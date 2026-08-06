#!/usr/bin/env python3
"""T-1758 -- trace, at source, the rule that sets the carried per-token
scale at the layer-25/layer-26 residual-stream site, and what it costs.

T-1751 (Claude/Brunel/t1751-e2-layer26-collapse-remeasure-2026-08-05.md)
found that the emitted carried (m, e) scale at row26 ("layer 25"), position
0 does not equal the independent reference-float peak at that (position,
row) divided by 127 -- the peak channel's int8 code is -127 (saturated),
and the carried step (38.6458) is smaller than peak/127 (51.9055). T-1751
explicitly did not attribute the divergence.

Reading the source (checked_chain_funnel.cpp's RequantChainChecked,
forward_sites.cpp's ResidualReconcileSite) resolves this:

  - The int8 CODES at this site are set by a genuine per-token max-abs rule:
    D' = MaxAbsReduceWide(wide, hidden_size) over the row actually being
    quantized -- `wide[i] = reconciled_branch[i] + stream_code[i]`
    (forward_sites.cpp:759, ResidualReconcileSite), an INTEGER accumulator
    row already carrying the whole prior chain's scale implicitly (it is
    NOT the row's raw float values). RequantTokenCodeWide then maps this
    row's own peak (D') to code +-127 by construction -- a channel at the
    row's own max magnitude saturates exactly, every time, by design.

  - The EMITTED carried (m, e) scale (`*out_scale`, what downstream sites
    and this ticket's dumps read as "the carried scale") is NOT D'/127
    alone. It is `running`, the LEFT-ASSOCIATED CombineCarriedScale fold of
    {stream_scale (the incoming carried scale from the PRIOR row),
    site_constant (an artifact-carried KVC1 constant for this site),
    d_prime_factor = CarriedScale{ns.dn, -ns.s} (D' itself, canonical)} --
    checked_chain_funnel.cpp:306-321. This is a composed product carried
    across the WHOLE chain, not a fresh per-token max-abs of this row alone.

  - T-1751's comparator ("peak/127") used the INDEPENDENT reference-float
    forward's peak value (t1740_pooled_float_dump.py -- a separate
    HuggingFace bf16 forward, never derived from the int8 engine's own
    pre-quantization row) as the numerator. That reference has accumulated
    25 layers' worth of divergence from the engine's OWN internal
    computation by this point -- a divergence with nothing to do with this
    site's scale rule. So "carried step != peak_float/127" was never
    guaranteed to hold even under an exactly-max-abs rule at this site; the
    two sides of that equality are different quantities by construction
    (one is the engine's own composed chain product, the other mixes an
    independent reference model's value with a naive single-row model).

This script does not re-derive that source reading (already done by
reading the .cpp/.h files directly, cited above and in the accompanying
build log). It exists to answer, BY EXECUTION rather than by reading alone,
the two things source-reading cannot settle on its own:

  1. Is saturation at this site confined to the row's own single true-max
     channel (the design's intended one-channel full-scale mapping), or
     does the clamp in RequantTokenCodeWide/RequantTokenCodeWide's formula
     engage on more than one channel (which would indicate real information
     loss from clipping, not just the designed peak-channel mapping)? This
     is checked directly off the real captured int8 codes -- count of
     |code| == 127 at position 0, row26 ("layer 25") and row27 ("layer 26"),
     at each of the three prompt lengths.

  2. How much (reference-float) energy is held by the channels that
     saturate, at layer 25 and layer 26, at position 0 -- computed the same
     way T-1751 computed the zeroed-channel energy share, so the two costs
     (zeroing vs. saturation) are reported on the same basis and are
     directly comparable.

Reuses T-1751's exact instrument shape (a new, standalone script reading
already-captured, already-self-checked dumps -- no new forward pass) and
T-1751's own row-index mapping (row 0 = embedding output, row k = output of
layer k-1; D-SLM444's "layer L" = row L+1).

Usage
-----
    python tools\\t1758_carried_scale_rule_probe.py \\
        --int8 out\\t1740\\long.int8.bin --float out\\t1740\\long.float.bin --label long
"""

from __future__ import annotations

import argparse
import struct
import sys

import numpy as np

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass


def load_int8(path):
    with open(path, "rb") as f:
        n, r, h, fp = struct.unpack("<QQQQ", f.read(32))
        scales = np.zeros((n, r, 2), dtype=np.int64)
        codes = np.zeros((n, r, h), dtype=np.int8)
        for p in range(n):
            for row in range(r):
                m, e = struct.unpack("<qq", f.read(16))
                scales[p, row, 0] = m
                scales[p, row, 1] = e
                codes[p, row, :] = np.frombuffer(f.read(h), dtype=np.int8)
    return n, r, h, fp, scales, codes


def load_float(path):
    with open(path, "rb") as f:
        n, r, h, fp = struct.unpack("<QQQQ", f.read(32))
        data = np.frombuffer(f.read(), dtype=np.float32, count=n * r * h)
    return n, r, h, fp, data.reshape(n, r, h).astype(np.float64)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--int8", required=True)
    ap.add_argument("--float", required=True)
    ap.add_argument("--label", required=True)
    args = ap.parse_args()

    n8, r8, h8, fp8, scales, codes = load_int8(args.int8)
    nf, rf, hf, fpf, fvals = load_float(args.float)

    if (n8, r8, h8) != (nf, rf, hf):
        print(f"[{args.label}] SHAPE MISMATCH int8={n8,r8,h8} float={nf,rf,hf}", file=sys.stderr)
        return 1
    if fp8 != fpf:
        print(f"[{args.label}] PROMPT FINGERPRINT MISMATCH int8={fp8} float={fpf}", file=sys.stderr)
        return 1

    n, r, h = n8, r8, h8
    print(f"[{args.label}] positions={n} rows={r} hidden={h} fingerprint={fp8} (matched)")

    def row_for_layer(layer_index_0based: int) -> int:
        return layer_index_0based + 1

    row25 = row_for_layer(25)  # D-SLM444 "layer 25"
    row26 = row_for_layer(26)  # D-SLM444 "layer 26"

    codes_row25 = codes[0, row25, :]
    sat_mask = np.abs(codes_row25) == 127
    n_sat = int(np.sum(sat_mask))
    sat_channels = np.nonzero(sat_mask)[0].tolist()
    zero_mask = codes_row25 == 0
    n_zeroed = int(np.sum(zero_mask))

    m, e = int(scales[0, row25, 0]), int(scales[0, row25, 1])
    step = m * (2.0 ** e)

    print(f"  -- saturation, position 0, row={row25} (layer 25) --")
    print(f"    saturated channels (|code|==127): {n_sat} of {h}  -- indices: {sat_channels}")
    print(f"    zeroed channels (code==0): {n_zeroed} of {h}")
    print(f"    carried step (m*2^e) = {step:.4f}")

    energy25 = fvals[0, row25, :] ** 2
    energy26 = fvals[0, row26, :] ** 2
    total25 = float(np.sum(energy25))
    total26 = float(np.sum(energy26))

    sat_energy25 = float(np.sum(energy25[sat_mask]))
    sat_energy26 = float(np.sum(energy26[sat_mask]))
    zero_energy25 = float(np.sum(energy25[zero_mask]))
    zero_energy26 = float(np.sum(energy26[zero_mask]))

    sat_share25 = 100.0 * sat_energy25 / total25 if total25 != 0 else float("nan")
    sat_share26 = 100.0 * sat_energy26 / total26 if total26 != 0 else float("nan")
    zero_share25 = 100.0 * zero_energy25 / total25 if total25 != 0 else float("nan")
    zero_share26 = 100.0 * zero_energy26 / total26 if total26 != 0 else float("nan")

    print(f"    saturated-channel (reference-float) energy share: layer25={sat_share25:.4f}% layer26={sat_share26:.4f}%")
    print(f"    zeroed-channel (reference-float) energy share:    layer25={zero_share25:.4f}% layer26={zero_share26:.4f}%")

    # Reconstruction gap at each saturated channel: code*step (engine's own
    # composed-chain reconstruction) vs. the independent reference-float value
    # at the SAME channel -- reported for completeness, on the same basis
    # T-1751 already reported it for the single peak channel.
    for c in sat_channels:
        code_val = int(codes_row25[c])
        recon = code_val * step
        true_val = float(fvals[0, row25, c])
        print(f"    channel {c}: code={code_val} recon(code*step)={recon:.4f} "
              f"reference_float={true_val:.4f} gap={true_val - recon:.4f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

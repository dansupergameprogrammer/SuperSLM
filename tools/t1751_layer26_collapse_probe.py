#!/usr/bin/env python3
"""T-1751 -- re-measure the E2/D-SLM444 layer-26 collapse mechanism against
the compiled engine's own captured forward.

D-SLM444 (Python spike, S3a parity tap harness, `s3a_parity/run_position0_followups.py`,
float-side unquantized forward) found, at layers 25/26 of the residual stream:

  (1) Float-side norm at position 0 vs. the sequence-median position: 220x at
      layer 1, 55x at layer 25, collapsing to 1.5x at layer 26 -- while the
      median position's own norm grows monotonically (17.6 -> 238) across the
      stack.
  (2) The per-token max-abs int8 quantization scale at layer 25 is set by
      position 0's own outlier peak (6615 -> scale 52.09) and zeroes 1278 of
      1536 channels at that position. Those zeroed channels hold 0.136% of
      layer 25's own (float) energy at position 0, but 25.85% of layer 26's.

This script re-derives both findings from the compiled C++ engine's own
already-captured, already-self-checked forward -- the raw per-position,
per-row dumps T-1740 produced and self-checked (bit-for-bit production
agreement, resumed-vs-one-shot equivalence, both passed on all three
prompts; see Claude/Brunel/t1740-engine-pooled-cosine-build-2026-08-05.md
Sec.3). No new forward pass is run. Per the standing rule against modifying
existing tools in place, this is a new script reading existing dumps, not an
edit to `tools/t1740_pooled_fidelity_report.py`.

Row convention (verified at source against the dumps' own header, 29 rows):
  row 0 = embedding output
  row k (1 <= k <= 28) = output of layer (k-1)

So D-SLM444's "layer 25" = row 26, "layer 26" = row 27 -- this mapping is
computed explicitly below, not assumed.

Dequantization: this codebase's own CarriedScale contract
(include/superslm/checked_chain_funnel.h: "value == m * 2^e"), one (m, e)
pair shared across all 1536 channels of a given (position, row) -- i.e. the
capture format IS the "per-token max-abs scale grid" D-SLM444 describes.
"Zeroed" channels are read directly off the real captured int8 codes
(code == 0), not re-derived by simulating quantization -- these are the
engine's own actual W8A8 residual codes.

Usage
-----
    python tools\\t1751_layer26_collapse_probe.py \\
        --int8 out\\t1740\\long.int8.bin --float out\\t1740\\long.float.bin --label long
"""

from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

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

    # Row convention: row 0 = embed, row k = layer(k-1). D-SLM444's "layer L" = row L+1.
    def row_for_layer(layer_index_0based: int) -> int:
        return layer_index_0based + 1

    row1 = row_for_layer(1)    # D-SLM444 "layer 1"
    row25 = row_for_layer(25)  # D-SLM444 "layer 25"
    row26 = row_for_layer(26)  # D-SLM444 "layer 26"

    # ---- (1) position-0-vs-median norm ratio, float side ----
    norms = np.linalg.norm(fvals, axis=2)  # [position, row]
    print(f"  -- finding 1: position-0 norm vs. median-position norm (float side) --")
    for label, row in (("layer1", row1), ("layer25", row25), ("layer26", row26)):
        pos0_norm = norms[0, row]
        other_norms = norms[1:, row]
        median_norm = float(np.median(other_norms))
        ratio = pos0_norm / median_norm if median_norm != 0 else float("inf")
        print(f"    row={row} ({label}): pos0_norm={pos0_norm:.4f} median_norm={median_norm:.4f} ratio={ratio:.4f}x")

    # Median-position norm growth across the full stack (rows 1..28, "layer0".."layer27")
    median_curve = [float(np.median(norms[1:, row])) for row in range(1, r)]
    print(f"    median-position norm, row1..row{r-1}: first={median_curve[0]:.4f} last={median_curve[-1]:.4f}"
          f" monotonic_nondecreasing={all(b >= a - 1e-9 for a, b in zip(median_curve, median_curve[1:]))}")

    # ---- (2) per-token max-abs scale grid + channel zeroing + energy share ----
    print(f"  -- finding 2: per-token int8 scale grid at layer25, position 0 --")
    m, e = int(scales[0, row25, 0]), int(scales[0, row25, 1])
    step = m * (2.0 ** e)
    pos0_peak_float = float(np.max(np.abs(fvals[0, row25, :])))
    codes_pos0_row25 = codes[0, row25, :]
    zero_mask = codes_pos0_row25 == 0
    n_zeroed = int(np.sum(zero_mask))
    print(f"    m={m} e={e} step={step:.4f} float_peak_at_pos0={pos0_peak_float:.4f}"
          f" step_from_peak/127={pos0_peak_float / 127.0:.4f}")
    print(f"    zeroed channels (code==0): {n_zeroed} of {h}")

    energy25 = fvals[0, row25, :] ** 2
    energy26 = fvals[0, row26, :] ** 2
    total25 = float(np.sum(energy25))
    total26 = float(np.sum(energy26))
    zeroed_energy25 = float(np.sum(energy25[zero_mask]))
    zeroed_energy26 = float(np.sum(energy26[zero_mask]))
    share25 = 100.0 * zeroed_energy25 / total25 if total25 != 0 else float("nan")
    share26 = 100.0 * zeroed_energy26 / total26 if total26 != 0 else float("nan")
    print(f"    zeroed-channel share of position-0 energy: layer25={share25:.4f}% layer26={share26:.4f}%")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

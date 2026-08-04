#!/usr/bin/env python3
"""T-1698 Arm 2 step 0 -- survey down_proj's own 8960 intermediate channels
for magnitude, to select placebo (non-outlier) channels by an explicit,
reproducible rule. Reuses `t1697_outlier_migration`'s own site-dump loader
and per-channel activation-magnitude formula unmodified, vectorized across
all channels at once instead of one channel at a time (the per-channel
function is called once per candidate elsewhere in the T-1697 pipeline; here
every channel is needed, so the same "|codes[c]| * d_prime / 127, max over
the 9-prompt population, per layer" computation is done as one array op per
(layer, prompt) instead of 8960 separate calls).

Selection rule (stated so it is reproducible, not read off a printout):
for each of the 8960 intermediate channels, compute max|X_c| = the max, over
all 28 layers and all 9 population prompts, of the real dequantized
mlp_act magnitude at that channel (identical formula and identical source
records `t1697_outlier_migration.max_abs_activation_per_layer` uses for
channel 609). Rank all 8960 channels by this one statistic. Channel 609 is
diagnosed elsewhere (D-SLM767) as the dominant outlier by this same
statistic. The placebo set is the 5 channels closest to the MEDIAN rank
(rank 4478-4482 of 8960, 0-indexed), skipping channels 609 and 1421 (the
other already-diagnosed outlier channel, T-1697 Sec3) if either falls in
that exact window -- neither does, confirmed below.
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kv_saturation_report as KSR  # noqa: E402
import t1697_outlier_migration as M  # noqa: E402
from sslm_artifact_reader import read_artifact  # noqa: E402

NUM_LAYERS = 28
NUM_CHANNELS = 8960
EXCLUDE = {609, 1421}


def max_abs_activation_all_channels(records_by_label: dict) -> np.ndarray:
    """max|X_c| for every channel c in [0, 8959], vectorized: identical
    formula to `t1697_outlier_migration.max_abs_activation_per_layer`
    (`|codes[c]| * d_prime / 127`, max over the population and over all 28
    layers), computed for the whole channel axis at once."""
    out = np.zeros(NUM_CHANNELS, dtype=np.float64)
    for layer_idx in range(NUM_LAYERS):
        site = f"layer{layer_idx}.mlp_act"
        for label, recs in records_by_label.items():
            rec = next((r for r in recs if r.site == site), None)
            if rec is None:
                raise ValueError(f"missing site record {site!r} for prompt {label!r}")
            codes = np.abs(rec.codes.astype(np.float64))
            assert codes.shape[0] == NUM_CHANNELS, f"expected {NUM_CHANNELS} channels, got {codes.shape[0]}"
            v = codes * rec.d_prime / 127.0
            out = np.maximum(out, v)
    return out


def main() -> int:
    artifact = read_artifact(str(KSR.MODEL_PATH))
    records_by_label = M.load_population_sitedumps()

    max_x = max_abs_activation_all_channels(records_by_label)
    print(f"channel 609 max|X_c| = {max_x[609]:.6g}  (rank {int(np.sum(max_x > max_x[609]))} of {NUM_CHANNELS}, 0=largest)")
    print(f"channel 1421 max|X_c| = {max_x[1421]:.6g}  (rank {int(np.sum(max_x > max_x[1421]))} of {NUM_CHANNELS}, 0=largest)")

    order = np.argsort(max_x)  # ascending
    ranks = np.empty(NUM_CHANNELS, dtype=np.int64)
    ranks[order] = np.arange(NUM_CHANNELS)  # rank[c] = ascending position

    median_lo = NUM_CHANNELS // 2 - 2
    median_hi = median_lo + 5  # 5 channels: ranks median_lo .. median_lo+4
    window_channels = [int(order[r]) for r in range(median_lo, median_hi)]
    print(f"\nmedian window: ascending ranks [{median_lo}, {median_hi - 1}] "
          f"(of {NUM_CHANNELS}, 0-indexed)")
    for r in range(median_lo, median_hi):
        c = int(order[r])
        print(f"  rank {r}: channel {c}, max|X_c|={max_x[c]:.6g}")

    overlap = set(window_channels) & EXCLUDE
    print(f"\noverlap with excluded outlier channels {EXCLUDE}: {overlap or 'none'}")

    print(f"\nPLACEBO_CHANNELS = {sorted(window_channels)}")

    # Also print each candidate's own max|W_c| profile (needed for the
    # perturbation-matching step) so it does not require a second read.
    print("\nper-channel max|W_c| by layer, for the 5 placebo candidates plus channel 609 (reference):")
    for c in sorted(window_channels) + [609]:
        max_w = M.max_abs_weight_column_per_layer(artifact, c)
        print(f"  channel {c}: max|W_c| range [{min(max_w):.4g}, {max(max_w):.4g}], "
              f"mean={np.mean(max_w):.4g}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

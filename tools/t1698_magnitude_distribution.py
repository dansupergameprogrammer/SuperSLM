#!/usr/bin/env python3
"""T-1698 -- full magnitude-distribution report at down_proj's input (all
8960 intermediate channels), plus per-layer/per-prompt top-channel identity
stability. Reuses the exact per-channel activation-magnitude formula
`t1697_outlier_migration.max_abs_activation_per_layer` already uses for one
channel at a time, vectorized here across the whole channel axis (same
approach as `t1698_channel_survey.py`, extended to report the full
distribution and per-(layer,prompt) argmax rather than only the median
window)."""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kv_saturation_report as KSR  # noqa: E402
import t1697_outlier_migration as M  # noqa: E402

NUM_LAYERS = 28
NUM_CHANNELS = 8960


def main() -> int:
    records_by_label = M.load_population_sitedumps()

    # Per (layer, channel) max over the 9-prompt population, AND
    # per (layer, prompt, channel) individually (for stability check).
    per_layer_per_prompt = {}  # (layer, label) -> ndarray[8960]
    for layer_idx in range(NUM_LAYERS):
        site = f"layer{layer_idx}.mlp_act"
        for label, recs in records_by_label.items():
            rec = next((r for r in recs if r.site == site), None)
            codes = np.abs(rec.codes.astype(np.float64))
            assert codes.shape[0] == NUM_CHANNELS
            per_layer_per_prompt[(layer_idx, label)] = codes * rec.d_prime / 127.0

    # Overall max|X_c| per channel across all layers and all prompts (the
    # SAME statistic t1698_channel_survey.py already reported for channels
    # 609/1421/the placebo window).
    overall_max = np.zeros(NUM_CHANNELS, dtype=np.float64)
    for arr in per_layer_per_prompt.values():
        overall_max = np.maximum(overall_max, arr)

    order_desc = np.argsort(-overall_max)  # descending: index 0 = largest
    n = NUM_CHANNELS

    print("=== Global max|X_c| distribution across all 8960 channels (max over 28 layers x 9 prompts) ===")
    print(f"{'rank':>6s} {'channel':>8s} {'max|X_c|':>14s}")
    for r in [0, 1, 2, 5, 10, 50, 100]:
        c = int(order_desc[r])
        print(f"{r:6d} {c:8d} {overall_max[c]:14.6g}")

    sorted_vals = overall_max[order_desc]  # descending
    median_val = float(np.median(overall_max))
    print(f"\n{'percentile':>12s} {'value':>14s}")
    for p in [99.9, 99, 95, 75, 50, 25, 1]:
        val = float(np.percentile(overall_max, p))
        print(f"{p:12.1f} {val:14.6g}")

    rank0 = float(sorted_vals[0])
    print(f"\nrank0 (largest) = {rank0:.6g}")
    print(f"median = {median_val:.6g}")
    print(f"rank0 / median = {rank0 / median_val:.4f}")
    print(f"channel 609 max|X_c| = {overall_max[609]:.6g}  "
          f"(rank {int(np.sum(overall_max > overall_max[609]))} of {n}, 0=largest, "
          f"{'ABOVE' if overall_max[609] > median_val else 'BELOW'} median, "
          f"ratio to median = {overall_max[609]/median_val:.4f})")
    print(f"channel 1421 max|X_c| = {overall_max[1421]:.6g}  "
          f"(rank {int(np.sum(overall_max > overall_max[1421]))} of {n}, 0=largest, "
          f"{'ABOVE' if overall_max[1421] > median_val else 'BELOW'} median, "
          f"ratio to median = {overall_max[1421]/median_val:.4f})")

    # --- Per-layer top-channel identity stability, across the 9-prompt population ---
    print("\n=== Per-layer top channel (max over the 9-prompt population at that layer) ===")
    per_layer_top = {}
    for layer_idx in range(NUM_LAYERS):
        layer_max = np.zeros(NUM_CHANNELS, dtype=np.float64)
        for label, _q, _r in KSR.POPULATION:
            layer_max = np.maximum(layer_max, per_layer_per_prompt[(layer_idx, label)])
        top_c = int(np.argmax(layer_max))
        per_layer_top[layer_idx] = top_c
        print(f"  layer{layer_idx:2d}: top channel = {top_c:5d}  value={layer_max[top_c]:.6g}")

    distinct_layer_tops = sorted(set(per_layer_top.values()))
    print(f"\ndistinct top-channel identities across 28 layers: {len(distinct_layer_tops)} "
          f"-> {distinct_layer_tops}")

    # --- Per-(layer,prompt) top-channel identity stability, across all 9 prompts individually ---
    print("\n=== Per-(layer,prompt) top channel -- does it wander within a layer, across prompts? ===")
    wander_count = 0
    total_cells = 0
    for layer_idx in range(NUM_LAYERS):
        tops_this_layer = set()
        for label, _q, _r in KSR.POPULATION:
            arr = per_layer_per_prompt[(layer_idx, label)]
            tops_this_layer.add(int(np.argmax(arr)))
        total_cells += 1
        if len(tops_this_layer) > 1:
            wander_count += 1
        print(f"  layer{layer_idx:2d}: distinct top channels across 9 prompts = {len(tops_this_layer)} "
              f"-> {sorted(tops_this_layer)}")
    print(f"\nlayers where the top channel WANDERS across prompts: {wander_count}/{total_cells}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

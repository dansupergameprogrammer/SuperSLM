#!/usr/bin/env python3
"""T-1700 -- full magnitude-distribution report at down_proj's OUTPUT (all
1536 hidden-size channels -- the residual-stream side), plus per-layer/
per-prompt top-channel identity stability. Extends `t1698_magnitude_
distribution.py` to the site the source (`src/forward/forward_sites.cpp`
line ~1486, `ProjectAndFunnel`'s `RequantChainChecked` call) actually names
`down_proj.requant` -- confirmed by execution here too: `layer0.down_proj.
requant` site records carry 1536-wide codes (hidden_size), not 8960
(intermediate_size). T-1698's own survey read `layer{L}.mlp_act` (the real
8960-wide INPUT to down_proj's matmul, quantized at a different site) and
never read `down_proj.requant` at all -- this script is the first survey of
the site that literal name actually belongs to.

Same statistic, same population, same driver as t1698_magnitude_
distribution.py: `max|X_c| = |codes[c]| * d_prime / 127`, max over 28 layers
x 9 prompts, from the SAME captured sitedumps under `out/t1697_population/`
(reused, not regenerated, per this task's scope)."""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kv_saturation_report as KSR  # noqa: E402
import t1697_outlier_migration as M  # noqa: E402

NUM_LAYERS = 28
NUM_CHANNELS = 1536
SITE_SUFFIX = "down_proj.requant"

# Damage locus named in T-1698/T-1699 (rows 21-26, onset sharp at 21-22, peak 25-26).
DAMAGE_LOCUS = set(range(21, 27))


def main() -> int:
    records_by_label = M.load_population_sitedumps()

    per_layer_per_prompt = {}  # (layer, label) -> ndarray[1536]
    for layer_idx in range(NUM_LAYERS):
        site = f"layer{layer_idx}.{SITE_SUFFIX}"
        for label, recs in records_by_label.items():
            rec = next((r for r in recs if r.site == site), None)
            if rec is None:
                raise ValueError(f"missing site record {site!r} for prompt {label!r}")
            codes = np.abs(rec.codes.astype(np.float64))
            assert codes.shape[0] == NUM_CHANNELS, (
                f"{site}: expected {NUM_CHANNELS} channels, got {codes.shape[0]}")
            per_layer_per_prompt[(layer_idx, label)] = codes * rec.d_prime / 127.0

    overall_max = np.zeros(NUM_CHANNELS, dtype=np.float64)
    for arr in per_layer_per_prompt.values():
        overall_max = np.maximum(overall_max, arr)

    order_desc = np.argsort(-overall_max)
    n = NUM_CHANNELS

    print(f"=== Global max|X_c| distribution across all {NUM_CHANNELS} channels "
          f"at {SITE_SUFFIX} (max over 28 layers x 9 prompts) ===")
    print(f"{'rank':>6s} {'channel':>8s} {'max|X_c|':>14s}")
    for r in [0, 1, 2, 5, 10, 50, 100]:
        c = int(order_desc[r])
        print(f"{r:6d} {c:8d} {overall_max[c]:14.6g}")

    sorted_vals = overall_max[order_desc]
    median_val = float(np.median(overall_max))
    print(f"\n{'percentile':>12s} {'value':>14s}")
    for p in [99.9, 99, 95, 75, 50, 25, 1]:
        val = float(np.percentile(overall_max, p))
        print(f"{p:12.1f} {val:14.6g}")

    rank0 = float(sorted_vals[0])
    print(f"\nrank0 (largest) = {rank0:.6g}")
    print(f"median = {median_val:.6g}")
    print(f"rank0 / median = {rank0 / median_val:.4f}")

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
    locus_wander_count = 0
    locus_total = 0
    for layer_idx in range(NUM_LAYERS):
        tops_this_layer = set()
        for label, _q, _r in KSR.POPULATION:
            arr = per_layer_per_prompt[(layer_idx, label)]
            tops_this_layer.add(int(np.argmax(arr)))
        total_cells += 1
        wanders = len(tops_this_layer) > 1
        if wanders:
            wander_count += 1
        if layer_idx in DAMAGE_LOCUS:
            locus_total += 1
            if wanders:
                locus_wander_count += 1
        marker = "  <-- damage locus" if layer_idx in DAMAGE_LOCUS else ""
        print(f"  layer{layer_idx:2d}: distinct top channels across 9 prompts = {len(tops_this_layer)} "
              f"-> {sorted(tops_this_layer)}{marker}")
    print(f"\nlayers where the top channel WANDERS across prompts: {wander_count}/{total_cells}")
    print(f"of which, within the damage locus (layers 21-26): "
          f"{locus_wander_count}/{locus_total}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""T-1698 Arm 2 step 1 -- for each placebo channel, search alpha to find the
value whose migration footprint best matches channel 609's own alpha=0.18
footprint (T-1697 Sec5.3: 20/28 layers migrated, 0.458% saturation,
max weight-quant error 14.06, mean RMSE 0.226 over migrated layers).

Computes the SAME per-layer statistics `t1697_outlier_migration.
build_migrated_artifact` computes (n_saturated, max_abs_weight_quant_error,
rmse_weight_quant_error, n_migrated_layers) directly from the already-parsed
artifact arrays (`layer.down_weight`, `layer.down_fold`) instead of
byte-patching a copy of the 1.5GB file to disk -- this is a search over many
candidate alphas per channel and a full disk write per candidate would be
wasteful; the arithmetic is identical (int8 column * s_c, round-half-away-
from-zero, clip to [-127,127], compared against real_scale-dequantized
target), confirmed to match `build_migrated_artifact`'s own byte-level
result for the finally-chosen alpha in `t1698_build_placebo_artifacts.py`
(same bytes_changed/n_saturated reported both ways for the search winner).

Matching rule (stated so it is reproducible): for each candidate alpha in a
coarse-to-fine grid, compute n_migrated_layers and total saturation %; the
chosen alpha is the one minimizing |n_migrated_layers - 20| first (the
primary match target, since it is what T-1697 Sec5.3.1 shows governs which
part of the network is touched), with saturation % as the tiebreaker
(closest to 0.458%).
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kv_saturation_report as KSR  # noqa: E402
import shadow_layer_recompute as S  # noqa: E402
import t1697_outlier_migration as M  # noqa: E402
from sslm_artifact_reader import read_artifact  # noqa: E402

REFERENCE = dict(n_migrated=20, sat_pct=0.458, max_err=14.06, mean_rmse=0.226)

PLACEBO_CHANNELS = [269, 870, 5967, 6015, 7693]

ALPHA_GRID = [round(0.14 + 0.005 * i, 4) for i in range(13)]  # 0.140 .. 0.200, step 0.005


def layer_stats_in_memory(artifact, channel: int, layer_scales: list[float]) -> dict:
    n_migrated = 0
    total_sat = 0
    total_elem = 0
    max_err = 0.0
    rmse_list = []
    for layer_idx, s_c in enumerate(layer_scales):
        layer = artifact.layers[layer_idx]
        out_channels_down = layer.down_weight.shape[0]
        total_elem += out_channels_down
        if s_c == 1.0:
            continue
        n_migrated += 1
        old_codes = layer.down_weight[:, channel].astype(np.float64)
        raw = old_codes * s_c
        new_codes = np.trunc(raw + np.where(raw >= 0, 0.5, -0.5))
        n_saturated = int(np.sum(np.abs(new_codes) > 127))
        new_codes = np.clip(new_codes, -127, 127)
        total_sat += n_saturated

        real_scale_down = S.real_scale(layer.down_fold[:, 1], layer.down_fold[:, 2], layer.down_fold[:, 0])
        w_old = old_codes * real_scale_down
        w_target = w_old * s_c
        w_new = new_codes * real_scale_down
        err = np.abs(w_new - w_target)
        max_err = max(max_err, float(np.max(err)))
        rmse_list.append(float(np.sqrt(np.mean(err ** 2))))
    return dict(
        n_migrated=n_migrated, sat_pct=100.0 * total_sat / total_elem if total_elem else 0.0,
        max_err=max_err, mean_rmse=float(np.mean(rmse_list)) if rmse_list else 0.0)


def main() -> int:
    artifact = read_artifact(str(KSR.MODEL_PATH))
    records_by_label = M.load_population_sitedumps()

    chosen = {}
    for channel in PLACEBO_CHANNELS:
        max_x = M.max_abs_activation_per_layer(channel, records_by_label)
        max_w = M.max_abs_weight_column_per_layer(artifact, channel)
        print(f"\n=== channel {channel} ===")
        best = None
        for alpha in ALPHA_GRID:
            layer_scales = M.derive_layer_scales(max_x, max_w, alpha)
            stats = layer_stats_in_memory(artifact, channel, layer_scales)
            score = (abs(stats["n_migrated"] - REFERENCE["n_migrated"]),
                     abs(stats["sat_pct"] - REFERENCE["sat_pct"]))
            print(f"  alpha={alpha:.2f}: n_migrated={stats['n_migrated']}/28  "
                  f"sat%={stats['sat_pct']:.3f}  max_err={stats['max_err']:.4g}  "
                  f"mean_rmse={stats['mean_rmse']:.4g}  score={score}")
            if best is None or score < best[0]:
                best = (score, alpha, stats)
        _, alpha, stats = best
        chosen[channel] = alpha
        print(f"  -> CHOSEN alpha={alpha:.2f} for channel {channel}: {stats}")

    print(f"\nCHOSEN_ALPHAS = {chosen}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

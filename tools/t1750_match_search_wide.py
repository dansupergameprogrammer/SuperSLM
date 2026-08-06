#!/usr/bin/env python3
"""T-1750: run T-1698's own perturbation-matching search
(`t1698_match_search.layer_stats_in_memory`, unmodified) over the 24 NEW
placebo channels found by `t1750_channel_survey_wide.py`, using the identical
matching rule against the identical reference (channel 609's own alpha=0.18
footprint). Reuses every primitive; only the channel list is new.

Usage: python tools\\t1750_match_search_wide.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kv_saturation_report as KSR  # noqa: E402
import t1697_outlier_migration as M  # noqa: E402
import t1698_match_search as MS  # noqa: E402
from sslm_artifact_reader import read_artifact  # noqa: E402

NEW_PLACEBO_CHANNELS = [5, 575, 581, 856, 2430, 2674, 3193, 3885, 4122, 4460, 4821, 5030,
                         5108, 5198, 6111, 6491, 7658, 7796, 7918, 8159, 8224, 8234, 8715, 8804]


def main() -> int:
    artifact = read_artifact(str(KSR.MODEL_PATH))
    records_by_label = M.load_population_sitedumps()

    chosen = {}
    for channel in NEW_PLACEBO_CHANNELS:
        max_x = M.max_abs_activation_per_layer(channel, records_by_label)
        max_w = M.max_abs_weight_column_per_layer(artifact, channel)
        print(f"\n=== channel {channel} ===")
        best = None
        for alpha in MS.ALPHA_GRID:
            layer_scales = M.derive_layer_scales(max_x, max_w, alpha)
            stats = MS.layer_stats_in_memory(artifact, channel, layer_scales)
            score = (abs(stats["n_migrated"] - MS.REFERENCE["n_migrated"]),
                     abs(stats["sat_pct"] - MS.REFERENCE["sat_pct"]))
            if best is None or score < best[0]:
                best = (score, alpha, stats)
        _, alpha, stats = best
        chosen[channel] = alpha
        print(f"  -> CHOSEN alpha={alpha:.3f} for channel {channel}: {stats}")

    print(f"\nCHOSEN_ALPHAS = {chosen}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""T-1750: widen T-1698's own placebo-channel selection window from 5 to 25
channels, using the SAME rule and the SAME statistic
(`t1698_channel_survey.max_abs_activation_all_channels`, unmodified) --
this measures the noise floor's own dispersion on a population large enough
to state a resolving power, per T-1698's own §5.4 observation that a 5-sample
null cannot resolve a small effect.

Selection rule (identical to T-1698, window widened): rank all 8,960
intermediate channels by max|X_c| (identical formula/source records).
Take the 25 channels closest to the median rank (ranks 4468-4492 of 8960,
0-indexed ascending), excluding channels 609 and 1421 (the diagnosed
outliers) if either falls in the window. T-1698's own original 5
(269, 870, 5967, 6015, 7693, ranks 4478-4482) are the innermost 5 of this
window and are re-used, not re-measured, since T-1750 already reproduced
them exactly (see build log §3).

Usage: python tools\\t1750_channel_survey_wide.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import t1697_outlier_migration as M  # noqa: E402
import t1698_channel_survey as CS  # noqa: E402

WINDOW_HALF = 12  # 25 channels total: median_lo-12 .. median_lo+12 (median_lo=4478 same as T-1698)
EXCLUDE = {609, 1421}
ORIGINAL_5 = {269, 870, 5967, 6015, 7693}


def main() -> int:
    records_by_label = M.load_population_sitedumps()
    max_x = CS.max_abs_activation_all_channels(records_by_label)

    order = np.argsort(max_x)  # ascending
    median_lo_orig = CS.NUM_CHANNELS // 2 - 2  # 4478, T-1698's own anchor
    lo = median_lo_orig - WINDOW_HALF
    hi = median_lo_orig + 5 + WINDOW_HALF  # covers original [4478,4482] plus WINDOW_HALF each side

    window_channels = []
    for r in range(lo, hi):
        c = int(order[r])
        window_channels.append((r, c, float(max_x[c])))

    print(f"wide window: ascending ranks [{lo}, {hi - 1}] of {CS.NUM_CHANNELS} (0-indexed)")
    print(f"n candidates in window: {len(window_channels)}")
    for r, c, v in window_channels:
        tag = "EXCLUDED(outlier)" if c in EXCLUDE else ("ORIGINAL-5" if c in ORIGINAL_5 else "NEW")
        print(f"  rank {r}: channel {c}, max|X_c|={v:.6g}  [{tag}]")

    new_channels = [c for _, c, _ in window_channels if c not in EXCLUDE and c not in ORIGINAL_5]
    print(f"\nNEW_PLACEBO_CHANNELS ({len(new_channels)}) = {sorted(new_channels)}")
    dropped = [c for _, c, _ in window_channels if c in EXCLUDE]
    if dropped:
        print(f"dropped as diagnosed outliers: {dropped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

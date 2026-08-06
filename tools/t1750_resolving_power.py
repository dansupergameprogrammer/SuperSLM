#!/usr/bin/env python3
"""T-1750: compute the achieved resolving power of T-1698's noise-floor
calibration, extended from N=5 to N=29 placebo perturbations (T-1698's own
original 5, reproduced exactly by this task, plus 24 new channels selected
and matched by the identical rule -- `t1750_channel_survey_wide.py`,
`t1750_match_search_wide.py`, `t1750_run_wide_placebo.py`).

Reports, per StandardsDocument §5.4 ("compute the achieved resolving power
and state the result against it"):
  - the pooled null-distribution statistics (mean, std, SE, min/max, IQR)
    for mech2 agreement % and mech2 mean Spearman, over the full N=29 sample
  - a bootstrap 95% CI on the null max (the quantity T-1711 §8.1's margin
    rule divides by) and on the null spread (null_max - null_min)
  - T-1698's own 5-sample statistic recomputed for comparison
  - the total token-comparison count the pooled statistic rests on (vs.
    T-1698's ~12 per single run)

Usage: python tools\\t1750_resolving_power.py
"""
from __future__ import annotations

import csv
import os
import sys

import numpy as np

REPO_ROOT = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
CSV_PATH = os.path.join(REPO_ROOT, "out", "t1750_wide_placebo_summary.csv")

# T-1698's own original 5, from its published build log (§5.3) -- reused
# verbatim rather than re-run, since this task reproduced them exactly.
ORIGINAL_5 = [
    dict(channel=269, alpha=0.175, mech2_agree=2, mech2_pos=10, mech2_pct=20.0, mech2_spearman=0.5844),
    dict(channel=870, alpha=0.165, mech2_agree=6, mech2_pos=12, mech2_pct=50.0, mech2_spearman=0.6773),
    dict(channel=5967, alpha=0.17, mech2_agree=4, mech2_pos=11, mech2_pct=36.4, mech2_spearman=0.6188),
    dict(channel=6015, alpha=0.17, mech2_agree=5, mech2_pos=12, mech2_pct=41.7, mech2_spearman=0.6717),
    dict(channel=7693, alpha=0.175, mech2_agree=7, mech2_pos=13, mech2_pct=53.8, mech2_spearman=0.7067),
]

REFERENCE_CH609 = dict(mech2_pct=61.5, mech2_spearman=0.7141)


def bootstrap_ci(values: np.ndarray, stat_fn, n_boot: int = 20000, alpha: float = 0.05, seed: int = 1750):
    rng = np.random.default_rng(seed)
    n = len(values)
    boots = np.empty(n_boot)
    for i in range(n_boot):
        sample = values[rng.integers(0, n, size=n)]
        boots[i] = stat_fn(sample)
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def report(label: str, pct: np.ndarray, sp: np.ndarray, total_pos: int) -> None:
    print(f"\n=== {label}: N={len(pct)} placebo perturbations, {total_pos} pooled mech2 token comparisons ===")
    print(f"mech2 agreement %%: mean={np.mean(pct):.2f}  std={np.std(pct, ddof=1):.2f}  "
          f"SE={np.std(pct, ddof=1)/np.sqrt(len(pct)):.2f}  min={np.min(pct):.1f}  max={np.max(pct):.1f}  "
          f"spread={np.max(pct)-np.min(pct):.1f}  median={np.median(pct):.1f}")
    print(f"mech2 spearman: mean={np.mean(sp):.4f}  std={np.std(sp, ddof=1):.4f}  "
          f"SE={np.std(sp, ddof=1)/np.sqrt(len(sp)):.4f}  min={np.min(sp):.4f}  max={np.max(sp):.4f}  "
          f"spread={np.max(sp)-np.min(sp):.4f}")

    if len(pct) >= 8:
        lo_max, hi_max = bootstrap_ci(pct, np.max)
        lo_spread, hi_spread = bootstrap_ci(pct, lambda x: np.max(x) - np.min(x))
        print(f"bootstrap 95% CI on null MAX (mech2 agreement %%): [{lo_max:.1f}, {hi_max:.1f}]")
        print(f"bootstrap 95% CI on null SPREAD (mech2 agreement %%): [{lo_spread:.1f}, {hi_spread:.1f}]")
        lo_max_sp, hi_max_sp = bootstrap_ci(sp, np.max)
        print(f"bootstrap 95% CI on null MAX (mech2 spearman): [{lo_max_sp:.4f}, {hi_max_sp:.4f}]")


def main() -> int:
    rows = list(ORIGINAL_5)
    if os.path.exists(CSV_PATH):
        with open(CSV_PATH, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                rows.append(dict(channel=int(r["channel"]), alpha=float(r["alpha"]),
                                  mech2_agree=int(r["mech2_agree"]), mech2_pos=int(r["mech2_pos"]),
                                  mech2_pct=float(r["mech2_pct"]), mech2_spearman=float(r["mech2_spearman"])))
    else:
        print(f"WARNING: {CSV_PATH} not found -- reporting on the original 5 only", file=sys.stderr)

    pct = np.array([r["mech2_pct"] for r in rows])
    sp = np.array([r["mech2_spearman"] for r in rows])
    total_pos = sum(r["mech2_pos"] for r in rows)

    report("T-1698 original (N=5, reproduced)", pct[:5], sp[:5], sum(r["mech2_pos"] for r in rows[:5]))
    if len(rows) > 5:
        report("T-1750 pooled (N=%d)" % len(rows), pct, sp, total_pos)

        margin_pct = (REFERENCE_CH609["mech2_pct"] - np.max(pct)) / (np.max(pct) - np.min(pct)) * 100.0
        margin_sp = (REFERENCE_CH609["mech2_spearman"] - np.max(sp)) / (np.max(sp) - np.min(sp)) * 100.0
        print(f"\nT-1697 ch609/alpha=0.18 reference (61.5%%, spearman 0.7141) against the N={len(rows)} "
              f"pooled null: margin = {margin_pct:.1f}%% of pooled spread (mech2 agreement), "
              f"{margin_sp:.1f}%% of pooled spread (spearman) -- per §8.1's own rule, >100%% is required "
              f"to be 'measurably outside' the floor.")

        print("\nfull per-channel table:")
        for r in rows:
            print(f"  channel {r['channel']:>5}  alpha={r['alpha']:.3f}  "
                  f"mech2={r['mech2_agree']}/{r['mech2_pos']} ({r['mech2_pct']:.1f}%)  "
                  f"spearman={r['mech2_spearman']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

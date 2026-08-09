#!/usr/bin/env python3
"""T-1869 attack D2 -- calibrate BOTH decision rules on the same delta shape.

D3 showed the sign-flip permutation test is anti-conservative on the primary
cell's left-skewed rank_top1 deltas. That is only half an attack: if the
campaign's own `|mean| / (1.96*SE) > 1.550038` rule is equally miscalibrated
on the same shape, the disagreement T-1868 reports is not evidence about
either rule. This measures the realised type-I rate of both, on populations
built from the observed deltas re-centred to a true mean of zero.
"""
from __future__ import annotations

import numpy as np
from scipy import stats

from t1869_probe_common import load_cell, metrics_for, ref_top1_index

ALPHA_CELL = 0.05 / 21
BONF_RATIO = 1.550038
TRIALS = 20_000


def main():
    rng = np.random.default_rng(18690)
    labels, mats, ref_mat, dom = load_cell("stageC")
    jstar = ref_top1_index(ref_mat)
    b = metrics_for(mats["base"], jstar)

    for arm, metric_idx, name in (("m1_g128", 1, "rank_top1"),
                                  ("m1_g128", 2, "margin_top1")):
        c = metrics_for(mats[arm], jstar)
        d = c[metric_idx] - b[metric_idx]
        centred = d - d.mean()
        n = len(d)
        print(f"\n=== {arm} / {name}: skew={stats.skew(d):+.3f}, "
              f"population re-centred to true mean 0, {TRIALS} resamples of n={n} ===")
        samp = rng.choice(centred, size=(TRIALS, n), replace=True)
        means = samp.mean(axis=1)
        ses = samp.std(axis=1, ddof=1) / np.sqrt(n)
        ratio = np.abs(means) / (1.96 * ses)
        fires_rp = float((ratio > BONF_RATIO).mean())
        tp = stats.ttest_1samp(samp, 0.0, axis=1).pvalue
        fires_t = float((tp < ALPHA_CELL).mean())
        print(f"  campaign rule |mean|/(1.96*SE) > {BONF_RATIO}: fires {fires_rp:.5f} "
              f"(nominal {ALPHA_CELL:.5f}, {fires_rp / ALPHA_CELL:.1f}x)")
        print(f"  paired t at alpha={ALPHA_CELL:.5f}:            fires {fires_t:.5f} "
              f"({fires_t / ALPHA_CELL:.1f}x)")


if __name__ == "__main__":
    main()

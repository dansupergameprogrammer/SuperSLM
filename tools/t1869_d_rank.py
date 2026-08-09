#!/usr/bin/env python3
"""T-1869 attack D -- the primary cell (`m1_g128`, `rank_top1`, pooled) and the
post-hoc permutation check T-1868 reports as disagreeing with it.

Four checks:
  D1  skewness of the 239 per-document rank_top1 deltas, and the information
      counts (documents with a nonzero delta per metric).
  D2  the sign-flip permutation test, re-run independently (200,000 draws).
  D3  CALIBRATION of that test on this exact delta shape. The sign-flip
      permutation null is "the paired differences are symmetric about zero" --
      not "no distributional assumption". This resamples from the observed
      deltas re-centred to mean zero (same skewed shape, true mean zero) and
      measures how often the test fires below the family-wise per-cell level.
  D4  fragility of the rank_top1 mean: how much of it rests on a few documents.
"""
from __future__ import annotations

import numpy as np
from scipy import stats

from t1869_probe_common import load_cell, metrics_for, paired, ref_top1_index

ALPHA_CELL = 0.05 / 21


def signflip_p(d: np.ndarray, draws: int, rng) -> float:
    obs = abs(d.mean())
    n = len(d)
    signs = rng.integers(0, 2, size=(draws, n)) * 2 - 1
    means = np.abs((signs * d).mean(axis=1))
    return float((means >= obs - 1e-15).sum() + 1) / (draws + 1)


def main():
    rng = np.random.default_rng(1869)
    labels, mats, ref_mat, dom = load_cell("stageC")
    jstar = ref_top1_index(ref_mat)
    b = metrics_for(mats["base"], jstar)
    c = metrics_for(mats["m1_g128"], jstar)
    d_rank = c[1] - b[1]
    d_rec = c[0] - b[0]
    d_mar = c[2] - b[2]

    print("--- D1: shape and information counts (m1_g128 vs base, stage C) ---")
    print(f"nonzero per-document delta: recall1={int((d_rec != 0).sum())}  "
          f"rank_top1={int((d_rank != 0).sum())}  margin_top1={int((d_mar != 0).sum())} of {len(d_rank)}")
    print(f"rank_top1 delta: mean={d_rank.mean():+.6f} sd={d_rank.std(ddof=1):.4f} "
          f"skew={stats.skew(d_rank):+.3f} min={d_rank.min():.0f} max={d_rank.max():.0f}")
    print(f"margin_top1 delta skew={stats.skew(d_mar):+.3f}")

    print("\n--- D2: sign-flip permutation, re-run independently ---")
    p = signflip_p(d_rank, 200_000, rng)
    print(f"two-sided sign-flip p = {p:.6f}   (family-wise per-cell alpha = {ALPHA_CELL:.6f})")
    print(f"paired t p = {stats.ttest_1samp(d_rank, 0.0).pvalue:.6f}")
    print(f"wilcoxon p = {stats.wilcoxon(d_rank[d_rank != 0]).pvalue:.6e}")

    print("\n--- D3: calibration of the sign-flip test on this delta shape ---")
    centred = d_rank - d_rank.mean()          # same skewed shape, true mean zero
    n = len(d_rank)
    fires = 0
    trials = 2000
    for _ in range(trials):
        samp = rng.choice(centred, size=n, replace=True)
        samp = samp - 0.0                      # population mean is zero by construction
        if signflip_p(samp, 4000, rng) < ALPHA_CELL:
            fires += 1
    print(f"nominal type-I rate at alpha={ALPHA_CELL:.6f}; observed {fires}/{trials} "
          f"= {fires / trials:.5f}  (ratio {fires / trials / ALPHA_CELL:.1f}x nominal)")

    print("\n--- D4: fragility of the rank_top1 mean ---")
    order = np.argsort(d_rank)                 # most negative (largest recovery) first
    for k in (0, 1, 2, 3, 5, 10):
        keep = np.ones(n, bool)
        keep[order[:k]] = False
        pk = paired(c[1][keep], b[1][keep])
        print(f"drop {k:>2d} largest-recovery docs (n={pk['n']}): delta={pk['delta']:+.5f} "
              f"ratio={pk['abs_ratio']:.3f}")
    print(f"the {int((d_rank < -5).sum())} documents with delta < -5 carry "
          f"{d_rank[d_rank < -5].sum() / d_rank.sum():.1%} of the total rank_top1 delta")


if __name__ == "__main__":
    main()

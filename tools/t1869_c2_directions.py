#!/usr/bin/env python3
"""T-1869 attack C2 -- is C's result special to the corpus-mean direction?

Repeats attack C at a fixed a=1e-4 (the largest swept value at which zero
documents change rank and zero change recall@1) for the corpus mean and for
eight random fixed directions, each scaled to the corpus mean's norm.
"""
from __future__ import annotations

import numpy as np

from t1869_probe_common import load_cell, metrics_for, paired, ref_top1_index

BONF = 1.550038
A = 1e-4


def main():
    labels, mats, ref_mat, dom = load_cell("stageC")
    jstar = ref_top1_index(ref_mat)
    base = mats["base"]
    b_recall, b_rank, b_margin = metrics_for(base, jstar)
    mu = base.mean(axis=0)
    scale = np.linalg.norm(mu)
    rng = np.random.default_rng(1869)
    dirs = [("corpus_mean", mu)]
    for k in range(8):
        d = rng.standard_normal(base.shape[1])
        dirs.append((f"random_{k}", d / np.linalg.norm(d) * scale))
    print(f"{'direction':>14} {'rank moved':>11} {'recall moved':>13} {'d_margin':>13} "
          f"{'ratio':>8} {'family-wise':>12}")
    for name, d in dirs:
        cand = base + A * d
        c_recall, c_rank, c_margin = metrics_for(cand, jstar)
        p = paired(c_margin, b_margin)
        print(f"{name:>14} {int((c_rank != b_rank).sum()):>11d} "
              f"{int((c_recall != b_recall).sum()):>13d} {p['delta']:>+13.3e} "
              f"{p['abs_ratio']:>8.3f} {str(p['abs_ratio'] > BONF):>12}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""T-1869 attack C -- a synthetic arm with EXACTLY zero retrieval effect.

Takes stage C's own `base` pooled vectors and adds a fixed multiple of the
corpus mean vector: v_i -> v_i + a * mean(v). Cosine similarities move; the
per-document ranking may or may not. The sweep reports, for each a, whether
recall@1 and rank_top1 are bit-identical to `base` on all 239 documents, and
what `margin_top1` reads.

An `a` at which recall@1 delta and rank_top1 delta are exactly zero on every
document is an arm with no retrieval effect of any kind that `recall@1` or
`rank_top1` can express. What `margin_top1` says about that arm is the test.
"""
from __future__ import annotations

import numpy as np

from t1869_probe_common import load_cell, metrics_for, paired, ref_top1_index

BONF = 1.550038


def main():
    labels, mats, ref_mat, dom = load_cell("stageC")
    jstar = ref_top1_index(ref_mat)
    base = mats["base"]
    b_recall, b_rank, b_margin = metrics_for(base, jstar)
    mu = base.mean(axis=0)

    print(f"{'a':>10} {'rank moved':>11} {'recall moved':>13} "
          f"{'d_margin':>13} {'ratio':>8} {'verdict':>22}")
    for a in (1e-1, 3e-2, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5):
        cand = base + a * mu
        c_recall, c_rank, c_margin = metrics_for(cand, jstar)
        nr = int((c_rank != b_rank).sum())
        nrec = int((c_recall != b_recall).sum())
        p = paired(c_margin, b_margin)
        verdict = ("RESOLVED family-wise" if p["abs_ratio"] > BONF
                   else ("resolved uncorrected" if p["abs_ratio"] > 1.0 else "not resolved"))
        print(f"{a:>10.0e} {nr:>11d} {nrec:>13d} {p['delta']:>+13.3e} "
              f"{p['abs_ratio']:>8.3f} {verdict:>22}")


if __name__ == "__main__":
    main()

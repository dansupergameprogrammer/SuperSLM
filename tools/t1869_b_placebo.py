#!/usr/bin/env python3
"""T-1869 attack B -- is `margin_top1`'s resolving power carried by retrieval
recovery, or by a global shift in each arm's own similarity geometry?

Three placebo statistics, each with the SAME functional form as `margin_top1`
but carrying no information about whether the correct neighbour was retrieved:

  placebo_rand   -- the identical gap statistic computed for a FIXED randomly
                    chosen document r[i] (r[i] != i, r[i] != jstar[i]) instead
                    of the reference's true top-1 neighbour. Same formula, same
                    exclusions, zero retrieval content.
  placebo_spread -- mean off-diagonal cosine similarity for query i. A pure
                    global-geometry statistic; no ranking, no correctness.
  placebo_own    -- the candidate arm's own top-1-to-runner-up gap, whichever
                    document that is. Measures decisiveness, not correctness.

If a placebo resolves at the same thresholds `margin_top1` does, the
resolution is a property of the arm's similarity geometry rather than of the
retrieval effect the packet attributes it to.

Also splits Delta margin_top1 by whether the document's rank of the correct
neighbour moved at all.
"""
from __future__ import annotations

import json

import numpy as np

from t1869_probe_common import load_cell, metrics_for, paired, ref_top1_index, rr

TEST_ARMS = {"stageB": ["ceiling", "middle"],
             "stageC": ["m1_g32", "m1_g128", "m2_residuals", "m2_site16", "m1m2_g128"]}


def placebos(mat, jstar, rand_j):
    sims = rr.cosine_sim_matrix(mat)
    n = sims.shape[0]
    p_rand = np.empty(n)
    p_spread = np.empty(n)
    p_own = np.empty(n)
    for i in range(n):
        row = sims[i].copy()
        row[i] = -np.inf
        finite = np.delete(sims[i], i)
        p_spread[i] = float(finite.mean())
        j = rand_j[i]
        v = row[j]
        row[j] = -np.inf
        p_rand[i] = float(v - row.max())
        row[j] = v
        srt = np.sort(row)[::-1]
        p_own[i] = float(srt[0] - srt[1])
    return p_rand, p_spread, p_own


def main():
    rng = np.random.default_rng(1869)
    out = {}
    for stage in ("stageB", "stageC"):
        labels, mats, ref_mat, dom = load_cell(stage)
        jstar = ref_top1_index(ref_mat)
        n = len(labels)
        rand_j = np.empty(n, dtype=np.int64)
        for i in range(n):
            while True:
                c = int(rng.integers(0, n))
                if c != i and c != jstar[i]:
                    rand_j[i] = c
                    break
        per = {name: metrics_for(m, jstar) for name, m in mats.items()}
        plc = {name: placebos(m, jstar, rand_j) for name, m in mats.items()}
        b, pb = per["base"], plc["base"]
        stage_out = {}
        for arm in TEST_ARMS[stage]:
            c, pc = per[arm], plc[arm]
            e = {
                "margin_top1": paired(c[2], b[2]),
                "placebo_rand": paired(pc[0], pb[0]),
                "placebo_spread": paired(pc[1], pb[1]),
                "placebo_own": paired(pc[2], pb[2]),
            }
            moved = c[1] != b[1]
            e["n_rank_moved"] = int(moved.sum())
            e["margin_on_rank_UNMOVED"] = paired(c[2][~moved], b[2][~moved])
            e["margin_on_rank_MOVED"] = paired(c[2][moved], b[2][moved]) if moved.sum() > 1 else None
            dmargin = c[2] - b[2]
            e["share_of_total_margin_delta_from_UNMOVED"] = float(
                dmargin[~moved].sum() / dmargin.sum())
            e["frac_docs_positive_dmargin"] = float((dmargin > 0).mean())
            stage_out[arm] = e
        out[stage] = stage_out
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()

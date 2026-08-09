#!/usr/bin/env python3
"""T-1869 attack E -- three remaining claims.

E1  the 54 "known-zero-effect" control arms: are they bit-identical to `base`
    at the VECTOR level? If so the control population has zero dispersion in
    every direction and can only detect a nondeterministic instrument, not a
    confounded metric.
E2  the per-domain (id/ood) claim: does the correctness-blind placebo resolve
    both subpopulations too, at the same threshold?
E3  a gap-scale-corrected margin. `margin_shortfall` = margin_top1 -
    (the arm's own top-1-to-runner-up gap). It is 0 exactly when the correct
    neighbour IS the arm's top-1, and negative by how far short it fell
    otherwise -- the same information margin_top1 carries, with the arm's own
    global gap scale divided out. What resolves after the confound is removed?
"""
from __future__ import annotations

import numpy as np

from t1869_probe_common import load_cell, metrics_for, paired, ref_top1_index, rr

BONF = 1.550038
ARMS = {"stageB": ["ceiling", "middle"],
        "stageC": ["m1_g32", "m1_g128", "m2_residuals", "m2_site16", "m1m2_g128"]}


def gap_stats(mat, jstar):
    sims = rr.cosine_sim_matrix(mat)
    n = sims.shape[0]
    own = np.empty(n)
    marg = np.empty(n)
    for i in range(n):
        row = sims[i].copy()
        row[i] = -np.inf
        srt = np.sort(row)[::-1]
        own[i] = srt[0] - srt[1]
        t = row[jstar[i]]
        row[jstar[i]] = -np.inf
        marg[i] = t - row.max()
    return own, marg


def verdict(r):
    return "FAMILY-WISE" if r > BONF else ("uncorrected" if r > 1.0 else "not resolved")


def main():
    print("=== E1: are the discarded-control arms bit-identical to base at the vector level? ===")
    for stage in ("stageB", "stageC"):
        labels, mats, ref_mat, dom = load_cell(stage)
        dups = [a for a in sorted(mats) if a.startswith("basedup")]
        worst = max(float(np.abs(mats[a] - mats["base"]).max()) for a in dups)
        print(f"  {stage}: {len(dups)} control arms, max |vector - base| over all of them = {worst:.3e}")

    for stage in ("stageB", "stageC"):
        labels, mats, ref_mat, dom = load_cell(stage)
        jstar = ref_top1_index(ref_mat)
        b_own, b_marg = gap_stats(mats["base"], jstar)
        b_m = metrics_for(mats["base"], jstar)
        print(f"\n=== {stage} ===")
        for arm in ARMS[stage]:
            c_own, c_marg = gap_stats(mats[arm], jstar)
            c_m = metrics_for(mats[arm], jstar)
            print(f"  {arm}")
            for pop, mask in (("pooled", np.ones(len(labels), bool)),
                              ("id", dom == "id"), ("ood", dom == "ood")):
                pm = paired(c_marg[mask], b_marg[mask])
                po = paired(c_own[mask], b_own[mask])
                ps = paired((c_marg - c_own)[mask], (b_marg - b_own)[mask])
                pr = paired(c_m[0][mask], b_m[0][mask])
                print(f"    {pop:>6} n={pm['n']:>3}  margin_top1 {pm['abs_ratio']:>6.3f} {verdict(pm['abs_ratio']):<12}"
                      f" | placebo_own(correctness-blind) {po['abs_ratio']:>6.3f} {verdict(po['abs_ratio']):<12}"
                      f" | margin_shortfall {ps['abs_ratio']:>6.3f} {verdict(ps['abs_ratio']):<12}"
                      f" | recall1 {pr['abs_ratio']:>6.3f}")


if __name__ == "__main__":
    main()

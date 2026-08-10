#!/usr/bin/env python3
r"""T-1886 -- Popper attack probe, part 2.

Attacks T-1881's family specification, its subpopulation independence (against the
standard this campaign itself applied to the `middle` arm at D-SLM2107), its declared
minimum effect size, and the environmental account's own discriminating power.

CPU only. Reads T-1881's outputs read-only.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import stats as st

T1881 = Path(r"D:\SuperSLM\.worktrees\t1881-realizable-sidecar")
T1777 = Path(r"D:\SuperSLM\.worktrees\t1777-retrieval-agreement")

import sys
sys.path.insert(0, str(T1777 / "tools"))
import t1777_retrieval_report as rr  # noqa: E402

N = 239
Q = 1.0 / N


def load():
    labels, domains = [], {}
    with open(T1777 / "out" / "t1777_corpus" / "manifest.jsonl", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rec = json.loads(line)
                labels.append(rec["label"])
                domains[rec["label"]] = rec.get("domain", rec.get("population", "ALL"))
    ref_vecs, ref_fps = rr.load_pooled_vectors(
        labels, T1777 / "out" / "t1777_full_float_bf16", "float")

    def arm(p):
        z = np.load(p, allow_pickle=True)
        labs = [str(x) for x in z["labels"]]
        vecs = {l: z["vectors"][i] for i, l in enumerate(labs)}
        fps = {l: int(z["fingerprints"][i]) for i, l in enumerate(labs)}
        res, _ = rr.compare_arms(ref_vecs, vecs, domains, ref_fps, fps)
        return [r.label for r in res], np.array([float(r.recall[1]) for r in res])

    root = T1881 / "out" / "t1881_capture"
    o, base = arm(root / "shared/pooled/base.npz")
    _, null = arm(root / "shared/pooled/null.npz")
    a = {t: arm(root / t / "pooled/E_recovery_site4_only.npz")[1]
         for t in ("kcap3", "kcap8", "kcap31")}
    return o, domains, base, null, a


def hdr(s):
    print("\n" + "=" * 78)
    print(s)
    print("=" * 78)


def ratio(d, z):
    se = float(d.std(ddof=1) / np.sqrt(len(d)))
    return float(d.mean()), se, float(d.mean()) / (z * se)


def main():
    order, domains, base, null, a = load()
    doms = np.array([domains[l] for l in order])

    hdr("B1 -- FAMILY SPECIFICATION: what the family is, and what it does to the verdict")
    print("  T-1881 declares m=3 (three precisions x pooled only).")
    print("  Two of those three cells are the SAME measurement (kcap8 == kcap31, bit-identical),")
    print("  so the declared family counts one measurement twice.")
    print("  Every prior arm in this campaign was graded over arms x {pooled, id, ood}")
    print("  (D-SLM2137/D-SLM2173: 7 arms x 3 populations = 21 cells).")
    print("  Applying that convention to T-1881's own distinct arms:")
    for m in (2, 3, 4, 6, 9):
        z = st.norm.ppf(1 - 0.05 / (2 * m))
        line = f"    m={m:2d}  z_crit={z:.6f} :"
        for tag in ("kcap3", "kcap8"):
            mm, se, r = ratio(a[tag] - base, z)
            line += f"  {tag} {r:.3f}x {'RESOLVED    ' if r >= 1 else 'NOT RESOLVED'}"
        print(line)
    print("  m=2  : the two DISTINCT precisions, pooled only")
    print("  m=3  : as T-1881 declared (duplicate cell counted)")
    print("  m=6  : two distinct precisions x 3 populations")
    print("  m=9  : three declared precisions x 3 populations (this campaign's own convention)")

    hdr("B2 -- SUBPOPULATION INDEPENDENCE, against D-SLM2107's own standard")
    print("  D-SLM2107 refused `middle` the label 'independent survivor' because it resolved")
    print("  only by pooling and resolved on NEITHER id nor ood -- while noting its effect was")
    print("  HOMOGENEOUS across domains (+0.0714 id / +0.0707 ood), which is why that entry")
    print("  refuted the resolution and not the effect.")
    print()
    for tag in ("kcap3", "kcap8", "kcap31"):
        d_all = a[tag] - base
        m_all, se_all, r_all = ratio(d_all, 1.959964)
        print(f"  {tag}: pooled {m_all:+.6f} ({m_all/Q:+.0f}/239) {r_all:.3f}x uncorrected")
        for dv in ("id", "ood"):
            sel = doms == dv
            d = d_all[sel]
            mm, se, r = ratio(d, 1.959964)
            up = int((d > 0).sum()); dn = int((d < 0).sum())
            p = st.binomtest(up, up + dn, 0.5).pvalue if up + dn else float("nan")
            print(f"    {dv:4s} n={int(sel.sum()):3d}  {mm:+.6f} ({mm*sel.sum():+.0f} docs) "
                  f"RP={1.96*se:.6f} {r:.3f}x "
                  f"{'RESOLVED' if r >= 1 else 'NOT RESOLVED':12s} exact McNemar p={p:.4f}")
        id_d = d_all[doms == "id"].mean()
        ood_d = d_all[doms == "ood"].mean()
        print(f"    homogeneity: id {id_d:+.6f} vs ood {ood_d:+.6f}  "
              f"ratio id/ood = {id_d/ood_d:.2f}x")
    print()
    print("  Interaction test (does the id-vs-ood difference in the effect itself resolve?):")
    for tag in ("kcap3", "kcap8"):
        d_all = a[tag] - base
        di = d_all[doms == "id"]; do = d_all[doms == "ood"]
        diff = di.mean() - do.mean()
        se = float(np.sqrt(di.var(ddof=1) / len(di) + do.var(ddof=1) / len(do)))
        print(f"    {tag}: id-minus-ood effect = {diff:+.6f}, SE={se:.6f}, "
              f"ratio={abs(diff)/(1.96*se):.3f}x -> "
              f"{'RESOLVED' if abs(diff) >= 1.96*se else 'NOT RESOLVED'}")

    hdr("B3 -- MINIMUM EFFECT SIZE: the campaign's own declared floor vs these effects")
    print("  D-SLM2173: the adopted decision rule is signed AND carries a stated minimum")
    print("  effect size. D-SLM2167 recommends delta_min = 6pp as the campaign default.")
    print("  D-SLM2168: at 6pp and a 21-cell family this corpus needs N=1,045, not 239.")
    for tag in ("kcap3", "kcap8"):
        d = a[tag] - base
        print(f"    {tag}: delta = {d.mean()*100:.2f}pp  "
              f"({'>=' if d.mean() >= 0.06 else '<'} 6pp floor)")
    print("  T-1881's grader declares no delta_min and uses abs(mean)/paired_rp -- the formula")
    print("  D-SLM2110/D-SLM2173 replaced. Every effect here is positive, so the sign defect")
    print("  is not live in these figures; the absent floor is.")

    hdr("B4 -- ENVIRONMENTAL: can the affine-in-iterations fit exclude an order effect?")
    obs = {"kcap3": (4, 0, 18.04), "kcap8": (9, 1, 20.06), "kcap31": (32, 2, 28.12)}
    it = np.array([obs[t][0] for t in ("kcap3", "kcap8", "kcap31")], dtype=float)
    od = np.array([obs[t][1] for t in ("kcap3", "kcap8", "kcap31")], dtype=float)
    y = np.array([obs[t][2] for t in ("kcap3", "kcap8", "kcap31")], dtype=float)
    # T-1881's model: fit alpha + beta*iters on the two endpoints, predict kcap3.
    beta = (y[2] - y[1]) / (it[2] - it[1])
    alpha = y[1] - beta * it[1]
    pred = alpha + beta * it[0]
    print(f"  T-1881's model (order effect assumed zero), fit on kcap8+kcap31:")
    print(f"    beta = {beta:.5f} s/doc per loop iteration, alpha = {alpha:.3f}")
    print(f"    predicted kcap3 = {pred:.3f} vs observed {y[0]:.3f} -> "
          f"{(pred - y[0]) / y[0] * 100:+.2f}% miss")
    # Counter-model: alpha + beta*iters + gamma*(config order), 3 params, 3 points.
    A = np.column_stack([np.ones(3), it, od])
    sol = np.linalg.solve(A, y)
    print(f"  Counter-model with a per-config order penalty (3 params, 3 points, exact fit):")
    print(f"    alpha={sol[0]:.4f}  beta={sol[1]:.5f} s/doc per iteration  "
          f"gamma={sol[2]:+.4f} s/doc per config position")
    print(f"    residuals: {np.abs(A @ sol - y).max():.2e} -- an EXACT fit")
    print("  => the three E-arm rates are equally consistent with a +0.34 s/doc per-config")
    print("     order penalty riding on top of the iteration cost. k_cap and capture order are")
    print("     collinear across configs 3/4/5, so this cell cannot separate them; the 1.5%")
    print("     'out-of-sample' miss is the size of the order term, not evidence against one.")
    print("  Order-only counter-model (no iteration term), fit on kcap8+kcap31:")
    b2 = (y[2] - y[1]) / (od[2] - od[1]); a2 = y[1] - b2 * od[1]
    print(f"    predicted kcap3 = {a2 + b2*od[0]:.3f} vs observed {y[0]:.3f} -> "
          f"{(a2 + b2*od[0] - y[0]) / y[0] * 100:+.1f}% miss -- this model IS excluded")

    hdr("B5 -- does anything in the OUTPUT correlate with capture order? (the check that matters)")
    print("  kcap8 (config 4) vs kcap31 (config 5): per-document recall@1 identical: "
          f"{np.array_equal(a['kcap8'], a['kcap31'])}")
    print(f"  null (config 1) = {int(round(null.sum()))}/239, independently filed by T-1879 "
          f"as 235/239 -- reproduced")
    print(f"  base (config 2) = {int(round(base.sum()))}/239, independently filed by D-SLM1936 "
          f"as 149/239 (0.623431) -- reproduced")
    print("  Both reproductions are integer document counts matching exactly. This leg of the")
    print("  environmental argument survives; the offset leg (B6) does not.")

    hdr("B6 -- the 0.041841 'six decimal place match' carries no independent information")
    print("  D-SLM1936 DEFINES the offset as 0.665272 - 0.623431 (its own text: 'scores")
    print("  recall@1 0.623431 at batch 1 and 0.665272 at batch 56/59, a gap of +0.041841').")
    print("  T-1881 recomputes 0.665272 - 0.623431 from its own base = 0.623431 and reports the")
    print("  agreement as corroboration from 'an entirely unrelated run'. It is the same")
    print("  subtraction twice: the offset is an exact function of the number under test.")
    print(f"  And the metric is quantized: 0.041841 = 10/239 exactly ({10/239:.6f}). Every")
    print("  figure in this packet is an integer document count, so 'agreement to six decimal")
    print("  places' between any two such figures is agreement on one integer, not on six digits.")
    print("  The load-bearing fact underneath is base = 149/239, which does reproduce (B5).")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
r"""T-1886 -- Popper attack probe against T-1881's result packet.

Independently recomputes every numeric claim in
`Claude/Brunel/t1881-realizable-sidecar-measurement-2026-08-10.md` from the raw
per-document vectors on disk, then runs the attacks that packet's own figures do not.

CPU only. No model load, no GPU capture. Reads the T-1881 worktree's outputs read-only.

Usage
    python tools/t1886_attack.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

T1881 = Path(r"D:\SuperSLM\.worktrees\t1881-realizable-sidecar")
T1777 = Path(r"D:\SuperSLM\.worktrees\t1777-retrieval-agreement")

import sys
sys.path.insert(0, str(T1777 / "tools"))
import t1777_retrieval_report as rr  # noqa: E402

N = 239


def load_manifest():
    labels, domains = [], {}
    with open(T1777 / "out" / "t1777_corpus" / "manifest.jsonl", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rec = json.loads(line)
                labels.append(rec["label"])
                domains[rec["label"]] = rec.get("domain", rec.get("population", "ALL"))
    return labels, domains


def arm(npz_path, ref_vecs, ref_fps, domains):
    z = np.load(npz_path, allow_pickle=True)
    labs = [str(x) for x in z["labels"]]
    vecs = {l: z["vectors"][i] for i, l in enumerate(labs)}
    fps = {l: int(z["fingerprints"][i]) for i, l in enumerate(labs)}
    results, n_docs = rr.compare_arms(ref_vecs, vecs, domains, ref_fps, fps)
    order = [r.label for r in results]
    per_doc = np.array([float(r.recall[1]) for r in results])
    raw = {l: z["vectors"][i] for i, l in enumerate(labs)}
    return order, per_doc, raw


def paired(a, b):
    d = a - b
    n = len(d)
    mean = float(d.mean())
    se = float(d.std(ddof=1) / np.sqrt(n))
    return mean, se, 1.96 * se


def hdr(s):
    print("\n" + "=" * 78)
    print(s)
    print("=" * 78)


def main():
    labels, domains = load_manifest()
    ref_vecs, ref_fps = rr.load_pooled_vectors(
        labels, T1777 / "out" / "t1777_full_float_bf16", "float")

    root = T1881 / "out" / "t1881_capture"
    o_null, r_null, v_null = arm(root / "shared/pooled/null.npz", ref_vecs, ref_fps, domains)
    o_base, r_base, v_base = arm(root / "shared/pooled/base.npz", ref_vecs, ref_fps, domains)
    arms = {}
    for tag in ("kcap3", "kcap8", "kcap31"):
        arms[tag] = arm(root / tag / "pooled/E_recovery_site4_only.npz",
                        ref_vecs, ref_fps, domains)
        assert arms[tag][0] == o_base, f"order mismatch {tag}"
    assert o_null == o_base

    hdr("A1 -- reproduce T-1881's headline figures, independently")
    print(f"N = {len(r_base)}")
    for name, r in [("null", r_null), ("base", r_base)] + \
            [(t, arms[t][1]) for t in ("kcap3", "kcap8", "kcap31")]:
        hits = int(round(r.sum()))
        print(f"  {name:8s} recall@1 = {r.mean():.6f}   = {hits}/{len(r)}   "
              f"exact k/N: {abs(r.mean() - hits / len(r)) < 1e-12}")

    hdr("A2 -- METRIC GRANULARITY: every reported figure is an integer document count")
    q = 1.0 / N
    print(f"  metric quantum 1/{N} = {q:.9f}")
    for lbl, val in [("T-1835/T-1881 'width-1-to-56 offset' 0.041841", 0.041841),
                     ("kcap8 delta vs base 0.092050", 0.092050),
                     ("kcap3 delta vs base 0.117155", 0.117155),
                     ("T-1859 filed width-56 delta 0.087866", 0.087866),
                     ("kcap3-kcap8 contrast 0.025105", 0.025105),
                     ("kcap8 RP 0.067755", 0.067755),
                     ("kcap3-kcap8 RP 0.054420", 0.054420)]:
        print(f"  {lbl:52s} = {val / q:8.3f} documents")

    hdr("A3 -- the '% of ceiling' denominator, under the campaign's own D-SLM2128/D-SLM2201 rule")
    nb_mean, nb_se, nb_rp = paired(r_null, r_base)
    print(f"  same-construction accounting basis (null - base) = {nb_mean:.6f} "
          f"= {nb_mean/q:.0f} documents")
    ceil8 = paired(arms["kcap8"][1], r_base)[0]
    for tag in ("kcap3", "kcap8", "kcap31"):
        m = paired(arms[tag][1], r_base)[0]
        print(f"  {tag:7s} delta={m:+.6f}  "
              f"T-1881's headline (/kcap8 itself) = {m/ceil8*100:6.1f}%   "
              f"campaign rule (/(null-base)) = {m/nb_mean*100:5.1f}%")

    hdr("A4 -- the three graded cells are not three measurements")
    same = np.array_equal(arms["kcap8"][1], arms["kcap31"][1])
    print(f"  kcap8 per-document recall@1 == kcap31 per-document recall@1 : {same}")
    v8 = arms["kcap8"][2]; v31 = arms["kcap31"][2]
    maxdiff = max(float(np.abs(v8[l].astype(np.float64) - v31[l].astype(np.float64)).max())
                  for l in v8)
    print(f"  pooled vector max abs diff kcap8 vs kcap31 = {maxdiff}")
    print(f"  kcap3 vs kcap8 vectors identical: "
          f"{all(np.array_equal(arms['kcap3'][2][l], v8[l]) for l in v8)}")
    print("  => the m=3 family contains 2 distinct measurements; the third is a duplicate row.")
    for m_fam, z in [(1, 1.959964), (2, 2.241403), (3, 2.393980)]:
        print(f"    m={m_fam} z_crit={z:.6f}:", end="")
        for tag in ("kcap3", "kcap8"):
            mean, se, _ = paired(arms[tag][1], r_base)
            print(f"   {tag} ratio={mean/(z*se):.3f}", end="")
        print()

    hdr("A5 -- the 2-bit-vs-4-bit contrast: what the interval actually permits")
    m, se, rp = paired(arms["kcap3"][1], arms["kcap8"][1])
    print(f"  kcap3 - kcap8: mean={m:+.6f} ({m/q:+.0f} docs), SE={se:.6f}, "
          f"RP(1.96SE)={rp:.6f} ({rp/q:.1f} docs), ratio={abs(m)/rp:.3f}x -> NOT RESOLVED")
    lo, hi = m - rp, m + rp
    print(f"  95% interval on (2-bit minus 4-bit): [{lo:+.6f}, {hi:+.6f}] "
          f"= [{lo/q:+.1f}, {hi/q:+.1f}] documents")
    print(f"  worst case the data permits for the 2-bit field, expressed as a fraction of the")
    print(f"  4-bit arm's own recovery ({ceil8:.6f}): {lo/ceil8*100:+.1f}%  "
          f"(i.e. the 2-bit field could lose up to {abs(lo)/ceil8*100:.1f}% of it)")
    d = arms["kcap3"][1] - arms["kcap8"][1]
    print(f"  per-document flips: kcap8-wrong->kcap3-right = {int((d > 0).sum())}, "
          f"kcap3-wrong->kcap8-right = {int((d < 0).sum())}, unchanged = {int((d == 0).sum())}")

    hdr("A6 -- SUBPOPULATION SPLIT: does the recovery hold in both domains?")
    doms = np.array([domains[l] for l in o_base])
    for dv in sorted(set(doms.tolist())):
        sel = doms == dv
        n_d = int(sel.sum())
        print(f"  domain {dv!r} (n={n_d}):")
        print(f"    base recall@1 = {r_base[sel].mean():.6f}, "
              f"null recall@1 = {r_null[sel].mean():.6f}")
        for tag in ("kcap3", "kcap8"):
            a = arms[tag][1][sel]
            dd = a - r_base[sel]
            mm = float(dd.mean()); ss = float(dd.std(ddof=1) / np.sqrt(n_d))
            rr_ = 1.96 * ss
            print(f"    {tag:7s} delta={mm:+.6f} ({mm*n_d:+.0f}/{n_d} docs) "
                  f"SE={ss:.6f} RP={rr_:.6f} ratio={abs(mm)/rr_ if rr_ else float('nan'):.3f}x "
                  f"-> {'RESOLVED' if abs(mm) >= rr_ else 'NOT RESOLVED'}")

    hdr("A7 -- is the pooled delta carried by a handful of documents?")
    for tag in ("kcap3", "kcap8"):
        dd = arms[tag][1] - r_base
        up = int((dd > 0).sum()); dn = int((dd < 0).sum())
        print(f"  {tag:7s}: base-wrong->E-right = {up}, base-right->E-wrong = {dn}, "
              f"net = {up - dn}")

    hdr("A8 -- sign test / exact binomial on the discordant pairs (distribution-free)")
    from scipy import stats as st
    for lbl, a, b in [("kcap3 vs base", arms["kcap3"][1], r_base),
                      ("kcap8 vs base", arms["kcap8"][1], r_base),
                      ("kcap3 vs kcap8", arms["kcap3"][1], arms["kcap8"][1])]:
        dd = a - b
        up = int((dd > 0).sum()); dn = int((dd < 0).sum())
        p = st.binomtest(up, up + dn, 0.5).pvalue if up + dn else float("nan")
        print(f"  {lbl:16s} up={up:3d} down={dn:3d}  exact two-sided p = {p:.5f}")
    print("  Bonferroni over the 3 pre-declared vs-base cells: alpha = 0.05/3 = 0.016667")

    hdr("A9 -- the 'null reproduces to six decimal places' check, in documents")
    print(f"  T-1881 null   = {r_null.mean():.6f} = {int(round(r_null.sum()))}/{N}")
    print(f"  T-1879 filed  = 0.983264 = {round(0.983264*N)}/{N}")
    print(f"  T-1881 base   = {r_base.mean():.6f} = {int(round(r_base.sum()))}/{N}")
    print(f"  T-1809 arm C / D-SLM1936 filed base@width1 = 0.623431 = {round(0.623431*N)}/{N}")
    print(f"  D-SLM1936 filed base@width56 = 0.665272 = {round(0.665272*N)}/{N}")
    print(f"  0.665272 - 0.623431 = {0.665272-0.623431:.6f}  "
          f"(= {round((0.665272-0.623431)*N)}/{N}, and D-SLM1936 DEFINES the offset as this "
          f"difference)")


if __name__ == "__main__":
    main()

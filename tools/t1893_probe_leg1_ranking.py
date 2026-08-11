#!/usr/bin/env python3
"""T-1893 DEBUNK PROBE (disposable). Leg 1's ranking null, checked past k=1.

The packet reports ONE ranking figure -- recall@1 -- and calls it not
distinguishable from zero. Three things that null does not establish, each
executed here:

  1. OTHER k. t1891_paired_recall_delta.py passes ks=(1,) into compare_arms.
     recall@5 / recall@10 / Spearman rho / the reference top-1 neighbour's
     candidate-side rank are all computed by the same instrument and were not
     reported. A directional effect visible at k=5 or in rho would mean the
     null is a property of the k that was reported, not of the arms.
  2. RESOLVING POWER OF THE INSTRUMENT ITSELF. recall@1 is a strict binary on
     a top-1 that the reference may itself barely resolve. If the arms disagree
     mostly where the reference's own similarity gap is near zero, recall@1 is
     reading near-ties, and "no difference detected" is partly a statement
     about the instrument.
  3. COMMON MODE. Per-document cosine-to-float improves detectably while
     ranking does not move. Ranking is invariant to any error component shared
     across documents. Measure how much of each arm's error vector is common
     mode; if most of it is, a per-document drift win is expected to be
     invisible to retrieval, and the two results are consistent rather than in
     tension -- which is a limit on what the drift figure can be used to argue.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import t1777_retrieval_report as trr  # noqa: E402

BASE = Path(r"D:\SuperSLM\.worktrees\t1777-retrieval-agreement\out\t1891_capture")
MANIFEST = Path(r"D:\SuperSLM\.worktrees\t1777-retrieval-agreement\out\t1777_corpus\manifest.jsonl")


def mcnemar_hw(p01, p10, n, z=1.96):
    d = p01 - p10
    return z * math.sqrt(max(p01 + p10 - d * d, 0.0) / n)


def paired_cont(a, b, z=1.96):
    d = np.array(a) - np.array(b)
    return float(d.mean()), z * float(d.std(ddof=1)) / math.sqrt(len(d))


def main() -> int:
    recs = [json.loads(l) for l in MANIFEST.read_text(encoding="utf-8").splitlines() if l.strip()]
    labels = [r["label"] for r in recs]
    domains = {r["label"]: r["domain"] for r in recs}

    fl, ffp = trr.load_pooled_vectors(labels, BASE / "float_bf16", "float")
    ol, ofp = trr.load_pooled_vectors(labels, BASE / "int8_old", "int8")
    fu, ufp = trr.load_pooled_vectors(labels, BASE / "int8_fused", "int8")

    ks = (1, 5, 10)
    ro, _ = trr.compare_arms(fl, ol, domains, ffp, ofp, ks=ks)
    rf, _ = trr.compare_arms(fl, fu, domains, ffp, ufp, ks=ks)
    O = {r.label: r for r in ro}
    F = {r.label: r for r in rf}
    common = sorted(set(O) & set(F))
    n = len(common)
    print(f"N={n}\n")

    print("=== 1. recall@k, paired old vs fused (packet reported k=1 only) ===")
    for k in ks:
        a = [O[l].recall[k] for l in common]
        b = [F[l].recall[k] for l in common]
        if k == 1:
            oc = [x >= 0.999 for x in a]
            fc = [x >= 0.999 for x in b]
            p01 = sum(1 for o, f in zip(oc, fc) if (not o) and f) / n
            p10 = sum(1 for o, f in zip(oc, fc) if o and (not f)) / n
            delta = sum(fc) / n - sum(oc) / n
            hw = max(1.0 / n, mcnemar_hw(p01, p10, n))
        else:
            d, hw = paired_cont(b, a)
            delta = d
            hw = max(1.0 / (n * k), hw)
        v = "DETECTABLE" if abs(delta) > hw else "NOT DISTINGUISHABLE FROM ZERO"
        print(f"recall@{k:<2}  old={np.mean(a):.4f}  fused={np.mean(b):.4f}  "
              f"delta(fused-old)={delta:+.4f}  rp={hw:.4f}  {v}")

    print("\n=== Spearman rho of each arm's full ranking vs the float reference ===")
    a = [O[l].spearman_rho for l in common]
    b = [F[l].spearman_rho for l in common]
    d, hw = paired_cont(b, a)
    print(f"rho     old={np.mean(a):.4f}  fused={np.mean(b):.4f}  delta(fused-old)={d:+.4f}  "
          f"rp={hw:.4f}  {'DETECTABLE' if abs(d) > hw else 'NOT DISTINGUISHABLE FROM ZERO'}")

    print("\n=== reference top-1 neighbour's candidate-side rank (lower = better) ===")
    a = [O[l].ref_top1_displacement for l in common]
    b = [F[l].ref_top1_displacement for l in common]
    d, hw = paired_cont(b, a)
    print(f"mean    old={np.mean(a):.3f}  fused={np.mean(b):.3f}  delta(fused-old)={d:+.3f}  "
          f"rp={hw:.3f}  {'DETECTABLE' if abs(d) > hw else 'NOT DISTINGUISHABLE FROM ZERO'}")
    print(f"median  old={np.median(a):.1f}   fused={np.median(b):.1f}")
    print(f"within top-5: old={np.mean([x < 5 for x in a]):.4f}  fused={np.mean([x < 5 for x in b]):.4f}")

    print("\n=== 2. is recall@1 reading near-ties the reference barely resolves? ===")
    # reference-side similarity gap between the reference's top-1 and top-2 for
    # the queries where each arm gets top-1 wrong vs right.
    ref_mat = np.stack([fl[l] for l in common])
    ref_sims = trr.cosine_sim_matrix(ref_mat)
    gaps = []
    for qi in range(n):
        _, ranked = trr.ranks_desc(ref_sims[qi], qi)
        gaps.append(float(ref_sims[qi, ranked[0]] - ref_sims[qi, ranked[1]]))
    gaps = np.array(gaps)
    for name, res in (("old", O), ("fused", F)):
        ok = np.array([res[l].recall[1] >= 0.999 for l in common])
        print(f"{name:<6} ref top1-top2 gap  |  correct: median={np.median(gaps[ok]):.5f}  "
              f"wrong: median={np.median(gaps[~ok]):.5f}  "
              f"ratio={np.median(gaps[ok]) / max(np.median(gaps[~ok]), 1e-12):.1f}x")
    print(f"overall ref top1-top2 gap: median={np.median(gaps):.5f} p10={np.percentile(gaps,10):.5f} "
          f"p90={np.percentile(gaps,90):.5f}")
    # what fraction of queries have a reference gap smaller than the typical
    # per-document cosine perturbation each arm introduces?
    for name, arm in (("old", ol), ("fused", fu)):
        cm = np.stack([arm[l] for l in common])
        csims = trr.cosine_sim_matrix(cm)
        perturb = np.median(np.abs(csims - ref_sims)[np.triu_indices(n, 1)])
        print(f"{name:<6} median |sim(cand)-sim(ref)| over all doc pairs = {perturb:.5f}  "
              f"-> {float((gaps < perturb).mean())*100:.1f}% of queries have a reference "
              f"top1-top2 gap SMALLER than that")

    print("\n=== 3. common-mode share of each arm's error vs float ===")
    for name, arm in (("old", ol), ("fused", fu)):
        E = np.stack([arm[l] - fl[l] for l in common])
        mu = E.mean(axis=0)
        tot = float((E ** 2).sum())
        cm = float(n * (mu ** 2).sum())
        print(f"{name:<6} ||mean error||^2 * N / total error energy = {cm / tot * 100:.1f}% "
              f"(common mode); residual {100 - cm / tot * 100:.1f}%")
    # and the common-mode component's alignment with the mean float vector
    mu_f = np.stack([fl[l] for l in common]).mean(axis=0)
    for name, arm in (("old", ol), ("fused", fu)):
        mu_e = np.stack([arm[l] - fl[l] for l in common]).mean(axis=0)
        c = float(np.dot(mu_e, mu_f) / (np.linalg.norm(mu_e) * np.linalg.norm(mu_f)))
        print(f"{name:<6} cos(mean error, mean float vector) = {c:+.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

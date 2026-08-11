#!/usr/bin/env python3
"""T-1893 DEBUNK PROBE (disposable). Two follow-ups to the leg-1 ranking probe.

A. The Spearman rho delta (fused - old = -0.0033, rp 0.0028) clears the packet's
   own DETECTABLE rule by only ~1.2x. Before it is reported as a counter-result
   it is checked three further ways that do not share the paired-t assumption:
   a per-document sign test, a bootstrap over documents, and the same figures
   split by domain (id / ood).

B. The mechanism. Ranking is invariant to any error component shared across all
   documents. If the fused arm's error is smaller overall but its
   RANKING-RELEVANT residual (the document-specific part, after removing the
   common mode) is not smaller -- or is larger -- then "fused sits closer to the
   float reference" is true of a component retrieval cannot see, while the
   component retrieval CAN see did not improve. Absolute energies, not shares.
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
RNG = np.random.default_rng(20260810)


def main() -> int:
    recs = [json.loads(l) for l in MANIFEST.read_text(encoding="utf-8").splitlines() if l.strip()]
    labels = [r["label"] for r in recs]
    domains = {r["label"]: r["domain"] for r in recs}
    fl, ffp = trr.load_pooled_vectors(labels, BASE / "float_bf16", "float")
    ol, ofp = trr.load_pooled_vectors(labels, BASE / "int8_old", "int8")
    fu, ufp = trr.load_pooled_vectors(labels, BASE / "int8_fused", "int8")

    ro, _ = trr.compare_arms(fl, ol, domains, ffp, ofp, ks=(1,))
    rf, _ = trr.compare_arms(fl, fu, domains, ffp, ufp, ks=(1,))
    O = {r.label: r for r in ro}
    F = {r.label: r for r in rf}
    common = sorted(set(O) & set(F))
    n = len(common)

    print("=== A. is the Spearman rho regression robust? ===")
    a = np.array([O[l].spearman_rho for l in common])
    b = np.array([F[l].spearman_rho for l in common])
    d = b - a
    print(f"N={n}  mean rho old={a.mean():.5f} fused={b.mean():.5f}  delta={d.mean():+.5f}")
    hw = 1.96 * d.std(ddof=1) / math.sqrt(n)
    print(f"paired-t 95% half-width = {hw:.5f}  -> ratio |delta|/rp = {abs(d.mean())/hw:.2f}x")

    nb, nw, nt = int((d < 0).sum()), int((d > 0).sum()), int((d == 0).sum())
    # two-sided exact sign test
    m = nb + nw
    k = min(nb, nw)
    p_sign = 2 * sum(math.comb(m, i) for i in range(k + 1)) / (2 ** m)
    print(f"per-document sign test: fused worse on {nb}, better on {nw}, tied {nt} "
          f"-> two-sided p = {min(p_sign,1.0):.4f}")

    boot = np.array([d[RNG.integers(0, n, n)].mean() for _ in range(20000)])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    print(f"bootstrap (20k, over documents) 95% CI on delta: [{lo:+.5f}, {hi:+.5f}]  "
          f"fraction of resamples with delta<0: {float((boot<0).mean()):.4f}")

    for dom in ("id", "ood"):
        idx = [i for i, l in enumerate(common) if domains[l] == dom]
        dd = d[idx]
        h = 1.96 * dd.std(ddof=1) / math.sqrt(len(dd))
        print(f"  domain={dom:<4} N={len(dd):<4} delta={dd.mean():+.5f}  rp={h:.5f}  "
              f"{'DETECTABLE' if abs(dd.mean())>h else 'NOT DISTINGUISHABLE FROM ZERO'}")

    print("\n=== B. common mode vs ranking-relevant residual, ABSOLUTE energies ===")
    stats = {}
    for name, arm in (("old", ol), ("fused", fu)):
        E = np.stack([arm[l] - fl[l] for l in common])
        mu = E.mean(axis=0)
        total = float((E ** 2).sum())
        cmode = float(n * (mu ** 2).sum())
        resid = total - cmode
        stats[name] = (total, cmode, resid)
        print(f"{name:<6} total={total:.4f}  common-mode={cmode:.4f} ({cmode/total*100:.1f}%)  "
              f"residual={resid:.4f} ({resid/total*100:.1f}%)")
    to, co, so = stats["old"]
    tf, cf, sf = stats["fused"]
    print(f"\ndelta (fused - old): total={tf-to:+.4f} ({(tf-to)/to*100:+.1f}%)  "
          f"common-mode={cf-co:+.4f} ({(cf-co)/co*100:+.1f}%)  "
          f"residual={sf-so:+.4f} ({(sf-so)/so*100:+.1f}%)")

    # per-document, scale-free: residual angular error after removing common mode
    print("\nper-document, after subtracting each arm's own mean error vector "
          "(the part ranking can actually see):")
    rows = {}
    for name, arm in (("old", ol), ("fused", fu)):
        E = np.stack([arm[l] - fl[l] for l in common])
        mu = E.mean(axis=0)
        R = E - mu
        rel = np.array([np.linalg.norm(R[i]) / np.linalg.norm(fl[l]) for i, l in enumerate(common)])
        rows[name] = rel
        print(f"  {name:<6} mean relative residual error = {rel.mean():.6f}")
    dr = rows["old"] - rows["fused"]
    h = 1.96 * dr.std(ddof=1) / math.sqrt(n)
    print(f"  paired delta (old - fused) = {dr.mean():+.6f}  rp={h:.6f}  "
          f"{'DETECTABLE' if abs(dr.mean())>h else 'NOT DISTINGUISHABLE FROM ZERO'} "
          f"(positive = fused better)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

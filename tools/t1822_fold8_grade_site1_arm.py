#!/usr/bin/env python3
r"""T-1822 fold 8 -- grade the site-1 oracle arms: forgone refinement IN vs OUT.

Grades through T-1777's own unmodified report functions (the t1809_grade_arms
pattern: compare_arms called directly, T-1777's committed true-bf16 dumps on the
reference side in every row).  Reports:

  * recall@1 (the binding metric, D-SLM1455) for both arms, with each arm's own
    resolving power;
  * the PAIRED contrast: per-document top-1 indicator difference
    (refined - row), its mean, and the achieved paired resolving power
    1.96 * SE(d_i) -- the arms share the corpus and query order, asserted;
  * the run-conditions bridge: byte-compare of this session's row-grid arm
    against T-1809's filed arm C dumps (the capture is deterministic on this
    machine if they are identical).

Usage (from D:\SuperSLM\.worktrees\t1822-fold8):
    python tools\t1822_fold8_grade_site1_arm.py \
        --t1777-tools D:\SuperSLM\.worktrees\t1777-retrieval-agreement\tools \
        --t1777-out   D:\SuperSLM\.worktrees\t1777-retrieval-agreement\out \
        --row-dir     out\t1822_arm_c_rerun \
        --refined-dir out\t1822_arm_s1r \
        --t1809-arm-c D:\SuperSLM\.worktrees\t1809-activation-interaction\out\t1809_arm_c
"""
from __future__ import annotations

import argparse
import filecmp
import json
import math
import sys
from pathlib import Path

import numpy as np


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--t1777-tools", required=True)
    ap.add_argument("--t1777-out", required=True)
    ap.add_argument("--row-dir", required=True)
    ap.add_argument("--refined-dir", required=True)
    ap.add_argument("--t1809-arm-c", default=None)
    args = ap.parse_args(argv)

    sys.path.insert(0, str(Path(args.t1777_tools).resolve()))
    import t1777_retrieval_report as rr

    out = Path(args.t1777_out)
    labels, domains = [], {}
    with open(out / "t1777_corpus" / "manifest.jsonl", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rec = json.loads(line)
                labels.append(rec["label"])
                domains[rec["label"]] = rec.get("domain", rec.get("population", "ALL"))

    ref_vecs, ref_fps = rr.load_pooled_vectors(labels, out / "t1777_full_float_bf16", "float")

    per_query = {}
    rows = {}
    for name, d in (("row (site 1 on the row grid)", Path(args.row_dir)),
                    ("refined (site 1 refinement retained)", Path(args.refined_dir))):
        cand_vecs, cand_fps = rr.load_pooled_vectors(labels, d, "float")
        results, n_docs = rr.compare_arms(ref_vecs, cand_vecs, domains, ref_fps, cand_fps)
        r1 = np.array([r.recall[1] for r in results], dtype=np.float64)
        mean = float(r1.mean())
        rp = float(max(rr.discrete_step_recall(n_docs, 1),
                       rr.normal_ci_halfwidth(mean, n_docs)))
        rows[name] = (mean, rp, int(r1.sum()), n_docs)
        per_query[name] = r1
        print(f"{name}: recall@1 = {mean:.6f} ({int(r1.sum())}/{n_docs}), "
              f"own resolving power {rp:.4f}")

    (row_name, ref_name) = list(rows)
    d = per_query[ref_name] - per_query[row_name]     # refined - row, per query
    n = d.size
    mean_d = float(d.mean())
    se = float(d.std(ddof=1) / math.sqrt(n))
    paired_rp = 1.96 * se
    print(f"\npaired contrast (refined - row), N = {n}:")
    print(f"  mean difference = {mean_d:+.6f}")
    print(f"  discordant documents: {int((d != 0).sum())} "
          f"({int((d > 0).sum())} refined-only correct, "
          f"{int((d < 0).sum())} row-only correct)")
    print(f"  achieved paired resolving power (1.96*SE) = {paired_rp:.6f}")
    if paired_rp > 0:
        ratio = abs(mean_d) / paired_rp
        verdict = (f"RESOLVED {ratio:.2f}x past the paired resolving power"
                   if ratio > 1.0 else
                   "NOT RESOLVED at this N -- no difference detected at this "
                   "resolution, which is not a proof of no difference at any resolution")
    else:
        verdict = ("ZERO PAIRED VARIANCE -- the two arms' per-document top-1 "
                   "indicators are identical on every document; the difference "
                   "is exactly zero at this N, resolution one document (1/239)")
    print(f"  verdict: {verdict}")

    if args.t1809_arm_c:
        filed = Path(args.t1809_arm_c)
        same = diff = missing = 0
        for lab in labels:
            a = Path(args.row_dir) / f"{lab}.float.bin"
            b = filed / f"{lab}.float.bin"
            if not a.exists() or not b.exists():
                missing += 1
            elif filecmp.cmp(a, b, shallow=False):
                same += 1
            else:
                diff += 1
        print(f"\nrun-conditions bridge: row arm vs T-1809's filed arm C dumps: "
              f"{same} byte-identical, {diff} differing, {missing} missing of "
              f"{len(labels)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

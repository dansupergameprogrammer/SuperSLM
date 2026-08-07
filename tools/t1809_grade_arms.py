#!/usr/bin/env python3
r"""T-1809 -- grade every arm through T-1777's own unmodified report functions.

Imports `t1777_retrieval_report`'s `load_pooled_vectors`, `compare_arms`,
`normal_ci_halfwidth`, and `discrete_step_recall` and calls them directly rather than
through the CLI, for the reason T-1805 recorded: the CLI fixes which directory plays
reference and which plays candidate in each of its two hardcoded comparisons, and the
`top1_within_top5` statistic is ASYMMETRIC in those roles. Calling `compare_arms` with the
roles named explicitly keeps T-1777's own reference (the true bf16 dumps) on the reference
side in every row, exactly as the engine baseline itself is scored.

Nothing in T-1777's worktree is written. Every figure printed here is computed by T-1777's
own functions on T-1777's own committed reference dumps.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--t1777-tools", required=True)
    ap.add_argument("--t1777-out", required=True,
                    help="T-1777's out/ directory (the committed reference dumps)")
    ap.add_argument("--arm", action="append", default=[], metavar="NAME=DIR",
                    help="an arm to grade, e.g. C=out/t1809_arm_c; repeatable")
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args(argv)

    sys.path.insert(0, str(Path(args.t1777_tools).resolve()))
    import t1777_retrieval_report as rr

    out = Path(args.t1777_out)
    manifest = out / "t1777_corpus" / "manifest.jsonl"
    labels, domains = [], {}
    with open(manifest, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rec = json.loads(line)
                labels.append(rec["label"])
                domains[rec["label"]] = rec.get("domain", rec.get("population", "ALL"))

    ref_vecs, ref_fps = rr.load_pooled_vectors(labels, out / "t1777_full_float_bf16", "float")

    arms = {
        "engine int8 (canonical baseline)": (out / "t1777_full_int8", "int8"),
        "reference self-consistency: true fp32": (out / "t1777_full_float_fp32", "float"),
    }
    for spec in args.arm:
        name, _, d = spec.partition("=")
        arms[f"T-1809 arm {name}"] = (Path(d), "float")

    rows = {}
    for name, (d, kind) in arms.items():
        cand_vecs, cand_fps = rr.load_pooled_vectors(labels, d, kind)
        results, n_docs = rr.compare_arms(ref_vecs, cand_vecs, domains, ref_fps, cand_fps)
        row = {"n": n_docs}
        for k in (1, 5, 10):
            vals = np.array([r.recall[k] for r in results])
            mean = float(vals.mean())
            step = rr.discrete_step_recall(n_docs, k)
            ci = rr.normal_ci_halfwidth(mean, n_docs)
            row[f"recall@{k}"] = mean
            row[f"rp@{k}"] = float(max(step, ci))
            row[f"num@{k}"] = float(vals.sum())
        top1 = np.array([r.ref_top1_displacement for r in results])
        w5 = float((top1 < 5).mean())
        row["within_top5"] = w5
        row["within_top5_n"] = int((top1 < 5).sum())
        row["within_top5_rp"] = float(max(1.0 / n_docs, rr.normal_ci_halfwidth(w5, n_docs)))
        rhos = np.array([r.spearman_rho for r in results])
        row["spearman_mean"] = float(rhos.mean())
        row["spearman_min"] = float(rhos.min())
        rows[name] = row

        print(f"\n=== {name}  (N={n_docs}) ===")
        for k in (1, 5, 10):
            print(f"  recall@{k:<2} = {row[f'recall@{k}']:.6f}   "
                  f"(sum {row[f'num@{k}']:.4f} / {n_docs})   "
                  f"resolving power {row[f'rp@{k}']:.4f}")
        print(f"  top1-within-top5 = {w5:.6f} ({row['within_top5_n']}/{n_docs})   "
              f"resolving power {row['within_top5_rp']:.4f}")
        print(f"  spearman mean = {row['spearman_mean']:.4f}  min = {row['spearman_min']:.4f}")

    def gap(a, b, k):
        """A - B with the two rows' resolving powers summed, per StandardsDocument 5.4."""
        d = rows[a][f"recall@{k}"] - rows[b][f"recall@{k}"]
        rp = rows[a][f"rp@{k}"] + rows[b][f"rp@{k}"]
        return d, rp, (abs(d) / rp if rp > 0 else float("nan"))

    print("\n=== pairwise gaps on recall@1 (the binding metric, D-SLM1455) ===")
    names = list(rows)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            d, rp, ratio = gap(a, b, 1)
            verdict = ("RESOLVED %.2fx past combined resolving power" % ratio) if ratio > 1.0 \
                else "NOT DISTINGUISHABLE at this N"
            print(f"  {a}\n    minus {b}\n    = {d:+.6f}  combined RP {rp:.4f}  -> {verdict}")

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(rows, f, indent=2)
        print(f"\nwritten: {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

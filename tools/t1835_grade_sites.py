#!/usr/bin/env python3
r"""T-1835 -- grade every per-site arm on the encoder oracle, through T-1777's own functions.

The oracle is T-1777's frozen 239-document corpus and its unmodified report functions;
`recall@1` is the binding metric (D-SLM1455) and `recall@5`/`recall@10`/top1-within-top5 are
diagnostics about the error's shape (D-SLM1458), never alternative bars. The reference on
every row is T-1777's own committed true-bf16 capture, produced before this ticket existed
and taking no input from anything this ticket computes.

`compare_arms` is called directly rather than through the CLI, for the reason T-1805
recorded and T-1809 repeated: the CLI fixes which directory plays reference in each of its
hardcoded comparisons and `top1_within_top5` is asymmetric in those roles.

TWO RESOLVING POWERS, and neither is dropped:
  * unpaired -- the two arms' own 95% normal-CI half-widths summed. T-1777's filed
    convention, which every cross-ticket comparison in this campaign uses.
  * paired   -- 1.96x the standard error of the per-document contrast. The arms share the
    corpus, the query order AND the batch, so a per-site contrast is genuinely paired and
    the unpaired sum overstates its noise. Reported against `base`, which is the arm every
    site figure is a contrast with.

Usage
    python tools\t1835_grade_sites.py --cell out\t1835_cellA --t1777-tools <dir> \
        --t1777-out D:\SuperSLM\.worktrees\t1777-retrieval-agreement\out \
        --json-out out\t1835_grades_cellA.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", required=True, help="a capture root, e.g. out/t1835_cellA")
    ap.add_argument("--t1777-tools", required=True)
    ap.add_argument("--t1777-out", required=True)
    ap.add_argument("--json-out", default=None)
    ap.add_argument("--baseline-arm", default="base")
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

    cell = Path(args.cell)
    arm_files = sorted(cell.glob("pooled/*.npz"))
    if not arm_files:
        raise SystemExit(f"no pooled vectors under {cell}/pooled")

    # The two standing reference points, read from T-1777's own committed dumps so this
    # ticket's table sits on the same axis as every other ticket's.
    rows = {}
    per_doc_recall1 = {}
    order = None

    cell_arms = []

    def grade(name, cand_vecs, cand_fps, is_cell_arm=False):
        """Grade one arm. The document-order check is enforced across the CELL's own arms
        only: a paired contrast is only defined when the two series index the same documents
        in the same order, and the two standing reference rows are graded over T-1777's own
        population rather than this cell's, so they carry no paired contrast and are excluded
        from the check rather than forced through it."""
        nonlocal order
        results, n_docs = rr.compare_arms(ref_vecs, cand_vecs, domains, ref_fps, cand_fps)
        these = [r.label for r in results]
        if is_cell_arm:
            cell_arms.append(name)
            if order is None:
                order = these
            elif these != order:
                raise SystemExit(f"{name}: document order differs from the cell's first arm")
        row = {"n": n_docs}
        for k in (1, 5, 10):
            vals = np.array([r.recall[k] for r in results])
            mean = float(vals.mean())
            row[f"recall@{k}"] = mean
            row[f"rp@{k}"] = float(max(rr.discrete_step_recall(n_docs, k),
                                       rr.normal_ci_halfwidth(mean, n_docs)))
            row[f"num@{k}"] = float(vals.sum())
        top1 = np.array([r.ref_top1_displacement for r in results])
        row["within_top5"] = float((top1 < 5).mean())
        row["within_top5_n"] = int((top1 < 5).sum())
        rhos = np.array([r.spearman_rho for r in results])
        row["spearman_mean"] = float(rhos.mean())
        row["spearman_min"] = float(rhos.min())
        rows[name] = row
        per_doc_recall1[name] = np.array([r.recall[1] for r in results])

    for nm, d, kind in (("engine int8 (canonical baseline)", out / "t1777_full_int8", "int8"),
                        ("reference self-consistency: true fp32",
                         out / "t1777_full_float_fp32", "float")):
        cv, cf = rr.load_pooled_vectors(labels, d, kind)
        grade(nm, cv, cf)

    for p in arm_files:
        name = p.stem
        z = np.load(p, allow_pickle=True)
        labs = [str(x) for x in z["labels"]]
        vecs = {l: z["vectors"][i] for i, l in enumerate(labs)}
        fps = {l: int(z["fingerprints"][i]) for i, l in enumerate(labs)}
        grade(name, vecs, fps, is_cell_arm=True)

    base = args.baseline_arm
    if base not in rows:
        raise SystemExit(f"baseline arm {base!r} not present in {cell}")

    n_docs = rows[base]["n"]

    def contrast_against(anchor):
        """Every cell arm's recall@1 contrast with one anchor arm, paired per document.

        Two anchors are meaningful and they answer different questions. Against `base` the
        contrast is what REMOVING a site buys in the presence of the other seventeen; against
        `null` it is what that site's quantization COSTS on its own. Both are computed rather
        than one being derived from the other, because deficit(base) - deficit(arm) and the
        paired contrast have different standard errors."""
        c = {}
        for name in cell_arms:
            if name == anchor:
                continue
            d = per_doc_recall1[name] - per_doc_recall1[anchor]
            mean = float(d.mean())
            se = float(d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else float("nan")
            paired_rp = 1.96 * se
            unpaired_rp = rows[name]["rp@1"] + rows[anchor]["rp@1"]
            # A zero (or NaN) paired RP means the paired contrast has no resolving power to
            # divide by -- whether because the arm is bit-identical to the anchor (mean and SE
            # both exactly zero, T-1863) or for any other reason the per-document SE collapses
            # to zero. `inf` in that slot made every such arm print RESOLVED regardless of
            # effect size (T-1862 found 52/52 duplicate-baseline arms doing exactly this). The
            # ratio is undefined, not infinite: report it as NaN and let the verdict below
            # refuse to call it resolved rather than silently passing a divide-by-zero through
            # as the strongest possible verdict.
            c[name] = {
                "delta_recall1": mean,
                "paired_rp": paired_rp,
                "unpaired_rp": unpaired_rp,
                "paired_ratio": abs(mean) / paired_rp if paired_rp > 0 else float("nan"),
                "unpaired_ratio": abs(mean) / unpaired_rp if unpaired_rp > 0 else float("nan"),
            }
        return c

    contrasts = contrast_against(base)
    contrasts_vs_null = contrast_against("null") if "null" in rows else {}

    print(f"=== {cell.name}: recall@1 per arm (N={n_docs}), reference = T-1777 true bf16 ===")
    for name in sorted(rows, key=lambda n: -rows[n]["recall@1"]):
        r = rows[name]
        print(f"  {name:38s} recall@1 = {r['recall@1']:.6f}  ({r['num@1']:.4f}/{r['n']})"
              f"   RP {r['rp@1']:.4f}")

    print(f"\n=== contrast against `{base}` on recall@1 ===")
    for name in sorted(contrasts, key=lambda n: -contrasts[n]["delta_recall1"]):
        c = contrasts[name]
        # A NaN paired_ratio (paired RP == 0) is refused explicitly rather than falling
        # through to "not resolved" -- `nan > 1.0` is already False in Python, so a silent
        # fallthrough would read identically to a real negative test and hide that no paired
        # comparison was actually possible here (T-1863).
        if math.isnan(c["paired_ratio"]):
            v = "UNDEFINED (paired RP = 0)"
        elif c["paired_ratio"] > 1.0:
            v = "RESOLVED"
        else:
            v = "not resolved"
        print(f"  {name:38s} {c['delta_recall1']:+.6f}   paired RP {c['paired_rp']:.4f} "
              f"({c['paired_ratio']:.2f}x, {v})   unpaired RP {c['unpaired_rp']:.4f} "
              f"({c['unpaired_ratio']:.2f}x)")

    if args.json_out:
        Path(args.json_out).write_text(
            json.dumps({"rows": rows, "contrasts": contrasts,
                        "contrasts_vs_null": contrasts_vs_null, "baseline": base,
                        "n_docs": n_docs}, indent=2), encoding="utf-8")
        print(f"\nwritten: {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

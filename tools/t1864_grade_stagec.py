#!/usr/bin/env python3
r"""T-1864 -- grade Stage C's attribution arms on the encoder oracle, POOLED and
PER-SUBPOPULATION (id/ood, T-1777's own manifest split), reporting BOTH the
paired and the unpaired ratio for every cell (Popper's T-1862 debunk of stage
B: `middle` resolved only pooled, and only on the paired convention; both
facts are reported here for every arm rather than left to be rediscovered).

Usage
    python tools\t1864_grade_stagec.py --cell out\t1864_stagec --t1777-tools <dir> \
        --t1777-out D:\SuperSLM\.worktrees\t1777-retrieval-agreement\out \
        --json-out out\t1864_grades.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np


def paired_unpaired(cand, anchor, n_cand, n_anchor, rp_cand, rp_anchor):
    d = cand - anchor
    mean = float(d.mean())
    se = float(d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else float("nan")
    paired_rp = 1.96 * se
    unpaired_rp = rp_cand + rp_anchor
    return {
        "n": len(d),
        "delta_recall1": mean,
        "paired_rp": paired_rp,
        "unpaired_rp": unpaired_rp,
        "paired_ratio": abs(mean) / paired_rp if paired_rp > 0 else float("nan"),
        "unpaired_ratio": abs(mean) / unpaired_rp if unpaired_rp > 0 else float("nan"),
    }


def verdict(ratio):
    if isinstance(ratio, float) and math.isnan(ratio):
        return "UNDEFINED (RP = 0)"
    return "RESOLVED" if ratio > 1.0 else "not resolved"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", required=True)
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

    per_doc = {}   # name -> {label: recall1}
    per_doc_domain = {}  # name -> {label: domain}
    rp1 = {}       # name -> unpaired rp@1 (pooled, T-1777 normal-CI convention)
    n_docs = None
    order = None

    for p in arm_files:
        name = p.stem
        z = np.load(p, allow_pickle=True)
        labs = [str(x) for x in z["labels"]]
        vecs = {l: z["vectors"][i] for i, l in enumerate(labs)}
        fps = {l: int(z["fingerprints"][i]) for i, l in enumerate(labs)}
        results, nd = rr.compare_arms(ref_vecs, vecs, domains, ref_fps, fps)
        these = [r.label for r in results]
        if order is None:
            order = these
            n_docs = nd
        elif these != order:
            raise SystemExit(f"{name}: document order differs from the cell's first arm")
        per_doc[name] = {r.label: r.recall[1] for r in results}
        per_doc_domain[name] = {r.label: r.domain for r in results}
        vals = np.array([r.recall[1] for r in results])
        rp1[name] = float(max(rr.discrete_step_recall(nd, 1),
                              rr.normal_ci_halfwidth(float(vals.mean()), nd)))

    base = args.baseline_arm
    if base not in per_doc:
        raise SystemExit(f"baseline arm {base!r} not present in {cell}")
    null_present = "null" in per_doc

    report = {"n_docs": n_docs, "baseline": base, "arms": {}}

    print(f"=== {cell.name}: Stage C attribution, N={n_docs}, pooled AND per-domain ===\n")

    for name in sorted(per_doc):
        if name in (base, "null") or name.startswith("basedup"):
            continue
        arm_report = {}
        for anchor in [a for a in (base, "null") if a in per_doc]:
            cand = np.array([per_doc[name][l] for l in order])
            anc = np.array([per_doc[anchor][l] for l in order])
            pooled = paired_unpaired(cand, anc, n_docs, n_docs, rp1[name], rp1[anchor])
            pooled["verdict_paired"] = verdict(pooled["paired_ratio"])
            pooled["verdict_unpaired"] = verdict(pooled["unpaired_ratio"])

            per_domain = {}
            for dom in ("id", "ood"):
                dom_labels = [l for l in order if per_doc_domain[name][l] == dom]
                cd = np.array([per_doc[name][l] for l in dom_labels])
                ad = np.array([per_doc[anchor][l] for l in dom_labels])
                nd_dom = len(dom_labels)
                rp_cand_dom = float(max(rr.discrete_step_recall(nd_dom, 1),
                                        rr.normal_ci_halfwidth(float(cd.mean()), nd_dom)))
                rp_anc_dom = float(max(rr.discrete_step_recall(nd_dom, 1),
                                       rr.normal_ci_halfwidth(float(ad.mean()), nd_dom)))
                dr = paired_unpaired(cd, ad, nd_dom, nd_dom, rp_cand_dom, rp_anc_dom)
                dr["verdict_paired"] = verdict(dr["paired_ratio"])
                dr["verdict_unpaired"] = verdict(dr["unpaired_ratio"])
                per_domain[dom] = dr

            arm_report[f"vs_{anchor}"] = {"pooled": pooled, "per_domain": per_domain}

            print(f"[{name} vs {anchor}]")
            print(f"  pooled   delta={pooled['delta_recall1']:+.6f}  "
                  f"paired {pooled['paired_ratio']:.2f}x ({pooled['verdict_paired']})  "
                  f"unpaired {pooled['unpaired_ratio']:.2f}x ({pooled['verdict_unpaired']})")
            for dom in ("id", "ood"):
                dr = per_domain[dom]
                print(f"  {dom:8s} delta={dr['delta_recall1']:+.6f}  n={dr['n']:3d}  "
                      f"paired {dr['paired_ratio']:.2f}x ({dr['verdict_paired']})  "
                      f"unpaired {dr['unpaired_ratio']:.2f}x ({dr['verdict_unpaired']})")
        print()
        report["arms"][name] = arm_report

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"written: {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

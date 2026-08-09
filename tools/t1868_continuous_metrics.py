#!/usr/bin/env python3
r"""T-1868 -- re-grade already-captured stage B / stage C pooled vectors on two
continuous metrics alongside `recall@1`, buying resolving power from the metric
rather than from more capture. Zero GPU: every input is a committed `.npz`
pooled-vector file already on disk; nothing here runs a forward pass.

Reuses `t1777_retrieval_report.py` UNMODIFIED for both the reference-vs-candidate
comparison (`compare_arms`) and its two building blocks (`cosine_sim_matrix`,
`ranks_desc`) -- the same module every `recall@1` figure in this campaign is
graded through. `compare_arms` already computes `DocResult.ref_top1_displacement`
(the candidate arm's rank of the reference's true top-1 neighbour) and discards
it into `recall@1`'s binary threshold; this ticket reads that existing field
instead of recomputing it, and additionally computes a score-margin quantity
from the same two building blocks `compare_arms` itself uses internally.

Two continuous metrics (full definitions: `Claude/Brunel/t1868-metric-regrade-
2026-08-09.md` S1.1):
  * rank_top1   -- 0-indexed candidate-arm rank of the reference's true top-1
                   neighbour. 0 == recall@1 hit. Lower is better.
  * margin_top1 -- candidate arm's own cosine-similarity gap between the true
                   top-1 neighbour and its best competitor. Sign reproduces
                   recall@1 exactly; magnitude is what recall@1 discards.
                   Higher is better.

Usage
    python tools\t1868_continuous_metrics.py \
        --cell out/t1861_full --label stageB --anchor-null null --anchor-base base \
        --t1777-tools D:\SuperSLM\.worktrees\t1777-retrieval-agreement\tools \
        --t1777-out   D:\SuperSLM\.worktrees\t1777-retrieval-agreement\out \
        --json-out out/t1868_stageb.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np


def load_cell_arms(cell: Path):
    """Read every pooled-vector .npz in a capture cell. Returns {name: (vecs, fps)}."""
    arms = {}
    for p in sorted(cell.glob("pooled/*.npz")):
        z = np.load(p, allow_pickle=True)
        labs = [str(x) for x in z["labels"]]
        vecs = {l: z["vectors"][i] for i, l in enumerate(labs)}
        fps = {l: int(z["fingerprints"][i]) for i, l in enumerate(labs)}
        arms[p.stem] = (vecs, fps)
    return arms


def margins_for_arm(rr, ref_vecs: dict, cand_vecs: dict, ref_top1_doc: dict, labels: list[str]):
    """For each query label, the candidate arm's own cosine-sim gap between the
    reference's true top-1 neighbour and that neighbour's best competitor in the
    CANDIDATE's own similarity row. Reuses rr.cosine_sim_matrix unmodified -- the
    identical building block compare_arms uses internally for its own ranking.
    `ref_top1_doc[label]` is the reference's true top-1 neighbour's label, supplied
    by the caller (computed once, from the reference arm, shared across every
    candidate arm graded against the same query set)."""
    cand_mat = np.stack([cand_vecs[l] for l in labels])
    cand_sims = rr.cosine_sim_matrix(cand_mat)
    idx = {l: i for i, l in enumerate(labels)}
    out = {}
    for qi, lab in enumerate(labels):
        j_star = idx[ref_top1_doc[lab]]
        row = cand_sims[qi].copy()
        self_sim = row[qi]
        row[qi] = -np.inf  # exclude self
        true_sim = row[j_star]
        row[j_star] = -np.inf  # exclude the true top-1 to find its best competitor
        competitor_sim = row.max()
        out[lab] = float(true_sim - competitor_sim)
        row[j_star] = true_sim  # restore (defensive; row is local)
    return out


def grade_arm(rr, ref_vecs, ref_fps, cand_vecs, cand_fps, domains, ref_top1_doc_global):
    """Runs compare_arms (unmodified) for recall@1 + rank_top1, and this ticket's
    own margin_top1 on the same label set. Returns per-document dict:
    {label: {"recall1":..., "rank_top1":..., "margin_top1":..., "domain":...}}"""
    results, n_docs = rr.compare_arms(ref_vecs, cand_vecs, domains, ref_fps, cand_fps)
    labels = [r.label for r in results]
    # ref_top1_doc: the reference's true top-1 neighbour's label, per query label.
    # Derived once from the reference arm alone (independent of any candidate),
    # supplied by the caller so every arm's margin is computed against the SAME
    # reference top-1 identity.
    margins = margins_for_arm(rr, ref_vecs, cand_vecs, ref_top1_doc_global, labels)
    per_doc = {}
    for r in results:
        per_doc[r.label] = {
            "recall1": float(r.recall[1]),
            "rank_top1": float(r.ref_top1_displacement),
            "margin_top1": margins[r.label],
            "domain": r.domain,
        }
    return per_doc, n_docs, labels


def compute_ref_top1_doc(rr, ref_vecs: dict, labels: list[str]):
    """The reference arm's own true top-1 neighbour label, per query label --
    computed once from the reference vectors alone via rr.ranks_desc (unmodified),
    so every candidate arm's margin_top1 is measured against the identical
    reference-side identity."""
    ref_mat = np.stack([ref_vecs[l] for l in labels])
    ref_sims = rr.cosine_sim_matrix(ref_mat)
    out = {}
    for qi, lab in enumerate(labels):
        _rank_of, ranked_idx = rr.ranks_desc(ref_sims[qi], qi)
        out[lab] = labels[ranked_idx[0]]
    return out


def paired_contrast(cand: np.ndarray, anchor: np.ndarray):
    d = cand - anchor
    mean = float(d.mean())
    se = float(d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else float("nan")
    paired_rp = 1.96 * se
    return {
        "delta": mean,
        "se": se,
        "paired_rp": paired_rp,
        "signed_ratio": (mean / paired_rp) if paired_rp > 0 else float("nan"),
        "abs_ratio": (abs(mean) / paired_rp) if paired_rp > 0 else float("nan"),
        "n": len(d),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", required=True)
    ap.add_argument("--label", required=True, help="stage label, e.g. stageB / stageC")
    ap.add_argument("--t1777-tools", required=True)
    ap.add_argument("--t1777-out", required=True)
    ap.add_argument("--baseline-arm", default="base")
    ap.add_argument("--null-arm", default="null")
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args(argv)

    sys.path.insert(0, str(Path(args.t1777_tools).resolve()))
    import t1777_retrieval_report as rr  # unmodified, reused

    out = Path(args.t1777_out)
    manifest = out / "t1777_corpus" / "manifest.jsonl"
    domains = {}
    with open(manifest, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rec = json.loads(line)
                domains[rec["label"]] = rec.get("domain", rec.get("population", "ALL"))

    ref_vecs, ref_fps = rr.load_pooled_vectors(list(domains.keys()), out / "t1777_full_float_bf16", "float")

    cell = Path(args.cell)
    arms = load_cell_arms(cell)
    if not arms:
        raise SystemExit(f"no pooled vectors under {cell}/pooled")

    base = args.baseline_arm
    null = args.null_arm
    if base not in arms:
        raise SystemExit(f"baseline arm {base!r} not present in {cell}")

    # Reference top-1 identity computed once, from labels common to every arm and
    # the reference (intersect over all cell arms so the identity set is stable).
    common_labels = None
    for name, (vecs, _fps) in arms.items():
        s = set(vecs) & set(ref_vecs)
        common_labels = s if common_labels is None else (common_labels & s)
    labels_sorted = sorted(common_labels)
    ref_top1_doc = compute_ref_top1_doc(rr, ref_vecs, labels_sorted)

    per_arm = {}
    order = None
    n_docs = None
    for name, (vecs, fps) in arms.items():
        per_doc, nd, labs = grade_arm(rr, ref_vecs, ref_fps, vecs, fps, domains, ref_top1_doc)
        these = sorted(labs)
        if order is None:
            order = these
            n_docs = nd
        elif these != order:
            raise SystemExit(f"{name}: document label set differs from the cell's first arm")
        per_arm[name] = per_doc

    metrics = ("recall1", "rank_top1", "margin_top1")

    # --- rows: per-arm mean of each metric, pooled -------------------------------
    rows = {}
    for name, per_doc in per_arm.items():
        row = {"n": n_docs}
        for m in metrics:
            vals = np.array([per_doc[l][m] for l in order])
            row[f"mean_{m}"] = float(vals.mean())
            row[f"sd_{m}"] = float(vals.std(ddof=1)) if n_docs > 1 else float("nan")
        rows[name] = row

    # --- contrasts vs base and vs null, pooled AND per-domain -------------------
    def contrasts_against(anchor_name):
        c = {}
        anchor_doc = per_arm[anchor_name]
        for name, per_doc in per_arm.items():
            if name == anchor_name:
                continue
            entry = {"pooled": {}}
            for m in metrics:
                cand_vals = np.array([per_doc[l][m] for l in order])
                anc_vals = np.array([anchor_doc[l][m] for l in order])
                entry["pooled"][m] = paired_contrast(cand_vals, anc_vals)
            for dom in ("id", "ood"):
                dom_labels = [l for l in order if per_doc[l]["domain"] == dom]
                if not dom_labels:
                    continue
                dom_entry = {}
                for m in metrics:
                    cand_vals = np.array([per_doc[l][m] for l in dom_labels])
                    anc_vals = np.array([anchor_doc[l][m] for l in dom_labels])
                    dom_entry[m] = paired_contrast(cand_vals, anc_vals)
                entry[dom] = dom_entry
            c[name] = entry
        return c

    contrasts_vs_base = contrasts_against(base) if base in per_arm else {}
    contrasts_vs_null = contrasts_against(null) if null in per_arm else {}

    report = {
        "label": args.label,
        "cell": str(cell),
        "n_docs": n_docs,
        "baseline": base,
        "null": null,
        "rows": rows,
        "contrasts_vs_base": contrasts_vs_base,
        "contrasts_vs_null": contrasts_vs_null,
    }

    print(f"=== T-1868 {args.label}: {cell.name} (N={n_docs}) ===")
    for name in sorted(rows, key=lambda n: rows[n]["mean_recall1"], reverse=True):
        r = rows[name]
        print(f"  {name:14s} recall1={r['mean_recall1']:.6f}  "
              f"rank_top1={r['mean_rank_top1']:8.4f}  margin_top1={r['mean_margin_top1']:+.6f}")

    print(f"\n--- contrasts vs `{base}`, pooled ---")
    for name in sorted(contrasts_vs_base):
        p = contrasts_vs_base[name]["pooled"]
        line = f"  {name:14s}"
        for m in metrics:
            c = p[m]
            line += f"  {m}: {c['delta']:+.6f} ({c['signed_ratio']:+.3f}x)"
        print(line)

    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_out).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nwritten: {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""T-1869 -- shared loader for the Popper probes against T-1868's result packet.

Read-only against every T-1868 input. Recomputes each arm's per-document
metric values independently from the committed pooled .npz vectors so no probe
below depends on T-1868's own JSON being right.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

T1777_TOOLS = Path(r"D:\SuperSLM\.worktrees\t1777-retrieval-agreement\tools")
T1777_OUT = Path(r"D:\SuperSLM\.worktrees\t1777-retrieval-agreement\out")
CELLS = {
    "stageB": Path(r"D:\SuperSLM\.worktrees\t1861-stageb\out\t1861_full"),
    "stageC": Path(r"D:\SuperSLM\.worktrees\t1864-stagec\out\t1864_stagec"),
}

sys.path.insert(0, str(T1777_TOOLS))
import t1777_retrieval_report as rr  # noqa: E402  (unmodified, reused)


def load_domains():
    domains = {}
    with open(T1777_OUT / "t1777_corpus" / "manifest.jsonl", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rec = json.loads(line)
                domains[rec["label"]] = rec.get("domain", rec.get("population", "ALL"))
    return domains


def load_cell(stage: str):
    """Returns labels (sorted), {arm: matrix (N,H)}, reference matrix, domains array."""
    domains = load_domains()
    ref_vecs, _ref_fps = rr.load_pooled_vectors(list(domains.keys()), T1777_OUT / "t1777_full_float_bf16", "float")
    cell = CELLS[stage]
    arms = {}
    common = None
    for p in sorted(cell.glob("pooled/*.npz")):
        z = np.load(p, allow_pickle=True)
        labs = [str(x) for x in z["labels"]]
        arms[p.stem] = {l: z["vectors"][i] for i, l in enumerate(labs)}
        s = set(labs) & set(ref_vecs)
        common = s if common is None else (common & s)
    labels = sorted(common)
    mats = {name: np.stack([v[l] for l in labels]) for name, v in arms.items()}
    ref_mat = np.stack([ref_vecs[l] for l in labels])
    dom = np.array([domains[l] for l in labels])
    return labels, mats, ref_mat, dom


def ref_top1_index(ref_mat: np.ndarray) -> np.ndarray:
    """Reference arm's own top-1 neighbour index per query (self excluded)."""
    sims = rr.cosine_sim_matrix(ref_mat)
    n = sims.shape[0]
    out = np.empty(n, dtype=np.int64)
    for i in range(n):
        _rank_of, ranked = rr.ranks_desc(sims[i], i)
        out[i] = ranked[0]
    return out


def metrics_for(mat: np.ndarray, jstar: np.ndarray):
    """recall1, rank_top1, margin_top1 per document, computed exactly as
    T-1868's instrument computes them (same building blocks, same exclusions)."""
    sims = rr.cosine_sim_matrix(mat)
    n = sims.shape[0]
    recall1 = np.empty(n)
    rank1 = np.empty(n)
    margin = np.empty(n)
    for i in range(n):
        rank_of, _ranked = rr.ranks_desc(sims[i], i)
        rank1[i] = rank_of[jstar[i]]
        recall1[i] = 1.0 if rank_of[jstar[i]] == 0 else 0.0
        row = sims[i].copy()
        row[i] = -np.inf
        true_sim = row[jstar[i]]
        row[jstar[i]] = -np.inf
        margin[i] = float(true_sim - row.max())
    return recall1, rank1, margin


def paired(cand: np.ndarray, anchor: np.ndarray):
    d = np.asarray(cand, float) - np.asarray(anchor, float)
    mean = float(d.mean())
    se = float(d.std(ddof=1) / np.sqrt(len(d)))
    rp = 1.96 * se
    return {"delta": mean, "se": se, "paired_rp": rp,
            "abs_ratio": (abs(mean) / rp) if rp > 0 else float("nan"), "n": len(d)}

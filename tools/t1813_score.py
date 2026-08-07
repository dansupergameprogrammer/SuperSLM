#!/usr/bin/env python3
"""T-1813 -- Route A driver, score phase.

Pools each arm's t1797_residual_arms.exe dumps (raw float32, layout
pos,state,hidden -- tools/t1797_residual_arms.cpp:29-31,817-828, read at
source) at the final state, over positions 1..n (excluding position 0, the
attention sink -- D-SLM447's ratified pooling scheme, the same reduction
tools/t1777_retrieval_report.py's own pooled_final_layer_float performs),
then calls that file's OWN compare_arms() function directly (imported, not
reimplemented -- T-1805's own established pattern,
Claude/Brunel/t1805-weights-only-isolation-arm-2026-08-07.md SS5) against
T-1777's existing, unmodified out/t1777_full_float_bf16 reference dumps.

Neither t1797_residual_arms.cpp nor t1777_retrieval_report.py is modified by
this script -- both are read-only inputs, imported/invoked from their
existing worktrees.

Usage:
    python tools/t1813_score.py --arms k0 k4 k8 rot
"""
from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

import numpy as np

T1777_WORKTREE = Path(r"D:\SuperSLM\.worktrees\t1777-retrieval-agreement")
T1777_CORPUS = T1777_WORKTREE / "out" / "t1777_corpus"
BF16_REF_DIR = T1777_WORKTREE / "out" / "t1777_full_float_bf16"

sys.path.insert(0, str(T1777_WORKTREE / "tools"))
import importlib
t1777_retrieval_report = importlib.import_module("t1777_retrieval_report")
compare_arms = t1777_retrieval_report.compare_arms
pooled_final_layer_float = t1777_retrieval_report.pooled_final_layer_float
fingerprint_of = t1777_retrieval_report.fingerprint_of
normal_ci_halfwidth = t1777_retrieval_report.normal_ci_halfwidth
discrete_step_recall = t1777_retrieval_report.discrete_step_recall

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
DUMP_ROOT = REPO_ROOT / "out" / "t1813_arms"


def read_t1797_meta(meta_path: Path) -> dict:
    d = {}
    for line in meta_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split(None, 1)
        if len(parts) == 2:
            d[parts[0]] = parts[1]
    return d


def pool_t1797_dump(bin_path: Path, meta_path: Path) -> np.ndarray:
    """Pools one t1797_residual_arms.exe dump at the final state, over
    positions 1..n (0-indexed positions 1..n-1), excluding position 0 -- the
    identical reduction pooled_final_layer_float performs against T-1777's
    own [header][pos,row,hidden] layout, reimplemented here against this
    tool's own headerless pos,state,hidden layout (t1797_residual_arms.cpp:
    817-828). The two source layouts differ only in whether a 32-byte header
    with an embedded fingerprint precedes the array."""
    meta = read_t1797_meta(meta_path)
    positions = int(meta["positions"])
    states = int(meta["states"])
    hidden = int(meta["hidden"])
    assert meta["dtype"] == "float32-le"
    assert meta["layout"] == "pos,state,hidden"
    data = np.fromfile(bin_path, dtype="<f4", count=positions * states * hidden)
    data = data.reshape(positions, states, hidden)
    final_state = states - 1  # last of n_states = num_hidden_layers + 1 rows (embed + 28 layers)
    if positions < 2:
        # single-token document: pooling over 1..n is empty by the D-SLM447
        # convention when n==1; fall back to including position 0 rather than
        # producing a NaN vector (documents this short are rare in this
        # corpus -- flagged, not silently divided by zero).
        return data[0:1, final_state, :].astype(np.float64).mean(axis=0)
    return data[1:, final_state, :].astype(np.float64).mean(axis=0)


def load_arm_vectors(arm: str, labels: list[str]) -> dict[str, np.ndarray]:
    vecs = {}
    arm_dir = DUMP_ROOT / arm
    for lab in labels:
        bin_path = arm_dir / f"{lab}_{arm}.bin"
        meta_path = arm_dir / f"{lab}_{arm}.meta"
        if not bin_path.exists() or not meta_path.exists():
            continue
        vecs[lab] = pool_t1797_dump(bin_path, meta_path)
    return vecs


def load_bf16_reference(labels: list[str]) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    vecs, fps = {}, {}
    for lab in labels:
        p = BF16_REF_DIR / f"{lab}.float.bin"
        if not p.exists():
            continue
        vecs[lab] = pooled_final_layer_float(p)
        fps[lab] = fingerprint_of(p, "float")
    return vecs, fps


def score_arm(arm: str, labels: list[str], domains: dict[str, str],
              bf16_vecs: dict, bf16_fps: dict, ks=(1, 5, 10)) -> dict:
    arm_vecs = load_arm_vectors(arm, labels)
    # bf16 is the REFERENCE (independent of the arm under test -- captured
    # before this ticket existed, unmodified by this session); the arm's
    # pooled vectors are the CANDIDATE. cand_fps intentionally omitted
    # (t1797_residual_arms.cpp writes no header/fingerprint at all) -- the
    # provenance guard is opt-in per label and fires only when BOTH dicts
    # carry the label (compare_arms, tools/t1777_retrieval_report.py:199),
    # so omitting the candidate side skips it cleanly rather than faking a
    # value, per T-1805's own established pattern (D-SLM1466-class note).
    results, n = compare_arms(bf16_vecs, arm_vecs, domains, bf16_fps, {}, ks=ks)
    out = {"arm": arm, "n_docs": n, "recall": {}, "spearman_mean": None}
    for k in ks:
        vals = np.array([r.recall[k] for r in results])
        mean = float(vals.mean())
        step = discrete_step_recall(n, k)
        ci = normal_ci_halfwidth(mean, n)
        rp = max(step, ci)
        out["recall"][k] = {"mean": mean, "n_correct_equiv": round(mean * n * k),
                             "resolving_power": rp, "discrete_step": step, "ci_halfwidth": ci}
    rhos = np.array([r.spearman_rho for r in results])
    out["spearman_mean"] = float(rhos.mean())
    top1_disp = np.array([r.ref_top1_displacement for r in results])
    out["top1_within_top5"] = float((top1_disp < 5).mean())
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=["k0", "k4", "k8", "rot"])
    args = ap.parse_args()

    manifest = []
    with open(T1777_CORPUS / "manifest.jsonl", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                manifest.append(json.loads(line))
    labels = [m["label"] for m in manifest]
    domains = {m["label"]: m["domain"] for m in manifest}

    bf16_vecs, bf16_fps = load_bf16_reference(labels)
    print(f"bf16 reference loaded: {len(bf16_vecs)}/{len(labels)}")

    all_results = {}
    for arm in args.arms:
        r = score_arm(arm, labels, domains, bf16_vecs, bf16_fps)
        all_results[arm] = r
        print(f"\n=== arm={arm}: n_docs={r['n_docs']} ===")
        for k, v in r["recall"].items():
            print(f"  recall@{k}: {v['mean']:.4f} ({v['n_correct_equiv']}/{r['n_docs']*k}) "
                  f"resolving_power={v['resolving_power']:.4f} "
                  f"(discrete_step={v['discrete_step']:.5f}, ci_halfwidth={v['ci_halfwidth']:.4f})")
        print(f"  spearman_mean={r['spearman_mean']:.4f}")
        print(f"  top1_within_top5={r['top1_within_top5']:.4f}")

    (REPO_ROOT / "out" / "t1813_score_results.json").write_text(
        json.dumps(all_results, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

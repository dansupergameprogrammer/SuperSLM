#!/usr/bin/env python3
"""T-1777 -- retrieval-agreement report: does the compiled engine's pooled
embedding rank and retrieve the same neighbours as the float model's?

Reads a corpus of per-document int8 dumps (tools/t1777_pooled_trace_batch.cpp
or tools/t1740_pooled_trace.cpp, identical binary layout) and float dumps at
one or two precisions (tools/t1777_pooled_float_dump_batch.py or
tools/t1740_pooled_float_dump.py), pools each side at the final layer
(D-SLM447's ratified scheme: mean over positions 1..n, excluding position 0),
then for every document AS A QUERY (leave-one-out against the rest of the
corpus) compares two "arms" -- a REFERENCE arm and a CANDIDATE arm -- and
reports:

  - recall@k (k in {1,5,10}): overlap between the reference arm's top-k
    nearest neighbours and the candidate arm's top-k nearest neighbours, as
    a fraction of k.
  - Spearman rank correlation between the two full rankings of the other
    N-1 documents.
  - rank displacement: for every (query, candidate-document) pair, the
    candidate arm's rank of that document minus its reference-arm rank --
    reported as a full distribution (percentiles), not only a mean.
  - discordant-pair near-tie sensitivity: for every pair of candidate
    documents whose RELATIVE order differs between the two arms, the
    reference arm's own similarity-score GAP between that pair. If
    disagreements cluster at a small reference-side gap, the two arms are
    disagreeing about a near-tie the reference itself barely resolves, not
    about a real ordering.

Two comparisons are run by this ticket's own build (T-1777):

  1. ENGINE vs FLOAT-BF16 (the question this ticket asks): does the
     compiled int8 engine's pooled embedding retrieve like the float
     model's actual forward-pass compute dtype (bfloat16, T-1782/
     D-SLM1212-1225 -- NOT float32; the "float32" the sibling T-1740
     scripts mention is HF's serialization dtype, never the compute dtype
     under torch_dtype="auto").
  2. FLOAT-BF16 vs FLOAT-FP32 (a REFERENCE SELF-CONSISTENCY baseline, no
     engine involved at all): how much do two different-precision runs of
     the SAME float model disagree with each other on ranking, purely from
     floating-point precision. This is the floor -- comparison 1 cannot be
     more precise than this baseline is, and any given comparison-1 result
     is only interpretable next to it.

Every population (in-distribution, out-of-distribution, and the two
combined) is reported separately for BOTH comparisons, each against its own
computed resolving power (StandardsDocument.md 5.4): the discrete step size
recall@k can move in given N queries and k, AND the practical statistical
detection floor (a normal-approximation 95% CI half-width on the observed
proportion).

Usage
-----
    python tools\\t1777_retrieval_report.py --int8-dir out\\t1777_full_int8 \\
        --float-bf16-dir out\\t1777_full_float_bf16 \\
        --float-fp32-dir out\\t1777_full_float_fp32 \\
        --manifest out\\t1777_corpus\\manifest.jsonl
"""
from __future__ import annotations

import argparse
import json
import math
import struct
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass


# --- dump readers (identical formats to tools/t1740_pooled_fidelity_report.py;
# reimplemented here rather than importing, matching that ticket's own
# precedent of reimplementing rather than modifying a reviewed instrument's
# import surface -- these are read-only parsers with no side effects). -------

def load_int8_dump(path: Path):
    with open(path, "rb") as f:
        num_positions, num_rows, hidden_size, fingerprint = struct.unpack("<QQQQ", f.read(32))
        scales = np.zeros((num_positions, num_rows, 2), dtype=np.int64)
        codes = np.zeros((num_positions, num_rows, hidden_size), dtype=np.int8)
        for pos in range(num_positions):
            for row in range(num_rows):
                m, e = struct.unpack("<qq", f.read(16))
                scales[pos, row, 0] = m
                scales[pos, row, 1] = e
                codes[pos, row, :] = np.frombuffer(f.read(hidden_size), dtype=np.int8)
    return num_positions, num_rows, hidden_size, fingerprint, scales, codes


def load_float_dump(path: Path):
    with open(path, "rb") as f:
        num_positions, num_rows, hidden_size, fingerprint = struct.unpack("<QQQQ", f.read(32))
        data = np.frombuffer(f.read(), dtype=np.float32, count=num_positions * num_rows * hidden_size)
    return num_positions, num_rows, hidden_size, fingerprint, data.reshape(num_positions, num_rows, hidden_size)


def pooled_final_layer_int8(path: Path) -> np.ndarray:
    n, r, h, _fp, scales, codes = load_int8_dump(path)
    row = r - 1  # final layer (layer num_hidden_layers - 1), row index num_hidden_layers
    codes_f = codes[1:, row, :].astype(np.float64)  # exclude position 0
    m = scales[1:, row, 0].astype(np.float64)[:, None]
    e = scales[1:, row, 1].astype(np.float64)
    real = codes_f * (m * np.exp2(e)[:, None])
    return real.mean(axis=0)


def pooled_final_layer_float(path: Path) -> np.ndarray:
    n, r, h, _fp, values = load_float_dump(path)
    row = r - 1
    return values[1:, row, :].astype(np.float64).mean(axis=0)


def fingerprint_of(path: Path, kind: str) -> int:
    with open(path, "rb") as f:
        _n, _r, _h, fp = struct.unpack("<QQQQ", f.read(32))
    return fp


def cosine_sim_matrix(vecs: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    unit = vecs / norms
    return unit @ unit.T


def ranks_desc(sims_row: np.ndarray, exclude: int):
    """Returns, for every index != exclude, its 0-based rank by descending
    similarity (0 = most similar). Ties broken by index (stable)."""
    idx = np.array([i for i in range(len(sims_row)) if i != exclude])
    order = np.argsort(-sims_row[idx], kind="stable")
    ranked_idx = idx[order]
    rank_of = np.empty(len(sims_row), dtype=np.int64)
    rank_of[:] = -1
    for r, i in enumerate(ranked_idx):
        rank_of[i] = r
    return rank_of, ranked_idx


def recall_at_k(ref_ranked_idx: np.ndarray, cand_ranked_idx: np.ndarray, k: int) -> float:
    kk = min(k, len(ref_ranked_idx))
    a = set(ref_ranked_idx[:kk].tolist())
    b = set(cand_ranked_idx[:kk].tolist())
    return len(a & b) / kk


def normal_ci_halfwidth(p: float, n: int, z: float = 1.96) -> float:
    if n <= 0:
        return float("nan")
    return z * math.sqrt(max(p * (1 - p), 1e-9) / n)


def discrete_step_recall(n: int, k: int) -> float:
    """The finest possible nonzero change in an aggregate recall@k averaged
    over n queries, each contributing a value on a 1/k grid."""
    return 1.0 / (n * k)


@dataclass
class DocResult:
    label: str
    domain: str
    recall: dict  # k -> value
    spearman_rho: float
    spearman_p: float
    displacement: np.ndarray  # cand_rank - ref_rank, over all other docs
    ref_top1_displacement: int  # candidate-arm rank of the reference top-1 neighbour
    discordant_ref_gaps: np.ndarray  # reference-side |sim gap| for every discordant pair this query contributes


def load_pooled_vectors(labels: list[str], dump_dir: Path, kind: str):
    """kind: 'int8' or 'float'. Returns {label: vector} and {label: fingerprint}."""
    vecs = {}
    fps = {}
    for lab in labels:
        ext = "int8" if kind == "int8" else "float"
        p = dump_dir / f"{lab}.{ext}.bin"
        if not p.exists():
            continue
        vecs[lab] = pooled_final_layer_int8(p) if kind == "int8" else pooled_final_layer_float(p)
        fps[lab] = fingerprint_of(p, kind)
    return vecs, fps


def compare_arms(ref_vecs: dict, cand_vecs: dict, domains: dict, ref_fps: dict, cand_fps: dict,
                  ks=(1, 5, 10)):
    """Generic reference-vs-candidate comparison over the labels present in
    BOTH arms. Provenance (fingerprint) is checked when both fingerprint
    dicts are non-empty for a label -- both sides must have embedded the
    exact same chat-templated text."""
    labels = sorted(set(ref_vecs) & set(cand_vecs))
    for lab in labels:
        if lab in ref_fps and lab in cand_fps and ref_fps[lab] != cand_fps[lab]:
            raise SystemExit(f"PROVENANCE MISMATCH {lab}: reference fingerprint 0x{ref_fps[lab]:016X} != "
                              f"candidate fingerprint 0x{cand_fps[lab]:016X}")

    n_docs = len(labels)
    if n_docs < 3:
        raise SystemExit(f"only {n_docs} documents present in both arms -- too few to measure")

    ref_mat = np.stack([ref_vecs[l] for l in labels])
    cand_mat = np.stack([cand_vecs[l] for l in labels])
    ref_sims = cosine_sim_matrix(ref_mat)
    cand_sims = cosine_sim_matrix(cand_mat)

    results: list[DocResult] = []
    for qi, lab in enumerate(labels):
        ref_rank_of, ref_ranked_idx = ranks_desc(ref_sims[qi], qi)
        cand_rank_of, cand_ranked_idx = ranks_desc(cand_sims[qi], qi)

        recall = {k: recall_at_k(ref_ranked_idx, cand_ranked_idx, k) for k in ks}

        others = np.array([i for i in range(n_docs) if i != qi])
        r_ranks = ref_rank_of[others]
        c_ranks = cand_rank_of[others]
        rho, pval = spearmanr(r_ranks, c_ranks)

        displacement = c_ranks - r_ranks
        ref_top1_doc = ref_ranked_idx[0]
        ref_top1_displacement = int(cand_rank_of[ref_top1_doc])

        # Discordant-pair near-tie sensitivity: among all C(len(others),2)
        # pairs of OTHER documents, find pairs whose relative order flips
        # between the reference and candidate arms, and record the
        # reference arm's own similarity gap for that pair.
        ref_row = ref_sims[qi, others]
        cand_row = cand_sims[qi, others]
        # sign of (ref_row[i] - ref_row[j]) vs sign of (cand_row[i] - cand_row[j])
        ref_diff = ref_row[:, None] - ref_row[None, :]
        cand_diff = cand_row[:, None] - cand_row[None, :]
        discordant = (np.sign(ref_diff) != np.sign(cand_diff)) & (ref_diff != 0)
        iu = np.triu_indices(len(others), k=1)
        disc_mask = discordant[iu]
        ref_gap_all = np.abs(ref_diff[iu])
        discordant_gaps = ref_gap_all[disc_mask]

        results.append(DocResult(lab, domains[lab], recall, float(rho), float(pval), displacement,
                                  ref_top1_displacement, discordant_gaps))

    return results, n_docs


def summarize(results: list[DocResult], population_name: str, ref_name: str, cand_name: str, ks=(1, 5, 10)):
    n = len(results)
    if n == 0:
        print(f"\n--- {population_name}: 0 queries -- no result ---")
        return
    print(f"\n--- population: {population_name} -- N={n} queries -- reference={ref_name} candidate={cand_name} ---")

    for k in ks:
        vals = np.array([r.recall[k] for r in results])
        mean = vals.mean()
        step = discrete_step_recall(n, k)
        ci = normal_ci_halfwidth(mean, n)
        floor = max(step, ci)
        verdict = "DETECTABLE below 1.0" if (1.0 - mean) > floor else "NOT DISTINGUISHABLE from 1.0 at this N"
        print(f"recall@{k}: mean={mean:.4f}  discrete_step(1/(N*k))={step:.5f}  "
              f"95%_CI_halfwidth={ci:.4f}  resolving_power={floor:.4f}  {verdict}")

    rhos = np.array([r.spearman_rho for r in results])
    print(f"spearman rank correlation (candidate vs reference ranking): mean={rhos.mean():.4f} "
          f"min={rhos.min():.4f} max={rhos.max():.4f} std={rhos.std():.4f}")

    all_disp = np.concatenate([r.displacement for r in results])
    abs_disp = np.abs(all_disp)
    pct = np.percentile(abs_disp, [50, 75, 90, 95, 99, 100])
    print(f"rank displacement |candidate_rank - reference_rank| over all (query, candidate-doc) pairs "
          f"(n_pairs={len(all_disp)}): median={pct[0]:.1f} p75={pct[1]:.1f} p90={pct[2]:.1f} "
          f"p95={pct[3]:.1f} p99={pct[4]:.1f} max={pct[5]:.1f}")

    top1_disp = np.array([r.ref_top1_displacement for r in results])
    print(f"reference top-1 neighbour's candidate-side rank: mean={top1_disp.mean():.2f}  "
          f"median={np.median(top1_disp):.1f}  "
          f"fraction still candidate-rank 0={float((top1_disp == 0).mean()):.4f}  "
          f"fraction within candidate top-5={float((top1_disp < 5).mean()):.4f}")

    all_disc_gaps = np.concatenate([r.discordant_ref_gaps for r in results]) if results else np.array([])
    n_disc = len(all_disc_gaps)
    n_pairs_total = sum(len(r.displacement) * (len(r.displacement) - 1) // 2 for r in results)
    if n_disc > 0:
        dp = np.percentile(all_disc_gaps, [10, 25, 50, 75, 90])
        print(f"discordant pairs (relative order disagrees between arms): {n_disc} of {n_pairs_total} "
              f"candidate-doc pairs ({100.0 * n_disc / max(1, n_pairs_total):.2f}%)")
        print(f"  reference-side |similarity gap| on discordant pairs: p10={dp[0]:.5f} p25={dp[1]:.5f} "
              f"median={dp[2]:.5f} p75={dp[3]:.5f} p90={dp[4]:.5f}")
    else:
        print("discordant pairs: none found (candidate arm reproduces the reference's full pairwise order)")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--int8-dir", required=True)
    parser.add_argument("--float-bf16-dir", required=True)
    parser.add_argument("--float-fp32-dir", default=None,
                         help="optional: if given, also runs the bf16-vs-fp32 reference "
                              "self-consistency baseline")
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args(argv)

    manifest = []
    with open(args.manifest, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                manifest.append(json.loads(line))
    labels = [m["label"] for m in manifest]
    domains = {m["label"]: m["domain"] for m in manifest}

    int8_vecs, int8_fps = load_pooled_vectors(labels, Path(args.int8_dir), "int8")
    bf16_vecs, bf16_fps = load_pooled_vectors(labels, Path(args.float_bf16_dir), "float")
    print(f"loaded: int8={len(int8_vecs)}/{len(labels)}  float-bf16={len(bf16_vecs)}/{len(labels)}")

    print("\n" + "=" * 78)
    print("COMPARISON 1: ENGINE (int8) vs FLOAT REFERENCE (bfloat16, this checkpoint's actual "
          "compute dtype -- T-1782/D-SLM1212-1225)")
    print("=" * 78)
    results1, n1 = compare_arms(bf16_vecs, int8_vecs, domains, bf16_fps, int8_fps)
    print(f"corpus size (documents present in both arms): {n1}")
    summarize(results1, "ALL (id + ood combined)", "float-bf16", "engine-int8")
    summarize([r for r in results1 if r.domain == "id"], "in-distribution (shopkeeper)", "float-bf16", "engine-int8")
    summarize([r for r in results1 if r.domain == "ood"], "out-of-distribution", "float-bf16", "engine-int8")

    if args.float_fp32_dir:
        fp32_vecs, fp32_fps = load_pooled_vectors(labels, Path(args.float_fp32_dir), "float")
        print(f"\nloaded: float-fp32={len(fp32_vecs)}/{len(labels)}")
        print("\n" + "=" * 78)
        print("COMPARISON 2: REFERENCE SELF-CONSISTENCY -- FLOAT bfloat16 vs FLOAT float32 "
              "(no engine involved; isolates precision-only disagreement)")
        print("=" * 78)
        results2, n2 = compare_arms(fp32_vecs, bf16_vecs, domains, fp32_fps, bf16_fps)
        print(f"corpus size (documents present in both arms): {n2}")
        summarize(results2, "ALL (id + ood combined)", "float-fp32", "float-bf16")
        summarize([r for r in results2 if r.domain == "id"], "in-distribution (shopkeeper)", "float-fp32", "float-bf16")
        summarize([r for r in results2 if r.domain == "ood"], "out-of-distribution", "float-fp32", "float-bf16")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

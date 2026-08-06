#!/usr/bin/env python3
"""T-1784: express this engine's attention fidelity (T-1778's own captured probability
dumps) in the metrics published prior-art work actually reports -- cosine similarity,
relative L1 error, and RMSE -- alongside total variation distance (TVD), so a comparison
against IntAttention's (arXiv:2511.21513) Table 9 figures is unit-matched rather than
directional.

Reads T-1778's own raw dumps unchanged (`out/t1778/p{1,2,3}.txt` -- engine Q15 integer
probabilities, and `out/t1778/p{1,2,3}_float.txt` -- the independent float32 reference,
both already self-checked and precision-verified per
`Claude/Brunel/t1778-float-attn-reference-2026-08-06.md`). Computes NOTHING from
`IExpScaleConstants`/`q_ln2`/`kIExpClipN` and reads no derived engine constant -- same
independence property T-1778's own comparison script established.

Definitions implemented (stated explicitly per this ticket's own requirement -- see the
build log for the citation and the sensitivity check against the alternative
aggregation):

  - TVD (per row): 0.5 * sum_k |p_e[k] - p_f[k]|                          -- unchanged
    from T-1778's own `t1778_compare_probs.py`/`t1778_pool_compare.py`.
  - Cosine similarity (per row): dot(p_e, p_f) / (||p_e||_2 * ||p_f||_2).
  - Relative L1 error (per row): sum_k |p_e[k]-p_f[k]| / sum_k |p_f[k]|. Since every
    captured row is a normalized probability distribution (p_f sums to 1 by
    construction -- softmax output; verified below), sum_k|p_f[k]| = 1 and this reduces
    algebraically to sum_k|p_e[k]-p_f[k]| = 2*TVD_row for THIS population. That
    reduction is checked by direct computation below, not assumed.
  - RMSE (per row): sqrt(mean_k (p_e[k]-p_f[k])^2).
  - Relative L2 error (per row, reported for completeness, not cited against
    IntAttention which does not report it): ||p_e-p_f||_2 / ||p_f||_2.

Two aggregation scopes are computed for every metric, because the paper's own text
(fetched from arXiv:2511.21513, Section 4.4 "P Matrix Quantization") does not state
which one it used: (a) per-row, then mean over rows ("row-mean"); (b) pooled by
weighted element/vector census over the whole population ("pooled"). For TVD/relL1/
RMSE the two scopes provably coincide when every row is a valid probability
distribution (proven and checked in the build log); for cosine similarity they do NOT
coincide in general, so both are reported and the delta is the stated sensitivity.
"""

from __future__ import annotations

import math
import sys
from collections import defaultdict

PAIRS = [
    ("p1", "../t1778-float-attn-reference/out/t1778/p1.txt", "../t1778-float-attn-reference/out/t1778/p1_float.txt"),
    ("p2", "../t1778-float-attn-reference/out/t1778/p2.txt", "../t1778-float-attn-reference/out/t1778/p2_float.txt"),
    ("p3", "../t1778-float-attn-reference/out/t1778/p3.txt", "../t1778-float-attn-reference/out/t1778/p3_float.txt"),
]


def read_engine_dump(path: str):
    rows = {}
    with open(path, "r", encoding="ascii") as f:
        header = f.readline().split()
        num_rows, width = int(header[0]), int(header[1])
        for _ in range(num_rows):
            parts = f.readline().split()
            layer, head, w = int(parts[0]), int(parts[1]), int(parts[2])
            probs_q15 = [int(x) for x in parts[3 : 3 + w]]
            rows[(layer, head)] = [p / 32768.0 for p in probs_q15]
    return rows, width


def read_float_dump(path: str):
    rows = {}
    with open(path, "r", encoding="ascii") as f:
        header = f.readline().split()
        num_rows, width = int(header[0]), int(header[1])
        for _ in range(num_rows):
            parts = f.readline().split()
            layer, head, w = int(parts[0]), int(parts[1]), int(parts[2])
            probs = [float(x) for x in parts[3 : 3 + w]]
            rows[(layer, head)] = probs
    return rows, width


def row_metrics(pe, pf):
    diffs = [a - b for a, b in zip(pe, pf)]
    abs_diffs = [abs(d) for d in diffs]
    sum_abs = sum(abs_diffs)
    tvd = 0.5 * sum_abs
    sq = [d * d for d in diffs]
    sum_sq = sum(sq)
    n = len(pe)
    rmse = math.sqrt(sum_sq / n)
    norm_e = math.sqrt(sum(p * p for p in pe))
    norm_f = math.sqrt(sum(p * p for p in pf))
    dot = sum(a * b for a, b in zip(pe, pf))
    cos = dot / (norm_e * norm_f) if norm_e > 0 and norm_f > 0 else float("nan")
    sum_pf = sum(pf)
    rel_l1 = sum_abs / sum_pf if sum_pf > 0 else float("nan")
    l2_diff = math.sqrt(sum_sq)
    rel_l2 = l2_diff / norm_f if norm_f > 0 else float("nan")
    return {
        "tvd": tvd,
        "cos": cos,
        "rel_l1": rel_l1,
        "rmse": rmse,
        "rel_l2": rel_l2,
        "sum_pf": sum_pf,
        "sum_pe": sum(pe),
        "n": n,
        "sum_abs": sum_abs,
        "sum_sq": sum_sq,
        "dot": dot,
        "norm_e": norm_e,
        "norm_f": norm_f,
    }


def main():
    per_layer_rows = defaultdict(list)  # layer -> list of row_metrics dicts
    per_layer_prompt_rows = defaultdict(lambda: defaultdict(list))  # layer -> prompt -> [row_metrics]
    row_sum_pf_all = []

    for prompt, engine_path, float_path in PAIRS:
        engine_rows, ew = read_engine_dump(engine_path)
        float_rows, fw = read_float_dump(float_path)
        assert ew == fw, (engine_path, ew, fw)
        assert set(engine_rows) == set(float_rows)
        for (layer, head), pe in engine_rows.items():
            pf = float_rows[(layer, head)]
            assert len(pe) == len(pf)
            m = row_metrics(pe, pf)
            per_layer_rows[layer].append(m)
            per_layer_prompt_rows[layer][prompt].append(m)
            row_sum_pf_all.append(m["sum_pf"])

    # Self-check: every reference row sums to 1 (verifies the relL1==2*TVD reduction's
    # own premise, not assumed).
    min_sum, max_sum = min(row_sum_pf_all), max(row_sum_pf_all)
    print(f"Self-check: float reference row sums range [{min_sum:.6f}, {max_sum:.6f}] "
          f"over {len(row_sum_pf_all)} rows (expect ~1.0 -- softmax output)")
    max_rel_l1_minus_2tvd = 0.0
    for layer, rows in per_layer_rows.items():
        for m in rows:
            d = abs(m["rel_l1"] - 2 * m["tvd"])
            max_rel_l1_minus_2tvd = max(max_rel_l1_minus_2tvd, d)
    print(f"Self-check: max_row |relL1 - 2*TVD| = {max_rel_l1_minus_2tvd:.10f} "
          f"(algebraic identity for two rows both summing to 1; near-zero confirms it "
          f"holds on this population, small residual is float rounding + engine Q15 "
          f"quantization of its own row sum away from exactly 1.0)\n")

    print("=== Per-layer, row-mean aggregation (mean of each row's own metric; TVD/relL1/RMSE match T-1778 §7's own 'mean over 12 heads' convention pooled over 3 prompts too) ===")
    hdr = f"{'layer':>5} {'n_rows':>6} {'mean TVD':>9} {'mean cos':>9} {'mean relL1':>11} {'mean RMSE':>10} {'mean relL2':>10}"
    print(hdr)
    layer_summary = {}
    for layer in sorted(per_layer_rows.keys()):
        rows = per_layer_rows[layer]
        n = len(rows)
        mean_tvd = sum(r["tvd"] for r in rows) / n
        mean_cos = sum(r["cos"] for r in rows) / n
        mean_rel_l1 = sum(r["rel_l1"] for r in rows) / n
        mean_rmse = sum(r["rmse"] for r in rows) / n
        mean_rel_l2 = sum(r["rel_l2"] for r in rows) / n
        layer_summary[layer] = (mean_tvd, mean_cos, mean_rel_l1, mean_rmse, mean_rel_l2)
        print(f"{layer:>5} {n:>6} {mean_tvd:>9.6f} {mean_cos:>9.6f} {mean_rel_l1:>11.6f} {mean_rmse:>10.6f} {mean_rel_l2:>10.6f}")

    print("\n=== Per-layer, pooled-census aggregation (all elements from all rows at that layer treated as one population; cosine over the flattened concatenation of all rows at that layer) ===")
    print(f"{'layer':>5} {'n_elems':>8} {'pooled TVD(elemsum/2)':>22} {'pooled cos(flat)':>17} {'pooled relL1':>13} {'pooled RMSE':>12}")
    for layer in sorted(per_layer_rows.keys()):
        rows = per_layer_rows[layer]
        total_sum_abs = sum(r["sum_abs"] for r in rows)
        total_sum_sq = sum(r["sum_sq"] for r in rows)
        total_n = sum(r["n"] for r in rows)
        total_sum_pf = sum(r["sum_pf"] for r in rows)
        total_dot = sum(r["dot"] for r in rows)
        total_norm_e = math.sqrt(sum(r["norm_e"] ** 2 for r in rows))
        total_norm_f = math.sqrt(sum(r["norm_f"] ** 2 for r in rows))
        pooled_tvd_equiv = 0.5 * total_sum_abs / len(rows)  # matches T-1778's "mean row TVD" (per-row TVD averaged, i.e. weighted by row not by element)
        pooled_cos_flat = total_dot / (total_norm_e * total_norm_f)
        pooled_rel_l1 = total_sum_abs / total_sum_pf
        pooled_rmse = math.sqrt(total_sum_sq / total_n)
        print(f"{layer:>5} {total_n:>8} {pooled_tvd_equiv:>22.6f} {pooled_cos_flat:>17.6f} {pooled_rel_l1:>13.6f} {pooled_rmse:>12.8f}")

    print("\n=== Per-layer, per-prompt row-mean cosine (resolving-power context, mirrors T-1778 §7/§8's per-prompt table) ===")
    print(f"{'layer':>5} {'p1 cos':>9} {'p2 cos':>9} {'p3 cos':>9} {'spread(max-min)':>16}")
    cos_gaps_vs_spread = []
    for layer in sorted(per_layer_prompt_rows.keys()):
        vals = []
        for prompt in ("p1", "p2", "p3"):
            rows = per_layer_prompt_rows[layer][prompt]
            mean_cos = sum(r["cos"] for r in rows) / len(rows)
            vals.append(mean_cos)
        spread = max(vals) - min(vals)
        cos_gaps_vs_spread.append((layer, spread))
        print(f"{layer:>5} {vals[0]:>9.6f} {vals[1]:>9.6f} {vals[2]:>9.6f} {spread:>16.6f}")

    print("\n=== Per-layer, per-prompt row-mean RMSE (resolving-power context) ===")
    print(f"{'layer':>5} {'p1 RMSE':>9} {'p2 RMSE':>9} {'p3 RMSE':>9} {'spread(max-min)':>16}")
    for layer in sorted(per_layer_prompt_rows.keys()):
        vals = []
        for prompt in ("p1", "p2", "p3"):
            rows = per_layer_prompt_rows[layer][prompt]
            mean_rmse = sum(r["rmse"] for r in rows) / len(rows)
            vals.append(mean_rmse)
        print(f"{layer:>5} {vals[0]:>9.6f} {vals[1]:>9.6f} {vals[2]:>9.6f} {max(vals)-min(vals):>16.6f}")

    print("\nCosine layer-vs-layer gaps compared to each layer's own inter-prompt spread:")
    layer0_cos = layer_summary[0][1]
    for layer, spread in cos_gaps_vs_spread:
        if layer == 0:
            continue
        gap = abs(layer_summary[layer][1] - layer0_cos)
        print(f"  layer 0 vs layer {layer}: |cos gap|={gap:.6f}  layer {layer}'s own inter-prompt spread={spread:.6f}  "
              f"gap {'EXCEEDS' if gap > spread else 'within'} that layer's own spread")

    print("\n=== Grand pooled (all 6 layers, all 3 prompts, all 12 heads together) ===")
    all_rows = [r for rows in per_layer_rows.values() for r in rows]
    n_rows = len(all_rows)
    total_sum_abs = sum(r["sum_abs"] for r in all_rows)
    total_sum_sq = sum(r["sum_sq"] for r in all_rows)
    total_n = sum(r["n"] for r in all_rows)
    total_sum_pf = sum(r["sum_pf"] for r in all_rows)
    total_dot = sum(r["dot"] for r in all_rows)
    total_norm_e = math.sqrt(sum(r["norm_e"] ** 2 for r in all_rows))
    total_norm_f = math.sqrt(sum(r["norm_f"] ** 2 for r in all_rows))
    grand_row_mean_cos = sum(r["cos"] for r in all_rows) / n_rows
    grand_pooled_cos = total_dot / (total_norm_e * total_norm_f)
    grand_row_mean_tvd = sum(r["tvd"] for r in all_rows) / n_rows
    grand_row_mean_rel_l1 = sum(r["rel_l1"] for r in all_rows) / n_rows
    grand_pooled_rel_l1 = total_sum_abs / total_sum_pf
    grand_row_mean_rmse = sum(r["rmse"] for r in all_rows) / n_rows
    grand_pooled_rmse = math.sqrt(total_sum_sq / total_n)
    print(f"n_rows={n_rows}  n_elems={total_n}")
    print(f"row-mean:  TVD={grand_row_mean_tvd:.6f}  cos={grand_row_mean_cos:.6f}  relL1={grand_row_mean_rel_l1:.6f}  RMSE={grand_row_mean_rmse:.6f}")
    print(f"pooled:    cos(flat)={grand_pooled_cos:.6f}  relL1={grand_pooled_rel_l1:.6f}  RMSE(elemwise)={grand_pooled_rmse:.8f}")
    print(f"cosine sensitivity (row-mean vs pooled-flat): {grand_row_mean_cos - grand_pooled_cos:+.6f}")


if __name__ == "__main__":
    main()

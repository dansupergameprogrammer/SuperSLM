#!/usr/bin/env python3
"""T-2604: three-way localization inside Qwen3 decoder block 0.

This probe uses the converter's persisted ``QuantizedModel`` cache as the exact
input to ``tools/reference_pipeline`` arithmetic.  It walks only block 0 and
captures every site named by the ticket.  The float arm is the pinned Hugging
Face checkpoint executed at float32; Q/K tensors are permuted into the engine's
adjacent-pair RoPE layout before comparison.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import subprocess
import time
from pathlib import Path

import numpy as np

from reference_pipeline import intmath, pipeline, rope, silu_lut
from reference_pipeline.artifact_cache import load_artifact


QUERY_PREFIX = (
    "Instruct: Given a web search query, retrieve relevant passages that answer the query\n"
    "Query:"
)
PROB_FRAC_BITS = pipeline.PROB_FRAC_BITS
NORM_FRAC_BITS = pipeline.NORM_FRAC_BITS


def scale_value(scale: tuple[int, int]) -> float:
    return math.ldexp(float(scale[0]), int(scale[1]))


def dequant(codes, scale: tuple[int, int]) -> np.ndarray:
    return np.asarray(codes, dtype=np.float64) * scale_value(scale)


def clamp_int8(values) -> tuple[np.ndarray, int]:
    values = np.asarray(values, dtype=np.int64)
    count = int(np.count_nonzero((values < -127) | (values > 127)))
    return np.clip(values, -127, 127).astype(np.int8), count


def chain(model, site: str, wide: np.ndarray, incoming: list[tuple[int, int]]):
    """The reference pipeline's own C19-C26 chain, one row at a time."""
    rows = np.asarray(wide, dtype=np.int64)
    out = np.empty(rows.shape, dtype=np.int8)
    scales: list[tuple[int, int]] = []
    records: list[dict] = []
    pipeline._CURRENT_COMPOSITION_CONSTANTS[0] = model.composition_constants
    for token_index, row in enumerate(rows):
        row_incoming = incoming[token_index] if incoming and isinstance(incoming[0], list) else incoming
        codes, scale = pipeline._chain_record(site, token_index, row.tolist(), row_incoming, records)
        out[token_index], _ = clamp_int8(codes)
        scales.append(tuple(int(v) for v in scale))
    return out, scales, records


def fold_rows(rows: np.ndarray, folds) -> np.ndarray:
    return np.asarray(
        [pipeline.fold_projection_accumulator(row.tolist(), folds) for row in rows],
        dtype=np.int64,
    )


def project(rows: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Bound here is <= 3072 * 127**2 < 2**26, so numpy int64 matmul is exact.
    return np.asarray(rows, dtype=np.int64) @ np.asarray(weight, dtype=np.int64).T


def rmsnorm_integer(model, site: str, rows: np.ndarray, gains: np.ndarray):
    wide = np.empty(rows.shape, dtype=np.int64)
    width = rows.shape[-1]
    gains = np.asarray(gains, dtype=np.int64)
    for t, row in enumerate(np.asarray(rows, dtype=np.int64)):
        total = int(np.dot(row, row))
        root = max(intmath.i_sqrt((total << (2 * NORM_FRAC_BITS)) // width), 1)
        wide[t] = np.asarray(
            [((int(row[i]) << (2 * NORM_FRAC_BITS)) // root) * int(gains[i]) for i in range(width)],
            dtype=np.int64,
        )
    return chain(model, site, wide, [])


def rotate_integer(rows: np.ndarray, cos_table, sin_table):
    rows = np.asarray(rows, dtype=np.int8)
    out = np.empty_like(rows)
    raw = np.empty(rows.shape, dtype=np.int64)
    count = 0
    for t in range(rows.shape[0]):
        for h in range(rows.shape[1]):
            for pair in range(rows.shape[2] // 2):
                x, y = rope.rope_apply_pair(
                    int(rows[t, h, 2 * pair]), int(rows[t, h, 2 * pair + 1]),
                    int(cos_table[t][pair]), int(sin_table[t][pair]),
                )
                raw[t, h, 2 * pair:2 * pair + 2] = (x, y)
                if x < -127 or x > 127:
                    count += 1
                if y < -127 or y > 127:
                    count += 1
                out[t, h, 2 * pair:2 * pair + 2] = (
                    max(-127, min(127, x)), max(-127, min(127, y)))
    return out, raw, count


def integer_block0(model, token_ids: list[int]) -> dict[str, np.ndarray | list | dict]:
    cfg = model.config
    steps = len(token_ids)
    prefix = "layer0"
    captures: dict[str, object] = {}
    clamp_counts: dict[str, int] = {}
    site_records: list[dict] = []

    embed_wide = np.asarray(model.weights["embed"])[token_ids].astype(np.int64)
    hidden, hidden_scales, embed_trace = chain(model, "embed", embed_wide, [])
    captures["embedding"] = np.stack([dequant(hidden[t], hidden_scales[t]) for t in range(steps)])
    captures["_trace_embedding"] = embed_trace

    attn_gain = np.asarray(model.weights[f"{prefix}.attn_norm.gain"])
    normed, norm_scales, trace = rmsnorm_integer(model, f"{prefix}.attn_norm", hidden, attn_gain)
    captures["input_rmsnorm"] = np.stack([dequant(normed[t], norm_scales[t]) for t in range(steps)])
    captures["_trace_input_rmsnorm"] = trace

    projection_codes: dict[str, np.ndarray] = {}
    projection_scales: dict[str, list[tuple[int, int]]] = {}
    raw_projection: dict[str, np.ndarray] = {}
    for leaf in ("q_proj", "k_proj", "v_proj"):
        raw = project(normed, np.asarray(model.weights[f"{prefix}.{leaf}"]))
        folds, _ = pipeline._reference_fold(model.weight_scales[f"{prefix}.{leaf}"])
        folded = fold_rows(raw, folds)
        raw_projection[leaf] = folded
        if leaf == "q_proj":
            codes, scales, qtrace = chain(
                model, f"{prefix}.{leaf}.requant", folded,
                [[norm_scales[t]] for t in range(steps)],
            )
            projection_codes[leaf] = codes
            projection_scales[leaf] = scales
            captures[leaf] = np.stack([dequant(codes[t], scales[t]) for t in range(steps)])
            captures[f"_trace_{leaf}"] = qtrace

    head_dim = cfg.head_dim
    num_kv = cfg.num_key_value_heads
    k_codes = np.empty((steps, num_kv, head_dim), dtype=np.int8)
    v_codes = np.empty_like(k_codes)
    k_scale_raw: list[list[tuple[int, int]]] = [[(0, 0)] * steps for _ in range(num_kv)]
    v_scale_raw: list[list[tuple[int, int]]] = [[(0, 0)] * steps for _ in range(num_kv)]
    kv_trace: list[dict] = []
    landing_saturation = 0
    for t in range(steps):
        m_a, e_a = norm_scales[t]
        for h in range(num_kv):
            for leaf, target in (("k_proj", k_codes), ("v_proj", v_codes)):
                seg = raw_projection[leaf][t, h * head_dim:(h + 1) * head_dim]
                m_out, e_out = model.kv_landing_scales[f"{prefix}.{leaf[0]}_head{h}"]
                _m_t, e_t, r_t = model.kv_landing_reciprocals[f"{prefix}.{leaf[0]}_head{h}"]
                landed_raw = np.asarray([
                    intmath.residual_reconcile(int(x), m_a, r_t, e_a, e_t) for x in seg
                ], dtype=np.int64)
                landed, sat = clamp_int8(landed_raw)
                landing_saturation += sat
                target[t, h] = landed
                (k_scale_raw if leaf == "k_proj" else v_scale_raw)[h][t] = (m_out, e_out)
                kv_trace.append({
                    "site": f"{prefix}.{leaf}.requant", "token_index": t, "head": h,
                    "x_int": tuple(int(v) for v in seg), "m_in": m_a, "e_in": e_a,
                    "codes": tuple(int(v) for v in landed), "m_out": m_out, "e_out": e_out,
                })
    clamp_counts["kv_landing"] = landing_saturation
    captures["k_proj"] = np.stack([
        np.stack([dequant(k_codes[t, h], k_scale_raw[h][t]) for h in range(num_kv)])
        for t in range(steps)])
    captures["v_proj"] = np.stack([
        np.stack([dequant(v_codes[t, h], v_scale_raw[h][t]) for h in range(num_kv)])
        for t in range(steps)])
    captures["_trace_kv"] = kv_trace

    q_codes = projection_codes["q_proj"].reshape(steps, cfg.num_attention_heads, head_dim).copy()
    q_scales_by_head: list[list[tuple[int, int]]] = [list(projection_scales["q_proj"])
                                                       for _ in range(cfg.num_attention_heads)]
    q_norm_gain = np.asarray(model.weights[f"{prefix}.q_norm.gain"])
    q_norm_values = np.empty(q_codes.shape, dtype=np.float64)
    q_norm_trace: list[dict] = []
    for h in range(cfg.num_attention_heads):
        codes, scales, records = rmsnorm_integer(model, f"{prefix}.q_norm", q_codes[:, h], q_norm_gain)
        for record in records:
            record["head"] = h
        q_codes[:, h] = codes
        q_scales_by_head[h] = scales
        q_norm_values[:, h] = np.stack([dequant(codes[t], scales[t]) for t in range(steps)])
        q_norm_trace.extend(records)
    captures["q_norm"] = q_norm_values
    captures["_trace_q_norm"] = q_norm_trace

    k_norm_gain = np.asarray(model.weights[f"{prefix}.k_norm.gain"])
    k_norm_values = np.empty(k_codes.shape, dtype=np.float64)
    k_consumed_values = np.empty(k_codes.shape, dtype=np.float64)
    k_norm_trace: list[dict] = []
    k_norm_landing_saturation = 0
    k_consumed_scales: list[tuple[int, int]] = []
    for h in range(num_kv):
        codes, scales, records = rmsnorm_integer(model, f"{prefix}.k_norm", k_codes[:, h], k_norm_gain)
        for record in records:
            record["head"] = h
        k_norm_trace.extend(records)
        m_t, e_t, r_t = model.kv_landing_reciprocals[f"{prefix}.k_normed_head{h}"]
        k_consumed_scales.append((m_t, e_t))
        for t in range(steps):
            k_norm_values[t, h] = dequant(codes[t], scales[t])
            m_a, e_a = scales[t]
            landed_raw = np.asarray([
                intmath.residual_reconcile(int(x), m_a, r_t, e_a, e_t) for x in codes[t]
            ], dtype=np.int64)
            landed, sat = clamp_int8(landed_raw)
            k_norm_landing_saturation += sat
            k_codes[t, h] = landed
            k_consumed_values[t, h] = dequant(landed, (m_t, e_t))
            site_records.append({
                "site": f"{prefix}.k_norm_landed", "token": t, "head": h,
                "x_int": [], "codes": landed.tolist(),
            })
    captures["k_norm"] = k_norm_values
    captures["k_norm_landed"] = k_consumed_values
    captures["_trace_k_norm"] = k_norm_trace
    clamp_counts["k_norm_landing"] = k_norm_landing_saturation

    cos_table = np.asarray(model.rope_tables[0]).tolist()
    sin_table = np.asarray(model.rope_tables[1]).tolist()
    q_rot, q_rot_raw, q_rope_sat = rotate_integer(q_codes, cos_table, sin_table)
    k_rot, k_rot_raw, k_rope_sat = rotate_integer(k_codes, cos_table, sin_table)
    captures["rope_q"] = np.stack([
        np.stack([dequant(q_rot[t, h], q_scales_by_head[h][t])
                  for h in range(cfg.num_attention_heads)]) for t in range(steps)])
    captures["rope_k"] = np.stack([
        np.stack([dequant(k_rot[t, h], k_consumed_scales[h]) for h in range(num_kv)])
        for t in range(steps)])
    captures["_raw_rope_q"] = q_rot_raw
    captures["_raw_rope_k"] = k_rot_raw
    clamp_counts["rope_q"] = q_rope_sat
    clamp_counts["rope_k"] = k_rope_sat
    for t in range(steps):
        for h in range(cfg.num_attention_heads):
            site_records.append({
                "site": f"{prefix}.rope_q", "token": t, "head": h,
                "x_int": [], "codes": q_rot[t, h].tolist(),
            })
        for h in range(num_kv):
            site_records.append({
                "site": f"{prefix}.rope_k", "token": t, "head": h,
                "x_int": [], "codes": k_rot[t, h].tolist(),
            })

    group = cfg.num_attention_heads // num_kv
    score_values: list[np.ndarray] = []
    probability_values: list[np.ndarray] = []
    weighted = np.empty((steps, cfg.num_attention_heads, head_dim), dtype=np.float64)
    context_wide = np.empty((steps, cfg.num_attention_heads * head_dim), dtype=np.int64)
    for t in range(steps):
        width = t + 1
        scores_t = np.empty((cfg.num_attention_heads, width), dtype=np.float64)
        probs_t = np.empty_like(scores_t)
        for h in range(cfg.num_attention_heads):
            kv_h = h // group
            scores = np.asarray(q_rot[t, h], dtype=np.int64) @ np.asarray(
                k_rot[:width, kv_h], dtype=np.int64).T
            site_records.append({
                "site": f"{prefix}.attention_scores", "token": t, "head": h,
                "x_int": scores.tolist(), "codes": [],
            })
            softmax_static = model.composition_constants[f"{prefix}.softmax_khead{kv_h}"]
            sm_scale = intmath.carried_scale_product([q_scales_by_head[h][t], softmax_static])
            scores_t[h] = dequant(scores, sm_scale)
            q_ln2, q_b, q_c = intmath.iexp_scale_constants(
                sm_scale[0], sm_scale[1], pipeline._IEXP_LN2_Q, pipeline._IEXP_QFMT,
                pipeline._IEXP_B_Q, pipeline._IEXP_QFMT, pipeline._IEXP_CA_Q,
                pipeline._IEXP_QFMT)
            shifted = intmath.shift_by_max(scores.tolist())
            exponentials = [intmath.i_exp_from_constants(int(q), q_ln2, q_b, q_c)
                            for q in shifted]
            total = sum(exponentials)
            probs = np.asarray([(e << PROB_FRAC_BITS) // max(total, 1)
                                for e in exponentials], dtype=np.int64)
            site_records.append({
                "site": f"{prefix}.softmax", "token": t, "head": h,
                "x_int": probs.tolist(), "codes": [],
            })
            probs_t[h] = probs.astype(np.float64) / (1 << PROB_FRAC_BITS)
            ctx_acc = probs @ np.asarray(v_codes[:width, kv_h], dtype=np.int64)
            site_records.append({
                "site": f"{prefix}.weighted_sum", "token": t, "head": h,
                "x_int": ctx_acc.tolist(), "codes": [],
            })
            weighted[t, h] = (
                ctx_acc.astype(np.float64) * scale_value(v_scale_raw[kv_h][t]) /
                (1 << PROB_FRAC_BITS))
            fold = None
            v_scale = model.scales.scale(f"{prefix}.v_head{kv_h}.scale")
            vmax = max(model.scales.scale(f"{prefix}.v_head{i}.scale") for i in range(num_kv))
            if v_scale != vmax:
                fold = pipeline.quantize_multiplier(v_scale / vmax)
            base = h * head_dim
            context_wide[t, base:base + head_dim] = np.asarray(
                pipeline.fold_projection_accumulator(ctx_acc.tolist(), [fold] * head_dim),
                dtype=np.int64)
        score_values.append(scores_t)
        probability_values.append(probs_t)
    captures["attention_scores"] = score_values
    captures["softmax"] = probability_values
    captures["weighted_sum"] = weighted

    attn_codes, attn_scales, trace = chain(model, f"{prefix}.attn_ctx", context_wide, [])
    captures["attention_context"] = np.stack([
        dequant(attn_codes[t], attn_scales[t]) for t in range(steps)]).reshape(
            steps, cfg.num_attention_heads, head_dim)
    captures["_trace_attention_context"] = trace

    o_raw = project(attn_codes, np.asarray(model.weights[f"{prefix}.o_proj"]))
    folds, _ = pipeline._reference_fold(model.weight_scales[f"{prefix}.o_proj"])
    o_folded = fold_rows(o_raw, folds)
    o_codes, o_scales, trace = chain(
        model, f"{prefix}.o_proj.requant", o_folded,
        [[attn_scales[t]] for t in range(steps)])
    captures["o_proj"] = np.stack([dequant(o_codes[t], o_scales[t]) for t in range(steps)])
    captures["_trace_o_proj"] = trace

    residual_wide = np.empty(hidden.shape, dtype=np.int64)
    for t in range(steps):
        m_h, e_h = hidden_scales[t]
        m_b, e_b = o_scales[t]
        r_h = intmath.dynamic_scale_reciprocal(m_h)
        residual_wide[t] = np.asarray([
            int(hidden[t, i]) + intmath.residual_reconcile(
                int(o_codes[t, i]), m_b, r_h, e_b, e_h)
            for i in range(cfg.hidden_size)], dtype=np.int64)
    residual1, residual1_scales, trace = chain(
        model, f"{prefix}.attn_residual", residual_wide,
        [[hidden_scales[t]] for t in range(steps)])
    captures["residual1"] = np.stack([
        dequant(residual1[t], residual1_scales[t]) for t in range(steps)])
    captures["_trace_residual1"] = trace

    mlp_gain = np.asarray(model.weights[f"{prefix}.mlp_norm.gain"])
    mlp_normed, mlp_norm_scales, trace = rmsnorm_integer(
        model, f"{prefix}.mlp_norm", residual1, mlp_gain)
    captures["post_attention_rmsnorm"] = np.stack([
        dequant(mlp_normed[t], mlp_norm_scales[t]) for t in range(steps)])
    captures["_trace_post_attention_rmsnorm"] = trace

    mlp_codes: dict[str, np.ndarray] = {}
    mlp_scales: dict[str, list[tuple[int, int]]] = {}
    for leaf in ("gate_proj", "up_proj"):
        raw = project(mlp_normed, np.asarray(model.weights[f"{prefix}.{leaf}"]))
        folds, _ = pipeline._reference_fold(model.weight_scales[f"{prefix}.{leaf}"])
        folded = fold_rows(raw, folds)
        codes, scales, trace = chain(
            model, f"{prefix}.{leaf}.requant", folded,
            [[mlp_norm_scales[t]] for t in range(steps)])
        mlp_codes[leaf], mlp_scales[leaf] = codes, scales
        captures[leaf] = np.stack([dequant(codes[t], scales[t]) for t in range(steps)])
        captures[f"_trace_{leaf}"] = trace

    silu_values = np.empty(mlp_codes["gate_proj"].shape, dtype=np.float64)
    act_wide = np.empty(mlp_codes["gate_proj"].shape, dtype=np.int64)
    for t in range(steps):
        m_g, e_g = mlp_scales["gate_proj"][t]
        sig = np.asarray([
            silu_lut.silu_sigmoid_q15(int(c), m_g, e_g)
            for c in mlp_codes["gate_proj"][t]], dtype=np.int64)
        silu_values[t] = (
            dequant(mlp_codes["gate_proj"][t], (m_g, e_g)) * sig /
            (1 << silu_lut.TABLE_FRAC_BITS))
        act_wide[t] = (
            mlp_codes["gate_proj"][t].astype(np.int64) * sig *
            mlp_codes["up_proj"][t].astype(np.int64))
    captures["silu"] = silu_values
    act_codes, act_scales, trace = chain(
        model, f"{prefix}.mlp_act", act_wide,
        [[mlp_scales["gate_proj"][t], mlp_scales["up_proj"][t]] for t in range(steps)])
    captures["mlp_activation"] = np.stack([
        dequant(act_codes[t], act_scales[t]) for t in range(steps)])
    captures["_trace_mlp_activation"] = trace

    down_raw = project(act_codes, np.asarray(model.weights[f"{prefix}.down_proj"]))
    folds, _ = pipeline._reference_fold(model.weight_scales[f"{prefix}.down_proj"])
    down_folded = fold_rows(down_raw, folds)
    down_codes, down_scales, trace = chain(
        model, f"{prefix}.down_proj.requant", down_folded,
        [[act_scales[t]] for t in range(steps)])
    captures["down_proj"] = np.stack([
        dequant(down_codes[t], down_scales[t]) for t in range(steps)])
    captures["_trace_down_proj"] = trace

    residual2_wide = np.empty(residual1.shape, dtype=np.int64)
    for t in range(steps):
        m_h, e_h = residual1_scales[t]
        m_b, e_b = down_scales[t]
        r_h = intmath.dynamic_scale_reciprocal(m_h)
        residual2_wide[t] = np.asarray([
            int(residual1[t, i]) + intmath.residual_reconcile(
                int(down_codes[t, i]), m_b, r_h, e_b, e_h)
            for i in range(cfg.hidden_size)], dtype=np.int64)
    residual2, residual2_scales, trace = chain(
        model, f"{prefix}.mlp_residual", residual2_wide,
        [[residual1_scales[t]] for t in range(steps)])
    captures["residual2"] = np.stack([
        dequant(residual2[t], residual2_scales[t]) for t in range(steps)])
    captures["_trace_residual2"] = trace
    captures["_codes"] = {
        "q_rope": q_rot, "k_rope": k_rot, "attention_scores": score_values,
        "softmax": probability_values,
    }
    for record in kv_trace:
        site_records.append({
            "site": record["site"], "token": record["token_index"], "head": record["head"],
            "x_int": list(record["x_int"]), "codes": list(record["codes"]),
        })
    chain_groups = (
        embed_trace, captures["_trace_input_rmsnorm"], captures["_trace_q_proj"],
        q_norm_trace, k_norm_trace, captures["_trace_attention_context"],
        captures["_trace_o_proj"], captures["_trace_residual1"],
        captures["_trace_post_attention_rmsnorm"], captures["_trace_gate_proj"],
        captures["_trace_up_proj"], captures["_trace_mlp_activation"],
        captures["_trace_down_proj"], captures["_trace_residual2"],
    )
    expected: list[dict] = []
    for group_records in chain_groups:
        for record in group_records:
            expected.append({
                "kind": "chain", "site": record["site"], "token": record["token_index"],
                "head": record.get("head", 0), "x_int": list(record["x_int"]),
                "d_prime": record["Dprime"], "dn": record["Dn"], "s": record["s"],
                "r": record["R"], "codes": list(record["codes"]),
                "m_out": record["m_out"], "e_out": record["e_out"],
            })
    expected.extend({"kind": "site", **record} for record in site_records)
    captures["_cpp_expected"] = expected
    captures["_diagnostics"] = {
        "k_projection_codes": int(k_codes.size),
        "k_projection_nonzero_codes": int(np.count_nonzero(np.asarray([
            record["codes"] for record in kv_trace if record["site"].endswith("k_proj.requant")
        ]))),
        "k_projection_scale_per_code": scale_value(k_scale_raw[0][0]),
    }
    captures["clamps"] = clamp_counts
    return captures


def float_block0(model, token_ids: list[int]) -> dict[str, np.ndarray | list]:
    import torch
    import torch.nn.functional as F
    from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb, repeat_kv

    device = next(model.parameters()).device
    ids = torch.tensor([token_ids], dtype=torch.long, device=device)
    steps = len(token_ids)
    position_ids = torch.arange(steps, device=device).unsqueeze(0)
    layer = model.layers[0]
    captures: dict[str, object] = {}
    with torch.no_grad():
        hidden = model.embed_tokens(ids)
        captures["embedding"] = hidden[0].float().cpu().numpy()
        normed = layer.input_layernorm(hidden)
        captures["input_rmsnorm"] = normed[0].float().cpu().numpy()
        q_proj_native = layer.self_attn.q_proj(normed).view(
            1, steps, model.config.num_attention_heads, model.config.head_dim)
        k_proj_native = layer.self_attn.k_proj(normed).view(
            1, steps, model.config.num_key_value_heads, model.config.head_dim)
        v_native = layer.self_attn.v_proj(normed).view(
            1, steps, model.config.num_key_value_heads, model.config.head_dim)
        order = pipeline._rope_pair_permutation(model.config.head_dim)
        captures["q_proj"] = q_proj_native[0].float().cpu().numpy()[:, :, order]
        captures["k_proj"] = k_proj_native[0].float().cpu().numpy()[:, :, order]
        captures["v_proj"] = v_native[0].float().cpu().numpy()
        q_norm_native = layer.self_attn.q_norm(q_proj_native)
        k_norm_native = layer.self_attn.k_norm(k_proj_native)
        captures["q_norm"] = q_norm_native[0].float().cpu().numpy()[:, :, order]
        captures["k_norm"] = k_norm_native[0].float().cpu().numpy()[:, :, order]
        captures["k_norm_landed"] = captures["k_norm"]

        q = q_norm_native.transpose(1, 2)
        k = k_norm_native.transpose(1, 2)
        v = v_native.transpose(1, 2)
        cos, sin = model.rotary_emb(hidden, position_ids)
        q_rot, k_rot = apply_rotary_pos_emb(q, k, cos, sin)
        captures["rope_q"] = q_rot[0].transpose(0, 1).float().cpu().numpy()[:, :, order]
        captures["rope_k"] = k_rot[0].transpose(0, 1).float().cpu().numpy()[:, :, order]

        keys = repeat_kv(k_rot, layer.self_attn.num_key_value_groups)
        values = repeat_kv(v, layer.self_attn.num_key_value_groups)
        raw_scores = torch.matmul(q_rot, keys.transpose(2, 3)) * layer.self_attn.scaling
        causal = torch.triu(
            torch.full((steps, steps), float("-inf"), device=device, dtype=raw_scores.dtype),
            diagonal=1,
        ).view(1, 1, steps, steps)
        masked_scores = raw_scores + causal
        probs = F.softmax(masked_scores, dim=-1, dtype=torch.float32).to(q_rot.dtype)
        weighted = torch.matmul(probs, values).transpose(1, 2).contiguous()
        captures["attention_scores"] = [
            raw_scores[0, :, t, :t + 1].float().cpu().numpy() for t in range(steps)]
        captures["softmax"] = [
            probs[0, :, t, :t + 1].float().cpu().numpy() for t in range(steps)]
        captures["weighted_sum"] = weighted[0].float().cpu().numpy()
        captures["attention_context"] = captures["weighted_sum"]
        attention_flat = weighted.reshape(1, steps, -1)
        attention = layer.self_attn.o_proj(attention_flat)
        captures["o_proj"] = attention[0].float().cpu().numpy()
        residual1 = hidden + attention
        captures["residual1"] = residual1[0].float().cpu().numpy()
        mlp_normed = layer.post_attention_layernorm(residual1)
        captures["post_attention_rmsnorm"] = mlp_normed[0].float().cpu().numpy()
        gate = layer.mlp.gate_proj(mlp_normed)
        up = layer.mlp.up_proj(mlp_normed)
        silu = layer.mlp.act_fn(gate)
        activation = silu * up
        captures["gate_proj"] = gate[0].float().cpu().numpy()
        captures["up_proj"] = up[0].float().cpu().numpy()
        captures["silu"] = silu[0].float().cpu().numpy()
        captures["mlp_activation"] = activation[0].float().cpu().numpy()
        down = layer.mlp.down_proj(activation)
        captures["down_proj"] = down[0].float().cpu().numpy()
        residual2 = residual1 + down
        captures["residual2"] = residual2[0].float().cpu().numpy()

        # Independently invoke the real layer body and require our captured composition to match.
        direct = layer(
            hidden, attention_mask=causal, position_ids=position_ids,
            position_embeddings=(cos, sin), use_cache=False)
        if not torch.equal(residual2, direct):
            delta = float((residual2.float() - direct.float()).abs().max().item())
            raise RuntimeError(f"manual float block differs from Qwen3DecoderLayer: max_abs={delta}")
    return captures


SITE_ORDER = (
    "embedding", "input_rmsnorm", "q_proj", "k_proj", "v_proj", "q_norm", "k_norm",
    "k_norm_landed", "rope_q", "rope_k", "attention_scores", "softmax", "weighted_sum",
    "attention_context", "o_proj", "residual1", "post_attention_rmsnorm", "gate_proj",
    "up_proj", "silu", "mlp_activation", "down_proj", "residual2",
)


def flatten_site(value) -> np.ndarray:
    if isinstance(value, list):
        return np.concatenate([np.asarray(v, dtype=np.float64).ravel() for v in value])
    return np.asarray(value, dtype=np.float64).ravel()


def metrics(candidate, reference) -> dict[str, float]:
    candidate = flatten_site(candidate)
    reference = flatten_site(reference)
    if candidate.shape != reference.shape:
        raise RuntimeError(f"metric shape mismatch: {candidate.shape} != {reference.shape}")
    denom = float(np.max(np.abs(reference), initial=0.0))
    return {
        "cosine": float(np.dot(candidate, reference) /
                        (np.linalg.norm(candidate) * np.linalg.norm(reference))),
        "max_abs_relative_error": float(np.max(np.abs(candidate - reference), initial=0.0) /
                                        max(denom, np.finfo(np.float64).tiny)),
        "max_abs_error": float(np.max(np.abs(candidate - reference), initial=0.0)),
        "reference_max_abs": denom,
        "elements": int(reference.size),
    }


def terminal_site(value):
    return value[-1]


def verify_cpp_trace(args, ids: list[int], expected: list[dict], dump: Path) -> dict:
    dump.parent.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(
        [str(args.cpp_exe), str(args.sslm_model), ",".join(str(v) for v in ids), str(dump)],
        check=False, capture_output=True, text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"C++ trace failed ({completed.returncode}): {completed.stdout}\n{completed.stderr}")
    rows = [json.loads(line) for line in dump.read_text(encoding="utf-8").splitlines()]
    if rows[0] != {"kind": "header", "token_ids": ids}:
        raise RuntimeError("C++ trace token provenance mismatch")
    actual = [row for row in rows if row["kind"] in ("chain", "site")]
    key = lambda row: (row["kind"], row["site"], row["token"], row["head"])
    expected_map = {key(row): row for row in expected}
    actual_map = {key(row): row for row in actual}
    if len(expected_map) != len(expected) or len(actual_map) != len(actual):
        raise RuntimeError("duplicate C++/Python trace identity")
    if expected_map.keys() != actual_map.keys():
        missing = sorted(expected_map.keys() - actual_map.keys())[:5]
        extra = sorted(actual_map.keys() - expected_map.keys())[:5]
        raise RuntimeError(f"C++ trace record-set mismatch: missing={missing}, extra={extra}")
    mismatches = []
    for identity, expected_row in expected_map.items():
        if actual_map[identity] != expected_row:
            differing = [field for field in expected_row if expected_row[field] != actual_map[identity].get(field)]
            mismatches.append({"identity": identity, "fields": differing})
            if len(mismatches) == 5:
                break
    if mismatches:
        raise RuntimeError(f"C++ differs from exact Python integer reference: {mismatches}")
    cpp_clamps = {
        name: int(sum(row[name] for row in rows if row["kind"] == "clamps"))
        for name in ("kv_landing", "k_norm_landing", "rope_q", "rope_k")
    }
    return {
        "exact_records": len(actual), "mismatch_records": 0, "clamps": cpp_clamps,
        "dump": str(dump),
    }


def selected_rows(corpus: Path):
    rows = [json.loads(line) for line in corpus.read_text(encoding="utf-8").splitlines() if line]
    found = []
    seen = set()
    for index, row in enumerate(rows):
        if row["class"] in seen:
            continue
        try:
            row["text"].encode("ascii")
        except UnicodeEncodeError:
            continue
        seen.add(row["class"])
        found.append((index, row))
    classes = {row["class"] for row in rows}
    if seen != classes:
        raise RuntimeError(f"no ASCII specimen for {classes - seen}")
    return found


def run(args) -> dict:
    import torch
    from transformers import AutoModel, AutoTokenizer

    samples = selected_rows(args.corpus)
    tokenizer = AutoTokenizer.from_pretrained(str(args.hf_model), local_files_only=True)
    model_int = load_artifact(args.integer_cache)
    started = time.perf_counter()
    model_fp32 = AutoModel.from_pretrained(
        str(args.hf_model), local_files_only=True, dtype=torch.float32,
        attn_implementation="eager").eval().to(args.device)
    output: dict[str, object] = {
        "provenance": {
            "corpus": str(args.corpus),
            "corpus_sha256": hashlib.sha256(args.corpus.read_bytes()).hexdigest(),
            "integer_cache": str(args.integer_cache),
            "integer_cache_fingerprint": json.loads(
                (args.integer_cache / "manifest.json").read_text(encoding="utf-8"))[
                    "source_fingerprint"],
            "hf_model": str(args.hf_model), "device": args.device,
            "sslm_model": str(args.sslm_model), "cpp_exe": str(args.cpp_exe),
            "sslm_sha256": hashlib.sha256(args.sslm_model.read_bytes()).hexdigest(),
            "float_dtype": "torch.float32", "sample_count_per_path": len(samples),
            "metric_definition": (
                "cosine and max(abs(candidate-reference))/max(abs(reference)), flattened over "
                "every element of the named site for one specimen; aggregates are arithmetic "
                "means across the seven one-per-class specimens"),
        },
        "paths": {"query": [], "document": []},
    }
    fp32_outputs: dict[tuple[str, str], np.ndarray] = {}
    for kind in ("query", "document"):
        for _index, row in samples:
            text = QUERY_PREFIX + row["text"] if kind == "query" else row["text"]
            ids = tokenizer(text)["input_ids"]
            float_sites = float_block0(model_fp32, ids)
            int_sites = integer_block0(model_int, ids)
            cpp = verify_cpp_trace(
                args, ids, int_sites["_cpp_expected"],
                args.cpp_dir / f"{kind}-{row['label']}.jsonl")
            if cpp["clamps"] != int_sites["clamps"]:
                raise RuntimeError(
                    f"C++ clamp counters differ for {kind}/{row['label']}: "
                    f"{cpp['clamps']} != {int_sites['clamps']}")
            specimen = {
                "label": row["label"], "class": row["class"], "token_count": len(ids),
                "sites": {site: {
                    "python_integer_vs_float32": metrics(int_sites[site], float_sites[site]),
                    "terminal_python_integer_vs_float32": metrics(
                        terminal_site(int_sites[site]), terminal_site(float_sites[site])),
                }
                          for site in SITE_ORDER},
                "clamps": int_sites["clamps"],
                "cpp_vs_python_integer": cpp,
                "diagnostics": int_sites["_diagnostics"],
            }
            specimen["diagnostics"]["float32_k_projection_max_abs"] = float(
                np.max(np.abs(float_sites["k_proj"]), initial=0.0))
            output["paths"][kind].append(specimen)
            fp32_outputs[(kind, row["label"])] = np.asarray(float_sites["residual2"][-1]).copy()

    del model_fp32
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    model_bf16 = AutoModel.from_pretrained(
        str(args.hf_model), local_files_only=True, dtype=torch.bfloat16,
        attn_implementation="eager").eval().to(args.device)
    precision_rows = []
    for kind in ("query", "document"):
        for _index, row in samples:
            text = QUERY_PREFIX + row["text"] if kind == "query" else row["text"]
            ids = tokenizer(text)["input_ids"]
            bf16 = float_block0(model_bf16, ids)["residual2"][-1]
            precision_rows.append({
                "path": kind, "label": row["label"],
                "cosine": metrics(bf16, fp32_outputs[(kind, row["label"])])["cosine"],
            })
    del model_bf16
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    output["bf16_vs_fp32_block0_terminal"] = {
        "rows": precision_rows,
        "mean_cosine": float(np.mean([row["cosine"] for row in precision_rows])),
        "query_mean_cosine": float(np.mean([row["cosine"] for row in precision_rows
                                             if row["path"] == "query"])),
        "document_mean_cosine": float(np.mean([row["cosine"] for row in precision_rows
                                                if row["path"] == "document"])),
    }
    aggregates = {}
    for kind, specimens in output["paths"].items():
        aggregates[kind] = {}
        for site in SITE_ORDER:
            rows = [item["sites"][site]["python_integer_vs_float32"] for item in specimens]
            aggregates[kind][site] = {
                "mean_cosine": float(np.mean([r["cosine"] for r in rows])),
                "min_cosine": float(np.min([r["cosine"] for r in rows])),
                "mean_max_abs_relative_error": float(np.mean(
                    [r["max_abs_relative_error"] for r in rows])),
                "max_max_abs_relative_error": float(np.max(
                    [r["max_abs_relative_error"] for r in rows])),
                "terminal_mean_cosine": float(np.mean([
                    item["sites"][site]["terminal_python_integer_vs_float32"]["cosine"]
                    for item in specimens])),
                "terminal_mean_max_abs_relative_error": float(np.mean([
                    item["sites"][site]["terminal_python_integer_vs_float32"][
                        "max_abs_relative_error"] for item in specimens])),
            }
        aggregates[kind]["clamps"] = {
            name: {"total": int(sum(s["clamps"][name] for s in specimens)),
                   "per_specimen": [int(s["clamps"][name]) for s in specimens]}
            for name in ("kv_landing", "k_norm_landing", "rope_q", "rope_k")
        }
        aggregates[kind]["cpp_vs_python_integer"] = {
            "exact_records": int(sum(s["cpp_vs_python_integer"]["exact_records"]
                                     for s in specimens)),
            "mismatch_records": 0,
            "all_clamp_counters_exact": True,
        }
        k_codes = sum(s["diagnostics"]["k_projection_codes"] for s in specimens)
        k_nonzero = sum(s["diagnostics"]["k_projection_nonzero_codes"] for s in specimens)
        aggregates[kind]["k_projection_landing"] = {
            "scale_per_code": specimens[0]["diagnostics"]["k_projection_scale_per_code"],
            "codes": k_codes, "nonzero_codes": k_nonzero,
            "nonzero_fraction": k_nonzero / k_codes,
            "max_float32_abs_on_measured_specimens": float(max(
                s["diagnostics"]["float32_k_projection_max_abs"] for s in specimens)),
        }
    k_output_scale = float(model_int.scales.output_scale("layer0.k_proj.requant"))
    output["localization"] = {
        "classification": "converter/calibrator defect",
        "first_departure": "layer0.k_proj.requant static landing",
        "artifact_projection_output_scales": {
            leaf: float(model_int.scales.output_scale(f"layer0.{leaf}.requant"))
            for leaf in ("q_proj", "k_proj", "v_proj")
        },
        "k_scale_implied_calibration_peak": k_output_scale * 127.0,
        "source_mechanism": (
            "_float_layer observes post-qk_norm/post-RoPE K under the raw layer0.k key; "
            "_projection_scale later converts that union maximum into the pre-norm K landing scale"),
    }
    output["aggregate"] = aggregates
    output["elapsed_seconds"] = time.perf_counter() - started
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--integer-cache", type=Path, required=True)
    parser.add_argument("--hf-model", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--cpp-exe", type=Path, required=True)
    parser.add_argument("--sslm-model", type=Path, required=True)
    parser.add_argument("--cpp-dir", type=Path, required=True)
    args = parser.parse_args()
    result = run(args)
    args.result.parent.mkdir(parents=True, exist_ok=True)
    args.result.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["aggregate"], indent=2))
    print(json.dumps(result["bf16_vs_fp32_block0_terminal"], indent=2))
    print(f"result_written={args.result}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

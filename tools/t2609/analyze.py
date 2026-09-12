#!/usr/bin/env python3
"""T-2609: localize score-to-softmax loss and measure a softmax counterfactual."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import subprocess
import time
from pathlib import Path

import numpy as np

from reference_pipeline import intmath, pipeline, rope
from reference_pipeline.artifact_cache import load_artifact


QUERY_PREFIX = (
    "Instruct: Given a web search query, retrieve relevant passages that answer the query\n"
    "Query:"
)
PROB_ONE = 1 << pipeline.PROB_FRAC_BITS
TARGET_LAYERS = (1, 14, 27)


def load_t2604(path: Path):
    spec = importlib.util.spec_from_file_location("t2604_analyze", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def softmax(values) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64)
    e = np.exp(x - np.max(x))
    return e / np.sum(e)


def cosine(candidate, reference) -> float:
    a = flatten(candidate)
    b = flatten(reference)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def flatten(value) -> np.ndarray:
    if isinstance(value, list):
        return np.concatenate([np.asarray(row, dtype=np.float64).ravel() for row in value])
    return np.asarray(value, dtype=np.float64).ravel()


def exact_probabilities(scores) -> list[np.ndarray]:
    return [np.stack([softmax(row) for row in np.asarray(token)]) for token in scores]


def probability_exp_stream(probabilities) -> list[int]:
    # SoftmaxRowQ15 only consumes ratios. A shared 2**50 denominator preserves the
    # supplied float distribution to far below its Q15 output resolution.
    stream: list[int] = []
    for token in probabilities:
        for row in np.asarray(token, dtype=np.float64):
            stream.extend(max(1, int(round(float(p) * (1 << 50)))) for p in row)
    return stream


def run_block0_override(t2604, model_int, ids, probabilities):
    stream = probability_exp_stream(probabilities)
    cursor = 0
    original = intmath.i_exp_from_constants

    def supplied(_q, _q_ln2, _q_b, _q_c):
        nonlocal cursor
        if cursor >= len(stream):
            raise RuntimeError("softmax override stream exhausted early")
        value = stream[cursor]
        cursor += 1
        return value

    intmath.i_exp_from_constants = supplied
    try:
        result = t2604.integer_block0(model_int, ids)
    finally:
        intmath.i_exp_from_constants = original
    if cursor != len(stream):
        raise RuntimeError(f"softmax override consumed {cursor}/{len(stream)} exponentials")
    return result


def kernel_probabilities_from_float_scores(model_int, int_sites, float_scores, layer_index=0):
    q_records = {(int(r["token_index"]), int(r["head"])): r
                 for r in int_sites["_trace_q_norm"]}
    group = model_int.config.num_attention_heads // model_int.config.num_key_value_heads
    output = []
    quantization = []
    prefix = f"layer{layer_index}"
    for t, token in enumerate(float_scores):
        rows = []
        for head, float_row in enumerate(np.asarray(token, dtype=np.float64)):
            kv_head = head // group
            qr = q_records[(t, head)]
            static = model_int.composition_constants[f"{prefix}.softmax_khead{kv_head}"]
            sm_scale = intmath.carried_scale_product(
                [(int(qr["m_out"]), int(qr["e_out"])), static])
            scale = math.ldexp(float(sm_scale[0]), int(sm_scale[1]))
            codes = np.rint(float_row / scale).astype(np.int64)
            shifted = intmath.shift_by_max(codes.tolist())
            q_ln2, q_b, q_c = intmath.iexp_scale_constants(
                sm_scale[0], sm_scale[1], pipeline._IEXP_LN2_Q, pipeline._IEXP_QFMT,
                pipeline._IEXP_B_Q, pipeline._IEXP_QFMT, pipeline._IEXP_CA_Q,
                pipeline._IEXP_QFMT)
            exps = [intmath.i_exp_from_constants(int(q), q_ln2, q_b, q_c) for q in shifted]
            total = max(sum(exps), 1)
            rows.append(np.asarray([(e << pipeline.PROB_FRAC_BITS) // total for e in exps],
                                   dtype=np.float64) / PROB_ONE)
            quantization.append({
                "scale": scale,
                "max_abs_error": float(np.max(np.abs(codes * scale - float_row), initial=0.0)),
                "q_ln2": int(q_ln2),
            })
        output.append(np.stack(rows))
    return output, quantization


def centered_score_diagnostics(integer_scores, float_scores) -> dict:
    slopes = []
    range_ratios = []
    centered_cosines = []
    for int_token, float_token in zip(integer_scores, float_scores):
        for a, b in zip(np.asarray(int_token, dtype=np.float64),
                        np.asarray(float_token, dtype=np.float64)):
            if a.size < 2:
                continue
            ac = a - np.mean(a)
            bc = b - np.mean(b)
            aa = float(np.dot(ac, ac))
            if aa > 0 and np.linalg.norm(bc) > 0:
                slopes.append(float(np.dot(ac, bc) / aa))
                centered_cosines.append(float(np.dot(ac, bc) /
                                               (np.linalg.norm(ac) * np.linalg.norm(bc))))
            ar = float(np.max(a) - np.min(a))
            br = float(np.max(b) - np.min(b))
            if ar > 0:
                range_ratios.append(br / ar)
    return {
        "rows": len(slopes),
        "median_float_over_integer_centered_slope": float(np.median(slopes)),
        "mean_float_over_integer_centered_slope": float(np.mean(slopes)),
        "median_float_over_integer_range_ratio": float(np.median(range_ratios)),
        "mean_centered_cosine": float(np.mean(centered_cosines)),
    }


def score_hybrids(model_int, int_sites, float_sites, layer_index=0):
    cfg = model_int.config
    prefix = f"layer{layer_index}"
    group = cfg.num_attention_heads // cfg.num_key_value_heads
    q_records = {(int(r["token_index"]), int(r["head"])): r
                 for r in int_sites["_trace_q_norm"]}
    k_records = {(int(r["token_index"]), int(r["head"])): r
                 for r in int_sites["_trace_k_norm"]}
    int_q = np.asarray(int_sites["rope_q"], dtype=np.float64)
    int_k = np.asarray(int_sites["rope_k"], dtype=np.float64)
    float_q = np.asarray(float_sites["rope_q"], dtype=np.float64)
    float_k = np.asarray(float_sites["rope_k"], dtype=np.float64)
    unclamped_q = np.empty_like(int_q)
    unlanded_k = np.empty_like(int_k)
    for t in range(int_q.shape[0]):
        for head in range(cfg.num_attention_heads):
            record = q_records[(t, head)]
            scale = math.ldexp(float(record["m_out"]), int(record["e_out"]))
            unclamped_q[t, head] = np.asarray(
                int_sites["_raw_rope_q"][t, head], dtype=np.float64) * scale
    cos_table, sin_table = model_int.rope_tables
    for t in range(int_k.shape[0]):
        for head in range(cfg.num_key_value_heads):
            record = k_records[(t, head)]
            codes = np.asarray(record["codes"], dtype=np.int64)
            rotated = np.empty(cfg.head_dim, dtype=np.int64)
            for pair in range(cfg.head_dim // 2):
                x, y = rope.rope_apply_pair(
                    int(codes[2 * pair]), int(codes[2 * pair + 1]),
                    int(cos_table[t][pair]), int(sin_table[t][pair]))
                rotated[2 * pair:2 * pair + 2] = (
                    max(-127, min(127, x)), max(-127, min(127, y)))
            scale = math.ldexp(float(record["m_out"]), int(record["e_out"]))
            unlanded_k[t, head] = rotated.astype(np.float64) * scale

    def scores(q_values, k_values):
        rows = []
        for t in range(q_values.shape[0]):
            rows.append(np.stack([
                q_values[t, head] @ k_values[:t + 1, head // group].T /
                math.sqrt(cfg.head_dim)
                for head in range(cfg.num_attention_heads)
            ]))
        return rows

    variants = {
        "recomputed_integer": scores(int_q, int_k),
        "float_q_integer_k": scores(float_q, int_k),
        "integer_q_float_k": scores(int_q, float_k),
        "unclamped_integer_q_integer_k": scores(unclamped_q, int_k),
        "integer_q_unlanded_integer_k": scores(int_q, unlanded_k),
        "float_q_unlanded_integer_k": scores(float_q, unlanded_k),
    }
    return variants, {name: exact_probabilities(value) for name, value in variants.items()}


def block0_campaign(args, t2604, tokenizer, model_fp32, model_int) -> dict:
    samples = t2604.selected_rows(args.corpus)
    paths = {}
    for kind in ("query", "document"):
        rows = []
        for _index, specimen in samples:
            text = QUERY_PREFIX + specimen["text"] if kind == "query" else specimen["text"]
            ids = tokenizer(text)["input_ids"]
            float_sites = t2604.float_block0(model_fp32, ids)
            baseline = t2604.integer_block0(model_int, ids)
            hybrid_scores, hybrid_probs = score_hybrids(model_int, baseline, float_sites)
            exact_on_integer_probs = exact_probabilities(baseline["attention_scores"])
            exact_on_integer = run_block0_override(
                t2604, model_int, ids, exact_on_integer_probs)
            float_softmax = run_block0_override(
                t2604, model_int, ids, float_sites["softmax"])
            float_score_kernel, quantization = kernel_probabilities_from_float_scores(
                model_int, baseline, float_sites["attention_scores"])
            float_score_walk = run_block0_override(
                t2604, model_int, ids, float_score_kernel)
            rows.append({
                "label": specimen["label"],
                "token_count": len(ids),
                "cosines": {
                    "attention_scores_baseline": cosine(
                        baseline["attention_scores"], float_sites["attention_scores"]),
                    "softmax_baseline": cosine(baseline["softmax"], float_sites["softmax"]),
                    "softmax_exact_on_integer_scores": cosine(
                        exact_on_integer["softmax"], float_sites["softmax"]),
                    "softmax_integer_kernel_on_float_scores": cosine(
                        float_score_kernel, float_sites["softmax"]),
                    "softmax_float_q_integer_k": cosine(
                        hybrid_probs["float_q_integer_k"], float_sites["softmax"]),
                    "softmax_integer_q_float_k": cosine(
                        hybrid_probs["integer_q_float_k"], float_sites["softmax"]),
                    "softmax_unclamped_integer_q_integer_k": cosine(
                        hybrid_probs["unclamped_integer_q_integer_k"], float_sites["softmax"]),
                    "softmax_integer_q_unlanded_integer_k": cosine(
                        hybrid_probs["integer_q_unlanded_integer_k"], float_sites["softmax"]),
                    "softmax_float_q_unlanded_integer_k": cosine(
                        hybrid_probs["float_q_unlanded_integer_k"], float_sites["softmax"]),
                    "weighted_sum_baseline": cosine(
                        baseline["weighted_sum"], float_sites["weighted_sum"]),
                    "weighted_sum_exact_on_integer_scores": cosine(
                        exact_on_integer["weighted_sum"], float_sites["weighted_sum"]),
                    "weighted_sum_float_softmax_override": cosine(
                        float_softmax["weighted_sum"], float_sites["weighted_sum"]),
                    "weighted_sum_float_score_integer_kernel": cosine(
                        float_score_walk["weighted_sum"], float_sites["weighted_sum"]),
                    "residual2_baseline": cosine(baseline["residual2"], float_sites["residual2"]),
                    "residual2_exact_on_integer_scores": cosine(
                        exact_on_integer["residual2"], float_sites["residual2"]),
                    "residual2_float_softmax_override": cosine(
                        float_softmax["residual2"], float_sites["residual2"]),
                    "residual2_float_score_integer_kernel": cosine(
                        float_score_walk["residual2"], float_sites["residual2"]),
                    "terminal_residual2_baseline": cosine(
                        baseline["residual2"][-1], float_sites["residual2"][-1]),
                    "terminal_residual2_float_softmax_override": cosine(
                        float_softmax["residual2"][-1], float_sites["residual2"][-1]),
                    "terminal_residual2_float_score_integer_kernel": cosine(
                        float_score_walk["residual2"][-1], float_sites["residual2"][-1]),
                },
                "score_shape": centered_score_diagnostics(
                    baseline["attention_scores"], float_sites["attention_scores"]),
                "score_recomposition": {
                    "max_abs_error": float(np.max(np.abs(
                        flatten(hybrid_scores["recomputed_integer"]) -
                        flatten(baseline["attention_scores"])), initial=0.0)),
                    "cosine": cosine(
                        hybrid_scores["recomputed_integer"], baseline["attention_scores"]),
                },
                "float_score_requant": {
                    "max_abs_error": max(q["max_abs_error"] for q in quantization),
                    "q_ln2_min": min(q["q_ln2"] for q in quantization),
                    "q_ln2_max": max(q["q_ln2"] for q in quantization),
                },
            })
        keys = rows[0]["cosines"].keys()
        means = {key: float(np.mean([r["cosines"][key] for r in rows])) for key in keys}
        base = means["residual2_baseline"]
        counter = means["residual2_float_softmax_override"]
        score_counter = means["residual2_float_score_integer_kernel"]
        paths[kind] = {
            "rows": rows,
            "mean_cosines": means,
            "softmax_counterfactual": {
                "residual2_cosine_gain": counter - base,
                "residual2_cosine_loss_recovered_fraction": (counter - base) / (1.0 - base),
                "terminal_residual2_cosine_gain": (
                    means["terminal_residual2_float_softmax_override"] -
                    means["terminal_residual2_baseline"]),
                "float_score_integer_kernel_residual2_cosine_gain": score_counter - base,
                "float_score_integer_kernel_loss_recovered_fraction": (
                    (score_counter - base) / (1.0 - base)),
                "float_score_integer_kernel_terminal_residual2_cosine_gain": (
                    means["terminal_residual2_float_score_integer_kernel"] -
                    means["terminal_residual2_baseline"]),
            },
            "score_shape": {
                key: float(np.mean([r["score_shape"][key] for r in rows]))
                for key in ("median_float_over_integer_centered_slope",
                            "mean_float_over_integer_centered_slope",
                            "median_float_over_integer_range_ratio",
                            "mean_centered_cosine")
            },
            "float_score_requant": {
                "max_abs_error": max(r["float_score_requant"]["max_abs_error"] for r in rows),
                "q_ln2_min": min(r["float_score_requant"]["q_ln2_min"] for r in rows),
                "q_ln2_max": max(r["float_score_requant"]["q_ln2_max"] for r in rows),
            },
        }
    return paths


def float_target_layers(model, ids, layers):
    import torch
    import torch.nn.functional as F
    from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb, repeat_kv

    device = next(model.parameters()).device
    token_tensor = torch.tensor([ids], dtype=torch.long, device=device)
    steps = len(ids)
    position_ids = torch.arange(steps, device=device).unsqueeze(0)
    causal = torch.triu(torch.full((steps, steps), float("-inf"), device=device), diagonal=1).view(
        1, 1, steps, steps)
    with torch.no_grad():
        full = model(input_ids=token_tensor, attention_mask=causal,
                     output_hidden_states=True, use_cache=False)
        output = {}
        for layer_index in layers:
            hidden = full.hidden_states[layer_index]
            layer = model.layers[layer_index]
            normed = layer.input_layernorm(hidden)
            q_native = layer.self_attn.q_proj(normed).view(
                1, steps, model.config.num_attention_heads, model.config.head_dim)
            k_native = layer.self_attn.k_proj(normed).view(
                1, steps, model.config.num_key_value_heads, model.config.head_dim)
            v_native = layer.self_attn.v_proj(normed).view(
                1, steps, model.config.num_key_value_heads, model.config.head_dim)
            q_norm = layer.self_attn.q_norm(q_native)
            k_norm = layer.self_attn.k_norm(k_native)
            q = q_norm.transpose(1, 2)
            k = k_norm.transpose(1, 2)
            v = v_native.transpose(1, 2)
            cos_table, sin_table = model.rotary_emb(hidden, position_ids)
            q_rot, k_rot = apply_rotary_pos_emb(q, k, cos_table, sin_table)
            keys = repeat_kv(k_rot, layer.self_attn.num_key_value_groups)
            values = repeat_kv(v, layer.self_attn.num_key_value_groups)
            scores = torch.matmul(q_rot, keys.transpose(2, 3)) * layer.self_attn.scaling
            probs = F.softmax(scores + causal, dim=-1, dtype=torch.float32).to(q_rot.dtype)
            weighted = torch.matmul(probs, values).transpose(1, 2).contiguous()
            order = pipeline._rope_pair_permutation(model.config.head_dim)
            output[layer_index] = {
                "rope_q": q_rot[0].transpose(0, 1).float().cpu().numpy()[:, :, order],
                "rope_k": k_rot[0].transpose(0, 1).float().cpu().numpy()[:, :, order],
                "attention_scores": [scores[0, :, t, :t + 1].float().cpu().numpy()
                                     for t in range(steps)],
                "softmax": [probs[0, :, t, :t + 1].float().cpu().numpy()
                            for t in range(steps)],
                "weighted_sum": weighted[0].float().cpu().numpy(),
            }
    return output


def run_cpp(exe: Path, artifact: Path, ids, dump: Path):
    dump.parent.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(
        [str(exe), str(artifact), ",".join(str(v) for v in ids), str(dump)],
        capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"C++ full-stack probe failed: {completed.stdout}\n{completed.stderr}")
    return [json.loads(line) for line in dump.read_text(encoding="utf-8").splitlines()]


def cpp_target_layers(model_int, records, layers, steps):
    by_key = {(r["site"], int(r["token"]), int(r["head"])): r
              for r in records if r["kind"] != "header"}
    cfg = model_int.config
    group = cfg.num_attention_heads // cfg.num_key_value_heads
    output = {}
    for layer_index in layers:
        prefix = f"layer{layer_index}"
        rope_q = np.empty((steps, cfg.num_attention_heads, cfg.head_dim), dtype=np.float64)
        rope_k = np.empty((steps, cfg.num_key_value_heads, cfg.head_dim), dtype=np.float64)
        scores, probs = [], []
        weighted = np.empty((steps, cfg.num_attention_heads, cfg.head_dim), dtype=np.float64)
        for t in range(steps):
            score_rows, prob_rows = [], []
            for head in range(cfg.num_attention_heads):
                qr = by_key[(f"{prefix}.q_norm", t, head)]
                q_scale = (int(qr["m_out"]), int(qr["e_out"]))
                q_real_scale = math.ldexp(float(q_scale[0]), q_scale[1])
                rope_q[t, head] = np.asarray(
                    by_key[(f"{prefix}.rope_q", t, head)]["codes"], dtype=np.float64) * q_real_scale
                kv_head = head // group
                static = model_int.composition_constants[f"{prefix}.softmax_khead{kv_head}"]
                sm_scale = intmath.carried_scale_product([q_scale, static])
                score_rows.append(np.asarray(
                    by_key[(f"{prefix}.attention_scores", t, head)]["x_int"],
                    dtype=np.float64) * math.ldexp(float(sm_scale[0]), sm_scale[1]))
                prob_rows.append(np.asarray(
                    by_key[(f"{prefix}.softmax", t, head)]["x_int"],
                    dtype=np.float64) / PROB_ONE)
                v_scale = model_int.scales.scale(f"{prefix}.v_head{kv_head}.scale")
                weighted[t, head] = np.asarray(
                    by_key[(f"{prefix}.weighted_sum", t, head)]["x_int"],
                    dtype=np.float64) * v_scale / PROB_ONE
            scores.append(np.stack(score_rows))
            probs.append(np.stack(prob_rows))
            for head in range(cfg.num_key_value_heads):
                k_scale = model_int.scales.scale(f"{prefix}.k_normed_head{head}.scale")
                rope_k[t, head] = np.asarray(
                    by_key[(f"{prefix}.rope_k", t, head)]["codes"], dtype=np.float64) * k_scale
        output[layer_index] = {
            "rope_q": rope_q, "rope_k": rope_k, "attention_scores": scores,
            "softmax": probs, "weighted_sum": weighted,
        }
    return output


def depth_readings(args, t2604, tokenizer, model_fp32, model_int) -> dict:
    specimen = t2604.selected_rows(args.corpus)[0][1]
    output = {}
    for kind in ("query", "document"):
        text = QUERY_PREFIX + specimen["text"] if kind == "query" else specimen["text"]
        ids = tokenizer(text)["input_ids"]
        float_layers = float_target_layers(model_fp32, ids, TARGET_LAYERS)
        records = run_cpp(args.cpp_exe, args.sslm_model, ids, args.cpp_dir / f"{kind}.jsonl")
        int_layers = cpp_target_layers(model_int, records, TARGET_LAYERS, len(ids))
        layer_rows = {}
        for layer in TARGET_LAYERS:
            layer_rows[str(layer)] = {
                site: cosine(int_layers[layer][site], float_layers[layer][site])
                for site in ("rope_q", "rope_k", "attention_scores", "softmax", "weighted_sum")
            }
            layer_rows[str(layer)]["score_to_softmax_cosine_drop"] = (
                layer_rows[str(layer)]["attention_scores"] - layer_rows[str(layer)]["softmax"])
        output[kind] = {
            "label": specimen["label"], "class": specimen["class"],
            "token_count": len(ids), "layers": layer_rows,
            "cpp_records": len(records) - 1,
        }
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--integer-cache", type=Path, required=True)
    parser.add_argument("--hf-model", type=Path, required=True)
    parser.add_argument("--sslm-model", type=Path, required=True)
    parser.add_argument("--cpp-exe", type=Path, required=True)
    parser.add_argument("--cpp-dir", type=Path, required=True)
    parser.add_argument("--fixed-result", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    import torch
    from transformers import AutoModel, AutoTokenizer

    started = time.perf_counter()
    t2604 = load_t2604(Path(__file__).parents[1] / "t2604" / "analyze.py")
    tokenizer = AutoTokenizer.from_pretrained(str(args.hf_model), local_files_only=True)
    model_int = load_artifact(args.integer_cache)
    model_fp32 = AutoModel.from_pretrained(
        str(args.hf_model), local_files_only=True, dtype=torch.float32,
        attn_implementation="eager").eval().to(args.device)

    fixed = json.loads(args.fixed_result.read_text(encoding="utf-8"))
    result = {
        "provenance": {
            "corpus": str(args.corpus),
            "corpus_sha256": hashlib.sha256(args.corpus.read_bytes()).hexdigest(),
            "integer_cache": str(args.integer_cache),
            "sslm_model": str(args.sslm_model),
            "sslm_sha256": hashlib.sha256(args.sslm_model.read_bytes()).hexdigest(),
            "hf_model": str(args.hf_model), "float_dtype": "torch.float32",
            "block0_population": "seven ASCII one-per-class specimens per path",
            "depth_population": "first ASCII one-per-class specimen, once per path",
        },
        "fixed_artifact_three_way": {
            kind: {
                site: fixed["aggregate"][kind][site]
                for site in ("rope_q", "rope_k", "attention_scores", "softmax", "weighted_sum",
                             "residual2")
            } | {"cpp_vs_python_integer": fixed["aggregate"][kind]["cpp_vs_python_integer"]}
            for kind in ("query", "document")
        },
        "block0_interventions": block0_campaign(
            args, t2604, tokenizer, model_fp32, model_int),
        "depth_readings": depth_readings(
            args, t2604, tokenizer, model_fp32, model_int),
        "artifact_constants": {
            "layer0.softmax.input": model_int.scales.scale("layer0.softmax.input"),
            "layer0.q_norm_gain_scale": model_int.weight_scales["layer0.q_norm.gain"][0],
            "layer0.k_norm_gain_scale": model_int.weight_scales["layer0.k_norm.gain"][0],
            "layer0.k_normed_scale": model_int.scales.scale("layer0.k_normed_head0.scale"),
            "layer0.softmax_khead_scale": math.ldexp(
                float(model_int.composition_constants["layer0.softmax_khead0"][0]),
                int(model_int.composition_constants["layer0.softmax_khead0"][1])),
        },
    }
    result["elapsed_seconds"] = time.perf_counter() - started
    args.result.parent.mkdir(parents=True, exist_ok=True)
    args.result.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "block0": {k: v["mean_cosines"] for k, v in result["block0_interventions"].items()},
        "counterfactual": {k: v["softmax_counterfactual"]
                           for k, v in result["block0_interventions"].items()},
        "depth": result["depth_readings"],
        "constants": result["artifact_constants"],
        "elapsed_seconds": result["elapsed_seconds"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

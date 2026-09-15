#!/usr/bin/env python3
"""T-2703 leaf split, input-matched local attribution, and diagnostic prices.

This tool consumes the accepted T-2703 float/floor vectors, captures the
engine's selected regions in one self-checking tracer invocation per item, and
keeps every counterfactual outside production code.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from reference_pipeline import artifact_cache, pipeline  # noqa: E402
from t2703_fidelity_localization import (  # noqa: E402
    LONG_SENTENCES, SMOKE_SENTENCES, capture_hf, cosine_loss, summarize,
)


LAYERS = (0, 3, 8, 13, 18, 22, 25, 26, 27)
PROB_SCALE = 1 << 15


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def physical(pair) -> float:
    return math.ldexp(float(pair[0]), int(pair[1]))


def chain_vector(rows, site: str, field: str = "codes", scaled: bool = True) -> np.ndarray:
    matches = [row for row in rows if row["type"] == "chain" and row["site"] == site]
    if not matches:
        raise KeyError(f"missing engine chain site {site}")
    chunks = []
    for row in matches:
        values = np.asarray(row[field], dtype=np.float64)
        if scaled:
            values = values * physical((row["m"], row["e"]))
        chunks.append(values)
    return np.concatenate(chunks)


def raw_weight(model, name: str) -> np.ndarray:
    codes = np.asarray(model.weights[name], dtype=np.float64)
    scales = np.asarray(model.weight_scales[name], dtype=np.float64)
    if scales.size == 1:
        return codes * float(scales.item())
    return codes * scales.reshape((scales.size,) + (1,) * (codes.ndim - 1))


def rmsnorm(value: np.ndarray, gain: np.ndarray, eps: float) -> np.ndarray:
    return value / math.sqrt(float(np.mean(value * value)) + eps) * gain


def silu(value: np.ndarray) -> np.ndarray:
    return value / (1.0 + np.exp(-np.clip(value, -80.0, 80.0)))


def loss(left, right) -> float:
    return cosine_loss(np.asarray(left).reshape(-1), np.asarray(right).reshape(-1))


def capture_engine(layer_trace: Path, artifact: Path, token_ids, output: Path) -> None:
    dump_dir = output / "engine-dumps"
    dump_dir.mkdir(parents=True, exist_ok=True)
    selected = ",".join(map(str, LAYERS))
    for index, ids in enumerate(token_ids):
        dump = dump_dir / f"item-{index:02d}.bin"
        sites = dump_dir / f"item-{index:02d}.sites.jsonl"
        stdout = dump_dir / f"item-{index:02d}.stdout.txt"
        command = [str(layer_trace), str(artifact), "-", "--token-ids", ",".join(map(str, ids)),
                   "--dump", str(dump), "--include-final-norm", "--site-dump", str(sites),
                   "--site-layers", selected]
        completed = subprocess.run(command, text=True, capture_output=True, check=False)
        stdout.write_text(completed.stdout + completed.stderr, encoding="utf-8")
        if completed.returncode:
            raise RuntimeError(f"engine capture failed for item {index}:\n{completed.stdout}\n"
                               f"{completed.stderr}")
        print(f"engine leaf capture {index + 1}/{len(token_ids)}", flush=True)


def read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def engine_value(rows, layer: int, head: int) -> np.ndarray:
    matching = [row for row in rows if row["type"] == "value" and
                row["layer"] == layer and row["head"] == head]
    if len(matching) != 1:
        raise ValueError(f"expected one V record for layer {layer} head {head}, got {len(matching)}")
    return np.asarray(matching[0]["codes"], dtype=np.float64)


def engine_attention(rows, layer: int):
    matching = [row for row in rows if row["type"] == "attention" and row["layer"] == layer]
    matching.sort(key=lambda row: row["head"])
    if len(matching) != 16:
        raise ValueError(f"expected 16 attention heads at layer {layer}, got {len(matching)}")
    return matching


def local_item(model, vectors, floor_sites, float_sites, rows, item: int, layer: int):
    cfg = model.config
    prefix = f"layer{layer}"
    group = cfg.num_attention_heads // cfg.num_key_value_heads
    result = {}
    global_result = {}

    engine_block_in = vectors["engine"][item, layer]
    floor_block_in = vectors["floor"][item, layer]

    def compare(name, engine_out, engine_ideal, floor_out, floor_ideal):
        result[name] = (loss(engine_out, engine_ideal), loss(floor_out, floor_ideal))

    # Attention RMSNorm.
    attn_gain = raw_weight(model, f"{prefix}.attn_norm.gain")
    engine_attn_norm = chain_vector(rows, f"{prefix}.attn_norm")
    floor_attn_norm = floor_sites[f"{prefix}.attn_norm"].reshape(-1)
    compare("attn_norm", engine_attn_norm,
            rmsnorm(engine_block_in, attn_gain, cfg.rms_norm_eps), floor_attn_norm,
            rmsnorm(floor_block_in, attn_gain, cfg.rms_norm_eps))

    # Q projection and QK norm; engine order is the artifact's adjacent-pair order.
    q_weight = raw_weight(model, f"{prefix}.q_proj")
    engine_q = chain_vector(rows, f"{prefix}.q_proj.requant")
    floor_q = floor_sites[f"{prefix}.q_proj"].reshape(-1)
    floor_q_weight = raw_weight(model, f"{prefix}.q_proj")
    order = pipeline._rope_pair_permutation(cfg.head_dim)
    inverse = np.empty_like(order)
    inverse[order] = np.arange(order.size)
    floor_q_weight = floor_q_weight.reshape(cfg.num_attention_heads, cfg.head_dim, -1)[:, inverse].reshape(
        floor_q_weight.shape)
    compare("q_proj", engine_q, q_weight @ engine_attn_norm,
            floor_q, floor_q_weight @ floor_attn_norm)
    if f"{prefix}.q_norm.gain" in model.weights:
        q_gain = raw_weight(model, f"{prefix}.q_norm.gain")
        engine_q_heads = engine_q.reshape(cfg.num_attention_heads, cfg.head_dim)
        engine_qnorm = chain_vector(rows, f"{prefix}.q_norm").reshape(
            cfg.num_attention_heads, cfg.head_dim)
        floor_q_heads = floor_q.reshape(cfg.num_attention_heads, cfg.head_dim)
        floor_qnorm = floor_sites[f"{prefix}.q_norm"]
        q_gain_hf = q_gain[inverse]
        engine_ideal = np.stack([rmsnorm(row, q_gain, cfg.rms_norm_eps) for row in engine_q_heads])
        floor_ideal = np.stack([rmsnorm(row, q_gain_hf, cfg.rms_norm_eps) for row in floor_q_heads])
        compare("q_norm", engine_qnorm, engine_ideal, floor_qnorm, floor_ideal)

    # V projection on the current token.  This direct observation is reached
    # only after the accumulator and fold checks below prove locally exact.
    v_weight = raw_weight(model, f"{prefix}.v_proj")
    engine_v = np.stack([engine_value(rows, layer, head)[-1]
                         * physical(model.kv_landing_scales[f"{prefix}.v_head{head}"])
                         for head in range(cfg.num_key_value_heads)])
    floor_v = floor_sites[f"{prefix}.v_proj"]
    v_ideal_engine = (v_weight @ engine_attn_norm).reshape(cfg.num_key_value_heads, cfg.head_dim)
    v_ideal_floor = (v_weight @ floor_attn_norm).reshape(cfg.num_key_value_heads, cfg.head_dim)
    compare("v_proj", engine_v, v_ideal_engine, floor_v, v_ideal_floor)
    global_result["v_proj"] = (loss(engine_v, float_sites[f"{prefix}.v_proj"]),
                               loss(floor_v, float_sites[f"{prefix}.v_proj"]))

    attention = engine_attention(rows, layer)
    softmax_engine, softmax_ideal = [], []
    ctx_acc, ctx_acc_ideal, ctx_wide = [], [], []
    for head, entry in enumerate(attention):
        kv_head = head // group
        q_scale = physical(([
            row for row in rows if row["type"] == "chain" and row["site"] == f"{prefix}.q_norm"
        ][head]["m"], [
            row for row in rows if row["type"] == "chain" and row["site"] == f"{prefix}.q_norm"
        ][head]["e"])) if f"{prefix}.q_norm.gain" in model.weights else physical((
            [row for row in rows if row["type"] == "chain" and row["site"] == f"{prefix}.q_proj.requant"][0]["m"],
            [row for row in rows if row["type"] == "chain" and row["site"] == f"{prefix}.q_proj.requant"][0]["e"]))
        static = physical(pipeline._qk_softmax_khead_pair(model, prefix, kv_head))
        score = np.asarray(entry["scores"], dtype=np.float64) * q_scale * static
        shifted = score - np.max(score)
        ideal_prob = np.exp(shifted) / np.exp(shifted).sum()
        engine_prob = np.asarray(entry["probs"], dtype=np.float64) / PROB_SCALE
        softmax_engine.append(engine_prob)
        softmax_ideal.append(ideal_prob)
        v_scale = physical(model.kv_landing_scales[f"{prefix}.v_head{kv_head}"])
        v_values = engine_value(rows, layer, kv_head) * v_scale
        ideal_acc = engine_prob @ v_values
        acc = np.asarray(entry["ctx_acc"], dtype=np.float64) * v_scale / PROB_SCALE
        max_v_scale = max(physical(model.kv_landing_scales[f"{prefix}.v_head{h}"])
                          for h in range(cfg.num_key_value_heads))
        wide = np.asarray(entry["ctx_wide"], dtype=np.float64) * max_v_scale / PROB_SCALE
        ctx_acc.append(acc)
        ctx_acc_ideal.append(ideal_acc)
        ctx_wide.append(wide)
    softmax_engine = np.stack(softmax_engine)
    softmax_ideal = np.stack(softmax_ideal)
    ctx_acc = np.stack(ctx_acc)
    ctx_acc_ideal = np.stack(ctx_acc_ideal)
    ctx_wide = np.stack(ctx_wide)
    floor_prob = floor_sites[f"{prefix}.softmax"]
    floor_scores = floor_sites[f"{prefix}.scores"]
    floor_softmax_ideal = np.exp(floor_scores - floor_scores.max(axis=-1, keepdims=True))
    floor_softmax_ideal /= floor_softmax_ideal.sum(axis=-1, keepdims=True)
    compare("softmax", softmax_engine, softmax_ideal, floor_prob, floor_softmax_ideal)
    floor_values = floor_sites[f"{prefix}.v_proj"]
    # The HF site dump retains only the current V projection row.  Its context
    # was already formed in float; hence the floor accumulator's local error is
    # structurally zero and the next landing comparison carries its only loss.
    compare("context_accumulate", ctx_acc, ctx_acc_ideal,
            floor_sites[f"{prefix}.context"], floor_sites[f"{prefix}.context"])
    compare("ctx_fold", ctx_wide, ctx_acc,
            floor_sites[f"{prefix}.context"], floor_sites[f"{prefix}.context"])
    engine_ctx = chain_vector(rows, f"{prefix}.attn_ctx").reshape(
        cfg.num_attention_heads, cfg.head_dim)
    floor_ctx = floor_sites[f"{prefix}.attn_ctx"]
    compare("attn_ctx_landing", engine_ctx, ctx_wide, floor_ctx,
            floor_sites[f"{prefix}.context"])
    global_result["softmax"] = (loss(softmax_engine, float_sites[f"{prefix}.softmax"]),
                                loss(floor_prob, float_sites[f"{prefix}.softmax"]))
    global_result["ctx_acc"] = (loss(ctx_acc, float_sites[f"{prefix}.context"]),
                                loss(floor_sites[f"{prefix}.context"],
                                     float_sites[f"{prefix}.context"]))
    global_result["ctx_wide"] = (loss(ctx_wide, float_sites[f"{prefix}.context"]),
                                 loss(floor_sites[f"{prefix}.context"],
                                      float_sites[f"{prefix}.context"]))
    global_result["attn_ctx"] = (loss(engine_ctx, float_sites[f"{prefix}.attn_ctx"]),
                                 loss(floor_ctx, float_sites[f"{prefix}.attn_ctx"]))

    # Output projection and attention residual.
    o_weight = raw_weight(model, f"{prefix}.o_proj")
    engine_o = chain_vector(rows, f"{prefix}.o_proj.requant")
    floor_o = floor_sites[f"{prefix}.o_proj"].reshape(-1)
    compare("o_proj", engine_o, o_weight @ engine_ctx.reshape(-1),
            floor_o, o_weight @ floor_ctx.reshape(-1))
    engine_attn_res = chain_vector(rows, f"{prefix}.attn_residual")
    floor_attn_res = floor_sites[f"{prefix}.attn_residual"].reshape(-1)
    compare("attn_residual", engine_attn_res, engine_block_in + engine_o,
            floor_attn_res, floor_block_in + floor_o)

    # MLP norm, projections, SwiGLU, down projection, and residual.
    mlp_gain = raw_weight(model, f"{prefix}.mlp_norm.gain")
    engine_mlp_norm = chain_vector(rows, f"{prefix}.mlp_norm")
    floor_mlp_norm = floor_sites[f"{prefix}.mlp_norm"].reshape(-1)
    compare("mlp_norm", engine_mlp_norm,
            rmsnorm(engine_attn_res, mlp_gain, cfg.rms_norm_eps), floor_mlp_norm,
            rmsnorm(floor_attn_res, mlp_gain, cfg.rms_norm_eps))
    projected = {}
    for leaf in ("gate_proj", "up_proj"):
        weight = raw_weight(model, f"{prefix}.{leaf}")
        engine_out = chain_vector(rows, f"{prefix}.{leaf}.requant")
        floor_out = floor_sites[f"{prefix}.{leaf}"].reshape(-1)
        compare(leaf, engine_out, weight @ engine_mlp_norm,
                floor_out, weight @ floor_mlp_norm)
        projected[leaf] = (engine_out, floor_out)
    engine_act = chain_vector(rows, f"{prefix}.mlp_act")
    floor_act = floor_sites[f"{prefix}.mlp_act"].reshape(-1)
    compare("mlp_act", engine_act, silu(projected["gate_proj"][0]) * projected["up_proj"][0],
            floor_act, silu(projected["gate_proj"][1]) * projected["up_proj"][1])
    down_weight = raw_weight(model, f"{prefix}.down_proj")
    engine_down = chain_vector(rows, f"{prefix}.down_proj.requant")
    floor_down = floor_sites[f"{prefix}.down_proj"].reshape(-1)
    compare("down_proj", engine_down, down_weight @ engine_act,
            floor_down, down_weight @ floor_act)
    engine_mlp_res = chain_vector(rows, f"{prefix}.mlp_residual")
    floor_mlp_res = floor_sites[f"{prefix}.mlp_residual"].reshape(-1)
    compare("mlp_residual", engine_mlp_res, engine_attn_res + engine_down,
            floor_mlp_res, floor_attn_res + floor_down)
    return result, global_result


def final_norm_item(model, vectors, rows, item: int):
    gain = raw_weight(model, "final_norm.gain")
    engine_in = vectors["engine"][item, -2]
    floor_in = vectors["floor"][item, -2]
    engine_out = chain_vector(rows, "final_norm")
    floor_out = vectors["floor"][item, -1]
    return (loss(engine_out, rmsnorm(engine_in, gain, model.config.rms_norm_eps)),
            loss(floor_out, rmsnorm(floor_in, gain, model.config.rms_norm_eps)))


def aggregate(model, baseline: Path, output: Path, item_count: int):
    vectors = np.load(baseline / "vectors.npz")
    if vectors["engine"].shape[0] != item_count:
        raise ValueError("baseline vector item count does not match capture")
    local = {layer: {} for layer in LAYERS}
    global_sites = {layer: {} for layer in LAYERS}
    final = []
    for item in range(item_count):
        floor_sites = dict(np.load(baseline / "floor-site-dumps" / f"item-{item:02d}.npz"))
        float_sites = dict(np.load(baseline / "float-site-dumps" / f"item-{item:02d}.npz"))
        rows = read_jsonl(output / "engine-dumps" / f"item-{item:02d}.sites.jsonl")
        for layer in LAYERS:
            item_local, item_global = local_item(model, vectors, floor_sites, float_sites,
                                                 rows, item, layer)
            for name, pair in item_local.items():
                local[layer].setdefault(name, [[], []])
                local[layer][name][0].append(pair[0])
                local[layer][name][1].append(pair[1])
            for name, pair in item_global.items():
                global_sites[layer].setdefault(name, [[], []])
                global_sites[layer][name][0].append(pair[0])
                global_sites[layer][name][1].append(pair[1])
        final.append(final_norm_item(model, vectors, rows, item))

    local_table = []
    region_carriers = []
    for layer in LAYERS:
        candidates = []
        for name, values in local[layer].items():
            engine_summary, floor_summary = summarize(values[0]), summarize(values[1])
            excess = engine_summary["median"] - floor_summary["median"]
            local_table.append({"region": f"layer{layer}", "operation": name,
                                "engine_local": engine_summary, "floor_local": floor_summary,
                                "median_excess": excess})
            candidates.append((excess, name))
        region_carriers.append({"region": f"layer{layer}",
                                "largest_local_excess_operation": max(candidates)[1],
                                "median_excess": max(candidates)[0]})
    final_engine = summarize([pair[0] for pair in final])
    final_floor = summarize([pair[1] for pair in final])
    local_table.append({"region": "final_norm", "operation": "rmsnorm",
                        "engine_local": final_engine, "floor_local": final_floor,
                        "median_excess": final_engine["median"] - final_floor["median"]})
    region_carriers.append({"region": "final_norm", "largest_local_excess_operation": "rmsnorm",
                            "median_excess": final_engine["median"] - final_floor["median"]})
    global_table = []
    for layer in LAYERS:
        for name, values in global_sites[layer].items():
            global_table.append({"region": f"layer{layer}", "stage": name,
                                 "engine_vs_float": summarize(values[0]),
                                 "floor_vs_float": summarize(values[1])})
    return local_table, global_table, region_carriers


def final_curve(rows: np.ndarray, floating: np.ndarray):
    losses = [loss(rows[i, -1], floating[i, -1]) for i in range(rows.shape[0])]
    stats = summarize(losses)
    cosines = 1.0 - np.asarray(losses, dtype=np.float64)
    return {"cosine_median": float(np.median(cosines)),
            "cosine_p90": float(np.quantile(cosines, 0.90)),
            "cosine_p10": float(np.quantile(cosines, 0.10)),
            "cosine_p90_loss": 1.0 - stats["p90"],
            "loss": stats}


def loss_curve(rows: np.ndarray, floating: np.ndarray):
    return [summarize([loss(rows[item, row], floating[item, row])
                       for item in range(rows.shape[0])]) for row in range(rows.shape[1])]


def run_option1_reproduction(hf_model: Path, model, token_ids, baseline: Path, output: Path):
    vectors = np.load(baseline / "vectors.npz")
    (output / "price-vectors").mkdir(parents=True, exist_ok=True)
    emulated, _ = capture_hf(hf_model, model, token_ids, floor=True,
                             emulate_attention_residual=True)
    np.savez_compressed(output / "price-vectors" / "engine-residual-emulation.npz", rows=emulated)
    engine_curve = loss_curve(vectors["engine"], vectors["floating"])
    emulated_curve = loss_curve(emulated, vectors["floating"])
    floor_curve = loss_curve(vectors["floor"], vectors["floating"])
    margins = [abs(emulated_curve[row]["median"] - engine_curve[row]["median"])
               for row in range(len(engine_curve))]
    return {
        "status": "FALSIFIED",
        "candidate_identified_operations": "attention residual reconciliation at every block",
        "emulation_scope": "diagnostic host analogue of engine residual scale reconciliation, applied to the floor",
        "engine_arithmetic_emulation": final_curve(emulated, vectors["floating"]),
        "engine_measured": final_curve(vectors["engine"], vectors["floating"]),
        "all_nonlanding_arithmetic_exact_upper_bound": final_curve(
            vectors["floor"], vectors["floating"]),
        "reproduction_margin": {
            "final_median_cosine_absolute": abs(
                final_curve(emulated, vectors["floating"])["cosine_median"] -
                final_curve(vectors["engine"], vectors["floating"])["cosine_median"]),
            "maximum_row_median_loss_absolute": max(margins),
            "median_row_median_loss_absolute": float(np.median(margins)),
        },
        "price": {"artifact_bytes_added": 0,
                  "runtime_vram_bytes_added": 2 * model.config.hidden_size,
                  "basis": "one FP16 residual reconciliation buffer; weights and artifact unchanged",
                  "disposition": "not a supported repair price because the reproduction test failed"},
        "engine_curve": engine_curve, "emulated_curve": emulated_curve, "floor_curve": floor_curve,
    }


def run_prices(hf_model: Path, model, token_ids, baseline: Path, output: Path):
    vectors = np.load(baseline / "vectors.npz")
    floating = vectors["floating"]
    prices = []
    variant_dir = output / "price-vectors"
    variant_dir.mkdir(parents=True, exist_ok=True)
    for candidate in ("v", "mlp", "down_residual"):
        rows, _ = capture_hf(hf_model, model, token_ids, floor=True, mixed_precision=candidate)
        np.savez_compressed(variant_dir / f"mixed-{candidate}.npz", rows=rows)
        if candidate == "v":
            parameters = int(np.prod(model.weights["layer27.v_proj"].shape))
            activation_delta = model.config.context_cap * model.config.num_key_value_heads * model.config.head_dim
        elif candidate == "mlp":
            parameters = sum(int(np.prod(model.weights[f"layer27.{leaf}"].shape))
                             for leaf in ("gate_proj", "up_proj", "down_proj"))
            activation_delta = 3 * model.config.intermediate_size
        else:
            parameters = int(np.prod(model.weights["layer27.down_proj"].shape))
            activation_delta = model.config.intermediate_size + 2 * model.config.hidden_size
        prices.append({"candidate": candidate, "status": "conditional floor-side sensitivity; "
                                                         "Option-1 reproduction failed",
                       **final_curve(rows, floating),
                       "artifact_bytes_added": parameters,
                       "runtime_vram_bytes_added": parameters + activation_delta,
                       "cost_basis": "FP16 replaces int8: +1 byte per promoted parameter; runtime adds "
                                     "the same resident-weight delta plus one-byte activation/cache widening"})
        print(f"mixed-precision price {candidate} complete", flush=True)
    clipping = []
    for multiplier in (1.0, 1.10, 1.25):
        rows, site_sets = capture_hf(hf_model, model, token_ids, floor=True,
                                     qkc_headroom=multiplier)
        np.savez_compressed(variant_dir / f"qkc-headroom-{multiplier:.2f}.npz", rows=rows)
        clipped = 0
        observed = 0
        max_ratio = 0.0
        for sites in site_sets:
            for layer in range(model.config.num_hidden_layers):
                raw = sites[f"layer{layer}.k_rope_pre_qkc_all"]
                source = np.asarray(model.qk_channel_peaks[f"layer{layer}"], dtype=np.float64)
                order = pipeline._rope_pair_permutation(model.config.head_dim)
                inverse = np.empty_like(order)
                inverse[order] = np.arange(order.size)
                scale = (source[:, inverse] * multiplier / 127.0)[:, None, :]
                ratio = np.abs(raw) / scale
                clipped += int(np.count_nonzero(ratio > 127.0))
                observed += int(ratio.size)
                max_ratio = max(max_ratio, float(ratio.max(initial=0.0)))
        clipping.append({"qkc_scale_multiplier": multiplier, **final_curve(rows, floating),
                         "clipped_values": clipped, "observed_values": observed,
                         "maximum_unclamped_code_magnitude": max_ratio})
        print(f"QKC headroom price {multiplier:.2f} complete", flush=True)
    return prices, clipping


def control_residual_summary(control_model, baseline: Path, capture: Path, layers):
    vectors = np.load(baseline / "vectors.npz")
    result = []
    final_pairs = []
    for item in range(vectors["engine"].shape[0]):
        floor_sites = dict(np.load(baseline / "floor-site-dumps" / f"item-{item:02d}.npz"))
        rows = read_jsonl(capture / f"item-{item:02d}.sites.jsonl")
        for layer in layers:
            prefix = f"layer{layer}"
            block_in_e = vectors["engine"][item, layer]
            block_in_f = vectors["floor"][item, layer]
            o_e = chain_vector(rows, f"{prefix}.o_proj.requant")
            o_f = floor_sites[f"{prefix}.o_proj"].reshape(-1)
            ar_e = chain_vector(rows, f"{prefix}.attn_residual")
            ar_f = floor_sites[f"{prefix}.attn_residual"].reshape(-1)
            down_e = chain_vector(rows, f"{prefix}.down_proj.requant")
            down_f = floor_sites[f"{prefix}.down_proj"].reshape(-1)
            mr_e = chain_vector(rows, f"{prefix}.mlp_residual")
            mr_f = floor_sites[f"{prefix}.mlp_residual"].reshape(-1)
            while len(result) <= layers.index(layer):
                result.append({"layer": layer, "attn_engine": [], "attn_floor": [],
                               "mlp_engine": [], "mlp_floor": []})
            slot = result[layers.index(layer)]
            slot["attn_engine"].append(loss(ar_e, block_in_e + o_e))
            slot["attn_floor"].append(loss(ar_f, block_in_f + o_f))
            slot["mlp_engine"].append(loss(mr_e, ar_e + down_e))
            slot["mlp_floor"].append(loss(mr_f, ar_f + down_f))
        final_pairs.append(final_norm_item(control_model, vectors, rows, item))
    table = []
    for slot in result:
        for kind in ("attn", "mlp"):
            engine = summarize(slot[f"{kind}_engine"])
            floor = summarize(slot[f"{kind}_floor"])
            table.append({"region": f"layer{slot['layer']}", "operation": f"{kind}_residual",
                          "engine_local": engine, "floor_local": floor,
                          "median_excess": engine["median"] - floor["median"]})
    final_engine = summarize([pair[0] for pair in final_pairs])
    final_floor = summarize([pair[1] for pair in final_pairs])
    return {"local_residual_table": table,
            "final_norm": {"engine_local": final_engine, "floor_local": final_floor,
                           "median_excess": final_engine["median"] - final_floor["median"]}}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", required=True, type=Path)
    parser.add_argument("--artifact-cache", required=True, type=Path)
    parser.add_argument("--hf-model", required=True, type=Path)
    parser.add_argument("--layer-trace", required=True, type=Path)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--skip-engine", action="store_true")
    parser.add_argument("--skip-prices", action="store_true")
    parser.add_argument("--control-artifact-cache", type=Path)
    parser.add_argument("--control-baseline", type=Path)
    parser.add_argument("--control-capture", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(args.hf_model), local_files_only=True)
    sentences = SMOKE_SENTENCES + LONG_SENTENCES
    token_ids = [tokenizer(sentence, add_special_tokens=True)["input_ids"] for sentence in sentences]
    model = artifact_cache.load_artifact(args.artifact_cache)
    if not args.skip_engine:
        capture_engine(args.layer_trace.resolve(), args.artifact.resolve(), token_ids, args.output)
    local, global_sites, carriers = aggregate(model, args.baseline, args.output, len(token_ids))
    if args.skip_prices:
        prior = json.loads((args.output / "summary.json").read_text(encoding="utf-8")) \
            if (args.output / "summary.json").is_file() else {}
        mixed = prior.get("mixed_precision_prices", [])
        clipping = prior.get("pass_c_clipping_prices", [])
        option1 = prior.get("option1_exact_operations", None)
    else:
        option1 = run_option1_reproduction(args.hf_model, model, token_ids,
                                           args.baseline, args.output)
        print("option-1 residual reproduction complete", flush=True)
        mixed, clipping = run_prices(args.hf_model, model, token_ids, args.baseline, args.output)
    control = None
    if args.control_artifact_cache and args.control_baseline and args.control_capture:
        control_model = artifact_cache.load_artifact(args.control_artifact_cache)
        control = control_residual_summary(control_model, args.control_baseline,
                                           args.control_capture, [0, 3, 8, 13, 18, 22, 23])
    result = {
        "artifact": str(args.artifact.resolve()), "artifact_sha256": sha256(args.artifact),
        "baseline_summary": str((args.baseline / "summary.json").resolve()),
        "baseline_summary_sha256": sha256(args.baseline / "summary.json"),
        "layers": list(LAYERS), "item_count": len(token_ids),
        "local_operation_table": local, "global_stage_table": global_sites,
        "region_carriers": carriers, "mixed_precision_prices": mixed,
        "option1_exact_operations": option1,
        "pass_c_clipping_prices": clipping,
        "qwen2p5_control_local": control,
        "definitions": {
            "local_engine": "engine operation output vs float operation on the engine operation's own input",
            "local_floor": "artifact-weight float operation plus that site's int8 landing, on the floor's own input",
            "context_units": "ctx_acc * per-head V scale / 32768; ctx_wide * max V-head scale / 32768",
            "cosine_p90_loss": "1 minus the p90 cosine loss (a lower-tail cosine, not p90 cosine)",
        },
    }
    summary = args.output / "summary.json"
    summary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"leaf summary={summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

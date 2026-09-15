#!/usr/bin/env python3
"""T-2703 real-vector site splicing inside one causally selected Qwen3 block."""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from reference_pipeline import artifact_cache, pipeline  # noqa: E402
from t2703_fidelity_localization import (  # noqa: E402
    LONG_SENTENCES,
    SMOKE_SENTENCES,
    CaptureController,
    install_artifact_weights,
    patched_rope,
)
from t2703_fidelity_splice import (  # noqa: E402
    cosine,
    distribution,
    final_curve,
    norm_ratio,
    run_trace,
    sha256,
)


PROB_SCALE = 32768.0
SITE_ORDER = (
    "attn_norm", "q_proj", "q_norm_rope", "v_proj", "fused_k",
    "scores", "softmax", "context", "attn_ctx", "o_proj",
    "attn_residual", "mlp_norm", "gate_proj", "up_proj", "mlp_act",
    "down_proj", "mlp_residual",
)


def physical(pair) -> float:
    return math.ldexp(float(pair[0]), int(pair[1]))


def chain(rows, name) -> tuple[np.ndarray, np.ndarray]:
    matching = [row for row in rows if row["type"] == "chain" and row["site"] == name]
    if not matching:
        raise ValueError(f"missing engine chain {name}")
    values = []
    scales = []
    for row in matching:
        scale = physical((row["m"], row["e"]))
        values.append(np.asarray(row["codes"], dtype=np.float32) * scale)
        scales.append(scale)
    return np.concatenate(values), np.asarray(scales, dtype=np.float32)


def inverse_rope_order(values, cfg, heads):
    order = pipeline._rope_pair_permutation(cfg.head_dim)
    inverse = np.empty_like(order)
    inverse[order] = np.arange(order.size)
    shaped = np.asarray(values).reshape(heads, cfg.head_dim)
    return shaped[:, inverse]


def parse_position(rows, model, layer: int):
    cfg = model.config
    prefix = f"layer{layer}"
    result = {}
    result["attn_norm"] = chain(rows, f"{prefix}.attn_norm")[0]
    result["q_proj"] = inverse_rope_order(
        chain(rows, f"{prefix}.q_proj.requant")[0], cfg, cfg.num_attention_heads).reshape(-1)
    qnorm, q_scales = chain(rows, f"{prefix}.q_norm")
    result["q_norm_rope"] = inverse_rope_order(qnorm, cfg, cfg.num_attention_heads)
    result["q_norm_scales"] = q_scales

    values = []
    for head in range(cfg.num_key_value_heads):
        matches = [row for row in rows if row["type"] == "value" and
                   row["layer"] == layer and row["head"] == head]
        if len(matches) != 1:
            raise ValueError(f"expected one V row for {prefix} head {head}")
        values.append(np.asarray(matches[0]["codes"][-1], dtype=np.float32) *
                      physical(model.kv_landing_scales[f"{prefix}.v_head{head}"]))
    result["v_proj"] = np.stack(values).reshape(-1)

    fused = [row for row in rows if row["type"] == "fused_k" and row["layer"] == layer]
    fused.sort(key=lambda row: (row["head"], row["channel"]))
    if len(fused) != cfg.num_key_value_heads * cfg.head_dim:
        raise ValueError(f"incomplete fused K for {prefix}: {len(fused)}")
    k_codes = np.asarray([np.clip(row["landing_raw"], -127, 127) for row in fused], dtype=np.float32)
    k_scales = np.asarray(model.qk_channel_peaks[prefix], dtype=np.float32) / 127.0
    result["fused_k"] = inverse_rope_order(
        k_codes.reshape(cfg.num_key_value_heads, cfg.head_dim) * k_scales,
        cfg, cfg.num_key_value_heads)

    attention = [row for row in rows if row["type"] == "attention" and row["layer"] == layer]
    attention.sort(key=lambda row: row["head"])
    if len(attention) != cfg.num_attention_heads:
        raise ValueError(f"incomplete attention rows for {prefix}: {len(attention)}")
    group = cfg.num_attention_heads // cfg.num_key_value_heads
    max_v_scale = max(physical(model.kv_landing_scales[f"{prefix}.v_head{head}"])
                      for head in range(cfg.num_key_value_heads))
    scores, probs, context = [], [], []
    for head, row in enumerate(attention):
        static = physical(pipeline._qk_softmax_khead_pair(model, prefix, head // group))
        scores.append(np.asarray(row["scores"], dtype=np.float32) * q_scales[head] * static)
        probs.append(np.asarray(row["probs"], dtype=np.float32) / PROB_SCALE)
        context.append(np.asarray(row["ctx_wide"], dtype=np.float32) * max_v_scale / PROB_SCALE)
    result["scores"] = np.stack(scores)
    result["softmax"] = np.stack(probs)
    result["context"] = np.stack(context)

    for target, source in (
        ("attn_ctx", "attn_ctx"),
        ("o_proj", "o_proj.requant"),
        ("attn_residual", "attn_residual"),
        ("mlp_norm", "mlp_norm"),
        ("gate_proj", "gate_proj.requant"),
        ("up_proj", "up_proj.requant"),
        ("mlp_act", "mlp_act"),
        ("down_proj", "down_proj.requant"),
        ("mlp_residual", "mlp_residual"),
    ):
        result[target] = chain(rows, f"{prefix}.{source}")[0]
    return result


def capture_prefix_sites(layer_trace: Path, artifact: Path, token_ids, model, layer: int,
                         output: Path):
    root = output / f"layer-{layer:02d}-prefix-sites"
    root.mkdir(parents=True, exist_ok=True)
    all_items = []
    for item, ids in enumerate(token_ids):
        cache = root / f"item-{item:02d}.npz"
        if cache.is_file():
            with np.load(cache) as saved:
                if np.array_equal(saved["token_ids"], np.asarray(ids, dtype=np.int32)):
                    all_items.append({key: saved[key] for key in saved.files if key != "token_ids"})
                    continue
        item_root = root / f"item-{item:02d}"
        item_root.mkdir(parents=True, exist_ok=True)
        positions = []
        for position in range(len(ids)):
            stem = f"position-{position:03d}"
            site_dump = item_root / f"{stem}.jsonl"
            run_trace(layer_trace, artifact, ids[:position + 1], item_root / f"{stem}.bin",
                      item_root / f"{stem}.stdout.txt", site_dump, layer)
            rows = [json.loads(line) for line in site_dump.read_text(encoding="utf-8").splitlines()]
            positions.append(parse_position(rows, model, layer))
        packed = {}
        for key in SITE_ORDER + ("q_norm_scales",):
            if key in ("scores", "softmax"):
                value = np.zeros((len(ids), model.config.num_attention_heads, len(ids)), dtype=np.float32)
                for position, record in enumerate(positions):
                    value[position, :, :position + 1] = record[key]
                packed[key] = value
            else:
                packed[key] = np.stack([record[key] for record in positions]).astype(np.float32)
        np.savez_compressed(cache, token_ids=np.asarray(ids, dtype=np.int32), **packed)
        all_items.append(packed)
        print(f"site-prefix layer={layer} item={item + 1}/{len(token_ids)} positions={len(ids)}", flush=True)
    return all_items


def capture_prefix_sites_multi(layer_trace: Path, artifact: Path, token_ids, model, layers,
                               output: Path):
    layers = tuple(layers)
    root = output / ("layers-" + "-".join(map(str, layers)) + "-prefix-sites")
    root.mkdir(parents=True, exist_ok=True)
    all_layers = {layer: [] for layer in layers}
    for item, ids in enumerate(token_ids):
        cache = root / f"item-{item:02d}.npz"
        if cache.is_file():
            with np.load(cache) as saved:
                if np.array_equal(saved["token_ids"], np.asarray(ids, dtype=np.int32)):
                    for layer in layers:
                        all_layers[layer].append({
                            key: saved[f"layer{layer}.{key}"]
                            for key in SITE_ORDER + ("q_norm_scales",)
                        })
                    continue
        item_root = root / f"item-{item:02d}"
        item_root.mkdir(parents=True, exist_ok=True)
        positions = {layer: [] for layer in layers}
        for position in range(len(ids)):
            stem = f"position-{position:03d}"
            site_dump = item_root / f"{stem}.jsonl"
            run_trace(layer_trace, artifact, ids[:position + 1], item_root / f"{stem}.bin",
                      item_root / f"{stem}.stdout.txt", site_dump, layers)
            rows = [json.loads(line) for line in site_dump.read_text(encoding="utf-8").splitlines()]
            for layer in layers:
                positions[layer].append(parse_position(rows, model, layer))
        saved_values = {"token_ids": np.asarray(ids, dtype=np.int32)}
        for layer in layers:
            packed = {}
            for key in SITE_ORDER + ("q_norm_scales",):
                if key in ("scores", "softmax"):
                    value = np.zeros((len(ids), model.config.num_attention_heads, len(ids)),
                                     dtype=np.float32)
                    for position, record in enumerate(positions[layer]):
                        value[position, :, :position + 1] = record[key]
                    packed[key] = value
                else:
                    packed[key] = np.stack(
                        [record[key] for record in positions[layer]]).astype(np.float32)
                saved_values[f"layer{layer}.{key}"] = packed[key]
            all_layers[layer].append(packed)
        np.savez_compressed(cache, **saved_values)
        print(f"site-prefix layers={layers} item={item + 1}/{len(token_ids)} positions={len(ids)}",
              flush=True)
    return all_layers


def tensor_replacement(torch, source, transform=None, after=None):
    def hook(_module, _inputs, output):
        value = torch.from_numpy(np.asarray(source, dtype=np.float32)).to(
            device=output.device, dtype=output.dtype)
        if transform is not None:
            value = transform(value)
        if after is not None:
            after(value)
        return value
    return hook


def input_replacement(torch, source, transform=None):
    def hook(_module, inputs):
        value = torch.from_numpy(np.asarray(source, dtype=np.float32)).to(
            device=inputs[0].device, dtype=inputs[0].dtype)
        if transform is not None:
            value = transform(value)
        return (value, *inputs[1:])
    return hook


@contextlib.contextmanager
def patched_attention(model, layer: int, kind: str | None, source, torch):
    if kind not in {"scores", "softmax"}:
        yield
        return
    module = sys.modules[model.__class__.__module__]
    original = module.eager_attention_forward

    def wrapped(attention, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
        key_states = module.repeat_kv(key, attention.num_key_value_groups)
        value_states = module.repeat_kv(value, attention.num_key_value_groups)
        weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
        if attention_mask is not None:
            weights = weights + attention_mask
        replacement = torch.from_numpy(np.asarray(source, dtype=np.float32)).to(
            device=weights.device, dtype=weights.dtype).permute(1, 0, 2).unsqueeze(0)
        lower = torch.tril(torch.ones(weights.shape[-2:], dtype=torch.bool, device=weights.device))
        if attention.layer_idx == layer and kind == "scores":
            weights = torch.where(lower.reshape(1, 1, *lower.shape), replacement, weights)
        weights = torch.nn.functional.softmax(weights, dim=-1, dtype=torch.float32).to(query.dtype)
        if attention.layer_idx == layer and kind == "softmax":
            weights = replacement.to(weights.dtype)
        result = torch.matmul(weights, value_states).transpose(1, 2).contiguous()
        return result, weights

    module.eager_attention_forward = wrapped
    try:
        yield
    finally:
        module.eager_attention_forward = original


def install_site_hook(torch, model, controller, layer: int, site: str, data):
    block = model.layers[layer]
    attn = block.self_attn
    mlp = block.mlp
    handles = []
    if site == "attn_norm":
        handles.append(block.input_layernorm.register_forward_hook(
            tensor_replacement(torch, data, lambda value: value.unsqueeze(0))))
    elif site == "q_proj":
        handles.append(attn.q_proj.register_forward_hook(
            tensor_replacement(torch, data, lambda value: value.unsqueeze(0))))
    elif site == "q_norm_rope":
        def scales(_value):
            scale = torch.from_numpy(np.asarray(data[1], dtype=np.float32)).to(
                device=_value.device, dtype=_value.dtype)
            controller.q_scales[layer] = scale.transpose(0, 1).reshape(
                1, scale.shape[1], scale.shape[0], 1)
        handles.append(attn.q_norm.register_forward_hook(
            tensor_replacement(torch, data[0], lambda value: value.unsqueeze(0), scales)))
    elif site == "v_proj":
        handles.append(attn.v_proj.register_forward_hook(
            tensor_replacement(torch, data, lambda value: value.unsqueeze(0))))
    elif site == "fused_k":
        # The real fused-K landing is installed by an extra RoPE wrapper in run_site_splices.
        pass
    elif site == "context":
        handles.append(attn.o_proj.register_forward_pre_hook(
            input_replacement(torch, data, lambda value: value.reshape(1, value.shape[0], -1)),
            prepend=True))
    elif site == "attn_ctx":
        handles.append(attn.o_proj.register_forward_pre_hook(
            input_replacement(torch, data, lambda value: value.unsqueeze(0))))
    elif site == "o_proj":
        handles.append(attn.o_proj.register_forward_hook(
            tensor_replacement(torch, data, lambda value: value.unsqueeze(0))))
    elif site == "attn_residual":
        handles.append(block.post_attention_layernorm.register_forward_pre_hook(
            input_replacement(torch, data, lambda value: value.unsqueeze(0))))
    elif site == "mlp_norm":
        handles.append(block.post_attention_layernorm.register_forward_hook(
            tensor_replacement(torch, data, lambda value: value.unsqueeze(0))))
    elif site == "gate_proj":
        handles.append(mlp.gate_proj.register_forward_hook(
            tensor_replacement(torch, data, lambda value: value.unsqueeze(0))))
    elif site == "up_proj":
        handles.append(mlp.up_proj.register_forward_hook(
            tensor_replacement(torch, data, lambda value: value.unsqueeze(0))))
    elif site == "mlp_act":
        handles.append(mlp.down_proj.register_forward_pre_hook(
            input_replacement(torch, data, lambda value: value.unsqueeze(0))))
    elif site == "down_proj":
        handles.append(mlp.down_proj.register_forward_hook(
            tensor_replacement(torch, data, lambda value: value.unsqueeze(0))))
    elif site == "mlp_residual":
        # Install at the consumer boundary.  A layer-output hook races the
        # model wrapper's hidden-state bookkeeping on some transformers builds;
        # the next-layer input is the observable residual-state contract used
        # by D(L), and guarantees that this endpoint is tested in the identical
        # coordinate and continuation path.
        target = model.norm if layer == len(model.layers) - 1 else model.layers[layer + 1]
        handles.append(target.register_forward_pre_hook(
            input_replacement(torch, data, lambda value: value.unsqueeze(0))))
    elif site not in {"scores", "softmax"}:
        raise ValueError(f"unsupported site {site}")
    return handles


@contextlib.contextmanager
def fused_k_override(controller, layer: int, source, torch):
    original = controller.rope

    def rope_factory(base):
        wrapped = original(base)
        def replace(q, k, cos, sin, *args, **kwargs):
            q_rot, k_rot = wrapped(q, k, cos, sin, *args, **kwargs)
            current = controller.rope_layer - 1
            if current == layer:
                value = torch.from_numpy(np.asarray(source, dtype=np.float32)).to(
                    device=k_rot.device, dtype=k_rot.dtype).permute(1, 0, 2).unsqueeze(0)
                k_rot = value
                controller.rope_k[layer] = k_rot.detach()
            return q_rot, k_rot
        return replace
    controller.rope = rope_factory
    try:
        yield
    finally:
        controller.rope = original


def run_site_splices(hf_path: Path, quantized_model, token_ids, engine_positions, sites,
                     block_vectors, floating_final, layer: int):
    import torch
    from transformers import AutoModel

    model = AutoModel.from_pretrained(str(hf_path), local_files_only=True, dtype=torch.float32,
                                      attn_implementation="eager").cuda().eval()
    install_artifact_weights(torch, model, quantized_model)
    controller = CaptureController(torch, quantized_model, True)
    controller.install(model)
    results = {}
    try:
        with torch.inference_mode():
            for mode in ("isolated", "cumulative"):
                mode_vectors = np.empty((len(SITE_ORDER), len(token_ids), quantized_model.config.hidden_size),
                                        dtype=np.float32)
                for site_index, site in enumerate(SITE_ORDER):
                    selected = (site,) if mode == "isolated" else SITE_ORDER[:site_index + 1]
                    for item, ids in enumerate(token_ids):
                        controller.reset()
                        state = engine_positions[item][:, layer, :]
                        block_input = model.layers[layer].register_forward_pre_hook(
                            input_replacement(torch, state, lambda value: value.unsqueeze(0)))
                        handles = [block_input]
                        try:
                            for selected_site in selected:
                                value = sites[item][selected_site]
                                if selected_site == "q_norm_rope":
                                    value = (value, sites[item]["q_norm_scales"])
                                handles.extend(install_site_hook(
                                    torch, model, controller, layer, selected_site, value))
                            attention_site = ("softmax" if "softmax" in selected else
                                              ("scores" if "scores" in selected else None))
                            attention_source = (sites[item][attention_site]
                                                if attention_site else None)
                            fused_source = (sites[item]["fused_k"]
                                            if "fused_k" in selected else None)
                            with fused_k_override(controller, layer, fused_source, torch) if fused_source is not None else contextlib.nullcontext():
                                with patched_rope(controller, model), patched_attention(
                                        model, layer, attention_site, attention_source, torch):
                                    input_ids = torch.tensor([ids], dtype=torch.long, device="cuda")
                                    output = model(input_ids=input_ids,
                                                   attention_mask=torch.ones_like(input_ids),
                                                   output_hidden_states=False, use_cache=False,
                                                   return_dict=True)
                                    mode_vectors[site_index, item] = output.last_hidden_state[
                                        0, -1].detach().float().cpu().numpy()
                        finally:
                            for handle in handles:
                                handle.remove()
                    print(f"site-splice mode={mode} layer={layer} site={site}", flush=True)
                results[mode] = mode_vectors
    finally:
        controller.close()
        del model
        torch.cuda.empty_cache()

    baseline = block_vectors[layer]
    block_endpoint = block_vectors[layer + 1]
    block_increment = ((1.0 - final_curve(block_endpoint, floating_final)["cosine"]["median"]) -
                       (1.0 - final_curve(baseline, floating_final)["cosine"]["median"]))
    table = []
    previous = baseline
    for index, site in enumerate(SITE_ORDER):
        isolated = final_curve(results["isolated"][index], floating_final)
        cumulative = final_curve(results["cumulative"][index], floating_final)
        previous_curve = final_curve(previous, floating_final)
        isolated_delta = ((1.0 - isolated["cosine"]["median"]) -
                          (1.0 - final_curve(baseline, floating_final)["cosine"]["median"]))
        cumulative_delta = ((1.0 - cumulative["cosine"]["median"]) -
                            (1.0 - previous_curve["cosine"]["median"]))
        table.append({
            "site": site,
            "isolated_final": isolated,
            "isolated_causal_delta_median_loss": isolated_delta,
            "cumulative_final": cumulative,
            "cumulative_increment_median_loss": cumulative_delta,
            "reverse_floor_value_final": previous_curve,
            "reverse_floor_value_increment_median_loss": -cumulative_delta,
            "isolated_share_of_block_increment": (isolated_delta / block_increment
                                                    if block_increment != 0 else None),
            "reverse_method": "with all earlier engine sites held fixed, omit this engine splice so the "
                              "floor computes this site and all downstream sites",
        })
        previous = results["cumulative"][index]
    endpoint_losses = [1.0 - cosine(results["cumulative"][-1, item], block_endpoint[item])
                       for item in range(len(token_ids))]
    endpoint_ratios = [abs(norm_ratio(results["cumulative"][-1, item], block_endpoint[item]) - 1.0)
                       for item in range(len(token_ids))]
    return results, table, {
        "block_increment_median_cosine_loss": block_increment,
        "cumulative_last_vs_D_L_cosine_loss": distribution(endpoint_losses),
        "cumulative_last_vs_D_L_norm_ratio_absolute_error": distribution(endpoint_ratios),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", required=True, type=Path)
    parser.add_argument("--artifact-cache", required=True, type=Path)
    parser.add_argument("--hf-model", required=True, type=Path)
    parser.add_argument("--layer-trace", required=True, type=Path)
    parser.add_argument("--block-summary", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--layers", required=True,
                        help="comma-separated causally dominant layer numbers")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(args.hf_model), local_files_only=True)
    token_ids = [tokenizer(sentence, add_special_tokens=True)["input_ids"]
                 for sentence in SMOKE_SENTENCES + LONG_SENTENCES]
    model = artifact_cache.load_artifact(args.artifact_cache)
    layers = tuple(int(value) for value in args.layers.split(","))
    if not layers or any(not 0 <= layer < model.config.num_hidden_layers for layer in layers):
        parser.error("--layers contains a layer outside model")

    engine_positions = []
    for item, ids in enumerate(token_ids):
        with np.load(args.block_summary.parent / "prefix-engine" / f"item-{item:02d}.npz") as saved:
            if not np.array_equal(saved["token_ids"], np.asarray(ids, dtype=np.int32)):
                raise ValueError(f"prefix token mismatch for item {item}")
            engine_positions.append(saved["states"])
    with np.load(args.block_summary.parent / "block-splice-vectors.npz") as saved:
        block_vectors = saved["block_vectors"]
        floating_final = saved["floating_final"]

    sites_by_layer = capture_prefix_sites_multi(
        args.layer_trace.resolve(), args.artifact.resolve(), token_ids, model, layers, args.output)
    layer_results = {}
    vector_values = {"floating_final": floating_final, "site_order": np.asarray(SITE_ORDER)}
    for layer in layers:
        vectors, table, reproduction = run_site_splices(
            args.hf_model, model, token_ids, engine_positions, sites_by_layer[layer], block_vectors,
            floating_final, layer)
        layer_results[str(layer)] = {"table": table, "reproduction": reproduction}
        vector_values[f"layer{layer}.isolated"] = vectors["isolated"]
        vector_values[f"layer{layer}.cumulative"] = vectors["cumulative"]
    vector_path = args.output / "dominant-site-splice-vectors.npz"
    np.savez_compressed(vector_path, **vector_values)
    summary = {
        "artifact": str(args.artifact.resolve()),
        "artifact_sha256": sha256(args.artifact),
        "layers": layers,
        "site_order": SITE_ORDER,
        "unavailable_direct_engine_sites": {
            "k_proj": "tracer exposes V cache and fused-K, not a separate K-projection row",
            "k_norm": "tracer exposes fused-K after K norm/RoPE/QKC1, not a separate K-norm row",
            "q_rope": "q_norm splice is propagated through RoPE; no separate full Q-RoPE vector is emitted",
        },
        "layer_results": layer_results,
        "evidence": {"vectors": str(vector_path.resolve())},
    }
    summary_path = args.output / "dominant-site-summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"summary={summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

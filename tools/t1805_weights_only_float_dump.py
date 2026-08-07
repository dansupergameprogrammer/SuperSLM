#!/usr/bin/env python3
"""T-1805 -- the weights-only isolation arm.

Falsifier named at `Claude/Vitruvius/t1803-fidelity-ceiling-feasibility-2026-08-07.md`
Section 9 (this project's records, not this repo's). Prices the entire weight-quantization
calibration class's ceiling in one arm: take the float (bfloat16) checkpoint, replace every
§6.1 projection weight `W` with `dequantize_weight_per_channel(quantize_weight_per_channel(W))`
-- this project's own int8 per-output-channel max-abs weight quantizer, symmetric grid, no
clipping, no calibration -- leaving everything else (activations, RMSNorm, RoPE, the forward
path itself) untouched and in float. No engine involvement anywhere in this script.

This is a NEW file in a new worktree. It REUSES, unmodified and by import, two things it does
not own:
  - `superslm_spike.quantize_weight_per_channel` / `dequantize_weight_per_channel`
    (`D:\\Wizard\\Tools\\superslm_spike\\pipeline.py:406-440`) -- the engine's own weight
    quantizer, so this arm measures what the engine actually does rather than a re-derived
    approximation of it.
  - `t1740_pooled_float_dump.capture_all_positions` / `endpoint_self_check_last_position`
    (`D:\\SuperSLM\\.worktrees\\t1777-retrieval-agreement\\tools\\t1740_pooled_float_dump.py`,
    read-only import from that worktree, not copied) -- the identical capture mechanism
    `t1777_pooled_float_dump_batch.py` itself reuses, so the dump this script writes is
    byte-layout-identical to T-1777's own float dumps and readable by its unmodified report
    tool.

Two cheap checkpoint statistics are computed in the same pass, before any weight is
overwritten, over exactly the same 196 tensors this arm degrades (28 layers x 7 projections:
q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj):
  - per-tensor and per-channel relative reconstruction error
    ||W - dequant(quant(W))|| / ||W||
  - per-channel ratio of the 99.9th percentile of |W| to the channel's own max
Written to <out-dir>/checkpoint_stats.json.

Usage
-----
    python tools\\t1805_weights_only_float_dump.py --docs <path\\to\\t1777\\docs.jsonl> \\
        --out-dir out\\t1805_weights_only \\
        --t1777-tools D:\\SuperSLM\\.worktrees\\t1777-retrieval-agreement\\tools
"""
from __future__ import annotations

import argparse
import json
import struct
import sys
import time
from pathlib import Path

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

_PROJECTIONS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
_ATTN_PROJ = ("q_proj", "k_proj", "v_proj", "o_proj")
_MLP_PROJ = ("gate_proj", "up_proj", "down_proj")

SYSTEM_PROMPT = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."


def _iter_projection_modules(model):
    """Yields (name, nn.Linear module) for every §6.1 projection in every decoder layer.

    Name is "layer{i}.{proj}", matching `superslm_spike.pipeline._weight_shapes`'s own
    naming so the two are directly comparable by eye.
    """
    n_layers = model.config.num_hidden_layers
    for i in range(n_layers):
        layer = model.model.layers[i]
        for proj in _ATTN_PROJ:
            yield f"layer{i}.{proj}", getattr(layer.self_attn, proj)
        for proj in _MLP_PROJ:
            yield f"layer{i}.{proj}", getattr(layer.mlp, proj)


def degrade_weights_and_measure(model, quantize_weight_per_channel, dequantize_weight_per_channel):
    """Round-trips every §6.1 projection weight through the engine's own int8 per-output-channel
    quantizer, in place, and returns the per-tensor / per-channel checkpoint statistics computed
    from the SAME pass, before the tensor under inspection is overwritten.

    output_axis=0: an nn.Linear weight is [out_features, in_features]; the engine's own
    `_PROJECTION_OUTPUT_AXIS = 0` scales per OUTPUT channel, i.e. per row of this layout --
    `superslm_spike.pipeline.py:204,1229-1230` confirms both the axis and that these seven
    projection names are exactly the tensors quantized through this function (embed_tokens,
    lm_head, and RMSNorm gains take a different, single-scale-per-tensor path and are not
    touched here).
    """
    import numpy as np
    import torch

    n_touched = 0
    stats = {}  # name -> {"rel_l2_tensor": float, "rel_l2_channel": [...], "pct999_over_max_channel": [...]}

    for name, module in _iter_projection_modules(model):
        w = module.weight.detach()
        orig_dtype = w.dtype
        orig_device = w.device
        w_np = w.to(torch.float32).cpu().numpy().astype(np.float64)

        codes, scales = quantize_weight_per_channel(w_np, output_axis=0)
        deq = dequantize_weight_per_channel(codes, scales, output_axis=0)

        # --- checkpoint statistics, computed on the original vs. round-tripped array,
        # before the tensor is overwritten. ---
        diff = w_np - deq
        rel_l2_tensor = float(np.linalg.norm(diff) / np.linalg.norm(w_np)) if np.linalg.norm(w_np) > 0 else 0.0

        # Per-output-channel (per row, axis 0): ||row_diff|| / ||row_orig||, and the
        # 99.9th-percentile-of-|W| / max ratio, per row.
        row_diff_norm = np.linalg.norm(diff, axis=1)
        row_orig_norm = np.linalg.norm(w_np, axis=1)
        rel_l2_channel = np.where(row_orig_norm > 0, row_diff_norm / np.where(row_orig_norm > 0, row_orig_norm, 1.0), 0.0)

        abs_w = np.abs(w_np)
        row_max = abs_w.max(axis=1)
        row_p999 = np.percentile(abs_w, 99.9, axis=1)
        pct999_over_max = np.where(row_max > 0, row_p999 / np.where(row_max > 0, row_max, 1.0), 0.0)

        stats[name] = {
            "shape": list(w_np.shape),
            "rel_l2_tensor": rel_l2_tensor,
            "rel_l2_channel_min": float(rel_l2_channel.min()),
            "rel_l2_channel_median": float(np.median(rel_l2_channel)),
            "rel_l2_channel_max": float(rel_l2_channel.max()),
            "pct999_over_max_channel_min": float(pct999_over_max.min()),
            "pct999_over_max_channel_median": float(np.median(pct999_over_max)),
            "pct999_over_max_channel_max": float(pct999_over_max.max()),
        }

        # Overwrite in place, cast back to the checkpoint's own compute dtype.
        deq_t = torch.from_numpy(deq.astype(np.float32)).to(dtype=orig_dtype, device=orig_device)
        with torch.no_grad():
            module.weight.copy_(deq_t)

        n_touched += 1

    assert n_touched == 196, (
        f"expected 196 projection tensors (28 layers x 7 projections), touched {n_touched} -- "
        "the model's layer count or projection naming does not match the assumption this "
        "script's scope was verified against"
    )
    return stats


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--docs", required=True, help="T-1777's own docs.jsonl (read-only)")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--model", default=None, help="override the default checkpoint path")
    parser.add_argument("--system", default=SYSTEM_PROMPT)
    parser.add_argument(
        "--t1777-tools", required=True,
        help=r"path to T-1777's tools\ directory, e.g. "
             r"D:\SuperSLM\.worktrees\t1777-retrieval-agreement\tools -- imported read-only, "
             r"never copied or modified",
    )
    parser.add_argument(
        "--spike-root", default=r"D:\Wizard\Tools",
        help="parent directory of the superslm_spike package (the engine's own weight "
             "quantizer lives at superslm_spike.pipeline) -- imported read-only",
    )
    args = parser.parse_args(argv)

    sys.path.insert(0, str(Path(args.t1777_tools).resolve()))
    sys.path.insert(0, str(Path(args.spike_root).resolve()))
    import t1740_pooled_float_dump as fd  # noqa: E402  (T-1777's own reused module, read-only)
    from superslm_spike.pipeline import (  # noqa: E402
        quantize_weight_per_channel,
        dequantize_weight_per_channel,
    )

    docs = []
    with open(args.docs, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            docs.append(json.loads(line))

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_arg = args.model if args.model is not None else str(fd.DEFAULT_MODEL)
    model_path = fd._resolve_default_model(Path(model_arg))
    t0 = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(str(model_path), local_files_only=True, torch_dtype="auto")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    print(f"model+tokenizer load: {time.perf_counter() - t0:.2f}s device={device} documents={len(docs)} "
          f"resolved_model_dtype={model.dtype}", flush=True)

    t_deg = time.perf_counter()
    stats = degrade_weights_and_measure(model, quantize_weight_per_channel, dequantize_weight_per_channel)
    print(f"weight degradation: 196/196 §6.1 projection tensors round-tripped through the engine's "
          f"own int8 per-output-channel quantizer in {time.perf_counter() - t_deg:.2f}s", flush=True)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "checkpoint_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print(f"checkpoint_stats written: {out_dir / 'checkpoint_stats.json'}", flush=True)

    n_ok = 0
    n_failed = 0
    t_start = time.perf_counter()
    for i, doc in enumerate(docs):
        label = doc["label"]
        text = doc["text"]
        t_doc = time.perf_counter()
        messages = [{"role": "system", "content": args.system}, {"role": "user", "content": text}]
        prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        templated = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
        input_ids = templated["input_ids"].to(device)
        n_positions = input_ids.shape[1]

        try:
            captured = fd.capture_all_positions(model, input_ids, device)
            fd.endpoint_self_check_last_position(model, captured, input_ids, device)
        except AssertionError as e:
            print(f"FAILED label={label}: {e}", flush=True)
            n_failed += 1
            continue

        n_layers = model.config.num_hidden_layers
        hidden_size = model.config.hidden_size
        fingerprint = fd.fnv1a64(prompt_text)
        dump_path = out_dir / f"{label}.float.bin"
        with open(dump_path, "wb") as fbin:
            fbin.write(struct.pack("<QQQQ", n_positions, n_layers + 1, hidden_size, fingerprint))
            for pos in range(n_positions):
                for idx in range(n_layers + 1):
                    fbin.write(captured[idx][pos].cpu().numpy().astype("float32").tobytes())
        del captured
        if device == "cuda":
            torch.cuda.empty_cache()
        n_ok += 1
        dt = time.perf_counter() - t_doc
        if (i + 1) % 10 == 0 or i == 0:
            elapsed = time.perf_counter() - t_start
            print(f"  [{i+1}/{len(docs)}] label={label} n_pos={n_positions} this_doc={dt:.2f}s "
                  f"elapsed={elapsed:.1f}s avg={elapsed/(i+1):.2f}s/doc", flush=True)

    total = time.perf_counter() - t_start
    print(f"batch_done: {n_ok} ok, {n_failed} failed (of {len(docs)}), capture_total={total:.1f}s "
          f"avg={total/max(1,len(docs)):.2f}s/doc", flush=True)
    return 0 if n_failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

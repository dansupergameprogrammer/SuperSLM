#!/usr/bin/env python3
"""T-1797 E0: float32-verified residual-stream reference at EVERY prompt position.

A minimal extension of tools/t1795_float_residual_reference.py (branch
claude/t1795-residual-drift@9eef187, read in full; its precision discipline reused
UNCHANGED: explicit dtype=torch.float32, attn_implementation="eager", token-at-a-time
incremental forward with an explicit DynamicCache, precision verified by execution at
capture time). The single change: the residual states (embedding output + each of the 28
decoder layers' committed output) are captured at EVERY position t, not only the last,
and written as raw float32 binary [position][state 0..28][hidden] with a .meta sidecar --
matching tools/t1797_multipos_probe.cpp's own engine-side layout for direct comparison.

The argmax cross-check against the model's own standard forward is retained unchanged.
Offline only (local_files_only=True).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

DEFAULT_MODEL = Path(
    r"D:\hf_cache\hub\models--Qwen--Qwen2.5-1.5B-Instruct\snapshots"
    r"\989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
)


def recompute_residual_stream_all_positions(model, input_ids, device):
    """Identical composition to t1795_float_residual_reference.recompute_residual_stream,
    with the capture predicate changed from `is_last` to every position."""
    from transformers import DynamicCache

    from transformers.masking_utils import create_causal_mask

    import torch

    n_layers = model.config.num_hidden_layers
    per_position: list = []  # list over t of list over 29 states of 1536-d float64 tensors

    cache = DynamicCache()
    with torch.no_grad():
        for t in range(input_ids.shape[1]):
            step_ids = input_ids[:, t : t + 1]
            inputs_embeds = model.model.embed_tokens(step_ids)
            position_ids = torch.tensor([[t]], device=device, dtype=torch.long)
            position_embeddings = model.model.rotary_emb(inputs_embeds, position_ids)
            cos, sin = position_embeddings

            mask_kwargs = {
                "config": model.config,
                "inputs_embeds": inputs_embeds,
                "attention_mask": None,
                "past_key_values": cache,
                "position_ids": position_ids,
            }
            causal_mask_full = create_causal_mask(**mask_kwargs)

            states_t: list = []
            hidden_states = inputs_embeds
            states_t.append(hidden_states.detach()[0, 0].double().clone())

            for i, layer in enumerate(model.model.layers[:n_layers]):
                attn = layer.self_attn
                normed = layer.input_layernorm(hidden_states)

                head_dim = attn.head_dim
                input_shape = normed.shape[:-1]
                hidden_shape = (*input_shape, -1, head_dim)

                from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb, repeat_kv

                q_lin = attn.q_proj(normed)
                k_lin = attn.k_proj(normed)
                v_lin = attn.v_proj(normed)

                query_states = q_lin.view(hidden_shape).transpose(1, 2)
                key_states = k_lin.view(hidden_shape).transpose(1, 2)
                value_states = v_lin.view(hidden_shape).transpose(1, 2)

                query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
                key_states_full, value_states_full = cache.update(key_states, value_states, i)

                num_kv_groups = model.config.num_attention_heads // model.config.num_key_value_heads
                key_rep = repeat_kv(key_states_full, num_kv_groups)
                value_rep = repeat_kv(value_states_full, num_kv_groups)
                scaling = head_dim**-0.5
                scores = torch.matmul(query_states, key_rep.transpose(2, 3)) * scaling
                if causal_mask_full is not None:
                    scores = scores + causal_mask_full
                weights = torch.nn.functional.softmax(scores, dim=-1, dtype=torch.float32).to(
                    query_states.dtype
                )
                attn_out = torch.matmul(weights, value_rep).transpose(1, 2).contiguous()
                attn_out = attn_out.reshape(*input_shape, -1)
                attn_out = attn.o_proj(attn_out)

                residual = hidden_states
                hidden_states = residual + attn_out
                residual = hidden_states
                normed2 = layer.post_attention_layernorm(hidden_states)
                hidden_states = residual + layer.mlp(normed2)

                states_t.append(hidden_states.detach()[0, 0].double().clone())

            per_position.append(states_t)

    return per_position


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prompt", help="the user prompt text")
    parser.add_argument(
        "--system",
        default="You are Qwen, created by Alibaba Cloud. You are a helpful assistant.",
    )
    parser.add_argument("--model", default=str(DEFAULT_MODEL))
    parser.add_argument("--dump", required=True, help="path to write the raw float32 dump (.bin)")
    args = parser.parse_args(argv)

    model_path = Path(args.model)
    snaps = model_path / "snapshots"
    if snaps.is_dir():
        entries = [d for d in snaps.iterdir() if d.is_dir()]
        if len(entries) == 1:
            model_path = entries[0]
    if not model_path.exists():
        raise SystemExit(f"model path does not exist: {model_path}")

    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path), local_files_only=True, dtype=torch.float32, attn_implementation="eager"
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    if getattr(model.config, "_attn_implementation", None) != "eager":
        raise SystemExit("attn_implementation did not take effect")

    # Precision verified by execution (StandardsDocument.md 5.4; D-SLM1212-1225's lesson).
    param_dtypes = {p.dtype for p in model.parameters()}
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
    print(f"param dtypes: {param_dtypes}")
    print(f"model.dtype: {model.dtype}")
    print(f"config._attn_implementation: {model.config._attn_implementation}")
    if device == "cuda":
        print(f"allow_tf32 matmul: {torch.backends.cuda.matmul.allow_tf32}")
        print(f"float32 matmul precision: {torch.get_float32_matmul_precision()}")
    if param_dtypes != {torch.float32}:
        raise SystemExit(f"FAILED: model parameters are not genuine float32: {param_dtypes}")

    messages = [
        {"role": "system", "content": args.system},
        {"role": "user", "content": args.prompt},
    ]
    templated = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
    input_ids = templated["input_ids"].to(device) if hasattr(templated, "to") else templated.to(device)
    prompt_len = input_ids.shape[1]

    per_position = recompute_residual_stream_all_positions(model, input_ids, device)
    n_layers = model.config.num_hidden_layers
    expected_states = n_layers + 1
    if len(per_position) != prompt_len or any(len(s) != expected_states for s in per_position):
        raise SystemExit("FAILED: capture shape mismatch")
    hidden_size = per_position[0][0].shape[0]
    print(f"captured: {prompt_len} positions x {expected_states} states x {hidden_size}")

    # Retained self-check: the manual composition's final state selects the same next-token
    # argmax as the model's own standard forward.
    with torch.no_grad():
        final_state = per_position[-1][-1].float().to(device).unsqueeze(0)
        manual_logits = model.lm_head(model.model.norm(final_state))
        manual_argmax = int(manual_logits.argmax(dim=-1).item())
        ref_out = model(input_ids=input_ids, use_cache=False)
        ref_argmax = int(ref_out.logits[0, -1].argmax(dim=-1).item())
    print(f"argmax_cross_check: manual={manual_argmax} standard={ref_argmax} "
          f"match={manual_argmax == ref_argmax}")
    if manual_argmax != ref_argmax:
        raise AssertionError("manual composition diverged from the model's own forward")

    arr = np.zeros((prompt_len, expected_states, hidden_size), dtype=np.float32)
    for t, states_t in enumerate(per_position):
        for s, vec in enumerate(states_t):
            arr[t, s, :] = vec.cpu().numpy().astype(np.float32)
    dump_path = Path(args.dump)
    dump_path.parent.mkdir(parents=True, exist_ok=True)
    arr.tofile(dump_path)
    with open(str(dump_path).replace(".bin", ".meta"), "w", encoding="ascii") as f:
        f.write(f"positions {prompt_len}\nstates {expected_states}\nhidden {hidden_size}\n"
                f"dtype float32-le\nlayout pos,state,hidden\n")
    print(f"dump written: {dump_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

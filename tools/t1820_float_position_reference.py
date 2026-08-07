#!/usr/bin/env python3
"""T-1820: float32-verified residual-stream reference at EVERY prompt position.

This is tools/t1795_float_residual_reference.py (branch claude/t1795-residual-drift@9eef187,
read in full) with one change: `recompute_residual_stream` records the 29 residual states at
every step of the token-at-a-time incremental forward, not only the last. Its precision
discipline, its manual per-layer composition (independent of `eager_attention_forward` below
`apply_rotary_pos_emb`/`repeat_kv`), its DynamicCache stepping, and its next-token-argmax
cross-check are carried over unmodified and re-verified by execution every run.

Output: a .npy array of shape (num_positions, 29, hidden_size), float32, plus a .tokens.npy
of the prompt token ids -- so the engine dump's own token ids can be checked to agree.

Offline only (local_files_only=True); never touches the network.
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


def _resolve_default_model(p: Path) -> Path:
    snaps = p / "snapshots"
    if snaps.is_dir():
        entries = [d for d in snaps.iterdir() if d.is_dir()]
        if len(entries) == 1:
            return entries[0]
        if len(entries) > 1:
            raise SystemExit(f"{p} has {len(entries)} snapshots; pass --model with an explicit one")
    return p


def recompute_residual_stream(model, input_ids, device):
    """T-1795's own `recompute_residual_stream`, with the `is_last` gate removed so every
    position's 29 states are recorded. The forward composition itself is unchanged."""
    from transformers import DynamicCache
    from transformers.masking_utils import create_causal_mask
    from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb, repeat_kv

    import torch

    n_layers = model.config.num_hidden_layers
    per_position: list[list] = []

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

            states = []
            hidden_states = inputs_embeds
            states.append(hidden_states.detach()[0, 0].float().clone())

            for i, layer in enumerate(model.model.layers[:n_layers]):
                attn = layer.self_attn
                normed = layer.input_layernorm(hidden_states)

                head_dim = attn.head_dim
                input_shape = normed.shape[:-1]
                hidden_shape = (*input_shape, -1, head_dim)

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

                states.append(hidden_states.detach()[0, 0].float().clone())

            per_position.append(states)

    return per_position


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("prompt", help="the user prompt text")
    parser.add_argument(
        "--system",
        default="You are Qwen, created by Alibaba Cloud. You are a helpful assistant.",
        help="system prompt (must match the .sslm side's manually-built prompt exactly)",
    )
    parser.add_argument("--model", default=str(DEFAULT_MODEL))
    parser.add_argument("--dump", required=True, help="path to write the (T,29,H) float32 .npy")
    args = parser.parse_args(argv)

    model_path = _resolve_default_model(Path(args.model))
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
        raise SystemExit(
            f"attn_implementation did not take effect: {getattr(model.config, '_attn_implementation', None)!r}"
        )

    # Precision verified by execution, per StandardsDocument.md 5.4 -- never read off the
    # dtype= argument's documented semantics.
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

    per_position = recompute_residual_stream(model, input_ids, device)
    n_layers = model.config.num_hidden_layers
    if len(per_position) != prompt_len:
        raise SystemExit(f"FAILED: expected {prompt_len} positions, got {len(per_position)}")
    for t, states in enumerate(per_position):
        if len(states) != n_layers + 1:
            raise SystemExit(f"FAILED: position {t} carries {len(states)} states, expected {n_layers+1}")

    # T-1795's own end-to-end check on the manual composition, unchanged: the last position's
    # final state must select the same next-token argmax as the model's own standard forward.
    with torch.no_grad():
        final_normed = model.model.norm(per_position[-1][-1].to(device).unsqueeze(0))
        manual_argmax = int(model.lm_head(final_normed).argmax(dim=-1).item())
        ref_out = model(input_ids=input_ids, use_cache=False)
        ref_argmax = int(ref_out.logits[0, -1].argmax(dim=-1).item())
    print(f"argmax_cross_check: manual={manual_argmax} standard={ref_argmax} "
          f"match={manual_argmax == ref_argmax}")
    if manual_argmax != ref_argmax:
        raise AssertionError("manual per-layer composition diverges from the model's own forward")

    arr = np.stack([np.stack([s.cpu().numpy() for s in states]) for states in per_position]).astype(
        np.float32
    )
    dump_path = Path(args.dump)
    dump_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(dump_path, arr)
    np.save(dump_path.with_suffix(".tokens.npy"), input_ids[0].cpu().numpy().astype(np.int32))
    print(f"float dump written: {dump_path} shape={arr.shape} prompt_tokens={prompt_len}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

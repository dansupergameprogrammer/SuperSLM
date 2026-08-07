#!/usr/bin/env python3
"""T-1795: dump the float model's own recursively-computed attention-branch output
(`attn_out`, pre-residual-add) at the 6 checkpoint layers, last prompt token only -- the
comparand tools/t1795_layer_substitution.py uses for attention's OWN branch (a different
population from the isolated positionwise sub-blocks: this is the float model's own
upstream-consistent input at that layer, not the engine's drifted one, named as such at
the call site per StandardsDocument.md 5.4).

Reuses tools/t1795_float_residual_reference.py's own per-layer composition unchanged
(same precision discipline, re-verified here), extended only to also capture `attn_out`
before it is added to the residual.
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
CHECKPOINT_LAYERS = (0, 1, 2, 9, 18, 27)


def _resolve_default_model(p: Path) -> Path:
    snaps = p / "snapshots"
    if snaps.is_dir():
        entries = [d for d in snaps.iterdir() if d.is_dir()]
        if len(entries) == 1:
            return entries[0]
    return p


def recompute_attn_out(model, input_ids, device):
    from transformers import DynamicCache
    from transformers.masking_utils import create_causal_mask
    from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb, repeat_kv

    import torch

    n_layers = model.config.num_hidden_layers
    captured = {}

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
            is_last = t == input_ids.shape[1] - 1

            hidden_states = inputs_embeds
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
                weights = torch.nn.functional.softmax(scores, dim=-1, dtype=torch.float32).to(query_states.dtype)
                attn_out = torch.matmul(weights, value_rep).transpose(1, 2).contiguous()
                attn_out = attn_out.reshape(*input_shape, -1)
                attn_out = attn.o_proj(attn_out)

                if is_last and i in CHECKPOINT_LAYERS:
                    captured[i] = attn_out.detach()[0, 0].double().clone()

                residual = hidden_states
                hidden_states = residual + attn_out
                residual = hidden_states
                normed2 = layer.post_attention_layernorm(hidden_states)
                hidden_states = residual + layer.mlp(normed2)

    return captured


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prompt")
    parser.add_argument("--system", default="You are Qwen, created by Alibaba Cloud. You are a helpful assistant.")
    parser.add_argument("--model", default=str(DEFAULT_MODEL))
    parser.add_argument("--dump", required=True)
    args = parser.parse_args(argv)

    model_path = _resolve_default_model(Path(args.model))
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path), local_files_only=True, dtype=torch.float32, attn_implementation="eager"
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")

    messages = [{"role": "system", "content": args.system}, {"role": "user", "content": args.prompt}]
    templated = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
    input_ids = templated.to(device) if not hasattr(templated, "input_ids") else templated["input_ids"].to(device)

    captured = recompute_attn_out(model, input_ids, device)
    dump_path = Path(args.dump)
    dump_path.parent.mkdir(parents=True, exist_ok=True)
    with open(dump_path, "w", encoding="ascii") as f:
        f.write(f"{len(captured)}\n")
        for layer in CHECKPOINT_LAYERS:
            vals = captured[layer].tolist()
            f.write(f"layer {layer} " + str(len(vals)) + "\n")
            f.write(f"{len(vals)} " + " ".join(f"{v!r}" for v in vals) + "\n")
    print(f"attn_out dump written: {dump_path} ({len(captured)} layers)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

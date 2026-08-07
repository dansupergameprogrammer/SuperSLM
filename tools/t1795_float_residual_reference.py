#!/usr/bin/env python3
"""T-1795: float32-verified residual-stream reference, independent of the engine's integer
construction -- the last prompt token's own residual hidden state entering the embedding and
leaving each of the 28 decoder layers (29 states), for comparison against the engine's real
committed residual stream (tools/t1795_residual_probe.cpp's own "*_residual.txt" dump).

Built directly on T-1778's own proven float instrument
(tools/t1778_float_attn_reference.py, branch claude/t1778-float-attn-reference@5df48fc, read
in full before reuse) -- this script reuses its precision discipline UNCHANGED (explicit
dtype=torch.float32, attn_implementation="eager", token-at-a-time incremental forward with an
explicit DynamicCache, verified by execution rather than asserted per StandardsDocument.md
5.4) and its `oracle_recompute_attn`'s own manual per-layer composition (independent of
`eager_attention_forward`, sharing no code with the model's in-band path below
`apply_rotary_pos_emb`/`repeat_kv`) -- extended only to capture the residual stream itself
(`hidden_states` entering the embed and leaving each layer) rather than attention weights.
Every property T-1778 SS4-SS6 established (independence from the engine's own int8
construction -- zero imports of superslm/intmath/dynamic_engine/the vendored spike reference;
genuine float32 compute with CUDA TF32 confirmed disabled) applies unchanged; both are
re-verified by direct execution below rather than inherited by assertion.

Offline only (local_files_only=True); never touches the network.

Dump format: 29 lines, "<hidden_size> v0 v1 ... v(hidden_size-1)" (matches
t1795_residual_probe.cpp's own WriteVec convention for easy cross-language diffing), in
order: embedding output, then each of the 28 decoder layers' own committed output, all at the
LAST prompt token's own row.
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
    """Independent, direct-composition per-layer forward -- identical structure to T-1778's
    own `oracle_recompute_attn`, extended to record `hidden_states` at every layer boundary
    for the LAST step only (matching the engine's own "last prompt token" capture)."""
    from transformers import DynamicCache
    from transformers.masking_utils import create_causal_mask

    import torch

    n_layers = model.config.num_hidden_layers
    residual_states: list = []  # filled only on the final step

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
            if is_last:
                residual_states.append(hidden_states.detach()[0, 0].double().clone())

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

                if is_last:
                    residual_states.append(hidden_states.detach()[0, 0].double().clone())

    return residual_states


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("prompt", help="the user prompt text")
    parser.add_argument(
        "--system",
        default="You are Qwen, created by Alibaba Cloud. You are a helpful assistant.",
        help="system prompt (should match the .sslm side's manually-built prompt exactly)",
    )
    parser.add_argument("--model", default=str(DEFAULT_MODEL), help="path to a local HF checkpoint directory")
    parser.add_argument("--dump", required=True, help="path to write the 29-state residual dump")
    args = parser.parse_args(argv)

    model_path = _resolve_default_model(Path(args.model))
    if not model_path.exists():
        raise SystemExit(f"model path does not exist: {model_path}")

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

    # Precision verified by execution, per StandardsDocument.md 5.4 -- not read from the
    # dtype=torch.float32 argument's documented semantics (D-SLM1212-1225's own finding: this
    # project's other float instruments silently ran bfloat16 while asserting float32).
    param_dtypes = {p.dtype for p in model.parameters()}
    allow_tf32_before = torch.backends.cuda.matmul.allow_tf32 if device == "cuda" else None
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
    print(f"param dtypes: {param_dtypes}")
    print(f"model.dtype: {model.dtype}")
    print(f"config._attn_implementation: {model.config._attn_implementation}")
    print(f"q_proj weight dtype layer0: {model.model.layers[0].self_attn.q_proj.weight.dtype}")
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

    residual_states = recompute_residual_stream(model, input_ids, device)
    n_layers = model.config.num_hidden_layers
    expected_states = n_layers + 1
    if len(residual_states) != expected_states:
        raise SystemExit(
            f"FAILED: expected {expected_states} residual states (embed + {n_layers} layers), "
            f"got {len(residual_states)}"
        )
    hidden_size = residual_states[0].shape[0]
    print(f"residual states captured: {len(residual_states)} (embed + {n_layers} layers), "
          f"hidden_size={hidden_size}, prompt_tokens={prompt_len}")

    # A second, independently-composed check that this manual per-layer composition really is
    # this model's forward: the standard (non-hooked) `model(...)` full-prompt logits and the
    # last state's own logits via `model.lm_head`/`model.model.norm` should select the same
    # argmax next token as a full ordinary generate call -- a coarse but real end-to-end check
    # that the manual per-layer recomposition has not silently diverged from the model's own
    # forward.
    with torch.no_grad():
        final_normed = model.model.norm(residual_states[-1].float().to(device).unsqueeze(0))
        manual_logits = model.lm_head(final_normed)
        manual_argmax = int(manual_logits.argmax(dim=-1).item())
        ref_out = model(input_ids=input_ids, use_cache=False)
        ref_argmax = int(ref_out.logits[0, -1].argmax(dim=-1).item())
    print(f"argmax_cross_check: manual composition argmax={manual_argmax}, "
          f"standard forward argmax={ref_argmax}, match={manual_argmax == ref_argmax}")
    if manual_argmax != ref_argmax:
        raise AssertionError(
            "manual per-layer composition's own final-layer state disagrees with the standard "
            "forward's own next-token argmax -- the manual composition has diverged from the "
            "model's real forward"
        )

    dump_path = Path(args.dump)
    dump_path.parent.mkdir(parents=True, exist_ok=True)
    with open(dump_path, "w", encoding="ascii") as f:
        for state in residual_states:
            vals = state.tolist()
            f.write(f"{len(vals)} " + " ".join(f"{v!r}" for v in vals) + "\n")
    print(f"residual dump written: {dump_path} ({len(residual_states)} states)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

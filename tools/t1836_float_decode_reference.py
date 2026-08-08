#!/usr/bin/env python3
"""T-1836: float32-verified residual-stream reference at every PROMPT position and every
GENERATED position.

This is tools/t1820_float_position_reference.py (branch brunel/t1820-drift-mechanism, read in
full) with one change: after the prompt is consumed the walk keeps going -- the last state is
passed through the model's own final norm and head, the argmax is taken, and that token is fed
back in, for `--max-new` steps. Its precision discipline, its manual per-layer composition, its
DynamicCache stepping, and its next-token-argmax cross-check against the model's own standard
forward are carried over unmodified and re-verified by execution every run.

The generated ids this produces are the sequence the engine probe is teacher-forced onto, which
is what makes the interior comparison a comparison of the same sequence on both sides.

Output:
  <dump>            (T, 29, H) float32, T = prompt_len + max_new
  <dump>.tokens.npy (T,) int32   -- the token consumed at each position
  <dump>.meta.json  -- prompt_len, max_new, generated ids, and the executed precision facts

Offline only (local_files_only=True); never touches the network.
"""

from __future__ import annotations

import argparse
import json
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


def recompute_residual_stream(model, input_ids, device, max_new: int, forced=None):
    """T-1820's own `recompute_residual_stream`, continued past the prompt.

    The forward composition inside the position loop is unchanged. What is new is that once
    the prompt is exhausted the next input token is selected from the position's own final
    state, through the model's own `model.model.norm` and `model.lm_head` -- the float
    analogue of the engine's `RmsNormSite("final_norm")` -> `LogitsSite` ->
    `ArgmaxLowestIndexTieBreak` selection step, in the same place in the loop.
    """
    from transformers import DynamicCache
    from transformers.masking_utils import create_causal_mask
    from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb, repeat_kv

    import torch

    n_layers = model.config.num_hidden_layers
    prompt_len = input_ids.shape[1]
    total = prompt_len + max_new
    per_position: list[list] = []
    consumed: list[int] = []
    argmaxes: list[int] = []

    cache = DynamicCache()
    with torch.no_grad():
        for t in range(total):
            if t < prompt_len:
                token = int(input_ids[0, t].item())
            elif forced is not None:
                # Teacher forcing: the continuation is a supplied id list (the ENGINE's own
                # generation), so the float side reads the sequence the engine actually
                # produced. The model's own argmax is still recorded at every step, so the two
                # trajectories can be reconciled offline.
                token = int(forced[t - prompt_len])
            else:
                token = argmaxes[-1]
            consumed.append(token)
            step_ids = torch.tensor([[token]], device=device, dtype=torch.long)
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
            # The selection step, from the last prompt position onward -- the same place in
            # the loop the engine's own decode takes it.
            if t + 1 >= prompt_len:
                final_normed = model.model.norm(states[-1].unsqueeze(0))
                argmaxes.append(int(model.lm_head(final_normed).argmax(dim=-1).item()))
            else:
                argmaxes.append(-1)

    return per_position, consumed, argmaxes


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
    parser.add_argument("--max-new", type=int, default=24)
    parser.add_argument("--force-tokens", default=None,
                        help="comma-separated ids to teacher-force the continuation onto (the "
                             "engine's own generation), instead of the model's own greedy "
                             "choice. The complement of the engine probe's own --force-tokens: "
                             "together the two give an interior comparison along EITHER side's "
                             "trajectory, with matched inputs on both sides in both cases.")
    parser.add_argument("--dump-suffix", default="",
                        help="suffix appended to the dump stem, so a forced run does not "
                             "overwrite the free one")
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
            f"attn_implementation did not take effect: "
            f"{getattr(model.config, '_attn_implementation', None)!r}"
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
    templated = tokenizer.apply_chat_template(messages, add_generation_prompt=True,
                                              return_tensors="pt")
    input_ids = templated["input_ids"].to(device) if hasattr(templated, "to") else templated.to(device)
    prompt_len = input_ids.shape[1]

    forced = None
    if args.force_tokens:
        forced = [int(x) for x in args.force_tokens.split(",") if x]
        if len(forced) < args.max_new:
            raise SystemExit(f"--force-tokens has {len(forced)} ids, --max-new is {args.max_new}")
    per_position, consumed, argmaxes = recompute_residual_stream(
        model, input_ids, device, args.max_new, forced)
    n_layers = model.config.num_hidden_layers
    total = prompt_len + args.max_new
    if len(per_position) != total:
        raise SystemExit(f"FAILED: expected {total} positions, got {len(per_position)}")
    for t, states in enumerate(per_position):
        if len(states) != n_layers + 1:
            raise SystemExit(f"FAILED: position {t} carries {len(states)} states, "
                             f"expected {n_layers+1}")

    # T-1795's own end-to-end check on the manual composition, unchanged in what it asserts and
    # applied at the prompt's last position: the manual composition's next-token argmax must
    # equal the model's own standard forward's. This is the check that the generation below
    # rests on -- if the manual composition selected a different first token, every generated
    # position after it would be a different sequence.
    with torch.no_grad():
        final_normed = model.model.norm(per_position[prompt_len - 1][-1].unsqueeze(0))
        manual_argmax = int(model.lm_head(final_normed).argmax(dim=-1).item())
        ref_out = model(input_ids=input_ids, use_cache=False)
        ref_argmax = int(ref_out.logits[0, -1].argmax(dim=-1).item())
    print(f"argmax_cross_check: manual={manual_argmax} standard={ref_argmax} "
          f"match={manual_argmax == ref_argmax}")
    if manual_argmax != ref_argmax:
        raise AssertionError("manual per-layer composition diverges from the model's own forward")

    # A second, independent view of the generation: the model's own `generate`, greedy. This
    # is a REPORTED figure rather than a gate past its first element, and the reason is a real
    # property of the thing being measured rather than a softened check. `generate` consumes
    # the prompt in ONE batched forward; this walk consumes it a token at a time. The two are
    # the same arithmetic in different reduction orders, so they can select different tokens
    # at some step without either being wrong -- and how many steps they stay together is
    # itself a measurement of how sharp the greedy trajectory is. Element 0 IS gated: it is
    # the same quantity `argmax_cross_check` above already asserts, and a disagreement there
    # would be a wiring defect rather than a reduction-order one.
    if forced is not None:
        agree_forced = 0
        while (agree_forced < args.max_new
               and forced[agree_forced] == argmaxes[prompt_len - 1 + agree_forced]):
            agree_forced += 1
        print(f"forced_vs_float_argmax: the forced continuation and the float model's own "
              f"choice agree for the first {agree_forced}/{args.max_new} generated tokens")
    with torch.no_grad():
        gen = model.generate(input_ids=input_ids, max_new_tokens=args.max_new, do_sample=False,
                             num_beams=1, use_cache=True,
                             pad_token_id=tokenizer.eos_token_id)
    hf_generated = [int(x) for x in gen[0, prompt_len:].tolist()]
    manual_generated = [argmaxes[prompt_len - 1 + g] for g in range(args.max_new)]
    agree = 0
    for a, b in zip(manual_generated, hf_generated):
        if a != b:
            break
        agree += 1
    print(f"generate_cross_check: the incremental walk and model.generate (batched prefill) "
          f"agree on {agree}/{len(hf_generated)} generated tokens")
    if agree == 0 and forced is None:
        raise AssertionError(
            f"the manual walk's FIRST generated token differs from model.generate's: "
            f"manual={manual_generated[0]} hf={hf_generated[0]}")

    arr = np.stack([np.stack([s.cpu().numpy() for s in states]) for states in per_position]).astype(
        np.float32
    )
    dump_path = Path(args.dump)
    if args.dump_suffix:
        dump_path = dump_path.with_name(
            dump_path.name.replace(".npy", f"{args.dump_suffix}.npy"))
    dump_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(dump_path, arr)
    np.save(dump_path.with_suffix(".tokens.npy"), np.array(consumed, dtype=np.int32))
    meta = {
        "prompt_len": prompt_len,
        "max_new": args.max_new,
        "consumed": consumed,
        "argmax": argmaxes,
        "generated": manual_generated,
        "param_dtypes": sorted(str(d) for d in param_dtypes),
        "attn_implementation": model.config._attn_implementation,
        "allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32) if device == "cuda" else None,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
    }
    Path(str(dump_path) + ".meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"float dump written: {dump_path} shape={arr.shape} prompt_tokens={prompt_len} "
          f"generated={args.max_new}")
    print(f"generated ids: {','.join(str(t) for t in manual_generated)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

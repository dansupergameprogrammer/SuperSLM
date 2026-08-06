#!/usr/bin/env python3
"""T-1787: unmodified copy of T-1786's own `tools/t1786_float_attn_reference.py` (branch
`claude/t1786-error-budget`, itself an unmodified copy of T-1778's own float reference, read in
full before reuse), extended by exactly ONE addition: this script ALSO dumps the row's own
PRE-SOFTMAX real-valued scores (`softmax(...)`'s own input tensor, in the SAME oracle
composition T-1786/T-1778 already proved independent and self-consistent), alongside the
post-softmax probabilities T-1786's own copy dumps. Nothing else changes: the two-path
capture/oracle discipline, the independence-from-the-engine's-own-construction property, and the
float32-throughout load are all identical to T-1786's own file, which is itself identical to
T-1778's (`Claude/Brunel/t1778-float-attn-reference-2026-08-06.md`).

WHY THE ADDITION, AND WHY IT DOES NOT BREAK "reference is read once, never recomputed."
T-1787's own commission (`Claude/Brunel/t1787-raw-score-divergence-2026-08-06.md`) is to grade
production's RAW INTEGER attention scores against the float reference's own PRE-softmax scores --
a comparison T-1786's own dump (probabilities only) cannot answer, because a post-softmax
probability and a pre-softmax score are different quantities on different scales
(`StandardsDocument.md` 5.4, "two stages of one pipeline compared as though they were one"). This
script computes the IDENTICAL SAME real number T-1786's own oracle path already computes
internally as `scores` (`oracle_recompute_attn`'s own local variable, `torch.matmul(query_states,
key_rep.transpose(2,3)) * scaling`, plus the causal mask) -- it is not a new derivation of what
attention "should" compute, it is capturing a value the EXISTING proven-independent oracle
computation already produces internally and previously discarded. The `torch.equal` two-path
self-check below is extended to also compare `softmax(captured_scores)` against the
independently-captured `attn_weights` tensor -- proving `captured_scores` really is that
attention module's own pre-softmax input, not a separately-derived quantity.

DOES THIS SCRIPT TOUCH ANY DERIVED INTEGER CONSTANT FROM THE ENGINE'S OWN CONSTRUCTION? No,
unchanged from T-1786/T-1778: no import of `superslm`, `intmath`, `dynamic_engine`, or any
vendored Python reference; `q_ln2`/`kIExpClipN`/`IExpScaleConstants`/`IdealProbsQ15` do not
appear anywhere in this file.

Dump format, one row per (layer, head), text:

    <num_rows> <width>
    <layer> <head> <width> <s_0> <s_1> ... <s_{width-1}> <p_0> <p_1> ... <p_{width-1}>
        (s_k = pre-softmax real score, p_k = post-softmax probability, both float64 repr)

Usage
-----
    python tools\\t1787_float_prescore_reference.py \\
        "What are some tips for keeping houseplants alive during winter?" \\
        --system "You are Qwen, created by Alibaba Cloud. You are a helpful assistant." \\
        --dump out\\t1787\\p1_float.txt
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
        if len(entries) > 1:
            raise SystemExit(f"{p} has {len(entries)} snapshots; pass --model with an explicit one")
    return p


def capture_incremental_attn(model, input_ids, device):
    """Path (a), PRIMARY: forward-hook the real `self_attn` modules; keep only the LAST step's
    attn_weights per layer. Unchanged from T-1786/T-1778."""
    from transformers import DynamicCache

    import torch

    n_layers = model.config.num_hidden_layers
    captured: dict[int, "torch.Tensor"] = {}

    def make_hook(idx):
        def hook(module, args, output):
            attn_weights = output[1]
            if attn_weights is None:
                raise AssertionError(
                    f"layer {idx}: self_attn returned attn_weights=None -- attn_implementation "
                    f"is not actually eager (check model.config._attn_implementation)"
                )
            captured[idx] = attn_weights.detach()[0].float().clone()  # [num_heads, 1, key_len]

        return hook

    handles = [model.model.layers[i].self_attn.register_forward_hook(make_hook(i)) for i in range(n_layers)]

    try:
        cache = DynamicCache()
        with torch.no_grad():
            for t in range(input_ids.shape[1]):
                model(input_ids=input_ids[:, t : t + 1], past_key_values=cache, use_cache=True)
    finally:
        for h in handles:
            h.remove()

    return captured


def oracle_recompute_attn(model, input_ids, device):
    """Path (b), ORACLE: an independent, direct-composition recomputation of the SAME last step's
    attention weights AND pre-softmax scores, for every layer. Unchanged from T-1786/T-1778
    except that `scores` (previously computed and discarded) is now also captured at t ==
    last position, per layer -- read-only observation of an existing intermediate, not a new
    computation path."""
    from transformers import DynamicCache
    from transformers.masking_utils import create_causal_mask
    from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb, repeat_kv

    import torch

    n_layers = model.config.num_hidden_layers
    head_dim = model.config.head_dim if hasattr(model.config, "head_dim") and model.config.head_dim else (
        model.config.hidden_size // model.config.num_attention_heads
    )
    num_kv_groups = model.config.num_attention_heads // model.config.num_key_value_heads

    qkv_captured: dict[int, dict[str, "torch.Tensor"]] = {i: {} for i in range(n_layers)}

    def make_qkv_hook(idx, name):
        def hook(module, args, output):
            qkv_captured[idx][name] = output.detach().clone()

        return hook

    handles = []
    for i in range(n_layers):
        attn = model.model.layers[i].self_attn
        handles.append(attn.q_proj.register_forward_hook(make_qkv_hook(i, "q")))
        handles.append(attn.k_proj.register_forward_hook(make_qkv_hook(i, "k")))

    oracle_captured: dict[int, "torch.Tensor"] = {}
    oracle_scores: dict[int, "torch.Tensor"] = {}

    try:
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

                hidden_states = inputs_embeds
                for i, layer in enumerate(model.model.layers[:n_layers]):
                    attn = layer.self_attn
                    normed = layer.input_layernorm(hidden_states)

                    input_shape = normed.shape[:-1]
                    hidden_shape = (*input_shape, -1, head_dim)

                    qkv_captured[i].clear()
                    q_lin = attn.q_proj(normed)
                    k_lin = attn.k_proj(normed)
                    v_lin = attn.v_proj(normed)

                    query_states = q_lin.view(hidden_shape).transpose(1, 2)
                    key_states = k_lin.view(hidden_shape).transpose(1, 2)
                    value_states = v_lin.view(hidden_shape).transpose(1, 2)

                    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

                    key_states_full, value_states_full = cache.update(key_states, value_states, i)

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

                    if t == input_ids.shape[1] - 1:
                        oracle_captured[i] = weights.detach()[0].float().clone()
                        oracle_scores[i] = scores.detach()[0].float().clone()

                    residual = hidden_states
                    hidden_states = residual + attn_out
                    residual = hidden_states
                    normed2 = layer.post_attention_layernorm(hidden_states)
                    hidden_states = residual + layer.mlp(normed2)
    finally:
        for h in handles:
            h.remove()

    return oracle_captured, oracle_scores


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("prompt", help="the user prompt text")
    parser.add_argument(
        "--system",
        default="You are Qwen, created by Alibaba Cloud. You are a helpful assistant.",
        help="system prompt (should match the .sslm side's manually-built prompt exactly)",
    )
    parser.add_argument("--model", default=str(DEFAULT_MODEL), help="path to a local HF checkpoint directory")
    parser.add_argument("--dump", required=True, help="path to write the per-(layer,head) score+probability dump")
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

    messages = [
        {"role": "system", "content": args.system},
        {"role": "user", "content": args.prompt},
    ]
    templated = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
    input_ids = templated["input_ids"].to(device)
    prompt_len = input_ids.shape[1]

    n_layers = model.config.num_hidden_layers
    num_heads = model.config.num_attention_heads

    hook_captured = capture_incremental_attn(model, input_ids, device)
    print(f"capture: direct forward hooks (self_attn), token-at-a-time incremental (DynamicCache), "
          f"{n_layers} layers x {num_heads} heads, prompt_tokens={prompt_len}")

    oracle_captured, oracle_scores = oracle_recompute_attn(model, input_ids, device)

    mismatches = []
    for i in range(n_layers):
        a = hook_captured[i]
        b = oracle_captured[i]
        if not torch.equal(a, b):
            max_abs = (a - b).abs().max().item()
            mismatches.append((i, max_abs))
    if mismatches:
        raise AssertionError(
            "oracle self-check FAILED at layers "
            + ", ".join(f"{i} (max|delta|={m:.3e})" for i, m in mismatches)
            + " -- hook-captured and independently-recomposed attention weights diverge"
        )
    print(f"oracle_self_check: all {n_layers} layers, hook-captured and independently-recomposed "
          f"attention weights are bit-for-bit identical (torch.equal)")

    # NEW: prove oracle_scores really is the pre-softmax input to the hook-captured weights --
    # softmax(oracle_scores[i]) must equal hook_captured[i] (float32 softmax rounding only, tight
    # tolerance).
    score_mismatches = []
    for i in range(n_layers):
        recomputed = torch.nn.functional.softmax(oracle_scores[i], dim=-1, dtype=torch.float32)
        if not torch.allclose(recomputed, hook_captured[i], atol=1e-6):
            max_abs = (recomputed - hook_captured[i]).abs().max().item()
            score_mismatches.append((i, max_abs))
    if score_mismatches:
        raise AssertionError(
            "score self-check FAILED at layers "
            + ", ".join(f"{i} (max|delta|={m:.3e})" for i, m in score_mismatches)
            + " -- softmax(captured pre-softmax scores) does not reproduce the hook-captured "
              "attention weights"
        )
    print(f"score_self_check: softmax(captured pre-softmax scores) reproduces the hook-captured "
          f"attention weights within 1e-6, all {n_layers} layers")

    for i in range(n_layers):
        row_sums = hook_captured[i].sum(dim=-1)
        if not torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-4):
            raise AssertionError(f"layer {i}: attention row sums are not ~1.0: {row_sums.tolist()}")
    print("row_sum_sanity_check: every captured row sums to 1.0 within 1e-4 (float32 rounding), all layers")

    dump_path = Path(args.dump)
    dump_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for layer in CHECKPOINT_LAYERS:
        w = hook_captured[layer]  # [num_heads, 1, key_len]
        s = oracle_scores[layer]  # [num_heads, 1, key_len]
        key_len = w.shape[-1]
        for head in range(num_heads):
            probs = w[head, 0, :].double().tolist()
            scores = s[head, 0, :].double().tolist()
            rows.append((layer, head, key_len, scores, probs))

    with open(dump_path, "w", encoding="ascii") as f:
        f.write(f"{len(rows)} {prompt_len}\n")
        for layer, head, key_len, scores, probs in rows:
            f.write(
                f"{layer} {head} {key_len} "
                + " ".join(f"{v!r}" for v in scores)
                + " "
                + " ".join(f"{v!r}" for v in probs)
                + "\n"
            )

    print(f"layer_dump_written: {len(rows)} (layer,head) rows, width={prompt_len}, "
          f"layers={list(CHECKPOINT_LAYERS)} -> {dump_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

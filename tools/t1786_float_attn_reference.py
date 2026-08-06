#!/usr/bin/env python3
"""T-1786: unmodified copy of T-1778's own `tools/t1778_float_attn_reference.py`
(branch `claude/t1778-float-attn-reference`, read in full before reuse), renamed only so this
ticket's own worktree does not depend on another seat's branch. Every property T-1778 §4-§6
established (independence from the engine's integer construction, genuine float32 compute
confirmed by execution, two-path oracle cross-check) applies unchanged -- nothing below this
docstring differs from the source file. Used here to build the SAME fixed, independent
reference for three NEW held-out prompts (distinct from every prompt set used anywhere else in
this campaign), which this ticket's stage-substitution measurements are graded against and
which never moves across any substitution (`StandardsDocument.md` §5.4).

ORIGINAL DOCSTRING FOLLOWS, verbatim:

T-1778: float attention-probability reference, independent of the engine's own integer
i-exp construction entirely -- built to close the hole named by
Claude/Popper/t1776-t1772-t1773-fixed-reference-debunk-2026-08-06.md (D-SLM1195-1198): every
prior "is layer 0's attention worse" comparison in this chain (T-1762 through T-1776) graded
the engine against `IdealProbsQ15`, a reference that derives its own clip window from the
SAME `q_ln2` the engine derives and consumes -- so every measurement to date compared the
construction against a reference sharing its own construction. This script computes the
float CHECKPOINT MODEL's own attention probabilities -- ordinary `softmax(QK^T/sqrt(d))` in
float32, via HuggingFace `transformers`' own eager attention path -- for the SAME rows (same
prompt, same last-token query, same causal key range, same per-head structure) the engine's
own `tools/t1778_engine_attn_probe.cpp` captures. NOTHING here reads `IExpScaleConstants`,
`IdealProbsQ15`, `q_ln2`, `kIExpClipN`, or any other derived integer constant from the
engine's own construction -- checked explicitly at the end of this docstring.

WHAT THIS DOES AND WHY, READ BEFORE TRUSTING ANY OUTPUT.

This is built on `tools/float_reference_layer_dump.py`'s own proven capture discipline (read
in full before reuse), carried over unchanged for the two reasons that script's own docstring
gives, both apply identically here:

  1. NEVER a single batched full-prompt forward. The int8 engine's own attention is an
     incrementally populated K/V cache, computed ONE TOKEN PER forward call
     (forward_sites.cpp's RunLayerLoop, one call per prompt token during prefill). A batched
     forward is NOT the same arithmetic composition. This script therefore runs a
     token-at-a-time incremental forward with an explicit `DynamicCache`, one new token per
     call, for every prompt token in order -- matching the int8 side's own composition -- and
     captures only the LAST call's attention weights (the last prompt token's own per-head,
     per-layer softmax row over the full causal key range) -- the same row the engine probe
     captures (its own "last prompt token" forward, `width = context_length + 1`).

  2. `attn_implementation="eager"` is forced at model load. Only the eager path computes and
     returns an explicit `nn.functional.softmax(...)` probability tensor
     (`transformers/models/qwen2/modeling_qwen2.py::eager_attention_forward`); SDPA/flash
     paths never materialize it. This is the model's actual, executed attention computation
     for this checkpoint's own float weights -- not a redefinition or a second derivation of
     what attention "should" compute, and not the fused-kernel fast path (irrelevant here:
     eager and SDPA compute the identical mathematical softmax, to float32 rounding, and this
     script needs the EXPLICIT tensor, not a fused kernel's internal state).

TWO INDEPENDENT CAPTURE PATHS, cross-checked against each other every invocation (the
oracle-independence discipline `float_reference_layer_dump.py::interior_row_oracle` already
established for hidden states, extended here to attention weights):

  (a) PRIMARY: a `register_forward_hook` on each `model.model.layers[i].self_attn` module,
      reading that module's own returned `attn_weights` (the second element of
      `Qwen2Attention.forward`'s return tuple) -- the exact tensor the model's real forward
      pass computes and would discard if `output_attentions` were not requested. This is the
      float model's genuine, in-band attention computation, captured without altering it.
  (b) ORACLE: a SECOND, independently-composed computation for the SAME step: hooks on
      `q_proj`/`k_proj`/`v_proj` capture the pre-RoPE projections directly; RoPE is applied via
      the model's own `apply_rotary_pos_emb` (imported, called directly -- not through
      `Qwen2Attention.forward`); GQA head expansion via the model's own `repeat_kv` (same
      import, same direct call); the score/softmax composition
      (`matmul -> * scaling -> softmax(dim=-1, dtype=float32)`) is written OUT IN THIS SCRIPT,
      not by calling `eager_attention_forward` -- so the two paths share no code below
      `apply_rotary_pos_emb`/`repeat_kv` (deterministic library utilities, not a source of
      attention-specific arithmetic) and independently arrive at the same tensor, or the
      self-check fails loudly.

  `torch.equal(a, b)` (bit-for-bit, not `allclose`) is required between (a) and (b) at every
  checkpoint layer, every prompt, or this script raises AssertionError and writes no dump.

DOES THIS SCRIPT TOUCH ANY DERIVED INTEGER CONSTANT FROM THE ENGINE'S OWN CONSTRUCTION?
Checked explicitly, per this ticket's own requirement: no. This script imports nothing from
`superslm`, `intmath`, `dynamic_engine`, or any vendored Python reference
(`D:\\Wizard\\Tools\\superslm_spike\\...`) -- the exact objects T-1776 named as sharing the
construction under test. It loads the ORIGINAL HuggingFace float checkpoint
(`D:\\hf_cache\\hub\\models--Qwen--Qwen2.5-1.5B-Instruct\\...`) and runs `transformers`'
stock `Qwen2ForCausalLM` forward, unmodified apart from `attn_implementation="eager"` (a
library flag selecting which attention KERNEL runs, not a numeric input to any formula) and
the two hook installations described above (read-only observation, not a computation change).
`q_ln2`, `kIExpClipN`, `IExpScaleConstants`, and `IdealProbsQ15` do not appear anywhere in
this file.

Offline only. Loads the local HuggingFace cache with local_files_only=True; never touches the
network.

Dump format, one row per (layer, head), text (matches `t1778_engine_attn_probe.cpp`'s own
dump convention for easy cross-language diffing):

    <num_rows> <width>
    <layer> <head> <width> <p_0> <p_1> ... <p_{width-1}>   (repeated num_rows times, p_k as
                                                             float64 repr, full precision)

Usage
-----
    python tools\\t1778_float_attn_reference.py \\
        "Can you recommend a good recipe for a beginner baker making bread for the first time?" \\
        --system "You are Qwen, created by Alibaba Cloud. You are a helpful assistant." \\
        --dump out\\t1778\\p1_float.txt
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

CHECKPOINT_LAYERS = (0, 1, 2, 9, 18, 27)  # identical convention to t1778_engine_attn_probe.cpp


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
    """Path (a), PRIMARY: forward-hook the real `self_attn` modules; keep only the LAST
    step's attn_weights per layer (matching float_reference_layer_dump.py's own "only the
    last call's captured values are dumped" convention -- the int8 side's own prefill calls
    build KV-cache state without being individually dumped, and this is the corresponding
    row: the LAST prompt token's own softmax, over the full causal key range at that point)."""
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
    """Path (b), ORACLE: an independent, direct-composition recomputation of the SAME last
    step's attention weights for every layer, sharing no code with `eager_attention_forward`
    or with path (a)'s hook dispatch below `apply_rotary_pos_emb`/`repeat_kv`. Hooks on
    q_proj/k_proj/v_proj capture pre-RoPE projections directly; RoPE and GQA expansion reuse
    the model's own (deterministic, library) `apply_rotary_pos_emb`/`repeat_kv`; the
    score/softmax composition is written out here, independently of eager_attention_forward's
    own body."""
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

                    # Same library RoPE call the model's own Qwen2Attention.forward makes --
                    # deterministic, not a source of attention-specific arithmetic.
                    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

                    key_states_full, value_states_full = cache.update(key_states, value_states, i)

                    # Independent score/softmax composition -- NOT eager_attention_forward.
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

                    residual = hidden_states
                    hidden_states = residual + attn_out
                    residual = hidden_states
                    normed2 = layer.post_attention_layernorm(hidden_states)
                    hidden_states = residual + layer.mlp(normed2)
    finally:
        for h in handles:
            h.remove()

    return oracle_captured


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("prompt", help="the user prompt text")
    parser.add_argument(
        "--system",
        default="You are Qwen, created by Alibaba Cloud. You are a helpful assistant.",
        help="system prompt (should match the .sslm side's manually-built prompt exactly)",
    )
    parser.add_argument("--model", default=str(DEFAULT_MODEL), help="path to a local HF checkpoint directory")
    parser.add_argument("--dump", required=True, help="path to write the per-(layer,head) probability dump")
    args = parser.parse_args(argv)

    model_path = _resolve_default_model(Path(args.model))
    if not model_path.exists():
        raise SystemExit(f"model path does not exist: {model_path}")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # float32, not torch_dtype="auto" (which would load this checkpoint's native bfloat16 and
    # cast the softmax output down to it, introducing ~1e-3-scale bf16 rounding into the
    # reference itself -- exactly the kind of reference-side noise that would confound a
    # measurement whose entire purpose is grading engine error against ground truth). Loading
    # in float32 upcasts the stored bf16 weights once at load and does all arithmetic
    # (including the softmax this script grades against) at float32 precision -- a materially
    # tighter reference than the model's own inference-time dtype, appropriate for a ground-
    # truth oracle even though it is not how this checkpoint is normally served.
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

    oracle_captured = oracle_recompute_attn(model, input_ids, device)

    mismatches = []
    for i in range(n_layers):
        a = hook_captured[i]  # [num_heads, 1, key_len]
        b = oracle_captured[i]  # [num_heads, 1, key_len]
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

    # Row-sum sanity check (every row is a genuine probability distribution). The model runs
    # in float32 end-to-end (see the load call above), so this is float32 accumulation
    # rounding only -- a tight tolerance is appropriate.
    for i in range(n_layers):
        row_sums = hook_captured[i].sum(dim=-1)  # [num_heads, 1]
        if not torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-4):
            raise AssertionError(f"layer {i}: attention row sums are not ~1.0: {row_sums.tolist()}")
    print("row_sum_sanity_check: every captured row sums to 1.0 within 1e-4 (float32 rounding), all layers")

    dump_path = Path(args.dump)
    dump_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for layer in CHECKPOINT_LAYERS:
        w = hook_captured[layer]  # [num_heads, 1, key_len]
        key_len = w.shape[-1]
        for head in range(num_heads):
            probs = w[head, 0, :].double().tolist()
            rows.append((layer, head, key_len, probs))

    with open(dump_path, "w", encoding="ascii") as f:
        f.write(f"{len(rows)} {prompt_len}\n")
        for layer, head, key_len, probs in rows:
            f.write(f"{layer} {head} {key_len} " + " ".join(f"{p!r}" for p in probs) + "\n")

    print(f"layer_dump_written: {len(rows)} (layer,head) rows, width={prompt_len}, "
          f"layers={list(CHECKPOINT_LAYERS)} -> {dump_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

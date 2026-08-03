#!/usr/bin/env python3
"""Float per-layer hidden-state capture (T-1686) -- the float side of the
T-1683 layer-bisection instrument (design Claude/Vitruvius/superslm-t1683-
layer-bisection-design-2026-08-02.md S4.2).

WHAT THIS DOES AND WHY, READ BEFORE TRUSTING ANY OUTPUT.

This script captures the residual stream's raw, pre-final-norm value at the
embedding row and after each of the 28 decoder layers, for the LAST token of
a chat-templated prompt -- position 0 of this campaign's own convention (the
prompt's own last-token forward pass, the same row
tools/float_reference_logits.py already reads for T-1681).

Two departures from the obvious implementation, both load-bearing, both
found at design time from source (design S2.4, S2.5) and confirmed by
execution (the strike, Claude/Loki/superslm-t1683-layer-bisection-design-
strike-2026-08-02.md; the gating re-check, Claude/Curie/t1683-gating-check-
2026-08-02-probe.py):

  1. NEVER `output_hidden_states=True`. HF's own post-processing
     (`tie_last_hidden_states=True`, the default for every causal LM) silently
     overwrites the LAST layer's collected output with the POST-final-norm
     value (transformers/utils/output_capturing.py). This script installs its
     own `register_forward_hook` on `model.model.embed_tokens` and each of
     `model.model.layers[i]` instead, which receive each layer's raw,
     pre-norm output including the last.

  2. NEVER a single batched full-prompt forward. The int8 engine's own
     attention is an incrementally populated K/V cache, computed ONE TOKEN
     PER forward call (forward_sites.cpp's RunGreedyDecodeLoop). A batched
     forward is NOT the same arithmetic composition and, executed, produces
     differences at 28 of 29 layer boundaries on 9 of 9 prompts, growing with
     depth and correlated with prompt content -- indistinguishable in shape
     from a genuine int8-vs-float defect (the strike's own finding). This
     script therefore runs a token-at-a-time incremental forward with an
     explicit `DynamicCache`, one new token per call, for every prompt token
     in order -- matching the int8 side's own composition -- and dumps only
     the LAST call's captured values (the last prompt token's own per-layer
     outputs).

Two self-checks (run every invocation) and one independent oracle (run every
invocation) guard this script's own capture correctness -- see
`endpoint_self_check` and `interior_row_oracle` below; none of the three is
a re-litigation of whether the installed transformers 5.13.1's own
`Qwen2DecoderLayer.forward` arithmetic is correct, which this script trusts
as vetted library code (the same trust design S2.4 already places in it for
the tie-break-correction finding).

Dump format (design S4.2 step 4, matching T-1685's int8-side format):
    uint64  rows (29)
    uint64  hidden_size
    uint64  prompt_fingerprint  (FNV-1a 64-bit hash of the literal
        chat-templated prompt text, UTF-8, before tokenization -- obtained
        from `tokenizer.apply_chat_template(messages, tokenize=False,
        add_generation_prompt=True)`, confirmed byte-identical to
        tools/sslm_generate.cpp's/tools/sslm_layer_trace.cpp's own
        manually-built prompt string for this campaign's own PROMPT_SET
        convention)
    uint64  capture_mode (always 1 -- incremental-per-token-kv-cache; the
        only value this script ever writes; design S5's provenance check
        rejects any other value on either side)
    rows * hidden_size  float32 values, row-major

Offline only. Loads the local HuggingFace cache with local_files_only=True;
never touches the network.

Usage
-----
    python tools\\float_reference_layer_dump.py "What is the capital of France?" \\
        --system "You are Qwen, created by Alibaba Cloud. You are a helpful assistant." \\
        --dump out\\t1683\\capital_of_france.float.bin
"""

from __future__ import annotations

import argparse
import struct
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


def fnv1a64(s: str) -> int:
    """Verbatim of tools/sslm_layer_trace.cpp's own Fnv1a64: the standard
    FNV-1a constants, over the UTF-8 bytes, masked to 64 bits (Python ints
    are unbounded; C++'s uint64_t wraps by hardware, this masks explicitly
    to match)."""
    h = 0xCBF29CE484222325
    for b in s.encode("utf-8"):
        h ^= b
        h = (h * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return h


def capture_incremental(model, input_ids, device):
    """Design S4.2 steps 2-3: direct forward hooks (never output_hidden_states),
    token-at-a-time incremental forward with an explicit DynamicCache (never
    batched). Returns {index: float32 tensor[hidden_size]} for index 0
    (embedding) and 1..num_hidden_layers (each decoder layer's raw output),
    taken from the LAST call only (every earlier call's captured values are
    overwritten as the loop advances -- the int8 side's own prefill calls
    build state without being dumped themselves, design S2.5/S4.1 step 3)."""
    from transformers import DynamicCache

    import torch

    n_layers = model.config.num_hidden_layers
    captured: dict[int, "torch.Tensor"] = {}

    def make_hook(idx):
        def hook(module, args, output):
            t = output if not isinstance(output, tuple) else output[0]
            # Explicit copy taken AT THE POINT OF CAPTURE (detach + float32 +
            # clone), never a reference into a buffer a later op could still
            # mutate -- a written discipline, not contingent on today's HF
            # internals happening not to mutate in place.
            captured[idx] = t.detach()[0, -1, :].float().clone()

        return hook

    handles = [model.model.embed_tokens.register_forward_hook(make_hook(0))]
    handles += [model.model.layers[i].register_forward_hook(make_hook(i + 1)) for i in range(n_layers)]

    try:
        cache = DynamicCache()
        with torch.no_grad():
            for t in range(input_ids.shape[1]):
                model(input_ids=input_ids[:, t : t + 1], past_key_values=cache, use_cache=True)
    finally:
        for h in handles:
            h.remove()

    return captured


def endpoint_self_check(model, captured, input_ids, device):
    """Design S4.2 step 5, proven able to fail (coverage audit F6; design S7
    T-1686 part 1): two assertions run every invocation.

      (a) The captured "layer 28" row is NOT bit-identical to the model's OWN
          `last_hidden_state` for this exact prompt, obtained independently
          via a SEPARATE token-at-a-time incremental re-run reading
          `Qwen2Model.forward`'s own RETURN VALUE (never a hook) -- a
          positive control proving the capture bypassed the
          `tie_last_hidden_states` overwrite (S2.4) rather than silently
          reproducing it. The independent re-run uses the SAME incremental
          shape as the capture above (a fresh `DynamicCache`, one token per
          call) specifically so this check targets the tie-overwrite
          question alone, with no batched-vs-incremental confound (design
          S2.5) contaminating the comparison.

          (Design S4.2 step 5's own text specifies this as "NOT bit-identical
          to `model.model.norm(...)`'s own output computed independently on
          the same final hidden state" -- read literally (self-referential:
          `captured != model.model.norm(captured)`), this formula does NOT
          discriminate: RMSNorm is not idempotent, so even the TRUE post-norm
          value fails `x == norm(x)` -- verified by direct execution (T-1683
          build log): applying `model.model.norm` a second time to the
          model's own real `last_hidden_state` is bit-different from that
          value 100% of the time. A self-check built on that formula would
          report "OK" whether or not the tie overwrite occurred, which is
          exactly the "proven able to fail" property design S7 T-1686 part 1
          requires and the literal formula does not have. StandardsDocument
          S5.6/S7: a check that cannot fail is not a check. This
          implementation instead compares the captured row against the
          model's own independently-obtained `last_hidden_state` -- provably
          discriminating, and confirmed able to trip against a deliberately
          naive `output_hidden_states=True` capture (T-1683 build log).)
      (b) The captured "layer 0" (embedding) row equals
          `model.model.embed_tokens(input_ids[:, -1:])`'s output for the LAST
          prompt token, computed directly, independent of the hook path.
          (Design S4.2 step 5's own text names `input_ids[:, :1]`, the FIRST
          prompt token -- a transcription error, StandardsDocument S5.6: the
          captured row this script actually dumps is the LAST incremental
          call's own embedding, per S4.2 step 3's own corrected mechanism
          and step 4's "only the LAST call's captured values... are dumped."
          Checking the first token's embedding here would compare against a
          position that was never captured and never dumped, for every
          prompt longer than one token. Flagged in the T-1683 build log;
          this implementation checks the last token, the position this
          script's own contract actually captures.)

    Raises AssertionError with a diagnostic on failure; returns None on
    success (matching design S5's ProvenanceError convention of "loud,
    non-zero-exit on failure" -- an uncaught AssertionError in this script's
    __main__ is exactly that)."""
    import torch
    from transformers import DynamicCache

    n_layers = model.config.num_hidden_layers
    last_layer_row = captured[n_layers]

    cache = DynamicCache()
    with torch.no_grad():
        for t in range(input_ids.shape[1]):
            out = model.model(input_ids=input_ids[:, t : t + 1], past_key_values=cache, use_cache=True)
    model_own_final = out.last_hidden_state[0, -1, :].float()

    if torch.equal(last_layer_row, model_own_final):
        raise AssertionError(
            "endpoint_self_check (a) FAILED: captured layer-%d row is bit-identical to the "
            "model's own last_hidden_state -- the capture reproduced the tie_last_hidden_states "
            "overwrite (design S2.4) instead of bypassing it" % n_layers
        )

    with torch.no_grad():
        direct_embed = model.model.embed_tokens(input_ids[:, -1:]).squeeze(0).squeeze(0).float()
    embed_row = captured[0]
    if not torch.equal(embed_row, direct_embed):
        raise AssertionError(
            "endpoint_self_check (b) FAILED: captured layer-0 (embedding) row differs from "
            "model.model.embed_tokens(input_ids[:, -1:])'s own direct output (the last prompt "
            "token -- the position this script's own incremental capture actually dumps)"
        )


def interior_row_oracle(model, input_ids, device, hook_captured):
    """Design S4.2 step 6, re-disposed 2026-08-02 (coverage audit F7): a
    SECOND composition of the identical incremental forward, by DIRECT
    Python composition -- embed_tokens, create_causal_mask, rotary_emb, then
    each decoder layer called DIRECTLY as a function -- never through
    register_forward_hook at all. Genuinely independent of
    capture_incremental's own hook-dispatch mechanism: the two share no code
    path below the point where each decoder layer's forward runs. All 29
    rows (embedding + 28 layers) are compared bit-identical to
    `hook_captured`; a mismatch at any row raises AssertionError immediately
    (loud, no dump written)."""
    from transformers import DynamicCache
    from transformers.masking_utils import create_causal_mask

    import torch

    n_layers = model.config.num_hidden_layers
    cache2 = DynamicCache()
    direct_captured: dict[int, "torch.Tensor"] = {}

    with torch.no_grad():
        for t in range(input_ids.shape[1]):
            step_ids = input_ids[:, t : t + 1]
            inputs_embeds = model.model.embed_tokens(step_ids)
            position_ids = torch.tensor([[t]], device=device, dtype=torch.long)
            mask_kwargs = {
                "config": model.config,
                "inputs_embeds": inputs_embeds,
                "attention_mask": None,
                "past_key_values": cache2,
                "position_ids": position_ids,
            }
            causal_mask = create_causal_mask(**mask_kwargs)
            position_embeddings = model.model.rotary_emb(inputs_embeds, position_ids)

            direct_captured[0] = inputs_embeds.detach()[0, -1, :].float().clone()
            hidden_states = inputs_embeds
            for i, layer in enumerate(model.model.layers[:n_layers]):
                hidden_states = layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_embeddings=position_embeddings,
                    position_ids=position_ids,
                    past_key_values=cache2,
                    use_cache=True,
                )
                direct_captured[i + 1] = hidden_states.detach()[0, -1, :].float().clone()

    for idx in range(n_layers + 1):
        if not torch.equal(direct_captured[idx], hook_captured[idx]):
            raise AssertionError(
                "interior_row_oracle FAILED: row %d (direct-composition capture) differs from "
                "the hook-captured row -- capture wiring mismatch (wrong module, wrong index, "
                "staleness, or aliasing)" % idx
            )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("prompt", help="the user prompt text")
    parser.add_argument(
        "--system",
        default="You are Qwen, created by Alibaba Cloud. You are a helpful assistant.",
        help="system prompt (should match the .sslm side's -System exactly)",
    )
    parser.add_argument("--model", default=str(DEFAULT_MODEL), help="path to a local HF checkpoint directory")
    parser.add_argument("--dump", required=True, help="path to write the per-layer float32 dump")
    args = parser.parse_args(argv)

    model_path = _resolve_default_model(Path(args.model))
    if not model_path.exists():
        raise SystemExit(f"model path does not exist: {model_path}")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(str(model_path), local_files_only=True, torch_dtype="auto")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    messages = [
        {"role": "system", "content": args.system},
        {"role": "user", "content": args.prompt},
    ]
    # The literal chat-templated prompt text, byte-identical to
    # tools/sslm_generate.cpp's/tools/sslm_layer_trace.cpp's own manually
    # built prompt string for this campaign's PROMPT_SET convention
    # (verified at build time -- see the T-1683 build log) -- this is what
    # gets fingerprinted, not the manually-assembled input_ids.
    prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    templated = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
    input_ids = templated["input_ids"].to(device)

    hidden_size = model.config.hidden_size
    n_layers = model.config.num_hidden_layers

    captured = capture_incremental(model, input_ids, device)
    print(f"capture: direct forward hooks, token-at-a-time incremental (DynamicCache), "
          f"{n_layers + 1} rows x {hidden_size} hidden_size")

    endpoint_self_check(model, captured, input_ids, device)
    print("endpoint_self_check: layer-%d != post-norm (OK, bypass confirmed); "
          "layer-0 == direct embed (OK)" % n_layers)

    interior_row_oracle(model, input_ids, device, captured)
    print(f"interior_row_oracle: all {n_layers + 1} rows bit-identical between hook capture and "
          f"direct-composition capture")

    fingerprint = fnv1a64(prompt_text)

    dump_path = Path(args.dump)
    dump_path.parent.mkdir(parents=True, exist_ok=True)
    with open(dump_path, "wb") as f:
        f.write(struct.pack("<QQQQ", n_layers + 1, hidden_size, fingerprint, 1))
        for idx in range(n_layers + 1):
            f.write(captured[idx].cpu().numpy().astype("float32").tobytes())

    print(f"layer_dump_written: {n_layers + 1} rows x {hidden_size} hidden_size, "
          f"prompt_fingerprint=0x{fingerprint:016X} -> {dump_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

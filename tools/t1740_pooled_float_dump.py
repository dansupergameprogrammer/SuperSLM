#!/usr/bin/env python3
"""T-1740 scratch instrument: float-reference per-position, per-layer hidden
state capture -- the float side of the pooled-fidelity measurement.

This is a NEW file, not a modification of the reviewed
tools/float_reference_layer_dump.py (T-1686): that script deliberately keeps
only the LAST incremental call's captured values (one sequence position per
run, design S4.2 step 3/4). T-1740 needs every position's per-layer residual
from one prompt, so it could not be gotten by using that script unmodified.
The capture mechanism (direct register_forward_hook on model.model.embed_tokens
and each decoder layer, never output_hidden_states=True; a token-at-a-time
incremental forward with an explicit DynamicCache, never batched) is reused
verbatim from tools/float_reference_layer_dump.py's own capture_incremental
and its docstring's rationale (both departures are load-bearing there and
apply identically here) -- only the accumulation is different: every
position's hook output is kept, not just the last call's.

Dump format (custom to this scratch tool -- read by
tools/t1740_pooled_fidelity_report.py, matching tools/t1740_pooled_trace.cpp's
own layout so one report script reads both sides):
    uint64  num_positions (== the prompt's own token count, n)
    uint64  num_rows      (== num_hidden_layers + 1, embedding + each layer)
    uint64  hidden_size
    uint64  prompt_fingerprint (FNV-1a 64-bit hash of the literal
        chat-templated prompt text, UTF-8, before tokenization -- identical
        formula to tools/float_reference_layer_dump.py's own fnv1a64)
    then num_positions * num_rows float32 values, row-major within each
    position (position-major, row-minor -- position 0's num_rows rows, then
    position 1's, ...)

One self-check, run every invocation, before any dump is written: the LAST
position's captured layer-(num_hidden_layers) row is NOT bit-identical to the
model's own independently-obtained last_hidden_state for this prompt --
tools/float_reference_layer_dump.py's own endpoint_self_check(a), reused
verbatim, proving this capture bypasses the tie_last_hidden_states overwrite
(HF's default post-processing for causal LMs) at the one position an
independent oracle (the model's own return value) exists to check against.
This tool does not repeat that script's interior_row_oracle (a second,
direct-composition capture of the SAME one position) -- reproducing it at
every position would roughly double this tool's own runtime for no new
coverage beyond what capture_incremental's own hook mechanism already
provides identically at every position (the hook fires the same way whether
or not the position is the last one dumped).

Offline only. Loads the local HuggingFace cache with local_files_only=True;
never touches the network.

Usage
-----
    python tools\\t1740_pooled_float_dump.py "What is the capital of France?" \\
        --system "You are Qwen, created by Alibaba Cloud. You are a helpful assistant." \\
        --dump out\\t1740\\capital_of_france.float.bin
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
    """Verbatim of tools/float_reference_layer_dump.py's own fnv1a64 (itself
    a transcription of tools/sslm_layer_trace.cpp's Fnv1a64/
    tools/t1740_pooled_trace.cpp's Fnv1a64)."""
    h = 0xCBF29CE484222325
    for b in s.encode("utf-8"):
        h ^= b
        h = (h * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return h


def capture_all_positions(model, input_ids, device):
    """Every position's per-layer output, from a single token-at-a-time
    incremental forward with an explicit DynamicCache (never batched) --
    matching the int8 engine's own composition (design S2.5, restated in
    tools/float_reference_layer_dump.py's own docstring). Returns
    {index: list[torch.Tensor]}, index 0 == embedding, 1..n_layers == each
    decoder layer's raw (pre-final-norm) output; list[t] is position t's row."""
    from transformers import DynamicCache

    import torch

    n_layers = model.config.num_hidden_layers
    captured: dict[int, list["torch.Tensor"]] = {i: [] for i in range(n_layers + 1)}

    def make_hook(idx):
        def hook(module, args, output):
            t = output if not isinstance(output, tuple) else output[0]
            captured[idx].append(t.detach()[0, -1, :].float().clone())

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


def endpoint_self_check_last_position(model, captured, input_ids, device):
    """tools/float_reference_layer_dump.py's own endpoint_self_check(a),
    applied to this tool's LAST captured position (position n-1 -- the same
    position that script's own single-position capture dumps): the captured
    final-layer row must NOT be bit-identical to the model's own
    last_hidden_state for this exact prompt, obtained independently via a
    SEPARATE incremental re-run reading Qwen2Model.forward's own return
    value (never a hook) -- proves the capture bypassed the
    tie_last_hidden_states overwrite rather than silently reproducing it."""
    import torch
    from transformers import DynamicCache

    n_layers = model.config.num_hidden_layers
    last_layer_row = captured[n_layers][-1]

    cache = DynamicCache()
    with torch.no_grad():
        for t in range(input_ids.shape[1]):
            out = model.model(input_ids=input_ids[:, t : t + 1], past_key_values=cache, use_cache=True)
    model_own_final = out.last_hidden_state[0, -1, :].float()

    if torch.equal(last_layer_row, model_own_final):
        raise AssertionError(
            "endpoint_self_check FAILED: captured layer-%d row at the last position is "
            "bit-identical to the model's own last_hidden_state -- the capture reproduced the "
            "tie_last_hidden_states overwrite instead of bypassing it" % n_layers
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
    parser.add_argument("--dump", required=True, help="path to write the per-position per-layer float32 dump")
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
    prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    templated = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
    input_ids = templated["input_ids"].to(device)

    hidden_size = model.config.hidden_size
    n_layers = model.config.num_hidden_layers
    n_positions = input_ids.shape[1]

    print(f"prompt_tokens={n_positions}")
    captured = capture_all_positions(model, input_ids, device)
    print(f"capture: direct forward hooks, token-at-a-time incremental (DynamicCache), "
          f"{n_positions} positions x {n_layers + 1} rows x {hidden_size} hidden_size")

    endpoint_self_check_last_position(model, captured, input_ids, device)
    print("endpoint_self_check: layer-%d != post-norm at the last position (OK, tie_last_hidden_states "
          "bypass confirmed)" % n_layers)

    fingerprint = fnv1a64(prompt_text)

    dump_path = Path(args.dump)
    dump_path.parent.mkdir(parents=True, exist_ok=True)
    with open(dump_path, "wb") as f:
        f.write(struct.pack("<QQQQ", n_positions, n_layers + 1, hidden_size, fingerprint))
        for pos in range(n_positions):
            for idx in range(n_layers + 1):
                f.write(captured[idx][pos].cpu().numpy().astype("float32").tobytes())

    print(f"pooled_float_dump_written: {n_positions} positions x {n_layers + 1} rows x {hidden_size} "
          f"hidden_size, prompt_fingerprint=0x{fingerprint:016X} -> {dump_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

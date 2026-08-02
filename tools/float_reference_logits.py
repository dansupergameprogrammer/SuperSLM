#!/usr/bin/env python3
"""Float reference greedy decode WITH per-step logit capture (T-1681).

WHAT THIS DOES AND DOES NOT ESTABLISH (read this before trusting any output).
This script is `tools/float_reference_generate.py`'s sibling for a different
question. `float_reference_generate.py` answers "what does the float path
say" for `tools/compare_float.ps1`'s coarse side-by-side. This script answers
a narrower, harder question: at each generated position, was the float path's
choice a close call (a tied top-2, where int8 rounding noise flipping the
argmax is expected and NOT a defect) or a clear call (float strongly prefers
one token and the int8 engine picked a different one anyway, which points at
a defect)? Answering that needs the full logit row, not just the chosen
token id -- so this script decodes step by step (never `model.generate`,
which does not expose a clean row-per-position logit trace under KV-cached
beam-free greedy decoding) and writes every step's full float32 logit row to
a binary file for `tools/logit_margin_report.py` to compare against the int8
engine's own `--dump-logits` output from `tools/sslm_generate.cpp`.

This script establishes nothing about correctness by itself. It is the float
half of a two-sided capture; the comparison and its normalisation live in
`tools/logit_margin_report.py`, and that tool's own docstring states what the
comparison does and does not support.

Decode is byte-for-byte the same greedy policy as `float_reference_generate.py`
(do_sample=False, no beams, no temperature/top_p/top_k) -- implemented as an
explicit step loop with a KV cache instead of a `model.generate` call, so
argmax at each step matches `model.generate`'s own argmax exactly (both are a
plain `argmax` over the identical logit row; `model.generate` runs no other
transform under `do_sample=False`).

Offline only. Loads the local HuggingFace cache with local_files_only=True;
never touches the network.

Usage
-----
    python tools\\float_reference_logits.py "What is 12 + 15? Give just the number." \\
        --system "You are Qwen, created by Alibaba Cloud. You are a helpful assistant." \\
        --max-new 8 --stop 151645 151643 \\
        --dump-logits out\\logits\\f_12plus15_digit.bin

Prints the same machine-parseable fields as float_reference_generate.py:

    prompt_tokens: <n>
    output_ids: <space-separated ids, generated tokens only, stop id included if emitted>
    stop_reason: 1|0            (1 = stopped on a supplied stop id, 0 = hit the token budget)
    wall_time_seconds: <float>
    ---DECODED---
    <decoded text, stop id(s) stripped>

Dump format (little-endian, matching tools/sslm_generate.cpp's --dump-logits
so tools/logit_margin_report.py reads both with one code path shape):
    uint64  rows_produced
    uint64  vocab_size
    rows_produced * vocab_size  float32 values, row-major (row i = the full
        logit row the model computed immediately before choosing output
        token i -- i.e. BEFORE that token was appended to the sequence,
        exactly matching RunGreedyDecodeLoop's own out_logit_rows contract
        that this tool's counterpart on the int8 side reads from).
"""

from __future__ import annotations

import argparse
import struct
import sys
import time
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
    """Resolve a hub-cache repo directory to its one snapshot, if needed."""
    snaps = p / "snapshots"
    if snaps.is_dir():
        entries = [d for d in snaps.iterdir() if d.is_dir()]
        if len(entries) == 1:
            return entries[0]
        if len(entries) > 1:
            raise SystemExit(
                f"{p} has {len(entries)} snapshots; pass --model with an explicit one"
            )
    return p


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prompt", help="the user prompt text")
    parser.add_argument(
        "--system",
        default="You are Qwen, created by Alibaba Cloud. You are a helpful assistant.",
        help="system prompt (should match the .sslm side's -System exactly)",
    )
    parser.add_argument("--max-new", type=int, default=8, dest="max_new")
    parser.add_argument(
        "--model",
        default=str(DEFAULT_MODEL),
        help="path to a local HF checkpoint directory (weights + tokenizer)",
    )
    parser.add_argument(
        "--stop",
        nargs="+",
        type=int,
        default=[151645, 151643],
        help="stop token ids (default: <|im_end|>, <|endoftext|>)",
    )
    parser.add_argument(
        "--dump-logits",
        required=True,
        help="path to write the per-step float32 logit rows (see this file's docstring)",
    )
    args = parser.parse_args(argv)

    model_path = _resolve_default_model(Path(args.model))
    if not model_path.exists():
        raise SystemExit(f"model path does not exist: {model_path}")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        local_files_only=True,
        torch_dtype="auto",
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    messages = [
        {"role": "system", "content": args.system},
        {"role": "user", "content": args.prompt},
    ]
    templated = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt"
    )
    input_ids = templated["input_ids"].to(device)
    prompt_len = input_ids.shape[1]
    stop_set = set(args.stop)

    t0 = time.time()
    output_ids: list[int] = []
    logit_rows: list["torch.Tensor"] = []
    with torch.no_grad():
        # Prefill: one forward over the whole prompt, KV cache retained.
        out = model(input_ids=input_ids, use_cache=True)
        past = out.past_key_values
        next_logits = out.logits[0, -1, :]
        cur_input = None
        for _step in range(args.max_new):
            row = next_logits.detach().to(torch.float32).cpu()
            logit_rows.append(row)
            next_id = int(torch.argmax(next_logits).item())
            output_ids.append(next_id)
            if next_id in stop_set:
                break
            cur_input = torch.tensor([[next_id]], device=device)
            out = model(input_ids=cur_input, past_key_values=past, use_cache=True)
            past = out.past_key_values
            next_logits = out.logits[0, -1, :]
    wall = time.time() - t0

    stopped = 1 if (output_ids and output_ids[-1] in stop_set) else 0
    decoded = tokenizer.decode(
        [i for i in output_ids if i not in stop_set], skip_special_tokens=True
    )

    vocab_size = logit_rows[0].shape[0] if logit_rows else 0
    dump_path = Path(args.dump_logits)
    dump_path.parent.mkdir(parents=True, exist_ok=True)
    with open(dump_path, "wb") as f:
        f.write(struct.pack("<QQ", len(logit_rows), vocab_size))
        for row in logit_rows:
            f.write(row.numpy().tobytes())

    print(f"prompt_tokens: {prompt_len}")
    print(f"output_ids: {' '.join(str(i) for i in output_ids)}")
    print(f"stop_reason: {stopped}")
    print(f"wall_time_seconds: {wall:.3f}")
    print(f"logit_rows_dumped: {len(logit_rows)} rows x {vocab_size} vocab -> {dump_path}")
    print("---DECODED---")
    print(decoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

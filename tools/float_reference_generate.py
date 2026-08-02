#!/usr/bin/env python3
"""Float reference greedy decode for the T-1679 side-by-side comparison.

WHAT THIS DOES AND DOES NOT ESTABLISH (read this before trusting any output).
This script runs the same base checkpoint's own float (fp32/bf16, whatever
`from_pretrained`'s default dtype resolves to) weights through
`transformers.AutoModelForCausalLM.generate` with greedy decoding
(`do_sample=False`). It exists so `tools/compare_float.ps1` can show this
output next to the .sslm int8 engine's output for the SAME prompt, SAME chat
template, SAME token budget, and SAME stop ids. Divergence in token choice
between an int8-quantized path and a float path is EXPECTED and is not by
itself a defect -- the two paths are not required to agree token-for-token,
and this is not bit-equality (plan section 12 criterion 2, which is a
separate, unbuilt thing). This script establishes nothing about correctness;
it is one half of a coarse behavioral side-by-side, and its caller states the
same limitation again in its own output.

Offline only. Loads the local HuggingFace cache with local_files_only=True;
never touches the network.

Usage
-----
    python tools\\float_reference_generate.py "What's on the menu today?" \\
        --system "You are Qwen, created by Alibaba Cloud. You are a helpful assistant." \\
        --max-new 256 --stop 151645 151643

Prints, one field per line, machine-parseable by the PowerShell wrapper:

    prompt_tokens: <n>
    output_ids: <space-separated ids, generated tokens only, stop id included if emitted>
    stop_reason: 1|0            (1 = stopped on a supplied stop id, 0 = hit the token budget)
    wall_time_seconds: <float>
    ---DECODED---
    <decoded text, stop id(s) stripped>
"""

from __future__ import annotations

import argparse
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
    parser.add_argument("--max-new", type=int, default=256, dest="max_new")
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
    # transformers 5.13.1's apply_chat_template(..., return_tensors="pt") returns a
    # BatchEncoding (dict-like), not a bare tensor -- index into it rather than
    # treating the return value itself as the tensor.
    input_ids = templated["input_ids"].to(device)
    prompt_len = input_ids.shape[1]

    t0 = time.time()
    with torch.no_grad():
        out = model.generate(
            input_ids,
            max_new_tokens=args.max_new,
            do_sample=False,
            num_beams=1,
            temperature=None,
            top_p=None,
            top_k=None,
            eos_token_id=args.stop,
            pad_token_id=(tokenizer.pad_token_id or args.stop[0]),
        )
    wall = time.time() - t0

    output_ids = out[0, prompt_len:].tolist()
    stopped = 1 if (output_ids and output_ids[-1] in args.stop) else 0

    decoded = tokenizer.decode(
        [i for i in output_ids if i not in args.stop], skip_special_tokens=True
    )

    print(f"prompt_tokens: {prompt_len}")
    print(f"output_ids: {' '.join(str(i) for i in output_ids)}")
    print(f"stop_reason: {stopped}")
    print(f"wall_time_seconds: {wall:.3f}")
    print("---DECODED---")
    print(decoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

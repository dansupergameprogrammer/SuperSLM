#!/usr/bin/env python3
"""T-1781 cell C10 -- float reference capture over LITERAL RAW TEXT, never a
chat-templated (system, user) pair.

WHY THIS EXISTS AND HOW IT DIFFERS FROM ITS SIBLING. `float_reference_layer_
dump.py` (T-1686) always re-applies `apply_chat_template` to a system+user
message pair and captures the LAST token of that templated prompt --
position 0 of this campaign's own convention. It has no way to express "the
prompt, plus several tokens of the model's OWN self-generated continuation"
because an assistant continuation is not a message role the chat template
accepts as input.

This script captures the identical quantity at an ARBITRARY later position:
given literal text (the same chat-templated prefix PLUS a decoded
self-generated continuation, produced by `tools/sslm_decode_kv_probe.cpp`),
it tokenizes that text DIRECTLY (no chat template applied a second time) and
captures the LAST token's own per-layer hidden state, using the exact same
capture/self-check/oracle machinery `float_reference_layer_dump.py` already
built and this campaign already trusts -- imported, not re-implemented, so
this script carries no new capture logic of its own.

The one new guard this script adds: `--expect-tokens N` is REQUIRED, and the
tokenized length of the literal text must equal it exactly, checked BEFORE
any capture runs. `tools/sslm_decode_kv_probe.cpp` computes N as the exact
token count of the engine-side sequence (prompt tokens + the self-generated
continuation being probed) and a round-trip check on the ENGINE'S OWN
tokenizer already confirmed decoding then re-encoding that same text
reproduces the identical token id sequence (its own text_roundtrip_ok/
TEXT_ROUNDTRIP_FAILED lines). This script's own `--expect-tokens` check is
the analogous guard on the FLOAT side's tokenizer -- the two guards
together are what let the resulting int8/float dump PAIR be trusted as
"the same position in the same token sequence" without assuming the two
tokenizers segment identically; a length mismatch here means they did not,
and this script fails loud rather than silently comparing two different
positions.

Dump format: identical to float_reference_layer_dump.py's own (T-1686 design
S4.2 step 4) -- uint64 rows, uint64 hidden_size, uint64 prompt_fingerprint,
uint64 capture_mode(=1), then rows*hidden_size float32 values.

Offline only, local_files_only=True, matching every other tool in this tree.

Usage
-----
    python tools\\t1781_float_raw_text_dump.py --text-file out\\t1781\\NAME.text.txt \\
        --dump out\\t1781\\NAME.float.bin
(--text-file points at a file whose first line is the literal text, matching
tools\\sslm_decode_kv_probe.cpp's own .text.txt output format: line 1 is the
text, followed by `expect_tokens=N` and `fingerprint=F` lines this script
reads instead of requiring them on the command line.)
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
import float_reference_layer_dump as fld  # noqa: E402  -- reuse, not reimplement

DEFAULT_MODEL = fld.DEFAULT_MODEL


def _parse_text_file(path: Path) -> tuple[str, int, int]:
    """Reads tools/sslm_decode_kv_probe.cpp's own .text.txt sidecar: line 1 is
    the literal text (which may itself contain embedded newlines from the
    chat template, so everything up to the LAST two lines is text), the
    second-to-last line is `expect_tokens=N`, the last is `fingerprint=F`."""
    raw = path.read_text(encoding="utf-8")
    lines = raw.split("\n")
    # Last line is a trailing empty string from the file's own trailing
    # newline (the C++ side writes "<<\n" after fingerprint); drop it if so.
    if lines and lines[-1] == "":
        lines = lines[:-1]
    if len(lines) < 3:
        raise SystemExit(f"{path}: expected at least 3 lines (text, expect_tokens, fingerprint), got {len(lines)}")
    fp_line = lines[-1]
    tok_line = lines[-2]
    text_lines = lines[:-2]
    text = "\n".join(text_lines)
    if not tok_line.startswith("expect_tokens="):
        raise SystemExit(f"{path}: expected 'expect_tokens=N' on line {len(lines)-1}, got {tok_line!r}")
    if not fp_line.startswith("fingerprint="):
        raise SystemExit(f"{path}: expected 'fingerprint=F' on the last line, got {fp_line!r}")
    expect_tokens = int(tok_line[len("expect_tokens="):])
    fingerprint = int(fp_line[len("fingerprint="):])
    return text, expect_tokens, fingerprint


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--text-file", required=True, help="path to the .text.txt sidecar (see docstring)")
    parser.add_argument("--model", default=str(DEFAULT_MODEL), help="path to a local HF checkpoint directory")
    parser.add_argument("--dump", required=True, help="path to write the per-layer float32 dump")
    args = parser.parse_args(argv)

    text, expect_tokens, fingerprint = _parse_text_file(Path(args.text_file))

    model_path = fld._resolve_default_model(Path(args.model))
    if not model_path.exists():
        raise SystemExit(f"model path does not exist: {model_path}")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(str(model_path), local_files_only=True, torch_dtype="auto")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    # RAW tokenization -- no apply_chat_template. `text` already IS the
    # fully templated prefix plus the decoded self-generated continuation,
    # produced by tools/sslm_decode_kv_probe.cpp's own text reconstruction.
    encoded = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    input_ids = encoded["input_ids"].to(device)

    # The guard this script adds (see docstring): the float-side tokenizer
    # must segment this literal text into EXACTLY the token count the
    # engine side already confirmed (by its own round-trip check) for its
    # own tokenizer. A mismatch means the two tokenizers did not segment
    # identically and the "last token" on each side is NOT the same
    # position -- fail loud, do not compare.
    actual_tokens = input_ids.shape[1]
    if actual_tokens != expect_tokens:
        raise SystemExit(
            f"EXPECT_TOKENS_MISMATCH: text-file {args.text_file} expects {expect_tokens} tokens "
            f"(engine-side count), float tokenizer produced {actual_tokens} -- the two "
            f"tokenizers did not segment this text identically; this position is NOT "
            f"comparable and no dump is written"
        )

    hidden_size = model.config.hidden_size
    n_layers = model.config.num_hidden_layers

    captured = fld.capture_incremental(model, input_ids, device)
    print(f"capture: direct forward hooks, token-at-a-time incremental (DynamicCache), "
          f"{n_layers + 1} rows x {hidden_size} hidden_size, {actual_tokens} input tokens (raw text, "
          f"no chat template re-applied)")

    fld.endpoint_self_check(model, captured, input_ids, device)
    print("endpoint_self_check: layer-%d != post-norm (OK, bypass confirmed); "
          "layer-0 == direct embed (OK)" % n_layers)

    fld.interior_row_oracle(model, input_ids, device, captured)
    print(f"interior_row_oracle: all {n_layers + 1} rows bit-identical between hook capture and "
          f"direct-composition capture")

    dump_path = Path(args.dump)
    dump_path.parent.mkdir(parents=True, exist_ok=True)
    with open(dump_path, "wb") as f:
        f.write(struct.pack("<QQQQ", n_layers + 1, hidden_size, fingerprint, 1))
        for idx in range(n_layers + 1):
            f.write(captured[idx].cpu().numpy().astype("float32").tobytes())

    print(f"layer_dump_written: {n_layers + 1} rows x {hidden_size} hidden_size, "
          f"prompt_fingerprint=0x{fingerprint:016X} (engine-supplied, raw-text Fnv1a64) -> {dump_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

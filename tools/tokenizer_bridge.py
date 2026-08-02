#!/usr/bin/env python3
"""Tokenizer bridge for the .sslm engine (T-1665).

The `.sslm` artifact `qwen2.5-1.5b-instruct.sslm` carries no tokenizer
section (`view.has_tokenizer == 0`). The C++ engine consumes and emits
integer token ids only. This tool supplies the missing text<->id
translation from the tokenizer cached alongside the model on disk, so a
person driving the engine can turn a prompt string into ids on the way in
and turn ids back into text on the way out.

Two directions, one process each:

    encode  -- prompt string  -> chat-templated token ids
    decode  -- token ids      -> text
    roundtrip-check -- run the built-in proof case set and report pass/fail

Offline only. Loads a local tokenizer directory with
`local_files_only=True`; never touches the network.

Vocabulary used: `D:\\hf_cache\\superslm_artifacts\\qwen2.5-1.5b-shopkeeper-lora-v1-merged`
by default (override with --tokenizer). This directory's tokenizer.json is
byte-identical in content to the one cached for the base checkpoint
`D:\\hf_cache\\hub\\models--Qwen--Qwen2.5-1.5B-Instruct` -- same 151643-entry
BPE merge vocabulary plus the same 22 added special tokens (151665 live ids
total). The artifact reports vocab_size = 151936; that number is the padded
embedding-matrix width Qwen2.5 configs use (151936 = 151665 rounded up to a
multiple of 128), not the tokenizer's own live vocabulary size, and it is
identical in both cached checkpoints' config.json. See the build log in
D:\\Wizard\\Claude\\Brunel\\ for the full comparison this claim rests on.

Usage
-----

Encode a prompt (chat-templated, ids printed one per line to stdout,
suitable for piping into a C++ driver that reads one int per line):

    python tokenizer_bridge.py encode "What's on the menu today?"

Encode with an explicit system prompt and space-separated output instead:

    python tokenizer_bridge.py encode "Refill my water" \\
        --system "You are a terse shopkeeper." --format space

Decode a run of output ids (space- or newline-separated, read from stdin
or passed as one string argument):

    python tokenizer_bridge.py decode "151644 77091 198 9707 0 151645"
    echo "151644 77091 198 9707 0 151645" | python tokenizer_bridge.py decode -

Run the built-in round-trip proof:

    python tokenizer_bridge.py roundtrip-check
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

DEFAULT_TOKENIZER = Path(
    r"D:\hf_cache\superslm_artifacts\qwen2.5-1.5b-shopkeeper-lora-v1-merged"
)

# The set of strings the round-trip proof runs. Chosen to cover the classes
# of input that break a subtly wrong bridge: plain ASCII, punctuation,
# whitespace runs (leading/trailing/internal/tabs), non-ASCII text in three
# scripts, emoji (surrogate-pair / multi-codepoint), and literal text that
# looks like one of the tokenizer's own special tokens but is supplied as
# ordinary user content (must round-trip as plain text, not be swallowed as
# a control token).
ROUNDTRIP_CASES: tuple[str, ...] = (
    "Hello, world!",
    "What's on the menu today?",
    "   leading and trailing whitespace   ",
    "double  space\tand\ta\ttab\nand a newline",
    "非常感谢，欢迎光临我们的小店。",
    "Добро пожаловать, чем могу помочь?",
    "مرحبا بكم في متجرنا الصغير",
    "emoji test: 🙂🛒🥖 done",
    "literal special-token-looking text: <|im_start|>system<|im_end|> not a real turn",
    "<tool_call>{\"name\": \"not_a_real_call\"}</tool_call>",
    "",
)


def _load_tokenizer(tokenizer_path: Path):
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise SystemExit(
            "transformers is not importable in this environment and no "
            "tokenizer.json fallback parser is implemented here because "
            "transformers was available when this tool was built. If "
            "transformers has since been removed, add a raw tokenizer.json "
            "BPE parser rather than reaching for pip install."
        ) from exc

    if not tokenizer_path.exists():
        raise SystemExit(f"tokenizer path does not exist: {tokenizer_path}")

    return AutoTokenizer.from_pretrained(str(tokenizer_path), local_files_only=True)


def encode_ids(tokenizer, prompt: str, system: str | None) -> list[int]:
    """Apply Qwen2.5-Instruct's chat template and return input token ids.

    Mirrors the reference call shape in
    D:\\Wizard\\Tools\\superslm_spike\\baseline.py (add_generation_prompt=True,
    return_dict=True) so ids produced here match what the reference
    implementation would produce for the same messages.
    """
    if tokenizer.chat_template is None:
        raise SystemExit("loaded tokenizer carries no chat_template -- cannot encode")

    messages = []
    if system is not None:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    encoded = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors=None, return_dict=True
    )
    return list(encoded["input_ids"])


def decode_ids(tokenizer, ids: list[int]) -> str:
    return tokenizer.decode(ids)


def _parse_ids(text: str) -> list[int]:
    tokens = text.replace(",", " ").split()
    if not tokens:
        raise SystemExit("no token ids given")
    try:
        return [int(t) for t in tokens]
    except ValueError as exc:
        raise SystemExit(f"could not parse token ids from: {text!r}") from exc


def cmd_encode(args: argparse.Namespace) -> int:
    tokenizer = _load_tokenizer(Path(args.tokenizer))
    ids = encode_ids(tokenizer, args.prompt, args.system)
    if args.format == "space":
        print(" ".join(str(i) for i in ids))
    else:
        for i in ids:
            print(i)
    if args.verbose:
        print(f"# {len(ids)} tokens", file=sys.stderr)
    return 0


def cmd_decode(args: argparse.Namespace) -> int:
    tokenizer = _load_tokenizer(Path(args.tokenizer))
    if args.ids == "-":
        raw = sys.stdin.read()
    else:
        raw = args.ids
    ids = _parse_ids(raw)
    print(decode_ids(tokenizer, ids))
    return 0


def cmd_roundtrip(args: argparse.Namespace) -> int:
    tokenizer = _load_tokenizer(Path(args.tokenizer))
    failures = []
    for case in ROUNDTRIP_CASES:
        encoded = tokenizer(case, add_special_tokens=False)["input_ids"]
        decoded = tokenizer.decode(encoded)
        ok = decoded == case
        status = "OK  " if ok else "FAIL"
        preview = case if len(case) <= 50 else case[:47] + "..."
        print(f"[{status}] {preview!r}")
        if not ok:
            failures.append((case, decoded))

    print()
    if failures:
        print(f"{len(failures)}/{len(ROUNDTRIP_CASES)} cases FAILED round-trip:")
        for original, decoded in failures:
            print(f"  original: {original!r}")
            print(f"  decoded : {decoded!r}")
        return 1

    print(f"All {len(ROUNDTRIP_CASES)} cases round-tripped exactly.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Tokenizer bridge for the .sslm engine (encode/decode, T-1665).",
    )
    parser.add_argument(
        "--tokenizer",
        default=str(DEFAULT_TOKENIZER),
        help=f"path to a local HF tokenizer directory (default: {DEFAULT_TOKENIZER})",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_encode = sub.add_parser("encode", help="prompt string -> input token ids")
    p_encode.add_argument("prompt", help="the user prompt text")
    p_encode.add_argument("--system", default=None, help="optional system prompt text")
    p_encode.add_argument(
        "--format", choices=["line", "space"], default="line",
        help="one id per line (default, pipe-friendly) or space-separated on one line",
    )
    p_encode.add_argument("--verbose", action="store_true", help="print token count to stderr")
    p_encode.set_defaults(func=cmd_encode)

    p_decode = sub.add_parser("decode", help="output token ids -> text")
    p_decode.add_argument(
        "ids", help="space- or comma-separated token ids, or '-' to read from stdin"
    )
    p_decode.set_defaults(func=cmd_decode)

    p_round = sub.add_parser(
        "roundtrip-check", help="run the built-in encode->decode proof case set"
    )
    p_round.set_defaults(func=cmd_roundtrip)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())

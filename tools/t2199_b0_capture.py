"""Capture/commission T-2199 B0 greedy logit rows with the current sslm_generate binary.

Existing dumps may be reused only after their per-row lowest-index argmax token stream is proven
identical to the frozen T-2193 baseline. Shorter captures may back a longer length cell only when
the baseline itself stopped before the shorter ceiling and the frozen streams are identical.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np


def load_baseline(path: Path, length: int) -> dict[str, dict]:
    rows = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row["max_new_tokens"] == length:
            rows[row["prompt_id"]] = row
    if len(rows) != 48:
        raise RuntimeError(f"{path}: expected 48 baseline rows at length {length}, got {len(rows)}")
    return rows


def dump_argmax_tokens(path: Path) -> list[int]:
    with path.open("rb") as fh:
        header = fh.read(16)
        if len(header) != 16:
            raise RuntimeError(f"{path}: short dump header")
        n_rows, vocab = struct.unpack("<QQ", header)
        if not n_rows or not vocab:
            raise RuntimeError(f"{path}: zero rows/vocab")
    expected_bytes = 16 + n_rows * vocab * 4
    if path.stat().st_size != expected_bytes:
        raise RuntimeError(f"{path}: size {path.stat().st_size}, expected {expected_bytes}")
    rows = np.memmap(path, dtype="<i4", mode="r", offset=16, shape=(n_rows, vocab))
    # numpy.argmax returns the first index at a tie, exactly the engine's lowest-token rule.
    return rows.argmax(axis=1).astype(np.int64).tolist()


def parse_output_tokens(stdout: str) -> list[int]:
    match = re.search(r"^output_tokens \((\d+)\):(.*)$", stdout, re.MULTILINE)
    if not match:
        raise RuntimeError("sslm_generate stdout has no output_tokens row")
    tokens = [int(v) for v in match.group(2).split()]
    if len(tokens) != int(match.group(1)):
        raise RuntimeError("sslm_generate output token header/list mismatch")
    return tokens


def validate_tokens(label: str, actual: list[int], expected: list[int]) -> None:
    if actual != expected:
        first = next((i for i, (a, b) in enumerate(zip(actual, expected)) if a != b),
                     min(len(actual), len(expected)))
        raise RuntimeError(
            f"{label}: token stream differs from frozen baseline at index {first}; "
            f"actual_len={len(actual)} expected_len={len(expected)}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exe", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--tokenizer", required=True, type=Path)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--harness-dir", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--length", required=True, type=int)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--reuse-dir", type=Path)
    parser.add_argument("--reuse-length", type=int, default=100)
    parser.add_argument("--timeout", type=int, default=3600)
    args = parser.parse_args()

    sys.path.insert(0, str(args.harness_dir))
    from prompts import all_prompts  # pylint: disable=import-error,import-outside-toplevel

    baseline = load_baseline(args.baseline, args.length)
    prompts = list(all_prompts())
    if len(prompts) != 48:
        raise RuntimeError(f"expected 48 T-2193 prompts, got {len(prompts)}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    logs = args.out_dir / "stdout"
    logs.mkdir(exist_ok=True)
    metadata_path = args.out_dir / "capture.jsonl"

    with metadata_path.open("a", encoding="utf-8") as metadata:
        for index, (prompt_id, category, _user_text, rendered) in enumerate(prompts, 1):
            expected = list(baseline[prompt_id]["output_tokens"])
            target = args.out_dir / f"{prompt_id}.logits"
            source = args.reuse_dir / target.name if args.reuse_dir else None

            if target.exists():
                validate_tokens(str(target), dump_argmax_tokens(target), expected)
                print(f"[{index}/48] VALID {prompt_id}", flush=True)
                continue

            can_reuse = (source is not None and source.exists() and
                         (args.length == args.reuse_length or len(expected) < args.reuse_length))
            if can_reuse:
                source_tokens = dump_argmax_tokens(source)
                validate_tokens(str(source), source_tokens, expected)
                os.link(source, target)
                print(f"[{index}/48] LINK {prompt_id} ({len(expected)} rows)", flush=True)
                continue

            command = [str(args.exe), str(args.model), str(args.tokenizer), rendered,
                       "--max-new", str(args.length), "--stop", "151645,151643",
                       "--dump-logits", str(target)]
            print(f"[{index}/48] RUN {args.checkpoint} len={args.length} {prompt_id}", flush=True)
            result = subprocess.run(command, capture_output=True, text=True, timeout=args.timeout,
                                    check=False)
            (logs / f"{prompt_id}.stdout.log").write_text(result.stdout, encoding="utf-8")
            (logs / f"{prompt_id}.stderr.log").write_text(result.stderr, encoding="utf-8")
            if result.returncode:
                raise RuntimeError(f"{prompt_id}: sslm_generate exited {result.returncode}: "
                                   f"{result.stderr[-1000:]}")
            actual = parse_output_tokens(result.stdout)
            validate_tokens(prompt_id, actual, expected)
            validate_tokens(str(target), dump_argmax_tokens(target), expected)
            record = {
                "checkpoint": args.checkpoint,
                "length": args.length,
                "prompt_id": prompt_id,
                "category": category,
                "rows": len(actual),
                "dump": str(target),
                "baseline_exact": True,
            }
            metadata.write(json.dumps(record, sort_keys=True) + "\n")
            metadata.flush()

    commission = {
        "checkpoint": args.checkpoint,
        "length": args.length,
        "engine": str(args.exe.resolve()),
        "model": str(args.model.resolve()),
        "tokenizer": str(args.tokenizer.resolve()),
        "baseline": str(args.baseline.resolve()),
        "prompt_population": 48,
        "dumps_present": len(list(args.out_dir.glob("*.logits"))),
        "baseline_exact": True,
    }
    (args.out_dir / "commission.json").write_text(
        json.dumps(commission, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"CAPTURE COMMISSIONED: {args.checkpoint} length={args.length}, 48/48 exact", flush=True)


if __name__ == "__main__":
    main()

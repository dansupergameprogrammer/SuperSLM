"""Run genuine autoregressive damped-greedy confirmation for a B0 candidate row."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path


EOS = {151645, 151643}


def parse_stdout(text: str):
    token_match = re.search(r"^output_tokens \((\d+)\):(.*)$", text, re.MULTILINE)
    stop_match = re.search(r"^stop_reason: (\d+)$", text, re.MULTILINE)
    wall_match = re.search(r"^wall_time_seconds: ([\d.]+)$", text, re.MULTILINE)
    output_match = re.search(r"^output_text_bytes \((\d+)\):\n(.*?)\noutput_text_end$",
                             text, re.MULTILINE | re.DOTALL)
    if not token_match or not stop_match:
        raise RuntimeError("incomplete sslm_generate stdout")
    tokens = [int(v) for v in token_match.group(2).split()]
    if len(tokens) != int(token_match.group(1)):
        raise RuntimeError("output token header/list mismatch")
    output_text = output_match.group(2) if output_match else None
    if output_match and len(output_text.encode("utf-8")) != int(output_match.group(1)):
        raise RuntimeError("output_text byte-count mismatch")
    return tokens, int(stop_match.group(1)), float(wall_match.group(1)) if wall_match else None, output_text


def load_baseline(path: Path, length: int):
    rows = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row["max_new_tokens"] == length:
            rows[row["prompt_id"]] = row
    if len(rows) != 48:
        raise RuntimeError(f"expected 48 baseline rows at length {length}, got {len(rows)}")
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exe", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--tokenizer", required=True, type=Path)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--harness-dir", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--length", required=True, type=int)
    parser.add_argument("--alpha", required=True, type=float)
    parser.add_argument("--order", required=True, type=int)
    parser.add_argument("--top-k", required=True, type=int)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--timeout", default=3600, type=int)
    args = parser.parse_args()

    sys.path.insert(0, str(args.harness_dir))
    from metrics import compute_all_metrics, strip_trailing_stop_ids  # pylint: disable=import-error,import-outside-toplevel
    from prompts import all_prompts  # pylint: disable=import-error,import-outside-toplevel

    baseline = load_baseline(args.baseline, args.length)
    existing = {}
    if args.out.exists():
        for line in args.out.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            existing[row["prompt_id"]] = row
    args.out.parent.mkdir(parents=True, exist_ok=True)
    alpha_q15 = round(args.alpha * 32768)
    mode = "a" if args.out.exists() else "w"
    with args.out.open(mode, encoding="utf-8") as output:
        for index, (prompt_id, category, _user_text, rendered) in enumerate(all_prompts(), 1):
            if prompt_id in existing:
                print(f"[{index}/48] SKIP {prompt_id}", flush=True)
                continue
            command = [str(args.exe), str(args.model), str(args.tokenizer), rendered,
                       "--max-new", str(args.length), "--stop", "151645,151643",
                       "--decode-mode", "damped-greedy", "--alpha-q15", str(alpha_q15),
                       "--anti-lm-order", str(args.order), "--top-k", str(args.top_k)]
            print(f"[{index}/48] RUN a={args.alpha} n={args.order} k={args.top_k} {prompt_id}",
                  flush=True)
            result = subprocess.run(command, capture_output=True, text=True, encoding="utf-8",
                                    errors="strict", timeout=args.timeout, check=False)
            if result.returncode:
                raise RuntimeError(f"{prompt_id}: sslm_generate exited {result.returncode}: "
                                   f"{result.stderr[-1000:]}")
            tokens, stop_reason, wall, output_text = parse_stdout(result.stdout)
            scored, stripped = strip_trailing_stop_ids(tokens, EOS)
            metrics = compute_all_metrics(scored)
            base = baseline[prompt_id]
            row = {
                "checkpoint": args.checkpoint, "length": args.length,
                "alpha": args.alpha, "alpha_q15": alpha_q15, "n": args.order, "k": args.top_k,
                "prompt_id": prompt_id, "category": category,
                "output_tokens": tokens, "output_text": output_text,
                "stop_reason": stop_reason, "wall_time_seconds": wall,
                "n_eos_stripped": stripped, "diverged_from_greedy": tokens != base["output_tokens"],
                **metrics,
            }
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
            output.flush()

    rows = [json.loads(line) for line in args.out.read_text(encoding="utf-8").splitlines()]
    if len(rows) != 48 or any(row["n_generated"] <= 0 for row in rows):
        raise RuntimeError(f"rollout incomplete or contains zero-token rows: {len(rows)}/48")
    summary = {
        "checkpoint": args.checkpoint, "length": args.length,
        "alpha": args.alpha, "alpha_q15": alpha_q15, "n": args.order, "k": args.top_k,
        "generations": 48,
        "reach": sum(row["diverged_from_greedy"] for row in rows),
        "loop_lock_count": sum(row["loop_lock"] for row in rows),
        "rep_3_mean": sum(row["rep_3"] for row in rows) / 48,
        "all_generated_nonempty": True,
    }
    summary_path = args.out.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

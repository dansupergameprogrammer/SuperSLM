#!/usr/bin/env python3
"""T-1800 -- the SLM-side (decode-level) calibration oracle.

WHAT THIS MEASURES. Per-position greedy-decode token agreement between the
compiled int8 engine (`out/sslm_generate.exe`, the real production driver,
T-1664/T-1666/T-1657-family) and the float reference checkpoint, run through
the SAME prompt, SAME chat template, SAME system prompt, SAME token budget,
SAME stop ids -- reusing the exact invocation this project already reviewed
and committed for this purpose (`tools/compare_float.ps1`/T-1679 for the
side-by-side shape, `tools/float_reference_generate.py`/T-1679 and
`tools/logit_margin_report.py`/T-1681 for the per-position agreement
convention). This script is new because neither existing tool aggregates
agreement over a POPULATION with a computed resolving power -- T-1679 is a
single-prompt side-by-side, T-1681 measures 9 fixed prompts for margin
diagnosis, not calibration-arm scoring. This one is built to be re-run,
unmodified, over any future calibration arm's `.sslm` artifact: point
--model at the arm's artifact, everything else is identical.

THE REFERENCE. "Float reference" here means the checkpoint's own VERIFIED
actual compute dtype -- bfloat16 under `torch_dtype="auto"`, independently
confirmed twice in this project (T-1782/D-SLM1212-1225, re-verified again in
T-1777/D-SLM1254-1258 sec.2) -- not the nominal "float32" some earlier
scripts' docstrings name, which is HF's on-disk serialization dtype, never
the compute dtype. Both this oracle and T-1777's encoder-side oracle now use
the SAME reference precision, for the same reason: comparability across the
two oracles' baselines rests on grading against the same ground truth, not
two different ones with different confounds. `tools/float_reference_generate.py`
already resolves `torch_dtype="auto"` correctly; this script calls it
unmodified.

WHAT AGREEMENT MEANS AND DOES NOT MEAN. Per StandardsDocument.md 5.4's
comparability rule: both sides compute the SAME quantity -- the argmax
token id chosen by greedy decoding at each generated position -- over the
SAME prompt population, and the reference (the float checkpoint) takes NO
input derived from the int8 engine or from any calibration constant under
test: it is the unmodified HuggingFace checkpoint, loaded fresh, with no
dependency on this project's quantization code at all. This is therefore a
valid comparison under 5.4's rule (not a case of "the reference shares a
parameter with the implementation").

Divergence between the two paths is not, by itself, a defect (T-1679's own
finding stands: quantization noise near a float top-1/top-2 tie can flip an
argmax with no forward-path error at all -- T-1681 exists to separate that
explanation from a real defect, and is a DIFFERENT, complementary
instrument; this script does not attempt that separation). What this script
measures is the RATE, over a population, with its resolving power -- the
number a calibration arm is scored against, exactly parallel to the
encoder-side oracle's recall@k.

Usage:
    python tools\\t1800_decode_agreement.py --model <path.sslm> --max-new 8 \\
        --out out\\t1800\\decode_agreement_<label>.json
"""
from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DRIVER_EXE = REPO_ROOT / "out" / "sslm_generate.exe"
TOKENIZER_PATH = REPO_ROOT / "tests" / "fixtures" / "qwen2.5-1.5b.tok.sslm"
FLOAT_MODEL_PATH = Path(
    r"D:\hf_cache\hub\models--Qwen--Qwen2.5-1.5B-Instruct\snapshots"
    r"\989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
)
SYSTEM_PROMPT = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
STOP_IDS = [151645, 151643]

# 32 original, self-authored prompts (no external/copyrighted source, matching
# T-1777's own convention for its out-of-distribution set), spanning plain
# factual recall, arithmetic, simple instruction-following, and short
# open-ended asks -- deliberately NOT the same 9 prompts T-1681's
# logit_margin_report.py already used, so this baseline is an independent
# population rather than a re-read of an existing one.
PROMPTS: list[str] = [
    "What is the capital of Italy?",
    "What is the capital of Canada?",
    "What is the capital of Egypt?",
    "Name the smallest planet in the solar system.",
    "Name the longest river in the world.",
    "How many legs does a spider have?",
    "How many continents are there on Earth?",
    "What is 7 plus 9?",
    "What is 14 minus 6?",
    "What is 8 times 6?",
    "What is 100 divided by 4?",
    "What is 23 plus 19?",
    "Write one word that means happy.",
    "Write one word that means fast.",
    "Give me a synonym for the word 'large'.",
    "Name a color that is not red, blue, or green.",
    "Say the days of the week starting from Monday.",
    "List three fruits.",
    "List two animals that live in the ocean.",
    "What language is spoken in Brazil?",
    "What is the freezing point of water in Celsius?",
    "What is the boiling point of water in Celsius?",
    "Who wrote the play Romeo and Juliet?",
    "What gas do plants absorb from the air?",
    "What is the chemical symbol for gold?",
    "How many sides does a hexagon have?",
    "What is the opposite of hot?",
    "What is the opposite of up?",
    "Translate the word 'hello' into French.",
    "Complete the sequence: 2, 4, 6, 8, ...",
    "What year did World War II end?",
    "Name the largest ocean on Earth.",
]


def build_prompt(question: str) -> str:
    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{question}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def run_int8(model_path: Path, question: str, max_new: int) -> dict:
    if not DRIVER_EXE.exists():
        raise SystemExit(f"driver not built: {DRIVER_EXE} (run tools\\build_generate.bat)")
    prompt = build_prompt(question)
    stop_arg = ",".join(str(i) for i in STOP_IDS)
    cmd = [
        str(DRIVER_EXE),
        str(model_path),
        str(TOKENIZER_PATH),
        prompt,
        "--max-new",
        str(max_new),
        "--stop",
        stop_arg,
    ]
    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    wall = time.time() - t0
    if proc.returncode != 0:
        raise SystemExit(f"int8 driver failed (exit {proc.returncode}) on {question!r}:\n{proc.stdout}\n{proc.stderr}")
    m = re.search(r"^output_tokens \((\d+)\):(.*)$", proc.stdout, re.MULTILINE)
    if not m:
        raise SystemExit(f"int8 driver produced no output_tokens line on {question!r}:\n{proc.stdout}")
    ids = [int(x) for x in m.group(2).split()]
    return {"output_ids": ids, "wall_time_seconds": wall}


def run_float(question: str, max_new: int) -> dict:
    cmd = [
        sys.executable,
        str(REPO_ROOT / "tools" / "float_reference_generate.py"),
        question,
        "--system",
        SYSTEM_PROMPT,
        "--max-new",
        str(max_new),
        "--model",
        str(FLOAT_MODEL_PATH),
        "--stop",
        *[str(i) for i in STOP_IDS],
    ]
    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    wall = time.time() - t0
    if proc.returncode != 0:
        raise SystemExit(f"float reference failed (exit {proc.returncode}) on {question!r}:\n{proc.stdout}\n{proc.stderr}")
    m = re.search(r"^output_ids: (.*)$", proc.stdout, re.MULTILINE)
    if not m:
        raise SystemExit(f"float reference produced no output_ids line on {question!r}:\n{proc.stdout}")
    ids = [int(x) for x in m.group(1).split()] if m.group(1).strip() else []
    return {"output_ids": ids, "wall_time_seconds": wall}


def wilson_halfwidth(k: int, n: int, z: float = 1.96) -> float:
    """95% Wilson-score CI half-width on a proportion k/n -- same construction
    T-1777's own oracle (tools/t1777_retrieval_report.py) and this project's
    other resolving-power statements use (StandardsDocument.md 5.4)."""
    if n == 0:
        return float("nan")
    p = k / n
    denom = 1 + z * z / n
    center = p + z * z / (2 * n)
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    lo = (center - margin) / denom
    hi = (center + margin) / denom
    return (hi - lo) / 2


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="path to the .sslm artifact under test (the int8 side)")
    parser.add_argument("--max-new", type=int, default=8, dest="max_new")
    parser.add_argument("--out", default=None, help="path to write the full JSON result set")
    parser.add_argument("--label", default=None, help="label for this run, printed in the summary")
    args = parser.parse_args(argv)

    model_path = Path(args.model)
    if not model_path.exists():
        raise SystemExit(f"model artifact not found: {model_path}")

    label = args.label or model_path.name
    print("=" * 100)
    print(f"T-1800 decode-agreement oracle -- label={label}")
    print(f"model:      {model_path}")
    print(f"reference:  {FLOAT_MODEL_PATH} (bfloat16 compute dtype, verified T-1782/D-SLM1212-1225)")
    print(f"population: {len(PROMPTS)} prompts (self-authored, listed in this script), max_new={args.max_new}")
    print("=" * 100)

    per_prompt = []
    total_positions = 0
    total_agree = 0
    full_sequence_matches = 0

    for i, q in enumerate(PROMPTS):
        int8_res = run_int8(model_path, q, args.max_new)
        float_res = run_float(q, args.max_new)
        int8_ids = int8_res["output_ids"]
        float_ids = float_res["output_ids"]
        n = min(len(int8_ids), len(float_ids))
        agree_positions = [int(int8_ids[p] == float_ids[p]) for p in range(n)]
        n_agree = sum(agree_positions)
        total_positions += n
        total_agree += n_agree
        seq_match = (int8_ids == float_ids)
        full_sequence_matches += int(seq_match)
        per_prompt.append(
            {
                "index": i,
                "question": q,
                "int8_output_ids": int8_ids,
                "float_output_ids": float_ids,
                "n_positions_compared": n,
                "n_agree": n_agree,
                "full_sequence_match": seq_match,
            }
        )
        print(
            f"[{i:2d}] {q!r:55s} int8_n={len(int8_ids):2d} float_n={len(float_ids):2d} "
            f"compared={n:2d} agree={n_agree:2d} seq_match={seq_match}"
        )

    rate = total_agree / total_positions if total_positions else float("nan")
    hw = wilson_halfwidth(total_agree, total_positions)
    seq_rate = full_sequence_matches / len(PROMPTS)

    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)
    print(f"cell: {len(PROMPTS)} prompts x max_new={args.max_new}, positions compared (min of both "
          f"arms' output length per prompt) N={total_positions}, model={model_path.name}, "
          f"reference=bfloat16 Qwen2.5-1.5B-Instruct greedy decode")
    print(f"per-position token agreement: {total_agree}/{total_positions} = {rate:.4f} "
          f"(95% Wilson half-width = {hw:.4f})")
    print(f"full-sequence exact match: {full_sequence_matches}/{len(PROMPTS)} = {seq_rate:.4f}")

    result = {
        "label": label,
        "model": str(model_path),
        "reference_model": str(FLOAT_MODEL_PATH),
        "reference_dtype": "bfloat16 (torch_dtype=auto, verified T-1782/D-SLM1212-1225)",
        "max_new": args.max_new,
        "n_prompts": len(PROMPTS),
        "total_positions": total_positions,
        "total_agree": total_agree,
        "agreement_rate": rate,
        "wilson_95_halfwidth": hw,
        "full_sequence_match_count": full_sequence_matches,
        "full_sequence_match_rate": seq_rate,
        "per_prompt": per_prompt,
    }
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2))
        print(f"\nfull result set written to {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

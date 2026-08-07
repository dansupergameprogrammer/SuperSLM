#!/usr/bin/env python3
"""T-1806 -- the bfloat16 float reference's own run-to-run REPEAT floor on
the SLM (decode) side: is the reference a fixed point of itself, or does it
move between two runs at the identical dtype, same machine, same session?

WHY THIS EXISTS. `t1800_bf16_fp32_floor.py` (T-1800) measures bf16-vs-fp32 --
two DIFFERENT compute precisions of the same checkpoint. That is a
cross-precision comparison, not a repeat measurement, and it is the number
both adopted calibration oracles currently carry as their reference's own
error term (D-SLM1408, D-SLM1429-1433). Two standing records the T-1804
campaign gap audit connects (`Claude/Mendeleev/t1804-campaign-gap-audit-
2026-08-07.md` finding A1; decision D-SLM1435) already point at a gap this
script fills:
  - D-SLM1266: the SAME prompt, SAME bf16 arm, run in two sessions two days
    apart, produced different token content -- filed OBSERVED, not
    investigated.
  - D-SLM1418: an independently-composed bf16-compute forward disagrees with
    the library bf16 reference at Spearman 0.9056-0.9336, while the SAME
    comparison at float32 agrees at 0.9964-0.9983 -- two bf16
    implementations disagree far more than the fp32 pair.

Neither was cited when the two calibration oracles adopted bf16 as their
primary reference (D-SLM1403/1404/1408). This script runs the SAME reference
checkpoint, at the SAME dtype (bfloat16, `torch_dtype="auto"`'s resolved
value for this checkpoint, T-1782/D-SLM1212-1225), TWICE, over the identical
32-prompt population `t1800_decode_agreement.py` uses, in the same process,
same session, same machine -- and reports how often the reference agrees
with ITSELF.

INVOKES, DOES NOT MODIFY, `tools/float_reference_generate.py` (unmodified on
`main`, no `--dtype` flag needed here: the default `torch_dtype="auto"` path
already resolves to bfloat16 for this checkpoint, verified). The PROMPTS
list below is copied BY VALUE from `t1800_decode_agreement.py`
(`D:\\SuperSLM\\.worktrees\\t1800-dual-oracle-baseline\\tools\\
t1800_decode_agreement.py`), textually identical -- verified in the build
record -- matching that project's own convention
(`t1800_bf16_fp32_floor.py`'s own docstring) for a script that must not take
an import-time dependency on a sibling worktree's driver being built.

Usage:
    python tools\\t1806_bf16_repeat_floor.py --max-new 8 --out out\\t1806\\bf16_repeat_floor.json
"""
from __future__ import annotations

import argparse
import json
import math
import platform
import re
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
FLOAT_MODEL_PATH = Path(
    r"D:\hf_cache\hub\models--Qwen--Qwen2.5-1.5B-Instruct\snapshots"
    r"\989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
)
SYSTEM_PROMPT = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
STOP_IDS = [151645, 151643]

# Copied BY VALUE from t1800_decode_agreement.py (D:\SuperSLM\.worktrees\
# t1800-dual-oracle-baseline\tools\t1800_decode_agreement.py), textually
# identical -- verified in the build record, not imported (that worktree's
# driver build is not this script's dependency).
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


def run_float(question: str, max_new: int, generator_script: Path) -> dict:
    cmd = [
        sys.executable,
        str(generator_script),
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
    parser.add_argument("--max-new", type=int, default=8, dest="max_new")
    parser.add_argument("--out", default=None)
    parser.add_argument(
        "--generator",
        default=str(REPO_ROOT / "tools" / "float_reference_generate.py"),
        help="path to float_reference_generate.py to invoke (unmodified; default is this "
             "worktree's own copy on main, which has no --dtype flag and always resolves "
             "torch_dtype='auto' to bfloat16 for this checkpoint)",
    )
    args = parser.parse_args(argv)
    generator_script = Path(args.generator)
    if not generator_script.exists():
        raise SystemExit(f"generator script not found: {generator_script}")

    import torch

    print("=" * 100)
    print("T-1806 bf16 reference REPEAT floor -- reference run against itself, same dtype, twice")
    print(f"reference model: {FLOAT_MODEL_PATH}")
    print(f"generator script invoked: {generator_script}")
    print(f"population: {len(PROMPTS)} prompts, max_new={args.max_new}")
    print(f"torch: {torch.__version__}  cuda_available: {torch.cuda.is_available()}  "
          f"device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}")
    print(f"platform: {platform.platform()}  python: {platform.python_version()}")
    print("=" * 100)

    per_prompt = []
    total_positions = 0
    total_agree = 0
    full_sequence_matches = 0

    for i, q in enumerate(PROMPTS):
        run_a = run_float(q, args.max_new, generator_script)
        run_b = run_float(q, args.max_new, generator_script)
        a_ids = run_a["output_ids"]
        b_ids = run_b["output_ids"]
        n = min(len(a_ids), len(b_ids))
        n_agree = sum(1 for p in range(n) if a_ids[p] == b_ids[p])
        total_positions += n
        total_agree += n_agree
        seq_match = (a_ids == b_ids)
        full_sequence_matches += int(seq_match)
        per_prompt.append(
            {
                "index": i,
                "question": q,
                "run_a_output_ids": a_ids,
                "run_b_output_ids": b_ids,
                "n_positions_compared": n,
                "n_agree": n_agree,
                "full_sequence_match": seq_match,
            }
        )
        print(
            f"[{i:2d}] {q!r:55s} runA_n={len(a_ids):2d} runB_n={len(b_ids):2d} "
            f"compared={n:2d} agree={n_agree:2d} seq_match={seq_match}"
        )

    rate = total_agree / total_positions if total_positions else float("nan")
    hw = wilson_halfwidth(total_agree, total_positions)
    seq_rate = full_sequence_matches / len(PROMPTS)

    print("\n" + "=" * 100)
    print("SUMMARY -- bf16 reference REPEAT floor (run A vs run B, same dtype, same process)")
    print("=" * 100)
    print(f"cell: {len(PROMPTS)} prompts x max_new={args.max_new}, N={total_positions} positions compared")
    print(f"per-position repeat agreement: {total_agree}/{total_positions} = {rate:.4f} "
          f"(95% Wilson half-width = {hw:.4f})")
    print(f"full-sequence exact match: {full_sequence_matches}/{len(PROMPTS)} = {seq_rate:.4f}")

    result = {
        "max_new": args.max_new,
        "n_prompts": len(PROMPTS),
        "total_positions": total_positions,
        "total_agree": total_agree,
        "agreement_rate": rate,
        "wilson_95_halfwidth": hw,
        "full_sequence_match_count": full_sequence_matches,
        "full_sequence_match_rate": seq_rate,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "platform": platform.platform(),
        "python_version": platform.python_version(),
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

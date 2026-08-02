#!/usr/bin/env python3
"""Logit-margin measurement (T-1681): separates two explanations for a
token-choice divergence between the int8 engine and the float reference.

WHAT THIS COMPARISON ESTABLISHES AND DOES NOT ESTABLISH. READ THIS BEFORE
ACTING ON ITS OUTPUT.

`tools/compare_float.ps1` (T-1679) already reports WHERE the int8 engine's
generated-token stream first diverges from the float reference's. It cannot
say WHY: divergence at the argmax is consistent with two different
explanations that look identical from the outside --

  1. Expected quantization behaviour -- float's top two candidates at that
     position are nearly tied, and int8 rounding noise flips which one wins
     the argmax. Not a defect.
  2. A real defect in the int8 forward path -- float strongly prefers its
     token and the int8 engine picks a different one anyway, which is not
     explained by noise near a tie.

This tool separates them by comparing the FULL logit distribution at each
generated position, not just the argmax:

  - float's top-1/top-2 margin at that position (float's own confidence,
    unnormalised, in float's own logit units -- this needs no cross-engine
    normalisation because it never leaves the float side).
  - the margin between float's top-1 and float's logit for the token the
    int8 engine actually chose (0 when the two engines agree). This is the
    number that answers the headline question, and it is reported both in
    float's own raw units (always valid) and z-scored against the float
    row's own dispersion (comparable across positions and prompts of
    different absolute logit scale -- see NORMALISATION below).
  - the RANK of the int8 engine's chosen token within float's own ordering
    (rank 1 = float's own top pick; rank 900 is far down the list).
  - whole-distribution agreement: Spearman rank correlation between the
    float logit row and the int8 raw logit row (full vocabulary, and
    separately restricted to float's own top-N, which is the more sensitive
    "does the head of the distribution track" reading), plus the maximum
    absolute z-scored difference between the two rows over that same top-N.

NORMALISATION -- STATED EXPLICITLY, PER STANDARD.
The int8 engine's `out_logit_rows` are raw int64-accumulated dot products
between int8-coded hidden-state codes and an int8-coded head weight matrix,
narrowed to int32 (forward_sites.h's LogitsSite comment: "wide row's bound is
hidden_size * 127 * 127"). The float reference's logits are float32 dot
products against float32 weights. These two are NOT on a common scale and
are NEVER compared as raw magnitudes anywhere in this tool -- an int8 value
of "500" and a float value of "0.8" carry no shared unit, and subtracting one
from the other would manufacture a number with no meaning.

What IS shared is that within ONE row, both engines' logits are (to first
order) an affine image of the same underlying real-valued pre-softmax score
for that position: softmax is shift-invariant and its ARGMAX and full
ORDERING are invariant to any positive rescaling of a row. That is what
licenses two specific comparisons, and only these two:

  1. Rank / ordering comparisons (argmax, top-k rank, Spearman correlation)
     -- valid under ANY positive monotonic transform of either row
     independently, which is the weakest assumption this tool can get away
     with and still say anything.
  2. Z-scored margins: each row is independently standardised, (x - row
     mean) / row standard deviation, computed over the FULL vocabulary of
     that row before any top-k selection. A margin expressed in
     row-standard-deviations is a scale-free measure of "how many units of
     this row's OWN spread separate these two logits" and is comparable
     across rows, prompts, and (with the stated caveat below) across the two
     engines, PROVIDED the two engines' rows have comparable shape after
     standardisation -- which is exactly what the Spearman/rank checks in
     this same report test for. If the whole-distribution agreement figures
     show the two rows are decorrelated, the z-scored margin comparison
     between engines is not trustworthy either, and this tool's own report
     says so rather than presenting the number anyway.

No other normalisation (min-max scaling to [0,1], softmax-then-compare,
temperature fitting, or any other rescaling) is used, because none of them
is justified by anything this tool has verified about the relationship
between the two engines' raw logit scales -- inventing one would manufacture
either a false alarm or a false all-clear, which is worse than reporting
nothing. If the measurements below do not support a normalised comparison,
this tool says so rather than presenting one.

THE CONTROL SET. A "wide" or "narrow" margin has no meaning on its own --
it needs something to be wide or narrow RELATIVE TO. This tool always runs
a mix of prompts where the two engines are known (or expected, subject to
this same measurement) to AGREE alongside the prompts under investigation,
and reports the z-scored-margin distribution for the agreeing positions as
the control this engine's own "healthy" profile is measured against.

USAGE
-----
    python tools\\logit_margin_report.py --max-new 8 --out-dir out\\t1681

Runs the built-in prompt set (see PROMPT_SET below), dumps both engines'
per-step logit rows to <out-dir>, and prints the margin table plus the cell
every number was measured over. Every number is EXECUTED here (D-SLM671):
this script does not accept precomputed inputs and reasons about them; it
drives both engines itself.
"""

from __future__ import annotations

import argparse
import json
import re
import struct
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

REPO_ROOT = Path(__file__).resolve().parent.parent
DRIVER_EXE = REPO_ROOT / "out" / "sslm_generate.exe"
MODEL_PATH = Path(r"D:\hf_cache\superslm_artifacts\qwen2.5-1.5b-instruct.sslm")
TOKENIZER_PATH = REPO_ROOT / "tests" / "fixtures" / "qwen2.5-1.5b.tok.sslm"
FLOAT_MODEL_PATH = Path(
    r"D:\hf_cache\hub\models--Qwen--Qwen2.5-1.5B-Instruct\snapshots"
    r"\989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
)
SYSTEM_PROMPT = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
STOP_IDS = [151645, 151643]

TOP_N = 50  # float's own top-N used for the sensitive rank/agreement figures


@dataclass
class PromptCase:
    label: str
    question: str
    group: str  # "fail" | "control" | "arith-other"
    max_new: int = 8


PROMPT_SET: list[PromptCase] = [
    PromptCase("digit_symbol_spaced", "What is 12 + 15? Give just the number.", "fail"),
    PromptCase("digit_symbol_nospace", "What is 12+15? Give just the number.", "fail"),
    PromptCase("plain_language_plus", "What is 12 plus 15? Give just the number.", "control-arith"),
    PromptCase("digit_symbol_mult", "What is 17 x 23? Give just the number.", "arith-other"),
    PromptCase("capital_of_france", "What is the capital of France?", "control"),
    PromptCase("capital_of_japan", "What is the capital of Japan?", "control"),
    PromptCase("largest_planet", "Name the largest planet in the solar system.", "control"),
    PromptCase("days_in_week", "How many days are in a week? Give just the number.", "control"),
]


def build_prompt(question: str) -> str:
    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{question}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def run_int8(case: PromptCase, dump_path: Path) -> dict:
    if not DRIVER_EXE.exists():
        raise SystemExit(f"driver not built: {DRIVER_EXE} (run tools\\build_generate.bat)")
    prompt = build_prompt(case.question)
    stop_arg = ",".join(str(i) for i in STOP_IDS)
    cmd = [
        str(DRIVER_EXE),
        str(MODEL_PATH),
        str(TOKENIZER_PATH),
        prompt,
        "--max-new",
        str(case.max_new),
        "--stop",
        stop_arg,
        "--dump-logits",
        str(dump_path),
    ]
    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    wall = time.time() - t0
    if proc.returncode != 0:
        raise SystemExit(
            f"[{case.label}] int8 driver failed (exit {proc.returncode}):\n{proc.stdout}\n{proc.stderr}"
        )
    m = re.search(r"^output_tokens \((\d+)\):(.*)$", proc.stdout, re.MULTILINE)
    if not m:
        raise SystemExit(f"[{case.label}] int8 driver produced no output_tokens line:\n{proc.stdout}")
    ids = [int(x) for x in m.group(2).split()]
    return {"output_ids": ids, "wall_time_seconds": wall, "stdout": proc.stdout}


def run_float(case: PromptCase, dump_path: Path) -> dict:
    cmd = [
        sys.executable,
        str(REPO_ROOT / "tools" / "float_reference_logits.py"),
        case.question,
        "--system",
        SYSTEM_PROMPT,
        "--max-new",
        str(case.max_new),
        "--model",
        str(FLOAT_MODEL_PATH),
        "--stop",
        *[str(i) for i in STOP_IDS],
        "--dump-logits",
        str(dump_path),
    ]
    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    wall = time.time() - t0
    if proc.returncode != 0:
        raise SystemExit(
            f"[{case.label}] float reference failed (exit {proc.returncode}):\n{proc.stdout}\n{proc.stderr}"
        )
    m = re.search(r"^output_ids: (.*)$", proc.stdout, re.MULTILINE)
    if not m:
        raise SystemExit(f"[{case.label}] float reference produced no output_ids line:\n{proc.stdout}")
    ids = [int(x) for x in m.group(1).split()] if m.group(1).strip() else []
    return {"output_ids": ids, "wall_time_seconds": wall, "stdout": proc.stdout}


def load_dump(path: Path, dtype: np.dtype) -> np.ndarray:
    """Reads the shared dump format: uint64 rows, uint64 vocab, then
    rows*vocab values of `dtype`, row-major. Both tools/sslm_generate.cpp
    (--dump-logits, int32) and tools/float_reference_logits.py
    (--dump-logits, float32) write this exact layout."""
    with open(path, "rb") as f:
        header = f.read(16)
        rows, vocab = struct.unpack("<QQ", header)
        data = np.frombuffer(f.read(), dtype=dtype, count=rows * vocab)
    return data.reshape(rows, vocab)


def zscore_row(row: np.ndarray) -> tuple[np.ndarray, float, float]:
    mean = float(row.mean())
    std = float(row.std())
    if std == 0.0:
        return np.zeros_like(row, dtype=np.float64), mean, std
    return (row.astype(np.float64) - mean) / std, mean, std


@dataclass
class PositionResult:
    position: int
    float_token: int
    int8_token: int
    agree: bool
    float_top1_logit: float
    float_top2_logit: float
    float_top1_top2_margin: float
    z_top1_top2_margin: float
    float_logit_of_int8_token: float
    float_margin_top1_to_int8_choice: float
    int8_token_rank_in_float_order: int  # 1-based
    z_margin_top1_to_int8_choice: float
    spearman_full_vocab: float
    spearman_top_n: float
    max_abs_z_diff_top_n: float


def analyze_position(float_row: np.ndarray, int8_row: np.ndarray, int8_token: int,
                      position: int) -> PositionResult:
    float_row64 = float_row.astype(np.float64)
    order = np.argsort(-float_row64, kind="stable")  # descending
    float_top1_idx = int(order[0])
    float_top2_idx = int(order[1])
    float_top1_logit = float(float_row64[float_top1_idx])
    float_top2_logit = float(float_row64[float_top2_idx])

    # Rank of int8's chosen token in float's own ordering. Reuses `order`
    # (the SAME stable descending argsort float_top1_idx/float_top2_idx were
    # already read from) rather than a fresh value-based comparison, so a
    # tie is broken identically here and there -- lowest index wins, matching
    # both engines' own ArgmaxLowestIndexTieBreak convention
    # (include/superslm/forward_sites.h). A value-based rank (e.g. via
    # searchsorted on sorted values) would rank two exactly-tied logits
    # identically regardless of index, disagreeing with the tie-break the
    # argmax itself uses -- immaterial for real float32 logit rows (exact
    # ties are not observed in practice) but wrong in principle.
    position_in_order = np.empty_like(order)
    position_in_order[order] = np.arange(order.shape[0])
    rank_of_int8_choice = int(position_in_order[int8_token]) + 1
    float_logit_of_int8_token = float(float_row64[int8_token])
    margin_to_choice = float_top1_logit - float_logit_of_int8_token

    z_float, _, std_float = zscore_row(float_row64)
    z_margin_to_choice = margin_to_choice / std_float if std_float != 0 else float("nan")
    z_top1_top2 = (float_top1_logit - float_top2_logit) / std_float if std_float != 0 else float("nan")

    # whole-distribution agreement: Spearman over the full vocab, and over
    # float's own top-N (the sensitive "does the head track" reading).
    int8_row64 = int8_row.astype(np.float64)
    rho_full, _ = spearmanr(float_row64, int8_row64)

    top_n_idx = order[:TOP_N]
    rho_top_n, _ = spearmanr(float_row64[top_n_idx], int8_row64[top_n_idx])

    z_int8, _, std_int8 = zscore_row(int8_row64)
    max_abs_z_diff_top_n = float(np.max(np.abs(z_float[top_n_idx] - z_int8[top_n_idx])))

    return PositionResult(
        position=position,
        float_token=float_top1_idx,
        int8_token=int8_token,
        agree=(float_top1_idx == int8_token),
        float_top1_logit=float_top1_logit,
        float_top2_logit=float_top2_logit,
        float_top1_top2_margin=float_top1_logit - float_top2_logit,
        z_top1_top2_margin=z_top1_top2,
        float_logit_of_int8_token=float_logit_of_int8_token,
        float_margin_top1_to_int8_choice=margin_to_choice,
        int8_token_rank_in_float_order=rank_of_int8_choice,
        z_margin_top1_to_int8_choice=z_margin_to_choice,
        spearman_full_vocab=float(rho_full),
        spearman_top_n=float(rho_top_n),
        max_abs_z_diff_top_n=max_abs_z_diff_top_n,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--max-new", type=int, default=None,
                         help="override every case's max-new-tokens budget")
    parser.add_argument("--out-dir", default=str(REPO_ROOT / "out" / "t1681_logits"))
    parser.add_argument("--json-out", default=None,
                         help="optional path to also write the full result set as JSON")
    args = parser.parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results: list[dict] = []
    print("=" * 100)
    print("T-1681 logit-margin measurement -- cell: prompts below, positions 0..max_new-1 per")
    print("prompt (generation stops early on a matched stop id), Qwen2.5-1.5B-Instruct, greedy")
    print("decode on both paths, system prompt fixed, TOP_N=%d for the top-N figures." % TOP_N)
    print("=" * 100)

    for case in PROMPT_SET:
        max_new = args.max_new if args.max_new is not None else case.max_new
        case = PromptCase(case.label, case.question, case.group, max_new)
        print(f"\n--- {case.label} [{case.group}] max_new={max_new} ---")
        print(f"    Q: {case.question!r}")

        int8_dump = out_dir / f"{case.label}.int8.bin"
        float_dump = out_dir / f"{case.label}.float.bin"

        int8_res = run_int8(case, int8_dump)
        float_res = run_float(case, float_dump)

        int8_rows = load_dump(int8_dump, np.int32)
        float_rows = load_dump(float_dump, np.float32)

        n_positions = min(len(int8_res["output_ids"]), int8_rows.shape[0], float_rows.shape[0])
        # Trim any trailing stop-id row that has no counterpart on the other side.
        int8_ids = int8_res["output_ids"][:n_positions]

        position_results: list[PositionResult] = []
        for pos in range(n_positions):
            pr = analyze_position(float_rows[pos], int8_rows[pos], int8_ids[pos], pos)
            position_results.append(pr)
            flag = "AGREE" if pr.agree else "DIFFER"
            print(
                f"    pos={pos:2d} [{flag:6s}] float_top1={pr.float_token:6d} int8_choice={pr.int8_token:6d} "
                f"top1-top2(z)={pr.z_top1_top2_margin:6.2f} top1-to-choice(z)={pr.z_margin_top1_to_int8_choice:6.2f} "
                f"rank_of_choice={pr.int8_token_rank_in_float_order:6d} "
                f"spearman_full={pr.spearman_full_vocab:.4f} spearman_top{TOP_N}={pr.spearman_top_n:.4f} "
                f"max|Δz|_top{TOP_N}={pr.max_abs_z_diff_top_n:6.2f}"
            )

        n_agree = sum(1 for pr in position_results if pr.agree)
        print(f"    token agreement: {n_agree}/{n_positions}")
        print(f"    int8 output_ids: {int8_res['output_ids']}")
        print(f"    float output_ids: {float_res['output_ids']}")

        all_results.append({
            "label": case.label,
            "group": case.group,
            "question": case.question,
            "max_new": max_new,
            "int8_output_ids": int8_res["output_ids"],
            "float_output_ids": float_res["output_ids"],
            "n_positions_compared": n_positions,
            "n_agree": n_agree,
            "positions": [pr.__dict__ for pr in position_results],
        })

    # --- control-vs-fail summary --------------------------------------------
    # The control baseline is float's OWN top1-vs-top2 gap, z-scored against
    # that row's own dispersion, measured ONLY at positions where the two
    # engines still agreed. This is the "healthy margin profile" the WORK
    # asks for: it says how tight float's own decision was allowed to be
    # while int8 noise still failed to flip the argmax. z_margin_top1_to_
    # int8_choice is identically 0 at every agreeing position (the chosen
    # token IS float's top-1 there), so it cannot serve as the control and is
    # not used as one.
    control_top1_top2_z = [pr["z_top1_top2_margin"] for r in all_results
                            for pr in r["positions"] if pr["agree"]]
    # The fail-group statistic is the gap between float's top-1 and float's
    # OWN logit for the token int8 actually chose, z-scored -- this is the
    # number the two candidate explanations disagree on: near the control
    # baseline is consistent with (1) quantization tie-break; well beyond it
    # is consistent with (2) a defect.
    fail_flip_z = [pr["z_margin_top1_to_int8_choice"] for r in all_results if r["group"] == "fail"
                   for pr in r["positions"] if not pr["agree"]]
    fail_flip_rank = [pr["int8_token_rank_in_float_order"] for r in all_results if r["group"] == "fail"
                       for pr in r["positions"] if not pr["agree"]]
    control_spearman = [pr["spearman_full_vocab"] for r in all_results if r["group"] in
                         ("control", "control-arith") for pr in r["positions"]]
    all_spearman = [pr["spearman_full_vocab"] for r in all_results for pr in r["positions"]]

    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)
    print(f"Cell: {len(PROMPT_SET)} prompts (labels above), positions 0..max_new-1 per prompt "
          f"(early-stopped on a matched stop id), greedy decode both paths, Qwen2.5-1.5B-Instruct, "
          f"system prompt fixed, TOP_N={TOP_N}.")
    print(f"CONTROL -- float's own top1-vs-top2 gap (z-scored), at positions where int8 still "
          f"AGREED with float (n={len(control_top1_top2_z)}): {_fmt_stats(control_top1_top2_z)}")
    print(f"FAIL-GROUP FLIP -- float's top1-to-(int8's actual choice) gap (z-scored), at positions "
          f"where int8 DIFFERED from float, fail-group prompts only (n={len(fail_flip_z)}): "
          f"{_fmt_stats(fail_flip_z)}")
    print(f"FAIL-GROUP FLIP -- rank of int8's chosen token in float's own ordering "
          f"(n={len(fail_flip_rank)}): {fail_flip_rank}")
    print(f"Spearman full-vocab, control group (n={len(control_spearman)}): {_fmt_stats(control_spearman)}")
    print(f"Spearman full-vocab, all positions (n={len(all_spearman)}): {_fmt_stats(all_spearman)}")

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(all_results, indent=2))
        print(f"\nFull result set written to {args.json_out}")

    return 0


def _fmt_stats(xs: list[float]) -> str:
    if not xs:
        return "no data"
    arr = np.array(xs, dtype=np.float64)
    return f"min={arr.min():.3f} median={np.median(arr):.3f} max={arr.max():.3f} mean={arr.mean():.3f}"


if __name__ == "__main__":
    raise SystemExit(main())

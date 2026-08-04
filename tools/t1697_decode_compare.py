#!/usr/bin/env python3
"""T-1697 phase 4: the real engine's nine-prompt decode, plus two genuinely
held-out arithmetic prompts, against the float reference -- for the baseline
artifact and for each alpha's migrated artifact. Reuses
`tools/logit_margin_report.py`'s own `run_int8`/`run_float`/`analyze_position`
directly (no reimplementation): only `logit_margin_report.MODEL_PATH` is
overridden per artifact before calling `run_int8`, since that module reads
the module-level constant at call time, not at import time.

The two held-out prompts ("34 + 48", "9 + 86") are never in the 9-prompt
`kv_saturation_report.POPULATION`/`logit_margin_report.PROMPT_SET` used to
derive s_c (activation statistics) or reported in the sweep table -- they
were established as held-out, tuned-on-nothing two-digit arithmetic prompts
by Popper's T-1682 overfit-null probe
(`Claude/Popper/superslm-t1682-logit-margin-flip-attack-2026-08-02.md` §4),
which found them showing zero divergence against the (then-unfixed) int8
baseline; reused here as this task's own held-out set (never seen by this
task's alpha derivation) rather than inventing a fresh pair.

Usage: python tools\\t1697_decode_compare.py <artifact_path> <run_label> [--float-cache-only]
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import logit_margin_report as LMR  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
FLOAT_CACHE_DIR = REPO_ROOT / "out" / "t1697_float_cache"
INT8_OUT_DIR_BASE = REPO_ROOT / "out" / "t1697_decode"

# The 9-prompt population, unchanged -- LMR.PROMPT_SET.
MECH2_LABELS = {"digit_symbol_spaced", "digit_symbol_nospace", "plain_language_plus", "digit_symbol_mult"}

HELDOUT_CASES = [
    LMR.PromptCase("heldout_34plus48", "What is 34 + 48? Give just the number.", "heldout-arith"),
    LMR.PromptCase("heldout_9plus86", "What is 9 + 86? Give just the number.", "heldout-arith"),
]

ALL_CASES = list(LMR.PROMPT_SET) + HELDOUT_CASES


def get_float_result(case) -> tuple[dict, np.ndarray]:
    FLOAT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    dump_path = FLOAT_CACHE_DIR / f"{case.label}.float.bin"
    ids_path = FLOAT_CACHE_DIR / f"{case.label}.float.ids.txt"
    if dump_path.exists() and ids_path.exists():
        ids = [int(x) for x in ids_path.read_text().split()] if ids_path.read_text().strip() else []
        rows = LMR.load_dump(dump_path, np.float32)
        return {"output_ids": ids}, rows
    res = LMR.run_float(case, dump_path)
    ids_path.write_text(" ".join(str(i) for i in res["output_ids"]))
    rows = LMR.load_dump(dump_path, np.float32)
    return res, rows


def run_against_artifact(artifact_path: str, run_label: str) -> list[dict]:
    LMR.MODEL_PATH = Path(artifact_path)
    out_dir = INT8_OUT_DIR_BASE / run_label
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results = []
    for case in ALL_CASES:
        int8_dump = out_dir / f"{case.label}.int8.bin"
        int8_res = LMR.run_int8(case, int8_dump)
        int8_rows = LMR.load_dump(int8_dump, np.int32)
        float_res, float_rows = get_float_result(case)

        n_positions = min(len(int8_res["output_ids"]), int8_rows.shape[0], float_rows.shape[0])
        int8_ids = int8_res["output_ids"][:n_positions]

        position_results = []
        for pos in range(n_positions):
            pr = LMR.analyze_position(float_rows[pos], int8_rows[pos], int8_ids[pos], pos)
            position_results.append(pr)

        n_agree = sum(1 for pr in position_results if pr.agree)
        group = "mech2" if case.label in MECH2_LABELS else (
            "heldout-arith" if case.group == "heldout-arith" else "control")
        spearman_full = [pr.spearman_full_vocab for pr in position_results]
        z_flip = [pr.z_margin_top1_to_int8_choice for pr in position_results if not pr.agree]

        print(f"  [{run_label}] {case.label} ({group}): {n_agree}/{n_positions} agree, "
              f"mean_spearman_full={np.mean(spearman_full):.4f}, "
              f"n_flip={len(z_flip)}, mean_z_flip={np.mean(z_flip) if z_flip else 0.0:.3f}")

        all_results.append(dict(
            label=case.label, group=group, n_positions=n_positions, n_agree=n_agree,
            spearman_full=spearman_full, z_flip=z_flip,
            int8_output_ids=int8_res["output_ids"], float_output_ids=float_res["output_ids"]))
    return all_results


def summarize(run_label: str, results: list[dict]) -> None:
    mech2 = [r for r in results if r["group"] == "mech2"]
    control = [r for r in results if r["group"] == "control"]
    heldout = [r for r in results if r["group"] == "heldout-arith"]

    def agg(rs, key_spearman="spearman_full"):
        all_sp = [v for r in rs for v in r[key_spearman]]
        total_pos = sum(r["n_positions"] for r in rs)
        total_agree = sum(r["n_agree"] for r in rs)
        return total_agree, total_pos, float(np.mean(all_sp)) if all_sp else float("nan")

    m_agree, m_pos, m_sp = agg(mech2)
    c_agree, c_pos, c_sp = agg(control)
    h_agree, h_pos, h_sp = agg(heldout)
    print(f"\n[{run_label}] SUMMARY: mech2 agree={m_agree}/{m_pos} ({100.0*m_agree/m_pos:.1f}%) "
          f"mean_spearman={m_sp:.4f} | control agree={c_agree}/{c_pos} ({100.0*c_agree/c_pos:.1f}%) "
          f"mean_spearman={c_sp:.4f} | heldout-arith agree={h_agree}/{h_pos} "
          f"({100.0*h_agree/h_pos if h_pos else float('nan'):.1f}%) mean_spearman={h_sp:.4f}")


def main() -> int:
    if len(sys.argv) < 3:
        print("usage: t1697_decode_compare.py <artifact_path> <run_label>", file=sys.stderr)
        return 1
    artifact_path, run_label = sys.argv[1], sys.argv[2]
    results = run_against_artifact(artifact_path, run_label)
    summarize(run_label, results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

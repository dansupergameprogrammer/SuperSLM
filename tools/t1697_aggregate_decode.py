#!/usr/bin/env python3
"""T-1697 phase 4 aggregation: reads the ALREADY-CAPTURED int8 dumps
(`out/t1697_decode/<run_label>/<case>.int8.bin`) and the cached float
reference dumps (`out/t1697_float_cache/<case>.float.bin`), and computes
per-run, per-group divergence -- no new engine runs, no re-derivation.
Reuses `logit_margin_report.analyze_position`/`load_dump` unmodified.
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import logit_margin_report as LMR  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
DECODE_DIR = REPO_ROOT / "out" / "t1697_decode"
FLOAT_CACHE_DIR = REPO_ROOT / "out" / "t1697_float_cache"

MECH2_LABELS = {"digit_symbol_spaced", "digit_symbol_nospace", "plain_language_plus", "digit_symbol_mult"}
CONTROL_LABELS = {"capital_of_germany", "capital_of_france", "capital_of_japan", "largest_planet",
                   "days_in_week"}
HELDOUT_LABELS = {"heldout_34plus48", "heldout_9plus86"}
ALL_LABELS = MECH2_LABELS | CONTROL_LABELS | HELDOUT_LABELS


def read_int8_output_ids(path: Path) -> list[int]:
    """int8_res['output_ids'] is not persisted separately by the decode-
    compare run; re-derive the emitted token count from the dump's own row
    count (`load_dump`'s own header) -- the token id sequence itself is not
    needed for this aggregation (only the per-position logit rows and the
    stored `int8_token` argmax, which IS derivable from the int8 dump's own
    logit rows: `sslm_generate.cpp --dump-logits` writes the model's raw
    logits, and greedy decode is `argmax` over them by construction, matching
    `float_reference_logits.py`'s own greedy policy) -- so int8's chosen
    token at each position is `argmax(int8_row)`, never re-run."""
    rows = LMR.load_dump(path, np.int32)
    return [int(np.argmax(row)) for row in rows]


def analyze_run(run_label: str) -> dict:
    run_dir = DECODE_DIR / run_label
    per_label = {}
    for label in ALL_LABELS:
        int8_path = run_dir / f"{label}.int8.bin"
        float_path = FLOAT_CACHE_DIR / f"{label}.float.bin"
        if not int8_path.exists() or not float_path.exists():
            raise FileNotFoundError(f"{run_label}: missing dump for {label!r} "
                                     f"({int8_path.exists()=}, {float_path.exists()=})")
        int8_rows = LMR.load_dump(int8_path, np.int32)
        float_rows = LMR.load_dump(float_path, np.float32)
        int8_ids = read_int8_output_ids(int8_path)
        n_positions = min(len(int8_ids), int8_rows.shape[0], float_rows.shape[0])

        position_results = []
        for pos in range(n_positions):
            pr = LMR.analyze_position(float_rows[pos], int8_rows[pos], int8_ids[pos], pos)
            position_results.append(pr)
        per_label[label] = position_results
    return per_label


def group_stats(per_label: dict, labels: set[str]) -> dict:
    all_pr = [pr for label in labels for pr in per_label[label]]
    n_positions = len(all_pr)
    n_agree = sum(1 for pr in all_pr if pr.agree)
    spearman_full = [pr.spearman_full_vocab for pr in all_pr]
    spearman_top_n = [pr.spearman_top_n for pr in all_pr]
    z_flip = [pr.z_margin_top1_to_int8_choice for pr in all_pr if not pr.agree]
    ranks_flip = [pr.int8_token_rank_in_float_order for pr in all_pr if not pr.agree]
    return dict(
        n_positions=n_positions, n_agree=n_agree,
        agree_rate=n_agree / n_positions if n_positions else float("nan"),
        mean_spearman_full=float(np.mean(spearman_full)) if spearman_full else float("nan"),
        mean_spearman_top_n=float(np.mean(spearman_top_n)) if spearman_top_n else float("nan"),
        n_flip=len(z_flip),
        mean_z_flip=float(np.mean(z_flip)) if z_flip else 0.0,
        max_flip_rank=max(ranks_flip) if ranks_flip else 0,
    )


def main() -> int:
    runs = ["baseline", "alpha0.000", "alpha0.150", "alpha0.180", "alpha0.200", "alpha0.250", "alpha0.300"]
    print(f"{'run':14s} {'group':14s} {'n_pos':>6s} {'agree':>6s} {'agree%':>7s} "
          f"{'meanSpearFull':>14s} {'meanSpearTopN':>14s} {'n_flip':>7s} {'meanZflip':>10s} {'maxRank':>8s}")
    all_results = {}
    for run in runs:
        per_label = analyze_run(run)
        all_results[run] = per_label
        for group_name, labels in [("mech2", MECH2_LABELS), ("control", CONTROL_LABELS),
                                    ("heldout-arith", HELDOUT_LABELS)]:
            gs = group_stats(per_label, labels)
            print(f"{run:14s} {group_name:14s} {gs['n_positions']:6d} {gs['n_agree']:6d} "
                  f"{100.0*gs['agree_rate']:6.1f}% {gs['mean_spearman_full']:14.4f} "
                  f"{gs['mean_spearman_top_n']:14.4f} {gs['n_flip']:7d} {gs['mean_z_flip']:10.3f} "
                  f"{gs['max_flip_rank']:8d}")
        # per-prompt breakdown within mech2 and heldout (small n, worth naming individually)
        for label in sorted(MECH2_LABELS | HELDOUT_LABELS):
            prs = per_label[label]
            n_agree = sum(1 for pr in prs if pr.agree)
            sp = float(np.mean([pr.spearman_full_vocab for pr in prs]))
            print(f"    {run:10s} {label:24s} agree={n_agree}/{len(prs)} mean_spearman_full={sp:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

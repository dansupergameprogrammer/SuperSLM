#!/usr/bin/env python3
"""T-1750: resume `t1750_run_wide_placebo.py` after the session that started
it ended mid-sweep (15 of 24 new channels had landed in the CSV; a 16th,
channel 6491, was build+partial-decode and was cleaned up and re-run here
from scratch rather than trusted half-finished). Identical per-arm pipeline,
appends to the same summary CSV instead of overwriting it.

Usage: python tools\\t1750_run_wide_placebo_resume.py
"""
from __future__ import annotations

import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import t1750_run_wide_placebo as W  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
SUMMARY_PATH = os.path.join(REPO_ROOT, "out", "t1750_wide_placebo_summary.csv")


def already_done() -> set[int]:
    done = set()
    if os.path.exists(SUMMARY_PATH):
        with open(SUMMARY_PATH, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                done.add(int(r["channel"]))
    return done


def main() -> int:
    done = already_done()
    remaining = {c: a for c, a in W.CHOSEN_ALPHAS.items() if c not in done}
    print(f"already done: {sorted(done)}")
    print(f"remaining ({len(remaining)}): {sorted(remaining)}")

    file_exists = os.path.exists(SUMMARY_PATH)
    with open(SUMMARY_PATH, "a", encoding="utf-8") as f:
        if not file_exists:
            f.write("channel,alpha,run_label,mech2_agree,mech2_pos,mech2_pct,mech2_spearman,"
                     "control_agree,control_pos,control_pct,control_spearman\n")
        for channel, alpha in remaining.items():
            print(f"\n########## RESUME channel {channel} alpha={alpha} ##########", flush=True)
            row = W.run_one(channel, alpha)
            f.write(f"{row['channel']},{row['alpha']},{row['run_label']},"
                    f"{row['mech2_agree']},{row['mech2_pos']},{row['mech2_pct']:.4f},"
                    f"{row['mech2_spearman']:.4f},{row['control_agree']},{row['control_pos']},"
                    f"{row['control_pct']:.4f},{row['control_spearman']:.4f}\n")
            f.flush()
    print("\nRESUME COMPLETE.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

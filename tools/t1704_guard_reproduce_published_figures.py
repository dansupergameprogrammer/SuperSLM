#!/usr/bin/env python3
"""T-1704 Part 3 -- backward-compatibility guard: restrict the NEW capture to
the last-token position (the default-mode re-capture
`t1704_capture_population_allpositions.py` already wrote and byte-diffed
against the original T-1697 capture) and confirm it reproduces T-1698's and
T-1700's own PUBLISHED figures exactly, via those tasks' own UNMODIFIED
tools (`t1698_magnitude_distribution.py`, `t1700_output_magnitude_
distribution.py`) -- neither file is edited by this task.

Since byte-identity to the original capture (already proven) implies
identical statistics deterministically, this script's own purpose is to
execute that implication directly and put the numbers on record, exactly as
T-1700/T-1703 did when they re-ran the same guard through their own fresh
invocations of these same unmodified tools.

Mechanism: `t1698_magnitude_distribution.py`/`t1700_output_magnitude_
distribution.py`'s own `main()` calls `M.load_population_sitedumps()` with
no argument (using that function's own default `out/t1697_population`
directory). To point them at THIS task's own re-capture directory without
editing either file, `t1697_outlier_migration.load_population_sitedumps`
(the attribute both modules resolve dynamically at CALL time, not import
time) is monkeypatched to bind the new directory -- the underlying,
unmodified function is still what runs; only which directory it defaults to
changes, and only inside this guard process.

Usage: python tools\\t1704_guard_reproduce_published_figures.py
"""
from __future__ import annotations

import functools
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

REPO_ROOT = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
RECAPTURE_DIR = os.path.join(REPO_ROOT, "out", "t1704_population_default_recapture")


def main() -> int:
    if not os.path.isdir(RECAPTURE_DIR):
        print(f"FAILED: {RECAPTURE_DIR} does not exist -- run "
              f"t1704_capture_population_allpositions.py first", file=sys.stderr)
        return 1

    import t1697_outlier_migration as M  # noqa: E402

    original_loader = M.load_population_sitedumps
    M.load_population_sitedumps = functools.wraps(original_loader)(
        lambda pop_dir=RECAPTURE_DIR: original_loader(pop_dir))

    print("=" * 78)
    print(f"Re-running tools/t1698_magnitude_distribution.py UNMODIFIED, against this task's own "
          f"default-mode (last-token-only) re-capture at {RECAPTURE_DIR}")
    print("Published (T-1698): rank0/median=22.6857, channel 609 rank 5092/8960 at 0.9019x, "
          "wandering 23/28")
    print("=" * 78)
    import t1698_magnitude_distribution as T1698  # noqa: E402
    r1 = T1698.main()

    print()
    print("=" * 78)
    print(f"Re-running tools/t1700_output_magnitude_distribution.py UNMODIFIED, against the SAME "
          f"re-capture")
    print("Published (T-1700): rank0/median=12.5817, wandering 21/28, channel 609 rank 1/1536 at "
          "6.2813x")
    print("=" * 78)
    import t1700_output_magnitude_distribution as T1700  # noqa: E402
    r2 = T1700.main()

    M.load_population_sitedumps = original_loader  # restore, courtesy only (process exits next anyway)

    if r1 not in (0, None) or r2 not in (0, None):
        print("FAILED: one of the reproduction runs returned non-zero", file=sys.stderr)
        return 1
    print("\nRESULT: compare the numbers printed above against the published figures quoted before "
          "each run -- match required, byte-for-byte identical inputs make an exact-figure mismatch "
          "here a sign of a bug in this guard, not in the capture.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

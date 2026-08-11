#!/usr/bin/env python3
"""T-1891 gate G2 -- wide-primitive correctness, the INDEPENDENT-CHECK half.

DISPOSABLE. Branch brunel/t1891-optionG-spike only, never merged.

Reads the CSV `tools/t1891_optiong_gate2_probe.cpp` writes (one row per case: the
inputs the C++ engine's REAL `RopeApplyPairWide` was driven at, and that primitive's
own output). For each row, recomputes the expected rotated pair independently --
Python's native `int` (arbitrary precision, exact) and this repo's own scalar
`rope.rope_apply_pair`/`intmath.rounding_divide_by_pot` (already-pinned reference
primitives, sharing no code with the C++ engine's SignedU128 facility this probes) --
and separately derives whether the TRUE (unrounded-magnitude) rotated result fits
int64_t, by direct comparison against INT64_MIN/INT64_MAX rather than by re-deriving
the engine's own fits-check. Reports PASS/FAIL per row and a summary.

This is gate G2's own "independent code, not the same expression twice"
(StandardsDocument §5.4): the C++ primitive's 128-bit intermediate is a hand-rolled
sign-magnitude U128 facility; this checker's "wide" arithmetic is Python's own
arbitrary-precision integer type, a different runtime with a different
implementation, computing the identical formula from T-1822 §29.4's own citation --
`(x*cos - y*sin, x*sin + y*cos)`, one C3 (ties-away-from-zero) rounding at
ROPE_FRAC_BITS.

Usage: python tools/t1891_optiong_gate2_check.py [csv_path]
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "tests" / "reference"))

from superslm_spike import rope  # noqa: E402

INT64_MIN = -(2**63)
INT64_MAX = 2**63 - 1


def expected(x: int, y: int, cos_q30: int, sin_q30: int) -> tuple[int, int, bool]:
    """The independent reference: exact Python-int rotation via the repo's own
    already-pinned scalar `rope.rope_apply_pair` (arbitrary precision -- no width
    limit of its own), then a direct int64 range check on the TRUE result."""
    rx, ry = rope.rope_apply_pair(x, y, cos_q30, sin_q30)
    fits = (INT64_MIN <= rx <= INT64_MAX) and (INT64_MIN <= ry <= INT64_MAX)
    return rx, ry, fits


def main(argv: list[str]) -> int:
    csv_path = Path(argv[1]) if len(argv) > 1 else REPO_ROOT / "out" / "t1891_gate2_cases.csv"
    if not csv_path.exists():
        print(f"CSV not found: {csv_path} -- run tools\\t1891_optiong_gate2_probe.exe first",
              file=sys.stderr)
        return 2

    total = 0
    failures: list[dict] = []
    label_counts: dict[str, int] = {}
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            total += 1
            label_counts[row["label"]] = label_counts.get(row["label"], 0) + 1
            x = int(row["x"])
            y = int(row["y"])
            cos_q30 = int(row["cos_q30"])
            sin_q30 = int(row["sin_q30"])
            engine_out_x = int(row["out_x"])
            engine_out_y = int(row["out_y"])
            engine_in_domain = row["in_domain"] == "1"

            exp_x, exp_y, exp_fits = expected(x, y, cos_q30, sin_q30)

            if exp_fits != engine_in_domain:
                failures.append({**row, "reason": "in_domain mismatch",
                                  "expected_in_domain": exp_fits})
                continue
            if not exp_fits:
                # Engine correctly refused; its out_x/out_y are documented as
                # meaningful only when in_domain is true (RopeApplyPairWide's own
                # contract), so no value comparison applies here -- REFUSAL is the
                # entire claim for this row, and it just matched.
                continue
            if exp_x != engine_out_x or exp_y != engine_out_y:
                failures.append({**row, "reason": "value mismatch",
                                  "expected_out_x": exp_x, "expected_out_y": exp_y})

    print(f"T-1891 gate G2: {total} cases checked, cell breakdown: {label_counts}")
    if failures:
        print(f"FAIL: {len(failures)} / {total} cases disagreed with the independent reference")
        for f_row in failures[:20]:
            print(f"  {f_row}")
        if len(failures) > 20:
            print(f"  ... and {len(failures) - 20} more")
        return 1

    print(f"PASS: all {total} cases agree with the independent (Python, arbitrary-precision) "
          f"reference -- {label_counts.get('random_sweep', 0)} randomized, "
          f"{total - label_counts.get('random_sweep', 0)} domain-extremity/truncation-corner/"
          f"overflow-forcing")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

#!/usr/bin/env python3
"""Tiled-matmul plan slice 1, cell 10.1: the decision rule over two sslm_gemm_bench builds.

Runs n interleaved pairs (base, candidate) of `sslm_gemm_bench --shape=S --m=M --reps=R --one` on this
host, in one session, and forms r_k = t_base / t_cand per pair (each t the best of R timed calls). It
reports the median, the lower quartile Q1 and the upper quartile Q3 of the n ratios (for n = 15, the 4th
and 12th order statistics) and the verdict of the plan's rule:

  PASS          iff median >= floor and Q1 >= 1.0 and (Q3 - Q1) / median <= 0.25;
  INCONCLUSIVE  iff median >= floor and Q1 >= 1.0 but the spread exceeds 0.25 (repeat the run once;
                a second INCONCLUSIVE is FAIL);
  FAIL          otherwise.

The must-reject is the same run with the D-infinity build as BOTH base and candidate; it must FAIL.

Usage: sslm_gemm_bench_rule.py BASE_BIN CAND_BIN [--shape=1.5B.gate_up] [--m=32] [--n=15] [--reps=5]
                               [--floor=1.4]
Exit code: 0 PASS, 1 FAIL, 2 INCONCLUSIVE.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys


def one(binary: str, shape: str, m: int, reps: int) -> float:
    out = subprocess.run([binary, f"--shape={shape}", f"--m={m}", f"--reps={reps}", "--one"],
                         check=True, capture_output=True, text=True).stdout
    return float(re.search(r"time_ms=([0-9.]+)", out).group(1))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("base")
    ap.add_argument("cand")
    ap.add_argument("--shape", default="1.5B.gate_up")
    ap.add_argument("--m", type=int, default=32)
    ap.add_argument("--n", type=int, default=15)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--floor", type=float, default=1.4)
    a = ap.parse_args()

    ratios = []
    for k in range(a.n):
        tb = one(a.base, a.shape, a.m, a.reps)
        tc = one(a.cand, a.shape, a.m, a.reps)
        ratios.append(tb / tc)
        print(f"pair {k + 1:2d}: base {tb:9.3f} ms  cand {tc:9.3f} ms  r = {tb / tc:.3f}")
    s = sorted(ratios)
    n = len(s)
    median = s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2
    q1 = s[(n + 1) // 4 - 1]       # n = 15: the 4th order statistic
    q3 = s[3 * (n + 1) // 4 - 1]   # n = 15: the 12th
    spread = (q3 - q1) / median
    if median >= a.floor and q1 >= 1.0:
        verdict = "PASS" if spread <= 0.25 else "INCONCLUSIVE"
    else:
        verdict = "FAIL"
    print(f"{a.shape} M={a.m}: n={n} median {median:.3f}  Q1 {q1:.3f}  Q3 {q3:.3f}  "
          f"(Q3-Q1)/median {spread:.3f}  floor {a.floor}  -> {verdict}")
    return {"PASS": 0, "FAIL": 1, "INCONCLUSIVE": 2}[verdict]


if __name__ == "__main__":
    sys.exit(main())

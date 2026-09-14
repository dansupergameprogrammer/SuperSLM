"""T-2700 step 0: compare the independent Python oracle with compiled C++.

Run manually after CMake builds `t2700_fused_k_oracle_{sse2,avx2}_forced`:
`T2700_FUSED_K_ORACLE=<exe> python tools/test_t2700_fused_k_compiled_oracle.py`.
"""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import fused_k_calibration_oracle as oracle


def _vectors():
    for gain in range(-128, 128):
        for code in tuple(range(-64, 0)) + tuple(range(1, 64)):
            yield gain, code, 1 << 30, 0, 1 << 31
    # The product/sum and Q31 boundary rows named by the numeric spec.
    yield 127, 63, 1 << 30, -(1 << 30), 1 << 31
    yield -128, -1, 1 << 30, 0, 1


def _expected(gain: int, code: int, cos_q30: int, sin_q30: int, ratio_q31: int):
    row = oracle.evaluate({
        "gain_code": gain, "k_norm_m": 1 << 30, "k_norm_e": -30,
        "k_codes": [code, 1], "q_codes": [127, -127],
        "cos_q30": cos_q30, "sin_q30": sin_q30,
        "landing_scale_bits": 0x405FC00000000000,  # binary64(127.0)
        "norm_numerator": 1 << 30, "norm_denominator": 1 << 30,
    })
    # Source equals target, so C27's landed values are the rotated integers;
    # score is over the actual clamped int8 operand accepted by C++ Q31.
    landed = [max(-127, min(127, value)) for value in row["rotated"]]
    score = oracle.round_nearest_away(sum(q * k * ratio_q31
                                          for q, k in zip((127, -127), landed)), 1 << 31)
    return [*row["wide"], *row["rotated"], *row["rotated"], score]


def main() -> int:
    exe = os.environ.get("T2700_FUSED_K_ORACLE")
    if not exe:
        raise SystemExit("T2700_FUSED_K_ORACLE must name a compiled forced-tier probe")
    pairs = list(_vectors())
    run = subprocess.run([exe], input="".join("%d %d %d %d %d\n" % pair for pair in pairs),
                         text=True, capture_output=True, check=True)
    rows = [[int(value) for value in line.split()] for line in run.stdout.splitlines()]
    assert len(rows) == len(pairs) == 32_514
    for pair, got in zip(pairs, rows):
        want = _expected(*pair)
        assert got == want, (pair, got, want)
    print(f"compiled-oracle: {len(rows)} census rows, 0 mismatches")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Independent integer oracle and deterministic fixture generator for T-2704.

The oracle owns its integer normalization, reciprocal and round-half-away
operations.  It imports no SuperSLM module and never calls the production
residual site or funnel primitives.  ``generate()`` is deliberately pure so
the regeneration gate can compare its bytes with the checked-in header.
"""
from __future__ import annotations

from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT_PATH = HERE.parent / "sslm_t2704_residual_oracle_fixtures.h"


def abs_unsigned(value: int) -> int:
    return -value if value < 0 else value


def normalize(magnitude: int) -> tuple[int, int]:
    if not 1 <= magnitude <= (1 << 31):
        raise ValueError("selected magnitude is outside T-2704's admitted domain")
    shift = 0
    denominator = magnitude
    while denominator < (1 << 30):
        denominator <<= 1
        shift += 1
    if denominator == (1 << 31):
        denominator >>= 1
        shift -= 1
    return denominator, shift


def reciprocal(denominator: int) -> int:
    """C19's positive integer reciprocal, implemented independently."""
    if not (1 << 30) <= denominator < (1 << 31):
        raise ValueError("normalization did not produce C19's denominator domain")
    return ((1 << 62) + denominator // 2) // denominator


def round_half_away(numerator: int, power: int) -> int:
    if power < 0:
        return numerator << -power
    divisor = 1 << power
    magnitude = abs_unsigned(numerator)
    quotient, remainder = divmod(magnitude, divisor)
    if remainder * 2 >= divisor:
        quotient += 1
    return -quotient if numerator < 0 else quotient


def landing(code: int, source_m: int, target_reciprocal: int, source_e: int,
            target_e: int, target_shift: int) -> int:
    return round_half_away(code * source_m * target_reciprocal,
                           62 - (source_e - target_e + target_shift))


def generate() -> str:
    # This row is the gate's independently stated physical witness.  The
    # selected stream code is retained directly; a current residual site
    # re-lands branch onto stream through an out-of-contract reciprocal and
    # returns final scale (0,19), whereas the physical result is (0,20).
    branch_m = stream_m = 1
    branch_e = stream_e = -17
    branch_code = stream_code = -127
    denominator, shift = normalize(abs_unsigned(stream_m))
    target_reciprocal = reciprocal(denominator)
    raw = landing(branch_code, branch_m, target_reciprocal, branch_e, stream_e, shift)
    oriented = raw
    wide_sum = stream_code + oriented
    assert (denominator, shift, target_reciprocal, raw, oriented, wide_sum) == (
        1 << 30, 30, 1 << 32, -127, -127, -254)

    lines = [
        "// GENERATED FILE. Do not hand-edit.",
        "// Produced by tests/reference/t2704_residual_oracle.py.",
        "// Re-running the generator must reproduce this file byte-for-byte.",
        "#ifndef SUPERSLM_TESTS_SSLM_T2704_RESIDUAL_ORACLE_FIXTURES_H",
        "#define SUPERSLM_TESTS_SSLM_T2704_RESIDUAL_ORACLE_FIXTURES_H",
        "",
        "#include <cstdint>",
        "",
        "namespace superslm_test {",
        "struct T2704ResidualOracleRow {",
        "  int64_t branch_m, branch_e, stream_m, stream_e;",
        "  int8_t branch_code, stream_code;",
        "  int64_t selected_magnitude, normalized_denominator, normalization_shift, reciprocal;",
        "  int64_t selected_direct, nonselected_raw, nonselected_oriented, wide_sum;",
        "  int8_t final_code; int64_t final_scale_m, final_scale_e;",
        "};",
        "inline constexpr T2704ResidualOracleRow kT2704M1ExactTie = {",
        f"  {branch_m}LL, {branch_e}LL, {stream_m}LL, {stream_e}LL, {branch_code}, {stream_code},",
        f"  {abs_unsigned(stream_m)}LL, {denominator}LL, {shift}LL, {target_reciprocal}LL,",
        f"  {stream_code}LL, {raw}LL, {oriented}LL, {wide_sum}LL,",
        "  -127, 0LL, 20LL,",
        "};",
        "}  // namespace superslm_test",
        "",
        "#endif  // SUPERSLM_TESTS_SSLM_T2704_RESIDUAL_ORACLE_FIXTURES_H",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    OUT_PATH.write_text(generate(), encoding="ascii", newline="\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

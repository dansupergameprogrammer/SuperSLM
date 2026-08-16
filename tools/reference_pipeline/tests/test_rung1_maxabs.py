"""C20 boundary fixtures for max_abs_reduce (SuperSLM_Plan.md §6.8 C20, D-SLM50, packet 3).

Tests the int64-widened-before-abs reduction primitive that guards against int32 overflow.
The naive int32 form is genuine UB (four distinct wrong values across six real toolchain runs);
these fixtures are the ones that actually attain the boundary that breaks a narrower
implementation.
"""

import pytest

from conftest import api
import rung1_ref


def test_max_abs_reduce_with_int32_min_present():
    """INT32_MIN present alongside ordinary values.

    Asserts that max_abs_reduce handles the case where INT32_MIN appears in the input.
    This is the value a naive abs() on an int32-typed accumulator cannot represent
    (abs(INT32_MIN) overflows int32) — the exact case D-SLM50 names as genuine UB.
    """
    max_abs_reduce = api("reference_pipeline.intmath", "max_abs_reduce")

    x = [5, rung1_ref.INT32_MIN, -3, 100, 0, 2 ** 30]
    result = max_abs_reduce(x)
    oracle_result = rung1_ref.max_abs_reduce_oracle(x)

    assert result == 2 ** 31
    assert result == oracle_result


def test_max_abs_reduce_all_int32_min():
    """Every element is INT32_MIN.

    Asserts that max_abs_reduce correctly handles repeated INT32_MIN values.
    Distinct from test_max_abs_reduce_with_int32_min_present (single occurrence)
    because an implementation that "fixes" the single-occurrence case with a special
    first/last-element check can still fail when every reduction step sees the same
    extreme value.
    """
    max_abs_reduce = api("reference_pipeline.intmath", "max_abs_reduce")

    x = [-(2 ** 31)] * 8
    result = max_abs_reduce(x)
    oracle_result = rung1_ref.max_abs_reduce_oracle(x)

    assert result == 2 ** 31
    assert result == oracle_result


def test_max_abs_reduce_yields_d_prime_2_31():
    """A token whose max-abs is exactly D' = 2**31 via a boundary NOT at INT32_MIN alone.

    Confirms the reduction reaches 2**31 however it is attained, and confirms the exact
    boundary value structurally. A silently-narrowed int32 result would wrap negative
    rather than equal 2**31, so we assert both the value and the type/bit-length.
    """
    max_abs_reduce = api("reference_pipeline.intmath", "max_abs_reduce")

    x = [2 ** 31 - 1, -(2 ** 31), 1]
    result = max_abs_reduce(x)
    oracle_result = rung1_ref.max_abs_reduce_oracle(x)

    assert result == 2 ** 31
    assert result == oracle_result
    assert type(result) is int
    assert result.bit_length() == 32


def test_max_abs_reduce_all_zero_token_yields_the_guard_value():
    """All-zero token, the max(D, 1) guard.

    D-SLM50: D' = max(D, 1) guards the all-zero token and is exact on that class
    (D=0 is reachable only from it, proven exhaustively).
    """
    max_abs_reduce = api("reference_pipeline.intmath", "max_abs_reduce")

    x = [0, 0, 0, 0]
    result = max_abs_reduce(x)
    oracle_result = rung1_ref.max_abs_reduce_oracle(x)

    assert result == 1
    assert result == oracle_result

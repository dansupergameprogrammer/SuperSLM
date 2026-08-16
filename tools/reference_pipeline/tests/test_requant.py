"""Plan §6.2 — requantization arithmetic pinned to the gemmlowp reference definitions.

`SaturatingRoundingDoublingHighMul` and `RoundingDivideByPOT` are normative scalar
arithmetic: the plan pins "their exact nudge, tie, and saturation semantics". This is
the reason Forge W6 exists — the spike's torch simulation must implement *this*
arithmetic, because torch's default round-half-even measures a different pipeline than
the one that ships.

The oracles here are the mathematical definitions of the two operations, evaluated in
exact rational arithmetic, not a transcription of gemmlowp's nudge-and-truncate trick.
A test that recoded the trick would pass any implementation that recoded it the same
wrong way.

    saturating_rounding_doubling_high_mul(a, b)
        == round_half_toward_positive_infinity(2*a*b / 2**32), saturated to INT32_MAX
    rounding_divide_by_pot(x, e)
        == round_half_away_from_zero(x / 2**e)

The two tie rules DIFFER, and that difference is measured here rather than assumed.
§6.8 pins both, and the cells cite the row:

  * **C2** — `SaturatingRoundingDoublingHighMul` ties toward **+∞**, both signs.
  * **C3** — `RoundingDivideByPOT` ties **away from zero**.

§15 previously glossed both as "ties away from zero", which is wrong for C2; D-SLM36
corrected §15 after this suite measured the difference (record C-3).
"""

from fractions import Fraction

import pytest

from conftest import api

MODULE = "reference_pipeline.intmath"

INT32_MIN = -(2 ** 31)
INT32_MAX = 2 ** 31 - 1


# --- the definitional oracles --------------------------------------------------

def round_half_toward_positive_infinity(value):
    """Nearest integer to `value` (a Fraction); exact .5 ties go toward +infinity."""
    f = Fraction(value)
    return (2 * f.numerator + f.denominator) // (2 * f.denominator)


def round_half_away_from_zero(value):
    """Nearest integer to `value` (a Fraction); exact .5 ties go away from zero."""
    f = Fraction(value)
    magnitude = (2 * abs(f.numerator) + f.denominator) // (2 * f.denominator)
    return magnitude if f.numerator >= 0 else -magnitude


def srdhm_truth(a, b):
    """The defined result of SaturatingRoundingDoublingHighMul(a, b)."""
    exact = round_half_toward_positive_infinity(Fraction(2 * a * b, 2 ** 32))
    return INT32_MAX if exact > INT32_MAX else exact


# ==============================================================================
# SaturatingRoundingDoublingHighMul
# ==============================================================================

def test_srdhm_int32_min_squared_saturates_to_int32_max():
    """§6.2 / §13.1: the INT32_MIN saturation case. 2*(-2^31)^2/2^32 == 2^31, one
    past INT32_MAX — the single input pair whose defined result overflows int32."""
    srdhm = api(MODULE, "saturating_rounding_doubling_high_mul")
    assert srdhm(INT32_MIN, INT32_MIN) == INT32_MAX


def test_srdhm_int32_min_negative_one_does_not_saturate():
    """The neighbouring case that must NOT saturate. An implementation that guards
    on `a == INT32_MIN` alone, rather than on the pair, returns INT32_MAX here."""
    srdhm = api(MODULE, "saturating_rounding_doubling_high_mul")
    assert srdhm(INT32_MIN, -1) == 1


def test_srdhm_positive_tie_rounds_up():
    """2*a*b/2^32 == +0.5 exactly. Ties round toward +infinity: 1, not 0."""
    srdhm = api(MODULE, "saturating_rounding_doubling_high_mul")
    assert Fraction(2 * (1 << 30) * 1, 2 ** 32) == Fraction(1, 2)  # the fixture is a tie
    assert srdhm(1 << 30, 1) == 1


def test_srdhm_negative_tie_rounds_toward_zero():
    """2*a*b/2^32 == -0.5 exactly. gemmlowp's negative nudge is `1 - (1 << 30)`, one
    short of symmetric, so negative ties round toward +infinity (== toward zero
    here): 0, not -1.

    §6.8 C2 — **PINNED**: ties toward +∞, both signs. C3 pins the sibling primitive the
    other way (away from zero), which is why homogenising the two is a live defect and why
    this cell exists rather than deferring to "the obvious rounding".
    """
    srdhm = api(MODULE, "saturating_rounding_doubling_high_mul")
    assert Fraction(2 * -(1 << 30) * 1, 2 ** 32) == Fraction(-1, 2)  # the fixture is a tie
    assert srdhm(-(1 << 30), 1) == 0


@pytest.mark.parametrize("k", range(-40, 41))
def test_srdhm_every_exact_tie_rounds_toward_positive_infinity(k):
    """Sweep exact .5 ties in both directions: a*b == k*2^31 + 2^30 makes
    2ab/2^32 land exactly on k + 0.5. Each must round to k+1."""
    srdhm = api(MODULE, "saturating_rounding_doubling_high_mul")
    ab = k * (1 << 31) + (1 << 30)
    assert Fraction(2 * ab, 2 ** 32) == Fraction(2 * k + 1, 2)  # the fixture is a tie
    assert srdhm(ab, 1) == k + 1 == srdhm_truth(ab, 1)


def test_srdhm_negative_operands_match_the_definition():
    """Sign handling across all four quadrants, against the exact-rational oracle."""
    srdhm = api(MODULE, "saturating_rounding_doubling_high_mul")
    operands = [
        -(2 ** 31), -(2 ** 30) - 7, -(2 ** 24), -12345, -3, -1, 0,
        1, 3, 12345, 2 ** 24, 2 ** 30 + 7, 2 ** 31 - 1,
    ]
    for a in operands:
        for b in operands:
            assert srdhm(a, b) == srdhm_truth(a, b), f"srdhm({a}, {b})"


def test_srdhm_result_always_fits_int32():
    """The 'Saturating' in the name: no input pair may produce an out-of-range value."""
    srdhm = api(MODULE, "saturating_rounding_doubling_high_mul")
    extremes = [INT32_MIN, INT32_MIN + 1, -(2 ** 30), -1, 0, 1, 2 ** 30, INT32_MAX - 1, INT32_MAX]
    for a in extremes:
        for b in extremes:
            result = srdhm(a, b)
            assert isinstance(result, int) and not isinstance(result, bool)
            assert INT32_MIN <= result <= INT32_MAX, f"srdhm({a}, {b}) == {result}"


def test_srdhm_zero_is_absorbing():
    srdhm = api(MODULE, "saturating_rounding_doubling_high_mul")
    for other in [INT32_MIN, -12345, -1, 0, 1, 12345, INT32_MAX]:
        assert srdhm(0, other) == 0
        assert srdhm(other, 0) == 0


def test_srdhm_one_half_identity():
    """b == 2^30 is the multiplicative identity of the doubling-high-mul: it returns
    a/2 rounded. A wrong doubling factor (2^31 or 2^29) fails here."""
    srdhm = api(MODULE, "saturating_rounding_doubling_high_mul")
    for a in [-(2 ** 31), -1000001, -4, 0, 4, 1000001, 2 ** 31 - 1]:
        assert srdhm(a, 1 << 30) == round_half_toward_positive_infinity(Fraction(a, 2))


# ==============================================================================
# RoundingDivideByPOT
# ==============================================================================

def test_rdbypot_exponent_zero_is_the_identity():
    """Shift boundary: exponent 0 divides by 1 and must return x unchanged."""
    rdbypot = api(MODULE, "rounding_divide_by_pot")
    for x in [INT32_MIN, -12345, -2, -1, 0, 1, 2, 12345, INT32_MAX]:
        assert rdbypot(x, 0) == x


def test_rdbypot_max_exponent_shift_boundary():
    """Shift boundary: exponent 31 is the widest defined shift for an int32 input."""
    rdbypot = api(MODULE, "rounding_divide_by_pot")
    assert rdbypot(INT32_MIN, 31) == -1                 # exactly -1.0
    assert rdbypot(INT32_MAX, 31) == 1                  # 0.99999... -> 1
    assert rdbypot(0, 31) == 0
    assert rdbypot(1 << 30, 31) == 1                    # exactly +0.5, ties away -> 1
    assert rdbypot(-(1 << 30), 31) == -1                # exactly -0.5, ties away -> -1


def test_rdbypot_positive_tie_rounds_away_from_zero():
    """6/4 == +1.5 exactly. Ties away from zero: 2, not 1."""
    rdbypot = api(MODULE, "rounding_divide_by_pot")
    assert rdbypot(6, 2) == 2


def test_rdbypot_negative_tie_rounds_away_from_zero():
    """-6/4 == -1.5 exactly. Ties away from zero: -2, not -1.

    This is the direction round-half-even (torch's default) and round-half-up both
    get wrong, and the direction in which this primitive's tie rule differs from
    SaturatingRoundingDoublingHighMul's. Forge W6's whole subject.
    """
    rdbypot = api(MODULE, "rounding_divide_by_pot")
    assert rdbypot(-6, 2) == -2


def test_rdbypot_half_even_would_disagree_here():
    """A round-half-even implementation returns 2 for both of these (2 is even).
    Ties-away-from-zero returns 1 and 3. Pins the tie rule against torch's default."""
    rdbypot = api(MODULE, "rounding_divide_by_pot")
    assert rdbypot(2, 2) == 1     # +0.5 -> 1  (half-even would give 0)
    assert rdbypot(10, 2) == 3    # +2.5 -> 3  (half-even would give 2)
    assert rdbypot(-2, 2) == -1   # -0.5 -> -1 (half-even would give 0)
    assert rdbypot(-10, 2) == -3  # -2.5 -> -3 (half-even would give -2)


@pytest.mark.parametrize("exponent", [1, 2, 3, 8, 15, 30, 31])
def test_rdbypot_every_exact_tie_rounds_away_from_zero(exponent):
    """Sweep exact .5 ties in both directions at each shift width."""
    rdbypot = api(MODULE, "rounding_divide_by_pot")
    half = 1 << (exponent - 1)
    for k in range(-24, 25):
        x = k * (1 << exponent) + half
        if not (INT32_MIN <= x <= INT32_MAX):
            continue
        truth = Fraction(x, 1 << exponent)
        assert truth.denominator == 2  # the fixture is a tie
        assert rdbypot(x, exponent) == round_half_away_from_zero(truth), f"rdbypot({x}, {exponent})"


@pytest.mark.parametrize("exponent", [0, 1, 2, 5, 16, 30, 31])
def test_rdbypot_matches_the_definition_over_the_domain(exponent):
    """Exhaustive-in-shape sweep against the exact-rational oracle, including
    negative operands and both saturation boundaries."""
    rdbypot = api(MODULE, "rounding_divide_by_pot")
    step = (1 << exponent) or 1
    values = [INT32_MIN, INT32_MIN + 1, INT32_MAX - 1, INT32_MAX, 0]
    for k in range(-5, 6):
        for offset in range(-3, 4):
            candidate = k * step + offset
            if INT32_MIN <= candidate <= INT32_MAX:
                values.append(candidate)
    for x in values:
        expected = round_half_away_from_zero(Fraction(x, 1 << exponent))
        assert rdbypot(x, exponent) == expected, f"rdbypot({x}, {exponent})"


def test_rdbypot_is_monotone_non_decreasing():
    """A defect in the negative-branch threshold shows up as a non-monotone step."""
    rdbypot = api(MODULE, "rounding_divide_by_pot")
    previous = None
    for x in range(-600, 601):
        current = rdbypot(x, 4)
        if previous is not None:
            assert current >= previous, f"rdbypot({x}, 4) < rdbypot({x - 1}, 4)"
        previous = current


# ==============================================================================
# The composed requant — §6.2 names the pair as the requantization arithmetic
# ==============================================================================

def test_multiply_by_quantized_multiplier_composes_the_two_primitives():
    """§6.2's requant is SaturatingRoundingDoublingHighMul followed by
    RoundingDivideByPOT. The composite must be exactly that pair, in that order —
    a single fused rounding would be a different pipeline."""
    mbqm, srdhm, rdbypot = api(
        MODULE,
        "multiply_by_quantized_multiplier",
        "saturating_rounding_doubling_high_mul",
        "rounding_divide_by_pot",
    )
    cases = [
        (1234567, 1073741824, 0), (1234567, 1073741824, 7), (-1234567, 1073741824, 7),
        (INT32_MAX, 2147483647, 31), (INT32_MIN, 2147483647, 31), (INT32_MIN, 1, 0),
        (0, 1073741824, 12), (-1, 1073741824, 1), (1, 1073741824, 1),
    ]
    for x, multiplier, shift in cases:
        assert mbqm(x, multiplier, shift) == rdbypot(srdhm(x, multiplier), shift), (x, multiplier, shift)


def test_multiply_by_quantized_multiplier_result_fits_int32():
    mbqm = api(MODULE, "multiply_by_quantized_multiplier")
    for x in [INT32_MIN, -1, 0, 1, INT32_MAX]:
        for multiplier in [1, 1 << 30, INT32_MAX]:
            for shift in [0, 1, 15, 31]:
                result = mbqm(x, multiplier, shift)
                assert isinstance(result, int) and not isinstance(result, bool)
                assert INT32_MIN <= result <= INT32_MAX, (x, multiplier, shift, result)


# ==============================================================================
# Purity — the arithmetic is integer, and no float leaks into it (Forge W6)
# ==============================================================================

def test_requant_primitives_return_exact_ints_not_floats():
    """W6: the simulation must implement the shipped integer arithmetic. A float or
    numpy/torch scalar leaking out is the defect this cell exists to catch — it
    would carry round-half-even semantics through the rest of the pipeline."""
    srdhm, rdbypot, mbqm = api(
        MODULE,
        "saturating_rounding_doubling_high_mul",
        "rounding_divide_by_pot",
        "multiply_by_quantized_multiplier",
    )
    for result in [srdhm(123456789, 987654321), rdbypot(-12345, 3), mbqm(123456, 1 << 30, 4)]:
        assert type(result) is int, f"{result!r} is {type(result).__name__}, not a plain int"


def test_requant_primitives_are_pure():
    """Same inputs, same outputs — no accumulated state between calls."""
    srdhm, rdbypot = api(MODULE, "saturating_rounding_doubling_high_mul", "rounding_divide_by_pot")
    first = [srdhm(a, 1 << 30) for a in range(-50, 51)] + [rdbypot(x, 3) for x in range(-50, 51)]
    second = [srdhm(a, 1 << 30) for a in range(-50, 51)] + [rdbypot(x, 3) for x in range(-50, 51)]
    assert first == second

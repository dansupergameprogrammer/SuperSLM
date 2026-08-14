"""Wilson score interval sanity cells -- reference values and edge cases, so P2's precision/recall
CIs are trusted before they gate anything (plan Sec.8 P2's own CI requirement)."""

import math

import pytest

from ..judge_a.stats import wilson_interval


def test_known_reference_value_50_of_100():
    # 50/100 at 95% -- a standard textbook Wilson interval, verified against the closed-form
    # computation independently (not against this module's own arithmetic): center ~0.5, half-width
    # ~0.098 for z=1.96.
    result = wilson_interval(50, 100)
    assert math.isclose(result.point_estimate, 0.5)
    assert 0.40 < result.lower < 0.41
    assert 0.59 < result.upper < 0.60


def test_perfect_score_interval_does_not_touch_1_with_finite_n():
    result = wilson_interval(100, 100)
    assert result.point_estimate == 1.0
    assert result.upper == 1.0
    assert result.lower < 1.0  # a finite sample never certifies p=1 exactly


def test_zero_score_interval_does_not_go_below_0():
    # Executed: with p_hat=0 the algebraic lower bound is exactly 0, but center and half_width are
    # computed via different floating-point paths (one direct, one through sqrt), so the raw
    # subtraction lands at ~3.47e-18, not bit-exact 0 -- the max(0.0, ...) clamp exists precisely so
    # the reported interval never goes negative, not so it lands at bit-exact 0.
    result = wilson_interval(0, 100)
    assert result.point_estimate == 0.0
    assert 0.0 <= result.lower < 1e-10
    assert result.upper > 0.0


def test_lower_bound_increases_with_more_successes_at_fixed_n():
    a = wilson_interval(70, 100)
    b = wilson_interval(90, 100)
    assert b.lower > a.lower


def test_interval_narrows_as_n_grows_at_fixed_proportion():
    small = wilson_interval(85, 100)
    large = wilson_interval(850, 1000)
    assert (large.upper - large.lower) < (small.upper - small.lower)


def test_n_zero_rejected():
    with pytest.raises(ValueError):
        wilson_interval(0, 0)


def test_successes_out_of_range_rejected():
    with pytest.raises(ValueError):
        wilson_interval(11, 10)
    with pytest.raises(ValueError):
        wilson_interval(-1, 10)


def test_085_bar_example_from_plan_is_computable():
    """A sanity check on the exact shape P2's gate uses: a judge scoring 95/100 correct clears the
    0.85 lower-CI-bound bar (executed: lower=0.8882); a judge scoring 90/100 does not, despite a
    0.90 point estimate (executed: lower=0.8256) -- the point estimate alone is not the gate, the
    lower CI bound is (illustrative, not a claim about any real judge's performance)."""

    clears = wilson_interval(95, 100)
    assert clears.lower >= 0.85

    point_estimate_looks_fine_but_ci_does_not_clear = wilson_interval(90, 100)
    assert point_estimate_looks_fine_but_ci_does_not_clear.point_estimate == 0.90
    assert point_estimate_looks_fine_but_ci_does_not_clear.lower < 0.85

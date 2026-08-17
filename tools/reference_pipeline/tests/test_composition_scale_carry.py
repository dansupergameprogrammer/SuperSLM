"""C26 machinery: canonical carried scale and residual reconciliation primitives.

SuperSLM_Plan.md §6.8 C26 (D-SLM56, 2026-07-16), packet E6.
Charter: canonical half-even offline emission, runtime scale products (high-mul +
renormalization), and the residual reconciliation composite over adversarial scale pairs.
"""

import random
from fractions import Fraction

from conftest import api
import rung1_ref
from composition_ref import (
    canonical_scale_oracle,
    scale_mul_oracle,
    residual_reconcile_oracle,
)


def test_canonical_scale_rounds_the_mantissa_half_even():
    """C26 offline emission: mantissa rounded HALF-EVEN (C14's precedent, cited by C26).

    The canonical scale is (m, e) with m ∈ [2^30, 2^31), m rounded half-even per the offline
    constant-emission rule. This cell asserts the tie fixtures and the exact power-of-two case,
    then sweeps 2,000 random values (seeded, deterministic) against the oracle.
    """
    canonical_scale = api("reference_pipeline.pipeline", "canonical_scale")

    # --- Tie fixtures (executed) ---
    # Tie 1: (2^31 + 1) / 2 -> rounds to EVEN mantissa 2^30
    # The boundary is at m_rounded = 2^30.5, so at this tie the even neighbor is DOWN.
    tie1_value = Fraction(2**31 + 1, 2)
    m1, e1 = canonical_scale(tie1_value)
    oracle_m1, oracle_e1 = canonical_scale_oracle(tie1_value)
    assert m1 == 2**30, f"tie 1: got m={m1}, expected 2^30 (even)"
    assert m1 == oracle_m1 and e1 == oracle_e1

    # Tie 2: (2^31 + 3) / 2 -> rounds to EVEN mantissa 2^30 + 2
    # At this tie, the even neighbor is UP.
    tie2_value = Fraction(2**31 + 3, 2)
    m2, e2 = canonical_scale(tie2_value)
    oracle_m2, oracle_e2 = canonical_scale_oracle(tie2_value)
    assert m2 == 2**30 + 2, f"tie 2: got m={m2}, expected 2^30+2 (even)"
    assert m2 == oracle_m2 and e2 == oracle_e2

    # Exact power of two: 1/128 = 2^-7
    exact_value = Fraction(1, 128)
    m_exact, e_exact = canonical_scale(exact_value)
    oracle_m_exact, oracle_e_exact = canonical_scale_oracle(exact_value)
    assert (m_exact, e_exact) == (2**30, -37), f"exact: got ({m_exact}, {e_exact})"
    assert m_exact == oracle_m_exact and e_exact == oracle_e_exact

    # --- Seeded sweep (2,000 random Fractions) ---
    rng = random.Random(6001)
    for _ in range(2000):
        a = rng.randint(1, 2**40)
        b = rng.randint(1, 2**40)
        frac_val = Fraction(a, b)

        m, e = canonical_scale(frac_val)
        oracle_m, oracle_e = canonical_scale_oracle(frac_val)

        # Assert implementation == oracle
        assert m == oracle_m and e == oracle_e, \
            f"mismatch at F({a},{b}): got ({m},{e}), oracle ({oracle_m},{oracle_e})"

        # Assert postcondition: m ∈ [2^30, 2^31)
        assert 2**30 <= m < 2**31, \
            f"F({a},{b}): m={m} outside [2^30, 2^31)"


def test_scale_mul_matches_the_oracle_and_renormalizes_exactly():
    """C26 runtime scale product: one C2-class high-mul on mantissas, exact left-shift
    renormalization into [2^30, 2^31).

    The executed renormalization boundary: (2^30, -20)^2 -> (2^30, -10),
    representing exactly 2^20. Also sweeps 5,000 random scale pairs.
    """
    scale_mul = api("reference_pipeline.intmath", "scale_mul")

    # --- Boundary fixture (executed) ---
    # scale_mul(2^30, -20, 2^30, -20) should renormalize to (2^30, -10).
    # High-mul lands at 2^29, below the floor, so left-shift by 1.
    m_result, e_result = scale_mul(2**30, -20, 2**30, -20)
    oracle_m, oracle_e = scale_mul_oracle(2**30, -20, 2**30, -20)
    assert m_result == 2**30 and e_result == -10, \
        f"boundary: got ({m_result}, {e_result}), expected (2^30, -10)"
    assert m_result == oracle_m and e_result == oracle_e

    # --- Seeded sweep (5,000 random (m, e) pairs) ---
    rng = random.Random(6002)
    for _ in range(5000):
        m_a = rng.randint(2**30, 2**31 - 1)
        e_a = rng.randint(-64, 0)
        m_b = rng.randint(2**30, 2**31 - 1)
        e_b = rng.randint(-64, 0)

        m, e = scale_mul(m_a, e_a, m_b, e_b)
        oracle_m, oracle_e = scale_mul_oracle(m_a, e_a, m_b, e_b)

        # Assert implementation == oracle
        assert m == oracle_m and e == oracle_e, \
            f"mismatch at ({m_a},{e_a})*({m_b},{e_b}): got ({m},{e}), oracle ({oracle_m},{oracle_e})"

        # Assert postcondition: m ∈ [2^30, 2^31)
        assert 2**30 <= m < 2**31, \
            f"({m_a},{e_a})*({m_b},{e_b}): m={m} outside [2^30, 2^31)"


def test_residual_reconcile_matches_the_oracle_over_adversarial_scale_pairs():
    """C26 residual reconciliation composite: round_half_away_from_zero(
    (branch_code · m_b · R_h) / 2^(62 − (e_b − e_h))).

    Sweeps 5,000 adversarial tuples with extreme ratios (decisiveness guard: at least one
    pair has |e_b - e_h| >= 15). Includes mantissa edges 2^30 and 2^31-1.
    """
    residual_reconcile = api("reference_pipeline.intmath", "residual_reconcile")

    # --- Seeded adversarial sweep ---
    rng = random.Random(6003)
    found_extreme_ratio = False

    for _ in range(5000):
        branch_code = rng.randint(-127, 127)
        m_b = rng.randint(2**30, 2**31 - 1)
        m_h = rng.randint(2**30, 2**31 - 1)
        e_b = rng.randint(-40, -20)
        e_h = rng.randint(-40, -20)

        # Compute R_h via the oracle
        r_h = rung1_ref.reciprocal_oracle(m_h)

        result = residual_reconcile(branch_code, m_b, r_h, e_b, e_h)
        oracle_result = residual_reconcile_oracle(branch_code, m_b, r_h, e_b, e_h)

        assert result == oracle_result, \
            f"mismatch at code={branch_code}, m_b={m_b}, m_h={m_h}, e_b={e_b}, e_h={e_h}: " \
            f"got {result}, oracle {oracle_result}"

        # Track decisiveness: have we seen an extreme ratio?
        if abs(e_b - e_h) >= 15:
            found_extreme_ratio = True

    # Decisiveness guard: the sweep MUST contain at least one extreme ratio
    assert found_extreme_ratio, "sweep failed decisiveness guard: no pair with |e_b - e_h| >= 15"

    # --- Named mantissa edges ---
    for m_b_edge in [2**30, 2**31 - 1]:
        for m_h_edge in [2**30, 2**31 - 1]:
            r_h = rung1_ref.reciprocal_oracle(m_h_edge)
            result = residual_reconcile(50, m_b_edge, r_h, -30, -25)
            oracle_result = residual_reconcile_oracle(50, m_b_edge, r_h, -30, -25)
            assert result == oracle_result, \
                f"mantissa edge ({m_b_edge}, {m_h_edge}): got {result}, oracle {oracle_result}"


def test_residual_reconcile_at_ratio_one_preserves_the_branch_code():
    """C26 at the charter's named ratio-1 case: m_b == m_h, e_b == e_h.

    At ratio 1, the rescale is the identity within C26's rounding, and the reconciled value
    equals the branch code (after rounding). The executed fixture shows this lands exactly.
    This is a measured property: in general, R_h carries C19's own rounding, so the result
    can differ from the code; the equality at these fixtures proves the identity at this
    regime.
    """
    residual_reconcile = api("reference_pipeline.intmath", "residual_reconcile")

    # Executed fixture: m = 2^30 + 7, code 100
    m = 2**30 + 7
    r_h = rung1_ref.reciprocal_oracle(m)
    result = residual_reconcile(100, m, r_h, -25, -25)
    oracle_result = residual_reconcile_oracle(100, m, r_h, -25, -25)

    assert result == 100, \
        f"ratio-1 fixture (m={m}, code=100): got {result}, expected 100"
    assert result == oracle_result

    # Spread of codes at several equal (m, e) pairs
    for code in [-127, -100, -1, 0, 1, 100, 127]:
        for m_pair in [2**30, 2**30 + 1024, 2**31 - 1]:
            r_h = rung1_ref.reciprocal_oracle(m_pair)
            result = residual_reconcile(code, m_pair, r_h, -30, -30)
            oracle_result = residual_reconcile_oracle(code, m_pair, r_h, -30, -30)

            assert result == oracle_result
            # At ratio 1, the branch code is preserved (identity property)
            assert result == code, \
                f"ratio-1 (m={m_pair}, code={code}): got {result}, expected {code}"


def test_residual_reconcile_at_the_wide_intermediate_boundary():
    """C26's ~2^70 wide intermediate boundary: C20/C22's width lesson at C26's own composite.

    The executed fixture: m_h = 2^30 (gives R_h = 2^32 exactly, no rounding),
    branch_code = 127, m_b = 2^31 - 1, (e_b, e_h) = (-20, -21).
    The numerator 127 · (2^31−1) · 2^32 is exactly 70 bits; an int64-only
    implementation cannot produce the reconciled value 508.
    """
    residual_reconcile = api("reference_pipeline.intmath", "residual_reconcile")

    # Setup the wide-boundary fixture
    m_h = 2**30
    r_h = rung1_ref.reciprocal_oracle(m_h)

    # Assert R_h is exactly 2^32 (2^62 / 2^30)
    assert r_h == 2**32, f"R_h mismatch: got {r_h}, expected 2^32"

    branch_code = 127
    m_b = 2**31 - 1
    e_b = -20
    e_h = -21

    # The numerator should be exactly 70 bits
    numerator = branch_code * m_b * r_h
    assert numerator.bit_length() == 70, \
        f"numerator bit_length: got {numerator.bit_length()}, expected 70"

    # Execute the reconciliation
    result = residual_reconcile(branch_code, m_b, r_h, e_b, e_h)
    oracle_result = residual_reconcile_oracle(branch_code, m_b, r_h, e_b, e_h)

    # The oracle-recomputed value is 508
    assert oracle_result == 508, \
        f"oracle result: got {oracle_result}, expected 508"

    # Implementation must match the oracle
    assert result == oracle_result, \
        f"wide boundary: got {result}, oracle {oracle_result}"

    # Sanity: int64 range is [-(2^63), 2^63 - 1]; 2^70 exceeds it
    assert numerator >= 2**63, \
        f"numerator {numerator} does not exceed int64 range; test fixture is wrong"

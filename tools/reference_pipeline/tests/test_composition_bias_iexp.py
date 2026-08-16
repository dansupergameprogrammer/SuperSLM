"""C28 bias reconciliation + C30 i-exp constant derivation (unit cells).

Construction C28/C30 (SuperSLM_Plan.md §6.8, D-SLM56): C28 CORRECTED formula
(second-pass S2-1, the exponent ADDS e_a and CARRIES the Q-format), and C30
R-COMPOSED derivation (second-pass W2-4, conformance recomputes the R-COMPOSED
formula; the exact-division twin is the negative-control fixture, never the
equality target). See packet-07.md.

Charter: §17 G-8 item 15 (C28 bias boundary, sign-inverted negative control)
and item 16 (C30 cells, R-composed conformance, twin negative control).
"""

import random

import pytest

from conftest import api
from composition_ref import (
    bias_reconcile_oracle,
    bias_reconcile_sign_inverted,
    iexp_constants_oracle,
    iexp_constants_exact_twin,
    FineScaleRejected,
    reciprocal_oracle,
)

MODULE_INTMATH = "reference_pipeline.intmath"

# Placeholder coefficient integers at Q30, POSITIVE per second-pass N2-5 (C30's floor-shift
# claim holds on positive operands only). VALUES ride C7/C8's S2 pin — these are fixture
# inputs to the FORMULA under test, not normative constants.
LN2_Q30, B_Q30, CA_Q30 = 744261117, 1452772687, 1030312935


# ==============================================================================
# Cell 1: C28 exactness, both signs, scale extremes, exact ties
# ==============================================================================


def test_bias_reconcile_matches_the_corrected_formula_on_both_signs():
    """C28 exactness: the corrected bias reconciliation formula
    b'[j] = round_half_away_from_zero((B[j] * R_a) / 2^(q_B + 62 + e_a))
    holds exactly across a seeded sweep of both positive and negative signs, and
    on exact-tie fixtures where the away-from-zero rule is load-bearing.
    """
    bias_reconcile = api(MODULE_INTMATH, "bias_reconcile")

    # Seeded sweep: 5,000 tuples with both signs
    rng = random.Random(7001)
    b_values = [rng.randint(-(2 ** 40), 2 ** 40) for _ in range(5000)]
    q_b_values = [rng.choice([20, 30]) for _ in range(5000)]
    m_a_values = [rng.randint(2 ** 30, 2 ** 31 - 1) for _ in range(5000)]
    e_a_values = [rng.randint(-40, -30) for _ in range(5000)]

    # Guard: the sweep must draw both signs
    has_positive = any(b > 0 for b in b_values)
    has_negative = any(b < 0 for b in b_values)
    assert has_positive and has_negative, "sweep did not draw both signs"

    # Conformance: each tuple via the oracle
    for b, q_b, m_a, e_a in zip(b_values, q_b_values, m_a_values, e_a_values):
        r_a = reciprocal_oracle(m_a)
        expected = bias_reconcile_oracle(b, q_b, r_a, e_a)
        actual = bias_reconcile(b, q_b, r_a, e_a)
        assert actual == expected

    # Executed exact-tie fixtures: m_a = 2^30 gives r_a = 2^32 exactly.
    # Assert decisiveness of the reciprocal.
    m_a = 2 ** 30
    r_a = reciprocal_oracle(m_a)
    assert r_a == 2 ** 32, f"expected r_a = 2^32 at m_a = 2^30, got {r_a}"

    # Divisor = 2^56 at q_b = 30, e_a = -36.
    # b = ±2^23 puts b·r_a = ±2^55 — an exact half.
    # The corrected rule rounds AWAY from zero: ±2^23 → ±1.
    q_b, e_a = 30, -36
    b_pos = 2 ** 23
    b_neg = -(2 ** 23)

    expected_pos = bias_reconcile_oracle(b_pos, q_b, r_a, e_a)
    expected_neg = bias_reconcile_oracle(b_neg, q_b, r_a, e_a)
    actual_pos = bias_reconcile(b_pos, q_b, r_a, e_a)
    actual_neg = bias_reconcile(b_neg, q_b, r_a, e_a)

    assert expected_pos == 1, f"oracle: expected +1, got {expected_pos}"
    assert expected_neg == -1, f"oracle: expected -1, got {expected_neg}"
    assert actual_pos == 1, f"implementation: expected +1, got {actual_pos}"
    assert actual_neg == -1, f"implementation: expected -1, got {actual_neg}"


# ==============================================================================
# Cell 2: C28 negative control — the sign-inverted form fails
# ==============================================================================


def test_bias_reconcile_fails_the_sign_inverted_form():
    """G-8 item 15: the pre-correction form (2^(62 - e_a), Q-format dropped) is what
    an implementer transcribing the OLD row would build. The cell asserts:
    1. The old form and the corrected oracle DISAGREE on the decisive fixture.
    2. The implementation matches the CORRECTED oracle, so an implementation of the
       old form FAILS here by construction.

    The fixture: (b = ±2^40, q_b = 30, r_a = reciprocal_oracle(2^30 + 12345),
    e_a = -36). At real exponents, the corrected formula gives ±65535 and the old
    form gives 0 — the old form UNDERFLOWS THE BIAS TO ZERO.
    """
    bias_reconcile = api(MODULE_INTMATH, "bias_reconcile")

    # Decisive fixture
    m_a = 2 ** 30 + 12345
    r_a = reciprocal_oracle(m_a)
    q_b, e_a = 30, -36
    b_pos = 2 ** 40
    b_neg = -(2 ** 40)

    # Assertion 1: Decisiveness guard on the fixture
    # The old form and the oracle must disagree (if they agree, the fixture cannot see the class)
    old_form_pos = bias_reconcile_sign_inverted(b_pos, q_b, r_a, e_a)
    oracle_pos = bias_reconcile_oracle(b_pos, q_b, r_a, e_a)
    assert old_form_pos != oracle_pos, (
        f"fixture not decisive: sign-inverted and oracle agree at "
        f"(b={b_pos}, q_b={q_b}, r_a={r_a}, e_a={e_a})"
    )

    # Same for negative sign
    old_form_neg = bias_reconcile_sign_inverted(b_neg, q_b, r_a, e_a)
    oracle_neg = bias_reconcile_oracle(b_neg, q_b, r_a, e_a)
    assert old_form_neg != oracle_neg, (
        f"fixture not decisive: sign-inverted and oracle agree at "
        f"(b={b_neg}, q_b={q_b}, r_a={r_a}, e_a={e_a})"
    )

    # Assertion 2: The implementation matches the CORRECTED oracle
    # An implementation computing the old form FAILS here
    actual_pos = bias_reconcile(b_pos, q_b, r_a, e_a)
    actual_neg = bias_reconcile(b_neg, q_b, r_a, e_a)
    assert actual_pos == oracle_pos, (
        f"implementation does not match oracle at b={b_pos}"
    )
    assert actual_neg == oracle_neg, (
        f"implementation does not match oracle at b={b_neg}"
    )


# ==============================================================================
# Cell 3: C30 exactness at operating exponents
# ==============================================================================


def test_iexp_constants_match_the_r_composed_oracle_at_operating_exponents():
    """C30 exactness (W2-4): the R-COMPOSED derivation recomputes via the reciprocal
    oracle and holds exactly at the operating-exponent band e ∈ {-40, -36, -32}.

    Conformance = arbitrary-precision recomputation of the R-COMPOSED formula
    (via rung1_ref.reciprocal_oracle, not intmath's Newton).
    """
    iexp_scale_constants = api(MODULE_INTMATH, "iexp_scale_constants")

    # Mantissa set: the named edges plus 2,000 seeded samples
    m_edges = [2 ** 30, 2 ** 30 + 1, 2 ** 31 - 1]
    rng = random.Random(7003)
    m_samples = [rng.randint(2 ** 30, 2 ** 31 - 1) for _ in range(2000)]
    m_values = sorted(set(m_edges + m_samples))

    # Operating exponents
    e_values = [-40, -36, -32]

    # Conformance: (m, e) → oracle must match implementation
    for m in m_values:
        for e in e_values:
            expected = iexp_constants_oracle(
                m, e,
                LN2_Q30, 30,  # ln2_q, ln2_fmt
                B_Q30, 30,    # b_q, b_fmt
                CA_Q30, 30    # ca_q, ca_fmt
            )
            actual = iexp_scale_constants(
                m, e,
                LN2_Q30, 30,
                B_Q30, 30,
                CA_Q30, 30
            )
            assert actual == expected, (
                f"mismatch at m={m}, e={e}: "
                f"expected {expected}, got {actual}"
            )


# ==============================================================================
# Cell 4: C30 fine-scale rejection edges
# ==============================================================================


def test_iexp_constant_derivation_rejects_the_fine_scale_edge():
    """C30 fine-scale (overflow-end) rejection boundaries: negative shifts are rejected.
    Per the coordinator's charter and W2-3:

    - e = -78 (below q_c's threshold at ca_fmt = 30): must raise ValueError.
    - e = -77 (shift_c = 0, the exact boundary): must compute (no raise).

    Do NOT assert exactness-vs-twin in this band (that is cell 5's subject) and
    the operating-exponent exactness is cell 3's.
    """
    iexp_scale_constants = api(MODULE_INTMATH, "iexp_scale_constants")

    m = 2 ** 30

    # e = -78: must raise ValueError (or a subclass, but assert ValueError catches all)
    with pytest.raises(ValueError):
        iexp_scale_constants(
            m, -78,
            LN2_Q30, 30,
            B_Q30, 30,
            CA_Q30, 30
        )

    # e = -77: shift_c = 0, must compute (no raise)
    try:
        result = iexp_scale_constants(
            m, -77,
            LN2_Q30, 30,
            B_Q30, 30,
            CA_Q30, 30
        )
        # Success means the call did not raise; q_c is a huge int at shift 0
        assert isinstance(result, tuple) and len(result) == 3
    except ValueError as e:
        pytest.fail(f"e = -77 (boundary) should compute, not raise: {e}")


# ==============================================================================
# Cell 5: C30 twin negative control (W2-4)
# ==============================================================================


def test_iexp_conformance_cell_can_see_the_exact_division_twin():
    """G-8 item 16, W2-4: the twin negative control. The pinned constructed divergent pair
    (m, e) = (2^30 + 1, -76) — oracle ≠ twin there — proves the suite can see the class.

    The cell:
    1. Asserts in-cell DECISIVENESS: the oracle and twin DISAGREE at this fixture.
    2. Asserts conformance: the implementation equals the ORACLE.
       An implementation computing the twin (exact division) FAILS here.

    Reason (W2-4): twin-equality as the conformance target would fail a CONFORMANT
    implementation at legitimate boundary divergences — the R-composed side is the definition.
    """
    iexp_scale_constants = api(MODULE_INTMATH, "iexp_scale_constants")

    m, e = 2 ** 30 + 1, -76

    # Assertion 1: Decisiveness guard
    # The oracle and twin must disagree (if they agree, the fixture cannot see the class)
    oracle = iexp_constants_oracle(
        m, e,
        LN2_Q30, 30,
        B_Q30, 30,
        CA_Q30, 30
    )
    twin = iexp_constants_exact_twin(
        m, e,
        LN2_Q30, 30,
        B_Q30, 30,
        CA_Q30, 30
    )
    assert oracle != twin, (
        f"fixture not decisive: oracle and twin agree at (m={m}, e={e})"
    )

    # Assertion 2: The implementation matches the ORACLE
    # An implementation computing the twin FAILS here
    actual = iexp_scale_constants(
        m, e,
        LN2_Q30, 30,
        B_Q30, 30,
        CA_Q30, 30
    )
    assert actual == oracle, (
        f"implementation must match oracle at (m={m}, e={e}); "
        f"got {actual}, expected {oracle}"
    )

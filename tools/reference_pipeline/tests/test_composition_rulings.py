"""The D-SLM57/D-SLM58 ruling cells — §17 G-8 items 12 (extension), 13 (extension), 14.

Curie-authored (design record A-7 §15.3 decision: single-construction cells against this
seat's own oracle primitives, below packet ceremony — no worker packet). Charters:
- item 13 extension: the association-order discriminating fixture — the executed witness
  triple; the suite must fail a right-associated implementation (D-SLM57).
- item 12 extension: the attn_ctx per-head pre-fold, identity-head discrimination included
  (D-SLM57; C24's pass-through pin at the context row).
- item 14 (re-authored charter): the k/v landing composite with the OFFLINE reciprocal
  (D-SLM58), with the runtime-reciprocal and two-rounding forms as executed negative
  controls — witnesses recomputed through composition_ref, never copied.

Test-design record: Claude/Curie/SuperSLM_Section15_Eval_TestDesign-2026-07-16.md §16 (A-7).
"""

from fractions import Fraction

from conftest import api
import rung1_ref
from composition_ref import (
    scale_mul_oracle,
    carried_scale_product_oracle,
    canonical_scale_oracle,
    residual_reconcile_oracle,
    round_half_away_from_zero_ratio,
    mbqm_oracle,
    fold_row_oracle,
    quantize_row_oracle,
)

MODULE = "reference_pipeline.pipeline"
INTMATH = "reference_pipeline.intmath"

# The A-5 §14.1 executed association-order witness triple (reproduced at the D-SLM57 pin).
WITNESS_TRIPLE = ((1607244023, -32), (1359881984, -28), (1259667695, -16))
# (1259667695, -16) is (Dn, -s) for the witness's s = 16 shift class.


def test_carried_scale_product_is_left_associated_incoming_first():
    """D-SLM57 (item 13): the multi-factor runtime scale product is LEFT-ASSOCIATED in
    composition order, incoming carried scale first — a right-associated implementation
    must fail at the executed witness triple (left mantissa 1194013661, right 1194013662).
    """
    carried_scale_product = api(INTMATH, "carried_scale_product")

    (a, b, c) = WITNESS_TRIPLE
    left = scale_mul_oracle(*scale_mul_oracle(*a, *b), *c)
    right = scale_mul_oracle(*a, *scale_mul_oracle(*b, *c))
    # In-cell decisiveness (executed A-5 §14.1: 81,034/200,000 diverge; this triple is
    # the first witness): the two association orders must differ here, or the fixture
    # cannot see the class.
    assert left != right, "fixture not decisive: associations agree at the witness triple"
    assert left == (1194013661, -15) and right == (1194013662, -15)

    got = carried_scale_product([a, b, c])
    assert tuple(got) == left, (
        f"carried_scale_product is not left-associated: got {got}, left {left}, "
        f"right {right} (D-SLM57)"
    )
    # Two-factor degenerate case: equals one scale_mul.
    assert tuple(carried_scale_product([a, b])) == scale_mul_oracle(*a, *b)


def test_attn_ctx_prefold_folds_head_segments_and_passes_the_identity_head():
    """D-SLM57 (item 12 extension): the attn_ctx per-head pre-fold — offline (0,1]
    constants per non-identity head SEGMENT through the C1 composite, the identity
    (max-S_v) head a TRUE pass-through (C24) — applied BEFORE the max-abs reduce, so the
    divergence class is token-wide through the shared D' (C25's class).

    Unit-level on the same fold machinery (the fold function does not know it is a
    context row); the identity-head discrimination reuses the executed C24 witness
    (2^30+1 with x = 215593831: pass-through code 25 vs near-identity 26).
    """
    fold_projection_accumulator = api(MODULE, "fold_projection_accumulator")
    quantize_multiplier = api(MODULE, "quantize_multiplier")

    head_dim = 2
    # Two heads: head 0 the identity (max S_v), head 1 folded at S_v ratio 0.75.
    fold_pair = quantize_multiplier(0.75)
    seg_folds = [None] * head_dim + [fold_pair] * head_dim

    row = [2 ** 30 + 1, 215593831, -(2 ** 20), 3 * 2 ** 19]  # head 0 | head 1
    folded = list(fold_projection_accumulator(row, seg_folds))
    assert folded == fold_row_oracle(row, seg_folds)

    # Identity head untouched (true pass-through — C24 at the context row).
    assert folded[:head_dim] == row[:head_dim]
    # Non-identity segment folded by exactly the C1 composite.
    assert folded[head_dim:] == [mbqm_oracle(v, *fold_pair) for v in row[head_dim:]]

    # Identity-head discrimination through the shared D' (the executed C24 witness):
    # the max element lives in the identity head; a near-identity treatment moves it
    # and the whole token's codes move with D'.
    assert quantize_row_oracle(folded)[0][1] == 25
    near = [mbqm_oracle(row[0], (1 << 31) - 1, 0)] + folded[1:]
    assert quantize_row_oracle(near)[0][1] == 26  # decisive — the fold must not do this


# --- item 14: the D-SLM58 landing's two executed negative controls -----------------
#
# Decisive fixtures constructed and executed 2026-07-16 (Curie bench, seed 20260717 —
# recomputed through composition_ref, never copied from the plan row):
#   runtime-reciprocal: acc=438386018, m_a=1691877598, e_a=-34, S_kh/S_ref=129/56
#       -> pinned 18741489 vs wrong-form 10254377238 (gross divergence)
#   two-rounding:       acc=-615068592, m_a=1130814510, e_a=-30, S_kh/S_ref=169/175
#       -> pinned -670758828 vs wrong-form -670758829 (off by one - the bit class)


def _landing_fixture(acc, m_a, e_a, num, den):
    m_t, e_t = canonical_scale_oracle(Fraction(num, den))
    r_t = rung1_ref.reciprocal_oracle(m_t)
    return m_t, e_t, r_t, residual_reconcile_oracle(acc, m_a, r_t, e_a, e_t)


def test_kv_landing_rejects_the_runtime_reciprocal_form():
    """D-SLM58 negative control 1 (item 14): an implementation realizing the OLD
    parenthetical — the reciprocal taken at runtime over the per-token mantissa, roles
    swapped — must FAIL. The pinned form's only reciprocal is OFFLINE, over the static
    target mantissa; the per-token quantity (m_a) is the composite's MULTIPLIER.
    """
    residual_reconcile = api(INTMATH, "residual_reconcile")

    acc, m_a, e_a, num, den = 438386018, 1691877598, -34, 129, 56
    m_t, e_t, r_t, pinned_value = _landing_fixture(acc, m_a, e_a, num, den)

    # The wrong form: recip over m_a (runtime, per token), the static mantissa as the
    # multiplier — the roles the old parenthetical implied.
    r_a = rung1_ref.reciprocal_oracle(m_a)
    wrong = residual_reconcile_oracle(acc, m_t, r_a, e_t, e_a)
    assert wrong != pinned_value, "fixture not decisive for the runtime-reciprocal form"
    assert pinned_value == 18741489 and wrong == 10254377238  # executed literals

    # Conformance: the implementation primitive at the pinned roles equals the oracle —
    # an implementation computing the wrong form fails here by construction.
    assert residual_reconcile(acc, m_a, r_t, e_a, e_t) == pinned_value


def test_kv_landing_rejects_the_two_rounding_form():
    """D-SLM58 negative control 2 (item 14): the rejected two-rounding form — a runtime
    scale product first (one high-mul rounding), then the element multiply (a second
    rounding) — diverges from the pinned ONE-rounding composite on ~19.9% of the domain
    (D-SLM58's executed rate) and must FAIL. Divergence here is exactly one code-unit —
    the C7 two-faithful-forms class the pin exists to close.
    """
    residual_reconcile = api(INTMATH, "residual_reconcile")

    acc, m_a, e_a, num, den = -615068592, 1130814510, -30, 169, 175
    m_t, e_t, r_t, pinned_value = _landing_fixture(acc, m_a, e_a, num, den)

    # The wrong form: (m_f, e_f) = scale_mul((m_a, e_a), canonical(S_ref/S_kh)), then
    # the per-element multiply with its own rounding.
    m_f, e_f = scale_mul_oracle(m_a, e_a, *canonical_scale_oracle(Fraction(den, num)))
    numerator = acc * m_f
    wrong = (round_half_away_from_zero_ratio(numerator, 2 ** (-e_f))
             if e_f <= 0 else numerator * 2 ** e_f)
    assert wrong != pinned_value, "fixture not decisive for the two-rounding form"
    assert pinned_value == -670758828 and wrong == -670758829  # executed literals

    assert residual_reconcile(acc, m_a, r_t, e_a, e_t) == pinned_value

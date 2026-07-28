"""Curie's regeneration gate for gen_s3_3_red_regression_fixtures.py.

WHY THIS EXISTS. Same shape as test_gen_s3_3_fixtures_regenerates.py's own
precedent: the generated header states "Re-running this script must
reproduce this file byte-for-byte", and this gate is what checks that claim
on every CI run (`python -m pytest tests/ci/ -v` already collects every
module here).

MECHANISM. Calls the generator's own module-level `_build_*()` functions
directly (not `main()`, which writes to disk), so each witness's own
mechanism claim is proven live against the real vendored reference: the
overflow witness genuinely overflows int64 forming q_b*q_b + q_c; the
negative-m_a witness is genuinely odd-symmetric in m_a; the extreme-exponent
witness's true magnitude genuinely exceeds the clamp range; the q_ln2==0
witness genuinely degenerates via the real iexp_scale_constants.

Casebooks:
Claude/Poirot/c314a64-s3.3-attention-interior-review-2026-07-28.md
Claude/Popper/superslm-c27-kv-landing-domain-bounds-debunk-2026-07-28.md
"""
import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_TESTS_DIR = os.path.normpath(os.path.join(_THIS_DIR, os.pardir))
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)

import gen_s3_3_red_regression_fixtures as gen  # noqa: E402


def _read_committed_header() -> str:
    with open(gen.OUT_PATH, "r", encoding="ascii") as f:
        return f.read()


def test_regenerating_reproduces_the_committed_header_byte_for_byte():
    committed = _read_committed_header()
    regenerated = gen.generate()
    assert regenerated == committed, (
        "tests/gen_s3_3_red_regression_fixtures.py's generate() output no longer matches the "
        "committed tests/sslm_s3_3_red_regression_fixtures.h -- regenerate and commit the "
        "header, or the generator's own logic changed without the fixture being regenerated "
        "to match"
    )


def test_the_gate_has_something_to_lose_a_nonvacuous_floor():
    committed = _read_committed_header()
    for needle in (
        "kSoftmaxRowOverflowWitness", "kLandingNegativeMaWitness",
        "kLandingExtremeExponentWitness", "kSoftmaxQLn2ZeroWitness",
        "kLandingRoundDivideInBandWitness",
    ):
        assert needle in committed, f"expected witness group {needle!r} missing from the committed header"
    assert len(committed) > 1500, (
        f"the committed header is only {len(committed)} bytes -- suspiciously small for a "
        f"fixture claiming five distinct witness groups"
    )


def test_overflow_witness_genuinely_overflows_int64_and_exceeds_the_ceiling():
    w = gen._build_overflow_witness()
    true_m = w["q_b"] * w["q_b"] + w["q_c"]
    assert true_m > gen.INT64_MAX, "the witness must overflow int64 forming q_b*q_b + q_c"
    assert true_m > gen.K_SOFTMAX_ROW_MAX_SAFE_EXPONENT, "the witness must exceed the ratified 2**47 ceiling"


def test_negative_ma_witness_is_odd_symmetric_and_does_not_saturate():
    w = gen._build_negative_ma_witness()
    assert w["correct_raw_neg"] == -w["correct_raw_pos"]
    assert -127 < w["correct_raw_pos"] < 127 and -127 < w["correct_raw_neg"] < 127, (
        "neither sign of this witness may saturate the clamp -- a saturating witness cannot "
        "distinguish a wrong magnitude that also saturates"
    )


def test_extreme_exponent_witness_true_magnitude_exceeds_the_clamp_range():
    import intmath as im  # the vendored reference, already on sys.path via gen module import

    w = gen._build_extreme_et_witness()
    reference = im.residual_reconcile(w["branch_code"], w["m_a"], w["r_t"], w["e_a"], w["e_t"])
    assert abs(reference) > 127
    assert (reference > 0) == w["reference_positive"]
    assert reference.bit_length() == w["reference_bit_length"]


def test_qln2_zero_witness_genuinely_degenerates_via_the_real_derivation():
    w = gen._build_qln2_zero_witness()
    assert w["q_ln2"] == 0
    assert w["q_b"] == 0 and w["q_c"] == 0


def test_round_divide_in_band_witness_lands_on_the_round_divide_branch_and_exceeds_the_clamp():
    import intmath as im  # the vendored reference, already on sys.path via gen module import

    w = gen._build_round_divide_in_band_witness()
    k = 62 - (w["e_a"] - w["e_t"])
    assert k >= 0, "this witness must land on LandingRescale's round-divide branch (k >= 0)"
    reference = im.residual_reconcile(w["branch_code"], w["m_a"], w["r_t"], w["e_a"], w["e_t"])
    assert abs(reference) > 127
    assert (reference > 0) == w["reference_positive"]
    assert reference.bit_length() == w["reference_bit_length"]

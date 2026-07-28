"""Curie's regeneration gate for gen_s3_3_fixtures.py
(SuperSLM_S3a_WalkingSkeleton_Plan.md Sec11 S3.3, the attention interior).

WHY THIS EXISTS. Same shape as test_gen_s3_2_fixtures_regenerates.py's own
precedent: the generated header states "Re-running this script must reproduce
this file byte-for-byte", and this gate is what checks that claim on every CI
run (`python -m pytest tests/ci/ -v` already collects every module here).

MECHANISM. Calls the generator's own module-level `_build_*()` functions
directly (not `main()`, which writes to disk), so each of this pass's own
mechanism claims is proven live: the C33 clamp witnesses genuinely exceed
int8/int32 range via the REAL vendored RopeApplyPair; the C32 numerator/sum
witnesses genuinely straddle the two named candidate thresholds; the landing
composite's two negative controls genuinely diverge from the pin on
non-saturating witnesses; the ctx_fold join's independent QuantizeMultiplier
decomposition round-trips against MultiplyByQuantizedMultiplier; the
KvLandingReciprocals bound is exactly what DynamicScaleReciprocal's own domain
endpoints produce.

Line-ending normalization matches the S3.1/S3.2 precedent (LF; this Windows
checkout has core.autocrlf=true).

Test-design record:
Claude/Curie/superslm-s3.3-attention-interior-test-design-2026-07-28.md
"""
import math
import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_TESTS_DIR = os.path.normpath(os.path.join(_THIS_DIR, os.pardir))
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)

import gen_s3_3_fixtures as gen  # noqa: E402


def _read_committed_header() -> str:
    with open(gen.OUT_PATH, "r", encoding="ascii") as f:
        return f.read()


def test_regenerating_reproduces_the_committed_header_byte_for_byte():
    committed = _read_committed_header()
    regenerated = gen.generate()
    assert regenerated == committed, (
        "tests/gen_s3_3_fixtures.py's generate() output no longer matches the "
        "committed tests/sslm_s3_3_fixtures.h -- regenerate and commit the "
        "header, or the generator's own logic changed without the fixture "
        "being regenerated to match"
    )


def test_the_gate_has_something_to_lose_a_nonvacuous_floor():
    committed = _read_committed_header()
    for needle in (
        "kC33BothSignsCase", "kC33Int32OverflowCase",
        "kC32WidthDomainCases[]", "kC32BandCase",
        "kLandingCompositeCases[]", "kCtxFoldJoinCase",
        "kKvLandingReciprocalMin", "kKvLandingReciprocalMax",
        "kNumeratorLimit2Pow48Minus1", "kNumeratorLimit2Pow47",
    ):
        assert needle in committed, f"expected witness group {needle!r} missing from the committed header"
    assert len(committed) > 3000, (
        f"the committed header is only {len(committed)} bytes -- suspiciously "
        f"small for a fixture claiming six distinct witness groups"
    )


def test_c33_both_signs_witness_genuinely_exceeds_int8_range_both_signs():
    c = gen._build_c33_cases()["both_signs"]
    assert c["raw_x"] > 127, "the positive-side witness must exceed +127"
    assert c["raw_x_neg"] < -127, "the negative-side witness must exceed -127"
    assert c["raw_x_neg"] == -c["raw_x"] and c["raw_y_neg"] == -c["raw_y"], (
        "RopeApplyPair is linear in (x, y); negating the input must exactly negate the output"
    )


def test_c33_int32_overflow_witness_genuinely_exceeds_int32_range():
    c = gen._build_c33_cases()["int32_overflow"]
    assert c["raw_x"] > gen.INT32_MAX or c["raw_x"] < gen.INT32_MIN, (
        "the int32-overflow witness must exceed int32's own representable range"
    )


def test_c32_numerator_overflow_witness_exceeds_both_candidate_thresholds():
    cases = {c["label"]: c for c in gen._build_c32_cases()}
    c = cases["numerator_overflow_single_element"]
    assert c["shipped_check_ok"], "the witness must still pass the existing shipped int64-representability check"
    assert c["row_max"] > gen.NUM_LIMIT_2POW48_MINUS_1
    assert c["row_max"] > gen.NUM_LIMIT_2POW47
    assert c["width"] == 1, "a numerator overflow needs only one element to demonstrate"


def test_c32_sum_overflow_witness_is_numerator_safe_but_sum_unsafe():
    cases = {c["label"]: c for c in gen._build_c32_cases()}
    c = cases["sum_overflow_small_width_numerator_safe"]
    assert c["row_max"] <= gen.NUM_LIMIT_2POW47, "the witness's own row_max must be numerator-safe under BOTH candidates"
    assert c["width"] * c["row_max"] > (1 << 63) - 1, "the chosen width must overflow the int64 sum"
    assert (c["width"] - 1) * c["row_max"] <= (1 << 63) - 1, (
        "width - 1 must NOT overflow -- otherwise this is not the smallest triggering width this construction found"
    )


def test_c32_accept_witness_is_safe_under_every_check():
    cases = {c["label"]: c for c in gen._build_c32_cases()}
    c = cases["accept_realistic_width"]
    assert c["shipped_check_ok"]
    assert c["row_max"] <= gen.NUM_LIMIT_2POW47
    assert c["width"] * c["row_max"] <= (1 << 63) - 1


def test_c32_band_case_is_where_the_two_candidate_thresholds_disagree():
    band = gen._build_c32_band_case()
    ok_48 = band["row_max"] <= gen.NUM_LIMIT_2POW48_MINUS_1
    ok_47 = band["row_max"] <= gen.NUM_LIMIT_2POW47
    assert ok_48 and not ok_47, (
        "the band witness must be ACCEPTED under the looser 2**48-1 threshold and "
        "REJECTED under the stricter 2**47 threshold -- otherwise it does not "
        "demonstrate the open disagreement"
    )
    assert band["shipped_check_ok"], "the band witness must still pass the existing shipped check"


def test_landing_composite_negative_controls_genuinely_diverge_and_do_not_saturate():
    cases = {c["label"]: c for c in gen._build_landing_cases()}
    for label in ("positive_branch_diverges", "negative_branch_diverges"):
        c = cases[label]
        assert c["wrong_pk_diverges"], f"{label}: per-key-reciprocal control has no discriminating power"
        assert c["wrong_2r_diverges"], f"{label}: two-rounding control has no discriminating power"
        assert -127 < c["correct"] < 127, (
            f"{label}: the correct result must not saturate the clamp -- a saturated "
            f"witness cannot distinguish a wrong construction that also saturates"
        )
    zero = cases["zero_branch"]
    assert zero["correct"] == 0 and zero["wrong_per_key_reciprocal"] == 0 and zero["wrong_two_rounding"] == 0, (
        "a zero branch code must reconcile to zero under every construction (numerator is 0 in all three)"
    )


def test_ctx_fold_join_independent_decomposition_round_trips_the_shipped_kernel():
    import intmath as im  # the vendored reference, already on sys.path via gen module import

    case = gen._build_ctx_fold_join_case()
    # The independent decomposition, applied through the SAME shipped
    # multiply_by_quantized_multiplier the vendored reference (and the real
    # C++ MultiplyByQuantizedMultiplier) implement, must reproduce
    # round(ctx1 * ratio) to within one requantization step -- i.e. the
    # decomposition is a genuine approximation of the ratio, not an
    # arbitrary pass.
    approx = im.multiply_by_quantized_multiplier(case["ctx1"], case["mult"], case["shift"])
    assert approx == case["expected1"]
    exact = case["ctx1"] * case["ratio"]
    assert abs(approx - exact) <= 2, (
        f"the independent decomposition's output ({approx}) must approximate "
        f"ctx1*ratio ({exact}) closely -- a decomposition bug would diverge by far more"
    )
    assert case["expected0"] == case["ctx0"], "the identity head must leave ctx0 exactly unchanged"


def test_kv_landing_reciprocal_bound_matches_the_real_primitives_own_domain_endpoints():
    import intmath as im

    domain = gen._build_kv_landing_domain()
    assert im.dynamic_scale_reciprocal(1 << 30) == domain["r_max"] == 1 << 32
    assert im.dynamic_scale_reciprocal((1 << 31) - 1) == domain["r_min"]
    assert domain["r_min"] == (1 << 31) + 1
    assert domain["m_min"] == 1 << 30
    assert domain["m_max"] == (1 << 31) - 1

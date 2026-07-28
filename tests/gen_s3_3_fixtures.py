#!/usr/bin/env python3
"""Generates sslm_s3_3_fixtures.h -- derived witnesses for S3.3, the attention
interior (Claude/Plans/SuperSLM_S3a_WalkingSkeleton_Plan.md Sec11 S3.3; C27,
C32, C33, F-S3-3, F-S3-6, T-518), plus D-SLM365/D-SLM366's owed C32 width
predicate (Claude/Vitruvius/SuperSLM_S3.3_C32WidthBound_Derivation-2026-07-28.md).

WHAT THIS PINS, AND WHAT IT DOES NOT (read before extending). Every value below
is either (a) computed by CALLING the vendored, hash-pinned reference
(tests/reference/superslm_spike/intmath.py, rope.py) -- never re-derived from
its formula in this module -- or (b) computed by an INDEPENDENT construction
this module writes from a public algorithm description (the gemmlowp
QuantizeMultiplier decomposition, Sec4 below), never from reading
Tools/superslm_spike/pipeline.py's own source, so a join cell built from it is
not comparing a thing to itself (F-S3-3's own finding, Plan Sec4.3).

D-SLM365/D-SLM366's own derivation record already gives the closed form
(M = q_b^2 + q_c, the row maximum) and the two conditions C32 needs
(numerator fits int64_t: M <= INT64_MAX >> PROB_FRAC_BITS; sum fits int64_t:
W * M <= INT64_MAX). This module does not re-derive the closed form -- it
recomputes concrete witness POINTS against it, by direct execution, rather
than transcribing the derivation record's own sampled table (per Curie's own
discipline: never copied from a probe or a prior record, recomputed).

THE THRESHOLD IS NOT PINNED HERE (routed by the coordinator, not this
module's call). Tools/superslm_spike/pipeline.py's own `_guard_probability_width`
(called from `_vec_softmax`) refuses at `(2**62) >> PROB_FRAC_BITS` = 2**47 --
one bit STRICTER than the true int64-safe ceiling `INT64_MAX >> PROB_FRAC_BITS`
(2**48 - 1) this module derives independently below. Tools/superslm_spike's
OWN forward (`dynamic_engine.py`, S3a's actual criterion-2 reference,
Sec12 item 2) carries NO guard of any kind at the inline probs computation.
Three different answers to "what is the numerator ceiling" exist in this
tree's own record today, and choosing among them is Dan's call, not this
generator's. So this module emits BOTH named candidate thresholds
(`kNumeratorLimit2Pow48Minus1`, `kNumeratorLimit2Pow47`) plus, for every case,
which of the two the case falls under -- and a dedicated BAND witness where
the two thresholds disagree (Sec6). No case's `expected_*` field hardcodes
one choice; the C++ side reads its own not-yet-declared named constant
(test-design record Sec5) and this header hands it both candidates to check
against, whichever Brunel/Dan's ruling wires up.

Re-running this script must reproduce the emitted header byte-for-byte.

Test-design record: Claude/Curie/superslm-s3.3-attention-interior-test-design-2026-07-28.md
"""
from __future__ import annotations

import json
import math
import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REFERENCE_DIR = os.path.normpath(os.path.join(_THIS_DIR, "reference"))
_SPIKE_DIR = os.path.join(_REFERENCE_DIR, "superslm_spike")
sys.path.insert(0, _SPIKE_DIR)
sys.path.insert(0, _REFERENCE_DIR)

import intmath as im  # noqa: E402  (the vendored, hash-pinned reference)
import rope as rope_ref  # noqa: E402

with open(os.path.join(_SPIKE_DIR, "rope_tables_pinned.json"), "r", encoding="ascii") as _f:
    _PINNED_ROPE = json.load(_f)

# Same pin every S3.x generator applies (S-HARDEN-5 design Sec3.2): im._LN2
# resolved from the pinned double, never from this runner's live libm.
im._LN2 = float.fromhex(_PINNED_ROPE["ln2"]["hex"])

OUT_PATH = os.path.join(_THIS_DIR, "sslm_s3_3_fixtures.h")

INT32_MIN = -(1 << 31)
INT32_MAX = (1 << 31) - 1
INT64_MIN = -(1 << 63)
INT64_MAX = (1 << 63) - 1
PROB_FRAC_BITS = 15
QFMT = 30

_IEXP_LN2_Q = int(im._LN2 * (1 << QFMT))
_IEXP_B_Q = int(im._POLY_B * (1 << QFMT))
_IEXP_CA_Q = int((im._POLY_C / im._POLY_A) * (1 << QFMT))

ROPE_ONE = 1 << 30  # rope.ROPE_ONE, restated as an int literal (no float import needed here)


# =====================================================================
# Sec1 -- C33: the post-rotation clamp (Plan Sec5.3)
# =====================================================================
#
# RopeApplyPair is CALLED (never re-derived) via the vendored rope.py's own
# `rope_apply_pair`, which is a straight, unrounded-a-second-time port of
# intmath.h's own construction (a single RoundingDivideByPOT). Two witnesses:
# a realistic int8-domain pair whose rotated output exceeds [-127, 127] on
# BOTH signs at once (the rotation gain sqrt(2) on a full-magnitude int8
# input already clears int8 range -- no exotic input needed), and a second,
# wide-input pair whose rotated output exceeds int32's own range (proving the
# clamp reads the int64 RopePair member, not a narrowed copy, Plan Sec5.3).


def _build_c33_cases() -> dict:
    # Both-signs witness: two full-magnitude, opposite-signed int8 codes,
    # rotated by a near-diagonal angle so both output components have
    # sizeable magnitude (neither collapses to 0 by symmetry).
    x0, y0 = 127, -127
    cos0 = int(round(math.cos(0.7) * (1 << 30)))
    sin0 = int(round(math.sin(0.7) * (1 << 30)))
    rx0, ry0 = rope_ref.rope_apply_pair(x0, y0, cos0, sin0)
    assert rx0 > 127 or rx0 < -127 or ry0 > 127 or ry0 < -127, (
        "C33 both-signs witness must exceed int8 range on at least one component"
    )
    # Force a case with one component clearly positive-out-of-range and one
    # clearly negative-out-of-range, so the clamp cell exercises both bounds
    # in one fixture (mirrors the plan's own "on both signs" requirement).
    x1, y1 = 127, 127
    cos1 = int(round(math.cos(-0.78) * (1 << 30)))
    sin1 = int(round(math.sin(-0.78) * (1 << 30)))
    rx1, ry1 = rope_ref.rope_apply_pair(x1, y1, cos1, sin1)
    # RopeApplyPair is linear in (x, y), so negating the input negates the
    # output exactly -- (-127, -127) at the SAME angle produces -rx1, not a
    # coincidence to re-derive.
    x2, y2 = -x1, -y1
    rx2, ry2 = rope_ref.rope_apply_pair(x2, y2, cos1, sin1)
    assert rx2 == -rx1 and ry2 == -ry1, "RopeApplyPair linearity check failed"

    both_signs = {
        "x": x1, "y": y1, "cos_q30": cos1, "sin_q30": sin1,
        "raw_x": rx1, "raw_y": ry1,
        "x_neg": x2, "y_neg": y2, "raw_x_neg": rx2, "raw_y_neg": ry2,
    }
    assert rx1 > 127, f"positive witness must exceed +127 on x, got {rx1}"
    assert rx2 < -127, f"negative witness must exceed -127 on x, got {rx2}"

    # int32-overflow witness: near-INT32_MAX inputs, near-diagonal angle.
    x3, y3 = INT32_MAX, INT32_MAX
    rx3, ry3 = rope_ref.rope_apply_pair(x3, y3, cos1, sin1)
    assert rx3 > INT32_MAX or rx3 < INT32_MIN, (
        f"int32-overflow witness must exceed int32 range, got raw_x={rx3}"
    )

    int32_overflow = {"x": x3, "y": y3, "cos_q30": cos1, "sin_q30": sin1, "raw_x": rx3, "raw_y": ry3}

    def clamp(v, lo, hi):
        return max(lo, min(hi, v))

    for case in (both_signs, int32_overflow):
        pass

    return {
        "both_signs": both_signs,
        "int32_overflow": int32_overflow,
        "clamp127": lambda v: clamp(v, -127, 127),
        "clamp128": lambda v: clamp(v, -128, 127),
    }


# =====================================================================
# Sec2 -- C32/D-SLM366: the softmax row width predicate's witnesses
# =====================================================================


def _softmax_row_max(m: int, e: int):
    """M = the row maximum i-exp value at carried scale (m, e), q=0 (the
    dominant element ShiftByMax always sends to zero). Recomputed by CALLING
    the vendored iexp_scale_constants + i_exp_from_constants -- the exact
    closed form D-SLM365 derived and confirmed by execution (M = q_b^2 +
    q_c), never re-typed as a formula here."""
    q_ln2, q_b, q_c = im.iexp_scale_constants(
        m, e, _IEXP_LN2_Q, QFMT, _IEXP_B_Q, QFMT, _IEXP_CA_Q, QFMT
    )
    M = im.i_exp_from_constants(0, q_ln2, q_b, q_c)
    return q_ln2, q_b, q_c, M


def _shipped_check_ok(q_ln2: int, q_b: int, q_c: int) -> bool:
    """The SAME arbitrary-precision decomposition oracle every S3.1 sweep
    fixture uses for IExpConstantsInDomain (tests/gen_iexp_domain_fixtures.py's
    own formula) -- q=0 is the only element that needs checking (every other
    element's z >= 0 divides the same or a smaller magnitude)."""
    I_EXP_CLIP_N = 30
    if q_ln2 < 1 or q_ln2 > INT64_MAX // I_EXP_CLIP_N:
        return False
    clip_lo = -I_EXP_CLIP_N * q_ln2
    clipped = max(0, clip_lo)
    z = (-clipped) // q_ln2
    q_p = clipped + z * q_ln2
    base = q_p + q_b
    value = (base * base + q_c) >> z
    return INT64_MIN <= value <= INT64_MAX


NUM_LIMIT_2POW48_MINUS_1 = INT64_MAX >> PROB_FRAC_BITS  # 2**48 - 1, this module's own derivation
NUM_LIMIT_2POW47 = (1 << 62) >> PROB_FRAC_BITS           # 2**47, pipeline.py's _guard_probability_width


def _build_c32_cases() -> list[dict]:
    cases = []

    # 1) Numerator overflow on a SINGLE element (W=1 already fails, at
    #    either threshold, and even the EXISTING shipped check's own int64
    #    representability holds -- this is what D-SLM366 names "at e=-61 the
    #    numerator overflows on a single element").
    m, e = (1 << 31) - 1, -61
    q_ln2, q_b, q_c, M = _softmax_row_max(m, e)
    shipped_ok = _shipped_check_ok(q_ln2, q_b, q_c)
    assert shipped_ok, "numerator-overflow witness must still pass the existing shipped check"
    assert M > NUM_LIMIT_2POW48_MINUS_1 and M > NUM_LIMIT_2POW47, (
        "numerator-overflow witness must exceed BOTH candidate thresholds"
    )
    cases.append({
        "label": "numerator_overflow_single_element",
        "m": m, "e": e, "q_ln2": q_ln2, "q_b": q_b, "q_c": q_c, "row_max": M,
        "width": 1,
        "shipped_check_ok": shipped_ok,
        "ok_under_2pow48m1": M <= NUM_LIMIT_2POW48_MINUS_1,
        "ok_under_2pow47": M <= NUM_LIMIT_2POW47,
    })

    # 2) Sum overflow at the smallest width this construction can reach,
    #    numerator itself SAFE under BOTH candidate thresholds (D-SLM366's
    #    "at e=-60 a row of just 2 keys overflows the sum" shape, recomputed
    #    at this module's own chosen (m, e) rather than copied from the
    #    derivation record). Whenever a row's numerator is safe under the
    #    STRICTER (2**47) threshold, W*M <= INT64_MAX requires
    #    W <= INT64_MAX / M -- minimized by MAXIMIZING M subject to staying
    #    numerator-safe, which is what this scan searches for (the opposite
    #    of minimizing M, which would maximize the safe width instead).
    best = None
    for e_try in range(-60, 0):
        for frac in (0.0, 0.001, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 0.999, 1.0):
            m_try = int((1 << 30) + frac * (((1 << 31) - 1) - (1 << 30)))
            m_try = min(m_try, (1 << 31) - 1)
            try:
                q_ln2_t, q_b_t, q_c_t, M_t = _softmax_row_max(m_try, e_try)
            except ValueError:
                continue
            if M_t <= NUM_LIMIT_2POW47 and (best is None or M_t > best[5]):
                best = (m_try, e_try, q_ln2_t, q_b_t, q_c_t, M_t)
    assert best is not None, "no numerator-safe row found in the scanned (m, e) range"
    m2, e2, q_ln2_2, q_b_2, q_c_2, M2 = best[0], best[1], best[2], best[3], best[4], best[5]
    max_w_48 = NUM_LIMIT_2POW48_MINUS_1 // M2 if M2 > 0 else 0
    max_w_true = INT64_MAX // M2 if M2 > 0 else 0
    width = max_w_true + 1  # one past the TRUE int64-sum ceiling -- must overflow under any candidate
    cases.append({
        "label": "sum_overflow_small_width_numerator_safe",
        "m": m2, "e": e2, "q_ln2": q_ln2_2, "q_b": q_b_2, "q_c": q_c_2, "row_max": M2,
        "width": width,
        "shipped_check_ok": _shipped_check_ok(q_ln2_2, q_b_2, q_c_2),
        "ok_under_2pow48m1": M2 <= NUM_LIMIT_2POW48_MINUS_1,
        "ok_under_2pow47": M2 <= NUM_LIMIT_2POW47,
        "max_safe_width_true_int64": max_w_true,
        "max_safe_width_2pow48m1": max_w_48,
    })

    # 3) Accept case: small M, small realistic width -- both thresholds and
    #    the sum check pass. Also the guard-vitality "silent control".
    m3, e3 = (1 << 31) - 1, -40
    q_ln2_3, q_b_3, q_c_3, M3 = _softmax_row_max(m3, e3)
    width3 = 4096  # a realistic context_cap-scale width
    assert M3 <= NUM_LIMIT_2POW47 and width3 * M3 <= INT64_MAX
    cases.append({
        "label": "accept_realistic_width",
        "m": m3, "e": e3, "q_ln2": q_ln2_3, "q_b": q_b_3, "q_c": q_c_3, "row_max": M3,
        "width": width3,
        "shipped_check_ok": _shipped_check_ok(q_ln2_3, q_b_3, q_c_3),
        "ok_under_2pow48m1": M3 <= NUM_LIMIT_2POW48_MINUS_1,
        "ok_under_2pow47": M3 <= NUM_LIMIT_2POW47,
    })

    return cases


def _build_c32_band_case() -> dict:
    """Sec6's routed finding, realized as data: a row where the row maximum
    M satisfies 2**47 < M <= 2**48-1 -- accepted under this module's own
    derived true int64-safe ceiling, REJECTED under pipeline.py's stricter
    `_guard_probability_width`. Found by direct execution over the same
    canonical-mantissa domain the numerator-safe scan above already covers,
    not guessed."""
    for e_try in range(-55, -50):
        for frac in (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0):
            m_try = int((1 << 30) + frac * (((1 << 31) - 1) - (1 << 30)))
            m_try = min(m_try, (1 << 31) - 1)
            try:
                q_ln2_t, q_b_t, q_c_t, M_t = _softmax_row_max(m_try, e_try)
            except ValueError:
                continue
            if NUM_LIMIT_2POW47 < M_t <= NUM_LIMIT_2POW48_MINUS_1:
                return {
                    "m": m_try, "e": e_try, "q_ln2": q_ln2_t, "q_b": q_b_t, "q_c": q_c_t,
                    "row_max": M_t, "width": 1,
                    "shipped_check_ok": _shipped_check_ok(q_ln2_t, q_b_t, q_c_t),
                }
    raise AssertionError("no band witness found where the two candidate thresholds disagree")


# =====================================================================
# Sec3 -- C27/D-SLM57: the K/V landing composite (Plan Sec8.1) and its two
# named negative controls, via the vendored residual_reconcile.
# =====================================================================


def _wrong_per_key_token_reciprocal(branch_code: int, m_b: int, r_h: int, e_b: int, e_h: int) -> int:
    """A per-key-token-reciprocal implementation (Plan Sec8.1's first named
    negative control): recomputes a reciprocal AT THE BRANCH's own mantissa
    (via the vendored DynamicScaleReciprocal-equivalent) instead of using the
    fixed home-scale reciprocal R_h the pin actually calls for -- the wrong
    per-key-token reciprocal `im.dynamic_scale_reciprocal(m_b)` stands in for
    R_h in the same formula."""
    r_wrong = im.dynamic_scale_reciprocal(m_b)
    numerator = branch_code * m_b * r_wrong
    k = 62 - (e_b - e_h)
    if k >= 0:
        return im._round_half_away_from_zero_ratio(numerator, 1 << k)
    return numerator << (-k)


def _wrong_two_rounding_scale_product(branch_code: int, m_b: int, r_h: int, e_b: int, e_h: int) -> int:
    """The two-rounding scale-product form (Plan Sec8.1's second named
    negative control): rounds branch_code*m_b to a Q31 intermediate FIRST
    (one rounding), then divides that already-rounded intermediate by R_h's
    own reciprocal-scale power a second time (a second rounding) -- instead
    of the pin's single joint round over the full numerator."""
    step1 = im._round_half_away_from_zero_ratio(branch_code * m_b, 1 << 31)
    numerator2 = step1 * r_h
    k2 = 62 - 31 - (e_b - e_h)
    if k2 >= 0:
        return im._round_half_away_from_zero_ratio(numerator2, 1 << k2)
    return numerator2 << (-k2)


def _build_landing_cases() -> list[dict]:
    cases = []
    # A witness with a genuinely negative branch code (signed) and distinct
    # e_b/e_h so the exponent term is exercised, at a realistic magnitude
    # (post-projection accumulator scale, well within int32).
    # r_h is C19's reciprocal of a DIFFERENT "home" mantissa (1,500,000,000)
    # than the branch's own m_b (2**30) -- deliberately, so the wrong
    # per-key-token-reciprocal construction (which recomputes a reciprocal
    # from m_b itself) actually diverges from the pin instead of coinciding
    # with it by construction. Both branch_code witnesses below were found
    # by scanning small in-range codes for the smallest ones where BOTH
    # named negative controls diverge from the correct result (neither
    # witness saturates the [-127,127] clamp, so the clamp cannot mask
    # either divergence).
    _r_h = im.dynamic_scale_reciprocal(1500000000)
    witnesses = [
        {"label": "positive_branch_diverges", "branch_code": 3, "m_b": 1073741824,
         "r_h": _r_h, "e_b": 0, "e_h": 0},
        {"label": "negative_branch_diverges", "branch_code": -55, "m_b": 1073741824,
         "r_h": _r_h, "e_b": 0, "e_h": 0},
        {"label": "zero_branch", "branch_code": 0, "m_b": 1073741824,
         "r_h": 4294967296, "e_b": -10, "e_h": -10},
    ]
    for w in witnesses:
        correct_raw = im.residual_reconcile(w["branch_code"], w["m_b"], w["r_h"], w["e_b"], w["e_h"])
        correct = max(-127, min(127, correct_raw))
        wrong_pk_raw = _wrong_per_key_token_reciprocal(w["branch_code"], w["m_b"], w["r_h"], w["e_b"], w["e_h"])
        wrong_pk = max(-127, min(127, wrong_pk_raw))
        wrong_2r_raw = _wrong_two_rounding_scale_product(w["branch_code"], w["m_b"], w["r_h"], w["e_b"], w["e_h"])
        wrong_2r = max(-127, min(127, wrong_2r_raw))
        cases.append({
            **w,
            "correct_raw": correct_raw, "correct": correct,
            "wrong_per_key_reciprocal_raw": wrong_pk_raw, "wrong_per_key_reciprocal": wrong_pk,
            "wrong_two_rounding_raw": wrong_2r_raw, "wrong_two_rounding": wrong_2r,
            "wrong_pk_diverges": wrong_pk != correct,
            "wrong_2r_diverges": wrong_2r != correct,
        })
    for c in cases:
        if c["label"] != "zero_branch":
            assert c["wrong_pk_diverges"], f"{c['label']}: per-key-reciprocal control has no discriminating power"
            assert c["wrong_2r_diverges"], f"{c['label']}: two-rounding control has no discriminating power"
    return cases


# =====================================================================
# Sec4 -- the ctx_fold join oracle: an INDEPENDENT gemmlowp
# QuantizeMultiplier decomposition, written from the published algorithm
# description (frexp-based mantissa/exponent split), never from reading
# Tools/superslm_spike/pipeline.py's own `quantize_multiplier` source
# (F-S3-3's own correlated-oracle finding, Plan Sec4.3 -- the emitter and
# dynamic_engine.py's runtime derivation are already one implementation, so
# a join oracle built from either of them proves nothing).
# =====================================================================


def _independent_quantize_multiplier(real_multiplier: float) -> tuple[int, int]:
    """gemmlowp's QuantizeMultiplier, independently re-derived: decompose a
    ratio in (0, 1] into (mult: int32 in [2**30, 2**31), shift: nonneg int)
    such that MultiplyByQuantizedMultiplier(x, mult, shift) approximates
    round(x * real_multiplier). frexp gives real_multiplier = q * 2**exp with
    q in [0.5, 1); mult = round(q * 2**31) (rounding to nearest, ties to
    even, matching Python's own round()); shift = -exp (nonneg since
    real_multiplier <= 1 implies exp <= 0)."""
    if not (0.0 < real_multiplier <= 1.0):
        raise ValueError("this ctx_fold join witness is only exercised for f_v/s_v_max in (0, 1]")
    q, exp = math.frexp(real_multiplier)
    mult = int(round(q * (1 << 31)))
    shift = -exp
    if mult == (1 << 31):
        mult >>= 1
        shift -= 1
    assert (1 << 30) <= mult < (1 << 31), f"independent decomposition postcondition violated: mult={mult}"
    assert shift >= 0, f"independent decomposition postcondition violated: shift={shift} < 0"
    return mult, shift


def _build_ctx_fold_join_case() -> dict:
    ratio = 0.5  # f_v / s_v_max for the non-identity head
    mult, shift = _independent_quantize_multiplier(ratio)
    ctx0, ctx1 = 1073741825, -536870912  # arbitrary in-range wide context values
    expected0 = ctx0  # head 0: identity, unchanged
    expected1 = im.multiply_by_quantized_multiplier(ctx1, mult, shift)  # head 1: non-identity
    return {
        "ratio": ratio, "mult": mult, "shift": shift,
        "ctx0": ctx0, "ctx1": ctx1,
        "expected0": expected0, "expected1": expected1,
    }


# =====================================================================
# Sec5 -- the KvLandingScales/KvLandingReciprocals joint-domain descriptor
# (Plan Sec7.2a's third limb -- S3.3's own derivation obligation, the same
# shape BIA1's magnitude bound was derived in S3.2's own record).
# =====================================================================


def _build_kv_landing_domain() -> dict:
    # m_t: canonical, matching every other artifact-carried KVC1 scale
    # mantissa in this tree (the funnel's own CombineCarriedScale
    # precondition and the format's canonical-scale convention).
    m_min = 1 << 30
    m_max = (1 << 31) - 1
    # R_t: C19's own output range -- R_t IS "the offline per-(head,
    # projection) reciprocal" (Plan Sec8.1), so any value outside what
    # DynamicScaleReciprocal itself can ever produce is definitionally not a
    # reciprocal of anything, confirmed by calling the real primitive at
    # its domain's two endpoints rather than restating its documented range.
    r_max = im.dynamic_scale_reciprocal(1 << 30)
    r_min = im.dynamic_scale_reciprocal((1 << 31) - 1)
    assert r_max == 1 << 32, f"R_t max must be exactly 2**32, got {r_max}"
    assert r_min == (1 << 31) + 1, f"R_t min must be exactly 2**31+1, got {r_min}"
    return {"m_min": m_min, "m_max": m_max, "r_min": r_min, "r_max": r_max}


# =====================================================================
# Emission
# =====================================================================


def _fmt_bool(b: bool) -> str:
    return "true" if b else "false"


def generate() -> str:
    c33 = _build_c33_cases()
    c32_cases = _build_c32_cases()
    c32_band = _build_c32_band_case()
    landing_cases = _build_landing_cases()
    ctx_fold = _build_ctx_fold_join_case()
    kv_domain = _build_kv_landing_domain()

    lines = [
        "// GENERATED FILE. Do not hand-edit.",
        "//",
        "// Produced by tests/gen_s3_3_fixtures.py -- S3.3's derived witnesses",
        "// (attention interior: C27/D-SLM57 landing, C32/D-SLM366 softmax width,",
        "// C33 clamp, the ctx_fold join oracle, KvLandingScales/Reciprocals'",
        "// joint domain). RopeApplyPair/residual_reconcile/dynamic_scale_reciprocal",
        "// values are computed by CALLING the vendored reference",
        "// (tests/reference/superslm_spike/{intmath,rope}.py); the ctx_fold",
        "// join's (mult, shift) is an INDEPENDENT gemmlowp QuantizeMultiplier",
        "// decomposition written from the published algorithm, never from",
        "// Tools/superslm_spike/pipeline.py's own source (F-S3-3, Plan Sec4.3).",
        "//",
        "// The C32 numerator threshold is NOT resolved here -- Dan's ruling is open",
        "// between INT64_MAX>>PROB_FRAC_BITS (2**48-1, this module's own int64-safety",
        "// derivation) and pipeline.py's stricter _guard_probability_width (2**47).",
        "// Every C32 case below carries BOTH candidate verdicts",
        "// (ok_under_2pow48m1/ok_under_2pow47); kC32BandCase is a witness where they",
        "// disagree. See the test-design record Sec5 for the named-constant routing.",
        "//",
        "// Re-running this script must reproduce this file byte-for-byte.",
        "//",
        "// Test-design record:",
        "// Claude/Curie/superslm-s3.3-attention-interior-test-design-2026-07-28.md",
        "#ifndef SUPERSLM_TESTS_SSLM_S3_3_FIXTURES_H",
        "#define SUPERSLM_TESTS_SSLM_S3_3_FIXTURES_H",
        "",
        "#include <cstddef>",
        "#include <cstdint>",
        "",
        "namespace superslm_test {",
        "",
        "// --- Sec1: C33's post-rotation clamp witnesses ---",
        "",
        "struct C33BothSignsCase {",
        "\tint32_t x, y, cos_q30, sin_q30;",
        "\tint64_t raw_x, raw_y;",
        "\tint32_t x_neg, y_neg;",
        "\tint64_t raw_x_neg, raw_y_neg;",
        "};",
        "",
        "inline constexpr C33BothSignsCase kC33BothSignsCase = {",
        f"\t/*x=*/{c33['both_signs']['x']}, /*y=*/{c33['both_signs']['y']}, "
        f"/*cos_q30=*/{c33['both_signs']['cos_q30']}, /*sin_q30=*/{c33['both_signs']['sin_q30']},",
        f"\t/*raw_x=*/{c33['both_signs']['raw_x']}LL, /*raw_y=*/{c33['both_signs']['raw_y']}LL,",
        f"\t/*x_neg=*/{c33['both_signs']['x_neg']}, /*y_neg=*/{c33['both_signs']['y_neg']},",
        f"\t/*raw_x_neg=*/{c33['both_signs']['raw_x_neg']}LL, /*raw_y_neg=*/{c33['both_signs']['raw_y_neg']}LL,",
        "};",
        "",
        "struct C33Int32OverflowCase {",
        "\tint32_t x, y, cos_q30, sin_q30;",
        "\tint64_t raw_x, raw_y;",
        "};",
        "",
        "inline constexpr C33Int32OverflowCase kC33Int32OverflowCase = {",
        f"\t/*x=*/{c33['int32_overflow']['x']}, /*y=*/{c33['int32_overflow']['y']}, "
        f"/*cos_q30=*/{c33['int32_overflow']['cos_q30']}, /*sin_q30=*/{c33['int32_overflow']['sin_q30']},",
        f"\t/*raw_x=*/{c33['int32_overflow']['raw_x']}LL, /*raw_y=*/{c33['int32_overflow']['raw_y']}LL,",
        "};",
        "",
        "// --- Sec2: C32/D-SLM366's softmax-row-width predicate witnesses ---",
        "",
        "inline constexpr int64_t kNumeratorLimit2Pow48Minus1 = " f"{NUM_LIMIT_2POW48_MINUS_1}LL;"
        "  // INT64_MAX >> PROB_FRAC_BITS",
        "inline constexpr int64_t kNumeratorLimit2Pow47 = " f"{NUM_LIMIT_2POW47}LL;"
        "  // pipeline.py's _guard_probability_width",
        "",
        "struct C32WidthDomainCase {",
        "\tconst char* label;",
        "\tint64_t m; int e;",
        "\tint64_t q_ln2, q_b, q_c;",
        "\tint64_t row_max;         // M = q_b^2 + q_c, the row's peak i-exp value (q=0)",
        "\tsize_t width;",
        "\tbool shipped_check_ok;   // IExpConstantsInDomain(0, q_ln2, q_b, q_c)",
        "\tbool ok_under_2pow48m1;  // row_max <= kNumeratorLimit2Pow48Minus1",
        "\tbool ok_under_2pow47;    // row_max <= kNumeratorLimit2Pow47",
        "};",
        "",
        "inline constexpr C32WidthDomainCase kC32WidthDomainCases[] = {",
    ]
    for c in c32_cases:
        lines.append(
            f"\t{{ \"{c['label']}\", {c['m']}LL, {c['e']}, {c['q_ln2']}LL, {c['q_b']}LL, {c['q_c']}LL, "
            f"{c['row_max']}LL, {c['width']}u, {_fmt_bool(c['shipped_check_ok'])}, "
            f"{_fmt_bool(c['ok_under_2pow48m1'])}, {_fmt_bool(c['ok_under_2pow47'])} }},"
        )
    lines += [
        "};",
        "inline constexpr size_t kC32WidthDomainCasesCount = "
        "sizeof(kC32WidthDomainCases) / sizeof(kC32WidthDomainCases[0]);",
        "",
        "// The routed band witness (Sec6): the two candidate thresholds DISAGREE on",
        "// this row -- accepted under 2**48-1, rejected under 2**47. Not resolved by",
        "// this suite; see the test-design record's routed finding.",
        "inline constexpr C32WidthDomainCase kC32BandCase = {",
        f"\t\"band_2pow47_vs_2pow48m1_disagree\", {c32_band['m']}LL, {c32_band['e']}, "
        f"{c32_band['q_ln2']}LL, {c32_band['q_b']}LL, {c32_band['q_c']}LL, {c32_band['row_max']}LL, "
        f"{c32_band['width']}u, {_fmt_bool(c32_band['shipped_check_ok'])}, "
        f"{_fmt_bool(c32_band['row_max'] <= NUM_LIMIT_2POW48_MINUS_1)}, "
        f"{_fmt_bool(c32_band['row_max'] <= NUM_LIMIT_2POW47)}",
        "};",
        "",
        "// --- Sec3: C27/D-SLM57's K/V landing composite (residual_reconcile) ---",
        "",
        "struct LandingCompositeCase {",
        "\tconst char* label;",
        "\tint64_t branch_code, m_b, r_h; int e_b, e_h;",
        "\tint64_t correct_raw; int8_t correct;",
        "\tint64_t wrong_per_key_reciprocal_raw; int8_t wrong_per_key_reciprocal;",
        "\tint64_t wrong_two_rounding_raw; int8_t wrong_two_rounding;",
        "\tbool wrong_pk_diverges;",
        "\tbool wrong_2r_diverges;",
        "};",
        "",
        "inline constexpr LandingCompositeCase kLandingCompositeCases[] = {",
    ]
    for c in landing_cases:
        lines.append(
            f"\t{{ \"{c['label']}\", {c['branch_code']}LL, {c['m_b']}LL, {c['r_h']}LL, {c['e_b']}, {c['e_h']}, "
            f"{c['correct_raw']}LL, {c['correct']}, "
            f"{c['wrong_per_key_reciprocal_raw']}LL, {c['wrong_per_key_reciprocal']}, "
            f"{c['wrong_two_rounding_raw']}LL, {c['wrong_two_rounding']}, "
            f"{_fmt_bool(c['wrong_pk_diverges'])}, {_fmt_bool(c['wrong_2r_diverges'])} }},"
        )
    lines += [
        "};",
        "inline constexpr size_t kLandingCompositeCasesCount = "
        "sizeof(kLandingCompositeCases) / sizeof(kLandingCompositeCases[0]);",
        "",
        "// --- Sec4: the ctx_fold join oracle (independent QuantizeMultiplier) ---",
        "",
        "struct CtxFoldJoinCase {",
        "\tdouble ratio; int32_t mult; int32_t shift;",
        "\tint64_t ctx0, ctx1;",
        "\tint64_t expected0, expected1;",
        "};",
        "",
        "inline constexpr CtxFoldJoinCase kCtxFoldJoinCase = {",
        f"\t{ctx_fold['ratio']}, {ctx_fold['mult']}, {ctx_fold['shift']},",
        f"\t{ctx_fold['ctx0']}LL, {ctx_fold['ctx1']}LL,",
        f"\t{ctx_fold['expected0']}LL, {ctx_fold['expected1']}LL,",
        "};",
        "",
        "// --- Sec5: KvLandingScales/KvLandingReciprocals' joint domain (Sec7.2a) ---",
        "",
        f"inline constexpr int64_t kKvLandingScaleMantissaMin = {kv_domain['m_min']}LL;",
        f"inline constexpr int64_t kKvLandingScaleMantissaMax = {kv_domain['m_max']}LL;",
        f"inline constexpr int64_t kKvLandingReciprocalMin = {kv_domain['r_min']}LL;",
        f"inline constexpr int64_t kKvLandingReciprocalMax = {kv_domain['r_max']}LL;",
        "",
        "}  // namespace superslm_test",
        "",
        "#endif  // SUPERSLM_TESTS_SSLM_S3_3_FIXTURES_H",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    text = generate()
    with open(OUT_PATH, "w", encoding="ascii", newline="\n") as f:
        f.write(text)
    print(f"wrote {OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

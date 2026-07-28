#!/usr/bin/env python3
"""Generates sslm_s3_3_red_regression_fixtures.h -- witnesses for the six
S3.3 attention-interior defects confirmed by execution
(Claude/Poirot/c314a64-s3.3-attention-interior-review-2026-07-28.md;
Claude/Popper/superslm-c27-kv-landing-domain-bounds-debunk-2026-07-28.md)
against the S3.3 GREEN build at D:\\SuperSLM@c314a64. This is a Curie
red-regression pass authored AFTER a green suite already exists (S-HARDEN
posture, not first-build red-first) -- the suite is currently green and
wrong; every cell this generator feeds must fail against the shipped
c314a64 code and pass once each finding is fixed.

WHAT THIS PINS, AND HOW. Every derived-constant witness below (q_ln2, q_b,
q_c from `iexp_scale_constants`; the K/V landing composite's `correct_raw`
from `residual_reconcile`) is computed by CALLING the vendored, hash-pinned
reference (tests/reference/superslm_spike/intmath.py) directly -- never
re-implemented from its formula in this module, matching
tests/gen_s3_3_fixtures.py's own established discipline. The reference
module and the ln2 pin (`im._LN2`, `rope_tables_pinned.json`) are the exact
same objects gen_s3_3_fixtures.py already uses, so a witness computed here
and a witness computed there are comparable without a second import path.

Re-running this script must reproduce this file byte-for-byte.

Casebooks:
Claude/Poirot/c314a64-s3.3-attention-interior-review-2026-07-28.md
Claude/Popper/superslm-c27-kv-landing-domain-bounds-debunk-2026-07-28.md
"""
from __future__ import annotations

import json
import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REFERENCE_DIR = os.path.normpath(os.path.join(_THIS_DIR, "reference"))
_SPIKE_DIR = os.path.join(_REFERENCE_DIR, "superslm_spike")
sys.path.insert(0, _SPIKE_DIR)
sys.path.insert(0, _REFERENCE_DIR)

import intmath as im  # noqa: E402  (the vendored, hash-pinned reference)

with open(os.path.join(_SPIKE_DIR, "rope_tables_pinned.json"), "r", encoding="ascii") as _f:
    _PINNED_ROPE = json.load(_f)

# Same pin gen_s3_3_fixtures.py applies (S-HARDEN-5 design Sec3.2): im._LN2
# resolved from the pinned double, never from this runner's live libm.
im._LN2 = float.fromhex(_PINNED_ROPE["ln2"]["hex"])

OUT_PATH = os.path.join(_THIS_DIR, "sslm_s3_3_red_regression_fixtures.h")

INT64_MAX = (1 << 63) - 1
QFMT = 30
_IEXP_LN2_Q = int(im._LN2 * (1 << QFMT))
_IEXP_B_Q = int(im._POLY_B * (1 << QFMT))
_IEXP_CA_Q = int((im._POLY_C / im._POLY_A) * (1 << QFMT))
K_SOFTMAX_ROW_MAX_SAFE_EXPONENT = (1 << 62) >> 15  # checked_chain_funnel.h's kSoftmaxRowMaxSafeExponent


# =====================================================================
# Finding 1 (Critical) -- CheckSoftmaxRowWidthDomain forms `q_b*q_b + q_c`
# in int64 and overflows on a point the canonical mantissa domain reaches
# (checked_chain_funnel.cpp:326; intmath.h:391-395's own documented reason
# a caller must not re-derive this). Witness: m=2^30, e=-61 -- one of the
# 108 points Poirot's probe found where (q_b, q_c) is fully int64-
# representable but q_b*q_b + q_c is not.
# =====================================================================


def _build_overflow_witness() -> dict:
    m, e = 1 << 30, -61
    q_ln2, q_b, q_c = im.iexp_scale_constants(
        m, e, _IEXP_LN2_Q, QFMT, _IEXP_B_Q, QFMT, _IEXP_CA_Q, QFMT
    )
    true_m = q_b * q_b + q_c  # arbitrary-precision Python int -- not the int64 the shipped predicate forms
    assert true_m > INT64_MAX, "the witness must overflow int64 forming q_b*q_b + q_c, or it no longer pins finding 1"
    assert true_m > K_SOFTMAX_ROW_MAX_SAFE_EXPONENT, "the witness's true M must exceed the ratified 2**47 ceiling"
    return {"m": m, "e": e, "q_ln2": q_ln2, "q_b": q_b, "q_c": q_c, "width": 1}


# =====================================================================
# Finding 2 -- LandingRescale (forward_sites.cpp:172-181) casts m_a to
# uint64_t unconditionally on a comment claiming it is "positive by
# construction"; a mid-composition carried mantissa need only fit int32_t's
# range (checked_chain_funnel.h), so a negative m_a is reachable. Witness:
# branch_code=5, m_a=-2^30, r_t=3e9, e_a=e_t=0 (Popper Sec3.1), plus the
# positive-sign control at the same magnitude -- the mutation the shipped
# sign handling must distinguish and does not.
# =====================================================================


def _build_negative_ma_witness() -> dict:
    branch_code, r_t, e_a, e_t = 5, 3_000_000_000, 0, 0
    m_a_pos = 1 << 30
    m_a_neg = -(1 << 30)
    correct_raw_pos = im.residual_reconcile(branch_code, m_a_pos, r_t, e_a, e_t)
    correct_raw_neg = im.residual_reconcile(branch_code, m_a_neg, r_t, e_a, e_t)
    assert correct_raw_neg == -correct_raw_pos, (
        "residual_reconcile must be odd-symmetric in m_a at fixed branch_code/r_t/e_a/e_t "
        f"-- got {correct_raw_pos} at +m_a and {correct_raw_neg} at -m_a"
    )
    return {
        "branch_code": branch_code, "r_t": r_t, "e_a": e_a, "e_t": e_t,
        "m_a_pos": m_a_pos, "m_a_neg": m_a_neg,
        "correct_raw_pos": correct_raw_pos, "correct_raw_neg": correct_raw_neg,
    }


# =====================================================================
# Finding 3 -- neither KVC1 landing exponent word (e_t, e_a) has a domain
# check anywhere in the tree. Witness: branch_code=5, m_a=2^30 (canonical),
# r_t=3e9, e_a=0, e_t=-1000 (Popper Sec3.2) -- every OTHER operand at a
# realistic, in-domain value, isolating this one axis. The true reference
# result does not fit int64_t (it is a 1000+ bit integer); what this
# fixture pins is that its MAGNITUDE exceeds the clamp range, so the
# saturation counter must fire -- the one safety net D-SLM201 built for
# exactly this class of event.
# =====================================================================


def _build_extreme_et_witness() -> dict:
    branch_code, m_a, r_t, e_a, e_t = 5, 1 << 30, 3_000_000_000, 0, -1000
    reference = im.residual_reconcile(branch_code, m_a, r_t, e_a, e_t)
    assert abs(reference) > 127, (
        "the reference's true magnitude must exceed the clamp range, or this witness no longer "
        "pins 'the saturation counter must fire but does not'"
    )
    return {
        "branch_code": branch_code, "m_a": m_a, "r_t": r_t, "e_a": e_a, "e_t": e_t,
        "reference_bit_length": reference.bit_length(), "reference_positive": reference > 0,
    }


# =====================================================================
# Finding 5 -- SoftmaxRowQ15 discards IExpConstruct's [[nodiscard]]
# outcome (src/intmath.cpp:552), and CheckSoftmaxRowWidthDomain does not
# take q_ln2, so no gate anywhere covers the kBadQLn2 outcome. Witness:
# m=2^30 (canonical), e=-10 -- one of the 150 (m, e) points in the
# canonical mantissa domain (all of e >= -10, the coarse-scale end) where
# iexp_scale_constants degenerates to (q_ln2, q_b, q_c) = (0, 0, 0).
# =====================================================================


def _build_qln2_zero_witness() -> dict:
    m, e = 1 << 30, -10
    q_ln2, q_b, q_c = im.iexp_scale_constants(
        m, e, _IEXP_LN2_Q, QFMT, _IEXP_B_Q, QFMT, _IEXP_CA_Q, QFMT
    )
    assert q_ln2 == 0, "the witness must degenerate to q_ln2 == 0, or it no longer pins finding 5"
    return {"m": m, "e": e, "q_ln2": q_ln2, "q_b": q_b, "q_c": q_c, "width": 3}


# =====================================================================
# New finding A (Critical) -- Claude/Poirot/ad6bd09-s3.3-remediation-
# confirmation-review-2026-07-28.md. LandingRescale's round-divide branch
# (k >= 0, forward_sites.cpp's `raw = static_cast<int64_t>(U128ShrToU64(...))`)
# narrows a >64-bit true quotient with NO loss detection of its own -- the
# identical class the left-shift branch's own fix (commit 3bbb11e) closed,
# left open on this branch, whose own comment claims the opposite ("this
# branch needs no loss detection of its own"). Witness: branch_code=100,
# m_a=-2^31+1, r_t=2^31+1, e_a=2, e_t=-60 (k=0) -- one of 111 rows the
# review's own probe found where the true residual_reconcile result is a
# 69-bit integer and the shipped narrowing returns a value INSIDE
# [-127, 127]: the wrong answer is indistinguishable from an ordinary
# activation code, so the symptom alone cannot catch it -- only the
# saturation counter can, and it does not fire.
# =====================================================================


def _build_round_divide_in_band_witness() -> dict:
    branch_code, m_a, r_t, e_a, e_t = 100, -2147483647, 2147483649, 2, -60
    k = 62 - (e_a - e_t)
    assert k >= 0, (
        "this witness must land on the round-divide branch (k >= 0), or it no longer pins finding A"
    )
    reference = im.residual_reconcile(branch_code, m_a, r_t, e_a, e_t)
    assert abs(reference) > 127, (
        "the reference's true magnitude must exceed the clamp range, or this witness no longer "
        "pins 'the saturation counter must fire but does not'"
    )
    return {
        "branch_code": branch_code, "m_a": m_a, "r_t": r_t, "e_a": e_a, "e_t": e_t,
        "reference_bit_length": reference.bit_length(), "reference_positive": reference > 0,
    }


# =====================================================================
# Emission
# =====================================================================


def _fmt_bool(b: bool) -> str:
    return "true" if b else "false"


def generate() -> str:
    overflow = _build_overflow_witness()
    neg_ma = _build_negative_ma_witness()
    extreme_et = _build_extreme_et_witness()
    qln2_zero = _build_qln2_zero_witness()
    round_divide_in_band = _build_round_divide_in_band_witness()

    lines = [
        "// GENERATED FILE. Do not hand-edit.",
        "//",
        "// Produced by tests/gen_s3_3_red_regression_fixtures.py -- witnesses for",
        "// the six S3.3 attention-interior defects confirmed by execution against",
        "// the green build at D:\\SuperSLM@c314a64 (Claude/Poirot/c314a64-s3.3-",
        "// attention-interior-review-2026-07-28.md; Claude/Popper/superslm-c27-kv-",
        "// landing-domain-bounds-debunk-2026-07-28.md), PLUS one new finding a",
        "// remediation of those six introduced (Claude/Poirot/ad6bd09-s3.3-",
        "// remediation-confirmation-review-2026-07-28.md finding A). iexp_scale_",
        "// constants/residual_reconcile values are computed by CALLING the vendored",
        "// reference (tests/reference/superslm_spike/intmath.py), never re-derived",
        "// from its formula in this module.",
        "//",
        "// Re-running this script must reproduce this file byte-for-byte.",
        "//",
        "// Test-design record:",
        "// Claude/Curie/superslm-s3.3-attention-interior-red-regression-2026-07-28.md",
        "// Claude/Curie/superslm-s3.3-remediation-confirmation-red-regression-",
        "// 2026-07-28.md",
        "#ifndef SUPERSLM_TESTS_SSLM_S3_3_RED_REGRESSION_FIXTURES_H",
        "#define SUPERSLM_TESTS_SSLM_S3_3_RED_REGRESSION_FIXTURES_H",
        "",
        "#include <cstddef>",
        "#include <cstdint>",
        "",
        "namespace superslm_test {",
        "",
        "// --- Finding 1 (Critical): CheckSoftmaxRowWidthDomain's own int64",
        "// overflow forming q_b*q_b + q_c (checked_chain_funnel.cpp:326) ---",
        "",
        "struct SoftmaxRowOverflowWitness {",
        "\tint64_t m; int e;",
        "\tint64_t q_ln2, q_b, q_c;",
        "\tsize_t width;",
        "};",
        "",
        "inline constexpr SoftmaxRowOverflowWitness kSoftmaxRowOverflowWitness = {",
        f"\t/*m=*/{overflow['m']}LL, /*e=*/{overflow['e']},",
        f"\t/*q_ln2=*/{overflow['q_ln2']}LL, /*q_b=*/{overflow['q_b']}LL, /*q_c=*/{overflow['q_c']}LL,",
        f"\t/*width=*/{overflow['width']}u,",
        "};",
        "",
        "// --- Finding 2: LandingRescale's unconditional uint64_t cast of m_a",
        "// on a false 'positive by construction' precondition ---",
        "// (forward_sites.cpp:172-181) ---",
        "",
        "struct LandingNegativeMaWitness {",
        "\tint64_t branch_code, r_t; int e_a, e_t;",
        "\tint64_t m_a_pos, m_a_neg;",
        "\tint64_t correct_raw_pos, correct_raw_neg;",
        "};",
        "",
        "inline constexpr LandingNegativeMaWitness kLandingNegativeMaWitness = {",
        f"\t/*branch_code=*/{neg_ma['branch_code']}LL, /*r_t=*/{neg_ma['r_t']}LL, "
        f"/*e_a=*/{neg_ma['e_a']}, /*e_t=*/{neg_ma['e_t']},",
        f"\t/*m_a_pos=*/{neg_ma['m_a_pos']}LL, /*m_a_neg=*/{neg_ma['m_a_neg']}LL,",
        f"\t/*correct_raw_pos=*/{neg_ma['correct_raw_pos']}LL, "
        f"/*correct_raw_neg=*/{neg_ma['correct_raw_neg']}LL,",
        "};",
        "",
        "// --- Finding 3: neither KVC1 landing exponent word (e_t, e_a) has a",
        "// domain check anywhere in the tree; an extreme e_t silently returns 0",
        "// with the saturation counter untouched (forward_sites.cpp) ---",
        "",
        "struct LandingExtremeExponentWitness {",
        "\tint64_t branch_code, m_a, r_t; int e_a, e_t;",
        "\t// The true reference (residual_reconcile) does not fit int64_t -- its",
        "\t// bit length and sign are recorded for documentation; what the suite",
        "\t// asserts is that its magnitude exceeds the clamp range (127), so the",
        "\t// saturation counter must fire.",
        "\tint reference_bit_length;",
        "\tbool reference_positive;",
        "};",
        "",
        "inline constexpr LandingExtremeExponentWitness kLandingExtremeExponentWitness = {",
        f"\t/*branch_code=*/{extreme_et['branch_code']}LL, /*m_a=*/{extreme_et['m_a']}LL, "
        f"/*r_t=*/{extreme_et['r_t']}LL, /*e_a=*/{extreme_et['e_a']}, /*e_t=*/{extreme_et['e_t']},",
        f"\t/*reference_bit_length=*/{extreme_et['reference_bit_length']}, "
        f"/*reference_positive=*/{_fmt_bool(extreme_et['reference_positive'])},",
        "};",
        "",
        "// --- Finding 5: SoftmaxRowQ15 discards IExpConstruct's [[nodiscard]]",
        "// outcome, and CheckSoftmaxRowWidthDomain does not take q_ln2",
        "// (src/intmath.cpp:552) ---",
        "",
        "struct SoftmaxQLn2ZeroWitness {",
        "\tint64_t m; int e;",
        "\tint64_t q_ln2, q_b, q_c;",
        "\tsize_t width;",
        "};",
        "",
        "inline constexpr SoftmaxQLn2ZeroWitness kSoftmaxQLn2ZeroWitness = {",
        f"\t/*m=*/{qln2_zero['m']}LL, /*e=*/{qln2_zero['e']},",
        f"\t/*q_ln2=*/{qln2_zero['q_ln2']}LL, /*q_b=*/{qln2_zero['q_b']}LL, /*q_c=*/{qln2_zero['q_c']}LL,",
        f"\t/*width=*/{qln2_zero['width']}u,",
        "};",
        "",
        "// --- New finding A (Critical, Poirot ad6bd09 remediation-confirmation",
        "// review): LandingRescale's round-divide branch (k >= 0) narrows a",
        "// >64-bit true quotient with no loss detection, and the shipped raw",
        "// result lands INSIDE [-127, 127] -- an in-band code indistinguishable",
        "// from an ordinary activation (forward_sites.cpp) ---",
        "",
        "struct LandingRoundDivideInBandWitness {",
        "\tint64_t branch_code, m_a, r_t; int e_a, e_t;",
        "\t// The true reference (residual_reconcile) does not fit int64_t. What",
        "\t// this suite asserts is that its magnitude exceeds the clamp range",
        "\t// (127), so the saturation counter must fire -- the shipped `raw` the",
        "\t// test itself computes and prints on failure is the dangerous part:",
        "\t// it is INSIDE the clamp band, not merely wrong.",
        "\tint reference_bit_length;",
        "\tbool reference_positive;",
        "};",
        "",
        "inline constexpr LandingRoundDivideInBandWitness kLandingRoundDivideInBandWitness = {",
        f"\t/*branch_code=*/{round_divide_in_band['branch_code']}LL, "
        f"/*m_a=*/{round_divide_in_band['m_a']}LL, /*r_t=*/{round_divide_in_band['r_t']}LL, "
        f"/*e_a=*/{round_divide_in_band['e_a']}, /*e_t=*/{round_divide_in_band['e_t']},",
        f"\t/*reference_bit_length=*/{round_divide_in_band['reference_bit_length']}, "
        f"/*reference_positive=*/{_fmt_bool(round_divide_in_band['reference_positive'])},",
        "};",
        "",
        "}  // namespace superslm_test",
        "",
        "#endif  // SUPERSLM_TESTS_SSLM_S3_3_RED_REGRESSION_FIXTURES_H",
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

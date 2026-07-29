#!/usr/bin/env python3
"""Generates sslm_c32_softmax_row_width_gate_fixtures.h -- witnesses for the
three C32 softmax-row width-gate defects confirmed by execution
(Claude/Popper/superslm-c32-softmax-denominator-2026-07-28.md) against the
green build at D:\\SuperSLM@1bf6638. `CheckSoftmaxRowWidthDomain` gates the
row on the closed-form proxy `M = q_b*q_b + q_c` (the i-exp value at the
shifted-max element) and returns `Ok` on every one of these triples, while
`SoftmaxRowQ15`'s int64 `total` genuinely overflows on the first witness.

WHAT THIS PINS, AND HOW. Every numeric truth that cannot be read off the
predicate's own int64 arithmetic (the row's REAL per-element i-exp values,
the exact-precision sum, and its two's-complement wrap to int64) is computed
by CALLING the vendored, hash-pinned reference
(tests/reference/superslm_spike/intmath.py's `i_exp_from_constants`, and
pipeline_prob_width_ceiling.py's `PROB_WIDTH_CEILING` -- the narrow
pipeline.py excerpt PROVENANCE.md already documents; the rest of pipeline.py
is deliberately never read here, matching gen_s3_3_red_regression_fixtures.py's
own established discipline) -- never re-implemented from either function's
formula in this module.

Re-running this script must reproduce this file byte-for-byte.

Casebook: Claude/Popper/superslm-c32-softmax-denominator-2026-07-28.md
Test-design record:
Claude/Curie/superslm-c32-softmax-row-width-gate-test-design-2026-07-28.md
"""
from __future__ import annotations

import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REFERENCE_DIR = os.path.normpath(os.path.join(_THIS_DIR, "reference"))
_SPIKE_DIR = os.path.join(_REFERENCE_DIR, "superslm_spike")
sys.path.insert(0, _SPIKE_DIR)
sys.path.insert(0, _REFERENCE_DIR)

import intmath as im  # noqa: E402  (the vendored, hash-pinned reference)
import pipeline_prob_width_ceiling as pc  # noqa: E402  (the vendored, narrow excerpt)

OUT_PATH = os.path.join(_THIS_DIR, "sslm_c32_softmax_row_width_gate_fixtures.h")

INT64_MAX = (1 << 63) - 1
INT64_MIN = -(1 << 63)


def _wrap_to_int64(n: int) -> int:
    """Two's-complement wrap of an arbitrary-precision int into int64_t's range --
    the same mirror Popper's own casebook cross-checked the real C++ accumulator
    against (Null 3)."""
    n &= (1 << 64) - 1
    return n - (1 << 64) if n >= (1 << 63) else n


# =====================================================================
# Witness 1 -- the off-ratio witness (Popper Null 1/Null 3). M = q_b*q_b +
# q_c = 0 is not an upper bound on this row's real i-exp values off the
# undocumented shortcut ratio `2*q_b >= q_ln2 - 1`, and
# CheckSoftmaxRowWidthDomain has no q_ln2 parameter with which to check that
# ratio. scores[1] is a legitimate max-shifted logit (still inside the z=0
# ln2-band, not clipped) whose real i-exp value is ~9e16 times M.
# =====================================================================


def _build_off_ratio_witness() -> dict:
    q_ln2, q_b, q_c = 3000000001, 10, -100
    width = 3
    scores = [0, -3000000000, -2999999999]
    m = q_b * q_b + q_c
    assert m == 0, f"this witness's own premise (M == 0) no longer holds -- got M={m}"

    exact_values = [im.i_exp_from_constants(s, q_ln2, q_b, q_c) for s in scores]
    exact_total = sum(exact_values)
    peak = max(exact_values)
    assert peak > pc.PROB_WIDTH_CEILING, (
        "the witness's real per-element peak must exceed PROB_WIDTH_CEILING (2**47), or M's "
        "unsoundness as a bound is no longer demonstrated by this witness"
    )
    assert exact_total > INT64_MAX, (
        "the witness's exact-precision total must exceed INT64_MAX, or SoftmaxRowQ15's `total` no "
        "longer genuinely overflows on it"
    )
    wrapped_total = _wrap_to_int64(exact_total)
    return {
        "q_ln2": q_ln2, "q_b": q_b, "q_c": q_c, "width": width, "scores": scores,
        "exact_total_wrapped_i64": wrapped_total,
    }


# =====================================================================
# Witness 2 -- the m == 0 short-circuit is independent of width (Popper
# Null 2, first bullet). Same (q_b, q_c) as witness 1 (M == 0); a width near
# the artifact format's own admitted context_cap ceiling (uint32_t,
# model.h/model.cpp:463) rather than the small width=3 witness 1 uses, so
# the pin does not depend on width being small by coincidence.
# =====================================================================


def _build_m_zero_wide_witness() -> dict:
    q_b, q_c = 10, -100
    m = q_b * q_b + q_c
    assert m == 0, f"this witness's own premise (M == 0) no longer holds -- got M={m}"
    width = 4_000_000_000  # inside [1, 2**32 - 1], the artifact's own admitted context_cap range
    return {"q_b": q_b, "q_c": q_c, "width": width}


# =====================================================================
# Witness 3 -- CheckSoftmaxRowWidthDomain's sum limb must not become MORE
# permissive when M's sign flips negative at fixed magnitude (Popper Null 2,
# second bullet: a negative m defeats `width > INT64_MAX / m` via a
# signed-to-size_t cast). `m_magnitude` is the largest value representable
# under the D-SLM367 ceiling (2**47 - 1); `reject_width` is derived, not
# guessed, from INT64_MAX // m_magnitude, so the positive-sign control is
# independently confirmed to be genuinely rejected by the sum limb before
# the negative-sign case is asserted to wrongly pass it.
# =====================================================================


def _build_sign_asymmetry_witness() -> dict:
    m_magnitude = pc.PROB_WIDTH_CEILING - 1  # 2**47 - 1, the largest M under the ceiling
    reject_threshold_width = INT64_MAX // m_magnitude
    reject_width = reject_threshold_width + 33464  # comfortably clear of the threshold, not adjacent
    assert reject_width > reject_threshold_width, (
        "reject_width must exceed INT64_MAX // m_magnitude, or the positive-sign control below is "
        "not actually rejected by the sum limb"
    )
    return {
        "m_magnitude": m_magnitude,
        "width": reject_width,
    }


# =====================================================================
# Witness 4 -- the kernel's own per-element ceiling check
# (src/intmath.cpp:598-602, D-SLM380) is pinned by nothing (Poirot
# fa3189a-s3.3-rope-site-and-c32-softmax-review-2026-07-28.md, Significant 3):
# every witness above carries `q_c <= 0`, so `CheckSoftmaxRowWidthDomain`'s own
# `q_c < 0` rejection (checked_chain_funnel.cpp:400-402) satisfies every
# gate-then-kernel property cell before the kernel is ever reached. Same
# q_ln2/q_b/scores as witness 1 -- the off-ratio mechanism is identical (at
# q=0 the shifted-max element returns exactly M; at q=-3000000000 the huge
# q_ln2 paired with a tiny q_b sends the real evaluated value far past M) --
# only q_c differs: 0 instead of -100, the smallest change that clears the
# gate's `q_c >= 0` rejection while leaving M trivially inside the ratified
# ceiling, so the gate accepts and the kernel is the only thing left standing
# between this row and a reported well-formed result.
# =====================================================================


def _build_off_ratio_nonnegative_qc_witness() -> dict:
    q_ln2, q_b, q_c = 3000000001, 10, 0
    width = 3
    scores = [0, -3000000000, -2999999999]
    m = q_b * q_b + q_c
    assert q_c >= 0, "this witness must clear the gate's own q_c >= 0 rejection"
    assert 0 <= m <= pc.PROB_WIDTH_CEILING, (
        f"M must be inside the ratified ceiling so the gate accepts this witness, got M={m}"
    )
    exact_values = [im.i_exp_from_constants(s, q_ln2, q_b, q_c) for s in scores]
    peak = max(exact_values)
    assert peak > m, (
        "the witness's real per-element peak must exceed M, or the off-ratio premise this cell "
        "exists to pin is not actually demonstrated"
    )
    return {"q_ln2": q_ln2, "q_b": q_b, "q_c": q_c, "width": width, "scores": scores}


# =====================================================================
# Witness 5 -- T-1324 (Poirot 72b0c7f-s3.3-rope-site-and-c32-softmax-
# confirmation-2026-07-28.md, Significant 4 / D-SLM409): `m_usable`
# (src/intmath.cpp) admits any M representable in int64_t and >= 1 -- it
# does not bound M against `kSoftmaxRowMaxSafeExponent` (2**47), the
# ceiling the header's own contract (intmath.h:521-528) attributes the
# `exps[k] << kProbFracBits` shift's safety to. A single-element row
# (width=1, scores=[0]) whose shifted-max element evaluates to EXACTLY M
# (q=0 gives z=0, base=q_b, so i_exp_from_constants == q_b**2 + q_c == M)
# isolates the shift from every other moving part: q_b=0 puts the whole
# magnitude in q_c, chosen far past the safe ceiling but still safely
# representable in int64_t, so `m_usable`'s existing two conjuncts both
# pass and only the missing third conjunct stands between this row and a
# reported well-formed result. Reachable only by a caller that calls the
# kernel directly without the width gate (CheckSoftmaxRowWidthDomain
# genuinely rejects this M, grounded below) -- exactly the "ungated
# caller" case the header's own text names and does not cover for this
# specific claim.
# =====================================================================


def _build_ungated_shift_overflow_witness() -> dict:
    q_ln2, q_b, q_c = 3000000001, 0, 1 << 60
    width = 1
    scores = [0]
    m = q_b * q_b + q_c
    assert m == 1 << 60, f"this witness's own premise (M == 2**60) no longer holds -- got M={m}"
    assert m > pc.PROB_WIDTH_CEILING, (
        "M must exceed kSoftmaxRowMaxSafeExponent (2**47), or this witness no longer "
        "demonstrates an M the safe ceiling would reject"
    )
    assert m <= INT64_MAX, (
        "M must still be representable in int64_t (m_usable's existing, correct conjunct), or "
        "this witness stops isolating the missing ceiling conjunct from the existing "
        "representability one"
    )
    real_value = im.i_exp_from_constants(scores[0], q_ln2, q_b, q_c)
    assert real_value == m, (
        "the shifted-max element (scores[0] == 0, the row's only element) must evaluate to "
        f"EXACTLY M, or this witness no longer isolates the shift from the row's other "
        f"arithmetic -- got real_value={real_value}, M={m}"
    )
    return {"q_ln2": q_ln2, "q_b": q_b, "q_c": q_c, "width": width, "scores": scores, "m": m}


# =====================================================================
# Emission
# =====================================================================


def generate() -> str:
    off_ratio = _build_off_ratio_witness()
    off_ratio_nonneg_qc = _build_off_ratio_nonnegative_qc_witness()
    m_zero_wide = _build_m_zero_wide_witness()
    sign_asym = _build_sign_asymmetry_witness()
    ungated_shift_overflow = _build_ungated_shift_overflow_witness()

    scores_line = ", ".join(f"{s}LL" for s in off_ratio["scores"])
    nonneg_qc_scores_line = ", ".join(f"{s}LL" for s in off_ratio_nonneg_qc["scores"])
    shift_overflow_scores_line = ", ".join(f"{s}LL" for s in ungated_shift_overflow["scores"])

    lines = [
        "// GENERATED FILE. Do not hand-edit.",
        "//",
        "// Produced by tests/gen_c32_softmax_row_width_gate_fixtures.py -- witnesses for",
        "// the three C32 softmax-row width-gate defects confirmed by execution against",
        "// the green build at D:\\SuperSLM@1bf6638",
        "// (Claude/Popper/superslm-c32-softmax-denominator-2026-07-28.md), plus a fifth",
        "// witness (T-1324, D-SLM409) pinning the missing bound on `e[k] << PROB_FRAC_BITS`",
        "// that plan Sec14.1 owes (Poirot 72b0c7f-s3.3-rope-site-and-c32-softmax-",
        "// confirmation-2026-07-28.md, Significant 4).",
        "// exact_total_wrapped_i64 and the peak/ceiling comparison are computed by CALLING",
        "// the vendored reference (tests/reference/superslm_spike/intmath.py's",
        "// i_exp_from_constants, pipeline_prob_width_ceiling.py's PROB_WIDTH_CEILING),",
        "// never re-derived from either function's formula in this module.",
        "//",
        "// Re-running this script must reproduce this file byte-for-byte.",
        "//",
        "// Test-design record:",
        "// Claude/Curie/superslm-c32-softmax-row-width-gate-test-design-2026-07-28.md,",
        "// Claude/Curie/72b0c7f-s3.3-rope-site-and-c32-softmax-confirmation-test-design-",
        "// 2026-07-28.md",
        "#ifndef SUPERSLM_TESTS_SSLM_C32_SOFTMAX_ROW_WIDTH_GATE_FIXTURES_H",
        "#define SUPERSLM_TESTS_SSLM_C32_SOFTMAX_ROW_WIDTH_GATE_FIXTURES_H",
        "",
        "#include <cstddef>",
        "#include <cstdint>",
        "",
        "namespace superslm_test {",
        "",
        "// --- Witness 1: the off-ratio witness (Popper Null 1/Null 3). M = q_b*q_b + q_c",
        "// == 0 is not an upper bound on this row's real i-exp values off the",
        "// undocumented shortcut ratio 2*q_b >= q_ln2 - 1, and",
        "// CheckSoftmaxRowWidthDomain has no q_ln2 parameter with which to check it. ---",
        "",
        "struct SoftmaxRowOffRatioWitness {",
        "\tint64_t q_ln2, q_b, q_c;",
        "\tsize_t width;",
        "\tint64_t scores[3];",
        "\t// Two's-complement wrap of the EXACT-PRECISION sum of this row's real i-exp",
        "\t// values into int64_t's range -- the same mirror Popper's own casebook",
        "\t// cross-checked the real C++ accumulator against, bit-identical.",
        "\tint64_t exact_total_wrapped_i64;",
        "};",
        "",
        "inline constexpr SoftmaxRowOffRatioWitness kSoftmaxRowOffRatioWitness = {",
        f"\t/*q_ln2=*/{off_ratio['q_ln2']}LL, /*q_b=*/{off_ratio['q_b']}LL, /*q_c=*/{off_ratio['q_c']}LL,",
        f"\t/*width=*/{off_ratio['width']}u,",
        f"\t/*scores=*/{{{scores_line}}},",
        f"\t/*exact_total_wrapped_i64=*/{off_ratio['exact_total_wrapped_i64']}LL,",
        "};",
        "",
        "// --- Witness 4 (Poirot fa3189a review, Significant 3): the same off-ratio",
        "// mechanism as witness 1, with q_c >= 0 so CheckSoftmaxRowWidthDomain's own",
        "// `q_c < 0` rejection does not intercept it before the kernel's per-element",
        "// check (src/intmath.cpp:598-602) is reached. M = q_b*q_b + q_c is trivially",
        "// inside the ratified ceiling; the row's real evaluated peak (at scores[1])",
        "// vastly exceeds it regardless. ---",
        "",
        "struct SoftmaxRowOffRatioNonnegativeQcWitness {",
        "\tint64_t q_ln2, q_b, q_c;",
        "\tsize_t width;",
        "\tint64_t scores[3];",
        "};",
        "",
        "inline constexpr SoftmaxRowOffRatioNonnegativeQcWitness kSoftmaxRowOffRatioNonnegativeQcWitness = {",
        f"\t/*q_ln2=*/{off_ratio_nonneg_qc['q_ln2']}LL, /*q_b=*/{off_ratio_nonneg_qc['q_b']}LL, "
        f"/*q_c=*/{off_ratio_nonneg_qc['q_c']}LL,",
        f"\t/*width=*/{off_ratio_nonneg_qc['width']}u,",
        f"\t/*scores=*/{{{nonneg_qc_scores_line}}},",
        "};",
        "",
        "// --- Witness 2: the m == 0 short-circuit is independent of width (Popper Null 2,",
        "// first bullet). Same (q_b, q_c) as witness 1 (M == 0); width near the",
        "// artifact format's own admitted context_cap ceiling. ---",
        "",
        "struct SoftmaxRowMZeroWideWitness {",
        "\tint64_t q_b, q_c;",
        "\tsize_t width;",
        "};",
        "",
        "inline constexpr SoftmaxRowMZeroWideWitness kSoftmaxRowMZeroWideWitness = {",
        f"\t/*q_b=*/{m_zero_wide['q_b']}LL, /*q_c=*/{m_zero_wide['q_c']}LL,",
        f"\t/*width=*/{m_zero_wide['width']}u,",
        "};",
        "",
        "// --- Witness 3: the sum limb must not become MORE permissive when M's sign",
        "// flips negative at fixed magnitude (Popper Null 2, second bullet).",
        "// m_magnitude is the largest M representable under the D-SLM367 ceiling",
        "// (2**47 - 1); width is derived from INT64_MAX // m_magnitude so the",
        "// positive-sign control is independently confirmed rejected by the sum limb",
        "// before the negative-sign case is asserted to wrongly pass it. ---",
        "",
        "struct SoftmaxRowSignAsymmetryWitness {",
        "\tint64_t m_magnitude;",
        "\tsize_t width;",
        "};",
        "",
        "inline constexpr SoftmaxRowSignAsymmetryWitness kSoftmaxRowSignAsymmetryWitness = {",
        f"\t/*m_magnitude=*/{sign_asym['m_magnitude']}LL,",
        f"\t/*width=*/{sign_asym['width']}u,",
        "};",
        "",
        "// --- Witness 5 (Poirot 72b0c7f confirmation, Significant 4 / D-SLM409): a",
        "// single-element row whose shifted-max element evaluates to EXACTLY M, with M",
        "// chosen past kSoftmaxRowMaxSafeExponent (2**47) but still representable in",
        "// int64_t -- isolates the missing third m_usable conjunct from its two existing,",
        "// correct ones. Reachable only by a caller that calls SoftmaxRowQ15 directly",
        "// without the width gate (CheckSoftmaxRowWidthDomain genuinely rejects this M). ---",
        "",
        "struct SoftmaxRowUngatedShiftOverflowWitness {",
        "\tint64_t q_ln2, q_b, q_c;",
        "\tsize_t width;",
        "\tint64_t scores[1];",
        "\t// M = q_b*q_b + q_c, computed once here so the C++ side never re-derives it.",
        "\tint64_t m;",
        "};",
        "",
        "inline constexpr SoftmaxRowUngatedShiftOverflowWitness kSoftmaxRowUngatedShiftOverflowWitness = {",
        f"\t/*q_ln2=*/{ungated_shift_overflow['q_ln2']}LL, /*q_b=*/{ungated_shift_overflow['q_b']}LL, "
        f"/*q_c=*/{ungated_shift_overflow['q_c']}LL,",
        f"\t/*width=*/{ungated_shift_overflow['width']}u,",
        f"\t/*scores=*/{{{shift_overflow_scores_line}}},",
        f"\t/*m=*/{ungated_shift_overflow['m']}LL,",
        "};",
        "",
        "}  // namespace superslm_test",
        "",
        "#endif  // SUPERSLM_TESTS_SSLM_C32_SOFTMAX_ROW_WIDTH_GATE_FIXTURES_H",
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

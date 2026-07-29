// GENERATED FILE. Do not hand-edit.
//
// Produced by tests/gen_c32_softmax_row_width_gate_fixtures.py -- witnesses for
// the three C32 softmax-row width-gate defects confirmed by execution against
// the green build at D:\SuperSLM@1bf6638
// (Claude/Popper/superslm-c32-softmax-denominator-2026-07-28.md), plus a fifth
// witness (T-1324, D-SLM409) pinning the missing bound on `e[k] << PROB_FRAC_BITS`
// that plan Sec14.1 owes (Poirot 72b0c7f-s3.3-rope-site-and-c32-softmax-
// confirmation-2026-07-28.md, Significant 4).
// exact_total_wrapped_i64 and the peak/ceiling comparison are computed by CALLING
// the vendored reference (tests/reference/superslm_spike/intmath.py's
// i_exp_from_constants, pipeline_prob_width_ceiling.py's PROB_WIDTH_CEILING),
// never re-derived from either function's formula in this module.
//
// Re-running this script must reproduce this file byte-for-byte.
//
// Test-design record:
// Claude/Curie/superslm-c32-softmax-row-width-gate-test-design-2026-07-28.md,
// Claude/Curie/72b0c7f-s3.3-rope-site-and-c32-softmax-confirmation-test-design-
// 2026-07-28.md
#ifndef SUPERSLM_TESTS_SSLM_C32_SOFTMAX_ROW_WIDTH_GATE_FIXTURES_H
#define SUPERSLM_TESTS_SSLM_C32_SOFTMAX_ROW_WIDTH_GATE_FIXTURES_H

#include <cstddef>
#include <cstdint>

namespace superslm_test {

// --- Witness 1: the off-ratio witness (Popper Null 1/Null 3). M = q_b*q_b + q_c
// == 0 is not an upper bound on this row's real i-exp values off the
// undocumented shortcut ratio 2*q_b >= q_ln2 - 1, and
// CheckSoftmaxRowWidthDomain has no q_ln2 parameter with which to check it. ---

struct SoftmaxRowOffRatioWitness {
	int64_t q_ln2, q_b, q_c;
	size_t width;
	int64_t scores[3];
	// Two's-complement wrap of the EXACT-PRECISION sum of this row's real i-exp
	// values into int64_t's range -- the same mirror Popper's own casebook
	// cross-checked the real C++ accumulator against, bit-identical.
	int64_t exact_total_wrapped_i64;
};

inline constexpr SoftmaxRowOffRatioWitness kSoftmaxRowOffRatioWitness = {
	/*q_ln2=*/3000000001LL, /*q_b=*/10LL, /*q_c=*/-100LL,
	/*width=*/3u,
	/*scores=*/{0LL, -3000000000LL, -2999999999LL},
	/*exact_total_wrapped_i64=*/-446744199709551595LL,
};

// --- Witness 4 (Poirot fa3189a review, Significant 3): the same off-ratio
// mechanism as witness 1, with q_c >= 0 so CheckSoftmaxRowWidthDomain's own
// `q_c < 0` rejection does not intercept it before the kernel's per-element
// check (src/intmath.cpp:598-602) is reached. M = q_b*q_b + q_c is trivially
// inside the ratified ceiling; the row's real evaluated peak (at scores[1])
// vastly exceeds it regardless. ---

struct SoftmaxRowOffRatioNonnegativeQcWitness {
	int64_t q_ln2, q_b, q_c;
	size_t width;
	int64_t scores[3];
};

inline constexpr SoftmaxRowOffRatioNonnegativeQcWitness kSoftmaxRowOffRatioNonnegativeQcWitness = {
	/*q_ln2=*/3000000001LL, /*q_b=*/10LL, /*q_c=*/0LL,
	/*width=*/3u,
	/*scores=*/{0LL, -3000000000LL, -2999999999LL},
};

// --- Witness 2: the m == 0 short-circuit is independent of width (Popper Null 2,
// first bullet). Same (q_b, q_c) as witness 1 (M == 0); width near the
// artifact format's own admitted context_cap ceiling. ---

struct SoftmaxRowMZeroWideWitness {
	int64_t q_b, q_c;
	size_t width;
};

inline constexpr SoftmaxRowMZeroWideWitness kSoftmaxRowMZeroWideWitness = {
	/*q_b=*/10LL, /*q_c=*/-100LL,
	/*width=*/4000000000u,
};

// --- Witness 3: the sum limb must not become MORE permissive when M's sign
// flips negative at fixed magnitude (Popper Null 2, second bullet).
// m_magnitude is the largest M representable under the D-SLM367 ceiling
// (2**47 - 1); width is derived from INT64_MAX // m_magnitude so the
// positive-sign control is independently confirmed rejected by the sum limb
// before the negative-sign case is asserted to wrongly pass it. ---

struct SoftmaxRowSignAsymmetryWitness {
	int64_t m_magnitude;
	size_t width;
};

inline constexpr SoftmaxRowSignAsymmetryWitness kSoftmaxRowSignAsymmetryWitness = {
	/*m_magnitude=*/140737488355327LL,
	/*width=*/99000u,
};

// --- Witness 5 (Poirot 72b0c7f confirmation, Significant 4 / D-SLM409; T-1324
// closed at 703dca6, D-SLM412): a single-element row whose shifted-max element
// evaluates to EXACTLY M, with M chosen past kSoftmaxRowMaxSafeExponent (2**47)
// but still representable in int64_t -- before the fix, isolated the missing
// third m_usable conjunct from its two existing, correct ones; the kernel now
// enforces that conjunct directly (intmath.h:515-528). Reachable only by a
// caller that calls SoftmaxRowQ15 directly without the width gate
// (CheckSoftmaxRowWidthDomain genuinely rejects this M). ---

struct SoftmaxRowUngatedShiftOverflowWitness {
	int64_t q_ln2, q_b, q_c;
	size_t width;
	int64_t scores[1];
	// M = q_b*q_b + q_c, computed once here so the C++ side never re-derives it.
	int64_t m;
};

inline constexpr SoftmaxRowUngatedShiftOverflowWitness kSoftmaxRowUngatedShiftOverflowWitness = {
	/*q_ln2=*/3000000001LL, /*q_b=*/0LL, /*q_c=*/1152921504606846976LL,
	/*width=*/1u,
	/*scores=*/{0LL},
	/*m=*/1152921504606846976LL,
};

}  // namespace superslm_test

#endif  // SUPERSLM_TESTS_SSLM_C32_SOFTMAX_ROW_WIDTH_GATE_FIXTURES_H

// GENERATED FILE. Do not hand-edit.
//
// Produced by tests/gen_s3_3_red_regression_fixtures.py -- witnesses for
// the six S3.3 attention-interior defects confirmed by execution against
// the green build at D:\SuperSLM@c314a64 (Claude/Poirot/c314a64-s3.3-
// attention-interior-review-2026-07-28.md; Claude/Popper/superslm-c27-kv-
// landing-domain-bounds-debunk-2026-07-28.md). iexp_scale_constants/
// residual_reconcile values are computed by CALLING the vendored reference
// (tests/reference/superslm_spike/intmath.py), never re-derived from its
// formula in this module.
//
// Re-running this script must reproduce this file byte-for-byte.
//
// Test-design record:
// Claude/Curie/superslm-s3.3-attention-interior-red-regression-2026-07-28.md
#ifndef SUPERSLM_TESTS_SSLM_S3_3_RED_REGRESSION_FIXTURES_H
#define SUPERSLM_TESTS_SSLM_S3_3_RED_REGRESSION_FIXTURES_H

#include <cstddef>
#include <cstdint>

namespace superslm_test {

// --- Finding 1 (Critical): CheckSoftmaxRowWidthDomain's own int64
// overflow forming q_b*q_b + q_c (checked_chain_funnel.cpp:326) ---

struct SoftmaxRowOverflowWitness {
	int64_t m; int e;
	int64_t q_ln2, q_b, q_c;
	size_t width;
};

inline constexpr SoftmaxRowOverflowWitness kSoftmaxRowOverflowWitness = {
	/*m=*/1073741824LL, /*e=*/-61,
	/*q_ln2=*/1488522234LL, /*q_b=*/2905545374LL, /*q_c=*/4425160360470773760LL,
	/*width=*/1u,
};

// --- Finding 2: LandingRescale's unconditional uint64_t cast of m_a
// on a false 'positive by construction' precondition ---
// (forward_sites.cpp:172-181) ---

struct LandingNegativeMaWitness {
	int64_t branch_code, r_t; int e_a, e_t;
	int64_t m_a_pos, m_a_neg;
	int64_t correct_raw_pos, correct_raw_neg;
};

inline constexpr LandingNegativeMaWitness kLandingNegativeMaWitness = {
	/*branch_code=*/5LL, /*r_t=*/3000000000LL, /*e_a=*/0, /*e_t=*/0,
	/*m_a_pos=*/1073741824LL, /*m_a_neg=*/-1073741824LL,
	/*correct_raw_pos=*/3LL, /*correct_raw_neg=*/-3LL,
};

// --- Finding 3: neither KVC1 landing exponent word (e_t, e_a) has a
// domain check anywhere in the tree; an extreme e_t silently returns 0
// with the saturation counter untouched (forward_sites.cpp) ---

struct LandingExtremeExponentWitness {
	int64_t branch_code, m_a, r_t; int e_a, e_t;
	// The true reference (residual_reconcile) does not fit int64_t -- its
	// bit length and sign are recorded for documentation; what the suite
	// asserts is that its magnitude exceeds the clamp range (127), so the
	// saturation counter must fire.
	int reference_bit_length;
	bool reference_positive;
};

inline constexpr LandingExtremeExponentWitness kLandingExtremeExponentWitness = {
	/*branch_code=*/5LL, /*m_a=*/1073741824LL, /*r_t=*/3000000000LL, /*e_a=*/0, /*e_t=*/-1000,
	/*reference_bit_length=*/1002, /*reference_positive=*/true,
};

// --- Finding 5: SoftmaxRowQ15 discards IExpConstruct's [[nodiscard]]
// outcome, and CheckSoftmaxRowWidthDomain does not take q_ln2
// (src/intmath.cpp:552) ---

struct SoftmaxQLn2ZeroWitness {
	int64_t m; int e;
	int64_t q_ln2, q_b, q_c;
	size_t width;
};

inline constexpr SoftmaxQLn2ZeroWitness kSoftmaxQLn2ZeroWitness = {
	/*m=*/1073741824LL, /*e=*/-10,
	/*q_ln2=*/0LL, /*q_b=*/0LL, /*q_c=*/0LL,
	/*width=*/3u,
};

}  // namespace superslm_test

#endif  // SUPERSLM_TESTS_SSLM_S3_3_RED_REGRESSION_FIXTURES_H

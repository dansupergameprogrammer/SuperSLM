// GENERATED FILE. Do not hand-edit.
//
// Produced by tests/gen_s3_3_fixtures.py -- S3.3's derived witnesses
// (attention interior: C27/D-SLM57 landing, C32/D-SLM366 softmax width,
// C33 clamp, the ctx_fold join oracle, KvLandingScales/Reciprocals'
// joint domain). RopeApplyPair/residual_reconcile/dynamic_scale_reciprocal
// values are computed by CALLING the vendored reference
// (tests/reference/superslm_spike/{intmath,rope}.py); the ctx_fold
// join's (mult, shift) is an INDEPENDENT gemmlowp QuantizeMultiplier
// decomposition written from the published algorithm, never from
// Tools/superslm_spike/pipeline.py's own source (F-S3-3, Plan Sec4.3).
//
// The C32 numerator threshold is NOT resolved here -- Dan's ruling is open
// between INT64_MAX>>PROB_FRAC_BITS (2**48-1, this module's own int64-safety
// derivation) and pipeline.py's stricter _guard_probability_width (2**47).
// Every C32 case below carries BOTH candidate verdicts
// (ok_under_2pow48m1/ok_under_2pow47); kC32BandCase is a witness where they
// disagree. See the test-design record Sec5 for the named-constant routing.
//
// Re-running this script must reproduce this file byte-for-byte.
//
// Test-design record:
// Claude/Curie/superslm-s3.3-attention-interior-test-design-2026-07-28.md
#ifndef SUPERSLM_TESTS_SSLM_S3_3_FIXTURES_H
#define SUPERSLM_TESTS_SSLM_S3_3_FIXTURES_H

#include <cstddef>
#include <cstdint>

namespace superslm_test {

// --- Sec1: C33's post-rotation clamp witnesses ---

struct C33BothSignsCase {
	int32_t x, y, cos_q30, sin_q30;
	int64_t raw_x, raw_y;
	int32_t x_neg, y_neg;
	int64_t raw_x_neg, raw_y_neg;
};

inline constexpr C33BothSignsCase kC33BothSignsCase = {
	/*x=*/127, /*y=*/127, /*cos_q30=*/763337599, /*sin_q30=*/-755140526,
	/*raw_x=*/180LL, /*raw_y=*/1LL,
	/*x_neg=*/-127, /*y_neg=*/-127,
	/*raw_x_neg=*/-180LL, /*raw_y_neg=*/-1LL,
};

struct C33Int32OverflowCase {
	int32_t x, y, cos_q30, sin_q30;
	int64_t raw_x, raw_y;
};

inline constexpr C33Int32OverflowCase kC33Int32OverflowCase = {
	/*x=*/2147483647, /*y=*/2147483647, /*cos_q30=*/763337599, /*sin_q30=*/-755140526,
	/*raw_x=*/3036956249LL, /*raw_y=*/16394146LL,
};

// --- Sec2: C32/D-SLM366's softmax-row-width predicate witnesses ---

inline constexpr int64_t kNumeratorLimit2Pow48Minus1 = 281474976710655LL;  // INT64_MAX >> PROB_FRAC_BITS
inline constexpr int64_t kNumeratorLimit2Pow47 = 140737488355328LL;  // pipeline.py's _guard_probability_width

struct C32WidthDomainCase {
	const char* label;
	int64_t m; int e;
	int64_t q_ln2, q_b, q_c;
	int64_t row_max;         // M = q_b^2 + q_c, the row's peak i-exp value (q=0)
	size_t width;
	bool shipped_check_ok;   // IExpConstantsInDomain(0, q_ln2, q_b, q_c)
	bool ok_under_2pow48m1;  // row_max <= kNumeratorLimit2Pow48Minus1
	bool ok_under_2pow47;    // row_max <= kNumeratorLimit2Pow47
};

inline constexpr C32WidthDomainCase kC32WidthDomainCases[] = {
	{ "numerator_overflow_single_element", 2147483647LL, -61, 744261117LL, 1452772687LL, 1106290091148006375LL, 3216838571241206344LL, 1u, true, false, false },
	{ "sum_overflow_small_width_numerator_safe", 1342177279LL, -53, 4651631LL, 9079829LL, 43214456700561LL, 125657751369802LL, 73401u, true, true, true },
	{ "accept_realistic_width", 2147483647LL, -40, 354LL, 692LL, 251541LL, 730405LL, 4096u, true, true, true },
};
inline constexpr size_t kC32WidthDomainCasesCount = sizeof(kC32WidthDomainCases) / sizeof(kC32WidthDomainCases[0]);

// The routed band witness (Sec6): the two candidate thresholds DISAGREE on
// this row -- accepted under 2**48-1, rejected under 2**47. Not resolved by
// this suite; see the test-design record's routed finding.
inline constexpr C32WidthDomainCase kC32BandCase = {
	"band_2pow47_vs_2pow48m1_disagree", 1879048191LL, -54, 6645188LL, 12971184LL, 88192768786923LL, 256444383148779LL, 1u, true, true, false
};

// --- Sec3: C27/D-SLM57's K/V landing composite (residual_reconcile) ---

struct LandingCompositeCase {
	const char* label;
	int64_t branch_code, m_b, r_h; int e_b, e_h;
	int64_t correct_raw; int8_t correct;
	int64_t wrong_per_key_reciprocal_raw; int8_t wrong_per_key_reciprocal;
	int64_t wrong_two_rounding_raw; int8_t wrong_two_rounding;
	bool wrong_pk_diverges;
	bool wrong_2r_diverges;
};

inline constexpr LandingCompositeCase kLandingCompositeCases[] = {
	{ "positive_branch_diverges", 3LL, 1073741824LL, 3074457346LL, 0, 0, 2LL, 2, 3LL, 3, 3LL, 3, true, true },
	{ "negative_branch_diverges", -55LL, 1073741824LL, 3074457346LL, 0, 0, -39LL, -39, -55LL, -55, -40LL, -40, true, true },
	{ "zero_branch", 0LL, 1073741824LL, 4294967296LL, -10, -10, 0LL, 0, 0LL, 0, 0LL, 0, false, false },
};
inline constexpr size_t kLandingCompositeCasesCount = sizeof(kLandingCompositeCases) / sizeof(kLandingCompositeCases[0]);

// --- Sec4: the ctx_fold join oracle (independent QuantizeMultiplier) ---

struct CtxFoldJoinCase {
	double ratio; int32_t mult; int32_t shift;
	int64_t ctx0, ctx1;
	int64_t expected0, expected1;
};

inline constexpr CtxFoldJoinCase kCtxFoldJoinCase = {
	0.5, 1073741824, 0,
	1073741825LL, -536870912LL,
	1073741825LL, -268435456LL,
};

// --- Sec5: KvLandingScales/KvLandingReciprocals' joint domain (Sec7.2a) ---

inline constexpr int64_t kKvLandingScaleMantissaMin = 1073741824LL;
inline constexpr int64_t kKvLandingScaleMantissaMax = 2147483647LL;
inline constexpr int64_t kKvLandingReciprocalMin = 2147483649LL;
inline constexpr int64_t kKvLandingReciprocalMax = 4294967296LL;

}  // namespace superslm_test

#endif  // SUPERSLM_TESTS_SSLM_S3_3_FIXTURES_H

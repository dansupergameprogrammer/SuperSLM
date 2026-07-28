// Hand-authored fixture data for SuperSLM_S3a_WalkingSkeleton_Plan.md Sec11 S3.1
// (F-S3-7, T-200, Sec17.3 cell 5) -- the int64 (wide) row-width siblings of the
// already-generated S2.1 intmath fixtures, plus the funnel's own T-1254 witness
// rows.
//
// NOT a generator output, unlike sslm_intmath_fixtures.h and
// sslm_s3_1_c30_iexp_domain_sweep_fixtures.h. Every wide MaxAbsReduceWide/
// RequantTokenCodeWide case here is the SAME data already generated and
// witnessed at int32 input width in sslm_intmath_fixtures.h (kMaxAbsCases,
// kRequantCases), reused unchanged because the formula and the expected outputs
// do not change when the input's C++ type widens from int32_t to int64_t --
// only the value domain the parameter can carry does (SuperSLM_S3a_
// WalkingSkeleton_Plan.md Sec11 S3.1, Sec4.7's correction). No new derivation
// happens in this file; hand-authoring it is not exempt from that rule, it
// simply has nothing left to derive. RowBoundsWide is new (Sec5.5) and its
// fixture is constructed directly from the design's own T-1254 witness rows,
// which are pinned in the plan text verbatim (Sec11 S3.1, "the strike's own
// witness (T-1254), required green").
//
// Test-design record:
// Claude/Curie/superslm-s3.1-checked-chain-funnel-test-design-2026-07-28.md
#ifndef SUPERSLM_TESTS_SSLM_S3_1_WIDE_INTMATH_FIXTURES_H
#define SUPERSLM_TESTS_SSLM_S3_1_WIDE_INTMATH_FIXTURES_H

#include "superslm/intmath.h"

#include <cstddef>
#include <cstdint>

namespace superslm_test {

// --- C20 at int64 width: MaxAbsReduceWide ----------------------------------

// The T-1254 witness row (SuperSLM_S3a_WalkingSkeleton_Plan.md Sec11 S3.1):
// shared verbatim with the RequantChainChecked/NarrowRowChecked witness rows
// below, so a regression in one surfaces at every entry point that reads it.
inline constexpr int64_t kWideT1254WitnessPositiveRow[] = {
	INT64_C(2147483648),  // +2^31
	INT64_C(1073741824),  // 2^30
	INT64_C(-5),
	INT64_C(0),
};
inline constexpr int64_t kWideT1254WitnessNegatedRow[] = {
	INT64_C(-2147483648),  // -2^31 (the extreme element negated)
	INT64_C(1073741824),   // 2^30
	INT64_C(-5),
	INT64_C(0),
};
inline constexpr size_t kWideT1254WitnessRowLen = 4;
inline constexpr int64_t kWideT1254WitnessDPrime = INT64_C(2147483648);  // 2^31

// A single-element row at D' = 2^31 + 1 -- one past the T-1254 witness's own
// D', and one past C29's domain boundary (Sec11 S3.1's own red cell: "a wide
// row at D' = 2^31 + 1 returns ChainInputOutOfDomain ... and computes
// nothing"). "Any row" per the plan's own wording; a single element is
// sufficient and keeps the reject-path fixture minimal.
inline constexpr int64_t kWideOverC29DomainRow[] = {INT64_C(2147483649)};  // 2^31 + 1
inline constexpr size_t kWideOverC29DomainRowLen = 1;

// Widened S2.1 MaxAbsReduce cases (kMaxAbsCases, sslm_intmath_fixtures.h):
// same multisets, int64_t element type, same expected magnitude -- the
// generator's own values, reused rather than re-derived, because widening the
// element type changes nothing about the max-abs reduction of values that
// already fit comfortably in int32.
inline constexpr int64_t kMaxAbsWideData0[] = {INT64_C(1), INT64_C(-5), INT64_C(3), INT64_C(-10), INT64_C(7)};  // typical
inline constexpr int64_t kMaxAbsWideData1[] = {INT64_C(5), INT64_C(-2147483648), INT64_C(-3)};  // contains_int32_min
inline constexpr int64_t kMaxAbsWideData2[] = {INT64_C(0), INT64_C(0), INT64_C(0), INT64_C(0)};  // all_zero_guard
inline constexpr int64_t kMaxAbsWideData3[] = {INT64_C(42)};  // single_positive
inline constexpr int64_t kMaxAbsWideData4[] = {INT64_C(-42)};  // single_negative
inline constexpr int64_t kMaxAbsWideData5[] = {INT64_C(-2147483648)};  // single_int32_min
inline constexpr int64_t kMaxAbsWideData6[] = {INT64_C(0)};  // single_zero
inline constexpr int64_t kMaxAbsWideData7[] = {INT64_C(1), INT64_C(2147483647), INT64_C(-7)};  // contains_int32_max
inline constexpr int64_t kMaxAbsWideData8[] = {INT64_C(3), INT64_C(-100), INT64_C(7), INT64_C(-55), INT64_C(100)};  // order_perm_a
inline constexpr int64_t kMaxAbsWideData9[] = {INT64_C(100), INT64_C(-55), INT64_C(7), INT64_C(-100), INT64_C(3)};  // order_perm_b
inline constexpr int64_t kMaxAbsWideData10[] = {INT64_C(-100), INT64_C(-55), INT64_C(3), INT64_C(100), INT64_C(7)};  // order_perm_c_shuffled

struct MaxAbsWideCase {
	const char* label;
	const int64_t* data;
	size_t n;
	int64_t expected;
};

inline constexpr MaxAbsWideCase kMaxAbsWideCases[] = {
	{"t1254_witness_row_d_prime_2_pow_31", kWideT1254WitnessPositiveRow, kWideT1254WitnessRowLen, kWideT1254WitnessDPrime},
	{"t1254_witness_row_negated_same_d_prime", kWideT1254WitnessNegatedRow, kWideT1254WitnessRowLen, kWideT1254WitnessDPrime},
	{"typical", kMaxAbsWideData0, 5, INT64_C(10)},
	{"contains_int32_min", kMaxAbsWideData1, 3, INT64_C(2147483648)},
	{"all_zero_guard", kMaxAbsWideData2, 4, INT64_C(1)},
	{"single_positive", kMaxAbsWideData3, 1, INT64_C(42)},
	{"single_negative", kMaxAbsWideData4, 1, INT64_C(42)},
	{"single_int32_min", kMaxAbsWideData5, 1, INT64_C(2147483648)},
	{"single_zero", kMaxAbsWideData6, 1, INT64_C(1)},
	{"contains_int32_max", kMaxAbsWideData7, 3, INT64_C(2147483647)},
	{"order_perm_a", kMaxAbsWideData8, 5, INT64_C(100)},
	{"order_perm_b", kMaxAbsWideData9, 5, INT64_C(100)},
	{"order_perm_c_shuffled", kMaxAbsWideData10, 5, INT64_C(100)},
};
inline constexpr size_t kMaxAbsWideCasesCount = 13;

// --- New (Sec5.5): RowBoundsWide --------------------------------------------
//
// Signed max/min, order-free -- distinct from MaxAbsReduceWide because int32
// narrowing (NarrowRowChecked, C35) depends on the row's signed extremes, not
// its magnitude. A magnitude-only bound cannot express the asymmetric int32
// target range [INT32_MIN, INT32_MAX]; that is exactly what the funnel-level
// C35-vs-C29 negative control (test_main.cpp) demonstrates.

inline constexpr int64_t kRowBoundsInRangeRow[] = {INT64_C(100), INT64_C(-200), INT64_C(50), INT64_C(-1)};
inline constexpr int64_t kRowBoundsInRangeRowPermA[] = {INT64_C(-1), INT64_C(100), INT64_C(-200), INT64_C(50)};
inline constexpr int64_t kRowBoundsInRangeRowPermB[] = {INT64_C(50), INT64_C(-1), INT64_C(100), INT64_C(-200)};

struct RowBoundsWideCase {
	const char* label;
	const int64_t* data;
	size_t n;
	int64_t expected_max;
	int64_t expected_min;
};

inline constexpr RowBoundsWideCase kRowBoundsWideCases[] = {
	// T-1254's own positive-extreme row: max is exactly 2^31, min is the row's
	// most-negative element (-5).
	{"t1254_witness_row_positive_extreme", kWideT1254WitnessPositiveRow, kWideT1254WitnessRowLen,
	 INT64_C(2147483648), INT64_C(-5)},
	// The same row with the extreme negated to -2^31: min is exactly -2^31 --
	// the T-1254 assertion this cell exists to realize (Sec4.2) -- and max is
	// the row's largest remaining element (2^30).
	{"t1254_witness_row_negated_extreme", kWideT1254WitnessNegatedRow, kWideT1254WitnessRowLen,
	 INT64_C(1073741824), INT64_C(-2147483648)},
	// Entirely inside [INT32_MIN, INT32_MAX] with room to spare -- both bounds
	// hold without approaching either edge.
	{"in_range_with_room_to_spare", kRowBoundsInRangeRow, 4, INT64_C(100), INT64_C(-200)},
	// Order-independence: the same multiset as the in-range row, permuted
	// twice -- both permutations must report the identical (max, min) pair.
	{"in_range_perm_a", kRowBoundsInRangeRowPermA, 4, INT64_C(100), INT64_C(-200)},
	{"in_range_perm_b", kRowBoundsInRangeRowPermB, 4, INT64_C(100), INT64_C(-200)},
};
inline constexpr size_t kRowBoundsWideCasesCount = 5;

// --- C22 at int64 input width: RequantTokenCodeWide -------------------------
//
// Reuses kRequantCases (sslm_intmath_fixtures.h) verbatim: C22's formula is
// unchanged at the wider input width (Sec4.7's correction) -- only x_i's C++
// type widens from int32_t to int64_t, and every expected code is identical.

struct RequantWideCase {
	const char* label;
	int64_t x_i;
	int64_t r;
	int s;
	int8_t expected;
};

inline constexpr RequantWideCase kRequantWideCases[] = {
	{"typical_pos_small", INT64_C(1000), INT64_C(3000000000), 10, INT8_C(0)},
	{"typical_neg_small", INT64_C(-1000), INT64_C(3000000000), 10, INT8_C(0)},
	{"typical_pos_mid", INT64_C(5000000), INT64_C(4294967296), 0, INT8_C(1)},
	{"typical_neg_mid", INT64_C(-5000000), INT64_C(4294967296), 0, INT8_C(-1)},
	{"zero_x_i", INT64_C(0), INT64_C(4000000000), 20, INT8_C(0)},
	{"clamp_saturate_pos", INT64_C(2147483647), INT64_C(4294967296), 0, INT8_C(127)},
	{"clamp_saturate_neg", INT64_C(-2147483648), INT64_C(4294967296), 0, INT8_C(-127)},
	{"clamp_saturate_pos_large_r", INT64_C(2000000000), INT64_C(4294967000), 5, INT8_C(127)},
	{"clamp_saturate_neg_large_r", INT64_C(-2000000000), INT64_C(4294967000), 5, INT8_C(-127)},
	{"width_boundary_int32_min_r_2pow32", INT64_C(-2147483648), INT64_C(4294967296), 0, INT8_C(-127)},
	{"width_boundary_int32_min_r_2pow32_s30", INT64_C(-2147483648), INT64_C(4294967296), 30, INT8_C(-127)},
	{"neg_tie_away_from_zero_exp32", INT64_C(-1), INT64_C(2147483648), 30, INT8_C(-64)},
	{"pos_tie_exp32", INT64_C(1), INT64_C(2147483648), 30, INT8_C(64)},
	{"neg_tie_away_from_zero_exp40", INT64_C(-1), INT64_C(549755813888), 22, INT8_C(-64)},
	{"pos_tie_exp40", INT64_C(1), INT64_C(549755813888), 22, INT8_C(64)},
	{"neg_tie_away_from_zero_exp50", INT64_C(-1), INT64_C(562949953421312), 12, INT8_C(-64)},
	{"pos_tie_exp50", INT64_C(1), INT64_C(562949953421312), 12, INT8_C(64)},
	{"neg_tie_away_from_zero_exp63", INT64_C(-1), INT64_C(4611686018427387904), -1, INT8_C(-64)},
	{"pos_tie_exp63", INT64_C(1), INT64_C(4611686018427387904), -1, INT8_C(64)},
	// Wide-only: an x_i value that does not fit int32_t at all (the whole
	// reason RequantTokenCodeWide exists at F-S3-7 -- the row is NEVER
	// narrowed to int32 before this call, T-1254's fold). Magnitude exceeds
	// INT32_MAX, must still clamp-saturate identically to the in-int32-range
	// large-magnitude cases above.
	{"wide_only_exceeds_int32_range_pos", INT64_C(3000000000), INT64_C(4294967296), 0, INT8_C(127)},
	{"wide_only_exceeds_int32_range_neg", INT64_C(-3000000000), INT64_C(4294967296), 0, INT8_C(-127)},
};
inline constexpr size_t kRequantWideCasesCount = 21;

// --- Funnel-level witness rows (RequantChainChecked / NarrowRowChecked) -----
//
// The T-1254 required-green witness, reproduced verbatim (Sec11 S3.1): the
// same two rows above, through the funnel's two entry points instead of the
// bare primitives.

struct FunnelWitnessRow {
	const char* label;
	const int64_t* row;
	size_t n;
	int8_t expected_codes[4];  // RequantChainChecked's expected out_codes on accept
	int32_t expected_narrowed;  // NarrowRowChecked's expected out_i32[0] on accept
};

inline constexpr FunnelWitnessRow kFunnelWitnessRows[] = {
	{"positive_extreme", kWideT1254WitnessPositiveRow, kWideT1254WitnessRowLen,
	 {INT8_C(127), INT8_C(64), INT8_C(0), INT8_C(0)}, /*unused (this row must be REJECTED by NarrowRowChecked)*/ 0},
	{"negated_extreme", kWideT1254WitnessNegatedRow, kWideT1254WitnessRowLen,
	 {INT8_C(-127), INT8_C(64), INT8_C(0), INT8_C(0)}, /*expected_narrowed=*/ INT32_C(-2147483648)},
};
inline constexpr size_t kFunnelWitnessRowsCount = 2;

}  // namespace superslm_test

#endif  // SUPERSLM_TESTS_SSLM_S3_1_WIDE_INTMATH_FIXTURES_H

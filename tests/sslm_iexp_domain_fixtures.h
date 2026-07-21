// GENERATED FILE. Do not hand-edit.
//
// Produced by tests/gen_iexp_domain_fixtures.py: every expected value is computed
// there from IExpFromConstants's documented five-line decomposition using Python
// arbitrary-precision integers -- never by calling IExpShift/IExpBase/
// IExpConstantsInDomain (which do not exist yet) and never by re-deriving the bound
// in fixed-width int64 arithmetic (that re-derivation reproduces the D-SLM81 defect:
// base**2 alone overflows int64). Re-running the generator must reproduce this file
// byte-for-byte.
//
// Test-design record:
// Claude/Curie/superslm-s2.6-softmax-iexp-domain-test-design-2026-07-21.md
// Strike casebook: Claude/Loki/softmax-s2.6-strike-2026-07-21.md
#ifndef SUPERSLM_TESTS_SSLM_IEXP_DOMAIN_FIXTURES_H
#define SUPERSLM_TESTS_SSLM_IEXP_DOMAIN_FIXTURES_H

#include <cstddef>
#include <cstdint>

namespace superslm_test {

// --- IExpShift / IExpBase: independently-derived (z, base) pairs, keyed to the
//     SAME (label, q, q_ln2, q_b, q_c) fixtures already shipped in
//     sslm_intmath_fixtures.h's kIExpCases (this generator asserts its own
//     recomposition, (base**2+q_c)>>z, equals that file's already-independent
//     'expected' golden before ever emitting a row -- see the assert above). ---

struct IExpAccessorCase {
	const char* label;
	int64_t q;
	int64_t q_ln2;
	int64_t q_b;
	int64_t expected_z;
	int64_t expected_base;
	int64_t expected_iexp_from_constants;  // cross-reference only: kIExpCases' own golden
};

inline constexpr IExpAccessorCase kIExpAccessorCases[] = {
	{"realistic_s0_q_max_element", INT64_C(0), INT64_C(6), INT64_C(13), INT64_C(0), INT64_C(13), INT64_C(264)},
	{"realistic_s0_q_neg1", INT64_C(-1), INT64_C(6), INT64_C(13), INT64_C(0), INT64_C(12), INT64_C(239)},
	{"realistic_s0_q_neg_one_ln2_step", INT64_C(-6), INT64_C(6), INT64_C(13), INT64_C(1), INT64_C(13), INT64_C(132)},
	{"realistic_s0_q_neg_five_ln2_steps", INT64_C(-30), INT64_C(6), INT64_C(13), INT64_C(5), INT64_C(13), INT64_C(8)},
	{"realistic_s0_clip_boundary", INT64_C(-180), INT64_C(6), INT64_C(13), INT64_C(30), INT64_C(13), INT64_C(0)},
	{"realistic_s0_beyond_clip", INT64_C(-186), INT64_C(6), INT64_C(13), INT64_C(30), INT64_C(13), INT64_C(0)},
	{"realistic_s1_q_max_element", INT64_C(0), INT64_C(13), INT64_C(27), INT64_C(0), INT64_C(27), INT64_C(1112)},
	{"realistic_s1_q_neg1", INT64_C(-1), INT64_C(13), INT64_C(27), INT64_C(0), INT64_C(26), INT64_C(1059)},
	{"realistic_s1_q_neg_one_ln2_step", INT64_C(-13), INT64_C(13), INT64_C(27), INT64_C(1), INT64_C(27), INT64_C(556)},
	{"realistic_s1_q_neg_five_ln2_steps", INT64_C(-65), INT64_C(13), INT64_C(27), INT64_C(5), INT64_C(27), INT64_C(34)},
	{"realistic_s1_clip_boundary", INT64_C(-390), INT64_C(13), INT64_C(27), INT64_C(30), INT64_C(27), INT64_C(0)},
	{"realistic_s1_beyond_clip", INT64_C(-403), INT64_C(13), INT64_C(27), INT64_C(30), INT64_C(27), INT64_C(0)},
	{"realistic_s2_q_max_element", INT64_C(0), INT64_C(34), INT64_C(67), INT64_C(0), INT64_C(67), INT64_C(6887)},
	{"realistic_s2_q_neg1", INT64_C(-1), INT64_C(34), INT64_C(67), INT64_C(0), INT64_C(66), INT64_C(6754)},
	{"realistic_s2_q_neg_one_ln2_step", INT64_C(-34), INT64_C(34), INT64_C(67), INT64_C(1), INT64_C(67), INT64_C(3443)},
	{"realistic_s2_q_neg_five_ln2_steps", INT64_C(-170), INT64_C(34), INT64_C(67), INT64_C(5), INT64_C(67), INT64_C(215)},
	{"realistic_s2_clip_boundary", INT64_C(-1020), INT64_C(34), INT64_C(67), INT64_C(30), INT64_C(67), INT64_C(0)},
	{"realistic_s2_beyond_clip", INT64_C(-1054), INT64_C(34), INT64_C(67), INT64_C(30), INT64_C(67), INT64_C(0)},
	{"realistic_s3_q_max_element", INT64_C(0), INT64_C(138), INT64_C(270), INT64_C(0), INT64_C(270), INT64_C(111282)},
	{"realistic_s3_q_neg1", INT64_C(-1), INT64_C(138), INT64_C(270), INT64_C(0), INT64_C(269), INT64_C(110743)},
	{"realistic_s3_q_neg_one_ln2_step", INT64_C(-138), INT64_C(138), INT64_C(270), INT64_C(1), INT64_C(270), INT64_C(55641)},
	{"realistic_s3_q_neg_five_ln2_steps", INT64_C(-690), INT64_C(138), INT64_C(270), INT64_C(5), INT64_C(270), INT64_C(3477)},
	{"realistic_s3_clip_boundary", INT64_C(-4140), INT64_C(138), INT64_C(270), INT64_C(30), INT64_C(270), INT64_C(0)},
	{"realistic_s3_beyond_clip", INT64_C(-4278), INT64_C(138), INT64_C(270), INT64_C(30), INT64_C(270), INT64_C(0)},
	{"realistic_s4_q_max_element", INT64_C(0), INT64_C(88), INT64_C(171), INT64_C(0), INT64_C(171), INT64_C(44717)},
	{"realistic_s4_q_neg1", INT64_C(-1), INT64_C(88), INT64_C(171), INT64_C(0), INT64_C(170), INT64_C(44376)},
	{"realistic_s4_q_neg_one_ln2_step", INT64_C(-88), INT64_C(88), INT64_C(171), INT64_C(1), INT64_C(171), INT64_C(22358)},
	{"realistic_s4_q_neg_five_ln2_steps", INT64_C(-440), INT64_C(88), INT64_C(171), INT64_C(5), INT64_C(171), INT64_C(1397)},
	{"realistic_s4_clip_boundary", INT64_C(-2640), INT64_C(88), INT64_C(171), INT64_C(30), INT64_C(171), INT64_C(0)},
	{"realistic_s4_beyond_clip", INT64_C(-2728), INT64_C(88), INT64_C(171), INT64_C(30), INT64_C(171), INT64_C(0)},
	{"qln2_min_q0", INT64_C(0), INT64_C(1), INT64_C(2), INT64_C(0), INT64_C(2), INT64_C(7)},
	{"qln2_min_q_neg1", INT64_C(-1), INT64_C(1), INT64_C(2), INT64_C(1), INT64_C(2), INT64_C(3)},
	{"qln2_min_clip_boundary", INT64_C(-30), INT64_C(1), INT64_C(2), INT64_C(30), INT64_C(2), INT64_C(0)},
	{"qln2_min_beyond_clip", INT64_C(-35), INT64_C(1), INT64_C(2), INT64_C(30), INT64_C(2), INT64_C(0)},
	{"width_probe_q0_max_element", INT64_C(0), INT64_C(887904998), INT64_C(1733160715), INT64_C(0), INT64_C(1733160715), INT64_C(4578377597039742585)},
	{"width_probe_q_neg_one_ln2_step", INT64_C(-887904998), INT64_C(887904998), INT64_C(1733160715), INT64_C(1), INT64_C(1733160715), INT64_C(2289188798519871292)},
};
inline constexpr size_t kIExpAccessorCasesCount = 36;

// --- IExpConstantsInDomain. Every expected_in_domain is computed in Python
//     arbitrary precision by gen_iexp_domain_fixtures.py, transcribing the
//     documented formula directly -- never by calling the primitive under test. ---

struct IExpDomainCase {
	const char* label;
	int64_t q;
	int64_t q_ln2;
	int64_t q_b;
	int64_t q_c;
	bool expected_in_domain;
};

inline constexpr IExpDomainCase kIExpDomainCases[] = {
	{"strike_exact_input_out_of_domain", INT64_C(0), INT64_C(887904998), INT64_C(1733160715), INT64_C(9223372036854775807), false},
	{"z0_boundary_last_in_domain", INT64_C(0), INT64_C(887904998), INT64_C(1733160715), INT64_C(6219525972835464582), true},
	{"z0_boundary_first_out_of_domain", INT64_C(0), INT64_C(887904998), INT64_C(1733160715), INT64_C(6219525972835464583), false},
	{"z1_same_qc_as_strike_recovers_in_domain", INT64_C(-887904998), INT64_C(887904998), INT64_C(1733160715), INT64_C(9223372036854775807), true},
	{"z_at_clip_ceiling_strike_qc_in_domain", INT64_C(-26637149940), INT64_C(887904998), INT64_C(1733160715), INT64_C(9223372036854775807), true},
	{"z0_base_squared_alone_exceeds_domain_even_at_qc_one", INT64_C(0), INT64_C(887904998), INT64_C(3500000000), INT64_C(1), false},
	{"z1_boundary_last_in_domain_large_base", INT64_C(-887904998), INT64_C(887904998), INT64_C(3500000000), INT64_C(6196744073709551615), true},
	{"z1_boundary_first_out_of_domain_large_base", INT64_C(-887904998), INT64_C(887904998), INT64_C(3500000000), INT64_C(6196744073709551616), false},
	{"z_at_clip_ceiling_always_in_domain_large_base", INT64_C(-26637149940), INT64_C(887904998), INT64_C(3500000000), INT64_C(9223372036854775807), true},
	{"z0_lower_bound_equality_qc_min", INT64_C(0), INT64_C(1), INT64_C(0), INT64_MIN, true},
	{"z1_lower_bound_slack_qc_min", INT64_C(-7), INT64_C(7), INT64_C(0), INT64_MIN, true},
	{"z_at_clip_ceiling_lower_bound_slack_qc_min", INT64_C(-210), INT64_C(7), INT64_C(0), INT64_MIN, true},
	{"typical_small_in_domain", INT64_C(0), INT64_C(6), INT64_C(13), INT64_C(95), true},
	{"realistic_operating_scale_0p01_in_domain", INT64_C(0), INT64_C(69), INT64_C(135), INT64_C(9595), true},
	{"existing_fixture_realistic_s0_q_max_element", INT64_C(0), INT64_C(6), INT64_C(13), INT64_C(95), true},
	{"existing_fixture_realistic_s0_q_neg1", INT64_C(-1), INT64_C(6), INT64_C(13), INT64_C(95), true},
	{"existing_fixture_realistic_s0_q_neg_one_ln2_step", INT64_C(-6), INT64_C(6), INT64_C(13), INT64_C(95), true},
	{"existing_fixture_realistic_s0_q_neg_five_ln2_steps", INT64_C(-30), INT64_C(6), INT64_C(13), INT64_C(95), true},
	{"existing_fixture_realistic_s0_clip_boundary", INT64_C(-180), INT64_C(6), INT64_C(13), INT64_C(95), true},
	{"existing_fixture_realistic_s0_beyond_clip", INT64_C(-186), INT64_C(6), INT64_C(13), INT64_C(95), true},
	{"existing_fixture_realistic_s1_q_max_element", INT64_C(0), INT64_C(13), INT64_C(27), INT64_C(383), true},
	{"existing_fixture_realistic_s1_q_neg1", INT64_C(-1), INT64_C(13), INT64_C(27), INT64_C(383), true},
	{"existing_fixture_realistic_s1_q_neg_one_ln2_step", INT64_C(-13), INT64_C(13), INT64_C(27), INT64_C(383), true},
	{"existing_fixture_realistic_s1_q_neg_five_ln2_steps", INT64_C(-65), INT64_C(13), INT64_C(27), INT64_C(383), true},
	{"existing_fixture_realistic_s1_clip_boundary", INT64_C(-390), INT64_C(13), INT64_C(27), INT64_C(383), true},
	{"existing_fixture_realistic_s1_beyond_clip", INT64_C(-403), INT64_C(13), INT64_C(27), INT64_C(383), true},
	{"existing_fixture_realistic_s2_q_max_element", INT64_C(0), INT64_C(34), INT64_C(67), INT64_C(2398), true},
	{"existing_fixture_realistic_s2_q_neg1", INT64_C(-1), INT64_C(34), INT64_C(67), INT64_C(2398), true},
	{"existing_fixture_realistic_s2_q_neg_one_ln2_step", INT64_C(-34), INT64_C(34), INT64_C(67), INT64_C(2398), true},
	{"existing_fixture_realistic_s2_q_neg_five_ln2_steps", INT64_C(-170), INT64_C(34), INT64_C(67), INT64_C(2398), true},
	{"existing_fixture_realistic_s2_clip_boundary", INT64_C(-1020), INT64_C(34), INT64_C(67), INT64_C(2398), true},
	{"existing_fixture_realistic_s2_beyond_clip", INT64_C(-1054), INT64_C(34), INT64_C(67), INT64_C(2398), true},
	{"existing_fixture_realistic_s3_q_max_element", INT64_C(0), INT64_C(138), INT64_C(270), INT64_C(38382), true},
	{"existing_fixture_realistic_s3_q_neg1", INT64_C(-1), INT64_C(138), INT64_C(270), INT64_C(38382), true},
	{"existing_fixture_realistic_s3_q_neg_one_ln2_step", INT64_C(-138), INT64_C(138), INT64_C(270), INT64_C(38382), true},
	{"existing_fixture_realistic_s3_q_neg_five_ln2_steps", INT64_C(-690), INT64_C(138), INT64_C(270), INT64_C(38382), true},
	{"existing_fixture_realistic_s3_clip_boundary", INT64_C(-4140), INT64_C(138), INT64_C(270), INT64_C(38382), true},
	{"existing_fixture_realistic_s3_beyond_clip", INT64_C(-4278), INT64_C(138), INT64_C(270), INT64_C(38382), true},
	{"existing_fixture_realistic_s4_q_max_element", INT64_C(0), INT64_C(88), INT64_C(171), INT64_C(15476), true},
	{"existing_fixture_realistic_s4_q_neg1", INT64_C(-1), INT64_C(88), INT64_C(171), INT64_C(15476), true},
	{"existing_fixture_realistic_s4_q_neg_one_ln2_step", INT64_C(-88), INT64_C(88), INT64_C(171), INT64_C(15476), true},
	{"existing_fixture_realistic_s4_q_neg_five_ln2_steps", INT64_C(-440), INT64_C(88), INT64_C(171), INT64_C(15476), true},
	{"existing_fixture_realistic_s4_clip_boundary", INT64_C(-2640), INT64_C(88), INT64_C(171), INT64_C(15476), true},
	{"existing_fixture_realistic_s4_beyond_clip", INT64_C(-2728), INT64_C(88), INT64_C(171), INT64_C(15476), true},
	{"existing_fixture_qln2_min_q0", INT64_C(0), INT64_C(1), INT64_C(2), INT64_C(3), true},
	{"existing_fixture_qln2_min_q_neg1", INT64_C(-1), INT64_C(1), INT64_C(2), INT64_C(3), true},
	{"existing_fixture_qln2_min_clip_boundary", INT64_C(-30), INT64_C(1), INT64_C(2), INT64_C(3), true},
	{"existing_fixture_qln2_min_beyond_clip", INT64_C(-35), INT64_C(1), INT64_C(2), INT64_C(3), true},
	{"existing_fixture_width_probe_q0_max_element", INT64_C(0), INT64_C(887904998), INT64_C(1733160715), INT64_C(1574531533020431360), true},
	{"existing_fixture_width_probe_q_neg_one_ln2_step", INT64_C(-887904998), INT64_C(887904998), INT64_C(1733160715), INT64_C(1574531533020431360), true},
};
inline constexpr size_t kIExpDomainCasesCount = 50;

}  // namespace superslm_test

#endif  // SUPERSLM_TESTS_SSLM_IEXP_DOMAIN_FIXTURES_H

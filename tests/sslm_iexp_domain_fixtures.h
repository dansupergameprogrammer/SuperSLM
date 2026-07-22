// GENERATED FILE. Do not hand-edit.
//
// Produced by tests/gen_iexp_domain_fixtures.py: every expected value is computed
// there from the documented five-line i-exp decomposition using Python
// arbitrary-precision integers -- never by calling IExpConstruct/IExpEvaluate/
// IExpConstantsInDomain and never by re-deriving the bound
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

// --- IExpConstruct's (z, base): independently-derived pairs, keyed to the
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
	{"q_ln2_ceiling_last_in_domain", INT64_C(0), INT64_C(307445734561825860), INT64_C(1), INT64_C(0), true},
	{"q_ln2_ceiling_first_out_of_domain", INT64_C(0), INT64_C(307445734561825861), INT64_C(1), INT64_C(0), false},
	{"q_ln2_beyond_ceiling_2pow60_was_false_all_clear", INT64_C(0), INT64_C(1152921504606846976), INT64_C(1), INT64_C(0), false},
	{"q_ln2_beyond_ceiling_int64_max_was_false_all_clear", INT64_C(0), INT64_C(9223372036854775807), INT64_C(1), INT64_C(0), false},
	{"z0_base_squared_alone_exceeds_domain_even_at_qc_one", INT64_C(0), INT64_C(887904998), INT64_C(3500000000), INT64_C(1), false},
	{"z1_boundary_last_in_domain_large_base", INT64_C(-887904998), INT64_C(887904998), INT64_C(3500000000), INT64_C(6196744073709551615), true},
	{"z1_boundary_first_out_of_domain_large_base", INT64_C(-887904998), INT64_C(887904998), INT64_C(3500000000), INT64_C(6196744073709551616), false},
	{"z_at_clip_ceiling_always_in_domain_large_base", INT64_C(-26637149940), INT64_C(887904998), INT64_C(3500000000), INT64_C(9223372036854775807), true},
	{"z0_lower_bound_equality_qc_min", INT64_C(0), INT64_C(1), INT64_C(0), INT64_MIN, true},
	{"z1_lower_bound_slack_qc_min", INT64_C(-7), INT64_C(7), INT64_C(0), INT64_MIN, true},
	{"z_at_clip_ceiling_lower_bound_slack_qc_min", INT64_C(-210), INT64_C(7), INT64_C(0), INT64_MIN, true},
	{"typical_small_in_domain", INT64_C(0), INT64_C(6), INT64_C(13), INT64_C(95), true},
	{"realistic_operating_scale_0p01_in_domain", INT64_C(0), INT64_C(69), INT64_C(135), INT64_C(9595), true},
	{"shortcut_holds_interior_A_q0", INT64_C(0), INT64_C(3000000000), INT64_C(1800000000), INT64_C(7783372039254775806), false},
	{"shortcut_holds_interior_A_far_end", INT64_C(-2999999999), INT64_C(3000000000), INT64_C(1800000000), INT64_C(7783372039254775806), true},
	{"shortcut_fails_interior_B_q0", INT64_C(0), INT64_C(3000000000), INT64_C(1000000000), INT64_C(8223372036854775807), true},
	{"shortcut_fails_interior_B_far_end", INT64_C(-2999999999), INT64_C(3000000000), INT64_C(1000000000), INT64_C(8223372036854775807), false},
	{"shortcut_boundary_holds_C_q0", INT64_C(0), INT64_C(3000000000), INT64_C(1500000000), INT64_C(6973372036854775807), true},
	{"shortcut_boundary_holds_C_far_end", INT64_C(-2999999999), INT64_C(3000000000), INT64_C(1500000000), INT64_C(6973372036854775807), true},
	{"shortcut_boundary_fails_D_q0", INT64_C(0), INT64_C(3000000000), INT64_C(1499999999), INT64_C(6973372039854775806), true},
	{"shortcut_boundary_fails_D_far_end", INT64_C(-2999999999), INT64_C(3000000000), INT64_C(1499999999), INT64_C(6973372039854775806), false},
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
inline constexpr size_t kIExpDomainCasesCount = 62;

// --- S-HARDEN-0 LAYER B: kIExpConstructCases, for the checked `IExpConstruct` /
//     `IExpEvaluate` entry point (final API, S-HARDEN-0). expected_domain is one
//     of "kOk", "kNotRepresentable", "kBadQ", "kBadQLn2", "kBadQB" -- a
//     string, not superslm::IExpDomain directly, so this header stays free of a
//     compile-time dependency on that enum's exact values; the consuming test maps
//     IExpDomain to its name and string-compares.
//
//     expected_z/expected_base/expected_value are valid (IExpConstruction is
//     filled, and IExpEvaluate on it is asserted) whenever expected_domain is
//     "kOk" or "kNotRepresentable" -- the decomposition is well-formed for both,
//     and IExpEvaluate is TOTAL over both per the final API. expected_value is the
//     exact wrapped int64 IExpEvaluate must return -- see gen_iexp_domain_fixtures
//     .py's wrap_to_int64() docstring for how a kNotRepresentable row's value is
//     derived. All three fields are 0 placeholders, never asserted, for the three
//     kBad* rows, where *out is contractually left untouched (checked against a
//     priming construction, not against these placeholders).
//
//     Includes the ported S-HARDEN-0 population cells (retired kIExpGuardOrderCases
//     -- see this generator's own comment at the retirement site) and Poirot's
//     a1d7986 code-review Finding 1 strips (stripa_*/stripb_*). Test-design record:
//     Claude/Curie/superslm-s-harden-0-test-design-2026-07-21.md ---

struct IExpConstructCase {
	const char* label;
	int64_t q;
	int64_t q_ln2;
	int64_t q_b;
	int64_t q_c;
	const char* expected_domain;
	int64_t expected_z;
	int64_t expected_base;
	int64_t expected_value;  // IExpEvaluate(...); valid for kOk/kNotRepresentable only
};

inline constexpr IExpConstructCase kIExpConstructCases[] = {
	{"badq_last_valid_q_zero", INT64_C(0), INT64_C(6), INT64_C(13), INT64_C(95), "kOk", INT64_C(0), INT64_C(13), INT64_C(264)},
	{"badq_first_invalid_q_one", INT64_C(1), INT64_C(6), INT64_C(13), INT64_C(95), "kBadQ", INT64_C(0), INT64_C(0), INT64_C(0)},
	{"badq_interior_invalid_large_positive_q", INT64_C(1000000), INT64_C(6), INT64_C(13), INT64_C(95), "kBadQ", INT64_C(0), INT64_C(0), INT64_C(0)},
	{"badqln2_last_valid_qln2_one", INT64_C(0), INT64_C(1), INT64_C(5), INT64_C(10), "kOk", INT64_C(0), INT64_C(5), INT64_C(35)},
	{"badqln2_first_invalid_qln2_zero", INT64_C(0), INT64_C(0), INT64_C(5), INT64_C(10), "kBadQLn2", INT64_C(0), INT64_C(0), INT64_C(0)},
	{"badqln2_interior_invalid_qln2_negative", INT64_C(0), INT64_C(-1000), INT64_C(5), INT64_C(10), "kBadQLn2", INT64_C(0), INT64_C(0), INT64_C(0)},
	{"badqln2_last_valid_ceiling", INT64_C(0), INT64_C(307445734561825860), INT64_C(1), INT64_C(0), "kOk", INT64_C(0), INT64_C(1), INT64_C(1)},
	{"badqln2_first_invalid_ceiling", INT64_C(0), INT64_C(307445734561825861), INT64_C(1), INT64_C(0), "kBadQLn2", INT64_C(0), INT64_C(0), INT64_C(0)},
	{"badqln2_interior_invalid_ceiling", INT64_C(0), INT64_C(614891469123651720), INT64_C(1), INT64_C(0), "kBadQLn2", INT64_C(0), INT64_C(0), INT64_C(0)},
	{"badqln2_f9_witness_int64_max", INT64_C(0), INT64_C(9223372036854775807), INT64_C(1), INT64_C(0), "kBadQLn2", INT64_C(0), INT64_C(0), INT64_C(0)},
	{"badqb_f21_witness_qb_int64_min", INT64_C(-1), INT64_C(1000), INT64_MIN, INT64_C(0), "kBadQB", INT64_C(0), INT64_C(0), INT64_C(0)},
	{"badqb_boundary_first_unsafe", INT64_C(-999), INT64_C(1000), INT64_C(-9223372036854774810), INT64_C(0), "kBadQB", INT64_C(0), INT64_C(0), INT64_C(0)},
	{"badqb_interior_unsafe", INT64_C(-999), INT64_C(1000), INT64_C(-9223372036854775308), INT64_C(0), "kBadQB", INT64_C(0), INT64_C(0), INT64_C(0)},
	{"badqb_representable_boundary_square_not_representable", INT64_C(-999), INT64_C(1000), INT64_C(-9223372036854774809), INT64_C(0), "kNotRepresentable", INT64_C(0), INT64_MIN, INT64_C(0)},
	{"notrepresentable_strike_witness", INT64_C(0), INT64_C(887904998), INT64_C(1733160715), INT64_C(9223372036854775807), "kNotRepresentable", INT64_C(0), INT64_C(1733160715), INT64_C(-6219525972835464584)},
	{"notrepresentable_boundary_last_ok", INT64_C(0), INT64_C(887904998), INT64_C(1733160715), INT64_C(6219525972835464582), "kOk", INT64_C(0), INT64_C(1733160715), INT64_C(9223372036854775807)},
	{"notrepresentable_boundary_first_not_representable", INT64_C(0), INT64_C(887904998), INT64_C(1733160715), INT64_C(6219525972835464583), "kNotRepresentable", INT64_C(0), INT64_C(1733160715), INT64_MIN},
	{"stripa_review_smallest_witness", INT64_C(1), INT64_C(1000), INT64_C(0), INT64_C(0), "kBadQ", INT64_C(0), INT64_C(0), INT64_C(0)},
	{"stripa_boundary_q_just_below_qln2", INT64_C(999), INT64_C(1000), INT64_C(0), INT64_C(0), "kBadQ", INT64_C(0), INT64_C(0), INT64_C(0)},
	{"stripa_interior", INT64_C(500), INT64_C(1000), INT64_C(0), INT64_C(0), "kBadQ", INT64_C(0), INT64_C(0), INT64_C(0)},
	{"stripb_boundary_qln2_neg_one", INT64_C(0), INT64_C(-1), INT64_C(0), INT64_C(0), "kBadQLn2", INT64_C(0), INT64_C(0), INT64_C(0)},
	{"stripb_boundary_qln2_most_negative", INT64_C(0), INT64_C(-307445734561825860), INT64_C(0), INT64_C(0), "kBadQLn2", INT64_C(0), INT64_C(0), INT64_C(0)},
	{"stripb_interior", INT64_C(0), INT64_C(-153722867280912930), INT64_C(0), INT64_C(0), "kBadQLn2", INT64_C(0), INT64_C(0), INT64_C(0)},
};
inline constexpr size_t kIExpConstructCasesCount = 23;

}  // namespace superslm_test

#endif  // SUPERSLM_TESTS_SSLM_IEXP_DOMAIN_FIXTURES_H

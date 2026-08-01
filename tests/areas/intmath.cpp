// tests/areas/intmath.cpp -- T-1574 test suite split, Stage 6.
// Area #5 (Claude/Plans/SuperSLM_TestSuiteSplit_Plan.md §4): tests calling
// through src/intmath.cpp's public contract (the §6.2 requant primitives,
// the §6.2 dynamic-scale chain, the §6.3 nonlinear scalars (ISqrt/IExp),
// §6.4 RoPE rotation, and the §6.2/§5.2 C32 softmax row kernel).
// Extracted verbatim from tests/test_main.cpp -- order keys 175-182, 184-188,
// 196, 206-225, 337-338, 355-356, 364, 382 -- plus the
// row_bounds_wide_zero_len_null_ptr crash-probe registration this area owns.
//
// **Order key 183 stays in tests/test_main.cpp, not here.**
// TestCarriedScaleReciprocalIsAForwardOfDynamicScaleReciprocalOverC19Domain
// is a join test: it calls both CarriedScaleReciprocal
// (checked_chain_funnel.h, candidate area #6) and DynamicScaleReciprocal
// (intmath.h, this area) to prove the former forwards to the latter. Per
// §3.1's join rule, a join files under the COMPOSING function's area --
// CarriedScaleReciprocal is the function under test (the wrapper being
// proven), DynamicScaleReciprocal is its own already-certified oracle -- so
// this test belongs to candidate area #6 checked_chain_funnel.cpp, extracted
// there.
//
// **The plan's own Stage 6 table row and the design record's grounding pass
// both misattribute the narrow_row_checked_zero_len_null_ptr crash probe to
// this area** ("including its two owned row_bounds_wide_zero_len_null_ptr/
// narrow_row_checked_zero_len_null_ptr crash-probe registrations"). Reading
// the actual probe body: it calls superslm::NarrowRowChecked
// (checked_chain_funnel.h), never anything declared in intmath.h. Per
// §3.1's rule this probe belongs to checked_chain_funnel.cpp (candidate
// area #6), not here -- extracted there (Stage 9), where its own §5 exit
// check gains the /O2+/Od dual-configuration verification the plan's table
// text describes for this stage, carried to the area that actually owns the
// probe rather than dropped. This area (Stage 6) is verified dual-config for
// its own single owned probe, row_bounds_wide_zero_len_null_ptr, below.

#include "superslm/intmath.h"
#include "sslm_fixtures.h"
#include "sslm_iexp_domain_fixtures.h"
#include "sslm_intmath_fixtures.h"
#include "sslm_kvc1_hostile_fixtures.h"
#include "sslm_s3_1_c30_iexp_domain_sweep_fixtures.h"
#include "sslm_s3_1_wide_intmath_fixtures.h"
#include "sslm_s3_3_fixtures.h"

#include "support/crash_probe.h"
#include "support/test_harness.h"
#include "support/test_registry.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

using namespace superslm;
using namespace superslm_test;
using namespace superslm_test_registry;

// ---------------------------------------------------------------------------
// Curie's S2.1 intmath red suite (SuperSLM_Plan.md S2.1; §6.8 C1/C2/C3 +
// C19-C22; Claude/Curie/superslm-s2.1-intmath-test-design-2026-07-19.md).
// Every golden value in sslm_intmath_fixtures.h is computed by CALLING
// Tools/superslm_spike/intmath.py (D-SLM52, the pinned reference) via
// tests/gen_intmath_fixtures.py — never hand-computed. src/intmath.cpp is
// fully ported and bit-exact against the pinned reference (shipped at
// S-HARDEN-1); every cell below is green against it.
// ---------------------------------------------------------------------------

SSLM_TEST(TestC2SaturatingRoundingDoublingHighMul, 175) {
	using namespace superslm_test;
	for (size_t i = 0; i < kC2CasesCount; ++i) {
		const C2Case& c = kC2Cases[i];
		int32_t got = superslm::SaturatingRoundingDoublingHighMul(c.a, c.b);
		CHECK_MSG(got == c.expected, "%s: SaturatingRoundingDoublingHighMul(%d, %d) == %d, want %d",
		          c.label, c.a, c.b, got, c.expected);
	}
}

SSLM_TEST(TestC1C3RoundingDivideByPOT, 176) {
	using namespace superslm_test;
	for (size_t i = 0; i < kC1C3CasesCount; ++i) {
		const C1C3Case& c = kC1C3Cases[i];
		int32_t got = superslm::RoundingDivideByPOT(c.x, c.exponent);
		CHECK_MSG(got == c.expected, "%s: RoundingDivideByPOT(%d, %d) == %d, want %d",
		          c.label, c.x, c.exponent, got, c.expected);
	}
}

SSLM_TEST(TestMultiplyByQuantizedMultiplier, 177) {
	using namespace superslm_test;
	for (size_t i = 0; i < kMbqmCasesCount; ++i) {
		const MbqmCase& c = kMbqmCases[i];
		int32_t got = superslm::MultiplyByQuantizedMultiplier(c.x, c.multiplier, c.shift);
		CHECK_MSG(got == c.expected, "%s: MultiplyByQuantizedMultiplier(%d, %d, %d) == %d, want %d",
		          c.label, c.x, c.multiplier, c.shift, got, c.expected);
	}
}

SSLM_TEST(TestClz64, 178) {
	using namespace superslm_test;
	for (size_t i = 0; i < kClz64CasesCount; ++i) {
		const Clz64Case& c = kClz64Cases[i];
		int got = superslm::Clz64(c.n);
		CHECK_MSG(got == c.expected, "%s: Clz64(%llu) == %d, want %d", c.label,
		          static_cast<unsigned long long>(c.n), got, c.expected);
	}
}

SSLM_TEST(TestMaxAbsReduce, 179) {
	using namespace superslm_test;
	for (size_t i = 0; i < kMaxAbsCasesCount; ++i) {
		const MaxAbsCase& c = kMaxAbsCases[i];
		int64_t got = superslm::MaxAbsReduce(c.data, c.n);
		CHECK_MSG(got == c.expected, "%s: MaxAbsReduce(n=%zu) == %lld, want %lld", c.label, c.n,
		          static_cast<long long>(got), static_cast<long long>(c.expected));
	}

	// Order-independence, checked directly rather than only via matching
	// expected values: the three "order_perm_*" fixture cases are the same
	// multiset under three different orderings, so their live results must
	// agree with each other, not merely each with its own precomputed golden.
	int64_t perm_a = superslm::MaxAbsReduce(kMaxAbsData8, 5);
	int64_t perm_b = superslm::MaxAbsReduce(kMaxAbsData9, 5);
	int64_t perm_c = superslm::MaxAbsReduce(kMaxAbsData10, 5);
	CHECK_MSG(perm_a == perm_b && perm_b == perm_c,
	          "MaxAbsReduce is not order-independent: perm_a=%lld perm_b=%lld perm_c=%lld",
	          static_cast<long long>(perm_a), static_cast<long long>(perm_b),
	          static_cast<long long>(perm_c));
}

SSLM_TEST(TestNormalizeScale, 180) {
	using namespace superslm_test;
	for (size_t i = 0; i < kNormalizeScaleCasesCount; ++i) {
		const NormalizeScaleCase& c = kNormalizeScaleCases[i];
		superslm::NormalizedScale got = superslm::NormalizeScale(c.d_prime);
		CHECK_MSG(got.dn == c.expected_dn, "%s: NormalizeScale(%lld).dn == %lld, want %lld", c.label,
		          static_cast<long long>(c.d_prime), static_cast<long long>(got.dn),
		          static_cast<long long>(c.expected_dn));
		CHECK_MSG(got.s == c.expected_s, "%s: NormalizeScale(%lld).s == %d, want %d", c.label,
		          static_cast<long long>(c.d_prime), got.s, c.expected_s);
		// Postcondition (C21): 2^30 <= Dn < 2^31 must hold on every case,
		// independent of whether it matches the golden — a bug that produces
		// a wrong-but-still-in-range Dn should not also silently violate the
		// documented contract.
		CHECK_MSG(got.dn >= (INT64_C(1) << 30) && got.dn < (INT64_C(1) << 31),
		          "%s: NormalizeScale(%lld).dn == %lld violates the C21 postcondition [2^30, 2^31)",
		          c.label, static_cast<long long>(c.d_prime), static_cast<long long>(got.dn));
	}
}

SSLM_TEST(TestDynamicScaleReciprocalNamed, 181) {
	using namespace superslm_test;
	for (size_t i = 0; i < kDynRecipNamedCasesCount; ++i) {
		const DynRecipNamedCase& c = kDynRecipNamedCases[i];
		int64_t got = superslm::DynamicScaleReciprocal(c.dn);
		CHECK_MSG(got == c.expected_r, "%s: DynamicScaleReciprocal(%lld) == %lld, want %lld", c.label,
		          static_cast<long long>(c.dn), static_cast<long long>(got),
		          static_cast<long long>(c.expected_r));
	}
}

SSLM_TEST(TestDynamicScaleReciprocalDenseSample, 182) {
	using namespace superslm_test;
	size_t mismatches = 0;
	for (size_t i = 0; i < kDynRecipDenseCasesCount; ++i) {
		const DynRecipDenseCase& c = kDynRecipDenseCases[i];
		int64_t got = superslm::DynamicScaleReciprocal(c.dn);
		++GChecks;
		if (got != c.expected_r) {
			++GFailures;
			++mismatches;
			if (mismatches <= 20) {
				std::printf("FAIL %s:%d: DynamicScaleReciprocal(%lld) == %lld, want %lld (dense sample)\n",
				            __FILE__, __LINE__, static_cast<long long>(c.dn), static_cast<long long>(got),
				            static_cast<long long>(c.expected_r));
			}
		}
	}
	if (mismatches > 20) {
		std::printf("... %zu additional DynamicScaleReciprocal dense-sample mismatches suppressed\n",
		            mismatches - 20);
	}
}


SSLM_TEST(TestRequantTokenCode, 184) {
	using namespace superslm_test;
	for (size_t i = 0; i < kRequantCasesCount; ++i) {
		const RequantCase& c = kRequantCases[i];
		int8_t got = superslm::RequantTokenCode(c.x_i, c.r, c.s);
		CHECK_MSG(got == c.expected, "%s: RequantTokenCode(%d, %lld, %d) == %d, want %d", c.label,
		          c.x_i, static_cast<long long>(c.r), c.s, static_cast<int>(got),
		          static_cast<int>(c.expected));
		CHECK_MSG(got >= -127 && got <= 127, "%s: RequantTokenCode(%d, %lld, %d) == %d, out of [-127, 127]",
		          c.label, c.x_i, static_cast<long long>(c.r), c.s, static_cast<int>(got));
	}
}

// --- F-S3-7 / Sec11 S3.1: wide-row (int64 input width) siblings --------------
//
// MaxAbsReduceWide / RowBoundsWide / RequantTokenCodeWide have real, shipped
// bodies (src/intmath.cpp:270-281 and neighbouring lines), fixed against the
// INT64_MIN signed-overflow and n==0 read-before-check defects Poirot's
// ac34677 review found (S1, S3) -- the unsigned-magnitude form and the
// defined-empty-reduction early return are both in place. Every cell below is
// specified completely against
// Claude/Curie/superslm-s3.1-checked-chain-funnel-test-design-2026-07-28.md
// Sec4.1/Sec4.2/Sec4.3.

SSLM_TEST(TestMaxAbsReduceWide, 186) {
	using namespace superslm_test;
	for (size_t i = 0; i < kMaxAbsWideCasesCount; ++i) {
		const MaxAbsWideCase& c = kMaxAbsWideCases[i];
		int64_t got = superslm::MaxAbsReduceWide(c.data, c.n);
		CHECK_MSG(got == c.expected, "%s: MaxAbsReduceWide(n=%zu) == %lld, want %lld", c.label, c.n,
		          static_cast<long long>(got), static_cast<long long>(c.expected));
	}

	// Order-independence, checked directly rather than only via matching
	// expected values -- mirrors TestMaxAbsReduce's own discipline at the
	// narrow width, widened.
	int64_t perm_a = superslm::MaxAbsReduceWide(kMaxAbsWideData8, 5);
	int64_t perm_b = superslm::MaxAbsReduceWide(kMaxAbsWideData9, 5);
	int64_t perm_c = superslm::MaxAbsReduceWide(kMaxAbsWideData10, 5);
	CHECK_MSG(perm_a == perm_b && perm_b == perm_c,
	          "MaxAbsReduceWide is not order-independent: perm_a=%lld perm_b=%lld perm_c=%lld",
	          static_cast<long long>(perm_a), static_cast<long long>(perm_b),
	          static_cast<long long>(perm_c));
}

SSLM_TEST(TestRowBoundsWide, 187) {
	using namespace superslm_test;
	for (size_t i = 0; i < kRowBoundsWideCasesCount; ++i) {
		const RowBoundsWideCase& c = kRowBoundsWideCases[i];
		int64_t got_max = INT64_C(0xdeadbeef);
		int64_t got_min = INT64_C(0xdeadbeef);
		superslm::RowBoundsWide(c.data, c.n, &got_max, &got_min);
		CHECK_MSG(got_max == c.expected_max, "%s: RowBoundsWide(n=%zu).out_max == %lld, want %lld",
		          c.label, c.n, static_cast<long long>(got_max), static_cast<long long>(c.expected_max));
		CHECK_MSG(got_min == c.expected_min, "%s: RowBoundsWide(n=%zu).out_min == %lld, want %lld",
		          c.label, c.n, static_cast<long long>(got_min), static_cast<long long>(c.expected_min));
	}

	// Order-independence on the two permuted in-range rows: both permutations
	// of the same multiset must report the identical (max, min) pair.
	int64_t max_a = 0, min_a = 0, max_b = 0, min_b = 0;
	superslm::RowBoundsWide(kRowBoundsInRangeRowPermA, 4, &max_a, &min_a);
	superslm::RowBoundsWide(kRowBoundsInRangeRowPermB, 4, &max_b, &min_b);
	CHECK_MSG(max_a == max_b && min_a == min_b,
	          "RowBoundsWide is not order-independent: (max_a=%lld,min_a=%lld) vs (max_b=%lld,min_b=%lld)",
	          static_cast<long long>(max_a), static_cast<long long>(min_a), static_cast<long long>(max_b),
	          static_cast<long long>(min_b));
}

SSLM_TEST(TestRequantTokenCodeWide, 188) {
	using namespace superslm_test;
	for (size_t i = 0; i < kRequantWideCasesCount; ++i) {
		const RequantWideCase& c = kRequantWideCases[i];
		int8_t got = superslm::RequantTokenCodeWide(c.x_i, c.r, c.s);
		CHECK_MSG(got == c.expected, "%s: RequantTokenCodeWide(%lld, %lld, %d) == %d, want %d", c.label,
		          static_cast<long long>(c.x_i), static_cast<long long>(c.r), c.s, static_cast<int>(got),
		          static_cast<int>(c.expected));
		CHECK_MSG(got >= -127 && got <= 127,
		          "%s: RequantTokenCodeWide(%lld, %lld, %d) == %d, out of [-127, 127]", c.label,
		          static_cast<long long>(c.x_i), static_cast<long long>(c.r), c.s, static_cast<int>(got));
	}
}

SSLM_TEST(TestIntmathPipelineComposition, 185) {
	// The feature oracle (dim 10) for the C19-C22 chain: MaxAbsReduce ->
	// NormalizeScale -> DynamicScaleReciprocal -> RequantTokenCode composed
	// exactly as the runtime rung-1 per-token quantizer composes them,
	// asserted against intmath.py's own end-to-end composition rather than
	// against each primitive's isolated golden alone — proves the four
	// C++ ports agree with each other's outputs in sequence, not merely
	// each in isolation.
	using namespace superslm_test;
	for (size_t i = 0; i < kPipelineCasesCount; ++i) {
		const PipelineCase& c = kPipelineCases[i];
		int64_t d_prime = superslm::MaxAbsReduce(c.xs, c.n);
		CHECK_MSG(d_prime == c.expected_d_prime, "%s: pipeline MaxAbsReduce == %lld, want %lld", c.label,
		          static_cast<long long>(d_prime), static_cast<long long>(c.expected_d_prime));

		superslm::NormalizedScale ns = superslm::NormalizeScale(d_prime);
		CHECK_MSG(ns.dn == c.expected_dn, "%s: pipeline NormalizeScale(D').dn == %lld, want %lld", c.label,
		          static_cast<long long>(ns.dn), static_cast<long long>(c.expected_dn));
		CHECK_MSG(ns.s == c.expected_s, "%s: pipeline NormalizeScale(D').s == %d, want %d", c.label, ns.s,
		          c.expected_s);

		int64_t r = superslm::DynamicScaleReciprocal(ns.dn);
		CHECK_MSG(r == c.expected_r, "%s: pipeline DynamicScaleReciprocal(Dn) == %lld, want %lld", c.label,
		          static_cast<long long>(r), static_cast<long long>(c.expected_r));

		for (size_t j = 0; j < c.n; ++j) {
			int8_t code = superslm::RequantTokenCode(c.xs[j], r, ns.s);
			CHECK_MSG(code == c.expected_codes[j],
			          "%s: pipeline RequantTokenCode(x[%zu]=%d, R, s) == %d, want %d", c.label, j, c.xs[j],
			          static_cast<int>(code), static_cast<int>(c.expected_codes[j]));
		}
	}
}

// ---------------------------------------------------------------------------
// Curie's S2.2 nonlinear scalar primitives red suite. src/intmath.cpp's
// ISqrt/ISqrtTrace/ShiftByMax/IExpFromConstants bodies are fully ported
// (shipped at S-HARDEN-1); every cell below is green against them.
// Test-design record: Claude/Curie/
// superslm-s2.2-nonlinear-test-design-2026-07-19.md.
// ---------------------------------------------------------------------------

SSLM_TEST(TestISqrt, 206) {
	using namespace superslm_test;
	for (size_t i = 0; i < kISqrtCasesCount; ++i) {
		const ISqrtCase& c = kISqrtCases[i];
		int64_t got = superslm::ISqrt(c.n);
		CHECK_MSG(got == c.expected_root, "%s: ISqrt(%lld) == %lld, want %lld", c.label,
		          static_cast<long long>(c.n), static_cast<long long>(got),
		          static_cast<long long>(c.expected_root));
	}
}

SSLM_TEST(TestISqrtTrace, 207) {
	// The §17 blind cell: the fixed 32-iteration recurrence. Every case asserts
	// BOTH that the trace is exactly I_SQRT_ITERATIONS entries long AND that
	// EVERY iterate matches the reference's i_sqrt_trace(n)[k] individually —
	// not only the final root — so a data-dependent early exit (the `while bit
	// > n` prologue intmath.py's docstring explicitly forbids) is caught even
	// if it happens to still land on the correct final root.
	using namespace superslm_test;
	for (size_t i = 0; i < kISqrtCasesCount; ++i) {
		const ISqrtCase& c = kISqrtCases[i];
		int64_t got_iterates[superslm::I_SQRT_ITERATIONS];
		// Poison the buffer so an implementation that writes fewer than
		// I_SQRT_ITERATIONS entries is caught by the per-iterate comparison
		// below rather than silently reading stale/uninitialized data as a
		// coincidental pass.
		for (int k = 0; k < superslm::I_SQRT_ITERATIONS; ++k) got_iterates[k] = INT64_C(-777777777);
		superslm::ISqrtTrace(c.n, got_iterates);
		for (int k = 0; k < superslm::I_SQRT_ITERATIONS; ++k) {
			CHECK_MSG(got_iterates[k] == c.expected_iterates[k],
			          "%s: ISqrtTrace(%lld)[%d] == %lld, want %lld (i_sqrt_trace reference iterate)",
			          c.label, static_cast<long long>(c.n), k, static_cast<long long>(got_iterates[k]),
			          static_cast<long long>(c.expected_iterates[k]));
		}
		CHECK_MSG(got_iterates[superslm::I_SQRT_ITERATIONS - 1] == c.expected_root,
		          "%s: ISqrtTrace(%lld)'s last iterate == %lld, want the ISqrt root %lld", c.label,
		          static_cast<long long>(c.n), static_cast<long long>(got_iterates[superslm::I_SQRT_ITERATIONS - 1]),
		          static_cast<long long>(c.expected_root));
	}
}

SSLM_TEST(TestShiftByMax, 208) {
	using namespace superslm_test;
	for (size_t i = 0; i < kShiftByMaxCasesCount; ++i) {
		const ShiftByMaxCase& c = kShiftByMaxCases[i];
		std::vector<int64_t> got(c.n, INT64_C(-777777777));
		superslm::ShiftByMax(c.logits, c.n, got.data());
		for (size_t j = 0; j < c.n; ++j) {
			CHECK_MSG(got[j] == c.expected[j], "%s: ShiftByMax(...)[%zu] == %lld, want %lld", c.label, j,
			          static_cast<long long>(got[j]), static_cast<long long>(c.expected[j]));
		}
		// The feature claim itself (C9): the maximum is at 0 and every result
		// is <= 0 — i-exp's domain requirement — checked live against this
		// call's own output, not only against the precomputed golden array.
		int64_t max_out = got.empty() ? 0 : got[0];
		for (size_t j = 0; j < c.n; ++j) {
			if (got[j] > max_out) max_out = got[j];
			CHECK_MSG(got[j] <= 0, "%s: ShiftByMax(...)[%zu] == %lld, want <= 0", c.label, j,
			          static_cast<long long>(got[j]));
		}
		CHECK_MSG(max_out == 0, "%s: ShiftByMax(...)'s maximum output == %lld, want exactly 0", c.label,
		          static_cast<long long>(max_out));
	}
}

// PORTED (this session, S-HARDEN-0 final API): IExpFromConstants is REMOVED --
// evaluation is now construct-then-evaluate. Every one of these 36 goldens is
// documented in-domain (each already produced a value that fits int64_t, by
// construction of gen_intmath_fixtures.py's own width-probe search), so
// IExpConstruct is asserted to return kOk before IExpEvaluate is ever called;
// the expected numeric value is UNCHANGED from the original suite (per the
// commission: if any of these 36 values had moved, that would be a real
// regression, not a port detail -- none did, confirmed by this port compiling
// and passing against the unedited kIExpCases table).
SSLM_TEST(TestIExpConstructAndEvaluateMatchGoldenCasesAcrossKIExpCases, 209) {
	using namespace superslm_test;
	for (size_t i = 0; i < kIExpCasesCount; ++i) {
		const IExpCase& c = kIExpCases[i];
		superslm::IExpConstruction out;
		superslm::IExpDomain d = superslm::IExpConstruct(c.q, c.q_ln2, c.q_b, c.q_c, &out);
		CHECK_MSG(d == superslm::IExpDomain::kOk,
		          "%s: IExpConstruct(q=%lld, q_ln2=%lld, q_b=%lld, q_c=%lld) returned domain %d, want kOk "
		          "-- every kIExpCases golden is documented in-domain",
		          c.label, static_cast<long long>(c.q), static_cast<long long>(c.q_ln2),
		          static_cast<long long>(c.q_b), static_cast<long long>(c.q_c), static_cast<int>(d));
		if (d != superslm::IExpDomain::kOk) continue;
		int64_t got = superslm::IExpEvaluate(out);
		CHECK_MSG(got == c.expected,
		          "%s: IExpEvaluate(IExpConstruct(q=%lld, q_ln2=%lld, q_b=%lld, q_c=%lld)) == %lld, want %lld",
		          c.label, static_cast<long long>(c.q), static_cast<long long>(c.q_ln2),
		          static_cast<long long>(c.q_b), static_cast<long long>(c.q_c), static_cast<long long>(got),
		          static_cast<long long>(c.expected));
	}
}

// PORTED (this session): construct-then-evaluate, same claim.
SSLM_TEST(TestIExpConstructAndEvaluateClipClampsIdenticallyAcrossFamily, 210) {
	// C8's clip claim, checked live across every "realistic_s*"/"qln2_min"
	// family in the fixture set: the clip-boundary input and the beyond-clip
	// input for the SAME (q_ln2, q_b, q_c) triple must produce the SAME
	// result, executed here rather than only asserted equal inside the
	// generator that produced the golden values.
	using namespace superslm_test;
	size_t pairs_checked = 0;
	for (size_t i = 0; i + 1 < kIExpCasesCount; ++i) {
		const IExpCase& clip = kIExpCases[i];
		const IExpCase& beyond = kIExpCases[i + 1];
		std::string clip_label(clip.label);
		std::string beyond_label(beyond.label);
		bool is_clip_pair = clip_label.find("clip_boundary") != std::string::npos &&
		                     beyond_label.find("beyond_clip") != std::string::npos &&
		                     clip.q_ln2 == beyond.q_ln2 && clip.q_b == beyond.q_b && clip.q_c == beyond.q_c;
		if (!is_clip_pair) continue;
		++pairs_checked;
		superslm::IExpConstruction clip_out, beyond_out;
		superslm::IExpDomain clip_d = superslm::IExpConstruct(clip.q, clip.q_ln2, clip.q_b, clip.q_c, &clip_out);
		superslm::IExpDomain beyond_d =
		    superslm::IExpConstruct(beyond.q, beyond.q_ln2, beyond.q_b, beyond.q_c, &beyond_out);
		CHECK_MSG(clip_d == superslm::IExpDomain::kOk && beyond_d == superslm::IExpDomain::kOk,
		          "%s/%s: both are documented in-domain fixtures; IExpConstruct returned %d/%d, want kOk/kOk",
		          clip.label, beyond.label, static_cast<int>(clip_d), static_cast<int>(beyond_d));
		if (clip_d != superslm::IExpDomain::kOk || beyond_d != superslm::IExpDomain::kOk) continue;
		int64_t got_clip = superslm::IExpEvaluate(clip_out);
		int64_t got_beyond = superslm::IExpEvaluate(beyond_out);
		CHECK_MSG(got_clip == got_beyond,
		          "%s/%s: clip-boundary result %lld != beyond-clip result %lld for the same constants",
		          clip.label, beyond.label, static_cast<long long>(got_clip), static_cast<long long>(got_beyond));
	}
	CHECK_MSG(pairs_checked == 6, "expected 6 clip/beyond-clip fixture pairs (5 realistic + 1 qln2_min), found %zu",
	          pairs_checked);
}

// ---------------------------------------------------------------------------
// Curie's S2.6-amendment suite for IExpConstantsInDomain (D-SLM78/79/81;
// Claude/Loki/softmax-s2.6-strike-2026-07-21.md; Claude/Curie/
// superslm-s2.6-softmax-iexp-domain-test-design-2026-07-21.md). This primitive
// is declared and shipped in src/intmath.cpp / include/superslm/intmath.h.
//
// The two accessor cells originally here (TestIExpShiftMatchesIndependentlyDerivedZ,
// TestIExpBaseMatchesIndependentlyDerivedBase, against IExpShift/IExpBase) were
// PORTED to TestIExpConstructMatchesAccessorCasesZAndBase above (Brunel, mid-build,
// 2026-07-21): IExpShift/IExpBase are removed from the public header by S-HARDEN-0,
// so every cell using them had to move to IExpConstruct's IExpConstruction output.
// kIExpAccessorCases itself (36 rows) is unchanged -- its values do not depend on
// which function reads them.
//
// Every expected value in sslm_iexp_domain_fixtures.h is computed by
// tests/gen_iexp_domain_fixtures.py, transcribing IExpFromConstants's documented
// five-line decomposition directly in Python arbitrary precision -- never by
// calling the primitives under test and never by re-deriving the bound in
// fixed-width int64 (the exact shape of the D-SLM81 defect this amendment
// fixes).
// ---------------------------------------------------------------------------

SSLM_TEST(TestIExpConstantsInDomainAcrossCorpus, 211) {
	// The full domain-predicate corpus: the strike's exact input, both the z=0 and
	// the z>=1 in-domain/out-of-domain boundaries forced EXACTLY on both sides (a
	// boundary is only authored where a valid int64_t q_c actually sits on the
	// transition -- see gen_iexp_domain_fixtures.py's note on the rejected
	// 1733160715-base z=1/z=30 construction, which forces nothing because the
	// theoretical boundary q_c there does not fit int64_t), the q_b-alone overflow
	// axis (large q_b, q_c=1, distinct from the strike's q_c-dominated overflow),
	// a realistic operating-scale positive case, and every one of the 36 fixtures
	// already shipped in sslm_intmath_fixtures.h's kIExpCases (each must be
	// in-domain, since each already produced a golden that fits int64_t).
	using namespace superslm_test;
	for (size_t i = 0; i < kIExpDomainCasesCount; ++i) {
		const IExpDomainCase& c = kIExpDomainCases[i];
		bool got = superslm::IExpConstantsInDomain(c.q, c.q_ln2, c.q_b, c.q_c);
		CHECK_MSG(got == c.expected_in_domain,
		          "%s: IExpConstantsInDomain(q=%lld, q_ln2=%lld, q_b=%lld, q_c=%lld) == %s, want %s", c.label,
		          static_cast<long long>(c.q), static_cast<long long>(c.q_ln2), static_cast<long long>(c.q_b),
		          static_cast<long long>(c.q_c), got ? "true" : "false", c.expected_in_domain ? "true" : "false");
	}
}

SSLM_TEST(TestIExpConstantsInDomainRejectsStrikeExactInput, 212) {
	// Standalone, individually diagnosable regression cell for the defect itself
	// (Claude/Loki/softmax-s2.6-strike-2026-07-21.md): this exact call is
	// contract-legal under every documented LOWER-bound precondition (q<=0,
	// q_ln2>=1) and yet base^2+q_c = 12,227,218,100,874,087,032 exceeds INT64_MAX.
	// Redundant with one row of TestIExpConstantsInDomainAcrossCorpus by design --
	// this is the one cell this whole amendment exists to force, and it must be
	// able to fail on its own without scanning a table's output to find it.
	bool in_domain = superslm::IExpConstantsInDomain(INT64_C(0), INT64_C(887904998), INT64_C(1733160715),
	                                                  INT64_C(9223372036854775807));
	CHECK_MSG(!in_domain,
	          "IExpConstantsInDomain(0, 887904998, 1733160715, 2^63-1) == true, want false -- this is the "
	          "strike's exact contract-legal input for which IExpFromConstants returns a NEGATIVE value "
	          "(Claude/Loki/softmax-s2.6-strike-2026-07-21.md)");
}

namespace {

// The header's claimed sufficient-AND-necessary condition for
// IExpConstantsInDomain's one-call-per-triple shortcut (include/superslm/
// intmath.h, corrected at c33843d: "The one-call-per-triple shortcut holds
// iff 2*q_b >= q_ln2 - 1"). This is prose about how a CALLER may use the
// primitive, not behavior the primitive itself performs, so nothing in
// src/intmath.cpp encodes it; TestIExpConstantsInDomainShortcutConditionMatchesHeaderClaim
// below is the only place that does, and is therefore the only place a wrong
// version of this claim (Claude/Poirot/93622d3-s2.2-iexp-amendment-close-
// round-review-2026-07-21.md, finding N7) can be caught by the suite rather
// than by a reviewer re-deriving it by hand.
bool IExpShortcutHolds(int64_t q_ln2, int64_t q_b) { return 2 * q_b >= q_ln2 - 1; }

}  // namespace

SSLM_TEST(TestIExpConstantsInDomainShortcutConditionMatchesHeaderClaim, 213) {
	// Four scenarios authored in tests/gen_iexp_domain_fixtures.py at
	// q_ln2 = 3,000,000,000 (labels "shortcut_*_q0" / "shortcut_*_far_end" in
	// kIExpDomainCases), each placing q_c so one end of the row -- q=0 or the
	// far end q=-(q_ln2-1) -- sits exactly at its own in-domain boundary:
	//
	//   A (q_b=1,800,000,000): condition HOLDS.       q0=false, far=true.
	//   B (q_b=1,000,000,000): condition FAILS.       q0=true,  far=false.
	//   C (q_b=1,500,000,000, the least satisfying):  condition HOLDS.  q0=true,  far=true.
	//   D (q_b=1,499,999,999, one less than C):       condition FAILS.  q0=true,  far=false.
	//
	// For every scenario this test (a) pins IExpShortcutHolds's own return
	// value against the scenario's independently-known intent, and (b) checks
	// that the REAL primitive's behavior at q=0 and at the row's other
	// extreme is consistent with that claim: where the shortcut holds, a
	// caller who discharges the row at q=0 alone can never get a false
	// all-clear (q0=true implies far=true); where it fails, this suite
	// constructed an actual witness of exactly that false all-clear
	// (q0=true, far=false) -- proving the condition is not merely sufficient
	// but tight, i.e. genuinely necessary at these constants.
	using namespace superslm_test;

	struct Scenario {
		const char* q0_label;
		const char* far_label;
		int64_t q_ln2;
		int64_t q_b;
		bool expected_holds;
	};
	static const Scenario kScenarios[] = {
	    {"shortcut_holds_interior_A_q0", "shortcut_holds_interior_A_far_end", INT64_C(3000000000),
	     INT64_C(1800000000), true},
	    {"shortcut_fails_interior_B_q0", "shortcut_fails_interior_B_far_end", INT64_C(3000000000),
	     INT64_C(1000000000), false},
	    {"shortcut_boundary_holds_C_q0", "shortcut_boundary_holds_C_far_end", INT64_C(3000000000),
	     INT64_C(1500000000), true},
	    {"shortcut_boundary_fails_D_q0", "shortcut_boundary_fails_D_far_end", INT64_C(3000000000),
	     INT64_C(1499999999), false},
	};

	for (const Scenario& s : kScenarios) {
		bool computed_holds = IExpShortcutHolds(s.q_ln2, s.q_b);
		CHECK_MSG(computed_holds == s.expected_holds,
		          "IExpShortcutHolds(q_ln2=%lld, q_b=%lld) == %s, want %s (%s) -- the header's condition "
		          "no longer matches this scenario's known intent",
		          static_cast<long long>(s.q_ln2), static_cast<long long>(s.q_b), computed_holds ? "true" : "false",
		          s.expected_holds ? "true" : "false", s.q0_label);

		const IExpDomainCase* q0_case = nullptr;
		const IExpDomainCase* far_case = nullptr;
		for (size_t i = 0; i < kIExpDomainCasesCount; ++i) {
			const IExpDomainCase& c = kIExpDomainCases[i];
			if (std::strcmp(c.label, s.q0_label) == 0) q0_case = &c;
			if (std::strcmp(c.label, s.far_label) == 0) far_case = &c;
		}
		CHECK_MSG(q0_case != nullptr, "fixture label not found: %s", s.q0_label);
		CHECK_MSG(far_case != nullptr, "fixture label not found: %s", s.far_label);
		if (q0_case == nullptr || far_case == nullptr) continue;

		bool got_q0 = superslm::IExpConstantsInDomain(q0_case->q, q0_case->q_ln2, q0_case->q_b, q0_case->q_c);
		bool got_far = superslm::IExpConstantsInDomain(far_case->q, far_case->q_ln2, far_case->q_b, far_case->q_c);

		if (computed_holds) {
			CHECK_MSG(!(got_q0 && !got_far),
			          "%s: shortcut claimed to hold (q_ln2=%lld, q_b=%lld) but q=0 answered true while the far "
			          "end answered false -- discharging at q=0 alone is unsound here, contradicting the header's "
			          "claim",
			          s.q0_label, static_cast<long long>(s.q_ln2), static_cast<long long>(s.q_b));
		} else {
			CHECK_MSG(got_q0 && !got_far,
			          "%s: this scenario's witness (q_ln2=%lld, q_b=%lld) no longer demonstrates q=0-alone "
			          "unsoundness (got_q0=%s, got_far=%s) -- the constructed q_c must produce true at q=0 and "
			          "false at the far end to prove the shortcut genuinely fails here",
			          s.q0_label, static_cast<long long>(s.q_ln2), static_cast<long long>(s.q_b),
			          got_q0 ? "true" : "false", got_far ? "true" : "false");
		}
	}
}

// D-SLM79 part 2's internal domain assert on IExpFromConstants no longer has a
// subject under the final API (IExpFromConstants is removed; IExpConstruct/
// IExpEvaluate assert nothing, in any build configuration). The wrapped-value
// golden it used to pin survives as
// TestIExpConstructAndEvaluateProducesKnownWrappedValueForOutOfDomainConstants,
// defined further below -- see that function's own comment for the full account.

// ---------------------------------------------------------------------------
// Curie's S2.3 RopeApplyPair red suite. src/intmath.cpp's RopeApplyPair
// body is fully ported (shipped at S-HARDEN-1); every cell below is green
// against it. Test-design record: Claude/Curie/
// superslm-s2.3-rope-test-design-2026-07-19.md.
// ---------------------------------------------------------------------------

SSLM_TEST(TestRopeApplyPair, 221) {
	using namespace superslm_test;
	for (size_t i = 0; i < kRopeCasesCount; ++i) {
		const RopeCase& c = kRopeCases[i];
		superslm::RopePair got = superslm::RopeApplyPair(c.x, c.y, c.cos_q30, c.sin_q30);
		CHECK_MSG(got.x == c.expected_x,
		          "%s: RopeApplyPair(x=%d, y=%d, cos=%d, sin=%d).x == %lld, want %lld", c.label, c.x, c.y,
		          c.cos_q30, c.sin_q30, static_cast<long long>(got.x), static_cast<long long>(c.expected_x));
		CHECK_MSG(got.y == c.expected_y,
		          "%s: RopeApplyPair(x=%d, y=%d, cos=%d, sin=%d).y == %lld, want %lld", c.label, c.x, c.y,
		          c.cos_q30, c.sin_q30, static_cast<long long>(got.y), static_cast<long long>(c.expected_y));
	}
}

SSLM_TEST(TestRopeApplyPairIdentityIsExact, 222) {
	// The identity/passthrough claim (dimension 10, feature oracle): rotating by
	// angle 0 (cos=ROPE_ONE, sin=0) must reproduce (x, y) EXACTLY — the
	// x*ROPE_ONE / ROPE_ONE division has zero remainder, so no rounding may
	// perturb the result. Checked live against the executed call, not only
	// against the precomputed golden in kRopeCases.
	using namespace superslm_test;
	const int32_t kOne = superslm::ROPE_ONE;
	const int32_t xs[] = {0, 12345, -12345, superslm::kInt32Max, superslm::kInt32Min};
	for (int32_t x : xs) {
		superslm::RopePair got = superslm::RopeApplyPair(x, x, kOne, 0);
		CHECK_MSG(got.x == static_cast<int64_t>(x), "identity x=%d: RopeApplyPair(...).x == %lld, want %d", x,
		          static_cast<long long>(got.x), x);
		CHECK_MSG(got.y == static_cast<int64_t>(x), "identity x=%d: RopeApplyPair(...).y == %lld, want %d", x,
		          static_cast<long long>(got.y), x);
	}
}

SSLM_TEST(TestRopeApplyPairQuarterTurnIsExact, 223) {
	// The quarter-turn claim, executed live: cos=0 isolates the pure-sin
	// rotation, and (-y, x) / (y, -x) must hold exactly for every (x, y) in
	// this suite's kRopeCases quarter-turn cells (found by label prefix).
	using namespace superslm_test;
	size_t checked = 0;
	for (size_t i = 0; i < kRopeCasesCount; ++i) {
		const RopeCase& c = kRopeCases[i];
		std::string label(c.label);
		if (label.rfind("quarter_pos_sin_", 0) == 0) {
			superslm::RopePair got = superslm::RopeApplyPair(c.x, c.y, 0, superslm::ROPE_ONE);
			CHECK_MSG(got.x == -static_cast<int64_t>(c.y) && got.y == static_cast<int64_t>(c.x),
			          "%s: cos=0,sin=ROPE_ONE must give (-y, x) == (%lld, %lld), got (%lld, %lld)", c.label,
			          static_cast<long long>(-c.y), static_cast<long long>(c.x), static_cast<long long>(got.x),
			          static_cast<long long>(got.y));
			++checked;
		} else if (label.rfind("quarter_neg_sin_", 0) == 0) {
			superslm::RopePair got = superslm::RopeApplyPair(c.x, c.y, 0, -superslm::ROPE_ONE);
			CHECK_MSG(got.x == static_cast<int64_t>(c.y) && got.y == -static_cast<int64_t>(c.x),
			          "%s: cos=0,sin=-ROPE_ONE must give (y, -x) == (%lld, %lld), got (%lld, %lld)", c.label,
			          static_cast<long long>(c.y), static_cast<long long>(-c.x), static_cast<long long>(got.x),
			          static_cast<long long>(got.y));
			++checked;
		}
	}
	CHECK_MSG(checked == 8, "expected 8 quarter-turn fixture cells (4 pos-sin + 4 neg-sin), found %zu", checked);
}

SSLM_TEST(TestRopeApplyPairWideInputExceedsInt32Range, 224) {
	// The int64-return-type claim (dimension 10, load-bearing): at least one
	// wide-input cell's result must exceed INT32_MAX in magnitude, checked live
	// against this suite's own executed calls — not only asserted true of the
	// precomputed golden inside the generator. A RopePair whose fields were
	// silently truncated to int32 would fail this even if the C++ under test
	// otherwise computed the right 64-bit value internally.
	using namespace superslm_test;
	int wide_exceeding = 0;
	int64_t largest_magnitude = 0;
	for (size_t i = 0; i < kRopeCasesCount; ++i) {
		const RopeCase& c = kRopeCases[i];
		std::string label(c.label);
		if (label.rfind("wide_", 0) != 0) continue;
		superslm::RopePair got = superslm::RopeApplyPair(c.x, c.y, c.cos_q30, c.sin_q30);
		int64_t mag_x = got.x < 0 ? -got.x : got.x;
		int64_t mag_y = got.y < 0 ? -got.y : got.y;
		if (mag_x > largest_magnitude) largest_magnitude = mag_x;
		if (mag_y > largest_magnitude) largest_magnitude = mag_y;
		if (mag_x > superslm::kInt32Max || mag_y > superslm::kInt32Max) ++wide_exceeding;
	}
	CHECK_MSG(wide_exceeding >= 1,
	          "at least one wide-input case must produce a result exceeding INT32_MAX in magnitude "
	          "(the int64 return type is load-bearing); found %d",
	          wide_exceeding);
	CHECK_MSG(largest_magnitude > superslm::kInt32Max,
	          "largest constructed magnitude %lld must exceed INT32_MAX (%d)",
	          static_cast<long long>(largest_magnitude), superslm::kInt32Max);
}

SSLM_TEST(TestRopeApplyPairTieRoundsAwayFromZero, 225) {
	// The tie-inheritance claim, executed live: RopeApplyPair's rounding must
	// move strictly off zero on an exact-half input, on both signs, for the
	// x-component alone, the y-component alone, and both simultaneously —
	// proving RoundingDivideByPOT's away-from-zero rule (C3) is inherited
	// correctly rather than merely matching a precomputed golden that happened
	// to be produced the same (wrong) way.
	using namespace superslm_test;
	const char* kTieLabels[] = {"tie_x_pos",  "tie_x_neg",   "tie_y_pos",
	                             "tie_y_neg",  "tie_both_pos", "tie_both_neg"};
	int checked = 0;
	for (const char* want_label : kTieLabels) {
		bool found = false;
		for (size_t i = 0; i < kRopeCasesCount; ++i) {
			const RopeCase& c = kRopeCases[i];
			if (std::strcmp(c.label, want_label) != 0) continue;
			found = true;
			++checked;
			superslm::RopePair got = superslm::RopeApplyPair(c.x, c.y, c.cos_q30, c.sin_q30);
			CHECK_MSG(got.x == c.expected_x, "%s: RopeApplyPair(...).x == %lld, want %lld (away-from-zero tie)",
			          c.label, static_cast<long long>(got.x), static_cast<long long>(c.expected_x));
			CHECK_MSG(got.y == c.expected_y, "%s: RopeApplyPair(...).y == %lld, want %lld (away-from-zero tie)",
			          c.label, static_cast<long long>(got.y), static_cast<long long>(c.expected_y));
			bool moved_off_zero = (got.x != 0) || (got.y != 0);
			CHECK_MSG(moved_off_zero, "%s: an exact-half tie must round strictly off zero", c.label);
		}
		CHECK_MSG(found, "expected a fixture cell labeled \"%s\"", want_label);
	}
	CHECK_MSG(checked == 6, "expected 6 tie fixture cells, found %d", checked);
}

// S3 (Poirot review ac34677, 2026-07-28): RowBoundsWide (src/intmath.cpp:
// 279-280) reads x[0] before testing n at all -- with a null data pointer
// and n == 0 this is a null-pointer dereference, and the reviewer's own
// probe process terminated at the call (P7). Unlike matmul_zero_in_
// channels above, this is a genuine memory access, not an assert(): it is
// not compiled out under NDEBUG, so it must not crash in EITHER build
// configuration once fixed.
static int ProbeRowBoundsWideZeroLenNullPtr() {
	int64_t out_max = 0;
	int64_t out_min = 0;
	std::printf("%s\n", CrashProbeBeganMarker("row_bounds_wide_zero_len_null_ptr").c_str());
	std::printf("crash-probe row_bounds_wide_zero_len_null_ptr: calling RowBoundsWide "
	            "with a null data pointer and n=0 (no n >= 1 precondition is documented "
	            "on this primitive or on NarrowRowChecked, S3)\n");
	std::fflush(stdout);
	superslm::RowBoundsWide(nullptr, /*n=*/0, &out_max, &out_min);
	std::printf("PROBE DID NOT CRASH (out_max=%lld out_min=%lld)\n",
	            static_cast<long long>(out_max), static_cast<long long>(out_min));
	return 0;
}
namespace { CrashProbeRegistrar gRegisterProbeRowBoundsWideZeroLenNullPtr(
    "row_bounds_wide_zero_len_null_ptr", &ProbeRowBoundsWideZeroLenNullPtr); }


// S3 (Poirot review ac34677, 2026-07-28; SuperSLM_S3a_WalkingSkeleton_Plan.md
// Sec11 S3.1, F-S3-7): RowBoundsWide reads x[0] before testing n at all
// (src/intmath.cpp:279-280); with a null data pointer and n == 0 -- a
// degenerate but in-contract input (neither RowBoundsWide's own header nor
// NarrowRowChecked's contract documents an n >= 1 precondition) -- this is a
// null-pointer dereference. Unlike the assert()-gated contract violation
// above, a null-pointer dereference is NOT compiled out under NDEBUG (it is
// a genuine memory access, not a debug check), so this cell asserts
// kRanNoCrash unconditionally, with no #ifdef NDEBUG branch: the sibling
// primitive MaxAbsReduceWide already treats n == 0 as an in-contract,
// defined case (D' == 1, "the empty reduction", intmath.h:150-152), so n ==
// 0 is not a caller-ensures contract violation this primitive is entitled
// to leave undefined either.
SSLM_TEST(TestRowBoundsWideZeroLenNullPtrDoesNotCrash, 196) {
	static const char* kProbeName = "row_bounds_wide_zero_len_null_ptr";
	std::string tail;
	CrashProbeOutcome outcome = RunsCrashProbeAndCrashes(kProbeName, &tail);
	CHECK_MSG(outcome == CrashProbeOutcome::kRanNoCrash,
	          "RowBoundsWide(nullptr, 0, &out_max, &out_min) must not crash in any build "
	          "configuration -- outcome was %s, child output was: %s",
	          CrashProbeOutcomeName(outcome), tail.c_str());
}


// REWORKED 2026-07-22 (S-HARDEN-0 final API port). This cell used to pin TWO things
// under the old two-function API: a debug-build abort (IExpFromConstants's internal
// "asserts success" contract), and the exact NDEBUG wrapped value
// (-6219525972835464584) for IExpFromConstants(0, 887904998, 1733160715, INT64_MAX).
//
// Under the final API there is no contract-violating call left to abort on.
// IExpConstruct(0, 887904998, 1733160715, INT64_MAX, &out) does not violate any
// contract -- q<=0 and 1<=q_ln2<=ceiling both hold, and q_p+q_b is representable --
// it returns kNotRepresentable, a documented, non-erroneous outcome that FILLS *out,
// and IExpEvaluate(out) is TOTAL on that construction in EVERY build configuration
// (no assert anywhere in IExpConstruct or IExpEvaluate -- confirmed by reading both
// bodies in src/intmath.cpp, S-HARDEN-0). There is no debug-vs-release split left to
// assert, because there is no code path in this call that ever asserts, in any
// configuration. The assert half of this cell's original claim therefore has no
// surviving subject, and is not silently dropped -- it is named here as retired,
// with the reason, per this project's "a gap is a finding, not a silent omission"
// discipline.
//
// The wrapped-value golden MUST survive, and does: this is exactly
// kIExpConstructCases' "notrepresentable_strike_witness" row (D-SLM78's original
// strike input), so this cell is also a standalone, individually diagnosable
// regression check for that one row, checked directly rather than by scanning a
// table (the same rationale TestIExpConstantsInDomainRejectsStrikeExactInput already
// uses for the predicate side of the same input).
SSLM_TEST(TestIExpConstructAndEvaluateProducesKnownWrappedValueForOutOfDomainConstants, 214) {
	superslm::IExpConstruction out;
	superslm::IExpDomain d = superslm::IExpConstruct(INT64_C(0), INT64_C(887904998), INT64_C(1733160715),
	                                                  INT64_C(9223372036854775807), &out);
	CHECK_MSG(d == superslm::IExpDomain::kNotRepresentable,
	          "IExpConstruct(0, 887904998, 1733160715, 2^63-1) returned domain %d, want "
	          "kNotRepresentable -- this is the strike's exact contract-legal input for which the "
	          "decomposition is well-formed but base^2+q_c does not fit int64 "
	          "(Claude/Loki/softmax-s2.6-strike-2026-07-21.md)",
	          static_cast<int>(d));
	if (d != superslm::IExpDomain::kNotRepresentable) return;
	CHECK_MSG(out.z() == 0 && out.base() == INT64_C(1733160715) && out.q_c() == INT64_C(9223372036854775807),
	          "IExpConstruct(...) is kNotRepresentable and must FILL *out with the well-formed "
	          "decomposition -- got z=%lld base=%lld q_c=%lld, want z=0 base=1733160715 q_c=%lld",
	          static_cast<long long>(out.z()), static_cast<long long>(out.base()),
	          static_cast<long long>(out.q_c()), static_cast<long long>(INT64_C(9223372036854775807)));
	int64_t got = superslm::IExpEvaluate(out);
	CHECK_MSG(got == INT64_C(-6219525972835464584),
	          "IExpEvaluate(construction from the strike's out-of-domain constants) == %lld, want "
	          "-6219525972835464584 (the exact wrapped value the strike observed, "
	          "Claude/Loki/softmax-s2.6-strike-2026-07-21.md, D-SLM80 behaviour-preservation) -- this "
	          "value must be identical in EVERY build configuration: IExpEvaluate has no assert and "
	          "no NDEBUG split, unlike the retired IExpFromConstants this cell used to pin",
	          static_cast<long long>(got));
}

// ---------------------------------------------------------------------------
// RETIRED 2026-07-22 (S-HARDEN-0 final API port). Curie's S-HARDEN-0 population
// suite, LAYER A, used to live here: a crash-probe population over the OLD
// two-function API (IExpFromConstants / IExpConstantsInDomain), distinguishing an
// "eval" call path (F9's finding: the evaluator consults its accessors, which
// overflow, before ever asserting the domain) from a "pred" call path (F21's
// finding: the guard's own internal accessor call overflows on an unbounded q_b
// before the guard can answer). Both required child-process isolation because,
// at f078403 and through this branch's a1d7986 revision, the un-fixed call really
// could crash the ENTIRE test process.
//
// The final API collapses the distinction this population existed to test.
// IExpFromConstants is removed; the only function that takes raw (q, q_ln2, q_b,
// q_c) is IExpConstruct, and it is TOTAL -- asserts nothing, executes no UB, in
// every build configuration (read: both IExpConstruct's and IExpEvaluate's bodies
// in src/intmath.cpp, S-HARDEN-0; confirmed further by Poirot's a1d7986 review,
// 5,684,354 executed quadruples under UBSan, 0 diagnostics). There is no longer a
// second call path for "eval" to name, and no longer a crash to isolate a child
// process against -- an evaluation is now only ever reachable through a
// construction IExpConstruct itself validated (or the safe default {0,0,0}), which
// TestIExpConstructionDefaultIsSafeToEvaluate and the two IExpEvaluate-totality
// static_asserts below prove structurally rather than by population.
//
// No witness value is dropped. Every input this population's ten rows drove is a
// row of kIExpConstructCases (tests/sslm_iexp_domain_fixtures.h) now: F9's four
// ceiling witnesses are badqln2_last_valid_ceiling / badqln2_first_invalid_ceiling
// / badqln2_interior_invalid_ceiling / badqln2_f9_witness_int64_max; F21's four
// q_b witnesses are badqb_f21_witness_qb_int64_min / badqb_boundary_first_unsafe /
// badqb_interior_unsafe / badqb_representable_boundary_square_not_representable
// (the last one carries a corrected label: it IS the old "boundary_last_safe"
// witness, exact same (q, q_ln2, q_b) -- accurate about q_p+q_b being
// representable, but the old label implied overall domain membership, which this
// input does not have once base=INT64_MIN is squared; see that row's own comment
// in gen_iexp_domain_fixtures.py). The two "_eval_" duplicates named no witness
// value the "pred" rows above did not already carry, so nothing beyond those eight
// is owed. TestIExpConstructMatchesIndependentOracleAcrossCases (below) sweeps all
// of kIExpConstructCases, including these rows, in-process, under whichever
// sanitizer configuration the suite is built with -- the same population, proven
// the same way execution always proved it, without a probe.
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Curie's S-HARDEN-0 population suite (LAYER B): the checked `IExpConstruct` /
// `IExpConstruction` / `IExpDomain` / `IExpEvaluate` entry point, final API
// (SuperSLM_Plan.md's S-HARDEN-0 sub-slot; Claude/Curie/
// superslm-s-harden-0-test-design-2026-07-21.md). `IExpConstruction`'s fields are
// PRIVATE (`z()`/`base()`/`q_c()` accessors only, populated solely by
// `IExpConstruct`'s friend access) and `IExpEvaluate` takes only the construction --
// this section's own two structural static_asserts, immediately below, pin both of
// those properties at compile time, and TestIExpConstructionDefaultIsSafeToEvaluate
// pins that the only OTHER reachable origin (a default-constructed object) is safe.
// ---------------------------------------------------------------------------

// Property: "IExpEvaluate takes only a construction; q_c is carried, never passed
// separately." A separate q_c parameter would let a caller validate against one
// constant and evaluate against another -- the exact two-derivations-drift defect
// this slot's own header comment names (F10). The final API makes that call
// unwritable: IExpEvaluate's signature has no q_c parameter at all. This is a
// property OF THE SIGNATURE, so it is pinned at compile time -- a future edit that
// re-introduces a second q_c argument fails to compile this file, immediately,
// rather than waiting for a test to notice a caller passing mismatched values.
static_assert(std::is_same<decltype(&superslm::IExpEvaluate), int64_t (*)(const superslm::IExpConstruction&)>::value,
              "IExpEvaluate must take ONLY a construction -- q_c must be carried by it, not accepted "
              "as a separate argument (S-HARDEN-0: a separate q_c would let a caller validate "
              "against one constant and evaluate against another, the two-derivations-drift defect "
              "F10 already recorded)");

// Property: "IExpConstruction's fields are private, and only IExpConstruct can
// populate one." A type with private non-static data members is never an
// aggregate (C++ [dcl.init.aggr]), so this is equivalent to "no caller can
// brace-initialize or field-assign an IExpConstruction into a state IExpConstruct
// never produced" -- the exact defect this slot's header comment names: "a public
// {z, base} aggregate would admit z = 999 from any caller," which is an unchecked
// shift and undefined behaviour in IExpEvaluate. Pinned at compile time for the
// same reason as the assert above: this is a property of the TYPE, not of any one
// call, so a test that only ever calls IExpConstruct correctly could never observe
// a regression here -- only the type system can.
static_assert(!std::is_aggregate<superslm::IExpConstruction>::value,
              "IExpConstruction must not be an aggregate -- z_/base_/q_c_ must stay private and "
              "unsettable directly by any caller (S-HARDEN-0: a public aggregate would admit an "
              "unchecked z outside [0, I_EXP_CLIP_N], which IExpEvaluate's shift cannot survive)");

// Property: the OTHER reachable origin of an IExpConstruction -- besides one
// IExpConstruct itself validated -- is a default-constructed object, and the header
// documents it as safe ("a default-constructed one is {0, 0, 0}, which is safe: z =
// 0 is a legal shift"). Checked directly rather than assumed: default-construct,
// read the three accessors, and evaluate it, confirming the documented zero state
// and that evaluating it executes no UB (run under ASan+UBSan, this cell would trap
// if z were anything other than a legal shift amount) and returns the value the
// formula predicts ((0^2 + 0) >> 0 == 0).
SSLM_TEST(TestIExpConstructionDefaultIsSafeToEvaluate, 215) {
	superslm::IExpConstruction c;
	CHECK_MSG(c.z() == 0 && c.base() == 0 && c.q_c() == 0,
	          "a default-constructed IExpConstruction must be {z=0, base=0, q_c=0} -- got "
	          "z=%lld base=%lld q_c=%lld",
	          static_cast<long long>(c.z()), static_cast<long long>(c.base()), static_cast<long long>(c.q_c()));
	int64_t got = superslm::IExpEvaluate(c);
	CHECK_MSG(got == 0,
	          "IExpEvaluate(default-constructed IExpConstruction) == %lld, want 0 -- (0^2+0)>>0 == 0, "
	          "and z=0 is documented as a legal (no-op) shift",
	          static_cast<long long>(got));
}

// Stringifies IExpDomain for CHECK_MSG output and for comparison against the
// fixture's expected_domain field (a string, kept free of a compile-time dependency
// on IExpDomain's exact enum values -- see sslm_iexp_domain_fixtures.h's own
// comment). Mirrors this file's existing CrashProbeOutcomeName pattern.
static const char* IExpDomainName(IExpDomain d) {
	switch (d) {
		case IExpDomain::kOk:
			return "kOk";
		case IExpDomain::kNotRepresentable:
			return "kNotRepresentable";
		case IExpDomain::kBadQ:
			return "kBadQ";
		case IExpDomain::kBadQLn2:
			return "kBadQLn2";
		case IExpDomain::kBadQB:
			return "kBadQB";
	}
	return "(unknown IExpDomain)";
}

// "*out is FILLED whenever the decomposition is well-formed -- that is, for kOk AND
// kNotRepresentable, and IExpEvaluate is TOTAL over both." True for exactly those
// two outcome names.
static bool IExpDomainNameIsWellFormed(const char* name) {
	return std::strcmp(name, "kOk") == 0 || std::strcmp(name, "kNotRepresentable") == 0;
}

// The primary sweep: every row's returned IExpDomain against the independent
// oracle, and -- for the two well-formed outcomes -- out.z()/out.base()/out.q_c()
// AND IExpEvaluate(out) against the same oracle. This is the cell that proves
// IExpEvaluate's totality (dimension: "IExpEvaluate is total on any construction it
// can be given, including one from a kNotRepresentable outcome, which still yields
// the exact wrapped value") across the FULL population, not only the single
// standalone golden TestIExpConstructAndEvaluateProducesKnownWrappedValueForOutOfDomainConstants
// pins. Does not poison *out with a sentinel (that is the dedicated concern of
// TestIExpConstructOutContractPerOutcome below); this function is the one place a
// reader confirms "does the entry point classify, decompose, and evaluate
// correctly" without also having to reason about the untouched-vs-filled contract
// at the same time.
SSLM_TEST(TestIExpConstructMatchesIndependentOracleAcrossCases, 216) {
	using namespace superslm_test;
	for (size_t i = 0; i < kIExpConstructCasesCount; ++i) {
		const IExpConstructCase& c = kIExpConstructCases[i];
		IExpConstruction out;
		IExpDomain got = IExpConstruct(c.q, c.q_ln2, c.q_b, c.q_c, &out);
		CHECK_MSG(std::strcmp(IExpDomainName(got), c.expected_domain) == 0,
		          "%s: IExpConstruct(%lld, %lld, %lld, %lld) returned %s, independently-derived "
		          "oracle expects %s",
		          c.label, static_cast<long long>(c.q), static_cast<long long>(c.q_ln2),
		          static_cast<long long>(c.q_b), static_cast<long long>(c.q_c), IExpDomainName(got),
		          c.expected_domain);
		if (std::strcmp(IExpDomainName(got), c.expected_domain) != 0) continue;
		if (IExpDomainNameIsWellFormed(c.expected_domain)) {
			CHECK_MSG(out.z() == c.expected_z,
			          "%s: IExpConstruct(%lld, %lld, %lld, %lld) [%s].z() == %lld, oracle expects %lld",
			          c.label, static_cast<long long>(c.q), static_cast<long long>(c.q_ln2),
			          static_cast<long long>(c.q_b), static_cast<long long>(c.q_c), c.expected_domain,
			          static_cast<long long>(out.z()), static_cast<long long>(c.expected_z));
			CHECK_MSG(out.base() == c.expected_base,
			          "%s: IExpConstruct(%lld, %lld, %lld, %lld) [%s].base() == %lld, oracle expects %lld",
			          c.label, static_cast<long long>(c.q), static_cast<long long>(c.q_ln2),
			          static_cast<long long>(c.q_b), static_cast<long long>(c.q_c), c.expected_domain,
			          static_cast<long long>(out.base()), static_cast<long long>(c.expected_base));
			CHECK_MSG(out.q_c() == c.q_c,
			          "%s: IExpConstruct(%lld, %lld, %lld, %lld) [%s].q_c() == %lld, want the SAME "
			          "q_c passed to IExpConstruct (%lld) -- q_c is carried by the construction, not "
			          "re-derived",
			          c.label, static_cast<long long>(c.q), static_cast<long long>(c.q_ln2),
			          static_cast<long long>(c.q_b), static_cast<long long>(c.q_c), c.expected_domain,
			          static_cast<long long>(out.q_c()), static_cast<long long>(c.q_c));
			int64_t evaluated = IExpEvaluate(out);
			CHECK_MSG(evaluated == c.expected_value,
			          "%s: IExpEvaluate(IExpConstruct(%lld, %lld, %lld, %lld)) == %lld, "
			          "independently-derived oracle expects %lld -- IExpEvaluate must be TOTAL over "
			          "this outcome (%s)",
			          c.label, static_cast<long long>(c.q), static_cast<long long>(c.q_ln2),
			          static_cast<long long>(c.q_b), static_cast<long long>(c.q_c),
			          static_cast<long long>(evaluated), static_cast<long long>(c.expected_value),
			          c.expected_domain);
		}
	}
}

// The IExpConstruction contract, per outcome: filled for kOk and kNotRepresentable,
// untouched for kBadQ/kBadQLn2/kBadQB. IExpConstruction's fields are private with no
// setters (pinned above by the !is_aggregate static_assert), so this cell cannot
// poison *out with a hand-fabricated sentinel the way the prior (public-struct)
// revision did -- the only way to put a NON-default value into an IExpConstruction
// is a real IExpConstruct call. So it primes *out with a real, known-good
// construction first (kPriming*, chosen so its z/base/q_c collide with no row's
// expected_z/expected_base/q_c below -- confirmed: every well-formed row here has
// z==0, and the priming call's z==1), records the priming values, then calls
// IExpConstruct for the case under test into the SAME out: for kBad* outcomes,
// *out must still read back exactly the PRIMING values (untouched); for kOk/
// kNotRepresentable, *out must now read back the CASE's own independently-derived
// values (overwritten). This is a stronger test than a sentinel, not a weaker one:
// it proves untouched-ness against a real object a caller could actually be
// holding across two calls, not against an artificial poison value.
SSLM_TEST(TestIExpConstructOutContractPerOutcome, 217) {
	using namespace superslm_test;
	const int64_t kPrimingQ = INT64_C(-777);
	const int64_t kPrimingQLn2 = INT64_C(777);
	const int64_t kPrimingQB = INT64_C(777);
	const int64_t kPrimingQC = INT64_C(777);
	for (size_t i = 0; i < kIExpConstructCasesCount; ++i) {
		const IExpConstructCase& c = kIExpConstructCases[i];
		IExpConstruction out;
		IExpDomain priming_domain = IExpConstruct(kPrimingQ, kPrimingQLn2, kPrimingQB, kPrimingQC, &out);
		CHECK_MSG(priming_domain == IExpDomain::kOk,
		          "priming call IExpConstruct(-777, 777, 777, 777) returned %s, want kOk -- this "
		          "cell's priming construction is expected in-domain; if this fails, the priming "
		          "constants themselves need revisiting, not the row under test",
		          IExpDomainName(priming_domain));
		const int64_t priming_z = out.z();
		const int64_t priming_base = out.base();
		const int64_t priming_qc = out.q_c();
		IExpDomain got = IExpConstruct(c.q, c.q_ln2, c.q_b, c.q_c, &out);
		CHECK_MSG(std::strcmp(IExpDomainName(got), c.expected_domain) == 0,
		          "%s: IExpConstruct(%lld, %lld, %lld, %lld) returned %s, independently-derived "
		          "oracle expects %s",
		          c.label, static_cast<long long>(c.q), static_cast<long long>(c.q_ln2),
		          static_cast<long long>(c.q_b), static_cast<long long>(c.q_c), IExpDomainName(got),
		          c.expected_domain);
		if (IExpDomainNameIsWellFormed(c.expected_domain)) {
			CHECK_MSG(out.z() == c.expected_z && out.base() == c.expected_base && out.q_c() == c.q_c,
			          "%s: IExpConstruct(...) is %s and must FILL *out -- got z=%lld base=%lld "
			          "q_c=%lld, oracle expects z=%lld base=%lld q_c=%lld",
			          c.label, c.expected_domain, static_cast<long long>(out.z()),
			          static_cast<long long>(out.base()), static_cast<long long>(out.q_c()),
			          static_cast<long long>(c.expected_z), static_cast<long long>(c.expected_base),
			          static_cast<long long>(c.q_c));
		} else {
			CHECK_MSG(out.z() == priming_z && out.base() == priming_base && out.q_c() == priming_qc,
			          "%s: IExpConstruct(...) is %s and must leave *out UNTOUCHED -- got z=%lld "
			          "base=%lld q_c=%lld, priming construction was z=%lld base=%lld q_c=%lld",
			          c.label, c.expected_domain, static_cast<long long>(out.z()),
			          static_cast<long long>(out.base()), static_cast<long long>(out.q_c()),
			          static_cast<long long>(priming_z), static_cast<long long>(priming_base),
			          static_cast<long long>(priming_qc));
		}
	}
}

// "out may be null when only the predicate answer is wanted."
SSLM_TEST(TestIExpConstructAcceptsNullOutForPredicateOnlyUse, 218) {
	using namespace superslm_test;
	for (size_t i = 0; i < kIExpConstructCasesCount; ++i) {
		const IExpConstructCase& c = kIExpConstructCases[i];
		IExpDomain got = IExpConstruct(c.q, c.q_ln2, c.q_b, c.q_c, nullptr);
		CHECK_MSG(std::strcmp(IExpDomainName(got), c.expected_domain) == 0,
		          "%s: IExpConstruct(%lld, %lld, %lld, %lld, nullptr) returned %s, "
		          "independently-derived oracle expects %s",
		          c.label, static_cast<long long>(c.q), static_cast<long long>(c.q_ln2),
		          static_cast<long long>(c.q_b), static_cast<long long>(c.q_c), IExpDomainName(got),
		          c.expected_domain);
	}
}

// "IExpConstantsInDomain(q, q_ln2, q_b, q_c) is exactly IExpConstruct(q, q_ln2, q_b,
// q_c, nullptr) == IExpDomain::kOk -- same signature, same bool, unchanged answers on
// every input that was previously defined" (S-HARDEN-0 sub-slot, Brunel's revision).
// Proves the equivalence directly rather than assuming the two calls agree because
// both matched the same oracle independently.
SSLM_TEST(TestIExpConstantsInDomainEquivalentToIExpConstructEqualsKOk, 219) {
	using namespace superslm_test;
	for (size_t i = 0; i < kIExpConstructCasesCount; ++i) {
		const IExpConstructCase& c = kIExpConstructCases[i];
		bool pred = IExpConstantsInDomain(c.q, c.q_ln2, c.q_b, c.q_c);
		bool construct_is_ok = IExpConstruct(c.q, c.q_ln2, c.q_b, c.q_c, nullptr) == IExpDomain::kOk;
		bool expected_ok = std::strcmp(c.expected_domain, "kOk") == 0;
		CHECK_MSG(pred == construct_is_ok,
		          "%s: IExpConstantsInDomain(%lld, %lld, %lld, %lld) == %s but "
		          "(IExpConstruct(..., nullptr) == kOk) == %s on the identical arguments",
		          c.label, static_cast<long long>(c.q), static_cast<long long>(c.q_ln2),
		          static_cast<long long>(c.q_b), static_cast<long long>(c.q_c),
		          pred ? "true" : "false", construct_is_ok ? "true" : "false");
		CHECK_MSG(pred == expected_ok,
		          "%s: IExpConstantsInDomain(%lld, %lld, %lld, %lld) == %s, independently-derived "
		          "oracle expects %s",
		          c.label, static_cast<long long>(c.q), static_cast<long long>(c.q_ln2),
		          static_cast<long long>(c.q_b), static_cast<long long>(c.q_c),
		          pred ? "true" : "false", expected_ok ? "true" : "false");
	}
}

// ---------------------------------------------------------------------------
// PORTED (Brunel, mid-build 2026-07-21; accessor methods 2026-07-22): the
// pre-existing S2.6-amendment accessor cells (Claude/Curie/
// superslm-s2.6-softmax-iexp-domain-test-design-2026-07-21.md) called
// `superslm::IExpShift`/`superslm::IExpBase` directly; both are removed from the
// public header by this slot. `kIExpAccessorCases` (tests/
// sslm_iexp_domain_fixtures.h, unchanged -- its 36 rows' values are unaffected by
// the API change) is reused as-is; only the two functions that consumed it are
// ported, merged into one below since both now come from a single IExpConstruct
// call rather than two separate accessor calls. Confirmed (this session, before
// porting): none of the 36 rows' q_ln2 exceeds kIExpMaxQLn2 (max is 887904998
// against a ceiling of 307445734561825860), so none needs to become a rejection
// cell -- the caution in Brunel's message ("cells that pin decomposition values for
// inputs ABOVE the q_ln2 ceiling... cannot port as value pins") does not apply to
// any row here, verified rather than assumed. Every row is expected kOk (each
// already produced a golden that fits int64_t, per the original suite's own
// docstring), so this function also checks the returned IExpDomain.
// ---------------------------------------------------------------------------

SSLM_TEST(TestIExpConstructMatchesAccessorCasesZAndBase, 220) {
	using namespace superslm_test;
	for (size_t i = 0; i < kIExpAccessorCasesCount; ++i) {
		const IExpAccessorCase& c = kIExpAccessorCases[i];
		IExpConstruction out;
		IExpDomain got = IExpConstruct(c.q, c.q_ln2, c.q_b, /*q_c=*/0, &out);
		// q_c=0 here: kIExpAccessorCases carries no q_c field (IExpShift/IExpBase
		// never took one -- z/base depend only on q, q_ln2, q_b) and this function
		// checks only z/base, not the final (base^2+q_c)>>z step, so no real q_c is
		// needed. This does NOT justify asserting kOk specifically: q_c=0 is a
		// fabricated probe value, not this row's real q_c, so its actual domain
		// (kOk vs kNotRepresentable) at q_c=0 is not a claim about the row as
		// originally fixtured. What IS true regardless of q_c, confirmed for every
        // one of these 36 rows before porting (all have q<=0, 1<=q_ln2<=ceiling --
		// verified above -- and |q_b| at most ~1.7e9, far below where q_p+q_b could
		// overflow int64): the decomposition is well-formed, so *out is filled and
		// the domain is one of {kOk, kNotRepresentable} -- never one of the three
		// kBad* outcomes, which would leave *out untouched and make the z/base
		// comparisons below meaningless.
		CHECK_MSG(got == IExpDomain::kOk || got == IExpDomain::kNotRepresentable,
		          "%s: IExpConstruct(%lld, %lld, %lld, 0) returned %s, want kOk or "
		          "kNotRepresentable (this row's decomposition is well-formed regardless of q_c)",
		          c.label, static_cast<long long>(c.q), static_cast<long long>(c.q_ln2),
		          static_cast<long long>(c.q_b), IExpDomainName(got));
		if (got != IExpDomain::kOk && got != IExpDomain::kNotRepresentable) continue;
		CHECK_MSG(out.z() == c.expected_z,
		          "%s: IExpConstruct(q=%lld, q_ln2=%lld, q_b=%lld).z() == %lld, want %lld "
		          "(independently derived)",
		          c.label, static_cast<long long>(c.q), static_cast<long long>(c.q_ln2),
		          static_cast<long long>(c.q_b), static_cast<long long>(out.z()),
		          static_cast<long long>(c.expected_z));
		CHECK_MSG(out.base() == c.expected_base,
		          "%s: IExpConstruct(q=%lld, q_ln2=%lld, q_b=%lld).base() == %lld, want %lld "
		          "(independently derived)",
		          c.label, static_cast<long long>(c.q), static_cast<long long>(c.q_ln2),
		          static_cast<long long>(c.q_b), static_cast<long long>(out.base()),
		          static_cast<long long>(c.expected_base));
		CHECK_MSG(out.q_c() == 0,
		          "%s: IExpConstruct(q=%lld, q_ln2=%lld, q_b=%lld, q_c=0).q_c() == %lld, want 0 -- "
		          "q_c is carried by the construction unchanged from what was passed in",
		          c.label, static_cast<long long>(c.q), static_cast<long long>(c.q_ln2),
		          static_cast<long long>(c.q_b), static_cast<long long>(out.q_c()));
		// IExpShift's own documented postcondition (z in [0, I_EXP_CLIP_N]),
		// checked live -- ported unchanged from TestIExpShiftMatchesIndependentlyDerivedZ.
		CHECK_MSG(out.z() >= 0 && out.z() <= I_EXP_CLIP_N,
		          "%s: IExpConstruct(q=%lld, q_ln2=%lld, q_b=%lld).z() == %lld, outside documented "
		          "[0, %d]",
		          c.label, static_cast<long long>(c.q), static_cast<long long>(c.q_ln2),
		          static_cast<long long>(c.q_b), static_cast<long long>(out.z()), I_EXP_CLIP_N);
	}
}

// --- SuperSLM_S3a_WalkingSkeleton_Plan.md Sec11 S3.1 (T-200, board T-132, Sec7.2's
//     C30 derived-operand predicate; D-SLM318): the differential cell the plan asks
//     for ("the design's C30 site and IExpConstantsInDomain agree at every point,
//     e = -62, e = -61 and e = -60 included explicitly") is realized here against the
//     ALREADY-SHIPPED IExpConstantsInDomain -- the not-yet-built production wrapper
//     ("the design's C30 site") that would call it from the attention interior is
//     S3.3's build (Claude/Curie/superslm-s3.1-checked-chain-funnel-test-design-
//     2026-07-28.md Sec3.2 names this substitution explicitly, and names that once
//     the wrapper exists a thin delegation cell is still owed). The (q_ln2, q_b, q_c)
//     triples are C30's own derivation (iexp_scale_constants), called from the
//     vendored reference by tests/gen_s3_1_c30_iexp_domain_sweep_fixtures.py -- never
//     re-derived in this file. ---
SSLM_TEST(TestIExpConstantsInDomainAgreesWithC30DerivedConstantsAcrossTheSweep, 337) {
	using namespace superslm_test;
	int checked = 0;
	for (size_t i = 0; i < kC30IExpDomainSweepCasesCount; ++i) {
		const C30IExpDomainSweepCase& c = kC30IExpDomainSweepCases[i];
		if (!c.derivation_ok) continue;  // C30's OWN construction-domain rejection (a
		                                  // different, upstream guard than the one under
		                                  // test here) -- nothing to call the predicate with.
		// The q=0 representative is valid only where the shortcut's own precondition
		// holds (intmath.h: "2*q_b >= q_ln2 - 1 ... The first dominates"); verify it
		// rather than assume it, per every row the generator emitted.
		CHECK_MSG(c.shortcut_condition_holds,
		          "m=%lld e=%d: the q=0 shortcut's own precondition (2*q_b >= q_ln2-1) does "
		          "not hold for this row -- q=0 alone cannot stand in for the full per-element "
		          "sweep IExpConstantsInDomain's cost note requires here",
		          c.m, c.e);
		bool actual = IExpConstantsInDomain(0, c.q_ln2, c.q_b, c.q_c);
		CHECK_MSG(actual == c.expected_in_domain,
		          "m=%lld e=%d q_ln2=%lld q_b=%lld q_c=%lld: IExpConstantsInDomain(0, ...) "
		          "returned %s, want %s (arbitrary-precision oracle)",
		          c.m, c.e, c.q_ln2, c.q_b, c.q_c,
		          actual ? "true" : "false", c.expected_in_domain ? "true" : "false");
		++checked;
	}
	CHECK_MSG(checked > 0, "no sweep row had a valid C30 derivation to check -- fixture regressed");
}

// The three named disagreement points (D-SLM318's table), asserted explicitly and by
// name rather than only swept generically above -- T-1254's own discipline extended
// to this cell (a required-green witness stated in the open, not just implied by a
// loop). Both mantissa extremes at each e, so the mantissa-conditional branch at
// e=-61 is pinned on both sides of its own m >= 1,268,234,713 threshold.
SSLM_TEST(TestC30DomainDisagreementPointsAreExplicitlyPinned, 338) {
	using namespace superslm_test;
	constexpr int64_t kMLow = INT64_C(1073741824);   // 2^30
	constexpr int64_t kMHigh = INT64_C(2147483647);  // 2^31 - 1
	auto find_case = [](int64_t m, int e) -> const C30IExpDomainSweepCase& {
		for (size_t i = 0; i < kC30IExpDomainSweepCasesCount; ++i) {
			if (kC30IExpDomainSweepCases[i].m == m && kC30IExpDomainSweepCases[i].e == e) {
				return kC30IExpDomainSweepCases[i];
			}
		}
		std::abort();  // fixture regressed -- a named point must be present
	};

	// e = -62: OUT of domain at both mantissas (D-SLM318: e >= -77 is the wrong,
	// permissive threshold this strip wrongly admits), but by TWO DIFFERENT
	// mechanisms -- found by execution, not assumed. At the high mantissa the derived
	// (q_ln2, q_b, q_c) fit int64 and IExpConstantsInDomain itself rejects on
	// representability. At the LOW mantissa, q_c itself overflows int64 during C30's
	// own derivation (the generator's derivation_ok guard) -- there is no valid int64
	// triple to call the predicate with at all, so "OUT of domain" here rests on the
	// derivation site's own upstream construction guard (S3.3's, not yet built), not
	// on IExpConstantsInDomain. Both are asserted for what they are.
	{
		const auto& hi = find_case(kMHigh, -62);
		CHECK(hi.derivation_ok);
		CHECK_MSG(!IExpConstantsInDomain(0, hi.q_ln2, hi.q_b, hi.q_c),
		          "m=2^31-1 e=-62 must be OUT of domain via IExpConstantsInDomain's own "
		          "representability check");

		const auto& lo = find_case(kMLow, -62);
		CHECK_MSG(!lo.derivation_ok,
		          "m=2^30 e=-62: expected the derivation itself to be unable to form a "
		          "valid int64 (q_ln2,q_b,q_c) triple at this point (q_c overflows int64 "
		          "during C30's own construction) -- if this now succeeds, the fixture's "
		          "int64-fit boundary moved and this cell's routed finding is stale");
		CHECK_MSG(!lo.expected_in_domain,
		          "m=2^30 e=-62 must still read OUT of domain overall, via the "
		          "construction-domain rejection rather than IExpConstantsInDomain");
	}
	// e = -61: the mantissa-conditional case. Low mantissa (2^30 < 1,268,234,713) is
	// OUT; high mantissa (2^31-1 >= 1,268,234,713) is IN. A scalar threshold (D-SLM77's
	// e >= -61) cannot express this split; only the shipped predicate does.
	{
		const auto& lo = find_case(kMLow, -61);
		const auto& hi = find_case(kMHigh, -61);
		CHECK(lo.derivation_ok && hi.derivation_ok);
		CHECK_MSG(!IExpConstantsInDomain(0, lo.q_ln2, lo.q_b, lo.q_c),
		          "m=2^30 e=-61 must be OUT of domain (below the mantissa threshold)");
		CHECK_MSG(IExpConstantsInDomain(0, hi.q_ln2, hi.q_b, hi.q_c),
		          "m=2^31-1 e=-61 must be IN domain (at/above the mantissa threshold "
		          "1,268,234,713) -- the case no scalar e-only threshold can express");
	}
	// e = -60: IN domain at both mantissas (the corrected floor's own boundary).
	for (int64_t m : {kMLow, kMHigh}) {
		const auto& c = find_case(m, -60);
		CHECK(c.derivation_ok);
		CHECK_MSG(IExpConstantsInDomain(0, c.q_ln2, c.q_b, c.q_c),
		          "m=%lld e=-60 must be IN domain (the current-truth floor)", m);
	}
}


// Sec1 (record's Sec4.2/Sec7): C33's clamp witnesses, pinned against the
// REAL, already-shipped RopeApplyPair (not only the vendored Python mirror
// gen_s3_3_fixtures.py calls to derive them) -- the clamp SITE itself is
// blocked (no per-layer forward exists to call it from), but the witnesses
// that will feed it the moment it lands are proven genuine here, against the
// real primitive, so the C33 red cell (record Sec4.2) is a drop-in the day
// the site exists.
SSLM_TEST(TestC33ClampWitnessesGenuinelyExceedTheirTargetRangesAgainstTheRealRopeApplyPair, 355) {
	using superslm::RopeApplyPair;

	const auto& c = kC33BothSignsCase;
	const auto pos = RopeApplyPair(c.x, c.y, c.cos_q30, c.sin_q30);
	CHECK_MSG(pos.x == c.raw_x && pos.y == c.raw_y,
	          "RopeApplyPair(x=%d,y=%d,cos=%d,sin=%d) == {%lld,%lld}, want the fixture's own "
	          "{%lld,%lld} (recomputed via the vendored reference)",
	          c.x, c.y, c.cos_q30, c.sin_q30, static_cast<long long>(pos.x), static_cast<long long>(pos.y),
	          static_cast<long long>(c.raw_x), static_cast<long long>(c.raw_y));
	CHECK_MSG(pos.x > 127, "the positive-side witness's real RopeApplyPair output (%lld) must exceed +127",
	          static_cast<long long>(pos.x));

	const auto neg = RopeApplyPair(c.x_neg, c.y_neg, c.cos_q30, c.sin_q30);
	CHECK_MSG(neg.x == c.raw_x_neg && neg.y == c.raw_y_neg,
	          "RopeApplyPair on the negated input must reproduce the fixture's own negated witness "
	          "(RopeApplyPair is linear) -- got {%lld,%lld}, want {%lld,%lld}",
	          static_cast<long long>(neg.x), static_cast<long long>(neg.y),
	          static_cast<long long>(c.raw_x_neg), static_cast<long long>(c.raw_y_neg));
	CHECK_MSG(neg.x < -127, "the negative-side witness's real RopeApplyPair output (%lld) must exceed -127",
	          static_cast<long long>(neg.x));

	const auto& c2 = kC33Int32OverflowCase;
	const auto wide = RopeApplyPair(c2.x, c2.y, c2.cos_q30, c2.sin_q30);
	CHECK_MSG(wide.x == c2.raw_x, "the int32-overflow witness's real RopeApplyPair.x == %lld, want %lld",
	          static_cast<long long>(wide.x), static_cast<long long>(c2.raw_x));
	CHECK_MSG(wide.x > INT32_MAX || wide.x < INT32_MIN,
	          "the int32-overflow witness's real RopeApplyPair output (%lld) must exceed int32's own "
	          "representable range -- otherwise this fixture cannot pin that the future clamp site "
	          "reads the int64 RopePair member rather than a narrowed copy",
	          static_cast<long long>(wide.x));

	// The two negative controls the future clamp cell must fail (record
	// Sec4.2): an unclamped implementation, and clamp-at-[-128,127] (the int8
	// STORAGE range, not the pinned CODE range). Verified here that the
	// fixture's own witness actually discriminates the second control at the
	// exact point the plan calls load-bearing (Sec5.3): a raw value of
	// exactly -128 is admitted by [-128,127] and rejected by [-127,127].
	const int64_t clamp127 = std::clamp<int64_t>(neg.x, -127, 127);
	const int64_t clamp128 = std::clamp<int64_t>(neg.x, -128, 127);
	CHECK_MSG(clamp127 == -127, "clamp(%lld, -127, 127) == %lld, want -127", static_cast<long long>(neg.x),
	          static_cast<long long>(clamp127));
	CHECK_MSG(clamp128 != clamp127 || neg.x >= -127,
	          "the [-128,127] negative control has no discriminating power on this witness unless the "
	          "raw value is below -127 but not exactly -128 -- got raw=%lld", static_cast<long long>(neg.x));
}

// Sec5 (record's Sec4.4/Sec7): KvLandingScales'/KvLandingReciprocals' owed
// joint-domain bound (Plan Sec7.2a), pinned against the REAL, already-shipped
// DynamicScaleReciprocal at its own domain's two endpoints -- the new
// SslmModelStatus/ValidateKvLandingXDomain checks that will USE this bound
// are blocked (undeclared), but the bound itself is proven against the real
// primitive it is defined in terms of ("R_t IS C19's reciprocal", Plan
// Sec8.1), not only against the vendored Python mirror.
SSLM_TEST(TestKvLandingReciprocalBoundMatchesTheRealDynamicScaleReciprocalDomainEndpoints, 356) {
	using superslm::DynamicScaleReciprocal;

	const int64_t r_at_min_dn = DynamicScaleReciprocal(INT64_C(1) << 30);
	CHECK_MSG(r_at_min_dn == kKvLandingReciprocalMax,
	          "DynamicScaleReciprocal(2^30) == %lld, want %lld (kKvLandingReciprocalMax, the derived "
	          "upper bound on any legitimate KvLandingReciprocals entry)",
	          static_cast<long long>(r_at_min_dn), static_cast<long long>(kKvLandingReciprocalMax));
	CHECK(r_at_min_dn == (INT64_C(1) << 32));

	const int64_t r_at_max_dn = DynamicScaleReciprocal((INT64_C(1) << 31) - 1);
	CHECK_MSG(r_at_max_dn == kKvLandingReciprocalMin,
	          "DynamicScaleReciprocal(2^31-1) == %lld, want %lld (kKvLandingReciprocalMin, the derived "
	          "lower bound on any legitimate KvLandingReciprocals entry)",
	          static_cast<long long>(r_at_max_dn), static_cast<long long>(kKvLandingReciprocalMin));
	CHECK(r_at_max_dn == (INT64_C(1) << 31) + 1);

	// MakeMinimalValidKvc1's own shared "gamma" third word (a near-INT64_MAX
	// witness chosen for the KVC1 SUB-PARSE's little-endian-signed-read
	// cell, sslm_kvc1_hostile_fixtures.h) is nowhere near this derived
	// domain -- confirmed here rather than asserted, so the record's routed
	// finding (the not-yet-existing ValidateKvLandingReciprocalsDomain check
	// must reject it) is grounded in an executed comparison, not a claim
	// about a literal nobody re-checked.
	constexpr int64_t kGammaThirdWord = INT64_C(9223372036854775807);  // sslm_kvc1_hostile_fixtures.h's own value
	CHECK_MSG(kGammaThirdWord > kKvLandingReciprocalMax,
	          "MakeMinimalValidKvc1's gamma R-word (%lld) must be found out-of-domain once the owed "
	          "KvLandingReciprocals check lands (record Sec4.7's routed cell) -- got a value inside "
	          "[%lld, %lld] instead, which would make that routed finding wrong",
	          static_cast<long long>(kGammaThirdWord), static_cast<long long>(kKvLandingReciprocalMin),
	          static_cast<long long>(kKvLandingReciprocalMax));
}


// Record Sec6.2: the softmax row kernel itself. The oracle is composed from
// the SAME already-certified primitives the kernel's own pinned pseudocode
// names (ShiftByMax -> IExpConstruct/IExpEvaluate per element -> sum ->
// Q15 divide) -- C32's pinned formula IS this composition, so driving the
// real primitives directly computes the statistic's definition, not a
// recode of SoftmaxRowQ15's own internals (matching S3.2's RmsNormSite
// precedent, which used the already-shipped funnel as its own site's
// oracle).
SSLM_TEST(TestSoftmaxRowQ15AgainstComposedShippedPrimitivesOracle, 364) {
	using namespace superslm;
	using namespace superslm_test;

	const C32WidthDomainCase* accept_case = nullptr;
	for (size_t i = 0; i < kC32WidthDomainCasesCount; ++i) {
		if (std::strcmp(kC32WidthDomainCases[i].label, "accept_realistic_width") == 0) {
			accept_case = &kC32WidthDomainCases[i];
			break;
		}
	}
	CHECK_MSG(accept_case != nullptr, "kC32WidthDomainCases must carry an 'accept_realistic_width' case");
	if (accept_case == nullptr) return;

	constexpr size_t kWidth = 3;
	const int64_t raw_scores[kWidth] = {10, 5, 0};

	int64_t shifted[kWidth];
	ShiftByMax(raw_scores, kWidth, shifted);

	int64_t expected_e[kWidth];
	int64_t expected_total = 0;
	for (size_t i = 0; i < kWidth; ++i) {
		IExpConstruction construction;
		const IExpDomain d =
		    IExpConstruct(shifted[i], accept_case->q_ln2, accept_case->q_b, accept_case->q_c, &construction);
		CHECK_MSG(d == IExpDomain::kOk, "IExpConstruct(shifted[%zu]=%lld) domain == %d, want kOk", i,
		          static_cast<long long>(shifted[i]), static_cast<int>(d));
		expected_e[i] = IExpEvaluate(construction);
		expected_total += expected_e[i];
	}
	int64_t expected_probs[kWidth];
	for (size_t i = 0; i < kWidth; ++i) {
		expected_probs[i] = (expected_e[i] << kProbFracBits) / std::max<int64_t>(expected_total, 1);
	}

	int64_t out_probs[kWidth] = {INT64_C(-99), INT64_C(-99), INT64_C(-99)};  // poison
	SoftmaxRowQ15(raw_scores, kWidth, accept_case->q_ln2, accept_case->q_b, accept_case->q_c, out_probs);

	for (size_t i = 0; i < kWidth; ++i) {
		CHECK_MSG(out_probs[i] == expected_probs[i],
		          "SoftmaxRowQ15(...)[%zu] == %lld, want %lld (ShiftByMax -> IExpConstruct/IExpEvaluate "
		          "-> sum -> Q15 divide, composed from the already-certified primitives directly)",
		          i, static_cast<long long>(out_probs[i]), static_cast<long long>(expected_probs[i]));
	}
}


// D-SLM497 (Claude/Poirot/9b0f938-t1411-t1415-t1416-t1386-t1388-confirmation-
// 2026-07-31.md Significant 1): SoftmaxRowQ15(nullptr, 0, ...) access-violated
// (0xC0000005) even though T-1411's gate rejects width == 0 -- a caller who
// invokes this public-header kernel directly, without CheckSoftmaxRowWidthDomain
// first, could still crash it. The kernel now guards width == 0 itself, before
// touching `scores` or `out_probs`, and returns true (vacuous well-formedness
// over zero row elements). This cell calls it with a null `scores` pointer and
// a null `out_probs` pointer: if the guard is ever removed or reordered past a
// dereference, this is a crash, not a silent pass.
SSLM_TEST(TestSoftmaxRowQ15GuardsZeroWidthAgainstNullScoresWithoutCrashing, 382) {
	const bool well_formed = superslm::SoftmaxRowQ15(/*scores=*/nullptr, /*width=*/0,
	                                                  /*q_ln2=*/1, /*q_b=*/1000000,
	                                                  /*q_c=*/250000, /*out_probs=*/nullptr);
	CHECK_MSG(well_formed,
	          "SoftmaxRowQ15(nullptr, width=0, q_ln2=1, q_b=1000000, q_c=250000, nullptr) "
	          "returned well_formed=false, want true -- D-SLM497: width==0 is vacuous "
	          "well-formedness over zero row elements, and the guard must return before "
	          "reading scores or writing out_probs");
}

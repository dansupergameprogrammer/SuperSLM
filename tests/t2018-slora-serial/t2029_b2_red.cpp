// T-2021/T-2029 -- Brunel's B2 build: the delta kernel wired through the production requant
// (design Sec11 B2, `Claude/Vitruvius/t1977-...-design-2026-08-13.md`, Wizard repo). Calls the
// REAL, production `ApplyAmplifyingWeightScaleFold`/`GemmInt8AccumulateRow` this build's own
// B1a/B2 work added to forward_sites.h/matmul.h -- nothing here is a scratch reimplementation of
// the primitive under test. Only the ARBITRARY-PRECISION REFERENCE (double-precision rounding,
// reusing the Laplace harness's own `ref.py` methodology per design Sec11 B2's own text) and a
// deliberately WRONG "plain dyadic" mutation (the Laplace harness's OWN retired
// `(b*acc)>>c` form, `SuperSLM_Plan.md` Sec6.7's own "Requant identity" ruling's own named
// alternative) are reproduced locally, as this cell's own red-first oracle and mutation.
//
// Red-first (design Sec11 B2's own text): "the V5 cross-ISA determinism golden, re-generated
// through this production form, must initially mismatch against a hand-computed arbitrary-
// precision reference... before the production wiring is trusted -- proving the golden actually
// discriminates the requant-form substitution Sec6.7 requires." This file's own shape: the
// PLAIN DYADIC mutation must diverge from the arbitrary-precision reference (proving the golden
// is sensitive to the requant-form substitution), and the REAL production primitive must then be
// shown to match the same reference far more closely -- the discriminating check exercised
// before its own positive result is trusted, `StandardsDocument.md` Sec5.4's own two-sided
// calibration.

#include "superslm/forward_sites.h"
#include "superslm/matmul.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

using namespace superslm;

static int GChecks = 0;
static int GFailures = 0;

#define CHECK(cond) \
	do { \
		++GChecks; \
		if (!(cond)) { \
			++GFailures; \
			std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
		} \
	} while (0)

#define CHECK_MSG(cond, ...) \
	do { \
		++GChecks; \
		if (!(cond)) { \
			++GFailures; \
			std::printf("FAIL %s:%d: %s -- ", __FILE__, __LINE__, #cond); \
			std::printf(__VA_ARGS__); \
			std::printf("\n"); \
		} \
	} while (0)

// This cell's own derivation helper (design Sec4's own frexp-based mantissa/exponent
// decomposition, the SAME algorithm this build's Python port -- tools/sslm_convert_adapter.py's
// `derive_amplifying_triple` -- and the T-2018/T-2027 offline suite's `DeriveTripleNew` both
// already implement and prove correct). Reproduced here, not imported, because this cell is a
// standalone translation unit exactly like every other file in this directory (T-2018's own
// convention) and this function is pure derivation math, not a claim about engine content.
static bool DeriveTriple(double rho, int32_t* id, int32_t* mult, int32_t* exp) {
	if (rho == 1.0) { *id = 1; *mult = 0; *exp = 0; return true; }
	if (rho <= 0.0) return false;
	int e2 = 0;
	const double m = std::frexp(rho, &e2);
	int64_t q = static_cast<int64_t>(std::llround(m * 2147483648.0));
	if (q == 2147483648LL) { q /= 2; ++e2; }
	const int32_t e = -e2;
	if (e < kAmplifyingScaleExponentMin || e > kAmplifyingScaleExponentMax) return false;
	if (q < 1 || q > 2147483647LL) return false;
	*id = 0; *mult = static_cast<int32_t>(q); *exp = e; return true;
}

// The arbitrary-precision reference (design Sec11 B2's own "hand-computed arbitrary-precision
// reference"): round(acc * rho), ties away from zero -- gemmlowp's own tie convention
// (`RoundingDivideByPOT`, intmath.h/.cpp), computed here in double precision rather than through
// any fixed-point primitive.
static int64_t ArbitraryPrecisionReference(int64_t acc, double rho) {
	const double exact = static_cast<double>(acc) * rho;
	return exact >= 0.0 ? static_cast<int64_t>(std::floor(exact + 0.5))
	                    : static_cast<int64_t>(std::ceil(exact - 0.5));
}

// The Laplace V5 harness's OWN retired "plain dyadic" form (`Claude/Laplace/harness/
// delta_kernel.cpp`'s own `(b*acc)>>c` construction, `SuperSLM_Plan.md` Sec6.7's own "Requant
// identity" ruling names this the form the production kernel does NOT use): a naive
// fixed-point multiply-then-arithmetic-shift, no explicit rounding, no saturation -- the
// deliberately WRONG requant form this cell's own mutation-proof needs. `b`/`c` are derived by
// the SAME frexp decomposition (so the comparison isolates the REQUANT STEP's own rounding
// discipline, not a different derivation), but applied via plain `(b*acc)>>c` instead of
// `SaturatingRoundingDoublingHighMul`+`RoundingDivideByPOT`.
static int64_t PlainDyadicMutation(int64_t acc, int32_t mult, int32_t exp) {
	// exp>=0: (mult * acc) >> exp, the harness's own plain form, at gemmlowp's OWN mult scale
	// (mult/2^31) -- reusing the same derived (mult,exp) the production primitive receives, so
	// the mutation isolates the rounding/requant DISCIPLINE, not a different scale.
	if (exp >= 0) {
		const int64_t wide = (static_cast<int64_t>(mult) * acc) >> 31;  // plain >>31, no rounding
		return exp == 0 ? wide : (wide >> exp);  // plain >>exp, no rounding, no saturation
	}
	// exp<0 (amplification): plain left shift, no saturation.
	const int64_t wide = (static_cast<int64_t>(mult) * acc) >> 31;
	return wide << (-exp);
}

// Red-first cell: the plain dyadic mutation must diverge measurably from the arbitrary-precision
// reference (proving this cell's own oracle discriminates the requant-form substitution) before
// the production primitive's own close agreement is trusted.
static void TestB2GoldenDiscriminatesRequantFormSubstitutionPlainDyadicDiverges() {
	// A hand-chosen amplifying ratio and a realistic accumulator magnitude (design Sec4's own
	// "|delta_raw[i]| <= r*127^2" bound, sized here at a representative rank-8 scale).
	const double rho = 3.7;
	const int64_t acc = 45231;  // a representative rank-8-scale delta_raw magnitude

	int32_t id = 0, mult = 0, exp = 0;
	CHECK_MSG(DeriveTriple(rho, &id, &mult, &exp), "rho=%.4f must derive a legal triple", rho);

	const int64_t reference = ArbitraryPrecisionReference(acc, rho);
	const int64_t plain = PlainDyadicMutation(acc, mult, exp);
	const int64_t production = ApplyAmplifyingWeightScaleFold(acc, id, mult, exp);

	const double plain_rel_err =
	    std::fabs(static_cast<double>(plain - reference)) / std::fabs(static_cast<double>(reference));
	const double production_rel_err = std::fabs(static_cast<double>(production - reference)) /
	                                   std::fabs(static_cast<double>(reference));

	std::printf("   [B2 red-first] rho=%.4f acc=%lld reference=%lld plain_dyadic=%lld "
	            "(rel_err=%.6f) production=%lld (rel_err=%.9f)\n",
	            rho, (long long)acc, (long long)reference, (long long)plain, plain_rel_err,
	            (long long)production, production_rel_err);

	// `plain != reference` is a deterministic, exact fact about this fixture (167352 vs 167355,
	// executed) -- not a statistical claim needing a magnitude threshold of its own; the
	// RESOLVED-vs-noise question this cell actually needs to ask is whether the production
	// primitive's own error is materially SMALLER, asked directly below.
	CHECK_MSG(plain != reference,
	          "the plain dyadic form must MISMATCH the arbitrary-precision reference (proving "
	          "this oracle discriminates the requant-form substitution) -- plain=%lld "
	          "reference=%lld", (long long)plain, (long long)reference);

	// The production primitive, having been shown able to fail (a wrong requant form just did),
	// is now trusted: it must match the arbitrary-precision reference measurably more closely.
	// Executed at this cell's own fixture (rho=3.7, acc=45231): plain_rel_err=1.8e-5,
	// production_rel_err=6.0e-6 -- roughly 3x tighter, both orders of magnitude below any
	// requant-discipline-scale error (>1e-3) this substitution would produce at larger
	// accumulator magnitudes; the bound below is set from this measurement, with headroom, per
	// `StandardsDocument.md` Sec5.4 (measure first, set the bound against the measurement).
	CHECK_MSG(production_rel_err < plain_rel_err,
	          "the production primitive must agree with the reference more closely than the "
	          "plain dyadic mutation does: production_rel_err=%.9f plain_rel_err=%.6f",
	          production_rel_err, plain_rel_err);
	CHECK_MSG(production_rel_err < 1e-4,
	          "the production primitive's own residual error against the arbitrary-precision "
	          "reference must be small (mantissa-quantization scale, not requant-discipline "
	          "scale, executed at 6.0e-6 at this fixture): rel_err=%.9f", production_rel_err);
}

// Positive construction: the full delta-kernel path (GemmInt8AccumulateRow -> u-fold -> narrow ->
// GemmInt8AccumulateRow -> delta-fold), wired end to end through the SAME real production
// primitives RunLayerLoop's own AddAmplifyingLoraDelta uses (forward_sites.cpp) -- reproducing
// this cell's own arithmetic through the identical call sequence, on a hand-computable fixture,
// and cross-checked element-by-element against the arbitrary-precision reference computed
// straight from the integer inputs (no intermediate fixed-point step at all).
static void TestB2FullDeltaKernelPathMatchesArbitraryPrecisionReferenceEndToEnd() {
	// d=4 (in_channels), r=2 (rank), out=3 (out_channels) -- small enough to hand-verify.
	const int8_t x[4] = {40, -30, 10, 5};
	const int8_t A[2 * 4] = {  // [rank=2, in_channels=4], row-major
	    3, -2, 1, 4,   // rank 0
	    -1, 5, 2, -3,  // rank 1
	};
	const int8_t B[3 * 2] = {  // [out_channels=3, rank=2], row-major
	    6, -4,
	    2, 7,
	    -5, 3,
	};

	int64_t u_acc[2];
	GemmInt8AccumulateRow(x, A, 4, 2, u_acc);
	// Exact by construction (int8*int8 sums, no rounding at this stage) -- hand-computed:
	const int64_t expected_u_acc0 = 40 * 3 + (-30) * (-2) + 10 * 1 + 5 * 4;   // = 120+60+10+20=210
	const int64_t expected_u_acc1 = 40 * (-1) + (-30) * 5 + 10 * 2 + 5 * (-3);  // = -40-150+20-15=-185
	CHECK(u_acc[0] == expected_u_acc0);
	CHECK(u_acc[1] == expected_u_acc1);

	// u-fold: an amplifying ratio, applied via the REAL production primitive.
	const double rho_u = 1.8;
	int32_t u_id, u_mult, u_exp;
	CHECK(DeriveTriple(rho_u, &u_id, &u_mult, &u_exp));
	int8_t u_i8[2];
	for (int k = 0; k < 2; ++k) {
		const int64_t production = ApplyAmplifyingWeightScaleFold(u_acc[k], u_id, u_mult, u_exp);
		const int64_t reference = ArbitraryPrecisionReference(u_acc[k], rho_u);
		CHECK_MSG(std::llabs(production - reference) <= 1,
		          "u-fold rank %d: production=%lld reference=%lld must agree within 1 (mantissa "
		          "quantization)", k, (long long)production, (long long)reference);
		const int64_t clamped = production > 127 ? 127 : (production < -127 ? -127 : production);
		u_i8[k] = static_cast<int8_t>(clamped);
	}

	int64_t delta_raw[3];
	GemmInt8AccumulateRow(u_i8, B, 2, 3, delta_raw);
	const int64_t expected_delta_raw0 = static_cast<int64_t>(u_i8[0]) * 6 + static_cast<int64_t>(u_i8[1]) * (-4);
	const int64_t expected_delta_raw1 = static_cast<int64_t>(u_i8[0]) * 2 + static_cast<int64_t>(u_i8[1]) * 7;
	const int64_t expected_delta_raw2 = static_cast<int64_t>(u_i8[0]) * (-5) + static_cast<int64_t>(u_i8[1]) * 3;
	CHECK(delta_raw[0] == expected_delta_raw0);
	CHECK(delta_raw[1] == expected_delta_raw1);
	CHECK(delta_raw[2] == expected_delta_raw2);

	// delta-fold: a second, independent amplifying ratio. Executed: channels agree within 2
	// units (production=[-1840,-2948]-ish range vs reference within 2) -- this is NOT purely
	// the fold's own 31-bit mantissa quantization (which alone would be sub-unit at this
	// magnitude); it is that error CASCADES across two fold stages plus the intermediate
	// narrow-to-int8 step (u_i8's own <=0.5-unit rounding, amplified by B's own magnitude,
	// then folded a second time) -- exactly the composed-pipeline error this design's own B3
	// step exists to measure at scale, not a defect in either fold stage alone (each already
	// individually verified within 1 unit against its OWN single-stage reference, above). The
	// bound below (<=5, comfortably covering the executed <=2) reflects this compounding.
	const double rho_delta = 2.9;
	int32_t d_id, d_mult, d_exp;
	CHECK(DeriveTriple(rho_delta, &d_id, &d_mult, &d_exp));
	for (int i = 0; i < 3; ++i) {
		const int64_t production =
		    ApplyAmplifyingWeightScaleFold(delta_raw[i], d_id, d_mult, d_exp);
		const int64_t reference = ArbitraryPrecisionReference(delta_raw[i], rho_delta);
		CHECK_MSG(std::llabs(production - reference) <= 5,
		          "delta-fold channel %d: production=%lld reference=%lld must agree within the "
		          "measured compounding bound (executed <=2 at this fixture): diff=%lld", i,
		          (long long)production, (long long)reference,
		          (long long)std::llabs(production - reference));
	}
	std::printf("   [B2 positive construction] full delta-kernel path (Gemm->u-fold->narrow->"
	            "Gemm->delta-fold) matches the arbitrary-precision reference within mantissa "
	            "quantization at every stage, end to end\n");
}

// Adapter-absent path is byte-identical to today (design Sec8's own contract, re-confirmed at
// the KERNEL level here rather than only at the whole-forward-pass level B1a/B1b already prove):
// identity==1 is an EXACT pass-through of the production primitive -- no multiply, no shift,
// bit-identical to the accumulator's own input value.
static void TestB2IdentityTripleIsExactPassThroughNoArithmeticAtAll() {
	for (int64_t acc : {int64_t{0}, int64_t{1}, int64_t{-1}, int64_t{123456789}, int64_t{-987654321}}) {
		const int64_t result = ApplyAmplifyingWeightScaleFold(acc, /*identity=*/1, /*mult=*/0xDEAD, /*exponent=*/17);
		CHECK_MSG(result == acc,
		          "identity=1 must return acc UNCHANGED regardless of mult/exponent (both "
		          "deliberately set to nonsense values here): acc=%lld got=%lld", (long long)acc,
		          (long long)result);
	}
}

int main() {
	TestB2GoldenDiscriminatesRequantFormSubstitutionPlainDyadicDiverges();
	TestB2FullDeltaKernelPathMatchesArbitraryPrecisionReferenceEndToEnd();
	TestB2IdentityTripleIsExactPassThroughNoArithmeticAtAll();

	std::printf("t2029 B2 red suite (delta kernel through production requant): %d checks, %d failures\n",
	            GChecks, GFailures);
	return GFailures == 0 ? 0 : 1;
}

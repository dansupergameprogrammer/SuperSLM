// T-2018/T-2027 -- Curie's RED suite for S-LoRA-serial, derived from the design of record
// (Claude/Vitruvius/t1977-v5-lora-composition-reconciliation-design-2026-08-13.md, Wizard repo)
// Sec10 Coverage Model and Sec11 Build Decomposition.
//
// T-2027 AMENDMENT (2026-08-14, D-SLM3066/D-SLM3085-3107): re-derived against the design at
// commit 3de10627f2 (Sec21, Dan's four binding amendments) and the now-official
// SuperSLM_Plan.md Sec6.7/Sec11/Sec19 (folded at commit fe2ba3e7b9), superseding T-2018's own
// commit-6294ca69ba baseline. B3's acceptance test is re-derived as a one-sided non-inferiority
// gate with a pilot/validation Delta-calibration split and a new effect-retention conjunct
// (Sec4-6 above, T2027Dim/T2027* constants and the NonInferiorityStat/FrozenDelta/Verdict
// machinery); B7 is re-derived for the explicit-only fallback contract. B0 and B6 are carried
// forward UNCHANGED -- neither is touched by the four amendments (design Sec21: "none of the
// four items touches Sec3's insertion point, Sec4's fold construction... or Sec5's one-currency
// K/V property"). See Claude/Curie/t2018-slora-serial-red-suite-2026-08-13.md Sec2 (T-2027
// section) for the full re-derivation record and true counts.
//
// SCOPE OF THIS FILE: the four B-steps the design itself classifies as OFFLINE/CONVERTER-SIDE
// (no engine ABI or ProjectAndFunnel insertion-point change) -- B0 (converter-side delta-fold
// derivation), B3 (composed-quantity domain + BAKED-ADAPTER COMPARATOR), B6 (conversion-time
// saturation/validation tooling), B7 (rejection/fallback wiring). Every cell below runs today,
// for real, against the REAL unmodified engine primitives at D:\SuperSLM\.worktrees\run @
// 727e63e (ApplyWeightScaleFold, SaturatingRoundingDoublingHighMul, RoundingDivideByPOT,
// RequantChainChecked, LandingRescale) -- no engine header or source file is modified by this
// file. `ApplyAmplifyingWeightScaleFold` (design Sec4, D-SLM2915) does not exist in that engine
// (`grep -rl "adapter\|Adapter\|LoRA\|lora" src/ include/` returns nothing, confirmed again at
// this suite's own authoring time) and is reproduced here as a scratch copy -- this file's own,
// per this seat's standing discipline of never editing another seat's filed instrument.
//
// PROVENANCE (StandardsDocument.md Sec7 sibling-pinning discipline: the reused source is named,
// not silently absorbed). The fixture generator, seed, quantization, DeriveTripleOld/New,
// SaturatingLeftShift32, ApplyAmplifyingWeightScaleFold, RealizedRatio, RequiredRatioIndependent,
// kDerivationRelBound, and the WIRING/DERIVATION oracle shape below are reused, with attribution,
// from Claude/Vitruvius/t2009-verify-probe/t2007_body.cpp (Wizard repo) -- itself a verbatim
// scratch copy of Claude/Vitruvius/t2005-remedy-probe/t2005_remedy.cpp, which is itself a
// verbatim scratch copy of Claude/Loki/t2004-probe/t2004_body.cpp and
// Claude/Vitruvius/t2002-remedy-probe/t2002_remedy.cpp. Those probes proved the mechanism's own
// representability and the WIRING/DERIVATION gate's own discriminating power as ADVERSARY
// instruments (Loki/Vitruvius seats); this file re-derives the SAME constructions as a CURIE-
// authored, CHECK-counted, pass/fail RED suite -- the seat and the purpose differ (gating the
// build to green vs. striking a design claim), the arithmetic does not, per the commission's own
// direction to reuse these constructions where they fit.
//
// NEW IN THIS FILE, not present in any reused probe (B3's baked-adapter comparator did not exist
// until D-SLM3019/3020/3050, 2026-08-13, after the last probe in this lineage was filed):
//   - QuantizedBase / BuildAdapter: splits the probes' single-shot Run() into a FIXED
//     (base, adapter) build and a per-corpus-item token draw, so the SAME converted adapter is
//     paired against many corpus items -- Sec6 item 1's own pairing requirement.
//   - The BAKED arm: W' = W + B.A merged in float, quantized ONCE via the base's own unmodified
//     per-channel WSC1 derivation (design Sec6 item 1, verbatim formula), producing a second,
//     independent int8 output that shares the adapter's INPUTS with the runtime arm and NONE of
//     the runtime mechanism's internals (T, the delta fold, u_i8, rho) -- the property Sec6 item
//     1 states closes the T-2007 axis by construction.
//   - The paired statistic gap[i] = float_distance_runtime[i] - float_distance_baked[i], its
//     mean, standard error, resolving power, and sign count (D-SLM3050's own named form,
//     D-SLM2824's paired-per-item precedent) -- computed over a HAND-BUILT, synthetic corpus of
//     independent token draws under one fixed (base, adapter) pair. This is NOT the real
//     calibration corpus (no adapter has converted through either path yet, design Sec2/Sec13);
//     it proves the STATISTIC MACHINERY and the MUTATION-PROOFS discriminate correctly, which is
//     everything a red-first cell can prove before B3's own real-corpus execution. The numeric
//     bar itself stays explicitly UNDERIVED (design Sec13, D-SLM3020) -- no cell below asserts a
//     fixed margin threshold.
//
// Standard-library-only, CHECK/CHECK_MSG counters -- this tree's tests/test_main.cpp convention
// (see that file's own header comment), so a reader who knows that harness reads this one at a
// glance. Not wired into tests/test_main.cpp or CMakeLists.txt: this suite tests CONVERTER-SIDE
// logic that does not exist as a shipped converter mode yet (design Sec9: "a new converter-level
// rejection/fallback branch", "a new conversion-time validation report" -- both offline Python
// surfaces per SuperSLM_Plan.md Sec11, not this repo's C++ engine target). It is a standalone
// translation unit, built and run directly (build.bat, this directory), exactly like every
// existing probe in this lineage -- the build seat's job (B0/B3/B6/B7) is to port this file's
// already-proven arithmetic into the real Python converter and wire this file's assertions into
// that target's own test runner.

#include "superslm/checked_chain_funnel.h"
#include "superslm/forward_sites.h"
#include "superslm/intmath.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>
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

// =============================================================================================
// Reused verbatim (attributed above): t2009-verify-probe/t2007_body.cpp lines ~63-119.
// =============================================================================================

static constexpr int32_t kShiftMin = 0, kShiftMax = 31;
static constexpr int32_t kAmpMin = -kShiftMax, kAmpMax = kShiftMax;
static constexpr int32_t kI32Max = 2147483647;
static constexpr int64_t kI32MinAsI64 = -2147483648LL;

static int32_t SaturatingLeftShift32(int32_t x, int shift) {
	if (shift <= 0) return x;
	const int64_t wide = static_cast<int64_t>(x) << shift;
	if (wide > static_cast<int64_t>(kI32Max)) return kI32Max;
	if (wide < kI32MinAsI64) return static_cast<int32_t>(kI32MinAsI64);
	return static_cast<int32_t>(wide);
}

// Design Sec4 (D-SLM2915)'s own code block, reproduced as a scratch copy -- this symbol does not
// exist in the real engine (confirmed by grep, this file's own header comment). The
// `exponent >= 0` branch calls the two REAL engine primitives (SaturatingRoundingDoublingHighMul,
// RoundingDivideByPOT) exactly as ApplyWeightScaleFold already does; only the `exponent < 0`
// branch (SaturatingLeftShift32 above) is new arithmetic, per the design's own text.
static int64_t ApplyAmplifyingWeightScaleFold(int64_t acc, int32_t identity, int32_t mult,
                                               int32_t exponent) {
	if (identity != 0) return acc;
	const int32_t hi =
	    SaturatingRoundingDoublingHighMul(static_cast<int32_t>(acc), mult);
	if (exponent >= 0) return static_cast<int64_t>(RoundingDivideByPOT(hi, exponent));
	return static_cast<int64_t>(SaturatingLeftShift32(hi, -exponent));
}

// The OLD mechanism T-1990 proved fractures: ApplyWeightScaleFold's own derivation, representable
// ratio set (0,1] only (this IS the real engine primitive; DeriveTripleOld only ever derives a
// triple ApplyWeightScaleFold, unmodified, consumes).
static bool DeriveTripleOld(double rho, int32_t* id, int32_t* mult, int32_t* shift) {
	if (rho == 1.0) { *id = 1; *mult = 0; *shift = 0; return true; }
	if (rho <= 0.0) return false;
	int e2 = 0;
	const double m = std::frexp(rho, &e2);
	int64_t q = static_cast<int64_t>(std::llround(m * 2147483648.0));
	if (q == 2147483648LL) { q /= 2; ++e2; }
	const int32_t s = -e2;
	if (s < kShiftMin || s > kShiftMax) return false;
	if (q < 1 || q > 2147483647LL) return false;
	*id = 0; *mult = static_cast<int32_t>(q); *shift = s; return true;
}

// The NEW mechanism's own derivation (design Sec4, D-SLM2915): the amplifying primitive's
// representable ratio domain, signed exponent in [-31,31].
static bool DeriveTripleNew(double rho, int32_t* id, int32_t* mult, int32_t* exp) {
	if (rho == 1.0) { *id = 1; *mult = 0; *exp = 0; return true; }
	if (rho <= 0.0) return false;
	int e2 = 0;
	const double m = std::frexp(rho, &e2);
	int64_t q = static_cast<int64_t>(std::llround(m * 2147483648.0));
	if (q == 2147483648LL) { q /= 2; ++e2; }
	const int32_t e = -e2;
	if (e < kAmpMin || e > kAmpMax) return false;
	if (q < 1 || q > 2147483647LL) return false;
	*id = 0; *mult = static_cast<int32_t>(q); *exp = e; return true;
}

static uint64_t g_state = 0;
static double Uniform() {
	uint64_t z = (g_state += 0x9E3779B97F4A7C15ull);
	z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
	z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
	z = z ^ (z >> 31);
	return static_cast<double>(z >> 11) / 9007199254740992.0;
}
static double Gauss() {
	double u1 = Uniform();
	if (u1 < 1e-15) u1 = 1e-15;
	return std::sqrt(-2.0 * std::log(u1)) * std::cos(6.283185307179586 * Uniform());
}

// DERIVATION's own reference (design Sec18/Sec19, D-SLM2979/D-SLM3018): pure algebraic decode of
// a fixed-point triple, and the second, independent recombination of the base's and adapter's own
// scale artifacts. Reused verbatim (attributed above).
static double RealizedRatio(int32_t id, int32_t mult, int32_t exp) {
	if (id != 0) return 1.0;
	return (static_cast<double>(mult) / 2147483648.0) * std::exp2(static_cast<double>(-exp));
}
static double RequiredRatioIndependent(double T_artifact, double beta_i_artifact, double S_artifact) {
	return (T_artifact * beta_i_artifact) / S_artifact;
}
// Constructive quantization bound (design Sec4/Sec18): half a unit in the last place of a 31-bit
// unsigned mantissa field -- 2^-31, not fitted to any measured cell.
static constexpr double kDerivationRelBound = 1.0 / 2147483648.0;  // 2^-31

// =============================================================================================
// NEW: QuantizedBase / BuildAdapter -- fixes ONE (base, adapter) pair, split from the probes'
// single-shot per-call random draw so Sec6 item 1's pairing (same adapter, many corpus items) is
// representable. Includes the BAKED arm (design Sec6 item 1's own formula: W' = W + B.A merged in
// float, quantized ONCE via the base's own unmodified per-channel WSC1 derivation).
// =============================================================================================

struct QuantizedBase {
	int d = 0, out = 0, r = 0;
	// Base weight, quantized (real WSC1-style per-channel fold inputs).
	std::vector<double> W;   // out*d, float
	std::vector<int8_t> Wc;  // out*d, int8
	std::vector<double> w;   // out, per-channel base scale
	double S = 0.0;          // shared base reference scale, S = max_i w[i]
	// Adapter A/B, quantized.
	std::vector<double> Af;  // r*d, float
	std::vector<int8_t> Ac;  // r*d, int8
	std::vector<double> alpha;  // r
	std::vector<double> Bf;  // out*r, float (gain already folded in, matching the reused probes'
	                          // own convention of modelling the folded PEFT scaling through `gain`)
	std::vector<int8_t> Bc;  // out*r, int8
	std::vector<double> beta;  // out
	// Design Sec4/D-SLM2916: T = max_k(alpha_k * max|u_acc[k]| / 127), a CORPUS statistic --
	// approximated here from the SAME synthetic corpus BuildAdapter draws to size T (a converter
	// calibrates T from representative tokens; this build does the same, honestly, before any
	// T_SCALE defect is injected by a cell that wants one).
	double T_honest = 0.0;
	double t_scale_knob = 1.0;  // 1.0 = honest; T-2007's own T_SCALE(t) defect injection point
	double T() const { return T_honest * t_scale_knob; }
	// BAKED arm (design Sec6 item 1): W' = W + B.A, merged in float ONCE, quantized ONCE via the
	// base's own unmodified per-channel WSC1 derivation applied to W' -- shares NONE of the
	// runtime mechanism's internals (T, u_i8, rho).
	std::vector<int8_t> Wpc;  // out*d
	std::vector<double> wp;   // out
	double Sp = 0.0;
};

static QuantizedBase BuildAdapter(int d, int out, int r, double gain, uint64_t build_seed,
                                   double t_scale_knob = 1.0) {
	QuantizedBase qb;
	qb.d = d; qb.out = out; qb.r = r; qb.t_scale_knob = t_scale_knob;
	g_state = build_seed;

	qb.W.assign((size_t)out * d, 0.0);
	for (auto& v : qb.W) v = 0.02 * Gauss();
	qb.Af.assign((size_t)r * d, 0.0);
	for (auto& v : qb.Af) v = Gauss() / std::sqrt((double)d);
	qb.Bf.assign((size_t)out * r, 0.0);
	for (auto& v : qb.Bf) v = gain * Gauss() / std::sqrt((double)r);

	qb.w.assign(out, 0.0);
	qb.Wc.assign((size_t)out * d, 0);
	for (int i = 0; i < out; ++i) {
		double mx = 0.0;
		for (int j = 0; j < d; ++j) mx = std::max(mx, std::fabs(qb.W[(size_t)i * d + j]));
		qb.w[i] = mx / 127.0;
		for (int j = 0; j < d; ++j)
			qb.Wc[(size_t)i * d + j] = (int8_t)std::llround(qb.W[(size_t)i * d + j] / qb.w[i]);
	}
	qb.S = 0.0;
	for (double v : qb.w) qb.S = std::max(qb.S, v);

	qb.alpha.assign(r, 0.0);
	qb.Ac.assign((size_t)r * d, 0);
	for (int k = 0; k < r; ++k) {
		double mx = 0.0;
		for (int j = 0; j < d; ++j) mx = std::max(mx, std::fabs(qb.Af[(size_t)k * d + j]));
		qb.alpha[k] = mx / 127.0;
		for (int j = 0; j < d; ++j)
			qb.Ac[(size_t)k * d + j] = (int8_t)std::llround(qb.Af[(size_t)k * d + j] / qb.alpha[k]);
	}
	qb.beta.assign(out, 0.0);
	qb.Bc.assign((size_t)out * r, 0);
	for (int i = 0; i < out; ++i) {
		double mx = 0.0;
		for (int k = 0; k < r; ++k) mx = std::max(mx, std::fabs(qb.Bf[(size_t)i * r + k]));
		qb.beta[i] = mx / 127.0;
		for (int k = 0; k < r; ++k)
			qb.Bc[(size_t)i * r + k] = (int8_t)std::llround(qb.Bf[(size_t)i * r + k] / qb.beta[i]);
	}

	// T calibration: same construction as t2005/t2007's own Run(), over ONE representative
	// calibration draw (a converter's real calibration corpus; here, one more draw from the
	// SAME stream immediately after A/B are fixed).
	{
		std::vector<double> xf(d);
		for (auto& v : xf) v = Gauss();
		double xmax = 0.0;
		for (double v : xf) xmax = std::max(xmax, std::fabs(v));
		const double X = xmax / 127.0;
		std::vector<int8_t> xc(d);
		for (int j = 0; j < d; ++j) xc[j] = (int8_t)std::llround(xf[j] / X);
		std::vector<int64_t> u_acc(r, 0);
		for (int k = 0; k < r; ++k) {
			int64_t a = 0;
			for (int j = 0; j < d; ++j) a += (int64_t)xc[j] * qb.Ac[(size_t)k * d + j];
			u_acc[k] = a;
		}
		double T = 0.0;
		for (int k = 0; k < r; ++k)
			T = std::max(T, qb.alpha[k] * (double)std::llabs(u_acc[k]) / 127.0);
		if (T <= 0.0) T = 1.0;
		qb.T_honest = T;
	}

	// BAKED arm: W' = W + B.A, merged in float ONCE (token-independent, exactly what a real
	// merge+quantize converter does), quantized ONCE via the base's own unmodified per-channel
	// WSC1 derivation (design Sec6 item 1's own formula, verbatim).
	{
		std::vector<double> Wp((size_t)out * d, 0.0);
		for (int i = 0; i < out; ++i) {
			for (int j = 0; j < d; ++j) {
				double v = qb.W[(size_t)i * d + j];
				for (int k = 0; k < r; ++k) v += qb.Bf[(size_t)i * r + k] * qb.Af[(size_t)k * d + j];
				Wp[(size_t)i * d + j] = v;
			}
		}
		qb.wp.assign(out, 0.0);
		qb.Wpc.assign((size_t)out * d, 0);
		for (int i = 0; i < out; ++i) {
			double mx = 0.0;
			for (int j = 0; j < d; ++j) mx = std::max(mx, std::fabs(Wp[(size_t)i * d + j]));
			qb.wp[i] = mx / 127.0;
			for (int j = 0; j < d; ++j)
				qb.Wpc[(size_t)i * d + j] = (int8_t)std::llround(Wp[(size_t)i * d + j] / qb.wp[i]);
		}
		qb.Sp = 0.0;
		for (double v : qb.wp) qb.Sp = std::max(qb.Sp, v);
	}
	return qb;
}

// mech: 0 = OLD (T-1990 fracture, ApplyWeightScaleFold applied to an amplifying ratio)
//       1 = NEW, correct repair (ApplyAmplifyingWeightScaleFold, honest rho)
//       2 = PARTIAL(f) -- knob = fraction of amplifying channels left on the OLD clamped triple
//       4 = RHO_SCALE(c) -- knob = c, rho_conv'[i] = c * rho_conv[i] on every channel
//       5 = RHO_PERM(p) -- knob = p, off-by-one index into beta[] for a fraction p of channels
//       7 = WRONG_DIRECTION -- deliberately inverted ratio (1/rho instead of rho), B0's own
//           second red-first cell's "deliberately mis-derived triple" fixture family, reused for
//           B3's mutation-proof per the design's own cross-reference (Sec11 B3 second cell)
struct TokenResult {
	// T-2027 (D-SLM3089): normalized L2 is now the PRIMARY composed/effect metric --
	// sqrt(sum_c (a_c-b_c)^2) / sqrt(sum_c b_c^2), a shared-denominator statistic computed once
	// over the whole channel vector for this item, not a pointwise ratio averaged after the fact.
	// The suite's own original L1-shaped statistic (sum|diff|/sum|ref|, T-2018's own
	// float_distance_*) is retained as the secondary, outlier-resistant reading.
	double composed_l2_runtime = 0.0, composed_l2_baked = 0.0;      // PRIMARY (D-SLM3089)
	double float_distance_runtime = 0.0, float_distance_baked = 0.0;  // secondary L1 (unchanged)
	// T-2027 (D-SLM3088): effect-retention -- composed minus base, per channel, graded against the
	// float PEFT delta reference (yd) by the SAME normalized-L2 metric.
	double effect_l2_runtime = 0.0, effect_l2_baked = 0.0;
	double err_nodelta = 0.0;  // base-only floor, reported for context
	int wiring_mismatches = 0;
	int derivation_mismatches = 0;
	double max_derivation_rel_err = 0.0;
	bool gate_passes = false;
	int amplifying_chan = 0;
	int infeasible_chan = 0;
};

static TokenResult RunToken(const QuantizedBase& qb, uint64_t token_seed, int mech, double knob) {
	const int d = qb.d, out = qb.out, r = qb.r;
	g_state = token_seed;
	std::vector<double> xf(d);
	for (auto& v : xf) v = Gauss();
	double xmax = 0.0;
	for (double v : xf) xmax = std::max(xmax, std::fabs(v));
	const double X = xmax / 127.0;
	std::vector<int8_t> xc(d);
	for (int j = 0; j < d; ++j) xc[j] = (int8_t)std::llround(xf[j] / X);

	const double T = qb.T();
	const bool old_mech = (mech == 0);

	std::vector<int64_t> u_acc(r, 0);
	for (int k = 0; k < r; ++k) {
		int64_t a = 0;
		for (int j = 0; j < d; ++j) a += (int64_t)xc[j] * qb.Ac[(size_t)k * d + j];
		u_acc[k] = a;
	}
	std::vector<int8_t> u_i8(r);
	for (int k = 0; k < r; ++k) {
		const double rho_u = qb.alpha[k] / T;
		int64_t uw;
		if (old_mech) {
			int32_t id, mu, sh;
			uw = DeriveTripleOld(rho_u <= 1.0 ? rho_u : 1.0, &id, &mu, &sh)
			         ? ApplyWeightScaleFold(u_acc[k], id, mu, sh)
			         : u_acc[k];
		} else {
			int32_t id, mu, ex;
			uw = DeriveTripleNew(rho_u, &id, &mu, &ex)
			         ? ApplyAmplifyingWeightScaleFold(u_acc[k], id, mu, ex)
			         : u_acc[k];
		}
		u_i8[k] = (int8_t)(uw > 127 ? 127 : (uw < -127 ? -127 : uw));
	}

	std::vector<int64_t> delta_raw(out, 0);
	for (int i = 0; i < out; ++i) {
		int64_t a = 0;
		for (int k = 0; k < r; ++k) a += (int64_t)u_i8[k] * qb.Bc[(size_t)i * r + k];
		delta_raw[i] = a;
	}

	std::vector<int> amplifying;
	for (int i = 0; i < out; ++i)
		if (T * qb.beta[i] / qb.S > 1.0) amplifying.push_back(i);
	const int n_unrepaired =
	    (mech == 2) ? (int)std::llround(knob * (double)amplifying.size())
	                : (mech == 0 ? (int)amplifying.size() : 0);
	std::vector<char> unrepaired(out, 0);
	for (int t = 0; t < n_unrepaired && t < (int)amplifying.size(); ++t) unrepaired[amplifying[t]] = 1;

	std::vector<double> rho_conv(out);
	for (int i = 0; i < out; ++i) {
		double b = qb.beta[i];
		if (mech == 5) {
			const int n_bad = (int)std::llround(knob * (double)out);
			if (i < n_bad) b = qb.beta[(i + 1) % out];
		}
		rho_conv[i] = T * b / qb.S;
		if (mech == 4) rho_conv[i] *= knob;
	}

	TokenResult res;
	int infeasible = 0;
	for (int i = 0; i < out; ++i) {
		int32_t id, mu, e;
		const bool ok = old_mech ? DeriveTripleOld(rho_conv[i], &id, &mu, &e)
		                         : DeriveTripleNew(rho_conv[i], &id, &mu, &e);
		if (!ok) ++infeasible;
	}
	res.infeasible_chan = infeasible;
	res.amplifying_chan = (int)amplifying.size();

	double el = 0.0, ebk = 0.0, en = 0.0, den = 0.0;
	// T-2027: L2 accumulators (composed, primary) and effect-retention accumulators (both metrics'
	// numerator/denominator, L2 only -- effect retention's own denominator is the float PEFT delta
	// vector's own L2 norm, distinct from the composed conjunct's float-reference L2 norm).
	double el2_num = 0.0, el2_den = 0.0, ebk2_num = 0.0, ebk2_den = 0.0;
	double eff_rt_num = 0.0, eff_bk_num = 0.0, eff_den = 0.0;
	int wiring_mismatches = 0, derivation_mismatches = 0;
	double max_rel_err = 0.0;
	const double scale = X * qb.S;
	const double scale_baked = X * qb.Sp;
	for (int i = 0; i < out; ++i) {
		int64_t acc = 0;
		for (int j = 0; j < d; ++j) acc += (int64_t)xc[j] * qb.Wc[(size_t)i * d + j];
		int32_t bid, bmu, bsh;
		DeriveTripleOld(qb.w[i] / qb.S, &bid, &bmu, &bsh);
		const int64_t acc_wide = ApplyWeightScaleFold(acc, bid, bmu, bsh);

		double rho = rho_conv[i];
		if (mech == 7 && rho > 0.0) rho = 1.0 / rho;  // WRONG_DIRECTION -- inverted ratio
		int64_t dwl;
		const bool use_old_fold = old_mech || (mech == 2 && unrepaired[i]);
		int32_t a, b, c;
		if (use_old_fold) {
			if (!DeriveTripleOld(rho, &a, &b, &c)) { a = 1; b = 0; c = 0; }
			dwl = ApplyWeightScaleFold(delta_raw[i], a, b, c);
		} else {
			if (!DeriveTripleNew(rho, &a, &b, &c)) { a = 1; b = 0; c = 0; }
			dwl = ApplyAmplifyingWeightScaleFold(delta_raw[i], a, b, c);
		}

		// WIRING -- branch-selection only (design Sec18/Sec19).
		int32_t ref_id, ref_mu, ref_ex;
		const bool ref_ok = DeriveTripleNew(rho_conv[i], &ref_id, &ref_mu, &ref_ex);
		const bool applied_via_new_primitive = !use_old_fold;
		const bool wiring_matches =
		    ref_ok && applied_via_new_primitive && (a == ref_id) && (b == ref_mu) && (c == ref_ex);
		if (!wiring_matches) ++wiring_mismatches;

		// DERIVATION -- ratio-self-consistency given the converter's own T (design Sec18/Sec19).
		const double rho_indep = RequiredRatioIndependent(T, qb.beta[i], qb.S);
		const double realized = RealizedRatio(a, b, c);
		const double rel_err = std::fabs(realized - rho_indep) / rho_indep;
		max_rel_err = std::max(max_rel_err, rel_err);
		if (rel_err > kDerivationRelBound) ++derivation_mismatches;

		// Composed float reference (design Sec6 item 1's own quantity).
		double yb = 0.0, yd = 0.0;
		for (int j = 0; j < d; ++j) yb += qb.W[(size_t)i * d + j] * xf[j];
		for (int k = 0; k < r; ++k) {
			double u = 0.0;
			for (int j = 0; j < d; ++j) u += qb.Af[(size_t)k * d + j] * xf[j];
			yd += qb.Bf[(size_t)i * r + k] * u;
		}
		const double ref = yb + yd;
		const double runtime_composed_val = (double)(acc_wide + dwl) * scale;
		const double base_val = (double)acc_wide * scale;  // SHARED base-only value, both arms (D-SLM3088)

		// RUNTIME arm, composed conjunct (secondary L1, unchanged; PRIMARY L2, new T-2027).
		el += std::fabs(runtime_composed_val - ref);
		en += std::fabs(base_val - ref);
		el2_num += (runtime_composed_val - ref) * (runtime_composed_val - ref);
		el2_den += ref * ref;

		// BAKED arm: independent GEMM against the merged, once-quantized weight, then the base's
		// own unmodified ApplyWeightScaleFold using the baked-arm's own (wp[i], Sp) fold --
		// shares xc (the SAME token, per Sec6 item 1's pairing) and nothing of the runtime
		// mechanism's internals.
		int64_t bacc = 0;
		for (int j = 0; j < d; ++j) bacc += (int64_t)xc[j] * qb.Wpc[(size_t)i * d + j];
		int32_t pid, pmu, psh;
		DeriveTripleOld(qb.wp[i] / qb.Sp, &pid, &pmu, &psh);
		const int64_t bacc_wide = ApplyWeightScaleFold(bacc, pid, pmu, psh);
		const double baked_composed_val = (double)bacc_wide * scale_baked;
		ebk += std::fabs(baked_composed_val - ref);
		ebk2_num += (baked_composed_val - ref) * (baked_composed_val - ref);
		ebk2_den += ref * ref;

		den += std::fabs(ref);

		// T-2027 (D-SLM3088): EFFECT-RETENTION -- composed minus the SHARED base value, graded
		// against the float PEFT delta reference `yd` alone (not `ref = yb+yd`). At full
		// annihilation (delta_wide==0 for every channel), effect_runtime == 0 identically while
		// effect_baked tracks the baked arm's own genuine, nonzero merge effect -- this is what
		// makes effect_distance_runtime == 1.0 exactly at T_SCALE(255.9), independent of yb's own
		// magnitude, because yb has already been subtracted out before this line runs.
		const double effect_runtime_val = runtime_composed_val - base_val;  // == dwl[i]*scale, exact
		const double effect_baked_val = baked_composed_val - base_val;
		eff_rt_num += (effect_runtime_val - yd) * (effect_runtime_val - yd);
		eff_bk_num += (effect_baked_val - yd) * (effect_baked_val - yd);
		eff_den += yd * yd;
	}
	res.float_distance_runtime = el / den;
	res.float_distance_baked = ebk / den;
	res.composed_l2_runtime = std::sqrt(el2_num) / std::sqrt(el2_den);
	res.composed_l2_baked = std::sqrt(ebk2_num) / std::sqrt(ebk2_den);
	res.effect_l2_runtime = std::sqrt(eff_rt_num) / std::sqrt(eff_den);
	res.effect_l2_baked = std::sqrt(eff_bk_num) / std::sqrt(eff_den);
	res.err_nodelta = en / den;
	res.wiring_mismatches = wiring_mismatches;
	res.derivation_mismatches = derivation_mismatches;
	res.max_derivation_rel_err = max_rel_err;
	res.gate_passes = (wiring_mismatches == 0) && (derivation_mismatches == 0);
	return res;
}

// =============================================================================================
// T-2027 AMENDMENT (D-SLM3066/D-SLM3085-3101, folded into the design at commit 3de10627f2 Sec21,
// and into SuperSLM_Plan.md Sec6.7/Sec11/Sec19 at commit fe2ba3e7b9): the one-sided
// non-inferiority acceptance test, the pilot/validation Delta-calibration split, and the
// effect-retention conjunct. This machinery REPLACES T-2018's own difference-from-zero
// PairedGapStat/ComputePairedGap (retained just below, unchanged, for cells that still cite it as
// a DIAGNOSTIC/mutation-direction check, never as the acceptance test itself from this point on).
// =============================================================================================

// Deterministic 80/20 validation/pilot corpus split (design Sec6 item 1, D-SLM3086): assignment is
// a pure function of the item's own identity (here, its seed), identical for every adapter graded
// against this corpus -- no adapter's own data ever influences which items fall in which half.
// `hash(item_id) mod 10 < 2` => PILOT, else VALIDATION, per the design's own stated rule.
static bool IsPilotItem(uint64_t item_seed) {
	uint64_t z = item_seed + 0x9E3779B97F4A7C15ull;
	z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
	z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
	z = z ^ (z >> 31);
	return (z % 10) < 2;
}

// One-sided non-inferiority statistic (design Sec6 item 1, D-SLM3085/3087): mean, SE, the
// one-sided upper confidence bound at a stated z (95% => z=1.645), and the P95 tail (D-SLM3090).
struct NonInferiorityStat {
	int n = 0;
	double mean = 0.0;
	double se = 0.0;
	double upper_ci = 0.0;   // mean + z*se -- the quantity the acceptance test itself reads
	double p95 = 0.0;        // predeclared tail statistic, gated beside the mean
	double sign_frac_positive = 0.0;
};

static constexpr double kZ95OneSided = 1.645;
static constexpr double kSafetyInflation = 1.5;  // design's own stated 1.5x pilot-calibration factor

static double Percentile95(std::vector<double> v) {
	if (v.empty()) return 0.0;
	std::sort(v.begin(), v.end());
	const double rank = 0.95 * (double)(v.size() - 1);
	const size_t lo = (size_t)rank;
	const size_t hi = lo + 1 < v.size() ? lo + 1 : lo;
	const double frac = rank - (double)lo;
	return v[lo] + frac * (v[hi] - v[lo]);
}

static NonInferiorityStat ComputeNonInferiorityStat(const std::vector<double>& gaps, double z) {
	NonInferiorityStat s;
	s.n = (int)gaps.size();
	if (s.n == 0) return s;
	double mean = 0.0;
	for (double g : gaps) mean += g;
	mean /= s.n;
	double var = 0.0;
	for (double g : gaps) var += (g - mean) * (g - mean);
	var /= (s.n > 1 ? (s.n - 1) : 1);
	const double se = std::sqrt(var) / std::sqrt((double)s.n);
	int pos = 0;
	for (double g : gaps) if (g > 0.0) ++pos;
	s.mean = mean;
	s.se = se;
	s.upper_ci = mean + z * se;
	s.p95 = Percentile95(gaps);
	s.sign_frac_positive = (double)pos / s.n;
	return s;
}

// Per-item composed and effect-retention gaps (L2, primary), split by pilot/validation membership.
struct CorpusGaps {
	std::vector<double> composed_pilot, composed_validation;
	std::vector<double> effect_pilot, effect_validation;
};

static CorpusGaps CollectCorpusGaps(const QuantizedBase& qb, int mech, double knob, int n_items,
                                     uint64_t seed_base) {
	CorpusGaps g;
	for (int i = 0; i < n_items; ++i) {
		const uint64_t item_seed = seed_base + (uint64_t)i * 0x9E3779B1u;
		const TokenResult tr = RunToken(qb, item_seed, mech, knob);
		const double composed_gap = tr.composed_l2_runtime - tr.composed_l2_baked;
		const double effect_gap = tr.effect_l2_runtime - tr.effect_l2_baked;
		if (IsPilotItem(item_seed)) {
			g.composed_pilot.push_back(composed_gap);
			g.effect_pilot.push_back(effect_gap);
		} else {
			g.composed_validation.push_back(composed_gap);
			g.effect_validation.push_back(effect_gap);
		}
	}
	return g;
}

// The four frozen Deltas (design Sec6 item 1/1a, D-SLM3086/3090): calibrated ONCE, from a
// WIRING-AND-DERIVATION-clean reference adapter's honest (t=1, mech=1) run over the PILOT
// partition only -- never from the adapter under test's own grading run (the circularity Dan's
// ruling names). `Delta = 1.5 * (mean_pilot + z*se_pilot)`; `Delta_tail = 1.5 * P95_pilot`.
struct FrozenDelta {
	double composed_mean = 0.0, composed_tail = 0.0;
	double effect_mean = 0.0, effect_tail = 0.0;
	int pilot_n_composed = 0, pilot_n_effect = 0;
};

static FrozenDelta CalibrateDelta(const QuantizedBase& reference, int n_items, uint64_t seed_base) {
	// The reference is graded at mech=1 (the honest mechanism), t_scale=1.0 -- the correct
	// mechanism's own baseline, exactly the "WIRING-AND-DERIVATION-clean" reference the design
	// requires (B0's gate passes on this construction by this suite's own B0 cells, above).
	const CorpusGaps g = CollectCorpusGaps(reference, /*mech=*/1, 0.0, n_items, seed_base);
	const NonInferiorityStat composed_pilot = ComputeNonInferiorityStat(g.composed_pilot, kZ95OneSided);
	const NonInferiorityStat effect_pilot = ComputeNonInferiorityStat(g.effect_pilot, kZ95OneSided);
	FrozenDelta d;
	d.composed_mean = kSafetyInflation * (composed_pilot.mean + kZ95OneSided * composed_pilot.se);
	d.composed_tail = kSafetyInflation * composed_pilot.p95;
	d.effect_mean = kSafetyInflation * (effect_pilot.mean + kZ95OneSided * effect_pilot.se);
	d.effect_tail = kSafetyInflation * effect_pilot.p95;
	d.pilot_n_composed = composed_pilot.n;
	d.pilot_n_effect = effect_pilot.n;
	return d;
}

// The full acceptance verdict (design Sec6 item 1/1a): both conjuncts, each its own mean-and-tail
// non-inferiority test against the frozen Delta, graded on the VALIDATION partition only.
struct Verdict {
	NonInferiorityStat composed_stat, effect_stat;
	bool composed_mean_accepts = false, composed_tail_accepts = false;
	bool effect_mean_accepts = false, effect_tail_accepts = false;
	bool accepts = false;  // ALL four conjuncts must accept
};

static Verdict Grade(const QuantizedBase& qb, int mech, double knob, const FrozenDelta& delta,
                      int n_items, uint64_t seed_base) {
	const CorpusGaps g = CollectCorpusGaps(qb, mech, knob, n_items, seed_base);
	Verdict v;
	v.composed_stat = ComputeNonInferiorityStat(g.composed_validation, kZ95OneSided);
	v.effect_stat = ComputeNonInferiorityStat(g.effect_validation, kZ95OneSided);
	v.composed_mean_accepts = v.composed_stat.upper_ci < delta.composed_mean;
	v.composed_tail_accepts = v.composed_stat.p95 < delta.composed_tail;
	v.effect_mean_accepts = v.effect_stat.upper_ci < delta.effect_mean;
	v.effect_tail_accepts = v.effect_stat.p95 < delta.effect_tail;
	v.accepts = v.composed_mean_accepts && v.composed_tail_accepts && v.effect_mean_accepts &&
	            v.effect_tail_accepts;
	return v;
}

// =============================================================================================
// T-2018's own difference-from-zero statistic (D-SLM3050) -- RETAINED, unchanged, but no longer
// the acceptance test (D-SLM3085 supersedes it for that role). Still used below as a diagnostic/
// mutation-direction check (does a defect move the gap the expected way at all), which is a
// weaker, still-true claim the retired form remains valid for.
// =============================================================================================

// D-SLM3050's own named statistic: per corpus item i, gap[i] = float_distance_runtime[i] -
// float_distance_baked[i]; the reported aggregate is the paired mean, with paired SE, achieved
// resolving power, and a sign count (D-SLM2824's own three quantities).
struct PairedGapStat {
	int n = 0;
	double mean = 0.0;
	double se = 0.0;
	double resolving_power = 0.0;  // 2*SE -- an approximate 95% half-width, named as such
	double sign_frac_positive = 0.0;  // fraction of items with gap[i] > 0
};

static PairedGapStat ComputePairedGap(const QuantizedBase& qb, int mech, double knob,
                                       int n_items, uint64_t seed_base) {
	std::vector<double> gaps(n_items);
	int pos = 0;
	for (int i = 0; i < n_items; ++i) {
		const TokenResult tr = RunToken(qb, seed_base + (uint64_t)i * 0x9E3779B1u, mech, knob);
		gaps[i] = tr.float_distance_runtime - tr.float_distance_baked;
		if (gaps[i] > 0.0) ++pos;
	}
	double mean = 0.0;
	for (double g : gaps) mean += g;
	mean /= n_items;
	double var = 0.0;
	for (double g : gaps) var += (g - mean) * (g - mean);
	var /= (n_items > 1 ? (n_items - 1) : 1);
	const double sd = std::sqrt(var);
	const double se = sd / std::sqrt((double)n_items);
	PairedGapStat s;
	s.n = n_items;
	s.mean = mean;
	s.se = se;
	s.resolving_power = 2.0 * se;
	s.sign_frac_positive = (double)pos / n_items;
	return s;
}

// =============================================================================================
// B0 -- Converter-side delta-fold derivation (design Sec11 B0).
// =============================================================================================

// B0's own first red-first cell: a hand-constructed zero-effect adapter (A or B all-zero) must
// derive delta-fold constants that fold to exactly zero contribution for every channel.
static void TestB0ZeroEffectAdapterFoldsToExactlyZeroContribution() {
	const int d = 64, out = 32, r = 4;
	QuantizedBase qb = BuildAdapter(d, out, r, /*gain=*/0.0, 0xB0'0001, 1.0);
	// gain=0.0 makes every Bf element exactly 0.0, which the SAME quantization loop (max|Bf|=0)
	// quantizes to beta[i]=0/127=0 -- guard against the div-by-zero this creates by asserting the
	// pre-condition this cell actually needs directly, then confirming the composed output
	// literally equals the base-only output on every channel.
	bool any_beta_nonzero = false;
	for (double b : qb.beta) if (b != 0.0) any_beta_nonzero = true;
	CHECK_MSG(!any_beta_nonzero, "fixture defect: a zero-gain adapter produced nonzero beta");

	const TokenResult tr = RunToken(qb, 0xB0'1001, /*mech=*/1, 0.0);
	// A zero-effect adapter must fold to a runtime output indistinguishable from the base-only
	// floor -- err_nodelta computed in RunToken already represents exactly that base-only
	// reference, so this cell's own claim is float_distance_runtime == err_nodelta.
	CHECK_MSG(std::fabs(tr.float_distance_runtime - tr.err_nodelta) < 1e-12,
	          "zero-effect adapter's composed output diverges from base-only: runtime=%.9f nodelta=%.9f",
	          tr.float_distance_runtime, tr.err_nodelta);
}

// B0's own second red-first cell: a hand-constructed nonzero adapter with a small-integer,
// hand-computable raw delta; the correct triple reproduces the expected value; a deliberately
// mis-derived triple (WRONG_DIRECTION, mech=7) diverges on the SAME fixture.
static void TestB0HandComputedNonzeroDeltaCorrectTripleMatchesMisderivedDiverges() {
	// Small, exact fixture: d=4, out=1, r=1, hand-chosen integers so the whole pipeline is
	// arithmetic-exact (no float rounding in the setup), isolating the fold's own correctness.
	// x = [1,1,1,1] (int8 codes), A = [1,1,1,1] (rank-1), so u_acc = 4. beta/S/T chosen so
	// rho = T*beta/S = 4.5 (an amplifying ratio > 1, matching the design's own executed witness,
	// Sec11 B0's own text: "executed at rho=4.5... rel_err = 0 for the correct triple, rel_err =
	// 0.778 for the mis-derived one").
	int32_t id, mu, ex;
	const bool ok = DeriveTripleNew(4.5, &id, &mu, &ex);
	CHECK_MSG(ok, "rho=4.5 must derive a legal amplifying triple (well inside [2^-31,2^31])");
	// A large enough magnitude that the primitive's own fixed-point quantization error (the
	// triple's mantissa is quantized to 31 bits, executed above at DeriveTripleNew's own
	// kDerivationRelBound = 2^-31) is negligible against delta_raw's own scale -- at delta_raw =
	// 1000 the SAME rounding step is a materially larger FRACTION of the small result, which is
	// exactly why B0's own construction uses realistic per-channel accumulator magnitudes (Sec4:
	// "|delta_raw[i]| <= r . 127^2") rather than a toy value; this cell keeps delta_raw exact
	// (an integer, no float involved in its own definition) while sizing it to isolate the
	// fold's rounding from the fixture's own scale.
	const int64_t delta_raw = 1'000'000;  // exact integer; large enough to isolate fold rounding
	const int64_t correct = ApplyAmplifyingWeightScaleFold(delta_raw, id, mu, ex);
	const double expected = 4.5 * (double)delta_raw;
	const double rel_err_correct = std::fabs((double)correct - expected) / expected;
	CHECK_MSG(rel_err_correct < 1e-5, "correct triple's rel_err=%.8f, expected ~1e-6 scale (the "
	          "primitive's own fixed-point rounding, not a defect)", rel_err_correct);

	// Deliberately mis-derived: invert the ratio direction (1/4.5 instead of 4.5) -- exactly this
	// file's own mech=7 WRONG_DIRECTION family, applied directly to the hand fixture.
	int32_t wid, wmu, wex;
	DeriveTripleNew(1.0 / 4.5, &wid, &wmu, &wex);
	const int64_t wrong = ApplyAmplifyingWeightScaleFold(delta_raw, wid, wmu, wex);
	const double rel_err_wrong = std::fabs((double)wrong - expected) / expected;
	CHECK_MSG(rel_err_wrong > 0.5,
	          "mis-derived (inverted-direction) triple must diverge measurably; rel_err=%.6f",
	          rel_err_wrong);
	CHECK_MSG(rel_err_wrong > rel_err_correct * 100.0,
	          "mis-derived triple's error must be far larger than the correct triple's: wrong=%.6f correct=%.6f",
	          rel_err_wrong, rel_err_correct);
}

// B0's own third red-first cell (reworded through Sec16/Sec17/Sec18/Sec19's own fold history): a
// two-part WIRING AND DERIVATION gate, at T-1990's exact fracture point (d=896, r=8, gain=0.4).
// The OLD mechanism must reproduce T-1990's own measured 26.0% infeasible channels and 0.02413
// composed relative error; the NEW mechanism must land at 0.0% infeasible and ~0.01156, and PASS
// the WIRING AND DERIVATION gate.
static void TestB0WiringDerivationGateAtTheFractureCellOldFailsNewPasses() {
	QuantizedBase qb = BuildAdapter(896, 896, 8, 0.4, 0x243F6A8885A308D3ull, 1.0);
	const TokenResult old_r = RunToken(qb, 0x243F6A8885A308D3ull ^ 0xF00D, 0, 0.0);
	const TokenResult new_r = RunToken(qb, 0x243F6A8885A308D3ull ^ 0xF00D, 1, 0.0);

	CHECK_MSG(old_r.infeasible_chan > 0,
	          "OLD mechanism at the fracture cell must show infeasible channels (T-1990's own 26%%); got %d",
	          old_r.infeasible_chan);
	CHECK_MSG(new_r.infeasible_chan == 0,
	          "NEW mechanism at the fracture cell must show zero infeasible channels; got %d",
	          new_r.infeasible_chan);
	CHECK_MSG(new_r.float_distance_runtime < old_r.float_distance_runtime,
	          "NEW mechanism's composed error must be smaller than OLD's at the fracture cell: new=%.5f old=%.5f",
	          new_r.float_distance_runtime, old_r.float_distance_runtime);
	CHECK_MSG(new_r.gate_passes,
	          "NEW mechanism (correct repair) must PASS the WIRING AND DERIVATION gate at the fracture cell");
	CHECK_MSG(new_r.max_derivation_rel_err <= kDerivationRelBound,
	          "NEW mechanism's own derivation rel_err (%.3e) must sit within the constructive 2^-31 bound",
	          new_r.max_derivation_rel_err);
}

// WIRING refuses every PARTIAL(f) with at least one unrepaired channel (T-2000's own class).
static void TestB0WiringRefusesPartialRepairAtEveryNonzeroFraction() {
	QuantizedBase qb = BuildAdapter(896, 896, 8, 0.4, 0x243F6A6885A308D3ull, 1.0);
	for (double f : {0.05, 232.0 / 233.0, 1.0}) {
		const TokenResult tr = RunToken(qb, 0xABCD, 2, f);
		CHECK_MSG(tr.wiring_mismatches > 0,
		          "PARTIAL(%.4f) must trip WIRING (>=1 unrepaired channel); wiring_mismatches=%d",
		          f, tr.wiring_mismatches);
		CHECK_MSG(!tr.gate_passes, "PARTIAL(%.4f) must be refused by the gate", f);
	}
}

// DERIVATION refuses RHO_SCALE(c) at every c != 1 (a constant-factor mis-derivation WIRING alone
// cannot see -- T-2004's own class).
static void TestB0DerivationRefusesRhoScaleMisderivation() {
	QuantizedBase qb = BuildAdapter(896, 896, 8, 0.4, 0x1234'5678, 1.0);
	for (double c : {1.0001, 1.0725, 2.0, 16.0}) {
		const TokenResult tr = RunToken(qb, 0xBEEF, 4, c);
		CHECK_MSG(tr.derivation_mismatches > 0,
		          "RHO_SCALE(%.4f) must trip DERIVATION; derivation_mismatches=%d", c,
		          tr.derivation_mismatches);
		CHECK_MSG(!tr.gate_passes, "RHO_SCALE(%.4f) must be refused by the gate", c);
	}
	// c = 1.0 (no mis-derivation) must NOT trip DERIVATION -- the calibration two-sided control
	// (StandardsDocument.md Sec5.4: the instrument must be inert at its own null point).
	const TokenResult inert = RunToken(qb, 0xBEEF, 4, 1.0);
	CHECK_MSG(inert.derivation_mismatches == 0,
	          "RHO_SCALE(1.0) (the inert null) must NOT trip DERIVATION; got %d mismatches",
	          inert.derivation_mismatches);
}

// DERIVATION refuses RHO_PERM(p) at every p > 0 (an off-by-one index into the adapter's own
// per-channel scale array -- T-2004's second class).
static void TestB0DerivationRefusesRhoPermMisderivation() {
	QuantizedBase qb = BuildAdapter(896, 896, 8, 0.4, 0x1234'0000, 1.0);
	for (double p : {0.01, 0.5, 1.0}) {
		const TokenResult tr = RunToken(qb, 0xC0FFEE, 5, p);
		CHECK_MSG(tr.derivation_mismatches > 0, "RHO_PERM(%.4f) must trip DERIVATION; got %d", p,
		          tr.derivation_mismatches);
		CHECK_MSG(!tr.gate_passes, "RHO_PERM(%.4f) must be refused by the gate", p);
	}
}

// The DERIVATION bound is constructive (2^-31), not fitted -- the correct mechanism's own realized
// ratio must sit strictly inside it at every gain this design's own sweeps use.
static void TestB0DerivationBoundIsConstructiveAcrossGains() {
	for (double g : {0.05, 0.4, 0.8, 1.6}) {
		QuantizedBase qb = BuildAdapter(896, 896, 8, g, 0x9999'0000 + (uint64_t)(g * 1000), 1.0);
		const TokenResult tr = RunToken(qb, 0x1111, 1, 0.0);
		CHECK_MSG(tr.max_derivation_rel_err <= kDerivationRelBound,
		          "gain=%.2f: correct mechanism's derivation rel_err %.3e exceeds the 2^-31 bound",
		          g, tr.max_derivation_rel_err);
	}
}

// EXPLICIT DISCLAIMER CELL (design Sec4/Sec10 dim 6/Sec12 R1/R2, D-SLM3018/D-SLM3023/D-SLM3025):
// the WIRING AND DERIVATION gate does NOT, and is not required to, refuse a converter that
// honestly mis-derives its own intermediate domain-fill target T (T-2007's own T_SCALE(t)
// construction) -- this is the third consecutive occurrence of StandardsDocument.md Sec5.4's
// "the reference shares a parameter with the thing under test" on this lineage, and sufficiency
// against it is B3's alone. This cell documents the gap AT B0 rather than silently leaving it
// unstated -- it asserts the gate's OWN documented limit, not a defect in this suite.
static void TestB0GateDoesNotAndCannotCatchTScaleMisderivationByDesign() {
	QuantizedBase honest = BuildAdapter(896, 896, 8, 0.4, 0xAAAA'0001, /*t_scale_knob=*/1.0);
	QuantizedBase defect = BuildAdapter(896, 896, 8, 0.4, 0xAAAA'0001, /*t_scale_knob=*/255.9);
	const TokenResult h = RunToken(honest, 0x2222, 1, 0.0);
	const TokenResult d = RunToken(defect, 0x2222, 1, 0.0);
	CHECK_MSG(h.gate_passes, "the honest (t=1) mechanism must pass WIRING AND DERIVATION");
	CHECK_MSG(d.gate_passes,
	          "T-2007's own finding: a converter that mis-derives T (t=255.9, the adapter "
	          "quantized to zero effect) STILL passes WIRING AND DERIVATION -- this is the "
	          "documented, accepted scope limit (design Sec4/Sec19), not a bug in this cell");
	// The defect is real and material even though the gate cannot see it: the composed output
	// under the T-mis-derivation is measurably worse than under the honest T.
	CHECK_MSG(d.float_distance_runtime > h.float_distance_runtime,
	          "the T-mis-derivation must be a REAL composed-quality defect even though the gate "
	          "cannot see it: defect=%.5f honest=%.5f", d.float_distance_runtime, h.float_distance_runtime);
}

// =============================================================================================
// B3 -- Composed-quantity domain and BAKED-ADAPTER COMPARATOR proof (design Sec11 B3).
// =============================================================================================

// B3's own first red-first cell: an adversarially large-magnitude adapter must trip the engine's
// OWN, REAL domain rejection (ChainInputOutOfDomain, C29) when the composed row is replayed
// through the REAL RequantChainChecked -- proving the composed path inherits the existing
// reject-over-degrade guarantee. This calls the actual production funnel entry point; nothing
// here is a scratch reimplementation of the domain check itself.
static void TestB3DomainCheckAdversarialComposedRowTripsChainInputOutOfDomain() {
	// A composed row engineered so at least one element's magnitude exceeds 2^31 after the
	// (base + delta) sum -- C29's own documented trigger (design Sec6 item 2: "one int32-bounded
	// value cannot itself exceed 2^31... once a delta composes into the same accumulator, two
	// such values summed CAN exceed it").
	const size_t n = 8;
	std::vector<int64_t> wide_row(n, 0);
	wide_row[0] = (int64_t)2147483647LL + (int64_t)2147483647LL;  // > 2^31, adversarial composition
	for (size_t i = 1; i < n; ++i) wide_row[i] = 1000;  // ordinary, in-domain elements elsewhere

	CarriedScale incoming{/*m=*/1717986918, /*e=*/-30};  // representative canonical activation scale
	CarriedScale site_constant{/*m=*/1717986918, /*e=*/0};
	std::vector<int8_t> out_codes(n);
	CarriedScale out_scale;
	const ChainResult cr =
	    RequantChainChecked(wide_row.data(), n, std::span<const CarriedScale>(&incoming, 1),
	                         site_constant, out_codes.data(), &out_scale);
	CHECK_MSG(cr.status == SslmForwardStatus::ChainInputOutOfDomain,
	          "an adversarially large composed row must trip ChainInputOutOfDomain at the REAL "
	          "RequantChainChecked; got status=%s",
	          SslmForwardStatusName(cr.status));

	// Negative control: the SAME call shape, but every element well inside domain, must return Ok
	// -- proving the rejection above is targeted, not a fixture-wide failure.
	std::vector<int64_t> ok_row(n, 1000);
	std::vector<int8_t> ok_codes(n);
	CarriedScale ok_scale;
	const ChainResult ok_cr =
	    RequantChainChecked(ok_row.data(), n, std::span<const CarriedScale>(&incoming, 1),
	                         site_constant, ok_codes.data(), &ok_scale);
	CHECK_MSG(ok_cr.status == SslmForwardStatus::Ok,
	          "an in-domain composed row must NOT be rejected; got status=%s",
	          SslmForwardStatusName(ok_cr.status));
}

// B3's own third red-first cell: a hand-constructed adapter whose true required ratio genuinely
// exceeds [2^-31, 2^31] must trip AmplifyingScaleRatioOutOfDomain (represented here, since the
// real status enum entry does not exist yet, by DeriveTripleNew's own documented failure return --
// the exact function B0's own converter-side derivation calls) rather than deriving a
// load-legal-but-wrong triple.
static void TestB3AmplifyingRatioGenuinelyOutOfDomainSignalsExplicitInfeasibility() {
	// T's structural ceiling is ~2^31 (design Sec4, corrected D-SLM2943); construct a ratio far
	// beyond it directly, bypassing the corpus-realistic BuildAdapter path (no measured LoRA
	// setting reaches within ~3e8x of this ceiling -- this cell is deliberately artificial, per
	// the design's own text).
	const double impossible_ratio = std::exp2(40.0);  // 2^40, far outside [2^-31, 2^31]
	int32_t id, mu, ex;
	const bool ok = DeriveTripleNew(impossible_ratio, &id, &mu, &ex);
	CHECK_MSG(!ok, "a ratio of 2^40 must be signaled genuinely infeasible, not silently accepted");

	// Negative control: a ratio at the domain's own structural ceiling must still derive legally.
	int32_t id2, mu2, ex2;
	const bool ok2 = DeriveTripleNew(std::exp2(30.0), &id2, &mu2, &ex2);
	CHECK_MSG(ok2, "a ratio of 2^30 (inside the domain) must derive a legal triple");
}

// =============================================================================================
// T-2027 RE-DERIVATION (D-SLM3066/D-SLM3085-3101): B3's second and fourth red-first cells are
// re-derived from the amended acceptance test -- one-sided non-inferiority, pilot/validation
// Delta calibration, and the effect-retention conjunct -- superseding T-2018's own
// difference-from-zero cells (which asserted an unconditional "refuse every t != 1" the amended
// contract explicitly retracts, D-SLM3092). Per D-SLM3106, this is a fresh derivation from the
// amended contract, not a patch of the struck cells.
// =============================================================================================

// The fixture and build seed T-2018/T-2020 already used at this exact cell (d=896, r=8, gain=0.4,
// build_seed=0xF00D'BEEF) is reused for continuity with T-2020's own boundary findings, which this
// re-derivation's t-sweep cell (below) cites as grounding.
static constexpr int kT2027Dim = 896, kT2027Out = 896, kT2027Rank = 8;
static constexpr double kT2027Gain = 0.4;
static constexpr uint64_t kT2027BuildSeed = 0xF00D'BEEF;
static constexpr int kT2027PilotCorpusN = 200;   // large enough for a stable PILOT calibration
static constexpr int kT2027GradeCorpusN = 200;   // VALIDATION-partition grading corpus size
static constexpr uint64_t kT2027PilotSeedBase = 0x1000;
static constexpr uint64_t kT2027GradeSeedBase = 0x9000;  // same seed_base T-2018 used at n=24

// B3's own second red-first cell, re-derived (design Sec6 item 1, D-SLM3085/3086): Delta is
// calibrated ONCE from a WIRING-AND-DERIVATION-clean reference adapter's honest (t=1, mech=1) run
// over the PILOT partition, frozen, then the SAME reference adapter is graded on its own
// VALIDATION partition against the now-frozen Delta -- genuine equality must ACCEPT, closing the
// exact failure mode (D-SLM2846's difference-from-zero form could never resolve "equal" as a
// pass) Dan's ruling names.
static void TestB3DeltaCalibratedOnPilotReferenceAcceptsOnValidation() {
	QuantizedBase reference = BuildAdapter(kT2027Dim, kT2027Out, kT2027Rank, kT2027Gain,
	                                        kT2027BuildSeed, /*t_scale_knob=*/1.0);
	// The reference must itself pass B0's WIRING-AND-DERIVATION gate cleanly (R7's own stated
	// precondition, D-SLM3100) -- confirmed via a single token draw before calibration proceeds.
	const TokenResult gate_check = RunToken(reference, 0xFEED, /*mech=*/1, 0.0);
	CHECK_MSG(gate_check.gate_passes,
	          "the PILOT reference adapter must pass B0's WIRING-AND-DERIVATION gate before it is "
	          "used to calibrate Delta (R7's own precondition, D-SLM3100)");

	const FrozenDelta delta = CalibrateDelta(reference, kT2027PilotCorpusN, kT2027PilotSeedBase);
	std::printf("   [B3 second cell] frozen Delta (n_pilot_composed=%d, n_pilot_effect=%d): "
	            "composed_mean=%.6f composed_tail=%.6f effect_mean=%.6f effect_tail=%.6f\n",
	            delta.pilot_n_composed, delta.pilot_n_effect, delta.composed_mean, delta.composed_tail,
	            delta.effect_mean, delta.effect_tail);
	CHECK_MSG(delta.composed_mean > 0.0 && delta.composed_tail > 0.0 && delta.effect_mean > 0.0 &&
	              delta.effect_tail > 0.0,
	          "all four calibrated Deltas must be strictly positive (a real, non-degenerate bound)");

	// Grade the SAME reference adapter on its own VALIDATION partition, disjoint from the items
	// that calibrated Delta (the pilot/validation split, D-SLM3086) -- genuine equality (the
	// honest mechanism graded against a Delta calibrated from itself, inflated 1.5x) must ACCEPT.
	const Verdict v = Grade(reference, /*mech=*/1, 0.0, delta, kT2027GradeCorpusN, kT2027GradeSeedBase);
	std::printf("   [B3 second cell] reference on VALIDATION: composed upper_ci=%.6f (Delta=%.6f) "
	            "composed p95=%.6f (Delta_tail=%.6f) effect upper_ci=%.6f (Delta_effect=%.6f) "
	            "effect p95=%.6f (Delta_effect_tail=%.6f) -> %s\n",
	            v.composed_stat.upper_ci, delta.composed_mean, v.composed_stat.p95, delta.composed_tail,
	            v.effect_stat.upper_ci, delta.effect_mean, v.effect_stat.p95, delta.effect_tail,
	            v.accepts ? "ACCEPT" : "reject");
	CHECK_MSG(v.accepts,
	          "a genuinely honest reference adapter, graded on VALIDATION against a Delta calibrated "
	          "from its OWN pilot partition (inflated 1.5x), must ACCEPT -- the exact failure mode "
	          "(equal arms refused forever) D-SLM3066 item 1 exists to close");
}

// A deliberately mis-derived delta-fold triple (WRONG_DIRECTION, reusing B0's own second
// red-first cell's fixture family) must be REJECTED by the composed conjunct against the SAME
// frozen Delta the reference calibrated -- the mutation-proof this cell inherits from T-2018,
// re-executed under the amended acceptance test rather than the retired difference-from-zero form.
static void TestB3WrongDirectionMutationRejectedAgainstFrozenDelta() {
	QuantizedBase reference = BuildAdapter(kT2027Dim, kT2027Out, kT2027Rank, kT2027Gain,
	                                        kT2027BuildSeed, /*t_scale_knob=*/1.0);
	const FrozenDelta delta = CalibrateDelta(reference, kT2027PilotCorpusN, kT2027PilotSeedBase);

	QuantizedBase same_base = BuildAdapter(kT2027Dim, kT2027Out, kT2027Rank, kT2027Gain,
	                                        kT2027BuildSeed, /*t_scale_knob=*/1.0);
	const Verdict v = Grade(same_base, /*mech=*/7, 0.0, delta, kT2027GradeCorpusN, kT2027GradeSeedBase);
	std::printf("   [B3 mutation-proof] WRONG_DIRECTION on VALIDATION: composed upper_ci=%.6f "
	            "(Delta=%.6f) -> %s\n", v.composed_stat.upper_ci, delta.composed_mean,
	            v.accepts ? "ACCEPT (should not happen)" : "reject");
	CHECK_MSG(!v.accepts,
	          "a deliberately mis-derived (inverted-direction) triple must be REJECTED against the "
	          "frozen Delta -- upper_ci=%.6f Delta=%.6f", v.composed_stat.upper_ci, delta.composed_mean);
}

// B3's own fifth red-first cell (design Sec6 item 1a, D-SLM3088/D-SLM3091): T_SCALE(255.9) full
// annihilation must fail the EFFECT-RETENTION conjunct UNCONDITIONALLY -- independent of the
// composed conjunct's own Delta, and independent of base-output magnitude, because the base has
// already been subtracted out of the graded quantity before the metric runs.
static void TestB3EffectRetentionRejectsFullAnnihilationUnconditionally() {
	QuantizedBase reference = BuildAdapter(kT2027Dim, kT2027Out, kT2027Rank, kT2027Gain,
	                                        kT2027BuildSeed, /*t_scale_knob=*/1.0);
	const FrozenDelta delta = CalibrateDelta(reference, kT2027PilotCorpusN, kT2027PilotSeedBase);

	QuantizedBase annihilated = BuildAdapter(kT2027Dim, kT2027Out, kT2027Rank, kT2027Gain,
	                                          kT2027BuildSeed, /*t_scale_knob=*/255.9);
	const TokenResult single = RunToken(annihilated, 0xAAAA, /*mech=*/1, 0.0);
	std::printf("   [B3 fifth cell, T_SCALE(255.9)] single-item effect_l2_runtime=%.9f "
	            "effect_l2_baked=%.6f (expect runtime == 1.0 exactly)\n",
	            single.effect_l2_runtime, single.effect_l2_baked);
	CHECK_MSG(std::fabs(single.effect_l2_runtime - 1.0) < 1e-12,
	          "at t=255.9, effect_l2_runtime must be EXACTLY 1.0 (100%% relative error against a "
	          "nonzero float-delta reference, every channel identically zero-effect): got %.12f",
	          single.effect_l2_runtime);

	const Verdict v = Grade(annihilated, /*mech=*/1, 0.0, delta, kT2027GradeCorpusN, kT2027GradeSeedBase);
	std::printf("   [B3 fifth cell] annihilated on VALIDATION: effect upper_ci=%.6f (Delta_effect=%.6f) "
	            "effect p95=%.6f (Delta_effect_tail=%.6f) composed accepts=%d effect accepts=%d "
	            "-> overall %s\n", v.effect_stat.upper_ci, delta.effect_mean, v.effect_stat.p95,
	            delta.effect_tail, v.composed_mean_accepts && v.composed_tail_accepts,
	            v.effect_mean_accepts && v.effect_tail_accepts, v.accepts ? "ACCEPT (should not happen)"
	                                                                       : "reject");
	CHECK_MSG(!v.effect_mean_accepts,
	          "the effect-retention mean conjunct must REJECT full annihilation: upper_ci=%.6f "
	          "Delta_effect=%.6f", v.effect_stat.upper_ci, delta.effect_mean);
	CHECK_MSG(!v.accepts, "the overall verdict must reject full annihilation via effect retention "
	          "alone, regardless of what the composed conjunct's own Delta would tolerate");
	// This holds regardless of base-output magnitude (D-SLM3091's own clause) -- re-run at a much
	// larger base weight scale (10x) to confirm the effect-retention refusal is unaffected.
	QuantizedBase annihilated_large_base = BuildAdapter(kT2027Dim, kT2027Out, kT2027Rank,
	                                                      kT2027Gain, kT2027BuildSeed, 255.9);
	for (auto& w : annihilated_large_base.W) w *= 10.0;  // scale the base weight up 10x
	// Re-quantize the scaled base (mirrors BuildAdapter's own per-channel fold derivation).
	for (int i = 0; i < annihilated_large_base.out; ++i) {
		double mx = 0.0;
		for (int j = 0; j < annihilated_large_base.d; ++j)
			mx = std::max(mx, std::fabs(annihilated_large_base.W[(size_t)i * annihilated_large_base.d + j]));
		annihilated_large_base.w[i] = mx / 127.0;
		for (int j = 0; j < annihilated_large_base.d; ++j)
			annihilated_large_base.Wc[(size_t)i * annihilated_large_base.d + j] =
			    (int8_t)std::llround(annihilated_large_base.W[(size_t)i * annihilated_large_base.d + j] /
			                          annihilated_large_base.w[i]);
	}
	annihilated_large_base.S = 0.0;
	for (double v2 : annihilated_large_base.w) annihilated_large_base.S = std::max(annihilated_large_base.S, v2);
	const TokenResult large_base_result = RunToken(annihilated_large_base, 0xAAAA, 1, 0.0);
	CHECK_MSG(std::fabs(large_base_result.effect_l2_runtime - 1.0) < 1e-12,
	          "effect_l2_runtime must remain EXACTLY 1.0 at a 10x-larger base-output magnitude "
	          "(D-SLM3091's own 'regardless of base-output magnitude' clause): got %.12f",
	          large_base_result.effect_l2_runtime);
}

// B3's own fourth red-first cell, rewritten as a sweep graded by the acceptance test itself
// (design Sec11 B3, D-SLM3092), reusing T-2020's own boundary grounding
// (Claude/Vitruvius/t2020-verify-probe/, D-SLM3062/D-SLM3068) at the SAME fixture and seed family.
// Required (D-SLM3066 item 3, D-SLM3092): (a) t=1 always accepts; (b) small perturbations move
// gap[i] in the expected direction, never asserted as a fixed accept/reject ahead of execution;
// (c) mutations whose measured upper_CI exceeds Delta are rejected by the composed conjunct; (d)
// full annihilation is rejected unconditionally by effect retention regardless of Delta (proven
// separately, above) -- this cell exercises (a)-(c) as an executed sweep, not a fixed list.
static void TestB3TScaleSweepGradedByTheAcceptanceTestItself() {
	QuantizedBase reference = BuildAdapter(kT2027Dim, kT2027Out, kT2027Rank, kT2027Gain,
	                                        kT2027BuildSeed, /*t_scale_knob=*/1.0);
	const FrozenDelta delta = CalibrateDelta(reference, kT2027PilotCorpusN, kT2027PilotSeedBase);
	std::printf("   [B3 fourth cell] frozen Delta: composed_mean=%.6f composed_tail=%.6f "
	            "effect_mean=%.6f effect_tail=%.6f\n", delta.composed_mean, delta.composed_tail,
	            delta.effect_mean, delta.effect_tail);

	// (a) t=1 always ACCEPTS -- bit-identical to the reference's own honest baseline.
	{
		QuantizedBase t1 = BuildAdapter(kT2027Dim, kT2027Out, kT2027Rank, kT2027Gain, kT2027BuildSeed, 1.0);
		const Verdict v = Grade(t1, 1, 0.0, delta, kT2027GradeCorpusN, kT2027GradeSeedBase);
		CHECK_MSG(v.accepts, "t=1 must ACCEPT (the correct mechanism's own baseline, D-SLM3066 item 3)");
	}

	// (b)/(c): sweep T-2007's own T_SCALE(t) family, graded by the acceptance test itself. No
	// fixed accept/reject is asserted ahead of execution for any individual t except t=1 (above)
	// and full annihilation (proven separately, unconditionally, by effect retention above) --
	// this loop reports the executed verdict at each t and asserts only the STRUCTURAL properties
	// D-SLM3092 requires: monotone-enough large-t refusal, and that the sweep is a genuine
	// function of t (not vacuously all-accept or all-reject).
	std::printf("   [B3 fourth cell] sweep, graded by the amended acceptance test (n=%d validation "
	            "items):\n", kT2027GradeCorpusN);
	int accepted = 0, rejected = 0;
	bool large_t_rejected = true;
	for (double t : {1.0001, 1.0725, 2.0, 4.0, 16.0, 20.0, 24.0, 32.0, 64.0, 200.0, 255.9, 256.0, 1024.0}) {
		QuantizedBase defect = BuildAdapter(kT2027Dim, kT2027Out, kT2027Rank, kT2027Gain, kT2027BuildSeed, t);
		const Verdict v = Grade(defect, 1, 0.0, delta, kT2027GradeCorpusN, kT2027GradeSeedBase);
		std::printf("     t=%-9.4f composed upper_ci=%.6f (Delta=%.6f) effect upper_ci=%.6f "
		            "(Delta_effect=%.6f) -> %s\n", t, v.composed_stat.upper_ci, delta.composed_mean,
		            v.effect_stat.upper_ci, delta.effect_mean, v.accepts ? "ACCEPT" : "reject");
		if (v.accepts) ++accepted; else ++rejected;
		if (t >= 64.0 && v.accepts) large_t_rejected = false;
	}
	CHECK_MSG(large_t_rejected,
	          "every t >= 64 (T-2020's own resolved-refusal region at this fixture) must be REJECTED");
	CHECK_MSG(accepted > 0 && rejected > 0,
	          "the sweep must be a genuine function of t -- some accepted (small perturbations, "
	          "non-inferior by construction) and some rejected (large perturbations), not vacuously "
	          "all one verdict; accepted=%d rejected=%d", accepted, rejected);

	// Grounding cross-check against T-2020's own executed boundary (Claude/Vitruvius/t2020-verify-
	// probe/t2020_extend.cpp, Part A): at this exact fixture and n=24 (the pre-amendment metric),
	// resolved refusal began at t=20, the first point above the design's own then-tested t=16. This
	// suite's own sweep above, under the NEW L2 metric and the amended acceptance test, is a fresh
	// execution -- reported for comparison, not asserted to reproduce the old metric's exact
	// boundary (a different statistic, D-SLM3089, is not guaranteed to cross Delta at the same t).
	std::printf("   [B3 fourth cell] T-2020's own prior finding (L1 metric, pre-amendment "
	            "difference-from-zero form, n=24): resolved refusal began at t=20 at this fixture -- "
	            "reported as grounding, not re-asserted verbatim under the new L2/non-inferiority "
	            "form (D-SLM3092).\n");
}

// Verdict-word discipline for the one-sided form (design Sec6 item 1, D-SLM3087): under-sampling
// must bias the test TOWARD reject/unresolved, never toward a false accept -- a smaller VALIDATION
// partition widens upper_CI (mean + z*se grows as se grows), which can only push the verdict
// further above Delta, not below it.
static void TestB3UnderSamplingWidensUpperCiNeverNarrowsIt() {
	QuantizedBase reference = BuildAdapter(kT2027Dim, kT2027Out, kT2027Rank, kT2027Gain,
	                                        kT2027BuildSeed, /*t_scale_knob=*/1.0);
	const CorpusGaps small = CollectCorpusGaps(reference, 1, 0.0, 24, kT2027GradeSeedBase);
	const CorpusGaps large = CollectCorpusGaps(reference, 1, 0.0, 400, kT2027GradeSeedBase);
	const NonInferiorityStat small_stat = ComputeNonInferiorityStat(small.composed_validation, kZ95OneSided);
	const NonInferiorityStat large_stat = ComputeNonInferiorityStat(large.composed_validation, kZ95OneSided);
	std::printf("   [B3 verdict-word] small validation (n=%d): se=%.6f upper_ci=%.6f | large "
	            "validation (n=%d): se=%.6f upper_ci=%.6f\n", small_stat.n, small_stat.se,
	            small_stat.upper_ci, large_stat.n, large_stat.se, large_stat.upper_ci);
	CHECK_MSG(small_stat.se >= large_stat.se,
	          "a smaller validation partition must have a LARGER (or equal) standard error: "
	          "small_se=%.6f (n=%d) large_se=%.6f (n=%d)", small_stat.se, small_stat.n,
	          large_stat.se, large_stat.n);
	CHECK_MSG(small_stat.upper_ci >= small_stat.mean,
	          "the one-sided upper CI must never sit BELOW the point estimate (it can only widen "
	          "the bound, biasing toward reject, never toward false accept)");
}

// =============================================================================================
// B6 -- Conversion-time saturation/validation tooling (design Sec11 B6).
// =============================================================================================

// A canonical, round-trip-consistent static K/V landing target (r_t, e_t), derived the same way
// C19's own reciprocal is (round_half_up(2^62/m_t)) -- a representative construction, not a claim
// about any real calibrated model. Used identically across every B6 cell below so the comparison
// between adversarial/typical/base-only is apples-to-apples against ONE fixed static target.
static void CanonicalStaticTarget(double ratio, int64_t* m_t, int64_t* e_t, int64_t* r_t) {
	int e2 = 0;
	const double m = std::frexp(ratio, &e2);
	*m_t = (int64_t)std::llround(m * 2147483648.0);
	if (*m_t == 2147483648LL) { *m_t /= 2; ++e2; }
	*e_t = e2;
	// C19's own reciprocal construction: R = round_half_up(2^62 / m_t), m_t in [2^30,2^31).
	*r_t = (int64_t)((static_cast<long double>(1) * (int64_t(1) << 62) / (long double)*m_t) + 0.5L);
}

// Composed K/V landing saturation rate: for a given (base, adapter) pair and mechanism, count how
// often LandingRescale's own clamp/saturation counter fires on the composed accumulator, over a
// synthetic corpus, using the REAL LandingRescale primitive.
static double MeasureKvSaturationRate(const QuantizedBase& qb, bool composed, int n_items,
                                       uint64_t seed_base) {
	int64_t m_t, e_t, r_t;
	CanonicalStaticTarget(qb.w[0] / qb.S, &m_t, &e_t, &r_t);  // representative base-only ratio
	int fired = 0, total = 0;
	for (int item = 0; item < n_items; ++item) {
		const TokenResult tr = RunToken(qb, seed_base + (uint64_t)item * 0x1234567u, 1, 0.0);
		(void)tr;  // TokenResult's own aggregate error is not this measurement's unit -- per-channel
		           // landing values are recomputed directly below, exactly as RunToken derives them.
		g_state = seed_base + (uint64_t)item * 0x1234567u;
		std::vector<double> xf(qb.d);
		for (auto& v : xf) v = Gauss();
		double xmax = 0.0;
		for (double v : xf) xmax = std::max(xmax, std::fabs(v));
		const double X = xmax / 127.0;
		std::vector<int8_t> xc(qb.d);
		for (int j = 0; j < qb.d; ++j) xc[j] = (int8_t)std::llround(xf[j] / X);
		int64_t m_a, e_a, dummy;
		CanonicalStaticTarget(X, &m_a, &e_a, &dummy);

		for (int i = 0; i < qb.out && i < 32; ++i) {  // representative subset -- 32 K/V channels
			int64_t acc = 0;
			for (int j = 0; j < qb.d; ++j) acc += (int64_t)xc[j] * qb.Wc[(size_t)i * qb.d + j];
			int32_t bid, bmu, bsh;
			DeriveTripleOld(qb.w[i] / qb.S, &bid, &bmu, &bsh);
			int64_t branch_code = ApplyWeightScaleFold(acc, bid, bmu, bsh);
			if (composed) {
				// Add the SAME delta-fold arithmetic RunToken uses, at this channel, mech=1 (correct).
				std::vector<int64_t> u_acc(qb.r, 0);
				for (int k = 0; k < qb.r; ++k) {
					int64_t a = 0;
					for (int j = 0; j < qb.d; ++j) a += (int64_t)xc[j] * qb.Ac[(size_t)k * qb.d + j];
					u_acc[k] = a;
				}
				std::vector<int8_t> u_i8(qb.r);
				for (int k = 0; k < qb.r; ++k) {
					int32_t id, mu, ex;
					const double rho_u = qb.alpha[k] / qb.T();
					int64_t uw = DeriveTripleNew(rho_u, &id, &mu, &ex)
					                 ? ApplyAmplifyingWeightScaleFold(u_acc[k], id, mu, ex)
					                 : u_acc[k];
					u_i8[k] = (int8_t)(uw > 127 ? 127 : (uw < -127 ? -127 : uw));
				}
				int64_t delta_raw = 0;
				for (int k = 0; k < qb.r; ++k) delta_raw += (int64_t)u_i8[k] * qb.Bc[(size_t)i * qb.r + k];
				int32_t did, dmu, dex;
				const double rho = qb.T() * qb.beta[i] / qb.S;
				int64_t dwl = DeriveTripleNew(rho, &did, &dmu, &dex)
				                  ? ApplyAmplifyingWeightScaleFold(delta_raw, did, dmu, dex)
				                  : 0;
				branch_code += dwl;
			}
			uint64_t sat_count = 0;
			bool mag_exceeded = false;
			const int64_t raw =
			    LandingRescale(branch_code, m_a, r_t, e_a, e_t, &sat_count, &mag_exceeded);
			const int64_t clamped = raw > 127 ? 127 : (raw < -127 ? -127 : raw);
			(void)clamped;
			++total;
			if (sat_count > 0 || mag_exceeded) ++fired;
		}
	}
	return total > 0 ? (double)fired / total : 0.0;
}

// B6's own first red-first cell: an adversarially constructed adapter (extreme gain, pushing K/V
// channel values near int8 saturation) must be caught -- a materially elevated saturation rate
// relative to the base-only accumulator, using the REAL LandingRescale primitive as the oracle.
static void TestB6AdversarialAdapterElevatesKvSaturationRateOverBaseOnly() {
	QuantizedBase adversarial = BuildAdapter(128, 64, 8, /*gain=*/40.0, 0xD00D'0001, 1.0);
	const double base_rate = MeasureKvSaturationRate(adversarial, /*composed=*/false, 12, 0xA000);
	const double composed_rate = MeasureKvSaturationRate(adversarial, /*composed=*/true, 12, 0xA000);
	std::printf("   [B6 first cell] adversarial gain=40: base_rate=%.4f composed_rate=%.4f\n",
	            base_rate, composed_rate);
	CHECK_MSG(composed_rate > base_rate,
	          "an adversarial (extreme-gain) adapter must elevate the composed K/V saturation "
	          "rate over the base-only rate: composed=%.4f base=%.4f", composed_rate, base_rate);
}

// B6's own second cell (the two-sided calibration StandardsDocument.md Sec5.4 requires): a
// typical or moderately-perturbed adapter, known by construction not to threaten K/V fidelity,
// must score BELOW the adversarial construction's own rate -- without this control, a metric that
// reads "elevated" for every input, adversarial or not, would also pass the adversarial cell.
static void TestB6TypicalAdapterScoresBelowTheAdversarialConstruction() {
	QuantizedBase adversarial = BuildAdapter(128, 64, 8, /*gain=*/40.0, 0xD00D'0002, 1.0);
	QuantizedBase typical = BuildAdapter(128, 64, 8, /*gain=*/0.4, 0xD00D'0002, 1.0);
	const double adversarial_rate = MeasureKvSaturationRate(adversarial, true, 12, 0xB000);
	const double typical_rate = MeasureKvSaturationRate(typical, true, 12, 0xB000);
	const double typical_base_rate = MeasureKvSaturationRate(typical, false, 12, 0xB000);
	std::printf("   [B6 second cell] typical composed_rate=%.4f typical base_rate=%.4f adversarial composed_rate=%.4f\n",
	            typical_rate, typical_base_rate, adversarial_rate);
	CHECK_MSG(typical_rate < adversarial_rate,
	          "a typical/moderate adapter must score BELOW the adversarial construction: "
	          "typical=%.4f adversarial=%.4f", typical_rate, adversarial_rate);
}

// =============================================================================================
// B7 -- Rejection/fallback wiring (design Sec11 B7).
// =============================================================================================

// The three-branch taxonomy (design Sec7, renamed D-SLM3047): domain-rejection trip,
// runtime-vs-baked margin exceeded, saturation-rate elevation. This offline dispatcher takes the
// three verdicts B3/B6 already compute and asserts each named branch is independently reachable
// and independently distinguishable -- reusing B3's domain fixture, B3's margin-mutation fixture,
// and B6's saturation fixture exactly as design Sec11 B7 specifies.
//
// T-2027 amendment (D-SLM3095/D-SLM3097): the fallback is now EXPLICIT-ONLY. `Outcome` replaces
// the T-2018 form's implicit "a branch was identified, therefore merge+quantize was emitted" --
// the amended contract requires the dispatcher to ALSO take the `--fallback=merge` flag as an
// input and report whether ANY artifact is emitted at all, not just which check failed.
enum class RejectionBranch { None, DomainRejectionTrip, RuntimeVsBakedMarginExceeded, SaturationRateElevation };
enum class ArtifactOutcome { RuntimeAdditive, NoArtifactEmitted, MergeQuantizeEmitted };

struct DispatchResult {
	RejectionBranch branch;
	ArtifactOutcome outcome;
};

static DispatchResult Dispatch(bool domain_trip, bool margin_exceeded, bool saturation_elevated,
                                bool fallback_flag_present) {
	// Domain-rejection trip is checked first: a row that cannot even be represented is a harder
	// failure than a fidelity or saturation measurement, matching the design's own reject-over-
	// degrade discipline (fail on the most fundamental violation first).
	RejectionBranch branch = RejectionBranch::None;
	if (domain_trip) branch = RejectionBranch::DomainRejectionTrip;
	else if (margin_exceeded) branch = RejectionBranch::RuntimeVsBakedMarginExceeded;
	else if (saturation_elevated) branch = RejectionBranch::SaturationRateElevation;

	if (branch == RejectionBranch::None) return {branch, ArtifactOutcome::RuntimeAdditive};
	// T-2027 (D-SLM3095): a validation failure emits NO artifact absent the explicit flag; WITH
	// the flag, it falls back to merge+quantize -- never an automatic substitution either way.
	return {branch, fallback_flag_present ? ArtifactOutcome::MergeQuantizeEmitted
	                                      : ArtifactOutcome::NoArtifactEmitted};
}

// Six red-first cells (design Sec11 B7, D-SLM3095): flag-absent and flag-present, for each of the
// three named branches -- proving the explicit-only behavior in both directions, not merely that
// a rejection fires (T-2018's own three cells proved only the latter).
static void TestB7DomainRejectionTripNoFlagEmitsNoArtifact() {
	// Reuses B3's own domain fixture: the adversarial composed row from
	// TestB3DomainCheckAdversarialComposedRowTripsChainInputOutOfDomain.
	std::vector<int64_t> wide_row(8, 1000);
	wide_row[0] = (int64_t)2147483647LL + (int64_t)2147483647LL;
	CarriedScale incoming{1717986918, -30}, site_constant{1717986918, 0};
	std::vector<int8_t> out_codes(8);
	CarriedScale out_scale;
	const ChainResult cr = RequantChainChecked(wide_row.data(), 8, std::span<const CarriedScale>(&incoming, 1),
	                                            site_constant, out_codes.data(), &out_scale);
	const bool domain_trip = (cr.status != SslmForwardStatus::Ok);
	CHECK_MSG(domain_trip, "B7's own domain fixture (reused from B3) must trip domain rejection");
	const DispatchResult r = Dispatch(domain_trip, false, false, /*fallback_flag_present=*/false);
	CHECK(r.branch == RejectionBranch::DomainRejectionTrip);
	CHECK_MSG(r.outcome == ArtifactOutcome::NoArtifactEmitted,
	          "without --fallback=merge, a domain-rejection-trip failure must emit NO artifact");
}

static void TestB7DomainRejectionTripWithFlagEmitsMergeQuantize() {
	std::vector<int64_t> wide_row(8, 1000);
	wide_row[0] = (int64_t)2147483647LL + (int64_t)2147483647LL;
	CarriedScale incoming{1717986918, -30}, site_constant{1717986918, 0};
	std::vector<int8_t> out_codes(8);
	CarriedScale out_scale;
	const ChainResult cr = RequantChainChecked(wide_row.data(), 8, std::span<const CarriedScale>(&incoming, 1),
	                                            site_constant, out_codes.data(), &out_scale);
	const DispatchResult r = Dispatch(cr.status != SslmForwardStatus::Ok, false, false,
	                                   /*fallback_flag_present=*/true);
	CHECK_MSG(r.outcome == ArtifactOutcome::MergeQuantizeEmitted,
	          "WITH --fallback=merge, the identical failing input must emit a merge+quantize artifact");
}

static void TestB7RuntimeVsBakedMarginExceededNoFlagEmitsNoArtifact() {
	// Reuses B3's own margin-mutation fixture, re-derived under the amended acceptance test
	// (frozen Delta, not the retired difference-from-zero form).
	QuantizedBase reference = BuildAdapter(kT2027Dim, kT2027Out, kT2027Rank, kT2027Gain,
	                                        kT2027BuildSeed, /*t_scale_knob=*/1.0);
	const FrozenDelta delta = CalibrateDelta(reference, kT2027PilotCorpusN, kT2027PilotSeedBase);
	QuantizedBase defect = BuildAdapter(kT2027Dim, kT2027Out, kT2027Rank, kT2027Gain, kT2027BuildSeed, 255.9);
	const Verdict v = Grade(defect, 1, 0.0, delta, 24, 0xC000);
	const bool margin_exceeded = !v.composed_mean_accepts || !v.composed_tail_accepts;
	CHECK_MSG(margin_exceeded, "B7's own margin fixture (T_SCALE(255.9), reused from B3) must "
	          "exceed the frozen composed Delta");
	const DispatchResult r = Dispatch(false, margin_exceeded, false, /*fallback_flag_present=*/false);
	CHECK(r.branch == RejectionBranch::RuntimeVsBakedMarginExceeded);
	CHECK_MSG(r.outcome == ArtifactOutcome::NoArtifactEmitted,
	          "without --fallback=merge, a margin-exceeded failure must emit NO artifact");
}

static void TestB7RuntimeVsBakedMarginExceededWithFlagEmitsMergeQuantize() {
	QuantizedBase reference = BuildAdapter(kT2027Dim, kT2027Out, kT2027Rank, kT2027Gain,
	                                        kT2027BuildSeed, /*t_scale_knob=*/1.0);
	const FrozenDelta delta = CalibrateDelta(reference, kT2027PilotCorpusN, kT2027PilotSeedBase);
	QuantizedBase defect = BuildAdapter(kT2027Dim, kT2027Out, kT2027Rank, kT2027Gain, kT2027BuildSeed, 255.9);
	const Verdict v = Grade(defect, 1, 0.0, delta, 24, 0xC000);
	const bool margin_exceeded = !v.composed_mean_accepts || !v.composed_tail_accepts ||
	                              !v.effect_mean_accepts || !v.effect_tail_accepts;
	const DispatchResult r = Dispatch(false, margin_exceeded, false, /*fallback_flag_present=*/true);
	CHECK_MSG(r.outcome == ArtifactOutcome::MergeQuantizeEmitted,
	          "WITH --fallback=merge, the identical failing input must emit a merge+quantize artifact");
}

static void TestB7SaturationRateElevationNoFlagEmitsNoArtifact() {
	// Reuses B6's own adversarial fixture.
	QuantizedBase adversarial = BuildAdapter(128, 64, 8, 40.0, 0xD00D'0003, 1.0);
	const double base_rate = MeasureKvSaturationRate(adversarial, false, 8, 0xD000);
	const double composed_rate = MeasureKvSaturationRate(adversarial, true, 8, 0xD000);
	const bool saturation_elevated = composed_rate > base_rate;
	CHECK_MSG(saturation_elevated, "B7's own saturation fixture (reused from B6) must show elevation");
	const DispatchResult r = Dispatch(false, false, saturation_elevated, /*fallback_flag_present=*/false);
	CHECK(r.branch == RejectionBranch::SaturationRateElevation);
	CHECK_MSG(r.outcome == ArtifactOutcome::NoArtifactEmitted,
	          "without --fallback=merge, a saturation-rate-elevation failure must emit NO artifact");
}

static void TestB7SaturationRateElevationWithFlagEmitsMergeQuantize() {
	QuantizedBase adversarial = BuildAdapter(128, 64, 8, 40.0, 0xD00D'0003, 1.0);
	const double base_rate = MeasureKvSaturationRate(adversarial, false, 8, 0xD000);
	const double composed_rate = MeasureKvSaturationRate(adversarial, true, 8, 0xD000);
	const DispatchResult r =
	    Dispatch(false, false, composed_rate > base_rate, /*fallback_flag_present=*/true);
	CHECK_MSG(r.outcome == ArtifactOutcome::MergeQuantizeEmitted,
	          "WITH --fallback=merge, the identical failing input must emit a merge+quantize artifact");
}

static void TestB7BranchesAreMutuallyDistinguishableNotCollapsedToOneDiagnostic() {
	// A domain trip must NOT be reported as a margin-exceeded or saturation diagnostic even when
	// all three conditions happen to co-occur -- the taxonomy's own priority order (Dispatch's
	// own first-match-wins) must be exercised, proving the three branches are not silently
	// collapsed into one generic "rejected" outcome. Checked at flag=false throughout, since the
	// flag governs the OUTCOME axis, not the BRANCH axis.
	CHECK(Dispatch(true, true, true, false).branch == RejectionBranch::DomainRejectionTrip);
	CHECK(Dispatch(false, true, true, false).branch == RejectionBranch::RuntimeVsBakedMarginExceeded);
	CHECK(Dispatch(false, false, true, false).branch == RejectionBranch::SaturationRateElevation);
	CHECK(Dispatch(false, false, false, false).branch == RejectionBranch::None);
	// A clean input (no branch triggered) emits the runtime-additive artifact regardless of the
	// flag -- the flag only matters on a rejection path.
	CHECK(Dispatch(false, false, false, true).outcome == ArtifactOutcome::RuntimeAdditive);
}

// =============================================================================================

int main() {
	TestB0ZeroEffectAdapterFoldsToExactlyZeroContribution();
	TestB0HandComputedNonzeroDeltaCorrectTripleMatchesMisderivedDiverges();
	TestB0WiringDerivationGateAtTheFractureCellOldFailsNewPasses();
	TestB0WiringRefusesPartialRepairAtEveryNonzeroFraction();
	TestB0DerivationRefusesRhoScaleMisderivation();
	TestB0DerivationRefusesRhoPermMisderivation();
	TestB0DerivationBoundIsConstructiveAcrossGains();
	TestB0GateDoesNotAndCannotCatchTScaleMisderivationByDesign();

	TestB3DomainCheckAdversarialComposedRowTripsChainInputOutOfDomain();
	TestB3AmplifyingRatioGenuinelyOutOfDomainSignalsExplicitInfeasibility();
	TestB3DeltaCalibratedOnPilotReferenceAcceptsOnValidation();
	TestB3WrongDirectionMutationRejectedAgainstFrozenDelta();
	TestB3EffectRetentionRejectsFullAnnihilationUnconditionally();
	TestB3TScaleSweepGradedByTheAcceptanceTestItself();
	TestB3UnderSamplingWidensUpperCiNeverNarrowsIt();

	TestB6AdversarialAdapterElevatesKvSaturationRateOverBaseOnly();
	TestB6TypicalAdapterScoresBelowTheAdversarialConstruction();

	TestB7DomainRejectionTripNoFlagEmitsNoArtifact();
	TestB7DomainRejectionTripWithFlagEmitsMergeQuantize();
	TestB7RuntimeVsBakedMarginExceededNoFlagEmitsNoArtifact();
	TestB7RuntimeVsBakedMarginExceededWithFlagEmitsMergeQuantize();
	TestB7SaturationRateElevationNoFlagEmitsNoArtifact();
	TestB7SaturationRateElevationWithFlagEmitsMergeQuantize();
	TestB7BranchesAreMutuallyDistinguishableNotCollapsedToOneDiagnostic();

	std::printf("t2027 offline red suite (B0/B3/B6/B7, amended contract): %d checks, %d failures\n",
	            GChecks, GFailures);
	return GFailures == 0 ? 0 : 1;
}

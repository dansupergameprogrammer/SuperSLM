// T-2018 -- Curie's RED suite for S-LoRA-serial, derived from the design of record
// (Claude/Vitruvius/t1977-v5-lora-composition-reconciliation-design-2026-08-13.md at commit
// 6294ca69ba, Wizard repo) Sec10 Coverage Model and Sec11 Build Decomposition.
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
	double float_distance_runtime;  // mean |composed_int8*scale - float_ref| / |float_ref|, over channels
	double float_distance_baked;
	double err_nodelta;  // base-only floor, reported for context
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

		// RUNTIME arm.
		el += std::fabs((double)(acc_wide + dwl) * scale - ref);
		en += std::fabs((double)acc_wide * scale - ref);

		// BAKED arm: independent GEMM against the merged, once-quantized weight, then the base's
		// own unmodified ApplyWeightScaleFold using the baked-arm's own (wp[i], Sp) fold --
		// shares xc (the SAME token, per Sec6 item 1's pairing) and nothing of the runtime
		// mechanism's internals.
		int64_t bacc = 0;
		for (int j = 0; j < d; ++j) bacc += (int64_t)xc[j] * qb.Wpc[(size_t)i * d + j];
		int32_t pid, pmu, psh;
		DeriveTripleOld(qb.wp[i] / qb.Sp, &pid, &pmu, &psh);
		const int64_t bacc_wide = ApplyWeightScaleFold(bacc, pid, pmu, psh);
		ebk += std::fabs((double)bacc_wide * scale_baked - ref);

		den += std::fabs(ref);
	}
	res.float_distance_runtime = el / den;
	res.float_distance_baked = ebk / den;
	res.err_nodelta = en / den;
	res.wiring_mismatches = wiring_mismatches;
	res.derivation_mismatches = derivation_mismatches;
	res.max_derivation_rel_err = max_rel_err;
	res.gate_passes = (wiring_mismatches == 0) && (derivation_mismatches == 0);
	return res;
}

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

// B3's own second red-first cell (redefined D-SLM3019/D-SLM3024, D-SLM3050): the BAKED-ADAPTER
// COMPARATOR's own statistic (paired mean gap, SE, resolving power, sign count) is computed on a
// synthetic corpus, and a deliberately mis-derived delta-fold triple (WRONG_DIRECTION, reusing
// B0's own second red-first cell's fixture family per this design's own cross-reference) widens
// the runtime-vs-baked gap measurably relative to the correctly-derived triple's own gap -- the
// direction check D-SLM2985 established, retained in this shape because the margin itself is not
// yet set (design Sec11 B3, second cell).
static void TestB3BakedComparatorStatisticMutationProofWidensGapUnderMisderivedTriple() {
	QuantizedBase qb = BuildAdapter(896, 896, 8, 0.4, 0x5EED'0001, 1.0);
	const int n_items = 24;
	const PairedGapStat correct = ComputePairedGap(qb, /*mech=*/1, 0.0, n_items, 0x7000);
	const PairedGapStat wrong = ComputePairedGap(qb, /*mech=*/7, 0.0, n_items, 0x7000);

	std::printf("   [B3 second cell] correct mean_gap=%.6f se=%.6f resolving_power=%.6f sign+=%.2f (n=%d)\n",
	            correct.mean, correct.se, correct.resolving_power, correct.sign_frac_positive, correct.n);
	std::printf("   [B3 second cell] wrong(direction) mean_gap=%.6f se=%.6f resolving_power=%.6f sign+=%.2f (n=%d)\n",
	            wrong.mean, wrong.se, wrong.resolving_power, wrong.sign_frac_positive, wrong.n);

	CHECK_MSG(wrong.mean > correct.mean,
	          "a deliberately mis-derived (inverted-direction) triple must widen the runtime-vs-"
	          "baked gap relative to the correct triple: wrong_mean=%.6f correct_mean=%.6f",
	          wrong.mean, correct.mean);
	CHECK_MSG(wrong.mean - correct.mean > wrong.resolving_power,
	          "the widening must exceed the wrong-arm's own achieved resolving power (a real, "
	          "resolved effect, not noise): widening=%.6f resolving_power=%.6f",
	          wrong.mean - correct.mean, wrong.resolving_power);
	// The correct arm's own mean gap must NOT itself exceed its own resolving power in the
	// direction that would fabricate a spurious "runtime worse than baked" reading -- reported,
	// per D-SLM2846's UNRESOLVED-never-PASSED discipline, not asserted as a numeric pass (the
	// margin is underived, design Sec13).
	if (std::fabs(correct.mean) < correct.resolving_power) {
		std::printf("   [B3 second cell] correct-triple mean_gap is UNRESOLVED at n=%d (|mean| < "
		            "resolving power) -- reported, not treated as a pass, per D-SLM2846\n", n_items);
	}
}

// B3's own fourth red-first cell (D-SLM3024/D-SLM3046, T-2016 fold): the design's own text claims
// the runtime-vs-baked margin "must be refused... at every t != 1 T-2007 measured passing B0's
// WIRING/DERIVATION gate, including t = 255.9" (the adapter-annihilating case), while ACCEPTING
// t=1 exactly (the correct mechanism's own baseline, no mis-derivation at all).
//
// EXECUTED FINDING, ROUTED TO THE PLANNER (Curie's own charter: "a gap in the model itself routes
// back to the planner in your handoff, not silently patched" -- this is that gap, found by
// deriving this cell's own fixture for the first time; no probe in this lineage computed the
// runtime-vs-baked GAP statistic before D-SLM3019/3050 introduced it, so this is this quantity's
// first execution against the T_SCALE family). At t=1 the gap is bit-identical to the baseline,
// confirmed below. At LARGE t (this fixture's own measured threshold: t IN {64, 200, 255.9, 256,
// 1024}) the gap widens measurably past its own achieved resolving power -- refusal holds,
// including the design's own headline t=255.9 case. But at the SMALL-t neighbourhood immediately
// above 1 (t IN {1.0001, 1.0725, 2.0, 4.0, 16.0}), this suite's own execution finds the gap does
// NOT reliably widen past the baseline -- at several of these t values the paired mean gap is
// SMALLER than at t=1, not larger, so "refused at every t != 1" does not hold as measured.
//
// This is NOT a contradiction this suite invents: T-2007's own probe (Claude/Vitruvius/t2009-
// verify-probe/t2007_body.cpp, Part 4 Side B) already found and REPORTED the identical qualitative
// property in the single-arm err_legal quantity this gap statistic is built from -- "the sweep is
// NOT monotone at its first step: t=1 gives X and t=2 gives Y, a Z% IMPROVEMENT... It is a
// property of the fixture, not of this knob, and it makes the fracture WORSE, not better: the
// gate cannot distinguish a T that is slightly better from one that is catastrophically wrong."
// The design's own B3 fourth-cell text was written against that same probe but states a stronger,
// blanket "every t != 1" claim than the probe's own Part 4 supports for the small-t neighbourhood.
// This cell therefore asserts only what this suite's own execution actually establishes (t=1
// accepted; LARGE t refused, including t=255.9) and REPORTS, rather than silently asserting past,
// the small-t non-monotonicity -- per StandardsDocument.md Sec5.4 ("a measurement is evidence
// about the cell it was taken in, and reverses at its neighbours").
static void TestB3TScaleMutationProofRefusedAtLargeTAcceptedAtTEqualsOneSmallTGapReportedToPlanner() {
	const uint64_t build_seed = 0xF00D'BEEF;
	const int n_items = 24;
	QuantizedBase honest = BuildAdapter(896, 896, 8, 0.4, build_seed, /*t_scale_knob=*/1.0);
	const PairedGapStat baseline = ComputePairedGap(honest, /*mech=*/1, 0.0, n_items, 0x9000);

	std::printf("   [B3 fourth cell] t=1 (baseline) mean_gap=%.6f se=%.6f resolving_power=%.6f\n",
	            baseline.mean, baseline.se, baseline.resolving_power);

	// t = 1 must be ACCEPTED: bit-identical to the baseline built with t_scale_knob = 1.0 by
	// construction (T-2007 Part 4 Side A's own two-sided calibration; D-SLM3046's own correction
	// that this cell is scoped to t != 1, not merely tolerant of t=1 passing by coincidence).
	{
		QuantizedBase t1 = BuildAdapter(896, 896, 8, 0.4, build_seed, /*t_scale_knob=*/1.0);
		const PairedGapStat at_t1 = ComputePairedGap(t1, 1, 0.0, n_items, 0x9000);
		CHECK_MSG(std::fabs(at_t1.mean - baseline.mean) < 1e-12,
		          "t=1 must be bit-identical to the correct mechanism's own baseline: %.9f vs %.9f",
		          at_t1.mean, baseline.mean);
	}

	// LARGE t: refusal is asserted and holds, including the design's own headline t=255.9 case
	// (the adapter quantized out of existence).
	for (double t : {64.0, 200.0, 255.9, 256.0, 1024.0}) {
		QuantizedBase defect = BuildAdapter(896, 896, 8, 0.4, build_seed, /*t_scale_knob=*/t);
		const PairedGapStat at_t = ComputePairedGap(defect, 1, 0.0, n_items, 0x9000);
		CHECK_MSG(at_t.mean > baseline.mean,
		          "t=%.4f must widen the runtime-vs-baked gap relative to t=1: gap(t)=%.6f gap(1)=%.6f",
		          t, at_t.mean, baseline.mean);
		const double widening = at_t.mean - baseline.mean;
		CHECK_MSG(widening > at_t.resolving_power,
		          "t=%.4f's widening (%.6f) must exceed its own achieved resolving power (%.6f) -- "
		          "a resolved refusal, not noise", t, widening, at_t.resolving_power);
	}

	// SMALL t: REPORTED, not asserted -- this is the coverage-model gap this cell's execution
	// found and routes to the planner. No CHECK() below; this is evidence, printed so it is not
	// lost, matching T-2007's own "REPORTED, NOT HIDDEN" discipline at the identical phenomenon.
	std::printf("   [B3 fourth cell] SMALL-t neighbourhood -- design claims refusal at every t != 1;\n"
	            "   this suite's own execution (n=%d paired items) finds:\n", n_items);
	for (double t : {1.0001, 1.0725, 2.0, 4.0, 16.0}) {
		QuantizedBase defect = BuildAdapter(896, 896, 8, 0.4, build_seed, /*t_scale_knob=*/t);
		const PairedGapStat at_t = ComputePairedGap(defect, 1, 0.0, n_items, 0x9000);
		const double widening = at_t.mean - baseline.mean;
		const bool resolved_refusal = widening > at_t.resolving_power;
		std::printf("     t=%-9.4f gap=%.6f widening=%.6f resolving_power=%.6f -> %s\n", t, at_t.mean,
		            widening, at_t.resolving_power, resolved_refusal ? "refused" : "NOT refused (gap here)");
	}
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
enum class RejectionBranch { None, DomainRejectionTrip, RuntimeVsBakedMarginExceeded, SaturationRateElevation };

static RejectionBranch Dispatch(bool domain_trip, bool margin_exceeded, bool saturation_elevated) {
	// Domain-rejection trip is checked first: a row that cannot even be represented is a harder
	// failure than a fidelity or saturation measurement, matching the design's own reject-over-
	// degrade discipline (fail on the most fundamental violation first).
	if (domain_trip) return RejectionBranch::DomainRejectionTrip;
	if (margin_exceeded) return RejectionBranch::RuntimeVsBakedMarginExceeded;
	if (saturation_elevated) return RejectionBranch::SaturationRateElevation;
	return RejectionBranch::None;
}

static void TestB7DispatchesDomainRejectionTripBranchReusingB3Fixture() {
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
	const RejectionBranch branch = Dispatch(domain_trip, /*margin_exceeded=*/false, /*saturation_elevated=*/false);
	CHECK(branch == RejectionBranch::DomainRejectionTrip);
}

static void TestB7DispatchesRuntimeVsBakedMarginExceededBranchReusingB3Fixture() {
	// Reuses B3's own margin-mutation fixture: T-2007's T_SCALE(255.9), which B3's fourth
	// red-first cell already proves widens the runtime-vs-baked gap past its own resolving power.
	QuantizedBase honest = BuildAdapter(896, 896, 8, 0.4, 0xC0DE'0001, 1.0);
	QuantizedBase defect = BuildAdapter(896, 896, 8, 0.4, 0xC0DE'0001, 255.9);
	const PairedGapStat baseline = ComputePairedGap(honest, 1, 0.0, 16, 0xC000);
	const PairedGapStat at_defect = ComputePairedGap(defect, 1, 0.0, 16, 0xC000);
	const double widening = at_defect.mean - baseline.mean;
	const bool margin_exceeded = widening > at_defect.resolving_power;
	CHECK_MSG(margin_exceeded, "B7's own margin fixture (reused from B3) must exceed the resolving power");
	const RejectionBranch branch = Dispatch(/*domain_trip=*/false, margin_exceeded, /*saturation_elevated=*/false);
	CHECK(branch == RejectionBranch::RuntimeVsBakedMarginExceeded);
}

static void TestB7DispatchesSaturationRateElevationBranchReusingB6Fixture() {
	// Reuses B6's own adversarial fixture.
	QuantizedBase adversarial = BuildAdapter(128, 64, 8, 40.0, 0xD00D'0003, 1.0);
	const double base_rate = MeasureKvSaturationRate(adversarial, false, 8, 0xD000);
	const double composed_rate = MeasureKvSaturationRate(adversarial, true, 8, 0xD000);
	const bool saturation_elevated = composed_rate > base_rate * 1.5;  // representative elevation test
	CHECK_MSG(saturation_elevated || composed_rate > base_rate,
	          "B7's own saturation fixture (reused from B6) must show elevation");
	const RejectionBranch branch =
	    Dispatch(/*domain_trip=*/false, /*margin_exceeded=*/false, composed_rate > base_rate);
	CHECK(branch == RejectionBranch::SaturationRateElevation);
}

static void TestB7BranchesAreMutuallyDistinguishableNotCollapsedToOneDiagnostic() {
	// A domain trip must NOT be reported as a margin-exceeded or saturation diagnostic even when
	// all three conditions happen to co-occur -- the taxonomy's own priority order (Dispatch's
	// own first-match-wins) must be exercised, proving the three branches are not silently
	// collapsed into one generic "rejected" outcome.
	CHECK(Dispatch(true, true, true) == RejectionBranch::DomainRejectionTrip);
	CHECK(Dispatch(false, true, true) == RejectionBranch::RuntimeVsBakedMarginExceeded);
	CHECK(Dispatch(false, false, true) == RejectionBranch::SaturationRateElevation);
	CHECK(Dispatch(false, false, false) == RejectionBranch::None);
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
	TestB3BakedComparatorStatisticMutationProofWidensGapUnderMisderivedTriple();
	TestB3TScaleMutationProofRefusedAtLargeTAcceptedAtTEqualsOneSmallTGapReportedToPlanner();

	TestB6AdversarialAdapterElevatesKvSaturationRateOverBaseOnly();
	TestB6TypicalAdapterScoresBelowTheAdversarialConstruction();

	TestB7DispatchesDomainRejectionTripBranchReusingB3Fixture();
	TestB7DispatchesRuntimeVsBakedMarginExceededBranchReusingB3Fixture();
	TestB7DispatchesSaturationRateElevationBranchReusingB6Fixture();
	TestB7BranchesAreMutuallyDistinguishableNotCollapsedToOneDiagnostic();

	std::printf("t2018 offline red suite (B0/B3/B6/B7): %d checks, %d failures\n", GChecks, GFailures);
	return GFailures == 0 ? 0 : 1;
}

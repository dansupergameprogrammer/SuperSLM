// Loki strike probe, T-1655/T-1656 §2 promise, limb 2.
//
// Claim under attack (design record §2): "T-1656's red suite uses synthetic
// fixtures with in-bound bias codes ... and none of those fixtures are affected
// by where the load-time bound sits. T-1656 is real, buildable, and testable
// today, independently of T-1657."
//
// The bound kBia1MagnitudeBound == INT32_MAX (src/model.cpp:736-745) is derived
// from "R_a's maximum over the C19 reciprocal's own domain is 2^32, so keeping
// B[j]*R_a inside int64 requires |B[j]| <= INT32_MAX".  R_a <= 2^32 holds only
// for a CANONICAL mantissa Dn in [2^30, 2^31).  Design §5.3's inserted call site
// computes r_a = CarriedScaleReciprocal(in_scale.m) where in_scale.m is a
// MID-COMPOSITION carried mantissa, which design §3 itself states is required
// only to fit int32_t, not to be canonical.
//
// This probe runs the real, unmodified shipped primitives at 9cd2b93 and asks:
// can a legal fixture -- one whose every value passes every check the design
// specifies -- drive B*R_a out of int64 with an IN-BOUND bias code?
//
// Build: probe_build.bat probe_bias_reciprocal.cpp

#include <cstdint>
#include <intrin.h>
#include <cstdio>
#include <string>
#include <vector>

#include "superslm/checked_chain_funnel.h"
#include "superslm/forward_sites.h"
#include "superslm/intmath.h"

using namespace superslm;

static constexpr int64_t kBia1MagnitudeBound = INT32_MAX;  // src/model.cpp:745
static constexpr int64_t kBiasQFormat = 30;                // design §5.2

// Detect signed-int64 overflow of b*r without executing it (MSVC has no
// __int128; _mul128 gives the exact 128-bit signed product).
static bool MulOverflows64(int64_t a, int64_t b, long double* mag) {
	int64_t hi = 0;
	const int64_t lo = _mul128(a, b, &hi);
	*mag = static_cast<long double>(hi) * 18446744073709551616.0L +
	       static_cast<long double>(static_cast<uint64_t>(lo));
	return hi != (lo >> 63);
}

int main() {
	std::printf("=== Part A: CarriedScaleReciprocal over the domain the int32 door admits ===\n");
	std::printf("%14s %22s %10s %26s\n", "in_scale.m", "r_a = recip(m)", "log2(r_a)", "|INT32_MAX * r_a| fits?");
	const int64_t ms[] = {1, 2, 4, 8, 1024, int64_t{1} << 20, int64_t{1} << 29,
	                      int64_t{1} << 30, (int64_t{1} << 31) - 1};
	for (int64_t m : ms) {
		const int64_t r = CarriedScaleReciprocal(m);
		long double mag = 0;
		const bool ov = MulOverflows64(kBia1MagnitudeBound, r, &mag);
		double l2 = 0;
		for (int64_t t = r; t > 1; t >>= 1) l2 += 1;
		std::printf("%14lld %22lld %10.0f %26s\n", (long long)m, (long long)r, l2,
		            ov ? "NO -- int64 OVERFLOW" : "yes");
	}

	std::printf("\n=== Part B: reachability -- can a legal fixture present a non-canonical in_scale? ===\n");
	// RmsNormSite produces `normed_scale`, which is exactly the `in_scale`
	// ProjectAndFunnel's q_proj call receives (forward_sites.cpp:1118-1126), and
	// which design §5.3 feeds to CarriedScaleReciprocal.  Its `site_constant`
	// comes straight out of LayerWeights (`lw.attn_norm_site_constant`) -- a
	// caller-resolved fixture value whose ONLY check is
	// CarriedScaleMantissaFitsInt32 (RequantChainChecked step 0).
	const size_t hidden_size = 8;
	std::vector<int8_t> h(hidden_size), out(hidden_size);
	std::vector<int32_t> g(hidden_size);
	for (size_t i = 0; i < hidden_size; ++i) {
		h[i] = static_cast<int8_t>(17 + 3 * i);
		g[i] = 1 << 14;  // NORM_FRAC_BITS-style unit gain
	}

	// A fixture-supplied site constant with a small (non-canonical) mantissa.
	// m = 1 is well inside int32_t; e chosen so the composed exponent lands in
	// CheckRoundingDivideByPotExponentDomain's accepted window.
	struct Case { int64_t sc_m; int64_t sc_e; int64_t in_m; int64_t in_e; const char* label; };
	const Case cases[] = {
	    {1, -60, int64_t{1} << 30, -31, "site_constant.m = 1 (legal, non-canonical)"},
	    {2, -60, int64_t{1} << 30, -31, "site_constant.m = 2"},
	    {int64_t{1} << 30, -60, int64_t{1} << 30, -31, "site_constant.m = 2^30 (canonical control)"},
	};

	for (const Case& c : cases) {
		CarriedScale normed_scale{};
		const SslmForwardStatus st =
		    RmsNormSite(h.data(), g.data(), hidden_size, CarriedScale{c.in_m, c.in_e},
		                CarriedScale{c.sc_m, c.sc_e}, out.data(), &normed_scale, "attn_norm", 0,
		                nullptr);
		std::printf("\n-- %s\n", c.label);
		std::printf("   RmsNormSite status      : %s\n", SslmForwardStatusName(st));
		if (st != SslmForwardStatus::Ok) continue;
		std::printf("   normed_scale            : m=%lld  e=%lld   (canonical range is [2^30, 2^31))\n",
		            (long long)normed_scale.m, (long long)normed_scale.e);
		std::printf("   mantissa fits int32_t   : %s\n",
		            (normed_scale.m >= INT32_MIN && normed_scale.m <= INT32_MAX) ? "yes" : "no");

		// === Design §5.3's inserted block, verbatim, on this in_scale. ===
		const CarriedScale in_scale = normed_scale;
		const SslmForwardStatus gate =
		    CheckRoundingDivideByPotExponentDomain(kBiasQFormat, in_scale.e);
		std::printf("   §5.3 exponent gate      : %s  (q_B+62+e_a = %lld)\n",
		            SslmForwardStatusName(gate), (long long)(kBiasQFormat + 62 + in_scale.e));
		if (gate != SslmForwardStatus::Ok) continue;

		const int64_t r_a = CarriedScaleReciprocal(in_scale.m);
		std::printf("   r_a                     : %lld\n", (long long)r_a);

		// Every bias code the CURRENT load-time gate admits is |b| <= INT32_MAX.
		// Find the smallest such in-bound code whose b*r_a leaves int64.
		int64_t first_bad = 0;
		for (int64_t b = 1; b <= kBia1MagnitudeBound; b <<= 1) {
			long double mag = 0;
			if (MulOverflows64(b, r_a, &mag)) { first_bad = b; break; }
		}
		if (first_bad == 0) {
			std::printf("   b*r_a                   : in range for every in-bound bias code\n");
		} else {
			long double mag = 0;
			MulOverflows64(kBia1MagnitudeBound, r_a, &mag);
			std::printf("   b*r_a                   : SIGNED INT64 OVERFLOW (UB) from |b| >= %lld\n",
			            (long long)first_bad);
			std::printf("                             at the bound itself (|b| = %lld) the true "
			            "product is %.4Le\n",
			            (long long)kBia1MagnitudeBound, mag);
			std::printf("                             INT64_MAX is                                "
			            "%.4Le\n",
			            static_cast<long double>(INT64_MAX));
			std::printf("   => an IN-BOUND bias code, at a fixture whose every value passes every\n"
			            "      check design §5.3 specifies, executes signed-overflow UB.\n");

			// The corpse: run the real shipped BiasReconcile at this point and
			// compare what it returns against the value the reference formula
			// defines (computed in exact 128-bit).
			const int64_t b = kBia1MagnitudeBound;
			const int64_t observed = BiasReconcile(b, kBiasQFormat, r_a, in_scale.e);
			int64_t hi = 0;
			const int64_t lo = _mul128(b, r_a, &hi);
			const int k = static_cast<int>(kBiasQFormat + 62 + in_scale.e);
			// exact round-half-away-from-zero of (b*r_a) / 2^k, from the true 128-bit product
			long double truth = (static_cast<long double>(hi) * 18446744073709551616.0L +
			                     static_cast<long double>(static_cast<uint64_t>(lo)));
			for (int t = 0; t < k; ++t) truth /= 2.0L;
			std::printf("   BiasReconcile(b=%lld)   : returns %lld\n", (long long)b,
			            (long long)observed);
			std::printf("   reference value         : %.6Le   (ratio observed/true = %.6Lf)\n", truth,
			            static_cast<long double>(observed) / truth);
		}
	}

	std::printf("\n=== Part C: where does kBia1MagnitudeBound stop protecting? ===\n");
	// The bound's derivation assumes R_a <= 2^32, true only on the canonical
	// mantissa range.  Bisect the largest in_scale.m at which INT32_MAX * r_a
	// still leaves int64.
	int64_t lo_m = 1, hi_m = (int64_t{1} << 31) - 1;
	while (lo_m + 1 < hi_m) {
		const int64_t mid = lo_m + (hi_m - lo_m) / 2;
		long double mag = 0;
		if (MulOverflows64(kBia1MagnitudeBound, CarriedScaleReciprocal(mid), &mag)) {
			lo_m = mid;
		} else {
			hi_m = mid;
		}
	}
	std::printf("largest in_scale.m that OVERFLOWS at |b| = kBia1MagnitudeBound : %lld\n",
	            (long long)lo_m);
	std::printf("smallest in_scale.m that is SAFE                              : %lld\n",
	            (long long)hi_m);
	std::printf("the canonical floor (2^30)                                    : %lld\n",
	            (long long)(int64_t{1} << 30));
	std::printf("=> the bound is sound exactly on the canonical range and nowhere below it.\n");
	std::printf("   in_scale.m in [1, %lld] all pass CarriedScaleMantissaFitsInt32 and all\n",
	            (long long)lo_m);
	std::printf("   break the bound's own derivation.\n");
	return 0;
}

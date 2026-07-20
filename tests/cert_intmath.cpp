// SuperSLM S2.1 exhaustive per-device-tier certifier (SuperSLM_Plan.md §6.8, the
// Popper one-time-per-tier boundary). NOT a per-CI test — a standalone proof run once
// per 1.0 device tier that the C19 reciprocal and C21 normalization are correct over
// their ENTIRE domains, not merely a sampled subset (that sample is the ongoing G-8 CI
// check that rides on top of this).
//
//   C19  DynamicScaleReciprocal(Dn) == round_half_up(2^62 / Dn) for ALL Dn in
//        [2^30, 2^31)  — 1,073,741,824 values.
//   C21  NormalizeScale(D') yields Dn in [2^30, 2^31) with the pinned shift for ALL
//        D' in [1, 2^31]  — 2,147,483,648 values.
//
// The reference round_half_up(2^62/Dn) is computed in pure int64 (no 128-bit needed):
// with N = 2^62, q = N/Dn and rem = N - q*Dn both fit int64 (q <= 2^32, q*Dn <= 2^62),
// and the half-up nudge is `2*rem >= Dn` (2*rem < 2^32). This is an INDEPENDENT oracle,
// not the runtime construction.
//
// Build: build_cert.bat (MSVC).  Run: out\cert_intmath.exe
#include "superslm/intmath.h"

#include <cstdint>
#include <cstdio>

using namespace superslm;

// Independent reference: round_half_up(2^62 / dn), ties toward +infinity.
static int64_t ReferenceReciprocal(int64_t dn) {
	constexpr int64_t kNum = int64_t{1} << 62;
	int64_t q = kNum / dn;
	int64_t rem = kNum - q * dn;  // in [0, dn)
	if (2 * rem >= dn) ++q;        // half-up (tie rounds up)
	return q;
}

int main() {
	constexpr int64_t kLo = int64_t{1} << 30;  // 2^30
	constexpr int64_t kHi = int64_t{1} << 31;  // 2^31 (exclusive for C19)

	// --- C19: exhaustive over Dn in [2^30, 2^31) --------------------------------
	std::printf("C19 DynamicScaleReciprocal: sweeping Dn in [2^30, 2^31) = %lld values...\n",
	            static_cast<long long>(kHi - kLo));
	int64_t c19_mismatches = 0;
	int64_t first_bad_dn = -1, first_got = 0, first_want = 0;
	for (int64_t dn = kLo; dn < kHi; ++dn) {
		int64_t got = DynamicScaleReciprocal(dn);
		int64_t want = ReferenceReciprocal(dn);
		if (got != want) {
			if (c19_mismatches == 0) {
				first_bad_dn = dn;
				first_got = got;
				first_want = want;
			}
			++c19_mismatches;
		}
	}
	if (c19_mismatches == 0) {
		std::printf("C19: PASS — correctly rounded over all %lld normalized values.\n",
		            static_cast<long long>(kHi - kLo));
	} else {
		std::printf("C19: FAIL — %lld mismatches. First at Dn=%lld: got %lld, want %lld.\n",
		            static_cast<long long>(c19_mismatches), static_cast<long long>(first_bad_dn),
		            static_cast<long long>(first_got), static_cast<long long>(first_want));
	}

	// --- C21: exhaustive over D' in [1, 2^31] -----------------------------------
	const int64_t kDMax = int64_t{1} << 31;  // 2^31 inclusive
	std::printf("C21 NormalizeScale: sweeping D' in [1, 2^31] = %lld values...\n",
	            static_cast<long long>(kDMax));
	int64_t c21_mismatches = 0;
	int64_t first_bad_d = -1;
	for (int64_t d = 1; d <= kDMax; ++d) {
		NormalizedScale ns = NormalizeScale(d);
		// Independent oracle: derive the top-bit index by a plain loop (NOT the runtime's
		// clz-based branch), then s = 30 - p and the exact shift. This certifies both the
		// range postcondition (Dn in [2^30, 2^31)) and the exact shift value.
		int p = 0;
		for (int64_t t = d; t > 1; t >>= 1) ++p;  // p = floor(log2(d))
		int expect_s = 30 - p;
		int64_t expect_dn = (expect_s >= 0) ? (d << expect_s) : (d >> 1);
		bool range_ok = (ns.dn >= kLo && ns.dn < kHi);
		bool map_ok = (ns.dn == expect_dn && ns.s == expect_s);
		if (!range_ok || !map_ok) {
			if (c21_mismatches == 0) first_bad_d = d;
			++c21_mismatches;
		}
	}
	if (c21_mismatches == 0) {
		std::printf("C21: PASS — Dn in [2^30, 2^31) for all %lld values of D'.\n",
		            static_cast<long long>(kDMax));
	} else {
		std::printf("C21: FAIL — %lld violations. First at D'=%lld.\n",
		            static_cast<long long>(c21_mismatches), static_cast<long long>(first_bad_d));
	}

	bool ok = (c19_mismatches == 0) && (c21_mismatches == 0);
	std::printf("\nS2.1 exhaustive certification: %s\n", ok ? "PASS" : "FAIL");
	return ok ? 0 : 1;
}

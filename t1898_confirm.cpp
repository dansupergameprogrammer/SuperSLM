// T-1898 strike, confirmation pass -- DISPOSABLE.
// Isolates the region where LandingRescale's `out_magnitude_exceeded_int64`
// -- the signal T-1822 Sec 31.2.2 promotes to "the authoritative gate" -- is
// FALSE while the emitted K code is wrong.
//
// Calls the real LandingRescale. Truth is decided by the Python reference
// (t1898_truth_reference.py) from the documented formula; this pass emits the
// same CSV schema so that reference judges it.

#include <cstdint>
#include <cstdio>
#include <climits>

#include "superslm/forward_sites.h"

using superslm::LandingRescale;

namespace {
constexpr int64_t kCompositionScaleMaxAbsM = INT32_MAX;
constexpr int64_t kKvLandingReciprocalMax = int64_t{1} << 32;

void Emit(std::FILE* out, int64_t bc, int64_t m_a, int64_t r_t, int64_t e_a, int64_t e_t) {
	bool exceeded = false;
	uint64_t sat = 0;
	const int64_t raw = LandingRescale(bc, m_a, r_t, e_a, e_t, &sat, &exceeded);
	std::fprintf(out, "%lld,%lld,%lld,%lld,%lld,%lld,%d\n", (long long)bc, (long long)m_a,
	             (long long)r_t, (long long)e_a, (long long)e_t, (long long)raw, exceeded ? 1 : 0);
}
}  // namespace

int main() {
	std::FILE* out = std::fopen("t1898_confirm.csv", "w");
	if (!out) return 1;
	std::fprintf(out, "bc,m_a,r_t,e_a,e_t,raw,exceeded\n");

	// The k >= 0 (rounding-divide) branch only. k = 62 - (e_a - e_t); sweep the
	// composed exponent over the whole range the load-time domain admits, at
	// every operand scale from 2^0 to the int64 ceiling Option G's own
	// RopeApplyPairWide guard permits.
	for (int b = 0; b <= 63; ++b) {
		const int64_t bc = (b == 63) ? INT64_MAX : (int64_t{1} << b);
		for (int64_t m_a : {int64_t{1}, int64_t{1} << 15, kCompositionScaleMaxAbsM,
		                    -kCompositionScaleMaxAbsM}) {
			for (int64_t r_t : {(int64_t{1} << 31) + 1, int64_t{3} << 30, kKvLandingReciprocalMax}) {
				for (int64_t e_a = -80; e_a <= 39; ++e_a) {
					for (int64_t e_t = -60; e_t <= 39; ++e_t) {
						const int64_t k = 62 - (e_a - e_t);
						if (k < 0 || k > 127) continue;
						Emit(out, bc, m_a, r_t, e_a, e_t);
					}
				}
			}
		}
	}
	std::fclose(out);
	std::printf("wrote t1898_confirm.csv\n");
	return 0;
}

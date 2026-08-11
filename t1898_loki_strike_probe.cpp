// T-1898 adversary strike probe -- DISPOSABLE. Never merges; never enters the
// test suite. Calls the REAL LandingRescale (src/forward/forward_sites.cpp),
// never a re-implementation. Emits raw observations only; the TRUTH reference
// is computed independently, in exact Python integers, from the formula
// LandingRescale's own header comment states -- so the comparison is not two
// copies of one arithmetic.
//
// Target promise, T-1822 design-of-record Sec 31.2.2 (verbatim):
//   (a) "...makes the check correct for ANY artifact's real exponent
//       distribution, not only the one this campaign has measured -- the
//       re-derived `-60` constant becomes a cheap early-exit (skip the
//       per-element check when kv_landing_e_t_k[h] is comfortably below it),
//       never the sole safety boundary."
//   (b) "This removes the '34/56 heads at zero margin' property entirely:
//       correctness no longer depends on any static exponent floor being
//       conservative enough for whatever artifact is loaded -- it depends on
//       LandingRescale's own already-computed per-element result."
//
// Two questions, separated on purpose:
//   Q1 -- is `out_magnitude_exceeded_int64` authoritative? i.e. is there a
//         load-legal input whose emitted K code is wrong while the flag is
//         false? If yes, remedy 2 itself is broken. If no, remedy 2 is sound
//         and the defect must live in what turns it off.
//   Q2 -- what does the exponent band [-60,-25) contain? That is the band the
//         design's re-derived floor admits and the spike's own -25 floor
//         refuses, and it is the band the specified early-exit hands to a
//         path with no per-element check at all.

#include <cstdint>
#include <cstdio>
#include <climits>
#include <random>

#include "superslm/forward_sites.h"

using superslm::LandingRescale;

namespace {

// Load-time domain constants, quoted at source (never re-derived here).
constexpr int64_t kKvLandingExponentMin = -60;                 // model.cpp:810
constexpr int64_t kOptionGFusedExponentMinSpike = -25;         // spike fwd_sites.cpp:68
constexpr int64_t kCompositionScaleMaxAbsM = INT32_MAX;        // model.cpp:617
constexpr int64_t kKvLandingReciprocalMin = (int64_t{1} << 31) + 1;  // model.cpp:753
constexpr int64_t kKvLandingReciprocalMax = int64_t{1} << 32;        // model.cpp:754
constexpr int64_t kEaMin = -80;                                // model.cpp:798
constexpr int64_t kEaMax = 39;                                 // model.cpp:798

void Emit(std::FILE* out, int64_t bc, int64_t m_a, int64_t r_t, int64_t e_a, int64_t e_t) {
	bool exceeded = false;
	uint64_t sat = 0;
	const int64_t raw = LandingRescale(bc, m_a, r_t, e_a, e_t, &sat, &exceeded);
	std::fprintf(out, "%lld,%lld,%lld,%lld,%lld,%lld,%d\n", static_cast<long long>(bc),
	             static_cast<long long>(m_a), static_cast<long long>(r_t),
	             static_cast<long long>(e_a), static_cast<long long>(e_t),
	             static_cast<long long>(raw), exceeded ? 1 : 0);
}

}  // namespace

int main() {
	std::FILE* out = std::fopen("t1898_observations.csv", "w");
	if (out == nullptr) return 1;
	std::fprintf(out, "bc,m_a,r_t,e_a,e_t,raw,exceeded\n");

	// ---- Structured sweep: the corners the domain contract actually admits.
	const int64_t bcs[] = {1,
	                       3,
	                       int64_t{1} << 10,
	                       int64_t{1} << 20,
	                       (int64_t{1} << 27) + 12345,   // the shipped accumulator's own scale
	                       int64_t{1} << 28,             // ~2^27.5 rotated, the design's substituted bound
	                       int64_t{1} << 31,
	                       int64_t{1} << 40,
	                       int64_t{1} << 50,
	                       int64_t{1} << 62,
	                       INT64_MAX,
	                       -1,
	                       -(int64_t{1} << 28),
	                       -(int64_t{1} << 40),
	                       INT64_MIN + 1};
	const int64_t mas[] = {1, -1, 12345, int64_t{1} << 20, kCompositionScaleMaxAbsM,
	                       -kCompositionScaleMaxAbsM};
	const int64_t rts[] = {kKvLandingReciprocalMin, (int64_t{3} << 30), kKvLandingReciprocalMax};
	const int64_t eas[] = {kEaMin, -30, 0, 2, 20, 30, 35, kEaMax};

	for (int64_t bc : bcs)
		for (int64_t m_a : mas)
			for (int64_t r_t : rts)
				for (int64_t e_a : eas)
					for (int64_t e_t = kKvLandingExponentMin; e_t <= 8; ++e_t)
						Emit(out, bc, m_a, r_t, e_a, e_t);

	// ---- Random sweep over the same load-legal domain, to widen the
	// population beyond corners this seat chose by hand.
	std::mt19937_64 rng(20260811u);
	std::uniform_int_distribution<int64_t> d_bc(INT64_MIN + 1, INT64_MAX);
	std::uniform_int_distribution<int64_t> d_ma(-kCompositionScaleMaxAbsM, kCompositionScaleMaxAbsM);
	std::uniform_int_distribution<int64_t> d_rt(kKvLandingReciprocalMin, kKvLandingReciprocalMax);
	std::uniform_int_distribution<int64_t> d_ea(kEaMin, kEaMax);
	std::uniform_int_distribution<int64_t> d_et(kKvLandingExponentMin, 8);
	std::uniform_int_distribution<int> d_shift(0, 63);
	for (int i = 0; i < 400000; ++i) {
		// Draw magnitudes across the whole exponent range, not just the top bits.
		int64_t bc = d_bc(rng) >> d_shift(rng);
		Emit(out, bc, d_ma(rng), d_rt(rng), d_ea(rng), d_et(rng));
	}

	std::fclose(out);
	std::printf("wrote t1898_observations.csv\n");
	return 0;
}

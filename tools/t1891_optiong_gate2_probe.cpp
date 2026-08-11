// T-1891 gate G2 -- wide-primitive correctness, the DATA-GENERATION half.
//
// DISPOSABLE. Branch brunel/t1891-optionG-spike only, never merged. Links against the
// real `superslm::RopeApplyPairWide` (option_g_spike.h / forward_sites.cpp) -- not a
// copy of it -- and drives it over three populations: domain extremes (2^57, 2^58,
// INT64_MAX/INT64_MIN-class magnitudes, plus a truncation-corner case), a randomized
// sweep, and an overflow-forcing corner. Dumps every (input, output, in_domain) tuple
// as one CSV row; `tools/t1891_optiong_gate2_check.py` recomputes the expected value
// independently (Python's arbitrary-precision int, sharing no code with this file's
// C++ SignedU128 facility) and reports pass/fail -- the comparison itself does not
// happen in this program, so this program alone proves nothing; it is half of gate G2.
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#include "superslm/option_g_spike.h"

using superslm::RopeApplyPairWide;
using superslm::RopePairWide;

namespace {

struct Case {
	int64_t x;
	int64_t y;
	int32_t cos_q30;
	int32_t sin_q30;
	const char* label;
};

// ROPE_ONE = 2^30 (Q2.30's own "1.0"), the domain ceiling `ValidateRopeTablesDomain`
// enforces on every table entry this primitive's real caller ever passes (this
// spike's own K-landing call site, forward_sites.cpp) -- so every case below uses
// |cos_q30|/|sin_q30| <= 2^30, matching the primitive's actual production domain
// rather than testing a magnitude no caller can reach.
constexpr int32_t kRopeOne = 1 << 30;

void AppendDomainExtremityCases(std::vector<Case>* out) {
	// 2^57 / 2^58 / INT64_MAX-class magnitudes (T-1822 §12's own Option-G coverage
	// bullet: "at least the 2^57/2^58/INT64_MAX-class magnitudes already used for
	// the RoPE-pair primitive Option G's sibling extends"), at both signs, against
	// both a maximal cos (sin=0, pure real-axis product) and a maximal sin (cos=0,
	// pure imaginary-axis product) so both the x-term and the cross-term reach
	// their own extremes independently.
	const int64_t magnitudes[] = {
	    int64_t{1} << 57,
	    int64_t{1} << 58,
	    INT64_MAX,
	    INT64_MIN,
	    INT64_MAX - 1,
	    INT64_MIN + 1,
	};
	for (int64_t m : magnitudes) {
		out->push_back({m, 0, kRopeOne, 0, "extreme_x_cos1_sin0"});
		out->push_back({0, m, kRopeOne, 0, "extreme_y_cos1_sin0"});
		out->push_back({m, 0, 0, kRopeOne, "extreme_x_cos0_sin1"});
		out->push_back({0, m, 0, kRopeOne, "extreme_y_cos0_sin1"});
		out->push_back({m, m, kRopeOne, kRopeOne, "extreme_both_cos1_sin1"});
		out->push_back({m, -m, kRopeOne, -kRopeOne, "extreme_both_mixed_sign"});
	}
	// A mid-magnitude, exact-Q2.30 pair (cos=sin=ROPE_ONE/sqrt(2)-ish, using an
	// exact half-scale value instead so the arithmetic is exact and the case is
	// reproducible without float): both terms contribute comparably, unlike the
	// axis-aligned cases above where one product is always zero.
	out->push_back({int64_t{1} << 57, int64_t{1} << 57, kRopeOne / 2, kRopeOne / 2,
	                 "extreme_both_terms_comparable"});
}

void AppendTruncationCornerCase(std::vector<Case>* out) {
	// T-1822 §12's own second domain-extremity requirement: "an int64-legal kacc
	// input whose 128-bit rotated composite truncates to an in-domain low word" --
	// mirroring LandingRescale's own truncation-corner defect class (this file's
	// sibling, forward_sites.cpp: "a 69-bit true quotient narrows to an in-band,
	// wrong-sign raw=100 with the saturation counter silent"). Constructed the
	// same way: an x/y/cos/sin combination whose PRE-rounding 128-bit magnitude
	// is large enough that a low-64-bit-only narrowing (the defect class, not
	// this primitive's actual behaviour) would silently wrap into a small,
	// plausible-looking in-domain value, while the true rotated result does not
	// fit int64_t. `RopeApplyPairWide`'s own `out_in_domain` is exercised here
	// directly: this case's own expected `in_domain` is `false` (checked in the
	// Python side against the arbitrary-precision computation), and the C++
	// primitive's OWN full-128-bit `quotient.hi != 0` check (mirroring
	// LandingRescale's own fix for this exact defect class) is what this case
	// exists to confirm did not regress.
	out->push_back({INT64_MAX, INT64_MAX, kRopeOne, kRopeOne, "truncation_corner"});
	out->push_back({INT64_MIN, INT64_MAX, kRopeOne, -kRopeOne, "truncation_corner_mixed"});
}

void AppendOverflowForcingCases(std::vector<Case>* out) {
	// T-1891 gate G2's own "overflow is refuse-not-wrap, and the refusal is
	// exercised in a test" requirement. A rotation can raise a pair's magnitude
	// by up to sqrt(2); at cos=sin=ROPE_ONE/2 (both terms contributing at 1/2
	// scale, matching the "both terms comparable" domain-extremity case above but
	// at FULL magnitude rather than 2^57) the combined magnitude is engineered to
	// cross int64_t's ceiling once rounded -- x=y=INT64_MAX/INT64_MIN at
	// cos=sin=ROPE_ONE is the sharpest witness: xr = x*cos - y*sin = 0 exactly
	// (cancels), but yr = x*sin + y*cos = 2*INT64_MAX*ROPE_ONE / 2^30 =
	// 2*INT64_MAX, which does not fit int64_t (INT64_MAX's own magnitude doubled
	// exceeds INT64_MAX by construction).
	out->push_back({INT64_MAX, INT64_MAX, kRopeOne, kRopeOne, "overflow_yr_doubled_max"});
	out->push_back({INT64_MIN, INT64_MIN, kRopeOne, kRopeOne, "overflow_yr_doubled_min"});
	out->push_back({INT64_MAX, INT64_MIN, kRopeOne, kRopeOne, "overflow_xr_doubled_max"});
	out->push_back({INT64_MIN, INT64_MAX, kRopeOne, kRopeOne, "overflow_xr_doubled_min"});
}

void AppendRandomSweep(std::vector<Case>* out, uint64_t seed, int count) {
	std::mt19937_64 rng(seed);
	std::uniform_int_distribution<int64_t> wide_dist(INT64_MIN, INT64_MAX);
	std::uniform_int_distribution<int32_t> table_dist(-kRopeOne, kRopeOne);
	for (int i = 0; i < count; ++i) {
		Case c;
		c.x = wide_dist(rng);
		c.y = wide_dist(rng);
		c.cos_q30 = table_dist(rng);
		c.sin_q30 = table_dist(rng);
		c.label = "random_sweep";
		out->push_back(c);
	}
}

}  // namespace

int main(int argc, char** argv) {
	const char* out_path = argc > 1 ? argv[1] : "out/t1891_gate2_cases.csv";
	const int sweep_count = argc > 2 ? std::atoi(argv[2]) : 200000;

	std::vector<Case> cases;
	AppendDomainExtremityCases(&cases);
	AppendTruncationCornerCase(&cases);
	AppendOverflowForcingCases(&cases);
	AppendRandomSweep(&cases, /*seed=*/0x54313839'31474332ULL, sweep_count);

	std::FILE* f = std::fopen(out_path, "w");
	if (f == nullptr) {
		std::fprintf(stderr, "could not open %s for writing\n", out_path);
		return 1;
	}
	std::fprintf(f, "label,x,y,cos_q30,sin_q30,out_x,out_y,in_domain\n");
	for (const Case& c : cases) {
		bool in_domain = false;
		const RopePairWide r = RopeApplyPairWide(c.x, c.y, c.cos_q30, c.sin_q30, &in_domain);
		std::fprintf(f, "%s,%lld,%lld,%d,%d,%lld,%lld,%d\n", c.label,
		             static_cast<long long>(c.x), static_cast<long long>(c.y), c.cos_q30,
		             c.sin_q30, static_cast<long long>(r.x), static_cast<long long>(r.y),
		             in_domain ? 1 : 0);
	}
	std::fclose(f);
	std::fprintf(stderr, "wrote %zu cases to %s (%zu domain-extremity, %d random)\n",
	             cases.size(), out_path, cases.size() - static_cast<size_t>(sweep_count),
	             sweep_count);
	return 0;
}

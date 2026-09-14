// T-2700: line-oriented compiled arithmetic probe for the independent fused-K oracle.
// Input: `gain code cos_q30 sin_q30 ratio_q31` per line.  k=[code,1],
// q=[127,-127], and source=target=127 are fixed.
// Output: wide0 wide1 rotated0 rotated1 landed0 landed1 q31_score.

#include <array>
#include <cstdint>
#include <cstdio>

#include "superslm/forward_sites.h"

int main() {
	int gain = 0;
	int code = 0, cos_q30 = 0, sin_q30 = 0;
	long long ratio_q31 = 0;
	while (std::scanf("%d %d %d %d %lld", &gain, &code, &cos_q30, &sin_q30,
	                  &ratio_q31) == 5) {
		const int64_t wide0 = superslm::FloorDivI64(
		    static_cast<int64_t>(code) * gain * (INT64_C(1) << 30), INT64_C(1) << 30);
		const int64_t wide1 = superslm::FloorDivI64(
		    static_cast<int64_t>(1) * gain * (INT64_C(1) << 30), INT64_C(1) << 30);
		bool in_domain = false;
		const superslm::RopePairWide rotated = superslm::RopeApplyPairWide(
		    wide0, wide1, cos_q30, sin_q30, &in_domain);
		if (!in_domain) return 2;
		// canonical_scale(127) and its C19 reciprocal, generated independently
		// by the Python oracle's stated source/target scale relation.
		constexpr int64_t kM = INT64_C(2130706432);
		constexpr int64_t kE = -24;
		constexpr int64_t kR = INT64_C(2164392968);
		const int64_t raw0 = superslm::LandingRescale(rotated.x, kM, kR, kE, kE);
		const int64_t raw1 = superslm::LandingRescale(rotated.y, kM, kR, kE, kE);
		const std::array<int8_t, 2> q = {127, -127};
		const std::array<int8_t, 2> k = {
		    static_cast<int8_t>(superslm::ClampRopeCode(raw0)),
		    static_cast<int8_t>(superslm::ClampRopeCode(raw1))};
		const std::array<int64_t, 2> ratio = {static_cast<int64_t>(ratio_q31),
		                                      static_cast<int64_t>(ratio_q31)};
		const int64_t score = superslm::QkQ31Score(q.data(), k.data(), ratio.data(), q.size());
		std::printf("%lld %lld %lld %lld %lld %lld %lld\n",
		            static_cast<long long>(wide0), static_cast<long long>(wide1),
		            static_cast<long long>(rotated.x), static_cast<long long>(rotated.y),
		            static_cast<long long>(raw0), static_cast<long long>(raw1),
		            static_cast<long long>(score));
	}
	return std::feof(stdin) ? 0 : 2;
}

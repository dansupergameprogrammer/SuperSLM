// Paged-KV plan (rev 16.1) §7, step C2: cell 6.2, GemmProbQ15AccumulateInto (§3.2 item 3).
//
// The context step of per-page attention zeroes `out_ctx` once and then adds one page run at a time
// through GemmProbQ15AccumulateInto. Its exactness claim is that the same int64 terms are summed in
// the same ascending order, so the chained result equals one GemmProbQ15Accumulate over the whole
// row. The cell pins it at the extremes the claim's bound is about: every probability at 2^15 (Q15
// 1.0), every value at +127 or -127, width = cap, at each geometry's head_dim. The reference is the
// shipped one-call GemmProbQ15Accumulate and, beside it, the exact sum written here.

#include "pkv_engine_helpers.h"

#include <algorithm>
#include <vector>

namespace {

using namespace pkv_engine;

constexpr int64_t kQ15One = int64_t{1} << 15;

// Value patterns at the extremes: all +127, all -127, alternating by position, alternating by
// (position + d), and a fixed pseudo-random sign.
std::vector<int8_t> Values(int pattern, size_t width, size_t D) {
	std::vector<int8_t> v(width * D);
	uint32_t x = 0x9E3779B9u;
	for (size_t k = 0; k < width; ++k)
		for (size_t d = 0; d < D; ++d) {
			bool plus = true;
			switch (pattern) {
			case 0: plus = true; break;
			case 1: plus = false; break;
			case 2: plus = (k & 1) == 0; break;
			case 3: plus = ((k + d) & 1) == 0; break;
			default:
				x = x * 1664525u + 1013904223u;
				plus = (x >> 31) != 0;
				break;
			}
			v[k * D + d] = static_cast<int8_t>(plus ? 127 : -127);
		}
	return v;
}

// The exact sum, written here: out[d] = sum_k probs[k] * values[k*D + d], int64 throughout.
std::vector<int64_t> ExactSum(const std::vector<int64_t>& probs, const std::vector<int8_t>& values, size_t width,
                              size_t D) {
	std::vector<int64_t> out(D, 0);
	for (size_t k = 0; k < width; ++k)
		for (size_t d = 0; d < D; ++d) out[d] += probs[k] * static_cast<int64_t>(values[k * D + d]);
	return out;
}

// §3.2 item 3's loop: out zeroed once (here: seeded), then one Into call per page run of B rows.
std::vector<int64_t> Chained(const std::vector<int64_t>& probs, const std::vector<int8_t>& values, size_t width,
                             size_t D, size_t B, const std::vector<int64_t>& seed) {
	std::vector<int64_t> out = seed;
	for (size_t p0 = 0; p0 < width; p0 += B) {
		const size_t rows = std::min(B, width - p0);
		superslm::GemmProbQ15AccumulateInto(probs.data() + p0, values.data() + p0 * D, rows, D, out.data());
	}
	return out;
}

// 6.2 [C2]: chained over page runs equals one GemmProbQ15Accumulate, at probs 2^15, values +-127,
// width = cap.
void Cell62() {
	// The cell geometries: cap 4096 (pkv_def, pkv_qk, the 0.5B and 1.5B cohorts) at the head_dims of
	// pkv_def (48), the 0.5B (64) and pkv_qk / the 1.5B (128); and cap 32768 (pkv_32k) at 48.
	const struct {
		size_t cap, D;
	} geometries[] = {{4096, 48}, {4096, 64}, {4096, 128}, {32768, 48}};
	for (const auto& g : geometries)
		for (int pattern = 0; pattern < 5; ++pattern) {
			const size_t cap = g.cap, D = g.D;
			// width = cap is the cell's; cap - 1 adds a short last run at every B.
			for (size_t width : {cap, cap - 1}) {
				const std::vector<int64_t> probs(width, kQ15One);
				const std::vector<int8_t> values = Values(pattern, width, D);
				std::vector<int64_t> one(D, int64_t{0x7EADBEEF});  // the one-call form zeroes it itself
				superslm::GemmProbQ15Accumulate(probs.data(), values.data(), width, D, one.data());
				const std::vector<int64_t> exact = ExactSum(probs, values, width, D);
				PKV_CHECK_MSG(one == exact, "cap %zu D %zu pattern %d width %zu: the shipped one-call sum is not exact",
				              cap, D, pattern, width);
				// B = 1 (every row its own run), the two engine views (4, 16), a run that does not divide
				// the width (1,000), and a single run (the one-page view's one call).
				for (size_t B : {size_t{1}, size_t{4}, size_t{16}, size_t{1000}, width}) {
					const std::vector<int64_t> zero(D, 0);
					const std::vector<int64_t> got = Chained(probs, values, width, D, B, zero);
					// kills: Into that zeroes out_ctx (only the last run survives), Into that assigns
					// instead of adding, and a SIMD tail that drops rows when a run is not a multiple of
					// the vector width (runs of 1, 4 and 15 rows).
					PKV_CHECK_MSG(got == one, "cap %zu D %zu pattern %d width %zu B %zu: chained Into != one call", cap,
					              D, pattern, width, B);
					// Into adds onto what out_ctx already holds: seeded with a large negative offset, the
					// chain ends at seed + the sum, term for term.
					std::vector<int64_t> seed(D);
					for (size_t d = 0; d < D; ++d) seed[d] = -(int64_t{1} << 40) + static_cast<int64_t>(d) * 977;
					const std::vector<int64_t> seeded = Chained(probs, values, width, D, B, seed);
					bool ok = true;
					for (size_t d = 0; d < D; ++d) ok = ok && seeded[d] == seed[d] + one[d];
					// kills: Into that zeroes out_ctx on entry, or zeroes it only when it is the first run.
					PKV_CHECK_MSG(ok, "cap %zu D %zu pattern %d width %zu B %zu: Into does not add onto out_ctx", cap,
					              D, pattern, width, B);
				}
			}
		}
}

PKV_CELL("6.2", "C2", Cell62);

}  // namespace

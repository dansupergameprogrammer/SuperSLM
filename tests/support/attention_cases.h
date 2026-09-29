// Attention and per-row sites plan, slice S2 (prob·V on int16 multiply-add): the fixed input set that
// the digest section `c32_attention`, the golden-pin generator (tools/gen_attn_rowsite_golden.cpp) and
// the suite's golden and grid cells all run through GemmProbQ15Accumulate. Header-only, so it needs no
// build entry: the digest and the generator include it by relative path, the suite through `tests/`.
// Later slices (S4, S5, S6) append their own entries to the same section and their own hashes.
//
// Everything here calls only GemmProbQ15Accumulate, whose signature is unchanged since v1.9.0, so the
// generator can be built against the v1.9.0 tag's library and the pin takes no input from the code it
// grades (plan §3.3 evidence 3).
//
// The set is plan §8 4.S2's grid: head_dim {4, 8, 12, 16, 60, 64, 100, 128, 132, 256, 512} x width
// {1, 2, 3, 7, 8, 9, 63, 64, 65, 1,024, 4,097} with realistic and peaked rows (both sides of
// head_dim % 16, odd widths for the unpaired last key), plus the int16 condition's corners (p = 32,767
// with Sum p = 2^15 exactly and 2^15 + 1, one-hot rows at width 1 and inside a wider row, Sum p = 2^15
// over many keys with every value -128 or 127, the lane bound's corners), width 0, and 2.S2's four
// hostile rows, each failing exactly one conjunct. Each call contributes width, head_dim and its whole
// output row (poisoned before the call, so the zeroing contract is digested too).
#ifndef SUPERSLM_TESTS_SUPPORT_ATTENTION_CASES_H
#define SUPERSLM_TESTS_SUPPORT_ATTENTION_CASES_H

#include <cstddef>
#include <cstdint>
#include <vector>

#include "superslm/matmul.h"

namespace superslm_attention_cases {

// splitmix64, integer-only (the same generator as rowsite_cases.h, restated so this header stands alone).
struct Rng {
	uint64_t s;
	explicit Rng(uint64_t seed) : s(seed) {}
	uint64_t Next() {
		s += 0x9e3779b97f4a7c15ULL;
		uint64_t z = s;
		z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
		z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
		return z ^ (z >> 31);
	}
	int64_t InRange(int64_t lo, int64_t hi) {
		const uint64_t span = static_cast<uint64_t>(hi - lo) + 1ULL;
		return lo + static_cast<int64_t>(Next() % span);
	}
};

inline constexpr size_t kPvHeadDims[] = {4, 8, 12, 16, 60, 64, 100, 128, 132, 256, 512};
inline constexpr size_t kPvWidths[] = {1, 2, 3, 7, 8, 9, 63, 64, 65, 1024, 4097};
inline constexpr int64_t kPvPoison = INT64_C(0x5A5A5A5A5A5A5A5A);

// One prob·V call: its row, its values and a label.
struct PvCase {
	const char* label;
	size_t width;
	size_t head_dim;
	std::vector<int64_t> probs;   // exactly `width` elements
	std::vector<int8_t> values;   // exactly width * head_dim elements
};

// A realistic row: random weights e_k, p_k = floor(e_k * 2^15 / Sum e), as the softmax forms them
// (Sum p <= 2^15 by construction). `spread` sets how peaked it is: weights are 2^(random in [0, spread]).
inline std::vector<int64_t> RealisticRow(size_t width, int spread, Rng& rng) {
	std::vector<int64_t> e(width), p(width);
	int64_t total = 0;
	for (size_t k = 0; k < width; ++k) {
		const int sh = static_cast<int>(rng.InRange(0, spread));
		e[k] = (INT64_C(1) << sh) + rng.InRange(0, (INT64_C(1) << sh) - 1);
		total += e[k];
	}
	for (size_t k = 0; k < width; ++k) p[k] = total == 0 ? 0 : (e[k] << 15) / total;
	return p;
}

inline std::vector<int8_t> RandomValues(size_t n, Rng& rng) {
	std::vector<int8_t> v(n);
	for (auto& x : v) x = static_cast<int8_t>(rng.InRange(-128, 127));
	return v;
}

// Calls fn(const PvCase&) for every case of the set, in a fixed order.
template <class Fn>
void ForEachProbVCase(Fn&& fn) {
	Rng rng(0x5332505653455431ULL);  // "S2PVSET1"
	// The grid: a realistic row and a peaked row at every (head_dim, width).
	for (size_t hd : kPvHeadDims)
		for (size_t w : kPvWidths)
			for (int spread : {4, 24}) {
				PvCase c{spread == 4 ? "4.S2 grid, realistic row" : "4.S2 grid, peaked row", w, hd,
				         RealisticRow(w, spread, rng), RandomValues(w * hd, rng)};
				fn(c);
			}

	const auto values_of = [&rng](size_t n, int mode) {
		std::vector<int8_t> v(n);
		for (auto& x : v) x = mode == 0 ? static_cast<int8_t>(rng.InRange(-128, 127)) : static_cast<int8_t>(mode);
		return v;
	};
	// The int16 condition's inside and outside corners (4.S2, 7.S2).
	for (size_t hd : {size_t{16}, size_t{64}, size_t{60}}) {
		fn(PvCase{"4.S2 corner p = 32,767 and 1 (Sum p = 2^15)", 2, hd, {32767, 1}, values_of(2 * hd, 0)});
		fn(PvCase{"4.S2 corner p = 32,767 and 2 (Sum p = 2^15 + 1)", 2, hd, {32767, 2}, values_of(2 * hd, 0)});
		fn(PvCase{"4.S2 corner width-1 one-hot (p = 2^15)", 1, hd, {32768}, values_of(hd, 0)});
		fn(PvCase{"4.S2 corner one-hot inside a width-3 row", 3, hd, {0, 32768, 0}, values_of(3 * hd, 0)});
		fn(PvCase{"4.S2 corner width 0", 0, hd, {}, {}});
	}
	// Sum p = 2^15 exactly over many keys, every value at an extreme: each lane reaches 128 * 2^15 = 2^22
	// (all -128) or 127 * 2^15 (all 127), the §5.2 lane bound's corner, at an odd width.
	for (int mode : {-128, 127}) {
		std::vector<int64_t> p(1025, 0);
		for (size_t k = 0; k < 1024; ++k) p[k] = 32;  // 1,024 x 32 = 2^15; the 1,025th key is 0
		fn(PvCase{mode < 0 ? "4.S2 corner Sum p = 2^15, every v = -128" : "4.S2 corner Sum p = 2^15, every v = 127",
		          1025, 64, p, values_of(1025 * 64, mode)});
	}
	{
		std::vector<int64_t> p(3, 0);
		p[2] = 32767;  // the unpaired last key carries the maximum
		fn(PvCase{"4.S2 corner odd last key p = 32,767", 3, 64, p, values_of(3 * 64, -128)});
	}

	// 2.S2: each row fails exactly one conjunct of the guard (head_dim 64).
	{
		std::vector<int64_t> p = RealisticRow(64, 8, rng);
		for (auto& x : p) x /= 2;  // keeps Sum p far from 2^15 whatever key 17 holds
		p[17] = -32769;            // only p >= 0 fails; the int16 pack would truncate it
		fn(PvCase{"2.S2 p = -32,769 at one key", 64, 64, p, values_of(64 * 64, 0)});
	}
	{
		std::vector<int64_t> p(1024);
		std::vector<int8_t> v(1024 * 64);
		for (size_t k = 0; k < 1024; ++k) {
			p[k] = (k % 2 == 0) ? 32767 : -32767;  // Sum p = 0: only p >= 0 fails
			for (size_t d = 0; d < 64; ++d) v[k * 64 + d] = (k % 2 == 0) ? 127 : -127;  // every product positive
		}
		fn(PvCase{"2.S2 alternating p = +-32,767, v sign-matched, W = 1,024", 1024, 64, p, v});
	}
	fn(PvCase{"2.S2 width-1 one-hot p = 32,768", 1, 64, {32768}, values_of(64, 0)});
	fn(PvCase{"2.S2 W = 1,024, p = 32,767, v = 127", 1024, 64, std::vector<int64_t>(1024, 32767),
	          values_of(1024 * 64, 127)});
}

// The whole S2 set through GemmProbQ15Accumulate: per call, width, head_dim and every output value.
template <class Emit>
void RunProbVCases(Emit& emit) {
	ForEachProbVCase([&emit](const PvCase& c) {
		std::vector<int64_t> out(c.head_dim, kPvPoison);
		superslm::GemmProbQ15Accumulate(c.probs.data(), c.values.data(), c.width, c.head_dim, out.data());
		emit(static_cast<int64_t>(c.width));
		emit(static_cast<int64_t>(c.head_dim));
		for (int64_t x : out) emit(x);
	});
}

}  // namespace superslm_attention_cases

#endif  // SUPERSLM_TESTS_SUPPORT_ATTENTION_CASES_H

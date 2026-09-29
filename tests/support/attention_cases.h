// Attention and per-row sites plan, slice S2 (prob·V on int16 multiply-add): the fixed input set that
// the digest section `c32_attention`, the golden-pin generator (tools/gen_attn_rowsite_golden.cpp) and
// the suite's golden and grid cells all run through GemmProbQ15Accumulate. Header-only, so it needs no
// build entry: the digest and the generator include it by relative path, the suite through `tests/`.
// Later slices (S4, S5, S6) append their own entries to the same section and their own hashes.
//
// Everything here calls only GemmProbQ15Accumulate, SoftmaxRowQ15 and IExpScaleConstants, whose
// signatures are unchanged since v1.9.0, so the generator can be built against the v1.9.0 tag's library
// and the pin takes no input from the code it grades (plan §3.3 evidence 3).
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
#include <array>
#include <cstdint>
#include <vector>

#include "superslm/intmath.h"
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

// ==== Slice S4: the guarded softmax (plan §4.4, §5.4, §8 2.S4 and 4.S4) =============================
//
// The S4 set runs SoftmaxRowQ15 over plan §8 4.S4's grid and corners and 2.S4's hostile rows. Its rows
// are chosen by two test-side copies written from the plan, never by the build under test:
//
//   * TestSoftmaxGuard, §5.4's row guard in dependency order, each conjunct reported separately so a 2.S4
//     row can be shown to fail exactly one;
//   * SoftmaxEstimateReplica, the fast path's estimate-and-correct arithmetic element by element, which
//     reports which corrections a row needs, so the correction rows are chosen, not hoped for.
//
// The estimates are integer, not IEEE double (docs/attention-rowsites-s4-progress.md, deviation 1: the
// library is floating-point-free and the fp-free scan gates it). §5.4's own floating-point bullet is what
// carries over: the estimates only need to land within one of the floor, because exactness comes from the
// exact integer corrections. Per element, after the max shift (a = min(max - s, 30 q_ln2) >= 0):
//
//   z   estimate: (a * inv_z) >> kz, inv_z = floor(2^kz / q_ln2), kz = 30 + bit_width(q_ln2). inv_z is at
//       most 2^31, a * inv_z < 2^61, and the estimate never exceeds floor(a / q_ln2) (a floored
//       reciprocal) and is at most one below it. So only the UPWARD z correction can fire, exactly as
//       §5.4 step 3 argues for the double estimate; the downward one is kept as defensive code.
//   p   estimate: (e * R) >> 47, R = round(2^62 / denom) = (2^62 + floor(denom / 2)) / denom. e <= 2^47 and
//       e <= denom give e * R <= 2^62 + 2^46, and |error| <= e / 2^48 <= 1/2, so the estimate lands on the
//       floor or one either side of it: BOTH p corrections are live, as §5.4 step 4 has them.
//
// Integer-only, so this header builds anywhere the suite does (no __int128: the MSVC legs build it).

// §5.4's guard, one flag per conjunct. `scores` is read only when width is in range.
struct SoftmaxGuard {
	bool width_ok = true, q_ln2_ge1 = true, q_c_ge0 = true, m_ge1 = true, m_le = true, ratio = true, scores_ok = true;
	bool Fast() const { return width_ok && q_ln2_ge1 && q_c_ge0 && m_ge1 && m_le && ratio && scores_ok; }
	int Failing() const { return !width_ok + !q_ln2_ge1 + !q_c_ge0 + !m_ge1 + !m_le + !ratio + !scores_ok; }
};

inline constexpr size_t kSmMaxWidth = size_t{1} << 14;
inline constexpr int64_t kSmMaxM = int64_t{1} << 47;
inline constexpr int64_t kSmScoreLimit = int64_t{1} << 61;

inline SoftmaxGuard TestSoftmaxGuard(const int64_t* scores, size_t width, int64_t q_ln2, int64_t q_b, int64_t q_c) {
	SoftmaxGuard g;
	g.width_ok = width >= 1 && width <= kSmMaxWidth;
	g.q_ln2_ge1 = q_ln2 >= 1;
	g.q_c_ge0 = q_c >= 0;
	// M = q_b^2 + q_c, judged exactly without 128-bit arithmetic: |q_b| > 2^32 makes q_b^2 > 2^64 > 2^47 + |q_c|.
	const uint64_t ab = q_b < 0 ? 0 - static_cast<uint64_t>(q_b) : static_cast<uint64_t>(q_b);
	if (ab > (uint64_t{1} << 32)) {
		g.m_le = false;
	} else {
		const uint64_t sq = ab == (uint64_t{1} << 32) ? UINT64_MAX : ab * ab;  // 2^64 saturates; still > 2^47 + 2^63
		if (q_c >= 0) {
			g.m_ge1 = sq > 0 || q_c >= 1;
			g.m_le = sq <= static_cast<uint64_t>(kSmMaxM) && sq + static_cast<uint64_t>(q_c) <= static_cast<uint64_t>(kSmMaxM);
		} else {
			const uint64_t mag = 0 - static_cast<uint64_t>(q_c);  // |q_c| <= 2^63
			g.m_ge1 = sq > mag;                                     // M >= 1
			g.m_le = sq <= mag || sq - mag <= static_cast<uint64_t>(kSmMaxM);
		}
	}
	// q_ln2 <= 2 q_b + 1, exact: q_b is an int64, so 2 q_b + 1 is formed from its halves.
	if (q_b >= 0) {
		g.ratio = q_ln2 <= q_b || q_b == INT64_MAX || q_ln2 - q_b <= q_b + 1;
	} else {
		g.ratio = q_ln2 < 0 && q_ln2 - q_b <= q_b + 1;  // q_ln2 - q_b cannot overflow for q_ln2 < 0 <= -q_b
	}
	if (g.width_ok)
		for (size_t k = 0; k < width; ++k)
			if (scores[k] > kSmScoreLimit || scores[k] < -kSmScoreLimit) g.scores_ok = false;
	return g;
}

// Which corrections the fast path's estimates need on one row, element counts.
struct SoftmaxCorrections {
	long long z_up = 0, z_down = 0, p_up = 0, p_down = 0;
};

inline int BitWidth64(uint64_t x) {
	int b = 0;
	while (x != 0) {
		++b;
		x >>= 1;
	}
	return b;
}

// The replica of the fast path (above), for a row the guard copy admits: fills `out` (width values) with
// the probabilities and `c` with the number of elements on which each correction fires. `skip` names one
// correction the replica leaves out (0 none, 1 z up, 2 p up, 3 p down), which is how a test states what
// the §9 "correction skipped" mutant would compute. Returns false, touching nothing, outside the guard.
inline bool SoftmaxEstimateReplica(const int64_t* scores, size_t width, int64_t q_ln2, int64_t q_b, int64_t q_c,
                                   int64_t* out, SoftmaxCorrections* c, int skip = 0) {
	if (!TestSoftmaxGuard(scores, width, q_ln2, q_b, q_c).Fast()) return false;
	int64_t mx = scores[0];
	for (size_t k = 1; k < width; ++k)
		if (scores[k] > mx) mx = scores[k];
	const int kz = 30 + BitWidth64(static_cast<uint64_t>(q_ln2));
	const uint64_t inv_z = (uint64_t{1} << kz) / static_cast<uint64_t>(q_ln2);
	const uint64_t clip = 30 * static_cast<uint64_t>(q_ln2);
	std::vector<int64_t> e(width);
	int64_t total = 0;
	for (size_t k = 0; k < width; ++k) {
		uint64_t a = static_cast<uint64_t>(mx - scores[k]);  // <= 2^62
		if (a > clip) a = clip;
		int64_t z = static_cast<int64_t>((a * inv_z) >> kz);
		int64_t r = static_cast<int64_t>(a) - z * q_ln2;
		if (r >= q_ln2) {
			++c->z_up;
			if (skip != 1) {
				++z;
				r -= q_ln2;
			}
		}
		if (r < 0) {
			++c->z_down;
			--z;
			r += q_ln2;
		}
		const int64_t base = q_b - r;  // q_p + q_b with q_p = -r
		e[k] = (base * base + q_c) >> z;
		total += e[k];
	}
	const uint64_t denom = static_cast<uint64_t>(total);
	const uint64_t R = ((uint64_t{1} << 62) + (denom >> 1)) / denom;
	for (size_t k = 0; k < width; ++k) {
		const uint64_t num = static_cast<uint64_t>(e[k]) << 15;
		uint64_t p = (static_cast<uint64_t>(e[k]) * R) >> 47;
		uint64_t prod = p * denom;
		if (prod > num) {
			++c->p_down;
			if (skip != 3) {
				--p;
				prod -= denom;
			}
		}
		if (prod + denom <= num) {
			++c->p_up;
			if (skip != 2) ++p;
		}
		out[k] = static_cast<int64_t>(p);
	}
	return true;
}

// Integer square root, floor, for 0 <= x < 2^62.
inline int64_t ISqrtFloor(int64_t x) {
	uint64_t lo = 0, hi = uint64_t{1} << 31;
	while (lo < hi) {
		const uint64_t mid = (lo + hi + 1) / 2;
		if (mid * mid <= static_cast<uint64_t>(x))
			lo = mid;
		else
			hi = mid - 1;
	}
	return static_cast<int64_t>(lo);
}

// One SoftmaxRowQ15 call of the set.
struct SmCase {
	const char* label;
	int64_t q_ln2, q_b, q_c;
	std::vector<int64_t> scores;  // the row, width = size()
	bool aliased;                 // scores == out_probs
};

inline constexpr int64_t kSmPoison = INT64_C(0x3C3C3C3C3C3C3C3C);

// The realistic constant triples: IExpScaleConstants over the forward's scale range, C30's pinned
// coefficients (the production call site's), mantissas in [2^30, 2^31) and exponents -52..-31 (inside the
// guard; the 0.5B-width forward's rows sit at about -41 to -39), plus -55 and -53 (M above 2^47: outside)
// and -30 (q_ln2 = 0: outside).
inline std::vector<std::array<int64_t, 3>> SmRealisticConstants(Rng& rng, int per_exponent) {
	std::vector<std::array<int64_t, 3>> out;
	for (int e = -55; e <= -30; ++e) {
		if (e == -54) continue;
		for (int t = 0; t < per_exponent; ++t) {
			const int64_t m = (INT64_C(1) << 30) + rng.InRange(0, (INT64_C(1) << 30) - 1);
			int64_t a = 0, b = 0, c = 0;
			if (superslm::IExpScaleConstants(m, e, superslm::kIExpLn2Q, 30, superslm::kIExpBQ, 30, superslm::kIExpCaQ, 30,
			                                 &a, &b, &c) == superslm::IExpScaleDomain::kOk)
				out.push_back({a, b, c});
		}
	}
	return out;
}

// A score row of `width` with `kind`: 0 realistic (a spread of 2^0..2^20, int8 dot products at head_dim 64
// reach 2^20), 1 one dominant score, 2 clip-heavy (most scores past 30 q_ln2 below the max: z = 30), 3 exact
// multiples of q_ln2 below the max (z at every value 0..30).
inline std::vector<int64_t> SmScoreRow(size_t width, int kind, int64_t q_ln2, Rng& rng) {
	std::vector<int64_t> s(width);
	const int bits = static_cast<int>(rng.InRange(0, 20));
	for (auto& x : s) {
		switch (kind) {
			case 0: x = rng.InRange(-(INT64_C(1) << bits), INT64_C(1) << bits); break;
			case 1: x = rng.InRange(-40 * q_ln2, -q_ln2); break;
			case 2: x = rng.InRange(-64 * q_ln2, 0); break;
			default: x = -q_ln2 * rng.InRange(0, 31); break;
		}
	}
	if (kind == 1 && width > 0) s[rng.Next() % width] = rng.InRange(0, 1000);
	return s;
}

// The steered generator for rows the p DOWNWARD correction needs (plan §5.4 step 4, §8 4.S4, after
// probes/rev3/probe_pdown.cpp): the denominator first, then a row summing to it. The probe steered the
// double estimate; this one steers the integer estimate above, whose error comes from R's rounding. The
// estimate for e = M is floor(M R / 2^47). For a reciprocal R = k, take N = floor(M k / 2^47) and
// denom = ceil(M 2^15 / N) + j for small j: the true quotient M 2^15 / denom is then just below N, so the
// estimate overshoots whenever denom still rounds to R = k (which holds when M k / 2^47 sits less than about
// M / 2^48 above N, so about half of all k). Each (k, j) is kept only if the overshoot is confirmed. The row
// is floor(denom / M) scores at the maximum (each e = M) plus fine elements whose e values sum to the exact
// remainder: z = 0 elements with e = base^2 + q_c (base from an integer square root) while the remainder is
// >= 2^18, then one z = 29 element whose e is what is left. From each seed k0 the generator tries k0, k0 + 1,
// ... and keeps at most `per_k` rows; widths run from about 2^15 / k0.
inline void SmSteeredPDownRows(int64_t q_ln2, int64_t q_b, int64_t q_c, const std::vector<int64_t>& seeds, int per_k,
                               const char* label, std::vector<SmCase>& out) {
	const int64_t M = q_b * q_b + q_c;                                                  // <= 2^47
	const auto elem = [&](int64_t j, int64_t z) { return -(z * q_ln2 + (q_b - j)); };  // r = q_b - j, base = j
	for (int64_t k0 : seeds) {
		int kept = 0;
		for (int64_t k = k0; k < k0 + 32 && kept < per_k; ++k) {
			const int64_t N = static_cast<int64_t>((static_cast<uint64_t>(M) * static_cast<uint64_t>(k)) >> 47);
			if (N < 1) continue;
			for (int64_t j = 0; j < 8 && kept < per_k; ++j) {
				const int64_t D = (M * 32768 + N - 1) / N + j;  // M 2^15 <= 2^62
				const int64_t n = D / M;
				if (n < 1 || static_cast<size_t>(n) > kSmMaxWidth) continue;
				const uint64_t R = ((uint64_t{1} << 62) + (static_cast<uint64_t>(D) >> 1)) / static_cast<uint64_t>(D);
				const uint64_t p_est = (static_cast<uint64_t>(M) * R) >> 47;
				if (!(p_est * static_cast<uint64_t>(D) > (static_cast<uint64_t>(M) << 15))) continue;
				int64_t rem = D - n * M;
				std::vector<int64_t> s(static_cast<size_t>(n), 0);
				bool ok = true;
				while (rem >= (INT64_C(1) << 18)) {
					int64_t b = ISqrtFloor(rem - q_c > 0 ? rem - q_c : 0);
					if (b > q_b) b = q_b;
					if (b * b + q_c > rem || b * b + q_c <= 0) {
						ok = false;
						break;
					}
					s.push_back(elem(b, 0));
					rem -= b * b + q_c;
				}
				if (ok && rem > 0) {  // one z = 29 element with (b^2 + q_c) >> 29 == rem
					int64_t b = ISqrtFloor((rem << 29) > q_c ? (rem << 29) - q_c : 0);
					while (b > 0 && ((b * b + q_c) >> 29) >= rem) --b;
					while (((b * b + q_c) >> 29) < rem) ++b;
					if (((b * b + q_c) >> 29) != rem || b > q_b || q_b - b >= q_ln2) ok = false;
					else s.push_back(elem(b, 29));
				}
				if (!ok || s.size() > kSmMaxWidth) continue;
				out.push_back(SmCase{label, q_ln2, q_b, q_c, std::move(s), false});
				++kept;
			}
		}
	}
}

// Calls fn(const SmCase&) for every case of the S4 set, in a fixed order.
template <class Fn>
void ForEachSoftmaxCase(Fn&& fn) {
	Rng rng(0x5334534D53455431ULL);  // "S4SMSET1"
	const std::vector<std::array<int64_t, 3>> triples = SmRealisticConstants(rng, 2);
	// 4.S4: the grid. Plan widths {1, 2, 3, 4, 5, 2^14}, plus the kernels' block edges (4 and 8 lanes) and
	// the forward's widths, at every realistic triple (inside and outside the guard), four row kinds.
	for (size_t w : {size_t{1}, size_t{2}, size_t{3}, size_t{4}, size_t{5}, size_t{7}, size_t{8}, size_t{9},
	                 size_t{15}, size_t{16}, size_t{17}, size_t{63}, size_t{64}, size_t{65}, size_t{301}})
		for (const auto& t : triples)
			for (int kind = 0; kind < 4; ++kind) {
				const bool aliased = (rng.Next() & 7) == 0;
				fn(SmCase{aliased ? "4.S4 grid, aliased" : "4.S4 grid", t[0], t[1], t[2],
				          SmScoreRow(w, kind, t[0], rng), aliased});
			}
	for (int kind = 0; kind < 4; ++kind) {
		const auto& t = triples[static_cast<size_t>(rng.Next() % triples.size())];
		fn(SmCase{"4.S4 grid, width 1,024", t[0], t[1], t[2], SmScoreRow(1024, kind, t[0], rng), kind == 3});
		fn(SmCase{"4.S4 grid, width 2^14", t[0], t[1], t[2], SmScoreRow(kSmMaxWidth, kind, t[0], rng), kind == 1});
	}
	// 4.S4: total = 1 (M = 1 and every other element clipped to e = 0) and one element equal to total
	// (p = 2^15); z at 0 and 30 on every clip-heavy row above.
	fn(SmCase{"4.S4 total = 1, p = 2^15", 3, 1, 0, {0, -1000, -2000}, false});
	fn(SmCase{"4.S4 total = 1, p = 2^15, aliased", 3, 1, 0, {0, -1000, -2000}, true});
	// 4.S4 inside corners, each fast.
	fn(SmCase{"4.S4 inside: M = 2^47, width 2^14, equal scores (denom = 2^61)", 1, 0, kSmMaxM,
	          std::vector<int64_t>(kSmMaxWidth, 5), false});
	{
		const int64_t qb = 636211, qc = 212166321733;  // the -49 triple's q_b and q_c, q_ln2 at its ceiling
		fn(SmCase{"4.S4 inside: q_ln2 = 2 q_b + 1, realistic spread", 2 * qb + 1, qb, qc,
		          SmScoreRow(301, 0, 2 * qb + 1, rng), false});
		fn(SmCase{"4.S4 inside: q_ln2 = 2 q_b + 1, exact multiples", 2 * qb + 1, qb, qc,
		          SmScoreRow(301, 3, 2 * qb + 1, rng), false});
	}
	fn(SmCase{"4.S4 inside: scores of exactly +-2^61", 9, 4, 0, {kSmScoreLimit, -kSmScoreLimit, 0}, false});
	fn(SmCase{"4.S4 inside: M = 1", 3, 1, 0, {0, -1, -2, -100}, false});
	fn(SmCase{"4.S4 inside: q_c = 0", 9, 4, 0, {0, -3, -9, -1000}, false});
	fn(SmCase{"4.S4 inside: q_ln2 = 1", 1, 0, kSmMaxM, {0, -1, -5, -29, -30, -31}, false});
	// 4.S4 correction rows, chosen by the replica (at least 16 each; 24 are kept). z up: scores at exact
	// multiples of q_ln2 below the max, over many q_ln2. p up: ordinary rows. A row is kept only where the
	// replica says the correction fires AND leaving it out changes the row's output (the i-exp construction
	// halves its value per ln 2 step, so a skipped z correction often lands on the same probability; those
	// rows would not decide the §9 mutant).
	{
		int z_up = 0, p_up = 0;
		std::vector<int64_t> good, skipped;
		for (int it = 0; it < 20000 && (z_up < 24 || p_up < 24); ++it) {
			const auto& t = triples[static_cast<size_t>(rng.Next() % triples.size())];
			const size_t w = static_cast<size_t>(rng.InRange(2, 40));
			std::vector<int64_t> s = SmScoreRow(w, it % 2 == 0 ? 3 : 0, t[0], rng);
			good.assign(w, 0);
			SoftmaxCorrections c;
			if (!SoftmaxEstimateReplica(s.data(), w, t[0], t[1], t[2], good.data(), &c)) continue;
			const auto decides = [&](int skip) {
				SoftmaxCorrections unused;
				skipped.assign(w, 0);
				SoftmaxEstimateReplica(s.data(), w, t[0], t[1], t[2], skipped.data(), &unused, skip);
				return skipped != good;
			};
			if (c.z_up > 0 && z_up < 24 && decides(1)) {
				++z_up;
				fn(SmCase{"4.S4 correction row: z up", t[0], t[1], t[2], s, false});
			} else if (c.p_up > 0 && p_up < 24 && decides(2)) {
				++p_up;
				fn(SmCase{"4.S4 correction row: p up", t[0], t[1], t[2], s, false});
			}
		}
	}
	// p down: the steered generator. The plan's constants put M within 2^-24 of 2^47, where M R / 2^47 sits
	// just below the integer R and the integer estimate for e = M can never overshoot (progress file,
	// deviation 1), so the constants here put M near 3/4 and 7/10 of 2^47, in the plan's two shapes: q_ln2 =
	// 2 q_b + 1 with q_c = 0, and q_ln2 = q_b with q_c = 12,345. The widths run from 6 to 8,193 (the second
	// set reaches N = 4, width 2^15 / 4, because 4 / 0.7 rounds up).
	{
		std::vector<SmCase> rows;
		const std::vector<int64_t> ks = {3, 24, 71, 200, 553, 1648, 2600, 3500, 5052, 7000, 9000, 12000};
		SmSteeredPDownRows(2 * 10273742 + 1, 10273742, 0, ks, 2, "4.S4 correction row: p down (steered)", rows);
		SmSteeredPDownRows(9925730, 9925730, 12345, ks, 2, "4.S4 correction row: p down (steered, q_c = 12,345)", rows);
		for (const SmCase& c : rows) fn(c);
	}
	// 2.S4: each row fails exactly one conjunct of §5.4's guard.
	// The width and score rows use the first realistic triple inside the guard, so each fails one conjunct.
	std::array<int64_t, 3> rt{};
	for (const auto& t : triples) {
		const int64_t zero = 0;
		if (TestSoftmaxGuard(&zero, 1, t[0], t[1], t[2]).Fast()) {
			rt = t;
			break;
		}
	}
	fn(SmCase{"2.S4 q_ln2 = 0", 0, 0, 1, {0, -3, -7}, false});
	fn(SmCase{"2.S4 q_ln2 = -1", -1, 0, 1, {0, -3, -7}, false});
	fn(SmCase{"2.S4 q_c = -50 (q_b 10, q_ln2 21)", 21, 10, -50, {0, -10}, false});
	fn(SmCase{"2.S4 M = 0 (q_b 0, q_c 0, q_ln2 1), width 3", 1, 0, 0, {0, 0, 0}, false});
	fn(SmCase{"2.S4 M = 2^47 + 1", 1, 0, kSmMaxM + 1, {0, -1}, false});
	fn(SmCase{"2.S4 q_ln2 = 2 q_b + 2 (q_b 4, q_ln2 10)", 10, 4, 0, {0, -9}, false});
	fn(SmCase{"2.S4 width 2^14 + 1", rt[0], rt[1], rt[2], SmScoreRow(kSmMaxWidth + 1, 0, rt[0], rng), false});
	{
		std::vector<int64_t> s(64, 0);
		s[17] = kSmScoreLimit + 1;
		fn(SmCase{"2.S4 one score 2^61 + 1", rt[0], rt[1], rt[2], s, false});
		s[17] = -kSmScoreLimit - 1;
		fn(SmCase{"2.S4 one score -(2^61 + 1)", rt[0], rt[1], rt[2], s, false});
	}
	// The off-ratio witness (tests/sslm_c32_softmax_row_width_gate_fixtures.h, kSoftmaxRowOffRatioWitness),
	// restated here so the generator needs no other header: a realistic hostile row that fails more than one
	// conjunct (q_c >= 0, M >= 1 and q_ln2 <= 2 q_b + 1), so it kills no single-conjunct mutant.
	fn(SmCase{"2.S4 off-ratio witness", INT64_C(3000000001), 10, -100, {0, INT64_C(-3000000000), INT64_C(-2999999999)},
	          false});
}

// The whole S4 set through SoftmaxRowQ15: per call, width, the bool and every output value (the output
// row is poisoned before the call; an aliased call writes over its own scores).
template <class Emit>
void RunSoftmaxCases(Emit& emit) {
	ForEachSoftmaxCase([&emit](const SmCase& c) {
		const size_t w = c.scores.size();
		std::vector<int64_t> out(w, kSmPoison);
		bool ok;
		if (c.aliased) {
			out = c.scores;
			ok = superslm::SoftmaxRowQ15(out.data(), w, c.q_ln2, c.q_b, c.q_c, out.data());
		} else {
			ok = superslm::SoftmaxRowQ15(c.scores.data(), w, c.q_ln2, c.q_b, c.q_c, out.data());
		}
		emit(static_cast<int64_t>(w));
		emit(ok ? 1 : 0);
		for (int64_t x : out) emit(x);
	});
}

// ==== Slice S5: the Q31 score in three 16-bit pieces, per head (plan §4.5, §5.5, §8 4.S5 and 7.S5) =====
//
// The S5 set scores one query head against `width` keys through a caller-supplied row function: the
// digest and the suite pass the build's QkQ31ScoreRow; the golden-pin generator, built against the v1.9.0
// tag (which has no row entry), passes a loop over the v1.9.0 per-key QkQ31Score. The set is §8 4.S5's grid,
// head_dim {4, 8, 60, 64, 128, 132, 256, 512, 513, 516} x width {1, 7, 8, 9, 1,024} plus widths 15, 16
// and 17 (the AVX-512 body's 16-key block, full and partial), in three operand kinds; 7.S5b's margin
// corners at head_dim 512 (and the same rows at 516, past the guard); and 7.S5c's rounding ties.
//
// Every ratio here is in [1, 2^31], the loader's range (G6), inside S5's fast-path range [0, 2^32): §3.3
// keeps pin and digest inputs in contract, because outside [0, 2^32) the v1.9.0 SIMD tiers of QkQ31Score
// already differ from its scalar reference, so no single pin could hold there. The out-of-contract rows
// (2.S5) are suite-only cells compared with the same binary's per-key QkQ31Score.

inline constexpr size_t kQ31HeadDims[] = {4, 8, 60, 64, 128, 132, 256, 512, 513, 516};
inline constexpr size_t kQ31Widths[] = {1, 7, 8, 9, 15, 16, 17, 1024};
inline constexpr int64_t kQ31Poison = INT64_C(0x2B2B2B2B2B2B2B2B);
inline constexpr int64_t kQ31RatioMax = INT64_C(1) << 31;  // the loader's maximum

// One score-row call: one query head's q, `width` key rows of head_dim (the K store's layout for one KV
// head, positions 0 .. width - 1), and the KV head's per-channel ratios.
struct Q31Case {
	const char* label;
	size_t head_dim;
	size_t width;
	std::vector<int8_t> q;       // head_dim
	std::vector<int8_t> keys;    // width * head_dim
	std::vector<int64_t> ratio;  // head_dim
};

// kind 0: uniform q and k in [-128, 127], ratio in [1, 2^31]; kind 1: q and k at the int8 extremes
// {-128, 127}, ratio in {1, 2^31 - 1, 2^31}; kind 2: uniform q and k, ratio 2^31 on every channel (the
// real value, as the QK-norm fixture carries).
inline Q31Case MakeQ31GridCase(size_t head_dim, size_t width, int kind, Rng& rng) {
	static const char* const kLabels[] = {"4.S5 grid, uniform", "4.S5 grid, int8 extremes", "4.S5 grid, ratio 2^31"};
	Q31Case c{kLabels[kind], head_dim, width, std::vector<int8_t>(head_dim), std::vector<int8_t>(width * head_dim),
	          std::vector<int64_t>(head_dim)};
	auto code = [&]() -> int8_t {
		if (kind == 1) return rng.Next() & 1 ? int8_t{127} : int8_t{-128};
		return static_cast<int8_t>(rng.InRange(-128, 127));
	};
	for (auto& x : c.q) x = code();
	for (auto& x : c.keys) x = code();
	for (auto& r : c.ratio) {
		if (kind == 0) r = rng.InRange(1, kQ31RatioMax);
		else if (kind == 1) r = (rng.Next() % 3 == 0) ? 1 : (rng.Next() & 1 ? kQ31RatioMax : kQ31RatioMax - 1);
		else r = kQ31RatioMax;
	}
	return c;
}

// 7.S5b: every limb sum at §5.5's int32 margin. Channel products w = q * ratio with a0 = a1 = 32,767 (w
// is -1, or 2^30 - 1) against keys of -128 (or 127) on every channel: at head_dim 512 each limb's per-key
// sum is -128 * 32,767 * 512 = -2,147,418,112 (margin 65,535 to int32's minimum) or 127 * 32,767 * 512 =
// 2,130,690,048. At 516 the same rows are past the guard (and would wrap an int32 lane).
inline Q31Case MakeQ31MarginCase(size_t head_dim, size_t width, bool negative_w, int8_t key) {
	Q31Case c{negative_w ? "7.S5b margin corner, w = -1, key fill" : "7.S5b margin corner, w = 2^30 - 1, key fill",
	          head_dim, width, std::vector<int8_t>(head_dim, negative_w ? int8_t{-1} : int8_t{1}),
	          std::vector<int8_t>(width * head_dim, key),
	          std::vector<int64_t>(head_dim, negative_w ? INT64_C(1) : (INT64_C(1) << 30) - 1)};
	return c;
}

// 7.S5c: totals on and beside the rounding ties x = +-2^30 (mod 2^31). Two channels carry the products:
// ratio 2^30 - 5 and 5 (every limb nonzero), q = +-1 on both, key t on both, so the key's total is exactly
// q * t * 2^30; odd t is a tie, which RoundingDivideByPOT rounds away from zero. The remaining channels
// add +-1 (ratio 1) on some keys to land one beside the tie.
inline Q31Case MakeQ31TieCase(int8_t q_sign) {
	const size_t hd = 64, width = 24;
	Q31Case c{q_sign > 0 ? "7.S5c ties, q = +1" : "7.S5c ties, q = -1", hd, width, std::vector<int8_t>(hd, 0),
	          std::vector<int8_t>(width * hd, 0), std::vector<int64_t>(hd, 1)};
	c.q[0] = q_sign;
	c.q[1] = q_sign;
	c.q[2] = 1;
	c.ratio[0] = (INT64_C(1) << 30) - 5;
	c.ratio[1] = 5;
	static constexpr int8_t kT[] = {1, -1, 3, -3, 5, -5, 127, -127, -128, 2, -2, 0};
	for (size_t j = 0; j < width; ++j) {
		const int8_t t = kT[j % 12];
		c.keys[j * hd + 0] = t;
		c.keys[j * hd + 1] = t;
		c.keys[j * hd + 2] = j < 12 ? int8_t{0} : (j % 2 ? int8_t{1} : int8_t{-1});  // beside the tie
	}
	return c;
}

template <class Fn>
void ForEachQ31Case(Fn&& fn) {
	Rng rng(0x5E5E'0031'7153'0005ULL);
	for (size_t hd : kQ31HeadDims)
		for (size_t w : kQ31Widths)
			for (int kind = 0; kind < 3; ++kind) fn(MakeQ31GridCase(hd, w, kind, rng));
	for (size_t hd : {size_t{512}, size_t{516}})
		for (size_t w : {size_t{1}, size_t{17}})
			for (bool neg : {true, false})
				for (int8_t key : {int8_t{-128}, int8_t{127}}) fn(MakeQ31MarginCase(hd, w, neg, key));
	fn(MakeQ31TieCase(1));
	fn(MakeQ31TieCase(-1));
}

// The whole S5 set through `row(q, keys, ratio, head_dim, width, out)`: per call, width, head_dim and every
// score (the output row is poisoned before the call).
template <class Emit, class Row>
void RunQ31Cases(Emit& emit, Row&& row) {
	ForEachQ31Case([&](const Q31Case& c) {
		std::vector<int64_t> out(c.width, kQ31Poison);
		row(c.q.data(), c.keys.data(), c.ratio.data(), c.head_dim, c.width, out.data());
		emit(static_cast<int64_t>(c.width));
		emit(static_cast<int64_t>(c.head_dim));
		for (int64_t x : out) emit(x);
	});
}

}  // namespace superslm_attention_cases

#endif  // SUPERSLM_TESTS_SUPPORT_ATTENTION_CASES_H

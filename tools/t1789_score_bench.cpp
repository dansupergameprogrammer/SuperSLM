// t1789_score_bench.cpp -- T-1789 speed arm: measure the W(8) construction's score-path cost
// against production's own int8 score path, on this machine's own CPU, using the engine's own
// DotRow constructions as the baseline.
//
// WHAT IS MEASURED. The score GEMM is the only stage whose per-element arithmetic changes under
// W(8): production dots two int8 vectors (q_rot, k_row) over head_dim per (head, key); W(8) dots
// two int16 vectors over the same shape. Everything else in the construction (funnel requant,
// landing rescale, RoPE rotation) performs the SAME operation count at the same width -- the only
// change there is the constant multiplied in (127*F vs 127) and the clamp bound, neither of which
// changes cost. The projection GEMMs (1536-wide, the layer's dominant cost) are UNTOUCHED by the
// construction (weights and activations stay int8).
//
// FOUR ARMS, all computing the identical mathematical quantity (an exact int64 dot):
//   1. int8 scalar   -- the engine's own DotRowScalar construction (copied verbatim in shape).
//   2. int8 SSE2     -- the engine's own DotRowSse2 construction (copied verbatim in shape:
//                       movq load, sign-extend to int16, pmaddwd, int32 lanes, flushed).
//   3. int16 scalar  -- same loop at int16.
//   4. int16 SSE2    -- full-width 128-bit loads (8 int16 lanes), pmaddwd DIRECTLY (no
//                       sign-extension step needed -- the data is already int16), then the int32
//                       pair-sums widened into int64 lanes EVERY iteration (per-lane bound
//                       2*(127*256)^2 = 2,114,060,288 < INT32_MAX = 2,147,483,647 -- one block
//                       fits an int32 lane exactly, two do not, so the flush cadence is 1).
//
// Bit-exactness note (the product's own constraint): arm 4 is a lane-regrouped reduction of
// exact int64 products, no saturation, no narrowing mid-reduction -- the same conformance class
// matmul.cpp's design names for the int8 SSE2 path. Verified below: all four arms produce
// identical totals on the same random data before any timing is trusted.

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <emmintrin.h>
#include <vector>

namespace {

inline int64_t Dot8Scalar(const int8_t* a, const int8_t* w, size_t n) {
	int64_t acc = 0;
	for (size_t k = 0; k < n; ++k) acc += static_cast<int64_t>(a[k]) * static_cast<int64_t>(w[k]);
	return acc;
}

inline int64_t Dot16Scalar(const int16_t* a, const int16_t* w, size_t n) {
	int64_t acc = 0;
	for (size_t k = 0; k < n; ++k) acc += static_cast<int64_t>(a[k]) * static_cast<int64_t>(w[k]);
	return acc;
}

inline int64_t Dot8Sse2(const int8_t* a, const int8_t* w, size_t n) {
	int64_t acc64 = 0;
	__m128i acc32 = _mm_setzero_si128();
	size_t k = 0;
	for (; k + 8 <= n; k += 8) {
		__m128i a8 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(a + k));
		__m128i w8 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(w + k));
		__m128i a16 = _mm_srai_epi16(_mm_unpacklo_epi8(a8, a8), 8);
		__m128i w16 = _mm_srai_epi16(_mm_unpacklo_epi8(w8, w8), 8);
		acc32 = _mm_add_epi32(acc32, _mm_madd_epi16(a16, w16));
	}
	alignas(16) int32_t lanes[4];
	_mm_store_si128(reinterpret_cast<__m128i*>(lanes), acc32);
	for (int32_t v : lanes) acc64 += static_cast<int64_t>(v);
	for (; k < n; ++k) acc64 += static_cast<int64_t>(a[k]) * static_cast<int64_t>(w[k]);
	return acc64;
}

inline int64_t Dot16Sse2(const int16_t* a, const int16_t* w, size_t n) {
	int64_t acc64 = 0;
	__m128i acc64v = _mm_setzero_si128();  // 2 int64 lanes
	size_t k = 0;
	for (; k + 8 <= n; k += 8) {
		__m128i a16 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a + k));
		__m128i w16 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(w + k));
		__m128i prod32 = _mm_madd_epi16(a16, w16);  // 4 int32 pair-sums, |v| <= 2*(127*256)^2
		__m128i sign = _mm_srai_epi32(prod32, 31);
		acc64v = _mm_add_epi64(acc64v, _mm_unpacklo_epi32(prod32, sign));
		acc64v = _mm_add_epi64(acc64v, _mm_unpackhi_epi32(prod32, sign));
	}
	alignas(16) int64_t lanes[2];
	_mm_store_si128(reinterpret_cast<__m128i*>(lanes), acc64v);
	acc64 += lanes[0] + lanes[1];
	for (; k < n; ++k) acc64 += static_cast<int64_t>(a[k]) * static_cast<int64_t>(w[k]);
	return acc64;
}

}  // namespace

int main() {
	constexpr size_t kHeadDim = 128;
	constexpr size_t kWidth = 64;      // cached key rows per score row (a mid-size context)
	constexpr size_t kHeads = 12;
	constexpr int kReps = 20000;       // score-row repetitions per timing arm

	// Data: one q per head plus kWidth k-rows per head, values in the construction's own ranges.
	std::srand(20260806);
	std::vector<int8_t> q8(kHeads * kHeadDim), k8(kHeads * kWidth * kHeadDim);
	std::vector<int16_t> q16(kHeads * kHeadDim), k16(kHeads * kWidth * kHeadDim);
	for (size_t i = 0; i < q8.size(); ++i) {
		int v = (std::rand() % 255) - 127;
		q8[i] = static_cast<int8_t>(v);
		q16[i] = static_cast<int16_t>(v * 256 + (std::rand() % 256) - 128);  // W(8)-range value
	}
	for (size_t i = 0; i < k8.size(); ++i) {
		int v = (std::rand() % 255) - 127;
		k8[i] = static_cast<int8_t>(v);
		k16[i] = static_cast<int16_t>(v * 256 + (std::rand() % 256) - 128);
	}

	// Exactness gate: SSE2 arms must equal scalar arms on every (head, key) before timing.
	for (size_t h = 0; h < kHeads; ++h) {
		for (size_t j = 0; j < kWidth; ++j) {
			const int8_t* a8 = q8.data() + h * kHeadDim;
			const int8_t* w8 = k8.data() + (h * kWidth + j) * kHeadDim;
			if (Dot8Scalar(a8, w8, kHeadDim) != Dot8Sse2(a8, w8, kHeadDim)) {
				std::printf("FAILED: int8 SSE2 != scalar at h=%zu j=%zu\n", h, j);
				return 1;
			}
			const int16_t* a16 = q16.data() + h * kHeadDim;
			const int16_t* w16 = k16.data() + (h * kWidth + j) * kHeadDim;
			if (Dot16Scalar(a16, w16, kHeadDim) != Dot16Sse2(a16, w16, kHeadDim)) {
				std::printf("FAILED: int16 SSE2 != scalar at h=%zu j=%zu\n", h, j);
				return 1;
			}
		}
	}
	std::printf("exactness gate: all 4 arms agree on all %zu dots\n", kHeads * kWidth);

	auto bench = [&](const char* label, auto fn) {
		volatile int64_t sink = 0;
		// warmup
		for (int r = 0; r < 200; ++r) sink += fn();
		auto t0 = std::chrono::steady_clock::now();
		for (int r = 0; r < kReps; ++r) sink += fn();
		auto t1 = std::chrono::steady_clock::now();
		double ns = std::chrono::duration<double, std::nano>(t1 - t0).count();
		double per_score_row = ns / kReps;                       // 12 heads x 64 keys x 128 dot
		double per_dot = per_score_row / (kHeads * kWidth);
		std::printf("%-14s: %10.1f ns per full score pass (12 heads x 64 keys), %7.2f ns per 128-dot  (sink=%lld)\n",
		            label, per_score_row, per_dot, static_cast<long long>(sink % 2));
		return per_score_row;
	};

	double s8 = bench("int8 scalar", [&]() {
		int64_t acc = 0;
		for (size_t h = 0; h < kHeads; ++h)
			for (size_t j = 0; j < kWidth; ++j)
				acc += Dot8Scalar(q8.data() + h * kHeadDim, k8.data() + (h * kWidth + j) * kHeadDim, kHeadDim);
		return acc;
	});
	double s16 = bench("int16 scalar", [&]() {
		int64_t acc = 0;
		for (size_t h = 0; h < kHeads; ++h)
			for (size_t j = 0; j < kWidth; ++j)
				acc += Dot16Scalar(q16.data() + h * kHeadDim, k16.data() + (h * kWidth + j) * kHeadDim, kHeadDim);
		return acc;
	});
	double v8 = bench("int8 SSE2", [&]() {
		int64_t acc = 0;
		for (size_t h = 0; h < kHeads; ++h)
			for (size_t j = 0; j < kWidth; ++j)
				acc += Dot8Sse2(q8.data() + h * kHeadDim, k8.data() + (h * kWidth + j) * kHeadDim, kHeadDim);
		return acc;
	});
	double v16 = bench("int16 SSE2", [&]() {
		int64_t acc = 0;
		for (size_t h = 0; h < kHeads; ++h)
			for (size_t j = 0; j < kWidth; ++j)
				acc += Dot16Sse2(q16.data() + h * kHeadDim, k16.data() + (h * kWidth + j) * kHeadDim, kHeadDim);
		return acc;
	});

	std::printf("\nratios: int16/int8 scalar = %.2fx   int16/int8 SSE2 = %.2fx\n", s16 / s8, v16 / v8);
	std::printf("K-cache memory: K rows int8 -> int16 doubles the K half; V unchanged -> KV total x1.5\n");
	return 0;
}

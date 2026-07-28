// SuperSLM raw int8x8 matmul accumulate (C17/C19-C22 site, S2.5).
//
// See include/superslm/matmul.h for the runtime contract and
// SuperSLM_matmul_subslot_design-2026-07-20.md §3/§5/§7 for the construction.
//
// The scalar reference (design §5) is the normative construction: every intermediate
// int64, both factors widened to int64 before the multiply, no saturation, no
// rounding, no branch, no data-dependent trip count. The SSE2 path below (design §7's
// "widening (non-saturating) multiply-accumulate" conformant class) is a lane-
// regrouped reduction of the exact same int64 sum -- by design §4's associativity
// argument, any traversal order of exact int64 products must produce the bit-identical
// total, and this path never narrows or saturates mid-reduction. SSE2 is the
// unconditional x64 baseline (no runtime CPUID dispatch needed -- every x64 chip has
// it), so it is used as the always-available SIMD specialization; non-x64 builds fall
// back to the scalar reference, which is equally conformant, only slower.
#include "superslm/matmul.h"

#include <cassert>

// S-HARDEN-6 (T-180, design Sec2.1 component 3): a build-time override that
// forces the scalar reference path even on x64, so the S-HARDEN-6 digest
// jobs can compare a scalar-forced digest against the SIMD-enabled one --
// the "scalar-forced vs SIMD-enabled" axis component 2's digest-and-compare
// pattern needs and today has no build-time override to select. Checked
// BEFORE the architecture test, so defining it wins regardless of target.
#if defined(SUPERSLM_FORCE_SCALAR_MATMUL)
#define SUPERSLM_MATMUL_HAVE_SSE2 0
#elif defined(_M_X64) || defined(__x86_64__)
#define SUPERSLM_MATMUL_HAVE_SSE2 1
#include <emmintrin.h>
#else
#define SUPERSLM_MATMUL_HAVE_SSE2 0
#endif

namespace superslm {
namespace {

// --- The scalar reference (design §5, normative) -------------------------------

inline int64_t DotRowScalar(const int8_t* activations, const int8_t* weights,
                             size_t in_channels) {
	int64_t acc = 0;  // int64 throughout -- never narrowed mid-reduction
	for (size_t k = 0; k < in_channels; ++k) {
		acc += static_cast<int64_t>(activations[k]) * static_cast<int64_t>(weights[k]);
	}
	return acc;
}

#if SUPERSLM_MATMUL_HAVE_SSE2

// --- The SSE2 specialization (design §7's conformant, widening-non-saturating
//     class) ---------------------------------------------------------------------
//
// Processes 8 int8 lanes per iteration:
//   1. Unaligned-safe 64-bit loads (movq -- no alignment precondition, design §12
//      dim 4's unaligned-pointer cell).
//   2. Manual SSE2 sign-extend int8 -> int16 (unpack-with-self then arithmetic-shift
//      right by 8) -- no SSE4.1 pmovsx dependency, and exact: int8's full range fits
//      int16 with no truncation.
//   3. _mm_madd_epi16: a WIDENING (not saturating) multiply of int16 pairs, pairwise-
//      summed into int32 lanes. Per-lane magnitude bound: two products, each at most
//      |−128 * 127| = 16256 (the attainable extreme) or 128*128=16384 (the
//      conservative bound design §8 derives against) -- pairwise sum <= 32768, far
//      under INT32_MAX. This is the exact instruction design §7 names as conformant
//      ("the same shape the scalar reference itself uses"); it is not the forbidden
//      saturating-int16-intermediate idiom (§7 excludes that construction outright).
//   4. The int32 partial-sum register is folded into the int64 running accumulator
//      every kFlushBlocks iterations -- chosen with a large safety margin so the
//      int32 partial can never approach overflow even for the suite's deep
//      (up to 5,000,000-element) in_channels cases; folding widens to int64 before
//      adding, so the running total is exact, matching the scalar reference bit-for-
//      bit by associativity (design §4).
//
// Tail remainder (in_channels not a multiple of 8, design §12 dim 4's shape hazard)
// falls through to the scalar accumulation for the last < 8 elements.
inline int64_t DotRowSse2(const int8_t* activations, const int8_t* weights,
                           size_t in_channels) {
	int64_t acc64 = 0;
	__m128i acc32 = _mm_setzero_si128();  // 4 int32 lanes, partial sum for the current flush window

	// Per flush-window bound: each block contributes at most 32768 magnitude to any
	// one lane (see the derivation above). kFlushBlocks * 32768 must stay comfortably
	// under INT32_MAX (2,147,483,647); 16384 gives 536,870,912 -- a 4x margin.
	constexpr size_t kFlushBlocks = 16384;

	size_t k = 0;
	size_t blocks_since_flush = 0;
	for (; k + 8 <= in_channels; k += 8) {
		__m128i a8 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(activations + k));
		__m128i w8 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(weights + k));

		__m128i a16 = _mm_srai_epi16(_mm_unpacklo_epi8(a8, a8), 8);
		__m128i w16 = _mm_srai_epi16(_mm_unpacklo_epi8(w8, w8), 8);

		__m128i prod32 = _mm_madd_epi16(a16, w16);  // widening, non-saturating
		acc32 = _mm_add_epi32(acc32, prod32);

		if (++blocks_since_flush == kFlushBlocks) {
			alignas(16) int32_t lanes[4];
			_mm_store_si128(reinterpret_cast<__m128i*>(lanes), acc32);
			for (int32_t v : lanes) acc64 += static_cast<int64_t>(v);
			acc32 = _mm_setzero_si128();
			blocks_since_flush = 0;
		}
	}
	{
		alignas(16) int32_t lanes[4];
		_mm_store_si128(reinterpret_cast<__m128i*>(lanes), acc32);
		for (int32_t v : lanes) acc64 += static_cast<int64_t>(v);
	}

	for (; k < in_channels; ++k) {  // scalar tail remainder
		acc64 += static_cast<int64_t>(activations[k]) * static_cast<int64_t>(weights[k]);
	}
	return acc64;
}

#endif  // SUPERSLM_MATMUL_HAVE_SSE2

inline int64_t DotRow(const int8_t* activations, const int8_t* weights, size_t in_channels) {
#if SUPERSLM_MATMUL_HAVE_SSE2
	return DotRowSse2(activations, weights, in_channels);
#else
	return DotRowScalar(activations, weights, in_channels);
#endif
}

}  // namespace

int64_t DotRowScalarRef(const int8_t* activations, const int8_t* weights, size_t in_channels) {
	// Test-reachable wrapper around the anonymous-namespace scalar reference (design
	// §5) -- see matmul.h's declaration. Calls the exact same construction DotRow falls
	// back to on non-x64 builds; does not participate in DotRow's dispatch, so the
	// shipping SSE2 selection above is unchanged.
	return DotRowScalar(activations, weights, in_channels);
}

void GemmInt8AccumulateRow(const int8_t* activations, const int8_t* weights,
                            size_t in_channels, size_t out_channels, int64_t* out_acc) {
	assert(in_channels > 0 && "GemmInt8AccumulateRow: in_channels below the architectural floor "
	                          "(design §12 dim 4) -- caller contract violation");
	assert(out_channels > 0 && "GemmInt8AccumulateRow: out_channels below the architectural floor "
	                           "(design §12 dim 4) -- caller contract violation");
	for (size_t j = 0; j < out_channels; ++j) {
		out_acc[j] = DotRow(activations, weights + j * in_channels, in_channels);
	}
}

void GemmInt8Accumulate(const int8_t* activations, const int8_t* weights,
                         size_t num_tokens, size_t in_channels, size_t out_channels,
                         int64_t* out_acc) {
	assert(num_tokens > 0 && "GemmInt8Accumulate: num_tokens must be >= 1");
	// Each row is independent (design §3/§9's "no cross-row reduction") -- literally
	// GemmInt8AccumulateRow applied per row, stacked row-major [num_tokens,
	// out_channels]. This is not an optimization shortcut; it is the construction
	// design §3 specifies and design §12 dim 7's stacking-equivalence cell asserts
	// against.
	for (size_t t = 0; t < num_tokens; ++t) {
		GemmInt8AccumulateRow(activations + t * in_channels, weights, in_channels, out_channels,
		                      out_acc + t * out_channels);
	}
}

void NarrowAccumulatorToI32(const int64_t* wide_row, size_t n, int32_t* out_i32) {
	// The ONLY narrowing point (design §3/§4). Caller-ensures convention: UB if the
	// declared MatmulAccumWidth was wrong for this tensor (i.e. a value does not fit
	// int32) -- matching MaxAbsReduce/ShiftByMax, not a runtime-checked cast.
	for (size_t i = 0; i < n; ++i) {
		out_i32[i] = static_cast<int32_t>(wide_row[i]);
	}
}

// F-S3-6/C32 (Claude/Curie/superslm-s3.3-attention-interior-test-design-
// 2026-07-28.md §6.4, §11): the probability x value context accumulate.
void GemmProbQ15Accumulate(const int64_t* probs, const int8_t* values, size_t width,
                            size_t head_dim, int64_t* out_ctx) {
	// out_ctx[d] = Sum_k probs[k] * values[k*head_dim + d]. Exact int64
	// accumulation, no saturation, no rounding (F-S3-6's derived bound:
	// |Sum_k p_k*v_k| <= 2^15*127 < 2^22, independent of context length --
	// far inside int64, so no intermediate can overflow).
	for (size_t d = 0; d < head_dim; ++d) {
		out_ctx[d] = 0;
	}
	for (size_t k = 0; k < width; ++k) {
		const int64_t p = probs[k];
		const int8_t* row = values + k * head_dim;
		for (size_t d = 0; d < head_dim; ++d) {
			out_ctx[d] += p * static_cast<int64_t>(row[d]);
		}
	}
}

}  // namespace superslm

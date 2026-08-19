// SuperSLM raw int8x8 matmul accumulate (C17/C19-C22 site, S2.5).
//
// See include/superslm/matmul.h for the runtime contract and
// SuperSLM_matmul_subslot_design-2026-07-20.md §3/§5/§7 for the construction, and
// Claude/Vitruvius/t2149-avx-kernel-design-2026-08-18.md §5-§8 (T-2149) for the
// widened SSE2/AVX2/AVX-512 dispatch this file implements.
//
// The scalar reference (design §5) is the normative construction: every intermediate
// int64, both factors widened to int64 before the multiply, no saturation, no
// rounding, no branch, no data-dependent trip count. Every SIMD specialization below
// (T-2149 design §7's "widening (non-saturating) multiply-accumulate" conformant
// class) is a lane-regrouped reduction of the exact same int64 sum -- by design §4's
// associativity argument, any traversal order of exact int64 products must produce
// the bit-identical total, and no path ever narrows or saturates mid-reduction.
//
// Dispatch (T-2149 design §6.1): SSE2 is the unconditional x64 architectural floor
// (every x64 chip has it); AVX2 and AVX-512 are wider tiers selected at runtime, by a
// cached CPUID+XGETBV probe (design §6.2), whenever neither a force macro nor a
// non-x64 target is in play. Four mutually-exclusive force macros
// (SUPERSLM_FORCE_{SCALAR,SSE2,AVX2,AVX512}_MATMUL) let a build pin one tier
// deliberately, independent of ambient hardware or scheduler behavior (design §6.3) --
// non-x64 builds fall back to the scalar reference, which is equally conformant, only
// slower.
#include "superslm/matmul.h"

#include <atomic>
#include <cassert>
#include <cstdint>

#ifdef SUPERSLM_ENABLE_MATMUL_DISPATCH_INSTRUMENT
// T-2158 red suite test-only seam (design §10 dimensions 1/3) -- mirrors
// src/bad_alloc_wrap.h's own SUPERSLM_ENABLE_BAD_ALLOC_INJECTION convention exactly.
// The superslm_test_injection library target and the three SSE2/AVX2/AVX512 *_forced
// library targets (all under SUPERSLM_T2149_AVX_TIERS_BUILT, CMakeLists.txt) define this
// macro so both halves of the dispatch-cache cell -- "== 1" in the non-forced binary,
// "== 0" in each forced binary -- exist and run (design §10 dimension 1); the production
// superslm library and sslm_verify never do, so a release build never references
// tests/support/* at all.
#include "support/matmul_dispatch_instrument.h"
#endif

// T-2149 design §6.1: force selection (mutually exclusive; at most one defined by any
// single build) -- enforced by #error on the invalid combination so a future
// build-system typo fails at compile time, not by producing a silently-wrong dispatch.
#if (defined(SUPERSLM_FORCE_SCALAR_MATMUL) + defined(SUPERSLM_FORCE_SSE2_MATMUL) + \
     defined(SUPERSLM_FORCE_AVX2_MATMUL) + defined(SUPERSLM_FORCE_AVX512_MATMUL)) > 1
#error "At most one of SUPERSLM_FORCE_SCALAR_MATMUL / SUPERSLM_FORCE_SSE2_MATMUL / " \
       "SUPERSLM_FORCE_AVX2_MATMUL / SUPERSLM_FORCE_AVX512_MATMUL may be defined in a " \
       "single build (design §6.1) -- a build defining more than one has stated a " \
       "contradictory intent, and this is a compile-time build-system logic error, " \
       "not a silent substitution."
#endif

// T-2149 design §6.1: compile-time capability -- SUPERSLM_MATMUL_HAVE_SIMD_X64 (renamed
// from the old SUPERSLM_MATMUL_HAVE_SSE2, same condition) is defined once, in
// include/superslm/matmul.h, so the T-2158 test seam can scope its own coverage cells
// to the identical condition (fold round 4, D-SLM3568) rather than maintain a second
// copy. Unconditional (not gated by any force macro): this is a fact about the target,
// never about which tier a build has chosen to pin.

#if SUPERSLM_MATMUL_HAVE_SIMD_X64
#include <emmintrin.h>   // SSE2
#include <immintrin.h>   // AVX2 / AVX-512 intrinsics
#if defined(_MSC_VER)
#include <intrin.h>       // __cpuid, __cpuidex, _xgetbv
#else
#include <cpuid.h>        // __cpuid_count
#endif
#endif  // SUPERSLM_MATMUL_HAVE_SIMD_X64

// T-2149 design §6.4: GCC/Clang require the target ISA enabled per-function (function
// multiversioning) for an AVX2/AVX-512 intrinsic to compile; a translation-unit-wide
// -mavx2/-mavx512bw flag is rejected (it would let the auto-vectorizer use those
// instructions anywhere else in this TU, breaking the "must run on hardware without
// AVX-512" requirement for everything else this file compiles -- confirmed standing,
// unconditional, by tools/ci/check_matmul_avx_isolation.py, design §10 dimension 7d).
// MSVC does not gate intrinsic use by /arch at all (documented MSVC behavior, design
// §6.4) -- DotRowAvx2/DotRowAvx512 need no attribute there. ClangCL follows MSVC's
// command-line conventions but its code generator is LLVM's, which gates AVX2/AVX-512
// intrinsics on target features exactly as Clang does on Linux (design §6.4, corrected
// fold round 4, D-SLM3567) -- clang-cl also does not define __GNUC__, so it needs the
// attributed path for two independent reasons.
#if defined(__clang__) || (defined(__GNUC__) && !defined(_MSC_VER))
#define SUPERSLM_AVX2_TARGET __attribute__((target("avx2")))
#define SUPERSLM_AVX512_TARGET __attribute__((target("avx512f,avx512bw")))
#else
#define SUPERSLM_AVX2_TARGET
#define SUPERSLM_AVX512_TARGET
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

#if SUPERSLM_MATMUL_HAVE_SIMD_X64

// --- The SSE2 specialization (design §7's conformant, widening-non-saturating
//     class; the unconditional x64 architectural floor) -------------------------
//
// Processes 8 int8 lanes per iteration:
//   1. Unaligned-safe 64-bit loads (movq -- no alignment precondition, design §12
//      dim 4's unaligned-pointer cell).
//   2. Manual SSE2 sign-extend int8 -> int16 (unpack-with-self then arithmetic-shift
//      right by 8) -- no SSE4.1 pmovsx dependency, and exact: int8's full range fits
//      int16 with no truncation.
//   3. _mm_madd_epi16: a WIDENING (not saturating) multiply of int16 pairs, pairwise-
//      summed into int32 lanes. Per-lane magnitude bound: two products, each at most
//      |-128 * 127| = 16256 (the attainable extreme) or 128*128=16384 (the
//      conservative bound design §5.2 derives against) -- pairwise sum <= 32768, far
//      under INT32_MAX. This is the exact instruction design §7 names as conformant
//      ("the same shape the scalar reference itself uses"); it is not the forbidden
//      saturating-int16-intermediate idiom (§7 excludes that construction outright).
//   4. The int32 partial-sum register is folded into the int64 running accumulator
//      every kFlushBlocks iterations -- chosen with a large safety margin so the
//      int32 partial can never approach overflow even for the suite's deep
//      (up to 5,000,000-element) in_channels cases; folding widens to int64 before
//      adding, so the running total is exact, matching the scalar reference bit-for-
//      bit by associativity (design §4). The bound is per-lane and per-iteration,
//      independent of lane count (design §5.2), so kFlushBlocks is the SAME literal
//      value, shared unchanged, across all three SIMD tiers below.
//
// Tail remainder (in_channels not a multiple of 8, design §12 dim 4's shape hazard)
// falls through to the scalar accumulation for the last < 8 elements -- one main loop,
// one scalar tail per tier, no cascading intermediate-width fallthrough (design §5.1,
// D-SLM3504).
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

// --- The AVX2 specialization (design §5, 16 int8 lanes/iteration) --------------
//
// Structurally identical to DotRowSse2 one tier up:
//   1. Unaligned 128-bit load (_mm_loadu_si128, 16 bytes) per operand.
//   2. Sign-extend via _mm256_cvtepi8_epi16 (VPMOVSXBW, AVX2) -- an exact
//      sign-extension instruction, replacing DotRowSse2's manual unpack+shift; no
//      precision or range hazard for int8 into int16.
//   3. _mm256_madd_epi16 (VPMADDWD, 256-bit form) -- the identical per-lane operation
//      _mm_madd_epi16 performs, replicated across 8 lanes instead of 4.
//   4. Flush to int64 every kFlushBlocks iterations (same named constant, same value
//      -- design §5.2's bound is per-lane, width-independent), horizontal-summing all
//      8 int32 lanes.
//   5. Scalar tail for in_channels % 16, same convention as DotRowSse2's own tail.
SUPERSLM_AVX2_TARGET
inline int64_t DotRowAvx2(const int8_t* activations, const int8_t* weights,
                           size_t in_channels) {
	int64_t acc64 = 0;
	__m256i acc32 = _mm256_setzero_si256();
	constexpr size_t kFlushBlocks = 16384;

	size_t k = 0;
	size_t blocks_since_flush = 0;
	for (; k + 16 <= in_channels; k += 16) {
		__m128i a8 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(activations + k));
		__m128i w8 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(weights + k));

		__m256i a16 = _mm256_cvtepi8_epi16(a8);
		__m256i w16 = _mm256_cvtepi8_epi16(w8);

		__m256i prod32 = _mm256_madd_epi16(a16, w16);  // widening, non-saturating
		acc32 = _mm256_add_epi32(acc32, prod32);

		if (++blocks_since_flush == kFlushBlocks) {
			alignas(32) int32_t lanes[8];
			_mm256_store_si256(reinterpret_cast<__m256i*>(lanes), acc32);
			for (int32_t v : lanes) acc64 += static_cast<int64_t>(v);
			acc32 = _mm256_setzero_si256();
			blocks_since_flush = 0;
		}
	}
	{
		alignas(32) int32_t lanes[8];
		_mm256_store_si256(reinterpret_cast<__m256i*>(lanes), acc32);
		for (int32_t v : lanes) acc64 += static_cast<int64_t>(v);
	}

	for (; k < in_channels; ++k) {  // scalar tail remainder
		acc64 += static_cast<int64_t>(activations[k]) * static_cast<int64_t>(weights[k]);
	}
	return acc64;
}

// --- The AVX-512 specialization (design §5, 32 int8 lanes/iteration) -----------
//
// Structurally identical to DotRowAvx2 one tier up:
//   1. Unaligned 256-bit load (_mm256_loadu_si256, 32 bytes) per operand.
//   2. Sign-extend via _mm512_cvtepi8_epi16 (VPMOVSXBW, AVX512BW).
//   3. _mm512_madd_epi16 (VPMADDWD, 512-bit form) -- same per-lane semantics,
//      16xint32 partial-sum lanes.
//   4. Flush every kFlushBlocks iterations (same shared constant), horizontal-summing
//      16 int32 lanes.
//   5. Scalar tail for in_channels % 32.
SUPERSLM_AVX512_TARGET
inline int64_t DotRowAvx512(const int8_t* activations, const int8_t* weights,
                             size_t in_channels) {
	int64_t acc64 = 0;
	__m512i acc32 = _mm512_setzero_si512();
	constexpr size_t kFlushBlocks = 16384;

	size_t k = 0;
	size_t blocks_since_flush = 0;
	for (; k + 32 <= in_channels; k += 32) {
		__m256i a8 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(activations + k));
		__m256i w8 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(weights + k));

		__m512i a16 = _mm512_cvtepi8_epi16(a8);
		__m512i w16 = _mm512_cvtepi8_epi16(w8);

		__m512i prod32 = _mm512_madd_epi16(a16, w16);  // widening, non-saturating
		acc32 = _mm512_add_epi32(acc32, prod32);

		if (++blocks_since_flush == kFlushBlocks) {
			alignas(64) int32_t lanes[16];
			_mm512_store_si512(reinterpret_cast<void*>(lanes), acc32);
			for (int32_t v : lanes) acc64 += static_cast<int64_t>(v);
			acc32 = _mm512_setzero_si512();
			blocks_since_flush = 0;
		}
	}
	{
		alignas(64) int32_t lanes[16];
		_mm512_store_si512(reinterpret_cast<void*>(lanes), acc32);
		for (int32_t v : lanes) acc64 += static_cast<int64_t>(v);
	}

	for (; k < in_channels; ++k) {  // scalar tail remainder
		acc64 += static_cast<int64_t>(activations[k]) * static_cast<int64_t>(weights[k]);
	}
	return acc64;
}

// --- Runtime CPUID+XGETBV probe (design §6.2, fold round 2's corrected four-condition
//     AVX-512 gate) --------------------------------------------------------------

enum class DotRowTier { kSse2, kAvx2, kAvx512 };

#if defined(_MSC_VER)
inline void QueryCpuId(int leaf, int subleaf, int regs[4]) {
	__cpuidex(regs, leaf, subleaf);
}
inline unsigned long long QueryXcr0() {
	return _xgetbv(0);  // _XCR_XFEATURE_ENABLED_MASK == 0
}
#else
inline void QueryCpuId(int leaf, int subleaf, int regs[4]) {
	unsigned int eax = 0, ebx = 0, ecx = 0, edx = 0;
	__cpuid_count(static_cast<unsigned int>(leaf), static_cast<unsigned int>(subleaf), eax, ebx,
	              ecx, edx);
	regs[0] = static_cast<int>(eax);
	regs[1] = static_cast<int>(ebx);
	regs[2] = static_cast<int>(ecx);
	regs[3] = static_cast<int>(edx);
}
inline unsigned long long QueryXcr0() {
	unsigned int eax = 0, edx = 0;
	__asm__ volatile("xgetbv" : "=a"(eax), "=d"(edx) : "c"(0));
	return (static_cast<unsigned long long>(edx) << 32) | eax;
}
#endif

// Pure decision logic (design §6.2, corrected T-2189 finding 1 / D-SLM3689 to add the
// max-basic-leaf guard below) -- takes already-fetched (or synthetic) register fields and
// touches no hardware itself, so it is independently testable with fabricated CPUID data
// without a hardware CPUID shim (see ResolveDotRowTier, declared in matmul.h,
// mirroring DotRowScalarRef's own test-reachable-wrapper pattern below). The four-step
// resolution:
//   0. CPUID leaf 0 -- EAX is the highest basic leaf the CPU supports. Leaf 7 is
//      architecturally undefined on any CPU whose max basic leaf is below 7 (some
//      implementations echo back the highest supported leaf's output or return stale
//      register state instead of zero) -- so leaf 7 is consulted ONLY when max_leaf >= 7;
//      below that, every leaf-7-derived bit is treated as unset and resolution falls
//      through to the unconditional SSE2 floor. This was the previous defect: leaf 7 was
//      queried and its bits interpreted unconditionally, with no max-leaf check.
//   1. CPUID leaf 7, sub-leaf 0 -- EBX bit 5 (AVX2), bit 16 (AVX512F), bit 30 (AVX512BW).
//   2. CPUID leaf 1 -- ECX bit 27 (OSXSAVE, confirms the OS exposes XGETBV).
//   3. XGETBV(0), only if OSXSAVE is set -- bits 1-2 (XMM/YMM state), bits 5-7
//      (opmask/ZMM_hi256/Hi16_ZMM state).
//   4. AVX-512 selected only if (AVX512F AND AVX512BW) AND (bits 1-2 AND bits 5-7);
//      else AVX2 if (AVX2 bit) AND (bits 1-2); else SSE2 (the unconditional floor).
inline DotRowTier ResolveDotRowTierFromFields(int max_basic_leaf, int leaf1_ecx,
                                               int leaf7_ebx, unsigned long long xcr0) {
	const bool osxsave = (leaf1_ecx & (1 << 27)) != 0;  // leaf 1, ECX bit 27

	bool avx2_bit = false;       // leaf 7/0, EBX bit 5
	bool avx512f_bit = false;    // leaf 7/0, EBX bit 16
	bool avx512bw_bit = false;   // leaf 7/0, EBX bit 30
	if (max_basic_leaf >= 7) {
		avx2_bit = (leaf7_ebx & (1 << 5)) != 0;
		avx512f_bit = (leaf7_ebx & (1 << 16)) != 0;
		avx512bw_bit = (leaf7_ebx & (1 << 30)) != 0;
	}

	bool xmm_ymm_state = false;     // XGETBV(0) bits 1-2
	bool opmask_zmm_state = false;  // XGETBV(0) bits 5-7
	if (osxsave) {
		xmm_ymm_state = (xcr0 & 0x6ULL) == 0x6ULL;
		opmask_zmm_state = (xcr0 & 0xE0ULL) == 0xE0ULL;
	}

	if (avx512f_bit && avx512bw_bit && xmm_ymm_state && opmask_zmm_state) {
		return DotRowTier::kAvx512;
	}
	if (avx2_bit && xmm_ymm_state) {
		return DotRowTier::kAvx2;
	}
	return DotRowTier::kSse2;
}

// Called exactly once per process via the function-local static initializer in
// DotRow's non-forced, SIMD_X64 branch below (C++11 magic-static: thread-safe by
// standard guarantee, the write happens-before every subsequent read through the
// same static). Fetches the real CPUID/XGETBV fields and hands them to the pure
// resolver above -- see ResolveDotRowTierFromFields for the resolution rules.
inline DotRowTier DetectBestDotRowTier() {
#ifdef SUPERSLM_ENABLE_MATMUL_DISPATCH_INSTRUMENT
	superslm_test::g_dot_row_tier_probe_invocations.fetch_add(1, std::memory_order_relaxed);
#endif

	int regs0[4] = {0, 0, 0, 0};
	QueryCpuId(0, 0, regs0);
	const int max_basic_leaf = regs0[0];  // leaf 0, EAX: highest supported basic leaf

	int regs1[4] = {0, 0, 0, 0};
	QueryCpuId(1, 0, regs1);

	int regs7[4] = {0, 0, 0, 0};
	if (max_basic_leaf >= 7) {
		QueryCpuId(7, 0, regs7);
	}

	const bool osxsave = (regs1[2] & (1 << 27)) != 0;  // leaf 1, ECX bit 27
	unsigned long long xcr0 = 0;
	if (osxsave) {
		xcr0 = QueryXcr0();
	}

	return ResolveDotRowTierFromFields(max_basic_leaf, regs1[2], regs7[1], xcr0);
}

#endif  // SUPERSLM_MATMUL_HAVE_SIMD_X64

// --- DotRow's resolution (design §6.1, fold round 3: the three x86-only force arms
//     nested inside a platform check, D-SLM3519/§6.1.1) -------------------------
inline int64_t DotRow(const int8_t* activations, const int8_t* weights, size_t in_channels) {
#if defined(SUPERSLM_FORCE_SCALAR_MATMUL)
	return DotRowScalar(activations, weights, in_channels);
#elif defined(SUPERSLM_FORCE_SSE2_MATMUL) || defined(SUPERSLM_FORCE_AVX2_MATMUL) || \
    defined(SUPERSLM_FORCE_AVX512_MATMUL)
#if !SUPERSLM_MATMUL_HAVE_SIMD_X64
#error "SUPERSLM_FORCE_{SSE2,AVX2,AVX512}_MATMUL selects an x86-64 SIMD tier and only " \
       "compiles on an x86-64 target; use SUPERSLM_FORCE_SCALAR_MATMUL, or leave every " \
       "SUPERSLM_FORCE_*_MATMUL macro undefined, on this platform."
#elif defined(SUPERSLM_FORCE_SSE2_MATMUL)
	return DotRowSse2(activations, weights, in_channels);
#elif defined(SUPERSLM_FORCE_AVX2_MATMUL)
	return DotRowAvx2(activations, weights, in_channels);
#else  // SUPERSLM_FORCE_AVX512_MATMUL
	return DotRowAvx512(activations, weights, in_channels);
#endif
#elif SUPERSLM_MATMUL_HAVE_SIMD_X64
	static const DotRowTier tier = DetectBestDotRowTier();
	switch (tier) {
		case DotRowTier::kAvx512:
			return DotRowAvx512(activations, weights, in_channels);
		case DotRowTier::kAvx2:
			return DotRowAvx2(activations, weights, in_channels);
		case DotRowTier::kSse2:
		default:
			return DotRowSse2(activations, weights, in_channels);
	}
#else
	return DotRowScalar(activations, weights, in_channels);
#endif
}

}  // namespace

int64_t DotRowScalarRef(const int8_t* activations, const int8_t* weights, size_t in_channels) {
	// Test-reachable wrapper around the anonymous-namespace scalar reference (design
	// §5) -- see matmul.h's declaration. Calls the exact same construction DotRow falls
	// back to on non-x64 builds; does not participate in DotRow's dispatch, so the
	// dispatch resolution above is unchanged.
	return DotRowScalar(activations, weights, in_channels);
}

#if SUPERSLM_MATMUL_HAVE_SIMD_X64
int ResolveDotRowTier(int max_basic_leaf, int leaf1_ecx, int leaf7_ebx,
                       unsigned long long xcr0) {
	// Test-reachable wrapper around the anonymous-namespace pure resolver
	// (ResolveDotRowTierFromFields above) -- see matmul.h's declaration. T-2189 finding 1
	// (D-SLM3689): this is the seam the mutation proof drives directly with fabricated
	// leaf-0/leaf-1/leaf-7/XCR0 values, entirely independent of the host's real CPUID --
	// no hardware CPUID shim is needed because the decision logic itself takes no
	// hardware input. Mirrors DotRowScalarRef's own test-reachable-wrapper pattern.
	// Return value matches DotRowTier's declaration order: 0=SSE2, 1=AVX2, 2=AVX512.
	switch (ResolveDotRowTierFromFields(max_basic_leaf, leaf1_ecx, leaf7_ebx, xcr0)) {
		case DotRowTier::kSse2:
			return 0;
		case DotRowTier::kAvx2:
			return 1;
		case DotRowTier::kAvx512:
			return 2;
	}
	return 0;  // unreachable; every enumerator handled above
}
#endif  // SUPERSLM_MATMUL_HAVE_SIMD_X64

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
	// T-2147/D-SLM3488 (Vitruvius fold 2026-08-17, correcting D-SLM3481): this primitive has TWO
	// independently-true properties, and they are not the same claim.
	//
	// (1) Row/cell independence (design §3/§9's "no cross-row reduction", extended one axis by
	//     D-SLM3488 to "no cross-column reduction" either): cell (t, j)'s own value is
	//     DotRow(activations[t], weights[j], in_channels) -- a scalar reduction over
	//     in_channels that reads only row t's activations and row j's weights, sharing no
	//     accumulator or state with any other cell in EITHER dimension. This means any traversal
	//     order over the (t, j) grid -- t outer or j outer, any grouping, any split -- produces
	//     bit-identical per-cell output, because every cell reduces to the identical DotRow
	//     expression regardless of visit order. This is a CORRECTNESS property (which values may
	//     be computed in which grouping), and design §12 dim 7's stacking-equivalence cell
	//     asserts against it.
	//
	// (2) Memory-traffic behavior: which loop is OUTER determines how many times `weights` is
	//     re-read from the start. j (out_channels) outer, t (num_tokens) inner means each
	//     weight row (`weights + j*in_channels`, in_channels bytes) is read ONCE per call and
	//     stays cache-resident across every token that consumes it -- weights streamed once per
	//     CHUNK. The reverse nesting (t outer, j inner -- this function's own shape before
	//     D-SLM3488) re-reads the FULL weight matrix from the start on every token, i.e. once
	//     per TOKEN across a chunk, delivering zero weight-bandwidth amortization regardless of
	//     num_tokens. D-SLM3481's original ruling asserted "streams that layer's weight matrix
	//     once for the whole chunk" from property (1) alone, without reading this function's own
	//     loop order at source -- false about (2), measured by T-2147's own 1.07x speedup
	//       (Claude/Brunel/t2147-batched-prefill-2026-08-17.md), corrected by D-SLM3488.
	//
	// The loop nest below is j-outer, t-middle, k-innermost (DotRow's own reduction) --
	// property (2)'s actual delivery mechanism. Property (1) makes this permutation free: no
	// cell's own accumulation order changes (DotRow's k-ascending reduction is untouched), only
	// which cells are visited in which order, so this is a pure traversal restructuring with
	// zero new arithmetic (D-SLM3488's own "no new arithmetic" ruling).
	for (size_t j = 0; j < out_channels; ++j) {
		const int8_t* const weight_row = weights + j * in_channels;  // read ONCE per chunk here,
		                                                              // held cache-resident across
		                                                              // every token below
		for (size_t t = 0; t < num_tokens; ++t) {
			out_acc[t * out_channels + j] = DotRow(activations + t * in_channels, weight_row, in_channels);
		}
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

// SuperSLM raw int8x8 matmul accumulate.
//
// The raw int8-weight-matrix x int8-activation-row dot product that produces the wide
// accumulator row the per-token dynamic-scale requant chain (intmath.h) consumes. This header
// declares the kernel only; it pins no new requant arithmetic -- every primitive downstream of
// the accumulator narrowing step is already normative and unmodified elsewhere.
//
// Standard library only -- no float on the reproducible path.
//
// The declarations below are the approved API surface. Every body, including the scalar
// reference, is a real construction backed by matmul.cpp.
#ifndef SUPERSLM_MATMUL_H
#define SUPERSLM_MATMUL_H

#include <cstddef>
#include <cstdint>

// T-2149 design §6.1: compile-time capability -- the same fact about the target that
// gates matmul.cpp's own SIMD dispatch (DotRow/DetectBestDotRowTier only exist under
// this condition). Declared here, not privately inside matmul.cpp, because the T-2158
// test-only seam (tests/support/matmul_dispatch_instrument.h) and its two coverage
// cells (design §10 dimensions 1/3) need the identical condition to scope themselves
// out on a non-x64 target where the mechanism under test does not exist (fold round 4,
// D-SLM3568) -- a second, independently-maintained copy of this condition is exactly
// the drift the design's own standards forbid (StandardsDocument.md §4).
#if defined(_M_X64) || defined(__x86_64__)
#define SUPERSLM_MATMUL_HAVE_SIMD_X64 1
#else
#define SUPERSLM_MATMUL_HAVE_SIMD_X64 0
#endif

namespace superslm {

// C17/C19-C22 -- raw int8x8 matmul accumulate: one activation row (int8 codes, in
// [-127,127] per C22's output range -- the already-shipped RequantTokenCode never emits
// -128) dotted against every row of an [out_channels, in_channels] int8 weight matrix
// (WGT1 layout, row-major, one row per output channel -- the nn.Linear (out_features,
// in_features) convention). The reduction is ALWAYS carried in int64 intermediate
// arithmetic regardless of the tensor's declared accumulator width (design §4's C20
// "widen before" discipline, applied here to a genuine-overflow op) -- this function
// never touches int32.
//
// Caller ensures (contract, not runtime-checked -- the same caller-ensures convention as
// MaxAbsReduce/ShiftByMax): `activations` has `in_channels` elements; `weights`
// has `out_channels * in_channels` elements; `out_acc` has `out_channels` elements (one
// int64 sum per output channel). No accumulation order is pinned -- integer addition of
// exact int64 products is exactly associative and commutative (design §4), so any
// traversal order (sequential, tree-reduced, SIMD-lane-regrouped) must produce the
// bit-identical int64 sum.
void GemmInt8AccumulateRow(const int8_t* activations, const int8_t* weights,
                            size_t in_channels, size_t out_channels, int64_t* out_acc);

// Multi-row form (the prefill shape): `num_tokens` independent activation rows against
// the SAME weight matrix. Each output row is GemmInt8AccumulateRow applied
// independently -- no cross-row reduction, no shared state between rows; every row
// carries its own per-token dynamic scale and is requantized independently upstream.
//
// Caller ensures: `activations` has `num_tokens * in_channels` elements; `weights` has
// `out_channels * in_channels` elements; `out_acc` has `num_tokens * out_channels`
// elements, row-major -- output row t occupies out_acc[t*out_channels ..
// (t+1)*out_channels) and equals GemmInt8AccumulateRow on activation row t alone
// (design §3).
//
// On the AVX2 and AVX-512BW tiers a call with num_tokens >= 8 runs a register-tiled kernel over
// weights packed per call (the tiled-matmul plan's slice 1): the same vpmaddwd instruction and the
// same per-lane flush bound as the one-cell-at-a-time loop, with only the lane-to-cell assignment
// changed, so every output is bit-identical to the scalar reference. That path allocates two
// scratch buffers on the calling thread (the int16-widened activations and one packed weight
// panel group) and can therefore throw std::bad_alloc; the call sites already run inside the
// ABI's allocation-failure boundary.
void GemmInt8Accumulate(const int8_t* activations, const int8_t* weights,
                         size_t num_tokens, size_t in_channels, size_t out_channels,
                         int64_t* out_acc);

// Internal: not part of the public contract, and no consumer outside the engine and its tests may
// call it. The tiled-matmul plan's slice 1 (§4.3, §4.7) lands it so slice 2 can split the output
// columns across tasks without touching matmul.cpp.
namespace detail {

// Column-range entry. Writes exactly columns [j_begin, j_end) of the row-major
// [num_tokens][out_channels] output, and never writes any other element of `out_acc`.
// GemmInt8Accumulate is GemmInt8AccumulateCols(..., 0, out_channels, out_acc).
// Caller ensures: j_begin <= j_end <= out_channels, plus GemmInt8Accumulate's own contract.
void GemmInt8AccumulateCols(const int8_t* activations, const int8_t* weights, size_t num_tokens,
                            size_t in_channels, size_t out_channels, size_t j_begin, size_t j_end,
                            int64_t* out_acc);

// The pre-widened sibling: `activations16` holds num_tokens rows of `widened_stride` int16 values,
// row t being activation row t widened exactly (int8 -> int16) and ZERO past in_channels, with
// widened_stride = in_channels rounded up to even. That zero pad is part of the specification:
// it is what makes the packed weights' own K pad harmless. Same output contract as
// GemmInt8AccumulateCols. Slice 2 widens once per call and hands every task this buffer.
void GemmInt8AccumulateColsWidened(const int16_t* activations16, size_t widened_stride,
                                   const int8_t* weights, size_t num_tokens, size_t in_channels,
                                   size_t out_channels, size_t j_begin, size_t j_end,
                                   int64_t* out_acc);

// The tier GemmInt8AccumulateCols dispatches on. On x64 it is the cached CPUID tier (or the
// forced one); on any other target it is kScalar.
enum class GemmTier : int { kScalar = 0, kSse2 = 1, kAvx2 = 2, kAvx512 = 3 };
enum class GemmPath : int { kDotRowLoop = 0, kTiled = 1 };

// The call count at which the tiled path starts (the plan's kTiledMinTokens, 8, unless a test
// build overrides it with SUPERSLM_TEST_TILED_MIN_TOKENS).
size_t TiledMinTokens();

// The pure path selector (cell 4.7(a)), compiled into every build and testable with any
// arguments on any runner: the tiled path is chosen only on the AVX2 and AVX-512 tiers, only at
// num_tokens >= TiledMinTokens(), and, on the AVX-512 tier of an MSVC or clang-cl build
// (`is_msvc_build`), only when `msvc_avx512_switch` is nonzero (plan §4.5: off until an MSVC
// AVX-512 build has executed the tiled kernel).
GemmPath SelectGemmPath(GemmTier tier, size_t num_tokens, int msvc_avx512_switch,
                        bool is_msvc_build);

// The call-site wiring: SelectGemmPath with this build's own switch value and compiler identity.
// GemmInt8AccumulateCols and its widened sibling decide their path through this function and
// nothing else, so a test on any runner can observe what a given tier would dispatch to.
GemmPath DispatchGemmPath(GemmTier tier, size_t num_tokens);

// The tier this process's GEMM dispatches on (see GemmTier).
GemmTier ActiveGemmTier();

// Attention and per-row sites plan (rev 3.1, §3.2, cell 11.2): which body the attention kernels of
// slices S2-S6 run. kShipped is the v1.9.0 code; kAvx2 and kAvx512 are the new SIMD bodies.
enum class SitesKernel : int { kShipped = 0, kAvx2 = 1, kAvx512 = 2 };

// The pure selector, compiled into every build and testable with any arguments on any runner: the
// new kernels run only on the AVX2 and AVX-512 tiers, and on the AVX-512 tier of an MSVC or clang-cl
// build (`is_msvc_build`) only when `msvc_avx512_switch` is nonzero (§3.2: SUPERSLM_SITES_AVX512_MSVC,
// default 0, independent of the tiled GEMM's switch).
SitesKernel SelectSitesKernel(GemmTier tier, int msvc_avx512_switch, bool is_msvc_build);

// The call-site wiring: SelectSitesKernel with this build's own switch value and compiler identity.
// Every S2-S6 dispatch decides through this function and nothing else. SitesKernel and both selectors
// are internal C++ declarations like the GemmPath ones above: not exported, not part of the C API.
SitesKernel DispatchSitesKernel(GemmTier tier);

}  // namespace detail

// C17 -- narrow one accumulator row to int32 AFTER a conversion-time proof (design §4,
// §8) that this tensor's declared MatmulAccumWidth is Int32 (i.e. in_channels is within
// §8's derived bound, 131,071). This is the ONLY point an int32 array is ever
// materialized -- GemmInt8Accumulate itself never produces int32.
//
// Caller-ensures convention (matching MaxAbsReduce / ShiftByMax): UB if the declared
// width was wrong for this tensor (i.e. the wide row's values do not fit int32). `n`
// elements each direction.
//
// **The i-exp primitives no longer belong on that list (S-HARDEN-0).** They moved from
// caller-ensures to checked, because a guard that is undefined on the input it screens is
// not a guard (F9, F21). Whether the rest of Layer 1 should follow is a live design
// question and is NOT answered here -- this comment records which convention this
// function actually uses, not which one it ought to.
void NarrowAccumulatorToI32(const int64_t* wide_row, size_t n, int32_t* out_i32);

// C17 -- the per-tensor accumulator-width choice, recorded in the artifact (design §4,
// owed registration -- not yet a format field; see design §13). Every currently-scoped
// candidate is Int32 per design §8's derivation; no Int64 candidate exists yet. The
// int64-domain MaxAbsReduceWide/RequantTokenCodeWide overloads an Int64 candidate would
// require ARE built (intmath.h, S3a §7.2/F-S3-7) -- this enum has no Int64-consuming
// caller yet, which is the part still owed.
enum class MatmulAccumWidth : int32_t { Int32 = 0, Int64 = 1 };

// design §5 -- the scalar reference construction, exposed for verification only. The
// shipping dispatch (matmul.cpp's internal DotRow) never calls this on an x64 build --
// x64 builds runtime-dispatch among SSE2 (the unconditional architectural floor), AVX2,
// and AVX-512 by cached CPUID+XGETBV probe (T-2149 design §6.2), or resolve one tier at
// compile time when a SUPERSLM_FORCE_*_MATMUL macro pins it (design §6.1). This
// declaration makes the normative scalar path reachable from a test so scalar == SIMD ==
// oracle can be asserted directly (design §11 item 4) instead of only transitively
// through GemmInt8Accumulate.
// Same contract as the row dot product inside GemmInt8AccumulateRow: `activations` and
// `weights` each have `in_channels` elements; every intermediate is int64, both int8
// factors widened to int64 before the multiply, no saturation, no rounding.
int64_t DotRowScalarRef(const int8_t* activations, const int8_t* weights, size_t in_channels);

#if SUPERSLM_MATMUL_HAVE_SIMD_X64
// design §6.2 -- the pure CPUID-field tier resolver, exposed for verification only, mirroring
// DotRowScalarRef's own pattern above. The shipping dispatch (matmul.cpp's internal
// DetectBestDotRowTier()) calls the anonymous-namespace resolver this wraps; this declaration
// makes that decision logic reachable from a test with fabricated register values, no real
// hardware CPUID shim needed, so the mutation proof for the max-basic-leaf guard (leaf 7 is
// architecturally undefined below basic leaf 7 and must not be consulted) can drive it
// directly. `max_basic_leaf` is CPUID leaf 0's EAX; `leaf1_ecx` is CPUID leaf 1's ECX (OSXSAVE,
// bit 27); `leaf7_ebx` is CPUID leaf 7 sub-leaf 0's EBX (AVX2 bit 5, AVX512F bit 16, AVX512BW
// bit 30) -- consulted only when `max_basic_leaf >= 7`; `xcr0` is the XGETBV(0) value (XMM/YMM
// state bits 1-2, opmask/ZMM state bits 5-7) -- consulted only when OSXSAVE is set. Returns
// 0=SSE2, 1=AVX2, 2=AVX512, matching the internal DotRowTier enum's declaration order.
int ResolveDotRowTier(int max_basic_leaf, int leaf1_ecx, int leaf7_ebx,
                       unsigned long long xcr0);
// Shipping CPU probe, sharing DotRow's cached CPUID/XGETBV result.  Exposed
// for other integer SIMD kernels that must select the same safe ISA tier.
int DetectBestDotRowTierForCpu();
#endif  // SUPERSLM_MATMUL_HAVE_SIMD_X64

// F-S3-6/C32 (SuperSLM_S3a_WalkingSkeleton_Plan.md §4.6, §11 S3.3 §6.4) — the
// probability x value context accumulate: `out_ctx[d] = Sum_k probs[k] *
// values[k*head_dim + d]`. `probs` are C32's Q15 row-normalized probabilities
// (each row sums to at most 2^15 by construction, kProbFracBits, intmath.h);
// `values` are int8 codes. Exact int64 accumulation, no saturation, no
// rounding: the derived bound is `|Sum_k p_k*v_k| <= 2^15 * 127 < 2^22`,
// INDEPENDENT of context length (matching C27's own stated bound), so no
// per-tensor width choice is owed and int64 accumulation is far more than
// sufficient. Same no-order-pin property as GemmInt8AccumulateRow above:
// exact int64 products are exactly associative and commutative, so any
// traversal order must produce the bit-identical `out_ctx`.
//
// Attention and per-row sites plan, slice S2 (§4.2, §5.2): on the AVX2 and AVX-512BW tiers a call
// whose head_dim is a multiple of 16 and whose row passes the int16 condition (every p in
// [0, 32767] and Sum p <= 2^15, checked by the function itself in one pass) accumulates p_k*v_k[d] +
// p_{k+1}*v_{k+1}[d] with vpmaddwd into one int32 lane per output dimension, then widens to int64.
// Every lane's running sum is bounded by 128 * Sum p <= 2^22, so each output equals the int64 sum
// exactly; any other row takes the shipped loop. No allocation, no new status, same contract.
//
// Caller ensures (contract, not runtime-checked -- the same convention as
// GemmInt8AccumulateRow above): `probs` has `width` elements; `values` has
// `width * head_dim` elements, row-major (`values[k*head_dim + d]` is key
// k's value-vector element d); `out_ctx` has `head_dim` elements.
void GemmProbQ15Accumulate(const int64_t* probs, const int8_t* values, size_t width,
                            size_t head_dim, int64_t* out_ctx);

// Paged-KV plan §3.2 item 3: GemmProbQ15Accumulate without the zeroing -- `out_ctx[d] += Sum_k
// probs[k] * values[k*head_dim + d]`, through the same tiered accumulate-into core (the same SIMD
// guard and bodies, the same scalar loop), so every tier's result is bit-identical to the one-call
// form's. Per-page attention zeroes `out_ctx` once and then calls this once per page run (the
// values of one page run are contiguous; consecutive runs are not). Exactness of the chain: the
// same int64 terms are summed, integer addition is associative, and the bound above is
// width-independent, so no intermediate overflows (cell 6.2). Same caller-ensures contract as
// GemmProbQ15Accumulate.
void GemmProbQ15AccumulateInto(const int64_t* probs, const int8_t* values, size_t width,
                               size_t head_dim, int64_t* out_ctx);

}  // namespace superslm

#endif  // SUPERSLM_MATMUL_H

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
void GemmInt8Accumulate(const int8_t* activations, const int8_t* weights,
                         size_t num_tokens, size_t in_channels, size_t out_channels,
                         int64_t* out_acc);

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
// it unconditionally selects the SSE2 specialization there, since every x64 chip has
// SSE2 and no runtime CPUID dispatch is needed. This declaration makes the normative
// scalar path reachable from a test so scalar == SIMD == oracle can be asserted
// directly (design §11 item 4) instead of only transitively through GemmInt8Accumulate.
// Same contract as the row dot product inside GemmInt8AccumulateRow: `activations` and
// `weights` each have `in_channels` elements; every intermediate is int64, both int8
// factors widened to int64 before the multiply, no saturation, no rounding.
int64_t DotRowScalarRef(const int8_t* activations, const int8_t* weights, size_t in_channels);

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
// Caller ensures (contract, not runtime-checked -- the same convention as
// GemmInt8AccumulateRow above): `probs` has `width` elements; `values` has
// `width * head_dim` elements, row-major (`values[k*head_dim + d]` is key
// k's value-vector element d); `out_ctx` has `head_dim` elements.
void GemmProbQ15Accumulate(const int64_t* probs, const int8_t* values, size_t width,
                            size_t head_dim, int64_t* out_ctx);

}  // namespace superslm

#endif  // SUPERSLM_MATMUL_H

// SuperSLM integer-arithmetic kernels — Layer-1 requant primitives and the
// per-token dynamic-scale chain (SuperSLM_Plan.md §6.2, §6.8 C1/C2/C3 + C19–C22).
//
// This is the compiled C++ port of the offline reference `Tools/superslm_spike/
// intmath.py`. Every function here is a bit-for-bit reimplementation of the pinned
// reference: conformance is OUTPUT EXACTNESS against that reference over the whole
// domain, never a blessing of any particular decomposition. Standard library only
// (`<bit>` for the pinned scalar count-leading-zeros); no float on any reproducible
// path — floats appear only as offline constants folded to integers by the converter,
// never here.
//
// The two families:
//   * C1/C2/C3 — gemmlowp's fixed-point requant primitives
//     (`SaturatingRoundingDoublingHighMul`, `RoundingDivideByPOT`) and their
//     composition. Tie directions are pinned and opposite by design: the doubling
//     high-mul rounds toward +infinity (C2), the divide-by-POT rounds away from zero
//     (C3). §15 / D-SLM36.
//   * C19/C20/C21/C22 — the rung-1 per-token dynamic quantizer: max-abs reduction
//     (C20), scale normalization (C21), fixed-point reciprocal (C19, corrected Newton),
//     and the per-token requant composition — the "127 scale wrapper" (C22, D-SLM52).
//
// Cost determinism (§14): every op count here is data-INDEPENDENT. The reciprocal runs
// exactly 3 Newton iterations plus exactly 2 branch-free correction steps; normalization
// runs one bit-scan plus one shift plus one select. No early exit, no data-dependent
// trip count. This property is a source obligation Poirot verifies by reading the code,
// on top of the exhaustive value certification.
//
// Wide intermediates: C19's reciprocal Newton products and C22's `x_i · 127 · R`
// (magnitude up to ~2^70) exceed int64. The port carries them in a portable 128-bit
// intermediate (no reliance on a compiler `__int128`, which MSVC lacks). Conformance is
// by output exactness against the pinned formula, not by the decomposition chosen.
#ifndef SUPERSLM_INTMATH_H
#define SUPERSLM_INTMATH_H

#include <cstddef>
#include <cstdint>

namespace superslm {

inline constexpr int32_t kInt32Min = -2147483647 - 1;  // -2^31
inline constexpr int32_t kInt32Max = 2147483647;       //  2^31 - 1

// --- §6.2 requant primitives (C1/C2/C3) --------------------------------------

// C2 — gemmlowp `SaturatingRoundingDoublingHighMul`: (2·a·b) / 2^32, rounded, ties
// toward +infinity (the whole primitive is `(a*b + (1<<30)) >> 31`, both signs).
// The product `a·b` is formed widened to int64; the arithmetic right shift is the
// C++20-guaranteed floor shift. `(INT32_MIN, INT32_MIN)` is the only int32 pair whose
// exact result exceeds INT32_MAX, and it saturates to INT32_MAX.
int32_t SaturatingRoundingDoublingHighMul(int32_t a, int32_t b);

// C1/C3 — gemmlowp `RoundingDivideByPOT`: x / 2^exponent, ties AWAY FROM ZERO.
// Defined for exponent in [0, 31]; exponent 0 is the identity. The `+1` carried on the
// negative branch of the threshold is what turns the floor shift's half-up behaviour
// into half-away-from-zero.
int32_t RoundingDivideByPOT(int32_t x, int exponent);

// C1 — the two primitives composed in order, both roundings applied:
// `RoundingDivideByPOT(SaturatingRoundingDoublingHighMul(x, m), shift)`.
int32_t MultiplyByQuantizedMultiplier(int32_t x, int32_t quantized_multiplier, int shift);

// --- §6.2 rung-1 per-token dynamic-scale chain (C19/C20/C21/C22) --------------

// C21 helper — count leading zeros over 64 bits. Defined for n in [1, 2^64 - 1]; the
// bit-scan is pinned to scalar-reference behaviour (§18 GPU-semantics row). n == 0 is
// out of contract (undefined leading-zero count) and never reached: its only caller,
// NormalizeScale, receives D' >= 1 by C20's guard.
int Clz64(uint64_t n);

// C20 — max-abs reduction: D = max_i |x_i| over the token's int32 activation vector,
// each element WIDENED to int64 before the abs (so x_i = INT32_MIN yields |x_i| = 2^31
// exactly rather than int32 UB), then the all-zero-token guard D' = max(D, 1). Returns
// D' in [1, 2^31]. Order-free (max applies no mid-stream rounding); every element is
// visited unconditionally. `n == 0` yields D' = 1 (the empty reduction), matching the
// reference; the runtime never sees a zero-length vector — hidden dims are architectural
// constants >= 1, rejected at artifact validation.
int64_t MaxAbsReduce(const int32_t* x, size_t n);

// C21 — scale normalization. Result of NormalizeScale: Dn in [2^30, 2^31) and the shift
// exponent s that rides into the C22 composition (1/D' = R · 2^s / 2^62).
struct NormalizedScale {
	int64_t dn;  // normalized denominator in [2^30, 2^31)
	int s;       // shift exponent, s = 30 - (63 - clz64(D'))
};

// C21 — map D' in [1, 2^31] into Dn in [2^30, 2^31) by a pure shift:
// p = 63 - clz64(D'), s = 30 - p, Dn = s >= 0 ? D' << s : D' >> 1. The only negative-s
// case is D' = 2^31 (even), whose single right-shift loses no bits (Dn = 2^30). Exact,
// no rounding. Exactly one bit-scan + one shift + one branch-free select.
NormalizedScale NormalizeScale(int64_t d_prime);

// C19 — fixed-point reciprocal: R = round_half_up(2^62 / Dn) for Dn in [2^30, 2^31),
// computed DIVISION-FREE via corrected Newton — exactly 3 Newton iterations then exactly
// 2 unconditional branch-free correction steps. Bare fixed-count Newton is deterministically
// wrong on 54–62% of the domain (error biased high → activation clipping); the correction
// closes it to correctly-rounded over the entire normalized domain. Returns R in
// (2^31, 2^32] (R = 2^32 exactly at Dn = 2^30). The seed is not pinned — any construction
// reaching output-exactness is conformant.
int64_t DynamicScaleReciprocal(int64_t dn);

// C22 — per-token requant composition, the "127 scale wrapper" (D-SLM52), R-composed:
// q_i = clamp(round_half_away_from_zero((x_i · 127 · R) / 2^(62 − s)), −127, 127),
// with R from C19 and s from C21. Ties away from zero (C3's direction — the composite IS
// a rounding divide by a power of two). `|x_i · 127 · R|` reaches ~2^70 (a token containing
// INT32_MIN forces R = 2^32 with x_i = −2^31 present), carried in a 128-bit intermediate.
// Returns the int8 code in [−127, 127].
int8_t RequantTokenCode(int32_t x_i, int64_t r, int s);

}  // namespace superslm

#endif  // SUPERSLM_INTMATH_H

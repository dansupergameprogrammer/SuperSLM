// SuperSLM SwiGLU SiLU via a fixed-point sigmoid lookup table.
//
// The runtime half of the construction: map an int8 gate code plus its per-token carried scale
// to a position in a pinned Q15 sigmoid table, linearly interpolate between the two nearest
// nodes, and round once. Entirely integer, entirely division-free. The table itself is
// generated offline in double precision by the converter and carried in the artifact's SIL1
// section (model.h ParseSigmoidLut / SigmoidLutValue); this header is the runtime lookup only.
//
// Standard library only — no float on the reproducible path. Both roundings reuse the shipped
// RoundingDivideByPOT primitive (int64 for the index placement, int32 for the interpolation),
// so the tie rule is not re-implemented here.
#ifndef SUPERSLM_SILU_LUT_H
#define SUPERSLM_SILU_LUT_H

#include <cstdint>

namespace superslm {

// Pinned build-time table geometry (SuperSLM_S2.4_SiLU_LUT_Design §4, §5). These fix the
// SIL1 the converter emits and the index math the runtime applies; they are compile-time,
// never carried per-artifact.
inline constexpr int kSiluLutN = 1024;                     // node count N (must match kSigmoidLutNodes)
inline constexpr int kSiluLutX = 16;                       // domain half-width X: x in [-16, 16]
inline constexpr int kSiluLutLog2K = 5;                    // k = log2(N / 2X) = log2(32); K = 2^k
inline constexpr int kSiluLutQIdx = 12;                    // sub-node index fractional bits (§5, §7)

// Named so the KVC1 (m,e) no-UB floor's upper bound on `e`
// (this header's own `kCompositionScaleMaxE`, below) can be pinned by `static_assert` to the exact
// shift at which SiluSigmoidQ15's left branch (`term << shift`, src/silu_lut.cpp) overflows
// int64 — 26, given |term| < 2^38 from |code| <= 127 (< 2^7) and |m| bounded by the floor's own
// `kCompositionScaleMaxAbsM`. A prose-only bound (a comment citing "26") is checked by
// nobody; this constant is what the pin in model.cpp compares against.
// `26` is hand-derived from |term| < 2^38 and int64's headroom.
// If a future change widens the KVC1 m-domain past INT32_MAX, model.cpp's
// `kCompositionScaleMaxAbsM == kInt32Max` static_assert fires first — but this constant must then
// be RE-DERIVED, because the shift-side pin would still pass against a now-stale overflow point.
inline constexpr int kSiluLutTermLeftShiftOverflowExponent = 26;

// The load-time no-UB floor for CompositionConstants (KVC1) (m, e),
// exact from source (silu_lut.cpp:20-39) and sufficient
// against all three of SiluSigmoidQ15's UB sites. Public (moved out of
// src/model.cpp's anonymous namespace) so the runtime no-UB domain
// predicate below (checked_chain_funnel.h's CheckSiluCompositionScaleDomain) can
// pin its own ceiling against this one by static_assert rather than a duplicated
// literal that could drift out of step with it silently. The tighter swept
// envelope (`-shift in [15,19]`, i.e. `e in [-36,-32]`) is deliberately NOT
// enforced here -- this constant ships the floor only.
inline constexpr int64_t kCompositionScaleMaxAbsM = (INT64_C(1) << 31) - 1;  // |m| <= 2^31-1
inline constexpr int64_t kCompositionScaleMinE = -80;   // silu_lut.cpp:35 right branch, -shift <= 63
inline constexpr int64_t kCompositionScaleMaxE = 7;     // silu_lut.cpp:35 left branch, term << shift exact

// sigmoid(code * realscale) in Q15 via the LUT, where realscale = m * 2^e and `code` is the
// int8 SwiGLU gate value (the codebase's activation format, in [-127, 127]). `table` is
// kSigmoidLutEntries (= N+1) native-endian int32 Q15 nodes (the caller materializes it from the
// SIL1 section once at load). Division-free; exactly TWO C3 roundings on the path — one to place
// the sub-node index (§5, when e+k+Q_idx < 0) and one to interpolate (§6). Returns the
// interpolated Q15 sigmoid value in [0, 2^15]. Saturates to table[0] / table[N] at the domain
// edges (§5/§6 resolve saturation to the true extreme node).
int32_t SiluSigmoidQ15(const int32_t* table, int8_t code, int64_t m, int e);

}  // namespace superslm

#endif  // SUPERSLM_SILU_LUT_H

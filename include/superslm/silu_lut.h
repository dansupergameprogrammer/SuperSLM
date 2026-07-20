// SuperSLM SwiGLU SiLU via a fixed-point sigmoid LUT (C10, S2.4).
//
// The runtime half of the C10 construction (D-SLM68): map an int8 gate code plus its
// per-token (m, e) carried scale to a position in a pinned Q15 sigmoid table, linearly
// interpolate between the two nearest nodes, and round once (C3). Entirely integer, entirely
// DIVISION-FREE — off §18's int64-division GPU-semantics row (SuperSLM_S2.4_SiLU_LUT_Design
// §5, §6). The table itself is generated OFFLINE in double precision by the converter and
// carried in the artifact's SIL1 section (model.h ParseSigmoidLut / SigmoidLutValue); this
// header is the runtime lookup only.
//
// Standard library only — no float on the reproducible path (Layer 1, D-SLM13). The two
// roundings both reuse a shipped RoundingDivideByPOT primitive (int64 for the index placement,
// int32 for the interpolation), so the tie rule (C3) is not re-implemented here.
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

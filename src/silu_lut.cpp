// SuperSLM SwiGLU SiLU via a fixed-point sigmoid LUT (C10, S2.4) — STUB (red-phase).
//
// See include/superslm/silu_lut.h for the runtime contract and SuperSLM_S2.4_SiLU_LUT_Design
// §5/§6 for the index derivation and interpolation this greens into. The stub returns a
// deliberately-wrong sentinel so Curie's S2.4 op-level red suite compiles+links+fails; Brunel
// greens it with the real division-free lookup.
#include "superslm/silu_lut.h"

namespace superslm {

int32_t SiluSigmoidQ15(const int32_t* /*table*/, int8_t /*code*/, int64_t /*m*/, int /*e*/) {
	return INT32_MIN;  // stub — no valid Q15 sigmoid is negative, so every real cell fails
}

}  // namespace superslm

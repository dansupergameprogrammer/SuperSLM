// SuperSLM integer-arithmetic kernels — STUB (S2.1 red-phase).
//
// These are deliberately-wrong sentinel bodies so Curie's red suite compiles, links,
// and FAILS against the port's real semantics. Brunel replaces each body with the
// bit-exact port of Tools/superslm_spike/intmath.py in the S2.1 green phase.
#include "superslm/intmath.h"

namespace superslm {

int32_t SaturatingRoundingDoublingHighMul(int32_t, int32_t) {
	return 0;  // stub
}

int32_t RoundingDivideByPOT(int32_t, int) {
	return 0;  // stub
}

int32_t MultiplyByQuantizedMultiplier(int32_t, int32_t, int) {
	return 0;  // stub
}

int Clz64(uint64_t) {
	return -1;  // stub
}

int64_t MaxAbsReduce(const int32_t*, size_t) {
	return 0;  // stub (violates the D' >= 1 postcondition on purpose)
}

NormalizedScale NormalizeScale(int64_t) {
	return NormalizedScale{0, 0};  // stub
}

int64_t DynamicScaleReciprocal(int64_t) {
	return 0;  // stub
}

int8_t RequantTokenCode(int32_t, int64_t, int) {
	return 0;  // stub
}

}  // namespace superslm

// SuperSLM raw int8x8 matmul accumulate (C17/C19-C22 site, S2.5).
//
// See include/superslm/matmul.h for the runtime contract and
// SuperSLM_matmul_subslot_design-2026-07-20.md §3/§5 for the construction.
//
// S2.5 CONTRACT STEP (red-first TDD): every body below is a STUB. Each asserts false
// when called -- it deliberately does not implement the scalar reference (design §5).
// The real implementation lands at the next build step (Brunel green), against Curie's
// red suite authored from this contract. Do not add arithmetic here until that step.
#include "superslm/matmul.h"

#include <cassert>

namespace superslm {

void GemmInt8AccumulateRow(const int8_t* activations, const int8_t* weights,
                            size_t in_channels, size_t out_channels, int64_t* out_acc) {
	(void)activations;
	(void)weights;
	(void)in_channels;
	(void)out_channels;
	(void)out_acc;
	assert(false && "GemmInt8AccumulateRow: stub -- S2.5 contract step, not yet implemented");
}

void GemmInt8Accumulate(const int8_t* activations, const int8_t* weights,
                         size_t num_tokens, size_t in_channels, size_t out_channels,
                         int64_t* out_acc) {
	(void)activations;
	(void)weights;
	(void)num_tokens;
	(void)in_channels;
	(void)out_channels;
	(void)out_acc;
	assert(false && "GemmInt8Accumulate: stub -- S2.5 contract step, not yet implemented");
}

void NarrowAccumulatorToI32(const int64_t* wide_row, size_t n, int32_t* out_i32) {
	(void)wide_row;
	(void)n;
	(void)out_i32;
	assert(false && "NarrowAccumulatorToI32: stub -- S2.5 contract step, not yet implemented");
}

}  // namespace superslm

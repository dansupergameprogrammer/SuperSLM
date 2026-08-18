// GENERATED FILE. Do not hand-edit.
//
// Produced by tools/gen_matmul_golden.py. The pinned hash below is computed by that
// script in arbitrary-precision Python integer arithmetic reproducing the design's S5
// scalar reference (SuperSLM_matmul_subslot_design-2026-07-20.md), NOT by recording
// what a C++ build printed -- the golden is generated from the reference, so a matching
// C++ hash is agreement with an independent computation.
//
// Design record: SuperSLM_matmul_subslot_design-2026-07-20.md S10 step 5 / S11 item 6.
// Re-running the generator must reproduce this file byte-for-byte.
#ifndef SUPERSLM_TESTS_MATMUL_GOLDEN_PIN_H
#define SUPERSLM_TESTS_MATMUL_GOLDEN_PIN_H

#include <cstddef>
#include <cstdint>

namespace superslm_test {

// kind: 0 = pinned-LCG fill, 1 = alternating int8-extremes pattern, 2 = single-signed
// attainable extremes (act +127, wgt -128), 3 = single-signed the other direction (act
// -127, wgt -128). All four are mirrored in test_main.cpp. Kinds 2 and 3 are the only
// fills whose sums leave int32: an LCG's mixed signs cancel, so a golden without them is
// reproduced bit-for-bit by a kernel that narrows mid-reduction.
struct MatmulGoldenCase {
	const char* label;
	uint64_t seed;
	size_t in_channels;
	size_t out_channels;
	size_t num_tokens;
	int kind;
	bool int32_safe;  // S8's 131,071 bound declares MatmulAccumWidth::Int32 for this case
};

inline constexpr MatmulGoldenCase kMatmulGoldenCases[] = {
	{"real_hidden_1536", 1ull, 1536, 1536, 2, 0, true},
	{"real_hidden_960", 2ull, 960, 960, 1, 0, true},
	{"real_intermediate_8960", 3ull, 8960, 64, 1, 0, true},
	{"real_intermediate_2560", 4ull, 2560, 64, 1, 0, true},
	{"synthetic_tail5_1533", 5ull, 1533, 8, 2, 0, true},
	{"synthetic_tail7_1535", 9ull, 1535, 8, 1, 0, true},
	{"floor_in1_out1", 6ull, 1, 1, 1, 0, true},
	{"int8_extremes_1024", 0ull, 1024, 4, 1, 1, true},
	{"deep_flush_lcg_132105", 8ull, 132105, 2, 1, 0, false},
	{"deep_beyond_i32_neg", 0ull, 132105, 1, 1, 2, false},
	{"deep_beyond_i32_pos", 0ull, 132105, 1, 1, 3, false},
	{"deep_beyond_all_windows_neg", 0ull, 4194305, 1, 1, 2, false},
	{"deep_beyond_all_windows_pos", 0ull, 4194305, 1, 1, 3, false},
};

// The generator's own first 16 bytes from seed 1. Pinned separately so a drift in the
// LCG mirror fails as its own named check rather than as a kernel determinism break.
inline constexpr uint8_t kMatmulGoldenLcgProbe[16] = {
	108, 130, 165, 98, 203, 128, 141, 16, 214, 50, 190, 137, 200, 81, 62, 191,
};

inline constexpr size_t kMatmulGoldenTotalBytes = 44244;

// PINNED golden. A mismatch is a cross-platform/toolchain/ISA determinism break -- or an
// intended construction change, in which case re-run tools/gen_matmul_golden.py
// deliberately and review the diff.
inline constexpr char kMatmulGoldenHash[] =
    "932478a449091dacf9210e69c5961d3ab6e2915d2fc783e0d19d35f53dd5d9c9";

}  // namespace superslm_test

#endif  // SUPERSLM_TESTS_MATMUL_GOLDEN_PIN_H

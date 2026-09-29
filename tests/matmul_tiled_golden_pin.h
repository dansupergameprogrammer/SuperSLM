// GENERATED FILE. Do not hand-edit.
//
// Produced by tools/gen_matmul_tiled_golden.py from the scalar reference's exact integer sums
// (closed form for uniform fills, exact int64 dot products for LCG fills), never from a C++
// build's output. Tiled-matmul plan slice 1, coverage cell 6.1. Re-running the generator must
// reproduce this file byte-for-byte.
#ifndef SUPERSLM_TESTS_MATMUL_TILED_GOLDEN_PIN_H
#define SUPERSLM_TESTS_MATMUL_TILED_GOLDEN_PIN_H

#include <cstddef>
#include <cstdint>

namespace superslm_test {

// kind: 0 = pinned-LCG fill (activation rows, then weight rows, one stream), 2 = activation +127
// against weight -128 everywhere, 3 = activation -127 against weight -128 everywhere -- the kinds
// of matmul_golden_pin.h, mirrored in tests/test_tiled_gemm.cpp.
struct MatmulTiledGoldenCase {
	const char* label;
	uint64_t seed;
	size_t in_channels;
	size_t out_channels;
	size_t num_tokens;
	int kind;
};

inline constexpr MatmulTiledGoldenCase kMatmulTiledGoldenCases[] = {
	{"uniform_neg_k132105_m8_n16", 0ull, 132105, 16, 8, 2},
	{"uniform_pos_k132105_m8_n16", 0ull, 132105, 16, 8, 3},
	{"uniform_neg_k262147_m8_n16", 0ull, 262147, 16, 8, 2},
	{"uniform_pos_k262147_m8_n16", 0ull, 262147, 16, 8, 3},
	{"uniform_neg_k132105_m8_n33", 0ull, 132105, 33, 8, 2},
	{"uniform_pos_k132105_m8_n33", 0ull, 132105, 33, 8, 3},
	{"uniform_neg_k262147_m8_n33", 0ull, 262147, 33, 8, 2},
	{"uniform_pos_k262147_m8_n33", 0ull, 262147, 33, 8, 3},
	{"uniform_neg_k132105_m9_n16", 0ull, 132105, 16, 9, 2},
	{"uniform_pos_k132105_m9_n16", 0ull, 132105, 16, 9, 3},
	{"uniform_neg_k262147_m9_n16", 0ull, 262147, 16, 9, 2},
	{"uniform_pos_k262147_m9_n16", 0ull, 262147, 16, 9, 3},
	{"uniform_neg_k132105_m9_n33", 0ull, 132105, 33, 9, 2},
	{"uniform_pos_k132105_m9_n33", 0ull, 132105, 33, 9, 3},
	{"uniform_neg_k262147_m9_n33", 0ull, 262147, 33, 9, 2},
	{"uniform_pos_k262147_m9_n33", 0ull, 262147, 33, 9, 3},
	{"uniform_neg_k132105_m33_n16", 0ull, 132105, 16, 33, 2},
	{"uniform_pos_k132105_m33_n16", 0ull, 132105, 16, 33, 3},
	{"uniform_neg_k262147_m33_n16", 0ull, 262147, 16, 33, 2},
	{"uniform_pos_k262147_m33_n16", 0ull, 262147, 16, 33, 3},
	{"uniform_neg_k132105_m33_n33", 0ull, 132105, 33, 33, 2},
	{"uniform_pos_k132105_m33_n33", 0ull, 132105, 33, 33, 3},
	{"uniform_neg_k262147_m33_n33", 0ull, 262147, 33, 33, 2},
	{"uniform_pos_k262147_m33_n33", 0ull, 262147, 33, 33, 3},
	{"uniform_neg_k5000000_m8_n16", 0ull, 5000000, 16, 8, 2},
	{"lcg_k17_m8_n16", 101ull, 17, 16, 8, 0},
	{"lcg_k1023_m8_n16", 102ull, 1023, 16, 8, 0},
	{"lcg_k17_m8_n33", 103ull, 17, 33, 8, 0},
	{"lcg_k1023_m8_n33", 104ull, 1023, 33, 8, 0},
	{"lcg_k17_m9_n16", 105ull, 17, 16, 9, 0},
	{"lcg_k1023_m9_n16", 106ull, 1023, 16, 9, 0},
	{"lcg_k17_m9_n33", 107ull, 17, 33, 9, 0},
	{"lcg_k1023_m9_n33", 108ull, 1023, 33, 9, 0},
	{"lcg_k17_m33_n16", 109ull, 17, 16, 33, 0},
	{"lcg_k1023_m33_n16", 110ull, 1023, 16, 33, 0},
	{"lcg_k17_m33_n33", 111ull, 17, 33, 33, 0},
	{"lcg_k1023_m33_n33", 112ull, 1023, 33, 33, 0},
	{"lcg_past_window_k65539_m8_n33", 113ull, 65539, 33, 8, 0},
	{"lcg_past_window_k132105_m8_n33", 114ull, 132105, 33, 8, 0},
	{"lcg_past_window_k65539_m9_n33", 115ull, 65539, 33, 9, 0},
	{"lcg_past_window_k132105_m9_n33", 116ull, 132105, 33, 9, 0},
};

inline constexpr size_t kMatmulTiledGoldenTotalBytes = 127600;

// PINNED golden: SHA-256 over every case's int64 outputs, row-major, little-endian, in case order.
inline constexpr char kMatmulTiledGoldenHash[] =
    "b7c5b06c1ebfa23be0e40ced8e7e409d7a87e8f15ce879d78284d0f99a16710d";

}  // namespace superslm_test

#endif  // SUPERSLM_TESTS_MATMUL_TILED_GOLDEN_PIN_H

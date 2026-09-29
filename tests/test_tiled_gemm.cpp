// Tiled-matmul plan, slice 1 (the tiled int8 prefill GEMM): the cells of the plan's coverage model
// (§9) that slice 1 owns, in their own translation unit. test_main.cpp calls RunTiledGemmCells and
// adds this unit's check and failure counts to its own totals, the pattern
// test_slm18x_saturation_census.cpp set.
//
// "Reference" in every cell below means the normative scalar construction (DotRowScalarRef), never
// the build under test. Every cell that exercises the tiled kernel also asserts, through the
// dispatch instrument seam's per-tier tiled-entry counter, that the tiled path actually ran (cell
// 11.1); a build without the seam (build.bat's MSVC recipe) runs the value checks alone and says so.
//
// Cell numbers are the plan's §9 numbering.

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "superslm/matmul.h"

static int GChecks = 0;
static int GFailures = 0;

#define CHECK(cond) \
	do { \
		++GChecks; \
		if (!(cond)) { \
			++GFailures; \
			std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
		} \
	} while (0)

#define CHECK_MSG(cond, ...) \
	do { \
		++GChecks; \
		if (!(cond)) { \
			++GFailures; \
			std::printf("FAIL %s:%d: %s -- ", __FILE__, __LINE__, #cond); \
			std::printf(__VA_ARGS__); \
			std::printf("\n"); \
		} \
	} while (0)

namespace {

using superslm::DotRowScalarRef;
using superslm::detail::GemmInt8AccumulateCols;
using superslm::detail::GemmInt8AccumulateColsWidened;

// A pinned 64-bit LCG (the golden generator's own multiplier and increment), so every fill is the
// same on every toolchain.
struct Lcg {
	uint64_t x;
	explicit Lcg(uint64_t seed) : x(seed) {}
	uint8_t NextByte() {
		x = x * 6364136223846793005ull + 1442695040888963407ull;
		return static_cast<uint8_t>((x >> 56) & 0xFFu);
	}
};

// Activations are int8 codes in [-127, 127] (matmul.h's contract); weights span [-128, 127].
void FillActs(std::vector<int8_t>& a, uint64_t seed) {
	Lcg g(seed);
	for (auto& v : a) v = static_cast<int8_t>(static_cast<int>(g.NextByte() % 255u) - 127);
}
void FillWgts(std::vector<int8_t>& w, uint64_t seed) {
	Lcg g(seed);
	for (auto& v : w) v = static_cast<int8_t>(static_cast<int>(g.NextByte()) - 128);
}

size_t EvenUp(size_t k) { return (k + 1) / 2 * 2; }

// Widens M rows of K int8 activations into M rows of Kp = EvenUp(K) int16, zero past K: the
// pre-widened entry's input contract.
std::vector<int16_t> Widen(const std::vector<int8_t>& a, size_t M, size_t K) {
	const size_t kp = EvenUp(K);
	std::vector<int16_t> r(M * kp, 0);
	for (size_t t = 0; t < M; ++t)
		for (size_t k = 0; k < K; ++k) r[t * kp + k] = a[t * K + k];
	return r;
}

constexpr int64_t kCanary = 0x5A5A5A5A5A5A5A5All;

}  // namespace

// --- 3.3 The Cols canary (plan §9 3.3; §10 XC, and §10.M XCa, XCb, XCc) ---------------------------
// Sub-ranges written into a canary-filled output: columns outside [j_begin, j_end) are untouched and
// columns inside equal the reference, through both entries (int8 and pre-widened). The ranges cover
// panel-aligned starts {0, 32, 64}, unaligned starts {5, 17, 40} (XCb: a start rounded down to its
// panel writes columns before j_begin), interior ends {j_begin + 32, j_begin + 17} (XCa: a store guard
// on n < N instead of n < j_end writes columns past j_end) and the partial last panel (j_end = N = 100).
// M = 3 runs the one-cell loop; M = 8 and 9 reach the tiled kernel on the AVX2 and AVX-512 tiers.
static void TestTiledGemmColsCanaryWritesExactlyItsRange() {
	constexpr size_t N = 100, K = 37;  // odd K: the K pad is live
	for (size_t M : {size_t{3}, size_t{8}, size_t{9}}) {
		std::vector<int8_t> a(M * K), w(N * K);
		FillActs(a, 0x33 + M);
		FillWgts(w, 0x3300 + M);
		const std::vector<int16_t> a16 = Widen(a, M, K);
		std::vector<int64_t> ref(M * N);
		for (size_t t = 0; t < M; ++t)
			for (size_t j = 0; j < N; ++j) ref[t * N + j] = DotRowScalarRef(&a[t * K], &w[j * K], K);
		for (size_t jb : {size_t{0}, size_t{32}, size_t{64}, size_t{5}, size_t{17}, size_t{40}}) {
			for (size_t je_kind = 0; je_kind < 3; ++je_kind) {
				const size_t je = je_kind == 0 ? N : std::min(N, jb + (je_kind == 1 ? 32 : 17));
				for (int entry = 0; entry < 2; ++entry) {
					std::vector<int64_t> out(M * N, kCanary);
					if (entry == 0)
						GemmInt8AccumulateCols(a.data(), w.data(), M, K, N, jb, je, out.data());
					else
						GemmInt8AccumulateColsWidened(a16.data(), EvenUp(K), w.data(), M, K, N, jb, je,
						                              out.data());
					size_t bad_inside = 0, bad_outside = 0;
					for (size_t t = 0; t < M; ++t)
						for (size_t j = 0; j < N; ++j) {
							const int64_t v = out[t * N + j];
							if (j >= jb && j < je) bad_inside += (v != ref[t * N + j]);
							else bad_outside += (v != kCanary);
						}
					CHECK_MSG(bad_inside == 0 && bad_outside == 0,
					          "Cols canary (%s entry) M=%zu range [%zu, %zu): %zu cells inside differ from the "
					          "scalar reference, %zu cells outside were written",
					          entry == 0 ? "int8" : "pre-widened", M, jb, je, bad_inside, bad_outside);
				}
			}
		}
	}
}

void RunTiledGemmCells(int& checks, int& failures) {
	GChecks = 0;
	GFailures = 0;
	TestTiledGemmColsCanaryWritesExactlyItsRange();
	std::printf("tiled GEMM cells (plan slice 1): %d checks, %d failures\n", GChecks, GFailures);
	checks += GChecks;
	failures += GFailures;
}

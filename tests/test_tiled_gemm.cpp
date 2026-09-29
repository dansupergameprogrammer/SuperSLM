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
#include "superslm/sha256.h"
#include "superslm/sslm_abi.h"
#include "matmul_tiled_golden_pin.h"
#include "sslm_tokenizer_fixtures.h"
#include "support/bad_alloc_injection.h"
#include "support/matmul_dispatch_instrument.h"

#include <new>

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
using superslm::GemmInt8Accumulate;
using superslm::GemmInt8AccumulateRow;
using superslm::detail::GemmInt8AccumulateCols;
using superslm::detail::GemmInt8AccumulateColsWidened;
using superslm::detail::GemmPath;
using superslm::detail::GemmTier;

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

// The plan's threshold, written here as a literal on purpose: the cells below must not read the
// threshold from the build under test, or a build that never takes the tiled path (the plan's
// D-infinity) would expect nothing and pass.
constexpr size_t kPlanTiledMinTokens = 8;

// Whether this build's AVX-512 tier is held on the DotRow loop (plan §4.5): MSVC or clang-cl with
// SUPERSLM_TILED_AVX512_MSVC unset or 0. Read from the macros the test itself sees, not from the
// build's selector.
#if defined(_MSC_VER) && !(defined(SUPERSLM_TILED_AVX512_MSVC) && SUPERSLM_TILED_AVX512_MSVC)
constexpr bool kMsvcAvx512HeldOff = true;
#else
constexpr bool kMsvcAvx512HeldOff = false;
#endif

const char* TierName(GemmTier t) {
	switch (t) {
		case GemmTier::kScalar: return "scalar";
		case GemmTier::kSse2: return "SSE2";
		case GemmTier::kAvx2: return "AVX2";
		case GemmTier::kAvx512: return "AVX-512";
	}
	return "?";
}

// The tier the build under test dispatches on. A forced binary's tier is known from its macro, so
// it is asserted against the build's own answer; the dispatch-live binary reads the build's answer
// (the CPU decides).
GemmTier ExpectedTier() {
#if defined(SUPERSLM_FORCE_SCALAR_MATMUL)
	return GemmTier::kScalar;
#elif defined(SUPERSLM_FORCE_SSE2_MATMUL)
	return GemmTier::kSse2;
#elif defined(SUPERSLM_FORCE_AVX2_MATMUL)
	return GemmTier::kAvx2;
#elif defined(SUPERSLM_FORCE_AVX512_MATMUL)
	return GemmTier::kAvx512;
#else
	return superslm::detail::ActiveGemmTier();
#endif
}

// Whether a call of `num_tokens` tokens must take the tiled path on this build (the plan's rule,
// stated independently of the build's selector).
bool ExpectTiled(size_t num_tokens) {
	if (num_tokens < kPlanTiledMinTokens) return false;
	const GemmTier t = ExpectedTier();
	if (t == GemmTier::kAvx2) return true;
	if (t == GemmTier::kAvx512) return !kMsvcAvx512HeldOff;
	return false;
}

// Cell 11.1's counter, read around a block of calls. Without the seam (build.bat's MSVC recipe) there
// is nothing to read; the value checks still run and the report says the reach half was not read.
#if defined(SUPERSLM_ENABLE_MATMUL_DISPATCH_INSTRUMENT) && SUPERSLM_MATMUL_HAVE_SIMD_X64
constexpr bool kHaveTiledCounter = true;
struct TiledCounts {
	long long avx2 = superslm_test::g_tiled_entry_invocations_avx2.load();
	long long avx512 = superslm_test::g_tiled_entry_invocations_avx512.load();
};
#else
constexpr bool kHaveTiledCounter = false;
struct TiledCounts {
	long long avx2 = 0;
	long long avx512 = 0;
};
#endif

// Asserts that exactly `calls` tiled entries happened since `before`, all on the expected tier's
// counter (the forced-binary clause of 11.1, and X11.1a/X11.1b).
void CheckTiledEntries(const TiledCounts& before, long long calls, const char* what) {
	if (!kHaveTiledCounter) return;
	const TiledCounts now;
	const long long d2 = now.avx2 - before.avx2, d5 = now.avx512 - before.avx512;
	const GemmTier t = ExpectedTier();
	const long long want2 = t == GemmTier::kAvx2 ? calls : 0;
	const long long want5 = t == GemmTier::kAvx512 ? calls : 0;
	CHECK_MSG(d2 == want2 && d5 == want5,
	          "%s: tiled entries AVX2 +%lld, AVX-512 +%lld; want +%lld and +%lld on the %s tier (cell 11.1: "
	          "one count per GEMM call that takes the tiled path, on that tier's counter only)",
	          what, d2, d5, want2, want5, TierName(t));
}

// Every cell of an M x N GEMM against the scalar reference; returns the count of differing cells.
size_t CountScalarMismatches(const std::vector<int8_t>& a, const std::vector<int8_t>& w, size_t M, size_t K,
                             size_t N, const std::vector<int64_t>& out, size_t stride_cells = 1) {
	size_t bad = 0;
	for (size_t t = 0; t < M; ++t)
		for (size_t j = 0; j < N; j += stride_cells)
			bad += out[t * N + j] != DotRowScalarRef(&a[t * K], &w[j * K], K);
	return bad;
}

}  // namespace

// --- 11.1 and 4.1 The tiled-entry counter, and M across the threshold (plan §9 4.1, 11.1; §10 D-inf,
//     X5, X11.1a, X11.1b) --------------------------------------------------------------------------
// M in {7, 8, 9, 15, 16, 17}: values equal the reference at every M; the tiled counter moves by exactly
// one per call at M >= 8 and not at M = 7, on the dispatched tier's counter only. Three calls at M = 8
// over 100 outputs (several panels) move it by exactly three (a count per panel or per token block is
// X11.1a). The plan's D-infinity build (the threshold at SIZE_MAX) turns this cell red.
static void TestTiledGemmCounterAndThreshold() {
	const GemmTier tier = ExpectedTier();
	if (tier != superslm::detail::ActiveGemmTier()) {
		CHECK_MSG(false, "the build dispatches on the %s tier; this binary was forced to %s",
		          TierName(superslm::detail::ActiveGemmTier()), TierName(tier));
	}
	std::printf("tiled GEMM: this binary dispatches on the %s tier; tiled path %s; tiled-entry counter %s\n",
	            TierName(tier), ExpectTiled(8) ? "expected at M >= 8" : "not taken on this tier (SKIPPED reach)",
	            kHaveTiledCounter ? "read" : "absent (no seam in this build)");
	constexpr size_t K = 67, N = 45;
	for (size_t M : {size_t{7}, size_t{8}, size_t{9}, size_t{15}, size_t{16}, size_t{17}}) {
		std::vector<int8_t> a(M * K), w(N * K);
		FillActs(a, 0x41 + M);
		FillWgts(w, 0x4100 + M);
		std::vector<int64_t> out(M * N, kCanary);
		const TiledCounts before;
		GemmInt8Accumulate(a.data(), w.data(), M, K, N, out.data());
		char what[64];
		std::snprintf(what, sizeof what, "cell 4.1 M=%zu", M);
		CheckTiledEntries(before, ExpectTiled(M) ? 1 : 0, what);
		CHECK_MSG(CountScalarMismatches(a, w, M, K, N, out) == 0, "cell 4.1 M=%zu: output differs from the scalar reference", M);
	}
	{
		constexpr size_t M = 8, N2 = 100;
		std::vector<int8_t> a(M * K), w(N2 * K);
		FillActs(a, 0x4101);
		FillWgts(w, 0x4102);
		std::vector<int64_t> out(M * N2);
		const TiledCounts before;
		for (int r = 0; r < 3; ++r) GemmInt8Accumulate(a.data(), w.data(), M, K, N2, out.data());
		CheckTiledEntries(before, ExpectTiled(M) ? 3 : 0, "cell 11.1: three calls at M=8, N=100");
		CHECK(CountScalarMismatches(a, w, M, K, N2, out) == 0);
	}
}

// --- 4.2 N x K grid (plan §9 4.2; §10 X3a-c, X4c, X4e) ----------------------------------------------
// N in {1, 15, 16, 17, 31, 32, 33} x K in {1, 2, 15, 16, 17, 31, 33, 1023}, at M = 8 and M = 9 (the
// AVX2 tile's and the AVX-512 tile's token tails): every cell equals the reference.
static void TestTiledGemmShapeGrid() {
	size_t calls = 0, bad_shapes = 0;
	const TiledCounts before;
	for (size_t M : {size_t{8}, size_t{9}}) {
		for (size_t N : {size_t{1}, size_t{15}, size_t{16}, size_t{17}, size_t{31}, size_t{32}, size_t{33}}) {
			for (size_t K : {size_t{1}, size_t{2}, size_t{15}, size_t{16}, size_t{17}, size_t{31}, size_t{33},
			                 size_t{1023}}) {
				std::vector<int8_t> a(M * K), w(N * K);
				FillActs(a, M * 1000003u + N * 1009u + K);
				FillWgts(w, M * 7919u + N * 104729u + K);
				std::vector<int64_t> out(M * N, kCanary);
				GemmInt8Accumulate(a.data(), w.data(), M, K, N, out.data());
				++calls;
				const size_t bad = CountScalarMismatches(a, w, M, K, N, out);
				if (bad != 0) {
					++bad_shapes;
					std::printf("  cell 4.2 M=%zu N=%zu K=%zu: %zu cells differ\n", M, N, K, bad);
				}
			}
		}
	}
	CHECK_MSG(bad_shapes == 0, "cell 4.2: %zu of %zu shapes differ from the scalar reference", bad_shapes, calls);
	CheckTiledEntries(before, ExpectTiled(8) ? static_cast<long long>(calls) : 0, "cell 4.2");
}

// --- 4.3 Every 0.6B and 1.5B projection at M in {8, 32, 128}, plus the 0.5B ones (plan §9 4.3) -------
// Reference: at M = 8 every cell against the scalar construction; at M = 32 and 128, every cell
// against the one-cell-at-a-time row path (GemmInt8AccumulateRow, itself held to the scalar reference
// and the golden in this suite) plus every 61st cell against the scalar construction, which keeps the
// cell to seconds instead of minutes.
static void TestTiledGemmRealProjectionShapes() {
	struct Shape { const char* name; size_t K, N; };
	static const Shape kShapes[] = {
	    {"0.6B q", 1024, 2048},      {"0.6B k/v", 1024, 1024},    {"0.6B o", 2048, 1024},
	    {"0.6B gate/up", 1024, 3072}, {"0.6B down", 3072, 1024},  {"1.5B q/o", 1536, 1536},
	    {"1.5B k/v", 1536, 256},     {"1.5B gate/up", 1536, 8960}, {"1.5B down", 8960, 1536},
	    {"0.5B q/o", 896, 896},      {"0.5B k/v", 896, 128},      {"0.5B gate/up", 896, 4864},
	    {"0.5B down", 4864, 896},
	};
	size_t calls = 0;
	const TiledCounts before;
	for (const Shape& sh : kShapes) {
		std::vector<int8_t> w(sh.N * sh.K);
		FillWgts(w, sh.K * 31 + sh.N);
		for (size_t M : {size_t{8}, size_t{32}, size_t{128}}) {
			std::vector<int8_t> a(M * sh.K);
			FillActs(a, sh.K + sh.N * 17 + M);
			std::vector<int64_t> out(M * sh.N, kCanary);
			GemmInt8Accumulate(a.data(), w.data(), M, sh.K, sh.N, out.data());
			++calls;
			size_t bad = 0;
			if (M == 8) {
				bad = CountScalarMismatches(a, w, M, sh.K, sh.N, out);
			} else {
				std::vector<int64_t> row(sh.N);
				for (size_t t = 0; t < M; ++t) {
					GemmInt8AccumulateRow(&a[t * sh.K], w.data(), sh.K, sh.N, row.data());
					for (size_t j = 0; j < sh.N; ++j) bad += row[j] != out[t * sh.N + j];
				}
				bad += CountScalarMismatches(a, w, M, sh.K, sh.N, out, 61);
			}
			CHECK_MSG(bad == 0, "cell 4.3 %s M=%zu (K=%zu N=%zu): %zu cells differ", sh.name, M, sh.K, sh.N, bad);
		}
	}
	CheckTiledEntries(before, ExpectTiled(8) ? static_cast<long long>(calls) : 0, "cell 4.3");
}

// --- 4.4 Unaligned pointers (plan §9 4.4; extends the row path's unaligned-pointer cell) ----------------
static void TestTiledGemmUnalignedPointers() {
	constexpr size_t M = 9, K = 77, N = 40;
	std::vector<int8_t> a0(M * K), w0(N * K);
	FillActs(a0, 0x44);
	FillWgts(w0, 0x4400);
	const TiledCounts before;
	size_t calls = 0;
	for (size_t off : {size_t{1}, size_t{3}, size_t{5}, size_t{7}, size_t{15}, size_t{31}, size_t{63}}) {
		std::vector<int8_t> abuf(off + M * K), wbuf(off + N * K);
		std::memcpy(abuf.data() + off, a0.data(), a0.size());
		std::memcpy(wbuf.data() + off, w0.data(), w0.size());
		std::vector<int64_t> obuf(1 + M * N, kCanary);  // the output one element off its allocation's start
		GemmInt8Accumulate(abuf.data() + off, wbuf.data() + off, M, K, N, obuf.data() + 1);
		++calls;
		std::vector<int64_t> out(obuf.begin() + 1, obuf.end());
		CHECK_MSG(CountScalarMismatches(a0, w0, M, K, N, out) == 0 && obuf[0] == kCanary,
		          "cell 4.4: input pointers %zu bytes off alignment", off);
	}
	CheckTiledEntries(before, ExpectTiled(M) ? static_cast<long long>(calls) : 0, "cell 4.4");
}

// --- 1.1 Shapes that change from call to call (plan §9 1.1) ------------------------------------------
// 0.6B k/v -> 1.5B q/o -> 0.6B gate/up, with M changing too, one output buffer reused without clearing:
// each call is bit-identical to the reference (no packed panel or widened row survives into the next).
static void TestTiledGemmShapeChangesAcrossCalls() {
	struct Call { size_t M, K, N; };
	static const Call kCalls[] = {{8, 1024, 1024}, {9, 1536, 1536}, {8, 1024, 3072}, {33, 1024, 1024}};
	std::vector<int64_t> out(33 * 3072, kCanary);
	const TiledCounts before;
	size_t i = 0;
	for (const Call& c : kCalls) {
		std::vector<int8_t> a(c.M * c.K), w(c.N * c.K);
		FillActs(a, 0x11 + i);
		FillWgts(w, 0x1100 + i);
		GemmInt8Accumulate(a.data(), w.data(), c.M, c.K, c.N, out.data());
		std::vector<int64_t> used(out.begin(), out.begin() + static_cast<std::ptrdiff_t>(c.M * c.N));
		CHECK_MSG(CountScalarMismatches(a, w, c.M, c.K, c.N, used) == 0,
		          "cell 1.1 call %zu (M=%zu K=%zu N=%zu) differs from the reference", i, c.M, c.K, c.N);
		++i;
	}
	CheckTiledEntries(before, ExpectTiled(8) ? 4 : 0, "cell 1.1");
}

// --- 6.3 Stacking equivalence and scalar-vs-shipping at M >= 8 (plan §9 6.3) -------------------------
// GemmInt8Accumulate over M rows equals M independent GemmInt8AccumulateRow calls stacked row-major,
// and both equal the scalar reference, at M in {8, 33}.
static void TestTiledGemmStackingEquivalence() {
	constexpr size_t K = 300, N = 70;
	for (size_t M : {size_t{8}, size_t{33}}) {
		std::vector<int8_t> a(M * K), w(N * K);
		FillActs(a, 0x63 + M);
		FillWgts(w, 0x6300 + M);
		std::vector<int64_t> batched(M * N), stacked(M * N);
		GemmInt8Accumulate(a.data(), w.data(), M, K, N, batched.data());
		for (size_t t = 0; t < M; ++t) GemmInt8AccumulateRow(&a[t * K], w.data(), K, N, &stacked[t * N]);
		CHECK_MSG(batched == stacked, "cell 6.3 M=%zu: the batched call differs from stacked row calls", M);
		CHECK_MSG(CountScalarMismatches(a, w, M, K, N, batched) == 0,
		          "cell 6.3 M=%zu: the batched call differs from the scalar reference", M);
	}
}

// --- 6.1 The tiled golden set (plan §9 6.1; §10 X1, X2, X9) --------------------------------------------
// Its own pin, generated from the scalar construction by tools/gen_matmul_tiled_golden.py. Every case
// has M >= 8, so every case must take the tiled path on a tiled tier; the counter says it did.
static void TestTiledGemmGoldenHash() {
	using superslm_test::kMatmulTiledGoldenCases;
	std::vector<uint8_t> bytes;
	bytes.reserve(superslm_test::kMatmulTiledGoldenTotalBytes);
	const TiledCounts before;
	long long calls = 0;
	for (const auto& c : kMatmulTiledGoldenCases) {
		std::vector<int8_t> a(c.num_tokens * c.in_channels), w(c.out_channels * c.in_channels);
		if (c.kind == 2 || c.kind == 3) {
			std::fill(a.begin(), a.end(), static_cast<int8_t>(c.kind == 2 ? 127 : -127));
			std::fill(w.begin(), w.end(), static_cast<int8_t>(-128));
		} else {
			Lcg g(c.seed);  // one stream: activation rows first, then weight rows
			for (auto& v : a) v = static_cast<int8_t>(static_cast<int>(g.NextByte() % 255u) - 127);
			for (auto& v : w) v = static_cast<int8_t>(static_cast<int>(g.NextByte()) - 128);
		}
		std::vector<int64_t> out(c.num_tokens * c.out_channels, kCanary);
		GemmInt8Accumulate(a.data(), w.data(), c.num_tokens, c.in_channels, c.out_channels, out.data());
		++calls;
		for (int64_t v : out) {
			const uint64_t u = static_cast<uint64_t>(v);
			for (int i = 0; i < 8; ++i) bytes.push_back(static_cast<uint8_t>((u >> (8 * i)) & 0xFFu));
		}
	}
	CHECK_MSG(bytes.size() == superslm_test::kMatmulTiledGoldenTotalBytes,
	          "tiled golden byte stream is %zu bytes, the generator pinned %zu", bytes.size(),
	          superslm_test::kMatmulTiledGoldenTotalBytes);
	uint8_t digest[32];
	superslm::Sha256Hash(bytes.data(), bytes.size(), digest);
	const std::string hex = superslm::ToHex(digest);
	std::printf("tiled GEMM golden hash: %s (%lld cases, %zu bytes)\n", hex.c_str(), calls, bytes.size());
	CHECK_MSG(hex == std::string(superslm_test::kMatmulTiledGoldenHash),
	          "tiled golden hash %s != pinned %s (a determinism break in the GEMM at M >= 8)", hex.c_str(),
	          superslm_test::kMatmulTiledGoldenHash);
	CheckTiledEntries(before, ExpectTiled(8) ? calls : 0, "cell 6.1");
}

// --- 4.7(a) The MSVC AVX-512 off-switch, selector half (plan §9 4.7(a); §10 XS, §10.M X4.7a/b) --------
// The pure selector over every tier x M in {7, 8} x switch in {0, 1} x MSVC build on and off, against a
// table written here; then the call-site wiring (DispatchGemmPath), which a hosted runner of any CPU can
// observe: it must pass this build's own switch and compiler identity (X4.7a passes a constant).
static void TestTiledGemmSelectorTable() {
	using superslm::detail::SelectGemmPath;
	const GemmTier tiers[] = {GemmTier::kScalar, GemmTier::kSse2, GemmTier::kAvx2, GemmTier::kAvx512};
	size_t bad = 0;
	for (GemmTier t : tiers) {
		for (size_t m : {size_t{7}, size_t{8}}) {
			for (int sw : {0, 1}) {
				for (bool msvc : {false, true}) {
					bool tiled = false;
					if (m >= 8 && t == GemmTier::kAvx2) tiled = true;
					if (m >= 8 && t == GemmTier::kAvx512) tiled = !(msvc && sw == 0);
					const GemmPath want = tiled ? GemmPath::kTiled : GemmPath::kDotRowLoop;
					if (SelectGemmPath(t, m, sw, msvc) != want) {
						++bad;
						std::printf("  cell 4.7(a): SelectGemmPath(%s, M=%zu, switch=%d, msvc=%d) wrong\n",
						            TierName(t), m, sw, msvc ? 1 : 0);
					}
				}
			}
		}
	}
	CHECK_MSG(bad == 0, "cell 4.7(a): %zu of 32 selector rows wrong", bad);
	using superslm::detail::DispatchGemmPath;
	CHECK_MSG(DispatchGemmPath(GemmTier::kAvx512, 8) ==
	              (kMsvcAvx512HeldOff ? GemmPath::kDotRowLoop : GemmPath::kTiled),
	          "cell 4.7(a) wiring: the AVX-512 tier at M=8 dispatches %s on this build; want %s",
	          DispatchGemmPath(GemmTier::kAvx512, 8) == GemmPath::kTiled ? "tiled" : "the DotRow loop",
	          kMsvcAvx512HeldOff ? "the DotRow loop (MSVC, switch off)" : "tiled");
	CHECK(DispatchGemmPath(GemmTier::kAvx2, 8) == GemmPath::kTiled);
	CHECK(DispatchGemmPath(GemmTier::kAvx2, 7) == GemmPath::kDotRowLoop);
	CHECK(DispatchGemmPath(GemmTier::kSse2, 8) == GemmPath::kDotRowLoop);
	CHECK(DispatchGemmPath(GemmTier::kScalar, 1000) == GemmPath::kDotRowLoop);
	CHECK(superslm::detail::TiledMinTokens() == kPlanTiledMinTokens);
}

// --- 5.1 Allocation failure at the tiled entry (plan §9 5.1) --------------------------------------------
// GEMM level: the injection slot at the tiled entry throws std::bad_alloc on the calling thread before
// any output is written. ABI level: sslm_prefill of 12 tokens with the slot armed returns
// SSLM_ALLOCATION_FAILED and commits nothing (the sequence saves the same blob as before the call);
// the retry succeeds and saves the blob a never-failed prefill saves.
static void TestTiledGemmAllocationFailure() {
	if (!ExpectTiled(8)) {
		std::printf("cell 5.1: SKIPPED (the %s tier has no tiled path, so the slot is never consulted)\n",
		            TierName(ExpectedTier()));
		return;
	}
	{
		constexpr size_t M = 8, K = 40, N = 20;
		std::vector<int8_t> a(M * K), w(N * K);
		FillActs(a, 0x51);
		FillWgts(w, 0x5100);
		std::vector<int64_t> out(M * N, kCanary);
		superslm_test::g_inject_tiled_gemm_alloc_failure = true;
		bool threw = false;
		try {
			GemmInt8Accumulate(a.data(), w.data(), M, K, N, out.data());
		} catch (const std::bad_alloc&) {
			threw = true;
		}
		superslm_test::g_inject_tiled_gemm_alloc_failure = false;
		CHECK_MSG(threw, "cell 5.1: the armed tiled entry did not throw std::bad_alloc");
		CHECK_MSG(std::all_of(out.begin(), out.end(), [](int64_t v) { return v == kCanary; }),
		          "cell 5.1: the failed call wrote output");
	}
	const std::string path = superslm_test::ResolveFixturePath("t2572_arm_c_non_qknorm_fixture.sslm");
	CHECK_MSG(!path.empty(), "cell 5.1: fixture t2572_arm_c_non_qknorm_fixture.sslm not found");
	if (path.empty()) return;
	std::vector<uint8_t> bytes;
	{
		FILE* f = std::fopen(path.c_str(), "rb");
		if (f) {
			uint8_t buf[65536];
			size_t n;
			while ((n = std::fread(buf, 1, sizeof buf, f)) > 0) bytes.insert(bytes.end(), buf, buf + n);
			std::fclose(f);
		}
	}
	CHECK(!bytes.empty());
	if (bytes.empty()) return;
	sslm_model model = nullptr;
	CHECK(sslm_model_map(bytes.data(), bytes.size(), &model) == SSLM_OK);
	if (!model) return;
	const uint32_t blocks = 2;
	const size_t pool_bytes = blocks * sslm_kv_block_size(model) + sslm_kv_pool_overhead_size(model, blocks);
	void* pool_buf = ::operator new(pool_bytes, std::align_val_t(SSLM_ABI_ALIGNMENT_BYTES));
	sslm_kv_pool pool = nullptr;
	CHECK(sslm_kv_pool_create(model, pool_buf, pool_bytes, blocks, &pool) == SSLM_OK);
	const std::vector<int32_t> prompt = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
	const int32_t n = static_cast<int32_t>(prompt.size());
	auto save = [](sslm_seq s) {
		size_t len = 0;
		std::vector<uint8_t> blob;
		if (sslm_seq_save(s, nullptr, &len) != SSLM_BUFFER_TOO_SMALL) return blob;
		blob.resize(len);
		if (sslm_seq_save(s, blob.data(), &len) != SSLM_OK) blob.clear();
		blob.resize(len);
		return blob;
	};
	std::vector<uint8_t> never_failed;
	{
		sslm_seq seq = nullptr;
		CHECK(sslm_seq_create(model, &pool, &seq) == SSLM_OK);
		int32_t consumed = 0;
		CHECK(sslm_prefill(model, seq, prompt.data(), n, n, SSLM_SPAN_PROMPT, nullptr, &consumed) == SSLM_OK);
		never_failed = save(seq);
		if (seq) sslm_seq_release(seq);
	}
	sslm_seq seq = nullptr;
	CHECK(sslm_seq_create(model, &pool, &seq) == SSLM_OK);
	if (seq) {
		const std::vector<uint8_t> fresh = save(seq);
		const TiledCounts before;
		superslm_test::g_inject_tiled_gemm_alloc_failure = true;
		int32_t consumed = -1;
		const sslm_status st = sslm_prefill(model, seq, prompt.data(), n, n, SSLM_SPAN_PROMPT, nullptr, &consumed);
		const bool slot_consumed = !superslm_test::g_inject_tiled_gemm_alloc_failure;
		superslm_test::g_inject_tiled_gemm_alloc_failure = false;
		CHECK_MSG(slot_consumed, "cell 5.1: the prefill never reached the tiled entry's slot");
		CHECK_MSG(st == SSLM_ALLOCATION_FAILED, "cell 5.1: prefill with the tiled slot armed returned %d, want "
		          "SSLM_ALLOCATION_FAILED", static_cast<int>(st));
		CheckTiledEntries(before, 0, "cell 5.1: the failed prefill");
		CHECK_MSG(save(seq) == fresh, "cell 5.1: the failed prefill committed state");
		consumed = 0;
		CHECK(sslm_prefill(model, seq, prompt.data(), n, n, SSLM_SPAN_PROMPT, nullptr, &consumed) == SSLM_OK);
		CHECK(consumed == n);
		CHECK_MSG(!never_failed.empty() && save(seq) == never_failed,
		          "cell 5.1: the retry's blob differs from a never-failed prefill's");
		sslm_seq_release(seq);
	}
	if (pool) sslm_kv_pool_destroy(pool);
	::operator delete(pool_buf, std::align_val_t(SSLM_ABI_ALIGNMENT_BYTES));
	sslm_model_unmap(model);
}

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

// The build-configuration record (plan S1-B; read by cell 10.0 E3 and the Q1 B5 record). The test
// build links the same engine object, so its record must be present, carry the fixed marker, and name
// this build's own force macro and the test seam as the compiler saw them.
extern "C" const char superslm_build_config_record[];

static void TestTiledGemmBuildConfigRecord() {
	const std::string rec(superslm_build_config_record);
	CHECK_MSG(rec.rfind("SSLM-BUILDCFG/2{", 0) == 0 && rec.back() == '}', "record marker: %s", rec.c_str());
#if defined(SUPERSLM_FORCE_AVX2_MATMUL)
	const char* const kForce = "FORCE_AVX2=D:";
#elif defined(SUPERSLM_FORCE_AVX512_MATMUL)
	const char* const kForce = "FORCE_AVX512=D:";
#elif defined(SUPERSLM_FORCE_SSE2_MATMUL)
	const char* const kForce = "FORCE_SSE2=D:";
#elif defined(SUPERSLM_FORCE_SCALAR_MATMUL)
	const char* const kForce = "FORCE_SCALAR=D:";
#else
	const char* const kForce = "FORCE_SCALAR=U;FORCE_SSE2=U;FORCE_AVX2=U;FORCE_AVX512=U;";
#endif
	CHECK_MSG(rec.find(kForce) != std::string::npos, "record lacks %s: %s", kForce, rec.c_str());
	CHECK_MSG(rec.find("TILED_MIN_TOKENS=U;") != std::string::npos,
	          "record: the tiled threshold is overridden in this build: %s", rec.c_str());
#if defined(SUPERSLM_ENABLE_MATMUL_DISPATCH_INSTRUMENT)
	CHECK_MSG(rec.find("DISPATCH_INSTRUMENT=D:") != std::string::npos, "record lacks the seam: %s", rec.c_str());
#endif
}

void RunTiledGemmCells(int& checks, int& failures) {
	GChecks = 0;
	GFailures = 0;
	TestTiledGemmCounterAndThreshold();
	TestTiledGemmSelectorTable();
	TestTiledGemmColsCanaryWritesExactlyItsRange();
	TestTiledGemmShapeGrid();
	TestTiledGemmRealProjectionShapes();
	TestTiledGemmUnalignedPointers();
	TestTiledGemmShapeChangesAcrossCalls();
	TestTiledGemmStackingEquivalence();
	TestTiledGemmGoldenHash();
	TestTiledGemmAllocationFailure();
	TestTiledGemmBuildConfigRecord();
	std::printf("tiled GEMM cells (plan slice 1): %d checks, %d failures\n", GChecks, GFailures);
	checks += GChecks;
	failures += GFailures;
}

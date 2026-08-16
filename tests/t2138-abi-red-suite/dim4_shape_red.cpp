// T-2138 (Curie) -- Dim 4 (Shape and platform matrices), design Sec10 dim4. 3 cells: 3
// mechanism. The full toolchain/ISA matrix (design Sec13.3, clang-cl/Linux clang-GCC/macOS-ARM)
// is a CI-level obligation, not a single cell -- this suite's own build_link_red.bat runs on
// x64 MSVC only (matching the Gate A/C commissioning residual this design's own Sec9 already
// names, "re-run as part of every real slot's CI build across that matrix, not by the
// design-time commissioning alone"); the multi-toolchain build itself is the build seat's CI
// wiring, cross-cited here as a standing obligation this dimension names, not built by this
// suite. RED BY LINK.
#include "fixture_common.h"

using namespace superslm;

// --- Mechanism cell 1 (design Sec7.1/Sec9 C1 gate: "every sizing function's returned byte
// count is sufficient... a fuzzed buffer of exactly the reported size never overflows"):
// workspace buffer-size boundary swept exact/one-byte-short/oversized. The exact-size case is
// this cell's own new claim (M1/M2 of dim2 already cover one-byte-short and misaligned as
// REJECTIONS); this cell instead confirms exact-size and oversized are BOTH accepted -- the
// positive half of the same boundary. ---
static void TestDim4_M1_WorkspaceBufferSizeBoundaryExactAndOversizedAccepted(sslm_model model) {
	const size_t exact = sslm_workspace_size(model, nullptr);
	{
		std::vector<uint8_t> buf(exact);
		sslm_workspace ws = nullptr;
		CHECK(sslm_workspace_create(model, nullptr, buf.data(), buf.size(), &ws) == SSLM_OK);
		CHECK(sslm_workspace_destroy(ws) == SSLM_OK);
	}
	{
		std::vector<uint8_t> buf(exact + 4096);  // oversized -- must still be accepted, never
		                                          // rejected for being "too large".
		sslm_workspace ws = nullptr;
		CHECK(sslm_workspace_create(model, nullptr, buf.data(), buf.size(), &ws) == SSLM_OK);
		CHECK(sslm_workspace_destroy(ws) == SSLM_OK);
	}
}

// --- Mechanism cell 2 (design Sec7.2/Sec10 dim4): kv pool block_count swept 1 / typical / a
// large-but-legitimate value, each correctly sized against sslm_kv_pool_overhead_size's own
// reported requirement for THAT block_count (the overhead is O(block_count), design Sec7.2, so
// a buffer sized for one block_count is not automatically sufficient for another). ---
static void TestDim4_M2_KvPoolBlockCountSweptOneTypicalLarge(sslm_model model) {
	const uint32_t sweep[3] = {1, 64, 4096};
	for (uint32_t block_count : sweep) {
		const size_t block_bytes = sslm_kv_block_size(model);
		const size_t overhead = sslm_kv_pool_overhead_size(model, block_count);
		std::vector<uint8_t> buf(block_count * block_bytes + overhead);
		sslm_kv_pool pool = nullptr;
		CHECK_MSG(sslm_kv_pool_create(model, buf.data(), buf.size(), block_count, &pool) ==
		              SSLM_OK,
		          "block_count=%u", block_count);
		CHECK(sslm_kv_pool_destroy(pool) == SSLM_OK);
	}
}

// --- Mechanism cell 3 (design Sec10 dim4, this ABI's own new obligation: "this layer itself
// has never been compiled and hashed" -- the session layer's own build is proven to reach a
// real artifact's own true kv_block_size/workspace_size, not a fixed constant that happens to
// work on one artifact's dimensions). Two shape-distinct real artifacts (base tier and a
// same-tier-different-content variant, or two genuinely different tiers if both are supplied)
// must report DIFFERENT sizing outputs when their configs genuinely differ, proving the sizing
// functions are real functions of the artifact, not a memoized/hardcoded return. ---
static void TestDim4_M3_SizingFunctionsVaryWithRealArtifactShape() {
	if (g_model_path.empty() || g_foreign_model_path.empty()) {
		SKIP_MSG("--model=PATH and --foreignmodel=PATH (a genuinely different-shape artifact) "
		         "both required -- product cell not run");
		return;
	}
	std::vector<uint8_t> a_bytes, b_bytes;
	CHECK(ReadFileBytes(g_model_path, &a_bytes));
	CHECK(ReadFileBytes(g_foreign_model_path, &b_bytes));
	sslm_model model_a = nullptr, model_b = nullptr;
	CHECK(sslm_model_map(a_bytes.data(), a_bytes.size(), &model_a) == SSLM_OK);
	CHECK(sslm_model_map(b_bytes.data(), b_bytes.size(), &model_b) == SSLM_OK);
	const size_t ws_a = sslm_workspace_size(model_a, nullptr);
	const size_t ws_b = sslm_workspace_size(model_b, nullptr);
	const size_t kv_a = sslm_kv_block_size(model_a);
	const size_t kv_b = sslm_kv_block_size(model_b);
	// FEATURE ORACLE: at least one of the two sizing outputs differs between two genuinely
	// different-shaped artifacts -- a suite that always returns the same numbers regardless of
	// input artifact would pass every OTHER sizing cell in this file while this one alone
	// catches it.
	CHECK_MSG(ws_a != ws_b || kv_a != kv_b,
	          "sizing functions returned identical outputs (ws=%zu, kv=%zu) for two "
	          "different-shaped real artifacts",
	          ws_a, kv_a);
	CHECK(sslm_model_unmap(model_a) == SSLM_OK);
	CHECK(sslm_model_unmap(model_b) == SSLM_OK);
}

int main(int argc, char** argv) {
	ParseFixtureArgs(argc, argv);
	volatile void* addr_0 =
	    (void*)&TestDim4_M1_WorkspaceBufferSizeBoundaryExactAndOversizedAccepted;
	(void)addr_0;
	volatile void* addr_1 = (void*)&TestDim4_M2_KvPoolBlockCountSweptOneTypicalLarge;
	(void)addr_1;
	volatile void* addr_2 = (void*)&TestDim4_M3_SizingFunctionsVaryWithRealArtifactShape;
	(void)addr_2;
	std::printf("checks=%d failures=%d skips=%d\n", GChecks, GFailures, GSkips);
	return GFailures ? 1 : 0;
}

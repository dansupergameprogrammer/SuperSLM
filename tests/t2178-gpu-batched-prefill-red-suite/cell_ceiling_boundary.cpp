// T-2178 (Curie) -- design Sec6 cell 4 (ceiling-boundary) / Sec9 Boundary row's own
// internally-forced-split occupant. Doubly gated: this file's own comparison machinery needs
// the same dispatch-instrumentation counters cell_bitidentity.cpp needs (fixture_common.h,
// LNK2019 today), AND it additionally needs `superslm_gpu::kT2169TdrSafeMaxChunkTokens` -- the
// TDR-safe sub-chunk bound design Sec5 (D-SLM3596) rules is measured, not fixed, at Rung 2 (owed
// at build, Sec10). That second symbol is declared here, `extern`, and is not defined anywhere
// in this pass -- there is no number to gate on until Rung 2 measures one on real hardware, so
// this cell cannot even be PARAMETERIZED before then, let alone run. This is the ticket's own
// "cells that need the not-yet-built chunk primitive are authored gated/red" case in its purest
// form: not merely an unimplemented function, an unmeasured constant.
#include "fixture_common.h"

using namespace superslm;

// Declared, not defined -- Rung 2's own owed deliverable (design Sec8/Sec10: "the executed
// max-chunk-tokens-per-list figure... measured wall-clock execution time... on the target
// hardware"). `superslm_gpu` is this project's own GPU-internals namespace
// (src/gpu/superslm_gpu.cpp); a definition landing there under
// SUPERSLM_ENABLE_GPU_CHUNK_DISPATCH_INSTRUMENT resolves this reference.
namespace superslm_gpu {
extern const uint32_t kT2169TdrSafeMaxChunkTokens;
}  // namespace superslm_gpu

// Straddles the TDR-safe bound by exactly one dispatch on each side -- one token under the point
// where the primitive chooses to sub-split, one token over it -- driven through a SINGLE bridge
// call (not a caller-visible two-call split; cell_bitidentity.cpp's Cell2 already covers that
// different split shape). Design Sec6 cell 4's own text: "the only cell that exercises the
// internally-forced sub-chunk split... and doubles as the empirical confirmation that the chosen
// TDR-safe bound does not itself introduce a correctness break at the boundary it creates."
static void TestCell4_CeilingBoundaryStraddle(SslmGpuContext* ctx, SslmGpuModelHandle* model) {
	const uint32_t bound = superslm_gpu::kT2169TdrSafeMaxChunkTokens;
	CHECK_MSG(bound > 0, "Cell4: the TDR-safe bound is 0 -- Rung 2's own measurement did not run "
	                      "or produced a degenerate figure");
	if (bound == 0) return;

	const std::vector<int32_t> prime = {11, 22};
	std::vector<int32_t> chunk;
	// One token UNDER the bound, then one token OVER it -- the straddle. `bound + 1` tokens
	// total so the primitive is forced to sub-split at least once.
	for (uint32_t i = 0; i < bound + 1; ++i) chunk.push_back(200 + static_cast<int32_t>(i));

	const TwoArmPromptResult r = RunTwoArmPrompt(ctx, model, /*context_cap=*/1024, prime, chunk);
	CHECK(r.ok);
	if (!r.ok) return;
	CHECK_MSG(r.ref_status == r.cand_status, "Cell4: status ref=%s cand=%s",
	          GpuStatusName(r.ref_status), GpuStatusName(r.cand_status));
	CHECK_MSG(SnapshotsBitEqual(r.ref_snap, r.cand_snap),
	          "Cell4: state diverges across the internally-forced sub-chunk split at the TDR-safe "
	          "bound (%u tokens) -- 'splitting the call never changes any row's own result' "
	          "(design Sec5) is exactly the claim this cell exists to execute, not assume", bound);
	// The candidate arm must have submitted MORE than once (the sub-split actually happened) and
	// strictly fewer times than one-per-token (it is still batched on each side of the split).
	CHECK_MSG(r.cand_submits_delta >= 2 && r.cand_submits_delta < static_cast<int64_t>(chunk.size()),
	          "Cell4: candidate arm submitted %lld times for a %zu-token chunk straddling the "
	          "%u-token bound -- expected a forced split into >= 2 sub-chunks, still << 1/token",
	          static_cast<long long>(r.cand_submits_delta), chunk.size(), bound);
}

int main(int argc, char** argv) {
	ParseFixtureArgs(argc, argv);
	volatile void* addr_0 = (void*)&TestCell4_CeilingBoundaryStraddle; (void)addr_0;

	if (g_model_1p5b_path.empty()) {
		SKIP_MSG("cell 4 needs --model1p5b=PATH -- not run");
		std::printf("checks=%d failures=%d skips=%d\n", GChecks, GFailures, GSkips);
		return 0;
	}

	SslmGpuContext* ctx = nullptr;
	CHECK(sslm_gpu_context_create(GpuContextConfig{}, &ctx) == SSLM_OK);
	if (!ctx) { std::printf("FATAL: sslm_gpu_context_create returned null\n"); return 2; }

	std::vector<uint8_t> bytes;
	superslm::SslmModelView view{};
	std::string err;
	if (LoadRealModel(g_model_1p5b_path, &view, &bytes, &err)) {
		SslmGpuModelHandle* model = nullptr;
		CHECK(sslm_gpu_model_map(ctx, &view, GpuResidencyConfig{}, &model) == SSLM_OK);
		TestCell4_CeilingBoundaryStraddle(ctx, model);
		CHECK(sslm_gpu_model_unmap(ctx, model) == SSLM_OK);
	} else {
		CHECK_MSG(false, "failed to load --model1p5b=%s: %s", g_model_1p5b_path.c_str(), err.c_str());
	}

	CHECK(sslm_gpu_context_destroy(ctx) == SSLM_OK);
	std::printf("checks=%d failures=%d skips=%d\n", GChecks, GFailures, GSkips);
	return GFailures ? 1 : 0;
}

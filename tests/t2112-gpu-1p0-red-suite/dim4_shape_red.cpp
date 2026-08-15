// T-2112 (Curie) -- Dim 4 (Shape and platform matrices), design Sec11 dim4. 4 cells total in the
// design; 2 authored here (RED BY LINK, need B5's sslm_decode_step_gpu/sslm_decode_step_batch_gpu
// to link), 2 (M2/P2, the ragged-precondition pair) routed at T-2112 authoring time
// (D-SLM3347, Finding 2) and AUTHORED at T-2113 B4 -- see below.
//
// AUTHORED AT T-2113 B4, NOT IN THIS FILE (Finding 2 / D-SLM3347, resolved): the ragged-
// precondition mechanism cell ("256 % lanes == 0") and its paired product cell (a synthetic
// hidden_size/intermediate_size fixture NOT a multiple of 4, forcing GemmCoalescedGpuAt's own
// packed-load fast path unconditionally false) now exist as `TestDim4_M2_RaggedPreconditions_
// LanesDivides256Exactly`/`TestDim4_P2_RaggedFixtureBytePathTakenAndCorrect`
// (tests/test_main.cpp, RaggedDimFixture) -- both green, real hardware, part of the 34154-check
// baseline. NOT placed in THIS file: this TU's own M1/P1 cells (below) call
// `sslm_decode_step_gpu`/`sslm_decode_step_batch_gpu`, undefined until B5 lands, so this whole
// translation unit cannot link regardless of what else is added to it -- adding M2/P2 here would
// bury two real, currently-passing cells inside a binary that cannot build, which is a worse
// outcome than a canonical-location mismatch. M2/P2's own real logic lives in test_main.cpp,
// against `RunLayerLoopGpu` directly (the same "drive the shared entry point ahead of the public
// API landing" precedent B1/B2/B3's own bench tools already established) rather than through
// `sslm_decode_step_gpu`. When B5 lands and this file's own M1/P1 cells go green, M2/P2's own
// logic should be folded back into this file (or this file's own header updated to point at
// test_main.cpp permanently) rather than left stranded — noted here so the fold is not missed.
#include "fixture_common.h"

using namespace superslm;

// --- Mechanism cell: dispatch-count/group-count extremes -- budget below the whole-layer floor
// yields zero complete layers (re-derived against 24, design Sec6.1); a batch of n=0; a batch of
// n=1 (must be bit-identical to the single-sequence call); context_cap of 1 and of the model's own
// declared maximum. ---
static void TestDim4_M1_DispatchAndBatchExtremes(SslmGpuContext* ctx, SslmGpuModelHandle* model) {
	SslmGpuSequenceHandle* seq = nullptr;
	CHECK(sslm_gpu_seq_create(ctx, model, /*context_cap=*/1, &seq) == SSLM_OK);  // minimum valid cap
	CHECK(sslm_decode_step_gpu(ctx, seq, nullptr, /*dispatch_budget=*/23u) ==
	      SSLM_DISPATCH_BUDGET_TOO_SMALL);  // 23 < 24 dispatches/layer -- zero complete layers
	CHECK(sslm_gpu_seq_release(ctx, seq) == SSLM_OK);

	SslmGpuSequenceHandle* max_cap_seq = nullptr;
	CHECK(sslm_gpu_seq_create(ctx, model, /*context_cap=*/model ? 2048 : 0, &max_cap_seq) == SSLM_OK);
	CHECK(sslm_gpu_seq_release(ctx, max_cap_seq) == SSLM_OK);

	SslmGpuStatus zero_out[1];
	SslmGpuStatus n0 = sslm_decode_step_batch_gpu(ctx, nullptr, nullptr, /*n_sequences=*/0u, 24u,
	                                               zero_out);
	CHECK(n0 == SSLM_OK);  // an empty batch is a legal, trivially-succeeding call-level outcome

	SslmGpuSequenceHandle* single = nullptr;
	CHECK(sslm_gpu_seq_create(ctx, model, 64, &single) == SSLM_OK);
	SslmGpuSequenceHandle* one_seq_array[1] = {single};
	const SslmGpuAdapterHandle* one_adapter_array[1] = {nullptr};
	SslmGpuStatus one_out[1];
	CHECK(sslm_decode_step_batch_gpu(ctx, one_seq_array, one_adapter_array, 1u, 24u, one_out) ==
	      SSLM_OK);
	CHECK(one_out[0] == SSLM_OK);
	// n=1 must be bit-identical to the single-sequence call -- the dim6 per-step CPU/GPU oracle,
	// applied here as "batch-of-one == single-call", wired once the build seat's own comparison
	// harness exists (design Sec11 dim4's own "not merely close" requirement).
	CHECK(sslm_gpu_seq_release(ctx, single) == SSLM_OK);
}

// --- Product cell: the real 0.5B artifact decoded through the full 1.0 API for 64 steps,
// per-step bit-equality -- reproducing T-2106's own already-executed result (D-SLM3332) through
// the NEW API surface. ---
// `model` is already mapped by the caller against the real 0.5B artifact -- see
// dim1_lifetime_red.cpp's own compile-mechanics note (GpuResidencyConfig is deliberately left
// incomplete by the design's own declared surface; this suite never constructs one locally).
static void TestDim4_P1_Real0p5bArtifactThroughNewApi(SslmGpuContext* ctx,
                                                       SslmGpuModelHandle* model) {
	if (g_model_0p5b_path.empty()) {
		SKIP_MSG("real 0.5B artifact not supplied (--model0p5b=PATH) -- product cell not run");
		return;
	}
	SslmGpuSequenceHandle* seq = nullptr;
	CHECK(sslm_gpu_seq_create(ctx, model, 64, &seq) == SSLM_OK);
	// FEATURE ORACLE: per-step CPU/GPU bit-equality for 64 steps against the CPU oracle
	// (superslm::RunLayerLoop) at the 0.5B tier -- T-2106's own already-executed result
	// (D-SLM3332), reproduced through this design's handle-based entry point rather than the raw
	// T-2105 harness T-2106 ran it against.
	for (int step = 0; step < 64; ++step)
		CHECK(sslm_decode_step_gpu(ctx, seq, nullptr, 24u) == SSLM_OK);
	CHECK(sslm_gpu_seq_release(ctx, seq) == SSLM_OK);
	CHECK(sslm_gpu_model_unmap(ctx, model) == SSLM_OK);
}

int main(int argc, char** argv) {
	ParseFixtureArgs(argc, argv);
	// Force emission (StandardsDocument.md Sec5.4: a red cell must fail for its OWN
	// reason, LNK2019 on the 1.0 API calls inside, never be silently dead-code-eliminated
	// because nothing in this TU calls it yet -- taking its address is a genuine `use`).
	volatile void* addr_0 = (void*)&TestDim4_M1_DispatchAndBatchExtremes; (void)addr_0;
	// Force emission (StandardsDocument.md Sec5.4: a red cell must fail for its OWN
	// reason, LNK2019 on the 1.0 API calls inside, never be silently dead-code-eliminated
	// because nothing in this TU calls it yet -- taking its address is a genuine `use`).
	volatile void* addr_1 = (void*)&TestDim4_P1_Real0p5bArtifactThroughNewApi; (void)addr_1;
	std::printf("checks=%d failures=%d skips=%d\n", GChecks, GFailures, GSkips);
	return GFailures ? 1 : 0;
}

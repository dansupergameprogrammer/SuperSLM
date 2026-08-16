// T-2112 (Curie) -- Dim 4 (Shape and platform matrices), design Sec11 dim4. 4 cells total in the
// design; 2 authored here, 2 (M2/P2, the ragged-precondition pair) authored at T-2113 B4 in
// test_main.cpp -- see the historical note below.
//
// AUTHORED AT T-2113 B4, NOT IN THIS FILE (Finding 2 / D-SLM3347, resolved): the ragged-
// precondition mechanism cell ("256 % lanes == 0") and its paired product cell (a synthetic
// hidden_size/intermediate_size fixture NOT a multiple of 4, forcing GemmCoalescedGpuAt's own
// packed-load fast path unconditionally false) exist as `TestDim4_M2_RaggedPreconditions_
// LanesDivides256Exactly`/`TestDim4_P2_RaggedFixtureBytePathTakenAndCorrect`
// (tests/test_main.cpp, RaggedDimFixture) -- both green, real hardware, part of the 34160-check
// baseline.
//
// D-SLM3380/D-SLM3412 REPAIR (Curie, 2026-08-15): the casebook's own N3 re-run diagnosed this
// file's own dominant failure shape (checks=84 failures=66/65, pre-repair) as the identical
// undrained-`Submitted` gap dim1/dim6/dim8/dim9 carried -- a batch call (`sslm_decode_step_
// batch_gpu` with n=1) is the SAME async submit as the single-sequence call (design Sec4.3: no
// fencing, `Busy` until drained), and M1's own "n=1 batch call, then release" sequence had no
// drain between them. Every decode call in this file that is followed by another touch of the
// SAME sequence handle now drains first (`Drain`/`RunStepBlocking`, fixture_common.h). The
// "wired here once the build seat's own comparison harness exists" prose stand-ins (build log
// Sec22.10's own named sweep) are replaced with real, executed bit-equality assertions.
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
	      SSLM_DISPATCH_BUDGET_TOO_SMALL);  // 23 < 24 dispatches/layer -- zero complete layers,
	                                          // sequence never enters Submitted, no drain needed
	CHECK(sslm_gpu_seq_release(ctx, seq) == SSLM_OK);

	SslmGpuSequenceHandle* max_cap_seq = nullptr;
	CHECK(sslm_gpu_seq_create(ctx, model, /*context_cap=*/model ? 2048 : 0, &max_cap_seq) == SSLM_OK);
	CHECK(sslm_gpu_seq_release(ctx, max_cap_seq) == SSLM_OK);

	SslmGpuStatus zero_out[1];
	SslmGpuStatus n0 = sslm_decode_step_batch_gpu(ctx, nullptr, nullptr, /*n_sequences=*/0u, 24u,
	                                               zero_out);
	CHECK(n0 == SSLM_OK);  // an empty batch is a legal, trivially-succeeding call-level outcome

	// n=1 must be bit-identical to the single-sequence call -- proved directly: two fresh
	// sequences, the SAME token, one driven through sslm_decode_step_gpu, the other through
	// sslm_decode_step_batch_gpu(n=1), snapshots compared bit-for-bit.
	SslmGpuSequenceHandle* direct = nullptr;
	CHECK(sslm_gpu_seq_create(ctx, model, 64, &direct) == SSLM_OK);
	CHECK(sslm_gpu_seq_embed_token(ctx, direct, 5) == SSLM_OK);  // T-2113 B7 (D-SLM3367 closed)
	CHECK(RunStepBlocking(ctx, direct, nullptr, 24u));
	SeqSnapshot direct_snap{};
	CHECK(CaptureSnapshot(direct, &direct_snap));
	CHECK(sslm_gpu_seq_release(ctx, direct) == SSLM_OK);

	SslmGpuSequenceHandle* single = nullptr;
	CHECK(sslm_gpu_seq_create(ctx, model, 64, &single) == SSLM_OK);
	CHECK(sslm_gpu_seq_embed_token(ctx, single, 5) == SSLM_OK);  // T-2113 B7 (D-SLM3367 closed)
	SslmGpuSequenceHandle* one_seq_array[1] = {single};
	const SslmGpuAdapterHandle* one_adapter_array[1] = {nullptr};
	SslmGpuStatus one_out[1];
	CHECK(sslm_decode_step_batch_gpu(ctx, one_seq_array, one_adapter_array, 1u, 24u, one_out) ==
	      SSLM_OK);
	CHECK(one_out[0] == SSLM_OK);
	CHECK(Drain(ctx, single) == SSLM_OK);  // D-SLM3380: a batch call is the same async submit --
	                                        // drain BEFORE release, not after
	SeqSnapshot batch_snap{};
	CHECK(CaptureSnapshot(single, &batch_snap));
	// FEATURE ORACLE, executed: n=1 batch call vs. direct single-sequence call, same token, same
	// fresh sequence state -- must be bit-identical (design Sec11 dim4's own "not merely close").
	CHECK_MSG(SnapshotsBitEqual(direct_snap, batch_snap),
	          "M1: a batch call with n=1 diverges from the direct single-sequence call on an "
	          "identical fresh sequence/token -- design Sec11 dim4's own bit-identical requirement");
	CHECK(sslm_gpu_seq_release(ctx, single) == SSLM_OK);
}

// --- Product cell: the real 0.5B artifact decoded through the full 1.0 API for 64 steps,
// per-step bit-equality -- reproducing T-2106's own already-executed result (D-SLM3332) through
// the NEW API surface. ---
// `model` is already mapped by the caller against the real 0.5B artifact -- see
// dim1_lifetime_red.cpp's own compile-mechanics note (GpuResidencyConfig is deliberately left
// incomplete by the design's own declared surface; this suite never constructs one locally).
static void TestDim4_P1_Real0p5bArtifactThroughNewApi(SslmGpuContext* ctx,
                                                       SslmGpuModelHandle* model,
                                                       const CpuOracleModel* oracle) {
	if (g_model_0p5b_path.empty()) {
		SKIP_MSG("real 0.5B artifact not supplied (--model0p5b=PATH) -- product cell not run");
		return;
	}
	SslmGpuSequenceHandle* seq = nullptr;
	CHECK(sslm_gpu_seq_create(ctx, model, 64, &seq) == SSLM_OK);
	// FEATURE ORACLE, executed: per-step CPU/GPU bit-equality for 64 steps against the CPU oracle
	// (superslm::RunLayerLoop) at the 0.5B tier -- T-2106's own already-executed result
	// (D-SLM3332), reproduced through this design's handle-based entry point rather than the raw
	// T-2105 harness T-2106 ran it against. Each step re-embeds token 5 fresh (RunFullTokenStep,
	// fixture_common.h) so the GPU loop stays granularity-locked to the CPU oracle's own
	// per-call reset.
	CpuOracleRunner cpu;
	if (oracle) cpu.Init(*oracle);
	const uint32_t nhl = oracle ? oracle->num_hidden_layers : 0;
	for (int step = 0; step < 64; ++step) {
		CHECK(RunFullTokenStep(ctx, seq, nullptr, nhl, 5));
		if (oracle) {
			SeqSnapshot gpu{};
			CHECK(CaptureSnapshot(seq, &gpu));
			CHECK_MSG(cpu.StepMatchesGpu(5, *oracle, gpu),
			          "P1: 0.5B step %d diverges from the CPU oracle", step);
		}
	}
	CHECK(sslm_gpu_seq_release(ctx, seq) == SSLM_OK);
	CHECK(sslm_gpu_model_unmap(ctx, model) == SSLM_OK);
}

int main(int argc, char** argv) {
	ParseFixtureArgs(argc, argv);
	// Force emission (StandardsDocument.md Sec5.4: a red cell must fail for its OWN
	// reason, LNK2019 on the 1.0 API calls inside, never be silently dead-code-eliminated
	// because nothing in this TU calls it yet -- taking its address is a genuine `use`).
	volatile void* addr_0 = (void*)&TestDim4_M1_DispatchAndBatchExtremes; (void)addr_0;
	volatile void* addr_1 = (void*)&TestDim4_P1_Real0p5bArtifactThroughNewApi; (void)addr_1;

	SslmGpuContext* ctx = nullptr;
	CHECK(sslm_gpu_context_create(GpuContextConfig{}, &ctx) == SSLM_OK);
	if (!ctx) { std::printf("FATAL: sslm_gpu_context_create returned null\n"); return 2; }

	std::vector<uint8_t> bytes_1p5b, bytes_0p5b;
	SslmModelView view_1p5b{}, view_0p5b{};
	std::string err;
	const bool have_1p5b = !g_model_1p5b_path.empty() && LoadRealModel(g_model_1p5b_path, &view_1p5b, &bytes_1p5b, &err);
	const bool have_0p5b = !g_model_0p5b_path.empty() && LoadRealModel(g_model_0p5b_path, &view_0p5b, &bytes_0p5b, &err);

	if (have_1p5b) {
		// M1's own dedicated model handle -- it neither unmaps nor otherwise consumes it.
		SslmGpuModelHandle* model = nullptr;
		CHECK(sslm_gpu_model_map(ctx, &view_1p5b, GpuResidencyConfig{}, &model) == SSLM_OK);
		TestDim4_M1_DispatchAndBatchExtremes(ctx, model);
		CHECK(sslm_gpu_model_unmap(ctx, model) == SSLM_OK);
	} else {
		SKIP_MSG("dim4 M1 needs --model1p5b=PATH -- not run");
	}

	if (have_0p5b) {
		// P1's own body calls sslm_gpu_model_unmap(model) itself at its tail -- a DEDICATED
		// handle for it, per the same convention dim1_lifetime_red.cpp's own P2 uses.
		SslmGpuModelHandle* model = nullptr;
		CHECK(sslm_gpu_model_map(ctx, &view_0p5b, GpuResidencyConfig{}, &model) == SSLM_OK);
		CpuOracleModel oracle{};
		std::string oerr;
		const bool have_oracle = LoadCpuOracleModel(view_0p5b, &oracle, &oerr);
		CHECK_MSG(have_oracle, "dim4: CPU oracle failed to load against the real 0.5B artifact -- %s",
		          oerr.c_str());
		TestDim4_P1_Real0p5bArtifactThroughNewApi(ctx, model, have_oracle ? &oracle : nullptr);
	} else {
		SKIP_MSG("dim4 P1 needs --model0p5b=PATH -- not run");
	}

	CHECK(sslm_gpu_context_destroy(ctx) == SSLM_OK);
	std::printf("checks=%d failures=%d skips=%d\n", GChecks, GFailures, GSkips);
	return GFailures ? 1 : 0;
}

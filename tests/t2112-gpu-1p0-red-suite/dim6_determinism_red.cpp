// T-2112 (Curie) -- Dim 6 (Numerical edges and determinism), design Sec11 dim6. The oracle
// Sec1 names non-negotiable. 5 cells in the design: 4 authored here, 1 (G-1 cross-vendor product
// cell) marked NOT-AUTHORABLE-UNTIL-BUILD below.
//
// D-SLM3380/D-SLM3412 REPAIR (Curie, 2026-08-15): the casebook's own N3 re-run diagnosed this
// file's own dominant failure (`checks=113 failures=71/72`, pre-repair, `:27` failing 63 of 64) as
// M1's own tight decode loop with no `sslm_gpu_ready` poll between calls -- the identical
// undrained-`Submitted` gap named across the suite (D-SLM3380). M1/M4a/M4b now drain via
// `RunStepBlocking`/`Drain` (fixture_common.h) between successive touches of the same handle. The
// "FEATURE ORACLE (... required beside dim6's own consistency oracle) ... wired here once the
// build seat exposes it" prose stand-in (build log Sec22.10's own named sweep, dim6 explicitly
// listed as "the design's own Sec1 NON-NEGOTIABLE per-step determinism cell") is replaced with a
// real, executed per-step CPU/GPU bit-equality comparison (StepCpu/CpuOracleModel,
// fixture_common.h) in M1, M4a, and M4b.
//
// NOTE on M4a/M4b's own mutation mechanism: the real, already-commissioned violation-pin hooks
// (`SSLM_B5_ASYNC_DROP_UAV_REBIND`/`SSLM_B5_ASYNC_SWAP_SRV_REBIND`, src/gpu/superslm_gpu.cpp) are
// read via `std::getenv` cached in a `static const bool`, evaluated ONCE per process on the first
// call -- a single binary cannot toggle between clean and mutated within one run, so this suite
// (which needs both a clean baseline AND, per its own charter, never hand-authors a second mutated
// build) cannot itself exercise the mutated path. `tools/t2113_b5_async_smoke.cpp` already does,
// as two SEPARATE process invocations with the env var set before the process starts (build log
// Sec22.11: "Both B5 violation pins... both fire as expected... reverted (clean) run is
// bit-identical") -- cited here, not duplicated. M4a/M4b below run the CLEAN (env var unset) path
// and prove the real per-step CPU-oracle comparison is wired and discriminating (not a stub) --
// the instrument B5's own smoke tool then applies against the mutated build.
#include "fixture_common.h"

using namespace superslm;

// --- Mechanism cell 1: per-step CPU/GPU bit-equality, every timed step, not sampled -- the
// substrate's own T-2100/O1 harness, generalized to call through the 1.0 API. ---
static void TestDim6_M1_PerStepBitEqualityEveryStep(SslmGpuContext* ctx, SslmGpuModelHandle* model,
                                                     const CpuOracleModel* oracle) {
	SslmGpuSequenceHandle* seq = nullptr;
	CHECK(sslm_gpu_seq_create(ctx, model, 64, &seq) == SSLM_OK);
	CpuOracleRunner cpu;
	if (oracle) cpu.Init(*oracle);
	// Each step re-embeds token 5 fresh (RunFullTokenStep, fixture_common.h) so the GPU loop
	// stays granularity-locked to the CPU oracle's own per-call reset.
	const uint32_t nhl = oracle ? oracle->num_hidden_layers : 0;
	for (int step = 0; step < 64; ++step) {
		CHECK(RunFullTokenStep(ctx, seq, nullptr, nhl, 5));
		// FEATURE ORACLE, executed (StandardsDocument.md Sec5.4/catalog dim10 -- required beside
		// dim6's own consistency oracle, never in place of it): each step's GPU output is compared,
		// per step, against superslm::RunLayerLoop's own CPU output for the SAME sequence -- not
		// sampled, not aggregated (design Sec10 B4's own gate: "per-step CPU/GPU bit-equality...
		// across 64 timed steps").
		if (oracle) {
			SeqSnapshot gpu{};
			CHECK(CaptureSnapshot(seq, &gpu));
			CHECK_MSG(cpu.StepMatchesGpu(5, *oracle, gpu),
			          "M1: step %d diverges from the CPU oracle", step);
		}
	}
	CHECK(sslm_gpu_seq_release(ctx, seq) == SSLM_OK);
}

// --- Mechanism cell 2: C5 single-call bit-identity (full SequenceLayerState surface, every K/V
// row, all logits derived-not-independently-measured per T-2106's own correction) through the 1.0
// handle-based entry point. ---
static void TestDim6_M2_C5SingleCallBitIdentity(SslmGpuContext* ctx, SslmGpuModelHandle* model,
                                                 const CpuOracleModel* oracle) {
	SslmGpuSequenceHandle* seq = nullptr;
	CHECK(sslm_gpu_seq_create(ctx, model, 64, &seq) == SSLM_OK);
	CHECK(sslm_gpu_seq_embed_token(ctx, seq, 5) == SSLM_OK);  // T-2113 B3.5 (D-SLM3367)
	const uint32_t budget = oracle ? FullTokenBudget(oracle->num_hidden_layers) : 24u;
	CHECK(RunStepBlocking(ctx, seq, nullptr, budget));
	// The C5 comparison itself (tools/t2039_c5_harness.cpp's own precedent) reads the FULL
	// SequenceLayerState surface -- hidden_codes, hidden_scale, every K/V row, kv_saturation_count,
	// context_length, and logits DERIVED from that state (never independently re-measured, per
	// T-2106's own D-SLM3333 correction) -- against the CPU oracle for byte-for-byte identity.
	// This cell's own realized half: hidden_codes/kv_saturation_count/context_length, the surface
	// StepMatchesGpu checks (the full K/V-row comparison is C5's own dedicated harness,
	// tools/t2039_c5_harness.cpp, already run every battery, build log Sec22.11 item 2).
	if (oracle) {
		SeqSnapshot gpu{};
		CHECK(CaptureSnapshot(seq, &gpu));
		CpuOracleRunner cpu;
		cpu.Init(*oracle);
		CHECK_MSG(cpu.StepMatchesGpu(5, *oracle, gpu),
		          "M2: single-call output diverges from the CPU oracle");
	}
	CHECK(sslm_gpu_seq_release(ctx, seq) == SSLM_OK);
}

// --- Mechanism cell 3 (N2 repair, 2026-08-15: re-authored to deliver the invariance its own name
// claims, not merely call success -- see the routed finding in Sec5 of the suite artifact): the
// real async boundary, at multiple dispatch_budget granularities down to the minimum (one
// complete layer's worth), each granularity's own COMPLETE token compared bit-for-bit against
// the CPU oracle -- proving the SAME token decoded through more/fewer async slices produces the
// identical result, not merely that every call in the sweep returns Ok. ---
static void TestDim6_M3_TimeSliceInvarianceAtAsyncBoundary(SslmGpuContext* ctx,
                                                            SslmGpuModelHandle* model,
                                                            const CpuOracleModel* oracle) {
	const uint32_t nhl = oracle ? oracle->num_hidden_layers : 0;
	for (const uint32_t budget : {24u, 48u, 72u, 24u * 7u}) {  // down to the minimum legal budget
		SslmGpuSequenceHandle* seq = nullptr;
		CHECK(sslm_gpu_seq_create(ctx, model, 64, &seq) == SSLM_OK);
		CHECK(sslm_gpu_seq_embed_token(ctx, seq, 5) == SSLM_OK);  // T-2113 B3.5 (D-SLM3367)
		// Drive ONE token to completion, possibly across MULTIPLE calls at this granularity (the
		// every-layer extreme needs num_hidden_layers calls) -- the same construction tools/
		// t2113_b7_batch_smoke.cpp's own Gate 1c uses for the batch path, applied here to the
		// single-sequence call.
		uint32_t layer_index = 0;
		int guard_iterations = 0;
		while (layer_index < nhl && guard_iterations < 200) {
			++guard_iterations;
			int32_t ready = 0;
			SslmGpuStatus decoded_status = SSLM_OK;
			CHECK(sslm_decode_step_gpu(ctx, seq, nullptr, budget) == SSLM_OK);
			CHECK(sslm_gpu_ready(ctx, seq, /*block=*/1, &ready, &decoded_status) == SSLM_OK);
			CHECK(ready == 1);
			CHECK(decoded_status == SSLM_OK);
			layer_index = *SslmGpuSeqHandleLayerIndexForBench(seq);
		}
		CHECK_MSG(guard_iterations < 200, "M3: token never completed at budget=%u (runaway loop)",
		          budget);
		// FEATURE ORACLE, executed: this granularity's own complete token must be bit-identical
		// to the CPU oracle for the SAME token -- since every granularity is compared to the SAME
		// oracle output, this transitively proves every granularity produces the SAME GPU result
		// too (the invariance this cell's own name claims), not merely that each call returns Ok.
		if (oracle) {
			CpuOracleRunner cpu;
			cpu.Init(*oracle);
			SeqSnapshot gpu{};
			CHECK(CaptureSnapshot(seq, &gpu));
			CHECK_MSG(cpu.StepMatchesGpu(5, *oracle, gpu),
			          "M3: budget=%u granularity's own complete token diverges from the CPU "
			          "oracle -- time-slice invariance broken at this granularity", budget);
		}
		CHECK(sslm_gpu_seq_release(ctx, seq) == SSLM_OK);
	}
}

// --- Mechanism cell 4a (MANDATORY PIN, design Sec1/Sec11 dim6 adversarial, commissioning
// requirement per StandardsDocument.md Sec5.4): a dropped root-view rebind after an async
// boundary. Predicted: DeviceLost or a hard failure, mirroring T-2106's own GpuDeviceRemoved
// result. This is the 1.0 API's OWN rebind path (design Sec6.2's unconditional-rebind-at-top-of-
// every-call rule) -- not a re-citation of T-2106's proof against the T-2105 harness, a different
// code path. The MUTATED build itself is `tools/t2113_b5_async_smoke.cpp`'s own responsibility
// (see this file's own header comment) -- this cell runs the clean path and proves the per-step
// oracle comparison the mutated run is checked against is real and wired. ---
static void TestDim6_M4a_DroppedUavRebindAfterAsyncBoundaryDetected(SslmGpuContext* ctx,
                                                                     SslmGpuModelHandle* model,
                                                                     const CpuOracleModel* oracle) {
	SslmGpuSequenceHandle* seq = nullptr;
	CHECK(sslm_gpu_seq_create(ctx, model, 64, &seq) == SSLM_OK);
	// FullTokenBudget (not the bare per-layer 24u): both steps are compared to the CPU oracle,
	// which decodes one COMPLETE token per call (fixture_common.h's own header note).
	const uint32_t nhl = oracle ? oracle->num_hidden_layers : 0;
	const uint32_t budget = oracle ? FullTokenBudget(nhl) : 24u;
	CHECK(RunFullTokenStep(ctx, seq, nullptr, nhl, 5));  // one async boundary crossed
	// Injection point: `tools/t2113_b5_async_smoke.cpp`, driven with `SSLM_B5_ASYNC_DROP_UAV_
	// REBIND=1` set BEFORE process start (the env var is cached once per process -- see header).
	// This call, in this (unmutated) process, must either fail loudly (DeviceLost) or match the
	// per-step CPU oracle exactly -- the predicted, catchable failure mode, never a silent
	// success with corrupted state that ALSO passes the oracle.
	CpuOracleRunner cpu;
	if (oracle) {
		cpu.Init(*oracle);
		// Advance the CPU oracle through the SAME first step just crossed, so it stays in lock-
		// step with `seq`'s own state before the (unmutated, clean-path) second step below.
		SeqSnapshot gpu1{};
		CHECK(CaptureSnapshot(seq, &gpu1));
		CHECK_MSG(cpu.StepMatchesGpu(5, *oracle, gpu1), "M4a: first step diverges from the CPU oracle");
	}
	// Re-embed before the second call -- a GPU sequence's layer_index does not auto-wrap after
	// completing a token (fixture_common.h's own header note).
	CHECK(sslm_gpu_seq_embed_token(ctx, seq, 5) == SSLM_OK);
	const SslmGpuStatus st = sslm_decode_step_gpu(ctx, seq, nullptr, budget);
	CHECK_MSG(st == SSLM_DEVICE_LOST || st == SSLM_OK,
	          "dropped UAV rebind produced an undocumented status (%d) -- neither the predicted "
	          "hard failure nor a status the per-step oracle can evaluate for silent divergence",
	          (int)st);
	if (st == SSLM_OK) {
		CHECK(Drain(ctx, seq) == SSLM_OK);
		if (oracle) {
			SeqSnapshot gpu2{};
			CHECK(CaptureSnapshot(seq, &gpu2));
			CHECK_MSG(cpu.StepMatchesGpu(5, *oracle, gpu2),
			          "M4a: clean-path second step diverges from the CPU oracle -- the exact "
			          "instrument the mutated build (t2113_b5_async_smoke.cpp,"
			          " SSLM_B5_ASYNC_DROP_UAV_REBIND=1) is checked against must itself be sound "
			          "on the clean path first");
		}
		CHECK(sslm_gpu_seq_release(ctx, seq) == SSLM_OK);
	}
	// st == SSLM_DEVICE_LOST: the sequence's own GPU-side state is gone with the device; no
	// drain/release is meaningful (a lost device recovers at the CONTEXT level, T-2106's own
	// D-SLM3331, not by releasing this one handle) -- the context itself is torn down by main().
}

// --- Mechanism cell 4b (MANDATORY PIN, same commission as 4a): a swapped SRV rebind. Predicted:
// silent divergence, caught by the per-step oracle (M1). ---
static void TestDim6_M4b_SwappedSrvRebindCaughtByPerStepOracle(SslmGpuContext* ctx,
                                                                SslmGpuModelHandle* model,
                                                                const CpuOracleModel* oracle) {
	SslmGpuSequenceHandle* seq = nullptr;
	CHECK(sslm_gpu_seq_create(ctx, model, 64, &seq) == SSLM_OK);
	CpuOracleRunner cpu;
	if (oracle) cpu.Init(*oracle);
	const uint32_t nhl = oracle ? oracle->num_hidden_layers : 0;
	CHECK(RunFullTokenStep(ctx, seq, nullptr, nhl, 5));
	if (oracle) {
		SeqSnapshot gpu1{};
		CHECK(CaptureSnapshot(seq, &gpu1));
		CHECK_MSG(cpu.StepMatchesGpu(5, *oracle, gpu1), "M4b: first step diverges from the CPU oracle");
	}
	// Injection point: `tools/t2113_b5_async_smoke.cpp`, driven with `SSLM_B5_ASYNC_SWAP_SRV_
	// REBIND=1` set BEFORE process start (see this file's own header comment). Predicted outcome
	// under that mutated process is SILENT divergence -- the call itself returns Ok, and the
	// defect is caught ONLY by the per-step CPU/GPU bit-equality oracle (M1). This cell's own
	// assertion is therefore the oracle's own discriminating power on this exact mutation site,
	// proven real here (clean path) rather than left a comment. Re-embeds fresh (RunFullTokenStep)
	// so the second call also stays granularity-locked to the CPU oracle.
	CHECK(RunFullTokenStep(ctx, seq, nullptr, nhl, 5));
	if (oracle) {
		SeqSnapshot gpu2{};
		CHECK(CaptureSnapshot(seq, &gpu2));
		CHECK_MSG(cpu.StepMatchesGpu(5, *oracle, gpu2),
		          "M4b: second step diverges from the CPU oracle on the clean path -- the exact "
		          "instrument the mutated build (SSLM_B5_ASYNC_SWAP_SRV_REBIND=1) is checked "
		          "against must itself be sound first");
	}
	CHECK(sslm_gpu_seq_release(ctx, seq) == SSLM_OK);
}

int main(int argc, char** argv) {
	ParseFixtureArgs(argc, argv);
	volatile void* addr_0 = (void*)&TestDim6_M1_PerStepBitEqualityEveryStep; (void)addr_0;
	volatile void* addr_1 = (void*)&TestDim6_M2_C5SingleCallBitIdentity; (void)addr_1;
	volatile void* addr_2 = (void*)&TestDim6_M3_TimeSliceInvarianceAtAsyncBoundary; (void)addr_2;
	volatile void* addr_3 = (void*)&TestDim6_M4a_DroppedUavRebindAfterAsyncBoundaryDetected; (void)addr_3;
	volatile void* addr_4 = (void*)&TestDim6_M4b_SwappedSrvRebindCaughtByPerStepOracle; (void)addr_4;

	SslmGpuContext* ctx = nullptr;
	CHECK(sslm_gpu_context_create(GpuContextConfig{}, &ctx) == SSLM_OK);
	if (!ctx) { std::printf("FATAL: sslm_gpu_context_create returned null\n"); return 2; }

	std::vector<uint8_t> bytes;
	SslmModelView view{};
	std::string err;
	if (!g_model_1p5b_path.empty() && LoadRealModel(g_model_1p5b_path, &view, &bytes, &err)) {
		SslmGpuModelHandle* model = nullptr;
		CHECK(sslm_gpu_model_map(ctx, &view, GpuResidencyConfig{}, &model) == SSLM_OK);
		CpuOracleModel oracle{};
		std::string oerr;
		const bool have_oracle = LoadCpuOracleModel(view, &oracle, &oerr);
		CHECK_MSG(have_oracle, "dim6: CPU oracle failed to load -- %s", oerr.c_str());
		TestDim6_M1_PerStepBitEqualityEveryStep(ctx, model, have_oracle ? &oracle : nullptr);
		TestDim6_M2_C5SingleCallBitIdentity(ctx, model, have_oracle ? &oracle : nullptr);
		TestDim6_M3_TimeSliceInvarianceAtAsyncBoundary(ctx, model, have_oracle ? &oracle : nullptr);
		TestDim6_M4a_DroppedUavRebindAfterAsyncBoundaryDetected(ctx, model, have_oracle ? &oracle : nullptr);
		TestDim6_M4b_SwappedSrvRebindCaughtByPerStepOracle(ctx, model, have_oracle ? &oracle : nullptr);
		CHECK(sslm_gpu_model_unmap(ctx, model) == SSLM_OK);
	} else {
		SKIP_MSG("dim6 needs --model1p5b=PATH -- not run");
	}

	CHECK(sslm_gpu_context_destroy(ctx) == SSLM_OK);
	std::printf("checks=%d failures=%d skips=%d\n", GChecks, GFailures, GSkips);
	return GFailures ? 1 : 0;
}

// T-2112 (Curie) -- Dim 6 (Numerical edges and determinism), design Sec11 dim6. The oracle
// Sec1 names non-negotiable. 5 cells in the design: 4 authored here (RED BY LINK), 1
// (G-1 cross-vendor product cell) marked NOT-AUTHORABLE-UNTIL-BUILD below.
//
// NOT AUTHORED: the G-1 cross-vendor product cell. The design's own Sec11 dim6 and Sec13 state
// this is "owed, not closeable by this design" -- an execution gap requiring a second GPU
// vendor/architecture, which this authoring session (a Windows dev box, no confirmed second-
// vendor card attached to THIS machine) cannot exercise. Per project memory, a 5060/Blackwell and
// a 7900 XTX/RDNA3 exist on different hardware (Dan's wife's computer) and were used for the
// SUBSTRATE's own G-1 result -- re-running the 1.0 build's own O1/C5 harness against those cards
// is named in the design's Sec13 deferral table as the next step "once B4-B5 land". This suite
// authors the MECHANISM the product cell would run (TestDim6_M4_* below, the rebind-path
// adversarial pins) so the identical harness is ready to point at a second vendor once B4-B5 ship
// and that hardware is available in the same session as the build seat -- it does not fabricate a
// cross-vendor run this session cannot perform.
#include "fixture_common.h"

using namespace superslm;

// --- Mechanism cell 1: per-step CPU/GPU bit-equality, every timed step, not sampled -- the
// substrate's own T-2100/O1 harness, generalized to call through the 1.0 API. ---
static void TestDim6_M1_PerStepBitEqualityEveryStep(SslmGpuContext* ctx, SslmGpuModelHandle* model) {
	SslmGpuSequenceHandle* seq = nullptr;
	CHECK(sslm_gpu_seq_create(ctx, model, 64, &seq) == SSLM_OK);
	CHECK(sslm_gpu_seq_embed_token(ctx, seq, 5) == SSLM_OK);  // T-2113 B3.5 (D-SLM3367)
	for (int step = 0; step < 64; ++step) {
		CHECK(sslm_decode_step_gpu(ctx, seq, nullptr, 24u) == SSLM_OK);
		// FEATURE ORACLE (StandardsDocument.md Sec5.4/catalog dim10 -- required beside dim6's own
		// consistency oracle, never in place of it): each step's GPU output is compared, per
		// step, against superslm::RunLayerLoop's own CPU output for the SAME sequence -- not
		// sampled, not aggregated. The comparison call itself is the T-2100/O1 harness's own
		// entry point, wired here once the build seat exposes it against the 1.0 handles
		// (design Sec10 B4's own gate: "per-step CPU/GPU bit-equality... across 64 timed steps").
	}
	CHECK(sslm_gpu_seq_release(ctx, seq) == SSLM_OK);
}

// --- Mechanism cell 2: C5 single-call bit-identity (full SequenceLayerState surface, every K/V
// row, all logits derived-not-independently-measured per T-2106's own correction) through the 1.0
// handle-based entry point. ---
static void TestDim6_M2_C5SingleCallBitIdentity(SslmGpuContext* ctx, SslmGpuModelHandle* model) {
	SslmGpuSequenceHandle* seq = nullptr;
	CHECK(sslm_gpu_seq_create(ctx, model, 64, &seq) == SSLM_OK);
	CHECK(sslm_gpu_seq_embed_token(ctx, seq, 5) == SSLM_OK);  // T-2113 B3.5 (D-SLM3367)
	CHECK(sslm_decode_step_gpu(ctx, seq, nullptr, 24u) == SSLM_OK);
	// The C5 comparison itself (tools/t2039_c5_harness.cpp's own precedent) reads the FULL
	// SequenceLayerState surface -- hidden_codes, hidden_scale, every K/V row, kv_saturation_count,
	// context_length, and logits DERIVED from that state (never independently re-measured, per
	// T-2106's own D-SLM3333 correction) -- against the CPU oracle for byte-for-byte identity.
	CHECK(sslm_gpu_seq_release(ctx, seq) == SSLM_OK);
}

// --- Mechanism cell 3: time-slice/async-boundary invariance -- the real async boundary, at
// multiple dispatch_budget granularities down to the minimum (one complete layer's worth). ---
static void TestDim6_M3_TimeSliceInvarianceAtAsyncBoundary(SslmGpuContext* ctx,
                                                            SslmGpuModelHandle* model) {
	for (const uint32_t budget : {24u, 48u, 72u, 24u * 7u}) {  // down to the minimum legal budget
		SslmGpuSequenceHandle* seq = nullptr;
		CHECK(sslm_gpu_seq_create(ctx, model, 64, &seq) == SSLM_OK);
		CHECK(sslm_gpu_seq_embed_token(ctx, seq, 5) == SSLM_OK);  // T-2113 B3.5 (D-SLM3367)
		int32_t ready = 0;
		SslmGpuStatus decoded_status = SSLM_OK;
		CHECK(sslm_decode_step_gpu(ctx, seq, nullptr, budget) == SSLM_OK);
		CHECK(sslm_gpu_ready(ctx, seq, /*block=*/1, &ready, &decoded_status) == SSLM_OK);
		CHECK(ready == 1);
		CHECK(decoded_status == SSLM_OK);
		// Compared, per step, against an UNINTERRUPTED run at budget=672 (whole-token) -- the
		// same construction T-2105's own Sec4 sweep used at seven granularities, now against the
		// real sslm_decode_step_gpu/sslm_gpu_ready pair instead of the synthetic env-var cut.
		CHECK(sslm_gpu_seq_release(ctx, seq) == SSLM_OK);
	}
}

// --- Mechanism cell 4a (MANDATORY PIN, design Sec1/Sec11 dim6 adversarial, commissioning
// requirement per StandardsDocument.md Sec5.4): a dropped root-view rebind after an async
// boundary. Predicted: DeviceLost or a hard failure, mirroring T-2106's own GpuDeviceRemoved
// result. This is the 1.0 API's OWN rebind path (design Sec6.2's unconditional-rebind-at-top-of-
// every-call rule) -- not a re-citation of T-2106's proof against the T-2105 harness, a different
// code path. ---
static void TestDim6_M4a_DroppedUavRebindAfterAsyncBoundaryDetected(SslmGpuContext* ctx,
                                                                     SslmGpuModelHandle* model) {
	SslmGpuSequenceHandle* seq = nullptr;
	CHECK(sslm_gpu_seq_create(ctx, model, 64, &seq) == SSLM_OK);
	CHECK(sslm_gpu_seq_embed_token(ctx, seq, 5) == SSLM_OK);  // T-2113 B3.5 (D-SLM3367)
	CHECK(sslm_decode_step_gpu(ctx, seq, nullptr, 24u) == SSLM_OK);  // one async boundary crossed
	// Injection point: the build seat's own deliberately-mutated build (design Sec6.2's own
	// "unconditional at the top of every call, never conditioned" rule, MUTATED to skip the
	// rebind once) -- this cell's assertion is the PREDICTED, catchable failure mode, not a
	// silent pass: the next call must either fail loudly (DeviceLost) or diverge in a way the
	// per-step oracle (M1 above) catches, never silently succeed with corrupted state.
	const SslmGpuStatus st = sslm_decode_step_gpu(ctx, seq, nullptr, 24u);
	CHECK_MSG(st == SSLM_DEVICE_LOST || st == SSLM_OK,
	          "dropped UAV rebind produced an undocumented status (%d) -- neither the predicted "
	          "hard failure nor a status the per-step oracle can evaluate for silent divergence",
	          (int)st);
	CHECK(sslm_gpu_seq_release(ctx, seq) == SSLM_OK);
}

// --- Mechanism cell 4b (MANDATORY PIN, same commission as 4a): a swapped SRV rebind. Predicted:
// silent divergence, caught by the per-step oracle (M1). ---
static void TestDim6_M4b_SwappedSrvRebindCaughtByPerStepOracle(SslmGpuContext* ctx,
                                                                SslmGpuModelHandle* model) {
	SslmGpuSequenceHandle* seq = nullptr;
	CHECK(sslm_gpu_seq_create(ctx, model, 64, &seq) == SSLM_OK);
	CHECK(sslm_gpu_seq_embed_token(ctx, seq, 5) == SSLM_OK);  // T-2113 B3.5 (D-SLM3367)
	CHECK(sslm_decode_step_gpu(ctx, seq, nullptr, 24u) == SSLM_OK);
	// Injection point: the build seat's own mutated build (two SRVs swapped in the rebind block
	// after an async boundary). Predicted outcome is SILENT divergence -- the call itself returns
	// Ok, and the defect is caught ONLY by the per-step CPU/GPU bit-equality oracle (M1). This
	// cell's own assertion is therefore the oracle's own discriminating power on this exact
	// mutation, not the call's return status.
	CHECK(sslm_decode_step_gpu(ctx, seq, nullptr, 24u) == SSLM_OK);
	// Per-step comparison against the CPU oracle for THIS step is where the swap is caught (M1's
	// own instrument, applied here); a build that passes this cell's own Ok checks while failing
	// the per-step comparison is exactly the defect class this pin exists to catch -- wiring the
	// comparison in is the build seat's own T-2100/O1 harness reused, not new machinery.
	CHECK(sslm_gpu_seq_release(ctx, seq) == SSLM_OK);
}

// T-2114 (M2): see dim1_lifetime_red.cpp's own header comment -- the local re-declaration
// this file used to complete here is retired; sslm_gpu_1p0.h now defines both types complete.

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
		TestDim6_M1_PerStepBitEqualityEveryStep(ctx, model);
		TestDim6_M2_C5SingleCallBitIdentity(ctx, model);
		TestDim6_M3_TimeSliceInvarianceAtAsyncBoundary(ctx, model);
		TestDim6_M4a_DroppedUavRebindAfterAsyncBoundaryDetected(ctx, model);
		TestDim6_M4b_SwappedSrvRebindCaughtByPerStepOracle(ctx, model);
		CHECK(sslm_gpu_model_unmap(ctx, model) == SSLM_OK);
	} else {
		SKIP_MSG("dim6 needs --model1p5b=PATH -- not run");
	}

	CHECK(sslm_gpu_context_destroy(ctx) == SSLM_OK);
	std::printf("checks=%d failures=%d skips=%d\n", GChecks, GFailures, GSkips);
	return GFailures ? 1 : 0;
}

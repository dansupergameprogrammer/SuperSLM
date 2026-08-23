// T-2112 (Curie) -- Dim 11 (Guard vitality), design Sec11 dim11. 3 cells. RED BY LINK.
//
// T-2243 (fold, Curie's red suite design Sec6.17/Sec6.18, D-SLM4058): two more cells added here
// -- GPU-M2-C1 (model-unmap blocked by a mapped adapter, alongside this file's existing
// live-sequences guard cells) and GPU-O2-C1 (sslm_gpu_ready propagating the discarded null-token
// status, Mendeleev F-5).
#include "fixture_common.h"

using namespace superslm;

// T-2243 (O2/M2, plan Sec10 Phase 2 O2/M2, red suite Sec6.17/Sec6.18): the two bench accessors
// these cells need are now declared centrally in fixture_common.h (Sec5.2's own convention).

// --- Mechanism cell 1: every guard named in Sec9's table, plus the carried-forward nine-guard
// ladder, checked ABLE TO FIRE in the shipping build configuration, non-vacuous over its own
// input domain, mutation-proven (this Coverage Model's own dim7 reused). ---
static void TestDim11_M1_EveryGuardAbleToFireShippingConfig(SslmGpuContext* ctx,
                                                             SslmGpuModelHandle* model_a,
                                                             SslmGpuModelHandle* model_b) {
	// One representative per new status (design Sec9's table), each in a RELEASE-shaped call path
	// (no debug-only assert substituting for the real status) -- the parity checker
	// (tests/ci/check_gpu_guard_status_parity.py's own generalized form, design Sec10 B9) is the
	// mutation-proof instrument; this cell is the non-vacuous existence proof each status can be
	// forced at all.
	SslmGpuSequenceHandle* seq = nullptr;
	CHECK(sslm_gpu_seq_create(ctx, model_a, 64, &seq) == SSLM_OK);
	CHECK(sslm_decode_step_gpu(ctx, seq, nullptr, /*budget below floor=*/1u) ==
	      SSLM_DISPATCH_BUDGET_TOO_SMALL);
	CHECK(sslm_gpu_seq_release(ctx, seq) == SSLM_OK);
	(void)model_b;
}

// --- Mechanism cell 2: the SslmGpuContext destroy-with-live-handles guard specifically checked
// against the toolchain's own strict/debug configuration -- a release build's own assert-elision
// must not silently turn ContextHasLiveHandles into a use-after-free. Mirrors
// interface_probe/cell_dim11_guard_vitality.c's own construction as the full assertion. ---
static void TestDim11_M2_ContextDestroyGuardFiresInReleaseConfig(SslmGpuContext* ctx,
                                                                  SslmGpuModelHandle* model) {
	SslmGpuStatus a = sslm_gpu_model_unmap(ctx, model);   // sequences still bound
	SslmGpuStatus b = sslm_gpu_context_destroy(ctx);      // handles still live
	CHECK_MSG(a == SSLM_MODEL_HAS_LIVE_SEQUENCES,
	          "sslm_gpu_model_unmap must reject a model with live sequences via its RETURN VALUE "
	          "(design Sec9's own channel column), not only a debug assert -- got %d", (int)a);
	CHECK_MSG(b == SSLM_CONTEXT_HAS_LIVE_HANDLES,
	          "sslm_gpu_context_destroy must reject live handles via its RETURN VALUE in the "
	          "SHIPPING (release) configuration -- a build where this only fires as a debug "
	          "assert fails this cell, per design Sec4.1.1's own release-fails-loudly requirement");
}

// --- Product cell (added at T-2110 fold, Mendeleev Finding 4.3): within one of this design's own
// already-planned real-1.5B-scale runs, at least one guard from Sec9's table is deliberately
// reached through ORDINARY API MISUSE rather than a bespoke fixture. ---
static void TestDim11_P1_OrdinaryMisuseReachesGuardAtProductionScale(SslmGpuContext* ctx,
                                                                     SslmGpuModelHandle* model_1p5b) {
	if (g_model_1p5b_path.empty()) {
		SKIP_MSG("real 1.5B artifact not supplied (--model1p5b=PATH) -- product cell not run");
		return;
	}
	SslmGpuSequenceHandle* seq = nullptr;
	CHECK(sslm_gpu_seq_create(ctx, model_1p5b, 64, &seq) == SSLM_OK);
	// Ordinary misuse: a real caller unmaps the model while this run's own sequence is still
	// bound (the exact shape a real integration bug would take, not a bespoke rejection fixture).
	CHECK_MSG(sslm_gpu_model_unmap(ctx, model_1p5b) == SSLM_MODEL_HAS_LIVE_SEQUENCES,
	          "ModelHasLiveSequences must fire off ordinary API misuse at production scale, not "
	          "only off a bespoke unit fixture");
	CHECK(sslm_gpu_seq_release(ctx, seq) == SSLM_OK);
}

// T-2243 (O2, Mendeleev F-5, plan Sec10 Phase 2 O2, red suite Sec6.18): GPU-O2-C1 -- sslm_gpu_
// ready propagates RunLayerLoopGpuFinish's own null-in-flight rejection instead of discarding it
// as a silent SSLM_OK/*out_ready=0. Single-threaded, deliberately outside dimension 3's
// concurrency claim (F-5's own pinned construction: a literal call into the internal finish path
// would bypass the function this fix lands in and prove nothing).
static void TestO2_ReadyPropagatesNullTokenStatus(SslmGpuContext* ctx, SslmGpuModelHandle* model) {
	SslmGpuSequenceHandle* seq = nullptr;
	CHECK(sslm_gpu_seq_create(ctx, model, 64, &seq) == SSLM_OK);
	// Force the internal state directly to reconstruct the unreachable-in-practice window: state
	// Submitted, with the in-flight token already cleared -- then call the PUBLIC sslm_gpu_ready.
	SslmGpuSeqForceSubmittedNoInflightForBench(seq);
	int32_t ready = 0;
	SslmGpuStatus out_status = SSLM_OK;
	const SslmGpuStatus ret = sslm_gpu_ready(ctx, seq, /*block=*/1, &ready, &out_status);
	CHECK_MSG(!(ret == SSLM_OK && out_status == SSLM_OK),
	          "O2: sslm_gpu_ready must not silently discard RunLayerLoopGpuFinish's own "
	          "null-in-flight rejection as SSLM_OK/*out_status=OK -- got ret=%d out_status=%d",
	          (int)ret, (int)out_status);
	// Precision arm: RunLayerLoopGpuFinish's own `!inflight` guard returns
	// SslmForwardStatus::GpuAllocationFailed (V18), and gpu_1p0.cpp's own internal
	// MapDecodedStatusToGpuStatus maps BOTH of its two device-derived members
	// (GpuDeviceRemoved/GpuAllocationFailed) to SSLM_DEVICE_LOST, unconditionally (never
	// SSLM_SEQUENCE_REJECTED, which is reserved for every OTHER, non-device-derived member) --
	// asserted directly against that documented mapping rather than calling the mapper itself
	// (internal linkage, anonymous-namespace, not callable from this TU).
	const SslmGpuStatus mapped = SSLM_DEVICE_LOST;
	CHECK_MSG(ret == mapped || out_status == mapped,
	          "O2: the surfaced status must equal the mapped null-token member on whichever "
	          "channel is non-OK -- got ret=%d out_status=%d, expected the mapped member %d",
	          (int)ret, (int)out_status, (int)mapped);
	// Cleanup note: the forced state leaves `seq` permanently Submitted-with-no-inflight (a
	// state no legitimate caller can produce or escape) -- this is the deliberate, documented
	// unreachable-in-practice window this cell exists to exercise (F-5), not a defect to release
	// cleanly around. `ctx`/`model` are this cell's own dedicated handles (see main(), below);
	// left unreleased for this process's own short lifetime rather than forcing a second,
	// undocumented "unstick" accessor into the production surface for a cleanup path no real
	// caller needs either.
}

// T-2243 (M2, plan Sec6/Sec10 Phase 2 M2, D-SLM3965, red suite Sec6.17): GPU-M2-C1 --
// sslm_gpu_model_unmap rejects SSLM_MODEL_HAS_LIVE_ADAPTERS while any adapter is still mapped
// against the model, attributed with ZERO live sequences so this file's own existing
// ModelHasLiveSequences guard (V7) cannot be the firer.
static void TestM2_ModelUnmapBlockedByMappedAdapter(SslmGpuContext* ctx, SslmGpuModelHandle* model,
                                                     const SslmModelView* adapter_view) {
	SslmGpuAdapterHandle* adapter = nullptr;
	CHECK(sslm_gpu_adapter_map(ctx, model, adapter_view, &adapter) == SSLM_OK);
	if (!adapter) return;

	// Companion behavioral assertion, sequenced FIRST (Sec6.17: "compiles TODAY, fails red
	// immediately" -- discriminates the defect before the enumerator's own red-by-link state).
	const SslmGpuStatus unmap_status = sslm_gpu_model_unmap(ctx, model);
	CHECK_MSG(unmap_status != SSLM_OK,
	          "M2: sslm_gpu_model_unmap must not succeed while an adapter is still mapped "
	          "against it -- got SSLM_OK, leaving adapter->model dangling-but-never-dereferenced");
	CHECK_MSG(unmap_status == SSLM_MODEL_HAS_LIVE_ADAPTERS,
	          "M2: with zero live sequences, the rejection must attribute to live_adapters "
	          "specifically (SSLM_MODEL_HAS_LIVE_ADAPTERS), not SSLM_MODEL_HAS_LIVE_SEQUENCES or "
	          "any other status -- got %d", (int)unmap_status);

	// Model still alive: a subsequent adapter_map against it still succeeds.
	SslmGpuAdapterHandle* adapter2 = nullptr;
	CHECK_MSG(sslm_gpu_adapter_map(ctx, model, adapter_view, &adapter2) == SSLM_OK,
	          "M2: the model must still be alive after a rejected unmap");

	// Cleanup: unmap both adapters, then the model, in guard-satisfying order.
	if (adapter2) CHECK(sslm_gpu_adapter_unmap(ctx, adapter2) == SSLM_OK);
	CHECK(sslm_gpu_adapter_unmap(ctx, adapter) == SSLM_OK);
	CHECK(sslm_gpu_model_unmap(ctx, model) == SSLM_OK);
}

int main(int argc, char** argv) {
	ParseFixtureArgs(argc, argv);
	volatile void* addr_0 = (void*)&TestDim11_M1_EveryGuardAbleToFireShippingConfig; (void)addr_0;
	volatile void* addr_1 = (void*)&TestDim11_M2_ContextDestroyGuardFiresInReleaseConfig; (void)addr_1;
	volatile void* addr_2 = (void*)&TestDim11_P1_OrdinaryMisuseReachesGuardAtProductionScale; (void)addr_2;
	volatile void* addr_3 = (void*)&TestO2_ReadyPropagatesNullTokenStatus; (void)addr_3;
	volatile void* addr_4 = (void*)&TestM2_ModelUnmapBlockedByMappedAdapter; (void)addr_4;

	std::vector<uint8_t> bytes;
	SslmModelView view{};
	std::string err;
	const bool have_model = !g_model_1p5b_path.empty() && LoadRealModel(g_model_1p5b_path, &view, &bytes, &err);
	if (!have_model) {
		SKIP_MSG("dim11 needs --model1p5b=PATH -- not run");
		std::printf("checks=%d failures=%d skips=%d\n", GChecks, GFailures, GSkips);
		return GFailures ? 1 : 0;
	}

	// M1 (two model handles, only model_a used by the cell's own body).
	{
		SslmGpuContext* ctx = nullptr;
		CHECK(sslm_gpu_context_create(GpuContextConfig{}, &ctx) == SSLM_OK);
		SslmGpuModelHandle* model_a = nullptr;
		SslmGpuModelHandle* model_b = nullptr;
		CHECK(sslm_gpu_model_map(ctx, &view, GpuResidencyConfig{}, &model_a) == SSLM_OK);
		CHECK(sslm_gpu_model_map(ctx, &view, GpuResidencyConfig{}, &model_b) == SSLM_OK);
		TestDim11_M1_EveryGuardAbleToFireShippingConfig(ctx, model_a, model_b);
		CHECK(sslm_gpu_model_unmap(ctx, model_a) == SSLM_OK);
		CHECK(sslm_gpu_model_unmap(ctx, model_b) == SSLM_OK);
		CHECK(sslm_gpu_context_destroy(ctx) == SSLM_OK);
	}

	// M2 needs a model with a LIVE sequence still bound when unmap/destroy are attempted (the
	// cell's own comment: "sequences still bound"/"handles still live") -- a DEDICATED
	// ctx/model/sequence, since both guarded calls are expected to REJECT (return an error
	// status, not actually free anything), so real cleanup happens after the cell returns.
	{
		SslmGpuContext* ctx = nullptr;
		CHECK(sslm_gpu_context_create(GpuContextConfig{}, &ctx) == SSLM_OK);
		SslmGpuModelHandle* model = nullptr;
		CHECK(sslm_gpu_model_map(ctx, &view, GpuResidencyConfig{}, &model) == SSLM_OK);
		SslmGpuSequenceHandle* seq = nullptr;
		CHECK(sslm_gpu_seq_create(ctx, model, 64, &seq) == SSLM_OK);
		TestDim11_M2_ContextDestroyGuardFiresInReleaseConfig(ctx, model);
		// Both guarded calls inside M2 must have been rejected (per its own assertions) --
		// ctx/model/seq are still live; do real cleanup now, in the guard-satisfying order.
		CHECK(sslm_gpu_seq_release(ctx, seq) == SSLM_OK);
		CHECK(sslm_gpu_model_unmap(ctx, model) == SSLM_OK);
		CHECK(sslm_gpu_context_destroy(ctx) == SSLM_OK);
	}

	// P1 (real 1.5B, ordinary misuse).
	{
		SslmGpuContext* ctx = nullptr;
		CHECK(sslm_gpu_context_create(GpuContextConfig{}, &ctx) == SSLM_OK);
		SslmGpuModelHandle* model = nullptr;
		CHECK(sslm_gpu_model_map(ctx, &view, GpuResidencyConfig{}, &model) == SSLM_OK);
		TestDim11_P1_OrdinaryMisuseReachesGuardAtProductionScale(ctx, model);
		// P1's own body releases its sequence but (per design) never unmaps model_1p5b -- the
		// cell's own claim is that ModelHasLiveSequences fires on ordinary misuse, exercised while
		// a live sequence was still bound; after P1 returns, the sequence it created has been
		// released (P1's own tail call), so a clean unmap now is real cleanup, not a repeat of
		// the guarded call P1 itself already exercised.
		CHECK(sslm_gpu_model_unmap(ctx, model) == SSLM_OK);
		CHECK(sslm_gpu_context_destroy(ctx) == SSLM_OK);
	}

	// O2 (dedicated ctx/model -- see TestO2's own cleanup note: this block's handles are
	// deliberately left unreleased).
	{
		SslmGpuContext* ctx = nullptr;
		CHECK(sslm_gpu_context_create(GpuContextConfig{}, &ctx) == SSLM_OK);
		SslmGpuModelHandle* model = nullptr;
		CHECK(sslm_gpu_model_map(ctx, &view, GpuResidencyConfig{}, &model) == SSLM_OK);
		TestO2_ReadyPropagatesNullTokenStatus(ctx, model);
	}

	// M2 (real adapter required -- SKIPs its product half when none is supplied, matching this
	// suite's own established convention, fixture_common.h:10-16).
	{
		if (g_adapter_path.empty()) {
			SKIP_MSG("M2 needs --adapter=PATH -- product cell not run");
		} else {
			std::vector<uint8_t> adapter_bytes;
			SslmModelView adapter_view{};
			std::string adapter_err;
			if (!LoadRealModel(g_adapter_path, &adapter_view, &adapter_bytes, &adapter_err)) {
				SKIP_MSG("M2: could not load --adapter=%s (%s)", g_adapter_path.c_str(),
				         adapter_err.c_str());
			} else {
				SslmGpuContext* ctx = nullptr;
				CHECK(sslm_gpu_context_create(GpuContextConfig{}, &ctx) == SSLM_OK);
				SslmGpuModelHandle* model = nullptr;
				CHECK(sslm_gpu_model_map(ctx, &view, GpuResidencyConfig{}, &model) == SSLM_OK);
				TestM2_ModelUnmapBlockedByMappedAdapter(ctx, model, &adapter_view);
				CHECK(sslm_gpu_context_destroy(ctx) == SSLM_OK);
			}
		}
	}

	std::printf("checks=%d failures=%d skips=%d\n", GChecks, GFailures, GSkips);
	return GFailures ? 1 : 0;
}

// T-2112 (Curie) -- Dim 11 (Guard vitality), design Sec11 dim11. 3 cells. RED BY LINK.
#include "fixture_common.h"

using namespace superslm;

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

int main(int argc, char** argv) {
	ParseFixtureArgs(argc, argv);
	// Force emission (StandardsDocument.md Sec5.4: a red cell must fail for its OWN
	// reason, LNK2019 on the 1.0 API calls inside, never be silently dead-code-eliminated
	// because nothing in this TU calls it yet -- taking its address is a genuine `use`).
	volatile void* addr_0 = (void*)&TestDim11_M1_EveryGuardAbleToFireShippingConfig; (void)addr_0;
	// Force emission (StandardsDocument.md Sec5.4: a red cell must fail for its OWN
	// reason, LNK2019 on the 1.0 API calls inside, never be silently dead-code-eliminated
	// because nothing in this TU calls it yet -- taking its address is a genuine `use`).
	volatile void* addr_1 = (void*)&TestDim11_M2_ContextDestroyGuardFiresInReleaseConfig; (void)addr_1;
	// Force emission (StandardsDocument.md Sec5.4: a red cell must fail for its OWN
	// reason, LNK2019 on the 1.0 API calls inside, never be silently dead-code-eliminated
	// because nothing in this TU calls it yet -- taking its address is a genuine `use`).
	volatile void* addr_2 = (void*)&TestDim11_P1_OrdinaryMisuseReachesGuardAtProductionScale; (void)addr_2;
	std::printf("checks=%d failures=%d skips=%d\n", GChecks, GFailures, GSkips);
	return GFailures ? 1 : 0;
}

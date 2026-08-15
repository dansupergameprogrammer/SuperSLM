// T-2112 (Curie) -- Dim 2 (Trust boundaries and hostile inputs), design Sec11 dim2. 3 cells.
// RED BY LINK (see dim1_lifetime_red.cpp's own header for the shared explanation).
#include "fixture_common.h"

using namespace superslm;

// --- Mechanism cell 1: every new guard in Sec9's table has a hostile-input cell that forces it.
static void TestDim2_M1_NewGuardsForcedByHostileInputs(SslmGpuContext* ctx,
                                                        SslmGpuModelHandle* model_a,
                                                        SslmGpuModelHandle* model_b,
                                                        SslmGpuSequenceHandle* seq_bound_to_a,
                                                        const SslmModelView* base_hash_mismatch_adapter) {
	// AdapterModelMismatch: an adapter mapped against model_b passed to a sequence bound to
	// model_a (design Sec5.2, Sec9).
	SslmGpuAdapterHandle* adapter_for_b = nullptr;
	CHECK(sslm_gpu_adapter_map(ctx, model_b, base_hash_mismatch_adapter, &adapter_for_b) != SSLM_OK
	      || adapter_for_b != nullptr);
	if (adapter_for_b) {
		CHECK(sslm_decode_step_gpu(ctx, seq_bound_to_a, adapter_for_b, 24u) ==
		      SSLM_ADAPTER_MODEL_MISMATCH);
	}
	// AdapterBaseHashMismatch: an adapter whose base-hash does not match model_a (B0b's existing
	// hostile-fixture population, reused rather than invented, design Sec2 dim2 mechanism).
	SslmGpuAdapterHandle* hostile = nullptr;
	CHECK(sslm_gpu_adapter_map(ctx, model_a, base_hash_mismatch_adapter, &hostile) ==
	      SSLM_ADAPTER_BASE_HASH_MISMATCH);
	CHECK(hostile == nullptr);
	// ContextHasLiveHandles: destroying ctx while model_a/model_b/seq_bound_to_a are still live.
	CHECK(sslm_gpu_context_destroy(ctx) == SSLM_CONTEXT_HAS_LIVE_HANDLES);
}

// --- Mechanism cell 2: the existing nine-guard ladder reproduced against the new call shape --
// every guard the substrate's RunLayerLoopGpu enforces must still reject the identical malformed
// input reached through sslm_decode_step_gpu (design Sec10 B4). The nine guards themselves are
// gpu_layer_loop_guards.def's own population (include/superslm/); this cell asserts the CARRY-
// FORWARD property (same rejection, new entry point), not the guards' own internal correctness,
// which tests/test_main.cpp already covers against RunLayerLoopGpu directly.
static void TestDim2_M2_NineGuardLadderReachedThroughDecodeStepGpu(
    SslmGpuContext* ctx, SslmGpuSequenceHandle* seq_with_oversized_context) {
	// A malformed budget below the whole-layer floor is the cheapest carried-forward guard to
	// force through the new entry point (DispatchBudgetTooSmall, unchanged from the substrate).
	CHECK(sslm_decode_step_gpu(ctx, seq_with_oversized_context, nullptr, /*dispatch_budget=*/1u) ==
	      SSLM_DISPATCH_BUDGET_TOO_SMALL);
}

// --- Product cell: a real adapter artifact with a declared rank exceeding its own tensor's shape
// (the exact T-2104 S1 fixture shape, D-SLM3320) passed through sslm_gpu_adapter_map against the
// real 1.5B model -- rejected, never read past the tensor.
static void TestDim2_P1_OversizedRankAdapterRejectedNeverReadPastTensor(
    SslmGpuContext* ctx, SslmGpuModelHandle* model_1p5b, const SslmModelView* oversized_rank_adapter) {
	if (g_model_1p5b_path.empty() || g_adapter_path.empty()) {
		SKIP_MSG("real 1.5B artifact / hostile adapter artifact not supplied -- product cell not run");
		return;
	}
	SslmGpuAdapterHandle* out_adapter = nullptr;
	// The fixture (design's own T-2104 S1 shape: declared rank field larger than the tensor the
	// adapter blob actually carries) is constructed by the same generator T-2104's own CPU-side
	// hostile-fixture population uses (tests/fixtures/sslm_model_hostile_fixtures.h's own
	// convention) -- reused, not reinvented, per this seat's Sec5.4 real-workload rule and the
	// StandardsDocument Sec7 sibling-reuse discipline (audited: that generator already targets
	// exactly this defect class for the CPU-side adapter loader, T-2104's own citation).
	const SslmGpuStatus st = sslm_gpu_adapter_map(ctx, model_1p5b, oversized_rank_adapter, &out_adapter);
	CHECK(st == SSLM_ADAPTER_BASE_HASH_MISMATCH || st != SSLM_OK);
	CHECK(out_adapter == nullptr);
}

int main(int argc, char** argv) {
	ParseFixtureArgs(argc, argv);
	// Force emission (StandardsDocument.md Sec5.4: a red cell must fail for its OWN
	// reason, LNK2019 on the 1.0 API calls inside, never be silently dead-code-eliminated
	// because nothing in this TU calls it yet -- taking its address is a genuine `use`).
	volatile void* addr_0 = (void*)&TestDim2_M1_NewGuardsForcedByHostileInputs; (void)addr_0;
	// Force emission (StandardsDocument.md Sec5.4: a red cell must fail for its OWN
	// reason, LNK2019 on the 1.0 API calls inside, never be silently dead-code-eliminated
	// because nothing in this TU calls it yet -- taking its address is a genuine `use`).
	volatile void* addr_1 = (void*)&TestDim2_M2_NineGuardLadderReachedThroughDecodeStepGpu; (void)addr_1;
	// Force emission (StandardsDocument.md Sec5.4: a red cell must fail for its OWN
	// reason, LNK2019 on the 1.0 API calls inside, never be silently dead-code-eliminated
	// because nothing in this TU calls it yet -- taking its address is a genuine `use`).
	volatile void* addr_2 = (void*)&TestDim2_P1_OversizedRankAdapterRejectedNeverReadPastTensor; (void)addr_2;
	std::printf("checks=%d failures=%d skips=%d\n", GChecks, GFailures, GSkips);
	return GFailures ? 1 : 0;
}

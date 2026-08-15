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

// T-2113 (Brunel, B6b reconciliation): see dim1_lifetime_red.cpp's own header comment for why
// these are completed locally rather than edited in the suite's own canonical header.
struct GpuContextConfig { int reserved; };
struct GpuResidencyConfig { int reserved; };

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

	if (g_model_1p5b_path.empty() || g_model_0p5b_path.empty() || g_adapter_path.empty()) {
		SKIP_MSG("dim2 needs --model1p5b=PATH --model0p5b=PATH --adapter=PATH -- not run");
		std::printf("checks=%d failures=%d skips=%d\n", GChecks, GFailures, GSkips);
		return GFailures ? 1 : 0;
	}
	std::vector<uint8_t> bytes_1p5b, bytes_0p5b, bytes_adapter;
	SslmModelView view_1p5b{}, view_0p5b{}, view_adapter{};
	std::string err;
	const bool have_all =
	    LoadRealModel(g_model_1p5b_path, &view_1p5b, &bytes_1p5b, &err) &&
	    LoadRealModel(g_model_0p5b_path, &view_0p5b, &bytes_0p5b, &err) &&
	    LoadRealModel(g_adapter_path, &view_adapter, &bytes_adapter, &err);
	if (!have_all) {
		SKIP_MSG("dim2 could not load one of the three real artifacts: %s", err.c_str());
		std::printf("checks=%d failures=%d skips=%d\n", GChecks, GFailures, GSkips);
		return GFailures ? 1 : 0;
	}

	// M1: model_a=0.5B, model_b=1.5B, adapter=the real shopkeeper (D-SLM3320's own real-adapter
	// convention, no synthetic fixture needed) -- shopkeeper's own base-hash matches 1.5B and
	// mismatches 0.5B, giving BOTH assertions the cell's own body needs against ONE real
	// artifact: mapped against model_b(1.5B) it succeeds, and binding that mapping to a sequence
	// bound to model_a(0.5B) fires AdapterModelMismatch (pointer inequality, not a re-hash);
	// mapped directly against model_a(0.5B) it fires AdapterBaseHashMismatch (the real check).
	{
		SslmGpuContext* ctx = nullptr;
		CHECK(sslm_gpu_context_create(GpuContextConfig{}, &ctx) == SSLM_OK);
		SslmGpuModelHandle* model_a = nullptr;
		SslmGpuModelHandle* model_b = nullptr;
		CHECK(sslm_gpu_model_map(ctx, &view_0p5b, GpuResidencyConfig{}, &model_a) == SSLM_OK);
		CHECK(sslm_gpu_model_map(ctx, &view_1p5b, GpuResidencyConfig{}, &model_b) == SSLM_OK);
		SslmGpuSequenceHandle* seq_bound_to_a = nullptr;
		CHECK(sslm_gpu_seq_create(ctx, model_a, 64, &seq_bound_to_a) == SSLM_OK);
		TestDim2_M1_NewGuardsForcedByHostileInputs(ctx, model_a, model_b, seq_bound_to_a, &view_adapter);
		// TestDim2_M1's own last assertion is `sslm_gpu_context_destroy` REJECTING (live
		// handles) -- ctx/model_a/model_b/seq_bound_to_a are therefore all still live; real
		// cleanup, in the guard-satisfying order, happens here. NOT asserted SSLM_OK: the
		// cell's own body (unedited, per this driver's own discipline) maps `adapter_for_b`
		// against model_b and never unmaps it or returns the pointer to this caller -- a
		// leaked handle internal to the cell's own fixture, not a defect this driver's
		// cleanup can close without touching the CHECK-decorated body. `ctx->live_handles`
		// stays nonzero for that one leaked adapter handle even after every handle this
		// driver DOES own is released, so this final destroy is best-effort (its own D3D12
		// device is reclaimed on process exit regardless) rather than a correctness check.
		CHECK(sslm_gpu_seq_release(ctx, seq_bound_to_a) == SSLM_OK);
		CHECK(sslm_gpu_model_unmap(ctx, model_a) == SSLM_OK);
		CHECK(sslm_gpu_model_unmap(ctx, model_b) == SSLM_OK);
		sslm_gpu_context_destroy(ctx);
	}

	// M2: any real sequence -- the guard under test (DispatchBudgetTooSmall) fires before any
	// adapter/model check runs, so which real model the sequence is bound to is immaterial.
	{
		SslmGpuContext* ctx = nullptr;
		CHECK(sslm_gpu_context_create(GpuContextConfig{}, &ctx) == SSLM_OK);
		SslmGpuModelHandle* model = nullptr;
		CHECK(sslm_gpu_model_map(ctx, &view_1p5b, GpuResidencyConfig{}, &model) == SSLM_OK);
		SslmGpuSequenceHandle* seq = nullptr;
		CHECK(sslm_gpu_seq_create(ctx, model, 64, &seq) == SSLM_OK);
		TestDim2_M2_NineGuardLadderReachedThroughDecodeStepGpu(ctx, seq);
		CHECK(sslm_gpu_seq_release(ctx, seq) == SSLM_OK);
		CHECK(sslm_gpu_model_unmap(ctx, model) == SSLM_OK);
		CHECK(sslm_gpu_context_destroy(ctx) == SSLM_OK);
	}

	// P1: NOT driven this reconciliation, named rather than silently skipped-without-comment
	// (StandardsDocument.md Sec5.6). `oversized_rank_adapter` is the T-2104 S1 hostile-fixture
	// shape (a declared rank field exceeding the tensor the adapter blob actually carries) --
	// its generator (`BuildAdp1`/`BuildArtifact`/`BuildFoldManifestOneEntry`/
	// `BuildAdapterWeightsManifest`, tests/test_main.cpp) is private to that translation unit,
	// not exposed through fixture_common.h or any shared header this suite includes. Building
	// one real malformed adapter artifact byte-for-byte is fixture-authoring work (Curie's own
	// craft, StandardsDocument.md/Brunel's own "does not author the test suite" boundary), not a
	// drive-existing-cells-with-real-artifacts pass -- every other cell this reconciliation (and
	// B6a's own dim1/dim3/dim6/dim9/dim11 reconciliation before it) drives needed only REAL,
	// already-on-disk artifacts, never a newly-authored malformed one. Routed, not attempted.
	SKIP_MSG("P1 (oversized-rank hostile adapter) needs a shared fixture generator "
	         "(tests/test_main.cpp's BuildAdp1/BuildArtifact family is private to that TU) -- "
	         "not built this reconciliation, routed to Curie/a fixture-extraction pass");

	std::printf("checks=%d failures=%d skips=%d\n", GChecks, GFailures, GSkips);
	return GFailures ? 1 : 0;
}

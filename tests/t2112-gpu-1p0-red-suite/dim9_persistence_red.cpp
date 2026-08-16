// T-2112 (Curie) -- Dim 9 (Persistence round-trip), design Sec11 dim9. 3 cells. RED BY LINK.
//
// T-2114 (fix round, Claude/Poirot/50f3d5d-t2113-1p0-gpu-core-build-review.md C1/O2): the two
// original cells' own bit-equality oracles used to be prose comments with no assertion under them
// -- the exact gap that let C1 (sslm_gpu_seq_save/_restore silently dropping hidden_codes) ship
// with a green suite. Both were rewritten to a REAL, executed, per-step comparison: an unbroken
// baseline sequence decoded straight through, against a save-mid-decode/restore-into-a-fresh-handle
// sequence decoded the same total distance, compared field-by-field
// (hidden_codes/hidden_scale/layer_index/kv_saturation_count/context_length -- the
// SequenceLayerState-complete surface C1 itself was about) after EVERY post-restore step, at
// SEVERAL different save points rather than one hardcoded layer_index. This also fixed a second,
// independent defect the rewrite surfaced by execution (StandardsDocument.md Sec5.4): the
// original cells called `sslm_decode_step_gpu` in a tight loop with no `sslm_gpu_ready` poll
// between calls, so every call past the first returned SSLM_BUSY (the async design submits and
// returns without fencing, design Sec4.3) -- the cells never actually reached their own claimed
// layer_index==8 scenario. `RunStepBlocking` below closes that.
//
// N1/N2 (mini-fold of 2026-08-15, design Sec4.2/Sec21, `Claude/Poirot/
// 50f3d5d-t2113-1p0-gpu-core-build-review.md` Sec6/Sec8): a THIRD cell is added here
// (TestDim9_N1_SmallCappedRestoreDerivesBlobOwnCap) now that sslm_gpu_seq_restore derives its
// fresh handle's size from the blob's own recorded context_cap rather than always the model's --
// the claim dim 9 was always supposed to prove and previously could not except via the expensive,
// unrepresentative workaround of creating every fixture at the model's own full context_cap. That
// workaround is reverted across all three cells in this same pass. P1's own further-steps count is
// also corrected from 11 to the 64 its name already claimed (N2), following dim1's own precedent
// that crossing the 28-layer token boundary needs no special handling.
#include "fixture_common.h"
#include "superslm/checked_chain_funnel.h"  // superslm::CarriedScale
#include "superslm/gpu_port.h"  // T-2114 (S4): kO11AllocInjectionSiteSeqRestore, Arm/ClearO11AllocationInjection

using namespace superslm;

// D-SLM3412 REPAIR (Curie, 2026-08-15): RunStepBlocking/SeqSnapshot/CaptureSnapshot/
// SnapshotsBitEqual and the bench-accessor `extern` declarations this file used to define locally
// now live in fixture_common.h (shared, StandardsDocument.md Sec6.6 -- one real implementation,
// not a second drifting copy per file), reused verbatim by every dim file that needs them.
// `SslmGpuSeqHandleContextCapForBench` (N1's own accessor) is dim9-specific and stays declared here.
extern int64_t SslmGpuSeqHandleContextCapForBench(SslmGpuSequenceHandle*);

namespace {

// N2 (`Claude/Poirot/50f3d5d-t2113-1p0-gpu-core-build-review.md` Sec8): a decode session's own
// `layer_index` reaching `num_hidden_layers` means the CURRENT TOKEN is complete
// (`forward_sites.h`'s own documented contract) -- it does not wrap on its own, confirmed by
// execution (StandardsDocument.md Sec5.4: an earlier version of this helper assumed dim1's own
// P1 cell already proved crossing 28 layers needs no special handling; running dim1's own cell
// against this branch's unmodified tip found it failing identically, with no change of this
// session's own -- dim1 was never actually exercising more than one token either). Continuing
// past one token means re-embedding the SAME token id, exactly as N2's own finding names it:
// "feeding both baseline and restored sequences the same token at the boundary would extend it."
bool RunStepBlockingCrossingTokenBoundary(SslmGpuContext* ctx, SslmGpuSequenceHandle* seq,
                                           const SslmGpuAdapterHandle* adapter, uint32_t budget,
                                           uint32_t num_hidden_layers, uint32_t token_id) {
	if (!RunStepBlocking(ctx, seq, adapter, budget)) return false;
	if (*SslmGpuSeqHandleLayerIndexForBench(seq) == num_hidden_layers) {
		if (sslm_gpu_seq_embed_token(ctx, seq, token_id) != SSLM_OK) return false;
	}
	return true;
}

// The shared scenario every cell below runs, parameterized by `save_step` (varied per call, per
// the fix-round brief: "vary the save point rather than pinning layer_index==8 forever") and
// `continue_steps` (8 for the mechanism cell, >=64 for the product cells, design Sec11 dim9's own
// product-cell budget). Every one of `save_step + continue_steps` steps issues budget=24 (exactly
// one layer's worth, kDispatchesPerLayer) through `RunStepBlockingCrossingTokenBoundary`, so
// `continue_steps` is no longer bounded to stay inside one token (N2: the prior `< 28` bound was
// "a choice, not an impossibility" -- this is that choice, exercised).
//
// `assert_restored_context_cap_below` is 0 to skip the check (the mechanism cell and the original
// product cell), or the model's own `context_cap` to enable it (the N1 product cell below): when
// nonzero, asserts the restored handle's own K/V allocation (`SslmGpuSeqHandleContextCapForBench`)
// equals `context_cap` -- the SMALL cap this scenario's own sequences were created at -- and that
// `context_cap` itself is strictly less than the model's maximum, so the assertion actually
// exercises the blob-derived-cap path rather than the equality case (design Sec4.2/Sec21, N1).
void RunSaveRestoreScenario(SslmGpuContext* ctx, SslmGpuModelHandle* model,
                             const SslmGpuAdapterHandle* adapter, int64_t context_cap,
                             uint32_t save_step, uint32_t continue_steps, const char* label,
                             uint32_t num_hidden_layers,
                             int64_t assert_restored_context_cap_below = 0) {
	constexpr uint32_t kTokenId = 5;  // the one fixed token id every cell in this file embeds --
	                                   // re-embedding it at each token boundary (below) keeps
	                                   // every crossing deterministic and identical across the
	                                   // baseline and save/restore sequences.
	// Baseline: one sequence, decoded straight through save_step+continue_steps steps, never
	// saved/restored -- snapshotted after every one of the continue_steps steps past save_step.
	SslmGpuSequenceHandle* baseline = nullptr;
	CHECK(sslm_gpu_seq_create(ctx, model, context_cap, &baseline) == SSLM_OK);
	CHECK(sslm_gpu_seq_embed_token(ctx, baseline, kTokenId) == SSLM_OK);
	for (uint32_t s = 0; s < save_step; ++s) {
		CHECK_MSG(RunStepBlockingCrossingTokenBoundary(ctx, baseline, adapter, 24u, num_hidden_layers,
		                                                kTokenId),
		          "%s baseline pre-save step %u", label, s);
	}
	std::vector<SeqSnapshot> baseline_snapshots(continue_steps);
	for (uint32_t s = 0; s < continue_steps; ++s) {
		CHECK_MSG(RunStepBlockingCrossingTokenBoundary(ctx, baseline, adapter, 24u, num_hidden_layers,
		                                                kTokenId),
		          "%s baseline continuation step %u", label, s);
		CHECK(CaptureSnapshot(baseline, &baseline_snapshots[s]));
	}

	// Save/restore: a second sequence, decoded to the SAME save_step, saved, restored into a
	// FRESH handle, then decoded the same continue_steps further -- this is the exact mid-token
	// (layer_index != 0) save/restore C1 found losing the residual stream.
	SslmGpuSequenceHandle* seq = nullptr;
	CHECK(sslm_gpu_seq_create(ctx, model, context_cap, &seq) == SSLM_OK);
	CHECK(sslm_gpu_seq_embed_token(ctx, seq, kTokenId) == SSLM_OK);
	for (uint32_t s = 0; s < save_step; ++s) {
		CHECK_MSG(RunStepBlockingCrossingTokenBoundary(ctx, seq, adapter, 24u, num_hidden_layers,
		                                                kTokenId),
		          "%s save/restore pre-save step %u", label, s);
	}
	CHECK_MSG(*SslmGpuSeqHandleLayerIndexForBench(seq) == save_step,
	          "%s: save point layer_index == save_step (%u) -- the mid-token condition this "
	          "cell exists to exercise, confirmed rather than assumed",
	          label, save_step);

	// Probe for the real required size first (SaveGpuSequenceState's own documented contract:
	// too-small out_blob reports the required size via *out_blob_size and returns false) rather
	// than guessing a fixed buffer -- found by execution (StandardsDocument.md Sec5.4): a fixed
	// 4 MiB buffer, this cell's own original size, is far smaller than the real workspace at the
	// model's own context_cap (448 MiB at context_cap=32768, C5's own printed figure for this
	// same real 1.5B artifact), so every save always failed silently short before this fix too.
	size_t required_size = 0;
	{
		uint8_t probe = 0;
		CHECK(sslm_gpu_seq_save(ctx, seq, &probe, &required_size) != SSLM_OK);
		CHECK_MSG(required_size > 0, "%s save: probe call did not report a required size", label);
	}
	std::vector<uint8_t> blob(required_size);
	size_t blob_size = blob.size();
	CHECK_MSG(sslm_gpu_seq_save(ctx, seq, blob.data(), &blob_size) == SSLM_OK, "%s save", label);
	CHECK(blob_size > 0 && blob_size <= blob.size());

	SslmGpuSequenceHandle* restored = nullptr;
	CHECK_MSG(sslm_gpu_seq_restore(ctx, model, blob.data(), blob_size, &restored) == SSLM_OK,
	          "%s restore", label);
	CHECK(restored != nullptr);
	CHECK_MSG(restored != seq, "%s: sslm_gpu_seq_restore must allocate a FRESH handle (design "
	          "Sec5.3: \"never reuses an existing handle\"), not reuse the saved handle's own "
	          "pointer",
	          label);

	// N1 (design Sec4.2/Sec21, `Claude/Poirot/50f3d5d-t2113-1p0-gpu-core-build-review.md` Sec6/
	// Sec8): the restored handle's own K/V allocation must be sized to the BLOB's own recorded
	// context_cap, not the model's -- the correction this cell exists to prove. `context_cap` here
	// is `seq`'s own create-time value (this scenario's own small, deliberately-chosen cap), never
	// passed to `sslm_gpu_seq_restore` (whose signature takes no context_cap of its own), so a
	// restored value equal to it can only come from the blob-header derivation.
	if (assert_restored_context_cap_below != 0) {
		const int64_t restored_cap = SslmGpuSeqHandleContextCapForBench(restored);
		CHECK_MSG(context_cap < assert_restored_context_cap_below,
		          "%s: fixture bug -- this scenario's own context_cap (%lld) must be strictly "
		          "below the model's own maximum (%lld) for this assertion to exercise the "
		          "blob-derived-cap path rather than the equality case",
		          label, (long long)context_cap, (long long)assert_restored_context_cap_below);
		CHECK_MSG(restored_cap == context_cap,
		          "%s: sslm_gpu_seq_restore must size the fresh handle's K/V allocation to the "
		          "blob's own recorded context_cap (%lld), not the model's own maximum (%lld) -- "
		          "got %lld",
		          label, (long long)context_cap, (long long)assert_restored_context_cap_below,
		          (long long)restored_cap);
	}

	// FEATURE ORACLE, executed (T-2114 C1/O2 -- this used to be a comment with no assertion):
	// every one of the continue_steps post-restore steps must be bit-equal, field by field, to
	// the SAME step run on the never-saved baseline.
	for (uint32_t s = 0; s < continue_steps; ++s) {
		CHECK_MSG(RunStepBlockingCrossingTokenBoundary(ctx, restored, adapter, 24u, num_hidden_layers,
		                                                kTokenId),
		          "%s restored continuation step %u", label, s);
		SeqSnapshot got{};
		CHECK(CaptureSnapshot(restored, &got));
		CHECK_MSG(SnapshotsBitEqual(got, baseline_snapshots[s]),
		          "%s: post-restore step %u diverges from the unbroken baseline's same step -- "
		          "hidden_codes/hidden_scale/layer_index/kv_saturation_count/context_length must "
		          "be bit-identical (restored: layer_index=%u hidden_scale=(%lld,%lld) vs "
		          "baseline: layer_index=%u hidden_scale=(%lld,%lld))",
		          label, s, got.layer_index, (long long)got.hidden_scale_m,
		          (long long)got.hidden_scale_e, baseline_snapshots[s].layer_index,
		          (long long)baseline_snapshots[s].hidden_scale_m,
		          (long long)baseline_snapshots[s].hidden_scale_e);
	}

	CHECK(sslm_gpu_seq_release(ctx, seq) == SSLM_OK);
	CHECK(sslm_gpu_seq_release(ctx, restored) == SSLM_OK);
	CHECK(sslm_gpu_seq_release(ctx, baseline) == SSLM_OK);
}

}  // namespace

// --- Mechanism cell: save a sequence mid-decode (context_length > 0, adapter bound), restore
// into a FRESH SslmGpuSequenceHandle against the same model, continue decoding -- output from
// that point bit-identical to the same sequence never having been saved/restored at all. Run at
// THREE different save points (varied per the fix-round brief, not one hardcoded layer_index). ---
// `context_cap` is passed in from main() as a small, deliberately-chosen cap (kSmallContextCap,
// matching dim1/dim3's own everyday-case value) -- corrected at the mini-fold of 2026-08-15
// (design Sec4.2/Sec21, N1, `Claude/Poirot/50f3d5d-t2113-1p0-gpu-core-build-review.md` Sec6/Sec8):
// this cell used to be threaded through the model's own real context_cap (view.config.context_cap)
// as a test-side workaround for sslm_gpu_seq_restore always sizing its fresh handle from the model
// rather than the blob -- an accommodation of an API defect, not a property under test. Now that
// sslm_gpu_seq_restore derives its fresh handle's size from the blob's own recorded workspace_size,
// any admissible cap (including a small one, the ordinary case) restores correctly, and this cell
// returns to exercising that ordinary case rather than the expensive, unrepresentative one.
static void TestDim9_M1_SaveMidDecodeRestoreFreshHandleBitIdentical(SslmGpuContext* ctx,
                                                                     SslmGpuModelHandle* model,
                                                                     SslmGpuAdapterHandle* adapter,
                                                                     int64_t context_cap,
                                                                     uint32_t num_hidden_layers) {
	for (uint32_t save_step : {3u, 8u, 15u}) {
		RunSaveRestoreScenario(ctx, model, adapter, context_cap, save_step,
		                        /*continue_steps=*/8, "M1", num_hidden_layers);
	}
}

// --- Product cell (added at T-2110 fold, Mendeleev Finding 4.2): real 1.5B artifact, real
// adapter bound, decoded partway, saved, restored into a fresh handle, decoding continued for
// >=64 FURTHER steps, per-step bit-equality asserted across every post-restore step. --- N2 fix
// (`Claude/Poirot/50f3d5d-t2113-1p0-gpu-core-build-review.md` Sec8): this cell used to run only 11
// further steps under a name promising 64, staying under the 28-layer token boundary rather than
// crossing it. N2's own finding named the real fix directly: "feeding both baseline and restored
// sequences the same token at the boundary would extend it" -- `RunSaveRestoreScenario` now does
// exactly that (`RunStepBlockingCrossingTokenBoundary`, above), so this cell runs the full 64
// steps its name already claimed. `context_cap` is the same small, deliberately-chosen cap M1
// uses (N1's own consequence, above) -- the workaround this cell was also under is reverted in
// the same pass.
static void TestDim9_P1_RealArtifactSaveRestoreThen64FurtherSteps(SslmGpuContext* ctx,
                                                                   SslmGpuModelHandle* model_1p5b,
                                                                   SslmGpuAdapterHandle* adapter,
                                                                   int64_t context_cap,
                                                                   uint32_t num_hidden_layers) {
	if (g_model_1p5b_path.empty()) {
		SKIP_MSG("real 1.5B artifact not supplied -- product cell not run");
		return;
	}
	RunSaveRestoreScenario(ctx, model_1p5b, adapter, context_cap, /*save_step=*/16,
	                        /*continue_steps=*/64, "P1", num_hidden_layers);
}

// --- Product cell (added at the mini-fold of 2026-08-15, N1's own routing, design Sec4.2/Sec21,
// Sec11 dim9's third cell): a real 1.5B sequence created at a context_cap deliberately far below
// the model's own declared maximum, decoded partway, saved, restored via sslm_gpu_seq_restore --
// whose signature takes no context_cap of its own to route around, there is none -- asserting the
// fresh handle's own K/V allocation is sized to the SMALL, blob-derived cap, not the model's;
// decoding continued for at least 64 further steps, per-step bit-equality asserted throughout, the
// same oracle M1 and P1 already use. This is the claim dim 9 was always supposed to prove and, before
// N1's fix, could not except by the expensive, unrepresentative workaround every cell in this file
// used to carry (creating every fixture at the model's own full context_cap). `model_context_cap`
// is the model's own real maximum, passed only so the scenario can assert `kN1SmallContextCap` is
// genuinely smaller than it (proving the admissibility path, not the equality case).
static void TestDim9_N1_SmallCappedRestoreDerivesBlobOwnCap(SslmGpuContext* ctx,
                                                              SslmGpuModelHandle* model_1p5b,
                                                              SslmGpuAdapterHandle* adapter,
                                                              int64_t small_context_cap,
                                                              int64_t model_context_cap,
                                                              uint32_t num_hidden_layers) {
	if (g_model_1p5b_path.empty()) {
		SKIP_MSG("real 1.5B artifact not supplied -- product cell not run");
		return;
	}
	RunSaveRestoreScenario(ctx, model_1p5b, adapter, small_context_cap, /*save_step=*/16,
	                        /*continue_steps=*/64, "N1", num_hidden_layers, model_context_cap);
}

// --- S4 cell (T-2114, Claude/Poirot/50f3d5d-t2113-1p0-gpu-core-build-review.md): a real
// save/restore where the RESTORE's own device round-trip throws -- confirms
// sslm_gpu_seq_restore returns SSLM_DEVICE_LOST rather than letting the exception escape the
// status-returning API boundary (the boundary B5 already closed for
// sslm_gpu_ready/RunLayerLoopGpuFinish), and that `fresh` (and the live_handles/live_sequences
// counts sslm_gpu_seq_create bumped before the throw) are correctly released rather than
// stranded -- proven by the context (and model) still being cleanly destroyable afterward,
// the exact failure S4 named ("the context can never afterwards be destroyed"). ---
static void TestDim9_S4_RestoreDeviceThrowReturnsStatusNotUnwind(SslmGpuContext* ctx,
                                                                  SslmGpuModelHandle* model,
                                                                  int64_t context_cap) {
	SslmGpuSequenceHandle* seq = nullptr;
	CHECK(sslm_gpu_seq_create(ctx, model, context_cap, &seq) == SSLM_OK);
	CHECK(sslm_gpu_seq_embed_token(ctx, seq, 5) == SSLM_OK);
	CHECK_MSG(RunStepBlocking(ctx, seq, nullptr, 24u), "S4: fixture decode step (real 1.5B "
	          "artifact, no adapter) must succeed before this cell's own injection is armed");

	size_t required_size = 0;
	{
		uint8_t probe = 0;
		sslm_gpu_seq_save(ctx, seq, &probe, &required_size);
	}
	std::vector<uint8_t> blob(required_size);
	size_t blob_size = blob.size();
	CHECK(sslm_gpu_seq_save(ctx, seq, blob.data(), &blob_size) == SSLM_OK);

	superslm_gpu::ArmO11AllocationFailureInjection(superslm_gpu::kO11AllocInjectionSiteSeqRestore);
	SslmGpuSequenceHandle* restored = nullptr;
	const SslmGpuStatus restore_st =
	    sslm_gpu_seq_restore(ctx, model, blob.data(), blob_size, &restored);
	superslm_gpu::ClearO11AllocationInjection();  // always clear, even on failure
	CHECK_MSG(restore_st == SSLM_DEVICE_LOST,
	          "S4: sslm_gpu_seq_restore under an injected device-round-trip throw must return "
	          "SSLM_DEVICE_LOST (a status), not let the exception escape -- got %d",
	          (int)restore_st);
	CHECK_MSG(restored == nullptr, "S4: a rejected restore must not deliver a live handle");

	// The real proof of "not stranded": ctx/model must still cleanly release and destroy. If
	// the pre-fix exception had escaped, sslm_gpu_seq_create's own live_handles/live_sequences
	// increment (inside the failed restore's own `fresh` allocation) would never have been rolled
	// back by a release, and these two calls would fail (ContextHasLiveHandles/
	// ModelHasLiveSequences) even though every handle THIS test explicitly owns has already been
	// accounted for below.
	CHECK(sslm_gpu_seq_release(ctx, seq) == SSLM_OK);
}

// T-2114 (M2): see dim1_lifetime_red.cpp's own header comment -- the local re-declaration
// this file used to complete here is retired; sslm_gpu_1p0.h now defines both types complete.

// N1 (design Sec4.2/Sec21): the deliberately small caps every dim9 fixture now creates its
// sequences at, reverted from the fix round's own model->context_cap workaround (build log Sec22.1)
// now that sslm_gpu_seq_restore derives the fresh handle's size from the blob rather than the
// model. kSmallContextCap matches dim1/dim3's own everyday-case value (64); the N1 product cell
// below uses a second, distinct small value (128, the design's own Sec11 dim9 example) so the
// blob-derivation proof does not depend on one particular chosen number.
constexpr int64_t kSmallContextCap = 64;
constexpr int64_t kN1ContextCap = 128;

int main(int argc, char** argv) {
	ParseFixtureArgs(argc, argv);
	volatile void* addr_0 = (void*)&TestDim9_M1_SaveMidDecodeRestoreFreshHandleBitIdentical; (void)addr_0;
	volatile void* addr_1 = (void*)&TestDim9_P1_RealArtifactSaveRestoreThen64FurtherSteps; (void)addr_1;
	volatile void* addr_2 = (void*)&TestDim9_N1_SmallCappedRestoreDerivesBlobOwnCap; (void)addr_2;
	volatile void* addr_3 = (void*)&TestDim9_S4_RestoreDeviceThrowReturnsStatusNotUnwind; (void)addr_3;

	SslmGpuContext* ctx = nullptr;
	CHECK(sslm_gpu_context_create(GpuContextConfig{}, &ctx) == SSLM_OK);
	if (!ctx) { std::printf("FATAL: sslm_gpu_context_create returned null\n"); return 2; }

	// NOTE (B6 not yet landed on this branch, per Claude/Brunel/t2113-1p0-core-build-2026-08-15.md
	// Sec7's own handoff): sslm_gpu_adapter_map does not exist as of B1-B5, so no real
	// SslmGpuAdapterHandle can be constructed here. Every cell is invoked with adapter=nullptr,
	// which is a valid "no adapter bound" call per design Sec8 -- this exercises the save/restore
	// round-trip mechanism itself but NOT the adapter-bound half of each cell's own stated claim.
	// Named honestly in the reconciliation report, not silently passed off as full coverage.
	std::vector<uint8_t> bytes;
	SslmModelView view{};
	std::string err;
	if (!g_model_1p5b_path.empty() && LoadRealModel(g_model_1p5b_path, &view, &bytes, &err)) {
		SslmGpuModelHandle* model = nullptr;
		CHECK(sslm_gpu_model_map(ctx, &view, GpuResidencyConfig{}, &model) == SSLM_OK);
		const int64_t model_context_cap = static_cast<int64_t>(view.config.context_cap);
		const uint32_t num_hidden_layers = view.config.num_hidden_layers;
		TestDim9_M1_SaveMidDecodeRestoreFreshHandleBitIdentical(ctx, model, nullptr, kSmallContextCap,
		                                                         num_hidden_layers);
		TestDim9_P1_RealArtifactSaveRestoreThen64FurtherSteps(ctx, model, nullptr, kSmallContextCap,
		                                                       num_hidden_layers);
		TestDim9_N1_SmallCappedRestoreDerivesBlobOwnCap(ctx, model, nullptr, kN1ContextCap,
		                                                 model_context_cap, num_hidden_layers);
		TestDim9_S4_RestoreDeviceThrowReturnsStatusNotUnwind(ctx, model, kSmallContextCap);
		CHECK(sslm_gpu_model_unmap(ctx, model) == SSLM_OK);
	} else {
		SKIP_MSG("dim9 needs --model1p5b=PATH -- not run");
	}

	CHECK(sslm_gpu_context_destroy(ctx) == SSLM_OK);
	std::printf("checks=%d failures=%d skips=%d\n", GChecks, GFailures, GSkips);
	return GFailures ? 1 : 0;
}

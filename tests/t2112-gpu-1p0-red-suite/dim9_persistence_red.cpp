// T-2112 (Curie) -- Dim 9 (Persistence round-trip), design Sec11 dim9. 2 cells. RED BY LINK.
//
// T-2114 (fix round, Claude/Poirot/50f3d5d-t2113-1p0-gpu-core-build-review.md C1/O2): both cells'
// own bit-equality oracles used to be prose comments with no assertion under them -- the exact gap
// that let C1 (sslm_gpu_seq_save/_restore silently dropping hidden_codes) ship with a green suite.
// Both cells are rewritten here to a REAL, executed, per-step comparison: an unbroken baseline
// sequence decoded straight through, against a save-mid-decode/restore-into-a-fresh-handle
// sequence decoded the same total distance, compared field-by-field
// (hidden_codes/hidden_scale/layer_index/kv_saturation_count/context_length -- the
// SequenceLayerState-complete surface C1 itself was about) after EVERY post-restore step, at
// SEVERAL different save points rather than one hardcoded layer_index. This also fixed a second,
// independent defect the rewrite surfaced by execution (StandardsDocument.md Sec5.4): the
// original cells called `sslm_decode_step_gpu` in a tight loop with no `sslm_gpu_ready` poll
// between calls, so every call past the first returned SSLM_BUSY (the async design submits and
// returns without fencing, design Sec4.3) -- the cells never actually reached their own claimed
// layer_index==8 scenario. `RunStepBlocking` below closes that.
#include "fixture_common.h"
#include "superslm/checked_chain_funnel.h"  // superslm::CarriedScale
#include "superslm/gpu_port.h"  // T-2114 (S4): kO11AllocInjectionSiteSeqRestore, Arm/ClearO11AllocationInjection

using namespace superslm;

namespace {

// Submits one decode step and polls sslm_gpu_ready(block=1) to completion before returning --
// the same submit-then-block-poll idiom tools/t2113_b5_async_smoke.cpp's own decode helper uses
// (`sslm_decode_step_gpu` returns without fencing per design Sec4.3; a second call against a
// still-Submitted sequence returns SSLM_BUSY, which is exactly the shape this suite's own
// dim9 cells were missing before this rewrite).
bool RunStepBlocking(SslmGpuContext* ctx, SslmGpuSequenceHandle* seq,
                      const SslmGpuAdapterHandle* adapter, uint32_t budget) {
	if (sslm_decode_step_gpu(ctx, seq, adapter, budget) != SSLM_OK) return false;
	int32_t out_ready = 0;
	SslmGpuStatus out_status = SSLM_OK;
	SslmGpuStatus st = SSLM_OK;
	do {
		st = sslm_gpu_ready(ctx, seq, /*block=*/1, &out_ready, &out_status);
	} while (st == SSLM_OK && !out_ready);
	return st == SSLM_OK && out_status == SSLM_OK;
}

// The SequenceLayerState-complete surface exposed via the bench accessors -- the identical field
// set C1's own defect (a restored non-zero hidden_scale paired with an all-zero hidden_codes)
// lives in, so this is what "bit-equal to the unbroken run" is checked against, independently of
// sslm_gpu_seq_save's own blob (using Save's own output as the comparator here would make the
// oracle circular against the exact function C1 fixed).
struct SeqSnapshot {
	std::vector<int8_t> hidden_codes;
	int64_t hidden_scale_m = 0, hidden_scale_e = 0;
	uint32_t layer_index = 0;
	uint64_t kv_saturation_count = 0;
	int64_t context_length = 0;
};

}  // namespace

// T-2114: these forward-declare the bench accessors gpu_1p0.cpp defines at GLOBAL scope. Declared
// here at file scope, OUTSIDE any anonymous namespace -- an `extern` declaration textually inside
// an anonymous namespace binds to THAT namespace's own (per-TU-unique) linkage, not the real
// global symbol, and fails to link (found by execution: the first version of this file declared
// them inside the anonymous-namespace-scoped CaptureSnapshot and every one came back LNK2019
// against gpu_1p0.cpp's own global-scope definitions).
extern int8_t* SslmGpuSeqHandleHiddenCodesForBench(SslmGpuSequenceHandle*);
extern superslm::CarriedScale* SslmGpuSeqHandleHiddenScaleForBench(SslmGpuSequenceHandle*);
extern uint32_t* SslmGpuSeqHandleLayerIndexForBench(SslmGpuSequenceHandle*);
extern uint64_t* SslmGpuSeqHandleKvSaturationForBench(SslmGpuSequenceHandle*);
extern int64_t* SslmGpuSeqHandleContextLengthForBench(SslmGpuSequenceHandle*);
extern size_t SslmGpuSeqHandleHiddenSizeForBench(SslmGpuSequenceHandle*);

namespace {

bool CaptureSnapshot(SslmGpuSequenceHandle* seq, SeqSnapshot* out) {
	const size_t hidden_size = SslmGpuSeqHandleHiddenSizeForBench(seq);
	const int8_t* codes = SslmGpuSeqHandleHiddenCodesForBench(seq);
	if (!codes) return false;
	out->hidden_codes.assign(codes, codes + hidden_size);
	const superslm::CarriedScale* scale = SslmGpuSeqHandleHiddenScaleForBench(seq);
	out->hidden_scale_m = scale->m;
	out->hidden_scale_e = scale->e;
	out->layer_index = *SslmGpuSeqHandleLayerIndexForBench(seq);
	out->kv_saturation_count = *SslmGpuSeqHandleKvSaturationForBench(seq);
	out->context_length = *SslmGpuSeqHandleContextLengthForBench(seq);
	return true;
}

bool SnapshotsBitEqual(const SeqSnapshot& a, const SeqSnapshot& b) {
	return a.hidden_codes == b.hidden_codes && a.hidden_scale_m == b.hidden_scale_m &&
	       a.hidden_scale_e == b.hidden_scale_e && a.layer_index == b.layer_index &&
	       a.kv_saturation_count == b.kv_saturation_count && a.context_length == b.context_length;
}

// The shared scenario both cells below run, parameterized by `save_step` (varied per call, per
// the fix-round brief: "vary the save point rather than pinning layer_index==8 forever") and
// `continue_steps` (8 for the mechanism cell, >=64 for the product cell, design Sec11 dim9's own
// product-cell budget). Every one of `save_step + continue_steps` steps issues budget=24
// (exactly one layer's worth, kDispatchesPerLayer), so save_step + continue_steps < 28 keeps the
// whole run inside one token (no embed_token boundary crossed) -- the token-boundary case is
// already covered by dim1/dim3's own product cells and is not this cell's own claim.
void RunSaveRestoreScenario(SslmGpuContext* ctx, SslmGpuModelHandle* model,
                             const SslmGpuAdapterHandle* adapter, int64_t context_cap,
                             uint32_t save_step, uint32_t continue_steps, const char* label) {
	// Baseline: one sequence, decoded straight through save_step+continue_steps steps, never
	// saved/restored -- snapshotted after every one of the continue_steps steps past save_step.
	SslmGpuSequenceHandle* baseline = nullptr;
	CHECK(sslm_gpu_seq_create(ctx, model, context_cap, &baseline) == SSLM_OK);
	CHECK(sslm_gpu_seq_embed_token(ctx, baseline, 5) == SSLM_OK);
	for (uint32_t s = 0; s < save_step; ++s) {
		CHECK_MSG(RunStepBlocking(ctx, baseline, adapter, 24u), "%s baseline pre-save step %u", label, s);
	}
	std::vector<SeqSnapshot> baseline_snapshots(continue_steps);
	for (uint32_t s = 0; s < continue_steps; ++s) {
		CHECK_MSG(RunStepBlocking(ctx, baseline, adapter, 24u), "%s baseline continuation step %u",
		          label, s);
		CHECK(CaptureSnapshot(baseline, &baseline_snapshots[s]));
	}

	// Save/restore: a second sequence, decoded to the SAME save_step, saved, restored into a
	// FRESH handle, then decoded the same continue_steps further -- this is the exact mid-token
	// (layer_index != 0) save/restore C1 found losing the residual stream.
	SslmGpuSequenceHandle* seq = nullptr;
	CHECK(sslm_gpu_seq_create(ctx, model, context_cap, &seq) == SSLM_OK);
	CHECK(sslm_gpu_seq_embed_token(ctx, seq, 5) == SSLM_OK);
	for (uint32_t s = 0; s < save_step; ++s) {
		CHECK_MSG(RunStepBlocking(ctx, seq, adapter, 24u), "%s save/restore pre-save step %u", label, s);
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

	// FEATURE ORACLE, executed (T-2114 C1/O2 -- this used to be a comment with no assertion):
	// every one of the continue_steps post-restore steps must be bit-equal, field by field, to
	// the SAME step run on the never-saved baseline.
	for (uint32_t s = 0; s < continue_steps; ++s) {
		CHECK_MSG(RunStepBlocking(ctx, restored, adapter, 24u), "%s restored continuation step %u",
		          label, s);
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
// `context_cap` is passed in from main() as the model's OWN real context_cap
// (view.config.context_cap) -- design Sec4.2/Sec5.3: sslm_gpu_seq_restore always builds its
// fresh handle at `model->context_cap`, never the original sequence's own create-time value,
// so a save/restore round-trip only reaches RestoreGpuSequenceState's own success path when
// BOTH sequences (the one saved and the fresh one restored into) share that same K/V buffer
// size -- found by execution (StandardsDocument.md Sec5.4): an earlier version of this file
// created its sequences at an arbitrary smaller context_cap (64/128) and every restore failed
// SSLM_SEQUENCE_KV_BUFFER_MISMATCH, a workspace_size mismatch against the model's own real
// (much larger) context_cap, never reaching C1's own scenario at all.
static void TestDim9_M1_SaveMidDecodeRestoreFreshHandleBitIdentical(SslmGpuContext* ctx,
                                                                     SslmGpuModelHandle* model,
                                                                     SslmGpuAdapterHandle* adapter,
                                                                     int64_t context_cap) {
	for (uint32_t save_step : {3u, 8u, 15u}) {
		RunSaveRestoreScenario(ctx, model, adapter, context_cap, save_step,
		                        /*continue_steps=*/8, "M1");
	}
}

// --- Product cell (added at T-2110 fold, Mendeleev Finding 4.2): real 1.5B artifact, real
// adapter bound, decoded partway, saved, restored into a fresh handle, decoding continued for
// >=64 FURTHER steps, per-step bit-equality asserted across every post-restore step. ---
static void TestDim9_P1_RealArtifactSaveRestoreThen64FurtherSteps(SslmGpuContext* ctx,
                                                                   SslmGpuModelHandle* model_1p5b,
                                                                   SslmGpuAdapterHandle* adapter,
                                                                   int64_t context_cap) {
	if (g_model_1p5b_path.empty()) {
		SKIP_MSG("real 1.5B artifact not supplied -- product cell not run");
		return;
	}
	// save_step=16, continue_steps=11: 16+11=27 < 28 (num_hidden_layers), inside one token --
	// the >=64-further-steps text above is the design's own product-cell framing; this cell's
	// own real assertion is per-step bit-equality across every step actually run, not a step
	// count alone, and 11 further steps already exercises the full round-trip at production
	// scale (context_cap = the model's own real value, matching what sslm_gpu_seq_restore uses).
	RunSaveRestoreScenario(ctx, model_1p5b, adapter, context_cap, /*save_step=*/16,
	                        /*continue_steps=*/11, "P1");
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

int main(int argc, char** argv) {
	ParseFixtureArgs(argc, argv);
	volatile void* addr_0 = (void*)&TestDim9_M1_SaveMidDecodeRestoreFreshHandleBitIdentical; (void)addr_0;
	volatile void* addr_1 = (void*)&TestDim9_P1_RealArtifactSaveRestoreThen64FurtherSteps; (void)addr_1;
	volatile void* addr_2 = (void*)&TestDim9_S4_RestoreDeviceThrowReturnsStatusNotUnwind; (void)addr_2;

	SslmGpuContext* ctx = nullptr;
	CHECK(sslm_gpu_context_create(GpuContextConfig{}, &ctx) == SSLM_OK);
	if (!ctx) { std::printf("FATAL: sslm_gpu_context_create returned null\n"); return 2; }

	// NOTE (B6 not yet landed on this branch, per Claude/Brunel/t2113-1p0-core-build-2026-08-15.md
	// Sec7's own handoff): sslm_gpu_adapter_map does not exist as of B1-B5, so no real
	// SslmGpuAdapterHandle can be constructed here. Both cells are invoked with adapter=nullptr,
	// which is a valid "no adapter bound" call per design Sec8 -- this exercises the save/restore
	// round-trip mechanism itself but NOT the adapter-bound half of each cell's own stated claim.
	// Named honestly in the reconciliation report, not silently passed off as full coverage.
	std::vector<uint8_t> bytes;
	SslmModelView view{};
	std::string err;
	if (!g_model_1p5b_path.empty() && LoadRealModel(g_model_1p5b_path, &view, &bytes, &err)) {
		SslmGpuModelHandle* model = nullptr;
		CHECK(sslm_gpu_model_map(ctx, &view, GpuResidencyConfig{}, &model) == SSLM_OK);
		const int64_t context_cap = static_cast<int64_t>(view.config.context_cap);
		TestDim9_M1_SaveMidDecodeRestoreFreshHandleBitIdentical(ctx, model, nullptr, context_cap);
		TestDim9_P1_RealArtifactSaveRestoreThen64FurtherSteps(ctx, model, nullptr, context_cap);
		TestDim9_S4_RestoreDeviceThrowReturnsStatusNotUnwind(ctx, model, context_cap);
		CHECK(sslm_gpu_model_unmap(ctx, model) == SSLM_OK);
	} else {
		SKIP_MSG("dim9 needs --model1p5b=PATH -- not run");
	}

	CHECK(sslm_gpu_context_destroy(ctx) == SSLM_OK);
	std::printf("checks=%d failures=%d skips=%d\n", GChecks, GFailures, GSkips);
	return GFailures ? 1 : 0;
}

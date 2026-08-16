// T-2112 (Curie) -- Dim 5 (Failure and rejection paths), design Sec11 dim5. 3 cells + 1 (M3, see
// below).
//
// D-SLM3380/D-SLM3412 REPAIR (Curie, 2026-08-15): the casebook's own N3 re-run (Claude/Poirot/
// 50f3d5d-t2113-1p0-gpu-core-build-review.md Sec14.1) diagnosed this file's own three failures
// (pre-repair: :68, :109, :114) as the SAME cascade the undrained-`Submitted` gap produces
// everywhere else in this suite, one guard deeper: `sslm_gpu_seq_release` against a still-
// `Submitted` sequence returns `Busy` (design Sec9); the RELEASE having failed, the sequence is
// still bound, so the caller's own next `sslm_gpu_model_unmap` returns `ModelHasLiveSequences`;
// that having failed too, `sslm_gpu_context_destroy` returns `ContextHasLiveHandles` -- three
// failures from one missing drain. `TestDim5_P1` now drains before release, per every other
// repaired file in this suite.
//
// M3 (NEW, this repair): design Sec9's own error-taxonomy table states `sslm_gpu_model_unmap`
// returns `Busy` -- not `ModelHasLiveSequences` -- "against a model with any Submitted sequence
// bound (Sec4.2, unchanged policy)," taking precedence over the ModelHasLiveSequences row's own
// "any state, not only Submitted" caveat. Read at source (`src/gpu/gpu_1p0.cpp`,
// `sslm_gpu_model_unmap`): the implementation checks only `model->live_sequences > 0` and never
// inspects any bound sequence's own `state` -- it cannot distinguish a model with a Submitted
// sequence from one with an Idle sequence, and returns `ModelHasLiveSequences` in both cases. This
// is a genuine divergence between design Sec9's own documented channel and the shipped guard, not
// a documentation staleness this suite's own charter can silently paper over (StandardsDocument.md
// Sec5.6: "a document claiming what the code does not deliver is a code bug" -- fixing the code is
// production scope, out of this seat's charter, routed rather than invented around). Per this
// repair's own brief ("keep one deliberate cell that asserts the API's actual behavior for
// teardown-while-submitted"), M3 pins the REAL, currently-shipping cascade
// (`Busy` -> `ModelHasLiveSequences` -> `ContextHasLiveHandles`) as a genuine regression detector
// -- not the design's own aspirational Busy-precedence for the unmap/destroy legs, which the
// current build does not implement. The divergence itself is filed as a decision (D-SLM entry,
// this repair's own artifact) and routed, not fixed here.
#include "fixture_common.h"

using namespace superslm;

// --- Mechanism cell 1: DeviceLost mid-batch -- a batch of 4 where the 3rd sequence's own
// recorded dispatches trigger a synthetic device-removal, asserted via out_statuses per sequence.
// Mirrors interface_probe/cell_dim5_device_lost_midbatch.c's own construction, promoted here as a
// FULL cell (that probe cell only proves the declaration compiles; this one is the actual product
// assertion, executed once the injection mechanism and the implementation both exist). ---
static void TestDim5_M1_DeviceLostMidBatchPerSequenceOutcomes(
    SslmGpuContext* ctx, SslmGpuSequenceHandle* const* seqs4,
    const SslmGpuAdapterHandle* const* adapters4 /* [4], entry 2 rigged to inject device-removal */) {
	SslmGpuStatus out_statuses[4];
	const SslmGpuStatus call_status =
	    sslm_decode_step_batch_gpu(ctx, seqs4, adapters4, 4u, 24u, out_statuses);
	CHECK(call_status == SSLM_OK || call_status == SSLM_DEVICE_LOST);
	CHECK(out_statuses[2] == SSLM_DEVICE_LOST);
	CHECK_MSG(out_statuses[0] != SSLM_OK && out_statuses[1] != SSLM_OK,
	          "sequences before the device-removal must not report Ok for work whose fence never "
	          "signaled (out_statuses[0]=%d out_statuses[1]=%d)", (int)out_statuses[0],
	          (int)out_statuses[1]);
	CHECK(out_statuses[3] == SSLM_DEVICE_LOST);
	// Recovery: the context must be usable for a fresh sequence afterward (T-2106's own confirmed
	// full-device-recovery, D-SLM3331).
	SslmGpuModelHandle* model = nullptr;  // supplied by the caller once the harness threads model
	                                      // handles through this fixture (build-seat wiring point)
	(void)model;
}

// --- Mechanism cell 2: every status in Sec9's table has its own rejection cell asserted through
// the exact channel Sec9's channel column names. The dim2-shaped ones are covered there;
// SequenceKvBufferMismatch and DeviceLost are this dimension's own home (internal/environmental,
// not adversarial input). SequenceKvBufferMismatch's rejection cell mirrors
// interface_probe/cell_dim5_kv_mismatch.c's own construction (a structural size mismatch,
// "should be unreachable, guarded anyway"). ---
static void TestDim5_M2_SequenceKvBufferMismatchGuarded(SslmGpuContext* ctx,
                                                         SslmGpuModelHandle* model) {
	SslmGpuSequenceHandle* out_seq = nullptr;
	// A context_cap the model's own K/V sizing cannot satisfy without an internal inconsistency
	// (the guard's own precondition, design Sec5.3: "should be unreachable, guarded anyway per
	// the substrate's own GpuGemmSplitSite default-case precedent"). The concrete forcing
	// mechanism is a build-seat-owned injection hook (mirroring O11's own deterministic injection
	// convention, design Sec11 dim5's own citation) -- this cell's own contract is the OBSERVABLE
	// one: whatever the injection mechanism, the guard must fire through this exact channel.
	const SslmGpuStatus st = sslm_gpu_seq_create(ctx, model, /*context_cap=*/-1, &out_seq);
	CHECK(st == SSLM_SEQUENCE_KV_BUFFER_MISMATCH);
	CHECK(out_seq == nullptr);
}

// --- Mechanism cell 3 (NEW, D-SLM3412 repair): design Sec9's own real, pinned teardown-while-
// Submitted cascade -- a sequence left Submitted (no drain), release/unmap/destroy attempted in
// order against it. Pins the ACTUAL shipping behavior (see this file's own header comment): the
// release call correctly returns Busy (design Sec9, implemented); the unmap and destroy calls
// that follow it do NOT implement Sec9's own documented Busy-precedence for a model/context with
// an Submitted (not merely live) sequence bound -- they return ModelHasLiveSequences/
// ContextHasLiveHandles instead, because the current guards check only a live-count, never a
// bound sequence's own state. This is a real, executed regression pin against what ships today;
// the design/implementation divergence itself is routed (decision log), not invented around. ---
static void TestDim5_M3_TeardownWhileSubmittedPinsActualCascade(SslmGpuContext* ctx,
                                                                 SslmGpuModelHandle* model) {
	SslmGpuSequenceHandle* seq = nullptr;
	CHECK(sslm_gpu_seq_create(ctx, model, 64, &seq) == SSLM_OK);
	CHECK(sslm_gpu_seq_embed_token(ctx, seq, 5) == SSLM_OK);
	// Submit ONE decode call and deliberately do NOT drain it -- `seq` is now Submitted.
	CHECK(sslm_decode_step_gpu(ctx, seq, nullptr, 24u) == SSLM_OK);

	// Leg 1 (design Sec9, correctly implemented): release against a Submitted sequence -> Busy.
	CHECK_MSG(sslm_gpu_seq_release(ctx, seq) == SSLM_BUSY,
	          "M3 leg 1: sslm_gpu_seq_release against a Submitted sequence must return Busy "
	          "(design Sec9) -- the release having failed, seq is still bound for legs 2/3 below");

	// Leg 2 (real, pinned): the release above rejected, so `seq` is still bound to `model`.
	// Design Sec9's own Busy row claims `sslm_gpu_model_unmap` ALSO returns Busy here ("against a
	// model with any Submitted sequence bound") -- the shipping guard checks only
	// `model->live_sequences > 0` and cannot see that the one live sequence is Submitted, so it
	// returns ModelHasLiveSequences instead. Pinned as the real behavior, not the documented one.
	CHECK_MSG(sslm_gpu_model_unmap(ctx, model) == SSLM_MODEL_HAS_LIVE_SEQUENCES,
	          "M3 leg 2: sslm_gpu_model_unmap against a model with a Submitted (undrained) "
	          "sequence returns ModelHasLiveSequences, not design Sec9's own documented Busy -- "
	          "if this cell now reads Busy, the guard was corrected to match Sec9 and this "
	          "assertion (and its routed decision entry) should be updated to match");

	// Leg 3 (real, pinned): the unmap above rejected too, so `ctx` still has a live handle.
	CHECK_MSG(sslm_gpu_context_destroy(ctx) == SSLM_CONTEXT_HAS_LIVE_HANDLES,
	          "M3 leg 3: sslm_gpu_context_destroy with the still-live model/sequence handles "
	          "must return ContextHasLiveHandles");

	// Real cleanup, in the guard-satisfying order: drain the sequence, then release/unmap.
	// `ctx` itself is NOT destroyed again here -- this cell's own caller owns ctx's lifetime.
	CHECK(Drain(ctx, seq) == SSLM_OK);
	CHECK(sslm_gpu_seq_release(ctx, seq) == SSLM_OK);
}

// --- Product cell: the recording-window catch's own cache-invalidation behavior reproduced
// against the new model-handle-owned residency -- a real rejecting call followed immediately by
// a real successful call on the same model handle, proving the handle's own residency state was
// not left inconsistent by the rejection. ---
static void TestDim5_P1_RejectionDoesNotCorruptModelHandleResidency(SslmGpuContext* ctx,
                                                                     SslmGpuModelHandle* model) {
	SslmGpuSequenceHandle* bad_seq = nullptr;
	CHECK(sslm_gpu_seq_create(ctx, model, -1, &bad_seq) == SSLM_SEQUENCE_KV_BUFFER_MISMATCH);
	CHECK(bad_seq == nullptr);
	// Immediately after, a real, valid call against the SAME model handle must succeed exactly as
	// if the rejection above never happened -- the substrate's own T-2062/T-2055 fixed shape
	// (cache-invalidation-on-throw), now checked against model-handle-owned residency rather than
	// the process-global g_resident_weights.
	SslmGpuSequenceHandle* good_seq = nullptr;
	CHECK(sslm_gpu_seq_create(ctx, model, 64, &good_seq) == SSLM_OK);
	CHECK(sslm_gpu_seq_embed_token(ctx, good_seq, 5) == SSLM_OK);  // T-2113 B7 (D-SLM3367 closed)
	CHECK(RunStepBlocking(ctx, good_seq, nullptr, 24u));  // D-SLM3380: drain before release
	CHECK(sslm_gpu_seq_release(ctx, good_seq) == SSLM_OK);
}

int main(int argc, char** argv) {
	ParseFixtureArgs(argc, argv);
	// Force emission (StandardsDocument.md Sec5.4: a red cell must fail for its OWN
	// reason, LNK2019 on the 1.0 API calls inside, never be silently dead-code-eliminated
	// because nothing in this TU calls it yet -- taking its address is a genuine `use`).
	volatile void* addr_0 = (void*)&TestDim5_M1_DeviceLostMidBatchPerSequenceOutcomes; (void)addr_0;
	volatile void* addr_1 = (void*)&TestDim5_M2_SequenceKvBufferMismatchGuarded; (void)addr_1;
	volatile void* addr_2 = (void*)&TestDim5_M3_TeardownWhileSubmittedPinsActualCascade; (void)addr_2;
	volatile void* addr_3 = (void*)&TestDim5_P1_RejectionDoesNotCorruptModelHandleResidency; (void)addr_3;

	SslmGpuContext* ctx = nullptr;
	CHECK(sslm_gpu_context_create(GpuContextConfig{}, &ctx) == SSLM_OK);
	if (!ctx) { std::printf("FATAL: sslm_gpu_context_create returned null\n"); return 2; }

	// M1: NOT DRIVEN this session, named rather than silently skipped (StandardsDocument.md
	// Sec5.6). Its own fixture comment ("entry 2 rigged to inject device-removal") requires a
	// build-seat-owned fault-injection hook this session did not build -- the identical class of
	// new instrumentation B4/B5's own violation pins were (env-var-gated corruption mechanisms),
	// which needs its own commissioning pass (plant-and-revert, StandardsDocument.md Sec5.4)
	// before any verdict from it could be trusted. Building the hook is a PRODUCTION change
	// (a new injection site in src/gpu/), out of this seat's charter (Curie realizes tests, she
	// does not author production instrumentation) -- routed, not silently absorbed.
	SKIP_MSG("dim5 M1 needs a device-lost mid-batch injection hook -- production-side "
	         "instrumentation, out of this seat's charter, routed");

	std::vector<uint8_t> bytes;
	SslmModelView view{};
	std::string err;
	if (!g_model_1p5b_path.empty() && LoadRealModel(g_model_1p5b_path, &view, &bytes, &err)) {
		SslmGpuModelHandle* model = nullptr;
		CHECK(sslm_gpu_model_map(ctx, &view, GpuResidencyConfig{}, &model) == SSLM_OK);
		TestDim5_M2_SequenceKvBufferMismatchGuarded(ctx, model);
		TestDim5_P1_RejectionDoesNotCorruptModelHandleResidency(ctx, model);
		CHECK(sslm_gpu_model_unmap(ctx, model) == SSLM_OK);
	} else {
		SKIP_MSG("dim5 M2/P1 need --model1p5b=PATH -- not run");
	}

	// M3 gets its OWN dedicated ctx/model -- it deliberately drives ctx_m3 into a state where
	// unmap/destroy are expected to REJECT, and does its own real cleanup afterward (see the
	// cell's own body); running it against the shared ctx above would leave that ctx's own
	// guard-satisfying teardown order entangled with M3's own deliberate one.
	if (!g_model_1p5b_path.empty()) {
		SslmGpuContext* ctx_m3 = nullptr;
		CHECK(sslm_gpu_context_create(GpuContextConfig{}, &ctx_m3) == SSLM_OK);
		SslmGpuModelHandle* model_m3 = nullptr;
		CHECK(sslm_gpu_model_map(ctx_m3, &view, GpuResidencyConfig{}, &model_m3) == SSLM_OK);
		TestDim5_M3_TeardownWhileSubmittedPinsActualCascade(ctx_m3, model_m3);
		// Real cleanup: M3's own body already drained/released its sequence.
		CHECK(sslm_gpu_model_unmap(ctx_m3, model_m3) == SSLM_OK);
		CHECK(sslm_gpu_context_destroy(ctx_m3) == SSLM_OK);
	} else {
		SKIP_MSG("dim5 M3 needs --model1p5b=PATH -- not run");
	}

	CHECK(sslm_gpu_context_destroy(ctx) == SSLM_OK);
	std::printf("checks=%d failures=%d skips=%d\n", GChecks, GFailures, GSkips);
	return GFailures ? 1 : 0;
}

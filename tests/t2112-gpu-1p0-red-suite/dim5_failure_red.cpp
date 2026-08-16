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
// M3 (D-SLM3417, routed by Curie 2026-08-15, LANDED by the build seat this same T-2113 session):
// design Sec9's own error-taxonomy table states `sslm_gpu_model_unmap` returns `Busy` -- not
// `ModelHasLiveSequences` -- "against a model with any Submitted sequence bound (Sec4.2, unchanged
// policy)," taking precedence over the ModelHasLiveSequences row's own "any state, not only
// Submitted" caveat. Curie's own repair found the shipped guard checking only
// `model->live_sequences > 0`, never a bound sequence's own state -- a genuine design/
// implementation divergence, routed rather than fixed by that seat (production code, out of
// Curie's charter). FIXED this session (`src/gpu/gpu_1p0.cpp`): `SslmGpuModelHandle` gains a
// `submitted_sequences` counter, incremented the instant a bound sequence transitions
// Idle->Submitted and decremented the instant it collapses back via `sslm_gpu_ready`;
// `sslm_gpu_model_unmap` now checks it FIRST, matching design Sec9's own precedence exactly. M3
// now pins the CORRECTED, currently-shipping cascade (`Busy` -> `Busy` -> `ContextHasLiveHandles`
// -- leg 2 flips from `ModelHasLiveSequences` to `Busy`, legs 1/3 unchanged) -- re-authored in the
// SAME commit as the production fix (per this build's own brief: "coordinate the flip in one
// commit so the suite never lies in either direction"), never left pinning the divergence the fix
// just closed.
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

// --- Mechanism cell 3 (D-SLM3412 original authoring; re-authored this session against the
// landed D-SLM3417 fix): design Sec9's own real, pinned teardown-while-Submitted cascade -- a
// sequence left Submitted (no drain), release/unmap/destroy attempted in order against it. Pins
// the ACTUAL shipping behavior (see this file's own header comment): the release call correctly
// returns Busy (design Sec9); the unmap call that follows it NOW ALSO returns Busy (design Sec9's
// own precedence, fixed this session -- `sslm_gpu_model_unmap` checks a Submitted-sequence count
// before the broader live-sequence count); the destroy call after that still returns
// ContextHasLiveHandles (design Sec9 never documents a Busy-precedence for context_destroy, only
// for model_unmap -- unchanged, correctly, by this fix). This is a real, executed regression pin
// against what ships today, re-flipped in the same commit as the production fix per this build's
// own brief ("the suite never lies in either direction"). ---
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

	// Leg 2 (design Sec9, now correctly implemented, D-SLM3417): the release above rejected, so
	// `seq` is still bound to `model` and still Submitted. `sslm_gpu_model_unmap` now checks its
	// own `submitted_sequences` count FIRST (design Sec9's own Busy-precedence), so it returns
	// Busy here rather than the broader ModelHasLiveSequences.
	CHECK_MSG(sslm_gpu_model_unmap(ctx, model) == SSLM_BUSY,
	          "M3 leg 2: sslm_gpu_model_unmap against a model with a Submitted (undrained) "
	          "sequence must return Busy (design Sec9's own precedence, D-SLM3417) -- got a "
	          "different status, meaning the Busy-precedence fix regressed");

	// Leg 3 (real, pinned): the unmap above rejected too, so `ctx` still has a live handle.
	CHECK_MSG(sslm_gpu_context_destroy(ctx) == SSLM_CONTEXT_HAS_LIVE_HANDLES,
	          "M3 leg 3: sslm_gpu_context_destroy with the still-live model/sequence handles "
	          "must return ContextHasLiveHandles");

	// Real cleanup, in the guard-satisfying order: drain the sequence, then release/unmap.
	// `ctx` itself is NOT destroyed again here -- this cell's own caller owns ctx's lifetime.
	CHECK(Drain(ctx, seq) == SSLM_OK);
	CHECK(sslm_gpu_seq_release(ctx, seq) == SSLM_OK);
}

// --- Mechanism cell 4 (T-2124, D-SLM3446 P0-3, external review): the adapter-handle analogue of
// M3 above. Design Sec5.2 used to claim "an adapter handle holds no in-flight GPU work," which is
// false on the async submit/finish split -- `sslm_decode_step_gpu` records a command list that
// reads the bound adapter's own resident buffers via root descriptors and returns before the fence
// signals. Fixed the same way M3's own gap was fixed (D-SLM3417's Busy-precedence, mirrored onto
// `SslmGpuAdapterHandle::submitted_sequences`): a sequence left Submitted with an adapter bound, no
// drain, `sslm_gpu_adapter_unmap` attempted against it. Pins the ACTUAL shipping behavior: the
// unmap call returns Busy (not Ok -- which would delete the adapter's own lora_ab_buf/fold_buf
// while the GPU may still be reading them via the in-flight command list); draining the sequence
// then unmapping again succeeds. ---
static void TestDim5_M4_AdapterUnmapWhileSubmittedReturnsBusyThenSucceedsAfterDrain(
    SslmGpuContext* ctx, SslmGpuModelHandle* model, SslmGpuAdapterHandle* adapter) {
	SslmGpuSequenceHandle* seq = nullptr;
	CHECK(sslm_gpu_seq_create(ctx, model, 64, &seq) == SSLM_OK);
	CHECK(sslm_gpu_seq_embed_token(ctx, seq, 5) == SSLM_OK);
	// Submit ONE decode call with `adapter` bound and deliberately do NOT drain it -- `seq` is now
	// Submitted, and `adapter`'s own submitted_sequences count is 1.
	CHECK(sslm_decode_step_gpu(ctx, seq, adapter, 24u) == SSLM_OK);

	CHECK_MSG(sslm_gpu_adapter_unmap(ctx, adapter) == SSLM_BUSY,
	          "M4: sslm_gpu_adapter_unmap against an adapter bound to a Submitted (undrained) "
	          "sequence must return Busy (T-2124, D-SLM3446 P0-3) -- returning Ok here would free "
	          "lora_ab_buf/fold_buf while the GPU may still be reading them via the already-"
	          "recorded, not-yet-fenced command list, a genuine use-after-free/device-removal "
	          "hazard, not a theoretical one");

	// Real cleanup: drain the sequence (the fence now signals, and the adapter's own
	// submitted_sequences count collapses back to 0), THEN the SAME unmap call must succeed --
	// proving Busy was a precondition, not a permanent rejection of this handle.
	CHECK(Drain(ctx, seq) == SSLM_OK);
	CHECK_MSG(sslm_gpu_adapter_unmap(ctx, adapter) == SSLM_OK,
	          "M4: sslm_gpu_adapter_unmap against the SAME adapter, after the sequence that bound "
	          "it has fully drained, must now succeed");
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
	volatile void* addr_4 =
	    (void*)&TestDim5_M4_AdapterUnmapWhileSubmittedReturnsBusyThenSucceedsAfterDrain; (void)addr_4;

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

	// M4 gets its OWN dedicated ctx/model/adapter -- same isolation reasoning as M3 above (it
	// deliberately leaves a sequence Submitted and drives adapter_unmap into a state expected to
	// reject before its own real cleanup).
	if (!g_model_1p5b_path.empty() && !g_adapter_path.empty()) {
		std::vector<uint8_t> adapter_bytes;
		SslmModelView adapter_view{};
		std::string aerr;
		if (LoadRealModel(g_adapter_path, &adapter_view, &adapter_bytes, &aerr)) {
			SslmGpuContext* ctx_m4 = nullptr;
			CHECK(sslm_gpu_context_create(GpuContextConfig{}, &ctx_m4) == SSLM_OK);
			SslmGpuModelHandle* model_m4 = nullptr;
			CHECK(sslm_gpu_model_map(ctx_m4, &view, GpuResidencyConfig{}, &model_m4) == SSLM_OK);
			SslmGpuAdapterHandle* adapter_m4 = nullptr;
			CHECK(sslm_gpu_adapter_map(ctx_m4, model_m4, &adapter_view, &adapter_m4) == SSLM_OK);
			TestDim5_M4_AdapterUnmapWhileSubmittedReturnsBusyThenSucceedsAfterDrain(ctx_m4, model_m4,
			                                                                        adapter_m4);
			// Real cleanup: M4's own body already drained the sequence and unmapped the adapter.
			CHECK(sslm_gpu_model_unmap(ctx_m4, model_m4) == SSLM_OK);
			CHECK(sslm_gpu_context_destroy(ctx_m4) == SSLM_OK);
		} else {
			CHECK_MSG(false, "dim5 M4: --adapter=%s could not be loaded -- %s",
			          g_adapter_path.c_str(), aerr.c_str());
		}
	} else {
		SKIP_MSG("dim5 M4 needs --model1p5b=PATH --adapter=PATH -- not run");
	}

	CHECK(sslm_gpu_context_destroy(ctx) == SSLM_OK);
	std::printf("checks=%d failures=%d skips=%d\n", GChecks, GFailures, GSkips);
	return GFailures ? 1 : 0;
}

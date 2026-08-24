// T-2243 (Curie's red suite design, Sec5.1/Sec6, D-SLM4058) -- S2 (sslm_gpu_seq_bind_adapter,
// plan Sec6.1/Sec10 Phase 2 S2) cells (a), (b), (c), (d), (e), (g), (h), (i), (l), (m), (o).
// Cell (n) lives in cell_rebind_serial.cpp (own two-adapter fork/rebind fixture); cells (f)/(k)
// extend dim8_composition_red.cpp; cell (j) extends dim9_persistence_red.cpp -- per this design's
// own fixture-scaffolding plan (Sec5.1).
//
// Common preamble every cell below shares (Sec6's own header): ctx -> real 1.5B model M -> real
// adapter A (mapped against M) -> [adapter B, when the cell needs two] -> per-cell sequence(s).
// Every setup call is CHECK'd SSLM_OK before a cell's own assertions run.
#include "fixture_common.h"

using namespace superslm;

namespace {

// --- S2-A: bind-time argument guards (plan Sec10 S2(a); red suite Sec6.1) ---
void TestS2_A_BindTimeArgumentGuards(SslmGpuContext* ctx, const SslmModelView* model_view,
                                      const SslmModelView* adapter_view,
                                      SslmGpuModelHandle* model_a, const SslmGpuAdapterHandle* AA) {
	// Leg 1: model mismatch. Map the SAME view a second time -> model_b (a distinct handle).
	SslmGpuModelHandle* model_b = nullptr;
	CHECK(sslm_gpu_model_map(ctx, model_view, GpuResidencyConfig{}, &model_b) == SSLM_OK);
	SslmGpuAdapterHandle* AB = nullptr;
	CHECK(sslm_gpu_adapter_map(ctx, model_b, adapter_view, &AB) == SSLM_OK);  // AB bound to model_b
	SslmGpuSequenceHandle* seq_a = nullptr;
	CHECK(sslm_gpu_seq_create(ctx, model_a, 64, &seq_a) == SSLM_OK);

	CHECK_MSG(sslm_gpu_seq_bind_adapter(ctx, seq_a, AB) == SSLM_ADAPTER_MODEL_MISMATCH,
	          "S2-A leg1: binding an adapter mapped against a DIFFERENT model handle must reject "
	          "SSLM_ADAPTER_MODEL_MISMATCH");
	CHECK_MSG(sslm_gpu_seq_bind_adapter(ctx, seq_a, AA) == SSLM_OK,
	          "S2-A leg1 control: binding the adapter mapped against seq_a own model must "
	          "succeed -- proves the rejection above is the comparison, not the verb");
	CHECK(*SslmGpuSequenceHandleBoundAdapterForBench(seq_a) == AA);
	CHECK(sslm_gpu_seq_bind_adapter(ctx, seq_a, nullptr) == SSLM_OK);  // unbind for cleanup

	CHECK(sslm_gpu_seq_release(ctx, seq_a) == SSLM_OK);
	CHECK(sslm_gpu_adapter_unmap(ctx, AB) == SSLM_OK);
	CHECK(sslm_gpu_model_unmap(ctx, model_b) == SSLM_OK);

	// Leg 2: foreign-context adapter (documented limitation -- the deterministic public outcome
	// is ADAPTER_MODEL_MISMATCH, since a cross-context adapter necessarily carries a different
	// ->model pointer too; the pure ctx-mismatch arm is pinned by cell (h) instead).
	if (g_model_0p5b_path.empty()) {
		SKIP_MSG("S2-A leg2 needs --model0p5b=PATH -- not run");
		return;
	}
	std::vector<uint8_t> bytes05;
	SslmModelView view05{};
	std::string err;
	if (!LoadRealModel(g_model_0p5b_path, &view05, &bytes05, &err)) {
		SKIP_MSG("S2-A leg2: could not load --model0p5b=%s (%s)", g_model_0p5b_path.c_str(), err.c_str());
		return;
	}
	SslmGpuContext* ctx2 = nullptr;
	CHECK(sslm_gpu_context_create(GpuContextConfig{}, &ctx2) == SSLM_OK);
	SslmGpuModelHandle* model05 = nullptr;
	CHECK(sslm_gpu_model_map(ctx2, &view05, GpuResidencyConfig{}, &model05) == SSLM_OK);
	// This leg needs a real adapter whose own base_artifact_hash matches model05 -- no such
	// artifact is supplied this session (--model0p5b names a BASE model, not an adapter shaped
	// for it); a genuine SKIP of this leg's product half, matching this suite's own established
	// convention (fixture_common.h:10-16), not a construction this cell can pass a non-adapter
	// view into and expect to succeed.
	SslmGpuAdapterHandle* foreign_adapter = nullptr;
	const SslmGpuStatus foreign_map_status = sslm_gpu_adapter_map(ctx2, model05, &view05, &foreign_adapter);
	if (foreign_map_status != SSLM_OK || !foreign_adapter) {
		SKIP_MSG("S2-A leg2: no real adapter artifact shaped for the 0.5B model is available -- "
		         "adapter_map status %d -- product half not run", (int)foreign_map_status);
		CHECK(sslm_gpu_model_unmap(ctx2, model05) == SSLM_OK);
		CHECK(sslm_gpu_context_destroy(ctx2) == SSLM_OK);
		return;
	}

	SslmGpuSequenceHandle* seq_a2 = nullptr;
	CHECK(sslm_gpu_seq_create(ctx, model_a, 64, &seq_a2) == SSLM_OK);
	CHECK_MSG(sslm_gpu_seq_bind_adapter(ctx, seq_a2, foreign_adapter) == SSLM_ADAPTER_MODEL_MISMATCH,
	          "S2-A leg2: a foreign-context adapter (different model AND different ctx) must "
	          "reject through the model-mismatch arm, per the verb own rule ordering (model "
	          "checked before ctx)");
	CHECK(sslm_gpu_seq_release(ctx, seq_a2) == SSLM_OK);

	CHECK(sslm_gpu_adapter_unmap(ctx2, foreign_adapter) == SSLM_OK);
	CHECK(sslm_gpu_model_unmap(ctx2, model05) == SSLM_OK);
	CHECK(sslm_gpu_context_destroy(ctx2) == SSLM_OK);
}

// --- S2-B: mid-token rejection, corrected predicate (plan Sec10 S2(b); red suite Sec6.2) ---
void TestS2_B_MidTokenRejection(SslmGpuContext* ctx, SslmGpuModelHandle* model,
                                 const SslmGpuAdapterHandle* A, uint32_t num_hidden_layers) {
	// Control row: a sub-floor budget cannot stand in for a mid-token state (floor-division).
	{
		SslmGpuSequenceHandle* seq = nullptr;
		CHECK(sslm_gpu_seq_create(ctx, model, 64, &seq) == SSLM_OK);
		CHECK(sslm_gpu_seq_embed_token(ctx, seq, 5) == SSLM_OK);
		CHECK_MSG(sslm_decode_step_gpu(ctx, seq, nullptr, /*budget=*/1u) == SSLM_DISPATCH_BUDGET_TOO_SMALL,
		          "S2-B control: a sub-floor dispatch_budget must reject DispatchBudgetTooSmall");
		CHECK(*SslmGpuSeqHandleLayerIndexForBench(seq) == 0);
		CHECK(sslm_gpu_seq_release(ctx, seq) == SSLM_OK);
	}

	// The mid-token cell proper: issue exactly 3 layers (kDispatchesPerLayer*3), never drained
	// past that -- a real, verified mid-token rest.
	SslmGpuSequenceHandle* seq = nullptr;
	CHECK(sslm_gpu_seq_create(ctx, model, 64, &seq) == SSLM_OK);
	CHECK(sslm_gpu_seq_embed_token(ctx, seq, 5) == SSLM_OK);
	CHECK(sslm_decode_step_gpu(ctx, seq, nullptr, kDispatchesPerLayer * 3u) == SSLM_OK);
	CHECK(Drain(ctx, seq) == SSLM_OK);
	CHECK_MSG(*SslmGpuSeqHandleLayerIndexForBench(seq) == 3,
	          "S2-B: setup precondition -- the fixture must actually be mid-token (layer_index==3) "
	          "before the bind under test is evaluated");

	CHECK_MSG(sslm_gpu_seq_bind_adapter(ctx, seq, A) == SSLM_BUSY,
	          "S2-B: bind must reject SSLM_BUSY while 0 < layer_index < num_hidden_layers");
	// Sequence still usable afterwards.
	CHECK(sslm_gpu_seq_reset(ctx, seq) == SSLM_OK);
	CHECK(sslm_gpu_seq_embed_token(ctx, seq, 5) == SSLM_OK);
	CHECK(sslm_gpu_seq_release(ctx, seq) == SSLM_OK);
	(void)num_hidden_layers;
}

// --- S2-C: rebind moves the counter (plan Sec10 S2(c); red suite Sec6.3) ---
void TestS2_C_RebindMovesCounter(SslmGpuContext* ctx, SslmGpuModelHandle* model,
                                  const SslmGpuAdapterHandle* A, const SslmGpuAdapterHandle* B) {
	if (!B) {
		SKIP_MSG("S2-C needs --adapter2=PATH -- not run");
		return;
	}
	SslmGpuSequenceHandle* seq = nullptr;
	CHECK(sslm_gpu_seq_create(ctx, model, 64, &seq) == SSLM_OK);
	CHECK(sslm_gpu_seq_bind_adapter(ctx, seq, A) == SSLM_OK);
	CHECK(sslm_gpu_seq_bind_adapter(ctx, seq, B) == SSLM_OK);

	CHECK_MSG(*SslmGpuAdapterHandleBoundSequencesForBench(const_cast<SslmGpuAdapterHandle*>(A)) == 0,
	          "S2-C: rebind must decrement the OLD adapter bound_sequences");
	CHECK_MSG(*SslmGpuAdapterHandleBoundSequencesForBench(const_cast<SslmGpuAdapterHandle*>(B)) == 1,
	          "S2-C: rebind must increment the NEW adapter bound_sequences");

	CHECK_MSG(sslm_gpu_adapter_unmap(ctx, const_cast<SslmGpuAdapterHandle*>(B)) ==
	          SSLM_ADAPTER_HAS_BOUND_SEQUENCES,
	          "S2-C: B still holds the bind -- unmap must reject");
	// A own count reached 0 via the rebind -- but A is a shared fixture handle owned by
	// main(), never unmapped here; the accessor check above already proved the count.

	CHECK(sslm_gpu_seq_bind_adapter(ctx, seq, nullptr) == SSLM_OK);  // unbind B
	CHECK(sslm_gpu_seq_release(ctx, seq) == SSLM_OK);
}

// --- S2-D: release decrements (plan Sec10 S2(d); red suite Sec6.4) ---
void TestS2_D_ReleaseDecrements(SslmGpuContext* ctx, SslmGpuModelHandle* model,
                                 const SslmGpuAdapterHandle* A) {
	SslmGpuSequenceHandle* seq = nullptr;
	CHECK(sslm_gpu_seq_create(ctx, model, 64, &seq) == SSLM_OK);
	CHECK(sslm_gpu_seq_bind_adapter(ctx, seq, A) == SSLM_OK);
	CHECK(sslm_gpu_seq_release(ctx, seq) == SSLM_OK);  // never explicitly unbound

	CHECK_MSG(*SslmGpuAdapterHandleBoundSequencesForBench(const_cast<SslmGpuAdapterHandle*>(A)) == 0,
	          "S2-D: sslm_gpu_seq_release must decrement its adapter bound_sequences even when "
	          "the caller never explicitly unbound first");
}

// --- S2-E: unmap blocked while bound (plan Sec10 S2(e); red suite Sec6.5) ---
void TestS2_E_UnmapBlockedWhileBound(SslmGpuContext* ctx, SslmGpuModelHandle* model,
                                      const SslmGpuAdapterHandle* A_view_adapter_artifact,
                                      const SslmModelView* model_view) {
	// Needs a DEDICATED adapter (unmapped inside this cell, unlike the shared fixture adapters).
	SslmGpuAdapterHandle* A2 = nullptr;
	CHECK(sslm_gpu_adapter_map(ctx, model, model_view, &A2) == SSLM_OK);
	SslmGpuSequenceHandle* seq = nullptr;
	CHECK(sslm_gpu_seq_create(ctx, model, 64, &seq) == SSLM_OK);
	CHECK(sslm_gpu_seq_bind_adapter(ctx, seq, A2) == SSLM_OK);

	CHECK_MSG(sslm_gpu_adapter_unmap(ctx, A2) == SSLM_ADAPTER_HAS_BOUND_SEQUENCES,
	          "S2-E: adapter_unmap with bound_sequences > 0 must reject "
	          "SSLM_ADAPTER_HAS_BOUND_SEQUENCES, not SSLM_BUSY");
	// Adapter still alive: a subsequent unbind succeeds (proves the rejected unmap did not
	// free the handle).
	CHECK(sslm_gpu_seq_bind_adapter(ctx, seq, nullptr) == SSLM_OK);
	CHECK(sslm_gpu_adapter_unmap(ctx, A2) == SSLM_OK);
	CHECK(sslm_gpu_seq_release(ctx, seq) == SSLM_OK);
	(void)A_view_adapter_artifact;
}

// --- S2-G: the guard releases -- unblock after unbind (plan Sec10 S2(g), Mendeleev F-1;
// red suite Sec6.7) ---
void TestS2_G_UnblockAfterUnbind(SslmGpuContext* ctx, SslmGpuModelHandle* model,
                                  const SslmModelView* model_view) {
	SslmGpuAdapterHandle* A2 = nullptr;
	CHECK(sslm_gpu_adapter_map(ctx, model, model_view, &A2) == SSLM_OK);
	SslmGpuSequenceHandle* seq = nullptr;
	CHECK(sslm_gpu_seq_create(ctx, model, 64, &seq) == SSLM_OK);
	CHECK(sslm_gpu_seq_bind_adapter(ctx, seq, A2) == SSLM_OK);
	CHECK(sslm_gpu_adapter_unmap(ctx, A2) == SSLM_ADAPTER_HAS_BOUND_SEQUENCES);  // setup, reused

	CHECK(sslm_gpu_seq_bind_adapter(ctx, seq, nullptr) == SSLM_OK);
	CHECK(*SslmGpuAdapterHandleBoundSequencesForBench(A2) == 0);
	CHECK_MSG(sslm_gpu_adapter_unmap(ctx, A2) == SSLM_OK,
	          "S2-G: after every binding sequence unbinds, the guard must actually release, not "
	          "merely block while nonzero");
	CHECK(sslm_gpu_seq_release(ctx, seq) == SSLM_OK);

	// Release-route arm: a fresh sequence bound to a second adapter, released (no explicit
	// unbind) -- unmap must still succeed.
	SslmGpuAdapterHandle* A3 = nullptr;
	CHECK(sslm_gpu_adapter_map(ctx, model, model_view, &A3) == SSLM_OK);
	SslmGpuSequenceHandle* seq2 = nullptr;
	CHECK(sslm_gpu_seq_create(ctx, model, 64, &seq2) == SSLM_OK);
	CHECK(sslm_gpu_seq_bind_adapter(ctx, seq2, A3) == SSLM_OK);
	CHECK(sslm_gpu_seq_release(ctx, seq2) == SSLM_OK);
	CHECK(sslm_gpu_adapter_unmap(ctx, A3) == SSLM_OK);
}

// --- S2-H: malformed-handle boilerplate (plan Sec10 S2(h); red suite Sec6.8) ---
void TestS2_H_MalformedHandleBoilerplate(SslmGpuContext* ctx, SslmGpuModelHandle* model,
                                          const SslmGpuAdapterHandle* A) {
	SslmGpuSequenceHandle* seq_a = nullptr;
	CHECK(sslm_gpu_seq_create(ctx, model, 64, &seq_a) == SSLM_OK);

	CHECK_MSG(sslm_gpu_seq_bind_adapter(nullptr, seq_a, A) == SSLM_SEQUENCE_KV_BUFFER_MISMATCH,
	          "S2-H row1: null ctx must reject SSLM_SEQUENCE_KV_BUFFER_MISMATCH");
	CHECK_MSG(sslm_gpu_seq_bind_adapter(ctx, nullptr, A) == SSLM_SEQUENCE_KV_BUFFER_MISMATCH,
	          "S2-H row2: null seq must reject SSLM_SEQUENCE_KV_BUFFER_MISMATCH");

	SslmGpuContext* ctx_b = nullptr;
	CHECK(sslm_gpu_context_create(GpuContextConfig{}, &ctx_b) == SSLM_OK);
	CHECK_MSG(sslm_gpu_seq_bind_adapter(ctx_b, seq_a, A) == SSLM_SEQUENCE_KV_BUFFER_MISMATCH,
	          "S2-H row3: a seq handle from a DIFFERENT context must reject "
	          "SSLM_SEQUENCE_KV_BUFFER_MISMATCH");
	CHECK(sslm_gpu_context_destroy(ctx_b) == SSLM_OK);

	CHECK(sslm_gpu_seq_release(ctx, seq_a) == SSLM_OK);
}

// --- S2-I: idempotent double-bind (plan Sec10 S2(i); red suite Sec6.9) ---
void TestS2_I_IdempotentDoubleBind(SslmGpuContext* ctx, SslmGpuModelHandle* model,
                                    const SslmModelView* model_view) {
	SslmGpuAdapterHandle* A2 = nullptr;
	CHECK(sslm_gpu_adapter_map(ctx, model, model_view, &A2) == SSLM_OK);
	SslmGpuSequenceHandle* seq = nullptr;
	CHECK(sslm_gpu_seq_create(ctx, model, 64, &seq) == SSLM_OK);

	CHECK(sslm_gpu_seq_bind_adapter(ctx, seq, A2) == SSLM_OK);
	CHECK_MSG(sslm_gpu_seq_bind_adapter(ctx, seq, A2) == SSLM_OK,
	          "S2-I: binding the SAME adapter a second time must return SSLM_OK (idempotent "
	          "success, not an error)");

	CHECK(sslm_gpu_seq_bind_adapter(ctx, seq, nullptr) == SSLM_OK);  // ONE unbind
	CHECK_MSG(sslm_gpu_adapter_unmap(ctx, A2) == SSLM_OK,
	          "S2-I: a double-increment bug would leave the count at 1 after one unbind and "
	          "reject this unmap -- observed count must be exactly 1 bind worth");
	CHECK(sslm_gpu_seq_release(ctx, seq) == SSLM_OK);
}

// --- S2-L: admission after granular full-depth drained rest (plan Sec10 S2(l);
// red suite Sec6.12) ---
void TestS2_L_AdmissionAfterGranularFullDepthDrainedRest(SslmGpuContext* ctx,
                                                          SslmGpuModelHandle* model,
                                                          const SslmGpuAdapterHandle* A,
                                                          uint32_t num_hidden_layers) {
	SslmGpuSequenceHandle* seq = nullptr;
	CHECK(sslm_gpu_seq_create(ctx, model, 64, &seq) == SSLM_OK);
	CHECK(sslm_gpu_seq_embed_token(ctx, seq, 5) == SSLM_OK);
	uint32_t guard = 0;
	while (*SslmGpuSeqHandleLayerIndexForBench(seq) < num_hidden_layers) {
		CHECK(sslm_decode_step_gpu(ctx, seq, nullptr, kDispatchesPerLayer) == SSLM_OK);
		CHECK(Drain(ctx, seq) == SSLM_OK);
		if (++guard > 200) break;
	}
	CHECK_MSG(*SslmGpuSeqHandleLayerIndexForBench(seq) == num_hidden_layers,
	          "S2-L: setup precondition -- must actually reach full depth, granular, undrained "
	          "past that (the natural rest of a caller driving the granular 1.0 surface)");

	CHECK_MSG(sslm_gpu_seq_bind_adapter(ctx, seq, A) == SSLM_OK,
	          "S2-L: bind must ADMIT from the granular path own drained-at-full-depth rest -- "
	          "the fractured predicate rejected this state");
	CHECK(*SslmGpuSequenceHandleBoundAdapterForBench(seq) == A);
	CHECK(sslm_gpu_seq_bind_adapter(ctx, seq, nullptr) == SSLM_OK);
	CHECK(sslm_gpu_seq_release(ctx, seq) == SSLM_OK);
}

// --- S2-M: admission after the chunk-prefill choke point (plan Sec10 S2(m);
// red suite Sec6.13) ---
void TestS2_M_AdmissionAfterChunkPrefillChokePoint(SslmGpuContext* ctx, SslmGpuModelHandle* model,
                                                    const SslmGpuAdapterHandle* A,
                                                    uint32_t num_hidden_layers) {
	SslmGpuSequenceHandle* seq = nullptr;
	CHECK(sslm_gpu_seq_create(ctx, model, 64, &seq) == SSLM_OK);
	const int32_t prompt[] = {5, 6, 7};
	CHECK(SslmGpuSeqPrefillPromptForG5Bridge(ctx, seq, prompt, 3, kDispatchesPerLayer) == SSLM_OK);
	CHECK(*SslmGpuSeqHandleLayerIndexForBench(seq) == num_hidden_layers);

	CHECK_MSG(sslm_gpu_seq_bind_adapter(ctx, seq, A) == SSLM_OK,
	          "S2-M: bind must ADMIT from the state every chunk-prefill caller is left in -- the "
	          "single most load-bearing admission case (the S2 remedy own chunk choke point "
	          "leaves every caller exactly here)");
	CHECK(*SslmGpuSequenceHandleBoundAdapterForBench(seq) == A);
	CHECK(sslm_gpu_seq_bind_adapter(ctx, seq, nullptr) == SSLM_OK);
	CHECK(sslm_gpu_seq_release(ctx, seq) == SSLM_OK);
}

// --- S2-O: reset preserves the binding (plan Sec10 S2(o); red suite Sec6.15) ---
void TestS2_O_ResetPreservesBinding(SslmGpuContext* ctx, SslmGpuModelHandle* model,
                                     const SslmModelView* model_view) {
	SslmGpuAdapterHandle* A2 = nullptr;
	CHECK(sslm_gpu_adapter_map(ctx, model, model_view, &A2) == SSLM_OK);
	SslmGpuSequenceHandle* seq = nullptr;
	CHECK(sslm_gpu_seq_create(ctx, model, 64, &seq) == SSLM_OK);
	const int32_t prompt[] = {5, 6};
	CHECK(SslmGpuSeqPrefillPromptForG5Bridge(ctx, seq, prompt, 2, kDispatchesPerLayer) == SSLM_OK);
	CHECK(sslm_gpu_seq_bind_adapter(ctx, seq, A2) == SSLM_OK);

	CHECK_MSG(sslm_gpu_seq_reset(ctx, seq) == SSLM_OK, "S2-O: reset itself must still succeed");
	CHECK_MSG(sslm_gpu_adapter_unmap(ctx, A2) == SSLM_ADAPTER_HAS_BOUND_SEQUENCES,
	          "S2-O: the binding must SURVIVE reset -- matching the bound_schema_index precedent "
	          "(a caller own standing configuration, not token-generation content)");
	CHECK(*SslmGpuSequenceHandleBoundAdapterForBench(seq) == A2);

	CHECK(sslm_gpu_seq_bind_adapter(ctx, seq, nullptr) == SSLM_OK);
	CHECK(sslm_gpu_adapter_unmap(ctx, A2) == SSLM_OK);
	CHECK(sslm_gpu_seq_release(ctx, seq) == SSLM_OK);
}

}  // namespace

int main(int argc, char** argv) {
	std::setvbuf(stdout, nullptr, _IONBF, 0);  // unbuffered -- a crash mid-cell must not lose
	                                            // the transcript already printed before it
	ParseFixtureArgs(argc, argv);

	if (g_model_1p5b_path.empty()) {
		SKIP_MSG("s2_bind_red needs --model1p5b=PATH -- suite not run");
		std::printf("checks=%d failures=%d skips=%d\n", GChecks, GFailures, GSkips);
		return GFailures ? 1 : 0;
	}
	if (g_adapter_path.empty()) {
		SKIP_MSG("s2_bind_red needs --adapter=PATH -- suite not run");
		std::printf("checks=%d failures=%d skips=%d\n", GChecks, GFailures, GSkips);
		return GFailures ? 1 : 0;
	}

	std::vector<uint8_t> mbytes;
	SslmModelView mview{};
	std::string err;
	if (!LoadRealModel(g_model_1p5b_path, &mview, &mbytes, &err)) {
		std::printf("FATAL: could not load --model1p5b=%s (%s)\n", g_model_1p5b_path.c_str(), err.c_str());
		return 2;
	}
	std::printf("s2_bind_red: resolved model path %s\n", g_model_1p5b_path.c_str());

	std::vector<uint8_t> abytes;
	SslmModelView aview{};
	if (!LoadRealModel(g_adapter_path, &aview, &abytes, &err)) {
		std::printf("FATAL: could not load --adapter=%s (%s)\n", g_adapter_path.c_str(), err.c_str());
		return 2;
	}
	std::printf("s2_bind_red: resolved adapter path %s\n", g_adapter_path.c_str());

	std::vector<uint8_t> bbytes;
	SslmModelView bview{};
	bool have_b = false;
	if (!g_adapter2_path.empty()) {
		have_b = LoadRealModel(g_adapter2_path, &bview, &bbytes, &err);
		if (!have_b) {
			std::printf("s2_bind_red: could not load --adapter2=%s (%s) -- two-adapter cells SKIP\n",
			            g_adapter2_path.c_str(), err.c_str());
		} else {
			std::printf("s2_bind_red: resolved adapter2 path %s\n", g_adapter2_path.c_str());
		}
	}

	SslmGpuContext* ctx = nullptr;
	CHECK(sslm_gpu_context_create(GpuContextConfig{}, &ctx) == SSLM_OK);
	SslmGpuModelHandle* model = nullptr;
	CHECK(sslm_gpu_model_map(ctx, &mview, GpuResidencyConfig{}, &model) == SSLM_OK);
	SslmGpuAdapterHandle* A = nullptr;
	CHECK(sslm_gpu_adapter_map(ctx, model, &aview, &A) == SSLM_OK);
	SslmGpuAdapterHandle* B = nullptr;
	if (have_b) {
		CHECK(sslm_gpu_adapter_map(ctx, model, &bview, &B) == SSLM_OK);
	}
	const uint32_t num_hidden_layers = mview.config.num_hidden_layers;

	if (ctx && model && A) {
		std::printf("cell S2-A\n");
		TestS2_A_BindTimeArgumentGuards(ctx, &mview, &aview, model, A);
		std::printf("cell S2-B\n");
		TestS2_B_MidTokenRejection(ctx, model, A, num_hidden_layers);
		std::printf("cell S2-C\n");
		TestS2_C_RebindMovesCounter(ctx, model, A, B);
		std::printf("cell S2-D\n");
		TestS2_D_ReleaseDecrements(ctx, model, A);
		std::printf("cell S2-E\n");
		TestS2_E_UnmapBlockedWhileBound(ctx, model, A, &aview);
		std::printf("cell S2-G\n");
		TestS2_G_UnblockAfterUnbind(ctx, model, &aview);
		std::printf("cell S2-H\n");
		TestS2_H_MalformedHandleBoilerplate(ctx, model, A);
		std::printf("cell S2-I\n");
		TestS2_I_IdempotentDoubleBind(ctx, model, &aview);
		std::printf("cell S2-L\n");
		TestS2_L_AdmissionAfterGranularFullDepthDrainedRest(ctx, model, A, num_hidden_layers);
		std::printf("cell S2-M\n");
		TestS2_M_AdmissionAfterChunkPrefillChokePoint(ctx, model, A, num_hidden_layers);
		std::printf("cell S2-O\n");
		TestS2_O_ResetPreservesBinding(ctx, model, &aview);
		std::printf("all S2 cells returned\n");
	}

	if (B) CHECK(sslm_gpu_adapter_unmap(ctx, B) == SSLM_OK);
	if (A) CHECK(sslm_gpu_adapter_unmap(ctx, A) == SSLM_OK);
	if (model) CHECK(sslm_gpu_model_unmap(ctx, model) == SSLM_OK);
	CHECK(sslm_gpu_context_destroy(ctx) == SSLM_OK);

	std::printf("checks=%d failures=%d skips=%d\n", GChecks, GFailures, GSkips);
	return GFailures ? 1 : 0;
}

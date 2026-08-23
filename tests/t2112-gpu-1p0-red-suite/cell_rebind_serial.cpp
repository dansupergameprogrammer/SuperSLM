// T-2243 (S2 cell (n), plan Sec6.1/Sec10 Phase 2 S2(n) repaired D-SLM4030, red suite Sec6.14,
// D-SLM4058) -- the campaign's one achievement claim, strongest form: a caller decodes token 1
// under adapter A through the recommended one-call bridge, rebinds to adapter B on the SAME live
// sequence, and decodes token 2 under B -- bit-for-bit identical to applying B through the
// pre-existing per-call adapter_or_null argument over identical history. This is the use case the
// whole S2 remedy exists to serve (runtime-lora-serial-ue-tooling); under the fractured predicate
// it cost an unwanted token or the whole K/V state.
//
// Own file (not folded into s2_bind_red.cpp) because it needs TWO real adapters and its own
// dedicated cleanup chain (Sec5.1's own fixture-scaffolding plan).
#include "fixture_common.h"

using namespace superslm;

int main(int argc, char** argv) {
	ParseFixtureArgs(argc, argv);

	if (g_model_1p5b_path.empty() || g_adapter_path.empty() || g_adapter2_path.empty()) {
		SKIP_MSG("cell_rebind_serial needs --model1p5b=PATH --adapter=PATH --adapter2=PATH -- "
		         "not run");
		std::printf("checks=%d failures=%d skips=%d\n", GChecks, GFailures, GSkips);
		return GFailures ? 1 : 0;
	}

	std::vector<uint8_t> mbytes, abytes, bbytes;
	SslmModelView mview{}, aview{}, bview{};
	std::string err;
	if (!LoadRealModel(g_model_1p5b_path, &mview, &mbytes, &err)) {
		std::printf("FATAL: could not load --model1p5b=%s (%s)\n", g_model_1p5b_path.c_str(), err.c_str());
		return 2;
	}
	std::printf("cell_rebind_serial: resolved model path %s\n", g_model_1p5b_path.c_str());
	if (!LoadRealModel(g_adapter_path, &aview, &abytes, &err)) {
		std::printf("FATAL: could not load --adapter=%s (%s)\n", g_adapter_path.c_str(), err.c_str());
		return 2;
	}
	std::printf("cell_rebind_serial: resolved adapter A path %s\n", g_adapter_path.c_str());
	if (!LoadRealModel(g_adapter2_path, &bview, &bbytes, &err)) {
		std::printf("FATAL: could not load --adapter2=%s (%s)\n", g_adapter2_path.c_str(), err.c_str());
		return 2;
	}
	std::printf("cell_rebind_serial: resolved adapter B path %s\n", g_adapter2_path.c_str());

	SslmGpuContext* ctx = nullptr;
	CHECK(sslm_gpu_context_create(GpuContextConfig{}, &ctx) == SSLM_OK);
	SslmGpuModelHandle* model = nullptr;
	CHECK(sslm_gpu_model_map(ctx, &mview, GpuResidencyConfig{}, &model) == SSLM_OK);
	SslmGpuAdapterHandle* A = nullptr;
	CHECK(sslm_gpu_adapter_map(ctx, model, &aview, &A) == SSLM_OK);
	SslmGpuAdapterHandle* B = nullptr;
	CHECK(sslm_gpu_adapter_map(ctx, model, &bview, &B) == SSLM_OK);
	const uint32_t num_hidden_layers = mview.config.num_hidden_layers;

	// D-SLM4057's own prediction, mandatory: a fixed second-token id shared by both arms, and NO
	// prefill anywhere before the first bridge call (a prior prefill would set ready_for_logits
	// and make the first bridge call consume the shortcut instead of embedding under A).
	constexpr int32_t kFirstToken = 5;
	constexpr int32_t kSecondToken = 9;

	if (ctx && model && A && B) {
		// 1-2. seq_p starts EMPTY; bind A; decode token 1 through the recommended one-call
		// bridge -- embeds token 1 under A, drives to full depth, finishes.
		SslmGpuSequenceHandle* seq_p = nullptr;
		CHECK(sslm_gpu_seq_create(ctx, model, 64, &seq_p) == SSLM_OK);
		CHECK(sslm_gpu_seq_bind_adapter(ctx, seq_p, A) == SSLM_OK);
		int32_t tok1 = -1;
		CHECK_MSG(SslmGpuSeqDecodeStepForG5Bridge(ctx, seq_p, kFirstToken,
		                                          FullTokenBudget(num_hidden_layers), &tok1) == SSLM_OK,
		          "S2-N: primary's own first bridge-composed token, under A");
		CHECK_MSG(*SslmGpuSeqHandleLayerIndexForBench(seq_p) == 0,
		          "S2-N: after one bridge-completed token, the primary rests at layer_index==0 -- "
		          "the admission state the rebind under test pins");

		// 4. Fork: two-call save -> restore into a fresh handle (the clone inherits the
		// primary's REAL token-1 K/V history, computed under A).
		size_t required_size = 0;
		{
			uint8_t probe = 0;
			CHECK(sslm_gpu_seq_save(ctx, seq_p, &probe, &required_size) != SSLM_OK);
			CHECK(required_size > 0);
		}
		std::vector<uint8_t> blob(required_size);
		size_t blob_size = blob.size();
		CHECK(sslm_gpu_seq_save(ctx, seq_p, blob.data(), &blob_size) == SSLM_OK);
		SslmGpuSequenceHandle* seq_c = nullptr;
		CHECK_MSG(sslm_gpu_seq_restore(ctx, model, blob.data(), blob_size, &seq_c) == SSLM_OK,
		          "S2-N: fork -- restore into a fresh clone handle");
		CHECK(seq_c != nullptr);
		CHECK(*SslmGpuSequenceHandleBoundAdapterForBench(seq_c) == nullptr);  // cell J's own claim

		// 5. Rebind the primary to B -- admission from the post-token rest.
		CHECK_MSG(sslm_gpu_seq_bind_adapter(ctx, seq_p, B) == SSLM_OK,
		          "S2-N: rebind to B must admit from the state one bridge-completed token leaves "
		          "the primary in");

		// 6. Primary, token 2: through the bridge, unchanged -- composes embed+drive+finish
		// under B via bound_adapter indirection.
		int32_t tok2_p = -1;
		CHECK_MSG(SslmGpuSeqDecodeStepForG5Bridge(ctx, seq_p, kSecondToken,
		                                          FullTokenBudget(num_hidden_layers), &tok2_p) == SSLM_OK,
		          "S2-N: primary's own second bridge-composed token, under B via bound_adapter");

		// 7. Clone, token 2: hand-composed, B EXPLICIT through the pre-existing per-call
		// argument -- the genuinely independent reference this cell's own claim needs.
		CHECK(sslm_gpu_seq_embed_token(ctx, seq_c, kSecondToken) == SSLM_OK);
		uint32_t guard = 0;
		while (*SslmGpuSeqHandleLayerIndexForBench(seq_c) < num_hidden_layers) {
			CHECK(sslm_decode_step_gpu(ctx, seq_c, B, kDispatchesPerLayer) == SSLM_OK);
			CHECK(Drain(ctx, seq_c) == SSLM_OK);
			if (++guard > 200) break;
		}
		int32_t tok2_c = -1;
		CHECK_MSG(SslmGpuSeqFinishTokenForG5Bridge(ctx, seq_c, &tok2_c) == SSLM_OK,
		          "S2-N: clone's own hand-composed second token, B passed explicitly");

		// 8. The product claim: bit-for-bit identical.
		CHECK_MSG(tok2_p == tok2_c,
		          "S2-N: the primary's rebind-path second token must be bit-for-bit identical to "
		          "the clone's direct-argument second token -- tok2_p=%d tok2_c=%d", tok2_p, tok2_c);
		SeqSnapshot snap_p{}, snap_c{};
		CHECK(CaptureSnapshot(seq_p, &snap_p));
		CHECK(CaptureSnapshot(seq_c, &snap_c));
		CHECK_MSG(SnapshotsBitEqual(snap_p, snap_c),
		          "S2-N: post-finish snapshots (hidden_codes/scale/kv_saturation/context_length) "
		          "must be bit-equal on both arms");

		CHECK(sslm_gpu_seq_bind_adapter(ctx, seq_p, nullptr) == SSLM_OK);
		CHECK(sslm_gpu_seq_release(ctx, seq_p) == SSLM_OK);
		CHECK(sslm_gpu_seq_release(ctx, seq_c) == SSLM_OK);
	}

	if (B) CHECK(sslm_gpu_adapter_unmap(ctx, B) == SSLM_OK);
	if (A) CHECK(sslm_gpu_adapter_unmap(ctx, A) == SSLM_OK);
	if (model) CHECK(sslm_gpu_model_unmap(ctx, model) == SSLM_OK);
	CHECK(sslm_gpu_context_destroy(ctx) == SSLM_OK);

	std::printf("checks=%d failures=%d skips=%d\n", GChecks, GFailures, GSkips);
	return GFailures ? 1 : 0;
}

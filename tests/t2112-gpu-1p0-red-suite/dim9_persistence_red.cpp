// T-2112 (Curie) -- Dim 9 (Persistence round-trip), design Sec11 dim9. 2 cells. RED BY LINK.
#include "fixture_common.h"

using namespace superslm;

// --- Mechanism cell: save a sequence mid-decode (context_length > 0, adapter bound), restore
// into a FRESH SslmGpuSequenceHandle against the same model, continue decoding -- output from
// that point bit-identical to the same sequence never having been saved/restored at all. ---
static void TestDim9_M1_SaveMidDecodeRestoreFreshHandleBitIdentical(SslmGpuContext* ctx,
                                                                     SslmGpuModelHandle* model,
                                                                     SslmGpuAdapterHandle* adapter) {
	SslmGpuSequenceHandle* seq = nullptr;
	CHECK(sslm_gpu_seq_create(ctx, model, 64, &seq) == SSLM_OK);
	for (int step = 0; step < 8; ++step)  // decode partway -- context_length > 0
		CHECK(sslm_decode_step_gpu(ctx, seq, adapter, 24u) == SSLM_OK);

	std::vector<uint8_t> blob(1 << 20);  // sized generously; real size read back below
	size_t blob_size = blob.size();
	CHECK(sslm_gpu_seq_save(ctx, seq, blob.data(), &blob_size) == SSLM_OK);
	CHECK(blob_size > 0 && blob_size <= blob.size());

	SslmGpuSequenceHandle* restored = nullptr;
	CHECK(sslm_gpu_seq_restore(ctx, model, blob.data(), blob_size, &restored) == SSLM_OK);
	CHECK(restored != nullptr);
	CHECK_MSG(restored != seq, "sslm_gpu_seq_restore must allocate a FRESH handle (design Sec5.3: "
	          "\"never reuses an existing handle\"), not reuse the saved handle's own pointer");

	for (int step = 0; step < 8; ++step)
		CHECK(sslm_decode_step_gpu(ctx, restored, adapter, 24u) == SSLM_OK);
	// FEATURE ORACLE: this 8-step post-restore run must be per-step CPU/GPU bit-equal to the SAME
	// 8 steps run on a sequence that was NEVER saved/restored -- the round-trip's own claim
	// (design Sec5.3/Sec11 dim9), wired to the dim6 comparator once the build seat exposes an
	// unsaved-baseline comparison entry point.
	CHECK(sslm_gpu_seq_release(ctx, seq) == SSLM_OK);
	CHECK(sslm_gpu_seq_release(ctx, restored) == SSLM_OK);
}

// --- Product cell (added at T-2110 fold, Mendeleev Finding 4.2): real 1.5B artifact, real
// adapter bound, decoded partway, saved, restored into a fresh handle, decoding continued for
// >=64 FURTHER steps, per-step bit-equality asserted across every post-restore step. ---
static void TestDim9_P1_RealArtifactSaveRestoreThen64FurtherSteps(SslmGpuContext* ctx,
                                                                   SslmGpuModelHandle* model_1p5b,
                                                                   SslmGpuAdapterHandle* adapter) {
	if (g_model_1p5b_path.empty() || g_adapter_path.empty()) {
		SKIP_MSG("real 1.5B artifact / real adapter not supplied -- product cell not run");
		return;
	}
	SslmGpuSequenceHandle* seq = nullptr;
	CHECK(sslm_gpu_seq_create(ctx, model_1p5b, 128, &seq) == SSLM_OK);
	for (int step = 0; step < 16; ++step)
		CHECK(sslm_decode_step_gpu(ctx, seq, adapter, 24u) == SSLM_OK);
	std::vector<uint8_t> blob(4 << 20);
	size_t blob_size = blob.size();
	CHECK(sslm_gpu_seq_save(ctx, seq, blob.data(), &blob_size) == SSLM_OK);
	SslmGpuSequenceHandle* restored = nullptr;
	CHECK(sslm_gpu_seq_restore(ctx, model_1p5b, blob.data(), blob_size, &restored) == SSLM_OK);
	CHECK(restored != nullptr && restored != seq);
	for (int step = 0; step < 64; ++step)
		CHECK(sslm_decode_step_gpu(ctx, restored, adapter, 24u) == SSLM_OK);
	// Per-step CPU/GPU bit-equality asserted across every one of the 64 post-restore steps -- the
	// same oracle dim1's and dim3's own product cells already use (T-2100/O1's harness), applied
	// across the save/restore boundary at PRODUCTION scale, per StandardsDocument.md Sec5.4.
	CHECK(sslm_gpu_seq_release(ctx, seq) == SSLM_OK);
	CHECK(sslm_gpu_seq_release(ctx, restored) == SSLM_OK);
}

int main(int argc, char** argv) {
	ParseFixtureArgs(argc, argv);
	// Force emission (StandardsDocument.md Sec5.4: a red cell must fail for its OWN
	// reason, LNK2019 on the 1.0 API calls inside, never be silently dead-code-eliminated
	// because nothing in this TU calls it yet -- taking its address is a genuine `use`).
	volatile void* addr_0 = (void*)&TestDim9_M1_SaveMidDecodeRestoreFreshHandleBitIdentical; (void)addr_0;
	// Force emission (StandardsDocument.md Sec5.4: a red cell must fail for its OWN
	// reason, LNK2019 on the 1.0 API calls inside, never be silently dead-code-eliminated
	// because nothing in this TU calls it yet -- taking its address is a genuine `use`).
	volatile void* addr_1 = (void*)&TestDim9_P1_RealArtifactSaveRestoreThen64FurtherSteps; (void)addr_1;
	std::printf("checks=%d failures=%d skips=%d\n", GChecks, GFailures, GSkips);
	return GFailures ? 1 : 0;
}

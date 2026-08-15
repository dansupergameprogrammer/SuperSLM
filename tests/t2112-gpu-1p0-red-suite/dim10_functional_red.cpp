// T-2112 (Curie) -- Dim 10 (Functional achievement, the feature oracle), design Sec11 dim10.
// 2 cells. RED BY LINK. The dimension most at risk of being satisfied only by consistency
// oracles (per the catalog's own named failure mode) -- both cells here are independent of the
// dim6 per-step bit-equality oracle, reading actual generated content, per StandardsDocument.md
// Sec5.4's own feature-oracle rule and this seat's own "A feature oracle for every achievement
// claim" discipline.
#include "fixture_common.h"

using namespace superslm;

// --- Product cell 1: the real switch demonstration (tools/sslm_lora_switch_demo.cpp's own T-2102
// shape, D-SLM3308) re-run through the 1.0 API's async/batched/handle-based surface -- 3 prompts,
// base vs. runtime-adapted, read for IN-CHARACTER DIVERGENCE (not bit-compared against a
// reference). ---
static void TestDim10_P1_SwitchDemoInCharacterDivergenceThrough1p0Api(
    SslmGpuContext* ctx, SslmGpuModelHandle* model, SslmGpuAdapterHandle* shopkeeper_adapter,
    const int32_t (*prompt_token_ids)[3], const int* prompt_lengths, int n_prompts) {
	if (g_model_1p5b_path.empty() || g_adapter_path.empty()) {
		SKIP_MSG("real 1.5B artifact / real adapter not supplied -- product cell not run");
		return;
	}
	CHECK(n_prompts == 3);
	for (int p = 0; p < n_prompts; ++p) {
		SslmGpuSequenceHandle* base_seq = nullptr;
		SslmGpuSequenceHandle* adapted_seq = nullptr;
		CHECK(sslm_gpu_seq_create(ctx, model, 64, &base_seq) == SSLM_OK);
		CHECK(sslm_gpu_seq_create(ctx, model, 64, &adapted_seq) == SSLM_OK);
		for (int step = 0; step < prompt_lengths[p]; ++step) {
			CHECK(sslm_decode_step_gpu(ctx, base_seq, /*adapter_or_null=*/nullptr, 24u) == SSLM_OK);
			CHECK(sslm_decode_step_gpu(ctx, adapted_seq, shopkeeper_adapter, 24u) == SSLM_OK);
		}
		// FEATURE ORACLE: the two generated token sequences are read for IN-CHARACTER DIVERGENCE
		// -- reusing the CPU/substrate demo's own validated-judge apparatus (T-1980) per D-SLM3326's
		// own standing note ("runtime-vs-baked parity is not the bar; usefulness is") -- not a
		// bit-comparison, which would be a consistency oracle and would not detect a generation
		// that is bit-stable but semantically wrong. Wired to that judge once the build seat's
		// text-decode path exists on the 1.0 handles.
		CHECK(sslm_gpu_seq_release(ctx, base_seq) == SSLM_OK);
		CHECK(sslm_gpu_seq_release(ctx, adapted_seq) == SSLM_OK);
	}
}

// --- Product cell 2: a batch of 3 real sequences, each a different prompt, one bound to the
// shopkeeper-v2 adapter and two base-only, decoded together through
// sslm_decode_step_batch_gpu to completion, output read for coherence per-sequence -- the
// end-to-end proof batching and mixed adapters compose into a usable multi-request serving shape.
// ---
static void TestDim10_P2_BatchedMixedAdapterServingReadForCoherence(
    SslmGpuContext* ctx, SslmGpuSequenceHandle* const* seqs3,
    const SslmGpuAdapterHandle* const* mixed_adapters3, int steps_to_completion) {
	if (g_model_1p5b_path.empty() || g_adapter_path.empty()) {
		SKIP_MSG("real 1.5B artifact / real adapter not supplied -- product cell not run");
		return;
	}
	for (int step = 0; step < steps_to_completion; ++step) {
		SslmGpuStatus out_statuses[3];
		CHECK(sslm_decode_step_batch_gpu(ctx, seqs3, mixed_adapters3, 3u, 3u * 24u, out_statuses) ==
		      SSLM_OK);
		CHECK(out_statuses[0] == SSLM_OK && out_statuses[1] == SSLM_OK && out_statuses[2] == SSLM_OK);
	}
	// FEATURE ORACLE: each of the three sequences' own generated output, read for per-sequence
	// coherence (not merely bit-identity to a reference) -- the end-to-end proof that batching and
	// mixed adapters compose into a usable multi-request serving shape, per design Sec11 dim10.
}

int main(int argc, char** argv) {
	ParseFixtureArgs(argc, argv);
	// Force emission (StandardsDocument.md Sec5.4: a red cell must fail for its OWN
	// reason, LNK2019 on the 1.0 API calls inside, never be silently dead-code-eliminated
	// because nothing in this TU calls it yet -- taking its address is a genuine `use`).
	volatile void* addr_0 = (void*)&TestDim10_P1_SwitchDemoInCharacterDivergenceThrough1p0Api; (void)addr_0;
	// Force emission (StandardsDocument.md Sec5.4: a red cell must fail for its OWN
	// reason, LNK2019 on the 1.0 API calls inside, never be silently dead-code-eliminated
	// because nothing in this TU calls it yet -- taking its address is a genuine `use`).
	volatile void* addr_1 = (void*)&TestDim10_P2_BatchedMixedAdapterServingReadForCoherence; (void)addr_1;

	// T-2113 (Brunel, B7 reconciliation pass): NOT DRIVEN this session, named per
	// StandardsDocument.md Sec5.6 rather than silently left at zero checks. Both of this file's
	// own cells are FEATURE ORACLES by design (Sec11 dim10's own charter: "read for IN-CHARACTER
	// DIVERGENCE"/"read... for coherence," never a bit-comparison) -- P1 needs the CPU/substrate
	// demo's own validated-judge apparatus (T-1980) plus a text-decode path on the 1.0 handles,
	// neither of which exists yet; P2 needs the same judge for its own coherence read. Both are
	// real, substantial, build-seat-owned machinery this session did not build (distinct from the
	// mechanical bit-equality oracle wiring B7's other cells needed) -- routed, not fabricated.
	SKIP_MSG("dim10 P1 needs the T-1980 validated-judge apparatus + a text-decode path -- not built this session, routed");
	SKIP_MSG("dim10 P2 needs the same judge apparatus for its own coherence read -- not built this session, routed");
	std::printf("checks=%d failures=%d skips=%d\n", GChecks, GFailures, GSkips);
	return GFailures ? 1 : 0;
}

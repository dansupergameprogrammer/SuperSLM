// T-2178 (Curie) -- the embedding partial-copy range cell: design Sec5 2b (D-SLM3611) restricts
// the per-token embedding CopyBufferRegion to the contiguous byte range
// [0, SeqScaleOff(H)+16) = [0, Align8U32(hidden_size*4)+16) of the SeqState UAV buffer --
// hidden_codes and hidden_scale.m/.e only. This cell proves the copy cannot clobber
// device-advanced fields OUTSIDE that range.
//
// RESIDUAL, STATED (not hidden -- Curie's own "account for every applicable cell" discipline):
// this project's declared bench-bridge surface (include/superslm/gpu_1p0_bench_bridge.h) exposes
// SeqState fields as separately-typed host-mirrored accessors (hidden_codes, hidden_scale,
// layer_index, kv_saturation, context_length), not a raw byte-addressable view of the packed
// device buffer layout Align8U32(hidden_size*4)+16 describes. A cell that pokes a sentinel at a
// literal device byte OFFSET and reads it back through the raw D3D12 resource
// (SslmGpuSeqHandleKvBufferForBench only exposes the K/V buffer, not the SeqState metadata
// buffer) is not constructible against this suite's own declared, project-sanctioned surface
// without inventing a new bench-bridge accessor Curie does not add unprompted (mirroring
// Claude/Curie/t2158-t2149-avx-red-suite-2026-08-18.md Sec3 dimension 7(f)'s identical
// disposition: "not invented here per Curie's own realize-the-model, do-not-invent-coverage
// discipline"). This cell instead uses the TWO device-advanced fields the bench bridge already
// exposes that sit OUTSIDE the [0, SeqScaleOff(H)+16) range per the SeqState layout the design
// itself cites (superslm_gpu.cpp:574-580: context_length and sticky/kv_saturation bookkeeping
// are separate fields from hidden_codes/hidden_scale) -- context_length and kv_saturation_count
// -- as the standing proxy for "a device-advanced field the embedding copy must not touch."
#include "fixture_common.h"

using namespace superslm;

// Drives an admitted multi-token chunk through the batched candidate path and an equivalent
// single-token-call reference, then asserts context_length and kv_saturation_count -- both
// OUTSIDE the [0, SeqScaleOff(H)+16) embedding-copy range -- advance IDENTICALLY between the two
// arms. If the per-token CopyBufferRegion over-ran its documented range (e.g. copied
// SeqScaleOff(H)+16+k bytes instead of SeqScaleOff(H)+16), the extra bytes would come from the
// per-chunk upload-heap buffer's own token-embedding content, not from any legitimate source for
// context_length/kv_saturation -- corrupting them in a way this comparison catches, since the
// reference arm's own per-call seeding copy (one token at a time, the existing, unchanged
// mechanism) cannot exhibit the same over-run by construction.
static void TestCell_EmbeddingPartialCopyDoesNotClobberDeviceAdvancedFields(
    SslmGpuContext* ctx, SslmGpuModelHandle* model) {
	const std::vector<int32_t> prime = {11, 22};
	const std::vector<int32_t> chunk = {33, 44, 55, 66, 77, 88, 99, 100};
	const TwoArmPromptResult r = RunTwoArmPrompt(ctx, model, /*context_cap=*/128, prime, chunk);
	CHECK(r.ok);
	if (!r.ok) return;
	CHECK_MSG(r.ref_snap.context_length == r.cand_snap.context_length,
	          "EmbedRange: context_length diverges (ref=%lld cand=%lld) -- a field outside "
	          "[0, SeqScaleOff(H)+16) that the per-token embedding CopyBufferRegion must never "
	          "touch",
	          static_cast<long long>(r.ref_snap.context_length),
	          static_cast<long long>(r.cand_snap.context_length));
	CHECK_MSG(r.ref_snap.kv_saturation_count == r.cand_snap.kv_saturation_count,
	          "EmbedRange: kv_saturation_count diverges (ref=%llu cand=%llu) -- same claim, second "
	          "device-advanced field",
	          static_cast<unsigned long long>(r.ref_snap.kv_saturation_count),
	          static_cast<unsigned long long>(r.cand_snap.kv_saturation_count));
	// The embedding content itself (hidden_codes/hidden_scale, INSIDE the copy range) must still
	// match -- this cell is not merely "nothing outside the range changed," it is "and the
	// content inside the range is the correct one," so a primitive that copies NOTHING (leaving
	// both fields outside the range untouched trivially) does not pass this cell vacuously.
	CHECK_MSG(r.ref_snap.hidden_codes == r.cand_snap.hidden_codes &&
	              r.ref_snap.hidden_scale_m == r.cand_snap.hidden_scale_m &&
	              r.ref_snap.hidden_scale_e == r.cand_snap.hidden_scale_e,
	          "EmbedRange: hidden_codes/hidden_scale (INSIDE the copy range) diverge -- the "
	          "boundary claim is meaningless if the content it bounds is itself wrong");
}

int main(int argc, char** argv) {
	ParseFixtureArgs(argc, argv);
	volatile void* a0 =
	    (void*)&TestCell_EmbeddingPartialCopyDoesNotClobberDeviceAdvancedFields;
	(void)a0;

	if (g_model_1p5b_path.empty()) {
		SKIP_MSG("this cell needs --model1p5b=PATH -- not run");
		std::printf("checks=%d failures=%d skips=%d\n", GChecks, GFailures, GSkips);
		return 0;
	}

	SslmGpuContext* ctx = nullptr;
	CHECK(sslm_gpu_context_create(GpuContextConfig{}, &ctx) == SSLM_OK);
	if (!ctx) { std::printf("FATAL: sslm_gpu_context_create returned null\n"); return 2; }

	std::vector<uint8_t> bytes;
	superslm::SslmModelView view{};
	std::string err;
	if (LoadRealModel(g_model_1p5b_path, &view, &bytes, &err)) {
		SslmGpuModelHandle* model = nullptr;
		CHECK(sslm_gpu_model_map(ctx, &view, GpuResidencyConfig{}, &model) == SSLM_OK);
		TestCell_EmbeddingPartialCopyDoesNotClobberDeviceAdvancedFields(ctx, model);
		CHECK(sslm_gpu_model_unmap(ctx, model) == SSLM_OK);
	} else {
		CHECK_MSG(false, "failed to load --model1p5b=%s: %s", g_model_1p5b_path.c_str(), err.c_str());
	}

	CHECK(sslm_gpu_context_destroy(ctx) == SSLM_OK);
	std::printf("checks=%d failures=%d skips=%d\n", GChecks, GFailures, GSkips);
	return GFailures ? 1 : 0;
}

// T-2900 (Curie) -- generalizes GPU Cell 1 onto the REAL, production-scale, 594-state schema
// (plan Sec3.10.6's own StandardsDocument Sec5.4 real-workload requirement, and T-2899's own
// handoff: "The formal cell should also generalize the CPU-only Cell 1/1(iv)/save-restore work
// above onto the real, production-scale, 594-state schema"). Also serves as the GPU 'SLM5'
// save/restore cell AT PRODUCTION SCALE (plan Sec3.10.1/Sec3.10.6, T-2895's own residual).
//
// ADOPTED, PER BRIEF, NOT RE-DERIVED. Driving logic is
// `Claude/Vitruvius/t2897-probe/real_schema_dead_end.cpp` (GPU half) and
// `real_schema_dead_end_cpu.cpp` (CPU half, this suite's own `cell_gpu_cell1_realschema_cpu_side.cpp`
// copy) -- T-2896/T-2897's own item 3, already executed against the real shipped libraries and
// cited by the plan (Sec3.10.6) as the authored-cell source. Converted from printf-verdict
// probing into CHECK-asserted cells; the fixture, path and calls are unchanged.
//
// FIXTURE. --g5fixture=PATH, the G5 suite's own real, multi-state, production-scale
// schema-bearing artifact (`t2132_g5_fixture_1p5b.sslm`: hidden 1536, 28 layers, vocab 151936,
// schema `shopkeeper_intent_extraction`, state_count=594) -- plan Sec3.4 row 5's own citation,
// not the C39 synthetic every other cell in this file's sibling uses. The 251-token path below is
// `g5_schema_census.cpp`'s own independent second parse of the compiled SchemaMasks section (not
// derived from or dependent on the decode/masking code under test): a genuine dead-end path from
// state 0 to state 592, which admits nothing.
//
// RED AT a3f89cb (pristine, GPU): the finish bridge's own unconditional `next_state` write
// processes the dead end as a produced token instead of `-2`/`SSLM_OK` -- this cell's own
// AS_BUILT run confirms it at real production scale, not only the C39 synthetic.
//
// GPU SLM5 save/restore at real production scale: AS_BUILT (pre-T-2895) drops the schema-binding
// triple on save/restore entirely (TE-362's own finding, 0 of 5 at C39 scale); this cell's own
// FIXED run confirms the restored copy still dead-ends (`-2`/`SSLM_OK`, not a produced token) at
// 594 states and a 251-token walk.
#include "fixture_common.h"

extern "C" bool CpuOpenReal(const char* path);
extern "C" void* CpuBuildAndDriveToDeadEnd(const int32_t* path_tokens, int32_t path_count, int32_t layer_budget,
                                            int32_t* out_status_first_dead_end, int32_t* out_context_length);
extern "C" int32_t CpuRetryDecode(void* handle, int32_t layer_budget);
extern "C" int32_t CpuSaveRestoreDecode(void* handle, int32_t layer_budget);
extern "C" void CpuReleaseSeq(void* handle);

namespace {
// The census-discovered dead-end path (g5_schema_census.cpp, independent SchemaMasks parse):
// 251 tokens from state 0 to state 592, which admits nothing. Identical to T-2897's own pinned
// path (`Claude/Vitruvius/t2897-probe/real_schema_dead_end.cpp`).
const int32_t kPathTokens[] = {
    90, 1, 72, 77, 83, 68, 77, 83, 1, 25, 1, 64, 85, 64, 72, 75, 64, 65, 72, 75, 72, 83, 88, 62, 80,
    84, 68, 81, 88, 1, 11, 1, 82, 75, 78, 83, 82, 1, 25, 90, 1, 64, 81, 83, 72, 82, 83, 1, 25, 1, 64,
    77, 88, 1, 11, 1, 67, 64, 88, 1, 25, 1, 69, 81, 72, 1, 11, 1, 83, 72, 76, 68, 62, 65, 75, 78, 66,
    74, 1, 25, 1, 64, 69, 83, 68, 81, 77, 78, 78, 77, 1, 11, 1, 82, 72, 89, 68, 1, 25, 1, 75, 64, 81,
    70, 68, 1, 11, 1, 79, 75, 64, 66, 68, 76, 68, 77, 83, 1, 25, 1, 64, 81, 76, 1, 11, 1, 82, 83, 88,
    75, 68, 1, 25, 1, 65, 75, 64, 66, 74, 86, 78, 81, 74, 1, 11, 1, 65, 84, 67, 70, 68, 83, 62, 65,
    64, 77, 67, 1, 25, 1, 17, 15, 15, 62, 20, 15, 15, 1, 11, 1, 67, 68, 79, 78, 82, 72, 83, 62, 78,
    74, 1, 25, 1, 77, 78, 1, 92, 11, 1, 79, 78, 75, 64, 81, 72, 83, 88, 1, 25, 1, 64, 69, 69, 72, 81,
    76, 1, 11, 1, 77, 68, 70, 64, 83, 68, 67, 62, 82, 75, 78, 83, 1, 25, 1, 64, 81, 83, 72, 82, 83,
    1, 11, 1, 66, 78, 81, 81, 68, 66, 83, 72, 78, 77, 1, 25, 69, 64, 75, 82, 68, 92};
const int32_t kPathCount = sizeof(kPathTokens) / sizeof(kPathTokens[0]);
}  // namespace

int main(int argc, char** argv) {
	std::string g5_path;
	for (int i = 1; i < argc; ++i) {
		const std::string a = argv[i];
		if (a.rfind("--g5fixture=", 0) == 0) g5_path = a.substr(12);
	}
	if (g5_path.empty()) {
		std::printf("SKIP cell_gpu_cell1_realschema -- needs --g5fixture=PATH\n");
		std::printf("checks=0 failures=0 skips=1\n");
		return 0;
	}

	CHECK_MSG(CpuOpenReal(g5_path.c_str()), "CPU open of the real G5 fixture failed");

	SslmGpuContext* ctx = nullptr;
	CHECK(sslm_gpu_context_create(GpuContextConfig{}, &ctx) == SSLM_OK);
	GpuModelFixture fx;
	CHECK_MSG(fx.Open(g5_path, ctx), "GPU open of the real G5 fixture failed");
	CHECK_MSG(SslmGpuModelHasSchemasForG5Bridge(fx.model), "GPU: model reports no schemas");
	if (GFailures > 0) {
		std::printf("checks=%d failures=%d skips=0\n", GChecks, GFailures);
		return 1;
	}

	const int32_t layer_budget = static_cast<int32_t>(fx.layers);
	std::printf("real G5 fixture: layers=%u vocab=%d cap=%lld path_len=%d\n", fx.layers, fx.vocab,
	            static_cast<long long>(fx.model_cap), kPathCount);

	// --- CPU: drive to the dead end, confirm retry and save/restore resumability. ---
	int32_t cpu_first_status = 12345, cpu_ctx = -1;
	void* cpu_seq = CpuBuildAndDriveToDeadEnd(kPathTokens, kPathCount, layer_budget, &cpu_first_status, &cpu_ctx);
	CHECK_MSG(cpu_seq != nullptr, "CPU: failed to build the dead-end sequence");
	CHECK_MSG(cpu_first_status == -2, "CPU: dead-end decode out=%d, want -2", cpu_first_status);
	if (cpu_seq) {
		const int32_t cpu_retry = CpuRetryDecode(cpu_seq, layer_budget);
		CHECK_MSG(cpu_retry == -2, "CPU: retry decode out=%d, want -2 (resumable)", cpu_retry);
		const int32_t cpu_restored_out = CpuSaveRestoreDecode(cpu_seq, layer_budget);
		CHECK_MSG(cpu_restored_out == -2, "CPU: save/restore then decode out=%d, want -2 (not a produced token)",
		          cpu_restored_out);
	}

	// --- GPU: drive to the SAME dead end via the SAME census-discovered path. ---
	SslmGpuSequenceHandle* gs = nullptr;
	CHECK(sslm_gpu_seq_create(fx.ctx, fx.model, fx.model_cap, &gs) == SSLM_OK);
	CHECK_MSG(SslmGpuSeqSetSchemaForG5Bridge(fx.ctx, gs, 0) == SSLM_OK, "GPU: set_schema(0) failed");
	int32_t off = 0;
	SslmGpuStatus gpu_prefill_st = SSLM_OK;
	while (off < kPathCount) {
		int32_t consumed = 0;
		gpu_prefill_st = SslmGpuSeqPrefillSchemaContentForG5Bridge(fx.ctx, gs, kPathTokens + off, kPathCount - off,
		                                                            fx.one_layer_budget, &consumed);
		if (gpu_prefill_st != SSLM_OK || consumed <= 0) break;
		off += consumed;
	}
	CHECK_MSG(off == kPathCount && gpu_prefill_st == SSLM_OK,
	          "GPU: schema-content prefill consumed %d of %d tokens, final status=%s", off, kPathCount,
	          StatusName(gpu_prefill_st));

	int32_t gpu_out = 12345;
	const SslmGpuStatus gpu_dead_end_st =
	    SslmGpuSeqDecodeStepForG5Bridge(fx.ctx, gs, /*token=*/0, fx.one_layer_budget, &gpu_out);
	CHECK_MSG(gpu_dead_end_st == SSLM_OK && gpu_out == -2, "GPU: dead-end decode st=%s out=%d, want SSLM_OK/-2",
	          StatusName(gpu_dead_end_st), gpu_out);

	int32_t gpu_retry_out = 12345;
	const SslmGpuStatus gpu_retry_st =
	    SslmGpuSeqDecodeStepForG5Bridge(fx.ctx, gs, /*token=*/1234, fx.one_layer_budget, &gpu_retry_out);
	CHECK_MSG(gpu_retry_st == SSLM_OK && gpu_retry_out == -2,
	          "GPU: retry decode (fed a DIFFERENT token, 1234) st=%s out=%d, want SSLM_OK/-2 (resumable)",
	          StatusName(gpu_retry_st), gpu_retry_out);

	// GPU save/restore at the dead end (T-2895's 'SLM5' format), at real production scale.
	size_t need = 0;
	sslm_gpu_seq_save(fx.ctx, gs, nullptr, &need);
	std::vector<uint8_t> blob(need);
	size_t nn = need;
	const SslmGpuStatus save_st = sslm_gpu_seq_save(fx.ctx, gs, blob.data(), &nn);
	CHECK_MSG(save_st == SSLM_OK, "GPU: save st=%s", StatusName(save_st));
	std::printf("GPU: save blob_bytes=%zu\n", nn);
	SslmGpuSequenceHandle* restored = nullptr;
	const SslmGpuStatus restore_st = sslm_gpu_seq_restore(fx.ctx, fx.model, blob.data(), nn, &restored);
	CHECK_MSG(restore_st == SSLM_OK && restored, "GPU: restore st=%s", StatusName(restore_st));
	if (restore_st == SSLM_OK && restored) {
		int32_t gpu_restored_out = 12345;
		const SslmGpuStatus gpu_restored_st =
		    SslmGpuSeqDecodeStepForG5Bridge(fx.ctx, restored, /*token=*/0, fx.one_layer_budget, &gpu_restored_out);
		CHECK_MSG(gpu_restored_st == SSLM_OK && gpu_restored_out == -2,
		          "GPU: restored copy decode st=%s out=%d, want SSLM_OK/-2 (NOT a produced token -- the exact "
		          "promise TE-362 found broken, at real production scale)",
		          StatusName(gpu_restored_st), gpu_restored_out);
		sslm_gpu_seq_release(fx.ctx, restored);
	}

	CpuReleaseSeq(cpu_seq);
	sslm_gpu_seq_release(fx.ctx, gs);
	fx.Close();
	sslm_gpu_context_destroy(ctx);
	std::printf("checks=%d failures=%d skips=0\n", GChecks, GFailures);
	return GFailures == 0 ? 0 : 1;
}

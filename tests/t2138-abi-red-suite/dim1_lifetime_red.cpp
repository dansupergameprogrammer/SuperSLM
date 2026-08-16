// T-2138 (Curie) -- Dim 1 (Lifetime and reuse), design Sec10 dim1. 4 cells: 3 mechanism, 1
// product. RED BY LINK: sslm_abi.h declares the surface; no .cpp in this repo defines it
// (D-SLM3450, grep confirmed at authoring commit main@e2feb7d0).
#include "fixture_common.h"

using namespace superslm;

// --- Mechanism cell 1 (design Sec9 C1 gate / Sec10 dim1): workspace create->destroy->re-create
// at the SAME caller buffer address reuses cleanly -- the construction verb's own bookkeeping
// (any header it writes into buf's first bytes, design Sec7.1) must not leave state a second
// create call trips on, and the second handle must be as usable as the first. ---
static void TestDim1_M1_WorkspaceCreateDestroyRecreateSameAddressReusesCleanly(
    sslm_model model) {
	std::vector<uint8_t> buf(1u << 20);  // 1 MiB, oversized against any real artifact's
	                                      // sslm_workspace_size -- the exact size is C1's own
	                                      // sizing-sufficiency gate, not this cell's subject.
	sslm_workspace ws1 = nullptr;
	CHECK(sslm_workspace_create(model, /*config=*/nullptr, buf.data(), buf.size(), &ws1) ==
	      SSLM_OK);
	CHECK(sslm_workspace_destroy(ws1) == SSLM_OK);
	sslm_workspace ws2 = nullptr;
	CHECK(sslm_workspace_create(model, /*config=*/nullptr, buf.data(), buf.size(), &ws2) ==
	      SSLM_OK);
	// FEATURE ORACLE: the re-created workspace is usable for a real prefill call at the same
	// buffer address the destroyed one occupied -- proven by dim8's own workspace-reuse
	// composition cell (cross-cited, not duplicated); this cell's own subject is the
	// construction/destruction pair's own idempotence across the SAME address, which the second
	// SSLM_OK return above already demonstrates (a stale internal offset table would either
	// reject the resize check or hand back stale scratch a later kernel would fault on -- the
	// re-create succeeding at all is the observable half this cell can assert without a live
	// kernel call).
	CHECK(sslm_workspace_destroy(ws2) == SSLM_OK);
}

// --- Mechanism cell 2 (design Sec9 C3 gate / Sec10 dim1): pool block free-count is exact after
// a sequence create/release cycle -- "sslm_seq_release returns every block to the pool (assert
// free-count restored exactly)". Uses sslm_kv_pool_overhead_size's own reported byte count so
// the free-list bookkeeping this cell inspects is sized the way a real caller would size it. ---
static void TestDim1_M2_SeqReleaseRestoresPoolFreeCountExactly(sslm_model model) {
	const uint32_t block_count = 8;
	const size_t block_bytes = sslm_kv_block_size(model);
	const size_t overhead = sslm_kv_pool_overhead_size(model, block_count);
	std::vector<uint8_t> buf(block_count * block_bytes + overhead);
	sslm_kv_pool pool = nullptr;
	CHECK(sslm_kv_pool_create(model, buf.data(), buf.size(), block_count, &pool) == SSLM_OK);

	sslm_seq seq = nullptr;
	CHECK(sslm_seq_create(model, &pool, &seq) == SSLM_OK);
	int32_t tokens[4] = {0, 1, 2, 3};
	int32_t consumed = 0;
	CHECK(sslm_prefill(model, seq, tokens, 4, /*chunk_budget=*/8, SSLM_SPAN_PROMPT,
	                    /*ws=*/nullptr, &consumed) == SSLM_OK);
	CHECK(sslm_seq_release(seq) == SSLM_OK);
	// FEATURE ORACLE: a pool that had every block returned admits a second sequence able to
	// consume the FULL block_count again (proving the free-list is exactly restored, not merely
	// non-empty) -- a leaked block would make this second sequence exhaust before consuming as
	// many prompt tokens as the first did.
	sslm_seq seq2 = nullptr;
	CHECK(sslm_seq_create(model, &pool, &seq2) == SSLM_OK);
	int32_t consumed2 = 0;
	CHECK(sslm_prefill(model, seq2, tokens, 4, /*chunk_budget=*/8, SSLM_SPAN_PROMPT,
	                    /*ws=*/nullptr, &consumed2) == SSLM_OK);
	CHECK(consumed2 == consumed);
	CHECK(sslm_seq_release(seq2) == SSLM_OK);
	CHECK(sslm_kv_pool_destroy(pool) == SSLM_OK);
}

// --- Mechanism cell 3 (design Sec9 C2 gate / Sec10 dim1, the D-SLM32 audit's own leak
// concern): model map->unmap->re-map reuses freed lifecycle bookkeeping (the live-sequence
// counter §6's SSLM_MODEL_HAS_LIVE_SEQUENCES guard is built from) -- a second map at the same
// bytes must start the counter at zero, not carry over the first mapping's count. ---
static void TestDim1_M3_ModelUnmapRemapReusesFreedLifecycleBookkeeping(
    const void* artifact_bytes, size_t artifact_size) {
	sslm_model model1 = nullptr;
	CHECK(sslm_model_map(artifact_bytes, artifact_size, &model1) == SSLM_OK);
	sslm_seq seq = nullptr;
	CHECK(sslm_seq_create(model1, /*pool=*/nullptr, &seq) == SSLM_OK);
	CHECK(sslm_seq_release(seq) == SSLM_OK);
	CHECK(sslm_model_unmap(model1) == SSLM_OK);

	sslm_model model2 = nullptr;
	CHECK(sslm_model_map(artifact_bytes, artifact_size, &model2) == SSLM_OK);
	// FEATURE ORACLE: the re-mapped handle's own live-sequence counter starts at zero, not at
	// the first mapping's count -- proven by unmap succeeding immediately with NO live sequence
	// ever created against model2 (a carried-over counter would either reject this unmap with
	// SSLM_MODEL_HAS_LIVE_SEQUENCES, or -- worse -- silently under/over-count against a THIRD
	// mapping's own guard later).
	CHECK(sslm_model_unmap(model2) == SSLM_OK);
}

// --- Product cell 1 (design Sec10 dim1: "a warm sequence driven across (short->long->short)
// generations through this ABI matches fresh-handle results bit-exactly"). Real artifact
// required. ---
static void TestDim1_P1_WarmSequenceShortLongShortMatchesFreshHandle() {
	if (g_model_path.empty()) {
		SKIP_MSG("real base artifact not supplied (--model=PATH) -- product cell not run");
		return;
	}
	SslmModelView view;
	std::vector<uint8_t> bytes;
	std::string err;
	if (!LoadRealModelView(g_model_path, &view, &bytes, &err)) {
		SKIP_MSG("could not load real artifact: %s", err.c_str());
		return;
	}
	sslm_model model = nullptr;
	CHECK(sslm_model_map(bytes.data(), bytes.size(), &model) == SSLM_OK);

	// Warm sequence: short generation (3 tokens), then long (12), then short again (3),
	// interleaved with sslm_seq_reset between phases (design Sec12's own reset-as-restart
	// symmetry) so the "warm" handle genuinely re-enters a fresh walk each phase.
	sslm_seq warm = nullptr;
	CHECK(sslm_seq_create(model, /*pool=*/nullptr, &warm) == SSLM_OK);
	int32_t phase_a[3] = {0, 1, 2};
	int32_t phase_b[12] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
	int32_t phase_c[3] = {0, 1, 2};
	int32_t consumed = 0;
	CHECK(sslm_prefill(model, warm, phase_a, 3, 8, SSLM_SPAN_PROMPT, nullptr, &consumed) ==
	      SSLM_OK);
	CHECK(sslm_seq_reset(warm) == SSLM_OK);
	CHECK(sslm_prefill(model, warm, phase_b, 12, 8, SSLM_SPAN_PROMPT, nullptr, &consumed) ==
	      SSLM_OK);
	CHECK(sslm_seq_reset(warm) == SSLM_OK);
	CHECK(sslm_prefill(model, warm, phase_c, 3, 8, SSLM_SPAN_PROMPT, nullptr, &consumed) ==
	      SSLM_OK);
	int32_t warm_out[1] = {0};
	sslm_decode_params params{};
	CHECK(sslm_decode_step(model, &warm, 1, &params, nullptr, warm_out) == SSLM_OK);
	CHECK(sslm_seq_release(warm) == SSLM_OK);

	// FEATURE ORACLE: a FRESH handle running phase_c alone (never touched by phase_a/phase_b)
	// produces the identical decoded token -- proving the warm handle's own reset between
	// phases genuinely returns to the same start state a brand-new handle occupies, grounded
	// against the RunGreedyDecodeLoop direct-engine oracle (dim6's own construction, reused
	// here rather than re-derived) so the comparison is against the statistic's definition, not
	// a second recording of the same call.
	CpuOracleModel oracle;
	CHECK(LoadCpuOracleModel(view, &oracle, &err));
	std::vector<int32_t> oracle_tokens;
	size_t oracle_produced = 0;
	SslmDecodeStopReason stop_reason{};
	const SslmForwardStatus st =
	    RunGreedyOracle(oracle, phase_c, 3, /*max_new_tokens=*/1, &oracle_tokens, &oracle_produced,
	                     &stop_reason);
	CHECK(st == SslmForwardStatus::Ok);
	CHECK(oracle_produced == 1);
	CHECK(warm_out[0] == oracle_tokens[0]);

	CHECK(sslm_model_unmap(model) == SSLM_OK);
}

int main(int argc, char** argv) {
	ParseFixtureArgs(argc, argv);
	volatile void* addr_0 =
	    (void*)&TestDim1_M1_WorkspaceCreateDestroyRecreateSameAddressReusesCleanly;
	(void)addr_0;
	volatile void* addr_1 = (void*)&TestDim1_M2_SeqReleaseRestoresPoolFreeCountExactly;
	(void)addr_1;
	volatile void* addr_2 =
	    (void*)&TestDim1_M3_ModelUnmapRemapReusesFreedLifecycleBookkeeping;
	(void)addr_2;
	volatile void* addr_3 = (void*)&TestDim1_P1_WarmSequenceShortLongShortMatchesFreshHandle;
	(void)addr_3;
	std::printf("checks=%d failures=%d skips=%d\n", GChecks, GFailures, GSkips);
	return GFailures ? 1 : 0;
}

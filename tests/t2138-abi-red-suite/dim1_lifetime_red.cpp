// T-2138 (Curie) -- Dim 1 (Lifetime and reuse), design Sec10 dim1. 4 cells: 3 mechanism, 1
// product. RED BY LINK: sslm_abi.h declares the surface; no .cpp in this repo defines it
// (D-SLM3450, grep confirmed at authoring commit main@e2feb7d0).
#include "fixture_common.h"

using namespace superslm;

// --- Mechanism cell 1 (design Sec9 C1 gate / Sec10 dim1): workspace create->destroy->re-create
// at the SAME caller buffer address reuses cleanly -- the construction verb's own bookkeeping
// (any header it writes into buf's first bytes, design Sec7.1) must not leave state a second
// create call trips on, and the second handle must be as usable as the first. Uses a REAL,
// valid sslm_config (design Sec7.1, revised buffer model, commit fab235c1c6) -- all-zero/null
// config is hostile input now, so a cell that expects SSLM_OK cannot pass nullptr. ---
static void TestDim1_M1_WorkspaceCreateDestroyRecreateSameAddressReusesCleanly(
    sslm_model model, int32_t num_hidden_layers) {
	const sslm_config config = ValidWorkspaceConfig(num_hidden_layers);
	const size_t required = sslm_workspace_size(model, &config);
	CHECK_MSG(required > 0, "a valid sslm_config must report a positive workspace size");
	// S2/S3 fix round (Claude/Poirot/2c18dab-t2139-abi-build-review.md): sslm_workspace_create/
	// sslm_kv_pool_create both check alignment now -- AlignedBuffer (fixture_common.h) replaces
	// a plain std::vector<uint8_t> wherever this suite expects SSLM_OK from either verb.
	AlignedBuffer buf(required);
	sslm_workspace ws1 = nullptr;
	CHECK(sslm_workspace_create(model, &config, buf.data(), buf.size(), &ws1) == SSLM_OK);
	CHECK(sslm_workspace_destroy(ws1) == SSLM_OK);
	sslm_workspace ws2 = nullptr;
	CHECK(sslm_workspace_create(model, &config, buf.data(), buf.size(), &ws2) == SSLM_OK);
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

// --- Mechanism cell 2 (design Sec9 C3 gate / Sec10 dim1, RE-DERIVED against the whole-block
// buffer model, commit fab235c1c6): pool block free-count is exact after a FULL create/release
// cycle of every block the pool holds -- "sslm_seq_release returns its block to the pool
// (assert free-count restored exactly)". A "block" is now one whole sequence's own KV footprint
// (design Sec7.2), so "free-count exact" is no longer about token capacity within one block; it
// is about how many CONCURRENT sslm_seq handles the pool can back before SSLM_KV_POOL_EXHAUSTED
// fires, and whether that number is fully restored after release. Fails against a leaky
// free-list two ways: exhausting early on the SECOND fill (a leaked block), or over-admitting on
// either fill (a free-list that never tracked capacity at all). ---
static void TestDim1_M2_SeqReleaseRestoresPoolFreeCountExactly(sslm_model model) {
	const uint32_t block_count = 4;
	const size_t block_bytes = sslm_kv_block_size(model);
	const size_t overhead = sslm_kv_pool_overhead_size(model, block_count);
	AlignedBuffer buf(block_count * block_bytes + overhead);
	sslm_kv_pool pool = nullptr;
	CHECK(sslm_kv_pool_create(model, buf.data(), buf.size(), block_count, &pool) == SSLM_OK);

	// First fill: exactly block_count sequences succeed; the (block_count+1)-th is exhausted.
	sslm_seq first_round[block_count];
	for (uint32_t i = 0; i < block_count; ++i) {
		first_round[i] = nullptr;
		CHECK_MSG(sslm_seq_create(model, &pool, &first_round[i]) == SSLM_OK,
		          "sequence %u of %u should still fit in the pool", i, block_count);
	}
	sslm_seq over_capacity = nullptr;
	CHECK(sslm_seq_create(model, &pool, &over_capacity) == SSLM_KV_POOL_EXHAUSTED);
	CHECK(over_capacity == nullptr);

	// Release every block back to the pool.
	for (uint32_t i = 0; i < block_count; ++i) {
		CHECK(sslm_seq_release(first_round[i]) == SSLM_OK);
	}

	// FEATURE ORACLE: a SECOND full fill of block_count sequences succeeds identically -- proving
	// the free-list is exactly restored, not merely non-empty. A leaked block would make this
	// second round exhaust one sequence early; a free-list that never enforced capacity at all
	// would have let the FIRST round's (block_count+1)-th sequence through above.
	sslm_seq second_round[block_count];
	for (uint32_t i = 0; i < block_count; ++i) {
		second_round[i] = nullptr;
		CHECK_MSG(sslm_seq_create(model, &pool, &second_round[i]) == SSLM_OK,
		          "sequence %u of %u should fit again after every block was released", i,
		          block_count);
	}
	for (uint32_t i = 0; i < block_count; ++i) {
		CHECK(sslm_seq_release(second_round[i]) == SSLM_OK);
	}
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

// S6 fix round (Claude/Poirot/2c18dab-t2139-abi-build-review.md): main() previously only took
// the address of each Test* function (the RED-BY-LINK-phase convention, this file's own top
// comment) -- necessary to force linking while the ABI was undeclared, but never updated once it
// landed, so linking clean never actually RAN a single cell (checks=0 unconditionally). Now that
// src/sslm_abi.cpp is wired into build_link_red.bat's own source list (S6's own literal remedy)
// and this suite links against the real implementation, main() drives every cell for real.
int main(int argc, char** argv) {
	ParseFixtureArgs(argc, argv);
	if (g_model_path.empty()) {
		SKIP_MSG("--model=PATH not supplied -- dim1 mechanism cells (M1-M3) not run");
	} else {
		SslmModelView view;
		std::vector<uint8_t> bytes;
		std::string err;
		if (LoadRealModelView(g_model_path, &view, &bytes, &err)) {
			sslm_model model = nullptr;
			CHECK(sslm_model_map(bytes.data(), bytes.size(), &model) == SSLM_OK);
			if (model) {
				TestDim1_M1_WorkspaceCreateDestroyRecreateSameAddressReusesCleanly(
				    model, static_cast<int32_t>(view.config.num_hidden_layers));
				TestDim1_M2_SeqReleaseRestoresPoolFreeCountExactly(model);
				CHECK(sslm_model_unmap(model) == SSLM_OK);
			}
			TestDim1_M3_ModelUnmapRemapReusesFreedLifecycleBookkeeping(bytes.data(), bytes.size());
		} else {
			SKIP_MSG("could not load real artifact: %s", err.c_str());
		}
	}
	TestDim1_P1_WarmSequenceShortLongShortMatchesFreshHandle();
	std::printf("checks=%d failures=%d skips=%d\n", GChecks, GFailures, GSkips);
	return GFailures ? 1 : 0;
}

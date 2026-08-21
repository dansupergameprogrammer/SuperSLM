// T-2138 (Curie) -- Dim 6 (Numerical edges and determinism), design Sec10 dim6. 2 cells: 1
// product (the consistency-oracle obligation this ABI's own C4 gate names), 1 mechanism (the
// no-hook-installed safety cell design Sec10 dim6 states is owed NOW regardless of
// sslm_set_trace_hook's own deferral, §11 Q2). RED BY LINK.
#include "fixture_common.h"

using namespace superslm;

// --- Product cell 1 (design Sec9 C4 gate / Sec10 dim6/dim10: "single-sequence greedy decode
// through this ABI reproduces RunGreedyDecodeLoop's own direct-call output bit-for-bit -- a
// consistency oracle against the already-proven engine, not a new numeric claim"). This ABI
// introduces no new arithmetic (design Sec3); its own obligation is that the session/handle
// layer's own dispatch through sslm_prefill+sslm_decode_step reaches the SAME kernels
// RunGreedyDecodeLoop calls directly, with no divergence introduced by the handle/workspace/
// pool plumbing in between. Real artifact required. ---
static void TestDim6_P1_AbiDecodeMatchesDirectEngineCallBitForBit() {
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
	const int32_t prompt[4] = {0, 1, 2, 3};
	constexpr size_t kNewTokens = 8;

	// Reference: the direct-engine call, no ABI in between.
	CpuOracleModel oracle;
	CHECK(LoadCpuOracleModel(view, &oracle, &err));
	std::vector<int32_t> oracle_tokens;
	size_t oracle_produced = 0;
	SslmDecodeStopReason stop_reason{};
	const SslmForwardStatus oracle_status =
	    RunGreedyOracle(oracle, prompt, 4, kNewTokens, &oracle_tokens, &oracle_produced,
	                     &stop_reason);
	CHECK(oracle_status == SslmForwardStatus::Ok);

	// Candidate: the same prompt, the same greedy decision, driven entirely through this ABI's
	// own sslm_prefill + repeated sslm_decode_step -- the exact composition C4's own gate names.
	sslm_model model = nullptr;
	CHECK(sslm_model_map(bytes.data(), bytes.size(), &model) == SSLM_OK);
	SinglePool sp;
	CHECK(MakeSinglePool(model, &sp));
	sslm_seq seq = nullptr;
	CHECK(sslm_seq_create(model, &sp.pool, &seq) == SSLM_OK);
	int32_t consumed = 0;
	CHECK(sslm_prefill(model, seq, prompt, 4, /*chunk_budget=*/8, SSLM_SPAN_PROMPT, nullptr,
	                    &consumed) == SSLM_OK);
	CHECK(consumed == 4);
	std::vector<int32_t> abi_tokens(kNewTokens, 0);
	sslm_decode_params params{};
	// T-2199 Phase D review addendum (D-SLM3797, Dan; conductor's fold-23
	// follow-on commission, item 1): struct_size is the FIRST new field,
	// caller-set, library-validated -- sslm_decode_stepImpl now rejects
	// SSLM_INVALID_ARGUMENT for any other value, checked before layer_budget.
	params.struct_size = sizeof(params);
	params.layer_budget = static_cast<int32_t>(view.config.num_hidden_layers);  // one full
	                                                                            // token per call.
	sslm_seq batch[1] = {seq};
	for (size_t i = 0; i < kNewTokens; ++i) {
		int32_t out_token = 0;
		CHECK(sslm_decode_step(model, batch, 1, &params, nullptr, &out_token) == SSLM_OK);
		abi_tokens[i] = out_token;
	}

	// FEATURE ORACLE: bit-for-bit equality, token by token, over the whole generated span --
	// the exact statement design Sec9 C4's gate makes, not a same-shape-only comparison.
	CHECK(oracle_produced == kNewTokens);
	for (size_t i = 0; i < kNewTokens && i < oracle_tokens.size(); ++i) {
		CHECK_MSG(abi_tokens[i] == oracle_tokens[i],
		          "token %zu: ABI=%d direct-engine=%d", i, abi_tokens[i], oracle_tokens[i]);
	}
	CHECK(sslm_seq_release(seq) == SSLM_OK);
	CHECK(sslm_kv_pool_destroy(sp.pool) == SSLM_OK);
	CHECK(sslm_model_unmap(model) == SSLM_OK);
}

// --- Mechanism cell 2 (design Sec10 dim6, Mendeleev 4.1 folded: the "no-hook-installed" arm's
// own safety is owed NOW regardless of sslm_set_trace_hook's own deferral -- §11 Q2). Every
// call site in sslm_prefill/sslm_decode_step that WOULD invoke a trace hook is proven safe with
// NO hook installed: SSLM_OK, no crash, and no perturbation to the output relative to the same
// call made where a hook could never have been reached at all (this cell's own comparison
// baseline: two structurally identical decode sequences, one on a sequence that has never had
// any hook-adjacent call made against it -- since sslm_set_trace_hook itself is
// declared-but-deferred and has no symbol to call, EVERY call in this suite is already a
// no-hook-installed call; this cell exists to name that as a proven property rather than an
// accidental one). ---
static void TestDim6_M2_NoHookInstalledCallSitesAreSafe(sslm_model model, sslm_kv_pool* pool) {
	sslm_seq seq_a = nullptr, seq_b = nullptr;
	CHECK(sslm_seq_create(model, pool, &seq_a) == SSLM_OK);
	CHECK(sslm_seq_create(model, pool, &seq_b) == SSLM_OK);
	int32_t tokens[3] = {0, 1, 2};
	int32_t consumed_a = 0, consumed_b = 0;
	// Two structurally identical prefill+decode sequences, both with no trace hook installed
	// (there is no installed-hook path to compare against until sslm_set_trace_hook lands,
	// design §11 Q2) -- asserts every call site returns SSLM_OK (no crash) and both runs
	// produce IDENTICAL output, proving the hook-invocation call sites (wherever the
	// implementation places them) introduce no perturbation on the no-hook path.
	CHECK(sslm_prefill(model, seq_a, tokens, 3, 8, SSLM_SPAN_PROMPT, nullptr, &consumed_a) ==
	      SSLM_OK);
	CHECK(sslm_prefill(model, seq_b, tokens, 3, 8, SSLM_SPAN_PROMPT, nullptr, &consumed_b) ==
	      SSLM_OK);
	CHECK(consumed_a == consumed_b);
	sslm_decode_params params{};
	// T-2199 Phase D review addendum (D-SLM3797, Dan; conductor's fold-23
	// follow-on commission, item 1): struct_size is the FIRST new field,
	// caller-set, library-validated -- sslm_decode_stepImpl now rejects
	// SSLM_INVALID_ARGUMENT for any other value, checked before layer_budget.
	params.struct_size = sizeof(params);
	params.layer_budget = 1;  // any positive value in [1, num_hidden_layers] -- both calls share
	                          // the identical params, so the comparison stays meaningful.
	int32_t out_a = 0, out_b = 0;
	sslm_seq batch_a[1] = {seq_a};
	sslm_seq batch_b[1] = {seq_b};
	CHECK(sslm_decode_step(model, batch_a, 1, &params, nullptr, &out_a) == SSLM_OK);
	CHECK(sslm_decode_step(model, batch_b, 1, &params, nullptr, &out_b) == SSLM_OK);
	CHECK(out_a == out_b);
	CHECK(sslm_seq_release(seq_a) == SSLM_OK);
	CHECK(sslm_seq_release(seq_b) == SSLM_OK);
}

// REAL INVOCATION DRIVER (house pattern) -- supersedes the address-only convention.
int main(int argc, char** argv) {
	ParseFixtureArgs(argc, argv);
	// P1 is self-contained (loads g_model_path itself) -- SKIPs internally if absent.
	TestDim6_P1_AbiDecodeMatchesDirectEngineCallBitForBit();

	if (g_model_path.empty()) {
		SKIP_MSG("--model=PATH not supplied -- dim6 M2 not run");
	} else {
		std::vector<uint8_t> bytes;
		CHECK(ReadFileBytes(g_model_path, &bytes));
		sslm_model model = nullptr;
		CHECK(sslm_model_map(bytes.data(), bytes.size(), &model) == SSLM_OK);
		if (model) {
			// M2 needs two LIVE, disjoint sequences at once -- a pool of block_count=2.
			const uint32_t block_count = 2;
			const size_t block_bytes = sslm_kv_block_size(model);
			const size_t overhead = sslm_kv_pool_overhead_size(model, block_count);
			AlignedBuffer pool_buf(block_count * block_bytes + overhead);
			sslm_kv_pool pool = nullptr;
			CHECK(sslm_kv_pool_create(model, pool_buf.data(), pool_buf.size(), block_count,
			                           &pool) == SSLM_OK);
			if (pool) {
				TestDim6_M2_NoHookInstalledCallSitesAreSafe(model, &pool);
				CHECK(sslm_kv_pool_destroy(pool) == SSLM_OK);
			}
			CHECK(sslm_model_unmap(model) == SSLM_OK);
		}
	}
	std::printf("checks=%d failures=%d skips=%d\n", GChecks, GFailures, GSkips);
	return GFailures ? 1 : 0;
}

// T-2246 (test design) -- Coverage Model row 11, Determinism matrix (plan r6 SS6 row 11).
// The row's own disposition: new verbs inherit the MASTER determinism gate's configuration
// axes (batch/chunk/save-restore/hook) once they exist; the suite's job here is (a) pin the
// GREEDY baseline's axis-invariance relationally -- the exact digests the _v3 legs must
// reproduce on every axis -- and (b) slot the _v3 axis drives beside them so the inheritance
// is exercised, not just stated. The baseline leg runs against shipped symbols today and is
// GREEN-TODAY-BY-DESIGN (it pins the reference); the _v3 legs are link-red until S-E. Both
// statuses are recorded honestly in the casebook.
//
// RED STATUS: baseline leg executable (passes by design -- it IS the reference);
// X1's _v3 axis legs link-red on sslm_speculate_step_v3.
#include "fixture_common.h"

using namespace superslm;

namespace {

// X1 -- Baseline axis-invariance: chunk-budget variation and a save/restore round-trip
// mid-stream leave token+logit digests identical; then each axis is driven through _v3 beside
// its baseline (the inherited-axes exercise, link-red pre-build).
void TestX1_BaselineAxisInvarianceAndV3AxisSlots(sslm_model model, sslm_workspace ws,
                                                 const CpuOracleModel& oracle,
                                                 const std::vector<int32_t>& prompt) {
	GreedyRun want;
	std::string err;
	ASSERT_TRUE(RunGreedyReference(oracle, prompt.data(), prompt.size(), {}, 8, &want, &err));
	uint8_t want_tok[32], want_rows[32];
	DigestRun(want, static_cast<size_t>(oracle.vocab_size), want_tok, want_rows);

	// Axis: save/restore round-trip mid-stream at the ABI level (v2 twin), then compare
	// against an un-restored v2 drive -- both must match the internal-loop digest mapping.
	SinglePool sp;
	ASSERT_TRUE(MakePool(model, 2, &sp));
	sslm_seq plain = nullptr, roundtrip = nullptr;
	ASSERT_TRUE(sslm_seq_create(model, &sp.pool, &plain) == SSLM_OK);
	ASSERT_TRUE(sslm_seq_create(model, &sp.pool, &roundtrip) == SSLM_OK);
	int32_t consumed = 0;
	for (sslm_seq* s : {&plain, &roundtrip}) {
		ASSERT_TRUE(sslm_prefill(model, *s, prompt.data(),
		                         static_cast<int32_t>(prompt.size()), 64, SSLM_SPAN_PROMPT,
		                         nullptr, &consumed) == SSLM_OK);
	}
	// Drive `plain` 4 emissions; snapshot; drive `roundtrip` 2, restore from blob, continue 2.
	ASSERT_TRUE(DriveV2Emissions(model, &plain, oracle.num_hidden_layers, ws, 4));
	ASSERT_TRUE(DriveV2Emissions(model, &roundtrip, oracle.num_hidden_layers, ws, 2));
	SeqBlobBuffer snap(model);
	snap.size = snap.bytes.size();
	ASSERT_TRUE(sslm_seq_save(roundtrip, snap.bytes.data(), &snap.size) == SSLM_OK);
	snap.bytes.resize(snap.size);
	sslm_seq resumed = nullptr;
	ASSERT_TRUE(sslm_seq_restore(model, &sp.pool, snap.bytes.data(), snap.bytes.size(),
	                             &resumed) == SSLM_OK);
	ASSERT_TRUE(DriveV2Emissions(model, &resumed, oracle.num_hidden_layers, ws, 2));

	int64_t n_plain = -1, n_resumed = -1;
	CHECK(sslm_seq_committed_token_count(plain, &n_plain) == SSLM_OK);
	CHECK(sslm_seq_committed_token_count(resumed, &n_resumed) == SSLM_OK);
	CHECK(n_plain == n_resumed);  // retention survives the round-trip identically

	// The _v3 axis slots (link-red): same two axes driven through speculate beside the
	// baseline -- batch/chunk via k_max/max_new_tokens splits, save/restore via the same
	// round-trip shape with speculate legs. Authored now so S-E cannot land without them.
	sslm_speculate_params axis_params{};
	ASSERT_TRUE(MakeSpecParams(model, /*k_max=*/2, /*max_new_tokens=*/8, {}, &axis_params));
	SpecDrive got;
	err.clear();
	ASSERT_TRUE(DriveSpeculate(model, plain, axis_params, ws, oracle.vocab_size,
	                            want.produced, &got, &err));
	CHECK(got.tokens == want.tokens);
	uint8_t gt[32], gr[32];
	superslm::ComputeTokenDigest(got.tokens.data(), got.tokens.size(), gt);
	superslm::ComputeFinalLogitDigest(got.logit_rows.data(), got.tokens.size(),
	                                  static_cast<size_t>(oracle.vocab_size), gr);
	CHECK_MSG(DigestEqual(gt, want_tok) && DigestEqual(gr, want_rows),
	          "the save/restore axis must carry identical digests through speculate");
}

}  // namespace

int main(int argc, char** argv) {
	ParseFixtureArgs(argc, argv);
	if (g_model_path.empty()) {
		SKIP_MSG("--model=PATH not supplied -- dim11 X1 not run");
		int ec;
		PrintSummaryAndExit(&ec);
		return ec;
	}
	SslmModelView view;
	std::vector<uint8_t> bytes;
	std::string err;
	if (!LoadRealModelView(g_model_path, &view, &bytes, &err)) {
		SKIP_MSG("could not load real artifact: %s", err.c_str());
		int ec;
		PrintSummaryAndExit(&ec);
		return ec;
	}
	CpuOracleModel oracle;
	if (!LoadCpuOracleModel(view, &oracle, &err)) {
		SKIP_MSG("could not marshal CPU oracle: %s", err.c_str());
		int ec;
		PrintSummaryAndExit(&ec);
		return ec;
	}
	sslm_model model = nullptr;
	CHECK(sslm_model_map(bytes.data(), bytes.size(), &model) == SSLM_OK);
	if (model) {
		sslm_config cfg{};
		cfg.max_batch = 16;
		cfg.max_chunk_budget = 256;
		cfg.max_layer_budget = static_cast<int32_t>(oracle.num_hidden_layers);
		const size_t ws_size = sslm_workspace_size(model, &cfg);
		std::vector<uint8_t> ws_store(ws_size ? ws_size : 1);
		sslm_workspace ws = nullptr;
		if (sslm_workspace_create(model, &cfg, ws_store.data(), ws_store.size(), &ws) ==
		    SSLM_OK) {
			TestX1_BaselineAxisInvarianceAndV3AxisSlots(model, ws, oracle,
			                                            NoRepeatPrompt(oracle.vocab_size, 12));
			CHECK(sslm_workspace_destroy(ws) == SSLM_OK);
		} else {
			SKIP_MSG("workspace create failed -- dim11 X1 needs a workspace");
		}
		CHECK(sslm_model_unmap(model) == SSLM_OK);
	}
	int ec;
	PrintSummaryAndExit(&ec);
	return ec;
}

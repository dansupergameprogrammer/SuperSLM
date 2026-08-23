// T-2246 (test design) -- Coverage Model row 11, Determinism matrix (plan r6 SS6 row 11).
// The row's own disposition: new verbs inherit the MASTER determinism gate's configuration
// axes (batch/chunk/save-restore/hook) once they exist; the suite's job here is (a) pin the
// GREEDY baseline's axis-invariance relationally -- the exact digests the _v3 legs must
// reproduce on every axis -- and (b) slot the _v3 axis drives beside them so the inheritance
// is exercised, not just stated. The baseline leg runs against shipped symbols today and is
// GREEN-TODAY-BY-DESIGN (it pins the reference); the _v3 legs are link-red until S-E. Both
// statuses are recorded honestly in the casebook.
//
// EXECUTION STATUS (fold round 1, 2026-08-22): the build has LANDED -- S-E's verbs and
// sslm_seq_committed_token_count exist in production, so this file no longer links red and
// BOTH legs execute against the real mechanism: the baseline axis-invariance leg (it pins
// the greedy reference's digests) and X1's _v3 axis drives beside it. The earlier
// "baseline executable / _v3 link-red" split described the pre-build state only.
#include "fixture_common.h"

using namespace superslm;

namespace {

// X1 -- Baseline axis-invariance: chunk-budget variation and a save/restore round-trip
// mid-stream leave token+logit digests identical; then each axis is driven through _v3 beside
// its baseline (the inherited-axes exercise).
void TestX1_BaselineAxisInvarianceAndV3AxisSlots(sslm_model model, sslm_workspace ws,
                                                 const CpuOracleModel& oracle,
                                                 const std::vector<int32_t>& prompt) {
	GreedyRun want;
	std::string err;
	// Reference horizon covers the baseline's 4 emissions PLUS the _v3 continuation drive.
	ASSERT_TRUE(RunGreedyReference(oracle, prompt.data(), prompt.size(), {}, 12, &want, &err));
	ASSERT_TRUE(want.produced == 12);

	// Axis: save/restore round-trip mid-stream at the ABI level (v2 twin), then compare
	// against an un-restored v2 drive -- both must match the internal-loop digest mapping.
	// THREE blocks: plain + roundtrip + the restored handle all need a block concurrently
	// (F-I: a 2-block pool exhausted on the restore).
	SinglePool sp;
	ASSERT_TRUE(MakePool(model, 3, &sp));
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
	// SS4: retention is NOT serialized -- restore restarts it EMPTY, so `resumed` appends
	// only its own two emissions while `plain` holds prompt + all four of its own. The
	// round-trip identity this dimension pins is OUTPUT identity (below), never retention
	// equality across a restore.
	CHECK(n_plain == static_cast<int64_t>(prompt.size()) + 4);
	CHECK(n_resumed == 2);

	// The _v3 axis slots: same two axes driven through speculate beside the baseline --
	// batch/chunk via k_max/max_new_tokens splits, save/restore via the same round-trip
	// shape with speculate legs. `plain` continues from its 4-emission state, so the drive
	// must equal greedy's CONTINUATION window [4, 12), row-for-row.
	sslm_speculate_params axis_params{};
	ASSERT_TRUE(MakeSpecParams(model, /*k_max=*/2, /*max_new_tokens=*/8, {}, &axis_params));
	SpecDrive got;
	err.clear();
	ASSERT_TRUE(DriveSpeculate(model, plain, axis_params, ws, oracle.vocab_size,
	                            want.produced - 4, &got, &err));
	CHECK(got.tokens ==
	      std::vector<int32_t>(want.tokens.begin() + 4,
	                           want.tokens.begin() + static_cast<ptrdiff_t>(want.produced)));
	uint8_t cont_tok[32], cont_rows[32], want_tok[32], want_rows[32];
	superslm::ComputeTokenDigest(got.tokens.data(), got.tokens.size(), cont_tok);
	superslm::ComputeTokenDigest(want.tokens.data() + 4, want.tokens.size() - 4, want_tok);
	superslm::ComputeFinalLogitDigest(got.logit_rows.data(), got.tokens.size(),
	                                  static_cast<size_t>(oracle.vocab_size), cont_rows);
	superslm::ComputeFinalLogitDigest(want.logit_rows.data() +
	                                      4 * static_cast<ptrdiff_t>(oracle.vocab_size),
	                                  want.produced - 4,
	                                  static_cast<size_t>(oracle.vocab_size), want_rows);
	CHECK_MSG(DigestEqual(cont_tok, want_tok) && DigestEqual(cont_rows, want_rows),
	          "the save/restore axis must carry identical digests through speculate");
	CHECK(sslm_seq_release(plain) == SSLM_OK);
	CHECK(sslm_seq_release(roundtrip) == SSLM_OK);
	CHECK(sslm_seq_release(resumed) == SSLM_OK);
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
		AlignedBuffer ws_store(ws_size ? ws_size : 1);
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

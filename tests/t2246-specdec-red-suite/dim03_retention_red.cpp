// T-2246 (test design) -- Coverage Model row 3, Committed-token retention lifetime
// (plan r6 SS3.0 lifecycle table, SS6 row 3 with the r6-corrected bound, fold-3's terminal
// checkpoint cell, audit G-6). Output oracles are structurally blind to retention state, so
// every cell asserts the state DIRECTLY through the expected read-back pair
// (sslm_seq_committed_token_count / _peek -- planner routing CM-G1).
//
// Bound under test: committed_tokens.size() <= context_cap + 1 at EVERY checkpoint. The +1 is
// derived (plan SS3.0: retention tracks occupancy plus the always-present unlanded pending
// token); an implementation carrying the superseded cap-only bound fails T7 at exactly that
// checkpoint (+1 excess, probe-executed shape from Claude/Loki/t2246-specdec-strike-2026-08-22.md).
//
// RED STATUS: link-red on the read-back pair and sslm_speculate_step_v3 until S-E/S-B.
#include "fixture_common.h"

using namespace superslm;

namespace {

void CheckBoundAt(sslm_seq seq, int64_t context_cap, const char* checkpoint) {
	int64_t n = -1;
	CHECK(sslm_seq_committed_token_count(seq, &n) == SSLM_OK);
	CHECK_MSG(n >= 0 && n <= context_cap + 1,
	          "retention bound %lld <= context_cap + 1 failed at checkpoint %s",
	          static_cast<long long>(n), checkpoint);
}

// T1/T6 -- Prefill appends admitted ids; each speculate step's emissions append; exact
// lengths at every checkpoint; the bound holds everywhere.
void TestT1_PrefillAndSpeculateAppendsExactLengths(sslm_model model, sslm_workspace ws,
                                                   const CpuOracleModel& oracle,
                                                   const std::vector<int32_t>& prompt) {
	SinglePool sp;
	ASSERT_TRUE(MakeSinglePool(model, &sp));
	sslm_seq seq = nullptr;
	ASSERT_TRUE(sslm_seq_create(model, &sp.pool, &seq) == SSLM_OK);

	int64_t n = -1;
	CHECK(sslm_seq_committed_token_count(seq, &n) == SSLM_OK);
	CHECK(n == 0);  // fresh handle: empty

	int32_t consumed = 0;
	ASSERT_TRUE(sslm_prefill(model, seq, prompt.data(), static_cast<int32_t>(prompt.size()),
	                         64, SSLM_SPAN_PROMPT, nullptr, &consumed) == SSLM_OK);
	CHECK(sslm_seq_committed_token_count(seq, &n) == SSLM_OK);
	CHECK_MSG(n == static_cast<int64_t>(prompt.size()),
	          "prefill must append exactly the %d admitted prompt ids",
	          static_cast<int32_t>(prompt.size()));
	CheckBoundAt(seq, oracle.context_cap, "post-prefill");

	sslm_speculate_params params{};
	ASSERT_TRUE(MakeSpecParams(model, 4, 8, {}, &params));
	SpecDrive got;
	std::string err;
	ASSERT_TRUE(DriveSpeculate(model, seq, params, ws, oracle.vocab_size, 8, &got, &err));
	const int64_t emitted = static_cast<int64_t>(got.tokens.size());
	CHECK(sslm_seq_committed_token_count(seq, &n) == SSLM_OK);
	CHECK_MSG(n == static_cast<int64_t>(prompt.size()) + emitted,
	          "each emission appends its id immediately (SS3.0 decode-step row)");
	// Exact retained CONTENT: the prompt followed by the emissions in order.
	std::vector<int32_t> expect(prompt.begin(), prompt.end());
	expect.insert(expect.end(), got.tokens.begin(), got.tokens.end());
	std::vector<int32_t> peeked(expect.size() + 1, 0);
	int64_t io = static_cast<int64_t>(peeked.size());
	CHECK(sslm_seq_committed_tokens_peek(seq, 0, peeked.data(), &io) == SSLM_OK);
	peeked.resize(static_cast<size_t>(io));
	CHECK(peeked == expect);
	CheckBoundAt(seq, oracle.context_cap, "post-speculate");
}

// T2 -- Reset clears retention together with KV occupancy (no reset lag arm exists:
// sslm_abi.cpp:1833-1834 zeroes both).
void TestT2_ResetClearsRetention(sslm_model model, sslm_workspace ws,
                                 const CpuOracleModel& oracle,
                                 const std::vector<int32_t>& prompt) {
	SinglePool sp;
	ASSERT_TRUE(MakeSinglePool(model, &sp));
	sslm_seq seq = nullptr;
	ASSERT_TRUE(sslm_seq_create(model, &sp.pool, &seq) == SSLM_OK);
	int32_t consumed = 0;
	ASSERT_TRUE(sslm_prefill(model, seq, prompt.data(), static_cast<int32_t>(prompt.size()),
	                         64, SSLM_SPAN_PROMPT, nullptr, &consumed) == SSLM_OK);
	int64_t n = -1;
	CHECK(sslm_seq_committed_token_count(seq, &n) == SSLM_OK);
	CHECK(n > 0);
	CHECK(sslm_seq_reset(seq) == SSLM_OK);
	CHECK(sslm_seq_committed_token_count(seq, &n) == SSLM_OK);
	CHECK_MSG(n == 0, "reset clears retention (SS3.0 table row 4)");
	(void)ws;
	(void)oracle;
}

// T3/T5 -- Adopt clears and restarts retention EMPTY while occupancy copies the prefix
// (sslm_abi.cpp:1909-1918): the lag arm adopt creates, asserted as state here; forced as
// behavior by dim09 W2.
void TestT3_AdoptClearsRestartsEmpty(sslm_model model, const CpuOracleModel& oracle,
                                     const std::vector<int32_t>& prefix_ids) {
	SinglePool sp;
	ASSERT_TRUE(MakePool(model, 2, &sp));
	sslm_prefix prefix = nullptr;
	ASSERT_TRUE(sslm_prefix_begin(model, &sp.pool, &prefix) == SSLM_OK);
	int32_t consumed = 0;
	ASSERT_TRUE(sslm_prefix_prefill(model, prefix, prefix_ids.data(),
	                                static_cast<int32_t>(prefix_ids.size()), 64,
	                                SSLM_SPAN_PROMPT, nullptr, &consumed) == SSLM_OK);
	ASSERT_TRUE(sslm_prefix_freeze(prefix) == SSLM_OK);
	sslm_seq seq = nullptr;
	ASSERT_TRUE(sslm_seq_create(model, &sp.pool, &seq) == SSLM_OK);
	ASSERT_TRUE(sslm_seq_adopt_prefix(seq, prefix) == SSLM_OK);
	int64_t n = -1;
	CHECK(sslm_seq_committed_token_count(seq, &n) == SSLM_OK);
	CHECK_MSG(n == 0, "adopt clears retention; it restarts empty over a copied prefix");
	SeqBlobBuffer blob(model);
	blob.size = blob.bytes.size();
	ASSERT_TRUE(sslm_seq_save(seq, blob.bytes.data(), &blob.size) == SSLM_OK);
	blob.bytes.resize(blob.size);
	CHECK(BlobContextLength(blob.bytes) ==
	      static_cast<int64_t>(prefix_ids.size()));  // occupancy copied with the prefix
	CHECK(sslm_prefix_release(prefix) == SSLM_OK);
	(void)oracle;
}

// T4 -- Restore starts retention EMPTY; occupancy resumes from the blob's context_length
// (SS4 persistence ruling: drafter/retention state is NOT serialized).
void TestT4_RestoreStartsEmpty(sslm_model model, const CpuOracleModel& oracle,
                               const std::vector<int32_t>& prompt) {
	SinglePool sp;
	ASSERT_TRUE(MakePool(model, 2, &sp));
	sslm_seq src = nullptr;
	ASSERT_TRUE(sslm_seq_create(model, &sp.pool, &src) == SSLM_OK);
	int32_t consumed = 0;
	ASSERT_TRUE(sslm_prefill(model, src, prompt.data(), static_cast<int32_t>(prompt.size()),
	                         64, SSLM_SPAN_PROMPT, nullptr, &consumed) == SSLM_OK);
	SeqBlobBuffer blob(model);
	blob.size = blob.bytes.size();
	ASSERT_TRUE(sslm_seq_save(src, blob.bytes.data(), &blob.size) == SSLM_OK);
	blob.bytes.resize(blob.size);
	int64_t before = -1;
	CHECK(sslm_seq_committed_token_count(src, &before) == SSLM_OK);
	CHECK(before == static_cast<int64_t>(prompt.size()));

	sslm_seq restored = nullptr;
	ASSERT_TRUE(
	    RestoreWithTamperedContextLength(model, &sp.pool, blob.bytes,
	                                     BlobContextLength(blob.bytes), &restored));
	int64_t after = -1;
	CHECK(sslm_seq_committed_token_count(restored, &after) == SSLM_OK);
	CHECK_MSG(after == 0, "restore starts retention EMPTY (not serialized)");
	SeqBlobBuffer rblob(model);
	rblob.size = rblob.bytes.size();
	ASSERT_TRUE(sslm_seq_save(restored, rblob.bytes.data(), &rblob.size) == SSLM_OK);
	rblob.bytes.resize(rblob.size);
	CHECK(BlobContextLength(rblob.bytes) == BlobContextLength(blob.bytes));  // occupancy resumes
	(void)oracle;
}

// T7 -- Cache-filling terminal checkpoint at toy scale (fold-3's own cell): drive the
// retention state machine directly to occupancy context_cap -- pure bookkeeping, no model
// weights beyond seeding one real tiny sequence for a valid blob -- and assert the EXACT
// terminal length context_cap + 1. An implementation bounded by cap alone fails right here.
void TestT7_TerminalCheckpointCapPlusOne(sslm_model model, sslm_workspace ws,
                                         const CpuOracleModel& oracle,
                                         const std::vector<int32_t>& prompt) {
	const int64_t cap = oracle.context_cap;
	if (cap < 8) {
		SKIP_MSG("context_cap too small to stage the terminal checkpoint");
		return;
	}
	SinglePool sp;
	ASSERT_TRUE(MakeSinglePool(model, &sp));
	sslm_seq seed = nullptr;
	ASSERT_TRUE(sslm_seq_create(model, &sp.pool, &seed) == SSLM_OK);
	int32_t consumed = 0;
	ASSERT_TRUE(sslm_prefill(model, seed, prompt.data(), static_cast<int32_t>(prompt.size()),
	                         64, SSLM_SPAN_PROMPT, nullptr, &consumed) == SSLM_OK);
	SeqBlobBuffer seed_blob(model);
	size_t sz = seed_blob.size;
	ASSERT_TRUE(sslm_seq_save(seed, seed_blob.bytes.data(), &sz) == SSLM_OK);
	seed_blob.bytes.resize(sz);
	CHECK(sslm_seq_release(seed) == SSLM_OK);

	// Terminal-step entry state: occupancy cap-1 with the pending token unlanded (the shape
	// the shipped guard admits and plan SS3.0 derives the +1 from).
	sslm_seq near_end = nullptr;
	ASSERT_TRUE(RestoreWithTamperedContextLength(model, &sp.pool, seed_blob.bytes, cap - 1,
	                                             &near_end));
	int64_t n = -1;
	CHECK(sslm_seq_committed_token_count(near_end, &n) == SSLM_OK);
	CHECK_MSG(n == 0, "restored window starts empty");

	// One final speculated emission: enters at cap-1, lands the final row, commits its
	// emission -- retention MUST read cap + 1 exactly.
	sslm_speculate_params params{};
	ASSERT_TRUE(MakeSpecParams(model, 1, 1, {}, &params));
	std::vector<int32_t> tok(2, 0), rows(2 * static_cast<size_t>(oracle.vocab_size), 0);
	int32_t produced = 0, stop = -1;
	const sslm_status st =
	    sslm_speculate_step_v3(model, near_end, &params, ws, tok.data(), 2, rows.data(),
	                           static_cast<int32_t>(rows.size()), &produced, &stop);
	CHECK(st == SSLM_OK);
	CHECK(produced == 1);
	CHECK(sslm_seq_committed_token_count(near_end, &n) == SSLM_OK);
	CHECK_MSG(n == cap + 1,
	          "terminal checkpoint: retention reads %lld exactly, not the superseded cap-only "
	          "bound",
	          static_cast<long long>(cap) + 1);
	CheckBoundAt(near_end, cap, "terminal");
}

}  // namespace

int main(int argc, char** argv) {
	ParseFixtureArgs(argc, argv);
	if (g_model_path.empty()) {
		SKIP_MSG("--model=PATH not supplied -- dim03 T1-T7 not run");
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
		if (sslm_workspace_create(model, &cfg, ws_store.data(), ws_store.size(), &ws) !=
		    SSLM_OK) {
			SKIP_MSG("workspace create failed -- dim03 cells need a workspace");
		}
		const std::vector<int32_t> prompt = NoRepeatPrompt(oracle.vocab_size, 12);
		if (ws) {
			TestT1_PrefillAndSpeculateAppendsExactLengths(model, ws, oracle, prompt);
			TestT2_ResetClearsRetention(model, ws, oracle, prompt);
			TestT3_AdoptClearsRestartsEmpty(model, oracle, prompt);
			TestT4_RestoreStartsEmpty(model, oracle, prompt);
			TestT7_TerminalCheckpointCapPlusOne(model, ws, oracle, prompt);
		}
		if (ws) CHECK(sslm_workspace_destroy(ws) == SSLM_OK);
		CHECK(sslm_model_unmap(model) == SSLM_OK);
	}
	int ec;
	PrintSummaryAndExit(&ec);
	return ec;
}

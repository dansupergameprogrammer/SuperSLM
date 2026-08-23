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
// EXECUTION STATUS (fold round 1, 2026-08-22): S-B..S-E have LANDED -- the read-back pair and
// sslm_speculate_step_v3 exist in production under the recorded names; every cell executes
// against the real mechanism.
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
	CHECK(sslm_seq_release(seq) == SSLM_OK);
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
	CHECK(sslm_seq_release(seq) == SSLM_OK);
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
	CHECK(sslm_seq_release(seq) == SSLM_OK);
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
	CHECK(sslm_seq_release(src) == SSLM_OK);
	CHECK(sslm_seq_release(restored) == SSLM_OK);
	(void)oracle;
}

// T7 -- Cache-filling terminal checkpoint (fold-3's own cell, fold-round-1 reconciled against
// SS3.0's lifecycle table): retention appends happen on PREFILL ADMISSIONS and on EMISSIONS
// only, and restore/adopt start retention EMPTY -- so the terminal shape occupancy ==
// context_cap with retention == context_cap + 1 is realizable exactly one way: prefill
// cap-1 admissions, then two v2 emissions. The first rides the prefill's logits-ready state
// for free; the second lands its predecessor's row at position cap-1 and commits the
// (cap+1)-th retained id as the always-present unlanded pending token. An implementation
// bounded by cap alone fails right here (+1 excess). The old restore-staged staging was
// self-contradictory: a restored window starts EMPTY, so one emission from it can never read
// cap + 1. Speculate is likewise unusable as the driver here: from occupancy cap-1 a
// pending-token entry needs k+1 rows and a logits-ready entry emits only its free token --
// neither reaches the terminal shape.
void TestT7_TerminalCheckpointCapPlusOne(sslm_model model, sslm_workspace ws,
                                         const CpuOracleModel& oracle,
                                         const std::vector<int32_t>& prompt) {
	const int64_t cap = oracle.context_cap;
	// Drivable bound (dated 2026-08-22): the honest construction prefills cap-1 admissions,
	// so an artifact whose cap exceeds this bound SKIPs rather than hangs the suite.
	constexpr int64_t kDrivableCap = 512;
	if (cap < 8) {
		SKIP_MSG("context_cap too small to stage the terminal checkpoint");
		return;
	}
	if (cap > kDrivableCap) {
		SKIP_MSG("context_cap %lld exceeds the drivable bound %lld -- the terminal-checkpoint "
		         "construction needs cap-1 real admissions",
		         static_cast<long long>(cap), static_cast<long long>(kDrivableCap));
		return;
	}
	SinglePool sp;
	ASSERT_TRUE(MakeSinglePool(model, &sp));
	sslm_seq seq = nullptr;
	ASSERT_TRUE(sslm_seq_create(model, &sp.pool, &seq) == SSLM_OK);

	std::vector<int32_t> window;
	window.reserve(static_cast<size_t>(cap - 1));
	for (int64_t i = 0; i < cap - 1; ++i) {
		window.push_back(static_cast<int32_t>(
		    prompt[static_cast<size_t>(i) % prompt.size()] %
		    static_cast<int32_t>(oracle.vocab_size)));
	}
	int32_t consumed = 0;
	ASSERT_TRUE(sslm_prefill(model, seq, window.data(), static_cast<int32_t>(window.size()),
	                         64, SSLM_SPAN_PROMPT, nullptr, &consumed) == SSLM_OK);
	ASSERT_TRUE(consumed == static_cast<int32_t>(cap - 1));
	int64_t n = -1;
	CHECK(sslm_seq_committed_token_count(seq, &n) == SSLM_OK);
	CHECK_MSG(n == cap - 1, "prefill appended exactly its cap-1 admissions");
	CheckBoundAt(seq, cap, "post-prefill near-cap");

	sslm_decode_params dp{};
	dp.struct_size = sizeof(dp);
	dp.layer_budget = static_cast<int32_t>(oracle.num_hidden_layers);
	int32_t t0 = -1;
	CHECK(sslm_decode_step_v2(model, &seq, 1, &dp, ws, &t0) == SSLM_OK);
	ASSERT_TRUE(t0 >= 0);
	CHECK(sslm_seq_committed_token_count(seq, &n) == SSLM_OK);
	CHECK_MSG(n == cap, "the free logits-ready emission appends without landing a row");

	int32_t t1 = -1;
	CHECK(sslm_decode_step_v2(model, &seq, 1, &dp, ws, &t1) == SSLM_OK);
	ASSERT_TRUE(t1 >= 0);
	CHECK(sslm_seq_committed_token_count(seq, &n) == SSLM_OK);
	CHECK_MSG(n == cap + 1,
	          "terminal checkpoint: retention reads %lld exactly (occupancy at cap plus the "
	          "unlanded pending token), not the superseded cap-only bound",
	          static_cast<long long>(cap) + 1);
	CheckBoundAt(seq, cap, "terminal");

	SeqBlobBuffer term_blob(model);
	term_blob.size = term_blob.bytes.size();
	ASSERT_TRUE(sslm_seq_save(seq, term_blob.bytes.data(), &term_blob.size) == SSLM_OK);
	term_blob.bytes.resize(term_blob.size);
	CHECK(BlobContextLength(term_blob.bytes) == cap);  // occupancy is genuinely AT the cap

	// The terminal state is stable: with zero rows free, another v2 emission must reject --
	// pinning that the count above was earned at real full occupancy, not by skipped guards.
	int32_t t2 = -1;
	const sslm_status st = sslm_decode_step_v2(model, &seq, 1, &dp, ws, &t2);
	CHECK_MSG(st == SSLM_CONTEXT_CAP_EXCEEDED,
	          "a further emission past full occupancy rejects SSLM_CONTEXT_CAP_EXCEEDED");
	CHECK(sslm_seq_release(seq) == SSLM_OK);
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
		AlignedBuffer ws_store(ws_size ? ws_size : 1);
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

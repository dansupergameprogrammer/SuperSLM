// T-2246 (test design) -- Coverage Model row 2, Rollback exactness (plan r6 SS6 row 2, SS3.3,
// SS5 S-A rollback cells, audit N-4's adopted scoping). Every equality is asserted over the
// canonicalized save-blob comparison -- fixture_common.h's CanonicalizedBlobsEqual implements
// plan SS3.3 verbatim: header fields (including kv_saturation_count and current_token),
// residual, and anti-LM regions byte-exact; live KV rows below context_length byte-exact;
// raw bytes beyond excluded (path-dependent by design, sslm_abi.cpp:1864-1869).
//
// EXECUTION STATUS (fold round 1, 2026-08-22): S-B..S-E have LANDED -- sslm_speculate_step_v3
// and sslm_seq_committed_token_count exist in production under the recorded names, so this
// file no longer links red; every cell executes against the real mechanism.
#include "fixture_common.h"

using namespace superslm;

namespace {

struct TwinFixture {
	SinglePool pool;
	sslm_seq spec_seq = nullptr;   // driven through _v3
	sslm_seq greedy_seq = nullptr; // never speculated; the never-speculated state witness
};

bool MakeTwin(sslm_model model, const std::vector<int32_t>& prompt, TwinFixture* out) {
	if (!MakePool(model, 2, &out->pool)) return false;
	if (sslm_seq_create(model, &out->pool.pool, &out->spec_seq) != SSLM_OK) return false;
	if (sslm_seq_create(model, &out->pool.pool, &out->greedy_seq) != SSLM_OK) return false;
	int32_t consumed = 0;
	return sslm_prefill(model, out->spec_seq, prompt.data(),
	                    static_cast<int32_t>(prompt.size()), 64, SSLM_SPAN_PROMPT, nullptr,
	                    &consumed) == SSLM_OK &&
	       sslm_prefill(model, out->greedy_seq, prompt.data(),
	                    static_cast<int32_t>(prompt.size()), 64, SSLM_SPAN_PROMPT, nullptr,
	                    &consumed) == SSLM_OK;
}

void SaveInto(sslm_seq seq, SeqBlobBuffer* blob) {
	blob->size = blob->bytes.size();
	const sslm_status st = sslm_seq_save(seq, blob->bytes.data(), &blob->size);
	CHECK(st == SSLM_OK);
	blob->bytes.resize(blob->size);
}

// RS-G1 direct retention read-back: count and CONTENT, not just outputs. Asserts the
// sequence's retained window reads pre + emitted ids -- the prompt followed by exactly this
// drive's emissions in order.
void CheckRetentionReadBack(sslm_seq seq, int64_t pre, const std::vector<int32_t>& prompt,
                            const std::vector<int32_t>& emitted) {
	int64_t n = -1;
	CHECK(sslm_seq_committed_token_count(seq, &n) == SSLM_OK);
	CHECK_MSG(n == pre + static_cast<int64_t>(emitted.size()),
	          "post-drive retention must read prompt + emitted prefix");
	std::vector<int32_t> expect(prompt.begin(), prompt.end());
	expect.insert(expect.end(), emitted.begin(), emitted.end());
	if (n != pre + static_cast<int64_t>(emitted.size())) return;
	std::vector<int32_t> peeked(expect.size() + 1, 0);
	int64_t io = static_cast<int64_t>(peeked.size());
	CHECK(sslm_seq_committed_tokens_peek(seq, 0, peeked.data(), &io) == SSLM_OK);
	peeked.resize(static_cast<size_t>(io));
	CHECK_MSG(peeked == expect, "retained content must be the prompt followed by the "
	                            "drive's emitted ids in order");
}

// B1 -- Mismatch rollback exactness: a natural divergence leaves post-step sequence state
// IDENTICAL to the never-speculated twin at the same emitted prefix -- full struct including
// kv_saturation_count (verify lands K/V for rejected drafts where greedy never does; the
// snapshot restore must undo every one) and carried walk-state at greedy's between-steps
// value. Digests over both drives equal too.
void TestB1_PostRejectionFullStructEquality(sslm_model model, sslm_workspace ws,
                                            const CpuOracleModel& oracle,
                                            const std::vector<int32_t>& prompt) {
	TwinFixture fx;
	ASSERT_TRUE(MakeTwin(model, prompt, &fx));

	// RS-G1: record both twins' retention BEFORE driving (prefill pins it at the prompt
	// length) so the post-drive read-back asserts count AND content, not just outputs.
	int64_t spec_pre = -1, twin_pre = -1;
	CHECK(sslm_seq_committed_token_count(fx.spec_seq, &spec_pre) == SSLM_OK);
	CHECK(spec_pre == static_cast<int64_t>(prompt.size()));
	CHECK(sslm_seq_committed_token_count(fx.greedy_seq, &twin_pre) == SSLM_OK);
	CHECK(twin_pre == static_cast<int64_t>(prompt.size()));

	GreedyRun want;
	std::string err;
	ASSERT_TRUE(RunGreedyReference(oracle, prompt.data(), prompt.size(), {}, 16, &want, &err));

	// Spec side: high K maximizes mismatch exposure; drive to the twin prefix length.
	sslm_speculate_params params{};
	ASSERT_TRUE(MakeSpecParams(model, /*k_max=*/8, 16, {}, &params));
	SpecDrive got;
	err.clear();
	ASSERT_TRUE(DriveSpeculate(model, fx.spec_seq, params, ws, oracle.vocab_size,
	                           want.produced, &got, &err));
	CHECK(got.tokens == want.tokens);

	// Twin side: same emitted prefix through pure v2 single-token steps (never speculated).
	ASSERT_TRUE(DriveV2Emissions(model, &fx.greedy_seq, oracle.num_hidden_layers, ws,
	                             want.produced));

	SeqBlobBuffer want_blob(model), got_blob(model);
	SaveInto(fx.greedy_seq, &want_blob);   // never-speculated twin at the SAME prefix
	SaveInto(fx.spec_seq, &got_blob);
	CHECK_MSG(got.tokens.size() == want.tokens.size(), "same emitted prefix required");
	if (got.tokens.size() == want.tokens.size()) {
		std::string why;
		const bool equal = CanonicalizedBlobsEqual(want_blob.bytes, got_blob.bytes,
		                                           oracle.hidden_size,
		                                           static_cast<size_t>(
		                                               sslm_kv_block_size(model)),
		                                           oracle.context_cap, oracle.num_hidden_layers, oracle.num_kv_heads, oracle.head_dim, &why);
		CHECK_MSG(equal, "post-rejection state must equal never-speculated state (%s)",
		          why.c_str());
	}
	CheckRetentionReadBack(fx.spec_seq, spec_pre, prompt, got.tokens);
	CheckRetentionReadBack(fx.greedy_seq, twin_pre, prompt, want.tokens);
	CHECK(sslm_seq_release(fx.spec_seq) == SSLM_OK);
	CHECK(sslm_seq_release(fx.greedy_seq) == SSLM_OK);
	fx.spec_seq = nullptr;
	fx.greedy_seq = nullptr;
}

// B2/B3 -- Save-blob byte equality vs pure greedy (audit N-4, scoped): after a full-K
// ACCEPTANCE and after a REJECTION, sslm_seq_save bytes equal the pure-greedy run's bytes at
// the same emitted prefix over the same scoping (saturation restored / landings replicated).
// The acceptance leg reuses the full-K fixture family's search; when no fixture is found on
// this artifact/corpus the cell SKIPs honestly rather than asserting nothing.
void TestB2_PostAcceptanceBlobEquality(sslm_model model, sslm_workspace ws,
                                       const CpuOracleModel& oracle,
                                       const std::vector<int32_t>& corpus_stream) {
	constexpr int32_t kK = 4;
	std::vector<int32_t> prompt;
	std::string err;
	if (!FindFullKAcceptancePrompt(oracle, corpus_stream, kK, 160, &prompt, &err)) {
		SKIP_MSG("full-K fixture not found: %s", err.c_str());
		return;
	}
	TwinFixture fx;
	ASSERT_TRUE(MakeTwin(model, prompt, &fx));
	GreedyRun want;
	ASSERT_TRUE(RunGreedyReference(oracle, prompt.data(), prompt.size(), {},
	                               static_cast<size_t>(kK) + 1, &want, &err));

	sslm_speculate_params params{};
	ASSERT_TRUE(MakeSpecParams(model, kK, kK + 1, {}, &params));
	SpecDrive got;
	err.clear();
	ASSERT_TRUE(DriveSpeculate(model, fx.spec_seq, params, ws, oracle.vocab_size,
	                           want.produced, &got, &err));
	CHECK(got.tokens == want.tokens);

	// Twin side: same prefix via pure v2 steps.
	ASSERT_TRUE(DriveV2Emissions(model, &fx.greedy_seq, oracle.num_hidden_layers, ws,
	                             want.produced));

	SeqBlobBuffer want_blob(model), got_blob(model);
	SaveInto(fx.greedy_seq, &want_blob);
	SaveInto(fx.spec_seq, &got_blob);
	std::string why;
	const bool equal =
	    CanonicalizedBlobsEqual(want_blob.bytes, got_blob.bytes, oracle.hidden_size,
	                            static_cast<size_t>(sslm_kv_block_size(model)),
	                            oracle.context_cap, oracle.num_hidden_layers, oracle.num_kv_heads, oracle.head_dim, &why);
	CHECK_MSG(equal, "post-full-K save-blob must equal pure greedy's at the same prefix (%s)",
	          why.c_str());
	CHECK(sslm_seq_release(fx.spec_seq) == SSLM_OK);
	CHECK(sslm_seq_release(fx.greedy_seq) == SSLM_OK);
	fx.spec_seq = nullptr;
	fx.greedy_seq = nullptr;
}

}  // namespace

int main(int argc, char** argv) {
	ParseFixtureArgs(argc, argv);
	if (g_model_path.empty()) {
		SKIP_MSG("--model=PATH not supplied -- dim02 B1-B2 not run");
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
			SKIP_MSG("workspace create failed -- dim02 cells need a workspace");
		}

		std::vector<std::vector<int32_t>> prompts;
		std::vector<int32_t> corpus_stream;
		bool have_corpus = false;
		if (!g_corpus_path.empty() && !g_model_tok_path.empty()) {
			sslm_model tok_model = nullptr;
			std::vector<uint8_t> tok_bytes;
			if (ReadFileBytes(g_model_tok_path, &tok_bytes)) {
				CHECK(sslm_model_map(tok_bytes.data(), tok_bytes.size(), &tok_model) == SSLM_OK);
			}
			if (tok_model) {
				std::vector<std::string> utterances;
				if (LoadCorpusUtterances(g_corpus_path, 24, &utterances)) {
					for (const auto& u : utterances) {
						std::vector<int32_t> ids;
						if (TokenizeUtf8(tok_model, u, &ids) && ids.size() >= 8) {
							prompts.push_back(ids);
							corpus_stream.insert(corpus_stream.end(), ids.begin(), ids.end());
						}
					}
					have_corpus = !prompts.empty();
				}
				CHECK(sslm_model_unmap(tok_model) == SSLM_OK);
			}
		}
		if (!have_corpus) prompts.push_back(NoRepeatPrompt(oracle.vocab_size, 12));

		if (ws) {
			TestB1_PostRejectionFullStructEquality(model, ws, oracle, prompts.front());
			if (have_corpus) {
				TestB2_PostAcceptanceBlobEquality(model, ws, oracle, corpus_stream);
			} else {
				SKIP_MSG("B2 full-K fixture requires a tokenized corpus");
			}
		}
		if (ws) CHECK(sslm_workspace_destroy(ws) == SSLM_OK);
		CHECK(sslm_model_unmap(model) == SSLM_OK);
	}
	int ec;
	PrintSummaryAndExit(&ec);
	return ec;
}


// T-2246 (test design) -- Coverage Model row 9, Persistence (plan r6 SS4 ruling, SS6 row 9,
// audit Delta-G3): cold-start-after-restore and post-adopt-prefix output identity -- the two
// lag arms where retention is empty while occupancy is not, which convict any implementation
// deriving verify-entry lengths from retention count instead of the sequence's own
// context_length (SS3.0 positioning invariant).
//
// RED STATUS: link-red on sslm_speculate_step_v3 / sslm_seq_committed_token_count until S-E.
#include "fixture_common.h"

using namespace superslm;

namespace {

// W1 -- Cold-start after restore: save a real driven sequence, restore into a fresh handle,
// speculate IMMEDIATELY: stream + both digests equal pure greedy driven identically from the
// same prompt history; retention reads EMPTY at speculate entry.
void TestW1_PostRestoreImmediateSpeculateIdentity(sslm_model model, sslm_workspace ws,
                                                  const CpuOracleModel& oracle,
                                                  const std::vector<int32_t>& corpus_stream) {
	constexpr int32_t kK = 4;
	std::vector<int32_t> prompt;
	std::string err;
	if (!corpus_stream.empty() &&
	    FindFullKAcceptancePrompt(oracle, corpus_stream, kK, 160, &prompt, &err)) {
		// an accepting window maximizes the discriminator's bite, but any prompt works
	} else {
		prompt = NoRepeatPrompt(oracle.vocab_size, 12);
	}
	GreedyRun want;
	ASSERT_TRUE(RunGreedyReference(oracle, prompt.data(), prompt.size(), {}, 8, &want, &err));

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
	CHECK(sslm_seq_release(src) == SSLM_OK);

	sslm_seq restored = nullptr;
	ASSERT_TRUE(sslm_seq_restore(model, &sp.pool, blob.bytes.data(), blob.bytes.size(),
	                             &restored) == SSLM_OK);
	int64_t n = -1;
	CHECK(sslm_seq_committed_token_count(restored, &n) == SSLM_OK);
	CHECK_MSG(n == 0, "restoration leaves retention EMPTY (SS4 persistence ruling)");

	sslm_speculate_params params{};
	ASSERT_TRUE(MakeSpecParams(model, kK, 8, {}, &params));
	SpecDrive got;
	err.clear();
	ASSERT_TRUE(DriveSpeculate(model, restored, params, ws, oracle.vocab_size, want.produced,
	                           &got, &err));
	CHECK(got.tokens == want.tokens);
	uint8_t wt[32], wr[32], gt[32], gr[32];
	DigestRun(want, static_cast<size_t>(oracle.vocab_size), wt, wr);
	superslm::ComputeTokenDigest(got.tokens.data(), got.tokens.size(), gt);
	superslm::ComputeFinalLogitDigest(got.logit_rows.data(), got.tokens.size(),
	                                  static_cast<size_t>(oracle.vocab_size), gr);
	CHECK(DigestEqual(gt, wt));
	CHECK(DigestEqual(gr, wr));
}

// W2 -- Post-adopt positioning arm (audit Delta-G3): prefill -> adopt_prefix (longer frozen
// prefix) -> immediate speculate equals pure greedy from the adopted state with retention
// EMPTY at entry. Deriving the verify window from retention yields start = 0 over truncated
// history and diverges on the first step -- exactly what this cell convicts.
void TestW2_PostAdoptImmediateSpeculateIdentity(sslm_model model, sslm_workspace ws,
                                                const CpuOracleModel& oracle,
                                                const std::vector<int32_t>& prefix_ids) {
	GreedyRun want;
	std::string err;
	ASSERT_TRUE(
	    RunGreedyReference(oracle, prefix_ids.data(), prefix_ids.size(), {}, 8, &want, &err));

	SinglePool sp;
	ASSERT_TRUE(MakePool(model, 3, &sp));
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
	CHECK_MSG(n == 0, "adoption leaves retention empty over the copied occupancy");

	sslm_speculate_params params{};
	ASSERT_TRUE(MakeSpecParams(model, 4, 8, {}, &params));
	SpecDrive got;
	err.clear();
	ASSERT_TRUE(DriveSpeculate(model, seq, params, ws, oracle.vocab_size, want.produced, &got,
	                           &err));
	CHECK_MSG(got.tokens ==
	              std::vector<int32_t>(want.tokens.begin(),
	                                   want.tokens.begin() +
	                                       static_cast<ptrdiff_t>(got.tokens.size())),
	          "post-adopt speculation must derive its window from occupancy, never retention");
	uint8_t wt[32], wr[32], gt[32], gr[32];
	DigestRunPrefix(want, got.tokens.size(), static_cast<size_t>(oracle.vocab_size), wt, wr);
	superslm::ComputeTokenDigest(got.tokens.data(), got.tokens.size(), gt);
	superslm::ComputeFinalLogitDigest(got.logit_rows.data(), got.tokens.size(),
	                                  static_cast<size_t>(oracle.vocab_size), gr);
	CHECK(DigestEqual(gt, wt));
	CHECK(DigestEqual(gr, wr));

	CHECK(sslm_prefix_release(prefix) == SSLM_OK);
}

// W3 -- Blob format regression guard: SSB3 is UNCHANGED by this mechanism (plan SS6 row 9).
// A never-speculated sequence's save must restore cleanly in a fresh handle at the identical
// size -- the format-stability half the row claims.
void TestW3_BlobFormatUnchangedRegression(sslm_model model, const CpuOracleModel& oracle,
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

	size_t required = 0;
	const sslm_status sizing =
	    sslm_seq_save(src, nullptr, &required);  // BUFFER_TOO_SMALL names the exact size
	CHECK(sizing == SSLM_BUFFER_TOO_SMALL);
	CHECK(required == blob.bytes.size());

	sslm_seq twin = nullptr;
	ASSERT_TRUE(sslm_seq_restore(model, &sp.pool, blob.bytes.data(), blob.bytes.size(),
	                             &twin) == SSLM_OK);
	SeqBlobBuffer rblob(model);
	rblob.size = rblob.bytes.size();
	ASSERT_TRUE(sslm_seq_save(twin, rblob.bytes.data(), &rblob.size) == SSLM_OK);
	rblob.bytes.resize(rblob.size);
	CHECK(rblob.bytes == blob.bytes);  // round-trip byte stability
	(void)oracle;
}

}  // namespace

int main(int argc, char** argv) {
	ParseFixtureArgs(argc, argv);
	if (g_model_path.empty()) {
		SKIP_MSG("--model=PATH not supplied -- dim09 W1-W3 not run");
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
			SKIP_MSG("workspace create failed -- dim09 cells need a workspace");
		}
		std::vector<int32_t> corpus_stream;
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
							corpus_stream.insert(corpus_stream.end(), ids.begin(), ids.end());
						}
					}
				}
				CHECK(sslm_model_unmap(tok_model) == SSLM_OK);
			}
		}
		if (ws) {
			TestW1_PostRestoreImmediateSpeculateIdentity(model, ws, oracle, corpus_stream);
			TestW2_PostAdoptImmediateSpeculateIdentity(model, ws, oracle,
			                                           NoRepeatPrompt(oracle.vocab_size, 14));
			TestW3_BlobFormatUnchangedRegression(model, oracle,
			                                     NoRepeatPrompt(oracle.vocab_size, 12));
			CHECK(sslm_workspace_destroy(ws) == SSLM_OK);
		}
		CHECK(sslm_model_unmap(model) == SSLM_OK);
	}
	int ec;
	PrintSummaryAndExit(&ec);
	return ec;
}

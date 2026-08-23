// T-2246 (test design) -- Coverage Model row 6, Concurrency (audit G-7, plan r6 SS6 row 6):
// the dimension is disposed as an extension obligation on the standing TSan job ("no new
// oracle"), routed not asserted; THIS cell authors the deterministic body those schedules
// wrap -- >=2 disjoint sequences driving speculate-ACCEPT and speculate-REJECT cycles
// interleaved, outputs hashed PER SEQUENCE against solo references. It carries the same
// greens-with-S-E slot the plan assigns to the TSan extension; when the harness extension
// lands it drives these exact fixtures under ThreadSanitizer.
//
// RED STATUS: link-red on sslm_speculate_step_v3 until S-E.
#include "fixture_common.h"

using namespace superslm;

namespace {

// CC1 -- Two disjoint sequences, interleaved drives: sequence A runs accept-seeking cycles on
// a corpus prompt; sequence B runs all-reject cycles on a never-repeating prompt. Each
// sequence's accumulated stream and digest pair must equal its own solo reference run --
// interleaving may change pacing, never content.
void TestCC1_DisjointSequencesAcceptAndRejectCycles(sslm_model model, sslm_workspace ws,
                                                    const CpuOracleModel& oracle,
                                                    const std::vector<int32_t>& accept_prompt) {
	const std::vector<int32_t> reject_prompt = NoRepeatPrompt(oracle.vocab_size, 12);

	GreedyRun want_a, want_b;
	std::string err;
	ASSERT_TRUE(RunGreedyReference(oracle, accept_prompt.data(), accept_prompt.size(), {}, 12,
	                               &want_a, &err));
	ASSERT_TRUE(RunGreedyReference(oracle, reject_prompt.data(), reject_prompt.size(), {}, 8,
	                               &want_b, &err));

	SinglePool sp;
	ASSERT_TRUE(MakePool(model, 2, &sp));
	sslm_seq seq_a = nullptr, seq_b = nullptr;
	ASSERT_TRUE(sslm_seq_create(model, &sp.pool, &seq_a) == SSLM_OK);
	ASSERT_TRUE(sslm_seq_create(model, &sp.pool, &seq_b) == SSLM_OK);
	int32_t consumed = 0;
	ASSERT_TRUE(sslm_prefill(model, seq_a, accept_prompt.data(),
	                         static_cast<int32_t>(accept_prompt.size()), 64, SSLM_SPAN_PROMPT,
	                         nullptr, &consumed) == SSLM_OK);
	ASSERT_TRUE(sslm_prefill(model, seq_b, reject_prompt.data(),
	                         static_cast<int32_t>(reject_prompt.size()), 64, SSLM_SPAN_PROMPT,
	                         nullptr, &consumed) == SSLM_OK);

	sslm_speculate_params pa{}, pb{};
	ASSERT_TRUE(MakeSpecParams(model, 6, 2, {}, &pa));  // A: deep drafts, short batches
	ASSERT_TRUE(MakeSpecParams(model, 4, 1, {}, &pb));  // B: shallow drafts, single emissions

	SpecDrive got_a, got_b;
	size_t emitted_b = 0;
	// Strict alternation until both reach their budget or stop.
	while (got_a.tokens.size() < want_a.produced || emitted_b < want_b.produced) {
		if (got_a.tokens.size() < want_a.produced) {
			pa.max_new_tokens = static_cast<int32_t>(
			    want_a.produced - got_a.tokens.size() > 2 ? 2
			                                             : want_a.produced - got_a.tokens.size());
			std::vector<int32_t> tok(static_cast<size_t>(pa.max_new_tokens) + 1, 0);
			std::vector<int32_t> rows((static_cast<size_t>(pa.max_new_tokens) + 1) *
			                              static_cast<size_t>(oracle.vocab_size), 0);
			int32_t produced = 0, stop = -1;
			CHECK(sslm_speculate_step_v3(model, seq_a, &pa, ws, tok.data(),
			                             static_cast<int32_t>(tok.size()), rows.data(),
			                             static_cast<int32_t>(rows.size()), &produced,
			                             &stop) == SSLM_OK);
			got_a.tokens.insert(got_a.tokens.end(), tok.begin(), tok.begin() + produced);
			got_a.logit_rows.insert(
			    got_a.logit_rows.end(), rows.begin(),
			    rows.begin() +
			        static_cast<ptrdiff_t>(static_cast<size_t>(produced) *
			                               static_cast<size_t>(oracle.vocab_size)));
			if (stop == SSLM_SPECULATE_STOP_TOKEN_MATCHED) break;
		}
		if (emitted_b < want_b.produced) {
			std::vector<int32_t> tok(8, 0), rows(8 * static_cast<size_t>(oracle.vocab_size), 0);
			int32_t produced = 0, stop = -1;
			CHECK(sslm_speculate_step_v3(model, seq_b, &pb, ws, tok.data(), 8, rows.data(), 8,
			                             &produced, &stop) == SSLM_OK);
			got_b.tokens.insert(got_b.tokens.end(), tok.begin(), tok.begin() + produced);
			got_b.logit_rows.insert(
			    got_b.logit_rows.end(), rows.begin(),
			    rows.begin() +
			        static_cast<ptrdiff_t>(static_cast<size_t>(produced) *
			                                       static_cast<size_t>(oracle.vocab_size)));
			emitted_b = got_b.tokens.size();
			if (stop == SSLM_SPECULATE_STOP_TOKEN_MATCHED) break;
		}
	}

	CHECK(got_a.tokens ==
	      std::vector<int32_t>(want_a.tokens.begin(),
	                           want_a.tokens.begin() + static_cast<ptrdiff_t>(got_a.tokens.size())));
	CHECK(got_b.tokens ==
	      std::vector<int32_t>(want_b.tokens.begin(),
	                           want_b.tokens.begin() + static_cast<ptrdiff_t>(got_b.tokens.size())));
	uint8_t ta[32], ra[32], tb[32], rb[32];
	superslm::ComputeTokenDigest(got_a.tokens.data(), got_a.tokens.size(), ta);
	superslm::ComputeFinalLogitDigest(got_a.logit_rows.data(), got_a.tokens.size(),
	                                  static_cast<size_t>(oracle.vocab_size), ra);
	superslm::ComputeTokenDigest(want_a.tokens.data(), got_a.tokens.size(), tb);
	superslm::ComputeFinalLogitDigest(want_a.logit_rows.data(), got_a.tokens.size(),
	                                  static_cast<size_t>(oracle.vocab_size), rb);
	CHECK(DigestEqual(ta, tb));
	CHECK(DigestEqual(ra, rb));
}

}  // namespace

int main(int argc, char** argv) {
	ParseFixtureArgs(argc, argv);
	if (g_model_path.empty()) {
		SKIP_MSG("--model=PATH not supplied -- dim06 CC1 not run");
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
			SKIP_MSG("workspace create failed -- dim06 CC1 needs a workspace");
		}
		std::vector<int32_t> accept_prompt = NoRepeatPrompt(oracle.vocab_size, 12);
		if (!g_corpus_path.empty() && !g_model_tok_path.empty()) {
			sslm_model tok_model = nullptr;
			std::vector<uint8_t> tok_bytes;
			if (ReadFileBytes(g_model_tok_path, &tok_bytes)) {
				CHECK(sslm_model_map(tok_bytes.data(), tok_bytes.size(), &tok_model) == SSLM_OK);
			}
			if (tok_model) {
				std::vector<std::string> utterances;
				if (LoadCorpusUtterances(g_corpus_path, 8, &utterances)) {
					for (const auto& u : utterances) {
						std::vector<int32_t> ids;
						if (TokenizeUtf8(tok_model, u, &ids) && ids.size() >= 8) {
							accept_prompt = ids;
							break;
						}
					}
				}
				CHECK(sslm_model_unmap(tok_model) == SSLM_OK);
			}
		}
		if (ws) {
			TestCC1_DisjointSequencesAcceptAndRejectCycles(model, ws, oracle, accept_prompt);
			CHECK(sslm_workspace_destroy(ws) == SSLM_OK);
		}
		CHECK(sslm_model_unmap(model) == SSLM_OK);
	}
	int ec;
	PrintSummaryAndExit(&ec);
	return ec;
}

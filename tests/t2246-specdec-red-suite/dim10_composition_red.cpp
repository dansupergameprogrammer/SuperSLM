// T-2246 (test design) -- Coverage Model row 10, Composition (audit G-2 + the fold's 2-sequence
// widen, plan r6 SS6 row 10): versioned-additive ABI means v2 callers keep driving sequences
// the new verbs also drive. One sequence is driven _v3-speculate <-> v2-decode alternately in
// BOTH orders, across an acceptance and across a rejection, with the v2 legs driving a
// TWO-sequence batch (the speculated sequence plus a bystander); stream + digests equal a pure
// v2 run of the same prompt; retention is asserted DIRECTLY at each event (output oracles
// cannot see retention staleness). P2 is ST-1's stream-identity symmetry leg.
//
// RED STATUS: link-red on sslm_speculate_step_v3 / sslm_seq_committed_token_count until S-E.
#include "fixture_common.h"

using namespace superslm;

namespace {

struct InterleaveState {
	sslm_seq spec_seq = nullptr;
	sslm_seq bystander = nullptr;
	int64_t committed = -1;
};

// P1/P2 -- Mixed-drive identity and direct retention assertions.
void TestP1_InterleaveV2V3BothOrdersAcrossAcceptAndReject(
    sslm_model model, sslm_workspace ws, const CpuOracleModel& oracle,
    const std::vector<int32_t>& corpus_stream) {
	constexpr int32_t kK = 4;
	std::vector<int32_t> prompt;
	std::string err;
	const bool have_window =
	    !corpus_stream.empty() &&
	    FindFullKAcceptancePrompt(oracle, corpus_stream, kK, 160, &prompt, &err);
	if (!have_window) prompt = NoRepeatPrompt(oracle.vocab_size, 12);

	GreedyRun want;
	ASSERT_TRUE(RunGreedyReference(oracle, prompt.data(), prompt.size(), {}, 12, &want, &err));

	SinglePool sp;
	ASSERT_TRUE(MakePool(model, 3, &sp));
	InterleaveState st;
	st.committed = -1;
	ASSERT_TRUE(sslm_seq_create(model, &sp.pool, &st.spec_seq) == SSLM_OK);
	ASSERT_TRUE(sslm_seq_create(model, &sp.pool, &st.bystander) == SSLM_OK);
	int32_t consumed = 0;
	ASSERT_TRUE(sslm_prefill(model, st.spec_seq, prompt.data(),
	                         static_cast<int32_t>(prompt.size()), 64, SSLM_SPAN_PROMPT,
	                         nullptr, &consumed) == SSLM_OK);
	CHECK(sslm_seq_committed_token_count(st.spec_seq, &st.committed) == SSLM_OK);
	const int64_t committed_start = st.committed;

	sslm_decode_params dp{};
	dp.struct_size = sizeof(dp);
	dp.layer_budget = static_cast<int32_t>(oracle.num_hidden_layers);
	sslm_speculate_params vp{};
	ASSERT_TRUE(MakeSpecParams(model, kK, 3, {}, &vp));

	SpecDrive got;
	size_t emitted = 0;
	int order_flip = 0;  // alternates which verb LEADS each pair of calls: both orders covered
	while (emitted < want.produced) {
		const bool speculate_first = (order_flip % 2 == 0);
		++order_flip;

		auto do_speculate = [&]() {
			vp.max_new_tokens =
			    static_cast<int32_t>(want.produced - emitted > 3 ? 3 : want.produced - emitted);
			std::vector<int32_t> tok(static_cast<size_t>(vp.max_new_tokens) + 1, 0);
			std::vector<int32_t> rows((static_cast<size_t>(vp.max_new_tokens) + 1) *
			                              static_cast<size_t>(oracle.vocab_size), 0);
			int32_t produced = 0, stop = -1;
			const sslm_status s =
			    sslm_speculate_step_v3(model, st.spec_seq, &vp, ws, tok.data(),
			                           static_cast<int32_t>(tok.size()), rows.data(),
			                           static_cast<int32_t>(rows.size()), &produced, &stop);
			CHECK(s == SSLM_OK);
			if (s != SSLM_OK) return false;
			got.tokens.insert(got.tokens.end(), tok.begin(), tok.begin() + produced);
			got.logit_rows.insert(
			    got.logit_rows.end(), rows.begin(),
			    rows.begin() + static_cast<ptrdiff_t>(static_cast<size_t>(produced) *
			                                          oracle.vocab_size));
			emitted = got.tokens.size();
			int64_t n = -1;
			CHECK(sslm_seq_committed_token_count(st.spec_seq, &n) == SSLM_OK);
			CHECK(n == committed_start + static_cast<int64_t>(emitted));  // direct retention read
			return stop != SSLM_SPECULATE_STOP_TOKEN_MATCHED && produced > 0;
		};
		auto do_v2_pair = [&](int32_t budget) {
			for (int32_t i = 0; i < budget && emitted < want.produced; ++i) {
				// TWO-sequence batch: speculated sequence + bystander, the fold's widening.
				int32_t outs[2] = {0, 0};
				sslm_seq batch[2] = {st.spec_seq, st.bystander};
				const sslm_status s = sslm_decode_step_v2(model, batch, 2, &dp, ws, outs);
				CHECK(s == SSLM_OK);
				if (s != SSLM_OK || outs[0] < 0) return false;
				got.tokens.push_back(outs[0]);
				emitted = got.tokens.size();
				int64_t n = -1;
				CHECK(sslm_seq_committed_token_count(st.spec_seq, &n) == SSLM_OK);
				CHECK(n == committed_start + static_cast<int64_t>(emitted));
			}
			return true;
		};
		(void)do_v2_pair;  // driven below through both orders

		if (speculate_first) {
			if (!do_speculate()) break;
			if (!do_v2_pair(1)) break;
		} else {
			if (!do_v2_pair(1)) break;
			if (!do_speculate()) break;
		}
	}
	CHECK(got.tokens ==
	      std::vector<int32_t>(want.tokens.begin(),
	                           want.tokens.begin() + static_cast<ptrdiff_t>(got.tokens.size())));
	uint8_t gt[32], gr[32], wt[32], wr[32];
	superslm::ComputeTokenDigest(got.tokens.data(), got.tokens.size(), gt);
	superslm::ComputeFinalLogitDigest(got.logit_rows.data(), got.tokens.size(),
	                                  static_cast<size_t>(oracle.vocab_size), gr);
	DigestRunPrefix(want, got.tokens.size(), static_cast<size_t>(oracle.vocab_size), wt, wr);
	CHECK(DigestEqual(gt, wt));
	CHECK(DigestEqual(gr, wr));

	// Rejection leg: hostile params mid-mixed-drive reject and leave BOTH batch members live.
	sslm_speculate_params bad{};
	ASSERT_TRUE(MakeSpecParams(model, 4, 4, {}, &bad));
	bad.struct_size = sizeof(bad) - 4u;
	std::vector<int32_t> tok(8, 0), rows(8 * static_cast<size_t>(oracle.vocab_size), 0);
	int32_t produced = -1, stop = -1;
	CHECK(sslm_speculate_step_v3(model, st.spec_seq, &bad, ws, tok.data(), 8, rows.data(), 8,
	                             &produced, &stop) == SSLM_INVALID_ARGUMENT);
	int32_t outs[2] = {0, 0};
	sslm_seq batch[2] = {st.spec_seq, st.bystander};
	CHECK(sslm_decode_step_v2(model, batch, 2, &dp, ws, outs) == SSLM_OK);
}

}  // namespace

int main(int argc, char** argv) {
	ParseFixtureArgs(argc, argv);
	if (g_model_path.empty()) {
		SKIP_MSG("--model=PATH not supplied -- dim10 P1 not run");
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
			SKIP_MSG("workspace create failed -- dim10 P1 needs a workspace");
		}
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
							corpus_stream.insert(corpus_stream.end(), ids.begin(), ids.end());
						}
					}
					have_corpus = !corpus_stream.empty();
				}
				CHECK(sslm_model_unmap(tok_model) == SSLM_OK);
			}
		}
		(void)have_corpus;
		if (ws) {
			TestP1_InterleaveV2V3BothOrdersAcrossAcceptAndReject(model, ws, oracle,
			                                                     corpus_stream);
			CHECK(sslm_workspace_destroy(ws) == SSLM_OK);
		}
		CHECK(sslm_model_unmap(model) == SSLM_OK);
	}
	int ec;
	PrintSummaryAndExit(&ec);
	return ec;
}

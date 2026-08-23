// T-2246 (test design) -- Coverage Model row 8, Guard vitality composed with batch emission
// (plan r6 SS3.2 batch walk, SS3.5 guards, SS5 S-A boundary cells; audits Delta-G1/Delta-G2).
// This file also carries ALL FOUR commissioning-time residuals as pinned cells:
//   R2-W1 -> V1/V2/V3's feed-accounting assertions (occupancy and kv_saturation_count stated
//            in FEED terms: E emissions from pending-token entry cost E feeds and end at
//            context_length C + E with the final emission resting unfed-pending --
//            forward_sites.cpp:2595-2630 loop shape; logits-ready/adopt entry costs E-1).
//   R2-W2 + r-G2 -> V3 (the zero-budget entry arm; one cell satisfying both seats' occupants,
//            filed independently by plan review and coverage audit).
//   r-G1  -> V2's post-step state leg (cell (b)'s missing fifth property).
//
// RED STATUS: link-red on sslm_speculate_step_v3 / sslm_seq_committed_token_count until S-E.
#include "fixture_common.h"

using namespace superslm;

namespace {

struct FullKFx {
	std::vector<int32_t> prompt;
	GreedyRun want;
};

bool FindFullK(const CpuOracleModel& oracle, const std::vector<int32_t>& corpus_stream,
               FullKFx* out) {
	std::string err;
	if (!corpus_stream.empty() &&
	    FindFullKAcceptancePrompt(oracle, corpus_stream, 4, 160, &out->prompt, &err)) {
		return RunGreedyReference(oracle, out->prompt.data(), out->prompt.size(), {},
		                          /*max_new=*/16, &out->want, &err);
	}
	return false;
}

// V1 -- Cap-straddling batch (boundary cell (a)): remaining budget r strictly inside the
// would-be-emitted length emits EXACTLY r, both digests cover exactly those r logit rows,
// MaxTokensReached matches capped pure greedy, and post-step sequence state equals capped
// greedy over the canonicalized comparison -- with R2-W1's feed accounting pinned numerically.
void TestV1_CapStraddlingBatchTruncationAndFeedAccounting(sslm_model model, sslm_workspace ws,
                                                          const CpuOracleModel& oracle,
                                                          const FullKFx& fx) {
	constexpr size_t kR = 2;  // 0 < r < would-be emitted (>= K+1 = 5)
	SinglePool sp;
	ASSERT_TRUE(MakePool(model, 2, &sp));
	sslm_seq spec = nullptr, twin = nullptr;
	ASSERT_TRUE(sslm_seq_create(model, &sp.pool, &spec) == SSLM_OK);
	ASSERT_TRUE(sslm_seq_create(model, &sp.pool, &twin) == SSLM_OK);
	int32_t consumed = 0;
	ASSERT_TRUE(sslm_prefill(model, spec, fx.prompt.data(),
	                         static_cast<int32_t>(fx.prompt.size()), 64, SSLM_SPAN_PROMPT,
	                         nullptr, &consumed) == SSLM_OK);
	ASSERT_TRUE(sslm_prefill(model, twin, fx.prompt.data(),
	                         static_cast<int32_t>(fx.prompt.size()), 64, SSLM_SPAN_PROMPT,
	                         nullptr, &consumed) == SSLM_OK);

	SeqBlobBuffer pre_blob(model);
	pre_blob.size = pre_blob.bytes.size();
	ASSERT_TRUE(sslm_seq_save(spec, pre_blob.bytes.data(), &pre_blob.size) == SSLM_OK);
	const int64_t ctx_start = BlobContextLength(pre_blob.bytes);

	sslm_speculate_params params{};
	ASSERT_TRUE(MakeSpecParams(model, 4, static_cast<int32_t>(kR), {}, &params));
	std::vector<int32_t> tok(16, 0), rows(16 * static_cast<size_t>(oracle.vocab_size), 0);
	int32_t produced = -1, stop = -1;
	ASSERT_TRUE(sslm_speculate_step_v3(model, spec, &params, ws, tok.data(), 16, rows.data(),
	                                   static_cast<int32_t>(rows.size()), &produced,
	                                   &stop) == SSLM_OK);
	CHECK_MSG(produced == static_cast<int32_t>(kR),
	          "exactly r = %d tokens emit past an exhausted budget", static_cast<int>(kR));
	CHECK(stop == SSLM_SPECULATE_STOP_MAX_TOKENS);
	for (size_t i = 0; i < static_cast<size_t>(produced); ++i) {
		CHECK(tok[i] == fx.want.tokens[i]);  // identical to capped pure greedy
	}

	uint8_t got_tok[32], got_rows[32], want_tok[32], want_rows[32];
	superslm::ComputeTokenDigest(tok.data(), static_cast<size_t>(produced), got_tok);
	superslm::ComputeFinalLogitDigest(rows.data(), static_cast<size_t>(produced),
	                                  static_cast<size_t>(oracle.vocab_size), got_rows);
	DigestRunPrefix(fx.want, kR, static_cast<size_t>(oracle.vocab_size), want_tok, want_rows);
	CHECK(DigestEqual(got_tok, want_tok));
	CHECK(DigestEqual(got_rows, want_rows));  // exactly the r logit rows covered

	// Twin: capped pure greedy through the SAME r emissions via v2 single-token steps.
	ASSERT_TRUE(DriveV2Emissions(model, &twin, oracle.num_hidden_layers, ws, kR));
	SeqBlobBuffer twin_blob(model), spec_blob(model);
	twin_blob.size = twin_blob.bytes.size();
	spec_blob.size = spec_blob.bytes.size();
	ASSERT_TRUE(sslm_seq_save(twin, twin_blob.bytes.data(), &twin_blob.size) == SSLM_OK);
	twin_blob.bytes.resize(twin_blob.size);
	ASSERT_TRUE(sslm_seq_save(spec, spec_blob.bytes.data(), &spec_blob.size) == SSLM_OK);
	spec_blob.bytes.resize(spec_blob.size);
	std::string why;
	CHECK_MSG(CanonicalizedBlobsEqual(twin_blob.bytes, spec_blob.bytes, oracle.hidden_size,
	                                  static_cast<size_t>(sslm_kv_block_size(model)),
	                                  oracle.context_cap, &why),
	          "truncated-commit contract: post-step state equals capped greedy (%s)", why.c_str());
	// R2-W1 pin, numeric form: pending-entry Emissions E=r => E feeds => ctx = start + r,
	// final emission resting unfed-pending as current_token.
	CHECK(BlobContextLength(spec_blob.bytes) == ctx_start + static_cast<int64_t>(kR));
	if (produced > 0) {
		CHECK(BlobReadLE32(spec_blob.bytes.data() + BlobLayout::kCurrentToken) ==
		      static_cast<uint32_t>(tok[produced - 1]));
	}
	int64_t retained = -1;
	CHECK(sslm_seq_committed_token_count(spec, &retained) == SSLM_OK);
	CHECK(retained == ctx_start + static_cast<int64_t>(kR));  // retention holds exactly emitted
}

// V1b -- Reason precedence corner: the r-th token ITSELF is a stop id -> StopTokenMatched,
// not MaxTokensReached (stop test precedes cap test per position, forward_sites.cpp:2615-2628;
// an inverted test order fails here).
void TestV1b_StopPrecedesCapAtTheBoundaryCorner(sslm_model model, sslm_workspace ws,
                                                const CpuOracleModel& oracle,
                                                const FullKFx& fx) {
	ASSERT_TRUE(fx.want.produced >= 2);
	const int32_t stop_id = fx.want.tokens[1];  // the r-th (second) would-be emission
	const std::vector<int32_t> stop_ids{stop_id};
	SinglePool sp;
	ASSERT_TRUE(MakeSinglePool(model, &sp));
	sslm_seq seq = nullptr;
	ASSERT_TRUE(sslm_seq_create(model, &sp.pool, &seq) == SSLM_OK);
	int32_t consumed = 0;
	ASSERT_TRUE(sslm_prefill(model, seq, fx.prompt.data(),
	                         static_cast<int32_t>(fx.prompt.size()), 64, SSLM_SPAN_PROMPT,
	                         nullptr, &consumed) == SSLM_OK);
	sslm_speculate_params params{};
	ASSERT_TRUE(MakeSpecParams(model, 4, /*max_new_tokens=*/2, stop_ids, &params));
	std::vector<int32_t> tok(16, 0), rows(16 * static_cast<size_t>(oracle.vocab_size), 0);
	int32_t produced = -1, stop = -1;
	ASSERT_TRUE(sslm_speculate_step_v3(model, seq, &params, ws, tok.data(), 16, rows.data(),
	                                   static_cast<int32_t>(rows.size()), &produced,
	                                   &stop) == SSLM_OK);
	CHECK_MSG(produced == 2 && stop == SSLM_SPECULATE_STOP_TOKEN_MATCHED,
	          "a matched stop AT the budget edge reports StopTokenMatched (stop precedes cap)");
	CHECK(produced >= 1 && tok[static_cast<size_t>(produced) - 1] == stop_id);
}

// V2 -- Stop-inside-batch (boundary cell (b)) WITH its post-step state leg (residual r-G1):
// stop id placed inside a would-be-accepted batch terminates at p INCLUSIVE, drafts beyond p
// discarded, both digests cover through p only, StopTokenMatched identical to greedy, AND
// post-step state equals stopped greedy: occupancy through the last EMITTED token, retention
// exactly the p+1 emitted ids including the stop id, saturation counting emitted landings
// only, carried walk-state resting unfed-pending (greedy's between-steps value). Fixture
// route per plan: seed committed history via prefill CONTAINING the stop id (prefill performs
// no stop filtering -- decode-only filter, forward_sites.cpp:2531-2532), so the suffix
// continuation proposes it into the batch.
void TestV2_StopInsideBatchCutWithPostStateLeg(sslm_model model, sslm_workspace ws,
                                               const CpuOracleModel& oracle,
                                               const FullKFx& fx) {
	ASSERT_TRUE(fx.want.produced >= 3);
	constexpr size_t kP = 1;  // stop lands at batch position 1 (inside a 4-proposal batch)
	const int32_t stop_id = fx.want.tokens[kP];
	const std::vector<int32_t> stop_ids{stop_id};
	// Seeded prompt: original window plus the first would-be emission, ending just before the
	// stop id -- the drafter's continuation then proposes the stop id INTO the batch.
	std::vector<int32_t> seeded(fx.prompt.begin(), fx.prompt.end());
	seeded.push_back(fx.want.tokens[0]);

	SinglePool sp;
	ASSERT_TRUE(MakePool(model, 2, &sp));
	sslm_seq spec = nullptr, twin = nullptr;
	ASSERT_TRUE(sslm_seq_create(model, &sp.pool, &spec) == SSLM_OK);
	ASSERT_TRUE(sslm_seq_create(model, &sp.pool, &twin) == SSLM_OK);
	int32_t consumed = 0;
	ASSERT_TRUE(sslm_prefill(model, spec, seeded.data(), static_cast<int32_t>(seeded.size()),
	                         64, SSLM_SPAN_PROMPT, nullptr, &consumed) == SSLM_OK);
	ASSERT_TRUE(sslm_prefill(model, twin, seeded.data(), static_cast<int32_t>(seeded.size()),
	                         64, SSLM_SPAN_PROMPT, nullptr, &consumed) == SSLM_OK);

	sslm_speculate_params params{};
	ASSERT_TRUE(MakeSpecParams(model, 4, 8, stop_ids, &params));
	std::vector<int32_t> tok(16, 0), rows(16 * static_cast<size_t>(oracle.vocab_size), 0);
	int32_t produced = -1, stop = -1;
	ASSERT_TRUE(sslm_speculate_step_v3(model, spec, &params, ws, tok.data(), 16, rows.data(),
	                                   static_cast<int32_t>(rows.size()), &produced,
	                                   &stop) == SSLM_OK);
	CHECK_MSG(produced == static_cast<int32_t>(kP) + 1,
	          "termination lands exactly at the stop position p inclusive");
	CHECK(stop == SSLM_SPECULATE_STOP_TOKEN_MATCHED);
	CHECK(tok[kP] == stop_id);
	for (size_t i = 0; i <= kP; ++i) CHECK(tok[i] == fx.want.tokens[i]);  // drafts beyond cut

	uint8_t gt[32], gr[32], wt[32], wr[32];
	superslm::ComputeTokenDigest(tok.data(), static_cast<size_t>(produced), gt);
	superslm::ComputeFinalLogitDigest(rows.data(), static_cast<size_t>(produced),
	                                  static_cast<size_t>(oracle.vocab_size), gr);
	DigestRunPrefix(fx.want, kP + 1, static_cast<size_t>(oracle.vocab_size), wt, wr);
	CHECK(DigestEqual(gt, wt));
	CHECK(DigestEqual(gr, wr));  // coverage through p only

	// Twin: stopped pure greedy via v2 steps (the twin stops after the same emission; the
	// shipped loop's stop semantics are the reference the walk must reproduce).
	ASSERT_TRUE(DriveV2Emissions(model, &twin, oracle.num_hidden_layers, ws, kP + 1));
	SeqBlobBuffer tb(model), sb(model);
	tb.size = tb.bytes.size();
	sb.size = sb.bytes.size();
	ASSERT_TRUE(sslm_seq_save(twin, tb.bytes.data(), &tb.size) == SSLM_OK);
	tb.bytes.resize(tb.size);
	ASSERT_TRUE(sslm_seq_save(spec, sb.bytes.data(), &sb.size) == SSLM_OK);
	sb.bytes.resize(sb.size);
	std::string why;
	CHECK_MSG(CanonicalizedBlobsEqual(tb.bytes, sb.bytes, oracle.hidden_size,
	                                  static_cast<size_t>(sslm_kv_block_size(model)),
	                                  oracle.context_cap, &why),
	          "r-G1 post-step state leg: stop-arm disposal leaves state equal to stopped "
	          "greedy (%s)",
	          why.c_str());
	// Retention holds EXACTLY the p+1 emitted ids including the stop id.
	int64_t retained = -1;
	CHECK(sslm_seq_committed_token_count(spec, &retained) == SSLM_OK);
	CHECK(retained == BlobContextLength(tb.bytes));
}

// V3 -- Zero-budget entry arm (residuals R2-W2 + r-G2): speculate entered with exhausted
// budget and a non-empty would-be draft emits nothing, reports MaxTokensReached, changes no
// counter, and passes no landings -- canonicalized post-state equal to pre-state.
void TestV3_ZeroBudgetEntryArm(sslm_model model, sslm_workspace ws, const CpuOracleModel& oracle,
                               const FullKFx& fx, const std::vector<int32_t>& plain_prompt) {
	const bool have_accepting_window = !fx.prompt.empty();
	const std::vector<int32_t>& prompt =
	    have_accepting_window ? fx.prompt : plain_prompt;

	SinglePool sp;
	ASSERT_TRUE(MakeSinglePool(model, &sp));
	sslm_seq seq = nullptr;
	ASSERT_TRUE(sslm_seq_create(model, &sp.pool, &seq) == SSLM_OK);
	int32_t consumed = 0;
	ASSERT_TRUE(sslm_prefill(model, seq, prompt.data(), static_cast<int32_t>(prompt.size()),
	                         64, SSLM_SPAN_PROMPT, nullptr, &consumed) == SSLM_OK);
	SeqBlobBuffer pre(model);
	pre.size = pre.bytes.size();
	ASSERT_TRUE(sslm_seq_save(seq, pre.bytes.data(), &pre.size) == SSLM_OK);
	pre.bytes.resize(pre.size);

	sslm_speculate_params zero{};
	ASSERT_TRUE(MakeSpecParams(model, /*k_max=*/4, /*max_new_tokens=*/0, {}, &zero));
	std::vector<int32_t> tok(8, -777), rows(8 * static_cast<size_t>(oracle.vocab_size), 0);
	int32_t produced = -1, stop = -1;
	ASSERT_TRUE(sslm_speculate_step_v3(model, seq, &zero, ws, tok.data(), 8, rows.data(),
	                                   static_cast<int32_t>(rows.size()), &produced,
	                                   &stop) == SSLM_OK);
	CHECK_MSG(produced == 0, "exhausted budget emits nothing");
	CHECK(stop == SSLM_SPECULATE_STOP_MAX_TOKENS);
	CHECK(tok[0] == -777);  // no output byte written past the contract

	SeqBlobBuffer post(model);
	post.size = post.bytes.size();
	ASSERT_TRUE(sslm_seq_save(seq, post.bytes.data(), &post.size) == SSLM_OK);
	post.bytes.resize(post.size);
	std::string why;
	CHECK_MSG(CanonicalizedBlobsEqual(pre.bytes, post.bytes, oracle.hidden_size,
	                                  static_cast<size_t>(sslm_kv_block_size(model)),
	                                  oracle.context_cap, &why),
	          "no landings, no counters moved, no retention append (%s)", why.c_str());
}

}  // namespace

int main(int argc, char** argv) {
	ParseFixtureArgs(argc, argv);
	if (g_model_path.empty()) {
		SKIP_MSG("--model=PATH not supplied -- dim08 V1-V3 not run");
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
			SKIP_MSG("workspace create failed -- dim08 cells need a workspace");
		}
		FullKFx fx;
		bool have_fx = false;
		if (!g_corpus_path.empty() && !g_model_tok_path.empty()) {
			sslm_model tok_model = nullptr;
			std::vector<uint8_t> tok_bytes;
			if (ReadFileBytes(g_model_tok_path, &tok_bytes)) {
				CHECK(sslm_model_map(tok_bytes.data(), tok_bytes.size(), &tok_model) == SSLM_OK);
			}
			if (tok_model) {
				std::vector<std::string> utterances;
				std::vector<int32_t> stream;
				if (LoadCorpusUtterances(g_corpus_path, 24, &utterances)) {
					for (const auto& u : utterances) {
						std::vector<int32_t> ids;
						if (TokenizeUtf8(tok_model, u, &ids) && ids.size() >= 8) {
							stream.insert(stream.end(), ids.begin(), ids.end());
						}
					}
				}
				have_fx = FindFullK(oracle, stream, &fx);
				if (!have_fx) {
					SKIP_MSG("full-K accepting window not found on this artifact/corpus -- "
					         "V1/V1b/V2 fall back or skip");
				}
				CHECK(sslm_model_unmap(tok_model) == SSLM_OK);
			}
		}
		if (ws) {
			if (have_fx) {
				TestV1_CapStraddlingBatchTruncationAndFeedAccounting(model, ws, oracle, fx);
				TestV1b_StopPrecedesCapAtTheBoundaryCorner(model, ws, oracle, fx);
				TestV2_StopInsideBatchCutWithPostStateLeg(model, ws, oracle, fx);
			} else {
				SKIP_MSG("V1/V1b/V2 require the full-K accepting-window fixture");
			}
			TestV3_ZeroBudgetEntryArm(model, ws, oracle, fx,
			                          NoRepeatPrompt(oracle.vocab_size, 12));
			CHECK(sslm_workspace_destroy(ws) == SSLM_OK);
		}
		CHECK(sslm_model_unmap(model) == SSLM_OK);
	}
	int ec;
	PrintSummaryAndExit(&ec);
	return ec;
}

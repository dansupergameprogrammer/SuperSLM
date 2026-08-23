// T-2246 (test design) -- Coverage Model row 1, Greedy-equivalence oracle (plan r6 SS6 row 1,
// SS5 S-A flagship), plus the full-K bonus family (audit G-1) and the minimal-margin runner-up
// fixture class (audit M-1). Every cell asserts the mechanism's emitted stream AND both digests
// equal the shipped greedy loop's on identical drives -- the feature oracle for plan SS1's
// identity claim ("identical emitted-token stream and identical digests versus pure greedy").
//
// RED STATUS: every cell below calls sslm_speculate_step_v3 / sslm_speculate_params_init --
// declared in sslm_specdec_red_contract.h, undefined at pin f409bda -- so each file links RED
// BY LINK until S-E lands. The greedy-reference and margin-search legs use only shipped
// symbols and run for real once linked.
#include "fixture_common.h"

using namespace superslm;

using superslm::SslmDecodeStopReason;

namespace {

// E1 -- Flagship greedy-equivalence with digests: drive _v3 to N emissions from a real prompt;
// compare token stream, token digest, and logit-row digest against RunGreedyDecodeLoop driven
// identically. Fallback prompt needs no corpus/tokenizer, so this cell runs wherever a base
// artifact exists.
void TestE1_FlagshipGreedyEquivalenceWithDigests(sslm_model model, sslm_workspace ws,
                                                 const CpuOracleModel& oracle,
                                                 const std::vector<int32_t>& prompt) {
	GreedyRun want;
	std::string err;
	ASSERT_TRUE(RunGreedyReference(oracle, prompt.data(), prompt.size(), {}, 16, &want, &err));
	if (want.produced < 4) return;  // too short to be a meaningful equivalence sample

	SinglePool sp;
	ASSERT_TRUE(MakeSinglePool(model, &sp));
	sslm_seq seq = nullptr;
	ASSERT_TRUE(sslm_seq_create(model, &sp.pool, &seq) == SSLM_OK);
	int32_t consumed = 0;
	ASSERT_TRUE(sslm_prefill(model, seq, prompt.data(), static_cast<int32_t>(prompt.size()),
	                         64, SSLM_SPAN_PROMPT, nullptr, &consumed) == SSLM_OK);

	sslm_speculate_params params{};
	ASSERT_TRUE(MakeSpecParams(model, /*k_max=*/4, /*max_new_tokens=*/16, {}, &params));
	SpecDrive got;
	err.clear();
	const bool drove = DriveSpeculate(model, seq, params, ws, oracle.vocab_size,
	                                  want.produced, &got, &err);
	ASSERT_TRUE(drove);

	CHECK(got.tokens == want.tokens);
	uint8_t want_tok[32], want_rows[32], got_tok[32], got_rows[32];
	DigestRun(want, static_cast<size_t>(oracle.vocab_size), want_tok, want_rows);
	superslm::ComputeTokenDigest(got.tokens.data(), got.tokens.size(), got_tok);
	superslm::ComputeFinalLogitDigest(got.logit_rows.data(), got.tokens.size(),
	                                  static_cast<size_t>(oracle.vocab_size), got_rows);
	CHECK(DigestEqual(got_tok, want_tok));
	CHECK_MSG(DigestEqual(got_rows, want_rows),
	          "logit-row digest must pair every emitted token with its own target-argmax row "
	          "(plan SS1 digest mapping)");
}

// E2 -- Minimal-margin runner-up class (M-1): acceptance must flip exactly at integer
// equality. The harness sweeps corpus prompts through the SHIPPED loop, reads each emitted
// position's argmax-vs-runner-up margin straight off the reference's own logit rows, and
// selects the prompts carrying the smallest positive margins (plus any exact ties found).
// The _v3 leg then asserts stream+digest equality ON THOSE PROMPTS. A +-tau-tolerant
// comparator mutant flips acceptance at any margin <= tau position and diverges from greedy
// there, failing this cell -- which is why exact ties alone cannot carry the claim.
struct MarginSample {
	size_t prompt_index;
	int32_t margin;      // max - runner-up at the tightest position of this prompt (>0)
	bool has_exact_tie;  // some emitted position had a second index equal to the max
};

void TestE2_AcceptanceFlipsExactlyAtEquality(sslm_model model, sslm_workspace ws,
                                             const CpuOracleModel& oracle,
                                             const std::vector<std::vector<int32_t>>& prompts) {
	std::vector<MarginSample> samples;
	for (size_t p = 0; p < prompts.size(); ++p) {
		GreedyRun run;
		std::string err;
		if (!RunGreedyReference(oracle, prompts[p].data(), prompts[p].size(), {}, 12, &run,
		                        &err)) {
			continue;
		}
		MarginSample best{p, INT32_MAX, false};
		for (size_t t = 0; t < run.produced; ++t) {
			const int32_t* row =
			    run.logit_rows.data() + t * static_cast<size_t>(oracle.vocab_size);
			int32_t v1 = INT32_MIN, v2 = INT32_MIN;
			for (size_t j = 0; j < static_cast<size_t>(oracle.vocab_size); ++j) {
				if (row[j] > v1) {
					v2 = v1;
					v1 = row[j];
				} else if (row[j] > v2) {
					v2 = row[j];
				}
			}
			const int32_t margin = v1 - v2;  // 0 exactly when an argmax tie exists
			if (margin >= 0 && margin < best.margin) best.margin = margin;
			best.has_exact_tie = best.has_exact_tie || (margin == 0);
		}
		if (best.margin != INT32_MAX) samples.push_back(best);
	}
	if (samples.empty()) {
		SKIP_MSG("no margin samples collected -- corpus prompts produced no runs");
		return;
	}
	// Tightest positive-margin prompt first; prefer an exact-tie prompt when one exists.
	const MarginSample* chosen = &samples[0];
	for (const auto& s : samples) {
		if (s.margin > 0 && s.margin < chosen->margin) chosen = &s;
		if (s.has_exact_tie) {
			chosen = &s;  // an exact tie dominates: it pins lowest-index discipline end to end
			break;
		}
	}
	CHECK(chosen->margin < INT32_MAX);

	const std::vector<int32_t>& prompt = prompts[chosen->prompt_index];
	GreedyRun want;
	std::string err;
	ASSERT_TRUE(
	    RunGreedyReference(oracle, prompt.data(), prompt.size(), {}, 12, &want, &err));

	SinglePool sp;
	ASSERT_TRUE(MakeSinglePool(model, &sp));
	sslm_seq seq = nullptr;
	ASSERT_TRUE(sslm_seq_create(model, &sp.pool, &seq) == SSLM_OK);
	int32_t consumed = 0;
	ASSERT_TRUE(sslm_prefill(model, seq, prompt.data(), static_cast<int32_t>(prompt.size()),
	                         64, SSLM_SPAN_PROMPT, nullptr, &consumed) == SSLM_OK);
	sslm_speculate_params params{};
	ASSERT_TRUE(MakeSpecParams(model, 4, 12, {}, &params));
	SpecDrive got;
	err.clear();
	ASSERT_TRUE(DriveSpeculate(model, seq, params, ws, oracle.vocab_size, want.produced, &got,
	                           &err));
	CHECK_MSG(got.tokens == want.tokens,
	         "at minimal margin %d the acceptor must flip exactly at equality -- a tolerant "
	         "comparator diverges here",
	         chosen->margin);
	uint8_t wt[32], wr[32], gt[32], gr[32];
	DigestRun(want, static_cast<size_t>(oracle.vocab_size), wt, wr);
	superslm::ComputeTokenDigest(got.tokens.data(), got.tokens.size(), gt);
	superslm::ComputeFinalLogitDigest(got.logit_rows.data(), got.tokens.size(),
	                                  static_cast<size_t>(oracle.vocab_size), gr);
	CHECK(DigestEqual(gt, wt));
	CHECK(DigestEqual(gr, wr));
}

// E3 -- Full-K bonus branch (G-1): history containing an exact K-token repetition so the
// longest-suffix match proposes K, verification accepts all K, and the bonus target emits.
// Fixture selection uses FindFullKAcceptancePrompt's search over the tokenized corpus (the
// plan's own stated construction route); K=1 is the minimal sibling driven on the same
// prompt. Asserts stream + both digests vs greedy, occupancy advancing K+1 per step, and
// retention appending per SS3.0.
void TestE3_FullKBonusBranchAndK1Sibling(sslm_model model, sslm_workspace ws,
                                         const CpuOracleModel& oracle,
                                         const std::vector<int32_t>& corpus_stream) {
	constexpr int32_t kK = 4;
	std::vector<int32_t> prompt;
	std::string err;
	if (!FindFullKAcceptancePrompt(oracle, corpus_stream, kK, /*max_attempts=*/160, &prompt,
	                               &err)) {
		SKIP_MSG("full-K fixture not found on this artifact/corpus: %s", err.c_str());
		return;
	}
	GreedyRun want;
	ASSERT_TRUE(RunGreedyReference(oracle, prompt.data(), prompt.size(), {},
	                               static_cast<size_t>(kK) + 1, &want, &err));
	CHECK_MSG(want.produced == static_cast<size_t>(kK) + 1,
	          "fixture premise: greedy emits all K accepted drafts then the bonus");

	SinglePool sp;
	ASSERT_TRUE(MakeSinglePool(model, &sp));
	sslm_seq seq = nullptr;
	ASSERT_TRUE(sslm_seq_create(model, &sp.pool, &seq) == SSLM_OK);
	int32_t consumed = 0;
	ASSERT_TRUE(sslm_prefill(model, seq, prompt.data(), static_cast<int32_t>(prompt.size()),
	                         64, SSLM_SPAN_PROMPT, nullptr, &consumed) == SSLM_OK);

	SeqBlobBuffer blob_before(model);
	size_t sz_before = blob_before.size;
	ASSERT_TRUE(sslm_seq_save(seq, blob_before.bytes.data(), &sz_before) == SSLM_OK);
	blob_before.bytes.resize(sz_before);
	int64_t committed_before = -1;
	CHECK(sslm_seq_committed_token_count(seq, &committed_before) == SSLM_OK);

	sslm_speculate_params params{};
	ASSERT_TRUE(MakeSpecParams(model, kK, kK + 1, {}, &params));
	SpecDrive got;
	err.clear();
	ASSERT_TRUE(DriveSpeculate(model, seq, params, ws, oracle.vocab_size, want.produced, &got,
	                           &err));
	CHECK(got.tokens == want.tokens);
	uint8_t wt[32], wr[32], gt[32], gr[32];
	DigestRun(want, static_cast<size_t>(oracle.vocab_size), wt, wr);
	superslm::ComputeTokenDigest(got.tokens.data(), got.tokens.size(), gt);
	superslm::ComputeFinalLogitDigest(got.logit_rows.data(), got.tokens.size(),
	                                  static_cast<size_t>(oracle.vocab_size), gr);
	CHECK(DigestEqual(gt, wt));
	CHECK(DigestEqual(gr, wr));

	SeqBlobBuffer blob_after(model);
	size_t sz_after = blob_after.size;
	ASSERT_TRUE(sslm_seq_save(seq, blob_after.bytes.data(), &sz_after) == SSLM_OK);
	blob_after.bytes.resize(sz_after);
	CHECK(BlobContextLength(blob_after.bytes) - BlobContextLength(blob_before.bytes) ==
	      static_cast<int64_t>(want.produced));  // occupancy advances K+1 (plan SS3.2)
	int64_t committed_after = -1;
	CHECK(sslm_seq_committed_token_count(seq, &committed_after) == SSLM_OK);
	CHECK(committed_after - committed_before == static_cast<int64_t>(want.produced));

	// K=1 minimal sibling: same fixture, one draft proposed, accepted, bonus emitted.
	SinglePool sp1;
	ASSERT_TRUE(MakeSinglePool(model, &sp1));
	sslm_seq seq1 = nullptr;
	ASSERT_TRUE(sslm_seq_create(model, &sp1.pool, &seq1) == SSLM_OK);
	ASSERT_TRUE(sslm_prefill(model, seq1, prompt.data(), static_cast<int32_t>(prompt.size()),
	                         64, SSLM_SPAN_PROMPT, nullptr, &consumed) == SSLM_OK);
	sslm_speculate_params params1{};
	ASSERT_TRUE(MakeSpecParams(model, /*k_max=*/1, /*max_new_tokens=*/kK + 1, {}, &params1));
	SpecDrive got1;
	err.clear();
	ASSERT_TRUE(DriveSpeculate(model, seq1, params1, ws, oracle.vocab_size, want.produced,
	                           &got1, &err));
	CHECK(got1.tokens == want.tokens);
}

// E4 -- Ceiling boundary: identity across a longer drive whose occupancy stays under the
// cache cap (both sides from the same real prompt history), then the ceiling GUARD arm --
// an occupied window parked just under context_cap via the cheap tampered-context restore
// route, with K sized so the K+1 verify chunk crosses the cap. The whole-chunk guard must
// reject UP FRONT, before any landing (forward_sites.cpp:2133-2136): kv_saturation_count and
// context_length are unchanged by the rejected call.
void TestE4_CeilingBoundaryEquivalenceAndGuard(sslm_model model, sslm_workspace ws,
                                               const CpuOracleModel& oracle,
                                               const std::vector<int32_t>& prompt) {
	const size_t budget = 16;
	GreedyRun want;
	std::string err;
	ASSERT_TRUE(RunGreedyReference(oracle, prompt.data(), prompt.size(), {}, budget, &want, &err));

	SinglePool sp;
	ASSERT_TRUE(MakeSinglePool(model, &sp));
	sslm_seq seq = nullptr;
	ASSERT_TRUE(sslm_seq_create(model, &sp.pool, &seq) == SSLM_OK);
	int32_t consumed = 0;
	ASSERT_TRUE(sslm_prefill(model, seq, prompt.data(), static_cast<int32_t>(prompt.size()),
	                         64, SSLM_SPAN_PROMPT, nullptr, &consumed) == SSLM_OK);

	sslm_speculate_params params{};
	ASSERT_TRUE(MakeSpecParams(model, 4, static_cast<int32_t>(budget), {}, &params));
	SpecDrive got;
	err.clear();
	ASSERT_TRUE(DriveSpeculate(model, seq, params, ws, oracle.vocab_size, want.produced, &got,
	                           &err));
	CHECK(got.tokens == want.tokens);
	uint8_t wt[32], wr[32], gt[32], gr[32];
	DigestRun(want, static_cast<size_t>(oracle.vocab_size), wt, wr);
	superslm::ComputeTokenDigest(got.tokens.data(), got.tokens.size(), gt);
	superslm::ComputeFinalLogitDigest(got.logit_rows.data(), got.tokens.size(),
	                                  static_cast<size_t>(oracle.vocab_size), gr);
	CHECK(DigestEqual(gt, wt));
	CHECK(DigestEqual(gr, wr));

	// Ceiling guard arm: occupancy parked at cap-4 via the tampered-restore route (real
	// sequence saved first so every other blob field stays valid), then a K=8 speculate whose
	// K+1 verify chunk cannot fit the remaining capacity. Expected: up-front rejection before
	// any landing -- context_length and kv_saturation_count unchanged by the rejected call.
	{
		SinglePool sp2;
		ASSERT_TRUE(MakePool(model, 3, &sp2));
		sslm_seq seed = nullptr;
		ASSERT_TRUE(sslm_seq_create(model, &sp2.pool, &seed) == SSLM_OK);
		ASSERT_TRUE(sslm_prefill(model, seed, prompt.data(),
		                         static_cast<int32_t>(prompt.size()), 64, SSLM_SPAN_PROMPT,
		                         nullptr, &consumed) == SSLM_OK);
		SeqBlobBuffer seed_blob(model);
		size_t seed_sz = seed_blob.size;
		ASSERT_TRUE(sslm_seq_save(seed, seed_blob.bytes.data(), &seed_sz) == SSLM_OK);
		seed_blob.bytes.resize(seed_sz);
		CHECK(sslm_seq_release(seed) == SSLM_OK);

		sslm_seq near = nullptr;
		ASSERT_TRUE(RestoreWithTamperedContextLength(model, &sp2.pool, seed_blob.bytes,
		                                             oracle.context_cap - 4, &near));
		ASSERT_TRUE(near != nullptr);
		SeqBlobBuffer pre(model);
		size_t pre_sz = pre.size;
		ASSERT_TRUE(sslm_seq_save(near, pre.bytes.data(), &pre_sz) == SSLM_OK);
		pre.bytes.resize(pre_sz);

		sslm_speculate_params over{};
		ASSERT_TRUE(MakeSpecParams(model, /*k_max=*/8, /*max_new_tokens=*/8, {}, &over));
		std::vector<int32_t> tok(16, 0);
		std::vector<int32_t> rows(16 * static_cast<size_t>(oracle.vocab_size), 0);
		int32_t produced = -1, stop = -1;
		const sslm_status st = sslm_speculate_step_v3(model, near, &over, ws, tok.data(), 16,
		                                              rows.data(),
		                                              static_cast<int32_t>(rows.size()),
		                                              &produced, &stop);
		CHECK_MSG(st != SSLM_OK,
		          "a drafted block crossing the remaining capacity must reject (guard 1)");
		if (st != SSLM_OK) {
			SeqBlobBuffer post(model);
			size_t post_sz = post.size;
			ASSERT_TRUE(sslm_seq_save(near, post.bytes.data(), &post_sz) == SSLM_OK);
			post.bytes.resize(post_sz);
			CHECK(BlobContextLength(post.bytes) == BlobContextLength(pre.bytes));
			CHECK(BlobSaturationCount(post.bytes) == BlobSaturationCount(pre.bytes));
		}
		CHECK(sslm_seq_release(near) == SSLM_OK);
	}
}

}  // namespace

int main(int argc, char** argv) {
	ParseFixtureArgs(argc, argv);
	if (g_model_path.empty()) {
		SKIP_MSG("--model=PATH not supplied -- dim01 E1-E4 not run");
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
		// One workspace sized like the reference driver's (per-call scratch convention).
		const size_t kv_bytes = static_cast<size_t>(oracle.num_hidden_layers) *
		                        static_cast<size_t>(oracle.context_cap) * oracle.num_kv_heads *
		                        oracle.head_dim * 2;
		std::vector<uint8_t> ws_buf(kv_bytes);
		sslm_config cfg{};
		cfg.max_batch = 16;
		cfg.max_chunk_budget = 256;
		cfg.max_layer_budget = static_cast<int32_t>(oracle.num_hidden_layers);
		const size_t ws_size = sslm_workspace_size(model, &cfg);
		std::vector<uint8_t> ws_store(ws_size ? ws_size : 1);
		sslm_workspace ws = nullptr;
		const bool have_ws =
		    sslm_workspace_create(model, &cfg, ws_store.data(), ws_store.size(), &ws) == SSLM_OK;
		if (!have_ws) SKIP_MSG("workspace create failed -- dim01 cells need a workspace");

		// Prompts: corpus-backed when available, arithmetic no-repeat fallback otherwise.
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
		if (!have_corpus) {
			SKIP_MSG("--corpus=PATH/--modeltok=PATH not supplied -- E2/E3 fall back or skip");
			prompts.push_back(NoRepeatPrompt(oracle.vocab_size, 12));
		}

		if (ws) {
			TestE1_FlagshipGreedyEquivalenceWithDigests(model, ws, oracle, prompts.front());
			TestE2_AcceptanceFlipsExactlyAtEquality(model, ws, oracle, prompts);
			if (have_corpus) {
				TestE3_FullKBonusBranchAndK1Sibling(model, ws, oracle, corpus_stream);
			} else {
				SKIP_MSG("E3 full-K fixture requires a tokenized corpus");
			}
			TestE4_CeilingBoundaryEquivalenceAndGuard(model, ws, oracle, prompts.front());
		}
		if (ws) CHECK(sslm_workspace_destroy(ws) == SSLM_OK);
		CHECK(sslm_model_unmap(model) == SSLM_OK);
	}
	int ec;
	PrintSummaryAndExit(&ec);
	return ec;
}

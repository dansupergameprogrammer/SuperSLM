// T-2246 (test design) -- Coverage Model row 7, Failure paths (plan r6 SS3.3 staging rule,
// SS3.5 guard 1, SS5 S-A injected-failure + empty-draft cells, audit G-4): the verify
// primitive's failure containment. The empty-draft fallback cell doubles as the equivalence
// witness for the broken-drafter negative control (dim12 M2 drives the same fixture).
//
// RED STATUS: link-red on sslm_speculate_step_v3 / superslm_test::g_inject_specdec_fault
// (expected seam, planner routing CM-G3) until S-B/S-E.
#include "fixture_common.h"

using namespace superslm;

namespace {

// F1 -- Empty-draft fallback: a history repeating no token proposes nothing, so EVERY step
// must fall back to the single-token path -- stream and both digests still equal greedy.
void TestF1_EmptyDraftFallsBackToSingleToken(sslm_model model, sslm_workspace ws,
                                             const CpuOracleModel& oracle) {
	const std::vector<int32_t> prompt = NoRepeatPrompt(oracle.vocab_size, 12);
	GreedyRun want;
	std::string err;
	ASSERT_TRUE(
	    RunGreedyReference(oracle, prompt.data(), prompt.size(), {}, 10, &want, &err));

	SinglePool sp;
	ASSERT_TRUE(MakeSinglePool(model, &sp));
	sslm_seq seq = nullptr;
	ASSERT_TRUE(sslm_seq_create(model, &sp.pool, &seq) == SSLM_OK);
	int32_t consumed = 0;
	ASSERT_TRUE(sslm_prefill(model, seq, prompt.data(), static_cast<int32_t>(prompt.size()),
	                         64, SSLM_SPAN_PROMPT, nullptr, &consumed) == SSLM_OK);

	// Direct drafter pin on this fixture's premise: no repetition => zero proposals.
	std::vector<int32_t> draft(4, 0);
	const size_t nd = superslm::SpecdecDraftPropose(prompt, 4, &draft);
	CHECK(nd == 0);

	sslm_speculate_params params{};
	ASSERT_TRUE(MakeSpecParams(model, 4, 10, {}, &params));
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
}

// F2 -- A rejected speculate call leaves the sequence resumable: after hostile-params
// rejections, a VALID call produces exactly the reference continuation.
void TestF2_RejectedCallLeavesSequenceResumable(sslm_model model, sslm_workspace ws,
                                                const CpuOracleModel& oracle,
                                                const std::vector<int32_t>& prompt) {
	GreedyRun want;
	std::string err;
	ASSERT_TRUE(RunGreedyReference(oracle, prompt.data(), prompt.size(), {}, 8, &want, &err));

	SinglePool sp;
	ASSERT_TRUE(MakeSinglePool(model, &sp));
	sslm_seq seq = nullptr;
	ASSERT_TRUE(sslm_seq_create(model, &sp.pool, &seq) == SSLM_OK);
	int32_t consumed = 0;
	ASSERT_TRUE(sslm_prefill(model, seq, prompt.data(), static_cast<int32_t>(prompt.size()),
	                         64, SSLM_SPAN_PROMPT, nullptr, &consumed) == SSLM_OK);

	sslm_speculate_params bad{};
	ASSERT_TRUE(MakeSpecParams(model, 4, 8, {}, &bad));
	bad.struct_size = sizeof(bad) - 4u;  // malformed: rejected before anything else
	std::vector<int32_t> tok(16, -1), rows(16 * static_cast<size_t>(oracle.vocab_size), 0);
	int32_t produced = -1, stop = -1;
	CHECK(sslm_speculate_step_v3(model, seq, &bad, ws, tok.data(), 16, rows.data(), 16,
	                             &produced, &stop) == SSLM_INVALID_ARGUMENT);

	sslm_speculate_params good{};
	ASSERT_TRUE(MakeSpecParams(model, 4, 8, {}, &good));
	SpecDrive got;
	err.clear();
	ASSERT_TRUE(DriveSpeculate(model, seq, good, ws, oracle.vocab_size, want.produced, &got,
	                           &err));
	CHECK(got.tokens == want.tokens);
}

// F3 -- Injected mid-verify non-Ok is contained per the staging rule: post-call sequence
// state equals pre-call over the canonicalized full struct, the sequence is resumable, and
// continued output equals greedy. The injection seam is an EXPECTED symbol (CM-G3): the plan
// names the cell but no seam owner; shape follows tests/support/bad_alloc_injection.h.
void TestF3_InjectedMidVerifyNonOkContained(sslm_model model, sslm_workspace ws,
                                            const CpuOracleModel& oracle,
                                            const std::vector<int32_t>& corpus_stream) {
	constexpr int32_t kK = 4;
	std::vector<int32_t> prompt;
	std::string err;
	if (!FindFullKAcceptancePrompt(oracle, corpus_stream, kK, 160, &prompt, &err)) {
		SKIP_MSG("full-K fixture not found (accepting window needed to stage mid-verify): %s",
		         err.c_str());
		return;
	}
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

	superslm_test::g_inject_specdec_fault =
	    superslm_test::SpecDecFaultKind::kNonOkMidVerify;  // arms ONE fault
	sslm_speculate_params params{};
	ASSERT_TRUE(MakeSpecParams(model, kK, kK + 1, {}, &params));
	std::vector<int32_t> tok(16, 0), rows(16 * static_cast<size_t>(oracle.vocab_size), 0);
	int32_t produced = -1, stop = -1;
	const sslm_status st =
	    sslm_speculate_step_v3(model, seq, &params, ws, tok.data(), 16, rows.data(),
	                           static_cast<int32_t>(rows.size()), &produced, &stop);
	CHECK_MSG(st != SSLM_OK, "the armed mid-verify fault must surface as a non-Ok return");
	CHECK(superslm_test::g_inject_specdec_fault ==
	      superslm_test::SpecDecFaultKind::kNone);  // single-shot disarm

	SeqBlobBuffer post(model);
	post.size = post.bytes.size();
	ASSERT_TRUE(sslm_seq_save(seq, post.bytes.data(), &post.size) == SSLM_OK);
	post.bytes.resize(post.size);
	std::string why;
	CHECK_MSG(CanonicalizedBlobsEqual(pre.bytes, post.bytes, oracle.hidden_size,
	                                  static_cast<size_t>(sslm_kv_block_size(model)),
	                                  oracle.context_cap, &why),
	          "a failed verify call must leave every sequence-state byte untouched (%s)",
	          why.c_str());

	// Resumable: continued output from the untouched state equals greedy.
	GreedyRun want;
	ASSERT_TRUE(RunGreedyReference(oracle, prompt.data(), prompt.size(), {},
	                               static_cast<size_t>(kK) + 2, &want, &err));
	sslm_speculate_params retry{};
	ASSERT_TRUE(MakeSpecParams(model, kK, kK + 2, {}, &retry));
	SpecDrive got;
	err.clear();
	ASSERT_TRUE(DriveSpeculate(model, seq, retry, ws, oracle.vocab_size, want.produced, &got,
	                           &err));
	CHECK(got.tokens == want.tokens);
}

// F4 -- Within-window exhaustion is the UP-FRONT whole-chunk capacity guard: occupancy near
// cap, K+1 chunk larger than the room left -> rejection BEFORE any landing (saturation
// unchanged), sequence unharmed and resumable.
void TestF4_WithinWindowCapacityGuardBeforeAnyLanding(sslm_model model, sslm_workspace ws,
                                                      const CpuOracleModel& oracle,
                                                      const std::vector<int32_t>& prompt) {
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

	sslm_seq near = nullptr;
	ASSERT_TRUE(RestoreWithTamperedContextLength(model, &sp.pool, seed_blob.bytes,
	                                             oracle.context_cap - 2, &near));
	SeqBlobBuffer pre(model);
	pre.size = pre.bytes.size();
	ASSERT_TRUE(sslm_seq_save(near, pre.bytes.data(), &pre.size) == SSLM_OK);
	pre.bytes.resize(pre.size);

	sslm_speculate_params over{};
	ASSERT_TRUE(MakeSpecParams(model, /*k_max=*/8, 8, {}, &over));  // K+1 = 9 > 2 remaining
	std::vector<int32_t> tok(16, 0), rows(16 * static_cast<size_t>(oracle.vocab_size), 0);
	int32_t produced = -1, stop = -1;
	const sslm_status st =
	    sslm_speculate_step_v3(model, near, &over, ws, tok.data(), 16, rows.data(),
	                           static_cast<int32_t>(rows.size()), &produced, &stop);
	CHECK(st != SSLM_OK);
	if (st != SSLM_OK) {
		SeqBlobBuffer post(model);
		post.size = post.bytes.size();
		ASSERT_TRUE(sslm_seq_save(near, post.bytes.data(), &post.size) == SSLM_OK);
		post.bytes.resize(post.size);
		CHECK(BlobSaturationCount(post.bytes) == BlobSaturationCount(pre.bytes));
		CHECK(BlobContextLength(post.bytes) == BlobContextLength(pre.bytes));
	}
}

}  // namespace

int main(int argc, char** argv) {
	ParseFixtureArgs(argc, argv);
	if (g_model_path.empty()) {
		SKIP_MSG("--model=PATH not supplied -- dim07 F1-F4 not run");
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
			SKIP_MSG("workspace create failed -- dim07 cells need a workspace");
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

		const std::vector<int32_t> prompt = NoRepeatPrompt(oracle.vocab_size, 12);
		if (ws) {
			TestF1_EmptyDraftFallsBackToSingleToken(model, ws, oracle);
			TestF2_RejectedCallLeavesSequenceResumable(model, ws, oracle, prompt);
			TestF4_WithinWindowCapacityGuardBeforeAnyLanding(model, ws, oracle, prompt);
			if (have_corpus) {
				TestF3_InjectedMidVerifyNonOkContained(model, ws, oracle, corpus_stream);
			} else {
				SKIP_MSG("--corpus=PATH/--modeltok=PATH not supplied -- F3 needs a real "
				         "accepting window");
			}
			CHECK(sslm_workspace_destroy(ws) == SSLM_OK);
		}
		CHECK(sslm_model_unmap(model) == SSLM_OK);
	}
	int ec;
	PrintSummaryAndExit(&ec);
	return ec;
}

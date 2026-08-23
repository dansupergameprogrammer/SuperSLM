// T-2246 (test design) -- Coverage Model row 4, Drafter determinism (plan r6 SS3.1, SS6 row 4,
// audit SS5 item 3): the drafter is a pure function of committed_tokens proposing up to K ids
// by longest-suffix match over the retained history, extending the matched suffix greedily by
// its continuation in history, with the EARLIEST occurrence winning equal-length ties
// (C16-style lowest-index discipline). Output oracles are blind to drafter defects by
// construction (the plan states this); these direct pure-function cells are what the
// dimension requires. Every expected proposal below is hand-decoded from the rule against a
// constructed history -- never by recoding the drafter inside the test.
//
// EXECUTION STATUS (fold round 1, 2026-08-22): S-D has LANDED -- superslm::SpecdecDraftPropose
// exists in production under the recorded name, so this file no longer links red; every cell
// executes against the real drafter.
#include "fixture_common.h"

using namespace superslm;

namespace {

std::vector<int32_t> Vec(std::initializer_list<int32_t> ids) { return std::vector<int32_t>(ids); }

// D1 -- Determinism: identical input yields an identical proposal on every call.
void TestD1_SameInputSameProposal() {
	const auto history = Vec({7, 8, 9, 4, 7, 8, 9});  // tail [7,8,9] repeats at index 0
	std::vector<int32_t> a, b;
	const size_t na = superslm::SpecdecDraftPropose(history, 3, &a);
	const size_t nb = superslm::SpecdecDraftPropose(history, 3, &b);
	CHECK(na == nb);
	CHECK(a == b);
}

// D2 -- Earliest-occurrence tie rule: the tail [1,2,3] occurs twice (index 0 with
// continuation 40; index 4 at the very end with NO continuation). Equal lengths => the
// EARLIEST occurrence wins, so the proposal extends 40's branch ([40, 1, 2, ...]). An
// implementation resolving ties toward the LATER occurrence finds no continuation there and
// proposes nothing -- caught by the first check alone.
void TestD2_EarliestOccurrenceTieRule() {
	const auto history = Vec({1, 2, 3, 40, 1, 2, 3});
	std::vector<int32_t> draft;
	const size_t n = superslm::SpecdecDraftPropose(history, /*k_max=*/3, &draft);
	ASSERT_TRUE(n >= 1);
	CHECK_MSG(draft[0] == 40,
	          "equal-length matches must take the EARLIEST occurrence's continuation "
	          "(earlier branch continues 40; the later branch has none)");
	if (n >= 2) CHECK_MSG(draft[1] == 1,
	                      "greedy extension follows the earliest occurrence's own history "
	                      "(after 40 comes 1)");
}

// D3 -- Longest-suffix priority: the tail [5,6,7,8,9] occurs at index 0 (length 5, earliest)
// whose continuation is 0. Any shorter-match or most-recent-match resolution proposes from a
// different point or nothing at all.
void TestD3_LongestSuffixPriority() {
	const auto history = Vec({5, 6, 7, 8, 9, 0, 5, 6, 7, 8, 9});
	std::vector<int32_t> draft;
	const size_t n = superslm::SpecdecDraftPropose(history, 3, &draft);
	ASSERT_TRUE(n >= 1);
	CHECK_MSG(draft[0] == 0,
	          "the longest match's own continuation must win (earliest length-5 match "
	          "continues with 0)");
}

// D4 -- Pure function of committed_tokens: interleaved unrelated traffic on another handle
// (a real v2 decode step) must not perturb this handle's proposals.
void TestD4_PureFunctionNoHiddenState(sslm_model model, sslm_workspace ws,
                                      const CpuOracleModel& oracle,
                                      const std::vector<int32_t>& prompt) {
	const auto history = Vec({3, 1, 4, 1, 5, 9, 2, 6});
	std::vector<int32_t> before, after;
	superslm::SpecdecDraftPropose(history, 3, &before);

	SinglePool sp;
	ASSERT_TRUE(MakeSinglePool(model, &sp));
	sslm_seq seq = nullptr;
	ASSERT_TRUE(sslm_seq_create(model, &sp.pool, &seq) == SSLM_OK);
	int32_t consumed = 0;
	ASSERT_TRUE(sslm_prefill(model, seq, prompt.data(), static_cast<int32_t>(prompt.size()),
	                         64, SSLM_SPAN_PROMPT, nullptr, &consumed) == SSLM_OK);
	sslm_decode_params dp{};
	dp.struct_size = sizeof(dp);
	dp.layer_budget = static_cast<int32_t>(oracle.num_hidden_layers);
	int32_t t0 = -1;
	CHECK(sslm_decode_step_v2(model, &seq, 1, &dp, ws, &t0) == SSLM_OK);

	superslm::SpecdecDraftPropose(history, 3, &after);
	CHECK_MSG(after == before, "unrelated decode traffic must not perturb proposals");
	CHECK(sslm_seq_release(seq) == SSLM_OK);
}

// D5 -- No match proposes nothing: a history repeating no token yields the empty draft --
// the fallback path's input condition and the negative control's fixture premise.
void TestD5_NoMatchProposesNothing() {
	const auto history = Vec({11, 108, 205, 302, 399});
	std::vector<int32_t> draft;
	const size_t n = superslm::SpecdecDraftPropose(history, 3, &draft);
	CHECK_MSG(n == 0 && draft.empty(), "a never-matching history proposes zero drafts");
}

}  // namespace

int main(int argc, char** argv) {
	ParseFixtureArgs(argc, argv);
	TestD1_SameInputSameProposal();
	TestD2_EarliestOccurrenceTieRule();
	TestD3_LongestSuffixPriority();
	TestD5_NoMatchProposesNothing();
	if (g_model_path.empty()) {
		SKIP_MSG("--model=PATH not supplied -- dim04 D4 not run");
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
			TestD4_PureFunctionNoHiddenState(model, ws, oracle,
			                                 NoRepeatPrompt(oracle.vocab_size, 12));
			CHECK(sslm_workspace_destroy(ws) == SSLM_OK);
		} else {
			SKIP_MSG("workspace create failed -- dim04 D4 not run");
		}
		CHECK(sslm_model_unmap(model) == SSLM_OK);
	}
	int ec;
	PrintSummaryAndExit(&ec);
	return ec;
}

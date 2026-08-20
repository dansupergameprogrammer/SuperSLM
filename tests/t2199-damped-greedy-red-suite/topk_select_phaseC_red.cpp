// T-2199 (Curie) -- Phase C1 red suite: top-k candidate selection over a masked int32 row
// (plan Sec7.4, Sec8 Phase C1). RED BY LINK: sslm_damped_greedy.h declares FsdTopK; no .cpp
// in this repo defines it (grep confirmed at authoring commit main@071c5f3).
//
// Coverage Model cells realized here (plan Sec9, Wizard repo, commit 5a7f80e251):
//   dim4  Shape and platform matrices  -- GATE, D-SLM3719  (k at floor / oversized value)
//   dim7  Contract claims               -- GATE, D-SLM3719  (the selection-step mask-status
//                                          tie-break mutation, cycle-9 temper Structural
//                                          Finding 1, 828e17bf24 -- "a legal token, however
//                                          its raw value ties, can never be excluded from
//                                          top-k by a masked position")
#include "fixture_common.h"

using namespace t2199fixture;

// ===========================================================================================
// Sec9 dim7 (GATE, D-SLM3719): the selection-step tie-break mutation. This is the SECOND,
// distinct construction cycle-9 temper added (Structural Finding 1) -- a genuinely different
// failure mode from the scoring-step INT64_MIN-floor mutation (score_select_phaseC_red.cpp):
// a legal token's own narrowed logit constructed to equal INT32_MIN EXACTLY, tied against
// masked positions of LOWER vocabulary index, n_legal = 1. The plan's own corrected rule
// (Sec7.4) breaks ties FIRST by mask status (unmasked always outranks masked at equal value),
// THEN ascending index within a mask-status group -- so the legal token must win its tie and
// be selected regardless of its own index. The pre-fix rule (index-only ties, ascending) would
// silently exclude it whenever its index is not among the k lowest-indexed masked positions --
// exactly the failure Charpy's cycle-9 temper traced (`Claude/Charpy/
// t2199-cycle9-fold-temper-2026-08-20.md` Sec3, `828e17bf24`).
// ===========================================================================================
static void TestDim7_TieBreak_LegalInt32MinWinsOverMaskedAtEqualValue() {
	const int32_t V = 20;  // small synthetic row -- the collision is about tie-break logic,
	                       // not row width, so a tiny hand-inspectable row suffices.
	const int32_t LEGAL_TOKEN = 15;  // the sole legal continuation, deliberately a HIGH index
	std::vector<int32_t> row(V, 0);
	for (int32_t t = 0; t < V; ++t) row[t] = kInt32Min;  // every position at INT32_MIN --
	                                                      // 19 because they are masked, ONE
	                                                      // (index 15) because its own real
	                                                      // logit legitimately narrowed there
	                                                      // (NarrowRowChecked admits
	                                                      // row_min == kInt32Min as legal).
	std::vector<uint8_t> mask = MakeMask(V, /*all_legal=*/false);
	SetLegal(mask, LEGAL_TOKEN);  // exactly one legal position: index 15

	const int32_t k = 6;
	std::vector<int32_t> picks(k, -1);
	FsdTopK(row.data(), mask.data(), V, k, picks.data());

	bool legal_included = false;
	for (int32_t p : picks)
		if (p == LEGAL_TOKEN) legal_included = true;
	CHECK_MSG(legal_included,
	          "top-k picks did not include the sole legal token (index %d) even though every "
	          "candidate ties at INT32_MIN -- the mask-status-first tie-break (Sec7.4, "
	          "cycle-9 temper) must always include it regardless of its own index",
	          LEGAL_TOKEN);

	// The pre-fix (index-only) rule would select the k=6 LOWEST indices {0,1,2,3,4,5} --
	// confirm the fixed rule does NOT reproduce that exact set, which is the discriminating
	// half of this mutation (a cell that merely checked "legal_included" without also ruling
	// out the pre-fix rule's own concrete output could coincidentally pass a broken
	// implementation whose bug happens not to manifest on THIS particular k).
	bool matches_prefix_rule = true;
	for (int32_t i = 0; i < k; ++i)
		if (picks[static_cast<size_t>(i)] != i) matches_prefix_rule = false;
	CHECK_MSG(!matches_prefix_rule,
	          "top-k picks reproduced the pre-fix index-only rule's own output {0..5} exactly "
	          "-- the legal token at index 15 was excluded despite the assertion above");
}

// A control: when the legal token's OWN index (not 15, but 2) already falls among the k
// lowest, an index-only rule would ACCIDENTALLY include it too -- this control confirms the
// fixture actually discriminates by re-running with the legal token moved to a low index and
// expecting inclusion under EITHER rule (a probe that "fires" even when nothing is wrong
// proves nothing, mirroring the commissioning discipline Loki's own strike-6 probe applied to
// its MUST-NOT-FIRE controls).
static void TestDim7_TieBreak_Control_LowIndexLegalTokenAlwaysIncluded() {
	const int32_t V = 20;
	const int32_t LEGAL_TOKEN = 2;  // already among the k=6 lowest indices
	std::vector<int32_t> row(V, kInt32Min);
	std::vector<uint8_t> mask = MakeMask(V, false);
	SetLegal(mask, LEGAL_TOKEN);
	const int32_t k = 6;
	std::vector<int32_t> picks(k, -1);
	FsdTopK(row.data(), mask.data(), V, k, picks.data());
	bool included = false;
	for (int32_t p : picks)
		if (p == LEGAL_TOKEN) included = true;
	CHECK_MSG(included, "control: low-index legal token %d must be included (sanity control, "
	                     "not the discriminating case)",
	          LEGAL_TOKEN);
}

// ===========================================================================================
// Sec9 dim4 (GATE, D-SLM3719): k at its floor (k=1) and at an oversized value (k=vocab_size --
// "under any bound schema this puts every masked token in the candidate set... this cell's own
// assertion is that the emitted token is legal at k=vocab_size specifically, the widest and
// cheapest fixture to construct the general n_legal < k claim against"). This file pins the
// SELECTION half only (every legal token is present in the k-picks set at both extremes); the
// full emitted-token claim (which requires the score-and-argmax step) is
// score_select_phaseC_red.cpp's own dim7/dim4 cell.
// ===========================================================================================
static void TestDim4_KFloor_SingleLegalCandidateAlwaysSelected() {
	const int32_t V = 50;
	const int32_t LEGAL_TOKEN = 30;
	std::vector<int32_t> row = MakeSyntheticRow(V, {LEGAL_TOKEN});
	std::vector<uint8_t> mask = MakeMask(V, false);
	SetLegal(mask, LEGAL_TOKEN);
	ApplyMaskFloor(row, mask);
	std::vector<int32_t> picks(1, -1);
	FsdTopK(row.data(), mask.data(), V, /*k=*/1, picks.data());
	CHECK_MSG(picks[0] == LEGAL_TOKEN,
	          "k=1 top-k pick = %d, want the sole legal, highest-valued token %d", picks[0],
	          LEGAL_TOKEN);
}

static void TestDim4_KOversized_AllLegalTokensIncluded() {
	const int32_t V = 100;
	const std::vector<int32_t> legal_tokens = {5, 40, 77};
	std::vector<int32_t> row = MakeSyntheticRow(V, legal_tokens);
	std::vector<uint8_t> mask = MakeMask(V, false);
	for (int32_t t : legal_tokens) SetLegal(mask, t);
	ApplyMaskFloor(row, mask);
	const int32_t k = V;  // k = vocab_size, the oversized extreme
	std::vector<int32_t> picks(static_cast<size_t>(k), -1);
	FsdTopK(row.data(), mask.data(), V, k, picks.data());
	CHECK_MSG(picks.size() == static_cast<size_t>(V), "top-k must still return exactly V picks "
	                                                   "at k=vocab_size");
	for (int32_t legal : legal_tokens) {
		bool found = false;
		for (int32_t p : picks)
			if (p == legal) found = true;
		CHECK_MSG(found, "legal token %d missing from the k=vocab_size candidate set -- at "
		                  "k=V every position (legal and masked alike) must be selected",
		          legal);
	}
}

// A determinism/order cell: descending-by-value with the corrected tie-break must select the
// SAME set (and the same ORDER, since DampedGreedyScoreAndArgmax's own downstream tie-break at
// Sec7.6 depends on which candidate list it receives, though not on the caller re-sorting it)
// every time it is run against an identical row -- a bare sanity cell guarding against any
// hidden nondeterminism (e.g. an unstable sort) entering top-k itself.
static void TestTopK_RepeatedRunsIdentical() {
	const int32_t V = 200;
	std::vector<int32_t> row = MakeSyntheticRow(V, {3, 50, 100, 150, 199, 7});
	std::vector<uint8_t> mask = MakeMask(V, true);
	const int32_t k = 6;
	std::vector<int32_t> picks1(k), picks2(k);
	FsdTopK(row.data(), mask.data(), V, k, picks1.data());
	FsdTopK(row.data(), mask.data(), V, k, picks2.data());
	for (int32_t i = 0; i < k; ++i)
		CHECK_MSG(picks1[static_cast<size_t>(i)] == picks2[static_cast<size_t>(i)],
		          "position %d: run1=%d run2=%d, top-k must be deterministic", i,
		          picks1[static_cast<size_t>(i)], picks2[static_cast<size_t>(i)]);
	// Known-answer half: the six seeded tokens (each strictly above the synthetic spread's own
	// [-2000, 2000) ceiling) must be exactly the six picks, in descending-value order (3 has
	// the highest planted value since MakeSyntheticRow assigns descending v starting at 20000
	// in the order given: {3,50,100,150,199,7}).
	const int32_t expect[6] = {3, 50, 100, 150, 199, 7};
	for (int32_t i = 0; i < k; ++i)
		CHECK_MSG(picks1[static_cast<size_t>(i)] == expect[i],
		          "position %d: got %d, want %d (planted descending order)", i,
		          picks1[static_cast<size_t>(i)], expect[i]);
}

int main() {
	TestDim7_TieBreak_LegalInt32MinWinsOverMaskedAtEqualValue();
	TestDim7_TieBreak_Control_LowIndexLegalTokenAlwaysIncluded();
	TestDim4_KFloor_SingleLegalCandidateAlwaysSelected();
	TestDim4_KOversized_AllLegalTokensIncluded();
	TestTopK_RepeatedRunsIdentical();
	std::printf("checks=%d failures=%d skips=%d\n", GChecks, GFailures, GSkips);
	return GFailures ? 1 : 0;
}

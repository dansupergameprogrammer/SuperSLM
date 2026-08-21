// T-2199 (Curie) -- fold 21, 2026-08-20: O2 REVERTED (plan commit `dbda73ab31`, ruling on
// `Claude/Poirot/927bbda-t2199-confirmation.md` S7/S8). This file INVERTS
// `o2_masked_not_queried_red.cpp` (removed, this commit) rather than deleting the
// construction outright -- the counting mock and the reason a public-interface cell can't
// observe this any other way are both still exactly right; only the RULED shape changed.
//
// The O2 fix-round remedy (queried the anti-LM only for unmasked picks) crossed in flight
// with the plan's own fold-20 correction of Sec7.5/Sec7.4, which states a masked pick's
// q_theta/p_omega reach `IExpConstruct` and the anti-LM lookup "exactly like an unmasked
// pick's -- nothing is skipped for it." Consequence, executed (Poirot S7):
// `DampedGreedyDiagnostics.pomspread_q15` read 3,276 where the true all-k spread was 29,491
// (and 16,384 where the true spread was 0) -- and pomspread is an operand of MODEL-GOVERNS,
// whose own worst-case margin is already 0.70x (plan Sec5.6.2). Ruling: O2 is REVERTED, not
// kept -- restoring the anti-LM query to every one of the k gathered picks, masked and
// unmasked alike (matching plan Sec7.5 as landed, the design of record). The score loop's
// own mask-floor branch is unaffected: a masked pick's score is still INT64_MIN regardless
// of its own (now-real) p_omega -- this cell's own construction confirms the anti-LM IS
// queried for masked picks; it does not, and does not need to, re-confirm the floor itself
// (that invariant is `TestDim7_MaskedNeverSelected_*`'s own subject, score_select_phaseC_red.cpp,
// unaffected by whether p_omega is real or fabricated for the losing candidate).
//
// Not observable through the public interface's own RETURN VALUES alone (AntiLmPenalize is a
// const, read-only query on an opaque state; a masked pick's own p_omega, real or fabricated,
// never changes which token wins, since the score floor ignores it either way). This cell
// links the REAL, unmodified `src/damped_greedy_topk.cpp` against a TEST-ONLY counting
// AntiLmState (`o2_counting_antilm_mock.cpp`, this directory, unchanged by this fold) instead
// of the real `src/damped_greedy_antilm.cpp` -- a counting construction, reachable without
// touching production code (a different .cpp providing the same extern symbols, substituted
// at the link step `build_o2_mock.bat` governs; `src/damped_greedy_topk.cpp` itself is never
// edited or aware of the substitution).
//
// RED-FIRST: EXECUTED, against this suite's own immediate parent commit -- `927bbda` (the
// fix-round pin round, O2 still in effect: the anti-LM queried only for unmasked picks).
// Running THIS inverted cell there reproduces exactly the shape the conductor's own dispatch
// named ("the o2 mock cell now fails 5/10"): the four masked candidates are absent from
// `g_o2_queried_candidates` (this cell's own new assertion, "must appear," fails for each),
// and the total-query-count assertion (want 6, the fix-round code produces 2) also fails --
// 5 of this file's own 11 checks red (measured). Against `b1dffd7` (the O2 revert, current tip): all six
// candidates appear, every check green. See this campaign's own test-design record for the
// transcript.
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#include "superslm/intmath.h"
#include "superslm/sslm_damped_greedy.h"

using namespace superslm;

namespace superslm {
extern std::vector<int32_t> g_o2_queried_candidates;
}  // namespace superslm
using superslm::g_o2_queried_candidates;

static int GChecks = 0;
static int GFailures = 0;

#define CHECK(cond) \
	do { \
		++GChecks; \
		if (!(cond)) { \
			++GFailures; \
			std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
			std::fflush(stdout); \
		} \
	} while (0)

#define CHECK_MSG(cond, ...) \
	do { \
		++GChecks; \
		if (!(cond)) { \
			++GFailures; \
			std::printf("FAIL %s:%d: %s -- ", __FILE__, __LINE__, #cond); \
			std::printf(__VA_ARGS__); \
			std::printf("\n"); \
			std::fflush(stdout); \
		} \
	} while (0)

int main() {
	const int32_t V = 20;
	const int32_t LEGAL_A = 3, LEGAL_B = 15;  // the two unmasked candidates
	const int32_t MASKED[] = {0, 1, 2, 4};    // four masked candidates, filling out k=6 with
	                                          // LEGAL_A/LEGAL_B (well inside n_legal < k, the
	                                          // ordinary condition under a bound schema)

	std::vector<int32_t> row(static_cast<std::size_t>(V), 0);
	// Descending, well-separated planted values so FsdTopK's own top-6 is exactly
	// {LEGAL_A, LEGAL_B, MASKED[0..3]} regardless of tie-break details.
	row[static_cast<std::size_t>(LEGAL_A)] = 20000;
	row[static_cast<std::size_t>(LEGAL_B)] = 19000;
	row[static_cast<std::size_t>(MASKED[0])] = 18000;
	row[static_cast<std::size_t>(MASKED[1])] = 17000;
	row[static_cast<std::size_t>(MASKED[2])] = 16000;
	row[static_cast<std::size_t>(MASKED[3])] = 15000;
	for (int32_t t = 5; t < V; ++t)
		if (t != LEGAL_B) row[static_cast<std::size_t>(t)] = -2000;  // filler, well below

	std::vector<uint8_t> mask(static_cast<std::size_t>((V + 7) / 8), 0);
	mask[static_cast<std::size_t>(LEGAL_A) >> 3] |=
	    static_cast<uint8_t>(1u << (LEGAL_A & 7));
	mask[static_cast<std::size_t>(LEGAL_B) >> 3] |=
	    static_cast<uint8_t>(1u << (LEGAL_B & 7));
	// MASKED[] positions: bit left clear (0) -- masked.

	AntiLmState* alm = AntiLmCreate(1);
	CHECK(alm != nullptr);

	int32_t out_token = -1;
	bool out_refused = false;
	// Any real, domain-valid (q_ln2, q_b, q_c) -- calibration is irrelevant to this cell's own
	// subject (which candidates AntiLmPenalize is called with). Derived directly here (this
	// file does not include fixture_common.h, to keep the mock build's own link line minimal
	// and independent of the rest of the suite) via the same certified IExpScaleConstants
	// search this suite's own DeriveScaleConstants helper uses, targeting the plan's own
	// corrected starting scale (q_ln2 = 493, plan Sec5.6.2) -- never a hand-typed constant.
	int64_t q_ln2 = 0, q_b = 0, q_c = 0;
	for (int64_t e = -80; e <= 0 && !q_ln2; ++e) {
		for (int64_t m = (int64_t)1 << 30; m < ((int64_t)1 << 31); m += ((int64_t)1 << 20)) {
			int64_t a = 0, b = 0, c = 0;
			if (IExpScaleConstants(m, e, kIExpLn2Q, 30, kIExpBQ, 30, kIExpCaQ, 30, &a, &b, &c) !=
			    IExpScaleDomain::kOk)
				continue;
			if (a == 493) {
				q_ln2 = a;
				q_b = b;
				q_c = c;
				break;
			}
		}
	}
	CHECK_MSG(q_ln2 == 493, "test fixture error: could not derive q_ln2=493 via "
	                        "IExpScaleConstants");
	const bool ok = DampedGreedyScoreAndArgmax(row.data(), mask.data(), V, /*k=*/6, alm,
	                                            /*alpha_q15=*/0, q_ln2, q_b, q_c, &out_token,
	                                            &out_refused);
	CHECK(ok);
	// The score-floor invariant this construction does NOT re-test, stated so a reader does
	// not look for it here: LEGAL_A/LEGAL_B are the model's own top two candidates by
	// construction (20000/19000, strictly above every MASKED value), so the winner is
	// necessarily one of them regardless of whether the anti-LM's own (now-real) p_omega for
	// the masked picks is ever read -- that floor is `TestDim7_MaskedNeverSelected_*`'s own
	// subject (score_select_phaseC_red.cpp), which sweeps a real anti-LM history and a real
	// alpha specifically to test it; this cell fixes alpha_q15=0 and a trivial anti-LM
	// precisely to keep that variable OUT of the way of the one thing under test here.
	CHECK_MSG(out_token == LEGAL_A || out_token == LEGAL_B,
	          "selected token %d is neither unmasked candidate -- fixture error, not this "
	          "cell's own subject",
	          out_token);

	// RULED SHAPE (fold 21, plan commit dbda73ab31, Sec7.5 as landed): the anti-LM is queried
	// for EVERY one of the k gathered picks, masked included.
	for (int32_t masked_cand : MASKED) {
		bool queried = false;
		for (int32_t q : g_o2_queried_candidates)
			if (q == masked_cand) queried = true;
		CHECK_MSG(queried,
		          "masked candidate %d is ABSENT from AntiLmPenalize's own query record -- "
		          "fold 21 reverted O2: the anti-LM must be queried for every one of the k "
		          "gathered picks, masked and unmasked alike (plan Sec7.5 as landed, "
		          "'a masked pick's own q_theta/p_omega reach IExpConstruct and the anti-LM "
		          "lookup exactly like an unmasked pick's')",
		          masked_cand);
	}
	for (int32_t legal_cand : {LEGAL_A, LEGAL_B}) {
		bool queried = false;
		for (int32_t q : g_o2_queried_candidates)
			if (q == legal_cand) queried = true;
		CHECK_MSG(queried, "unmasked candidate %d never appears in AntiLmPenalize's own query "
		                    "record",
		          legal_cand);
	}
	CHECK_MSG(g_o2_queried_candidates.size() == 6,
	          "AntiLmPenalize was called with %zu total candidate entries across this step, "
	          "want exactly 6 (all k gathered picks, masked and unmasked alike -- the O2 "
	          "revert's own shape) -- ScoreAndSelect makes exactly one AntiLmPenalize call "
	          "per step, batched over the full k-set",
	          g_o2_queried_candidates.size());

	AntiLmDestroy(alm);
	std::printf("checks=%d failures=%d skips=0\n", GChecks, GFailures);
	return GFailures ? 1 : 0;
}

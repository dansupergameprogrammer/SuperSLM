// T-2199 (Curie) -- fix round 2026-08-20, Poirot O2 (`Claude/Poirot/
// 7be9508-t2199-phaseAC-review.md`, `Claude/Brunel/t2199-phaseAC-build-2026-08-20.md`
// Sec12.1): "the shared ScoreAndSelect helper now queries AntiLmPenalize only for the
// UNMASKED picks among the k gathered candidates."
//
// Not observable through the public interface alone (AntiLmPenalize is a const, read-only
// query on an opaque state; its own result is never read for a masked pick either way, so no
// output value can distinguish "queried and discarded" from "never queried"). This cell links
// the REAL, unmodified `src/damped_greedy_topk.cpp` against a TEST-ONLY counting AntiLmState
// (`o2_counting_antilm_mock.cpp`, this directory) instead of the real
// `src/damped_greedy_antilm.cpp` -- a counting construction, per the commission's own named
// fallback for this remedy, reachable without touching production code (a different .cpp
// providing the same extern symbols, substituted at the link step build_o2_mock.bat governs;
// src/damped_greedy_topk.cpp itself is never edited or aware of the substitution).
//
// RED-FIRST: EXECUTED. Against `c473dad` (pre-fix `ScoreAndSelect`, which queries
// AntiLmPenalize over the WHOLE k-pick set including masked members -- Poirot's own O2
// finding, reproduced directly here): every one of the six gathered candidates appears in
// `g_o2_queried_candidates`, including the four MASKED ones -- this cell's own assertion
// (masked candidates never appear) FAILS. Against `826f607` (post-fix): only the two
// unmasked candidates appear. See this campaign's own test-design record for the transcript.
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

	for (int32_t masked_cand : MASKED) {
		bool queried = false;
		for (int32_t q : g_o2_queried_candidates)
			if (q == masked_cand) queried = true;
		CHECK_MSG(!queried,
		          "masked candidate %d appears in AntiLmPenalize's own query record -- a "
		          "masked pick's p_omega is never read (Sec7.5's own INT64_MIN floor), so it "
		          "must never be queried at all (Poirot O2)",
		          masked_cand);
	}
	// Positive control: the two UNMASKED candidates must have been queried -- a cell that only
	// checked "masked candidates absent" could pass a stub that queries NOTHING at all.
	for (int32_t legal_cand : {LEGAL_A, LEGAL_B}) {
		bool queried = false;
		for (int32_t q : g_o2_queried_candidates)
			if (q == legal_cand) queried = true;
		CHECK_MSG(queried, "unmasked candidate %d never appears in AntiLmPenalize's own query "
		                    "record -- the anti-LM must still be queried for every UNMASKED "
		                    "pick",
		          legal_cand);
	}
	CHECK_MSG(g_o2_queried_candidates.size() == 2,
	          "AntiLmPenalize was called with %zu total candidate entries across this step, "
	          "want exactly 2 (the unmasked picks only) -- ScoreAndSelect makes exactly one "
	          "AntiLmPenalize call per step, batched over the unmasked subset",
	          g_o2_queried_candidates.size());

	AntiLmDestroy(alm);
	std::printf("checks=%d failures=%d skips=0\n", GChecks, GFailures);
	return GFailures ? 1 : 0;
}

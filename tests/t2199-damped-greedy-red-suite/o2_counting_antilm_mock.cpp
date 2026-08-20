// o2_counting_antilm_mock.cpp -- TEST-ONLY. A replacement AntiLmState implementation, linked
// INSTEAD OF src/damped_greedy_antilm.cpp, purely for o2_masked_not_queried_red.cpp's own
// cell. Never touches production code: this file lives in the suite directory and is
// substituted at the LINK step (a different .cpp providing the same extern symbols
// sslm_damped_greedy.h declares), which src/damped_greedy_topk.cpp's own ScoreAndSelect does
// not and cannot distinguish from the real implementation -- it only ever calls AntiLmPenalize
// through the declared interface.
//
// Poirot O2 (`Claude/Poirot/7be9508-t2199-phaseAC-review.md`,
// `Claude/Brunel/t2199-phaseAC-build-2026-08-20.md` Sec12.1): "the shared ScoreAndSelect
// helper now queries AntiLmPenalize only for the UNMASKED picks among the k gathered
// candidates (a masked pick's p_omega is never read -- its score always floors to INT64_MIN
// regardless)." This claim has no observable consequence through the PUBLIC interface alone
// (AntiLmPenalize is a const, side-effect-free query per its own documented contract; the
// production AntiLmState is opaque, exposing no counter) -- the only way to OBSERVE which
// candidates a real caller queries, without touching src/damped_greedy_antilm.cpp, is to
// substitute an entirely different AntiLmState that records what it is asked, and link
// src/damped_greedy_topk.cpp (unmodified, real) against THAT instead.
#include "superslm/sslm_damped_greedy.h"

#include <cstddef>
#include <vector>

namespace superslm {

// The mock's own AntiLmState -- a private definition, distinct from and never linked
// alongside the production one (this file replaces damped_greedy_antilm.cpp entirely in this
// cell's own build recipe, see build_o2_mock.bat).
class AntiLmState {
public:
	int max_order = 1;
};

// Test-visible record of every candidate token id ever passed to AntiLmPenalize, across every
// call, in call order -- declared here, read by the test .cpp via `extern`.
std::vector<int32_t> g_o2_queried_candidates;

AntiLmState* AntiLmCreate(int max_order) {
	if (max_order < 1) return nullptr;
	AntiLmState* s = new AntiLmState();
	s->max_order = max_order;
	return s;
}

void AntiLmDestroy(AntiLmState* state) { delete state; }

void AntiLmUpdate(AntiLmState*, int32_t) {
	// No state to mutate -- this mock's only observable behavior is AntiLmPenalize's own
	// recording, below.
}

void AntiLmPenalize(const AntiLmState*, const int32_t* candidates, std::size_t k,
                     int64_t* out_p_omega_q15) {
	for (std::size_t i = 0; i < k; ++i) {
		g_o2_queried_candidates.push_back(candidates[i]);
		// The returned value is never read for a masked pick (O2's own claim, this cell's own
		// subject) and is irrelevant to which token wins among unmasked picks as long as it is
		// a small, fixed, non-adversarial constant -- 0 keeps s(v) = q_theta(v) exactly for
		// every unmasked candidate, so DampedGreedyScoreAndArgmax still resolves a
		// deterministic, legal winner in this cell's own construction.
		out_p_omega_q15[i] = 0;
	}
}

std::size_t AntiLmRetainedBytes(const AntiLmState*) { return 0; }

}  // namespace superslm

#ifndef SUPERSLM_INCLUDE_SPECDEC_DRAFTER_H
#define SUPERSLM_INCLUDE_SPECDEC_DRAFTER_H
// T-2246 speculative decoding (plan Claude/Plans/SuperSLM_SpecDecoding_SubPlan_2026-08-22.md
// SS3.1, CM-G2): the n-gram suffix drafter over a sequence's retained committed tokens.
//
// Pure function of `committed_tokens`: identical input yields an identical proposal, and no
// engine state is read or written. Proposes up to `k_max` ids by longest-suffix match -- the
// longest suffix of `committed_tokens` that occurs at an earlier position -- with the
// EARLIEST occurrence winning equal-length ties (C16's lowest-index discipline), then extends
// the matched occurrence greedily through its own continuation run in history.
//
// An occurrence qualifies only when its continuation index (`q + match_len`) still names a
// token inside history; the tail-only occurrence of the suffix therefore never wins by
// itself. A history repeating no token proposes zero ids -- the empty-draft fallback arm's
// input condition.
//
// Header-only on purpose: every build recipe in this tree compiles an explicit source list,
// and an inline definition lands the drafter in each consumer without touching any of them.
#ifndef __cplusplus
#error "superslm::SpecdecDraftPropose requires C++"
#endif

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

namespace superslm {

inline size_t SpecdecDraftPropose(const std::vector<int32_t>& committed_tokens, int32_t k_max,
                                   std::vector<int32_t>* out_drafts) {
	if (out_drafts == nullptr) return 0;
	out_drafts->clear();
	const size_t end = committed_tokens.size();
	if (end < 2 || k_max < 1) return 0;

	// Longest suffix length L in [1, end-1] whose earliest qualifying occurrence starts at q:
	// committed_tokens[q .. q+L) == committed_tokens[end-L .. end), with q + L <= end-1 so a
	// continuation token exists. Lengths descend; the first hit fixes both terms, which pins
	// the earliest-occurrence tie rule (q ascends within a length).
	size_t match_len = 0;
	size_t match_pos = 0;
	for (size_t len = end - 1; len >= 1; --len) {
		bool found = false;
		const size_t tail_start = end - len;
		const size_t q_max = end - 1 - len;  // last start with a continuation token
		for (size_t q = 0; q <= q_max; ++q) {
			if (std::memcmp(committed_tokens.data() + q, committed_tokens.data() + tail_start,
			                len * sizeof(int32_t)) == 0) {
				match_len = len;
				match_pos = q;
				found = true;
				break;
			}
		}
		if (found || len == 1) break;
	}
	if (match_len == 0) return 0;

	// Greedy extension: follow the matched occurrence's own continuation run in history, up
	// to k_max proposals or the end of what history holds.
	size_t idx = match_pos + match_len;
	size_t proposed = 0;
	while (idx < end && proposed < static_cast<size_t>(k_max)) {
		out_drafts->push_back(committed_tokens[idx]);
		++idx;
		++proposed;
	}
	return proposed;
}

}  // namespace superslm

#endif /* SUPERSLM_INCLUDE_SPECDEC_DRAFTER_H */

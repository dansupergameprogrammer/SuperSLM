// damped_greedy_antilm.cpp -- T-2199 Phase A: the n-gram anti-LM (plan Sec7.2, Sec8 Phase
// A1/A2). Design of record: Claude/Plans/superslm-1p2-fsd-plan-2026-08-19.md (Wizard repo).
//
// Fix round 2026-08-20 (Claude/Poirot/7be9508-t2199-phaseAC-review.md, FIX-THEN-SHIP):
// closes M4 (AntiLmCreate's own domain -- a negative max_order previously terminated the
// process, a zero max_order was silently accepted as a permanently-disabled anti-LM) and
// recalibrates S3 (AntiLmRetainedBytes read 3.0-5.9x low against measured process
// retention) -- its own residual then stated honestly rather than left as "recalibrated"
// alone (fold 21, plan Sec9 dim1, S9 of Claude/Poirot/927bbda-t2199-confirmation.md; see
// AntiLmRetainedBytes's own comment below).
#include "superslm/sslm_damped_greedy.h"

#include <unordered_map>
#include <vector>

#include "superslm/intmath.h"
#include "detail/context_hash.h"

namespace superslm {

namespace {

// ContextView RETIRED -- aliased to the design's own type (src/detail/context_hash.h),
// so every existing local-variable construction below (`ctx_view`, `ctx`) needs no edit.
using superslm::detail::ContextView;
// VecHash/VecEq RETIRED -- deleted outright. GrowableContextMap (src/detail/context_hash.h)
// computes HashContext/Eq internally and takes no policy-class template argument for
// either; nothing in this file constructs a VecHash or VecEq once tables_'s type changes
// below.

// Poirot S3 (2026-08-20) fit these constants to model MSVC's own `std::unordered_map`
// shape: a doubly-linked-list node, a cached hash, CRT small-allocation bookkeeping, and an
// amortized bucket-array share at load factor ~1. T-2296 (2026-08-26) replaced tables_/counts
// with the open-addressing GrowableContextMap/GrowableIntMap (src/detail/context_hash.h,
// src/detail/int_hash.h) -- not one clause of that model describes the container these
// constants now sit beside: an open-addressing slot is inline in one contiguous vector at
// load factor <= 0.5, with no linked-list node, no cached hash, and no per-element
// allocation (T-2299, S2). The constants below are UNCHANGED from Poirot's 2026-08-20 values
// -- they remain a stale, unretired model of a container this file no longer has, kept as a
// documented LOWER BOUND rather than refit, because refitting them to the real container
// shape is a separate, not-yet-authorized change to what AntiLmRetainedBytes returns (see the
// header's own comment for the honest, measured-workload-dependent understatement this
// produces post-T-2296, T-2302).
constexpr std::size_t kNodeOverhead = 48;         // stale: linked-list node + hash cache +
                                                   // CRT bookkeeping + bucket-array share,
                                                   // per element -- not this file's own
                                                   // container shape (see comment above)
constexpr std::size_t kSeparateAllocOverhead = 16;  // stale: a second heap allocation's own
                                                     // CRT bookkeeping (the context vector's
                                                     // backing buffer)
constexpr std::size_t kContextBaseOverhead = kNodeOverhead + kSeparateAllocOverhead;  // 64
constexpr std::size_t kContextPerTokenOverhead = sizeof(int32_t);
constexpr std::size_t kCandidateOverhead =
    sizeof(int32_t) + sizeof(int64_t) + kNodeOverhead;  // 60

}  // namespace

class AntiLmState {
public:
	explicit AntiLmState(int max_order)
	    : tables_(static_cast<size_t>(max_order)), max_order_(max_order) {}

	int max_order() const { return max_order_; }

	struct ContextEntry {
		superslm::detail::GrowableIntMap<int32_t, int64_t, superslm::detail::MixKeyS32> counts;
		int64_t total = 0;
	};

	// tables_[i-1] holds order i's context -> {candidate counts, total}. Exact-key lookup
	// only: every read below is a direct `find`, never a traversal of the map's own bucket
	// order (Sec7.2's own determinism argument).
	std::vector<superslm::detail::GrowableContextMap<ContextEntry>> tables_;
	std::vector<int32_t> history_;
	std::size_t retained_bytes_ = 0;

private:
	int max_order_;
};

// Domain: max_order >= 1 (Sec7.2's own "orders 1..N"). Poirot M4, executed:
// `AntiLmCreate(-1)` previously cast to a huge `size_t` in `tables_`'s own constructor and
// terminated the process (`0xC0000409`); `AntiLmCreate(0)` was silently accepted and
// produced a state that always returns `p_omega = 0` -- a permanently-disabled anti-LM
// rather than a rejection. Returns nullptr for max_order < 1; `AntiLmDestroy(nullptr)` is
// safe (matches `delete nullptr`).
AntiLmState* AntiLmCreate(int max_order) {
	if (max_order < 1) return nullptr;
	return new AntiLmState(max_order);
}

void AntiLmDestroy(AntiLmState* state) { delete state; }

std::size_t AntiLmHistorySize(const AntiLmState* state) { return state->history_.size(); }

int32_t AntiLmHistoryTokenAt(const AntiLmState* state, std::size_t index) {
	return state->history_[index];
}

void AntiLmUpdate(AntiLmState* state, int32_t token) {
	const std::size_t hist_size = state->history_.size();
	for (int order = 1; order <= state->max_order(); ++order) {
		const std::size_t ctx_len = static_cast<std::size_t>(order - 1);
		if (ctx_len > hist_size) continue;  // not enough history yet to form this order's context
		const ContextView ctx_view{ctx_len ? state->history_.data() + hist_size - ctx_len : nullptr,
		                              ctx_len};
		auto& table = state->tables_[static_cast<size_t>(order - 1)];
		AntiLmState::ContextEntry* existing = table.Find(ctx_view);
		if (!existing) {
			state->retained_bytes_ += kContextBaseOverhead + ctx_view.size * kContextPerTokenOverhead;
		}
		AntiLmState::ContextEntry& entry = existing ? *existing : table.FindOrEmplace(ctx_view);
		if (!entry.counts.Find(token)) { state->retained_bytes_ += kCandidateOverhead; }
		entry.counts[token] += 1;
		entry.total += 1;
	}
	state->history_.push_back(token);
}

namespace {

// Q15 fixed-point beta=0.9 decay (Sec7.2's own paper-cited exponential decay), and the
// order weight beta^(N - i) for order i of N (i=N, the highest/most specific order, gets
// the largest raw weight -- "gives more weight to higher, more specific orders when they
// have data", per this suite's own cross-reference cell).
constexpr int64_t kBetaQ15 = 29491;  // round(0.9 * 32768)

int64_t Q15Pow(int64_t base_q15, int exponent) {
	int64_t result = int64_t{1} << kProbFracBits;  // 1.0 in Q15
	for (int i = 0; i < exponent; ++i) result = (result * base_q15) >> kProbFracBits;
	return result;
}

}  // namespace

void AntiLmPenalize(const AntiLmState* state, const int32_t* candidates, std::size_t k,
                     int64_t* out_p_omega_q15) {
	const int max_order = state->max_order();
	const std::size_t hist_size = state->history_.size();

	// First pass finds the active orders and their exact normalization denominator. The
	// second pass repeats the exact-key lookups and mixes directly into caller storage. This
	// avoids the former ActiveOrder/raw-weight/normalized-weight heap vectors on every token.
	int active_count = 0;
	int64_t raw_sum = 0;
	for (int order = 1; order <= max_order; ++order) {
		const std::size_t ctx_len = static_cast<std::size_t>(order - 1);
		if (ctx_len > hist_size) continue;
		const ContextView ctx{ctx_len ? state->history_.data() + hist_size - ctx_len : nullptr,
		                      ctx_len};
		const auto& table = state->tables_[static_cast<size_t>(order - 1)];
		const AntiLmState::ContextEntry* entry = table.Find(ctx);
		if (!entry) continue;  // context never observed -- order excluded, not zeroed
		++active_count;
		raw_sum += Q15Pow(kBetaQ15, max_order - order);
	}

	if (active_count == 0) {
		for (std::size_t c = 0; c < k; ++c) out_p_omega_q15[c] = 0;
		return;
	}

	for (std::size_t c = 0; c < k; ++c) out_p_omega_q15[c] = 0;
	int active_index = 0;
	int64_t normalized_sum = 0;
	for (int order = 1; order <= max_order; ++order) {
		const std::size_t ctx_len = static_cast<std::size_t>(order - 1);
		if (ctx_len > hist_size) continue;
		const ContextView ctx{ctx_len ? state->history_.data() + hist_size - ctx_len : nullptr,
		                      ctx_len};
		const auto& table = state->tables_[static_cast<size_t>(order - 1)];
		const AntiLmState::ContextEntry* entry = table.Find(ctx);
		if (!entry) continue;
		++active_index;
		const int64_t normalized_weight =
		    (active_index == active_count)
		        ? ((int64_t{1} << kProbFracBits) - normalized_sum)
		        : ((Q15Pow(kBetaQ15, max_order - order) << kProbFracBits) / raw_sum);
		normalized_sum += normalized_weight;
		for (std::size_t c = 0; c < k; ++c) {
			const int64_t* found = entry->counts.Find(candidates[c]);
			const int64_t count = found ? *found : 0;
			const int64_t ratio_q15 = (count << kProbFracBits) / entry->total;
			out_p_omega_q15[c] += (normalized_weight * ratio_q15) >> kProbFracBits;
		}
	}
	for (std::size_t c = 0; c < k; ++c) {
		if (out_p_omega_q15[c] < 0) out_p_omega_q15[c] = 0;
		if (out_p_omega_q15[c] > (int64_t{1} << kProbFracBits)) {
			out_p_omega_q15[c] = int64_t{1} << kProbFracBits;
		}
	}
}

// Reports the n-gram COUNT TABLE's own retained memory -- the component Sec7.2 claims is
// "O(distinct n-grams seen so far)" and Sec8 Phase A2's own memory-growth cell pins as
// UNCHANGED after replaying an already-seen sequence a second time
// (`TestPhaseA_MemoryGrowth_BoundedByDistinctNgrams`, this suite). It deliberately does NOT
// include `history_` (Poirot S3): that buffer grows by one token on every `AntiLmUpdate`
// call regardless of repeats, so folding it into this same figure would grow the reported
// total on a replay of already-seen content and break the cited cell's own passing
// assertion -- a real, suite-pinned property of THIS metric, not an oversight. `history_`'s
// own footprint is a separately, exactly computable quantity a caller pricing the anti-LM's
// TOTAL state adds directly: `AntiLmRetainedBytes(state) + generation_length_so_far *
// sizeof(int32_t)` (plus a small constant for the `std::vector<int32_t>` object itself),
// where `generation_length_so_far` is the caller's own count of `AntiLmUpdate` calls made
// against this state (this interface has no accessor for it, matching the suite's own
// declared two-operation surface -- update/penalize -- which this build does not extend).
//
// The table-portion residual itself is STATED HONESTLY, not merely "recalibrated": this
// reading is a LOWER BOUND, and the gap between it and the real footprint is
// workload-dependent, not a stable per-order ratio -- MEASURED (T-2302, 2026-08-26, following
// T-2299's own method) at 1.31x-2.61x low (max_order=1), 3.85x-4.98x low (max_order=3), and
// 4.54x-5.60x low (max_order=5), across the vocab sizes and generation lengths T-2302 sampled.
// See the production header's own copy of this note for the full range table and the caller
// guidance it carries.
std::size_t AntiLmRetainedBytes(const AntiLmState* state) { return state->retained_bytes_; }

}  // namespace superslm

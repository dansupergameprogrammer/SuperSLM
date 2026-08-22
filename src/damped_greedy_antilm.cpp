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

namespace superslm {

namespace {

struct ContextView {
	const int32_t* data;
	std::size_t size;
};

// Transparent hash/equality let hot-path lookups use a non-owning suffix view. A vector is
// allocated only when a genuinely new context becomes persistent table state.
struct VecHash {
	using is_transparent = void;
	std::size_t operator()(ContextView v) const noexcept {
		std::size_t h = 1469598103934665603ull;
		for (std::size_t i = 0; i < v.size; ++i) {
			h ^= static_cast<std::size_t>(static_cast<uint32_t>(v.data[i]));
			h *= 1099511628211ull;
		}
		return h;
	}
	std::size_t operator()(const std::vector<int32_t>& v) const noexcept {
		return (*this)(ContextView{v.data(), v.size()});
	}
};

struct VecEq {
	using is_transparent = void;
	bool operator()(ContextView a, ContextView b) const noexcept {
		if (a.size != b.size) return false;
		return a.size == 0 || std::equal(a.data, a.data + a.size, b.data);
	}
	bool operator()(const std::vector<int32_t>& a, const std::vector<int32_t>& b) const noexcept {
		return (*this)(ContextView{a.data(), a.size()}, ContextView{b.data(), b.size()});
	}
	bool operator()(const std::vector<int32_t>& a, ContextView b) const noexcept {
		return (*this)(ContextView{a.data(), a.size()}, b);
	}
	bool operator()(ContextView a, const std::vector<int32_t>& b) const noexcept {
		return (*this)(a, ContextView{b.data(), b.size()});
	}
};

// Poirot S3, recalibrated 2026-08-20 against measured process-memory deltas (the review's
// own table: reported figures read 2.97-5.92x low against `PrivateUsage` at max_order in
// {1,3,5}, most stably ~5.8x at this design's own default max_order=3). The prior constants
// (32 / 4 / 28 bytes) modeled only the raw key/value payload plus a small guess at node
// overhead; these model MSVC's own `unordered_map` shape more closely -- each element is a
// SEPARATE heap allocation (a doubly-linked-list node: ~16 bytes prev/next + ~8 bytes cached
// hash, plus typical CRT small-allocation bookkeeping ~16 bytes, plus an amortized bucket-
// array share at load factor ~1, ~8 bytes/element -- summing to ~48 bytes of overhead per
// element beyond its own key+value payload), and a context key additionally owns a SECOND,
// separately-allocated heap buffer (the `std::vector<int32_t>` context tuple's own backing
// array), taxed at the same ~16 bytes of per-allocation bookkeeping. This remains a MODEL,
// not a byte-exact accounting (real allocator layout is platform- and load-factor-
// dependent) -- it is now calibrated toward a measured population rather than hand-picked,
// per `StandardsDocument.md` Sec5.4.
constexpr std::size_t kNodeOverhead = 48;         // linked-list node + hash cache + CRT
                                                   // per-allocation bookkeeping + amortized
                                                   // bucket-array share, per element
constexpr std::size_t kSeparateAllocOverhead = 16;  // a second heap allocation's own CRT
                                                     // bookkeeping (the context vector's
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
		std::unordered_map<int32_t, int64_t> counts;
		int64_t total = 0;
	};

	// tables_[i-1] holds order i's context -> {candidate counts, total}. Exact-key lookup
	// only: every read below is a direct `find`, never a traversal of the map's own bucket
	// order (Sec7.2's own determinism argument).
	std::vector<std::unordered_map<std::vector<int32_t>, ContextEntry, VecHash, VecEq>> tables_;
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
		auto it = table.find(ctx_view);
		if (it == table.end()) {
			std::vector<int32_t> persistent_ctx(
			    state->history_.end() - static_cast<long>(ctx_len), state->history_.end());
			state->retained_bytes_ +=
			    kContextBaseOverhead + persistent_ctx.size() * kContextPerTokenOverhead;
			it = table.emplace(std::move(persistent_ctx), AntiLmState::ContextEntry{}).first;
		}
		auto& entry = it->second;
		if (entry.counts.find(token) == entry.counts.end()) {
			state->retained_bytes_ += kCandidateOverhead;
		}
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
		auto it = table.find(ctx);
		if (it == table.end()) continue;  // context never observed -- order excluded, not zeroed
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
		const auto it = table.find(ctx);
		if (it == table.end()) continue;
		++active_index;
		const int64_t normalized_weight =
		    (active_index == active_count)
		        ? ((int64_t{1} << kProbFracBits) - normalized_sum)
		        : ((Q15Pow(kBetaQ15, max_order - order) << kProbFracBits) / raw_sum);
		normalized_sum += normalized_weight;
		for (std::size_t c = 0; c < k; ++c) {
			const auto found = it->second.counts.find(candidates[c]);
			const int64_t count = (found == it->second.counts.end()) ? 0 : found->second;
			const int64_t ratio_q15 = (count << kProbFracBits) / it->second.total;
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
// The table-portion residual itself is STATED HONESTLY, not merely "recalibrated" (fold 21,
// plan Sec9 dim1, S9 of `Claude/Poirot/927bbda-t2199-confirmation.md`): this reading is a
// LOWER BOUND, still ~2.9x low at this design's own default max_order=3 against measured
// process-memory deltas -- the recalibration above reasoned the constants forward from an
// allocator model rather than fitting them to the measured population, which is why the gap
// narrowed (from ~5.8x) rather than closed. See the production header's own copy of this note
// for the full per-order ratio table.
std::size_t AntiLmRetainedBytes(const AntiLmState* state) { return state->retained_bytes_; }

}  // namespace superslm

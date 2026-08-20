// damped_greedy_antilm.cpp -- T-2199 Phase A: the n-gram anti-LM (plan Sec7.2, Sec8 Phase
// A1/A2). Design of record: Claude/Plans/superslm-1p2-fsd-plan-2026-08-19.md (Wizard repo).
#include "superslm/sslm_damped_greedy.h"

#include <unordered_map>
#include <vector>

#include "superslm/intmath.h"

namespace superslm {

namespace {

// ---- a small hash for a variable-length token-context key --------------------------
struct VecHash {
	std::size_t operator()(const std::vector<int32_t>& v) const noexcept {
		std::size_t h = 1469598103934665603ull;  // FNV-1a offset basis
		for (int32_t t : v) {
			h ^= static_cast<std::size_t>(static_cast<uint32_t>(t));
			h *= 1099511628211ull;  // FNV-1a prime
		}
		return h;
	}
};

// Rough, deliberately simple per-entry byte costs -- AntiLmRetainedBytes only needs to grow
// monotonically with the number of DISTINCT n-grams observed (Sec7.2's own "Memory is
// O(distinct n-grams seen so far)" claim), not report an exact allocator size.
constexpr std::size_t kContextBaseOverhead = 32;
constexpr std::size_t kContextPerTokenOverhead = sizeof(int32_t);
constexpr std::size_t kCandidateOverhead = sizeof(int32_t) + sizeof(int64_t) + 16;

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
	std::vector<std::unordered_map<std::vector<int32_t>, ContextEntry, VecHash>> tables_;
	std::vector<int32_t> history_;
	std::size_t retained_bytes_ = 0;

private:
	int max_order_;
};

AntiLmState* AntiLmCreate(int max_order) { return new AntiLmState(max_order); }

void AntiLmDestroy(AntiLmState* state) { delete state; }

void AntiLmUpdate(AntiLmState* state, int32_t token) {
	const std::size_t hist_size = state->history_.size();
	for (int order = 1; order <= state->max_order(); ++order) {
		const std::size_t ctx_len = static_cast<std::size_t>(order - 1);
		if (ctx_len > hist_size) continue;  // not enough history yet to form this order's context
		std::vector<int32_t> ctx(state->history_.end() - static_cast<long>(ctx_len),
		                          state->history_.end());
		auto& table = state->tables_[static_cast<size_t>(order - 1)];
		auto it = table.find(ctx);
		if (it == table.end()) {
			state->retained_bytes_ += kContextBaseOverhead + ctx.size() * kContextPerTokenOverhead;
			it = table.emplace(std::move(ctx), AntiLmState::ContextEntry{}).first;
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

	// One lookup per order (query context is state's own current history, shared by every
	// candidate this call), never per candidate -- exact-key lookup only.
	struct ActiveOrder {
		const AntiLmState::ContextEntry* entry;
		int order;  // 1-based
	};
	std::vector<ActiveOrder> active;
	active.reserve(static_cast<size_t>(max_order));
	for (int order = 1; order <= max_order; ++order) {
		const std::size_t ctx_len = static_cast<std::size_t>(order - 1);
		if (ctx_len > hist_size) continue;
		std::vector<int32_t> ctx(state->history_.end() - static_cast<long>(ctx_len),
		                          state->history_.end());
		const auto& table = state->tables_[static_cast<size_t>(order - 1)];
		auto it = table.find(ctx);
		if (it == table.end()) continue;  // context never observed -- order excluded, not zeroed
		active.push_back(ActiveOrder{&it->second, order});
	}

	if (active.empty()) {
		for (std::size_t c = 0; c < k; ++c) out_p_omega_q15[c] = 0;
		return;
	}

	// Normalized mixing weights over the ACTIVE orders only (Sec7.2's own "normalized before
	// mixing... a genuine convex combination"), exact-sum-to-2^15 via largest-remainder: floor
	// every weight but the last active order, then let the last absorb the residual so the
	// weights sum to exactly kProbFracBits's own 1<<15 regardless of rounding.
	std::vector<int64_t> raw_w(active.size());
	int64_t raw_sum = 0;
	for (std::size_t j = 0; j < active.size(); ++j) {
		raw_w[j] = Q15Pow(kBetaQ15, max_order - active[j].order);
		raw_sum += raw_w[j];
	}
	std::vector<int64_t> norm_w(active.size());
	int64_t norm_sum = 0;
	for (std::size_t j = 0; j + 1 < active.size(); ++j) {
		norm_w[j] = (raw_w[j] << kProbFracBits) / raw_sum;
		norm_sum += norm_w[j];
	}
	norm_w.back() = (int64_t{1} << kProbFracBits) - norm_sum;

	for (std::size_t c = 0; c < k; ++c) {
		const int32_t cand = candidates[c];
		int64_t mixed = 0;
		for (std::size_t j = 0; j < active.size(); ++j) {
			const auto& entry = *active[j].entry;
			const auto found = entry.counts.find(cand);
			const int64_t count = (found == entry.counts.end()) ? 0 : found->second;
			const int64_t ratio_q15 = (count << kProbFracBits) / entry.total;  // total >= 1 always
			mixed += (norm_w[j] * ratio_q15) >> kProbFracBits;
		}
		if (mixed < 0) mixed = 0;
		if (mixed > (int64_t{1} << kProbFracBits)) mixed = int64_t{1} << kProbFracBits;
		out_p_omega_q15[c] = mixed;
	}
}

std::size_t AntiLmRetainedBytes(const AntiLmState* state) { return state->retained_bytes_; }

}  // namespace superslm

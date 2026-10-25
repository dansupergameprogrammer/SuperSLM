// src/detail/anti_lm_update.h -- AntiLmUpdate in two phases, for a caller that must make every fallible
// step of a token before it publishes any of them.
//
// Schema fast-decode plan rev 1.3.3 §4.5 (slice TF): the decode step's finish is a transaction. Its
// fallible steps all run before its commit, and AntiLmUpdate is one of them, so it splits into a
// fallible reserve and a no-throw apply:
//   AntiLmReserveUpdate  every allocation the token needs (AntiLmUpdate's phase 1), held in a
//                        caller-owned AntiLmPendingUpdate. The state's content -- its history and
//                        every count -- is unchanged, whether it returns or throws std::bad_alloc;
//                        only spare capacity may grow.
//   AntiLmApplyUpdate    publishes the reserved token (AntiLmUpdate's phase 2). Cannot fail.
// AntiLmUpdate (sslm_damped_greedy.h) is the two in sequence, with the same allocations, in the same
// order, as before the split.
//
// The reservation lives with the caller rather than in AntiLmState so that the state's layout, which
// the t2296 capacity cells pin through its allocation sizes, is unchanged. Between the reserve and the
// apply the state must not be updated or destroyed; reading it (AntiLmPenalize, the history accessors)
// is fine. A reservation that is never applied publishes nothing and is released with its object.
#ifndef SUPERSLM_SRC_DETAIL_ANTI_LM_UPDATE_H
#define SUPERSLM_SRC_DETAIL_ANTI_LM_UPDATE_H

#include <cstddef>
#include <cstdint>
#include <vector>

#include "superslm/sslm_damped_greedy.h"
#include "int_hash.h"

namespace superslm {

// One order's context entry: its candidate counts and their total. AntiLmState::ContextEntry is this
// type (damped_greedy_antilm.cpp); it is named here so a reservation can hold a new context's entry.
struct AntiLmContextEntry {
	// {1}: the construction-site hint (fold round 26, D-SLM4769), unchanged by the move to this header.
	superslm::detail::GrowableIntMap<int32_t, int64_t, superslm::detail::MixKeyS32> counts{1};
	int64_t total = 0;
};

class AntiLmPendingUpdate;
void AntiLmReserveUpdate(AntiLmState* state, int32_t token, AntiLmPendingUpdate* pending);
void AntiLmApplyUpdate(AntiLmState* state, AntiLmPendingUpdate* pending) noexcept;

class AntiLmPendingUpdate {
public:
	AntiLmPendingUpdate() = default;
	AntiLmPendingUpdate(const AntiLmPendingUpdate&) = delete;
	AntiLmPendingUpdate& operator=(const AntiLmPendingUpdate&) = delete;

	// True between a reserve that returned and the apply.
	bool reserved() const noexcept { return state_ != nullptr; }

private:
	friend void AntiLmReserveUpdate(AntiLmState* state, int32_t token, AntiLmPendingUpdate* pending);
	friend void AntiLmApplyUpdate(AntiLmState* state, AntiLmPendingUpdate* pending) noexcept;

	struct Order {
		std::vector<int32_t> key;   // a new context's key
		AntiLmContextEntry entry;   // a new context's entry, already holding {token: 1}
		bool is_new = false;
	};
	static constexpr int kInlineOrders = 8;  // no heap scratch at the orders hosts use

	Order inline_[kInlineOrders];
	std::vector<Order> heap_;  // only above kInlineOrders
	Order* orders_ = nullptr;
	AntiLmState* state_ = nullptr;
	std::size_t hist_size_ = 0;
	int32_t token_ = -1;
};

}  // namespace superslm

#endif  // SUPERSLM_SRC_DETAIL_ANTI_LM_UPDATE_H

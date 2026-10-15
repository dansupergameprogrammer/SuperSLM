// Paged-KV plan rev 16.21 cell 7.9 and §20.2 (D-SLM8370): the scattered reserve, through the ABI only.
//
// The scatter guard grades the measured holder's mapped page ids by the mode-step count M against M99(n, P), the 99th
// percentile of M over uniformly random reserves. A fresh pool hands a whole-reserve holder its pages in order (M =
// n - 1, rejected), and rev 16.15's pair churn (budget-1 holders of two pages each, released in a shuffled order) hands
// them back in adjacent pairs (M about n/2, rejected: the order the guard exists to reject). This construction takes
// the pool's pages ONE AT A TIME, then returns them in a fixed shuffled order, so the next holder draws a permutation of
// single pages:
//   1. repeatedly: sslm_prefix_begin_budgeted(model, pool, 1) reserves R(1) = 2 pages; one prompt token is prefilled,
//      which maps the first; sslm_prefix_freeze returns the unused reserve page to the pool. Each frozen prefix holds
//      exactly one page. This stops when a begin is refused SSLM_KV_POOL_EXHAUSTED (one page is left free), so P - 1
//      prefixes hold P - 1 single pages (T-3162's note, executed here);
//   2. the prefixes are released in a Fisher-Yates order of a fixed xorshift seed, so the pool's free order becomes that
//      permutation of single pages (the page module's free list returns the last page released first; the order is a
//      permutation either way);
//   3. the caller then restores (or creates) the measured holder, which draws its reserve from that order.
// The seed is fixed, not per process: cell 7.9's five processes must read equal id lists (the driver's id_lists_agree),
// so every process builds the same reserve. Only public ABI verbs are used, so the production pair (tools/
// paged_kv_pair.cpp, row79) calls it before its paged restore. Cost: P - 1 one-token prefills of the model per process
// (P = 64 or 512 at B = 64; 256 or 2,048 for 7.9-MR-OPT at B = 16).
//
// Executed on the bench at every row geometry, against the builder's T4 tree, with the T4 driver's own guard: the test
// author's record, T-3189 §11.

#ifndef SUPERSLM_TESTS_PKV_SCATTER_RESERVE_H
#define SUPERSLM_TESTS_PKV_SCATTER_RESERVE_H

#include "superslm/sslm_abi.h"

#include <cstdint>
#include <vector>

namespace pkv {

// Cell 7.9's reserve: one fixed order for all five processes of every row. The guard is a 99th-percentile bound, so a
// uniformly random reserve fails a given row about 1% of the time; across 200 seeds this construction's pass rate at
// the 0.5B rows equals a uniform reserve's (R2a 0.990 against 0.995, R2c 0.995 against 0.996, R2b 1.000 against 0.999).
// Seed 7970 drew one of those 1% at R2a (M = 4 against M99 = 3); seed 1 passes every row at both page sizes. The order
// depends only on P and the verb sequence, not on the model, so the bench's verdicts are the box's (T-3189 §11).
constexpr uint32_t kScatterReserveSeed = 1u;

// Builds the scattered reserve in `pool` (which must hold no holder yet). Returns SSLM_OK with *singles = the number of
// single pages taken and returned (P - 1 on a fresh pool), or the first refusal other than the expected
// SSLM_KV_POOL_EXHAUSTED, with every prefix it made released.
inline sslm_status ScatterReserve(sslm_model model, sslm_kv_pool* pool, uint32_t seed, uint32_t* singles) {
	std::vector<sslm_prefix> held;
	sslm_status st = SSLM_OK;
	for (;;) {
		sslm_prefix p = nullptr;
		st = sslm_prefix_begin_budgeted(model, pool, 1, &p);
		if (st == SSLM_KV_POOL_EXHAUSTED) {
			st = SSLM_OK;
			break;
		}
		if (st != SSLM_OK) break;
		const int32_t token = 0;
		int32_t consumed = 0;
		st = sslm_prefix_prefill(model, p, &token, 1, 1, SSLM_SPAN_PROMPT, nullptr, &consumed);
		if (st == SSLM_OK && consumed != 1) st = SSLM_INVALID_ARGUMENT;
		if (st == SSLM_OK) st = sslm_prefix_freeze(p);
		if (st != SSLM_OK) {
			sslm_prefix_release(p);
			break;
		}
		held.push_back(p);
	}
	if (singles) *singles = static_cast<uint32_t>(held.size());
	std::vector<uint32_t> order(held.size());
	for (uint32_t i = 0; i < order.size(); ++i) order[i] = i;
	uint32_t x = seed ? seed : 1u;
	for (uint32_t i = static_cast<uint32_t>(order.size()); i > 1; --i) {
		x ^= x << 13;
		x ^= x >> 17;
		x ^= x << 5;
		const uint32_t k = x % i, t = order[i - 1];
		order[i - 1] = order[k];
		order[k] = t;
	}
	for (uint32_t i : order) sslm_prefix_release(held[i]);
	return st;
}

}  // namespace pkv

#endif  // SUPERSLM_TESTS_PKV_SCATTER_RESERVE_H

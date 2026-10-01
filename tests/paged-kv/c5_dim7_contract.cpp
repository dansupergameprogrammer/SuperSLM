// Paged-KV plan (rev 16.1) §7 dimension 7, the budget-holder parts (step C5): 7.1, 7.2, 7.3, 7.5,
// 7.6's budget part, 7.8's budget-verb sentences, 7.10's budget part, 7.11's SSB6 size (moved to a
// budget holder by the carrier row RB16-CELLS), 7.12's budget part, 7.13's budget-ABI part and 7.14.
//
// Page counts are graded by the two-sided fill probe or the legacy-create admission count (§8),
// never by the stats verbs; 7.5 is the one cell whose subject is the stats verbs themselves.
//
// This file replaces the global operator new with a counting one (7.10's allocation counter, the
// construction of tests/t2138-abi-red-suite/dim7_contract_red.cpp and
// tests/decode_threading_alloc_main.cpp). The replacement counts and forwards to malloc; it is
// process-wide for superslm_pkv_c5, so no other C5 translation unit may replace operator new.

#include "pkv_budget_b_helpers.h"

#include <atomic>
#include <climits>
#include <cstdint>
#include <cstdlib>
#include <new>

namespace {
std::atomic<long long> g_new_calls{0};
}  // namespace

// GCC's -Wmismatched-new-delete cannot see that these replacements pair malloc with free.
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmismatched-new-delete"
#endif
void* operator new(std::size_t size) {
	g_new_calls.fetch_add(1, std::memory_order_relaxed);
	if (void* p = std::malloc(size ? size : 1)) return p;
	throw std::bad_alloc{};
}
void* operator new[](std::size_t size) { return ::operator new(size); }
void* operator new(std::size_t size, std::align_val_t al) {
	g_new_calls.fetch_add(1, std::memory_order_relaxed);
	const size_t a = static_cast<size_t>(al);
	const size_t n = (size + a - 1) / a * a;
	if (void* p = std::aligned_alloc(a, n ? n : a)) return p;
	throw std::bad_alloc{};
}
void* operator new[](std::size_t size, std::align_val_t al) { return ::operator new(size, al); }
void operator delete(void* p) noexcept { std::free(p); }
void operator delete(void* p, std::size_t) noexcept { std::free(p); }
void operator delete[](void* p) noexcept { std::free(p); }
void operator delete[](void* p, std::size_t) noexcept { std::free(p); }
void operator delete(void* p, std::align_val_t) noexcept { std::free(p); }
void operator delete(void* p, std::size_t, std::align_val_t) noexcept { std::free(p); }
void operator delete[](void* p, std::align_val_t) noexcept { std::free(p); }
void operator delete[](void* p, std::size_t, std::align_val_t) noexcept { std::free(p); }
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic pop
#endif

namespace {

using namespace pkv;
using namespace pkv::budget_b;

// A world-only adopter: a budget prefix of `len` tokens (Stream 51, the prefix_lengths scenario's)
// and a budget-b sequence that adopts it, in a pool the caller sizes.
struct WorldAdopter {
	sslm_prefix px = nullptr;
	sslm_seq s = nullptr;
	WorldAdopter(const Fixture& fx, sslm_kv_pool* pool, int32_t len, int32_t prefix_budget, int32_t b) {
		px = MakePrefix(fx, pool, Stream(51, len, fx.vocab), prefix_budget);
		s = px ? MakeBudgetSeq(fx, pool, b) : nullptr;
	}
	~WorldAdopter() {
		if (s) sslm_seq_release(s);
		if (px) sslm_prefix_release(px);
	}
};

// The legacy twin of a world-only adopter: a legacy pool, a legacy prefix of the same tokens, a
// legacy sequence that copies it, and `n` tokens decoded.
std::vector<int32_t> LegacyTwinTokens(const Fixture& fx, int32_t len, int n) {
	LegacyPool pool(fx.model, 2);
	std::vector<int32_t> t;
	sslm_prefix px = MakePrefix(fx, &pool.pool, Stream(51, len, fx.vocab), 0);
	sslm_seq s = px ? MakeLegacySeq(fx, &pool.pool) : nullptr;
	if (s && sslm_seq_adopt_prefix(s, px) == SSLM_OK) DecodeN(fx, s, n, &t);
	if (s) sslm_seq_release(s);
	if (px) sslm_prefix_release(px);
	return t;
}

// ---- 7.1 [C5] -------------------------------------------------------------------------------------

// "Adopt never draws from the pool" (5.4's construction): a budget-1,008 prefix of length 1,000
// (mid-page, so share adopt copies a tail page) and a budget-512 adopter in a pool of
// R(1008) + R(512) + 1 pages. Freeze returns R(1008) - 63 = 1 page, so 2 are free once the adopter
// exists; a budget-1 create (2 pages) takes them, and a second is refused, so the pool is exhausted.
// The adopt then succeeds, and the adopter runs its whole budget.
void Cell71() {
	const Fixture& fx = GetFixture("pkv_def");
	if (!fx.ok) return;
	for (int32_t len : {1000, 1008}) {
		PagePool pool(fx.model, static_cast<uint32_t>(fx.R(1008) + fx.R(512) + 1));
		WorldAdopter w(fx, &pool.pool, len, 1008, 512);
		if (!w.s) continue;
		sslm_seq filler = MakeBudgetSeq(fx, &pool.pool, 1);
		sslm_seq extra = nullptr;
		PKV_CHECK_EQ(sslm_seq_create_budgeted(fx.model, &pool.pool, 1, &extra), SSLM_KV_POOL_EXHAUSTED);
		if (extra) sslm_seq_release(extra);
		// kills: adopt that draws its tail page from the pool (refused here at 1,000; at 1,008 there is
		// no tail, so that mutant is equivalent there, and the aligned case is the control)
		PKV_CHECK_MSG(sslm_seq_adopt_prefix(w.s, w.px) == SSLM_OK, "adopt of a %d-token prefix refused in an exhausted pool", len);
		sslm_status refusal = SSLM_OK;
		const std::vector<int32_t> got = RunToRefusal(fx, w.s, &refusal, 600);
		PKV_CHECK_EQ(got.size(), 513u);
		PKV_CHECK_EQ(refusal, PKV_KV_BUDGET_EXCEEDED);
		PKV_CHECK_MSG(got == LegacyTwinTokens(fx, len, 513), "len %d: tokens differ from the legacy copy-adopt twin", len);
		if (filler) sslm_seq_release(filler);
	}
}

// ---- 7.2 [C5] -------------------------------------------------------------------------------------

// "No write inside the declared budget is refused" (4.3's construction): every origin offset
// 0..B-1 (a prefix of 32 + o tokens) x budgets {1, B-1, B, B+1, 512}: a holder resting ready at its
// origin emits exactly 1 + budget tokens (§3.4) and then SSLM_KV_BUDGET_EXCEEDED. And the named
// construction: a world-only adopt at position 1,000 with budget 512.
void Cell72() {
	const Fixture& fx = GetFixture("pkv_def");
	if (!fx.ok) return;
	for (int64_t o = 0; o < fx.B(); ++o)
		for (int32_t b : {1, 15, 16, 17, 512}) {
			const int32_t plen = static_cast<int32_t>(32 + o);
			PagePool pool(fx.model, static_cast<uint32_t>(fx.R(plen) + fx.R(b)));
			WorldAdopter w(fx, &pool.pool, plen, plen, b);
			if (!w.s) continue;
			PKV_CHECK_EQ(sslm_seq_adopt_prefix(w.s, w.px), SSLM_OK);
			sslm_status refusal = SSLM_OK;
			const std::vector<int32_t> got = RunToRefusal(fx, w.s, &refusal, b + 4);
			PKV_CHECK_MSG(static_cast<int64_t>(got.size()) == 1 + b && refusal == PKV_KV_BUDGET_EXCEEDED,
			              "origin offset %lld budget %d: %zu tokens then status %d (want %d then %d)",
			              static_cast<long long>(o), b, got.size(), static_cast<int>(refusal), 1 + b,
			              static_cast<int>(PKV_KV_BUDGET_EXCEEDED));
		}
	// kills: R = ceil(budget/B), which at origin 1,000 (offset 8) runs out of reserve 8 tokens early,
	// at position 1,504, with SSLM_KV_POOL_EXHAUSTED after 505 tokens
	PagePool pool(fx.model, static_cast<uint32_t>(fx.R(1000) + fx.R(512)));
	WorldAdopter w(fx, &pool.pool, 1000, 1000, 512);
	if (!w.s) return;
	PKV_CHECK_EQ(sslm_seq_adopt_prefix(w.s, w.px), SSLM_OK);
	sslm_status refusal = SSLM_OK;
	const std::vector<int32_t> got = RunToRefusal(fx, w.s, &refusal, 600);
	PKV_CHECK_EQ(got.size(), 513u);
	PKV_CHECK_EQ(refusal, PKV_KV_BUDGET_EXCEEDED);
}

// ---- 7.3 [C5] -------------------------------------------------------------------------------------

// "A pool sized by the old verb for N blocks admits N handles, including after a prefix release with
// adopters live". Legacy verbs only, so it runs on today's build and must pass there unchanged.
void Cell73() {
	const Fixture& fx = GetFixture("pkv_def");
	if (!fx.ok) return;
	const uint32_t N = 4;
	{
		// N sequences.
		LegacyPool pool(fx.model, N);
		sslm_status refusal = SSLM_OK;
		PKV_CHECK_EQ(CountLegacyCreates(fx.model, &pool.pool, &refusal), N);
		PKV_CHECK_EQ(refusal, SSLM_KV_POOL_EXHAUSTED);
	}
	{
		// N prefixes, the (N+1)th refused.
		LegacyPool pool(fx.model, N);
		std::vector<sslm_prefix> made;
		for (uint32_t i = 0; i < N; ++i) {
			sslm_prefix p = nullptr;
			PKV_CHECK_EQ(sslm_prefix_begin(fx.model, &pool.pool, &p), SSLM_OK);
			if (p) made.push_back(p);
		}
		sslm_prefix p = nullptr;
		PKV_CHECK_EQ(sslm_prefix_begin(fx.model, &pool.pool, &p), SSLM_KV_POOL_EXHAUSTED);
		for (sslm_prefix q : made) sslm_prefix_release(q);
	}
	// §3.5's construction: 4 blocks (1,024 pages), a prefix of 1,000 tokens (62 full pages and a
	// tail), 3 legacy adopters, then the prefix released. Today's build frees the prefix's block, so
	// the fourth handle is admitted. A sharing legacy adopt leaves the 62 full pages referenced by the
	// adopters, 194 pages free, and refuses it.
	LegacyPool pool(fx.model, N);
	sslm_prefix px = MakePrefix(fx, &pool.pool, Stream(51, 1000, fx.vocab), 0);
	std::vector<sslm_seq> adopters;
	for (uint32_t i = 0; i + 1 < N && px; ++i) {
		sslm_seq s = MakeLegacySeq(fx, &pool.pool);
		if (!s) break;
		PKV_CHECK_EQ(sslm_seq_adopt_prefix(s, px), SSLM_OK);
		adopters.push_back(s);
	}
	PKV_CHECK_EQ(CountLegacyCreates(fx.model, &pool.pool), 0);
	if (px) PKV_CHECK_EQ(sslm_prefix_release(px), SSLM_OK);
	// kills: legacy adopt that shares (exactly one block is free only if no page is shared)
	PKV_CHECK_EQ(CountLegacyCreates(fx.model, &pool.pool), 1);
	// The adopters still decode their copy: tokens equal a fresh copy-adopt twin.
	for (sslm_seq s : adopters) {
		std::vector<int32_t> t;
		DecodeN(fx, s, 20, &t);
		PKV_CHECK_MSG(t == LegacyTwinTokens(fx, 1000, 20), "an adopter's tokens changed when its prefix was released");
	}
	for (sslm_seq s : adopters) sslm_seq_release(s);
}

// ---- 7.5 [C5] -------------------------------------------------------------------------------------

struct WantSeq {
	uint32_t mode;
	int64_t origin;
	int32_t budget;
	int64_t limit, context_length;
	uint32_t priv, shared, reserve, materialized;
};

void ExpectSeqStats(sslm_seq s, const WantSeq& w, const char* what) {
	sslm_seq_kv_stats_out o{};
	o.struct_size = sizeof(o);
	o.reserved0 = 77;
	PKV_CHECK_EQ(sslm_seq_kv_stats(s, &o), SSLM_OK);
	PKV_CHECK_MSG(o.mode == w.mode && o.origin == w.origin && o.budget == w.budget && o.reserved0 == 0 &&
	                  o.limit == w.limit && o.context_length == w.context_length,
	              "%s: mode %u origin %lld budget %d reserved0 %d limit %lld L %lld; want %u %lld %d 0 %lld %lld", what,
	              o.mode, static_cast<long long>(o.origin), o.budget, o.reserved0, static_cast<long long>(o.limit),
	              static_cast<long long>(o.context_length), w.mode, static_cast<long long>(w.origin), w.budget,
	              static_cast<long long>(w.limit), static_cast<long long>(w.context_length));
	PKV_CHECK_MSG(o.mapped_private_pages == w.priv && o.shared_pages == w.shared && o.reserve_pages == w.reserve &&
	                  o.materialized_pages == w.materialized,
	              "%s: private %u shared %u reserve %u materialized %u; want %u %u %u %u", what, o.mapped_private_pages,
	              o.shared_pages, o.reserve_pages, o.materialized_pages, w.priv, w.shared, w.reserve, w.materialized);
}

void ExpectPoolStats(sslm_kv_pool pool, uint32_t page_count, uint32_t free_pages, uint32_t shared, const char* what) {
	sslm_kv_pool_stats_out o{};
	o.struct_size = sizeof(o);
	PKV_CHECK_EQ(sslm_kv_pool_stats(pool, &o), SSLM_OK);
	PKV_CHECK_MSG(o.page_count == page_count && o.free_pages == free_pages && o.shared_pages == shared &&
	                  o.page_positions == 16,
	              "%s: pool page_count %u free %u shared %u B %lld; want %u %u %u 16", what, o.page_count, o.free_pages,
	              o.shared_pages, static_cast<long long>(o.page_positions), page_count, free_pages, shared);
}

// 7.5: the stats verbs report the page counts a hand-built cohort geometry predicts by §3.4's
// formula, written here. The verbs' own sentences: free_pages; shared_pages as the distinct pages
// some live holder maps as shared, by provenance (§3.3), so a released world's pages stay shared
// while personas map them; per holder mode, origin, budget, limit, context_length, and the
// reservation split into private mapped pages (materialized ones included: §3.9 "the reservation is
// the holder's reserve plus its private mapped pages", which is R + E after a private restore),
// shared pages, reserve pages and materialized pages. Diagnostic only: no verdict elsewhere reads them.
void Cell75() {
	const Fixture& fx = GetFixture("pkv_def");
	if (!fx.ok) return;
	const uint32_t P = 400;
	const int32_t b = 512;
	const uint32_t R = static_cast<uint32_t>(fx.R(b));  // 33
	PagePool pool(fx.model, P);
	PKV_CHECK_EQ(pool.status, SSLM_OK);
	if (pool.status != SSLM_OK) return;
	ExpectPoolStats(pool.pool, P, P, 0, "empty");
	sslm_prefix world = MakePrefix(fx, &pool.pool, WorldTokens(fx.vocab), kWorldLen);
	ExpectPoolStats(pool.pool, P, P - 63, 0, "world frozen");
	sslm_prefix persona = world ? MakeChild(fx, world, PersonaTokens(fx.vocab), kPersonaLen) : nullptr;
	// kills: shared_pages read as refcount >= 2 (the same 62 here), or counting the world's tail page
	ExpectPoolStats(pool.pool, P, P - 76, 62, "persona frozen");
	if (!persona) return;
	std::vector<sslm_seq> seqs;
	for (int i = 0; i < 3; ++i) {
		sslm_seq s = MakeBudgetSeq(fx, &pool.pool, b);
		if (!s) return;
		PKV_CHECK_EQ(sslm_seq_adopt_prefix(s, persona), SSLM_OK);
		seqs.push_back(s);
	}
	ExpectPoolStats(pool.pool, P, P - 76 - 3 * R, 75, "three adopters");
	ExpectSeqStats(seqs[0], {1, 1200, b, 1712, 1200, 0, 75, R, 0}, "adopter");
	std::vector<int32_t> t;
	DecodeN(fx, seqs[0], 20, &t);  // the ready token, then rows 1,200..1,218: pages 75 and 76
	ExpectSeqStats(seqs[0], {1, 1200, b, 1712, 1219, 2, 75, R - 2, 0}, "adopter after 20 tokens");
	PKV_CHECK_EQ(sslm_prefix_release(world), SSLM_OK);
	// kills: shared_pages by refcount (the world's 62 pages drop to refcount 4, still shared), or a
	// prefix release that frees pages its sharers map
	ExpectPoolStats(pool.pool, P, P - 75 - 3 * R, 75, "world released");
	PKV_CHECK_EQ(sslm_seq_reset(seqs[1]), SSLM_OK);
	ExpectSeqStats(seqs[1], {1, 0, b, b, 0, 0, 0, R, 0}, "adopter after reset");
	const std::vector<uint8_t> blob = Save(seqs[0]);
	sslm_seq r1 = nullptr, r2 = nullptr;
	PKV_CHECK_EQ(sslm_seq_restore(fx.model, &pool.pool, blob.data(), blob.size(), &r1), SSLM_OK);
	// E = min(floor(1200/16), ceil(4096/16) - R(512)) = 75 materialized pages.
	if (r1) ExpectSeqStats(r1, {1, 1200, b, 1712, 1219, 75 + 2, 0, R - 2, 75}, "restored without a handle");
	ExpectPoolStats(pool.pool, P, P - 75 - 3 * R - (R + 75), 75, "after the private restore");
	uint32_t shared_out = 0;
	PKV_CHECK_EQ(sslm_seq_restore_shared(fx.model, &pool.pool, blob.data(), blob.size(), persona, &r2, &shared_out), SSLM_OK);
	if (r2) ExpectSeqStats(r2, {1, 1200, b, 1712, 1219, 2, 75, R - 2, 0}, "restored with the persona handle");
	ExpectPoolStats(pool.pool, P, P - 75 - 3 * R - (R + 75) - R, 75, "after the shared restore");
	if (r1) {
		PKV_CHECK_EQ(sslm_seq_reset(r1), SSLM_OK);
		ExpectSeqStats(r1, {1, 0, b, b, 0, 0, 0, R, 0}, "private restore after reset");
		ExpectPoolStats(pool.pool, P, P - 75 - 3 * R - R - R, 75, "materialized pages returned");
	}
	if (r1) sslm_seq_release(r1);
	if (r2) sslm_seq_release(r2);
	for (sslm_seq s : seqs) sslm_seq_release(s);
	sslm_prefix_release(persona);
	ExpectPoolStats(pool.pool, P, P, 0, "torn down");
	// A whole_reserve holder in a page pool: mode 0, budget = cap, its reservation one old block.
	PagePool lp(fx.model, static_cast<uint32_t>(fx.CapPages()));
	sslm_seq l = MakeLegacySeq(fx, &lp.pool);
	if (!l) return;
	const int64_t cap = fx.geo.context_cap;
	const uint32_t cp = static_cast<uint32_t>(fx.CapPages());
	ExpectSeqStats(l, {0, 0, static_cast<int32_t>(cap), cap, 0, 0, 0, cp, 0}, "whole_reserve holder");
	PKV_CHECK_EQ(PrefillAll(fx.model, l, Stream(5, 40, fx.vocab), 16), SSLM_OK);
	ExpectSeqStats(l, {0, 0, static_cast<int32_t>(cap), cap, 40, 3, 0, cp - 3, 0}, "whole_reserve holder after 40");
	ExpectPoolStats(lp.pool, cp, 0, 0, "one legacy holder");
	sslm_seq_release(l);
}

// ---- 7.6 [C5] -------------------------------------------------------------------------------------

// 7.6's budget part: sslm_stats' kv_blocks_resident stays the constant 1 (G14) for every budget-mode
// holder: fresh, written across pages, an adopter of a nested prefix, restored with and without a
// handle, reset; and for a whole_reserve holder in a page pool.
void Cell76Budget() {
	const Fixture& fx = GetFixture("pkv_def");
	if (!fx.ok) return;
	auto resident = [&](sslm_seq s, const char* what) {
		sslm_stats_out o{};
		PKV_CHECK_EQ(sslm_stats(fx.model, s, &o), SSLM_OK);
		// kills: kv_blocks_resident reporting pages mapped, or the reservation, or 0 for a shared holder
		PKV_CHECK_MSG(o.kv_blocks_resident == 1, "%s: kv_blocks_resident %d", what, o.kv_blocks_resident);
	};
	PagePool pool(fx.model, 700);
	Handles h;
	const Chain c = BuildChain(fx, &pool.pool, &h);
	if (!c.persona) return;
	sslm_seq s = MakeBudgetSeq(fx, &pool.pool, 512);
	if (!s) return;
	h.seqs.push_back(s);
	resident(s, "fresh budget holder");
	PKV_CHECK_EQ(PrefillAll(fx.model, s, Stream(5, 40, fx.vocab), 16), SSLM_OK);
	resident(s, "budget holder after 40 rows");
	PKV_CHECK_EQ(sslm_seq_adopt_prefix(s, c.persona), SSLM_OK);
	resident(s, "persona adopter");
	std::vector<int32_t> t;
	DecodeN(fx, s, 10, &t);
	const std::vector<uint8_t> blob = Save(s);
	sslm_seq r1 = nullptr, r2 = nullptr;
	PKV_CHECK_EQ(sslm_seq_restore(fx.model, &pool.pool, blob.data(), blob.size(), &r1), SSLM_OK);
	if (r1) h.seqs.push_back(r1), resident(r1, "restored without a handle");
	PKV_CHECK_EQ(sslm_seq_restore_shared(fx.model, &pool.pool, blob.data(), blob.size(), c.persona, &r2, nullptr), SSLM_OK);
	if (r2) h.seqs.push_back(r2), resident(r2, "restored with the persona handle");
	PKV_CHECK_EQ(sslm_seq_reset(s), SSLM_OK);
	resident(s, "reset budget holder");
	sslm_seq l = MakeLegacySeq(fx, &pool.pool);
	if (l) h.seqs.push_back(l), resident(l, "whole_reserve holder in a page pool");
}

// ---- 7.8 [C5] -------------------------------------------------------------------------------------

// 7.8's budget-verb part: each new verb's header behaviour sentences (§3.6's verb table, the header
// to be) have a failing cell where cheap. One block per verb, each named for its sentence.
void Cell78Budget() {
	const Fixture& fx = GetFixture("pkv_def");
	const Fixture& odd = GetFixture("pkv_odd");
	const Fixture& big = GetFixture("pkv_32k");
	if (!fx.ok || !odd.ok || !big.ok) return;
	const int32_t cap = static_cast<int32_t>(fx.geo.context_cap);

	// sslm_kv_page_positions: "B (§3.1); 0 on a null model".
	PKV_CHECK_EQ(sslm_kv_page_positions(nullptr), 0);
	PKV_CHECK_EQ(sslm_kv_page_positions(fx.model), 16);
	PKV_CHECK_EQ(sslm_kv_page_positions(big.model), 16);
	PKV_CHECK_EQ(sslm_kv_page_positions(odd.model), 4100);  // kills: B = kv_block_size when it does not divide cap

	// sslm_kv_page_size: "page_bytes; 0 on null/overflow". page_bytes = L * 2 * H_kv * B * D, written here.
	PKV_CHECK_EQ(sslm_kv_page_size(nullptr), 0u);
	PKV_CHECK_EQ(sslm_kv_page_size(fx.model), fx.PageBytes());
	PKV_CHECK_EQ(sslm_kv_page_size(odd.model), odd.PageBytes());  // one page is the whole block there
	PKV_CHECK_EQ(sslm_kv_page_size(odd.model), sslm_kv_block_size(odd.model));

	// sslm_kv_pages_for_budget: "R(budget); 0 for a budget outside [1, cap]".
	for (int32_t b : {1, 15, 16, 17, 512, 1000, 3200, 4079, 4080, 4081, 4096})
		PKV_CHECK_MSG(sslm_kv_pages_for_budget(fx.model, b) == static_cast<size_t>(fx.R(b)), "R(%d) = %zu, want %lld", b,
		              sslm_kv_pages_for_budget(fx.model, b), static_cast<long long>(fx.R(b)));
	// kills: the clamp missing (R(4096) would be 257), the +1 missing (R(512) would be 32)
	for (int32_t b : {0, -1, cap + 1, INT32_MAX, INT32_MIN})
		PKV_CHECK_MSG(sslm_kv_pages_for_budget(fx.model, b) == 0, "pages_for_budget(%d) is not 0", b);
	PKV_CHECK_EQ(sslm_kv_pages_for_budget(nullptr, 16), 0u);
	PKV_CHECK_EQ(sslm_kv_pages_for_budget(odd.model, 1), 1u);  // B = cap: ceil(cap/B) = 1 clamps

	// sslm_kv_page_pool_overhead_size: "same saturating convention as the block verb": 0 for a null
	// model, never decreasing in page_count, and never wrapping to a smaller value.
	PKV_CHECK_EQ(sslm_kv_page_pool_overhead_size(nullptr, 4), 0u);
	size_t prev = 0;
	for (uint32_t n : {1u, 2u, 256u, 65536u, 0x7FFFFFFFu, 0xFFFFFFFFu}) {
		const size_t o = sslm_kv_page_pool_overhead_size(fx.model, n);
		PKV_CHECK_MSG(o >= prev && o > 0, "overhead(%u) = %zu after %zu", n, o, prev);
		prev = o;
	}

	// sslm_kv_page_pool_create: "buf_size >= page_count * page_bytes + overhead; same alignment and
	// ordering of refusals as sslm_kv_pool_create". The exact size is admitted, one byte less is not,
	// and each defect (and pair of defects) gets the status the block verb gives the same defects on
	// the same bytes (one block = ceil(cap/B) pages).
	{
		const uint32_t pages = static_cast<uint32_t>(fx.CapPages());
		const size_t need = pages * sslm_kv_page_size(fx.model) + sslm_kv_page_pool_overhead_size(fx.model, pages);
		const size_t need_block = sslm_kv_block_size(fx.model) + sslm_kv_pool_overhead_size(fx.model, 1);
		AlignedBuf mem(std::max(need, need_block) + 2 * SSLM_ABI_ALIGNMENT_BYTES);
		sslm_kv_pool p = nullptr;
		PKV_CHECK_EQ(sslm_kv_page_pool_create(fx.model, mem.p, need, pages, &p), SSLM_OK);
		if (p) sslm_kv_pool_destroy(p), p = nullptr;
		PKV_CHECK_EQ(sslm_kv_page_pool_create(fx.model, mem.p, need - 1, pages, &p), SSLM_BUFFER_TOO_SMALL);
		struct Case {
			const char* what;
			sslm_model m;
			bool null_buf, misaligned, short_by_one, zero_count, null_out;
		};
		const Case cases[] = {
		    {"misaligned", fx.model, false, true, false, false, false},
		    {"misaligned and short", fx.model, false, true, true, false, false},
		    {"zero count", fx.model, false, false, false, true, false},
		    {"zero count and misaligned", fx.model, false, true, false, true, false},
		    {"null buffer", fx.model, true, false, false, false, false},
		    {"null model", nullptr, false, false, false, false, false},
		    {"null out", fx.model, false, false, false, false, true},
		};
		for (const Case& c : cases) {
			uint8_t* buf = c.null_buf ? nullptr : mem.p + (c.misaligned ? 1 : 0);
			sslm_kv_pool a = nullptr, b2 = nullptr;
			const sslm_status page_st = sslm_kv_page_pool_create(c.m, buf, need - (c.short_by_one ? 1 : 0),
			                                                     c.zero_count ? 0u : pages, c.null_out ? nullptr : &a);
			if (a) sslm_kv_pool_destroy(a);
			const sslm_status block_st = sslm_kv_pool_create(c.m, buf, need_block - (c.short_by_one ? 1 : 0),
			                                                 c.zero_count ? 0u : 1u, c.null_out ? nullptr : &b2);
			if (b2) sslm_kv_pool_destroy(b2);
			PKV_CHECK_MSG(page_st == block_st && page_st != SSLM_OK, "%s: page pool %d, block pool %d", c.what,
			              static_cast<int>(page_st), static_cast<int>(block_st));
		}
	}

	// sslm_seq_create_budgeted: "budget mode; 1 <= budget <= cap else SSLM_INVALID_ARGUMENT". A refused
	// create draws nothing: a pool of exactly R(cap) pages still admits a create of budget cap.
	{
		PagePool pool(fx.model, static_cast<uint32_t>(fx.CapPages()));
		for (int32_t b : {0, -1, cap + 1, INT32_MAX}) {
			sslm_seq s = OutSentinel<sslm_seq>();  // non-null; the verb must null it on the refusal
			// kills: a refusal that returns before writing *out
			PKV_CHECK_MSG(sslm_seq_create_budgeted(fx.model, &pool.pool, b, &s) == SSLM_INVALID_ARGUMENT && !s,
			              "create_budgeted(%d) not refused", b);
			if (s && s != OutSentinel<sslm_seq>()) sslm_seq_release(s);
		}
		sslm_seq s = MakeBudgetSeq(fx, &pool.pool, cap);  // kills: a refusal that drew before validating
		if (s) sslm_seq_release(s);
	}

	// sslm_prefix_begin_budgeted: "budget-mode prefix": it draws R(budget), not a block (a pool of
	// R(budget) pages admits it), and a budget-mode prefix's freeze keeps exactly ceil(len/B) pages.
	{
		auto build = [&]() {
			auto st = std::make_unique<PageState>(fx.model, static_cast<uint32_t>(fx.R(1000) + 40));
			sslm_prefix p = MakePrefix(fx, st->Pool(), Stream(51, 1000, fx.vocab), 1000);
			if (p) st->h.prefixes.push_back(p);
			return std::unique_ptr<ProbeState>(std::move(st));
		};
		// kills: begin_budgeted reserving ceil(cap/B) (the pool is smaller than a block), or freeze
		// keeping its reserve page
		ProbeExactlyFree(fx, build, fx.R(1000) + 40 - 63);
	}

	// sslm_prefix_begin_from: "parent must be frozen (SSLM_INVALID_ARGUMENT otherwise); draws
	// R(budget) from the parent's pool".
	{
		PagePool pool(fx.model, 200);
		sslm_prefix open = nullptr;
		PKV_CHECK_EQ(sslm_prefix_begin_budgeted(fx.model, &pool.pool, 64, &open), SSLM_OK);
		if (open) {
			PKV_CHECK_EQ(PrefixPrefillAll(fx.model, open, Stream(5, 20, fx.vocab), 8), SSLM_OK);
			sslm_prefix child = OutSentinel<sslm_prefix>();  // non-null; the verb must null it on the refusal
			// kills: a refusal that returns before writing *out
			PKV_CHECK_MSG(sslm_prefix_begin_from(open, 32, &child) == SSLM_INVALID_ARGUMENT && !child,
			              "begin_from an unfrozen parent was not refused");
			if (child && child != OutSentinel<sslm_prefix>()) sslm_prefix_release(child);
			sslm_prefix_release(open);
		}
		auto build = [&]() {
			auto st = std::make_unique<PageState>(fx.model, 200);
			sslm_prefix w = MakePrefix(fx, st->Pool(), WorldTokens(fx.vocab), kWorldLen);
			if (w) st->h.prefixes.push_back(w);
			sslm_prefix child = nullptr;
			if (w && sslm_prefix_begin_from(w, kPersonaLen, &child) == SSLM_OK) st->h.prefixes.push_back(child);
			return std::unique_ptr<ProbeState>(std::move(st));
		};
		// kills: begin_from drawing a block (refused in this pool), drawing R(budget) + 1 for the
		// tail it copies (the tail page is one of the R(200) = 14), or drawing from another pool
		ProbeExactlyFree(fx, build, 200 - 63 - fx.R(kPersonaLen));
	}

	// sslm_seq_restore_shared: "a null handle, or one whose identity does not match, gives a private
	// restore with *out_shared_pages = 0 and SSLM_OK. out_shared_pages may be null. The existing
	// sslm_seq_restore ... behaves as this verb with a null prefix."
	{
		PagePool pool(fx.model, 600);
		Handles h;
		const Chain c = BuildChain(fx, &pool.pool, &h);
		sslm_seq s = c.persona ? MakeBudgetSeq(fx, &pool.pool, 512) : nullptr;
		if (!s) return;
		h.seqs.push_back(s);
		PKV_CHECK_EQ(sslm_seq_adopt_prefix(s, c.persona), SSLM_OK);
		std::vector<int32_t> t;
		DecodeN(fx, s, 12, &t);
		const std::vector<uint8_t> blob = Save(s);
		sslm_seq a = nullptr, b = nullptr, m = nullptr, n = nullptr;
		uint32_t shared = 99;
		PKV_CHECK_EQ(sslm_seq_restore_shared(fx.model, &pool.pool, blob.data(), blob.size(), nullptr, &a, &shared), SSLM_OK);
		PKV_CHECK_EQ(shared, 0u);
		PKV_CHECK_EQ(sslm_seq_restore(fx.model, &pool.pool, blob.data(), blob.size(), &b), SSLM_OK);
		shared = 99;
		// A mismatched handle (the world: context_length 1,000, not the blob's origin 1,200).
		PKV_CHECK_EQ(sslm_seq_restore_shared(fx.model, &pool.pool, blob.data(), blob.size(), c.world, &m, &shared), SSLM_OK);
		PKV_CHECK_EQ(shared, 0u);
		// A matching handle with a null out_shared_pages.
		PKV_CHECK_EQ(sslm_seq_restore_shared(fx.model, &pool.pool, blob.data(), blob.size(), c.persona, &n, nullptr), SSLM_OK);
		for (sslm_seq r : {a, b, m, n})
			if (r) h.seqs.push_back(r);
		if (a && b && m && n) {
			// kills: sslm_seq_restore diverging from restore_shared(null) (a different budget, origin or rows)
			PKV_CHECK_MSG(Save(a) == Save(b) && Save(m) == Save(b) && Save(n) == Save(b), "the four restores re-save differently");
			std::vector<int32_t> ta, tb, tn;
			DecodeN(fx, a, 8, &ta);
			DecodeN(fx, b, 8, &tb);
			DecodeN(fx, n, 8, &tn);
			PKV_CHECK(ta == tb && tn == tb);
		}
	}
}

// ---- 7.10 [C5] ------------------------------------------------------------------------------------

// 7.10's budget part: decode a budget sequence with a sized workspace across a page boundary. The
// step that maps a new page from the holder's reserve takes the pool mutex 0 times (the C4 seam
// counter) and allocates no more than the step that maps nothing (the counting operator new above);
// a prefill that maps several pages takes the pool mutex 0 times too.
void Cell710Budget() {
	const Fixture& fx = GetFixture("pkv_def");
	if (!fx.ok) return;
	const sslm_config cfg{1, 64, static_cast<int32_t>(fx.geo.layers), 0u};
	AlignedBuf wsmem(sslm_workspace_size(fx.model, &cfg));
	sslm_workspace ws = nullptr;
	PKV_CHECK_EQ(sslm_workspace_create(fx.model, &cfg, wsmem.p, wsmem.n, &ws), SSLM_OK);
	PagePool pool(fx.model, static_cast<uint32_t>(fx.R(256)));
	sslm_seq s = MakeBudgetSeq(fx, &pool.pool, 256);
	if (!s || !ws) return;
	const uint64_t m0 = sslm_pkv_test_only_pool_mutex_acquisitions(pool.pool);
	const std::vector<int32_t> prompt = Stream(9, 31, fx.vocab);
	int32_t consumed = 0;
	PKV_CHECK_EQ(sslm_prefill(fx.model, s, prompt.data(), 31, 64, SSLM_SPAN_PROMPT, ws, &consumed), SSLM_OK);
	PKV_CHECK_EQ(consumed, 31);
	// kills: prefill mapping its two pages through the pool
	PKV_CHECK_EQ(sslm_pkv_test_only_pool_mutex_acquisitions(pool.pool) - m0, 0u);
	sslm_decode_params p{};
	p.layer_budget = static_cast<int32_t>(fx.geo.layers);
	sslm_seq batch[1] = {s};
	auto step = [&](long long* news, uint64_t* locks) {
		int32_t tok = -1;
		const uint64_t l0 = sslm_pkv_test_only_pool_mutex_acquisitions(pool.pool);
		const long long n0 = g_new_calls.load();
		const sslm_status st = sslm_decode_step(fx.model, batch, 1, &p, ws, &tok);
		*news = g_new_calls.load() - n0;
		*locks = sslm_pkv_test_only_pool_mutex_acquisitions(pool.pool) - l0;
		PKV_CHECK_MSG(st == SSLM_OK && tok >= 0, "decode step: status %d token %d", static_cast<int>(st), tok);
	};
	long long news_ready = 0, news_inside = 0, news_cross = 0;
	uint64_t locks_ready = 0, locks_inside = 0, locks_cross = 0;
	step(&news_ready, &locks_ready);    // the ready token: no row
	step(&news_inside, &locks_inside);  // row 31: page 1, already mapped
	step(&news_cross, &locks_cross);    // row 32: maps page 2 from the reserve
	std::printf("  7.10: allocations per step: ready %lld, inside a page %lld, crossing %lld\n", news_ready,
	            news_inside, news_cross);
	// kills: the mapping call taking the pool mutex (a map through the pool)
	PKV_CHECK_EQ(locks_inside, 0u);
	PKV_CHECK_EQ(locks_cross, 0u);
	// kills: a mapping call that allocates (a table grown on demand, a pool-side list)
	PKV_CHECK_MSG(news_cross == news_inside, "the page-crossing step allocates %lld times, the step inside a page %lld",
	              news_cross, news_inside);
	PKV_CHECK_EQ(news_ready, 0);  // the ready path allocates nothing (dim7_contract_red.cpp's half 1)
	sslm_seq_release(s);
	sslm_workspace_destroy(ws);
}

// ---- 7.11 on a budget holder (RB16-CELLS) ------------------------------------------------------------

// The exact SSB6 size is 172 + residual + 4 * history + 8 + L' * bytes_per_token at L' in
// {0, 1, B, 1,000}, on a budget holder's save (no legacy holder writes SSB6). Residual is
// hidden_size bytes (written whenever hidden_size > 0, so a zero-token holder saves no K/V bytes,
// not a header-only blob); history is the blob's own history_count field. Also at a mid-token save
// (L' = context_length + 1) and on an adopter (L' = its origin).
void Cell711Budget() {
	const Fixture& fx = GetFixture("pkv_def");
	if (!fx.ok) return;
	const size_t hidden = HiddenSize(fx), bpt = fx.BytesPerToken();
	auto expect_size = [&](sslm_seq s, int64_t Lp, const char* what) {
		const std::vector<uint8_t> b = Save(s);
		const BlobView v = ParseBlobWithHidden(b, hidden);
		PKV_CHECK_MSG(IsMagic(b, "SSB6") && v.ok, "%s: not an SSB6 blob", what);
		const size_t want = kSsb6Header + hidden + 4 * static_cast<size_t>(v.history_count) + 8 + static_cast<size_t>(Lp) * bpt;
		// kills: the save gathering [0, cap) (the size then carries cap * bytes_per_token)
		PKV_CHECK_MSG(b.size() == want, "%s: SSB6 size %zu, want %zu (L' = %lld)", what, b.size(), want,
		              static_cast<long long>(Lp));
		PKV_CHECK_EQ(static_cast<int64_t>(v.kv_positions), Lp);
	};
	PagePool pool(fx.model, 200);
	for (int64_t Lp : {int64_t{0}, int64_t{1}, fx.B(), int64_t{1000}}) {
		sslm_seq s = MakeBudgetSeq(fx, &pool.pool, 1024);
		if (!s) return;
		if (Lp > 0) PKV_CHECK_EQ(PrefillAll(fx.model, s, Stream(13, static_cast<int32_t>(Lp), fx.vocab), 64), SSLM_OK);
		const std::string what = "L' = " + std::to_string(Lp);
		expect_size(s, Lp, what.c_str());
		if (Lp > 0) {
			std::vector<int32_t> t;
			DecodeN(fx, s, 1, &t);
			EnterMidToken(fx, s);
			expect_size(s, Lp + 1, (what + " + 1, mid-token").c_str());
		}
		sslm_seq_release(s);
	}
	WorldAdopter w(fx, &pool.pool, 1000, 1000, 512);
	if (w.s && sslm_seq_adopt_prefix(w.s, w.px) == SSLM_OK) expect_size(w.s, 1000, "adopter at origin 1,000");
}

// ---- 7.12 [C5] ------------------------------------------------------------------------------------

// 7.12's budget part, the E clamp: a budget-3,200 sequence at origin 1,000, restored without a
// handle, draws R(3200) + E with E = min(floor(1000/16), ceil(4096/16) - R(3200)) = min(62, 55) = 55
// materialized pages, not 62: a pool of 256 = 201 + 55 pages admits the restore and 255 refuses it.
// After its reset, the fill probe shows a reservation of exactly R(3200) = 201 (55 pages free).
void Cell712Budget() {
	const Fixture& fx = GetFixture("pkv_def");
	if (!fx.ok) return;
	PKV_CHECK_EQ(fx.R(3200), 201);
	std::vector<uint8_t> blob;
	std::vector<int32_t> cont;
	{
		PagePool src(fx.model, static_cast<uint32_t>(fx.R(1000) + fx.R(3200)));
		WorldAdopter w(fx, &src.pool, 1000, 1000, 3200);
		if (!w.s || sslm_seq_adopt_prefix(w.s, w.px) != SSLM_OK) {
			PKV_CHECK_MSG(false, "7.12: the source holder could not be built");
			return;
		}
		std::vector<int32_t> t;
		DecodeN(fx, w.s, 40, &t);
		blob = Save(w.s);
		DecodeN(fx, w.s, 16, &cont);
	}
	const int64_t E = std::min<int64_t>(1000 / fx.B(), fx.CapPages() - fx.R(3200));
	PKV_CHECK_EQ(E, 55);
	const uint32_t need = static_cast<uint32_t>(fx.R(3200) + E);
	{
		PagePool pool(fx.model, need);
		sslm_seq r = nullptr;
		// kills: the unclamped E (62 pages: 263 needed, refused here)
		PKV_CHECK_EQ(sslm_seq_restore(fx.model, &pool.pool, blob.data(), blob.size(), &r), SSLM_OK);
		if (r) {
			std::vector<int32_t> t;
			DecodeN(fx, r, 16, &t);
			PKV_CHECK_MSG(t == cont, "the restored holder's tokens differ from the unsaved original's");
			sslm_seq_release(r);
		}
	}
	{
		PagePool pool(fx.model, need - 1);
		sslm_seq r = OutSentinel<sslm_seq>();  // non-null; the verb must null it on the refusal
		// kills: an E below 55 (a restore that does not hold the prefix span privately)
		PKV_CHECK_EQ(sslm_seq_restore(fx.model, &pool.pool, blob.data(), blob.size(), &r), SSLM_KV_POOL_EXHAUSTED);
		PKV_CHECK(r == nullptr);  // kills: a refusal that returns before writing *out
		if (r && r != OutSentinel<sslm_seq>()) sslm_seq_release(r);
	}
	auto build = [&]() {
		auto st = std::make_unique<PageState>(fx.model, need);
		sslm_seq r = nullptr;
		if (sslm_seq_restore(fx.model, st->Pool(), blob.data(), blob.size(), &r) == SSLM_OK) {
			st->h.seqs.push_back(r);
			PKV_CHECK_EQ(sslm_seq_reset(r), SSLM_OK);
		}
		return std::unique_ptr<ProbeState>(std::move(st));
	};
	// kills: materialized pages sent to the reserve at reset (0 free), or a reset keeping them mapped
	ProbeExactlyFree(fx, build, E);
}

// ---- 7.13 [C5] ------------------------------------------------------------------------------------

// 9.9's fixed case (the construction 7.13 runs under the sentinel): a budget-512 adopter of the
// persona (origin 1,200) decodes 300 tokens, is saved, and restored twice, with and without the
// persona handle. Each twin continues to its refusal with the unsaved original's 213 tokens (300 +
// 213 = 1 + 512), and after a reset and re-adopt runs to refusal again with 513.
void Fixed99(const Fixture& fx) {
	PagePool pool(fx.model, 600);
	Handles h;
	const Chain c = BuildChain(fx, &pool.pool, &h);
	sslm_seq s = c.persona ? MakeBudgetSeq(fx, &pool.pool, 512) : nullptr;
	if (!s) return;
	h.seqs.push_back(s);
	PKV_CHECK_EQ(sslm_seq_adopt_prefix(s, c.persona), SSLM_OK);
	std::vector<int32_t> t;
	DecodeN(fx, s, 300, &t);
	const std::vector<uint8_t> blob = Save(s);
	sslm_status st = SSLM_OK;
	const std::vector<int32_t> rest = RunToRefusal(fx, s, &st, 600);
	PKV_CHECK_EQ(rest.size(), 213u);
	for (int with_handle = 0; with_handle <= 1; ++with_handle) {
		sslm_seq r = nullptr;
		PKV_CHECK_EQ(sslm_seq_restore_shared(fx.model, &pool.pool, blob.data(), blob.size(),
		                                     with_handle ? c.persona : nullptr, &r, nullptr),
		             SSLM_OK);
		if (!r) continue;
		h.seqs.push_back(r);
		const std::vector<int32_t> got = RunToRefusal(fx, r, &st, 600);
		PKV_CHECK_MSG(got == rest && st == PKV_KV_BUDGET_EXCEEDED, "9.9 twin (%s handle): %zu tokens, status %d",
		              with_handle ? "with" : "without", got.size(), static_cast<int>(st));
		PKV_CHECK_EQ(sslm_seq_reset(r), SSLM_OK);
		PKV_CHECK_EQ(sslm_seq_adopt_prefix(r, c.persona), SSLM_OK);
		PKV_CHECK_EQ(RunToRefusal(fx, r, &st, 600).size(), 513u);
	}
}

// 9.10's shorter-handle (e) case: the cohort blob (origin 1,200, budget 512) restored with the world
// handle (context_length 1,000) into a pool of exactly the world's 63 pages + R(512) + E(75). (e)
// rejects the handle, so the restore is private: SSLM_OK, and a 2-page create is refused straight
// after it. A build that evaluates (f) first reads the world's table to floor(1200/16) = 75 entries,
// 12 above its mapped 63, and traps on the sentinel.
void ShorterHandle910(const Fixture& fx) {
	std::vector<uint8_t> blob;
	std::vector<int32_t> cont;
	{
		PagePool src(fx.model, 600);
		Handles h;
		const Chain c = BuildChain(fx, &src.pool, &h);
		sslm_seq s = c.persona ? MakeBudgetSeq(fx, &src.pool, 512) : nullptr;
		if (!s) return;
		h.seqs.push_back(s);
		PKV_CHECK_EQ(sslm_seq_adopt_prefix(s, c.persona), SSLM_OK);
		std::vector<int32_t> t;
		DecodeN(fx, s, 30, &t);
		blob = Save(s);
		DecodeN(fx, s, 12, &cont);
	}
	const int64_t E = std::min<int64_t>(kPersonaOrigin / fx.B(), fx.CapPages() - fx.R(512));
	PagePool dst(fx.model, static_cast<uint32_t>(PrefixPages(fx, kWorldLen) + fx.R(512) + E));  // the world keeps 63
	sslm_prefix world = MakePrefix(fx, &dst.pool, WorldTokens(fx.vocab), kWorldLen);
	if (!world) return;
	sslm_seq r = nullptr;
	PKV_CHECK_EQ(sslm_seq_restore_shared(fx.model, &dst.pool, blob.data(), blob.size(), world, &r, nullptr), SSLM_OK);
	sslm_seq extra = nullptr;
	PKV_CHECK_EQ(sslm_seq_create_budgeted(fx.model, &dst.pool, static_cast<int32_t>(fx.B()), &extra), SSLM_KV_POOL_EXHAUSTED);
	if (extra) sslm_seq_release(extra);
	if (r) {
		std::vector<int32_t> t;
		DecodeN(fx, r, 12, &t);
		PKV_CHECK_MSG(t == cont, "9.10 (e) shorter handle: tokens after restore differ from the original's");
		sslm_seq_release(r);
	}
	sslm_prefix_release(world);
}

// 7.13's budget-ABI part: "table entries at or above mapped are never read". 1.1's budget half,
// 1.10, 9.9 and 9.10 run with the table sentinel on (every new table filled with 0xFFFFFFFF, every
// unmapped entry rewritten to it, the address path trapping on it), each in a forked child graded by
// its exit status. Mutant: condition (f) reads the handle's table to floor(origin/B) before
// condition (e) has rejected a shorter handle (9.10's world at 1,000 against origin 1,200), and traps.
void Cell713Budget() {
	const Fixture& fx = GetFixture("pkv_def");
	if (!fx.ok) return;
	(void)Reference("v1.11.0", fx);  // parsed in the parent, so each child inherits it
	struct Leg {
		const char* what;
		std::function<void()> body;
	};
	const Leg legs[] = {
	    {"1.1's budget half",
	     [&] { ExpectMatchesReference("v1.11.0", fx, "lifecycle", BudgetLifecycle(fx), Compare::kTokensAndRows); }},
	    {"1.10 (the budget-mode adopt census)", [&] { PKV_CHECK_EQ(RunBudgetCensus(fx).mismatches, 0); }},
	    {"9.9's fixed case", [&] { Fixed99(fx); }},
	    {"9.10's shorter-handle (e) case", [&] { ShorterHandle910(fx); }},
	};
	for (const Leg& leg : legs) {
		const std::string how = RunUnderSentinel(leg.body);
		// kills: a read of a table entry at or above mapped (the save gather, the address path, or
		// restore_shared's (f) before (e)), which traps on the sentinel and kills the child
		PKV_CHECK_MSG(how.empty(), "%s under the table sentinel: %s", leg.what, how.c_str());
	}
}

// ---- 7.14 [C5] ------------------------------------------------------------------------------------

// 7.14: mixed-pool admission, priced. A pool made by sslm_kv_pool_create for 2 blocks (512 pages); a
// 4,000-token prefix (250 full pages), once legacy and once budget-mode, shared by one
// sslm_seq_create_budgeted(cap) holder and then released. The pool then holds R(cap) + 250 = 506
// pages by §3.5's accounting: a legacy create is refused, and the fill probe shows exactly 6 free.
// After the holder's reset the legacy create is admitted. Control: a legacy holder in the same
// construction copies, and the legacy create is admitted with the prefix released.
void Cell714() {
	const Fixture& fx = GetFixture("pkv_def");
	if (!fx.ok) return;
	const int32_t cap = static_cast<int32_t>(fx.geo.context_cap);
	const std::vector<int32_t> toks = Stream(77, 4000, fx.vocab);
	PKV_CHECK_EQ(512 - (fx.R(cap) + 250), 6);
	for (int prefix_budget : {0, 1}) {
		const char* mode = prefix_budget ? "budget prefix" : "legacy prefix";
		auto build = [&](bool budget_holder, bool reset) {
			auto st = std::make_unique<LegacyState>(fx.model, 2);
			sslm_prefix px = MakePrefix(fx, st->Pool(), toks, prefix_budget ? 4000 : 0);
			sslm_seq s = px ? (budget_holder ? MakeBudgetSeq(fx, st->Pool(), cap) : MakeLegacySeq(fx, st->Pool())) : nullptr;
			if (s) {
				st->h.seqs.push_back(s);
				PKV_CHECK_EQ(sslm_seq_adopt_prefix(s, px), SSLM_OK);
			}
			if (px) PKV_CHECK_EQ(sslm_prefix_release(px), SSLM_OK);
			if (s && reset) PKV_CHECK_EQ(sslm_seq_reset(s), SSLM_OK);
			return st;
		};
		{
			auto st = build(true, false);
			sslm_status refusal = SSLM_OK;
			// kills: a budget sharer whose shared pages leave the pool's accounting at the prefix's release
			PKV_CHECK_MSG(CountLegacyCreates(fx.model, st->Pool(), &refusal) == 0 && refusal == SSLM_KV_POOL_EXHAUSTED,
			              "%s, budget sharer: a legacy create was admitted with 250 shared pages held", mode);
		}
		ProbeExactlyFree(fx, [&] { return std::unique_ptr<ProbeState>(build(true, false)); }, 6);
		{
			auto st = build(true, true);
			// kills: a reset that keeps the shared pages mapped (or sends them to the holder's reserve)
			PKV_CHECK_MSG(CountLegacyCreates(fx.model, st->Pool()) == 1, "%s: the legacy create is refused after the reset",
			              mode);
		}
		{
			auto st = build(false, false);
			// kills: legacy adopt that shares (the 2-block promise breaks with the prefix released)
			PKV_CHECK_MSG(CountLegacyCreates(fx.model, st->Pool()) == 1, "%s, legacy holder: the legacy create is refused",
			              mode);
		}
	}
}

PKV_CELL("7.1", "C5", Cell71);
PKV_CELL("7.2", "C5", Cell72);
PKV_CELL("7.3", "C5", Cell73);
PKV_CELL("7.5", "C5", Cell75);
PKV_CELL("7.6/C5", "C5", Cell76Budget);
PKV_CELL("7.8/C5", "C5", Cell78Budget);
PKV_CELL("7.10/C5", "C5", Cell710Budget);
PKV_CELL("7.11/C5", "C5", Cell711Budget);
PKV_CELL("7.12/C5", "C5", Cell712Budget);
PKV_CELL("7.13/C5", "C5", Cell713Budget);
PKV_CELL("7.14", "C5", Cell714);

}  // namespace

// Paged-KV plan (rev 16.1) §7 dimension 3, concurrency: the C5-owned cells (budget adopters sharing
// prefix pages, begin_from, sslm_seq_restore_shared). Red by link at C1.
//
// A race cannot be rebuilt, so every cell here grades its one torn-down state with the fill probe's
// one-state form (§8, rev 6): creates totalling k - 1 admitted, a 2-page create refused, one admitted
// create of r pages released and r + 1 admitted. Each race runs RaceReps() times (100, the planner
// default of §8), and the verdict needs every run's probe to pass. Tokens are compared with serial
// runs of the same construction. The data-race verdict is the TSan leg's: the cloud's linux-clang
// TSan job builds superslm_pkv_c5 with -fsanitize=thread and runs these cells; here they run, and
// grade, without it. Threads never call the CHECK macros; each records what it saw and the main
// thread checks.

#include "pkv_budget_a_helpers.h"

namespace {

using namespace pkv;
using namespace pkv::budget_a;

constexpr int kThreads = 16;

std::vector<int32_t> P1000(const Fixture& fx) { return Stream(51, 1000, fx.vocab); }

// Reports the first failing repetition once, so a red race prints one line, not a hundred.
struct RepVerdict {
	int failed = 0;
	std::string first;
	void Fail(int rep, const std::string& why) {
		if (failed++ == 0) first = "rep " + std::to_string(rep) + ": " + why;
	}
	void Check(const char* cell, int reps) {
		PKV_CHECK_MSG(failed == 0, "%s: %d of %d repetitions failed; first: %s", cell, failed, reps, first.c_str());
	}
};

// ---- 3.1 [C5] -----------------------------------------------------------------------------------
// 16 threads adopt one frozen prefix and decode concurrently (each: budget-64 create, adopt, its own
// suffix, 8 tokens, release). Tokens equal serial runs; after the prefix's release the one-state
// fill probe admits the whole pool.
constexpr uint32_t k31Pool = 160;

std::vector<int32_t> Suffix31(const Fixture& fx, int i) {
	return Stream(300 + static_cast<uint32_t>(i), 5 + i % 7, fx.vocab);
}

// One adopter's whole life; returns its tokens, or a status-coded marker on any failure.
std::vector<int32_t> Adopter31(const Fixture& fx, sslm_kv_pool* pool, sslm_prefix p, int i) {
	sslm_seq s = nullptr;
	std::vector<int32_t> t;
	if (sslm_seq_create_budgeted(fx.model, pool, 64, &s) != SSLM_OK || !s) return {-100};
	if (sslm_seq_adopt_prefix(s, p) != SSLM_OK) t = {-101};
	else if (PrefillAll(fx.model, s, Suffix31(fx, i), 8) != SSLM_OK) t = {-102};
	else if (DecodeN(fx.model, s, 8, &t) != SSLM_OK) t.push_back(-103);
	if (sslm_seq_release(s) != SSLM_OK) t.push_back(-104);
	return t;
}

void Cell31() {
	const Fixture& fx = GetFixture("pkv_def");
	if (!fx.ok) return;
	std::vector<std::vector<int32_t>> want(kThreads);
	{
		Scene sc;
		PagePool& pool = sc.AddPool(fx.model, k31Pool);
		sslm_prefix p = FrozenBudgetPrefix(fx, &pool.pool, P1000(fx), sc.h);
		for (int i = 0; i < kThreads && p; ++i) want[static_cast<size_t>(i)] = Adopter31(fx, &pool.pool, p, i);
		PKV_CHECK_MSG(p && want[0].size() == 8, "3.1: the serial runs did not complete");
	}
	RepVerdict v;
	const int reps = RaceReps();
	for (int rep = 0; rep < reps; ++rep) {
		Scene sc;
		PagePool& pool = sc.AddPool(fx.model, k31Pool);
		sslm_prefix p = FrozenBudgetPrefix(fx, &pool.pool, P1000(fx), sc.h);
		if (!p) {
			v.Fail(rep, "prefix not built");
			break;
		}
		std::vector<std::vector<int32_t>> got(kThreads);
		RunTogether(kThreads, [&](int i) { got[static_cast<size_t>(i)] = Adopter31(fx, &pool.pool, p, i); });
		for (int i = 0; i < kThreads; ++i)  // kills: a shared page torn by a concurrent adopt or release
			if (got[static_cast<size_t>(i)] != want[static_cast<size_t>(i)]) v.Fail(rep, "thread " + std::to_string(i) + " tokens differ");
		ReleasePrefix(sc.h, p);
		// kills: a refcount update outside the pool mutex (a lost +1/-1 leaks or double-frees a page).
		if (!ProbeExactlyFreeOneState(fx, &pool.pool, k31Pool)) v.Fail(rep, "one-state probe");
		if (v.failed) break;
	}
	v.Check("3.1", reps);
}

// ---- 3.2 [C5] -----------------------------------------------------------------------------------
// Concurrent releases of sequences sharing pages: 16 budget adopters of one prefix, the prefix
// released, then all 16 released at once. Every shared page is freed exactly once: each page any of
// them wrote reads 0xCD, and the one-state probe admits exactly the pool (a page freed twice shows
// one over it, 161 against 160, and the probe's 2-page create is admitted).
constexpr uint32_t k32Pool = 160;

void Cell32() {
	const Fixture& fx = GetFixture("pkv_def");
	if (!fx.ok) return;
	RepVerdict v;
	const int reps = RaceReps();
	for (int rep = 0; rep < reps; ++rep) {
		Scene sc;
		PagePool& pool = sc.AddPool(fx.model, k32Pool);
		sslm_prefix p = FrozenBudgetPrefix(fx, &pool.pool, P1000(fx), sc.h);
		std::set<uint32_t> written;
		if (p)
			for (uint32_t pg : PrefixTable(p)) written.insert(pg);  // 62 shared pages and the prefix's tail page
		std::vector<sslm_seq> a;
		for (int i = 0; i < kThreads && p; ++i) {
			sslm_seq s = nullptr;
			if (sslm_seq_create_budgeted(fx.model, &pool.pool, 64, &s) != SSLM_OK || !s) break;
			a.push_back(s);
			if (sslm_seq_adopt_prefix(s, p) == SSLM_OK) DecodeN(fx.model, s, 2, nullptr);
			const std::vector<uint32_t> t = SeqTable(s);
			for (size_t j = 1000 / 16; j < t.size(); ++j) written.insert(t[j]);  // its private tail copy and new row
		}
		if (static_cast<int>(a.size()) != kThreads || written.size() < 63 + kThreads) {
			v.Fail(rep, "setup incomplete");
			for (sslm_seq s : a) sslm_seq_release(s);
			break;
		}
		ReleasePrefix(sc.h, p);  // the adopters are now the only sharers of the 62 pages
		std::vector<sslm_status> st(kThreads, SSLM_INVALID_ARGUMENT);
		RunTogether(kThreads, [&](int i) { st[static_cast<size_t>(i)] = sslm_seq_release(a[static_cast<size_t>(i)]); });
		for (sslm_status s : st)
			if (s != SSLM_OK) v.Fail(rep, "a release returned " + std::to_string(s));
		int not_poisoned = 0;  // kills: a page freed without its poison (a lost dirty flag in the race)
		for (uint32_t pg : written) not_poisoned += !AllBytes(PeekPage(fx, pool.pool, pg), 0xCD);
		if (not_poisoned) v.Fail(rep, std::to_string(not_poisoned) + " freed pages do not read 0xCD");
		// kills: a shared page freed twice (or never) under concurrent release.
		if (!ProbeExactlyFreeOneState(fx, &pool.pool, k32Pool)) v.Fail(rep, "one-state probe");
		if (v.failed) break;
	}
	v.Check("3.2", reps);
}

// ---- 3.7 [C5] -----------------------------------------------------------------------------------
// A prefix's release racing its last sharer's release (or reset) on the same 62 pages: each page is
// freed exactly once, inside the critical section that takes it to 0. After the race the shared pages
// read 0xCD; with the reset variant the holder keeps exactly R(64) (one-state probe at pool - 5), and
// after the teardown the probe admits the whole pool.
constexpr uint32_t k37Pool = 120;

void Cell37() {
	const Fixture& fx = GetFixture("pkv_def");
	if (!fx.ok) return;
	for (int reset_variant = 0; reset_variant <= 1; ++reset_variant) {
		const char* name = reset_variant ? "3.7 (prefix release vs sharer reset)" : "3.7 (prefix release vs sharer release)";
		RepVerdict v;
		const int reps = RaceReps();
		for (int rep = 0; rep < reps; ++rep) {
			Scene sc;
			PagePool& pool = sc.AddPool(fx.model, k37Pool);
			sslm_prefix p = FrozenBudgetPrefix(fx, &pool.pool, P1000(fx), sc.h);
			sslm_seq h = BudgetSeq(fx, &pool.pool, 64, sc.h);
			if (!p || !h || sslm_seq_adopt_prefix(h, p) != SSLM_OK) {
				v.Fail(rep, "setup");
				break;
			}
			DecodeN(fx.model, h, 2, nullptr);
			const std::vector<uint32_t> pt = PrefixTable(p);
			const std::set<uint32_t> shared(pt.begin(), pt.begin() + std::min<size_t>(pt.size(), 1000 / 16));
			sc.h.prefixes.clear();
			if (!reset_variant) sc.h.seqs.clear();
			sslm_status st[2] = {SSLM_INVALID_ARGUMENT, SSLM_INVALID_ARGUMENT};
			RunTogether(2, [&](int i) {
				st[i] = i == 0 ? sslm_prefix_release(p) : reset_variant ? sslm_seq_reset(h) : sslm_seq_release(h);
			});
			if (st[0] != SSLM_OK || st[1] != SSLM_OK) v.Fail(rep, "a racing call failed");
			int not_poisoned = 0;  // kills: a decrement to 0 whose free is lost when the two sections interleave
			for (uint32_t pg : shared) not_poisoned += !AllBytes(PeekPage(fx, pool.pool, pg), 0xCD);
			if (shared.size() != 62 || not_poisoned) v.Fail(rep, std::to_string(not_poisoned) + " shared pages not freed");
			if (reset_variant) {
				// kills: the reset sending the last sharer's shared pages to its reserve (pool - 5 - 62).
				if (!ProbeExactlyFreeOneState(fx, &pool.pool, k37Pool - fx.R(64))) v.Fail(rep, "one-state probe after reset");
				ReleaseSeq(sc.h, h);
			}
			// kills: a page freed by both racing calls (one over), or by neither (62 under).
			if (!ProbeExactlyFreeOneState(fx, &pool.pool, k37Pool)) v.Fail(rep, "one-state probe after teardown");
			if (v.failed) break;
		}
		v.Check(name, reps);
	}
}

// ---- 3.8 [C5] -----------------------------------------------------------------------------------
// A sharer decoding while its prefix is released (refcount 2 -> 1): its tokens equal the serial run
// in which the prefix stays. Then the teardown's one-state probe admits the whole pool.
constexpr uint32_t k38Pool = 120;
constexpr int k38Tokens = 32;

void Cell38() {
	const Fixture& fx = GetFixture("pkv_def");
	if (!fx.ok) return;
	std::vector<int32_t> want;
	{
		Scene sc;
		PagePool& pool = sc.AddPool(fx.model, k38Pool);
		sslm_prefix p = FrozenBudgetPrefix(fx, &pool.pool, P1000(fx), sc.h);
		sslm_seq h = BudgetSeq(fx, &pool.pool, 64, sc.h);
		if (p && h && sslm_seq_adopt_prefix(h, p) == SSLM_OK) DecodeN(fx.model, h, k38Tokens, &want);
		PKV_CHECK_EQ(want.size(), k38Tokens);
	}
	RepVerdict v;
	const int reps = RaceReps();
	for (int rep = 0; rep < reps; ++rep) {
		Scene sc;
		PagePool& pool = sc.AddPool(fx.model, k38Pool);
		sslm_prefix p = FrozenBudgetPrefix(fx, &pool.pool, P1000(fx), sc.h);
		sslm_seq h = BudgetSeq(fx, &pool.pool, 64, sc.h);
		if (!p || !h || sslm_seq_adopt_prefix(h, p) != SSLM_OK) {
			v.Fail(rep, "setup");
			break;
		}
		sc.h.prefixes.clear();
		std::atomic<int> emitted{0};
		std::atomic<bool> done{false};
		std::vector<int32_t> got;
		sslm_status rel = SSLM_INVALID_ARGUMENT;
		RunTogether(2, [&](int i) {
			if (i == 0) {
				for (int t = 0; t < k38Tokens; ++t) {
					if (DecodeN(fx.model, h, 1, &got) != SSLM_OK) break;
					emitted.fetch_add(1, std::memory_order_release);
				}
				done.store(true, std::memory_order_release);
			} else {
				// Released while the sharer is mid-run (after its 4th token, whatever the timing after that).
				while (emitted.load(std::memory_order_acquire) < 4 && !done.load(std::memory_order_acquire))
					std::this_thread::yield();
				rel = sslm_prefix_release(p);
			}
		});
		if (rel != SSLM_OK) v.Fail(rep, "prefix release failed");
		// kills: a prefix release that frees (and poisons) pages a live sharer maps.
		if (got != want) v.Fail(rep, "tokens differ from the serial run");
		ReleaseSeq(sc.h, h);
		if (!ProbeExactlyFreeOneState(fx, &pool.pool, k38Pool)) v.Fail(rep, "one-state probe");
		if (v.failed) break;
	}
	v.Check("3.8", reps);
}

// ---- 3.9 [C5] -----------------------------------------------------------------------------------
// 16 threads call sslm_prefix_begin_from on one parent (each then prefills its own child, freezes it,
// has a budget holder adopt it and decode 4, and releases both), then the parent is released. The
// one-state probe admits the whole pool, and each thread's tokens equal a serial run's.
constexpr uint32_t k39Pool = 400;

void Cell39() {
	const Fixture& fx = GetFixture("pkv_def");
	if (!fx.ok) return;
	// The child's whole life, given the pool its holder draws from.
	auto life = [&fx](sslm_kv_pool* pool, sslm_prefix parent, int i) -> std::vector<int32_t> {
		sslm_prefix c = nullptr;
		std::vector<int32_t> t;
		if (sslm_prefix_begin_from(parent, 32, &c) != SSLM_OK || !c) return {-100};
		sslm_seq s = nullptr;
		if (PrefixPrefillAll(fx.model, c, Stream(400 + static_cast<uint32_t>(i), 3 + i, fx.vocab), 8) != SSLM_OK ||
		    sslm_prefix_freeze(c) != SSLM_OK)
			t = {-101};
		else if (sslm_seq_create_budgeted(fx.model, pool, 16, &s) != SSLM_OK || !s)
			t = {-102};
		else if (sslm_seq_adopt_prefix(s, c) != SSLM_OK)
			t = {-103};
		else if (DecodeN(fx.model, s, 4, &t) != SSLM_OK)
			t.push_back(-104);
		if (s && sslm_seq_release(s) != SSLM_OK) t.push_back(-105);
		if (sslm_prefix_release(c) != SSLM_OK) t.push_back(-106);
		return t;
	};
	std::vector<std::vector<int32_t>> want(kThreads);
	{
		Scene sc;
		PagePool& pool = sc.AddPool(fx.model, k39Pool);
		sslm_prefix w = FrozenBudgetPrefix(fx, &pool.pool, P1000(fx), sc.h);
		for (int i = 0; i < kThreads && w; ++i) want[static_cast<size_t>(i)] = life(&pool.pool, w, i);
		PKV_CHECK_MSG(w && want[0].size() == 4, "3.9: the serial runs did not complete");
	}
	RepVerdict v;
	const int reps = RaceReps();
	for (int rep = 0; rep < reps; ++rep) {
		Scene sc;
		PagePool& pool = sc.AddPool(fx.model, k39Pool);
		sslm_prefix w = FrozenBudgetPrefix(fx, &pool.pool, P1000(fx), sc.h);
		if (!w) {
			v.Fail(rep, "parent not built");
			break;
		}
		std::vector<std::vector<int32_t>> got(kThreads);
		RunTogether(kThreads, [&](int i) { got[static_cast<size_t>(i)] = life(&pool.pool, w, i); });
		for (int i = 0; i < kThreads; ++i)
			if (got[static_cast<size_t>(i)] != want[static_cast<size_t>(i)]) v.Fail(rep, "thread " + std::to_string(i) + " differs");
		ReleasePrefix(sc.h, w);
		// kills: begin_from's share (+1 on 62 pages) or its R(budget) draw racing another's outside the mutex.
		if (!ProbeExactlyFreeOneState(fx, &pool.pool, k39Pool)) v.Fail(rep, "one-state probe");
		if (v.failed) break;
	}
	v.Check("3.9", reps);
}

// ---- 3.10 [C5] ----------------------------------------------------------------------------------
// 16 threads call sslm_seq_restore_shared with one handle (the game host's load): each restores its
// own blob (a budget-512 holder that adopted the same 1,000-token prefix in another pool) and
// decodes 8. The pool is exactly the prefix's 63 pages plus 16 x R(512), so all 16 are admitted only
// if each shares the prefix's 62 full pages and draws exactly R(512) (a private restore needs E = 62
// more). Tokens equal the unsaved originals'; after the teardown the one-state probe admits the pool.
void Cell310() {
	const Fixture& fx = GetFixture("pkv_def");
	if (!fx.ok) return;
	std::vector<std::vector<uint8_t>> blobs(kThreads);
	std::vector<std::vector<int32_t>> want(kThreads);
	{
		Scene sc;
		PagePool& pool = sc.AddPool(fx.model, 120);
		sslm_prefix p = FrozenBudgetPrefix(fx, &pool.pool, P1000(fx), sc.h);
		for (int i = 0; i < kThreads && p; ++i) {
			sslm_seq s = BudgetSeq(fx, &pool.pool, 512, sc.h);
			if (!s || sslm_seq_adopt_prefix(s, p) != SSLM_OK) break;
			PrefillAll(fx.model, s, Stream(500 + static_cast<uint32_t>(i), 3, fx.vocab), 8);
			DecodeN(fx.model, s, 4, nullptr);
			blobs[static_cast<size_t>(i)] = Save(s);
			DecodeN(fx.model, s, 8, &want[static_cast<size_t>(i)]);
			ReleaseSeq(sc.h, s);
		}
		PKV_CHECK_MSG(want[kThreads - 1].size() == 8, "3.10: the blobs were not built");
	}
	const uint32_t pool_pages = static_cast<uint32_t>(PrefixPages(fx, 1000) + kThreads * fx.R(512));  // 591
	RepVerdict v;
	const int reps = RaceReps();
	for (int rep = 0; rep < reps; ++rep) {
		Scene sc;
		PagePool& pool = sc.AddPool(fx.model, pool_pages);
		sslm_prefix p = FrozenBudgetPrefix(fx, &pool.pool, P1000(fx), sc.h);
		if (!p) {
			v.Fail(rep, "prefix not built");
			break;
		}
		std::vector<std::vector<int32_t>> got(kThreads);
		RunTogether(kThreads, [&](int i) {
			const std::vector<uint8_t>& b = blobs[static_cast<size_t>(i)];
			sslm_seq r = nullptr;
			uint32_t shared = 0;
			std::vector<int32_t>& t = got[static_cast<size_t>(i)];
			const sslm_status st = sslm_seq_restore_shared(fx.model, &pool.pool, b.data(), b.size(), p, &r, &shared);
			if (st != SSLM_OK || !r) {
				t = {-1000 - static_cast<int32_t>(st)};
				return;
			}
			if (DecodeN(fx.model, r, 8, &t) != SSLM_OK) t.push_back(-1);
			if (sslm_seq_release(r) != SSLM_OK) t.push_back(-2);
		});
		// kills: a restore that cannot share under concurrency (refused in the exact pool), or that maps
		// a page another restore is still scattering.
		for (int i = 0; i < kThreads; ++i)
			if (got[static_cast<size_t>(i)] != want[static_cast<size_t>(i)]) v.Fail(rep, "thread " + std::to_string(i) + " differs");
		ReleasePrefix(sc.h, p);
		if (!ProbeExactlyFreeOneState(fx, &pool.pool, pool_pages)) v.Fail(rep, "one-state probe");
		if (v.failed) break;
	}
	v.Check("3.10", reps);
}

// ---- 3.11 [C5] ----------------------------------------------------------------------------------
// sslm_seq_restore_shared (with the prefix's handle) racing a sharer's release on the same pages:
// every page is freed exactly once. The restored holder continues as the unsaved original; after the
// teardown the one-state probe admits the whole pool.
constexpr uint32_t k311Pool = 200;

void Cell311() {
	const Fixture& fx = GetFixture("pkv_def");
	if (!fx.ok) return;
	std::vector<uint8_t> blob;
	std::vector<int32_t> want;
	{
		Scene sc;
		PagePool& pool = sc.AddPool(fx.model, 120);
		sslm_prefix p = FrozenBudgetPrefix(fx, &pool.pool, P1000(fx), sc.h);
		sslm_seq s = BudgetSeq(fx, &pool.pool, 512, sc.h);
		if (p && s && sslm_seq_adopt_prefix(s, p) == SSLM_OK && DecodeN(fx.model, s, 4, nullptr) == SSLM_OK) {
			blob = Save(s);
			DecodeN(fx.model, s, 8, &want);
		}
		PKV_CHECK_MSG(want.size() == 8, "3.11: the blob was not built");
	}
	RepVerdict v;
	const int reps = RaceReps();
	for (int rep = 0; rep < reps; ++rep) {
		Scene sc;
		PagePool& pool = sc.AddPool(fx.model, k311Pool);
		sslm_prefix p = FrozenBudgetPrefix(fx, &pool.pool, P1000(fx), sc.h);
		sslm_seq sharer = BudgetSeq(fx, &pool.pool, 64, sc.h);
		if (!p || !sharer || sslm_seq_adopt_prefix(sharer, p) != SSLM_OK) {
			v.Fail(rep, "setup");
			break;
		}
		DecodeN(fx.model, sharer, 2, nullptr);
		sc.h.seqs.clear();
		sslm_seq r = nullptr;
		sslm_status st[2] = {SSLM_INVALID_ARGUMENT, SSLM_INVALID_ARGUMENT};
		RunTogether(2, [&](int i) {
			uint32_t shared = 0;
			st[i] = i == 0 ? sslm_seq_restore_shared(fx.model, &pool.pool, blob.data(), blob.size(), p, &r, &shared)
			               : sslm_seq_release(sharer);
		});
		if (st[0] != SSLM_OK || st[1] != SSLM_OK || !r) {
			v.Fail(rep, "restore " + std::to_string(st[0]) + ", release " + std::to_string(st[1]));
			if (r) sslm_seq_release(r);
			break;
		}
		std::vector<int32_t> got;
		DecodeN(fx.model, r, 8, &got);
		if (got != want) v.Fail(rep, "the restored holder's tokens differ");
		sslm_seq_release(r);
		ReleasePrefix(sc.h, p);
		// kills: a +1 (restore) and a -1 (release) on one page interleaving outside the pool mutex.
		if (!ProbeExactlyFreeOneState(fx, &pool.pool, k311Pool)) v.Fail(rep, "one-state probe");
		if (v.failed) break;
	}
	v.Check("3.11", reps);
}

PKV_CELL("3.1/C5", "C5", Cell31);
PKV_CELL("3.2/C5", "C5", Cell32);
PKV_CELL("3.7/C5", "C5", Cell37);
PKV_CELL("3.8/C5", "C5", Cell38);
PKV_CELL("3.9/C5", "C5", Cell39);
PKV_CELL("3.10/C5", "C5", Cell310);
PKV_CELL("3.11/C5", "C5", Cell311);

}  // namespace

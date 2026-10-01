// Paged-KV plan (rev 16.1) §7 dimension 3 (concurrency), the cells step C4 owns: legacy holders
// on pages under concurrent lifecycle calls.
//
//   3.3  concurrent create/release driving a legacy pool to exhaustion and back
//   3.4  release during an in-flight batched decode of the same sequence (the teardown-during-
//        flight construction of tests/t2199-damped-greedy-red-suite/phaseD3_teardown_red.cpp)
//
// Each cell writes its race and grades the state it leaves: statuses, tokens against a serial run,
// page bytes through the peek, and the legacy-create admission count (§8). A race cannot be
// rebuilt, so each count grades the one torn-down state. The TSan leg (Linux clang in the cloud,
// ASan on MSVC on the box) runs these same cells; this file states the race, the sanitizer run is
// that leg's.

#include "pkv_legacy_a_helpers.h"

#include <atomic>
#include <thread>

namespace {

using namespace pkv;
using namespace pkv::legacy_a;

// One racing run of 3.3: `threads` threads, each `iters` times: a legacy create, and when admitted
// a 20-token prefill (so the holder maps and dirties pages) and a release. Returns the statistics;
// the pool is left torn down for the caller to grade.
struct ChurnResult {
	int admitted = 0;
	int refused = 0;
	int bad_status = 0;
	int bad_prefill = 0;
	int max_live = 0;
};

ChurnResult Churn(const Fixture& fx, sslm_kv_pool* pool, int threads, int iters) {
	std::atomic<int> admitted{0}, refused{0}, bad_status{0}, bad_prefill{0}, live{0}, max_live{0};
	std::atomic<bool> go{false};
	std::vector<std::thread> ts;
	for (int t = 0; t < threads; ++t)
		ts.emplace_back([&, t] {
			while (!go.load(std::memory_order_acquire)) std::this_thread::yield();
			const std::vector<int32_t> prompt = Stream(80 + static_cast<uint32_t>(t), 20, fx.vocab);
			for (int i = 0; i < iters; ++i) {
				sslm_seq s = nullptr;
				const sslm_status st = sslm_seq_create(fx.model, pool, &s);
				if (st == SSLM_KV_POOL_EXHAUSTED) {
					refused.fetch_add(1);
					if (s) bad_status.fetch_add(1);
					continue;
				}
				if (st != SSLM_OK || !s) {
					bad_status.fetch_add(1);
					continue;
				}
				admitted.fetch_add(1);
				const int now = live.fetch_add(1) + 1;
				int seen = max_live.load();
				while (now > seen && !max_live.compare_exchange_weak(seen, now)) {
				}
				if (PrefillAll(fx.model, s, prompt, 16) != SSLM_OK) bad_prefill.fetch_add(1);
				live.fetch_sub(1);
				if (sslm_seq_release(s) != SSLM_OK) bad_status.fetch_add(1);
			}
		});
	go.store(true, std::memory_order_release);
	for (auto& t : ts) t.join();
	ChurnResult r;
	r.admitted = admitted.load();
	r.refused = refused.load();
	r.bad_status = bad_status.load();
	r.bad_prefill = bad_prefill.load();
	r.max_live = max_live.load();
	return r;
}

// 3.3 [C4]. Concurrent create/release driving the pool to exhaustion and back. 8 threads churn a
// 3-block pool (768 pages, 3 legacy reservations of 256): creates are refused
// SSLM_KV_POOL_EXHAUSTED while 3 are live and admitted again as others release. Graded on the one
// torn-down state of each run by the legacy-create admission count, which sees a one-page error in
// one direction per pool size (§8), so the race runs twice:
//   - run A, a 3-block pool: back to exactly 3 admitted. A page leaked or left in a reserve (a
//     deficit) leaves 767 free and admits 2;
//   - run B, a 4-block pool holding a frozen 16-token legacy prefix throughout: the prefix keeps
//     ceil(16 / B) = 1 page and its freeze returns 255, so 3 * 256 + 255 are free and 3 are
//     admitted. A page freed twice (an excess) makes 4 * 256 free and admits 4.
void Cell33() {
	const Fixture& fx = GetFixture("pkv_def");
	if (!fx.ok) return;
	constexpr int kThreads = 8, kIters = 150, kN = 3;

	{  // run A
		LegacyPool pool(fx.model, kN);
		PKV_CHECK_EQ(pool.status, SSLM_OK);
		if (!pool.pool) return;
		const ChurnResult r = Churn(fx, &pool.pool, kThreads, kIters);
		// kills: a create that admits a holder the pool cannot reserve whole (more than 3 live)
		PKV_CHECK_MSG(r.max_live <= kN, "run A: %d holders live at once in a %d-block pool", r.max_live, kN);
		// kills: a refusal reported as anything but SSLM_KV_POOL_EXHAUSTED, or a failed release
		PKV_CHECK_EQ(r.bad_status, 0);
		PKV_CHECK_EQ(r.bad_prefill, 0);
		PKV_CHECK_MSG(r.refused > 0, "run A reached exhaustion (%d refusals)", r.refused);
		PKV_CHECK_MSG(r.admitted > kN, "run A came back from exhaustion (%d admitted)", r.admitted);
		PKV_CHECK_EQ(r.admitted + r.refused, kThreads * kIters);
		sslm_status refusal = SSLM_OK;
		// kills: a page leaked by a release racing a create (the count falls to 2)
		PKV_CHECK_EQ(CountLegacyCreates(fx.model, &pool.pool, &refusal), kN);
		PKV_CHECK_EQ(refusal, SSLM_KV_POOL_EXHAUSTED);
		PKV_CHECK_EQ(sslm_kv_pool_destroy(pool.pool), SSLM_OK);
		pool.pool = nullptr;
	}
	{  // run B
		LegacyPool pool(fx.model, kN + 1);
		PKV_CHECK_EQ(pool.status, SSLM_OK);
		if (!pool.pool) return;
		sslm_prefix px = nullptr;
		PKV_CHECK_EQ(sslm_prefix_begin(fx.model, &pool.pool, &px), SSLM_OK);
		if (!px) return;
		PKV_CHECK_EQ(PrefixPrefillAll(fx.model, px, Stream(79, 16, fx.vocab), 16), SSLM_OK);
		PKV_CHECK_EQ(sslm_prefix_freeze(px), SSLM_OK);
		PKV_CHECK_EQ((16 + fx.B() - 1) / fx.B(), 1);  // §3.4: the prefix keeps one page
		PKV_CHECK_EQ(CountLegacyCreates(fx.model, &pool.pool), kN);  // before the race: 3 * 256 + 255 free
		const ChurnResult r = Churn(fx, &pool.pool, kThreads, kIters);
		PKV_CHECK_MSG(r.max_live <= kN, "run B: %d holders live at once beside the prefix", r.max_live);
		PKV_CHECK_EQ(r.bad_status, 0);
		PKV_CHECK_EQ(r.bad_prefill, 0);
		PKV_CHECK_MSG(r.refused > 0 && r.admitted > kN, "run B reached exhaustion (%d) and came back (%d)", r.refused,
		              r.admitted);
		// kills: a page freed twice by racing releases (the count rises to 4)
		PKV_CHECK_EQ(CountLegacyCreates(fx.model, &pool.pool), kN);
		PKV_CHECK_EQ(sslm_prefix_release(px), SSLM_OK);
		PKV_CHECK_EQ(CountLegacyCreates(fx.model, &pool.pool), kN + 1);
	}
}

// The serial twin of one 3.4 trial: A and B prefilled 16 tokens each (different prompts), one
// batched decode that emits the ready tokens, and a second batched decode that writes row 16 (the
// first row of page 1, mapped by the decode path). Returns the second call's tokens.
bool SerialTokens(const Fixture& fx, int32_t out[2]) {
	LegacyPool pool(fx.model, 2);
	if (!pool.pool) return false;
	sslm_seq a = nullptr, b = nullptr;
	if (sslm_seq_create(fx.model, &pool.pool, &a) != SSLM_OK || sslm_seq_create(fx.model, &pool.pool, &b) != SSLM_OK)
		return false;
	bool ok = PrefillAll(fx.model, a, Stream(91, 16, fx.vocab), 16) == SSLM_OK &&
	          PrefillAll(fx.model, b, Stream(92, 16, fx.vocab), 16) == SSLM_OK;
	sslm_decode_params p{};
	p.layer_budget = static_cast<int32_t>(fx.geo.layers);
	sslm_seq batch[2] = {a, b};
	int32_t first[2] = {-1, -1};
	ok = ok && sslm_decode_step(fx.model, batch, 2, &p, nullptr, first) == SSLM_OK;
	ok = ok && sslm_decode_step(fx.model, batch, 2, &p, nullptr, out) == SSLM_OK;
	sslm_seq_release(b);
	sslm_seq_release(a);
	return ok && out[0] >= 0 && out[1] >= 0;
}

// 3.4 [C4]. Release during an in-flight batched decode of the same sequence, stays safe with
// tables. phaseD3_teardown_red.cpp's construction, ported to legacy holders on pages: per trial, a
// fresh 2-block pool over memory filled with 0x5A, sequences A and B prefilled 16 tokens and past
// their ready tokens, then one batched decode {A, B} on one thread raced against
// sslm_seq_release(B) on another, with no barrier. That decode maps page 1 for each sequence from
// its reserve, so the release can land before, during or after B's map and write.
// Each trial asserts:
//   - A's token equals the serial twin's (no cross-sequence corruption through a table);
//   - B's slot is the serial twin's token (the release lost) or -3, the dead-sequence sentinel;
//   - every pool page outside A's table reads wholly 0xCD (B's written pages, poisoned on
//     release) or wholly 0x5A (never written): a decode write landing in a page after B's release
//     freed it, or a page freed while still mapped, leaves a page that is neither;
//   - after A's release the pool admits its 2 legacy creates again, and destroys cleanly.
void Cell34() {
	const Fixture& fx = GetFixture("pkv_def");
	if (!fx.ok) return;
	int32_t serial[2] = {-1, -1};
	PKV_CHECK(SerialTokens(fx, serial));
	constexpr int kTrials = 200;
	int a_bad = 0, b_bad = 0, b_won = 0, b_lost = 0, page_bad = 0, peek_bad = 0, count_bad = 0, status_bad = 0;
	for (int trial = 0; trial < kTrials; ++trial) {
		LegacyPool pool(fx.model, 2, 0x5A);
		if (!pool.pool) {
			++status_bad;
			continue;
		}
		sslm_seq a = nullptr, b = nullptr;
		if (sslm_seq_create(fx.model, &pool.pool, &a) != SSLM_OK || sslm_seq_create(fx.model, &pool.pool, &b) != SSLM_OK) {
			++status_bad;
			continue;
		}
		sslm_decode_params p{};
		p.layer_budget = static_cast<int32_t>(fx.geo.layers);
		sslm_seq batch[2] = {a, b};
		int32_t ready[2] = {-1, -1}, out[2] = {-1, -1};
		if (PrefillAll(fx.model, a, Stream(91, 16, fx.vocab), 16) != SSLM_OK ||
		    PrefillAll(fx.model, b, Stream(92, 16, fx.vocab), 16) != SSLM_OK ||
		    sslm_decode_step(fx.model, batch, 2, &p, nullptr, ready) != SSLM_OK)
			++status_bad;
		std::thread decode_thread([&] { sslm_decode_step(fx.model, batch, 2, &p, nullptr, out); });
		std::thread release_thread([&] { sslm_seq_release(b); });
		decode_thread.join();
		release_thread.join();

		// kills: A's decode reading or writing through B's table, or a page B freed mid-flight
		// handed to A
		if (out[0] != serial[0]) ++a_bad;
		if (out[1] == -3) ++b_won;
		else if (out[1] == serial[1]) ++b_lost;
		else ++b_bad;  // kills: B decoded through a freed table (garbage token or the retired -1)

		std::vector<uint32_t> a_table;
		if (!SeqTable(a, &a_table)) ++peek_bad;
		const std::set<uint32_t> mine(a_table.begin(), a_table.end());
		for (int64_t i = 0; i < 2 * fx.CapPages(); ++i) {
			if (mine.count(static_cast<uint32_t>(i))) continue;
			const std::vector<uint8_t> pg = PeekPage(pool.pool, static_cast<uint32_t>(i), fx.PageBytes());
			if (pg.empty()) {
				++peek_bad;
				break;
			}
			// kills: a write after free (B's row 16 landing in a page its release already poisoned
			// and returned), and a release that frees B's pages without poisoning them
			if (!AllBytesAre(pg, 0xCD) && !AllBytesAre(pg, 0x5A)) ++page_bad;
		}
		if (sslm_seq_release(a) != SSLM_OK) ++status_bad;
		// kills: a page leaked or double-freed by the racing release (a count other than 2)
		if (CountLegacyCreates(fx.model, &pool.pool) != 2) ++count_bad;
		if (sslm_kv_pool_destroy(pool.pool) != SSLM_OK) ++status_bad;
		pool.pool = nullptr;
	}
	std::printf("  3.4: %d trials, release won %d, lost %d\n", kTrials, b_won, b_lost);
	PKV_CHECK_EQ(status_bad, 0);
	PKV_CHECK_MSG(a_bad == 0, "%d trials: the surviving sequence's token differs from the serial run", a_bad);
	PKV_CHECK_MSG(b_bad == 0, "%d trials: the released slot is neither the serial token nor -3", b_bad);
	PKV_CHECK_EQ(peek_bad, 0);
	PKV_CHECK_MSG(page_bad == 0, "%d pages outside the survivor's table are neither poison nor untouched", page_bad);
	PKV_CHECK_MSG(count_bad == 0, "%d trials: the torn-down pool does not admit exactly 2 legacy creates", count_bad);
}

PKV_CELL("3.3", "C4", Cell33);
PKV_CELL("3.4", "C4", Cell34);

}  // namespace

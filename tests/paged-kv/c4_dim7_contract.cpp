// Paged-KV plan (rev 16.1) §7 dimension 7, the legacy holders' contract cells (step C4): 7.6, 7.8,
// 7.10, 7.11, 7.12 and 7.13, each the part its bracket gives C4.
//
// This file replaces the global operator new for the whole superslm_pkv_c4 executable, with the
// counting operator new of tests/t2138-abi-red-suite/dim7_contract_red.cpp:64 (the counter the
// existing zero-allocation cell, its cell 1a, reads; tests/decode_threading_alloc_main.cpp reuses
// it the same way). It counts and forwards to malloc, so no other cell sees a difference, and no
// other c4_*.cpp may define one.
//
// Verdict sources (§7 grading rule): ABI statuses counted as admissions against pool sizes the
// test computes from §3.4; tokens against the R0 reference (v1.11.0) or an unsaved twin; blob
// bytes and sizes; and, in 7.10 only, the two seam counters that are that cell's subject.

#include "pkv_legacy_b_helpers.h"

#include <atomic>
#include <cstdlib>
#include <new>

#if !defined(_WIN32)
#include <sys/wait.h>
#include <unistd.h>
#endif

namespace {
std::atomic<long long> g_new_calls{0};
}  // namespace

// GCC's -Wmismatched-new-delete cannot see that these replacements pair malloc with free. (No
// function-pointer indirection: under MSVC's /MD, std::malloc's address is a DLL import, so such a
// pointer is initialized at run time, after another file's static registration has already called
// this operator new.)
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmismatched-new-delete"
#endif
void* operator new(std::size_t size) {
	g_new_calls.fetch_add(1, std::memory_order_relaxed);
	if (void* p = std::malloc(size ? size : 1)) return p;
	throw std::bad_alloc{};
}
void* operator new[](std::size_t size) { return operator new(size); }
void operator delete(void* p) noexcept { std::free(p); }
void operator delete(void* p, std::size_t) noexcept { std::free(p); }
void operator delete[](void* p) noexcept { std::free(p); }
void operator delete[](void* p, std::size_t) noexcept { std::free(p); }
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic pop
#endif

namespace {

using namespace pkv;
using namespace pkv::legacy_b;

// ---- 7.6 [C4 whole_reserve] ----------------------------------------------------------------------
//
// "kv_blocks_resident stays 1 in every mode" (§3.6, G14): page accounting goes to the new stats
// verb, so sslm_stats' documented reading does not change. The subject is sslm_stats' own field,
// read for whole_reserve holders in every state C4 can make: fresh, after a 1,500-token prefill
// (94 pages mapped), mid-token, copy-adopted (1,008), reset, and restored from 1.9.0's SSB5, from
// v1.8.1's SSB4 and from a hand-built SSB6 whole_reserve blob (A2).
// Mutant killed: "kv_blocks_resident reports the holder's mapped (or reserved) page count".

void ExpectResidentOne(const Fixture& fx, sslm_seq s, const char* state) {
	sslm_stats_out o{};
	const sslm_status st = sslm_stats(fx.model, s, &o);
	// kills: kv_blocks_resident reports mapped or reserved pages instead of the constant 1
	PKV_CHECK_MSG(st == SSLM_OK && o.kv_blocks_resident == 1, "7.6 %s: sslm_stats %d, kv_blocks_resident %d", state,
	              static_cast<int>(st), o.kv_blocks_resident);
}

void Cell76WholeReserveResident() {
	const Fixture& fx = GetFixture("pkv_def");
	if (!fx.ok) return;
	LegacyPool pool(fx.model, 4);
	sslm_seq s = nullptr;
	PKV_CHECK_EQ(sslm_seq_create(fx.model, &pool.pool, &s), SSLM_OK);
	if (!s) return;
	ExpectResidentOne(fx, s, "fresh");
	PKV_CHECK_EQ(PrefillAll(fx.model, s, Stream(11, 1500, fx.vocab), 64), SSLM_OK);
	ExpectResidentOne(fx, s, "after prefill 1500");
	EnterMidToken(fx.model, s);
	ExpectResidentOne(fx, s, "mid-token");
	std::vector<uint8_t> ssb5;
	PKV_CHECK(SaveBlob(s, &ssb5));
	PKV_CHECK_EQ(sslm_seq_reset(s), SSLM_OK);
	ExpectResidentOne(fx, s, "reset");
	sslm_prefix px = nullptr;
	PKV_CHECK_EQ(sslm_prefix_begin(fx.model, &pool.pool, &px), SSLM_OK);
	if (px) {
		PKV_CHECK_EQ(PrefixPrefillAll(fx.model, px, Stream(51, 1008, fx.vocab), 64), SSLM_OK);
		PKV_CHECK_EQ(sslm_prefix_freeze(px), SSLM_OK);
		PKV_CHECK_EQ(sslm_seq_adopt_prefix(s, px), SSLM_OK);
		ExpectResidentOne(fx, s, "copy-adopted 1008");
	}
	sslm_seq_release(s);
	if (px) sslm_prefix_release(px);
	struct Src {
		const char* what;
		std::vector<uint8_t> blob;
	};
	std::vector<Src> blobs = {{"restored SSB5 (own save)", ssb5},
	                          {"restored v1.11.0 SSB5 pin", Pin("v1.11.0", "persist", "saved")},
	                          {"restored v1.8.1 SSB4 pin", Pin("v1.8.1", "persist", "saved")},
	                          {"restored hand-built SSB6 whole_reserve", Ssb6WholeReserveFromSsb5(ssb5, fx, 1000)}};
	for (const Src& b : blobs) {
		sslm_seq r = nullptr;
		PKV_CHECK_MSG(Restore(fx, &pool.pool, b.blob, &r) == SSLM_OK, "7.6 %s: restore", b.what);
		if (!r) continue;
		ExpectResidentOne(fx, r, b.what);
		sslm_seq_release(r);
	}
}

// ---- 7.8 [C4] --------------------------------------------------------------------------------------
//
// "Each new verb's header behaviour sentences have a failing cell where cheap." C4 adds no verb;
// it changes legacy verbs' behaviour in three sentences §3.4 and §3.6 state, each cheap:
//   (a) a legacy prefix's first freeze returns its unused reserve to the pool (§3.4, §3.5: "only
//       ever increases free pages; it can admit a handle today's build would refuse"). Two legacy
//       prefixes of 1,000 tokens (63 pages each) fill a 2-block pool, so a legacy create is
//       refused; after both freezes 512 - 126 = 386 pages are free and exactly one create is
//       admitted. That create then prefills 300 tokens into pages the freezes returned, and an
//       adopter of each prefix still decodes v1.11.0's tokens, so no mapped page was returned.
//       The second freeze's own sentence (no pool effect) is 1.13's legacy half.
//   (b) sslm_kv_pool_overhead_size "saturates for the same counts" as the u32 page-count refusal
//       (§3.6): SIZE_MAX at the first block_count whose page count exceeds UINT32_MAX, and a
//       finite value one block below. The thresholds are §3.6's, recomputed here: 2,097,152 at cap
//       32,768 and 16,777,216 at cap 4,096 (B = 16).
//   (c) "sslm_kv_block_size keeps its meaning", and sslm_kv_pool_create(N) accepts exactly the
//       buffer the two size verbs name and refuses one byte less (as today): the page pool lives
//       in the same buffer size (§3.6).
// Mutants killed: freeze keeps the whole reserve (a: 0 creates admitted); freeze returns mapped
// pages (a: the adopters' tokens change once the create writes into them); the overhead verb
// keeps the block-count arithmetic (b); a page pool that needs more than the verbs report (c).

void Cell78LegacyHeaderSentences() {
	const Fixture& fx = GetFixture("pkv_def");
	if (!fx.ok) return;
	{  // (a)
		LegacyPool pool(fx.model, 2);
		sslm_prefix a = nullptr, b = nullptr;
		PKV_CHECK_EQ(sslm_prefix_begin(fx.model, &pool.pool, &a), SSLM_OK);
		PKV_CHECK_EQ(sslm_prefix_begin(fx.model, &pool.pool, &b), SSLM_OK);
		if (a && b) {
			const std::vector<int32_t> prompt = Stream(51, 1000, fx.vocab);
			PKV_CHECK_EQ(PrefixPrefillAll(fx.model, a, prompt, 64), SSLM_OK);
			PKV_CHECK_EQ(PrefixPrefillAll(fx.model, b, prompt, 64), SSLM_OK);
			sslm_status refusal = SSLM_OK;
			PKV_CHECK_EQ(CountLegacyCreates(fx.model, &pool.pool, &refusal), 0);  // both blocks' pages held
			PKV_CHECK_EQ(refusal, SSLM_KV_POOL_EXHAUSTED);
			PKV_CHECK_EQ(sslm_prefix_freeze(a), SSLM_OK);
			PKV_CHECK_EQ(CountLegacyCreates(fx.model, &pool.pool), 0);  // 256 - 63 + 0 = 193 free
			PKV_CHECK_EQ(sslm_prefix_freeze(b), SSLM_OK);
			// kills: freeze keeps the whole reserve (today's whole-block prefix admits 0)
			PKV_CHECK_EQ(CountLegacyCreates(fx.model, &pool.pool), 1);  // 386 free: floor(386 / 256)
			sslm_seq writer = nullptr;
			PKV_CHECK_EQ(sslm_seq_create(fx.model, &pool.pool, &writer), SSLM_OK);
			if (writer) {
				PKV_CHECK_EQ(PrefillAll(fx.model, writer, Stream(7, 300, fx.vocab), 64), SSLM_OK);
				sslm_seq_release(writer);
			}
			// A second pool holds the adopters (a whole_reserve adopter copies across pools, §3.5).
			LegacyPool other(fx.model, 2);
			const RefRecord* d8 = RefLookup(Reference("v1.11.0", fx), "prefix_lengths", "p1000_decode8");
			for (sslm_prefix px : {a, b}) {
				sslm_seq s = nullptr;
				PKV_CHECK_EQ(sslm_seq_create(fx.model, &other.pool, &s), SSLM_OK);
				if (!s) continue;
				PKV_CHECK_EQ(sslm_seq_adopt_prefix(s, px), SSLM_OK);
				// kills: freeze returns mapped pages, which the writer then overwrote
				PKV_CHECK_MSG(d8 && NextTokens(fx.model, s, 8) == d8->tokens, "7.8 (a): a frozen prefix's rows changed");
				sslm_seq_release(s);
			}
		}
		if (a) sslm_prefix_release(a);
		if (b) sslm_prefix_release(b);
	}
	// (b), at both caps.
	for (const char* stem : {"pkv_def", "pkv_32k"}) {
		const Fixture& f = GetFixture(stem);
		if (!f.ok) continue;
		const uint64_t threshold = (uint64_t{UINT32_MAX} / static_cast<uint64_t>(f.CapPages())) + 1;  // first N over
		PKV_CHECK_EQ(threshold, f.geo.context_cap == 32768 ? 2097152 : 16777216);
		// kills: the overhead verb keeps counting blocks, not pages
		PKV_CHECK_MSG(sslm_kv_pool_overhead_size(f.model, static_cast<uint32_t>(threshold)) == SIZE_MAX,
		              "7.8 (b) %s: overhead at block_count %llu does not saturate", stem,
		              static_cast<unsigned long long>(threshold));
		PKV_CHECK_MSG(sslm_kv_pool_overhead_size(f.model, static_cast<uint32_t>(threshold - 1)) != SIZE_MAX,
		              "7.8 (b) %s: overhead saturates one block below the threshold", stem);
	}
	{  // (c)
		PKV_CHECK_EQ(sslm_kv_block_size(fx.model), fx.PageBytes() * static_cast<size_t>(fx.CapPages()));
		for (uint32_t n : {1u, 3u}) {
			const size_t exact = sslm_kv_block_size(fx.model) * n + sslm_kv_pool_overhead_size(fx.model, n);
			AlignedBuf mem(exact);
			sslm_kv_pool p = OutSentinel<sslm_kv_pool>();  // non-null; the verb must null it on the refusal
			PKV_CHECK_EQ(sslm_kv_pool_create(fx.model, mem.p, exact - 1, n, &p), SSLM_BUFFER_TOO_SMALL);
			PKV_CHECK(p == nullptr);  // kills: a refusal that returns before writing *out
			if (p && p != OutSentinel<sslm_kv_pool>()) sslm_kv_pool_destroy(p);
			p = nullptr;
			// kills: a page pool needing more bytes than sslm_kv_block_size and the overhead verb name
			PKV_CHECK_EQ(sslm_kv_pool_create(fx.model, mem.p, exact, n, &p), SSLM_OK);
			if (p) {
				PKV_CHECK_EQ(CountLegacyCreates(fx.model, &p), n);
				sslm_kv_pool_destroy(p);
			}
		}
	}
}

// ---- 7.10 [C4 legacy] ------------------------------------------------------------------------------
//
// Hot path (§3.6 rule; §3.3 "the hot path never takes the pool mutex"): a legacy sequence with a
// sized workspace decodes across a page boundary. Prefilled to 15 positions, it emits its ready
// token (no row), then decodes whole tokens that write rows 15 (page 0), 16 (the first row of
// page 1: the mapping call), 17, ... 33 (a second boundary at 32). Each decode call is measured
// alone: the pool-mutex acquisition seam counter must not move. Allocations are counted by the
// counting operator new above, the counter the existing zero-allocation cell reads. That cell's
// zero is the ready_for_logits path only; a whole decode token allocates the engine's own
// scratch, a disclosed data-independent count (dim7_contract_red.cpp's law (b); 81 per token on
// pkv_def with this workspace at v1.11.0). So "0 allocations on the mapping call" is graded as
// the map's own share: a decode that maps a page (rows 16 and 32) allocates exactly what every
// decode that maps none allocates, and the ready-token call allocates 0 as the existing cell
// asserts. Then one-token prefills cross the next boundary (row 48) under the mutex counter.
// Mutants killed: "the decode maps through the pool" (the counter moves by >= 1 at row 16), and
// "the map grows the table or reserve with an allocation" (rows 16 and 32 count one more than
// their neighbours). The budget half (C5) repeats this on a budget holder.

void Cell710LegacyHotPath() {
	const Fixture& fx = GetFixture("pkv_def");
	if (!fx.ok) return;
	LegacyPool pool(fx.model, 1);
	const sslm_config cfg{1, 64, static_cast<int32_t>(fx.geo.layers), 0u};
	const size_t wsz = sslm_workspace_size(fx.model, &cfg);
	AlignedBuf ws_mem(wsz);
	sslm_workspace ws = nullptr;
	PKV_CHECK_EQ(sslm_workspace_create(fx.model, &cfg, ws_mem.p, ws_mem.n, &ws), SSLM_OK);
	sslm_seq s = nullptr;
	PKV_CHECK_EQ(sslm_seq_create(fx.model, &pool.pool, &s), SSLM_OK);
	if (!s || !ws) {
		if (s) sslm_seq_release(s);
		if (ws) sslm_workspace_destroy(ws);
		return;
	}
	const std::vector<int32_t> prompt = Stream(71, 15, fx.vocab);
	int32_t consumed = 0;
	PKV_CHECK_EQ(sslm_prefill(fx.model, s, prompt.data(), 15, 64, SSLM_SPAN_PROMPT, ws, &consumed), SSLM_OK);
	sslm_decode_params p{};
	p.layer_budget = static_cast<int32_t>(fx.geo.layers);  // one whole token per call
	sslm_seq batch[1] = {s};
	// The ready token, then whole tokens writing rows 15 .. 33.
	long long interior = -1;  // the allocation count of a decode that maps no page
	for (int64_t row = 14; row <= 33; ++row) {
		int32_t tok = -1;
		const uint64_t m0 = sslm_pkv_test_only_pool_mutex_acquisitions(pool.pool);
		const long long a0 = g_new_calls.load(std::memory_order_relaxed);
		const sslm_status st = sslm_decode_step(fx.model, batch, 1, &p, ws, &tok);
		const long long allocs = g_new_calls.load(std::memory_order_relaxed) - a0;
		const uint64_t mutex = sslm_pkv_test_only_pool_mutex_acquisitions(pool.pool) - m0;
		PKV_CHECK_MSG(st == SSLM_OK && tok >= 0, "7.10 decode writing row %lld: status %d", static_cast<long long>(row + 1),
		              static_cast<int>(st));
		const int64_t written = row == 14 ? -1 : row;  // the first call is the ready token: no row
		// kills: the decode maps its page through the pool (takes the pool mutex)
		char label[64];
		if (written < 0) std::snprintf(label, sizeof label, "the ready-token call");
		else std::snprintf(label, sizeof label, "decode writing row %lld%s", static_cast<long long>(written),
		                   written % fx.B() == 0 ? " (a page boundary)" : "");
		PKV_CHECK_MSG(mutex == 0, "7.10 %s: %llu pool-mutex acquisitions", label, static_cast<unsigned long long>(mutex));
		if (written < 0) {
			PKV_CHECK_MSG(allocs == 0, "7.10 the ready-token call: %lld allocations (the existing cell's zero)", allocs);
		} else if (written % fx.B() != 0) {
			if (interior < 0) interior = allocs;
			PKV_CHECK_MSG(allocs == interior, "7.10 decode writing row %lld: %lld allocations, row 15 made %lld",
			              static_cast<long long>(written), allocs, interior);
		} else {
			// kills: the map allocates (table or reserve grown on the hot path)
			PKV_CHECK_MSG(interior >= 0 && allocs == interior,
			              "7.10 decode writing row %lld (a page boundary): %lld allocations, a decode mapping no page makes %lld",
			              static_cast<long long>(written), allocs, interior);
		}
	}
	// The sequence rests at context_length 34; one-token prefills write rows 34 .. 49 (48 starts
	// page 3).
	for (int32_t i = 0; i < 16; ++i) {
		const int32_t tok = (i * 37 + 11) % fx.vocab;
		const uint64_t m0 = sslm_pkv_test_only_pool_mutex_acquisitions(pool.pool);
		const sslm_status st = sslm_prefill(fx.model, s, &tok, 1, 64, SSLM_SPAN_PROMPT, ws, &consumed);
		const uint64_t mutex = sslm_pkv_test_only_pool_mutex_acquisitions(pool.pool) - m0;
		PKV_CHECK_MSG(st == SSLM_OK && consumed == 1, "7.10 prefill of row %d: status %d", 34 + i, static_cast<int>(st));
		// kills: the prefill maps its pages through the pool
		PKV_CHECK_MSG(mutex == 0, "7.10 prefill writing row %d: %llu pool-mutex acquisitions", 34 + i,
		              static_cast<unsigned long long>(mutex));
	}
	sslm_seq_release(s);
	sslm_workspace_destroy(ws);
}

// ---- 7.11 [C4] -------------------------------------------------------------------------------------
//
// What stays at C4 after the carrier row RB16-CELLS: the exact SSB6 size, 172 + residual +
// 4 * history + 8 + L' * bytes_per_token, and its [0, cap) mutant, run on a budget holder's save
// (C5), because no legacy holder writes SSB6. A legacy holder writes 1.9.0's SSB5, whose exact
// size is 1.9.0's formula, read at R0 from v1.11.0's sslm_seq_save: 156 + residual + 4 * history +
// 4 + block_size, the whole block. This cell pins that at the same L' values {0, 1, B, 1,000},
// resting and mid-token, with and without an anti-LM history (damped greedy), and reads
// kv_block_count = 1 and the history count off the blob.
// Mutant killed: a legacy holder that saves only [0, L') (an SSB6-shaped K/V section under the
// SSB5 magic, or the SSB6 writer chosen for a whole_reserve holder). The [0, cap) mutant is the
// correct SSB5 size here, so it does not separate at C4 (the plan says so); it separates at C5.

void ExpectSsb5Size(const Fixture& fx, sslm_seq s, int64_t want_lprime, const char* what) {
	std::vector<uint8_t> b;
	PKV_CHECK(SaveBlob(s, &b));
	PKV_CHECK_MSG(IsMagic(b, "SSB5"), "7.11 %s: magic", what);
	const uint64_t history = Le64(b, 112);
	PKV_CHECK_EQ(BlobLPrime(b), want_lprime);
	// kills: a legacy save that gathers only [0, L') or writes the SSB6 layout
	PKV_CHECK_MSG(b.size() == Ssb5Size(fx, history), "7.11 %s: SSB5 size %zu, 1.9.0's formula gives %zu", what, b.size(),
	              Ssb5Size(fx, history));
	PKV_CHECK_EQ(Le32(b, kSsb5Header + HiddenSize(fx) + 4 * static_cast<size_t>(history)), 1);  // kv_block_count
}

void Cell711LegacySsb5Size() {
	const Fixture& fx = GetFixture("pkv_def");
	if (!fx.ok) return;
	LegacyPool pool(fx.model, 1);
	struct Case {
		int64_t lprime;
		bool mid;
	};
	// L' = 1 mid-token does not exist (a fresh holder has no token to embed), so mid-token runs
	// at B and 1,000: prefill L' - 1, emit the ready token, run one layer of the next.
	for (const Case c : {Case{0, false}, Case{1, false}, Case{fx.B(), false}, Case{fx.B(), true}, Case{1000, false},
	                     Case{1000, true}}) {
		sslm_seq s = nullptr;
		PKV_CHECK_EQ(sslm_seq_create(fx.model, &pool.pool, &s), SSLM_OK);
		if (!s) continue;
		char what[64];
		std::snprintf(what, sizeof what, "L'=%lld %s", static_cast<long long>(c.lprime), c.mid ? "mid-token" : "resting");
		const int64_t prefill = c.mid ? c.lprime - 1 : c.lprime;
		if (prefill > 0) PKV_CHECK_EQ(PrefillAll(fx.model, s, Stream(81, static_cast<int32_t>(prefill), fx.vocab), 64), SSLM_OK);
		if (c.mid) EnterMidToken(fx.model, s);
		ExpectSsb5Size(fx, s, c.lprime, what);
		sslm_seq_release(s);
	}
	// With an anti-LM history: damped greedy from a 40-token prompt, 24 whole tokens, saved
	// resting and then mid-token.
	sslm_seq s = nullptr;
	PKV_CHECK_EQ(sslm_seq_create(fx.model, &pool.pool, &s), SSLM_OK);
	if (!s) return;
	PKV_CHECK_EQ(PrefillAll(fx.model, s, Stream(82, 40, fx.vocab), 64), SSLM_OK);
	sslm_decode_params p{};
	PKV_CHECK_EQ(sslm_decode_params_init(fx.model, SSLM_DECODE_MODE_DAMPED_GREEDY, static_cast<int32_t>(fx.geo.layers), &p),
	             SSLM_OK);
	sslm_seq b[1] = {s};
	for (int i = 0; i < 24; ++i) {
		int32_t tok = -1;
		PKV_CHECK_EQ(sslm_decode_step_v2(fx.model, b, 1, &p, nullptr, &tok), SSLM_OK);
	}
	std::vector<uint8_t> blob;
	PKV_CHECK(SaveBlob(s, &blob));
	PKV_CHECK_MSG(Le64(blob, 112) > 0, "7.11: the damped run recorded no anti-LM history");
	ExpectSsb5Size(fx, s, BlobLPrime(blob), "damped resting");
	p.layer_budget = 1;
	int32_t tok = -1;
	PKV_CHECK_EQ(sslm_decode_step_v2(fx.model, b, 1, &p, nullptr, &tok), SSLM_OK);
	PKV_CHECK(SaveBlob(s, &blob));
	PKV_CHECK_EQ(Le32(blob, 68), 1);
	ExpectSsb5Size(fx, s, BlobLPrime(blob), "damped mid-token");
	sslm_seq_release(s);
}

// ---- 7.12 [C4 whole_reserve] -----------------------------------------------------------------------
//
// The E clamp (§3.7): a private restore draws R(budget) + E, E = min(floor(origin/B),
// ceil(cap/B) - R(budget)). For whole_reserve, R(cap) = ceil(cap/B), so E = 0. A hand-built SSB6
// whole_reserve blob with origin 1,000 (no library writer emits one, rev 16) restores into a
// one-block (256-page) pool with SSLM_OK and continues equal to v1.11.0; N such blobs are admitted
// by an N-block pool and the (N+1)th is refused. The blob is built from a legacy adopter of the
// 1,000-token prefix, whose own SSB5 first equals v1.11.0's record, so its rows are the
// reference's.
// Mutants killed: the unclamped E = floor(1000/16) = 62 (318 pages: refused in 256, and in an
// N-block pool only floor(256N / 318) are admitted); a whole_reserve restore that draws fewer than
// ceil(cap/B) pages (the (N+1)th is admitted).

void Cell712WholeReserveClamp() {
	const Fixture& fx = GetFixture("pkv_def");
	if (!fx.ok) return;
	const RefFile& ref = Reference("v1.11.0", fx);
	const int64_t origin = 1000;
	const int64_t E = std::min<int64_t>(origin / fx.B(), fx.CapPages() - fx.R(fx.geo.context_cap));
	const int64_t E_unclamped = origin / fx.B();
	PKV_CHECK_EQ(E, 0);
	PKV_CHECK_EQ(fx.R(fx.geo.context_cap) + E_unclamped, 318);  // the mutant's draw, > 256
	std::vector<uint8_t> ssb5;
	{
		LegacyPool pool(fx.model, 2);
		sslm_prefix px = nullptr;
		sslm_seq s = nullptr;
		PKV_CHECK_EQ(sslm_prefix_begin(fx.model, &pool.pool, &px), SSLM_OK);
		if (px) PKV_CHECK_EQ(PrefixPrefillAll(fx.model, px, Stream(51, 1000, fx.vocab), 64), SSLM_OK);
		if (px) PKV_CHECK_EQ(sslm_prefix_freeze(px), SSLM_OK);
		PKV_CHECK_EQ(sslm_seq_create(fx.model, &pool.pool, &s), SSLM_OK);
		if (s && px) PKV_CHECK_EQ(sslm_seq_adopt_prefix(s, px), SSLM_OK);
		if (s) PKV_CHECK(SaveBlob(s, &ssb5));
		if (s) sslm_seq_release(s);
		if (px) sslm_prefix_release(px);
	}
	const RefRecord* adopted = RefLookup(ref, "prefix_lengths", "p1000_adopted");
	const RefRecord* d8 = RefLookup(ref, "prefix_lengths", "p1000_decode8");
	const RefRecord* last = RefLookup(ref, "prefix_lengths", "p1000_prefill20+decode4");
	PKV_CHECK_MSG(adopted && BlobSha(ssb5) == adopted->blob_sha, "7.12: the source SSB5 is not v1.11.0's");
	const std::vector<uint8_t> blob = Ssb6WholeReserveFromSsb5(ssb5, fx, origin);
	PKV_CHECK(IsMagic(blob, "SSB6"));
	{  // one block
		LegacyPool pool(fx.model, 1);
		sslm_seq r = nullptr;
		// kills: the unclamped E (256 + 62 pages wanted in a 256-page pool)
		PKV_CHECK_EQ(Restore(fx, &pool.pool, blob, &r), SSLM_OK);
		if (r) {
			PKV_CHECK_EQ(CountLegacyCreates(fx.model, &pool.pool), 0);
			PKV_CHECK_MSG(d8 && NextTokens(fx.model, r, 8) == d8->tokens, "7.12: continuation differs from v1.11.0");
			std::vector<uint8_t> resaved;
			PKV_CHECK(SaveBlob(r, &resaved));
			// The restored whole_reserve holder writes 1.9.0's SSB5, byte-equal to v1.11.0's.
			PKV_CHECK_MSG(d8 && BlobSha(resaved) == d8->blob_sha, "7.12: re-save after decode8 differs from v1.11.0");
			PKV_CHECK_EQ(PrefillAll(fx.model, r, Stream(52, 20, fx.vocab), 64), SSLM_OK);
			PKV_CHECK_MSG(last && NextTokens(fx.model, r, 4) == last->tokens, "7.12: prefill20+decode4 differs");
			sslm_seq_release(r);
		}
	}
	for (uint32_t n : {2u, 4u}) {  // N blobs in an N-block pool
		LegacyPool pool(fx.model, n);
		std::vector<sslm_seq> live;
		for (uint32_t i = 0; i < n; ++i) {
			sslm_seq r = nullptr;
			// kills: the unclamped E (floor(256N / 318) < N restores fit)
			PKV_CHECK_MSG(Restore(fx, &pool.pool, blob, &r) == SSLM_OK, "7.12: restore %u of %u refused", i + 1, n);
			if (r) live.push_back(r);
		}
		sslm_seq extra = OutSentinel<sslm_seq>();  // non-null; the verb must null it on the refusal
		// kills: a whole_reserve restore that reserves fewer than ceil(cap/B) pages
		PKV_CHECK_EQ(Restore(fx, &pool.pool, blob, &extra), SSLM_KV_POOL_EXHAUSTED);
		PKV_CHECK(extra == nullptr);  // kills: a refusal that returns before writing *out
		if (extra && extra != OutSentinel<sslm_seq>()) sslm_seq_release(extra);
		for (sslm_seq r : live) PKV_CHECK_MSG(d8 && NextTokens(fx.model, r, 8) == d8->tokens, "7.12: N-block continuation");
		for (sslm_seq r : live) sslm_seq_release(r);
	}
}

// ---- 7.13 [C4 legacy ABI] --------------------------------------------------------------------------
//
// "Table entries at or above `mapped` are never read." 1.1's legacy half -- one handle driven
// fresh -> prefill 100 -> reset -> prefill 40 -> reset -> adopt -> decode, saved at every stage
// (the lifecycle scenario of R0) -- runs with the table sentinel on, so every table entry at or
// above `mapped` holds 0xFFFFFFFF (rewritten on every unmap, so a warm table traps as a fresh one
// does) and the address path traps on it. A fresh holder's save (mapped 0) is added in front.
// The correct build finishes and equals v1.11.0 record for record. The run is made in a forked
// child (POSIX), so the trap is graded by exit status: exit 0 is green; a SIGABRT is the trap.
// Mutant killed: "the save gather reads one entry past `mapped`", at the fresh stage and again
// after the reset stage (prefill 100, reset, prefill 40): the child traps both times. On a host
// without fork (MSVC on the box) the run is in-process, where a trap ends the executable.

int RunLifecycleUnderSentinel(const Fixture& fx) {
	const int before = State().failures;
	sslm_pkv_test_only_set_table_sentinel(1);
	{
		LegacyPool pool(fx.model, 1);
		sslm_seq s = nullptr;
		PKV_CHECK_EQ(sslm_seq_create(fx.model, &pool.pool, &s), SSLM_OK);
		if (s) {
			std::vector<uint8_t> b;
			PKV_CHECK(SaveBlob(s, &b));  // mapped 0: the gather may read no entry at all
			PKV_CHECK_EQ(b.size(), Ssb5Size(fx, 0));
			sslm_seq_release(s);
		}
	}
	ExpectMatchesReference("v1.11.0", fx, "lifecycle", RunScenario(fx, "lifecycle"), Compare::kTokensRowsAndBlob);
	sslm_pkv_test_only_set_table_sentinel(0);
	return State().failures - before;
}

void Cell713LegacySentinel() {
	const Fixture& fx = GetFixture("pkv_def");
	if (!fx.ok) return;
	(void)Reference("v1.11.0", fx);  // load before the fork, so a missing file fails here
#if defined(_WIN32)
	RunLifecycleUnderSentinel(fx);
#else
	std::fflush(stdout);
	std::fflush(stderr);
	const pid_t pid = fork();
	PKV_CHECK_MSG(pid >= 0, "7.13: fork failed");
	if (pid == 0) {
		const int failures = RunLifecycleUnderSentinel(fx);
		std::fflush(stderr);
		_exit(failures == 0 ? 0 : 3);
	}
	if (pid < 0) return;
	int status = 0;
	PKV_CHECK(waitpid(pid, &status, 0) == pid);
	// kills: the save gather (or any address path) reads an entry at or past `mapped`
	PKV_CHECK_MSG(!WIFSIGNALED(status), "7.13: the lifecycle run trapped under the table sentinel (signal %d)",
	              WIFSIGNALED(status) ? WTERMSIG(status) : 0);
	PKV_CHECK_MSG(WIFSIGNALED(status) || (WIFEXITED(status) && WEXITSTATUS(status) == 0),
	              "7.13: the lifecycle run under the sentinel failed its reference checks (exit %d; see the child's FAIL lines)",
	              WIFEXITED(status) ? WEXITSTATUS(status) : -1);
#endif
}

PKV_CELL("7.6/C4", "C4", Cell76WholeReserveResident);
PKV_CELL("7.8/C4", "C4", Cell78LegacyHeaderSentences);
PKV_CELL("7.10/C4", "C4", Cell710LegacyHotPath);
PKV_CELL("7.11/C4", "C4", Cell711LegacySsb5Size);
PKV_CELL("7.12/C4", "C4", Cell712WholeReserveClamp);
PKV_CELL("7.13/C4", "C4", Cell713LegacySentinel);

}  // namespace

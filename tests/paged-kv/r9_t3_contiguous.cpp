// Paged-KV plan rev 16.21 §23.8 step T3, the contiguous pool option (§21.2, D-SLM8398):
// sslm_kv_pool_contiguous_size and sslm_kv_pool_create_contiguous, a pool whose page is `cap`
// positions, one page per sequence, through the same page module.
//
//   4.14/T3:bit-identity   tokens and K/V rows equal the page pool's at 4.1's widths, cap - 1 and cap
//   4.14/T3:batch          batch decode with the option: two contiguous holders, and a page holder beside a
//                          contiguous one in both orders, against the n = 1 twin (rev 16.24, D-SLM8441)
//   4.4/T3:batch           the same at pkv_odd across the partial last page, then the cap refusal
//   4.14/T3:admission      N blocks admit exactly N legacy and N budget creates of any budget
//   4.14/T3:sizing         the size verb equals the test's N * cap * q + 8N + 32
//   4.14/T3:saves          SSB6 and SSB5 save and restore across pool kinds with rows equal
//   4.14/T3:adopt          adopt and begin_from copy; one page per holder
//   4.14/T3:lifecycle      the fill probe across create, release, reset and re-adopt; pool-kind reads
//                          (stats B = cap, page_count N; the model-level verbs answer the derived B);
//                          freeze returns no reserve page; restore_shared at origin 1,200 shares 0 pages
//   4.14/T3:hostile        refusal order, the overflow pair 3,054,198,961 / 3,054,198,962 (L-3) at the
//                          Qwen3-8B geometry with Int16 K/V at cap 40,960, null out and *out nulling (L-6)
//   4.14/T3:reservation    a budget holder created, and one restored, in the option reserves exactly one page
//                          (T-3195's lesser plane; mutant: reserving by the model's B)
//   5.6/T3                 allocation-failure injection in sslm_kv_pool_create_contiguous
//   7.8/T3                 the public header declares both verbs (their behaviour sentences are 4.14's)
//   6.7/T3:option          a blob saved in the option restores in the page pool and back (pkv_def)
//   6.7/T3:non-dividing    the same at cap 4,100 (pkv_odd), and the legacy save byte-equal across kinds
//   4.15/T3                copy adopt across page geometries refused in both directions, nothing changed
//                          (rev 16.23, D-SLM8434); the 0.5B member is 4.15/box
//   7.9/C6:fixtures        the cloud twin of cell 7.9: pkv_def in a page pool against pkv_def in the
//                          option, per-token random order, one blob restored into both arms; the token
//                          streams must be equal (timings are recorded, never graded in the cloud)
//
// On 2ebe5e3 both verbs are the T0 stubs (tests/paged-kv/pkv_r9_stubs.cpp), so every cell fails at
// its first use of the option.

#include "pkv_r9.h"
#include "pkv_r9_adopt415.h"

#include "support/bad_alloc_injection.h"

#include <chrono>
#include <fstream>
#include <random>

namespace {

using namespace pkv;
using namespace pkv::r9;

constexpr int64_t kCap = 4096;

const Fixture& Def() { return GetFixture("pkv_def"); }
int64_t DefB() { return RuleB(48, kCap, 1); }  // 128
Lay DefLay(int64_t B) { return LayoutOf(Geometry{2, 2, 48, kCap}, 1, B); }

size_t ContigBytes(const Lay& y, uint32_t n) {
	bool fits = false;
	return static_cast<size_t>(ContiguousSize(y, n, &fits));
}

// Prefill W of the widths stream, ready token, and (below the cap) one decode; rows [0, L) read from
// the holder's page(s) at page size `B` (the option's B is the cap: one page, slot r).
struct ArmRun {
	AbiWidth w;
	std::vector<uint8_t> rows;
	bool rows_ok = false;
};
ArmRun RunArm(const Fixture& fx, sslm_kv_pool* pool, int64_t B, int64_t W) {
	ArmRun r;
	sslm_seq s = nullptr;
	if (sslm_seq_create(fx.model, pool, &s) != SSLM_OK) return r;
	r.w = RunAbiWidth(fx.model, s, W, fx.vocab, W < kCap);
	r.rows_ok = SeqRows(*pool, s, fx.geo, 1, B, W < kCap ? W + 1 : W, Rot::kRotated, &r.rows);
	sslm_seq_release(s);
	return r;
}

void Cell414BitIdentity() {
	const Fixture& fx = Def();
	if (!fx.ok) return;
	const int64_t B = DefB();
	const Lay y = DefLay(B);
	for (int64_t W : {int64_t{1}, B - 1, B, B + 1, 2 * B, int64_t{1064}, kCap - 1, kCap}) {
		ContigPoolOf cp(fx.model, 1, ContigBytes(y, 1));
		BlockPoolOf pp(fx.model, 1, y);
		PKV_CHECK_MSG(cp.status == SSLM_OK, "W %lld: the option's create: %s", static_cast<long long>(W), AbiStatusName(cp.status));
		if (!cp.pool || !pp.pool) continue;
		const ArmRun c = RunArm(fx, &cp.pool, kCap, W), p = RunArm(fx, &pp.pool, B, W);
		PKV_CHECK_MSG(c.w.prefill == SSLM_OK && c.w.ready == p.w.ready && c.w.decode1 == p.w.decode1,
		              "W %lld: tokens differ between the option and the page pool", static_cast<long long>(W));
		PKV_CHECK_MSG(c.rows_ok && p.rows_ok && c.rows == p.rows, "W %lld: K/V rows differ between pool kinds",
		              static_cast<long long>(W));
	}
}

void Cell414Admission() {
	const Fixture& fx = Def();
	if (!fx.ok) return;
	const Lay y = DefLay(DefB());
	for (uint32_t N : {1u, 3u})
		for (int32_t budget : {0, 1, 100, 4096}) {
			ContigPoolOf cp(fx.model, N, ContigBytes(y, N));
			PKV_CHECK_MSG(cp.status == SSLM_OK, "N %u: the option's create: %s", N, AbiStatusName(cp.status));
			if (!cp.pool) continue;
			sslm_status refusal = SSLM_OK;
			const int n = CountCreates(fx.model, &cp.pool, budget, &refusal);
			// kills: the option ignored (derived B = 128): budget-1 holders reserve 2 of 32 pages per block
			PKV_CHECK_MSG(n == static_cast<int>(N) && refusal == SSLM_KV_POOL_EXHAUSTED,
			              "%u blocks admit %d %s creates (budget %d), refused %s; %u owed", N, n, budget ? "budget" : "legacy",
			              budget, AbiStatusName(refusal), N);
		}
	PKV_CHECK(32 / Reservation(1, kCap, DefB()) > 1);  // separation: the mutant's budget-1 count per block
}

void Cell414Sizing() {
	const Fixture& fx = Def();
	if (!fx.ok) return;
	const Lay y = DefLay(DefB());
	for (uint32_t N : {1u, 2u, 7u}) {
		const size_t want = N * kCap * y.bytes_per_position + 8 * N + 32;
		PKV_CHECK_MSG(sslm_kv_pool_contiguous_size(fx.model, N) == want, "contiguous_size(%u) = %zu, %zu owed", N,
		              sslm_kv_pool_contiguous_size(fx.model, N), want);
	}
}

void Cell414Saves() {
	const Fixture& fx = Def();
	if (!fx.ok) return;
	const int64_t B = DefB();
	const Lay y = DefLay(B);
	for (int32_t budget : {0, 1200}) {
		for (int dir = 0; dir < 2; ++dir) {
			ContigPoolOf cp(fx.model, 1, ContigBytes(y, 1));
			PagePoolOf pp(fx.model, static_cast<uint32_t>(y.pages), y.page_bytes);
			if (!cp.pool || !pp.pool) {
				PKV_CHECK_MSG(false, "pools: option %s, page %s", AbiStatusName(cp.status), AbiStatusName(pp.status));
				continue;
			}
			sslm_kv_pool* src = dir == 0 ? &cp.pool : &pp.pool;
			sslm_kv_pool* dst = dir == 0 ? &pp.pool : &cp.pool;
			const int64_t dstB = dir == 0 ? B : kCap;
			sslm_seq s = nullptr, r = nullptr;
			PKV_CHECK_EQ(budget ? sslm_seq_create_budgeted(fx.model, src, budget, &s) : sslm_seq_create(fx.model, src, &s), SSLM_OK);
			if (!s) continue;
			PKV_CHECK_EQ(PrefillAll(fx.model, s, Stream(31, 1100, fx.vocab), 64), SSLM_OK);
			bc::DecodeN(fx.model, s, 4);
			const std::vector<uint8_t> blob = bc::Save(s);
			PKV_CHECK(IsMagic(blob, budget ? "SSB6" : "SSB5"));
			PKV_CHECK_EQ(sslm_seq_restore(fx.model, dst, blob.data(), blob.size(), &r), SSLM_OK);
			if (r) {
				std::vector<uint8_t> rows, brows;
				PKV_CHECK_MSG(SeqRows(*dst, r, fx.geo, 1, dstB, 1103, Rot::kRotated, &rows) &&
				                  BlobRows(blob, fx, HiddenSize(fx), 1103, &brows) && rows == brows,
				              "%s %s: restored rows equal the blob's", budget ? "SSB6" : "SSB5",
				              dir == 0 ? "option -> page pool" : "page pool -> option");
				PKV_CHECK(bc::DecodeN(fx.model, r, 4) == bc::DecodeN(fx.model, s, 4));
				sslm_seq_release(r);
			}
			sslm_seq_release(s);
		}
	}
}

void Cell414Adopt() {
	const Fixture& fx = Def();
	if (!fx.ok) return;
	const Lay y = DefLay(DefB());
	ContigPoolOf cp(fx.model, 4, ContigBytes(y, 4));
	PKV_CHECK_MSG(cp.status == SSLM_OK, "the option's create: %s", AbiStatusName(cp.status));
	if (!cp.pool) return;
	sslm_prefix px = bc::BudgetPrefix(fx, &cp.pool, Stream(51, 1000, fx.vocab));
	sslm_seq s = nullptr;
	sslm_prefix child = nullptr;
	PKV_CHECK_EQ(sslm_seq_create_budgeted(fx.model, &cp.pool, 64, &s), SSLM_OK);
	if (!px || !s) return;
	PKV_CHECK_EQ(sslm_seq_adopt_prefix(s, px), SSLM_OK);
	PKV_CHECK_EQ(sslm_prefix_begin_from(px, 64, &child), SSLM_OK);
	uint32_t pp = 0, sp = 0, cpg = 0, sm = 0, cm = 0;
	sslm_pkv_test_only_prefix_table_entry(px, 0, &pp, nullptr);
	PKV_CHECK_EQ(sslm_pkv_test_only_seq_table_entry(s, 0, &sp, &sm), SSLM_OK);
	if (child) PKV_CHECK_EQ(sslm_pkv_test_only_prefix_table_entry(child, 0, &cpg, &cm), SSLM_OK);
	PKV_CHECK_MSG(sm == 1 && sp != pp, "adopt in the option maps %u page(s), the prefix's page shared: %d", sm, sp == pp);
	PKV_CHECK_MSG(cm == 1 && cpg != pp, "begin_from in the option maps %u page(s), the parent's page shared: %d", cm, cpg == pp);
	std::vector<uint8_t> a, c, p;
	std::vector<uint32_t> tp = {pp}, tc = {cpg};
	PKV_CHECK(SeqRows(cp.pool, s, fx.geo, 1, kCap, 1000, Rot::kRotated, &a) &&
	          RowsFromTable(cp.pool, tp, fx.geo, 1, kCap, 1000, Rot::kRotated, &p) && a == p);
	if (child) PKV_CHECK(RowsFromTable(cp.pool, tc, fx.geo, 1, kCap, 1000, Rot::kRotated, &c) && c == p);
	// the fill probe: three holders of one page each leave exactly one of four
	sslm_status refusal = SSLM_OK;
	PKV_CHECK_EQ(CountCreates(fx.model, &cp.pool, 1, &refusal), 1);
	if (child) sslm_prefix_release(child);
	sslm_seq_release(s);
	sslm_prefix_release(px);
}

void Cell414Lifecycle() {
	const Fixture& fx = Def();
	if (!fx.ok) return;
	const Lay y = DefLay(DefB());
	const uint32_t N = 3;
	ContigPoolOf cp(fx.model, N, ContigBytes(y, N));
	PKV_CHECK_MSG(cp.status == SSLM_OK, "the option's create: %s", AbiStatusName(cp.status));
	if (!cp.pool) return;
	auto probe = [&](const char* when) {
		sslm_status refusal = SSLM_OK;
		const int n = CountCreates(fx.model, &cp.pool, 1, &refusal);
		PKV_CHECK_MSG(n == static_cast<int>(N), "%s: the fill probe admits %d, %u owed", when, n, N);
	};
	probe("fresh");
	sslm_prefix px = bc::BudgetPrefix(fx, &cp.pool, Stream(51, 300, fx.vocab));
	for (int32_t budget : {1, 512, 4096}) {
		sslm_seq s = nullptr;
		PKV_CHECK_EQ(sslm_seq_create_budgeted(fx.model, &cp.pool, budget, &s), SSLM_OK);
		if (!s) continue;
		PrefillAll(fx.model, s, Stream(9, std::min(budget, 64), fx.vocab), 64);
		PKV_CHECK_EQ(sslm_seq_reset(s), SSLM_OK);
		if (px) PKV_CHECK_EQ(sslm_seq_adopt_prefix(s, px), SSLM_OK);
		sslm_seq_release(s);
	}
	if (px) sslm_prefix_release(px);
	probe("after create, reset, re-adopt and release across budget shapes");
	sslm_kv_pool_stats_out ps{};
	ps.struct_size = sizeof ps;
	PKV_CHECK_EQ(sslm_kv_pool_stats(cp.pool, &ps), SSLM_OK);
	PKV_CHECK_MSG(ps.page_positions == kCap && ps.page_count == N, "pool stats: page_positions %lld, page_count %u",
	              static_cast<long long>(ps.page_positions), ps.page_count);  // mutant: stats report the derived B
	PKV_CHECK_EQ(sslm_kv_page_positions(fx.model), DefB());  // model-level: the derived B, not the pool's
	PKV_CHECK_EQ(sslm_kv_page_size(fx.model), y.page_bytes);
	PKV_CHECK_EQ(sslm_kv_pages_for_budget(fx.model, 100), static_cast<size_t>(Reservation(100, kCap, DefB())));
	// freeze returns no reserve page in the option: a legacy prefix holds its one page before and after
	{
		sslm_prefix lp = nullptr;
		PKV_CHECK_EQ(sslm_prefix_begin(fx.model, &cp.pool, &lp), SSLM_OK);
		if (lp) {
			PrefixPrefillAll(fx.model, lp, Stream(51, 100, fx.vocab), 64);
			PKV_CHECK_EQ(sslm_prefix_freeze(lp), SSLM_OK);
			sslm_status refusal = SSLM_OK;
			PKV_CHECK_EQ(CountCreates(fx.model, &cp.pool, 1, &refusal), static_cast<int>(N) - 1);
			sslm_prefix_release(lp);
		}
	}
	// restore_shared at origin 1,200 with a live prefix handle shares floor(1,200 / cap) = 0 pages
	{
		sslm_prefix p12 = bc::BudgetPrefix(fx, &cp.pool, Stream(51, 1200, fx.vocab));
		sslm_seq s = nullptr, r = nullptr;
		PKV_CHECK_EQ(sslm_seq_create_budgeted(fx.model, &cp.pool, 64, &s), SSLM_OK);
		if (p12 && s && sslm_seq_adopt_prefix(s, p12) == SSLM_OK) {
			bc::DecodeN(fx.model, s, 3);
			const std::vector<uint8_t> blob = bc::Save(s);
			sslm_seq_release(s);
			s = nullptr;
			uint32_t shared = 9;
			PKV_CHECK_EQ(sslm_seq_restore_shared(fx.model, &cp.pool, blob.data(), blob.size(), p12, &r, &shared), SSLM_OK);
			PKV_CHECK_EQ(shared, 0u);  // mutant: restore_shared takes the page pool's share path (9 pages)
			if (r) {
				std::vector<uint8_t> rows, brows;
				PKV_CHECK(SeqRows(cp.pool, r, fx.geo, 1, kCap, 1202, Rot::kRotated, &rows) &&
				          BlobRows(blob, fx, HiddenSize(fx), 1202, &brows) && rows == brows);
				sslm_seq_release(r);
			}
		}
		if (s) sslm_seq_release(s);
		if (p12) sslm_prefix_release(p12);
	}
}

void Cell414Hostile() {
	const Fixture& fx = Def();
	if (!fx.ok) return;
	const Lay y = DefLay(DefB());
	const size_t need = ContigBytes(y, 2);
	PKV_CHECK_EQ(sslm_kv_pool_contiguous_size(nullptr, 2), 0u);
	PKV_CHECK_EQ(sslm_kv_pool_contiguous_size(fx.model, 0), 0u);
	AlignedBuf mem(need + 64, 0x77);
	const std::vector<uint8_t> before(mem.p, mem.p + mem.n);
	struct Case { const char* what; sslm_model m; void* buf; size_t size; uint32_t n; sslm_status want; };
	const Case cases[] = {
	    {"block_count 0", fx.model, mem.p, need, 0, SSLM_INVALID_ARGUMENT},
	    {"null model", nullptr, mem.p, need, 2, SSLM_INVALID_ARGUMENT},
	    {"null buffer", fx.model, nullptr, need, 2, SSLM_INVALID_ARGUMENT},
	    {"one byte below", fx.model, mem.p, need - 1, 2, SSLM_BUFFER_TOO_SMALL},
	    {"misaligned", fx.model, mem.p + 1, need, 2, SSLM_MISALIGNED_BUFFER},
	    // both faults: sslm_kv_pool_create's precedence, the size before the alignment (src/sslm_abi.cpp, the
	    // too-small-and-misaligned case); kills an option that checks the alignment first (T-3196 F-4)
	    {"one byte below and misaligned", fx.model, mem.p + 1, need - 1, 2, SSLM_BUFFER_TOO_SMALL},
	};
	for (const Case& c : cases) {
		sslm_kv_pool out = legacy_a::OutSentinel<sslm_kv_pool>();
		const sslm_status st = sslm_kv_pool_create_contiguous(c.m, c.buf, c.size, c.n, &out);
		PKV_CHECK_MSG(st == c.want, "%s: %s, %s owed", c.what, AbiStatusName(st), AbiStatusName(c.want));
		PKV_CHECK_MSG(out == nullptr, "%s: *out is not nulled on the refusal", c.what);
		if (out && out != legacy_a::OutSentinel<sslm_kv_pool>()) sslm_kv_pool_destroy(out);
	}
	PKV_CHECK_MSG(std::equal(before.begin(), before.end(), mem.p), "a refusal wrote into the caller's buffer");
	PKV_CHECK_EQ(sslm_kv_pool_create_contiguous(fx.model, mem.p, need, 2, nullptr), SSLM_INVALID_ARGUMENT);  // null out
	PKV_CHECK_MSG(std::equal(before.begin(), before.end(), mem.p), "the null-out refusal wrote into the caller's buffer");
	// the precedence against the existing create's, on the same both-faults buffer (green on 2ebe5e3: the reading)
	{
		sslm_kv_pool out = legacy_a::OutSentinel<sslm_kv_pool>();
		PKV_CHECK_EQ(sslm_kv_pool_create(fx.model, mem.p + 1, 1, 2, &out), SSLM_BUFFER_TOO_SMALL);
		PKV_CHECK(out == nullptr);
	}
	{
		sslm_kv_pool out = nullptr;
		PKV_CHECK_EQ(sslm_kv_pool_create_contiguous(fx.model, mem.p, need, 2, &out), SSLM_OK);  // exactly the size
		if (out) sslm_kv_pool_destroy(out);
	}
	// The overflow pair (L-3, D-SLM8422): the Qwen3-8B K/V geometry (36 layers, 8 KV heads, D = 128),
	// Int16, cap 40,960: a block of 6,039,797,760 bytes. The sum block_count * (block + 8) + 32 first
	// exceeds 2^64 - 1 at 3,054,198,962; the product alone at 3,054,198,967.
	const Fixture& q8 = Q8();
	if (!q8.ok) return;
	Fixture big;
	Variant v;
	v.cap = 40960;
	v.kv_precision = 1;
	v.strip_rope = true;
	sslm_status ms = SSLM_OK;
	PKV_CHECK_MSG(MakeVariant(q8, v, &big, &ms), "the Qwen3-8B-shaped fixture at Int16, cap 40,960 maps (%s)", AbiStatusName(ms));
	if (!big.ok) return;
	const Lay yq = LayoutOf(big.geo, 2, RuleB(128, 40960, 2));
	PKV_CHECK_EQ(yq.contiguous_block_bytes, 6039797760ull);
	bool fits = false;
	ContiguousSize(yq, 3054198961ull, &fits);
	PKV_CHECK(fits);
	ContiguousSize(yq, 3054198962ull, &fits);
	PKV_CHECK(!fits);
	PKV_CHECK(3054198966ull <= UINT64_MAX / 6039797760ull && 3054198967ull > UINT64_MAX / 6039797760ull);  // the product
	PKV_CHECK_MSG(sslm_kv_pool_contiguous_size(big.model, 3054198961u) != 0, "the size at 3,054,198,961 is 0");
	// At 3,054,198,962 the engine's product N * (block + 8) overflows first; it is checked before the + 32 (D-SLM8499).
	// Kills: that product unchecked, and the plan's sum N * block + (8N + 32) unchecked (§23.3). Not the engine's + 32
	// unchecked: at 961 the product is 6,033,832,567 below 2^64 - 1, so the + 32 cannot carry, and no cell kills it.
	PKV_CHECK_MSG(sslm_kv_pool_contiguous_size(big.model, 3054198962u) == 0, "the size at 3,054,198,962 is not 0");
	// At 3,054,198,967 the bare product N * block overflows (T-3212 C-2). Kills: the plan's product N * block unchecked
	// with 8N + 32 added checked, which reads 0 at 962 to 966 (its checked sum catches them) and a wrapped size from
	// 967 (30,204,954,072 bytes at 967, executed in T-3212 §5.1); and any decomposition whose product is unchecked.
	PKV_CHECK_MSG(sslm_kv_pool_contiguous_size(big.model, 3054198967u) == 0, "the size at 3,054,198,967 is not 0");
	for (uint32_t n : {3054198962u, 3054198967u}) {
		// Where the size verb wrapped (already a failed check above), a create of n blocks over SIZE_MAX would build
		// the pool in this small buffer and corrupt the process, so that create is not attempted.
		if (sslm_kv_pool_contiguous_size(big.model, n) != 0) continue;
		const std::vector<uint8_t> before_overflow(mem.p, mem.p + mem.n);  // the admitted create above wrote its pool
		sslm_kv_pool out = legacy_a::OutSentinel<sslm_kv_pool>();
		const sslm_status st = sslm_kv_pool_create_contiguous(big.model, mem.p, SIZE_MAX, n, &out);
		PKV_CHECK_MSG(st == SSLM_INVALID_ARGUMENT, "the create at %u blocks: %s, SSLM_INVALID_ARGUMENT owed", n,
		              AbiStatusName(st));
		PKV_CHECK_MSG(out == nullptr, "the create at %u blocks: *out is not nulled", n);
		PKV_CHECK_MSG(std::equal(before_overflow.begin(), before_overflow.end(), mem.p),
		              "the overflow refusal at %u blocks wrote into the caller's buffer", n);
	}
}

void Cell56() {
	const Fixture& fx = Def();
	if (!fx.ok) return;
	const Lay y = DefLay(DefB());
	AlignedBuf mem(ContigBytes(y, 2));
	sslm_kv_pool out = legacy_a::OutSentinel<sslm_kv_pool>();
	superslm_test::ArmInjectedFault(superslm_test::InjectThrowKind::kBadAlloc);
	const sslm_status st = sslm_kv_pool_create_contiguous(fx.model, mem.p, mem.n, 2, &out);
	const bool consulted = superslm_test::g_inject_throw == superslm_test::InjectThrowKind::kNone;
	superslm_test::DisarmInjectedFault();
	PKV_CHECK_MSG(st == SSLM_ALLOCATION_FAILED, "armed bad_alloc: %s, SSLM_ALLOCATION_FAILED owed", AbiStatusName(st));
	PKV_CHECK_MSG(consulted, "the create never consulted the injection seam");
	PKV_CHECK(out == nullptr);
	if (out && out != legacy_a::OutSentinel<sslm_kv_pool>()) sslm_kv_pool_destroy(out);
	// nothing allocated: the same buffer then makes a pool that admits both blocks
	ContigPoolOf again(fx.model, 2, mem.n);
	sslm_status refusal = SSLM_OK;
	if (again.pool) PKV_CHECK_EQ(CountCreates(fx.model, &again.pool, 0, &refusal), 2);
	else PKV_CHECK_MSG(false, "the option's create after the fault: %s", AbiStatusName(again.status));
}

// 7.8: the public header declares both verbs with §3.6's signatures (their sentences are 4.14's cells).
void Cell78() {
	std::ifstream f(std::string(PKV_REFERENCE_DIR) + "/../../../include/superslm/sslm_abi_functions.inc");
	const std::string text((std::istreambuf_iterator<char>(f)), {});
	PKV_CHECK_MSG(!text.empty(), "include/superslm/sslm_abi_functions.inc not read");
	PKV_CHECK_MSG(text.find("size_t sslm_kv_pool_contiguous_size(sslm_model model, uint32_t block_count)") != std::string::npos,
	              "the header does not declare sslm_kv_pool_contiguous_size(sslm_model, uint32_t)");
	PKV_CHECK_MSG(text.find("sslm_kv_pool_create_contiguous(sslm_model model, void* buf, size_t buf_size, uint32_t block_count,") !=
	                  std::string::npos,
	              "the header does not declare sslm_kv_pool_create_contiguous with §3.6's parameters");
}

// 6.7 [T3]: the option directions at a dividing cap (pkv_def): saved in the option, restored in the
// derived-B page pool, and back; legacy and budget holders. (4.14/T3:saves runs the same construction
// graded on rows; this cell adds the restoring holder's reservation: a legacy restore in the page pool
// reserves ceil(cap/B) = 32 pages, so a 32-page pool takes exactly one.)
void Cell67Option() {
	const Fixture& fx = Def();
	if (!fx.ok) return;
	const Lay y = DefLay(DefB());
	ContigPoolOf cp(fx.model, 1, ContigBytes(y, 1));
	PagePoolOf pp(fx.model, static_cast<uint32_t>(y.pages), y.page_bytes);
	if (!cp.pool || !pp.pool) {
		PKV_CHECK_MSG(false, "pools: option %s, page %s", AbiStatusName(cp.status), AbiStatusName(pp.status));
		return;
	}
	sslm_seq s = nullptr, r = nullptr, extra = nullptr;
	PKV_CHECK_EQ(sslm_seq_create(fx.model, &cp.pool, &s), SSLM_OK);
	if (!s) return;
	PrefillAll(fx.model, s, Stream(31, 700, fx.vocab), 64);
	const std::vector<uint8_t> blob = bc::Save(s);
	PKV_CHECK_EQ(sslm_seq_restore(fx.model, &pp.pool, blob.data(), blob.size(), &r), SSLM_OK);
	PKV_CHECK_EQ(sslm_seq_create_budgeted(fx.model, &pp.pool, 1, &extra), SSLM_KV_POOL_EXHAUSTED);
	if (extra) sslm_seq_release(extra);
	if (r) {
		PKV_CHECK(bc::DecodeN(fx.model, r, 6) == bc::DecodeN(fx.model, s, 6));
		sslm_seq_release(r);
	}
	sslm_seq_release(s);
}

// 7.9/C6:fixtures, the cloud twin: one blob (prefill 1,200 of the 7.9 harness stream) restored into a
// legacy holder in a page pool at the derived B and into one in the option; 104 pairs in a random
// order per pair, each token timed alone; the streams must be equal token for token. Timings print
// for the record; no verdict reads them in the cloud.
void Cell79Fixtures() {
	const Fixture& fx = Def();
	if (!fx.ok) return;
	const Lay y = DefLay(DefB());
	BlockPoolOf src(fx.model, 1, y);
	sslm_seq s = nullptr;
	if (!src.pool || sslm_seq_create(fx.model, &src.pool, &s) != SSLM_OK) return;
	PKV_CHECK_EQ(PrefillAll(fx.model, s, Stream(79, 1200, fx.vocab), 64), SSLM_OK);
	const std::vector<uint8_t> blob = bc::Save(s);
	sslm_seq_release(s);
	ContigPoolOf cp(fx.model, 1, ContigBytes(y, 1));
	BlockPoolOf pp(fx.model, 1, y);
	PKV_CHECK_MSG(cp.status == SSLM_OK, "the option's create: %s", AbiStatusName(cp.status));
	if (!cp.pool || !pp.pool) return;
	sslm_seq a = nullptr, b = nullptr;
	PKV_CHECK_EQ(sslm_seq_restore(fx.model, &pp.pool, blob.data(), blob.size(), &a), SSLM_OK);
	PKV_CHECK_EQ(sslm_seq_restore(fx.model, &cp.pool, blob.data(), blob.size(), &b), SSLM_OK);
	if (!a || !b) return;
	std::mt19937 rng(7919u);
	int mismatch = -1;
	double paged = 0, contiguous = 0;
	for (int i = 0; i < 4 + 104; ++i) {
		const bool first = (rng() & 1) != 0;
		auto one = [&](sslm_seq q, double* acc) {
			const auto t0 = std::chrono::steady_clock::now();
			const int32_t t = bc::Token(fx.model, q);
			if (i >= 4) *acc += std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count();
			return t;
		};
		int32_t ta = 0, tb = 0;
		if (first) { ta = one(a, &paged); tb = one(b, &contiguous); }
		else { tb = one(b, &contiguous); ta = one(a, &paged); }
		if (ta != tb && mismatch < 0) mismatch = i;
	}
	PKV_CHECK_MSG(mismatch < 0, "token-stream mismatch at pair %d", mismatch);
	std::printf("  7.9/C6:fixtures (recorded, not graded): 104 pairs, paged %.1f us, contiguous %.1f us\n", paged, contiguous);
	sslm_seq_release(a);
	sslm_seq_release(b);
}

// 6.7 [T3], moved from T2 by rev 16.23 (T-3189 G-3): at the non-dividing cap 4,100 (pkv_odd, B = 256), a blob
// saved in the contiguous option restores into the page pool and one saved in the page pool restores into the
// option, at widths 1, 255, 256, 4,099 and 4,100; rows [0, L') equal and the next tokens agree. Mutant: a restore
// that writes floor(L/B) * B rows loses the tail (fails at 4,099). Rev 16.22 (D-SLM8427): the legacy blob saved at
// width 4,099 is byte-equal across pool kinds and is 160 + hidden_size + 4 * history + the flat block; a section
// sized by each pool's own reservation (cap * q in the option, 17 pages in the page pool) differs by 96,768 bytes.
void Cell67NonDividingOption() {
	const Fixture& odd = GetFixture("pkv_odd");
	if (!odd.ok) return;
	const int64_t B = RuleB(48, 4100, 1);
	PKV_CHECK_MSG(sslm_kv_page_positions(odd.model) == B, "pkv_odd runs at B = %lld; the rule gives 256",
	              static_cast<long long>(sslm_kv_page_positions(odd.model)));
	const Lay y = LayoutOf(odd.geo, 1, B);
	bool fits = false;
	const uint64_t csize = ContiguousSize(y, 2, &fits);
	for (int64_t W : {int64_t{1}, int64_t{255}, int64_t{256}, int64_t{4099}, int64_t{4100}}) {
		ContigPoolOf cp(odd.model, 2, static_cast<size_t>(csize));
		BlockPoolOf pp(odd.model, 2, y);
		PKV_CHECK_MSG(cp.status == SSLM_OK, "W %lld: the contiguous pool: %s", static_cast<long long>(W), AbiStatusName(cp.status));
		PKV_CHECK_EQ(pp.status, SSLM_OK);
		if (!cp.pool || !pp.pool) continue;
		for (int dir = 0; dir < 2; ++dir) {
			sslm_kv_pool* src = dir == 0 ? &cp.pool : &pp.pool;
			sslm_kv_pool* dst = dir == 0 ? &pp.pool : &cp.pool;
			const int64_t dstB = dir == 0 ? B : 4100;
			sslm_seq s = nullptr, r = nullptr;
			PKV_CHECK_EQ(sslm_seq_create(odd.model, src, &s), SSLM_OK);
			if (!s) continue;
			PKV_CHECK_EQ(PrefillAll(odd.model, s, Stream(41, static_cast<int32_t>(W), odd.vocab), 64), SSLM_OK);
			const std::vector<uint8_t> blob = bc::Save(s);
			PKV_CHECK_EQ(sslm_seq_restore(odd.model, dst, blob.data(), blob.size(), &r), SSLM_OK);
			if (r) {
				std::vector<uint8_t> rows, brows;
				// kills: a restore that writes floor(L/B) * B rows loses the tail (fails at 4,099)
				PKV_CHECK_MSG(SeqRows(*dst, r, odd.geo, 1, dstB, W, Rot::kRotated, &rows) &&
				                  BlobRows(blob, odd, HiddenSize(odd), W, &brows) && rows == brows,
				              "W %lld, %s: restored rows equal the blob's", static_cast<long long>(W),
				              dir == 0 ? "option -> page pool" : "page pool -> option");
				if (W < 4100) PKV_CHECK(bc::DecodeN(odd.model, r, 2) == bc::DecodeN(odd.model, s, 2));
				sslm_seq_release(r);
			}
			sslm_seq_release(s);
		}
	}
	// Rev 16.22 (D-SLM8427): the legacy blob saved at width 4,099 is byte-equal across pool kinds, and
	// is 160 + hidden_size + 4 * history + the flat block. Mutant: a section sized by each pool's own
	// reservation (cap * q in the option, 17 pages in the page pool) differs by 96,768 bytes.
	{
		bool f1 = false;
		ContigPoolOf cp(odd.model, 1, static_cast<size_t>(ContiguousSize(y, 1, &f1)));
		BlockPoolOf pp(odd.model, 1, y);
		std::vector<uint8_t> blobs[2];
		for (int k = 0; k < 2; ++k) {
			sslm_kv_pool* pool = k == 0 ? &cp.pool : &pp.pool;
			sslm_seq s = nullptr;
			if (*pool && sslm_seq_create(odd.model, pool, &s) == SSLM_OK) {
				PKV_CHECK_EQ(PrefillAll(odd.model, s, Stream(41, 4099, odd.vocab), 64), SSLM_OK);
				blobs[k] = bc::Save(s);
				sslm_seq_release(s);
			}
		}
		PKV_CHECK_MSG(!blobs[0].empty() && blobs[0] == blobs[1],
		              "the legacy saves of one state from the option (%zu bytes) and the page pool (%zu) differ",
		              blobs[0].size(), blobs[1].size());
		for (const auto& b : blobs)
			if (!b.empty())
				PKV_CHECK_MSG(b.size() == 160 + 192 + 4 * Le64(b, 112) + y.block_bytes,
				              "a legacy save is %zu bytes; 160 + 192 + 4 * %llu + %zu owed", b.size(),
				              static_cast<unsigned long long>(Le64(b, 112)), y.block_bytes);
	}
}


// 4.14 [T3], T-3195's lesser plane (rev 16.23): a budget holder in a contiguous pool reserves exactly one page,
// the whole cap (§21.2), whether created or restored. At 2ebe5e3 the four budget reservation sites compute
// ReservationPages from the model's B (sslm_abi.cpp:2184, :2206, :2396, :3988), not the pool's; 4.14's other members
// reach three of them and a budget restore into a contiguous pool reaches none. Graded by the fill probe: after one
// budget holder, a pool of N blocks admits exactly N - 1 more budget-1 creates. Mutant: the reservation taken by the
// model's B (R(100) = min(ceil(100/128) + 1, 32) = 2 pages of the pool's cap-sized pages), which admits N - 2.
void Cell414Reservation() {
	const Fixture& fx = Def();
	if (!fx.ok) return;
	const Lay y = DefLay(DefB());
	const uint32_t N = 3;
	PKV_CHECK(Reservation(100, kCap, DefB()) == 2 && Reservation(100, kCap, kCap) == 1);  // the mutant separates
	// a budget blob to restore, saved from a page pool at the derived B
	std::vector<uint8_t> blob;
	{
		PagePoolOf src(fx.model, 8, y.page_bytes);
		sslm_seq s = nullptr;
		if (src.pool && sslm_seq_create_budgeted(fx.model, &src.pool, 100, &s) == SSLM_OK) {
			PrefillAll(fx.model, s, Stream(31, 60, fx.vocab), 64);
			blob = bc::Save(s);
			sslm_seq_release(s);
		}
		PKV_CHECK_MSG(IsMagic(blob, "SSB6"), "the budget-100 blob to restore was not made");
	}
	for (int restored = 0; restored < 2; ++restored) {
		ContigPoolOf cp(fx.model, N, ContigBytes(y, N));
		PKV_CHECK_MSG(cp.status == SSLM_OK, "the option's create: %s", AbiStatusName(cp.status));
		if (!cp.pool) continue;
		sslm_seq h = nullptr;
		const sslm_status st = restored ? sslm_seq_restore(fx.model, &cp.pool, blob.data(), blob.size(), &h)
		                                : sslm_seq_create_budgeted(fx.model, &cp.pool, 100, &h);
		PKV_CHECK_MSG(st == SSLM_OK, "a budget-100 holder %s in the option: %s", restored ? "restored" : "created",
		              AbiStatusName(st));
		if (!h) continue;
		sslm_status refusal = SSLM_OK;
		const int n = CountCreates(fx.model, &cp.pool, 1, &refusal);
		PKV_CHECK_MSG(n == static_cast<int>(N) - 1, "after one %s budget holder, %u blocks admit %d budget-1 creates; %u owed",
		              restored ? "restored" : "created", N, n, N - 1);
		sslm_seq_release(h);
	}
}

using namespace pkv::r9::c415;

void Cell415() {
	Cell415On(GetFixture("pkv_def"), RuleB(48, 4096, 1), static_cast<int32_t>(RuleB(48, 4096, 1)), 700,
	          "g5_minimal_one_field");  // B = 128
	Cell415On(GetFixture("pkv_odd"), RuleB(48, 4100, 1), static_cast<int32_t>(RuleB(48, 4100, 1)), 700,
	          "g5_minimal_one_field");  // B = 256
	// One geometry: D = 48 at cap 100, where the rule's B is the cap; both directions admitted, rows equal.
	const Fixture& one = Get("r9_d48_c100", Geometry{2, 2, 48, 100});
	if (!one.ok) return;
	PKV_CHECK_EQ(RuleB(48, 100, 1), 100);
	for (int32_t len : {50, 99}) {
		KindPools k(one, 100, 3);
		PKV_CHECK_MSG(k.Ok(), "cap 100: the pools (contiguous: %s)", AbiStatusName(k.contig->status));
		if (!k.Ok()) continue;
		for (int dir = 0; dir < 2; ++dir) {
			sslm_kv_pool* sp = dir == 0 ? &k.page->pool : &k.contig->pool;
			sslm_kv_pool* pp = dir == 0 ? &k.contig->pool : &k.page->pool;
			sslm_prefix p = LegacyPrefixOf(one, pp, len);
			sslm_seq s = nullptr;
			const bool made = p && sslm_seq_create(one.model, sp, &s) == SSLM_OK;
			PKV_CHECK_MSG(made, "cap 100: the prefix of %d or its adopter was not made", len);
			if (made)
				ExpectAdmitted(one, *sp, 100, *pp, 100, s, p, len,
				               std::string("cap 100 (one geometry): ") + (dir == 0 ? "page from contiguous" : "contiguous from page") +
				                   ", len " + std::to_string(len));
			if (s) sslm_seq_release(s);
			if (p) sslm_prefix_release(p);
		}
	}
}

// 4.14 [T3] batch (plan rev 16.24 §26.1, §26.8 item 1; D-SLM8441). pkv_def, a blob at width 1,020: (a) two legacy
// holders in one two-block contiguous pool; (b) one in a one-block page pool and one in a one-block contiguous pool,
// in both batch orders. Rows 1,020 to 1,031. Kills H1 by (b), H2 by (a) and (b).
void Cell414Batch() {
	const Fixture& fx = Def();
	if (!fx.ok) return;
	const int64_t B = DefB();
	const Lay y = DefLay(B);
	const int64_t W = 1020;
	const std::vector<uint8_t> blob = WidthsBlob(fx, B, W);
	{
		ContigPoolOf cp(fx.model, 2, ContigBytes(y, 2), 0x5A);
		PKV_CHECK_MSG(cp.status == SSLM_OK, "(a): the two-block option create: %s", AbiStatusName(cp.status));
		if (cp.pool)
			BatchAgainstTwin(fx, blob, W, B, {&cp.pool, kCap, "contiguous"}, {&cp.pool, kCap, "contiguous"}, false,
			                 "4.14 (a) contiguous + contiguous");
	}
	for (bool page_first : {true, false}) {
		BlockPoolOf pp(fx.model, 1, y, 0x5A);
		ContigPoolOf cp(fx.model, 1, ContigBytes(y, 1), 0x5A);
		PKV_CHECK_MSG(pp.status == SSLM_OK && cp.status == SSLM_OK, "(b): pools: page %s, option %s", AbiStatusName(pp.status),
		              AbiStatusName(cp.status));
		if (!pp.pool || !cp.pool) continue;
		const BatchArm page{&pp.pool, B, "page"}, contig{&cp.pool, kCap, "contiguous"};
		BatchAgainstTwin(fx, blob, W, B, page_first ? page : contig, page_first ? contig : page, false,
		                 page_first ? "4.14 (b) page first" : "4.14 (b) contiguous first");
	}
}

// 4.4 [T3] batch (§26.1, §26.8 item 2). pkv_odd (B = 256, 17 pages), a blob at width 4,088: a page holder (one-block
// page pool) beside a contiguous holder (one-block option pool), both orders, through rows 4,088 to 4,099 in the
// partial last page; then the step past the cap is refused. Kills H1 and H2.
void Cell44Batch() {
	const Fixture& odd = GetFixture("pkv_odd");
	if (!odd.ok) return;
	const int64_t B = RuleB(48, 4100, 1);
	const Lay y = LayoutOf(odd.geo, 1, B);
	const int64_t W = 4088;
	PKV_CHECK_EQ(y.pages, 17);
	const std::vector<uint8_t> blob = WidthsBlob(odd, B, W);
	for (bool page_first : {true, false}) {
		BlockPoolOf pp(odd.model, 1, y, 0x5A);
		bool fits = false;
		ContigPoolOf cp(odd.model, 1, static_cast<size_t>(ContiguousSize(y, 1, &fits)), 0x5A);
		PKV_CHECK_MSG(pp.status == SSLM_OK && cp.status == SSLM_OK, "pools: page %s, option %s", AbiStatusName(pp.status),
		              AbiStatusName(cp.status));
		if (!pp.pool || !cp.pool) continue;
		const BatchArm page{&pp.pool, B, "page"}, contig{&cp.pool, 4100, "contiguous"};
		BatchAgainstTwin(odd, blob, W, B, page_first ? page : contig, page_first ? contig : page, true,
		                 page_first ? "4.4 page first" : "4.4 contiguous first");
	}
}

PKV_CELL("4.14/T3:bit-identity", "T3", Cell414BitIdentity);
PKV_CELL("4.14/T3:batch", "T3", Cell414Batch);
PKV_CELL("4.4/T3:batch", "T3", Cell44Batch);
PKV_CELL("4.15/T3", "T3", Cell415);
PKV_CELL("6.7/T3:non-dividing", "T3", Cell67NonDividingOption);
PKV_CELL("4.14/T3:admission", "T3", Cell414Admission);
PKV_CELL("4.14/T3:sizing", "T3", Cell414Sizing);
PKV_CELL("4.14/T3:saves", "T3", Cell414Saves);
PKV_CELL("4.14/T3:adopt", "T3", Cell414Adopt);
PKV_CELL("4.14/T3:lifecycle", "T3", Cell414Lifecycle);
PKV_CELL("4.14/T3:hostile", "T3", Cell414Hostile);
PKV_CELL("4.14/T3:reservation", "T3", Cell414Reservation);
PKV_CELL("5.6/T3", "T3", Cell56);
PKV_CELL("7.8/T3", "T3", Cell78);
PKV_CELL("6.7/T3:option", "T3", Cell67Option);
PKV_CELL("7.9/C6:fixtures", "T3", Cell79Fixtures);

}  // namespace

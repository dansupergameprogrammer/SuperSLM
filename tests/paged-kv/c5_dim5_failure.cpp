// Paged-KV plan (rev 16.1) §7 dimension 5, failure and rejection paths: the C5-owned cells (the
// budget limit on prefill and decode, the cap status at limit == cap, exhaustion, the freeze return,
// and allocation failure in the new verbs). Red by link at C1.
//
// Token counts are the §3.4 formula written here (a holder resting ready at c emits 1 + (limit - c));
// "nothing leaked" is the two-sided fill probe (§8).

#include "pkv_budget_a_helpers.h"

#include "support/bad_alloc_injection.h"

#include <type_traits>

namespace {

using namespace pkv;
using namespace pkv::budget_a;

// ---- 5.1 [C5] -----------------------------------------------------------------------------------
// Prefill past the limit: partial consumption, SSLM_KV_BUDGET_EXCEEDED, resumable. A budget-100
// holder given 150 tokens consumes exactly 100 (in one call, and across chunked calls where the
// second call stops partway), refuses the rest, and its state is the 100-token state: rows equal a
// fresh holder that prefilled exactly those 100, its one ready token is that holder's, and it is
// refused after it; reset, it prefills 100 again. A bad token id before the limit still reports
// SSLM_TOKEN_ID_OUT_OF_RANGE (with partial consumption up to it); past the limit the budget wins.
void Cell51() {
	const Fixture& fx = GetFixture("pkv_def");
	if (!fx.ok) return;
	Scene sc;
	PagePool& pool = sc.AddPool(fx.model, 64);
	const std::vector<int32_t> t150 = Stream(1, 150, fx.vocab);
	const std::vector<int32_t> t100(t150.begin(), t150.begin() + 100);
	std::vector<uint8_t> twin_blob;
	std::vector<int32_t> twin_tok;
	{
		sslm_seq t = BudgetSeq(fx, &pool.pool, 100, sc.h);
		PKV_CHECK_EQ(t ? PrefillAll(fx.model, t, t100, 64) : SSLM_INVALID_ARGUMENT, SSLM_OK);
		twin_blob = Save(t);
		if (t) DecodeN(fx.model, t, 1, &twin_tok);
		if (t) ReleaseSeq(sc.h, t);
	}
	for (int chunked = 0; chunked <= 1; ++chunked) {
		sslm_seq s = BudgetSeq(fx, &pool.pool, 100, sc.h);
		if (!s) {
			PKV_CHECK_MSG(false, "5.1: create");
			continue;
		}
		int64_t consumed = 0;
		sslm_status st = SSLM_OK;
		if (chunked) {
			st = PrefillAll(fx.model, s, t150, 64, &consumed);  // 64, then 36 of the next 64
		} else {
			int32_t c = -1;
			st = sslm_prefill(fx.model, s, t150.data(), 150, 150, SSLM_SPAN_PROMPT, nullptr, &c);
			consumed = c;
		}
		// kills: the limit headroom missing from the admission pre-scan (150 consumed), or an
		// all-or-nothing refusal (0 consumed).
		PKV_CHECK_EQ(st, PKV_KV_BUDGET_EXCEEDED);
		PKV_CHECK_EQ(consumed, 100);
		int32_t c2 = -1;
		PKV_CHECK_EQ(sslm_prefill(fx.model, s, t150.data() + 100, 1, 1, SSLM_SPAN_PROMPT, nullptr, &c2), PKV_KV_BUDGET_EXCEEDED);
		PKV_CHECK_EQ(c2, 0);
		const std::vector<uint8_t> b = Save(s);
		// kills: a refusal that wrote a row past the limit, or left the consumed rows half-written.
		PKV_CHECK_MSG(BlobContextLength(b) == 100 && RowsSha(fx, b, 100) == RowsSha(fx, twin_blob, 100),
		              "5.1 (%s): the stopped holder's rows differ from the 100-token twin's", chunked ? "chunked" : "one call");
		std::vector<int32_t> tok;
		sslm_status refusal = SSLM_OK;
		PKV_CHECK_EQ(TokensUntilRefusal(fx.model, s, &refusal, 8192, &tok), 1);  // 1 + (limit - c), c = limit
		PKV_CHECK_EQ(refusal, PKV_KV_BUDGET_EXCEEDED);
		PKV_CHECK(tok == twin_tok);
		PKV_CHECK_EQ(sslm_seq_reset(s), SSLM_OK);  // resumable: the reservation is kept, the budget runs again
		int64_t again = 0;
		PKV_CHECK_EQ(PrefillAll(fx.model, s, t100, 64, &again), SSLM_OK);
		PKV_CHECK_EQ(again, 100);
		ReleaseSeq(sc.h, s);
	}
	struct {
		int at;
		sslm_status want;
		int64_t consumed;
	} ids[] = {{50, SSLM_TOKEN_ID_OUT_OF_RANGE, 50}, {0, SSLM_TOKEN_ID_OUT_OF_RANGE, 0}, {120, PKV_KV_BUDGET_EXCEEDED, 100}};
	for (auto& c : ids) {
		std::vector<int32_t> t = t150;
		t[static_cast<size_t>(c.at)] = fx.vocab;  // one past the vocabulary
		sslm_seq s = BudgetSeq(fx, &pool.pool, 100, sc.h);
		int32_t consumed = -1;
		const sslm_status st = s ? sslm_prefill(fx.model, s, t.data(), 150, 150, SSLM_SPAN_PROMPT, nullptr, &consumed)
		                         : SSLM_INVALID_ARGUMENT;
		// kills: the budget check given precedence over the token-id check (or the reverse past the limit).
		PKV_CHECK_MSG(st == c.want && consumed == c.consumed, "5.1 bad id at %d: -> %d, consumed %d", c.at,
		              static_cast<int>(st), consumed);
		if (s) ReleaseSeq(sc.h, s);
	}
}

// ---- 5.2 [C5] -----------------------------------------------------------------------------------
// Decode at the limit: out_tokens[i] = -1 and the status is returned, with the batch behaviour of
// today's cap check. A holder resting ready at context_length c emits exactly 1 + (limit - c) tokens
// (§3.4): c = 25 under budget 40 emits 16; an adopter of a ready 1,000-token prefix with budget 20
// emits 21; a holder resting at c = limit emits 1. In a batch, a budget holder at its limit and a
// legacy holder at the cap give the same pattern: in [Y, X] Y emits its token and X is refused; in
// [X, Y] the call stops at X and Y is untouched.
void Cell52() {
	const Fixture& fx = GetFixture("pkv_def");
	if (!fx.ok) return;
	Scene sc;
	PagePool& pool = sc.AddPool(fx.model, 160);
	struct {
		const char* name;
		int32_t budget, prefill, prefix;
	} counts[] = {{"c 25, budget 40", 40, 25, 0}, {"adopter at 1,000, budget 20", 20, 0, 1000}, {"c = limit", 25, 25, 0}};
	for (auto& c : counts) {
		sslm_prefix p = c.prefix ? FrozenBudgetPrefix(fx, &pool.pool, Stream(51, c.prefix, fx.vocab), sc.h) : nullptr;
		sslm_seq s = BudgetSeq(fx, &pool.pool, c.budget, sc.h);
		if (!s || (c.prefix && (!p || sslm_seq_adopt_prefix(s, p) != SSLM_OK)) ||
		    (c.prefill && PrefillAll(fx.model, s, Stream(1, c.prefill, fx.vocab), 64) != SSLM_OK)) {
			PKV_CHECK_MSG(false, "5.2 %s: setup", c.name);
			continue;
		}
		const int64_t rest = c.prefix ? c.prefix : c.prefill, limit = rest + c.budget;
		sslm_status refusal = SSLM_OK;
		// kills: the budget check on the ready path (one token fewer) or after the embed (a row past the
		// limit and one token more).
		PKV_CHECK_MSG(TokensUntilRefusal(fx.model, s, &refusal) == 1 + (limit - rest), "5.2 %s: token count", c.name);
		PKV_CHECK_EQ(refusal, PKV_KV_BUDGET_EXCEEDED);
		ReleaseSeq(sc.h, s);
		if (p) ReleasePrefix(sc.h, p);
	}
	// X at its limit (budget) or at the cap (legacy), each resting with a pending token to embed.
	sslm_seq xb = BudgetSeq(fx, &pool.pool, 40, sc.h);
	LegacyPool& lp = sc.AddLegacyPool(fx.model, 1);
	sslm_seq xl = nullptr;
	PKV_CHECK_EQ(sslm_seq_create(fx.model, &lp.pool, &xl), SSLM_OK);
	if (xl) sc.h.seqs.push_back(xl);
	if (!xb || !xl) return;
	PrefillAll(fx.model, xb, Stream(1, 25, fx.vocab), 64);
	DecodeN(fx.model, xb, 16, nullptr);  // now at limit 40
	PrefillAll(fx.model, xl, Stream(1, static_cast<int32_t>(fx.geo.context_cap), fx.vocab), 512);
	DecodeN(fx.model, xl, 1, nullptr);  // now at the cap
	struct Pattern {
		sslm_status st;
		int32_t out_x, out_y;
		std::vector<uint8_t> y_after;
	};
	auto run = [&](sslm_seq x, bool x_first) {
		Pattern r{};
		sslm_seq y = BudgetSeq(fx, &pool.pool, 64, sc.h);
		if (!y) return r;
		PrefillAll(fx.model, y, Stream(9, 10, fx.vocab), 64);
		sslm_seq batch[2] = {x_first ? x : y, x_first ? y : x};
		int32_t out[2] = {-77, -77};
		sslm_decode_params prm{};
		prm.layer_budget = 2;
		r.st = sslm_decode_step(fx.model, batch, 2, &prm, nullptr, out);
		r.out_x = out[x_first ? 0 : 1];
		r.out_y = out[x_first ? 1 : 0];
		r.y_after = Save(y);
		ReleaseSeq(sc.h, y);
		return r;
	};
	for (bool x_first : {false, true}) {
		const Pattern b = run(xb, x_first), l = run(xl, x_first);
		PKV_CHECK_EQ(b.st, PKV_KV_BUDGET_EXCEEDED);
		PKV_CHECK_EQ(l.st, SSLM_CONTEXT_CAP_EXCEEDED);  // today's check, unchanged for a legacy holder
		PKV_CHECK_EQ(b.out_x, -1);
		PKV_CHECK_EQ(l.out_x, -1);
		// kills: a budget refusal with its own batch semantics (skipping X and finishing the batch, or
		// failing the whole call before Y in [Y, X]).
		PKV_CHECK_MSG(b.out_y == l.out_y && b.y_after == l.y_after, "5.2 batch [%s]: the budget refusal's batch behaviour differs from the cap's",
		              x_first ? "X, Y" : "Y, X");
	}
}

// ---- 5.3 [C5] -----------------------------------------------------------------------------------
// limit == cap reports SSLM_CONTEXT_CAP_EXCEEDED, never the budget status: a budget-cap holder from
// origin 0; an adopter at 1,000 with budget 4,000 (limit min(5,000, 4,096) = cap); and one with budget
// 3,096 (origin + budget = cap exactly). Prefill past and decode past both.
void Cell53() {
	const Fixture& fx = GetFixture("pkv_def");
	if (!fx.ok) return;
	const int64_t cap = fx.geo.context_cap;
	Scene sc;
	PagePool& pool = sc.AddPool(fx.model, static_cast<uint32_t>(fx.CapPages() + 80));
	sslm_prefix p = FrozenBudgetPrefix(fx, &pool.pool, Stream(51, 1000, fx.vocab), sc.h);
	struct {
		const char* name;
		int32_t budget;
		bool adopt;
	} cases[] = {{"budget cap, origin 0", static_cast<int32_t>(cap), false},
	             {"origin 1,000, budget 4,000", 4000, true},
	             {"origin 1,000, budget 3,096", static_cast<int32_t>(cap - 1000), true}};
	for (auto& c : cases) {
		for (int via_decode = 0; via_decode <= 1; ++via_decode) {
			sslm_seq s = BudgetSeq(fx, &pool.pool, c.budget, sc.h);
			if (!s || (c.adopt && (!p || sslm_seq_adopt_prefix(s, p) != SSLM_OK))) {
				PKV_CHECK_MSG(false, "5.3 %s: setup", c.name);
				continue;
			}
			const int64_t origin = c.adopt ? 1000 : 0;
			// Fill to one short of the cap, so the decode writes the last row and then meets the cap.
			int64_t got = 0;
			PKV_CHECK_EQ(PrefillAll(fx.model, s, Stream(5, static_cast<int32_t>(cap - origin - 1), fx.vocab), 512, &got), SSLM_OK);
			if (via_decode) {
				sslm_status refusal = SSLM_OK;
				PKV_CHECK_EQ(TokensUntilRefusal(fx.model, s, &refusal), 2);  // 1 + (cap - (cap - 1))
				// kills: the budget check ahead of the cap check at limit == cap.
				PKV_CHECK_MSG(refusal == SSLM_CONTEXT_CAP_EXCEEDED, "5.3 %s, decode: -> %d", c.name, static_cast<int>(refusal));
			} else {
				int32_t n = -1;
				const std::vector<int32_t> two = Stream(6, 2, fx.vocab);
				const sslm_status st = sslm_prefill(fx.model, s, two.data(), 2, 2, SSLM_SPAN_PROMPT, nullptr, &n);
				PKV_CHECK_MSG(st == SSLM_CONTEXT_CAP_EXCEEDED && n == 1, "5.3 %s, prefill: -> %d, consumed %d", c.name,
				              static_cast<int>(st), n);
			}
			ReleaseSeq(sc.h, s);
		}
	}
}

// ---- 5.4 [C5] -----------------------------------------------------------------------------------
// Against an exhausted pool, every verb that draws is refused SSLM_KV_POOL_EXHAUSTED with a null
// out-handle and nothing leaked: budget and legacy create, budget and legacy begin, begin_from, an
// SSB6 restore (sslm_seq_restore, restore_shared with no handle and with the prefix's handle) and a
// 1.9.0 SSB5 restore. Adopt against the exhausted pool succeeds and its holder runs its whole budget
// (64 + 1 tokens). The adopt case uses a mid-page prefix (1,000): an aligned prefix has no tail, so
// the tail-from-pool mutant is equivalent there (run at 1,008 as the must-accept only).
constexpr uint32_t k54Pool = 120;

struct Out54 {
	std::vector<std::string> wrong;
	sslm_status adopt = SSLM_INVALID_ARGUMENT;
	int64_t tokens = -1;
	sslm_status refusal = SSLM_OK;
};

std::vector<uint8_t> Blob54(const Fixture& fx) {
	Scene sc;
	PagePool& pool = sc.AddPool(fx.model, 120);
	sslm_prefix p = FrozenBudgetPrefix(fx, &pool.pool, Stream(51, 1000, fx.vocab), sc.h);
	sslm_seq s = BudgetSeq(fx, &pool.pool, 64, sc.h);
	if (p && s && sslm_seq_adopt_prefix(s, p) == SSLM_OK) DecodeN(fx.model, s, 4, nullptr);
	return Save(s);
}

std::unique_ptr<ProbeState> Build54(int32_t prefix_len, const std::vector<uint8_t>& ssb6, const std::vector<uint8_t>& ssb5,
                                    Out54* out) {
	auto sc = std::make_unique<Scene>();
	const Fixture& fx = GetFixture("pkv_def");
	PagePool& pool = sc->AddPool(fx.model, k54Pool);
	if (pool.status != SSLM_OK) return sc;
	sslm_prefix p = FrozenBudgetPrefix(fx, &pool.pool, Stream(51, prefix_len, fx.vocab), sc->h);
	sslm_seq h = BudgetSeq(fx, &pool.pool, 64, sc->h);
	const int64_t rest = k54Pool - PrefixPages(fx, prefix_len) - fx.R(64);
	std::vector<sslm_seq> fill;
	if (!p || !h || !AdmitPages(fx, &pool.pool, rest, &fill)) {
		if (out) out->wrong.push_back("setup: the pool was not filled");
		for (sslm_seq s : fill) sslm_seq_release(s);
		return sc;
	}
	// Each out-handle is preset to a non-null sentinel and read only after its verb returns (through the
	// pointer, so argument order cannot read it first), and a refusal must null it.
	// kills: a refusal that returns before writing *out (the preset left standing)
	auto expect = [&](const char* name, sslm_status st, auto* handle) {
		using H = std::remove_pointer_t<decltype(handle)>;
		if (out && (st != SSLM_KV_POOL_EXHAUSTED || *handle != nullptr))
			out->wrong.push_back(std::string(name) + " -> " + std::to_string(st));
		if (*handle && *handle != OutSentinel<H>()) {
			if constexpr (std::is_same_v<H, sslm_seq>) sslm_seq_release(*handle);
			else sslm_prefix_release(*handle);
		}
	};
	const sslm_seq kSeqPreset = OutSentinel<sslm_seq>();
	const sslm_prefix kPrefixPreset = OutSentinel<sslm_prefix>();
	sslm_seq s = kSeqPreset;
	sslm_prefix q = kPrefixPreset;
	uint32_t shared = 0;
	expect("seq_create_budgeted", sslm_seq_create_budgeted(fx.model, &pool.pool, 1, &s), &s);
	s = kSeqPreset;
	expect("seq_create", sslm_seq_create(fx.model, &pool.pool, &s), &s);
	expect("prefix_begin_budgeted", sslm_prefix_begin_budgeted(fx.model, &pool.pool, 1, &q), &q);
	q = kPrefixPreset;
	expect("prefix_begin", sslm_prefix_begin(fx.model, &pool.pool, &q), &q);
	q = kPrefixPreset;
	expect("prefix_begin_from", sslm_prefix_begin_from(p, 1, &q), &q);
	s = kSeqPreset;
	expect("seq_restore (SSB6)", sslm_seq_restore(fx.model, &pool.pool, ssb6.data(), ssb6.size(), &s), &s);
	s = kSeqPreset;
	expect("restore_shared (SSB6, null)", sslm_seq_restore_shared(fx.model, &pool.pool, ssb6.data(), ssb6.size(), nullptr, &s, &shared), &s);
	s = kSeqPreset;
	expect("restore_shared (SSB6, the prefix)", sslm_seq_restore_shared(fx.model, &pool.pool, ssb6.data(), ssb6.size(), p, &s, &shared), &s);
	s = kSeqPreset;
	expect("seq_restore (1.9.0 SSB5)", sslm_seq_restore(fx.model, &pool.pool, ssb5.data(), ssb5.size(), &s), &s);
	const sslm_status adopt = sslm_seq_adopt_prefix(h, p);
	sslm_status refusal = SSLM_OK;
	const int64_t n = adopt == SSLM_OK ? TokensUntilRefusal(fx.model, h, &refusal) : -1;
	if (out) {
		out->adopt = adopt;
		out->tokens = n;
		out->refusal = refusal;
	}
	for (sslm_seq f : fill) sslm_seq_release(f);
	return sc;
}

void Cell54() {
	const Fixture& fx = GetFixture("pkv_def");
	if (!fx.ok) return;
	const std::vector<uint8_t> ssb6 = Blob54(fx);
	const std::vector<uint8_t> ssb5 = Pin("v1.11.0", "persist", "saved");
	PKV_CHECK(IsMagic(ssb6, "SSB6") && IsMagic(ssb5, "SSB5"));
	for (int32_t len : {1000, 1008}) {
		Out54 o;
		Build54(len, ssb6, ssb5, &o);
		// kills: a draw that does not check availability first (admitted over the pool's end).
		PKV_CHECK_MSG(o.wrong.empty(), "5.4 prefix %d: %zu wrong, first: %s", len, o.wrong.size(),
		              o.wrong.empty() ? "" : o.wrong[0].c_str());
		// kills (at 1,000): adopt drawing its tail page from the pool -- refused in the exhausted pool.
		PKV_CHECK_MSG(o.adopt == SSLM_OK, "5.4 prefix %d: adopt against an exhausted pool -> %d", len, static_cast<int>(o.adopt));
		PKV_CHECK_EQ(o.tokens, 64 + 1);
		PKV_CHECK_EQ(o.refusal, PKV_KV_BUDGET_EXCEEDED);
		// kills: a refused verb that keeps part of its draw (the leak this cell's "nothing leaked" names).
		ProbeExactlyFree(
		    fx, [len, &ssb6, &ssb5] { return Build54(len, ssb6, ssb5, nullptr); },
		    k54Pool - PrefixPages(fx, len) - fx.R(64));
	}
}

// ---- 5.5 [C5] -----------------------------------------------------------------------------------
// A create refused before a prefix freeze succeeds after it. Budget: in a pool of exactly R(1,008) +
// 1 = 65 pages a budget-1,008 prefix prefilled to 1,000 leaves 1 page, so a 2-page create is refused;
// freeze returns the unused reserve page and the same create is admitted. Legacy prefix in a page
// pool: R(cap) + 255 = 511 pages leave 255 after the begin, so a legacy create is refused; freeze
// returns 256 - 63 = 193 and it is admitted.
void Cell55() {
	const Fixture& fx = GetFixture("pkv_def");
	if (!fx.ok) return;
	{
		Scene sc;
		PagePool& pool = sc.AddPool(fx.model, static_cast<uint32_t>(fx.R(1008) + 1));
		sslm_prefix p = nullptr;
		PKV_CHECK_EQ(sslm_prefix_begin_budgeted(fx.model, &pool.pool, 1008, &p), SSLM_OK);
		if (!p) return;
		sc.h.prefixes.push_back(p);
		PKV_CHECK_EQ(PrefixPrefillAll(fx.model, p, Stream(51, 1000, fx.vocab), 64), SSLM_OK);
		sslm_seq s = nullptr;
		PKV_CHECK_EQ(sslm_seq_create_budgeted(fx.model, &pool.pool, static_cast<int32_t>(fx.B()), &s), SSLM_KV_POOL_EXHAUSTED);
		if (s) sc.h.seqs.push_back(s);
		PKV_CHECK_EQ(sslm_prefix_freeze(p), SSLM_OK);
		// kills: freeze that returns nothing (the budget prefix keeps R(1,008) pages).
		PKV_CHECK(BudgetSeq(fx, &pool.pool, static_cast<int32_t>(fx.B()), sc.h) != nullptr);
	}
	{
		Scene sc;
		PagePool& pool = sc.AddPool(fx.model, static_cast<uint32_t>(2 * fx.CapPages() - 1));
		sslm_prefix p = nullptr;
		PKV_CHECK_EQ(sslm_prefix_begin(fx.model, &pool.pool, &p), SSLM_OK);
		if (!p) return;
		sc.h.prefixes.push_back(p);
		PKV_CHECK_EQ(PrefixPrefillAll(fx.model, p, Stream(51, 1000, fx.vocab), 64), SSLM_OK);
		sslm_seq s = nullptr;
		PKV_CHECK_EQ(sslm_seq_create(fx.model, &pool.pool, &s), SSLM_KV_POOL_EXHAUSTED);
		if (s) sc.h.seqs.push_back(s);
		PKV_CHECK_EQ(sslm_prefix_freeze(p), SSLM_OK);
		s = nullptr;
		PKV_CHECK_EQ(sslm_seq_create(fx.model, &pool.pool, &s), SSLM_OK);  // kills: a legacy prefix that keeps its reserve
		if (s) sc.h.seqs.push_back(s);
	}
}

// ---- 5.6 [C5] -----------------------------------------------------------------------------------
// Allocation-failure injection in every new verb, through the existing seam
// (SUPERSLM_ENABLE_BAD_ALLOC_INJECTION, tests/support/bad_alloc_injection.h, armed as
// tests/test_main.cpp arms it; superslm_test_injection is the library the suite links). Each verb,
// armed with a bad_alloc, returns SSLM_ALLOCATION_FAILED with a null out-handle, the seam was consulted
// (it disarmed itself), and the fill probe admits exactly what was free before the call: every page
// the verb drew went back.
//
// The seam fires where the builder consults it. For "pages returned" to be tested rather than
// vacuous, each verb consults it after its pool draw (for example at the holder's heap allocation),
// so the fault unwinds a real draw; consulted before the draw, the cell still passes and proves less.
enum class Verb56 { kPagePoolCreate, kCreateBudgeted, kBeginBudgeted, kBeginFrom, kRestoreNull, kRestoreShared, kRestoreSsb6 };

struct Case56 {
	const char* name;
	Verb56 verb;
	uint32_t pool;
	int64_t free_after;  // pages free after the refused call
};

std::unique_ptr<ProbeState> Build56(const Case56& c, const std::vector<uint8_t>& blob, sslm_status* st, bool* consulted,
                                    bool* out_null) {
	auto sc = std::make_unique<Scene>();
	const Fixture& fx = GetFixture("pkv_def");
	PagePool& pool = sc->AddPool(fx.model, c.pool);
	*st = SSLM_OK;
	if (pool.status != SSLM_OK) return sc;
	sslm_prefix parent = nullptr;
	if (c.verb == Verb56::kBeginFrom) parent = FrozenBudgetPrefix(fx, &pool.pool, Stream(51, 37, fx.vocab), sc->h);
	if (c.verb == Verb56::kRestoreShared) parent = FrozenBudgetPrefix(fx, &pool.pool, Stream(51, 1000, fx.vocab), sc->h);
	sslm_seq s = reinterpret_cast<sslm_seq>(&pool);
	sslm_prefix p = reinterpret_cast<sslm_prefix>(&pool);
	sslm_kv_pool np = reinterpret_cast<sslm_kv_pool>(&pool);
	uint32_t shared = 0;
	AlignedBuf mem;
	superslm_test::ArmInjectedFault(superslm_test::InjectThrowKind::kBadAlloc);
	switch (c.verb) {
		case Verb56::kPagePoolCreate: {
			const size_t size = sslm_kv_page_size(fx.model) * 8 + sslm_kv_page_pool_overhead_size(fx.model, 8);
			superslm_test::DisarmInjectedFault();  // sizing first, then arm for the create alone
			mem.Reset(size);
			superslm_test::ArmInjectedFault(superslm_test::InjectThrowKind::kBadAlloc);
			*st = sslm_kv_page_pool_create(fx.model, mem.p, mem.n, 8, &np);
			*out_null = np == nullptr;
			break;
		}
		case Verb56::kCreateBudgeted:
			*st = sslm_seq_create_budgeted(fx.model, &pool.pool, 64, &s);
			*out_null = s == nullptr;
			break;
		case Verb56::kBeginBudgeted:
			*st = sslm_prefix_begin_budgeted(fx.model, &pool.pool, 64, &p);
			*out_null = p == nullptr;
			break;
		case Verb56::kBeginFrom:
			*st = parent ? sslm_prefix_begin_from(parent, 64, &p) : SSLM_INVALID_ARGUMENT;
			*out_null = p == nullptr;
			break;
		case Verb56::kRestoreNull:
			*st = sslm_seq_restore_shared(fx.model, &pool.pool, blob.data(), blob.size(), nullptr, &s, &shared);
			*out_null = s == nullptr;
			break;
		case Verb56::kRestoreShared:
			*st = parent ? sslm_seq_restore_shared(fx.model, &pool.pool, blob.data(), blob.size(), parent, &s, &shared)
			             : SSLM_INVALID_ARGUMENT;
			*out_null = s == nullptr;
			break;
		case Verb56::kRestoreSsb6:
			*st = sslm_seq_restore(fx.model, &pool.pool, blob.data(), blob.size(), &s);
			*out_null = s == nullptr;
			break;
	}
	*consulted = superslm_test::g_inject_throw == superslm_test::InjectThrowKind::kNone;
	superslm_test::DisarmInjectedFault();
	if (*st == SSLM_OK) {  // a verb that ignored the fault: own what it made, so the scene tears down
		if (c.verb == Verb56::kPagePoolCreate && np) sslm_kv_pool_destroy(np);
		if ((c.verb == Verb56::kBeginBudgeted || c.verb == Verb56::kBeginFrom) && p) sc->h.prefixes.push_back(p);
		if (c.verb != Verb56::kPagePoolCreate && c.verb != Verb56::kBeginBudgeted && c.verb != Verb56::kBeginFrom && s)
			sc->h.seqs.push_back(s);
	}
	return sc;
}

void Cell56() {
	const Fixture& fx = GetFixture("pkv_def");
	if (!fx.ok) return;
	const std::vector<uint8_t> blob = Blob54(fx);  // a budget-64 SSB6 at origin 1,000
	PKV_CHECK(IsMagic(blob, "SSB6"));
	const Case56 cases[] = {
	    {"sslm_kv_page_pool_create", Verb56::kPagePoolCreate, 64, 64},
	    {"sslm_seq_create_budgeted", Verb56::kCreateBudgeted, 64, 64},
	    {"sslm_prefix_begin_budgeted", Verb56::kBeginBudgeted, 64, 64},
	    {"sslm_prefix_begin_from", Verb56::kBeginFrom, 64, 64 - PrefixPages(fx, 37)},
	    {"sslm_seq_restore_shared (no handle)", Verb56::kRestoreNull, 160, 160},
	    {"sslm_seq_restore_shared (matching handle)", Verb56::kRestoreShared, 160, 160 - PrefixPages(fx, 1000)},
	    {"sslm_seq_restore (budget SSB6)", Verb56::kRestoreSsb6, 160, 160},
	};
	for (const Case56& c : cases) {
		sslm_status st = SSLM_OK;
		bool consulted = false, out_null = false;
		Build56(c, blob, &st, &consulted, &out_null);
		// kills: a new verb with no allocation-failure path (the injected fault never reaches it, or it
		// escapes the C boundary).
		PKV_CHECK_MSG(consulted, "5.6 %s: the injection seam was never consulted", c.name);
		PKV_CHECK_MSG(st == SSLM_ALLOCATION_FAILED && out_null, "5.6 %s: -> %d (out %s)", c.name, static_cast<int>(st),
		              out_null ? "null" : "set");
		// kills: an allocation failure that unwinds without returning the pages already drawn.
		ProbeExactlyFree(
		    fx,
		    [&c, &blob] {
			    sslm_status s2;
			    bool a, b;
			    return Build56(c, blob, &s2, &a, &b);
		    },
		    c.free_after);
	}
}

PKV_CELL("5.1/C5", "C5", Cell51);
PKV_CELL("5.2/C5", "C5", Cell52);
PKV_CELL("5.3/C5", "C5", Cell53);
PKV_CELL("5.4/C5", "C5", Cell54);
PKV_CELL("5.5/C5", "C5", Cell55);
PKV_CELL("5.6/C5", "C5", Cell56);

}  // namespace

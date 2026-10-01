// Paged-KV plan (rev 16.1) §7 dimension 1, lifetime and reuse: the C5-owned cells and parts (budget
// holders and page pools). Red by link at C1: every cell here calls a §3.6 verb C5 defines, or a C4
// seam.
//
// Grading (§7, the three rules): tokens and K/V rows against the R0 reference (v1.11.0, the plan's
// "v1.9.0") or a fresh/unsaved twin, compared as tokens and rows because a budget holder saves SSB6
// and the reference 1.9.0's SSB5; page counts by the two-sided fill probe (§8), never by
// sslm_kv_pool_stats or sslm_seq_kv_stats; freed pages by the page-index peek reading 0xCD.

#include "pkv_budget_a_helpers.h"

namespace {

using namespace pkv;
using namespace pkv::budget_a;

// ---- 1.1 [C5 budget] ----------------------------------------------------------------------------
// One budget handle driven fresh -> prefill 100 -> reset -> prefill 40 -> reset -> adopt -> decode,
// crossing page boundaries at every stage (B = 16): the R0 lifecycle scenario with a budget prefix
// and a budget holder. Tokens and the K/V rows of its SSB6 blob equal v1.11.0's at each stage.
void Cell11Budget() {
	for (const char* stem : {"pkv_def"}) {
		const Fixture& fx = GetFixture(stem);
		if (!fx.ok) continue;
		PagePool pool(fx.model, 64);
		PKV_CHECK_EQ(pool.status, SSLM_OK);
		BRun d(fx, "SSB6");  // kills: a budget holder that saves 1.9.0's SSB5 (the legacy writer)
		BudgetLifecycle(d, &pool.pool, 128, 300);
		PKV_CHECK_MSG(d.fail.empty(), "budget lifecycle on %s: %s", stem, d.fail.c_str());
		// kills: any stage whose tokens or K/V rows leave the reference: a reset that leaves stale rows
		// visible, a share adopt that maps the wrong pages, a tail copy of the wrong bytes.
		ExpectMatchesReference("v1.11.0", fx, "lifecycle", d.records, Compare::kTokensAndRows);
	}
}

// ---- 1.3 [C5] -----------------------------------------------------------------------------------
// 1,000 mixed create / adopt / reset / release cycles (budget and legacy holders, prefixes released
// and rebuilt while adopters live), then a full teardown: the fill probe admits exactly the pool.
struct Mix13Counts {
	int creates = 0, legacy_creates = 0, adopts = 0, resets = 0, releases = 0, prefix_rebuilds = 0;
};

constexpr uint32_t k13Pool = 400;

std::unique_ptr<ProbeState> Build13(Mix13Counts* counts) {
	auto sc = std::make_unique<Scene>();
	const Fixture& fx = GetFixture("pkv_def");
	PagePool& pool = sc->AddPool(fx.model, k13Pool);
	if (pool.status != SSLM_OK) return sc;
	Mix13Counts n;
	Lcg g(13);
	const int32_t lens[3] = {37, 300, 1000};  // mid-page, mid-page, mid-page: every adopt copies a tail
	sslm_prefix px[3] = {};
	auto make_prefix = [&](int k) {
		px[k] = FrozenBudgetPrefix(fx, &pool.pool, Stream(130 + static_cast<uint32_t>(k), lens[k], fx.vocab), sc->h);
	};
	for (int k = 0; k < 3; ++k) make_prefix(k);
	std::vector<sslm_seq> live;
	auto pick = [&]() { return live[g.Next() % live.size()]; };
	for (int c = 0; c < 1000; ++c) {
		switch (g.Next() % 8) {
			case 0:
			case 1: {  // budget create: budgets at and around a page, and two larger
				if (live.size() >= 12) break;
				static const int32_t kBudgets[] = {1, 15, 16, 17, 64, 200};
				sslm_seq s = nullptr;
				if (sslm_seq_create_budgeted(fx.model, &pool.pool, kBudgets[g.Next() % 6], &s) == SSLM_OK && s) {
					live.push_back(s);
					sc->h.seqs.push_back(s);
					++n.creates;
				}
				break;
			}
			case 2: {  // legacy create: 256 pages, admitted only when the mix leaves room
				if (live.size() >= 12) break;
				sslm_seq s = nullptr;
				if (sslm_seq_create(fx.model, &pool.pool, &s) == SSLM_OK && s) {
					live.push_back(s);
					sc->h.seqs.push_back(s);
					++n.legacy_creates;
				}
				break;
			}
			case 3:
			case 4: {  // adopt onto a used or fresh holder, then decode one or two tokens
				if (live.empty()) break;
				sslm_seq s = pick();
				const int k = static_cast<int>(g.Next() % 3);
				if (px[k] && sslm_seq_adopt_prefix(s, px[k]) == SSLM_OK) {
					++n.adopts;
					DecodeN(fx.model, s, 1 + static_cast<int>(g.Next() % 2), nullptr);  // a refusal at a tiny budget is fine
				}
				break;
			}
			case 5: {  // reset, then write a few rows
				if (live.empty()) break;
				sslm_seq s = pick();
				if (sslm_seq_reset(s) == SSLM_OK) ++n.resets;
				int64_t got = 0;
				PrefillAll(fx.model, s, Stream(900 + static_cast<uint32_t>(c), 3, fx.vocab), 3, &got);
				break;
			}
			case 6: {  // release
				if (live.empty()) break;
				const size_t i = g.Next() % live.size();
				if (ReleaseSeq(sc->h, live[i]) == SSLM_OK) ++n.releases;
				live.erase(live.begin() + static_cast<std::ptrdiff_t>(i));
				break;
			}
			case 7: {  // release a prefix while its adopters live, and build it again
				const int k = static_cast<int>(g.Next() % 3);
				if (px[k]) ReleasePrefix(sc->h, px[k]);
				make_prefix(k);
				++n.prefix_rebuilds;
				break;
			}
		}
	}
	for (sslm_seq s : live) ReleaseSeq(sc->h, s);
	for (sslm_prefix p : px)
		if (p) ReleasePrefix(sc->h, p);
	sc->h.ReleaseAll();  // and any half-built prefix a refused rebuild left behind
	if (counts) *counts = n;
	return sc;
}

void Cell13() {
	const Fixture& fx = GetFixture("pkv_def");
	if (!fx.ok) return;
	Mix13Counts n;
	Build13(&n);  // one pass to see the mix ran the verbs it names
	PKV_CHECK_MSG(n.creates >= 100 && n.adopts >= 100 && n.resets >= 50 && n.releases >= 50 && n.prefix_rebuilds >= 20,
	              "1.3's mix ran %d creates, %d adopts, %d resets, %d releases, %d prefix rebuilds", n.creates, n.adopts,
	              n.resets, n.releases, n.prefix_rebuilds);
	// kills: a page leaked (k + 1 refused at k) or freed twice (k + 1 admitted) anywhere in the mix.
	ProbeExactlyFree(fx, [] { return Build13(nullptr); }, k13Pool);
}

// ---- 1.4 [C5] -----------------------------------------------------------------------------------
// A prefix released while budget adopters live: each adopter continues byte-equal to the run where
// it stayed (R0's prefix_lengths p1000 stages, in which the prefix lives throughout). After the last
// adopter releases, the fill probe admits the whole pool.
constexpr uint32_t k14Pool = 160;
constexpr int k14Adopters = 3;
constexpr int32_t k14Budget = 32;  // the adopter writes [1000, 1030)

enum class Stop14 { kAfterPrefixRelease, kTeardown };

std::unique_ptr<ProbeState> Build14(Stop14 stop, std::vector<BRun>* runs) {
	auto sc = std::make_unique<Scene>();
	const Fixture& fx = GetFixture("pkv_def");
	PagePool& pool = sc->AddPool(fx.model, k14Pool);
	if (pool.status != SSLM_OK) return sc;
	sslm_prefix p = FrozenBudgetPrefix(fx, &pool.pool, Stream(51, 1000, fx.vocab), sc->h);
	std::vector<BRun> local;
	std::vector<BRun>& d = runs ? *runs : local;
	d.clear();
	std::vector<sslm_seq> a;
	for (int i = 0; i < k14Adopters; ++i) {
		d.emplace_back(fx);
		sslm_seq s = BudgetSeq(fx, &pool.pool, k14Budget, sc->h);
		a.push_back(s);
		d[i].Ok(s && p ? SSLM_OK : SSLM_INVALID_ARGUMENT, "setup");
		if (s && p) d[i].Ok(sslm_seq_adopt_prefix(s, p), "adopt") && d[i].Mark("p1000_adopted", s);
	}
	for (int i = 0; i < k14Adopters; ++i) d[i].Decode(a[i], 4);
	if (p) ReleasePrefix(sc->h, p);  // refcount of the 62 shared pages: 4 -> 3, the prefix's tail page freed
	if (stop == Stop14::kAfterPrefixRelease) return sc;
	for (int i = 0; i < k14Adopters; ++i) d[i].Decode(a[i], 4) && d[i].Mark("p1000_decode8", a[i]);
	for (int i = 0; i < k14Adopters; ++i)
		d[i].Prefill(a[i], Stream(52, 20, fx.vocab), 64) && d[i].Decode(a[i], 4) &&
		    d[i].Mark("p1000_prefill20+decode4", a[i]);
	for (sslm_seq s : a)
		if (s) ReleaseSeq(sc->h, s);
	sc->h.ReleaseAll();
	return sc;
}

void Cell14() {
	const Fixture& fx = GetFixture("pkv_def");
	if (!fx.ok) return;
	std::vector<BRun> runs;
	Build14(Stop14::kTeardown, &runs);
	PKV_CHECK_EQ(runs.size(), k14Adopters);
	for (BRun& d : runs) {
		PKV_CHECK_MSG(d.fail.empty(), "1.4 adopter: %s", d.fail.c_str());
		PKV_CHECK_EQ(d.records.size(), 3);
		// kills: a prefix release that frees pages its adopters still map (the next rows read another
		// holder's bytes, or poison) -- the adopter leaves the run in which the prefix stayed.
		for (const Record& r : d.records) ExpectRecord("v1.11.0", fx, "prefix_lengths", r, "1.4 adopter");
	}
	// With the adopters live, the prefix's release returns only its own tail page: the pool holds the
	// 62 shared pages and each adopter's R(32) = 3. kills: a prefix release that frees shared pages
	// (more free) or keeps its tail page (fewer).
	const int64_t live_free = k14Pool - 62 - k14Adopters * fx.R(k14Budget);
	ProbeExactlyFree(fx, [] { return Build14(Stop14::kAfterPrefixRelease, nullptr); }, live_free);
	// kills: the last adopter's release leaving the shared pages referenced (or freeing one twice).
	ProbeExactlyFree(fx, [] { return Build14(Stop14::kTeardown, nullptr); }, k14Pool);
}

// ---- 1.5 [C5] -----------------------------------------------------------------------------------
// Nested: a world -> 10 personas (begin_from) -> 2 sequences each. The world is released, then the
// personas, then the sequences, each group in a shuffled order, with every live sequence decoding
// between releases. Every sequence's tokens and final K/V rows equal a fresh unshared twin's (the
// same prompt prefilled whole into one budget holder), and the final fill probe admits the pool.
constexpr uint32_t k15Pool = 200;
constexpr int k15Personas = 10, k15PerPersona = 2;
constexpr int32_t k15SeqBudget = 48;

struct Seq15 {
	int persona = 0, j = 0;
	sslm_seq s = nullptr;
	std::vector<int32_t> tokens;
	std::vector<uint8_t> final_blob;
};

std::vector<int32_t> World15(const Fixture& fx) { return Stream(81, 300, fx.vocab); }
std::vector<int32_t> Persona15(const Fixture& fx, int i) {
	return Stream(90 + static_cast<uint32_t>(i), 10 + 3 * i, fx.vocab);
}
std::vector<int32_t> Suffix15(const Fixture& fx, int i, int j) {
	return Stream(200 + 2 * static_cast<uint32_t>(i) + static_cast<uint32_t>(j), 5, fx.vocab);
}

std::unique_ptr<ProbeState> Build15(std::vector<Seq15>* out) {
	auto sc = std::make_unique<Scene>();
	const Fixture& fx = GetFixture("pkv_def");
	PagePool& pool = sc->AddPool(fx.model, k15Pool);
	if (pool.status != SSLM_OK) return sc;
	sslm_prefix world = FrozenBudgetPrefix(fx, &pool.pool, World15(fx), sc->h);
	std::vector<sslm_prefix> personas;
	for (int i = 0; i < k15Personas && world; ++i) {
		sslm_prefix p = nullptr;
		if (sslm_prefix_begin_from(world, 64, &p) != SSLM_OK || !p) break;
		sc->h.prefixes.push_back(p);
		PrefixPrefillAll(fx.model, p, Persona15(fx, i), 16);
		sslm_prefix_freeze(p);
		personas.push_back(p);
	}
	std::vector<Seq15> seqs;
	for (int i = 0; i < static_cast<int>(personas.size()); ++i)
		for (int j = 0; j < k15PerPersona; ++j) {
			Seq15 q;
			q.persona = i;
			q.j = j;
			q.s = BudgetSeq(fx, &pool.pool, k15SeqBudget, sc->h);
			if (q.s && sslm_seq_adopt_prefix(q.s, personas[static_cast<size_t>(i)]) == SSLM_OK)
				PrefillAll(fx.model, q.s, Suffix15(fx, i, j), 8);
			seqs.push_back(std::move(q));
		}
	auto decode_live = [&](int n) {
		for (Seq15& q : seqs)
			if (q.s) DecodeN(fx.model, q.s, n, &q.tokens);
	};
	Lcg g(15);
	decode_live(4);
	if (world) ReleasePrefix(sc->h, world);
	decode_live(4);
	std::vector<sslm_prefix> order = personas;
	g.Shuffle(order);
	for (sslm_prefix p : order) {
		ReleasePrefix(sc->h, p);
		decode_live(1);
	}
	std::vector<size_t> sorder(seqs.size());
	for (size_t i = 0; i < sorder.size(); ++i) sorder[i] = i;
	g.Shuffle(sorder);
	for (size_t i : sorder) {
		Seq15& q = seqs[i];
		if (q.s) {
			q.final_blob = Save(q.s);
			ReleaseSeq(sc->h, q.s);
			q.s = nullptr;
		}
		decode_live(1);
	}
	sc->h.ReleaseAll();
	if (out) *out = std::move(seqs);
	return sc;
}

void Cell15() {
	const Fixture& fx = GetFixture("pkv_def");
	if (!fx.ok) return;
	std::vector<Seq15> seqs;
	Build15(&seqs);
	PKV_CHECK_EQ(seqs.size(), k15Personas * k15PerPersona);
	for (const Seq15& q : seqs) {
		// The twin: the whole prompt prefilled into one fresh budget holder of its own, no sharing.
		std::vector<int32_t> prompt = World15(fx);
		const std::vector<int32_t> pp = Persona15(fx, q.persona), sx = Suffix15(fx, q.persona, q.j);
		prompt.insert(prompt.end(), pp.begin(), pp.end());
		prompt.insert(prompt.end(), sx.begin(), sx.end());
		Scene tw;
		PagePool& pool = tw.AddPool(fx.model, static_cast<uint32_t>(fx.CapPages()));
		sslm_seq t = BudgetSeq(fx, &pool.pool, static_cast<int32_t>(fx.geo.context_cap), tw.h);
		std::vector<int32_t> want;
		if (t) {
			PrefillAll(fx.model, t, prompt, 64);
			DecodeN(fx.model, t, static_cast<int>(q.tokens.size()), &want);
		}
		const std::vector<uint8_t> twin_blob = Save(t);
		PKV_CHECK_MSG(!q.tokens.empty() && q.tokens.size() >= 18, "1.5 seq (%d,%d) decoded %zu tokens", q.persona, q.j,
		              q.tokens.size());
		// kills: a release (of the world, a persona, or a sibling) that frees or poisons a page a live
		// sequence still maps -- its next tokens or rows leave the unshared twin's.
		PKV_CHECK_MSG(q.tokens == want, "1.5 seq (%d,%d): tokens differ from the unshared twin", q.persona, q.j);
		const int64_t L = BlobContextLength(q.final_blob);
		PKV_CHECK_MSG(L > 0 && L == BlobContextLength(twin_blob) && RowsSha(fx, q.final_blob, L) == RowsSha(fx, twin_blob, L),
		              "1.5 seq (%d,%d): final K/V rows differ from the unshared twin", q.persona, q.j);
	}
	// kills: a nested release order that leaks a shared page or frees one twice.
	ProbeExactlyFree(fx, [] { return Build15(nullptr); }, k15Pool);
}

// ---- 1.6 [C5, the page-pool variant] -----------------------------------------------------------
// Dirty accounting. The caller's buffer is filled with 0x5A before sslm_kv_page_pool_create. A
// budget-512 holder (33 reserve pages) writes 100 rows (7 pages), is reset, then released. Every page
// it wrote reads 0xCD through the page-index peek, and every other page of the pool -- the 26 clean
// reserve pages and the 7 never drawn -- still reads 0x5A.
void Cell16Budget() {
	const Fixture& fx = GetFixture("pkv_def");
	if (!fx.ok) return;
	constexpr uint32_t kPages = 40;
	PagePool pool(fx.model, kPages, 0x5A);
	PKV_CHECK_EQ(pool.status, SSLM_OK);
	sslm_seq s = nullptr;
	PKV_CHECK_EQ(sslm_seq_create_budgeted(fx.model, &pool.pool, 512, &s), SSLM_OK);
	if (!s) return;
	PKV_CHECK_EQ(PrefillAll(fx.model, s, Stream(1, 100, fx.vocab), 64), SSLM_OK);
	const std::vector<uint32_t> written = SeqTable(s);
	PKV_CHECK_EQ(written.size(), CeilDiv(100, fx.B()));
	PKV_CHECK_EQ(sslm_seq_reset(s), SSLM_OK);
	PKV_CHECK_EQ(sslm_seq_release(s), SSLM_OK);
	const std::set<uint32_t> dirty(written.begin(), written.end());
	int poisoned = 0, clean = 0, wrong = 0;
	for (uint32_t p = 0; p < kPages; ++p) {
		const std::vector<uint8_t> b = PeekPage(fx, pool.pool, p);
		if (dirty.count(p)) {
			poisoned += AllBytes(b, 0xCD);
			wrong += !AllBytes(b, 0xCD);  // kills: poison removed (the S4 pin's mutant)
		} else {
			clean += AllBytes(b, 0x5A);
			wrong += !AllBytes(b, 0x5A);  // kills: release ignoring the dirty flag (a 0x5A page turns 0xCD)
		}
	}
	PKV_CHECK_MSG(wrong == 0 && poisoned == static_cast<int>(dirty.size()) && clean == static_cast<int>(kPages - dirty.size()),
	              "1.6: %d written pages read 0xCD (of %zu), %d other pages read 0x5A (of %zu), %d wrong", poisoned,
	              dirty.size(), clean, kPages - dirty.size(), wrong);
}

// ---- 1.10 [C5] ----------------------------------------------------------------------------------
// Adopt onto a used budget holder. The census population of
// tests/t2899-schema-deadend-red-suite/cell_adopt_prefix_census.cpp -- 10 resting states x 3 prefix
// kinds x {live, restored} -- run in budget mode: budget prefixes in a page pool, budget holders.
// Per row: the adopt's status, the whole SSB6 blob and the next 4 tokens equal a fresh budget holder
// (same budget, same schema binding) adopting the same prefix; the fill probe admits exactly
// pool - prefix pages - R(budget) straight after the adopt; and once the adopter is released, every
// page it held privately reads 0xCD while the shared pages read unchanged.
//
// The census's prompts are widened so pages are in play: the prefix is 37 tokens (2 full pages
// shared, a 5-row tail copied), and each adopter's own prompt is 40 tokens (3 pages), so adopting
// onto a used holder unmaps real private pages.
constexpr uint32_t k110Pool = 80;
constexpr int32_t k110Budget = 256;
constexpr int32_t k110PrefixLen = 37;

enum Pk110 { P_PROMPT, P_BOUND_START, P_PROGRESS };
enum Sk110 { FRESH, POST_PREFILL, POST_GREEDY, MIDTOKEN, B_FRESH, B_POST_PREFILL, B_ADVANCED, B_FORCED, B_DEADEND,
             B_FORCED_DEAD };
const char* const kPk110[] = {"P_PROMPT", "P_BOUND_START", "P_PROGRESS"};
const char* const kSk110[] = {"FRESH",      "POST_PREFILL", "POST_GREEDY", "MIDTOKEN", "B_FRESH",
                              "B_POST_PREFILL", "B_ADVANCED", "B_FORCED",    "B_DEADEND", "B_FORCED_DEAD"};

// The census's Greedy: one decode call per step at layer budget `budget`.
std::vector<int32_t> Greedy110(const Fixture& fx, sslm_seq s, int steps, int32_t budget) {
	sslm_decode_params p{};
	p.layer_budget = budget;
	sslm_seq b[1] = {s};
	std::vector<int32_t> t;
	for (int i = 0; i < steps; ++i) {
		int32_t tok = -7;
		const sslm_status st = sslm_decode_step(fx.model, b, 1, &p, nullptr, &tok);
		t.push_back(st == SSLM_OK ? tok : -1000 - static_cast<int>(st));
	}
	return t;
}

bool PrefillPrompt110(const Fixture& fx, sslm_seq s) {
	int64_t got = 0;
	return PrefillAll(fx.model, s, Stream(72, 40, fx.vocab), 8, &got) == SSLM_OK && got == 40;
}

bool EnterMidToken110(const Fixture& fx, sslm_seq s) {
	sslm_decode_params p{};
	p.layer_budget = 1;
	sslm_seq b[1] = {s};
	int32_t tok = 0;
	if (sslm_decode_step(fx.model, b, 1, &p, nullptr, &tok) != SSLM_OK) return false;
	tok = 0;
	if (sslm_decode_step(fx.model, b, 1, &p, nullptr, &tok) != SSLM_OK) return false;
	return tok < 0;
}

struct Census110 {
	sslm_schema schema = nullptr;
	int32_t admitted = -1;
	int32_t L = 2;
};

// Builds one row in a fresh scene: pool, prefix of kind pk, the holder resting in state sk (restored
// through its own SSB6 save when asked), and the adopt. Returns the adopt's status; the adopter is
// sc.h.seqs.back(), and `before` receives its table straight before the adopt.
sslm_status Row110(const Fixture& fx, const Census110& c, int pk, int sk, bool restored, Scene& sc, sslm_prefix* prefix_out,
                   sslm_seq* target_out, std::vector<uint32_t>* before) {
	*target_out = nullptr;
	PagePool& pool = sc.AddPool(fx.model, k110Pool);
	if (pool.status != SSLM_OK) return pool.status;
	sslm_prefix prefix = nullptr;
	if (sslm_prefix_begin_budgeted(fx.model, &pool.pool, 64, &prefix) != SSLM_OK || !prefix) return SSLM_INVALID_ARGUMENT;
	sc.h.prefixes.push_back(prefix);
	*prefix_out = prefix;
	if (pk != P_PROMPT && sslm_prefix_set_schema(prefix, c.schema) != SSLM_OK) return SSLM_INVALID_ARGUMENT;
	if (PrefixPrefillAll(fx.model, prefix, Stream(71, k110PrefixLen, fx.vocab), 8) != SSLM_OK) return SSLM_INVALID_ARGUMENT;
	if (pk == P_PROGRESS) {
		int32_t c2 = 0;
		if (sslm_prefix_prefill(fx.model, prefix, &c.admitted, 1, 8, SSLM_SPAN_SCHEMA_CONTENT, nullptr, &c2) != SSLM_OK ||
		    c2 != 1)
			return SSLM_INVALID_ARGUMENT;
	}
	if (sslm_prefix_freeze(prefix) != SSLM_OK) return SSLM_INVALID_ARGUMENT;
	const bool bound = sk >= B_FRESH;
	sslm_seq s = BudgetSeq(fx, &pool.pool, k110Budget, sc.h);
	if (!s || (bound && sslm_seq_set_schema(s, c.schema) != SSLM_OK)) return SSLM_INVALID_ARGUMENT;
	bool ok = true;
	switch (sk) {
		case FRESH:
		case B_FRESH:
			break;
		case POST_PREFILL:
		case B_POST_PREFILL:
			ok = PrefillPrompt110(fx, s);
			break;
		case POST_GREEDY:
			ok = PrefillPrompt110(fx, s);
			Greedy110(fx, s, 2, c.L);
			break;
		case MIDTOKEN:
			ok = PrefillPrompt110(fx, s) && EnterMidToken110(fx, s);
			break;
		case B_ADVANCED:
			ok = PrefillPrompt110(fx, s);
			Greedy110(fx, s, 1, c.L);
			break;
		case B_FORCED:
		case B_FORCED_DEAD: {
			int32_t n = 0;
			ok = PrefillPrompt110(fx, s) &&
			     sslm_prefill(fx.model, s, &c.admitted, 1, 8, SSLM_SPAN_SCHEMA_CONTENT, nullptr, &n) == SSLM_OK && n == 1;
			if (sk == B_FORCED_DEAD) Greedy110(fx, s, 2, c.L);
			break;
		}
		case B_DEADEND:
			ok = PrefillPrompt110(fx, s);
			Greedy110(fx, s, 3, c.L);
			break;
	}
	if (!ok) return SSLM_INVALID_ARGUMENT;
	sslm_seq target = s;
	if (restored) {
		const std::vector<uint8_t> b = Save(s);
		target = nullptr;
		if (sslm_seq_restore(fx.model, &pool.pool, b.data(), b.size(), &target) != SSLM_OK || !target)
			return SSLM_INVALID_ARGUMENT;
		sc.h.seqs.push_back(target);
		ReleaseSeq(sc.h, s);
	}
	*target_out = target;
	if (before) *before = SeqTable(target);
	return sslm_seq_adopt_prefix(target, prefix);
}

// The checks 1.10 and 1.11 share, on an adopter straight after its adopt: blob and next 4 tokens
// against a fresh budget holder adopting the same prefix (bound to `schema` when given), then the
// adopter's release poisons its private pages and leaves the shared ones as they were.
void ExpectLikeFreshAdopter(const Fixture& fx, sslm_kv_pool* pool, Scene& sc, sslm_seq target, sslm_prefix prefix,
                            int32_t budget, sslm_schema schema, int32_t layer_budget,
                            const std::vector<uint32_t>& before, const char* what) {
	const std::vector<uint8_t> got_blob = Save(target);
	const std::vector<int32_t> got_next = Greedy110(fx, target, 4, layer_budget);
	const std::vector<uint32_t> after = SeqTable(target);
	const int64_t origin = BlobContextLength(got_blob);
	const size_t shared = static_cast<size_t>(origin / fx.B());
	std::set<uint32_t> freed(before.begin(), before.end());
	std::vector<std::vector<uint8_t>> shared_bytes;
	for (size_t i = 0; i < after.size(); ++i) {
		if (i < shared) shared_bytes.push_back(PeekPage(fx, *pool, after[i]));
		else freed.insert(after[i]);
	}
	PKV_CHECK_EQ(ReleaseSeq(sc.h, target), SSLM_OK);
	// kills: a release that skips the poison of a dirty private page, or that sends one to nothing.
	if (!freed.empty()) ExpectPoisoned(fx, *pool, freed, what);
	for (size_t i = 0; i < shared && i < after.size(); ++i)  // kills: a sharer's release that frees a page the prefix still holds
		PKV_CHECK_MSG(PeekPage(fx, *pool, after[i]) == shared_bytes[i], "%s: shared page %u changed at the adopter's release",
		              what, after[i]);
	sslm_seq ref = BudgetSeq(fx, pool, budget, sc.h);
	if (ref && schema) PKV_CHECK_EQ(sslm_seq_set_schema(ref, schema), SSLM_OK);
	PKV_CHECK_EQ(ref ? sslm_seq_adopt_prefix(ref, prefix) : SSLM_INVALID_ARGUMENT, SSLM_OK);
	const std::vector<uint8_t> want_blob = Save(ref);
	const std::vector<int32_t> want_next = Greedy110(fx, ref, 4, layer_budget);
	if (ref) ReleaseSeq(sc.h, ref);
	PKV_CHECK_MSG(IsMagic(got_blob, "SSB6"), "%s: the adopter's save is SSB6", what);
	// kills: adopt onto a used holder that keeps stale state (walk, forced count, rows past origin) or
	// maps the wrong pages: the whole SSB6 blob differs from a fresh adopter's.
	PKV_CHECK_MSG(!got_blob.empty() && got_blob == want_blob, "%s: SSB6 blob differs from a fresh adopter's", what);
	PKV_CHECK_MSG(got_next == want_next, "%s: next 4 tokens differ from a fresh adopter's", what);
}

void Cell110() {
	const Fixture& fx = GetFixture("pkv_def");
	if (!fx.ok) return;
	Census110 c;
	PKV_CHECK_EQ(sslm_schema_lookup(fx.model, "g5_minimal_one_field", &c.schema), SSLM_OK);
	{  // one schema-admitted token from the start state, as the census derives it
		Scene sc;
		PagePool& pool = sc.AddPool(fx.model, 40);
		sslm_seq d = BudgetSeq(fx, &pool.pool, k110Budget, sc.h);
		if (d && sslm_seq_set_schema(d, c.schema) == SSLM_OK && PrefillPrompt110(fx, d)) c.admitted = Greedy110(fx, d, 1, c.L)[0];
	}
	PKV_CHECK_MSG(c.admitted >= 0, "1.10: no admitted token from the start state");
	if (c.admitted < 0) return;
	const int64_t k = static_cast<int64_t>(k110Pool) - PrefixPages(fx, k110PrefixLen + 1) - fx.R(k110Budget);
	int rows = 0;
	for (int pk = P_PROMPT; pk <= P_PROGRESS; ++pk)
		for (int sk = FRESH; sk <= B_FORCED_DEAD; ++sk)
			for (int restored = 0; restored <= 1; ++restored) {
				char what[96];
				std::snprintf(what, sizeof what, "1.10 %s %s %s", kPk110[pk], kSk110[sk], restored ? "restored" : "live");
				Scene sc;
				sslm_prefix prefix = nullptr;
				sslm_seq target = nullptr;
				std::vector<uint32_t> before;
				const sslm_status st = Row110(fx, c, pk, sk, restored != 0, sc, &prefix, &target, &before);
				PKV_CHECK_MSG(st == SSLM_OK, "%s: adopt -> %d", what, static_cast<int>(st));
				++rows;
				if (st != SSLM_OK || !target) continue;
				ExpectLikeFreshAdopter(fx, &sc.pools.front()->pool, sc, target, prefix, k110Budget,
				                       sk >= B_FRESH ? c.schema : nullptr, c.L, before, what);
				// kills: adopt drawing from the pool, a used holder's private pages sent to the pool
				// (more free) or held beyond R(budget) (fewer free), a restored holder's materialized
				// pages not returned.
				ProbeExactlyFree(
				    fx,
				    [&]() -> std::unique_ptr<ProbeState> {
					    auto s2 = std::make_unique<Scene>();
					    sslm_prefix p2 = nullptr;
					    sslm_seq t2 = nullptr;
					    Row110(fx, c, pk, sk, restored != 0, *s2, &p2, &t2, nullptr);
					    return s2;
				    },
				    k);
			}
	PKV_CHECK_EQ(rows, 60);
}

// ---- 1.11 [C5] ----------------------------------------------------------------------------------
// (a) Re-adopt of the same prefix without a reset: each shared page goes -1 then +1 inside one
// critical section. (b) A holder sharing prefix P, after P's release, adopts prefix Q: P's pages
// reach 0 through the adopt and are freed (they read 0xCD at once). The assertions of 1.10.
constexpr uint32_t k111Pool = 120;
constexpr int32_t k111Budget = 64;

std::unique_ptr<ProbeState> Build111(bool case_b, sslm_prefix* adopted, sslm_seq* holder, std::vector<uint32_t>* before,
                                     std::vector<uint32_t>* p_pages, sslm_status* st) {
	auto sc = std::make_unique<Scene>();
	const Fixture& fx = GetFixture("pkv_def");
	PagePool& pool = sc->AddPool(fx.model, k111Pool);
	*st = SSLM_INVALID_ARGUMENT;
	if (pool.status != SSLM_OK) return sc;
	sslm_prefix p = FrozenBudgetPrefix(fx, &pool.pool, Stream(51, 1000, fx.vocab), sc->h);
	sslm_prefix q = case_b ? FrozenBudgetPrefix(fx, &pool.pool, Stream(53, 300, fx.vocab), sc->h) : nullptr;
	sslm_seq h = BudgetSeq(fx, &pool.pool, k111Budget, sc->h);
	if (!p || !h || (case_b && !q) || sslm_seq_adopt_prefix(h, p) != SSLM_OK) return sc;
	DecodeN(fx.model, h, 5, nullptr);
	if (before) {  // the holder's private entries; the leading 62 are P's shared pages
		const std::vector<uint32_t> t = SeqTable(h);
		before->assign(t.begin() + std::min<size_t>(t.size(), 1000 / 16), t.end());
	}
	if (case_b) {
		if (p_pages) {
			const std::vector<uint32_t> t = SeqTable(h);
			p_pages->assign(t.begin(), t.begin() + std::min<size_t>(t.size(), 1000 / 16));
		}
		ReleasePrefix(sc->h, p);  // h is now the last sharer of P's 62 full pages
		*st = sslm_seq_adopt_prefix(h, q);
		*adopted = q;
	} else {
		*st = sslm_seq_adopt_prefix(h, p);  // no reset
		*adopted = p;
	}
	*holder = h;
	return sc;
}

void Cell111() {
	const Fixture& fx = GetFixture("pkv_def");
	if (!fx.ok) return;
	for (int case_b = 0; case_b <= 1; ++case_b) {
		const char* what = case_b ? "1.11(b) adopt of Q by P's last sharer" : "1.11(a) re-adopt without reset";
		sslm_prefix adopted = nullptr;
		sslm_seq h = nullptr;
		std::vector<uint32_t> before, p_pages;
		sslm_status st = SSLM_OK;
		std::unique_ptr<ProbeState> ps = Build111(case_b != 0, &adopted, &h, &before, &p_pages, &st);
		Scene& sc = static_cast<Scene&>(*ps);
		PKV_CHECK_MSG(st == SSLM_OK, "%s -> %d", what, static_cast<int>(st));
		if (st != SSLM_OK || !h) continue;
		if (case_b) {
			PKV_CHECK_EQ(p_pages.size(), 62);
			// kills: shared pages reaching 0 through adopt sent to the holder's reserve (by a refcount
			// reading) instead of freed -- they would keep P's bytes, not 0xCD.
			ExpectPoisoned(fx, sc.pools.front()->pool, std::set<uint32_t>(p_pages.begin(), p_pages.end()), what);
		}
		ExpectLikeFreshAdopter(fx, &sc.pools.front()->pool, sc, h, adopted, k111Budget, nullptr, 2, before, what);
		// (a): P's 63 pages + the holder's R(64); (b): Q's 19 pages + R(64), P's 63 all free again.
		// kills: (a) a re-adopt that leaks or double-draws a page across its -1/+1 (more or fewer free);
		// (b) P's pages kept in the holder's reserve or never freed.
		// Not separated here: the -1/+1 split over two critical sections. P is live in (a), so every page
		// the holder shares is at refcount >= 2 and the -1 never reaches 0; a count of 1 needs P released,
		// and re-adopting a released prefix is a use of a released handle (caller UB, §3.3 G11). The
		// path where a shared page does reach 0 through adopt is (b)'s.
		const int64_t k = k111Pool - (case_b ? PrefixPages(fx, 300) : PrefixPages(fx, 1000)) - fx.R(k111Budget);
		ProbeExactlyFree(
		    fx,
		    [case_b]() {
			    sslm_prefix a = nullptr;
			    sslm_seq hh = nullptr;
			    sslm_status s2 = SSLM_OK;
			    return Build111(case_b != 0, &a, &hh, nullptr, nullptr, &s2);
		    },
		    k);
	}
}

// ---- 1.12 [C5] ----------------------------------------------------------------------------------
// sslm_kv_pool_destroy refuses SSLM_POOL_HAS_LIVE_HANDLES while a released prefix's pages are still
// shared by a live adopter, and returns SSLM_OK only after a full teardown in a shuffled order.
void Cell112() {
	const Fixture& fx = GetFixture("pkv_def");
	if (!fx.ok) return;
	for (uint32_t seed = 0; seed < 5; ++seed) {
		auto pool = std::make_unique<PagePool>(fx.model, 120);
		PKV_CHECK_EQ(pool->status, SSLM_OK);
		if (pool->status != SSLM_OK) return;
		Handles h;
		sslm_prefix p = FrozenBudgetPrefix(fx, &pool->pool, Stream(51, 1000, fx.vocab), h);
		std::vector<sslm_seq> a;
		for (int i = 0; i < 3; ++i) {
			sslm_seq s = BudgetSeq(fx, &pool->pool, 32, h);
			if (s && p && sslm_seq_adopt_prefix(s, p) == SSLM_OK) DecodeN(fx.model, s, 2, nullptr);
			a.push_back(s);
		}
		// The release order: seed 0 releases the prefix first (the case the cell names); the others
		// shuffle the prefix among its adopters.
		std::vector<int> order = {0, 1, 2, 3};  // 0 = the prefix, i = adopter i - 1
		if (seed > 0) Lcg(112 + seed).Shuffle(order);
		for (size_t i = 0; i < order.size(); ++i) {
			const sslm_status rs = order[i] == 0 ? ReleasePrefix(h, p) : ReleaseSeq(h, a[static_cast<size_t>(order[i] - 1)]);
			PKV_CHECK_EQ(rs, SSLM_OK);
			if (i + 1 < order.size()) {
				// kills: live_refs counting only prefixes, or dropped when the prefix's pages are
				// "released" -- a destroy that frees pages a live adopter still maps.
				PKV_CHECK_MSG(sslm_kv_pool_destroy(pool->pool) == SSLM_POOL_HAS_LIVE_HANDLES,
				              "1.12 seed %u: destroy with %zu handles live", seed, order.size() - i - 1);
			}
		}
		// kills: a handle count that never returns to 0 after a shared-page teardown.
		PKV_CHECK_EQ(sslm_kv_pool_destroy(pool->pool), SSLM_OK);
		pool->pool = nullptr;
	}
}

// ---- 1.13 [C5 budget prefix] --------------------------------------------------------------------
// sslm_prefix_freeze called twice. A budget-1,008 prefix of length 1,000 in a 300-page pool: R(1,008)
// = 64 drawn, 63 mapped, the first freeze returns 1, so 237 pages are free after it -- and after the
// second call too. The named mutant, "freeze returns R(budget) - ceil(len/B) pages, computed
// arithmetically, on every call", returns one more page on the second call (238); the two-sided fill
// probe at k = 237 rejects it. Again at budget 1,040 (3 pages: 237 against 240).
std::unique_ptr<ProbeState> Build113(int32_t budget, int freezes) {
	auto sc = std::make_unique<Scene>();
	const Fixture& fx = GetFixture("pkv_def");
	PagePool& pool = sc->AddPool(fx.model, 300);
	if (pool.status != SSLM_OK) return sc;
	sslm_prefix p = nullptr;
	if (sslm_prefix_begin_budgeted(fx.model, &pool.pool, budget, &p) != SSLM_OK || !p) return sc;
	sc->h.prefixes.push_back(p);
	PrefixPrefillAll(fx.model, p, Stream(51, 1000, fx.vocab), 64);
	for (int i = 0; i < freezes; ++i) PKV_CHECK_EQ(sslm_prefix_freeze(p), SSLM_OK);  // G33: idempotent, SSLM_OK
	return sc;
}

void Cell113Budget() {
	const Fixture& fx = GetFixture("pkv_def");
	if (!fx.ok) return;
	PKV_CHECK_EQ(fx.R(1008) - PrefixPages(fx, 1000), 1);
	PKV_CHECK_EQ(fx.R(1040) - PrefixPages(fx, 1000), 3);
	const int64_t k = 300 - PrefixPages(fx, 1000);  // 237
	for (int32_t budget : {1008, 1040}) {
		// kills: a first freeze that returns nothing, or the wrong count.
		ProbeExactlyFree(fx, [budget] { return Build113(budget, 1); }, k);
		// kills: freeze returns R(budget) - ceil(len/B) on every call (238 / 240 at the second call).
		ProbeExactlyFree(fx, [budget] { return Build113(budget, 2); }, k);
	}
}

PKV_CELL("1.1/C5", "C5", Cell11Budget);
PKV_CELL("1.3/C5", "C5", Cell13);
PKV_CELL("1.4/C5", "C5", Cell14);
PKV_CELL("1.5/C5", "C5", Cell15);
PKV_CELL("1.6/C5", "C5", Cell16Budget);
PKV_CELL("1.10/C5", "C5", Cell110);
PKV_CELL("1.11/C5", "C5", Cell111);
PKV_CELL("1.12/C5", "C5", Cell112);
PKV_CELL("1.13/C5", "C5", Cell113Budget);

}  // namespace

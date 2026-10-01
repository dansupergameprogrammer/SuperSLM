// Paged-KV plan (rev 16.1) §7 dimension 2, trust boundaries and hostile inputs: the C5-owned cells
// and parts (the budget verbs, the page pool, budget-mode SSB6, begin_from, the cross-pool refusal and
// sslm_seq_restore_shared's refusal paths). Red by link at C1.
//
// "Nothing drawn" is graded by the two-sided fill probe (§8) on a rebuilt state, never by the stats
// verbs; "unchanged" by a re-saved blob equal to the one saved before.

#include "pkv_budget_a_helpers.h"

#include "support/bad_alloc_injection.h"

#include <climits>

namespace {

using namespace pkv;
using namespace pkv::budget_a;

// ---- 2.1 [C5] -----------------------------------------------------------------------------------
// A budget <= 0, above cap, or INT32_MAX: SSLM_INVALID_ARGUMENT, a null out-handle and nothing drawn,
// on all three budget verbs (sslm_seq_create_budgeted, sslm_prefix_begin_budgeted,
// sslm_prefix_begin_from). sslm_kv_pages_for_budget returns 0 for each (§3.6).
std::vector<int32_t> BadBudgets(const Fixture& fx) {
	return {0, -1, INT32_MIN, static_cast<int32_t>(fx.geo.context_cap + 1), INT32_MAX};
}

constexpr uint32_t k21Pool = 64;

std::unique_ptr<ProbeState> Build21(std::vector<std::string>* bad) {
	auto sc = std::make_unique<Scene>();
	const Fixture& fx = GetFixture("pkv_def");
	PagePool& pool = sc->AddPool(fx.model, k21Pool);
	if (pool.status != SSLM_OK) return sc;
	sslm_prefix parent = FrozenBudgetPrefix(fx, &pool.pool, Stream(51, 37, fx.vocab), sc->h);  // keeps 3 pages
	for (int32_t b : BadBudgets(fx)) {
		sslm_seq s = reinterpret_cast<sslm_seq>(&pool);  // a non-null sentinel: the verb must null it
		sslm_prefix p = reinterpret_cast<sslm_prefix>(&pool), c = reinterpret_cast<sslm_prefix>(&pool);
		const sslm_status s1 = sslm_seq_create_budgeted(fx.model, &pool.pool, b, &s);
		const sslm_status s2 = sslm_prefix_begin_budgeted(fx.model, &pool.pool, b, &p);
		const sslm_status s3 = parent ? sslm_prefix_begin_from(parent, b, &c) : SSLM_INVALID_ARGUMENT;
		// A handle a refusal returned anyway is owned by the scene, so it is torn down with it.
		if (s1 == SSLM_OK && s) sc->h.seqs.push_back(s);
		if (s2 == SSLM_OK && p) sc->h.prefixes.push_back(p);
		if (s3 == SSLM_OK && c) sc->h.prefixes.push_back(c);
		if (bad && (s1 != SSLM_INVALID_ARGUMENT || s != nullptr))
			bad->push_back("create_budgeted(" + std::to_string(b) + ") -> " + std::to_string(s1));
		if (bad && (s2 != SSLM_INVALID_ARGUMENT || p != nullptr))
			bad->push_back("prefix_begin_budgeted(" + std::to_string(b) + ") -> " + std::to_string(s2));
		if (bad && (s3 != SSLM_INVALID_ARGUMENT || (parent && c != nullptr)))
			bad->push_back("prefix_begin_from(" + std::to_string(b) + ") -> " + std::to_string(s3));
	}
	if (bad && !parent) bad->push_back("the frozen parent was not built");
	return sc;
}

void Cell21() {
	const Fixture& fx = GetFixture("pkv_def");
	if (!fx.ok) return;
	for (int32_t b : BadBudgets(fx)) PKV_CHECK_EQ(sslm_kv_pages_for_budget(fx.model, b), 0);  // §3.6 sentence
	std::vector<std::string> bad;
	Build21(&bad);
	// kills: a budget clamped into [1, cap] instead of refused, and a refusal that leaves *out set.
	PKV_CHECK_MSG(bad.empty(), "2.1: %zu refusals wrong, first: %s", bad.size(), bad.empty() ? "" : bad[0].c_str());
	// kills: a verb that draws R(budget) before validating and keeps it on the refusal path.
	ProbeExactlyFree(fx, [] { return Build21(nullptr); }, k21Pool - PrefixPages(fx, 37));
	// The must-accept edges, so the refusals above are about the value: budgets 1 and cap admitted.
	Scene sc;
	PagePool& pool = sc.AddPool(fx.model, static_cast<uint32_t>(fx.CapPages() + 8));
	sslm_prefix parent = FrozenBudgetPrefix(fx, &pool.pool, Stream(51, 37, fx.vocab), sc.h);
	PKV_CHECK(BudgetSeq(fx, &pool.pool, 1, sc.h) != nullptr);
	sslm_prefix c = nullptr;
	PKV_CHECK_EQ(parent ? sslm_prefix_begin_from(parent, 1, &c) : SSLM_INVALID_ARGUMENT, SSLM_OK);
	if (c) sc.h.prefixes.push_back(c);
	sc.h.ReleaseAll();
	PKV_CHECK(BudgetSeq(fx, &pool.pool, static_cast<int32_t>(fx.geo.context_cap), sc.h) != nullptr);
}

// ---- 2.2 [C5] -----------------------------------------------------------------------------------
// sslm_kv_page_pool_create's refusals, in sslm_kv_pool_create's order: page_count 0; the
// page_count * page_bytes overflow; a buffer one byte short -> SSLM_BUFFER_TOO_SMALL; a misaligned
// buffer -> SSLM_MISALIGNED_BUFFER (and a short, misaligned one -> SSLM_BUFFER_TOO_SMALL first); and a
// pool of another model, refused by the budget verbs with nothing drawn.
void Cell22() {
	const Fixture& fx = GetFixture("pkv_def");
	const Fixture& other = GetFixture("pkv_qk");
	if (!fx.ok || !other.ok) return;
	// §3.1, §3.6: the sizing verbs agree with the formulas written in the test.
	PKV_CHECK_EQ(sslm_kv_page_positions(fx.model), fx.B());
	PKV_CHECK_EQ(sslm_kv_page_size(fx.model), fx.PageBytes());
	constexpr uint32_t kN = 8;
	const size_t required = sslm_kv_page_size(fx.model) * kN + sslm_kv_page_pool_overhead_size(fx.model, kN);
	AlignedBuf mem(required + 2 * SSLM_ABI_ALIGNMENT_BYTES);
	auto create = [&](sslm_model m, void* buf, size_t size, uint32_t n, sslm_status* st) {
		sslm_kv_pool p = reinterpret_cast<sslm_kv_pool>(&mem);  // non-null: a refusal must null it
		*st = sslm_kv_page_pool_create(m, buf, size, n, &p);
		return p;
	};
	sslm_status st = SSLM_OK;
	sslm_kv_pool p = create(fx.model, mem.p, mem.n, 0, &st);  // kills: a zero-page pool admitted
	PKV_CHECK_EQ(st, SSLM_INVALID_ARGUMENT);
	PKV_CHECK(p == nullptr);
	p = create(fx.model, mem.p, required - 1, kN, &st);  // kills: the size check off by one (>= vs >)
	PKV_CHECK_EQ(st, SSLM_BUFFER_TOO_SMALL);
	PKV_CHECK(p == nullptr);
	p = create(fx.model, mem.p + 1, required, kN, &st);  // kills: no alignment check
	PKV_CHECK_EQ(st, SSLM_MISALIGNED_BUFFER);
	PKV_CHECK(p == nullptr);
	p = create(fx.model, mem.p + 1, required - 1, kN, &st);  // kills: alignment checked before size
	PKV_CHECK_EQ(st, SSLM_BUFFER_TOO_SMALL);
	PKV_CHECK(p == nullptr);
	p = create(nullptr, mem.p, required, kN, &st);
	PKV_CHECK_EQ(st, SSLM_INVALID_ARGUMENT);
	p = create(fx.model, nullptr, required, kN, &st);
	PKV_CHECK_EQ(st, SSLM_INVALID_ARGUMENT);
	PKV_CHECK_EQ(sslm_kv_page_pool_create(fx.model, mem.p, required, kN, nullptr), SSLM_INVALID_ARGUMENT);
	// The overflow. page_count is u32, so page_count * page_bytes overflows size_t only when page_bytes
	// exceeds SIZE_MAX / UINT32_MAX. No cloud fixture's page does on a 64-bit size_t (pkv_def's is 6,144
	// bytes, pkv_odd's 1,574,400), so there the cell pins the saturating sizing instead: the overhead
	// at UINT32_MAX pages is finite, the exact sum the test computes is the requirement, and one byte
	// short of it is refused before the verb touches the buffer. A 32-bit build takes the first branch.
	const size_t pb = sslm_kv_page_size(fx.model);
	if (pb > SIZE_MAX / UINT32_MAX) {
		p = create(fx.model, mem.p, SIZE_MAX, UINT32_MAX, &st);  // kills: the product trusted unchecked
		PKV_CHECK_EQ(st, SSLM_INVALID_ARGUMENT);
	} else {
		const size_t ov = sslm_kv_page_pool_overhead_size(fx.model, UINT32_MAX);
		PKV_CHECK(ov != SIZE_MAX && ov <= SIZE_MAX - pb * size_t{UINT32_MAX});
		p = create(fx.model, mem.p, pb * size_t{UINT32_MAX} + ov - 1, UINT32_MAX, &st);
		PKV_CHECK_EQ(st, SSLM_BUFFER_TOO_SMALL);
		PKV_CHECK(p == nullptr);
	}
	// A pool of another model: refused by the budget verbs, nothing drawn.
	auto build = []() -> std::unique_ptr<ProbeState> {
		auto sc = std::make_unique<Scene>();
		const Fixture& a = GetFixture("pkv_def");
		const Fixture& b = GetFixture("pkv_qk");
		PagePool& pool = sc->AddPool(a.model, kN);
		sslm_seq s = nullptr;
		sslm_prefix px = nullptr;
		// kills: a budget verb that skips the pool's model check and draws from a foreign pool.
		PKV_CHECK_EQ(sslm_seq_create_budgeted(b.model, &pool.pool, 16, &s), SSLM_INVALID_ARGUMENT);
		PKV_CHECK_EQ(sslm_prefix_begin_budgeted(b.model, &pool.pool, 16, &px), SSLM_INVALID_ARGUMENT);
		if (s) sc->h.seqs.push_back(s);
		if (px) sc->h.prefixes.push_back(px);
		return sc;
	};
	ProbeExactlyFree(fx, build, kN);
}

// ---- 2.3 [C5 budget-mode cases] ------------------------------------------------------------------
// Hostile SSB6, budget mode, through sslm_seq_restore: budget 0, negative or above cap; origin
// negative or above context_length; context_length above min(origin + budget, cap); kv_mode not in
// {0, 1}; kv_positions inconsistent; truncated K/V. Each -> SSLM_INVALID_ARGUMENT, a null out-handle,
// nothing drawn. The source blob is a budget-64 holder's save after adopting a 300-token prefix and
// decoding 8 (origin 300, context_length 307); the unedited blob restores (the must-accept).
std::vector<uint8_t> Blob23(const Fixture& fx) {
	Scene sc;
	PagePool& pool = sc.AddPool(fx.model, 64);
	sslm_prefix p = FrozenBudgetPrefix(fx, &pool.pool, Stream(3, 300, fx.vocab), sc.h);
	sslm_seq s = BudgetSeq(fx, &pool.pool, 64, sc.h);
	if (p && s && sslm_seq_adopt_prefix(s, p) == SSLM_OK) DecodeN(fx.model, s, 8, nullptr);
	std::vector<uint8_t> b = Save(s);
	const BlobView v = ParseBlobWithHidden(b, HiddenSize(fx));
	PKV_CHECK_MSG(IsMagic(b, "SSB6") && v.ok && v.kv_mode == 1 && v.budget == 64 && v.origin == 300 &&
	                  v.context_length == 307 && v.kv_positions == 307,
	              "2.3: the source blob is a budget holder's SSB6 at origin 300, L 307");
	return b;
}

constexpr uint32_t k23Pool = 64;

std::unique_ptr<ProbeState> Build23(const std::vector<uint8_t>& blob, sslm_status* st, bool* out_null) {
	auto sc = std::make_unique<Scene>();
	const Fixture& fx = GetFixture("pkv_def");
	PagePool& pool = sc->AddPool(fx.model, k23Pool);
	sslm_seq r = reinterpret_cast<sslm_seq>(&pool);
	*st = sslm_seq_restore(fx.model, &pool.pool, blob.data(), blob.size(), &r);
	*out_null = r == nullptr;
	if (*st == SSLM_OK && r) sc->h.seqs.push_back(r);
	return sc;
}

void Cell23Budget() {
	const Fixture& fx = GetFixture("pkv_def");
	if (!fx.ok) return;
	const std::vector<uint8_t> good = Blob23(fx);
	sslm_status st = SSLM_INVALID_ARGUMENT;
	bool out_null = true;
	Build23(good, &st, &out_null);
	PKV_CHECK_EQ(st, SSLM_OK);  // must-accept: the edits below are what the refusals see
	for (const HostileCase& h : HostileSsb6Cases(fx, good)) {
		Build23(h.blob, &st, &out_null);
		// kills: each §3.7 validation removed (the restore admitted, or refused with another status).
		PKV_CHECK_MSG(st == SSLM_INVALID_ARGUMENT && out_null, "2.3 %s: restore -> %d (out %s)", h.name,
		              static_cast<int>(st), out_null ? "null" : "set");
		// kills: validation after the draw, with the draw kept on the refusal path.
		const std::vector<uint8_t>& b = h.blob;
		ProbeExactlyFree(
		    fx,
		    [&b] {
			    sslm_status s2;
			    bool n2;
			    return Build23(b, &s2, &n2);
		    },
		    k23Pool);
	}
}

// ---- 2.5 [C5] -----------------------------------------------------------------------------------
// sslm_prefix_begin_from on an unfrozen parent (a budget prefix and a legacy one) or a null parent:
// SSLM_INVALID_ARGUMENT, a null out-handle, nothing drawn (§3.9 A19).
//
// "Or across models" is not constructible: begin_from takes no model, and draws from the parent's
// own pool, whose model is the parent's (the carrier's CLASS-G row; the probe drops this case). It is
// stated here, not tested.
constexpr uint32_t k25Pool = 300;

std::unique_ptr<ProbeState> Build25(std::vector<std::string>* bad) {
	auto sc = std::make_unique<Scene>();
	const Fixture& fx = GetFixture("pkv_def");
	PagePool& pool = sc->AddPool(fx.model, k25Pool);
	if (pool.status != SSLM_OK) return sc;
	sslm_prefix budget_parent = nullptr, legacy_parent = nullptr;
	if (sslm_prefix_begin_budgeted(fx.model, &pool.pool, 40, &budget_parent) == SSLM_OK && budget_parent) {
		sc->h.prefixes.push_back(budget_parent);
		PrefixPrefillAll(fx.model, budget_parent, Stream(51, 37, fx.vocab), 64);  // not frozen
	}
	if (sslm_prefix_begin(fx.model, &pool.pool, &legacy_parent) == SSLM_OK && legacy_parent) {
		sc->h.prefixes.push_back(legacy_parent);
		PrefixPrefillAll(fx.model, legacy_parent, Stream(52, 20, fx.vocab), 64);  // not frozen
	}
	struct {
		const char* name;
		sslm_prefix parent;
	} cases[] = {{"unfrozen budget parent", budget_parent}, {"unfrozen legacy parent", legacy_parent}, {"null parent", nullptr}};
	for (auto& c : cases) {
		sslm_prefix child = reinterpret_cast<sslm_prefix>(&pool);
		const sslm_status st = sslm_prefix_begin_from(c.parent, 16, &child);
		if (st == SSLM_OK && child) sc->h.prefixes.push_back(child);
		// kills: begin_from that skips the frozen check (a child sharing pages its parent still writes).
		if (bad && (st != SSLM_INVALID_ARGUMENT || child != nullptr))
			bad->push_back(std::string(c.name) + " -> " + std::to_string(st));
	}
	if (bad && (!budget_parent || !legacy_parent)) bad->push_back("a parent was not built");
	return sc;
}

void Cell25() {
	const Fixture& fx = GetFixture("pkv_def");
	if (!fx.ok) return;
	std::vector<std::string> bad;
	Build25(&bad);
	PKV_CHECK_MSG(bad.empty(), "2.5: %zu wrong, first: %s", bad.size(), bad.empty() ? "" : bad[0].c_str());
	// The two unfrozen parents hold R(40) = 4 and ceil(cap/B) = 256 pages. kills: a draw before the
	// frozen check, kept on the refusal path.
	ProbeExactlyFree(fx, [] { return Build25(nullptr); }, k25Pool - fx.R(40) - fx.CapPages());
}

// ---- 2.8 [C5] -----------------------------------------------------------------------------------
// A budget sequence in pool A adopts a frozen prefix in pool B of the same model: SSLM_INVALID_ARGUMENT,
// the sequence unchanged (its re-saved blob equals the one saved before), and after the prefix's
// release the fill probe on pool B admits its initial count. The mutant that shares across pools
// returns SSLM_OK, and the status kills it; its foreign pages would also hold pool B's count down. A
// legacy sequence doing the same adopts by copy, with tokens and rows equal to v1.11.0's.
constexpr uint32_t k28PoolB = 40;

std::unique_ptr<ProbeState> Build28(sslm_status* st, std::vector<uint8_t>* before, std::vector<uint8_t>* after) {
	auto sc = std::make_unique<Scene>();
	const Fixture& fx = GetFixture("pkv_def");
	PagePool& a = sc->AddPool(fx.model, 64);
	PagePool& b = sc->AddPool(fx.model, k28PoolB);
	sc->probe = &b.pool;
	*st = SSLM_ALLOCATION_FAILED;
	sslm_seq s = BudgetSeq(fx, &a.pool, 64, sc->h);
	if (s) {
		PrefillAll(fx.model, s, Stream(2, 40, fx.vocab), 16);
		DecodeN(fx.model, s, 4, nullptr);
	}
	sslm_prefix p = FrozenBudgetPrefix(fx, &b.pool, Stream(3, 300, fx.vocab), sc->h);
	if (!s || !p) return sc;
	if (before) *before = Save(s);
	*st = sslm_seq_adopt_prefix(s, p);
	if (after) *after = Save(s);
	ReleasePrefix(sc->h, p);
	return sc;
}

void Cell28() {
	const Fixture& fx = GetFixture("pkv_def");
	if (!fx.ok) return;
	sslm_status st = SSLM_OK;
	std::vector<uint8_t> before, after;
	Build28(&st, &before, &after);
	PKV_CHECK_EQ(st, SSLM_INVALID_ARGUMENT);  // kills: the mutant that shares across pools (SSLM_OK)
	PKV_CHECK_MSG(!before.empty() && before == after, "2.8: the refused sequence's re-saved blob changed");  // kills: a check after the unmap
	ProbeExactlyFree(
	    fx,
	    [] {
		    sslm_status s2;
		    return Build28(&s2, nullptr, nullptr);
	    },
	    k28PoolB);  // kills: foreign pages referenced by pool A's holder
	// The legacy half: copy adopt across pools, as today (§3.5, G33). (i) legacy pools and a legacy
	// prefix, which runs on every build; (ii) a legacy holder adopting the budget prefix of a page pool.
	for (int variant = 0; variant < 2; ++variant) {
		Scene sc;
		LegacyPool& a = sc.AddLegacyPool(fx.model, 1);
		sslm_kv_pool* pool_b = nullptr;
		sslm_prefix p = nullptr;
		if (variant == 0) {
			pool_b = &sc.AddLegacyPool(fx.model, 1).pool;
			if (sslm_prefix_begin(fx.model, pool_b, &p) == SSLM_OK && p) {
				sc.h.prefixes.push_back(p);
				if (PrefixPrefillAll(fx.model, p, Stream(3, 300, fx.vocab), 64) != SSLM_OK || sslm_prefix_freeze(p) != SSLM_OK)
					p = nullptr;
			}
		} else {
			pool_b = &sc.AddPool(fx.model, 40).pool;
			p = FrozenBudgetPrefix(fx, pool_b, Stream(3, 300, fx.vocab), sc.h);
		}
		sslm_seq s = nullptr;
		BRun d(fx, "SSB5");  // a legacy holder writes 1.9.0's SSB5
		if (d.Ok(p ? SSLM_OK : SSLM_INVALID_ARGUMENT, "prefix") && d.Ok(sslm_seq_create(fx.model, &a.pool, &s), "seq_create")) {
			sc.h.seqs.push_back(s);
			// kills: a legacy holder refused across pools (the budget rule applied to every adopter).
			d.Ok(sslm_seq_adopt_prefix(s, p), "cross-pool copy adopt") && d.Mark("adopt300", s) && d.Decode(s, 8) &&
			    d.Mark("adopt300+decode8", s);
		}
		PKV_CHECK_MSG(d.fail.empty(), "2.8 legacy half (%d): %s", variant, d.fail.c_str());
		for (const Record& r : d.records) ExpectRecord("v1.11.0", fx, "lifecycle", r, "2.8 legacy copy adopt");
	}
}

// ---- 2.10 [C5] ----------------------------------------------------------------------------------
// Refusal paths of sslm_seq_restore_shared. The persona chain (world 1,000 -> persona 1,200 by
// begin_from, world released) holds 75 pages; the cohort blob is a budget-512 holder adopted at origin
// 1,200 (R(512) = 33, E = 75).
//  (i) The second draw, with a null handle and with a mismatched handle (only the private path draws
//      E): in a pool of 75 + R(512) + E - 1 = 182 the restore is refused SSLM_KV_POOL_EXHAUSTED and the
//      fill probe admits exactly the 107 pages free before the call. The mutant that skips the
//      rollback of the first draw leaks R(512) = 33 (74). With the matching handle the same restore is
//      admitted, drawing exactly 33 (74 left).
// (ii) Validation, with the matching handle: each 2.3 hostile case, and bad-alloc injection, refused;
//      after the handle and every sharer are released the fill probe admits all 200 pages. The mutant
//      that shares before validating leaves 75 pages referenced (125).
// Each mutant is equivalent at the other's construction (validation precedes every draw; the private
// path shares nothing), so each part kills exactly one.
enum class Handle210 { kNull, kMismatched, kMatching };

std::unique_ptr<ProbeState> Build210i(Handle210 kind, const std::vector<uint8_t>& blob, sslm_status* st) {
	auto sc = std::make_unique<Scene>();
	const Fixture& fx = GetFixture("pkv_def");
	const uint32_t pool_pages = static_cast<uint32_t>(kChainPages + fx.R(512) + MaterializedE(fx, kPersonaLen, 512) - 1);
	PagePool& pool = sc->AddPool(fx.model, pool_pages);
	*st = SSLM_ALLOCATION_FAILED;
	if (pool.status != SSLM_OK) return sc;
	sslm_prefix persona = PersonaChain(fx, &pool.pool, sc->h, kind == Handle210::kMismatched ? 1 : 0);
	if (!persona) return sc;
	sslm_seq r = nullptr;
	uint32_t shared = 0;
	*st = sslm_seq_restore_shared(fx.model, &pool.pool, blob.data(), blob.size(),
	                              kind == Handle210::kNull ? nullptr : persona, &r, &shared);
	if (*st == SSLM_OK && r) sc->h.seqs.push_back(r);
	return sc;
}

constexpr uint32_t k210iiPool = 200;

// `armed` injects a bad_alloc into the restore (the existing seam, SUPERSLM_ENABLE_BAD_ALLOC_INJECTION,
// as tests/test_main.cpp arms it); *consulted reports whether the verb took the armed fault.
std::unique_ptr<ProbeState> Build210ii(const std::vector<uint8_t>& blob, bool armed, sslm_status* st, bool* consulted) {
	auto sc = std::make_unique<Scene>();
	const Fixture& fx = GetFixture("pkv_def");
	PagePool& pool = sc->AddPool(fx.model, k210iiPool);
	*st = SSLM_OK;
	if (pool.status != SSLM_OK) return sc;
	sslm_prefix persona = PersonaChain(fx, &pool.pool, sc->h);
	if (!persona) return sc;
	sslm_seq r = nullptr;
	uint32_t shared = 0;
	if (armed) superslm_test::ArmInjectedFault(superslm_test::InjectThrowKind::kBadAlloc);
	*st = sslm_seq_restore_shared(fx.model, &pool.pool, blob.data(), blob.size(), persona, &r, &shared);
	if (consulted) *consulted = superslm_test::g_inject_throw == superslm_test::InjectThrowKind::kNone;
	superslm_test::DisarmInjectedFault();
	if (*st == SSLM_OK && r) sc->h.seqs.push_back(r);
	ReleasePrefix(sc->h, persona);
	sc->h.ReleaseAll();  // every sharer, so the pool should be whole
	return sc;
}

void Cell210() {
	const Fixture& fx = GetFixture("pkv_def");
	if (!fx.ok) return;
	PKV_CHECK_EQ(fx.R(512), 33);
	PKV_CHECK_EQ(MaterializedE(fx, kPersonaLen, 512), 75);
	const std::vector<uint8_t> blob = CohortBlob(fx, 100);
	const BlobView v = ParseBlobWithHidden(blob, HiddenSize(fx));
	PKV_CHECK(v.ok && v.kv_mode == 1 && v.budget == 512 && v.origin == kPersonaLen);
	// (i)
	const int64_t free_before = 182 - kChainPages;  // 107
	for (Handle210 kind : {Handle210::kNull, Handle210::kMismatched, Handle210::kMatching}) {
		const char* name = kind == Handle210::kNull ? "null" : kind == Handle210::kMismatched ? "mismatched" : "matching";
		sslm_status st = SSLM_OK;
		Build210i(kind, blob, &st);
		if (kind == Handle210::kMatching) {
			PKV_CHECK_MSG(st == SSLM_OK, "2.10(i) matching handle: restore -> %d (must be admitted)", static_cast<int>(st));
		} else {
			// kills: the private path that draws E without checking it is there (an over-draw admitted).
			PKV_CHECK_MSG(st == SSLM_KV_POOL_EXHAUSTED, "2.10(i) %s handle: restore -> %d", name, static_cast<int>(st));
		}
		// kills: the rollback of the first draw skipped (74 free, not 107); and, with the matching
		// handle, a restore that draws more than R(512) or shares less than 75 pages.
		const int64_t k = kind == Handle210::kMatching ? free_before - fx.R(512) : free_before;
		ProbeExactlyFree(
		    fx,
		    [kind, &blob] {
			    sslm_status s2;
			    return Build210i(kind, blob, &s2);
		    },
		    k);
	}
	// (ii)
	{
		sslm_status st = SSLM_INVALID_ARGUMENT;
		Build210ii(blob, false, &st, nullptr);
		PKV_CHECK_EQ(st, SSLM_OK);  // must-accept: the unedited blob with the matching handle
	}
	for (const HostileCase& h : HostileSsb6Cases(fx, blob)) {
		sslm_status st = SSLM_OK;
		Build210ii(h.blob, false, &st, nullptr);
		PKV_CHECK_MSG(st == SSLM_INVALID_ARGUMENT, "2.10(ii) %s with the matching handle: -> %d", h.name, static_cast<int>(st));
		// kills: sharing before validating (75 pages stay referenced: 125, not 200).
		const std::vector<uint8_t>& b = h.blob;
		ProbeExactlyFree(
		    fx,
		    [&b] {
			    sslm_status s2;
			    return Build210ii(b, false, &s2, nullptr);
		    },
		    k210iiPool);
	}
	{
		sslm_status st = SSLM_OK;
		bool consulted = false;
		Build210ii(blob, true, &st, &consulted);
		PKV_CHECK_MSG(consulted, "2.10(ii) bad-alloc: sslm_seq_restore_shared never consulted the injection seam");
		PKV_CHECK_EQ(st, SSLM_ALLOCATION_FAILED);
		ProbeExactlyFree(
		    fx,
		    [&blob] {
			    sslm_status s2;
			    return Build210ii(blob, true, &s2, nullptr);
		    },
		    k210iiPool);
	}
}

PKV_CELL("2.1/C5", "C5", Cell21);
PKV_CELL("2.2/C5", "C5", Cell22);
PKV_CELL("2.3/C5", "C5", Cell23Budget);
PKV_CELL("2.5/C5", "C5", Cell25);
PKV_CELL("2.8/C5", "C5", Cell28);
PKV_CELL("2.10/C5", "C5", Cell210);

}  // namespace

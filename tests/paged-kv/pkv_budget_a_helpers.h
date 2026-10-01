// Paged-KV plan (rev 16.1) step C1, group budget_a: helpers the C5-owned cells of dimensions 1 to 5
// share (tests/paged-kv/c5_dim{1..5}_*.cpp). Everything here is inline and in pkv::budget_a, so it
// cannot collide with another group's helpers in the same superslm_pkv_c5 executable.
//
//   - BRun: a record-producing driver for budget holders, the budget twin of pkv_scenarios.h's
//     Driver. Its records carry the same stage names as the R0 reference, and their K/V rows are
//     read out of the holder's SSB6 blob by pkv_common.h's BlobRows, so ExpectMatchesReference
//     compares them with Compare::kTokensAndRows (a budget holder saves SSB6, the reference SSB5).
//   - the budget twins of three R0 scenarios: lifecycle (1.1), prefix_lengths (4.2) and widths
//     (4.1), built with the budget verbs and the same token streams, chunks and stage names.
//   - Scene: a ProbeState that owns its pools and handles, so a cell can hand ProbeExactlyFree a
//     builder that replays its construction.
//   - the persona chain of 2.10, 3.10 and 9.9 (world 1,000 -> begin_from to 1,200, world released:
//     75 pages), SSB6 field editing, table and page reads through the C4 seams, and a start gate
//     and repetition count for the racing cells.
//
// Nothing here reads sslm_kv_pool_stats or sslm_seq_kv_stats: page counts are graded only by the
// fill probe (§7, Grading).

#ifndef SUPERSLM_TESTS_PKV_BUDGET_A_HELPERS_H
#define SUPERSLM_TESTS_PKV_BUDGET_A_HELPERS_H

#include "pkv_common.h"

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <memory>
#include <set>
#include <string>
#include <thread>
#include <vector>

namespace pkv {
namespace budget_a {

// The value an out-handle is preset to before a verb whose refusal a cell checks by a null out:
// non-null, so a verb that never writes *out leaves it standing and the check fails. Never released
// or dereferenced -- release a handle only when it is neither null nor this.
template <class H>
inline H OutSentinel() {
	return reinterpret_cast<H>(uintptr_t{0x1});
}

// ---- small arithmetic, written in the test (§3.1, §3.4, §3.7) ---------------------------------

inline int64_t CeilDiv(int64_t a, int64_t b) { return (a + b - 1) / b; }

// Pages a frozen prefix of length `len` keeps (§3.4, freeze row): ceil(len / B).
inline int64_t PrefixPages(const Fixture& fx, int64_t len) { return CeilDiv(len, fx.B()); }

// §3.7's E for a private restore at `origin` with `budget`.
inline int64_t MaterializedE(const Fixture& fx, int64_t origin, int64_t budget) {
	const int64_t a = origin / fx.B(), b = fx.CapPages() - fx.R(budget);
	return a < b ? a : b;
}

// ---- token and decode helpers ------------------------------------------------------------------

// Decodes up to `n` tokens (NextToken, one layer per call). Returns the first non-OK status, and
// appends every emitted token to `out`.
inline sslm_status DecodeN(sslm_model model, sslm_seq s, int n, std::vector<int32_t>* out) {
	for (int i = 0; i < n; ++i) {
		sslm_status st = SSLM_OK;
		const int32_t t = NextToken(model, s, &st);
		if (st != SSLM_OK) return st;
		if (t < 0) return SSLM_INVALID_ARGUMENT;
		if (out) out->push_back(t);
	}
	return SSLM_OK;
}

// Decodes until the first refusal and returns how many tokens were emitted before it; the refusal
// is returned in *refusal. `guard` bounds the loop (a build that never refuses fails the caller's
// count check, never hangs).
inline int64_t TokensUntilRefusal(sslm_model model, sslm_seq s, sslm_status* refusal, int64_t guard = 8192,
                                  std::vector<int32_t>* out = nullptr) {
	int64_t n = 0;
	sslm_status st = SSLM_OK;
	for (; n < guard; ++n) {
		const int32_t t = NextToken(model, s, &st);
		if (st != SSLM_OK || t < 0) break;
		if (out) out->push_back(t);
	}
	if (refusal) *refusal = st;
	return n;
}

inline std::vector<uint8_t> Save(sslm_seq s) {
	std::vector<uint8_t> b;
	if (!s || !SaveBlob(s, &b)) b.clear();
	return b;
}

// Rows [0, L) of a blob (either magic), digested as the reference digests them.
inline std::string RowsSha(const Fixture& fx, const std::vector<uint8_t>& blob, int64_t L) {
	std::vector<uint8_t> rows;
	if (!BlobRows(blob, fx, HiddenSize(fx), L, &rows)) return "(unreadable)";
	return Sha(rows.data(), rows.size());
}

// ---- BRun: the budget scenario driver ---------------------------------------------------------

// Records in pkv_scenarios.h's Record shape. `expect_magic` is checked on every save, so a budget
// holder that wrote 1.9.0's SSB5 (or a legacy one that wrote SSB6) fails where it happens.
struct BRun {
	const Fixture& fx;
	const char* expect_magic;
	std::string fail;
	std::vector<Record> records;
	std::vector<int32_t> pending;

	BRun(const Fixture& f, const char* magic = "SSB6") : fx(f), expect_magic(magic) {}

	bool Ok(sslm_status s, const char* what) {
		if (s == SSLM_OK) return true;
		if (fail.empty()) fail = std::string(what) + " -> " + std::to_string(static_cast<int>(s));
		return false;
	}
	bool Prefill(sslm_seq s, const std::vector<int32_t>& t, int32_t chunk) {
		int64_t got = 0;
		return Ok(PrefillAll(fx.model, s, t, chunk, &got), "prefill") &&
		       Ok(got == static_cast<int64_t>(t.size()) ? SSLM_OK : SSLM_INVALID_ARGUMENT, "prefill short");
	}
	bool PrefixPrefill(sslm_prefix p, const std::vector<int32_t>& t, int32_t chunk) {
		return Ok(PrefixPrefillAll(fx.model, p, t, chunk), "prefix_prefill");
	}
	bool Decode(sslm_seq s, int32_t n) { return Ok(DecodeN(fx.model, s, n, &pending), "decode"); }
	// One record; with `s`, its blob's context length and the rows [0, L) of its K/V section.
	bool Mark(const char* stage, sslm_seq s, std::vector<uint8_t>* keep = nullptr) {
		if (!fail.empty()) return false;
		Record r;
		r.stage = stage;
		r.tokens.swap(pending);
		if (s) {
			std::vector<uint8_t> blob;
			if (!SaveBlob(s, &blob)) return Ok(SSLM_INVALID_ARGUMENT, "save");
			if (!IsMagic(blob, expect_magic)) return Ok(SSLM_INVALID_ARGUMENT, "save magic");
			r.context_length = BlobContextLength(blob);
			r.saturation = BlobSaturationCount(blob);
			r.blob_sha = Sha(blob.data(), blob.size());
			r.rows_sha = RowsSha(fx, blob, r.context_length);
			if (keep) keep->swap(blob);
		} else {
			r.blob_sha = "-";
			r.rows_sha = "-";
		}
		records.push_back(std::move(r));
		return true;
	}
};

// 1.1's budget half: pkv_scenarios.h's Lifecycle with a budget prefix (sslm_prefix_begin_budgeted)
// and a budget holder (sslm_seq_create_budgeted) in `pool`. The fresh stages write [0, 111) and the
// adopted ones [300, 347), so `seq_budget` >= 111 and `prefix_budget` >= 300.
inline void BudgetLifecycle(BRun& d, sslm_kv_pool* pool, int32_t seq_budget, int32_t prefix_budget) {
	const int32_t V = d.fx.vocab;
	sslm_prefix px = nullptr;
	sslm_seq s = nullptr;
	if (!d.Ok(sslm_prefix_begin_budgeted(d.fx.model, pool, prefix_budget, &px), "prefix_begin_budgeted")) return;
	if (d.PrefixPrefill(px, Stream(3, 300, V), 64) && d.Ok(sslm_prefix_freeze(px), "prefix_freeze") &&
	    d.Ok(sslm_seq_create_budgeted(d.fx.model, pool, seq_budget, &s), "seq_create_budgeted")) {
		(void)(d.Prefill(s, Stream(1, 100, V), 64) && d.Mark("prefill100", s) && d.Decode(s, 4) &&
		       d.Mark("prefill100+decode4", s) && d.Decode(s, 8) && d.Mark("continue8", nullptr) &&
		       d.Ok(sslm_seq_reset(s), "reset") && d.Mark("reset1", s) && d.Prefill(s, Stream(2, 40, V), 16) &&
		       d.Decode(s, 4) && d.Mark("prefill40+decode4", s) && d.Ok(sslm_seq_reset(s), "reset") &&
		       d.Ok(sslm_seq_adopt_prefix(s, px), "adopt") && d.Mark("adopt300", s) && d.Decode(s, 8) &&
		       d.Mark("adopt300+decode8", s) && d.Prefill(s, Stream(4, 37, V), 7) && d.Decode(s, 4) &&
		       d.Mark("adopt+prefill37+decode4", s));
	}
	if (s) sslm_seq_release(s);
	sslm_prefix_release(px);
}

// 4.2's budget half: pkv_scenarios.h's PrefixLengths with a budget prefix of budget max(len, 1)
// and a budget adopter of `seq_budget` (>= 30: the adopter writes [len, len + 30)).
inline void BudgetPrefixLengths(BRun& d, sslm_kv_pool* pool, int32_t seq_budget) {
	const int32_t V = d.fx.vocab;
	for (int32_t len : {0, 1, 15, 16, 17, 1000, 1008}) {
		sslm_prefix px = nullptr;
		sslm_seq s = nullptr;
		if (!d.Ok(sslm_prefix_begin_budgeted(d.fx.model, pool, len > 0 ? len : 1, &px), "prefix_begin_budgeted"))
			return;
		const std::string tag = "p" + std::to_string(len);
		const bool ok = (len == 0 || d.PrefixPrefill(px, Stream(51, len, V), 64)) &&
		                d.Ok(sslm_prefix_freeze(px), "prefix_freeze") &&
		                d.Ok(sslm_seq_create_budgeted(d.fx.model, pool, seq_budget, &s), "seq_create_budgeted") &&
		                d.Ok(sslm_seq_adopt_prefix(s, px), "adopt") && d.Mark((tag + "_adopted").c_str(), s) &&
		                (len == 0 || (d.Decode(s, 8) && d.Mark((tag + "_decode8").c_str(), s))) &&
		                d.Prefill(s, Stream(52, 20, V), 64) && d.Decode(s, 4) &&
		                d.Mark((tag + "_prefill20+decode4").c_str(), s);
		if (s) sslm_seq_release(s);
		sslm_prefix_release(px);
		if (!ok) return;
	}
}

// 4.1's budget half: pkv_scenarios.h's Widths, one budget holder per width W with the budget its
// writes need and no more (min(W + 1, cap): prefill W rows, then the decode that writes row W).
inline void BudgetWidths(BRun& d, sslm_kv_pool* pool, const std::vector<int64_t>& widths) {
	const int64_t cap = d.fx.geo.context_cap;
	for (int64_t w : widths) {
		sslm_seq s = nullptr;
		const int32_t budget = static_cast<int32_t>(w < cap ? w + 1 : cap);
		if (!d.Ok(sslm_seq_create_budgeted(d.fx.model, pool, budget, &s), "seq_create_budgeted")) return;
		const std::string tag = "w" + std::to_string(w);
		bool ok = d.Prefill(s, Stream(41, static_cast<int32_t>(w), d.fx.vocab), 64) &&
		          d.Mark((tag + "_prefill").c_str(), s) && d.Decode(s, 1) && d.Mark((tag + "_ready").c_str(), nullptr);
		if (ok && w < cap) ok = d.Decode(s, 1) && d.Mark((tag + "_decode1").c_str(), s);
		sslm_seq_release(s);
		if (!ok) return;
	}
}

// The widths the R0 reference recorded for a fixture (pkv_scenarios.h's Widths).
inline std::vector<int64_t> ReferenceWidths(const Fixture& fx) {
	std::vector<int64_t> w = {1, 15, 16, 17, 32, 1064};
	if (fx.geo.context_cap <= 4100) {
		w.push_back(fx.geo.context_cap - 1);
		w.push_back(fx.geo.context_cap);
	}
	return w;
}

// Compares one record against the reference record of the same stage (tokens, context length,
// rows), for cells that produce only some of a scenario's stages.
inline void ExpectRecord(const char* tag, const Fixture& fx, const char* scenario, const Record& got,
                         const char* what) {
	const RefRecord* e = RefLookup(Reference(tag, fx), scenario, got.stage);
	if (!e) return;
	PKV_CHECK_MSG(got.tokens == e->tokens, "%s: %s/%s tokens differ from %s", what, scenario, got.stage.c_str(), tag);
	PKV_CHECK_MSG(got.context_length == e->context_length, "%s: %s/%s L %lld vs %lld", what, scenario,
	              got.stage.c_str(), static_cast<long long>(got.context_length),
	              static_cast<long long>(e->context_length));
	PKV_CHECK_MSG(e->rows_sha == "-" || got.rows_sha == e->rows_sha, "%s: %s/%s K/V rows differ from %s", what,
	              scenario, got.stage.c_str(), tag);
}

// ---- scenes for the fill probe ----------------------------------------------------------------

// A rebuildable state: page pools and the handles in them. Handles are declared after the pools,
// so they are released first. `probe` names the pool the fill probe reads (default: pools[0]).
struct Scene : ProbeState {
	std::vector<std::unique_ptr<PagePool>> pools;
	std::vector<std::unique_ptr<LegacyPool>> legacy_pools;
	Handles h;
	sslm_kv_pool* probe = nullptr;
	sslm_kv_pool* Pool() override { return probe ? probe : &pools.front()->pool; }
	PagePool& AddPool(sslm_model model, uint32_t pages, uint8_t fill = 0) {
		pools.push_back(std::make_unique<PagePool>(model, pages, fill));
		return *pools.back();
	}
	LegacyPool& AddLegacyPool(sslm_model model, uint32_t blocks) {
		legacy_pools.push_back(std::make_unique<LegacyPool>(model, blocks));
		return *legacy_pools.back();
	}
	~Scene() override { h.ReleaseAll(); }
};

using SceneBuilder = std::function<std::unique_ptr<ProbeState>()>;

// Releases one handle owned by a Handles list (and forgets it), so a scene can release in its own
// order and still be torn down in full.
inline sslm_status ReleaseSeq(Handles& h, sslm_seq s) {
	auto it = std::find(h.seqs.begin(), h.seqs.end(), s);
	if (it != h.seqs.end()) h.seqs.erase(it);
	return sslm_seq_release(s);
}
inline sslm_status ReleasePrefix(Handles& h, sslm_prefix p) {
	auto it = std::find(h.prefixes.begin(), h.prefixes.end(), p);
	if (it != h.prefixes.end()) h.prefixes.erase(it);
	return sslm_prefix_release(p);
}

// A frozen budget prefix of `tokens` (budget max(len, 1) unless given), owned by `h`. Null on any
// failure (the caller's checks then fail on what follows).
inline sslm_prefix FrozenBudgetPrefix(const Fixture& fx, sslm_kv_pool* pool, const std::vector<int32_t>& tokens,
                                      Handles& h, int32_t budget = 0) {
	sslm_prefix p = nullptr;
	const int32_t b = budget > 0 ? budget : (tokens.empty() ? 1 : static_cast<int32_t>(tokens.size()));
	if (sslm_prefix_begin_budgeted(fx.model, pool, b, &p) != SSLM_OK || !p) return nullptr;
	h.prefixes.push_back(p);
	if ((!tokens.empty() && PrefixPrefillAll(fx.model, p, tokens, 64) != SSLM_OK) || sslm_prefix_freeze(p) != SSLM_OK)
		return nullptr;
	return p;
}

// A budget holder owned by `h`, or null.
inline sslm_seq BudgetSeq(const Fixture& fx, sslm_kv_pool* pool, int32_t budget, Handles& h) {
	sslm_seq s = nullptr;
	if (sslm_seq_create_budgeted(fx.model, pool, budget, &s) != SSLM_OK || !s) return nullptr;
	h.seqs.push_back(s);
	return s;
}

// ---- the persona chain (2.10, 3.10; the cohort geometry of §3.7 and 9.9) ----------------------
//
// A world of 1,000 tokens (budget 1,000: R = 64 pages, 63 kept at freeze), a persona made by
// sslm_prefix_begin_from(world, 200) that shares the world's 62 full pages, copies its tail and
// prefills 200 more to 1,200 (13 private pages kept at freeze), then the world released, which
// frees its private tail page. The chain then holds 62 + 13 = 75 pages (§3.7, 9.9).
constexpr int32_t kWorldLen = 1000;
constexpr int32_t kPersonaLen = 1200;
constexpr int64_t kChainPages = 75;

inline std::vector<int32_t> WorldTokens(const Fixture& fx) { return Stream(61, kWorldLen, fx.vocab); }
// `variant` != 0 changes the persona's first token (position 1,000), so the chain's rows from
// page 62 on differ: a handle that fails only §3.7's condition (f).
inline std::vector<int32_t> PersonaTokens(const Fixture& fx, int variant = 0) {
	std::vector<int32_t> t = Stream(62, kPersonaLen - kWorldLen, fx.vocab);
	if (variant) t[0] = (t[0] + variant) % fx.vocab;
	return t;
}

// Builds the chain in `pool`; returns the persona (owned by `h`), or null.
inline sslm_prefix PersonaChain(const Fixture& fx, sslm_kv_pool* pool, Handles& h, int variant = 0) {
	sslm_prefix world = FrozenBudgetPrefix(fx, pool, WorldTokens(fx), h);
	if (!world) return nullptr;
	sslm_prefix persona = nullptr;
	if (sslm_prefix_begin_from(world, kPersonaLen - kWorldLen, &persona) != SSLM_OK || !persona) return nullptr;
	h.prefixes.push_back(persona);
	if (PrefixPrefillAll(fx.model, persona, PersonaTokens(fx, variant), 64) != SSLM_OK ||
	    sslm_prefix_freeze(persona) != SSLM_OK)
		return nullptr;
	if (ReleasePrefix(h, world) != SSLM_OK) return nullptr;
	return persona;
}

// A cohort blob (§3.7's construction): a budget-512 holder adopts the persona (origin 1,200) and
// decodes `decode` tokens; its SSB6 save. Built in a pool of its own; `next` receives the unsaved
// original's next `next_n` tokens after the save.
inline std::vector<uint8_t> CohortBlob(const Fixture& fx, int decode, std::vector<int32_t>* next = nullptr,
                                       int next_n = 0) {
	Scene sc;
	PagePool& pool = sc.AddPool(fx.model, 200);
	sslm_prefix persona = PersonaChain(fx, &pool.pool, sc.h);
	sslm_seq s = persona ? BudgetSeq(fx, &pool.pool, 512, sc.h) : nullptr;
	std::vector<uint8_t> blob;
	if (s && sslm_seq_adopt_prefix(s, persona) == SSLM_OK && DecodeN(fx.model, s, decode, nullptr) == SSLM_OK) {
		blob = Save(s);
		if (next) DecodeN(fx.model, s, next_n, next);
	}
	PKV_CHECK_MSG(IsMagic(blob, "SSB6"), "cohort blob: a budget holder's save is SSB6");
	return blob;
}

// ---- SSB6 field editing (§3.7 layout; pkv_common.h's offsets) --------------------------------

inline size_t Ssb6KvPositionsOffset(const Fixture& fx, const std::vector<uint8_t>& b) {
	return kSsb6Header + HiddenSize(fx) + 4 * static_cast<size_t>(Le64(b, 112));
}

// One hostile edit of a valid budget-mode SSB6 blob (2.3's budget-mode cases, reused by 2.10).
struct HostileCase {
	const char* name;
	std::vector<uint8_t> blob;
};

inline std::vector<HostileCase> HostileSsb6Cases(const Fixture& fx, const std::vector<uint8_t>& good) {
	std::vector<HostileCase> out;
	const BlobView v = ParseBlobWithHidden(good, HiddenSize(fx));
	if (!v.ok || !IsMagic(good, "SSB6")) {
		PKV_CHECK_MSG(false, "hostile cases need a valid SSB6 blob");
		return out;
	}
	const int64_t cap = fx.geo.context_cap;
	const size_t kvp = Ssb6KvPositionsOffset(fx, good);
	const size_t row = fx.BytesPerToken();
	auto edit32 = [&](const char* name, size_t at, uint32_t value) {
		std::vector<uint8_t> b = good;
		PutLe32(b, at, value);
		out.push_back({name, std::move(b)});
	};
	auto edit64 = [&](const char* name, size_t at, uint64_t value) {
		std::vector<uint8_t> b = good;
		PutLe64(b, at, value);
		out.push_back({name, std::move(b)});
	};
	edit32("budget 0", 160, 0u);
	edit32("budget -1", 160, static_cast<uint32_t>(-1));
	edit32("budget INT32_MIN", 160, 0x80000000u);
	edit32("budget cap + 1", 160, static_cast<uint32_t>(cap + 1));
	edit64("origin -1", 164, static_cast<uint64_t>(int64_t{-1}));
	edit64("origin context_length + 1", 164, static_cast<uint64_t>(v.context_length + 1));
	// context_length above min(origin + budget, cap): the budget shrunk so origin + budget lands one
	// below the saved context length (every other field still valid).
	edit32("context_length above origin + budget", 160, static_cast<uint32_t>(v.context_length - v.origin - 1));
	edit32("kv_mode 2", 156, 2u);
	edit32("kv_mode 0xFFFFFFFF", 156, 0xFFFFFFFFu);
	{  // kv_positions one past L', with one row appended so the size still matches it
		std::vector<uint8_t> b = good;
		PutLe64(b, kvp, v.kv_positions + 1);
		b.resize(b.size() + row, 0);
		out.push_back({"kv_positions L' + 1 (size consistent)", std::move(b)});
	}
	if (v.kv_positions > 0) {  // kv_positions one below L', with one row removed
		std::vector<uint8_t> b = good;
		PutLe64(b, kvp, v.kv_positions - 1);
		b.resize(b.size() - row);
		out.push_back({"kv_positions L' - 1 (size consistent)", std::move(b)});
	}
	{
		std::vector<uint8_t> b = good;
		b.pop_back();
		out.push_back({"K/V truncated by one byte", std::move(b)});
	}
	{
		std::vector<uint8_t> b = good;
		b.resize(b.size() - row);
		out.push_back({"K/V truncated by one row", std::move(b)});
	}
	return out;
}

// ---- tables and pages through the C4 seams ----------------------------------------------------

inline std::vector<uint32_t> SeqTable(sslm_seq s) {
	std::vector<uint32_t> t;
	uint32_t page = 0, mapped = 0;
	if (!s || sslm_pkv_test_only_seq_table_entry(s, 0, &page, &mapped) != SSLM_OK) return t;
	for (uint32_t i = 0; i < mapped; ++i)
		if (sslm_pkv_test_only_seq_table_entry(s, i, &page, nullptr) == SSLM_OK) t.push_back(page);
	return t;
}
inline std::vector<uint32_t> PrefixTable(sslm_prefix p) {
	std::vector<uint32_t> t;
	uint32_t page = 0, mapped = 0;
	if (!p || sslm_pkv_test_only_prefix_table_entry(p, 0, &page, &mapped) != SSLM_OK) return t;
	for (uint32_t i = 0; i < mapped; ++i)
		if (sslm_pkv_test_only_prefix_table_entry(p, i, &page, nullptr) == SSLM_OK) t.push_back(page);
	return t;
}

inline std::vector<uint8_t> PeekPage(const Fixture& fx, sslm_kv_pool pool, uint32_t page) {
	std::vector<uint8_t> b(fx.PageBytes(), 0);
	if (sslm_pkv_test_only_peek_page_bytes(pool, page, b.data(), b.size()) != SSLM_OK) b.clear();
	return b;
}

inline bool AllBytes(const std::vector<uint8_t>& b, uint8_t v) {
	return !b.empty() && std::all_of(b.begin(), b.end(), [v](uint8_t x) { return x == v; });
}

// Every listed page reads 0xCD through the page-index peek (§3.3: a freed dirty page is poisoned).
inline void ExpectPoisoned(const Fixture& fx, sslm_kv_pool pool, const std::set<uint32_t>& pages, const char* what) {
	int bad = 0;
	for (uint32_t p : pages)
		if (!AllBytes(PeekPage(fx, pool, p), 0xCD)) ++bad;
	PKV_CHECK_MSG(!pages.empty() && bad == 0, "%s: %d of %zu freed pages do not read 0xCD", what, bad, pages.size());
}

// ---- racing cells ------------------------------------------------------------------------------

// §8: each racing cell runs its race 100 times (planner default). PKV_RACE_REPS overrides it for a
// quick local run; the TSan leg runs the default.
inline int RaceReps() {
	const char* e = std::getenv("PKV_RACE_REPS");
	const int n = e ? std::atoi(e) : 0;
	return n > 0 ? n : 100;
}

// A start gate: every thread arrives, then all leave together, so the calls under test overlap.
struct StartGate {
	std::atomic<int> arrived{0};
	int n;
	explicit StartGate(int count) : n(count) {}
	void Wait() {
		arrived.fetch_add(1, std::memory_order_acq_rel);
		while (arrived.load(std::memory_order_acquire) < n) std::this_thread::yield();
	}
};

// Runs fn(i) on `n` threads released together by one gate, and joins them.
inline void RunTogether(int n, const std::function<void(int)>& fn) {
	StartGate gate(n);
	std::vector<std::thread> t;
	t.reserve(static_cast<size_t>(n));
	for (int i = 0; i < n; ++i)
		t.emplace_back([&, i] {
			gate.Wait();
			fn(i);
		});
	for (auto& th : t) th.join();
}

// A deterministic generator for shuffles and operation mixes.
struct Lcg {
	uint32_t x;
	explicit Lcg(uint32_t seed) : x(seed * 2654435761u + 1u) {}
	uint32_t Next() {
		x = x * 1664525u + 1013904223u;
		return x >> 8;
	}
	template <class T>
	void Shuffle(std::vector<T>& v) {
		for (size_t i = v.size(); i > 1; --i) std::swap(v[i - 1], v[Next() % i]);
	}
};

}  // namespace budget_a
}  // namespace pkv

#endif  // SUPERSLM_TESTS_PKV_BUDGET_A_HELPERS_H

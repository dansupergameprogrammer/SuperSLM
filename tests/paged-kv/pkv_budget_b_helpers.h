// Paged-KV plan (rev 16.1) step C1, group budget_b: the helpers the budget-holder cells of
// dimensions 6, 7 and 8 share (c5_dim6_determinism.cpp, c5_dim7_contract.cpp,
// c5_dim8_composition.cpp).
//
// Everything here is written from the plan's formulas and the R0 scenario scripts, never by
// calling the code under test for an expected value:
//   - budget twins of pkv_scenarios.h's lifecycle, widths and prefix_lengths scenarios, which emit
//     records in the reference's own shape, so a budget holder's tokens and K/V rows compare with
//     v1.11.0 through ExpectMatchesReference(..., Compare::kTokensAndRows) (its blob is SSB6, so
//     the whole-blob digest differs from the reference's SSB5 by design, §3.7);
//   - the persona chain of §7's cohort cells (a 1,000-token world, a 200-token persona through
//     begin_from, budget-512 sequences at origin 1,200);
//   - the used-state adopt census of tests/t2899-schema-deadend-red-suite/cell_adopt_prefix_census.cpp
//     in budget mode (cells 1.10, 8.11 and 7.13's budget part);
//   - a synthetic rank-1 adapter for pkv_def (8.4, 8.12), built from the fixture's own Config and
//     SigmoidLut sections and an ADP1 provenance written field by field (design §24.2's layout, as
//     tests/test_main.cpp's BuildAdp1 writes it);
//   - the table-sentinel child (7.13): a construction run in a forked child with the sentinel on,
//     graded by the child's exit status, so a trap (std::abort) fails one cell, not the runner.
//
// Helpers that call a C5 verb are inline and emitted only where a cell calls them (pkv_common.h's
// rule), so this header adds no symbol to an executable whose cells do not use it.

#ifndef SUPERSLM_TESTS_PKV_BUDGET_B_HELPERS_H
#define SUPERSLM_TESTS_PKV_BUDGET_B_HELPERS_H

#include "pkv_common.h"
#include "sslm_model_hostile_fixtures.h"
#include "superslm/artifact.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#if defined(__unix__) || defined(__APPLE__)
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#define PKV_BUDGET_B_HAVE_FORK 1
#else
#define PKV_BUDGET_B_HAVE_FORK 0
#endif

namespace pkv {
namespace budget_b {

// ---- blobs and records --------------------------------------------------------------------------

// L' of §3.7: context_length + (layer_index > 0 ? 1 : 0), read from the blob's fixed header (the
// same offsets in 1.9.0's SSB5 and SSB6).
inline int64_t BlobLp(const std::vector<uint8_t>& b) {
	const BlobView v = ParseBlobWithHidden(b, 192);
	return v.context_length + (v.layer_index > 0 ? 1 : 0);
}

// The rows of positions [0, L) from either format, or an empty vector (and a failed check).
inline std::vector<uint8_t> Rows(const std::vector<uint8_t>& blob, const Fixture& fx, int64_t L) {
	std::vector<uint8_t> rows;
	const bool ok = BlobRows(blob, fx, HiddenSize(fx), L, &rows);
	PKV_CHECK_MSG(ok, "blob rows [0, %lld) unreadable (magic %.4s, %zu bytes)", static_cast<long long>(L),
	              blob.size() >= 4 ? reinterpret_cast<const char*>(blob.data()) : "????", blob.size());
	return ok ? rows : std::vector<uint8_t>{};
}

inline std::vector<uint8_t> Save(sslm_seq s) {
	std::vector<uint8_t> b;
	PKV_CHECK_MSG(SaveBlob(s, &b), "sslm_seq_save failed");
	return b;
}

// One record in pkv_scenarios.h's shape, from a save of `s` (any format): the context length and
// saturation count from the fixed header, the whole blob's digest, and the digest of the rows
// [0, context_length) read by the layout formula written in the test (BlobRows).
inline Record MakeRecord(const Fixture& fx, const char* stage, sslm_seq s, std::vector<int32_t>* pending,
                         std::vector<uint8_t>* keep = nullptr) {
	Record r;
	r.stage = stage;
	r.tokens.swap(*pending);
	if (!s) {
		r.blob_sha = "-";
		r.rows_sha = "-";
		return r;
	}
	std::vector<uint8_t> blob = Save(s);
	r.context_length = BlobContextLength(blob);
	r.saturation = BlobSaturationCount(blob);
	r.blob_sha = Sha(blob.data(), blob.size());
	const std::vector<uint8_t> rows = Rows(blob, fx, r.context_length);
	r.rows_sha = Sha(rows.data(), rows.size());
	if (keep) keep->swap(blob);
	return r;
}

// `n` greedy tokens, appended to `out`; false at the first token the step does not produce (its
// status in *st when given).
inline bool DecodeN(const Fixture& fx, sslm_seq s, int n, std::vector<int32_t>* out, sslm_status* st = nullptr) {
	for (int i = 0; i < n; ++i) {
		sslm_status s1 = SSLM_OK;
		const int32_t t = NextToken(fx.model, s, &s1);
		if (st) *st = s1;
		if (t < 0) {
			PKV_CHECK_MSG(false, "decode %d of %d produced no token (status %d)", i, n, static_cast<int>(s1));
			return false;
		}
		out->push_back(t);
	}
	return true;
}

// Decodes until the first step that produces no token; returns the tokens and the refusing
// status. `cap_steps` bounds the loop.
inline std::vector<int32_t> RunToRefusal(const Fixture& fx, sslm_seq s, sslm_status* refusal, int cap_steps = 70000) {
	std::vector<int32_t> t;
	sslm_status st = SSLM_OK;
	for (int i = 0; i < cap_steps; ++i) {
		const int32_t tok = NextToken(fx.model, s, &st);
		if (tok < 0) break;
		t.push_back(tok);
	}
	if (refusal) *refusal = st;
	return t;
}

// Leaves `s` mid-token: one decode call with layer_budget 1 on a sequence that is not ready for
// logits (it has emitted its ready token), which writes layer 0 of row context_length and stops.
inline bool EnterMidToken(const Fixture& fx, sslm_seq s) {
	sslm_decode_params p{};
	p.layer_budget = 1;
	sslm_seq b[1] = {s};
	int32_t tok = 0;
	const sslm_status st = sslm_decode_step(fx.model, b, 1, &p, nullptr, &tok);
	PKV_CHECK_MSG(st == SSLM_OK && tok == -1, "mid-token step: status %d, token %d (want 0, -1)", static_cast<int>(st), tok);
	return st == SSLM_OK && tok == -1;
}

inline std::vector<int32_t> Concat(std::vector<int32_t> a, const std::vector<int32_t>& b) {
	a.insert(a.end(), b.begin(), b.end());
	return a;
}

// A holder's page table through the C4 seam: entries [0, mapped). Empty when nothing is mapped.
inline std::vector<uint32_t> SeqTable(sslm_seq s) {
	std::vector<uint32_t> t;
	uint32_t mapped = 0, page = 0;
	if (sslm_pkv_test_only_seq_table_entry(s, 0, &page, &mapped) != SSLM_OK) return t;
	for (uint32_t i = 0; i < mapped; ++i) {
		PKV_CHECK_EQ(sslm_pkv_test_only_seq_table_entry(s, i, &page, nullptr), SSLM_OK);
		t.push_back(page);
	}
	return t;
}
inline std::vector<uint32_t> PrefixTable(sslm_prefix p) {
	std::vector<uint32_t> t;
	uint32_t mapped = 0, page = 0;
	if (sslm_pkv_test_only_prefix_table_entry(p, 0, &page, &mapped) != SSLM_OK) return t;
	for (uint32_t i = 0; i < mapped; ++i) {
		PKV_CHECK_EQ(sslm_pkv_test_only_prefix_table_entry(p, i, &page, nullptr), SSLM_OK);
		t.push_back(page);
	}
	return t;
}

inline std::vector<uint8_t> PeekPage(sslm_kv_pool pool, const Fixture& fx, uint32_t page) {
	std::vector<uint8_t> b(fx.PageBytes());
	PKV_CHECK_EQ(sslm_pkv_test_only_peek_page_bytes(pool, page, b.data(), b.size()), SSLM_OK);
	return b;
}

inline bool AllBytes(const std::vector<uint8_t>& b, uint8_t v) {
	return !b.empty() && std::all_of(b.begin(), b.end(), [v](uint8_t x) { return x == v; });
}

// Compares records against the named reference stages one by one (tokens, context length and the
// K/V rows' digest): for a cell that runs part of a scenario, where ExpectMatchesReference's
// whole-scenario record count does not apply.
inline void ExpectStagesMatch(const char* tag, const Fixture& fx, const char* scenario, const std::vector<Record>& got) {
	const RefFile& ref = Reference(tag, fx);
	PKV_CHECK_MSG(!got.empty(), "%s: no records", scenario);
	for (const Record& r : got) {
		const RefRecord* e = RefLookup(ref, scenario, r.stage);
		if (!e) continue;
		PKV_CHECK_MSG(r.tokens == e->tokens, "%s/%s: tokens differ from %s", scenario, r.stage.c_str(), tag);
		PKV_CHECK_MSG(r.context_length == e->context_length, "%s/%s: L %lld vs %lld", scenario, r.stage.c_str(),
		              static_cast<long long>(r.context_length), static_cast<long long>(e->context_length));
		PKV_CHECK_MSG(r.rows_sha == e->rows_sha, "%s/%s: K/V rows differ from %s", scenario, r.stage.c_str(), tag);
	}
}

// ---- pools ------------------------------------------------------------------------------------

// A fill-probe state over a page pool: the pool, its handles, and nothing else.
struct PageState : ProbeState {
	std::unique_ptr<PagePool> pool;
	Handles h;
	PageState(sslm_model m, uint32_t pages) : pool(std::make_unique<PagePool>(m, pages)) {
		PKV_CHECK_MSG(pool->status == SSLM_OK, "sslm_kv_page_pool_create(%u) -> %d", pages, static_cast<int>(pool->status));
	}
	~PageState() override { h.ReleaseAll(); }
	sslm_kv_pool* Pool() override { return &pool->pool; }
};

struct LegacyState : ProbeState {
	std::unique_ptr<LegacyPool> pool;
	Handles h;
	LegacyState(sslm_model m, uint32_t blocks) : pool(std::make_unique<LegacyPool>(m, blocks)) {
		PKV_CHECK_MSG(pool->status == SSLM_OK, "sslm_kv_pool_create(%u) -> %d", blocks, static_cast<int>(pool->status));
	}
	~LegacyState() override { h.ReleaseAll(); }
	sslm_kv_pool* Pool() override { return &pool->pool; }
};

// ---- prefixes ---------------------------------------------------------------------------------

// A frozen prefix of `tokens`: budget mode when budget > 0 (sslm_prefix_begin_budgeted), legacy
// (sslm_prefix_begin) when budget == 0. Null on any failure (a failed check).
inline sslm_prefix MakePrefix(const Fixture& fx, sslm_kv_pool* pool, const std::vector<int32_t>& tokens,
                              int32_t budget, int32_t chunk = 64) {
	sslm_prefix p = nullptr;
	const sslm_status st = budget > 0 ? sslm_prefix_begin_budgeted(fx.model, pool, budget, &p)
	                                  : sslm_prefix_begin(fx.model, pool, &p);
	PKV_CHECK_MSG(st == SSLM_OK, "prefix begin (budget %d) -> %d", budget, static_cast<int>(st));
	if (st != SSLM_OK) return nullptr;
	if (!tokens.empty()) PKV_CHECK_EQ(PrefixPrefillAll(fx.model, p, tokens, chunk), SSLM_OK);
	PKV_CHECK_EQ(sslm_prefix_freeze(p), SSLM_OK);
	return p;
}

// A child of a frozen parent through begin_from, prefilled with `tokens` and frozen.
inline sslm_prefix MakeChild(const Fixture& fx, sslm_prefix parent, const std::vector<int32_t>& tokens, int32_t budget) {
	sslm_prefix p = nullptr;
	const sslm_status st = sslm_prefix_begin_from(parent, budget, &p);
	PKV_CHECK_MSG(st == SSLM_OK, "begin_from (budget %d) -> %d", budget, static_cast<int>(st));
	if (st != SSLM_OK) return nullptr;
	if (!tokens.empty()) PKV_CHECK_EQ(PrefixPrefillAll(fx.model, p, tokens, 64), SSLM_OK);
	PKV_CHECK_EQ(sslm_prefix_freeze(p), SSLM_OK);
	return p;
}

inline sslm_seq MakeBudgetSeq(const Fixture& fx, sslm_kv_pool* pool, int32_t budget) {
	sslm_seq s = nullptr;
	const sslm_status st = sslm_seq_create_budgeted(fx.model, pool, budget, &s);
	PKV_CHECK_MSG(st == SSLM_OK, "sslm_seq_create_budgeted(%d) -> %d", budget, static_cast<int>(st));
	return st == SSLM_OK ? s : nullptr;
}

inline sslm_seq MakeLegacySeq(const Fixture& fx, sslm_kv_pool* pool) {
	sslm_seq s = nullptr;
	const sslm_status st = sslm_seq_create(fx.model, pool, &s);
	PKV_CHECK_MSG(st == SSLM_OK, "sslm_seq_create -> %d", static_cast<int>(st));
	return st == SSLM_OK ? s : nullptr;
}

// The cohort chain of §7 (9.9-9.11, 10.x): a 1,000-token world and a 200-token persona through
// begin_from, so a persona adopter's origin is 1,200 (75 full pages, aligned). Page counts by §3.4,
// written in the test: the world draws R(1000) = 64 and keeps ceil(1000/16) = 63 after freeze; the
// persona draws R(200) = 14, shares the world's 62 full pages, copies its tail into one reserve page,
// maps 13 private pages for [992, 1200) and returns 1 at freeze. The chain's persona table is 75 pages.
constexpr int32_t kWorldLen = 1000;
constexpr int32_t kPersonaLen = 200;
constexpr int32_t kPersonaOrigin = kWorldLen + kPersonaLen;  // 1,200
inline std::vector<int32_t> WorldTokens(int32_t vocab) { return Stream(61, kWorldLen, vocab); }
inline std::vector<int32_t> PersonaTokens(int32_t vocab, uint32_t which = 0) {
	return Stream(62 + which, kPersonaLen, vocab);
}
// Pages the chain holds with the world live: the world's 63 and the persona's 13 private ones.
constexpr int64_t kChainPagesWorldLive = 63 + 13;
// Pages the persona's table maps (62 shared from the world, 13 private): the "persona chain 75".
constexpr int64_t kPersonaTablePages = 75;

struct Chain {
	sslm_prefix world = nullptr;
	sslm_prefix persona = nullptr;
};

inline Chain BuildChain(const Fixture& fx, sslm_kv_pool* pool, Handles* h, uint32_t which = 0) {
	Chain c;
	c.world = MakePrefix(fx, pool, WorldTokens(fx.vocab), kWorldLen);
	if (c.world) h->prefixes.push_back(c.world);
	if (c.world) c.persona = MakeChild(fx, c.world, PersonaTokens(fx.vocab, which), kPersonaLen);
	if (c.persona) h->prefixes.push_back(c.persona);
	return c;
}

// ---- the budget twins of the R0 scenarios --------------------------------------------------------

// pkv_scenarios.h Lifecycle with a budget holder (1.1's budget half): a budget-300 prefix of 300
// tokens and one sequence of budget 112 driven fresh -> prefill 100 -> reset -> prefill 40 -> reset
// -> adopt -> decode, the reference's stages. Budget 112 covers the largest generation of the
// script (100 + 3 + 8 rows from origin 0; 7 + 37 + 3 rows from origin 300), so nothing is refused.
// Records compare with the reference's "lifecycle" records, tokens and rows.
inline std::vector<Record> BudgetLifecycle(const Fixture& fx, int32_t seq_budget = 112) {
	std::vector<Record> recs;
	std::vector<int32_t> pending;
	PagePool pool(fx.model, static_cast<uint32_t>(fx.R(300) + fx.R(seq_budget) + 4));
	PKV_CHECK_EQ(pool.status, SSLM_OK);
	if (pool.status != SSLM_OK) return recs;
	sslm_prefix px = MakePrefix(fx, &pool.pool, Stream(3, 300, fx.vocab), 300);
	sslm_seq s = MakeBudgetSeq(fx, &pool.pool, seq_budget);
	if (px && s) {
		bool ok = PrefillAll(fx.model, s, Stream(1, 100, fx.vocab), 64) == SSLM_OK;
		if (ok) recs.push_back(MakeRecord(fx, "prefill100", s, &pending));
		ok = ok && DecodeN(fx, s, 4, &pending);
		if (ok) recs.push_back(MakeRecord(fx, "prefill100+decode4", s, &pending));
		ok = ok && DecodeN(fx, s, 8, &pending);
		if (ok) recs.push_back(MakeRecord(fx, "continue8", nullptr, &pending));
		ok = ok && sslm_seq_reset(s) == SSLM_OK;
		if (ok) recs.push_back(MakeRecord(fx, "reset1", s, &pending));
		ok = ok && PrefillAll(fx.model, s, Stream(2, 40, fx.vocab), 16) == SSLM_OK && DecodeN(fx, s, 4, &pending);
		if (ok) recs.push_back(MakeRecord(fx, "prefill40+decode4", s, &pending));
		ok = ok && sslm_seq_reset(s) == SSLM_OK && sslm_seq_adopt_prefix(s, px) == SSLM_OK;
		if (ok) recs.push_back(MakeRecord(fx, "adopt300", s, &pending));
		ok = ok && DecodeN(fx, s, 8, &pending);
		if (ok) recs.push_back(MakeRecord(fx, "adopt300+decode8", s, &pending));
		ok = ok && PrefillAll(fx.model, s, Stream(4, 37, fx.vocab), 7) == SSLM_OK && DecodeN(fx, s, 4, &pending);
		if (ok) recs.push_back(MakeRecord(fx, "adopt+prefill37+decode4", s, &pending));
		PKV_CHECK_MSG(ok, "budget lifecycle stopped after %zu records", recs.size());
	}
	if (s) sslm_seq_release(s);
	if (px) sslm_prefix_release(px);
	return recs;
}

// pkv_scenarios.h Widths with budget holders: one budget sequence per width W, budget
// min(W + 1, cap), so the widths' last write is its limit's last admissible position, in a page
// pool of exactly R(budget) pages. Records compare with the reference's "widths" records.
inline std::vector<Record> BudgetWidths(const Fixture& fx) {
	std::vector<Record> recs;
	std::vector<int32_t> pending;
	const int64_t cap = fx.geo.context_cap;
	std::vector<int64_t> widths = {1, 15, 16, 17, 32, 1064};
	if (cap <= 4100) {
		widths.push_back(cap - 1);
		widths.push_back(cap);
	}
	for (int64_t w : widths) {
		const int32_t budget = static_cast<int32_t>(std::min<int64_t>(w + 1, cap));
		PagePool pool(fx.model, static_cast<uint32_t>(fx.R(budget)));
		PKV_CHECK_EQ(pool.status, SSLM_OK);
		sslm_seq s = pool.status == SSLM_OK ? MakeBudgetSeq(fx, &pool.pool, budget) : nullptr;
		if (!s) return recs;
		const std::string tag = "w" + std::to_string(w);
		bool ok = PrefillAll(fx.model, s, Stream(41, static_cast<int32_t>(w), fx.vocab), 64) == SSLM_OK;
		if (ok) recs.push_back(MakeRecord(fx, (tag + "_prefill").c_str(), s, &pending));
		ok = ok && DecodeN(fx, s, 1, &pending);
		if (ok) recs.push_back(MakeRecord(fx, (tag + "_ready").c_str(), nullptr, &pending));
		if (ok && w < cap) {
			ok = DecodeN(fx, s, 1, &pending);
			if (ok) recs.push_back(MakeRecord(fx, (tag + "_decode1").c_str(), s, &pending));
		}
		sslm_seq_release(s);
		if (!ok) return recs;
	}
	return recs;
}

// pkv_scenarios.h PrefixLengths with a budget prefix (budget max(P, 1)) and a budget adopter of
// budget 30, which is exactly the rows the script writes after the adopt (7 + 20 + 3), so the
// adopter runs to its limit's last position. Records compare with "prefix_lengths".
inline std::vector<Record> BudgetPrefixLengths(const Fixture& fx) {
	std::vector<Record> recs;
	std::vector<int32_t> pending;
	PagePool pool(fx.model, static_cast<uint32_t>(fx.R(1008) + fx.R(30)));
	PKV_CHECK_EQ(pool.status, SSLM_OK);
	if (pool.status != SSLM_OK) return recs;
	for (int32_t len : {0, 1, 15, 16, 17, 1000, 1008}) {
		sslm_prefix px = MakePrefix(fx, &pool.pool, len ? Stream(51, len, fx.vocab) : std::vector<int32_t>{},
		                            std::max(len, 1));
		sslm_seq s = px ? MakeBudgetSeq(fx, &pool.pool, 30) : nullptr;
		const std::string tag = "p" + std::to_string(len);
		bool ok = px && s && sslm_seq_adopt_prefix(s, px) == SSLM_OK;
		if (ok) recs.push_back(MakeRecord(fx, (tag + "_adopted").c_str(), s, &pending));
		if (ok && len > 0) {
			ok = DecodeN(fx, s, 8, &pending);
			if (ok) recs.push_back(MakeRecord(fx, (tag + "_decode8").c_str(), s, &pending));
		}
		ok = ok && PrefillAll(fx.model, s, Stream(52, 20, fx.vocab), 64) == SSLM_OK && DecodeN(fx, s, 4, &pending);
		if (ok) recs.push_back(MakeRecord(fx, (tag + "_prefill20+decode4").c_str(), s, &pending));
		if (s) sslm_seq_release(s);
		if (px) sslm_prefix_release(px);
		if (!ok) return recs;
	}
	return recs;
}

// ---- the used-state adopt census in budget mode (1.10; 8.11; 7.13) ----------------------------------
//
// cell_adopt_prefix_census.cpp's population on pkv_def's g5_minimal_one_field schema: 10 resting
// states of the adopting sequence x 3 prefix kinds x {live, restored}, with budget prefixes and
// budget sequences in one page pool. The prompts are lengthened so every state crosses a page
// boundary: the prefix holds 37 prompt tokens (two full pages and a mid-page tail, so share adopt
// maps two pages and copies one), the sequence's own prompt is 20 tokens. Each row asserts, against
// a fresh budget holder adopting the same prefix: the adopt status, the whole SSB6 blob, sslm_stats'
// forced_token_count and schema_accepting, and the next 4 tokens; then the pool's free pages,
// exactly, by the fill probe's one-state form on the live state (§8); then that every private page
// the adopter maps reads 0xCD through the page-index peek after its release (dim 1).
struct CensusResult {
	int rows = 0;
	int mismatches = 0;
};

constexpr int32_t kCensusPrefixBudget = 64;  // R = 5; the frozen prefix keeps ceil(38/16) = 3
constexpr int32_t kCensusSeqBudget = 48;     // R = 4
constexpr int64_t kCensusPrefixPrompt = 37;

inline std::vector<int32_t> CensusGreedy(const Fixture& fx, sslm_seq s, int steps) {
	sslm_decode_params p{};
	p.layer_budget = static_cast<int32_t>(fx.geo.layers);
	sslm_seq b[1] = {s};
	std::vector<int32_t> t;
	for (int i = 0; i < steps; ++i) {
		int32_t tok = -7;
		const sslm_status st = sslm_decode_step(fx.model, b, 1, &p, nullptr, &tok);
		t.push_back(st == SSLM_OK ? tok : -1000 - static_cast<int>(st));
	}
	return t;
}

inline CensusResult RunBudgetCensus(const Fixture& fx) {
	CensusResult res;
	sslm_schema schema = nullptr;
	PKV_CHECK_EQ(sslm_schema_lookup(fx.model, "g5_minimal_one_field", &schema), SSLM_OK);
	if (!schema) return res;
	// Pool: the prefix's 3 pages, the adopter, its saved original while a restore is made, the fresh
	// reference, and 13 free pages for the one-state probe (k = 13 >= 3).
	const int64_t r_seq = fx.R(kCensusSeqBudget);
	const int64_t slack = 13;
	const uint32_t P = static_cast<uint32_t>(fx.R(kCensusPrefixBudget) + 3 * r_seq + slack);
	PagePool pool(fx.model, P);
	PKV_CHECK_EQ(pool.status, SSLM_OK);
	if (pool.status != SSLM_OK) return res;
	const std::vector<int32_t> seq_prompt = Stream(72, 20, fx.vocab);
	auto prefill_prompt = [&](sslm_seq s) { PKV_CHECK_EQ(PrefillAll(fx.model, s, seq_prompt, 8), SSLM_OK); };
	// One schema-admitted token from the start state, derived on the fixture (the census's way).
	int32_t admitted = -1;
	{
		sslm_seq d = MakeBudgetSeq(fx, &pool.pool, kCensusSeqBudget);
		if (!d) return res;
		PKV_CHECK_EQ(sslm_seq_set_schema(d, schema), SSLM_OK);
		prefill_prompt(d);
		admitted = CensusGreedy(fx, d, 1)[0];
		sslm_seq_release(d);
	}
	PKV_CHECK_MSG(admitted >= 0, "census: no schema-admitted token from the start state");
	if (admitted < 0) return res;
	enum PK { P_PROMPT, P_BOUND_START, P_PROGRESS };
	enum SK { FRESH, POST_PREFILL, POST_GREEDY, MIDTOKEN, B_FRESH, B_POST_PREFILL, B_ADVANCED, B_FORCED, B_DEADEND,
		      B_FORCED_DEAD };
	static const char* const kPk[] = {"P_PROMPT", "P_BOUND_START", "P_PROGRESS"};
	static const char* const kSk[] = {"FRESH",          "POST_PREFILL", "POST_GREEDY", "MIDTOKEN", "B_FRESH",
	                                  "B_POST_PREFILL", "B_ADVANCED",   "B_FORCED",    "B_DEADEND", "B_FORCED_DEAD"};
	for (int pk = P_PROMPT; pk <= P_PROGRESS; ++pk) {
		sslm_prefix prefix = nullptr;
		PKV_CHECK_EQ(sslm_prefix_begin_budgeted(fx.model, &pool.pool, kCensusPrefixBudget, &prefix), SSLM_OK);
		if (!prefix) return res;
		if (pk != P_PROMPT) PKV_CHECK_EQ(sslm_prefix_set_schema(prefix, schema), SSLM_OK);
		PKV_CHECK_EQ(PrefixPrefillAll(fx.model, prefix, Stream(71, static_cast<int32_t>(kCensusPrefixPrompt), fx.vocab), 8),
		             SSLM_OK);
		int64_t plen = kCensusPrefixPrompt;
		if (pk == P_PROGRESS) {
			int32_t c = 0;
			PKV_CHECK(sslm_prefix_prefill(fx.model, prefix, &admitted, 1, 8, SSLM_SPAN_SCHEMA_CONTENT, nullptr, &c) ==
			              SSLM_OK &&
			          c == 1);
			++plen;
		}
		PKV_CHECK_EQ(sslm_prefix_freeze(prefix), SSLM_OK);
		const int64_t held_by_prefix = (plen + fx.B() - 1) / fx.B();
		for (int sk = FRESH; sk <= B_FORCED_DEAD; ++sk) {
			const bool bound = sk >= B_FRESH;
			for (int restored = 0; restored <= 1; ++restored) {
				sslm_seq s = MakeBudgetSeq(fx, &pool.pool, kCensusSeqBudget);
				if (!s) continue;
				if (bound) PKV_CHECK_EQ(sslm_seq_set_schema(s, schema), SSLM_OK);
				switch (sk) {
					case FRESH:
					case B_FRESH:
						break;
					case POST_PREFILL:
					case B_POST_PREFILL:
						prefill_prompt(s);
						break;
					case POST_GREEDY:
						prefill_prompt(s);
						CensusGreedy(fx, s, 2);
						break;
					case MIDTOKEN:
						prefill_prompt(s);
						CensusGreedy(fx, s, 1);  // the ready token
						EnterMidToken(fx, s);
						break;
					case B_ADVANCED:
						prefill_prompt(s);
						CensusGreedy(fx, s, 1);
						break;
					case B_FORCED:
					case B_FORCED_DEAD: {
						prefill_prompt(s);
						int32_t c = 0;
						PKV_CHECK(sslm_prefill(fx.model, s, &admitted, 1, 8, SSLM_SPAN_SCHEMA_CONTENT, nullptr, &c) ==
						              SSLM_OK &&
						          c == 1);
						if (sk == B_FORCED_DEAD) CensusGreedy(fx, s, 2);
						break;
					}
					case B_DEADEND:
						prefill_prompt(s);
						CensusGreedy(fx, s, 3);
						break;
				}
				sslm_seq target = s;
				if (restored) {
					const std::vector<uint8_t> b = Save(s);
					target = nullptr;
					PKV_CHECK_EQ(sslm_seq_restore(fx.model, &pool.pool, b.data(), b.size(), &target), SSLM_OK);
					sslm_seq_release(s);
					if (!target) continue;
				}
				const sslm_status st = sslm_seq_adopt_prefix(target, prefix);
				sslm_seq ref = MakeBudgetSeq(fx, &pool.pool, kCensusSeqBudget);
				if (!ref) {
					sslm_seq_release(target);
					continue;
				}
				if (bound) PKV_CHECK_EQ(sslm_seq_set_schema(ref, schema), SSLM_OK);
				const sslm_status rst = sslm_seq_adopt_prefix(ref, prefix);
				std::string why;
				if (st != rst) why += " status";
				if (st == SSLM_OK && rst == SSLM_OK) {
					if (Save(target) != Save(ref)) why += " blob";  // kills: share adopt that keeps any used-state byte
					sslm_stats_out a{}, b{};
					PKV_CHECK_EQ(sslm_stats(fx.model, target, &a), SSLM_OK);
					PKV_CHECK_EQ(sslm_stats(fx.model, ref, &b), SSLM_OK);
					if (a.forced_token_count != b.forced_token_count) why += " forced_token_count";
					if (a.schema_accepting != b.schema_accepting) why += " schema_accepting";
					if (CensusGreedy(fx, target, 4) != CensusGreedy(fx, ref, 4)) why += " next4";
				}
				++res.rows;
				if (!why.empty()) ++res.mismatches;
				PKV_CHECK_MSG(why.empty(), "census %s %s %s:%s", kPk[pk], kSk[sk], restored ? "restored" : "live",
				              why.c_str());
				// kills: an adopt onto a used holder that leaks or double-counts a page (a restored
				// holder's materialized pages, E = 0 here at origin 0, return to the pool). Free
				// pages: P minus the prefix's pages minus two reservations of R(48).
				const int64_t k = static_cast<int64_t>(P) - held_by_prefix - 2 * r_seq;
				ProbeExactlyFreeOneState(fx, &pool.pool, k);
				// The adopter's private mapped pages, found through the table seam, read 0xCD after its
				// release (they were written, so they are dirty; §3.3 "Freeing a page").
				const std::vector<uint32_t> table = SeqTable(target);
				const size_t shared = static_cast<size_t>(plen / fx.B());
				const std::vector<uint32_t> priv(table.begin() + static_cast<std::ptrdiff_t>(std::min(shared, table.size())),
				                                 table.end());
				sslm_seq_release(target);
				for (uint32_t pg : priv)
					PKV_CHECK_MSG(AllBytes(PeekPage(pool.pool, fx, pg), 0xCD), "census %s %s: freed page %u does not read 0xCD",
					              kPk[pk], kSk[sk], pg);
				sslm_seq_release(ref);
			}
		}
		sslm_prefix_release(prefix);
	}
	PKV_CHECK_EQ(res.rows, 60);
	return res;
}

// ---- a synthetic adapter (8.4, 8.12) ------------------------------------------------------------
//
// A rank-1 LoRA on layer 0's q_proj and layer 1's v_proj, identity fold triples, built against the
// fixture's own Config and SigmoidLut sections and its integrity hash (sslm_adapter_map's
// base-hash check). The v_proj delta changes the K/V rows an adopter writes after its origin, so a
// cell crossing adapters with share adopt compares rows that the adapter itself moved.
inline std::vector<uint8_t> BuildAdapterBytes(const Fixture& fx) {
	using namespace superslm_test;
	superslm::SslmArtifact art;
	superslm::SslmError err;
	if (superslm::SslmArtifact::OpenFromMemory(fx.bytes.p, fx.size, art, &err) != superslm::SslmStatus::Ok) {
		PKV_CHECK_MSG(false, "adapter: the fixture does not open as an artifact");
		return {};
	}
	const superslm::SslmSectionView* cfg = art.Section(superslm::SslmSectionType::Config);
	const superslm::SslmSectionView* sil = art.Section(superslm::SslmSectionType::SigmoidLut);
	if (!cfg || !sil) {
		PKV_CHECK_MSG(false, "adapter: the fixture has no Config or SigmoidLut section");
		return {};
	}
	std::vector<uint8_t> adp = {'A', 'D', 'P', '1'};
	WriteU32LE(adp, 1);            // version
	WriteU32LE(adp, 1);            // rank
	WriteU32LE(adp, 0x1u | 0x40u);  // q_proj, v_proj
	const auto hash = art.RawIntegrityHash();
	adp.insert(adp.end(), hash.begin(), hash.end());
	const double alpha = 8.0;
	uint64_t bits = 0;
	std::memcpy(&bits, &alpha, 8);
	WriteU64LE(adp, bits);
	WriteU32LE(adp, 0);  // use_rslora
	WriteU32LE(adp, 0);  // reserved
	const std::string name = "pkv-budget-b";
	WriteU32LE(adp, static_cast<uint32_t>(name.size()));
	WriteBytes(adp, name);
	struct Proj {
		std::string name;
		uint32_t in, out;
	};
	const uint32_t hidden = static_cast<uint32_t>(HiddenSize(fx));
	const std::vector<Proj> projs = {{"layer0.q_proj", hidden, 4u * fx.geo.head_dim},
	                                 {"layer1.v_proj", hidden, fx.geo.kv_heads * fx.geo.head_dim}};
	std::vector<ManifestTensorSpec> wt, dt, ut;
	for (const Proj& p : projs) {
		wt.push_back({p.name + ".lora_A", {1, p.in}});
		wt.push_back({p.name + ".lora_B", {p.out, 1}});
		dt.push_back({p.name, {p.out * 3}});
		ut.push_back({p.name, {3}});
	}
	BuiltManifest w = BuildManifest(reinterpret_cast<const uint8_t*>("WGT1"), 1, wt);
	BuiltManifest d = BuildManifest(superslm::kDeltaFoldScalesMagic, 4, dt);
	BuiltManifest u = BuildManifest(superslm::kUFoldScalesMagic, 4, ut);
	for (BuiltManifest* m : {&d, &u})
		for (size_t t = 0; t < m->tensor_data_off.size(); ++t)
			for (uint64_t r = 0; r < m->tensor_elem_count[t] / 3; ++r) {
				const int32_t triple[3] = {1, 0, 0};
				std::memcpy(m->bytes.data() + m->tensor_data_off[t] + r * 12, triple, 12);
			}
	auto copy = [](superslm::SslmSectionType t, const superslm::SslmSectionView* v) {
		return MakeSection(t, v->dtype, std::vector<uint8_t>(v->data, v->data + v->byte_size));
	};
	BuiltArtifact a = BuildArtifact({copy(superslm::SslmSectionType::Config, cfg),
	                                 copy(superslm::SslmSectionType::SigmoidLut, sil),
	                                 MakeSection(superslm::SslmSectionType::Provenance, superslm::SslmDtype::Raw, adp),
	                                 MakeSection(superslm::SslmSectionType::Weights, superslm::SslmDtype::Int8, w.bytes),
	                                 MakeSection(superslm::SslmSectionType::DeltaFoldScales, superslm::SslmDtype::Int32, d.bytes),
	                                 MakeSection(superslm::SslmSectionType::UFoldScales, superslm::SslmDtype::Int32, u.bytes)});
	return a.bytes;
}

struct Adapter {
	std::vector<uint8_t> bytes;
	sslm_adapter a = nullptr;
	explicit Adapter(const Fixture& fx) : bytes(BuildAdapterBytes(fx)) {
		if (!bytes.empty()) PKV_CHECK_EQ(sslm_adapter_map(bytes.data(), bytes.size(), fx.model, &a), SSLM_OK);
	}
	~Adapter() {
		if (a) sslm_adapter_release(a);
	}
};

// ---- the table-sentinel child (7.13) ------------------------------------------------------------
//
// Runs `body` in a forked child with sslm_pkv_test_only_set_table_sentinel(1). The child exits 0
// when every check it made passed, 3 when a check failed; a read of a sentinel entry aborts it
// (SIGABRT). Returns a description of how the child ended ("" when it exited 0). On a platform
// without fork the body runs in-process: a trap then ends the runner, which still fails the step.
inline std::string RunUnderSentinel(const std::function<void()>& body) {
#if PKV_BUDGET_B_HAVE_FORK
	std::fflush(stdout);
	std::fflush(stderr);
	const pid_t pid = fork();
	if (pid < 0) return "fork failed";
	if (pid == 0) {
		const int before = State().failures;
		sslm_pkv_test_only_set_table_sentinel(1);
		body();
		std::fflush(stdout);
		std::fflush(stderr);
		std::_Exit(State().failures > before ? 3 : 0);
	}
	int status = 0;
	if (waitpid(pid, &status, 0) != pid) return "waitpid failed";
	if (WIFEXITED(status)) {
		if (WEXITSTATUS(status) == 0) return "";
		return "the child's own checks failed (exit " + std::to_string(WEXITSTATUS(status)) + ")";
	}
	if (WIFSIGNALED(status))
		return "the child was killed by signal " + std::to_string(WTERMSIG(status)) +
		       " (a read of a table entry at or above mapped trapped on the sentinel)";
	return "the child ended abnormally";
#else
	const int before = State().failures;
	sslm_pkv_test_only_set_table_sentinel(1);
	body();
	sslm_pkv_test_only_set_table_sentinel(0);
	return State().failures > before ? "checks failed under the sentinel" : "";
#endif
}

}  // namespace budget_b
}  // namespace pkv

#endif  // SUPERSLM_TESTS_PKV_BUDGET_B_HELPERS_H

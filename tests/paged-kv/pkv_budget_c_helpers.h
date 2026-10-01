// Paged-KV plan (rev 16.1) step C1: helpers shared by the budget_c group's cells (C5's dimension 9
// and 11 cells, C6's timing and achievement cells).
//
// Everything here is test-side: token drivers over the public verbs, the persona chain of §3.7 and
// §5 (world 1,000 -> persona 1,200 by `begin_from`), page arithmetic written from §3.1 and §3.4,
// readers of the page-index peek addressed by the §3.1 formula, and rebuildable states for the
// two-sided fill probe (§8). Nothing reads the code under test's own stats. Helpers that call a
// C5 verb are inline, so an executable that never calls them never references the verb.

#ifndef SUPERSLM_TESTS_PKV_BUDGET_C_HELPERS_H
#define SUPERSLM_TESTS_PKV_BUDGET_C_HELPERS_H

#include "pkv_common.h"

#include <algorithm>
#include <memory>
#include <string>
#include <vector>

namespace pkv {
namespace bc {

// The value an out-handle is preset to before a verb whose refusal a cell checks by a null out:
// non-null, so a verb that never writes *out leaves it standing and the check fails. Never released
// or dereferenced -- release a handle only when it is neither null nor this.
template <class H>
inline H OutSentinel() {
	return reinterpret_cast<H>(uintptr_t{0x1});
}

constexpr int32_t kChunk = 64;
constexpr int32_t kVocab = 256;  // every token stream stays inside the fixtures' vocabulary (and any real one)

// ---- the persona chain's prompts (§5's cohort: a world of 1,000, ten personas of 200) -----------

inline std::vector<int32_t> WorldTokens() { return Stream(61, 1000, kVocab); }
inline std::vector<int32_t> PersonaTokens(int i) { return Stream(71 + static_cast<uint32_t>(i), 200, kVocab); }
// A plain prefix of length `len` for the population cells (seed 0: the "matching" content).
inline std::vector<int32_t> PrefixTokens(int32_t len, uint32_t seed = 0) { return Stream(51 + seed * 7, len, kVocab); }

// ---- page arithmetic, written from §3.1 / §3.4 / §3.7 ---------------------------------------------

inline int64_t CeilDiv(int64_t a, int64_t b) { return (a + b - 1) / b; }
// Pages a frozen prefix of `len` positions keeps: ceil(len / B) (§3.4, freeze).
inline int64_t PrefixPages(const Fixture& fx, int64_t len) { return CeilDiv(len, fx.B()); }
// Full pages a matching restore or a budget adopt shares: floor(origin / B) (§3.5, §3.7).
inline int64_t SharedPages(const Fixture& fx, int64_t origin) { return origin / fx.B(); }
// §3.7's materialized pages for a private restore: E = min(floor(origin/B), ceil(cap/B) - R(budget)).
inline int64_t MaterializedPages(const Fixture& fx, int64_t origin, int64_t budget) {
	const int64_t a = origin / fx.B(), b = fx.CapPages() - fx.R(budget);
	return a < b ? a : b;
}
inline int64_t Limit(const Fixture& fx, int64_t origin, int64_t budget) {
	return std::min<int64_t>(origin + budget, fx.geo.context_cap);
}

// ---- token drivers ----------------------------------------------------------------------------

// One whole greedy token, one layer per call (the ABI caps layer_budget at the model's layer count,
// which the fixtures and the real artifacts do not share). Returns the token, or -1 with the status in
// *st; a -2 or -3 sentinel is returned as is.
inline int32_t Token(sslm_model model, sslm_seq s, sslm_status* st = nullptr) {
	sslm_decode_params p{};
	p.layer_budget = 1;
	sslm_seq b[1] = {s};
	int32_t tok = -1;
	sslm_status r = SSLM_OK;
	for (int guard = 0; tok == -1 && guard < 4096; ++guard) {
		r = sslm_decode_step(model, b, 1, &p, nullptr, &tok);
		if (r != SSLM_OK) break;
	}
	if (st) *st = r;
	return r == SSLM_OK ? tok : -1;
}

// Up to `n` greedy tokens; stops at the first non-OK status (returned in *last, SSLM_OK if none).
inline std::vector<int32_t> DecodeN(sslm_model model, sslm_seq s, int64_t n, sslm_status* last = nullptr) {
	std::vector<int32_t> out;
	sslm_status st = SSLM_OK;
	for (int64_t i = 0; i < n; ++i) {
		const int32_t t = Token(model, s, &st);
		if (st != SSLM_OK || t < 0) break;  // a schema dead end (-2) ends the run with SSLM_OK
		out.push_back(t);
	}
	if (last) *last = st;
	return out;
}

// Greedy tokens until the first refusal; the refusal's status in *refusal.
inline std::vector<int32_t> RunToRefusal(sslm_model model, sslm_seq s, sslm_status* refusal) {
	return DecodeN(model, s, 1 << 20, refusal);
}

// One layer of the next token (layer_budget 1): leaves a two-layer fixture mid-token.
inline sslm_status LayerStep(sslm_model model, sslm_seq s, int32_t* out_token = nullptr) {
	sslm_decode_params p{};
	p.layer_budget = 1;
	sslm_seq b[1] = {s};
	int32_t tok = -1;
	const sslm_status r = sslm_decode_step(model, b, 1, &p, nullptr, &tok);
	if (out_token) *out_token = tok;
	return r;
}

// The "next turn" of §3.9's keeps rows: greedy to the first refusal. A holder with nothing to decode
// from (an empty prefix adopted, no token yet: decode refuses SSLM_INVALID_ARGUMENT) first takes a
// one-token user turn. The same rule runs on the original and on its twin.
struct Turn {
	std::vector<int32_t> tokens;
	sslm_status prefill = SSLM_OK;  // the one-token turn's status, SSLM_OK when none was needed
	sslm_status refusal = SSLM_OK;
	bool operator==(const Turn& o) const { return tokens == o.tokens && prefill == o.prefill && refusal == o.refusal; }
};
inline Turn NextTurn(sslm_model model, sslm_seq s) {
	Turn t;
	sslm_status st = SSLM_OK;
	const int32_t first = Token(model, s, &st);
	if (st == SSLM_INVALID_ARGUMENT) {
		const std::vector<int32_t> one = Stream(82, 1, kVocab);
		int32_t consumed = 0;
		t.prefill = sslm_prefill(model, s, one.data(), 1, 1, SSLM_SPAN_PROMPT, nullptr, &consumed);
		t.tokens = RunToRefusal(model, s, &t.refusal);
		return t;
	}
	if (st != SSLM_OK) {
		t.refusal = st;
		return t;
	}
	t.tokens.push_back(first);
	const std::vector<int32_t> rest = RunToRefusal(model, s, &t.refusal);
	t.tokens.insert(t.tokens.end(), rest.begin(), rest.end());
	return t;
}

inline std::vector<uint8_t> Save(sslm_seq s) {
	std::vector<uint8_t> b;
	const bool ok = s && SaveBlob(s, &b);
	PKV_CHECK_MSG(ok, "sslm_seq_save failed");
	if (!ok) b.clear();
	return b;
}

// ---- prefixes -------------------------------------------------------------------------------

// A budget-mode prefix of `tokens` (budget = max(len, 1)), frozen unless `freeze` is false.
inline sslm_prefix BudgetPrefix(const Fixture& fx, sslm_kv_pool* pool, const std::vector<int32_t>& tokens,
                                bool freeze = true, sslm_status* st_out = nullptr) {
	sslm_prefix p = nullptr;
	const int32_t budget = tokens.empty() ? 1 : static_cast<int32_t>(tokens.size());
	sslm_status st = sslm_prefix_begin_budgeted(fx.model, pool, budget, &p);
	if (st == SSLM_OK && !tokens.empty()) st = PrefixPrefillAll(fx.model, p, tokens, kChunk);
	if (st == SSLM_OK && freeze) st = sslm_prefix_freeze(p);
	if (st_out) *st_out = st;
	else PKV_CHECK_MSG(st == SSLM_OK, "budget prefix of %zu tokens: status %d", tokens.size(), static_cast<int>(st));
	if (st != SSLM_OK && p) {
		sslm_prefix_release(p);
		p = nullptr;
	}
	return p;
}

// A legacy (whole_reserve) prefix of `tokens`, frozen.
inline sslm_prefix LegacyPrefix(const Fixture& fx, sslm_kv_pool* pool, const std::vector<int32_t>& tokens) {
	sslm_prefix p = nullptr;
	sslm_status st = sslm_prefix_begin(fx.model, pool, &p);
	if (st == SSLM_OK && !tokens.empty()) st = PrefixPrefillAll(fx.model, p, tokens, kChunk);
	if (st == SSLM_OK) st = sslm_prefix_freeze(p);
	PKV_CHECK_MSG(st == SSLM_OK, "legacy prefix of %zu tokens: status %d", tokens.size(), static_cast<int>(st));
	if (st != SSLM_OK && p) {
		sslm_prefix_release(p);
		p = nullptr;
	}
	return p;
}

// The persona chain: a budget world of 1,000 (budget 1,000, R = 64, 63 pages after freeze), then a
// persona by `begin_from(world, 200)` (R(200) = 14 drawn; 13 private pages after its 200 tokens and
// freeze). With the world released the chain holds 62 + 13 = 75 pages, §7 9.9's "persona chain 75
// pages". `persona_tokens` replaces the persona's 200 tokens when given (9.10's producers).
struct PersonaChain {
	sslm_prefix world = nullptr;
	sslm_prefix persona = nullptr;
	bool ok = false;
	void Release() {
		if (persona) sslm_prefix_release(persona);
		if (world) sslm_prefix_release(world);
		persona = world = nullptr;
	}
};
inline PersonaChain BuildPersonaChain(const Fixture& fx, sslm_kv_pool* pool, int persona_index, bool release_world,
                                      const std::vector<int32_t>* persona_tokens = nullptr, bool freeze_persona = true) {
	PersonaChain c;
	sslm_status st = sslm_prefix_begin_budgeted(fx.model, pool, 1000, &c.world);
	if (st == SSLM_OK) st = PrefixPrefillAll(fx.model, c.world, WorldTokens(), kChunk);
	if (st == SSLM_OK) st = sslm_prefix_freeze(c.world);
	const std::vector<int32_t> pt = persona_tokens ? *persona_tokens : PersonaTokens(persona_index);
	if (st == SSLM_OK) st = sslm_prefix_begin_from(c.world, static_cast<int32_t>(pt.size()), &c.persona);
	if (st == SSLM_OK) st = PrefixPrefillAll(fx.model, c.persona, pt, kChunk);
	if (st == SSLM_OK && freeze_persona) st = sslm_prefix_freeze(c.persona);
	if (st == SSLM_OK && release_world) {
		st = sslm_prefix_release(c.world);
		c.world = nullptr;
	}
	PKV_CHECK_MSG(st == SSLM_OK, "persona chain %d: status %d", persona_index, static_cast<int>(st));
	c.ok = st == SSLM_OK;
	return c;
}
// The chain's pages, from §3.4 written here: the world's ceil(1000/B) less its tail page once it is
// released (the persona copied it), plus the persona's own pages from the world's tail page on.
inline int64_t PersonaChainPages(const Fixture& fx, int64_t persona_len, bool world_released) {
	const int64_t world = PrefixPages(fx, 1000);
	const int64_t persona_own = PrefixPages(fx, 1000 + persona_len) - SharedPages(fx, 1000);
	return (world_released ? SharedPages(fx, 1000) : world) + persona_own;
}

// ---- the page-index peek, addressed by §3.1's formula --------------------------------------------

inline std::vector<uint8_t> PeekPage(const Fixture& fx, sslm_kv_pool pool, uint32_t page) {
	std::vector<uint8_t> b(fx.PageBytes());
	const sslm_status st = sslm_pkv_test_only_peek_page_bytes(pool, page, b.data(), b.size());
	PKV_CHECK_MSG(st == SSLM_OK, "peek page %u: status %d", page, static_cast<int>(st));
	return b;
}

// Every mapped page of a prefix, by its own table (seam), as (page index, bytes).
struct PageSnap {
	std::vector<uint32_t> index;
	std::vector<std::vector<uint8_t>> bytes;
};
inline PageSnap SnapPrefix(const Fixture& fx, sslm_kv_pool pool, sslm_prefix p) {
	PageSnap s;
	uint32_t mapped = 0, page = 0;
	if (sslm_pkv_test_only_prefix_table_entry(p, 0, &page, &mapped) != SSLM_OK) mapped = 0;
	for (uint32_t i = 0; i < mapped; ++i) {
		PKV_CHECK(sslm_pkv_test_only_prefix_table_entry(p, i, &page, nullptr) == SSLM_OK);
		s.index.push_back(page);
		s.bytes.push_back(PeekPage(fx, pool, page));
	}
	return s;
}
inline std::vector<uint32_t> SeqPages(sslm_seq s, uint32_t count) {
	std::vector<uint32_t> out;
	for (uint32_t i = 0; i < count; ++i) {
		uint32_t page = 0;
		if (sslm_pkv_test_only_seq_table_entry(s, i, &page, nullptr) != SSLM_OK) break;
		out.push_back(page);
	}
	return out;
}
inline uint32_t SeqMapped(sslm_seq s) {
	uint32_t page = 0, mapped = 0;
	return sslm_pkv_test_only_seq_table_entry(s, 0, &page, &mapped) == SSLM_OK ? mapped : 0;
}

// Byte offset of (layer, half, head, pos % B, 0) inside a page (§3.1).
inline size_t InPage(const Fixture& fx, uint32_t layer, uint32_t half, uint32_t head, int64_t pos) {
	const size_t B = static_cast<size_t>(fx.B()), D = fx.geo.head_dim;
	return ((size_t{layer} * 2 + half) * fx.geo.kv_heads + head) * B * D + static_cast<size_t>(pos % fx.B()) * D;
}

// The canonical SSB6 row (layer, half, head, pos) of a blob: [layer][K|V][head][pos < L'][d] (§3.7).
inline const uint8_t* Ssb6Row(const std::vector<uint8_t>& blob, const Fixture& fx, uint32_t layer, uint32_t half,
                              uint32_t head, int64_t pos) {
	const BlobView v = ParseBlobWithHidden(blob, HiddenSize(fx));
	const size_t D = fx.geo.head_dim, Lp = static_cast<size_t>(v.kv_positions);
	return blob.data() + v.kv_offset + ((size_t{layer} * 2 + half) * fx.geo.kv_heads + head) * Lp * D +
	       static_cast<size_t>(pos) * D;
}

// Does the prefix's span [0, floor(origin/B)*B) equal the blob's rows there, every layer, K and V?
// The construction check of 9.10's cases (which conditions a handle meets), read through the peek.
// `differs_at` (optional) receives the first differing position, or -1.
inline bool PrefixSpanEqualsBlob(const Fixture& fx, sslm_kv_pool pool, sslm_prefix p, const std::vector<uint8_t>& blob,
                                 int64_t origin, int64_t* differs_at = nullptr) {
	const int64_t span = SharedPages(fx, origin) * fx.B();
	const PageSnap snap = SnapPrefix(fx, pool, p);
	const size_t D = fx.geo.head_dim;
	if (differs_at) *differs_at = -1;
	if (static_cast<int64_t>(snap.index.size()) < SharedPages(fx, origin)) {
		if (differs_at) *differs_at = static_cast<int64_t>(snap.index.size()) * fx.B();
		return false;
	}
	for (int64_t pos = 0; pos < span; ++pos)
		for (uint32_t l = 0; l < fx.geo.layers; ++l)
			for (uint32_t half = 0; half < 2; ++half)
				for (uint32_t h = 0; h < fx.geo.kv_heads; ++h) {
					const uint8_t* a = snap.bytes[static_cast<size_t>(pos / fx.B())].data() + InPage(fx, l, half, h, pos);
					if (std::memcmp(a, Ssb6Row(blob, fx, l, half, h, pos), D) != 0) {
						if (differs_at) *differs_at = pos;
						return false;
					}
				}
	return true;
}

// The digest of rows [0, L) of a blob of either format (pkv_scenarios.h's position-major order).
inline std::string RowsSha(const std::vector<uint8_t>& blob, const Fixture& fx) {
	const BlobView v = ParseBlobWithHidden(blob, HiddenSize(fx));
	std::vector<uint8_t> rows;
	if (!v.ok || !BlobRows(blob, fx, HiddenSize(fx), v.context_length, &rows)) return "(unreadable)";
	return Sha(rows.data(), rows.size());
}

// ---- rebuildable states for the two-sided fill probe (§8) ---------------------------------------

// A page pool of `pages` and the handles made in it (released before the pool).
struct Rig : ProbeState {
	std::unique_ptr<PagePool> pool;
	Handles h;
	PersonaChain chain;
	Rig(const Fixture& fx, uint32_t pages, uint8_t fill = 0) : pool(new PagePool(fx.model, pages, fill)) {
		PKV_CHECK_MSG(pool->status == SSLM_OK, "page pool of %u pages: status %d", pages, static_cast<int>(pool->status));
	}
	~Rig() override {
		h.ReleaseAll();
		chain.Release();
	}
	sslm_kv_pool* Pool() override { return &pool->pool; }
	sslm_seq Seq(sslm_seq s) {
		if (s) h.seqs.push_back(s);
		return s;
	}
	sslm_prefix Prefix(sslm_prefix p) {
		if (p) h.prefixes.push_back(p);
		return p;
	}
	// Releases one tracked sequence now (and forgets it).
	void ReleaseSeq(sslm_seq s) {
		auto it = std::find(h.seqs.begin(), h.seqs.end(), s);
		if (it != h.seqs.end()) h.seqs.erase(it);
		sslm_seq_release(s);
	}
	void ReleasePrefix(sslm_prefix p) {
		auto it = std::find(h.prefixes.begin(), h.prefixes.end(), p);
		if (it != h.prefixes.end()) h.prefixes.erase(it);
		sslm_prefix_release(p);
	}
};

// A 2-page create (the smallest the probe makes): admitted or not, released again.
inline bool TwoPageCreateAdmitted(const Fixture& fx, sslm_kv_pool* pool) {
	sslm_seq s = nullptr;
	const sslm_status st = sslm_seq_create_budgeted(fx.model, pool, static_cast<int32_t>(fx.B()), &s);
	if (s) sslm_seq_release(s);
	return st == SSLM_OK;
}

// Creates totalling exactly `k` pages admitted (and released again): the probe's admission leg alone,
// for a construction that only needs "at least k free".
inline bool AdmitsPages(const Fixture& fx, sslm_kv_pool* pool, int64_t k) {
	std::vector<sslm_seq> made;
	const bool all = AdmitPages(fx, pool, k, &made);
	for (auto it = made.rbegin(); it != made.rend(); ++it) sslm_seq_release(*it);
	return all;
}

// ---- real artifacts (box) ---------------------------------------------------------------------

// A real artifact read from SUPERSLM_PAGED_KV_REAL_ARTIFACT_DIR/<file>; a missing variable or file is a
// failed check. `geo` is the artifact's geometry as the cell states it; the cell checks it against
// sslm_kv_block_size (§3.1's formula) before using it.
inline const Fixture& GetArtifact(const char* file, Geometry geo) {
	static std::map<std::string, std::unique_ptr<Fixture>> cache;
	auto it = cache.find(file);
	if (it != cache.end()) {
		PKV_CHECK_MSG(it->second->ok, "artifact %s did not load (see its first failure)", file);
		return *it->second;
	}
	auto f = std::make_unique<Fixture>();
	std::string stem = file;
	if (stem.size() > 5 && stem.compare(stem.size() - 5, 5, ".sslm") == 0) stem.resize(stem.size() - 5);
	f->stem = stem;
	f->geo = geo;
	const char* dir = std::getenv("SUPERSLM_PAGED_KV_REAL_ARTIFACT_DIR");
	PKV_CHECK_MSG(dir && *dir, "SUPERSLM_PAGED_KV_REAL_ARTIFACT_DIR is not set (box cell: it names the real artifacts)");
	std::vector<uint8_t> raw;
	if (dir && *dir && ReadAll(std::string(dir) + "/" + file, &raw) && !raw.empty()) {
		f->size = raw.size();
		f->bytes.Reset(raw.size());
		std::memcpy(f->bytes.p, raw.data(), raw.size());
		f->sha = Sha(raw.data(), raw.size());
		const sslm_status st = sslm_model_map(f->bytes.p, f->size, &f->model);
		PKV_CHECK_MSG(st == SSLM_OK, "sslm_model_map(%s) == %d", file, static_cast<int>(st));
		f->ok = st == SSLM_OK;
		if (f->ok) {
			const size_t block = sslm_kv_block_size(f->model);
			PKV_CHECK_MSG(block == static_cast<size_t>(geo.context_cap) * f->BytesPerToken(),
			              "artifact %s: kv block %zu does not match the stated geometry", file, block);
		}
	} else if (dir && *dir) {
		PKV_CHECK_MSG(false, "artifact %s/%s not found or empty", dir, file);
	}
	const Fixture& ref = *f;
	cache[file] = std::move(f);
	return ref;
}

}  // namespace bc
}  // namespace pkv

#endif  // SUPERSLM_TESTS_PKV_BUDGET_C_HELPERS_H

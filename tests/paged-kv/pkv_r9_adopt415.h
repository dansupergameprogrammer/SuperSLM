// Paged-KV plan rev 16.23 cell 4.15 (D-SLM8434): the construction shared by 4.15/T3 (tests/paged-kv/
// r9_t3_contiguous.cpp) and its 0.5B member 4.15/box (tests/paged-kv/r9box_real.cpp).

#ifndef SUPERSLM_TESTS_PKV_R9_ADOPT415_H
#define SUPERSLM_TESTS_PKV_R9_ADOPT415_H

#include "pkv_r9.h"

#include <memory>

namespace pkv {
namespace r9 {
namespace c415 {

// ---- 4.15 [C4, at T3]: copy adopt across page geometries (rev 16.23, D-SLM8434; T-3192's red candidate) ----------
//
// One model, a page pool and a contiguous pool. A whole_reserve (legacy) sequence copy-adopts a frozen legacy prefix
// of the other pool kind: refused SSLM_INVALID_ARGUMENT in both directions, with the sequence's re-saved blob, both
// pools' raw pages (every page, through the peek) and both pools' admission counts unchanged, and the sequence's bind
// permission kept. Admitted: the same adopts between two pools of one kind (rows [0, len) equal the prefix's), the
// cross-kind pair at a cap where the rule's B is the cap (one geometry), and, after a refusal, an adopt from the
// sequence's own pool. Precedence: a cross-geometry prefix bound to a schema with progress is refused
// SSLM_INVALID_ARGUMENT, not SSLM_PREFIX_SCHEMA_MISMATCH, and the bind permission is kept.
// Mutants: the check removed (SSLM_OK in both directions); the check after bind_eligible is cleared (the bind
// assertion); the check on pool kind rather than page_positions (the one-geometry pair is refused).

struct KindPools {
	Lay y;
	std::unique_ptr<BlockPoolOf> page, page2;
	std::unique_ptr<ContigPoolOf> contig, contig2;
	KindPools(const Fixture& fx, int64_t B, uint32_t blocks) : y(LayoutOf(fx.geo, 1, B)) {
		page = std::make_unique<BlockPoolOf>(fx.model, blocks, y);
		page2 = std::make_unique<BlockPoolOf>(fx.model, blocks, y);
		contig = std::make_unique<ContigPoolOf>(fx.model, blocks, ContigBytesOf(y, blocks));
		contig2 = std::make_unique<ContigPoolOf>(fx.model, blocks, ContigBytesOf(y, blocks));
	}
	static size_t ContigBytesOf(const Lay& y, uint32_t n) {
		bool fits = false;
		return static_cast<size_t>(ContiguousSize(y, n, &fits));
	}
	bool Ok() const { return page->pool && page2->pool && contig->pool && contig2->pool; }
};

// Every page of a pool through the page-index peek: `pages` pages of `page_bytes`.
inline std::vector<std::vector<uint8_t>> SnapPool(sslm_kv_pool pool, uint32_t pages, size_t page_bytes) {
	std::vector<std::vector<uint8_t>> out;
	for (uint32_t i = 0; i < pages; ++i) out.push_back(legacy_a::PeekPage(pool, i, page_bytes));
	return out;
}

inline std::vector<uint32_t> PrefixTable(sslm_prefix p) {
	std::vector<uint32_t> t;
	uint32_t page = 0, mapped = 0;
	for (uint32_t i = 0; sslm_pkv_test_only_prefix_table_entry(p, i, &page, &mapped) == SSLM_OK && i < mapped; ++i)
		t.push_back(page);
	return t;
}

inline sslm_prefix LegacyPrefixOf(const Fixture& fx, sslm_kv_pool* pool, int32_t len, uint32_t seed = 51) {
	sslm_prefix p = nullptr;
	if (sslm_prefix_begin(fx.model, pool, &p) != SSLM_OK) return nullptr;
	if (PrefixPrefillAll(fx.model, p, Stream(seed, len, fx.vocab), 64) != SSLM_OK || sslm_prefix_freeze(p) != SSLM_OK) {
		sslm_prefix_release(p);
		return nullptr;
	}
	return p;
}

// An admitted copy adopt: SSLM_OK, and the sequence's rows [0, len) read at its pool's page size equal the
// prefix's rows read at the prefix pool's page size.
inline void ExpectAdmitted(const Fixture& fx, sslm_kv_pool seq_pool, int64_t seq_B, sslm_kv_pool pre_pool, int64_t pre_B,
                    sslm_seq s, sslm_prefix p, int32_t len, const std::string& what) {
	const sslm_status st = sslm_seq_adopt_prefix(s, p);
	PKV_CHECK_MSG(st == SSLM_OK, "%s: adopt returned %s, SSLM_OK owed", what.c_str(), AbiStatusName(st));
	if (st != SSLM_OK) return;
	std::vector<uint8_t> got, want;
	PKV_CHECK_MSG(SeqRows(seq_pool, s, fx.geo, 1, seq_B, len, Rot::kRotated, &got) &&
	                  RowsFromTable(pre_pool, PrefixTable(p), fx.geo, 1, pre_B, len, Rot::kRotated, &want) && got == want,
	              "%s: the adopter's rows [0, %d) differ from the prefix's", what.c_str(), len);
}

// A refused cross-kind adopt: SSLM_INVALID_ARGUMENT and nothing changed.
inline void ExpectRefused(const Fixture& fx, KindPools& k, bool seq_in_page, int32_t len, bool prefix_last, int64_t B,
                   sslm_schema schema, const std::string& what) {
	sslm_kv_pool* seq_pool = seq_in_page ? &k.page->pool : &k.contig->pool;
	sslm_kv_pool* pre_pool = seq_in_page ? &k.contig->pool : &k.page->pool;
	const int64_t seq_B = seq_in_page ? B : fx.geo.context_cap;
	std::vector<sslm_seq> others;
	auto other = [&](sslm_kv_pool* pool) {
		sslm_seq o = nullptr;
		if (sslm_seq_create(fx.model, pool, &o) == SSLM_OK) others.push_back(o);
	};
	if (prefix_last) other(pre_pool);  // the prefix is then its pool's last holder
	sslm_prefix p = LegacyPrefixOf(fx, pre_pool, len);
	PKV_CHECK_MSG(p != nullptr, "%s: the prefix of %d was not made, so the member did not run", what.c_str(), len);
	if (!prefix_last) other(pre_pool);  // the prefix is its pool's first holder
	sslm_seq s = nullptr;
	PKV_CHECK_EQ(sslm_seq_create(fx.model, seq_pool, &s), SSLM_OK);
	if (p && s) {
		const uint32_t page_pages = static_cast<uint32_t>(3 * k.y.pages), contig_pages = 3;
		const auto page_before = SnapPool(k.page->pool, page_pages, k.y.page_bytes);
		const auto contig_before = SnapPool(k.contig->pool, contig_pages, k.y.block_bytes);
		const std::vector<uint8_t> blob_before = bc::Save(s);
		// the admission count on both pools before (creates that are released write nothing a peek reads)
		sslm_status r0 = SSLM_OK;
		const int page_admits = CountCreates(fx.model, &k.page->pool, 0, &r0);
		const int contig_admits = CountCreates(fx.model, &k.contig->pool, 0, &r0);
		const sslm_status st = sslm_seq_adopt_prefix(s, p);
		PKV_CHECK_MSG(st == SSLM_INVALID_ARGUMENT, "%s: adopt returned %s, SSLM_INVALID_ARGUMENT owed", what.c_str(),
		              AbiStatusName(st));
		PKV_CHECK_MSG(bc::Save(s) == blob_before, "%s: the refused adopt changed the sequence", what.c_str());
		PKV_CHECK_MSG(SnapPool(k.page->pool, page_pages, k.y.page_bytes) == page_before &&
		                  SnapPool(k.contig->pool, contig_pages, k.y.block_bytes) == contig_before,
		              "%s: a refused adopt changed a pool's pages", what.c_str());
		sslm_status r1 = SSLM_OK;
		PKV_CHECK_MSG(CountCreates(fx.model, &k.page->pool, 0, &r1) == page_admits &&
		                  CountCreates(fx.model, &k.contig->pool, 0, &r1) == contig_admits,
		              "%s: the admission count of a pool moved", what.c_str());
		PKV_CHECK_MSG(schema && sslm_seq_set_schema(s, schema) == SSLM_OK, "%s: the bind permission was revoked%s",
		              what.c_str(), schema ? "" : " (no schema to bind: the member did not run)");
		// after the refusal the sequence adopts a prefix of its own pool, rows equal
		sslm_prefix own = LegacyPrefixOf(fx, seq_pool, std::min<int32_t>(len, 64), 53);
		PKV_CHECK_MSG(own != nullptr, "%s: its own pool's prefix was not made", what.c_str());
		if (own) {
			ExpectAdmitted(fx, *seq_pool, seq_B, *seq_pool, seq_B, s, own, std::min<int32_t>(len, 64), what + ", then its own pool's prefix");
			sslm_prefix_release(own);
		}
	}
	if (s) sslm_seq_release(s);
	for (sslm_seq o : others) sslm_seq_release(o);
	if (p) sslm_prefix_release(p);
}

// `schema_name` is a schema the artifact carries: the bench fixtures' g5_minimal_one_field, the 0.5B artifact's
// potion_shop_order (its one SCM1 schema; T-3212 C-1). A failed lookup is a failed check (T0 F-11).
inline void Cell415On(const Fixture& fx, int64_t B, int32_t page_from_contig_len_base, int32_t contig_from_page_len,
                      const char* schema_name) {
	if (!fx.ok) return;
	PKV_CHECK_MSG(sslm_kv_page_positions(fx.model) == B, "%s runs at B = %lld, the rule's %lld owed", fx.stem.c_str(),
	              static_cast<long long>(sslm_kv_page_positions(fx.model)), static_cast<long long>(B));
	sslm_schema schema = nullptr;
	const sslm_status ls = sslm_schema_lookup(fx.model, schema_name, &schema);
	PKV_CHECK_MSG(ls == SSLM_OK && schema, "%s: the lookup of schema %s returned %s; the bind and precedence members need it",
	              fx.stem.c_str(), schema_name, AbiStatusName(ls));
	const int64_t cap = fx.geo.context_cap;
	// Both directions refused.
	for (int32_t len : {page_from_contig_len_base, page_from_contig_len_base + 1}) {
		KindPools k(fx, B, 3);
		PKV_CHECK_MSG(k.Ok(), "%s: the pools (contiguous: %s)", fx.stem.c_str(), AbiStatusName(k.contig->status));
		if (k.Ok()) ExpectRefused(fx, k, true, len, false, B, schema, fx.stem + ": page from contiguous, len " + std::to_string(len));
	}
	for (bool last : {false, true}) {
		KindPools k(fx, B, 3);
		PKV_CHECK_MSG(k.Ok(), "%s: the pools (contiguous from page)", fx.stem.c_str());
		if (k.Ok())
			ExpectRefused(fx, k, false, contig_from_page_len, last, B, schema,
			              fx.stem + ": contiguous from page, the prefix its pool's " + (last ? "last" : "first") + " holder");
	}
	// Admitted: the same adopts within one kind.
	for (int32_t len : {page_from_contig_len_base, page_from_contig_len_base + 1, contig_from_page_len}) {
		KindPools k(fx, B, 3);
		PKV_CHECK_MSG(k.Ok(), "%s: the control pools", fx.stem.c_str());
		if (!k.Ok()) continue;
		for (int kind = 0; kind < 2; ++kind) {
			sslm_kv_pool* sp = kind == 0 ? &k.page->pool : &k.contig->pool;
			sslm_kv_pool* pp = kind == 0 ? &k.page2->pool : &k.contig2->pool;
			const int64_t b = kind == 0 ? B : cap;
			sslm_prefix p = LegacyPrefixOf(fx, pp, len);
			sslm_seq s = nullptr;
			const bool made = p && sslm_seq_create(fx.model, sp, &s) == SSLM_OK;
			PKV_CHECK_MSG(made, "%s: the within-kind control (kind %d, len %d) was not made", fx.stem.c_str(), kind, len);
			if (made)
				ExpectAdmitted(fx, *sp, b, *pp, b, s, p, len,
				               fx.stem + (kind == 0 ? ": page from page" : ": contiguous from contiguous") + ", len " + std::to_string(len));
			if (s) sslm_seq_release(s);
			if (p) sslm_prefix_release(p);
		}
	}
	// Precedence over A18: a cross-geometry prefix with schema progress, an unbound sequence.
	if (schema) {
		KindPools k(fx, B, 3);
		int32_t admitted = -1;
		{
			BlockPoolOf one(fx.model, 1, k.y);
			sslm_seq d = nullptr;
			if (one.pool && sslm_seq_create(fx.model, &one.pool, &d) == SSLM_OK) {
				if (sslm_seq_set_schema(d, schema) == SSLM_OK && PrefillAll(fx.model, d, Stream(72, 20, fx.vocab), 8) == SSLM_OK)
					admitted = bc::Token(fx.model, d);
				sslm_seq_release(d);
			}
		}
		sslm_prefix p = nullptr;
		sslm_seq s = nullptr;
		if (k.Ok() && admitted >= 0 && sslm_prefix_begin(fx.model, &k.contig->pool, &p) == SSLM_OK) {
			int32_t c = 0;
			PKV_CHECK_EQ(sslm_prefix_set_schema(p, schema), SSLM_OK);
			PKV_CHECK_EQ(PrefixPrefillAll(fx.model, p, Stream(84, 15, fx.vocab), 8), SSLM_OK);
			PKV_CHECK(sslm_prefix_prefill(fx.model, p, &admitted, 1, 8, SSLM_SPAN_SCHEMA_CONTENT, nullptr, &c) == SSLM_OK && c == 1);
			PKV_CHECK_EQ(sslm_prefix_freeze(p), SSLM_OK);
			PKV_CHECK_EQ(sslm_seq_create(fx.model, &k.page->pool, &s), SSLM_OK);
			if (s) {
				const sslm_status st = sslm_seq_adopt_prefix(s, p);
				PKV_CHECK_MSG(st == SSLM_INVALID_ARGUMENT, "%s precedence: %s, SSLM_INVALID_ARGUMENT (not the schema mismatch) owed",
				              fx.stem.c_str(), AbiStatusName(st));
				PKV_CHECK_MSG(sslm_seq_set_schema(s, schema) == SSLM_OK, "%s precedence: the bind permission was revoked",
				              fx.stem.c_str());
			}
		} else {
			PKV_CHECK_MSG(false, "%s precedence: construction (pools %d, admitted token %d)", fx.stem.c_str(), k.Ok() ? 1 : 0, admitted);
		}
		if (s) sslm_seq_release(s);
		if (p) sslm_prefix_release(p);
	}
}

}  // namespace c415
}  // namespace r9
}  // namespace pkv

#endif  // SUPERSLM_TESTS_PKV_R9_ADOPT415_H

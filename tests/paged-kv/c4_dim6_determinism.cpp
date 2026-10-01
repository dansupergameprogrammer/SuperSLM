// Paged-KV plan (rev 16.1) §7 dimension 6, the legacy holders' cells (step C4): 6.1's legacy-ABI
// part and 6.4a.
//
// A legacy holder on the paged build is a whole_reserve holder whose rows live in pool pages
// (§3.4, §3.5), and whose save is 1.9.0's SSB5 gathered from those pages with the zero
// substitution of §3.7. Its tokens, its K/V bytes and its blob are v1.11.0's (the reference that
// replaces the plan's "v1.9.0", R0 re-anchor note). Each verdict here reads the R0 reference, a
// fresh twin on the build under test, raw page bytes through the C4 peek seams addressed by the
// §3.1 formula written in the test, or blob bytes; never the stats verbs.

#include "pkv_legacy_b_helpers.h"

namespace {

using namespace pkv;
using namespace pkv::legacy_b;

// ---- 6.1 [C4] ----------------------------------------------------------------------------------
//
// "Tokens and per-position K/V bytes equal the reference": the legacy ABI, for every cell of 4.1
// (widths 1, B-1, B, B+1, 2B, prefix + 64, cap-1, cap) and 4.2 (prefix lengths 0, 1, B-1, B, B+1,
// 1,000 and 1,008), plus the long prefill whose 64-token chunks cross a page boundary each. Two
// halves, each graded against R0's records:
//   (a) the ABI half: the scenarios run through the legacy verbs on the build under test and
//       equal v1.11.0 record for record -- tokens, context length, the rows read out of the blob
//       by the flat formula, and the whole SSB5 blob (a legacy holder's save is byte-equal to
//       1.9.0's, §3.7). On pkv_def (B = 16), pkv_qk (the direct_qk score branch run per page,
//       4.6) and pkv_odd (B = cap, the one-page fallback of §3.1);
//   (b) the layout half (§8, "layout pinning independent of the code under test"): the same
//       states rebuilt live, their rows read raw out of the pool through the table-entry seam and
//       the page-index peek, addressed by §3.1's formula written in pkv_legacy_b_helpers.h, and
//       digested in the reference's own row order. The digest equals the record's rows_sha, so
//       the bytes sit where §3.1 says, not only where the save path finds them.
// Mutants killed: any per-page attention or landing change that moves a token or a row (a);
// a page layout other than §3.1's, a table entry pointing at the wrong page, or a copy adopt that
// lands the prefix rows at other offsets (b) -- each digests differently from the reference.

const char* const kWidthStems[] = {"pkv_def", "pkv_qk", "pkv_odd"};

void LayoutHalfWidths(const Fixture& fx, const RefFile& ref) {
	// 4.1's widths at B = 16, as the widths scenario records them (on pkv_odd, B = cap, the same
	// widths all sit in its one page).
	const std::vector<int64_t> widths = {1, 15, 16, 17, 32, 1064, fx.geo.context_cap - 1, fx.geo.context_cap};
	for (int64_t w : widths) {
		const std::string tag = "w" + std::to_string(w);
		const RefRecord* pre = RefLookup(ref, "widths", tag + "_prefill");
		if (!pre) continue;
		LegacyPool pool(fx.model, 1);
		sslm_seq s = nullptr;
		PKV_CHECK_EQ(sslm_seq_create(fx.model, &pool.pool, &s), SSLM_OK);
		if (!s) continue;
		PKV_CHECK_EQ(PrefillAll(fx.model, s, Stream(41, static_cast<int32_t>(w), fx.vocab), 64), SSLM_OK);
		std::vector<uint8_t> rows;
		PKV_CHECK_MSG(PeekRows(fx, pool.pool, s, w, &rows), "%s width %lld: peek", fx.stem.c_str(), static_cast<long long>(w));
		// kills: a page layout other than §3.1's, or a table entry naming the wrong page
		PKV_CHECK_MSG(Sha(rows.data(), rows.size()) == pre->rows_sha, "%s width %lld: peeked rows differ from v1.11.0",
		              fx.stem.c_str(), static_cast<long long>(w));
		const int32_t ready = NextToken(fx.model, s);
		const RefRecord* rdy = RefLookup(ref, "widths", tag + "_ready");
		PKV_CHECK_MSG(rdy && rdy->tokens.size() == 1 && rdy->tokens[0] == ready, "%s width %lld: ready token",
		              fx.stem.c_str(), static_cast<long long>(w));
		if (w < fx.geo.context_cap) {
			const int32_t t = NextToken(fx.model, s);
			const RefRecord* d1 = RefLookup(ref, "widths", tag + "_decode1");
			PKV_CHECK_MSG(d1 && d1->tokens.size() == 1 && d1->tokens[0] == t, "%s width %lld: decode1 token",
			              fx.stem.c_str(), static_cast<long long>(w));
			// row w is the first row of a fresh page whenever w % B == 0: the map at a page boundary
			PKV_CHECK_MSG(PeekRows(fx, pool.pool, s, w + 1, &rows), "%s width %lld: peek after decode",
			              fx.stem.c_str(), static_cast<long long>(w));
			PKV_CHECK_MSG(d1 && Sha(rows.data(), rows.size()) == d1->rows_sha,
			              "%s width %lld: peeked rows after decode1 differ from v1.11.0", fx.stem.c_str(),
			              static_cast<long long>(w));
		}
		sslm_seq_release(s);
	}
}

void LayoutHalfPrefixLengths(const Fixture& fx, const RefFile& ref) {
	for (int32_t len : {0, 1, 15, 16, 17, 1000, 1008}) {
		const std::string tag = "p" + std::to_string(len);
		LegacyPool pool(fx.model, 2);
		sslm_prefix px = nullptr;
		sslm_seq s = nullptr;
		PKV_CHECK_EQ(sslm_prefix_begin(fx.model, &pool.pool, &px), SSLM_OK);
		if (len > 0) PKV_CHECK_EQ(PrefixPrefillAll(fx.model, px, Stream(51, len, fx.vocab), 64), SSLM_OK);
		PKV_CHECK_EQ(sslm_prefix_freeze(px), SSLM_OK);
		PKV_CHECK_EQ(sslm_seq_create(fx.model, &pool.pool, &s), SSLM_OK);
		PKV_CHECK_EQ(sslm_seq_adopt_prefix(s, px), SSLM_OK);
		if (!s || !px) {
			if (s) sslm_seq_release(s);
			if (px) sslm_prefix_release(px);
			continue;
		}
		std::vector<uint8_t> rows;
		const RefRecord* ad = RefLookup(ref, "prefix_lengths", tag + "_adopted");
		// kills: a copy adopt that lands the prefix rows anywhere but §3.1's offsets in the
		// adopter's own pages
		PKV_CHECK_MSG(PeekRows(fx, pool.pool, s, len, &rows) && ad && Sha(rows.data(), rows.size()) == ad->rows_sha,
		              "%s prefix %d: adopted rows through the peek differ from v1.11.0", fx.stem.c_str(), len);
		// The prefix is released now: a copy adopter references no prefix page (§3.5), so its rows
		// read back unchanged after the prefix's pages are freed and poisoned.
		sslm_prefix_release(px);
		px = nullptr;
		if (len > 0) {
			const std::vector<int32_t> toks = NextTokens(fx.model, s, 8);
			const RefRecord* d8 = RefLookup(ref, "prefix_lengths", tag + "_decode8");
			PKV_CHECK_MSG(d8 && toks == d8->tokens, "%s prefix %d: decode8 tokens", fx.stem.c_str(), len);
			PKV_CHECK_MSG(d8 && PeekRows(fx, pool.pool, s, d8->context_length, &rows) &&
			                  Sha(rows.data(), rows.size()) == d8->rows_sha,
			              "%s prefix %d: rows after decode8 differ from v1.11.0", fx.stem.c_str(), len);
		}
		PKV_CHECK_EQ(PrefillAll(fx.model, s, Stream(52, 20, fx.vocab), 64), SSLM_OK);
		const std::vector<int32_t> toks = NextTokens(fx.model, s, 4);
		const RefRecord* last = RefLookup(ref, "prefix_lengths", tag + "_prefill20+decode4");
		PKV_CHECK_MSG(last && toks == last->tokens, "%s prefix %d: prefill20+decode4 tokens", fx.stem.c_str(), len);
		PKV_CHECK_MSG(last && PeekRows(fx, pool.pool, s, last->context_length, &rows) &&
		                  Sha(rows.data(), rows.size()) == last->rows_sha,
		              "%s prefix %d: rows after prefill20+decode4 differ from v1.11.0", fx.stem.c_str(), len);
		sslm_seq_release(s);
	}
}

void LayoutHalfLongPrefill(const Fixture& fx, const RefFile& ref) {
	LegacyPool pool(fx.model, 1);
	sslm_seq s = nullptr;
	PKV_CHECK_EQ(sslm_seq_create(fx.model, &pool.pool, &s), SSLM_OK);
	if (!s) return;
	PKV_CHECK_EQ(PrefillAll(fx.model, s, Stream(11, 1500, fx.vocab), 64), SSLM_OK);
	std::vector<uint8_t> rows;
	const RefRecord* a = RefLookup(ref, "long_prefill", "prefill1500");
	PKV_CHECK_MSG(a && PeekRows(fx, pool.pool, s, 1500, &rows) && Sha(rows.data(), rows.size()) == a->rows_sha,
	              "%s long prefill: rows through the peek differ from v1.11.0", fx.stem.c_str());
	const std::vector<int32_t> toks = NextTokens(fx.model, s, 16);
	const RefRecord* b = RefLookup(ref, "long_prefill", "prefill1500+decode16");
	PKV_CHECK_MSG(b && toks == b->tokens, "%s long prefill: decode16 tokens", fx.stem.c_str());
	PKV_CHECK_MSG(b && PeekRows(fx, pool.pool, s, b->context_length, &rows) && Sha(rows.data(), rows.size()) == b->rows_sha,
	              "%s long prefill: rows after decode16 differ from v1.11.0", fx.stem.c_str());
	sslm_seq_release(s);
}

void Cell61Legacy() {
	for (const char* stem : kWidthStems) {
		const Fixture& fx = GetFixture(stem);
		if (!fx.ok) continue;
		// (a) the ABI half, against the reference record for record.
		for (const char* sc : {"widths", "prefix_lengths", "long_prefill"})
			ExpectMatchesReference("v1.11.0", fx, sc, RunScenario(fx, sc), Compare::kTokensRowsAndBlob);
		// (b) the layout half, on B = 16 (pkv_def) and B = cap (pkv_odd: one page per holder, the
		// one-page identity of §3.1). pkv_qk's geometry differs only in head_dim, which (a) covers.
		if (std::strcmp(stem, "pkv_qk") == 0) continue;
		const RefFile& ref = Reference("v1.11.0", fx);
		LayoutHalfWidths(fx, ref);
		LayoutHalfPrefixLengths(fx, ref);
		LayoutHalfLongPrefill(fx, ref);
	}
}

// ---- 6.4a [C4] ---------------------------------------------------------------------------------
//
// The determinism rule, forced (§3.3 "No zero fill anywhere", §3.7 Writers). A sequence is reset
// and reused so that the row at its new context_length still holds the prior generation's K/V in
// its pages (the reset no longer memsets, §3.3 rev 15.2). It is saved mid-token there, so the
// row at context_length is written for layer 0 only, and layers >= layer_index must be zero in
// the blob whatever the page holds; every position past L' must be zero too. Its 1.9.0 SSB5
// equals a fresh holder's, saved at the same state. The fresh twin's pool is pre-filled with
// 0x5A, so a gather that reads memory past the valid rows differs on that side as well. Run at a
// mid-page position (40) and at a page-aligned one (48, whose mid-token row is the first row of
// a page the prior generation filled).
// Mutant killed: "the save gathers the mid-token row (or the rows past L') from memory instead of
// substituting zeros" -- the reused blob then carries the prior generation's row and differs.
// The SSB6 half of the same row's zero substitution runs on a budget holder (carrier row
// RB16-CELLS), at C5.

void Cell64aResetReuseMidToken() {
	const Fixture& fx = GetFixture("pkv_def");
	if (!fx.ok) return;
	const size_t bpt = fx.BytesPerToken();
	for (int32_t pos : {40, 48}) {
		LegacyPool reused_pool(fx.model, 1);
		sslm_seq s = nullptr;
		PKV_CHECK_EQ(sslm_seq_create(fx.model, &reused_pool.pool, &s), SSLM_OK);
		if (!s) continue;
		// The prior generation: 100 positions, so row `pos` and every row up to 99 hold its K/V.
		PKV_CHECK_EQ(PrefillAll(fx.model, s, Stream(61, 100, fx.vocab), 64), SSLM_OK);
		NextTokens(fx.model, s, 2);
		std::vector<uint8_t> g1;
		PKV_CHECK(SaveBlob(s, &g1));
		// The construction separates only if the prior generation's row `pos` is nonzero in the
		// layers the mid-token save must zero (layer 1): checked on its own blob.
		std::vector<uint8_t> g1rows;
		PKV_CHECK(BlobRows(g1, fx, HiddenSize(fx), 100, &g1rows));
		bool stale_nonzero = false;
		if (g1rows.size() >= static_cast<size_t>(pos + 1) * bpt)
			for (size_t i = static_cast<size_t>(pos) * bpt + bpt / 2; i < static_cast<size_t>(pos + 1) * bpt; ++i)
				stale_nonzero = stale_nonzero || g1rows[i] != 0;
		PKV_CHECK_MSG(stale_nonzero, "6.4a construction: the prior generation's row %d layer 1 is all zero", pos);

		// Reset and reuse: a new generation of `pos` tokens, then one layer of the next token.
		PKV_CHECK_EQ(sslm_seq_reset(s), SSLM_OK);
		PKV_CHECK_EQ(PrefillAll(fx.model, s, Stream(62, pos, fx.vocab), 16), SSLM_OK);
		std::vector<int32_t> ready_reused, ready_fresh;
		EnterMidToken(fx.model, s, &ready_reused);
		std::vector<uint8_t> reused;
		PKV_CHECK(SaveBlob(s, &reused));

		// The fresh twin, over a pool pre-filled with 0x5A.
		LegacyPool fresh_pool(fx.model, 1, 0x5A);
		sslm_seq t = nullptr;
		PKV_CHECK_EQ(sslm_seq_create(fx.model, &fresh_pool.pool, &t), SSLM_OK);
		if (!t) {
			sslm_seq_release(s);
			continue;
		}
		PKV_CHECK_EQ(PrefillAll(fx.model, t, Stream(62, pos, fx.vocab), 16), SSLM_OK);
		EnterMidToken(fx.model, t, &ready_fresh);
		std::vector<uint8_t> fresh;
		PKV_CHECK(SaveBlob(t, &fresh));

		PKV_CHECK_MSG(ready_reused == ready_fresh, "6.4a pos %d: ready tokens differ", pos);
		PKV_CHECK_MSG(IsMagic(reused, "SSB5"), "6.4a pos %d: a legacy holder saves 1.9.0's SSB5", pos);
		PKV_CHECK_EQ(Le64(reused, 60), pos);
		PKV_CHECK_EQ(Le32(reused, 68), 1);  // mid-token: layer_index 1, L' = pos + 1
		// kills: the save gathers the mid-token row or the rows past L' from the pages
		PKV_CHECK_MSG(reused == fresh, "6.4a pos %d: the reused holder's SSB5 differs from a fresh holder's", pos);
		// The same fact read off the blob bytes directly: row `pos`, layer 1 (K and V, both heads)
		// and every position past L' are zero in the whole block (§3.7).
		const BlobView v = ParseBlobWithHidden(reused, HiddenSize(fx));
		bool zeros = v.ok;
		if (v.ok) {
			const size_t cap = static_cast<size_t>(fx.geo.context_cap), D = fx.geo.head_dim;
			for (uint32_t l = 0; l < fx.geo.layers && zeros; ++l)
				for (uint32_t half = 0; half < 2 && zeros; ++half)
					for (uint32_t h = 0; h < fx.geo.kv_heads && zeros; ++h) {
						const uint8_t* base = reused.data() + v.kv_offset + ((size_t{l} * 2 + half) * fx.geo.kv_heads + h) * cap * D;
						const size_t from = static_cast<size_t>(l >= 1 ? pos : pos + 1) * D;
						for (size_t i = from; i < cap * D && zeros; ++i) zeros = base[i] == 0;
					}
		}
		PKV_CHECK_MSG(zeros, "6.4a pos %d: the mid-token row's layer 1, or a row past L', is not zero in the blob", pos);
		// Both finish the token and continue alike.
		PKV_CHECK_MSG(NextTokens(fx.model, s, 4) == NextTokens(fx.model, t, 4), "6.4a pos %d: continuations differ", pos);
		sslm_seq_release(t);
		sslm_seq_release(s);
	}
}

PKV_CELL("6.1/C4", "C4", Cell61Legacy);
PKV_CELL("6.4a/C4", "C4", Cell64aResetReuseMidToken);

}  // namespace

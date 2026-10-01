// Paged-KV plan (rev 16.1) §7 dimension 1 (lifetime and reuse), the parts step C4 owns: legacy
// holders (no-budget verbs, whole_reserve mode) on the paged build.
//
//   1.1/C4   the lifecycle scenario's tokens and 1.9.0 SSB5 blobs equal v1.11.0's, stage by stage
//   1.2      the leak check: a written page reads 0xCD through the page-index peek after release
//   1.6/C4   dirty accounting: written pages poisoned, clean reserve pages untouched (0x5A)
//   1.13/C4  a legacy prefix's freeze returns its reserve once, graded by the legacy-create count
//
// Grading (§7 verdict provenance, §8): byte-equality against the R0 reference; page bytes through
// the C4 peek seam, located through the table-entry seam; page counts by the legacy-create
// admission count. No cell reads the stats verbs.

#include "pkv_legacy_a_helpers.h"

namespace {

using namespace pkv;
using namespace pkv::legacy_a;

// 1.1 [C4 legacy]. One handle driven fresh -> prefill 100 -> reset -> prefill 40 -> reset -> adopt a
// 300-token prefix -> decode, crossing 16-position page boundaries at every stage (B = 16 on
// pkv_def): R0's `lifecycle` scenario, run on this build and compared record for record with
// v1.11.0, whole blob included (a legacy holder writes 1.9.0's SSB5, §3.7). The reference is the
// v1.11.0 tag's own run, never the flat path through the new code (§8).
void Cell11Legacy() {
	const Fixture& fx = GetFixture("pkv_def");
	if (!fx.ok) return;
	// kills: any stage whose tokens, K/V rows or saved bytes differ from 1.9.0's, e.g. a reset that
	// leaves pages mapped (the 40-token generation reads stale rows), an adopt that copies the tail
	// page short, or a save gather that reads past `mapped` or skips the zero substitution
	ExpectMatchesReference("v1.11.0", fx, "lifecycle", RunScenario(fx, "lifecycle"), Compare::kTokensRowsAndBlob);
}

// 1.2 [C4]. Port of the S4 pin (G19; tools/t2132_s4_leak_guard_mutation_pin.cpp) from blocks to
// pages: a one-block pool, one sequence writes rows 0-4 (page table entry 0), the page holds real
// bytes before release, and after release the page-index peek reads 0xCD over the whole page.
void Cell12() {
	const Fixture& fx = GetFixture("pkv_def");
	if (!fx.ok) return;
	LegacyPool pool(fx.model, 1);
	PKV_CHECK_EQ(pool.status, SSLM_OK);
	if (!pool.pool) return;
	sslm_seq s = nullptr;
	PKV_CHECK_EQ(sslm_seq_create(fx.model, &pool.pool, &s), SSLM_OK);
	if (!s) return;
	PKV_CHECK_EQ(PrefillAll(fx.model, s, Stream(71, 5, fx.vocab), 5), SSLM_OK);
	uint32_t page = 0, mapped = 0;
	PKV_CHECK_EQ(sslm_pkv_test_only_seq_table_entry(s, 0, &page, &mapped), SSLM_OK);
	PKV_CHECK_MSG(mapped >= 1, "a 5-row sequence maps its first page (mapped %u)", mapped);
	const std::vector<uint8_t> before = PeekPage(pool.pool, page, fx.PageBytes());
	PKV_CHECK_MSG(!before.empty() && !AllBytesAre(before, 0xCD),
	              "page %u holds the written rows before release (not already poison)", page);
	PKV_CHECK_EQ(sslm_seq_release(s), SSLM_OK);
	const std::vector<uint8_t> after = PeekPage(pool.pool, page, fx.PageBytes());
	// kills: the poison fill removed from the page free path (the page keeps the released
	// holder's K/V, which the next holder to draw it could observe)
	PKV_CHECK_MSG(AllBytesAre(after, 0xCD), "page %u reads 0xCD over all %zu bytes after release", page,
	              fx.PageBytes());
}

// 1.6 [C4]. Dirty accounting. The caller-owned pool buffer is filled with 0x5A before
// sslm_kv_pool_create, so every page nobody wrote reads 0x5A (no zero fill anywhere, §3.3). One
// legacy sequence (a one-block pool: its 256-page reservation is the whole pool) writes positions
// [0, 113): prefill 100 maps pages 0-6 and decode maps page 7 through the decode path. Its table is
// read through the seam, then it is reset and released. Every written page reads 0xCD; every other
// page (the 248 clean reserve pages) still reads 0x5A. Rev 2's "exactly once" is dropped (§7).
void Cell16Legacy() {
	const Fixture& fx = GetFixture("pkv_def");
	if (!fx.ok) return;
	LegacyPool pool(fx.model, 1, 0x5A);
	PKV_CHECK_EQ(pool.status, SSLM_OK);
	if (!pool.pool) return;
	sslm_seq s = nullptr;
	PKV_CHECK_EQ(sslm_seq_create(fx.model, &pool.pool, &s), SSLM_OK);
	if (!s) return;
	PKV_CHECK_EQ(PrefillAll(fx.model, s, Stream(72, 100, fx.vocab), 64), SSLM_OK);
	for (int i = 0; i < 14; ++i) PKV_CHECK(NextToken(fx.model, s) >= 0);  // the ready token, then rows 100-112
	std::vector<uint32_t> table;
	PKV_CHECK(SeqTable(s, &table));
	// §3.4 written in the test: rows [0, 113) span ceil(113 / B) pages.
	const int64_t written_pages = (113 + fx.B() - 1) / fx.B();
	PKV_CHECK_MSG(static_cast<int64_t>(table.size()) >= written_pages, "table covers the %lld written pages (%zu)",
	              static_cast<long long>(written_pages), table.size());
	const std::set<uint32_t> written(table.begin(),
	                                 table.begin() + std::min<std::ptrdiff_t>(written_pages, static_cast<std::ptrdiff_t>(table.size())));
	PKV_CHECK_EQ(written.size(), written_pages);
	PKV_CHECK_EQ(sslm_seq_reset(s), SSLM_OK);
	PKV_CHECK_EQ(sslm_seq_release(s), SSLM_OK);
	const int64_t pages = fx.CapPages();  // one block = ceil(cap / B) pages (§3.6)
	int dirty_bad = 0, clean_bad = 0, peek_refused = 0;
	for (int64_t i = 0; i < pages; ++i) {
		const std::vector<uint8_t> b = PeekPage(pool.pool, static_cast<uint32_t>(i), fx.PageBytes());
		if (b.empty()) {
			++peek_refused;
			continue;
		}
		if (written.count(static_cast<uint32_t>(i))) dirty_bad += AllBytesAre(b, 0xCD) ? 0 : 1;
		else clean_bad += AllBytesAre(b, 0x5A) ? 0 : 1;
	}
	PKV_CHECK_EQ(peek_refused, 0);
	// kills: the poison fill skipped (or the dirty flag never set) for a page the holder wrote,
	// including the page decode mapped and the pages reset handed back to the reserve
	PKV_CHECK_MSG(dirty_bad == 0, "%d of %lld written pages do not read 0xCD after reset and release", dirty_bad,
	              static_cast<long long>(written_pages));
	// kills: the release that ignores the dirty flag and poisons every reserve page (a 0x5A page
	// turns 0xCD), and any zero fill at create, draw or reset
	PKV_CHECK_MSG(clean_bad == 0, "%d of %lld clean pages do not read 0x5A after reset and release", clean_bad,
	              static_cast<long long>(pages - written_pages));
}

// 1.13 [C4 legacy prefix]. sslm_prefix_freeze called twice on a legacy prefix (budget = cap) of
// length 1,000 in a 2-block pool (512 pages). §3.4 written in the test: the prefix keeps
// ceil(1000 / B) = 63 pages and the first call returns 256 - 63 = 193, leaving 449 free, which
// admits floor(449 / 256) = 1 legacy create. The named mutant ("freeze returns R(budget) -
// ceil(len/B) pages, computed arithmetically, on every call") returns 193 more on the second call:
// 642 free, 2 creates. Graded by the legacy-create admission count (§8); the earlier mutant
// ("returns the reserve on every call") is equivalent, the reserve being empty after the first
// call.
void Cell113Legacy() {
	const Fixture& fx = GetFixture("pkv_def");
	if (!fx.ok) return;
	const int64_t P = 2 * fx.CapPages();
	const int64_t kept = (1000 + fx.B() - 1) / fx.B();
	const int64_t returned = fx.R(fx.geo.context_cap) - kept;
	PKV_CHECK_EQ(kept, 63);
	PKV_CHECK_EQ(returned, 193);
	const int64_t correct = (P - kept) / fx.CapPages(), mutant = (P - kept + returned) / fx.CapPages();
	PKV_CHECK_EQ(correct, 1);
	PKV_CHECK_EQ(mutant, 2);  // the construction separates the mutant

	LegacyPool pool(fx.model, 2);
	PKV_CHECK_EQ(pool.status, SSLM_OK);
	if (!pool.pool) return;
	sslm_prefix px = nullptr;
	PKV_CHECK_EQ(sslm_prefix_begin(fx.model, &pool.pool, &px), SSLM_OK);
	if (!px) return;
	PKV_CHECK_EQ(PrefixPrefillAll(fx.model, px, Stream(73, 1000, fx.vocab), 64), SSLM_OK);
	sslm_status refusal = SSLM_OK;
	PKV_CHECK_EQ(sslm_prefix_freeze(px), SSLM_OK);
	PKV_CHECK_EQ(CountLegacyCreates(fx.model, &pool.pool, &refusal), correct);
	PKV_CHECK_EQ(refusal, SSLM_KV_POOL_EXHAUSTED);
	PKV_CHECK_EQ(sslm_prefix_freeze(px), SSLM_OK);  // idempotent (G33)
	// kills: freeze returns R(budget) - ceil(len/B) pages on every call (2 creates admitted)
	PKV_CHECK_EQ(CountLegacyCreates(fx.model, &pool.pool, &refusal), correct);
	PKV_CHECK_EQ(refusal, SSLM_KV_POOL_EXHAUSTED);
	PKV_CHECK_EQ(sslm_prefix_release(px), SSLM_OK);
	PKV_CHECK_EQ(CountLegacyCreates(fx.model, &pool.pool), 2);  // the prefix's 63 pages come back
}

PKV_CELL("1.1/C4", "C4", Cell11Legacy);
PKV_CELL("1.2", "C4", Cell12);
PKV_CELL("1.6/C4", "C4", Cell16Legacy);
PKV_CELL("1.13/C4", "C4", Cell113Legacy);

}  // namespace

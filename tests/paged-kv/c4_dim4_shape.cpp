// Paged-KV plan (rev 16.1) §7 dimension 4 (shape and platform matrices), the parts step C4 owns:
// the legacy ABI on pages across widths, prefix lengths, page sizes and geometries.
//
//   4.1/C4  widths 1, B-1, B, B+1, 2B, prefix + 64, cap-1 and cap through the legacy ABI
//   4.2/C4  prefix lengths 0, 1, B-1, B, B+1, 1,000 and 1,008 through legacy prefixes
//   4.4/C4  the B fallback: pkv_odd (cap 4100, B = cap), legacy outputs equal to the reference
//   4.6/C4  pkv_def (plain score branch) and pkv_qk (direct_qk) through the legacy ABI
//   4.9     the B = 1 geometry (G36) through the legacy ABI
//
// A legacy holder's outputs on the paged build are promised equal to v1.11.0's (§3.7), so every
// cell compares R0's scenario records against the v1.11.0 reference: tokens, context length, the
// K/V rows by the flat layout's formula, and the whole 1.9.0 SSB5 blob. 4.4 and 4.9 also read
// the rows out of the pool's pages through the C4 seams, by §3.1's formula written in the test.

#include "pkv_legacy_a_helpers.h"

namespace {

using namespace pkv;
using namespace pkv::legacy_a;

constexpr const char* kAllScenarios[] = {"lifecycle", "long_prefill", "shared_preamble", "persist",
                                         "saturating", "widths", "prefix_lengths"};

// The persist scenario's saved state (prefill 100, decode 4; L = 103) rebuilt on a fresh one-block
// legacy pool of `fx`, its rows read from the pages with page size `B` and compared with the
// reference record `ref_fx`/persist/saved: the decoded tokens and the rows' digest.
void PagesHoldReferenceRows(const Fixture& fx, int64_t B, const Fixture& ref_fx) {
	const RefRecord* saved = RefLookup(Reference("v1.11.0", ref_fx), "persist", "saved");
	if (!saved) return;
	LegacyPool pool(fx.model, 1);
	PKV_CHECK_EQ(pool.status, SSLM_OK);
	if (!pool.pool) return;
	sslm_seq s = nullptr;
	PKV_CHECK_EQ(sslm_seq_create(fx.model, &pool.pool, &s), SSLM_OK);
	if (!s) return;
	PKV_CHECK_EQ(PrefillAll(fx.model, s, Stream(31, 100, fx.vocab), 64), SSLM_OK);
	PKV_CHECK_MSG(NextTokens(fx.model, s, 4) == saved->tokens, "%s: tokens of the saved state", fx.stem.c_str());
	std::vector<uint8_t> rows;
	// kills: a page size other than §3.1's B for this geometry (rows land at another page or
	// another in-page offset), and a table that does not cover the written positions
	PKV_CHECK_MSG(RowsFromPages(pool.pool, s, fx.geo, B, saved->context_length, &rows),
	              "%s: the table covers the %lld written positions at B = %lld", fx.stem.c_str(),
	              static_cast<long long>(saved->context_length), static_cast<long long>(B));
	PKV_CHECK_MSG(Sha(rows.data(), rows.size()) == saved->rows_sha, "%s: rows read from the pages at B = %lld equal v1.11.0's",
	              fx.stem.c_str(), static_cast<long long>(B));
	PKV_CHECK_EQ(sslm_seq_release(s), SSLM_OK);
}

// 4.1 [C4 ABI]. R0's `widths` scenario on pkv_def (B = 16): one fresh legacy sequence per width
// W in {1, 15, 16, 17, 32, 1,064 (a 1,000-token prefix + 64), 4,095, 4,096}, prefilled in
// 64-token chunks, its ready token decoded, and below the cap one more token (row W, attending over
// W + 1 positions), each stage saved and compared whole. Then the cap is the last admissible
// write: a holder at context_length = cap emits its ready token and is refused the next row with
// today's SSLM_CONTEXT_CAP_EXCEEDED (§3.6: limit == cap never reports the budget status, nor a pool
// status from an empty reserve).
void Cell41Legacy() {
	const Fixture& fx = GetFixture("pkv_def");
	if (!fx.ok) return;
	PKV_CHECK_EQ(fx.B(), 16);
	// kills: a page-run loop that drops or repeats the last partial run (W = 15, 17, 1,064), a run
	// boundary off by one at W = B or 2B, and a coverage guard that refuses the last page (W = cap)
	ExpectMatchesReference("v1.11.0", fx, "widths", RunScenario(fx, "widths"), Compare::kTokensRowsAndBlob);

	LegacyPool pool(fx.model, 1);
	PKV_CHECK_EQ(pool.status, SSLM_OK);
	if (!pool.pool) return;
	sslm_seq s = nullptr;
	PKV_CHECK_EQ(sslm_seq_create(fx.model, &pool.pool, &s), SSLM_OK);
	if (!s) return;
	const int32_t cap = static_cast<int32_t>(fx.geo.context_cap);
	PKV_CHECK_EQ(PrefillAll(fx.model, s, Stream(41, cap, fx.vocab), 64), SSLM_OK);
	sslm_status st = SSLM_OK;
	PKV_CHECK(NextToken(fx.model, s, &st) >= 0);  // the ready token writes no row
	PKV_CHECK_EQ(NextToken(fx.model, s, &st), -1);
	// kills: a legacy holder's cap refusal reported as SSLM_KV_POOL_EXHAUSTED (an empty reserve at
	// the cap) or as the budget status
	PKV_CHECK_EQ(st, SSLM_CONTEXT_CAP_EXCEEDED);
	const int32_t one = 0;
	int32_t consumed = -1;
	PKV_CHECK_EQ(sslm_prefill(fx.model, s, &one, 1, 1, SSLM_SPAN_PROMPT, nullptr, &consumed), SSLM_CONTEXT_CAP_EXCEEDED);
	PKV_CHECK_EQ(consumed, 0);
	PKV_CHECK_EQ(sslm_seq_release(s), SSLM_OK);
}

// 4.2 [C4 legacy]. R0's `prefix_lengths` scenario on pkv_def: a legacy prefix of length P in
// {0, 1, 15, 16, 17, 1,000 (mid-page), 1,008 (aligned)} frozen and copy-adopted by a fresh
// legacy sequence, which decodes 8 tokens, then prefills 20 and decodes 4; every stage saved and
// compared whole.
void Cell42Legacy() {
	const Fixture& fx = GetFixture("pkv_def");
	if (!fx.ok) return;
	// kills: a copy adopt that copies floor(P / B) pages (the mid-page tail lost at 1, 15, 17,
	// 1,000), or one page too many past an aligned prefix (16, 1,008), and an empty prefix (0)
	// that maps a page
	ExpectMatchesReference("v1.11.0", fx, "prefix_lengths", RunScenario(fx, "prefix_lengths"),
	                       Compare::kTokensRowsAndBlob);
}

// 4.4 [C4 legacy]. The B fallback (§3.1): pkv_odd is pkv_def re-emitted at cap 4100, which 16 does
// not divide, so B = cap and a legacy holder has one page, today's block. Every R0 scenario's
// legacy outputs equal v1.11.0's, blob included, and the one page holds the rows at B = 4,100 (the
// one-page identity: the paged address formula with B = cap is today's flat layout).
void Cell44Legacy() {
	const Fixture& fx = GetFixture("pkv_odd");
	if (!fx.ok) return;
	PKV_CHECK_EQ(fx.B(), 4100);
	PKV_CHECK_EQ(fx.CapPages(), 1);
	PKV_CHECK_EQ(fx.PageBytes(), sslm_kv_block_size(fx.model));  // one page is one block
	// kills: B = kv_block_size taken without the divisibility test (257 pages a reservation,
	// 256.25 pages a block: the scenarios' pools refuse their creates)
	for (const char* sc : kAllScenarios)
		ExpectMatchesReference("v1.11.0", fx, sc, RunScenario(fx, sc), Compare::kTokensRowsAndBlob);
	PagesHoldReferenceRows(fx, fx.B(), fx);
	LegacyPool pool(fx.model, 3);
	PKV_CHECK_EQ(pool.status, SSLM_OK);
	if (pool.pool) PKV_CHECK_EQ(CountLegacyCreates(fx.model, &pool.pool), 3);  // N blocks admit N handles (§3.5)
}

// 4.6 [C4 ABI]. Geometries through the legacy ABI: pkv_def (H_kv 2, D 48, QK-norm stripped: the
// GemmInt8AccumulateRow score branch) and pkv_qk (H_kv 2, D 128, QK-norm kept: the direct_qk
// QkQ31Score branch), so both score branches (G2) run per page. The box leg adds the 0.5B (D 64)
// and 1.5B (D 128) artifacts; the cloud fixtures keep their page arithmetic (B = 16, 256 pages a
// cap). Every R0 scenario, blob included, against each fixture's own v1.11.0 reference.
void Cell46Legacy() {
	for (const char* stem : {"pkv_def", "pkv_qk"}) {
		const Fixture& fx = GetFixture(stem);
		if (!fx.ok) continue;
		PKV_CHECK_EQ(fx.B(), 16);
		// kills: a per-page score loop wrong on one branch only (e.g. direct_qk scoring each page
		// run from position 0 of the row, or the plain branch striding by cap instead of B)
		for (const char* sc : kAllScenarios)
			ExpectMatchesReference("v1.11.0", fx, sc, RunScenario(fx, sc), Compare::kTokensRowsAndBlob);
	}
}

// 4.9 [C4]. The B = 1 geometry, where every position is a page boundary and R(cap) = min(cap + 1,
// cap) = cap leaves the +1 page of slack unused. G36: test_main.cpp's 21 hand-built fixture specs
// set kv_block_size = 1, and their flat engine paths stay in the existing suites, green at C4; this
// cell names the geometry in the matrix and runs it through the legacy ABI, so a regression there is
// attributed. The model is pkv_def with CFG1's kv_block_size patched to 1 (integrity hash
// recomputed): §3.1 gives B = 1, while the weights, and so the tokens and K/V rows, are pkv_def's.
// Tokens and rows equal pkv_def's v1.11.0 reference (the blobs carry the twin's own model hash, so
// they are compared by rows, not whole), and the rows read from the pages at B = 1 equal them too.
void Cell49() {
	const Fixture& def = GetFixture("pkv_def");
	if (!def.ok) return;
	Fixture twin;
	PKV_CHECK_MSG(PatchedBlockSizeTwin(def, 1, &twin), "pkv_def with kv_block_size 1 maps");
	if (!twin.ok) return;
	const int64_t B = 1;  // §3.1: kv_block_size 1 <= cap and divides it
	PKV_CHECK_EQ(sslm_kv_block_size(twin.model), sslm_kv_block_size(def.model));
	// kills: page arithmetic that assumes B > 1 (a zero-size tail page, a +1 slack page drawn
	// past cap / B, a page run of length 0) anywhere in prefill, decode, reset, copy adopt or save
	for (const char* sc : kAllScenarios)
		ExpectMatchesReference("v1.11.0", def, sc, RunScenario(twin, sc), Compare::kTokensAndRows);
	PagesHoldReferenceRows(twin, B, def);
	// N blocks of the old verb admit N legacy handles: a reservation is cap / 1 = 4,096 pages,
	// exactly one block, with no slack page.
	LegacyPool pool(twin.model, 2);
	PKV_CHECK_EQ(pool.status, SSLM_OK);
	if (pool.pool) PKV_CHECK_EQ(CountLegacyCreates(twin.model, &pool.pool), 2);
}

PKV_CELL("4.1/C4", "C4", Cell41Legacy);
PKV_CELL("4.2/C4", "C4", Cell42Legacy);
PKV_CELL("4.4/C4", "C4", Cell44Legacy);
PKV_CELL("4.6/C4", "C4", Cell46Legacy);
PKV_CELL("4.9", "C4", Cell49);

}  // namespace

// Paged-KV plan rev 16.21 §23.8 step T1, the derived-B crossing:
//
//   4.13/T1:legacy   4.1 and 4.2 (the widths and prefix-lengths scenarios) and 6.1's token and K/V
//                    equality, unpinned: pkv_def at B = 128, pkv_32k at 256, pkv_qk at 64
//   4.13/T1:budget   4.3 unpinned: every origin offset 0..B-1 x budgets {1, B-1, B, B+1, 512}, each
//                    running its full budget (b + 1 tokens) and refused one past; on pkv_def (128)
//                    and pkv_qk (64), and pkv_32k (256)
//   4.13/T1:restore  9.9's and 9.10's populations at the derived B: a mid-page origin restored with
//                    and without its prefix handle (E materialized pages), rows and tokens equal
//   4.6/T1           both score branches per page at the derived B, rotated: pkv_def (plain) and
//                    pkv_qk (direct_qk), every scenario against v1.11.0 (the real-artifact half is
//                    4.6/box in superslm_pkv_r9_box)
//
// The references are the v1.11.0 records, which do not depend on the page layout; on 2ebe5e3 they
// match at B = 16, so every cell here is red on the derived B (and the rotated rows read through the
// peek), not on its tokens. 4.13's forced-SIMD-tier half runs this file in superslm_pkv_r9_<tier>_forced.

#include "pkv_r9.h"

namespace {

using namespace pkv;
using namespace pkv::r9;

struct Member { const char* stem; uint32_t D; int64_t cap; };
const Member kMembers[] = {{"pkv_def", 48, 4096}, {"pkv_32k", 48, 32768}, {"pkv_qk", 128, 4096}};

bool AtDerived(const Fixture& fx, uint32_t D, int64_t* B) {
	*B = RuleB(D, static_cast<uint64_t>(fx.geo.context_cap), 1);
	const int64_t got = sslm_kv_page_positions(fx.model);
	PKV_CHECK_MSG(got == *B, "%s runs at B = %lld; the derived B is %lld", fx.stem.c_str(), static_cast<long long>(got),
	              static_cast<long long>(*B));
	return got == *B;
}

void PersistRowsAtDerived(const Fixture& fx, int64_t B) {
	const RefRecord* saved = RefLookup(Reference("v1.11.0", fx), "persist", "saved");
	if (!saved) return;
	LegacyPool pool(fx.model, 1);
	sslm_seq s = nullptr;
	PKV_CHECK_EQ(sslm_seq_create(fx.model, &pool.pool, &s), SSLM_OK);
	if (!s) return;
	PKV_CHECK_EQ(PrefillAll(fx.model, s, Stream(31, 100, fx.vocab), 64), SSLM_OK);
	PKV_CHECK(bc::DecodeN(fx.model, s, 4) == saved->tokens);
	std::vector<uint8_t> rows;
	PKV_CHECK_MSG(SeqRows(pool.pool, s, fx.geo, 1, B, saved->context_length, Rot::kRotated, &rows) &&
	                  Sha(rows.data(), rows.size()) == saved->rows_sha,
	              "%s: rows read from the pages at the derived B = %lld, rotated, equal v1.11.0's", fx.stem.c_str(),
	              static_cast<long long>(B));
	sslm_seq_release(s);
}

void Cell413Legacy() {
	for (const Member& m : kMembers) {
		const Fixture& fx = GetFixture(m.stem);
		if (!fx.ok) continue;
		int64_t B = 0;
		AtDerived(fx, m.D, &B);
		for (const char* sc : {"widths", "prefix_lengths"})
			ExpectMatchesReference("v1.11.0", fx, sc, RunScenario(fx, sc), Compare::kTokensRowsAndBlob);
		PersistRowsAtDerived(fx, B);
		// 4.1's widths and 4.2's prefix lengths re-chosen at the derived B (the scenarios above are B = 16's), the
		// aligned length 1,024 run, not asserted (T-3196 F-2)
		DerivedBoundaries(fx, B, kBoundaryWidths | kBoundaryPrefixes, "4.13/T1:legacy");
	}
}

void Cell413Budget() {
	for (const Member& m : kMembers) {
		const Fixture& fx = GetFixture(m.stem);
		if (!fx.ok) continue;
		int64_t B = 0;
		AtDerived(fx, m.D, &B);
		const Lay y = LayoutOf(fx.geo, 1, B);
		const int32_t budgets[] = {1, static_cast<int32_t>(B - 1), static_cast<int32_t>(B), static_cast<int32_t>(B + 1), 512};
		int bad = 0;
		std::string first;
		for (int64_t off = 0; off < B; ++off) {
			const int32_t origin = static_cast<int32_t>(1024 - B + off);  // offsets 0..B-1 within one page
			PagePoolOf pool(fx.model, static_cast<uint32_t>(CeilDiv(origin, B) + 2 * Reservation(512, fx.geo.context_cap, B) + 2),
			                y.page_bytes);
			if (!pool.pool) {
				if (bad++ == 0) first = "pool refused: " + std::string(AbiStatusName(pool.status));
				continue;
			}
			sslm_prefix p = bc::BudgetPrefix(fx, &pool.pool, Stream(51, origin, fx.vocab), true, nullptr);
			for (int32_t b : budgets) {
				sslm_seq s = nullptr;
				sslm_status refusal = SSLM_OK;
				int64_t n = -1;
				if (p && sslm_seq_create_budgeted(fx.model, &pool.pool, b, &s) == SSLM_OK &&
				    sslm_seq_adopt_prefix(s, p) == SSLM_OK)
					n = static_cast<int64_t>(bc::RunToRefusal(fx.model, s, &refusal).size());
				if (s) sslm_seq_release(s);
				if (n != b + 1 || refusal != PKV_KV_BUDGET_EXCEEDED) {
					if (bad++ == 0)
						first = Fmt("%s origin %d budget %d: %lld tokens, refused %s", m.stem, origin, b,
						            static_cast<long long>(n), AbiStatusName(refusal));
				}
			}
			if (p) sslm_prefix_release(p);
		}
		PKV_CHECK_MSG(bad == 0, "%s at B = %lld: %d of %lld cases wrong; first: %s", m.stem, static_cast<long long>(B), bad,
		              static_cast<long long>(B * 5), first.c_str());
	}
}

void Cell413Restore() {
	const Fixture& fx = GetFixture("pkv_def");
	if (!fx.ok) return;
	int64_t B = 0;
	AtDerived(fx, 48, &B);
	const Lay y = LayoutOf(fx.geo, 1, B);
	for (int32_t origin : {1200, 1024, 1000}) {
		PagePoolOf pool(fx.model, 120, y.page_bytes);
		PKV_CHECK_EQ(pool.status, SSLM_OK);
		if (!pool.pool) continue;
		sslm_prefix p = bc::BudgetPrefix(fx, &pool.pool, Stream(51, origin, fx.vocab));
		sslm_seq s = nullptr;
		PKV_CHECK_EQ(sslm_seq_create_budgeted(fx.model, &pool.pool, 512, &s), SSLM_OK);
		if (!p || !s) continue;
		PKV_CHECK_EQ(sslm_seq_adopt_prefix(s, p), SSLM_OK);
		bc::DecodeN(fx.model, s, 6);
		const std::vector<uint8_t> blob = bc::Save(s);
		const int64_t L = origin + 5;
		for (bool with_handle : {true, false}) {
			sslm_seq r = nullptr;
			uint32_t shared = 0;
			PKV_CHECK_EQ(sslm_seq_restore_shared(fx.model, &pool.pool, blob.data(), blob.size(), with_handle ? p : nullptr,
			                                     &r, &shared),
			             SSLM_OK);
			if (!r) continue;
			PKV_CHECK_MSG(shared == (with_handle ? static_cast<uint32_t>(origin / B) : 0u),
			              "origin %d %s: shared %u pages (diagnostic), %lld owed", origin,
			              with_handle ? "with its handle" : "without", shared,
			              static_cast<long long>(with_handle ? origin / B : 0));
			std::vector<uint8_t> rows, brows;
			PKV_CHECK_MSG(SeqRows(pool.pool, r, fx.geo, 1, B, L, Rot::kRotated, &rows) &&
			                  BlobRows(blob, fx, HiddenSize(fx), L, &brows) && rows == brows,
			              "origin %d %s: restored rows at B = %lld equal the blob's", origin,
			              with_handle ? "with its handle" : "without", static_cast<long long>(B));
			if (!with_handle) {
				// E = min(floor(origin/B), ceil(cap/B) - R(budget)) materialized pages: the mapped table holds
				// them privately, and they go back to the pool at the first reset
				std::vector<uint32_t> t;
				legacy_a::SeqTable(r, &t);
				PKV_CHECK_EQ(static_cast<int64_t>(t.size()), CeilDiv(L, B));
			}
			PKV_CHECK(bc::DecodeN(fx.model, r, 4) == bc::DecodeN(fx.model, s, 4));
			// rewind s's extra tokens: re-make s from the blob so the next restore compares like with like
			sslm_seq_release(r);
			sslm_seq_release(s);
			s = nullptr;
			PKV_CHECK_EQ(sslm_seq_restore_shared(fx.model, &pool.pool, blob.data(), blob.size(), p, &s, nullptr), SSLM_OK);
			if (!s) break;
		}
		if (s) sslm_seq_release(s);
		sslm_prefix_release(p);
	}
}

void Cell46() {
	const char* const scenarios[] = {"lifecycle", "long_prefill", "shared_preamble", "persist", "saturating", "widths",
	                                 "prefix_lengths"};
	for (const Member& m : {kMembers[0], kMembers[2]}) {
		const Fixture& fx = GetFixture(m.stem);
		if (!fx.ok) continue;
		auto e = EngineOf(fx);
		PKV_CHECK_MSG(e->ok && e->DirectQk() == (std::strcmp(m.stem, "pkv_qk") == 0),
		              "%s: the score branch is not the one the cell is for", m.stem);
		int64_t B = 0;
		AtDerived(fx, m.D, &B);
		for (const char* sc : scenarios)
			ExpectMatchesReference("v1.11.0", fx, sc, RunScenario(fx, sc), Compare::kTokensRowsAndBlob);
		PersistRowsAtDerived(fx, B);
		DerivedBoundaries(fx, B, kBoundaryWidths | kBoundaryPrefixes, "4.6/T1");  // each branch across every page edge (F-2)
	}
}

PKV_CELL("4.13/T1:legacy", "T1", Cell413Legacy);
PKV_CELL("4.13/T1:budget", "T1", Cell413Budget);
PKV_CELL("4.13/T1:restore", "T1", Cell413Restore);
PKV_CELL("4.6/T1", "T1", Cell46);

}  // namespace

// Paged-KV plan rev 16.21 §23.8, the real-artifact members of the R9 red suite: superslm_pkv_r9_box.
// AUTHORED AND COMPILED AT T0, BOX-OWED: the release box is serialized and unstable (D-SLM8235), so
// none of these runs on the bench. Each reads SUPERSLM_PAGED_KV_REAL_ARTIFACT_DIR.
//
//   4.6/box    the 0.5B (H_kv 2, D 64, cap 4,096) and the 1.5B (H_kv 2, D 128, cap 32,768), unpinned at
//              the derived B = 64, rotated: tokens and rows of a page-pool legacy holder equal the
//              same artifact's contiguous-option holder (T3's verb) at widths 1, 63, 64, 65, 1,064 and
//              cap - 1, and the rows read through the peek equal the holder's own legacy save
//   2.9b/box   the 0.5B at cap 32,768 (B = 128): 16,777,215 blocks pass the page-count check, one more is
//              refused before the buffer check
//   7.4c/box   on the unpinned 0.5B, a mid-page share adopt copies exactly page_bytes, read through the
//              peek: 384 KiB at cap 4,096 and 768 KiB at cap 32,768; the mutant copying B = 16's page fails
//   4.15/box   copy adopt across page geometries on the 0.5B at cap 4,096 (B = 64): page from contiguous at
//              len 64 and 65, contiguous from page at len 100, refused; the controls admitted (rev 16.23); the bind
//              and precedence members bind the artifact's own schema, potion_shop_order (T-3212 C-1)
//   4.12/box   the harness check: every real-artifact cell records the B it ran at and asserts §3.1's rule
//              (each cell above calls RecordB first)
//
// Artifacts (the box's names, tools/paged_kv_box_common.ps1 and cell 7.4's construction):
//   qwen2.5-0.5b-instruct-cap4096-aex.sslm, qwen2.5-0.5b-instruct-cap32768-aex.sslm, qwen2.5-1.5b-instruct.sslm

#include "pkv_r9.h"
#include "pkv_r9_adopt415.h"

namespace {

using namespace pkv;
using namespace pkv::r9;

constexpr const char* k05B = "qwen2.5-0.5b-instruct-cap4096-aex.sslm";
constexpr const char* k05B32k = "qwen2.5-0.5b-instruct-cap32768-aex.sslm";
constexpr const char* k15B = "qwen2.5-1.5b-instruct.sslm";
const Geometry kGeo05B{24, 2, 64, 4096};
const Geometry kGeo05B32k{24, 2, 64, 32768};
const Geometry kGeo15B{28, 2, 128, 32768};

// 4.12's harness check: the B this run is at, recorded, and equal to the rule.
int64_t RecordB(const Fixture& fx) {
	const int64_t got = sslm_kv_page_positions(fx.model), want = RuleB(fx.geo.head_dim, static_cast<uint64_t>(fx.geo.context_cap), 1);
	std::printf("  %s: B = %lld (rule %lld)\n", fx.stem.c_str(), static_cast<long long>(got), static_cast<long long>(want));
	PKV_CHECK_MSG(got == want, "%s ran at B = %lld, the rule gives %lld", fx.stem.c_str(), static_cast<long long>(got),
	              static_cast<long long>(want));
	return want;
}

// The K/V rows [0, L) of a legacy SSB5 save: its last sslm_kv_block_size bytes are the flat block.
std::vector<uint8_t> LegacyBlobRows(const Fixture& fx, const std::vector<uint8_t>& blob, int64_t L) {
	const size_t block = sslm_kv_block_size(fx.model);
	if (blob.size() < block) return {};
	return RowsFromBlock(blob.data() + blob.size() - block, fx.geo, L);
}

void Box46On(const Fixture& fx) {
	if (!fx.ok) return;
	const int64_t B = RecordB(fx);
	const Lay y = LayoutOf(fx.geo, 1, B);
	const int64_t cap = fx.geo.context_cap;
	for (int64_t W : {int64_t{1}, B - 1, B, B + 1, int64_t{1064}, cap - 1}) {
		BlockPoolOf pp(fx.model, 1, y);
		bool fits = false;
		ContigPoolOf cp(fx.model, 1, static_cast<size_t>(ContiguousSize(y, 1, &fits)));
		if (!pp.pool || !cp.pool) {
			PKV_CHECK_MSG(false, "W %lld: pools %s / %s", static_cast<long long>(W), AbiStatusName(pp.status), AbiStatusName(cp.status));
			continue;
		}
		sslm_seq a = nullptr, b = nullptr;
		PKV_CHECK_EQ(sslm_seq_create(fx.model, &pp.pool, &a), SSLM_OK);
		PKV_CHECK_EQ(sslm_seq_create(fx.model, &cp.pool, &b), SSLM_OK);
		if (!a || !b) continue;
		const AbiWidth wa = RunAbiWidth(fx.model, a, W, 151936, true), wb = RunAbiWidth(fx.model, b, W, 151936, true);
		PKV_CHECK_MSG(wa.ready == wb.ready && wa.decode1 == wb.decode1 && wa.ready >= 0,
		              "%s W %lld: tokens differ between the page pool and the option", fx.stem.c_str(), static_cast<long long>(W));
		std::vector<uint8_t> rows;
		const std::vector<uint8_t> blob = bc::Save(a);
		PKV_CHECK_MSG(SeqRows(pp.pool, a, fx.geo, 1, B, W + 1, Rot::kRotated, &rows) && rows == LegacyBlobRows(fx, blob, W + 1),
		              "%s W %lld: rows read at B = %lld rotated differ from the legacy save's", fx.stem.c_str(),
		              static_cast<long long>(W), static_cast<long long>(B));
		PKV_CHECK_MSG(bc::Save(b) == blob, "%s W %lld: the legacy saves differ between pool kinds", fx.stem.c_str(),
		              static_cast<long long>(W));
		sslm_seq_release(a);
		sslm_seq_release(b);
	}
}

void Box46() {
	Box46On(bc::GetArtifact(k05B, kGeo05B));
	Box46On(bc::GetArtifact(k15B, kGeo15B));
}

void Box29b() {
	const Fixture& fx = bc::GetArtifact(k05B32k, kGeo05B32k);
	if (!fx.ok) return;
	PKV_CHECK_EQ(RecordB(fx), 128);
	auto tiny = [&](uint32_t n) {
		AlignedBuf small(4096);
		sslm_kv_pool pool = nullptr;
		const sslm_status st = sslm_kv_pool_create(fx.model, small.p, small.n, n, &pool);
		if (pool) sslm_kv_pool_destroy(pool);
		return st;
	};
	PKV_CHECK_EQ(tiny(16777215u), SSLM_BUFFER_TOO_SMALL);
	PKV_CHECK_EQ(tiny(16777216u), SSLM_INVALID_ARGUMENT);
}

// 7.4c: a mid-page share adopt (prefix 1,000) copies exactly one page of page_bytes: the adopter's tail
// page equals the prefix's tail page byte for byte over the whole page, in a pool pre-filled with 0x5A
// (so a copy of B = 16's 96 KiB leaves the rest of the destination page at 0x5A and fails).
void Box74cOn(const Fixture& fx, size_t want_page_bytes) {
	if (!fx.ok) return;
	const int64_t B = RecordB(fx);
	const Lay y = LayoutOf(fx.geo, 1, B);
	PKV_CHECK_EQ(y.page_bytes, want_page_bytes);
	PagePoolOf pool(fx.model, 60, y.page_bytes, 0x5A);
	PKV_CHECK_EQ(pool.status, SSLM_OK);
	if (!pool.pool) return;
	sslm_prefix px = nullptr;
	sslm_seq s = nullptr;
	PKV_CHECK_EQ(sslm_prefix_begin_budgeted(fx.model, &pool.pool, 1000, &px), SSLM_OK);
	if (px) PKV_CHECK_EQ(PrefixPrefillAll(fx.model, px, Stream(51, 1000, 151936), 64), SSLM_OK);
	if (px) PKV_CHECK_EQ(sslm_prefix_freeze(px), SSLM_OK);
	PKV_CHECK_EQ(sslm_seq_create_budgeted(fx.model, &pool.pool, 64, &s), SSLM_OK);
	if (px && s) {
		PKV_CHECK_EQ(sslm_seq_adopt_prefix(s, px), SSLM_OK);
		const uint32_t tail = static_cast<uint32_t>(1000 / B);
		uint32_t sp = 0, pp = 0;
		PKV_CHECK_EQ(sslm_pkv_test_only_seq_table_entry(s, tail, &sp, nullptr), SSLM_OK);
		PKV_CHECK_EQ(sslm_pkv_test_only_prefix_table_entry(px, tail, &pp, nullptr), SSLM_OK);
		PKV_CHECK_MSG(sp != pp, "%s: the tail page is shared, not copied", fx.stem.c_str());
		const std::vector<uint8_t> a = legacy_a::PeekPage(pool.pool, sp, y.page_bytes), b = legacy_a::PeekPage(pool.pool, pp, y.page_bytes);
		PKV_CHECK_MSG(!a.empty() && a == b, "%s: the copied tail page is not the prefix's page over its %zu bytes",
		              fx.stem.c_str(), y.page_bytes);
	}
	if (s) sslm_seq_release(s);
	if (px) sslm_prefix_release(px);
}

void Box74c() {
	Box74cOn(bc::GetArtifact(k05B, kGeo05B), 384 * 1024);
	Box74cOn(bc::GetArtifact(k05B32k, kGeo05B32k), 768 * 1024);
}

// The 0.5B artifact carries one schema, potion_shop_order, and no g5_minimal_one_field (T-3212 C-1: its one SCM1
// section, read at source); the bind and precedence members bind the artifact's own schema.
constexpr const char* k05BSchema = "potion_shop_order";

void Box415() {
	const Fixture& fx = bc::GetArtifact(k05B, kGeo05B);
	if (!fx.ok) return;
	RecordB(fx);
	pkv::r9::c415::Cell415On(fx, 64, 64, 100, k05BSchema);
}

// bracketed T3 (plan rev 16.24 §26.4, D-SLM8444): its reference arm is the contiguous create, T3's verb
PKV_CELL("4.6/box", "T3", Box46);
PKV_CELL("4.15/box", "T3", Box415);
PKV_CELL("2.9b/box", "T2", Box29b);
PKV_CELL("7.4c/box", "T1", Box74c);

}  // namespace

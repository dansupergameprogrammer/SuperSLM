// Paged-KV plan (rev 16.1) §7 dimension 6, the budget-holder parts (step C5): 6.1's budget ABI
// part, 6.3, 6.4b, and the SSB6 half of 6.4a that the carrier row RB16-CELLS moves to the first
// SSB6 writer (a budget holder).
//
// Byte-equality for a budget holder compares tokens and K/V rows, never the whole blob, against
// the R0 reference (v1.11.0) or a legacy twin: a budget holder saves SSB6, so its blob differs from
// the reference's 1.9.0 SSB5 by design (§3.7). The rows are read out of each blob by the layout
// formula written in the test (pkv_common.h BlobRows), never by calling the engine.

#include "pkv_budget_b_helpers.h"

#include "superslm/matmul.h"

namespace {

using namespace pkv;
using namespace pkv::budget_b;

// ---- 6.1 [C5] ------------------------------------------------------------------------------------

// §3.1's address of (l, half, h, pos) inside a page, written here: ((l*2 + half)*H_kv + h)*B*D +
// (pos % B)*D. 6.1's layout half (§8, "Layout pinning independent of the code under test"): the
// bytes a budget holder's pages hold, read through the page-index peek and addressed by this
// formula, equal the rows its own blob carries.
void CheckPageLayout(const Fixture& fx, sslm_kv_pool pool, sslm_seq s, int64_t L) {
	const std::vector<uint8_t> blob = Save(s);
	const std::vector<uint8_t> rows = Rows(blob, fx, L);
	const std::vector<uint32_t> table = SeqTable(s);
	const int64_t B = fx.B();
	PKV_CHECK_EQ(static_cast<int64_t>(table.size()), (L + B - 1) / B);
	if (rows.empty() || static_cast<int64_t>(table.size()) < (L + B - 1) / B) return;
	const size_t D = fx.geo.head_dim, H = fx.geo.kv_heads;
	std::map<uint32_t, std::vector<uint8_t>> pages;
	size_t at = 0, bad = 0;
	for (int64_t p = 0; p < L; ++p)
		for (uint32_t l = 0; l < fx.geo.layers; ++l)
			for (uint32_t half = 0; half < 2; ++half)
				for (uint32_t h = 0; h < H; ++h) {
					const uint32_t page = table[static_cast<size_t>(p / B)];
					auto it = pages.find(page);
					if (it == pages.end()) it = pages.emplace(page, PeekPage(pool, fx, page)).first;
					const size_t off = ((size_t{l} * 2 + half) * H + h) * static_cast<size_t>(B) * D +
					                   static_cast<size_t>(p % B) * D;
					if (it->second.size() < off + D || std::memcmp(it->second.data() + off, rows.data() + at, D) != 0) ++bad;
					at += D;
				}
	// kills: a page layout that is self-consistent but not §3.1's (position-major within a page, or
	// the halves swapped), which the blob alone cannot see
	PKV_CHECK_MSG(bad == 0, "%zu of %lld rows differ between the pages (§3.1 formula) and the blob", bad,
	              static_cast<long long>(L * fx.geo.layers * 2 * H));
}

// 4.3's sweep with a legacy twin: a budget prefix of 32 + o tokens (origin offset o = 0..B-1 inside
// its page), a budget-b adopter run to its refusal, and a legacy holder adopting the same tokens by
// copy, run for the same number of tokens. Tokens and the rows [0, L') of the two saves are equal.
void Sweep43AgainstLegacyTwin(const Fixture& fx) {
	for (int64_t o = 0; o < fx.B(); ++o) {
		const int32_t plen = static_cast<int32_t>(32 + o);
		const std::vector<int32_t> ptoks = Stream(91, plen, fx.vocab);
		LegacyPool lpool(fx.model, 2);
		sslm_prefix lpx = MakePrefix(fx, &lpool.pool, ptoks, 0);
		for (int32_t b : {1, 15, 16, 17, 512}) {
			PagePool pool(fx.model, static_cast<uint32_t>(fx.R(plen) + fx.R(b)));
			sslm_prefix px = MakePrefix(fx, &pool.pool, ptoks, plen);
			sslm_seq s = px ? MakeBudgetSeq(fx, &pool.pool, b) : nullptr;
			sslm_seq t = lpx ? MakeLegacySeq(fx, &lpool.pool) : nullptr;
			if (s && t) {
				PKV_CHECK_EQ(sslm_seq_adopt_prefix(s, px), SSLM_OK);
				PKV_CHECK_EQ(sslm_seq_adopt_prefix(t, lpx), SSLM_OK);
				sslm_status refusal = SSLM_OK;
				const std::vector<int32_t> got = RunToRefusal(fx, s, &refusal, b + 4);
				std::vector<int32_t> want;
				DecodeN(fx, t, static_cast<int>(got.size()), &want);
				// kills: a page-run boundary inside the budget that drops, repeats or misplaces a row
				PKV_CHECK_MSG(got == want, "4.3 origin offset %lld budget %d: tokens differ from the legacy twin",
				              static_cast<long long>(o), b);
				const std::vector<uint8_t> bs = Save(s), bt = Save(t);
				const int64_t L = BlobLp(bs);
				PKV_CHECK_EQ(L, BlobLp(bt));
				PKV_CHECK_MSG(Rows(bs, fx, L) == Rows(bt, fx, L), "4.3 origin offset %lld budget %d: K/V rows differ",
				              static_cast<long long>(o), b);
			}
			if (s) sslm_seq_release(s);
			if (t) sslm_seq_release(t);
			if (px) sslm_prefix_release(px);
		}
		if (lpx) sslm_prefix_release(lpx);
	}
}

// 6.1 [C5]: tokens and per-position K/V bytes of the budget ABI equal the reference for every cell
// of 4.1 (widths 1, B-1, B, B+1, 2B, prefix + 64, cap-1, cap), 4.2 (prefix lengths 0, 1, B-1, B,
// B+1, 1,000, 1,008) and 4.3 (origin offsets 0..B-1 x budgets {1, B-1, B, B+1, 512}).
void Cell61Budget() {
	const Fixture& fx = GetFixture("pkv_def");
	if (!fx.ok) return;
	// 4.1: every width as a budget holder of budget min(W + 1, cap) in a pool of exactly R(budget).
	// kills: per-page attention that mis-chains a run at a page boundary (W = B-1, B, B+1, 2B);
	// a guard or map that stops one short of the limit (W = cap - 1 and cap are the last writes)
	ExpectMatchesReference("v1.11.0", fx, "widths", BudgetWidths(fx), Compare::kTokensAndRows);
	// The B = cap fallback runs the same widths at one page per holder (§3.1).
	const Fixture& odd = GetFixture("pkv_odd");
	if (odd.ok) ExpectMatchesReference("v1.11.0", odd, "widths", BudgetWidths(odd), Compare::kTokensAndRows);
	// 4.2: budget prefix x budget share adopt at every prefix length.
	// kills: share adopt that maps the tail page instead of copying it (1,000), or copies a tail an
	// aligned prefix does not have (1,008, 16); an empty prefix adopted as a one-row prefix (0)
	ExpectMatchesReference("v1.11.0", fx, "prefix_lengths", BudgetPrefixLengths(fx), Compare::kTokensAndRows);
	// 4.3 against a legacy twin (there is no reference record for these origins and budgets).
	Sweep43AgainstLegacyTwin(fx);
	// The layout half on a budget holder whose rows span 67 pages (prefix + 64 = 1,064, decoded once).
	{
		PagePool pool(fx.model, static_cast<uint32_t>(fx.R(1065)));
		sslm_seq s = MakeBudgetSeq(fx, &pool.pool, 1065);
		if (s) {
			PKV_CHECK_EQ(PrefillAll(fx.model, s, Stream(41, 1064, fx.vocab), 64), SSLM_OK);
			std::vector<int32_t> t;
			DecodeN(fx, s, 2, &t);
			CheckPageLayout(fx, pool.pool, s, 1065);
			sslm_seq_release(s);
		}
	}
}

// ---- 6.3 [C5] ------------------------------------------------------------------------------------

// pkv_scenarios.h LongPrefill on a budget holder of budget 1,515 (the 1,500 prefilled rows and the
// 15 the decode writes): 64-token chunks, so every chunk runs the batched (tiled, >= 8 tokens)
// path across page boundaries, then single-token decode over 95 pages.
std::vector<Record> BudgetLongPrefill(const Fixture& fx) {
	std::vector<Record> recs;
	std::vector<int32_t> pending;
	PagePool pool(fx.model, static_cast<uint32_t>(fx.R(1515)));
	sslm_seq s = MakeBudgetSeq(fx, &pool.pool, 1515);
	if (!s) return recs;
	if (PrefillAll(fx.model, s, Stream(11, 1500, fx.vocab), 64) == SSLM_OK) {
		recs.push_back(MakeRecord(fx, "prefill1500", s, &pending));
		if (DecodeN(fx, s, 16, &pending)) recs.push_back(MakeRecord(fx, "prefill1500+decode16", s, &pending));
	}
	sslm_seq_release(s);
	return recs;
}

// 6.3 [C5]: SIMD tiers are byte-equal (4.7). This is the determinism side; 4.7 (another part of
// the suite) is the toolchain matrix. The tier is fixed per library build: the dispatch probes CPUID
// once, and SUPERSLM_FORCE_{SCALAR,SSE2,AVX2,AVX512}_MATMUL pin it at compile time (matmul.cpp), so
// no single process can run two tiers. The cell therefore pins every tier to one fixed point: the
// budget holders' tokens and K/V rows equal the v1.11.0 reference, on whichever tier this build
// dispatches to, through the paths whose kernels the tiers implement (per-page score and context
// runs of every length 1..B, the tiled chunk path, single-token decode). Two tiers that each equal
// the reference equal each other byte for byte. The cloud leg runs the native tier (printed); the
// box/CI leg runs this same cell in each forced-tier build of the library (the four macros above,
// and MSVC and clang-cl per 4.7), which is what makes the matrix.
void Cell63Tiers() {
	const Fixture& fx = GetFixture("pkv_def");
	if (!fx.ok) return;
	std::printf("  6.3: GEMM tier of this build: %d (0 scalar, 1 SSE2, 2 AVX2, 3 AVX-512)\n",
	            static_cast<int>(superslm::detail::ActiveGemmTier()));
	// kills: a tier whose accumulate-into of a short page run (rows < its vector width) differs
	// from the scalar sum, on that tier's build
	ExpectMatchesReference("v1.11.0", fx, "long_prefill", BudgetLongPrefill(fx), Compare::kTokensAndRows);
	ExpectMatchesReference("v1.11.0", fx, "widths", BudgetWidths(fx), Compare::kTokensAndRows);
	// The direct_qk score branch (QkQ31ScoreRow per page run) has its own SIMD kernels; pkv_qk's
	// reference holds for the host that generated its fixture (pkv_common.h checks the digest).
	const Fixture& qk = GetFixture("pkv_qk");
	if (qk.ok) ExpectMatchesReference("v1.11.0", qk, "widths", BudgetWidths(qk), Compare::kTokensAndRows);
}

// ---- 6.4a, the SSB6 half (RB16-CELLS) -------------------------------------------------------------

// 6.4a's construction on a budget holder: reset and reuse so that the row at the new context_length
// holds a prior generation's K/V, then save mid-token at that position. The SSB6 blob equals a fresh
// budget holder's, and the mid-token row's layers >= layer_index are zero in it (§3.7). Both pools
// are pre-filled with 0x5A, so a row no generation wrote is not zero either: whichever page the
// reused holder maps there, the gather-from-memory mutant reads non-zero bytes.
void Cell64aSsb6() {
	const Fixture& fx = GetFixture("pkv_def");
	if (!fx.ok) return;
	const int32_t budget = 128;
	auto midtoken_at_40 = [&](sslm_seq s) {
		PKV_CHECK_EQ(PrefillAll(fx.model, s, Stream(2, 40, fx.vocab), 16), SSLM_OK);
		std::vector<int32_t> t;
		DecodeN(fx, s, 1, &t);  // the ready token: no row
		EnterMidToken(fx, s);   // layer 0 of row 40 written; layer 1 is not
	};
	PagePool reused_pool(fx.model, static_cast<uint32_t>(fx.R(budget)), 0x5A);
	PagePool fresh_pool(fx.model, static_cast<uint32_t>(fx.R(budget)), 0x5A);
	sslm_seq reused = MakeBudgetSeq(fx, &reused_pool.pool, budget);
	sslm_seq fresh = MakeBudgetSeq(fx, &fresh_pool.pool, budget);
	if (!reused || !fresh) return;
	// The prior generation: 112 rows, filling pages 0..6 whole.
	PKV_CHECK_EQ(PrefillAll(fx.model, reused, Stream(1, 100, fx.vocab), 64), SSLM_OK);
	std::vector<int32_t> t;
	DecodeN(fx, reused, 13, &t);
	PKV_CHECK_EQ(sslm_seq_reset(reused), SSLM_OK);
	midtoken_at_40(reused);
	midtoken_at_40(fresh);
	// Precondition, so the construction separates: memory at row 40, layer 1, holds non-zero bytes
	// in the reused holder's page (a prior generation's, or the pool's 0x5A).
	const std::vector<uint32_t> table = SeqTable(reused);
	if (table.size() > 2) {
		const std::vector<uint8_t> page = PeekPage(reused_pool.pool, fx, table[2]);
		const size_t D = fx.geo.head_dim, H = fx.geo.kv_heads, B = static_cast<size_t>(fx.B());
		const size_t off = ((1 * 2 + 0) * H + 0) * B * D + (40 % B) * D;
		bool nonzero = false;
		for (size_t d = 0; d < D && off + d < page.size(); ++d) nonzero = nonzero || page[off + d] != 0;
		PKV_CHECK_MSG(nonzero, "construction: row 40 layer 1 holds zero bytes in memory, so the mutant cannot separate");
	}
	const std::vector<uint8_t> a = Save(reused), b = Save(fresh);
	PKV_CHECK(IsMagic(a, "SSB6"));
	PKV_CHECK_EQ(BlobLp(a), 41);
	// kills: the save gathers row context_length's layers >= layer_index from memory instead of
	// writing zero bytes (the reused holder's blob then carries the prior generation's K/V)
	PKV_CHECK_MSG(a == b, "the reused budget holder's mid-token SSB6 differs from a fresh holder's");
	// The zero substitution read directly, by the layout formula: row 40, layer 1, K and V, both heads.
	const std::vector<uint8_t> rows = Rows(a, fx, 41);
	const size_t per_pos = fx.BytesPerToken();
	if (rows.size() == 41 * per_pos) {
		const size_t layer1 = 40 * per_pos + per_pos / fx.geo.layers;  // layer-major inside a position
		PKV_CHECK_MSG(std::all_of(rows.begin() + static_cast<std::ptrdiff_t>(layer1),
		                          rows.begin() + static_cast<std::ptrdiff_t>(41 * per_pos), [](uint8_t x) { return x == 0; }),
		              "row 40's layer >= layer_index bytes are not zero in the SSB6 blob");
	}
	sslm_seq_release(reused);
	sslm_seq_release(fresh);
}

// ---- 6.4b [C5] ------------------------------------------------------------------------------------

// 6.4b: legacy copy adopt and budget share adopt of one prefix give identical K/V rows [0, L') in
// their blobs (the legacy holder saves 1.9.0's SSB5, the budget holder SSB6, so rows are compared,
// not sections). Both adopters live in one page pool and adopt the same prefix handle, of each mode,
// at a mid-page (1,000) and an aligned (1,008) length; rows are compared straight after the adopt,
// after decode and prefill across page boundaries, and at a mid-token save (the zero substitution on
// both writers).
void Cell64b() {
	const Fixture& fx = GetFixture("pkv_def");
	if (!fx.ok) return;
	for (int32_t len : {1000, 1008}) {
		for (int prefix_budget : {0, 1}) {
			const int32_t pb = prefix_budget ? len : 0;
			// A legacy prefix begin draws ceil(cap/B) like a legacy create (§3.4), and keeps
			// ceil(len/B) after freeze; the legacy adopter reserves ceil(cap/B).
			PagePool pool(fx.model, static_cast<uint32_t>(2 * fx.CapPages() + fx.R(64) + 2));
			sslm_prefix px = MakePrefix(fx, &pool.pool, Stream(51, len, fx.vocab), pb);
			sslm_seq legacy = px ? MakeLegacySeq(fx, &pool.pool) : nullptr;
			sslm_seq budget = legacy ? MakeBudgetSeq(fx, &pool.pool, 64) : nullptr;
			if (!budget) {
				if (legacy) sslm_seq_release(legacy);
				if (px) sslm_prefix_release(px);
				continue;
			}
			PKV_CHECK_EQ(sslm_seq_adopt_prefix(legacy, px), SSLM_OK);
			PKV_CHECK_EQ(sslm_seq_adopt_prefix(budget, px), SSLM_OK);
			auto same_rows = [&](const char* stage) {
				const std::vector<uint8_t> a = Save(legacy), b = Save(budget);
				PKV_CHECK(IsMagic(a, "SSB5") && IsMagic(b, "SSB6"));
				const int64_t L = BlobLp(a);
				PKV_CHECK_EQ(L, BlobLp(b));
				// kills: share adopt that maps a page other than the prefix's, or copies the tail short;
				// copy adopt that copies a different span than the share maps
				PKV_CHECK_MSG(Rows(a, fx, L) == Rows(b, fx, L), "len %d prefix %s, %s: K/V rows [0, %lld) differ", len,
				              pb ? "budget" : "legacy", stage, static_cast<long long>(L));
			};
			same_rows("adopted");
			std::vector<int32_t> ta, tb;
			DecodeN(fx, legacy, 8, &ta);
			DecodeN(fx, budget, 8, &tb);
			PKV_CHECK_EQ(PrefillAll(fx.model, legacy, Stream(52, 20, fx.vocab), 7), SSLM_OK);
			PKV_CHECK_EQ(PrefillAll(fx.model, budget, Stream(52, 20, fx.vocab), 7), SSLM_OK);
			DecodeN(fx, legacy, 4, &ta);
			DecodeN(fx, budget, 4, &tb);
			PKV_CHECK_MSG(ta == tb, "len %d: tokens differ between copy and share adopt", len);
			same_rows("decode8+prefill20+decode4");
			EnterMidToken(fx, legacy);
			EnterMidToken(fx, budget);
			same_rows("mid-token");
			sslm_seq_release(budget);
			sslm_seq_release(legacy);
			sslm_prefix_release(px);
		}
	}
}

PKV_CELL("6.1/C5", "C5", Cell61Budget);
PKV_CELL("6.3/C5", "C5", Cell63Tiers);
PKV_CELL("6.4a/C5", "C5", Cell64aSsb6);
PKV_CELL("6.4b", "C5", Cell64b);

}  // namespace

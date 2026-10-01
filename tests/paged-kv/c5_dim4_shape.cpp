// Paged-KV plan (rev 16.1) §7 dimension 4, shape and platform matrices: the C5-owned cells and parts
// (budget holders at every width, prefix length, origin offset and budget; the B = cap fallback in
// budget mode; page pools at their edges; the SIMD tiers). Red by link at C1.
//
// Byte-equality is against the R0 reference (v1.11.0), compared as tokens and K/V rows because a
// budget holder saves SSB6; page counts by admission or the fill probe.

#include "pkv_budget_a_helpers.h"

#include "superslm/matmul.h"

namespace {

using namespace pkv;
using namespace pkv::budget_a;

// ---- 4.1 [C5 ABI] -------------------------------------------------------------------------------
// Widths 1, B-1, B, B+1, 2B, prefix + 64, cap-1 and cap (the last admissible write), each in a budget
// holder whose budget is exactly what its writes need: tokens and rows equal the reference's widths
// records, and one write past the budget is refused with the status §3.6 names (the budget status
// below the cap, the cap status at it). "Prefix + 64" runs twice: prefilled whole (the reference's
// own construction) and as an adopt of the first 1,000 tokens plus 64 more, which the reference also
// pins because rows are a function of tokens and positions only.
void Cell41Budget() {
	const Fixture& fx = GetFixture("pkv_def");
	if (!fx.ok) return;
	const int64_t cap = fx.geo.context_cap;
	{
		PagePool pool(fx.model, static_cast<uint32_t>(fx.CapPages() + 8));
		BRun d(fx);
		BudgetWidths(d, &pool.pool, ReferenceWidths(fx));
		PKV_CHECK_MSG(d.fail.empty(), "4.1 widths: %s", d.fail.c_str());
		// kills: a page-run boundary error in a budget holder's attention or row writes at any width.
		ExpectMatchesReference("v1.11.0", fx, "widths", d.records, Compare::kTokensAndRows);
	}
	for (int64_t w : ReferenceWidths(fx)) {
		Scene sc;
		PagePool& pool = sc.AddPool(fx.model, static_cast<uint32_t>(fx.CapPages() + 8));
		const int64_t limit = w < cap ? w + 1 : cap;
		sslm_seq s = BudgetSeq(fx, &pool.pool, static_cast<int32_t>(limit), sc.h);
		if (!s || PrefillAll(fx.model, s, Stream(41, static_cast<int32_t>(w), fx.vocab), 64) != SSLM_OK) {
			PKV_CHECK_MSG(false, "4.1 width %lld: setup", static_cast<long long>(w));
			continue;
		}
		sslm_status refusal = SSLM_OK;
		const int64_t n = TokensUntilRefusal(fx.model, s, &refusal);
		// §3.4: resting ready at c = w, it emits 1 + (limit - c) tokens. kills: the coverage guard or
		// the budget check one position early (fewer) or late (a write past the reservation).
		PKV_CHECK_MSG(n == 1 + (limit - w), "4.1 width %lld: %lld tokens, want %lld", static_cast<long long>(w),
		              static_cast<long long>(n), static_cast<long long>(1 + (limit - w)));
		PKV_CHECK_EQ(refusal, limit < cap ? PKV_KV_BUDGET_EXCEEDED : SSLM_CONTEXT_CAP_EXCEEDED);
	}
	{  // prefix + 64, adopted
		Scene sc;
		PagePool& pool = sc.AddPool(fx.model, 80);
		const std::vector<int32_t> all = Stream(41, 1064, fx.vocab);
		sslm_prefix p = FrozenBudgetPrefix(fx, &pool.pool, std::vector<int32_t>(all.begin(), all.begin() + 1000), sc.h);
		sslm_seq s = BudgetSeq(fx, &pool.pool, 65, sc.h);
		BRun d(fx);
		d.Ok(p && s ? SSLM_OK : SSLM_INVALID_ARGUMENT, "setup") && d.Ok(sslm_seq_adopt_prefix(s, p), "adopt") &&
		    d.Prefill(s, std::vector<int32_t>(all.begin() + 1000, all.end()), 64) && d.Mark("w1064_prefill", s) &&
		    d.Decode(s, 1) && d.Mark("w1064_ready", nullptr) && d.Decode(s, 1) && d.Mark("w1064_decode1", s);
		PKV_CHECK_MSG(d.fail.empty() && d.records.size() == 3, "4.1 prefix + 64: %s", d.fail.c_str());
		// kills: a shared page or the copied tail read at the wrong offset once the holder writes past it.
		for (const Record& r : d.records) ExpectRecord("v1.11.0", fx, "widths", r, "4.1 adopted prefix + 64");
	}
}

// ---- 4.2 [C5 budget] ----------------------------------------------------------------------------
// Prefix lengths 0, 1, B-1, B, B+1, 1,000 (mid-page) and 1,008 (aligned): budget prefixes adopted by
// budget-32 holders (R(32) = 3 pages, exactly the span [1,000, 1,030) needs from the mid-page origin).
// Tokens and rows equal the reference's prefix_lengths records.
void Cell42Budget() {
	const Fixture& fx = GetFixture("pkv_def");
	if (!fx.ok) return;
	PagePool pool(fx.model, 80);
	PKV_CHECK_EQ(pool.status, SSLM_OK);
	BRun d(fx);
	BudgetPrefixLengths(d, &pool.pool, 32);
	PKV_CHECK_MSG(d.fail.empty(), "4.2: %s", d.fail.c_str());
	// kills: share adopt mapping ceil(len/B) pages (writing into the frozen prefix's tail page), a tail
	// copy of the wrong rows, an empty or aligned prefix given a tail.
	ExpectMatchesReference("v1.11.0", fx, "prefix_lengths", d.records, Compare::kTokensAndRows);
}

// ---- 4.3 [C5] -----------------------------------------------------------------------------------
// Every origin offset 0..B-1 (origins 992..1,007, so 1,000 is among them) x budgets {1, B-1, B, B+1,
// 512}: an adopter runs its full budget with no refusal and is refused one past -- by decode (it emits
// exactly budget + 1 tokens, §3.4) and by prefill (it consumes exactly `budget` tokens, then 0). 7.2
// rests on this cell: the mutant R = ceil(budget/B) runs out of reserve before the limit whenever the
// span from a mid-page origin needs the +1 page (round 2's 8-token early refusal at origin 1,000,
// budget 512).
void Cell43() {
	const Fixture& fx = GetFixture("pkv_def");
	if (!fx.ok) return;
	const int64_t B = fx.B();
	const int32_t budgets[] = {1, static_cast<int32_t>(B - 1), static_cast<int32_t>(B), static_cast<int32_t>(B + 1), 512};
	int bad = 0;
	std::string first;
	for (int64_t off = 0; off < B; ++off) {
		const int32_t origin = static_cast<int32_t>(992 + off);
		Scene sc;
		PagePool& pool = sc.AddPool(fx.model, 160);
		sslm_prefix p = FrozenBudgetPrefix(fx, &pool.pool, Stream(51, origin, fx.vocab), sc.h);
		for (int32_t b : budgets) {
			char what[80];
			std::snprintf(what, sizeof what, "origin %d budget %d", origin, b);
			sslm_seq s = BudgetSeq(fx, &pool.pool, b, sc.h);
			sslm_status refusal = SSLM_OK;
			int64_t n = -1;
			if (p && s && sslm_seq_adopt_prefix(s, p) == SSLM_OK) n = TokensUntilRefusal(fx.model, s, &refusal);
			if (s) ReleaseSeq(sc.h, s);
			if (n != b + 1 || refusal != PKV_KV_BUDGET_EXCEEDED) {
				if (bad++ == 0)
					first = std::string(what) + ": decode emitted " + std::to_string(n) + ", refused " + std::to_string(refusal);
			}
			sslm_seq q = BudgetSeq(fx, &pool.pool, b, sc.h);
			int64_t consumed = -1, past = -1;
			sslm_status st1 = SSLM_INVALID_ARGUMENT, st2 = SSLM_OK;
			if (p && q && sslm_seq_adopt_prefix(q, p) == SSLM_OK) {
				st1 = PrefillAll(fx.model, q, Stream(77, b, fx.vocab), 64, &consumed);
				st2 = PrefillAll(fx.model, q, Stream(78, 1, fx.vocab), 1, &past);
			}
			if (q) ReleaseSeq(sc.h, q);
			if (st1 != SSLM_OK || consumed != b || st2 != PKV_KV_BUDGET_EXCEEDED || past != 0) {
				if (bad++ == 0)
					first = std::string(what) + ": prefill consumed " + std::to_string(consumed) + " (" + std::to_string(st1) +
					        "), then " + std::to_string(past) + " (" + std::to_string(st2) + ")";
			}
		}
	}
	// kills: R = ceil(budget/B) (an early refusal from an empty reserve), a limit computed from the
	// page rather than origin + budget, a refusal one position late.
	PKV_CHECK_MSG(bad == 0, "4.3: %d of %lld cases wrong; first: %s", bad, static_cast<long long>(2 * B * 5), first.c_str());
}

// ---- 4.4 [C5 budget] ----------------------------------------------------------------------------
// The B fallback in budget mode: pkv_odd (cap 4,100, which 16 does not divide) has B = cap, so every
// R(budget) is 1 page. The budget lifecycle and prefix-length twins equal the v1.11.0 pkv_odd
// reference; in a 4-page pool a budget world, a begin_from child and two budget holders (one at the
// whole cap) are admitted and a fifth create is refused; the child's adopter equals an unshared twin.
//
// The fill probe is not used here: its create sizes are 2 to ceil(cap/B) pages (§8), and at
// ceil(cap/B) = 1 no create spans 2 pages, so it cannot run at B = cap. Admission counts grade instead.
void Cell44Budget() {
	const Fixture& fx = GetFixture("pkv_odd");
	if (!fx.ok) return;
	PKV_CHECK_EQ(fx.B(), fx.geo.context_cap);
	PKV_CHECK_EQ(sslm_kv_page_positions(fx.model), fx.geo.context_cap);  // §3.1's fallback
	for (int32_t b : {1, 512, 4100}) PKV_CHECK_EQ(sslm_kv_pages_for_budget(fx.model, b), fx.R(b));  // = 1: the clamp
	{
		PagePool pool(fx.model, 4);
		BRun d(fx);
		BudgetLifecycle(d, &pool.pool, 128, 300);
		PKV_CHECK_MSG(d.fail.empty(), "4.4 lifecycle on pkv_odd: %s", d.fail.c_str());
		ExpectMatchesReference("v1.11.0", fx, "lifecycle", d.records, Compare::kTokensAndRows);
	}
	{
		PagePool pool(fx.model, 4);
		BRun d(fx);
		BudgetPrefixLengths(d, &pool.pool, 32);
		PKV_CHECK_MSG(d.fail.empty(), "4.4 prefix lengths on pkv_odd: %s", d.fail.c_str());
		ExpectMatchesReference("v1.11.0", fx, "prefix_lengths", d.records, Compare::kTokensAndRows);
	}
	Scene sc;
	PagePool& pool = sc.AddPool(fx.model, 4);
	sslm_prefix world = FrozenBudgetPrefix(fx, &pool.pool, Stream(51, 100, fx.vocab), sc.h, 512);
	sslm_prefix child = nullptr;
	PKV_CHECK_EQ(world ? sslm_prefix_begin_from(world, 512, &child) : SSLM_INVALID_ARGUMENT, SSLM_OK);
	if (child) {
		sc.h.prefixes.push_back(child);
		PKV_CHECK_EQ(PrefixPrefillAll(fx.model, child, Stream(52, 10, fx.vocab), 64), SSLM_OK);
		PKV_CHECK_EQ(sslm_prefix_freeze(child), SSLM_OK);
	}
	sslm_seq s1 = BudgetSeq(fx, &pool.pool, 512, sc.h);
	sslm_seq s2 = BudgetSeq(fx, &pool.pool, static_cast<int32_t>(fx.geo.context_cap), sc.h);
	PKV_CHECK(s1 && s2);  // kills: R unclamped (ceil(budget/B) + 1 = 2 pages per holder: the third holder refused)
	sslm_seq extra = nullptr;
	PKV_CHECK_EQ(sslm_seq_create_budgeted(fx.model, &pool.pool, 1, &extra), SSLM_KV_POOL_EXHAUSTED);  // kills: R = 0
	if (extra) sc.h.seqs.push_back(extra);
	if (s2) ReleaseSeq(sc.h, s2);
	PKV_CHECK(BudgetSeq(fx, &pool.pool, 1, sc.h) != nullptr);  // the released page is admitted again
	std::vector<int32_t> got, want;
	PKV_CHECK_EQ(s1 && child ? sslm_seq_adopt_prefix(s1, child) : SSLM_INVALID_ARGUMENT, SSLM_OK);
	if (s1) DecodeN(fx.model, s1, 8, &got);
	{
		Scene tw;
		PagePool& tp = tw.AddPool(fx.model, 1);
		sslm_seq t = BudgetSeq(fx, &tp.pool, static_cast<int32_t>(fx.geo.context_cap), tw.h);
		std::vector<int32_t> prompt = Stream(51, 100, fx.vocab);
		const std::vector<int32_t> c = Stream(52, 10, fx.vocab);
		prompt.insert(prompt.end(), c.begin(), c.end());
		if (t) {
			PrefillAll(fx.model, t, prompt, 64);
			DecodeN(fx.model, t, 8, &want);
		}
	}
	// kills: begin_from at B = cap sharing a page it should copy (the child's writes land in the world).
	PKV_CHECK_MSG(got.size() == 8 && got == want, "4.4: the begin_from child's adopter differs from the unshared twin");
}

// ---- 4.5 [C5] -----------------------------------------------------------------------------------
// Page pools of 1 page, of exactly one reservation, and of the maximum.
//  - 1 page: created; on pkv_def every create and begin is refused (no reservation is below 2
//    pages); on pkv_odd (B = cap) it admits exactly one budget holder, which runs.
//  - one reservation, R(512) = 33 pages: one budget-512 holder is admitted and runs its whole budget
//    (resting ready at c = 1 it emits 1 + (512 - 1) tokens), a second create is refused while it
//    lives, and after its release the fill probe admits exactly 33. Likewise R(cap) = 256 pages and one
//    legacy holder.
//  - the maximum, page_count = UINT32_MAX: the sizing verbs return the exact page_count * page_bytes +
//    overhead without saturating, and a create one byte short of it is refused before any use of the
//    buffer. A real pool of UINT32_MAX pages is 26 TB at pkv_def's 6,144-byte page, so no leg allocates
//    one; the cell pins the arithmetic and the refusal edge at the maximum instead. The count itself is
//    admissible (decision 24 refuses only counts above UINT32_MAX, and indices stop at UINT32_MAX - 1,
//    below kNoPage): the page module's acceptance of it is 2.9b/C3, the legacy verb's 2.9b [C4].
void Cell45() {
	const Fixture& fx = GetFixture("pkv_def");
	const Fixture& odd = GetFixture("pkv_odd");
	if (!fx.ok || !odd.ok) return;
	{
		Scene sc;
		PagePool& pool = sc.AddPool(fx.model, 1);
		PKV_CHECK_EQ(pool.status, SSLM_OK);
		sslm_seq s = nullptr;
		sslm_prefix p = nullptr;
		// kills: a reservation below 2 pages (R without the +1), and a legacy create below a block.
		PKV_CHECK_EQ(sslm_seq_create_budgeted(fx.model, &pool.pool, 1, &s), SSLM_KV_POOL_EXHAUSTED);
		PKV_CHECK_EQ(sslm_prefix_begin_budgeted(fx.model, &pool.pool, 1, &p), SSLM_KV_POOL_EXHAUSTED);
		PKV_CHECK_EQ(sslm_seq_create(fx.model, &pool.pool, &s), SSLM_KV_POOL_EXHAUSTED);
		PKV_CHECK_EQ(sslm_prefix_begin(fx.model, &pool.pool, &p), SSLM_KV_POOL_EXHAUSTED);
	}
	{
		Scene sc;
		PagePool& pool = sc.AddPool(odd.model, 1);
		PKV_CHECK_EQ(pool.status, SSLM_OK);
		sslm_seq s = BudgetSeq(odd, &pool.pool, 512, sc.h);
		PKV_CHECK(s != nullptr);
		if (s) {
			PKV_CHECK_EQ(PrefillAll(odd.model, s, Stream(1, 100, odd.vocab), 64), SSLM_OK);
			PKV_CHECK_EQ(DecodeN(odd.model, s, 4, nullptr), SSLM_OK);
		}
		sslm_seq t = nullptr;
		PKV_CHECK_EQ(sslm_seq_create_budgeted(odd.model, &pool.pool, 1, &t), SSLM_KV_POOL_EXHAUSTED);
		if (s) ReleaseSeq(sc.h, s);
		PKV_CHECK(BudgetSeq(odd, &pool.pool, 1, sc.h) != nullptr);
	}
	{
		const uint32_t pages = static_cast<uint32_t>(fx.R(512));
		Scene sc;
		PagePool& pool = sc.AddPool(fx.model, pages);
		sslm_seq s = BudgetSeq(fx, &pool.pool, 512, sc.h);
		PKV_CHECK(s != nullptr);
		if (s) {
			PKV_CHECK_EQ(PrefillAll(fx.model, s, Stream(1, 1, fx.vocab), 1), SSLM_OK);
			sslm_status refusal = SSLM_OK;
			// kills: a reservation one page short of a full-budget run (an early SSLM_KV_POOL_EXHAUSTED).
			PKV_CHECK_EQ(TokensUntilRefusal(fx.model, s, &refusal), 512);
			PKV_CHECK_EQ(refusal, PKV_KV_BUDGET_EXCEEDED);
		}
		sslm_seq t = nullptr;
		PKV_CHECK_EQ(sslm_seq_create_budgeted(fx.model, &pool.pool, 1, &t), SSLM_KV_POOL_EXHAUSTED);
		if (s) ReleaseSeq(sc.h, s);
		ProbeExactlyFree(
		    fx,
		    [pages] {
			    auto s2 = std::make_unique<Scene>();
			    s2->AddPool(GetFixture("pkv_def").model, pages);
			    return std::unique_ptr<ProbeState>(std::move(s2));
		    },
		    pages);
	}
	{
		Scene sc;
		PagePool& pool = sc.AddPool(fx.model, static_cast<uint32_t>(fx.CapPages()));
		sslm_seq s = nullptr, t = nullptr;
		PKV_CHECK_EQ(sslm_seq_create(fx.model, &pool.pool, &s), SSLM_OK);
		if (s) sc.h.seqs.push_back(s);
		PKV_CHECK_EQ(sslm_seq_create(fx.model, &pool.pool, &t), SSLM_KV_POOL_EXHAUSTED);
		if (t) sc.h.seqs.push_back(t);
	}
	{
		const size_t pb = sslm_kv_page_size(fx.model);
		const size_t ov = sslm_kv_page_pool_overhead_size(fx.model, UINT32_MAX);
		PKV_CHECK(pb == fx.PageBytes() && pb <= SIZE_MAX / UINT32_MAX);
		PKV_CHECK(ov != SIZE_MAX && ov <= SIZE_MAX - pb * size_t{UINT32_MAX});  // kills: a saturating overhead at the u32 maximum
		AlignedBuf mem(SSLM_ABI_ALIGNMENT_BYTES);
		sslm_kv_pool p = reinterpret_cast<sslm_kv_pool>(&mem);
		PKV_CHECK_EQ(sslm_kv_page_pool_create(fx.model, mem.p, pb * size_t{UINT32_MAX} + ov - 1, UINT32_MAX, &p),
		             SSLM_BUFFER_TOO_SMALL);
		PKV_CHECK(p == nullptr);
	}
}

// ---- 4.7 [C5] -----------------------------------------------------------------------------------
// Toolchains and DotRow tiers. The cell runs the budget twins of lifecycle, prefix_lengths and widths
// and requires tokens and rows equal to v1.11.0's -- recorded on the reference build's own tier -- so
// any tier that runs it is held byte-equal to that one (6.3).
//
// How the legs run it. Toolchains: the same superslm_pkv_c5 on every leg (MSVC and clang-cl on the
// box, Linux clang and GCC in the cloud, macOS-ARM in CI when it runs). Tiers: the box/CI leg links
// superslm_pkv_c5's sources against each forced library of CMakeLists.txt (superslm_sse2_forced,
// superslm_avx2_forced, superslm_avx512_forced; x86-64 only, EXCLUDE_FROM_ALL, and each carries the
// bad-alloc seam and the matvec seams the injection library has) in place of superslm_test_injection,
// and runs "4.7/C5" once per binary. The forced libraries define SUPERSLM_FORCE_<TIER>_MATMUL PUBLIC,
// so the cell checks the binary really dispatches on the tier it was built for; on the default
// library it reports the CPUID tier it ran.
void Cell47() {
	using superslm::detail::GemmTier;
	const GemmTier tier = superslm::detail::ActiveGemmTier();
	static const char* const kNames[] = {"scalar", "sse2", "avx2", "avx512"};
	std::printf("  4.7: DotRow tier %s\n", kNames[static_cast<int>(tier) & 3]);
#if defined(SUPERSLM_FORCE_SSE2_MATMUL)
	PKV_CHECK(tier == GemmTier::kSse2);
#elif defined(SUPERSLM_FORCE_AVX2_MATMUL)
	PKV_CHECK(tier == GemmTier::kAvx2);
#elif defined(SUPERSLM_FORCE_AVX512_MATMUL)
	PKV_CHECK(tier == GemmTier::kAvx512);
#elif defined(SUPERSLM_FORCE_SCALAR_MATMUL)
	PKV_CHECK(tier == GemmTier::kScalar);
#endif
	const Fixture& fx = GetFixture("pkv_def");
	if (!fx.ok) return;
	PagePool pool(fx.model, static_cast<uint32_t>(fx.CapPages() + 8));
	PKV_CHECK_EQ(pool.status, SSLM_OK);
	// kills: a tier whose per-page attention runs (or DotRow tails at page edges) differ in any byte.
	BRun life(fx);
	BudgetLifecycle(life, &pool.pool, 128, 300);
	PKV_CHECK_MSG(life.fail.empty(), "4.7 lifecycle: %s", life.fail.c_str());
	ExpectMatchesReference("v1.11.0", fx, "lifecycle", life.records, Compare::kTokensAndRows);
	BRun lens(fx);
	BudgetPrefixLengths(lens, &pool.pool, 32);
	PKV_CHECK_MSG(lens.fail.empty(), "4.7 prefix lengths: %s", lens.fail.c_str());
	ExpectMatchesReference("v1.11.0", fx, "prefix_lengths", lens.records, Compare::kTokensAndRows);
	BRun widths(fx);
	BudgetWidths(widths, &pool.pool, ReferenceWidths(fx));
	PKV_CHECK_MSG(widths.fail.empty(), "4.7 widths: %s", widths.fail.c_str());
	ExpectMatchesReference("v1.11.0", fx, "widths", widths.records, Compare::kTokensAndRows);
}

PKV_CELL("4.1/C5", "C5", Cell41Budget);
PKV_CELL("4.2/C5", "C5", Cell42Budget);
PKV_CELL("4.3/C5", "C5", Cell43);
PKV_CELL("4.4/C5", "C5", Cell44Budget);
PKV_CELL("4.5/C5", "C5", Cell45);
PKV_CELL("4.7/C5", "C5", Cell47);

}  // namespace

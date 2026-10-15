// Paged-KV plan rev 16.21 §23.8 step T1, the page-size rule as a pure function and the pin's reach:
//
//   4.10/T1:grid          sslm_kv_page_positions over D {2, 48, 64, 128, 512, 1,024} x caps {8, 4,095,
//                         4,096, 4,097, 4,100, 8,192, 32,767, 32,768, 40,960} against the test's rule
//   4.10/T1:below-page    D = 2, cap 8: B = 8, one page per holder, and what holds there
//   4.10/T1:boundary      D = 64, caps 63, 64, 65, 127: the page condition's boundary row (L-7, R-1)
//   4.10/T1:extremes      the floor, the element width, the cap term's boundary, a non-power-of-two D
//   4.10/T1:hostile-d     D = 2^20 and 2^31 on CFG1-only fixtures (model gap: they do not map)
//   4.10/T1:generality    every §17.1 model row's (D, cap, elem)
//   4.10/T1:seam-refusals the seam's null out, null artifact and refused artifact (D-SLM8446), out untouched
//   4.11/T1               CFG1 kv_block_size in {1, 16, 32, 4,096, 0xFFFFFFFF} gives the rule's B; 0 refused
//   4.12/T1:refusals      the pin's refusals and hostile pins, the model left unchanged
//   4.12/T1:pins          a pin that takes: 16, 1 (the hand-built B = 1 geometry), the cap
//   4.10/T2:grid, :boundary, :extremes, :seam-tie, 4.11/T2
//                         the same sweeps' rows at caps the rule's B does not divide (grid and 4.11 at 4,095,
//                         4,097, 4,100, 32,767; boundary at 65 and 127; extremes at 4,097; the seam tie's pkv_odd):
//                         T2's partial last page, re-bracketed from T1 after the builder's T1 stop (T-3198 §5.1)
//
// Each named mutant's separation is executed at the cell's own construction: the mutant's value
// (r9::RuleB with the mutant) is required to differ from the asserted value at the row the plan
// names, so a mutant the construction cannot see fails the cell rather than passing silently.
// 2ebe5e3 itself is r9::RuleMutant::kShipped (CFG1's kv_block_size, else the cap), so a red run on
// 2ebe5e3 is that mutant's kill against a real build.

#include "pkv_r9.h"

// T-3195 (re-strike of rev 16.23, lesser plane): 4.10's seam tie compares the seam, which reads the unloaded artifact
// and so the derived B, with the verbs on fixtures loaded by pkv_common.h's GetFixture. That loader pins the
// generated fixtures to B = 16 only where PKV_PIN_GENERATED_FIXTURES is defined (the C4 and C5 suites, D-SLM8439);
// this executable must load them unpinned, and this guard makes a build that would pin them refuse to compile.
#if defined(PKV_PIN_GENERATED_FIXTURES)
#error "superslm_pkv_r9 loads the generated fixtures unpinned (4.10's seam tie, 4.13); PKV_PIN_GENERATED_FIXTURES is the C4 and C5 suites' alone"
#endif

namespace {

using namespace pkv;
using namespace pkv::r9;

// A fixture variant at `cap` (RoPE stripped: these cells read geometry only), mapped fresh.
struct At {
	Fixture fx;
	sslm_status map = SSLM_INVALID_ARGUMENT;
	At(const Fixture& base, int64_t cap, int64_t kvbs = -1, int64_t prec = -1) {
		Variant v;
		v.cap = cap;
		v.kv_block_size = kvbs;
		v.kv_precision = prec;
		v.strip_rope = true;
		MakeVariant(base, v, &fx, &map);
	}
};

// Where a cell's assertion is `value == expected`, the named mutant must give something else there.
void Separates(RuleMutant m, uint64_t D, uint64_t cap, uint64_t elem, int64_t expected, const char* row) {
	const int64_t got = RuleB(D, cap, elem, m);
	PKV_CHECK_MSG(got != expected,
	              "separation: the mutant '%s' gives B = %lld at %s, the asserted value; the cell cannot kill it",
	              RuleMutantName(m), static_cast<long long>(got), row);
}

// The step a row belongs to (T-3198 §5.1, re-bracketed 2026-10-09). A cap the rule's B does not divide needs T2's
// partial last page (ceil(cap/B) pages, the slack in the overhead, D-SLM8427); until T2 the engine keeps the
// one-page fallback (§23.8). So each cell below that sweeps caps runs twice: its T1 registration takes the dividing
// rows, its T2 registration the non-dividing ones, which stay red until T2.
enum class Rows { kDividing, kNonDividing };
bool Takes(Rows rows, uint64_t D, uint64_t cap, uint64_t elem) {
	const int64_t B = RuleB(D, cap, elem);
	const bool dividing = B > 0 && cap % static_cast<uint64_t>(B) == 0;
	return dividing == (rows == Rows::kDividing);
}

struct DBase {
	uint32_t D;
	const Fixture& (*get)();
};

const Fixture& GetDef() { return GetFixture("pkv_def"); }
const Fixture& GetQk() { return GetFixture("pkv_qk"); }
const Fixture& GetD64() { return D64Cap(4096); }

const DBase kBases[] = {{2, D2Cap8}, {48, GetDef}, {64, GetD64}, {128, GetQk}, {512, D512}, {1024, D1024}};
const int64_t kGridCaps[] = {8, 4095, 4096, 4097, 4100, 8192, 32767, 32768, 40960};

// 4.10 [T1 dividing caps, T2 non-dividing]. The grid: every value equals the test's own evaluation of §3.1's rule.
// The T2 rows (caps 4,095, 4,097, 4,100 and 32,767 at every D) overlap 4.10/T2:non-dividing at five (D, cap) pairs;
// both stay: the grid asserts B across the whole sweep, non-dividing asserts B, the page count and the overhead's slack
// at its named rows.
void Cell410GridRows(Rows rows) {
	for (const DBase& b : kBases) {
		const Fixture& base = b.get();
		if (!base.ok) continue;
		for (int64_t cap : kGridCaps) {
			if (!Takes(rows, b.D, static_cast<uint64_t>(cap), 1)) continue;
			At a(base, cap);
			PKV_CHECK_MSG(a.fx.ok, "D = %u at cap %lld: the variant did not map (%s)", b.D, static_cast<long long>(cap),
			              AbiStatusName(a.map));
			if (!a.fx.ok) continue;
			const int64_t want = RuleB(b.D, static_cast<uint64_t>(cap), 1);
			const int64_t got = sslm_kv_page_positions(a.fx.model);
			// kills: every rule mutant at some grid row (each is asserted separately below and in
			// :extremes); 2ebe5e3's rule reads kv_block_size 16 here
			PKV_CHECK_MSG(got == want, "D = %u, cap %lld: sslm_kv_page_positions = %lld, the rule gives %lld", b.D,
			              static_cast<long long>(cap), static_cast<long long>(got), static_cast<long long>(want));
		}
	}
	if (rows == Rows::kNonDividing) return;
	// The grid's own reach: 2ebe5e3's rule (kv_block_size 16, else the cap) differs at D = 64, cap 4,096.
	Separates(RuleMutant::kShipped, 64, 4096, 1, 64, "D = 64, cap 4,096");
	Separates(RuleMutant::kReadsCfg1, 48, 4096, 1, 128, "D = 48, cap 4,096");
}
void Cell410Grid() { Cell410GridRows(Rows::kDividing); }
void Cell410GridT2() { Cell410GridRows(Rows::kNonDividing); }

// 4.10 [T1]. Below one page: D = 2, cap 8 -> B = 8, one page per holder (D-SLM8412). The fixture's
// CFG1 kv_block_size is patched to 4, a divisor of 8, so 2ebe5e3's rule (which reads it) gives
// B = 4 and two pages; the rule ignores it (4.11).
void Cell410BelowPage() {
	const Fixture& base = D2Cap8();
	if (!base.ok) return;
	Fixture fx;
	Variant v;
	v.kv_block_size = 4;
	PKV_CHECK_MSG(MakeVariant(base, v, &fx), "r9_d2_c8 with kv_block_size 4 maps");
	if (!fx.ok) return;
	const int64_t cap = 8, B = RuleB(2, 8, 1);
	PKV_CHECK_EQ(B, 8);
	const Lay y = LayoutOf(fx.geo, 1, B);
	// The first bullet's two assertions: the no-clamp mutant keeps B = 2,048 and a 2,048-position page.
	PKV_CHECK_EQ(sslm_kv_page_positions(fx.model), 8);
	PKV_CHECK_EQ(sslm_kv_block_size(fx.model), y.bytes_per_position * 8);  // one 8-position page
	Separates(RuleMutant::kNoClamp, 2, 8, 1, 8, "D = 2, cap 8 (B)");
	PKV_CHECK_MSG(LayoutOf(fx.geo, 1, RuleB(2, 8, 1, RuleMutant::kNoClamp)).bytes_per_position *
	                      static_cast<size_t>(RuleB(2, 8, 1, RuleMutant::kNoClamp)) !=
	                  y.bytes_per_position * 8,
	              "separation: the no-clamp mutant's page bytes equal one 8-position page");
	// The four after them (they hold under the no-clamp mutant too, as the plan says):
	for (int32_t b = 1; b <= 8; ++b) PKV_CHECK_EQ(sslm_kv_pages_for_budget(fx.model, b), 1);
	const uint32_t N = 3;
	for (int32_t budget = 0; budget <= 8; ++budget) {
		PagePoolOf pool(fx.model, N, y.page_bytes);
		PKV_CHECK_EQ(pool.status, SSLM_OK);
		if (!pool.pool) continue;
		sslm_status refusal = SSLM_OK;
		const int n = CountCreates(fx.model, &pool.pool, budget, &refusal);
		PKV_CHECK_MSG(n == static_cast<int>(N) && refusal == SSLM_KV_POOL_EXHAUSTED,
		              "a pool of %u pages admits %d %s creates (budget %d), refused with %s; %u owed", N, n,
		              budget ? "budget" : "legacy", budget, AbiStatusName(refusal), N);
	}
	// A share adopt of a frozen 5-token prefix maps 0 shared pages and copies one page.
	{
		PagePoolOf pool(fx.model, 4, y.page_bytes);
		PKV_CHECK_EQ(pool.status, SSLM_OK);
		if (pool.pool) {
			sslm_prefix px = nullptr;
			sslm_seq s = nullptr;
			PKV_CHECK_EQ(sslm_prefix_begin_budgeted(fx.model, &pool.pool, 5, &px), SSLM_OK);
			if (px) PKV_CHECK_EQ(PrefixPrefillAll(fx.model, px, Stream(51, 5, fx.vocab), 5), SSLM_OK);
			if (px) PKV_CHECK_EQ(sslm_prefix_freeze(px), SSLM_OK);
			PKV_CHECK_EQ(sslm_seq_create_budgeted(fx.model, &pool.pool, 1, &s), SSLM_OK);
			if (px && s) {
				PKV_CHECK_EQ(sslm_seq_adopt_prefix(s, px), SSLM_OK);
				uint32_t pp = 0, sp = 0, pm = 0, sm = 0;
				PKV_CHECK_EQ(sslm_pkv_test_only_prefix_table_entry(px, 0, &pp, &pm), SSLM_OK);
				PKV_CHECK_EQ(sslm_pkv_test_only_seq_table_entry(s, 0, &sp, &sm), SSLM_OK);
				PKV_CHECK_EQ(sm, 1);
				PKV_CHECK_MSG(sp != pp, "the adopter maps the prefix's page %u: a share at one page", pp);
				// The fill probe: two holders of one page each leave exactly 2 of 4 pages.
				sslm_status refusal = SSLM_OK;
				PKV_CHECK_EQ(CountCreates(fx.model, &pool.pool, 1, &refusal), 2);
				sslm_seq_kv_stats_out st{};
				st.struct_size = sizeof st;
				if (sslm_seq_kv_stats(s, &st) == SSLM_OK)
					std::printf("  diagnostic: adopter shared_pages=%u mapped_private_pages=%u\n", st.shared_pages,
					            st.mapped_private_pages);
			}
			if (s) sslm_seq_release(s);
			if (px) sslm_prefix_release(px);
		}
	}
	// Tokens and K/V rows equal the flat twin at widths 1, 7 and 8 (8 = the cap: ready token only).
	auto e = EngineOf(fx);
	if (!e->ok) return;
	for (int64_t W : {int64_t{1}, int64_t{7}, cap}) {
		const pkv_engine::WidthRun twin = FlatTwin(*e, W);
		PKV_CHECK_MSG(twin.fail.empty(), "flat twin at W = %lld: %s", static_cast<long long>(W), twin.fail.c_str());
		PagePoolOf pool(fx.model, 1, y.page_bytes);
		PKV_CHECK_EQ(pool.status, SSLM_OK);
		if (!pool.pool) continue;
		sslm_seq s = nullptr;
		PKV_CHECK_EQ(sslm_seq_create(fx.model, &pool.pool, &s), SSLM_OK);
		if (!s) continue;
		const AbiWidth w = RunAbiWidth(fx.model, s, W, fx.vocab, W < cap);
		PKV_CHECK_EQ(w.prefill, SSLM_OK);
		PKV_CHECK_EQ(w.ready, twin.ready);
		if (W < cap) PKV_CHECK_EQ(w.decode1, twin.decode1);
		std::vector<uint8_t> rows;
		const int64_t L = W < cap ? W + 1 : W;
		PKV_CHECK_MSG(SeqRows(pool.pool, s, fx.geo, 1, B, L, Rot::kRotated, &rows) &&
		                  Sha(rows.data(), rows.size()) == (W < cap ? twin.rows_decode1 : twin.rows_prefill),
		              "W = %lld: the rows read from the page at B = 8 equal the flat twin's", static_cast<long long>(W));
		sslm_seq_release(s);
	}
}

// 4.10 [T1 caps 63 and 64, T2 caps 65 and 127]. The page condition's boundary (T-3178 G-4, R-1, D-SLM8412): D = 64,
// Int8, unpinned. Each fixture keeps the converter's kv_block_size 16, so 2ebe5e3 gives 63, 16, 65 and 127. Caps 65
// and 127 are two pages with slack (T2's partial last page), so they and their two separations are T2's.
void Cell410BoundaryRows(Rows sel) {
	struct Row { int cap; int64_t B, pages, slack, budget1_pages; };
	const Row rows[] = {{63, 63, 1, 0, 1}, {64, 64, 1, 0, 1}, {65, 64, 2, 63, 2}, {127, 64, 2, 1, 2}};
	for (const Row& r : rows) {
		if (!Takes(sel, 64, static_cast<uint64_t>(r.cap), 1)) continue;
		const Fixture& fx = D64Cap(r.cap);
		if (!fx.ok) continue;
		const int64_t B = RuleB(64, static_cast<uint64_t>(r.cap), 1);
		PKV_CHECK_EQ(B, r.B);
		const Lay y = LayoutOf(fx.geo, 1, B);
		PKV_CHECK_EQ(y.pages, r.pages);
		PKV_CHECK_EQ(y.slack, r.slack);
		PKV_CHECK_MSG(sslm_kv_page_positions(fx.model) == B, "cap %d: B = %lld, the rule gives %lld", r.cap,
		              static_cast<long long>(sslm_kv_page_positions(fx.model)), static_cast<long long>(B));
		// The block stays 1.11.0's; pages and slack show in the overhead (rev 16.22, D-SLM8427).
		PKV_CHECK_MSG(sslm_kv_block_size(fx.model) == y.block_bytes, "cap %d: sslm_kv_block_size = %zu, 1.11.0's %zu owed",
		              r.cap, sslm_kv_block_size(fx.model), y.block_bytes);
		PKV_CHECK_MSG(sslm_kv_pool_overhead_size(fx.model, 1) == y.Overhead(1),
		              "cap %d: sslm_kv_pool_overhead_size(1) = %zu; 40 + %lld slack positions of %zu bytes is %zu", r.cap,
		              sslm_kv_pool_overhead_size(fx.model, 1), static_cast<long long>(y.slack), y.bytes_per_position,
		              y.Overhead(1));
		// A budget-1 holder reserves min(ceil(1/B) + 1, pages): graded by admission in a 4-page pool.
		{
			PagePoolOf pool(fx.model, 4, y.page_bytes);
			PKV_CHECK_EQ(pool.status, SSLM_OK);
			if (pool.pool) {
				sslm_status refusal = SSLM_OK;
				const int n = CountCreates(fx.model, &pool.pool, 1, &refusal);
				PKV_CHECK_MSG(n == 4 / r.budget1_pages, "cap %d: a 4-page pool admits %d budget-1 creates, %lld owed",
				              r.cap, n, static_cast<long long>(4 / r.budget1_pages));
			}
		}
		// A whole-cap holder writes to the cap and is refused at it.
		{
			BlockPoolOf pool(fx.model, 1, y);
			PKV_CHECK_EQ(pool.status, SSLM_OK);
			sslm_seq s = nullptr;
			if (pool.pool) PKV_CHECK_EQ(sslm_seq_create(fx.model, &pool.pool, &s), SSLM_OK);
			if (s) {
				PKV_CHECK_EQ(PrefillAll(fx.model, s, Stream(41, r.cap, fx.vocab), 16), SSLM_OK);
				sslm_status st = SSLM_OK;
				PKV_CHECK(Next(fx.model, s, &st) >= 0);
				PKV_CHECK_EQ(Next(fx.model, s, &st), -1);
				PKV_CHECK_EQ(st, SSLM_CONTEXT_CAP_EXCEEDED);
				// kills: a page count rounded down (cap 127 holds one page; the write to 64 has none)
				std::vector<uint8_t> rows;
				PKV_CHECK_MSG(SeqRows(pool.pool, s, fx.geo, 1, B, r.cap, Rot::kRotated, &rows),
				              "cap %d: the table covers all %d written positions at B = %lld", r.cap, r.cap,
				              static_cast<long long>(B));
				sslm_seq_release(s);
			}
		}
		// The adopt at the boundary: cap 64 copies the one page; cap 65 shares one full page.
		if (r.cap == 64 || r.cap == 65) {
			const int32_t len = r.cap == 64 ? 5 : 64;
			PagePoolOf pool(fx.model, 6, y.page_bytes);
			PKV_CHECK_EQ(pool.status, SSLM_OK);
			if (!pool.pool) continue;
			sslm_prefix px = nullptr;
			sslm_seq s = nullptr;
			PKV_CHECK_EQ(sslm_prefix_begin_budgeted(fx.model, &pool.pool, len, &px), SSLM_OK);
			if (px) PKV_CHECK_EQ(PrefixPrefillAll(fx.model, px, Stream(51, len, fx.vocab), 16), SSLM_OK);
			if (px) PKV_CHECK_EQ(sslm_prefix_freeze(px), SSLM_OK);
			PKV_CHECK_EQ(sslm_seq_create_budgeted(fx.model, &pool.pool, 1, &s), SSLM_OK);
			if (px && s) {
				PKV_CHECK_EQ(sslm_seq_adopt_prefix(s, px), SSLM_OK);
				uint32_t pp = 0, sp = 0, pm = 0, sm = 0;
				PKV_CHECK_EQ(sslm_pkv_test_only_prefix_table_entry(px, 0, &pp, &pm), SSLM_OK);
				PKV_CHECK_EQ(sslm_pkv_test_only_seq_table_entry(s, 0, &sp, &sm), SSLM_OK);
				if (r.cap == 64) {
					// claim (3) does not hold at exactly one page: 0 shared pages, the whole block copied
					PKV_CHECK_MSG(sp != pp, "cap 64: the adopter maps the prefix's page (a share at one page)");
				} else {
					// cap 65: the 64-token prefix's one full page is shared; nothing is copied
					PKV_CHECK_MSG(sp == pp, "cap 65: the adopter's page 0 is %u, the prefix's is %u (not shared)", sp, pp);
					PKV_CHECK_EQ(sm, 1);  // the shared page only: no tail page was copied
				}
			}
			if (s) sslm_seq_release(s);
			if (px) sslm_prefix_release(px);
		}
	}
	if (sel == Rows::kDividing) return;
	// Mutants: a clamp at 2B > cap gives cap 65 B = 65 and one page; floor(cap/B) gives cap 127 one page.
	Separates(RuleMutant::kClampAt2B, 64, 65, 1, 64, "cap 65");
	PKV_CHECK_MSG(CapPages(127, 64, true) != 2, "separation: the floor page count gives 2 pages at cap 127");
}
void Cell410Boundary() { Cell410BoundaryRows(Rows::kDividing); }
void Cell410BoundaryT2() { Cell410BoundaryRows(Rows::kNonDividing); }

// 4.10 [T1; the cap-4,097 row T2]. The floor, the element width, the cap term's boundary and a non-power-of-two D.
void Cell410ExtremesRows(Rows sel) {
	struct Row { const Fixture& (*base)(); uint32_t D; int64_t cap; int64_t prec; int64_t want; RuleMutant kills; };
	const Row rows[] = {
	    {D1024, 1024, 4096, -1, 16, RuleMutant::kNoFloor},   // the floor
	    {D1024, 1024, 32768, -1, 16, RuleMutant::kNoFloor},
	    {D512, 512, 4096, -1, 16, RuleMutant::kNone},        // 16 with or without the floor: a grid row
	    {GetD64, 64, 4096, 1, 32, RuleMutant::kNoElem},      // Int16 (elem 2)
	    {GetD64, 64, 4096, -1, 64, RuleMutant::kNoCapTerm},  // the cap term
	    {GetD64, 64, 4097, -1, 128, RuleMutant::kNone},
	    {GetD64, 64, 8192, -1, 128, RuleMutant::kNone},
	    {GetDef, 48, 4096, -1, 128, RuleMutant::kNoPow2},    // a non-power-of-two D
	};
	for (const Row& r : rows) {
		if (!Takes(sel, r.D, static_cast<uint64_t>(r.cap), r.prec == 1 ? 2 : 1)) continue;
		const Fixture& base = r.base();
		if (!base.ok) continue;
		At a(base, r.cap, -1, r.prec);
		PKV_CHECK_MSG(a.fx.ok, "D = %u, cap %lld, kv_precision %lld: the variant did not map (%s)", r.D,
		              static_cast<long long>(r.cap), static_cast<long long>(r.prec), AbiStatusName(a.map));
		if (!a.fx.ok) continue;
		const uint64_t elem = r.prec == 1 ? 2 : 1;
		PKV_CHECK_EQ(RuleB(r.D, static_cast<uint64_t>(r.cap), elem), r.want);
		const int64_t got = sslm_kv_page_positions(a.fx.model);
		PKV_CHECK_MSG(got == r.want, "D = %u, cap %lld, elem %llu: B = %lld, %lld owed", r.D,
		              static_cast<long long>(r.cap), static_cast<unsigned long long>(elem), static_cast<long long>(got),
		              static_cast<long long>(r.want));
		if (r.kills != RuleMutant::kNone) Separates(r.kills, r.D, static_cast<uint64_t>(r.cap), elem, r.want, "its row");
		if (r.cap == 4097)
			PKV_CHECK_MSG(sslm_kv_pool_overhead_size(a.fx.model, 1) == LayoutOf(a.fx.geo, 1, 128).Overhead(1),
			              "cap 4,097: 33 pages of 128 positions, so the overhead carries 127 slack positions");
	}
	if (sel == Rows::kNonDividing) return;
	// The 32-bit product wraps at D = 2^31 (the hostile row, model gap below); its separation:
	Separates(RuleMutant::kU32Product, uint64_t{1} << 31, 4096, 1, 16, "D = 2^31");
}
void Cell410Extremes() { Cell410ExtremesRows(Rows::kDividing); }
void Cell410ExtremesT2() { Cell410ExtremesRows(Rows::kNonDividing); }

// 4.10 [T1]. A hostile head_dim, D = 2^20 and 2^31 (both even, so the loader's CFG1 checks admit
// them), on the CFG1-and-sigmoid-only fixture the plan specifies; B is 16 with no 32-bit wrap. Rev 16.23
// (D-SLM8437, T-3189 G-1): neither fixture maps (sslm_model_map builds the engine cache from WGT1 weights they
// do not carry), so B is read through the geometry seam over the loaded config.
superslm_test::Cfg1Spec SmallCfg1(uint32_t layers, uint32_t heads, uint32_t kv_heads, uint32_t D, uint32_t cap,
                                  uint32_t precision) {
	superslm_test::Cfg1Spec spec;
	spec.hidden_size = 2;
	spec.num_hidden_layers = layers;
	spec.num_attention_heads = heads;
	spec.num_key_value_heads = kv_heads;
	spec.head_dim = D;
	spec.intermediate_size = 2;
	spec.vocab_size = 256;
	spec.context_cap = cap;
	spec.kv_precision = precision;
	spec.kv_block_size = 16;
	return spec;
}

bool SeamOf(const std::vector<uint8_t>& artifact, uint32_t blocks, KvPagesTestPoolGeometryOut* out, const char* what) {
	*out = KvPagesTestPoolGeometryOut{};
	const sslm_status st = KvPagesTestPoolGeometry(artifact.data(), artifact.size(), blocks, out);
	PKV_CHECK_MSG(st == SSLM_OK, "%s: KvPagesTestPoolGeometry = %s", what, AbiStatusName(st));
	return st == SSLM_OK;
}

void Cell410HostileD() {
	for (uint64_t D : {uint64_t{1} << 20, uint64_t{1} << 31}) {
		const std::vector<uint8_t> b = Cfg1OnlyArtifact(SmallCfg1(2, 1, 1, static_cast<uint32_t>(D), 4096, 0));
		std::string err;
		const superslm::SslmModelStatus ls = LoadStatus(b, &err);
		PKV_CHECK_MSG(ls == superslm::SslmModelStatus::Ok, "D = %llu: the loader refuses the fixture: %s (%s)",
		              static_cast<unsigned long long>(D), superslm::SslmModelStatusName(ls), err.c_str());
		KvPagesTestPoolGeometryOut g;
		const std::string what = "D = " + std::to_string(D);
		if (SeamOf(b, 1, &g, what.c_str()))
			PKV_CHECK_MSG(g.page_positions == 16, "%s: B = %lld through the seam, 16 owed", what.c_str(),
			              static_cast<long long>(g.page_positions));
		PKV_CHECK_EQ(RuleB(D, 4096, 1), 16);
	}
}

// 4.10 [T1]. Generality: the rule's B for every §17.1 model row, at its game cap 4,096 and its
// max_position_embeddings cap. The rule reads (D, cap, elem) only; each row is carried by a fixture
// with that D, cap and element width (layers and KV heads do not enter the rule). The plan specifies
// CFG1-only fixtures, which do not map (the gap of :hostile-d), so the rows run on mapped fixtures.
void Cell410Generality() {
	struct Row { const char* model; const Fixture& (*base)(); uint32_t D; int64_t cap; int64_t want; };
	const Row rows[] = {
	    {"Qwen2.5-0.5B", GetD64, 64, 4096, 64},         {"Qwen2.5-0.5B", GetD64, 64, 32768, 128},
	    {"Qwen2.5-1.5B", GetQk, 128, 4096, 64},         {"Qwen2.5-1.5B", GetQk, 128, 32768, 64},
	    {"Qwen2.5-3B", GetQk, 128, 32768, 64},          {"Qwen2.5-7B", GetQk, 128, 32768, 64},
	    {"Qwen3-0.6B/1.7B/4B/8B", GetQk, 128, 40960, 64}, {"Qwen3-0.6B/1.7B/4B/8B", GetQk, 128, 4096, 64},
	};
	for (const Row& r : rows) {
		const Fixture& base = r.base();
		if (!base.ok) continue;
		At a(base, r.cap);
		if (!a.fx.ok) {
			PKV_CHECK_MSG(false, "%s row (D %u, cap %lld) did not map", r.model, r.D, static_cast<long long>(r.cap));
			continue;
		}
		PKV_CHECK_EQ(RuleB(r.D, static_cast<uint64_t>(r.cap), 1), r.want);
		PKV_CHECK_MSG(sslm_kv_page_positions(a.fx.model) == r.want, "%s (D %u, cap %lld): B = %lld, %lld owed", r.model,
		              r.D, static_cast<long long>(r.cap), static_cast<long long>(sslm_kv_page_positions(a.fx.model)),
		              static_cast<long long>(r.want));
	}
	// Rev 16.23 (D-SLM8437): the same rows as CFG1-only fixtures at each model's own layers and heads, through the
	// geometry seam; the page bytes follow from them, so the seam's page_size is asserted too.
	struct Cfg { const char* model; uint32_t L, Hq, Hkv, D, cap; int64_t want; };
	const Cfg cfgs[] = {
	    {"Qwen2.5-0.5B", 24, 14, 2, 64, 4096, 64},  {"Qwen2.5-0.5B", 24, 14, 2, 64, 32768, 128},
	    {"Qwen2.5-1.5B", 28, 12, 2, 128, 4096, 64}, {"Qwen2.5-1.5B", 28, 12, 2, 128, 32768, 64},
	    {"Qwen2.5-3B", 36, 16, 2, 128, 4096, 64},   {"Qwen2.5-3B", 36, 16, 2, 128, 32768, 64},
	    {"Qwen2.5-7B", 28, 28, 4, 128, 4096, 64},   {"Qwen2.5-7B", 28, 28, 4, 128, 32768, 64},
	    {"Qwen3-0.6B/1.7B", 28, 16, 8, 128, 4096, 64}, {"Qwen3-0.6B/1.7B", 28, 16, 8, 128, 40960, 64},
	    {"Qwen3-4B/8B", 36, 32, 8, 128, 4096, 64},  {"Qwen3-4B/8B", 36, 32, 8, 128, 40960, 64},
	};
	for (const Cfg& c : cfgs) {
		const std::vector<uint8_t> b = Cfg1OnlyArtifact(SmallCfg1(c.L, c.Hq, c.Hkv, c.D, c.cap, 0));
		KvPagesTestPoolGeometryOut g;
		const std::string what = std::string(c.model) + " at cap " + std::to_string(c.cap);
		if (!SeamOf(b, 1, &g, what.c_str())) continue;
		PKV_CHECK_EQ(RuleB(c.D, c.cap, 1), c.want);
		PKV_CHECK_MSG(g.page_positions == c.want && g.page_size == size_t{c.L} * 2 * c.Hkv * c.D * static_cast<size_t>(c.want),
		              "%s: B %lld and page %zu through the seam; %lld and %zu owed", what.c_str(),
		              static_cast<long long>(g.page_positions), g.page_size, static_cast<long long>(c.want),
		              size_t{c.L} * 2 * c.Hkv * c.D * static_cast<size_t>(c.want));
	}
}

// 4.10 [T1], rev 16.23 (D-SLM8437, T-3189 G-1): the seam tie. On every fixture the suite maps, the seam's values
// equal the verbs' at block_count 1, 343 and the 2.9b and 2.9c thresholds: page_positions, page_size,
// block_size and overhead equal the verbs; page_count_fits and requirement_fits together equal the create's
// verdict on a buffer smaller than any requirement (SSLM_BUFFER_TOO_SMALL when both fit, SSLM_INVALID_ARGUMENT
// otherwise), and requirement_fits equals the verbs' own block and overhead summed in the test. Mutant: a seam
// with its own copy of the sizing that omits the slack term, which fails at pkv_odd's overhead (96,808 against 40).
// Re-bracketed (T-3198 §5.1): pkv_odd (cap 4,100, B = 256, 17 pages) is T2's partial last page, so its member, and the
// slack-omitting copy's separation that lives there, are 4.10/T2:seam-tie; the other nine fixtures divide.
struct Tied { std::string what; const Fixture* fx; };
void TieOne(const Tied& f);
void Cell410SeamTie() {
	std::vector<Tied> fixtures;
	for (const char* stem : {"pkv_def", "pkv_32k", "pkv_qk"}) fixtures.push_back({stem, &GetFixture(stem)});
	fixtures.push_back({"r9_d2_c8", &D2Cap8()});
	fixtures.push_back({"r9_d48_c100", &Get("r9_d48_c100", Geometry{2, 2, 48, 100})});
	fixtures.push_back({"r9_d64_c4096", &D64Cap(4096)});
	fixtures.push_back({"r9_d512_c64", &D512()});
	fixtures.push_back({"r9_d1024_c64", &D1024()});
	static Fixture q8big;  // 4.14's Qwen3-8B K/V shape, Int16, cap 40,960
	if (!q8big.ok && Q8().ok) {
		Variant v;
		v.cap = 40960;
		v.kv_precision = 1;
		v.strip_rope = true;
		MakeVariant(Q8(), v, &q8big);
	}
	fixtures.push_back({"r9_q8 Int16 cap 40,960", &q8big});
	for (const Tied& f : fixtures) TieOne(f);
}
void Cell410SeamTieT2() {
	TieOne({"pkv_odd", &GetFixture("pkv_odd")});
	// separation of the slack-omitting copy at pkv_odd (D = 48, cap 4,100, B = 256: 252 slack positions of 384 bytes)
	const Lay odd = LayoutOf(Geometry{2, 2, 48, 4100}, 1, 256);
	PKV_CHECK(odd.Overhead(1) == 96808u && 8u + 32u != odd.Overhead(1));
}
void TieOne(const Tied& f) {
	{
		// like with like: every fixture here is at the derived B (the seam's), never the C4 loader's pinned 16
		if (f.fx->ok)
			PKV_CHECK_MSG(sslm_kv_page_positions(f.fx->model) ==
			                  RuleB(f.fx->geo.head_dim, static_cast<uint64_t>(f.fx->geo.context_cap), f.what.find("Int16") != std::string::npos ? 2 : 1),
			              "%s is not at the rule's B (a pinned fixture would make the tie compare unlike with unlike)", f.what.c_str());
		if (!f.fx->ok) {
			PKV_CHECK_MSG(false, "%s did not map", f.what.c_str());
			return;
		}
		const std::vector<uint8_t> bytes(f.fx->bytes.p, f.fx->bytes.p + f.fx->size);
		const int64_t B = sslm_kv_page_positions(f.fx->model);
		const uint64_t pages = B > 0 ? static_cast<uint64_t>(CeilDiv(f.fx->geo.context_cap, B)) : 1;
		for (uint32_t n : {1u, 343u, static_cast<uint32_t>(UINT32_MAX / pages), 252645135u}) {
			KvPagesTestPoolGeometryOut g;
			const std::string what = f.what + " at " + std::to_string(n) + " blocks";
			if (!SeamOf(bytes, n, &g, what.c_str())) continue;
			const size_t block = sslm_kv_block_size(f.fx->model), overhead = sslm_kv_pool_overhead_size(f.fx->model, n);
			PKV_CHECK_MSG(g.page_positions == B && g.page_size == sslm_kv_page_size(f.fx->model) && g.block_size == block &&
			                  g.overhead == overhead,
			              "%s: seam (B %lld, page %zu, block %zu, overhead %zu) against the verbs (%lld, %zu, %zu, %zu)",
			              what.c_str(), static_cast<long long>(g.page_positions), g.page_size, g.block_size, g.overhead,
			              static_cast<long long>(B), sslm_kv_page_size(f.fx->model), block, overhead);
			// A buffer smaller than this requirement (T-3198 §5.2: a fixed 4,096 bytes is not, at r9_d2_c8's 168): one
			// byte below it when it fits in 4,096 bytes, else the whole 4,096. Every size below the requirement.
			AlignedBuf tiny(4096);
			const bool req_fits = overhead != SIZE_MAX && block <= (SIZE_MAX - overhead) / n;
			const size_t req = req_fits ? block * n + overhead : SIZE_MAX;
			const size_t below = req_fits && req <= tiny.n ? req - 1 : tiny.n;
			sslm_kv_pool pool = nullptr;
			const sslm_status st = sslm_kv_pool_create(f.fx->model, tiny.p, below, n, &pool);
			if (pool) sslm_kv_pool_destroy(pool);
			PKV_CHECK_MSG((g.page_count_fits && g.requirement_fits) == (st == SSLM_BUFFER_TOO_SMALL),
			              "%s: seam fits (pages %d, requirement %d) against the create's %s", what.c_str(), g.page_count_fits,
			              g.requirement_fits, AbiStatusName(st));
			const bool sum_fits = overhead != SIZE_MAX && block <= (SIZE_MAX - overhead) / n;
			PKV_CHECK_MSG((g.requirement_fits != 0) == sum_fits, "%s: requirement_fits %d, the verbs' sum fits %d",
			              what.c_str(), g.requirement_fits, sum_fits ? 1 : 0);
		}
	}
}

// 4.11 [T1 dividing caps; T2 the four non-dividing caps]. Header independence: kv_block_size in {1, 16, 32, 4,096,
// 0xFFFFFFFF} gives the rule's B at each (D, cap) of the grid, and 0 is still refused at load (T1).
void Cell411Rows(Rows sel) {
	const uint32_t kvbs[] = {1, 16, 32, 4096, 0xFFFFFFFFu};
	for (const DBase& b : kBases) {
		const Fixture& base = b.get();
		if (!base.ok) continue;
		for (int64_t cap : kGridCaps) {
			if (!Takes(sel, b.D, static_cast<uint64_t>(cap), 1)) continue;
			const int64_t want = RuleB(b.D, static_cast<uint64_t>(cap), 1);
			for (uint32_t k : kvbs) {
				At a(base, cap, k);
				if (!a.fx.ok) {
					PKV_CHECK_MSG(false, "D %u cap %lld kv_block_size %u did not map (%s)", b.D,
					              static_cast<long long>(cap), k, AbiStatusName(a.map));
					continue;
				}
				const int64_t got = sslm_kv_page_positions(a.fx.model);
				if (got != want)
					PKV_CHECK_MSG(false, "D %u, cap %lld, kv_block_size %u: B = %lld, the rule gives %lld", b.D,
					              static_cast<long long>(cap), k, static_cast<long long>(got),
					              static_cast<long long>(want));
				else
					PKV_CHECK(true);
			}
		}
		if (sel == Rows::kNonDividing) continue;
		At zero(base, 4096, 0);
		PKV_CHECK_MSG(!zero.fx.ok && zero.map == SSLM_ARTIFACT_REJECTED,
		              "D %u: kv_block_size 0 maps with %s; the loader must refuse it", b.D, AbiStatusName(zero.map));
	}
	if (sel == Rows::kDividing) Separates(RuleMutant::kReadsCfg1, 64, 4096, 1, 64, "D = 64, cap 4,096, kv_block_size 32");
}
void Cell411() { Cell411Rows(Rows::kDividing); }
void Cell411T2() { Cell411Rows(Rows::kNonDividing); }

// 4.12 [T1]. The pin's refusals in the test library, each leaving the model's B unchanged.
void Cell412Refusals() {
	const Fixture& def = GetFixture("pkv_def");
	if (!def.ok) return;
	const int64_t rule = RuleB(48, 4096, 1);  // 128
	// A pin after a pool exists is refused.
	{
		At a(def, 4096);
		if (a.fx.ok) {
			LegacyPool pool(a.fx.model, 1);
			PKV_CHECK_EQ(pool.status, SSLM_OK);
			PKV_CHECK_EQ(KvPagesTestPinPagePositions(a.fx.model, 16), SSLM_INVALID_ARGUMENT);
			PKV_CHECK_EQ(sslm_kv_page_positions(a.fx.model), rule);
		}
	}
	// A B that neither divides the cap nor is the cap (the cap-4,100 case is 4.12/T2:cap4100).
	{
		At a(def, 4096);
		if (a.fx.ok) {
			PKV_CHECK_EQ(KvPagesTestPinPagePositions(a.fx.model, 48), SSLM_INVALID_ARGUMENT);
			PKV_CHECK_EQ(sslm_kv_page_positions(a.fx.model), rule);
		}
	}
	// Hostile pins (G9): 0, -1, cap + 1, 2^31; the model is left unchanged.
	for (int64_t pin : {int64_t{0}, int64_t{-1}, int64_t{4097}, int64_t{1} << 31}) {
		At a(def, 4096);
		if (!a.fx.ok) continue;
		PKV_CHECK_MSG(KvPagesTestPinPagePositions(a.fx.model, pin) == SSLM_INVALID_ARGUMENT, "pin %lld is not refused",
		              static_cast<long long>(pin));
		PKV_CHECK_MSG(sslm_kv_page_positions(a.fx.model) == rule, "after the refused pin %lld, B = %lld, not the rule's %lld",
		              static_cast<long long>(pin), static_cast<long long>(sslm_kv_page_positions(a.fx.model)),
		              static_cast<long long>(rule));
	}
}

// 4.12 [T1]. A pin that takes: 16 (every generated-fixture cell's B), 1 (the hand-built fixtures'
// B = 1 geometry, G36; see the model gap in the record: the 21 hand-built specs are engine-level
// fixtures with no sslm_model, so the cell pins pkv_def to 1 as cell 4.9 stands in for them), and
// the cap. Without the pin the model reports the rule's B (128), not the pinned value.
void Cell412Pins() {
	const Fixture& def = GetFixture("pkv_def");
	if (!def.ok) return;
	for (int64_t pin : {int64_t{16}, int64_t{1}, int64_t{4096}, int64_t{4}}) {
		At a(def, 4096, 32);  // kv_block_size 32, so 2ebe5e3's own reading cannot supply the pinned value
		if (!a.fx.ok) continue;
		PKV_CHECK_MSG(sslm_kv_page_positions(a.fx.model) == 128, "unpinned: B = %lld, the rule's is 128",
		              static_cast<long long>(sslm_kv_page_positions(a.fx.model)));
		PKV_CHECK_MSG(KvPagesTestPinPagePositions(a.fx.model, pin) == SSLM_OK, "the pin to %lld is refused",
		              static_cast<long long>(pin));
		PKV_CHECK_MSG(sslm_kv_page_positions(a.fx.model) == pin, "after the pin to %lld, B = %lld",
		              static_cast<long long>(pin), static_cast<long long>(sslm_kv_page_positions(a.fx.model)));
		const Lay y = LayoutOf(a.fx.geo, 1, pin);
		PKV_CHECK_EQ(sslm_kv_page_size(a.fx.model), y.page_bytes);
	}
}

// The geometry seam's refusals (plan §25.6, §25.7 dimension 2, §26.5 N-3 / D-SLM8446; T-3196 F-8). A null `out` or a null `artifact` returns
// SSLM_INVALID_ARGUMENT; an artifact the loader refuses (an odd head_dim, model.cpp's parity check) returns the
// load's refusal, read as the status sslm_model_map gives the same bytes (SSLM_ARTIFACT_REJECTED), and writes nothing
// through `out`. The positive control, the same call over an admitted artifact returning SSLM_OK, is what keeps a seam
// that refuses everything (2ebe5e3's stub) from passing the two refusals: it is this cell's red reason there.
void Cell410SeamRefusals() {
	const std::vector<uint8_t> good = Cfg1OnlyArtifact(SmallCfg1(2, 1, 1, 64, 4096, 0));
	const std::vector<uint8_t> odd = Cfg1OnlyArtifact(SmallCfg1(2, 1, 1, 47, 4096, 0));
	std::string err;
	PKV_CHECK_MSG(LoadStatus(good, &err) == superslm::SslmModelStatus::Ok, "the admitted artifact: %s", err.c_str());
	PKV_CHECK_MSG(LoadStatus(odd, &err) == superslm::SslmModelStatus::BadConfigHeadDimParity,
	              "the loader does not refuse head_dim 47 on parity: %s", err.c_str());
	// the status the map gives the refused bytes (the reading the seam's refusal is held to)
	sslm_status map_status = SSLM_OK;
	{
		AlignedBuf b(odd.size(), 0);
		std::memcpy(b.p, odd.data(), odd.size());
		sslm_model m = nullptr;
		map_status = sslm_model_map(b.p, odd.size(), &m);
		if (m) sslm_model_unmap(m);
	}
	PKV_CHECK_EQ(map_status, SSLM_ARTIFACT_REJECTED);
	// positive control
	KvPagesTestPoolGeometryOut g;
	PKV_CHECK_MSG(SeamOf(good, 1, &g, "the admitted artifact (D = 64, cap 4,096)") && g.page_positions == 64,
	              "the seam over an admitted artifact: B = %lld, the rule's 64 owed", static_cast<long long>(g.page_positions));
	// a null out, and a null artifact (D-SLM8446: both SSLM_INVALID_ARGUMENT, checked before the load, out untouched)
	PKV_CHECK_EQ(KvPagesTestPoolGeometry(good.data(), good.size(), 1, nullptr), SSLM_INVALID_ARGUMENT);
	{
		KvPagesTestPoolGeometryOut s0;
		std::memset(&s0, 0xA5, sizeof s0);
		KvPagesTestPoolGeometryOut o0 = s0;
		PKV_CHECK_EQ(KvPagesTestPoolGeometry(nullptr, good.size(), 1, &o0), SSLM_INVALID_ARGUMENT);
		PKV_CHECK_MSG(std::memcmp(&o0, &s0, sizeof o0) == 0, "a null artifact: the seam wrote through out");
	}
	// a refused artifact: the load's refusal, and nothing written
	KvPagesTestPoolGeometryOut sentinel;
	std::memset(&sentinel, 0xA5, sizeof sentinel);
	KvPagesTestPoolGeometryOut out = sentinel;
	const sslm_status st = KvPagesTestPoolGeometry(odd.data(), odd.size(), 1, &out);
	PKV_CHECK_MSG(st == map_status, "head_dim 47: the seam returned %s; the map's %s owed", AbiStatusName(st),
	              AbiStatusName(map_status));
	PKV_CHECK_MSG(std::memcmp(&out, &sentinel, sizeof out) == 0, "head_dim 47: the refusing seam wrote through out");
	// a truncated image: the load refuses it too, the same way
	out = sentinel;
	const sslm_status tr = KvPagesTestPoolGeometry(good.data(), good.size() / 2, 1, &out);
	PKV_CHECK_MSG(tr != SSLM_OK && std::memcmp(&out, &sentinel, sizeof out) == 0,
	              "a truncated image: the seam returned %s and %s through out", AbiStatusName(tr),
	              std::memcmp(&out, &sentinel, sizeof out) == 0 ? "wrote nothing" : "wrote");
}

PKV_CELL("4.10/T1:grid", "T1", Cell410Grid);
PKV_CELL("4.10/T1:below-page", "T1", Cell410BelowPage);
PKV_CELL("4.10/T1:boundary", "T1", Cell410Boundary);
PKV_CELL("4.10/T1:extremes", "T1", Cell410Extremes);
PKV_CELL("4.10/T1:hostile-d", "T1", Cell410HostileD);
PKV_CELL("4.10/T1:generality", "T1", Cell410Generality);
PKV_CELL("4.10/T1:seam-tie", "T1", Cell410SeamTie);
PKV_CELL("4.10/T1:seam-refusals", "T1", Cell410SeamRefusals);
PKV_CELL("4.11/T1", "T1", Cell411);
PKV_CELL("4.12/T1:refusals", "T1", Cell412Refusals);
PKV_CELL("4.12/T1:pins", "T1", Cell412Pins);
// the non-dividing rows of the T1 sweeps, re-bracketed (T-3198 §5.1): red until T2's partial last page
PKV_CELL("4.10/T2:grid", "T2", Cell410GridT2);
PKV_CELL("4.10/T2:boundary", "T2", Cell410BoundaryT2);
PKV_CELL("4.10/T2:extremes", "T2", Cell410ExtremesT2);
PKV_CELL("4.10/T2:seam-tie", "T2", Cell410SeamTieT2);
PKV_CELL("4.11/T2", "T2", Cell411T2);

}  // namespace

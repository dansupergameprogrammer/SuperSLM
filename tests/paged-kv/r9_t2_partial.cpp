// Paged-KV plan rev 16.21 §23.8 step T2, the fallback's removal: the partial last page (ceil(cap/B)
// pages, the slack never addressed), the u32 threshold and the block product on ceil, the state size
// at a non-dividing cap. Fixture: pkv_odd (D = 48, cap 4,100), unpinned: B = 256, 17 pages, 252
// positions of slack. On 2ebe5e3 its B is the cap (one page), so every cell is red there.
//
//   4.4/T2              legacy tokens and rows at 4.1's widths, cap - 1 and cap; budget reservations
//   4.10/T2:non-dividing  4,095 -> 64 (D 64), 4,097 -> 128, 4,100 -> 128 (D 64) and 256 (D 48),
//                       32,767 -> 64 (D 128), each with ceil(cap/B) pages
//   4.12/T2:cap4100     a pin of 16 at cap 4,100 is refused; B stays 256
//   2.9/T2              pinned B = 16 at cap 32,768: 2,097,152 blocks refused before the buffer check
//   2.9b/T2             unpinned thresholds: D = 64 at cap 32,768 (B = 128) and D = 128 (B = 64)
//   2.9c/T2             pkv_odd: 252,645,135 blocks passes the page check, 252,645,136 is refused
//   2.9d/T2             the block product's overflow at a non-dividing cap, through the geometry seam
//                       (rev 16.23, D-SLM8437: the CFG1-only fixture loads and does not map)
//   10.9/T2             admission at a non-dividing cap, the writes to the cap, save, adopt, state size
//   10.9/T2:batch       two page holders in one 34-page pool through the partial last page and the cap refusal,
//                       against the n = 1 twin: 4.4/T3:batch's attribution control (rev 16.24, D-SLM8441)
//   4.13/T2:cap4100     4.2 (prefix lengths) whole on pkv_odd at B = 256
//   6.7/T2:non-dividing a pin of 4 -> the derived B = 256 page pool, the 17-page reservation, the legacy size
//   (rev 16.22: 2.4, 7.11, 9.2b and 9.3 at cap 4,100 are in r9_t2_saves.cpp)

#include "pkv_r9.h"

namespace {

using namespace pkv;
using namespace pkv::r9;

const char* const kAllScenarios[] = {"lifecycle", "long_prefill", "shared_preamble", "persist",
                                     "saturating", "widths", "prefix_lengths"};

int64_t OddB() { return RuleB(48, 4100, 1); }  // 256

void RequireOddB(const Fixture& odd) {
	PKV_CHECK_MSG(sslm_kv_page_positions(odd.model) == 256, "pkv_odd runs at B = %lld; the rule gives 256 with 17 pages",
	              static_cast<long long>(sslm_kv_page_positions(odd.model)));
}

// 4.4 [T2]. Legacy: tokens and K/V rows equal the v1.11.0 reference in every scenario (4.1's widths,
// cap - 1 and cap are the `widths` scenario's), rows read through the peek at B = 256 rotated.
// Budget: the reservation R(budget) = min(ceil(budget/B) + 1, 17), graded by admission.
void Cell44() {
	const Fixture& odd = GetFixture("pkv_odd");
	if (!odd.ok) return;
	RequireOddB(odd);
	const int64_t B = OddB();
	const Lay y = LayoutOf(odd.geo, 1, B);
	PKV_CHECK_EQ(y.pages, 17);
	PKV_CHECK_EQ(y.slack, 252);
	for (const char* sc : kAllScenarios)
		ExpectMatchesReference("v1.11.0", odd, sc, RunScenario(odd, sc), Compare::kTokensAndRows);
	// 4.1's widths at B = 256 (1, 255, 256, 257, 512, 1,512, 4,099, 4,100: the last two in the partial page), against
	// the flat twin; the `widths` scenario above runs B = 16's (T-3196 F-2)
	DerivedBoundaries(odd, B, kBoundaryWidths, "4.4/T2");
	const RefRecord* saved = RefLookup(Reference("v1.11.0", odd), "persist", "saved");
	BlockPoolOf pool(odd.model, 1, y);
	PKV_CHECK_EQ(pool.status, SSLM_OK);
	sslm_seq s = nullptr;
	if (pool.pool) PKV_CHECK_EQ(sslm_seq_create(odd.model, &pool.pool, &s), SSLM_OK);
	if (s && saved) {
		PKV_CHECK_EQ(PrefillAll(odd.model, s, Stream(31, 100, odd.vocab), 64), SSLM_OK);
		bc::DecodeN(odd.model, s, 4);
		std::vector<uint8_t> rows;
		PKV_CHECK_MSG(SeqRows(pool.pool, s, odd.geo, 1, B, saved->context_length, Rot::kRotated, &rows) &&
		                  Sha(rows.data(), rows.size()) == saved->rows_sha,
		              "persist/saved: rows read from the pages at B = 256 rotated equal v1.11.0's");
		sslm_seq_release(s);
	}
	for (int32_t budget : {1, 255, 256, 257, 1000, 4100}) {
		const int64_t R = Reservation(budget, 4100, B);
		PagePoolOf p(odd.model, 17, y.page_bytes);
		PKV_CHECK_EQ(p.status, SSLM_OK);
		if (!p.pool) continue;
		sslm_status refusal = SSLM_OK;
		const int n = CountCreates(odd.model, &p.pool, budget, &refusal);
		PKV_CHECK_MSG(n == 17 / R && refusal == SSLM_KV_POOL_EXHAUSTED,
		              "budget %d: R = %lld, so 17 pages admit %lld; admitted %d, refused with %s", budget,
		              static_cast<long long>(R), static_cast<long long>(17 / R), n, AbiStatusName(refusal));
	}
	// The fallback mutant: one 4,100-position page in the same bytes (4,352 positions) admits 1
	// budget-100 create against 8.
	PKV_CHECK(RuleB(48, 4100, 1, RuleMutant::kFallback) == 4100);
}

// 4.10 [T2]. Non-dividing caps take the rule's B with a partial last page.
void Cell410NonDividing() {
	struct Row { const Fixture& (*base)(); uint32_t D; int64_t cap, B, pages, slack; };
	auto d64 = []() -> const Fixture& { return D64Cap(4096); };
	auto def = []() -> const Fixture& { return GetFixture("pkv_def"); };
	auto qk = []() -> const Fixture& { return GetFixture("pkv_qk"); };
	const Row rows[] = {{d64, 64, 4095, 64, 64, 1}, {d64, 64, 4097, 128, 33, 127}, {d64, 64, 4100, 128, 33, 124},
	                    {def, 48, 4100, 256, 17, 252}, {qk, 128, 32767, 64, 512, 1}};
	for (const Row& r : rows) {
		const Fixture& base = r.base();
		if (!base.ok) continue;
		Variant v;
		v.cap = r.cap;
		v.strip_rope = true;
		Fixture fx;
		sslm_status ms = SSLM_OK;
		if (!MakeVariant(base, v, &fx, &ms)) {
			PKV_CHECK_MSG(false, "D %u cap %lld did not map (%s)", r.D, static_cast<long long>(r.cap), AbiStatusName(ms));
			continue;
		}
		PKV_CHECK_EQ(RuleB(r.D, static_cast<uint64_t>(r.cap), 1), r.B);
		const Lay y = LayoutOf(fx.geo, 1, r.B);
		PKV_CHECK_EQ(y.pages, r.pages);
		PKV_CHECK_EQ(y.slack, r.slack);
		PKV_CHECK_MSG(sslm_kv_page_positions(fx.model) == r.B, "D %u cap %lld: B = %lld, %lld owed", r.D,
		              static_cast<long long>(r.cap), static_cast<long long>(sslm_kv_page_positions(fx.model)),
		              static_cast<long long>(r.B));
		PKV_CHECK_MSG(sslm_kv_block_size(fx.model) == y.block_bytes, "D %u cap %lld: block %zu, 1.11.0's %zu owed", r.D,
		              static_cast<long long>(r.cap), sslm_kv_block_size(fx.model), y.block_bytes);
		PKV_CHECK_MSG(sslm_kv_pool_overhead_size(fx.model, 1) == y.Overhead(1),
		              "D %u cap %lld: overhead(1) %zu; %lld pages leave %lld slack positions, %zu owed", r.D,
		              static_cast<long long>(r.cap), sslm_kv_pool_overhead_size(fx.model, 1),
		              static_cast<long long>(r.pages), static_cast<long long>(r.slack), y.Overhead(1));
		PKV_CHECK_MSG(RuleB(r.D, static_cast<uint64_t>(r.cap), 1, RuleMutant::kFallback) != r.B,
		              "separation: the fallback gives the rule's B at D %u cap %lld", r.D, static_cast<long long>(r.cap));
	}
}

// 4.12 [T2]. A pin of 16 at cap 4,100 is refused, and the model's B stays 256.
void Cell412Cap4100() {
	const Fixture& odd = GetFixture("pkv_odd");
	if (!odd.ok) return;
	Fixture o;
	Variant v;
	v.kv_block_size = 16;
	if (!MakeVariant(odd, v, &o)) return;
	PKV_CHECK_EQ(KvPagesTestPinPagePositions(o.model, 16), SSLM_INVALID_ARGUMENT);
	PKV_CHECK_EQ(sslm_kv_page_positions(o.model), 256);
	// a pin of 4, which divides 4,100, takes (6.7's third direction)
	Fixture o4;
	if (MakeVariant(odd, v, &o4)) {
		PKV_CHECK_EQ(KvPagesTestPinPagePositions(o4.model, 4), SSLM_OK);
		PKV_CHECK_EQ(sslm_kv_page_positions(o4.model), 4);
	}
}

// A create with a buffer smaller than any real requirement: SSLM_BUFFER_TOO_SMALL when the page
// count passes, SSLM_INVALID_ARGUMENT when it is refused first. Nothing allocated either way.
sslm_status TinyCreate(sslm_model model, uint32_t blocks) {
	AlignedBuf tiny(4096);
	sslm_kv_pool pool = nullptr;
	const sslm_status st = sslm_kv_pool_create(model, tiny.p, tiny.n, blocks, &pool);
	if (pool) sslm_kv_pool_destroy(pool);
	return st;
}

// 2.9 [T2]. Pinned to 16 at cap 32,768: 2,097,152 blocks exceed UINT32_MAX pages -> refused first.
void Cell29() {
	const Fixture& base = GetFixture("pkv_32k");
	if (!base.ok) return;
	Fixture fx;
	Variant v;
	v.kv_block_size = 16;
	if (!MakeVariant(base, v, &fx)) return;
	PKV_CHECK_EQ(KvPagesTestPinPagePositions(fx.model, 16), SSLM_OK);
	PKV_CHECK_EQ(sslm_kv_page_positions(fx.model), 16);
	PKV_CHECK_EQ(TinyCreate(fx.model, 2097151), SSLM_BUFFER_TOO_SMALL);
	PKV_CHECK_EQ(TinyCreate(fx.model, 2097152), SSLM_INVALID_ARGUMENT);
}

// 2.9b [T2]. The unpinned u32 thresholds: floor(UINT32_MAX / ceil(cap/B)) at the derived B.
void Cell29b() {
	struct Row { const char* what; const Fixture& (*base)(); uint32_t D; int64_t B; uint32_t threshold; };
	auto d64 = []() -> const Fixture& { return D64Cap(4096); };
	auto qk = []() -> const Fixture& { return GetFixture("pkv_qk"); };
	const Row rows[] = {{"D = 64 at cap 32,768 (the 0.5B's geometry)", d64, 64, 128, 16777215u},
	                    {"D = 128 at cap 32,768", qk, 128, 64, 8388607u}};
	for (const Row& r : rows) {
		Variant v;
		v.cap = 32768;
		v.strip_rope = true;
		Fixture fx;
		if (!r.base().ok || !MakeVariant(r.base(), v, &fx)) continue;
		PKV_CHECK_EQ(RuleB(r.D, 32768, 1), r.B);
		PKV_CHECK_EQ(static_cast<uint64_t>(UINT32_MAX) / CapPages(32768, r.B), r.threshold);
		PKV_CHECK_MSG(TinyCreate(fx.model, r.threshold) == SSLM_BUFFER_TOO_SMALL,
		              "%s: %u blocks must pass the page-count check (B = 16's threshold refuses them)", r.what, r.threshold);
		PKV_CHECK_MSG(TinyCreate(fx.model, r.threshold + 1) == SSLM_INVALID_ARGUMENT,
		              "%s: %u blocks must be refused before the buffer check", r.what, r.threshold + 1);
		// mutant: a threshold computed from B = 16 refuses at the threshold
		PKV_CHECK(static_cast<uint64_t>(UINT32_MAX) / CapPages(32768, 16) < r.threshold);
	}
}

// 2.9c [T2]. pkv_odd: 17 pages per block; 17 * 252,645,135 = UINT32_MAX exactly.
void Cell29c() {
	const Fixture& odd = GetFixture("pkv_odd");
	if (!odd.ok) return;
	RequireOddB(odd);
	PKV_CHECK_EQ(17ull * 252645135ull, static_cast<unsigned long long>(UINT32_MAX));
	PKV_CHECK_EQ(TinyCreate(odd.model, 252645135u), SSLM_BUFFER_TOO_SMALL);
	PKV_CHECK_EQ(TinyCreate(odd.model, 252645136u), SSLM_INVALID_ARGUMENT);
	// mutant: floor(cap/B) = 16 pages per block puts the threshold at 268,435,455
	PKV_CHECK(static_cast<uint64_t>(UINT32_MAX) / CapPages(4100, 256, true) == 268435455u);
}

// 2.9d [T2]. The block product's overflow at a non-dividing cap, on the plan's CFG1-only fixture:
// L 2,090,000, H_kv 1,024, D 1,048,576, Int8, cap 4,100 (B = 16, 257 pages; q = L*2*H_kv*D per
// position; cap*q and 16*q fit 64 bits, 257*16*q does not).
// Rev 16.23 (D-SLM8437, T-3189 G-1): the fixture loads (SslmModel::Load) and does not map (sslm_model_map builds
// the engine cache from WGT1 weights it does not carry), which is not a bound on the window, so every value is
// read through the geometry seam over the loaded config; the create's refusal is requirement_fits == 0 at one block.
void Cell29d() {
	superslm_test::Cfg1Spec spec;
	spec.hidden_size = 1u << 30;
	spec.num_hidden_layers = 2090000;
	spec.num_attention_heads = 1024;
	spec.num_key_value_heads = 1024;
	spec.head_dim = 1048576;
	spec.intermediate_size = 2;
	spec.vocab_size = 256;
	spec.context_cap = 4100;
	spec.kv_precision = 0;
	spec.kv_block_size = 16;
	const uint64_t q = 2090000ull * 2 * 1024 * 1048576;
	PKV_CHECK_EQ(q, 4488240824320000ull);
	PKV_CHECK_EQ(RuleB(1048576, 4100, 1), 16);
	PKV_CHECK(q <= UINT64_MAX / 4100 && q <= UINT64_MAX / 16 && q > UINT64_MAX / (257ull * 16));
	const std::vector<uint8_t> b = Cfg1OnlyArtifact(spec);
	std::string err;
	const superslm::SslmModelStatus ls = LoadStatus(b, &err);
	PKV_CHECK_MSG(ls == superslm::SslmModelStatus::Ok, "the loader refuses the 2.9d fixture: %s (%s)",
	              superslm::SslmModelStatusName(ls), err.c_str());
	// Rev 16.22 (D-SLM8427): the block keeps 1.11.0's cap * q; the overhead carries 12 slack positions;
	// the create refuses through blocks + overhead > 2^64 - 1; the overhead saturates from 343 blocks.
	const uint64_t block = 4100ull * q, overhead1 = 8 + 32 + 12 * q;
	PKV_CHECK_EQ(block, 18401787379712000000ull);
	PKV_CHECK_EQ(overhead1, 53858889891840040ull);
	PKV_CHECK(block + overhead1 < block);                                       // the sum wraps
	PKV_CHECK_EQ(block + overhead1, 8902195894288424ull);                       // the unchecked-sum mutant's value
	PKV_CHECK(343ull * 12 * q / q != 343ull * 12 || 343ull * 12 * q < 342ull * 12 * q);  // the slack product wraps at 343
	PKV_CHECK(342ull * 12 * q / (12 * q) == 342ull);                            // and not at 342
	auto seam = [&](uint32_t n, KvPagesTestPoolGeometryOut* g) {
		*g = KvPagesTestPoolGeometryOut{};
		const sslm_status st = KvPagesTestPoolGeometry(b.data(), b.size(), n, g);
		PKV_CHECK_MSG(st == SSLM_OK, "KvPagesTestPoolGeometry(2.9d, %u blocks) = %s", n, AbiStatusName(st));
		return st == SSLM_OK;
	};
	KvPagesTestPoolGeometryOut g1, g342, g343;
	if (seam(1, &g1)) {
		PKV_CHECK_MSG(g1.block_size == block, "block %zu; 1.11.0's cap * q, 18,401,787,379,712,000,000, owed (rev 16.21's grown "
		              "block saturates)", g1.block_size);
		PKV_CHECK_MSG(g1.overhead == overhead1, "overhead(1) %zu; 8 + 32 + 12 q owed", g1.overhead);
		PKV_CHECK_MSG(g1.page_count_fits == 1 && g1.requirement_fits == 0,
		              "one block: page count fits %d, requirement fits %d; the create refuses through the sum (0 owed)",
		              g1.page_count_fits, g1.requirement_fits);
		PKV_CHECK_EQ(g1.page_positions, 16);
		PKV_CHECK_EQ(g1.page_size, static_cast<size_t>(16 * q));
	}
	if (seam(343, &g343)) PKV_CHECK_MSG(g343.overhead == SIZE_MAX, "overhead(343) %zu, SIZE_MAX owed", g343.overhead);
	if (seam(342, &g342)) PKV_CHECK_MSG(g342.overhead != SIZE_MAX, "overhead(342) saturates");
}

// 10.9 [T2]. Admission at a non-dividing cap (D-SLM8396).
void Cell109() {
	const Fixture& odd = GetFixture("pkv_odd");
	if (!odd.ok) return;
	RequireOddB(odd);
	const int64_t B = OddB();
	const Lay y = LayoutOf(odd.geo, 1, B);
	{  // a pool of 17 pages admits one legacy create and refuses a second
		PagePoolOf p(odd.model, 17, y.page_bytes);
		PKV_CHECK_EQ(p.status, SSLM_OK);
		sslm_status refusal = SSLM_OK;
		if (p.pool) PKV_CHECK_EQ(CountCreates(odd.model, &p.pool, 0, &refusal), 1);
		if (p.pool) PKV_CHECK_EQ(refusal, SSLM_KV_POOL_EXHAUSTED);
	}
	{  // eight budget-100 creates (R(100) = 2), the ninth refused
		PKV_CHECK_EQ(Reservation(100, 4100, B), 2);
		PagePoolOf p(odd.model, 17, y.page_bytes);
		PKV_CHECK_EQ(p.status, SSLM_OK);
		sslm_status refusal = SSLM_OK;
		if (p.pool) PKV_CHECK_EQ(CountCreates(odd.model, &p.pool, 100, &refusal), 8);
		if (p.pool) PKV_CHECK_EQ(refusal, SSLM_KV_POOL_EXHAUSTED);
	}
	// a legacy holder writes positions 0 to 4,099 and refuses 4,100; a save at width 4,099 restores
	{
		BlockPoolOf p(odd.model, 2, y);
		PKV_CHECK_EQ(p.status, SSLM_OK);
		sslm_seq s = nullptr;
		if (p.pool) PKV_CHECK_EQ(sslm_seq_create(odd.model, &p.pool, &s), SSLM_OK);
		if (s) {
			PKV_CHECK_EQ(PrefillAll(odd.model, s, Stream(41, 4099, odd.vocab), 64), SSLM_OK);
			const std::vector<uint8_t> blob = bc::Save(s);
			sslm_seq r = nullptr;
			PKV_CHECK_EQ(sslm_seq_restore(odd.model, &p.pool, blob.data(), blob.size(), &r), SSLM_OK);
			if (r) {
				std::vector<uint8_t> rows, brows;
				PKV_CHECK_MSG(SeqRows(p.pool, r, odd.geo, 1, B, 4099, Rot::kRotated, &rows) &&
				                  BlobRows(blob, odd, HiddenSize(odd), 4099, &brows) && rows == brows,
				              "the save at width 4,099 restores with rows equal (read at B = 256)");
				sslm_seq_release(r);
			}
			// The state size (rev 16.22, D-SLM8427): 180 + hidden_size 192 + 4 * 4,100 + the flat block
			// 1,574,400 = 1,591,172, v1.11.0's 1,591,152 plus 20. Mutant: rev 16.21's grown block, 1,687,940.
			const size_t odd_size = sslm_seq_state_size(odd.model);
			PKV_CHECK_EQ(y.block_bytes, 1574400u);
			PKV_CHECK_EQ(180u + 192u + 4u * 4100u + y.block_bytes, 1591172u);
			PKV_CHECK_MSG(odd_size == 1591172u, "sslm_seq_state_size at cap 4,100 = %zu, 1,591,172 owed", odd_size);
			PKV_CHECK(180u + 192u + 4u * 4100u + 17u * y.page_bytes == 1687940u);  // the mutant's value, separated
			PKV_CHECK(blob.size() <= odd_size);
			// the last row, then the cap
			PKV_CHECK_EQ(PrefillAll(odd.model, s, Stream(42, 1, odd.vocab), 1), SSLM_OK);
			sslm_status st = SSLM_OK;
			PKV_CHECK(Next(odd.model, s, &st) >= 0);  // the ready token at L = 4,100 writes no row
			PKV_CHECK_EQ(Next(odd.model, s, &st), -1);
			PKV_CHECK_EQ(st, SSLM_CONTEXT_CAP_EXCEEDED);
			std::vector<uint8_t> rows;
			// kills: the floor mutant (16 pages) has no page for positions 4,096 to 4,099
			PKV_CHECK_MSG(SeqRows(p.pool, s, odd.geo, 1, B, 4100, Rot::kRotated, &rows),
			              "the table maps 17 pages: positions 4,096 to 4,099 are on the partial last page");
			sslm_seq_release(s);
		}
	}
	// Pool sizing (rev 16.22): the block is 1.11.0's 1,574,400; the overhead carries the slack
	// (96,808 at one block, 193,584 at two; 1.11.0: 40 and 48); a create succeeds in exactly
	// N * block + overhead(N) bytes and is refused one byte below and at N * block + 8N + 32.
	PKV_CHECK_EQ(sslm_kv_block_size(odd.model), 1574400u);
	PKV_CHECK_EQ(y.Overhead(1), 96808u);
	PKV_CHECK_EQ(y.Overhead(2), 193584u);
	PKV_CHECK_EQ(sslm_kv_pool_overhead_size(odd.model, 1), 96808u);
	PKV_CHECK_EQ(sslm_kv_pool_overhead_size(odd.model, 2), 193584u);
	for (uint32_t n : {1u, 2u}) {
		const size_t exact = y.PoolBytes(n);  // 1,671,208 and 3,342,384: 17 and 34 pages plus 40 and 48
		PKV_CHECK_EQ(exact, n == 1 ? 1671208u : 3342384u);
		for (size_t size : {exact, exact - 1, n * y.block_bytes + 8 * n + 32}) {
			AlignedBuf mem(size);
			sslm_kv_pool pool = nullptr;
			const sslm_status st = sslm_kv_pool_create(odd.model, mem.p, size, n, &pool);
			const sslm_status want = size == exact ? SSLM_OK : SSLM_BUFFER_TOO_SMALL;
			// kills: an overhead without the slack term (the create succeeds 96,768 bytes short of its pages)
			PKV_CHECK_MSG(st == want, "%u block(s) in %zu bytes: %s, %s owed", n, size, AbiStatusName(st), AbiStatusName(want));
			if (pool && size == exact) {
				sslm_kv_pool_stats_out ps{};
				ps.struct_size = sizeof ps;
				PKV_CHECK_EQ(sslm_kv_pool_stats(pool, &ps), SSLM_OK);
				PKV_CHECK_EQ(ps.page_count, 17 * n);
				if (n == 2) {
					// The block-index peek seam strides by the pages a block holds (17 * page_bytes): block 1
					// reads what page 17 holds. Mutant: a stride of sslm_kv_block_size reads 96,768 bytes earlier.
					sslm_seq a = nullptr, b = nullptr;
					PKV_CHECK_EQ(sslm_seq_create(odd.model, &pool, &a), SSLM_OK);
					PKV_CHECK_EQ(sslm_seq_create(odd.model, &pool, &b), SSLM_OK);
					if (a) PrefillAll(odd.model, a, Stream(41, 300, odd.vocab), 64);
					if (b) PrefillAll(odd.model, b, Stream(43, 300, odd.vocab), 64);
					std::vector<uint8_t> via_block(y.page_bytes), via_page(y.page_bytes);
					PKV_CHECK_EQ(sslm_g5_test_only_peek_kv_block_bytes(pool, 1, via_block.data(), via_block.size()), SSLM_OK);
					PKV_CHECK_EQ(sslm_pkv_test_only_peek_page_bytes(pool, 17, via_page.data(), via_page.size()), SSLM_OK);
					PKV_CHECK_MSG(via_block == via_page, "the block-index peek of block 1 is not page 17's bytes");
					if (b) sslm_seq_release(b);
					if (a) sslm_seq_release(a);
				}
			}
			if (pool) sslm_kv_pool_destroy(pool);
		}
	}
	// a share adopt at origin 4,000 maps 15 full pages and copies one
	{
		PagePoolOf p(odd.model, 40, y.page_bytes);
		PKV_CHECK_EQ(p.status, SSLM_OK);
		if (p.pool) {
			sslm_prefix px = bc::BudgetPrefix(odd, &p.pool, Stream(51, 4000, odd.vocab));
			sslm_seq s = nullptr;
			PKV_CHECK_EQ(sslm_seq_create_budgeted(odd.model, &p.pool, 64, &s), SSLM_OK);
			if (px && s) {
				PKV_CHECK_EQ(sslm_seq_adopt_prefix(s, px), SSLM_OK);
				uint32_t shared_ok = 0, sm = 0;
				for (uint32_t i = 0; i < 16; ++i) {
					uint32_t a = 0, b = 0;
					sslm_pkv_test_only_seq_table_entry(s, i, &a, &sm);
					sslm_pkv_test_only_prefix_table_entry(px, i, &b, nullptr);
					shared_ok += (a == b) ? 1 : 0;
				}
				PKV_CHECK_MSG(sm == 16, "after the adopt the holder maps %u pages, 16 owed (15 shared + 1 copied)", sm);
				uint32_t last_s = 0, last_p = 0;
				sslm_pkv_test_only_seq_table_entry(s, 15, &last_s, nullptr);
				sslm_pkv_test_only_prefix_table_entry(px, 15, &last_p, nullptr);
				PKV_CHECK_MSG(shared_ok == 15 && last_s != last_p, "the first 15 pages are shared (%u) and page 15 is a copy",
				              shared_ok);
			}
			if (s) sslm_seq_release(s);
			if (px) sslm_prefix_release(px);
		}
	}
	// must-rejects, separated at this construction
	PKV_CHECK_MSG(17 / Reservation(100, 4100, 4100) == 17 && (4352 / 4100) == 1,
	              "separation: the fallback's one 4,100-position page in 17 * 256 = 4,352 positions holds 1 holder");
	PKV_CHECK(CapPages(4100, 256, true) == 16);
}

// 4.13 [T2]. The cap-4,100 member: 4.2 (prefix lengths) whole on pkv_odd at B = 256 with a partial
// last page; 1,024 is aligned at 256.
void Cell413Cap4100() {
	const Fixture& odd = GetFixture("pkv_odd");
	if (!odd.ok) return;
	RequireOddB(odd);
	ExpectMatchesReference("v1.11.0", odd, "prefix_lengths", RunScenario(odd, "prefix_lengths"), Compare::kTokensAndRows);
	// 4.2's lengths at B = 256 (0, 1, 255, 256, 257, 1,000, 1,024), the aligned 1,024 run (T-3196 F-2)
	DerivedBoundaries(odd, OddB(), kBoundaryPrefixes, "4.13/T2:cap4100");
}

// 6.7 [T2]. At a non-dividing cap (T-3172 G-9), the page-pool half (rev 16.23, T-3189 G-3: the option directions
// and the cross-kind byte-equality are 6.7/T3:non-dividing): a blob saved at a pinned B = 4, which divides 4,100,
// restores into the derived B = 256 page pool with rows equal; the restoring legacy holder reserves ceil(cap/B) =
// 17 pages; and the page pool's legacy save is 160 + hidden_size + 4 * history + the flat block (rev 16.22).
void Cell67NonDividing() {
	const Fixture& odd = GetFixture("pkv_odd");
	if (!odd.ok) return;
	RequireOddB(odd);
	const int64_t B = OddB();
	const Lay y = LayoutOf(odd.geo, 1, B);
	{  // the page pool's legacy save at width 4,099 is exactly 160 + hidden_size + 4 * history + the flat block
		BlockPoolOf pp(odd.model, 1, y);
		sslm_seq s = nullptr;
		if (pp.pool && sslm_seq_create(odd.model, &pp.pool, &s) == SSLM_OK) {
			PKV_CHECK_EQ(PrefillAll(odd.model, s, Stream(41, 4099, odd.vocab), 64), SSLM_OK);
			const std::vector<uint8_t> b = bc::Save(s);
			PKV_CHECK_MSG(b.size() == 160 + 192 + 4 * Le64(b, 112) + y.block_bytes,
			              "the legacy save is %zu bytes; 160 + 192 + 4 * %llu + %zu owed", b.size(),
			              static_cast<unsigned long long>(Le64(b, 112)), y.block_bytes);
			sslm_seq_release(s);
		}
	}
	// the restoring legacy holder reserves ceil(cap/B) = 17 pages (T-3196 F-3). A budget-1 create needs R(1) = 2
	// pages, so the observable is an 18-page pool: 17 reserved leave 1 and the create is refused; the floor's 16
	// leave 2 and it is admitted (the floor mutant dies here, where a 17-page pool refuses at both). A 19-page pool
	// admits it, so the refusal is the reservation's and not the pool's.
	{
		BlockPoolOf src(odd.model, 1, y);
		sslm_seq s = nullptr;
		std::vector<uint8_t> blob;
		if (src.pool && sslm_seq_create(odd.model, &src.pool, &s) == SSLM_OK) {
			PKV_CHECK_EQ(PrefillAll(odd.model, s, Stream(41, 300, odd.vocab), 64), SSLM_OK);
			blob = bc::Save(s);
			sslm_seq_release(s);
		}
		PKV_CHECK_MSG(!blob.empty(), "the legacy holder to restore was not made");
		PKV_CHECK_EQ(Reservation(1, 4100, B), 2);
		for (uint32_t pages : {18u, 19u}) {
			if (blob.empty()) break;
			PagePoolOf p(odd.model, pages, y.page_bytes);
			PKV_CHECK_EQ(p.status, SSLM_OK);
			if (!p.pool) continue;
			sslm_seq r = nullptr, extra = nullptr;
			PKV_CHECK_EQ(sslm_seq_restore(odd.model, &p.pool, blob.data(), blob.size(), &r), SSLM_OK);
			const sslm_status cs = sslm_seq_create_budgeted(odd.model, &p.pool, 1, &extra);
			const sslm_status want = pages == 18 ? SSLM_KV_POOL_EXHAUSTED : SSLM_OK;
			PKV_CHECK_MSG(r && cs == want, "a %u-page pool after the restore: budget-1 create %s, %s owed (17 pages "
			              "reserved; a floor reservation of 16 admits it at 18)", pages, AbiStatusName(cs), AbiStatusName(want));
			if (extra) sslm_seq_release(extra);
			if (r) sslm_seq_release(r);
		}
	}
	// the third direction: pinned B = 4 (divides 4,100) -> derived 256
	{
		Fixture o4;
		Variant v;
		v.kv_block_size = 16;
		if (MakeVariant(odd, v, &o4)) {
			PKV_CHECK_EQ(KvPagesTestPinPagePositions(o4.model, 4), SSLM_OK);
			const Lay y4 = LayoutOf(o4.geo, 1, 4);
			BlockPoolOf src(o4.model, 1, y4);
			BlockPoolOf dst(odd.model, 1, y);
			sslm_seq s = nullptr, r = nullptr;
			if (src.pool && sslm_seq_create(o4.model, &src.pool, &s) == SSLM_OK) {
				PKV_CHECK_EQ(PrefillAll(o4.model, s, Stream(41, 4099, o4.vocab), 64), SSLM_OK);
				const std::vector<uint8_t> blob = bc::Save(s);
				// the twin's model hash differs from pkv_odd's (kv_block_size patched), so the blob restores
				// on the pinned twin's identity: map a second copy of the twin unpinned
				Fixture o4u;
				if (MakeVariant(odd, v, &o4u)) {
					BlockPoolOf d2(o4u.model, 1, y);
					PKV_CHECK_EQ(sslm_seq_restore(o4u.model, &d2.pool, blob.data(), blob.size(), &r), SSLM_OK);
					if (r) {
						std::vector<uint8_t> rows, brows;
						PKV_CHECK_MSG(SeqRows(d2.pool, r, o4u.geo, 1, B, 4099, Rot::kRotated, &rows) &&
						                  BlobRows(blob, o4, HiddenSize(o4), 4099, &brows) && rows == brows,
						              "pinned 4 -> derived 256 at width 4,099: rows equal");
						sslm_seq_release(r);
					}
				}
				sslm_seq_release(s);
			}
		}
	}
}

// 10.9 [T2] batch (plan rev 16.24 §26.1, §26.8 item 3; D-SLM8441): 4.4/T3:batch's construction with both holders in one
// two-block page pool (34 pages) on pkv_odd at B = 256, through rows 4,088 to 4,099 and the cap refusal. It names no
// mutant: it is the attribution control for 4.4/T3:batch, so a 4.4 failure with a 10.9 pass is the pool kinds'.
void Cell109Batch() {
	const Fixture& odd = GetFixture("pkv_odd");
	if (!odd.ok) return;
	RequireOddB(odd);
	const int64_t B = OddB();
	const Lay y = LayoutOf(odd.geo, 1, B);
	const int64_t W = 4088;
	const std::vector<uint8_t> blob = WidthsBlob(odd, B, W);
	BlockPoolOf pp(odd.model, 2, y, 0x5A);
	PKV_CHECK_EQ(pp.status, SSLM_OK);
	if (!pp.pool) return;
	BatchAgainstTwin(odd, blob, W, B, {&pp.pool, B, "page"}, {&pp.pool, B, "page"}, true, "10.9 page + page");
}

PKV_CELL("4.4/T2", "T2", Cell44);
PKV_CELL("10.9/T2:batch", "T2", Cell109Batch);
PKV_CELL("4.10/T2:non-dividing", "T2", Cell410NonDividing);
PKV_CELL("4.12/T2:cap4100", "T2", Cell412Cap4100);
PKV_CELL("2.9/T2", "T2", Cell29);
PKV_CELL("2.9b/T2", "T2", Cell29b);
PKV_CELL("2.9c/T2", "T2", Cell29c);
PKV_CELL("2.9d/T2", "T2", Cell29d);
PKV_CELL("10.9/T2", "T2", Cell109);
PKV_CELL("4.13/T2:cap4100", "T2", Cell413Cap4100);
PKV_CELL("6.7/T2:non-dividing", "T2", Cell67NonDividing);

}  // namespace

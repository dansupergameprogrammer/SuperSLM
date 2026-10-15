// Paged-KV plan rev 16.21 §23.8 step T1, rotation (§3.1: run r of logical page j sits in slot
// (r + j) mod runs) and its whole-page consequences:
//
//   6.6a/T1           engine level, over the test's own page buffer: runs in {8, 12, 96, 112}, more
//                     than 64 pages, a permuted table; every address, each page's slots a permutation,
//                     the one-page view term for term
//   6.6b/T1           every ABI address site on a 3-layer fixture (runs = 12), B = 16, 70 pages:
//                     a legacy SSB5 save, a budget SSB6 save and restore, the sharing compare, batch
//                     decode and chunked prefill across j = 64, the page peek
//   6.7/T1:layouts    SSB6 and the legacy SSB5 saved at one B restore at another: pinned 16 -> derived,
//                     derived -> pinned 1, the one-page view (pinned to the cap) -> derived
//   6.7/T1:old-blobs  v1.11.0's pinned contiguous blobs restore at the derived B
//   6.7/T1:reshare    sslm_seq_restore_shared across layouts at origins 1,200 and 1,024: four must-accepts and
//                     the must-reject 128 -> 1 at 1,200 (rev 16.24, D-SLM8442; T-3196 F-1)
//   6.8/T1            adopt's tail copy, begin_from's tail copy, a materialized restore and release
//                     poison on a reserve scattered by churn, at the derived B
//
// On 2ebe5e3 nothing rotates and B is CFG1's 16, so the address assertions fail at j >= 1 and the
// derived-B assertions fail outright. The site paths that compare one engine read against another
// engine write (a save against its own restore) agree on 2ebe5e3, because every site is unrotated
// alike; those assertions are green there and exist to kill the per-site mutants 6.6b names once the
// sites are rotated. The red reason of 6.6b is the layout read through the peek.

#include "pkv_r9.h"

#include <set>

namespace {

using namespace pkv;
using namespace pkv::r9;

std::vector<uint32_t> Shuffled(uint32_t n, uint32_t seed) {
	std::vector<uint32_t> t(n);
	for (uint32_t i = 0; i < n; ++i) t[i] = i;
	uint32_t x = seed;
	for (uint32_t i = n; i > 1; --i) {
		x ^= x << 13;
		x ^= x >> 17;
		x ^= x << 5;
		std::swap(t[i - 1], t[x % i]);
	}
	return t;
}

// ---- 6.6a ----------------------------------------------------------------------------------------

void Cell66a() {
	struct G { uint32_t L, H; };
	const G gs[] = {{2, 2}, {3, 2}, {24, 2}, {28, 2}};  // runs 8, 12, 96, 112
	const uint32_t D = 2;
	const int64_t B = 4;
	const uint32_t pages = 70;  // > 64, so j = 64 is reached
	for (const G& g : gs) {
		const uint32_t runs = g.L * 2u * g.H;
		const Geometry geo{g.L, g.H, D, static_cast<int64_t>(pages) * B};
		const size_t page_bytes = size_t{runs} * static_cast<size_t>(B) * D;
		AlignedBuf mem(page_bytes * (pages + 1));
		const std::vector<uint32_t> table = Shuffled(pages, 2463534242u + runs);
		superslm::KvPageView view{};
		view.pool_base = mem.p;
		view.page_table = table.data();
		view.mapped_pages = pages;
		view.page_positions = B;
		view.page_bytes = page_bytes;
		SetRuns(view, runs);
		long long bad = 0;
		std::string first;
		std::vector<std::set<uint64_t>> slots(pages);
		for (int64_t pos = 0; pos < geo.context_cap; ++pos)
			for (uint32_t l = 0; l < g.L; ++l)
				for (uint32_t half = 0; half < 2; ++half)
					for (uint32_t h = 0; h < g.H; ++h) {
						const int8_t* got = half ? superslm::ValueRow(view, l, g.H, D, h, pos)
						                         : superslm::KeyRow(view, l, g.H, D, h, pos);
						const size_t j = static_cast<size_t>(pos / B);
						const int8_t* want = reinterpret_cast<const int8_t*>(
						    mem.p + size_t{table[j]} * page_bytes + InPage(geo, 1, B, l, half, h, pos));
						if (got != want && bad++ == 0)
							first = Fmt("l %u half %u h %u pos %lld (j %zu)", l, half, h, static_cast<long long>(pos), j);
						const size_t in_page = static_cast<size_t>(reinterpret_cast<const uint8_t*>(got) - mem.p) -
						                       size_t{table[j]} * page_bytes;
						if (pos % B == 0) slots[j].insert(in_page / (static_cast<size_t>(B) * D));
					}
		// kills: no rotation (2ebe5e3), and the four mutants below
		PKV_CHECK_MSG(bad == 0, "runs %u: %lld addresses differ from (r + j) mod runs; first at %s", runs, bad, first.c_str());
		long long not_perm = 0;
		for (const auto& s : slots)
			if (s.size() != runs || *s.rbegin() >= runs) ++not_perm;
		PKV_CHECK_MSG(not_perm == 0, "runs %u: %lld pages whose slots are not a permutation of [0, runs)", runs, not_perm);
		// The one-page view equals the flat formula term for term.
		const uint32_t zero[1] = {0};
		superslm::KvPageView one{};
		one.pool_base = mem.p;
		one.page_table = zero;
		one.mapped_pages = 1;
		one.page_positions = geo.context_cap;
		one.page_bytes = mem.n;
		SetRuns(one, runs);
		long long flat_bad = 0;
		for (int64_t pos = 0; pos < geo.context_cap; pos += 7)
			for (uint32_t l = 0; l < g.L; ++l)
				for (uint32_t h = 0; h < g.H; ++h)
					if (superslm::KeyRow(one, l, g.H, D, h, pos) !=
					        superslm::KeyRow(mem.p, l, geo.context_cap, g.H, D, h, pos) ||
					    superslm::ValueRow(one, l, g.H, D, h, pos) !=
					        superslm::ValueRow(mem.p, l, geo.context_cap, g.H, D, h, pos))
						++flat_bad;
		PKV_CHECK_MSG(flat_bad == 0, "runs %u: the one-page view differs from the flat accessors at %lld rows", runs, flat_bad);
		// Separation of 6.6a's mutants at this construction (plan: j & 63 separates where runs does
		// not divide 64; & (runs - 1) where runs is not a power of two).
		for (Rot m : {Rot::kJAnd63, Rot::kAndMask, Rot::kModPages, Rot::kPhysical, Rot::kNone}) {
			long long differ = 0;
			for (int64_t pos = 0; pos < geo.context_cap; pos += B)
				for (uint32_t l = 0; l < g.L; ++l)
					for (uint32_t half = 0; half < 2; ++half)
						for (uint32_t h = 0; h < g.H; ++h) {
							const uint32_t phys = table[static_cast<size_t>(pos / B)];
							if (InPage(geo, 1, B, l, half, h, pos, m, phys, pages) != InPage(geo, 1, B, l, half, h, pos))
								++differ;
						}
			const bool expect = !((m == Rot::kJAnd63 && 64 % runs == 0) || (m == Rot::kAndMask && (runs & (runs - 1)) == 0));
			PKV_CHECK_MSG((differ > 0) == expect, "runs %u: the mutant '%s' differs at %lld addresses (separation %s)", runs,
			              RotName(m), differ, expect ? "owed" : "not expected");
		}
	}
}

// ---- 6.6b ----------------------------------------------------------------------------------------

// The fixture pinned to 16 (its CFG1 kv_block_size is 16, so 2ebe5e3 also runs it at 16 and the pin
// refusal is the only difference the pin makes there).
bool Pinned(const Fixture& base, int64_t B, Fixture* out) {
	Variant v;
	v.kv_block_size = 16;
	if (!MakeVariant(base, v, out)) return false;
	const sslm_status st = KvPagesTestPinPagePositions(out->model, B);
	PKV_CHECK_MSG(st == SSLM_OK, "%s: the pin to %lld is refused (%s)", base.stem.c_str(), static_cast<long long>(B),
	              AbiStatusName(st));
	return true;
}

void CheckRowsVsBlob(const char* what, sslm_kv_pool pool, sslm_seq s, const Fixture& fx, int64_t B, int64_t L,
                     const std::vector<uint8_t>& blob) {
	std::vector<uint8_t> page_rows, blob_rows;
	const bool read = SeqRows(pool, s, fx.geo, 1, B, L, Rot::kRotated, &page_rows);
	const bool parsed = BlobRows(blob, fx, HiddenSize(fx), L, &blob_rows);
	PKV_CHECK_MSG(read && parsed && page_rows == blob_rows,
	              "%s: rows [0, %lld) read through the peek at (r + j) mod runs, B = %lld, %s the blob's rows", what,
	              static_cast<long long>(L), static_cast<long long>(B), !read ? "unreadable; " : parsed ? "differ from" : "vs an unparsed blob;");
}

void Cell66b() {
	const Fixture& base = L3();
	if (!base.ok) return;
	Fixture fx;
	if (!Pinned(base, 16, &fx)) return;
	const int64_t B = 16, W = 1100;  // j up to 68
	PKV_CHECK_EQ(sslm_kv_page_positions(fx.model), B);
	const Lay y = LayoutOf(fx.geo, 1, B);
	PKV_CHECK_EQ(y.runs, 12u);
	auto e = EngineOf(fx);
	const pkv_engine::WidthRun twin = e->ok ? FlatTwin(*e, W) : pkv_engine::WidthRun{};
	PKV_CHECK_MSG(e->ok && twin.fail.empty(), "flat twin at W = %lld: %s", static_cast<long long>(W), twin.fail.c_str());
	// A legacy SSB5 save (GatherFlatBlock's fast loop), and chunked prefill across j = 64.
	{
		BlockPoolOf pool(fx.model, 1, y);
		PKV_CHECK_EQ(pool.status, SSLM_OK);
		sslm_seq s = nullptr;
		if (pool.pool) PKV_CHECK_EQ(sslm_seq_create(fx.model, &pool.pool, &s), SSLM_OK);
		if (s) {
			const AbiWidth w = RunAbiWidth(fx.model, s, W, fx.vocab, true);
			PKV_CHECK_EQ(w.prefill, SSLM_OK);
			PKV_CHECK_EQ(w.ready, twin.ready);
			PKV_CHECK_EQ(w.decode1, twin.decode1);
			const std::vector<uint8_t> blob = bc::Save(s);
			PKV_CHECK(IsMagic(blob, "SSB5"));
			std::vector<uint8_t> rows;
			PKV_CHECK_MSG(BlobRows(blob, fx, HiddenSize(fx), W + 1, &rows) && Sha(rows.data(), rows.size()) == twin.rows_decode1,
			              "the legacy SSB5 save's rows equal the flat twin's");
			CheckRowsVsBlob("legacy holder", pool.pool, s, fx, B, W + 1, blob);
			sslm_seq_release(s);
		}
	}
	// A budget SSB6 save and restore (HolderRow through ScatterRows), and batch decode across j = 64.
	{
		PagePoolOf pool(fx.model, 3 * static_cast<uint32_t>(Reservation(1200, fx.geo.context_cap, B)), y.page_bytes);
		PKV_CHECK_EQ(pool.status, SSLM_OK);
		sslm_seq a = nullptr, b = nullptr;
		if (pool.pool) {
			PKV_CHECK_EQ(sslm_seq_create_budgeted(fx.model, &pool.pool, 1200, &a), SSLM_OK);
			PKV_CHECK_EQ(sslm_seq_create_budgeted(fx.model, &pool.pool, 1200, &b), SSLM_OK);
		}
		if (a && b) {
			const auto prompt = Stream(pkv_engine::kWidthsSeed, 1020, fx.vocab);
			PKV_CHECK_EQ(PrefillAll(fx.model, a, prompt, 64), SSLM_OK);
			PKV_CHECK_EQ(PrefillAll(fx.model, b, prompt, 64), SSLM_OK);
			// batch decode: both sequences in each sslm_decode_step, 12 tokens, positions 1,020 to 1,031
			sslm_decode_params p{};
			p.layer_budget = static_cast<int32_t>(fx.geo.layers);
			sslm_seq batch[2] = {a, b};
			bool same = true;
			for (int i = 0; i < 12; ++i) {
				int32_t out[2] = {-1, -1};
				PKV_CHECK_EQ(sslm_decode_step(fx.model, batch, 2, &p, nullptr, out), SSLM_OK);
				same = same && out[0] == out[1] && out[0] >= 0;
			}
			PKV_CHECK_MSG(same, "batch decode: the two sequences' tokens differ");
			const std::vector<uint8_t> blob = bc::Save(a);
			PKV_CHECK(IsMagic(blob, "SSB6"));
			const int64_t L = 1020 + 11;  // the ready token after 1,020 writes no row; 11 more do
			CheckRowsVsBlob("budget holder after batch decode", pool.pool, a, fx, B, L, blob);
			sslm_seq r = nullptr;
			PKV_CHECK_EQ(sslm_seq_restore(fx.model, &pool.pool, blob.data(), blob.size(), &r), SSLM_OK);
			if (r) {
				CheckRowsVsBlob("restored budget holder", pool.pool, r, fx, B, L, blob);
				PKV_CHECK(bc::DecodeN(fx.model, r, 6) == bc::DecodeN(fx.model, a, 6));
				sslm_seq_release(r);
			}
		}
		if (a) sslm_seq_release(a);
		if (b) sslm_seq_release(b);
	}
	// The gather mutant (plan 6.7: a gather that walks physical page order instead of logical order; T-3196 F-10).
	// It is invisible wherever a holder's table ascends in physical id, as every fresh-pool holder above does, so
	// this member scatters the reserve first (6.8's churn) and asserts from the engine's own table that the holder's
	// pages do not ascend. Its save's rows are then graded against the flat twin, which no gather order can fake,
	// and its restore against the save.
	{
		PagePoolOf pool(fx.model, 240, y.page_bytes);
		PKV_CHECK_EQ(pool.status, SSLM_OK);
		if (pool.pool) {
			std::vector<sslm_seq> made;
			for (int i = 0; i < 80; ++i) {
				sslm_seq c = nullptr;
				if (sslm_seq_create_budgeted(fx.model, &pool.pool, 1, &c) == SSLM_OK) made.push_back(c);
			}
			PKV_CHECK_EQ(made.size(), 80u);
			for (uint32_t i : Shuffled(static_cast<uint32_t>(made.size()), 66001u)) sslm_seq_release(made[i]);
			sslm_seq h = nullptr;
			PKV_CHECK_EQ(sslm_seq_create_budgeted(fx.model, &pool.pool, 1200, &h), SSLM_OK);
			if (h) {
				const AbiWidth w = RunAbiWidth(fx.model, h, W, fx.vocab, true);
				PKV_CHECK_EQ(w.prefill, SSLM_OK);
				PKV_CHECK_EQ(w.ready, twin.ready);
				PKV_CHECK_EQ(w.decode1, twin.decode1);
				std::vector<uint32_t> table;
				PKV_CHECK(legacy_a::SeqTable(h, &table) && static_cast<int64_t>(table.size()) >= CeilDiv(W + 1, B));
				const bool scattered = !std::is_sorted(table.begin(), table.end());
				PKV_CHECK_MSG(scattered, "construction: the churned holder's %zu pages ascend in physical id, so a "
				              "physical-order gather would pass", table.size());
				const std::vector<uint8_t> blob = bc::Save(h);
				PKV_CHECK(IsMagic(blob, "SSB6"));
				std::vector<uint8_t> rows;
				PKV_CHECK_MSG(BlobRows(blob, fx, HiddenSize(fx), W + 1, &rows) && Sha(rows.data(), rows.size()) == twin.rows_decode1,
				              "the scattered holder's save: rows [0, %lld) differ from the flat twin's (kills the "
				              "physical-order gather)", static_cast<long long>(W + 1));
				CheckRowsVsBlob("scattered budget holder", pool.pool, h, fx, B, W + 1, blob);
				sslm_seq r = nullptr;
				PKV_CHECK_EQ(sslm_seq_restore(fx.model, &pool.pool, blob.data(), blob.size(), &r), SSLM_OK);
				if (r) {
					CheckRowsVsBlob("its restore", pool.pool, r, fx, B, W + 1, blob);
					sslm_seq_release(r);
				}
				sslm_seq_release(h);
			}
		}
	}
	// SharingHandleMatches: a frozen prefix of 1,040 positions (65 pages) re-shared on restore.
	{
		PagePoolOf pool(fx.model, 200, y.page_bytes);
		PKV_CHECK_EQ(pool.status, SSLM_OK);
		if (!pool.pool) return;
		sslm_prefix px = bc::BudgetPrefix(fx, &pool.pool, Stream(51, 1040, fx.vocab));
		sslm_seq s = nullptr;
		PKV_CHECK_EQ(sslm_seq_create_budgeted(fx.model, &pool.pool, 64, &s), SSLM_OK);
		if (px && s) {
			PKV_CHECK_EQ(sslm_seq_adopt_prefix(s, px), SSLM_OK);
			bc::DecodeN(fx.model, s, 8);
			const std::vector<uint8_t> blob = bc::Save(s);
			sslm_seq r = nullptr;
			uint32_t shared = 0;
			PKV_CHECK_EQ(sslm_seq_restore_shared(fx.model, &pool.pool, blob.data(), blob.size(), px, &r, &shared), SSLM_OK);
			PKV_CHECK_MSG(shared == 65, "the sharing restore shared %u pages, 65 owed", shared);
			if (r) {
				uint32_t rp = 0, pp = 0;
				PKV_CHECK_EQ(sslm_pkv_test_only_seq_table_entry(r, 64, &rp, nullptr), SSLM_OK);
				PKV_CHECK_EQ(sslm_pkv_test_only_prefix_table_entry(px, 64, &pp, nullptr), SSLM_OK);
				PKV_CHECK_MSG(rp == pp, "logical page 64 is not the prefix's page after a matching restore");
				CheckRowsVsBlob("sharing restore", pool.pool, r, fx, B, 1040 + 7, blob);
				sslm_seq_release(r);
			}
		}
		if (s) sslm_seq_release(s);
		if (px) sslm_prefix_release(px);
	}
}

// ---- 6.7 -----------------------------------------------------------------------------------------

struct Side {
	Fixture fx;
	int64_t B = 0;
};

// pkv_def's bytes with kv_block_size 32 (so 2ebe5e3's own reading supplies neither 16, 1 nor the
// rule's 128), pinned to `pin` when pin > 0.
bool MakeSide(int64_t pin, Side* s) {
	const Fixture& def = GetFixture("pkv_def");
	Variant v;
	v.kv_block_size = 32;
	if (!def.ok || !MakeVariant(def, v, &s->fx)) return false;
	s->B = pin > 0 ? pin : RuleB(48, 4096, 1);
	if (pin > 0) {
		const sslm_status st = KvPagesTestPinPagePositions(s->fx.model, pin);
		PKV_CHECK_MSG(st == SSLM_OK, "the pin to %lld is refused (%s)", static_cast<long long>(pin), AbiStatusName(st));
	}
	PKV_CHECK_MSG(sslm_kv_page_positions(s->fx.model) == s->B, "the %s side runs at B = %lld, %lld owed",
	              pin > 0 ? "pinned" : "derived", static_cast<long long>(sslm_kv_page_positions(s->fx.model)),
	              static_cast<long long>(s->B));
	return true;
}

// Save a holder of `budget` (0: legacy) on `from` after prefilling 1,100 and decoding 4; restore on
// `to`; its rows read through the peek at to's B, and its next 8 tokens, equal the source's.
void CrossLayout(const char* name, Side& from, Side& to, int32_t budget) {
	const Lay yf = LayoutOf(from.fx.geo, 1, from.B), yt = LayoutOf(to.fx.geo, 1, to.B);
	BlockPoolOf pf(from.fx.model, 1, yf);
	BlockPoolOf pt(to.fx.model, 1, yt);
	PKV_CHECK_EQ(pf.status, SSLM_OK);
	PKV_CHECK_EQ(pt.status, SSLM_OK);
	if (!pf.pool || !pt.pool) return;
	sslm_seq s = nullptr, r = nullptr;
	PKV_CHECK_EQ(budget ? sslm_seq_create_budgeted(from.fx.model, &pf.pool, budget, &s)
	                    : sslm_seq_create(from.fx.model, &pf.pool, &s),
	             SSLM_OK);
	if (!s) return;
	PKV_CHECK_EQ(PrefillAll(from.fx.model, s, Stream(31, 1100, from.fx.vocab), 64), SSLM_OK);
	bc::DecodeN(from.fx.model, s, 4);
	const std::vector<uint8_t> blob = bc::Save(s);
	PKV_CHECK_MSG(IsMagic(blob, budget ? "SSB6" : "SSB5"), "%s: the blob's magic", name);
	const int64_t L = 1100 + 3;
	PKV_CHECK_EQ(sslm_seq_restore(to.fx.model, &pt.pool, blob.data(), blob.size(), &r), SSLM_OK);
	if (r) {
		std::vector<uint8_t> rows, blob_rows;
		PKV_CHECK_MSG(SeqRows(pt.pool, r, to.fx.geo, 1, to.B, L, Rot::kRotated, &rows) &&
		                  BlobRows(blob, from.fx, HiddenSize(from.fx), L, &blob_rows) && rows == blob_rows,
		              "%s: the restored rows at B = %lld equal the blob's", name, static_cast<long long>(to.B));
		PKV_CHECK_MSG(bc::DecodeN(to.fx.model, r, 8) == bc::DecodeN(from.fx.model, s, 8), "%s: next 8 tokens", name);
		sslm_seq_release(r);
	}
	sslm_seq_release(s);
}

void Cell67Layouts() {
	struct Dir { const char* name; int64_t from_pin, to_pin; };
	const Dir dirs[] = {{"pinned 16 -> derived", 16, 0}, {"derived -> pinned 1", 0, 1}, {"one-page view -> derived", 4096, 0}};
	for (const Dir& d : dirs)
		for (int32_t budget : {0, 1200}) {
			Side from, to;
			if (!MakeSide(d.from_pin, &from) || !MakeSide(d.to_pin, &to)) continue;
			CrossLayout(Fmt("%s, %s", d.name, budget ? "SSB6" : "SSB5").c_str(), from, to, budget);
		}
	// The gather mutant (physical page order instead of logical) is 6.6b's: its churned member asserts a scattered
	// table and grades the save against the flat twin (T-3196 F-10). These fresh-pool layouts ascend in physical id,
	// and 6.8 compares a restore with its own blob, so neither separates it.
}

// v1.11.0's contiguous blobs (the pinned SSB5 blobs) restore at the derived B, tokens equal the reference.
void Cell67OldBlobs() {
	const Fixture& def = GetFixture("pkv_def");
	if (!def.ok) return;
	PKV_CHECK_MSG(sslm_kv_page_positions(def.model) == RuleB(48, 4096, 1), "pkv_def runs at B = %lld, the derived 128 owed",
	              static_cast<long long>(sslm_kv_page_positions(def.model)));
	const std::vector<uint8_t> blob = Pin("v1.11.0", "persist", "saved");
	const RefRecord* cont = RefLookup(Reference("v1.11.0", def), "persist", "restored+continuation8");
	const RefRecord* saved = RefLookup(Reference("v1.11.0", def), "persist", "saved");
	LegacyPool pool(def.model, 1);
	sslm_seq r = nullptr;
	PKV_CHECK_EQ(sslm_seq_restore(def.model, &pool.pool, blob.data(), blob.size(), &r), SSLM_OK);
	if (!r || !cont || !saved) return;
	std::vector<uint8_t> rows;
	PKV_CHECK_MSG(SeqRows(pool.pool, r, def.geo, 1, 128, saved->context_length, Rot::kRotated, &rows) &&
	                  Sha(rows.data(), rows.size()) == saved->rows_sha,
	              "the restored v1.11.0 blob's rows, read at the derived B rotated, equal the reference");
	PKV_CHECK(bc::DecodeN(def.model, r, 8) == cont->tokens);
	sslm_seq_release(r);
}

// The re-share branch across layouts (T-3162 G-2; T-3196 F-1). The identity compare spans
// floor(origin/Bt) * Bt positions at the restoring Bt (SharingHandleMatches' memcmp), so a blob whose prefix
// differs from the live prefix's only inside [floor(origin/Bt) * Bt, origin) re-shares exactly floor(origin/Bt)
// pages, and one that differs inside the span takes the private path and shares none.
//
// Members, and the mutant each separates (plan cell 6.7's two: a span at the saving grain, a span of `origin`):
//   16 -> derived 128, origin 1,200, differs on [1,152, 1,200): must share 9. The saving grain's span is 1,200
//        (1,200 is a multiple of 16) and the span of `origin` is 1,200: both compare the differing tail, take the
//        private path and share 0. Both mutants die here.
//   16 -> derived 128, origin 1,024, no difference: must share 8 (aligned; neither mutant separates).
//   derived 128 -> pinned 1, origins 1,200 and 1,024, no difference: must share 1,200 and 1,024 (at B = 1 every
//        origin is aligned, so there is no in-page tail to differ on; neither mutant separates).
//   derived 128 -> pinned 1, origin 1,200, differs on [1,152, 1,200): must share 0 (the restoring span, 1,200,
//        covers the difference). The saving grain's span is 1,152, which stops short of it: that mutant passes the
//        compare and shares. This is the saving-grain mutant's separation in the derived -> pinned 1 direction.
//   The span of `origin` is the correct span at B = 1 (floor(origin/1) * 1 = origin), an equivalent mutant in this
//   direction by arithmetic; it is separated in the 16 -> derived direction above, and nowhere else.
// (Executed on 2ebe5e3 with the engine's B set by CFG1, both grains: the record's F-1 row.)
void Cell67Reshare() {
	struct Member { int64_t from_pin, to_pin; int32_t origin, differ_from; int64_t owed; const char* kills; };
	const Member members[] = {
	    {16, 0, 1200, 1152, 9, "the saving-grain span and the span of origin"},
	    {16, 0, 1024, 1024, 8, "-"},
	    {0, 1, 1200, 1200, 1200, "-"},
	    {0, 1, 1024, 1024, 1024, "-"},
	    {0, 1, 1200, 1152, 0, "the saving-grain span"},
	};
	for (const Member& m : members) {
		Side from, to;
		if (!MakeSide(m.from_pin, &from) || !MakeSide(m.to_pin, &to)) continue;
		const int64_t Bt = to.B;
		// The source prefix agrees with the live prefix on [0, differ_from) and differs on [differ_from, origin).
		std::vector<int32_t> live = Stream(51, m.origin, to.fx.vocab), src = live;
		for (int64_t i = m.differ_from; i < m.origin; ++i)
			src[static_cast<size_t>(i)] = (src[static_cast<size_t>(i)] + 1) % to.fx.vocab;
		const bool must_share = m.owed > 0;
		PKV_CHECK_MSG(must_share == (m.differ_from >= (m.origin / Bt) * Bt),
		              "construction: the difference [%d, %d) lies %s the restoring span [0, %lld)", m.differ_from, m.origin,
		              must_share ? "outside" : "inside", static_cast<long long>((m.origin / Bt) * Bt));
		const Lay yf = LayoutOf(from.fx.geo, 1, from.B), yt = LayoutOf(to.fx.geo, 1, to.B);
		PagePoolOf pf(from.fx.model, static_cast<uint32_t>(2 * yf.pages + 4), yf.page_bytes);
		PagePoolOf pt(to.fx.model, static_cast<uint32_t>(2 * yt.pages + 4), yt.page_bytes);
		if (pf.status != SSLM_OK || pt.status != SSLM_OK) {
			PKV_CHECK_MSG(false, "pools: %s / %s", AbiStatusName(pf.status), AbiStatusName(pt.status));
			continue;
		}
		sslm_prefix sp = bc::BudgetPrefix(from.fx, &pf.pool, src);
		sslm_prefix lp = bc::BudgetPrefix(to.fx, &pt.pool, live);
		sslm_seq s = nullptr, r = nullptr;
		PKV_CHECK_EQ(sslm_seq_create_budgeted(from.fx.model, &pf.pool, 64, &s), SSLM_OK);
		if (sp && lp && s) {
			PKV_CHECK_EQ(sslm_seq_adopt_prefix(s, sp), SSLM_OK);
			bc::DecodeN(from.fx.model, s, 4);
			const std::vector<uint8_t> blob = bc::Save(s);
			uint32_t shared = 999;
			PKV_CHECK_EQ(sslm_seq_restore_shared(to.fx.model, &pt.pool, blob.data(), blob.size(), lp, &r, &shared), SSLM_OK);
			PKV_CHECK_MSG(shared == static_cast<uint32_t>(m.owed),
			              "%lld -> %lld at origin %d, differing on [%d, %d): shared %u pages, %lld owed (kills: %s)",
			              static_cast<long long>(from.B), static_cast<long long>(Bt), m.origin, m.differ_from, m.origin,
			              shared, static_cast<long long>(m.owed), m.kills);
			if (r) {
				std::vector<uint8_t> rows, blob_rows;
				const int64_t L = m.origin + 3;
				PKV_CHECK_MSG(SeqRows(pt.pool, r, to.fx.geo, 1, Bt, L, Rot::kRotated, &rows) &&
				                  BlobRows(blob, from.fx, HiddenSize(from.fx), L, &blob_rows) && rows == blob_rows,
				              "origin %d: the restored rows equal the blob's", m.origin);
				PKV_CHECK_MSG(bc::DecodeN(to.fx.model, r, 4) == bc::DecodeN(from.fx.model, s, 4),
				              "origin %d: the restored holder's next 4 tokens equal the source's", m.origin);
				sslm_seq_release(r);
			}
		}
		if (s) sslm_seq_release(s);
		if (sp) sslm_prefix_release(sp);
		if (lp) sslm_prefix_release(lp);
	}
}

// ---- 6.8 -----------------------------------------------------------------------------------------

// A reserve scattered by churn (7.9's construction: budget-1 creates and releases in a fixed
// shuffled order), then the whole-page operations, each checked through the peek at the derived B.
void Cell68() {
	const Fixture& def = GetFixture("pkv_def");
	if (!def.ok) return;
	const int64_t B = RuleB(48, 4096, 1);
	PKV_CHECK_MSG(sslm_kv_page_positions(def.model) == B, "pkv_def runs at B = %lld, %lld owed",
	              static_cast<long long>(sslm_kv_page_positions(def.model)), static_cast<long long>(B));
	const Lay y = LayoutOf(def.geo, 1, B);
	PagePoolOf pool(def.model, 120, y.page_bytes);
	PKV_CHECK_EQ(pool.status, SSLM_OK);
	if (!pool.pool) return;
	{  // churn: 60 two-page holders made, then released in a fixed shuffled order
		std::vector<sslm_seq> made;
		for (int i = 0; i < 60; ++i) {
			sslm_seq s = nullptr;
			if (sslm_seq_create_budgeted(def.model, &pool.pool, 1, &s) == SSLM_OK) made.push_back(s);
		}
		const std::vector<uint32_t> order = Shuffled(static_cast<uint32_t>(made.size()), 99991u);
		for (uint32_t i : order) sslm_seq_release(made[i]);
	}
	sslm_prefix px = bc::BudgetPrefix(def, &pool.pool, Stream(51, 1000, def.vocab));  // mid-page: 1,000 = 7 * 128 + 104
	sslm_seq a = nullptr;
	sslm_prefix child = nullptr;
	PKV_CHECK_EQ(sslm_seq_create_budgeted(def.model, &pool.pool, 64, &a), SSLM_OK);
	if (!px || !a) return;
	// The budget adopt copies the tail page into the reserve page at the same logical index (7).
	PKV_CHECK_EQ(sslm_seq_adopt_prefix(a, px), SSLM_OK);
	const std::vector<uint8_t> pblob_rows = [&] {
		std::vector<uint32_t> t;
		uint32_t page = 0, mapped = 0;
		for (uint32_t i = 0; sslm_pkv_test_only_prefix_table_entry(px, i, &page, &mapped) == SSLM_OK && i < mapped; ++i)
			t.push_back(page);
		std::vector<uint8_t> rows;
		RowsFromTable(pool.pool, t, def.geo, 1, B, 1000, Rot::kRotated, &rows);
		return rows;
	}();
	std::vector<uint8_t> arows;
	PKV_CHECK_MSG(SeqRows(pool.pool, a, def.geo, 1, B, 1000, Rot::kRotated, &arows) && !pblob_rows.empty() &&
	                  arows == pblob_rows,
	              "adopt: the adopter's rows [0, 1,000) at the rotated layout equal the prefix's (the tail copy kept "
	              "logical index 7)");
	// begin_from's tail copy.
	PKV_CHECK_EQ(sslm_prefix_begin_from(px, 64, &child), SSLM_OK);
	if (child) {
		std::vector<uint32_t> t;
		uint32_t page = 0, mapped = 0;
		for (uint32_t i = 0; sslm_pkv_test_only_prefix_table_entry(child, i, &page, &mapped) == SSLM_OK && i < mapped; ++i)
			t.push_back(page);
		std::vector<uint8_t> crows;
		PKV_CHECK_MSG(RowsFromTable(pool.pool, t, def.geo, 1, B, 1000, Rot::kRotated, &crows) && crows == pblob_rows,
		              "begin_from: the child's rows equal the parent's");
	}
	// A materialized restore (no handle): rows equal the blob's.
	bc::DecodeN(def.model, a, 4);
	const std::vector<uint8_t> blob = bc::Save(a);
	sslm_seq m = nullptr;
	PKV_CHECK_EQ(sslm_seq_restore(def.model, &pool.pool, blob.data(), blob.size(), &m), SSLM_OK);
	if (m) {
		std::vector<uint8_t> mrows, brows;
		PKV_CHECK_MSG(SeqRows(pool.pool, m, def.geo, 1, B, 1003, Rot::kRotated, &mrows) &&
		                  BlobRows(blob, def, HiddenSize(def), 1003, &brows) && mrows == brows,
		              "materialized restore: rows equal the blob's");
	}
	// Release poison: the released holder's dirty pages read 0xCD; nobody else's page changes.
	std::vector<uint32_t> mine;
	legacy_a::SeqTable(a, &mine);
	std::vector<uint32_t> others;
	if (m) legacy_a::SeqTable(m, &others);
	std::map<uint32_t, std::vector<uint8_t>> before;
	for (uint32_t p : others) before[p] = legacy_a::PeekPage(pool.pool, p, y.page_bytes);
	std::set<uint32_t> shared_with_prefix;
	{
		uint32_t page = 0, mapped = 0;
		for (uint32_t i = 0; sslm_pkv_test_only_prefix_table_entry(px, i, &page, &mapped) == SSLM_OK && i < mapped; ++i)
			shared_with_prefix.insert(page);
	}
	sslm_seq_release(a);
	a = nullptr;
	for (uint32_t p : mine)
		if (!shared_with_prefix.count(p))
			PKV_CHECK_MSG(legacy_a::AllBytesAre(legacy_a::PeekPage(pool.pool, p, y.page_bytes), 0xCD),
			              "release: the holder's private page %u is not poisoned", p);
	for (auto& kv : before)
		PKV_CHECK_MSG(legacy_a::PeekPage(pool.pool, kv.first, y.page_bytes) == kv.second,
		              "release: another holder's page %u changed", kv.first);
	// Separation of 6.8's mutants: a tail copy re-rotated by the destination's physical id, or written at
	// logical index 0, places row 1,000 - 1 elsewhere than (r + 7) mod runs.
	PKV_CHECK(InPage(def.geo, 1, B, 0, 0, 0, 999, Rot::kPhysical, 5, 120) != InPage(def.geo, 1, B, 0, 0, 0, 999));
	PKV_CHECK(InPage(def.geo, 1, B, 0, 0, 0, 999 % B) != InPage(def.geo, 1, B, 0, 0, 0, 999));
	if (m) sslm_seq_release(m);
	if (child) sslm_prefix_release(child);
	sslm_prefix_release(px);
}

PKV_CELL("6.6a/T1", "T1", Cell66a);
PKV_CELL("6.6b/T1", "T1", Cell66b);
PKV_CELL("6.7/T1:layouts", "T1", Cell67Layouts);
PKV_CELL("6.7/T1:old-blobs", "T1", Cell67OldBlobs);
PKV_CELL("6.7/T1:reshare", "T1", Cell67Reshare);
PKV_CELL("6.8/T1", "T1", Cell68);

}  // namespace

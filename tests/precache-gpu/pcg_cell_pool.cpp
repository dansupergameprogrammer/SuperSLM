// Precached-prefixes plan §7, step F1 (green at F2): the GPU restore against the pool.
//   1.8 [F2]   GPU lifetime: restore A, release it, restore B into exactly A's pages (the pool left with
//              exactly ceil(L/B) free, A's; confirmed by page index), A's and B's rows differing on every
//              page; B's adopter tokens equal those of B's prefill-built twin. (The leg "device rows of B's
//              private pages, read back, equal the host mirror" has no device read-back seam on this tree;
//              the adopter tokens are the operand that sees the device rows. Recorded in the build record.)
//   2.6 [F2]   (2.7's hint half) a nested blob restored with hints failing each of (b)-(f) alone -- an
//              unfrozen parent, a parent of another model, of another pool, a parent whose length is not
//              the origin, and one whose rows differ in the last byte of the last shared page: private,
//              SSLM_OK, ceil(L/B) pages drawn (fill probe), re-saved byte-identical, the foreign parent's
//              pool unchanged. Matching hints from a prefill-built parent and a restored one share
//              floor(origin/B) and draw the rest.
//   5.10 [F2]  GPU twin: B = 16, cap 4096, a frozen parent of 1,000 resident (63 pages) and exactly
//              R(200) = 14 pages free. Control: begin_from(parent, 200), prefill to 1,200 and freeze are
//              admitted. Each private-hint disposition -> SSLM_GPU_KV_POOL_EXHAUSTED, nothing drawn (14
//              free before and after); the matching hint shares 62, draws 13 and is admitted.
//   keeps      X1/X3 on the GPU: adopters of a restored root prefix, and of a nested child restored with
//              and without sharing, emit the tokens adopters of the prefill-built prefix emit (6.2/6.3's
//              shape; 6.5 [F3] runs them on both certified GPUs).
//
// RED AT F1: the save stub returns SSLM_DEVICE_LOST, so no blob exists and the first check fails.
//
// Run: pcg_cell_pool.exe --fixture-dir=DIR (pkv_def).
#include "pcg_helpers.h"

#include <cstdio>

using namespace pcg;

namespace {

constexpr int32_t kWorld = 1000, kPersona = 200, kChild = kWorld + kPersona;
constexpr int64_t kTokens = 16;

struct PoolState : ProbeState {
	SslmGpuContext* c;
	SslmGpuModelHandle* m;
	SslmGpuKvPoolHandle* p;
	PoolState(SslmGpuContext* cc, SslmGpuModelHandle* mm, SslmGpuKvPoolHandle* pp) : c(cc), m(mm), p(pp) {}
	SslmGpuKvPoolHandle* Pool() override { return p; }
	SslmGpuContext* Ctx() override { return c; }
	SslmGpuModelHandle* Model() override { return m; }
	void DrainAll() override {}
};
bool FreeIs(const Env& e, Rig& rig, int64_t k, const std::string& what) {
	PoolState s(rig.ctx, rig.model, rig.pool);
	return ProbeExactlyFreeOneState(e, s, k, what.c_str());
}
// Exactly one page free, where the one-state probe (k >= 3) cannot read: a 2-page create is refused for
// capacity (at most 1 free), and the restore of a one-position blob, which draws exactly one page, is
// admitted (at least 1 free) and released again.
bool FreeIsOne(const Env& e, Rig& rig, const std::vector<uint8_t>& one, const std::string& what) {
	PoolState ps(rig.ctx, rig.model, rig.pool);
	const bool at_most = !TwoPageCreateAdmitted(e, ps);
	CHECK_MSG(at_most, "%s: a 2-page create was admitted: more than one page free", what.c_str());
	const RestoredPrefix r = RestorePrefix(rig, one);
	CHECK_MSG(r.status == GS::SSLM_OK && r.p, "%s: a one-page restore returned %s: no page free", what.c_str(), Name(r.status));
	if (r.p) {
		rig.prefixes.pop_back();
		sslm_gpu_prefix_release(rig.ctx, r.p);
	}
	return at_most && r.p != nullptr;
}

// A blob's rows of positions [p0, p1), every layer, half and head, concatenated.
std::vector<uint8_t> Rows(const Env& e, const std::vector<uint8_t>& b, int64_t p0, int64_t p1) {
	const Spx1 v = Spx1Parse(b, e.geo);
	std::vector<uint8_t> out;
	if (!v.size_exact) return out;
	for (uint32_t l = 0; l < e.geo.L; ++l)
		for (uint32_t half = 0; half < 2; ++half)
			for (uint32_t h = 0; h < e.geo.Hkv; ++h) {
				const size_t o = Spx1RowOff(e.geo, v.length, l, half, h, p0);
				out.insert(out.end(), b.begin() + static_cast<std::ptrdiff_t>(o),
				           b.begin() + static_cast<std::ptrdiff_t>(o + static_cast<size_t>(p1 - p0) * e.geo.D));
			}
	return out;
}

// ---- 1.8 ------------------------------------------------------------------------------------------------

void Cell18(const Env& e) {
	constexpr int32_t L = 40;  // ceil(40/16) = 3 pages >= 2
	const int64_t P = e.geo.PrefixPages(L);
	const std::vector<int32_t> ta = PrefixTokens(L, 11), tb = PrefixTokens(L, 12);
	std::vector<uint8_t> blob_a, blob_b;
	std::vector<int32_t> twin;
	{
		Rig twin_rig(e, static_cast<uint32_t>(2 * P + e.geo.R(64) + 2));
		SslmGpuPrefixHandle* a = Prefix(twin_rig, ta);
		SslmGpuPrefixHandle* b = Prefix(twin_rig, tb);
		if (!a || !b) return;
		blob_a = SavePrefix(e, a);
		blob_b = SavePrefix(e, b);
		twin = AdopterTokens(twin_rig, b, L, kTokens, tb.back(), 64);
	}
	if (blob_a.empty() || blob_b.empty()) return;
	// A's and B's rows differ on every page.
	for (int64_t j = 0; j < P; ++j) {
		const int64_t p1 = std::min<int64_t>((j + 1) * e.geo.B, L);
		CHECK_MSG(Rows(e, blob_a, j * e.geo.B, p1) != Rows(e, blob_b, j * e.geo.B, p1), "1.8: A's and B's rows agree on page %lld",
		          static_cast<long long>(j));
	}
	// The pool: the adopter's R(64) pages first, then exactly P free.
	Rig rig(e, static_cast<uint32_t>(e.geo.R(64) + P));
	GSeq adopter = rig.Budgeted(64);
	if (!adopter.h) return;
	FreeIs(e, rig, P, "1.8: exactly ceil(L/B) free before A");
	RestoredPrefix ra = RestorePrefix(rig, blob_a);
	CHECK_MSG(ra.status == GS::SSLM_OK && ra.p, "1.8: restore A: %s", Name(ra.status));
	if (!ra.p) return;
	std::vector<uint32_t> pa = TablePages(ra.p);
	rig.prefixes.pop_back();
	CHECK(sslm_gpu_prefix_release(rig.ctx, ra.p) == GS::SSLM_OK);
	RestoredPrefix rb = RestorePrefix(rig, blob_b);
	CHECK_MSG(rb.status == GS::SSLM_OK && rb.p, "1.8: restore B: %s", Name(rb.status));
	if (!rb.p) return;
	std::vector<uint32_t> pb = TablePages(rb.p);
	std::sort(pa.begin(), pa.end());
	std::sort(pb.begin(), pb.end());
	CHECK_MSG(static_cast<int64_t>(pa.size()) == P && pa == pb, "1.8: B's pages are not A's (A %zu pages, B %zu)", pa.size(),
	          pb.size());
	// B's mirror rows are B's (the host half of the leg).
	const std::vector<uint32_t> order = TablePages(rb.p);
	size_t bad = 0;
	for (int64_t pos = 0; pos < L && static_cast<int64_t>(order.size()) == P; ++pos) {
		const std::vector<uint8_t> page = PeekPage(e, rig.pool, order[static_cast<size_t>(pos / e.geo.B)]);
		for (uint32_t l = 0; l < e.geo.L; ++l)
			for (uint32_t half = 0; half < 2; ++half)
				for (uint32_t h = 0; h < e.geo.Hkv; ++h)
					bad += std::memcmp(page.data() + e.geo.InPage(l, half, h, pos),
					                   blob_b.data() + Spx1RowOff(e.geo, L, l, half, h, pos), e.geo.D) != 0;
	}
	CHECK_MSG(bad == 0, "1.8: %zu of B's rows in the host mirror are not B's", bad);
	// B's adopter tokens equal B's twin: the device rows are B's, not A's.
	const GS st = Adopt(rig, adopter, rb.p, L);
	CHECK_MSG(st == GS::SSLM_OK, "1.8: adopt B: %s", StatusName(st));
	adopter.last = tb.back();
	const std::vector<int32_t> got = st == GS::SSLM_OK ? DecodeN(e, adopter, kTokens) : std::vector<int32_t>{};
	CHECK_MSG(got == twin && static_cast<int64_t>(twin.size()) == kTokens,
	          "1.8: B's adopter emitted %zu tokens, %s B's prefill-built twin (%zu)", got.size(), got == twin ? "equal to" : "NOT equal to",
	          twin.size());
}

// ---- 2.6's GPU twin, 5.10's GPU twin, X3/X4 keeps ---------------------------------------------------------

void Nested(const Env& e) {
	const int64_t B = e.geo.B;
	const int64_t s = kWorld / B;                                   // 62
	const int64_t child_pages = e.geo.PrefixPages(kChild);         // 75
	const int64_t world_pages = e.geo.PrefixPages(kWorld);         // 63
	const int64_t R200 = e.geo.R(kPersona);                        // 14
	CHECK_MSG(s == 62 && child_pages == 75 && world_pages == 63 && R200 == 14, "geometry: s %lld, child %lld, world %lld, R %lld",
	          static_cast<long long>(s), static_cast<long long>(child_pages), static_cast<long long>(world_pages),
	          static_cast<long long>(R200));
	// Pool P: world, its restored twin, its (f)-edited twin, an unfrozen world, and room to spare.
	const uint32_t total = static_cast<uint32_t>(3 * world_pages + e.geo.R(kWorld) + child_pages + 32);
	Rig rig(e, total);
	if (!rig.pool) return;
	SslmGpuPrefixHandle* world = Prefix(rig, WorldTokens());
	if (!world) return;
	const std::vector<uint8_t> wblob = SavePrefix(e, world);
	if (wblob.empty()) return;
	RestoredPrefix wr = RestorePrefix(rig, wblob);
	CHECK_MSG(wr.status == GS::SSLM_OK && wr.p, "nested: restoring the world: %s", Name(wr.status));
	std::vector<uint8_t> fblob(wblob);
	{
		// (f): the last byte of the last shared page, last layer, V, last head.
		const size_t o = Spx1RowOff(e.geo, kWorld, e.geo.L - 1, 1, e.geo.Hkv - 1, s * B - 1) + e.geo.D - 1;
		fblob[o] ^= 0x01;
		Redigest(fblob);
	}
	RestoredPrefix wf = RestorePrefix(rig, fblob);
	CHECK_MSG(wf.status == GS::SSLM_OK && wf.p, "nested: restoring the (f)-edited world: %s", Name(wf.status));
	SslmGpuPrefixHandle* unfrozen = nullptr;
	{
		GS st = sslm_gpu_prefix_begin(rig.ctx, rig.model, rig.pool, kWorld, &unfrozen);
		if (st == GS::SSLM_OK) st = PrefixPrefillAll(rig.ctx, unfrozen, WorldTokens(), 0, e.full_budget);
		CHECK_MSG(st == GS::SSLM_OK, "nested: the unfrozen world: %s", StatusName(st));
		rig.Prefix(unfrozen);
	}
	// (e): a frozen prefix of P whose rows [0, s * B) are the world's but whose length (1,001) is not the
	// origin: begin_from(world, 8) and one more token, frozen.
	SslmGpuPrefixHandle* longer = nullptr;
	{
		GS st = sslm_gpu_prefix_begin_from(rig.ctx, world, 8, &longer);
		if (st == GS::SSLM_OK) st = PrefixPrefillAll(rig.ctx, longer, Stream(97, 1, kVocab), 0, e.full_budget);
		if (st == GS::SSLM_OK) st = sslm_gpu_prefix_freeze(rig.ctx, longer);
		CHECK_MSG(st == GS::SSLM_OK, "nested: the 1,001-token parent: %s", StatusName(st));
		rig.Prefix(longer);
	}
	if (!wr.p || !wf.p || !unfrozen || !Real(longer)) return;
	// Pool Q (same model, another pool) and pool M (another model) hold a restored world each.
	Rig q(e, static_cast<uint32_t>(world_pages + 8));
	RestoredPrefix wq = RestorePrefix(q, wblob);
	GpuModelFixture other;
	if (!other.Open(e.fx.path, e.ctx)) return;
	Rig mrig(e, static_cast<uint32_t>(world_pages + 8), e.ctx, other.model);
	RestoredPrefix wm = RestorePrefix(mrig, wblob);
	CHECK_MSG(wq.p && wm.p, "nested: the worlds of pool Q (%s) and of another model (%s)", Name(wq.status), Name(wm.status));
	if (!wq.p || !wm.p) return;
	// A one-position blob (from pool Q), the one-page probe's operand.
	std::vector<uint8_t> one;
	{
		SslmGpuPrefixHandle* p1 = Prefix(q, Stream(5, 1, kVocab));
		if (p1) {
			one = SavePrefix(e, p1);
			q.prefixes.pop_back();
			sslm_gpu_prefix_release(q.ctx, p1);  // pool Q back to world + 8 free
		}
		CHECK_MSG(!one.empty(), "nested: the one-position blob");
		if (one.empty()) return;
	}

	// Fill pool P to exactly R(200) free (5.10's construction).
	const int64_t held = 3 * world_pages + e.geo.R(kWorld) + 1;  // + the 1,001-token parent's own page
	const int64_t free_now = static_cast<int64_t>(total) - held;
	std::vector<SslmGpuSequenceHandle*> filler;
	{
		PoolState ps(rig.ctx, rig.model, rig.pool);
		CHECK_MSG(AdmitPages(e.geo, ps, free_now - R200, &filler), "5.10: filling pool P to %lld free", static_cast<long long>(R200));
	}
	FreeIs(e, rig, R200, "5.10: exactly R(200) free");
	// Control: begin_from(world, 200), prefill to 1,200, freeze: admitted. Its save is the nested blob.
	std::vector<uint8_t> cblob;
	std::vector<int32_t> twin;
	{
		SslmGpuPrefixHandle* child = nullptr;
		GS st = sslm_gpu_prefix_begin_from(rig.ctx, world, kPersona, &child);
		if (st == GS::SSLM_OK) st = PrefixPrefillAll(rig.ctx, child, PersonaTokens(0), 0, e.full_budget);
		if (st == GS::SSLM_OK) st = sslm_gpu_prefix_freeze(rig.ctx, child);
		CHECK_MSG(st == GS::SSLM_OK, "5.10 control: begin_from(parent, 200), prefill to 1,200, freeze: %s", StatusName(st));
		if (Real(child)) {
			cblob = SavePrefix(e, child);
			// The twin's adopter tokens, from pool Q (the adopter needs pages pool P does not have to spare).
			sslm_gpu_prefix_release(rig.ctx, child);
		}
	}
	if (cblob.empty()) return;
	const Spx1 cv = Spx1Parse(cblob, e.geo);
	CHECK_MSG(cv.length == kChild && cv.origin == kWorld, "nested blob: L %lld origin %lld", static_cast<long long>(cv.length),
	          static_cast<long long>(cv.origin));
	FreeIs(e, rig, R200, "5.10: R(200) free again after the control's release");
	struct Hint {
		const char* name;
		SslmGpuPrefixHandle* p;
		Rig* foreign;  // the hint's own pool, when it is not P
		int64_t foreign_free;
	};
	const Hint failing[] = {
	    {"null", nullptr, nullptr, 0},
	    {"(b) unfrozen", unfrozen, nullptr, 0},
	    {"(c) another model", wm.p, &mrig, 8},
	    {"(d) another pool", wq.p, &q, 8},
	    {"(e) context_length != origin", longer, nullptr, 0},
	    {"(f) rows differ in the last byte of the last shared page", wf.p, nullptr, 0},
	};
	for (const Hint& h : failing) {
		const RestoredPrefix r = RestorePrefix(rig, cblob, h.p);
		Refused(r, GS::SSLM_GPU_KV_POOL_EXHAUSTED, std::string("5.10: private hint ") + h.name + " at 14 free");
		FreeIs(e, rig, R200, std::string("5.10: nothing drawn, hint ") + h.name);
	}
	// The must-accept control: the matching hint shares 62 and draws 13.
	{
		const RestoredPrefix r = RestorePrefix(rig, cblob, world);
		CHECK_MSG(r.status == GS::SSLM_OK && r.p, "5.10: the matching hint at 14 free: %s", Name(r.status));
		if (r.p) {
			FreeIsOne(e, rig, one, "5.10: the matching hint drew ceil(L/B) - s = 13");
			const std::vector<uint8_t> re = SavePrefix(e, r.p);
			CHECK_MSG(re == cblob, "X3: the shared restore re-saves byte-identical");
			rig.prefixes.pop_back();
			sslm_gpu_prefix_release(rig.ctx, r.p);
		}
	}
	// A restored parent matches too.
	{
		const RestoredPrefix r = RestorePrefix(rig, cblob, wr.p);
		CHECK_MSG(r.status == GS::SSLM_OK && r.p, "2.6: a restored parent as the hint at 14 free: %s", Name(r.status));
		if (r.p) {
			rig.prefixes.pop_back();
			sslm_gpu_prefix_release(rig.ctx, r.p);
		}
	}
	// 2.6: room for a private restore; each failing hint gives SSLM_OK, ceil(L/B) pages drawn, the bytes kept.
	for (auto it = filler.rbegin(); it != filler.rend(); ++it) sslm_gpu_seq_release(rig.ctx, *it);
	filler.clear();
	FreeIs(e, rig, free_now, "2.6: the filler released");
	const int64_t base = free_now;
	for (const Hint& h : failing) {
		const RestoredPrefix r = RestorePrefix(rig, cblob, h.p);
		CHECK_MSG(r.status == GS::SSLM_OK && r.p, "2.6: hint %s: %s", h.name, Name(r.status));
		if (!r.p) continue;
		FreeIs(e, rig, base - child_pages, std::string("2.6: hint ") + h.name + ": private, ceil(L/B) = 75 drawn");
		CHECK_MSG(SavePrefix(e, r.p) == cblob, "2.6: hint %s: the private restore re-saves byte-identical", h.name);
		if (h.foreign) FreeIs(e, *h.foreign, h.foreign_free, std::string("2.6: hint ") + h.name + ": its pool unchanged");
		rig.prefixes.pop_back();
		sslm_gpu_prefix_release(rig.ctx, r.p);
	}
	// X3/X4 keeps: adopters of the shared and the private restore emit the same tokens (pool Q has the room).
	{
		Rig a(e, static_cast<uint32_t>(2 * child_pages + world_pages + 2 * e.geo.R(256) + 4));
		RestoredPrefix aw = RestorePrefix(a, wblob);
		RestoredPrefix shared = aw.p ? RestorePrefix(a, cblob, aw.p) : RestoredPrefix{};
		RestoredPrefix priv = RestorePrefix(a, cblob);
		if (shared.p && priv.p) {
			const std::vector<int32_t> t1 = AdopterTokens(a, shared.p, kChild, kTokens, PersonaTokens(0).back());
			const std::vector<int32_t> t2 = AdopterTokens(a, priv.p, kChild, kTokens, PersonaTokens(0).back());
			CHECK_MSG(static_cast<int64_t>(t1.size()) == kTokens && t1 == t2,
			          "X3/X4: adopters of the shared and the private restore differ (%zu, %zu tokens)", t1.size(), t2.size());
			twin = t1;
		} else {
			CHECK_MSG(false, "X3/X4: restores in the adopter pool: %s, %s", Name(shared.status), Name(priv.status));
		}
	}
	// The prefill-built twin's adopter (the persona built by begin_from in a fresh pool).
	{
		Rig t(e, static_cast<uint32_t>(world_pages + R200 + e.geo.R(256) + 4));
		GChain c = BuildPersonaChain(t, 0, false);
		if (c.ok && !twin.empty()) {
			const std::vector<int32_t> t3 = AdopterTokens(t, c.persona, kChild, kTokens, PersonaTokens(0).back());
			CHECK_MSG(t3 == twin, "X3: the restored child's adopter tokens differ from the prefill-built persona's");
		}
	}
}

// X1 keeps: a restored root prefix's adopter tokens equal the prefill-built prefix's.
void Root(const Env& e) {
	Rig rig(e, 64);
	const std::vector<int32_t> t = PrefixTokens(40, 21);
	SslmGpuPrefixHandle* p = Prefix(rig, t);
	if (!p) return;
	const std::vector<uint8_t> b = SavePrefix(e, p);
	if (b.empty()) return;
	const RestoredPrefix r = RestorePrefix(rig, b);
	CHECK_MSG(r.status == GS::SSLM_OK && r.p, "X1: restore: %s", Name(r.status));
	if (!r.p) return;
	const std::vector<int32_t> a1 = AdopterTokens(rig, p, 40, kTokens, t.back());
	const std::vector<int32_t> a2 = AdopterTokens(rig, r.p, 40, kTokens, t.back());
	CHECK_MSG(static_cast<int64_t>(a1.size()) == kTokens && a1 == a2, "X1: the restored prefix's adopter differs (%zu, %zu)",
	          a1.size(), a2.size());
	// begin_from a restored prefix behaves as from the original: the children's saves are identical.
	SslmGpuPrefixHandle *c1 = nullptr, *c2 = nullptr;
	CHECK(sslm_gpu_prefix_begin_from(rig.ctx, p, 8, &c1) == GS::SSLM_OK);
	CHECK(sslm_gpu_prefix_begin_from(rig.ctx, r.p, 8, &c2) == GS::SSLM_OK);
	rig.Prefix(c1);
	rig.Prefix(c2);
	if (Real(c1) && Real(c2)) {
		CHECK(PrefixPrefillAll(rig.ctx, c1, PrefixTokens(8, 22), 0, e.full_budget) == GS::SSLM_OK);
		CHECK(PrefixPrefillAll(rig.ctx, c2, PrefixTokens(8, 22), 0, e.full_budget) == GS::SSLM_OK);
		CHECK(sslm_gpu_prefix_freeze(rig.ctx, c1) == GS::SSLM_OK && sslm_gpu_prefix_freeze(rig.ctx, c2) == GS::SSLM_OK);
		CHECK_MSG(SavePrefix(e, c1) == SavePrefix(e, c2), "X1: begin_from children of the original and the restored differ");
	}
}

}  // namespace

int main(int argc, char** argv) {
	ParseFixtureArgs(argc, argv);
	{
		Env e;
		if (e.Open(FixturePath(argc, argv))) {
			Cell18(e);
			Root(e);
			Nested(e);
		}
	}
	return FinishSuite("pcg_cell_pool");
}

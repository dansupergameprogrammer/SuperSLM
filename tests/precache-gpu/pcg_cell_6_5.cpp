// Precached-prefixes plan §7 6.5 [F3]: "GPU: 6.1-6.4 on both vendors; GPU adopter tokens equal the GPU prefill twin's."
// One run per card per model; the cell is the same on both vendors.
//   6.1  for each shape: the restored prefix's rows, read from the pool's host mirror through its own table, byte-equal
//        the saved prefix's (every layer, K and V, head and position), and the re-saved blob's state fields (residual,
//        scale m and e, the five counts, current_token, schema hash, walk, L, origin) equal the saved blob's.
//   6.2  adopters of the restored prefix and of the prefill twin, 64 greedy steps each, with and without a bound schema:
//        identical outcome streams (a refused step is recorded as its status, so a prefix at the cap compares too).
//   6.3  a nested child restored with sharing (the restored parent as its hint), without (no hint), and built by
//        begin_from + prefill: the three children's adopters emit identical streams, and the shared restore's first
//        floor(origin/B) table pages are the parent's own pages (the GPU's peek of a page index, as the CPU's
//        "peek addresses equal").
//   6.4  save -> restore -> save byte-identical, for root, nested, empty and schema-bound prefixes.
//
// Shapes (6.1's "each shape of 4.1"): on pkv_def (the GPU suite's fixture, B = 16, cap 4,096), L in {0, 1, B-1, B,
// B+1, 1,000, cap-1, cap}, with the bound legs at L in {1, B+1, 1,000}. On a real artifact (--model), the shipped
// geometry: L in {0, 1, B-1, B, B+1, 1,000} and the nested world -> persona of tests/t3035-precache-refs/prompts (world.ids, 1,012
// ids; persona00.ids). The plan's other two readings of 6.1's shapes have no GPU counterpart, and the record says so:
// the synthetic 0.5B/1.5B shapes are CPU-suite inputs with no GPU fixture on the box or the kit, and the GPU has no
// damped-greedy decode and one adopter mode (a pooled budgeted sequence), so 6.2 runs greedy, budgeted.
//
// Run:
//   pcg_cell_6_5.exe --fixture-dir=DIR                                   pkv_def's shapes
//   pcg_cell_6_5.exe --model=FILE --tag=0p5|1p5 --world=FILE --persona=FILE   a real artifact
// Configuration: plain.
#include "pcg_f3_common.h"

#include <cstdio>

using namespace pcg;
using namespace pcg::f3;

namespace {

constexpr int64_t kSteps = 64;

// 6.1's rows: the restored prefix's table pages, peeked, against the blob's canonical rows.
void RowsEqual(const Env& e, Rig& rig, SslmGpuPrefixHandle* r, const std::vector<uint8_t>& blob, const std::string& what) {
	const Spx1 v = Spx1Parse(blob, e.geo);
	const std::vector<uint32_t> pages = TablePages(r);
	CHECK_MSG(static_cast<int64_t>(pages.size()) == e.geo.PrefixPages(v.length), "%s: 6.1 the restored prefix maps %zu "
	          "pages, want %lld", what.c_str(), pages.size(), static_cast<long long>(e.geo.PrefixPages(v.length)));
	if (static_cast<int64_t>(pages.size()) != e.geo.PrefixPages(v.length)) return;
	size_t bad = 0;
	std::vector<uint8_t> page;
	int64_t loaded = -1;
	for (int64_t pos = 0; pos < v.length; ++pos) {
		const int64_t j = pos / e.geo.B;
		if (j != loaded) {
			page = PeekPage(e, rig.pool, pages[static_cast<size_t>(j)]);
			loaded = j;
		}
		for (uint32_t l = 0; l < e.geo.L; ++l)
			for (uint32_t half = 0; half < 2; ++half)
				for (uint32_t h = 0; h < e.geo.Hkv; ++h)
					bad += std::memcmp(page.data() + e.geo.InPage(l, half, h, pos),
					                   blob.data() + Spx1RowOff(e.geo, v.length, l, half, h, pos), e.geo.D) != 0;
	}
	CHECK_MSG(bad == 0, "%s: 6.1 %zu restored rows differ from the saved blob's", what.c_str(), bad);
}

// 6.1's state fields and 6.4's identity: the restored prefix re-saved.
void Resave(const Env& e, SslmGpuPrefixHandle* r, const std::vector<uint8_t>& a, const std::string& what) {
	const std::vector<uint8_t> c = SavePrefix(e, r);
	if (c.empty()) return;
	const Spx1 va = Spx1Parse(a, e.geo), vc = Spx1Parse(c, e.geo);
	const bool state = va.m == vc.m && va.e == vc.e && va.token == vc.token && va.schema_hash == vc.schema_hash &&
	                   va.walk == vc.walk && va.length == vc.length && va.origin == vc.origin &&
	                   std::memcmp(va.counts, vc.counts, sizeof va.counts) == 0 && a.size() == c.size() &&
	                   std::memcmp(a.data() + kHeader, c.data() + kHeader, e.geo.hidden) == 0;
	CHECK_MSG(state, "%s: 6.1 the re-saved state fields differ (m %lld/%lld, e %lld/%lld, token %d/%d, walk %u/%u, "
	          "L %lld/%lld, origin %lld/%lld)", what.c_str(), static_cast<long long>(va.m), static_cast<long long>(vc.m),
	          static_cast<long long>(va.e), static_cast<long long>(vc.e), va.token, vc.token, va.walk, vc.walk,
	          static_cast<long long>(va.length), static_cast<long long>(vc.length), static_cast<long long>(va.origin),
	          static_cast<long long>(vc.origin));
	CHECK_MSG(a == c, "%s: 6.4 save -> restore -> save is not byte-identical (%zu vs %zu bytes)", what.c_str(), a.size(),
	          c.size());
}

// 6.2: adopters of the restored prefix and of its prefill twin. Returns the twin's stream.
std::vector<int32_t> SameAdopters(Rig& rig, SslmGpuPrefixHandle* twin, SslmGpuPrefixHandle* r, int64_t L,
                                  const std::string& what) {
	const int64_t cap = rig.e.geo.cap;
	const std::vector<int32_t> a = AdopterOutcomes(rig, twin, L, kSteps, static_cast<int32_t>(kSteps + 16));
	const std::vector<int32_t> b = AdopterOutcomes(rig, r, L, kSteps, static_cast<int32_t>(kSteps + 16));
	size_t first = 0;
	while (first < a.size() && first < b.size() && a[first] == b[first]) ++first;
	CHECK_MSG(a.size() == static_cast<size_t>(kSteps) && a == b, "%s: 6.2 the restored prefix's adopter differs from the "
	          "prefill twin's from step %zu (%zu and %zu steps)", what.c_str(), first, a.size(), b.size());
	// Two streams of identical refusals would pass the equality above: away from the empty prefix and the cap, the
	// twin's stream must open with a decoded token (code review 2026-10-24, N2).
	if (L >= 1 && L < cap - 1)
		CHECK_MSG(!a.empty() && a[0] >= 0, "%s: 6.2 the prefill twin's adopter decoded no first token (outcome %d)",
		          what.c_str(), a.empty() ? 0 : a[0]);
	return a;
}

// One root shape: built, saved, restored, and graded by 6.1, 6.2 and 6.4. Released before the next shape.
void RootShape(const Env& e, Rig& rig, const std::vector<int32_t>& tokens, int32_t schema, const std::string& what) {
	SslmGpuPrefixHandle* twin = BatchedPrefix(rig, tokens, schema);
	if (!twin) return;
	const std::vector<uint8_t> a = SavePrefix(e, twin);
	if (a.empty()) {
		Release(rig, twin);
		return;
	}
	const RestoredPrefix r = RestorePrefix(rig, a);
	CHECK_MSG(r.status == GS::SSLM_OK && r.p, "%s: restore %s", what.c_str(), Name(r.status));
	if (r.p) {
		RowsEqual(e, rig, r.p, a, what);
		Resave(e, r.p, a, what);
		const int64_t L = static_cast<int64_t>(tokens.size());
		const std::vector<int32_t> s = SameAdopters(rig, twin, r.p, L, what);
		std::printf("%s: L = %lld%s: rows, state, round trip and %lld adopter steps graded (first steps %s)\n", what.c_str(),
		            static_cast<long long>(L), schema >= 0 ? ", schema-bound" : "", static_cast<long long>(kSteps),
		            Join(std::vector<int32_t>(s.begin(), s.begin() + std::min<size_t>(s.size(), 4))).c_str());
		Release(rig, r.p);
	}
	Release(rig, twin);
}

// 6.3 and 6.4's nested leg: a world, a begin_from child; the child restored with the restored world as its hint, and
// without a hint; the three children's adopters equal; the shared restore's leading pages are the world's.
void Nested(const Env& e, Rig& rig, const std::vector<int32_t>& world, const std::vector<int32_t>& persona,
            const std::string& what) {
	SslmGpuPrefixHandle* w = BatchedPrefix(rig, world);
	if (!w) return;
	SslmGpuPrefixHandle* c = BatchedChild(rig, w, persona);
	if (!c) return;
	const std::vector<uint8_t> wb = SavePrefix(e, w), cb = SavePrefix(e, c);
	if (wb.empty() || cb.empty()) return;
	const RestoredPrefix wr = RestorePrefix(rig, wb);
	CHECK_MSG(wr.status == GS::SSLM_OK && wr.p, "%s: world restore %s", what.c_str(), Name(wr.status));
	if (!wr.p) return;
	Resave(e, wr.p, wb, what + " world");
	const RestoredPrefix shared = RestorePrefix(rig, cb, wr.p);
	const RestoredPrefix priv = RestorePrefix(rig, cb);
	CHECK_MSG(shared.status == GS::SSLM_OK && shared.p && priv.status == GS::SSLM_OK && priv.p,
	          "%s: child restores: with the hint %s, without %s", what.c_str(), Name(shared.status), Name(priv.status));
	if (!shared.p || !priv.p) return;
	const int64_t origin = static_cast<int64_t>(world.size()), L = origin + static_cast<int64_t>(persona.size());
	const int64_t s = e.geo.SharedPages(origin);
	const std::vector<uint32_t> wp = TablePages(wr.p), sp = TablePages(shared.p), pp = TablePages(priv.p);
	bool leading = static_cast<int64_t>(sp.size()) == e.geo.PrefixPages(L) && static_cast<int64_t>(wp.size()) >= s;
	for (int64_t i = 0; leading && i < s; ++i) leading = sp[static_cast<size_t>(i)] == wp[static_cast<size_t>(i)];
	CHECK_MSG(leading, "%s: 6.3 the shared restore's first %lld pages are not the restored world's (it maps %zu, the world "
	          "%zu)", what.c_str(), static_cast<long long>(s), sp.size(), wp.size());
	bool disjoint = static_cast<int64_t>(pp.size()) == e.geo.PrefixPages(L);
	for (uint32_t x : pp)
		for (uint32_t y : wp) disjoint = disjoint && x != y;
	CHECK_MSG(disjoint, "%s: 6.3 the unhinted restore shares a page with the world", what.c_str());
	CHECK_MSG(shared.shared == static_cast<uint32_t>(s), "%s: 6.3 the hinted restore reports %u shared pages, want %lld",
	          what.c_str(), shared.shared, static_cast<long long>(s));
	RowsEqual(e, rig, shared.p, cb, what + " shared child");
	RowsEqual(e, rig, priv.p, cb, what + " private child");
	Resave(e, shared.p, cb, what + " shared child");
	Resave(e, priv.p, cb, what + " private child");
	const std::vector<int32_t> built = AdopterOutcomes(rig, c, L, kSteps, static_cast<int32_t>(kSteps + 16));
	const std::vector<int32_t> a = AdopterOutcomes(rig, shared.p, L, kSteps, static_cast<int32_t>(kSteps + 16));
	const std::vector<int32_t> b = AdopterOutcomes(rig, priv.p, L, kSteps, static_cast<int32_t>(kSteps + 16));
	CHECK_MSG(built.size() == static_cast<size_t>(kSteps) && built == a && built == b,
	          "%s: 6.3 the three children's adopters differ (built/shared %s, built/private %s)", what.c_str(),
	          built == a ? "equal" : "DIFFER", built == b ? "equal" : "DIFFER");
	std::printf("%s: nested world %lld -> child %lld: shared %lld pages, three children's adopters %s\n", what.c_str(),
	            static_cast<long long>(origin), static_cast<long long>(L), static_cast<long long>(s),
	            built == a && built == b ? "equal" : "DIFFER");
	Release(rig, priv.p);
	Release(rig, shared.p);
	Release(rig, wr.p);
	Release(rig, c);
	Release(rig, w);
}

void PkvDef(const Env& e) {
	const Geo& g = e.geo;
	// The largest leg: a cap-length twin and its restored copy, an adopter of each, and a 1,200-position nested set.
	const uint32_t pages = static_cast<uint32_t>(2 * g.CapPages() + 4 * g.R(kSteps + 16) + 16);
	Rig rig(e, pages);
	if (!rig.pool || !RequireGpuGeometry(e, rig.pool, "6.5 pkv_def")) return;
	const int64_t B = g.B, cap = g.cap;
	for (int64_t L : {int64_t{0}, int64_t{1}, B - 1, B, B + 1, int64_t{1000}, cap - 1, cap})
		RootShape(e, rig, PrefixTokens(static_cast<int32_t>(L), 9), -1, "6.5 pkv_def root L=" + std::to_string(L));
	for (int64_t L : {int64_t{1}, B + 1, int64_t{1000}})
		RootShape(e, rig, PrefixTokens(static_cast<int32_t>(L), 11), 0, "6.5 pkv_def bound L=" + std::to_string(L));
	Nested(e, rig, WorldTokens(), PersonaTokens(0), "6.5 pkv_def nested");
}

void RealModel(const Env& e, int argc, char** argv) {
	const std::string tag = Arg(argc, argv, "--tag");
	const std::vector<int32_t> world = ReadIds(Arg(argc, argv, "--world")), persona = ReadIds(Arg(argc, argv, "--persona"));
	CHECK_MSG(world.size() >= 1000 && !persona.empty(), "6.5 %s: the world has %zu ids (needs 1,000), the persona %zu",
	          tag.c_str(), world.size(), persona.size());
	if (world.size() < 1000 || persona.empty()) return;
	const Geo& g = e.geo;
	const int64_t nested_len = static_cast<int64_t>(world.size() + persona.size());
	const uint32_t pages = static_cast<uint32_t>(4 * g.PrefixPages(nested_len) + 4 * g.R(kSteps + 16) + 16);
	Rig rig(e, pages);
	if (!rig.pool || !RequireGpuGeometry(e, rig.pool, ("6.5 " + tag).c_str())) return;
	// 4.1's page edges at the shipped B (0, B-1, B, B+1) and 1,000. cap-1 and cap (32,768 on both artifacts) are left
	// to pkv_def: two cap-length batched prefills per card do not fit a 15-minute step (code review N1).
	for (int64_t L : {int64_t{0}, int64_t{1}, g.B - 1, g.B, g.B + 1, int64_t{1000}})
		RootShape(e, rig, std::vector<int32_t>(world.begin(), world.begin() + L), -1,
		          "6.5 " + tag + " root L=" + std::to_string(L));
	Nested(e, rig, world, persona, "6.5 " + tag + " nested");
}

}  // namespace

int main(int argc, char** argv) {
	ParseFixtureArgs(argc, argv);
	{
		const std::string model = Arg(argc, argv, "--model");
		Env e;
		if (model.empty()) {
			if (e.Open(FixturePath(argc, argv))) PkvDef(e);
		} else if (e.Open(model, false)) {
			RealModel(e, argc, argv);
		}
	}
	return FinishSuite("pcg_cell_6_5");
}

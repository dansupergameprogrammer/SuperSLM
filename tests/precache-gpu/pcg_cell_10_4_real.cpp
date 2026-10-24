// Precached-prefixes plan §7 10.4 [F3], its first clause: "10.1-10.2 on both GPUs". The GPU twins of 10.1 and 10.2 on
// the real 0.5B artifact (10.1's artifact), each in two processes, so the restoring process builds nothing.
//
//   10.1 (GPU)  ten prefixes of 128-2,000 tokens (tests/t3035-precache-refs/prompts/flat00.ids..flat09.ids), built by
//               prefill and saved to files in the capture process, with the token streams of five adopters of each
//               (64 greedy tokens) recorded beside them. The process exits. A new process restores all ten from the
//               files; its fifty adopters decode 64 tokens each, and every stream equals the capture's stream of the
//               same adopter over the prefill-built prefix; and no prefix prefill ran in the restoring process
//               (GpuPrefixPrefillCallsForTest reads 0).
//   10.2 (GPU)  the PKV cohort's nested shape: a world (world.ids, 1,012 ids) and ten personas (persona00..09.ids)
//               built by begin_from + prefill and saved; in the restoring process the world is restored and each
//               persona restored against it as its hint: each shares s = floor(|world| / B) pages, the world's own
//               (its first s table pages are the world's), and draws ceil((|world| + |persona|) / B) - s, read by the
//               one-state fill probe around the restore; s and the draw are computed from the files read, never
//               from literals; every persona's adopter stream equals the capture's.
//
// The reference streams: the CPU's 10.1 grades against a population pinned at an anchor (§8). The GPU has none, so the
// capture process records the streams of adopters over the prefill-built prefixes, before any restore exists in either
// process; the restore path cannot influence them (it never runs in the capture process). The refs file carries the
// adapter, the model's SHA-256 and each prompt's SHA-256, and the restoring process refuses a refs file from another
// card, model or prompt. Five adopters per prefix (10.1's fifty): the GPU has one adopter mode (a pooled budgeted
// sequence) and no damped-greedy decode, so the five differ by their budgets (64, 80, 96, 128, 256 tokens), which
// differ in the pages each reserves; the record names this as the GPU reading of 10.1's "both adopter modes" and
// "greedy and damped-greedy".
//
// Run (0.5B; both phases, two processes):
//   pcg_cell_10_4_real.exe --cell=10.1|10.2 --phase=capture|restore --model=FILE --prompts=DIR --blobs=DIR
// Configuration: alloc (GpuPrefixPrefillCallsForTest is a test-build seam).
#include "pcg_f3_common.h"

#include <cstdio>
#include <map>

#if !defined(SUPERSLM_GPU_ALLOC_FAULT_INJECTION)
#error "pcg_cell_10_4_real needs GpuPrefixPrefillCallsForTest: build it in the alloc configuration"
#endif

using namespace pcg;
using namespace pcg::f3;

namespace {

constexpr int64_t kTokens = 64;
constexpr int32_t kBudgets[5] = {64, 80, 96, 128, 256};

struct Refs {
	std::map<std::string, std::string> header;
	std::map<std::string, std::vector<int32_t>> streams;  // "<prefix>#<adopter>" -> tokens
	bool ok = false;
};

std::map<std::string, std::string> Fields(const std::string& line) {
	std::map<std::string, std::string> f;
	std::istringstream in(line);
	std::string w;
	while (in >> w) {
		const size_t eq = w.find('=');
		if (eq != std::string::npos) f[w.substr(0, eq)] = w.substr(eq + 1);
	}
	return f;
}

Refs ReadRefs(const std::string& path) {
	Refs r;
	std::ifstream in(path);
	std::string line;
	if (!std::getline(in, line) || line.rfind("# ", 0) != 0) return r;
	r.header = Fields(line.substr(2));
	while (std::getline(in, line)) {
		if (!line.empty() && line.back() == '\r') line.pop_back();
		if (line.empty()) continue;
		const auto f = Fields(line);
		if (f.count("id") && f.count("tokens")) r.streams[f.at("id")] = Split(f.at("tokens"));
	}
	r.ok = true;
	return r;
}

std::string Header(const Env& e, const std::string& cell, const std::string& prompts_sha) {
	return "# cell=" + cell + " adapter=" + AdapterTag() + " model_sha256=" + Sha(e.fx.bytes) +
	       " prompts_sha256=" + prompts_sha + " tokens=" + std::to_string(kTokens) + "\n";
}
bool RefsMatch(const Env& e, const Refs& r, const std::string& cell, const std::string& prompts_sha, const char* what) {
	const auto h = [&](const char* k) { return r.header.count(k) ? r.header.at(k) : std::string(); };
	const bool ok = r.ok && h("cell") == cell && h("adapter") == AdapterTag() && h("model_sha256") == Sha(e.fx.bytes) &&
	                h("prompts_sha256") == prompts_sha;
	CHECK_MSG(ok, "%s: the capture's refs are absent or from another card, model or prompt set (cell %s, adapter %s, "
	          "model %s, prompts %s; here %s, %s)", what, h("cell").c_str(), h("adapter").c_str(),
	          h("model_sha256").substr(0, 12).c_str(), h("prompts_sha256").substr(0, 12).c_str(), AdapterTag().c_str(),
	          Sha(e.fx.bytes).substr(0, 12).c_str());
	return ok;
}

// The SHA-256 over the named prompt files' SHA-256s, in order: what the refs bind.
std::string PromptsSha(const std::string& dir, const std::vector<std::string>& names) {
	std::string all;
	for (const auto& n : names) all += ShaFile(dir + "/" + n + ".ids") + " " + n + "\n";
	return Sha(std::vector<uint8_t>(all.begin(), all.end()));
}

std::string Name2(const char* stem, int i) {
	char n[32];
	std::snprintf(n, sizeof n, "%s%02d", stem, i);
	return n;
}

// ---- 10.1 ---------------------------------------------------------------------------------------------------------------

void Cell101(const Env& e, const std::string& phase, const std::string& prompts, const std::string& blobs) {
	std::vector<std::string> names;
	for (int i = 0; i < 10; ++i) names.push_back(Name2("flat", i));
	std::vector<std::vector<int32_t>> ids;
	int64_t largest = 0;
	for (const auto& n : names) {
		ids.push_back(ReadIds(prompts + "/" + n + ".ids"));
		largest = std::max<int64_t>(largest, static_cast<int64_t>(ids.back().size()));
	}
	for (const auto& t : ids)
		if (t.size() < 128) {
			CHECK_MSG(false, "10.1: a prompt has %zu ids; 10.1's prefixes are 128-2,000", t.size());
			return;
		}
	const std::string psha = PromptsSha(prompts, names);
	// One prefix live at a time, plus one adopter of the largest budget.
	const uint32_t pages = static_cast<uint32_t>(e.geo.PrefixPages(largest) + e.geo.R(256) + 8);
	Rig rig(e, pages);
	if (!rig.pool || !RequireGpuGeometry(e, rig.pool, "10.1 GPU")) return;
	const std::string refs_path = blobs + "/10.1.refs";
	if (phase == "capture") {
		std::string refs = Header(e, "10.1", psha);
		for (size_t i = 0; i < names.size(); ++i) {
			SslmGpuPrefixHandle* p = BatchedPrefix(rig, ids[i]);
			if (!p) return;
			const std::vector<uint8_t> b = SavePrefix(e, p);
			CHECK_MSG(!b.empty() && WriteBytes(blobs + "/" + names[i] + ".spx1", b), "10.1 capture: %s not saved", names[i].c_str());
			for (int a = 0; a < 5; ++a) {
				const std::vector<int32_t> t = AdopterOutcomes(rig, p, static_cast<int64_t>(ids[i].size()), kTokens, kBudgets[a]);
				CHECK_MSG(AllTokens(t) && t.size() == static_cast<size_t>(kTokens), "10.1 capture %s adopter %d: %zu tokens",
				          names[i].c_str(), a, t.size());
				refs += "id=" + names[i] + "#" + std::to_string(a) + " budget=" + std::to_string(kBudgets[a]) +
				        " tokens=" + Join(t) + "\n";
			}
			std::printf("10.1 capture %s: %zu tokens, blob %zu bytes, sha256 %s\n", names[i].c_str(), ids[i].size(), b.size(),
			            Sha(b).c_str());
			Release(rig, p);
		}
		CHECK_MSG(WriteText(refs_path, refs), "10.1 capture: %s not written", refs_path.c_str());
		std::printf("10.1 capture: %zu prefixes and %zu adopter streams written to %s\n", names.size(), names.size() * 5,
		            blobs.c_str());
		return;
	}
	const Refs refs = ReadRefs(refs_path);
	if (!RefsMatch(e, refs, "10.1", psha, "10.1 restore")) return;
	const uint64_t calls0 = GpuPrefixPrefillCallsForTest();
	int graded = 0;
	for (size_t i = 0; i < names.size(); ++i) {
		std::vector<uint8_t> b;
		CHECK_MSG(ReadBytes(blobs + "/" + names[i] + ".spx1", &b) && !b.empty(), "10.1 restore: %s.spx1 is missing",
		          names[i].c_str());
		if (b.empty()) continue;
		const RestoredPrefix r = RestorePrefix(rig, b);
		CHECK_MSG(r.status == GS::SSLM_OK && r.p, "10.1 restore %s: %s", names[i].c_str(), Name(r.status));
		if (!r.p) continue;
		for (int a = 0; a < 5; ++a) {
			const std::string id = names[i] + "#" + std::to_string(a);
			const auto want = refs.streams.find(id);
			const std::vector<int32_t> got = AdopterOutcomes(rig, r.p, static_cast<int64_t>(ids[i].size()), kTokens, kBudgets[a]);
			const bool same = want != refs.streams.end() && got == want->second && got.size() == static_cast<size_t>(kTokens);
			CHECK_MSG(same, "10.1 %s: the restored prefix's adopter differs from the capture's prefill-built stream", id.c_str());
			if (same) ++graded;
		}
		Release(rig, r.p);
	}
	const uint64_t calls = GpuPrefixPrefillCallsForTest() - calls0;
	CHECK_MSG(graded == 50, "10.1: %d of 50 adopter streams equal the capture's", graded);
	CHECK_MSG(calls == 0, "10.1: %llu prefix prefill calls ran in the restoring process", static_cast<unsigned long long>(calls));
	std::printf("10.1 restore on %s: %d of 50 adopter streams equal the prefill-built ones; prefix prefill calls in this "
	            "process: %llu\n", AdapterTag().c_str(), graded, static_cast<unsigned long long>(GpuPrefixPrefillCallsForTest()));
}

// ---- 10.2 ---------------------------------------------------------------------------------------------------------------

struct PoolOf : ProbeState {
	Rig& r;
	explicit PoolOf(Rig& rr) : r(rr) {}
	SslmGpuKvPoolHandle* Pool() override { return r.pool; }
	SslmGpuContext* Ctx() override { return r.ctx; }
	SslmGpuModelHandle* Model() override { return r.model; }
	void DrainAll() override {}
};

void Cell102(const Env& e, const std::string& phase, const std::string& prompts, const std::string& blobs) {
	std::vector<std::string> names = {"world"};
	for (int i = 0; i < 10; ++i) names.push_back(Name2("persona", i));
	const std::vector<int32_t> world = ReadIds(prompts + "/world.ids");
	std::vector<std::vector<int32_t>> personas;
	for (int i = 0; i < 10; ++i) personas.push_back(ReadIds(prompts + "/" + names[static_cast<size_t>(i) + 1] + ".ids"));
	if (world.empty()) return;
	const std::string psha = PromptsSha(prompts, names);
	const Geo& g = e.geo;
	const int64_t W = static_cast<int64_t>(world.size());
	const int64_t s = W / g.B;  // floor(|world| / B), from the file read
	int64_t draws = 0;
	for (const auto& p : personas) draws += g.PrefixPages(W + static_cast<int64_t>(p.size())) - s;
	// The world, all ten children live (as the CPU's 10.2 keeps them), one adopter, and 8 pages the probe reads free.
	const int64_t slack = 8;
	const uint32_t pages = static_cast<uint32_t>(g.PrefixPages(W) + draws + g.R(kTokens + 16) + slack);
	Rig rig(e, pages);
	if (!rig.pool || !RequireGpuGeometry(e, rig.pool, "10.2 GPU")) return;
	const std::string refs_path = blobs + "/10.2.refs";
	if (phase == "capture") {
		SslmGpuPrefixHandle* w = BatchedPrefix(rig, world);
		if (!w) return;
		std::string refs = Header(e, "10.2", psha);
		const std::vector<uint8_t> wb = SavePrefix(e, w);
		CHECK_MSG(!wb.empty() && WriteBytes(blobs + "/world.spx1", wb), "10.2 capture: world not saved");
		for (int i = 0; i < 10; ++i) {
			SslmGpuPrefixHandle* c = BatchedChild(rig, w, personas[static_cast<size_t>(i)]);
			if (!c) return;
			const std::vector<uint8_t> cb = SavePrefix(e, c);
			const std::string& n = names[static_cast<size_t>(i) + 1];
			CHECK_MSG(!cb.empty() && WriteBytes(blobs + "/" + n + ".spx1", cb), "10.2 capture: %s not saved", n.c_str());
			const int64_t L = W + static_cast<int64_t>(personas[static_cast<size_t>(i)].size());
			const std::vector<int32_t> t = AdopterOutcomes(rig, c, L, kTokens, static_cast<int32_t>(kTokens + 16));
			CHECK_MSG(AllTokens(t) && t.size() == static_cast<size_t>(kTokens), "10.2 capture %s: %zu tokens", n.c_str(), t.size());
			refs += "id=" + n + " tokens=" + Join(t) + "\n";
			Release(rig, c);
		}
		CHECK_MSG(WriteText(refs_path, refs), "10.2 capture: %s not written", refs_path.c_str());
		std::printf("10.2 capture: world of %lld and ten personas written to %s\n", static_cast<long long>(W), blobs.c_str());
		return;
	}
	const Refs refs = ReadRefs(refs_path);
	if (!RefsMatch(e, refs, "10.2", psha, "10.2 restore")) return;
	const uint64_t calls0 = GpuPrefixPrefillCallsForTest();
	std::vector<uint8_t> wb;
	CHECK_MSG(ReadBytes(blobs + "/world.spx1", &wb) && !wb.empty(), "10.2 restore: world.spx1 is missing");
	if (wb.empty()) return;
	const RestoredPrefix w = RestorePrefix(rig, wb);
	CHECK_MSG(w.status == GS::SSLM_OK && w.p, "10.2 restore world: %s", Name(w.status));
	if (!w.p) return;
	const std::vector<uint32_t> wp = TablePages(w.p);
	int64_t free_now = static_cast<int64_t>(pages) - g.PrefixPages(W);
	PoolOf probe(rig);
	int graded = 0;
	for (int i = 0; i < 10; ++i) {
		const std::string& n = names[static_cast<size_t>(i) + 1];
		const int64_t L = W + static_cast<int64_t>(personas[static_cast<size_t>(i)].size());
		const int64_t draw = g.PrefixPages(L) - s;
		std::vector<uint8_t> cb;
		CHECK_MSG(ReadBytes(blobs + "/" + n + ".spx1", &cb) && !cb.empty(), "10.2 restore: %s.spx1 is missing", n.c_str());
		if (cb.empty()) continue;
		CHECK_MSG(free_now >= 3, "10.2: %lld pages free before %s; the probe needs 3", static_cast<long long>(free_now), n.c_str());
		const bool before = ProbeExactlyFreeOneState(e, probe, free_now, (std::string("10.2 before ") + n).c_str());
		const RestoredPrefix c = RestorePrefix(rig, cb, w.p);
		CHECK_MSG(c.status == GS::SSLM_OK && c.p, "10.2 restore %s against the world: %s", n.c_str(), Name(c.status));
		if (!c.p) continue;
		const bool after = free_now - draw >= 3 &&
		                   ProbeExactlyFreeOneState(e, probe, free_now - draw, (std::string("10.2 after ") + n).c_str());
		const std::vector<uint32_t> cp = TablePages(c.p);
		bool leading = static_cast<int64_t>(cp.size()) == g.PrefixPages(L) && static_cast<int64_t>(wp.size()) >= s;
		for (int64_t k = 0; leading && k < s; ++k) leading = cp[static_cast<size_t>(k)] == wp[static_cast<size_t>(k)];
		CHECK_MSG(leading, "10.2 %s: its first %lld table pages are not the restored world's", n.c_str(), static_cast<long long>(s));
		CHECK_MSG(c.shared == static_cast<uint32_t>(s), "10.2 %s: *out_shared_pages %u, want floor(%lld/%lld) = %lld", n.c_str(),
		          c.shared, static_cast<long long>(W), static_cast<long long>(g.B), static_cast<long long>(s));
		const auto want = refs.streams.find(n);
		const std::vector<int32_t> got = AdopterOutcomes(rig, c.p, L, kTokens, static_cast<int32_t>(kTokens + 16));
		const bool same = want != refs.streams.end() && got == want->second && got.size() == static_cast<size_t>(kTokens);
		CHECK_MSG(same, "10.2 %s: the adopter differs from the capture's", n.c_str());
		std::printf("10.2 %s: shares %lld (the world's pages), draws %lld (probe: %s), adopter %s\n", n.c_str(),
		            static_cast<long long>(s), static_cast<long long>(draw), before && after ? "exact" : "MISMATCH",
		            same ? "equal" : "DIFFERS");
		if (same && leading && before && after && c.shared == static_cast<uint32_t>(s)) ++graded;
		free_now -= draw;  // the child stays live, as the CPU's 10.2 keeps them
	}
	const uint64_t calls = GpuPrefixPrefillCallsForTest() - calls0;
	CHECK_MSG(graded == 10, "10.2: %d of 10 personas graded", graded);
	CHECK_MSG(calls == 0, "10.2: %llu prefix prefill calls ran in the restoring process", static_cast<unsigned long long>(calls));
	std::printf("10.2 restore on %s: %d of 10 personas share s = %lld and match; prefix prefill calls in this process: %llu\n",
	            AdapterTag().c_str(), graded, static_cast<long long>(s), static_cast<unsigned long long>(GpuPrefixPrefillCallsForTest()));
}

}  // namespace

int main(int argc, char** argv) {
	ParseFixtureArgs(argc, argv);
	{
		const std::string cell = Arg(argc, argv, "--cell"), phase = Arg(argc, argv, "--phase"), model = Arg(argc, argv, "--model");
		const std::string prompts = Arg(argc, argv, "--prompts"), blobs = Arg(argc, argv, "--blobs");
		const bool args = (cell == "10.1" || cell == "10.2") && (phase == "capture" || phase == "restore") && !model.empty() &&
		                  !prompts.empty() && !blobs.empty();
		CHECK_MSG(args, "pass --cell=10.1|10.2 --phase=capture|restore --model=FILE --prompts=DIR --blobs=DIR");
		Env e;
		if (args && e.Open(model, false)) {
			if (cell == "10.1") Cell101(e, phase, prompts, blobs);
			else Cell102(e, phase, prompts, blobs);
		}
	}
	return FinishSuite("pcg_cell_10_4_real");
}

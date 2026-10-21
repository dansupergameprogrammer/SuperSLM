// Precached-prefixes plan, cell 9.10 [E5; box] (rev 1.13, PC-13; rev 1.18: captured on v1.13.1, d697927): the
// GPU twin of 9.9. 'SLM5' (dedicated) and 'SLM6' (pooled, including a save between an adopt and the first
// finish) captured on a v1.13.1 build on each certified GPU, restored on the release build on the same device,
// re-saved byte-identical, and continued to the pinned tokens. 1.14.0's tree carries §4's GPU code (§3.8), so
// this is the proof that SLM5 and SLM6 are unchanged, not a regression check of an unchanged file set.
//
// ONE SOURCE, TWO BUILDS, as paged-KV 9.8's two legs (g1e_cell_9_8.cpp / g1e_old_reader_9_8.cpp):
//   - PCG_910_CAPTURE defined: the capture leg, built against the v1.13.1 tree (its include/, src/gpu and its
//     own tests/paged-kv-gpu/g1e_helpers.h; tests/paged-kv-gpu/build_paged_kv_gpu.bat :capture910). It writes
//     --out=DIR/v1.13.1_<vendor>_<device>/: captures.txt (a header `writer=v1.13.1 commit=<sha> vendor=<hex>
//     device=<hex> fixture=<sha256>`, then per case `<case> magic=<4cc> size=<n> blob=<sha256> last=<t>
//     ready=<0|1> mid=<0|1> tokens=<64 tokens>`) and <case>.bin, each blob as saved. The test-side token state
//     (last, ready, mid) is recorded because the GPU decode bridge needs it and the blob carries it device-side.
//   - Otherwise: the cell, built against the tree under work (the F2 table). It reads
//     --captures=DIR (or SUPERSLM_PCG_910_CAPTURES) /v1.13.1_<vendor>_<device>/ for the adapter it runs on.
//
// RED NOW, for the reason it names: no capture is pinned for any adapter (plan §9 E5: "Capture and pin 9.9's
// and 9.10's sequence saves on a v1.13.1 build (d697927)", on each certified GPU). On the cloud's software GPU
// there is no certified-GPU capture at all; this cell needs box inputs.
//
// Run: pcg_cell_9_10.exe --fixture-dir=DIR --captures=DIR
//      pcg_cell_9_10_capture.exe --fixture-dir=DIR --out=DIR   (the v1.13.1 build)
#include "g1e_helpers.h"
#include "superslm/sha256.h"

#include <dxgi1_6.h>
#include <wrl/client.h>

#include <cstdio>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

using namespace g1e;

namespace {

constexpr const char* kTag = "v1.13.1";
constexpr const char* kCommit = "d697927f6d5ce5fa73ecbb8660020a425518b923";
constexpr int64_t kDedicatedCap = 256;
constexpr int32_t kBudget = 512;
constexpr int32_t kPrefixLen = 300;
constexpr int32_t kContinue = 64;
constexpr uint32_t kPoolPages = 200;

enum class Kind { kDedicated, kPooled, kPooledAdopted };
struct Case {
	const char* name;
	Kind kind;
	bool mid;
	const char* magic;
};
const Case kCases[] = {
    {"slm5_dedicated_resting", Kind::kDedicated, false, "SLM5"},
    {"slm5_dedicated_mid_token", Kind::kDedicated, true, "SLM5"},
    {"slm6_pooled_resting", Kind::kPooled, false, "SLM6"},
    {"slm6_pooled_mid_token", Kind::kPooled, true, "SLM6"},
    // Plan 9.10: "a save between an adopt and the first finish": adopted, resting ready, nothing decoded yet.
    {"slm6_pooled_adopted_before_first_finish", Kind::kPooledAdopted, false, "SLM6"},
};

std::string Hex(const uint8_t* d, size_t n) {
	static const char* k = "0123456789abcdef";
	std::string s;
	for (size_t i = 0; i < n; ++i) {
		s.push_back(k[d[i] >> 4]);
		s.push_back(k[d[i] & 15]);
	}
	return s;
}
std::string Sha(const std::vector<uint8_t>& b) {
	uint8_t out[32];
	superslm::Sha256Hash(b.data(), b.size(), out);
	return Hex(out, 32);
}
std::string Join(const std::vector<int32_t>& t) {
	std::string o;
	for (size_t i = 0; i < t.size(); ++i) o += (i ? "," : "") + std::to_string(t[i]);
	return o;
}
bool ReadFile(const std::string& p, std::vector<uint8_t>* out) {
	std::ifstream f(p, std::ios::binary);
	if (!f) return false;
	out->assign(std::istreambuf_iterator<char>(f), {});
	return true;
}
std::string Arg(int argc, char** argv, const char* key) {
	const std::string k = key;
	for (int i = 1; i < argc; ++i) {
		const std::string a = argv[i];
		if (a.rfind(k, 0) == 0) return a.substr(k.size());
	}
	return {};
}

// The adapter the submission device uses, selected as the harness selects it (pcg_helpers.h's AdapterIds,
// restated: that header is the tree under work's and does not exist at v1.13.1, which the capture builds on).
bool AdapterIds910(uint32_t* vendor, uint32_t* device) {
	Microsoft::WRL::ComPtr<IDXGIFactory6> f;
	if (FAILED(CreateDXGIFactory1(IID_PPV_ARGS(&f)))) return false;
	const char* env = std::getenv("SSLM_GPU_ADAPTER_INDEX");
	Microsoft::WRL::ComPtr<IDXGIAdapter1> a;
	if (env && *env) {
		if (FAILED(f->EnumAdapters1(static_cast<UINT>(std::atoi(env)), &a))) return false;
	} else {
		for (UINT i = 0;; ++i) {
			Microsoft::WRL::ComPtr<IDXGIAdapter1> c;
			if (f->EnumAdapterByGpuPreference(i, DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE, IID_PPV_ARGS(&c)) ==
			    DXGI_ERROR_NOT_FOUND)
				break;
			DXGI_ADAPTER_DESC1 d{};
			c->GetDesc1(&d);
			if (d.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) continue;
			a = c;
			break;
		}
	}
	if (!a) return false;
	DXGI_ADAPTER_DESC1 d{};
	if (FAILED(a->GetDesc1(&d))) return false;
	*vendor = d.VendorId;
	*device = d.DeviceId;
	return true;
}
std::string AdapterDir() {
	uint32_t v = 0, d = 0;
	if (!AdapterIds910(&v, &d)) return {};
	char b[64];
	std::snprintf(b, sizeof b, "%s_%04x_%04x", kTag, v, d);
	return b;
}

// The state at the save point, built the same way on both builds. `px` receives the adopted prefix.
struct Built {
	GSeq seq;
	SslmGpuPrefixHandle* px = nullptr;
	bool dedicated = false;
};
bool Build(const Env& e, Rig& rig, const Case& c, Built* b) {
	if (c.kind == Kind::kDedicated) {
		SslmGpuSequenceHandle* h = nullptr;
		const GS st = sslm_gpu_seq_create(e.ctx, e.model(), kDedicatedCap, &h);
		CHECK_MSG(st == GS::SSLM_OK && h, "%s: sslm_gpu_seq_create: %s", c.name, StatusName(st));
		if (!h) return false;
		b->seq.h = h;
		b->dedicated = true;
	} else {
		b->seq = rig.Budgeted(kBudget);
		if (!b->seq.h) return false;
	}
	if (c.kind == Kind::kPooledAdopted) {
		b->px = BudgetPrefix(rig, PrefixTokens(kPrefixLen));
		if (!b->px) return false;
		const GS st = Adopt(rig, b->seq, b->px, kPrefixLen);
		CHECK_MSG(st == GS::SSLM_OK, "%s: adopt: %s", c.name, StatusName(st));
		return st == GS::SSLM_OK;
	}
	GS st = Prefill(e, b->seq, Stream(31, 100, kVocab));
	CHECK_MSG(st == GS::SSLM_OK, "%s: prefill: %s", c.name, StatusName(st));
	if (st != GS::SSLM_OK) return false;
	const std::vector<int32_t> four = DecodeN(e, b->seq, 4, &st);
	CHECK_MSG(four.size() == 4, "%s: 4 tokens before the save: %zu (%s)", c.name, four.size(), StatusName(st));
	if (four.size() != 4) return false;
	return !c.mid || EnterMidToken(e, b->seq);
}

#if defined(PCG_910_CAPTURE)

int Capture(int argc, char** argv) {
	const std::string out = Arg(argc, argv, "--out=");
	CHECK_MSG(!out.empty(), "usage: pcg_cell_9_10_capture --fixture-dir=DIR --out=DIR");
	const std::string sub = AdapterDir();
	CHECK_MSG(!sub.empty(), "no hardware adapter (DXGI)");
	if (out.empty() || sub.empty()) return 1;
	const std::string dir = out + "/" + sub;
	CreateDirectoryA(out.c_str(), nullptr);
	CreateDirectoryA(dir.c_str(), nullptr);
	Env e;
	const std::string fixture = FixturePath(argc, argv);
	if (!e.Open(fixture)) return 1;
	std::vector<uint8_t> fx_bytes;
	CHECK(ReadFile(fixture, &fx_bytes));
	std::ofstream rec(dir + "/captures.txt", std::ios::binary);
	rec << "writer=" << kTag << " commit=" << kCommit << " adapter=" << sub << " fixture=" << Sha(fx_bytes) << "\n";
	for (const Case& c : kCases) {
		Rig rig(e, kPoolPages);
		if (!rig.pool) continue;
		Built b;
		if (!Build(e, rig, c, &b)) continue;
		const std::vector<uint8_t> blob = Save(e, b.seq.h);
		CHECK_MSG(IsMagic(blob, c.magic), "%s: 1.13.1 saved another magic than %s", c.name, c.magic);
		const GSeq at = b.seq;
		GS st = GS::SSLM_OK;
		const std::vector<int32_t> next = DecodeN(e, b.seq, kContinue, &st);
		CHECK_MSG(next.size() == size_t(kContinue), "%s: 1.13.1's continuation: %zu tokens (%s)", c.name, next.size(),
		          StatusName(st));
		if (b.dedicated) sslm_gpu_seq_release(e.ctx, b.seq.h);
		if (!IsMagic(blob, c.magic) || next.size() != size_t(kContinue)) continue;
		std::ofstream bin(dir + "/" + c.name + ".bin", std::ios::binary);
		bin.write(reinterpret_cast<const char*>(blob.data()), static_cast<std::streamsize>(blob.size()));
		rec << c.name << " magic=" << c.magic << " size=" << blob.size() << " blob=" << Sha(blob) << " last=" << at.last
		    << " ready=" << (at.ready ? 1 : 0) << " mid=" << (at.mid ? 1 : 0) << " tokens=" << Join(next) << "\n";
	}
	return 0;
}

#else

struct Record {
	std::string magic, sha;
	size_t size = 0;
	int32_t last = -1;
	bool ready = false, mid = false;
	std::vector<int32_t> tokens;
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

void Cell(int argc, char** argv, const Env& e) {
	std::string root = Arg(argc, argv, "--captures=");
	if (root.empty() && std::getenv("SUPERSLM_PCG_910_CAPTURES")) root = std::getenv("SUPERSLM_PCG_910_CAPTURES");
	const std::string sub = AdapterDir();
	CHECK_MSG(!sub.empty(), "9.10: no hardware adapter (DXGI): the cell grades a certified GPU's v1.13.1 capture");
	const std::string dir = root + "/" + sub;
	std::ifstream in(dir + "/captures.txt", std::ios::binary);
	CHECK_MSG(!root.empty() && in, "9.10: no v1.13.1 capture is pinned for adapter %s (looked in '%s'; plan §9 E5 "
	          "captures on a v1.13.1 build, d697927, on each certified GPU)", sub.c_str(), dir.c_str());
	if (root.empty() || !in) return;
	std::string line;
	std::getline(in, line);
	const auto head = Fields(line);
	CHECK_MSG(head.count("writer") && head.at("writer") == kTag && head.count("commit") && head.at("commit") == kCommit,
	          "9.10: the capture was not written by %s at %s: %s", kTag, kCommit, line.c_str());
	std::vector<uint8_t> fx_bytes;
	CHECK(ReadFile(FixturePath(argc, argv), &fx_bytes));
	CHECK_MSG(head.count("fixture") && head.at("fixture") == Sha(fx_bytes), "9.10: the capture's pkv_def is not this one");
	std::map<std::string, Record> recs;
	while (std::getline(in, line)) {
		if (!line.empty() && line.back() == '\r') line.pop_back();
		if (line.empty()) continue;
		const std::string name = line.substr(0, line.find(' '));
		const auto f = Fields(line);
		Record r;
		r.magic = f.count("magic") ? f.at("magic") : "";
		r.sha = f.count("blob") ? f.at("blob") : "";
		r.size = f.count("size") ? size_t(std::strtoull(f.at("size").c_str(), nullptr, 10)) : 0;
		r.last = f.count("last") ? std::atoi(f.at("last").c_str()) : -1;
		r.ready = f.count("ready") && f.at("ready") == "1";
		r.mid = f.count("mid") && f.at("mid") == "1";
		if (f.count("tokens")) {
			std::istringstream t(f.at("tokens"));
			std::string tok;
			while (std::getline(t, tok, ','))
				if (!tok.empty()) r.tokens.push_back(std::atoi(tok.c_str()));
		}
		recs[name] = r;
	}
	for (const Case& c : kCases) {
		const auto it = recs.find(c.name);
		CHECK_MSG(it != recs.end(), "9.10: the capture has no record for %s", c.name);
		if (it == recs.end()) continue;
		const Record& r = it->second;
		std::vector<uint8_t> pinned;
		const bool read = ReadFile(dir + "/" + c.name + ".bin", &pinned);
		CHECK_MSG(read && Sha(pinned) == r.sha && pinned.size() == r.size && IsMagic(pinned, c.magic) && r.magic == c.magic,
		          "9.10 %s: the pinned blob is missing or does not match its record", c.name);
		CHECK_MSG(r.tokens.size() == size_t(kContinue), "9.10 %s: the pinned continuation is %zu tokens", c.name,
		          r.tokens.size());
		if (!read || Sha(pinned) != r.sha || !IsMagic(pinned, c.magic)) continue;
		GSeq like;
		like.last = r.last;
		like.ready = r.ready;
		like.mid = r.mid;
		like.primed = true;
		const auto grade = [&](GSeq s, const char* path) {
			CHECK_MSG(Save(e, s.h) == pinned, "9.10 %s via %s: the re-save differs from the v1.13.1 pin", c.name, path);
			GS st = GS::SSLM_OK;
			const std::vector<int32_t> next = DecodeN(e, s, kContinue, &st);
			CHECK_MSG(next == r.tokens, "9.10 %s via %s: the continuation differs from v1.13.1's (%zu tokens, %s)", c.name,
			          path, next.size(), StatusName(st));
		};
		if (c.kind == Kind::kDedicated) {
			SslmGpuSequenceHandle* h = OutSentinel<SslmGpuSequenceHandle>();
			const GS st = sslm_gpu_seq_restore(e.ctx, e.model(), pinned.data(), pinned.size(), &h);
			CHECK_MSG(st == GS::SSLM_OK && Real(h), "9.10 %s: sslm_gpu_seq_restore: %s", c.name, StatusName(st));
			if (!Real(h)) continue;
			GSeq s = like;
			s.h = h;
			grade(s, "sslm_gpu_seq_restore");
			sslm_gpu_seq_release(e.ctx, h);
			continue;
		}
		for (int with_prefix = 0; with_prefix <= (c.kind == Kind::kPooledAdopted ? 1 : 0); ++with_prefix) {
			Rig rig(e, kPoolPages);
			if (!rig.pool) continue;
			SslmGpuPrefixHandle* px = with_prefix ? BudgetPrefix(rig, PrefixTokens(kPrefixLen)) : nullptr;
			if (with_prefix && !px) continue;
			const Restored rs = RestoreIntoPool(rig, pinned, px, like);
			CHECK_MSG(rs.status == GS::SSLM_OK && rs.seq.h, "9.10 %s: restore_into_pool (%s prefix): %s", c.name,
			          with_prefix ? "matching" : "null", StatusName(rs.status));
			if (rs.seq.h) grade(rs.seq, with_prefix ? "restore_into_pool, matching prefix" : "restore_into_pool, null");
		}
	}
}

#endif

}  // namespace

int main(int argc, char** argv) {
	ParseFixtureArgs(argc, argv);
#if defined(PCG_910_CAPTURE)
	const int rc = Capture(argc, argv);
	const int suite = FinishSuite("pcg_cell_9_10_capture");
	return rc ? rc : suite;
#else
	{
		Env e;
		if (e.Open(FixturePath(argc, argv))) Cell(argc, argv, e);
	}
	return FinishSuite("pcg_cell_9_10");
#endif
}

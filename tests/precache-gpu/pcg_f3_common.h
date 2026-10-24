// Precached-prefixes plan §9 F3 (and F4's GPU legs): what the F3 certification cells share, on top of pcg_helpers.h.
//
//   pcg_cell_4_6           4.6 [F3]  byte identity CPU <-> GPU and card <-> card on the real 0.5B and 1.5B at L = 1,000;
//                                    the cross-card restore through the adapter-check seam
//   pcg_cell_6_5           6.5 [F3]  6.1-6.4 on the GPU (pkv_def's shapes, and the real artifacts)
//   pcg_cell_10_4_real     10.4 [F3] 10.1 and 10.2 on the GPU, two processes, against the capture's own prefill twins
//   pcg_cell_10_4_timing   10.4 [F3] the GPU speed bar's instrument (plain configuration only)
//
// These cells run on the real artifacts as well as on pkv_def, so nothing here assumes pkv_def's geometry: the
// environment is opened with Env::Open(path, false) and every page count is computed from the test-side Geo
// (§3.1's rule as the GPU pool applies it: B = kv_block_size when it divides the cap, else the cap), which
// RequireGpuGeometry checks against the pool's own reading before anything is graded.
//
// Prefill: the real cells build their prefixes with the batched prompt prefill (one call admits the whole prompt;
// the dispatch budget is per-token dispatches x tokens), the path 10.4's bar times ("restore over batched GPU
// prefill"), and the same path for a prefix and its twin, so the two are built alike.
#ifndef SSLM_PCG_F3_COMMON_H
#define SSLM_PCG_F3_COMMON_H

#include "pcg_helpers.h"

#include <chrono>
#include <cstdio>
#include <fstream>
#include <iterator>
#include <sstream>
#include <string>
#include <vector>

#if defined(_WIN32)
#include <windows.h>  // GetModuleFileNameA (the D3D12 headers pcg_helpers.h includes bring it in already)
#endif

namespace pcg {
namespace f3 {

inline std::string Arg(int argc, char** argv, const char* name) {
	const std::string prefix = std::string(name) + "=";
	for (int i = 1; i < argc; ++i) {
		const std::string a = argv[i];
		if (a.rfind(prefix, 0) == 0) return a.substr(prefix.size());
	}
	return std::string();
}

inline bool ReadBytes(const std::string& path, std::vector<uint8_t>* out) {
	std::ifstream in(path, std::ios::binary);
	if (!in) return false;
	out->assign(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
	return true;
}
// Written beside the target and renamed, so a crash never leaves a partial file under the final name.
inline bool WriteBytes(const std::string& path, const std::vector<uint8_t>& b) {
	const std::string tmp = path + ".partial";
	{
		std::ofstream out(tmp, std::ios::binary | std::ios::trunc);
		out.write(reinterpret_cast<const char*>(b.data()), static_cast<std::streamsize>(b.size()));
		if (!out.good()) return false;
	}
	std::remove(path.c_str());
	return std::rename(tmp.c_str(), path.c_str()) == 0;
}
inline bool WriteText(const std::string& path, const std::string& text) {
	return WriteBytes(path, std::vector<uint8_t>(text.begin(), text.end()));
}

// A prompt file: whitespace-separated token ids (tests/t3035-precache-refs/prompts/*.ids).
inline std::vector<int32_t> ReadIds(const std::string& path) {
	std::ifstream in(path);
	std::vector<int32_t> ids;
	int32_t id = 0;
	while (in >> id) ids.push_back(id);
	CHECK_MSG(!ids.empty(), "prompt ids %s are missing or empty", path.c_str());
	return ids;
}

inline std::string Hex(const uint8_t* d, size_t n) {
	static const char* k = "0123456789abcdef";
	std::string s;
	for (size_t i = 0; i < n; ++i) {
		s.push_back(k[d[i] >> 4]);
		s.push_back(k[d[i] & 15]);
	}
	return s;
}
inline std::string Sha(const std::vector<uint8_t>& b) {
	uint8_t out[32];
	superslm::Sha256Hash(b.data(), b.size(), out);
	return Hex(out, 32);
}
inline std::string ShaFile(const std::string& path) {
	std::vector<uint8_t> b;
	return ReadBytes(path, &b) ? Sha(b) : std::string();
}
// The SHA-256 of the running program, so a reading is tied to the build that took it.
inline std::string RunnerSha() {
#if defined(_WIN32)
	char path[4096] = {};
	const DWORD n = GetModuleFileNameA(nullptr, path, sizeof path);
	return n > 0 && n < sizeof path ? ShaFile(std::string(path, n)) : std::string();
#else
	return std::string();
#endif
}

// "vvvv_dddd", the adapter the context runs on (the captures' and blobs' naming).
inline std::string AdapterTag() {
	uint32_t v = 0, d = 0;
	if (!AdapterIds(&v, &d)) return "unknown";
	char t[16];
	std::snprintf(t, sizeof t, "%04x_%04x", v, d);
	return t;
}

// The geometry precondition: the test-side B (Geo, §3.1) equals the pool's own page_positions.
inline bool RequireGpuGeometry(const Env& e, SslmGpuKvPoolHandle* pool, const char* what) {
	SslmGpuKvPoolStats st{};
	st.struct_size = sizeof st;
	const GS s = sslm_gpu_kv_pool_stats(pool, &st);
	const bool ok = s == GS::SSLM_OK && st.page_positions == e.geo.B && e.geo.B > 0;
	CHECK_MSG(ok, "%s: the pool's page_positions is %lld (%s); the test-side B is %lld", what,
	          static_cast<long long>(st.page_positions), Name(s), static_cast<long long>(e.geo.B));
	std::printf("%s: layers %u, kv heads %u, head_dim %u, cap %lld, B %lld, %zu bytes per token\n", what, e.geo.L,
	            e.geo.Hkv, e.geo.D, static_cast<long long>(e.geo.cap), static_cast<long long>(e.geo.B),
	            e.geo.BytesPerToken());
	return ok;
}

// The batched prompt prefill: as few calls as the budget admits, each admitting every remaining token.
inline GS PrefillBatched(const Env& e, SslmGpuPrefixHandle* p, const std::vector<int32_t>& t) {
	size_t at = 0;
	GS st = GS::SSLM_OK;
	while (at < t.size()) {
		const uint64_t want = uint64_t{e.full_budget} * (t.size() - at);
		const uint32_t budget = want > 0xFFFFFFFFull ? 0xFFFFFFFFu : static_cast<uint32_t>(want);
		int32_t consumed = 0;
		st = sslm_gpu_prefix_prefill(e.ctx, p, t.data() + at, static_cast<int32_t>(t.size() - at), 0, budget, &consumed);
		at += static_cast<size_t>(consumed > 0 ? consumed : 0);
		if (st != GS::SSLM_OK || consumed <= 0) break;
	}
	return st;
}

// A frozen root prefix of `tokens`, built by the batched prefill; schema index `schema` bound first when >= 0.
inline SslmGpuPrefixHandle* BatchedPrefix(Rig& rig, const std::vector<int32_t>& tokens, int32_t schema = -1) {
	SslmGpuPrefixHandle* p = nullptr;
	const int32_t budget = tokens.empty() ? 1 : static_cast<int32_t>(tokens.size());
	GS st = sslm_gpu_prefix_begin(rig.ctx, rig.model, rig.pool, budget, &p);
	if (st == GS::SSLM_OK && schema >= 0) st = sslm_gpu_prefix_set_schema(rig.ctx, p, schema);
	if (st == GS::SSLM_OK && !tokens.empty()) st = PrefillBatched(rig.e, p, tokens);
	if (st == GS::SSLM_OK) st = sslm_gpu_prefix_freeze(rig.ctx, p);
	CHECK_MSG(st == GS::SSLM_OK, "GPU prefix of %zu tokens (schema %d), batched prefill: %s", tokens.size(), schema,
	          Name(st));
	if (st != GS::SSLM_OK) {
		if (Real(p)) sslm_gpu_prefix_release(rig.ctx, p);
		return nullptr;
	}
	return rig.Prefix(p);
}

// A begin_from child of the frozen `parent`, `tokens` prefilled (batched) and frozen.
inline SslmGpuPrefixHandle* BatchedChild(Rig& rig, SslmGpuPrefixHandle* parent, const std::vector<int32_t>& tokens) {
	SslmGpuPrefixHandle* c = nullptr;
	GS st = sslm_gpu_prefix_begin_from(rig.ctx, parent, static_cast<int32_t>(tokens.size()), &c);
	if (st == GS::SSLM_OK) st = PrefillBatched(rig.e, c, tokens);
	if (st == GS::SSLM_OK) st = sslm_gpu_prefix_freeze(rig.ctx, c);
	CHECK_MSG(st == GS::SSLM_OK, "GPU begin_from child of %zu tokens: %s", tokens.size(), Name(st));
	if (st != GS::SSLM_OK) {
		if (Real(c)) sslm_gpu_prefix_release(rig.ctx, c);
		return nullptr;
	}
	return rig.Prefix(c);
}

// Exactly `n` greedy steps of a fresh budgeted adopter of `p` (released after): the token, -2 for a schema dead end,
// or -1000 - status for a refused step, so a prefix near the cap compares its refusals too.
inline std::vector<int32_t> AdopterOutcomes(Rig& rig, SslmGpuPrefixHandle* p, int64_t prefix_len, int64_t n,
                                            int32_t budget) {
	GSeq s = rig.Budgeted(budget);
	if (!s.h) return {};
	const GS st = Adopt(rig, s, p, prefix_len);
	CHECK_MSG(st == GS::SSLM_OK, "adopt: %s", Name(st));
	std::vector<int32_t> out = st == GS::SSLM_OK ? StepOutcomes(rig.e, s, n) : std::vector<int32_t>{};
	rig.ReleaseSeq(s.h);
	return out;
}
inline bool AllTokens(const std::vector<int32_t>& t) {
	for (int32_t x : t)
		if (x < 0) return false;
	return !t.empty();
}
inline std::string Join(const std::vector<int32_t>& t) {
	std::string s;
	for (size_t i = 0; i < t.size(); ++i) s += (i ? "," : "") + std::to_string(t[i]);
	return s;
}
inline std::vector<int32_t> Split(const std::string& s) {
	std::vector<int32_t> t;
	std::stringstream in(s);
	std::string w;
	while (std::getline(in, w, ','))
		if (!w.empty()) t.push_back(static_cast<int32_t>(std::atoi(w.c_str())));
	return t;
}
inline void Release(Rig& rig, SslmGpuPrefixHandle* p) {
	for (auto it = rig.prefixes.begin(); it != rig.prefixes.end(); ++it)
		if (*it == p) {
			rig.prefixes.erase(it);
			break;
		}
	if (p) sslm_gpu_prefix_release(rig.ctx, p);
}

inline double Ms(std::chrono::steady_clock::time_point a, std::chrono::steady_clock::time_point b) {
	return std::chrono::duration<double, std::milli>(b - a).count();
}

}  // namespace f3
}  // namespace pcg

#endif  // SSLM_PCG_F3_COMMON_H

// T-2900 (Curie), CPU-side helper for cell_gpu_cell3_agreement.cpp -- own translation unit, same
// technique as cell_gpu_cell1_realschema_cpu_side.cpp (avoids the fixture_common.h
// `using enum SslmGpuStatus` collision with sslm_abi.h's plain enum). ADOPTED VERBATIM from
// `Claude/Vitruvius/t2897-probe/cell3_cpu_side.cpp` (T-2895/T-2897's own Cell 3 CPU half).
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <vector>
#include "superslm/sslm_abi.h"

namespace {
uint32_t Le32(const uint8_t* p) { return p[0] | (p[1] << 8) | (p[2] << 16) | (static_cast<uint32_t>(p[3]) << 24); }
uint64_t Le64(const uint8_t* p) { return static_cast<uint64_t>(Le32(p)) | (static_cast<uint64_t>(Le32(p + 4)) << 32); }

struct Cpu {
	std::vector<uint8_t> bytes;
	sslm_model model = nullptr;
	std::vector<uint8_t> pool_store;
	sslm_kv_pool pool = nullptr;
	sslm_schema schema = nullptr;
	int32_t layers = 0;
} g;

sslm_status PrefillAll(sslm_seq s, const std::vector<int32_t>& t, sslm_span_kind kind) {
	int32_t off = 0;
	while (off < static_cast<int32_t>(t.size())) {
		int32_t consumed = 0;
		sslm_status st =
		    sslm_prefill(g.model, s, t.data() + off, static_cast<int32_t>(t.size()) - off, g.layers, kind, nullptr,
		                 &consumed);
		if (st != SSLM_OK) return st;
		if (consumed <= 0) return SSLM_INVALID_ARGUMENT;
		off += consumed;
	}
	return SSLM_OK;
}
}  // namespace

extern "C" bool CpuOpenForCell3(const char* path, int32_t layers) {
	FILE* f = std::fopen(path, "rb");
	if (!f) return false;
	std::fseek(f, 0, SEEK_END);
	long sz = std::ftell(f);
	std::fseek(f, 0, SEEK_SET);
	g.bytes.resize(sz);
	std::fread(g.bytes.data(), 1, sz, f);
	std::fclose(f);
	if (sslm_model_map(g.bytes.data(), g.bytes.size(), &g.model) != SSLM_OK) return false;
	g.layers = layers;
	const uint32_t blocks = 8;
	size_t bb = sslm_kv_block_size(g.model), ov = sslm_kv_pool_overhead_size(g.model, blocks);
	size_t need = bb * blocks + ov;
	g.pool_store.assign(need + 63, 0);
	void* ap = g.pool_store.data();
	size_t space = g.pool_store.size();
	std::align(64, need, ap, space);
	if (sslm_kv_pool_create(g.model, ap, need, blocks, &g.pool) != SSLM_OK) return false;
	size_t n = 0;
	sslm_schema_name(g.model, 0, nullptr, &n);
	std::string name(n, '\0');
	sslm_schema_name(g.model, 0, name.data(), &n);
	return sslm_schema_lookup(g.model, name.c_str(), &g.schema) == SSLM_OK;
}

extern "C" void* CpuBuildRoute(const char* route, const int32_t* p_tokens, int32_t p_count, int32_t t0, int64_t cap) {
	std::vector<int32_t> P(p_tokens, p_tokens + p_count);
	sslm_seq s = nullptr;
	sslm_seq_create(g.model, &g.pool, &s);
	sslm_seq_set_schema(s, g.schema);
	if (!std::strcmp(route, "R")) {
		PrefillAll(s, P, SSLM_SPAN_PROMPT);
		PrefillAll(s, {t0}, SSLM_SPAN_SCHEMA_CONTENT);
	} else if (!std::strcmp(route, "D")) {
		PrefillAll(s, P, SSLM_SPAN_PROMPT);
	} else {  // C
		std::vector<int32_t> fill;
		for (int64_t i = 0; i < cap - 1; ++i) fill.push_back(P[static_cast<size_t>(i) % P.size()]);
		PrefillAll(s, fill, SSLM_SPAN_PROMPT);
		PrefillAll(s, {t0}, SSLM_SPAN_SCHEMA_CONTENT);
	}
	return s;
}

// Reads current_token (offset 72) and context_length (offset 60) WITHOUT decoding.
extern "C" void CpuPeekState(void* handle, int32_t* out_current_token, int64_t* out_ctx) {
	sslm_seq s = static_cast<sslm_seq>(handle);
	size_t n = sslm_seq_state_size(g.model);
	std::vector<uint8_t> b(n);
	size_t nn = n;
	sslm_seq_save(s, b.data(), &nn);
	*out_current_token = static_cast<int32_t>(Le32(&b[72]));
	*out_ctx = static_cast<int64_t>(Le64(&b[60]));
}

// One sslm_decode_step call. Returns status, out token, resulting context_length, and a content
// hash over the blob past its fixed header (offset 124 -- kv_sha8's own established technique).
extern "C" void CpuDecodeOnce(void* handle, int32_t* out_status, int32_t* out_token, int64_t* out_ctx,
                                uint64_t* out_hash) {
	sslm_seq s = static_cast<sslm_seq>(handle);
	sslm_decode_params p{};
	p.struct_size = sizeof(p);
	p.layer_budget = g.layers;
	int32_t out = 12345;
	sslm_status st = sslm_decode_step(g.model, &s, 1, &p, nullptr, &out);
	*out_status = static_cast<int32_t>(st);
	*out_token = out;
	size_t n = sslm_seq_state_size(g.model);
	std::vector<uint8_t> b(n);
	size_t nn = n;
	sslm_seq_save(s, b.data(), &nn);
	*out_ctx = static_cast<int64_t>(Le64(&b[60]));
	uint64_t h = 0;
	for (size_t i = 124; i < nn; ++i) h = h * 1099511628211ull + b[i];
	*out_hash = h;
}

extern "C" void CpuReleaseSeq(void* handle) { sslm_seq_release(static_cast<sslm_seq>(handle)); }

extern "C" int32_t CpuOkValue() { return static_cast<int32_t>(SSLM_OK); }

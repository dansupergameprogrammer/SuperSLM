// T-2900 (Curie), CPU-side helper for cell_gpu_cell1_realschema.cpp -- own translation unit,
// same technique as cell3_cpu_side.cpp (avoids the fixture_common.h `using enum SslmGpuStatus`
// collision with sslm_abi.h's plain enum). ADOPTED VERBATIM from
// `Claude/Vitruvius/t2897-probe/real_schema_dead_end_cpu.cpp` (T-2896/T-2897's own item 3): drives
// the REAL G5 1.5B artifact's own real, compiled, multi-state schema
// ("shopkeeper_intent_extraction", state_count=594) to a genuine dead end via the
// census-discovered 251-token path (g5_schema_census.cpp, independent second parse), then
// exercises the Sec3.10 dead-end cell and a save/restore round trip, at production scale, on CPU.
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <vector>
#include "superslm/sslm_abi.h"

namespace {
struct Cpu {
	std::vector<uint8_t> bytes;
	sslm_model model = nullptr;
	std::vector<uint8_t> pool_store;
	sslm_kv_pool pool = nullptr;
	sslm_schema schema = nullptr;
	int32_t layers = 0;
} g;
}  // namespace

extern "C" bool CpuOpenReal(const char* path) {
	FILE* f = std::fopen(path, "rb");
	if (!f) return false;
	std::fseek(f, 0, SEEK_END);
	long sz = std::ftell(f);
	std::fseek(f, 0, SEEK_SET);
	g.bytes.resize(sz);
	std::fread(g.bytes.data(), 1, sz, f);
	std::fclose(f);
	if (sslm_model_map(g.bytes.data(), g.bytes.size(), &g.model) != SSLM_OK) return false;
	char name[256];
	size_t n = sizeof(name) - 1;
	if (sslm_schema_name(g.model, 0, name, &n) != SSLM_OK) return false;
	name[n] = 0;
	if (std::strcmp(name, "shopkeeper_intent_extraction") != 0) {
		std::printf("CPU: schema[0] name mismatch: %s\n", name);
		return false;
	}
	if (sslm_schema_lookup(g.model, name, &g.schema) != SSLM_OK) return false;
	const uint32_t blocks = 4;
	size_t bb = sslm_kv_block_size(g.model), ov = sslm_kv_pool_overhead_size(g.model, blocks);
	size_t need = bb * blocks + ov;
	g.pool_store.assign(need + 63, 0);
	void* ap = g.pool_store.data();
	size_t space = g.pool_store.size();
	std::align(64, need, ap, space);
	if (sslm_kv_pool_create(g.model, ap, need, blocks, &g.pool) != SSLM_OK) return false;
	return true;
}

extern "C" void* CpuBuildAndDriveToDeadEnd(const int32_t* path_tokens, int32_t path_count, int32_t layer_budget,
                                            int32_t* out_status_first_dead_end, int32_t* out_context_length) {
	g.layers = layer_budget;
	sslm_seq s = nullptr;
	if (sslm_seq_create(g.model, &g.pool, &s) != SSLM_OK) return nullptr;
	if (sslm_seq_set_schema(s, g.schema) != SSLM_OK) {
		sslm_seq_release(s);
		return nullptr;
	}
	int32_t off = 0;
	while (off < path_count) {
		int32_t consumed = 0;
		sslm_status st =
		    sslm_prefill(g.model, s, path_tokens + off, path_count - off, layer_budget, SSLM_SPAN_SCHEMA_CONTENT,
		                 nullptr, &consumed);
		if (st != SSLM_OK) {
			*out_status_first_dead_end = static_cast<int32_t>(st);
			*out_context_length = -1;
			return s;
		}
		if (consumed <= 0) {
			sslm_seq_release(s);
			return nullptr;
		}
		off += consumed;
	}
	sslm_decode_params dp{};
	dp.struct_size = sizeof(dp);
	dp.layer_budget = layer_budget;
	int32_t out = 12345;
	sslm_decode_step(g.model, &s, 1, &dp, nullptr, &out);
	size_t n = sslm_seq_state_size(g.model);
	std::vector<uint8_t> b(n);
	size_t nn = n;
	sslm_seq_save(s, b.data(), &nn);
	*out_status_first_dead_end = out;  // -2 expected
	*out_context_length = static_cast<int32_t>(static_cast<uint32_t>(b[60] | (b[61] << 8) | (b[62] << 16) | (b[63] << 24)));
	return s;
}

extern "C" int32_t CpuRetryDecode(void* handle, int32_t layer_budget) {
	sslm_seq s = static_cast<sslm_seq>(handle);
	sslm_decode_params dp{};
	dp.struct_size = sizeof(dp);
	dp.layer_budget = layer_budget;
	int32_t out = 12345;
	sslm_decode_step(g.model, &s, 1, &dp, nullptr, &out);
	return out;
}

extern "C" int32_t CpuSaveRestoreDecode(void* handle, int32_t layer_budget) {
	sslm_seq s = static_cast<sslm_seq>(handle);
	size_t n = sslm_seq_state_size(g.model);
	std::vector<uint8_t> blob(n);
	size_t nn = n;
	if (sslm_seq_save(s, blob.data(), &nn) != SSLM_OK) return -999;
	sslm_seq restored = nullptr;
	if (sslm_seq_restore(g.model, &g.pool, blob.data(), nn, &restored) != SSLM_OK) return -998;
	sslm_decode_params dp{};
	dp.struct_size = sizeof(dp);
	dp.layer_budget = layer_budget;
	int32_t out = 12345;
	sslm_decode_step(g.model, &restored, 1, &dp, nullptr, &out);
	sslm_seq_release(restored);
	return out;
}

extern "C" void CpuReleaseSeq(void* handle) { sslm_seq_release(static_cast<sslm_seq>(handle)); }

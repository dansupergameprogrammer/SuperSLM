// t2139_c4_smoke_negative.cpp -- Gate B's MUST-REJECT half for C4 (S9, see
// t2139_c2_smoke_negative.cpp's own header comment for this construction's shape). The hostile
// call: sslm_decode_step with a negative layer_budget over an otherwise perfectly real, freshly
// prefilled sequence.
//
// Usage: t2139_c4_smoke_negative.exe <path-to-real.sslm>
#include <cstdio>
#include <fstream>
#include <vector>

#include "superslm/sslm_abi.h"

namespace {
bool ReadFile(const char* path, std::vector<uint8_t>* out) {
	std::ifstream f(path, std::ios::binary);
	if (!f) return false;
	f.seekg(0, std::ios::end);
	const std::streamoff size = f.tellg();
	if (size < 0) return false;
	f.seekg(0, std::ios::beg);
	out->resize(static_cast<size_t>(size));
	if (size > 0) f.read(reinterpret_cast<char*>(out->data()), size);
	return static_cast<bool>(f) || f.eof();
}
}  // namespace

int main(int argc, char** argv) {
	if (argc < 2) {
		std::fprintf(stderr, "usage: %s <path-to-real.sslm>\n", argv[0]);
		return 1;
	}
	std::vector<uint8_t> bytes;
	if (!ReadFile(argv[1], &bytes)) {
		std::fprintf(stderr, "FAIL: could not read %s\n", argv[1]);
		return 1;
	}
	sslm_model model = nullptr;
	if (sslm_model_map(bytes.data(), bytes.size(), &model) != SSLM_OK || !model) {
		std::fprintf(stderr, "FAIL: sslm_model_map returned non-OK\n");
		return 1;
	}
	const uint32_t block_count = 1;
	const size_t block_bytes = sslm_kv_block_size(model);
	const size_t overhead = sslm_kv_pool_overhead_size(model, block_count);
	const size_t pool_buf_size = block_bytes * block_count + overhead;
	std::vector<uint8_t> pool_raw_storage(pool_buf_size + 63);
	void* pool_raw = pool_raw_storage.data();
	size_t pool_raw_space = pool_raw_storage.size();
	std::align(64, pool_buf_size, pool_raw, pool_raw_space);
	sslm_kv_pool pool = nullptr;
	if (sslm_kv_pool_create(model, pool_raw, pool_buf_size, block_count, &pool) != SSLM_OK) {
		std::fprintf(stderr, "FAIL: sslm_kv_pool_create returned non-OK\n");
		sslm_model_unmap(model);
		return 1;
	}
	sslm_seq seq = nullptr;
	if (sslm_seq_create(model, &pool, &seq) != SSLM_OK) {
		std::fprintf(stderr, "FAIL: sslm_seq_create returned non-OK\n");
		sslm_kv_pool_destroy(pool);
		sslm_model_unmap(model);
		return 1;
	}
	const int32_t tokens[1] = {0};
	int32_t consumed = 0;
	if (sslm_prefill(model, seq, tokens, 1, 1, SSLM_SPAN_PROMPT, nullptr, &consumed) != SSLM_OK) {
		std::fprintf(stderr, "FAIL: sslm_prefill returned non-OK\n");
		sslm_seq_release(seq);
		sslm_kv_pool_destroy(pool);
		sslm_model_unmap(model);
		return 1;
	}
	// The hostile call: a negative layer_budget.
	sslm_decode_params hostile_params{};
	// T-2199 Phase D review addendum (D-SLM3797, Dan): struct_size is the FIRST
	// new field the library now validates -- an unrecognized size is a loud rejection.
	hostile_params.struct_size = sizeof(hostile_params);
	hostile_params.layer_budget = -1;
	int32_t out_token = 0;
	sslm_seq batch[1] = {seq};
	const sslm_status st = sslm_decode_step(model, batch, 1, &hostile_params, nullptr, &out_token);
	sslm_seq_release(seq);
	sslm_kv_pool_destroy(pool);
	sslm_model_unmap(model);
	if (st != SSLM_INVALID_ARGUMENT) {
		std::fprintf(stderr,
		             "GATE B MUST-REJECT FAILED: sslm_decode_step(layer_budget=-1) returned %d, "
		             "expected SSLM_INVALID_ARGUMENT\n",
		             static_cast<int>(st));
		return 1;
	}
	std::printf("t2139_c4_smoke_negative: PASS (negative layer_budget correctly rejected)\n");
	return 0;
}

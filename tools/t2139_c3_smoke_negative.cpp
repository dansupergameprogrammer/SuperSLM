// t2139_c3_smoke_negative.cpp -- Gate B's MUST-REJECT half for C3 (S9, see t2139_c2_smoke_negative.cpp's
// own header comment for this construction's shape and why it differs from Gate A/C's negatives).
// The hostile call: sslm_seq_adopt_prefix against a prefix that was never frozen (M2's own
// SSLM_PREFIX_FROZEN_REJECTED naming confusion, fixed this same round, does not change WHICH
// status this returns -- it is now SSLM_INVALID_ARGUMENT; either way the call must be REJECTED,
// never SSLM_OK).
//
// Usage: t2139_c3_smoke_negative.exe <path-to-real.sslm>
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
	const uint32_t block_count = 2;
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
	sslm_prefix prefix = nullptr;
	sslm_seq seq = nullptr;
	if (sslm_prefix_begin(model, &pool, &prefix) != SSLM_OK ||
	    sslm_seq_create(model, &pool, &seq) != SSLM_OK) {
		std::fprintf(stderr, "FAIL: sslm_prefix_begin/sslm_seq_create returned non-OK\n");
		sslm_kv_pool_destroy(pool);
		sslm_model_unmap(model);
		return 1;
	}
	// The hostile call: adopt a prefix that was never frozen.
	const sslm_status st = sslm_seq_adopt_prefix(seq, prefix);
	sslm_seq_release(seq);
	sslm_prefix_release(prefix);
	sslm_kv_pool_destroy(pool);
	sslm_model_unmap(model);
	if (st != SSLM_INVALID_ARGUMENT) {
		std::fprintf(stderr,
		             "GATE B MUST-REJECT FAILED: sslm_seq_adopt_prefix(unfrozen prefix) returned "
		             "%d, expected SSLM_INVALID_ARGUMENT\n",
		             static_cast<int>(st));
		return 1;
	}
	std::printf("t2139_c3_smoke_negative: PASS (unfrozen-prefix adoption correctly rejected)\n");
	return 0;
}

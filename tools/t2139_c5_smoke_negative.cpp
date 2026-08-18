// t2139_c5_smoke_negative.cpp -- Gate B's MUST-REJECT half for C5 (S9, see
// t2139_c2_smoke_negative.cpp's own header comment for this construction's shape). The hostile
// call: sslm_seq_restore against a blob whose magic has been corrupted -- M1 (Claude/Poirot/
// 2c18dab-t2139-abi-build-review.md) retargeted this from SSLM_RESTORE_MODEL_MISMATCH to
// SSLM_INVALID_ARGUMENT this same fix round; this gate checks the CURRENT, correct status.
//
// Usage: t2139_c5_smoke_negative.exe <path-to-real.sslm>
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
	size_t need = 0;
	sslm_seq_save(seq, nullptr, &need);
	std::vector<uint8_t> blob(need);
	size_t n = blob.size();
	if (sslm_seq_save(seq, blob.data(), &n) != SSLM_OK) {
		std::fprintf(stderr, "FAIL: sslm_seq_save returned non-OK\n");
		sslm_seq_release(seq);
		sslm_kv_pool_destroy(pool);
		sslm_model_unmap(model);
		return 1;
	}
	// The hostile call: corrupt the magic, then restore.
	blob[0] = static_cast<uint8_t>(~blob[0]);
	sslm_seq restored = nullptr;
	const sslm_status st = sslm_seq_restore(model, &pool, blob.data(), n, &restored);
	sslm_seq_release(seq);
	sslm_kv_pool_destroy(pool);
	sslm_model_unmap(model);
	if (st != SSLM_INVALID_ARGUMENT || restored != nullptr) {
		std::fprintf(stderr,
		             "GATE B MUST-REJECT FAILED: sslm_seq_restore(corrupted magic) returned %d, "
		             "expected SSLM_INVALID_ARGUMENT\n",
		             static_cast<int>(st));
		return 1;
	}
	std::printf("t2139_c5_smoke_negative: PASS (corrupted magic correctly rejected)\n");
	return 0;
}

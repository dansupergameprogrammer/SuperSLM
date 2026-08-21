// t2139_c5_smoke.cpp -- Gate B for C5 (design Sec9: "a smoke TU that saves a real sequence
// mid-generation, restores it into a fresh handle, decodes one more step on each, compares").
// Also exercises the hostile-blob rejections design Sec9's own C5 gate names: a corrupted magic
// (SSLM_RESTORE_MODEL_MISMATCH -- the magic check IS the model-mismatch class per design
// Sec7.3's own hard-reject strategy), a corrupted model_hash (SSLM_RESTORE_MODEL_MISMATCH), a
// corrupted kv_precision (SSLM_RESTORE_KV_MISMATCH), and the two-call sizing convention
// (SSLM_BUFFER_TOO_SMALL with *n set to the real requirement).
//
// Usage: t2139_c5_smoke.exe <path-to-real.sslm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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
	sslm_status st = sslm_model_map(bytes.data(), bytes.size(), &model);
	if (st != SSLM_OK || !model) {
		std::fprintf(stderr, "FAIL: sslm_model_map returned %d\n", static_cast<int>(st));
		return 1;
	}

	// A 2-block pool: block 1 for the live sequence, block 2 for the restored one.
	const uint32_t block_count = 2;
	const size_t block_bytes = sslm_kv_block_size(model);
	const size_t overhead = sslm_kv_pool_overhead_size(model, block_count);
	// S2 (Claude/Poirot/2c18dab-t2139-abi-build-review.md): sslm_kv_pool_create now checks
	// alignment -- over-allocate and round up, matching tools/t2139_c2_smoke.cpp's own pattern.
	const size_t pool_buf_size = block_bytes * block_count + overhead;
	std::vector<uint8_t> pool_raw_storage(pool_buf_size + 63);
	void* pool_raw = pool_raw_storage.data();
	size_t pool_raw_space = pool_raw_storage.size();
	std::align(64, pool_buf_size, pool_raw, pool_raw_space);
	sslm_kv_pool pool = nullptr;
	st = sslm_kv_pool_create(model, pool_raw, pool_buf_size, block_count, &pool);
	if (st != SSLM_OK || !pool) {
		std::fprintf(stderr, "FAIL: sslm_kv_pool_create returned %d\n", static_cast<int>(st));
		return 1;
	}

	sslm_seq seq = nullptr;
	st = sslm_seq_create(model, &pool, &seq);
	if (st != SSLM_OK || !seq) {
		std::fprintf(stderr, "FAIL: sslm_seq_create returned %d\n", static_cast<int>(st));
		return 1;
	}

	const int32_t prompt_tokens[] = {1, 2, 3};
	int32_t consumed = 0;
	st = sslm_prefill(model, seq, prompt_tokens, 3, 3, SSLM_SPAN_PROMPT, nullptr, &consumed);
	if (st != SSLM_OK || consumed != 3) {
		std::fprintf(stderr, "FAIL: sslm_prefill returned %d consumed %d\n", static_cast<int>(st),
		             consumed);
		return 1;
	}

	sslm_decode_params params{};
	// T-2199 Phase D review addendum (D-SLM3797, Dan): struct_size is the FIRST
	// new field the library now validates -- an unrecognized size is a loud rejection.
	params.struct_size = sizeof(params);
	// A small bounded layer_budget so the saved state is genuinely "mid-generation" (design's
	// own smoke shape) rather than always resting between tokens.
	params.layer_budget = 1;
	sslm_seq seqs[1] = {seq};
	int32_t out_token = -1;
	// Drive a handful of bounded decode_step calls so the sequence is captured genuinely
	// mid-token (layer_index > 0, a nonzero residual) -- the exact "mid-generation" shape design
	// Sec9's own C5 gate names, not merely a between-tokens rest state.
	for (int i = 0; i < 3; ++i) {
		st = sslm_decode_step(model, seqs, 1, &params, nullptr, &out_token);
		if (st != SSLM_OK) {
			std::fprintf(stderr, "FAIL: sslm_decode_step returned %d\n", static_cast<int>(st));
			return 1;
		}
	}
	std::printf("driven mid-generation (bounded layer_budget=1, 3 calls, still pending): PASS\n");

	// --- two-call sizing convention ---
	size_t required = 0;
	st = sslm_seq_save(seq, nullptr, &required);
	if (st != SSLM_BUFFER_TOO_SMALL || required == 0) {
		std::fprintf(stderr,
		             "FAIL: sslm_seq_save(nullptr) returned %d, required=%zu -- expected "
		             "SSLM_BUFFER_TOO_SMALL with a real size\n",
		             static_cast<int>(st), required);
		return 1;
	}
	std::vector<uint8_t> blob(required);
	size_t n = blob.size();
	st = sslm_seq_save(seq, blob.data(), &n);
	if (st != SSLM_OK) {
		std::fprintf(stderr, "FAIL: sslm_seq_save returned %d\n", static_cast<int>(st));
		return 1;
	}
	blob.resize(n);
	std::printf("sslm_seq_save: PASS (%zu bytes, mid-token)\n", n);

	// --- hostile blobs ---
	{
		std::vector<uint8_t> corrupted = blob;
		corrupted[0] = static_cast<uint8_t>(~corrupted[0]);  // corrupt the magic
		sslm_seq bad = nullptr;
		st = sslm_seq_restore(model, &pool, corrupted.data(), corrupted.size(), &bad);
		// M1 (Claude/Poirot/2c18dab-t2139-abi-build-review.md): a bad magic is a malformed/
		// foreign-format blob, not a model mismatch -- SSLM_INVALID_ARGUMENT now, not
		// SSLM_RESTORE_MODEL_MISMATCH (which stays correct for the model_hash-mismatch cell
		// immediately below, a genuinely different-model blob against a well-formed magic).
		if (st != SSLM_INVALID_ARGUMENT || bad != nullptr) {
			std::fprintf(stderr,
			             "FAIL: sslm_seq_restore(corrupted magic) returned %d, expected "
			             "SSLM_INVALID_ARGUMENT\n",
			             static_cast<int>(st));
			return 1;
		}
	}
	{
		std::vector<uint8_t> corrupted = blob;
		corrupted[4] = static_cast<uint8_t>(~corrupted[4]);  // corrupt one byte of model_hash
		sslm_seq bad = nullptr;
		st = sslm_seq_restore(model, &pool, corrupted.data(), corrupted.size(), &bad);
		if (st != SSLM_RESTORE_MODEL_MISMATCH || bad != nullptr) {
			std::fprintf(stderr,
			             "FAIL: sslm_seq_restore(corrupted model_hash) returned %d, expected "
			             "SSLM_RESTORE_MODEL_MISMATCH\n",
			             static_cast<int>(st));
			return 1;
		}
	}
	{
		std::vector<uint8_t> corrupted = blob;
		corrupted[36] = static_cast<uint8_t>(corrupted[36] ^ 0xFF);  // corrupt kv_precision
		sslm_seq bad = nullptr;
		st = sslm_seq_restore(model, &pool, corrupted.data(), corrupted.size(), &bad);
		if (st != SSLM_RESTORE_KV_MISMATCH || bad != nullptr) {
			std::fprintf(stderr,
			             "FAIL: sslm_seq_restore(corrupted kv_precision) returned %d, expected "
			             "SSLM_RESTORE_KV_MISMATCH\n",
			             static_cast<int>(st));
			return 1;
		}
	}
	std::printf("hostile blob rejections (magic, model_hash, kv_precision): PASS\n");

	// --- design's own smoke shape: restore into a fresh handle, decode one more step on each,
	// compare ---
	sslm_seq restored = nullptr;
	st = sslm_seq_restore(model, &pool, blob.data(), blob.size(), &restored);
	if (st != SSLM_OK || !restored) {
		std::fprintf(stderr, "FAIL: sslm_seq_restore (well-formed) returned %d\n",
		             static_cast<int>(st));
		return 1;
	}

	// Drive both paths, bounded one layer at a time, until EACH produces a real token -- a
	// stronger check than a single one-more-step call (design's own minimum): if the restored
	// blob's residual/layer_index/KV bytes diverged from the original at all, the two paths
	// would very likely disagree either on WHEN a token completes or WHICH token it is.
	sslm_seq orig_seqs[1] = {seq};
	sslm_seq restored_seqs[1] = {restored};
	int32_t out_orig = -1, out_restored = -1;
	int guard = 0;
	while (out_orig < 0) {
		st = sslm_decode_step(model, orig_seqs, 1, &params, nullptr, &out_orig);
		if (st != SSLM_OK) {
			std::fprintf(stderr, "FAIL: sslm_decode_step (original) returned %d\n",
			             static_cast<int>(st));
			return 1;
		}
		if (++guard > 100000) {
			std::fprintf(stderr, "FAIL: original sequence never completed a token\n");
			return 1;
		}
	}
	guard = 0;
	while (out_restored < 0) {
		st = sslm_decode_step(model, restored_seqs, 1, &params, nullptr, &out_restored);
		if (st != SSLM_OK) {
			std::fprintf(stderr, "FAIL: sslm_decode_step (restored) returned %d\n",
			             static_cast<int>(st));
			return 1;
		}
		if (++guard > 100000) {
			std::fprintf(stderr, "FAIL: restored sequence never completed a token\n");
			return 1;
		}
	}
	if (out_orig != out_restored) {
		std::fprintf(stderr,
		             "FAIL: post-restore decode diverges: original=%d restored=%d\n", out_orig,
		             out_restored);
		return 1;
	}
	std::printf("save/restore + decode-to-next-token comparison: PASS (both produce %d)\n",
	            out_orig);

	sslm_seq_release(seq);
	sslm_seq_release(restored);
	sslm_kv_pool_destroy(pool);
	sslm_model_unmap(model);

	std::printf("t2139_c5_smoke: PASS\n");
	return 0;
}

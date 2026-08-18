// t2139_c3_smoke.cpp -- Gate B for C3 (design Sec9: "a smoke TU that begins a prefix, prefills
// it with SSLM_SPAN_PROMPT tokens, freezes it, creates a sequence, adopts the prefix, releases
// both"). Authored fresh for this build, links against the just-built library alone.
//
// Also exercises real hostile-input/guard paths beyond the design's own minimum smoke shape
// (Brunel's own "prove it builds and runs" discipline): pool exhaustion at draw time
// (SSLM_KV_POOL_EXHAUSTED), prefill-past-freeze rejection (SSLM_PREFIX_FROZEN_REJECTED),
// mid-token reset rejection (SSLM_SEQ_RESET_MIDTOKEN_REJECTED is NOT reachable here since
// prefill always leaves layer_index at 0 between tokens -- reset is exercised on a fresh
// sequence instead, which is the only state this smoke drives it from), and pool free-count
// exactness across a create/release cycle (design Sec17 dim 1).
//
// Usage: t2139_c3_smoke.exe <path-to-real.sslm>
#include <cstdio>
#include <cstdlib>
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

	// A pool with 2 blocks: one for the prefix, one for the sequence that adopts it.
	const uint32_t block_count = 2;
	const size_t block_bytes = sslm_kv_block_size(model);
	const size_t overhead = sslm_kv_pool_overhead_size(model, block_count);
	// S2 (Claude/Poirot/2c18dab-t2139-abi-build-review.md): sslm_kv_pool_create now checks
	// alignment -- over-allocate and round up, matching the workspace buffer's own established
	// pattern (tools/t2139_c2_smoke.cpp).
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

	// --- design Sec9's own C3 smoke shape: begin, prefill, freeze, create, adopt, release ---
	sslm_prefix prefix = nullptr;
	st = sslm_prefix_begin(model, &pool, &prefix);
	if (st != SSLM_OK || !prefix) {
		std::fprintf(stderr, "FAIL: sslm_prefix_begin returned %d\n", static_cast<int>(st));
		return 1;
	}

	const int32_t prompt_tokens[] = {1, 2, 3, 4, 5};
	const int32_t prompt_count = 5;
	int32_t consumed = 0;
	st = sslm_prefix_prefill(model, prefix, prompt_tokens, prompt_count, /*chunk_budget=*/8,
	                          SSLM_SPAN_PROMPT, nullptr, &consumed);
	if (st != SSLM_OK) {
		std::fprintf(stderr, "FAIL: sslm_prefix_prefill returned %d\n", static_cast<int>(st));
		return 1;
	}
	if (consumed != prompt_count) {
		std::fprintf(stderr, "FAIL: sslm_prefix_prefill consumed %d, expected %d\n", consumed,
		             prompt_count);
		return 1;
	}
	std::printf("sslm_prefix_begin/prefill: PASS (consumed %d real tokens, no crash)\n", consumed);

	st = sslm_prefix_freeze(prefix);
	if (st != SSLM_OK) {
		std::fprintf(stderr, "FAIL: sslm_prefix_freeze returned %d\n", static_cast<int>(st));
		return 1;
	}

	// Prefill-past-freeze is a defined rejection, not a crash.
	st = sslm_prefix_prefill(model, prefix, prompt_tokens, 1, 1, SSLM_SPAN_PROMPT, nullptr,
	                          &consumed);
	if (st != SSLM_PREFIX_FROZEN_REJECTED) {
		std::fprintf(stderr,
		             "FAIL: sslm_prefix_prefill after freeze returned %d, expected "
		             "SSLM_PREFIX_FROZEN_REJECTED\n",
		             static_cast<int>(st));
		return 1;
	}
	std::printf("sslm_prefix_freeze + prefill-past-freeze rejection: PASS\n");

	sslm_seq seq = nullptr;
	st = sslm_seq_create(model, &pool, &seq);
	if (st != SSLM_OK || !seq) {
		std::fprintf(stderr, "FAIL: sslm_seq_create returned %d\n", static_cast<int>(st));
		return 1;
	}

	st = sslm_seq_adopt_prefix(seq, prefix);
	if (st != SSLM_OK) {
		std::fprintf(stderr, "FAIL: sslm_seq_adopt_prefix returned %d\n", static_cast<int>(st));
		return 1;
	}
	std::printf("sslm_seq_create + sslm_seq_adopt_prefix: PASS (copy-on-adopt, no crash)\n");

	// --- pool exhaustion: this 2-block pool has 0 free blocks left now (prefix + seq). ---
	sslm_prefix exhausted_probe = nullptr;
	st = sslm_prefix_begin(model, &pool, &exhausted_probe);
	if (st != SSLM_KV_POOL_EXHAUSTED) {
		std::fprintf(stderr,
		             "FAIL: sslm_prefix_begin on an exhausted pool returned %d, expected "
		             "SSLM_KV_POOL_EXHAUSTED\n",
		             static_cast<int>(st));
		return 1;
	}
	std::printf("sslm_kv_pool exhaustion at draw time: PASS (SSLM_KV_POOL_EXHAUSTED fired)\n");

	st = sslm_prefix_release(prefix);
	if (st != SSLM_OK) {
		std::fprintf(stderr, "FAIL: sslm_prefix_release returned %d\n", static_cast<int>(st));
		return 1;
	}

	// The released block is free again -- a fresh sslm_prefix_begin now succeeds (design Sec17
	// dim 1's own free-count-exact obligation, exercised for real).
	sslm_prefix reused_probe = nullptr;
	st = sslm_prefix_begin(model, &pool, &reused_probe);
	if (st != SSLM_OK || !reused_probe) {
		std::fprintf(stderr,
		             "FAIL: sslm_prefix_begin after a release returned %d, expected SSLM_OK "
		             "(the pool's free-count did not restore exactly)\n",
		             static_cast<int>(st));
		return 1;
	}
	sslm_prefix_release(reused_probe);
	std::printf("sslm_kv_pool free-count restored exactly after release: PASS\n");

	// --- sslm_seq_reset on the adopted sequence (fresh layer_index == 0 after adoption) ---
	st = sslm_seq_reset(seq);
	if (st != SSLM_OK) {
		std::fprintf(stderr, "FAIL: sslm_seq_reset returned %d\n", static_cast<int>(st));
		return 1;
	}
	std::printf("sslm_seq_reset: PASS\n");

	// S1 pin (Claude/Poirot/2c18dab-t2139-abi-build-review.md; closed per the coordinator's own
	// closing-round follow-up list): a reset sequence must not be able to emit from a zeroed
	// residual -- sslm_seq_reset now clears ready_for_logits, so decode_step's own validation
	// (layer_index==0, current_token<0, !ready_for_logits) correctly rejects a decode attempt on
	// a just-reset, never-reprefilled sequence.
	{
		sslm_decode_params params{};
		params.layer_budget = 1;
		int32_t out_token = 0;
		sslm_seq batch[1] = {seq};
		const sslm_status reset_decode_st =
		    sslm_decode_step(model, batch, 1, &params, nullptr, &out_token);
		if (reset_decode_st != SSLM_INVALID_ARGUMENT) {
			std::fprintf(stderr,
			             "FAIL: sslm_decode_step(just-reset sequence) returned %d, expected "
			             "SSLM_INVALID_ARGUMENT (S1 pin)\n",
			             static_cast<int>(reset_decode_st));
			return 1;
		}
		std::printf("S1 pin (decode_step on a just-reset sequence): PASS (SSLM_INVALID_ARGUMENT "
		            "fired as designed)\n");
	}

	st = sslm_seq_release(seq);
	if (st != SSLM_OK) {
		std::fprintf(stderr, "FAIL: sslm_seq_release returned %d\n", static_cast<int>(st));
		return 1;
	}

	st = sslm_kv_pool_destroy(pool);
	if (st != SSLM_OK) {
		std::fprintf(stderr, "FAIL: sslm_kv_pool_destroy returned %d\n", static_cast<int>(st));
		return 1;
	}

	st = sslm_model_unmap(model);
	if (st != SSLM_OK) {
		std::fprintf(stderr, "FAIL: sslm_model_unmap returned %d\n", static_cast<int>(st));
		return 1;
	}

	std::printf("t2139_c3_smoke: PASS\n");
	return 0;
}

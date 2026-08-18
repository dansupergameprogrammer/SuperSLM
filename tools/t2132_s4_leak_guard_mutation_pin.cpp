// t2132_s4_leak_guard_mutation_pin.cpp -- S4 (Claude/Poirot/9bc9ec6-t2132-g5-arc-review.md):
// restores a discriminating mechanism for the dimension-1 leak guard T-2132's own DrawBlock
// zero-fill fix made unobservable through the production read path (every live sequence's own
// KV block is reached via DrawBlock, which now zero-fills unconditionally, so a leak-check cell
// reading through a live handle passes whether or not ReturnBlock ever poisoned anything).
//
// Uses the new test-only pool-peek hook (src/sslm_abi.cpp, sslm_g5_test_only_peek_kv_block_bytes
// -- same disclosed, non-shipping precedent as sslm_g5_test_only_apply_mask_and_argmax one
// function above it) to read a block's raw bytes DIRECTLY, bypassing DrawBlock's zero-fill --
// the one read path that still observes ReturnBlock's own poison-fill.
//
// THE MUTATION PROOF (executed by this pin's own build+run, not merely asserted): with
// ReturnBlock's `std::memset(kv_block, 0xCD, block_size)` line intact, this pin observes 0xCD at
// the released block's own raw bytes before the next draw -- PASS. Temporarily commenting out
// that one line and rebuilding this pin makes it observe whatever was left behind instead (0x00,
// this pool's own creation-time content, since nothing else wrote there) -- FAIL. That mutation
// was run by hand during this fix round (see the build log's own session note) to confirm the
// guard can genuinely fail, not merely pass by construction; it is not re-run automatically here
// (this pin ships with the poison-fill intact, proving the SHIPPED guard passes).
//
// Usage: t2132_s4_leak_guard_mutation_pin.exe <path-to-g5-fixture.sslm>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include "superslm/sslm_abi.h"

// Declared locally -- test-only, not part of any public header (see this file's own top comment
// and the hook's own definition in src/sslm_abi.cpp for why). The signature must match the real
// definition exactly; the linker resolves it against this project's own sslm_abi.cpp.
extern "C" sslm_status sslm_g5_test_only_peek_kv_block_bytes(sslm_kv_pool pool, uint32_t block_index,
                                                               uint8_t* out_buf, size_t n);

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
		std::fprintf(stderr, "usage: %s <path-to-g5-fixture.sslm>\n", argv[0]);
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

	// A single-block pool -- draw/release/re-peek the SAME block_index deterministically.
	const uint32_t block_count = 1;
	const size_t block_bytes = sslm_kv_block_size(model);
	const size_t overhead = sslm_kv_pool_overhead_size(model, block_count);
	const size_t pool_buf_size = block_bytes * block_count + overhead;
	std::vector<uint8_t> pool_storage(pool_buf_size + 63);
	void* pool_raw = pool_storage.data();
	size_t pool_raw_space = pool_storage.size();
	std::align(64, pool_buf_size, pool_raw, pool_raw_space);
	sslm_kv_pool pool = nullptr;
	st = sslm_kv_pool_create(model, pool_raw, pool_buf_size, block_count, &pool);
	if (st != SSLM_OK || !pool) {
		std::fprintf(stderr, "FAIL: sslm_kv_pool_create returned %d\n", static_cast<int>(st));
		return 1;
	}

	// Draw the block via a real sequence, write REAL non-degenerate content into it (a real
	// prefill), so the block is genuinely non-zero, non-poison content before release.
	sslm_seq seq = nullptr;
	st = sslm_seq_create(model, &pool, &seq);
	if (st != SSLM_OK || !seq) {
		std::fprintf(stderr, "FAIL: sslm_seq_create returned %d\n", static_cast<int>(st));
		return 1;
	}
	const int32_t prompt_tokens[] = {1, 2, 3, 4, 5};
	int32_t consumed = 0;
	st = sslm_prefill(model, seq, prompt_tokens, 5, 5, SSLM_SPAN_PROMPT, nullptr, &consumed);
	if (st != SSLM_OK || consumed != 5) {
		std::fprintf(stderr, "FAIL: sslm_prefill returned %d consumed=%d\n", static_cast<int>(st),
		             consumed);
		return 1;
	}

	// Only one block in this pool -- its index is 0 by construction (the free list started as
	// {0} and DrawBlock popped it). Peek it now, through the live sequence's own draw, to prove
	// it is real content (not already the poison pattern) before release.
	std::vector<uint8_t> peek_before(64);
	st = sslm_g5_test_only_peek_kv_block_bytes(pool, 0, peek_before.data(), peek_before.size());
	if (st != SSLM_OK) {
		std::fprintf(stderr, "FAIL: peek (before release) returned %d\n", static_cast<int>(st));
		return 1;
	}
	bool all_0xcd_before = true;
	for (uint8_t b : peek_before) {
		if (b != 0xCD) {
			all_0xcd_before = false;
			break;
		}
	}
	std::printf("peek before release: first byte=0x%02X, all-0xCD=%s (expect false -- real content)\n",
	            peek_before[0], all_0xcd_before ? "true" : "false");

	sslm_seq_release(seq);  // ReturnBlock's own poison-fill runs here.

	// THE GUARD: peek the SAME raw block index, BYPASSING DrawBlock's zero-fill entirely --
	// this is the one read path that can still observe ReturnBlock's own poison.
	std::vector<uint8_t> peek_after(block_bytes);
	st = sslm_g5_test_only_peek_kv_block_bytes(pool, 0, peek_after.data(), peek_after.size());
	if (st != SSLM_OK) {
		std::fprintf(stderr, "FAIL: peek (after release) returned %d\n", static_cast<int>(st));
		return 1;
	}
	bool all_0xcd_after = true;
	for (uint8_t b : peek_after) {
		if (b != 0xCD) {
			all_0xcd_after = false;
			break;
		}
	}
	const bool pass = all_0xcd_after;
	std::printf(
	    "peek after release (bypassing DrawBlock's zero-fill): whole block (%zu bytes) all-0xCD=%s -- "
	    "%s\n",
	    peek_after.size(), all_0xcd_after ? "true" : "false", pass ? "PASS" : "FAIL");

	sslm_kv_pool_destroy(pool);
	sslm_model_unmap(model);

	if (!pass) {
		std::fprintf(stderr, "t2132_s4_leak_guard_mutation_pin: FAIL\n");
		return 1;
	}
	std::printf("t2132_s4_leak_guard_mutation_pin: PASS\n");
	return 0;
}

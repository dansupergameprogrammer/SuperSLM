// t2132_s4_leak_guard_mutation_pin.cpp -- S4 (Claude/Poirot/9bc9ec6-t2132-g5-arc-review.md):
// a discriminating mechanism for the dimension-1 leak guard ("no content from the PRIOR sequence
// survives release"): it reads the released K/V's raw bytes directly from the pool, through a
// test-only peek, rather than through a live holder.
//
// Paged-KV plan (rev 16.1) §3.6, step C4: ported from blocks to pages (paged-KV cell 1.2 is the
// suite's own port). A pool is now a page pool and a page is poisoned with 0xCD on release iff it
// was mapped since its draw (§3.3); a clean reserve page is returned unpoisoned, so the old
// whole-block check no longer holds and no longer discriminates. The pin now asks the sequence's
// page table which page holds rows 0-4 (sslm_pkv_test_only_seq_table_entry), checks that page
// holds real content before release, and peeks the WHOLE page after release
// (sslm_pkv_test_only_peek_page_bytes).
//
// THE MUTATION PROOF: with the page free path's poison fill intact (src/kv_pages.cpp,
// PoisonIfDirty), this pin observes 0xCD over the released page -- PASS. Removing that fill makes
// it observe the released sequence's rows instead -- FAIL.
//
// Usage: t2132_s4_leak_guard_mutation_pin.exe <path-to-g5-fixture.sslm>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include "superslm/sslm_abi.h"

// Declared locally -- test-only, not part of any public header (tests/paged-kv/pkv_abi.h declares
// the same two). The signatures must match the definitions in src/sslm_abi.cpp exactly.
extern "C" sslm_status sslm_pkv_test_only_peek_page_bytes(sslm_kv_pool pool, uint32_t page_index,
                                                          uint8_t* out_buf, size_t n);
extern "C" sslm_status sslm_pkv_test_only_seq_table_entry(sslm_seq seq, uint32_t i, uint32_t* out_page,
                                                         uint32_t* out_mapped);

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

	// A single-block pool: one sequence's whole reservation.
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

	// A real sequence writes REAL non-degenerate content (a real prefill) into its first page, so
	// that page holds genuinely non-poison content before release.
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

	// The page that holds rows 0-4: table entry 0 of the live sequence. The page size is found as
	// the largest n the page peek accepts (it refuses n > page_bytes).
	uint32_t page = 0, mapped = 0;
	st = sslm_pkv_test_only_seq_table_entry(seq, 0, &page, &mapped);
	if (st != SSLM_OK || mapped < 1) {
		std::fprintf(stderr, "FAIL: table entry 0 returned %d (mapped %u)\n", static_cast<int>(st), mapped);
		return 1;
	}
	size_t page_bytes = 0;
	{
		std::vector<uint8_t> probe(block_bytes);
		size_t lo = 1, hi = block_bytes;
		while (lo <= hi) {
			const size_t mid = lo + (hi - lo) / 2;
			if (sslm_pkv_test_only_peek_page_bytes(pool, page, probe.data(), mid) == SSLM_OK) {
				page_bytes = mid;
				lo = mid + 1;
			} else {
				hi = mid - 1;
			}
		}
	}
	if (page_bytes == 0) {
		std::fprintf(stderr, "FAIL: the page peek accepted no size for page %u\n", page);
		return 1;
	}
	std::vector<uint8_t> peek_before(page_bytes);
	st = sslm_pkv_test_only_peek_page_bytes(pool, page, peek_before.data(), peek_before.size());
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
	std::printf("page %u (%zu bytes) before release: all-0xCD=%s (expect false -- real content)\n", page,
	            page_bytes, all_0xcd_before ? "true" : "false");

	sslm_seq_release(seq);  // the page free path's poison fill runs here (the page is dirty).

	// THE GUARD: peek the SAME page after release -- the one read path that observes the poison.
	std::vector<uint8_t> peek_after(page_bytes);
	st = sslm_pkv_test_only_peek_page_bytes(pool, page, peek_after.data(), peek_after.size());
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
	const bool pass = !all_0xcd_before && all_0xcd_after;
	std::printf("page %u after release: whole page (%zu bytes) all-0xCD=%s -- %s\n", page, peek_after.size(),
	            all_0xcd_after ? "true" : "false", pass ? "PASS" : "FAIL");

	sslm_kv_pool_destroy(pool);
	sslm_model_unmap(model);

	if (!pass) {
		std::fprintf(stderr, "t2132_s4_leak_guard_mutation_pin: FAIL\n");
		return 1;
	}
	std::printf("t2132_s4_leak_guard_mutation_pin: PASS\n");
	return 0;
}

// t2139_dim9_current_token_pin.cpp -- design commit 9e2995f4e7's own same-round pin (Sec10 dim
// 9): a sequence saved while resting BETWEEN sslm_decode_step calls (not freshly post-
// sslm_prefill), restored into a fresh handle, then both the live and restored sequences driven
// one further sslm_decode_step call each -- the produced tokens must be bit-identical. This is
// the cell that would have caught T-2139's own measured pre-amendment divergence (live
// continues with 315, the pre-amendment restore produced 0) had it existed first.
//
// Usage: t2139_dim9_current_token_pin.exe <path-to-real.sslm>
#include <cstdio>
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

	const uint32_t block_count = 2;  // live + restored, concurrently resident (D-SLM3454)
	const size_t block_bytes = sslm_kv_block_size(model);
	const size_t overhead = sslm_kv_pool_overhead_size(model, block_count);
	std::vector<uint8_t> pool_raw(block_bytes * block_count + overhead);
	sslm_kv_pool pool = nullptr;
	st = sslm_kv_pool_create(model, pool_raw.data(), pool_raw.size(), block_count, &pool);
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

	const int32_t prompt_tokens[] = {1, 2, 3, 4, 5};
	int32_t consumed = 0;
	st = sslm_prefill(model, seq, prompt_tokens, 5, 5, SSLM_SPAN_PROMPT, nullptr, &consumed);
	if (st != SSLM_OK || consumed != 5) {
		std::fprintf(stderr, "FAIL: sslm_prefill returned %d consumed %d\n", static_cast<int>(st),
		             consumed);
		return 1;
	}

	// Drive the sequence to REST BETWEEN decode steps (not freshly post-prefill): consume the
	// free ready_for_logits step (produces the first token), then complete a second, ordinary
	// decode step (embed + run) so the saved state is genuinely the "produced a token, not yet
	// embedded the next one" shape design commit 9e2995f4e7's own finding names.
	sslm_decode_params params{};
	params.layer_budget = 8;  // a real, bounded budget -- exercises the ordinary embed+RunLayerLoop
	                          // path for the second step, not just the free first-call shortcut.
	sslm_seq seqs[1] = {seq};
	for (int step = 0; step < 2; ++step) {
		int32_t out_token = -1;
		int guard = 0;
		while (out_token < 0) {
			st = sslm_decode_step(model, seqs, 1, &params, nullptr, &out_token);
			if (st != SSLM_OK) {
				std::fprintf(stderr, "FAIL: sslm_decode_step (drive step %d) returned %d\n", step,
				             static_cast<int>(st));
				return 1;
			}
			if (++guard > 100000) {
				std::fprintf(stderr, "FAIL: decode step %d never completed\n", step);
				return 1;
			}
		}
	}
	std::printf("driven to rest BETWEEN decode steps (2 completed steps): PASS\n");

	// --- save mid-rest, restore into a fresh handle ---
	size_t required = 0;
	st = sslm_seq_save(seq, nullptr, &required);
	if (st != SSLM_BUFFER_TOO_SMALL || required == 0) {
		std::fprintf(stderr, "FAIL: sslm_seq_save(nullptr) returned %d, required=%zu\n",
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

	sslm_seq restored = nullptr;
	st = sslm_seq_restore(model, &pool, blob.data(), blob.size(), &restored);
	if (st != SSLM_OK || !restored) {
		std::fprintf(stderr, "FAIL: sslm_seq_restore returned %d\n", static_cast<int>(st));
		return 1;
	}
	std::printf("sslm_seq_save (%zu bytes) + sslm_seq_restore: PASS\n", n);

	// --- drive live and restored one further step each; must be bit-identical ---
	sslm_seq live_seqs[1] = {seq};
	sslm_seq restored_seqs[1] = {restored};
	int32_t live_next = -1, restored_next = -1;
	{
		int guard = 0;
		while (live_next < 0) {
			st = sslm_decode_step(model, live_seqs, 1, &params, nullptr, &live_next);
			if (st != SSLM_OK) {
				std::fprintf(stderr, "FAIL: sslm_decode_step (live) returned %d\n", static_cast<int>(st));
				return 1;
			}
			if (++guard > 100000) {
				std::fprintf(stderr, "FAIL: live sequence never completed its next step\n");
				return 1;
			}
		}
	}
	{
		int guard = 0;
		while (restored_next < 0) {
			st = sslm_decode_step(model, restored_seqs, 1, &params, nullptr, &restored_next);
			if (st != SSLM_OK) {
				std::fprintf(stderr, "FAIL: sslm_decode_step (restored) returned %d\n",
				             static_cast<int>(st));
				return 1;
			}
			if (++guard > 100000) {
				std::fprintf(stderr, "FAIL: restored sequence never completed its next step\n");
				return 1;
			}
		}
	}

	if (live_next != restored_next) {
		std::fprintf(stderr,
		             "FAIL: dim-9 current_token cell: live=%d restored=%d -- MUST be bit-identical "
		             "(design commit 9e2995f4e7)\n",
		             live_next, restored_next);
		return 1;
	}
	std::printf("dim-9 current_token cell: PASS -- live and restored both produce %d (bit-identical)\n",
	            live_next);

	sslm_seq_release(seq);
	sslm_seq_release(restored);
	sslm_kv_pool_destroy(pool);
	sslm_model_unmap(model);

	std::printf("t2139_dim9_current_token_pin: PASS\n");
	return 0;
}

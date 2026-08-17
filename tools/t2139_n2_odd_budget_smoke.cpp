// t2139_n2_odd_budget_smoke.cpp -- N2 pin (Claude/Poirot/2c18dab-t2139-abi-build-review.md
// Sec6.3): an ODD max_chunk_budget (5, matching tools/t2139_sfreeze_example.cpp's own documented
// prompt -- "The capital of France is" tokenizes to 5 real tokens) is exactly the case that
// landed sslm_decode_step's own wide_logits/rms_wide int64_t* regions on a 4-mod-8 address
// pre-fix (4*max_chunk_budget's own residue mod 8 is 4 when max_chunk_budget is odd, 0 when
// even -- every other caller in this tree happened to use an even budget, so nothing asserted
// against it). Full prefill+decode under alignment-checked access: src/sslm_abi.cpp's own
// compiled-in `assert(reinterpret_cast<uintptr_t>(...) % alignof(int64_t) == 0)` at the exact
// point of use fires immediately if the fix regresses -- this smoke's own job is exercising that
// exact call shape for real, not re-deriving the check itself.
//
// Usage: t2139_n2_odd_budget_smoke.exe <path-to-real.sslm>
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
	sslm_status st = sslm_model_map(bytes.data(), bytes.size(), &model);
	if (st != SSLM_OK || !model) {
		std::fprintf(stderr, "FAIL: sslm_model_map returned %d\n", static_cast<int>(st));
		return 1;
	}

	const uint32_t block_count = 1;
	const size_t block_bytes = sslm_kv_block_size(model);
	const size_t overhead = sslm_kv_pool_overhead_size(model, block_count);
	const size_t pool_buf_size = block_bytes * block_count + overhead;
	std::vector<uint8_t> pool_storage(pool_buf_size + 63);
	void* pool_aligned = pool_storage.data();
	size_t pool_space = pool_storage.size();
	std::align(64, pool_buf_size, pool_aligned, pool_space);
	sslm_kv_pool pool = nullptr;
	st = sslm_kv_pool_create(model, pool_aligned, pool_buf_size, block_count, &pool);
	if (st != SSLM_OK || !pool) {
		std::fprintf(stderr, "FAIL: sslm_kv_pool_create returned %d\n", static_cast<int>(st));
		sslm_model_unmap(model);
		return 1;
	}

	// THE ODD BUDGET -- 5, the exact value N2's own finding traced to the real S-FREEZE prompt.
	const int32_t kOddChunkBudget = 5;
	sslm_config config{};
	config.max_batch = 1;
	config.max_chunk_budget = kOddChunkBudget;
	config.max_layer_budget = 1;  // real value substituted below once the model's own
	                              // num_hidden_layers is known -- placeholder to pass
	                              // ConfigDomainOk's own [1, num_hidden_layers] check for now.

	// sslm_workspace_size needs a real, valid max_layer_budget too (design Sec7.1) -- read the
	// model's own num_hidden_layers via a throwaway prefill-shaped probe is unnecessary; this
	// smoke only needs the value to be in-domain, and the sfreeze example's own precedent
	// (kNumHiddenLayers = 28 for Qwen2.5-1.5B, a real, named gap this ABI does not close) is the
	// established convention this tree already uses for exactly this reason.
	config.max_layer_budget = 28;

	const size_t ws_bytes = sslm_workspace_size(model, &config);
	if (ws_bytes == 0) {
		std::fprintf(stderr, "FAIL: sslm_workspace_size returned 0 for an odd max_chunk_budget\n");
		sslm_kv_pool_destroy(pool);
		sslm_model_unmap(model);
		return 1;
	}
	std::vector<uint8_t> ws_storage(ws_bytes + 63);
	void* ws_aligned = ws_storage.data();
	size_t ws_space = ws_storage.size();
	std::align(64, ws_bytes, ws_aligned, ws_space);
	sslm_workspace ws = nullptr;
	st = sslm_workspace_create(model, &config, ws_aligned, ws_bytes, &ws);
	if (st != SSLM_OK || !ws) {
		std::fprintf(stderr, "FAIL: sslm_workspace_create returned %d\n", static_cast<int>(st));
		sslm_kv_pool_destroy(pool);
		sslm_model_unmap(model);
		return 1;
	}
	std::printf("odd max_chunk_budget=%d: sslm_workspace_size=%zu, sslm_workspace_create: PASS\n",
	            kOddChunkBudget, ws_bytes);

	sslm_seq seq = nullptr;
	st = sslm_seq_create(model, &pool, &seq);
	if (st != SSLM_OK || !seq) {
		std::fprintf(stderr, "FAIL: sslm_seq_create returned %d\n", static_cast<int>(st));
		sslm_workspace_destroy(ws);
		sslm_kv_pool_destroy(pool);
		sslm_model_unmap(model);
		return 1;
	}

	// Full prefill, exactly kOddChunkBudget real tokens in one call (count == chunk_budget, the
	// S-FREEZE example's own shape), through the odd-budget workspace.
	int32_t tokens[kOddChunkBudget] = {0, 1, 2, 3, 4};
	int32_t consumed = 0;
	st = sslm_prefill(model, seq, tokens, kOddChunkBudget, kOddChunkBudget, SSLM_SPAN_PROMPT, ws,
	                   &consumed);
	if (st != SSLM_OK || consumed != kOddChunkBudget) {
		std::fprintf(stderr, "FAIL: sslm_prefill returned %d, consumed=%d\n", static_cast<int>(st),
		             consumed);
		sslm_seq_release(seq);
		sslm_workspace_destroy(ws);
		sslm_kv_pool_destroy(pool);
		sslm_model_unmap(model);
		return 1;
	}
	std::printf("full prefill (%d tokens, odd chunk_budget): PASS\n", kOddChunkBudget);

	// Full decode -- the ready_for_logits path first (the ABI-layer-owned final_norm/rms_wide
	// carve N3's own sibling round fixed), then a real, complete decode_step through
	// wide_logits/logit_row at the SAME odd-budget offsets.
	sslm_decode_params params{};
	params.layer_budget = 28;
	int32_t out_token = 0;
	sslm_seq batch[1] = {seq};
	st = sslm_decode_step(model, batch, 1, &params, ws, &out_token);
	if (st != SSLM_OK) {
		std::fprintf(stderr, "FAIL: sslm_decode_step returned %d\n", static_cast<int>(st));
		sslm_seq_release(seq);
		sslm_workspace_destroy(ws);
		sslm_kv_pool_destroy(pool);
		sslm_model_unmap(model);
		return 1;
	}
	std::printf("full decode_step (odd chunk_budget workspace, real wide_logits/rms_wide carve): "
	            "PASS (out_token=%d)\n",
	            out_token);

	sslm_seq_release(seq);
	sslm_workspace_destroy(ws);
	sslm_kv_pool_destroy(pool);
	sslm_model_unmap(model);
	std::printf("t2139_n2_odd_budget_smoke: PASS\n");
	return 0;
}

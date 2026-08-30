// t2425_trackbd_harness.cpp -- disposable spike harness (T-2425). Loads a real .sslm artifact
// through the real ABI (sslm_model_map, matching T-2423's own model-map harness precedent),
// then drives a real prefill and Track D's new sslm_seq_get_hidden_state verb against it --
// exercising Track B's new QK-norm call site (built into RunLayerLoopChunkBatched, the path
// sslm_prefill actually calls) as a side effect of the forward pass, since this artifact's own
// WGT1 manifest carries q_norm.gain/k_norm.gain tensors. Not part of the build.bat/CMake build
// graph; disposable, per this ticket's own brief.
//
// Usage: t2425_trackbd_harness <artifact.sslm>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include "superslm/model.h"
#include "superslm/sslm_abi.h"

int main(int argc, char** argv) {
	if (argc < 2) {
		std::fprintf(stderr, "usage: t2425_trackbd_harness <artifact.sslm>\n");
		return 2;
	}
	const char* path = argv[1];

	std::ifstream f(path, std::ios::binary);
	if (!f) {
		std::fprintf(stderr, "could not open %s\n", path);
		return 2;
	}
	std::string bytes((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
	std::fprintf(stdout, "artifact: %s (%zu bytes)\n", path, bytes.size());

	// Second, independent, read-only view purely for hidden_size/num_hidden_layers/vocab_size --
	// not exposed by the opaque CPU ABI's own public surface (matching
	// tools/t2132_diag_layer_bisect_cpu.cpp's own precedent for this exact need).
	superslm::SslmModelView view;
	std::string load_err;
	if (superslm::SslmModel::Load(reinterpret_cast<const uint8_t*>(bytes.data()), bytes.size(),
	                               view, &load_err) != superslm::SslmModelStatus::Ok) {
		std::fprintf(stdout, "SslmModel::Load (probe) REJECTED: %s\n", load_err.c_str());
		return 1;
	}
	const size_t hidden_size = view.config.hidden_size;
	const uint32_t num_hidden_layers = view.config.num_hidden_layers;
	std::fprintf(stdout, "config: hidden_size=%zu num_hidden_layers=%u num_attention_heads=%u "
	                     "num_key_value_heads=%u head_dim=%u vocab_size=%u\n",
	             hidden_size, num_hidden_layers, view.config.num_attention_heads,
	             view.config.num_key_value_heads, view.config.head_dim, view.config.vocab_size);

	sslm_model model = nullptr;
	sslm_status st = sslm_model_map(bytes.data(), bytes.size(), &model);
	std::fprintf(stdout, "sslm_model_map status = %d\n", static_cast<int>(st));
	if (st != SSLM_OK) {
		std::fprintf(stdout, "sslm_model_map: REJECTED (status=%d) -- MarshalLayer did not "
		                     "succeed for every layer\n", static_cast<int>(st));
		return 1;
	}
	std::fprintf(stdout, "sslm_model_map: OK -- model handle mapped (MarshalLayer succeeded for "
	                     "every layer, including q_norm/k_norm marshal if the artifact carries "
	                     "those tensors)\n");

	// A short, arbitrary in-range prompt -- correctness of the TOKENS is irrelevant to this
	// spike's own exit condition (execution, not numeric agreement, per this ticket's own §5
	// boundary); five small ids are safely within [0, vocab_size).
	const std::vector<int32_t> prompt_tokens = {1, 2, 3, 4, 5};
	const int32_t token_count = static_cast<int32_t>(prompt_tokens.size());

	const uint32_t block_count = 1;
	const size_t kv_block_bytes = sslm_kv_block_size(model);
	const size_t kv_overhead_bytes = sslm_kv_pool_overhead_size(model, block_count);
	const size_t kv_required = kv_block_bytes * block_count + kv_overhead_bytes;
	std::vector<uint8_t> pool_buf_raw(kv_required + (SSLM_ABI_ALIGNMENT_BYTES - 1));
	void* pool_buf_aligned = pool_buf_raw.data();
	size_t pool_buf_space = pool_buf_raw.size();
	std::align(SSLM_ABI_ALIGNMENT_BYTES, kv_required, pool_buf_aligned, pool_buf_space);
	sslm_kv_pool pool = nullptr;
	st = sslm_kv_pool_create(model, pool_buf_aligned, kv_required, block_count, &pool);
	std::fprintf(stdout, "sslm_kv_pool_create status = %d\n", static_cast<int>(st));
	if (st != SSLM_OK || !pool) {
		sslm_model_unmap(model);
		return 1;
	}

	sslm_config config{};
	config.max_batch = 1;
	config.max_chunk_budget = token_count;
	config.max_layer_budget = static_cast<int32_t>(num_hidden_layers);
	const size_t ws_bytes = sslm_workspace_size(model, &config);
	std::vector<uint8_t> ws_buf_raw(ws_bytes + (SSLM_ABI_ALIGNMENT_BYTES - 1));
	void* ws_aligned = ws_buf_raw.data();
	size_t ws_space = ws_buf_raw.size();
	std::align(SSLM_ABI_ALIGNMENT_BYTES, ws_bytes, ws_aligned, ws_space);
	sslm_workspace ws = nullptr;
	st = sslm_workspace_create(model, &config, ws_aligned, ws_bytes, &ws);
	std::fprintf(stdout, "sslm_workspace_create status = %d\n", static_cast<int>(st));
	if (st != SSLM_OK || !ws) {
		sslm_kv_pool_destroy(pool);
		sslm_model_unmap(model);
		return 1;
	}

	sslm_seq seq = nullptr;
	st = sslm_seq_create(model, &pool, &seq);
	std::fprintf(stdout, "sslm_seq_create status = %d\n", static_cast<int>(st));
	if (st != SSLM_OK || !seq) {
		sslm_workspace_destroy(ws);
		sslm_kv_pool_destroy(pool);
		sslm_model_unmap(model);
		return 1;
	}

	// Track D negative path, tried FIRST, on a fresh never-prefilled sequence: the precondition
	// (design §5) must reject before anything is prefilled -- SSLM_SEQUENCE_NOT_READY, a defined
	// rejection, per this ticket's own instruction that a negative path is as much a finding as a
	// positive one.
	{
		std::vector<int8_t> codes(hidden_size, 0);
		sslm_carried_scale scale{};
		const sslm_status pre_st = sslm_seq_get_hidden_state(model, seq, codes.data(), &scale);
		std::fprintf(stdout, "sslm_seq_get_hidden_state (BEFORE prefill) status = %d "
		                     "(expect SSLM_SEQUENCE_NOT_READY=%d)\n",
		             static_cast<int>(pre_st), static_cast<int>(SSLM_SEQUENCE_NOT_READY));
	}

	int32_t consumed = 0;
	st = sslm_prefill(model, seq, prompt_tokens.data(), token_count, token_count,
	                  SSLM_SPAN_PROMPT, ws, &consumed);
	std::fprintf(stdout, "sslm_prefill status = %d consumed = %d (of %d)\n", static_cast<int>(st),
	             consumed, token_count);
	if (st != SSLM_OK) {
		sslm_seq_release(seq);
		sslm_workspace_destroy(ws);
		sslm_kv_pool_destroy(pool);
		sslm_model_unmap(model);
		return 1;
	}

	// Track D positive path: the real point of this harness. Extract the just-completed
	// prefill's own final hidden state.
	std::vector<int8_t> codes(hidden_size, 0);
	sslm_carried_scale scale{};
	st = sslm_seq_get_hidden_state(model, seq, codes.data(), &scale);
	std::fprintf(stdout, "sslm_seq_get_hidden_state (AFTER prefill) status = %d\n",
	             static_cast<int>(st));
	if (st == SSLM_OK) {
		std::fprintf(stdout, "hidden_state: scale=(m=%lld, e=%lld) first_8_codes=[",
		             static_cast<long long>(scale.m), static_cast<long long>(scale.e));
		for (size_t i = 0; i < 8 && i < hidden_size; ++i) {
			std::fprintf(stdout, "%d%s", static_cast<int>(codes[i]), i + 1 < 8 ? "," : "");
		}
		int64_t sumsq = 0;
		for (size_t i = 0; i < hidden_size; ++i) sumsq += static_cast<int64_t>(codes[i]) * codes[i];
		std::fprintf(stdout, "] sum_of_squares(all %zu codes)=%lld\n", hidden_size,
		             static_cast<long long>(sumsq));

		// Non-mutation check (design §5 postcondition: "the sequence's own state is unmodified
		// by the call"): a second call, same seq, must return byte-identical codes/scale -- if
		// the first call had consumed or perturbed state, the second would diverge or reject.
		std::vector<int8_t> codes2(hidden_size, 0);
		sslm_carried_scale scale2{};
		const sslm_status st2 = sslm_seq_get_hidden_state(model, seq, codes2.data(), &scale2);
		const bool codes_match = codes == codes2;
		const bool scale_match = scale2.m == scale.m && scale2.e == scale.e;
		std::fprintf(stdout, "sslm_seq_get_hidden_state (SECOND call, same seq) status = %d "
		                     "codes_match=%d scale_match=%d\n",
		             static_cast<int>(st2), static_cast<int>(codes_match),
		             static_cast<int>(scale_match));
	}

	sslm_seq_release(seq);
	sslm_workspace_destroy(ws);
	sslm_kv_pool_destroy(pool);
	sslm_model_unmap(model);
	return st == SSLM_OK ? 0 : 1;
}

// t2132_g5_gpu_parity_cpu.cpp -- G5-5 (T-2132, Brunel): the CPU-side driver for the GPU parity
// tool's two gates. See t2132_g5_gpu_parity_shared.h for why this logic is split into its own
// TU (the sslm_status/SslmGpuStatus global-enum collision). Includes ONLY sslm_abi.h -- never
// gpu_1p0.h -- so it compiles as an ordinary CPU-ABI consumer, byte-for-byte the same
// construction t2132_g5_smoke.cpp already uses for G5-2's own real-workload demonstration.
#include <memory>

#include "superslm/sslm_abi.h"
#include "t2132_g5_gpu_parity_shared.h"

namespace {

void Check(G5ParityPathResult& r, bool cond, const char* msg) {
	++r.checks;
	if (!cond) {
		++r.failures;
		if (r.last_error.empty()) r.last_error = msg;
	}
}

}  // namespace

G5ParityPathResult RunCpuGates(const uint8_t* bytes, size_t byte_count, const std::string& prompt,
                                const std::string& schema_name, int32_t num_decode_steps,
                                int32_t forced_chain_len, std::vector<int32_t>* out_prompt_tokens) {
	G5ParityPathResult r;

	sslm_model model = nullptr;
	sslm_status st = sslm_model_map(bytes, byte_count, &model);
	Check(r, st == SSLM_OK && model != nullptr, "sslm_model_map failed");
	if (!model) return r;

	sslm_schema schema = nullptr;
	st = sslm_schema_lookup(model, schema_name.c_str(), &schema);
	Check(r, st == SSLM_OK && schema != nullptr, "sslm_schema_lookup failed");
	if (!schema) {
		sslm_model_unmap(model);
		return r;
	}

	int32_t token_count = 0;
	st = sslm_tokenize(model, prompt.c_str(), nullptr, &token_count);
	Check(r, st == SSLM_BUFFER_TOO_SMALL && token_count > 0, "sslm_tokenize size query failed");
	std::vector<int32_t> prompt_tokens(static_cast<size_t>(token_count > 0 ? token_count : 0));
	if (token_count > 0) {
		int32_t n = token_count;
		st = sslm_tokenize(model, prompt.c_str(), prompt_tokens.data(), &n);
		Check(r, st == SSLM_OK, "sslm_tokenize failed");
	}
	if (out_prompt_tokens) *out_prompt_tokens = prompt_tokens;

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
	Check(r, st == SSLM_OK && pool != nullptr, "sslm_kv_pool_create failed");

	const int32_t kNumHiddenLayers = 28;
	sslm_config config{};
	config.max_batch = 1;
	config.max_chunk_budget = token_count > 0 ? token_count : 1;
	config.max_layer_budget = kNumHiddenLayers;
	const size_t ws_bytes = sslm_workspace_size(model, &config);
	std::vector<uint8_t> ws_buf_raw(ws_bytes + (SSLM_ABI_ALIGNMENT_BYTES - 1));
	void* ws_aligned = ws_buf_raw.data();
	size_t ws_space = ws_buf_raw.size();
	std::align(SSLM_ABI_ALIGNMENT_BYTES, ws_bytes, ws_aligned, ws_space);
	sslm_workspace ws = nullptr;
	st = sslm_workspace_create(model, &config, ws_aligned, ws_bytes, &ws);
	Check(r, st == SSLM_OK && ws != nullptr, "sslm_workspace_create failed");

	if (!pool || !ws) {
		if (ws) sslm_workspace_destroy(ws);
		if (pool) sslm_kv_pool_destroy(pool);
		sslm_model_unmap(model);
		return r;
	}

	// ---- Gate 1: prompt-prefill + num_decode_steps ordinary masked decode calls. ----
	sslm_seq seq = nullptr;
	st = sslm_seq_create(model, &pool, &seq);
	Check(r, st == SSLM_OK && seq != nullptr, "sslm_seq_create (Gate 1) failed");
	if (seq) {
		st = sslm_seq_set_schema(seq, schema);
		Check(r, st == SSLM_OK, "sslm_seq_set_schema (Gate 1) failed");
		int32_t consumed = 0;
		st = sslm_prefill(model, seq, prompt_tokens.data(), token_count, token_count > 0 ? token_count : 1,
		                   SSLM_SPAN_PROMPT, ws, &consumed);
		Check(r, st == SSLM_OK && consumed == token_count, "sslm_prefill (Gate 1, prompt) failed");

		sslm_decode_params params{};
		params.layer_budget = kNumHiddenLayers;
		for (int32_t step = 0; step < num_decode_steps; ++step) {
			int32_t out_token = -1;
			sslm_seq seqs[1] = {seq};
			int32_t guard = 0;
			while (out_token < 0) {
				st = sslm_decode_step(model, seqs, 1, &params, ws, &out_token);
				if (st != SSLM_OK || ++guard > 10000) break;
			}
			Check(r, st == SSLM_OK, "sslm_decode_step (Gate 1) failed");
			if (st != SSLM_OK) break;
			r.decoded_tokens.push_back(out_token);
		}
		sslm_seq_release(seq);
	}

	// ---- Gate 2: jump-forward. The forced chain is DERIVED from Gate 1's own real decode
	// output above (r.decoded_tokens[0..forced_chain_len)) -- every one of those tokens is a
	// real schema-legal continuation by construction (it passed the mask at the step it was
	// produced), so this IS a real forced chain, not a fixture. ----
	if (static_cast<int32_t>(r.decoded_tokens.size()) >= forced_chain_len && forced_chain_len > 0) {
		std::vector<int32_t> forced_chain(r.decoded_tokens.begin(),
		                                  r.decoded_tokens.begin() + forced_chain_len);
		sslm_seq seq2 = nullptr;
		st = sslm_seq_create(model, &pool, &seq2);
		Check(r, st == SSLM_OK && seq2 != nullptr, "sslm_seq_create (Gate 2) failed");
		if (seq2) {
			st = sslm_seq_set_schema(seq2, schema);
			Check(r, st == SSLM_OK, "sslm_seq_set_schema (Gate 2) failed");
			int32_t consumed2 = 0;
			st = sslm_prefill(model, seq2, prompt_tokens.data(), token_count,
			                   token_count > 0 ? token_count : 1, SSLM_SPAN_PROMPT, ws, &consumed2);
			Check(r, st == SSLM_OK && consumed2 == token_count, "sslm_prefill (Gate 2, prompt) failed");

			int32_t forced_consumed = 0;
			st = sslm_prefill(model, seq2, forced_chain.data(), forced_chain_len, forced_chain_len,
			                   SSLM_SPAN_SCHEMA_CONTENT, ws, &forced_consumed);
			Check(r, st == SSLM_OK && forced_consumed == forced_chain_len,
			      "sslm_prefill (Gate 2, forced chain) failed or short-consumed");
			r.forced_consumed = forced_consumed;

			sslm_decode_params params{};
			params.layer_budget = kNumHiddenLayers;
			int32_t next_token = -1;
			sslm_seq seqs2[1] = {seq2};
			int32_t guard = 0;
			while (next_token < 0) {
				st = sslm_decode_step(model, seqs2, 1, &params, ws, &next_token);
				if (st != SSLM_OK || ++guard > 10000) break;
			}
			Check(r, st == SSLM_OK, "sslm_decode_step (Gate 2, post-forced-chain) failed");
			r.post_forced_token = next_token;
			sslm_seq_release(seq2);
		}
	} else {
		Check(r, false, "Gate 1 produced fewer tokens than forced_chain_len -- cannot derive Gate 2's chain");
	}

	sslm_workspace_destroy(ws);
	sslm_kv_pool_destroy(pool);
	sslm_model_unmap(model);

	r.setup_ok = true;
	return r;
}

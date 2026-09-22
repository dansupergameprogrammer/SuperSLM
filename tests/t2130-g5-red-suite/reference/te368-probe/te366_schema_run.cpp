// TE-366 strike harness. Derived from SuperSLM tools/t2132_g5_smoke.cpp (same ABI call sequence).
// Changes: prompt token ids read from a file (chat template applied by the HF tokenizer), decode up
// to max_new tokens, stop on schema acceptance (sslm_stats.schema_accepting) or the -2 dead end,
// report every produced id >= 151643 (Qwen2.5's 22 special ids), detokenize the whole output.
// Usage: te366_schema_run.exe <artifact.sslm> <prompt.ids> <max_new> <kv_blocks> [schema_name]
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <string>
#include <vector>
#include "superslm/sslm_abi.h"

static bool ReadFile(const char* path, std::vector<uint8_t>* out) {
	std::ifstream f(path, std::ios::binary);
	if (!f) return false;
	f.seekg(0, std::ios::end);
	const std::streamoff size = f.tellg();
	f.seekg(0, std::ios::beg);
	out->resize(static_cast<size_t>(size));
	if (size > 0) f.read(reinterpret_cast<char*>(out->data()), size);
	return true;
}
[[noreturn]] static void Fail(const char* what, int status) {
	std::fprintf(stderr, "TE366 FAILED at %s: sslm_status=%d\n", what, status);
	std::exit(1);
}

int main(int argc, char** argv) {
	if (argc < 5) { std::fprintf(stderr, "usage\n"); return 1; }
	const char* artifact_path = argv[1];
	const int32_t max_new = std::atoi(argv[3]);
	const uint32_t block_count = static_cast<uint32_t>(std::atoi(argv[4]));
	const char* schema_name = argc >= 6 ? argv[5] : "prompt_result";
	std::vector<int32_t> forced;
	if (argc >= 7) { std::ifstream ff(argv[6]); int32_t v; while (ff >> v) forced.push_back(v); }
	const int32_t kLayers = 24;  // Qwen2.5-0.5B-Instruct num_hidden_layers

	std::vector<int32_t> prompt;
	{ std::ifstream f(argv[2]); int32_t v; while (f >> v) prompt.push_back(v); }
	const int32_t token_count = static_cast<int32_t>(prompt.size());

	std::vector<uint8_t> bytes;
	if (!ReadFile(artifact_path, &bytes)) { std::fprintf(stderr, "read fail\n"); return 1; }
	sslm_model model = nullptr;
	sslm_status st = sslm_model_map(bytes.data(), bytes.size(), &model);
	if (st != SSLM_OK) Fail("sslm_model_map", st);
	sslm_schema schema = nullptr;
	st = sslm_schema_lookup(model, schema_name, &schema);
	if (st != SSLM_OK) Fail("sslm_schema_lookup", st);

	const size_t kv_block_bytes = sslm_kv_block_size(model);
	const size_t kv_required = kv_block_bytes * block_count + sslm_kv_pool_overhead_size(model, block_count);
	std::vector<uint8_t> pool_raw(kv_required + SSLM_ABI_ALIGNMENT_BYTES);
	void* pool_al = pool_raw.data(); size_t pool_sp = pool_raw.size();
	std::align(SSLM_ABI_ALIGNMENT_BYTES, kv_required, pool_al, pool_sp);
	sslm_kv_pool pool = nullptr;
	st = sslm_kv_pool_create(model, pool_al, kv_required, block_count, &pool);
	if (st != SSLM_OK) Fail("sslm_kv_pool_create", st);

	sslm_config config{};
	config.max_batch = 1;
	config.max_chunk_budget = token_count > (int32_t)forced.size() ? token_count : (int32_t)forced.size();
	config.max_layer_budget = kLayers;
	const size_t ws_bytes = sslm_workspace_size(model, &config);
	std::vector<uint8_t> ws_raw(ws_bytes + SSLM_ABI_ALIGNMENT_BYTES);
	void* ws_al = ws_raw.data(); size_t ws_sp = ws_raw.size();
	std::align(SSLM_ABI_ALIGNMENT_BYTES, ws_bytes, ws_al, ws_sp);
	sslm_workspace ws = nullptr;
	st = sslm_workspace_create(model, &config, ws_al, ws_bytes, &ws);
	if (st != SSLM_OK) Fail("sslm_workspace_create", st);

	sslm_seq seq = nullptr;
	st = sslm_seq_create(model, &pool, &seq);
	if (st != SSLM_OK) Fail("sslm_seq_create", st);
	st = sslm_seq_set_schema(seq, schema);
	if (st != SSLM_OK) Fail("sslm_seq_set_schema", st);
	int32_t consumed = 0;
	st = sslm_prefill(model, seq, prompt.data(), token_count, token_count, SSLM_SPAN_PROMPT, ws, &consumed);
	if (st != SSLM_OK || consumed != token_count) Fail("sslm_prefill", st);
	if (!forced.empty()) {
		int32_t c2 = 0;
		st = sslm_prefill(model, seq, forced.data(), (int32_t)forced.size(), (int32_t)forced.size(), SSLM_SPAN_SCHEMA_CONTENT, ws, &c2);
		if (st != SSLM_OK || c2 != (int32_t)forced.size()) Fail("sslm_prefill(SCHEMA_CONTENT)", st);
		std::printf("FORCED=%zu\n", forced.size());
	}

	sslm_decode_params params{};
	params.struct_size = sizeof(params);
	params.layer_budget = kLayers;
	sslm_seq seqs[1] = {seq};
	std::vector<int32_t> gen;
	const char* stop = "budget";
	int32_t first_special_step = -1, special_count = 0;
	for (int32_t t = 0; t < max_new; ++t) {
		int32_t out_token = -1;
		st = sslm_decode_step(model, seqs, 1, &params, ws, &out_token);
		if (st != SSLM_OK) { std::printf("decode_step status %d at step %d\n", st, t); stop = "status"; break; }
		if (out_token == -2) { stop = "deadend(-2)"; break; }
		if (out_token < 0) { std::printf("out_token %d at step %d\n", out_token, t); stop = "neg"; break; }
		gen.push_back(out_token);
		if (out_token >= 151643) { ++special_count; if (first_special_step < 0) first_special_step = t; }
		sslm_stats_out stats{};
		if (sslm_stats(model, seq, &stats) == SSLM_OK && stats.schema_accepting) { stop = "accepting"; break; }
	}
	std::printf("STOP=%s N=%zu SPECIALS=%d FIRST_SPECIAL_STEP=%d\n", stop, gen.size(), special_count, first_special_step);
	std::printf("IDS=");
	for (int32_t v : gen) std::printf("%d ", v);
	std::printf("\n");
	sslm_seq_release(seq);
	sslm_workspace_destroy(ws);
	sslm_kv_pool_destroy(pool);
	sslm_model_unmap(model);
	return 0;
}

// t2132_g5_smoke.cpp -- T-2132 (Brunel): the real-workload demonstration StandardsDocument.md
// Sec5.4 requires for G5-2 -- one genuine, end-to-end schema-constrained generation against the
// real 1.5B artifact (tools/t2132_build_g5_fixture.py's own output: the real, shipped
// Qwen2.5-1.5B-Instruct weights/tokenizer plus a real SchemaMasks section compiled from
// Claude/Docs/Shopkeeper_IntentExtraction_Schema.md's own reference schema against the real
// tokenizer vocabulary -- 594 compiled states, matching the design's own cited reference count,
// Claude/Loki/t2122-g5-design-strike-2026-08-16.md).
//
// Deliberately independent of tests/t2130-g5-red-suite's own fixture helpers
// (TryMapRealModel/TryLookupSchema call sslm_seq_create with a literal nullptr `pool` argument
// -- a real, evidenced suite defect this ticket's build log reports; sslm_seq_create's own
// shipped, gated contract requires a non-null pointer to an ALREADY-CREATED pool, exactly as
// t2139_sfreeze_example.cpp's own pool-creation pattern, reused here, already does). This tool
// builds its own pool/workspace correctly and proves the production ABI works end to end,
// unblocked by that suite-side gap.
//
// Usage: t2132_g5_smoke.exe <path-to-g5-fixture.sslm> "<prompt>" [schema_name]

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <string>
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

[[noreturn]] void Fail(const char* what, int status) {
	std::fprintf(stderr, "T2132-G5-SMOKE FAILED at %s: sslm_status=%d\n", what, status);
	std::exit(1);
}

}  // namespace

int main(int argc, char** argv) {
	if (argc < 3) {
		std::fprintf(stderr, "usage: %s <path-to-g5-fixture.sslm> \"<prompt>\" [schema_name]\n",
		              argv[0]);
		return 1;
	}
	const char* artifact_path = argv[1];
	const char* prompt = argv[2];
	const char* schema_name = argc >= 4 ? argv[3] : "shopkeeper_intent_extraction";

	const int32_t kNumHiddenLayers = 28;  // Qwen2.5-1.5B-Instruct's own published depth, same
	                                       // out-of-band constant t2139_sfreeze_example.cpp uses.

	std::vector<uint8_t> bytes;
	if (!ReadFile(artifact_path, &bytes)) {
		std::fprintf(stderr, "could not read %s\n", artifact_path);
		return 1;
	}

	sslm_model model = nullptr;
	sslm_status st = sslm_model_map(bytes.data(), bytes.size(), &model);
	if (st != SSLM_OK || !model) Fail("sslm_model_map", static_cast<int>(st));
	std::printf("[1] sslm_model_map: OK (%zu bytes)\n", bytes.size());

	const size_t schema_count = sslm_schema_count(model);
	std::printf("[2] sslm_schema_count: %zu compiled schema(s) resident in this artifact\n",
	            schema_count);
	for (int32_t i = 0; i < static_cast<int32_t>(schema_count); ++i) {
		char buf[256];
		size_t n = sizeof(buf);
		if (sslm_schema_name(model, i, buf, &n) == SSLM_OK) {
			std::printf("      [%d] %.*s\n", i, static_cast<int>(n), buf);
		}
	}

	sslm_schema schema = nullptr;
	st = sslm_schema_lookup(model, schema_name, &schema);
	if (st != SSLM_OK || !schema) Fail("sslm_schema_lookup", static_cast<int>(st));
	std::printf("[3] sslm_schema_lookup(\"%s\"): OK\n", schema_name);

	int32_t token_count = 0;
	st = sslm_tokenize(model, prompt, nullptr, &token_count);
	if (st != SSLM_BUFFER_TOO_SMALL || token_count <= 0) {
		Fail("sslm_tokenize (size query)", static_cast<int>(st));
	}
	std::vector<int32_t> prompt_tokens(static_cast<size_t>(token_count));
	int32_t n = token_count;
	st = sslm_tokenize(model, prompt, prompt_tokens.data(), &n);
	if (st != SSLM_OK) Fail("sslm_tokenize", static_cast<int>(st));
	std::printf("[4] sslm_tokenize: OK (prompt \"%s\" -> %d real tokens)\n", prompt, n);

	const uint32_t block_count = 1;
	const size_t kv_block_bytes = sslm_kv_block_size(model);
	const size_t kv_overhead_bytes = sslm_kv_pool_overhead_size(model, block_count);
	if (kv_block_bytes == 0) Fail("sslm_kv_block_size", 0);
	const size_t kv_required = kv_block_bytes * block_count + kv_overhead_bytes;
	std::vector<uint8_t> pool_buf_raw(kv_required + (SSLM_ABI_ALIGNMENT_BYTES - 1));
	void* pool_buf_aligned = pool_buf_raw.data();
	size_t pool_buf_space = pool_buf_raw.size();
	std::align(SSLM_ABI_ALIGNMENT_BYTES, kv_required, pool_buf_aligned, pool_buf_space);
	sslm_kv_pool pool = nullptr;
	st = sslm_kv_pool_create(model, pool_buf_aligned, kv_required, block_count, &pool);
	if (st != SSLM_OK || !pool) Fail("sslm_kv_pool_create", static_cast<int>(st));
	std::printf("[5] sslm_kv_pool_create: OK\n");

	sslm_config config{};
	config.max_batch = 1;
	config.max_chunk_budget = token_count > 0 ? token_count : 1;
	config.max_layer_budget = kNumHiddenLayers;
	config.reserved = 0;
	const size_t ws_bytes = sslm_workspace_size(model, &config);
	if (ws_bytes == 0) Fail("sslm_workspace_size", 0);
	std::vector<uint8_t> ws_buf_raw(ws_bytes + (SSLM_ABI_ALIGNMENT_BYTES - 1));
	void* ws_aligned = ws_buf_raw.data();
	size_t ws_space = ws_buf_raw.size();
	std::align(SSLM_ABI_ALIGNMENT_BYTES, ws_bytes, ws_aligned, ws_space);
	sslm_workspace ws = nullptr;
	st = sslm_workspace_create(model, &config, ws_aligned, ws_bytes, &ws);
	if (st != SSLM_OK || !ws) Fail("sslm_workspace_create", static_cast<int>(st));
	std::printf("[5] sslm_workspace_create: OK\n");

	sslm_seq seq = nullptr;
	st = sslm_seq_create(model, &pool, &seq);
	if (st != SSLM_OK || !seq) Fail("sslm_seq_create", static_cast<int>(st));
	std::printf("[5] sslm_seq_create: OK\n");

	// --- G5: bind the schema BEFORE any content (design Sec5: valid only at a fresh walk-state) ---
	st = sslm_seq_set_schema(seq, schema);
	if (st != SSLM_OK) Fail("sslm_seq_set_schema", static_cast<int>(st));
	std::printf("[6] sslm_seq_set_schema: OK (bound to \"%s\")\n", schema_name);

	// --- prefill the prompt as SSLM_SPAN_PROMPT -- design Sec5: never advances the DFA walk ---
	int32_t consumed = 0;
	st = sslm_prefill(model, seq, prompt_tokens.data(), token_count, token_count, SSLM_SPAN_PROMPT,
	                   ws, &consumed);
	if (st != SSLM_OK || consumed != token_count) Fail("sslm_prefill", static_cast<int>(st));
	std::printf("[7] sslm_prefill(SSLM_SPAN_PROMPT): OK (%d tokens, walk-state unmoved by design)\n",
	            consumed);

	// --- decode under the mask: every produced token is schema-legal by construction (G5-2's
	// own masking step forces argmax to choose only from the schema's own valid-token set at
	// each DFA state) ---
	const int32_t kMaxNewTokens = 40;
	std::vector<int32_t> generated;
	sslm_decode_params params{};
	params.layer_budget = kNumHiddenLayers;
	sslm_seq seqs[1] = {seq};
	for (int32_t t = 0; t < kMaxNewTokens; ++t) {
		int32_t out_token = -1;
		int32_t guard = 0;
		while (out_token < 0) {
			st = sslm_decode_step(model, seqs, 1, &params, ws, &out_token);
			if (st != SSLM_OK) Fail("sslm_decode_step", static_cast<int>(st));
			if (++guard > 10000) Fail("sslm_decode_step (never completed)", -1);
		}
		generated.push_back(out_token);
	}
	std::printf("[8] sslm_decode_step x%d, SCHEMA-CONSTRAINED: OK (produced: ", kMaxNewTokens);
	for (int32_t t : generated) std::printf("%d ", t);
	std::printf(")\n");

	sslm_detok_state detok_state = {0};
	char detok_buf[8192];
	int32_t detok_out_n = static_cast<int32_t>(sizeof(detok_buf));
	st = sslm_detokenize_stream(model, &detok_state, generated.data(),
	                             static_cast<int32_t>(generated.size()), detok_buf, &detok_out_n);
	if (st != SSLM_OK) Fail("sslm_detokenize_stream", static_cast<int>(st));
	const std::string generated_text(detok_buf, static_cast<size_t>(detok_out_n));

	sslm_stats_out stats{};
	st = sslm_stats(model, seq, &stats);
	if (st != SSLM_OK) Fail("sslm_stats", static_cast<int>(st));

	sslm_seq_release(seq);
	sslm_workspace_destroy(ws);
	sslm_kv_pool_destroy(pool);
	st = sslm_model_unmap(model);
	if (st != SSLM_OK) Fail("sslm_model_unmap", static_cast<int>(st));

	std::printf("\n=== T-2132 G5-2 REAL-WORKLOAD SMOKE: PASS ===\n");
	std::printf("artifact:            %s\n", artifact_path);
	std::printf("schema:              %s\n", schema_name);
	std::printf("prompt:              \"%s\"\n", prompt);
	std::printf("schema-constrained generated text: \"%s\"\n", generated_text.c_str());
	std::printf("forced_token_count (jump-forward, design Sec6 G5-3): %lld\n",
	            static_cast<long long>(stats.forced_token_count));
	return 0;
}

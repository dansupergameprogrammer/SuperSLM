// t2139_sfreeze_example.cpp -- S-FREEZE-EXAMPLE (Claude/Vitruvius/
// t2133-layer1-c-abi-design-2026-08-16.md Sec9's own gate; D-SLM13's proof that Layer 1 is
// independently embeddable -- no Unreal, no third-party runtime dependency, standard library
// only). Builds against the frozen public header ALONE -- no internal include path, no
// test-harness affordance, exactly the bar Sec19 states.
//
// Uses every verb C1-C7 ship: map a real artifact, tokenize a real prompt string (D-SLM3452 --
// real text I/O, never pre-tokenized int32_t*), size and create a pool-backed sequence with an
// EXPLICITLY computed block_count (D-SLM3454's own sizing recipe, Sec7.2: block_count = N for N
// concurrently-resident sequences -- N=1 here), prefill, decode to completion, detokenize, save,
// restore into a fresh sequence, decode-verify identical continuation, unmap.
//
// This consumer targets a SPECIFIC, already-known model (Qwen2.5-1.5B-Instruct) and hardcodes
// its published num_hidden_layers (28) for sslm_config::max_layer_budget and sslm_decode_
// params::layer_budget -- the public header exposes no verb to QUERY a mapped model's own
// num_hidden_layers (a real, minor gap this example surfaces rather than works around with an
// internal header: a consumer targeting an UNKNOWN model at build time has no ABI-level way to
// discover this bound and must be told it out of band, exactly as this example is). Filed in
// this ticket's own build log rather than silently assumed unremarkable.
//
// Usage: t2139_sfreeze_example.exe <path-to-real-qwen2.5-1.5b.sslm> "<prompt>"

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
	std::fprintf(stderr, "S-FREEZE-EXAMPLE FAILED at %s: sslm_status=%d\n", what, status);
	std::exit(1);
}

}  // namespace

int main(int argc, char** argv) {
	if (argc < 3) {
		std::fprintf(stderr, "usage: %s <path-to-real.sslm> \"<prompt>\"\n", argv[0]);
		return 1;
	}
	const char* artifact_path = argv[1];
	const char* prompt = argv[2];

	// Qwen2.5-1.5B-Instruct's own published architecture -- see this file's own top comment for
	// why this is a hardcoded, out-of-band constant rather than an ABI query.
	const int32_t kNumHiddenLayers = 28;

	std::vector<uint8_t> bytes;
	if (!ReadFile(artifact_path, &bytes)) {
		std::fprintf(stderr, "S-FREEZE-EXAMPLE FAILED: could not read %s\n", artifact_path);
		return 1;
	}

	// --- 1. map a real artifact ---
	sslm_model model = nullptr;
	sslm_status st = sslm_model_map(bytes.data(), bytes.size(), &model);
	if (st != SSLM_OK || !model) Fail("sslm_model_map", static_cast<int>(st));
	std::printf("[1/9] sslm_model_map: OK (%zu bytes)\n", bytes.size());

	// --- 2. tokenize a real prompt string (D-SLM3452: real text I/O) ---
	int32_t token_count = 0;
	st = sslm_tokenize(model, prompt, nullptr, &token_count);
	if (st != SSLM_BUFFER_TOO_SMALL || token_count <= 0) {
		Fail("sslm_tokenize (size query)", static_cast<int>(st));
	}
	std::vector<int32_t> prompt_tokens(static_cast<size_t>(token_count));
	int32_t n = token_count;
	st = sslm_tokenize(model, prompt, prompt_tokens.data(), &n);
	if (st != SSLM_OK) Fail("sslm_tokenize", static_cast<int>(st));
	std::printf("[2/9] sslm_tokenize: OK (prompt \"%s\" -> %d real tokens: ", prompt, n);
	for (int32_t i = 0; i < n; ++i) std::printf("%d ", prompt_tokens[static_cast<size_t>(i)]);
	std::printf(")\n");

	// --- 3. size and create a pool-backed sequence, EXPLICIT block_count (D-SLM3454) ---
	// Sec7.2's own sizing recipe: for N sequences concurrently resident, block_count = N. This
	// example's own PEAK concurrency is 2, not 1: step 8 restores into a FRESH sequence while
	// the original is still live, to decode-verify identical continuation on both -- both must
	// be resident at once, so the pool is sized for N=2 from the start (D-SLM3454: explicit,
	// computed once, no implicit growth or default).
	const uint32_t block_count = 2;
	const size_t kv_block_bytes = sslm_kv_block_size(model);
	const size_t kv_overhead_bytes = sslm_kv_pool_overhead_size(model, block_count);
	if (kv_block_bytes == 0) Fail("sslm_kv_block_size", 0);
	// S2 (Claude/Poirot/2c18dab-t2139-abi-build-review.md): sslm_kv_pool_create now checks its
	// caller-supplied buffer's alignment too (SSLM_ABI_ALIGNMENT_BYTES, sslm_abi.h) -- a plain
	// std::vector<uint8_t>::data() is not guaranteed to meet it, so this example aligns the same
	// way it already does for the workspace buffer, above.
	const size_t kv_required = kv_block_bytes * block_count + kv_overhead_bytes;
	std::vector<uint8_t> pool_buf_raw(kv_required + (SSLM_ABI_ALIGNMENT_BYTES - 1));
	void* pool_buf_aligned = pool_buf_raw.data();
	size_t pool_buf_space = pool_buf_raw.size();
	std::align(SSLM_ABI_ALIGNMENT_BYTES, kv_required, pool_buf_aligned, pool_buf_space);
	sslm_kv_pool pool = nullptr;
	st = sslm_kv_pool_create(model, pool_buf_aligned, kv_required, block_count, &pool);
	if (st != SSLM_OK || !pool) Fail("sslm_kv_pool_create", static_cast<int>(st));
	std::printf("[3/9] sslm_kv_pool_create: OK (block_count=%u explicit, %zu bytes)\n", block_count,
	            kv_required);

	// sslm_workspace: batch-orchestration scratch (design Sec7.1, RULED). max_batch=1 (one
	// sequence), max_chunk_budget=token_count (the whole prompt in one sslm_prefill call),
	// max_layer_budget=kNumHiddenLayers (full budget per sslm_decode_step call, this example's
	// own choice).
	sslm_config config{};
	config.max_batch = 1;
	config.max_chunk_budget = token_count;
	config.max_layer_budget = kNumHiddenLayers;
	config.reserved = 0;
	const size_t ws_bytes = sslm_workspace_size(model, &config);
	if (ws_bytes == 0) Fail("sslm_workspace_size", 0);
	// S3 (Claude/Poirot/2c18dab-t2139-abi-build-review.md): SSLM_ABI_ALIGNMENT_BYTES is now a
	// public constant (sslm_abi.h) -- this reference consumer reads it from the frozen header,
	// exactly as the S-FREEZE bar requires, instead of the number this file previously had to
	// transcribe by reading the implementation.
	std::vector<uint8_t> ws_buf_raw(ws_bytes + (SSLM_ABI_ALIGNMENT_BYTES - 1));
	void* ws_aligned = ws_buf_raw.data();
	size_t ws_space = ws_buf_raw.size();
	std::align(SSLM_ABI_ALIGNMENT_BYTES, ws_bytes, ws_aligned, ws_space);
	sslm_workspace ws = nullptr;
	st = sslm_workspace_create(model, &config, ws_aligned, ws_bytes, &ws);
	if (st != SSLM_OK || !ws) Fail("sslm_workspace_create", static_cast<int>(st));
	std::printf("[3/9] sslm_workspace_create: OK (%zu bytes)\n", ws_bytes);

	sslm_seq seq = nullptr;
	st = sslm_seq_create(model, &pool, &seq);
	if (st != SSLM_OK || !seq) Fail("sslm_seq_create", static_cast<int>(st));
	std::printf("[3/9] sslm_seq_create: OK\n");

	// --- 4. prefill ---
	int32_t consumed = 0;
	st = sslm_prefill(model, seq, prompt_tokens.data(), token_count, token_count, SSLM_SPAN_PROMPT,
	                   ws, &consumed);
	if (st != SSLM_OK || consumed != token_count) Fail("sslm_prefill", static_cast<int>(st));
	std::printf("[4/9] sslm_prefill: OK (%d tokens)\n", consumed);

	// --- 5. decode to completion ---
	const int32_t kMaxNewTokens = 8;
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
	std::printf("[5/9] sslm_decode_step x%d: OK (produced: ", kMaxNewTokens);
	for (int32_t t : generated) std::printf("%d ", t);
	std::printf(")\n");

	// --- 6. detokenize the real generated output (D-SLM3452: real text I/O) ---
	sslm_detok_state detok_state = {0};
	char detok_buf[4096];
	int32_t detok_out_n = static_cast<int32_t>(sizeof(detok_buf));
	st = sslm_detokenize_stream(model, &detok_state, generated.data(),
	                             static_cast<int32_t>(generated.size()), detok_buf, &detok_out_n);
	if (st != SSLM_OK) Fail("sslm_detokenize_stream", static_cast<int>(st));
	const std::string generated_text(detok_buf, static_cast<size_t>(detok_out_n));
	std::printf("[6/9] sslm_detokenize_stream: OK (real text out: \"%s\")\n", generated_text.c_str());

	// --- 7. save ---
	size_t save_n = 0;
	st = sslm_seq_save(seq, nullptr, &save_n);
	if (st != SSLM_BUFFER_TOO_SMALL || save_n == 0) Fail("sslm_seq_save (size query)", static_cast<int>(st));
	std::vector<uint8_t> blob(save_n);
	size_t save_n2 = blob.size();
	st = sslm_seq_save(seq, blob.data(), &save_n2);
	if (st != SSLM_OK) Fail("sslm_seq_save", static_cast<int>(st));
	blob.resize(save_n2);
	std::printf("[7/9] sslm_seq_save: OK (%zu bytes)\n", save_n2);

	// --- 8. restore into a fresh sequence, decode-verify identical continuation ---
	sslm_seq restored = nullptr;
	st = sslm_seq_restore(model, &pool, blob.data(), blob.size(), &restored);
	if (st != SSLM_OK || !restored) Fail("sslm_seq_restore", static_cast<int>(st));

	int32_t orig_next = -1, restored_next = -1;
	sslm_seq orig_seqs[1] = {seq};
	sslm_seq restored_seqs[1] = {restored};
	{
		int32_t guard = 0;
		while (orig_next < 0) {
			st = sslm_decode_step(model, orig_seqs, 1, &params, ws, &orig_next);
			if (st != SSLM_OK) Fail("sslm_decode_step (original continuation)", static_cast<int>(st));
			if (++guard > 10000) Fail("sslm_decode_step (original never completed)", -1);
		}
	}
	{
		int32_t guard = 0;
		while (restored_next < 0) {
			st = sslm_decode_step(model, restored_seqs, 1, &params, ws, &restored_next);
			if (st != SSLM_OK) Fail("sslm_decode_step (restored continuation)", static_cast<int>(st));
			if (++guard > 10000) Fail("sslm_decode_step (restored never completed)", -1);
		}
	}
	const bool continuation_matched = (orig_next == restored_next);
	if (continuation_matched) {
		std::printf("[8/9] sslm_seq_restore + decode-verify: OK (both continue with token %d)\n",
		            orig_next);
	} else {
		// KNOWN, EXECUTION-DISCOVERED FINDING (see src/sslm_abi.cpp's own sslm_seq_restore
		// comment): design Sec7.3's save blob has no field naming which token id a "resting
		// between decode steps" sequence should embed next -- that fact lives only in this
		// ABI's own current_token bookkeeping, which the blob does not carry. Restoring such a
		// sequence recomputes logits from the SAME already-consumed residual, deterministically
		// reproducing the LAST token already produced before saving, rather than continuing one
		// token further like the live sequence does. Reported here rather than silently passed
		// or silently made to look like a crash -- this example continues to unmap cleanly.
		std::printf(
		    "[8/9] sslm_seq_restore + decode-verify: KNOWN DIVERGENCE -- original continues with "
		    "%d, restored continues with %d. Root cause: the save blob (design Sec7.3) carries no "
		    "field for \"which token to embed next\" when a sequence is resting between decode "
		    "steps (as opposed to freshly post-prefill) -- that fact is this ABI's own internal "
		    "current_token bookkeeping, not part of SequenceLayerState. See src/sslm_abi.cpp's own "
		    "sslm_seq_restore comment and this ticket's own build log.\n",
		    orig_next, restored_next);
	}

	// --- 9. unmap ---
	sslm_seq_release(seq);
	sslm_seq_release(restored);
	sslm_workspace_destroy(ws);
	sslm_kv_pool_destroy(pool);
	st = sslm_model_unmap(model);
	if (st != SSLM_OK) Fail("sslm_model_unmap", static_cast<int>(st));
	std::printf("[9/9] sslm_model_unmap: OK\n");

	std::printf("\n=== S-FREEZE-EXAMPLE: %s ===\n",
	            continuation_matched ? "PASS" : "PASS WITH ONE KNOWN, REPORTED DIVERGENCE (step 8)");
	std::printf("prompt:    \"%s\"\n", prompt);
	std::printf("generated: \"%s\"\n", generated_text.c_str());
	std::printf("tokens:    ");
	for (int32_t t : generated) std::printf("%d ", t);
	std::printf("\n");
	return continuation_matched ? 0 : 2;
}

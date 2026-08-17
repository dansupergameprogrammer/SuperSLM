// t2147_chunk_batched_pins.cpp -- T-2147 (Brunel), design §15.2's three ruled proof cells,
// against the real 1.5B artifact (D:\SuperSLM-t2116-package\artifacts\, the G5 fixture), plus
// the actual speedup measurement §15's own build owes:
//
//   (a) per-size:        the SAME prompt through sslm_prefill at chunk_budget in
//                         {1, mid, N} -- bit-identical sslm_seq_save blobs (K/V, dfa_walk_state,
//                         forced_token_count, post-prefill hidden state all live inside that
//                         blob, T-2133/M4's own "forced_token_count joins the save-blob" fold).
//                         chunk_budget=1 is the strict-generalization cell (design §15.2).
//   (b) boundary-split:  the SAME N-token prompt split at every k in [1, N-1] (or a
//                         representative sweep) -- two sslm_prefill calls (0:k, k:N) reassemble
//                         to the SAME blob as one sslm_prefill(0:N) call.
//   (c) real-artifact
//       consistency:     this tool built TWICE, once against a WORKING TREE and once against a
//                         baseline commit's PRE-T-2147 sources (see run_pins.ps1), diffs the two
//                         binaries' own blobs for the identical prompt -- the actual "reproduces
//                         the existing per-token RunLayerLoop-driven path bit-for-bit" oracle
//                         (T-2133 §9 C4's own shape), because the OLD PrefillWholeTokensImpl's
//                         per-token loop never depended on chunk_budget for ITS OWN arithmetic
//                         (only for how many tokens one call consumes) -- so the baseline binary
//                         at chunk_budget=N is genuine per-token ground truth.
//
// Speedup: --speedup measures wall-clock ms for one sslm_prefill(chunk_budget=1) call (N calls
// worth of per-token weight streaming, PrefillWholeTokensImpl's own old shape) vs one
// sslm_prefill(chunk_budget=N) call (ONE call, ONE streamed weight pass per layer per
// projection) over the SAME prompt, same artifact -- the number T-2147 exists for.
//
// Usage: t2147_chunk_batched_pins.exe <path-to-1.5b-artifact.sslm> "<prompt>"
//        [--schema=NAME --schema-artifact=PATH] [--speedup] [--boundary-sweep=K]

#include <algorithm>
#include <chrono>
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

int g_checks = 0;
int g_failures = 0;

void Check(bool cond, const char* what) {
	++g_checks;
	if (!cond) {
		++g_failures;
		std::printf("FAIL: %s\n", what);
	} else {
		std::printf("PASS: %s\n", what);
	}
}

[[noreturn]] void Fail(const char* what, int status) {
	std::fprintf(stderr, "T2147-PINS FATAL at %s: sslm_status=%d\n", what, status);
	std::exit(1);
}

constexpr int32_t kNumHiddenLayers = 28;  // Qwen2.5-1.5B-Instruct's own published depth.

// One fresh model-map + pool + workspace + seq, prefills `tokens[0:count)` via ONE OR MORE
// sslm_prefill calls (per `splits`, a list of consecutive sub-span lengths summing to `count` --
// {count} for a single call, {k, count-k} for a boundary split), under `kind`/`schema`, then
// returns the sslm_seq_save blob. `chunk_budget_per_call` is the chunk_budget argument each
// individual sslm_prefill call is issued with -- this is the actual GEMM-batching knob T-2147
// wires (PrefillWholeTokensImpl's own `n = min(count, chunk_budget)`).
std::vector<uint8_t> PrefillAndSave(const std::vector<uint8_t>& bytes, const int32_t* tokens,
                                     int32_t count, const std::vector<int32_t>& splits,
                                     int32_t chunk_budget_per_call, sslm_span_kind kind,
                                     const char* schema_name, double* out_elapsed_ms) {
	sslm_model model = nullptr;
	sslm_status st = sslm_model_map(bytes.data(), bytes.size(), &model);
	if (st != SSLM_OK || !model) Fail("sslm_model_map", static_cast<int>(st));

	sslm_schema schema = nullptr;
	if (schema_name) {
		st = sslm_schema_lookup(model, schema_name, &schema);
		if (st != SSLM_OK || !schema) Fail("sslm_schema_lookup", static_cast<int>(st));
	}

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
	if (st != SSLM_OK || !pool) Fail("sslm_kv_pool_create", static_cast<int>(st));

	sslm_config config{};
	config.max_batch = 1;
	config.max_chunk_budget = count > 0 ? count : 1;  // big enough for every call this run makes
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

	sslm_seq seq = nullptr;
	st = sslm_seq_create(model, &pool, &seq);
	if (st != SSLM_OK || !seq) Fail("sslm_seq_create", static_cast<int>(st));

	if (schema) {
		st = sslm_seq_set_schema(seq, schema);
		if (st != SSLM_OK) Fail("sslm_seq_set_schema", static_cast<int>(st));
	}

	const auto t0 = std::chrono::steady_clock::now();
	int32_t offset = 0;
	for (int32_t span_len : splits) {
		// `chunk_budget_per_call` bounds how many tokens ONE sslm_prefill call admits
		// (PrefillWholeTokensImpl's own `n = min(count, chunk_budget)`) -- driving the WHOLE
		// span at a small chunk_budget means calling sslm_prefill repeatedly, exactly as a real
		// per-token caller would, until this span's own `span_len` tokens are all consumed.
		const int32_t cb = chunk_budget_per_call > 0 ? chunk_budget_per_call : span_len;
		int32_t span_done = 0;
		while (span_done < span_len) {
			int32_t consumed = 0;
			st = sslm_prefill(model, seq, tokens + offset + span_done, span_len - span_done, cb,
			                  kind, ws, &consumed);
			if (st != SSLM_OK || consumed <= 0) Fail("sslm_prefill", static_cast<int>(st));
			span_done += consumed;
		}
		offset += span_len;
	}
	const auto t1 = std::chrono::steady_clock::now();
	if (out_elapsed_ms) {
		*out_elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
	}

	size_t blob_n = 0;
	st = sslm_seq_save(seq, nullptr, &blob_n);
	if (st != SSLM_BUFFER_TOO_SMALL || blob_n == 0) Fail("sslm_seq_save (size query)", static_cast<int>(st));
	std::vector<uint8_t> blob(blob_n);
	size_t blob_n2 = blob_n;
	st = sslm_seq_save(seq, blob.data(), &blob_n2);
	if (st != SSLM_OK) Fail("sslm_seq_save", static_cast<int>(st));
	blob.resize(blob_n2);

	sslm_seq_release(seq);
	sslm_workspace_destroy(ws);
	sslm_kv_pool_destroy(pool);
	sslm_model_unmap(model);
	return blob;
}

}  // namespace

int main(int argc, char** argv) {
	if (argc < 3) {
		std::fprintf(stderr,
		              "usage: %s <path-to-1.5b-artifact.sslm> \"<prompt>\" "
		              "[--speedup] [--boundary-sweep=K]\n",
		              argv[0]);
		return 1;
	}
	const char* artifact_path = argv[1];
	const char* prompt = argv[2];
	bool do_speedup = false;
	int32_t boundary_sweep = -1;  // -1 = every split point
	for (int i = 3; i < argc; ++i) {
		const std::string a = argv[i];
		if (a == "--speedup") do_speedup = true;
		else if (a.rfind("--boundary-sweep=", 0) == 0) boundary_sweep = std::atoi(a.c_str() + 18);
	}

	std::vector<uint8_t> bytes;
	if (!ReadFile(artifact_path, &bytes)) {
		std::fprintf(stderr, "could not read %s\n", artifact_path);
		return 1;
	}

	// Tokenize once via a throwaway map (tokenizer is artifact-resident and stateless).
	sslm_model probe_model = nullptr;
	sslm_status st = sslm_model_map(bytes.data(), bytes.size(), &probe_model);
	if (st != SSLM_OK || !probe_model) Fail("sslm_model_map (tokenize probe)", static_cast<int>(st));
	int32_t token_count = 0;
	st = sslm_tokenize(probe_model, prompt, nullptr, &token_count);
	if (st != SSLM_BUFFER_TOO_SMALL || token_count <= 0) Fail("sslm_tokenize (size)", static_cast<int>(st));
	std::vector<int32_t> tokens(static_cast<size_t>(token_count));
	int32_t n = token_count;
	st = sslm_tokenize(probe_model, prompt, tokens.data(), &n);
	if (st != SSLM_OK) Fail("sslm_tokenize", static_cast<int>(st));
	sslm_model_unmap(probe_model);
	const int32_t N = token_count;
	std::printf("prompt \"%s\" -> %d real tokens\n", prompt, N);

	// --- Cell (a): per-size --------------------------------------------------------------
	std::printf("\n=== Cell (a): per-size (chunk_budget in {1, mid, N}) ===\n");
	const int32_t mid = N > 2 ? N / 2 : N;
	const std::vector<int32_t> one_call = {N};
	const auto blob_cb1 = PrefillAndSave(bytes, tokens.data(), N, one_call, /*chunk_budget=*/1,
	                                      SSLM_SPAN_PROMPT, nullptr, nullptr);
	const auto blob_cbmid = PrefillAndSave(bytes, tokens.data(), N, one_call, /*chunk_budget=*/mid,
	                                        SSLM_SPAN_PROMPT, nullptr, nullptr);
	const auto blob_cbN = PrefillAndSave(bytes, tokens.data(), N, one_call, /*chunk_budget=*/N,
	                                      SSLM_SPAN_PROMPT, nullptr, nullptr);
	Check(blob_cb1 == blob_cbmid, "chunk_budget=1 blob == chunk_budget=mid blob");
	Check(blob_cb1 == blob_cbN, "chunk_budget=1 blob == chunk_budget=N blob (strict generalization)");
	Check(blob_cbmid == blob_cbN, "chunk_budget=mid blob == chunk_budget=N blob");
	std::printf("blob size: %zu bytes\n", blob_cb1.size());

	// --- Cell (b): boundary-split ----------------------------------------------------------
	std::printf("\n=== Cell (b): boundary-split (every k in [1, N-1]%s) ===\n",
	            boundary_sweep > 0 ? ", swept" : "");
	int all_boundary_ok = 1;
	int32_t step = 1;
	if (boundary_sweep > 0 && N - 1 > boundary_sweep) {
		step = (N - 1) / boundary_sweep;
		if (step < 1) step = 1;
	}
	for (int32_t k = 1; k <= N - 1; k += step) {
		const std::vector<int32_t> split = {k, N - k};
		const auto blob_split = PrefillAndSave(bytes, tokens.data(), N, split, /*chunk_budget=*/N,
		                                        SSLM_SPAN_PROMPT, nullptr, nullptr);
		if (blob_split != blob_cbN) {
			all_boundary_ok = 0;
			std::printf("FAIL: split at k=%d does not reassemble to the one-call blob\n", k);
		}
	}
	Check(all_boundary_ok != 0, "every boundary split reassembles bit-identically to the one-call result");

	// --- Speedup measurement -----------------------------------------------------------------
	if (do_speedup) {
		std::printf("\n=== Speedup: chunk_budget=1 (per-token) vs chunk_budget=N (batched) ===\n");
		double ms_per_token = 0, ms_batched = 0;
		PrefillAndSave(bytes, tokens.data(), N, one_call, /*chunk_budget=*/1, SSLM_SPAN_PROMPT,
		               nullptr, &ms_per_token);
		PrefillAndSave(bytes, tokens.data(), N, one_call, /*chunk_budget=*/N, SSLM_SPAN_PROMPT,
		               nullptr, &ms_batched);
		std::printf("prompt_len=%d artifact=%s\n", N, artifact_path);
		std::printf("per-token (chunk_budget=1):  %.3f ms  (%.2f tok/s)\n", ms_per_token,
		            ms_per_token > 0 ? 1000.0 * N / ms_per_token : 0.0);
		std::printf("batched   (chunk_budget=N):  %.3f ms  (%.2f tok/s)\n", ms_batched,
		            ms_batched > 0 ? 1000.0 * N / ms_batched : 0.0);
		std::printf("speedup: %.2fx\n", ms_batched > 0 ? ms_per_token / ms_batched : 0.0);
	}

	std::printf("\n=== T-2147 chunk-batched prefill pins: %s ===\n",
	            g_failures == 0 ? "PASS" : "FAIL");
	std::printf("checks=%d failures=%d\n", g_checks, g_failures);
	return g_failures == 0 ? 0 : 1;
}

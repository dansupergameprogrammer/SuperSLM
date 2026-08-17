// t2139_c4_oracle.cpp -- C4's own gate (Claude/Vitruvius/t2133-layer1-c-abi-design-2026-08-16.md
// Sec9): "single-sequence greedy decode through this ABI reproduces RunGreedyDecodeLoop's own
// direct-call output bit-for-bit." Drives the SAME real prompt through two independent paths
// against the SAME real artifact:
//
//   (A) this ABI: sslm_model_map -> sslm_kv_pool_create(1) -> sslm_seq_create ->
//       sslm_prefill(whole prompt) -> sslm_decode_step (repeated, one token per call, full
//       layer_budget) -> N produced tokens, plus the sequence's own final KV block bytes.
//   (B) the direct engine: tests/t2138-abi-red-suite/fixture_common.h's own CpuOracleModel +
//       RunGreedyOracle (a thin, already-reviewed wrapper around superslm::RunGreedyDecodeLoop),
//       reused verbatim rather than re-derived (StandardsDocument Sec6.6) -- the SAME construction
//       T-2112's own sibling GPU-ABI gate and this suite's own dim6/dim10 cells already use for
//       the identical "this ABI's output bit-equals the direct-engine call" oracle.
//
// Compares: the produced token sequence (must be identical, element for element) and the
// sequence's own final KV block bytes against the oracle's own final workspace bytes (must be
// byte-for-byte identical over the region both paths actually wrote -- oracle's own KV region is
// exactly this ABI's own sslm_kv_block_size(model) bytes, RunGreedyOracle's own kv_bytes formula
// matching it at int8 kv_precision, the only decodable precision).
//
// Usage: t2139_c4_oracle.exe <path-to-real.sslm> [max_new_tokens]
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <vector>

#include "sslm_abi.h"
#include "fixture_common.h"

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
		std::fprintf(stderr, "usage: %s <path-to-real.sslm> [max_new_tokens] [layer_budget]\n",
		             argv[0]);
		return 1;
	}
	// layer_budget < num_hidden_layers exercises the mid-token PENDING (-1) resumption path --
	// sslm_decode_step called multiple times per token, each advancing layer_index by at most
	// layer_budget, until a call completes the token. Same oracle comparison either way: the
	// FINAL produced token sequence must still be bit-identical regardless of how many
	// bounded calls it took to get there (StandardsDocument Sec5.4 -- executed, not reasoned).
	const int32_t forced_layer_budget = argc >= 4 ? std::atoi(argv[3]) : 0;
	const size_t max_new_tokens = argc >= 3 ? static_cast<size_t>(std::atoi(argv[2])) : 8;

	std::vector<uint8_t> bytes;
	if (!ReadFile(argv[1], &bytes)) {
		std::fprintf(stderr, "FAIL: could not read %s\n", argv[1]);
		return 1;
	}

	// --- (B) the direct-engine oracle, built once from the SAME bytes ---
	superslm::SslmModelView oracle_view;
	std::string oracle_err;
	if (superslm::SslmModel::Load(bytes.data(), bytes.size(), oracle_view, &oracle_err) !=
	    superslm::SslmModelStatus::Ok) {
		std::fprintf(stderr, "FAIL: oracle SslmModel::Load: %s\n", oracle_err.c_str());
		return 1;
	}
	CpuOracleModel oracle;
	if (!LoadCpuOracleModel(oracle_view, &oracle, &oracle_err)) {
		std::fprintf(stderr, "FAIL: LoadCpuOracleModel: %s\n", oracle_err.c_str());
		return 1;
	}
	if (oracle.kv_precision != superslm::SslmKvPrecision::Int8) {
		std::printf("SKIP: artifact declares a non-Int8 kv_precision -- RunGreedyDecodeLoop "
		            "itself rejects Int16 (KvPrecisionUnsupported); no decodable oracle exists "
		            "to compare against.\n");
		return 0;
	}

	const int32_t prompt_tokens[] = {1, 2, 3, 4, 5};
	const size_t prompt_len = 5;

	std::vector<int32_t> oracle_tokens;
	size_t oracle_produced = 0;
	superslm::SslmDecodeStopReason oracle_stop{};
	const superslm::SslmForwardStatus ora_st =
	    RunGreedyOracle(oracle, prompt_tokens, prompt_len, max_new_tokens, &oracle_tokens,
	                     &oracle_produced, &oracle_stop);
	if (ora_st != superslm::SslmForwardStatus::Ok) {
		std::fprintf(stderr, "FAIL: RunGreedyOracle returned status %d\n", static_cast<int>(ora_st));
		return 1;
	}
	oracle_tokens.resize(oracle_produced);

	// --- (A) this ABI, over the SAME artifact bytes and the SAME prompt ---
	sslm_model model = nullptr;
	sslm_status st = sslm_model_map(bytes.data(), bytes.size(), &model);
	if (st != SSLM_OK || !model) {
		std::fprintf(stderr, "FAIL: sslm_model_map returned %d\n", static_cast<int>(st));
		return 1;
	}

	const uint32_t block_count = 1;
	const size_t block_bytes = sslm_kv_block_size(model);
	const size_t overhead = sslm_kv_pool_overhead_size(model, block_count);
	// S2 (Claude/Poirot/2c18dab-t2139-abi-build-review.md): sslm_kv_pool_create now checks
	// alignment -- over-allocate and round up, matching the workspace buffer's own established
	// pattern (tools/t2139_c2_smoke.cpp). `pool_raw` retained as the aligned void* base this
	// tool later reads directly (see its own comment below, near "This tool reaches into the
	// buffer it itself allocated").
	const size_t pool_buf_size = block_bytes * block_count + overhead;
	std::vector<uint8_t> pool_raw_storage(pool_buf_size + 63);
	void* pool_raw = pool_raw_storage.data();
	size_t pool_raw_space = pool_raw_storage.size();
	std::align(64, pool_buf_size, pool_raw, pool_raw_space);
	sslm_kv_pool pool = nullptr;
	st = sslm_kv_pool_create(model, pool_raw, pool_buf_size, block_count, &pool);
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

	int32_t consumed = 0;
	st = sslm_prefill(model, seq, prompt_tokens, static_cast<int32_t>(prompt_len),
	                   static_cast<int32_t>(prompt_len), SSLM_SPAN_PROMPT, nullptr, &consumed);
	if (st != SSLM_OK || consumed != static_cast<int32_t>(prompt_len)) {
		std::fprintf(stderr, "FAIL: sslm_prefill returned %d, consumed %d\n",
		             static_cast<int>(st), consumed);
		return 1;
	}

	std::vector<int32_t> abi_tokens;
	sslm_decode_params params{};
	params.layer_budget = forced_layer_budget > 0 ? forced_layer_budget
	                                               : static_cast<int32_t>(oracle.num_hidden_layers);
	sslm_seq seqs[1] = {seq};
	int32_t calls_this_run = 0;
	for (size_t t = 0; t < max_new_tokens; ++t) {
		int32_t out_token = -1;
		int32_t guard = 0;
		while (out_token < 0) {
			st = sslm_decode_step(model, seqs, 1, &params, nullptr, &out_token);
			++calls_this_run;
			if (st != SSLM_OK) {
				std::fprintf(stderr, "FAIL: sslm_decode_step (token %zu) returned %d\n", t,
				             static_cast<int>(st));
				return 1;
			}
			if (++guard > 10000) {
				std::fprintf(stderr, "FAIL: sslm_decode_step (token %zu) never completed -- "
				                      "possible resumption bug\n", t);
				return 1;
			}
		}
		abi_tokens.push_back(out_token);
	}
	if (forced_layer_budget > 0) {
		std::printf("bounded layer_budget=%d: %d sslm_decode_step calls to produce %zu tokens\n",
		            forced_layer_budget, calls_this_run, max_new_tokens);
	}

	// --- comparison ---
	bool ok = true;
	if (abi_tokens.size() != oracle_tokens.size()) {
		std::fprintf(stderr, "FAIL: token count diverges: ABI=%zu oracle=%zu\n",
		             abi_tokens.size(), oracle_tokens.size());
		ok = false;
	} else {
		for (size_t i = 0; i < abi_tokens.size(); ++i) {
			if (abi_tokens[i] != oracle_tokens[i]) {
				std::fprintf(stderr, "FAIL: token[%zu] diverges: ABI=%d oracle=%d\n", i,
				             abi_tokens[i], oracle_tokens[i]);
				ok = false;
			}
		}
	}
	if (ok) {
		std::printf("token sequence: BIT-IDENTICAL (%zu tokens): ", abi_tokens.size());
		for (int32_t t : abi_tokens) std::printf("%d ", t);
		std::printf("\n");
	}

	// KV bytes: compare this ABI's own drawn block against the oracle's own workspace, over the
	// region both paths actually wrote (min of the two byte counts -- both should be identical
	// by construction, checked explicitly rather than assumed).
	{
		const size_t oracle_kv_bytes = static_cast<size_t>(oracle.num_hidden_layers) *
		                                static_cast<size_t>(oracle.context_cap) *
		                                oracle.num_kv_heads * oracle.head_dim * 2;
		if (oracle_kv_bytes != block_bytes) {
			std::fprintf(stderr,
			             "FAIL: KV byte-count formulas diverge: ABI sslm_kv_block_size=%zu "
			             "oracle=%zu\n",
			             block_bytes, oracle_kv_bytes);
			ok = false;
		} else {
			// Re-run the oracle's own RunGreedyOracle a second time, independently, to obtain
			// its own final workspace bytes for comparison (RunGreedyOracle's own workspace is
			// local to that call and was not returned above) -- a fresh run against the
			// IDENTICAL prompt/max_new_tokens is bit-reproducible by this project's own proven
			// determinism property (S-HARDEN family), so comparing against a second run's own
			// buffer is equivalent to comparing against the first.
			const size_t kv_bytes = static_cast<size_t>(oracle.num_hidden_layers) *
			                         static_cast<size_t>(oracle.context_cap) * oracle.num_kv_heads *
			                         oracle.head_dim * 2;
			std::vector<uint8_t> oracle_workspace(kv_bytes);
			std::vector<int8_t> oracle_hidden(oracle.hidden_size);
			superslm::SequenceLayerState oracle_seq{};
			oracle_seq.hidden_codes = oracle_hidden.data();
			std::vector<int32_t> oracle_tokens2(max_new_tokens, 0);
			std::vector<int32_t> oracle_logit_rows(max_new_tokens *
			                                        static_cast<size_t>(oracle.vocab_size));
			size_t produced2 = 0;
			superslm::SslmDecodeStopReason stop2{};
			const superslm::SslmForwardStatus st2 = superslm::RunGreedyDecodeLoop(
			    oracle_seq, oracle.layers.data(), oracle.num_hidden_layers, oracle.hidden_size,
			    oracle.head_dim, oracle.num_kv_heads, oracle.intermediate_size, oracle.context_cap,
			    *oracle.rope_tables, prompt_tokens, prompt_len, oracle.embed_weights,
			    oracle.embed_site_constant, oracle.final_norm_gain.data(),
			    oracle.final_norm_site_constant, oracle.head_weights, oracle.vocab_size, nullptr,
			    0, max_new_tokens, oracle_workspace.data(), oracle_workspace.size(),
			    oracle_tokens2.data(), oracle_logit_rows.data(), oracle_tokens2.size(), &produced2,
			    &stop2, oracle.kv_precision, oracle.option_g_fused_k_landing);
			if (st2 != superslm::SslmForwardStatus::Ok) {
				std::fprintf(stderr, "FAIL: second oracle run returned status %d\n",
				             static_cast<int>(st2));
				ok = false;
			} else {
				// This pool's block 0 (the only block a 1-block pool can ever draw) sits at
				// pool_raw.data() + 0 -- sslm_kv_pool_create's own layout (src/sslm_abi.cpp)
				// places blocks starting at the caller buffer's own base address, with the
				// bookkeeping overhead this tool sized via sslm_kv_pool_overhead_size occupying
				// unused padding at the END of the buffer, never written into. This tool reaches
				// into the buffer it itself allocated and passed to sslm_kv_pool_create --
				// legitimate for THIS verification tool (it owns pool_raw), not a public ABI
				// accessor (none is declared; sslm_kv_pool stays opaque to every real caller).
				const uint8_t* abi_kv = static_cast<const uint8_t*>(pool_raw);
				if (std::memcmp(abi_kv, oracle_workspace.data(), kv_bytes) != 0) {
					std::fprintf(stderr, "FAIL: KV block bytes diverge from the oracle's own workspace\n");
					ok = false;
				} else {
					std::printf("KV block bytes: BIT-IDENTICAL (%zu bytes)\n", kv_bytes);
				}
			}
		}
	}

	if (!ok) {
		std::fprintf(stderr, "t2139_c4_oracle: FAIL\n");
		return 1;
	}
	std::printf("t2139_c4_oracle: PASS\n");
	return 0;
}

// t2139_c6_smoke.cpp -- Gate B for C6 (design Sec9: "a smoke TU that maps a real adapter, binds
// it to a decoding sequence, decodes, releases"). Also exercises: adapter-model mismatch
// (SSLM_ADAPTER_MODEL_MISMATCH, a foreign base), mid-token swap rejection
// (SSLM_ADAPTER_SWAP_MIDTOKEN_REJECTED), the product-scale lifecycle-guard cell design Sec10
// dim11 names ("reach at least one lifecycle guard through ORDINARY API sequencing" --
// sslm_adapter_release while the sequence it is bound to is still live, arising from this
// smoke's own normal call order, not a fixture hand-built solely to trigger the guard), and
// residency reporting.
//
// Usage: t2139_c6_smoke.exe <path-to-real-base.sslm> <path-to-real-adapter.sslm>
//        [path-to-foreign-base.sslm]
#include <cstdio>
#include <cstdlib>
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
	if (argc < 3) {
		std::fprintf(stderr,
		             "usage: %s <path-to-real-base.sslm> <path-to-real-adapter.sslm> "
		             "[path-to-foreign-base.sslm]\n",
		             argv[0]);
		return 1;
	}
	std::vector<uint8_t> base_bytes, adapter_bytes;
	if (!ReadFile(argv[1], &base_bytes)) {
		std::fprintf(stderr, "FAIL: could not read %s\n", argv[1]);
		return 1;
	}
	if (!ReadFile(argv[2], &adapter_bytes)) {
		std::fprintf(stderr, "FAIL: could not read %s\n", argv[2]);
		return 1;
	}

	sslm_model base = nullptr;
	sslm_status st = sslm_model_map(base_bytes.data(), base_bytes.size(), &base);
	if (st != SSLM_OK || !base) {
		std::fprintf(stderr, "FAIL: sslm_model_map(base) returned %d\n", static_cast<int>(st));
		return 1;
	}

	sslm_adapter adapter = nullptr;
	st = sslm_adapter_map(adapter_bytes.data(), adapter_bytes.size(), base, &adapter);
	if (st != SSLM_OK || !adapter) {
		std::fprintf(stderr, "FAIL: sslm_adapter_map returned %d\n", static_cast<int>(st));
		return 1;
	}
	const size_t residency = sslm_adapter_residency(adapter);
	if (residency != adapter_bytes.size()) {
		std::fprintf(stderr, "FAIL: sslm_adapter_residency=%zu, expected %zu\n", residency,
		             adapter_bytes.size());
		return 1;
	}
	std::printf("sslm_adapter_map: PASS (residency=%zu bytes)\n", residency);

	// --- adapter-model mismatch, if a foreign base was supplied ---
	if (argc >= 4) {
		std::vector<uint8_t> foreign_bytes;
		if (!ReadFile(argv[3], &foreign_bytes)) {
			std::fprintf(stderr, "FAIL: could not read %s\n", argv[3]);
			return 1;
		}
		sslm_model foreign = nullptr;
		st = sslm_model_map(foreign_bytes.data(), foreign_bytes.size(), &foreign);
		if (st != SSLM_OK || !foreign) {
			std::fprintf(stderr, "FAIL: sslm_model_map(foreign) returned %d\n", static_cast<int>(st));
			return 1;
		}
		sslm_adapter foreign_probe = nullptr;
		st = sslm_adapter_map(adapter_bytes.data(), adapter_bytes.size(), foreign, &foreign_probe);
		if (st != SSLM_ADAPTER_MODEL_MISMATCH || foreign_probe != nullptr) {
			std::fprintf(stderr,
			             "FAIL: sslm_adapter_map(adapter, foreign base) returned %d, expected "
			             "SSLM_ADAPTER_MODEL_MISMATCH\n",
			             static_cast<int>(st));
			return 1;
		}
		sslm_model_unmap(foreign);
		std::printf("sslm_adapter_map foreign-base rejection: PASS\n");
	}

	const uint32_t block_count = 1;
	const size_t block_bytes = sslm_kv_block_size(base);
	const size_t overhead = sslm_kv_pool_overhead_size(base, block_count);
	// S2 (Claude/Poirot/2c18dab-t2139-abi-build-review.md): sslm_kv_pool_create now checks
	// alignment -- over-allocate and round up, matching tools/t2139_c2_smoke.cpp's own pattern.
	const size_t pool_buf_size = block_bytes * block_count + overhead;
	std::vector<uint8_t> pool_raw_storage(pool_buf_size + 63);
	void* pool_raw = pool_raw_storage.data();
	size_t pool_raw_space = pool_raw_storage.size();
	std::align(64, pool_buf_size, pool_raw, pool_raw_space);
	sslm_kv_pool pool = nullptr;
	st = sslm_kv_pool_create(base, pool_raw, pool_buf_size, block_count, &pool);
	if (st != SSLM_OK || !pool) {
		std::fprintf(stderr, "FAIL: sslm_kv_pool_create returned %d\n", static_cast<int>(st));
		return 1;
	}

	sslm_seq seq = nullptr;
	st = sslm_seq_create(base, &pool, &seq);
	if (st != SSLM_OK || !seq) {
		std::fprintf(stderr, "FAIL: sslm_seq_create returned %d\n", static_cast<int>(st));
		return 1;
	}

	const int32_t prompt_tokens[] = {1, 2, 3};
	int32_t consumed = 0;
	st = sslm_prefill(base, seq, prompt_tokens, 3, 3, SSLM_SPAN_PROMPT, nullptr, &consumed);
	if (st != SSLM_OK || consumed != 3) {
		std::fprintf(stderr, "FAIL: sslm_prefill returned %d consumed %d\n", static_cast<int>(st),
		             consumed);
		return 1;
	}

	// bind the adapter (design's own smoke shape: "binds it to a decoding sequence")
	st = sslm_seq_set_adapter(seq, adapter);
	if (st != SSLM_OK) {
		std::fprintf(stderr, "FAIL: sslm_seq_set_adapter returned %d\n", static_cast<int>(st));
		return 1;
	}
	std::printf("sslm_seq_set_adapter: PASS\n");

	// decode a real step through the adapted path -- real LoRA delta composition, no crash.
	sslm_decode_params params{};
	// T-2199 Phase D review addendum (D-SLM3797, Dan): struct_size is the FIRST
	// new field the library now validates -- an unrecognized size is a loud rejection.
	params.struct_size = sizeof(params);
	params.layer_budget = 1;  // bounded, to force a genuine mid-token state on the SECOND call
	                          // below (the FIRST call after a completed prefill is always the
	                          // free ready_for_logits step, sslm_seq_s's own header comment --
	                          // no RunLayerLoop work, so layer_budget is irrelevant to it and a
	                          // real token completes in one call regardless).
	sslm_seq seqs[1] = {seq};
	int32_t out_token = -1;
	st = sslm_decode_step(base, seqs, 1, &params, nullptr, &out_token);
	if (st != SSLM_OK) {
		std::fprintf(stderr, "FAIL: sslm_decode_step (adapted) returned %d\n", static_cast<int>(st));
		return 1;
	}
	std::printf("sslm_decode_step through the adapted path: PASS (out_token=%d)\n", out_token);

	// A second, bounded call starts a NEW token (embed + RunLayerLoop at layer_budget=1) -- this
	// is the one that is genuinely mid-token afterward on any model with more than one layer.
	out_token = -1;
	st = sslm_decode_step(base, seqs, 1, &params, nullptr, &out_token);
	if (st != SSLM_OK) {
		std::fprintf(stderr, "FAIL: sslm_decode_step (adapted, second call) returned %d\n",
		             static_cast<int>(st));
		return 1;
	}
	if (out_token >= 0) {
		std::fprintf(stderr,
		             "FAIL: sslm_decode_step (second call) completed a token at layer_budget=1 "
		             "-- expected PENDING (-1) on a multi-layer model\n");
		return 1;
	}

	// mid-token swap is a defined rejection -- the sequence is genuinely mid-token now.
	st = sslm_seq_set_adapter(seq, nullptr);
	if (st != SSLM_ADAPTER_SWAP_MIDTOKEN_REJECTED) {
		std::fprintf(stderr,
		             "FAIL: sslm_seq_set_adapter mid-token returned %d, expected "
		             "SSLM_ADAPTER_SWAP_MIDTOKEN_REJECTED\n",
		             static_cast<int>(st));
		return 1;
	}
	std::printf("sslm_seq_set_adapter mid-token rejection: PASS\n");

	// --- the product-scale lifecycle-guard cell (design Sec10 dim11): reach
	// SSLM_ADAPTER_HAS_LIVE_SEQUENCES through this smoke's own ORDINARY call order -- the
	// sequence above is still live and still bound, so releasing the adapter now, before
	// releasing the sequence, is exactly the guard's own precondition, arising naturally rather
	// than via a fixture hand-built solely to trigger it. ---
	st = sslm_adapter_release(adapter);
	if (st != SSLM_ADAPTER_HAS_LIVE_SEQUENCES) {
		std::fprintf(stderr,
		             "FAIL: sslm_adapter_release while a live sequence is bound returned %d, "
		             "expected SSLM_ADAPTER_HAS_LIVE_SEQUENCES\n",
		             static_cast<int>(st));
		return 1;
	}
	std::printf("sslm_adapter_release guard (live sequence still bound): PASS\n");

	// design's own smoke shape's own final step: release, in the order that actually works --
	// unbind first (now that the sequence has finished its mid-token step and is resting), then
	// release the sequence, then the adapter.
	{
		// Drive the pending token to completion first (layer_index must return to 0 before
		// unbinding is legal, same rule sslm_seq_reset already enforces).
		int guard = 0;
		while (out_token < 0) {
			st = sslm_decode_step(base, seqs, 1, &params, nullptr, &out_token);
			if (st != SSLM_OK) {
				std::fprintf(stderr, "FAIL: sslm_decode_step (draining) returned %d\n",
				             static_cast<int>(st));
				return 1;
			}
			if (++guard > 100000) {
				std::fprintf(stderr, "FAIL: sequence never completed its pending token\n");
				return 1;
			}
		}
	}
	st = sslm_seq_set_adapter(seq, nullptr);
	if (st != SSLM_OK) {
		std::fprintf(stderr, "FAIL: sslm_seq_set_adapter(nullptr) returned %d\n",
		             static_cast<int>(st));
		return 1;
	}
	st = sslm_seq_release(seq);
	if (st != SSLM_OK) {
		std::fprintf(stderr, "FAIL: sslm_seq_release returned %d\n", static_cast<int>(st));
		return 1;
	}
	st = sslm_adapter_release(adapter);
	if (st != SSLM_OK) {
		std::fprintf(stderr, "FAIL: sslm_adapter_release (final) returned %d\n", static_cast<int>(st));
		return 1;
	}
	sslm_kv_pool_destroy(pool);
	sslm_model_unmap(base);

	std::printf("t2139_c6_smoke: PASS\n");
	return 0;
}

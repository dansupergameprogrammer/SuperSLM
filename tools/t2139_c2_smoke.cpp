// t2139_c2_smoke.cpp -- Gate B for C2 (design Sec9: "a smoke TU that maps a real artifact,
// calls sslm_seq_state_size on the resulting handle, unmaps"). Authored fresh for this build,
// never a repurposed dimN_red.cpp -- includes superslm/sslm_abi.h and links against the
// just-built library alone, no G5-only symbol reference. Exit 0, no crash, is the whole gate.
//
// Extended beyond design Sec9's own minimum (which names no Gate B for C1) to also exercise
// C1's own construction verbs -- sslm_workspace_size/_create/_destroy and
// sslm_kv_pool_overhead_size/_create/_destroy -- against the same real, mapped model, before
// unmapping it. Design Sec9 states "no Gate B here" for C1 because C1's two invented verb pairs
// have no real-suite caller to smoke-link against yet; this file is not that suite caller
// either (it is this build's own verification, not a red-suite cell), so it can and does drive
// them for real without contradicting that scoping -- Brunel's own "prove it builds and runs
// before handoff" discipline, not a claim that this satisfies a C1 Gate B design Sec9 does not
// require.
//
// Usage: t2139_c2_smoke.exe <path-to-real.sslm> [path-to-a-DIFFERENT-shaped-real.sslm]
// The optional second argument (a different tier, e.g. the 0.5B artifact against a 1.5B first
// argument) drives the C2 fix-round pool/model-mismatch pin (Claude/Poirot/
// 2c18dab-t2139-abi-build-review.md) -- SKIPs, not silently passes, when omitted.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
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
	if (st != SSLM_OK) {
		std::fprintf(stderr, "FAIL: sslm_model_map returned %d\n", static_cast<int>(st));
		return 1;
	}
	if (!model) {
		std::fprintf(stderr, "FAIL: sslm_model_map returned SSLM_OK with a null handle\n");
		return 1;
	}

	const size_t state_size = sslm_seq_state_size(model);
	if (state_size == 0) {
		std::fprintf(stderr, "FAIL: sslm_seq_state_size returned 0 for a real, successfully mapped model\n");
		sslm_model_unmap(model);
		return 1;
	}
	std::printf("sslm_seq_state_size: %zu bytes\n", state_size);

	// --- C1 construction verbs, exercised for real against this same live model handle ---
	{
		sslm_config cfg{};
		cfg.max_batch = 1;
		cfg.max_chunk_budget = 64;
		cfg.max_layer_budget = 1;  // any value in [1, num_hidden_layers] is domain-valid
		cfg.reserved = 0;
		const size_t ws_size = sslm_workspace_size(model, &cfg);
		if (ws_size == 0) {
			std::fprintf(stderr, "FAIL: sslm_workspace_size returned 0 for a domain-valid config\n");
			sslm_model_unmap(model);
			return 1;
		}
		// 64-byte-aligned heap buffer via std::unique_ptr<uint8_t[]> + manual alignment padding
		// (no aligned_alloc portability assumption made here -- over-allocate and round up).
		std::vector<uint8_t> raw(ws_size + 63);
		void* aligned = raw.data();
		size_t space = raw.size();
		void* p = aligned;
		std::align(64, ws_size, p, space);
		if (!p) {
			std::fprintf(stderr, "FAIL: could not align a workspace buffer\n");
			sslm_model_unmap(model);
			return 1;
		}
		sslm_workspace ws = nullptr;
		st = sslm_workspace_create(model, &cfg, p, ws_size, &ws);
		if (st != SSLM_OK || !ws) {
			std::fprintf(stderr, "FAIL: sslm_workspace_create returned %d\n", static_cast<int>(st));
			sslm_model_unmap(model);
			return 1;
		}
		st = sslm_workspace_destroy(ws);
		if (st != SSLM_OK) {
			std::fprintf(stderr, "FAIL: sslm_workspace_destroy returned %d\n", static_cast<int>(st));
			sslm_model_unmap(model);
			return 1;
		}
		std::printf("sslm_workspace_size/create/destroy: PASS (%zu bytes)\n", ws_size);

		// --- hostile-input paths (design Sec7.1/Sec9 C1 gate: "SSLM_BUFFER_TOO_SMALL/
		// SSLM_MISALIGNED_BUFFER fire on every under-sized/misaligned input"), executed for
		// real rather than reasoned from the code alone (StandardsDocument Sec5.4). ---
		{
			sslm_config hostile{};  // all-zero: every field hostile (design Sec7.1)
			if (sslm_workspace_size(model, &hostile) != 0) {
				std::fprintf(stderr, "FAIL: sslm_workspace_size did not sentinel-reject an all-zero config\n");
				sslm_model_unmap(model);
				return 1;
			}
			sslm_workspace bad_ws = nullptr;
			st = sslm_workspace_create(model, &hostile, p, ws_size, &bad_ws);
			if (st != SSLM_INVALID_ARGUMENT) {
				std::fprintf(stderr, "FAIL: sslm_workspace_create(hostile config) returned %d, expected SSLM_INVALID_ARGUMENT\n",
				              static_cast<int>(st));
				sslm_model_unmap(model);
				return 1;
			}
			st = sslm_workspace_create(model, &cfg, p, ws_size - 1, &bad_ws);
			if (st != SSLM_BUFFER_TOO_SMALL) {
				std::fprintf(stderr, "FAIL: sslm_workspace_create(1-byte-short buffer) returned %d, expected SSLM_BUFFER_TOO_SMALL\n",
				              static_cast<int>(st));
				sslm_model_unmap(model);
				return 1;
			}
			// A deliberately misaligned pointer (one byte off the 64-byte-aligned region), at a
			// SUFFICIENT buf_size -- isolating alignment as the only hostile variable, since
			// sslm_workspace_create checks buf_size before alignment and an also-too-small
			// buffer would report SSLM_BUFFER_TOO_SMALL first, testing nothing about alignment
			// (found by executing this exact case with ws_size - 1, StandardsDocument Sec5.4:
			// exactness verified by execution surfaced the test's own unisolated variable).
			// raw has ws_size + 63 bytes of slack from std::align above, so p + 1 sized at
			// ws_size stays within raw's real allocation.
			void* misaligned = static_cast<uint8_t*>(p) + 1;
			st = sslm_workspace_create(model, &cfg, misaligned, ws_size, &bad_ws);
			if (st != SSLM_MISALIGNED_BUFFER) {
				std::fprintf(stderr, "FAIL: sslm_workspace_create(misaligned buffer) returned %d, expected SSLM_MISALIGNED_BUFFER\n",
				              static_cast<int>(st));
				sslm_model_unmap(model);
				return 1;
			}
			std::printf("sslm_workspace_size/create hostile-input paths: PASS (sentinel 0, "
			            "SSLM_INVALID_ARGUMENT, SSLM_BUFFER_TOO_SMALL, SSLM_MISALIGNED_BUFFER "
			            "all fired as designed)\n");
		}

		const uint32_t block_count = 4;
		const size_t block_bytes = sslm_kv_block_size(model);
		const size_t overhead = sslm_kv_pool_overhead_size(model, block_count);
		if (block_bytes == 0 || overhead == 0) {
			std::fprintf(stderr, "FAIL: sslm_kv_block_size/sslm_kv_pool_overhead_size returned 0\n");
			sslm_model_unmap(model);
			return 1;
		}
		const size_t pool_buf_size = block_bytes * block_count + overhead;
		// S2 (Claude/Poirot/2c18dab-t2139-abi-build-review.md): sslm_kv_pool_create now checks
		// alignment too -- same over-allocate-and-round-up pattern as the workspace buffer above.
		std::vector<uint8_t> pool_raw_storage(pool_buf_size + 63);
		void* pool_aligned = pool_raw_storage.data();
		size_t pool_space = pool_raw_storage.size();
		std::align(64, pool_buf_size, pool_aligned, pool_space);
		if (!pool_aligned) {
			std::fprintf(stderr, "FAIL: could not align a pool buffer\n");
			sslm_model_unmap(model);
			return 1;
		}
		sslm_kv_pool pool = nullptr;
		st = sslm_kv_pool_create(model, pool_aligned, pool_buf_size, block_count, &pool);
		if (st != SSLM_OK || !pool) {
			std::fprintf(stderr, "FAIL: sslm_kv_pool_create returned %d\n", static_cast<int>(st));
			sslm_model_unmap(model);
			return 1;
		}
		{
			sslm_kv_pool bad_pool = nullptr;
			st = sslm_kv_pool_create(model, pool_aligned, pool_buf_size - 1, block_count, &bad_pool);
			if (st != SSLM_BUFFER_TOO_SMALL) {
				std::fprintf(stderr, "FAIL: sslm_kv_pool_create(1-byte-short buffer) returned %d, expected SSLM_BUFFER_TOO_SMALL\n",
				              static_cast<int>(st));
				sslm_model_unmap(model);
				return 1;
			}
			std::printf("sslm_kv_pool_create hostile-input path: PASS (SSLM_BUFFER_TOO_SMALL fired as designed)\n");
			// S2's own must-reject pin: a deliberately misaligned pointer at a sufficient
			// buf_size, isolating alignment as the only hostile variable (same isolation
			// discipline as the workspace's own misalignment cell above).
			void* pool_misaligned = static_cast<uint8_t*>(pool_aligned) + 1;
			st = sslm_kv_pool_create(model, pool_misaligned, pool_buf_size, block_count, &bad_pool);
			if (st != SSLM_MISALIGNED_BUFFER) {
				std::fprintf(stderr, "FAIL: sslm_kv_pool_create(misaligned buffer) returned %d, expected SSLM_MISALIGNED_BUFFER\n",
				              static_cast<int>(st));
				sslm_model_unmap(model);
				return 1;
			}
			std::printf("sslm_kv_pool_create misalignment pin: PASS (SSLM_MISALIGNED_BUFFER fired as designed)\n");
		}
		st = sslm_kv_pool_destroy(pool);
		if (st != SSLM_OK) {
			std::fprintf(stderr, "FAIL: sslm_kv_pool_destroy returned %d\n", static_cast<int>(st));
			sslm_model_unmap(model);
			return 1;
		}
		std::printf("sslm_kv_block_size/sslm_kv_pool_overhead_size/create/destroy: PASS "
		            "(block=%zu bytes, overhead=%zu bytes, pool=%zu bytes)\n",
		            block_bytes, overhead, pool_buf_size);
	}

	// C2 (Claude/Poirot/2c18dab-t2139-abi-build-review.md): the actual must-overflow-turned-
	// reject pin, against two REAL, differently-shaped artifacts -- exactly the pre-fix hazard's
	// own shape (measured, pre-fix: restoring a 1.5B sequence into a 0.5B-sized pool overran the
	// destination by 268,435,456 bytes; well-formed inputs, no malformed data anywhere). Optional
	// (argv[2]): a second real artifact of a DIFFERENT shape, e.g. the 0.5B tier against a 1.5B
	// `model`. Absent argv[2], this whole pin SKIPs rather than silently passing.
	if (argc >= 3) {
		std::vector<uint8_t> bytes_b;
		if (!ReadFile(argv[2], &bytes_b)) {
			std::fprintf(stderr, "FAIL: could not read %s (foreign model for C2 pin)\n", argv[2]);
			sslm_model_unmap(model);
			return 1;
		}
		sslm_model model_b = nullptr;
		st = sslm_model_map(bytes_b.data(), bytes_b.size(), &model_b);
		if (st != SSLM_OK) {
			std::fprintf(stderr, "FAIL: sslm_model_map(foreign) returned %d\n", static_cast<int>(st));
			sslm_model_unmap(model);
			return 1;
		}

		const uint32_t pin_block_count = 1;
		const size_t b_block_bytes = sslm_kv_block_size(model_b);
		const size_t b_overhead = sslm_kv_pool_overhead_size(model_b, pin_block_count);
		const size_t b_pool_buf_size = b_block_bytes * pin_block_count + b_overhead;
		std::vector<uint8_t> b_pool_storage(b_pool_buf_size + 63);
		void* b_pool_aligned = b_pool_storage.data();
		size_t b_pool_space = b_pool_storage.size();
		std::align(64, b_pool_buf_size, b_pool_aligned, b_pool_space);
		sslm_kv_pool pool_b = nullptr;
		st = sslm_kv_pool_create(model_b, b_pool_aligned, b_pool_buf_size, pin_block_count, &pool_b);
		if (st != SSLM_OK || !pool_b) {
			std::fprintf(stderr, "FAIL: sslm_kv_pool_create(model_b) returned %d\n", static_cast<int>(st));
			sslm_model_unmap(model_b);
			sslm_model_unmap(model);
			return 1;
		}

		sslm_seq bad_seq = nullptr;
		st = sslm_seq_create(model, &pool_b, &bad_seq);
		if (st != SSLM_INVALID_ARGUMENT || bad_seq != nullptr) {
			std::fprintf(stderr,
			              "FAIL: sslm_seq_create(model, mismatched-model pool) returned %d, "
			              "expected SSLM_INVALID_ARGUMENT\n",
			              static_cast<int>(st));
			sslm_kv_pool_destroy(pool_b);
			sslm_model_unmap(model_b);
			sslm_model_unmap(model);
			return 1;
		}
		sslm_prefix bad_prefix = nullptr;
		st = sslm_prefix_begin(model, &pool_b, &bad_prefix);
		if (st != SSLM_INVALID_ARGUMENT || bad_prefix != nullptr) {
			std::fprintf(stderr,
			              "FAIL: sslm_prefix_begin(model, mismatched-model pool) returned %d, "
			              "expected SSLM_INVALID_ARGUMENT\n",
			              static_cast<int>(st));
			sslm_kv_pool_destroy(pool_b);
			sslm_model_unmap(model_b);
			sslm_model_unmap(model);
			return 1;
		}
		std::printf("C2 pool/model mismatch pin (sslm_seq_create, sslm_prefix_begin): PASS "
		            "(SSLM_INVALID_ARGUMENT fired as designed)\n");

		// THE actual overflow site (src/sslm_abi.cpp, sslm_seq_restore): a real, valid,
		// well-formed saved blob from `model`'s own correctly-sized pool, restored against
		// pool_b (bound to `model_b`, a DIFFERENT model). Pre-fix this memcpy'd `model`'s own
		// block_size into pool_b's own (possibly smaller) per-block stride.
		const size_t a_block_bytes = sslm_kv_block_size(model);
		const size_t a_overhead = sslm_kv_pool_overhead_size(model, pin_block_count);
		const size_t a_pool_buf_size = a_block_bytes * pin_block_count + a_overhead;
		std::vector<uint8_t> a_pool_storage(a_pool_buf_size + 63);
		void* a_pool_aligned = a_pool_storage.data();
		size_t a_pool_space = a_pool_storage.size();
		std::align(64, a_pool_buf_size, a_pool_aligned, a_pool_space);
		sslm_kv_pool pool_a = nullptr;
		st = sslm_kv_pool_create(model, a_pool_aligned, a_pool_buf_size, pin_block_count, &pool_a);
		if (st != SSLM_OK || !pool_a) {
			std::fprintf(stderr, "FAIL: sslm_kv_pool_create(model) returned %d\n", static_cast<int>(st));
			sslm_kv_pool_destroy(pool_b);
			sslm_model_unmap(model_b);
			sslm_model_unmap(model);
			return 1;
		}
		sslm_seq seq_a = nullptr;
		st = sslm_seq_create(model, &pool_a, &seq_a);
		if (st != SSLM_OK || !seq_a) {
			std::fprintf(stderr, "FAIL: sslm_seq_create(model, pool_a) returned %d\n", static_cast<int>(st));
			sslm_kv_pool_destroy(pool_a);
			sslm_kv_pool_destroy(pool_b);
			sslm_model_unmap(model_b);
			sslm_model_unmap(model);
			return 1;
		}
		const int32_t prompt_tokens[1] = {0};  // token id 0 -- always in-domain for any real vocab
		int32_t consumed = 0;
		st = sslm_prefill(model, seq_a, prompt_tokens, 1, 1, SSLM_SPAN_PROMPT, nullptr, &consumed);
		if (st != SSLM_OK || consumed != 1) {
			std::fprintf(stderr, "FAIL: sslm_prefill(seq_a) returned %d, consumed=%d\n",
			              static_cast<int>(st), consumed);
			sslm_seq_release(seq_a);
			sslm_kv_pool_destroy(pool_a);
			sslm_kv_pool_destroy(pool_b);
			sslm_model_unmap(model_b);
			sslm_model_unmap(model);
			return 1;
		}
		size_t need = 0;
		st = sslm_seq_save(seq_a, nullptr, &need);
		if (st != SSLM_BUFFER_TOO_SMALL || need == 0) {
			std::fprintf(stderr, "FAIL: sslm_seq_save(sizing) returned %d, need=%zu\n",
			              static_cast<int>(st), need);
			sslm_seq_release(seq_a);
			sslm_kv_pool_destroy(pool_a);
			sslm_kv_pool_destroy(pool_b);
			sslm_model_unmap(model_b);
			sslm_model_unmap(model);
			return 1;
		}
		std::vector<uint8_t> real_blob(need);
		size_t n = real_blob.size();
		st = sslm_seq_save(seq_a, real_blob.data(), &n);
		if (st != SSLM_OK) {
			std::fprintf(stderr, "FAIL: sslm_seq_save(real) returned %d\n", static_cast<int>(st));
			sslm_seq_release(seq_a);
			sslm_kv_pool_destroy(pool_a);
			sslm_kv_pool_destroy(pool_b);
			sslm_model_unmap(model_b);
			sslm_model_unmap(model);
			return 1;
		}
		sslm_seq restored_bad = nullptr;
		st = sslm_seq_restore(model, &pool_b, real_blob.data(), n, &restored_bad);
		if (st != SSLM_INVALID_ARGUMENT || restored_bad != nullptr) {
			std::fprintf(stderr,
			              "FAIL: sslm_seq_restore(model, mismatched-model pool_b, REAL well-"
			              "formed blob) returned %d, expected SSLM_INVALID_ARGUMENT -- this IS "
			              "the pre-fix heap-overflow site\n",
			              static_cast<int>(st));
			sslm_seq_release(seq_a);
			sslm_kv_pool_destroy(pool_a);
			sslm_kv_pool_destroy(pool_b);
			sslm_model_unmap(model_b);
			sslm_model_unmap(model);
			return 1;
		}
		std::printf("C2 sslm_seq_restore pool/model mismatch pin: PASS (SSLM_INVALID_ARGUMENT "
		            "fired as designed -- the pre-fix 268,435,456-byte heap overflow's own real "
		            "call shape, now rejected)\n");

		sslm_seq_release(seq_a);
		sslm_kv_pool_destroy(pool_a);
		sslm_kv_pool_destroy(pool_b);
		st = sslm_model_unmap(model_b);
		if (st != SSLM_OK) {
			std::fprintf(stderr, "FAIL: sslm_model_unmap(model_b) returned %d\n", static_cast<int>(st));
			sslm_model_unmap(model);
			return 1;
		}
	} else {
		std::printf("C2 pool/model mismatch pin: SKIP (no argv[2] -- a second, differently-"
		            "shaped real artifact was not supplied on this invocation)\n");
	}

	st = sslm_model_unmap(model);
	if (st != SSLM_OK) {
		std::fprintf(stderr, "FAIL: sslm_model_unmap returned %d\n", static_cast<int>(st));
		return 1;
	}

	std::printf("t2139_c2_smoke: PASS\n");
	return 0;
}

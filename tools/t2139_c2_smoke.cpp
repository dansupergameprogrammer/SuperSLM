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
// Usage: t2139_c2_smoke.exe <path-to-real.sslm>
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
		std::vector<uint8_t> pool_raw(pool_buf_size);
		sslm_kv_pool pool = nullptr;
		st = sslm_kv_pool_create(model, pool_raw.data(), pool_raw.size(), block_count, &pool);
		if (st != SSLM_OK || !pool) {
			std::fprintf(stderr, "FAIL: sslm_kv_pool_create returned %d\n", static_cast<int>(st));
			sslm_model_unmap(model);
			return 1;
		}
		{
			sslm_kv_pool bad_pool = nullptr;
			st = sslm_kv_pool_create(model, pool_raw.data(), pool_buf_size - 1, block_count, &bad_pool);
			if (st != SSLM_BUFFER_TOO_SMALL) {
				std::fprintf(stderr, "FAIL: sslm_kv_pool_create(1-byte-short buffer) returned %d, expected SSLM_BUFFER_TOO_SMALL\n",
				              static_cast<int>(st));
				sslm_model_unmap(model);
				return 1;
			}
			std::printf("sslm_kv_pool_create hostile-input path: PASS (SSLM_BUFFER_TOO_SMALL fired as designed)\n");
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

	st = sslm_model_unmap(model);
	if (st != SSLM_OK) {
		std::fprintf(stderr, "FAIL: sslm_model_unmap returned %d\n", static_cast<int>(st));
		return 1;
	}

	std::printf("t2139_c2_smoke: PASS\n");
	return 0;
}

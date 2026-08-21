// t2132_m4_forced_token_count_pin.cpp -- M4 / D-SLM3486 (design Sec7.3, Claude/Vitruvius/
// t2133-layer1-c-abi-design-2026-08-16.md Sec7.3): forced_token_count now survives
// sslm_seq_save/sslm_seq_restore under the 'SSB2' blob format -- a sequence saved after real
// jump-forward-admitted (SSLM_SPAN_SCHEMA_CONTENT) tokens, restored into a fresh handle, must
// report the SAME forced_token_count via sslm_stats as the live sequence did at save time.
// Named precedent: tools/t2139_dim9_current_token_pin.cpp (design Sec10 dim 9's own round-trip
// cell for current_token, the prior 'SSB1' amendment).
//
// Also pins the 'SSB2' magic-rejection half of the ruling: a well-formed 'SSB1'-shaped blob (this
// pin hand-builds one, matching the PRE-fold 100-byte fixed-header layout byte-for-byte) is
// rejected by sslm_seq_restore on the magic check alone, SSLM_INVALID_ARGUMENT, never parsed as
// SSB2 and never left to default forced_token_count to 0 while accepting the rest of the blob.
//
// Usage: t2132_m4_forced_token_count_pin.exe <path-to-g5-fixture.sslm> [schema_name]
#include <cstdio>
#include <cstring>
#include <fstream>
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
}  // namespace

int main(int argc, char** argv) {
	if (argc < 2) {
		std::fprintf(stderr, "usage: %s <path-to-g5-fixture.sslm> [schema_name]\n", argv[0]);
		return 1;
	}
	const char* artifact_path = argv[1];
	const char* schema_name = argc >= 3 ? argv[2] : "shopkeeper_intent_extraction";

	std::vector<uint8_t> bytes;
	if (!ReadFile(artifact_path, &bytes)) {
		std::fprintf(stderr, "FAIL: could not read %s\n", artifact_path);
		return 1;
	}

	sslm_model model = nullptr;
	sslm_status st = sslm_model_map(bytes.data(), bytes.size(), &model);
	if (st != SSLM_OK || !model) {
		std::fprintf(stderr, "FAIL: sslm_model_map returned %d\n", static_cast<int>(st));
		return 1;
	}
	sslm_schema schema = nullptr;
	st = sslm_schema_lookup(model, schema_name, &schema);
	if (st != SSLM_OK || !schema) {
		std::fprintf(stderr, "FAIL: sslm_schema_lookup(\"%s\") returned %d\n", schema_name,
		             static_cast<int>(st));
		return 1;
	}

	const uint32_t block_count = 2;  // live + restored, concurrently resident (D-SLM3454)
	const size_t block_bytes = sslm_kv_block_size(model);
	const size_t overhead = sslm_kv_pool_overhead_size(model, block_count);
	const size_t pool_buf_size = block_bytes * block_count + overhead;
	std::vector<uint8_t> pool_storage(pool_buf_size + 63);
	void* pool_raw = pool_storage.data();
	size_t pool_raw_space = pool_storage.size();
	std::align(64, pool_buf_size, pool_raw, pool_raw_space);
	sslm_kv_pool pool = nullptr;
	st = sslm_kv_pool_create(model, pool_raw, pool_buf_size, block_count, &pool);
	if (st != SSLM_OK || !pool) {
		std::fprintf(stderr, "FAIL: sslm_kv_pool_create returned %d\n", static_cast<int>(st));
		return 1;
	}

	// --- build a real sequence, admit REAL jump-forward tokens under the schema ---
	sslm_seq seq = nullptr;
	st = sslm_seq_create(model, &pool, &seq);
	if (st != SSLM_OK || !seq) {
		std::fprintf(stderr, "FAIL: sslm_seq_create returned %d\n", static_cast<int>(st));
		return 1;
	}
	st = sslm_seq_set_schema(seq, schema);
	if (st != SSLM_OK) {
		std::fprintf(stderr, "FAIL: sslm_seq_set_schema returned %d\n", static_cast<int>(st));
		return 1;
	}

	// A real, schema-legal forced chain: decode a handful of unconstrained-under-the-mask steps
	// first (every one masked-valid by construction, since the schema is bound throughout), then
	// replay the first few PRODUCED tokens back in as a SSLM_SPAN_SCHEMA_CONTENT forced span --
	// exactly t2132_g5_gpu_parity_cpu.cpp's own Gate 2 derivation, reused here (a real forced
	// chain, not a fixture literal).
	const int32_t seed_prompt[] = {1};
	int32_t consumed = 0;
	st = sslm_prefill(model, seq, seed_prompt, 1, 1, SSLM_SPAN_PROMPT, nullptr, &consumed);
	if (st != SSLM_OK || consumed != 1) {
		std::fprintf(stderr, "FAIL: seed prefill returned %d consumed=%d\n", static_cast<int>(st),
		             consumed);
		return 1;
	}
	std::vector<int32_t> decoded;
	sslm_decode_params params{};
	// T-2199 Phase D review addendum (D-SLM3797, Dan): struct_size is the FIRST
	// new field the library now validates -- an unrecognized size is a loud rejection.
	params.struct_size = sizeof(params);
	params.layer_budget = 28;
	sslm_seq seqs[1] = {seq};
	for (int step = 0; step < 8 && decoded.size() < 3; ++step) {
		int32_t out_token = -1;
		int guard = 0;
		while (out_token < 0) {
			st = sslm_decode_step(model, seqs, 1, &params, nullptr, &out_token);
			if (st != SSLM_OK) {
				std::fprintf(stderr, "FAIL: sslm_decode_step (seeding forced chain) returned %d\n",
				             static_cast<int>(st));
				return 1;
			}
			if (++guard > 10000) {
				std::fprintf(stderr, "FAIL: decode step never completed\n");
				return 1;
			}
		}
		if (out_token >= 0) decoded.push_back(out_token);
	}
	if (decoded.size() < 3) {
		std::fprintf(stderr, "FAIL: could not derive 3 real decoded tokens to reuse as a forced chain\n");
		return 1;
	}
	sslm_seq_release(seq);

	// Fresh sequence: prefill the same seed, then admit the derived chain as a REAL forced
	// (SSLM_SPAN_SCHEMA_CONTENT) span -- this is what actually increments forced_token_count.
	st = sslm_seq_create(model, &pool, &seq);
	if (st != SSLM_OK || !seq) {
		std::fprintf(stderr, "FAIL: sslm_seq_create (2) returned %d\n", static_cast<int>(st));
		return 1;
	}
	st = sslm_seq_set_schema(seq, schema);
	if (st != SSLM_OK) {
		std::fprintf(stderr, "FAIL: sslm_seq_set_schema (2) returned %d\n", static_cast<int>(st));
		return 1;
	}
	consumed = 0;
	st = sslm_prefill(model, seq, seed_prompt, 1, 1, SSLM_SPAN_PROMPT, nullptr, &consumed);
	if (st != SSLM_OK || consumed != 1) {
		std::fprintf(stderr, "FAIL: seed prefill (2) returned %d consumed=%d\n", static_cast<int>(st),
		             consumed);
		return 1;
	}
	int32_t forced_consumed = 0;
	st = sslm_prefill(model, seq, decoded.data(), static_cast<int32_t>(decoded.size()),
	                   static_cast<int32_t>(decoded.size()), SSLM_SPAN_SCHEMA_CONTENT, nullptr,
	                   &forced_consumed);
	if (st != SSLM_OK || forced_consumed != static_cast<int32_t>(decoded.size())) {
		std::fprintf(stderr, "FAIL: forced-chain prefill returned %d consumed=%d\n",
		             static_cast<int>(st), forced_consumed);
		return 1;
	}

	sslm_stats_out live_stats{};
	st = sslm_stats(model, seq, &live_stats);
	if (st != SSLM_OK) {
		std::fprintf(stderr, "FAIL: sslm_stats (live) returned %d\n", static_cast<int>(st));
		return 1;
	}
	std::printf("live sequence forced_token_count after %zu real forced tokens: %lld\n",
	            decoded.size(), static_cast<long long>(live_stats.forced_token_count));
	if (live_stats.forced_token_count != static_cast<int64_t>(decoded.size())) {
		std::fprintf(stderr,
		             "FAIL: live forced_token_count (%lld) != tokens actually admitted (%zu)\n",
		             static_cast<long long>(live_stats.forced_token_count), decoded.size());
		return 1;
	}

	// --- save (real 'SSB2' blob), restore into a fresh handle, compare via sslm_stats ---
	size_t required = 0;
	sslm_seq_save(seq, nullptr, &required);
	std::vector<uint8_t> blob(required);
	size_t n = blob.size();
	st = sslm_seq_save(seq, blob.data(), &n);
	if (st != SSLM_OK) {
		std::fprintf(stderr, "FAIL: sslm_seq_save returned %d\n", static_cast<int>(st));
		return 1;
	}
	blob.resize(n);
	std::printf("saved 'SSB2' blob: %zu bytes (magic '%c%c%c%c')\n", n, blob[0], blob[1], blob[2],
	            blob[3]);

	int failures = 0;

	sslm_seq restored = nullptr;
	st = sslm_seq_restore(model, &pool, blob.data(), blob.size(), &restored);
	if (st != SSLM_OK || !restored) {
		std::fprintf(stderr, "FAIL: sslm_seq_restore returned %d\n", static_cast<int>(st));
		return 1;
	}
	sslm_stats_out restored_stats{};
	st = sslm_stats(model, restored, &restored_stats);
	if (st != SSLM_OK) {
		std::fprintf(stderr, "FAIL: sslm_stats (restored) returned %d\n", static_cast<int>(st));
		return 1;
	}
	const bool round_trip_ok = restored_stats.forced_token_count == live_stats.forced_token_count;
	std::printf(
	    "THE PIN -- forced_token_count round-trip: live=%lld restored=%lld -- %s\n",
	    static_cast<long long>(live_stats.forced_token_count),
	    static_cast<long long>(restored_stats.forced_token_count), round_trip_ok ? "PASS" : "FAIL");
	if (!round_trip_ok) ++failures;
	sslm_seq_release(restored);

	// --- the 'SSB1'-rejection half: a hand-built, well-formed PRE-fold blob (100-byte fixed
	// header, no forced_token_count field, magic 'SSB1') must be rejected outright on the magic
	// check, never parsed as SSB2. ---
	{
		// Slice the real SSB2 blob apart and reassemble it as a byte-exact SSB1-shaped blob: the
		// first 92 bytes (magic through kv_saturation_count's own first 92 bytes... actually
		// through offset 92, i.e. magic(4)+hash(32)+kv_precision(4)+schema_name_hash(8)+
		// dfa_walk_state(4)+adapter_binding_id(8)+context_length(8)+layer_index(4)+
		// current_token(4)+hidden_scale(16) = 92 bytes) are IDENTICAL in both formats (SSB2 only
		// inserts forced_token_count AFTER kv_saturation_count, at byte 100); kv_saturation_count
		// itself is bytes [92,100) in both. The SSB1 shape is: those first 100 bytes, magic
		// overwritten to 'SSB1', immediately followed by SSB2's OWN residual/kv_block_count/
		// kv_blocks tail (blob[108:]) -- skipping SSB2's own forced_token_count field at [100,108)
		// entirely, exactly what a real pre-fold writer would have produced for the same live
		// sequence state.
		if (blob.size() < 108) {
			std::fprintf(stderr, "FAIL: SSB2 blob unexpectedly short (%zu bytes)\n", blob.size());
			return 1;
		}
		std::vector<uint8_t> ssb1_shaped;
		ssb1_shaped.insert(ssb1_shaped.end(), blob.begin(), blob.begin() + 100);
		ssb1_shaped[0] = 'S';
		ssb1_shaped[1] = 'S';
		ssb1_shaped[2] = 'B';
		ssb1_shaped[3] = '1';
		ssb1_shaped.insert(ssb1_shaped.end(), blob.begin() + 108, blob.end());

		sslm_seq should_be_null = nullptr;
		st = sslm_seq_restore(model, &pool, ssb1_shaped.data(), ssb1_shaped.size(), &should_be_null);
		const bool rejected_ok = (st == SSLM_INVALID_ARGUMENT) && (should_be_null == nullptr);
		std::printf("SSB1-shaped blob (magic 'SSB1', 100-byte fixed header): status=%d restored=%p -- %s\n",
		            static_cast<int>(st), static_cast<void*>(should_be_null),
		            rejected_ok ? "PASS" : "FAIL");
		if (!rejected_ok) {
			++failures;
			if (should_be_null) sslm_seq_release(should_be_null);
		}
	}

	sslm_seq_release(seq);
	sslm_kv_pool_destroy(pool);
	sslm_model_unmap(model);

	if (failures > 0) {
		std::fprintf(stderr, "t2132_m4_forced_token_count_pin: FAIL (%d cell(s) failed)\n", failures);
		return 1;
	}
	std::printf("t2132_m4_forced_token_count_pin: PASS\n");
	return 0;
}

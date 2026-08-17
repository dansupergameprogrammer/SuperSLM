// t2132_s7_set_schema_freshness_pin.cpp -- S7 (Claude/Poirot/9bc9ec6-t2132-g5-arc-review.md):
// sslm_seq_set_schema's guard now matches its own shipped header's condition ("valid ONLY when
// the sequence's DFA-walk state is at its start -- a fresh sslm_seq_create, or immediately after
// sslm_seq_reset", design Sec5, ABI surface) via `current_token == -1`, mirroring
// sslm_prefix_set_schema's own already-correct, identically-shaped check. Before this fix,
// `dfa_walk_state` alone (kDfaWalkStateUnused forever on an UNBOUND sequence, however much
// unconstrained content had already been prefilled/decoded on it) could not detect a first-time
// bind attempted after real generation had already happened -- accepted where the header says it
// must reject.
//
// Usage: t2132_s7_set_schema_freshness_pin.exe <path-to-g5-fixture.sslm> [schema_name]
#include <cstdio>
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

	const uint32_t block_count = 1;
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

	int failures = 0;

	// (a) control -- immediately after sslm_seq_create, set_schema succeeds.
	{
		sslm_seq seq = nullptr;
		st = sslm_seq_create(model, &pool, &seq);
		if (st != SSLM_OK || !seq) {
			std::fprintf(stderr, "FAIL: sslm_seq_create (a) returned %d\n", static_cast<int>(st));
			return 1;
		}
		st = sslm_seq_set_schema(seq, schema);
		const bool pass = (st == SSLM_OK);
		std::printf("(a) control -- set_schema immediately after create: status=%d -- %s\n",
		            static_cast<int>(st), pass ? "PASS" : "FAIL");
		if (!pass) ++failures;
		sslm_seq_release(seq);
	}

	// (b) THE FIX: prefill a real prompt (unconstrained -- no schema ever bound), so
	// current_token != -1 and real generation has genuinely happened, THEN attempt a first-time
	// bind. Pre-fix, dfa_walk_state stayed at kDfaWalkStateUnused (nothing had touched it, since
	// no schema was ever bound), so the OLD guard would have accepted this -- not "at its start"
	// by the header's own two named examples (fresh create / immediately after reset), neither
	// of which this sequence is.
	{
		sslm_seq seq = nullptr;
		st = sslm_seq_create(model, &pool, &seq);
		if (st != SSLM_OK || !seq) {
			std::fprintf(stderr, "FAIL: sslm_seq_create (b) returned %d\n", static_cast<int>(st));
			return 1;
		}
		const int32_t prompt_tokens[] = {1, 2, 3};
		int32_t consumed = 0;
		st = sslm_prefill(model, seq, prompt_tokens, 3, 3, SSLM_SPAN_PROMPT, nullptr, &consumed);
		if (st != SSLM_OK || consumed != 3) {
			std::fprintf(stderr, "FAIL: sslm_prefill (b) returned %d consumed=%d\n",
			             static_cast<int>(st), consumed);
			return 1;
		}
		st = sslm_seq_set_schema(seq, schema);
		const bool pass = (st == SSLM_SCHEMA_BIND_REJECTED);
		std::printf(
		    "(b) THE FIX -- set_schema after real (unconstrained) generation: status=%d -- %s\n",
		    static_cast<int>(st), pass ? "PASS" : "FAIL");
		if (!pass) ++failures;
		sslm_seq_release(seq);
	}

	// (c) sslm_seq_reset genuinely re-freshens the sequence -- set_schema after reset succeeds,
	// proving (b) is not a blanket rejection of every post-generation sequence, only the
	// non-fresh ones the header actually names as invalid.
	{
		sslm_seq seq = nullptr;
		st = sslm_seq_create(model, &pool, &seq);
		if (st != SSLM_OK || !seq) {
			std::fprintf(stderr, "FAIL: sslm_seq_create (c) returned %d\n", static_cast<int>(st));
			return 1;
		}
		const int32_t prompt_tokens[] = {1, 2, 3};
		int32_t consumed = 0;
		st = sslm_prefill(model, seq, prompt_tokens, 3, 3, SSLM_SPAN_PROMPT, nullptr, &consumed);
		if (st != SSLM_OK || consumed != 3) {
			std::fprintf(stderr, "FAIL: sslm_prefill (c) returned %d consumed=%d\n",
			             static_cast<int>(st), consumed);
			return 1;
		}
		st = sslm_seq_reset(seq);
		if (st != SSLM_OK) {
			std::fprintf(stderr, "FAIL: sslm_seq_reset (c) returned %d\n", static_cast<int>(st));
			return 1;
		}
		st = sslm_seq_set_schema(seq, schema);
		const bool pass = (st == SSLM_OK);
		std::printf("(c) control -- set_schema immediately after reset: status=%d -- %s\n",
		            static_cast<int>(st), pass ? "PASS" : "FAIL");
		if (!pass) ++failures;
		sslm_seq_release(seq);
	}

	sslm_kv_pool_destroy(pool);
	sslm_model_unmap(model);

	if (failures > 0) {
		std::fprintf(stderr, "t2132_s7_set_schema_freshness_pin: FAIL (%d cells failed)\n", failures);
		return 1;
	}
	std::printf("t2132_s7_set_schema_freshness_pin: PASS\n");
	return 0;
}

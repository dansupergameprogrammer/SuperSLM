// t2132_c2_cross_model_schema_pin.cpp -- C2 (Claude/Poirot/9bc9ec6-t2132-g5-arc-review.md):
// sslm_seq_set_schema / sslm_prefix_set_schema now check the handle's owning model
// (sslm_schema_s::model) before binding -- executed against two REAL, independently-mapped
// artifacts: a schema handle resolved from model A (the 1.5B G5 fixture, real compiled schema)
// bound onto a plain sequence created on model B (the 0.5B artifact) must reject.
//
// Usage: t2132_c2_cross_model_schema_pin.exe <path-to-1.5b-g5-fixture.sslm>
//        <path-to-0.5b-plain.sslm> [schema_name]
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
	if (argc < 3) {
		std::fprintf(stderr,
		              "usage: %s <path-to-1.5b-g5-fixture.sslm> <path-to-0.5b-plain.sslm> "
		              "[schema_name]\n",
		              argv[0]);
		return 1;
	}
	const char* model_a_path = argv[1];
	const char* model_b_path = argv[2];
	const char* schema_name = argc >= 4 ? argv[3] : "shopkeeper_intent_extraction";

	std::vector<uint8_t> bytes_a, bytes_b;
	if (!ReadFile(model_a_path, &bytes_a)) {
		std::fprintf(stderr, "FAIL: could not read %s\n", model_a_path);
		return 1;
	}
	if (!ReadFile(model_b_path, &bytes_b)) {
		std::fprintf(stderr, "FAIL: could not read %s\n", model_b_path);
		return 1;
	}

	sslm_model model_a = nullptr, model_b = nullptr;
	sslm_status st = sslm_model_map(bytes_a.data(), bytes_a.size(), &model_a);
	if (st != SSLM_OK || !model_a) {
		std::fprintf(stderr, "FAIL: sslm_model_map(A) returned %d\n", static_cast<int>(st));
		return 1;
	}
	st = sslm_model_map(bytes_b.data(), bytes_b.size(), &model_b);
	if (st != SSLM_OK || !model_b) {
		std::fprintf(stderr, "FAIL: sslm_model_map(B) returned %d\n", static_cast<int>(st));
		return 1;
	}

	sslm_schema schema_a = nullptr;
	st = sslm_schema_lookup(model_a, schema_name, &schema_a);
	if (st != SSLM_OK || !schema_a) {
		std::fprintf(stderr, "FAIL: sslm_schema_lookup(A, \"%s\") returned %d\n", schema_name,
		             static_cast<int>(st));
		return 1;
	}

	auto MakePool = [](sslm_model m, sslm_kv_pool* out_pool, std::vector<uint8_t>* keepalive) -> bool {
		const uint32_t block_count = 1;
		const size_t block_bytes = sslm_kv_block_size(m);
		const size_t overhead = sslm_kv_pool_overhead_size(m, block_count);
		const size_t pool_buf_size = block_bytes * block_count + overhead;
		keepalive->assign(pool_buf_size + 63, 0);
		void* pool_raw = keepalive->data();
		size_t pool_raw_space = keepalive->size();
		std::align(64, pool_buf_size, pool_raw, pool_raw_space);
		return sslm_kv_pool_create(m, pool_raw, pool_buf_size, block_count, out_pool) == SSLM_OK &&
		       *out_pool != nullptr;
	};

	std::vector<uint8_t> pool_b_storage;
	sslm_kv_pool pool_b = nullptr;
	if (!MakePool(model_b, &pool_b, &pool_b_storage)) {
		std::fprintf(stderr, "FAIL: sslm_kv_pool_create(B) failed\n");
		return 1;
	}

	int failures = 0;

	// (a) sslm_seq_set_schema: schema_a (owned by model_a) bound onto a plain sequence created
	// on model_b -- must reject with SSLM_INVALID_ARGUMENT (this file's own existing cross-
	// handle-mismatch family: sslm_prefix_begin/sslm_seq_create/sslm_seq_adopt_prefix/
	// sslm_seq_restore all already return this status for a pool/model/prefix identity
	// mismatch), and the sequence's own bound_schema/dfa_walk_state must stay untouched
	// (checkable indirectly: a SUBSEQUENT bind of SSLM_SCHEMA_NONE -- an unbind -- must still
	// succeed, which it would not if the rejected call had left dfa_walk_state in a state that
	// then confuses a later legitimate call).
	{
		sslm_seq seq_b = nullptr;
		st = sslm_seq_create(model_b, &pool_b, &seq_b);
		if (st != SSLM_OK || !seq_b) {
			std::fprintf(stderr, "FAIL: sslm_seq_create(B) returned %d\n", static_cast<int>(st));
			return 1;
		}
		st = sslm_seq_set_schema(seq_b, schema_a);
		const bool pass = (st == SSLM_INVALID_ARGUMENT);
		std::printf("(a) sslm_seq_set_schema(model_b sequence, model_a schema): status=%d -- %s\n",
		            static_cast<int>(st), pass ? "PASS" : "FAIL");
		if (!pass) ++failures;

		// Confirm the rejected call left the sequence in its own fresh, still-usable state --
		// SSLM_SCHEMA_NONE unbind (a no-op on an already-unbound sequence) must still succeed.
		const sslm_status unbind_st = sslm_seq_set_schema(seq_b, SSLM_SCHEMA_NONE);
		const bool unbind_pass = (unbind_st == SSLM_OK);
		std::printf("    post-reject state still usable (unbind no-op): status=%d -- %s\n",
		            static_cast<int>(unbind_st), unbind_pass ? "PASS" : "FAIL");
		if (!unbind_pass) ++failures;

		sslm_seq_release(seq_b);
	}

	// (b) sslm_prefix_set_schema: the identical mismatch on a prefix-under-construction.
	{
		sslm_prefix prefix_b = nullptr;
		st = sslm_prefix_begin(model_b, &pool_b, &prefix_b);
		if (st != SSLM_OK || !prefix_b) {
			std::fprintf(stderr, "FAIL: sslm_prefix_begin(B) returned %d\n", static_cast<int>(st));
			return 1;
		}
		st = sslm_prefix_set_schema(prefix_b, schema_a);
		const bool pass = (st == SSLM_INVALID_ARGUMENT);
		std::printf("(b) sslm_prefix_set_schema(model_b prefix, model_a schema): status=%d -- %s\n",
		            static_cast<int>(st), pass ? "PASS" : "FAIL");
		if (!pass) ++failures;
		sslm_prefix_release(prefix_b);
	}

	// (c) same-model bind, as a control -- must SUCCEED, proving (a)/(b) reject the cross-model
	// case specifically and not schema binding in general.
	{
		sslm_seq seq_a = nullptr;
		std::vector<uint8_t> pool_a_storage;
		sslm_kv_pool pool_a = nullptr;
		if (!MakePool(model_a, &pool_a, &pool_a_storage)) {
			std::fprintf(stderr, "FAIL: sslm_kv_pool_create(A) failed\n");
			return 1;
		}
		st = sslm_seq_create(model_a, &pool_a, &seq_a);
		if (st != SSLM_OK || !seq_a) {
			std::fprintf(stderr, "FAIL: sslm_seq_create(A) returned %d\n", static_cast<int>(st));
			return 1;
		}
		st = sslm_seq_set_schema(seq_a, schema_a);
		const bool pass = (st == SSLM_OK);
		std::printf("(c) control -- same-model bind (model_a sequence, model_a schema): status=%d -- %s\n",
		            static_cast<int>(st), pass ? "PASS" : "FAIL");
		if (!pass) ++failures;
		sslm_seq_release(seq_a);
		sslm_kv_pool_destroy(pool_a);
	}

	sslm_kv_pool_destroy(pool_b);
	sslm_model_unmap(model_a);
	sslm_model_unmap(model_b);

	if (failures > 0) {
		std::fprintf(stderr, "t2132_c2_cross_model_schema_pin: FAIL (%d cells failed)\n", failures);
		return 1;
	}
	std::printf("t2132_c2_cross_model_schema_pin: PASS\n");
	return 0;
}

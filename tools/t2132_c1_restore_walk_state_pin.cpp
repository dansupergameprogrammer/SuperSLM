// t2132_c1_restore_walk_state_pin.cpp -- C1 (Claude/Poirot/9bc9ec6-t2132-g5-arc-review.md):
// sslm_seq_restore's own restored dfa_walk_state now bounded against the resolved schema's
// state_count before it is ever used to index mask_pages/state_offsets_le -- executed against a
// corrupted blob built from a REAL save (real 1.5B fixture, real compiled schema), not a
// synthetic construction.
//
// Two hostile constructions, both rejected:
//   (a) an out-of-range walk-state (larger than the schema's own real state_count) -- the
//       out-of-bounds read C1 named, at whatever offset `walk_state * mask_page_bytes` lands.
//   (b) the save-side sentinel (kDfaWalkStateUnused, 0xFFFFFFFF) written where a BOUND schema's
//       real state id belongs -- accepted as a state id and multiplied before this fix; C1's
//       own "worst form" of the same defect.
//
// Usage: t2132_c1_restore_walk_state_pin.exe <path-to-g5-fixture.sslm> [schema_name]
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

// Blob field offsets -- design Sec13.4 / src/sslm_abi.cpp's own save-blob header comment
// ("magic(4) + model_hash(32) + kv_precision(4) + schema_name_hash(8) + dfa_walk_state(4) +
// ..."): magic=0, model_hash=4, kv_precision=36, schema_name_hash=40, dfa_walk_state=48.
constexpr size_t kDfaWalkStateOffset = 48;

void WriteLE32At(std::vector<uint8_t>* blob, size_t off, uint32_t v) {
	(*blob)[off + 0] = static_cast<uint8_t>(v);
	(*blob)[off + 1] = static_cast<uint8_t>(v >> 8);
	(*blob)[off + 2] = static_cast<uint8_t>(v >> 16);
	(*blob)[off + 3] = static_cast<uint8_t>(v >> 24);
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

	// A real bound-schema sequence, saved at its own fresh (state 0) walk-state -- gives a real,
	// well-formed blob whose dfa_walk_state field this pin then corrupts.
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

	size_t required = 0;
	sslm_seq_save(seq, nullptr, &required);
	std::vector<uint8_t> valid_blob(required);
	size_t n = valid_blob.size();
	st = sslm_seq_save(seq, valid_blob.data(), &n);
	if (st != SSLM_OK) {
		std::fprintf(stderr, "FAIL: sslm_seq_save returned %d\n", static_cast<int>(st));
		return 1;
	}
	valid_blob.resize(n);
	sslm_seq_release(seq);

	// Sanity: the un-corrupted blob restores cleanly -- proves the corruption below, not some
	// unrelated blob malformation, is what triggers the rejection.
	{
		sslm_seq restored = nullptr;
		st = sslm_seq_restore(model, &pool, valid_blob.data(), valid_blob.size(), &restored);
		if (st != SSLM_OK || !restored) {
			std::fprintf(stderr,
			             "FAIL: sanity check -- the UNCORRUPTED blob failed to restore (status %d)\n",
			             static_cast<int>(st));
			return 1;
		}
		sslm_seq_release(restored);
		std::printf("sanity: uncorrupted blob restores cleanly: PASS\n");
	}

	int failures = 0;

	// (a) out-of-range walk-state: state_count for the real compiled reference schema is 594
	// (design Sec9.10.1's own cited figure) -- comfortably out of range, and large enough that
	// `walk_state * mask_page_bytes` (18,992-byte stride for this fixture's own vocab width)
	// lands well past the mapped mask-pages region.
	{
		std::vector<uint8_t> corrupt = valid_blob;
		WriteLE32At(&corrupt, kDfaWalkStateOffset, 999999u);
		sslm_seq restored = nullptr;
		st = sslm_seq_restore(model, &pool, corrupt.data(), corrupt.size(), &restored);
		const bool pass = (st == SSLM_RESTORE_SCHEMA_MISMATCH) && (restored == nullptr);
		std::printf("(a) out-of-range walk-state (999999): status=%d restored=%p -- %s\n",
		            static_cast<int>(st), static_cast<void*>(restored), pass ? "PASS" : "FAIL");
		if (!pass) {
			++failures;
			if (restored) sslm_seq_release(restored);
		}
	}

	// (b) the save-side sentinel (kDfaWalkStateUnused, 0xFFFFFFFF) where a BOUND schema's real
	// state id belongs -- schema_name_hash on this blob is nonzero (a real schema IS bound), so
	// the sentinel here is exactly the malformed-blob case C1 names, not the legitimate unbound
	// case (that one carries schema_name_hash == 0, exercised by (c) below).
	{
		std::vector<uint8_t> corrupt = valid_blob;
		WriteLE32At(&corrupt, kDfaWalkStateOffset, 0xFFFFFFFFu);
		sslm_seq restored = nullptr;
		st = sslm_seq_restore(model, &pool, corrupt.data(), corrupt.size(), &restored);
		const bool pass = (st == SSLM_RESTORE_SCHEMA_MISMATCH) && (restored == nullptr);
		std::printf("(b) sentinel-as-bound-state (0xFFFFFFFF): status=%d restored=%p -- %s\n",
		            static_cast<int>(st), static_cast<void*>(restored), pass ? "PASS" : "FAIL");
		if (!pass) {
			++failures;
			if (restored) sslm_seq_release(restored);
		}
	}

	// (c) O3 closed: schema_name_hash == 0 (unbound) with a NON-sentinel walk-state is the
	// symmetric malformation on the other side of the same field -- also rejected.
	{
		std::vector<uint8_t> corrupt = valid_blob;
		// Force schema_name_hash (offset 40, 8 bytes LE) to 0 -- "no schema bound" -- while
		// leaving dfa_walk_state (offset 48) at whatever real, non-sentinel value the fresh bind
		// above wrote (0, this fixture's own start state) -- a real, non-sentinel value on an
		// unbound blob.
		for (int k = 0; k < 8; ++k) corrupt[40 + k] = 0;
		sslm_seq restored = nullptr;
		st = sslm_seq_restore(model, &pool, corrupt.data(), corrupt.size(), &restored);
		const bool pass = (st == SSLM_RESTORE_SCHEMA_MISMATCH) && (restored == nullptr);
		std::printf("(c) O3 -- unbound blob with non-sentinel walk-state: status=%d restored=%p -- %s\n",
		            static_cast<int>(st), static_cast<void*>(restored), pass ? "PASS" : "FAIL");
		if (!pass) {
			++failures;
			if (restored) sslm_seq_release(restored);
		}
	}

	sslm_kv_pool_destroy(pool);
	sslm_model_unmap(model);

	if (failures > 0) {
		std::fprintf(stderr, "t2132_c1_restore_walk_state_pin: FAIL (%d/3 cells failed)\n", failures);
		return 1;
	}
	std::printf("t2132_c1_restore_walk_state_pin: PASS (3/3)\n");
	return 0;
}

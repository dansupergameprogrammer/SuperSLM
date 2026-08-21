// t2132_s2_dead_end_sentinel_pin.cpp -- D-SLM3476 (design Sec14.1, Claude/Poirot/
// 9bc9ec6-t2132-g5-arc-review.md S2): sslm_decode_step now reserves out_tokens[i] == -2 for a
// sequence whose schema-bound walk has reached a state with NO legal continuation at all
// (Transition finds no matching CSR entry for the masked-argmax-selected token -- which Sec13.2
// permits only at an accepting state with a legitimately all-zero mask page). Before this fix
// the failure was silently discarded and the walk froze, emitting token 0 forever at SSLM_OK.
//
// Executed against the REAL compiled reference schema, not a hand-built artifact: parses the
// artifact's own SchemaMasks section directly (schema_masks.h, the same collision-free parse the
// GPU-parity tool's own diagnostic block already uses) and BFS's the REAL compiled DFA for a
// reachable state whose own transition row is genuinely empty -- the natural, real "all-zero
// accepting page" construction the casebook names, found rather than fabricated. If the real
// 594-state reference schema happens to have no such state (every reachable state has at least
// one legal continuation), this pin SKIPs its own product half rather than fabricate one --
// StandardsDocument.md Sec5.4's own real-workload convention (see fixture_common.h's SKIP_MSG).
//
// Usage: t2132_s2_dead_end_sentinel_pin.exe <path-to-g5-fixture.sslm> [schema_name]
#include <cstdio>
#include <deque>
#include <fstream>
#include <string>
#include <vector>

#include "superslm/model.h"
#include "superslm/schema_masks.h"
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

uint32_t RowBegin(const superslm::SchemaEntry& e, uint32_t s) {
	const uint8_t* p = e.state_offsets_le + static_cast<size_t>(s) * 4;
	return static_cast<uint32_t>(p[0]) | (static_cast<uint32_t>(p[1]) << 8) |
	       (static_cast<uint32_t>(p[2]) << 16) | (static_cast<uint32_t>(p[3]) << 24);
}

// BFS from state 0 (the schema's own start state -- prompt tokens never touch it, so every real
// sequence's walk begins there) for the shortest path of LEGAL forced tokens to any state whose
// own CSR row is empty (RowBegin(s) == RowBegin(s+1), the state_offsets_le width Sec13.2's
// all-zero-page clause names). Returns true and fills `out_path` on success.
bool FindDeadEndPath(const superslm::SchemaMasksTable& table, const superslm::SchemaEntry& e,
                      std::vector<int32_t>* out_path) {
	std::vector<bool> visited(e.state_count, false);
	std::deque<std::pair<uint32_t, std::vector<int32_t>>> q;
	q.push_back({0u, {}});
	visited[0] = true;
	while (!q.empty()) {
		auto [state, path] = q.front();
		q.pop_front();
		const uint32_t begin = RowBegin(e, state);
		const uint32_t end = RowBegin(e, state + 1);
		if (begin == end) {
			*out_path = path;
			return true;
		}
		for (uint32_t k = begin; k < end; ++k) {
			const uint8_t* row = e.transitions_le + static_cast<size_t>(k) * 8;
			const uint32_t tok = static_cast<uint32_t>(row[0]) | (static_cast<uint32_t>(row[1]) << 8) |
			                      (static_cast<uint32_t>(row[2]) << 16) |
			                      (static_cast<uint32_t>(row[3]) << 24);
			const uint32_t next = static_cast<uint32_t>(row[4]) | (static_cast<uint32_t>(row[5]) << 8) |
			                       (static_cast<uint32_t>(row[6]) << 16) |
			                       (static_cast<uint32_t>(row[7]) << 24);
			if (next < visited.size() && !visited[next]) {
				visited[next] = true;
				std::vector<int32_t> next_path = path;
				next_path.push_back(static_cast<int32_t>(tok));
				q.push_back({next, next_path});
			}
		}
	}
	return false;
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

	std::vector<int32_t> dead_end_path;
	{
		superslm::SslmModelView view;
		std::string load_err;
		if (superslm::SslmModel::Load(bytes.data(), bytes.size(), view, &load_err) !=
		    superslm::SslmModelStatus::Ok) {
			std::fprintf(stderr, "FAIL: SslmModel::Load returned: %s\n", load_err.c_str());
			return 1;
		}
		const superslm::SslmSectionView* sec = view.Section(superslm::SslmSectionType::SchemaMasks);
		if (!sec) {
			std::fprintf(stderr, "FAIL: artifact carries no SchemaMasks section\n");
			return 1;
		}
		superslm::SchemaMasksTable table;
		std::string perr;
		if (!superslm::SchemaMasksTable::Parse(sec->data, sec->byte_size, view.config.vocab_size,
		                                        table, &perr)) {
			std::fprintf(stderr, "FAIL: SchemaMasksTable::Parse returned: %s\n", perr.c_str());
			return 1;
		}
		size_t idx = 0;
		const superslm::SchemaEntry* entry = table.ByName(schema_name, &idx);
		if (!entry) {
			std::fprintf(stderr, "FAIL: schema \"%s\" not found\n", schema_name);
			return 1;
		}
		std::printf("real compiled schema \"%s\": state_count=%u\n", schema_name, entry->state_count);
		if (!FindDeadEndPath(table, *entry, &dead_end_path)) {
			std::printf(
			    "SKIP: no reachable state with an empty transition row found in the real "
			    "compiled schema (every reachable state has at least one legal continuation) -- "
			    "no real input to drive the -2 path against. Not a Gate failure.\n");
			return 0;
		}
		std::printf("dead-end state found, reachable via %zu forced token(s)\n", dead_end_path.size());
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
		std::fprintf(stderr, "FAIL: sslm_schema_lookup returned %d\n", static_cast<int>(st));
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

	int failures = 0;
	if (dead_end_path.empty()) {
		// The start state IS the dead end -- seed one arbitrary SSLM_SPAN_PROMPT token (never
		// touches the walk) so sslm_decode_step has prior content to resume from.
		const int32_t seed[] = {1};
		int32_t consumed = 0;
		st = sslm_prefill(model, seq, seed, 1, 1, SSLM_SPAN_PROMPT, nullptr, &consumed);
		if (st != SSLM_OK || consumed != 1) {
			std::fprintf(stderr, "FAIL: seed prefill returned %d consumed=%d\n", static_cast<int>(st),
			             consumed);
			return 1;
		}
	} else {
		int32_t consumed = 0;
		st = sslm_prefill(model, seq, dead_end_path.data(), static_cast<int32_t>(dead_end_path.size()),
		                   static_cast<int32_t>(dead_end_path.size()), SSLM_SPAN_SCHEMA_CONTENT, nullptr,
		                   &consumed);
		const bool drove_ok = (st == SSLM_OK) && (consumed == static_cast<int32_t>(dead_end_path.size()));
		std::printf("drive to dead-end state via %zu forced token(s): status=%d consumed=%d -- %s\n",
		            dead_end_path.size(), static_cast<int>(st), consumed, drove_ok ? "PASS" : "FAIL");
		if (!drove_ok) {
			std::fprintf(stderr, "FAIL: could not drive to the dead-end state\n");
			return 1;
		}
	}

	// THE PIN: decode_step at the dead-end state must return -2, SSLM_OK overall, and change
	// nothing about the sequence's own schema state (checkable: calling again returns -2 again,
	// deterministically -- a torn/advancing state would not).
	sslm_decode_params params{};
	// T-2199 Phase D review addendum (D-SLM3797, Dan): struct_size is the FIRST
	// new field the library now validates -- an unrecognized size is a loud rejection.
	params.struct_size = sizeof(params);
	params.layer_budget = 28;
	sslm_seq seqs[1] = {seq};
	int32_t out_token = -1;
	st = sslm_decode_step(model, seqs, 1, &params, nullptr, &out_token);
	const bool pass1 = (st == SSLM_OK) && (out_token == -2);
	std::printf("decode_step at dead-end state: status=%d out_token=%d -- %s\n", static_cast<int>(st),
	            out_token, pass1 ? "PASS" : "FAIL");
	if (!pass1) ++failures;

	int32_t out_token2 = -1;
	st = sslm_decode_step(model, seqs, 1, &params, nullptr, &out_token2);
	const bool pass2 = (st == SSLM_OK) && (out_token2 == -2);
	std::printf("decode_step again (deterministic, resumable stop): status=%d out_token=%d -- %s\n",
	            static_cast<int>(st), out_token2, pass2 ? "PASS" : "FAIL");
	if (!pass2) ++failures;

	sslm_seq_release(seq);
	sslm_kv_pool_destroy(pool);
	sslm_model_unmap(model);

	if (failures > 0) {
		std::fprintf(stderr, "t2132_s2_dead_end_sentinel_pin: FAIL (%d cells failed)\n", failures);
		return 1;
	}
	std::printf("t2132_s2_dead_end_sentinel_pin: PASS\n");
	return 0;
}

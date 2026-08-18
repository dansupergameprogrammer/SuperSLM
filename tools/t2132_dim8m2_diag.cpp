// t2132_dim8m2_diag.cpp -- T-2132 (Brunel) diagnostic, disposable. Isolates the G5-6 mechanism
// cell 2 (prefix-sharing x schema) divergence Curie found (Claude/Curie/
// t2130-g5-red-suite-composition-joins-2026-08-17.md): seq_with_prefix's adopted-prefix K/V for
// the shared system-prompt span does not bit-match seq_no_prefix's directly-computed K/V for the
// identical tokens, first differing byte ~1152 (within token 0's own K/V span).
//
// This tool strips away every variable Curie's cell exercises together (6 PROMPT tokens, 3
// SCHEMA_CONTENT tokens, a bound schema) down to the single narrowest case: ONE token, no
// schema at all, prefix-adopt path vs direct-prefill path. If these two still diverge, the bug
// is in the prefix/adopt machinery itself, not in schema/multi-token interaction.
//
// Usage: t2132_dim8m2_diag.exe <path-to-g5-fixture.sslm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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

[[noreturn]] void Fail(const char* what, int status) {
	std::fprintf(stderr, "DIAG FAILED at %s: sslm_status=%d\n", what, status);
	std::exit(1);
}

// Mirrors RealKvPool (fixture_common.h), standalone so this tool has no test-tree dependency.
struct Pool {
	std::vector<uint8_t> storage;
	sslm_kv_pool pool = nullptr;
	bool Create(sslm_model model, uint32_t block_count) {
		const size_t block_bytes = sslm_kv_block_size(model);
		const size_t overhead = sslm_kv_pool_overhead_size(model, block_count);
		if (block_bytes == 0 || overhead == 0) return false;
		const size_t needed = block_bytes * (size_t)block_count + overhead;
		storage.assign(needed + 63, 0);
		void* aligned = storage.data();
		size_t space = storage.size();
		std::align(64, needed, aligned, space);
		if (!aligned) return false;
		return sslm_kv_pool_create(model, aligned, needed, block_count, &pool) == SSLM_OK && pool;
	}
	~Pool() { if (pool) sslm_kv_pool_destroy(pool); }
};

void DumpFirstDiff(const std::vector<uint8_t>& a, const std::vector<uint8_t>& b) {
	const size_t n = a.size();
	size_t first = (size_t)-1;
	size_t ndiff = 0;
	// Classifies every differing byte as "poison-shaped" (one side is exactly 0xCD -- the
	// ReturnBlock poison-fill constant -- and the other is exactly 0x00 -- pool-init zero, in
	// EITHER direction) or "unexplained" (neither side is 0xCD/0x00, or both sides are
	// non-poison values that still disagree) -- the latter would indicate a REAL semantic
	// divergence hiding inside the poison noise, not accounted for by pool-history alone.
	size_t poison_shaped = 0;
	size_t unexplained = 0;
	size_t first_unexplained = (size_t)-1;
	for (size_t i = 0; i < n; ++i) {
		if (a[i] != b[i]) {
			if (first == (size_t)-1) first = i;
			++ndiff;
			const bool poison_pattern =
			    (a[i] == 0xCD && b[i] == 0x00) || (a[i] == 0x00 && b[i] == 0xCD);
			if (poison_pattern) {
				++poison_shaped;
			} else {
				++unexplained;
				if (first_unexplained == (size_t)-1) first_unexplained = i;
			}
		}
	}
	std::printf("blob size=%zu differing_bytes=%zu first_diff_offset=%zd "
	            "poison_shaped=%zu unexplained=%zu first_unexplained_offset=%zd\n",
	            n, ndiff, first == (size_t)-1 ? -1 : (ptrdiff_t)first, poison_shaped, unexplained,
	            first_unexplained == (size_t)-1 ? -1 : (ptrdiff_t)first_unexplained);
	if (first != (size_t)-1) {
		std::printf("  around first diff: a=[");
		for (size_t i = first > 4 ? first - 4 : 0; i < first + 12 && i < n; ++i)
			std::printf("%02x ", a[i]);
		std::printf("] b=[");
		for (size_t i = first > 4 ? first - 4 : 0; i < first + 12 && i < n; ++i)
			std::printf("%02x ", b[i]);
		std::printf("]\n");
	}
	if (first_unexplained != (size_t)-1) {
		std::printf("  around FIRST UNEXPLAINED diff: a=[");
		for (size_t i = first_unexplained > 4 ? first_unexplained - 4 : 0;
		     i < first_unexplained + 12 && i < n; ++i)
			std::printf("%02x ", a[i]);
		std::printf("] b=[");
		for (size_t i = first_unexplained > 4 ? first_unexplained - 4 : 0;
		     i < first_unexplained + 12 && i < n; ++i)
			std::printf("%02x ", b[i]);
		std::printf("]\n");
	}
}

}  // namespace

int main(int argc, char** argv) {
	if (argc < 2) {
		std::fprintf(stderr, "usage: %s <path-to-g5-fixture.sslm>\n", argv[0]);
		return 1;
	}
	std::vector<uint8_t> bytes;
	if (!ReadFile(argv[1], &bytes)) {
		std::fprintf(stderr, "could not read %s\n", argv[1]);
		return 1;
	}
	sslm_model model = nullptr;
	sslm_status st = sslm_model_map(bytes.data(), bytes.size(), &model);
	if (st != SSLM_OK || !model) Fail("sslm_model_map", (int)st);
	std::printf("[1] model mapped OK\n");

	Pool pool;
	if (!pool.Create(model, 4)) Fail("pool.Create", -1);
	std::printf("[2] pool created OK\n");

	const size_t blob_sz = sslm_seq_state_size(model);
	std::printf("[3] blob size = %zu\n", blob_sz);

	// ---- Case A: ONE token via prefix-adopt path ----
	{
		sslm_prefix prefix = nullptr;
		if (sslm_prefix_begin(model, &pool.pool, &prefix) != SSLM_OK) Fail("prefix_begin", -1);
		int32_t tok[1] = {0};
		int32_t consumed = 0;
		st = sslm_prefix_prefill(model, prefix, tok, 1, 8, SSLM_SPAN_PROMPT, nullptr, &consumed);
		if (st != SSLM_OK) Fail("prefix_prefill", (int)st);
		if (sslm_prefix_freeze(prefix) != SSLM_OK) Fail("prefix_freeze", -1);

		sslm_seq seq_a = nullptr;
		if (sslm_seq_create(model, &pool.pool, &seq_a) != SSLM_OK) Fail("seq_create A", -1);
		if (sslm_seq_adopt_prefix(seq_a, prefix) != SSLM_OK) Fail("adopt_prefix", -1);

		std::vector<uint8_t> blob_a(blob_sz);
		size_t sz_a = blob_sz;
		if (sslm_seq_save(seq_a, blob_a.data(), &sz_a) != SSLM_OK) Fail("seq_save A", -1);
		blob_a.resize(sz_a);

		// ---- Case B: the SAME one token, direct prefill, no prefix at all ----
		sslm_seq seq_b = nullptr;
		if (sslm_seq_create(model, &pool.pool, &seq_b) != SSLM_OK) Fail("seq_create B", -1);
		int32_t consumed_b = 0;
		st = sslm_prefill(model, seq_b, tok, 1, 8, SSLM_SPAN_PROMPT, nullptr, &consumed_b);
		if (st != SSLM_OK) Fail("prefill B", (int)st);

		std::vector<uint8_t> blob_b(blob_sz);
		size_t sz_b = blob_sz;
		if (sslm_seq_save(seq_b, blob_b.data(), &sz_b) != SSLM_OK) Fail("seq_save B", -1);
		blob_b.resize(sz_b);

		std::printf("\n=== CASE: 1 token, prefix-adopt (A) vs direct-prefill (B), no schema ===\n");
		if (sz_a != sz_b) {
			std::printf("SIZE MISMATCH a=%zu b=%zu\n", sz_a, sz_b);
		} else {
			DumpFirstDiff(blob_a, blob_b);
		}

		sslm_seq_release(seq_a);
		sslm_seq_release(seq_b);
		sslm_prefix_release(prefix);
	}

	// ---- Case C: SIX tokens (Curie's own system_prompt_tokens), prefix-adopt vs direct,
	// still no schema at all -- isolates whether it's token-count-dependent. ----
	{
		int32_t toks[6] = {0, 1, 2, 3, 4, 5};

		sslm_prefix prefix = nullptr;
		if (sslm_prefix_begin(model, &pool.pool, &prefix) != SSLM_OK) Fail("prefix_begin C", -1);
		int32_t consumed = 0;
		st = sslm_prefix_prefill(model, prefix, toks, 6, 8, SSLM_SPAN_PROMPT, nullptr, &consumed);
		if (st != SSLM_OK) Fail("prefix_prefill C", (int)st);
		if (sslm_prefix_freeze(prefix) != SSLM_OK) Fail("prefix_freeze C", -1);

		sslm_seq seq_a = nullptr;
		if (sslm_seq_create(model, &pool.pool, &seq_a) != SSLM_OK) Fail("seq_create C-A", -1);
		if (sslm_seq_adopt_prefix(seq_a, prefix) != SSLM_OK) Fail("adopt_prefix C", -1);

		std::vector<uint8_t> blob_a(blob_sz);
		size_t sz_a = blob_sz;
		if (sslm_seq_save(seq_a, blob_a.data(), &sz_a) != SSLM_OK) Fail("seq_save C-A", -1);
		blob_a.resize(sz_a);

		sslm_seq seq_b = nullptr;
		if (sslm_seq_create(model, &pool.pool, &seq_b) != SSLM_OK) Fail("seq_create C-B", -1);
		int32_t consumed_b = 0;
		st = sslm_prefill(model, seq_b, toks, 6, 8, SSLM_SPAN_PROMPT, nullptr, &consumed_b);
		if (st != SSLM_OK) Fail("prefill C-B", (int)st);

		std::vector<uint8_t> blob_b(blob_sz);
		size_t sz_b = blob_sz;
		if (sslm_seq_save(seq_b, blob_b.data(), &sz_b) != SSLM_OK) Fail("seq_save C-B", -1);
		blob_b.resize(sz_b);

		std::printf("\n=== CASE: 6 tokens, prefix-adopt (A) vs direct-prefill (B), no schema ===\n");
		if (sz_a != sz_b) {
			std::printf("SIZE MISMATCH a=%zu b=%zu\n", sz_a, sz_b);
		} else {
			DumpFirstDiff(blob_a, blob_b);
		}

		sslm_seq_release(seq_a);
		sslm_seq_release(seq_b);
		sslm_prefix_release(prefix);
	}

	// ---- Case D: Curie's own mechanism cell 2, reproduced exactly -- 6 PROMPT tokens then 3
	// SSLM_SPAN_SCHEMA_CONTENT tokens, WITH a schema bound on the sequence, prefix-adopt (A) vs
	// direct-prefill (B). This is the actual failing cell; Case A/B/C above proved the
	// PROMPT-only path (no schema at all, either token count) is byte-identical, so if D fails,
	// the divergence is specifically in the SCHEMA_CONTENT follow-up call, not in adoption of
	// PROMPT content alone.
	//
	// Uses a FRESH pool, isolated from Cases A/B/C's own draw/release history -- the real test
	// binary (dim8_composition_red.exe) runs ONLY this shape against a pool that has never been
	// touched before it, so Cases A/B/C's prior poison-fill/redraw cycles (ReturnBlock's own
	// 0xCD memset) could leave this pool's blocks in a different history than the real binary's
	// first-ever draws -- ruled out here by giving Case D its own fresh pool. ----
	{
		Pool pool_d;
		if (!pool_d.Create(model, 4)) Fail("pool_d.Create", -1);
		sslm_kv_pool* pool_d_ptr = &pool_d.pool;
		sslm_schema schema = nullptr;
		if (sslm_schema_lookup(model, "shopkeeper_intent_extraction", &schema) != SSLM_OK ||
		    !schema) {
			std::printf("\n=== CASE D skipped: schema lookup failed ===\n");
		} else {
			// Derive a real, schema-legal 3-token forced span the same way fixture_common.h's
			// DeriveRealSchemaContentSpan does: a scratch sequence, seeded, decoded 3 steps.
			sslm_seq scratch = nullptr;
			if (sslm_seq_create(model, pool_d_ptr, &scratch) != SSLM_OK) Fail("scratch create", -1);
			if (sslm_seq_set_schema(scratch, schema) != SSLM_OK) Fail("scratch set_schema", -1);
			int32_t seed_tok[1] = {0};
			int32_t seed_consumed = 0;
			if (sslm_prefill(model, scratch, seed_tok, 1, 8, SSLM_SPAN_PROMPT, nullptr,
			                  &seed_consumed) != SSLM_OK) {
				Fail("scratch seed prefill", -1);
			}
			sslm_decode_params params{};
			params.layer_budget = 28;
			int32_t forced[3];
			for (int i = 0; i < 3; ++i) {
				if (sslm_decode_step(model, &scratch, 1, &params, nullptr, &forced[i]) != SSLM_OK) {
					Fail("scratch decode_step", -1);
				}
			}
			sslm_seq_release(scratch);
			std::printf("\n[derived forced span] %d %d %d\n", forced[0], forced[1], forced[2]);

			int32_t prompt_toks[6] = {0, 1, 2, 3, 4, 5};

			// Sequence A: adopt the shared prefix (6 PROMPT tokens, no schema on the prefix),
			// bind schema, jump-forward the forced span -- exactly Curie's seq_with_prefix.
			sslm_prefix prefix = nullptr;
			if (sslm_prefix_begin(model, pool_d_ptr, &prefix) != SSLM_OK) Fail("D prefix_begin", -1);
			int32_t pc = 0;
			if (sslm_prefix_prefill(model, prefix, prompt_toks, 6, 8, SSLM_SPAN_PROMPT, nullptr,
			                         &pc) != SSLM_OK) {
				Fail("D prefix_prefill", -1);
			}
			if (sslm_prefix_freeze(prefix) != SSLM_OK) Fail("D prefix_freeze", -1);

			sslm_seq seq_a = nullptr;
			if (sslm_seq_create(model, pool_d_ptr, &seq_a) != SSLM_OK) Fail("D seq_create A", -1);
			if (sslm_seq_set_schema(seq_a, schema) != SSLM_OK) Fail("D set_schema A", -1);
			if (sslm_seq_adopt_prefix(seq_a, prefix) != SSLM_OK) Fail("D adopt_prefix", -1);

			// Save state BEFORE the schema-content call, to see whether prompt-span K/V is
			// still intact at this point (isolates: did adoption already corrupt it, or does
			// the schema-content call corrupt it afterward?).
			std::vector<uint8_t> blob_a_pre(blob_sz);
			size_t sz_a_pre = blob_sz;
			if (sslm_seq_save(seq_a, blob_a_pre.data(), &sz_a_pre) != SSLM_OK) Fail("save A pre", -1);
			blob_a_pre.resize(sz_a_pre);

			int32_t ca = 0;
			if (sslm_prefill(model, seq_a, forced, 3, 8, SSLM_SPAN_SCHEMA_CONTENT, nullptr, &ca) !=
			    SSLM_OK) {
				Fail("D prefill A schema", -1);
			}

			std::vector<uint8_t> blob_a(blob_sz);
			size_t sz_a = blob_sz;
			if (sslm_seq_save(seq_a, blob_a.data(), &sz_a) != SSLM_OK) Fail("D save A", -1);
			blob_a.resize(sz_a);

			// Sequence B: direct, no prefix -- exactly Curie's seq_no_prefix.
			sslm_seq seq_b = nullptr;
			if (sslm_seq_create(model, pool_d_ptr, &seq_b) != SSLM_OK) Fail("D seq_create B", -1);
			if (sslm_seq_set_schema(seq_b, schema) != SSLM_OK) Fail("D set_schema B", -1);
			int32_t cb0 = 0;
			if (sslm_prefill(model, seq_b, prompt_toks, 6, 8, SSLM_SPAN_PROMPT, nullptr, &cb0) !=
			    SSLM_OK) {
				Fail("D prefill B prompt", -1);
			}

			std::vector<uint8_t> blob_b_pre(blob_sz);
			size_t sz_b_pre = blob_sz;
			if (sslm_seq_save(seq_b, blob_b_pre.data(), &sz_b_pre) != SSLM_OK) Fail("save B pre", -1);
			blob_b_pre.resize(sz_b_pre);

			int32_t cb = 0;
			if (sslm_prefill(model, seq_b, forced, 3, 8, SSLM_SPAN_SCHEMA_CONTENT, nullptr, &cb) !=
			    SSLM_OK) {
				Fail("D prefill B schema", -1);
			}

			std::vector<uint8_t> blob_b(blob_sz);
			size_t sz_b = blob_sz;
			if (sslm_seq_save(seq_b, blob_b.data(), &sz_b) != SSLM_OK) Fail("D save B", -1);
			blob_b.resize(sz_b);

			std::printf("\n=== CASE D-pre: after 6 PROMPT + schema bound, BEFORE schema-content "
			            "call ===\n");
			if (sz_a_pre != sz_b_pre) {
				std::printf("SIZE MISMATCH a=%zu b=%zu\n", sz_a_pre, sz_b_pre);
			} else {
				DumpFirstDiff(blob_a_pre, blob_b_pre);
			}

			std::printf("\n=== CASE D: full cell 2 repro, AFTER schema-content call ===\n");
			if (sz_a != sz_b) {
				std::printf("SIZE MISMATCH a=%zu b=%zu\n", sz_a, sz_b);
			} else {
				DumpFirstDiff(blob_a, blob_b);
			}

			sslm_seq_release(seq_a);
			sslm_seq_release(seq_b);
			sslm_prefix_release(prefix);
		}
	}

	std::printf("\n[DONE]\n");
	return 0;
}

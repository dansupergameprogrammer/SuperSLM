// T-2130 (Curie) -- shared CHECK harness and real-artifact loading helper for the G5
// constrained-decoding red suite. Reuses this repo's own established conventions rather than
// inventing new ones: the CHECK/CHECK_MSG/SKIP_MSG counter triple and the argv fixture
// convention are tests/t2112-gpu-1p0-red-suite/fixture_common.h's own shapes (itself
// tests/t2018-slora-serial/t2018_offline_red.cpp's convention, itself tests/test_main.cpp's);
// the real-artifact load path (SslmModel::Load over a file read into memory) is
// tools/t2100_gpu_throughput.cpp's own convention, reused verbatim.
//
// Model/schema/adapter paths are NEVER hardcoded (StandardsDocument.md Sec5.2, cold-read
// self-contained): every product cell that needs a real artifact takes it from
// g_model_*_path / g_adapter_path, populated from argv by each suite binary's own main(). A
// cell whose product half needs an artifact argv did not supply reports SKIP (counted
// separately from PASS/FAIL/RED so a suite run without artifacts on the invoking machine is
// still legible), never a silent pass.
#ifndef SSLM_T2130_FIXTURE_COMMON_H
#define SSLM_T2130_FIXTURE_COMMON_H

#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "superslm/artifact.h"
#include "superslm/model.h"

// The declared G5 ABI surface this whole suite is written against (Claude/Vitruvius/
// t2119-g5-constrained-decoding-design-2026-08-16.md Sec5, post rung-6 confirmation fold).
// Promoted copy, this directory -- see sslm_g5.h's own header comment.
#include "sslm_g5.h"

static int GChecks = 0;
static int GFailures = 0;
static int GSkips = 0;

#define CHECK(cond) \
	do { \
		++GChecks; \
		if (!(cond)) { \
			++GFailures; \
			std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
		} \
	} while (0)

#define CHECK_MSG(cond, ...) \
	do { \
		++GChecks; \
		if (!(cond)) { \
			++GFailures; \
			std::printf("FAIL %s:%d: %s -- ", __FILE__, __LINE__, #cond); \
			std::printf(__VA_ARGS__); \
			std::printf("\n"); \
		} \
	} while (0)

#define SKIP_MSG(...) \
	do { \
		++GSkips; \
		std::printf("SKIP %s:%d -- ", __FILE__, __LINE__); \
		std::printf(__VA_ARGS__); \
		std::printf("\n"); \
	} while (0)

// argv-populated real-artifact paths (StandardsDocument.md Sec5.4 real-workload rule).
// Empty means "not supplied on this invocation" -- the cell SKIPs its product half rather
// than silently passing or fabricating a result.
static std::string g_model_1p5b_path;    // real 1.5B artifact, WITH a compiled schema set
static std::string g_adversarial_schema_model_path;  // artifact compiled with the adversarial
                                                       // deep-nesting schema corpus (G5-1 gate)
static std::string g_adapter_path;       // real adapter artifact (shopkeeper-v2), for dim8/dim10
static std::string g_reference_schema_name = "shopkeeper_intent_extraction";  // design Sec3's
                                                       // reference task, the plan's own D-SLM32
                                                       // reference schema name -- overridable.
static std::string g_model_1p5b_variant_path;  // a SECOND artifact, content-hash-matched to
                                                // --model1p5b but compiled with a DIFFERENT (or
                                                // absent) schema set -- dim2 mechanism cell 4's
                                                // own "model_b" fixture (T-2132 harness fix).
static std::string g_secondary_schema_name = "g5_minimal_one_field";  // a schema distinct from
                                                // g_reference_schema_name, for cells needing two
                                                // non-identical schema handles (T-2132).

// Reuses tools/t2100_gpu_throughput.cpp's own load path verbatim (T-2100, this repo).
inline bool LoadRealModel(const std::string& path, superslm::SslmModelView* out_view,
                           std::vector<uint8_t>* out_bytes, std::string* out_err) {
	if (path.empty()) return false;
	std::FILE* f = std::fopen(path.c_str(), "rb");
	if (!f) {
		if (out_err) *out_err = "could not open " + path;
		return false;
	}
	std::fseek(f, 0, SEEK_END);
	const long sz = std::ftell(f);
	std::fseek(f, 0, SEEK_SET);
	out_bytes->resize(sz > 0 ? (size_t)sz : 0);
	if (sz > 0) {
		const size_t n = std::fread(out_bytes->data(), 1, (size_t)sz, f);
		std::fclose(f);
		if (n != (size_t)sz) {
			if (out_err) *out_err = "short read on " + path;
			return false;
		}
	} else {
		std::fclose(f);
	}
	const superslm::SslmModelStatus st =
	    superslm::SslmModel::Load(out_bytes->data(), out_bytes->size(), *out_view, out_err);
	return st == superslm::SslmModelStatus::Ok;
}

// Parses this suite's own argv convention: <binary> [--model1p5b=PATH] [--adversarial=PATH]
// [--adapter=PATH] [--schema=NAME] [--model1p5b_variant=PATH] [--schema2=NAME]. Every flag
// optional; a cell whose fixture is missing SKIPs rather than asserting against nothing.
inline void ParseFixtureArgs(int argc, char** argv) {
	for (int i = 1; i < argc; ++i) {
		const std::string a = argv[i];
		auto take = [&](const char* flag) -> const char* {
			const size_t n = std::strlen(flag);
			return a.compare(0, n, flag) == 0 ? a.c_str() + n : nullptr;
		};
		if (const char* v = take("--model1p5b_variant=")) g_model_1p5b_variant_path = v;
		else if (const char* v = take("--model1p5b=")) g_model_1p5b_path = v;
		else if (const char* v = take("--adversarial=")) g_adversarial_schema_model_path = v;
		else if (const char* v = take("--adapter=")) g_adapter_path = v;
		else if (const char* v = take("--schema2=")) g_secondary_schema_name = v;
		else if (const char* v = take("--schema=")) g_reference_schema_name = v;
	}
}

// --- Real-artifact fixture helpers (T-2132/Curie build-harness fix): each suite main() below
// uses these to actually CALL its cells with real handles when a real artifact is supplied on
// argv, and to SKIP -- never silently pass, never assert against a null handle -- the cells it
// cannot construct when one isn't. This replaces the prior address-take-only mains, whose
// clean-link runtime output was unconditionally checks=0 regardless of what the ABI does (see
// Claude/Brunel/t2132-g5-build-2026-08-16.md, "the harness's own construction" finding). Calling
// sslm_model_map here is safe today (it already ships, T-2139); sslm_schema_lookup and every
// other genuinely G5-only verb remain unresolved externals until G5-2 lands, so these helpers
// change nothing about this suite's RED BY LINK state pre-G5 -- a function that only takes the
// address of a symbol and a function that calls it both force the identical linker resolution.

// Loads a file fully into `keepalive` and maps it via sslm_model_map. `keepalive` must outlive
// every call made against the returned handle (sslm_model_map does not copy the bytes). Returns
// false -- never asserts -- when the path is empty, the file cannot be read, or the artifact is
// rejected; callers SKIP rather than call a cell with a null model handle.
inline bool TryMapRealModel(const std::string& path, std::vector<uint8_t>* keepalive,
                             sslm_model* out_model) {
	*out_model = nullptr;
	if (path.empty()) return false;
	std::FILE* f = std::fopen(path.c_str(), "rb");
	if (!f) return false;
	std::fseek(f, 0, SEEK_END);
	const long sz = std::ftell(f);
	std::fseek(f, 0, SEEK_SET);
	keepalive->resize(sz > 0 ? (size_t)sz : 0);
	if (sz > 0) {
		const size_t n = std::fread(keepalive->data(), 1, (size_t)sz, f);
		std::fclose(f);
		if (n != (size_t)sz) return false;
	} else {
		std::fclose(f);
	}
	return sslm_model_map(keepalive->data(), keepalive->size(), out_model) == SSLM_OK;
}

// Looks up a named schema on an already-mapped model. Returns false (leaves *out null) when the
// model is null or the lookup fails -- including "unresolved external", pre-G5-2, which never
// reaches runtime at all (the whole binary fails to link, per this suite's own RED BY LINK
// contract; this function's mere presence changes nothing about that).
inline bool TryLookupSchema(sslm_model model, const char* name, sslm_schema* out) {
	*out = nullptr;
	if (!model) return false;
	return sslm_schema_lookup(model, name, out) == SSLM_OK;
}

// sslm_decode_step's own precondition (src/sslm_abi.cpp, sslm_decode_stepImpl): `params->
// layer_budget` must be in [1, model's own num_hidden_layers] or the call rejects
// SSLM_INVALID_ARGUMENT outright -- every `sslm_decode_params params{};` in this suite
// originally left layer_budget value-initialized to 0, which is < 1 and therefore always
// invalid, a THIRD suite-fixture gap this ticket's own build log did not evidence (the suite
// was never previously run against a real fixture far enough to reach a decode_step call).
// Reproduced directly: every cell's own decode_step FAILs SSLM_INVALID_ARGUMENT at the real
// fixture regardless of the pool/token-id fixes above, until layer_budget is set. A
// layer_budget below the model's own layer count only partially advances one decode_step call
// (the runtime's own progressive per-layer throttle, S-HARDEN's speed-headroom design) and
// would silently under-produce a token were this suite to try to complete one in fewer calls
// than its own layer depth -- so every call here uses the FULL depth, guaranteeing one
// decode_step call completes exactly one token, matching every cell's own "one call, one
// token" assumption. kNumHiddenLayers reuses tools/t2132_g5_smoke.cpp's own already-proven
// constant verbatim (Qwen2.5-1.5B-Instruct's own published depth) -- the real 1.5B fixture is
// the only model this suite's own --model1p5b/--model1p5b_variant/--adversarial flags name.
constexpr int32_t kNumHiddenLayers = 28;

// Builds an sslm_decode_params with layer_budget set to the model's own full depth (see
// kNumHiddenLayers's own comment) -- every direct sslm_decode_step call in this suite uses
// this instead of a bare `sslm_decode_params params{};`, which value-initializes layer_budget
// to 0 and is unconditionally rejected.
inline sslm_decode_params MakeFullDepthDecodeParams() {
	sslm_decode_params params{};
	params.layer_budget = kNumHiddenLayers;
	return params;
}

// sslm_seq_save's own required size (src/sslm_abi.cpp: `kSeqBlobFixedHeaderBytes +
// residual_len + 4 + seq->block_size`) is dominated by seq->block_size -- one whole
// sequence's real KV footprint (sslm_kv_block_size's own contract), which for a real,
// multi-layer 1.5B model is many times larger than the fixed 65536-byte stack buffer every
// `sslm_seq_save`/`sslm_seq_restore` call in this suite originally declared (`unsigned char
// blob[65536]`) -- SSLM_BUFFER_TOO_SMALL against the real fixture, a FOURTH suite-fixture gap
// this ticket's own build log did not evidence for the identical reason as the layer_budget
// gap above (the suite was never previously run far enough against a real fixture to reach a
// save call). sslm_seq_state_size(model) reports the same worst-case formula's own upper
// bound (src/sslm_abi.cpp: fixed header + worst-case residual + kv_block_size) -- every
// dynamically-sized save-blob buffer below uses it instead of a fixed 64KB guess.
inline std::vector<uint8_t> AllocRealSaveBlobBuffer(sslm_model model) {
	const size_t n = model ? sslm_seq_state_size(model) : 0;
	return std::vector<uint8_t>(n > 0 ? n : 65536);
}

// sslm_prefill consumes AT MOST chunk_budget tokens PER CALL and reports the actual count via
// `consumed` -- it never internally loops across multiple chunks within one call
// (src/sslm_abi.cpp, PrefillWholeTokensImpl: `n = count < chunk_budget ? count : chunk_budget`,
// exactly one bounded chunk). The design states this is the intended contract, not an
// implementation shortcut (t2119-g5-constrained-decoding-design-2026-08-16.md, prefix-
// construction correction: "`consumed`... is what lets a caller drive a multi-call chunked
// ingestion loop at all" -- design Sec8.1's own SARATHI-pattern bounded-ingestion law, "every
// call performs a fixed, known amount of work"). Several cells in this suite originally called
// sslm_prefill ONCE for a token count larger than chunk_budget and expected full consumption
// from that one call -- a FIFTH suite-fixture gap this ticket's own build log did not
// evidence (the suite was never previously run far enough against a real fixture to reach a
// multi-chunk prefill). This helper loops until every token is consumed or a call fails,
// matching the design's own intended usage -- the identical pattern this suite's own
// g5_collision_regression_red.cpp RunScenario already used correctly for its own
// SSLM_SPAN_PROMPT half. Returns the failing status (or SSLM_OK on full success);
// `*out_consumed` is the running total actually consumed before any failure.
inline sslm_status PrefillLooped(sslm_model model, sslm_seq seq, const int32_t* tokens,
                                  int32_t count, int32_t chunk_budget, sslm_span_kind kind,
                                  sslm_workspace ws, int32_t* out_consumed) {
	*out_consumed = 0;
	int32_t offset = 0;
	while (offset < count) {
		int32_t this_consumed = 0;
		const sslm_status st = sslm_prefill(model, seq, tokens + offset, count - offset,
		                                     chunk_budget, kind, ws, &this_consumed);
		if (st != SSLM_OK) return st;
		if (this_consumed <= 0) return SSLM_INVALID_ARGUMENT;  // no progress; avoid an infinite loop
		offset += this_consumed;
		*out_consumed += this_consumed;
	}
	return SSLM_OK;
}

// --- Real KV-pool construction (T-2132/Curie fix, first suite-fixture defect): every
// sslm_seq_create/sslm_prefix_begin/sslm_seq_restore call in this suite originally passed a
// literal nullptr as its `pool` argument, which the shipped ABI's own precondition
// (`if (!model || !pool || !*pool) return SSLM_INVALID_ARGUMENT;`, src/sslm_abi.cpp) rejects
// outright -- reproduced directly, 28/28 FAIL at the first sslm_seq_create call against the
// real fixture (Claude/Brunel/t2132-g5-build-2026-08-16.md). RealKvPool constructs a genuine
// pool per the ABI's own C1 sizing verbs, reusing tools/t2139_c2_smoke.cpp's own construction
// pattern verbatim (over-allocate + std::align, sized from sslm_kv_block_size/
// sslm_kv_pool_overhead_size) so every pool-consuming call below can pass a real `sslm_kv_pool*`
// instead. ---

// The pool sizing every dimN_red.cpp binary in this suite creates one of, per real model
// mapped -- one whole KV block per concurrently-live sequence/prefix (sslm_kv_block_size's own
// contract: no token-count arithmetic). 4 matches tools/t2139_c2_smoke.cpp's own
// already-proven block_count; the suite's own maximum concurrent live handles against any one
// model, in any one cell, is 3 (dim3's three-sequence batch; dim8's shared prefix plus its two
// adopting sequences) -- 4 leaves headroom without inflating the real, multi-megabyte
// per-block allocation further than this suite needs.
constexpr uint32_t kRealPoolBlockCount = 4;

// Owns the storage buffer for a real sslm_kv_pool bound to one model, and destroys the pool on
// going out of scope (or on a subsequent Create() call). Construction failure (any ABI verb
// returning non-OK, or a zero-sized sizing query) leaves ptr() pointing at a null handle;
// callers SKIP rather than pass a null/partially-constructed pool into sslm_seq_create et al.
class RealKvPool {
 public:
	RealKvPool() = default;
	~RealKvPool() { Destroy(); }
	RealKvPool(const RealKvPool&) = delete;
	RealKvPool& operator=(const RealKvPool&) = delete;

	bool Create(sslm_model model, uint32_t block_count) {
		Destroy();
		if (!model) return false;
		const size_t block_bytes = sslm_kv_block_size(model);
		const size_t overhead = sslm_kv_pool_overhead_size(model, block_count);
		if (block_bytes == 0 || overhead == 0) return false;
		const size_t needed = block_bytes * static_cast<size_t>(block_count) + overhead;
		storage_.assign(needed + 63, 0);
		void* aligned = storage_.data();
		size_t space = storage_.size();
		std::align(64, needed, aligned, space);
		if (!aligned) return false;
		sslm_kv_pool pool = nullptr;
		if (sslm_kv_pool_create(model, aligned, needed, block_count, &pool) != SSLM_OK || !pool) {
			return false;
		}
		pool_ = pool;
		return true;
	}

	void Destroy() {
		if (pool_) {
			sslm_kv_pool_destroy(pool_);
			pool_ = nullptr;
		}
		storage_.clear();
	}

	// The `sslm_kv_pool*` shape every pool-consuming ABI verb below takes.
	sslm_kv_pool* ptr() { return &pool_; }

 private:
	std::vector<uint8_t> storage_;
	sslm_kv_pool pool_ = nullptr;
};

// sslm_decode_step's own precondition (src/sslm_abi.cpp, sslm_decode_stepImpl): a sequence
// with NO prior prompt/forced content (current_token < 0, layer_index == 0) and no
// already-computed resting state (!ready_for_logits) has nothing to produce a token from and
// is rejected SSLM_INVALID_ARGUMENT -- a freshly created (or just-reset) sequence's own
// natural state. Several cells in this suite call sslm_decode_step directly against such a
// sequence (to observe its own first masked-argmax choice, or a plain unconstrained decode)
// with no other reason to prefill first; this seeds exactly one arbitrary, in-range
// SSLM_SPAN_PROMPT token so the precondition is met without perturbing anything SSLM_SPAN_PROMPT
// does not touch -- the DFA-walk-state (design Sec5: "walk-state NEVER advances, whatever
// schema is bound"). Returns false -- callers SKIP -- if the seed prefill itself fails.
inline bool SeedPromptForDecodeStep(sslm_model model, sslm_seq seq) {
	int32_t seed_token[1] = {0};
	int32_t consumed = 0;
	return sslm_prefill(model, seq, seed_token, 1, /*chunk_budget=*/8, SSLM_SPAN_PROMPT,
	                     /*ws=*/nullptr, &consumed) == SSLM_OK;
}

// --- Real schema-content span derivation (T-2132/Curie fix, second suite-fixture defect):
// several product/functional cells originally embedded hardcoded literal token ids (e.g.
// {0,1,2}, i % 20) as SSLM_SPAN_SCHEMA_CONTENT spans, assuming -- with no document specifying
// the relationship -- that they were legal continuations of the reference schema's own
// compiled DFA from its start state; a real tokenizer's own BPE-assigned ids cannot be
// expected to satisfy that by construction (Claude/Brunel/t2132-g5-build-2026-08-16.md). ---

// Derives a real, schema-legal sequence of `count` token ids for use as an
// SSLM_SPAN_SCHEMA_CONTENT forced/fixed span, by running `count` ordinary masked
// sslm_decode_step calls against a scratch sequence freshly bound to `schema` on `pool`. The
// runtime's own masked argmax, by construction (G5-2's own mask-application step), never
// emits a token the schema's compiled DFA rejects at its current walk position -- so the ids
// collected here are honestly legal continuations from the schema's start state, derived from
// the real tokenizer/schema relationship at runtime, never literal ids assumed legal by
// construction. Returns false -- callers SKIP -- if the scratch sequence cannot be constructed
// or a decode step fails; `out_tokens` is left empty on failure.
inline bool DeriveRealSchemaContentSpan(sslm_model model, sslm_kv_pool* pool, sslm_schema schema,
                                         int32_t count, std::vector<int32_t>* out_tokens) {
	out_tokens->clear();
	if (!model || !pool || !schema || count <= 0) return false;
	sslm_seq scratch = nullptr;
	if (sslm_seq_create(model, pool, &scratch) != SSLM_OK || !scratch) return false;
	if (sslm_seq_set_schema(scratch, schema) != SSLM_OK) {
		sslm_seq_release(scratch);
		return false;
	}
	// Satisfy sslm_decode_step's own "has prior content" precondition -- see
	// SeedPromptForDecodeStep's own comment. SSLM_SPAN_PROMPT never advances the DFA walk, so
	// this does not change which tokens the schema's own start state admits below.
	if (!SeedPromptForDecodeStep(model, scratch)) {
		sslm_seq_release(scratch);
		return false;
	}
	sslm_decode_params params = MakeFullDepthDecodeParams();
	std::vector<int32_t> tokens(static_cast<size_t>(count));
	for (int32_t i = 0; i < count; ++i) {
		int32_t tok = 0;
		if (sslm_decode_step(model, &scratch, 1, &params, /*ws=*/nullptr, &tok) != SSLM_OK) {
			sslm_seq_release(scratch);
			return false;
		}
		tokens[static_cast<size_t>(i)] = tok;
	}
	if (sslm_seq_release(scratch) != SSLM_OK) return false;
	*out_tokens = std::move(tokens);
	return true;
}

#endif  // SSLM_T2130_FIXTURE_COMMON_H

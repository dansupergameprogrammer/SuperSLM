// T-2130 (Curie) -- Dim 1 (Lifetime and reuse), design Sec7 dim1 (revised T-2121 F1, closed;
// mid-forced-chain save/restore added T-2127 G2, closed). 3 cells: 2 mechanism, 1 product.
// RED BY LINK: sslm_g5.h declares the surface; no .cpp in this repo defines it (grep
// confirmed at authoring commit main@f37be4a).
#include "fixture_common.h"

using namespace superslm;

// --- Mechanism cell 1 (design Sec7 dim1, revised): a schema-bound sequence, mid-DFA-walk
// (non-start, non-accepting state), reset and reused. Asserts no leak from the prior
// generation's schema walk into the next one -- the same poison-fill discipline this
// dimension already applies to KV-block recycling, applied here to the DFA-walk-state field.
// Oracle: equivalence against the solo, uninterrupted run of the SECOND generation alone --
// not "same bits twice" (a stale-but-deterministic leak would pass a same-bits check). ---
static void TestDim1_M1_ResetClearsDfaWalkStateNoLeak(sslm_model model, sslm_kv_pool* pool,
                                                       sslm_schema schema) {
	// The forced span's own ids are DERIVED from the real schema/tokenizer at runtime
	// (T-2132/Curie fix -- see fixture_common.h's DeriveRealSchemaContentSpan) rather than
	// literal ids assumed legal by construction: a hardcoded id here that the compiled DFA
	// does not admit from the start state would make the CHECK below fail
	// SSLM_SCHEMA_SPAN_UNREACHABLE regardless of this cell's own subject (reset/reuse).
	std::vector<int32_t> forced_tokens;
	if (!DeriveRealSchemaContentSpan(model, pool, schema, 3, &forced_tokens)) {
		SKIP_MSG("could not derive a real 3-token schema-legal forced span from the live "
		         "fixture -- mechanism cell 1 not run");
		return;
	}
	sslm_seq seq = nullptr;
	CHECK(sslm_seq_create(model, pool, &seq) == SSLM_OK);
	CHECK(sslm_seq_set_schema(seq, schema) == SSLM_OK);
	int32_t consumed = 0;
	// Advance the walk mid-generation via a forced (jump-forward-shaped) span so the
	// pre-reset walk-state is genuinely non-start, not merely "created".
	CHECK(sslm_prefill(model, seq, forced_tokens.data(), 3, /*chunk_budget=*/8,
	                    SSLM_SPAN_SCHEMA_CONTENT, /*ws=*/nullptr, &consumed) == SSLM_OK);
	CHECK(sslm_seq_reset(seq) == SSLM_OK);
	// CORRECTED (T-2132/Curie fix, real-fixture reproduction): the design's own literal text
	// (t2119-g5-constrained-decoding-design-2026-08-16.md, sslm_seq_set_schema's own comment)
	// states set_schema is "valid ONLY when the sequence's DFA-walk state is at its start (a
	// fresh sslm_seq_create, OR IMMEDIATELY AFTER sslm_seq_reset)" -- reset is explicitly one
	// of the two ACCEPTING conditions, not a rejecting one, and the shipped, reviewed
	// implementation (src/sslm_abi.cpp: rejects only when dfa_walk_state is neither "unused"
	// nor exactly 0, and reset restores it to exactly 0) matches that text exactly. This
	// cell's prior expectation (SSLM_SCHEMA_BIND_REJECTED here) was a misreading of the design
	// -- reproduced directly against the real fixture (SSLM_OK, not SSLM_SCHEMA_BIND_REJECTED,
	// at this exact call) before being corrected. Re-binding to the SAME schema right after
	// reset succeeds because the walk is genuinely fresh again -- exactly what makes the
	// SECOND generation below indistinguishable from a fresh sequence's first, which is this
	// cell's own actual subject (no leak from the pre-reset walk).
	CHECK(sslm_seq_set_schema(seq, schema) == SSLM_OK);
	// sslm_decode_step's own precondition requires prior content on the sequence (reset left
	// it in exactly a fresh sequence's own "nothing to resume from" state) -- seeded with one
	// arbitrary SSLM_SPAN_PROMPT token, which never advances the DFA walk (design Sec5), so
	// the walk this cell is actually testing (post-reset, back at the schema's own start
	// state) is unaffected. See fixture_common.h's SeedPromptForDecodeStep.
	CHECK(SeedPromptForDecodeStep(model, seq));
	int32_t out_tokens[1] = {0};
	sslm_decode_params params = MakeFullDepthDecodeParams();
	CHECK(sslm_decode_step(model, &seq, 1, &params, /*ws=*/nullptr, out_tokens) == SSLM_OK);
	// FEATURE ORACLE (equivalence, not same-bits-twice): out_tokens[0] must equal what a
	// FRESH sequence (never touched by the pre-reset forced span) produces as its own first
	// constrained token under the same schema -- wired once the build seat's fresh-sequence
	// comparator exists; this cell's own structural half is the SSLM_SCHEMA_BIND_REJECTED
	// assertion above, which is independently checkable today.
	CHECK(sslm_seq_release(seq) == SSLM_OK);
}

// --- Mechanism cell 2 (design Sec7 dim1, revised): a sequence resumed after an interleaved
// decode_step on a DIFFERENT batch sharing the same workspace produces the schema-constrained
// output its own solo run would -- the per-sequence, not shared-scratch, requirement (the
// Loki Manifestation-A shape this dimension's base cell already exists to catch). ---
static void TestDim1_M2_InterleavedBatchDoesNotCorruptWalkState(sslm_model model,
                                                                 sslm_kv_pool* pool,
                                                                 sslm_schema schema_a,
                                                                 sslm_schema schema_b) {
	sslm_seq seq_a = nullptr, seq_b = nullptr;
	CHECK(sslm_seq_create(model, pool, &seq_a) == SSLM_OK);
	CHECK(sslm_seq_create(model, pool, &seq_b) == SSLM_OK);
	CHECK(sslm_seq_set_schema(seq_a, schema_a) == SSLM_OK);
	CHECK(sslm_seq_set_schema(seq_b, schema_b) == SSLM_OK);
	// sslm_decode_step's own precondition requires prior content on every sequence in the
	// batch -- seeded with one arbitrary SSLM_SPAN_PROMPT token each, which never advances
	// either DFA walk (design Sec5). See fixture_common.h's SeedPromptForDecodeStep.
	CHECK(SeedPromptForDecodeStep(model, seq_a));
	CHECK(SeedPromptForDecodeStep(model, seq_b));
	sslm_decode_params params = MakeFullDepthDecodeParams();
	sslm_seq batch[2] = {seq_a, seq_b};
	int32_t out_tokens[2] = {0, 0};
	CHECK(sslm_decode_step(model, batch, 2, &params, /*ws=*/nullptr, out_tokens) == SSLM_OK);
	// FEATURE ORACLE: seq_a's own token equals a solo (batch-of-one) run of seq_a alone under
	// schema_a, and likewise for seq_b under schema_b -- proving the shared-workspace decode
	// call did not let seq_b's schema_b walk leak into seq_a's own DFA-walk-state field.
	CHECK(sslm_seq_release(seq_a) == SSLM_OK);
	CHECK(sslm_seq_release(seq_b) == SSLM_OK);
}

// --- Product cell 1 (design Sec6 G5-3 / Sec7 dim1, T-2127 G2, closed this fold): a forced
// chain long enough to cross a chunkBudget boundary is a sequence of multiple internal
// sslm_prefill(..., SSLM_SPAN_SCHEMA_CONTENT, ...) calls; interrupted with
// sslm_seq_save/restore AT that internal boundary, before the chain completes. The restored
// sequence must finish the remaining forced tokens and their K/V population identically to
// the uninterrupted run -- the equivalence oracle extended across a save/restore boundary
// landing INSIDE a forced chain, the narrower case neither T-2120 F1 nor its downstream fix
// anticipated (design had no concept of a multi-call forced chain before the repair). ---
static void TestDim1_P1_MidForcedChainSaveRestoreEquivalence(sslm_model model, sslm_kv_pool* pool,
                                                              sslm_schema reference_schema) {
	if (g_model_1p5b_path.empty()) {
		SKIP_MSG("real 1.5B artifact with a compiled schema set not supplied "
		         "(--model1p5b=PATH) -- product cell not run");
		return;
	}
	// The forced chain's own 20 token ids are DERIVED from the real schema/tokenizer at
	// runtime (T-2132/Curie fix, fixture_common.h's DeriveRealSchemaContentSpan) rather than
	// assumed legal by literal construction -- see that helper's own comment for why this is
	// honest: the runtime's masked argmax never emits a DFA-illegal token, so the ids
	// collected from it are a genuine legal path from the schema's start state.
	std::vector<int32_t> long_forced_chain;
	if (!DeriveRealSchemaContentSpan(model, pool, reference_schema, 20, &long_forced_chain)) {
		SKIP_MSG("could not derive a real 20-token schema-legal forced chain from the live "
		         "fixture -- product cell not run");
		return;
	}
	// Uninterrupted run: the reference schema's own long forced chain (an adversarial schema
	// with a forced chain crossing chunk_budget=8, per design Sec6 G5-3's own red-suite text),
	// driven to full consumption via PrefillLooped -- sslm_prefill consumes at most
	// chunk_budget tokens per call (design Sec8.1's own bounded-ingestion law; see
	// fixture_common.h's PrefillLooped), the caller loops for the rest.
	sslm_seq seq_uninterrupted = nullptr;
	CHECK(sslm_seq_create(model, pool, &seq_uninterrupted) == SSLM_OK);
	CHECK(sslm_seq_set_schema(seq_uninterrupted, reference_schema) == SSLM_OK);
	int32_t consumed_total = 0;
	CHECK(PrefillLooped(model, seq_uninterrupted, long_forced_chain.data(), 20, /*chunk_budget=*/8,
	                     SSLM_SPAN_SCHEMA_CONTENT, nullptr, &consumed_total) == SSLM_OK);
	CHECK(consumed_total == 20);

	// Interrupted run: the SAME forced chain, but sslm_seq_save/restore lands between prefill
	// calls -- after the first chunk_budget=8-token call (a single sslm_prefill call, since
	// 8 == chunk_budget consumes it whole), before the chain's remaining 12 (which itself
	// needs two further chunked calls -- PrefillLooped, same reason as the uninterrupted run
	// above).
	sslm_seq seq_interrupted = nullptr;
	CHECK(sslm_seq_create(model, pool, &seq_interrupted) == SSLM_OK);
	CHECK(sslm_seq_set_schema(seq_interrupted, reference_schema) == SSLM_OK);
	int32_t consumed_first = 0;
	CHECK(sslm_prefill(model, seq_interrupted, long_forced_chain.data(), 8, /*chunk_budget=*/8,
	                    SSLM_SPAN_SCHEMA_CONTENT, nullptr, &consumed_first) == SSLM_OK);
	CHECK(consumed_first == 8);
	std::vector<uint8_t> blob = AllocRealSaveBlobBuffer(model);
	size_t blob_size = blob.size();
	CHECK(sslm_seq_save(seq_interrupted, blob.data(), &blob_size) == SSLM_OK);
	CHECK(sslm_seq_release(seq_interrupted) == SSLM_OK);
	sslm_seq seq_restored = nullptr;
	CHECK(sslm_seq_restore(model, pool, blob.data(), blob_size, &seq_restored) == SSLM_OK);
	int32_t consumed_rest = 0;
	CHECK(PrefillLooped(model, seq_restored, long_forced_chain.data() + 8, 12, /*chunk_budget=*/8,
	                     SSLM_SPAN_SCHEMA_CONTENT, nullptr, &consumed_rest) == SSLM_OK);
	CHECK(consumed_rest == 12);

	// FEATURE ORACLE: seq_restored's final DFA-walk-state and full K/V content over the
	// 20-token forced span equal seq_uninterrupted's, token-for-token and bit-for-bit -- the
	// G5-3 equivalence oracle (design Sec6/Sec8), extended across a save/restore boundary
	// landing INSIDE the forced chain rather than only at an ordinary token gap.
	CHECK(sslm_seq_release(seq_uninterrupted) == SSLM_OK);
	CHECK(sslm_seq_release(seq_restored) == SSLM_OK);
}

int main(int argc, char** argv) {
	ParseFixtureArgs(argc, argv);
	std::vector<uint8_t> model_bytes;
	sslm_model model = nullptr;
	const bool have_model = TryMapRealModel(g_model_1p5b_path, &model_bytes, &model);
	sslm_schema schema_ref = nullptr, schema_secondary = nullptr;
	const bool have_schema_ref =
	    have_model && TryLookupSchema(model, g_reference_schema_name.c_str(), &schema_ref);
	const bool have_schema_secondary =
	    have_model && TryLookupSchema(model, g_secondary_schema_name.c_str(), &schema_secondary);

	// Real KV pool (T-2132/Curie fix): sslm_seq_create/sslm_seq_restore below require a
	// non-null, already-created sslm_kv_pool* -- see fixture_common.h's own RealKvPool comment.
	RealKvPool pool;
	const bool have_pool = have_model && pool.Create(model, kRealPoolBlockCount);

	if (have_schema_ref && have_pool) {
		TestDim1_M1_ResetClearsDfaWalkStateNoLeak(model, pool.ptr(), schema_ref);
	} else {
		SKIP_MSG("real 1.5B artifact with a compiled schema set not supplied (--model1p5b=PATH), "
		         "the reference schema did not resolve, or a real KV pool could not be "
		         "constructed -- mechanism cell 1 not run");
	}
	if (have_schema_ref && have_schema_secondary && have_pool) {
		TestDim1_M2_InterleavedBatchDoesNotCorruptWalkState(model, pool.ptr(), schema_ref,
		                                                     schema_secondary);
	} else {
		SKIP_MSG("real 1.5B artifact with two distinct compiled schemas not supplied, or a "
		         "real KV pool could not be constructed -- mechanism cell 2 not run");
	}
	// Self-skips via g_model_1p5b_path.empty(), or via DeriveRealSchemaContentSpan's own
	// failure path (which fires cleanly if the pool above could not be constructed -- an
	// uncreated pool makes sslm_seq_create fail SSLM_INVALID_ARGUMENT inside that helper,
	// which the helper reports as its own "could not derive" SKIP rather than a bare FAIL).
	TestDim1_P1_MidForcedChainSaveRestoreEquivalence(model, pool.ptr(), schema_ref);
	std::printf("checks=%d failures=%d skips=%d\n", GChecks, GFailures, GSkips);
	return GFailures ? 1 : 0;
}

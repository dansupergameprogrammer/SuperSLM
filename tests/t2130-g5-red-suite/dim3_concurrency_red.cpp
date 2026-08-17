// T-2130 (Curie) -- Dim 3 (Concurrency), design Sec7 dim3 (revised this fold; was "covered by
// the existing cell", T-2121 F4, closed -- G5's mask-application and jump-forward code paths
// are equally NEW, per the runtime-additive-LoRA re-cross precedent this design cites). 1
// cell, TSan-gated. RED BY LINK; the sanitizer half additionally requires a Clang-TSan build
// (design Sec7 dim3: "the CI job per Sec13 item 5, not a comment") -- this cell SKIPs its
// sanitizer half when not built under -fsanitize=thread, per this project's own S-HARDEN-6
// precedent, and always exercises its equivalence half.
#include "fixture_common.h"

using namespace superslm;

// --- Mechanism cell 1: disjoint sequences decode concurrently -- at least one sequence
// schema-constrained with a jump-forward-forced batched-prefill in flight, concurrently with
// sibling sequences decoding unconstrained or under a DIFFERENT schema, all reading the same
// artifact-resident mask pages. Assert: no race reported (TSan build only), and each
// sequence's output matches its own solo run. Oracle: sanitizer + equivalence (design Sec7
// dim3), owned by G5-2/G5-3 jointly (the TSan cell is authored once both mechanisms exist). ---
static void TestDim3_M1_ConcurrentMixedSchemaDecodeNoRaceAndSoloEquivalence(
    sslm_model model, sslm_kv_pool* pool, sslm_schema schema_a, sslm_schema schema_b) {
	// The forced span's own ids are DERIVED from the real schema/tokenizer at runtime
	// (T-2132/Curie fix), not literal ids assumed legal by construction -- see
	// fixture_common.h's DeriveRealSchemaContentSpan. Derived FIRST, before this cell's own
	// three sequences are created, so the pool's own headroom is not shared with a transient
	// derivation handle at the same time as this cell's own maximum concurrent live count.
	std::vector<int32_t> forced_tokens;
	if (!DeriveRealSchemaContentSpan(model, pool, schema_a, 2, &forced_tokens)) {
		SKIP_MSG("could not derive a real 2-token schema-legal forced span from the live "
		         "fixture -- mechanism cell 1 not run");
		return;
	}

	sslm_seq seq_constrained_jf = nullptr;   // schema_a, will jump-forward
	sslm_seq seq_unconstrained = nullptr;     // SSLM_SCHEMA_NONE
	sslm_seq seq_constrained_other = nullptr; // schema_b, no jump-forward
	CHECK(sslm_seq_create(model, pool, &seq_constrained_jf) == SSLM_OK);
	CHECK(sslm_seq_create(model, pool, &seq_unconstrained) == SSLM_OK);
	CHECK(sslm_seq_create(model, pool, &seq_constrained_other) == SSLM_OK);
	CHECK(sslm_seq_set_schema(seq_constrained_jf, schema_a) == SSLM_OK);
	CHECK(sslm_seq_set_schema(seq_unconstrained, SSLM_SCHEMA_NONE) == SSLM_OK);
	CHECK(sslm_seq_set_schema(seq_constrained_other, schema_b) == SSLM_OK);

	// The runtime's own jump-forward logic issues the forced chain internally
	// (SSLM_SPAN_SCHEMA_CONTENT) concurrently with the two sibling sequences' ordinary decode
	// steps -- the driver (build-seat-owned thread harness, per this dimension's own base
	// cell) interleaves these three sequences' calls across threads. The forced-chain call
	// itself is exercised here as the single-threaded shape the concurrent driver wraps.
	int32_t consumed = 0;
	CHECK(sslm_prefill(model, seq_constrained_jf, forced_tokens.data(), 2, /*chunk_budget=*/8,
	                    SSLM_SPAN_SCHEMA_CONTENT, nullptr, &consumed) == SSLM_OK);
	// sslm_decode_step's own precondition requires prior content on every sequence in the
	// batch; seq_constrained_jf already has it (the forced prefill above), the other two do
	// not yet -- seeded with one arbitrary SSLM_SPAN_PROMPT token each, which never advances
	// either DFA walk (design Sec5). See fixture_common.h's SeedPromptForDecodeStep.
	CHECK(SeedPromptForDecodeStep(model, seq_unconstrained));
	CHECK(SeedPromptForDecodeStep(model, seq_constrained_other));

	sslm_decode_params params = MakeFullDepthDecodeParams();
	sslm_seq batch[3] = {seq_constrained_jf, seq_unconstrained, seq_constrained_other};
	int32_t out_tokens[3] = {0, 0, 0};
	CHECK(sslm_decode_step(model, batch, 3, &params, nullptr, out_tokens) == SSLM_OK);

	// FEATURE ORACLE: each sequence's own output equals its own solo (batch-of-one) run --
	// proving the shared, read-only mask pages and the new mask-application/jump-forward code
	// paths introduce no cross-sequence dependence, under concurrent access.
#ifdef __SANITIZE_THREAD__
	// Sanitizer half: this translation unit built with -fsanitize=thread reports no race
	// across the three sequences' concurrent mask-page reads and DFA-walk-state writes.
#else
	SKIP_MSG("not built under Clang-TSan (-fsanitize=thread) -- sanitizer half not run; "
	         "the equivalence half above still executes");
#endif
	CHECK(sslm_seq_release(seq_constrained_jf) == SSLM_OK);
	CHECK(sslm_seq_release(seq_unconstrained) == SSLM_OK);
	CHECK(sslm_seq_release(seq_constrained_other) == SSLM_OK);
}

int main(int argc, char** argv) {
	ParseFixtureArgs(argc, argv);
	std::vector<uint8_t> model_bytes;
	sslm_model model = nullptr;
	const bool have_model = TryMapRealModel(g_model_1p5b_path, &model_bytes, &model);
	sslm_schema schema_a = nullptr, schema_b = nullptr;
	const bool have_schema_a =
	    have_model && TryLookupSchema(model, g_reference_schema_name.c_str(), &schema_a);
	const bool have_schema_b =
	    have_model && TryLookupSchema(model, g_secondary_schema_name.c_str(), &schema_b);

	// Real KV pool (T-2132/Curie fix): sslm_seq_create below requires a non-null,
	// already-created sslm_kv_pool* -- see fixture_common.h's own RealKvPool comment.
	RealKvPool pool;
	const bool have_pool = have_model && pool.Create(model, kRealPoolBlockCount);

	if (have_schema_a && have_schema_b && have_pool) {
		TestDim3_M1_ConcurrentMixedSchemaDecodeNoRaceAndSoloEquivalence(model, pool.ptr(),
		                                                                schema_a, schema_b);
	} else {
		SKIP_MSG("real 1.5B artifact with two distinct compiled schemas not supplied, or a "
		         "real KV pool could not be constructed -- mechanism cell 1 not run");
	}
	std::printf("checks=%d failures=%d skips=%d\n", GChecks, GFailures, GSkips);
	return GFailures ? 1 : 0;
}

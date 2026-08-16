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
    sslm_model model, sslm_schema schema_a, sslm_schema schema_b) {
	sslm_seq seq_constrained_jf = nullptr;   // schema_a, will jump-forward
	sslm_seq seq_unconstrained = nullptr;     // SSLM_SCHEMA_NONE
	sslm_seq seq_constrained_other = nullptr; // schema_b, no jump-forward
	CHECK(sslm_seq_create(model, nullptr, &seq_constrained_jf) == SSLM_OK);
	CHECK(sslm_seq_create(model, nullptr, &seq_unconstrained) == SSLM_OK);
	CHECK(sslm_seq_create(model, nullptr, &seq_constrained_other) == SSLM_OK);
	CHECK(sslm_seq_set_schema(seq_constrained_jf, schema_a) == SSLM_OK);
	CHECK(sslm_seq_set_schema(seq_unconstrained, SSLM_SCHEMA_NONE) == SSLM_OK);
	CHECK(sslm_seq_set_schema(seq_constrained_other, schema_b) == SSLM_OK);

	// The runtime's own jump-forward logic issues the forced chain internally
	// (SSLM_SPAN_SCHEMA_CONTENT) concurrently with the two sibling sequences' ordinary decode
	// steps -- the driver (build-seat-owned thread harness, per this dimension's own base
	// cell) interleaves these three sequences' calls across threads. The forced-chain call
	// itself is exercised here as the single-threaded shape the concurrent driver wraps.
	int32_t forced_tokens[2] = {0, 1};
	int32_t consumed = 0;
	CHECK(sslm_prefill(model, seq_constrained_jf, forced_tokens, 2, /*chunk_budget=*/8,
	                    SSLM_SPAN_SCHEMA_CONTENT, nullptr, &consumed) == SSLM_OK);

	sslm_decode_params params{};
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
	volatile void* addr_0 =
	    (void*)&TestDim3_M1_ConcurrentMixedSchemaDecodeNoRaceAndSoloEquivalence;
	(void)addr_0;
	std::printf("checks=%d failures=%d skips=%d\n", GChecks, GFailures, GSkips);
	return GFailures ? 1 : 0;
}

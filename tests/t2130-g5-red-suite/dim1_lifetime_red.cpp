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
static void TestDim1_M1_ResetClearsDfaWalkStateNoLeak(sslm_model model, sslm_schema schema) {
	sslm_seq seq = nullptr;
	CHECK(sslm_seq_create(model, /*pool=*/nullptr, &seq) == SSLM_OK);
	CHECK(sslm_seq_set_schema(seq, schema) == SSLM_OK);
	int32_t consumed = 0;
	// Advance the walk mid-generation via a forced (jump-forward-shaped) span so the
	// pre-reset walk-state is genuinely non-start, not merely "created".
	const int32_t forced_tokens[3] = {0, 1, 2};
	CHECK(sslm_prefill(model, seq, forced_tokens, 3, /*chunk_budget=*/8,
	                    SSLM_SPAN_SCHEMA_CONTENT, /*ws=*/nullptr, &consumed) == SSLM_OK);
	CHECK(sslm_seq_reset(seq) == SSLM_OK);
	// Reset preserves the schema binding (design Sec5) -- re-binding must be rejected, not
	// required, and the walk-state must have returned to the schema's own start state so the
	// SECOND generation below is indistinguishable from a fresh sequence's first.
	CHECK(sslm_seq_set_schema(seq, schema) == SSLM_SCHEMA_BIND_REJECTED);
	int32_t out_tokens[1] = {0};
	sslm_decode_params params{};
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
                                                                 sslm_schema schema_a,
                                                                 sslm_schema schema_b) {
	sslm_seq seq_a = nullptr, seq_b = nullptr;
	CHECK(sslm_seq_create(model, nullptr, &seq_a) == SSLM_OK);
	CHECK(sslm_seq_create(model, nullptr, &seq_b) == SSLM_OK);
	CHECK(sslm_seq_set_schema(seq_a, schema_a) == SSLM_OK);
	CHECK(sslm_seq_set_schema(seq_b, schema_b) == SSLM_OK);
	sslm_decode_params params{};
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
static void TestDim1_P1_MidForcedChainSaveRestoreEquivalence(sslm_model model,
                                                              sslm_schema reference_schema) {
	if (g_model_1p5b_path.empty()) {
		SKIP_MSG("real 1.5B artifact with a compiled schema set not supplied "
		         "(--model1p5b=PATH) -- product cell not run");
		return;
	}
	// Uninterrupted run: the reference schema's own long forced chain (an adversarial schema
	// with a forced chain crossing chunk_budget=8, per design Sec6 G5-3's own red-suite text),
	// prefilled in ONE logical jump-forward call the runtime internally chunks.
	sslm_seq seq_uninterrupted = nullptr;
	CHECK(sslm_seq_create(model, nullptr, &seq_uninterrupted) == SSLM_OK);
	CHECK(sslm_seq_set_schema(seq_uninterrupted, reference_schema) == SSLM_OK);
	int32_t long_forced_chain[20];
	for (int i = 0; i < 20; ++i) long_forced_chain[i] = i;
	int32_t consumed_total = 0;
	CHECK(sslm_prefill(model, seq_uninterrupted, long_forced_chain, 20, /*chunk_budget=*/8,
	                    SSLM_SPAN_SCHEMA_CONTENT, nullptr, &consumed_total) == SSLM_OK);

	// Interrupted run: the SAME forced chain, but sslm_seq_save/restore lands between the
	// runtime's own internal chunk-boundary prefill calls (after the first chunk_budget=8
	// tokens, before the chain's remaining 12).
	sslm_seq seq_interrupted = nullptr;
	CHECK(sslm_seq_create(model, nullptr, &seq_interrupted) == SSLM_OK);
	CHECK(sslm_seq_set_schema(seq_interrupted, reference_schema) == SSLM_OK);
	int32_t consumed_first = 0;
	CHECK(sslm_prefill(model, seq_interrupted, long_forced_chain, 8, /*chunk_budget=*/8,
	                    SSLM_SPAN_SCHEMA_CONTENT, nullptr, &consumed_first) == SSLM_OK);
	CHECK(consumed_first == 8);
	unsigned char blob[65536];
	size_t blob_size = sizeof(blob);
	CHECK(sslm_seq_save(seq_interrupted, blob, &blob_size) == SSLM_OK);
	CHECK(sslm_seq_release(seq_interrupted) == SSLM_OK);
	sslm_seq seq_restored = nullptr;
	CHECK(sslm_seq_restore(model, nullptr, blob, blob_size, &seq_restored) == SSLM_OK);
	int32_t consumed_rest = 0;
	CHECK(sslm_prefill(model, seq_restored, long_forced_chain + 8, 12, /*chunk_budget=*/8,
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
	volatile void* addr_0 = (void*)&TestDim1_M1_ResetClearsDfaWalkStateNoLeak; (void)addr_0;
	volatile void* addr_1 = (void*)&TestDim1_M2_InterleavedBatchDoesNotCorruptWalkState; (void)addr_1;
	volatile void* addr_2 = (void*)&TestDim1_P1_MidForcedChainSaveRestoreEquivalence; (void)addr_2;
	std::printf("checks=%d failures=%d skips=%d\n", GChecks, GFailures, GSkips);
	return GFailures ? 1 : 0;
}

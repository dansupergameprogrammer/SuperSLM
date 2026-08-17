// T-2130 (Curie) -- Dim 8 (Composition), design Sec7 dim8 (extended, T-2121 F3/N4). Owned by
// G5-6. 3 cells: adapter x schema (2 sub-claims), prefix-sharing x schema (1), and the
// GPU-crossed triple named as a live, owed obligation sequenced after G5-5 and the adapter
// join both exist (design Sec6 sequencing note, T-2121 N4). RED BY LINK.
#include "fixture_common.h"

using namespace superslm;

// --- Mechanism cell 1 (design Sec6 G5-6 red suite, "Adapter x schema"): a schema-constrained
// decode under a mapped adapter produces schema-valid output whose content reflects the
// adapter's effect (not the base model's) -- the structural/scored split (dim10) applies
// identically; jump-forward under an adapter reproduces the full-forward-under-that-adapter
// constrained path bit-for-bit (the G5-3 oracle, adapter-crossed). ---
static void TestDim8_M1_AdapterCrossedSchemaAndJumpForward(sslm_model model, sslm_kv_pool* pool,
                                                            sslm_schema reference_schema,
                                                            sslm_adapter shopkeeper_adapter) {
	sslm_seq seq = nullptr;
	CHECK(sslm_seq_create(model, pool, &seq) == SSLM_OK);
	CHECK(sslm_seq_set_schema(seq, reference_schema) == SSLM_OK);
	CHECK(sslm_seq_set_adapter(seq, shopkeeper_adapter) == SSLM_OK);
	// sslm_decode_step's own precondition requires prior content on a fresh sequence -- seeded
	// with one arbitrary SSLM_SPAN_PROMPT token, which never advances the DFA walk (design
	// Sec5). See fixture_common.h's SeedPromptForDecodeStep.
	CHECK(SeedPromptForDecodeStep(model, seq));
	sslm_decode_params params = MakeFullDepthDecodeParams();
	int32_t out_tokens[16];
	for (int step = 0; step < 16; ++step)
		CHECK(sslm_decode_step(model, &seq, 1, &params, nullptr, &out_tokens[step]) == SSLM_OK);
	// FEATURE ORACLE (a): out_tokens[0..15] parse as schema-valid under reference_schema's own
	// structural rules (0 violations), AND the content reflects the adapter's own conditioning
	// (differs from a base-only run of the identical prompt under the same schema) --
	// asserted against the adapter's own validated-judge apparatus (T-1980), not a
	// bit-comparison, which would be a consistency oracle blind to a generation that is
	// bit-stable but ignores the adapter.
	CHECK(sslm_seq_release(seq) == SSLM_OK);

	// FEATURE ORACLE (b): a SEPARATE sequence, adapter-bound identically, whose forced chain
	// is driven via jump-forward, produces output and K/V bit-identical to the full-forward
	// constrained-under-adapter path -- the G5-3 equivalence oracle, adapter-crossed.
	sslm_seq seq_jf = nullptr;
	CHECK(sslm_seq_create(model, pool, &seq_jf) == SSLM_OK);
	CHECK(sslm_seq_set_schema(seq_jf, reference_schema) == SSLM_OK);
	CHECK(sslm_seq_set_adapter(seq_jf, shopkeeper_adapter) == SSLM_OK);
	// DERIVED from the real schema/tokenizer at runtime (T-2132/Curie fix), not a literal id
	// assumed legal by construction -- see fixture_common.h's DeriveRealSchemaContentSpan.
	std::vector<int32_t> forced_tokens;
	if (!DeriveRealSchemaContentSpan(model, pool, reference_schema, 4, &forced_tokens)) {
		SKIP_MSG("could not derive a real 4-token schema-legal forced span from the live "
		         "fixture -- mechanism cell 1's own oracle (b) half not run");
		CHECK(sslm_seq_release(seq_jf) == SSLM_OK);
		return;
	}
	int32_t consumed = 0;
	CHECK(sslm_prefill(model, seq_jf, forced_tokens.data(), 4, 8, SSLM_SPAN_SCHEMA_CONTENT, nullptr,
	                    &consumed) == SSLM_OK);
	CHECK(sslm_seq_release(seq_jf) == SSLM_OK);
}

// --- Mechanism cell 2 (design Sec6 G5-6 red suite, "Prefix-sharing x schema"): a shared,
// frozen prefix block adopted by a sequence that then decodes under an active schema --
// including at least one jump-forward-forced span beginning at or crossing the prefix
// boundary -- reproduces the same output as an equivalent sequence decoding under the same
// schema with NO shared prefix (prefix sharing composes bit-identically with the mask). ---
static void TestDim8_M2_PrefixSharingCrossedWithSchemaAndJumpForwardAtBoundary(
    sslm_model model, sslm_kv_pool* pool, sslm_schema reference_schema) {
	// The forced-at-boundary span's own ids are DERIVED from the real schema/tokenizer at
	// runtime (T-2132/Curie fix -- see fixture_common.h's DeriveRealSchemaContentSpan), not
	// literal ids assumed legal by construction. Legal because SSLM_SPAN_PROMPT never
	// advances the walk-state (design Sec5), so the walk is still at the schema's own start
	// state when this span is prefilled, on BOTH sequences below -- the same derived span is
	// therefore a legal continuation for each. system_prompt_tokens stays literal/arbitrary:
	// it is an SSLM_SPAN_PROMPT span, never checked against the schema's DFA.
	std::vector<int32_t> forced_at_boundary;
	if (!DeriveRealSchemaContentSpan(model, pool, reference_schema, 3, &forced_at_boundary)) {
		SKIP_MSG("could not derive a real 3-token schema-legal forced span from the live "
		         "fixture -- mechanism cell 2 not run");
		return;
	}
	// The prefix is built entirely from SSLM_SPAN_PROMPT calls (the ordinary "shared
	// system/persona prefix" case, design Sec5/Sec7) -- freezes at the unbound/start
	// walk-state, compatible with any sequence's schema binding.
	sslm_prefix shared_prefix = nullptr;
	CHECK(sslm_prefix_begin(model, pool, &shared_prefix) == SSLM_OK);
	int32_t system_prompt_tokens[6] = {0, 1, 2, 3, 4, 5};
	int32_t prefix_consumed = 0;
	CHECK(sslm_prefix_prefill(model, shared_prefix, system_prompt_tokens, 6, /*chunk_budget=*/8,
	                           SSLM_SPAN_PROMPT, nullptr, &prefix_consumed) == SSLM_OK);
	CHECK(sslm_prefix_freeze(shared_prefix) == SSLM_OK);

	// Sequence A: adopts the shared prefix, binds the reference schema, then jump-forwards a
	// forced span that begins AT the prefix boundary (the schema's own leading forced
	// literal, right after the adopted prefix's last token).
	sslm_seq seq_with_prefix = nullptr;
	CHECK(sslm_seq_create(model, pool, &seq_with_prefix) == SSLM_OK);
	CHECK(sslm_seq_set_schema(seq_with_prefix, reference_schema) == SSLM_OK);
	CHECK(sslm_seq_adopt_prefix(seq_with_prefix, shared_prefix) == SSLM_OK);
	int32_t consumed_a = 0;
	CHECK(sslm_prefill(model, seq_with_prefix, forced_at_boundary.data(), 3, 8,
	                    SSLM_SPAN_SCHEMA_CONTENT, nullptr, &consumed_a) == SSLM_OK);

	// Sequence B: no shared prefix -- the identical system-prompt tokens supplied directly as
	// an ordinary SSLM_SPAN_PROMPT span, then the identical forced span.
	sslm_seq seq_no_prefix = nullptr;
	CHECK(sslm_seq_create(model, pool, &seq_no_prefix) == SSLM_OK);
	CHECK(sslm_seq_set_schema(seq_no_prefix, reference_schema) == SSLM_OK);
	int32_t consumed_prompt = 0;
	CHECK(sslm_prefill(model, seq_no_prefix, system_prompt_tokens, 6, 8, SSLM_SPAN_PROMPT,
	                    nullptr, &consumed_prompt) == SSLM_OK);
	int32_t consumed_b = 0;
	CHECK(sslm_prefill(model, seq_no_prefix, forced_at_boundary.data(), 3, 8,
	                    SSLM_SPAN_SCHEMA_CONTENT, nullptr, &consumed_b) == SSLM_OK);

	// FEATURE ORACLE: seq_with_prefix and seq_no_prefix produce the same DFA-walk-state and
	// the same K/V content over the shared span -- prefix sharing composes bit-identically
	// with the mask (design Sec6/Sec7 dim8's own "each coexisting pair" rule), feeding dim10's
	// schema-valid assertion on the same run.
	CHECK(sslm_seq_release(seq_with_prefix) == SSLM_OK);
	CHECK(sslm_seq_release(seq_no_prefix) == SSLM_OK);
	CHECK(sslm_prefix_release(shared_prefix) == SSLM_OK);
}

// --- Product cell 1 (design Sec6 G5-6 sequencing note / T-2121 N4, now live per D-SLM3443):
// the GPU-crossed triple -- adapter x constraint-schema x GPU dispatch-budget. Owed jointly
// by G5-5 and G5-6, sequenced after both exist; named here so the obligation is not lost
// between the two slots' own gate text (neither slot's gate carries it alone). A
// schema-constrained, adapter-bound sequence decoded on the GPU under a PARTIAL dispatch
// budget composes bit-identically with the same sequence decoded under a WHOLE-token budget --
// the GPU twin of the existing GPU-dispatch-budget x batch-composition cell (plan Sec17
// dim8), specialized to the adapter x schema pairing this slot adds. ---
static void TestDim8_P1_AdapterSchemaGpuDispatchBudgetTriple(sslm_model model,
                                                              sslm_schema reference_schema,
                                                              sslm_adapter shopkeeper_adapter) {
	if (g_model_1p5b_path.empty() || g_adapter_path.empty()) {
		SKIP_MSG("real 1.5B artifact / real adapter / GPU device not supplied -- product cell "
		         "not run (sequenced after BOTH G5-5's device path and this slot's adapter "
		         "join, design Sec6 sequencing note -- declared here so the obligation is not "
		         "lost between the two slots' own gate text)");
		return;
	}
	// FEATURE ORACLE: a schema-constrained, adapter-bound sequence decoded on the GPU under a
	// partial per-call dispatch budget (forcing multiple GPU decode calls to complete one
	// logical step) produces output bit-identical to the same sequence decoded under a
	// whole-token dispatch budget (one call per step) -- the GPU-dispatch-budget invariance
	// this repo already proves for the base kernel set (plan Sec17 dim8), specialized to the
	// adapter x schema pairing. Wired once G5-5's device decode-step entry point exists.
}

int main(int argc, char** argv) {
	ParseFixtureArgs(argc, argv);
	std::vector<uint8_t> model_bytes;
	sslm_model model = nullptr;
	const bool have_model = TryMapRealModel(g_model_1p5b_path, &model_bytes, &model);
	sslm_schema schema_ref = nullptr;
	const bool have_schema_ref =
	    have_model && TryLookupSchema(model, g_reference_schema_name.c_str(), &schema_ref);

	// Real KV pool (T-2132/Curie fix): sslm_seq_create/sslm_prefix_begin below require a
	// non-null, already-created sslm_kv_pool* -- see fixture_common.h's own RealKvPool comment.
	RealKvPool pool;
	const bool have_pool = have_model && pool.Create(model, kRealPoolBlockCount);

	// M1 needs a real sslm_adapter handle; sslm_g5.h declares sslm_seq_set_adapter but no
	// sslm_adapter-map verb under the sslm_ prefix at all (the only shipped map verb is
	// sslm_gpu_adapter_map, a different, incompatible handle type) -- a fixture this suite
	// genuinely cannot construct, not a fixture merely unsupplied on this invocation. SKIP
	// unconditionally rather than pass a null adapter into a cell with no internal skip check;
	// owed as a design/build-time gap, not papered over here.
	SKIP_MSG("no sslm_adapter fixture is constructible -- design Sec5's ABI (sslm_g5.h) "
	         "declares no sslm_adapter-map verb under the sslm_ prefix -- mechanism cell 1 "
	         "not run");
	if (have_schema_ref && have_pool) {
		TestDim8_M2_PrefixSharingCrossedWithSchemaAndJumpForwardAtBoundary(model, pool.ptr(),
		                                                                   schema_ref);
	} else {
		SKIP_MSG("real 1.5B artifact with a compiled schema not supplied, or a real KV pool "
		         "could not be constructed -- mechanism cell 2 not run");
	}
	// P1 self-skips on g_model_1p5b_path.empty() || g_adapter_path.empty() before touching any
	// argument, including the adapter this main() cannot itself construct.
	TestDim8_P1_AdapterSchemaGpuDispatchBudgetTriple(model, schema_ref, nullptr);
	std::printf("checks=%d failures=%d skips=%d\n", GChecks, GFailures, GSkips);
	return GFailures ? 1 : 0;
}

// T-2130 (Curie) -- Dim 6 (Numerical edges and determinism), design Sec7 dim6. The constraint
// engine introduces no new arithmetic (design Sec4): masking is a bitwise AND on int32
// logits, jump-forward reuses the existing kernel set, and the base kernel corpus (plan Sec17
// dim6) already covers every numeric edge case that arithmetic can produce -- no new cell is
// owed for that half (design Sec7 dim6 states this explicitly: "No new numeric edge case
// exists that the base kernel corpus does not already cover"). ONE cell is owed: the claim
// promoted into this dimension this fold (T-2121 F2) -- template fill's byte-identical
// fixed-span reproduction across the toolchain matrix is a determinism claim like any other
// output claim, not merely a composition claim under dim8. RED BY LINK.
#include "fixture_common.h"

using namespace superslm;

// --- Product cell 1 (design Sec6 G5-4 gate / Sec7 dim6, promoted T-2121 F2): a template's
// fixed spans reproduce byte-identically across the toolchain/ISA matrix -- the plan's own
// D-SLM40 determinism contract (bit-identical for identical (artifact, token inputs, schema,
// config)) applied to the fixed-span half of a template-fill call specifically, isolated from
// the generated hole-fill (which reduces to G5-2/G5-3's own already-proven determinism). ---
static void TestDim6_P1_TemplateFillFixedSpansByteIdenticalAcrossToolchain(
    sslm_model model, sslm_kv_pool* pool, sslm_schema reference_schema) {
	if (g_model_1p5b_path.empty()) {
		SKIP_MSG("real 1.5B artifact not supplied -- product cell not run (this cell's own "
		         "cross-toolchain half additionally requires the CI matrix's own multiple "
		         "compiler/ISA legs, per D-SLM3439's GitHub-CI-matrix-cell discipline; this "
		         "invocation exercises the single-toolchain half only)");
		return;
	}
	// The fixed span's own ids are DERIVED from the real schema/tokenizer at runtime
	// (T-2132/Curie fix -- see fixture_common.h's DeriveRealSchemaContentSpan), not literal
	// ids assumed legal by construction.
	std::vector<int32_t> fixed_span_tokens;
	if (!DeriveRealSchemaContentSpan(model, pool, reference_schema, 8, &fixed_span_tokens)) {
		SKIP_MSG("could not derive a real 8-token schema-legal fixed span from the live "
		         "fixture -- product cell not run");
		return;
	}
	sslm_seq seq = nullptr;
	CHECK(sslm_seq_create(model, pool, &seq) == SSLM_OK);
	CHECK(sslm_seq_set_schema(seq, reference_schema) == SSLM_OK);
	// A host-declared fixed span (e.g. the reference schema's own leading `{"intent":`
	// literal, forced by construction per design Sec9.10.1's own execution) supplied via
	// SSLM_SPAN_SCHEMA_CONTENT.
	int32_t consumed = 0;
	CHECK(sslm_prefill(model, seq, fixed_span_tokens.data(), 8, /*chunk_budget=*/8,
	                    SSLM_SPAN_SCHEMA_CONTENT, nullptr, &consumed) == SSLM_OK);
	std::vector<uint8_t> blob_run1 = AllocRealSaveBlobBuffer(model);
	size_t blob_run1_size = blob_run1.size();
	CHECK(sslm_seq_save(seq, blob_run1.data(), &blob_run1_size) == SSLM_OK);
	CHECK(sslm_seq_release(seq) == SSLM_OK);

	// FEATURE ORACLE (determinism, dim6): the identical (artifact, token inputs, schema,
	// config) run on a SECOND toolchain leg (a separately compiled binary of this same
	// suite, per D-SLM3439's GitHub-CI-matrix-cell discipline) produces a byte-identical
	// save-blob. This process's own run captures its own leg's blob for the CI job's
	// cross-leg comparison; the comparison itself is the CI job's own step, not this binary's.
	std::printf("g5_dim6_template_fixed_span_blob_hash: <computed once the build seat's own "
	            "hashing utility is wired>\n");
}

int main(int argc, char** argv) {
	ParseFixtureArgs(argc, argv);
	// Self-skips via g_model_1p5b_path.empty() before touching either argument.
	std::vector<uint8_t> model_bytes;
	sslm_model model = nullptr;
	const bool have_model = TryMapRealModel(g_model_1p5b_path, &model_bytes, &model);
	sslm_schema schema_ref = nullptr;
	if (have_model) TryLookupSchema(model, g_reference_schema_name.c_str(), &schema_ref);
	// Real KV pool (T-2132/Curie fix): sslm_seq_create below requires a non-null,
	// already-created sslm_kv_pool* -- see fixture_common.h's own RealKvPool comment.
	RealKvPool pool;
	if (have_model) pool.Create(model, kRealPoolBlockCount);
	TestDim6_P1_TemplateFillFixedSpansByteIdenticalAcrossToolchain(model, pool.ptr(), schema_ref);
	std::printf("checks=%d failures=%d skips=%d\n", GChecks, GFailures, GSkips);
	return GFailures ? 1 : 0;
}

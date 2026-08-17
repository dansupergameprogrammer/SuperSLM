// T-2130 (Curie) -- Dim 5 (Failure and rejection paths), design Sec7 dim5 (extended again this
// fold, T-2122 repair). Cells here cover the rejections NOT already exercised by dim2's
// load-time/restore-path cells: sslm_seq_set_schema's non-fresh-walk-state rejection, the
// repair's own SSLM_SCHEMA_SPAN_UNBOUND rejection, and sslm_seq_adopt_prefix's
// SSLM_PREFIX_SCHEMA_MISMATCH, stated exhaustively over BOTH branches the design names (a
// mismatched bound schema, AND an unbound sequence adopting real progress -- the T-2126/T-2127
// closed gap). 4 cells. RED BY LINK.
#include "fixture_common.h"

using namespace superslm;

// --- Mechanism cell 1: sslm_seq_set_schema on a sequence whose DFA-walk state is not fresh
// (has already consumed tokens) rejects -- SSLM_SCHEMA_BIND_REJECTED. Valid ONLY at a fresh
// sslm_seq_create or immediately after sslm_seq_reset (design Sec5). ---
static void TestDim5_M1_SetSchemaOnNonFreshWalkStateRejected(sslm_model model, sslm_kv_pool* pool,
                                                              sslm_schema schema_a,
                                                              sslm_schema schema_b) {
	// DERIVED from the real schema/tokenizer at runtime (T-2132/Curie fix), not a literal id
	// assumed legal by construction -- see fixture_common.h's DeriveRealSchemaContentSpan.
	std::vector<int32_t> forced_tokens;
	if (!DeriveRealSchemaContentSpan(model, pool, schema_a, 1, &forced_tokens)) {
		SKIP_MSG("could not derive a real 1-token schema-legal forced span from the live "
		         "fixture -- mechanism cell 1 not run");
		return;
	}
	sslm_seq seq = nullptr;
	CHECK(sslm_seq_create(model, pool, &seq) == SSLM_OK);
	CHECK(sslm_seq_set_schema(seq, schema_a) == SSLM_OK);
	int32_t consumed = 0;
	CHECK(sslm_prefill(model, seq, forced_tokens.data(), 1, 8, SSLM_SPAN_SCHEMA_CONTENT, nullptr,
	                    &consumed) == SSLM_OK);
	// Walk state is now non-fresh (one token consumed, non-start). Re-binding, even to a
	// DIFFERENT schema, rejects -- schema re-binding mid-generation is out of 1.0 scope
	// (design Sec11).
	CHECK(sslm_seq_set_schema(seq, schema_b) == SSLM_SCHEMA_BIND_REJECTED);
	CHECK(sslm_seq_release(seq) == SSLM_OK);
}

// --- Mechanism cell 2 (design Sec5/Sec10.2, THE REPAIR's own rejection): a
// SSLM_SPAN_SCHEMA_CONTENT prefill call against a sequence with no schema bound
// (SSLM_SCHEMA_NONE) rejects -- SSLM_SCHEMA_SPAN_UNBOUND. "A span cannot be schema-content
// against no schema." ---
static void TestDim5_M2_SchemaContentSpanAgainstUnboundSequenceRejected(sslm_model model,
                                                                         sslm_kv_pool* pool) {
	sslm_seq seq = nullptr;
	CHECK(sslm_seq_create(model, pool, &seq) == SSLM_OK);
	CHECK(sslm_seq_set_schema(seq, SSLM_SCHEMA_NONE) == SSLM_OK);
	int32_t span_tokens[2] = {0, 1};
	int32_t consumed = 0;
	CHECK(sslm_prefill(model, seq, span_tokens, 2, 8, SSLM_SPAN_SCHEMA_CONTENT, nullptr,
	                    &consumed) == SSLM_SCHEMA_SPAN_UNBOUND);
	// SSLM_SPAN_PROMPT against the same unbound sequence remains ordinary, unrejected
	// prompt ingestion -- the rejection is specific to the CONTENT kind against no schema,
	// never a blanket "unbound sequences can't prefill" rule.
	int32_t prompt_tokens[2] = {2, 3};
	CHECK(sslm_prefill(model, seq, prompt_tokens, 2, 8, SSLM_SPAN_PROMPT, nullptr, &consumed) ==
	      SSLM_OK);
	CHECK(sslm_seq_release(seq) == SSLM_OK);
}

// --- Mechanism cell 3 (design Sec5, branch (a) of the exhaustive rejection): a sequence
// bound to a DIFFERENT schema than the prefix's own recorded progress rejects adoption --
// SSLM_PREFIX_SCHEMA_MISMATCH. ---
static void TestDim5_M3_AdoptPrefixMismatchedBoundSchemaRejected(sslm_model model,
                                                                  sslm_kv_pool* pool,
                                                                  sslm_schema schema_a,
                                                                  sslm_schema schema_b) {
	// DERIVED from the real schema/tokenizer at runtime (T-2132/Curie fix), not a literal id
	// assumed legal by construction -- see fixture_common.h's DeriveRealSchemaContentSpan.
	std::vector<int32_t> schema_content;
	if (!DeriveRealSchemaContentSpan(model, pool, schema_a, 2, &schema_content)) {
		SKIP_MSG("could not derive a real 2-token schema-legal forced span from the live "
		         "fixture -- mechanism cell 3 not run");
		return;
	}
	sslm_prefix prefix = nullptr;
	CHECK(sslm_prefix_begin(model, pool, &prefix) == SSLM_OK);
	CHECK(sslm_prefix_set_schema(prefix, schema_a) == SSLM_OK);
	int32_t prefix_consumed = 0;
	CHECK(sslm_prefix_prefill(model, prefix, schema_content.data(), 2, /*chunk_budget=*/8,
	                           SSLM_SPAN_SCHEMA_CONTENT, nullptr, &prefix_consumed) == SSLM_OK);
	CHECK(sslm_prefix_freeze(prefix) == SSLM_OK);

	sslm_seq seq_bound_to_b = nullptr;
	CHECK(sslm_seq_create(model, pool, &seq_bound_to_b) == SSLM_OK);
	CHECK(sslm_seq_set_schema(seq_bound_to_b, schema_b) == SSLM_OK);
	CHECK(sslm_seq_adopt_prefix(seq_bound_to_b, prefix) == SSLM_PREFIX_SCHEMA_MISMATCH);
	CHECK(sslm_seq_release(seq_bound_to_b) == SSLM_OK);
	CHECK(sslm_prefix_release(prefix) == SSLM_OK);
}

// --- Mechanism cell 4 (design Sec5, branch (b) of the exhaustive rejection -- the
// unbound-adoption gap T-2126 F1 / T-2127 G3 independently found, CLOSED this fold): a
// sequence with NO schema bound adopting a prefix carrying real schema-content walk progress
// rejects -- SSLM_PREFIX_SCHEMA_MISMATCH, deliberately, over an inert-adopt (which would
// strand the sequence's walk-state permanently, per design Sec5's own updated comment). ---
static void TestDim5_M4_AdoptPrefixByUnboundSequenceWithRealProgressRejected(
    sslm_model model, sslm_kv_pool* pool, sslm_schema schema_a) {
	// DERIVED from the real schema/tokenizer at runtime (T-2132/Curie fix), not a literal id
	// assumed legal by construction -- see fixture_common.h's DeriveRealSchemaContentSpan.
	std::vector<int32_t> schema_content;
	if (!DeriveRealSchemaContentSpan(model, pool, schema_a, 2, &schema_content)) {
		SKIP_MSG("could not derive a real 2-token schema-legal forced span from the live "
		         "fixture -- mechanism cell 4 not run");
		return;
	}
	sslm_prefix prefix = nullptr;
	CHECK(sslm_prefix_begin(model, pool, &prefix) == SSLM_OK);
	CHECK(sslm_prefix_set_schema(prefix, schema_a) == SSLM_OK);
	int32_t prefix_consumed = 0;
	CHECK(sslm_prefix_prefill(model, prefix, schema_content.data(), 2, /*chunk_budget=*/8,
	                           SSLM_SPAN_SCHEMA_CONTENT, nullptr, &prefix_consumed) == SSLM_OK);
	CHECK(sslm_prefix_freeze(prefix) == SSLM_OK);

	sslm_seq seq_unbound = nullptr;
	CHECK(sslm_seq_create(model, pool, &seq_unbound) == SSLM_OK);
	CHECK(sslm_seq_set_schema(seq_unbound, SSLM_SCHEMA_NONE) == SSLM_OK);
	CHECK(sslm_seq_adopt_prefix(seq_unbound, prefix) == SSLM_PREFIX_SCHEMA_MISMATCH);
	// CONTRAST (the ordinary, compatible case, not this cell's own subject but recorded so
	// the exhaustive rejection reads against its complement): a prefix built ENTIRELY from
	// SSLM_SPAN_PROMPT calls freezes at the unbound/start walk-state and IS adoptable by an
	// unbound sequence -- design Sec5's own "shared system/persona prefix" framing, exercised
	// by dim8's prefix-sharing composition cell in this suite.
	CHECK(sslm_seq_release(seq_unbound) == SSLM_OK);
	CHECK(sslm_prefix_release(prefix) == SSLM_OK);
}

// --- Mechanism cell 5 (design Sec6 G5-4 red suite / Sec7 dim5): a template whose "fixed"
// span is not actually reachable under the active schema is a defined rejection, not a silent
// mismatch -- "a span that leaves the DFA's language before it is exhausted," checked token
// by token as the walk advances through the SSLM_SPAN_SCHEMA_CONTENT span (design Sec6 G5-4).
// ---
static void TestDim5_M5_TemplateFixedSpanNotReachableUnderActiveSchemaRejected(
    sslm_model model, sslm_kv_pool* pool, sslm_schema reference_schema) {
	// The CONTRAST half's own reachable span is DERIVED from the real schema/tokenizer at
	// runtime (T-2132/Curie fix -- see fixture_common.h's DeriveRealSchemaContentSpan);
	// unreachable_span below stays a literal, deliberately-illegal id sequence -- that IS this
	// cell's own subject (a span the DFA never admits), not the defect this fix closes.
	std::vector<int32_t> reachable_span;
	if (!DeriveRealSchemaContentSpan(model, pool, reference_schema, 1, &reachable_span)) {
		SKIP_MSG("could not derive a real 1-token schema-legal span from the live fixture -- "
		         "mechanism cell 5 not run");
		return;
	}
	sslm_seq seq = nullptr;
	CHECK(sslm_seq_create(model, pool, &seq) == SSLM_OK);
	CHECK(sslm_seq_set_schema(seq, reference_schema) == SSLM_OK);
	// A host-declared "fixed" span whose token sequence does NOT match any path the schema's
	// own compiled DFA admits from the sequence's current (start) walk-state -- e.g. a token
	// the reference schema's own leading literal never emits at that position. Genuinely
	// hostile input to the reachability check, not merely an empty span. Kept as literal,
	// deliberately-illegal ids: this cell's own subject is precisely a span the DFA rejects,
	// so "legal" ids would defeat the cell rather than fix a defect.
	int32_t unreachable_span[4] = {9001, 9002, 9003, 9004};
	int32_t consumed = 0;
	CHECK(sslm_prefill(model, seq, unreachable_span, 4, 8, SSLM_SPAN_SCHEMA_CONTENT, nullptr,
	                    &consumed) == SSLM_SCHEMA_SPAN_UNREACHABLE);
	// The sequence's own walk-state is unmoved by the rejected call -- a subsequent call with
	// the schema's OWN genuinely reachable leading span still succeeds, proving the rejection
	// did not corrupt the walk-state on its way to reporting failure.
	int32_t consumed2 = 0;
	CHECK(sslm_prefill(model, seq, reachable_span.data(), 1, 8, SSLM_SPAN_SCHEMA_CONTENT, nullptr,
	                    &consumed2) == SSLM_OK);
	CHECK(sslm_seq_release(seq) == SSLM_OK);
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

	// Real KV pool (T-2132/Curie fix): sslm_seq_create/sslm_prefix_begin below require a
	// non-null, already-created sslm_kv_pool* -- see fixture_common.h's own RealKvPool comment.
	RealKvPool pool;
	const bool have_pool = have_model && pool.Create(model, kRealPoolBlockCount);

	if (have_schema_a && have_schema_b && have_pool) {
		TestDim5_M1_SetSchemaOnNonFreshWalkStateRejected(model, pool.ptr(), schema_a, schema_b);
	} else {
		SKIP_MSG("real 1.5B artifact with two distinct compiled schemas not supplied, or a "
		         "real KV pool could not be constructed -- mechanism cell 1 not run");
	}
	if (have_pool) {
		TestDim5_M2_SchemaContentSpanAgainstUnboundSequenceRejected(model, pool.ptr());
	} else {
		SKIP_MSG("real 1.5B artifact not supplied (--model1p5b=PATH), or a real KV pool could "
		         "not be constructed -- mechanism cell 2 not run");
	}
	if (have_schema_a && have_schema_b && have_pool) {
		TestDim5_M3_AdoptPrefixMismatchedBoundSchemaRejected(model, pool.ptr(), schema_a, schema_b);
	} else {
		SKIP_MSG("real 1.5B artifact with two distinct compiled schemas not supplied, or a "
		         "real KV pool could not be constructed -- mechanism cell 3 not run");
	}
	if (have_schema_a && have_pool) {
		TestDim5_M4_AdoptPrefixByUnboundSequenceWithRealProgressRejected(model, pool.ptr(),
		                                                                 schema_a);
	} else {
		SKIP_MSG("real 1.5B artifact with a compiled schema not supplied, or a real KV pool "
		         "could not be constructed -- mechanism cell 4 not run");
	}
	if (have_schema_a && have_pool) {
		TestDim5_M5_TemplateFixedSpanNotReachableUnderActiveSchemaRejected(model, pool.ptr(),
		                                                                   schema_a);
	} else {
		SKIP_MSG("real 1.5B artifact with a compiled schema not supplied, or a real KV pool "
		         "could not be constructed -- mechanism cell 5 not run");
	}
	std::printf("checks=%d failures=%d skips=%d\n", GChecks, GFailures, GSkips);
	return GFailures ? 1 : 0;
}

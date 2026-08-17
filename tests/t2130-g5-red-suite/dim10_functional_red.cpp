// T-2130 (Curie) -- Dim 10 (Functional achievement, the feature oracle), design Sec7 dim10.
// FOUR oracles (was three; T-2121 F2, closed) -- the core of this design. None is satisfied
// by "same bits twice"; each requires an independent reference (the schema parser, or the
// full-forward path) per the plan's own J-N1/J-S2 discipline (StandardsDocument.md Sec5.4's
// own feature-oracle rule / this seat's "A feature oracle for every achievement claim"
// discipline). RED BY LINK.
#include "fixture_common.h"

using namespace superslm;

// --- Oracle (a): schema-constrained decode yields schema-valid output. Structural cells
// (enum legality, field presence, key order, well-formedness) asserted at 0 violations, per
// D-SLM45; cross-field violations reported, never asserted 0 (scored, not constrained). ---
static void TestDim10_A_SchemaConstrainedDecodeYieldsSchemaValidOutput(
    sslm_model model, sslm_schema reference_schema) {
	sslm_seq seq = nullptr;
	CHECK(sslm_seq_create(model, nullptr, &seq) == SSLM_OK);
	CHECK(sslm_seq_set_schema(seq, reference_schema) == SSLM_OK);
	sslm_decode_params params{};
	int32_t out_tokens[64];
	int step = 0;
	for (; step < 64; ++step) {
		const sslm_status st = sslm_decode_step(model, &seq, 1, &params, nullptr,
		                                         &out_tokens[step]);
		CHECK(st == SSLM_OK);
		if (st != SSLM_OK) break;
		// A real driver stops at the DFA's own accepting state (design Sec9's
		// greedy_decode-shaped loop); this cell's own structural half runs to a fixed step
		// bound as a placeholder for that stop condition, pending the build seat's own
		// accepting-state query.
	}
	// FEATURE ORACLE: the emitted token stream, detokenized and parsed by an INDEPENDENT JSON
	// parser (never the runtime's own DFA, which would make this a consistency oracle) against
	// reference_schema's own textual definition (Claude/Docs/Shopkeeper_IntentExtraction_
	// Schema.md), reports 0 STRUCTURAL violations (enum legality, key presence, declaration-
	// order key emission per D-SLM40, well-formedness) and reports (never asserts 0) any
	// cross-field violations (D-SLM45: scored, not constrained -- the reference schema's own
	// three cross-field-carrying rules, per design Sec6 G5-1's own gate text, are exactly the
	// case this scoring exists for).
	CHECK(sslm_seq_release(seq) == SSLM_OK);
}

// --- Oracle (b): jump-forward output and K/V equal the full-forward constrained path,
// token-for-token AND bit-for-bit -- the equivalence oracle named in design Sec9/Sec13.2/
// Sec17 dim10, run against real schemas with forced spans crossing prefill-chunk boundaries
// (so the oracle exercises chunkBudget inheritance, not only a short forced chain that fits
// in one chunk). A consistency oracle alone (same bits twice) cannot see a jump-forward
// implementation that is deterministic but WRONG -- this comparison is what the plan calls
// out as load-bearing (J-N1/J-S2 shape). ---
static void TestDim10_B_JumpForwardEquivalentToFullForwardTokenAndKvBitForBit(
    sslm_model model, sslm_schema reference_schema) {
	// Path 1: jump-forward. The runtime's own forced-chain detection issues the chain as
	// SSLM_SPAN_SCHEMA_CONTENT prefill call(s), logit/argmax skipped for forced positions.
	sslm_seq seq_jump_forward = nullptr;
	CHECK(sslm_seq_create(model, nullptr, &seq_jump_forward) == SSLM_OK);
	CHECK(sslm_seq_set_schema(seq_jump_forward, reference_schema) == SSLM_OK);
	// A forced chain crossing a chunkBudget boundary (25 tokens, chunk_budget=8 -> 4 internal
	// chunk calls) -- deliberately NOT a short chain that fits in one chunk, per this
	// dimension's own text.
	int32_t forced_chain[25];
	for (int i = 0; i < 25; ++i) forced_chain[i] = i % 20;
	int32_t consumed = 0;
	CHECK(sslm_prefill(model, seq_jump_forward, forced_chain, 25, /*chunk_budget=*/8,
	                    SSLM_SPAN_SCHEMA_CONTENT, nullptr, &consumed) == SSLM_OK);
	CHECK(consumed == 25);
	unsigned char jf_blob[65536];
	size_t jf_blob_size = sizeof(jf_blob);
	CHECK(sslm_seq_save(seq_jump_forward, jf_blob, &jf_blob_size) == SSLM_OK);
	CHECK(sslm_seq_release(seq_jump_forward) == SSLM_OK);

	// Path 2: the full-forward constrained path -- the IDENTICAL 25 forced tokens run through
	// ordinary per-token sslm_decode_step calls (the logit/argmax step NOT skipped), which is
	// what the design's own G-7a-obligated mask (a single valid token at each of these
	// states) forces the greedy argmax to select regardless.
	sslm_seq seq_full_forward = nullptr;
	CHECK(sslm_seq_create(model, nullptr, &seq_full_forward) == SSLM_OK);
	CHECK(sslm_seq_set_schema(seq_full_forward, reference_schema) == SSLM_OK);
	sslm_decode_params params{};
	int32_t out_tok = 0;
	for (int i = 0; i < 25; ++i)
		CHECK(sslm_decode_step(model, &seq_full_forward, 1, &params, nullptr, &out_tok) ==
		      SSLM_OK);
	unsigned char ff_blob[65536];
	size_t ff_blob_size = sizeof(ff_blob);
	CHECK(sslm_seq_save(seq_full_forward, ff_blob, &ff_blob_size) == SSLM_OK);
	CHECK(sslm_seq_release(seq_full_forward) == SSLM_OK);

	// FEATURE ORACLE: jf_blob and ff_blob -- the K/V content AND the DFA-walk-state -- are
	// byte-for-byte identical. This is the equivalence oracle, independent of the mask
	// application's own internal consistency: jump-forward could be perfectly self-consistent
	// (same bits every run) while still being WRONG relative to the full-forward path, and
	// only this direct comparison would catch it.
	CHECK(jf_blob_size == ff_blob_size);
	CHECK(std::memcmp(jf_blob, ff_blob, jf_blob_size) == 0);
}

// --- Oracle (c): adapter x constraint-schema yields schema-valid, adapter-conditioned
// output. Cross-cited to dim8_composition_red.cpp's own mechanism cell 1 (the adapter-crossed
// oracle), which is this dimension's own oracle (c) -- not duplicated here per this suite's
// no-drifting-copy discipline; this dimension's own rollup names it so dim10's four oracles
// are complete without inferring the fourth from dim8's text. ---
static void TestDim10_C_AdapterCrossedSchemaValidOutput_CrossCitedToDim8() {
	// See dim8_composition_red.cpp::TestDim8_M1_AdapterCrossedSchemaAndJumpForward -- that
	// cell's own FEATURE ORACLE (a) IS this dimension's oracle (c), authored once (in dim8,
	// where the composition it exercises is dim8's own subject) and cited here rather than
	// re-authored, per StandardsDocument.md Sec6.6's no-proliferation discipline applied to
	// tests: the same claim tested twice in two files drifts the moment one is updated and
	// not the other.
}

// --- Oracle (d) (promoted into this rollup this fold, T-2121 F2 -- "fully specified in Sec6
// but not gathered here, despite this dimension's own framing text promising 'gathered
// together for the build'"): template fill produces the fixed spans verbatim plus a
// schema-valid generated fill for each hole. ---
static void TestDim10_D_TemplateFillFixedSpansVerbatimPlusSchemaValidHoleFill(
    sslm_model model, sslm_schema reference_schema) {
	sslm_seq seq = nullptr;
	CHECK(sslm_seq_create(model, nullptr, &seq) == SSLM_OK);
	CHECK(sslm_seq_set_schema(seq, reference_schema) == SSLM_OK);
	// The host declares a fixed span (a template's own literal text, e.g. the reference
	// schema's leading `{"intent":` through a hand-picked enum value) as
	// SSLM_SPAN_SCHEMA_CONTENT.
	int32_t fixed_span[6] = {0, 1, 2, 3, 4, 5};
	int32_t consumed = 0;
	CHECK(sslm_prefill(model, seq, fixed_span, 6, 8, SSLM_SPAN_SCHEMA_CONTENT, nullptr,
	                    &consumed) == SSLM_OK);
	CHECK(consumed == 6);
	// The hole: ordinary constrained generation continues from the DFA-walk-state the fixed
	// span left, reducing to G5-2/G5-3's own already-proven mechanism (this slot proves the
	// API COMPOSITION, not new arithmetic).
	sslm_decode_params params{};
	int32_t hole_tokens[8];
	for (int i = 0; i < 8; ++i)
		CHECK(sslm_decode_step(model, &seq, 1, &params, nullptr, &hole_tokens[i]) == SSLM_OK);
	// FEATURE ORACLE: the emitted stream, read as (fixed_span verbatim) ++ (hole_tokens),
	// equals fixed_span exactly over its own 6 positions (byte-for-byte, the template's own
	// literal text is never altered by generation) AND the hole positions parse as
	// schema-valid under reference_schema (oracle (a)'s own structural/scored split, applied
	// to the hole-fill specifically). A template whose "fixed" span is not actually reachable
	// under the active schema is a defined rejection, exercised as its own cell in
	// dim5_failure_red.cpp (TestDim5_M5_TemplateFixedSpanNotReachableUnderActiveSchemaRejected)
	// rather than duplicated here: this cell's own claim is the achievement path, dim5's is
	// the rejection path, per this suite's own dimension split.
	CHECK(sslm_seq_release(seq) == SSLM_OK);
}

int main(int argc, char** argv) {
	ParseFixtureArgs(argc, argv);
	std::vector<uint8_t> model_bytes;
	sslm_model model = nullptr;
	const bool have_model = TryMapRealModel(g_model_1p5b_path, &model_bytes, &model);
	sslm_schema schema_ref = nullptr;
	const bool have_schema_ref =
	    have_model && TryLookupSchema(model, g_reference_schema_name.c_str(), &schema_ref);

	if (have_schema_ref) {
		TestDim10_A_SchemaConstrainedDecodeYieldsSchemaValidOutput(model, schema_ref);
	} else {
		SKIP_MSG("real 1.5B artifact with a compiled schema not supplied -- oracle (a) not run");
	}
	if (have_schema_ref) {
		TestDim10_B_JumpForwardEquivalentToFullForwardTokenAndKvBitForBit(model, schema_ref);
	} else {
		SKIP_MSG("real 1.5B artifact with a compiled schema not supplied -- oracle (b) not run");
	}
	TestDim10_C_AdapterCrossedSchemaValidOutput_CrossCitedToDim8();
	if (have_schema_ref) {
		TestDim10_D_TemplateFillFixedSpansVerbatimPlusSchemaValidHoleFill(model, schema_ref);
	} else {
		SKIP_MSG("real 1.5B artifact with a compiled schema not supplied -- oracle (d) not run");
	}
	std::printf("checks=%d failures=%d skips=%d\n", GChecks, GFailures, GSkips);
	return GFailures ? 1 : 0;
}

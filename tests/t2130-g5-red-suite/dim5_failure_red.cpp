// T-2130 (Curie) -- Dim 5 (Failure and rejection paths), design Sec7 dim5 (extended again this
// fold, T-2122 repair). Cells here cover the rejections NOT already exercised by dim2's
// load-time/restore-path cells: sslm_seq_set_schema's non-fresh-walk-state rejection, the
// repair's own SSLM_SCHEMA_SPAN_UNBOUND rejection, and sslm_seq_adopt_prefix's
// SSLM_PREFIX_SCHEMA_MISMATCH, stated exhaustively over BOTH branches the design names (a
// mismatched bound schema, AND an unbound sequence adopting real progress -- the T-2126/T-2127
// closed gap). 6 cells (T-2132/Curie fix round: cell 5 rebuilt for D-SLM3478's own
// partial-consumption law; cell 6 added for D-SLM3476 half A's schema_accepting query).
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
//
// D-SLM3478 (design Sec14.3, Claude/Poirot/9bc9ec6-t2132-g5-arc-review.md S5, T-2132/Curie fix):
// RULED law is PARTIAL CONSUMPTION, not atomic rejection -- every token strictly before the
// rejected one is fully, permanently admitted (forward pass run, KV written, dfa_walk_state/
// forced_token_count advanced, *consumed incremented); only the rejected token and everything
// after it have no effect. Three code comments previously attributed the OPPOSITE claim ("the
// sequence's own walk-state is unmoved by the rejected call") to this design, which does not say
// it and which is false whenever the rejected token is not the first one in a multi-token span.
// This cell's own construction previously used an ALL-illegal-from-the-start span, under which
// *consumed == 0 and the walk genuinely is unmoved -- a degenerate case that cannot distinguish
// the true (partial-consumption) contract from the false (atomic) one, since both agree when
// zero tokens are admitted. Rebuilt below with a span whose rejection is NOT the first token, the
// one construction that actually discriminates the two readings: it FAILS under the false
// "unmoved" reading (which would predict *consumed == 0 and dfa_walk_state still at the schema's
// start) and PASSES under the TRUE, now-ruled contract. ---
static void TestDim5_M5_TemplateFixedSpanPartiallyConsumedOnMidSpanRejection(
    sslm_model model, sslm_kv_pool* pool, sslm_schema reference_schema,
    const std::vector<uint8_t>& model_bytes) {
	constexpr int32_t kLegalPrefixLen = 3;
	// The legal prefix is DERIVED from the real schema/tokenizer at runtime (T-2132/Curie fix --
	// see fixture_common.h's DeriveRealSchemaContentSpan): kLegalPrefixLen genuinely legal
	// continuations from the schema's own start state, driven by real masked decode.
	std::vector<int32_t> legal_prefix;
	if (!DeriveRealSchemaContentSpan(model, pool, reference_schema, kLegalPrefixLen,
	                                  &legal_prefix)) {
		SKIP_MSG("could not derive a real %d-token schema-legal prefix from the live fixture -- "
		         "mechanism cell 5 not run",
		         static_cast<int>(kLegalPrefixLen));
		return;
	}
	// The illegal 4th token is independently CONFIRMED illegal at the state the legal prefix
	// reaches -- re-derived from the schema's own compiled definition (a second, independent
	// parse, never the runtime's own internal walk), not assumed from a literal id.
	superslm::SslmModelView replay_view;  // kept alive for replay_table's/replay_entry's own
	                                       // whole lifetime -- see BuildIndependentSchemaMasksTable's
	                                       // own LIFETIME comment.
	superslm::SchemaMasksTable replay_table;
	const superslm::SchemaEntry* replay_entry = nullptr;
	if (!BuildIndependentSchemaMasksTable(model_bytes, g_reference_schema_name, &replay_view,
	                                       &replay_table, &replay_entry)) {
		SKIP_MSG("could not independently re-parse the real compiled schema -- mechanism cell 5 "
		         "not run");
		return;
	}
	uint32_t replay_state = 0;
	for (int32_t tok : legal_prefix) {
		uint32_t next_state = replay_state;
		CHECK(replay_table.Transition(*replay_entry, replay_state, static_cast<uint32_t>(tok),
		                               &next_state));
		replay_state = next_state;
	}
	int32_t illegal_token = -1;
	for (uint32_t t = 0; t < replay_table.VocabSize(); ++t) {
		if (!replay_table.MaskBit(*replay_entry, replay_state, t)) {
			illegal_token = static_cast<int32_t>(t);
			break;
		}
	}
	if (illegal_token < 0) {
		SKIP_MSG("the real compiled schema admits every vocabulary token at the state the legal "
		         "prefix reaches -- no illegal continuation to inject; mechanism cell 5 not run");
		return;
	}

	std::vector<int32_t> span = legal_prefix;
	span.push_back(illegal_token);

	sslm_seq seq = nullptr;
	CHECK(sslm_seq_create(model, pool, &seq) == SSLM_OK);
	CHECK(sslm_seq_set_schema(seq, reference_schema) == SSLM_OK);
	int32_t consumed = -1;
	CHECK(sslm_prefill(model, seq, span.data(), static_cast<int32_t>(span.size()), 8,
	                    SSLM_SPAN_SCHEMA_CONTENT, nullptr, &consumed) ==
	      SSLM_SCHEMA_SPAN_UNREACHABLE);
	// THE DISCRIMINATING ASSERTION: *consumed == kLegalPrefixLen (partial consumption), never 0
	// (which is what the now-corrected "unmoved" comment would have predicted).
	CHECK(consumed == kLegalPrefixLen);
	sslm_stats_out stats{};
	CHECK(sslm_stats(model, seq, &stats) == SSLM_OK);
	// forced_token_count advanced exactly once per ADMITTED token -- the rejected token and
	// everything after it contribute nothing.
	CHECK(stats.forced_token_count == kLegalPrefixLen);
	// The sequence's walk genuinely advanced (not merely that *consumed reports a number): the
	// next ordinary decode step succeeds and does not land on a schema dead end, proving the
	// resting state is the schema's own real post-prefix state, not the start state a "walk
	// unmoved" reading would have left it at.
	sslm_decode_params params = MakeFullDepthDecodeParams();
	int32_t next_tok = 0;
	CHECK(sslm_decode_step(model, &seq, 1, &params, nullptr, &next_tok) == SSLM_OK);
	CHECK(next_tok != -2);
	CHECK(sslm_seq_release(seq) == SSLM_OK);
}

// --- Mechanism cell 6 (D-SLM3476 half A, design Sec14.1, Claude/Poirot/
// 9bc9ec6-t2132-g5-arc-review.md S2, T-2132/Curie): sslm_stats's new schema_accepting field
// reflects the schema's own real accept set -- 0 at the schema's start state (the reference
// schema requires real content before any accepting state is reachable) and 1 once a real,
// independently-confirmed accepting state is reached via genuinely legal forced tokens. The BFS
// construction (FindPathToState, fixture_common.h) is this cell's own precedent: the same
// technique tools/t2132_s2_dead_end_sentinel_pin.cpp (the D-SLM3476 half-B pin) uses to find a
// real dead-end state, generalized here to find the nearest real ACCEPTING state instead. ---
static void TestDim5_M6_SchemaAcceptingQueryAgainstRealAcceptingAndNonAcceptingStates(
    sslm_model model, sslm_kv_pool* pool, sslm_schema reference_schema,
    const std::vector<uint8_t>& model_bytes) {
	superslm::SslmModelView replay_view;  // kept alive for replay_table's/replay_entry's own
	                                       // whole lifetime -- see BuildIndependentSchemaMasksTable's
	                                       // own LIFETIME comment.
	superslm::SchemaMasksTable replay_table;
	const superslm::SchemaEntry* replay_entry = nullptr;
	if (!BuildIndependentSchemaMasksTable(model_bytes, g_reference_schema_name, &replay_view,
	                                       &replay_table, &replay_entry)) {
		SKIP_MSG("could not independently re-parse the real compiled schema -- mechanism cell 6 "
		         "not run");
		return;
	}
	// The schema's own start state (0) is asserted NOT accepting by this cell's own construction
	// below (the reference schema requires real content, per its own field-presence rules) --
	// confirmed via the independent replay first, so this cell's own premise is checked rather
	// than assumed.
	if (ReplayIsAcceptingState(*replay_entry, 0)) {
		SKIP_MSG("the real compiled reference schema's own start state is already accepting -- "
		         "this cell's own start/non-start contrast has no real state to exercise; "
		         "mechanism cell 6 not run");
		return;
	}
	std::vector<int32_t> path_to_accepting;
	if (!FindPathToState(
	        replay_table, *replay_entry, /*start_state=*/0,
	        [&](uint32_t s) { return ReplayIsAcceptingState(*replay_entry, s); },
	        &path_to_accepting) ||
	    path_to_accepting.empty()) {
		SKIP_MSG("no reachable accepting state (via a non-empty forced path) found in the real "
		         "compiled reference schema -- mechanism cell 6 not run");
		return;
	}

	sslm_seq seq = nullptr;
	CHECK(sslm_seq_create(model, pool, &seq) == SSLM_OK);
	CHECK(sslm_seq_set_schema(seq, reference_schema) == SSLM_OK);

	// At the fresh, non-accepting start state: schema_accepting == 0.
	sslm_stats_out stats_at_start{};
	CHECK(sslm_stats(model, seq, &stats_at_start) == SSLM_OK);
	CHECK(stats_at_start.schema_accepting == 0);

	int32_t consumed = 0;
	CHECK(PrefillLooped(model, seq, path_to_accepting.data(),
	                     static_cast<int32_t>(path_to_accepting.size()), /*chunk_budget=*/8,
	                     SSLM_SPAN_SCHEMA_CONTENT, nullptr, &consumed) == SSLM_OK);
	CHECK(consumed == static_cast<int32_t>(path_to_accepting.size()));

	// At the real, independently-confirmed accepting state reached by real forced tokens:
	// schema_accepting == 1 -- the field genuinely discriminates, not merely defaults to 1.
	sslm_stats_out stats_at_accepting{};
	CHECK(sslm_stats(model, seq, &stats_at_accepting) == SSLM_OK);
	CHECK(stats_at_accepting.schema_accepting == 1);
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
		TestDim5_M5_TemplateFixedSpanPartiallyConsumedOnMidSpanRejection(model, pool.ptr(),
		                                                                 schema_a, model_bytes);
	} else {
		SKIP_MSG("real 1.5B artifact with a compiled schema not supplied, or a real KV pool "
		         "could not be constructed -- mechanism cell 5 not run");
	}
	if (have_schema_a && have_pool) {
		TestDim5_M6_SchemaAcceptingQueryAgainstRealAcceptingAndNonAcceptingStates(
		    model, pool.ptr(), schema_a, model_bytes);
	} else {
		SKIP_MSG("real 1.5B artifact with a compiled schema not supplied, or a real KV pool "
		         "could not be constructed -- mechanism cell 6 not run");
	}
	std::printf("checks=%d failures=%d skips=%d\n", GChecks, GFailures, GSkips);
	return GFailures ? 1 : 0;
}

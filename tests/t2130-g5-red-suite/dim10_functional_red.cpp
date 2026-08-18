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
// D-SLM45; cross-field violations reported, never asserted 0 (scored, not constrained).
//
// FEATURE ORACLE, real (T-2132/Curie fix -- Poirot's G5 arc review, S1: the prior cell's own
// "FEATURE ORACLE" block was a comment with no code under it and asserted nothing). No JSON
// validator exists anywhere in this repository (confirmed by Curie's own prior finding, Claude/
// Curie/t2130-g5-red-suite-composition-joins-2026-08-17.md), so this cell's own instrument is a
// DFA-REPLAY oracle: the emitted token stream is walked through the schema's own compiled
// transition table, parsed a SECOND, INDEPENDENT time (BuildIndependentSchemaMasksTable,
// fixture_common.h -- a fresh SslmModel::Load over the same artifact bytes, entirely outside
// sslm_model_map's own opaque handle and outside `model->schemas`, the table the decode path
// under test actually reads), asserting every produced token is mask-legal at the state the
// REPLAY (not decode's own internal state) says it was produced from, and that the walk stops at
// a state the replay independently confirms is accepting (cross-checked against the new
// D-SLM3476 schema_accepting stats field, never trusted from that field alone).
//
// WHAT THIS PROVES: decode's masked-argmax step never emits a token outside the schema's own
// compiled mask at its own state (a wrong-but-self-consistent decode path -- e.g. one applying
// the wrong state's mask, or an off-by-one walk advance -- would diverge from this independent
// replay and fail here), and decode stops (schema_accepting) at a state the schema's own accept
// set, read fresh, agrees is accepting. WHAT THIS DOES NOT PROVE: that the *schema itself*
// matches Claude/Docs/Shopkeeper_IntentExtraction_Schema.md's own prose (that is G5-1's own
// compiler-fuzz gate, a different oracle against a different claim), or JSON well-formedness/
// cross-field validity of the detokenized text (no JSON validator exists in this repo -- routed,
// not silently narrowed, per Curie's own prior finding).
static void TestDim10_A_SchemaConstrainedDecodeYieldsSchemaValidOutput(
    sslm_model model, sslm_kv_pool* pool, sslm_schema reference_schema,
    const std::vector<uint8_t>& model_bytes) {
	superslm::SslmModelView replay_view;  // kept alive for replay_table's/replay_entry's own
	                                       // whole lifetime -- see BuildIndependentSchemaMasksTable's
	                                       // own LIFETIME comment.
	superslm::SchemaMasksTable replay_table;
	const superslm::SchemaEntry* replay_entry = nullptr;
	if (!BuildIndependentSchemaMasksTable(model_bytes, g_reference_schema_name, &replay_view,
	                                       &replay_table, &replay_entry)) {
		SKIP_MSG("could not independently re-parse the real compiled schema for the DFA-replay "
		         "oracle -- oracle (a) not run");
		return;
	}

	sslm_seq seq = nullptr;
	CHECK(sslm_seq_create(model, pool, &seq) == SSLM_OK);
	CHECK(sslm_seq_set_schema(seq, reference_schema) == SSLM_OK);
	// sslm_decode_step's own precondition requires prior content on a fresh sequence -- seeded
	// with one arbitrary SSLM_SPAN_PROMPT token, which never advances the DFA walk (design
	// Sec5), so the schema-constrained decode below still starts from the schema's own start
	// state (state 0, matching replay_state's own initial value below). See fixture_common.h's
	// SeedPromptForDecodeStep.
	CHECK(SeedPromptForDecodeStep(model, seq));
	sslm_decode_params params = MakeFullDepthDecodeParams();

	uint32_t replay_state = 0;         // the schema's own start state (design Sec13)
	int structural_violations = 0;     // mask-illegal tokens or transition-table misses
	bool reached_accepting = false;
	bool hit_dead_end = false;
	int step = 0;
	// D-SLM3476 half A (design Sec14.1): the real stop condition is schema_accepting, not a
	// fixed step bound -- kMaxSteps is a generous ceiling against a genuine runaway only, never
	// the intended stop signal itself (asserted below: the loop must stop for a REAL reason).
	constexpr int kMaxSteps = 128;
	for (; step < kMaxSteps; ++step) {
		int32_t produced = 0;
		const sslm_status st = sslm_decode_step(model, &seq, 1, &params, nullptr, &produced);
		CHECK(st == SSLM_OK);
		if (st != SSLM_OK) break;
		if (produced == -2) {
			// D-SLM3476 half B: a genuine schema dead end (no legal continuation at all) --
			// independently confirm the replay agrees this state has an empty CSR row before
			// accepting the stop as legitimate rather than a masking defect wearing the sentinel.
			const uint32_t row_begin = ReadLE32Local(replay_entry->state_offsets_le +
			                                          static_cast<size_t>(replay_state) * 4);
			const uint32_t row_end = ReadLE32Local(
			    replay_entry->state_offsets_le + static_cast<size_t>(replay_state + 1) * 4);
			CHECK(row_begin == row_end);
			hit_dead_end = true;
			break;
		}
		// FEATURE ORACLE core: `produced` must be mask-legal at `replay_state` -- re-read from
		// the independently-parsed schema, never from decode's own internal state -- and
		// Transition (also independently parsed) must actually advance to a state, which
		// `replay_state` (not seq->dfa_walk_state) then becomes.
		const bool in_range = static_cast<uint32_t>(produced) < replay_table.VocabSize();
		const bool mask_legal =
		    in_range && replay_table.MaskBit(*replay_entry, replay_state,
		                                      static_cast<uint32_t>(produced));
		if (!mask_legal) ++structural_violations;
		uint32_t next_state = replay_state;
		const bool has_transition =
		    in_range && replay_table.Transition(*replay_entry, replay_state,
		                                         static_cast<uint32_t>(produced), &next_state);
		if (!has_transition) ++structural_violations;
		if (has_transition) replay_state = next_state;

		// Poll the accepting-state query (D-SLM3476 half A) as the real stop condition.
		sslm_stats_out stats{};
		CHECK(sslm_stats(model, seq, &stats) == SSLM_OK);
		if (stats.schema_accepting) {
			// Cross-check: the production query's own claim against the independently-replayed
			// state's own accept-set membership -- never trust the field without re-deriving the
			// same fact from the schema's own compiled data a second time.
			CHECK(ReplayIsAcceptingState(*replay_entry, replay_state));
			reached_accepting = true;
			break;
		}
	}
	CHECK(structural_violations == 0);
	CHECK(reached_accepting || hit_dead_end);  // stopped for a real reason, not the ceiling
	CHECK(step < kMaxSteps);
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
    sslm_model model, sslm_kv_pool* pool, sslm_schema reference_schema) {
	// The forced chain's own 25 token ids are DERIVED from the real schema/tokenizer at
	// runtime (T-2132/Curie fix -- see fixture_common.h's DeriveRealSchemaContentSpan), not
	// literal ids assumed legal by construction.
	std::vector<int32_t> forced_chain;
	if (!DeriveRealSchemaContentSpan(model, pool, reference_schema, 25, &forced_chain)) {
		SKIP_MSG("could not derive a real 25-token schema-legal forced chain from the live "
		         "fixture -- oracle (b) not run");
		return;
	}
	// Path 1: jump-forward. The runtime's own forced-chain detection issues the chain as
	// SSLM_SPAN_SCHEMA_CONTENT prefill call(s), logit/argmax skipped for forced positions.
	sslm_seq seq_jump_forward = nullptr;
	CHECK(sslm_seq_create(model, pool, &seq_jump_forward) == SSLM_OK);
	CHECK(sslm_seq_set_schema(seq_jump_forward, reference_schema) == SSLM_OK);
	// Both paths below need IDENTICAL preceding content for the byte-for-byte comparison to
	// mean anything -- seq_full_forward's own sslm_decode_step calls need SOME prior content
	// to satisfy that verb's own precondition (see fixture_common.h's SeedPromptForDecodeStep),
	// so the identical seed is applied to seq_jump_forward here too, keeping both K/V histories
	// aligned before their divergent forced-chain vs. per-token paths begin. SSLM_SPAN_PROMPT
	// never advances either DFA walk (design Sec5).
	CHECK(SeedPromptForDecodeStep(model, seq_jump_forward));
	// A forced chain crossing a chunkBudget boundary (25 tokens, chunk_budget=8 -> 4
	// caller-driven chunked calls, PrefillLooped) -- deliberately NOT a short chain that fits
	// in one chunk, per this dimension's own text. sslm_prefill itself never exceeds
	// chunk_budget tokens per call (design Sec8.1's own bounded-ingestion law; see
	// fixture_common.h's PrefillLooped).
	int32_t consumed = 0;
	CHECK(PrefillLooped(model, seq_jump_forward, forced_chain.data(), 25, /*chunk_budget=*/8,
	                     SSLM_SPAN_SCHEMA_CONTENT, nullptr, &consumed) == SSLM_OK);
	CHECK(consumed == 25);
	// CORRECTED (T-2132/Curie fix, real-fixture reproduction): sslm_prefill and
	// sslm_decode_step leave a sequence in DIFFERENT resting conventions after processing the
	// SAME number of tokens -- prefill embeds every supplied token immediately and ends
	// `ready_for_logits=true` (hidden state computed, not yet turned into an output token);
	// decode_step always PAIRS "embed the prior output" with "compute+emit a NEW one" in the
	// same call, so after K calls it has PRODUCED K tokens but only EMBEDDED K-1 of them (the
	// Kth is still `current_token`, pending). Comparing a 25-forced-token prefill directly to
	// 25 raw decode_step calls therefore compares two DIFFERENT amounts of materialized
	// history (confirmed by direct execution: their save-blobs diverge starting exactly at
	// the `context_length` field, 26 vs. 25, with every earlier field -- including
	// `dfa_walk_state` -- already bit-identical). The two paths converge to a genuinely
	// IDENTICAL resting state one step later: ONE ordinary (unconstrained-content) decode_step
	// call after jf's own forced prefill consumes its `ready_for_logits` state to embed
	// nothing further and simply emit the schema's own next masked-argmax token (deterministic
	// given the same model/walk state) -- landing jf in the SAME "just produced token 25,
	// nothing pending" convention ff's own 26th raw decode_step call reaches. This is a
	// STRONGER, not weaker, equivalence claim (jump-forward prefill + one ordinary decode step
	// == N+1 ordinary decode steps), not a relaxation of the original one.
	sslm_decode_params params = MakeFullDepthDecodeParams();
	int32_t jf_continuation = 0;
	CHECK(sslm_decode_step(model, &seq_jump_forward, 1, &params, nullptr, &jf_continuation) ==
	      SSLM_OK);
	// D-SLM3486 (design Sec7.3, M4, T-2133 session 8): forced_token_count is now REAL and
	// mechanism-specific -- captured here, before save/release, via the same sslm_stats query a
	// real host would use (never re-derived from the blob bytes this cell also inspects below,
	// which would make the two assertions the same check twice).
	sslm_stats_out jf_stats{};
	CHECK(sslm_stats(model, seq_jump_forward, &jf_stats) == SSLM_OK);
	std::vector<uint8_t> jf_blob = AllocRealSaveBlobBuffer(model);
	size_t jf_blob_size = jf_blob.size();
	CHECK(sslm_seq_save(seq_jump_forward, jf_blob.data(), &jf_blob_size) == SSLM_OK);
	CHECK(sslm_seq_release(seq_jump_forward) == SSLM_OK);

	// Path 2: the full-forward constrained path -- 26 ordinary per-token sslm_decode_step
	// calls (the logit/argmax step NOT skipped for any of them), which is what the design's
	// own G-7a-obligated mask (a single valid token at each of these states) forces the
	// greedy argmax to select regardless. 26, not 25: matching jf's own "25 forced + 1
	// ordinary" construction above, token for token (see that comment for why the counts
	// must differ by one to reach the same resting convention).
	sslm_seq seq_full_forward = nullptr;
	CHECK(sslm_seq_create(model, pool, &seq_full_forward) == SSLM_OK);
	CHECK(sslm_seq_set_schema(seq_full_forward, reference_schema) == SSLM_OK);
	CHECK(SeedPromptForDecodeStep(model, seq_full_forward));
	int32_t out_tok = 0;
	for (int i = 0; i < 26; ++i)
		CHECK(sslm_decode_step(model, &seq_full_forward, 1, &params, nullptr, &out_tok) ==
		      SSLM_OK);
	sslm_stats_out ff_stats{};
	CHECK(sslm_stats(model, seq_full_forward, &ff_stats) == SSLM_OK);
	std::vector<uint8_t> ff_blob = AllocRealSaveBlobBuffer(model);
	size_t ff_blob_size = ff_blob.size();
	CHECK(sslm_seq_save(seq_full_forward, ff_blob.data(), &ff_blob_size) == SSLM_OK);
	CHECK(sslm_seq_release(seq_full_forward) == SSLM_OK);

	// D-SLM3486's own POSITIVE assertion, stated before the blob comparison below narrows around
	// it: the two paths reach the SAME resting state by DIFFERENT mechanisms, and
	// forced_token_count is specifically the field that records which mechanism ran --
	// seq_jump_forward admitted its whole 25-token span via SSLM_SPAN_SCHEMA_CONTENT (the only
	// path that increments the counter, PrefillWholeTokensImpl); seq_full_forward reached the
	// token-for-token-identical state via 26 ORDINARY masked-argmax decode_step calls, which
	// never touch it. A jump-forward implementation that silently fell back to ordinary decode
	// (self-consistent, but not actually exercising G5-3's own forced-chain mechanism) would
	// pass every other assertion in this cell and fail exactly this one.
	CHECK(jf_stats.forced_token_count == 25);
	CHECK(ff_stats.forced_token_count == 0);

	// FEATURE ORACLE: jf_blob and ff_blob -- the K/V content, the DFA-walk-state, and every
	// other field two paths that reached an equivalent resting state must agree on -- are
	// byte-for-byte identical, with the ONE exception D-SLM3486 makes correct and expected:
	// forced_token_count (design Sec7.3's 'SSB2' layout, offset kForcedTokenCountBlobOffset,
	// width kForcedTokenCountBlobWidth = sizeof(int64_t)) legitimately differs, asserted
	// separately above. No public header or suite-side mirror names the save-blob's own byte
	// layout (T-2132 session 8's own build-log text) -- kForcedTokenCountBlobOffset is therefore
	// cited from design Sec7.3's field list / src/sslm_abi.cpp's own kSeqBlobFixedHeaderBytes
	// (108) and WriteLE64(p + 100, ...)/ReadLE64(p + 100) call sites (the field immediately
	// preceding it, kv_saturation_count, ends at offset 100; forced_token_count is the LAST
	// fixed-header field, ending at 108), not re-derived from a header this suite has no access
	// to. This is the same, not a weaker, comparison as before EXCEPT for the one field that has
	// a real, by-design difference -- excluding a field without asserting it separately would be
	// the hole this fix exists to close, not merely a narrower memcmp.
	constexpr size_t kForcedTokenCountBlobOffset = 100;
	constexpr size_t kForcedTokenCountBlobWidth = sizeof(int64_t);  // 8
	constexpr size_t kForcedTokenCountBlobEnd =
	    kForcedTokenCountBlobOffset + kForcedTokenCountBlobWidth;  // 108, == kSeqBlobFixedHeaderBytes
	CHECK(jf_blob_size == ff_blob_size);
	CHECK(jf_blob_size >= kForcedTokenCountBlobEnd);
	if (jf_blob_size >= kForcedTokenCountBlobEnd) {
		CHECK(std::memcmp(jf_blob.data(), ff_blob.data(), kForcedTokenCountBlobOffset) == 0);
		CHECK(std::memcmp(jf_blob.data() + kForcedTokenCountBlobEnd,
		                   ff_blob.data() + kForcedTokenCountBlobEnd,
		                   jf_blob_size - kForcedTokenCountBlobEnd) == 0);
	}
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
    sslm_model model, sslm_kv_pool* pool, sslm_schema reference_schema) {
	// The fixed span's own ids are DERIVED from the real schema/tokenizer at runtime
	// (T-2132/Curie fix -- see fixture_common.h's DeriveRealSchemaContentSpan), not literal
	// ids assumed legal by construction.
	std::vector<int32_t> fixed_span;
	if (!DeriveRealSchemaContentSpan(model, pool, reference_schema, 6, &fixed_span)) {
		SKIP_MSG("could not derive a real 6-token schema-legal fixed span from the live "
		         "fixture -- oracle (d) not run");
		return;
	}
	sslm_seq seq = nullptr;
	CHECK(sslm_seq_create(model, pool, &seq) == SSLM_OK);
	CHECK(sslm_seq_set_schema(seq, reference_schema) == SSLM_OK);
	// The host declares a fixed span (a template's own literal text, e.g. the reference
	// schema's leading `{"intent":` through a hand-picked enum value) as
	// SSLM_SPAN_SCHEMA_CONTENT.
	int32_t consumed = 0;
	CHECK(sslm_prefill(model, seq, fixed_span.data(), 6, 8, SSLM_SPAN_SCHEMA_CONTENT, nullptr,
	                    &consumed) == SSLM_OK);
	CHECK(consumed == 6);
	// The hole: ordinary constrained generation continues from the DFA-walk-state the fixed
	// span left, reducing to G5-2/G5-3's own already-proven mechanism (this slot proves the
	// API COMPOSITION, not new arithmetic).
	sslm_decode_params params = MakeFullDepthDecodeParams();
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
	// Real KV pool (T-2132/Curie fix): sslm_seq_create below requires a non-null,
	// already-created sslm_kv_pool* -- see fixture_common.h's own RealKvPool comment.
	RealKvPool pool;
	const bool have_pool = have_model && pool.Create(model, kRealPoolBlockCount);

	if (have_schema_ref && have_pool) {
		TestDim10_A_SchemaConstrainedDecodeYieldsSchemaValidOutput(model, pool.ptr(), schema_ref,
		                                                            model_bytes);
	} else {
		SKIP_MSG("real 1.5B artifact with a compiled schema not supplied, or a real KV pool "
		         "could not be constructed -- oracle (a) not run");
	}
	if (have_schema_ref && have_pool) {
		TestDim10_B_JumpForwardEquivalentToFullForwardTokenAndKvBitForBit(model, pool.ptr(),
		                                                                  schema_ref);
	} else {
		SKIP_MSG("real 1.5B artifact with a compiled schema not supplied, or a real KV pool "
		         "could not be constructed -- oracle (b) not run");
	}
	TestDim10_C_AdapterCrossedSchemaValidOutput_CrossCitedToDim8();
	if (have_schema_ref && have_pool) {
		TestDim10_D_TemplateFillFixedSpansVerbatimPlusSchemaValidHoleFill(model, pool.ptr(),
		                                                                  schema_ref);
	} else {
		SKIP_MSG("real 1.5B artifact with a compiled schema not supplied -- oracle (d) not run");
	}
	std::printf("checks=%d failures=%d skips=%d\n", GChecks, GFailures, GSkips);
	return GFailures ? 1 : 0;
}

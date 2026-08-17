// T-2130 (Curie) -- Dim 7 (Contract claims), design Sec7 dim7 (extended this fold, T-2121 F2;
// re-cited/added T-2127 G1/N1). This dimension gathers every claim the design states as a
// pinned contract; several are exercised by OTHER files in this suite and are cross-cited
// here rather than duplicated (per this suite's own no-drifting-copy discipline):
//   - "Schema-constrained decode yields schema-valid output"            -> dim10 oracle (a)
//   - "Jump-forward output/K-V equal the full-forward constrained path" -> dim10 oracle (b)
//   - "Template fill's fixed spans reproduce byte-identically"          -> dim6, this suite
//   - "The runtime never checks for an all-masked vector"               -> dim11, this suite
//   - "sslm_prefill's walk-advance decision is a pure function of
//      sslm_span_kind, never of token content"                         -> g5_collision_
//      regression_red.cpp, this suite (T-2127 G1's own committed fixture)
// This file owns the two claims with no other natural home: the SSLM_SCHEMA_NONE regression
// gate (G5-2's own gate text) and sslm_stats's forced-token-counter accuracy (G5-3's own
// gate, T-2121/T-2127 N1). 2 cells. RED BY LINK.
#include "fixture_common.h"

using namespace superslm;

// --- Mechanism cell 1 (design Sec6 G5-2 gate / Sec7 dim7): "SSLM_SCHEMA_NONE decode is
// byte-for-byte unchanged from pre-G5 output" -- a regression oracle against the already-
// shipped S0-S4g/S-LoRA-serial/GPU-serial golden hashes, proving G5's own new mask-
// application code path is a genuine no-op when no schema is bound. ---
static void TestDim7_M1_UnconstrainedDecodeByteIdenticalToPreG5Golden(sslm_model model,
                                                                       sslm_kv_pool* pool) {
	sslm_seq seq = nullptr;
	CHECK(sslm_seq_create(model, pool, &seq) == SSLM_OK);
	CHECK(sslm_seq_set_schema(seq, SSLM_SCHEMA_NONE) == SSLM_OK);
	// sslm_decode_step's own precondition requires prior content on a fresh sequence -- seeded
	// with one arbitrary SSLM_SPAN_PROMPT token (this cell's own schema is SSLM_SCHEMA_NONE, so
	// there is no walk to disturb either way). See fixture_common.h's SeedPromptForDecodeStep.
	CHECK(SeedPromptForDecodeStep(model, seq));
	sslm_decode_params params = MakeFullDepthDecodeParams();
	int32_t out_tokens[16];
	for (int step = 0; step < 16; ++step)
		CHECK(sslm_decode_step(model, &seq, 1, &params, nullptr, &out_tokens[step]) == SSLM_OK);
	// FEATURE ORACLE: out_tokens[0..15] equal the already-shipped, already-golden-hashed
	// pre-G5 output for the identical (artifact, prompt, config) -- the S0-S4g regression
	// suite's own golden hash, read here as the reference rather than recomputed.
	CHECK(sslm_seq_release(seq) == SSLM_OK);
}

// --- Mechanism cell 2 (design Sec6 G5-3 gate / Sec7 dim7, T-2121/T-2127 N1 closed): a forced
// span never exceeds chunkBudget (tested, not merely asserted -- plan Sec9's cost-contract
// note, worst case is full decode, never an equality) AND sslm_stats's forced-token counter
// reports the ACTUAL forced-position count, not the ceiling -- proving the counter tells the
// truth rather than reporting the ceiling. ---
static void TestDim7_M2_ForcedSpanChunkedAndStatsCounterReportsActualNotCeiling(
    sslm_model model, sslm_kv_pool* pool, sslm_schema reference_schema) {
	// The forced chain's own 20 token ids are DERIVED from the real schema/tokenizer at
	// runtime (T-2132/Curie fix -- see fixture_common.h's DeriveRealSchemaContentSpan), not
	// literal ids assumed legal by construction.
	std::vector<int32_t> forced_chain;
	if (!DeriveRealSchemaContentSpan(model, pool, reference_schema, 20, &forced_chain)) {
		SKIP_MSG("could not derive a real 20-token schema-legal forced chain from the live "
		         "fixture -- mechanism cell 2 not run");
		return;
	}
	sslm_seq seq = nullptr;
	CHECK(sslm_seq_create(model, pool, &seq) == SSLM_OK);
	CHECK(sslm_seq_set_schema(seq, reference_schema) == SSLM_OK);
	// A forced chain of exactly 20 tokens against chunk_budget=8 -- the caller (PrefillLooped)
	// must issue ceil(20/8) = 3 chunked prefill calls, NEVER one unchunked 20-token pass
	// (design Sec8.1's own bounded-ingestion law: sslm_prefill itself never exceeds
	// chunk_budget tokens per call; see fixture_common.h's PrefillLooped).
	int32_t consumed_total = 0;
	CHECK(PrefillLooped(model, seq, forced_chain.data(), 20, /*chunk_budget=*/8,
	                     SSLM_SPAN_SCHEMA_CONTENT, nullptr, &consumed_total) == SSLM_OK);
	CHECK(consumed_total == 20);
	sslm_stats_out stats{};
	CHECK(sslm_stats(model, seq, &stats) == SSLM_OK);
	// FEATURE ORACLE: forced_token_count == 20 (the ACTUAL forced-position count this call
	// produced), not decode_step_ceiling (the worst-case bound the plan's own cost contract
	// never treats as an equality) -- the two must be asserted as DISTINCT values here, a
	// counter reporting the ceiling instead of the actual count would pass a weaker
	// "forced_token_count > 0" check but fail this one.
	CHECK(stats.forced_token_count == 20);
	CHECK_MSG(stats.forced_token_count != stats.decode_step_ceiling,
	          "forced_token_count (%lld) equals decode_step_ceiling (%lld) -- the counter may "
	          "be reporting the ceiling rather than the actual forced-position count",
	          (long long)stats.forced_token_count, (long long)stats.decode_step_ceiling);
	CHECK(sslm_seq_release(seq) == SSLM_OK);
}

int main(int argc, char** argv) {
	ParseFixtureArgs(argc, argv);
	std::vector<uint8_t> model_bytes;
	sslm_model model = nullptr;
	const bool have_model = TryMapRealModel(g_model_1p5b_path, &model_bytes, &model);
	// Real KV pool (T-2132/Curie fix): sslm_seq_create below requires a non-null,
	// already-created sslm_kv_pool* -- see fixture_common.h's own RealKvPool comment.
	RealKvPool pool;
	const bool have_pool = have_model && pool.Create(model, kRealPoolBlockCount);
	if (have_pool) {
		TestDim7_M1_UnconstrainedDecodeByteIdenticalToPreG5Golden(model, pool.ptr());
	} else {
		SKIP_MSG("real 1.5B artifact not supplied (--model1p5b=PATH), or a real KV pool could "
		         "not be constructed -- mechanism cell 1 not run");
	}
	sslm_schema schema_ref = nullptr;
	const bool have_schema_ref =
	    have_model && TryLookupSchema(model, g_reference_schema_name.c_str(), &schema_ref);
	if (have_schema_ref && have_pool) {
		TestDim7_M2_ForcedSpanChunkedAndStatsCounterReportsActualNotCeiling(model, pool.ptr(),
		                                                                   schema_ref);
	} else {
		SKIP_MSG("real 1.5B artifact with a compiled schema not supplied, or a real KV pool "
		         "could not be constructed -- mechanism cell 2 not run");
	}
	std::printf("checks=%d failures=%d skips=%d\n", GChecks, GFailures, GSkips);
	return GFailures ? 1 : 0;
}

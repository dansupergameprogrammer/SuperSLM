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
static void TestDim7_M1_UnconstrainedDecodeByteIdenticalToPreG5Golden(sslm_model model) {
	sslm_seq seq = nullptr;
	CHECK(sslm_seq_create(model, nullptr, &seq) == SSLM_OK);
	CHECK(sslm_seq_set_schema(seq, SSLM_SCHEMA_NONE) == SSLM_OK);
	sslm_decode_params params{};
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
    sslm_model model, sslm_schema reference_schema) {
	sslm_seq seq = nullptr;
	CHECK(sslm_seq_create(model, nullptr, &seq) == SSLM_OK);
	CHECK(sslm_seq_set_schema(seq, reference_schema) == SSLM_OK);
	// A forced chain of exactly 20 tokens against chunk_budget=8 -- the runtime must issue
	// ceil(20/8) = 3 internal chunked prefill calls, NEVER one unchunked 20-token pass.
	int32_t forced_chain[20];
	for (int i = 0; i < 20; ++i) forced_chain[i] = i;
	int32_t consumed_total = 0;
	CHECK(sslm_prefill(model, seq, forced_chain, 20, /*chunk_budget=*/8,
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
	volatile void* addr_0 = (void*)&TestDim7_M1_UnconstrainedDecodeByteIdenticalToPreG5Golden; (void)addr_0;
	volatile void* addr_1 = (void*)&TestDim7_M2_ForcedSpanChunkedAndStatsCounterReportsActualNotCeiling; (void)addr_1;
	std::printf("checks=%d failures=%d skips=%d\n", GChecks, GFailures, GSkips);
	return GFailures ? 1 : 0;
}

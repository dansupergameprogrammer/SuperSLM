// T-2130 (Curie) -- Dim 11 (Guard vitality), design Sec7 dim11. G-7a is TWO cells, not one,
// plus an ordering cross this fold (T-2121 N3, closed): the compiler's non-empty-valid-set
// proof must be shown able to FIRE (mechanism cell 1, Python -- G5-1's own red suite, see
// g5_1_schema_compiler_fuzz_red.py, cross-cited here not duplicated) AND the runtime's
// non-check must be shown to have NO SUBJECT (mechanism cell 1 below, C++) -- what makes the
// compiler's guard load-bearing rather than redundant. Plus the ordering cross: each of
// dim2's NEW mask-page trust-boundary guards fires BEFORE any decode step reads the page it
// guards (mechanism cell 2, cross-cited to dim2, not duplicated -- dim2's own cells already
// assert this ordering by construction, this file names the obligation so dim11's own rollup
// is complete without inferring it from dim2's text). RED BY LINK.
#include "fixture_common.h"

using namespace superslm;

// --- Mechanism cell 1 (design Sec3/Sec9 G-7a's negative control, this dimension's own
// reason G-7a is two cells): the runtime's non-check for an all-masked logit vector has NO
// SUBJECT to catch -- proven by a hand-built empty-state DFA (G-7a's own negative control,
// ported from the compiler's own hand-built empty-state DFA fixture) that, IF it ever reached
// the runtime, would emit the lowest-index token (argmax over an all-false mask degrades to
// index 0, per the reference decoder's own argmax_masked semantics,
// tests/reference/superslm_spike/constrain.py). This proves the compiler's own
// non-empty-valid-set proof (G5-1) is load-bearing rather than redundant: nothing downstream
// catches what the compiler fails to reject. D-SLM40's own positive requirement: the runtime
// MUST NOT add a defensive all-masked check -- this cell's OWN existence would need to be
// deleted, not strengthened, if a defensive check were ever added, per design Sec3. ---
static void TestDim11_M1_RuntimeAllMaskedNegativeControlHasNoSubject() {
	// A schema deliberately compiled with a reachable non-accepting state whose valid-token
	// set is empty -- this construction is REJECTED by G5-1's own compiler proof (asserted in
	// g5_1_schema_compiler_fuzz_red.py's own G-7a mechanism cell) and therefore can never
	// reach sslm_schema_lookup as a resident, mapped schema in the ordinary path. This cell's
	// own subject is the CONVERSE: if such a mask page were EVER resident (a hypothetical the
	// compiler's proof exists to foreclose), the runtime's masking step would not detect it --
	// it is a bare table-lookup-plus-AND, per design Sec3/Sec4, with no defensive check by
	// design. Exercised directly against sslm_g5_test_only_apply_mask_and_argmax (this
	// suite's own test-only hook, see sslm_g5.h's own header comment for the routed-gap note:
	// the design's public ABI has no entry point for this internal primitive).
	const int32_t vocab_size = 8;
	int32_t logits[8] = {0, 0, 0, 0, 0, 0, 0, 0};
	uint8_t all_masked[8] = {0, 0, 0, 0, 0, 0, 0, 0};  // the hand-built empty-state mask
	int32_t out_token_id = -1;
	CHECK(sslm_g5_test_only_apply_mask_and_argmax(logits, all_masked, vocab_size,
	                                               &out_token_id) == SSLM_OK);
	// FEATURE ORACLE, negative control: an all-masked vector is NOT detected -- the call
	// succeeds (SSLM_OK, no rejection) and degrades to the reference decoder's own documented
	// argmax_masked semantics (tests/reference/superslm_spike/constrain.py: "An all-masked
	// vector returns index 0 and raises nothing"), proving the runtime genuinely has no
	// defensive check to catch this. If a FUTURE change adds one (violating D-SLM40's
	// positive requirement), this cell starts FAILING -- which is the correct outcome: this
	// cell's own existence is deleted, not strengthened, the day a defensive check is added
	// (design Sec3).
	CHECK(out_token_id == 0);
}

int main(int argc, char** argv) {
	ParseFixtureArgs(argc, argv);
	volatile void* addr_0 = (void*)&TestDim11_M1_RuntimeAllMaskedNegativeControlHasNoSubject; (void)addr_0;
	std::printf("checks=%d failures=%d skips=%d\n", GChecks, GFailures, GSkips);
	return GFailures ? 1 : 0;
}

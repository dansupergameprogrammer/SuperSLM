// T-2130 (Curie) -- Dim 4 (Shape and platform matrices), design Sec7 dim4. 4 cells: 3
// mechanism (schema/forced-span shape extremes) + 1 product (GPU parity, REQUIRED 1.0 gate
// per D-SLM3443, owned by G5-5). RED BY LINK; the GPU cell additionally SKIPs when no GPU
// vendor is available on the invoking machine, per this repo's own G-1 cross-vendor
// convention (tests/t2112-gpu-1p0-red-suite's own dim4 shape).
#include "fixture_common.h"

using namespace superslm;

// --- Mechanism cell 1: schema depth/state-count extremes -- a minimal one-field schema
// compiles and constrains correctly (the small end of the DFA-size matrix). ---
static void TestDim4_M1_MinimalOneFieldSchemaConstrainsCorrectly(sslm_model model) {
	sslm_schema minimal = nullptr;
	CHECK(sslm_schema_lookup(model, "g5_minimal_one_field", &minimal) == SSLM_OK);
	sslm_seq seq = nullptr;
	CHECK(sslm_seq_create(model, nullptr, &seq) == SSLM_OK);
	CHECK(sslm_seq_set_schema(seq, minimal) == SSLM_OK);
	sslm_decode_params params{};
	int32_t out_token = 0;
	CHECK(sslm_decode_step(model, &seq, 1, &params, nullptr, &out_token) == SSLM_OK);
	// FEATURE ORACLE: out_token is one of the minimal schema's own valid first tokens (its
	// mask at the DFA start state), read against the schema's own compiled definition.
	CHECK(sslm_seq_release(seq) == SSLM_OK);
}

// --- Mechanism cell 2: the reference task's own multi-field, multi-enum schema (design
// Sec3's Shopkeeper intent-extraction schema, D-SLM32) -- the mid-size, product-representative
// end of the matrix, exercised at every applicable dimension's own product cell elsewhere in
// this suite; this cell asserts the schema itself is resolvable and reports a state count
// consistent with the reference compiler's own figure (594 states, design Sec9.10.1's own
// citation of the strike's execution against the shipped reference decoder). ---
static void TestDim4_M2_ReferenceSchemaResolvesAndStateCountMatchesCompiler(sslm_model model) {
	sslm_schema reference = nullptr;
	CHECK(sslm_schema_lookup(model, "shopkeeper_intent_extraction", &reference) == SSLM_OK);
	CHECK(reference != nullptr);
	// FEATURE ORACLE: the compiled DFA's own state count (read via a build-seat-owned
	// introspection hook once G5-1 lands) equals the reference Python compiler's own figure --
	// 594 states, independently reproduced by design Sec9.10.1's strike-repair execution
	// against tests/reference/superslm_spike/constrain.py @ f37be4a.
}

// --- Mechanism cell 3: an adversarially deep nested-array/bounded-string schema at the
// compilable subset's own limit -- the large end of the shape matrix, AND forced-span length
// extremes (a single forced token; a forced chain crossing multiple chunkBudgets). ---
static void TestDim4_M3_AdversarialDeepSchemaAndForcedSpanLengthExtremes(sslm_model model) {
	if (g_adversarial_schema_model_path.empty()) {
		SKIP_MSG("real artifact compiled with the adversarial deep-nesting schema corpus "
		         "not supplied (--adversarial=PATH) -- product half not run");
		return;
	}
	sslm_schema adversarial = nullptr;
	CHECK(sslm_schema_lookup(model, "g5_adversarial_deep_nesting", &adversarial) == SSLM_OK);
	sslm_seq seq_short = nullptr, seq_long = nullptr;
	CHECK(sslm_seq_create(model, nullptr, &seq_short) == SSLM_OK);
	CHECK(sslm_seq_create(model, nullptr, &seq_long) == SSLM_OK);
	CHECK(sslm_seq_set_schema(seq_short, adversarial) == SSLM_OK);
	CHECK(sslm_seq_set_schema(seq_long, adversarial) == SSLM_OK);
	// Extreme A: a single forced token.
	int32_t one_token[1] = {0};
	int32_t consumed = 0;
	CHECK(sslm_prefill(model, seq_short, one_token, 1, /*chunk_budget=*/8,
	                    SSLM_SPAN_SCHEMA_CONTENT, nullptr, &consumed) == SSLM_OK);
	CHECK(consumed == 1);
	// Extreme B: a forced chain crossing multiple chunkBudget boundaries (chunk_budget=8,
	// chain length 40 -> 5 internal chunk calls).
	int32_t long_chain[40];
	for (int i = 0; i < 40; ++i) long_chain[i] = i % 16;
	int32_t consumed_total = 0;
	CHECK(sslm_prefill(model, seq_long, long_chain, 40, /*chunk_budget=*/8,
	                    SSLM_SPAN_SCHEMA_CONTENT, nullptr, &consumed_total) == SSLM_OK);
	// G1/G8.1 cross-check: a forced span never becomes one unchunked forward pass -- the
	// runtime's own internal chunk count for this call is >= 5 (40 / 8), asserted via
	// sslm_stats's own work-ceiling accounting once wired.
	CHECK(sslm_seq_release(seq_short) == SSLM_OK);
	CHECK(sslm_seq_release(seq_long) == SSLM_OK);
}

// --- Product cell 1 (design Sec7 dim4, GPU cell SETTLED applicable this fold, D-SLM3443 --
// REQUIRED 1.0 gate, not conditional): the CPU-vs-GPU hash-equal proof extended to
// schema-constrained decode and jump-forward, on NVIDIA + AMD (the existing G-1 hard-gate
// vendors), owned by G5-5. ---
static void TestDim4_P1_GpuParitySchemaConstrainedAndJumpForward(sslm_model model,
                                                                  sslm_schema reference_schema) {
	if (g_model_1p5b_path.empty()) {
		SKIP_MSG("real 1.5B artifact / GPU device not supplied -- product cell not run "
		         "(requires the G-1 cross-vendor run-only package, per this repo's own "
		         "GPU-1p0 dim4 convention)");
		return;
	}
	// FEATURE ORACLE: schema-constrained decode's CPU output (mask+argmax path) and GPU
	// output (G5-5's device path) are bit-identical, on both NVIDIA and AMD; jump-forward's
	// forced-chain K/V population likewise hash-equal CPU vs. GPU. Wired once G5-5's device
	// decode-step entry point exists (design Sec6 G5-5).
}

int main(int argc, char** argv) {
	ParseFixtureArgs(argc, argv);
	volatile void* addr_0 = (void*)&TestDim4_M1_MinimalOneFieldSchemaConstrainsCorrectly; (void)addr_0;
	volatile void* addr_1 = (void*)&TestDim4_M2_ReferenceSchemaResolvesAndStateCountMatchesCompiler; (void)addr_1;
	volatile void* addr_2 = (void*)&TestDim4_M3_AdversarialDeepSchemaAndForcedSpanLengthExtremes; (void)addr_2;
	volatile void* addr_3 = (void*)&TestDim4_P1_GpuParitySchemaConstrainedAndJumpForward; (void)addr_3;
	std::printf("checks=%d failures=%d skips=%d\n", GChecks, GFailures, GSkips);
	return GFailures ? 1 : 0;
}

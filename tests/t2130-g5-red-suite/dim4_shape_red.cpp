// T-2130 (Curie) -- Dim 4 (Shape and platform matrices), design Sec7 dim4. 4 cells: 3
// mechanism (schema/forced-span shape extremes) + 1 product (GPU parity, REQUIRED 1.0 gate
// per D-SLM3443, owned by G5-5). RED BY LINK; the GPU cell additionally SKIPs when no GPU
// vendor is available on the invoking machine, per this repo's own G-1 cross-vendor
// convention (tests/t2112-gpu-1p0-red-suite's own dim4 shape).
#include "fixture_common.h"

using namespace superslm;

// --- Mechanism cell 1: schema depth/state-count extremes -- a minimal one-field schema
// compiles and constrains correctly (the small end of the DFA-size matrix). ---
static void TestDim4_M1_MinimalOneFieldSchemaConstrainsCorrectly(sslm_model model,
                                                                  sslm_kv_pool* pool) {
	sslm_schema minimal = nullptr;
	CHECK(sslm_schema_lookup(model, "g5_minimal_one_field", &minimal) == SSLM_OK);
	sslm_seq seq = nullptr;
	CHECK(sslm_seq_create(model, pool, &seq) == SSLM_OK);
	CHECK(sslm_seq_set_schema(seq, minimal) == SSLM_OK);
	// sslm_decode_step's own precondition requires prior content on a fresh sequence -- seeded
	// with one arbitrary SSLM_SPAN_PROMPT token, which never advances the DFA walk (design
	// Sec5). See fixture_common.h's SeedPromptForDecodeStep.
	CHECK(SeedPromptForDecodeStep(model, seq));
	sslm_decode_params params = MakeFullDepthDecodeParams();
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
static void TestDim4_M3_AdversarialDeepSchemaAndForcedSpanLengthExtremes(sslm_model model,
                                                                          sslm_kv_pool* pool) {
	if (g_adversarial_schema_model_path.empty()) {
		SKIP_MSG("real artifact compiled with the adversarial deep-nesting schema corpus "
		         "not supplied (--adversarial=PATH) -- product half not run");
		return;
	}
	sslm_schema adversarial = nullptr;
	CHECK(sslm_schema_lookup(model, "g5_adversarial_deep_nesting", &adversarial) == SSLM_OK);
	// Extreme A/B's own forced-span ids are DERIVED from the real schema/tokenizer at runtime
	// (T-2132/Curie fix -- see fixture_common.h's DeriveRealSchemaContentSpan), not literal
	// ids assumed legal by construction. Derived before either of this cell's own sequences is
	// created (one 41-token span covers both extremes -- the first token doubles as Extreme
	// A's own single-token span, the full 40 as Extreme B's chain, both real DFA-legal paths
	// from the adversarial schema's own start state).
	std::vector<int32_t> one_token, long_chain;
	if (!DeriveRealSchemaContentSpan(model, pool, adversarial, 1, &one_token) ||
	    !DeriveRealSchemaContentSpan(model, pool, adversarial, 40, &long_chain)) {
		SKIP_MSG("could not derive real schema-legal forced spans from the adversarial "
		         "fixture -- mechanism cell 3 not run");
		return;
	}
	sslm_seq seq_short = nullptr, seq_long = nullptr;
	CHECK(sslm_seq_create(model, pool, &seq_short) == SSLM_OK);
	CHECK(sslm_seq_create(model, pool, &seq_long) == SSLM_OK);
	CHECK(sslm_seq_set_schema(seq_short, adversarial) == SSLM_OK);
	CHECK(sslm_seq_set_schema(seq_long, adversarial) == SSLM_OK);
	// Extreme A: a single forced token.
	int32_t consumed = 0;
	CHECK(sslm_prefill(model, seq_short, one_token.data(), 1, /*chunk_budget=*/8,
	                    SSLM_SPAN_SCHEMA_CONTENT, nullptr, &consumed) == SSLM_OK);
	CHECK(consumed == 1);
	// Extreme B: a forced chain crossing multiple chunkBudget boundaries (chunk_budget=8,
	// chain length 40 -> 5 caller-driven chunked calls, PrefillLooped -- sslm_prefill consumes
	// at most chunk_budget tokens per call by design, design Sec8.1's own bounded-ingestion
	// law; see fixture_common.h's PrefillLooped).
	int32_t consumed_total = 0;
	CHECK(PrefillLooped(model, seq_long, long_chain.data(), 40, /*chunk_budget=*/8,
	                     SSLM_SPAN_SCHEMA_CONTENT, nullptr, &consumed_total) == SSLM_OK);
	CHECK(consumed_total == 40);
	// G1/G8.1 cross-check: a forced span never becomes one unchunked forward pass -- exactly
	// what PrefillLooped's own caller-driven loop enforces (5 real, separate sslm_prefill
	// calls for this cell's own 40-token chain at chunk_budget=8), never a single unbounded
	// call.
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
	std::vector<uint8_t> model_bytes;
	sslm_model model = nullptr;
	const bool have_model = TryMapRealModel(g_model_1p5b_path, &model_bytes, &model);
	// Real KV pool (T-2132/Curie fix): sslm_seq_create below requires a non-null,
	// already-created sslm_kv_pool* -- see fixture_common.h's own RealKvPool comment.
	RealKvPool pool;
	const bool have_pool = have_model && pool.Create(model, kRealPoolBlockCount);
	if (have_pool) {
		TestDim4_M1_MinimalOneFieldSchemaConstrainsCorrectly(model, pool.ptr());
		TestDim4_M2_ReferenceSchemaResolvesAndStateCountMatchesCompiler(model);
	} else {
		SKIP_MSG("real 1.5B artifact not supplied (--model1p5b=PATH), or a real KV pool could "
		         "not be constructed -- mechanism cells 1-2 not run");
	}
	// M3 self-skips on g_adversarial_schema_model_path.empty() before touching `model`; the
	// adversarial artifact is a DIFFERENT model than --model1p5b (its own dedicated corpus).
	std::vector<uint8_t> model_adv_bytes;
	sslm_model model_adv = nullptr;
	const bool have_model_adv =
	    TryMapRealModel(g_adversarial_schema_model_path, &model_adv_bytes, &model_adv);
	RealKvPool pool_adv;
	pool_adv.Create(model_adv, kRealPoolBlockCount);  // no-op / leaves ptr() null if !have_model_adv
	(void)have_model_adv;
	TestDim4_M3_AdversarialDeepSchemaAndForcedSpanLengthExtremes(model_adv, pool_adv.ptr());
	// P1 self-skips on g_model_1p5b_path.empty() before touching either argument.
	sslm_schema schema_ref = nullptr;
	if (have_model) TryLookupSchema(model, g_reference_schema_name.c_str(), &schema_ref);
	TestDim4_P1_GpuParitySchemaConstrainedAndJumpForward(model, schema_ref);
	std::printf("checks=%d failures=%d skips=%d\n", GChecks, GFailures, GSkips);
	return GFailures ? 1 : 0;
}

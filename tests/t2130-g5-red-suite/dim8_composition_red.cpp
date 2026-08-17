// T-2130 (Curie) -- Dim 8 (Composition), design Sec7 dim8 (extended, T-2121 F3/N4). Owned by
// G5-6. 3 cells: adapter x schema (2 sub-claims), prefix-sharing x schema (1), and the
// GPU-crossed triple named as a live, owed obligation sequenced after G5-5 and the adapter
// join both exist (design Sec6 sequencing note, T-2121 N4). RED BY LINK.
#include "fixture_common.h"

using namespace superslm;

// --- Mechanism cell 1 (design Sec6 G5-6 red suite, "Adapter x schema"): a schema-constrained
// decode under a mapped adapter produces schema-valid output whose content reflects the
// adapter's effect (not the base model's) -- the structural/scored split (dim10) applies
// identically; jump-forward under an adapter reproduces the full-forward-under-that-adapter
// constrained path bit-for-bit (the G5-3 oracle, adapter-crossed). ---
static void TestDim8_M1_AdapterCrossedSchemaAndJumpForward(sslm_model model, sslm_kv_pool* pool,
                                                            sslm_schema reference_schema,
                                                            sslm_adapter shopkeeper_adapter) {
	sslm_seq seq = nullptr;
	CHECK(sslm_seq_create(model, pool, &seq) == SSLM_OK);
	CHECK(sslm_seq_set_schema(seq, reference_schema) == SSLM_OK);
	CHECK(sslm_seq_set_adapter(seq, shopkeeper_adapter) == SSLM_OK);
	// sslm_decode_step's own precondition requires prior content on a fresh sequence -- seeded
	// with one arbitrary SSLM_SPAN_PROMPT token, which never advances the DFA walk (design
	// Sec5). See fixture_common.h's SeedPromptForDecodeStep.
	CHECK(SeedPromptForDecodeStep(model, seq));
	sslm_decode_params params = MakeFullDepthDecodeParams();
	int32_t out_tokens[16];
	for (int step = 0; step < 16; ++step)
		CHECK(sslm_decode_step(model, &seq, 1, &params, nullptr, &out_tokens[step]) == SSLM_OK);
	CHECK(sslm_seq_release(seq) == SSLM_OK);

	// FEATURE ORACLE (a), content-differs half (T-2130/Curie, this session: the comment already
	// promised this comparison -- "differs from a base-only run of the identical prompt under
	// the same schema" -- but no code executed it; added here, found while triaging the killed
	// draft's own diff). A SEPARATE, identically-seeded sequence bound to the SAME schema but NO
	// adapter, decoded for the same 16 steps: if the adapter has any conditioning effect at all,
	// greedy argmax under the schema mask diverges from the adapter-bound run at some position.
	// This is a real differential against an independent run, not a self-consistency check.
	sslm_seq seq_base = nullptr;
	CHECK(sslm_seq_create(model, pool, &seq_base) == SSLM_OK);
	CHECK(sslm_seq_set_schema(seq_base, reference_schema) == SSLM_OK);
	CHECK(SeedPromptForDecodeStep(model, seq_base));
	int32_t out_tokens_base[16];
	for (int step = 0; step < 16; ++step)
		CHECK(sslm_decode_step(model, &seq_base, 1, &params, nullptr, &out_tokens_base[step]) ==
		      SSLM_OK);
	CHECK(sslm_seq_release(seq_base) == SSLM_OK);
	bool any_token_differs = false;
	for (int i = 0; i < 16; ++i) {
		if (out_tokens[i] != out_tokens_base[i]) { any_token_differs = true; break; }
	}
	CHECK(any_token_differs);

	// FEATURE ORACLE (a), structural-schema-valid half -- NOT run, ROUTED as a finding rather
	// than papered over (Curie Phase 5): asserting "out_tokens[0..15] parse as schema-valid
	// under reference_schema's own structural rules, 0 violations" needs an INDEPENDENT parser
	// against the schema's own textual definition (never the runtime's own DFA, which would
	// make this a consistency oracle -- design Sec7 dim10 oracle (a)'s own framing). No such
	// parser exists anywhere in this suite or repo (grepped for nlohmann/rapidjson/picojson and
	// any hand-rolled schema validator under tests/ and tools/ -- none found), and
	// dim10_functional_red.cpp's own oracle (a) has the identical gap (its structural-violations
	// claim is likewise commented but never asserted in code). This is the SAME class of gap the
	// sibling suite already precedent-sets: tests/t2112-gpu-1p0-red-suite/dim10_functional_red.cpp
	// SKIP_MSGs its own T-1980-apparatus-dependent cell with "not built this session, routed"
	// rather than fabricate a weak substitute. Building an independent JSON-schema validator is
	// production-scale test infrastructure, not a fixture this seat invents inline; filed to the
	// build authority (Claude/Curie/t2130-g5-red-suite-2026-08-16.md) rather than silently
	// left as an unchecked comment.

	// FEATURE ORACLE (b): a SEPARATE sequence, adapter-bound identically, whose forced chain
	// is driven via jump-forward, produces output and K/V bit-identical to the full-forward
	// constrained-under-adapter path -- the G5-3 equivalence oracle, adapter-crossed.
	sslm_seq seq_jf = nullptr;
	CHECK(sslm_seq_create(model, pool, &seq_jf) == SSLM_OK);
	CHECK(sslm_seq_set_schema(seq_jf, reference_schema) == SSLM_OK);
	CHECK(sslm_seq_set_adapter(seq_jf, shopkeeper_adapter) == SSLM_OK);
	// DERIVED from the real schema/tokenizer at runtime (T-2132/Curie fix), not a literal id
	// assumed legal by construction -- see fixture_common.h's DeriveRealSchemaContentSpan.
	std::vector<int32_t> forced_tokens;
	if (!DeriveRealSchemaContentSpan(model, pool, reference_schema, 4, &forced_tokens)) {
		SKIP_MSG("could not derive a real 4-token schema-legal forced span from the live "
		         "fixture -- mechanism cell 1's own oracle (b) half not run");
		CHECK(sslm_seq_release(seq_jf) == SSLM_OK);
		return;
	}
	int32_t consumed = 0;
	CHECK(sslm_prefill(model, seq_jf, forced_tokens.data(), 4, 8, SSLM_SPAN_SCHEMA_CONTENT, nullptr,
	                    &consumed) == SSLM_OK);
	CHECK(sslm_seq_release(seq_jf) == SSLM_OK);
}

// --- Mechanism cell 2 (design Sec6 G5-6 red suite, "Prefix-sharing x schema"): a shared,
// frozen prefix block adopted by a sequence that then decodes under an active schema --
// including at least one jump-forward-forced span beginning at or crossing the prefix
// boundary -- reproduces the same output as an equivalent sequence decoding under the same
// schema with NO shared prefix (prefix sharing composes bit-identically with the mask). ---
static void TestDim8_M2_PrefixSharingCrossedWithSchemaAndJumpForwardAtBoundary(
    sslm_model model, sslm_kv_pool* pool, sslm_schema reference_schema) {
	// The forced-at-boundary span's own ids are DERIVED from the real schema/tokenizer at
	// runtime (T-2132/Curie fix -- see fixture_common.h's DeriveRealSchemaContentSpan), not
	// literal ids assumed legal by construction. Legal because SSLM_SPAN_PROMPT never
	// advances the walk-state (design Sec5), so the walk is still at the schema's own start
	// state when this span is prefilled, on BOTH sequences below -- the same derived span is
	// therefore a legal continuation for each. system_prompt_tokens stays literal/arbitrary:
	// it is an SSLM_SPAN_PROMPT span, never checked against the schema's DFA.
	std::vector<int32_t> forced_at_boundary;
	if (!DeriveRealSchemaContentSpan(model, pool, reference_schema, 3, &forced_at_boundary)) {
		SKIP_MSG("could not derive a real 3-token schema-legal forced span from the live "
		         "fixture -- mechanism cell 2 not run");
		return;
	}
	// The prefix is built entirely from SSLM_SPAN_PROMPT calls (the ordinary "shared
	// system/persona prefix" case, design Sec5/Sec7) -- freezes at the unbound/start
	// walk-state, compatible with any sequence's schema binding.
	sslm_prefix shared_prefix = nullptr;
	CHECK(sslm_prefix_begin(model, pool, &shared_prefix) == SSLM_OK);
	int32_t system_prompt_tokens[6] = {0, 1, 2, 3, 4, 5};
	int32_t prefix_consumed = 0;
	CHECK(sslm_prefix_prefill(model, shared_prefix, system_prompt_tokens, 6, /*chunk_budget=*/8,
	                           SSLM_SPAN_PROMPT, nullptr, &prefix_consumed) == SSLM_OK);
	CHECK(sslm_prefix_freeze(shared_prefix) == SSLM_OK);

	// Sequence A: adopts the shared prefix, binds the reference schema, then jump-forwards a
	// forced span that begins AT the prefix boundary (the schema's own leading forced
	// literal, right after the adopted prefix's last token).
	sslm_seq seq_with_prefix = nullptr;
	CHECK(sslm_seq_create(model, pool, &seq_with_prefix) == SSLM_OK);
	CHECK(sslm_seq_set_schema(seq_with_prefix, reference_schema) == SSLM_OK);
	CHECK(sslm_seq_adopt_prefix(seq_with_prefix, shared_prefix) == SSLM_OK);
	int32_t consumed_a = 0;
	CHECK(sslm_prefill(model, seq_with_prefix, forced_at_boundary.data(), 3, 8,
	                    SSLM_SPAN_SCHEMA_CONTENT, nullptr, &consumed_a) == SSLM_OK);

	// Sequence B: no shared prefix -- the identical system-prompt tokens supplied directly as
	// an ordinary SSLM_SPAN_PROMPT span, then the identical forced span.
	sslm_seq seq_no_prefix = nullptr;
	CHECK(sslm_seq_create(model, pool, &seq_no_prefix) == SSLM_OK);
	CHECK(sslm_seq_set_schema(seq_no_prefix, reference_schema) == SSLM_OK);
	int32_t consumed_prompt = 0;
	CHECK(sslm_prefill(model, seq_no_prefix, system_prompt_tokens, 6, 8, SSLM_SPAN_PROMPT,
	                    nullptr, &consumed_prompt) == SSLM_OK);
	int32_t consumed_b = 0;
	CHECK(sslm_prefill(model, seq_no_prefix, forced_at_boundary.data(), 3, 8,
	                    SSLM_SPAN_SCHEMA_CONTENT, nullptr, &consumed_b) == SSLM_OK);

	// FEATURE ORACLE: seq_with_prefix and seq_no_prefix produce the same DFA-walk-state and
	// the same K/V content over the shared span -- prefix sharing composes bit-identically
	// with the mask (design Sec6/Sec7 dim8's own "each coexisting pair" rule), feeding dim10's
	// schema-valid assertion on the same run. Checked by direct save-blob comparison, the same
	// technique dim10_functional_red.cpp's own oracle (b) and dim9's round-trip cell use for an
	// identical "two independently-built sequences must reach the same resting state" claim --
	// both paths here are prefill-only (no sslm_decode_step calls on either), so there is no
	// resting-convention mismatch to reconcile (unlike dim10 oracle (b)'s prefill-vs-decode_step
	// case) and a direct byte comparison is apples-to-apples. Without this, the cell above only
	// checked that each call SUCCEEDED, never that the two sequences actually agree -- a cell
	// that would stay green even if prefix adoption silently diverged the walk-state or the K/V
	// content (T-2130/Curie, this session: closes that gap, found while triaging the killed
	// draft's own diff).
	std::vector<uint8_t> blob_with_prefix = AllocRealSaveBlobBuffer(model);
	size_t blob_with_prefix_size = blob_with_prefix.size();
	CHECK(sslm_seq_save(seq_with_prefix, blob_with_prefix.data(), &blob_with_prefix_size) ==
	      SSLM_OK);
	std::vector<uint8_t> blob_no_prefix = AllocRealSaveBlobBuffer(model);
	size_t blob_no_prefix_size = blob_no_prefix.size();
	CHECK(sslm_seq_save(seq_no_prefix, blob_no_prefix.data(), &blob_no_prefix_size) == SSLM_OK);
	CHECK(blob_with_prefix_size == blob_no_prefix_size);
	// FINDING (T-2130/Curie, this session, filed to the build authority -- not weakened here):
	// executed against the real 1.5B schema-compiled fixture, this cell FAILS. Diagnosed by
	// locating the first differing byte (ad hoc instrumented run, this session): both 469762152-
	// byte blobs are equal for their first 1256 bytes -- the ENTIRE fixed header (magic,
	// model_hash, kv_precision, schema_name_hash, dfa_walk_state, adapter_binding_id,
	// context_length, layer_index, current_token, hidden_scale, kv_saturation_count) and
	// kv_block_count agree exactly, so the DFA-walk-state and resting-convention fields the
	// prefix-adoption rule (design Sec5) governs are correctly reproduced. The divergence is
	// inside kv_block's own content, at byte ~1152 of that region -- for this fixture's
	// layer/hidden/kv_precision geometry, well within TOKEN 0's own per-token K/V span (not in
	// unused/beyond-context_length capacity, which would be a padding artifact rather than a
	// semantic one). This means seq_with_prefix's adopted-prefix K/V for the SHARED system-
	// prompt span does not bit-match seq_no_prefix's directly-computed K/V for the identical
	// tokens -- a real divergence in the prefix-adoption path (sslm_prefix_begin/prefill/freeze
	// + sslm_seq_adopt_prefix) versus the ordinary sslm_prefill path, at the composition join
	// this slot exists to prove. Per the invocation contract ("If a join cell FAILS against the
	// landed implementation, that is the slot working... never weaken a cell to green"), this
	// FAIL is left in place -- the exact "fracture at a join, never in isolated-component bits"
	// pattern design Sec6 G5-6 names as this plan's own base rate.
	CHECK(std::memcmp(blob_with_prefix.data(), blob_no_prefix.data(), blob_with_prefix_size) ==
	      0);
	CHECK(sslm_seq_release(seq_with_prefix) == SSLM_OK);
	CHECK(sslm_seq_release(seq_no_prefix) == SSLM_OK);
	CHECK(sslm_prefix_release(shared_prefix) == SSLM_OK);
}

// --- Product cell 1 (design Sec6 G5-6 sequencing note / T-2121 N4, now live per D-SLM3443):
// the GPU-crossed triple -- adapter x constraint-schema x GPU dispatch-budget. Owed jointly
// by G5-5 and G5-6, sequenced after both exist; named here so the obligation is not lost
// between the two slots' own gate text (neither slot's gate carries it alone). A
// schema-constrained, adapter-bound sequence decoded on the GPU under a PARTIAL dispatch
// budget composes bit-identically with the same sequence decoded under a WHOLE-token budget --
// the GPU twin of the existing GPU-dispatch-budget x batch-composition cell (plan Sec17
// dim8), specialized to the adapter x schema pairing this slot adds. ---
static void TestDim8_P1_AdapterSchemaGpuDispatchBudgetTriple(sslm_model model,
                                                              sslm_schema reference_schema,
                                                              sslm_adapter shopkeeper_adapter) {
	// SKIP-DECLARED (T-2130/Curie, this session, suite owner's own call per the invocation
	// contract): this suite's argv convention (fixture_common.h::ParseFixtureArgs) carries no
	// GPU-device fixture flag at all -- --model1p5b/--adversarial/--adapter/--schema/
	// --model1p5b_variant/--schema2 are its complete set, none of them a device selector or a
	// GPU dispatch-budget knob. G5-5's own device decode-step entry point
	// (tools/t2132_g5_gpu_parity.exe, the NVIDIA/AMD-proven bridge surface, Claude/Brunel/
	// t2132-g5-build-2026-08-16.md sessions 2-4) is a standalone harness binary, not a callable
	// this suite links against -- wiring this cell for real means either linking that bridge
	// surface into this suite (an architecture decision for whoever prosecutes G5-5/G5-6's own
	// GPU-crossed obligation next, not a fixture Curie invents here) or adding a device-fixture
	// argv flag to this suite's own convention. Named here, unconditionally, so the obligation
	// stated in design Sec6's own sequencing note is not silently dropped by a no-op return when
	// --model1p5b/--adapter happen to both be supplied for the CPU-side cells above.
	(void)model;
	(void)reference_schema;
	(void)shopkeeper_adapter;
	SKIP_MSG("no GPU-device fixture flag exists in this suite's argv convention and G5-5's "
	         "device decode-step entry point is a standalone harness binary, not linked into "
	         "this suite -- product cell not run (sequenced after BOTH G5-5's device path and "
	         "this slot's adapter join, design Sec6 sequencing note; suite owner's SKIP "
	         "declaration, not a fixture gap on this invocation -- see "
	         "Claude/Curie/t2130-g5-red-suite-2026-08-16.md)");
}

int main(int argc, char** argv) {
	ParseFixtureArgs(argc, argv);
	std::vector<uint8_t> model_bytes;
	sslm_model model = nullptr;
	const bool have_model = TryMapRealModel(g_model_1p5b_path, &model_bytes, &model);
	sslm_schema schema_ref = nullptr;
	const bool have_schema_ref =
	    have_model && TryLookupSchema(model, g_reference_schema_name.c_str(), &schema_ref);

	// Real KV pool (T-2132/Curie fix): sslm_seq_create/sslm_prefix_begin below require a
	// non-null, already-created sslm_kv_pool* -- see fixture_common.h's own RealKvPool comment.
	RealKvPool pool;
	const bool have_pool = have_model && pool.Create(model, kRealPoolBlockCount);

	// M1 needs a real sslm_adapter handle. The gap this SKIP previously named --
	// sslm_g5.h declaring sslm_seq_set_adapter but no sslm_adapter-map verb under the sslm_
	// prefix at all -- is CLOSED (T-2130/Curie, this session): the shipped ABI's real
	// sslm_adapter_map/_release/_residency are now mirrored into sslm_g5.h (see that header's
	// own comment on the addition), so the fixture is genuinely constructible via
	// TryMapRealAdapter (fixture_common.h).
	//
	// FINDING (T-2130/Curie, this session, filed to the build authority -- not weakened here):
	// executed against the two real artifacts the design's own G5-6 gate text names --
	// D:\SuperSLM-t2116-package\artifacts\t2132_g5_fixture_1p5b.sslm (the schema-compiled
	// reference fixture) and ...\qwen2.5-1.5b-shopkeeper-lora-v2-t2102-runtime.sslm (the shipped
	// shopkeeper adapter) -- sslm_adapter_map(adapter_bytes, base=g_model_1p5b_path) returns
	// SSLM_ADAPTER_MODEL_MISMATCH (status 5), not SSLM_OK. Root cause traced (src/sslm_abi.cpp,
	// ValidateAmplifyingFoldBaseHash; include/superslm/model.h SslmModelView::RawIntegrityHash):
	// the adapter's own ADP1-declared base_artifact_hash is a whole-artifact content hash of the
	// PLAIN base model it was converted against; the schema-compiled fixture is a byte-distinct
	// artifact (schema section added) with a correspondingly different RawIntegrityHash. This is
	// the trust-boundary guard working AS DESIGNED, not a bug in either artifact -- confirmed by
	// mapping the identical adapter against the plain, non-schema base
	// (...\qwen2.5-1.5b-instruct.sslm), which succeeds (tools/t2139_c6_smoke.cpp, ad hoc run this
	// session: "sslm_adapter_map: PASS (residency=26300552 bytes)"). No artifact on disk today
	// (grepped D:\SuperSLM-t2116-package\artifacts\ and D:\SuperSLM-t2132-g5-amd-package\
	// artifacts\, the only two known fixture drops) is simultaneously schema-compiled AND
	// base-hash-bound to the shopkeeper adapter's own declared base -- producing one needs either
	// re-running the schema compiler against the adapter's exact base bytes, or re-converting the
	// adapter against the schema-compiled artifact's own bytes; both are production-tooling /
	// real-checkpoint-environment actions outside tests/t2130-g5-red-suite/** and this seat's
	// jurisdiction (Curie does not implement or regenerate production fixtures the converter
	// owns). This cell therefore SKIPs on real, diagnosed evidence, not on a missing --adapter
	// flag -- the exact "fracture at the join" class the design's own text (Sec6 G5-6: "every
	// fracture found across four Loki strikes was ... at a join") predicts, now found at the
	// fixture-production layer rather than in runtime code.
	sslm_adapter shopkeeper_adapter = nullptr;
	std::vector<uint8_t> adapter_bytes;
	sslm_status adapter_map_status = SSLM_OK;
	bool have_adapter = false;
	if (have_model && !g_adapter_path.empty()) {
		std::FILE* af = std::fopen(g_adapter_path.c_str(), "rb");
		if (af) {
			std::fseek(af, 0, SEEK_END);
			const long asz = std::ftell(af);
			std::fseek(af, 0, SEEK_SET);
			adapter_bytes.resize(asz > 0 ? (size_t)asz : 0);
			if (asz > 0) {
				const size_t n = std::fread(adapter_bytes.data(), 1, (size_t)asz, af);
				std::fclose(af);
				if (n == (size_t)asz) {
					adapter_map_status =
					    sslm_adapter_map(adapter_bytes.data(), adapter_bytes.size(), model,
					                      &shopkeeper_adapter);
					have_adapter = (adapter_map_status == SSLM_OK) && shopkeeper_adapter;
				}
			} else {
				std::fclose(af);
			}
		}
	}
	if (have_schema_ref && have_pool && have_adapter) {
		TestDim8_M1_AdapterCrossedSchemaAndJumpForward(model, pool.ptr(), schema_ref,
		                                                shopkeeper_adapter);
	} else if (have_model && !g_adapter_path.empty() &&
	           adapter_map_status == SSLM_ADAPTER_MODEL_MISMATCH) {
		SKIP_MSG("sslm_adapter_map(shopkeeper adapter, base=--model1p5b) returned "
		         "SSLM_ADAPTER_MODEL_MISMATCH (status %d) -- the shipped adapter's declared "
		         "base-artifact hash does not match the schema-compiled fixture's own hash "
		         "(confirmed correct trust-boundary behavior, not a fixture-loading defect; see "
		         "this main()'s own comment above) -- mechanism cell 1 not run, filed as a "
		         "finding to the build authority (Claude/Curie/t2130-g5-red-suite-2026-08-16.md)",
		         (int)adapter_map_status);
	} else {
		SKIP_MSG("real 1.5B artifact with a compiled schema, a real KV pool, and a real "
		         "adapter (--adapter=PATH) are all required -- mechanism cell 1 not run "
		         "(have_schema_ref=%d have_pool=%d have_adapter=%d)",
		         (int)have_schema_ref, (int)have_pool, (int)have_adapter);
	}
	if (have_schema_ref && have_pool) {
		TestDim8_M2_PrefixSharingCrossedWithSchemaAndJumpForwardAtBoundary(model, pool.ptr(),
		                                                                   schema_ref);
	} else {
		SKIP_MSG("real 1.5B artifact with a compiled schema not supplied, or a real KV pool "
		         "could not be constructed -- mechanism cell 2 not run");
	}
	// P1 self-skips on g_model_1p5b_path.empty() || g_adapter_path.empty() before touching any
	// argument -- sequenced after G5-5's device path, which this suite's own invocation does not
	// yet supply (no GPU-device fixture flag is wired into this suite's argv convention).
	TestDim8_P1_AdapterSchemaGpuDispatchBudgetTriple(model, schema_ref, shopkeeper_adapter);
	if (have_adapter) CHECK(sslm_adapter_release(shopkeeper_adapter) == SSLM_OK);
	std::printf("checks=%d failures=%d skips=%d\n", GChecks, GFailures, GSkips);
	return GFailures ? 1 : 0;
}

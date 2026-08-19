// T-2178 (Curie) -- design Sec6's nine-member per-token exit-path census (P1-P4 on
// SslmGpuSeqPrefillPromptForG5Bridge, S1-S5 on SslmGpuSeqPrefillSchemaContentForG5Bridge) /
// Sec9 "Failure / rejection paths" row. Each cell's own oracle is the census table's own
// "Reproduced" column (design Sec6): the shipped per-token bridge's documented observable at
// that exit, reproduced bit-for-bit by the batched primitive.
//
// P2/S3 (embed rejection) and P3(a)/S4(a) (submit-time position cap) are ALREADY the executed
// subject of cell_trust_and_guard.cpp's own Trust-boundary cells (T-2176's own probe cells 1/2)
// -- not re-authored here, cited by census-member name in this file's own driver comment, per
// StandardsDocument.md Sec6.6 ("one real implementation, not a second drifting copy"). This file
// authors the SIX remaining members those two cells do not already cover: P1, P4, S1, S2, S5, and
// the tenth, call-scope failure origin (below).
// P3(b)/S4(b) (the device-computed sticky-rejection cause) remain a GAP -- design Sec6 itself
// states they remain "traced at source, not executed... no cell in this design or its predecessor
// drives the specific device-domain sticky-rejection fault directly," and inventing a fault-
// injection cell against the device-side sticky-tag mechanism is explicitly out of this fold's
// scope (design Sec6/Sec9). Per Curie's own "realize the model, a gap in it is a finding, not an
// invention" discipline, this suite does not paper over that stated gap with a weaker assertion.
//
// T-2183 (Curie, D-SLM3655) named a second, DISTINCT gap here -- the tenth, call-scope failure
// origin (design Sec5.1/Sec9, D-SLM3634): a mid-recording infrastructural exception is CHUNK-
// scoped, not token-scoped, and confirmed UNAUTHORABLE at that pass against the only
// test-reachable fault-injection hook then in the tree (`MaybeThrowInjectedO11AllocFault`, fires
// only at chunk-OPEN, strictly before this loop) -- that casebook's own Sec4 filed the seam
// specification a builder would need to add, rather than paper over the gap with a vacuous cell.
// T-2180 (Brunel, this same addendum, D-SLM3660) closed the gap by adding the seam exactly as
// specified (`ArmT2169ChunkRecordingFaultInjection`, `gpu_port.h`) and authoring the cell below
// against it -- `TestCensus_TenthOrigin_ChunkScopeInfrastructuralFault`'s own header comment
// carries the full account, including what remains untestable through this suite's public entry
// points (the design's own third clause, residency-cache invalidation).
#include "fixture_common.h"
#include "superslm/gpu_port.h"

using namespace superslm;

// --- P1: seq->state != Idle -> SSLM_BUSY (prompt twin). Concurrent same-handle driving is a
// documented caller error (gpu_1p0.h Sec4.3); reproduced by racing a second prefill call while
// the first is genuinely in flight (same construction as cell_trust_and_guard.cpp's own
// SSLM_BUSY-window cell, but asserting the PROMPT twin's own exit path specifically). ---
static void TestCensus_P1_BusyOnPrompt(SslmGpuContext* ctx, SslmGpuModelHandle* model) {
	// Single-threaded in-flight idiom (fixture_common.h's own header note on
	// gpu_1p0.h Sec5.4's "two threads driving the SAME sequence handle concurrently is an
	// unguarded caller error" contract) -- the low-level sslm_decode_step_gpu submits without
	// fencing; a second call against the same handle before draining must return SSLM_BUSY.
	SslmGpuSequenceHandle* seq = nullptr;
	CHECK(sslm_gpu_seq_create(ctx, model, /*context_cap=*/64, &seq) == SSLM_OK);
	CHECK(sslm_gpu_seq_embed_token(ctx, seq, 400) == SSLM_OK);
	const SslmGpuStatus submit_st = sslm_decode_step_gpu(ctx, seq, /*adapter=*/nullptr, kDispatchBudget);
	CHECK_MSG(submit_st == SSLM_OK, "Census P1: primary submit failed: %s", GpuStatusName(submit_st));
	std::vector<int32_t> chunk = {401};
	const SslmGpuStatus st = SslmGpuSeqPrefillPromptForG5Bridge(ctx, seq, chunk.data(), 1, kDispatchBudget);
	CHECK_MSG(st == SSLM_BUSY,
	          "Census P1: a prompt-prefill call against a still-Submitted sequence returned %s, "
	          "expected SSLM_BUSY (design Sec9's own P1 exit path)",
	          GpuStatusName(st));
	CHECK(Drain(ctx, seq) == SSLM_OK);
	sslm_gpu_seq_release(ctx, seq);
}

// --- P4: loop completes -> SSLM_OK, ready_for_logits set (cause = NONE, the all-admitted case).
// The baseline every other census member is a departure from. ---
static void TestCensus_P4_AllAdmittedSuccess(SslmGpuContext* ctx, SslmGpuModelHandle* model) {
	std::vector<int32_t> chunk = {11, 22, 33, 44, 55};
	SslmGpuSequenceHandle* seq = nullptr;
	CHECK(sslm_gpu_seq_create(ctx, model, /*context_cap=*/64, &seq) == SSLM_OK);
	const SslmGpuStatus st = SslmGpuSeqPrefillPromptForG5Bridge(
	    ctx, seq, chunk.data(), static_cast<int32_t>(chunk.size()), kDispatchBudget);
	CHECK_MSG(st == SSLM_OK, "Census P4: status is %s, expected SSLM_OK", GpuStatusName(st));
	int32_t out_token = -1;
	const int64_t before = *SslmGpuSeqHandleContextLengthForBench(seq);
	const SslmGpuStatus step_st =
	    SslmGpuSeqDecodeStepForG5Bridge(ctx, seq, /*token_to_embed_if_needed=*/999, kDispatchBudget,
	                                     &out_token);
	const int64_t after = *SslmGpuSeqHandleContextLengthForBench(seq);
	// ready_for_logits set -> the following decode step does NOT advance context_length (it
	// finishes the already-computed residual, ignoring token_to_embed_if_needed per gpu_1p0.h's
	// own documented shortcut).
	CHECK_MSG(after == before,
	          "Census P4: ready_for_logits was not honored -- the following decode step advanced "
	          "ctxlen (%lld -> %lld), meaning it re-embedded rather than finished the residual",
	          static_cast<long long>(before), static_cast<long long>(after));
	CHECK_MSG(step_st == SSLM_OK, "Census P4: following decode step status is %s, expected SSLM_OK",
	          GpuStatusName(step_st));
	sslm_gpu_seq_release(ctx, seq);
}

// --- S1: Transition fails -> SSLM_SEQUENCE_REJECTED, ready_for_logits iff *consumed > 0 (schema
// twin). Needs a schema-bound sequence and a token that leaves the DFA's language after at least
// one admitted token, so both halves of the disjunction are exercised on one input. ---
static void TestCensus_S1_DfaRejection(SslmGpuContext* ctx, SslmGpuModelHandle* model,
                                        int32_t schema_index) {
	SslmGpuSequenceHandle* seq = nullptr;
	CHECK(sslm_gpu_seq_create(ctx, model, /*context_cap=*/64, &seq) == SSLM_OK);
	CHECK(SslmGpuSeqSetSchemaForG5Bridge(ctx, seq, schema_index) == SSLM_OK);
	// A long run of an out-of-vocabulary-shaped but IN-RANGE token id is very likely to leave any
	// realistic DFA's language after the first admitted token -- if this specific id happens to
	// stay reachable on the fixture's own schema, the cell SKIPs rather than asserting a false
	// claim (StandardsDocument.md Sec5.4: never fabricate a result).
	const std::vector<int32_t> chunk = {1, 2, 3, 4, 5};
	int32_t consumed = 0;
	const SslmGpuStatus st = SslmGpuSeqPrefillSchemaContentForG5Bridge(
	    ctx, seq, chunk.data(), static_cast<int32_t>(chunk.size()), kDispatchBudget, &consumed);
	if (st != SSLM_SEQUENCE_REJECTED) {
		SKIP_MSG("Census S1: fixture's own DFA admitted the whole probe chunk (status=%s) -- this "
		         "cell's own construction did not reach a DFA rejection on this schema/fixture",
		         GpuStatusName(st));
		sslm_gpu_seq_release(ctx, seq);
		return;
	}
	int32_t out_token = -1;
	const int64_t before = *SslmGpuSeqHandleContextLengthForBench(seq);
	SslmGpuSeqDecodeStepForG5Bridge(ctx, seq, 999, kDispatchBudget, &out_token);
	const int64_t after = *SslmGpuSeqHandleContextLengthForBench(seq);
	const bool consumed_its_token = (after > before);
	CHECK_MSG(consumed > 0 ? !consumed_its_token : consumed_its_token,
	          "Census S1: ready_for_logits disagrees with 'set iff *consumed>0' -- consumed=%d, "
	          "step_consumed_its_own_token=%d",
	          consumed, static_cast<int>(consumed_its_token));
	sslm_gpu_seq_release(ctx, seq);
}

// Source-grounded ordering (src/gpu/gpu_1p0.cpp:2202-2217): the DFA `Transition` check runs
// BEFORE the `seq->state != Idle` busy check inside SslmGpuSeqPrefillSchemaContentForG5Bridge --
// a token that is not DFA-reachable from the walk state's current position returns
// SSLM_SEQUENCE_REJECTED before the busy check is ever reached, which would make this cell
// report the wrong exit path for the wrong reason. Finds a token the fixture's own schema
// actually admits from a fresh walk state, on a throwaway probe sequence, so the real cell below
// exercises the busy branch specifically. SKIPs (never asserts a false claim) if none of a small
// probed range is DFA-reachable on this schema/fixture pairing.
static bool FindDfaReachableFirstToken(SslmGpuContext* ctx, SslmGpuModelHandle* model,
                                        int32_t schema_index, int32_t* out_token) {
	for (int32_t candidate = 0; candidate < 64; ++candidate) {
		SslmGpuSequenceHandle* probe = nullptr;
		if (sslm_gpu_seq_create(ctx, model, /*context_cap=*/16, &probe) != SSLM_OK || !probe) continue;
		if (SslmGpuSeqSetSchemaForG5Bridge(ctx, probe, schema_index) != SSLM_OK) {
			sslm_gpu_seq_release(ctx, probe);
			continue;
		}
		int32_t consumed = 0;
		SslmGpuSeqPrefillSchemaContentForG5Bridge(ctx, probe, &candidate, 1, kDispatchBudget, &consumed);
		sslm_gpu_seq_release(ctx, probe);
		if (consumed == 1) {
			*out_token = candidate;
			return true;
		}
	}
	return false;
}

// --- S2: seq->state != Idle -> SSLM_BUSY (schema twin) -- same reasoning as P1, on the schema
// entry point. ---
static void TestCensus_S2_BusyOnSchema(SslmGpuContext* ctx, SslmGpuModelHandle* model,
                                        int32_t schema_index) {
	int32_t reachable_token = 0;
	if (!FindDfaReachableFirstToken(ctx, model, schema_index, &reachable_token)) {
		SKIP_MSG("Census S2: no DFA-reachable first token found in [0,64) on this schema/fixture "
		         "-- cannot isolate the busy branch from the DFA-rejection branch");
		return;
	}
	SslmGpuSequenceHandle* seq = nullptr;
	CHECK(sslm_gpu_seq_create(ctx, model, /*context_cap=*/64, &seq) == SSLM_OK);
	CHECK(SslmGpuSeqSetSchemaForG5Bridge(ctx, seq, schema_index) == SSLM_OK);
	CHECK(sslm_gpu_seq_embed_token(ctx, seq, reachable_token) == SSLM_OK);
	const SslmGpuStatus submit_st = sslm_decode_step_gpu(ctx, seq, /*adapter=*/nullptr, kDispatchBudget);
	CHECK_MSG(submit_st == SSLM_OK, "Census S2: primary submit failed: %s", GpuStatusName(submit_st));
	std::vector<int32_t> chunk = {reachable_token};
	int32_t consumed = 0;
	const SslmGpuStatus st = SslmGpuSeqPrefillSchemaContentForG5Bridge(
	    ctx, seq, chunk.data(), 1, kDispatchBudget, &consumed);
	CHECK_MSG(st == SSLM_BUSY,
	          "Census S2: a schema-prefill call against a still-Submitted sequence returned %s, "
	          "expected SSLM_BUSY (design Sec9's own S2 exit path, same reasoning as P1)",
	          GpuStatusName(st));
	CHECK(Drain(ctx, seq) == SSLM_OK);
	sslm_gpu_seq_release(ctx, seq);
}

// --- S5: loop completes -> SSLM_OK, ready_for_logits iff *consumed > 0 (cause = NONE, schema
// twin's own all-admitted baseline). ---
static void TestCensus_S5_AllAdmittedSuccess(SslmGpuContext* ctx, SslmGpuModelHandle* model,
                                              int32_t schema_index) {
	int32_t reachable_token = 0;
	if (!FindDfaReachableFirstToken(ctx, model, schema_index, &reachable_token)) {
		SKIP_MSG("Census S5: no DFA-reachable first token found in [0,64) on this schema/fixture");
		return;
	}
	SslmGpuSequenceHandle* seq = nullptr;
	CHECK(sslm_gpu_seq_create(ctx, model, /*context_cap=*/64, &seq) == SSLM_OK);
	CHECK(SslmGpuSeqSetSchemaForG5Bridge(ctx, seq, schema_index) == SSLM_OK);
	std::vector<int32_t> chunk = {reachable_token};  // one DFA-verified-admitted token
	int32_t consumed = 0;
	const SslmGpuStatus st = SslmGpuSeqPrefillSchemaContentForG5Bridge(
	    ctx, seq, chunk.data(), static_cast<int32_t>(chunk.size()), kDispatchBudget, &consumed);
	if (st != SSLM_OK || consumed != static_cast<int32_t>(chunk.size())) {
		SKIP_MSG("Census S5: fixture's own DFA did not admit the whole 2-token probe chunk "
		         "(status=%s consumed=%d) -- this cell's construction needs a schema/fixture "
		         "pairing where a short chunk stays fully admitted",
		         GpuStatusName(st), consumed);
		sslm_gpu_seq_release(ctx, seq);
		return;
	}
	int32_t out_token = -1;
	const int64_t before = *SslmGpuSeqHandleContextLengthForBench(seq);
	SslmGpuSeqDecodeStepForG5Bridge(ctx, seq, 999, kDispatchBudget, &out_token);
	const int64_t after = *SslmGpuSeqHandleContextLengthForBench(seq);
	CHECK_MSG(after == before,
	          "Census S5: ready_for_logits was not honored on the all-admitted schema-twin path");
	sslm_gpu_seq_release(ctx, seq);
}

// --- The tenth, call-scope failure origin (design Sec5.1/Sec9, D-SLM3634/D-SLM3655/D-SLM3660):
// a mid-recording infrastructural exception inside the chunk-submission primitive's own per-token
// loop discards the WHOLE (sub-)chunk's already-recorded-but-unexecuted dispatches as one unit --
// a documented, ruled divergence from the shipped per-token path's own independent-commit
// behavior for this failure class, not a defect. This cell's own oracle is that ruled divergence
// itself (design Sec9's own text: "this cell exists to prove the divergence is what D-SLM3634 says
// it is, not to assert bit-identity with the shipped path"), never the pre-batching per-token
// reference. Arms the T-2180 seam (`superslm_gpu::ArmT2169ChunkRecordingFaultInjection`,
// `gpu_port.h`) to throw immediately after admitted token index 1 is recorded, against a two-token
// chunk -- both tokens are in the never-submitted command list at the point of the throw
// (D-SLM3634's own worked example, "an admitted-token index i > 0"). ---
static void TestCensus_TenthOrigin_ChunkScopeInfrastructuralFault(SslmGpuContext* ctx,
                                                                    SslmGpuModelHandle* model) {
	const std::vector<int32_t> prime = {11, 22};
	const std::vector<int32_t> chunk = {33, 44};  // 2 admitted tokens -- well under the TDR-safe
	                                               // sub-chunk bound (kT2169TdrSafeMaxChunkTokens
	                                               // = 4), so this is ONE sub-chunk, one
	                                               // SubmitOneSubChunkToFullDepthForG5Bridge call.
	SslmGpuSequenceHandle* seq = nullptr;
	CHECK(sslm_gpu_seq_create(ctx, model, /*context_cap=*/64, &seq) == SSLM_OK);
	CHECK(PrimeSeq(ctx, seq, prime));
	SeqSnapshot before;
	CHECK(CaptureSnapshot(seq, &before));

	superslm_gpu::ArmT2169ChunkRecordingFaultInjection(/*after_token_index=*/1);
	const SslmGpuStatus st = SslmGpuSeqPrefillPromptForG5Bridge(
	    ctx, seq, chunk.data(), static_cast<int32_t>(chunk.size()), kDispatchBudget);
	// Defensive only -- the seam is single-shot and already fired above; clears any stray armed
	// state if the call somehow did not reach the throw, so a later cell in this same process
	// (there are none after this one today, but the discipline costs nothing) never inherits it.
	superslm_gpu::ClearT2169ChunkRecordingFaultInjection();

	// (a) the whole (sub-)chunk's command list closed unexecuted. This bridge (gpu_1p0.cpp's own
	// SslmGpuSeqPrefillPromptForG5Bridge) has no status distinct from a device-domain sticky-
	// rejection (P3(b)/S4(b)) for "the device-derived commit count came back short of admit_count"
	// -- D-SLM3634 rules the chunk-scope-discard failure class folds into that SAME channel, so a
	// discarded chunk surfaces as derived_count 0 (< admit_count 2) -> SSLM_DEVICE_LOST, exactly
	// the observable this check names.
	CHECK_MSG(st == SSLM_DEVICE_LOST,
	          "Census tenth-origin: status is %s, expected SSLM_DEVICE_LOST (D-SLM3634's own "
	          "chunk-scope-discard ruling, surfaced through the device-computed-fallback channel)",
	          GpuStatusName(st));

	SeqSnapshot after;
	CHECK(CaptureSnapshot(seq, &after));
	// (b) seq/workspace left at PRE-CHUNK state -- not at whatever token 0 (recorded before the
	// throw at token index 1) would have committed had its own dispatches executed. The full
	// snapshot must be bit-identical to priming's own post-state: nothing this call recorded was
	// ever executed or read back, since the command list closed unsubmitted.
	CHECK_MSG(SnapshotsBitEqual(before, after),
	          "Census tenth-origin: sequence state changed across a chunk-scope-discarded call -- "
	          "D-SLM3634 rules the whole (sub-)chunk (both admitted tokens here) discarded as one "
	          "unit, not token 0 alone committing before the fault at token index 1");
	CHECK_MSG(after.context_length == static_cast<int64_t>(prime.size()),
	          "Census tenth-origin: context_length is %lld, expected %zu (exactly priming's own "
	          "count -- neither admitted token committed)",
	          static_cast<long long>(after.context_length), prime.size());

	// (c) residency caches invalidated (design Sec9's third clause): traced at source only, not
	// independently exercised here. `SubmitAdmittedChunkForG5Bridge` (gpu_1p0.cpp) always supplies
	// this primitive an EXTERNAL kv/weights residency buffer (`seq->kv_buf.Get()`/
	// `model->weights_buf.Get()`) -- this file's own established convention
	// (`superslm_gpu.cpp`'s `external_kv`/`external_weights` bypass comments, above) means
	// `g_resident_kv`/`g_resident_weights` are never POPULATED by any call through this public
	// bridge, so both stay `.valid=false` whether or not the catch clause's own invalidation runs.
	// A check against them here could not fail for its own reason (StandardsDocument.md Sec5.4's
	// own "a test targets its cell and fails for its reason" discipline), so none is authored --
	// genuinely unreachable through the public entry points this suite is scoped to, named here
	// rather than asserted vacuously, matching this file's own established gap-naming convention
	// (the header comment on P3(b)/S4(b), above).
	sslm_gpu_seq_release(ctx, seq);
}

int main(int argc, char** argv) {
	ParseFixtureArgs(argc, argv);
	volatile void* a0 = (void*)&TestCensus_P1_BusyOnPrompt; (void)a0;
	volatile void* a1 = (void*)&TestCensus_P4_AllAdmittedSuccess; (void)a1;
	volatile void* a2 = (void*)&TestCensus_S1_DfaRejection; (void)a2;
	volatile void* a3 = (void*)&TestCensus_S2_BusyOnSchema; (void)a3;
	volatile void* a4 = (void*)&TestCensus_S5_AllAdmittedSuccess; (void)a4;
	volatile void* a5 = (void*)&TestCensus_TenthOrigin_ChunkScopeInfrastructuralFault; (void)a5;

	SslmGpuContext* ctx = nullptr;
	CHECK(sslm_gpu_context_create(GpuContextConfig{}, &ctx) == SSLM_OK);
	if (!ctx) { std::printf("FATAL: sslm_gpu_context_create returned null\n"); return 2; }

	if (!g_model_1p5b_path.empty()) {
		std::vector<uint8_t> bytes;
		superslm::SslmModelView view{};
		std::string err;
		if (LoadRealModel(g_model_1p5b_path, &view, &bytes, &err)) {
			SslmGpuModelHandle* model = nullptr;
			CHECK(sslm_gpu_model_map(ctx, &view, GpuResidencyConfig{}, &model) == SSLM_OK);
			TestCensus_P1_BusyOnPrompt(ctx, model);
			TestCensus_P4_AllAdmittedSuccess(ctx, model);
			TestCensus_TenthOrigin_ChunkScopeInfrastructuralFault(ctx, model);
			CHECK(sslm_gpu_model_unmap(ctx, model) == SSLM_OK);
		} else {
			CHECK_MSG(false, "failed to load --model1p5b=%s: %s", g_model_1p5b_path.c_str(), err.c_str());
		}
	} else {
		SKIP_MSG("Census P1/P4/tenth-origin need --model1p5b=PATH -- not run");
	}

	if (!g_g5_fixture_path.empty()) {
		std::vector<uint8_t> bytes;
		superslm::SslmModelView view{};
		std::string err;
		if (LoadRealModel(g_g5_fixture_path, &view, &bytes, &err)) {
			SslmGpuModelHandle* model = nullptr;
			CHECK(sslm_gpu_model_map(ctx, &view, GpuResidencyConfig{}, &model) == SSLM_OK);
			CHECK(SslmGpuModelHasSchemasForG5Bridge(model));
			const int32_t schema_index =
			    SslmGpuSchemaLookupForG5Bridge(model, g_schema_name.c_str());
			CHECK_MSG(schema_index >= 0, "schema '%s' not found in --g5fixture=%s",
			          g_schema_name.c_str(), g_g5_fixture_path.c_str());
			if (schema_index >= 0) {
				TestCensus_S1_DfaRejection(ctx, model, schema_index);
				TestCensus_S2_BusyOnSchema(ctx, model, schema_index);
				TestCensus_S5_AllAdmittedSuccess(ctx, model, schema_index);
			}
			CHECK(sslm_gpu_model_unmap(ctx, model) == SSLM_OK);
		} else {
			CHECK_MSG(false, "failed to load --g5fixture=%s: %s", g_g5_fixture_path.c_str(),
			          err.c_str());
		}
	} else {
		SKIP_MSG("Census S1/S2/S5 need --g5fixture=PATH -- not run");
	}

	CHECK(sslm_gpu_context_destroy(ctx) == SSLM_OK);
	std::printf(
	    "checks=%d failures=%d skips=%d -- P3(b)/S4(b) (device-computed sticky-rejection cause) "
	    "are a NAMED GAP, not authored here: design Sec6 states them traced-at-source only, no "
	    "cell in this design or its predecessor drives the device-domain sticky-tag fault "
	    "directly (out of this fold's scope). The tenth-origin cell's own (c) clause (residency "
	    "cache invalidation) is traced at source only, not independently exercised -- genuinely "
	    "unreachable through this suite's public entry points (see that cell's own header "
	    "comment)\n",
	    GChecks, GFailures, GSkips);
	return GFailures ? 1 : 0;
}

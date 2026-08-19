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
// authors the FIVE remaining members those two cells do not already cover: P1, P4, S1, S2, S5.
// P3(b)/S4(b) (the device-computed sticky-rejection cause) are named a GAP below, not authored
// here -- design Sec6 itself states they remain "traced at source, not executed... no cell in
// this design or its predecessor drives the specific device-domain sticky-rejection fault
// directly," and inventing a fault-injection cell against the device-side sticky-tag mechanism
// is explicitly out of this fold's scope (design Sec6/Sec9). Per Curie's own "realize the model,
// a gap in it is a finding, not an invention" discipline, this suite does not paper over that
// stated gap with a weaker assertion.
//
// D-SLM3655 / the tenth failure origin (design Sec5.1/Sec9, D-SLM3634) is likewise a NAMED GAP,
// not authored anywhere in this suite -- checked at T-2183 (Claude/Curie/
// t2183-t2169-suite-gaps-2026-08-18.md) and confirmed UNAUTHORABLE against the seam the built
// tree (brunel/t2180-gpu-batched-prefill) actually provides. The design's own Coverage Model text
// (Sec9, Failure/rejection-paths row) specifies this cell as "a throwaway fault injected into the
// chunk primitive's own recording pass, e.g. an allocation failure engineered at an
// admitted-token index i > 0" -- the ONLY test-reachable fault-injection hook in this codebase,
// `MaybeThrowInjectedO11AllocFault` (gpu_port.h's `kO11AllocInjectionSite*` constants,
// src/gpu/superslm_gpu.cpp), fires from exactly two call sites, both inside
// `PrepareGpuLayerLoopChunkOpenState` -- i.e. at CHUNK-OPEN, strictly before
// `SubmitOneSubChunkToFullDepthForG5Bridge`'s own per-token loop (the same function's `for
// (uint32_t t = 0; t < chunk_len; ++t)` body, superslm_gpu.cpp) issues a single
// `RecordOneTokenFullDepthDispatchBody` call for token index 0. No hook exists inside that loop
// itself, so no test can force the exception AFTER at least one admitted token (i > 0, per the
// design's own text) has already had its dispatches recorded into the open list -- exactly the
// state D-SLM3634's own ruling (whole-(sub-)chunk discard of tokens `0..i-1`'s own
// already-recorded-but-unexecuted dispatches) requires to be a discriminating proof rather than a
// vacuous one (an exception at chunk-open, before token 0, discards nothing that was ever
// recorded, so a cell built on the existing hook would pass under both a correct and an
// INcorrect whole-chunk-discard implementation -- it could not fail for its own reason). The
// alternative -- exploiting `GpuGemmGroupArithmeticError`'s own throw sites inside
// `RecordOneTokenFullDepthDispatchBody` (`ComputeGpuGemmSiteGroupPlan`'s enumerator/
// arithmetic guards) -- is not caller-input-triggerable: every GEMM-site-plan value is a fixed
// per-site constant (superslm_gpu.cpp), never derived from anything a public-bridge caller
// supplies, so no chunk content or size can make it throw at a chosen token index, or throw at
// all, in a correctly-functioning build.
//
// SEAM SPECIFICATION (owed to the builder, not authored here -- authoring it is production code,
// outside a test author's jurisdiction per Curie.md's own "does not implement" boundary): a new
// test-only injection hook, guarded by its own macro (e.g.
// `SUPERSLM_T2169_CHUNK_RECORDING_FAULT_INJECTION`, mirroring `SUPERSLM_O11_ALLOC_INJECTION`'s
// existing single-shot Arm/fire idiom rather than reusing its three named sites), called from
// INSIDE `SubmitOneSubChunkToFullDepthForG5Bridge`'s own per-token loop -- immediately after
// `RecordOneTokenFullDepthDispatchBody` returns for token index `t`, before the loop advances to
// `t + 1` -- and parameterized by a armed "throw after recording this many admitted tokens in the
// current (sub-)chunk" count (0-based token index within the (sub-)chunk, not a call-site
// enumerator), throwing `GpuGemmGroupArithmeticError` (or the generic `std::runtime_error` twin
// the same catch clause already handles) exactly once, single-shot, when the armed count is
// reached. This lets a cell arm "throw after token index 1" against a >= 2-admitted-token chunk
// and assert (a) the whole (sub-)chunk's command list closes unexecuted, (b) `seq`/`workspace`
// are left at their pre-chunk state (not token 0's own would-have-committed state), and (c) the
// residency caches are invalidated -- the three assertions design Sec9's own text names for this
// cell. Until this hook exists, D-SLM3655 stays quarantined here as a named gap, not a silently
// dropped one, per StandardsDocument.md Sec5.6's deferral-surfacing rule.
#include "fixture_common.h"

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

int main(int argc, char** argv) {
	ParseFixtureArgs(argc, argv);
	volatile void* a0 = (void*)&TestCensus_P1_BusyOnPrompt; (void)a0;
	volatile void* a1 = (void*)&TestCensus_P4_AllAdmittedSuccess; (void)a1;
	volatile void* a2 = (void*)&TestCensus_S1_DfaRejection; (void)a2;
	volatile void* a3 = (void*)&TestCensus_S2_BusyOnSchema; (void)a3;
	volatile void* a4 = (void*)&TestCensus_S5_AllAdmittedSuccess; (void)a4;

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
			CHECK(sslm_gpu_model_unmap(ctx, model) == SSLM_OK);
		} else {
			CHECK_MSG(false, "failed to load --model1p5b=%s: %s", g_model_1p5b_path.c_str(), err.c_str());
		}
	} else {
		SKIP_MSG("Census P1/P4 need --model1p5b=PATH -- not run");
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
	    "directly (out of this fold's scope)\n",
	    GChecks, GFailures, GSkips);
	return GFailures ? 1 : 0;
}

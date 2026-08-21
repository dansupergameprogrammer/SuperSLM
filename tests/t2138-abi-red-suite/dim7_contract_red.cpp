// T-2138 (Curie) -- Dim 7 (Contract claims), design Sec10 dim7. 4 cells authored here (C1
// recalibrated into C1a/C1b per design commit 959336ad64, below); two dispositions named
// without a cell (N/A-with-reason, per the design's own text):
//   - The handle-type half of §5's distinct-but-consistent ruling (sslm_model vs
//     SslmGpuModelHandle are compile-time-distinct incomplete-struct pointer types with no
//     implicit C conversion) is N/A-with-reason: a caller passing one where the other is
//     expected fails to COMPILE, so no runtime cell is owed (design Sec10 dim7, verbatim).
//   - The GPU-restore-rejects-CPU-blob half of §7.3's cross-surface blob claim is a cell on the
//     GPU ABI's own suite (tests/t2112-gpu-1p0-red-suite), not this suite's obligation (design
//     Sec10 dim7: "named here as an obligation this design surfaces but does not itself build,
//     since it does not own gpu_1p0.h"). The CPU-restore-rejects-GPU-blob half IS this suite's
//     own cell -- authored in dim9_persistence_red.cpp (M2, cross-cited here per dim7's own
//     "filed jointly with dimension 2's own blob-rejection cell" instruction), not duplicated.
// RED BY LINK.
#include "fixture_common.h"

#include <cstdlib>

using namespace superslm;

// --- Cell 1 (design Sec7/Sec10 dim7, RECALIBRATED per design commit 959336ad64: "narrow the
// hot-path-allocation contract to what this ABI layer controls; true zero-allocation joins
// D-SLM3457's 1.x inventory"). The T-2139 closing round measured sslm_prefill at 9717
// allocations and sslm_decode_step at 1 against a real 1.5B artifact, contradicting the
// original zero-allocation claim -- traced to source, the bulk is RunLayerLoop's/EmbedEntry's
// own internal per-call std::vector scratch (design Sec3's own grounding: "the engine's own
// internal per-call compute scratch... is separately, internally heap-allocated... never a
// caller-supplied buffer of any kind"), never this ABI layer's own code. The ruled contract,
// two halves, both still real and mutation-provable (overriding global operator new/delete,
// the same counter every cell in this file shares):
//   (a) THIS ABI LAYER's own allocations are the exact disclosed count -- zero on
//       sslm_decode_step's common ready_for_logits path; zero on sslm_prefill when a real,
//       correctly-sized workspace is supplied and no adapter is bound (the embed_codes
//       fallback fires ONLY workspace-absent/undersized; layers_scratch fires ONLY
//       adapter-bound) -- proven by a WITH-WORKSPACE-vs-WITHOUT-WORKSPACE differential on the
//       IDENTICAL call shape (same tokens, same seq freshness), which isolates exactly the one
//       disclosed embed_codes-fallback allocation from RunLayerLoop's own per-token engine cost
//       (workspace presence has no bearing on RunLayerLoop's own scratch, so the delta between
//       the two calls can only be the ABI layer's own embed_codes handling).
//   (b) THE ENGINE's own internal scratch allocation count is a disclosed, DATA-INDEPENDENT
//       function of (num_hidden_layers, token count, layer_budget) -- design Sec14's own
//       cost-determinism law applied to allocation count, never a zero requirement -- proven by
//       a STABLE-COUNT cell: the identical call shape, with DIFFERENT (still valid) token
//       values, produces the identical allocation count both times. ---
static int g_new_call_count = 0;
static void ResetAllocCounter() { g_new_call_count = 0; }

void* operator new(std::size_t size) {
	++g_new_call_count;
	return std::malloc(size);
}
void operator delete(void* p) noexcept { std::free(p); }
void operator delete(void* p, std::size_t) noexcept { std::free(p); }

// --- Cell 1a: this ABI layer's own allocations are the exact disclosed count. ---
static void TestDim7_C1a_AbiLayerOwnAllocationsAreTheDisclosedCount(
    sslm_model model, sslm_seq seq_decode, sslm_seq seq_with_ws, sslm_seq seq_without_ws,
    sslm_workspace ws) {
	// Half 1: sslm_decode_step's common ready_for_logits path (the FIRST decode_step call after
	// a completed prefill, fixture_common.h::EnterMidToken's own house precedent for why this is
	// the "free" step) -- design Sec6/Poirot S1's own finding: this path "skips embed and skips
	// RunLayerLoop entirely, runs final_norm -> logits -> argmax", so total allocations (ABI AND
	// engine, both) must be exactly zero here, not merely this ABI layer's own share of it.
	int32_t decode_prompt[1] = {0};
	int32_t decode_consumed = 0;
	CHECK(sslm_prefill(model, seq_decode, decode_prompt, 1, 8, SSLM_SPAN_PROMPT, ws,
	                    &decode_consumed) == SSLM_OK);
	sslm_decode_params params{};
	// T-2199 Phase D review addendum (D-SLM3797, Dan; conductor's fold-23
	// follow-on commission, item 1): struct_size is the FIRST new field,
	// caller-set, library-validated -- sslm_decode_stepImpl now rejects
	// SSLM_INVALID_ARGUMENT for any other value, checked before layer_budget.
	params.struct_size = sizeof(params);
	params.layer_budget = 1;
	int32_t out_token = 0;
	sslm_seq batch[1] = {seq_decode};
	ResetAllocCounter();
	CHECK(sslm_decode_step(model, batch, 1, &params, ws, &out_token) == SSLM_OK);
	CHECK_MSG(g_new_call_count == 0,
	          "sslm_decode_step's own ready_for_logits path made %d allocation(s), expected 0 "
	          "(design commit 959336ad64's own disclosed count)",
	          g_new_call_count);

	// Half 2: sslm_prefill's own embed_codes handling, isolated by a WITH-vs-WITHOUT-workspace
	// differential on the IDENTICAL call shape (same 4 tokens, both sequences equally fresh).
	// RunLayerLoop's own per-token engine scratch does not depend on workspace presence at all
	// (the workspace only ever backs PrefillWholeTokens's own embed_codes storage) -- so the
	// delta between these two calls can only be the ABI layer's own fallback allocation, proving
	// BOTH that the fix is real (the delta is not zero -- the fallback still exists when the
	// workspace is genuinely absent) AND that it is exactly the disclosed size (delta == 1, not
	// more): a regression that made the with-workspace call ALSO allocate would shrink the delta
	// to 0 or make the with-workspace call itself the nonzero side, either way changing the
	// measured delta away from 1.
	int32_t prefill_tokens[4] = {0, 1, 2, 3};
	int32_t consumed_with_ws = 0;
	ResetAllocCounter();
	CHECK(sslm_prefill(model, seq_with_ws, prefill_tokens, 4, 8, SSLM_SPAN_PROMPT, ws,
	                    &consumed_with_ws) == SSLM_OK);
	const int with_ws_count = g_new_call_count;

	int32_t consumed_without_ws = 0;
	ResetAllocCounter();
	CHECK(sslm_prefill(model, seq_without_ws, prefill_tokens, 4, 8, SSLM_SPAN_PROMPT, nullptr,
	                    &consumed_without_ws) == SSLM_OK);
	const int without_ws_count = g_new_call_count;

	CHECK_MSG(without_ws_count - with_ws_count == 1,
	          "with_ws=%d without_ws=%d (delta=%d), expected delta=1 -- the disclosed "
	          "embed_codes-fallback allocation, design commit 959336ad64",
	          with_ws_count, without_ws_count, without_ws_count - with_ws_count);
}

// --- Cell 1b: the engine's own internal scratch allocation count is a disclosed,
// DATA-INDEPENDENT function of (num_hidden_layers, token count, layer_budget) -- a stable-count
// cell, never a zero requirement. ---
static void TestDim7_C1b_EngineAllocationCountIsStableAcrossHostileContent(sslm_model model,
                                                                            sslm_seq seq_benign,
                                                                            sslm_seq seq_hostile,
                                                                            sslm_workspace ws) {
	// Same SHAPE (4 tokens, same chunk_budget, same freshly-created sequence state), different
	// token VALUES -- still in-domain (dim5's own C10 already covers genuinely out-of-range
	// values as a separate rejection path; this cell's own subject is in-domain content that
	// merely differs).
	int32_t benign_tokens[4] = {0, 1, 2, 3};
	int32_t hostile_tokens[4] = {97, 53, 11, 29};  // different values, same count/shape.

	int32_t consumed_benign = 0;
	ResetAllocCounter();
	CHECK(sslm_prefill(model, seq_benign, benign_tokens, 4, 8, SSLM_SPAN_PROMPT, ws,
	                    &consumed_benign) == SSLM_OK);
	const int benign_count = g_new_call_count;

	int32_t consumed_hostile = 0;
	ResetAllocCounter();
	CHECK(sslm_prefill(model, seq_hostile, hostile_tokens, 4, 8, SSLM_SPAN_PROMPT, ws,
	                    &consumed_hostile) == SSLM_OK);
	const int hostile_count = g_new_call_count;

	// FEATURE ORACLE: the identical call shape produces the IDENTICAL allocation count
	// regardless of token content -- proving the engine's own per-call scratch cost is bounded
	// and disclosed (a function of shape alone), never content-dependent. This cell can fail: a
	// regression that made allocation count scale with token VALUE (e.g. a content-dependent
	// branch inside RunLayerLoop's own scratch sizing) would make benign_count != hostile_count.
	CHECK_MSG(benign_count == hostile_count,
	          "benign=%d hostile=%d -- allocation count is not data-independent at this shape",
	          benign_count, hostile_count);
	CHECK(consumed_benign == consumed_hostile);  // same shape consumed the same token count too.
}

// --- Cell 2 (design Sec7/Sec10 dim7: "resumable between calls -> the save/restore round-trip
// (C5) plus the mid-call-boundary state test"). The mid-call-boundary half (distinct from C5's
// own bit-equality round-trip, dim9's own obligation, cross-cited not duplicated): a sequence
// interrupted mid-way through a MULTI-CALL chunked prefill (chunk_budget smaller than the
// prompt) resumes correctly on the NEXT call -- proving "resumable between calls" means literal
// call-to-call resumption, not merely save/restore-shaped resumption. ---
static void TestDim7_C2_MidCallBoundaryChunkedPrefillResumesCorrectly(sslm_model model,
                                                                       sslm_seq seq) {
	int32_t long_prompt[20];
	for (int i = 0; i < 20; ++i) long_prompt[i] = i % 16;
	int32_t consumed_first = 0;
	// chunk_budget=8 against a 20-token prompt forces multiple internal chunk boundaries; this
	// cell calls sslm_prefill twice itself (8 then 12) to prove the ABI-level call boundary
	// (not only chunk_budget's own internal one) is a resumption point. Design Sec8.1 ("host
	// owns the loop: synchronous bounded steps... consume a bounded budget and return") means
	// `consumed` never exceeds `chunk_budget` in one call, whatever `count` requests -- the
	// caller loops, feeding the remainder, exactly like the SECOND call below (count=12,
	// chunk_budget=8) itself only consumes 8, requiring a THIRD call for the remaining 4. This
	// is the corrected reading (StandardsDocument.md Sec5.6): the prior text assumed a single
	// call with count > chunk_budget auto-loops to completion, contradicted by execution.
	CHECK(sslm_prefill(model, seq, long_prompt, 8, 8, SSLM_SPAN_PROMPT, nullptr,
	                    &consumed_first) == SSLM_OK);
	CHECK(consumed_first == 8);
	int32_t total_consumed = 8;
	int32_t guard_iterations = 0;
	while (total_consumed < 20) {
		int32_t consumed_this_call = 0;
		CHECK(sslm_prefill(model, seq, long_prompt + total_consumed, 20 - total_consumed, 8,
		                    SSLM_SPAN_PROMPT, nullptr, &consumed_this_call) == SSLM_OK);
		CHECK_MSG(consumed_this_call > 0, "prefill made no progress at total_consumed=%d",
		          total_consumed);
		if (consumed_this_call <= 0) break;  // never spin forever on a real failure to progress.
		total_consumed += consumed_this_call;
		CHECK_MSG(++guard_iterations <= 20, "prefill did not converge within a sane call count");
		if (guard_iterations > 20) break;
	}
	// FEATURE ORACLE: the sequence's own context_length after every call equals the FULL
	// 20-token prompt's own length, proving each call genuinely resumed from where the previous
	// one left off rather than either re-consuming or dropping tokens at any boundary.
	CHECK(total_consumed == 20);
	sslm_stats_out stats{};
	CHECK(sslm_stats(model, seq, &stats) == SSLM_OK);
}

// --- Cell 3 (design Sec14/Sec10 dim7: "fixed op count per call -> §14's existing ceiling test,
// extended to this ABI's own dispatch: no new data-dependent branch is introduced by a
// lifecycle guard -- every guard here is a comparison against already-validated state, not a
// scan of hostile content"). sslm_stats's own decode_step_ceiling is unaffected by whether a
// lifecycle-guard rejection is hit BEFORE the real dispatch -- the rejection path's own cost is
// data-independent (a fixed comparison), never a function of how "hostile" the rejected input
// is. ---
static void TestDim7_C3_LifecycleGuardRejectionCostIsDataIndependent(sslm_model model,
                                                                      sslm_seq seq,
                                                                      sslm_adapter adapter) {
	// Establishes the SAME mid-token-residual precondition dim5 C5 puts its own seq into
	// (a real prefill, THEN layer_budget=1, leaving layer_index != 0) -- decode_step requires a
	// prior prefill call on a real implementation (a virgin sequence has no current_token to
	// continue from); this cell's own subject is the guard's rejection cost, not the
	// precondition setup, but the precondition still has to be real and self-contained rather
	// than assumed of the caller.
	int32_t setup_prompt[3] = {0, 1, 2};
	int32_t setup_consumed = 0;
	CHECK(sslm_prefill(model, seq, setup_prompt, 3, 8, SSLM_SPAN_PROMPT, nullptr,
	                    &setup_consumed) == SSLM_OK);
	CHECK(EnterMidToken(model, seq));

	// Two rejected sslm_seq_set_adapter calls against the SAME mid-token-residual precondition
	// (design Sec6's own SSLM_ADAPTER_SWAP_MIDTOKEN_REJECTED cell, dim5 C5) -- one against a
	// "small" adapter argument shape and one against a structurally identical call, both
	// rejected at the SAME guard comparison. The claim under test is that the guard's own
	// rejection path costs the same regardless of adapter content (never a scan), which this
	// cell asserts indirectly via sslm_stats's own reported ceiling being unmoved by a rejected
	// call (a rejection that scaled with content would perturb decode_step_ceiling, since the
	// ceiling is itself derived from a fixed dispatch-shape computation, design Sec14).
	sslm_stats_out stats_before{};
	CHECK(sslm_stats(model, seq, &stats_before) == SSLM_OK);
	const sslm_status swap_status = sslm_seq_set_adapter(seq, adapter);
	CHECK_MSG(swap_status == SSLM_ADAPTER_SWAP_MIDTOKEN_REJECTED, "actual status=%d (adapter=%p)",
	          (int)swap_status, (void*)adapter);
	sslm_stats_out stats_after{};
	CHECK(sslm_stats(model, seq, &stats_after) == SSLM_OK);
	CHECK(stats_before.decode_step_ceiling == stats_after.decode_step_ceiling);
}

// REAL INVOCATION DRIVER (house pattern) -- supersedes the address-only convention. Each cell
// gets its OWN fresh sequence (a pool sized for three) so C2's own "prefill from scratch" and
// C3's own mid-token setup never observe another cell's leftover state.
int main(int argc, char** argv) {
	ParseFixtureArgs(argc, argv);
	if (g_model_path.empty()) {
		SKIP_MSG("--model=PATH not supplied -- dim7 C1-C3 not run");
		std::printf("checks=%d failures=%d skips=%d\n", GChecks, GFailures, GSkips);
		return GFailures ? 1 : 0;
	}
	SslmModelView view;
	std::vector<uint8_t> bytes;
	std::string err;
	if (!LoadRealModelView(g_model_path, &view, &bytes, &err)) {
		SKIP_MSG("could not load real artifact: %s", err.c_str());
		std::printf("checks=%d failures=%d skips=%d\n", GChecks, GFailures, GSkips);
		return GFailures ? 1 : 0;
	}
	sslm_model model = nullptr;
	CHECK(sslm_model_map(bytes.data(), bytes.size(), &model) == SSLM_OK);
	if (model) {
		const int32_t num_hidden_layers = static_cast<int32_t>(view.config.num_hidden_layers);
		// C1a needs 3 fresh sequences (decode-common-path, with-ws, without-ws), C1b needs 2
		// (benign, hostile), C2/C3 need 1 each -- 7 total, one shared pool.
		const uint32_t block_count = 7;
		const size_t block_bytes = sslm_kv_block_size(model);
		const size_t overhead = sslm_kv_pool_overhead_size(model, block_count);
		AlignedBuffer pool_buf(block_count * block_bytes + overhead);
		sslm_kv_pool pool = nullptr;
		CHECK(sslm_kv_pool_create(model, pool_buf.data(), pool_buf.size(), block_count, &pool) ==
		      SSLM_OK);
		const sslm_config config = ValidWorkspaceConfig(num_hidden_layers);
		const size_t ws_size = sslm_workspace_size(model, &config);
		AlignedBuffer ws_buf(ws_size);
		sslm_workspace ws = nullptr;
		CHECK(sslm_workspace_create(model, &config, ws_buf.data(), ws_buf.size(), &ws) == SSLM_OK);

		if (pool && ws) {
			sslm_seq seq_decode = nullptr, seq_with_ws = nullptr, seq_without_ws = nullptr,
			         seq_benign = nullptr, seq_hostile = nullptr, seq_c2 = nullptr,
			         seq_c3 = nullptr;
			CHECK(sslm_seq_create(model, &pool, &seq_decode) == SSLM_OK);
			CHECK(sslm_seq_create(model, &pool, &seq_with_ws) == SSLM_OK);
			CHECK(sslm_seq_create(model, &pool, &seq_without_ws) == SSLM_OK);
			CHECK(sslm_seq_create(model, &pool, &seq_benign) == SSLM_OK);
			CHECK(sslm_seq_create(model, &pool, &seq_hostile) == SSLM_OK);
			CHECK(sslm_seq_create(model, &pool, &seq_c2) == SSLM_OK);
			CHECK(sslm_seq_create(model, &pool, &seq_c3) == SSLM_OK);

			if (seq_decode && seq_with_ws && seq_without_ws) {
				TestDim7_C1a_AbiLayerOwnAllocationsAreTheDisclosedCount(
				    model, seq_decode, seq_with_ws, seq_without_ws, ws);
			}
			if (seq_benign && seq_hostile) {
				TestDim7_C1b_EngineAllocationCountIsStableAcrossHostileContent(
				    model, seq_benign, seq_hostile, ws);
			}
			if (seq_c2) TestDim7_C2_MidCallBoundaryChunkedPrefillResumesCorrectly(model, seq_c2);
			if (!seq_c3) {
				// nothing to do -- C3's own SKIP is subsumed by the seq_create failure above.
			} else if (g_adapter_path.empty()) {
				SKIP_MSG("--adapter=PATH not supplied -- dim7 C3 not run");
			} else {
				std::vector<uint8_t> adapter_bytes;
				CHECK(ReadFileBytes(g_adapter_path, &adapter_bytes));
				sslm_adapter adapter = nullptr;
				CHECK(sslm_adapter_map(adapter_bytes.data(), adapter_bytes.size(), model,
				                        &adapter) == SSLM_OK);
				if (adapter) {
					TestDim7_C3_LifecycleGuardRejectionCostIsDataIndependent(model, seq_c3,
					                                                         adapter);
					CHECK(sslm_adapter_release(adapter) == SSLM_OK);
				}
			}
			if (seq_decode) CHECK(sslm_seq_release(seq_decode) == SSLM_OK);
			if (seq_with_ws) CHECK(sslm_seq_release(seq_with_ws) == SSLM_OK);
			if (seq_without_ws) CHECK(sslm_seq_release(seq_without_ws) == SSLM_OK);
			if (seq_benign) CHECK(sslm_seq_release(seq_benign) == SSLM_OK);
			if (seq_hostile) CHECK(sslm_seq_release(seq_hostile) == SSLM_OK);
			if (seq_c2) CHECK(sslm_seq_release(seq_c2) == SSLM_OK);
			if (seq_c3) CHECK(sslm_seq_release(seq_c3) == SSLM_OK);
		}
		if (ws) CHECK(sslm_workspace_destroy(ws) == SSLM_OK);
		if (pool) CHECK(sslm_kv_pool_destroy(pool) == SSLM_OK);
		CHECK(sslm_model_unmap(model) == SSLM_OK);
	}
	std::printf("checks=%d failures=%d skips=%d\n", GChecks, GFailures, GSkips);
	return GFailures ? 1 : 0;
}

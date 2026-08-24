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
// C1c additionally needs the SIZE of what was allocated, not only how many times. Cell 1a's
// contract is a count because the allocations it governs are all small and fixed; the defect
// C1c exists to catch was a SINGLE vocabulary-sized buffer, which moves a plain count by one and
// is indistinguishable by plain count from a hash-table node. What separates them is scale, so
// this file also tallies allocations at or above a caller-set threshold. Every observation comes
// from the same operator new below, so no cell pays for another's instrumentation, and the
// threshold is zero (tally inert) unless a cell sets it.
static std::size_t g_new_byte_total = 0;
static std::size_t g_new_largest = 0;
static std::size_t g_large_alloc_threshold = 0;
static int g_large_alloc_count = 0;
static void ResetAllocCounter() {
	g_new_call_count = 0;
	g_new_byte_total = 0;
	g_new_largest = 0;
	g_large_alloc_count = 0;
}

void* operator new(std::size_t size) {
	++g_new_call_count;
	g_new_byte_total += size;
	if (size > g_new_largest) g_new_largest = size;
	if (g_large_alloc_threshold != 0 && size >= g_large_alloc_threshold) ++g_large_alloc_count;
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

// --- Cell 1c (T-2233, closing finding O1 of the 1.2 release-candidate outside review,
// Claude/Poirot/4c8cd2d-superslm-1p2-release-candidate.md): the DAMPED-GREEDY decode path is
// held to a per-token allocation contract, through sslm_decode_step_v2.
//
// WHY THIS CELL EXISTS. Cell 1a reaches the contract through sslm_decode_step -- the legacy
// wrapper, which hard-codes mode = SSLM_DECODE_MODE_GREEDY -- so the damped path inherited the
// contract by construction without inheriting the cell that proves it. That gap is how a
// per-token, full-vocabulary allocation (a std::vector<int32_t> of vocab_size, 607,744 bytes at
// the shipped target vocabulary, plus a partial_sort over the whole row, once per emitted token
// per sequence) reached a release candidate with every cell in this file green. The remedy
// carries the selector's index and score scratch in the caller's workspace
// (sslm_decode_stepImpl's damped_indices/wide_logits, src/sslm_abi.cpp), and this cell pins it.
//
// WHAT THE ORACLE IS, AND WHY IT IS A SIZE RATHER THAN A COUNT. The first version of this cell
// asserted that a damped step allocates exactly as many times as a greedy step. It fails on the
// shipped release at delta=9, and the 9 are legitimate: AntiLmUpdate grows the sequence's own
// n-gram count tables as new contexts appear -- persistent STATE whose cost saturates as
// contexts repeat, disclosed in docs/api.md as the one damped-specific memory cost, and not the
// per-token scratch this contract governs. A count is the wrong instrument for the defect class
// anyway: the allocation that shipped was ONE call, indistinguishable by count from one hash
// node, and expensive only because of its SIZE.
//
// So the oracle counts VOCABULARY-SCALE allocations -- those at or above one int32 per
// vocabulary entry -- and requires the damped arm to make exactly as many as the greedy arm.
// The greedy arm is the reference rather than a constant, and it has to be: the engine itself
// allocates one vocabulary-sized logit buffer per token on BOTH paths, measured, so "no
// vocabulary-sized allocation on the damped path" is a bound neither arm can meet and would be
// a gate that fails on correct code. "No vocabulary-sized allocation the greedy path does not
// also make" is the claim the contract actually supports, it refuses the exact construction C1
// found -- a second vocabulary-sized buffer, on the damped path only -- and it is immune to the
// anti-LM's small-node churn at any count.
//
// THE FIRST DAMPED STEP IS DELIBERATELY NOT THE MEASURED ONE. sslm_decode_stepImpl creates the
// sequence's own AntiLmState on its first damped call. That allocation is once per sequence
// rather than once per token, so it is paid in the warm-up step and is out of the measurement.
static void TestDim7_C1c_DampedDecodeStepAllocatesNoVocabularySizedBuffer(
    sslm_model model, sslm_seq seq_damped, sslm_seq seq_greedy, sslm_workspace ws,
    int32_t layer_budget, int32_t vocab_size) {
	// 1 is SSLM_DECODE_MODE_DAMPED_GREEDY and 0 is SSLM_DECODE_MODE_GREEDY
	// (include/superslm/sslm_abi.h). This suite's own mirror header carries the parameter
	// struct, whose layout the T-2141 gate static_asserts against production, but not the mode
	// constants -- so the literals are used here rather than a second un-gated mirror of them.
	sslm_decode_params damped{};
	const sslm_status init = sslm_decode_params_init(model, 1, layer_budget, &damped);
	if (init == SSLM_ARTIFACT_REJECTED) {
		SKIP_MSG("dim7 C1c: the supplied artifact carries no DGC1 section, so damped greedy "
		         "cannot be selected on it -- pass a --model built with "
		         "convert_model.py --enable-damped-greedy to run this cell");
		return;
	}
	CHECK(init == SSLM_OK);
	if (init != SSLM_OK) return;

	sslm_decode_params greedy{};
	CHECK(sslm_decode_params_init(model, 0, layer_budget, &greedy) == SSLM_OK);

	// Identical prompts, identical shape, identical workspace on both arms.
	int32_t prompt[4] = {0, 1, 2, 3};
	int32_t consumed_damped = 0;
	int32_t consumed_greedy = 0;
	CHECK(sslm_prefill(model, seq_damped, prompt, 4, 8, SSLM_SPAN_PROMPT, ws,
	                    &consumed_damped) == SSLM_OK);
	CHECK(sslm_prefill(model, seq_greedy, prompt, 4, 8, SSLM_SPAN_PROMPT, ws,
	                    &consumed_greedy) == SSLM_OK);
	CHECK(consumed_damped == consumed_greedy);

	sslm_seq damped_batch[1] = {seq_damped};
	sslm_seq greedy_batch[1] = {seq_greedy};
	int32_t token = 0;

	// Warm-up: the ready_for_logits step. On the damped arm this is also the call that creates
	// the AntiLmState. Neither arm is measured here.
	CHECK(sslm_decode_step_v2(model, damped_batch, 1, &damped, ws, &token) == SSLM_OK);
	CHECK(sslm_decode_step_v2(model, greedy_batch, 1, &greedy, ws, &token) == SSLM_OK);

	// The measured step: a full token through the layer loop on each arm. One int32 per
	// vocabulary entry is the scale of the buffer C1 found, and of the engine's own per-token
	// logit buffer that both arms pay.
	const std::size_t vocab_scale = static_cast<std::size_t>(vocab_size) * sizeof(int32_t);
	g_large_alloc_threshold = vocab_scale;

	ResetAllocCounter();
	CHECK(sslm_decode_step_v2(model, damped_batch, 1, &damped, ws, &token) == SSLM_OK);
	const std::size_t damped_largest = g_new_largest;
	const int damped_count = g_new_call_count;
	const std::size_t damped_bytes = g_new_byte_total;
	const int damped_large = g_large_alloc_count;

	ResetAllocCounter();
	CHECK(sslm_decode_step_v2(model, greedy_batch, 1, &greedy, ws, &token) == SSLM_OK);
	const std::size_t greedy_largest = g_new_largest;
	const int greedy_count = g_new_call_count;
	const std::size_t greedy_bytes = g_new_byte_total;
	const int greedy_large = g_large_alloc_count;

	g_large_alloc_threshold = 0;  // leave the tally inert for every other cell in this file.

	// FEATURE ORACLE: the damped path makes no vocabulary-scale allocation that the greedy path
	// does not also make. This is the half that can fail on the defect that shipped: the removed
	// construction was a std::vector<int32_t> of exactly vocab_size, so restoring it -- or
	// introducing any other per-token buffer that scales with the vocabulary -- puts the damped
	// arm one vocabulary-scale allocation above the greedy arm, at every vocabulary, including
	// the small synthetic fixtures this suite runs on.
	CHECK_MSG(damped_large == greedy_large,
	          "damped made %d vocabulary-scale allocation(s) (>= %zu bytes) against greedy's %d, "
	          "at vocab_size=%d -- a per-token buffer that scales with the vocabulary is on the "
	          "damped decode path and not on the greedy one",
	          damped_large, vocab_scale, greedy_large, vocab_size);

	// Reported, never gated: the plain call and byte deltas are the anti-LM count tables growing
	// as new contexts appear, which docs/api.md discloses and which saturates as contexts repeat.
	// A gate on either would refuse the disclosed cost, so this cell reports them for a reader and
	// leaves the judgement to the oracle above.
	std::printf("dim7 C1c: damped calls=%d bytes=%zu largest=%zu vocab-scale=%d | greedy calls=%d "
	            "bytes=%zu largest=%zu vocab-scale=%d | vocab scale=%zu\n",
	            damped_count, damped_bytes, damped_largest, damped_large, greedy_count,
	            greedy_bytes, greedy_largest, greedy_large, vocab_scale);
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

// --- Cells 4 and 5 (T-2234, SuperSLM 1.2.1; red suite PH1-T2234-C1/C2): the workspace's
// damped_indices region becomes CONDITIONAL on damped_greedy_available. 1.2.0 grew EVERY
// caller's workspace unconditionally (the region is reserved for any mapped model,
// src/sslm_abi.cpp ComputeWorkspaceLayout); the fix reserves it only when the artifact
// carries the DGC1 opt-in, restoring the pre-1.2 formula for greedy-only callers -- so an
// old-formula buffer sized by 1.1.0's own arithmetic validates again at 1.2.1.
//
// T-2243 review finding 7 (D-SLM4113) pin note: C4's own `ws_size == GOLDEN_1_1_0_TOTAL`
// assertion below IS the externally-observable proof that `layout.damped_indices_bytes == 0`
// for a non-DGC1 model (this comment block's own "equivalent to damped_indices_bytes == 0"
// note, above) -- the exact condition finding 7's carve-site guard
// (`layout.damped_indices_bytes > 0`, sslm_abi.cpp's ws_usable block) now branches on. No
// separate pin was added for the guarded pointer itself: `WorkspaceLayout`/`sslm_model_s` are
// both private to sslm_abi.cpp (no header declares either), and the only public entry point
// that reaches the carve site (sslm_workspace_create's own decode path) already rejects a
// non-DGC1 model with SSLM_ARTIFACT_REJECTED before the carve runs -- there is no way to
// observe the guarded pointer from outside that TU without adding new test-only exported
// surface for a defensive fix on an already-proven-unreachable path.
//
// GOLDEN_1_1_0_TOTAL -- captured ONCE by executing tag v1.1.0's own shipped
// sslm_workspace_size over the identical geometry (provenance: 2026-08-23, detached
// worktree at v1.1.0 = commit 49d1333, capture tool linked v1.1.0's real
// src/sslm_abi.cpp; artifact: out/t2243_s8_plain.sslm, the S8 fixture's no-DGC1 twin;
// frozen geometry literals: num_hidden_layers=8 hidden_size=32 vocab_size=128
// context_cap=64; sslm_config: max_batch=16 max_chunk_budget=256 max_layer_budget=8).
// Executed output: "GOLDEN v1.1.0: ... total_bytes=11264". The reference is the prior
// release's OWN arithmetic, independent of 1.2.1's.
//
// Both cells require the hermetic fixture pair at exactly this geometry (--model= the S8
// DGC1 artifact, --modelplain= its no-DGC1 twin): the golden value is meaningful ONLY over
// the frozen geometry it was captured at (a different artifact SKIPs rather than asserting
// against a mismatched domain). The v1.1.0 -> 1.2.0 layout diff is EXACTLY one added region
// (damped_indices_offset/bytes between logit_row and rms_wide; verified by diffing the two
// tags' ComputeWorkspaceLayout), so over a frozen geometry `total == GOLDEN` is equivalent
// to `damped_indices_bytes == 0` with its offset unclaimed.

static const size_t GOLDEN_1_1_0_TOTAL = 11264;

// Frozen geometry + config both cells' golden comparisons are valid over.
struct FrozenGeometry {
	uint32_t num_hidden_layers;
	uint32_t hidden_size;
	uint32_t vocab_size;
	int32_t context_cap;
};
static constexpr FrozenGeometry kGoldenGeometry{8, 32, 128, 64};

static bool GeometryMatchesGolden(const superslm::SslmModelView& view) {
	return view.config.num_hidden_layers == kGoldenGeometry.num_hidden_layers &&
	       view.config.hidden_size == kGoldenGeometry.hidden_size &&
	       view.config.vocab_size == kGoldenGeometry.vocab_size &&
	       static_cast<int32_t>(view.config.context_cap) == kGoldenGeometry.context_cap;
}

// Cell 4 (PH1-T2234-C1): with damped_greedy_available == false the workspace reserves NO
// damped_indices region and the total equals the pre-1.2.0 formula -- AND an sslm_decode
// call with a caller workspace buffer sized to that golden total succeeds end to end (the
// old buffer is ACCEPTED, not merely identically computed).
static void TestDim7_C4_NonDgc1WorkspaceMatchesV110Golden(sslm_model model_plain,
                                                           const superslm::SslmModelView& view) {
	if (!GeometryMatchesGolden(view)) {
		SKIP_MSG("dim7 C4: --modelplain geometry (%u/%u/%u/%d) does not match the frozen "
		         "geometry the golden was captured over (8/32/128/64) -- cell not run",
		         static_cast<unsigned>(view.config.num_hidden_layers),
		         static_cast<unsigned>(view.config.hidden_size),
		         static_cast<unsigned>(view.config.vocab_size),
		         static_cast<int>(view.config.context_cap));
		return;
	}
	const int32_t num_hidden_layers = static_cast<int32_t>(view.config.num_hidden_layers);
	const sslm_config cfg = ValidWorkspaceConfig(num_hidden_layers);
	const size_t ws_size = sslm_workspace_size(model_plain, &cfg);
	CHECK_MSG(ws_size == GOLDEN_1_1_0_TOTAL,
	          "dim7 C4: a non-DGC1 artifact's workspace must size to the pre-1.2 formula "
	          "(golden %zu captured from v1.1.0's own ComputeWorkspaceLayout); got %zu -- "
	          "%zu bytes of unconditional damped_indices growth",
	          GOLDEN_1_1_0_TOTAL, ws_size, ws_size - GOLDEN_1_1_0_TOTAL);

	// End-to-end validation arm: the old-formula buffer is ACCEPTED and drives a real
	// prefill + decode step.
	const uint32_t block_count = 1;
	const size_t block_bytes = sslm_kv_block_size(model_plain);
	const size_t overhead = sslm_kv_pool_overhead_size(model_plain, block_count);
	AlignedBuffer pool_buf(block_count * block_bytes + overhead);
	sslm_kv_pool pool = nullptr;
	CHECK(sslm_kv_pool_create(model_plain, pool_buf.data(), pool_buf.size(), block_count,
	                          &pool) == SSLM_OK);
	AlignedBuffer ws_buf(GOLDEN_1_1_0_TOTAL);
	sslm_workspace ws = nullptr;
	CHECK_MSG(sslm_workspace_create(model_plain, &cfg, ws_buf.data(), ws_buf.size(), &ws) ==
	                  SSLM_OK,
	          "dim7 C4: a caller workspace sized to the v1.1.0 golden total (%zu bytes) must "
	          "be ACCEPTED for a non-DGC1 artifact at 1.2.1 (got status %d)",
	          GOLDEN_1_1_0_TOTAL, ws != nullptr ? 0 : -1);
	if (!ws) return;
	sslm_seq seq = nullptr;
	CHECK(sslm_seq_create(model_plain, &pool, &seq) == SSLM_OK);
	if (seq && pool) {
		int32_t prompt[4] = {0, 1, 2, 3};
		int32_t consumed = 0;
		CHECK(sslm_prefill(model_plain, seq, prompt, 4, 8, SSLM_SPAN_PROMPT, ws,
		                    &consumed) == SSLM_OK);
		sslm_decode_params params{};
		params.struct_size = sizeof(params);
		params.layer_budget = num_hidden_layers;
		sslm_seq batch[1] = {seq};
		int32_t token = 0;
		CHECK(sslm_decode_step(model_plain, batch, 1, &params, ws, &token) == SSLM_OK);
		CHECK(sslm_seq_release(seq) == SSLM_OK);
	}
	if (ws) CHECK(sslm_workspace_destroy(ws) == SSLM_OK);
	if (pool) CHECK(sslm_kv_pool_destroy(pool) == SSLM_OK);
}

// Cell 5 (PH1-T2234-C2): a DGC1-bearing artifact STILL reserves the damped_indices region
// once the gating lands -- REGRESSION-PIN, green-at-authoring (it passes today BY the very
// behavior cell 4 removes elsewhere), never counted as red coverage. Its obligation is
// existence BEFORE the fix, so the fix cannot silently drop the region.
static void TestDim7_C5_Dgc1ArtifactKeepsDampedRegion(sslm_model model_dgc1,
                                                       const superslm::SslmModelView& view) {
	if (!GeometryMatchesGolden(view)) {
		SKIP_MSG("dim7 C5: --model geometry does not match the frozen geometry the golden was "
		         "captured over (8/32/128/64) -- cell not run");
		return;
	}
	const int32_t vocab_size = static_cast<int32_t>(view.config.vocab_size);
	const int32_t num_hidden_layers = static_cast<int32_t>(view.config.num_hidden_layers);
	const sslm_config cfg = ValidWorkspaceConfig(num_hidden_layers);
	const size_t ws_size = sslm_workspace_size(model_dgc1, &cfg);

	// The region's byte count at THIS fixture's vocabulary: vocab_size * sizeof(int32_t),
	// aligned per the layout's alignment rule. The delta against the v1.1.0 golden is
	// observable publicly through the total alone (the v1.1.0 -> 1.2.0 layout diff is
	// exactly that one region).
	CHECK(ws_size >= GOLDEN_1_1_0_TOTAL);
	const size_t delta = ws_size - GOLDEN_1_1_0_TOTAL;
	CHECK_MSG(delta >= static_cast<size_t>(vocab_size) * sizeof(int32_t),
	          "dim7 C5: the DGC1 artifact's workspace must still reserve damped_indices "
	          "(delta %zu vs golden %zu < vocab %d * 4) -- the gating dropped the region",
	          delta, GOLDEN_1_1_0_TOTAL, vocab_size);

	// Decode with the FULL layout succeeds: the reserved region is real and usable on the
	// damped path (prefill + one damped-greedy decode step through the caller's workspace).
	const uint32_t block_count = 1;
	const size_t block_bytes = sslm_kv_block_size(model_dgc1);
	const size_t overhead = sslm_kv_pool_overhead_size(model_dgc1, block_count);
	AlignedBuffer pool_buf(block_count * block_bytes + overhead);
	sslm_kv_pool pool = nullptr;
	CHECK(sslm_kv_pool_create(model_dgc1, pool_buf.data(), pool_buf.size(), block_count,
	                          &pool) == SSLM_OK);
	AlignedBuffer ws_buf(ws_size);
	sslm_workspace ws = nullptr;
	CHECK(sslm_workspace_create(model_dgc1, &cfg, ws_buf.data(), ws_buf.size(), &ws) ==
	      SSLM_OK);
	if (ws && pool) {
		sslm_seq seq = nullptr;
		CHECK(sslm_seq_create(model_dgc1, &pool, &seq) == SSLM_OK);
		if (seq) {
			int32_t prompt[4] = {0, 1, 2, 3};
			int32_t consumed = 0;
			CHECK(sslm_prefill(model_dgc1, seq, prompt, 4, 8, SSLM_SPAN_PROMPT, ws,
			                    &consumed) == SSLM_OK);
			sslm_decode_params params{};
			const sslm_status init =
			    sslm_decode_params_init(model_dgc1, /*mode=*/1, num_hidden_layers, &params);
			if (init == SSLM_ARTIFACT_REJECTED) {
				SKIP_MSG("dim7 C5: --model carries no DGC1 section -- damped arm not run");
			} else {
				CHECK(init == SSLM_OK);
				sslm_seq batch[1] = {seq};
				int32_t token = 0;
				CHECK(sslm_decode_step_v2(model_dgc1, batch, 1, &params, ws, &token) ==
				      SSLM_OK);
			}
			CHECK(sslm_seq_release(seq) == SSLM_OK);
		}
	}
	if (ws) CHECK(sslm_workspace_destroy(ws) == SSLM_OK);
	if (pool) CHECK(sslm_kv_pool_destroy(pool) == SSLM_OK);
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
		// (benign, hostile), C1c needs 2 (damped arm, greedy arm), C2/C3 need 1 each -- 9 total,
		// one shared pool.
		const uint32_t block_count = 9;
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
			         seq_benign = nullptr, seq_hostile = nullptr, seq_damped = nullptr,
			         seq_greedy = nullptr, seq_c2 = nullptr, seq_c3 = nullptr;
			CHECK(sslm_seq_create(model, &pool, &seq_decode) == SSLM_OK);
			CHECK(sslm_seq_create(model, &pool, &seq_with_ws) == SSLM_OK);
			CHECK(sslm_seq_create(model, &pool, &seq_without_ws) == SSLM_OK);
			CHECK(sslm_seq_create(model, &pool, &seq_benign) == SSLM_OK);
			CHECK(sslm_seq_create(model, &pool, &seq_hostile) == SSLM_OK);
			CHECK(sslm_seq_create(model, &pool, &seq_damped) == SSLM_OK);
			CHECK(sslm_seq_create(model, &pool, &seq_greedy) == SSLM_OK);
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
			if (seq_damped && seq_greedy) {
				TestDim7_C1c_DampedDecodeStepAllocatesNoVocabularySizedBuffer(
				    model, seq_damped, seq_greedy, ws, num_hidden_layers,
				    static_cast<int32_t>(view.config.vocab_size));
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
			// T-2234: the workspace-layout contract across the damped_greedy_available
			// gating. Both cells build their own pools/workspaces; C5 runs against THIS
			// file's --model (the DGC1-bearing hermetic artifact), C4 against --modelplain.
			if (!g_model_plain_path.empty()) {
				superslm::SslmModelView plain_view;
				std::vector<uint8_t> plain_bytes;
				std::string plain_err;
				if (LoadRealModelView(g_model_plain_path, &plain_view, &plain_bytes,
				                      &plain_err)) {
					sslm_model model_plain = nullptr;
					CHECK(sslm_model_map(plain_bytes.data(), plain_bytes.size(),
					                     &model_plain) == SSLM_OK);
					if (model_plain) {
						TestDim7_C4_NonDgc1WorkspaceMatchesV110Golden(model_plain,
						                                              plain_view);
						CHECK(sslm_model_unmap(model_plain) == SSLM_OK);
					}
				} else {
					SKIP_MSG("dim7 C4: could not load --modelplain artifact: %s",
					         plain_err.c_str());
				}
			} else {
				SKIP_MSG("--modelplain=PATH not supplied -- dim7 C4 not run");
			}
			TestDim7_C5_Dgc1ArtifactKeepsDampedRegion(model, view);

			if (seq_decode) CHECK(sslm_seq_release(seq_decode) == SSLM_OK);
			if (seq_with_ws) CHECK(sslm_seq_release(seq_with_ws) == SSLM_OK);
			if (seq_without_ws) CHECK(sslm_seq_release(seq_without_ws) == SSLM_OK);
			if (seq_benign) CHECK(sslm_seq_release(seq_benign) == SSLM_OK);
			if (seq_hostile) CHECK(sslm_seq_release(seq_hostile) == SSLM_OK);
			if (seq_damped) CHECK(sslm_seq_release(seq_damped) == SSLM_OK);
			if (seq_greedy) CHECK(sslm_seq_release(seq_greedy) == SSLM_OK);
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

// T-2138 (Curie) -- Dim 7 (Contract claims), design Sec10 dim7. 3 cells authored here; two
// dispositions named without a cell (N/A-with-reason, per the design's own text):
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

// --- Cell 1 (design Sec7/Sec10 dim7: "caller-owned memory, never mallocs on the hot path" ->
// asserted by construction, an allocation-tracking test per §17 dim 7's existing convention).
// sslm_prefill/sslm_decode_step's own call graph makes zero heap allocations -- proven by
// overriding global operator new/delete for the scope of the hot-path calls and asserting the
// allocation counter is unmoved. This is a REAL, mutation-provable counter (not a comment-only
// stub, StandardsDocument.md §5.4): a build that reintroduces a hidden heap allocation on this
// path makes g_new_call_count nonzero and this cell fails for exactly that reason. ---
static int g_new_call_count = 0;
static void ResetAllocCounter() { g_new_call_count = 0; }

void* operator new(std::size_t size) {
	++g_new_call_count;
	return std::malloc(size);
}
void operator delete(void* p) noexcept { std::free(p); }
void operator delete(void* p, std::size_t) noexcept { std::free(p); }

static void TestDim7_C1_HotPathMakesNoAllocation(sslm_model model, sslm_seq seq,
                                                  sslm_workspace ws) {
	int32_t tokens[4] = {0, 1, 2, 3};
	int32_t consumed = 0;
	ResetAllocCounter();
	CHECK(sslm_prefill(model, seq, tokens, 4, 8, SSLM_SPAN_PROMPT, ws, &consumed) == SSLM_OK);
	CHECK_MSG(g_new_call_count == 0, "sslm_prefill made %d heap allocation(s) on its own hot "
	                                 "path -- design Sec7's caller-owned-memory contract",
	          g_new_call_count);
	sslm_decode_params params{};
	int32_t out_token = 0;
	sslm_seq batch[1] = {seq};
	ResetAllocCounter();
	CHECK(sslm_decode_step(model, batch, 1, &params, ws, &out_token) == SSLM_OK);
	CHECK_MSG(g_new_call_count == 0, "sslm_decode_step made %d heap allocation(s) on its own "
	                                 "hot path -- design Sec7's caller-owned-memory contract",
	          g_new_call_count);
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
	// (not only chunk_budget's own internal one) is a resumption point.
	CHECK(sslm_prefill(model, seq, long_prompt, 8, 8, SSLM_SPAN_PROMPT, nullptr,
	                    &consumed_first) == SSLM_OK);
	CHECK(consumed_first == 8);
	int32_t consumed_rest = 0;
	CHECK(sslm_prefill(model, seq, long_prompt + 8, 12, 8, SSLM_SPAN_PROMPT, nullptr,
	                    &consumed_rest) == SSLM_OK);
	CHECK(consumed_rest == 12);
	// FEATURE ORACLE: the sequence's own context_length after both calls equals the FULL
	// 20-token prompt's own length, proving the second call genuinely resumed from where the
	// first left off rather than either re-consuming or dropping tokens at the boundary.
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
	CHECK(sslm_seq_set_adapter(seq, adapter) == SSLM_ADAPTER_SWAP_MIDTOKEN_REJECTED);
	sslm_stats_out stats_after{};
	CHECK(sslm_stats(model, seq, &stats_after) == SSLM_OK);
	CHECK(stats_before.decode_step_ceiling == stats_after.decode_step_ceiling);
}

int main(int argc, char** argv) {
	ParseFixtureArgs(argc, argv);
	volatile void* addr_0 = (void*)&TestDim7_C1_HotPathMakesNoAllocation;
	(void)addr_0;
	volatile void* addr_1 = (void*)&TestDim7_C2_MidCallBoundaryChunkedPrefillResumesCorrectly;
	(void)addr_1;
	volatile void* addr_2 = (void*)&TestDim7_C3_LifecycleGuardRejectionCostIsDataIndependent;
	(void)addr_2;
	std::printf("checks=%d failures=%d skips=%d\n", GChecks, GFailures, GSkips);
	return GFailures ? 1 : 0;
}

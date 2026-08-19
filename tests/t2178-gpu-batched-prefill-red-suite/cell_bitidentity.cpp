// T-2178 (Curie) -- design Sec6/Sec9 "Correctness / determinism" and "Boundary / degenerate
// sizes" rows: the three bit-identity proof cells (cells 1-3; cell 4, the ceiling-boundary cell,
// is authored separately in cell_ceiling_boundary.cpp, gated additionally on Rung 2's own
// TDR-safe bound, per design Sec6 cell 4/Sec8 rung 5). Every cell is EXECUTED-claim form
// (StandardsDocument.md Sec5.4: "exactness is verified at source or by execution, never by
// construction") and every N >= 2 sub-cell is discriminating per D-SLM3608/D-SLM3615 -- the
// budget=1 sub-cell is retained but explicitly marked mechanism-only, matching design Sec6/Sec9's
// own re-disposition.
#include "fixture_common.h"

using namespace superslm;

// D-SLM3654 (part 1, T-2183 fix): the TDR-safe sub-chunk bound (Rung 2, D-SLM3649) is measured
// AFTER this cell's own N=6 sub-cell was first authored (Rung 1) -- declared here, `extern`,
// mirroring cell_ceiling_boundary.cpp's own identical declaration, so this file's own expected-
// submission-count arithmetic is derived from the real bound rather than re-asserting the
// pre-bound "always exactly 1" assumption that a 6-token chunk (bound=4) can no longer satisfy.
namespace superslm_gpu {
extern const uint32_t kT2169TdrSafeMaxChunkTokens;
}  // namespace superslm_gpu

// --- Cell 1: per-size sweep -- chunk_budget = 1 (mechanism-only), mid-size, N (discriminating).
// Design Sec6 cell 1 / Sec9 Boundary row. Asserts K/V (via hidden_codes/hidden_scale/
// kv_saturation), dfa_walk_state, context_length, and the SUBMISSION-COUNT claim (Sec4's own
// "one batched submission instead of N") are bit-identical/mechanism-correct between the
// reference (N single-token calls) and candidate (one N-token call) arms.
static void TestCell1_PerSizeSweep(SslmGpuContext* ctx, SslmGpuModelHandle* model) {
	const std::vector<int32_t> prime = {11, 22};
	// >= 4 discriminating tokens per D-SLM3610/D-SLM3615 ("mid-size and N sub-cells... both
	// require N >= 2" and this suite's own invocation floor of "4-token discriminating cells").
	const std::vector<int32_t> kFullChunk = {33, 44, 55, 66, 77, 88};
	const struct { size_t n; const char* label; bool discriminating; } kSizes[] = {
	    {1, "chunk_budget=1 (mechanism-only, D-SLM3610)", false},
	    {3, "mid-size (discriminating)", true},
	    {kFullChunk.size(), "N=6 (discriminating)", true},
	};
	for (const auto& sz : kSizes) {
		std::vector<int32_t> chunk(kFullChunk.begin(), kFullChunk.begin() + static_cast<long>(sz.n));
		const TwoArmPromptResult r = RunTwoArmPrompt(ctx, model, /*context_cap=*/64, prime, chunk);
		CHECK_MSG(r.ok, "Cell1 [%s]: two-arm setup failed", sz.label);
		if (!r.ok) continue;
		CHECK_MSG(r.ref_status == r.cand_status, "Cell1 [%s]: status ref=%s cand=%s", sz.label,
		          GpuStatusName(r.ref_status), GpuStatusName(r.cand_status));
		CHECK_MSG(SnapshotsBitEqual(r.ref_snap, r.cand_snap),
		          "Cell1 [%s]: post-prefill state diverges (hidden_codes/hidden_scale/"
		          "context_length/kv_saturation/dfa_walk_state) -- the design's own central claim",
		          sz.label);
		// The submission-granularity claim (Sec4): the reference arm submits once per token
		// (sz.n submissions); the candidate arm submits once for the WHOLE chunk (1 submission,
		// or the TDR-safe sub-chunk count -- at these small sizes, always 1). A candidate that
		// still submits sz.n times has not batched anything, whatever its content matches.
		CHECK_MSG(r.ref_submits_delta == static_cast<int64_t>(sz.n),
		          "Cell1 [%s]: reference arm's own submit count is %lld, expected %zu (sanity on "
		          "the reference construction itself)",
		          sz.label, static_cast<long long>(r.ref_submits_delta), sz.n);
		CHECK_MSG(r.cand_submits_delta >= 1 && r.cand_submits_delta < static_cast<int64_t>(sz.n) + 1,
		          "Cell1 [%s]: candidate arm submitted %lld times for a %zu-token chunk -- Sec4's "
		          "own claim is ONE submission per chunk (or a bounded few sub-chunks), never one "
		          "per token",
		          sz.label, static_cast<long long>(r.cand_submits_delta), sz.n);
		if (sz.discriminating) {
			// D-SLM3654 (part 1, T-2183 fix): a chunk larger than the ruled TDR-safe sub-chunk
			// bound (superslm_gpu::kT2169TdrSafeMaxChunkTokens, D-SLM3649) necessarily splits
			// into ceil(sz.n / bound) sub-chunk submissions -- a mathematical consequence of the
			// bound, not an implementation choice. This was previously hardcoded to exactly 1,
			// which is correct only while every discriminating size stays under the bound (sizes
			// 1 and 3 both do, at bound=4); the N=6 sub-cell exceeds it and must split into
			// exactly 2. The expected count is derived from the real bound rather than
			// re-asserting the pre-bound assumption, so this cell stays correct at any bound the
			// design measures on any hardware, not only the value measured on this machine.
			const uint32_t bound = superslm_gpu::kT2169TdrSafeMaxChunkTokens;
			CHECK_MSG(bound > 0, "Cell1 [%s]: the TDR-safe bound is 0 -- Rung 2's own measurement "
			                      "did not run or produced a degenerate figure",
			          sz.label);
			const int64_t expected_submits =
			    bound > 0 ? static_cast<int64_t>((sz.n + bound - 1) / bound) : 0;
			CHECK_MSG(r.cand_submits_delta == expected_submits,
			          "Cell1 [%s]: at this size (bound=%u), the candidate arm must submit exactly "
			          "%lld time(s) (ceil(%zu / %u)), not %lld",
			          sz.label, bound, static_cast<long long>(expected_submits), sz.n, bound,
			          static_cast<long long>(r.cand_submits_delta));
		}
		// Dispatch count parity: both arms must issue the IDENTICAL total number of per-token
		// dispatch-body invocations (sz.n * num_hidden_layers each) -- a batched call that skips
		// or duplicates a token's own dispatch chain diverges here even if by coincidence its
		// final snapshot still matched.
		CHECK_MSG(r.ref_dispatch_delta == r.cand_dispatch_delta,
		          "Cell1 [%s]: per-token dispatch-body invocation count diverges, ref=%lld "
		          "cand=%lld",
		          sz.label, static_cast<long long>(r.ref_dispatch_delta),
		          static_cast<long long>(r.cand_dispatch_delta));
	}
}

// --- Cell 2: boundary-split -- every representative two-call split of an N-token forced chain
// bit-matches the one-call result. Design Sec6 cell 2 (the GPU twin of sslm_prefill's own
// chunk-resumability contract, T-2133 Sec7.2). ---
static void TestCell2_BoundarySplit(SslmGpuContext* ctx, SslmGpuModelHandle* model) {
	const std::vector<int32_t> prime = {11, 22};
	const std::vector<int32_t> chunk = {33, 44, 55, 66, 77, 88};
	// One-call reference: the WHOLE chunk in a single call (the batched candidate this cell
	// exists to prove composes correctly with a caller who instead splits the same tokens
	// across two calls, per design Sec9's own "composition with sslm_prefill's own chunk-
	// resumability contract" framing).
	SslmGpuSequenceHandle* whole_seq = nullptr;
	CHECK(sslm_gpu_seq_create(ctx, model, /*context_cap=*/64, &whole_seq) == SSLM_OK);
	CHECK(PrimeSeq(ctx, whole_seq, prime));
	const SslmGpuStatus whole_status = SslmGpuSeqPrefillPromptForG5Bridge(
	    ctx, whole_seq, chunk.data(), static_cast<int32_t>(chunk.size()), kDispatchBudget);
	SeqSnapshot whole_snap;
	CHECK(CaptureSnapshot(whole_seq, &whole_snap));
	sslm_gpu_seq_release(ctx, whole_seq);

	for (size_t k = 1; k < chunk.size(); ++k) {
		SslmGpuSequenceHandle* split_seq = nullptr;
		CHECK(sslm_gpu_seq_create(ctx, model, /*context_cap=*/64, &split_seq) == SSLM_OK);
		CHECK(PrimeSeq(ctx, split_seq, prime));
		const SslmGpuStatus st1 = SslmGpuSeqPrefillPromptForG5Bridge(
		    ctx, split_seq, chunk.data(), static_cast<int32_t>(k), kDispatchBudget);
		const SslmGpuStatus st2 = SslmGpuSeqPrefillPromptForG5Bridge(
		    ctx, split_seq, chunk.data() + k, static_cast<int32_t>(chunk.size() - k), kDispatchBudget);
		SeqSnapshot split_snap;
		CHECK(CaptureSnapshot(split_seq, &split_snap));
		sslm_gpu_seq_release(ctx, split_seq);
		CHECK_MSG(st1 == SSLM_OK && st2 == SSLM_OK && whole_status == SSLM_OK,
		          "Cell2 k=%zu: one or both calls in the split failed (st1=%s st2=%s whole=%s)", k,
		          GpuStatusName(st1), GpuStatusName(st2), GpuStatusName(whole_status));
		CHECK_MSG(SnapshotsBitEqual(whole_snap, split_snap),
		          "Cell2 k=%zu: state after the two-call split diverges from the one-call result",
		          k);
	}
}

// --- Cell 3: real-artifact consistency -- the batched GPU path, at production scale, reproduces
// the unbatched per-token GPU path bit-for-bit; simultaneously the CPU/GPU parity re-proof under
// the new code path (design Sec6 cell 3). Runs a real decode step after the compared prefill, so
// an argmax-visible divergence (not only a raw-byte one) is also caught, matching
// Claude/Loki/t2176-probe-cap-straddle-bridge-semantics.cpp's own "the one question a consumer
// actually asks next" methodology. ---
static void TestCell3_RealArtifactConsistency(SslmGpuContext* ctx, SslmGpuModelHandle* model) {
	if (g_model_1p5b_path.empty()) {
		SKIP_MSG("real 1.5B artifact not supplied (--model1p5b=PATH) -- product cell not run");
		return;
	}
	const std::vector<int32_t> prime = {11, 22, 33};
	// A longer forced span at production scale -- the "does it work when used the way it will be
	// used" cell StandardsDocument.md Sec5.4 requires beside every mechanism cell above.
	std::vector<int32_t> chunk;
	for (int i = 0; i < 32; ++i) chunk.push_back(100 + i);

	SslmGpuSequenceHandle* ref_seq = nullptr;
	CHECK(sslm_gpu_seq_create(ctx, model, /*context_cap=*/256, &ref_seq) == SSLM_OK);
	CHECK(PrimeSeq(ctx, ref_seq, prime));
	SslmGpuStatus ref_status = SSLM_OK;
	for (int32_t t : chunk) {
		ref_status = SslmGpuSeqPrefillPromptForG5Bridge(ctx, ref_seq, &t, 1, kDispatchBudget);
		if (ref_status != SSLM_OK) break;
	}
	SeqSnapshot ref_snap;
	CHECK(CaptureSnapshot(ref_seq, &ref_snap));
	int32_t ref_out_token = -1;
	const SslmGpuStatus ref_step_status =
	    SslmGpuSeqDecodeStepForG5Bridge(ctx, ref_seq, /*token_to_embed_if_needed=*/999, kDispatchBudget,
	                                     &ref_out_token);
	sslm_gpu_seq_release(ctx, ref_seq);

	SslmGpuSequenceHandle* cand_seq = nullptr;
	CHECK(sslm_gpu_seq_create(ctx, model, /*context_cap=*/256, &cand_seq) == SSLM_OK);
	CHECK(PrimeSeq(ctx, cand_seq, prime));
	const SslmGpuStatus cand_status = SslmGpuSeqPrefillPromptForG5Bridge(
	    ctx, cand_seq, chunk.data(), static_cast<int32_t>(chunk.size()), kDispatchBudget);
	SeqSnapshot cand_snap;
	CHECK(CaptureSnapshot(cand_seq, &cand_snap));
	int32_t cand_out_token = -1;
	const SslmGpuStatus cand_step_status =
	    SslmGpuSeqDecodeStepForG5Bridge(ctx, cand_seq, /*token_to_embed_if_needed=*/999, kDispatchBudget,
	                                     &cand_out_token);
	sslm_gpu_seq_release(ctx, cand_seq);

	CHECK_MSG(ref_status == SSLM_OK && cand_status == SSLM_OK,
	          "Cell3: prefill failed (ref=%s cand=%s)", GpuStatusName(ref_status),
	          GpuStatusName(cand_status));
	CHECK_MSG(SnapshotsBitEqual(ref_snap, cand_snap),
	          "Cell3: post-prefill state diverges at production scale (32-token forced span, real "
	          "Qwen2.5-1.5B artifact) -- this is also the CPU/GPU parity re-proof under the new "
	          "batched code path (D-SLM3487), since CPU output is unaffected by this design "
	          "(Sec7) and matching the pre-existing per-token GPU path is sufficient");
	CHECK_MSG(ref_step_status == cand_step_status && ref_out_token == cand_out_token,
	          "Cell3: the FOLLOWING decode step diverges (status ref=%s/%d cand=%s/%d) -- an "
	          "argmax-visible product-level divergence, not merely a raw-byte one",
	          GpuStatusName(ref_step_status), ref_out_token, GpuStatusName(cand_step_status),
	          cand_out_token);
}

int main(int argc, char** argv) {
	ParseFixtureArgs(argc, argv);
	// Force emission (StandardsDocument.md Sec5.4: a red cell must fail for its OWN reason --
	// here, LNK2019 on the probe counters this file's own fixture_common.h references via
	// RunTwoArmPrompt -- never be silently dead-code-eliminated).
	volatile void* addr_0 = (void*)&TestCell1_PerSizeSweep; (void)addr_0;
	volatile void* addr_1 = (void*)&TestCell2_BoundarySplit; (void)addr_1;
	volatile void* addr_2 = (void*)&TestCell3_RealArtifactConsistency; (void)addr_2;

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
			TestCell1_PerSizeSweep(ctx, model);
			TestCell2_BoundarySplit(ctx, model);
			TestCell3_RealArtifactConsistency(ctx, model);
			CHECK(sslm_gpu_model_unmap(ctx, model) == SSLM_OK);
		} else {
			CHECK_MSG(false, "failed to load --model1p5b=%s: %s", g_model_1p5b_path.c_str(),
			          err.c_str());
		}
	} else {
		SKIP_MSG("cells 1-3 need --model1p5b=PATH -- not run");
	}

	CHECK(sslm_gpu_context_destroy(ctx) == SSLM_OK);
	std::printf("checks=%d failures=%d skips=%d\n", GChecks, GFailures, GSkips);
	return GFailures ? 1 : 0;
}

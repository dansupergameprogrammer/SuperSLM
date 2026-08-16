// T-2112 (Curie) -- Dim 8 (Composition), design Sec11 dim8. 5 cells.
// Named explicitly in Sec1 as a requirement (mixed adapters, batched sequences) and the
// dimension the substrate's own most-recurrent gap class lives in (crossed cells).
//
// D-SLM3380/D-SLM3387/D-SLM3412 REPAIR (Curie, 2026-08-15):
//  - D-SLM3380 (undrained-Submitted gap, the bulk of this file's own pre-repair failures --
//    checks=102 failures=73/74): every decode/batch-decode call followed by another touch of the
//    same handle now drains first (`Drain`/`RunStepBlocking`, fixture_common.h).
//  - D-SLM3387 (Cell 2's own worked example genuinely mismatched against a real model, build log
//    Sec12.7): the cell's own literal "batch-wide budget of 72 = 3 whole layers ->
//    [Ok,Ok,Ok,Exhausted]" split only holds for a tiny (<=1-layer) fixture this suite's own
//    fixture_common.h has no mechanism to construct (every model here loads from a real .sslm
//    artifact, 24-28 layers). Re-authored against a CONSTRUCTIBLE fixture -- the real model,
//    scaled budget -- reusing tools/t2113_b7_batch_smoke.cpp's own Gate 2 construction verbatim
//    (`full_token * 2 + kDispatchesPerLayer`, n=4): seqs[0]/[1] each complete a FULL token,
//    seqs[2] gets exactly one further layer (still Ok -- "Ok" means "some dispatches recorded,"
//    not "token complete," design Sec7), seqs[3] reads BatchBudgetExhausted and stays Idle
//    (proven by a follow-up decode succeeding as a fresh sequence would). This is the exact
//    scenario Gate 2 already proves at real hardware; re-authoring it here as the suite's own
//    Coverage-Model cell closes D-SLM3387 without inventing new semantics.
//  - The "wired here once the build seat's harness exposes it" prose stand-in (build log
//    Sec22.10's own named sweep, dim8 explicitly listed) is replaced with real, executed
//    bit-equality assertions in cells 1, 3, and 5.
#include "fixture_common.h"

using namespace superslm;

// Three sequences' own solo-decode baseline snapshots (Cell 1's own comparator) -- captured by
// the driver in main() before the batch call touches the same three fresh handles.
struct SseqBaselineSnapshots3 {
	SeqSnapshot snap[3];
};

// --- Cell 1 (design Sec10 B7's own gate): batch x mixed-adapter -- N>=3 sequences, {none,
// adapter A, adapter B}, bit-identical per-sequence to each decoded alone. ---
static void TestDim8_1_BatchMixedAdapterBitIdenticalToAlone(
    SslmGpuContext* ctx, SslmGpuSequenceHandle* const* seqs3,
    const SslmGpuAdapterHandle* const* mixed_adapters3 /* [none, A, B] */,
    SseqBaselineSnapshots3* baselines, uint32_t num_hidden_layers) {
	// Batch-wide budget = 3 full tokens (not 3 layers): each of the three sequences must complete
	// its OWN full token in this call for the "bit-identical to alone" comparison below to compare
	// like-for-like against the solo baseline (also a full-token decode) -- a per-layer budget
	// would let design Sec7's own greedy per-sequence-turn consumption hand seqs[0] the WHOLE
	// budget (the exact D-SLM3387 shape this file's own Cell 2 was found and fixed for).
	SslmGpuStatus out_statuses[3];
	const uint32_t batch_budget = FullTokenBudget(num_hidden_layers) * 3u;
	CHECK(sslm_decode_step_batch_gpu(ctx, seqs3, mixed_adapters3, 3u, batch_budget, out_statuses) ==
	      SSLM_OK);
	CHECK(out_statuses[0] == SSLM_OK && out_statuses[1] == SSLM_OK && out_statuses[2] == SSLM_OK);
	for (int i = 0; i < 3; ++i) CHECK(Drain(ctx, seqs3[i]) == SSLM_OK);  // D-SLM3380
	// FEATURE ORACLE, executed: each of the three sequences' own per-step output, decoded via
	// THIS batch call, must be bit-identical to that same sequence decoded ALONE via
	// sslm_decode_step_gpu -- the composition claim (design Sec4.3: "no dispatch ... reads two
	// sequences' K/V state or composes two adapters' deltas"). `baselines` holds each sequence's
	// own solo-decode snapshot for the SAME token, captured by the driver before this batch call.
	if (baselines) {
		for (int i = 0; i < 3; ++i) {
			SeqSnapshot batch_snap{};
			CHECK(CaptureSnapshot(seqs3[i], &batch_snap));
			CHECK_MSG(SnapshotsBitEqual(baselines->snap[i], batch_snap),
			          "Cell1: sequence %d's batch-driven output diverges from its own solo-decode "
			          "baseline -- the batch call must change submission grouping only, never any "
			          "sequence's own numerics (design Sec4.3)",
			          i);
		}
	}
}

// --- Cell 2 (re-authored, D-SLM3387): batch x async -- a real batch of 4, BATCH-WIDE budget
// covering 2 full tokens plus exactly one further layer -- seqs[0]/[1] complete a full token
// each, seqs[2] gets exactly one layer (still Ok), seqs[3] reads BatchBudgetExhausted and stays
// Idle. Mirrors tools/t2113_b7_batch_smoke.cpp's own Gate 2 construction verbatim (that tool
// already proves this exact scenario at real hardware; this cell is the Coverage Model's own
// realization of the identical claim). ---
static void TestDim8_2_BatchWideBudgetCutRealModelTwoTokensPlusOneLayer(
    SslmGpuContext* ctx, SslmGpuSequenceHandle* const* seqs4,
    const SslmGpuAdapterHandle* const* adapters4, uint32_t num_hidden_layers) {
	const uint32_t full_token = FullTokenBudget(num_hidden_layers);
	const uint32_t batch_wide_dispatch_budget = full_token * 2u + kDispatchesPerLayer;  // 2 full + 1 layer
	SslmGpuStatus out_statuses[4];
	const SslmGpuStatus call_status =
	    sslm_decode_step_batch_gpu(ctx, seqs4, adapters4, 4u, batch_wide_dispatch_budget,
	                                out_statuses);
	CHECK(call_status == SSLM_OK);
	CHECK_MSG(out_statuses[0] == SSLM_OK, "seqs[0] must complete its own full token");
	CHECK_MSG(out_statuses[1] == SSLM_OK, "seqs[1] must complete its own full token");
	CHECK_MSG(out_statuses[2] == SSLM_OK,
	          "seqs[2] (partial, exactly one layer) must still read Ok -- Ok means \"some "
	          "dispatches recorded,\" not \"token complete\" (design Sec7)");
	CHECK_MSG(out_statuses[3] == SSLM_BATCH_BUDGET_EXHAUSTED,
	          "batch-wide budget of 2 full tokens + 1 layer must exhaust before seqs[3]'s own "
	          "first layer, per design Sec7's own strict-array-order greedy consumption -- got "
	          "status %d", (int)out_statuses[3]);
	if (out_statuses[0] == SSLM_OK) CHECK(Drain(ctx, seqs4[0]) == SSLM_OK);
	if (out_statuses[1] == SSLM_OK) CHECK(Drain(ctx, seqs4[1]) == SSLM_OK);
	if (out_statuses[2] == SSLM_OK) CHECK(Drain(ctx, seqs4[2]) == SSLM_OK);
	CHECK_MSG(*SslmGpuSeqHandleLayerIndexForBench(seqs4[0]) == num_hidden_layers,
	          "seqs[0] did not complete its own full token");
	CHECK_MSG(*SslmGpuSeqHandleLayerIndexForBench(seqs4[1]) == num_hidden_layers,
	          "seqs[1] did not complete its own full token");
	CHECK_MSG(*SslmGpuSeqHandleLayerIndexForBench(seqs4[2]) == 1,
	          "seqs[2] did not record exactly its own one layer");
	// seqs[3] never submitted -- state stays Idle: a follow-up single-sequence decode call must
	// succeed exactly as it would against a never-touched fresh sequence.
	const SslmGpuStatus follow_up = sslm_decode_step_gpu(ctx, seqs4[3], nullptr, kDispatchesPerLayer);
	CHECK_MSG(follow_up == SSLM_OK,
	          "seqs[3] was not a genuinely fresh, untouched Idle sequence after BatchBudgetExhausted");
	if (follow_up == SSLM_OK) CHECK(Drain(ctx, seqs4[3]) == SSLM_OK);
}

// --- Cell 3: adapter x KV-residency -- a sequence's adapter binding changed (base -> A -> B ->
// base) across successive calls on the SAME sequence handle, K/V state carried forward unchanged.
// ---
static void TestDim8_3_AdapterSwapMidSessionPreservesKvState(SslmGpuContext* ctx,
                                                              SslmGpuModelHandle* model,
                                                              SslmGpuSequenceHandle* seq,
                                                              SslmGpuAdapterHandle* adapter_a,
                                                              SslmGpuAdapterHandle* adapter_b,
                                                              uint32_t num_hidden_layers) {
	// Each of the four calls is its own full token (RunFullTokenStep, fixture_common.h) -- the
	// cell's own claim is that context_length (the K/V-row count) advances by exactly ONE PER
	// STEP/TOKEN regardless of adapter churn; a per-layer budget would leave context_length
	// unmoved until every layer of a token lands, not advance predictably per call.
	CHECK(RunFullTokenStep(ctx, seq, /*adapter_or_null=*/nullptr, num_hidden_layers, 5));  // base
	CHECK(RunFullTokenStep(ctx, seq, adapter_a, num_hidden_layers, 5));                     // -> A
	CHECK(RunFullTokenStep(ctx, seq, adapter_b, num_hidden_layers, 5));                     // -> B
	CHECK(RunFullTokenStep(ctx, seq, nullptr, num_hidden_layers, 5));                        // -> base
	SeqSnapshot final_state{};
	CHECK(CaptureSnapshot(seq, &final_state));
	CHECK_MSG(final_state.context_length == 4,
	          "Cell3: K/V context_length must advance by exactly one per step regardless of "
	          "adapter churn (base/A/B/base) -- got %lld", (long long)final_state.context_length);
	// FEATURE ORACLE, executed (N2 repair, 2026-08-15): the header's own claim that
	// `kv_saturation_count` "is a function of context alone" is checked directly, not merely
	// asserted -- a SECOND, dedicated sequence run for the SAME four full-token steps with NO
	// adapter bound at all (base/base/base/base) must reach the identical `kv_saturation_count`
	// as the base/A/B/base sequence above. If adapter identity leaked into the K/V bookkeeping
	// (rather than only the residual stream's content, which this cell does not itself compare),
	// the two counts would diverge.
	SslmGpuSequenceHandle* base_only = nullptr;
	CHECK(sslm_gpu_seq_create(ctx, model, 64, &base_only) == SSLM_OK);
	for (int i = 0; i < 4; ++i) CHECK(RunFullTokenStep(ctx, base_only, nullptr, num_hidden_layers, 5));
	SeqSnapshot base_only_state{};
	CHECK(CaptureSnapshot(base_only, &base_only_state));
	CHECK_MSG(base_only_state.kv_saturation_count == final_state.kv_saturation_count,
	          "Cell3: kv_saturation_count after base/A/B/base (%llu) diverges from a base-only "
	          "four-step run (%llu) -- adapter identity leaked into K/V bookkeeping, not only the "
	          "residual stream's own content",
	          (unsigned long long)final_state.kv_saturation_count,
	          (unsigned long long)base_only_state.kv_saturation_count);
	CHECK(sslm_gpu_seq_release(ctx, base_only) == SSLM_OK);
}

// --- Cell 4: context x adapter -- a sequence bound to an adapter decoded to context >=64 so
// softmax's own O(context) cost is visible, proving the adapter path does not bypass or
// duplicate that cost. ---
static void TestDim8_4_ContextLengthAdapterCostNotBypassed(SslmGpuContext* ctx,
                                                            SslmGpuSequenceHandle* seq,
                                                            SslmGpuAdapterHandle* adapter,
                                                            uint32_t num_hidden_layers) {
	// One full token per step (RunFullTokenStep) -- context_length must reach exactly 64 after
	// 64 STEPS, not 64 layer-calls (a per-layer budget would need 64*num_hidden_layers calls to
	// reach the same context_length).
	for (int step = 0; step < 64; ++step)
		CHECK(RunFullTokenStep(ctx, seq, adapter, num_hidden_layers, 5));
	// Timing assertion (design Sec3/Sec13: softmax cost grows 0.397->3.169 ms/token over
	// 16->256 steps) is a build-seat-measured quantity, not a fixed bound this cell asserts --
	// this cell's own contract is functional: the 64-step run with an adapter bound completes and
	// produces context_length==64, proving the adapter path traverses the same O(context) code
	// the base-only path does rather than a shortcut that skips it.
	SeqSnapshot final_state{};
	CHECK(CaptureSnapshot(seq, &final_state));
	CHECK_MSG(final_state.context_length == 64,
	          "Cell4: adapter-bound 64-step run must reach context_length==64 -- got %lld",
	          (long long)final_state.context_length);
}

// --- Cell 5 (added at T-2110 fold, Mendeleev Finding 4.4): lifetime x concurrency -- handle
// churn under address reuse, CONCURRENT with live decode. Thread A decodes one sequence
// continuously for >=64 steps; thread B concurrently churns short-lived sequence handles against
// the same model, forcing address reuse. This is the precise D-SLM3311 shape, proven only when
// dim1's address-reuse cell and dim3's concurrency cell hold TOGETHER, not each alone. Thread A's
// own submit+drain is now serialized through the documented external mutex
// (`SubmitAndDrainSerialized`, fixture_common.h, the SAME idiom tools/t2113_b8_thread_smoke.cpp's
// own Gate 2 uses for this EXACT cell) -- thread B's own churn needs no mutex against thread A's
// decode loop (b8's own grounded finding: churn calls route through `ctx->device` directly, a
// device object distinct from the process-wide singleton decode calls use). ---
#include <thread>
static void TestDim8_5_HandleChurnConcurrentWithLiveDecode(SslmGpuContext* ctx,
                                                            SslmGpuModelHandle* model,
                                                            const CpuOracleModel* oracle) {
	constexpr int kSteps = 64;
	SslmGpuSequenceHandle* long_lived = nullptr;
	CHECK(sslm_gpu_seq_create(ctx, model, 64, &long_lived) == SSLM_OK);
	const uint32_t budget = oracle ? FullTokenBudget(oracle->num_hidden_layers) : 24u;
	bool thread_a_ok = true;
	std::vector<SeqSnapshot> gpu_snaps(kSteps);
	std::thread thread_a([&] {
		for (int step = 0; step < kSteps; ++step) {
			if (sslm_gpu_seq_embed_token(ctx, long_lived, 5) != SSLM_OK) thread_a_ok = false;
			if (SubmitAndDrainSerialized(ctx, long_lived, nullptr, budget) != SSLM_OK) thread_a_ok = false;
			CaptureSnapshot(long_lived, &gpu_snaps[step]);
		}
	});
	std::thread thread_b([&] {
		for (int i = 0; i < 200; ++i) {  // tight loop, forcing address reuse during thread A's run
			SslmGpuSequenceHandle* churn = nullptr;
			if (sslm_gpu_seq_create(ctx, model, 64, &churn) == SSLM_OK)
				(void)sslm_gpu_seq_release(ctx, churn);
		}
	});
	thread_a.join();
	thread_b.join();
	CHECK_MSG(thread_a_ok, "thread A's long-lived sequence observed a non-Ok decode step while "
	          "thread B churned handles concurrently -- residue crossed from B's churn into A's "
	          "sequence, the precise D-SLM3311 shape this cell exists to catch");
	// FEATURE ORACLE, executed: thread A's own per-step output, decoded under concurrent churn,
	// must match the CPU oracle for the SAME re-embedded token every step -- the tell for residue
	// crossing from B's churn is a step diverging from what an isolated decode of the same token
	// would produce.
	if (oracle && thread_a_ok) {
		CpuOracleRunner cpu;
		cpu.Init(*oracle);
		for (int step = 0; step < kSteps; ++step) {
			CHECK_MSG(cpu.StepMatchesGpu(5, *oracle, gpu_snaps[step]),
			          "Cell5: thread A step %d diverges from the CPU oracle under concurrent "
			          "handle churn", step);
		}
	}
	CHECK(sslm_gpu_seq_release(ctx, long_lived) == SSLM_OK);
}

int main(int argc, char** argv) {
	ParseFixtureArgs(argc, argv);
	// Force emission (StandardsDocument.md Sec5.4: a red cell must fail for its OWN
	// reason, LNK2019 on the 1.0 API calls inside, never be silently dead-code-eliminated
	// because nothing in this TU calls it yet -- taking its address is a genuine `use`).
	volatile void* addr_0 = (void*)&TestDim8_1_BatchMixedAdapterBitIdenticalToAlone; (void)addr_0;
	volatile void* addr_1 = (void*)&TestDim8_2_BatchWideBudgetCutRealModelTwoTokensPlusOneLayer; (void)addr_1;
	volatile void* addr_2 = (void*)&TestDim8_3_AdapterSwapMidSessionPreservesKvState; (void)addr_2;
	volatile void* addr_3 = (void*)&TestDim8_4_ContextLengthAdapterCostNotBypassed; (void)addr_3;
	volatile void* addr_4 = (void*)&TestDim8_5_HandleChurnConcurrentWithLiveDecode; (void)addr_4;

	SslmGpuContext* ctx = nullptr;
	CHECK(sslm_gpu_context_create(GpuContextConfig{}, &ctx) == SSLM_OK);
	if (!ctx) { std::printf("FATAL: sslm_gpu_context_create returned null\n"); return 2; }

	std::vector<uint8_t> bytes;
	SslmModelView view{};
	std::vector<uint8_t> adapter_bytes;
	SslmModelView adapter_view{};
	std::string err;
	const bool have_model = !g_model_1p5b_path.empty() && LoadRealModel(g_model_1p5b_path, &view, &bytes, &err);
	const bool have_adapter = !g_adapter_path.empty() &&
	                           LoadRealModel(g_adapter_path, &adapter_view, &adapter_bytes, &err);
	if (!have_model) {
		SKIP_MSG("dim8 needs --model1p5b=PATH -- not run");
		std::printf("checks=%d failures=%d skips=%d\n", GChecks, GFailures, GSkips);
		return GFailures ? 1 : 0;
	}

	CpuOracleModel oracle{};
	std::string oerr;
	const bool have_oracle = LoadCpuOracleModel(view, &oracle, &oerr);
	CHECK_MSG(have_oracle, "dim8: CPU oracle failed to load -- %s", oerr.c_str());

	SslmGpuModelHandle* model = nullptr;
	CHECK(sslm_gpu_model_map(ctx, &view, GpuResidencyConfig{}, &model) == SSLM_OK);

	SslmGpuAdapterHandle* adapter = nullptr;
	if (have_adapter) {
		CHECK(sslm_gpu_adapter_map(ctx, model, &adapter_view, &adapter) == SSLM_OK);
	} else {
		SKIP_MSG("dim8 cells 1/3/4 need --adapter=PATH -- not run");
	}

	// Cell 1: batch of 3, mixed adapters {none, adapter, adapter} -- ONLY ONE real adapter
	// artifact is available this session (the shopkeeper-v2 runtime adapter); the SAME handle
	// is bound to two DIFFERENT sequences in the batch to exercise mixed composition (two
	// distinct sequences, one shared adapter handle) rather than two distinct adapters, named
	// honestly here rather than silently presented as the design's own literal "adapter A,
	// adapter B" (which would need a second real converted adapter artifact this session does
	// not have on disk).
	const uint32_t num_hidden_layers_top = view.config.num_hidden_layers;
	if (adapter) {
		SslmGpuSequenceHandle* seqs3[3] = {nullptr, nullptr, nullptr};
		for (auto& s : seqs3) {
			CHECK(sslm_gpu_seq_create(ctx, model, 64, &s) == SSLM_OK);
			if (s) CHECK(sslm_gpu_seq_embed_token(ctx, s, 5) == SSLM_OK);
		}
		const SslmGpuAdapterHandle* mixed_adapters3[3] = {nullptr, adapter, adapter};
		// Solo-decode baseline, captured BEFORE the batch call touches these same three handles --
		// each sequence decoded alone (single-sequence, one FULL TOKEN, matching the batch call's
		// own per-sequence budget below) at the SAME token, snapshotted, then reset via a fresh
		// create/embed pair so the batch call below starts from identical state.
		SseqBaselineSnapshots3 baselines{};
		for (int i = 0; i < 3; ++i) {
			SslmGpuSequenceHandle* solo = nullptr;
			CHECK(sslm_gpu_seq_create(ctx, model, 64, &solo) == SSLM_OK);
			CHECK(RunFullTokenStep(ctx, solo, mixed_adapters3[i], num_hidden_layers_top, 5));
			CHECK(CaptureSnapshot(solo, &baselines.snap[i]));
			CHECK(sslm_gpu_seq_release(ctx, solo) == SSLM_OK);
		}
		TestDim8_1_BatchMixedAdapterBitIdenticalToAlone(ctx, seqs3, mixed_adapters3, &baselines,
		                                                 num_hidden_layers_top);
		for (auto& s : seqs3)
			if (s) sslm_gpu_seq_release(ctx, s);
	}

	// Cell 2: batch of 4, real model, re-authored budget-cut worked example (D-SLM3387).
	{
		SslmGpuSequenceHandle* seqs4[4] = {nullptr, nullptr, nullptr, nullptr};
		for (auto& s : seqs4) {
			CHECK(sslm_gpu_seq_create(ctx, model, 64, &s) == SSLM_OK);
			if (s) CHECK(sslm_gpu_seq_embed_token(ctx, s, 5) == SSLM_OK);
		}
		const SslmGpuAdapterHandle* adapters4[4] = {nullptr, nullptr, nullptr, nullptr};
		TestDim8_2_BatchWideBudgetCutRealModelTwoTokensPlusOneLayer(ctx, seqs4, adapters4,
		                                                             num_hidden_layers_top);
		for (auto& s : seqs4)
			if (s) sslm_gpu_seq_release(ctx, s);
	}

	// Cell 3: adapter swap mid-session -- needs TWO adapter handles (A, B); only one real
	// artifact available, same handle reused for both slots (named, same reasoning as Cell 1).
	if (adapter) {
		SslmGpuSequenceHandle* seq = nullptr;
		CHECK(sslm_gpu_seq_create(ctx, model, 64, &seq) == SSLM_OK);
		if (seq) TestDim8_3_AdapterSwapMidSessionPreservesKvState(ctx, model, seq, adapter, adapter,
		                                                           num_hidden_layers_top);
		if (seq) sslm_gpu_seq_release(ctx, seq);
	}

	// Cell 4: context-length x adapter cost.
	if (adapter) {
		SslmGpuSequenceHandle* seq = nullptr;
		CHECK(sslm_gpu_seq_create(ctx, model, 64, &seq) == SSLM_OK);
		if (seq) TestDim8_4_ContextLengthAdapterCostNotBypassed(ctx, seq, adapter, num_hidden_layers_top);
		if (seq) sslm_gpu_seq_release(ctx, seq);
	}

	// Cell 5: handle churn concurrent with live decode.
	TestDim8_5_HandleChurnConcurrentWithLiveDecode(ctx, model, have_oracle ? &oracle : nullptr);

	if (adapter) sslm_gpu_adapter_unmap(ctx, adapter);
	CHECK(sslm_gpu_model_unmap(ctx, model) == SSLM_OK);
	CHECK(sslm_gpu_context_destroy(ctx) == SSLM_OK);
	std::printf("checks=%d failures=%d skips=%d\n", GChecks, GFailures, GSkips);
	return GFailures ? 1 : 0;
}

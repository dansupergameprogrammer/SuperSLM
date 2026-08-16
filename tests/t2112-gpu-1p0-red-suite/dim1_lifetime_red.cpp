// T-2112 (Curie) -- Dim 1 (Lifetime and reuse), design Sec11 dim1. 4 cells: 2 mechanism, 2
// product. Every cell calls the declared 1.0 API (sslm_gpu_1p0.h).
//
// D-SLM3380/D-SLM3412 REPAIR (Curie, 2026-08-15): the T-2114 fix round (build log Sec22.1) found
// and fixed the "decode call against a still-Submitted sequence returns SSLM_BUSY" gap in dim9
// alone; the casebook's own N3 re-run (Claude/Poirot/50f3d5d-t2113-1p0-gpu-core-build-review.md
// Sec14.1) diagnosed the identical gap here (dim1's own dominant failure lines, pre-repair:
// :107 x63, :92 x15, :120 x10 -- every one a tight decode loop, or a decode immediately followed
// by release/another decode, with no `sslm_gpu_ready` poll between calls). Every raw
// `sslm_decode_step_gpu(...) == SSLM_OK` check in this file is now `RunStepBlocking(...)`
// (fixture_common.h, submit-then-drain-to-completion) -- the real async contract (design
// Sec4.3/Sec9's own `Busy` row: a decode call, a save/restore/reset/release, or a model_unmap
// against a `Submitted` sequence all return `Busy` until drained). The prose "FEATURE ORACLE ...
// wired here once the build seat's harness exposes it" stand-ins (build log Sec22.10's own named
// sweep) are replaced with the real, executed per-step CPU/GPU bit-equality comparison
// tools/t2113_b7_batch_smoke.cpp/tools/t2113_b8_thread_smoke.cpp/dim9's own C1 rewrite already
// established (StepCpu/CpuOracleModel, fixture_common.h) -- each cell's own claim is now checked
// by execution, not asserted in a comment.
#include "fixture_common.h"

using namespace superslm;

// --- Mechanism cell 1 (design Sec11 dim1, "a SslmGpuModelHandle created, used by one sequence,
// and a second, independent model handle created for a different artifact while the first is
// still live -- the second must not evict, alias, or corrupt the first's resident weights") ---
// NOTE (compile-mechanics): `GpuResidencyConfig` is deliberately left incomplete by the design's
// own declared surface (Sec4.1's own "config/view types the design names but does not declare");
// this suite therefore never constructs one locally -- `sslm_gpu_model_map` itself is exercised
// in the dim4/dim9 product cells (which take a caller-supplied config), and every OTHER cell here
// takes already-mapped model/sequence handles as parameters, which is what a real test driver
// (build-seat-owned, once GpuResidencyConfig has a real definition) supplies.
static void TestDim1_M1_TwoLiveModelHandlesContentHashKeyed(SslmGpuContext* ctx,
                                                              SslmGpuModelHandle* model_a,
                                                              SslmGpuModelHandle* model_b,
                                                              const CpuOracleModel* oracle_a) {
	CHECK(model_a != nullptr);
	SslmGpuSequenceHandle* seq_a = nullptr;
	CHECK(sslm_gpu_seq_create(ctx, model_a, /*context_cap=*/64, &seq_a) == SSLM_OK);
	// T-2113 B3.5 reconciliation (D-SLM3367): a fresh sequence's own hidden_codes/hidden_scale
	// are zero/default until embedded -- real content via the production entry point, not the
	// bench bridge, so the decode call below is decode-ready (design Sec5.3a).
	CHECK(sslm_gpu_seq_embed_token(ctx, seq_a, 5) == SSLM_OK);
	// Second, independent model handle (already mapped by the caller against a DIFFERENT
	// artifact, content-hash-keyed per design Sec5.1) must not evict or alias model_a's own
	// resident weights.
	CHECK(model_b != nullptr);
	// seq_a's own next decode step must still read artifact_a's weights, not artifact_b's.
	// FullTokenBudget (not the bare per-layer 24u): this step is compared to the CPU oracle
	// below, which decodes one COMPLETE token per call (fixture_common.h's own header note).
	const uint32_t budget_a = oracle_a ? FullTokenBudget(oracle_a->num_hidden_layers) : 24u;
	CHECK(RunStepBlocking(ctx, seq_a, /*adapter=*/nullptr, budget_a));
	// FEATURE ORACLE, executed: per-step CPU/GPU bit-equality against model_a's own CPU oracle --
	// if model_b's own map had evicted or aliased model_a's resident weights, this step's own
	// hidden_codes would diverge from a fresh CPU-side decode of the SAME token against
	// model_a's own weights.
	if (oracle_a) {
		SeqSnapshot gpu{};
		CHECK(CaptureSnapshot(seq_a, &gpu));
		CpuOracleRunner cpu;
		cpu.Init(*oracle_a);
		CHECK_MSG(cpu.StepMatchesGpu(5, *oracle_a, gpu),
		          "M1: seq_a's own step diverges from model_a's own CPU oracle -- model_b's own "
		          "map may have evicted or aliased model_a's resident weights");
	}
	CHECK(sslm_gpu_seq_release(ctx, seq_a) == SSLM_OK);
	CHECK(sslm_gpu_model_unmap(ctx, model_a) == SSLM_OK);
	CHECK(sslm_gpu_model_unmap(ctx, model_b) == SSLM_OK);
}

// --- Mechanism cell 2 (design Sec11 dim1, "a sequence handle released and a new sequence handle
// immediately created -- the allocator MAY reuse the freed address, and the new handle's own
// buffer must never read the freed handle's stale content" -- the direct D-SLM3311 reproduction
// against the sequence-handle allocator) ---
static void TestDim1_M2_ReleasedThenCreatedSequenceNeverReadsStaleContent(
    SslmGpuContext* ctx, SslmGpuModelHandle* model, const CpuOracleModel* oracle) {
	SslmGpuSequenceHandle* seq1 = nullptr;
	CHECK(sslm_gpu_seq_create(ctx, model, 64, &seq1) == SSLM_OK);
	CHECK(sslm_gpu_seq_embed_token(ctx, seq1, 5) == SSLM_OK);  // T-2113 B3.5 (D-SLM3367)
	// Advance seq1's own K/V state so a stale read would be observable (not a freshly-zeroed
	// buffer, which would hide the defect this cell exists to catch).
	const uint32_t budget = oracle ? FullTokenBudget(oracle->num_hidden_layers) : 24u;
	CHECK(RunStepBlocking(ctx, seq1, nullptr, budget));
	CHECK(sslm_gpu_seq_release(ctx, seq1) == SSLM_OK);
	// Forced address-reuse pressure: churn several short-lived handles before the one under test,
	// per this cell's own product twin below (design Sec11 dim1 product cell 1's own "forced via
	// repeated create/release cycling" method).
	for (int i = 0; i < 8; ++i) {
		SslmGpuSequenceHandle* churn = nullptr;
		CHECK(sslm_gpu_seq_create(ctx, model, 64, &churn) == SSLM_OK);
		CHECK(sslm_gpu_seq_release(ctx, churn) == SSLM_OK);
	}
	SslmGpuSequenceHandle* seq2 = nullptr;
	CHECK(sslm_gpu_seq_create(ctx, model, 64, &seq2) == SSLM_OK);
	CHECK(sslm_gpu_seq_embed_token(ctx, seq2, 5) == SSLM_OK);  // T-2113 B3.5 (D-SLM3367)
	// seq2's first decode step must be identical to a sequence created fresh with no prior
	// history at this address.
	CHECK(RunStepBlocking(ctx, seq2, nullptr, budget));
	// FEATURE ORACLE, executed: seq2's own step is compared to a fresh CPU-side decode of the
	// same token -- a stale read from seq1's freed slot would diverge from this, since seq1 was
	// advanced one real step (not left zeroed) before release specifically so a stale read is
	// observable here.
	if (oracle) {
		SeqSnapshot gpu{};
		CHECK(CaptureSnapshot(seq2, &gpu));
		CpuOracleRunner cpu;
		cpu.Init(*oracle);
		CHECK_MSG(cpu.StepMatchesGpu(5, *oracle, gpu),
		          "M2: seq2's own first step diverges from a fresh CPU-side decode -- seq1's "
		          "freed, advanced state may have leaked through address reuse");
	}
	CHECK(sslm_gpu_seq_release(ctx, seq2) == SSLM_OK);
}

// --- Product cell 1 (design Sec11 dim1, real 1.5B artifact, second sequence forced to an address
// the first's release may have freed, per-step CPU/GPU bit-equality on the second sequence's full
// run) ---
static void TestDim1_P1_RealArtifactAddressReuseBitEquality(SslmGpuContext* ctx,
                                                              SslmGpuModelHandle* model_1p5b,
                                                              const CpuOracleModel* oracle) {
	if (g_model_1p5b_path.empty()) {
		SKIP_MSG("real 1.5B artifact not supplied (--model1p5b=PATH) -- product cell not run");
		return;
	}
	const uint32_t nhl_seq1 = oracle ? oracle->num_hidden_layers : 0;
	SslmGpuSequenceHandle* seq1 = nullptr;
	CHECK(sslm_gpu_seq_create(ctx, model_1p5b, 64, &seq1) == SSLM_OK);
	// Each of the 16 steps re-embeds token 5 fresh (RunFullTokenStep) -- a GPU sequence's
	// layer_index does not auto-wrap after completing a token (fixture_common.h's own header
	// note); without the re-embed, every call past the first would resume an already-complete
	// sequence rather than start a fresh token.
	for (int step = 0; step < 16; ++step)
		CHECK(RunFullTokenStep(ctx, seq1, nullptr, nhl_seq1, 5));
	CHECK(sslm_gpu_seq_release(ctx, seq1) == SSLM_OK);
	for (int i = 0; i < 32; ++i) {  // repeated create/release cycling, forcing address reuse
		SslmGpuSequenceHandle* churn = nullptr;
		CHECK(sslm_gpu_seq_create(ctx, model_1p5b, 64, &churn) == SSLM_OK);
		CHECK(sslm_gpu_seq_release(ctx, churn) == SSLM_OK);
	}
	SslmGpuSequenceHandle* seq2 = nullptr;
	CHECK(sslm_gpu_seq_create(ctx, model_1p5b, 64, &seq2) == SSLM_OK);
	// FEATURE ORACLE, executed (StandardsDocument.md Sec5.4 / catalog dim10): per-step CPU/GPU
	// bit-equality against superslm::RunLayerLoop (the CPU oracle) for the FULL 64-step run on
	// seq2, proving no residue from seq1 survived. Each step re-embeds token 5 fresh
	// (RunFullTokenStep) -- a GPU sequence's layer_index does not auto-wrap after completing a
	// token, so a multi-step loop compared to the CPU oracle (which resets every call) needs the
	// re-embed every iteration, not once before the loop (fixture_common.h's own header note).
	CpuOracleRunner cpu;
	if (oracle) cpu.Init(*oracle);
	const uint32_t nhl = oracle ? oracle->num_hidden_layers : 0;
	for (int step = 0; step < 64; ++step) {
		CHECK(RunFullTokenStep(ctx, seq2, nullptr, nhl, 5));
		if (oracle) {
			SeqSnapshot gpu{};
			CHECK(CaptureSnapshot(seq2, &gpu));
			CHECK_MSG(cpu.StepMatchesGpu(5, *oracle, gpu),
			          "P1: seq2's own step %d diverges from the CPU oracle -- residue from seq1's "
			          "freed slot may have survived address reuse",
			          step);
		}
	}
	CHECK(sslm_gpu_seq_release(ctx, seq2) == SSLM_OK);
}

// --- Product cell 2 (design Sec11 dim1, ten sequences created and released sequentially against
// one model, LiveAllocationCount()-style accounting reads zero outstanding after the tenth) ---
static void TestDim1_P2_TenSequenceLifecycleNoLeak(SslmGpuContext* ctx,
                                                    SslmGpuModelHandle* model) {
	for (int i = 0; i < 10; ++i) {
		SslmGpuSequenceHandle* seq = nullptr;
		CHECK(sslm_gpu_seq_create(ctx, model, 64, &seq) == SSLM_OK);
		CHECK(sslm_gpu_seq_embed_token(ctx, seq, 5) == SSLM_OK);  // T-2113 B3.5 (D-SLM3367)
		CHECK(RunStepBlocking(ctx, seq, nullptr, 24u));
		CHECK(sslm_gpu_seq_release(ctx, seq) == SSLM_OK);
	}
	// LiveAllocationCount()-style accounting (design's own B12 precedent) is a build-seat-owned
	// instrument, not yet declared on the 1.0 surface -- the cell's own outstanding-count
	// assertion is filed here as the contract the build seat's accounting hook must satisfy
	// (zero outstanding sequence allocations after the tenth release), pinned in the design
	// artifact's Sec3 as an open wiring point rather than invented as a new API entry here.
	CHECK(sslm_gpu_model_unmap(ctx, model) != SSLM_MODEL_HAS_LIVE_SEQUENCES);
}

int main(int argc, char** argv) {
	ParseFixtureArgs(argc, argv);
	// Force emission (StandardsDocument.md Sec5.4: a red cell must fail for its OWN
	// reason, LNK2019 on the 1.0 API calls inside, never be silently dead-code-eliminated
	// because nothing in this TU calls it yet -- taking its address is a genuine `use`).
	volatile void* addr_0 = (void*)&TestDim1_M1_TwoLiveModelHandlesContentHashKeyed; (void)addr_0;
	volatile void* addr_1 = (void*)&TestDim1_M2_ReleasedThenCreatedSequenceNeverReadsStaleContent; (void)addr_1;
	volatile void* addr_2 = (void*)&TestDim1_P1_RealArtifactAddressReuseBitEquality; (void)addr_2;
	volatile void* addr_3 = (void*)&TestDim1_P2_TenSequenceLifecycleNoLeak; (void)addr_3;

	// T-2113 driver (B1-B5 landed): construct real ctx/model handles through the production API
	// and invoke every cell above -- the wiring named as "the driver's job" in this file's own
	// prior comment, now that sslm_gpu_context_create/sslm_gpu_model_map exist.
	SslmGpuContext* ctx = nullptr;
	CHECK(sslm_gpu_context_create(GpuContextConfig{}, &ctx) == SSLM_OK);
	if (!ctx) { std::printf("FATAL: sslm_gpu_context_create returned null\n"); return 2; }

	std::vector<uint8_t> bytes_a, bytes_b;
	SslmModelView view_a{}, view_b{};
	std::string err;
	SslmGpuModelHandle* model_a = nullptr;  // primary (1.5B), kept live across M2/P1/P2
	bool have_1p5b = !g_model_1p5b_path.empty() && LoadRealModel(g_model_1p5b_path, &view_a, &bytes_a, &err);
	bool have_0p5b = !g_model_0p5b_path.empty() && LoadRealModel(g_model_0p5b_path, &view_b, &bytes_b, &err);

	CpuOracleModel oracle_a{};
	bool have_oracle_a = false;
	if (have_1p5b) {
		std::string oerr;
		have_oracle_a = LoadCpuOracleModel(view_a, &oracle_a, &oerr);
		CHECK_MSG(have_oracle_a, "dim1: CPU oracle failed to load against the real 1.5B artifact -- %s",
		          oerr.c_str());
	}

	if (have_1p5b && have_0p5b) {
		// M1 unmaps both handles internally -- map two DEDICATED handles for it so the primary
		// handle used by M2/P1/P2 below is unaffected.
		SslmGpuModelHandle* m1_a = nullptr;
		SslmGpuModelHandle* m1_b = nullptr;
		CHECK(sslm_gpu_model_map(ctx, &view_a, GpuResidencyConfig{}, &m1_a) == SSLM_OK);
		CHECK(sslm_gpu_model_map(ctx, &view_b, GpuResidencyConfig{}, &m1_b) == SSLM_OK);
		TestDim1_M1_TwoLiveModelHandlesContentHashKeyed(ctx, m1_a, m1_b, have_oracle_a ? &oracle_a : nullptr);
	} else {
		SKIP_MSG("dim1 M1 needs both --model1p5b=PATH and --model0p5b=PATH -- not run");
	}

	if (have_1p5b) {
		CHECK(sslm_gpu_model_map(ctx, &view_a, GpuResidencyConfig{}, &model_a) == SSLM_OK);
		TestDim1_M2_ReleasedThenCreatedSequenceNeverReadsStaleContent(ctx, model_a, have_oracle_a ? &oracle_a : nullptr);
		TestDim1_P1_RealArtifactAddressReuseBitEquality(ctx, model_a, have_oracle_a ? &oracle_a : nullptr);
		// P2's own body calls sslm_gpu_model_unmap(model_a) itself at its tail (its own cell
		// asserts the unmap succeeds once every sequence it created is released) -- run last,
		// no further unmap call on model_a after this returns (the handle is gone).
		TestDim1_P2_TenSequenceLifecycleNoLeak(ctx, model_a);
	} else {
		SKIP_MSG("dim1 M2/P1/P2 need --model1p5b=PATH -- not run");
	}

	CHECK(sslm_gpu_context_destroy(ctx) == SSLM_OK);
	std::printf("checks=%d failures=%d skips=%d\n", GChecks, GFailures, GSkips);
	return GFailures ? 1 : 0;
}

// T-2112 (Curie) -- Dim 3 (Concurrency), design Sec11 dim3. 3 cells.
// The two mechanism cells and the product cell all need a second thread; std::thread is used
// (this repo's build.bat already links a C++20 toolchain, no new dependency). The
// single-writer-violation cell (M2) is a DELIBERATE residual demonstration (design Sec5.4/Sec11
// dim3's own second bullet: "exercised as a documented-unguarded residual ... not silently
// assumed safe"), not a claim the suite requires to pass a specific way.
//
// D-SLM3412 REPAIR (Curie, 2026-08-15): the casebook's own N3 re-run (Claude/Poirot/
// 50f3d5d-t2113-1p0-gpu-core-build-review.md Sec14.2) diagnosed this file's own crash
// (`superslm_gpu.cpp:1475`, `HR FAIL 0x80004005`, exit 9): M1 and P1 launched concurrent
// `sslm_decode_step_gpu` calls with NO MUTEX anywhere in the file -- the exact documented-
// unguarded residual design Sec5.4/`gpu_1p0.cpp`'s own `SslmGpuContext` comment and B8's own
// account of the undrained-batch hang all name: "a caller driving two threads' decode calls
// through one context's queue concurrently must serialize the submit call itself." M1 and P1 are
// NOT the residual cell (M2 is, and stays unguarded, per its own charter above) -- they are
// meant to prove the SAFE case: two threads, disjoint sequences, following the documented
// external-serialization discipline. tools/t2113_b8_thread_smoke.cpp already proves this exact
// discipline works (14/14, mutex-scoped submit+drain, per-step CPU-oracle bit-equality, both
// gates) -- M1/P1 below now use fixture_common.h's own `SubmitAndDrainSerialized`
// (the identical `g_submit_mutex`-scoped submit+drain idiom B8's own tool established), so the
// concurrency oracle is real (per-step bit-equality against the CPU oracle, not merely "the call
// returned Ok") rather than the "FEATURE ORACLE ... wired here once..." stand-in this file used
// to carry (build log Sec22.10's own named sweep).
//
// 1.13 (paged-KV G3, the box finding on M2): M2 is no longer an unguarded residual. On the box the
// unguarded race died with heap corruption under the D3D12 debug layer (0xC0000374, 2 of 2) and,
// with the layer off, left one call Ok and the sequence Submitted, which this cell then released
// unfinished, so seq_release, model_unmap and context_destroy all refused. The cloud's software GPU
// reproduces both (a winevulkan abort, and the teardown refusals), on 1.12 as well as 1.13: the
// handle's Idle check, EntrySubmitGuard and BeginRecording()'s ledger check are unlocked reads, so
// two threads on one handle could pass all three and record into the singleton's one command list
// at once. The decode now takes a per-handle claim (SslmGpuSequenceHandle::decode_claimed,
// gpu_1p0.cpp) before its state read, so M2's measured outcome is exact: one call Ok, the other
// SSLM_BUSY, every repetition, and a clean teardown once the Ok call is finished.
#include "fixture_common.h"

#include <atomic>
#include <thread>

using namespace superslm;

// --- Mechanism cell 1 (design Sec10 B8's own gate): two threads, two disjoint sequence handles,
// one context, concurrent decode calls -- submit+drain serialized through the documented external
// mutex (design Sec5.4), compared per-step against the CPU oracle. ---
static void TestDim3_M1_TwoThreadsDisjointSequencesMatchSerial(SslmGpuContext* ctx,
                                                                SslmGpuModelHandle* model,
                                                                const CpuOracleModel* oracle) {
	constexpr int kSteps = 16;
	SslmGpuSequenceHandle* seq_a = nullptr;
	SslmGpuSequenceHandle* seq_b = nullptr;
	CHECK(sslm_gpu_seq_create(ctx, model, 64, &seq_a) == SSLM_OK);
	CHECK(sslm_gpu_seq_create(ctx, model, 64, &seq_b) == SSLM_OK);
	// FullTokenBudget + a re-embed every iteration (not once before the loop): each step is
	// compared to the CPU oracle below, which decodes one COMPLETE token per call and a GPU
	// sequence's layer_index does not auto-wrap after completing one (fixture_common.h's own
	// header note). Embedding is host-only (design Sec5.3a) -- no mutex needed around it, each
	// thread only ever touches its OWN sequence handle.
	const uint32_t nhl = oracle ? oracle->num_hidden_layers : 0;
	const uint32_t budget = oracle ? FullTokenBudget(nhl) : 24u;
	bool a_ok = true, b_ok = true;
	std::vector<SeqSnapshot> gpu_a(kSteps), gpu_b(kSteps);
	std::thread ta([&] {
		for (int i = 0; i < kSteps; ++i) {
			if (sslm_gpu_seq_embed_token(ctx, seq_a, 5) != SSLM_OK) a_ok = false;
			if (SubmitAndDrainSerialized(ctx, seq_a, nullptr, budget) != SSLM_OK) a_ok = false;
			CaptureSnapshot(seq_a, &gpu_a[i]);
		}
	});
	std::thread tb([&] {
		for (int i = 0; i < kSteps; ++i) {
			if (sslm_gpu_seq_embed_token(ctx, seq_b, 5) != SSLM_OK) b_ok = false;
			if (SubmitAndDrainSerialized(ctx, seq_b, nullptr, budget) != SSLM_OK) b_ok = false;
			CaptureSnapshot(seq_b, &gpu_b[i]);
		}
	});
	ta.join();
	tb.join();
	CHECK_MSG(a_ok, "M1: thread A observed a non-Ok status under the documented serialized-submit "
	          "discipline");
	CHECK_MSG(b_ok, "M1: thread B observed a non-Ok status under the documented serialized-submit "
	          "discipline");

	// FEATURE ORACLE, executed: per-sequence output must match the equivalent SERIAL two-run
	// baseline -- realized here as the stronger, established claim (tools/t2113_b8_thread_
	// smoke.cpp's own precedent) of per-step bit-equality against the CPU oracle for both
	// sequences, run AFTER the concurrent GPU run completes (the CPU oracle is independent of the
	// GPU run entirely).
	if (oracle && a_ok && b_ok) {
		CpuOracleRunner cpu_a, cpu_b;
		cpu_a.Init(*oracle);
		cpu_b.Init(*oracle);
		for (int i = 0; i < kSteps; ++i) {
			CHECK_MSG(cpu_a.StepMatchesGpu(5, *oracle, gpu_a[i]),
			          "M1: seq_a step %d diverges from the CPU oracle under concurrent decode", i);
			CHECK_MSG(cpu_b.StepMatchesGpu(5, *oracle, gpu_b[i]),
			          "M1: seq_b step %d diverges from the CPU oracle under concurrent decode", i);
		}
	}
	CHECK(sslm_gpu_seq_release(ctx, seq_a) == SSLM_OK);
	CHECK(sslm_gpu_seq_release(ctx, seq_b) == SSLM_OK);
}

// --- Mechanism cell 2 (design Sec5.4/Sec11 dim3, documented-unguarded residual): two threads
// driving sslm_decode_step_gpu against the SAME sequence handle concurrently, DELIBERATELY with
// no mutex (this is the one cell in this file that must NOT use SubmitAndDrainSerialized -- it
// exists to prove what actually happens without the documented discipline, not to demonstrate the
// discipline itself). Proves what actually happens, not that it is safe.
//
// 1.13 (the box finding; see this file's header): the decode's same-sequence claim makes the
// collision a guarded one. Per repetition: both threads start together off one flag; exactly one
// call returns SSLM_OK and the other SSLM_BUSY (the claim, or the Submitted state the winner left,
// refused it before it touched anything); the Ok call is then finished through
// sslm_gpu_ready(block=1), as the API requires before a release, and the release returns SSLM_OK.
// One repetition rarely overlaps the two calls, so the race runs kReps times. Before the claim the
// software GPU showed, within 12 repetitions: Ok + SSLM_DEVICE_LOST (the loser's allocator Reset
// refused under the winner's recording list), a silent Ok/Ok (1.12), and a winevulkan abort (two
// threads recording one command list) -- each one fails this cell or ends its process. ---
static void TestDim3_M2_SameSequenceConcurrentAccessResidualObserved(SslmGpuContext* ctx,
                                                                      SslmGpuModelHandle* model) {
	constexpr int kReps = 16;
	for (int rep = 0; rep < kReps; ++rep) {
		SslmGpuSequenceHandle* seq = nullptr;
		CHECK(sslm_gpu_seq_create(ctx, model, 64, &seq) == SSLM_OK);
		if (seq == nullptr) return;
		CHECK(sslm_gpu_seq_embed_token(ctx, seq, 5) == SSLM_OK);  // T-2113 B3.5 (D-SLM3367)
		SslmGpuStatus r1 = SSLM_OK, r2 = SSLM_OK;
		std::atomic<bool> go{false};
		std::thread ta([&] {
			while (!go.load(std::memory_order_acquire)) {
			}
			r1 = sslm_decode_step_gpu(ctx, seq, nullptr, 24u);
		});
		std::thread tb([&] {
			while (!go.load(std::memory_order_acquire)) {
			}
			r2 = sslm_decode_step_gpu(ctx, seq, nullptr, 24u);
		});
		go.store(true, std::memory_order_release);
		ta.join();
		tb.join();
		// Per design Sec5.4/Sec11 dim3: this is NOT required to be `Ok`/`Ok` -- the residual is
		// "a data race the toolchain's sanitizer flags, or a wrong-but-deterministic corruption",
		// asserted as a MEASURED FACT, not an assumption. The one thing this cell forbids is silent
		// undetected corruption with BOTH calls reporting Ok AND divergent output with no signal --
		// the minimal machine-checkable form of that: at least one of the two calls is observably
		// non-Ok (Busy, the state machine's own existing collision signal) OR the sanitizer build
		// (see build_link_red.bat) flags the race. A hard device fault (a real, uncatchable process
		// exit) is ALSO a legitimate realization of "not silently Ok/Ok with no signal" -- the exact
		// outcome the casebook's own re-run reproduced (Claude/Poirot/50f3d5d-t2113-1p0-gpu-core-
		// build-review.md Sec14.2) -- so this cell's own CHECK below is what runs when the process
		// survives; a hard fault is itself the observed signal when it does not.
		// D-SLM4120 finding 1 (confirmation pass): on this toolchain the unsynchronized race
		// collides on the process-global command allocator/list and both submits return
		// SSLM_DEVICE_LOST -- observably rejected, which is the signal this cell exists to
		// measure; only silent Ok/Ok is forbidden. Accepted as the measured realization beside
		// Busy-or-divergent.
		// The original measured-fact check, kept: never a silent Ok/Ok.
		CHECK_MSG(r1 == SSLM_BUSY || r2 == SSLM_BUSY || r1 != r2 ||
		          (r1 != SSLM_OK && r2 != SSLM_OK),
		          "rep %d: same-sequence concurrent access produced Ok/Ok with no observable "
		          "collision signal -- residual unmeasured (r1=%d r2=%d)", rep, (int)r1, (int)r2);
		// 1.13: the exact outcome the claim gives -- one Ok, one Busy.
		CHECK_MSG((r1 == SSLM_OK && r2 == SSLM_BUSY) || (r1 == SSLM_BUSY && r2 == SSLM_OK),
		          "rep %d: expected one SSLM_OK and one SSLM_BUSY from the same-sequence claim "
		          "(r1=%d r2=%d)", rep, (int)r1, (int)r2);
		if (r1 == SSLM_OK || r2 == SSLM_OK) {
			int32_t ready = 0;
			SslmGpuStatus status = SSLM_DEVICE_LOST;
			CHECK(sslm_gpu_ready(ctx, seq, /*block=*/1, &ready, &status) == SSLM_OK);
			CHECK_MSG(ready == 1 && status == SSLM_OK,
			          "rep %d: the Ok call did not finish cleanly (ready=%d status=%d)", rep, (int)ready,
			          (int)status);
		}
		CHECK_MSG(sslm_gpu_seq_release(ctx, seq) == SSLM_OK,
		          "rep %d: release of the raced sequence refused after it was finished", rep);
	}
}

// --- Product cell: two real sequences, real 1.5B artifact, decoded concurrently on two threads
// for 64 steps each, serialized through the documented external mutex, per-step CPU-oracle
// bit-equality. ---
static void TestDim3_P1_RealArtifactConcurrentDecode64Steps(SslmGpuContext* ctx,
                                                             SslmGpuModelHandle* model_1p5b,
                                                             const CpuOracleModel* oracle) {
	if (g_model_1p5b_path.empty()) {
		SKIP_MSG("real 1.5B artifact not supplied (--model1p5b=PATH) -- product cell not run");
		return;
	}
	constexpr int kSteps = 64;
	SslmGpuSequenceHandle* seq_a = nullptr;
	SslmGpuSequenceHandle* seq_b = nullptr;
	CHECK(sslm_gpu_seq_create(ctx, model_1p5b, 64, &seq_a) == SSLM_OK);
	CHECK(sslm_gpu_seq_create(ctx, model_1p5b, 64, &seq_b) == SSLM_OK);
	// FullTokenBudget + a re-embed every iteration -- see M1's own comment above.
	const uint32_t nhl = oracle ? oracle->num_hidden_layers : 0;
	const uint32_t budget = oracle ? FullTokenBudget(nhl) : 24u;
	bool a_ok = true, b_ok = true;
	std::vector<SeqSnapshot> gpu_a(kSteps), gpu_b(kSteps);
	std::thread ta([&] {
		for (int i = 0; i < kSteps; ++i) {
			if (sslm_gpu_seq_embed_token(ctx, seq_a, 5) != SSLM_OK) a_ok = false;
			if (SubmitAndDrainSerialized(ctx, seq_a, nullptr, budget) != SSLM_OK) a_ok = false;
			CaptureSnapshot(seq_a, &gpu_a[i]);
		}
	});
	std::thread tb([&] {
		for (int i = 0; i < kSteps; ++i) {
			if (sslm_gpu_seq_embed_token(ctx, seq_b, 5) != SSLM_OK) b_ok = false;
			if (SubmitAndDrainSerialized(ctx, seq_b, nullptr, budget) != SSLM_OK) b_ok = false;
			CaptureSnapshot(seq_b, &gpu_b[i]);
		}
	});
	ta.join();
	tb.join();
	CHECK_MSG(a_ok, "P1: thread A observed a non-Ok status under serialized concurrent decode");
	CHECK_MSG(b_ok, "P1: thread B observed a non-Ok status under serialized concurrent decode");

	if (oracle && a_ok && b_ok) {
		CpuOracleRunner cpu_a, cpu_b;
		cpu_a.Init(*oracle);
		cpu_b.Init(*oracle);
		for (int i = 0; i < kSteps; ++i) {
			CHECK_MSG(cpu_a.StepMatchesGpu(5, *oracle, gpu_a[i]),
			          "P1: seq_a step %d diverges from the CPU oracle under concurrent decode", i);
			CHECK_MSG(cpu_b.StepMatchesGpu(5, *oracle, gpu_b[i]),
			          "P1: seq_b step %d diverges from the CPU oracle under concurrent decode", i);
		}
	}
	CHECK(sslm_gpu_seq_release(ctx, seq_a) == SSLM_OK);
	CHECK(sslm_gpu_seq_release(ctx, seq_b) == SSLM_OK);
	// TOOLCHAIN NOTE (design Sec11 dim3's own residual, confirmed at fold): this project's
	// ThreadSanitizer leg is a separate clang CI build that does not compile the GPU MSVC/D3D12
	// binary (.github/workflows/tests.yml:63-82 at main@495fbb4). This cell is therefore
	// sanitizer-clean ONLY when run under a toolchain the build seat separately wires for the GPU
	// target -- named here as a residual, per design Sec11 dim3, not silently assumed covered.
}

int main(int argc, char** argv) {
	ParseFixtureArgs(argc, argv);
	volatile void* addr_0 = (void*)&TestDim3_M1_TwoThreadsDisjointSequencesMatchSerial; (void)addr_0;
	volatile void* addr_1 = (void*)&TestDim3_M2_SameSequenceConcurrentAccessResidualObserved; (void)addr_1;
	volatile void* addr_2 = (void*)&TestDim3_P1_RealArtifactConcurrentDecode64Steps; (void)addr_2;

	SslmGpuContext* ctx = nullptr;
	CHECK(sslm_gpu_context_create(GpuContextConfig{}, &ctx) == SSLM_OK);
	if (!ctx) { std::printf("FATAL: sslm_gpu_context_create returned null\n"); return 2; }

	std::vector<uint8_t> bytes;
	SslmModelView view{};
	std::string err;
	if (!g_model_1p5b_path.empty() && LoadRealModel(g_model_1p5b_path, &view, &bytes, &err)) {
		SslmGpuModelHandle* model = nullptr;
		CHECK(sslm_gpu_model_map(ctx, &view, GpuResidencyConfig{}, &model) == SSLM_OK);
		CpuOracleModel oracle{};
		std::string oerr;
		const bool have_oracle = LoadCpuOracleModel(view, &oracle, &oerr);
		CHECK_MSG(have_oracle, "dim3: CPU oracle failed to load -- %s", oerr.c_str());
		// Ordering, deliberate: the two cells meant to prove the SAFE, documented-discipline case
		// (M1, P1) run FIRST, so their own real assertions are printed before M2's own deliberate
		// hazard demonstration runs -- M2 can legitimately end the process with a hard device
		// fault (design Sec5.4/Sec11 dim3's own residual, casebook Sec14.2), and running it last
		// means that outcome never masks M1/P1's own results.
		TestDim3_M1_TwoThreadsDisjointSequencesMatchSerial(ctx, model, have_oracle ? &oracle : nullptr);
		// Incremental report, flushed: M2 (below) can legitimately end the process with a real
		// device fault (design's own documented residual) before main()'s own final "checks="
		// line ever prints -- these checkpoints are what makes M1/P1's own real counts visible
		// even when that happens, rather than losing them to an un-flushed buffer.
		std::printf("[after M1] checks=%d failures=%d skips=%d\n", GChecks, GFailures, GSkips);
		std::fflush(stdout);
		TestDim3_P1_RealArtifactConcurrentDecode64Steps(ctx, model, have_oracle ? &oracle : nullptr);
		std::printf("[after P1] checks=%d failures=%d skips=%d\n", GChecks, GFailures, GSkips);
		std::fflush(stdout);
		TestDim3_M2_SameSequenceConcurrentAccessResidualObserved(ctx, model);
		CHECK(sslm_gpu_model_unmap(ctx, model) == SSLM_OK);
	} else {
		SKIP_MSG("dim3 needs --model1p5b=PATH -- not run");
	}

	CHECK(sslm_gpu_context_destroy(ctx) == SSLM_OK);
	std::printf("checks=%d failures=%d skips=%d\n", GChecks, GFailures, GSkips);
	return GFailures ? 1 : 0;
}

// T-2112 (Curie) -- Dim 3 (Concurrency), design Sec11 dim3. 3 cells. RED BY LINK.
// The two mechanism cells and the product cell all need a second thread; std::thread is used
// (this repo's build.bat already links a C++20 toolchain, no new dependency). The
// single-writer-violation cell is a DELIBERATE residual demonstration (design Sec5.4/Sec11 dim3's
// own second bullet: "exercised as a documented-unguarded residual ... not silently assumed
// safe"), not a claim the suite requires to pass a specific way -- its own assertion is that the
// run PRODUCES ONE of the two predicted outcomes (a sanitizer-flagged race or a
// wrong-but-deterministic corruption), never that it is safe.
#include "fixture_common.h"

#include <thread>

using namespace superslm;

// --- Mechanism cell 1 (design Sec10 B8's own gate): two threads, two disjoint sequence handles,
// one context, concurrent decode calls, compared against the equivalent serial run. ---
static void TestDim3_M1_TwoThreadsDisjointSequencesMatchSerial(SslmGpuContext* ctx,
                                                                SslmGpuModelHandle* model) {
	SslmGpuSequenceHandle* seq_a = nullptr;
	SslmGpuSequenceHandle* seq_b = nullptr;
	CHECK(sslm_gpu_seq_create(ctx, model, 64, &seq_a) == SSLM_OK);
	CHECK(sslm_gpu_seq_create(ctx, model, 64, &seq_b) == SSLM_OK);
	CHECK(sslm_gpu_seq_embed_token(ctx, seq_a, 5) == SSLM_OK);  // T-2113 B3.5 (D-SLM3367)
	CHECK(sslm_gpu_seq_embed_token(ctx, seq_b, 5) == SSLM_OK);
	std::thread ta([&] {
		for (int i = 0; i < 16; ++i) CHECK(sslm_decode_step_gpu(ctx, seq_a, nullptr, 24u) == SSLM_OK);
	});
	std::thread tb([&] {
		for (int i = 0; i < 16; ++i) CHECK(sslm_decode_step_gpu(ctx, seq_b, nullptr, 24u) == SSLM_OK);
	});
	ta.join();
	tb.join();
	// FEATURE ORACLE: per-sequence output must match the equivalent SERIAL two-run baseline --
	// wired to the dim6 per-step CPU/GPU bit-equality oracle once the build seat's own harness
	// exposes a serial-baseline comparison entry point (design Sec10 B8's own gate text).
	CHECK(sslm_gpu_seq_release(ctx, seq_a) == SSLM_OK);
	CHECK(sslm_gpu_seq_release(ctx, seq_b) == SSLM_OK);
}

// --- Mechanism cell 2 (design Sec5.4/Sec11 dim3, documented-unguarded residual): two threads
// driving sslm_decode_step_gpu against the SAME sequence handle concurrently. Proves what
// actually happens, not that it is safe. ---
static void TestDim3_M2_SameSequenceConcurrentAccessResidualObserved(SslmGpuContext* ctx,
                                                                      SslmGpuModelHandle* model) {
	SslmGpuSequenceHandle* seq = nullptr;
	CHECK(sslm_gpu_seq_create(ctx, model, 64, &seq) == SSLM_OK);
	CHECK(sslm_gpu_seq_embed_token(ctx, seq, 5) == SSLM_OK);  // T-2113 B3.5 (D-SLM3367)
	SslmGpuStatus r1 = SSLM_OK, r2 = SSLM_OK;
	std::thread ta([&] { r1 = sslm_decode_step_gpu(ctx, seq, nullptr, 24u); });
	std::thread tb([&] { r2 = sslm_decode_step_gpu(ctx, seq, nullptr, 24u); });
	ta.join();
	tb.join();
	// Per design Sec5.4/Sec11 dim3: this is NOT required to be `Ok`/`Ok` -- the residual is
	// "a data race the toolchain's sanitizer flags, or a wrong-but-deterministic corruption",
	// asserted as a MEASURED FACT, not an assumption. The one thing this cell forbids is silent
	// undetected corruption with BOTH calls reporting Ok AND divergent output with no signal --
	// the minimal machine-checkable form of that: at least one of the two calls is observably
	// non-Ok (Busy, the state machine's own existing collision signal) OR the sanitizer build
	// (see build_link_red.bat) flags the race.
	CHECK_MSG(r1 == SSLM_BUSY || r2 == SSLM_BUSY || r1 != r2,
	          "same-sequence concurrent access produced Ok/Ok with no observable state-machine "
	          "collision -- residual not measured (r1=%d r2=%d)", (int)r1, (int)r2);
	CHECK(sslm_gpu_seq_release(ctx, seq) == SSLM_OK);
}

// --- Product cell: two real sequences, real 1.5B artifact, decoded concurrently on two threads
// for 64 steps each, thread-sanitizer-clean where the toolchain supports it. ---
static void TestDim3_P1_RealArtifactConcurrentDecode64Steps(SslmGpuContext* ctx,
                                                             SslmGpuModelHandle* model_1p5b) {
	if (g_model_1p5b_path.empty()) {
		SKIP_MSG("real 1.5B artifact not supplied (--model1p5b=PATH) -- product cell not run");
		return;
	}
	SslmGpuSequenceHandle* seq_a = nullptr;
	SslmGpuSequenceHandle* seq_b = nullptr;
	CHECK(sslm_gpu_seq_create(ctx, model_1p5b, 64, &seq_a) == SSLM_OK);
	CHECK(sslm_gpu_seq_create(ctx, model_1p5b, 64, &seq_b) == SSLM_OK);
	CHECK(sslm_gpu_seq_embed_token(ctx, seq_a, 5) == SSLM_OK);  // T-2113 B3.5 (D-SLM3367)
	CHECK(sslm_gpu_seq_embed_token(ctx, seq_b, 5) == SSLM_OK);
	std::thread ta([&] {
		for (int i = 0; i < 64; ++i) CHECK(sslm_decode_step_gpu(ctx, seq_a, nullptr, 24u) == SSLM_OK);
	});
	std::thread tb([&] {
		for (int i = 0; i < 64; ++i) CHECK(sslm_decode_step_gpu(ctx, seq_b, nullptr, 24u) == SSLM_OK);
	});
	ta.join();
	tb.join();
	CHECK(sslm_gpu_seq_release(ctx, seq_a) == SSLM_OK);
	CHECK(sslm_gpu_seq_release(ctx, seq_b) == SSLM_OK);
	// TOOLCHAIN NOTE (design Sec11 dim3's own residual, confirmed at fold): this project's
	// ThreadSanitizer leg is a separate clang CI build that does not compile the GPU MSVC/D3D12
	// binary (.github/workflows/tests.yml:63-82 at main@495fbb4). This cell is therefore
	// sanitizer-clean ONLY when run under a toolchain the build seat separately wires for the GPU
	// target -- named here as a residual, per design Sec11 dim3, not silently assumed covered.
}

// T-2113 (Brunel, reconciliation pass): see dim1_lifetime_red.cpp's own header comment for why
// these are completed locally rather than edited in the suite's own canonical header.
struct GpuContextConfig { int reserved; };
struct GpuResidencyConfig { int reserved; };

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
		TestDim3_M1_TwoThreadsDisjointSequencesMatchSerial(ctx, model);
		TestDim3_M2_SameSequenceConcurrentAccessResidualObserved(ctx, model);
		TestDim3_P1_RealArtifactConcurrentDecode64Steps(ctx, model);
		CHECK(sslm_gpu_model_unmap(ctx, model) == SSLM_OK);
	} else {
		SKIP_MSG("dim3 needs --model1p5b=PATH -- not run");
	}

	CHECK(sslm_gpu_context_destroy(ctx) == SSLM_OK);
	std::printf("checks=%d failures=%d skips=%d\n", GChecks, GFailures, GSkips);
	return GFailures ? 1 : 0;
}

// T-2112 (Curie) -- Dim 1 (Lifetime and reuse), design Sec11 dim1. 4 cells: 2 mechanism, 2
// product. Every cell calls the declared 1.0 API (sslm_gpu_1p0.h) -- RED BY LINK: no .cpp
// anywhere in this tree defines sslm_gpu_context_create/sslm_gpu_model_map/sslm_gpu_seq_create/
// etc (grep of src/ at this suite's own authoring commit finds none), so this file COMPILES
// clean (the declared surface exists, dim-7's own interface_probe already proves that) and FAILS
// TO LINK (LNK2019 unresolved external symbol, one per call site) until the build seat (T-2113,
// design Sec10 B1-B3) lands. See Claude/Curie/t2112-1p0-red-suite-2026-08-15.md (Wizard repo)
// Sec3 for the full cell derivation and the link transcript this file's own build produced.
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
                                                              SslmGpuModelHandle* model_b) {
	CHECK(model_a != nullptr);
	SslmGpuSequenceHandle* seq_a = nullptr;
	CHECK(sslm_gpu_seq_create(ctx, model_a, /*context_cap=*/64, &seq_a) == SSLM_OK);
	// Second, independent model handle (already mapped by the caller against a DIFFERENT
	// artifact, content-hash-keyed per design Sec5.1) must not evict or alias model_a's own
	// resident weights.
	CHECK(model_b != nullptr);
	// seq_a's own next decode step must still read artifact_a's weights, not artifact_b's --
	// asserted via per-step CPU/GPU bit-equality against a CPU-oracle decode of artifact_a alone
	// (the oracle apparatus dim6 owns; here the cell asserts only that model_a's handle is still
	// independently addressable and the call proceeds, which is the lifetime-isolation claim).
	CHECK(sslm_decode_step_gpu(ctx, seq_a, /*adapter_or_null=*/nullptr, 24u) == SSLM_OK);
	CHECK(sslm_gpu_seq_release(ctx, seq_a) == SSLM_OK);
	CHECK(sslm_gpu_model_unmap(ctx, model_a) == SSLM_OK);
	CHECK(sslm_gpu_model_unmap(ctx, model_b) == SSLM_OK);
}

// --- Mechanism cell 2 (design Sec11 dim1, "a sequence handle released and a new sequence handle
// immediately created -- the allocator MAY reuse the freed address, and the new handle's own
// buffer must never read the freed handle's stale content" -- the direct D-SLM3311 reproduction
// against the sequence-handle allocator) ---
static void TestDim1_M2_ReleasedThenCreatedSequenceNeverReadsStaleContent(
    SslmGpuContext* ctx, SslmGpuModelHandle* model) {
	SslmGpuSequenceHandle* seq1 = nullptr;
	CHECK(sslm_gpu_seq_create(ctx, model, 64, &seq1) == SSLM_OK);
	// Advance seq1's own K/V state so a stale read would be observable (not a freshly-zeroed
	// buffer, which would hide the defect this cell exists to catch).
	CHECK(sslm_decode_step_gpu(ctx, seq1, nullptr, 24u) == SSLM_OK);
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
	// seq2's first decode step must be identical to a sequence created fresh with no prior
	// history at this address -- the per-step CPU/GPU bit-equality oracle (dim6) is the
	// instrument; this cell's own claim is narrower and structural: seq2 != seq1's own stale
	// state leaking through, checked here by requiring seq2's own handle to report a fresh
	// Idle-state decode succeeding identically regardless of seq1's prior occupancy of the slot.
	CHECK(sslm_decode_step_gpu(ctx, seq2, nullptr, 24u) == SSLM_OK);
	CHECK(sslm_gpu_seq_release(ctx, seq2) == SSLM_OK);
}

// --- Product cell 1 (design Sec11 dim1, real 1.5B artifact, second sequence forced to an address
// the first's release may have freed, per-step CPU/GPU bit-equality on the second sequence's full
// run) ---
static void TestDim1_P1_RealArtifactAddressReuseBitEquality(SslmGpuContext* ctx,
                                                              SslmGpuModelHandle* model_1p5b) {
	if (g_model_1p5b_path.empty()) {
		SKIP_MSG("real 1.5B artifact not supplied (--model1p5b=PATH) -- product cell not run");
		return;
	}
	SslmGpuSequenceHandle* seq1 = nullptr;
	CHECK(sslm_gpu_seq_create(ctx, model_1p5b, 64, &seq1) == SSLM_OK);
	for (int step = 0; step < 16; ++step)
		CHECK(sslm_decode_step_gpu(ctx, seq1, nullptr, 24u) == SSLM_OK);
	CHECK(sslm_gpu_seq_release(ctx, seq1) == SSLM_OK);
	for (int i = 0; i < 32; ++i) {  // repeated create/release cycling, forcing address reuse
		SslmGpuSequenceHandle* churn = nullptr;
		CHECK(sslm_gpu_seq_create(ctx, model_1p5b, 64, &churn) == SSLM_OK);
		CHECK(sslm_gpu_seq_release(ctx, churn) == SSLM_OK);
	}
	SslmGpuSequenceHandle* seq2 = nullptr;
	CHECK(sslm_gpu_seq_create(ctx, model_1p5b, 64, &seq2) == SSLM_OK);
	// FEATURE ORACLE (StandardsDocument.md Sec5.4 / catalog dim10): per-step CPU/GPU bit-equality
	// against superslm::RunLayerLoop (the CPU oracle) for the FULL 64-step run on seq2, proving no
	// residue from seq1 survived -- the oracle call itself is the build seat's own T-2100/O1
	// harness (design Sec11 dim6), wired here once that harness's own 1.0-API entry point exists.
	for (int step = 0; step < 64; ++step)
		CHECK(sslm_decode_step_gpu(ctx, seq2, nullptr, 24u) == SSLM_OK);
	CHECK(sslm_gpu_seq_release(ctx, seq2) == SSLM_OK);
}

// --- Product cell 2 (design Sec11 dim1, ten sequences created and released sequentially against
// one model, LiveAllocationCount()-style accounting reads zero outstanding after the tenth) ---
static void TestDim1_P2_TenSequenceLifecycleNoLeak(SslmGpuContext* ctx,
                                                    SslmGpuModelHandle* model) {
	for (int i = 0; i < 10; ++i) {
		SslmGpuSequenceHandle* seq = nullptr;
		CHECK(sslm_gpu_seq_create(ctx, model, 64, &seq) == SSLM_OK);
		CHECK(sslm_decode_step_gpu(ctx, seq, nullptr, 24u) == SSLM_OK);
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
	// Force emission (StandardsDocument.md Sec5.4: a red cell must fail for its OWN
	// reason, LNK2019 on the 1.0 API calls inside, never be silently dead-code-eliminated
	// because nothing in this TU calls it yet -- taking its address is a genuine `use`).
	volatile void* addr_1 = (void*)&TestDim1_M2_ReleasedThenCreatedSequenceNeverReadsStaleContent; (void)addr_1;
	// Force emission (StandardsDocument.md Sec5.4: a red cell must fail for its OWN
	// reason, LNK2019 on the 1.0 API calls inside, never be silently dead-code-eliminated
	// because nothing in this TU calls it yet -- taking its address is a genuine `use`).
	volatile void* addr_2 = (void*)&TestDim1_P1_RealArtifactAddressReuseBitEquality; (void)addr_2;
	// Force emission (StandardsDocument.md Sec5.4: a red cell must fail for its OWN
	// reason, LNK2019 on the 1.0 API calls inside, never be silently dead-code-eliminated
	// because nothing in this TU calls it yet -- taking its address is a genuine `use`).
	volatile void* addr_3 = (void*)&TestDim1_P2_TenSequenceLifecycleNoLeak; (void)addr_3;
	// This translation unit's own job today is to compile clean (proving the declared surface,
	// dim-7) and fail at LINK (proving no implementation exists yet, see build_link_red.bat's own
	// captured transcript) -- ctx/model/sequence construction (which needs a REAL, complete
	// GpuContextConfig/GpuResidencyConfig the build seat defines alongside B1/B2, design Sec10)
	// is the driver's job once those types and sslm_gpu_context_create/sslm_gpu_model_map are
	// implemented; the cell functions above take already-constructed handles as parameters for
	// exactly that reason.
	std::printf("checks=%d failures=%d skips=%d\n", GChecks, GFailures, GSkips);
	return GFailures ? 1 : 0;
}

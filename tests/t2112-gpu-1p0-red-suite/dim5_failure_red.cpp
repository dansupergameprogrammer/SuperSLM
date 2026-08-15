// T-2112 (Curie) -- Dim 5 (Failure and rejection paths), design Sec11 dim5. 3 cells. RED BY LINK.
#include "fixture_common.h"

using namespace superslm;

// --- Mechanism cell 1: DeviceLost mid-batch -- a batch of 4 where the 3rd sequence's own
// recorded dispatches trigger a synthetic device-removal, asserted via out_statuses per sequence.
// Mirrors interface_probe/cell_dim5_device_lost_midbatch.c's own construction, promoted here as a
// FULL cell (that probe cell only proves the declaration compiles; this one is the actual product
// assertion, executed once the injection mechanism and the implementation both exist). ---
static void TestDim5_M1_DeviceLostMidBatchPerSequenceOutcomes(
    SslmGpuContext* ctx, SslmGpuSequenceHandle* const* seqs4,
    const SslmGpuAdapterHandle* const* adapters4 /* [4], entry 2 rigged to inject device-removal */) {
	SslmGpuStatus out_statuses[4];
	const SslmGpuStatus call_status =
	    sslm_decode_step_batch_gpu(ctx, seqs4, adapters4, 4u, 24u, out_statuses);
	CHECK(call_status == SSLM_OK || call_status == SSLM_DEVICE_LOST);
	CHECK(out_statuses[2] == SSLM_DEVICE_LOST);
	CHECK_MSG(out_statuses[0] != SSLM_OK && out_statuses[1] != SSLM_OK,
	          "sequences before the device-removal must not report Ok for work whose fence never "
	          "signaled (out_statuses[0]=%d out_statuses[1]=%d)", (int)out_statuses[0],
	          (int)out_statuses[1]);
	CHECK(out_statuses[3] == SSLM_DEVICE_LOST);
	// Recovery: the context must be usable for a fresh sequence afterward (T-2106's own confirmed
	// full-device-recovery, D-SLM3331).
	SslmGpuModelHandle* model = nullptr;  // supplied by the caller once the harness threads model
	                                      // handles through this fixture (build-seat wiring point)
	(void)model;
}

// --- Mechanism cell 2: every status in Sec9's table has its own rejection cell asserted through
// the exact channel Sec9's channel column names. The dim2-shaped ones are covered there;
// SequenceKvBufferMismatch and DeviceLost are this dimension's own home (internal/environmental,
// not adversarial input). SequenceKvBufferMismatch's rejection cell mirrors
// interface_probe/cell_dim5_kv_mismatch.c's own construction (a structural size mismatch,
// "should be unreachable, guarded anyway"). ---
static void TestDim5_M2_SequenceKvBufferMismatchGuarded(SslmGpuContext* ctx,
                                                         SslmGpuModelHandle* model) {
	SslmGpuSequenceHandle* out_seq = nullptr;
	// A context_cap the model's own K/V sizing cannot satisfy without an internal inconsistency
	// (the guard's own precondition, design Sec5.3: "should be unreachable, guarded anyway per
	// the substrate's own GpuGemmSplitSite default-case precedent"). The concrete forcing
	// mechanism is a build-seat-owned injection hook (mirroring O11's own deterministic injection
	// convention, design Sec11 dim5's own citation) -- this cell's own contract is the OBSERVABLE
	// one: whatever the injection mechanism, the guard must fire through this exact channel.
	const SslmGpuStatus st = sslm_gpu_seq_create(ctx, model, /*context_cap=*/-1, &out_seq);
	CHECK(st == SSLM_SEQUENCE_KV_BUFFER_MISMATCH);
	CHECK(out_seq == nullptr);
}

// --- Product cell: the recording-window catch's own cache-invalidation behavior reproduced
// against the new model-handle-owned residency -- a real rejecting call followed immediately by
// a real successful call on the same model handle, proving the handle's own residency state was
// not left inconsistent by the rejection. ---
static void TestDim5_P1_RejectionDoesNotCorruptModelHandleResidency(SslmGpuContext* ctx,
                                                                     SslmGpuModelHandle* model) {
	SslmGpuSequenceHandle* bad_seq = nullptr;
	CHECK(sslm_gpu_seq_create(ctx, model, -1, &bad_seq) == SSLM_SEQUENCE_KV_BUFFER_MISMATCH);
	CHECK(bad_seq == nullptr);
	// Immediately after, a real, valid call against the SAME model handle must succeed exactly as
	// if the rejection above never happened -- the substrate's own T-2062/T-2055 fixed shape
	// (cache-invalidation-on-throw), now checked against model-handle-owned residency rather than
	// the process-global g_resident_weights.
	SslmGpuSequenceHandle* good_seq = nullptr;
	CHECK(sslm_gpu_seq_create(ctx, model, 64, &good_seq) == SSLM_OK);
	CHECK(sslm_decode_step_gpu(ctx, good_seq, nullptr, 24u) == SSLM_OK);
	CHECK(sslm_gpu_seq_release(ctx, good_seq) == SSLM_OK);
}

int main(int argc, char** argv) {
	ParseFixtureArgs(argc, argv);
	// Force emission (StandardsDocument.md Sec5.4: a red cell must fail for its OWN
	// reason, LNK2019 on the 1.0 API calls inside, never be silently dead-code-eliminated
	// because nothing in this TU calls it yet -- taking its address is a genuine `use`).
	volatile void* addr_0 = (void*)&TestDim5_M1_DeviceLostMidBatchPerSequenceOutcomes; (void)addr_0;
	// Force emission (StandardsDocument.md Sec5.4: a red cell must fail for its OWN
	// reason, LNK2019 on the 1.0 API calls inside, never be silently dead-code-eliminated
	// because nothing in this TU calls it yet -- taking its address is a genuine `use`).
	volatile void* addr_1 = (void*)&TestDim5_M2_SequenceKvBufferMismatchGuarded; (void)addr_1;
	// Force emission (StandardsDocument.md Sec5.4: a red cell must fail for its OWN
	// reason, LNK2019 on the 1.0 API calls inside, never be silently dead-code-eliminated
	// because nothing in this TU calls it yet -- taking its address is a genuine `use`).
	volatile void* addr_2 = (void*)&TestDim5_P1_RejectionDoesNotCorruptModelHandleResidency; (void)addr_2;
	std::printf("checks=%d failures=%d skips=%d\n", GChecks, GFailures, GSkips);
	return GFailures ? 1 : 0;
}

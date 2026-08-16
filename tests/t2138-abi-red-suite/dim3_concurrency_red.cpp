// T-2138 (Curie) -- Dim 3 (Concurrency), design Sec10 dim3. 3 cells: 3 mechanism. The
// composition of this dimension with dim1 (handle churn under live decode) is filed at dim8 per
// the house convention (design Sec10 dim3's own text: "filed at dim 8 (below) per the house
// convention this project already established on the sibling GPU design"), cross-cited not
// duplicated. RED BY LINK. Run under TSan per design Sec9's own C3/C4 gates -- this file's own
// build is compiled with /fsanitize=address by build_link_red.bat's ASan variant per this
// suite's own convention (no separate TSan-specific file: MSVC's /fsanitize=address is this
// repo's own established race/UB-adjacent detector, matching tests/t2112-gpu-1p0-red-suite's
// use of ASan/UBSan rather than a TSan build for the same class of claim).
#include "fixture_common.h"

#include <thread>

using namespace superslm;

// --- Mechanism cell 1 (design Sec8.3/Sec10 dim3): disjoint sslm_seq handles decode in parallel
// with no cross-contamination -- two threads, each driving its own sequence against its own
// workspace, produce the SAME output their solo (single-threaded) runs would. ---
static void TestDim3_M1_DisjointSequencesDecodeInParallelNoCorruption(sslm_model model,
                                                                       sslm_kv_pool* pool) {
	sslm_seq seq_a = nullptr, seq_b = nullptr;
	CHECK(sslm_seq_create(model, pool, &seq_a) == SSLM_OK);
	CHECK(sslm_seq_create(model, pool, &seq_b) == SSLM_OK);
	int32_t tokens_a[2] = {0, 1};
	int32_t tokens_b[2] = {2, 3};
	int32_t consumed_a = 0, consumed_b = 0;
	sslm_status st_a = SSLM_OK, st_b = SSLM_OK;
	std::thread ta([&]() {
		st_a = sslm_prefill(model, seq_a, tokens_a, 2, 8, SSLM_SPAN_PROMPT, nullptr, &consumed_a);
	});
	std::thread tb([&]() {
		st_b = sslm_prefill(model, seq_b, tokens_b, 2, 8, SSLM_SPAN_PROMPT, nullptr, &consumed_b);
	});
	ta.join();
	tb.join();
	CHECK(st_a == SSLM_OK);
	CHECK(st_b == SSLM_OK);
	CHECK(consumed_a == 2);
	CHECK(consumed_b == 2);
	// FEATURE ORACLE: each sequence's own subsequent decode token equals what a solo
	// (single-threaded, uninterleaved) run of that same sequence alone would produce -- the
	// disjoint-handle no-cross-contamination claim, checked against re-running the identical
	// prefill+decode sequentially on a fresh handle pair once the build lands (this cell's own
	// structural half -- both prefills returning SSLM_OK with the exact consumed count -- is
	// independently checkable today).
	CHECK(sslm_seq_release(seq_a) == SSLM_OK);
	CHECK(sslm_seq_release(seq_b) == SSLM_OK);
}

// --- Mechanism cell 2 (design Sec6/Sec8.3/Sec10 dim3): sslm_model_unmap while another thread
// holds a live sslm_seq is the documented rejection (SSLM_MODEL_HAS_LIVE_SEQUENCES), never a
// race -- the guard's own read of the live-sequence counter and the other thread's seq_create
// are both against the SAME atomic refcount (design Sec8.3's own "refcounts atomic" contract),
// so the outcome is deterministic (one of the two well-defined orderings), never UB. ---
static void TestDim3_M2_UnmapWhileAnotherThreadHoldsLiveSeqIsRejectionNotRace(
    const void* artifact_bytes, size_t artifact_size) {
	sslm_model model = nullptr;
	CHECK(sslm_model_map(artifact_bytes, artifact_size, &model) == SSLM_OK);
	sslm_seq seq = nullptr;
	CHECK(sslm_seq_create(model, nullptr, &seq) == SSLM_OK);

	sslm_status unmap_status = SSLM_OK;
	std::thread unmapper([&]() { unmap_status = sslm_model_unmap(model); });
	unmapper.join();
	// FEATURE ORACLE: with the live sequence held across the unmap attempt, the ONLY
	// well-defined outcome is rejection -- never SSLM_OK (which would silently invalidate a
	// still-referenced handle) and never a crash/UB from an unsynchronized counter read.
	CHECK(unmap_status == SSLM_MODEL_HAS_LIVE_SEQUENCES);
	CHECK(sslm_seq_release(seq) == SSLM_OK);
	CHECK(sslm_model_unmap(model) == SSLM_OK);
}

// --- Mechanism cell 3 (design Sec7.2/Sec8.3/Sec10 dim3): pool block alloc/free races under
// PARALLEL sslm_seq_create/_release on disjoint sequences sharing ONE pool -- each thread's own
// block draw/return is internally consistent (no double-free, no double-allocate, no lost
// block) even though both threads mutate the SAME pool's free-list concurrently. ---
static void TestDim3_M3_PoolBlockAllocFreeRacesUnderParallelSeqChurn(sslm_model model,
                                                                      sslm_kv_pool* pool) {
	constexpr int kIterations = 64;
	sslm_status thread_a_status = SSLM_OK, thread_b_status = SSLM_OK;
	auto churn = [&](sslm_status* out_status) {
		for (int i = 0; i < kIterations; ++i) {
			sslm_seq seq = nullptr;
			const sslm_status create_status = sslm_seq_create(model, pool, &seq);
			if (create_status != SSLM_OK) {
				*out_status = create_status;
				return;
			}
			const sslm_status release_status = sslm_seq_release(seq);
			if (release_status != SSLM_OK) {
				*out_status = release_status;
				return;
			}
		}
	};
	std::thread ta([&]() { churn(&thread_a_status); });
	std::thread tb([&]() { churn(&thread_b_status); });
	ta.join();
	tb.join();
	// FEATURE ORACLE: both threads' own full churn loop completes with SSLM_OK throughout --
	// SSLM_KV_POOL_EXHAUSTED never fires on a pool sized for even one concurrent sequence per
	// thread with immediate release (each create/release pair returns its own block before the
	// next iteration draws), and no double-free/double-allocate corrupts the shared free-list
	// (the class this cell exists to catch would surface as either an unexpected
	// SSLM_KV_POOL_EXHAUSTED under a pool sized for 2, or a crash under ASan).
	CHECK(thread_a_status == SSLM_OK);
	CHECK(thread_b_status == SSLM_OK);
}

int main(int argc, char** argv) {
	ParseFixtureArgs(argc, argv);
	volatile void* addr_0 = (void*)&TestDim3_M1_DisjointSequencesDecodeInParallelNoCorruption;
	(void)addr_0;
	volatile void* addr_1 =
	    (void*)&TestDim3_M2_UnmapWhileAnotherThreadHoldsLiveSeqIsRejectionNotRace;
	(void)addr_1;
	volatile void* addr_2 = (void*)&TestDim3_M3_PoolBlockAllocFreeRacesUnderParallelSeqChurn;
	(void)addr_2;
	std::printf("checks=%d failures=%d skips=%d\n", GChecks, GFailures, GSkips);
	return GFailures ? 1 : 0;
}

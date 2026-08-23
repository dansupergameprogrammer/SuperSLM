// T-2240/O3 (SuperSLM 1.2.1, plan Sec10 Phase 2 O3; red suite GPU-O3-C1) -- supplementary
// REAL, EXECUTING cell, host-only, no device required (the
// dispatch_geometry_policy_red.cpp shape in this directory). The ADAPTER_U scratch
// region's sizing expression is extracted into the namespace-scope pure helper
// `AdapterURegionBytes` (include/superslm/gpu_port.h, beside kDispatchesPerLayer) and
// consumed by the work_total site (src/gpu/superslm_gpu.cpp), so this cell asserts the
// layout arithmetic at its ONE source of definition.
//
// Claim: for every adapter_rank, the region's byte count is a MULTIPLE OF 4, at least the
// rank itself, and at least the 4-byte floor. The non-multiple-of-4 ranks are the
// discriminating population: the pre-fix formula (`max(4u, rank)` raw) fails ranks 5-7
// (5%4=1, 6%4=2, 7%4=3), so this cell is red for its own stated reason against the staged
// extraction, never vacuously.
#include <cstdio>
#include "superslm/gpu_port.h"

static int GChecks = 0;
static int GFailures = 0;
#define CHECK(cond) \
	do { \
		++GChecks; \
		if (!(cond)) { \
			++GFailures; \
			std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
		} \
	} while (0)
#define CHECK_MSG(cond, ...) \
	do { \
		++GChecks; \
		if (!(cond)) { \
			++GFailures; \
			std::printf("FAIL %s:%d: %s -- ", __FILE__, __LINE__, #cond); \
			std::printf(__VA_ARGS__); \
			std::printf("\n"); \
		} \
	} while (0)

int main() {
	const uint64_t kRanks[] = {1, 2, 3, 5, 6, 7};
	for (uint64_t rank : kRanks) {
		const uint64_t bytes = superslm_gpu::AdapterURegionBytes(rank);
		CHECK_MSG(bytes % 4 == 0,
		          "AdapterURegionBytes(%llu) = %llu is not a multiple of 4 -- the ADAPTER_U "
		          "region leaves a tail over-read open at this rank",
		          static_cast<unsigned long long>(rank),
		          static_cast<unsigned long long>(bytes));
		CHECK_MSG(bytes >= rank,
		          "AdapterURegionBytes(%llu) = %llu is smaller than the rank it must hold",
		          static_cast<unsigned long long>(rank),
		          static_cast<unsigned long long>(bytes));
		CHECK_MSG(bytes >= 4,
		          "AdapterURegionBytes(%llu) = %llu is below the 4-byte resource floor",
		          static_cast<unsigned long long>(rank),
		          static_cast<unsigned long long>(bytes));
	}
	// The floor and already-aligned ranks stay exact: no growth beyond the rounding.
	CHECK(superslm_gpu::AdapterURegionBytes(0) == 4);
	CHECK(superslm_gpu::AdapterURegionBytes(4) == 4);
	CHECK(superslm_gpu::AdapterURegionBytes(8) == 8);
	std::printf("checks=%d failures=%d\n", GChecks, GFailures);
	return GFailures ? 1 : 0;
}

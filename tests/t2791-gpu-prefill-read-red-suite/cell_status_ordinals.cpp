// T-2791 (Curie) -- plan Sec3.4 row 9 (persistence and version evolution): "The existing enum
// ordinals are unchanged (the switch pins every current enumerator)."
//
// Oracle: the v1.5.0 header's own enumerator order (include/superslm/gpu_1p0.h @ 321be46), read
// and transcribed here as literal ordinals, and plan Sec3.1's "Appended LAST to SslmGpuStatus;
// no existing ordinal moves". Neither is produced by the build under test.
//
// Red at v1.5.0 by COMPILE, and only on the two new enumerator names (C2838/C2065 against
// SSLM_OUTPUT_BUFFER_TOO_SMALL / SSLM_PREFILL_HIDDEN_UNAVAILABLE). Every other assertion in this
// file compiles at v1.5.0 and holds there, which is what makes it a pin rather than a restatement.
//
// Mutation proof: inserting either new status anywhere but last moves at least one existing
// ordinal below and breaks its static_assert; appending them in the other order breaks the pair
// of asserts at the end; appending another status breaks the exhaustive switch.
#include <cstdio>
#include <type_traits>

#include "superslm/gpu_1p0.h"

using enum SslmGpuStatus;

static_assert(std::is_same_v<std::underlying_type_t<SslmGpuStatus>, uint32_t>,
              "SslmGpuStatus's underlying type is part of the ABI and stays uint32_t");

#define PIN(name, ordinal) \
	static_assert(static_cast<uint32_t>(SslmGpuStatus::name) == (ordinal), #name " moved")

// The seventeen v1.5.0 ordinals, unchanged.
PIN(SSLM_OK, 0);
PIN(SSLM_DISPATCH_BUDGET_TOO_SMALL, 1);
PIN(SSLM_BUSY, 2);
PIN(SSLM_CONTEXT_HAS_LIVE_HANDLES, 3);
PIN(SSLM_MODEL_HAS_LIVE_SEQUENCES, 4);
PIN(SSLM_ADAPTER_MODEL_MISMATCH, 5);
PIN(SSLM_ADAPTER_BASE_HASH_MISMATCH, 6);
PIN(SSLM_SEQUENCE_KV_BUFFER_MISMATCH, 7);
PIN(SSLM_DEVICE_LOST, 8);
PIN(SSLM_BATCH_BUDGET_EXHAUSTED, 9);
PIN(SSLM_TOKEN_ID_OUT_OF_RANGE, 10);
PIN(SSLM_SEQUENCE_REJECTED, 11);
PIN(SSLM_RESTORE_MODEL_MISMATCH, 12);
PIN(SSLM_MODEL_HAS_LIVE_ADAPTERS, 13);
PIN(SSLM_ADAPTER_HAS_BOUND_SEQUENCES, 14);
PIN(SSLM_GPU_SHADER_BINARY_STALE, 15);
PIN(SSLM_GPU_ALLOCATION_FAILED, 16);

// The two 1.6.0 statuses, appended last and in the order plan Sec3.1 lists them.
PIN(SSLM_OUTPUT_BUFFER_TOO_SMALL, 17);
PIN(SSLM_PREFILL_HIDDEN_UNAVAILABLE, 18);
PIN(SSLM_GPU_SHADER_DIR_INVALID, 19);
PIN(SSLM_GPU_SHADER_DIR_CONFLICT, 20);
PIN(SSLM_GPU_PARALLEL_FOR_INVALID, 21);
PIN(SSLM_GPU_PARALLEL_FOR_INCOMPLETE, 22);
PIN(SSLM_GPU_RESIDENCY_FLAGS_INVALID, 23);

// "Last": a switch over every named enumerator with no default compiles warning-free under
// /W4 /we4062 only if the list below is complete. The appended T-2851 values are 21..23.
#pragma warning(error : 4062)  // enumerator not handled in switch: fails the build here
constexpr int AllNamed(SslmGpuStatus s) {
	switch (s) {
		case SSLM_OK:
		case SSLM_DISPATCH_BUDGET_TOO_SMALL:
		case SSLM_BUSY:
		case SSLM_CONTEXT_HAS_LIVE_HANDLES:
		case SSLM_MODEL_HAS_LIVE_SEQUENCES:
		case SSLM_ADAPTER_MODEL_MISMATCH:
		case SSLM_ADAPTER_BASE_HASH_MISMATCH:
		case SSLM_SEQUENCE_KV_BUFFER_MISMATCH:
		case SSLM_DEVICE_LOST:
		case SSLM_BATCH_BUDGET_EXHAUSTED:
		case SSLM_TOKEN_ID_OUT_OF_RANGE:
		case SSLM_SEQUENCE_REJECTED:
		case SSLM_RESTORE_MODEL_MISMATCH:
		case SSLM_MODEL_HAS_LIVE_ADAPTERS:
		case SSLM_ADAPTER_HAS_BOUND_SEQUENCES:
		case SSLM_GPU_SHADER_BINARY_STALE:
		case SSLM_GPU_ALLOCATION_FAILED:
		case SSLM_OUTPUT_BUFFER_TOO_SMALL:
		case SSLM_PREFILL_HIDDEN_UNAVAILABLE:
		case SSLM_GPU_SHADER_DIR_INVALID:
		case SSLM_GPU_SHADER_DIR_CONFLICT:
		case SSLM_GPU_PARALLEL_FOR_INVALID:
		case SSLM_GPU_PARALLEL_FOR_INCOMPLETE:
		case SSLM_GPU_RESIDENCY_FLAGS_INVALID:
			return 1;
	}
	return 0;
}
static_assert(AllNamed(SSLM_GPU_RESIDENCY_FLAGS_INVALID) == 1, "the switch above names every status");

int main() {
	std::printf("cell_status_ordinals: 24 ordinals pinned at compile time -> PASS\n");
	return 0;
}

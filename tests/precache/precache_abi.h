// Test-side declaration of the E2 CPU ABI. Keep this identical to the public
// declarations when SSLM_HAS_PREFIX_SAVE_ABI is introduced.
#pragma once
#include "superslm/sslm_abi.h"
#include <cstddef>
#include <cstdint>

#define PC_RESTORE_PRODUCER_MISMATCH static_cast<sslm_status>(30)
#ifdef SSLM_HAS_PREFIX_SAVE_ABI
static_assert(SSLM_RESTORE_PRODUCER_MISMATCH == PC_RESTORE_PRODUCER_MISMATCH,
              "producer mismatch must be appended at ordinal 30");
#endif
// Schema skip-ahead on the GPU (1.17.0) appends two names after it; the producer mismatch keeps 30.
static_assert(static_cast<int>(SSLM_GPU_FORCED_TABLE_ABSENT) == 31, "SSLM_GPU_FORCED_TABLE_ABSENT is appended at 31");
static_assert(static_cast<int>(SSLM_GPU_CONTEXT_CAP_EXCEEDED) == 32, "SSLM_GPU_CONTEXT_CAP_EXCEEDED is appended at 32");

extern "C" {
SUPERSLM_API sslm_status sslm_prefix_save(sslm_prefix prefix, void* buf, size_t* n);
SUPERSLM_API sslm_status sslm_prefix_restore(sslm_model model, sslm_kv_pool* pool,
    const void* buf, size_t n, sslm_prefix parent_or_null, sslm_prefix* out,
    uint32_t* out_shared_pages);

// Test-injection-library seams from plan §8. They are absent from the public ABI.
void sslm_pc_test_only_restore_fault_arm(int position, int page_status);
void sslm_pc_test_only_restore_fault_disarm();
int sslm_pc_test_only_restore_fault_read(uint32_t* free_pages, uint32_t* shared,
    uint32_t* mapped, uint32_t* reserve);
sslm_status sslm_pc_test_only_set_prefix_layer(sslm_prefix prefix, uint32_t layer);
uint64_t sslm_pc_test_only_prefix_prefill_calls();
sslm_status sslm_pc_test_only_set_current_token(sslm_prefix prefix, int32_t token);
}

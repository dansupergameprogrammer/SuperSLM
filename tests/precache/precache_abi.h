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

extern "C" {
SUPERSLM_API sslm_status sslm_prefix_save(sslm_prefix prefix, void* buf, size_t* n);
SUPERSLM_API sslm_status sslm_prefix_restore(sslm_model model, sslm_kv_pool* pool,
    const void* buf, size_t n, sslm_prefix parent_or_null, sslm_prefix* out,
    uint32_t* out_shared_pages);
}

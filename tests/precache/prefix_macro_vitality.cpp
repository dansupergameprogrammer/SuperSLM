#include "superslm/sslm_abi.h"

#ifndef SSLM_HAS_PREFIX_SAVE_ABI
#error SSLM_HAS_PREFIX_SAVE_ABI must be published by the E2 public header
#endif

static_assert(int(SSLM_RESTORE_PRODUCER_MISMATCH) == 30,
              "the public producer-mismatch ordinal changed");
// Schema skip-ahead on the GPU (1.17.0) appends two names after it.
static_assert(int(SSLM_GPU_FORCED_TABLE_ABSENT) == 31 && int(SSLM_GPU_CONTEXT_CAP_EXCEEDED) == 32,
              "the GPU skip-ahead statuses are appended at 31 and 32");

int main() { return 0; }

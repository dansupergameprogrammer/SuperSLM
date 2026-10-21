#include "superslm/sslm_abi.h"

#ifndef SSLM_HAS_PREFIX_SAVE_ABI
#error SSLM_HAS_PREFIX_SAVE_ABI must be published by the E2 public header
#endif

static_assert(int(SSLM_RESTORE_PRODUCER_MISMATCH) == 30,
              "the public producer-mismatch ordinal changed");

int main() { return 0; }

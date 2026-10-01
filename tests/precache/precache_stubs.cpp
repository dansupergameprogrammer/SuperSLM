#include "precache_abi.h"

// E1 runtime stubs. E2's public-header macro removes both definitions, so the
// same suite then links to the implementation. No valid status equals 0x7fff.
#ifndef SSLM_HAS_PREFIX_SAVE_ABI
extern "C" sslm_status sslm_prefix_save(sslm_prefix, void*, size_t*) {
    return static_cast<sslm_status>(0x7fff);
}
extern "C" sslm_status sslm_prefix_restore(sslm_model, sslm_kv_pool*, const void*,
    size_t, sslm_prefix, sslm_prefix*, uint32_t*) {
    return static_cast<sslm_status>(0x7fff);
}
extern "C" void sslm_pc_test_only_restore_fault_arm(int, int) {}
extern "C" void sslm_pc_test_only_restore_fault_disarm() {}
extern "C" int sslm_pc_test_only_restore_fault_read(uint32_t*, uint32_t*, uint32_t*, uint32_t*) { return 0; }
extern "C" sslm_status sslm_pc_test_only_set_prefix_layer(sslm_prefix, uint32_t) {
    return static_cast<sslm_status>(0x7fff);
}
extern "C" uint64_t sslm_pc_test_only_prefix_prefill_calls() { return UINT64_MAX; }
extern "C" sslm_status sslm_pc_test_only_set_current_token(sslm_prefix, int32_t) {
    return static_cast<sslm_status>(0x7fff);
}
#endif

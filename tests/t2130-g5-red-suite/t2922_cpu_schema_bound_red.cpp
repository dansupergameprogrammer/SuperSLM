// T-2933 cell cpu_bound_api -- independent compile/link/runtime witness.
#include "superslm/sslm_abi.h"
#include <cstdint>
#include <cstdio>

int main() {
    int32_t out = 0x10203040;
    const sslm_status malformed = sslm_seq_schema_bound(nullptr, &out);
    const bool unchanged = out == 0x10203040;
    const sslm_status null_out = sslm_seq_schema_bound(nullptr, nullptr);
    const bool held = malformed == SSLM_INVALID_ARGUMENT &&
                      null_out == SSLM_INVALID_ARGUMENT && unchanged;
    std::printf("CELL cpu_bound_api malformed=%d null_out=%d unchanged=%d held=%d\n",
                static_cast<int>(malformed), static_cast<int>(null_out), unchanged, held);
    return held ? 0 : 1;
}

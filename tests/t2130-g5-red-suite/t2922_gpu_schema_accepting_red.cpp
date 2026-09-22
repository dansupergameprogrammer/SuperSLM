// T-2933 cell gpu_accepting_api. Separate TU: one absent API cannot mask a sibling.
#include "superslm/gpu_1p0.h"
#include <cstdint>
#include <cstdio>

int main() {
    int32_t out = 0x13572468;
    const SslmGpuStatus malformed =
        SslmGpuSeqSchemaAcceptingForG5Bridge(nullptr, nullptr, &out);
    const bool unchanged = out == 0x13572468;
    const SslmGpuStatus null_out =
        SslmGpuSeqSchemaAcceptingForG5Bridge(nullptr, nullptr, nullptr);
    const bool held = malformed == SslmGpuStatus::SSLM_SEQUENCE_KV_BUFFER_MISMATCH &&
                      null_out == SslmGpuStatus::SSLM_SEQUENCE_KV_BUFFER_MISMATCH && unchanged;
    std::printf("CELL gpu_accepting_api malformed=%d null_out=%d unchanged=%d held=%d\n",
                static_cast<int>(malformed), static_cast<int>(null_out), unchanged, held);
    return held ? 0 : 1;
}

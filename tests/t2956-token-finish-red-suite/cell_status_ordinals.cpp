// T-2851 rows 4 and 9. The shader-directory statuses already occupy 19 and 20.
#include <cstddef>
#include <cstdint>
#include "superslm/gpu_1p0.h"

static_assert(sizeof(GpuResidencyConfig) == 4);
static_assert(offsetof(GpuResidencyConfig, flags) == 0);
static_assert(SSLM_GPU_RESIDENCY_HEAD_ON_DEVICE == 1u);
static_assert(static_cast<uint32_t>(SslmGpuStatus::SSLM_GPU_PARALLEL_FOR_INVALID) == 21);
static_assert(static_cast<uint32_t>(SslmGpuStatus::SSLM_GPU_PARALLEL_FOR_INCOMPLETE) == 22);
static_assert(static_cast<uint32_t>(SslmGpuStatus::SSLM_GPU_RESIDENCY_FLAGS_INVALID) == 23);
static_assert(static_cast<uint32_t>(SslmGpuStatus::SSLM_GPU_SHADER_DIR_INVALID) == 19);
static_assert(static_cast<uint32_t>(SslmGpuStatus::SSLM_GPU_SHADER_DIR_CONFLICT) == 20);

#pragma warning(error: 4062)
constexpr bool AllNamed(SslmGpuStatus status) {
    switch (status) {
        case SslmGpuStatus::SSLM_OK:
        case SslmGpuStatus::SSLM_DISPATCH_BUDGET_TOO_SMALL:
        case SslmGpuStatus::SSLM_BUSY:
        case SslmGpuStatus::SSLM_CONTEXT_HAS_LIVE_HANDLES:
        case SslmGpuStatus::SSLM_MODEL_HAS_LIVE_SEQUENCES:
        case SslmGpuStatus::SSLM_ADAPTER_MODEL_MISMATCH:
        case SslmGpuStatus::SSLM_ADAPTER_BASE_HASH_MISMATCH:
        case SslmGpuStatus::SSLM_SEQUENCE_KV_BUFFER_MISMATCH:
        case SslmGpuStatus::SSLM_DEVICE_LOST:
        case SslmGpuStatus::SSLM_BATCH_BUDGET_EXHAUSTED:
        case SslmGpuStatus::SSLM_TOKEN_ID_OUT_OF_RANGE:
        case SslmGpuStatus::SSLM_SEQUENCE_REJECTED:
        case SslmGpuStatus::SSLM_RESTORE_MODEL_MISMATCH:
        case SslmGpuStatus::SSLM_MODEL_HAS_LIVE_ADAPTERS:
        case SslmGpuStatus::SSLM_ADAPTER_HAS_BOUND_SEQUENCES:
        case SslmGpuStatus::SSLM_GPU_SHADER_BINARY_STALE:
        case SslmGpuStatus::SSLM_GPU_ALLOCATION_FAILED:
        case SslmGpuStatus::SSLM_OUTPUT_BUFFER_TOO_SMALL:
        case SslmGpuStatus::SSLM_PREFILL_HIDDEN_UNAVAILABLE:
        case SslmGpuStatus::SSLM_GPU_SHADER_DIR_INVALID:
        case SslmGpuStatus::SSLM_GPU_SHADER_DIR_CONFLICT:
        case SslmGpuStatus::SSLM_GPU_PARALLEL_FOR_INVALID:
        case SslmGpuStatus::SSLM_GPU_PARALLEL_FOR_INCOMPLETE:
        case SslmGpuStatus::SSLM_GPU_RESIDENCY_FLAGS_INVALID:
            return true;
    }
    return false;
}
static_assert(AllNamed(SslmGpuStatus::SSLM_GPU_RESIDENCY_FLAGS_INVALID));
int main() { return 0; }

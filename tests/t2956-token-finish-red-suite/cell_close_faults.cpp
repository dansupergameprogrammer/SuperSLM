// T-2851 row 5(xiv): phase-B Close retry, for bundle creation and dispatch.
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <vector>
#include "superslm/gpu_1p0.h"

struct SslmGpuDeviceLogitsBundleForTest;
SslmGpuStatus SslmGpuDeviceLogitsBundleCreateForTest(
    SslmGpuContext*, const int8_t*, uint32_t, uint32_t,
    SslmGpuDeviceLogitsBundleForTest**) noexcept;
SslmGpuStatus SslmGpuDeviceLogitsRunForTest(
    SslmGpuContext*, SslmGpuDeviceLogitsBundleForTest*, const int8_t*, int64_t*) noexcept;
void SslmGpuDeviceLogitsBundleDestroyForTest(SslmGpuDeviceLogitsBundleForTest*) noexcept;
void ArmGpuDeviceLogitsCloseFaultForTest(uint32_t) noexcept;

namespace {
constexpr uint32_t V = 65, H = 7;
const std::vector<int8_t> head(V * H, 1);
const std::vector<int8_t> x(H, 2);

bool Create(SslmGpuContext* ctx, SslmGpuDeviceLogitsBundleForTest** out) {
    return SslmGpuDeviceLogitsBundleCreateForTest(ctx, head.data(), V, H, out) ==
        SslmGpuStatus::SSLM_OK && *out;
}
bool Run(SslmGpuContext* ctx, SslmGpuDeviceLogitsBundleForTest* bundle) {
    int64_t row[V]{};
    if (SslmGpuDeviceLogitsRunForTest(ctx, bundle, x.data(), row) != SslmGpuStatus::SSLM_OK)
        return false;
    return std::all_of(std::begin(row), std::end(row), [](int64_t v) { return v == 14; });
}
SslmGpuContext* Context() {
    SslmGpuContext* ctx = nullptr;
    return sslm_gpu_context_create(GpuContextConfig{}, &ctx) == SslmGpuStatus::SSLM_OK ? ctx : nullptr;
}
}

int main() {
    // Must-accept neighbour and first-failure/retry-success path in creation.
    SslmGpuContext* ctx = Context();
    if (!ctx) return 2;
    SslmGpuDeviceLogitsBundleForTest* bundle = nullptr;
    if (!Create(ctx, &bundle) || !Run(ctx, bundle)) return 3;
    SslmGpuDeviceLogitsBundleDestroyForTest(bundle);
    bundle = nullptr;
    ArmGpuDeviceLogitsCloseFaultForTest(1);
    if (SslmGpuDeviceLogitsBundleCreateForTest(ctx, head.data(), V, H, &bundle) !=
            SslmGpuStatus::SSLM_DEVICE_LOST || bundle) return 4;
    if (!Create(ctx, &bundle) || !Run(ctx, bundle)) {
        std::fputs("FAIL Close creation: one failure made same context unusable\n", stderr);
        return 5;
    }

    // The same pair, this time on RunDeviceLogits and on the same bundle.
    int64_t row[V]{};
    ArmGpuDeviceLogitsCloseFaultForTest(1);
    if (SslmGpuDeviceLogitsRunForTest(ctx, bundle, x.data(), row) !=
        SslmGpuStatus::SSLM_DEVICE_LOST) return 6;
    if (!Run(ctx, bundle)) return 7;
    SslmGpuDeviceLogitsBundleDestroyForTest(bundle);
    if (sslm_gpu_context_destroy(ctx) != SslmGpuStatus::SSLM_OK) return 8;

    // Two failed Close attempts leave a terminal context, without hanging.
    ctx = Context();
    if (!ctx) return 9;
    ArmGpuDeviceLogitsCloseFaultForTest(2);
    bundle = nullptr;
    if (SslmGpuDeviceLogitsBundleCreateForTest(ctx, head.data(), V, H, &bundle) !=
            SslmGpuStatus::SSLM_DEVICE_LOST || bundle) return 10;
    if (SslmGpuDeviceLogitsBundleCreateForTest(ctx, head.data(), V, H, &bundle) !=
            SslmGpuStatus::SSLM_DEVICE_LOST || bundle) return 11;
    if (sslm_gpu_context_destroy(ctx) != SslmGpuStatus::SSLM_OK) return 12;

    ctx = Context();
    if (!ctx || !Create(ctx, &bundle)) return 13;
    ArmGpuDeviceLogitsCloseFaultForTest(2);
    if (SslmGpuDeviceLogitsRunForTest(ctx, bundle, x.data(), row) !=
        SslmGpuStatus::SSLM_DEVICE_LOST) return 14;
    if (SslmGpuDeviceLogitsRunForTest(ctx, bundle, x.data(), row) !=
        SslmGpuStatus::SSLM_DEVICE_LOST) return 15;
    SslmGpuDeviceLogitsBundleDestroyForTest(bundle);
    if (sslm_gpu_context_destroy(ctx) != SslmGpuStatus::SSLM_OK) return 16;
    std::puts("PASS Close one-failure recovery and two-failure terminal on create/run");
    return 0;
}

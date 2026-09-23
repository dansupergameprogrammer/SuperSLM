// T-2851 rows 5(viii) and 6: one GPU bundle, exact int64 rows, host narrowing.
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>
#include "superslm/checked_chain_funnel.h"
#include "superslm/forward_sites.h"
#include "superslm/gpu_1p0.h"
#include "superslm/model.h"

// Test-build seam contract from the design's section 4.6. The implementation
// must put matching declarations in gpu_port.h under the fault-injection macro.
struct SslmGpuDeviceLogitsBundleForTest;
SslmGpuStatus SslmGpuDeviceLogitsBundleCreateForTest(
    SslmGpuContext*, const int8_t*, uint32_t, uint32_t,
    SslmGpuDeviceLogitsBundleForTest**) noexcept;
SslmGpuStatus SslmGpuDeviceLogitsRunForTest(
    SslmGpuContext*, SslmGpuDeviceLogitsBundleForTest*, const int8_t*, int64_t*) noexcept;
void SslmGpuDeviceLogitsBundleResourcesForTest(
    const SslmGpuDeviceLogitsBundleForTest*, uint64_t out_gpu_vas[5]) noexcept;
void SslmGpuDeviceLogitsBundleDestroyForTest(SslmGpuDeviceLogitsBundleForTest*) noexcept;
void SslmGpuAllocCounterResetForTest() noexcept;
uint32_t SslmGpuAllocCountForTest() noexcept;

namespace {
std::vector<uint8_t> Read(const char* path) {
    FILE* f = nullptr;
    if (fopen_s(&f, path, "rb") || !f) return {};
    _fseeki64(f, 0, SEEK_END);
    auto n = _ftelli64(f);
    _fseeki64(f, 0, SEEK_SET);
    std::vector<uint8_t> b(static_cast<size_t>(n));
    bool ok = fread(b.data(), 1, b.size(), f) == b.size();
    fclose(f);
    return ok ? b : std::vector<uint8_t>{};
}
bool Same(const uint64_t* a, const uint64_t* b) {
    for (int i = 0; i < 5; ++i) if (a[i] != b[i]) return false;
    return true;
}
bool Run(SslmGpuContext* ctx, const int8_t* head, uint32_t V, uint32_t H, int count) {
    SslmGpuDeviceLogitsBundleForTest* bundle = nullptr;
    if (SslmGpuDeviceLogitsBundleCreateForTest(ctx, head, V, H, &bundle) !=
        SslmGpuStatus::SSLM_OK || !bundle) return false;
    uint64_t before[5]{}, after[5]{};
    SslmGpuDeviceLogitsBundleResourcesForTest(bundle, before);
    std::vector<int8_t> x(H);
    std::vector<int64_t> gpu(V), cpu(V);
    std::vector<int32_t> narrowed(V);
    uint32_t rng = 0x729551u;
    for (int trial = 0; trial < count; ++trial) {
        for (uint32_t k = 0; k < H; ++k) {
            if (trial == 0) x[k] = 127;
            else if (trial == 1) x[k] = k & 1 ? -127 : 127;
            else if (trial == 2) x[k] = -127;
            else {
                rng = rng * 1664525u + 1013904223u;
                x[k] = static_cast<int8_t>(static_cast<int>((rng >> 16) % 255) - 127);
            }
        }
        SslmGpuAllocCounterResetForTest();
        const auto status = SslmGpuDeviceLogitsRunForTest(ctx, bundle, x.data(), gpu.data());
        const auto allocations = SslmGpuAllocCountForTest();
        if (status != SslmGpuStatus::SSLM_OK || allocations != 0) {
            std::fprintf(stderr, "FAIL device run V=%u H=%u trial=%d status=%u allocations=%u\n",
                         V, H, trial, static_cast<unsigned>(status), allocations);
            return false;
        }
        if (superslm::LogitsSite(x.data(), H, head, V, cpu.data(), narrowed.data()) !=
            superslm::SslmForwardStatus::Ok || gpu != cpu) {
            std::fprintf(stderr, "FAIL wide row V=%u H=%u trial=%d\n", V, H, trial);
            return false;
        }
        SslmGpuDeviceLogitsBundleResourcesForTest(bundle, after);
        if (!Same(before, after)) return false;
    }
    SslmGpuDeviceLogitsBundleDestroyForTest(bundle);
    return true;
}
bool Overflow(SslmGpuContext* ctx) {
    constexpr uint32_t V = 4, H = 132112;
    std::vector<int8_t> head(static_cast<size_t>(V) * H, -128);
    std::vector<int8_t> x(H, -127);
    std::vector<int64_t> wide(V);
    std::vector<int32_t> narrow(V);
    SslmGpuDeviceLogitsBundleForTest* bundle = nullptr;
    if (SslmGpuDeviceLogitsBundleCreateForTest(ctx, head.data(), V, H, &bundle) !=
        SslmGpuStatus::SSLM_OK || !bundle) return false;
    uint64_t before[5]{}, after[5]{};
    SslmGpuDeviceLogitsBundleResourcesForTest(bundle, before);
    for (int phase = 0; phase < 3; ++phase) {
        std::fill(x.begin(), x.end(), static_cast<int8_t>(phase == 1 ? 0 : -127));
        SslmGpuAllocCounterResetForTest();
        const auto status = SslmGpuDeviceLogitsRunForTest(ctx, bundle, x.data(), wide.data());
        const auto allocations = SslmGpuAllocCountForTest();
        if (status != SslmGpuStatus::SSLM_OK || allocations != 0) {
            std::fprintf(stderr, "FAIL RO phase=%d status=%u allocations=%u\n", phase,
                         static_cast<unsigned>(status), allocations);
            return false;
        }
        for (int64_t value : wide) {
            const int64_t expected = phase == 1 ? 0 : 2147612672LL;
            if (value != expected) {
                std::fprintf(stderr, "FAIL RO phase=%d wide=%lld want=%lld\n", phase,
                             static_cast<long long>(value), static_cast<long long>(expected));
                return false;
            }
        }
        const auto result = superslm::NarrowRowChecked(wide.data(), V, narrow.data());
        if (result != (phase == 1 ? superslm::SslmForwardStatus::Ok :
                   superslm::SslmForwardStatus::LogitNarrowingOverflow)) return false;
    }
    SslmGpuDeviceLogitsBundleResourcesForTest(bundle, after);
    const bool stable = Same(before, after);
    SslmGpuDeviceLogitsBundleDestroyForTest(bundle);
    return stable;
}
}

int main(int argc, char** argv) {
    if (argc != 4) return 2;
    GpuContextConfig cfg{};
    SslmGpuContext* ctx = nullptr;
    if (sslm_gpu_context_create(cfg, &ctx) != SslmGpuStatus::SSLM_OK) return 3;
    for (int i = 1; i < 4; ++i) {
        auto bytes = Read(argv[i]);
        superslm::SslmModelView model;
        std::string err;
        if (bytes.empty() || superslm::SslmModel::Load(bytes.data(), bytes.size(), model, &err) !=
            superslm::SslmModelStatus::Ok) return 4;
        const auto* head = model.weights.Tensor(model.config.tie_word_embeddings ? "embed" : "lm_head");
        if (!head || !Run(ctx, reinterpret_cast<const int8_t*>(head->data),
                          model.config.vocab_size, model.config.hidden_size, 103)) return 5;
    }
    std::vector<int8_t> byte_path(65 * 7);
    for (size_t i = 0; i < byte_path.size(); ++i)
        byte_path[i] = static_cast<int8_t>(static_cast<int>(i % 251) - 125);
    if (!Run(ctx, byte_path.data(), 65, 7, 3)) return 8;
    if (!Overflow(ctx)) return 6;
    if (sslm_gpu_context_destroy(ctx) != SslmGpuStatus::SSLM_OK) return 7;
    std::printf("PASS device_rows: 103 vectors per artifact; byte path H=7; "
                "RO overflow/zero/overflow one bundle\n");
    return 0;
}

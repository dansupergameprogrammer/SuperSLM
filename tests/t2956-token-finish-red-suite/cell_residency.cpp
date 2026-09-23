// T-2851 rows 1 and 7: flag-clear residency, opt-in head, unmap, and
// the single host copy of tied heads.
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <dxgi1_4.h>
#include <wrl/client.h>
#include "superslm/gpu_1p0.h"
#include "superslm/model.h"

size_t SslmGpuHeadWeightCopySizeForTest(const SslmGpuModelHandle*) noexcept;

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
uint64_t Vram() {
    Microsoft::WRL::ComPtr<IDXGIFactory4> factory;
    if (FAILED(CreateDXGIFactory1(IID_PPV_ARGS(&factory)))) return UINT64_MAX;
    uint64_t best = 0;
    for (UINT i = 0;; ++i) {
        Microsoft::WRL::ComPtr<IDXGIAdapter1> adapter;
        if (factory->EnumAdapters1(i, &adapter) == DXGI_ERROR_NOT_FOUND) break;
        Microsoft::WRL::ComPtr<IDXGIAdapter3> modern;
        if (FAILED(adapter.As(&modern))) continue;
        DXGI_QUERY_VIDEO_MEMORY_INFO info{};
        if (SUCCEEDED(modern->QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &info)))
            best = std::max<uint64_t>(best, info.CurrentUsage);
    }
    return best;
}
constexpr uint64_t Round64K(uint64_t n) { return (n + 65535u) & ~uint64_t(65535u); }
bool One(const char* path, bool tied) {
    auto bytes = Read(path);
    superslm::SslmModelView view;
    std::string err;
    if (bytes.empty() || superslm::SslmModel::Load(bytes.data(), bytes.size(), view, &err) !=
        superslm::SslmModelStatus::Ok || view.config.tie_word_embeddings != tied) return false;
    GpuContextConfig cc{};
    SslmGpuContext* ctx = nullptr;
    if (sslm_gpu_context_create(cc, &ctx) != SslmGpuStatus::SSLM_OK) return false;
    const uint64_t initial = Vram();
    if (initial == UINT64_MAX) return false;
    GpuResidencyConfig clear{};
    SslmGpuModelHandle* model = nullptr;
    if (sslm_gpu_model_map(ctx, &view, clear, &model) != SslmGpuStatus::SSLM_OK) return false;
    const uint64_t clear_vram = Vram();
    const size_t want_copy = tied ? 0 :
        static_cast<size_t>(view.config.vocab_size) * view.config.hidden_size;
    const size_t clear_copy = SslmGpuHeadWeightCopySizeForTest(model);
    if (clear_copy != want_copy) {
        std::fprintf(stderr, "FAIL head copy %s clear=%zu expected=%zu\n",
                     path, clear_copy, want_copy);
        return false;
    }
    sslm_gpu_model_unmap(ctx, model);
    const uint64_t after_clear = Vram();
    if (after_clear > initial + 65536u) return false;
    GpuResidencyConfig flagged{};
#if defined(T2956_BASELINE)
    flagged.reserved = 1; // compile-only check; this cell cannot run on v1.6.0
#else
    flagged.flags = SSLM_GPU_RESIDENCY_HEAD_ON_DEVICE;
#endif
    model = nullptr;
    if (sslm_gpu_model_map(ctx, &view, flagged, &model) != SslmGpuStatus::SSLM_OK) return false;
    const uint64_t flagged_vram = Vram();
    const size_t flagged_copy = SslmGpuHeadWeightCopySizeForTest(model);
    if (flagged_copy != 0) {
        std::fprintf(stderr, "FAIL head copy %s flagged=%zu expected=0\n", path,
                     flagged_copy);
        return false;
    }
    const uint64_t head = static_cast<uint64_t>(view.config.vocab_size) * view.config.hidden_size;
    const uint64_t x = static_cast<uint64_t>(view.config.hidden_size) * 4;
    const uint64_t out = static_cast<uint64_t>(view.config.vocab_size) * 8;
    const uint64_t expected_extra = Round64K(head) + Round64K(x) + Round64K(out);
    const uint64_t got_extra = flagged_vram - initial - (clear_vram - initial);
    if (got_extra + 3 * 65536u < expected_extra ||
        got_extra > expected_extra + 3 * 65536u) {
        std::fprintf(stderr, "FAIL VRAM %s extra=%llu expected=%llu\n", path,
                     static_cast<unsigned long long>(got_extra),
                     static_cast<unsigned long long>(expected_extra));
        return false;
    }
    sslm_gpu_model_unmap(ctx, model);
    const uint64_t after_flagged = Vram();
    if (after_flagged > initial + 3 * 65536u) return false;
    sslm_gpu_context_destroy(ctx);
    std::printf("RESIDENCY %s clear=%llu flagged=%llu extra=%llu expected=%llu\n", path,
                static_cast<unsigned long long>(clear_vram - initial),
                static_cast<unsigned long long>(flagged_vram - initial),
                static_cast<unsigned long long>(got_extra),
                static_cast<unsigned long long>(expected_extra));
    return true;
}
}

int main(int argc, char** argv) {
    if (argc != 4) return 2;
    if (!One(argv[1], true) || !One(argv[2], true) || !One(argv[3], false)) return 3;
    std::printf("PASS tied/untied host copies and flag residency\n");
    return 0;
}

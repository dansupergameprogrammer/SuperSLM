// T-2851 row 5(xiii): count actual allocations for each claim-5 call, then
// fault every occurrence and retry on the same GPU context.
//
// TE-425 (SuperSLM 1.8.0 plan `te421-slm172-host-oom.md` Sec3.5 R1): E_OUTOFMEMORY at ANY counted
// allocation of a handle-creating call, on a device not reported removed, now expects
// SSLM_GPU_ALLOCATION_FAILED -- not only the device-logits bundle's (the non-bundle expectation was
// SSLM_DEVICE_LOST through v1.7.1). The sweep adds sslm_gpu_context_create (its own device setup's
// counted allocation). The two removed-device legs -- DXGI_ERROR_DEVICE_REMOVED returned by the
// allocation (rule 3), and E_OUTOFMEMORY with the classifier's removed-query seam armed (rule 1) --
// run on every kind, not the head map alone, and must read SSLM_DEVICE_LOST. A full (unfocused) run
// counts every nonconforming site and exits non-zero at the end, so its red reading carries a count;
// a focused run (the mutant runner's) still stops at its first failure, with the same messages.
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <dxgi1_4.h>
#include <wrl/client.h>
#include "superslm/gpu_1p0.h"
#include "superslm/model.h"

void SslmGpuAllocCounterResetForTest() noexcept;
uint32_t SslmGpuAllocCountForTest() noexcept;
void ArmGpuAllocFaultAtOccurrence(uint32_t, HRESULT) noexcept;
bool SslmGpuAllocInBundleForTest(uint32_t) noexcept;
void ArmGpuMapDeviceRemovedQueryInjection() noexcept;

namespace {
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
struct Fixture {
    std::vector<uint8_t> model_bytes, adapter_bytes, blob;
    superslm::SslmModelView model_view, adapter_view;
    SslmGpuContext* ctx = nullptr;
    SslmGpuModelHandle* model = nullptr;
    SslmGpuSequenceHandle* seq = nullptr;
    bool Setup(const char* model_path, const char* adapter_path) {
        model_bytes = Read(model_path);
        adapter_bytes = Read(adapter_path);
        std::string err;
        if (model_bytes.empty() || adapter_bytes.empty() ||
            superslm::SslmModel::Load(model_bytes.data(), model_bytes.size(), model_view, &err) !=
                superslm::SslmModelStatus::Ok ||
            superslm::SslmModel::Load(adapter_bytes.data(), adapter_bytes.size(), adapter_view, &err) !=
                superslm::SslmModelStatus::Ok) return false;
        GpuContextConfig cc{};
        GpuResidencyConfig rc{};
        if (sslm_gpu_context_create(cc, &ctx) != SslmGpuStatus::SSLM_OK ||
            sslm_gpu_model_map(ctx, &model_view, rc, &model) != SslmGpuStatus::SSLM_OK ||
            sslm_gpu_seq_create(ctx, model, 64, &seq) != SslmGpuStatus::SSLM_OK) return false;
        size_t n = 0;
        (void)sslm_gpu_seq_save(ctx, seq, nullptr, &n);
        if (!n) return false;
        blob.resize(n);
        return sslm_gpu_seq_save(ctx, seq, blob.data(), &n) == SslmGpuStatus::SSLM_OK;
    }
    SslmGpuStatus Call(int kind, void** created) {
        *created = nullptr;
        switch (kind) {
            case 0: case 1: {
                GpuResidencyConfig rc{};
                if (kind == 1) {
#if defined(T2956_BASELINE)
                    rc.reserved = 1; // compile-only check; no v1.6.0 execution of this cell
#else
                    rc.flags = SSLM_GPU_RESIDENCY_HEAD_ON_DEVICE;
#endif
                }
                SslmGpuModelHandle* h = nullptr;
                auto status = sslm_gpu_model_map(ctx, &model_view, rc, &h);
                *created = h;
                return status;
            }
            case 2: {
                SslmGpuAdapterHandle* h = nullptr;
                auto status = sslm_gpu_adapter_map(ctx, model, &adapter_view, &h);
                *created = h;
                return status;
            }
            case 3: {
                SslmGpuSequenceHandle* h = nullptr;
                auto status = sslm_gpu_seq_create(ctx, model, 64, &h);
                *created = h;
                return status;
            }
            case 5: {
                SslmGpuContext* h = nullptr;
                auto status = sslm_gpu_context_create(GpuContextConfig{}, &h);
                *created = h;
                return status;
            }
            default: {
                SslmGpuSequenceHandle* h = nullptr;
                auto status = sslm_gpu_seq_restore(ctx, model, blob.data(), blob.size(), &h);
                *created = h;
                return status;
            }
        }
    }
    void Release(int kind, void* handle) {
        if (!handle) return;
        if (kind < 2) sslm_gpu_model_unmap(ctx, static_cast<SslmGpuModelHandle*>(handle));
        else if (kind == 2) sslm_gpu_adapter_unmap(ctx, static_cast<SslmGpuAdapterHandle*>(handle));
        else if (kind == 5) sslm_gpu_context_destroy(static_cast<SslmGpuContext*>(handle));
        else sslm_gpu_seq_release(ctx, static_cast<SslmGpuSequenceHandle*>(handle));
    }
};
constexpr int kKinds = 6;
const char* kNames[kKinds] = {"model_clear", "model_head", "adapter", "sequence_create", "sequence_restore",
                              "context_create"};
}

int main(int argc, char** argv) {
    if (argc != 3 && argc != 5) return 2;
    const bool focused = argc == 5;
    int focus_kind = -1;
    uint32_t focus_slot = 0;
    if (focused) {
        for (int i = 0; i < kKinds; ++i)
            if (std::strcmp(argv[3], kNames[i]) == 0) focus_kind = i;
        focus_slot = static_cast<uint32_t>(std::atoi(argv[4]));
        if (focus_kind < 0 || focus_slot == 0) return 2;
    }
    Fixture fx;
    if (!fx.Setup(argv[1], argv[2])) return 3;
    // An unfocused run records every nonconforming site and continues; a focused run (one site, the
    // mutant runner's) stops at its first failure with that failure's exit code.
    uint32_t failures = 0, faulted = 0;
    int first_failure_code = 0;
    auto fail = [&](int code) {
        ++failures;
        if (!first_failure_code) first_failure_code = code;
        return focused;
    };
    std::vector<uint32_t> counts(kKinds, 0);
    std::vector<std::vector<bool>> bundles(kKinds);
    for (int kind = 0; kind < kKinds; ++kind) {
        if (focused && kind != focus_kind) continue;
        // Warm the call before counting. In particular, the first restore initializes
        // the process-wide device and allocates its timestamp buffer once. That
        // allocation is outside restore and must never be the fault target.
        void* warm = nullptr;
        const auto warm_status = fx.Call(kind, &warm);
        fx.Release(kind, warm);
        if (warm_status != SslmGpuStatus::SSLM_OK || !warm) {
            std::fprintf(stderr, "FAIL warm %s status=%u handle=%p\n", kNames[kind],
                         static_cast<unsigned>(warm_status), warm);
            return 10;
        }
        SslmGpuAllocCounterResetForTest();
        void* clean = nullptr;
        const auto clean_status = fx.Call(kind, &clean);
        const auto count = SslmGpuAllocCountForTest();
        std::vector<bool> bundle(count + 1);
        for (uint32_t k = 1; k <= count; ++k) bundle[k] = SslmGpuAllocInBundleForTest(k);
        const uint64_t clean_vram = kind == 1 ? Vram() : 0;
        fx.Release(kind, clean);
        if (clean_status != SslmGpuStatus::SSLM_OK || !clean || count == 0 ||
            (kind == 1 && clean_vram == UINT64_MAX)) {
            std::fprintf(stderr, "FAIL allocation setup %s count=%u status=%u\n", kNames[kind],
                         count, static_cast<unsigned>(clean_status));
            return 4;
        }
        counts[kind] = count;
        bundles[kind] = bundle;
        std::printf("COUNT %s n=%u\n", kNames[kind], count);
        uint32_t selected = focus_slot;
        if (focused && kind == 1) {
            selected = 0;
            uint32_t seen = 0;
            for (uint32_t k = 1; k <= count; ++k)
                if (bundle[k] && ++seen == focus_slot) { selected = k; break; }
        }
        if (focused && (selected == 0 || selected > count)) return 11;
        for (uint32_t k = 1; k <= count; ++k) {
            if (focused && k != selected) continue;
            ++faulted;
            SslmGpuAllocCounterResetForTest();
            ArmGpuAllocFaultAtOccurrence(k, E_OUTOFMEMORY);
            void* failed = nullptr;
            const auto got = fx.Call(kind, &failed);
            // TE-425: every counted allocation, bundle or not (plan Sec3.3; was bundle-only at v1.7.1).
            const auto want = SslmGpuStatus::SSLM_GPU_ALLOCATION_FAILED;
            if (got != want || failed) {
                std::fprintf(stderr, "FAIL %s k=%u got=%u want=%u handle=%p%s\n", kNames[kind],
                             k, static_cast<unsigned>(got), static_cast<unsigned>(want), failed,
                             bundle[k] ? " (device-logits bundle)" : "");
                fx.Release(kind, failed);
                if (fail(5)) return 5;
            }
            SslmGpuAllocCounterResetForTest();
            void* retry = nullptr;
            const auto retry_status = fx.Call(kind, &retry);
            const auto retry_count = SslmGpuAllocCountForTest();
            const uint64_t retry_vram = kind == 1 ? Vram() : 0;
            fx.Release(kind, retry);
            if (retry_status != SslmGpuStatus::SSLM_OK || !retry || retry_count != count) {
                std::fprintf(stderr, "FAIL retry %s k=%u status=%u count=%u expected=%u\n",
                             kNames[kind], k, static_cast<unsigned>(retry_status), retry_count, count);
                if (fail(6)) return 6;
                continue;
            }
            constexpr uint64_t tolerance = 6 * 65536u;
            if (kind == 1 && (retry_vram == UINT64_MAX ||
                retry_vram > clean_vram + tolerance || clean_vram > retry_vram + tolerance)) {
                std::fprintf(stderr, "FAIL model_head retry VRAM k=%u clean=%llu retry=%llu\n",
                             k, static_cast<unsigned long long>(clean_vram),
                             static_cast<unsigned long long>(retry_vram));
                if (fail(12)) return 12;
                continue;
            }
            std::printf("FAULT %s k=%u status=%u retry=0\n", kNames[kind], k,
                        static_cast<unsigned>(got));
        }
    }
    // The removed-device legs, on every kind (TE-425; the head map's alone through v1.7.1). The head
    // map's run first: at v1.7.1 the removed-query seam is consulted only by the map-time bundle, so an
    // arm another kind leaves unconsumed must not reach the head map's leg.
    if (!focused) {
        const int order[kKinds] = {1, 0, 2, 3, 4, 5};
        for (int kind : order) {
            uint32_t site = 1;
            if (kind == 1) {
                site = 0;
                for (uint32_t k = 1, seen = 0; k <= counts[kind]; ++k)
                    if (bundles[kind][k] && ++seen == 2) { site = k; break; }
                if (!site) return 7;
            }
            SslmGpuAllocCounterResetForTest();
            ArmGpuAllocFaultAtOccurrence(site, DXGI_ERROR_DEVICE_REMOVED);
            void* failed = nullptr;
            const auto removed_status = fx.Call(kind, &failed);
            ArmGpuAllocFaultAtOccurrence(0, S_OK);
            if (removed_status != SslmGpuStatus::SSLM_DEVICE_LOST || failed) {
                std::fprintf(stderr, "FAIL removed %s k=%u got=%u expected=8 handle=%p\n", kNames[kind],
                             site, static_cast<unsigned>(removed_status), failed);
                fx.Release(kind, failed);
                fail(8);
            }
            SslmGpuAllocCounterResetForTest();
            ArmGpuMapDeviceRemovedQueryInjection();
            ArmGpuAllocFaultAtOccurrence(site, E_OUTOFMEMORY);
            failed = nullptr;
            const auto query_status = fx.Call(kind, &failed);
            ArmGpuAllocFaultAtOccurrence(0, S_OK);
            if (query_status != SslmGpuStatus::SSLM_DEVICE_LOST || failed) {
                std::fprintf(stderr, "FAIL removed-query %s k=%u got=%u expected=8 handle=%p\n", kNames[kind],
                             site, static_cast<unsigned>(query_status), failed);
                fx.Release(kind, failed);
                fail(9);
            }
            std::printf("FAULT %s k=%u DXGI_ERROR_DEVICE_REMOVED=%u OOM_with_removed_reason=%u\n", kNames[kind],
                        site, static_cast<unsigned>(removed_status), static_cast<unsigned>(query_status));
        }
    }
    if (failures) {
        std::printf("FAIL %u of %u faulted allocations (plus the removed-device legs) did not conform\n",
                    failures, faulted);
        return first_failure_code;
    }
    sslm_gpu_seq_release(fx.ctx, fx.seq);
    sslm_gpu_model_unmap(fx.ctx, fx.model);
    sslm_gpu_context_destroy(fx.ctx);
    std::printf("PASS all counted allocations faulted with same-context retry\n");
    return 0;
}

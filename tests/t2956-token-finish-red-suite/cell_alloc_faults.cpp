// T-2851 row 5(xiii): count actual allocations for each claim-5 call, then
// fault every occurrence and retry on the same GPU context.
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <dxgi.h>
#include "superslm/gpu_1p0.h"
#include "superslm/model.h"

void SslmGpuAllocCounterResetForTest() noexcept;
uint32_t SslmGpuAllocCountForTest() noexcept;
void ArmGpuAllocFaultAtOccurrence(uint32_t, HRESULT) noexcept;
bool SslmGpuAllocInBundleForTest(uint32_t) noexcept;
void ArmGpuMapDeviceRemovedQueryInjection() noexcept;

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
        else sslm_gpu_seq_release(ctx, static_cast<SslmGpuSequenceHandle*>(handle));
    }
};
const char* kNames[] = {"model_clear", "model_head", "adapter", "sequence_create", "sequence_restore"};
}

int main(int argc, char** argv) {
    if (argc != 3 && argc != 5) return 2;
    const bool focused = argc == 5;
    int focus_kind = -1;
    uint32_t focus_slot = 0;
    if (focused) {
        for (int i = 0; i < 5; ++i)
            if (std::strcmp(argv[3], kNames[i]) == 0) focus_kind = i;
        focus_slot = static_cast<uint32_t>(std::atoi(argv[4]));
        if (focus_kind < 0 || focus_slot == 0) return 2;
    }
    Fixture fx;
    if (!fx.Setup(argv[1], argv[2])) return 3;
    for (int kind = 0; kind < 5; ++kind) {
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
        fx.Release(kind, clean);
        if (clean_status != SslmGpuStatus::SSLM_OK || !clean || count == 0) {
            std::fprintf(stderr, "FAIL allocation setup %s count=%u status=%u\n", kNames[kind],
                         count, static_cast<unsigned>(clean_status));
            return 4;
        }
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
            SslmGpuAllocCounterResetForTest();
            ArmGpuAllocFaultAtOccurrence(k, E_OUTOFMEMORY);
            void* failed = nullptr;
            const auto got = fx.Call(kind, &failed);
            const auto want = bundle[k] ? SslmGpuStatus::SSLM_GPU_ALLOCATION_FAILED :
                                          SslmGpuStatus::SSLM_DEVICE_LOST;
            if (got != want || failed) {
                std::fprintf(stderr, "FAIL %s k=%u got=%u want=%u handle=%p\n", kNames[kind],
                             k, static_cast<unsigned>(got), static_cast<unsigned>(want), failed);
                return 5;
            }
            SslmGpuAllocCounterResetForTest();
            void* retry = nullptr;
            const auto retry_status = fx.Call(kind, &retry);
            const auto retry_count = SslmGpuAllocCountForTest();
            fx.Release(kind, retry);
            if (retry_status != SslmGpuStatus::SSLM_OK || !retry || retry_count != count) {
                std::fprintf(stderr, "FAIL retry %s k=%u status=%u count=%u expected=%u\n",
                             kNames[kind], k, static_cast<unsigned>(retry_status), retry_count, count);
                return 6;
            }
            std::printf("FAULT %s k=%u status=%u retry=0\n", kNames[kind], k,
                        static_cast<unsigned>(got));
        }
        if (kind == 1 && !focused) {
            uint32_t second_bundle = 0;
            for (uint32_t k = 1, seen = 0; k <= count; ++k) {
                if (bundle[k] && ++seen == 2) { second_bundle = k; break; }
            }
            if (!second_bundle) return 7;
            SslmGpuAllocCounterResetForTest();
            ArmGpuAllocFaultAtOccurrence(second_bundle, DXGI_ERROR_DEVICE_REMOVED);
            void* failed = nullptr;
            const auto removed_status = fx.Call(kind, &failed);
            if (removed_status != SslmGpuStatus::SSLM_DEVICE_LOST || failed) {
                std::fprintf(stderr, "FAIL removed model_head k=%u got=%u expected=8 handle=%p\n",
                             second_bundle, static_cast<unsigned>(removed_status), failed);
                return 8;
            }
            SslmGpuAllocCounterResetForTest();
            ArmGpuMapDeviceRemovedQueryInjection();
            ArmGpuAllocFaultAtOccurrence(second_bundle, E_OUTOFMEMORY);
            failed = nullptr;
            const auto query_status = fx.Call(kind, &failed);
            if (query_status != SslmGpuStatus::SSLM_DEVICE_LOST || failed) {
                std::fprintf(stderr, "FAIL removed-query model_head k=%u got=%u expected=8 handle=%p\n",
                             second_bundle, static_cast<unsigned>(query_status), failed);
                return 9;
            }
            std::printf("FAULT model_head k=%u DXGI_ERROR_DEVICE_REMOVED=device_loss "
                        "OOM_with_removed_reason=device_loss\n", second_bundle);
        }
    }
    sslm_gpu_seq_release(fx.ctx, fx.seq);
    sslm_gpu_model_unmap(fx.ctx, fx.model);
    sslm_gpu_context_destroy(fx.ctx);
    std::printf("PASS all counted allocations faulted with same-context retry\n");
    return 0;
}

// T-2851 row 2: both hook setters enforce the same field domain.
#include <array>
#include <climits>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>
#include <malloc.h>
#include "superslm/parallel_for.h"
#include "superslm/sslm_abi.h"
#include "superslm/gpu_1p0.h"

extern "C" sslm_status sslm_workspace_set_parallel_for(sslm_workspace, const sslm_parallel_for*);
SslmGpuStatus sslm_gpu_context_set_host_parallel_for(SslmGpuContext*, const sslm_parallel_for*) noexcept;
#if defined(T2956_BASELINE)
constexpr SslmGpuStatus kInvalidHook = static_cast<SslmGpuStatus>(21);
#else
constexpr SslmGpuStatus kInvalidHook = SslmGpuStatus::SSLM_GPU_PARALLEL_FOR_INVALID;
#endif

void Inline(void*, int32_t count, sslm_task_fn task, void* ctx) {
    for (int32_t i = 0; i < count; ++i) task(ctx, i);
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
int main(int argc, char** argv) {
    if (argc != 2) return 2;
    auto file = Read(argv[1]);
    if (file.empty()) return 3;
    void* bytes = _aligned_malloc(file.size(), 64);
    std::memcpy(bytes, file.data(), file.size());
    sslm_model model = nullptr;
    if (sslm_model_map(bytes, file.size(), &model) != SSLM_OK) return 4;
    sslm_decode_params dp{};
    int layers = 0;
    for (int L = 64; L >= 1; --L)
        if (sslm_decode_params_init(model, SSLM_DECODE_MODE_GREEDY, L, &dp) == SSLM_OK) {
            layers = L; break;
        }
    if (!layers) return 5;
    sslm_config config{1, 64, layers, 0};
    size_t capacity = sslm_workspace_size(model, &config);
    void* storage = _aligned_malloc(capacity, 64);
    sslm_workspace ws = nullptr;
    if (sslm_workspace_create(model, &config, storage, capacity, &ws) != SSLM_OK) return 6;
    GpuContextConfig gc{};
    SslmGpuContext* gpu = nullptr;
    if (sslm_gpu_context_create(gc, &gpu) != SslmGpuStatus::SSLM_OK) return 7;
    if (sslm_workspace_set_parallel_for(nullptr, nullptr) != SSLM_INVALID_ARGUMENT ||
        sslm_gpu_context_set_host_parallel_for(nullptr, nullptr) !=
            SslmGpuStatus::SSLM_SEQUENCE_KV_BUFFER_MISMATCH) return 8;
    for (int tasks : std::array<int, 9>{0, 1, 2, 3, 4, 7, 64, 255, 256}) {
        sslm_parallel_for pf{};
        pf.run = &Inline;
        pf.max_tasks = tasks;
        if (sslm_workspace_set_parallel_for(ws, &pf) != SSLM_OK ||
            sslm_gpu_context_set_host_parallel_for(gpu, &pf) != SslmGpuStatus::SSLM_OK) {
            std::fprintf(stderr, "FAIL valid max_tasks=%d\n", tasks);
            return 9;
        }
    }
    for (int tasks : std::array<int, 3>{-1, 257, INT_MAX}) {
        sslm_parallel_for pf{};
        pf.run = &Inline;
        pf.max_tasks = tasks;
        if (sslm_workspace_set_parallel_for(ws, &pf) != SSLM_INVALID_ARGUMENT ||
            sslm_gpu_context_set_host_parallel_for(gpu, &pf) !=
                kInvalidHook) return 10;
    }
    sslm_parallel_for reserved{};
    reserved.run = &Inline;
    reserved.max_tasks = 2;
    reserved.reserved = 1;
    if (sslm_workspace_set_parallel_for(ws, &reserved) != SSLM_INVALID_ARGUMENT ||
        sslm_gpu_context_set_host_parallel_for(gpu, &reserved) !=
            kInvalidHook) return 11;
    sslm_parallel_for no_run{};
    no_run.max_tasks = 2;
    if (sslm_workspace_set_parallel_for(ws, &no_run) != SSLM_INVALID_ARGUMENT ||
        sslm_gpu_context_set_host_parallel_for(gpu, &no_run) !=
            kInvalidHook) return 12;
    if (sslm_workspace_set_parallel_for(ws, nullptr) != SSLM_OK ||
        sslm_gpu_context_set_host_parallel_for(gpu, nullptr) != SslmGpuStatus::SSLM_OK)
        return 13;
    sslm_workspace_destroy(ws);
    sslm_gpu_context_destroy(gpu);
    sslm_model_unmap(model);
    _aligned_free(storage);
    _aligned_free(bytes);
    std::printf("PASS hook setter validation; max_tasks=256 accepted, 257 and INT_MAX refused\n");
    return 0;
}

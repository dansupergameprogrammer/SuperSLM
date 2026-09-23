// T-2851 row 2: each rejected setter preserves a usable serial or installed hook.
#include <array>
#include <climits>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include <malloc.h>
#include "superslm/parallel_for.h"
#include "superslm/sslm_abi.h"
#include "superslm/gpu_1p0.h"
#include "superslm/gpu_1p0_g5_bridge.h"
#include "superslm/gpu_port.h"
#include "superslm/model.h"

extern "C" sslm_status sslm_workspace_set_parallel_for(sslm_workspace,
                                                       const sslm_parallel_for*);
SslmGpuStatus sslm_gpu_context_set_host_parallel_for(SslmGpuContext*,
                                                       const sslm_parallel_for*) noexcept;

namespace {
constexpr int32_t prompt[] = {40, 1035, 1075, 311, 3695, 264, 2820, 60108, 11, 4486, 13};
void Count(void* host, int32_t count, sslm_task_fn task, void* ctx) {
    ++*static_cast<int*>(host);
    for (int32_t i = 0; i < count; ++i) task(ctx, i);
}
std::vector<uint8_t> Read(const char* path) {
    FILE* f = nullptr;
    if (fopen_s(&f, path, "rb") || !f) return {};
    _fseeki64(f, 0, SEEK_END);
    const auto n = _ftelli64(f);
    _fseeki64(f, 0, SEEK_SET);
    std::vector<uint8_t> bytes(static_cast<size_t>(n));
    const bool okay = fread(bytes.data(), 1, bytes.size(), f) == bytes.size();
    fclose(f);
    return okay ? bytes : std::vector<uint8_t>{};
}
int CpuFinish(sslm_model model, sslm_kv_pool* pool, const std::vector<uint8_t>& blob,
              sslm_workspace ws, const sslm_decode_params& params) {
    sslm_seq seq = nullptr;
    if (sslm_seq_restore(model, pool, blob.data(), blob.size(), &seq) != SSLM_OK) return -1;
    int32_t token = -99;
    const auto status = sslm_decode_step_v2(model, &seq, 1, &params, ws, &token);
    sslm_seq_release(seq);
    return status == SSLM_OK ? token : -2;
}
std::vector<uint8_t> GpuSave(SslmGpuContext* ctx, SslmGpuSequenceHandle* seq) {
    size_t n = 0;
    (void)sslm_gpu_seq_save(ctx, seq, nullptr, &n);
    if (!n) return {};
    std::vector<uint8_t> bytes(n);
    if (sslm_gpu_seq_save(ctx, seq, bytes.data(), &n) != SslmGpuStatus::SSLM_OK) return {};
    bytes.resize(n);
    return bytes;
}
int GpuFinish(SslmGpuContext* ctx, SslmGpuModelHandle* model,
              const std::vector<uint8_t>& blob) {
    SslmGpuSequenceHandle* seq = nullptr;
    if (sslm_gpu_seq_restore(ctx, model, blob.data(), blob.size(), &seq) !=
        SslmGpuStatus::SSLM_OK || !seq) return -1;
    int32_t token = -99;
    const auto status = SslmGpuSeqFinishTokenForG5Bridge(ctx, seq, &token);
    sslm_gpu_seq_release(ctx, seq);
    return status == SslmGpuStatus::SSLM_OK ? token : -2;
}
}

int main(int argc, char** argv) {
    if (argc != 2) return 2;
    auto bytes = Read(argv[1]);
    if (bytes.empty()) return 3;
    void* aligned = _aligned_malloc(bytes.size(), 64);
    if (!aligned) return 4;
    std::memcpy(aligned, bytes.data(), bytes.size());
    sslm_model cpu_model = nullptr;
    if (sslm_model_map(aligned, bytes.size(), &cpu_model) != SSLM_OK) return 5;
    superslm::SslmModelView view;
    std::string error;
    if (superslm::SslmModel::Load(bytes.data(), bytes.size(), view, &error) !=
        superslm::SslmModelStatus::Ok) return 6;
    sslm_decode_params params{};
    int layers = 0;
    for (int n = 64; n >= 1; --n)
        if (sslm_decode_params_init(cpu_model, SSLM_DECODE_MODE_GREEDY, n, &params) == SSLM_OK) {
            layers = n;
            break;
        }
    if (!layers) return 7;
    sslm_config config{1, 64, layers, 0};
    const auto pool_bytes = sslm_kv_block_size(cpu_model) + sslm_kv_pool_overhead_size(cpu_model, 1);
    void* pool_storage = _aligned_malloc(pool_bytes, 64);
    sslm_kv_pool pool = nullptr;
    if (sslm_kv_pool_create(cpu_model, pool_storage, pool_bytes, 1, &pool) != SSLM_OK) return 8;
    const auto ws_bytes = sslm_workspace_size(cpu_model, &config);
    void* ws_storage = _aligned_malloc(ws_bytes, 64);
    sslm_workspace ws = nullptr;
    if (sslm_workspace_create(cpu_model, &config, ws_storage, ws_bytes, &ws) != SSLM_OK) return 9;
    sslm_seq source = nullptr;
    if (sslm_seq_create(cpu_model, &pool, &source) != SSLM_OK) return 10;
    int done = 0;
    while (done < static_cast<int>(sizeof prompt / sizeof prompt[0])) {
        int32_t consumed = 0;
        if (sslm_prefill(cpu_model, source, prompt + done,
                static_cast<int32_t>(sizeof prompt / sizeof prompt[0]) - done,
                64, SSLM_SPAN_PROMPT, ws, &consumed) != SSLM_OK || !consumed) return 11;
        done += consumed;
    }
    std::vector<uint8_t> cpu_blob(sslm_seq_state_size(cpu_model));
    size_t written = cpu_blob.size();
    if (sslm_seq_save(source, cpu_blob.data(), &written) != SSLM_OK) return 12;
    cpu_blob.resize(written);
    sslm_seq_release(source);

    SslmGpuContext* ctx = nullptr;
    if (sslm_gpu_context_create(GpuContextConfig{}, &ctx) != SslmGpuStatus::SSLM_OK) return 13;
    SslmGpuModelHandle* gpu_model = nullptr;
    if (sslm_gpu_model_map(ctx, &view, GpuResidencyConfig{}, &gpu_model) !=
        SslmGpuStatus::SSLM_OK) return 14;
    SslmGpuSequenceHandle* gpu_source = nullptr;
    if (sslm_gpu_seq_create(ctx, gpu_model, view.config.context_cap, &gpu_source) !=
        SslmGpuStatus::SSLM_OK) return 15;
    const uint32_t budget = superslm_gpu::kDispatchesPerLayer * view.config.num_hidden_layers;
    if (SslmGpuSeqPrefillPromptForG5Bridge(ctx, gpu_source, prompt,
            static_cast<int32_t>(sizeof prompt / sizeof prompt[0]), budget) !=
        SslmGpuStatus::SSLM_OK) return 16;
    const auto gpu_blob = GpuSave(ctx, gpu_source);
    sslm_gpu_seq_release(ctx, gpu_source);
    if (gpu_blob.empty()) return 17;

    const int cpu_baseline = CpuFinish(cpu_model, &pool, cpu_blob, ws, params);
    const int gpu_baseline = GpuFinish(ctx, gpu_model, gpu_blob);
    if (cpu_baseline < 0 || gpu_baseline != cpu_baseline) return 18;
    int cpu_calls = 0, gpu_calls = 0;
    sslm_parallel_for valid_cpu{}, valid_gpu{};
    valid_cpu.run = valid_gpu.run = &Count;
    valid_cpu.host_ctx = &cpu_calls;
    valid_gpu.host_ctx = &gpu_calls;
    valid_cpu.max_tasks = valid_gpu.max_tasks = 4;
    std::array<sslm_parallel_for, 5> invalid{};
    for (int i = 0; i < 3; ++i) {
        invalid[i].run = &Count;
        invalid[i].host_ctx = &cpu_calls;
    }
    invalid[0].max_tasks = -1;
    invalid[1].max_tasks = 257;
    invalid[2].max_tasks = INT_MAX;
    invalid[3] = valid_cpu;
    invalid[3].reserved = 1;
    invalid[4].max_tasks = 2; // NULL run
    for (int index = 0; index < static_cast<int>(invalid.size()); ++index) {
        const auto* pf = &invalid[index];
        if (sslm_workspace_set_parallel_for(ws, pf) != SSLM_INVALID_ARGUMENT ||
            sslm_gpu_context_set_host_parallel_for(ctx, pf) !=
                SslmGpuStatus::SSLM_GPU_PARALLEL_FOR_INVALID) return 19;
        if (CpuFinish(cpu_model, &pool, cpu_blob, ws, params) != cpu_baseline ||
            GpuFinish(ctx, gpu_model, gpu_blob) != gpu_baseline || cpu_calls || gpu_calls) {
            std::fprintf(stderr, "FAIL rejected setter changed serial decode case=%d\n", index);
            return 20;
        }
        if (sslm_workspace_set_parallel_for(ws, &valid_cpu) != SSLM_OK ||
            sslm_gpu_context_set_host_parallel_for(ctx, &valid_gpu) != SslmGpuStatus::SSLM_OK)
            return 21;
        const int prior_cpu = cpu_calls, prior_gpu = gpu_calls;
        if (sslm_workspace_set_parallel_for(ws, pf) != SSLM_INVALID_ARGUMENT ||
            sslm_gpu_context_set_host_parallel_for(ctx, pf) !=
                SslmGpuStatus::SSLM_GPU_PARALLEL_FOR_INVALID) return 22;
        if (CpuFinish(cpu_model, &pool, cpu_blob, ws, params) != cpu_baseline ||
            GpuFinish(ctx, gpu_model, gpu_blob) != gpu_baseline ||
            cpu_calls <= prior_cpu || gpu_calls <= prior_gpu) {
            std::fprintf(stderr, "FAIL rejected setter replaced installed hook case=%d\n", index);
            return 23;
        }
        if (sslm_workspace_set_parallel_for(ws, nullptr) != SSLM_OK ||
            sslm_gpu_context_set_host_parallel_for(ctx, nullptr) != SslmGpuStatus::SSLM_OK)
            return 24;
        cpu_calls = gpu_calls = 0;
    }
    if (sslm_gpu_model_unmap(ctx, gpu_model) != SslmGpuStatus::SSLM_OK ||
        sslm_gpu_context_destroy(ctx) != SslmGpuStatus::SSLM_OK) return 25;
    sslm_workspace_destroy(ws);
    sslm_kv_pool_destroy(pool);
    sslm_model_unmap(cpu_model);
    _aligned_free(ws_storage);
    _aligned_free(pool_storage);
    _aligned_free(aligned);
    std::printf("PASS refused setter state preserved cases=5 cpu/gpu token=%d\n", cpu_baseline);
    return 0;
}

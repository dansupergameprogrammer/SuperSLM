// T-2851 rows 1 and 9: GPU context hook reuse and hook/flag-free save blobs.
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <new>
#include <string>
#include <vector>
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include "superslm/gpu_1p0.h"
#include "superslm/gpu_1p0_g5_bridge.h"
#include "superslm/gpu_port.h"
#include "superslm/model.h"
#include "superslm/parallel_for.h"

SslmGpuStatus sslm_gpu_context_set_host_parallel_for(SslmGpuContext*,
                                                       const sslm_parallel_for*) noexcept;

namespace {
constexpr int32_t prompt[] = {40, 1035, 1075, 311, 3695, 264, 2820, 60108, 11, 4486, 13};
struct State {
    std::atomic<int> calls{0};
    std::atomic<int> wrong_thread{0};
    DWORD caller = 0;
};
void Run(void* host, int32_t count, sslm_task_fn task, void* ctx) {
    auto& state = *static_cast<State*>(host);
    state.calls.fetch_add(1);
    if (GetCurrentThreadId() != state.caller) state.wrong_thread.fetch_add(1);
    for (int32_t i = count - 1; i >= 0; --i) task(ctx, i);
}
sslm_parallel_for Hook(State* state) {
    sslm_parallel_for result{};
    result.run = &Run;
    result.host_ctx = state;
    result.max_tasks = 4;
    return result;
}
std::vector<uint8_t> Read(const char* path) {
    FILE* f = nullptr;
    if (fopen_s(&f, path, "rb") || !f) return {};
    _fseeki64(f, 0, SEEK_END);
    const auto size = _ftelli64(f);
    _fseeki64(f, 0, SEEK_SET);
    std::vector<uint8_t> bytes(static_cast<size_t>(size));
    const bool okay = fread(bytes.data(), 1, bytes.size(), f) == bytes.size();
    fclose(f);
    return okay ? bytes : std::vector<uint8_t>{};
}
std::vector<uint8_t> Save(SslmGpuContext* ctx, SslmGpuSequenceHandle* seq) {
    size_t size = 0;
    (void)sslm_gpu_seq_save(ctx, seq, nullptr, &size);
    if (!size) return {};
    std::vector<uint8_t> blob(size);
    if (sslm_gpu_seq_save(ctx, seq, blob.data(), &size) != SslmGpuStatus::SSLM_OK) return {};
    blob.resize(size);
    return blob;
}
SslmGpuSequenceHandle* Prefilled(SslmGpuContext* ctx, SslmGpuModelHandle* model,
                                 const superslm::SslmModelView& view) {
    SslmGpuSequenceHandle* seq = nullptr;
    if (sslm_gpu_seq_create(ctx, model, view.config.context_cap, &seq) !=
        SslmGpuStatus::SSLM_OK) return nullptr;
    const uint32_t budget = superslm_gpu::kDispatchesPerLayer * view.config.num_hidden_layers;
    if (SslmGpuSeqPrefillPromptForG5Bridge(ctx, seq, prompt,
            static_cast<int32_t>(sizeof prompt / sizeof prompt[0]), budget) !=
        SslmGpuStatus::SSLM_OK) return nullptr;
    return seq;
}
int FinishRestored(SslmGpuContext* ctx, SslmGpuModelHandle* model,
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
    superslm::SslmModelView view;
    std::string error;
    if (bytes.empty() || superslm::SslmModel::Load(bytes.data(), bytes.size(), view, &error) !=
        superslm::SslmModelStatus::Ok || !view.config.tie_word_embeddings) return 3;
    SslmGpuContext* ctx = nullptr;
    if (sslm_gpu_context_create(GpuContextConfig{}, &ctx) != SslmGpuStatus::SSLM_OK) return 4;
    SslmGpuModelHandle* model = nullptr;
    if (sslm_gpu_model_map(ctx, &view, GpuResidencyConfig{}, &model) !=
        SslmGpuStatus::SSLM_OK) return 5;
    SslmGpuSequenceHandle* seq = Prefilled(ctx, model, view);
    if (!seq) return 6;
    const auto no_hook_blob = Save(ctx, seq);
    if (no_hook_blob.empty()) return 7;
    State state{};
    state.caller = GetCurrentThreadId();
    auto hook = Hook(&state);
    if (sslm_gpu_context_set_host_parallel_for(ctx, &hook) != SslmGpuStatus::SSLM_OK) return 8;
    if (Save(ctx, seq) != no_hook_blob) {
        std::fputs("FAIL GPU hook changed save blob\n", stderr);
        return 9;
    }
    const int hooked = FinishRestored(ctx, model, no_hook_blob);
    if (hooked < 0 || state.calls.load() == 0 || state.wrong_thread.load()) return 10;
    const int calls = state.calls.load();
    if (sslm_gpu_context_set_host_parallel_for(ctx, nullptr) != SslmGpuStatus::SSLM_OK ||
        FinishRestored(ctx, model, no_hook_blob) != hooked || state.calls.load() != calls)
        return 11;
    if (sslm_gpu_context_set_host_parallel_for(ctx, &hook) != SslmGpuStatus::SSLM_OK ||
        FinishRestored(ctx, model, no_hook_blob) != hooked || state.calls.load() <= calls)
        return 12;
    if (sslm_gpu_context_set_host_parallel_for(ctx, nullptr) != SslmGpuStatus::SSLM_OK)
        return 13;
    sslm_gpu_seq_release(ctx, seq);
    if (sslm_gpu_model_unmap(ctx, model) != SslmGpuStatus::SSLM_OK) return 14;

    GpuResidencyConfig device{};
    device.flags = SSLM_GPU_RESIDENCY_HEAD_ON_DEVICE;
    if (sslm_gpu_model_map(ctx, &view, device, &model) != SslmGpuStatus::SSLM_OK) return 15;
    seq = Prefilled(ctx, model, view);
    if (!seq) return 16;
    const auto device_blob = Save(ctx, seq);
    if (device_blob != no_hook_blob) {
        std::fprintf(stderr, "FAIL GPU residency flag changed save blob clear=%zu device=%zu\n",
                     no_hook_blob.size(), device_blob.size());
        return 17;
    }
    if (FinishRestored(ctx, model, device_blob) != hooked) return 18;
    sslm_gpu_seq_release(ctx, seq);
    if (sslm_gpu_model_unmap(ctx, model) != SslmGpuStatus::SSLM_OK) return 19;

    void* page = VirtualAlloc(nullptr, 4096, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
    if (!page) return 20;
    auto* transient = new (page) State{};
    transient->caller = GetCurrentThreadId();
    auto temp_hook = Hook(transient);
    if (sslm_gpu_context_set_host_parallel_for(ctx, &temp_hook) != SslmGpuStatus::SSLM_OK ||
        sslm_gpu_context_set_host_parallel_for(ctx, nullptr) != SslmGpuStatus::SSLM_OK) return 21;
    transient->~State();
    DWORD prior = 0;
    if (!VirtualProtect(page, 4096, PAGE_NOACCESS, &prior)) return 22;
    const auto destroyed = sslm_gpu_context_destroy(ctx);
    VirtualFree(page, 0, MEM_RELEASE);
    if (destroyed != SslmGpuStatus::SSLM_OK) return 23;
    std::printf("PASS GPU hook clear/reinstall/save isolation and flag-free blob token=%d calls=%d\n",
                hooked, state.calls.load());
    return 0;
}

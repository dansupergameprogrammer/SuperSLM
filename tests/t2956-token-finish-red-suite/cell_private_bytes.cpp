// T-2851 row 7: process private bytes after a tied, flag-clear map.
// Build once against v1.6.0 and once against the candidate, then compare.
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <psapi.h>
#include "superslm/gpu_1p0.h"
#include "superslm/model.h"

int main(int argc, char** argv) {
    if (argc != 2) return 2;
    FILE* file = nullptr;
    if (fopen_s(&file, argv[1], "rb") || !file) return 3;
    _fseeki64(file, 0, SEEK_END);
    const auto size = _ftelli64(file);
    _fseeki64(file, 0, SEEK_SET);
    if (size <= 0) return 4;
    std::vector<uint8_t> bytes(static_cast<size_t>(size));
    const bool read = fread(bytes.data(), 1, bytes.size(), file) == bytes.size();
    fclose(file);
    if (!read) return 5;
    superslm::SslmModelView view;
    std::string error;
    if (superslm::SslmModel::Load(bytes.data(), bytes.size(), view, &error) !=
        superslm::SslmModelStatus::Ok || !view.config.tie_word_embeddings) return 6;
    SslmGpuContext* ctx = nullptr;
    if (sslm_gpu_context_create(GpuContextConfig{}, &ctx) != SslmGpuStatus::SSLM_OK) return 7;
    SslmGpuModelHandle* model = nullptr;
    if (sslm_gpu_model_map(ctx, &view, GpuResidencyConfig{}, &model) !=
        SslmGpuStatus::SSLM_OK || !model) return 8;
    PROCESS_MEMORY_COUNTERS_EX memory{};
    memory.cb = sizeof memory;
    if (!GetProcessMemoryInfo(GetCurrentProcess(),
                              reinterpret_cast<PROCESS_MEMORY_COUNTERS*>(&memory),
                              sizeof memory)) return 9;
    const auto head_bytes = static_cast<uint64_t>(view.config.vocab_size) * view.config.hidden_size;
    std::printf("PRIVATE_BYTES %llu HEAD_BYTES %llu\n",
                static_cast<unsigned long long>(memory.PrivateUsage),
                static_cast<unsigned long long>(head_bytes));
    if (sslm_gpu_model_unmap(ctx, model) != SslmGpuStatus::SSLM_OK ||
        sslm_gpu_context_destroy(ctx) != SslmGpuStatus::SSLM_OK) return 10;
    return 0;
}

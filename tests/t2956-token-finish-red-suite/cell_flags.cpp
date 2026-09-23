// T-2851 row 2: each unknown residency bit is refused independently.
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include "superslm/gpu_1p0.h"
#include "superslm/model.h"

int main(int argc, char** argv) {
    if (argc != 2) return 2;
    FILE* f = nullptr;
    if (fopen_s(&f, argv[1], "rb") || !f) return 3;
    _fseeki64(f, 0, SEEK_END);
    const auto n = _ftelli64(f);
    _fseeki64(f, 0, SEEK_SET);
    std::vector<uint8_t> bytes(static_cast<size_t>(n));
    if (fread(bytes.data(), 1, bytes.size(), f) != bytes.size()) return 4;
    fclose(f);
    superslm::SslmModelView view;
    std::string err;
    if (superslm::SslmModel::Load(bytes.data(), bytes.size(), view, &err) !=
        superslm::SslmModelStatus::Ok) return 5;
    GpuContextConfig cfg{};
    SslmGpuContext* ctx = nullptr;
    if (sslm_gpu_context_create(cfg, &ctx) != SslmGpuStatus::SSLM_OK) return 6;
    for (unsigned bit = 1; bit < 32; ++bit) {
        GpuResidencyConfig rc{};
        rc.flags = 1u << bit;
        SslmGpuModelHandle* handle = nullptr;
        const auto got = sslm_gpu_model_map(ctx, &view, rc, &handle);
        if (got != SslmGpuStatus::SSLM_GPU_RESIDENCY_FLAGS_INVALID || handle != nullptr) {
            std::fprintf(stderr, "FAIL unknown bit %u status=%u handle=%p\n",
                         bit, static_cast<unsigned>(got), static_cast<void*>(handle));
            return 7;
        }
    }
    GpuResidencyConfig clear{};
    SslmGpuModelHandle* healthy = nullptr;
    if (sslm_gpu_model_map(ctx, &view, clear, &healthy) != SslmGpuStatus::SSLM_OK || !healthy)
        return 8;
    sslm_gpu_model_unmap(ctx, healthy);
    sslm_gpu_context_destroy(ctx);
    std::printf("PASS unknown_flags bits=1..31 refused; clear accepted\n");
    return 0;
}

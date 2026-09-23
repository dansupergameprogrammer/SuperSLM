// T-2851 row 5: a flag-set map with logits_site.cso absent refuses cleanly.
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>
#include "superslm/gpu_1p0.h"
#include "superslm/model.h"

int main(int argc, char** argv) {
    if (argc != 3) return 2;
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
        superslm::SslmModelStatus::Ok) return 6;
    GpuContextConfig config{};
    config.shader_dir = argv[2];
    SslmGpuContext* ctx = nullptr;
    if (sslm_gpu_context_create(config, &ctx) != SslmGpuStatus::SSLM_OK) return 7;
    GpuResidencyConfig residence{};
    residence.flags = SSLM_GPU_RESIDENCY_HEAD_ON_DEVICE;
    SslmGpuModelHandle* handle = nullptr;
    const auto status = sslm_gpu_model_map(ctx, &view, residence, &handle);
    if (status != SslmGpuStatus::SSLM_DEVICE_LOST || handle) {
        std::fprintf(stderr, "FAIL missing logits_site.cso status=%u handle=%p\n",
                     static_cast<unsigned>(status), handle);
        return 8;
    }
    residence.flags = 0;
    if (sslm_gpu_model_map(ctx, &view, residence, &handle) != SslmGpuStatus::SSLM_OK ||
        !handle) return 9;
    if (sslm_gpu_model_unmap(ctx, handle) != SslmGpuStatus::SSLM_OK ||
        sslm_gpu_context_destroy(ctx) != SslmGpuStatus::SSLM_OK) return 10;
    std::puts("PASS missing logits_site.cso refused; same-context flag-clear retry succeeded");
    return 0;
}

#include "../t2791-gpu-prefill-read-red-suite/cpu_oracle.h"
#include <cstdio>

int main(int argc, char** argv) {
    if (argc != 3) return 2;
    const uint32_t target = static_cast<uint32_t>(std::stoul(argv[2]));
    SslmGpuContext* ctx = nullptr;
    if (sslm_gpu_context_create(GpuContextConfig{}, &ctx) != SSLM_OK || !ctx) return 2;
    GpuModelFixture fx;
    if (!fx.Open(argv[1], ctx)) return 2;
    CpuOracle cpu(fx);
    int32_t chosen = -1;
    for (int32_t token = 0; token < fx.vocab; ++token) {
        const CpuResult result = cpu.Run({token});
        if (result.st != superslm::SslmForwardStatus::Ok &&
            result.refused_index == 0 && result.refused_layer == target) {
            chosen = token;
            break;
        }
    }
    if (chosen < 0) return 2;
    SslmGpuSequenceHandle* seq = nullptr;
    if (sslm_gpu_seq_create(ctx, fx.model, 64, &seq) != SSLM_OK || !seq) return 2;
    const std::vector<int32_t> ids = {chosen};
    const SslmGpuStatus first = Prefill(fx, seq, ids);
    const uint32_t before = LayerIndex(seq);
    const SslmGpuStatus reset = sslm_gpu_seq_reset(ctx, seq);
    const uint32_t after = LayerIndex(seq);
    const SslmGpuStatus reused = Prefill(fx, seq, ids);
    const uint32_t after_reuse = LayerIndex(seq);
    const bool held = first == SSLM_SEQUENCE_REJECTED && before == target &&
        reset == SSLM_OK && after == 0 && reused == SSLM_SEQUENCE_REJECTED &&
        after_reuse == target;
    std::printf("CELL gpu_idle_reset layer=%u first=%s before=%u reset=%s after=%u "
                "reuse=%s reused_layer=%u held=%d\n", target, StatusName(first), before,
                StatusName(reset), after, StatusName(reused), after_reuse, held);
    sslm_gpu_seq_release(ctx, seq);
    fx.Close();
    sslm_gpu_context_destroy(ctx);
    return held ? 0 : 1;
}

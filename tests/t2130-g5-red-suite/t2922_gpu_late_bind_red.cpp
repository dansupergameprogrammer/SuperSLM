#include "../t2791-gpu-prefill-read-red-suite/fixture_common.h"
#include <cstdio>
#include <vector>

namespace {
bool DecodeOne(const GpuModelFixture& fx, SslmGpuSequenceHandle* seq) {
    int32_t token = -999;
    return SslmGpuSeqDecodeStepForG5Bridge(
               fx.ctx, seq, 7, fx.one_layer_budget, &token) == SSLM_OK && token >= 0;
}
}

int main(int argc, char** argv) {
    if (argc != 2) return 2;
    SslmGpuContext* ctx = nullptr;
    if (sslm_gpu_context_create(GpuContextConfig{}, &ctx) != SSLM_OK || !ctx) return 2;
    GpuModelFixture fx;
    if (!fx.Open(argv[1], ctx)) return 2;

    SslmGpuSequenceHandle* fresh = nullptr;
    SslmGpuSequenceHandle* live = nullptr;
    SslmGpuSequenceHandle* original = nullptr;
    if (sslm_gpu_seq_create(ctx, fx.model, fx.model_cap, &fresh) != SSLM_OK ||
        sslm_gpu_seq_create(ctx, fx.model, fx.model_cap, &live) != SSLM_OK ||
        sslm_gpu_seq_create(ctx, fx.model, fx.model_cap, &original) != SSLM_OK) return 2;

    const SslmGpuStatus fresh_bind = SslmGpuSeqSetSchemaForG5Bridge(ctx, fresh, 0);
    const bool live_decode = DecodeOne(fx, live);
    const SslmGpuStatus live_bind = SslmGpuSeqSetSchemaForG5Bridge(ctx, live, 0);
    const SslmGpuStatus live_reset = sslm_gpu_seq_reset(ctx, live);
    const SslmGpuStatus reset_bind = SslmGpuSeqSetSchemaForG5Bridge(ctx, live, 0);

    const bool restored_decode = DecodeOne(fx, original);
    size_t need = 0;
    const SslmGpuStatus measure = sslm_gpu_seq_save(ctx, original, nullptr, &need);
    std::vector<uint8_t> blob(need);
    size_t written = need;
    const SslmGpuStatus save = sslm_gpu_seq_save(ctx, original, blob.data(), &written);
    SslmGpuSequenceHandle* restored = nullptr;
    const SslmGpuStatus restore = sslm_gpu_seq_restore(
        ctx, fx.model, blob.data(), written, &restored);
    const SslmGpuStatus restored_bind = restored
        ? SslmGpuSeqSetSchemaForG5Bridge(ctx, restored, 0) : SSLM_DEVICE_LOST;

    const bool held = fresh_bind == SSLM_OK && live_decode &&
        live_bind == SSLM_SEQUENCE_REJECTED && live_reset == SSLM_OK &&
        reset_bind == SSLM_OK && restored_decode &&
        measure == SSLM_DEVICE_LOST && need > 0 && save == SSLM_OK && restore == SSLM_OK &&
        restored_bind == SSLM_SEQUENCE_REJECTED;
    std::printf("CELL gpu_late_bind fresh=%s live=%s reset=%s reset_bind=%s "
                "restore=%s restored_bind=%s held=%d\n",
                StatusName(fresh_bind), StatusName(live_bind), StatusName(live_reset),
                StatusName(reset_bind), StatusName(restore), StatusName(restored_bind), held);

    sslm_gpu_seq_release(ctx, fresh);
    sslm_gpu_seq_release(ctx, live);
    sslm_gpu_seq_release(ctx, original);
    if (restored) sslm_gpu_seq_release(ctx, restored);
    fx.Close();
    sslm_gpu_context_destroy(ctx);
    return held ? 0 : 1;
}

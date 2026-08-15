/* Design Sec9: `SequenceKvBufferMismatch` -- "a sequence's own K/V buffer
 * size does not match its own `context_cap` at create time ... guarded
 * anyway". Sec11 dim 5 requires every status in Sec9's table to have its
 * own rejection cell. REPAIRED at fold (T-2111 rung 5) for the out_seq
 * out-parameter the repaired signature (design Sec5.3) adds. */
#include "sslm_gpu_1p0.h"
int cell(SslmGpuContext* ctx, SslmGpuModelHandle* model) {
    SslmGpuSequenceHandle* out_seq = 0;
    SslmGpuStatus st = sslm_gpu_seq_create(ctx, model, 4096, &out_seq);
    return st == SSLM_SEQUENCE_KV_BUFFER_MISMATCH && out_seq == 0;
}

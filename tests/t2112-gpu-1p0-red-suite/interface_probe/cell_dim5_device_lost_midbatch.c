/* Design Sec9 line 562 + Sec11 dim 5, lines 722-727: a batch of 4 where the 3rd
 * sequence triggers device-removal; "sequences 1-2 must not report success for
 * work whose fence never signaled". That assertion requires per-sequence
 * outcomes out of the batch call. */
#include "sslm_gpu_1p0.h"
int cell(SslmGpuContext* ctx, SslmGpuSequenceHandle* const* seqs,
         const SslmGpuAdapterHandle* const* adapters) {
    SslmGpuStatus out_statuses[4];
    sslm_decode_step_batch_gpu(ctx, seqs, adapters, 4u, 24u, out_statuses);
    return out_statuses[2] == SSLM_DEVICE_LOST
        && out_statuses[0] != SSLM_OK && out_statuses[1] != SSLM_OK;
}

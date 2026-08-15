/* Design Sec7, lines 492-497: "one sequence's guard rejection ... does not abort
 * the whole batch ... the returned status is per-sequence (`out_statuses[n]`,
 * one slot per input sequence), never a single batch-wide status that would
 * force an all-or-nothing interpretation onto n independent sequences." */
#include "sslm_gpu_1p0.h"
int cell(SslmGpuContext* ctx, SslmGpuSequenceHandle* const* seqs,
         const SslmGpuAdapterHandle* const* adapters) {
    SslmGpuStatus out_statuses[4];
    sslm_decode_step_batch_gpu(ctx, seqs, adapters, 4u, 24u, out_statuses);
    /* seq 1 rejected, the other three must have proceeded: */
    return out_statuses[0] == SSLM_OK
        && out_statuses[1] == SSLM_ADAPTER_MODEL_MISMATCH
        && out_statuses[2] == SSLM_OK
        && out_statuses[3] == SSLM_OK;
}

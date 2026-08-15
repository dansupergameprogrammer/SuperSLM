/* Design Sec11 dim 8 product cell, RE-AUTHORED at fold (T-2111 rung 5) per
 * the strike's own vacuity finding (D-SLM3344): the pre-fold cell compiled
 * against a per-sequence budget and passed vacuously, because a per-sequence
 * budget can never be exhausted by an EARLIER sequence -- the cut it existed
 * to test could not occur.
 *
 * Repaired semantics (design Sec7): `dispatch_budget` on the batch call is
 * BATCH-WIDE. Four sequences, 24 dispatches/layer (Sec3/Sec6.1): a
 * whole-batch budget of 72 (= 3 x 24) covers exactly three sequences' own
 * first layer and is exhausted before the fourth's. This cell asserts the
 * PER-SEQUENCE consequence through `out_statuses` -- the channel the pre-fold
 * cell could not use, because the pre-fold signature had no such parameter --
 * not merely that the call type-checks. */
#include "sslm_gpu_1p0.h"
int cell(SslmGpuContext* ctx, SslmGpuSequenceHandle* const* seqs,
         const SslmGpuAdapterHandle* const* adapters) {
    const uint32_t batch_wide_dispatch_budget = 3u * 24u;  /* three whole layers, batch-wide */
    SslmGpuStatus out_statuses[4];
    SslmGpuStatus call_status = sslm_decode_step_batch_gpu(
        ctx, seqs, adapters, 4u, batch_wide_dispatch_budget, out_statuses);
    /* The call itself proceeded (recording started); the cut is visible only
     * in the per-sequence channel, never in the call-level return value. */
    return call_status == SSLM_OK
        && out_statuses[0] == SSLM_OK
        && out_statuses[1] == SSLM_OK
        && out_statuses[2] == SSLM_OK
        && out_statuses[3] == SSLM_BATCH_BUDGET_EXHAUSTED;   /* the cut, actually asserted */
}

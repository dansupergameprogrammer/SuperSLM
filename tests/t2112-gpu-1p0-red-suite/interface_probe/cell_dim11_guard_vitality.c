/* Design Sec11 dim 11 product cell, lines 891-902: "calling
 * `sslm_gpu_model_unmap` while that same run's sequences are still bound
 * (forcing `ModelHasLiveSequences`), or destroying the context mid-batch
 * (forcing `ContextHasLiveHandles`) -- asserting the release-build status fires
 * as a defined rejection, not a debug-assert-only guard". */
#include "sslm_gpu_1p0.h"
int cell(SslmGpuContext* ctx, SslmGpuModelHandle* model) {
    SslmGpuStatus a = sslm_gpu_model_unmap(ctx, model);   /* sequences still bound */
    SslmGpuStatus b = sslm_gpu_context_destroy(ctx);      /* handles still live    */
    return a == SSLM_MODEL_HAS_LIVE_SEQUENCES
        && b == SSLM_CONTEXT_HAS_LIVE_HANDLES;
}

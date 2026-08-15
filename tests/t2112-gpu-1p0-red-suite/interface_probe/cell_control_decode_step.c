/* CONTROL / must-accept. Design Sec4.3 + Sec9 row `AdapterModelMismatch`
 * ("checked at every decode-step call"). If this does not compile, the probe
 * is broken, not the design. */
#include "sslm_gpu_1p0.h"
int cell(SslmGpuContext* ctx, SslmGpuSequenceHandle* seq,
         const SslmGpuAdapterHandle* wrong_model_adapter) {
    SslmGpuStatus st = sslm_decode_step_gpu(ctx, seq, wrong_model_adapter, 24u);
    return st == SSLM_ADAPTER_MODEL_MISMATCH;   /* the caller learns which guard fired */
}

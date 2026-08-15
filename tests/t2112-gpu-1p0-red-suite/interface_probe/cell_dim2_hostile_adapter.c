/* Design Sec11 dim 2 product cell: a hostile adapter artifact "rejected,
 * never read past the tensor" at `sslm_gpu_adapter_map`, reported as
 * `AdapterBaseHashMismatch`. REPAIRED at fold (T-2111 rung 5) for the
 * out_adapter out-parameter the repaired signature (design Sec5.2) adds. */
#include "sslm_gpu_1p0.h"
int cell(SslmGpuContext* ctx, SslmGpuModelHandle* model,
         const SslmModelView* hostile_adapter) {
    SslmGpuAdapterHandle* out_adapter = 0;
    SslmGpuStatus st = sslm_gpu_adapter_map(ctx, model, hostile_adapter, &out_adapter);
    return st == SSLM_ADAPTER_BASE_HASH_MISMATCH && out_adapter == 0;
}

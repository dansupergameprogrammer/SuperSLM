// T-2932: public-schema-query red witness for TE-373 S2 / TE-374.
//
// This translation unit deliberately names only the public query surface.  At e822c70
// it fails to compile with the three missing declarations below.  The companion runner
// recognizes that exact failure as RED_API_SURFACE_ABSENT and records every modeled cell;
// any other compiler failure is an infrastructure failure, never evidence for a cell.
// Once the declarations land, this file becomes the runner's link-and-execute entry point.

#include "superslm/gpu_1p0.h"
#include "superslm/sslm_abi.h"

int main() {
    SslmGpuContext* gpu_context = nullptr;
    SslmGpuSequenceHandle* gpu_sequence = nullptr;
    int32_t accepting = -101;
    int32_t bound = -102;
    sslm_seq cpu_sequence = nullptr;

    // The calls are intentionally unevaluated at this pinned red revision: the declarations
    // themselves are the first contract the builder must supply.
    (void)SslmGpuSeqSchemaAcceptingForG5Bridge(gpu_context, gpu_sequence, &accepting);
    (void)SslmGpuSeqSchemaBoundForG5Bridge(gpu_context, gpu_sequence, &bound);
    (void)sslm_seq_schema_bound(cpu_sequence, &bound);
    return accepting + bound;
}

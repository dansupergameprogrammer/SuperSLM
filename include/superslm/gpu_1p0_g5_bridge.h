#ifndef SSLM_GPU_1P0_G5_BRIDGE_H
#define SSLM_GPU_1P0_G5_BRIDGE_H
// gpu_1p0_g5_bridge.h -- G5-5 (T-2132, Brunel). PROMOTED (design Sec14.2, D-SLM3477,
// Claude/Poirot/9bc9ec6-t2132-g5-arc-review.md S8, T-2132/Curie fix round): every G5 GPU verb
// this header used to declare now lives on the shipped 1.0 GPU API surface,
// include/superslm/gpu_1p0.h, alongside its own full rationale for the promotion -- a genuine
// product capability (schema-constrained decoding on the GPU, proven bit-identical to the CPU
// path at 80 real decode steps) had no shipped entry point, and this design's own rung-7
// precedent (Sec11.2) rules that shape promotes to production rather than staying test-only.
// D-SLM3443 (GPU parity REQUIRED, no dated deferral) independently forecloses the deferred
// branch.
//
// This file is retained -- not deleted -- as a compatibility include for every existing
// includer (src/gpu/gpu_1p0.cpp, tools/t2132_g5_gpu_parity_gpu.cpp,
// tools/t2132_g5_gpu_parity_shared.h, tools/t2132_diag_layer_bisect_gpu.cpp; this ticket's own
// invocation contract holds tools/ and src/gpu/ read-only, so their own #include lines are not
// touched) plus the one piece of this header's original content that is NOT a verb declaration
// and so is not part of D-SLM3477's own promotion: the walk-state-unused sentinel below. Every
// `#include "superslm/gpu_1p0_g5_bridge.h"` site continues to see the full G5 GPU surface,
// unchanged, via the transitive #include of gpu_1p0.h immediately below.
//
// Scope (design Claude/Vitruvius/t2119-g5-constrained-decoding-design-2026-08-16.md Sec6, Slot
// G5-5 -- "GPU parity for constrained decoding") is stated in full at gpu_1p0.h's own G5 section
// banner now; not repeated here.

#include <cstddef>
#include <cstdint>

#include "superslm/gpu_1p0.h"  // SslmGpuContext/SslmGpuModelHandle/SslmGpuSequenceHandle
                                // (opaque) AND, since D-SLM3477's promotion, every G5 GPU verb
                                // declaration this file used to carry directly.

// Sentinel matching the CPU ABI's own `kDfaWalkStateUnused` (src/sslm_abi.cpp) -- "no schema
// ever bound" on a freshly-created sequence handle. NOT a verb declaration -- D-SLM3477's own
// promotion scope is the eight functions (now in gpu_1p0.h); this constant stays here, this
// header's own remaining reason to exist as more than a bare #include.
constexpr uint32_t kSslmGpuDfaWalkStateUnused = 0xFFFFFFFFu;

#endif  // SSLM_GPU_1P0_G5_BRIDGE_H

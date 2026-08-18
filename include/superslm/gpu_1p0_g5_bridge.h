#ifndef SSLM_GPU_1P0_G5_BRIDGE_H
#define SSLM_GPU_1P0_G5_BRIDGE_H
// gpu_1p0_g5_bridge.h -- compatibility include for schema-constrained (G5) GPU decoding.
//
// The G5 GPU verbs this header used to declare directly now live on the shipped GPU API
// surface, include/superslm/gpu_1p0.h. This header is retained, not deleted, so every existing
// `#include "superslm/gpu_1p0_g5_bridge.h"` site continues to see the full G5 GPU surface
// unchanged, via the transitive #include of gpu_1p0.h below. Its only remaining unique content
// is the walk-state-unused sentinel declared below.

#include <cstddef>
#include <cstdint>

#include "superslm/gpu_1p0.h"  // SslmGpuContext/SslmGpuModelHandle/SslmGpuSequenceHandle
                                // (opaque) AND every G5 GPU verb declaration this file used
                                // to carry directly, now promoted to gpu_1p0.h.

// Sentinel matching the CPU ABI's own `kDfaWalkStateUnused` (src/sslm_abi.cpp) -- "no schema
// ever bound" on a freshly-created sequence handle. NOT a verb declaration -- the G5 verb
// promotion scope is the eight functions (now in gpu_1p0.h); this constant stays here, this
// header's own remaining reason to exist as more than a bare #include.
constexpr uint32_t kSslmGpuDfaWalkStateUnused = 0xFFFFFFFFu;

#endif  // SSLM_GPU_1P0_G5_BRIDGE_H

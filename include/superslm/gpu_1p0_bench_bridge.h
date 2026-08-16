#ifndef SSLM_GPU_1P0_BENCH_BRIDGE_H
#define SSLM_GPU_1P0_BENCH_BRIDGE_H
// T-2113 (B3): build-seat-only bench accessors for SslmGpuSequenceHandle -- NOT part of
// the shipped 1.0 API surface. `include/superslm/gpu_1p0.h` stays byte-for-byte identical
// to the red suite's own canonical header (T-2113 B1's own header banner) so the two link
// against each other; this file exists precisely so nothing needs to be added there.
//
// `sslm_decode_step_gpu` (design Sec4.3) does not exist yet -- B5 (design Sec10) is what
// builds the async decode-step call every real caller will use. Until it lands, the only
// way to drive a real forward pass THROUGH a real SslmGpuSequenceHandle's own dedicated
// K/V buffer (the thing B3 exists to prove correct, design Sec5.3/Sec10 B3's own gate) is
// to call the pre-1.0 `superslm_gpu::RunLayerLoopGpu` directly, through the external-buffer
// bridge B3 added to it (gpu_port.h's own trailing `external_kv_resident`/
// `io_external_kv_needs_resume_barrier` pair). That bridge needs the raw D3D12 resource
// pointer and the per-handle resume latch, both of which are private members of the
// (deliberately opaque, per design Sec4.1) `SslmGpuSequenceHandle` struct -- these
// accessors are the narrow, named, documented seam that exposes exactly those two fields
// (plus the host-mirrored SequenceLayerState surface a bench driver needs to build a real
// `superslm::SequenceLayerState&` from) to a build-seat bench tool, and nothing else.
//
// Retired the moment B5 lands and a real `sslm_decode_step_gpu` exists to exercise this
// path through the public surface instead (StandardsDocument.md Sec5.6: named here, not
// silently left as permanent surface).
//
// tools/t2113_b3_sequence_smoke.cpp is this header's only intended includer.

#include <cstdint>

#include "superslm/checked_chain_funnel.h"  // superslm::CarriedScale
#include "superslm/gpu_1p0.h"               // SslmGpuSequenceHandle (opaque)

struct ID3D12Resource;

ID3D12Resource* SslmGpuSeqHandleKvBufferForBench(SslmGpuSequenceHandle* seq);
bool* SslmGpuSeqHandleKvResumeFlagForBench(SslmGpuSequenceHandle* seq);
int8_t* SslmGpuSeqHandleHiddenCodesForBench(SslmGpuSequenceHandle* seq);
superslm::CarriedScale* SslmGpuSeqHandleHiddenScaleForBench(SslmGpuSequenceHandle* seq);
uint32_t* SslmGpuSeqHandleLayerIndexForBench(SslmGpuSequenceHandle* seq);
uint64_t* SslmGpuSeqHandleKvSaturationForBench(SslmGpuSequenceHandle* seq);
int64_t* SslmGpuSeqHandleContextLengthForBench(SslmGpuSequenceHandle* seq);
size_t SslmGpuSeqHandleHiddenSizeForBench(SslmGpuSequenceHandle* seq);
int64_t SslmGpuSeqHandleContextCapForBench(SslmGpuSequenceHandle* seq);  // T-2113 (N1)

#endif  // SSLM_GPU_1P0_BENCH_BRIDGE_H

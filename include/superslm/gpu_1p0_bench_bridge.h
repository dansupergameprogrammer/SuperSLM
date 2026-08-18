#ifndef SSLM_GPU_1P0_BENCH_BRIDGE_H
#define SSLM_GPU_1P0_BENCH_BRIDGE_H
// Benchmark-only accessors for SslmGpuSequenceHandle -- not part of the shipped GPU API
// surface, and not installed alongside it. `SslmGpuSequenceHandle` is deliberately opaque to
// ordinary callers; these accessors are a narrow, named seam that exposes the sequence handle's
// underlying D3D12 K/V resource, resume state, and host-mirrored layer state to a benchmark
// driver that needs to drive a forward pass through a real sequence handle directly, and
// nothing else.
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

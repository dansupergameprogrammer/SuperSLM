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

// T-2184 remedy C1 (Brunel fix round 1, Claude/Poirot/efeb9ba-t2184-t2169-gpu-batched-prefill-
// review.md; D-SLM3662): a bench-only entry point reproducing the GENUINELY shipped (pre-T-2169,
// `e35edc1`) per-token prefill composition -- one submit-and-fence round-trip per layer, issued
// once per token -- for a bench harness that needs a reference arm the T-2169 batching change did
// not touch. `tools/t2180_rung6_tokps.cpp` is this declaration's intended includer, alongside
// this header's existing accessors. See its own definition (src/gpu/gpu_1p0.cpp) for the full
// account of why the batched public bridge functions can no longer serve this role.
//
// T-2185 remedy N6/observation (Brunel fix round 2, D-SLM3677): gated behind
// `SUPERSLM_ENABLE_GPU_BENCH_PRE_BATCHING`, matching this codebase's own established injection-
// seam convention (`SUPERSLM_O11_ALLOC_INJECTION`/`SUPERSLM_T2169_CHUNK_RECORDING_FAULT_
// INJECTION`, gpu_port.h) -- an `#ifdef`-gated symbol, absent entirely (not merely undeclared)
// from any translation unit that does not define the macro. `build.bat`'s own
// `tools\t2180_rung6_tokps.cpp` compile line is the ONLY place this macro is ever defined; every
// other build target -- including the red-suite cells and every other bench/smoke tool, none of
// which call this function -- compiles `src/gpu/gpu_1p0.cpp` without it, so the symbol is not
// linked into any of them.
#ifdef SUPERSLM_ENABLE_GPU_BENCH_PRE_BATCHING
SslmGpuStatus SslmGpuSeqPrefillPromptPreBatchingBenchOnly(SslmGpuContext* ctx,
                                                            SslmGpuSequenceHandle* seq,
                                                            const int32_t* tokens, int32_t count,
                                                            uint32_t dispatch_budget);
#endif  // SUPERSLM_ENABLE_GPU_BENCH_PRE_BATCHING

#endif  // SSLM_GPU_1P0_BENCH_BRIDGE_H

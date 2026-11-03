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
#include "superslm/forward_sites.h"          // superslm::SequenceLayerState
#include "superslm/gpu_1p0.h"               // SslmGpuSequenceHandle (opaque)

struct ID3D12Resource;

ID3D12Resource* SslmGpuSeqHandleKvBufferForBench(SslmGpuSequenceHandle* seq);
bool* SslmGpuSeqHandleKvResumeFlagForBench(SslmGpuSequenceHandle* seq);
int8_t* SslmGpuSeqHandleHiddenCodesForBench(SslmGpuSequenceHandle* seq);
superslm::CarriedScale* SslmGpuSeqHandleHiddenScaleForBench(SslmGpuSequenceHandle* seq);
uint32_t* SslmGpuSeqHandleLayerIndexForBench(SslmGpuSequenceHandle* seq);
uint64_t* SslmGpuSeqHandleKvSaturationForBench(SslmGpuSequenceHandle* seq);
superslm::SequenceLayerState* SslmGpuSeqHandleLiveStateForBench(SslmGpuSequenceHandle* seq);
int64_t* SslmGpuSeqHandleContextLengthForBench(SslmGpuSequenceHandle* seq);
size_t SslmGpuSeqHandleHiddenSizeForBench(SslmGpuSequenceHandle* seq);
int64_t SslmGpuSeqHandleContextCapForBench(SslmGpuSequenceHandle* seq);  // T-2113 (N1)

// Paged-KV plan step G3: the test seams the GPU red suite (group g1a) names and the plan does not.
// Table entry i (< mapped) of a pooled GPU sequence; i at or past `mapped`, a dedicated sequence or
// a null `out_page` is SSLM_SEQUENCE_KV_BUFFER_MISMATCH. `out_mapped` (optional) gets `mapped`.
SslmGpuStatus sslm_gpu_test_only_seq_table_entry(const SslmGpuSequenceHandle* seq, uint32_t i, uint32_t* out_page,
                                                 uint32_t* out_mapped) noexcept;
// Arms a one-shot fault on the view the pooled `seq`'s next submission hands the GPU ladder: 1 the
// table one page short, 2 page_bytes one byte short, 3 a null pool base, 4 page_positions = 2^61;
// 0 disarms. A dedicated sequence or another value: SSLM_SEQUENCE_KV_BUFFER_MISMATCH.
SslmGpuStatus sslm_gpu_test_only_arm_view_fault(SslmGpuSequenceHandle* seq, uint32_t fault) noexcept;
// The superslm::SslmForwardStatus the ladder returned for `seq`'s last refused submission, as an
// integer; Ok when its last submission was not refused before recording.
SslmGpuStatus sslm_gpu_test_only_last_ladder_status(const SslmGpuSequenceHandle* seq,
                                                    int32_t* out_forward_status) noexcept;
// Precached prefixes (1.16.0): table entry i (< mapped) of a GPU prefix's internal holder, as
// sslm_gpu_test_only_seq_table_entry reads a pooled sequence's.
SslmGpuStatus sslm_gpu_test_only_prefix_table_entry(const SslmGpuPrefixHandle* prefix, uint32_t i, uint32_t* out_page,
                                                    uint32_t* out_mapped) noexcept;

// Schema fast-decode plan rev 1.4.1 §11.0: the GPU seams of slices G1-F, G1-M and G1-J. Test-only:
// declared and defined only under SUPERSLM_GPU_TEST_SEAMS, so they are absent from the production library.
#if defined(SUPERSLM_GPU_TEST_SEAMS)
// S-GSUB (G1-F): the submissions made so far on `ctx`'s device and on the process's submission device
// (when it is set up), read from the harness's fence_value_counter(). A cell reads it before and after a
// call; the difference is the call's submissions. A null argument: SSLM_SEQUENCE_KV_BUFFER_MISMATCH.
SslmGpuStatus sslm_gpu_test_only_submission_count(SslmGpuContext* ctx, uint64_t* out_count) noexcept;
// The GPU model's forced-next table, read (G12.4's golden hashes, as the CPU's cell 6.6 reads its own):
// next_forced(state) (-1 when unforced) and |F(state)| (0 when unforced) of schema `schema_index`.
// SSLM_GPU_FORCED_TABLE_ABSENT when the model has no table; SSLM_SEQUENCE_KV_BUFFER_MISMATCH for a null
// argument or an index out of range.
SslmGpuStatus sslm_gpu_test_only_forced_table_entry(const SslmGpuModelHandle* model, int32_t schema_index,
                                                    uint32_t state, int32_t* out_next_forced,
                                                    uint32_t* out_f_len) noexcept;
// S-GROWS (G1-M): one entry per token finish (SslmGpuSeqFinishTokenForG5Bridge, directly or through the decode
// bridge), in call order, process-wide: the head rows the finish scored (V for the full finish, n for the
// legal-row finish) and whether the device logits ran (1) or the host computed the row (0). Reset clears it.
// An index at or past the count, or a null output: SSLM_SEQUENCE_KV_BUFFER_MISMATCH.
void sslm_gpu_test_only_rows_log_reset() noexcept;
size_t sslm_gpu_test_only_rows_log_count() noexcept;
SslmGpuStatus sslm_gpu_test_only_rows_log_at(size_t i, uint64_t* out_rows, int32_t* out_device_logits) noexcept;
// S-GBOUND (G1-M): forces the model's LegalRowFinishExact predicate false (force 1) or back to its mapped value
// (force 0), the GPU twin of S-BOUND. A forced model finishes every bound token with the full finish. Must not
// race a finish on the model. A null model: SSLM_SEQUENCE_KV_BUFFER_MISMATCH.
SslmGpuStatus sslm_gpu_test_only_force_full_finish(SslmGpuModelHandle* model, int32_t force) noexcept;
// S-GCHUNKFAULT (G1-J): arms a single-shot fault for the next sslm_gpu_seq_jump_forward call's pass. Its sub-chunk
// `sub_chunk` (0 = the first; sub-chunks of 4 rows) is refused before it is recorded, as a failed submission is,
// with `status`: SSLM_GPU_ALLOCATION_FAILED or SSLM_DEVICE_LOST. Earlier sub-chunks run and commit; the verb's
// failure exit then restores its entry state. A negative `sub_chunk` disarms. Any other status:
// SSLM_SEQUENCE_KV_BUFFER_MISMATCH. Only the verb consumes it; the bridges never do.
SslmGpuStatus sslm_gpu_test_only_arm_chunk_fault(int32_t sub_chunk, SslmGpuStatus status) noexcept;
// S-GJUMPALLOC (G1-J, engine code review B1): arms a single-shot host std::bad_alloc in the next
// sslm_gpu_seq_jump_forward call's pass, after its entry snapshot is taken: site 0 before the chunk pre-scan (the
// prefill snapshot's flag already cleared), 1 after a pooled holder's pages are mapped, 2 after the pass's
// sub-chunks committed, before the walk and the ready flag move. A negative `site` disarms; any other value,
// SSLM_SEQUENCE_KV_BUFFER_MISMATCH. Only the verb consumes it.
SslmGpuStatus sslm_gpu_test_only_arm_jump_host_bad_alloc(int32_t site) noexcept;
// The context's own harness device (a superslm_gpu::harness::Device*, as void* so this header needs no harness),
// for the cells' debug-layer report at exit (engine code review N3). Null for a null or destroyed context.
void* sslm_gpu_test_only_context_device(SslmGpuContext* ctx) noexcept;
#endif  // SUPERSLM_GPU_TEST_SEAMS

#if defined(SUPERSLM_ENABLE_GPU_API_FAILURE_INJECTION)
enum class SslmGpuApiFailureInjectionPoint { BeforeHandler, AfterHandler };
void ArmSslmGpuApiFailureInjection(const char* api_name,
                                   SslmGpuApiFailureInjectionPoint point);
void ClearSslmGpuApiFailureInjection();
#endif

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

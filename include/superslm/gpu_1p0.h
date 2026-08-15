#ifndef SSLM_GPU_1P0_H
#define SSLM_GPU_1P0_H
// T-2113 (B1): the production 1.0 GPU API surface.
//
// This header is a byte-for-byte declaration match against the red suite's own
// canonical copy (D:\SuperSLM\.worktrees\t2112-red-suite\tests\t2112-gpu-1p0-red-suite\
// sslm_gpu_1p0.h, promoted from the T-2111 strike probe, transcribed verbatim from
// Claude/Vitruvius/t2107-gpu-core-1p0-design-2026-08-14.md Sec4/Sec5, post-fold). The
// suite includes ITS OWN copy, never this one -- the two must stay declaration-identical
// (checked at every B-section checkpoint: `diff` against the suite's copy) so that the
// object files this header's own .cpp produces link cleanly against the suite's call
// sites. Every function below is declared at GLOBAL scope with ordinary C++ linkage
// (not extern "C", not inside `namespace superslm_gpu`) because that is what the suite's
// own header declares, and C++ link-name matching requires the definition to share the
// caller's exact scope, not merely an equivalent signature in a different namespace.
//
// Every fallible call returns SslmGpuStatus; every value it produces (a handle, a ready
// flag, a decoded status, a batch's per-sequence outcomes) is delivered through an
// out-parameter, never through the return value (design Sec4.1.1, the T-2111 fold's own
// repair). B-section provenance is noted per declaration; only B1's symbols
// (sslm_gpu_context_create/destroy) are DEFINED as of this commit -- src/gpu/gpu_1p0.cpp.
// Every other declaration below exists so later B-sections extend one already-complete,
// already-reviewed header rather than re-declaring the surface piecemeal.

#include <stdint.h>
#include <stddef.h>

/* --- opaque handle types, design Sec4.1 --- */
typedef struct SslmGpuContext        SslmGpuContext;
typedef struct SslmGpuModelHandle    SslmGpuModelHandle;
typedef struct SslmGpuAdapterHandle  SslmGpuAdapterHandle;
typedef struct SslmGpuSequenceHandle SslmGpuSequenceHandle;

/* Same namespace-collision fix the suite's own header carries (T-2112 Finding 1,
 * D-SLM3346): SslmModelView must be THE real superslm::SslmModelView, not a second
 * distinct global-scope type of the same name, so production callers can load a real
 * .sslm artifact through this API. Declared once here for every 1.0 production TU. */
namespace superslm { struct SslmModelView; }
using superslm::SslmModelView;

/* Design Sec4.1.1/Sec5.1 name these two config types as call parameters but assign
 * no fields to either -- the suite's own dim1_lifetime_red.cpp names this
 * explicitly as "the build seat defines alongside B1/B2, design Sec10." Defined
 * here (B1) as reserved, zero-behavior structs: real by-value C++ types (not
 * incomplete forward declarations, which cannot be passed by value -- the gap
 * that broke the first B1 build attempt, Claude/Brunel/t2113-1p0-core-build-
 * 2026-08-15.md), with one reserved field each so a future field can be added
 * without changing either struct's calling convention (an all-zero-initialized
 * value is always a valid "no options requested" config under either shape). */
typedef struct GpuContextConfig {
	int reserved;  /* design Sec4.1.1 assigns no fields yet; zero-initialize */
} GpuContextConfig;
typedef struct GpuResidencyConfig {
	int reserved;  /* design Sec5.1 assigns no fields yet; zero-initialize */
} GpuResidencyConfig;

/* --- status enum ---
 * Global-scope C-style enum, matching the suite's own copy exactly (SSLM_-prefixed
 * enumerators -- the suite's own T-2111-fold naming, kept here rather than
 * "corrected" to the substrate's bare-enumerator convention, because this enum's
 * whole reason to exist at global scope is link-compatibility with the suite; the
 * substrate's OWN namespaced `superslm_gpu::SslmGpuStatus` (gpu_port.h:472) is a
 * separate C++ type that this design's Sec9 conceptually extends but cannot literally
 * share across the namespace boundary the suite's header itself introduced -- see the
 * B1 section of Claude/Brunel/t2113-1p0-core-build-2026-08-15.md for the routed note). */
typedef enum SslmGpuStatus {
    SSLM_OK = 0,
    SSLM_DISPATCH_BUDGET_TOO_SMALL,      /* substrate, gpu_port.h:472           */
    SSLM_BUSY,                           /* design Sec9                         */
    SSLM_CONTEXT_HAS_LIVE_HANDLES,       /* design Sec9 -- B1                   */
    SSLM_MODEL_HAS_LIVE_SEQUENCES,       /* design Sec9 -- B2                   */
    SSLM_ADAPTER_MODEL_MISMATCH,         /* design Sec9 -- B6                   */
    SSLM_ADAPTER_BASE_HASH_MISMATCH,     /* design Sec9 -- B6                   */
    SSLM_SEQUENCE_KV_BUFFER_MISMATCH,    /* design Sec9 -- B3                   */
    SSLM_DEVICE_LOST,                    /* design Sec9 -- B1/B2                */
    SSLM_BATCH_BUDGET_EXHAUSTED,         /* design Sec9 -- B7                   */
    SSLM_TOKEN_ID_OUT_OF_RANGE           /* design Sec9 -- B3.5 (D-SLM3367)     */
} SslmGpuStatus;

/* --- Sec4.1.1: context create/destroy. DEFINED as of B1 (src/gpu/gpu_1p0.cpp). --- */
SslmGpuStatus sslm_gpu_context_create(GpuContextConfig cfg, SslmGpuContext** out_ctx);
SslmGpuStatus sslm_gpu_context_destroy(SslmGpuContext* ctx);

/* --- Sec5.1: model map/unmap. Declared for B2. --- */
SslmGpuStatus sslm_gpu_model_map(SslmGpuContext* ctx, const SslmModelView* base,
                                  GpuResidencyConfig cfg, SslmGpuModelHandle** out_model);
SslmGpuStatus sslm_gpu_model_unmap(SslmGpuContext* ctx, SslmGpuModelHandle* model);

/* --- Sec5.2: adapter map/unmap. Declared for B6. --- */
SslmGpuStatus sslm_gpu_adapter_map(SslmGpuContext* ctx, SslmGpuModelHandle* model,
                                    const SslmModelView* adapter_artifact,
                                    SslmGpuAdapterHandle** out_adapter);
SslmGpuStatus sslm_gpu_adapter_unmap(SslmGpuContext* ctx, SslmGpuAdapterHandle* adapter);

/* --- Sec5.3: sequence create/release. Declared for B3. --- */
SslmGpuStatus sslm_gpu_seq_create(SslmGpuContext* ctx, SslmGpuModelHandle* model,
                                   int64_t context_cap, SslmGpuSequenceHandle** out_seq);
SslmGpuStatus sslm_gpu_seq_release(SslmGpuContext* ctx, SslmGpuSequenceHandle* seq);

/* --- Sec5.3a: the production token-feed entry point. Host-only -- no dispatch, no
 * state transition to Submitted. DEFINED as of B3.5 (src/gpu/gpu_1p0.cpp), added at the
 * mini-fold of 2026-08-15 routing D-SLM3367. Precondition: seq state Idle (Busy against
 * Submitted). Effect on success: overwrites seq's own hidden_codes/hidden_scale via the
 * identical EmbedEntry primitive the CPU path calls, resets layer_index to 0, returns Ok.
 * On a hostile token_id (outside [0, vocab_size)): returns TokenIdOutOfRange, seq's own
 * state left untouched. */
SslmGpuStatus sslm_gpu_seq_embed_token(SslmGpuContext* ctx, SslmGpuSequenceHandle* seq,
                                        int32_t token_id);

/* --- Sec4.2: save/restore/reset. Declared for B3/B5. --- */
SslmGpuStatus sslm_gpu_seq_save(SslmGpuContext* ctx, const SslmGpuSequenceHandle* seq,
                                 void* out_blob, size_t* out_blob_size);
SslmGpuStatus sslm_gpu_seq_restore(SslmGpuContext* ctx, SslmGpuModelHandle* model,
                                    const void* blob, size_t blob_size,
                                    SslmGpuSequenceHandle** out_seq);
SslmGpuStatus sslm_gpu_seq_reset(SslmGpuContext* ctx, SslmGpuSequenceHandle* seq);

/* --- Sec4.3: the two decode calls. Declared for B5/B7. --- */
SslmGpuStatus sslm_decode_step_gpu(
    SslmGpuContext* ctx,
    SslmGpuSequenceHandle* seq,
    const SslmGpuAdapterHandle* adapter_or_null,   /* per-sequence, Sec8 */
    uint32_t dispatch_budget);

SslmGpuStatus sslm_decode_step_batch_gpu(
    SslmGpuContext* ctx,
    SslmGpuSequenceHandle* const* seqs,
    const SslmGpuAdapterHandle* const* adapters_or_null,  /* parallel array, per-sequence, Sec8 */
    uint32_t n_sequences,
    uint32_t dispatch_budget,          /* BATCH-WIDE -- design Sec7 */
    SslmGpuStatus* out_statuses);       /* [n_sequences] -- design Sec7 */

/* --- Sec4.2: sslm_gpu_ready. Declared for B5. --- */
SslmGpuStatus sslm_gpu_ready(SslmGpuContext* ctx,
                              SslmGpuSequenceHandle* seq,
                              int32_t block,
                              int32_t* out_ready,
                              SslmGpuStatus* out_status);

#endif /* SSLM_GPU_1P0_H */

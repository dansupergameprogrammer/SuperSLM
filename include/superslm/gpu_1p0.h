#ifndef SSLM_GPU_1P0_H
#define SSLM_GPU_1P0_H
// T-2113 (B1): the production 1.0 GPU API surface.
//
// This header's own DECLARATIONS -- opaque handle typedefs, the SslmGpuStatus enum (every
// enumerator, same order, same values), the GpuContextConfig/GpuResidencyConfig struct
// bodies, and every function signature -- match the red suite's own canonical copy
// (D:\SuperSLM\.worktrees\t2112-red-suite\tests\t2112-gpu-1p0-red-suite\sslm_gpu_1p0.h,
// promoted from the T-2111 strike probe, transcribed verbatim from
// Claude/Vitruvius/t2107-gpu-core-1p0-design-2026-08-14.md Sec4/Sec5, post-fold). The suite
// includes ITS OWN copy, never this one -- the two must stay declaration-identical (checked at
// every B-section checkpoint) so that the object files this header's own .cpp produces link
// cleanly against the suite's call sites. **This is a declaration-shape parity, not a literal
// whole-file `diff`**: each copy carries its own explanatory prose (this file narrates the
// production build's own provenance per declaration; the suite's copy narrates the design
// section and promotion history), so a raw `diff` between the two files shows comment lines
// differing throughout by design -- corrected here (T-2114, M2,
// Claude/Poirot/50f3d5d-t2113-1p0-gpu-core-build-review.md) after that comment's own literal
// "checked... via diff" claim was found false: the suite's copy had GpuContextConfig/
// GpuResidencyConfig as INCOMPLETE forward declarations while this header defines them
// complete (by-value call parameters cannot be incomplete types), a real declaration-shape
// divergence a real `diff` would have caught, not merely a stale sentence -- both now define
// the same complete struct bodies. Every function below is declared at GLOBAL scope with
// ordinary C++ linkage (not extern "C", not inside `namespace superslm_gpu`) because that is
// what the suite's own header declares, and C++ link-name matching requires the definition to
// share the caller's exact scope, not merely an equivalent signature in a different namespace.
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
    SSLM_TOKEN_ID_OUT_OF_RANGE,          /* design Sec9 -- B3.5 (D-SLM3367)     */
    /* T-2114 (S1, Claude/Poirot/50f3d5d-t2113-1p0-gpu-core-build-review.md): a per-sequence
     * decode-time rejection that the CPU-domain guard ladder (RunLayerLoopGpuSubmit's own
     * pre-submission checks, or DecodeStickyTag's own post-dispatch decode) produced --
     * InvalidLayerBudget, ChainInputOutOfDomain, SoftmaxRowWidthOutOfDomain, and every other
     * superslm::SslmForwardStatus value that is neither Ok nor a real device-level failure
     * (GpuDeviceRemoved/GpuAllocationFailed, which map to SSLM_DEVICE_LOST below instead).
     * Design Sec9 deliberately assigns no dedicated 1.0 status per individual guard reason
     * (this enum does not grow one enumerator per CPU-domain check); this ONE status is the
     * real distinction that matters to a caller -- THIS sequence's own decode step was
     * rejected on a numeric/structural ground, the shared device is healthy, and (design
     * Sec7's own per-sequence independence) no other sequence in the same batch call is
     * affected. Before this status existed, both mapping functions
     * (MapSubmitRejectionToGpuStatus/MapDecodedStatusToGpuStatus, gpu_1p0.cpp) collapsed
     * every one of these into SSLM_DEVICE_LOST, which made sslm_decode_step_batch_gpu's own
     * "one sequence's guard rejection does not abort the batch" contract (Sec7) impossible to
     * honor -- the batch loop's own DeviceLost-poisons-the-rest logic (gpu_1p0.cpp) cannot
     * tell a real device loss apart from a healthy device's per-sequence rejection when both
     * arrive through the same value. */
    SSLM_SEQUENCE_REJECTED,
    /* T-2113 (P2, design Sec4.2/Sec9/Sec22, routed `Claude/Poirot/50f3d5d-t2113-1p0-gpu-core-
     * build-review.md` Sec15, D-SLM3415): `sslm_gpu_seq_restore` (Sec4.2/Sec5.3), the v3 blob's
     * own `model_content_hash` field does not match the TARGET model handle's own
     * `RawIntegrityHash()` -- a blob saved from one model restored against a different one, the
     * identity gap the N1 size-admissibility widening (Sec21) left open. Checked after the
     * size-derivation ladder (a malformed blob is rejected for that reason first) and before any
     * device work. Appended LAST, the same precedent SSLM_SEQUENCE_REJECTED (T-2114 S1) already
     * set: no existing enumerator value moves. */
    SSLM_RESTORE_MODEL_MISMATCH
} SslmGpuStatus;

/* --- Sec4.1.1: context create/destroy. DEFINED as of B1 (src/gpu/gpu_1p0.cpp). --- */
SslmGpuStatus sslm_gpu_context_create(GpuContextConfig cfg, SslmGpuContext** out_ctx);
SslmGpuStatus sslm_gpu_context_destroy(SslmGpuContext* ctx);

/* --- Sec5.1: model map/unmap. Declared for B2. --- */
SslmGpuStatus sslm_gpu_model_map(SslmGpuContext* ctx, const SslmModelView* base,
                                  GpuResidencyConfig cfg, SslmGpuModelHandle** out_model);
SslmGpuStatus sslm_gpu_model_unmap(SslmGpuContext* ctx, SslmGpuModelHandle* model);

/* --- Sec5.2: adapter map/unmap. Declared for B6.
 * `sslm_gpu_adapter_map`: on a base-hash mismatch against `model`, returns AdapterBaseHashMismatch,
 * `*out_adapter=nullptr`; on an upload/allocation failure or a foreign `model` (mapped against a
 * DIFFERENT context than `ctx`, T-2124 D-SLM3446 P1-4), returns DeviceLost, `*out_adapter=nullptr`.
 *
 * `sslm_gpu_adapter_unmap`: releases the adapter's own residency and returns Ok -- CARRIES A `Busy`
 * PRECONDITION (added T-2124, D-SLM3446 P0-3, Claude/Poirot/435f730-t2124-adapter-uaf-review.md S1/
 * S3; design Sec5.2/Sec9, mirrors sslm_gpu_model_unmap's own identical precondition, D-SLM3417):
 * returns Busy while any Submitted sequence's own in-flight decode call still has `adapter` bound
 * -- releasing while such a submission is in flight would free device buffers (lora_ab_buf/
 * fold_buf) the GPU may still be reading through an already-recorded, not-yet-fenced command list, a
 * real use-after-free. REMEDY: drain every sequence that bound this adapter (poll sslm_gpu_ready to
 * completion) and retry -- the same remedy every other Busy-returning release call in this header
 * already expects a caller to apply. `adapter` mapped against a DIFFERENT context than `ctx`
 * (T-2124 P1-4) returns DeviceLost, same disposition as the map call's own foreign-`model` case
 * above. `unmap(ctx, nullptr)` is a documented no-op, returns Ok, checked before the Busy/DeviceLost
 * preconditions above (matches sslm_gpu_model_unmap(ctx, nullptr)'s own precedent). --- */
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

/* --- Sec4.2: save/restore/reset. Declared for B3/B5.
 * `sslm_gpu_seq_save` writes a v3 blob ('SLM3'), carrying `model`'s own content hash (T-2113 P2,
 * design Sec22). `sslm_gpu_seq_restore` sizes the fresh handle to the blob's own recorded
 * context_cap (N1, Sec21) and rejects a v1/v2 blob outright on magic, a malformed/inadmissible
 * size derivation, OR a model_content_hash that does not match `model`'s own hash --
 * SSLM_RESTORE_MODEL_MISMATCH, distinct from the generic malformed-blob disposition (P2, Sec22).
 * --- */
SslmGpuStatus sslm_gpu_seq_save(SslmGpuContext* ctx, const SslmGpuSequenceHandle* seq,
                                 void* out_blob, size_t* out_blob_size);
SslmGpuStatus sslm_gpu_seq_restore(SslmGpuContext* ctx, SslmGpuModelHandle* model,
                                    const void* blob, size_t blob_size,
                                    SslmGpuSequenceHandle** out_seq);
SslmGpuStatus sslm_gpu_seq_reset(SslmGpuContext* ctx, SslmGpuSequenceHandle* seq);

/* --- Sec4.3: the two decode calls. Declared for B5/B7.
 * Thread-safety (design Sec5.4, T-2113 B8): safe to call concurrently from different threads
 * against DIFFERENT SslmGpuSequenceHandle values ("thread-safe execution over disjoint
 * sequences," D-SLM3294) -- BUT every call that submits GPU work (sslm_decode_step_gpu,
 * sslm_decode_step_batch_gpu, and sslm_gpu_ready(block=1) draining one) must be externally
 * serialized by the caller: this design does not build an internal queue-level lock (see
 * src/gpu/gpu_1p0.cpp's own SslmGpuContext comment for the full contract and the grounded,
 * currently-process-wide reason). Two threads driving the SAME sequence handle concurrently is
 * an unguarded caller error, not a supported use (see the same comment). */
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

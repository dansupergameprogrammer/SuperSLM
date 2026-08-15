/* sslm_gpu_1p0.h -- T-2111 probe.
 *
 * A verbatim transcription of the T-2107 design of record's OWN declared public
 * API surface, taken only from the design document
 * (Claude/Vitruvius/t2107-gpu-core-1p0-design-2026-08-14.md, records@3643bc8278).
 * Nothing here is invented: every declaration below cites the design line it was
 * copied from. Where the design gives a return type, that return type is used.
 * Where the design gives NO return type, `void` is used, because a C declaration
 * must have one and the design supplies none -- that substitution is itself under
 * test and is marked [NO RETURN TYPE IN DESIGN].
 *
 * The design's `name(args) -> Type` prose form is rendered as `Type name(args);`,
 * which is the most favourable possible reading of it.
 */
#ifndef SSLM_GPU_1P0_H
#define SSLM_GPU_1P0_H

#include <stdint.h>

/* --- opaque handle types, design line 244-247 (Sec4.1) --- */
typedef struct SslmGpuContext        SslmGpuContext;
typedef struct SslmGpuModelHandle    SslmGpuModelHandle;
typedef struct SslmGpuAdapterHandle  SslmGpuAdapterHandle;
typedef struct SslmGpuSequenceHandle SslmGpuSequenceHandle;

/* Config/view types the design names but does not declare; stubbed so the probe
 * compiles as far as the design's own surface allows. */
typedef struct GpuContextConfig   GpuContextConfig;
typedef struct GpuResidencyConfig GpuResidencyConfig;
typedef struct SslmModelView      SslmModelView;

/* --- status enum ---
 * Substrate at D:\SuperSLM main@495fbb4, include/superslm/gpu_port.h:472:
 *     enum class SslmGpuStatus { Ok, DispatchBudgetTooSmall, Busy };
 * Design Sec9 (line 545) "Extends, never replaces" it with the rows of its own
 * error-taxonomy table (design lines 556-562). All seven rows transcribed. */
typedef enum SslmGpuStatus {
    SSLM_OK = 0,
    SSLM_DISPATCH_BUDGET_TOO_SMALL,      /* substrate, gpu_port.h:472           */
    SSLM_BUSY,                           /* design line 556                     */
    SSLM_CONTEXT_HAS_LIVE_HANDLES,       /* design line 557                     */
    SSLM_MODEL_HAS_LIVE_SEQUENCES,       /* design line 558                     */
    SSLM_ADAPTER_MODEL_MISMATCH,         /* design line 559                     */
    SSLM_ADAPTER_BASE_HASH_MISMATCH,     /* design line 560                     */
    SSLM_SEQUENCE_KV_BUFFER_MISMATCH,    /* design line 561                     */
    SSLM_DEVICE_LOST                     /* design line 562                     */
} SslmGpuStatus;

/* --- Sec4.1.1, design line 251: context create --- */
SslmGpuContext* sslm_gpu_context_create(GpuContextConfig cfg);

/* --- Sec4.1.1, design line 266: context destroy ---
 * [NO RETURN TYPE IN DESIGN]. Design line 266-269 states this call
 * "asserts (debug) or fails loudly (release, a new status
 * `ContextHasLiveHandles`)". The declaration it gives is:
 *     sslm_gpu_context_destroy(SslmGpuContext*)                                */
void sslm_gpu_context_destroy(SslmGpuContext* ctx);

/* --- Sec5.1, design line 327: model map --- */
SslmGpuModelHandle* sslm_gpu_model_map(SslmGpuContext* ctx,
                                       const SslmModelView* base,
                                       GpuResidencyConfig cfg);

/* --- Sec5.1, design line 338: model unmap ---
 * [NO RETURN TYPE IN DESIGN]. Design line 339-340: freeing with live sequences
 * "is a caller error (`ModelHasLiveSequences`, Sec9)". Sec4.2's table (line 287)
 * additionally gives it the SSLM_BUSY precondition. Declaration given:
 *     sslm_gpu_model_unmap(SslmGpuContext*, SslmGpuModelHandle*)               */
void sslm_gpu_model_unmap(SslmGpuContext* ctx, SslmGpuModelHandle* model);

/* --- Sec5.2, design line 345: adapter map ---
 * Design line 349-350: "validates the adapter's base-hash ... an adapter
 * converted against a different base model is rejected here, at map time".
 * Sec9 line 560 assigns `AdapterBaseHashMismatch` to this call site.           */
SslmGpuAdapterHandle* sslm_gpu_adapter_map(SslmGpuContext* ctx,
                                           SslmGpuModelHandle* model,
                                           const SslmModelView* adapter_artifact);

/* --- adapter unmap: named at design line 587 (Sec10 B6) as
 * "`sslm_gpu_adapter_map`/`unmap`". NO SIGNATURE IS GIVEN ANYWHERE IN THE
 * DESIGN. Transcribed by analogy to model unmap, which is a guess, not a
 * transcription -- marked as such.                                            */
void sslm_gpu_adapter_unmap(SslmGpuContext* ctx, SslmGpuAdapterHandle* adapter);

/* --- Sec5.3, design line 369: sequence create ---
 * Sec9 line 561 assigns `SequenceKvBufferMismatch` to "create time".          */
SslmGpuSequenceHandle* sslm_gpu_seq_create(SslmGpuContext* ctx,
                                           SslmGpuModelHandle* model,
                                           int64_t context_cap);

/* --- Sec5.3, design line 379: sequence release ---
 * [NO RETURN TYPE IN DESIGN]. Sec4.2's table (line 286) requires this call to
 * return SSLM_BUSY against a `Submitted` sequence. Note also that Sec4.2 names
 * the same call `sslm_seq_release` while Sec5.3 names it `sslm_gpu_seq_release`. */
void sslm_gpu_seq_release(SslmGpuContext* ctx, SslmGpuSequenceHandle* seq);

/* --- Sec4.3, design lines 298-309: the two decode calls.
 * These two are the ONLY entry points in the design carrying a complete C
 * signature, and they are transcribed character-for-character.                */
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
    uint32_t dispatch_budget_per_sequence);

/* --- sslm_gpu_ready: D-SLM3294's own named half of the async lifecycle.
 * NO SIGNATURE BLOCK EXISTS IN THE DESIGN. Sec4.2's table (lines 284-285)
 * describes `block`, `*out_ready` and `*out_status`; the substrate's pure-policy
 * analogue is GpuReadySignalsCompletion(bool, int32_t*, ...) at gpu_port.h:502.
 * Reconstructed below -- a guess, not a transcription.                        */
SslmGpuStatus sslm_gpu_ready(SslmGpuContext* ctx,
                             SslmGpuSequenceHandle* seq,
                             int32_t block,
                             int32_t* out_ready,
                             SslmGpuStatus* out_status);

/* --- sslm_seq_save / sslm_seq_reset: named in Sec4.2's table (line 286) and
 * Sec5.3 (line 379). NO SIGNATURES EXIST IN THE DESIGN. Not reconstructed --
 * the probe does not need them, and inventing them would be the briefer's error
 * committed by the probe. */

#endif /* SSLM_GPU_1P0_H */

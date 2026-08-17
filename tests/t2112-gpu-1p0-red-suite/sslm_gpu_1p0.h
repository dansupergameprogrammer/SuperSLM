/* sslm_gpu_1p0.h -- T-2112 red suite, promoted from the T-2111 strike probe.
 *
 * PROMOTION (T-2112, Curie, 2026-08-15): this file is the canonical copy of
 * Claude/Loki/t2111-probe/sslm_gpu_1p0.h (Wizard repo), moved here per the
 * T-2107 design's own Sec10 B9 / Sec11 dim 7 instruction -- "the probe
 * pattern ... promoted from a one-time strike instrument to a standing suite
 * fixture" -- so it is not left behind in the casebook (T-2112 brief). Every
 * .cpp/.c file in this suite directory and in interface_probe/ includes THIS
 * copy (via `#include "sslm_gpu_1p0.h"` with an -I to this directory), so
 * there is exactly one declared-surface header in the suite, never two
 * drifting copies. Regenerate this file (never hand-patch it out of sync)
 * whenever the design's Sec4/Sec5/Sec9 changes; interface_probe/'s own cells
 * are the check that a regeneration still compiles against every claim.
 *
 * Below this note is the T-2111 fold's own file, unchanged (verbatim
 * transcription of the T-2107 design of record's repaired public API surface,
 * Claude/Vitruvius/t2107-gpu-core-1p0-design-2026-08-14.md, Sec4/Sec5,
 * post-fold). Every declaration cites the design section it was copied from.
 * The pre-fold version lives in git history (this file's own history in
 * Claude/Loki/t2111-probe/, Wizard repo), not as a second file here, per this
 * project's append-only-in-git, current-truth-in-HEAD discipline
 * (StandardsDocument.md Sec6.6).
 *
 * The repair's single structural move, stated once rather than at every
 * declaration: every fallible call returns SslmGpuStatus; every value it
 * produces (a handle, a ready flag, a decoded status, a batch's per-sequence
 * outcomes) is delivered through an out-parameter, never through the return
 * value. This is the substrate's own established convention
 * (gpu_port.h:100, "status-returning at the entry point") -- see design Sec
 * 4.1.1 for the fold's own citation of that grounding.
 */
#ifndef SSLM_GPU_1P0_H
#define SSLM_GPU_1P0_H

#include <stdint.h>
#include <stddef.h>

/* --- opaque handle types, design Sec4.1 --- */
typedef struct SslmGpuContext        SslmGpuContext;
typedef struct SslmGpuModelHandle    SslmGpuModelHandle;
typedef struct SslmGpuAdapterHandle  SslmGpuAdapterHandle;
typedef struct SslmGpuSequenceHandle SslmGpuSequenceHandle;

/* T-2112 SUITE-INTEGRATION FIX (Curie, 2026-08-15), routed back as a finding rather than silently
 * carried: the T-2111 probe's own `typedef struct SslmModelView SslmModelView;` below declared a
 * SECOND, DISTINCT global-scope type of that name -- fine for the probe's own compile-only cells
 * (never built alongside the real engine headers), but genuinely AMBIGUOUS (MSVC C2872) the
 * moment a real fixture-loading translation unit includes both this header and
 * `superslm/model.h` (which defines the REAL `superslm::SslmModelView`) under `using namespace
 * superslm;` -- two distinct types sharing one name, not one type seen twice. This suite's own
 * product cells need the REAL type (to load a real .sslm artifact via `SslmModel::Load`), so this
 * header now forward-declares `SslmModelView` INSIDE `namespace superslm` and imports it with a
 * `using`-declaration -- the design's own intent (the 1.0 API's model/adapter-artifact parameter
 * IS the engine's own model view, never a second placeholder type) restated so it actually
 * compiles against the real engine. See Claude/Curie/t2112-1p0-red-suite-2026-08-15.md
 * (Wizard repo) Sec5 for the full finding, routed to the conductor for the design record's own
 * Sec4.1 to fold explicitly the next time that section is touched. */
namespace superslm { struct SslmModelView; }
using superslm::SslmModelView;

/* T-2114 (M2, Claude/Poirot/50f3d5d-t2113-1p0-gpu-core-build-review.md): these were incomplete
 * forward declarations here while gpu_1p0.h (production) defines both as complete, by-value
 * types (design Sec4.1.1/Sec5.1 name them as call parameters, and an incomplete type cannot be
 * passed by value) -- a real divergence `diff` against gpu_1p0.h would show, not merely a stale
 * comment, and the exact reason every dim*_red.cpp file in this suite had to define its own
 * local, redundant `struct GpuContextConfig { int reserved; };` stand-in rather than using this
 * header's own type directly. Completed here to match gpu_1p0.h's own struct bodies field-for-
 * field -- the every-cell-file local stand-ins are now genuinely redundant and are removed in
 * the same fix round (each file's own struct declaration would otherwise conflict with this
 * header's now-complete one, a duplicate-definition error, not merely dead code). */
typedef struct GpuContextConfig {
	int reserved;  /* design Sec4.1.1 assigns no fields yet; zero-initialize */
} GpuContextConfig;
typedef struct GpuResidencyConfig {
	int reserved;  /* design Sec5.1 assigns no fields yet; zero-initialize */
} GpuResidencyConfig;

/* --- status enum ---
 * Substrate at D:\SuperSLM main@495fbb4, include/superslm/gpu_port.h:472:
 *     enum class SslmGpuStatus { Ok, DispatchBudgetTooSmall, Busy };
 * Design Sec9 (repaired channel table) extends it with the rows of its own
 * error-taxonomy table, now including BatchBudgetExhausted (added at this
 * fold, Sec7/Sec9). All nine extension rows transcribed, plus the two
 * substrate rows. Naming corrected to the substrate's own bare-enumerator
 * convention at this fold (was SSLM_-prefixed; design Sec9's own sweep note). */
typedef enum SslmGpuStatus {
    SSLM_OK = 0,
    SSLM_DISPATCH_BUDGET_TOO_SMALL,      /* substrate, gpu_port.h:472           */
    SSLM_BUSY,                           /* design Sec9                         */
    SSLM_CONTEXT_HAS_LIVE_HANDLES,       /* design Sec9                         */
    SSLM_MODEL_HAS_LIVE_SEQUENCES,       /* design Sec9                         */
    SSLM_ADAPTER_MODEL_MISMATCH,         /* design Sec9                         */
    SSLM_ADAPTER_BASE_HASH_MISMATCH,     /* design Sec9                         */
    SSLM_SEQUENCE_KV_BUFFER_MISMATCH,    /* design Sec9                         */
    SSLM_DEVICE_LOST,                    /* design Sec9                         */
    SSLM_BATCH_BUDGET_EXHAUSTED,         /* design Sec9, ADDED at this fold     */
    SSLM_TOKEN_ID_OUT_OF_RANGE,          /* design Sec9, ADDED at the 2026-08-15
                                           * mini-fold (Sec18), D-SLM3367       */
    SSLM_SEQUENCE_REJECTED,               /* design Sec9, ADDED T-2114 (S1, review
                                           * 50f3d5d): a per-sequence CPU-domain
                                           * guard/decode rejection, distinct from
                                           * SSLM_DEVICE_LOST -- see gpu_1p0.h's
                                           * own header comment on this enumerator
                                           * for the full account.               */
    SSLM_RESTORE_MODEL_MISMATCH          /* design Sec9/Sec22, ADDED at the 2026-08-15
                                           * mini-fold (P2, D-SLM3415): sslm_gpu_seq_restore's
                                           * v3 blob model_content_hash does not match the
                                           * target model handle's own RawIntegrityHash() --
                                           * see gpu_1p0.h's own header comment for the full
                                           * account. Appended LAST, same S1 precedent. */
} SslmGpuStatus;

/* --- Sec4.1.1: context create/destroy --- */
SslmGpuStatus sslm_gpu_context_create(GpuContextConfig cfg, SslmGpuContext** out_ctx);
SslmGpuStatus sslm_gpu_context_destroy(SslmGpuContext* ctx);

/* --- Sec5.1: model map/unmap --- */
SslmGpuStatus sslm_gpu_model_map(SslmGpuContext* ctx, const SslmModelView* base,
                                  GpuResidencyConfig cfg, SslmGpuModelHandle** out_model);
SslmGpuStatus sslm_gpu_model_unmap(SslmGpuContext* ctx, SslmGpuModelHandle* model);

/* --- Sec5.2: adapter map/unmap --- */
SslmGpuStatus sslm_gpu_adapter_map(SslmGpuContext* ctx, SslmGpuModelHandle* model,
                                    const SslmModelView* adapter_artifact,
                                    SslmGpuAdapterHandle** out_adapter);
SslmGpuStatus sslm_gpu_adapter_unmap(SslmGpuContext* ctx, SslmGpuAdapterHandle* adapter);

/* --- Sec5.3: sequence create/release --- */
SslmGpuStatus sslm_gpu_seq_create(SslmGpuContext* ctx, SslmGpuModelHandle* model,
                                   int64_t context_cap, SslmGpuSequenceHandle** out_seq);
SslmGpuStatus sslm_gpu_seq_release(SslmGpuContext* ctx, SslmGpuSequenceHandle* seq);

/* --- Sec5.3a: the production token-feed entry point, ADDED at the 2026-08-15 mini-fold
 * (Sec18), routing D-SLM3367. Host-only -- no dispatch, no state transition to Submitted.
 * Precondition: seq state Idle (Busy against Submitted). On success: overwrites hidden_codes/
 * hidden_scale via the identical EmbedEntry primitive the CPU path calls, resets layer_index
 * to 0. On a hostile token_id (outside [0, vocab_size)): TokenIdOutOfRange, seq state
 * untouched. */
SslmGpuStatus sslm_gpu_seq_embed_token(SslmGpuContext* ctx, SslmGpuSequenceHandle* seq,
                                        int32_t token_id);

/* --- Sec4.2: save/restore/reset. sslm_gpu_seq_save writes a v3 blob ('SLM3'), carrying the
 * model's own content hash (P2, Sec22, D-SLM3415). sslm_gpu_seq_restore rejects a model-hash
 * mismatch under SSLM_RESTORE_MODEL_MISMATCH, distinct from the generic malformed-blob
 * disposition its size-derivation ladder (N1, Sec21) uses. --- */
SslmGpuStatus sslm_gpu_seq_save(SslmGpuContext* ctx, const SslmGpuSequenceHandle* seq,
                                 void* out_blob, size_t* out_blob_size);
SslmGpuStatus sslm_gpu_seq_restore(SslmGpuContext* ctx, SslmGpuModelHandle* model,
                                    const void* blob, size_t blob_size,
                                    SslmGpuSequenceHandle** out_seq);
SslmGpuStatus sslm_gpu_seq_reset(SslmGpuContext* ctx, SslmGpuSequenceHandle* seq);

/* --- Sec4.3: the two decode calls.
 * sslm_decode_step_gpu is unchanged by this fold -- it already carried a
 * complete signature and was the strike's own CONTROL cell.
 * sslm_decode_step_batch_gpu is REPAIRED: dispatch_budget is now BATCH-WIDE
 * (was dispatch_budget_per_sequence) and out_statuses[n_sequences] is added,
 * resolving the Sec4.3-vs-Sec7 contradiction the strike's own specimen
 * finding named (D-SLM3344) and making the Sec11 dim 8 cell's cut possible
 * to construct at all. */
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

/* --- sslm_gpu_ready: D-SLM3294's own named half of the async lifecycle.
 * Full signature given at design Sec4.2, this fold. */
SslmGpuStatus sslm_gpu_ready(SslmGpuContext* ctx,
                              SslmGpuSequenceHandle* seq,
                              int32_t block,
                              int32_t* out_ready,
                              SslmGpuStatus* out_status);

/* ============================================================================
 * G5: schema-constrained GPU decoding -- PROMOTED to this shipped surface (T-2119 design
 * Sec14.2, D-SLM3477, Claude/Poirot/9bc9ec6-t2132-g5-arc-review.md S8). Mirrored here, in the
 * SAME commit as the production header's own promotion (include/superslm/gpu_1p0.h), per this
 * suite's own declaration-identical convention (this file's own top-of-file PROMOTION note) --
 * these eight declarations were on include/superslm/gpu_1p0_g5_bridge.h, a non-shipping header
 * this suite never mirrored (it was correctly outside the "declaration-identical to gpu_1p0.h"
 * contract while it stayed off the shipped surface); now that they are part of gpu_1p0.h, they
 * are part of what this file mirrors too. Declarations transcribed verbatim from
 * gpu_1p0_g5_bridge.h -- unchanged by the promotion, a relocation not a redesign.
 * ============================================================================ */

/* Sentinel matching the CPU ABI's own `kDfaWalkStateUnused` (src/sslm_abi.cpp) -- "no schema
 * ever bound" on a freshly-created sequence handle. NOT part of gpu_1p0.h's own promoted verb
 * set (it stayed on gpu_1p0_g5_bridge.h, D-SLM3477's own scope is the eight functions only);
 * mirrored here anyway so a suite TU that needs it has one declared source, matching this
 * header's own "one canonical copy" discipline. */
constexpr uint32_t kSslmGpuDfaWalkStateUnused = 0xFFFFFFFFu;

bool SslmGpuModelHasSchemasForG5Bridge(SslmGpuModelHandle* model);

int32_t SslmGpuSchemaLookupForG5Bridge(SslmGpuModelHandle* model, const char* name);

SslmGpuStatus SslmGpuSeqSetSchemaForG5Bridge(SslmGpuContext* ctx, SslmGpuSequenceHandle* seq,
                                              int32_t schema_index);

uint32_t SslmGpuSeqWalkStateForG5Bridge(SslmGpuSequenceHandle* seq);

SslmGpuStatus SslmGpuSeqPrefillPromptForG5Bridge(SslmGpuContext* ctx, SslmGpuSequenceHandle* seq,
                                                  const int32_t* tokens, int32_t count,
                                                  uint32_t dispatch_budget);

SslmGpuStatus SslmGpuSeqFinishTokenForG5Bridge(SslmGpuContext* ctx, SslmGpuSequenceHandle* seq,
                                                int32_t* out_token);

SslmGpuStatus SslmGpuSeqDecodeStepForG5Bridge(SslmGpuContext* ctx, SslmGpuSequenceHandle* seq,
                                               int32_t token_to_embed_if_needed,
                                               uint32_t dispatch_budget, int32_t* out_token);

SslmGpuStatus SslmGpuSeqPrefillSchemaContentForG5Bridge(SslmGpuContext* ctx,
                                                          SslmGpuSequenceHandle* seq,
                                                          const int32_t* tokens, int32_t count,
                                                          uint32_t dispatch_budget_per_token,
                                                          int32_t* consumed);

#endif /* SSLM_GPU_1P0_H */

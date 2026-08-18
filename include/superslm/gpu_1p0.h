#ifndef SSLM_GPU_1P0_H
#define SSLM_GPU_1P0_H
// SuperSLM GPU acceleration API (D3D12-backed) -- the public entry points for creating a GPU
// context; mapping a model and its LoRA adapters into device residency; creating, saving,
// restoring, and releasing per-sequence decode state; and running batched decode steps on the
// GPU.
//
// Every fallible call returns an SslmGpuStatus; every value it produces (a handle, a ready
// flag, a decoded status, a batch's per-sequence outcome) is delivered through an out-parameter,
// never through the return value. Every function below is declared at global scope with
// ordinary C++ linkage -- not extern "C", not inside a namespace -- and a caller linking
// against this header's implementation must match that scope exactly.
//
// A per-sequence decode-time rejection (an out-of-domain input, a guard check failing on that
// sequence's own step) is reported as SSLM_SEQUENCE_REJECTED, kept distinct from
// SSLM_DEVICE_LOST: the device stays healthy and no other sequence in the same batch call is
// affected. sslm_gpu_seq_restore additionally rejects a restore blob whose recorded
// model-content hash does not match the target model handle (SSLM_RESTORE_MODEL_MISMATCH), so
// a sequence saved from one model cannot be silently replayed against a different one.

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

/* ============================================================================
 * G5: schema-constrained GPU decoding -- PROMOTED to this shipped surface (design Sec14.2,
 * D-SLM3477, Claude/Poirot/9bc9ec6-t2132-g5-arc-review.md S8). Built and proven
 * (G5-5, T-2132) on include/superslm/gpu_1p0_g5_bridge.h, which stated it was NOT part of this
 * shipped surface -- a genuine product capability (a real host wanting schema-constrained
 * decoding on the GPU, the plan's own G-1 lineage) with proven bit-identical CPU/GPU parity (80
 * decode steps, matching SHA-256, Claude/Brunel/t2132-g5-build-2026-08-16.md session 4) had no
 * shipped entry point to reach it. Applying this design's own rung-7 precedent (Sec11.2: a
 * genuine capability promotes to production; only a state no legitimate host call could ever
 * reach stays test-only) points the opposite direction from the bridge's prior placement --
 * D-SLM3443 (GPU parity REQUIRED, no dated deferral) independently forecloses staying test-only.
 * Declarations below are UNCHANGED from gpu_1p0_g5_bridge.h's own signatures (a relocation, not a
 * redesign) -- that header now includes this one for its remaining, non-verb content (the
 * kSslmGpuDfaWalkStateUnused sentinel) and stays valid for every existing includer. Definitions
 * are unchanged, src/gpu/gpu_1p0.cpp. Promotion mechanics beyond the declaration move (CMake
 * install/export list inclusion, a dedicated declaration-parity gate against this surface's own
 * suite mirror) are owed to the builder, not performed here -- this ruling authorizes exactly the
 * declaration relocation and its suite-side mirror, per D-SLM3477's own "mechanics owed to the
 * builder" text. --- */

/* True iff `model`'s own mapped artifact carried a SchemaMasks (SCM1) section -- an artifact
 * with none is a valid, unconstrained-only artifact (design Sec13.1), exactly the CPU ABI's own
 * disposition; every schema-bound call below is meaningless against such a model. */
bool SslmGpuModelHasSchemasForG5Bridge(SslmGpuModelHandle* model);

/* Resolves a compiled schema by name against `model`'s own host-side parsed SchemaMasksTable
 * (built once, at sslm_gpu_model_map time, from the SAME section bytes the CPU ABI parses) --
 * the GPU-1.0 twin of `sslm_schema_lookup`. Returns the schema's own index (>= 0) on a match,
 * -1 on no match. */
int32_t SslmGpuSchemaLookupForG5Bridge(SslmGpuModelHandle* model, const char* name);

/* Binds `schema_index` (as returned by the lookup above; -1 unbinds, mirroring
 * SSLM_SCHEMA_NONE) to `seq`, valid ONLY when `seq`'s own DFA-walk-state is at a fresh/reset
 * start (mirrors `sslm_seq_set_schema`'s own "valid only when the sequence's DFA-walk state is
 * at its start" precondition, design Sec5) -- returns SSLM_SEQUENCE_REJECTED on a non-fresh walk
 * state. A caller-malformed handle (`seq`/`model` null, `ctx` mismatch) returns
 * SSLM_SEQUENCE_KV_BUFFER_MISMATCH, the existing "malformed handle" bucket every 1.0 entry point
 * already uses. */
SslmGpuStatus SslmGpuSeqSetSchemaForG5Bridge(SslmGpuContext* ctx, SslmGpuSequenceHandle* seq,
                                              int32_t schema_index);

/* Reads `seq`'s own current DFA-walk-state -- kSslmGpuDfaWalkStateUnused if no schema is bound. */
uint32_t SslmGpuSeqWalkStateForG5Bridge(SslmGpuSequenceHandle* seq);

/* The GPU-1.0 twin of `sslm_prefill(..., SSLM_SPAN_PROMPT, ...)` -- the REQUIRED way to prime a
 * fresh or reset sequence with a host prompt before decoding. Embeds and drives EVERY token in
 * `tokens` (including the last) to full depth, no walk-state touch, no masking. On success
 * (count > 0), sets an internal "ready for logits" flag mirroring `sslm_seq_s::ready_for_logits`
 * (src/sslm_abi.cpp) EXACTLY -- a caller's own next `SslmGpuSeqDecodeStepForG5Bridge` call
 * consumes this flag automatically. */
SslmGpuStatus SslmGpuSeqPrefillPromptForG5Bridge(SslmGpuContext* ctx, SslmGpuSequenceHandle* seq,
                                                  const int32_t* tokens, int32_t count,
                                                  uint32_t dispatch_budget);

/* Finishes a token once `seq`'s own layer loop has reached full depth (caller-ensures: drained
 * via `sslm_gpu_ready` to Idle, `seq`'s own layer_index == model->num_hidden_layers) -- runs
 * final_norm + logits (the SAME `RmsNormSite`/`LogitsSite` calls sslm_decode_step's own
 * finishing block uses) then, if a schema is bound, `superslm::ApplyMaskAndArgmax` indexed by
 * `seq`'s own walk-state -- advances the walk-state via `SchemaMasksTable::Transition`, exactly
 * `sslm_decode_step`'s own masked-argmax step. No schema bound: plain
 * `ArgmaxLowestIndexTieBreak`, byte-for-byte the pre-G5 path. On success, `*out_token` is the
 * produced token id and `seq`'s own layer_index resets to 0. Returns SSLM_SEQUENCE_REJECTED if
 * the precondition (full depth reached) does not hold. */
SslmGpuStatus SslmGpuSeqFinishTokenForG5Bridge(SslmGpuContext* ctx, SslmGpuSequenceHandle* seq,
                                                int32_t* out_token);

/* THE RECOMMENDED one-call-per-decode-step entry point -- the GPU-1.0 twin of
 * `sslm_decode_step`'s own composition (embed-if-needed, layer-loop-to-depth, finish), including
 * its `ready_for_logits` shortcut verbatim (src/sslm_abi.cpp): if a prior
 * `SslmGpuSeqPrefillPromptForG5Bridge`/`SslmGpuSeqPrefillSchemaContentForG5Bridge` call left that
 * flag set, `token_to_embed_if_needed` is IGNORED and this call finishes the already-computed
 * residual directly (no embed, no layer loop); otherwise it embeds `token_to_embed_if_needed`,
 * drives it to full depth, and finishes. A caller that always calls this once per decode step --
 * rather than hand-composing embed/`sslm_decode_step_gpu`/`sslm_gpu_ready`/
 * `SslmGpuSeqFinishTokenForG5Bridge` itself -- cannot reproduce the duplicate-KV-commit class of
 * bug session 3/4 of the T-2132 build found and fixed, by construction. */
SslmGpuStatus SslmGpuSeqDecodeStepForG5Bridge(SslmGpuContext* ctx, SslmGpuSequenceHandle* seq,
                                               int32_t token_to_embed_if_needed,
                                               uint32_t dispatch_budget, int32_t* out_token);

/* Jump-forward's own GPU twin. Drives `count` FORCED tokens (already known -- never chosen, no
 * masking/argmax involved, exactly `PrefillWholeTokensImpl`'s own SSLM_SPAN_SCHEMA_CONTENT
 * branch, src/sslm_abi.cpp) through the full embed -> layer-loop-to-depth -> commit sequence, ONE
 * token at a time (each token issued as up to `dispatch_budget_per_token`-sized
 * `sslm_decode_step_gpu` calls, internally). Reachability is checked BEFORE each token's own
 * forward pass (`SchemaMasksTable::Transition` against `seq`'s own current walk-state) -- a
 * token that leaves the DFA's language is rejected (SSLM_SEQUENCE_REJECTED). Only the REJECTED
 * token's own effects are withheld: its own walk-state transition never applies, its own forward
 * pass never runs, and `*consumed` is never incremented for it. Tokens admitted BEFORE it in the
 * same call already had their walk-state advance, K/V write, and layer loop run for real, and
 * `*consumed` already counts them -- the documented partial-consumption contract (matching
 * `sslm_prefill`'s own shape), not full-call atomicity (design Sec14.3, D-SLM3478 -- RULED design
 * text). Requires a schema already bound (SSLM_SEQUENCE_REJECTED otherwise). Sets the SAME
 * "ready for logits" flag `SslmGpuSeqPrefillPromptForG5Bridge` sets whenever `*consumed > 0` --
 * including on the rejection path, when earlier tokens in the same call were already admitted. */
SslmGpuStatus SslmGpuSeqPrefillSchemaContentForG5Bridge(SslmGpuContext* ctx,
                                                          SslmGpuSequenceHandle* seq,
                                                          const int32_t* tokens, int32_t count,
                                                          uint32_t dispatch_budget_per_token,
                                                          int32_t* consumed);

#endif /* SSLM_GPU_1P0_H */

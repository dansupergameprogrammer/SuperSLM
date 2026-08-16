/* sslm_abi.h -- T-2138 (Curie) red suite: the canonical declared surface for the Layer-1
 * CPU-side sslm_* consumer C ABI.
 *
 * Transcribed verbatim from the design of record, Claude/Vitruvius/
 * t2133-layer1-c-abi-design-2026-08-16.md Sec8 (Wizard repo) -- the exact code block that
 * section ships, itself derived verbatim (Sec8's own stated method) from
 * tests/t2130-g5-red-suite/sslm_g5.h @ curie/t2130-g5-red-suite@760b12b for every verb that
 * header already declares: same argument count, same argument order, same pointer-vs-value
 * convention, per verb, with no re-derivation from plan prose and no per-verb normalization.
 * This is the SAME promotion discipline tests/t2112-gpu-1p0-red-suite/sslm_gpu_1p0.h and
 * tests/t2130-g5-red-suite/sslm_g5.h both established: one canonical copy in this directory,
 * included by every .cpp fixture in this suite via `-I .`, never hand-patched out of sync with
 * the design -- regenerate this file from the design of record Sec8 if that section changes.
 *
 * Every function below is RED BY LINK against the real engine: this header declares the
 * surface; no .cpp anywhere in this repo (grep confirmed at authoring commit,
 * main@e2feb7d0) defines any `sslm_*` symbol under these names -- D-SLM3450, the design's own
 * Sec1 grounding. Every cell that calls one of these symbols compiles clean and fails to link
 * (LNK2019), per this suite's own build_link_red.bat and the house convention
 * tests/t2112-gpu-1p0-red-suite/build_link_red.bat / tests/t2130-g5-red-suite/build_link_red.bat
 * both established.
 *
 * TWO TYPES THIS HEADER CANNOT COMPLETE -- routed as a Coverage-Model gap (Curie's own
 * jurisdiction: realize the model, route what it does not say, never invent a body it never
 * gave). The design's Sec8 code block passes `const sslm_config*` (sslm_workspace_size,
 * sslm_workspace_create) and `sslm_detok_state*` (sslm_detokenize_stream) but defines the body
 * of NEITHER struct anywhere in the document -- Sec7/Sec8's own text states only that a
 * hostile `sslm_config` would carry "an out-of-range `layer_budget`, negative `chunk_budget`"
 * (Sec10 dimension 2), which names two PROBABLE field names, not a field layout Curie can
 * build a struct from with any confidence (no declared type width, no declared valid domain
 * beyond "out-of-range", no confirmation these are the ONLY fields). Declared here as
 * INCOMPLETE (forward-declared, no body) rather than guessed at, so every verb that takes a
 * pointer to either type still compiles and links red for the right reason; no cell in this
 * suite constructs an instance of either type or dereferences a field of either -- see this
 * suite's own test-design record (Claude/Curie/t2138-abi-red-suite-2026-08-16.md) for the
 * routed finding.
 */
#ifndef SSLM_T2138_ABI_H
#define SSLM_T2138_ABI_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* opaque handles -- design Sec8, verbatim */
typedef struct sslm_model_s*     sslm_model;
typedef struct sslm_seq_s*       sslm_seq;
typedef struct sslm_prefix_s*    sslm_prefix;
typedef struct sslm_adapter_s*   sslm_adapter;
typedef struct sslm_kv_pool_s*   sslm_kv_pool;
typedef struct sslm_workspace_s* sslm_workspace;

/* INCOMPLETE by design -- see this file's own header comment, "TWO TYPES THIS HEADER CANNOT
 * COMPLETE". Never given a body in Claude/Vitruvius/t2133-layer1-c-abi-design-2026-08-16.md. */
struct sslm_config;
struct sslm_detok_state;
typedef struct sslm_config sslm_config;
typedef struct sslm_detok_state sslm_detok_state;

/* status enum -- design Sec6's full per-cause taxonomy, 17 enumerators (design Sec10 dim5's own
 * reconciled count: 1 + 3 + 4 + 7 + 2 = 17). SSLM_RESTORE_SCHEMA_MISMATCH is explicitly NOT one
 * of these (design Sec10 dim5, Sec7.3) -- it is G5's own status, reserved-but-unbuilt here. */
typedef enum sslm_status {
    SSLM_OK = 0,

    /* argument/precondition rejections (design Sec6) */
    SSLM_INVALID_ARGUMENT,
    SSLM_BUFFER_TOO_SMALL,
    SSLM_MISALIGNED_BUFFER,

    /* artifact/content rejections (design Sec6) */
    SSLM_ARTIFACT_REJECTED,
    SSLM_ADAPTER_MODEL_MISMATCH,
    SSLM_RESTORE_MODEL_MISMATCH,
    SSLM_RESTORE_KV_MISMATCH,

    /* lifecycle/precondition-on-state rejections (design Sec6) */
    SSLM_MODEL_HAS_LIVE_SEQUENCES,
    SSLM_POOL_HAS_LIVE_HANDLES,
    SSLM_ADAPTER_HAS_LIVE_SEQUENCES,
    SSLM_ADAPTER_SWAP_MIDTOKEN_REJECTED,
    SSLM_SEQ_RESET_MIDTOKEN_REJECTED,
    SSLM_PREFIX_FROZEN_REJECTED,
    SSLM_KV_POOL_EXHAUSTED,

    /* numeric/domain rejections (design Sec6) */
    SSLM_TOKEN_ID_OUT_OF_RANGE,
    SSLM_CONTEXT_CAP_EXCEEDED
} sslm_status;

/* design Sec8: "carried unchanged from t2119-g5-constrained-decoding-design-2026-08-16.md
 * Sec5" -- transcribed here from tests/t2130-g5-red-suite/sslm_g5.h (the same source Sec8
 * itself cites), verbatim, since this design does not restate G5's own struct bodies. */
typedef enum sslm_span_kind {
    SSLM_SPAN_PROMPT = 0,
    SSLM_SPAN_SCHEMA_CONTENT = 1
} sslm_span_kind;

typedef struct sslm_decode_params {
    int32_t layer_budget;
} sslm_decode_params;

typedef struct sslm_stats_out {
    int64_t decode_step_ceiling;
    int64_t decode_step_actual;
    int64_t forced_token_count;
    int32_t kv_blocks_resident;
} sslm_stats_out;

/* ============================================================================
 * model lifecycle -- design Sec8: 3 args, NO config parameter (sslm_g5.h:149, verbatim).
 * ============================================================================ */
sslm_status sslm_model_map(const void* data, size_t size, sslm_model* out);
sslm_status sslm_model_unmap(sslm_model model);

/* ============================================================================
 * sizing (design Sec7) -- pure functions of an already-loaded sslm_model.
 * ============================================================================ */
size_t sslm_workspace_size(sslm_model model, const sslm_config* config);
size_t sslm_kv_block_size(sslm_model model);
size_t sslm_kv_pool_overhead_size(sslm_model model, uint32_t block_count);
size_t sslm_seq_state_size(sslm_model model);

/* ============================================================================
 * construction over caller memory (design Sec7 -- NEW, not in the plan's Sec12 sketch or
 * sslm_g5.h at all; this design's own invented pair, per verb, over caller-owned buffers).
 * ============================================================================ */
sslm_status sslm_workspace_create(sslm_model model, const sslm_config* config,
                                   void* buf, size_t buf_size, sslm_workspace* out);
sslm_status sslm_workspace_destroy(sslm_workspace ws);

/* block_count is explicit, no convenience default (D-SLM3454). Sizing recipe (design Sec7.2):
 * for N sequences concurrently resident at up to L tokens of context each,
 *   block_count = ceil((N * L) / sslm_kv_block_size(model))
 * -- single sequence (N=1, L=context_cap) reduces to ceil(context_cap / kv_block_size(model));
 * a shared prefix's tokens count once, not once per adopting sequence. */
sslm_status sslm_kv_pool_create(sslm_model model, void* buf, size_t buf_size,
                                 uint32_t block_count, sslm_kv_pool* out);
sslm_status sslm_kv_pool_destroy(sslm_kv_pool pool);

/* ============================================================================
 * prefix lifecycle -- design Sec8: kv_pool by POINTER, not value (sslm_g5.h:152, verbatim).
 * ============================================================================ */
sslm_status sslm_prefix_begin(sslm_model model, sslm_kv_pool* pool, sslm_prefix* out);

/* the token-admission verb the plan's own Sec12 sketch never named (Loki T-2136 Sec9, folded);
 * built at G5's rung-7-ruled 8-argument shape (sslm_g5.h) because Sec7.2's own prefix-sharing
 * scope needs SOME way to admit tokens before freeze, G5 or not. Valid only before
 * sslm_prefix_freeze -- after freeze or release, SSLM_PREFIX_FROZEN_REJECTED. */
sslm_status sslm_prefix_prefill(sslm_model model, sslm_prefix prefix, const int32_t* tokens,
                                 int32_t count, int32_t chunk_budget, sslm_span_kind kind,
                                 sslm_workspace ws, int32_t* consumed);
sslm_status sslm_prefix_freeze(sslm_prefix prefix);
sslm_status sslm_prefix_release(sslm_prefix prefix);

/* ============================================================================
 * sequence lifecycle -- design Sec8: kv_pool by POINTER, not value (sslm_g5.h:156/161,
 * verbatim).
 * ============================================================================ */
sslm_status sslm_seq_create(sslm_model model, sslm_kv_pool* pool, sslm_seq* out);
sslm_status sslm_seq_release(sslm_seq seq);
sslm_status sslm_seq_reset(sslm_seq seq);
sslm_status sslm_seq_adopt_prefix(sslm_seq seq, sslm_prefix prefix);
sslm_status sslm_seq_save(sslm_seq seq, void* buf, size_t* n);
sslm_status sslm_seq_restore(sslm_model model, sslm_kv_pool* pool, const void* buf, size_t n,
                              sslm_seq* out);
sslm_status sslm_seq_set_adapter(sslm_seq seq, sslm_adapter adapter);

/* ============================================================================
 * adapter lifecycle (design Sec4 -- S-LoRA-serial's own outstanding ABI debt).
 * ============================================================================ */
sslm_status sslm_adapter_map(const void* data, size_t size, sslm_model base, sslm_adapter* out);
sslm_status sslm_adapter_release(sslm_adapter adapter);
size_t      sslm_adapter_residency(sslm_adapter adapter);

/* ============================================================================
 * generation -- sslm_prefill at the G5-repaired 8-argument signature (design Sec4).
 * ============================================================================ */
sslm_status sslm_prefill(sslm_model model, sslm_seq seq, const int32_t* tokens, int32_t count,
                          int32_t chunk_budget, sslm_span_kind kind, sslm_workspace ws,
                          int32_t* consumed);
sslm_status sslm_decode_step(sslm_model model, sslm_seq* seqs, int32_t n,
                              const sslm_decode_params* params, sslm_workspace ws,
                              int32_t* out_tokens);

/* ============================================================================
 * text I/O (S-FREEZE only, design Sec4; settled build-now by D-SLM3452).
 * ============================================================================ */
sslm_status sslm_tokenize(sslm_model model, const char* utf8, int32_t* tokens, int32_t* n);
sslm_status sslm_detokenize_stream(sslm_model model, sslm_detok_state* state, const int32_t* tokens,
                                    int32_t n, char* utf8, int32_t* out_n);

sslm_status sslm_stats(sslm_model model, sslm_seq seq, sslm_stats_out* out);

#ifdef __cplusplus
}
#endif

#endif /* SSLM_T2138_ABI_H */

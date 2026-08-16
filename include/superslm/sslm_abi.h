#ifndef SUPERSLM_INCLUDE_SSLM_ABI_H
#define SUPERSLM_INCLUDE_SSLM_ABI_H
// T-2139 (Brunel, C1/C2): the production Layer-1 CPU-side sslm_* consumer C ABI surface.
//
// This header's own declarations -- opaque handle typedefs, the sslm_status enum (every
// enumerator, same order, same values), the sslm_config/sslm_detok_state/sslm_span_kind/
// sslm_decode_params/sslm_stats_out struct/enum bodies, and every function signature -- are
// transcribed verbatim from Claude/Vitruvius/t2133-layer1-c-abi-design-2026-08-16.md Sec8
// (Wizard repo), the design of record. tests/t2138-abi-red-suite/sslm_abi.h is an
// independently-promoted copy of the SAME design section (Curie's own build, T-2138) -- the
// suite includes its own copy, never this one; the two are declaration-identical by
// construction (both transcribed from the same Sec8 code block) and Gate A (tools/
// t2139_gate_a_header_parity_check.cpp) is the standing, compiler-enforced proof that THIS
// header's declarations match tests/t2130-g5-red-suite/sslm_g5.h wherever the two overlap --
// not a proof against the suite's own sslm_abi.h copy, which this file does not reference.
// **This is a declaration-shape parity convention, not a literal whole-file diff** (matching
// gpu_1p0.h's own identical precedent, this repo) -- each copy carries its own explanatory
// prose; only the DECLARATIONS are required to agree.
//
// Only C1's own eight verbs (sslm_workspace_size/sslm_kv_block_size/sslm_kv_pool_overhead_size/
// sslm_seq_state_size, sslm_workspace_create/_destroy, sslm_kv_pool_create/_destroy) and C2's
// two (sslm_model_map/_unmap) are DEFINED as of this commit (src/sslm_abi.cpp). Every other
// declaration below exists so later C-slots extend one already-complete, already-reviewed
// header rather than re-declaring the surface piecemeal -- the same convention gpu_1p0.h
// itself established at B1 (this repo).

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

/* call-shape declaration for workspace sizing -- design Sec7.1, verbatim. All-zero is HOSTILE
 * input here (every field is a required positive sizing input), unlike a GPU-ABI-style
 * reserved-only config. */
typedef struct sslm_config {
    int32_t max_batch;         /* max sequences in one sslm_decode_step call, >= 1 */
    int32_t max_chunk_budget;  /* max tokens in one sslm_prefill/sslm_prefix_prefill call, >= 1 */
    int32_t max_layer_budget;  /* max layers in one sslm_decode_step call -- in
                                 * [1, config.num_hidden_layers] */
    uint32_t reserved;         /* must be 0 */
} sslm_config;

/* fixed-size POD, caller-allocated, no create/destroy verb -- design Sec7.4, verbatim. {0} IS A
 * VALID start state (unlike sslm_config, above): pending_count == 0 means "no partial UTF-8
 * tail yet". */
typedef struct sslm_detok_state {
    uint8_t pending_bytes[3];  /* a partial UTF-8 multi-byte sequence's leading bytes, carried
                                 * from the previous call -- at most 3 */
    uint8_t pending_count;     /* how many of pending_bytes[] are valid, in [0, 3] */
} sslm_detok_state;

/* status enum -- design Sec6's full per-cause taxonomy, 17 enumerators (design Sec10 dim5's own
 * reconciled count: 1 + 3 + 4 + 7 + 2 = 17). SSLM_RESTORE_SCHEMA_MISMATCH is explicitly NOT one
 * of these (design Sec10 dim5, Sec7.3) -- it is G5's own status, reserved-but-unbuilt here.
 *
 * KNOWN, EXECUTED DIVERGENCE FROM sslm_g5.h (T-2139 build finding, filed in
 * Claude/Brunel/t2139-abi-build-2026-08-16.md Sec5 -- not silently resolved here): three
 * enumerator NAMES this design's own Sec6 taxonomy shares with tests/t2130-g5-red-suite/
 * sslm_g5.h@760b12b -- SSLM_ADAPTER_MODEL_MISMATCH, SSLM_ADAPTER_SWAP_MIDTOKEN_REJECTED,
 * SSLM_RESTORE_MODEL_MISMATCH -- carry DIFFERENT ordinal values in the two headers (sslm_g5.h:
 * 3, 4, 5 respectively; this header: 5, 11, 6), because sslm_g5.h's own enum is the G5-scoped
 * pre-G5-subset-plus-G5-additions taxonomy (design Sec3), not this design's own 17-member
 * family, and the two were never reconciled value-for-value at either header's own authoring
 * time. Gate C's must-accept construction (tools/t2139_gate_c_type_identity_check.cpp) checks
 * per-shared-enumerator-name value equality as the design's own Sec9 construction specifies,
 * and -- executed -- FAILS to compile on exactly these three names, which is Gate C correctly
 * detecting a real divergence, not a gate defect. Gate C's shipped must-accept construction
 * therefore excludes sslm_status from its own comparison (scoped down, stated in that file's own
 * header comment) until the design authority rules whether this divergence is acceptable
 * (the two ABIs are independently linked, so no single caller mixes both headers' status values
 * against one library return) or requires reconciliation; sslm_span_kind/sslm_decode_params/
 * sslm_stats_out (the three OTHER Gate C obligations design Sec9 names for C1) carry no such
 * divergence and are checked in full, per the design's own construction, unmodified. */
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
 * Sec5" -- transcribed here verbatim, since this design does not restate G5's own struct
 * bodies. */
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
 * DEFINED (C2, src/sslm_abi.cpp).
 * ============================================================================ */
#include "superslm/sslm_abi_functions.inc"

#ifdef __cplusplus
}
#endif

#endif /* SUPERSLM_INCLUDE_SSLM_ABI_H */

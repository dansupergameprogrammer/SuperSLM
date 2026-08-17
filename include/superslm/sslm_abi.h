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

/* status enum -- design Sec6's full per-cause taxonomy, 18 enumerators (design Sec10 dim5's own
 * reconciled count: 1 + 3 + 4 + 7 + 3 = 18, the numeric/domain group now carrying
 * SSLM_TOKEN_ID_UNMAPPED alongside SSLM_TOKEN_ID_OUT_OF_RANGE/SSLM_CONTEXT_CAP_EXCEEDED).
 * SSLM_RESTORE_SCHEMA_MISMATCH is explicitly NOT one of these (design Sec10 dim5, Sec7.3) -- it
 * is G5's own status, reserved-but-unbuilt here.
 *
 * SSLM_TOKEN_ID_UNMAPPED -- NEW (design commit 212de7742c, the padded-vocabulary ruling, Brunel
 * T-2139), appended at ordinal 17, the next free base ordinal, per the append-only reconciliation
 * law (never inserted within the numeric/domain group's own prior position, even though it is
 * semantically a numeric/domain rejection -- ordinal stability for every already-reconciled
 * enumerator wins). sslm_detokenize_stream's own rejection for a decode-output token id in
 * [tok_vocab, cfg_vocab) -- see that function's own header comment.
 *
 * *** COORDINATION NOTE FOR THE SUITE OWNER (tests/t2130-g5-red-suite/sslm_g5.h) ***
 * sslm_g5.h's own reconciled enum (curie/t2130-g5-red-suite@a7655dd) currently carries this
 * design's 0-16 base verbatim, with G5's own seven additions appended at 17-23 (24 total). This
 * new enumerator inserts at ordinal 17 in THIS design's own base -- sslm_g5.h's copy needs
 * SSLM_TOKEN_ID_UNMAPPED added at 17, with G5's own seven enumerators shifted to 18-24 (25
 * total), to stay reconciled. Not performed here (suite ownership, StandardsDocument Sec6.4/
 * Sec5.2's own persona-boundary discipline) -- Gate C (tools/t2139_gate_c_type_identity_check.cpp)
 * will need its own real-values check extended to cover this 18th enumerator once sslm_g5.h
 * reconciles; until then Gate C's existing 17-enumerator coverage is unaffected (it never checks
 * an enumerator neither side yet defines).
 *
 * RECONCILED against tests/t2130-g5-red-suite/sslm_g5.h (Claude/Brunel/
 * t2139-abi-build-2026-08-16.md Sec4/Sec5's own executed finding, since resolved). Executing
 * Gate C's own per-shared-enumerator-value construction once found three names --
 * SSLM_ADAPTER_MODEL_MISMATCH, SSLM_ADAPTER_SWAP_MIDTOKEN_REJECTED, SSLM_RESTORE_MODEL_MISMATCH
 * -- carrying different ordinal values between this header and sslm_g5.h's own then-current
 * enum (a G5-scoped pre-G5-subset-plus-G5-additions taxonomy the two were never reconciled
 * against). sslm_g5.h has since been reconciled (curie/t2130-g5-red-suite@a7655dd) for the prior
 * 17-enumerator base; see the coordination note above for this fold's own follow-up. Gate C's
 * must-accept construction (tools/t2139_gate_c_type_identity_check.cpp) checks every one of the
 * (pre-this-fold) 17 enumerators against sslm_g5.h's own real body, in full, with no exclusion. */
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
    SSLM_CONTEXT_CAP_EXCEEDED,

    /* NEW, appended (design commit 212de7742c, ordinal 17, the next free base ordinal --
     * append-only, never inserted above): sslm_detokenize_stream's own input id in
     * [tok_vocab, cfg_vocab) -- a legal decode-output id with no tokenizer entry (the padded-
     * vocabulary case ValidateTokenizerVocabSizeJoin's loosening admits, src/model.cpp).
     * Distinct from SSLM_TOKEN_ID_OUT_OF_RANGE (id >= cfg_vocab, never a legal decode output at
     * all). */
    SSLM_TOKEN_ID_UNMAPPED = 17,

    /* NEW, appended at ordinal 25 -- N3 (Claude/Poirot/2c18dab-t2139-abi-build-review.md Sec6.3),
     * executed under the coordinator's own explicit closing-round instruction ("catch at the
     * boundary and return the design's allocation-failure status per the taxonomy"). This
     * design's own Sec6 base taxonomy (0-16) plus SSLM_TOKEN_ID_UNMAPPED (17) is otherwise
     * exhaustively reconciled against tests/t2130-g5-red-suite/sslm_g5.h, whose own seven G5-
     * scoped additions occupy 18-24 (curie/t2130-g5-red-suite@59e26ff) -- 25 is the next ordinal
     * neither side has ever claimed, so this insertion needs no renumbering of anything already
     * coordinated (unlike SSLM_TOKEN_ID_UNMAPPED's own insertion at 17, which did). Seven
     * `extern "C"` verbs (sslm_workspace_create, sslm_kv_pool_create, sslm_model_map,
     * sslm_prefix_begin, sslm_seq_create, sslm_seq_restore, sslm_adapter_map) previously let
     * std::bad_alloc cross this ABI boundary under /EHc, where MSVC's own C4297 diagnostic (63
     * times a build.bat run) states plainly that a throwing extern "C" function's behavior is
     * undefined -- this design's own Sec6 deliberately declined a resource-exhaustion catch-all
     * when the base taxonomy was first cut ("What this design does NOT add: a SSLM_DEVICE_LOST-
     * shaped catch-all"), so no existing enumerator names this cause.
     *
     * RATIFIED at design commit 9f84d9e4ca ("ratify SSLM_ALLOCATION_FAILED (T-2139 N3) --
     * resource-exhaustion family, ordinal 25, append-only law held") -- P3 (Claude/Poirot/
     * 2c18dab-t2139-abi-build-review.md Sec7.5, third confirmation pass): this comment previously
     * said the ratification was still pending and told a reader to go chase an authorization that
     * by then already existed; corrected in place rather than left to mislead the next reader.
     *
     * P2 (same casebook, Sec7.4): this insertion left the ordinal space INTERLEAVED --
     * SSLM_ALLOCATION_FAILED (25) sits numerically ABOVE sslm_g5.h's own G5 block (18-24), not
     * below it, breaking the arc's own "base low, G5 appended above" arrangement and leaving no
     * rule for who takes 26. That governance question is being ruled at the planner in parallel
     * to this round; SSLM_STATUS_BASE_MAX below is this header's own honest, single-source-of-
     * truth answer to "what is the base taxonomy's real highest ordinal today" in the meantime --
     * update it in the SAME change that appends any future base enumerator, and take the
     * planner's own ruling at landing if it changes this arrangement's shape. */
    SSLM_ALLOCATION_FAILED = 25
} sslm_status;

/* P2 (Claude/Poirot/2c18dab-t2139-abi-build-review.md Sec7.4): the base taxonomy's own real
 * highest ordinal, maintained by hand alongside the enum above (a C enum has no built-in "my own
 * maximum" query) -- tools/t2139_gate_c_type_identity_check.cpp's own ordinal-disjointness
 * assertion reads THIS, not a specific enumerator's name, so its own message ("the base
 * taxonomy's own real maximum") stays true regardless of which enumerator is appended next. */
#define SSLM_STATUS_BASE_MAX SSLM_ALLOCATION_FAILED

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

/* S3 (Claude/Poirot/2c18dab-t2139-abi-build-review.md): the alignment sslm_workspace_create and
 * sslm_kv_pool_create both require of their caller-supplied `buf`, on pain of
 * SSLM_MISALIGNED_BUFFER -- previously an anonymous-namespace constant inside src/sslm_abi.cpp
 * with no counterpart in this frozen header, so a consumer holding only this header (the
 * S-FREEZE bar's own definition of "consumer") could only discover the requirement by receiving
 * SSLM_MISALIGNED_BUFFER and guessing. Exported here so a caller can pass it directly to
 * std::align (or its own aligned-allocation path) instead of transcribing the number from the
 * implementation. */
#define SSLM_ABI_ALIGNMENT_BYTES 64u

/* ============================================================================
 * model lifecycle -- design Sec8: 3 args, NO config parameter (sslm_g5.h:149, verbatim).
 * DEFINED (C2, src/sslm_abi.cpp).
 * ============================================================================ */
#include "superslm/sslm_abi_functions.inc"

#ifdef __cplusplus
}
#endif

#endif /* SUPERSLM_INCLUDE_SSLM_ABI_H */

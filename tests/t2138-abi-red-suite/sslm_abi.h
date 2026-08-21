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
 * sslm_config AND sslm_detok_state -- ROUTED GAP CLOSED (design commit 41b72091c2, "T-2133:
 * micro-fold -- define sslm_config and sslm_detok_state field layouts routed from the T-2138
 * red suite's model gaps", Sec7.1/Sec7.4). This suite's own first pass (Claude/Curie/
 * t2138-abi-red-suite-2026-08-16.md Sec6) found neither struct given a body anywhere in the
 * design and declared both incomplete rather than guess; the planner ruled both bodies and this
 * header now carries them verbatim from Sec7.1/Sec7.4/Sec8:
 *
 * sslm_config (Sec7.1) is the call-shape declaration sslm_workspace_size/sslm_workspace_create
 * size the workspace against. ALL-ZERO IS HOSTILE INPUT here (every field is a required
 * positive sizing input) -- the opposite domain from sslm_detok_state below. Domain:
 * max_batch >= 1, max_chunk_budget >= 1, 1 <= max_layer_budget <= num_hidden_layers,
 * reserved == 0. A structurally invalid config makes sslm_workspace_size return 0 (its only
 * channel) and sslm_workspace_create reject SSLM_INVALID_ARGUMENT (re-validated independently,
 * never trusting the size_t call's return) -- dim2_hostile_red.cpp's own five sub-cells drive
 * this per field.
 *
 * sslm_detok_state (Sec7.4) is a fixed-size, caller-allocated POD value type (no create/destroy
 * verb, unlike sslm_workspace/sslm_kv_pool) holding a partial UTF-8 tail across successive
 * sslm_detokenize_stream calls. THE ALL-ZERO VALUE IS A VALID START STATE here (pending_count
 * == 0 means "no partial tail pending") -- the opposite domain from sslm_config above.
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

/* status enum -- RECONCILED to Sec6's COMPLETE 26-entry ordinal registry (design commit
 * 4f4eb23896, Claude/Vitruvius/t2133-layer1-c-abi-design-2026-08-16.md Sec6 GOVERNANCE RULING,
 * FOLD RULING 2026-08-17 on Poirot's third confirmation casebook
 * 4466666-t2139-third-confirmation-review.md F1, correction 1: "sslm_abi.h also mirrors the
 * complete 26-entry registry, verbatim -- the identical convention already ruled for sslm_g5.h,
 * applied symmetrically. Neither header owns a subset of the registry to extend independently").
 * This is this suite's own THIRD, independently-maintained transcription of the enum (the same
 * casebook's F5, Minor: distinct from both include/superslm/sslm_abi.h and
 * tests/t2130-g5-red-suite/sslm_g5.h) -- carried here in the SAME implicit-sequential-value style
 * this file already used before this fold (no `= N` needed per entry; the declaration order below
 * IS the registry's own ordinal order, 0 through 25, then the sentinel).
 *
 * SSLM_RESTORE_SCHEMA_MISMATCH (ordinal 22) was previously named here as "explicitly NOT one of
 * these... reserved-but-unbuilt" -- that carve-out is retired by the symmetric-mirroring ruling
 * above: every registry entry is now declared as an enumerator VALUE this suite's own C1-C7
 * cells do not all exercise functionally, exactly the relationship this header already had to
 * SSLM_TOKEN_ID_UNMAPPED before the padded-vocabulary fold landed it. Declaring the name is not a
 * claim this suite drives G5 behavior through it.
 *
 * WHAT GUARDS THIS COPY, AND WHAT DOES NOT (F5's own residual, named so it is not silently
 * assumed swept): Gate A (tools/t2139_gate_a_header_parity_check.cpp) and Gate C
 * (tools/t2139_gate_c_type_identity_check.cpp) both compare the REAL include/superslm/sslm_abi.h
 * against tests/t2130-g5-red-suite/sslm_g5.h ONLY -- neither gate reads or references this file at
 * all. This copy has NO compile-time parity check of its own against either real header; it is
 * guarded only by the fact that the 564-cell suite this header serves links against the real
 * src/sslm_abi.cpp (tests/t2138-abi-red-suite/build_link_red.bat) -- and C linkage resolves calls
 * by SYMBOL NAME alone, never by re-checking argument or enum-value types across translation
 * units, so a divergence here (a wrong ordinal, a missing name) would NOT be caught by the linker;
 * it would silently pass the wrong int-valued status to a CHECK() that expects a different name. A
 * dedicated Gate A/C-style comparison against this copy is a suite-owned follow-up this fold does
 * not itself add (Curie ownership, same as the pin list this round derives). */
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

    /* appended at ordinal 17, design commit 212de7742c, the padded-vocabulary ruling, Brunel
     * T-2139: sslm_detokenize_stream's own rejection for a decode-output token id in
     * [tok_vocab, cfg_vocab) -- a legal decode-output id with no tokenizer entry (the
     * padded-vocabulary case ValidateTokenizerVocabSizeJoin's loosening admits, src/model.cpp).
     * Distinct from SSLM_TOKEN_ID_OUT_OF_RANGE (id >= cfg_vocab, never a legal decode output). */
    SSLM_TOKEN_ID_UNMAPPED,

    /* G5, design Sec5 -- ordinals 18-24, meaning owned by G5 (T-2119), ordinal allocated by Sec6.
     * Declared here per the symmetric-mirroring ruling above; not functionally driven by any C1-C7
     * verb this suite's own cells exercise. */
    SSLM_SCHEMA_NOT_FOUND,
    SSLM_SCHEMA_BIND_REJECTED,
    SSLM_SCHEMA_SPAN_UNBOUND,
    SSLM_PREFIX_SCHEMA_MISMATCH,
    SSLM_RESTORE_SCHEMA_MISMATCH,
    SSLM_SCHEMA_SPAN_UNREACHABLE,
    SSLM_SCHEMA_UNSATISFIABLE,

    /* Resource-exhaustion rejection, ordinal 25 -- design Sec6 RATIFIED, Brunel T-2139 Sec19/N3;
     * FOLD RULING 2026-08-17 (design Sec6, F2) narrows the boundary catch this status covers to
     * std::bad_alloc and std::length_error alone; a bare catch (...) returns SSLM_ARTIFACT_REJECTED
     * instead (no new ordinal). Distinct from SSLM_KV_POOL_EXHAUSTED (caller-supplied-memory
     * exhaustion, an entirely different cause). */
    SSLM_ALLOCATION_FAILED,
    /* T-2199 Phase D review fix S6 (Claude/Poirot/7a3b10a-t2199-phaseD-review.md): mirrored here
     * in the SAME commit -- see tests/t2130-g5-red-suite/sslm_g5.h's own identical addition for
     * the full rationale. */
    SSLM_NUMERIC_STEP_REFUSED,

    /* Sentinel, FOLD RULING 2026-08-17 (design Sec6, F1) -- the enum's own final member, no
     * explicit value, auto-valuing to one past this header's own last explicit member and moving
     * itself on the next append. Compared against the real headers' own identical last member by
     * whichever gate is extended to cover this copy (see the guard note above); never itself a
     * valid status value any call returns. */
    SSLM_STATUS_NEXT_FREE
} sslm_status;

/* S3 (Claude/Poirot/2c18dab-t2139-abi-build-review.md): the alignment sslm_workspace_create AND
 * sslm_kv_pool_create both require of their caller-supplied buf, on pain of
 * SSLM_MISALIGNED_BUFFER -- exported so a caller can pass it directly to std::align rather than
 * transcribe it from the implementation, matching include/superslm/sslm_abi.h's own export. */
#define SSLM_ABI_ALIGNMENT_BYTES 64u

/* design Sec8: "carried unchanged from t2119-g5-constrained-decoding-design-2026-08-16.md
 * Sec5" -- transcribed here from tests/t2130-g5-red-suite/sslm_g5.h (the same source Sec8
 * itself cites), verbatim, since this design does not restate G5's own struct bodies. */
typedef enum sslm_span_kind {
    SSLM_SPAN_PROMPT = 0,
    SSLM_SPAN_SCHEMA_CONTENT = 1
} sslm_span_kind;

// T-2199 Phase D1 (plan Sec8 D1 ruling, D-SLM3476 precedent -- M2 fix,
// Claude/Poirot/7a3b10a-t2199-phaseD-review.md: "D-SLM3476A" cited previously does not exist in
// Claude/Decisions/DecisionLog.md): sslm_decode_params gained 7
// additive fields directly on the real ABI header (include/superslm/sslm_abi.h). Widened here,
// in the SAME shape, as a STRUCTURAL SAFETY FIX rather than a test-content edit: this suite's
// own fixture_common.h includes this LOCAL mirror via a bare `#include "sslm_abi.h"`
// (deliberately decoupled from the real header, matching this suite's own S-FREEZE-mirror
// convention) and passes its address to the REAL, external sslm_decode_step -- which now reads
// fields past this struct's own end. Left at the old 4-byte size, every call in this suite
// would pass a pointer to an under-sized stack allocation, and the real function's own
// `params->mode` read becomes an out-of-bounds read of whatever stack memory happens to follow
// -- a genuine memory-safety hazard this widening closes, not a change to any assertion this
// suite's own .cpp files make. Field values are irrelevant here (this struct carries no
// initializer), only the LAYOUT matching the real header's own.
typedef struct sslm_decode_params {
    int32_t layer_budget;
    /* T-2199 Phase D1 (D-SLM3794, additive-field ruling): mirrored here in lockstep with the
     * production struct (include/superslm/sslm_abi.h) -- this mirror's own field ORDER, WIDTH,
     * and COUNT must match exactly, since tools/t2141_gate_c_t2138_suite_side_check.cpp now
     * static_asserts sizeof/alignof/offsetof identity against the REAL production header
     * directly (conductor's dispute-resolution commission, mirror-copy defect class closure,
     * 2026-08-20). */
    uint32_t struct_size;  // D-SLM3797 (Phase D review addendum): FIRST new field, caller-set/library-validated
    int32_t mode;
    int32_t alpha_q15;  // T-2199 Phase D review fix C2: int32_t, matches the real header now
    int32_t anti_lm_max_order;
    int32_t top_k;
    int64_t q_ln2;
    int64_t q_b;
    int64_t q_c;
} sslm_decode_params;

typedef struct sslm_stats_out {
    int64_t decode_step_ceiling;
    int64_t decode_step_actual;
    int64_t forced_token_count;
    int32_t kv_blocks_resident;
    /* D-SLM3476 (design Sec14.1, Claude/Poirot/9bc9ec6-t2132-g5-arc-review.md S2, T-2132/Curie):
     * mirrored here in lockstep with the production struct (include/superslm/sslm_abi.h) and
     * with tests/t2130-g5-red-suite/sslm_g5.h's own identical addition -- this suite's own
     * dim7_contract_red.cpp/dim8_composition_red.cpp both stack-allocate sslm_stats_out and pass
     * it to the real sslm_stats(), so growing the production struct without this mirror growing
     * in the same commit would leave those callers' own stack allocation sized by a STALE,
     * narrower definition (the exact hazard the production build session's own STOP-and-report
     * named against tests/t2130-g5-red-suite's identical struct). This suite has no G5-specific
     * concept of its own to name the field's own semantics against (schemas are G5's), so the
     * field is carried here purely for layout parity, unread by this suite's own cells. */
    int32_t schema_accepting;
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
/* one WHOLE sequence's entire KV footprint across every layer (design Sec7.2, RULED, commit
 * fab235c1c6), never a sub-sequence PagedAttention page -- num_hidden_layers * context_cap *
 * num_key_value_heads * head_dim * 2 (K+V) * kv_precision_width. */
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

/* block_count is explicit, no convenience default (D-SLM3454); it is a COUNT OF SEQUENCES this
 * pool can back, not a token count (design Sec7.2, RULED, commit fab235c1c6 -- sslm_kv_block_size
 * is now one WHOLE sequence's own entire KV footprint, never a sub-sequence page). Sizing recipe:
 * for N sequences the caller wants concurrently resident, block_count = N -- the single-sequence
 * case (N=1, the shape the S-FREEZE example uses) is block_count = 1. A prefix under construction
 * also draws exactly one whole block. Prefix adoption COPIES the prefix's own occupied bytes into
 * the adopting sequence's own block (Sec7.2's copy-on-adopt ruling) -- it does not extend how many
 * blocks the cohort needs beyond one per adopting sequence, since sharing is not physical. */
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

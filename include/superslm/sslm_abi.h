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

/* G5 (T-2119 design Sec5, T-2132 build): the schema handle -- transcribed verbatim from
 * tests/t2130-g5-red-suite/sslm_g5.h, the same declaration-shape-parity convention this
 * whole file's own header comment states for every type here. No map/release verb -- a schema
 * is already resident wherever the model is (Sec5). SSLM_SCHEMA_NONE is the unconstrained
 * sentinel, a null handle by construction, matching sslm_seq_set_adapter's own NULL-adapter
 * convention. */
typedef struct sslm_schema_s*    sslm_schema;
#define SSLM_SCHEMA_NONE ((sslm_schema)0)

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

/* status enum -- FULL-REGISTRY MIRROR, design Sec6's own single-authority complete ordinal
 * registry, 26 entries (0-25) plus one auto-valued sentinel (design Sec6 GOVERNANCE RULING,
 * design commit 4f4eb23896; symmetric-mirroring FOLD RULING on the third confirmation pass's F1,
 * design commit dated 2026-08-17, Claude/Poirot/4466666-t2139-third-confirmation-review.md).
 *
 * Prior to this fold this header declared only 19 of the 26 registry entries (ordinals 0-17 and
 * 25) -- G5's own schema block (18-24) existed only in tests/t2130-g5-red-suite/sslm_g5.h. F1
 * found the governance ruling's premise ("every one of the 26 entries is a shared name once
 * sslm_g5.h carries the complete list") false as written, because nothing had ever ruled the
 * symmetric obligation onto THIS header. The fold ruling corrects this: neither header owns a
 * subset of the registry to extend independently; both mirror the full, current, reconciled
 * enum verbatim, including the other side's block as enumerator VALUES this header's own C1-C7
 * build does not yet exercise functionally -- exactly the relationship sslm_g5.h already had to
 * SSLM_ALLOCATION_FAILED (25) before its own mirroring fold. Ordinal allocation stays Sec6's
 * alone; semantic ownership of G5's own schema-cause meanings (18-24) stays G5's (T-2119) --
 * only which header's enum body carries which NAMES changed.
 *
 * SSLM_STATUS_NEXT_FREE -- the enum's final member, auto-valued (no explicit numeric value:
 * the compiler assigns one past SSLM_ALLOCATION_FAILED, i.e. 26). This is the fold's OTHER
 * ruling: a plain #define "top" macro (SSLM_STATUS_BASE_MAX, retired this fold -- no design
 * record, no consumer beyond one test tool, F1 collateral) cannot move itself
 * when a header appends a new enumerator, so F1's four compiled mutations proved it silently
 * asserted nothing (Probe A: deleting the whole registry-top static_assert still compiled
 * clean). An auto-valued enum tag inside the enum body has no such gap -- it moves automatically
 * on whichever side appends, with nothing for an author to remember to hand-update. Gate C
 * (tools/t2139_gate_c_type_identity_check.cpp) asserts both headers' SSLM_STATUS_NEXT_FREE
 * equal, IN ADDITION TO (never replacing) its per-name checks: the per-name checks catch a
 * shared name claimed at different ordinals; the sentinel catches a shared ordinal claimed by
 * two different names (F1's Probe C, the collision the retired static_assert did not catch).
 *
 * The registry, complete (design Sec6 GOVERNANCE RULING table, verbatim ordinals) -- expressed
 * as an X-MACRO LIST rather than a hand-written enum body, per the fold ruling's own remedy #2:
 * "the per-name check list itself is generated from the header ... never hand-written, so a
 * newly appended enumerator cannot land without gaining a check line." SSLM_STATUS_ENUM_LIST is
 * that generation mechanism -- it is BOTH how this enum's own body is built (below) AND, by
 * design, the same list tools/t2139_gate_c_type_identity_check.cpp re-expands to emit one
 * T2139_GATE_C_STATUS_CHECK per name (see that file). A name added here without a matching Gate
 * C check is structurally impossible: there is no second place a name could be added to this
 * enum, and any TU that expands SSLM_STATUS_ENUM_LIST sees every name this one does. */
#define SSLM_STATUS_ENUM_LIST(X) \
    X(SSLM_OK) /* 0 -- the only non-error value */ \
    /* argument/precondition rejections (design Sec6) */ \
    X(SSLM_INVALID_ARGUMENT) /* 1 */ \
    X(SSLM_BUFFER_TOO_SMALL) /* 2 */ \
    X(SSLM_MISALIGNED_BUFFER) /* 3 */ \
    /* artifact/content rejections (design Sec6) */ \
    X(SSLM_ARTIFACT_REJECTED) /* 4 */ \
    X(SSLM_ADAPTER_MODEL_MISMATCH) /* 5 */ \
    X(SSLM_RESTORE_MODEL_MISMATCH) /* 6 */ \
    X(SSLM_RESTORE_KV_MISMATCH) /* 7 */ \
    /* lifecycle/precondition-on-state rejections (design Sec6) */ \
    X(SSLM_MODEL_HAS_LIVE_SEQUENCES) /* 8 */ \
    X(SSLM_POOL_HAS_LIVE_HANDLES) /* 9 */ \
    X(SSLM_ADAPTER_HAS_LIVE_SEQUENCES) /* 10 */ \
    X(SSLM_ADAPTER_SWAP_MIDTOKEN_REJECTED) /* 11 */ \
    X(SSLM_SEQ_RESET_MIDTOKEN_REJECTED) /* 12 */ \
    X(SSLM_PREFIX_FROZEN_REJECTED) /* 13 */ \
    X(SSLM_KV_POOL_EXHAUSTED) /* 14 */ \
    /* numeric/domain rejections (design Sec6) */ \
    X(SSLM_TOKEN_ID_OUT_OF_RANGE) /* 15 */ \
    X(SSLM_CONTEXT_CAP_EXCEEDED) /* 16 */ \
    /* sslm_detokenize_stream's own input id in [tok_vocab, cfg_vocab) -- a legal decode-output
     * id with no tokenizer entry (padded-vocabulary case, src/model.cpp). Distinct from
     * SSLM_TOKEN_ID_OUT_OF_RANGE (id >= cfg_vocab, never a legal decode output at all). */ \
    X(SSLM_TOKEN_ID_UNMAPPED) /* 17 */ \
    /* G5's own schema block (design Sec5, T-2119) -- ordinals 18-24 ALLOCATED by Sec6, MEANING
     * owned by G5; mirrored here verbatim, same convention sslm_g5.h already used for
     * SSLM_ALLOCATION_FAILED before this fold. Not exercised by this header's own C1-C7 build --
     * declared so the registry is complete and Gate C's per-name checks have something to
     * compare on both sides for every entry, not only the 19 that happened to already coincide. */ \
    X(SSLM_SCHEMA_NOT_FOUND) /* 18 -- sslm_schema_lookup: unknown name */ \
    X(SSLM_SCHEMA_BIND_REJECTED) /* 19 -- sslm_seq_set_schema on a non-fresh walk */ \
    X(SSLM_SCHEMA_SPAN_UNBOUND) /* 20 -- schema-content span, unbound sequence */ \
    X(SSLM_PREFIX_SCHEMA_MISMATCH) /* 21 -- sslm_seq_adopt_prefix schema mismatch */ \
    X(SSLM_RESTORE_SCHEMA_MISMATCH) /* 22 -- sslm_seq_restore schema does not resolve */ \
    X(SSLM_SCHEMA_SPAN_UNREACHABLE) /* 23 -- a fixed span not reachable under the DFA */ \
    X(SSLM_SCHEMA_UNSATISFIABLE) /* 24 -- G5-1 compiler rejection, CPU/converter-side */ \
    /* resource-exhaustion rejection (design Sec6, RATIFIED Brunel T-2139 Sec19/N3; FOLD RULING
     * on the third confirmation pass's F2, design commit dated 2026-08-17) -- process-level
     * allocation failure, distinct from SSLM_KV_POOL_EXHAUSTED (caller-supplied-memory
     * exhaustion). The boundary catch (src/sslm_abi.cpp, CatchAllocationFailure and both
     * SslmModel::Load call sites): catch(const std::bad_alloc&)/catch(const std::length_error&)
     * -- the only causes this family covers -- return this status; a final catch(...) is the
     * true, unconditional boundary and returns SSLM_ARTIFACT_REJECTED instead (no new ordinal
     * minted for it -- MapForwardStatus's own existing "no dedicated status" mapping, extended
     * here on the record). */ \
    X(SSLM_ALLOCATION_FAILED) /* 25 */

typedef enum sslm_status {
#define SSLM_STATUS_ENUM_VALUE_(name) name,
    SSLM_STATUS_ENUM_LIST(SSLM_STATUS_ENUM_VALUE_)
#undef SSLM_STATUS_ENUM_VALUE_
    /* Sentinel -- see the header comment above. NOT a real status; never returned by any verb,
     * never a valid argument. Auto-valued (no explicit value given here), always one past the
     * enum's own last real member, so appending a new enumerator to the X-macro list above moves
     * this automatically with nothing to hand-maintain. */
    SSLM_STATUS_NEXT_FREE
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

/* S3 (Claude/Poirot/2c18dab-t2139-abi-build-review.md): the alignment sslm_workspace_create and
 * sslm_kv_pool_create both require of their caller-supplied `buf`, on pain of
 * SSLM_MISALIGNED_BUFFER -- previously an anonymous-namespace constant inside src/sslm_abi.cpp
 * with no counterpart in this frozen header, so a consumer holding only this header (the
 * S-FREEZE bar's own definition of "consumer") could only discover the requirement by receiving
 * SSLM_MISALIGNED_BUFFER and guessing. Exported here so a caller can pass it directly to
 * std::align (or its own aligned-allocation path) instead of transcribing the number from the
 * implementation. */
#define SSLM_ABI_ALIGNMENT_BYTES 64u

#ifdef __cplusplus
}
#endif

#endif /* SUPERSLM_INCLUDE_SSLM_ABI_H */

/* ============================================================================
 * model lifecycle -- design Sec8: 3 args, NO config parameter (sslm_g5.h:149, verbatim).
 * DEFINED (C2, src/sslm_abi.cpp).
 * ============================================================================
 *
 * SUPERSLM_ABI_ENUM_ONLY (owed remedy 5, Claude/Poirot/3bcbe43-t2139-fourth-confirmation-
 * review.md S1): #define this BEFORE including this header to suppress the function-declaration
 * include below -- every type/enum/struct above (including sslm_status and the
 * SSLM_STATUS_ENUM_LIST macro itself) still declares normally, since that whole section is
 * governed by the SUPERSLM_INCLUDE_SSLM_ABI_H guard above and closes before this section starts.
 * This is Gate C's OWN escape hatch (tools/t2139_gate_c_real_suite_side_check.cpp): its
 * real-suite-side TU needs the REAL tests/t2130-g5-red-suite/sslm_g5.h's own function
 * declarations visible at the SAME scope this header's own functions would otherwise occupy, and
 * two extern "C" declarations of the same function name with different parameter types is
 * ill-formed ([dcl.link]) regardless of C++ namespace (C linkage collides by name only, ignoring
 * the enclosing namespace) -- so the two real headers' FUNCTIONS cannot both be visible in one TU
 * no matter how they are namespaced. Suppressing this header's own functions (which the enum-only
 * gate does not need) removes the only thing that would collide, while keeping
 * SSLM_STATUS_ENUM_LIST -- the single generation source both this enum body and Gate C's own
 * per-name checks re-expand -- available exactly as before. No consumer outside this one gate TU
 * ever defines this macro; every ordinary include of this header (the frozen public S-FREEZE
 * surface, D-SLM13) is completely unaffected.
 *
 * This section carries its OWN guard (SUPERSLM_ABI_FUNCTIONS_INCLUDED_), independent of and
 * outside SUPERSLM_INCLUDE_SSLM_ABI_H above (Claude/Poirot/ce5aff2-t2139-fifth-confirmation-
 * review.md S2): the enum-only pass sets the outer guard but must NOT be able to claim it also
 * satisfies this one, or a later plain #include of this header in the SAME TU becomes a silent
 * no-op -- the outer guard is already defined, so nothing past it re-runs, and the functions this
 * plain include was relying on to appear never do (executed: MSVC C3861 'sslm_model_map':
 * identifier not found, at the call site three includes away from the actual cause). With the
 * function section's guard independent of the outer one, an enum-only include followed by a
 * plain include in the same TU declares the functions on the SECOND include, exactly once,
 * because the type section is already satisfied and skips, while the function section's own
 * guard is still unset. */
#if !defined(SUPERSLM_ABI_ENUM_ONLY) && !defined(SUPERSLM_ABI_FUNCTIONS_INCLUDED_)
#define SUPERSLM_ABI_FUNCTIONS_INCLUDED_

#ifdef __cplusplus
extern "C" {
#endif

#include "superslm/sslm_abi_functions.inc"

#ifdef __cplusplus
}
#endif

#endif /* SUPERSLM_ABI_FUNCTIONS_INCLUDED_ */

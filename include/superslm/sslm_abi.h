#ifndef SUPERSLM_INCLUDE_SSLM_ABI_H
#define SUPERSLM_INCLUDE_SSLM_ABI_H
// SuperSLM Layer-1 C ABI -- the production, engine-agnostic C surface for a CPU-side consumer:
// workspace and KV-pool sizing and lifetime, model and adapter mapping, sequence lifetime
// (create, save, restore, reset, release), prefix and prefill, schema-constrained decode,
// tokenization and detokenization, and stats.
//
// Every declaration in this header has `extern "C"` linkage and a stable ABI shape -- opaque
// handles, POD config/state structs, and a single closed `sslm_status` enum whose enumerator
// values never change once shipped. New functionality is added by appending new enumerators
// and new functions, never by renumbering or repurposing an existing one.

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

/* G5 (design Sec5): the schema handle -- transcribed verbatim from
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
 * design commit dated 2026-08-17).
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
 * alone; semantic ownership of G5's own schema-cause meanings (18-24) stays G5's --
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
    /* G5's own schema block (design Sec5) -- ordinals 18-24 ALLOCATED by Sec6, MEANING
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
    /* resource-exhaustion rejection (design Sec6; FOLD RULING
     * on the third confirmation pass's F2, design commit dated 2026-08-17) -- process-level
     * allocation failure, distinct from SSLM_KV_POOL_EXHAUSTED (caller-supplied-memory
     * exhaustion). The boundary catch (src/sslm_abi.cpp, CatchAllocationFailure and both
     * SslmModel::Load call sites): catch(const std::bad_alloc&)/catch(const std::length_error&)
     * -- the only causes this family covers -- return this status; a final catch(...) is the
     * true, unconditional boundary and returns SSLM_ARTIFACT_REJECTED instead (no new ordinal
     * minted for it -- MapForwardStatus's own existing "no dedicated status" mapping, extended
     * here on the record). */ \
    X(SSLM_ALLOCATION_FAILED) /* 25 */ \
    /* T-2199 Phase D review fix S6 (Claude/Poirot/7a3b10a-t2199-phaseD-review.md): a per-step
     * NUMERIC gate declining on a VALID model and VALID params (TopKRenormalizeQ15's own
     * refusal surface, plan Sec7.5) is not an artifact defect -- MapForwardStatus's blanket
     * non-Ok -> SSLM_ARTIFACT_REJECTED collapse (src/sslm_abi.cpp) told a caller with a
     * perfectly good model and params to discard it, which is the wrong remedy for a
     * retry-safe, per-step numeric condition. Appended at the END of this list (never inserted
     * into the "numeric/domain rejections" block above, ordinals 15-17) so no already-shipped
     * ordinal renumbers -- this ABI's own additive-only discipline for a public C surface. */ \
    X(SSLM_NUMERIC_STEP_REFUSED) /* 26 */

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

/* design Sec8: carried unchanged from the G5 constrained-decoding design's own Sec5 --
 * transcribed here verbatim, since this design does not restate G5's own struct
 * bodies. */
typedef enum sslm_span_kind {
    SSLM_SPAN_PROMPT = 0,
    SSLM_SPAN_SCHEMA_CONTENT = 1
} sslm_span_kind;

/* T-2199 Phase D1 (plan Sec8 D1 RULING: an ADDITIVE FIELD, not a versioned successor struct --
 * grounded against this same struct family's own sslm_stats_out/schema_accepting precedent,
 * D-SLM3476, T-2132). T-2199 Phase D review fix M2 (Claude/Poirot/7a3b10a-t2199-phaseD-review.md):
 * this comment previously cited "D-SLM3794/D-SLM3794A-precedent" and "D-SLM3476A" -- neither
 * "D-SLM3794A" nor "D-SLM3476A" exists in Claude/Decisions/DecisionLog.md (only D-SLM3794 and
 * D-SLM3476 do), and separately, D-SLM3794's own text rules the ARTIFACT flags-bit mechanism
 * (Decision A, artifact.h's kDampedGreedyArtifactConstantsFlag) and says nothing about this C
 * struct -- the additive-field ABI decision this comment describes is the plan's own (Sec8 D1),
 * not D-SLM3794's, so that citation is dropped here rather than corrected to a decision entry
 * that does not actually make this ruling. `mode` selects the decode-step's own
 * selection mechanism: 0 (SSLM_DECODE_MODE_GREEDY, the default under zero-init, so every
 * EXISTING caller of sslm_decode_step -- which never sets these new fields -- is bit-unchanged)
 * or 1 (SSLM_DECODE_MODE_DAMPED_GREEDY). alpha_q15/anti_lm_max_order/top_k/q_ln2/q_b/q_c are
 * meaningful ONLY under damped-greedy mode (plan Sec7.1's own "Inputs" list; ignored, never
 * read, in greedy mode) -- q_ln2/q_b/q_c are the runtime i-exp scale constants a real caller
 * derives once per model load (plan Sec8 B3) and passes per call here rather than this ABI
 * caching them on the model handle, matching Phase D1's own build scope (Phase B3's own
 * model-handle-caching mechanism is not required by this phase; a caller may derive and cache
 * these itself). */
#define SSLM_DECODE_MODE_GREEDY 0
#define SSLM_DECODE_MODE_DAMPED_GREEDY 1

/* T-2199 Phase D review addendum (D-SLM3797, Dan, 2026-08-20): sslm_decode_params' FIRST new
 * field is `struct_size` -- caller-SET, library-VALIDATED. Supersedes the no-size-field reading
 * of the additive-field ruling above for THIS struct specifically (D-SLM3797's own text): this
 * struct is an IN parameter crossing the ABI trust boundary, and a silent partial/garbage read
 * on header/library version skew contradicts this codebase's own boundary discipline (the
 * loader's hostile-input model, applied here to a decode-time struct instead of an artifact).
 * The field is free to add only while 1.2's own shape is unshipped -- sslm_stats_out and every
 * other OUT struct in this ABI keep the pre-existing version-implied convention unchanged; this
 * is a scoped exception for this ONE struct, not a new blanket rule. A caller sets
 * `struct_size = sizeof(sslm_decode_params)`; sslm_decode_stepImpl rejects any other value with
 * SSLM_INVALID_ARGUMENT (checked before layer_budget/mode/anything else -- see that function's
 * own comment, sslm_abi.cpp) -- a wrong size is a defined, loud rejection, never a partial read
 * of a struct shape the library and the caller disagree about. */
typedef struct sslm_decode_params {
    int32_t layer_budget;
    uint32_t struct_size;        /* caller sets sizeof(sslm_decode_params); library validates */
    int32_t mode;               /* SSLM_DECODE_MODE_GREEDY (0, default) or _DAMPED_GREEDY (1) */
    /* T-2199 Phase D review fix, C2 (Claude/Poirot/7a3b10a-t2199-phaseD-review.md): int32_t per
     * plan Sec2.5's own stated field type -- the prior build shipped this as int64_t, which
     * removed the width-bound half of the overflow argument Sec2.5/Sec9 dim2 rest on
     * ("an int32_t field's own representable range already bounds alpha_q15 far below any
     * int64 overflow risk"). Restored, plus the domain check below (ValidateDampedGreedyParams,
     * damped_greedy_phaseD.cpp) enforcing the plan's own two-sided [0, 2^20) sanity ceiling --
     * both defenses, not one, matching the plan's own explicit "two independent bounds" framing. */
    int32_t alpha_q15;          /* damped-greedy only: Q15-scaled anti-repetition weight, [0, 2^20) */
    int32_t anti_lm_max_order;  /* damped-greedy only: the anti-LM's own n, >= 1 */
    int32_t top_k;              /* damped-greedy only: candidates scored per step, 1 <= k <= vocab_size */
    int64_t q_ln2;              /* damped-greedy only: runtime i-exp scale constant (plan Sec2.3/B3) */
    int64_t q_b;                /* damped-greedy only: runtime i-exp scale constant */
    int64_t q_c;                /* damped-greedy only: runtime i-exp scale constant */
} sslm_decode_params;

typedef struct sslm_stats_out {
    int64_t decode_step_ceiling;
    int64_t decode_step_actual;
    int64_t forced_token_count;
    int32_t kv_blocks_resident;
    /* (design Sec14.1): 1 iff the
     * sequence's current dfa_walk_state is a member of its bound schema's accept set
     * (accepting_le/accepting_count, design Sec13.2); 0 when it is not, and 0 when no schema is
     * bound (SSLM_SCHEMA_NONE) -- a host never needs to special-case whether a schema is even
     * bound before reading it. The query dim10's own placeholder comment named ("pending the
     * build seat's own accepting-state query"), added to this existing per-sequence stats
     * accessor rather than as a new verb, per the ruling's own "smallest sound thing" reasoning
     * (the same one already applied to forced_token_count). */
    int32_t schema_accepting;
} sslm_stats_out;

/* The alignment sslm_workspace_create and
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
 * SUPERSLM_ABI_ENUM_ONLY: #define this BEFORE including this header to suppress the function-declaration
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
 * surface) is completely unaffected.
 *
 * This section carries its OWN guard (SUPERSLM_ABI_FUNCTIONS_INCLUDED_), independent of and
 * outside SUPERSLM_INCLUDE_SSLM_ABI_H above: the enum-only pass sets the outer guard but must NOT be able to claim it also
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

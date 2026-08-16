/* sslm_g5.h -- T-2130 (Curie) red suite: the canonical declared surface for the G5
 * constrained-decoding engine.
 *
 * Transcribed verbatim from the design of record, Claude/Vitruvius/
 * t2119-g5-constrained-decoding-design-2026-08-16.md Sec5 (Wizard repo) -- the ABI surface
 * post rung-5 repair (T-2122/D-SLM3445) and rung-6 confirmation fold. Every G5-specific
 * declaration cites the design section it was copied from. This is the SAME promotion
 * discipline T-2112's sslm_gpu_1p0.h established (its own header comment, this repo,
 * tests/t2112-gpu-1p0-red-suite/sslm_gpu_1p0.h): one canonical copy in this directory,
 * included by every .cpp/.py fixture in this suite via `-I .`, never hand-patched out of
 * sync with the design -- regenerate this file from the design of record when Sec5 changes.
 *
 * SCOPE NOTE (routed as a finding, not silently resolved -- same shape as T-2112's own
 * SlmModelView-collision finding in its header comment): the design's Sec5 ABI is written
 * in the PLAN's own Sec12 API-sketch naming (`sslm_model`, `sslm_seq`, `sslm_prefix`,
 * `sslm_schema`, `sslm_prefill`, `sslm_seq_create`, ...) -- distinct from the ALREADY-SHIPPED
 * `SslmGpu*`-prefixed handle surface T-2107/T-2112/T-2113 built (include/superslm/gpu_1p0.h).
 * A grep of this repo's own src/ and include/ at this suite's authoring commit (main@f37be4a)
 * finds NO symbol under the `sslm_model_map`/`sslm_seq_create`/`sslm_prefill`/
 * `sslm_prefix_begin` names the plan's own Sec12 sketch and this design's Sec5 both use --
 * the Layer-1 C ABI the plan sketches has never been built under those literal names; every
 * shipped GPU-serial-port and S-LoRA-serial mechanism is reached through the `SslmGpu*`
 * handle surface instead. This suite is therefore written against the design's OWN Sec5 text,
 * not reconciled against the shipped `SslmGpu*` naming -- reconciling the two naming schemes
 * (or confirming the build seat intends a literal `sslm_`-prefixed Layer-1 ABI distinct from
 * `SslmGpu*`) is a build-time question for whoever prosecutes G5, not a test-authoring
 * decision Curie's own jurisdiction covers (Curie realizes the design's declared ABI in
 * tests; she does not resolve a naming question the design itself did not raise). Filed in
 * Claude/Curie/t2130-g5-red-suite-2026-08-16.md (Wizard repo).
 *
 * Every function below is therefore RED BY LINK against the real engine: this header
 * declares the surface, no .cpp anywhere in this repo (grep confirmed at authoring time)
 * defines it, so every cell that calls one of these symbols compiles clean and fails to
 * link (LNK2019), per this suite's own build_link_red.bat and the house convention
 * tests/t2112-gpu-1p0-red-suite/build_link_red.bat established.
 *
 * The pre-G5 Layer-1 primitives G5 extends (sslm_seq_create/release/reset/save/restore,
 * sslm_prefix_begin/freeze/release, sslm_seq_adopt_prefix, sslm_seq_set_adapter,
 * sslm_decode_step, sslm_stats) are declared here too, at the SAME plan Sec12 signatures,
 * because they are equally undeclared under these literal names in the current tree -- this
 * mirrors T-2112's own precedent of declaring more of the surface than just the new pieces,
 * for the identical reason (nothing exists yet under either name).
 */
#ifndef SSLM_G5_H
#define SSLM_G5_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* --- opaque handle types (plan Sec12; design Sec5 uses the identical shapes) --- */
typedef struct sslm_model_s*    sslm_model;
typedef struct sslm_seq_s*      sslm_seq;
typedef struct sslm_prefix_s*   sslm_prefix;
typedef struct sslm_adapter_s*  sslm_adapter;
typedef struct sslm_schema_s*   sslm_schema;      /* NEW, design Sec5 */
typedef struct sslm_kv_pool_s*  sslm_kv_pool;
typedef struct sslm_workspace_s* sslm_workspace;

/* --- status enum. Pre-G5 members are the plan's own Sec9/Sec12 taxonomy (named, not
 * enumerated exhaustively there); G5's own additions are transcribed verbatim from design
 * Sec5/Sec7 dim5. --- */
typedef enum sslm_status {
    SSLM_OK = 0,
    SSLM_INVALID_ARGUMENT,
    SSLM_MODEL_MAP_REJECTED,
    SSLM_ADAPTER_MODEL_MISMATCH,             /* plan Sec12 */
    SSLM_ADAPTER_SWAP_MIDTOKEN_REJECTED,      /* plan Sec12, sslm_seq_set_adapter */
    SSLM_RESTORE_MODEL_MISMATCH,              /* plan Sec12 (cited by design Sec5 as the
                                                * existing precedent SSLM_RESTORE_SCHEMA_MISMATCH
                                                * is symmetric with) */

    /* --- G5, design Sec5 / Sec7 dim5 (verbatim) --- */
    SSLM_SCHEMA_NOT_FOUND,                    /* sslm_schema_lookup: unknown name, design Sec5 */
    SSLM_SCHEMA_BIND_REJECTED,                /* sslm_seq_set_schema on a non-fresh walk state,
                                                * design Sec5 */
    SSLM_SCHEMA_SPAN_UNBOUND,                 /* sslm_prefill(..., SSLM_SPAN_SCHEMA_CONTENT, ...)
                                                * against an unbound sequence, design Sec5/Sec10.2 */
    SSLM_PREFIX_SCHEMA_MISMATCH,              /* sslm_seq_adopt_prefix: real schema-content walk
                                                * progress against a mismatched or unbound sequence,
                                                * design Sec5 (stated exhaustively, both branches) */
    SSLM_RESTORE_SCHEMA_MISMATCH,             /* sslm_seq_restore: bound schema does not resolve
                                                * against the currently-mapped artifact's schema set,
                                                * design Sec5 / Sec7 dim2/dim9 */
    SSLM_SCHEMA_SPAN_UNREACHABLE,             /* G5-4, design Sec6: a host-declared "fixed" span
                                                * that is not actually reachable under the active
                                                * schema's DFA from the sequence's current walk-
                                                * state -- "a span that leaves the DFA's language
                                                * before it is exhausted." Named exactly once,
                                                * uncapitalized in the design's own prose ("fixed
                                                * span not reachable"); this enumerator is this
                                                * suite's own transcription, since the design gives
                                                * no literal SSLM_-prefixed name for it -- a naming
                                                * gap distinct from sslm_prefix_prefill's own call
                                                * shape (see that declaration's comment, below),
                                                * which the planner has since ruled on
                                                * (Claude/Vitruvius/t2119-rung7-fold-2026-08-16.md,
                                                * Wizard repo, ruling 1). */
    SSLM_SCHEMA_UNSATISFIABLE                 /* G5-1 compiler rejection, design Sec3/Sec6 G-7a --
                                                * CPU/converter-side, listed here for completeness;
                                                * the C++ decode-time surface never emits it */
} sslm_status;

/* --- G5's own span-kind discriminator -- design Sec5, THE REPAIR (Sec10.2). The single
 * structural move of the rung-5 fold: `sslm_prefill` gains this parameter because no function
 * of the sequence's pre-existing observable state can discriminate the three span kinds it
 * must route (design Sec10.1, the strike's own pigeonhole proof, survivors = 0 without it). ---
 */
typedef enum sslm_span_kind {
    SSLM_SPAN_PROMPT = 0,          /* host context/utterance tokens. Walk-state NEVER advances,
                                     * whatever schema is bound (design Sec5). */
    SSLM_SPAN_SCHEMA_CONTENT = 1   /* tokens that ARE the schema-constrained output stream: a
                                     * jump-forward forced chain (runtime-supplied) or a
                                     * template-fill fixed span (host-supplied). Walk-state
                                     * ALWAYS advances, token by token (design Sec5). Rejected
                                     * (SSLM_SCHEMA_SPAN_UNBOUND) against SSLM_SCHEMA_NONE. */
} sslm_span_kind;

/* SSLM_SCHEMA_NONE -- the unconstrained sentinel schema handle, design Sec5
 * ("SSLM_SCHEMA_NONE = unconstrained, the pre-G5 behavior"). A null handle by construction,
 * matching the plan's own NULL-adapter convention for sslm_seq_set_adapter. */
#define SSLM_SCHEMA_NONE ((sslm_schema)0)

/* --- decode-time configuration, plan Sec12 (schema field REMOVED this design, Sec5: "is
 * superseded by this per-sequence binding" -- what remains is the field the plan's own Sec12
 * comment names alongside it). --- */
typedef struct sslm_decode_params {
    int32_t layer_budget;    /* plan Sec12 (S1) */
} sslm_decode_params;

/* --- sslm_stats_out, plan Sec12 ("ceiling + actual work, forced-token count, cache state").
 * forced_token_count is the field G5-3's own red suite pins (design Sec6 G5-3 gate / Sec7
 * dim7: "reports the actual forced-position count, not the ceiling"). --- */
typedef struct sslm_stats_out {
    int64_t decode_step_ceiling;      /* plan Sec14 cost-contract: worst case is full decode */
    int64_t decode_step_actual;
    int64_t forced_token_count;       /* G5-3, design Sec6/Sec7 dim7 */
    int32_t kv_blocks_resident;
} sslm_stats_out;

/* ============================================================================
 * Pre-G5 Layer-1 primitives (plan Sec12), undeclared under these literal names anywhere in
 * this repo at authoring time -- RED BY LINK identically to the G5-specific surface below.
 * ============================================================================ */

sslm_status sslm_model_map(const void* data, size_t size, sslm_model* out);
sslm_status sslm_model_unmap(sslm_model model);

sslm_status sslm_prefix_begin(sslm_model model, sslm_kv_pool* pool, sslm_prefix* out);
sslm_status sslm_prefix_freeze(sslm_prefix prefix);
sslm_status sslm_prefix_release(sslm_prefix prefix);

sslm_status sslm_seq_create(sslm_model model, sslm_kv_pool* pool, sslm_seq* out);
sslm_status sslm_seq_release(sslm_seq seq);
sslm_status sslm_seq_reset(sslm_seq seq);
sslm_status sslm_seq_adopt_prefix(sslm_seq seq, sslm_prefix prefix);
sslm_status sslm_seq_save(sslm_seq seq, void* buf, size_t* n);
sslm_status sslm_seq_restore(sslm_model model, sslm_kv_pool* pool, const void* buf, size_t n,
                              sslm_seq* out);

sslm_status sslm_seq_set_adapter(sslm_seq seq, sslm_adapter adapter);

sslm_status sslm_decode_step(sslm_model model, sslm_seq* seqs, int32_t n,
                              const sslm_decode_params* params, sslm_workspace ws,
                              int32_t* out_tokens);

sslm_status sslm_stats(sslm_model model, sslm_seq seq, sslm_stats_out* out);

size_t sslm_seq_state_size(sslm_model model);

/* ============================================================================
 * G5: the constrained-decoding surface (design Sec5, verbatim).
 * ============================================================================ */

/* Sec5: schema lookup, read-only, no map/release verb -- the schema is already resident
 * wherever the model is. Unknown name is SSLM_SCHEMA_NOT_FOUND. Lookup-by-name, D-SLM3442. */
sslm_status sslm_schema_lookup(sslm_model model, const char* name, sslm_schema* out);
size_t      sslm_schema_count(sslm_model model);
sslm_status sslm_schema_name(sslm_model model, int32_t index, char* buf, size_t* n);

/* Sec5: binds a compiled schema to a sequence. SSLM_SCHEMA_NONE = unconstrained. Valid ONLY
 * when the sequence's DFA-walk state is at its start (fresh sslm_seq_create, or immediately
 * after sslm_seq_reset) -- a non-fresh walk state rejects (SSLM_SCHEMA_BIND_REJECTED). No
 * mid-generation re-binding (design Sec5/Sec11: explicitly out of 1.0 scope). */
sslm_status sslm_seq_set_schema(sslm_seq seq, sslm_schema schema);

/* Sec5/Sec10.2, REPAIRED (T-2122/D-SLM3445): `kind` is the discriminator that makes the
 * walk-advance decision fully determined (design Sec10.3: repair_probe.py, survivors 0 -> 1,
 * both traces). Caller-declared CONFIGURATION, decided by which code path issues the call
 * (never derived from token content -- design Sec5's own cost-determinism argument, plan
 * Sec2/Sec14). SSLM_SPAN_PROMPT: every pre-G5 call site migrates here mechanically (design
 * Sec5, "Pre-freeze migration note"), behaviorally inert on an SSLM_SCHEMA_NONE sequence. */
sslm_status sslm_prefill(sslm_model model, sslm_seq seq, const int32_t* tokens, int32_t count,
                          int32_t chunk_budget, sslm_span_kind kind, sslm_workspace ws,
                          int32_t* consumed);

/* CURIE COVERAGE-MODEL-ADJACENT GAP #2 -- BLESSED, with a contract
 * (Claude/Vitruvius/t2119-rung7-fold-2026-08-16.md, Wizard repo, ruling 2): design Sec7
 * dim11 requires a guard-vitality cell proving the runtime's D-SLM40 non-check for an
 * all-masked logit vector has NO SUBJECT to catch -- but the design's OWN Sec5 ABI gives no
 * caller any way to construct an all-masked-vector state to observe this against (the
 * compiler's G-7a proof forecloses every legitimate path to one). This is a real
 * mask-application primitive with no ABI-level entry point named anywhere in Sec4/Sec5 --
 * internal to G5-2 by design. This declaration is this suite's own test-only hook, NOT part
 * of the design's own public ABI -- the planner's ruling confirms this disposition (not the
 * bench-bridge precedent's "promote to production" fix: no real host has any legitimate
 * reason to call raw mask+argmax directly, so promoting this would manufacture product
 * surface nothing needs). THE HOOK'S CONTRACT, per the ruling: whoever prosecutes G5-2
 * implements this as a THIN WRAPPER calling G5-2's own internal (non-exported) mask-
 * application primitive -- never a parallel reimplementation, or a passing negative control
 * proves nothing about the production guard (the project's own "producible by the real data
 * path" commissioning discipline). Its non-shipping disposition is enforced structurally:
 * declared ONLY in this suite-private header (never `include/superslm/`) and excluded from
 * this repo's CMake install/export rule set from the start -- verified at this suite's own
 * authoring commit: `sslm_g5.h` has no counterpart under `include/` and no CMake target in
 * this repo installs or exports anything from `tests/t2130-g5-red-suite/`. */
sslm_status sslm_g5_test_only_apply_mask_and_argmax(const int32_t* logits, const uint8_t* mask,
                                                      int32_t vocab_size, int32_t* out_token_id);

/* Sec5: "Prefix construction inherits the same discriminator." Valid only before any content
 * has been prefilled into the prefix under construction. */
sslm_status sslm_prefix_set_schema(sslm_prefix prefix, sslm_schema schema);

/* CURIE COVERAGE-MODEL-ADJACENT GAP -- RULED (Claude/Vitruvius/t2119-rung7-fold-2026-08-16.md,
 * Wizard repo, ruling 1; superseding the routed-not-invented note this comment previously
 * carried, see Claude/Curie/t2130-g5-red-suite-2026-08-16.md Sec5 for the original finding):
 * design Sec5 states prefix construction is "mechanically, prefilling into a scratch handle
 * before sslm_prefix_freeze converts it to a shared, read-only block" and that content
 * prefilled into a prefix under construction "carries the same sslm_span_kind parameter
 * sslm_prefill does." This suite's original reading took "the same ... parameter" to mean
 * `kind` alone (a four-parameter overload); the planner's ruling corrects this: "the same
 * parameter" means the same PARAMETER SET, not one field of it -- full mechanical symmetry
 * with sslm_prefill, `sslm_seq` swapped for `sslm_prefix`. `chunk_budget`/`consumed` are
 * required by the frame-budget law (design Sec2/Sec8.1, plan G1) that applies to every
 * token-admitting call with no prefix exemption; `sslm_workspace` and the explicit
 * `sslm_model` argument match every other such call's existing shape. This is now the RULED
 * shape, not an unconfirmed reading -- no further build-time confirmation is owed on the
 * signature itself. */
sslm_status sslm_prefix_prefill(sslm_model model, sslm_prefix prefix, const int32_t* tokens,
                                 int32_t count, int32_t chunk_budget, sslm_span_kind kind,
                                 sslm_workspace ws, int32_t* consumed);

#ifdef __cplusplus
}
#endif

#endif /* SSLM_G5_H */

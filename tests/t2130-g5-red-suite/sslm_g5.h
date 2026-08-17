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

/* --- status enum, MIRRORED verbatim from Sec6's complete ordinal registry (design commit
 * 4f4eb23896, Claude/Vitruvius/t2133-layer1-c-abi-design-2026-08-16.md Sec6 GOVERNANCE RULING:
 * "Sec6 is the COMPLETE ordinal registry for sslm_status; sslm_g5.h mirrors it verbatim rather
 * than owning a block", Poirot confirmation-pass casebook 2c18dab-t2139-abi-build-review.md
 * Sec7.4/P2, folded). This supersedes the prior "base low 0-17, G5 appended above at 18-24"
 * reconciliation (still visible in the per-enumerator comments below, which record each entry's
 * own history) -- SSLM_ALLOCATION_FAILED's ratification at ordinal 25 (past the 0-24 range, the
 * append-only law's own correct reasoning at the time) left the base family non-contiguous
 * (0-17 and 25) with G5's block (18-24) sitting INSIDE that range rather than cleanly above it,
 * and nothing governed who takes the next ordinal (26) from two independently-numbered blocks.
 * Sec6 closes this by becoming the single ordinal-allocation authority for BOTH families: base
 * enumerators and G5's schema enumerators alike are entries in one registry, assigned by Sec6
 * alone, never self-numbered by either header independently. Semantic ownership is unchanged --
 * G5 (design T-2119) still owns the MEANING of its own schema statuses; only ordinal allocation
 * centralizes. All 26 registry entries are carried here, verbatim, including
 * SSLM_ALLOCATION_FAILED at 25 (a prior gap in this header, closed this fold). Next-free ordinal
 * per the registry: 26 -- requested from and assigned by Sec6, whichever side needs it next.
 * Gate C's sslm_status exclusion lifted at the prior reconciliation and stays lifted -- the
 * per-enumerator check runs unconditionally over the full, reconciled enum. Gate C's own retired
 * "highest base enumerator strictly below first G5-only enumerator" ordering sentinel (false on
 * its own terms once SSLM_ALLOCATION_FAILED landed at 25) was NOT left retired -- FOLD RULING
 * 2026-08-17 (design Sec6, on Poirot's third confirmation casebook 4466666-t2139-third-
 * confirmation-review.md F1) reinstated a structural version of it: both this header and
 * include/superslm/sslm_abi.h gain a final enumerator, SSLM_STATUS_NEXT_FREE, with no explicit
 * value, so it auto-values one past whichever header's own last explicit member and moves itself
 * whenever either side appends -- see the enum body below, and Gate C's own generated per-name
 * checks plus the SSLM_STATUS_NEXT_FREE-pair static_assert this header's addition now makes
 * possible on the library side. */
typedef enum sslm_status {
    SSLM_OK = 0,                              /* the only non-error value */

    /* --- Argument/precondition rejections (T-2133 Sec6) --- */
    SSLM_INVALID_ARGUMENT = 1,                /* the one broad catch-all this taxonomy keeps --
                                                * only for malformed calls with no differentiated
                                                * remedy (a null required pointer, a negative
                                                * count, an out-of-range enum) */
    SSLM_BUFFER_TOO_SMALL = 2,                /* a caller-supplied buffer (workspace, KV pool,
                                                * save blob, sslm_seq_state_size-sized state) is
                                                * smaller than the artifact-derived requirement --
                                                * remedy: allocate the size the sizing function
                                                * reports */
    SSLM_MISALIGNED_BUFFER = 3,               /* a caller-supplied buffer's address does not meet
                                                * the artifact's declared alignment */

    /* --- Artifact/content rejections (T-2133 Sec6, Sec17 dim2 trust-boundary) --- */
    SSLM_ARTIFACT_REJECTED = 4,               /* RENAMED from SSLM_MODEL_MAP_REJECTED this fold --
                                                * the identical rejection T-2133 Sec6 names under
                                                * this identifier: the underlying SslmModel::Load
                                                * rejected the container or a section */
    SSLM_ADAPTER_MODEL_MISMATCH = 5,          /* RENUMBERED 3->5 this fold. plan Sec12's existing
                                                * name, kept -- op-set/dims/base-hash mismatch */
    SSLM_RESTORE_MODEL_MISMATCH = 6,          /* RENUMBERED 5->6 this fold. plan Sec12 (cited by
                                                * design Sec5 as the existing precedent
                                                * SSLM_RESTORE_SCHEMA_MISMATCH is symmetric with) */
    SSLM_RESTORE_KV_MISMATCH = 7,             /* a save-blob's kv_precision does not match the
                                                * target model/pool, Sec17 dim9 N2 */

    /* --- Lifecycle/precondition-on-state rejections (T-2133 Sec6) --- */
    SSLM_MODEL_HAS_LIVE_SEQUENCES = 8,        /* sslm_model_unmap while a live sslm_seq/
                                                * sslm_prefix/sslm_adapter still references it */
    SSLM_POOL_HAS_LIVE_HANDLES = 9,           /* sslm_kv_pool_destroy while a live sequence or
                                                * prefix still holds blocks from it */
    SSLM_ADAPTER_HAS_LIVE_SEQUENCES = 10,     /* sslm_adapter_release while a live sslm_seq still
                                                * points at it */
    SSLM_ADAPTER_SWAP_MIDTOKEN_REJECTED = 11, /* RENUMBERED 4->11 this fold. plan Sec12's existing
                                                * name, kept -- sslm_seq_set_adapter against a
                                                * non-zero mid-token residual marker */
    SSLM_SEQ_RESET_MIDTOKEN_REJECTED = 12,    /* plan Sec17's lifecycle addendum:
                                                * sslm_seq_reset against a non-zero residual
                                                * marker */
    SSLM_PREFIX_FROZEN_REJECTED = 13,         /* sslm_prefix_prefill called on a prefix past
                                                * sslm_prefix_freeze, or already released */
    SSLM_KV_POOL_EXHAUSTED = 14,              /* sslm_seq_create/sslm_prefix_begin: no free block
                                                * in the pool -- Sec17 dim5's "defined, resumable
                                                * failure" */
    SSLM_TOKEN_ID_OUT_OF_RANGE = 15,          /* a token id outside [0, vocab_size) reaching
                                                * sslm_prefill/sslm_decode_step's input, mirroring
                                                * the GPU ABI's own B3.5 addition, D-SLM3367 */
    SSLM_CONTEXT_CAP_EXCEEDED = 16,           /* a prefill/decode_step call that would push
                                                * context_length past the artifact's context_cap */
    SSLM_TOKEN_ID_UNMAPPED = 17,              /* NEW, appended this fold (design commit
                                                * 212de7742c, the padded-vocabulary ruling, Brunel
                                                * T-2139, executed SuperSLM@f9a0065): the base
                                                * taxonomy's numeric/domain group gains this
                                                * enumerator at ordinal 17, the next free base
                                                * ordinal, per the append-only reconciliation law --
                                                * never inserted within the group's own prior
                                                * position, even though it is semantically a
                                                * numeric/domain rejection, because ordinal
                                                * stability for every already-reconciled enumerator
                                                * wins. sslm_detokenize_stream's own rejection for a
                                                * decode-output token id in [tok_vocab, cfg_vocab) --
                                                * a legal decode output with no tokenizer entry
                                                * (the padded-vocabulary case
                                                * ValidateTokenizerVocabSizeJoin's loosening admits,
                                                * src/model.cpp), distinct from
                                                * SSLM_TOKEN_ID_OUT_OF_RANGE (id >= cfg_vocab, never
                                                * a legal decode output at all). This closed the
                                                * base family's own contiguous 0-17 run of 18
                                                * enumerators AT THE TIME; the registry has since
                                                * grown past that shape entirely -- Sec6's
                                                * GOVERNANCE RULING (below) makes this a single
                                                * 26-entry registry, one interleaved ordinal
                                                * sequence spanning base (0-17, then 25) and G5
                                                * (18-24) together, not two independently-counted
                                                * blocks. G5's own seven sit at 18-24, immediately
                                                * below. */

    /* --- G5, design Sec5 / Sec7 dim5 -- APPENDED at 18-24, existing relative order preserved
     * per the ruling's own append-only reasoning (G5's own internal ordering is T-2119's call,
     * not T-2133's). SHIFTED 17-23 -> 18-24 this fold to make room for
     * SSLM_TOKEN_ID_UNMAPPED's own base-taxonomy insertion at 17, above. --- */
    SSLM_SCHEMA_NOT_FOUND = 18,                /* sslm_schema_lookup: unknown name, design Sec5 */
    SSLM_SCHEMA_BIND_REJECTED = 19,            /* sslm_seq_set_schema on a non-fresh walk state,
                                                * design Sec5 */
    SSLM_SCHEMA_SPAN_UNBOUND = 20,             /* sslm_prefill(..., SSLM_SPAN_SCHEMA_CONTENT, ...)
                                                * against an unbound sequence, design Sec5/Sec10.2 */
    SSLM_PREFIX_SCHEMA_MISMATCH = 21,          /* sslm_seq_adopt_prefix: real schema-content walk
                                                * progress against a mismatched or unbound sequence,
                                                * design Sec5 (stated exhaustively, both branches) */
    SSLM_RESTORE_SCHEMA_MISMATCH = 22,         /* sslm_seq_restore: bound schema does not resolve
                                                * against the currently-mapped artifact's schema set,
                                                * design Sec5 / Sec7 dim2/dim9 */
    SSLM_SCHEMA_SPAN_UNREACHABLE = 23,         /* G5-4, design Sec6: a host-declared "fixed" span
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
    SSLM_SCHEMA_UNSATISFIABLE = 24,             /* G5-1 compiler rejection, design Sec3/Sec6 G-7a --
                                                * CPU/converter-side, listed here for completeness;
                                                * the C++ decode-time surface never emits it */

    /* --- Resource-exhaustion rejection (T-2133 Sec6, RATIFIED Brunel T-2139 Sec19/N3; registry
     * entry per Sec6's GOVERNANCE RULING, design commit 4f4eb23896) -- process-level allocation
     * failure, distinct from SSLM_KV_POOL_EXHAUSTED (caller-supplied-memory exhaustion, an
     * entirely different cause). Eight verbs plus two shared internal helpers (the swept set,
     * Claude/Poirot/4466666-t2139-third-confirmation-review.md Sec2/O2 -- a superset of the seven
     * originally named, corrected to the actual swept scope) carry a small number of remaining
     * internal std::vector/SslmModel::Load allocations this ABI layer cannot eliminate. FOLD
     * RULING 2026-08-17 (design Sec6, on the same casebook's F2) narrowed the boundary catch to
     * exactly two exception types for this status -- catch (const std::bad_alloc&) and
     * catch (const std::length_error&), the only causes this family is defined to cover -- with a
     * final catch (...) closing the extern "C" boundary unconditionally and returning
     * SSLM_ARTIFACT_REJECTED instead (no new ordinal minted; MapForwardStatus's own existing
     * "no dedicated status" mapping, extended). Ordinal 25 -- appended past the entire
     * already-coordinated 0-24 range at ratification time (the append-only law's own correct
     * reasoning then); this header previously did not carry it at all (0 occurrences, closed that
     * fold). */
    SSLM_ALLOCATION_FAILED = 25,                /* see catch-split above; the resource-exhaustion
                                                * cause alone, never a wrapper for a cause that
                                                * already has its own status */

    /* Sentinel, FOLD RULING 2026-08-17 (design Sec6, on F1) -- the enum's own final member, no
     * explicit value, so it auto-values to one past whichever entry above it is this header's own
     * last explicit member, and moves itself automatically on either header's own next append.
     * Paired with include/superslm/sslm_abi.h's identical last member via Gate C's own
     * static_assert(SSLM_STATUS_NEXT_FREE == SSLM_STATUS_NEXT_FREE) sentinel check -- catches a
     * shared ordinal claimed by two different names (F1's own Probe C), which the per-name checks
     * structurally cannot see. Never referenced as a valid status value by any call; it exists
     * only for this comparison. */
    SSLM_STATUS_NEXT_FREE
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

/* C1 KV-pool sizing/construction primitives (plan Sec12, RULED design commit fab235c1c6) --
 * mirrored verbatim from include/superslm/sslm_abi_functions_g5_comparable.inc, ADDED T-2132
 * (Curie) so this suite's own sslm_seq_create/sslm_prefix_begin/sslm_seq_restore calls below
 * can construct a REAL sslm_kv_pool per the shipped ABI's own precondition
 * (`if (!model || !pool || !*pool) return SSLM_INVALID_ARGUMENT;`, src/sslm_abi.cpp) instead
 * of passing nullptr -- this suite's ORIGINAL construction, which failed 28/28 checks at the
 * first sslm_seq_create call against the real fixture (Claude/Brunel/
 * t2132-g5-build-2026-08-16.md's pool=nullptr finding). sslm_kv_block_size: one WHOLE
 * sequence's entire KV footprint across every layer, never a sub-sequence PagedAttention page
 * (design commit fab235c1c6). sslm_kv_pool_create's block_count: how many concurrent
 * SEQUENCES the pool can back, one whole block each -- no token-count arithmetic. */
size_t sslm_kv_block_size(sslm_model model);
size_t sslm_kv_pool_overhead_size(sslm_model model, uint32_t block_count);
sslm_status sslm_kv_pool_create(sslm_model model, void* buf, size_t buf_size,
                                 uint32_t block_count, sslm_kv_pool* out);
sslm_status sslm_kv_pool_destroy(sslm_kv_pool pool);

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

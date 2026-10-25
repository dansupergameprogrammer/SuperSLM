/* sslm_abi_functions.inc -- the sole source of every sslm_* function DECLARATION this ABI
 * ships (design Sec8). include/superslm/sslm_abi.h #includes this file, inside its own
 * extern "C" block, after defining every type the declarations reference (sslm_model,
 * sslm_status, sslm_config, sslm_detok_state, etc.).
 *
 * Split into two pieces: sslm_abi_functions_g5_comparable.inc holds every verb whose signature
 * uses ONLY types tests/t2130-g5-red-suite/sslm_g5.h itself defines -- included here AND,
 * separately, from tools/t2139_gate_a_header_parity_check.cpp (macro-renamed) so Gate A checks
 * the declarations this library actually ships, not a hand-maintained copy of them. The two
 * workspace-sizing declarations below take the caller-memory call-shape struct, which has no
 * counterpart in sslm_g5.h at all (design Sec9 residual d) -- they stay directly in this file,
 * outside the g5-comparable subset, and are correspondingly outside Gate A's own comparison
 * surface until/unless that struct gains a second, independent definition to compare against.
 * The text-I/O declaration below takes the incremental-detokenize state struct, the same
 * residual's other type, for the identical reason.
 */
#include "superslm/api.h"
#include "superslm/sslm_abi_functions_g5_comparable.inc"

/* Versioned decode entry point. The legacy sslm_decode_step declaration lives in the
 * g5-comparable inventory because it is the v1.1 surface mirrored by that suite. This v2 symbol
 * is deliberately separate: it may validate and consume sslm_decode_params fields beyond the
 * shipped four-byte layer_budget prefix without reading past an old caller's allocation. */
SUPERSLM_API sslm_status sslm_decode_step_v2(sslm_model model, sslm_seq* seqs, int32_t n,
                                const sslm_decode_params* params, sslm_workspace ws,
                                int32_t* out_tokens);

/* Initializes the complete v2 params shape. Greedy needs no artifact extension. Damped greedy
 * selects the B0-ruled alpha/n/k defaults and derives q_ln2/q_b/q_c from this mapped model's
 * DGC1 section; a model converted without --enable-damped-greedy is rejected. */
SUPERSLM_API sslm_status sslm_decode_params_init(sslm_model model, int32_t mode, int32_t layer_budget,
                                    sslm_decode_params* out);

/* Takes the caller-memory call-shape struct (design Sec7.1) -- not part of the g5-comparable
 * subset (see above).
 *
 * Concurrency (matching
 * include/superslm/gpu_1p0.h's own per-declaration thread-safety convention): a single
 * sslm_workspace returned by sslm_workspace_create is reusable across an arbitrary SEQUENCE of
 * calls (design Sec7.1), but is NOT safe to pass to two calls running concurrently -- the
 * implementation carves per-call scratch out of it, and two concurrent calls sharing one
 * workspace handle will corrupt each other's scratch. A caller driving multiple sequences
 * concurrently needs one workspace per concurrently-active call (one per thread is the common
 * shape), exactly as sslm_kv_pool's own per-block accounting already requires one pool sized for
 * every concurrently-resident sequence (design Sec7.2). */
SUPERSLM_API size_t sslm_workspace_size(sslm_model model, const sslm_config* config);
SUPERSLM_API sslm_status sslm_workspace_create(sslm_model model, const sslm_config* config,
                                   void* buf, size_t buf_size, sslm_workspace* out);

/* Installs a caller-supplied parallel-for hook on this workspace, or clears it when `pf` is NULL
 * (superslm/parallel_for.h states the contract the hook's `run` must meet). Takes a type the
 * g5-comparable subset does not define, so it is declared here, outside that subset.
 *
 * - Copies *pf into the workspace. With no hook installed, or with ws NULL at the decode call,
 *   the token finish runs serially on the calling thread, as in every earlier release.
 * - Read by the logits step of the token finish in sslm_decode_step and sslm_decode_step_v2
 *   when this workspace is passed. With SSLM_PARALLEL_FOR_MATVEC set in pf->reserved, also read
 *   by every decode layer's one-row projections and by a prefill or prefix-prefill call that
 *   admits exactly one token (parallel_for.h states the rule). Every other call ignores it.
 *   Tokens are identical with any hook, any max_tasks, any accepted bits, and no hook.
 * - Returns SSLM_INVALID_ARGUMENT, and changes nothing, for: a NULL ws; a pf->reserved bit other
 *   than SSLM_PARALLEL_FOR_MATVEC; pf->max_tasks < 0 or > SSLM_PARALLEL_FOR_MAX_TASKS; pf->run
 *   NULL with pf->max_tasks > 1.
 * - Must not run concurrently with a call using the same workspace (the workspace's existing
 *   rule). pf->host_ctx must outlive every call made with this workspace while the hook is
 *   installed.
 * - A hook whose run omits, repeats or invents a task index fails that decode call with
 *   SSLM_INVALID_ARGUMENT. At the finish, the sequence is left resting with its final hidden
 *   state ready for logits and its schema walk and current token unchanged, so a retry recomputes
 *   the finish. In a layer (SSLM_PARALLEL_FOR_MATVEC), it rests at the start of that layer with
 *   the layer's saturation counts put back, and the next call resumes there. A failed prefill
 *   call that admitted nothing leaves the sequence's saturation counts as they were before it. */
SUPERSLM_API sslm_status sslm_workspace_set_parallel_for(sslm_workspace ws,
                                                          const sslm_parallel_for* pf);

/* Takes the incremental-detokenize state struct (design Sec7.4) -- not part of the
 * g5-comparable subset (see above). */
SUPERSLM_API sslm_status sslm_detokenize_stream(sslm_model model, sslm_detok_state* state, const int32_t* tokens,
                                    int32_t n, char* utf8, int32_t* out_n);

/* ---- Paged KV (1.12.0; paged-KV plan Sec3.4-Sec3.7): page pools, budget holders, shared prefixes ----
 * Outside the g5-comparable subset, like the workspace verbs. docs/api.md ("Paged KV memory") is
 * the full contract.
 *
 * A pool is pages of B positions (sslm_kv_page_positions). B is a function of the model's head_dim
 * (D), context cap and K/V element width: the smallest power of two of at least 16 whose page run,
 * B * D * element bytes, reaches 4 KiB (when D <= 64 and the cap is at most 4,096) or 8 KiB
 * (otherwise); the cap itself when that is larger than the cap. The CFG1 kv_block_size field does
 * not select B. Where B does not divide the cap, a holder's last page is partial: its slack positions
 * are never addressed, since writes stop at the cap. A holder made by a no-budget verb
 * (sslm_seq_create, sslm_prefix_begin, or a restore of a no-budget blob) reserves ceil(cap/B) pages,
 * one old block plus that slack, and behaves as in 1.11.0; sslm_kv_block_size stays 1.11.0's value,
 * and sslm_kv_pool_overhead_size carries each block's slack, so a pool sized as
 * N * sslm_kv_block_size + sslm_kv_pool_overhead_size(N) admits N such holders. A budget holder declares 1 <= budget <= cap positions it may write
 * itself and reserves R(budget) = min(ceil(budget/B) + 1, ceil(cap/B)) pages
 * (sslm_kv_pages_for_budget) when it is created; prefill and decode then map pages only from that
 * reserve, never from the pool. Its writes stop at limit = min(origin + budget, cap): below the
 * cap that is SSLM_KV_BUDGET_EXCEEDED (prefill admits the tokens below the limit and stops; decode
 * refuses at the token boundary with no row written, after emitting the ready token), at the cap
 * it is SSLM_CONTEXT_CAP_EXCEEDED as before. origin is 0 after create or reset, the prefix's
 * length after sslm_seq_adopt_prefix, and the parent's length for sslm_prefix_begin_from.
 *
 * Adopt: the adopter's mode decides, never the prefix's. A budget holder's adopt maps the frozen
 * prefix's full pages (shared, read-only) and copies at most one partial page; it never draws from
 * the pool, and a prefix of another pool is SSLM_INVALID_ARGUMENT. A no-budget holder's adopt
 * copies the prefix's rows into its own pages, across pools as before. A shared page is freed when
 * the prefix and every holder that maps it have released, reset or re-adopted, so a released
 * prefix's pages can stay in use; read sslm_kv_pool_stats' free_pages, not the handle count.
 * Releasing a prefix while an adopt, a begin_from or a sharing restore reads it is caller UB, as
 * for adopt in 1.11.0; releasing it while its sharers decode, reset or release is safe. Prefixes
 * stay base-only: a shared page is valid under any adapter the adopter binds.
 *
 * Save: a budget holder writes 'SSB6' (its live rows only); a no-budget holder keeps writing
 * 'SSB5' byte-equal to 1.11.0's. Output tokens and K/V values are bit-identical to 1.11.0 in
 * either mode. */

/* B, the positions per page (0 on a null model). */
SUPERSLM_API int64_t sslm_kv_page_positions(sslm_model model);
/* Bytes per page (0 on a null model or overflow). */
SUPERSLM_API size_t sslm_kv_page_size(sslm_model model);
/* R(budget), the pages a budget holder reserves; 0 for a budget outside [1, cap] or a null model. */
SUPERSLM_API size_t sslm_kv_pages_for_budget(sslm_model model, int32_t budget);
/* The contiguous pool option (1.12.0, CPU only): a pool whose page is the whole context cap, one page
 * per block, which keeps 1.11.0's contiguous layout for single-sequence speed. Every verb works on it as
 * on a page pool with B = cap: a legacy or budget holder takes one page, adopt and begin_from copy and
 * share no page, and saves are canonical, so blobs restore across pool kinds. It trades away what page
 * pools give a budget holder (sharing a prefix's pages, reserving by budget). Copy adopt between a
 * contiguous pool and a page pool of one model is SSLM_INVALID_ARGUMENT wherever their page sizes differ;
 * move state across pool kinds by save and restore. sslm_kv_page_positions, sslm_kv_page_size and
 * sslm_kv_pages_for_budget keep answering for the model's B; sslm_kv_pool_stats reports the pool's.
 * sslm_kv_pool_contiguous_size returns the bytes for `block_count` blocks, block_count *
 * (sslm_kv_block_size + 8) + 32, and 0 on a null model, a zero count or overflow. The create refuses in
 * sslm_kv_pool_create's order and alignment rule. */
SUPERSLM_API size_t sslm_kv_pool_contiguous_size(sslm_model model, uint32_t block_count);
SUPERSLM_API sslm_status sslm_kv_pool_create_contiguous(sslm_model model, void* buf, size_t buf_size, uint32_t block_count,
                                                        sslm_kv_pool* out);
/* The page-pool bookkeeping a buffer must hold beyond its pages; sslm_kv_pool_overhead_size's
 * convention (0 on a null model, SIZE_MAX where no buffer suffices). As with the block verb, the
 * pool's per-page bookkeeping (free list, reference counts, written and shared flags: 13 bytes
 * per page) is allocated on the heap at pool create, not in this part of the caller's buffer. */
SUPERSLM_API size_t sslm_kv_page_pool_overhead_size(sslm_model model, uint32_t page_count);
/* A pool of `page_count` pages over `buf` (buf_size >= page_count * sslm_kv_page_size +
 * sslm_kv_page_pool_overhead_size, SSLM_ABI_ALIGNMENT_BYTES-aligned), with sslm_kv_pool_create's
 * refusals in its order (arguments, a zero count, overflow, SSLM_BUFFER_TOO_SMALL, then
 * SSLM_MISALIGNED_BUFFER); every page_count up to UINT32_MAX is admissible by count, and heap
 * failure is SSLM_ALLOCATION_FAILED. Every holder verb takes it, budget or not; sslm_kv_pool_destroy destroys
 * it and refuses with SSLM_POOL_HAS_LIVE_HANDLES while any handle drawn from it lives. */
SUPERSLM_API sslm_status sslm_kv_page_pool_create(sslm_model model, void* buf, size_t buf_size, uint32_t page_count,
                                     sslm_kv_pool* out);
/* A budget-mode sequence: R(budget) pages drawn, SSLM_KV_POOL_EXHAUSTED with nothing drawn when the
 * pool has fewer; SSLM_INVALID_ARGUMENT for a budget outside [1, cap] or a pool of another model.
 * Reset keeps the reservation (it never draws from the pool) and sets origin to 0. */
SUPERSLM_API sslm_status sslm_seq_create_budgeted(sslm_model model, sslm_kv_pool* pool, int32_t budget, sslm_seq* out);
/* A budget-mode prefix (origin 0), as sslm_seq_create_budgeted; its prefill stops at the budget.
 * The first sslm_prefix_freeze of any prefix returns its unmapped pages to the pool, so a frozen
 * prefix holds ceil(length/B) pages; later freezes are no-ops returning SSLM_OK. */
SUPERSLM_API sslm_status sslm_prefix_begin_budgeted(sslm_model model, sslm_kv_pool* pool, int32_t budget, sslm_prefix* out);
/* A budget-mode prefix continuing a frozen `parent` of either mode (SSLM_INVALID_ARGUMENT if not
 * frozen, or for a budget outside [1, cap]): R(budget) drawn from the parent's pool
 * (SSLM_KV_POOL_EXHAUSTED with nothing drawn), the parent's full pages shared, its partial page
 * copied, and its state copied (including its schema binding and walk state); origin = the
 * parent's length. A child of a non-empty parent cannot bind a schema of its own. */
SUPERSLM_API sslm_status sslm_prefix_begin_from(sslm_prefix parent, int32_t budget, sslm_prefix* out);
/* Save a frozen prefix as canonical SPX1 bytes. A null or short buffer reports the exact size
 * through *n and returns SSLM_BUFFER_TOO_SMALL. Null handles, an unfrozen prefix, or a prefix
 * with nonzero layer_index return SSLM_INVALID_ARGUMENT without writing the buffer. Concurrent
 * saves of a frozen prefix are allowed; releasing it during a save is caller undefined behavior. */
SUPERSLM_API sslm_status sslm_prefix_save(sslm_prefix prefix, void* buf, size_t* n);
/* Restore an SPX1 prefix for this backend, engine producer, model and KV precision. The hint is
 * optional for fidelity. A nested blob (origin > 0) shares the hint's leading floor(origin/B) pages
 * only when the hint is frozen, of `model` and of `*pool`, holds exactly origin positions, and its
 * rows over those pages equal the blob's; the rest are drawn and filled from the blob, so the
 * restore draws ceil(length/B) - floor(origin/B) pages. Any other hint, a null one, or a root blob
 * restores privately (SSLM_OK, 0 shared). *out_shared_pages, when non-null, receives the count
 * shared. Without a matching hint, a nested prefix requires ceil(length/B) free pages; exhaustion
 * returns SSLM_KV_POOL_EXHAUSTED with no page drawn. Every refusal clears *out and returns every
 * page drawn and every shared reference taken. The parent may be released after the restore.
 * B is `pool`'s page size (the contiguous option's is the cap): SPX1 carries no page size, so a
 * blob restores into a pool of any page size, and the restored prefix is then adoptable only from
 * pools of that page size, as every prefix is (sslm_seq_adopt_prefix).
 * The SHA-256 digest detects accidental change, not intentional editing: a recomputed digest can
 * admit schema and walk state within model bounds, length within the context cap, origin within
 * length, a valid current token, an int32 mantissa, any int64 exponent, any saturation counts,
 * and arbitrary residual and KV bytes. Magic, backend, producer, model identity and precision
 * must match exactly. Releasing a hint while restore reads it is caller undefined behavior. */
SUPERSLM_API sslm_status sslm_prefix_restore(sslm_model model, sslm_kv_pool* pool, const void* buf,
                                              size_t n, sslm_prefix parent_or_null, sslm_prefix* out,
                                              uint32_t* out_shared_pages);
/* Page counts of a pool (either create verb): page_count, free_pages, shared_pages and B. The
 * caller sets out->struct_size. A diagnostic for sizing admissions, read under the pool lock. */
SUPERSLM_API sslm_status sslm_kv_pool_stats(sslm_kv_pool pool, sslm_kv_pool_stats_out* out);
/* A sequence's mode, origin, budget, limit, context length and pages; the caller sets
 * out->struct_size. Takes the sequence's lifecycle lock; must not race a prefill of the same
 * sequence (one caller per sequence at a time). A sequence restored from SSB5 or older reads
 * origin 0, whatever it adopted before the save; its limit and reservation do not depend on it. */
SUPERSLM_API sslm_status sslm_seq_kv_stats(sslm_seq seq, sslm_seq_kv_stats_out* out);
/* sslm_seq_restore with a sharing hint. A budget-mode 'SSB6' restores as the holder it was saved
 * from: its saved budget, origin and mode, so the same limit and the same remaining token count.
 * With `prefix_or_null` frozen, of this model and pool, at length == the blob's origin, and its
 * rows over the shared span byte-equal to the blob's (checked byte by byte), its full pages are
 * shared and *out_shared_pages receives their count. Otherwise the restore is private:
 * *out_shared_pages = 0 and SSLM_OK, never an error, and it also draws pages for the prefix span,
 * which go back to the pool at the sequence's first reset, adopt or release. A no-budget blob
 * (SSB5 and older, or a whole_reserve SSB6) ignores the handle. Every field is validated before
 * any page is drawn; SSLM_KV_POOL_EXHAUSTED draws nothing. out_shared_pages may be NULL.
 * sslm_seq_restore is this verb with a NULL prefix. */
SUPERSLM_API sslm_status sslm_seq_restore_shared(sslm_model model, sslm_kv_pool* pool, const void* blob, size_t size,
                                    sslm_prefix prefix_or_null, sslm_seq* out, uint32_t* out_shared_pages);
/* Schema jump-forward (1.15.0). The forced chain of a schema-bound sequence's current walk state:
 * from a state where the schema admits exactly one byte string before its next choice, the tokens
 * that spell it, as the forced-next table built at model map gives them (docs/api.md). Read-only:
 * writes up to `cap` ids (out_ids may be NULL when cap == 0); *out_count = the full chain length
 * (may exceed cap), 0 when no schema is bound or the state is unforced; *out_accepting = 1 iff the
 * walk state is accepting. Changes nothing, including bind eligibility. SSLM_ARTIFACT_REJECTED when
 * the model has no table (an artifact without SCM1 or TOK1). */
SUPERSLM_API sslm_status sslm_seq_forced_tokens(sslm_model model, sslm_seq seq,
                                                int32_t* out_ids, int32_t cap, int32_t* out_count,
                                                int32_t* out_accepting);
/* Admits the pending choice token (one decode_step produced and has not yet embedded), if any, then
 * forced-chain tokens, up to `chunk_budget` tokens in all, in one batched layer pass through the
 * prefill path; writes the admitted *forced* ids to out_forced (cap >= chunk_budget required) and
 * their count to *out_forced_count. `params` (NULL = greedy) is validated exactly as
 * sslm_decode_step_v2 validates it; damped params feed the forced tokens to the anti-LM history.
 * A no-op (OK, count 0) when no schema is bound or the state is unforced. A limit admits a prefix
 * and returns the limit status, as sslm_prefill does. The host loop: call this until
 * sslm_seq_forced_tokens reports 0, then decode_step (docs/api.md gives the contract).
 * `opts` (1.15.0; NULL = canonical, the behaviour above): with mode SSLM_JUMP_EXACT the verb
 * admits the pending token and a draft clipped to chunk_budget - 1 tokens and to the context
 * room, truncated at its first out-of-range, special, byteless or illegal token; checks every
 * drafted token with the decode step's finish; and keeps the pending token and the drafted
 * tokens the model confirms, reporting the confirmed ones in out_forced. It returns SSLM_OK for
 * any draft, and is a no-op (OK, count 0) when there is no pending token, the clipped draft is
 * empty, or the model lacks the exact legal-row finish. The next decode_step produces the
 * model's own next token (docs/api.md). */
SUPERSLM_API sslm_status sslm_seq_jump_forward(sslm_model model, sslm_seq seq, int32_t chunk_budget,
                                               const sslm_decode_params* params, /* NULL = greedy */
                                               const sslm_jump_options* opts,     /* NULL = canonical */
                                               sslm_workspace ws,
                                               int32_t* out_forced, int32_t cap, int32_t* out_forced_count);

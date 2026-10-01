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

/* ---- Paged KV (paged-KV plan Sec3.4-Sec3.7): page pools, budget holders, shared prefixes ----------
 * Outside the g5-comparable subset, like the workspace verbs.
 *
 * A pool is pages of B positions (sslm_kv_page_positions). A holder made by a no-budget verb
 * (sslm_seq_create, sslm_prefix_begin, or a restore of a no-budget blob) reserves ceil(cap/B)
 * pages, one old block, and behaves as before. A budget holder declares 1 <= budget <= cap
 * positions it may write itself and reserves R(budget) = min(ceil(budget/B) + 1, ceil(cap/B))
 * pages (sslm_kv_pages_for_budget); its writes stop at limit = min(origin + budget, cap) with
 * SSLM_KV_BUDGET_EXCEEDED below the cap. origin is 0 after create or reset, the prefix's length
 * after sslm_seq_adopt_prefix, and the parent's length for sslm_prefix_begin_from. A budget
 * holder's adopt shares the frozen prefix's full pages (it never draws from the pool, and a prefix
 * of another pool is SSLM_INVALID_ARGUMENT); a no-budget holder's adopt copies, as before. A
 * prefix's pages are freed when the prefix and every holder that shares them have released or
 * reset. Releasing a prefix while an adopt, a begin_from or a sharing restore reads it is caller
 * UB, as for adopt today. A budget holder saves the 'SSB6' blob; a no-budget holder keeps saving
 * 'SSB5'. */

/* B, the positions per page (0 on a null model). */
SUPERSLM_API int64_t sslm_kv_page_positions(sslm_model model);
/* Bytes per page (0 on a null model or overflow). */
SUPERSLM_API size_t sslm_kv_page_size(sslm_model model);
/* R(budget), the pages a budget holder reserves; 0 for a budget outside [1, cap] or a null model. */
SUPERSLM_API size_t sslm_kv_pages_for_budget(sslm_model model, int32_t budget);
/* The page-pool bookkeeping a buffer must hold beyond its pages; sslm_kv_pool_overhead_size's
 * convention (0 on a null model, SIZE_MAX where no buffer suffices). */
SUPERSLM_API size_t sslm_kv_page_pool_overhead_size(sslm_model model, uint32_t page_count);
/* A pool of `page_count` pages over `buf` (buf_size >= page_count * sslm_kv_page_size +
 * sslm_kv_page_pool_overhead_size, SSLM_ABI_ALIGNMENT_BYTES-aligned), with sslm_kv_pool_create's
 * refusals in its order. Every holder verb takes it; sslm_kv_pool_destroy destroys it. */
SUPERSLM_API sslm_status sslm_kv_page_pool_create(sslm_model model, void* buf, size_t buf_size, uint32_t page_count,
                                     sslm_kv_pool* out);
/* A budget-mode sequence: R(budget) pages drawn, SSLM_KV_POOL_EXHAUSTED with nothing drawn when the
 * pool has fewer; SSLM_INVALID_ARGUMENT for a budget outside [1, cap]. */
SUPERSLM_API sslm_status sslm_seq_create_budgeted(sslm_model model, sslm_kv_pool* pool, int32_t budget, sslm_seq* out);
/* A budget-mode prefix (origin 0), as sslm_seq_create_budgeted. Freeze returns its unmapped pages. */
SUPERSLM_API sslm_status sslm_prefix_begin_budgeted(sslm_model model, sslm_kv_pool* pool, int32_t budget, sslm_prefix* out);
/* A budget-mode prefix continuing a frozen `parent` (SSLM_INVALID_ARGUMENT if not frozen): R(budget)
 * drawn from the parent's pool, the parent's full pages shared, its state copied; origin = the
 * parent's length. */
SUPERSLM_API sslm_status sslm_prefix_begin_from(sslm_prefix parent, int32_t budget, sslm_prefix* out);
/* Page counts of a pool (either create verb). */
SUPERSLM_API sslm_status sslm_kv_pool_stats(sslm_kv_pool pool, sslm_kv_pool_stats_out* out);
/* A sequence's mode, origin, budget, limit and pages. Takes the sequence's lifecycle lock; must not
 * race a prefill of the same sequence (one caller per sequence at a time). */
SUPERSLM_API sslm_status sslm_seq_kv_stats(sslm_seq seq, sslm_seq_kv_stats_out* out);
/* sslm_seq_restore with a sharing hint. A budget-mode 'SSB6' restores with its saved budget, origin
 * and mode. With `prefix_or_null` frozen, of this model and pool, at length == the blob's origin,
 * and its rows over the shared span byte-equal to the blob's, its full pages are shared and
 * *out_shared_pages receives their count; otherwise the restore is private (it also draws the
 * prefix span's pages, returned at the first reset, adopt or release), *out_shared_pages = 0 and
 * SSLM_OK. A no-budget blob ignores the handle. out_shared_pages may be NULL. sslm_seq_restore is
 * this verb with a NULL prefix. */
SUPERSLM_API sslm_status sslm_seq_restore_shared(sslm_model model, sslm_kv_pool* pool, const void* blob, size_t size,
                                    sslm_prefix prefix_or_null, sslm_seq* out, uint32_t* out_shared_pages);

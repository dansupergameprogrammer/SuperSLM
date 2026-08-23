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
#include "superslm/sslm_abi_functions_g5_comparable.inc"

/* Versioned decode entry point. The legacy sslm_decode_step declaration lives in the
 * g5-comparable inventory because it is the v1.1 surface mirrored by that suite. This v2 symbol
 * is deliberately separate: it may validate and consume sslm_decode_params fields beyond the
 * shipped four-byte layer_budget prefix without reading past an old caller's allocation. */
sslm_status sslm_decode_step_v2(sslm_model model, sslm_seq* seqs, int32_t n,
                                const sslm_decode_params* params, sslm_workspace ws,
                                int32_t* out_tokens);

/* Initializes the complete v2 params shape. Greedy needs no artifact extension. Damped greedy
 * selects the B0-ruled alpha/n/k defaults and derives q_ln2/q_b/q_c from this mapped model's
 * DGC1 section; a model converted without --enable-damped-greedy is rejected. */
sslm_status sslm_decode_params_init(sslm_model model, int32_t mode, int32_t layer_budget,
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
size_t sslm_workspace_size(sslm_model model, const sslm_config* config);
sslm_status sslm_workspace_create(sslm_model model, const sslm_config* config,
                                   void* buf, size_t buf_size, sslm_workspace* out);

/* Takes the incremental-detokenize state struct (design Sec7.4) -- not part of the
 * g5-comparable subset (see above). */
sslm_status sslm_detokenize_stream(sslm_model model, sslm_detok_state* state, const int32_t* tokens,
                                    int32_t n, char* utf8, int32_t* out_n);

/* ---------------------------------------------------------------------------
 * T-2246 speculative decoding (plan SS3.6), appended per this file's own stated
 * evolution rule (appends are how this ABI grows).
 *
 * sslm_speculate_params_init fills a KNOWN-VALID params shape: max_draft_tokens
 * defaulted to 4, budget defaulted generous, no stop set -- the
 * sslm_decode_params_init precedent.
 *
 * sslm_speculate_step_v3 proposes up to params->max_draft_tokens drafts from the
 * sequence's retained committed tokens, verifies them integer-exactly against the
 * target argmax in ONE chunk-batched pass, emits accepted drafts plus the
 * bonus/divergence target applying the shipped per-token stop/cap tests in order
 * (append, stop test, cap test per position), commits on Ok, and restores the
 * snapshotted three-term sequence state on any rejection. Single sequence by
 * signature. Free-text greedy scope only: a schema-bound sequence rejects with
 * SSLM_INVALID_ARGUMENT.
 *
 * out_tokens receives up to out_tokens_capacity ids; out_logit_rows receives the
 * emitted tokens' target-argmax logit rows, row-major -- row i is the full int32
 * logit row composed immediately before the i-th emitted token was selected (the
 * caller-owned-rows convention RunGreedyDecodeLoop establishes). Capacity is
 * checked up front against the call's worst reachable emission count,
 * min(K+1, remaining budget); an undersized buffer rejects SSLM_BUFFER_TOO_SMALL
 * before any landing. On a truncated call, out slots beyond *out_tokens_produced
 * have unspecified content; nothing beyond the worst reachable count is ever
 * written. Digesting stays outside the ABI: the verb emits raw arrays only, so
 * ComputeTokenDigest/ComputeFinalLogitDigest remain pure functions of those
 * arrays. On any rejection the sequence's header fields, residual, and anti-LM
 * state are byte-identical to pre-call over the canonicalized save-blob
 * comparison (plan SS3.3); raw KV bytes beyond context_length are stale-but-
 * unread by design and excluded there, exactly as for every other path.
 *
 * sslm_seq_committed_token_count returns the sequence's retained committed-token
 * id count (plan SS3.0 lifecycle checkpoints' quantity of record).
 * sslm_seq_committed_tokens_peek copies up to *io_count retained ids starting at
 * start_index into out_ids and sets *io_count to the number written. Read-back
 * is observation only -- retention stays unserialized (plan SS4). A
 * [start_index, start_index + *io_count) range crossing past the retained count
 * rejects SSLM_INVALID_ARGUMENT; an undersized (or null) out_ids buffer rejects
 * SSLM_BUFFER_TOO_SMALL with *io_count set to the required count, the shipped
 * two-call sizing convention.
 * --------------------------------------------------------------------------- */
sslm_status sslm_speculate_params_init(sslm_model model, sslm_speculate_params* out);
sslm_status sslm_speculate_step_v3(sslm_model model, sslm_seq seq,
                                    const sslm_speculate_params* params, sslm_workspace ws,
                                    int32_t* out_tokens, int32_t out_tokens_capacity,
                                    int32_t* out_logit_rows, int32_t out_rows_capacity,
                                    int32_t* out_tokens_produced, int32_t* out_stop_reason);
sslm_status sslm_seq_committed_token_count(sslm_seq seq, int64_t* out_count);
sslm_status sslm_seq_committed_tokens_peek(sslm_seq seq, int64_t start_index,
                                            int32_t* out_ids, int64_t* io_count);

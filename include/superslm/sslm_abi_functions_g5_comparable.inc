/* sslm_abi_functions_g5_comparable.inc -- the subset of sslm_abi_functions.inc's own
 * declarations whose signature uses only types tests/t2130-g5-red-suite/sslm_g5.h itself
 * defines: sslm_model, sslm_seq, sslm_prefix, sslm_adapter, sslm_kv_pool, sslm_workspace,
 * sslm_status, sslm_span_kind, sslm_decode_params, sslm_stats_out, and C's own primitives.
 *
 * Split out so this file, unlike the whole of sslm_abi_functions.inc, can be included in a
 * translation unit where only sslm_g5.h's own types are visible. Two of this design's own
 * invented types have no counterpart anywhere in sslm_g5.h at all -- the caller-memory
 * call-shape struct and the incremental-detokenize state struct, Claude/Vitruvius/
 * t2133-layer1-c-abi-design-2026-08-16.md Sec9's own residual d -- and every declaration whose
 * signature names either of those two types is excluded from this file, staying directly in
 * sslm_abi_functions.inc instead: the two workspace-sizing declarations that take the call-shape
 * struct, and the one text-I/O declaration that takes the detokenize-state struct.
 *
 * sslm_abi_functions.inc includes this file for the shared subset, so there is exactly one
 * place each of these declarations is written -- not a second, maintained-in-parallel copy.
 * tools/t2139_gate_a_header_parity_check.cpp includes this same file too (macro-renamed) to
 * check the declarations this library actually ships against sslm_g5.h's own real names, which
 * is why the two excluded declarations above are correspondingly excluded from Gate A's own
 * comparison surface until/unless either invented type gains a second, independent definition
 * to compare against -- the same residual Gate C's own construction states for these two types.
 */
sslm_status sslm_model_map(const void* data, size_t size, sslm_model* out);
sslm_status sslm_model_unmap(sslm_model model);

size_t sslm_kv_block_size(sslm_model model);
size_t sslm_kv_pool_overhead_size(sslm_model model, uint32_t block_count);
size_t sslm_seq_state_size(sslm_model model);

sslm_status sslm_kv_pool_create(sslm_model model, void* buf, size_t buf_size,
                                 uint32_t block_count, sslm_kv_pool* out);
sslm_status sslm_kv_pool_destroy(sslm_kv_pool pool);
sslm_status sslm_workspace_destroy(sslm_workspace ws);

sslm_status sslm_prefix_begin(sslm_model model, sslm_kv_pool* pool, sslm_prefix* out);
sslm_status sslm_prefix_prefill(sslm_model model, sslm_prefix prefix, const int32_t* tokens,
                                 int32_t count, int32_t chunk_budget, sslm_span_kind kind,
                                 sslm_workspace ws, int32_t* consumed);
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

sslm_status sslm_adapter_map(const void* data, size_t size, sslm_model base, sslm_adapter* out);
sslm_status sslm_adapter_release(sslm_adapter adapter);
size_t      sslm_adapter_residency(sslm_adapter adapter);

sslm_status sslm_prefill(sslm_model model, sslm_seq seq, const int32_t* tokens, int32_t count,
                          int32_t chunk_budget, sslm_span_kind kind, sslm_workspace ws,
                          int32_t* consumed);
sslm_status sslm_decode_step(sslm_model model, sslm_seq* seqs, int32_t n,
                              const sslm_decode_params* params, sslm_workspace ws,
                              int32_t* out_tokens);

sslm_status sslm_tokenize(sslm_model model, const char* utf8, int32_t* tokens, int32_t* n);

sslm_status sslm_stats(sslm_model model, sslm_seq seq, sslm_stats_out* out);

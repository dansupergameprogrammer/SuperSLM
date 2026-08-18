/* sslm_abi_functions_g5_comparable.inc -- the subset of sslm_abi_functions.inc's own
 * declarations whose signature uses only types tests/t2130-g5-red-suite/sslm_g5.h itself
 * defines: sslm_model, sslm_seq, sslm_prefix, sslm_adapter, sslm_kv_pool, sslm_workspace,
 * sslm_status, sslm_span_kind, sslm_decode_params, sslm_stats_out, and C's own primitives.
 *
 * Split out so this file, unlike the whole of sslm_abi_functions.inc, can be included in a
 * translation unit where only sslm_g5.h's own types are visible. Two of this design's own
 * invented types have no counterpart anywhere in sslm_g5.h at all -- the caller-memory
 * call-shape struct and the incremental-detokenize state struct -- and every declaration whose
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
/* Exception-boundary contract (design Sec6, fold 2026-08-17 second pass -- matching this
 * file's own per-declaration convention), SCOPED to this verb's OWN
 * SslmModel::Load call specifically -- not the whole verb's contract: the Load call
 * is fronted by SslmModel::Load's public entry (model.h/model.cpp), which is itself wrapped in
 * internal::WrapBadAllocContract (src/bad_alloc_wrap.h) -- pre-existing, independently-governed
 * S-HARDEN-7 machinery this ABI does not own. WrapBadAllocContract narrows every internal
 * std::exception-derived failure OTHER than std::bad_alloc itself (std::length_error,
 * std::runtime_error, std::logic_error, and every other standard or user-defined std::exception
 * subtype) to std::bad_alloc before this ABI's own extern "C" boundary can classify it. For THE
 * LOAD CALL specifically, the delivered contract is therefore: any std::exception-derived internal
 * cause surfaces as SSLM_ALLOCATION_FAILED -- not narrowed to true allocation/size exhaustion the
 * way the general catch(bad_alloc)/catch(length_error)/catch(...) split (src/sslm_abi.cpp,
 * CatchAllocationFailure) is elsewhere in this file. SSLM_ARTIFACT_REJECTED still fires from the
 * Load call's own catch(...) arm, but only for a throw NOT derived from std::exception at all (a
 * foreign type, or a non-class throw), which WrapBadAllocContract does not narrow and lets
 * through unmolested. Proven by real fault injection through this exact call path
 * (tools/t2139_f2_length_error_pin.cpp's own kRuntimeError cell).
 *
 * This verb ALSO has a second exception-handling region -- its own post-Load construction step
 * (BuildEngineCache, src/sslm_abi.cpp), which is NOT fronted by WrapBadAllocContract and instead
 * follows CatchAllocationFailure's own general catch(bad_alloc)/catch(length_error)/catch(...)
 * split: a non-allocation std::exception raised there returns SSLM_ARTIFACT_REJECTED, not
 * SSLM_ALLOCATION_FAILED. Proven by real fault injection through this exact region
 * (tools/t2139_d3466_postload_region_pin.cpp). src/sslm_abi.cpp's own SCOPE paragraph (above
 * CatchAllocationFailure's definition) states this split for both call regions of both
 * Load-fronted verbs; this comment restates only the Load-call half because that is the half this
 * declaration's own signature fronts directly. */
sslm_status sslm_model_map(const void* data, size_t size, sslm_model* out);
sslm_status sslm_model_unmap(sslm_model model);

/* sslm_kv_block_size (design Sec7.2, RULED design commit fab235c1c6): one WHOLE sequence's
 * entire KV footprint across every layer, never a sub-sequence PagedAttention page (that
 * indirection is committed post-1.0 work) -- num_hidden_layers * context_cap *
 * num_key_value_heads * head_dim * 2 (K+V) * kv_precision_width. */
size_t sslm_kv_block_size(sslm_model model);
size_t sslm_kv_pool_overhead_size(sslm_model model, uint32_t block_count);
size_t sslm_seq_state_size(sslm_model model);

/* sslm_kv_pool_create's block_count (design Sec7.2, RULED): how many concurrent SEQUENCES this
 * pool can back, one whole block each -- for N sequences concurrently resident, block_count = N
 * (the single-sequence case, N = 1, is what the S-FREEZE example uses, design Sec9). No token-
 * count arithmetic: a block is no longer sub-divided. */
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

/* Exception-boundary contract -- identical to sslm_model_map's own comment above,
 * SCOPED THE SAME WAY (S2): this verb's own SslmModel::Load call is also fronted by
 * WrapBadAllocContract, so the same narrowing applies to that call specifically -- any
 * std::exception-derived internal cause reaching the Load call surfaces as SSLM_ALLOCATION_FAILED,
 * and that call's own catch(...) arm returns SSLM_ARTIFACT_REJECTED only for a throw not derived
 * from std::exception at all. This verb's own SEPARATE post-Load construction step
 * (PopulateAdapterFromView, adapter_marshal.h, wrapped via CatchAllocationFailure in
 * src/sslm_abi.cpp) is NOT fronted by WrapBadAllocContract and follows the general
 * catch(bad_alloc)/catch(length_error)/catch(...) split instead -- a non-allocation std::exception
 * there returns SSLM_ARTIFACT_REJECTED. Proven by real fault injection through this exact region
 * (tools/t2139_d3466_postload_region_pin.cpp). */
sslm_status sslm_adapter_map(const void* data, size_t size, sslm_model base, sslm_adapter* out);
sslm_status sslm_adapter_release(sslm_adapter adapter);
size_t      sslm_adapter_residency(sslm_adapter adapter);

sslm_status sslm_prefill(sslm_model model, sslm_seq seq, const int32_t* tokens, int32_t count,
                          int32_t chunk_budget, sslm_span_kind kind, sslm_workspace ws,
                          int32_t* consumed);
/* design Sec14.1: out_tokens[i]
 * reserves TWO sentinels, not one -- the existing -1 ("pending, mid-token, call again with the
 * same layer_budget-carried state") and -2 ("this sequence's schema-bound walk reached a state
 * with NO legal continuation at all -- the argmax-selected token has no matching CSR transition
 * entry, which Sec13.2 permits at an accepting state with a legitimately all-zero mask page").
 * -2 is a per-sequence outcome, not a call fault: the call's own return status stays SSLM_OK,
 * and nothing schema-related advances for that sequence (dfa_walk_state/current_token/
 * layer_index all stay exactly as this call found them) -- calling again on the same sequence
 * deterministically returns -2 again. No new sslm_status ordinal. */
sslm_status sslm_decode_step(sslm_model model, sslm_seq* seqs, int32_t n,
                              const sslm_decode_params* params, sslm_workspace ws,
                              int32_t* out_tokens);

sslm_status sslm_tokenize(sslm_model model, const char* utf8, int32_t* tokens, int32_t* n);

sslm_status sslm_stats(sslm_model model, sslm_seq seq, sslm_stats_out* out);

/* G5 (design Sec5): schema lookup, read-only, no map/release verb.
 * Unknown name is SSLM_SCHEMA_NOT_FOUND. Lookup-by-name. */
sslm_status sslm_schema_lookup(sslm_model model, const char* name, sslm_schema* out);
size_t      sslm_schema_count(sslm_model model);
/* *n query/fill shape matches sslm_tokenize's
 * own established convention (call with buf==nullptr or *n too small to get the required byte
 * count in *n via SSLM_BUFFER_TOO_SMALL, then call again with a buffer of that size). buf is
 * filled with EXACTLY *n bytes on success -- the schema's own name bytes, NOT NUL-terminated
 * (the converter's own name blob carries no terminator, sslm_convert_schema.py). A caller that
 * wants a C string appends '\0' itself in a buffer sized *n + 1. */
sslm_status sslm_schema_name(sslm_model model, int32_t index, char* buf, size_t* n);

/* G5 design Sec5: binds a compiled schema to a sequence (SSLM_SCHEMA_NONE = unconstrained).
 * Valid ONLY when the sequence's DFA-walk state is at its start (fresh sslm_seq_create, or
 * immediately after sslm_seq_reset) -- a non-fresh walk state rejects
 * (SSLM_SCHEMA_BIND_REJECTED). No mid-generation re-binding (design Sec5/Sec11, out of scope). */
sslm_status sslm_seq_set_schema(sslm_seq seq, sslm_schema schema);

/* G5 design Sec5: "Prefix construction inherits the same discriminator." Valid only before any
 * content has been prefilled into the prefix under construction. */
sslm_status sslm_prefix_set_schema(sslm_prefix prefix, sslm_schema schema);

#ifndef SSLM_GPU_1P0_G5_BRIDGE_H
#define SSLM_GPU_1P0_G5_BRIDGE_H
// gpu_1p0_g5_bridge.h -- G5-5 (T-2132, Brunel): the build-seat-owned schema/parity bridge for
// the 1.0 GPU API, mirroring gpu_1p0_bench_bridge.h's own established shape and disclosure
// (T-2113 B3): NOT part of the shipped 1.0 API surface `include/superslm/gpu_1p0.h` declares.
//
// Why a SEPARATE header rather than new functions appended to gpu_1p0.h: gpu_1p0.h's own
// banner comment states it stays byte-for-byte declaration-identical to the red suite's own
// canonical copy (tests/t2112-gpu-1p0-red-suite/sslm_gpu_1p0.h), checked at every B-section
// checkpoint -- that suite header is read-only per this ticket's invocation contract, so this
// build cannot extend it, and appending new declarations to gpu_1p0.h alone would desynchronize
// the two copies the existing checkpoint discipline requires stay identical. Every function
// below is therefore declared here instead, exactly the seam gpu_1p0_bench_bridge.h already
// established for T-2113 B3's own narrower need.
//
// Scope (design Claude/Vitruvius/t2119-g5-constrained-decoding-design-2026-08-16.md Sec6, Slot
// G5-5 -- "GPU parity for constrained decoding"): the GPU-1.0 layer-loop substrate
// (SslmGpuSequenceHandle) already proves its own hidden-state output bit-identical to the CPU
// oracle at the base (unconstrained) level (tests/t2112-gpu-1p0-red-suite's own dim6). Design
// Sec4's own architecture table states the constraint engine introduces NO NEW ARITHMETIC --
// masking is "table lookup + bitmask AND, int32 logits, pre-argmax," pure host arithmetic in
// both the CPU and GPU paths, never a device kernel. This bridge is therefore exactly what is
// needed to drive that already-proven substrate through a schema-bound sequence and finish each
// token with the SAME `superslm::ApplyMaskAndArgmax` primitive (forward_sites.h) the CPU ABI's
// own `sslm_decode_step` calls -- bit-parity is achievable BY CONSTRUCTION (identical function,
// fed identical bytes), which is what the parity harness below proves for real rather than by
// argument alone.
//
// tools/t2132_g5_gpu_parity.cpp is this header's own intended includer.

#include <cstddef>
#include <cstdint>

#include "superslm/gpu_1p0.h"  // SslmGpuContext/SslmGpuModelHandle/SslmGpuSequenceHandle (opaque)

// Sentinel matching the CPU ABI's own `kDfaWalkStateUnused` (src/sslm_abi.cpp) -- "no schema
// ever bound" on a freshly-created sequence handle.
constexpr uint32_t kSslmGpuDfaWalkStateUnused = 0xFFFFFFFFu;

// True iff `model`'s own mapped artifact carried a SchemaMasks (SCM1) section -- an artifact
// with none is a valid, unconstrained-only artifact (design Sec13.1), exactly the CPU ABI's own
// disposition; every schema-bound call below is meaningless against such a model.
bool SslmGpuModelHasSchemasForG5Bridge(SslmGpuModelHandle* model);

// Resolves a compiled schema by name against `model`'s own host-side parsed SchemaMasksTable
// (built once, at sslm_gpu_model_map time, from the SAME section bytes the CPU ABI parses) --
// the GPU-1.0 twin of `sslm_schema_lookup`. Returns the schema's own index (>= 0) on a match,
// -1 on no match -- this bridge's own small surface has no status-enum plumbing of its own, so
// "not found" is reported the same way `SchemaMasksTable::ByName` already does at its own layer.
int32_t SslmGpuSchemaLookupForG5Bridge(SslmGpuModelHandle* model, const char* name);

// Binds `schema_index` (as returned by the lookup above; -1 unbinds, mirroring
// SSLM_SCHEMA_NONE) to `seq`, valid ONLY when `seq`'s own DFA-walk-state is at a fresh/reset
// start (mirrors `sslm_seq_set_schema`'s own "valid only when the sequence's DFA-walk state is
// at its start" precondition, design Sec5) -- returns SSLM_SEQUENCE_REJECTED on a non-fresh walk
// state (the closest existing SslmGpuStatus disposition for a per-sequence structural rejection,
// design Sec9's own "no dedicated enumerator per guard reason" convention, gpu_1p0.h's own
// SSLM_SEQUENCE_REJECTED comment). A caller-malformed handle (`seq`/`model` null, `ctx`
// mismatch) returns SSLM_SEQUENCE_KV_BUFFER_MISMATCH, the existing "malformed handle" bucket
// every 1.0 entry point already uses.
SslmGpuStatus SslmGpuSeqSetSchemaForG5Bridge(SslmGpuContext* ctx, SslmGpuSequenceHandle* seq,
                                              int32_t schema_index);

// Reads `seq`'s own current DFA-walk-state -- kSslmGpuDfaWalkStateUnused if no schema is bound.
uint32_t SslmGpuSeqWalkStateForG5Bridge(SslmGpuSequenceHandle* seq);

// Finishes a token once `seq`'s own layer loop has reached full depth (caller-ensures: drained
// via `sslm_gpu_ready` to Idle, `seq`'s own layer_index == model->num_hidden_layers -- the exact
// precondition every existing 1.0 call in this file caller-ensures at its own layer, gpu_1p0.h's
// own established convention) -- runs final_norm + logits (the SAME `RmsNormSite`/`LogitsSite`
// calls sslm_decode_step's own finishing block uses, on `seq`'s own GPU-derived, already-
// read-back hidden_codes) then, if a schema is bound, `superslm::ApplyMaskAndArgmax` indexed by
// `seq`'s own walk-state (G5-5's own "no new arithmetic" claim, made checkable) -- advances the
// walk-state via `SchemaMasksTable::Transition`, exactly `sslm_decode_step`'s own masked-argmax
// step. No schema bound: plain `ArgmaxLowestIndexTieBreak`, byte-for-byte the pre-G5 path (the
// same SSLM_SCHEMA_NONE regression requirement the CPU ABI's own gate already proves, mirrored
// here for the GPU twin). On success, `*out_token` is the produced token id and `seq`'s own
// layer_index resets to 0 (the resting convention, matching CPU's identical reset). Returns
// SSLM_SEQUENCE_REJECTED if the precondition (full depth reached) does not hold -- no dedicated
// enumerator exists for "not ready yet," same "no more specific status" disposition this whole
// file already uses throughout.
SslmGpuStatus SslmGpuSeqFinishTokenForG5Bridge(SslmGpuContext* ctx, SslmGpuSequenceHandle* seq,
                                                int32_t* out_token);

// Jump-forward's own GPU twin (design Sec6 G5-5's own "Builds: jump-forward's batched-prefill
// K/V population issued as GPU dispatches under the existing bounded-dispatch-budget primitive
// (sslm_decode_step_gpu)"). Drives `count` FORCED tokens (already known -- never chosen, no
// masking/argmax involved, exactly `PrefillWholeTokensImpl`'s own SSLM_SPAN_SCHEMA_CONTENT
// branch, src/sslm_abi.cpp) through the full embed -> layer-loop-to-depth -> commit sequence,
// ONE token at a time (each token issued as up to `dispatch_budget_per_token`-sized
// `sslm_decode_step_gpu` calls, internally, so the caller's own dispatch-budget policy still
// governs how many layers land per GPU submission -- the SAME primitive, never a second one).
// Reachability is checked BEFORE each token's own forward pass (`SchemaMasksTable::Transition`
// against `seq`'s own current walk-state) -- a token that leaves the DFA's language is rejected
// (SSLM_SEQUENCE_REJECTED) with `seq`'s own walk-state, layer_index, and context_length left
// exactly at their last-good value (mirrors `PrefillWholeTokensImpl`'s own "the sequence's own
// walk-state is unmoved by the rejected call" contract) and `*consumed` reporting how many
// tokens were admitted before the rejection -- matching `sslm_prefill`'s own partial-consumption
// contract. Requires a schema already bound (SSLM_SEQUENCE_REJECTED otherwise -- the GPU twin of
// SSLM_SCHEMA_SPAN_UNBOUND, no dedicated enumerator on this bridge's own small surface).
SslmGpuStatus SslmGpuSeqPrefillSchemaContentForG5Bridge(SslmGpuContext* ctx,
                                                          SslmGpuSequenceHandle* seq,
                                                          const int32_t* tokens, int32_t count,
                                                          uint32_t dispatch_budget_per_token,
                                                          int32_t* consumed);

#endif  // SSLM_GPU_1P0_G5_BRIDGE_H

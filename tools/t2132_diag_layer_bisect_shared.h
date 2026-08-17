// t2132_diag_layer_bisect_shared.h -- DISPOSABLE diagnostic tool (T-2132, Brunel), localizing the
// >18-step CPU/GPU substrate divergence recorded in
// Claude/Brunel/t2132-g5-build-2026-08-16.md ("Session 2 ... walk-state replay diagnostic").
//
// NOT a production artifact -- delete this file and its two driver TUs, and revert the temporary
// T2132Diag* accessors appended to src/sslm_abi.cpp, before this diagnosis session's own exit
// (invocation contract: disposable tools/ instrumentation only, src/ and include/ diff clean at
// exit).
//
// Same split-TU shape as tools/t2132_g5_gpu_parity_shared.h and for the identical reason: the CPU
// ABI's `sslm_status` and the GPU-1.0 substrate's `SslmGpuStatus` are both plain C enums at global
// scope sharing several enumerator spellings, so neither header may appear in the same TU as the
// other. This header carries only plain PODs.
#ifndef T2132_DIAG_LAYER_BISECT_SHARED_H
#define T2132_DIAG_LAYER_BISECT_SHARED_H

#include <cstdint>
#include <string>
#include <vector>

// One layer boundary's own snapshot of the sequence's hidden state -- the SAME fields
// tools/t2113_b7_batch_smoke.cpp's own SnapshotsMatch already compares (hidden_codes byte-for-
// byte, hidden_scale.m/.e) -- taken after `layer_index` many layers have run for the token
// currently being decoded at the bisected step.
struct DiagLayerSnapshot {
	uint32_t layer_index_after = 0;
	std::vector<int8_t> hidden_codes;
	int32_t scale_m = 0;
	int32_t scale_e = 0;
	int64_t context_length = -1;  // KV positions committed so far, when the caller supplies it.
};

struct DiagStepResult {
	bool setup_ok = false;
	std::string last_error;
	// One entry per RunLayerLoop/DriveToFullDepth call at the bisected step -- num_hidden_layers
	// entries on a clean run (one per layer, since both paths are driven exactly one layer per
	// call at the bisected step: CPU via sslm_decode_step's own resumable layer_budget=1, GPU via
	// sslm_decode_step_gpu's existing kDispatchesPerLayer-sized budget).
	std::vector<DiagLayerSnapshot> layer_snapshots;
	int32_t produced_token = -1;  // this path's own real production output at the bisected step.

	// One entry per ORDINARY (full layer_budget) step in the prefix 0..target_step-1 -- the RAW,
	// pre-final_norm residual (seq->state.hidden_codes, untouched by finish -- RmsNormSite/
	// SslmGpuSeqFinishTokenForG5Bridge both READ it into a SEPARATE final_codes buffer, never
	// overwrite it) at the moment each ordinary step's own decode call returns. Lets the caller
	// check whether the two paths' hidden state already numerically differs at some step BEFORE
	// the bisected one, even though every prior step's own ARGMAX-chosen token still matched (a
	// tie-break-tolerant equality that can hide a real, already-present drift).
	std::vector<DiagLayerSnapshot> prefix_step_snapshots;
};

// Runs the REAL, unmodified production CPU ABI (sslm_model_map/sslm_prefill/sslm_decode_step)
// through `target_step` ordinary (full layer_budget) decode calls (steps 0..target_step-1), then
// drives the `target_step`-th call one layer at a time (params.layer_budget=1, S3.7's own
// resumable contract), snapshotting the real production sslm_seq's hidden state after every
// single-layer call via this session's temporary T2132DiagSeq*ForBench accessors
// (src/sslm_abi.cpp, disposable). `out_prompt_tokens`, if non-null, is filled with the real
// tokenized prompt so the GPU driver can reuse the identical token ids without its own tokenizer
// call.
DiagStepResult RunCpuDiagStep(const uint8_t* bytes, size_t byte_count, const std::string& prompt,
                               const std::string& schema_name, int32_t target_step,
                               std::vector<int32_t>* out_prompt_tokens);

// The GPU-side twin. Drives the real GPU-1.0 substrate (through the G5-5 bridge) with driving
// logic copied verbatim from tools/t2132_g5_gpu_parity_gpu.cpp's own already-verified
// PrefillPrompt/DriveToFullDepth (unmodified algorithm, only added snapshot capture) through
// `target_step` ordinary decode calls, then the `target_step`-th call one layer at a time
// (kDispatchBudget == kDispatchesPerLayer, t2113's own established one-layer-per-call
// convention), snapshotting via the EXISTING, already-public SslmGpuSeqHandleHiddenCodesForBench
// bench accessor (gpu_1p0_bench_bridge.h, T-2113 B3) after each call.
DiagStepResult RunGpuDiagStep(const uint8_t* bytes, size_t byte_count,
                               const std::vector<int32_t>& prompt_tokens,
                               const std::string& schema_name, int32_t target_step);

#endif  // T2132_DIAG_LAYER_BISECT_SHARED_H

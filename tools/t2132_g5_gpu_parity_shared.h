// t2132_g5_gpu_parity_shared.h -- G5-5 (T-2132, Brunel): the plain, POD-only interface between
// this tool's CPU driver TU (t2132_g5_gpu_parity_cpu.cpp) and GPU driver TU
// (t2132_g5_gpu_parity_gpu.cpp).
//
// Why two driver TUs at all: `include/superslm/sslm_abi.h`'s `sslm_status` and
// `include/superslm/gpu_1p0.h`'s `SslmGpuStatus` are BOTH plain C-style `typedef enum`s at
// GLOBAL scope, and they share several enumerator spellings verbatim (SSLM_OK,
// SSLM_ADAPTER_MODEL_MISMATCH, SSLM_MODEL_HAS_LIVE_SEQUENCES, SSLM_TOKEN_ID_OUT_OF_RANGE,
// SSLM_RESTORE_MODEL_MISMATCH) -- a real, previously-latent header collision this tool is the
// first thing in the tree to trip, because it is the first build target that needs the CPU
// production ABI and the GPU-1.0 substrate in the same translation unit. Neither header is
// available to fix here (both are frozen/declaration-pinned public surfaces this ticket's own
// invocation contract holds read-only in that respect -- gpu_1p0.h stays byte-for-byte
// declaration-identical to the red suite's own canonical copy). The sound fix is the ordinary
// C++ one: never let the two conflicting global-scope enums appear in the same TU. This header
// carries NOTHING from either -- only plain ints/strings/vectors -- so `t2132_g5_gpu_parity.cpp`
// (the driver `main()`) can call into both sides without ever including either header itself.
#ifndef T2132_G5_GPU_PARITY_SHARED_H
#define T2132_G5_GPU_PARITY_SHARED_H

#include <cstdint>
#include <string>
#include <vector>

// Gate 1 (masked decode) + Gate 2 (jump-forward) results for one path (CPU or GPU). A path
// whose own setup failed reports setup_ok=false and last_error; the caller (main()) treats that
// as a hard failure, not merely a check miss, since nothing downstream is comparable.
struct G5ParityPathResult {
	bool setup_ok = false;
	std::string last_error;
	int checks = 0;
	int failures = 0;

	// Gate 1: the produced-token stream from num_decode_steps ordinary masked decode calls,
	// starting immediately after the prompt is prefilled under the bound schema.
	std::vector<int32_t> decoded_tokens;

	// Gate 2: fed `forced_chain` (the CPU's own Gate-1 decoded_tokens[0..forced_chain_len) --
	// see t2132_g5_gpu_parity.cpp's own main() for why that is a real schema-legal forced chain
	// by construction) via the schema-content prefill path, against a FRESH sequence re-primed
	// with the same prompt; `forced_consumed` is how many of those tokens were admitted, and
	// `post_forced_token` is the ONE ordinary masked decode step run immediately afterward.
	int32_t forced_consumed = -1;
	int32_t post_forced_token = -1;
};

// Runs the CPU-side gates against `bytes[0, byte_count)` -- implemented in
// t2132_g5_gpu_parity_cpu.cpp, the ONLY TU in this tool that includes sslm_abi.h. Derives its
// own Gate-2 forced chain internally from its own Gate-1 decoded_tokens[0..forced_chain_len) (a
// real schema-legal chain by construction -- every one of those tokens passed the mask), so a
// single call runs both gates. `out_prompt_tokens`, if non-null, is filled with the real
// tokenized prompt (so the GPU side, which has no tokenizer call of its own reason to run twice,
// can reuse the exact same token ids).
G5ParityPathResult RunCpuGates(const uint8_t* bytes, size_t byte_count, const std::string& prompt,
                                const std::string& schema_name, int32_t num_decode_steps,
                                int32_t forced_chain_len, std::vector<int32_t>* out_prompt_tokens);

// The GPU-side twin -- implemented in t2132_g5_gpu_parity_gpu.cpp, the ONLY TU that includes
// gpu_1p0.h/gpu_1p0_g5_bridge.h/gpu_1p0_bench_bridge.h. `prompt_tokens` and `forced_chain` are
// supplied by the caller (main(), after RunCpuGates has already produced them) so both paths
// exercise the IDENTICAL prompt and forced chain.
G5ParityPathResult RunGpuGates(const uint8_t* bytes, size_t byte_count,
                                const std::vector<int32_t>& prompt_tokens,
                                const std::string& schema_name, int32_t num_decode_steps,
                                const std::vector<int32_t>& forced_chain);

// S6 (Claude/Poirot/9bc9ec6-t2132-g5-arc-review.md): Gate 3 -- CPU/GPU `ready_for_logits`
// parity on the PARTIAL-REJECTION path of a forced span. `chain_with_illegal_tail` is Gate 2's
// own real, schema-legal `forced_chain` (every token admitted, by construction) with ONE more
// token appended that is NOT a legal transition from the state that chain leaves the walk in --
// so the schema-content prefill call admits every token up to the last one, then rejects. Before
// this ticket's own S6 fix, the CPU ABI set `ready_for_logits = true` whenever `*consumed > 0`
// (unconditional on the call's own return status), but the GPU bridge returned the rejection
// status without ever reaching its own `ready_for_logits = true` set -- so a caller's immediate
// next decode step would, on GPU only, wrongly re-embed the last ADMITTED token instead of
// finishing its already-computed residual, diverging from the CPU ABI on the exact same input.
// `post_reject_token`, run identically on both paths immediately after the (identically) partial
// rejection, is what this gate compares.
struct G5Gate3Result {
	bool setup_ok = false;
	std::string last_error;
	int checks = 0;
	int failures = 0;
	int32_t forced_consumed = -1;    // tokens admitted before the illegal tail token rejects.
	bool rejected_as_expected = false;  // the call returned the expected per-path rejection status.
	int32_t post_reject_token = -1;  // ONE decode step run immediately after the rejection.
};

G5Gate3Result RunCpuGate3(const uint8_t* bytes, size_t byte_count,
                           const std::vector<int32_t>& prompt_tokens,
                           const std::string& schema_name,
                           const std::vector<int32_t>& chain_with_illegal_tail);

G5Gate3Result RunGpuGate3(const uint8_t* bytes, size_t byte_count,
                           const std::vector<int32_t>& prompt_tokens,
                           const std::string& schema_name,
                           const std::vector<int32_t>& chain_with_illegal_tail);

#endif  // T2132_G5_GPU_PARITY_SHARED_H

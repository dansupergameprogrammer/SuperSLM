// t2132_g5_gpu_parity_gpu.cpp -- G5-5 (T-2132, Brunel): the GPU-side driver for the GPU parity
// tool's two gates. See t2132_g5_gpu_parity_shared.h for why this logic is split into its own
// TU (the sslm_status/SslmGpuStatus global-enum collision). Includes ONLY gpu_1p0.h/
// gpu_1p0_g5_bridge.h/gpu_1p0_bench_bridge.h/model.h -- never sslm_abi.h.
#include "superslm/gpu_1p0.h"
#include "superslm/gpu_1p0_bench_bridge.h"  // SslmGpuSeqHandleLayerIndexForBench (read-only)
#include "superslm/gpu_1p0_g5_bridge.h"
#include "superslm/model.h"
#include "t2132_g5_gpu_parity_shared.h"

namespace {

void Check(G5ParityPathResult& r, bool cond, const char* msg) {
	++r.checks;
	if (!cond) {
		++r.failures;
		if (r.last_error.empty()) r.last_error = msg;
	}
}

// Drains one GPU sequence's own in-flight token to full depth -- possibly across several
// sslm_decode_step_gpu calls, exactly the caller-driven dispatch-budget loop every 1.0 caller
// runs (design Sec7).
bool DriveToFullDepth(SslmGpuContext* ctx, SslmGpuSequenceHandle* seq, uint32_t num_hidden_layers,
                       uint32_t dispatch_budget) {
	uint32_t guard = 0;
	while (*SslmGpuSeqHandleLayerIndexForBench(seq) < num_hidden_layers) {
		if (sslm_decode_step_gpu(ctx, seq, /*adapter_or_null=*/nullptr, dispatch_budget) != SSLM_OK) {
			return false;
		}
		int32_t ready = 0;
		SslmGpuStatus drained = SSLM_OK;
		if (sslm_gpu_ready(ctx, seq, /*block=*/1, &ready, &drained) != SSLM_OK) return false;
		if (drained != SSLM_OK) return false;
		if (++guard > 10000) return false;  // runaway-loop guard, never expected in practice.
	}
	return true;
}

// Embeds and drives every prompt token to full depth WITHOUT masking/argmax -- SSLM_SPAN_PROMPT
// semantics (design Sec5: "never advances the DFA walk, whatever schema is bound") is exactly
// the base substrate, mirroring PrefillWholeTokensImpl's own SSLM_SPAN_PROMPT branch on the CPU
// side (schema_entry stays null there too).
bool PrefillPrompt(SslmGpuContext* ctx, SslmGpuSequenceHandle* seq,
                    const std::vector<int32_t>& prompt_tokens, uint32_t num_hidden_layers,
                    uint32_t dispatch_budget) {
	for (int32_t tok : prompt_tokens) {
		if (sslm_gpu_seq_embed_token(ctx, seq, tok) != SSLM_OK) return false;
		if (!DriveToFullDepth(ctx, seq, num_hidden_layers, dispatch_budget)) return false;
	}
	return true;
}

}  // namespace

G5ParityPathResult RunGpuGates(const uint8_t* bytes, size_t byte_count,
                                const std::vector<int32_t>& prompt_tokens,
                                const std::string& schema_name, int32_t num_decode_steps,
                                const std::vector<int32_t>& forced_chain) {
	G5ParityPathResult r;
	constexpr uint32_t kDispatchBudget = 24;  // one whole layer's worth per call, t2113's own
	                                           // established kDispatchesPerLayer convention.

	superslm::SslmModelView view;
	std::string load_err;
	const superslm::SslmModelStatus load_st = superslm::SslmModel::Load(bytes, byte_count, view, &load_err);
	Check(r, load_st == superslm::SslmModelStatus::Ok, "SslmModel::Load (GPU view) failed");
	if (load_st != superslm::SslmModelStatus::Ok) {
		r.last_error += ": " + load_err;
		return r;
	}

	GpuContextConfig gcfg{};
	GpuResidencyConfig rcfg{};
	SslmGpuContext* ctx = nullptr;
	Check(r, sslm_gpu_context_create(gcfg, &ctx) == SSLM_OK && ctx != nullptr,
	      "sslm_gpu_context_create failed");
	if (!ctx) return r;

	SslmGpuModelHandle* model = nullptr;
	Check(r, sslm_gpu_model_map(ctx, &view, rcfg, &model) == SSLM_OK && model != nullptr,
	      "sslm_gpu_model_map failed");
	if (!model) {
		sslm_gpu_context_destroy(ctx);
		return r;
	}
	Check(r, SslmGpuModelHasSchemasForG5Bridge(model), "GPU model carries no SchemaMasks section");

	const int32_t schema_index = SslmGpuSchemaLookupForG5Bridge(model, schema_name.c_str());
	Check(r, schema_index >= 0, "SslmGpuSchemaLookupForG5Bridge failed");

	const int64_t context_cap = static_cast<int64_t>(view.config.context_cap);
	const uint32_t num_hidden_layers = view.config.num_hidden_layers;

	// ---- Gate 1: prompt-prefill + num_decode_steps ordinary masked decode calls. ----
	SslmGpuSequenceHandle* seq = nullptr;
	Check(r, sslm_gpu_seq_create(ctx, model, context_cap, &seq) == SSLM_OK && seq != nullptr,
	      "sslm_gpu_seq_create (Gate 1) failed");
	if (seq && schema_index >= 0) {
		Check(r, SslmGpuSeqSetSchemaForG5Bridge(ctx, seq, schema_index) == SSLM_OK,
		      "SslmGpuSeqSetSchemaForG5Bridge (Gate 1) failed");
		Check(r, PrefillPrompt(ctx, seq, prompt_tokens, num_hidden_layers, kDispatchBudget),
		      "GPU prompt prefill (Gate 1) failed");

		int32_t current_token = prompt_tokens.empty() ? 0 : prompt_tokens.back();
		for (int32_t step = 0; step < num_decode_steps; ++step) {
			if (sslm_gpu_seq_embed_token(ctx, seq, current_token) != SSLM_OK) {
				Check(r, false, "sslm_gpu_seq_embed_token (Gate 1) failed");
				break;
			}
			if (!DriveToFullDepth(ctx, seq, num_hidden_layers, kDispatchBudget)) {
				Check(r, false, "GPU layer loop (Gate 1) failed");
				break;
			}
			int32_t out_token = -1;
			if (SslmGpuSeqFinishTokenForG5Bridge(ctx, seq, &out_token) != SSLM_OK) {
				Check(r, false, "SslmGpuSeqFinishTokenForG5Bridge (Gate 1) failed");
				break;
			}
			r.decoded_tokens.push_back(out_token);
			current_token = out_token;
		}
		sslm_gpu_seq_release(ctx, seq);
	}

	// ---- Gate 2: jump-forward, driving the SAME forced chain the CPU side's own Gate 1 output
	// already produced (passed in by the caller). ----
	if (!forced_chain.empty()) {
		SslmGpuSequenceHandle* seq2 = nullptr;
		Check(r, sslm_gpu_seq_create(ctx, model, context_cap, &seq2) == SSLM_OK && seq2 != nullptr,
		      "sslm_gpu_seq_create (Gate 2) failed");
		if (seq2) {
			Check(r, SslmGpuSeqSetSchemaForG5Bridge(ctx, seq2, schema_index) == SSLM_OK,
			      "SslmGpuSeqSetSchemaForG5Bridge (Gate 2) failed");
			Check(r, PrefillPrompt(ctx, seq2, prompt_tokens, num_hidden_layers, kDispatchBudget),
			      "GPU prompt re-prefill (Gate 2) failed");

			int32_t forced_consumed = 0;
			const bool prefill_ok = SslmGpuSeqPrefillSchemaContentForG5Bridge(
			    ctx, seq2, forced_chain.data(), static_cast<int32_t>(forced_chain.size()),
			    kDispatchBudget, &forced_consumed) == SSLM_OK &&
			    forced_consumed == static_cast<int32_t>(forced_chain.size());
			Check(r, prefill_ok, "SslmGpuSeqPrefillSchemaContentForG5Bridge failed or short-consumed");
			r.forced_consumed = forced_consumed;

			// NO re-embed here: SslmGpuSeqPrefillSchemaContentForG5Bridge's own last forced
			// token already ran its full layer loop and left `seq2`'s hidden_codes holding that
			// token's complete final residual with layer_index at full depth (mirrors CPU's
			// `ready_for_logits` shortcut exactly -- sslm_decode_step also does NOT re-embed
			// after a prefill, see sslm_abi.cpp's own `seq->ready_for_logits` comment). Finishing
			// THIS state directly is "one ordinary masked step run immediately after the forced
			// chain" -- re-embedding the last forced token again here would silently double the
			// forward pass on the same token and diverge from the CPU oracle's own behavior.
			if (!DriveToFullDepth(ctx, seq2, num_hidden_layers, kDispatchBudget)) {
				Check(r, false, "GPU layer loop (Gate 2, post-chain) failed");
			} else {
				int32_t next_token = -1;
				if (SslmGpuSeqFinishTokenForG5Bridge(ctx, seq2, &next_token) == SSLM_OK) {
					r.post_forced_token = next_token;
				} else {
					Check(r, false, "SslmGpuSeqFinishTokenForG5Bridge (Gate 2, post-chain) failed");
				}
			}
			sslm_gpu_seq_release(ctx, seq2);
		}
	} else {
		Check(r, false, "Gate 2: no forced chain supplied");
	}

	sslm_gpu_model_unmap(ctx, model);
	sslm_gpu_context_destroy(ctx);

	r.setup_ok = true;
	return r;
}

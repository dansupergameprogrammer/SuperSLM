// t2132_g5_gpu_parity_gpu.cpp -- G5-5 (T-2132, Brunel): the GPU-side driver for the GPU parity
// tool's two gates. See t2132_g5_gpu_parity_shared.h for why this logic is split into its own
// TU (the sslm_status/SslmGpuStatus global-enum collision). Includes ONLY gpu_1p0.h/
// gpu_1p0_g5_bridge.h/model.h -- never sslm_abi.h.
//
// G5-5 session 3 fix (Claude/Brunel/t2132-g5-build-2026-08-16.md session 3): this file's own
// PREVIOUS `PrefillPrompt` + Gate-1 loop hand-composed embed/`sslm_decode_step_gpu`/
// `sslm_gpu_ready` directly and re-embedded the prompt's own last token for its first decode
// call, with no GPU-side equivalent of the CPU ABI's `ready_for_logits` shortcut -- committing a
// duplicate KV row that grew `context_length` +1 from decode step 0 onward and eventually
// flipped a produced token 19 real steps later (session 2's original finding). The fix landed in
// the bridge itself (`gpu_1p0_g5_bridge.h`'s new `SslmGpuSeqPrefillPromptForG5Bridge`/
// `SslmGpuSeqDecodeStepForG5Bridge`), not here, per the coordinator's own steer -- the bridge is
// this session's own new surface, and a future consumer reaching for its recommended entry
// points cannot re-trip the bug by construction. This driver now calls those entry points
// exclusively; it no longer hand-composes the embed/drive/finish sequence at all.
#include "superslm/gpu_1p0.h"
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

void Check(G5Gate3Result& r, bool cond, const char* msg) {
	++r.checks;
	if (!cond) {
		++r.failures;
		if (r.last_error.empty()) r.last_error = msg;
	}
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

	// ---- Gate 1: prompt-prefill + num_decode_steps ordinary masked decode calls. ----
	SslmGpuSequenceHandle* seq = nullptr;
	Check(r, sslm_gpu_seq_create(ctx, model, context_cap, &seq) == SSLM_OK && seq != nullptr,
	      "sslm_gpu_seq_create (Gate 1) failed");
	if (seq && schema_index >= 0) {
		Check(r, SslmGpuSeqSetSchemaForG5Bridge(ctx, seq, schema_index) == SSLM_OK,
		      "SslmGpuSeqSetSchemaForG5Bridge (Gate 1) failed");
		Check(r,
		      SslmGpuSeqPrefillPromptForG5Bridge(ctx, seq, prompt_tokens.data(),
		                                          static_cast<int32_t>(prompt_tokens.size()),
		                                          kDispatchBudget) == SSLM_OK,
		      "SslmGpuSeqPrefillPromptForG5Bridge (Gate 1) failed");

		// Step 0's own token_to_embed_if_needed is IGNORED (the prefill call above left
		// ready_for_logits set) -- it is supplied anyway, harmlessly, only so every iteration of
		// this loop has the same shape; every step from 1 onward genuinely uses it.
		int32_t current_token = prompt_tokens.empty() ? 0 : prompt_tokens.back();
		for (int32_t step = 0; step < num_decode_steps; ++step) {
			int32_t out_token = -1;
			if (SslmGpuSeqDecodeStepForG5Bridge(ctx, seq, current_token, kDispatchBudget,
			                                     &out_token) != SSLM_OK) {
				Check(r, false, "SslmGpuSeqDecodeStepForG5Bridge (Gate 1) failed");
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
			Check(r,
			      SslmGpuSeqPrefillPromptForG5Bridge(ctx, seq2, prompt_tokens.data(),
			                                          static_cast<int32_t>(prompt_tokens.size()),
			                                          kDispatchBudget) == SSLM_OK,
			      "SslmGpuSeqPrefillPromptForG5Bridge (Gate 2, re-prefill) failed");

			int32_t forced_consumed = 0;
			const bool prefill_ok = SslmGpuSeqPrefillSchemaContentForG5Bridge(
			    ctx, seq2, forced_chain.data(), static_cast<int32_t>(forced_chain.size()),
			    kDispatchBudget, &forced_consumed) == SSLM_OK &&
			    forced_consumed == static_cast<int32_t>(forced_chain.size());
			Check(r, prefill_ok, "SslmGpuSeqPrefillSchemaContentForG5Bridge failed or short-consumed");
			r.forced_consumed = forced_consumed;

			// token_to_embed_if_needed is IGNORED here too -- the forced-chain prefill above left
			// ready_for_logits set (session 3 fix), so this finishes the chain's own last token's
			// already-computed residual directly: "one ordinary masked step run immediately after
			// the forced chain," with no re-embed.
			int32_t next_token = -1;
			if (SslmGpuSeqDecodeStepForG5Bridge(ctx, seq2, forced_chain.back(), kDispatchBudget,
			                                     &next_token) == SSLM_OK) {
				r.post_forced_token = next_token;
			} else {
				Check(r, false, "SslmGpuSeqDecodeStepForG5Bridge (Gate 2, post-chain) failed");
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

G5Gate3Result RunGpuGate3(const uint8_t* bytes, size_t byte_count,
                           const std::vector<int32_t>& prompt_tokens,
                           const std::string& schema_name,
                           const std::vector<int32_t>& chain_with_illegal_tail) {
	G5Gate3Result r;
	constexpr uint32_t kDispatchBudget = 24;

	superslm::SslmModelView view;
	std::string load_err;
	const superslm::SslmModelStatus load_st = superslm::SslmModel::Load(bytes, byte_count, view, &load_err);
	Check(r, load_st == superslm::SslmModelStatus::Ok, "SslmModel::Load (GPU Gate 3 view) failed");
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

	const int32_t schema_index = SslmGpuSchemaLookupForG5Bridge(model, schema_name.c_str());
	Check(r, schema_index >= 0, "SslmGpuSchemaLookupForG5Bridge failed");
	const int64_t context_cap = static_cast<int64_t>(view.config.context_cap);

	SslmGpuSequenceHandle* seq = nullptr;
	Check(r, sslm_gpu_seq_create(ctx, model, context_cap, &seq) == SSLM_OK && seq != nullptr,
	      "sslm_gpu_seq_create (Gate 3) failed");
	if (seq && schema_index >= 0) {
		Check(r, SslmGpuSeqSetSchemaForG5Bridge(ctx, seq, schema_index) == SSLM_OK,
		      "SslmGpuSeqSetSchemaForG5Bridge (Gate 3) failed");
		Check(r,
		      SslmGpuSeqPrefillPromptForG5Bridge(ctx, seq, prompt_tokens.data(),
		                                          static_cast<int32_t>(prompt_tokens.size()),
		                                          kDispatchBudget) == SSLM_OK,
		      "SslmGpuSeqPrefillPromptForG5Bridge (Gate 3, prompt) failed");

		int32_t forced_consumed = 0;
		const SslmGpuStatus reject_st = SslmGpuSeqPrefillSchemaContentForG5Bridge(
		    ctx, seq, chain_with_illegal_tail.data(),
		    static_cast<int32_t>(chain_with_illegal_tail.size()), kDispatchBudget, &forced_consumed);
		r.forced_consumed = forced_consumed;
		r.rejected_as_expected =
		    (reject_st == SSLM_SEQUENCE_REJECTED) &&
		    (forced_consumed == static_cast<int32_t>(chain_with_illegal_tail.size()) - 1);
		Check(r, r.rejected_as_expected,
		      "SslmGpuSeqPrefillSchemaContentForG5Bridge did not partially reject as expected");

		// S6 fix under test: before it, this bridge returned SSLM_SEQUENCE_REJECTED here without
		// ever setting `ready_for_logits`, so the call below would wrongly re-embed
		// `chain_with_illegal_tail[forced_consumed - 1]` (the last ADMITTED token) instead of
		// finishing its already-computed residual -- diverging from the CPU reference above on
		// the identical input. token_to_embed_if_needed is supplied anyway (harmless if ignored,
		// load-bearing if the bug has regressed).
		const int32_t last_admitted =
		    forced_consumed > 0 ? chain_with_illegal_tail[static_cast<size_t>(forced_consumed) - 1]
		                        : (prompt_tokens.empty() ? 0 : prompt_tokens.back());
		int32_t next_token = -1;
		Check(r,
		      SslmGpuSeqDecodeStepForG5Bridge(ctx, seq, last_admitted, kDispatchBudget, &next_token) ==
		          SSLM_OK,
		      "SslmGpuSeqDecodeStepForG5Bridge (Gate 3, post-partial-reject) failed");
		r.post_reject_token = next_token;
		sslm_gpu_seq_release(ctx, seq);
	}

	sslm_gpu_model_unmap(ctx, model);
	sslm_gpu_context_destroy(ctx);

	r.setup_ok = true;
	return r;
}

// t2132_diag_layer_bisect_gpu.cpp -- DISPOSABLE (T-2132, Brunel). See
// t2132_diag_layer_bisect_shared.h for the why. Includes ONLY gpu_1p0.h/gpu_1p0_bench_bridge.h/
// gpu_1p0_g5_bridge.h/model.h -- never sslm_abi.h (the sslm_status/SslmGpuStatus global-enum
// collision, same reason tools/t2132_g5_gpu_parity_gpu.cpp is split the same way).
//
// PrefillPrompt/DriveToFullDepth below are copied VERBATIM (unmodified algorithm) from
// tools/t2132_g5_gpu_parity_gpu.cpp's own already-verified, already-committed driving logic --
// both are trivial compositions of ONLY public GPU-1.0 entry points plus the existing
// SslmGpuSeqHandleLayerIndexForBench read-only bench accessor, so copying them carries no
// behavioral risk; the only addition is a snapshot capture point inside the bisected step's own
// per-layer loop, never a change to the driving algorithm itself.
#include "superslm/gpu_1p0.h"
#include "superslm/gpu_1p0_bench_bridge.h"  // SslmGpuSeqHandle{LayerIndex,HiddenCodes,HiddenScale,HiddenSize}ForBench
#include "superslm/gpu_1p0_g5_bridge.h"
#include "superslm/model.h"
#include "t2132_diag_layer_bisect_shared.h"

namespace {

constexpr uint32_t kDispatchBudget = 24;  // t2113's own kDispatchesPerLayer -- one whole layer's
                                           // worth per call, verbatim from t2132_g5_gpu_parity_gpu.cpp.

void Check(DiagStepResult& r, bool cond, const char* msg) {
	if (!cond && r.last_error.empty()) r.last_error = msg;
}

// Copied verbatim from tools/t2132_g5_gpu_parity_gpu.cpp's own DriveToFullDepth.
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
		if (++guard > 10000) return false;
	}
	return true;
}

// Copied verbatim from tools/t2132_g5_gpu_parity_gpu.cpp's own PrefillPrompt.
bool PrefillPrompt(SslmGpuContext* ctx, SslmGpuSequenceHandle* seq,
                    const std::vector<int32_t>& prompt_tokens, uint32_t num_hidden_layers,
                    uint32_t dispatch_budget) {
	for (int32_t tok : prompt_tokens) {
		if (sslm_gpu_seq_embed_token(ctx, seq, tok) != SSLM_OK) return false;
		if (!DriveToFullDepth(ctx, seq, num_hidden_layers, dispatch_budget)) return false;
	}
	return true;
}

DiagLayerSnapshot Snapshot(SslmGpuSequenceHandle* seq) {
	DiagLayerSnapshot s;
	s.layer_index_after = *SslmGpuSeqHandleLayerIndexForBench(seq);
	const size_t hs = SslmGpuSeqHandleHiddenSizeForBench(seq);
	const int8_t* codes = SslmGpuSeqHandleHiddenCodesForBench(seq);
	s.hidden_codes.assign(codes, codes + hs);
	const superslm::CarriedScale* scale = SslmGpuSeqHandleHiddenScaleForBench(seq);
	s.scale_m = scale->m;
	s.scale_e = scale->e;
	return s;
}

}  // namespace

DiagStepResult RunGpuDiagStep(const uint8_t* bytes, size_t byte_count,
                               const std::vector<int32_t>& prompt_tokens,
                               const std::string& schema_name, int32_t target_step) {
	DiagStepResult r;

	superslm::SslmModelView view;
	std::string load_err;
	if (superslm::SslmModel::Load(bytes, byte_count, view, &load_err) != superslm::SslmModelStatus::Ok) {
		r.last_error = "SslmModel::Load (GPU view) failed: " + load_err;
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

	SslmGpuSequenceHandle* seq = nullptr;
	Check(r, sslm_gpu_seq_create(ctx, model, context_cap, &seq) == SSLM_OK && seq != nullptr,
	      "sslm_gpu_seq_create failed");
	if (seq && schema_index >= 0 && r.last_error.empty()) {
		Check(r, SslmGpuSeqSetSchemaForG5Bridge(ctx, seq, schema_index) == SSLM_OK,
		      "SslmGpuSeqSetSchemaForG5Bridge failed");
		Check(r, PrefillPrompt(ctx, seq, prompt_tokens, num_hidden_layers, kDispatchBudget),
		      "GPU prompt prefill failed");

		// Raw residual right after PrefillPrompt (the LAST prompt token's own post-layer-loop
		// hidden state) -- BEFORE the Gate-1 loop's own step-0 re-embed below. Mirrors the CPU
		// driver's identical post-prefill capture point exactly (sentinel step -1).
		{
			DiagLayerSnapshot snap;
			snap.layer_index_after = 0xFFFFFFFFu;
			snap.hidden_codes.assign(SslmGpuSeqHandleHiddenCodesForBench(seq),
			                          SslmGpuSeqHandleHiddenCodesForBench(seq) +
			                              SslmGpuSeqHandleHiddenSizeForBench(seq));
			const superslm::CarriedScale* pscale = SslmGpuSeqHandleHiddenScaleForBench(seq);
			snap.scale_m = pscale->m;
			snap.scale_e = pscale->e;
			snap.context_length = *SslmGpuSeqHandleContextLengthForBench(seq);
			r.prefix_step_snapshots.push_back(std::move(snap));
		}

		// ---- Ordinary decode calls for steps 0..target_step-1, verbatim RunGpuGates' own Gate 1
		// loop shape (embed current_token, drive to full depth, finish). ----
		int32_t current_token = prompt_tokens.empty() ? 0 : prompt_tokens.back();
		for (int32_t step = 0; step < target_step && r.last_error.empty(); ++step) {
			Check(r, sslm_gpu_seq_embed_token(ctx, seq, current_token) == SSLM_OK,
			      "sslm_gpu_seq_embed_token (ordinary prefix) failed");
			Check(r, DriveToFullDepth(ctx, seq, num_hidden_layers, kDispatchBudget),
			      "GPU layer loop (ordinary prefix) failed");
			int32_t out_token = -1;
			Check(r, SslmGpuSeqFinishTokenForG5Bridge(ctx, seq, &out_token) == SSLM_OK,
			      "SslmGpuSeqFinishTokenForG5Bridge (ordinary prefix) failed");
			current_token = out_token;

			// Raw pre-final_norm residual right after this ordinary step's own finish call --
			// mirrors the CPU driver's own identical capture point (SslmGpuSeqFinishTokenForG5Bridge
			// only READS seq's hidden_codes for RmsNormSite/LogitsSite, per its own header comment).
			DiagLayerSnapshot snap;
			snap.layer_index_after = static_cast<uint32_t>(step);  // repurposed as step index here.
			snap.hidden_codes.assign(SslmGpuSeqHandleHiddenCodesForBench(seq),
			                          SslmGpuSeqHandleHiddenCodesForBench(seq) +
			                              SslmGpuSeqHandleHiddenSizeForBench(seq));
			const superslm::CarriedScale* pscale = SslmGpuSeqHandleHiddenScaleForBench(seq);
			snap.scale_m = pscale->m;
			snap.scale_e = pscale->e;
			snap.context_length = *SslmGpuSeqHandleContextLengthForBench(seq);
			r.prefix_step_snapshots.push_back(std::move(snap));
		}

		if (r.last_error.empty()) {
			// ---- The bisected step: embed once, then drive one layer (kDispatchBudget ==
			// kDispatchesPerLayer) at a time, snapshotting the REAL GPU sequence handle's
			// hidden state after every single-layer call via the EXISTING, already-public
			// SslmGpuSeqHandleHiddenCodesForBench accessor. ----
			Check(r, sslm_gpu_seq_embed_token(ctx, seq, current_token) == SSLM_OK,
			      "sslm_gpu_seq_embed_token (bisected) failed");
			uint32_t guard = 0;
			while (*SslmGpuSeqHandleLayerIndexForBench(seq) < num_hidden_layers) {
				if (sslm_decode_step_gpu(ctx, seq, /*adapter_or_null=*/nullptr, kDispatchBudget) != SSLM_OK) {
					Check(r, false, "sslm_decode_step_gpu (bisected, single layer) failed");
					break;
				}
				int32_t ready = 0;
				SslmGpuStatus drained = SSLM_OK;
				if (sslm_gpu_ready(ctx, seq, /*block=*/1, &ready, &drained) != SSLM_OK || drained != SSLM_OK) {
					Check(r, false, "sslm_gpu_ready (bisected, single layer) failed");
					break;
				}
				r.layer_snapshots.push_back(Snapshot(seq));
				if (++guard > 10000) {
					Check(r, false, "bisected single-layer loop guard tripped");
					break;
				}
			}
			if (r.last_error.empty()) {
				int32_t out_token = -1;
				Check(r, SslmGpuSeqFinishTokenForG5Bridge(ctx, seq, &out_token) == SSLM_OK,
				      "SslmGpuSeqFinishTokenForG5Bridge (bisected) failed");
				r.produced_token = out_token;
			}
		}

		sslm_gpu_seq_release(ctx, seq);
	}

	sslm_gpu_model_unmap(ctx, model);
	sslm_gpu_context_destroy(ctx);

	r.setup_ok = r.last_error.empty();
	return r;
}

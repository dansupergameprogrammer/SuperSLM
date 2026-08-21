// t2132_diag_layer_bisect_cpu.cpp -- DISPOSABLE (T-2132, Brunel). See
// t2132_diag_layer_bisect_shared.h for the why. Includes ONLY sslm_abi.h (the real production CPU
// ABI) and model.h (read-only, purely to learn hidden_size/num_hidden_layers for sizing/printing
// -- a second, independent SslmModelView over the SAME bytes, exactly the pattern
// tools/t2132_g5_gpu_parity_gpu.cpp's own "GPU view" already establishes) -- never gpu_1p0.h.
#include <cstring>
#include <memory>

#include "superslm/checked_chain_funnel.h"  // superslm::CarriedScale
#include "superslm/model.h"
#include "superslm/sslm_abi.h"
#include "t2132_diag_layer_bisect_shared.h"

// Temporary, disposable accessors appended to src/sslm_abi.cpp for this diagnosis session only
// (see that file's own "T2132 DIAGNOSTIC ONLY" block, to be reverted before exit). Declared here
// rather than in any include/superslm/*.h -- never part of the shipped ABI surface.
extern "C" int8_t* T2132DiagSeqHiddenCodesForBench(sslm_seq seq);
extern "C" superslm::CarriedScale* T2132DiagSeqHiddenScaleForBench(sslm_seq seq);
extern "C" uint32_t* T2132DiagSeqLayerIndexForBench(sslm_seq seq);
extern "C" int64_t* T2132DiagSeqContextLengthForBench(sslm_seq seq);

namespace {

void Check(DiagStepResult& r, bool cond, const char* msg) {
	if (!cond && r.last_error.empty()) r.last_error = msg;
}

}  // namespace

DiagStepResult RunCpuDiagStep(const uint8_t* bytes, size_t byte_count, const std::string& prompt,
                               const std::string& schema_name, int32_t target_step,
                               std::vector<int32_t>* out_prompt_tokens) {
	DiagStepResult r;

	// Second, independent, read-only view purely for hidden_size (not exposed by the opaque CPU
	// ABI's own public surface).
	superslm::SslmModelView view;
	std::string load_err;
	if (superslm::SslmModel::Load(bytes, byte_count, view, &load_err) != superslm::SslmModelStatus::Ok) {
		r.last_error = "SslmModel::Load (hidden_size probe) failed: " + load_err;
		return r;
	}
	const size_t hidden_size = view.config.hidden_size;
	const uint32_t num_hidden_layers = view.config.num_hidden_layers;

	sslm_model model = nullptr;
	sslm_status st = sslm_model_map(bytes, byte_count, &model);
	Check(r, st == SSLM_OK && model != nullptr, "sslm_model_map failed");
	if (!model) return r;

	sslm_schema schema = nullptr;
	st = sslm_schema_lookup(model, schema_name.c_str(), &schema);
	Check(r, st == SSLM_OK && schema != nullptr, "sslm_schema_lookup failed");
	if (!schema) {
		sslm_model_unmap(model);
		return r;
	}

	int32_t token_count = 0;
	st = sslm_tokenize(model, prompt.c_str(), nullptr, &token_count);
	Check(r, st == SSLM_BUFFER_TOO_SMALL && token_count > 0, "sslm_tokenize size query failed");
	std::vector<int32_t> prompt_tokens(static_cast<size_t>(token_count > 0 ? token_count : 0));
	if (token_count > 0) {
		int32_t n = token_count;
		st = sslm_tokenize(model, prompt.c_str(), prompt_tokens.data(), &n);
		Check(r, st == SSLM_OK, "sslm_tokenize failed");
	}
	if (out_prompt_tokens) *out_prompt_tokens = prompt_tokens;

	const uint32_t block_count = 1;
	const size_t kv_block_bytes = sslm_kv_block_size(model);
	const size_t kv_overhead_bytes = sslm_kv_pool_overhead_size(model, block_count);
	const size_t kv_required = kv_block_bytes * block_count + kv_overhead_bytes;
	std::vector<uint8_t> pool_buf_raw(kv_required + (SSLM_ABI_ALIGNMENT_BYTES - 1));
	void* pool_buf_aligned = pool_buf_raw.data();
	size_t pool_buf_space = pool_buf_raw.size();
	std::align(SSLM_ABI_ALIGNMENT_BYTES, kv_required, pool_buf_aligned, pool_buf_space);
	sslm_kv_pool pool = nullptr;
	st = sslm_kv_pool_create(model, pool_buf_aligned, kv_required, block_count, &pool);
	Check(r, st == SSLM_OK && pool != nullptr, "sslm_kv_pool_create failed");

	sslm_config config{};
	config.max_batch = 1;
	config.max_chunk_budget = token_count > 0 ? token_count : 1;
	config.max_layer_budget = static_cast<int32_t>(num_hidden_layers);
	const size_t ws_bytes = sslm_workspace_size(model, &config);
	std::vector<uint8_t> ws_buf_raw(ws_bytes + (SSLM_ABI_ALIGNMENT_BYTES - 1));
	void* ws_aligned = ws_buf_raw.data();
	size_t ws_space = ws_buf_raw.size();
	std::align(SSLM_ABI_ALIGNMENT_BYTES, ws_bytes, ws_aligned, ws_space);
	sslm_workspace ws = nullptr;
	st = sslm_workspace_create(model, &config, ws_aligned, ws_bytes, &ws);
	Check(r, st == SSLM_OK && ws != nullptr, "sslm_workspace_create failed");

	if (!pool || !ws) {
		if (ws) sslm_workspace_destroy(ws);
		if (pool) sslm_kv_pool_destroy(pool);
		sslm_model_unmap(model);
		return r;
	}

	sslm_seq seq = nullptr;
	st = sslm_seq_create(model, &pool, &seq);
	Check(r, st == SSLM_OK && seq != nullptr, "sslm_seq_create failed");
	if (seq) {
		st = sslm_seq_set_schema(seq, schema);
		Check(r, st == SSLM_OK, "sslm_seq_set_schema failed");
		int32_t consumed = 0;
		st = sslm_prefill(model, seq, prompt_tokens.data(), token_count, token_count > 0 ? token_count : 1,
		                   SSLM_SPAN_PROMPT, ws, &consumed);
		Check(r, st == SSLM_OK && consumed == token_count, "sslm_prefill (prompt) failed");

		// Raw residual right after prefill (the LAST prompt token's own post-layer-loop hidden
		// state, ready_for_logits) -- BEFORE any decode step runs. step index -1 marks this.
		{
			DiagLayerSnapshot snap;
			snap.layer_index_after = 0xFFFFFFFFu;  // sentinel: "post-prefill, pre-decode" (step -1).
			const int8_t* codes = T2132DiagSeqHiddenCodesForBench(seq);
			snap.hidden_codes.assign(codes, codes + hidden_size);
			const superslm::CarriedScale* scale = T2132DiagSeqHiddenScaleForBench(seq);
			snap.scale_m = scale->m;
			snap.scale_e = scale->e;
			snap.context_length = *T2132DiagSeqContextLengthForBench(seq);
			r.prefix_step_snapshots.push_back(std::move(snap));
		}

		// ---- Ordinary full-layer_budget decode calls for steps 0..target_step-1 -- the REAL,
		// unmodified production path, exactly RunCpuGates' own loop (t2132_g5_gpu_parity_cpu.cpp)
		// -- reaching the state right before the bisected step. ----
		sslm_decode_params full_params{};
		// T-2199 Phase D review addendum (D-SLM3797, Dan): struct_size is the FIRST
		// new field the library now validates -- an unrecognized size is a loud rejection.
		full_params.struct_size = sizeof(full_params);
		full_params.layer_budget = static_cast<int32_t>(num_hidden_layers);
		for (int32_t step = 0; step < target_step; ++step) {
			int32_t out_token = -1;
			sslm_seq seqs[1] = {seq};
			int32_t guard = 0;
			while (out_token < 0) {
				st = sslm_decode_step(model, seqs, 1, &full_params, ws, &out_token);
				if (st != SSLM_OK || ++guard > 10000) break;
			}
			Check(r, st == SSLM_OK, "sslm_decode_step (ordinary prefix) failed");
			if (st != SSLM_OK) break;

			// Raw pre-final_norm residual right after this ordinary step's own decode call --
			// RmsNormSite/LogitsSite only READ seq->state.hidden_codes into a separate final_codes
			// buffer inside sslm_decode_step's own finish block, never overwrite it, so this is
			// genuinely this step's own layer-loop output, unperturbed by the finish stage.
			DiagLayerSnapshot snap;
			snap.layer_index_after = static_cast<uint32_t>(step);  // repurposed as step index here.
			const int8_t* codes = T2132DiagSeqHiddenCodesForBench(seq);
			snap.hidden_codes.assign(codes, codes + hidden_size);
			const superslm::CarriedScale* scale = T2132DiagSeqHiddenScaleForBench(seq);
			snap.scale_m = scale->m;
			snap.scale_e = scale->e;
			snap.context_length = *T2132DiagSeqContextLengthForBench(seq);
			r.prefix_step_snapshots.push_back(std::move(snap));
		}

		if (r.last_error.empty()) {
			// ---- The bisected call: params.layer_budget = 1, S3.7's own resumable-layer_budget
			// contract lets each call advance exactly one layer and resume next call at
			// seq->state.layer_index -- so num_hidden_layers single-layer calls reach the SAME
			// final state one full-budget call would. Snapshot the REAL production sslm_seq's
			// hidden state after EVERY call via the temporary bench accessors. ----
			sslm_decode_params one_layer{};
			// T-2199 Phase D review addendum (D-SLM3797, Dan): struct_size is the FIRST
			// new field the library now validates -- an unrecognized size is a loud rejection.
			one_layer.struct_size = sizeof(one_layer);
			one_layer.layer_budget = 1;
			for (uint32_t l = 0; l < num_hidden_layers; ++l) {
				int32_t out_token = -1;
				sslm_seq seqs[1] = {seq};
				st = sslm_decode_step(model, seqs, 1, &one_layer, ws, &out_token);
				Check(r, st == SSLM_OK, "sslm_decode_step (bisected, single layer) failed");
				if (st != SSLM_OK) break;

				DiagLayerSnapshot snap;
				snap.layer_index_after = *T2132DiagSeqLayerIndexForBench(seq);
				const int8_t* codes = T2132DiagSeqHiddenCodesForBench(seq);
				snap.hidden_codes.assign(codes, codes + hidden_size);
				const superslm::CarriedScale* scale = T2132DiagSeqHiddenScaleForBench(seq);
				snap.scale_m = scale->m;
				snap.scale_e = scale->e;
				r.layer_snapshots.push_back(std::move(snap));

				if (out_token >= 0) r.produced_token = out_token;
			}
		}

		sslm_seq_release(seq);
	}

	sslm_workspace_destroy(ws);
	sslm_kv_pool_destroy(pool);
	sslm_model_unmap(model);

	r.setup_ok = r.last_error.empty();
	return r;
}

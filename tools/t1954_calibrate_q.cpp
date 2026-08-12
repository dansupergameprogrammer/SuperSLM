// t1954_calibrate_q.cpp -- T-1954 (Brunel spike, disposable, never merges;
// T-1822 design Sec32 "Fused Q"). Calibration driver: loads the real,
// unmodified Qwen2.5-1.5B-Instruct artifact through the real production
// entry point (SslmModel::Load, the same marshaling adapter sslm_generate.cpp
// uses), runs the ordinary decode loop over a handful of real prompts with
// SSLM_OPTION_G_FUSED_Q_LANDING and SSLM_OPTION_G_FUSED_Q_LANDING_CALIBRATE
// both set, and dumps forward_sites.h's own OptionGFusedQCalibrationDump()
// per layer -- the observed post-RoPE Q peak (Sec32.2's own decided
// calibration policy: "canonical_scale of the observed post-RoPE peak") and
// the (m_a, e_a) in force when it was set. This tool computes NO landing
// constants itself; it only observes. A separate offline script
// (tools/t1954_derive_q_landing.py) turns this dump into q_landing_m_out/
// e_out/e_t/r_t, transplanting `_derive_composition_constants`'s own K/V
// formula (tests/reference/superslm_spike/pipeline.py) onto Q's own observed
// peak instead of K's calibrated output_scale.
//
// Usage: t1954_calibrate_q <model.sslm> <tokenizer.sslm>
// (both env vars above must already be set in the calling process before
// this binary starts -- OptionGFusedQLandingEnabled/CalibrateEnabled cache
// on first read.)

#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

#include "superslm/artifact.h"
#include "superslm/forward_sites.h"
#include "superslm/model.h"
#include "superslm/tokenizer.h"
#include "sslm_marshal.h"

using namespace superslm;
using superslm_marshal::LayerBacking;
using superslm_marshal::MarshalLayer;
using superslm_marshal::PreflightScanWscFolds;
using superslm_marshal::ReadCarriedScale;
using superslm_marshal::ReadFile;
using superslm_marshal::WidenGainToInt32;

namespace {

// A small, real-language calibration set -- not T-1777's own 239-document
// corpus (this is a spike-tier calibration pass, not a graded capture; NO
// graded capture runs this ticket per its own brief), but more than one
// prompt so the observed peak is not an artifact of a single sequence's own
// quirks. Chat-templated the same way sslm_generate.cpp's own callers
// template a prompt (T-1902's own 32-prompt population uses the identical
// wrapper) -- reused verbatim here rather than re-derived.
const char* kCalibrationPrompts[] = {
    "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a "
    "helpful assistant.<|im_end|>\n<|im_start|>user\nWhat is 12 + 15?<|im_end|>\n"
    "<|im_start|>assistant\n",
    "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a "
    "helpful assistant.<|im_end|>\n<|im_start|>user\nName three colors.<|im_end|>\n"
    "<|im_start|>assistant\n",
    "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a "
    "helpful assistant.<|im_end|>\n<|im_start|>user\nWrite one sentence about the "
    "ocean.<|im_end|>\n<|im_start|>assistant\n",
    "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a "
    "helpful assistant.<|im_end|>\n<|im_start|>user\nTranslate 'hello' to "
    "French.<|im_end|>\n<|im_start|>assistant\n",
    "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a "
    "helpful assistant.<|im_end|>\n<|im_start|>user\nWhat year did the Berlin Wall "
    "fall?<|im_end|>\n<|im_start|>assistant\n",
};

}  // namespace

int main(int argc, char** argv) {
	if (argc < 3) {
		std::fprintf(stderr, "usage: %s <model.sslm> <tokenizer.sslm>\n", argv[0]);
		return 2;
	}
	const std::string model_path = argv[1];
	const std::string tokenizer_path = argv[2];

	std::vector<uint8_t> tok_bytes;
	if (!ReadFile(tokenizer_path.c_str(), tok_bytes)) {
		std::fprintf(stderr, "FAILED stage=tokenizer_file_read path=\"%s\"\n", tokenizer_path.c_str());
		return 1;
	}
	SslmArtifact tok_artifact;
	SslmError tok_open_err;
	if (SslmArtifact::OpenFromMemory(tok_bytes.data(), tok_bytes.size(), tok_artifact,
	                                  &tok_open_err) != SslmStatus::Ok) {
		std::fprintf(stderr, "FAILED stage=tokenizer_artifact_open status=%s\n",
		             SslmStatusName(tok_open_err.code));
		return 1;
	}
	TokenizerView tokenizer;
	std::string tok_err;
	if (!TokenizerView::Open(tok_artifact, tokenizer, &tok_err)) {
		std::fprintf(stderr, "FAILED stage=tokenizer_view_open diagnostic=\"%s\"\n", tok_err.c_str());
		return 1;
	}

	std::vector<uint8_t> model_bytes;
	if (!ReadFile(model_path.c_str(), model_bytes)) {
		std::fprintf(stderr, "FAILED stage=model_file_read path=\"%s\"\n", model_path.c_str());
		return 1;
	}
	SslmModelView model_view;
	std::string model_err;
	const SslmModelStatus load_status =
	    SslmModel::Load(model_bytes.data(), model_bytes.size(), model_view, &model_err);
	if (load_status != SslmModelStatus::Ok) {
		std::fprintf(stderr, "FAILED stage=model_load status=%s diagnostic=\"%s\"\n",
		             SslmModelStatusName(load_status), model_err.c_str());
		return 1;
	}
	std::printf("model loaded: hidden_size=%u layers=%u heads=%u/%u head_dim=%u\n",
	            model_view.config.hidden_size, model_view.config.num_hidden_layers,
	            model_view.config.num_attention_heads, model_view.config.num_key_value_heads,
	            model_view.config.head_dim);

	const uint32_t num_heads = model_view.config.num_attention_heads;
	const uint32_t num_kv_heads = model_view.config.num_key_value_heads;
	const uint32_t num_hidden_layers = model_view.config.num_hidden_layers;
	const size_t hidden_size = model_view.config.hidden_size;

	PreflightScanWscFolds(model_view);

	std::vector<LayerBacking> backings(num_hidden_layers);
	std::vector<LayerWeights> layers(num_hidden_layers);
	for (uint32_t l = 0; l < num_hidden_layers; ++l) {
		std::string marshal_err;
		if (!MarshalLayer(model_view, l, num_heads, num_kv_heads, backings[l], layers[l],
		                   &marshal_err)) {
			std::fprintf(stderr, "FAILED stage=layer_weights_marshal layer=%u diagnostic=\"%s\"\n", l,
			             marshal_err.c_str());
			return 1;
		}
	}

	const SslmTensorView* embed_w = model_view.weights.Tensor("embed");
	const SslmTensorView* final_gain_w = model_view.weights.Tensor("final_norm.gain");
	if (!embed_w || !final_gain_w) {
		std::fprintf(stderr, "FAILED stage=head_marshal: missing embed or final_norm.gain\n");
		return 1;
	}
	std::vector<int32_t> final_norm_gain = WidenGainToInt32(*final_gain_w);
	bool ok = true;
	CarriedScale embed_site_constant =
	    ReadCarriedScale(model_view.composition_constants, "embed", &ok);
	CarriedScale final_norm_site_constant =
	    ReadCarriedScale(model_view.composition_constants, "final_norm", &ok);
	if (!ok) {
		std::fprintf(stderr, "FAILED stage=head_marshal: missing site constants\n");
		return 1;
	}
	const int8_t* embed_weights = reinterpret_cast<const int8_t*>(embed_w->data);
	const int8_t* head_weights = model_view.config.tie_word_embeddings
	                                  ? embed_weights
	                                  : reinterpret_cast<const int8_t*>(
	                                        model_view.weights.Tensor("lm_head")->data);

	const int64_t context_cap = static_cast<int64_t>(model_view.config.context_cap);
	const size_t kv_bytes = static_cast<size_t>(num_hidden_layers) *
	                        static_cast<size_t>(context_cap) * num_kv_heads *
	                        model_view.config.head_dim * 2;

	// Reset ONCE, before any prompt -- the peak accumulates across the WHOLE
	// calibration set (OptionGFusedQCalibrationDump reports the running max,
	// never reset between prompts), matching Sec32.2's own "the observed
	// post-RoPE peak" language (a peak over the calibration population, not
	// per-prompt).
	OptionGFusedQCalibrationReset(num_hidden_layers);

	int prompts_run = 0;
	for (const char* raw_prompt : kCalibrationPrompts) {
		const std::vector<int32_t> prompt_tokens = tokenizer.Encode(raw_prompt);
		if (prompt_tokens.empty()) {
			std::fprintf(stderr, "WARN: prompt encoded to zero tokens, skipped\n");
			continue;
		}
		std::vector<uint8_t> workspace(kv_bytes);
		std::vector<int8_t> hidden_codes(hidden_size);
		SequenceLayerState seq;
		seq.hidden_codes = hidden_codes.data();
		std::vector<int32_t> out_tokens(8);
		std::vector<int32_t> out_logit_rows(8 * static_cast<size_t>(model_view.config.vocab_size));
		size_t out_tokens_produced = 0;
		SslmDecodeStopReason stop_reason = SslmDecodeStopReason::MaxTokensReached;
		const std::vector<int32_t> stop_ids = {151645, 151643};

		const SslmForwardStatus decode_status = RunGreedyDecodeLoop(
		    seq, layers.data(), num_hidden_layers, hidden_size, model_view.config.head_dim,
		    num_kv_heads, model_view.config.intermediate_size, context_cap, model_view.rope_tables,
		    prompt_tokens.data(), prompt_tokens.size(), embed_weights, embed_site_constant,
		    final_norm_gain.data(), final_norm_site_constant, head_weights,
		    static_cast<int32_t>(model_view.config.vocab_size), stop_ids.data(), stop_ids.size(), 8,
		    workspace.data(), workspace.size(), out_tokens.data(), out_logit_rows.data(),
		    out_tokens.size(), &out_tokens_produced, &stop_reason, model_view.config.kv_precision,
		    model_view.option_g_fused_k_landing);

		if (decode_status != SslmForwardStatus::Ok) {
			std::fprintf(stderr,
			             "FAILED stage=calibration_decode prompt=%d status=%s -- calibration mode "
			             "should never refuse (landing is skipped entirely); this is a real defect\n",
			             prompts_run, SslmForwardStatusName(decode_status));
			return 1;
		}
		++prompts_run;
	}
	std::printf("calibration_prompts_run: %d\n", prompts_run);

	const std::vector<OptionGFusedQCalibrationSample> samples = OptionGFusedQCalibrationDump();
	// q_site_constant (canonical_scale(s_ref/127), already loaded from the
	// real artifact by MarshalLayer) is dumped alongside the calibration
	// sample so the offline derivation script can recover `s_ref` without a
	// second artifact-parsing pass.
	std::printf(
	    "layer,peak_abs_branch_code,m_a_at_peak,e_a_at_peak,q_site_constant_m,q_site_constant_e\n");
	for (uint32_t l = 0; l < samples.size(); ++l) {
		std::printf("%u,%lld,%lld,%lld,%lld,%lld\n", l,
		            static_cast<long long>(samples[l].peak_abs_branch_code),
		            static_cast<long long>(samples[l].m_a_at_peak),
		            static_cast<long long>(samples[l].e_a_at_peak),
		            static_cast<long long>(layers[l].q_site_constant.m),
		            static_cast<long long>(layers[l].q_site_constant.e));
	}
	return 0;
}

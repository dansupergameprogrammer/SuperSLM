// t1966_construction_anchor.cpp -- T-1966 (Brunel micro-round, disposable,
// never merges; D-SLM2787/D-SLM2788). The construction-anchor gate for the
// dynamic-fused Q path: does `q_rot`, dequantized, reproduce the exact
// wide-integer post-RoPE target to within the single quantization step's
// bound, on real data?
//
// Method: loads the real, unmodified artifact through the real production
// loader; runs one real decode with SSLM_OPTION_G_FUSED_Q_LANDING=dynamic
// AND SSLM_OPTION_G_FUSED_Q_ANCHOR_CAPTURE=1 both set (both env vars must
// already be set in the calling process before this binary starts, same
// convention as every other T-1954/T-1959/T-1966 tool); reads back
// OptionGDynamicQAnchorDump()'s own per-layer capture (the EXACT wide row
// RequantChainChecked narrowed and the EXACT codes it produced, captured
// inside the real call site -- never recomputed independently, so this
// check compares the certified construction's own output against itself,
// not against a second implementation that could itself diverge).
//
// The bound: for each layer, `d_prime = max_i |wide_rotated[i]|` (computable
// directly from the captured row -- this IS the quantity RequantChainChecked's
// own MaxAbsReduceWide computes internally, per Steps 1-2, source-verified
// in this ticket's own build log Sec18). The implied quantization step in
// raw wide-row units is `d_prime / 127` (an ordinary max-abs int8 scale).
// Reconstructing `codes[i] * d_prime / 127` and comparing against
// `wide_rotated[i]` should differ by at most ~1 step (rounding) for every
// element -- this is NOT a re-derivation of RequantTokenCodeWide's own exact
// formula (already covered by that function's own pre-existing test suite);
// it is confirmation that feeding a ROTATED row into the SAME machinery
// preserves the SAME quantization-fidelity property on real data, which is
// what this ticket's gate asks for.
//
// Usage: t1966_construction_anchor <model.sslm> <tokenizer.sslm>
// (SSLM_OPTION_G_FUSED_Q_LANDING=dynamic and SSLM_OPTION_G_FUSED_Q_ANCHOR_CAPTURE=1
// must already be set in the calling process.)

#include <algorithm>
#include <cmath>
#include <cstdio>
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
	std::printf("model loaded (real, unmodified artifact): hidden_size=%u layers=%u\n",
	            model_view.config.hidden_size, model_view.config.num_hidden_layers);

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
		// Dynamic-fused needs NO injected constants -- q_landing_m_out/e_out/
		// e_t/r_t stay at their default-member-initialized 0, genuinely
		// unread on this path (D-SLM2788's own "no static fields" claim,
		// confirmed here by simply never touching them).
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
	std::vector<uint8_t> workspace(kv_bytes);
	std::vector<int8_t> hidden_codes(hidden_size);
	SequenceLayerState seq;
	seq.hidden_codes = hidden_codes.data();

	OptionGDynamicQAnchorReset(num_hidden_layers);

	const std::string prompt = "What is 12 + 15?";
	const std::vector<int32_t> prompt_tokens = tokenizer.Encode(prompt);
	if (prompt_tokens.empty()) {
		std::fprintf(stderr, "FAILED stage=tokenize: zero tokens\n");
		return 1;
	}
	std::vector<int32_t> out_tokens(16);
	std::vector<int32_t> out_logit_rows(16 * static_cast<size_t>(model_view.config.vocab_size));
	size_t out_tokens_produced = 0;
	SslmDecodeStopReason stop_reason = SslmDecodeStopReason::MaxTokensReached;
	const std::vector<int32_t> stop_ids = {151645, 151643};

	const SslmForwardStatus decode_status = RunGreedyDecodeLoop(
	    seq, layers.data(), num_hidden_layers, hidden_size, model_view.config.head_dim,
	    num_kv_heads, model_view.config.intermediate_size, context_cap, model_view.rope_tables,
	    prompt_tokens.data(), prompt_tokens.size(), embed_weights, embed_site_constant,
	    final_norm_gain.data(), final_norm_site_constant, head_weights,
	    static_cast<int32_t>(model_view.config.vocab_size), stop_ids.data(), stop_ids.size(), 16,
	    workspace.data(), workspace.size(), out_tokens.data(), out_logit_rows.data(),
	    out_tokens.size(), &out_tokens_produced, &stop_reason, model_view.config.kv_precision,
	    /*option_g_fused_k_landing=*/false);

	if (decode_status != SslmForwardStatus::Ok) {
		std::fprintf(stderr, "FAILED stage=decode status=%s\n", SslmForwardStatusName(decode_status));
		return 1;
	}
	std::printf("decode completed: status=Ok, %zu tokens produced\n", out_tokens_produced);

	const std::vector<OptionGDynamicQAnchorSample> samples = OptionGDynamicQAnchorDump();

	uint32_t layers_checked = 0;
	double worst_layer_max_steps = 0.0;
	double overall_max_steps = 0.0;
	double overall_sum_steps = 0.0;
	uint64_t overall_elements = 0;
	uint32_t worst_layer_index = 0;

	for (uint32_t l = 0; l < samples.size(); ++l) {
		const OptionGDynamicQAnchorSample& s = samples[l];
		if (s.wide_rotated.empty() || s.codes.empty()) {
			continue;  // this layer's own forward step was never reached this run (e.g. beyond
			           // num_hidden_layers on a short decode) -- skip, not a failure.
		}
		if (s.wide_rotated.size() != hidden_size || s.codes.size() != hidden_size) {
			std::fprintf(stderr, "FAILED: layer=%u captured size mismatch (wide=%zu codes=%zu, "
			                     "expected %zu)\n",
			             l, s.wide_rotated.size(), s.codes.size(), hidden_size);
			return 1;
		}
		int64_t d_prime = 0;
		for (int64_t v : s.wide_rotated) {
			const int64_t av = v < 0 ? -v : v;
			if (av > d_prime) d_prime = av;
		}
		if (d_prime == 0) {
			std::fprintf(stderr, "FAILED: layer=%u wide_rotated row is all-zero -- degenerate, "
			                     "cannot check reconstruction fidelity\n",
			             l);
			return 1;
		}
		const double step = static_cast<double>(d_prime) / 127.0;
		double layer_max_steps = 0.0;
		for (size_t i = 0; i < hidden_size; ++i) {
			const double reconstructed = static_cast<double>(s.codes[i]) * step;
			const double target = static_cast<double>(s.wide_rotated[i]);
			const double err_steps = std::fabs(target - reconstructed) / step;
			layer_max_steps = std::max(layer_max_steps, err_steps);
			overall_max_steps = std::max(overall_max_steps, err_steps);
			overall_sum_steps += err_steps;
			++overall_elements;
		}
		++layers_checked;
		if (layer_max_steps > worst_layer_max_steps) {
			worst_layer_max_steps = layer_max_steps;
			worst_layer_index = l;
		}
		std::printf("layer=%u d_prime=%lld step=%.6f max_error_steps=%.4f\n", l,
		            static_cast<long long>(d_prime), step, layer_max_steps);
	}

	if (layers_checked == 0) {
		std::fprintf(stderr, "FAILED: zero layers captured -- anchor capture did not run (env var "
		                     "not set, or the dynamic-fused branch was never entered)\n");
		return 1;
	}

	const double mean_steps = overall_sum_steps / static_cast<double>(overall_elements);
	std::printf("layers_checked: %u/%u\n", layers_checked, num_hidden_layers);
	std::printf("elements_checked: %llu\n", static_cast<unsigned long long>(overall_elements));
	std::printf("overall_max_error_steps: %.4f (worst layer=%u)\n", overall_max_steps,
	            worst_layer_index);
	std::printf("overall_mean_error_steps: %.6f\n", mean_steps);

	// Bound: 1.5 quantization steps, per element, across every checked layer
	// -- generous enough to absorb the difference between this probe's own
	// simple max-abs/127 reconstruction and RequantChainChecked's own
	// power-of-2 NormalizeScale+reciprocal formula (a different but
	// closely-related scale derivation), while still catching any GROSS
	// divergence (a construction defect would produce errors of many steps,
	// not a fraction of one).
	constexpr double kMaxAllowedSteps = 1.5;
	if (overall_max_steps > kMaxAllowedSteps) {
		std::fprintf(stderr,
		             "FAILED: overall_max_error_steps=%.4f exceeds the %.1f-step bound\n",
		             overall_max_steps, kMaxAllowedSteps);
		return 1;
	}
	std::printf("CONSTRUCTION ANCHOR PASSED: reconstruction error stays within %.1f quantization "
	            "steps across %u layer(s), %llu elements.\n",
	            kMaxAllowedSteps, layers_checked, static_cast<unsigned long long>(overall_elements));
	return 0;
}

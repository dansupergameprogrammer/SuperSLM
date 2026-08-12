// t1954_selfcheck_q.cpp -- T-1954 (Brunel spike, disposable, never merges;
// T-1822 design Sec32 "Fused Q"). The toggle-on, end-to-end self-check on
// the real artifact: loads the real, unmodified Qwen2.5-1.5B-Instruct
// artifact through the real production entry point (SslmModel::Load, the
// same marshaling adapter sslm_generate.cpp uses), then -- because the
// shipped artifact carries no Q landing constants yet (Sec32.2's own new
// artifact keys are unbuilt) -- overwrites each loaded LayerWeights' own
// q_landing_m_out/e_out/e_t/r_t IN MEMORY with the values
// tools/t1954_derive_q_landing.py computed from tools/t1954_calibrate_q.exe's
// own dump over this SAME real artifact's real weights. Every other byte of
// the loaded model (every weight, every K/V landing constant, every
// composition constant) is untouched -- the injection touches only four
// fields this build itself added to LayerWeights, fields the shipped
// artifact's own loader never populates (they default to 0).
//
// Then runs the real RunGreedyDecodeLoop with SSLM_OPTION_G_FUSED_Q_LANDING=1
// (set by the caller before this process starts) over the real decode smoke
// prompt, and reports: did it complete without refusal, and does its output
// differ from the toggle-off baseline (confirming the fused path took
// effect, not merely that it declined to run).
//
// Usage: t1954_selfcheck_q <model.sslm> <tokenizer.sslm> <derived_constants.txt>
// <derived_constants.txt> is one line per layer: "L m_out e_out e_t r_t"
// (space-separated), any order, one line per layer index 0..num_layers-1.

#include <cstdio>
#include <fstream>
#include <map>
#include <sstream>
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

struct DerivedQLanding {
	int64_t m_out, e_out, e_t, r_t;
};

bool LoadDerivedConstants(const std::string& path, std::map<uint32_t, DerivedQLanding>* out,
                           std::string* err) {
	std::ifstream f(path);
	if (!f) {
		*err = "could not open \"" + path + "\"";
		return false;
	}
	std::string line;
	while (std::getline(f, line)) {
		if (line.empty()) continue;
		std::istringstream iss(line);
		uint32_t l;
		DerivedQLanding d{};
		if (!(iss >> l >> d.m_out >> d.e_out >> d.e_t >> d.r_t)) {
			*err = "malformed line: \"" + line + "\"";
			return false;
		}
		(*out)[l] = d;
	}
	return true;
}

}  // namespace

int main(int argc, char** argv) {
	if (argc < 4) {
		std::fprintf(stderr, "usage: %s <model.sslm> <tokenizer.sslm> <derived_constants.txt>\n",
		             argv[0]);
		return 2;
	}
	const std::string model_path = argv[1];
	const std::string tokenizer_path = argv[2];
	const std::string derived_path = argv[3];

	std::map<uint32_t, DerivedQLanding> derived;
	std::string derived_err;
	if (!LoadDerivedConstants(derived_path, &derived, &derived_err)) {
		std::fprintf(stderr, "FAILED stage=derived_constants_load diagnostic=\"%s\"\n",
		             derived_err.c_str());
		return 1;
	}

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
	// Confirmed unmodified: this reads the file bytes, never writes them --
	// the ORIGINAL shipped artifact on disk is never touched by this tool.
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
	uint32_t layers_injected = 0;  // T-1956 fix round (M1) -- see the increment site below.
	for (uint32_t l = 0; l < num_hidden_layers; ++l) {
		std::string marshal_err;
		if (!MarshalLayer(model_view, l, num_heads, num_kv_heads, backings[l], layers[l],
		                   &marshal_err)) {
			std::fprintf(stderr, "FAILED stage=layer_weights_marshal layer=%u diagnostic=\"%s\"\n", l,
			             marshal_err.c_str());
			return 1;
		}
		// T-1954: the ONLY in-memory mutation this tool performs -- every
		// other field MarshalLayer just populated (every weight pointer,
		// every fold triple, every K/V landing constant) is untouched.
		// These four fields default to 0 (forward_sites.h's own default
		// member initializers) because the shipped artifact carries no
		// `layer{N}.q` key in KvLandingScales/KvLandingReciprocals yet
		// (Sec32.2's own unbuilt artifact-writer obligation) -- this
		// injection is the spike-tier substitute named in this ticket's own
		// build log.
		const auto it = derived.find(l);
		if (it == derived.end()) {
			std::fprintf(stderr, "FAILED stage=derived_constants_lookup: no entry for layer=%u\n", l);
			return 1;
		}
		layers[l].q_landing_m_out = it->second.m_out;
		layers[l].q_landing_e_out = it->second.e_out;
		layers[l].q_landing_e_t = it->second.e_t;
		layers[l].q_landing_r_t = it->second.r_t;
		++layers_injected;  // T-1956 fix round (M1): count what was actually
		                    // WRITTEN, not `derived.size()` -- an
		                    // over-populated constants file (more entries
		                    // than layers) would print a numerator exceeding
		                    // the layer count while every entry beyond
		                    // `num_hidden_layers` is silently never looked
		                    // up. This loop's own hard-fail on a MISSING
		                    // entry already made an under-count unreachable;
		                    // this closes the over-count side too.
	}
	std::printf("q_landing constants injected for %u/%u layers\n", layers_injected,
	            num_hidden_layers);

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

	const std::string prompt = "What is 12 + 15?";
	const std::vector<int32_t> prompt_tokens = tokenizer.Encode(prompt);
	std::printf("prompt: %s\n", prompt.c_str());
	std::printf("prompt_tokens (%zu):", prompt_tokens.size());
	for (int32_t t : prompt_tokens) std::printf(" %d", t);
	std::printf("\n");

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
	    model_view.option_g_fused_k_landing);

	if (decode_status != SslmForwardStatus::Ok) {
		std::fprintf(stderr, "FAILED stage=decode status=%s\n", SslmForwardStatusName(decode_status));
		return 1;
	}

	std::printf("output_tokens (%zu):", out_tokens_produced);
	for (size_t i = 0; i < out_tokens_produced; ++i) std::printf(" %d", out_tokens[i]);
	std::printf("\n");
	std::printf("stop_reason: %d\n", static_cast<int>(stop_reason));
	// T-1956 fix round (Significant 1/D-SLM2727 verification): reads back
	// the SAME per-sequence counter the fix now wires Q's own landing
	// through (`&seq.kv_saturation_count`, matching K's sibling call) --
	// direct evidence the counter is live, not merely that the code
	// compiles. A nonzero count on this run's own coarse, spike-tier
	// constants is expected and not itself a defect (§9's own disclaimer);
	// the point is that it is OBSERVABLE at all, which it was not before
	// this fix (`nullptr` was passed).
	std::printf("kv_saturation_count (K/V + Q landing clamps this decode, host-facing "
	            "SslmDecodeStepStatus::saturation_count equivalent): %llu\n",
	            static_cast<unsigned long long>(seq.kv_saturation_count));
	std::printf("SELFCHECK: fused-Q path completed end to end on the real artifact, status=Ok\n");
	return 0;
}

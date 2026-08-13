// t1960_decode_q.cpp -- T-1960 decode-leg driver for T-1822 Sec32.5's incremental
// contrast (fused-Q spike, toggle-off vs toggle-on), built on T-1954's own
// certified spike (brunel/t1954-fusedq-spike@fca77bb).
//
// Mechanical generalization of tools/t1954_selfcheck_q.cpp (T-1954's own
// certified Gate B mechanism): the SAME in-memory Q-landing-constant injection
// (verbatim), but taking the prompt, max_new_tokens, and stop-token IDs as CLI
// arguments instead of T-1954's own hardcoded single smoke prompt, so this tool
// can be driven once per prompt over T-1800/T-1818's frozen 32-prompt decode
// population (Claude/Brunel/t1953-slm-side-probes-2026-08-12.md Sec3's own
// PROMPTS list) exactly as tools/sslm_generate.exe already is by that
// campaign's own capture scripts.
//
// Q's own fused/legacy toggle is NOT a parameter -- read internally via
// SSLM_OPTION_G_FUSED_Q_LANDING (T-1954 Sec2, D-SLM2748), exactly like every
// other T-1954 tool. The caller sets the env var before invoking this binary.
//
// Usage:
//   t1960_decode_q <model.sslm> <tokenizer.sslm> <derived_q_constants.txt>
//       <prompt text> --max-new N [--stop id1,id2,...]
//
// Output line format matches sslm_generate.exe's own convention
// ("output_tokens (N): id id id ...") so it can be parsed by the same regex
// T-1953's own t1953_capture_arm_decode.py already uses.

#include <cstdint>
#include <cstdio>
#include <cstdlib>
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

// Verbatim of tools/t1954_selfcheck_q.cpp's own LoadDerivedConstants.
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

std::vector<int32_t> ParseIntList(const std::string& csv) {
	std::vector<int32_t> out;
	std::istringstream iss(csv);
	std::string tok;
	while (std::getline(iss, tok, ',')) {
		if (!tok.empty()) out.push_back(std::atoi(tok.c_str()));
	}
	return out;
}

}  // namespace

int main(int argc, char** argv) {
	if (argc < 5) {
		std::fprintf(stderr,
		              "usage: %s <model.sslm> <tokenizer.sslm> <derived_constants.txt> <prompt> "
		              "[--max-new N] [--stop id1,id2,...]\n",
		              argv[0]);
		return 2;
	}
	const std::string model_path = argv[1];
	const std::string tokenizer_path = argv[2];
	const std::string derived_path = argv[3];
	const std::string prompt = argv[4];
	int max_new = 8;
	std::vector<int32_t> stop_ids;
	for (int i = 5; i < argc; ++i) {
		const std::string a = argv[i];
		if (a == "--max-new" && i + 1 < argc) {
			max_new = std::atoi(argv[++i]);
		} else if (a == "--stop" && i + 1 < argc) {
			stop_ids = ParseIntList(argv[++i]);
		}
	}
	if (max_new <= 0) {
		std::fprintf(stderr, "FAILED stage=args: --max-new must be positive\n");
		return 2;
	}

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
	// Confirmed unmodified: reads only, never writes the shipped artifact.
	SslmModelView model_view;
	std::string model_err;
	const SslmModelStatus load_status =
	    SslmModel::Load(model_bytes.data(), model_bytes.size(), model_view, &model_err);
	if (load_status != SslmModelStatus::Ok) {
		std::fprintf(stderr, "FAILED stage=model_load status=%s diagnostic=\"%s\"\n",
		             SslmModelStatusName(load_status), model_err.c_str());
		return 1;
	}

	const uint32_t num_heads = model_view.config.num_attention_heads;
	const uint32_t num_kv_heads = model_view.config.num_key_value_heads;
	const uint32_t num_hidden_layers = model_view.config.num_hidden_layers;
	const size_t hidden_size = model_view.config.hidden_size;

	PreflightScanWscFolds(model_view);

	std::vector<LayerBacking> backings(num_hidden_layers);
	std::vector<LayerWeights> layers(num_hidden_layers);
	uint32_t layers_injected = 0;
	for (uint32_t l = 0; l < num_hidden_layers; ++l) {
		std::string marshal_err;
		if (!MarshalLayer(model_view, l, num_heads, num_kv_heads, backings[l], layers[l],
		                   &marshal_err)) {
			std::fprintf(stderr, "FAILED stage=layer_weights_marshal layer=%u diagnostic=\"%s\"\n", l,
			             marshal_err.c_str());
			return 1;
		}
		const auto it = derived.find(l);
		if (it == derived.end()) {
			std::fprintf(stderr, "FAILED stage=derived_constants_lookup: no entry for layer=%u\n", l);
			return 1;
		}
		layers[l].q_landing_m_out = it->second.m_out;
		layers[l].q_landing_e_out = it->second.e_out;
		layers[l].q_landing_e_t = it->second.e_t;
		layers[l].q_landing_r_t = it->second.r_t;
		++layers_injected;
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

	const std::vector<int32_t> prompt_tokens = tokenizer.Encode(prompt);

	std::vector<int32_t> out_tokens(static_cast<size_t>(max_new));
	std::vector<int32_t> out_logit_rows(static_cast<size_t>(max_new) *
	                                     static_cast<size_t>(model_view.config.vocab_size));
	size_t out_tokens_produced = 0;
	SslmDecodeStopReason stop_reason = SslmDecodeStopReason::MaxTokensReached;

	const SslmForwardStatus decode_status = RunGreedyDecodeLoop(
	    seq, layers.data(), num_hidden_layers, hidden_size, model_view.config.head_dim,
	    num_kv_heads, model_view.config.intermediate_size, context_cap, model_view.rope_tables,
	    prompt_tokens.data(), prompt_tokens.size(), embed_weights, embed_site_constant,
	    final_norm_gain.data(), final_norm_site_constant, head_weights,
	    static_cast<int32_t>(model_view.config.vocab_size),
	    stop_ids.empty() ? nullptr : stop_ids.data(), stop_ids.size(),
	    static_cast<size_t>(max_new), workspace.data(), workspace.size(), out_tokens.data(),
	    out_logit_rows.data(), out_tokens.size(), &out_tokens_produced, &stop_reason,
	    model_view.config.kv_precision, model_view.option_g_fused_k_landing);

	if (decode_status != SslmForwardStatus::Ok) {
		std::printf("FAILED stage=decode status=%s\n", SslmForwardStatusName(decode_status));
		return 1;
	}

	const char* q_toggle_env = std::getenv("SSLM_OPTION_G_FUSED_Q_LANDING");
	const bool q_toggle_on = q_toggle_env && q_toggle_env[0] != '\0' && q_toggle_env[0] != '0';
	std::printf("prompt_tokens (%zu):", prompt_tokens.size());
	for (int32_t t : prompt_tokens) std::printf(" %d", t);
	std::printf("\n");
	std::printf("output_tokens (%zu):", out_tokens_produced);
	for (size_t i = 0; i < out_tokens_produced; ++i) std::printf(" %d", out_tokens[i]);
	std::printf("\n");
	std::printf("stop_reason: %d\n", static_cast<int>(stop_reason));
	std::printf("q_landing_injected: %u/%u\n", layers_injected, num_hidden_layers);
	std::printf("SSLM_OPTION_G_FUSED_Q_LANDING(read-back): %d\n", q_toggle_on ? 1 : 0);
	std::printf("kv_saturation_count: %llu\n",
	            static_cast<unsigned long long>(seq.kv_saturation_count));
	return 0;
}

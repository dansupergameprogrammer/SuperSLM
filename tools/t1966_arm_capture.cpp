// t1966_arm_capture.cpp -- T-1966 (Brunel micro-round, disposable, never
// merges; D-SLM2787/D-SLM2788). One capture per arm (legacy / static-fused
// / dynamic-fused), selected by the SAME env var every other T-1954/T-1959
// tool uses (SSLM_OPTION_G_FUSED_Q_LANDING: unset/0 = legacy, "1" =
// static-fused, "dynamic" = dynamic-fused), plus
// SSLM_OPTION_G_FUSED_Q_ANCHOR_CAPTURE=1 (both must be set by the caller
// before this process starts). Dumps forward_sites.h's own
// OptionGDynamicQAnchorDump() -- the SAME unified per-layer capture
// (forward_sites.cpp's own single arm-agnostic capture site) -- to a JSON
// file. tools/t1966_compare_arms.py reads three such dumps (one per arm,
// same token) and computes reconstructed-Q and QK-score error against the
// dynamic-fused capture's own wide_rotated target.
//
// The static-fused arm needs the SAME 28 derived landing constants
// t1954_selfcheck_q/t1954_trace_probe already use (out/t1954_derived_selfcheck.txt)
// -- passed as an optional 4th argument, required only when
// SSLM_OPTION_G_FUSED_Q_LANDING=1. Legacy and dynamic-fused need no
// constants at all (both confirmed by this ticket's own gates).
//
// Usage: t1966_arm_capture <model.sslm> <tokenizer.sslm> <out.json> [derived_constants.txt]

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
		std::fprintf(stderr, "usage: %s <model.sslm> <tokenizer.sslm> <out.json> "
		                     "[derived_constants.txt]\n",
		             argv[0]);
		return 2;
	}
	const std::string model_path = argv[1];
	const std::string tokenizer_path = argv[2];
	const std::string out_path = argv[3];
	const std::string derived_path = argc > 4 ? argv[4] : "";

	const char* arm_env = std::getenv("SSLM_OPTION_G_FUSED_Q_LANDING");
	const std::string arm_name = (arm_env == nullptr || arm_env[0] == '\0' ||
	                               (arm_env[0] == '0' && arm_env[1] == '\0'))
	                                  ? "legacy"
	                              : (std::string(arm_env) == "dynamic") ? "dynamic-fused"
	                                                                    : "static-fused";
	std::printf("arm: %s (SSLM_OPTION_G_FUSED_Q_LANDING=\"%s\")\n", arm_name.c_str(),
	            arm_env ? arm_env : "(unset)");

	std::map<uint32_t, DerivedQLanding> derived;
	if (arm_name == "static-fused") {
		if (derived_path.empty()) {
			std::fprintf(stderr, "FAILED: arm=static-fused requires <derived_constants.txt>\n");
			return 2;
		}
		std::string derived_err;
		if (!LoadDerivedConstants(derived_path, &derived, &derived_err)) {
			std::fprintf(stderr, "FAILED stage=derived_constants_load diagnostic=\"%s\"\n",
			             derived_err.c_str());
			return 1;
		}
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
	for (uint32_t l = 0; l < num_hidden_layers; ++l) {
		std::string marshal_err;
		if (!MarshalLayer(model_view, l, num_heads, num_kv_heads, backings[l], layers[l],
		                   &marshal_err)) {
			std::fprintf(stderr, "FAILED stage=layer_weights_marshal layer=%u diagnostic=\"%s\"\n", l,
			             marshal_err.c_str());
			return 1;
		}
		if (arm_name == "static-fused") {
			const auto it = derived.find(l);
			if (it == derived.end()) {
				std::fprintf(stderr, "FAILED stage=derived_constants_lookup: no entry for layer=%u\n",
				             l);
				return 1;
			}
			layers[l].q_landing_m_out = it->second.m_out;
			layers[l].q_landing_e_out = it->second.e_out;
			layers[l].q_landing_e_t = it->second.e_t;
			layers[l].q_landing_r_t = it->second.r_t;
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
	std::printf("decode completed: status=Ok, %zu tokens produced: ", out_tokens_produced);
	for (size_t i = 0; i < out_tokens_produced; ++i) std::printf("%d ", out_tokens[i]);
	std::printf("\n");

	const std::vector<OptionGDynamicQAnchorSample> samples = OptionGDynamicQAnchorDump();

	// T-1968 fix round (Critical 2/D-SLM2796): the JSON's own "arm" identity
	// is no longer the calling process's own claim (`arm_name`, derived from
	// the env var THIS process set) -- it is read back from the ENGINE's own
	// `arm_mode` field, captured inside `RunLayerLoopImpl` itself at
	// position 0 for every layer. Refuses (does not write a dump at all) if
	// any captured layer disagrees with the engine's own layer-0 answer, or
	// if the engine's own answer disagrees with what this process asked for
	// -- a real mismatch here means the toggle did not take effect the way
	// the caller believes, which is exactly the class of defect the
	// reviewer's own relabelling attack exploited when arm identity was a
	// caller-supplied string nothing checked against the engine.
	int32_t engine_arm_mode = -2;  // -2: "no captured layer found" (distinct from -1, "never written")
	bool engine_arm_mode_consistent = true;
	for (const OptionGDynamicQAnchorSample& s : samples) {
		if (!s.captured) continue;
		if (engine_arm_mode == -2) {
			engine_arm_mode = s.arm_mode;
		} else if (s.arm_mode != engine_arm_mode) {
			engine_arm_mode_consistent = false;
		}
	}
	if (!engine_arm_mode_consistent) {
		std::fprintf(stderr, "FAILED: engine's own arm_mode disagrees across captured layers -- "
		                     "refusing to write a dump with an inconsistent arm identity\n");
		return 1;
	}
	if (engine_arm_mode == -2) {
		std::fprintf(stderr, "FAILED: no layer was captured at position 0 -- anchor capture did "
		                     "not run (env var not set, or position 0 was never reached)\n");
		return 1;
	}
	const char* const kArmModeNames[3] = {"legacy", "static-fused", "dynamic-fused"};
	if (engine_arm_mode < 0 || engine_arm_mode > 2) {
		std::fprintf(stderr, "FAILED: engine's own arm_mode=%d is out of the known [0,2] range\n",
		             engine_arm_mode);
		return 1;
	}
	const std::string engine_arm_name = kArmModeNames[engine_arm_mode];
	if (engine_arm_name != arm_name) {
		std::fprintf(stderr,
		             "FAILED: this process asked for arm=\"%s\" (SSLM_OPTION_G_FUSED_Q_LANDING=\"%s\") "
		             "but the ENGINE's own readback reports arm=\"%s\" -- refusing to write a "
		             "mislabelled dump\n",
		             arm_name.c_str(), arm_env ? arm_env : "(unset)", engine_arm_name.c_str());
		return 1;
	}
	std::printf("engine's own arm readback (arm_mode, captured at position 0, consistent across "
	            "%zu layers): %s\n",
	            samples.size(), engine_arm_name.c_str());

	std::ofstream out(out_path);
	if (!out) {
		std::fprintf(stderr, "FAILED: could not open \"%s\" for writing\n", out_path.c_str());
		return 1;
	}
	// "arm" is the engine's own readback, not the caller's claim (both are
	// identical here, since the check above just refused otherwise -- but
	// what is WRITTEN is the verified value). "decode_tokens" is the
	// reviewer's own named remedy for Critical 2: the ACTUAL output of the
	// real 16-token decode this process just ran, so a vitality check can
	// compare the dump's own recorded tokens against a pinned baseline,
	// rather than a hardcoded literal compared to itself.
	out << "{\n  \"arm\": \"" << engine_arm_name << "\",\n  \"arm_mode\": " << engine_arm_mode
	    << ",\n  \"hidden_size\": " << hidden_size
	    << ",\n  \"num_hidden_layers\": " << num_hidden_layers << ",\n  \"decode_tokens\": [";
	for (size_t i = 0; i < out_tokens_produced; ++i) {
		if (i) out << ",";
		out << out_tokens[i];
	}
	out << "],\n  \"layers\": [\n";
	for (uint32_t l = 0; l < samples.size(); ++l) {
		const OptionGDynamicQAnchorSample& s = samples[l];
		out << "    {\"layer\": " << l << ", \"codes\": [";
		for (size_t i = 0; i < s.codes.size(); ++i) {
			if (i) out << ",";
			out << static_cast<int>(s.codes[i]);
		}
		out << "], \"q_scale_m\": " << s.q_scale_m << ", \"q_scale_e\": " << s.q_scale_e
		    << ", \"normed_scale_m\": " << s.normed_scale_m
		    << ", \"normed_scale_e\": " << s.normed_scale_e
		    << ", \"site_constant_m\": " << s.site_constant_m
		    << ", \"site_constant_e\": " << s.site_constant_e << ", \"wide_rotated\": [";
		for (size_t i = 0; i < s.wide_rotated.size(); ++i) {
			if (i) out << ",";
			out << s.wide_rotated[i];
		}
		out << "], \"k_row_head0\": [";
		for (size_t i = 0; i < s.k_row_head0.size(); ++i) {
			if (i) out << ",";
			out << static_cast<int>(s.k_row_head0[i]);
		}
		out << "]}" << (l + 1 < samples.size() ? "," : "") << "\n";
	}
	out << "  ]\n}\n";
	out.close();
	std::printf("wrote %zu layer records to \"%s\" (position 0 only, per T-1968's own C1 fix)\n",
	            samples.size(), out_path.c_str());
	return 0;
}

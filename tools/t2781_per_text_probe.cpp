// T-2781: independent one-text final-hidden probe for the v1.4 GPU identity cell.
// Usage: t2781_per_text_probe <model.sslm> <comma-separated-token-ids>
//        [--gpu] --dump-final-hidden <path> --final-hidden-only
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include "superslm/artifact.h"
#include "superslm/forward_sites.h"
#include "superslm/gpu_port.h"
#include "superslm/model.h"
#include "sslm_marshal.h"

using namespace superslm;
using superslm_marshal::LayerBacking;
using superslm_marshal::MarshalLayer;
using superslm_marshal::ReadCarriedScale;
using superslm_marshal::ReadFile;
using superslm_marshal::WidenGainToInt32;

namespace {

bool ParseTokenIds(const char* spec, std::vector<int32_t>* out) {
	std::string text(spec);
	for (size_t pos = 0; pos < text.size();) {
		const size_t end = text.find(',', pos);
		const std::string field = text.substr(pos, end == std::string::npos ? end : end - pos);
		if (field.empty()) return false;
		try { out->push_back(static_cast<int32_t>(std::stol(field))); }
		catch (...) { return false; }
		if (end == std::string::npos) break;
		pos = end + 1;
	}
	return !out->empty();
}

bool WriteFinalHiddenFrame(const char* path, const std::vector<int8_t>& codes,
	                         const CarriedScale& scale) {
	std::ofstream output(path, std::ios::binary | std::ios::trunc);
	if (!output) return false;
	const uint64_t magic = UINT64_C(0x54474D5331373032);  // T-2701 final-hidden frame.
	const uint64_t count = static_cast<uint64_t>(codes.size());
	output.write(reinterpret_cast<const char*>(&magic), sizeof(magic));
	output.write(reinterpret_cast<const char*>(&count), sizeof(count));
	output.write(reinterpret_cast<const char*>(&scale.m), sizeof(scale.m));
	output.write(reinterpret_cast<const char*>(&scale.e), sizeof(scale.e));
	output.write(reinterpret_cast<const char*>(codes.data()), static_cast<std::streamsize>(codes.size()));
	return static_cast<bool>(output);
}

bool ForwardOne(const std::vector<int32_t>& tokens, SslmModelView& model,
	            std::vector<LayerWeights>& layers, const int8_t* embed_weights,
	            const CarriedScale& embed_scale, const std::vector<int32_t>& final_gains,
	            const CarriedScale& final_scale, bool use_gpu, std::vector<uint8_t>& workspace,
	            std::vector<int8_t>& hidden_codes, std::vector<int8_t>* final_codes,
	            CarriedScale* final_hidden_scale, std::string* error) {
	const size_t hidden = model.config.hidden_size;
	const size_t head_dim = model.config.head_dim;
	const size_t kv_heads = model.config.num_key_value_heads;
	SequenceLayerState seq{};
	seq.hidden_codes = hidden_codes.data();
	const OptionGKLandingMode k_mode = model.option_g_fused_k_landing ? OptionGKLandingMode::kFused
	                                                                    : OptionGKLandingMode::kLegacy;
	std::vector<uint8_t> gpu_q_codes(model.config.num_attention_heads * head_dim);
	for (const int32_t token : tokens) {
		CarriedScale token_scale{};
		SslmForwardStatus status = EmbedEntry(token, static_cast<int32_t>(model.config.vocab_size),
		                                        embed_weights, hidden, embed_scale,
		                                        hidden_codes.data(), &token_scale);
		if (status != SslmForwardStatus::Ok) { *error = "embed"; return false; }
		seq.hidden_scale = token_scale;
		seq.layer_index = 0;
		if (use_gpu) {
			superslm_gpu::GpuLayerLoopInFlight* inflight = nullptr;
			status = superslm_gpu::RunLayerLoopGpuSubmit(
				seq, layers.data(), static_cast<uint32_t>(layers.size()), static_cast<uint32_t>(layers.size()),
				hidden, head_dim, kv_heads, model.config.intermediate_size, model.config.context_cap,
				model.rope_tables, workspace.data(), workspace.size(), nullptr, nullptr, &inflight, nullptr,
				nullptr, nullptr, false, 0, 0, nullptr, gpu_q_codes.size(), gpu_q_codes.data(),
				gpu_q_codes.size(), 1, layers[0].q_norm_gain != nullptr && layers[0].k_norm_gain != nullptr);
			if (status == SslmForwardStatus::Ok && inflight != nullptr) {
				int32_t ready = 0;
				status = superslm_gpu::RunLayerLoopGpuFinish(inflight, seq, workspace.data(), 1, &ready,
				                                               gpu_q_codes.data());
			}
		} else {
			status = RunLayerLoop(seq, layers.data(), static_cast<uint32_t>(layers.size()),
							  static_cast<uint32_t>(layers.size()), hidden, head_dim, kv_heads,
							  model.config.intermediate_size, model.config.context_cap, model.rope_tables,
							  workspace.data(), workspace.size(), k_mode, {}, 0, &model.trace_hook,
							  model.config.num_attention_heads * model.config.head_dim);
		}
		if (status != SslmForwardStatus::Ok) { *error = SslmForwardStatusName(status); return false; }
	}
	final_codes->resize(hidden);
	const SslmForwardStatus status = RmsNormSite(hidden_codes.data(), final_gains.data(), hidden,
	                                             seq.hidden_scale, final_scale, final_codes->data(),
	                                             final_hidden_scale, "final_norm");
	if (status != SslmForwardStatus::Ok) { *error = SslmForwardStatusName(status); return false; }
	return true;
}

}  // namespace

int main(int argc, char** argv) {
	if (argc < 5)
		return std::fprintf(stderr, "usage: %s <model.sslm> <token-ids> [--gpu] --dump-final-hidden <path> --final-hidden-only\\n", argv[0]), 2;
	bool use_gpu = false, final_hidden_only = false;
	const char* dump_path = nullptr;
	for (int i = 3; i < argc; ++i) {
		if (std::strcmp(argv[i], "--gpu") == 0) use_gpu = true;
		else if (std::strcmp(argv[i], "--dump-final-hidden") == 0 && i + 1 < argc) dump_path = argv[++i];
		else if (std::strcmp(argv[i], "--final-hidden-only") == 0) final_hidden_only = true;
		else return std::fprintf(stderr, "unknown option: %s\\n", argv[i]), 2;
	}
	if (!dump_path || !final_hidden_only)
		return std::fprintf(stderr, "--dump-final-hidden <path> and --final-hidden-only are required\\n"), 2;
	std::vector<int32_t> tokens;
	if (!ParseTokenIds(argv[2], &tokens)) return std::fprintf(stderr, "invalid token-id list\\n"), 2;
	std::vector<uint8_t> bytes;
	if (!ReadFile(argv[1], bytes)) return std::fprintf(stderr, "could not read model\\n"), 1;
	SslmModelView model;
	std::string error;
	if (SslmModel::Load(bytes.data(), bytes.size(), model, &error) != SslmModelStatus::Ok)
		return std::fprintf(stderr, "model load: %s\\n", error.c_str()), 1;
	std::vector<LayerBacking> backing(model.config.num_hidden_layers);
	std::vector<LayerWeights> layers(model.config.num_hidden_layers);
	for (uint32_t layer = 0; layer < model.config.num_hidden_layers; ++layer) {
		if (!MarshalLayer(model, layer, model.config.num_attention_heads, model.config.num_key_value_heads,
		                  backing[layer], layers[layer], &error))
			return std::fprintf(stderr, "marshal layer %u: %s\\n", layer, error.c_str()), 1;
	}
	const SslmTensorView* embed = model.weights.Tensor("embed");
	const SslmTensorView* final_gain = model.weights.Tensor("final_norm.gain");
	if (!embed || !final_gain) return std::fprintf(stderr, "missing embed/final_norm.gain\\n"), 1;
	bool constants_ok = true;
	const CarriedScale embed_scale = ReadCarriedScale(model.composition_constants, "embed", &constants_ok);
	const CarriedScale final_scale = ReadCarriedScale(model.composition_constants, "final_norm", &constants_ok);
	if (!constants_ok) return std::fprintf(stderr, "missing composition constants\\n"), 1;
	const size_t workspace_bytes = static_cast<size_t>(model.config.num_hidden_layers) *
	                               static_cast<size_t>(model.config.context_cap) *
	                               model.config.num_key_value_heads * model.config.head_dim * 2;
	std::vector<uint8_t> workspace(workspace_bytes);
	std::vector<int8_t> hidden_codes(model.config.hidden_size), final_codes;
	CarriedScale final_hidden_scale{};
	if (!ForwardOne(tokens, model, layers, reinterpret_cast<const int8_t*>(embed->data), embed_scale,
	                WidenGainToInt32(*final_gain), final_scale, use_gpu, workspace, hidden_codes,
	                &final_codes, &final_hidden_scale, &error))
		return std::fprintf(stderr, "forward: %s\\n", error.c_str()), 1;
	if (!WriteFinalHiddenFrame(dump_path, final_codes, final_hidden_scale))
		return std::fprintf(stderr, "could not write final-hidden frame\\n"), 1;
	return 0;
}

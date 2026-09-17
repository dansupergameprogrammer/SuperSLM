// T-2776: artifact-backed batch embedding driver for pre-tokenised text rows.
// Usage: t2776_batched_embed_driver <model.sslm> <token-rows.txt> <vectors.bin> [--gpu]
// Each non-empty input line is one comma-separated token-id sequence.  The output is the
// concatenation of T-2701 final-hidden dump frames, one frame per non-empty input row.
#include <chrono>
#include <cstdio>
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
using superslm_marshal::PreflightScanWscFolds;
using superslm_marshal::ReadCarriedScale;
using superslm_marshal::ReadFile;
using superslm_marshal::WidenGainToInt32;

namespace {

bool ParseTokenIds(const std::string& spec, std::vector<int32_t>* out) {
	for (size_t pos = 0; pos < spec.size();) {
		const size_t end = spec.find(',', pos);
		const std::string field = spec.substr(pos, end == std::string::npos ? end : end - pos);
		if (field.empty()) return false;
		try { out->push_back(static_cast<int32_t>(std::stol(field))); }
		catch (...) { return false; }
		if (end == std::string::npos) break;
		pos = end + 1;
	}
	return !out->empty();
}

bool WriteFinalHiddenFrame(std::ofstream& output, const std::vector<int8_t>& codes,
	                       const CarriedScale& scale) {
	const uint64_t magic = UINT64_C(0x54474D5331373032);  // T-2701 final-hidden dump frame.
	const uint64_t count = static_cast<uint64_t>(codes.size());
	output.write(reinterpret_cast<const char*>(&magic), sizeof(magic));
	output.write(reinterpret_cast<const char*>(&count), sizeof(count));
	output.write(reinterpret_cast<const char*>(&scale.m), sizeof(scale.m));
	output.write(reinterpret_cast<const char*>(&scale.e), sizeof(scale.e));
	output.write(reinterpret_cast<const char*>(codes.data()), static_cast<std::streamsize>(codes.size()));
	return static_cast<bool>(output);
}

bool EmbedRow(const std::vector<int32_t>& tokens, SslmModelView& model,
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
		SslmForwardStatus st = EmbedEntry(token, static_cast<int32_t>(model.config.vocab_size),
		                                  embed_weights, hidden, embed_scale, hidden_codes.data(), &token_scale);
		if (st != SslmForwardStatus::Ok) { *error = "embed"; return false; }
		seq.hidden_scale = token_scale;
		seq.layer_index = 0;
		if (use_gpu) {
			superslm_gpu::GpuLayerLoopInFlight* inflight = nullptr;
			st = superslm_gpu::RunLayerLoopGpuSubmit(
			    seq, layers.data(), static_cast<uint32_t>(layers.size()), static_cast<uint32_t>(layers.size()),
			    hidden, head_dim, kv_heads, model.config.intermediate_size, model.config.context_cap,
			    model.rope_tables, workspace.data(), workspace.size(), nullptr, nullptr, &inflight, nullptr,
			    nullptr, nullptr, false, 0, 0, nullptr, gpu_q_codes.size(), gpu_q_codes.data(),
			    gpu_q_codes.size(), 1, layers[0].q_norm_gain != nullptr && layers[0].k_norm_gain != nullptr);
			if (st == SslmForwardStatus::Ok && inflight != nullptr) {
				int32_t ready = 0;
				st = superslm_gpu::RunLayerLoopGpuFinish(inflight, seq, workspace.data(), 1, &ready,
				                                         gpu_q_codes.data());
			}
		} else {
			st = RunLayerLoop(seq, layers.data(), static_cast<uint32_t>(layers.size()),
			                  static_cast<uint32_t>(layers.size()), hidden, head_dim, kv_heads,
			                  model.config.intermediate_size, model.config.context_cap, model.rope_tables,
			                  workspace.data(), workspace.size(), k_mode, {}, 0, &model.trace_hook,
			                  model.config.num_attention_heads * model.config.head_dim);
		}
		if (st != SslmForwardStatus::Ok) { *error = SslmForwardStatusName(st); return false; }
	}
	final_codes->resize(hidden);
	const SslmForwardStatus st = RmsNormSite(hidden_codes.data(), final_gains.data(), hidden,
	                                        seq.hidden_scale, final_scale, final_codes->data(),
	                                        final_hidden_scale, "final_norm");
	if (st != SslmForwardStatus::Ok) { *error = SslmForwardStatusName(st); return false; }
	return true;
}

}  // namespace

int main(int argc, char** argv) {
	if (argc != 4 && argc != 5)
		return std::fprintf(stderr, "usage: %s <model.sslm> <token-rows.txt> <vectors.bin> [--gpu]\\n", argv[0]), 2;
	const bool use_gpu = argc == 5 && std::strcmp(argv[4], "--gpu") == 0;
	if (argc == 5 && !use_gpu) return std::fprintf(stderr, "unknown option: %s\\n", argv[4]), 2;
	std::vector<uint8_t> bytes;
	if (!ReadFile(argv[1], bytes)) return std::fprintf(stderr, "could not read model\\n"), 1;
	SslmModelView model;
	std::string error;
	if (SslmModel::Load(bytes.data(), bytes.size(), model, &error) != SslmModelStatus::Ok)
		return std::fprintf(stderr, "model load: %s\\n", error.c_str()), 1;
	PreflightScanWscFolds(model);
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
	const std::vector<int32_t> final_gains = WidenGainToInt32(*final_gain);
	const size_t workspace_bytes = static_cast<size_t>(model.config.num_hidden_layers) *
	                               static_cast<size_t>(model.config.context_cap) *
	                               model.config.num_key_value_heads * model.config.head_dim * 2;
	std::vector<uint8_t> workspace(workspace_bytes);
	std::vector<int8_t> hidden_codes(model.config.hidden_size);
	std::ifstream input(argv[2]);
	std::ofstream output(argv[3], std::ios::binary | std::ios::trunc);
	if (!input || !output) return std::fprintf(stderr, "could not open input or output\\n"), 1;
	const auto started = std::chrono::steady_clock::now();
	size_t rows = 0, tokens_total = 0, line_number = 0;
	std::string line;
	while (std::getline(input, line)) {
		++line_number;
		if (line.empty()) continue;
		std::vector<int32_t> tokens;
		if (!ParseTokenIds(line, &tokens))
			return std::fprintf(stderr, "invalid token ids at row %zu\\n", line_number), 2;
		std::vector<int8_t> final_codes;
		CarriedScale final_hidden_scale{};
		if (!EmbedRow(tokens, model, layers, reinterpret_cast<const int8_t*>(embed->data), embed_scale,
		              final_gains, final_scale, use_gpu, workspace, hidden_codes, &final_codes,
		              &final_hidden_scale, &error))
			return std::fprintf(stderr, "forward row %zu: %s\\n", line_number, error.c_str()), 1;
		if (!WriteFinalHiddenFrame(output, final_codes, final_hidden_scale))
			return std::fprintf(stderr, "could not write output frame %zu\\n", rows), 1;
		++rows;
		tokens_total += tokens.size();
	}
	const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
	std::printf("backend=%s rows=%zu tokens=%zu seconds=%.6f rows_per_second=%.6f tokens_per_second=%.6f\\n",
	            use_gpu ? "gpu" : "cpu", rows, tokens_total, seconds,
	            seconds == 0.0 ? 0.0 : rows / seconds, seconds == 0.0 ? 0.0 : tokens_total / seconds);
	return 0;
}

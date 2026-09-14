// T-2701: artifact-backed CPU full-forward digest for the Python parity cell.
// Usage: t2701_cpu_forward_probe <model.sslm> <comma-separated-token-ids>
// Prints the SHA-256 of the final raw int32 logits in little-endian row-major order.
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "superslm/artifact.h"
#include "superslm/forward_sites.h"
#include "superslm/gpu_port.h"
#include "superslm/model.h"
#include "superslm/sha256.h"
#include "sslm_marshal.h"

using namespace superslm;
using superslm_marshal::LayerBacking;
using superslm_marshal::MarshalLayer;
using superslm_marshal::PreflightScanWscFolds;
using superslm_marshal::ReadCarriedScale;
using superslm_marshal::ReadFile;
using superslm_marshal::WidenGainToInt32;

namespace {

bool ParseTokenIds(const char* spec, std::vector<int32_t>* out) {
	std::string s(spec);
	for (size_t pos = 0; pos < s.size();) {
		const size_t end = s.find(',', pos);
		const std::string field = s.substr(pos, end == std::string::npos ? end : end - pos);
		if (field.empty()) return false;
		try {
			out->push_back(static_cast<int32_t>(std::stol(field)));
		} catch (...) { return false; }
		if (end == std::string::npos) break;
		pos = end + 1;
	}
	return !out->empty();
}

}  // namespace

int main(int argc, char** argv) {
	if (argc != 3 && argc != 4) {
		std::fprintf(stderr, "usage: %s <model.sslm> <comma-separated-token-ids> [--gpu]\n", argv[0]);
		return 2;
	}
	const bool use_gpu = argc == 4 && std::strcmp(argv[3], "--gpu") == 0;
	if (argc == 4 && !use_gpu) return std::fprintf(stderr, "unknown option: %s\n", argv[3]), 2;
	std::vector<int32_t> tokens;
	if (!ParseTokenIds(argv[2], &tokens)) {
		std::fprintf(stderr, "invalid token-id list: %s\n", argv[2]);
		return 2;
	}
	std::vector<uint8_t> bytes;
	if (!ReadFile(argv[1], bytes)) return std::fprintf(stderr, "could not read model\n"), 1;
	SslmModelView model;
	std::string error;
	if (SslmModel::Load(bytes.data(), bytes.size(), model, &error) != SslmModelStatus::Ok)
		return std::fprintf(stderr, "model load: %s\n", error.c_str()), 1;

	const uint32_t layers_n = model.config.num_hidden_layers;
	const size_t hidden = model.config.hidden_size;
	const size_t head_dim = model.config.head_dim;
	const size_t kv_heads = model.config.num_key_value_heads;
	const int64_t context_cap = static_cast<int64_t>(model.config.context_cap);
	PreflightScanWscFolds(model);
	std::vector<LayerBacking> backing(layers_n);
	std::vector<LayerWeights> layers(layers_n);
	for (uint32_t layer = 0; layer < layers_n; ++layer) {
		if (!MarshalLayer(model, layer, model.config.num_attention_heads, model.config.num_key_value_heads,
		                  backing[layer], layers[layer], &error))
			return std::fprintf(stderr, "marshal layer %u: %s\n", layer, error.c_str()), 1;
	}
	const SslmTensorView* embed = model.weights.Tensor("embed");
	const SslmTensorView* final_gain = model.weights.Tensor("final_norm.gain");
	if (!embed || !final_gain) return std::fprintf(stderr, "missing embed/final_norm.gain\n"), 1;
	bool constants_ok = true;
	const CarriedScale embed_scale = ReadCarriedScale(model.composition_constants, "embed", &constants_ok);
	const CarriedScale final_scale = ReadCarriedScale(model.composition_constants, "final_norm", &constants_ok);
	if (!constants_ok) return std::fprintf(stderr, "missing composition constants\n"), 1;
	const int8_t* embed_weights = reinterpret_cast<const int8_t*>(embed->data);
	const int8_t* head_weights = embed_weights;
	if (!model.config.tie_word_embeddings) {
		const SslmTensorView* head = model.weights.Tensor("lm_head");
		if (!head) return std::fprintf(stderr, "missing lm_head\n"), 1;
		head_weights = reinterpret_cast<const int8_t*>(head->data);
	}
	const std::vector<int32_t> gains = WidenGainToInt32(*final_gain);
	const size_t kv_bytes = static_cast<size_t>(layers_n) * static_cast<size_t>(context_cap) * kv_heads * head_dim * 2;
	std::vector<uint8_t> workspace(kv_bytes);
	std::vector<int8_t> hidden_codes(hidden);
	SequenceLayerState seq;
	seq.hidden_codes = hidden_codes.data();
	const OptionGKLandingMode k_mode = model.option_g_fused_k_landing ? OptionGKLandingMode::kFused
	                                                                    : OptionGKLandingMode::kLegacy;
	for (const int32_t token : tokens) {
		CarriedScale token_scale{};
		SslmForwardStatus st = EmbedEntry(token, static_cast<int32_t>(model.config.vocab_size), embed_weights,
		                                 hidden, embed_scale, hidden_codes.data(), &token_scale);
		if (st != SslmForwardStatus::Ok) return std::fprintf(stderr, "embed: %s\n", SslmForwardStatusName(st)), 1;
		seq.hidden_scale = token_scale;
		seq.layer_index = 0;
		if (use_gpu) {
			st = superslm_gpu::RunLayerLoopGpu(
			    seq, layers.data(), layers_n, layers_n, hidden, head_dim, kv_heads,
			    model.config.intermediate_size, context_cap, model.rope_tables, workspace.data(), workspace.size(),
			    nullptr, nullptr, 1);
		} else {
			st = RunLayerLoop(seq, layers.data(), layers_n, layers_n, hidden, head_dim, kv_heads,
			                  model.config.intermediate_size, context_cap, model.rope_tables, workspace.data(),
			                  workspace.size(), k_mode, {}, 0, nullptr,
			                  model.config.num_attention_heads * model.config.head_dim);
		}
		if (st != SslmForwardStatus::Ok) return std::fprintf(stderr, "forward: %s\n", SslmForwardStatusName(st)), 1;
	}
	std::vector<int8_t> final_codes(hidden);
	CarriedScale ignored{};
	SslmForwardStatus st = RmsNormSite(hidden_codes.data(), gains.data(), hidden, seq.hidden_scale,
	                                  final_scale, final_codes.data(), &ignored, "final_norm");
	if (st != SslmForwardStatus::Ok) return std::fprintf(stderr, "final norm: %s\n", SslmForwardStatusName(st)), 1;
	std::vector<int64_t> wide(model.config.vocab_size);
	std::vector<int32_t> logits(model.config.vocab_size);
	st = LogitsSite(final_codes.data(), hidden, head_weights, model.config.vocab_size, wide.data(), logits.data());
	if (st != SslmForwardStatus::Ok) return std::fprintf(stderr, "logits: %s\n", SslmForwardStatusName(st)), 1;
	uint8_t digest[32];
	Sha256Hash(reinterpret_cast<const uint8_t*>(logits.data()), logits.size() * sizeof(int32_t), digest);
	std::printf("backend=%s tokens=%s logits_sha256=%s\n", use_gpu ? "gpu" : "cpu", argv[2],
	            ToHex(digest).c_str());
	return 0;
}

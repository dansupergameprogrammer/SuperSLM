// T-2701: public-API CPU/GPU greedy-token parity over a growing KV cache.
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "superslm/forward_sites.h"
#include "superslm/gpu_1p0.h"
#include "superslm/gpu_port.h"
#include "superslm/model.h"
#include "superslm/sha256.h"
#include "sslm_marshal.h"

using namespace superslm;
using enum SslmGpuStatus;
using superslm_marshal::LayerBacking;
using superslm_marshal::MarshalLayer;
using superslm_marshal::ReadCarriedScale;
using superslm_marshal::ReadFile;
using superslm_marshal::WidenGainToInt32;

namespace {

bool ParseTokens(const char* spec, std::vector<int32_t>* out) {
	for (std::string s(spec); !s.empty();) {
		const size_t end = s.find(',');
		try { out->push_back(static_cast<int32_t>(std::stol(s.substr(0, end)))); }
		catch (...) { return false; }
		if (end == std::string::npos) break;
		s.erase(0, end + 1);
	}
	return !out->empty();
}

std::string Hex(const uint8_t digest[32]) {
	static constexpr char kDigits[] = "0123456789abcdef";
	std::string out(64, '0');
	for (size_t i = 0; i < 32; ++i) {
		out[2 * i] = kDigits[digest[i] >> 4];
		out[2 * i + 1] = kDigits[digest[i] & 15];
	}
	return out;
}

void PrintTokens(const char* label, const std::vector<int32_t>& tokens) {
	std::printf("%s=", label);
	for (size_t i = 0; i < tokens.size(); ++i) std::printf("%s%d", i ? "," : "", tokens[i]);
	std::printf("\n");
}

}  // namespace

int main(int argc, char** argv) {
	if (argc != 3) return std::fprintf(stderr, "usage: %s <model.sslm> <comma-token-ids>\n", argv[0]), 2;
	std::vector<int32_t> prompt;
	if (!ParseTokens(argv[2], &prompt)) return std::fprintf(stderr, "invalid token list\n"), 2;
	std::vector<uint8_t> bytes;
	if (!ReadFile(argv[1], bytes)) return std::fprintf(stderr, "cannot read model\n"), 1;
	SslmModelView view;
	std::string error;
	if (SslmModel::Load(bytes.data(), bytes.size(), view, &error) != SslmModelStatus::Ok)
		return std::fprintf(stderr, "model load: %s\n", error.c_str()), 1;

	const uint32_t layers_n = view.config.num_hidden_layers;
	const size_t hidden = view.config.hidden_size;
	const size_t kv_heads = view.config.num_key_value_heads;
	std::vector<LayerBacking> backing(layers_n);
	std::vector<LayerWeights> layers(layers_n);
	for (uint32_t l = 0; l < layers_n; ++l) {
		if (!MarshalLayer(view, l, view.config.num_attention_heads, view.config.num_key_value_heads,
		                  backing[l], layers[l], &error))
			return std::fprintf(stderr, "marshal %u: %s\n", l, error.c_str()), 1;
	}
	const SslmTensorView* embed = view.weights.Tensor("embed");
	const SslmTensorView* final_gain = view.weights.Tensor("final_norm.gain");
	if (embed == nullptr || final_gain == nullptr) return std::fprintf(stderr, "missing head tensors\n"), 1;
	bool constants_ok = true;
	const CarriedScale embed_scale = ReadCarriedScale(view.composition_constants, "embed", &constants_ok);
	const CarriedScale final_scale = ReadCarriedScale(view.composition_constants, "final_norm", &constants_ok);
	if (!constants_ok) return std::fprintf(stderr, "missing head constants\n"), 1;
	const int8_t* embed_weights = reinterpret_cast<const int8_t*>(embed->data);
	const std::vector<int32_t> final_gain_wide = WidenGainToInt32(*final_gain);
	const int8_t* head_weights = embed_weights;
	if (!view.config.tie_word_embeddings) {
		const SslmTensorView* head = view.weights.Tensor("lm_head");
		if (head == nullptr) return std::fprintf(stderr, "missing lm_head\n"), 1;
		head_weights = reinterpret_cast<const int8_t*>(head->data);
	}
	const size_t workspace_bytes = static_cast<size_t>(layers_n) * static_cast<size_t>(view.config.context_cap) *
	                               kv_heads * view.config.head_dim * 2;
	std::vector<uint8_t> workspace(workspace_bytes);
	std::vector<int8_t> hidden_codes(hidden);
	SequenceLayerState cpu_seq;
	cpu_seq.hidden_codes = hidden_codes.data();
	std::vector<int32_t> cpu_tokens(16);
	std::vector<int32_t> cpu_logits(16 * static_cast<size_t>(view.config.vocab_size));
	size_t cpu_count = 0;
	SslmDecodeStopReason stop = SslmDecodeStopReason::MaxTokensReached;
	const SslmForwardStatus cpu_status = RunGreedyDecodeLoop(
		cpu_seq, layers.data(), layers_n, hidden, view.config.head_dim, kv_heads,
		view.config.intermediate_size, view.config.context_cap, view.rope_tables, prompt.data(), prompt.size(),
		embed_weights, embed_scale, final_gain_wide.data(), final_scale, head_weights,
		static_cast<int32_t>(view.config.vocab_size), nullptr, 0, cpu_tokens.size(), workspace.data(), workspace.size(),
		cpu_tokens.data(), cpu_logits.data(), cpu_tokens.size(), &cpu_count, &stop, view.config.kv_precision,
		view.option_g_fused_k_landing, view.config.num_attention_heads);
	if (cpu_status != SslmForwardStatus::Ok || cpu_count != cpu_tokens.size())
		return std::fprintf(stderr, "CPU greedy failed: %s count=%zu\n", SslmForwardStatusName(cpu_status), cpu_count), 1;

	GpuContextConfig context_config{};
	GpuResidencyConfig residency_config{};
	SslmGpuContext* context = nullptr;
	SslmGpuModelHandle* gpu_model = nullptr;
	SslmGpuSequenceHandle* gpu_seq = nullptr;
	if (sslm_gpu_context_create(context_config, &context) != SSLM_OK ||
		sslm_gpu_model_map(context, &view, residency_config, &gpu_model) != SSLM_OK ||
		sslm_gpu_seq_create(context, gpu_model, view.config.context_cap, &gpu_seq) != SSLM_OK)
		return std::fprintf(stderr, "GPU public API setup failed\n"), 1;
	if (SslmGpuSeqPrefillPromptForG5Bridge(context, gpu_seq, prompt.data(), static_cast<int32_t>(prompt.size()),
	                                        superslm_gpu::kDispatchesPerLayer) != SSLM_OK)
		return std::fprintf(stderr, "GPU prompt prefill failed\n"), 1;
	std::vector<int32_t> gpu_tokens;
	int32_t current = prompt.back();
	for (size_t step = 0; step < 16; ++step) {
		int32_t next = -1;
		if (SslmGpuSeqDecodeStepForG5Bridge(context, gpu_seq, current,
		                                    superslm_gpu::kDispatchesPerLayer, &next) != SSLM_OK)
			return std::fprintf(stderr, "GPU decode failed at step %zu\n", step), 1;
		gpu_tokens.push_back(next);
		current = next;
	}
	sslm_gpu_seq_release(context, gpu_seq);
	sslm_gpu_model_unmap(context, gpu_model);
	sslm_gpu_context_destroy(context);

	cpu_tokens.resize(cpu_count);
	PrintTokens("cpu_tokens", cpu_tokens);
	PrintTokens("gpu_tokens", gpu_tokens);
	for (size_t step = 0; step < cpu_count; ++step) {
		uint8_t digest[32];
		Sha256Hash(reinterpret_cast<const uint8_t*>(cpu_logits.data() + step * view.config.vocab_size),
		           static_cast<size_t>(view.config.vocab_size) * sizeof(int32_t), digest);
		std::printf("cpu_step_logits step=%zu sha256=%s\n", step, Hex(digest).c_str());
	}
	uint8_t all_steps_digest[32];
	Sha256Hash(reinterpret_cast<const uint8_t*>(cpu_logits.data()),
	           cpu_count * static_cast<size_t>(view.config.vocab_size) * sizeof(int32_t), all_steps_digest);
	std::printf("cpu_step_logits_aggregate_sha256=%s\n", Hex(all_steps_digest).c_str());
	const bool equal = cpu_tokens == gpu_tokens;
	std::printf("token_identity=%s\n", equal ? "PASS" : "FAIL");
	return equal ? 0 : 1;
}

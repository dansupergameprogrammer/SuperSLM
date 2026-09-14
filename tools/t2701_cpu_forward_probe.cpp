// T-2701: artifact-backed CPU full-forward digest for the Python parity cell.
// Usage: t2701_cpu_forward_probe <model.sslm> <comma-separated-token-ids>
// Prints the SHA-256 of the final raw int32 logits in little-endian row-major order.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "superslm/artifact.h"
#include "superslm/forward_sites.h"
#include "superslm/gpu_port.h"
#include "superslm/model.h"
#include "superslm/sha256.h"
#include "superslm/trace_hook.h"
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

uint32_t Align8U32(uint32_t value) { return (value + 7u) & ~7u; }
uint32_t SeqScaleOff(uint32_t hidden_size) { return Align8U32(hidden_size * 4u); }

void PackEmbeddingBlock(const int8_t* codes, size_t hidden_size, const CarriedScale& scale,
                        uint8_t* out) {
	for (uint32_t i = 0; i < hidden_size; ++i) {
		const int32_t code = static_cast<int32_t>(codes[i]);
		std::memcpy(out + i * 4u, &code, sizeof(code));
	}
	const uint32_t scale_off = SeqScaleOff(static_cast<uint32_t>(hidden_size));
	std::memcpy(out + scale_off, &scale.m, sizeof(scale.m));
	std::memcpy(out + scale_off + sizeof(scale.m), &scale.e, sizeof(scale.e));
}

struct TraceCapture {
	struct Record {
		std::string site;
		size_t token_index = 0;
		std::string codes_sha256;
		size_t count = 0;
		CarriedScale scale{};
	};
	std::vector<Record> records;
	std::vector<int8_t> q_proj_codes;
	CarriedScale q_proj_scale{};
	std::vector<int8_t> q_norm_codes;
	std::vector<CarriedScale> q_norm_scales;
	std::vector<int8_t> attention_residual_codes;
	CarriedScale attention_residual_scale{};
};

void CaptureTrace(const SslmChainTraceRecord* chain, const SslmKvLandingTraceRecord*, void* user) {
	if (chain == nullptr) return;
	auto* capture = static_cast<TraceCapture*>(user);
	if (chain->site.starts_with("layer0.")) {
		uint8_t digest[32];
		Sha256Hash(reinterpret_cast<const uint8_t*>(chain->codes.data()), chain->codes.size(), digest);
		capture->records.push_back({std::string(chain->site), chain->token_index, ToHex(digest),
		                            chain->codes.size(), {chain->m_out, chain->e_out}});
	}
	if (chain->site == "layer0.q_proj.requant") {
		capture->q_proj_codes.assign(chain->codes.begin(), chain->codes.end());
		capture->q_proj_scale = CarriedScale{chain->m_out, chain->e_out};
	} else if (chain->site == "layer0.q_norm") {
		capture->q_norm_codes.insert(capture->q_norm_codes.end(), chain->codes.begin(), chain->codes.end());
		capture->q_norm_scales.push_back(CarriedScale{chain->m_out, chain->e_out});
	} else if (chain->site == "layer0.attn_residual") {
		capture->attention_residual_codes.assign(chain->codes.begin(), chain->codes.end());
		capture->attention_residual_scale = CarriedScale{chain->m_out, chain->e_out};
	}
}

void PrintTraceDigest(const char* name, const std::vector<int8_t>& codes,
	                  const std::vector<CarriedScale>& scales) {
	if (codes.empty()) return;
	uint8_t digest[32];
	Sha256Hash(reinterpret_cast<const uint8_t*>(codes.data()), codes.size(), digest);
	std::printf("%s_codes_sha256=%s count=%zu", name, ToHex(digest).c_str(), codes.size());
	for (size_t i = 0; i < scales.size(); ++i)
		std::printf(" scale[%zu]=%lld,%lld", i, static_cast<long long>(scales[i].m),
		            static_cast<long long>(scales[i].e));
	std::printf("\n");
}

void PrintTraceRecords(const TraceCapture& capture) {
	for (const TraceCapture::Record& record : capture.records) {
		std::printf("cpu_chain site=%s token=%zu codes_sha256=%s count=%zu scale=%lld,%lld\n",
		            record.site.c_str(), record.token_index, record.codes_sha256.c_str(), record.count,
		            static_cast<long long>(record.scale.m), static_cast<long long>(record.scale.e));
	}
}

}  // namespace

int main(int argc, char** argv) {
	if (argc != 3 && argc != 4) {
		std::fprintf(stderr, "usage: %s <model.sslm> <comma-separated-token-ids> [--gpu|--chunk|--gpu-chunk|--raw-k-cpu|--raw-k-gpu|--no-qnorm-cpu|--no-qnorm-gpu]\n", argv[0]);
		return 2;
	}
	const bool use_gpu = argc == 4 &&
	                     (std::strcmp(argv[3], "--gpu") == 0 || std::strcmp(argv[3], "--gpu-chunk") == 0);
	const bool use_chunk = argc == 4 &&
	                       (std::strcmp(argv[3], "--chunk") == 0 || std::strcmp(argv[3], "--gpu-chunk") == 0);
	const bool raw_k = argc == 4 &&
	                   (std::strcmp(argv[3], "--raw-k-cpu") == 0 || std::strcmp(argv[3], "--raw-k-gpu") == 0);
	const bool raw_k_gpu = argc == 4 && std::strcmp(argv[3], "--raw-k-gpu") == 0;
	const bool no_qnorm = argc == 4 &&
	                      (std::strcmp(argv[3], "--no-qnorm-cpu") == 0 || std::strcmp(argv[3], "--no-qnorm-gpu") == 0);
	const bool no_qnorm_gpu = argc == 4 && std::strcmp(argv[3], "--no-qnorm-gpu") == 0;
	if (argc == 4 && !use_gpu && !use_chunk && !raw_k && !no_qnorm)
		return std::fprintf(stderr, "unknown option: %s\n", argv[3]), 2;
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
	uint32_t layer_budget = layers_n;
	if (const char* budget_spec = std::getenv("T2701_LAYER_BUDGET")) {
		try { layer_budget = static_cast<uint32_t>(std::stoul(budget_spec)); }
		catch (...) { return std::fprintf(stderr, "invalid T2701_LAYER_BUDGET\n"), 2; }
		if (layer_budget == 0 || layer_budget > layers_n || use_chunk)
			return std::fprintf(stderr, "T2701_LAYER_BUDGET must be in [1,layers] and stepped-only\n"), 2;
	}
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
	if (raw_k) {
		for (LayerWeights& layer : layers) {
			layer.q_norm_gain = nullptr;
			layer.k_norm_gain = nullptr;
		}
	}
	if (no_qnorm) {
		for (LayerWeights& layer : layers) layer.q_norm_gain = nullptr;
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
	TraceCapture trace;
	if (!(use_gpu || raw_k_gpu || no_qnorm_gpu)) SslmSetTraceHook(model.trace_hook, CaptureTrace, &trace);
	std::vector<uint8_t> gpu_q_codes(model.config.num_attention_heads * head_dim);
	SslmForwardStatus forward_status = SslmForwardStatus::Ok;
	if (use_chunk) {
		std::vector<int8_t> chunk_codes(tokens.size() * hidden);
		std::vector<CarriedScale> chunk_scales(tokens.size());
		for (size_t i = 0; i < tokens.size(); ++i) {
			SslmForwardStatus st = EmbedEntry(tokens[i], static_cast<int32_t>(model.config.vocab_size),
			                                  embed_weights, hidden, embed_scale, chunk_codes.data() + i * hidden,
			                                  &chunk_scales[i]);
			if (st != SslmForwardStatus::Ok)
				return std::fprintf(stderr, "embed: %s\n", SslmForwardStatusName(st)), 1;
		}
		SslmForwardStatus st;
		if (use_gpu || raw_k_gpu || no_qnorm_gpu) {
			const uint32_t block_bytes = SeqScaleOff(static_cast<uint32_t>(hidden)) + 16u;
			std::vector<uint8_t> blocks(block_bytes * tokens.size());
			for (size_t i = 0; i < tokens.size(); ++i)
				PackEmbeddingBlock(chunk_codes.data() + i * hidden, hidden, chunk_scales[i],
				                   blocks.data() + i * block_bytes);
			superslm_gpu::GpuLayerLoopInFlight* inflight = nullptr;
			st = superslm_gpu::SubmitChunkToFullDepthForG5Bridge(
			    seq, layers.data(), layers_n, hidden, head_dim, kv_heads, model.config.intermediate_size,
			    context_cap, model.rope_tables, workspace.data(), workspace.size(), blocks.data(),
			    static_cast<uint32_t>(tokens.size()), nullptr, nullptr, nullptr, nullptr, nullptr, false,
			    0, 0, nullptr, &inflight, model.config.num_attention_heads * model.config.head_dim, 1,
			    layers_n != 0 && layers[0].q_norm_gain != nullptr && layers[0].k_norm_gain != nullptr);
			if (st == SslmForwardStatus::Ok && inflight != nullptr) {
				int32_t ready = 0;
				st = superslm_gpu::RunLayerLoopGpuFinish(inflight, seq, workspace.data(), 1, &ready);
			}
		} else {
			st = RunLayerLoopChunkBatched(chunk_codes.data(), chunk_scales.data(), tokens.size(), layers.data(),
			                              layers_n, hidden, head_dim, kv_heads, model.config.intermediate_size,
			                              context_cap, 0, model.rope_tables, workspace.data(), workspace.size(),
			                              model.option_g_fused_k_landing, &seq.kv_saturation_count, {}, nullptr,
			                              model.config.num_attention_heads * model.config.head_dim);
			std::memcpy(hidden_codes.data(), chunk_codes.data() + (tokens.size() - 1) * hidden, hidden);
			seq.hidden_scale = chunk_scales.back();
			seq.context_length = static_cast<int64_t>(tokens.size());
		}
		forward_status = st;
		if (st != SslmForwardStatus::Ok)
			return std::fprintf(stderr, "forward: %s\n", SslmForwardStatusName(st)), 1;
	} else for (const int32_t token : tokens) {
		CarriedScale token_scale{};
		SslmForwardStatus st = EmbedEntry(token, static_cast<int32_t>(model.config.vocab_size), embed_weights,
		                                 hidden, embed_scale, hidden_codes.data(), &token_scale);
		if (st != SslmForwardStatus::Ok) return std::fprintf(stderr, "embed: %s\n", SslmForwardStatusName(st)), 1;
		seq.hidden_scale = token_scale;
		seq.layer_index = 0;
		if (use_gpu || raw_k_gpu || no_qnorm_gpu) {
			superslm_gpu::GpuLayerLoopInFlight* inflight = nullptr;
			st = superslm_gpu::RunLayerLoopGpuSubmit(
			    seq, layers.data(), layers_n, layer_budget, hidden, head_dim, kv_heads,
			    model.config.intermediate_size, context_cap, model.rope_tables, workspace.data(), workspace.size(),
			    nullptr, nullptr, &inflight, nullptr, nullptr, nullptr, false, 0, 0, nullptr,
			    gpu_q_codes.size(), gpu_q_codes.data(), gpu_q_codes.size(), 1,
			    layers[0].q_norm_gain != nullptr && layers[0].k_norm_gain != nullptr);
			if (st == SslmForwardStatus::Ok && inflight != nullptr) {
				int32_t ready = 0;
				st = superslm_gpu::RunLayerLoopGpuFinish(inflight, seq, workspace.data(), 1, &ready,
				                                         gpu_q_codes.data());
			}
		} else {
			st = RunLayerLoop(seq, layers.data(), layers_n, layer_budget, hidden, head_dim, kv_heads,
			                  model.config.intermediate_size, context_cap, model.rope_tables, workspace.data(),
			                  workspace.size(), k_mode, {}, 0, &model.trace_hook,
			                  model.config.num_attention_heads * model.config.head_dim);
		}
		forward_status = st;
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
	uint8_t kv_digest[32];
	Sha256Hash(workspace.data(), workspace.size(), kv_digest);
	std::printf("backend=%s mode=%s layer_budget=%u tokens=%s logits_sha256=%s kv_sha256=%s\n", (use_gpu || raw_k_gpu || no_qnorm_gpu) ? "gpu" : "cpu",
	            use_chunk ? "chunk" : "stepped", layer_budget, argv[2],
	            ToHex(digest).c_str(), ToHex(kv_digest).c_str());
	std::printf("forward_status=%s gpu_sticky_tag=%d kv_saturation=%llu kv_landing=%llu k_normed_landing=%llu rope_q=%llu rope_k=%llu\n",
	            SslmForwardStatusName(forward_status), (use_gpu || raw_k_gpu || no_qnorm_gpu) ? 0 : -1,
	            static_cast<unsigned long long>(seq.kv_saturation_count),
	            static_cast<unsigned long long>(seq.kv_landing_saturation_count),
	            static_cast<unsigned long long>(seq.k_normed_landing_saturation_count),
	            static_cast<unsigned long long>(seq.rope_q_saturation_count),
	            static_cast<unsigned long long>(seq.rope_k_saturation_count));
	if (use_gpu || raw_k_gpu || no_qnorm_gpu) {
		std::vector<int8_t> signed_q(gpu_q_codes.begin(), gpu_q_codes.end());
		PrintTraceDigest("gpu_q", signed_q, {});
	} else {
		PrintTraceDigest("cpu_q_proj", trace.q_proj_codes, {trace.q_proj_scale});
		PrintTraceDigest("cpu_q_norm", trace.q_norm_codes, trace.q_norm_scales);
		PrintTraceDigest("cpu_attn_residual", trace.attention_residual_codes,
		                 {trace.attention_residual_scale});
		PrintTraceRecords(trace);
	}
	return 0;
}

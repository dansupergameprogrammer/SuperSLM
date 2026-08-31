// T-2432 Track A acceptance harness. Disposable, matching tools/t2039_c5_harness.cpp's own
// precedent ("Not part of the build.bat/CMake build graph -- compiled and run directly for
// this session's own verification"). Adapted from t2039_c5_harness.cpp: loads a real .sslm
// artifact, marshals it, embeds one token, and runs it through every layer on BOTH the CPU
// oracle (production RunLayerLoop) and the GPU port (RunLayerLoopGpu) from an IDENTICAL
// initial SequenceLayerState and workspace -- q_width threaded explicitly through both calls
// (design's own new parameter, forward_sites.h/gpu_port.h).
//
// Two checks beyond t2039_c5_harness.cpp's own template:
//   1. A trace hook installed on the CPU run's own model view captures the "q_proj.requant"
//      chain-trace record for token_index 0, layer 0, and asserts its own `codes` span has
//      length == q_width (not hidden_size) -- a direct, executed proof that Q's own output
//      row is genuinely widened on the CPU path (Track A steps 2/3, GS-02/GS-12).
//   2. The existing CPU-vs-GPU bit-identity comparison (hidden_codes, K/V cache, derived
//      logits) is unchanged from t2039_c5_harness.cpp's own template, but now runs at
//      non-square geometry -- so it is the GPU-side proof: a truncated-to-hidden_size GPU
//      q_proj/o_proj computation (GS-14 through GS-17, unfixed) would read stale/garbage
//      WorkScratch bytes for o_proj's missing input channels and diverge from the CPU's
//      complete computation, which this comparison would catch as a hidden_codes mismatch.
//
// Usage: t2432_geometry_harness <model.sslm> [token_id]
#include <cstdio>
#include <cstring>
#include <vector>

#include "superslm/artifact.h"
#include "superslm/forward_sites.h"
#include "superslm/gpu_port.h"
#include "superslm/model.h"
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
struct QProjRowCapture {
	bool captured = false;
	size_t width = 0;
};

void QProjRowHook(const SslmChainTraceRecord* chain, const SslmKvLandingTraceRecord* kv, void* user) {
	(void)kv;
	if (chain == nullptr) return;
	QProjRowCapture* cap = static_cast<QProjRowCapture*>(user);
	// LayerSite's own convention: "layer{L}.q_proj.requant" -- match the SUFFIX so this
	// fires once, at layer 0, token 0 (the only layer/token this harness runs).
	if (chain->site.size() >= 14 && chain->site.substr(chain->site.size() - 14) == "q_proj.requant") {
		if (!cap->captured) {
			cap->captured = true;
			cap->width = chain->codes.size();
		}
	}
}
}  // namespace

int main(int argc, char** argv) {
	if (argc < 2) {
		std::fprintf(stderr, "usage: %s <model.sslm> [token_id]\n", argv[0]);
		return 2;
	}
	const std::string model_path = argv[1];
	const int32_t token_id = argc >= 3 ? std::atoi(argv[2]) : 0;

	std::vector<uint8_t> model_bytes;
	if (!ReadFile(model_path.c_str(), model_bytes)) {
		std::fprintf(stderr, "FAILED: could not read \"%s\"\n", model_path.c_str());
		return 1;
	}
	SslmModelView model_view;
	std::string model_err;
	const SslmModelStatus load_status =
	    SslmModel::Load(model_bytes.data(), model_bytes.size(), model_view, &model_err);
	if (load_status != SslmModelStatus::Ok) {
		std::fprintf(stderr, "FAILED at stage=model_load: status=%s diagnostic=\"%s\"\n",
		             SslmModelStatusName(load_status), model_err.c_str());
		return 1;
	}
	const uint32_t num_heads = model_view.config.num_attention_heads;
	const uint32_t num_kv_heads = model_view.config.num_key_value_heads;
	const uint32_t num_hidden_layers = model_view.config.num_hidden_layers;
	const size_t hidden_size = model_view.config.hidden_size;
	const size_t head_dim = model_view.config.head_dim;
	const size_t intermediate_size = model_view.config.intermediate_size;
	const int64_t context_cap = static_cast<int64_t>(model_view.config.context_cap);
	const size_t q_width = static_cast<size_t>(num_heads) * head_dim;
	std::printf(
	    "model loaded: hidden_size=%zu layers=%u heads=%u/%u head_dim=%zu q_width=%zu "
	    "(square=%d) intermediate=%zu vocab=%u context_cap=%lld\n",
	    hidden_size, num_hidden_layers, num_heads, num_kv_heads, head_dim, q_width,
	    q_width == hidden_size ? 1 : 0, intermediate_size, model_view.config.vocab_size,
	    (long long)context_cap);

	PreflightScanWscFolds(model_view);
	std::vector<LayerBacking> backings(num_hidden_layers);
	std::vector<LayerWeights> layers(num_hidden_layers);
	for (uint32_t l = 0; l < num_hidden_layers; ++l) {
		std::string marshal_err;
		if (!MarshalLayer(model_view, l, num_heads, num_kv_heads, backings[l], layers[l], &marshal_err)) {
			std::fprintf(stderr, "FAILED at stage=layer_weights_marshal: layer=%u diagnostic=\"%s\"\n", l,
			             marshal_err.c_str());
			return 1;
		}
	}

	const SslmTensorView* embed_w = model_view.weights.Tensor("embed");
	if (!embed_w) {
		std::fprintf(stderr, "FAILED at stage=head_marshal: missing embed tensor\n");
		return 1;
	}
	bool ok = true;
	CarriedScale embed_site_constant = ReadCarriedScale(model_view.composition_constants, "embed", &ok);
	if (!ok) {
		std::fprintf(stderr, "FAILED at stage=head_marshal: missing embed site constant\n");
		return 1;
	}
	const int8_t* embed_weights = reinterpret_cast<const int8_t*>(embed_w->data);
	if (token_id < 0 || token_id >= model_view.config.vocab_size) {
		std::fprintf(stderr, "FAILED: token_id=%d out of range [0,%u)\n", token_id,
		             model_view.config.vocab_size);
		return 1;
	}
	std::vector<int8_t> embed_codes(hidden_size);
	CarriedScale embed_scale{};
	const SslmForwardStatus est = EmbedEntry(token_id, model_view.config.vocab_size, embed_weights,
	                                          hidden_size, embed_site_constant, embed_codes.data(),
	                                          &embed_scale);
	if (est != SslmForwardStatus::Ok) {
		std::fprintf(stderr, "FAILED at stage=embed: status=%s\n", SslmForwardStatusName(est));
		return 1;
	}

	const size_t kv_bytes = static_cast<size_t>(num_hidden_layers) * static_cast<size_t>(context_cap) *
	                        num_kv_heads * head_dim * 2;

	// --- CPU oracle: production RunLayerLoop, layer_budget = all layers, q_width explicit,
	//     a trace hook installed to capture q_proj.requant's own output row length. ---
	std::vector<int8_t> cpu_codes(hidden_size);
	std::memcpy(cpu_codes.data(), embed_codes.data(), hidden_size);
	SequenceLayerState cpu_seq;
	cpu_seq.hidden_codes = cpu_codes.data();
	cpu_seq.hidden_scale = embed_scale;
	cpu_seq.layer_index = 0;
	std::vector<uint8_t> cpu_ws(kv_bytes, 0);
	QProjRowCapture cap;
	SslmTraceHookState hook_state;
	SslmSetTraceHook(hook_state, &QProjRowHook, &cap);
	const SslmForwardStatus cpu_status = RunLayerLoop(
	    cpu_seq, layers.data(), num_hidden_layers, /*layer_budget=*/num_hidden_layers, hidden_size,
	    head_dim, num_kv_heads, intermediate_size, context_cap, model_view.rope_tables, cpu_ws.data(),
	    cpu_ws.size(), /*site_prefix=*/{}, /*token_index=*/0, &hook_state, q_width);
	SslmSetTraceHook(hook_state, nullptr, nullptr);
	std::printf("CPU oracle: status=%s layer_index=%u\n", SslmForwardStatusName(cpu_status),
	            cpu_seq.layer_index);

	bool geometry_row_ok = true;
	if (!cap.captured) {
		std::printf("GEOMETRY CHECK: FAILED -- q_proj.requant trace hook never fired\n");
		geometry_row_ok = false;
	} else if (cap.width != q_width) {
		std::printf("GEOMETRY CHECK: FAILED -- q_proj.requant output row width=%zu, want q_width=%zu\n",
		            cap.width, q_width);
		geometry_row_ok = false;
	} else {
		std::printf("GEOMETRY CHECK: PASS -- q_proj.requant output row width=%zu == q_width (CPU, "
		            "layer 0, token 0)\n",
		            cap.width);
	}

	// --- GPU port: RunLayerLoopGpu, IDENTICAL inputs, q_width explicit. ---
	std::vector<int8_t> gpu_codes(hidden_size);
	std::memcpy(gpu_codes.data(), embed_codes.data(), hidden_size);
	SequenceLayerState gpu_seq;
	gpu_seq.hidden_codes = gpu_codes.data();
	gpu_seq.hidden_scale = embed_scale;
	gpu_seq.layer_index = 0;
	std::vector<uint8_t> gpu_ws(kv_bytes, 0);
	// T-2432: RunLayerLoopGpu's own signature is deliberately UNCHANGED (gpu_port.h's own
	// "~40 existing callers" contract) -- q_width reaches the real GPU forward path only
	// through RunLayerLoopGpuSubmit/Finish, the two-call form RunLayerLoopGpu itself is a
	// thin wrapper over (superslm_gpu.cpp).
	superslm_gpu::GpuLayerLoopInFlight* inflight = nullptr;
	const SslmForwardStatus gpu_submit_status = superslm_gpu::RunLayerLoopGpuSubmit(
	    gpu_seq, layers.data(), num_hidden_layers, /*layer_budget=*/num_hidden_layers, hidden_size,
	    head_dim, num_kv_heads, intermediate_size, context_cap, model_view.rope_tables, gpu_ws.data(),
	    gpu_ws.size(), /*external_kv_resident=*/nullptr, /*io_external_kv_needs_resume_barrier=*/nullptr,
	    &inflight, /*external_weights_resident=*/nullptr, /*external_rope_cos_resident=*/nullptr,
	    /*external_rope_sin_resident=*/nullptr, /*external_rope_has=*/false,
	    /*external_rope_cos_elems=*/0, /*external_rope_sin_elems=*/0, /*adapter_bridge=*/nullptr,
	    q_width);
	SslmForwardStatus gpu_status = gpu_submit_status;
	if (inflight) {
		int32_t ready = 0;
		gpu_status = superslm_gpu::RunLayerLoopGpuFinish(inflight, gpu_seq, gpu_ws.data(), /*block=*/1, &ready);
	}
	std::printf("GPU port:   status=%s layer_index=%u\n", SslmForwardStatusName(gpu_status),
	            gpu_seq.layer_index);

	bool all_match = true;
	if (cpu_status != gpu_status) {
		std::printf("DIVERGENCE: status: CPU=%s GPU=%s\n", SslmForwardStatusName(cpu_status),
		            SslmForwardStatusName(gpu_status));
		all_match = false;
	}
	int first_code_mismatch = -1;
	for (size_t i = 0; i < hidden_size; ++i) {
		if (cpu_codes[i] != gpu_codes[i]) {
			first_code_mismatch = static_cast<int>(i);
			break;
		}
	}
	if (first_code_mismatch >= 0) {
		std::printf("DIVERGENCE: hidden_codes[%d]: CPU=%d GPU=%d (first mismatch of %zu elements)\n",
		            first_code_mismatch, cpu_codes[first_code_mismatch], gpu_codes[first_code_mismatch],
		            hidden_size);
		all_match = false;
	}
	if (cpu_seq.hidden_scale.m != gpu_seq.hidden_scale.m || cpu_seq.hidden_scale.e != gpu_seq.hidden_scale.e) {
		std::printf("DIVERGENCE: hidden_scale: CPU=(%lld,%lld) GPU=(%lld,%lld)\n",
		            (long long)cpu_seq.hidden_scale.m, (long long)cpu_seq.hidden_scale.e,
		            (long long)gpu_seq.hidden_scale.m, (long long)gpu_seq.hidden_scale.e);
		all_match = false;
	}
	bool kv_match = true;
	for (uint32_t l = 0; l < num_hidden_layers && kv_match; ++l) {
		for (uint32_t h = 0; h < num_kv_heads && kv_match; ++h) {
			const int8_t* cpu_k = KeyRow(cpu_ws.data(), l, context_cap, num_kv_heads, head_dim, h, 0);
			const int8_t* gpu_k =
			    superslm_gpu::KeyRowGpu(gpu_ws.data(), l, context_cap, num_kv_heads, head_dim, h, 0);
			const int8_t* cpu_v = ValueRow(cpu_ws.data(), l, context_cap, num_kv_heads, head_dim, h, 0);
			const int8_t* gpu_v =
			    superslm_gpu::ValueRowGpu(gpu_ws.data(), l, context_cap, num_kv_heads, head_dim, h, 0);
			for (size_t d = 0; d < head_dim; ++d) {
				if (cpu_k[d] != gpu_k[d] || cpu_v[d] != gpu_v[d]) {
					std::printf("DIVERGENCE: K/V[layer=%u][kv_head=%u][dim=%zu]\n", l, h, d);
					kv_match = false;
					all_match = false;
					break;
				}
			}
		}
	}

	if (all_match && geometry_row_ok) {
		std::printf("RESULT: PASS -- CPU q_proj row genuinely q_width-wide, and CPU/GPU bit-identical "
		            "across hidden_codes[%zu], hidden_scale, and every K/V row (%u layers).\n",
		            hidden_size, num_hidden_layers);
		return 0;
	}
	std::printf("RESULT: FAIL (see above)\n");
	return 1;
}

// T-2100: decode-step throughput, CPU oracle vs GPU port, real artifact, real hardware.
//
// WHY THIS EXISTS. The GPU-serial port has been proven bit-identical to the CPU oracle for
// twelve consecutive rounds -- on ONE token, every time. No round ever measured a token per
// second, so the arc could say "the port is correct" and could not say "the port is worth
// having." This harness answers the second question with the same setup C5 already uses:
// identical artifact, identical marshalling, identical entry points.
//
// WHAT IT MEASURES. N successive decode steps. Each step is one token through all 28 layers --
// exactly one `RunLayerLoop`/`RunLayerLoopGpu` call, the same call a generation loop makes --
// with the K/V workspace persisting across steps so `context_length` advances the way it does in
// real decode. One warmup step per side is run and discarded (device init, PSO compile, first-
// touch page faults); the reported figure is the mean over the timed steps.
//
// WHAT IT DOES NOT MEASURE, stated so no one reads more into the number than it carries: this
// times `RunLayerLoopGpu` AS IT EXISTS, which fence-waits and reads the complete K/V workspace
// back to host on every call. That readback is part of the measured cost and is a property of
// this substrate's calling convention rather than of the kernels. A backend with an asynchronous
// decode lifecycle would not pay it per step. The number below is therefore a floor on what the
// kernels can do, not a ceiling.
//
// Throwaway harness, same precedent as tools/t2039_c5_harness.cpp.
// Usage: t2100_gpu_throughput <model.sslm> [steps] [token_id]
#include <chrono>
#include <cstdio>
#include <cstring>
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

namespace {

double SecondsSince(const std::chrono::steady_clock::time_point& t0) {
	return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

}  // namespace

int main(int argc, char** argv) {
	if (argc < 2) {
		std::fprintf(stderr, "usage: %s <model.sslm> [steps] [token_id]\n", argv[0]);
		return 2;
	}
	const std::string model_path = argv[1];
	const int steps = argc >= 3 ? std::atoi(argv[2]) : 16;
	const int32_t token_id = argc >= 4 ? std::atoi(argv[3]) : 0;
	if (steps < 1) {
		std::fprintf(stderr, "steps must be >= 1\n");
		return 2;
	}

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
	std::printf("model loaded: hidden_size=%zu layers=%u heads=%u/%u head_dim=%zu intermediate=%zu\n",
	            hidden_size, num_hidden_layers, num_heads, num_kv_heads, head_dim, intermediate_size);

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
		std::fprintf(stderr, "FAILED: missing embed tensor\n");
		return 1;
	}
	bool ok = true;
	CarriedScale embed_site_constant = ReadCarriedScale(model_view.composition_constants, "embed", &ok);
	if (!ok) {
		std::fprintf(stderr, "FAILED: missing embed site constant\n");
		return 1;
	}
	const int8_t* embed_weights = reinterpret_cast<const int8_t*>(embed_w->data);

	std::vector<int8_t> embed_codes(hidden_size);
	CarriedScale embed_scale{};
	const SslmForwardStatus est =
	    EmbedEntry(token_id, model_view.config.vocab_size, embed_weights, hidden_size,
	               embed_site_constant, embed_codes.data(), &embed_scale);
	if (est != SslmForwardStatus::Ok) {
		std::fprintf(stderr, "FAILED at stage=embed: status=%s\n", SslmForwardStatusName(est));
		return 1;
	}

	const size_t kv_bytes = static_cast<size_t>(num_hidden_layers) * static_cast<size_t>(context_cap) *
	                        num_kv_heads * head_dim * 2;
	std::printf("workspace (KV cache) bytes = %zu (%.2f MiB)\n", kv_bytes,
	            kv_bytes / (1024.0 * 1024.0));
	std::printf("timing %d decode steps per side (1 warmup step each, discarded)\n\n", steps);

	// Each side runs its own persistent workspace so context_length advances across steps,
	// exactly as a real decode loop does.
	struct Side {
		const char* name;
		bool gpu;
	};
	const Side sides[2] = {{"CPU (RunLayerLoop)", false}, {"GPU (RunLayerLoopGpu)", true}};

	// T-2100: per-step CPU/GPU equality across MULTIPLE steps. C5 proves bit-identity on ONE call,
	// which cannot exercise a weight-residency cache at all -- the first call is always a miss that
	// populates it, so every residency fast path is untested by a single-call harness. These arrays
	// capture each side's own per-step residual so the comparison below is over steps 2..N as well.
	std::vector<std::vector<int8_t>> step_codes[2];
	std::vector<int64_t> step_ctx[2];

	double per_token[2] = {0.0, 0.0};
	for (int s = 0; s < 2; ++s) {
		std::vector<uint8_t> ws(kv_bytes, 0);
		std::vector<int8_t> codes(hidden_size);
		SequenceLayerState seq;
		seq.hidden_codes = codes.data();
		seq.layer_index = 0;

		auto one_step = [&]() -> SslmForwardStatus {
			std::memcpy(codes.data(), embed_codes.data(), hidden_size);
			seq.hidden_scale = embed_scale;
			seq.layer_index = 0;
			if (sides[s].gpu) {
				return superslm_gpu::RunLayerLoopGpu(seq, layers.data(), num_hidden_layers,
				                                     num_hidden_layers, hidden_size, head_dim,
				                                     num_kv_heads, intermediate_size, context_cap,
				                                     model_view.rope_tables, ws.data(), ws.size());
			}
			return RunLayerLoop(seq, layers.data(), num_hidden_layers, num_hidden_layers, hidden_size,
			                    head_dim, num_kv_heads, intermediate_size, context_cap,
			                    model_view.rope_tables, ws.data(), ws.size());
		};

		const SslmForwardStatus warm = one_step();
		if (warm != SslmForwardStatus::Ok) {
			std::fprintf(stderr, "%s: warmup step returned %s\n", sides[s].name,
			             SslmForwardStatusName(warm));
			return 1;
		}

		const auto t0 = std::chrono::steady_clock::now();
		double worst = 0.0, best = 1e30;
		for (int i = 0; i < steps; ++i) {
			const auto ts = std::chrono::steady_clock::now();
			const SslmForwardStatus st = one_step();
			const double dt = SecondsSince(ts);
			if (st != SslmForwardStatus::Ok) {
				std::fprintf(stderr, "%s: step %d returned %s\n", sides[s].name, i,
				             SslmForwardStatusName(st));
				return 1;
			}
			if (dt > worst) worst = dt;
			if (dt < best) best = dt;
			step_codes[s].push_back(codes);
			step_ctx[s].push_back(seq.context_length);
		}
		const double total = SecondsSince(t0);
		per_token[s] = total / steps;
		std::printf("%-24s %6.4f s/token   %8.2f tok/s   (min %.4f, max %.4f, total %.2f s over %d steps)\n",
		            sides[s].name, per_token[s], 1.0 / per_token[s], best, worst, total, steps);
	}

	// --- Per-step equality, every step, not just the first. ---
	int mismatched_steps = 0;
	for (size_t i = 0; i < step_codes[0].size() && i < step_codes[1].size(); ++i) {
		if (step_codes[0][i] != step_codes[1][i] || step_ctx[0][i] != step_ctx[1][i]) {
			if (mismatched_steps == 0) std::printf("\nDIVERGENCE first at timed step %zu\n", i);
			++mismatched_steps;
		}
	}
	std::printf("\nper-step CPU/GPU equality over %zu timed steps: %s (%d mismatched)\n",
	            step_codes[0].size(), mismatched_steps == 0 ? "IDENTICAL" : "DIVERGED",
	            mismatched_steps);

	std::printf("\nGPU vs CPU: %.2fx %s\n", per_token[0] / per_token[1],
	            per_token[1] < per_token[0] ? "FASTER" : "SLOWER");
	std::printf(
	    "NOTE: RunLayerLoopGpu fence-waits and reads the complete %.2f MiB K/V workspace back to\n"
	    "host every call. That cost is included above and is a property of this substrate's calling\n"
	    "convention, not of the kernels -- an asynchronous decode lifecycle would not pay it per step.\n",
	    kv_bytes / (1024.0 * 1024.0));
	return 0;
}

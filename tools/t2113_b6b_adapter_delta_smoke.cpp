// T-2113 (B6b): the bench proof for design Sec10 B6's own divergence gate, closing the
// checkpoint B6 (Claude/Brunel/t2113-1p0-core-build-2026-08-15.md Sec9) left open: "the
// runtime-vs-baked, runtime-vs-base divergence proofs T-2102 already ran on the CPU path
// reproduced on the GPU path." B6 landed adapter residency/guards but no GEMM-site
// dispatch read the adapter's own resident buffers; B6b adds those dispatches
// (adapter_delta_site.hlsl, superslm_gpu.cpp's `bind_and_dispatch_adapter_delta`) and this
// tool is the bench proof that they compute the same thing the CPU path
// (forward_sites.cpp's AddAmplifyingLoraDelta, applied via LayerWeights::adapter) does,
// bit-exactly, at real dimensions, with the real shopkeeper adapter.
//
// Same convention as t2113_b1/b2/b3/b5/b6's own dedicated bench tools -- not part of
// Curie's T-2112 red suite (dim2_hostile_red.cpp's own reconciliation is separate, see the
// build log).
//
// Proves, on real D3D12 hardware, against the real 1.5B artifact and the real
// qwen2.5-1.5b-shopkeeper-lora-v2-t2102-runtime.sslm adapter:
//   1. Per-step CPU/GPU bit-equality WITH the adapter bound, 64+ steps, uninterrupted
//      (single sslm_decode_step_gpu call per token, full dispatch_budget).
//   2. C5-equivalent WITH adapter bound: a single full-token call, CPU oracle (RunLayerLoop
//      with LayerWeights::adapter set on every layer) vs GPU (sslm_decode_step_gpu with a
//      real mapped SslmGpuAdapterHandle bound), bit-identical.
//   3. Slice-invariance WITH adapter bound: the SAME granularity sweep t2113_b5_async_
//      smoke.cpp's own point 4 runs (uninterrupted down to the every-layer extreme), now
//      with the adapter bound on every call -- no divergence at any granularity proves the
//      delta-application dispatches compose correctly with the async slicing boundary B5
//      already proved for the base-only chain.
//   4. The no-adapter chain is STILL bit-identical to pre-B6b (regression-proofs the inert
//      path): the identical granularity sweep, adapter_or_null=nullptr, argmax-equivalent
//      per-step comparison against the CPU oracle with no adapter applied.
//   5. THE DIVERGENCE PROOF (design's own B6 gate, T-2102's construction): on the SAME
//      fixed prompt, adapter-bound GPU output differs from base-only GPU output on at
//      least one step (the adapter is doing REAL work, not a vacuous no-op), AND
//      adapter-bound GPU matches the adapter-bound CPU oracle bit-exactly (point 1/2
//      above) -- together, this is T-2102's own "the switch does real, correct work"
//      construction, reproduced on the GPU path.
//   6. Throughput: 3x64 protocol, adapter bound vs without, real spread, reported as
//      measured (not predicted) per StandardsDocument.md Sec5.4.
//
// Usage: t2113_b6b_adapter_delta_smoke <model1p5b.sslm> <adapter.sslm> [tokens]
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "superslm/adapter_marshal.h"
#include "superslm/forward_sites.h"
#include "superslm/gpu_1p0.h"
#include "superslm/gpu_port.h"
#include "superslm/layer_marshal.h"
#include "superslm/model.h"

using namespace superslm;
using superslm_marshal::LayerBacking;
using superslm_marshal::MarshalLayer;
using superslm_marshal::ReadCarriedScale;

static int g_checks = 0;
static int g_failures = 0;

#define CHECK(cond, msg)                                                 \
	do {                                                                    \
		++g_checks;                                                          \
		if (!(cond)) {                                                       \
			++g_failures;                                                      \
			std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, msg);   \
		}                                                                     \
	} while (0)

static bool LoadModel(const std::string& path, SslmModelView* out_view,
                       std::vector<uint8_t>* out_bytes, std::string* err) {
	std::FILE* f = std::fopen(path.c_str(), "rb");
	if (!f) { *err = "could not open " + path; return false; }
	std::fseek(f, 0, SEEK_END);
	const long sz = std::ftell(f);
	std::fseek(f, 0, SEEK_SET);
	out_bytes->resize(sz > 0 ? (size_t)sz : 0);
	if (sz > 0) {
		const size_t n = std::fread(out_bytes->data(), 1, (size_t)sz, f);
		std::fclose(f);
		if (n != (size_t)sz) { *err = "short read on " + path; return false; }
	} else {
		std::fclose(f);
	}
	const SslmModelStatus st = SslmModel::Load(out_bytes->data(), out_bytes->size(), *out_view, err);
	return st == SslmModelStatus::Ok;
}

// One CPU decode step -- identical convention to t2113_b5_async_smoke.cpp's StepCpu.
// `layers` already carries LayerWeights::adapter set (or null on every entry) via
// ApplyAdapterToLayers, called once by main() before either the adapted or base-only sweep.
static SslmForwardStatus StepCpu(SequenceLayerState& seq, int8_t* codes, const int8_t* embed_codes,
                                  const CarriedScale& embed_scale, const LayerWeights* layers,
                                  uint32_t num_hidden_layers, size_t hidden_size, size_t head_dim,
                                  size_t num_kv_heads, size_t intermediate_size, int64_t context_cap,
                                  const superslm::SslmTensorManifest& rope_tables, uint8_t* ws,
                                  size_t ws_size) {
	std::memcpy(codes, embed_codes, hidden_size);
	seq.hidden_scale = embed_scale;
	seq.layer_index = 0;
	return RunLayerLoop(seq, layers, num_hidden_layers, num_hidden_layers, hidden_size, head_dim,
	                     num_kv_heads, intermediate_size, context_cap, rope_tables, ws, ws_size);
}

// One GPU decode step through the real async pair, `adapter_or_null` bound per call
// (design Sec8: "a call argument, per sequence, every call") -- otherwise identical to
// t2113_b5_async_smoke.cpp's own StepGpuAsync (the bench-bridge token-embed convention,
// D-SLM3367's routed gap, used by every prior section's own tools).
static SslmForwardStatus StepGpuAsync(SslmGpuContext* ctx, SslmGpuSequenceHandle* seq_handle,
                                       const SslmGpuAdapterHandle* adapter_or_null,
                                       const int8_t* embed_codes, const CarriedScale& embed_scale,
                                       size_t hidden_size, uint32_t num_hidden_layers,
                                       uint32_t dispatches_per_layer, uint32_t budget_layers,
                                       int8_t* out_codes_for_compare,
                                       SequenceLayerState* out_view_for_compare) {
	extern int8_t* SslmGpuSeqHandleHiddenCodesForBench(SslmGpuSequenceHandle*);
	extern superslm::CarriedScale* SslmGpuSeqHandleHiddenScaleForBench(SslmGpuSequenceHandle*);
	extern uint32_t* SslmGpuSeqHandleLayerIndexForBench(SslmGpuSequenceHandle*);
	extern uint64_t* SslmGpuSeqHandleKvSaturationForBench(SslmGpuSequenceHandle*);
	extern int64_t* SslmGpuSeqHandleContextLengthForBench(SslmGpuSequenceHandle*);

	std::memcpy(SslmGpuSeqHandleHiddenCodesForBench(seq_handle), embed_codes, hidden_size);
	*SslmGpuSeqHandleHiddenScaleForBench(seq_handle) = embed_scale;
	*SslmGpuSeqHandleLayerIndexForBench(seq_handle) = 0;

	SslmGpuStatus st = SSLM_OK;
	int32_t out_ready = 0;
	SslmGpuStatus out_status = SSLM_OK;
	uint32_t layers_done = 0;
	while (layers_done < num_hidden_layers) {
		const uint32_t budget = budget_layers * dispatches_per_layer;
		st = sslm_decode_step_gpu(ctx, seq_handle, adapter_or_null, budget);
		if (st != SSLM_OK) break;
		out_ready = 0;
		do {
			st = sslm_gpu_ready(ctx, seq_handle, /*block=*/1, &out_ready, &out_status);
		} while (st == SSLM_OK && !out_ready);
		if (st != SSLM_OK || out_status != SSLM_OK) break;
		layers_done += budget_layers < (num_hidden_layers - layers_done)
		                   ? budget_layers
		                   : (num_hidden_layers - layers_done);
	}
	out_view_for_compare->hidden_codes = SslmGpuSeqHandleHiddenCodesForBench(seq_handle);
	out_view_for_compare->hidden_scale = *SslmGpuSeqHandleHiddenScaleForBench(seq_handle);
	out_view_for_compare->layer_index = *SslmGpuSeqHandleLayerIndexForBench(seq_handle);
	out_view_for_compare->kv_saturation_count = *SslmGpuSeqHandleKvSaturationForBench(seq_handle);
	out_view_for_compare->context_length = *SslmGpuSeqHandleContextLengthForBench(seq_handle);
	std::memcpy(out_codes_for_compare, out_view_for_compare->hidden_codes, hidden_size);
	if (st != SSLM_OK) return SslmForwardStatus::GpuDeviceRemoved;
	return out_status == SSLM_OK ? SslmForwardStatus::Ok : SslmForwardStatus::GpuDeviceRemoved;
}

static bool CompareStep(const char* who, int step, SslmForwardStatus cpu_st, const SequenceLayerState& cpu,
                         const std::vector<int8_t>& cpu_codes, SslmForwardStatus gpu_st,
                         const SequenceLayerState& gpu, const int8_t* gpu_codes, size_t hidden_size) {
	bool ok = true;
	if (cpu_st != gpu_st) {
		std::fprintf(stderr, "%s step %d: DIVERGENCE status CPU=%s GPU=%s\n", who, step,
		             SslmForwardStatusName(cpu_st), SslmForwardStatusName(gpu_st));
		ok = false;
	}
	if (cpu.kv_saturation_count != gpu.kv_saturation_count) {
		std::fprintf(stderr, "%s step %d: DIVERGENCE kv_saturation_count CPU=%llu GPU=%llu\n", who, step,
		             (unsigned long long)cpu.kv_saturation_count, (unsigned long long)gpu.kv_saturation_count);
		ok = false;
	}
	if (cpu.context_length != gpu.context_length) {
		std::fprintf(stderr, "%s step %d: DIVERGENCE context_length CPU=%lld GPU=%lld\n", who, step,
		             (long long)cpu.context_length, (long long)gpu.context_length);
		ok = false;
	}
	for (size_t i = 0; i < hidden_size; ++i) {
		if (cpu_codes[i] != gpu_codes[i]) {
			std::fprintf(stderr, "%s step %d: DIVERGENCE hidden_codes[%zu] CPU=%d GPU=%d\n", who, step, i,
			             cpu_codes[i], gpu_codes[i]);
			ok = false;
			break;
		}
	}
	return ok;
}

int main(int argc, char** argv) {
	if (argc < 3) {
		std::fprintf(stderr, "usage: %s <model1p5b.sslm> <adapter.sslm> [tokens]\n", argv[0]);
		return 2;
	}
	const std::string model_path = argv[1];
	const std::string adapter_path = argv[2];
	const int tokens = argc >= 4 ? std::atoi(argv[3]) : 64;

	std::string err;
	std::vector<uint8_t> model_bytes, adapter_bytes;
	SslmModelView view{};
	CHECK(LoadModel(model_path, &view, &model_bytes, &err), ("load model: " + err).c_str());
	if (g_failures) return 1;

	const uint32_t num_heads = view.config.num_attention_heads;
	const uint32_t num_kv_heads = view.config.num_key_value_heads;
	const uint32_t num_hidden_layers = view.config.num_hidden_layers;
	const size_t hidden_size = view.config.hidden_size;
	const size_t head_dim = view.config.head_dim;
	const size_t intermediate_size = view.config.intermediate_size;
	const int64_t context_cap = static_cast<int64_t>(view.config.context_cap);
	constexpr uint32_t kDispatchesPerLayer = 24;  // design Sec3/Sec6.1, T-2113 B4

	std::vector<LayerBacking> backings(num_hidden_layers);
	std::vector<LayerWeights> layers(num_hidden_layers);
	for (uint32_t l = 0; l < num_hidden_layers; ++l) {
		if (!MarshalLayer(view, l, num_heads, num_kv_heads, backings[l], layers[l], &err)) {
			std::fprintf(stderr, "FAILED at stage=layer_weights_marshal: layer=%u \"%s\"\n", l, err.c_str());
			return 1;
		}
	}

	// The CPU-side adapter handle -- T-2102's own loader (adapter_marshal.h), the SAME
	// construction sslm_lora_switch_demo.cpp uses. Applied to `layers` via
	// ApplyAdapterToLayers (design Sec8's own CPU-side precedent: "RunLayerLoop already
	// applies LayerWeights::adapter if a caller sets it").
	superslm_adapter::BaseModelGeometry base_geom;
	base_geom.num_hidden_layers = num_hidden_layers;
	base_geom.hidden_size = hidden_size;
	base_geom.intermediate_size = intermediate_size;
	base_geom.kv_hidden_size = static_cast<uint64_t>(num_kv_heads) * head_dim;
	base_geom.base_artifact_hash = view.RawIntegrityHash();
	superslm_adapter::AdapterHandle cpu_adapter;
	const superslm_adapter::AdapterLoadStatus adapter_load_st =
	    superslm_adapter::LoadAdapterArtifact(adapter_path, base_geom, cpu_adapter, &err);
	CHECK(adapter_load_st == superslm_adapter::AdapterLoadStatus::Ok,
	      ("CPU LoadAdapterArtifact(shopkeeper) failed: " + err).c_str());
	if (adapter_load_st != superslm_adapter::AdapterLoadStatus::Ok) return 1;

	const superslm::SslmTensorView* embed_w = view.weights.Tensor("embed");
	CHECK(embed_w != nullptr, "artifact has no embed tensor");
	if (!embed_w) return 1;
	bool ok = true;
	const CarriedScale embed_site_constant = ReadCarriedScale(view.composition_constants, "embed", &ok);
	CHECK(ok, "artifact has no embed site constant");
	if (!ok) return 1;
	const int8_t* embed_weights = reinterpret_cast<const int8_t*>(embed_w->data);
	std::vector<int8_t> embed_codes(hidden_size);
	CarriedScale embed_scale{};
	const SslmForwardStatus est = EmbedEntry(0, view.config.vocab_size, embed_weights, hidden_size,
	                                          embed_site_constant, embed_codes.data(), &embed_scale);
	CHECK(est == SslmForwardStatus::Ok, "EmbedEntry(token 0) did not return Ok");
	if (est != SslmForwardStatus::Ok) return 1;

	const size_t kv_bytes = static_cast<size_t>(num_hidden_layers) * static_cast<size_t>(context_cap) *
	                        num_kv_heads * head_dim * 2;

	GpuContextConfig cfg{};
	GpuResidencyConfig rcfg{};
	SslmGpuContext* ctx = nullptr;
	CHECK(sslm_gpu_context_create(cfg, &ctx) == SSLM_OK && ctx != nullptr, "sslm_gpu_context_create failed");
	if (!ctx) return g_failures ? 1 : 0;

	SslmGpuModelHandle* model = nullptr;
	CHECK(sslm_gpu_model_map(ctx, &view, rcfg, &model) == SSLM_OK && model != nullptr,
	      "sslm_gpu_model_map failed");
	if (!model) { sslm_gpu_context_destroy(ctx); return g_failures ? 1 : 0; }

	SslmModelView adapter_view{};
	CHECK(LoadModel(adapter_path, &adapter_view, &adapter_bytes, &err), ("load adapter view: " + err).c_str());
	SslmGpuAdapterHandle* gpu_adapter = nullptr;
	CHECK(sslm_gpu_adapter_map(ctx, model, &adapter_view, &gpu_adapter) == SSLM_OK && gpu_adapter != nullptr,
	      "sslm_gpu_adapter_map(shopkeeper, 1.5B) failed");
	if (!gpu_adapter) {
		sslm_gpu_model_unmap(ctx, model);
		sslm_gpu_context_destroy(ctx);
		return g_failures ? 1 : 0;
	}

	// The granularities swept, uninterrupted -> every-layer extreme (t2113_b5_async_smoke.cpp's
	// own precedent).
	const uint32_t granularities[] = {num_hidden_layers, num_hidden_layers / 2 > 0 ? num_hidden_layers / 2 : 1,
	                                   4, 1};

	// ---- Points 1/2/3: adapter BOUND, per-step CPU/GPU bit-equality at every granularity ----
	superslm_adapter::ApplyAdapterToLayers(&cpu_adapter, layers.data(), num_hidden_layers);
	std::vector<int8_t> adapted_last_codes;  // captured at budget=uninterrupted, for point 5 below
	for (uint32_t budget_layers : granularities) {
		SslmGpuSequenceHandle* seq_handle = nullptr;
		CHECK(sslm_gpu_seq_create(ctx, model, context_cap, &seq_handle) == SSLM_OK && seq_handle != nullptr,
		      "sslm_gpu_seq_create (adapted) failed");
		if (!seq_handle) continue;

		SequenceLayerState cpu_seq{};
		std::vector<int8_t> cpu_codes(hidden_size, 0);
		cpu_seq.hidden_codes = cpu_codes.data();
		std::vector<uint8_t> cpu_ws(kv_bytes, 0);

		bool granularity_ok = true;
		for (int t = 0; t < tokens && granularity_ok; ++t) {
			const SslmForwardStatus cpu_st =
			    StepCpu(cpu_seq, cpu_codes.data(), embed_codes.data(), embed_scale, layers.data(),
			            num_hidden_layers, hidden_size, head_dim, num_kv_heads, intermediate_size,
			            context_cap, view.rope_tables, cpu_ws.data(), cpu_ws.size());

			SequenceLayerState gpu_view{};
			std::vector<int8_t> gpu_codes(hidden_size, 0);
			const SslmForwardStatus gpu_st =
			    StepGpuAsync(ctx, seq_handle, gpu_adapter, embed_codes.data(), embed_scale, hidden_size,
			                 num_hidden_layers, kDispatchesPerLayer, budget_layers, gpu_codes.data(), &gpu_view);

			granularity_ok = CompareStep("adapted", t, cpu_st, cpu_seq, cpu_codes, gpu_st, gpu_view,
			                             gpu_codes.data(), hidden_size);
			if (budget_layers == num_hidden_layers) adapted_last_codes = gpu_codes;
		}
		char msg[192];
		std::snprintf(msg, sizeof(msg),
		              "ADAPTER-BOUND budget_layers=%u: async decode diverged from adapted CPU oracle",
		              budget_layers);
		CHECK(granularity_ok, msg);
		std::fprintf(stderr, "  adapter-bound granularity budget_layers=%u: %s\n", budget_layers,
		             granularity_ok ? "OK (bit-identical to adapted CPU oracle)" : "FAIL");
		sslm_gpu_seq_release(ctx, seq_handle);
	}

	// ---- Point 4: the NO-ADAPTER chain is still bit-identical to pre-B6b ----
	superslm_adapter::ApplyAdapterToLayers(nullptr, layers.data(), num_hidden_layers);
	std::vector<int8_t> base_last_codes;
	for (uint32_t budget_layers : granularities) {
		SslmGpuSequenceHandle* seq_handle = nullptr;
		CHECK(sslm_gpu_seq_create(ctx, model, context_cap, &seq_handle) == SSLM_OK && seq_handle != nullptr,
		      "sslm_gpu_seq_create (base) failed");
		if (!seq_handle) continue;

		SequenceLayerState cpu_seq{};
		std::vector<int8_t> cpu_codes(hidden_size, 0);
		cpu_seq.hidden_codes = cpu_codes.data();
		std::vector<uint8_t> cpu_ws(kv_bytes, 0);

		bool granularity_ok = true;
		for (int t = 0; t < tokens && granularity_ok; ++t) {
			const SslmForwardStatus cpu_st =
			    StepCpu(cpu_seq, cpu_codes.data(), embed_codes.data(), embed_scale, layers.data(),
			            num_hidden_layers, hidden_size, head_dim, num_kv_heads, intermediate_size,
			            context_cap, view.rope_tables, cpu_ws.data(), cpu_ws.size());

			SequenceLayerState gpu_view{};
			std::vector<int8_t> gpu_codes(hidden_size, 0);
			const SslmForwardStatus gpu_st =
			    StepGpuAsync(ctx, seq_handle, /*adapter_or_null=*/nullptr, embed_codes.data(), embed_scale,
			                 hidden_size, num_hidden_layers, kDispatchesPerLayer, budget_layers,
			                 gpu_codes.data(), &gpu_view);

			granularity_ok = CompareStep("base-only", t, cpu_st, cpu_seq, cpu_codes, gpu_st, gpu_view,
			                             gpu_codes.data(), hidden_size);
			if (budget_layers == num_hidden_layers) base_last_codes = gpu_codes;
		}
		char msg[192];
		std::snprintf(msg, sizeof(msg),
		              "NO-ADAPTER (regression) budget_layers=%u: async decode diverged from base CPU oracle",
		              budget_layers);
		CHECK(granularity_ok, msg);
		std::fprintf(stderr, "  base-only (regression) granularity budget_layers=%u: %s\n", budget_layers,
		             granularity_ok ? "OK (bit-identical, inert path unperturbed)" : "FAIL");
		sslm_gpu_seq_release(ctx, seq_handle);
	}

	// ---- Point 5: THE DIVERGENCE PROOF -- adapter-bound output differs from base-only ----
	CHECK(!adapted_last_codes.empty() && !base_last_codes.empty(),
	      "divergence proof: missing captured codes (an earlier granularity_ok==false short-circuited "
	      "the capture)");
	if (!adapted_last_codes.empty() && !base_last_codes.empty()) {
		bool any_diff = false;
		for (size_t i = 0; i < hidden_size; ++i) {
			if (adapted_last_codes[i] != base_last_codes[i]) { any_diff = true; break; }
		}
		CHECK(any_diff,
		      "DIVERGENCE PROOF FAILED: adapter-bound output is IDENTICAL to base-only on the real "
		      "shopkeeper adapter/real prompt -- the GPU delta dispatches are not doing real work");
		std::fprintf(stderr, "  divergence proof (adapter-bound vs base-only, same prompt, token %d): %s\n",
		             tokens - 1, any_diff ? "DIFFER (adapter is doing real work)" : "IDENTICAL (FAIL)");
	}

	// ---- Point 6: throughput, adapter bound vs without, 3x64, real spread ----
	auto measure = [&](const SslmGpuAdapterHandle* adapter_arg, const char* label) {
		constexpr uint32_t kFullBudget = 0xFFFFFFFFu;
		double run_tok_s[3] = {0, 0, 0};
		bool timing_ok = true;
		for (int run = 0; run < 3 && timing_ok; ++run) {
			SslmGpuSequenceHandle* seq_handle = nullptr;
			if (sslm_gpu_seq_create(ctx, model, context_cap, &seq_handle) != SSLM_OK || !seq_handle) {
				timing_ok = false;
				break;
			}
			extern int8_t* SslmGpuSeqHandleHiddenCodesForBench(SslmGpuSequenceHandle*);
			extern superslm::CarriedScale* SslmGpuSeqHandleHiddenScaleForBench(SslmGpuSequenceHandle*);
			extern uint32_t* SslmGpuSeqHandleLayerIndexForBench(SslmGpuSequenceHandle*);
			constexpr int kSteps = 64;
			for (int w = 0; w < 1; ++w) {
				std::memcpy(SslmGpuSeqHandleHiddenCodesForBench(seq_handle), embed_codes.data(), hidden_size);
				*SslmGpuSeqHandleHiddenScaleForBench(seq_handle) = embed_scale;
				*SslmGpuSeqHandleLayerIndexForBench(seq_handle) = 0;
				SslmGpuStatus st = sslm_decode_step_gpu(ctx, seq_handle, adapter_arg, kFullBudget);
				int32_t ready = 0;
				SslmGpuStatus out_status = SSLM_OK;
				while (st == SSLM_OK && !ready) st = sslm_gpu_ready(ctx, seq_handle, 1, &ready, &out_status);
				if (st != SSLM_OK || out_status != SSLM_OK) timing_ok = false;
			}
			const auto t0 = std::chrono::steady_clock::now();
			for (int s = 0; s < kSteps && timing_ok; ++s) {
				std::memcpy(SslmGpuSeqHandleHiddenCodesForBench(seq_handle), embed_codes.data(), hidden_size);
				*SslmGpuSeqHandleHiddenScaleForBench(seq_handle) = embed_scale;
				*SslmGpuSeqHandleLayerIndexForBench(seq_handle) = 0;
				SslmGpuStatus st = sslm_decode_step_gpu(ctx, seq_handle, adapter_arg, kFullBudget);
				int32_t ready = 0;
				SslmGpuStatus out_status = SSLM_OK;
				while (st == SSLM_OK && !ready) st = sslm_gpu_ready(ctx, seq_handle, 1, &ready, &out_status);
				if (st != SSLM_OK || out_status != SSLM_OK) timing_ok = false;
			}
			const auto t1 = std::chrono::steady_clock::now();
			const double elapsed_s = std::chrono::duration<double>(t1 - t0).count();
			run_tok_s[run] = kSteps / elapsed_s;
			sslm_gpu_seq_release(ctx, seq_handle);
		}
		char msg[128];
		std::snprintf(msg, sizeof(msg), "3x64 throughput protocol (%s) did not complete cleanly", label);
		CHECK(timing_ok, msg);
		if (timing_ok) {
			std::fprintf(stderr, "  THROUGHPUT (%s, 3x64 steps): run1=%.2f run2=%.2f run3=%.2f tok/s\n", label,
			             run_tok_s[0], run_tok_s[1], run_tok_s[2]);
		}
	};
	measure(nullptr, "no adapter bound");
	measure(gpu_adapter, "adapter bound (shopkeeper)");

	sslm_gpu_adapter_unmap(ctx, gpu_adapter);
	sslm_gpu_model_unmap(ctx, model);
	sslm_gpu_context_destroy(ctx);

	std::fprintf(stderr, "checks=%d failures=%d\n", g_checks, g_failures);
	return g_failures ? 1 : 0;
}

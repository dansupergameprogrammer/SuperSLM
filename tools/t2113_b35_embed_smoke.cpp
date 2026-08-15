// T-2113 (B3.5): the bench proof for design Sec10 B3.5's own delivery -- the production
// token-feed entry point (design Sec5.3a, mini-fold 2026-08-15, routing D-SLM3367):
// `sslm_gpu_seq_embed_token(ctx, seq, token_id)`.
//
// This closes D-SLM3367 (build log Sec8.3/design Sec18): the 1.0 API had no way to put a
// real token's content into a fresh sequence's own hidden_codes/hidden_scale before the
// first `sslm_decode_step_gpu` call, so every decode-driving red-suite cell and every
// prior B-section's own bench tool had to reach around the gap via the non-shipping
// `gpu_1p0_bench_bridge.h`. This tool drives the two new design Sec11 cells against real
// D3D12 hardware and a real 1.5B artifact, through the DECLARED public surface only --
// never the bench bridge:
//
//   1. Dim 2 (trust boundaries), mechanism: a hostile token_id (negative, and
//      >= vocab_size) returns SSLM_TOKEN_ID_OUT_OF_RANGE, and the sequence's own state
//      (hidden_codes/hidden_scale/layer_index) is left BYTE-FOR-BYTE untouched -- proven
//      by embedding a real token first, snapshotting the handle's own state via the B3
//      bench-bridge READ-ONLY accessors (comparison only, never a write), then issuing
//      the hostile call and re-snapshotting.
//   2. Dim 6 (numerical edges and determinism), product -- the conductor's own named
//      "obvious product cell": a real multi-token prefill, driven entirely through
//      sslm_gpu_seq_embed_token + sslm_decode_step_gpu in the CPU path's own per-token
//      pairing (embed, then run every layer -- RunWholeToken's own shape,
//      forward_sites.cpp), compared per-step, bit-for-bit, against the SAME prefill's own
//      CPU oracle (EmbedEntry + RunLayerLoop, run fully serially from an identical cold
//      start). No divergence at any prefill position proves the production embed+decode
//      call pair reproduces the CPU path's own per-token content exactly.
//
// Usage: t2113_b35_embed_smoke <model1p5b.sslm> [extra_decode_tokens]  (exits 0 on pass)
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "superslm/forward_sites.h"
#include "superslm/gpu_1p0.h"
#include "superslm/gpu_port.h"
#include "superslm/layer_marshal.h"
#include "superslm/model.h"

using superslm::CarriedScale;
using superslm::EmbedEntry;
using superslm::LayerWeights;
using superslm::RunLayerLoop;
using superslm::SequenceLayerState;
using superslm::SslmForwardStatus;
using superslm::SslmForwardStatusName;
using superslm::SslmModelStatus;
using superslm::SslmModelView;
using superslm_marshal::LayerBacking;
using superslm_marshal::MarshalLayer;
using superslm_marshal::ReadCarriedScale;

static int g_checks = 0;
static int g_failures = 0;

#define CHECK(cond, msg)                                                    \
	do {                                                                       \
		++g_checks;                                                             \
		if (!(cond)) {                                                          \
			++g_failures;                                                         \
			std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, msg);      \
		}                                                                        \
	} while (0)

// Read-only bench-bridge accessors (gpu_1p0_bench_bridge.h) -- used here ONLY to observe
// a sequence handle's own host-mirrored state for comparison, never to write into it (the
// whole point of this tool is that sslm_gpu_seq_embed_token, the DECLARED surface, is the
// only write path exercised).
extern "C" {}
extern int8_t* SslmGpuSeqHandleHiddenCodesForBench(SslmGpuSequenceHandle*);
extern superslm::CarriedScale* SslmGpuSeqHandleHiddenScaleForBench(SslmGpuSequenceHandle*);
extern uint32_t* SslmGpuSeqHandleLayerIndexForBench(SslmGpuSequenceHandle*);
extern uint64_t* SslmGpuSeqHandleKvSaturationForBench(SslmGpuSequenceHandle*);
extern int64_t* SslmGpuSeqHandleContextLengthForBench(SslmGpuSequenceHandle*);
extern size_t SslmGpuSeqHandleHiddenSizeForBench(SslmGpuSequenceHandle*);

static bool LoadModel(const std::string& path, SslmModelView* out_view,
                       std::vector<uint8_t>* out_bytes) {
	std::FILE* f = std::fopen(path.c_str(), "rb");
	if (!f) {
		std::fprintf(stderr, "could not open %s\n", path.c_str());
		return false;
	}
	std::fseek(f, 0, SEEK_END);
	const long sz = std::ftell(f);
	std::fseek(f, 0, SEEK_SET);
	out_bytes->resize(sz > 0 ? static_cast<size_t>(sz) : 0);
	if (sz > 0) {
		const size_t n = std::fread(out_bytes->data(), 1, static_cast<size_t>(sz), f);
		std::fclose(f);
		if (n != static_cast<size_t>(sz)) {
			std::fprintf(stderr, "short read on %s\n", path.c_str());
			return false;
		}
	} else {
		std::fclose(f);
	}
	std::string err;
	const SslmModelStatus st =
	    superslm::SslmModel::Load(out_bytes->data(), out_bytes->size(), *out_view, &err);
	if (st != SslmModelStatus::Ok) {
		std::fprintf(stderr, "model load failed for %s: %s\n", path.c_str(), err.c_str());
		return false;
	}
	return true;
}

// One CPU decode step, the oracle: EmbedEntry(token) then RunLayerLoop over every layer --
// the identical RunWholeToken shape (forward_sites.cpp), called with the same trailing
// defaults sslm_gpu_seq_embed_token's own production body uses.
static SslmForwardStatus StepCpu(SequenceLayerState& seq, int32_t token, int32_t vocab_size,
                                  const int8_t* embed_weights, size_t hidden_size,
                                  const CarriedScale& embed_site_constant, const LayerWeights* layers,
                                  uint32_t num_hidden_layers, size_t head_dim, size_t num_kv_heads,
                                  size_t intermediate_size, int64_t context_cap,
                                  const superslm::SslmTensorManifest& rope_tables, uint8_t* ws,
                                  size_t ws_size) {
	std::vector<int8_t> embed_codes(hidden_size);
	CarriedScale embed_scale{};
	const SslmForwardStatus est = EmbedEntry(token, vocab_size, embed_weights, hidden_size,
	                                          embed_site_constant, embed_codes.data(), &embed_scale);
	if (est != SslmForwardStatus::Ok) return est;
	std::memcpy(seq.hidden_codes, embed_codes.data(), hidden_size);
	seq.hidden_scale = embed_scale;
	seq.layer_index = 0;
	return RunLayerLoop(seq, layers, num_hidden_layers, num_hidden_layers, hidden_size, head_dim,
	                     num_kv_heads, intermediate_size, context_cap, rope_tables, ws, ws_size);
}

// One GPU decode step through the PRODUCTION public surface only:
// sslm_gpu_seq_embed_token(token) then sslm_decode_step_gpu(full budget) +
// sslm_gpu_ready(block=1). No bench-bridge WRITE accessor anywhere in this function.
static SslmForwardStatus StepGpu(SslmGpuContext* ctx, SslmGpuSequenceHandle* seq_handle,
                                  int32_t token, uint32_t num_hidden_layers, int8_t* out_codes,
                                  SequenceLayerState* out_view, size_t hidden_size) {
	const SslmGpuStatus est = sslm_gpu_seq_embed_token(ctx, seq_handle, token);
	if (est != SSLM_OK) return SslmForwardStatus::TokenIdOutOfRange;  // only rejection this call has

	constexpr uint32_t kFullBudget = 0xFFFFFFFFu;  // PlanDispatchBudgetGpu clamps to num_hidden_layers
	SslmGpuStatus st = sslm_decode_step_gpu(ctx, seq_handle, nullptr, kFullBudget);
	if (st != SSLM_OK) return SslmForwardStatus::GpuDeviceRemoved;
	int32_t ready = 0;
	SslmGpuStatus out_status = SSLM_OK;
	while (st == SSLM_OK && !ready) st = sslm_gpu_ready(ctx, seq_handle, /*block=*/1, &ready, &out_status);
	if (st != SSLM_OK) return SslmForwardStatus::GpuDeviceRemoved;

	out_view->hidden_codes = SslmGpuSeqHandleHiddenCodesForBench(seq_handle);
	out_view->hidden_scale = *SslmGpuSeqHandleHiddenScaleForBench(seq_handle);
	out_view->layer_index = *SslmGpuSeqHandleLayerIndexForBench(seq_handle);
	out_view->kv_saturation_count = *SslmGpuSeqHandleKvSaturationForBench(seq_handle);
	out_view->context_length = *SslmGpuSeqHandleContextLengthForBench(seq_handle);
	std::memcpy(out_codes, out_view->hidden_codes, hidden_size);
	(void)num_hidden_layers;
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
	if (argc < 2) {
		std::fprintf(stderr, "usage: %s <model1p5b.sslm> [extra_decode_tokens]\n", argv[0]);
		return 2;
	}
	const int extra_decode_tokens = argc >= 3 ? std::atoi(argv[2]) : 4;

	std::vector<uint8_t> model_bytes;
	SslmModelView view;
	if (!LoadModel(argv[1], &view, &model_bytes)) return 1;

	const uint32_t num_heads = view.config.num_attention_heads;
	const uint32_t num_kv_heads = view.config.num_key_value_heads;
	const uint32_t num_hidden_layers = view.config.num_hidden_layers;
	const size_t hidden_size = view.config.hidden_size;
	const size_t head_dim = view.config.head_dim;
	const size_t intermediate_size = view.config.intermediate_size;
	const int64_t context_cap = static_cast<int64_t>(view.config.context_cap);
	const int32_t vocab_size = view.config.vocab_size;

	std::vector<LayerBacking> backings(num_hidden_layers);
	std::vector<LayerWeights> layers(num_hidden_layers);
	for (uint32_t l = 0; l < num_hidden_layers; ++l) {
		std::string err;
		if (!MarshalLayer(view, l, num_heads, num_kv_heads, backings[l], layers[l], &err)) {
			std::fprintf(stderr, "FAILED at stage=layer_weights_marshal: layer=%u \"%s\"\n", l, err.c_str());
			return 1;
		}
	}

	const superslm::SslmTensorView* embed_w = view.weights.Tensor("embed");
	if (!embed_w) {
		std::fprintf(stderr, "FAILED: artifact has no embed tensor\n");
		return 1;
	}
	bool ok = true;
	const CarriedScale embed_site_constant = ReadCarriedScale(view.composition_constants, "embed", &ok);
	if (!ok) {
		std::fprintf(stderr, "FAILED: artifact has no embed site constant\n");
		return 1;
	}
	const int8_t* embed_weights = reinterpret_cast<const int8_t*>(embed_w->data);
	const size_t kv_bytes = static_cast<size_t>(num_hidden_layers) * static_cast<size_t>(context_cap) *
	                        num_kv_heads * head_dim * 2;

	GpuContextConfig cfg{};
	GpuResidencyConfig rcfg{};
	SslmGpuContext* ctx = nullptr;
	CHECK(sslm_gpu_context_create(cfg, &ctx) == SSLM_OK && ctx != nullptr, "sslm_gpu_context_create failed");
	if (!ctx) return g_failures ? 1 : 0;

	SslmGpuModelHandle* model = nullptr;
	CHECK(sslm_gpu_model_map(ctx, &view, rcfg, &model) == SSLM_OK && model != nullptr,
	      "sslm_gpu_model_map(1.5B) failed");
	if (!model) {
		sslm_gpu_context_destroy(ctx);
		return g_failures ? 1 : 0;
	}

	// --- Dim 2 mechanism: hostile token_id, mechanism + state-untouched proof. ---
	{
		SslmGpuSequenceHandle* seq_handle = nullptr;
		CHECK(sslm_gpu_seq_create(ctx, model, context_cap, &seq_handle) == SSLM_OK && seq_handle != nullptr,
		      "sslm_gpu_seq_create (dim2) failed");
		if (seq_handle) {
			// Embed a real token first (token 1, distinct from the fresh-handle all-zero
			// default so a "state left untouched" claim is actually observable -- a
			// hostile call against a still-all-zero handle would trivially "not touch"
			// zeros regardless of whether the guard runs before or after a write).
			CHECK(sslm_gpu_seq_embed_token(ctx, seq_handle, 1) == SSLM_OK,
			      "sslm_gpu_seq_embed_token(token=1) did not return Ok");

			std::vector<int8_t> before_codes(SslmGpuSeqHandleHiddenCodesForBench(seq_handle),
			                                  SslmGpuSeqHandleHiddenCodesForBench(seq_handle) +
			                                      SslmGpuSeqHandleHiddenSizeForBench(seq_handle));
			const CarriedScale before_scale = *SslmGpuSeqHandleHiddenScaleForBench(seq_handle);
			const uint32_t before_layer_index = *SslmGpuSeqHandleLayerIndexForBench(seq_handle);

			CHECK(sslm_gpu_seq_embed_token(ctx, seq_handle, -1) == SSLM_TOKEN_ID_OUT_OF_RANGE,
			      "token_id=-1 did not return SSLM_TOKEN_ID_OUT_OF_RANGE");
			CHECK(sslm_gpu_seq_embed_token(ctx, seq_handle, vocab_size) == SSLM_TOKEN_ID_OUT_OF_RANGE,
			      "token_id=vocab_size did not return SSLM_TOKEN_ID_OUT_OF_RANGE");
			CHECK(sslm_gpu_seq_embed_token(ctx, seq_handle, vocab_size + 1000000) ==
			          SSLM_TOKEN_ID_OUT_OF_RANGE,
			      "a wildly out-of-range token_id did not return SSLM_TOKEN_ID_OUT_OF_RANGE");

			const int8_t* after_codes = SslmGpuSeqHandleHiddenCodesForBench(seq_handle);
			const CarriedScale after_scale = *SslmGpuSeqHandleHiddenScaleForBench(seq_handle);
			const uint32_t after_layer_index = *SslmGpuSeqHandleLayerIndexForBench(seq_handle);
			bool state_untouched = before_scale.m == after_scale.m && before_scale.e == after_scale.e &&
			                        before_layer_index == after_layer_index;
			for (size_t i = 0; state_untouched && i < before_codes.size(); ++i) {
				if (before_codes[i] != after_codes[i]) state_untouched = false;
			}
			CHECK(state_untouched,
			      "a hostile token_id mutated seq's own hidden_codes/hidden_scale/layer_index");

			sslm_gpu_seq_release(ctx, seq_handle);
		}
	}

	// --- Busy precondition: embed against a Submitted sequence returns SSLM_BUSY. ---
	{
		SslmGpuSequenceHandle* seq_handle = nullptr;
		CHECK(sslm_gpu_seq_create(ctx, model, context_cap, &seq_handle) == SSLM_OK && seq_handle != nullptr,
		      "sslm_gpu_seq_create (busy) failed");
		if (seq_handle) {
			CHECK(sslm_gpu_seq_embed_token(ctx, seq_handle, 2) == SSLM_OK,
			      "sslm_gpu_seq_embed_token(token=2) did not return Ok");
			CHECK(sslm_decode_step_gpu(ctx, seq_handle, nullptr, 0xFFFFFFFFu) == SSLM_OK,
			      "decode_step_gpu did not return SSLM_OK");
			CHECK(sslm_gpu_seq_embed_token(ctx, seq_handle, 3) == SSLM_BUSY,
			      "sslm_gpu_seq_embed_token against a Submitted sequence did not return SSLM_BUSY");
			int32_t ready = 0;
			SslmGpuStatus out_status = SSLM_OK;
			SslmGpuStatus st = SSLM_OK;
			while (!ready) st = sslm_gpu_ready(ctx, seq_handle, 1, &ready, &out_status);
			CHECK(st == SSLM_OK && out_status == SSLM_OK, "draining the Submitted decode did not succeed");
			sslm_gpu_seq_release(ctx, seq_handle);
		}
	}

	// --- Dim 6 product: real multi-token prefill, then continued decode, bit-equality
	// against the CPU oracle at every step, driven entirely through
	// sslm_gpu_seq_embed_token + sslm_decode_step_gpu. Prompt tokens chosen to span a
	// real range of the vocabulary (not merely token 0 repeated), including the boundary
	// value vocab_size-1 and a token in the low range that legitimately re-occurs later
	// in the sequence (a real prompt can repeat a token; the K/V history must still
	// diverge from position to position). ---
	{
		const std::vector<int32_t> prompt_tokens = {
		    5, 128, 1000, 5,  // token 5 repeats -- proves position (not token identity)
		                      // drives the K/V row written, matching the CPU oracle's own
		                      // per-position RoPE application.
		    static_cast<int32_t>(vocab_size - 1), 42};
		std::vector<int32_t> all_tokens(prompt_tokens);
		for (int i = 0; i < extra_decode_tokens; ++i) {
			// Deterministic pseudo-generation tokens beyond the prompt -- still real,
			// in-range token ids, exercising sslm_gpu_seq_embed_token past prefill into
			// the ordinary per-generated-token re-entry the CPU path's own closure makes
			// no distinction from prefill (design Sec5.3a: "prefill is that closure
			// called once per prompt token, in order; ... every subsequently generated
			// token re-enters through the same closure").
			all_tokens.push_back((17 * (i + 1) + 9) % vocab_size);
		}

		SslmGpuSequenceHandle* seq_handle = nullptr;
		CHECK(sslm_gpu_seq_create(ctx, model, context_cap, &seq_handle) == SSLM_OK && seq_handle != nullptr,
		      "sslm_gpu_seq_create (dim6 prefill) failed");
		if (seq_handle) {
			SequenceLayerState cpu_seq{};
			std::vector<int8_t> cpu_codes(hidden_size, 0);
			cpu_seq.hidden_codes = cpu_codes.data();
			std::vector<uint8_t> cpu_ws(kv_bytes, 0);

			bool all_match = true;
			for (size_t t = 0; t < all_tokens.size() && all_match; ++t) {
				const SslmForwardStatus cpu_st =
				    StepCpu(cpu_seq, all_tokens[t], vocab_size, embed_weights, hidden_size,
				            embed_site_constant, layers.data(), num_hidden_layers, head_dim, num_kv_heads,
				            intermediate_size, context_cap, view.rope_tables, cpu_ws.data(), cpu_ws.size());

				SequenceLayerState gpu_view{};
				std::vector<int8_t> gpu_codes(hidden_size, 0);
				const SslmForwardStatus gpu_st = StepGpu(ctx, seq_handle, all_tokens[t], num_hidden_layers,
				                                          gpu_codes.data(), &gpu_view, hidden_size);

				const bool step_ok = CompareStep("dim6_prefill", static_cast<int>(t), cpu_st, cpu_seq,
				                                  cpu_codes, gpu_st, gpu_view, gpu_codes.data(), hidden_size);
				all_match = all_match && step_ok;
			}
			CHECK(all_match, "prefill-through-decode bit-equality diverged from the CPU oracle");
			std::fprintf(stderr, "  dim6 prefill (%zu prompt + %d decode tokens): %s\n", prompt_tokens.size(),
			             extra_decode_tokens, all_match ? "OK, bit-identical every step" : "FAIL");

			sslm_gpu_seq_release(ctx, seq_handle);
		}
	}

	sslm_gpu_model_unmap(ctx, model);
	sslm_gpu_context_destroy(ctx);

	std::fprintf(stderr, "T-2113 B3.5 embed-token smoke: checks=%d failures=%d\n", g_checks, g_failures);
	return g_failures ? 1 : 0;
}

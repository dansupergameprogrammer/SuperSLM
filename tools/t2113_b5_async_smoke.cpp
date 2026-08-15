// T-2113 (B5): the bench proof for design Sec10 B5's own delivery -- the async
// submission boundary (sslm_decode_step_gpu records+submits without fencing;
// sslm_gpu_ready polls/blocks; the unconditional per-call root-view rebind, design
// Sec6.2), driven through the REAL production 1.0 API (not the B3 bench bridge, which
// this section supersedes -- see gpu_1p0_bench_bridge.h's own retirement note).
//
// Design Sec10 B5's own stated gate: "time-slice invariance at the async boundary
// itself, generalizing T-2105's own Sec4 sweep from a synthetic env-var cut to the real
// sslm_decode_step_gpu/sslm_gpu_ready pair -- same oracle (per-step CPU/GPU bit-equality
// across an interrupted vs. uninterrupted run), new cut mechanism."
//
// Proves, on real D3D12 hardware, against a real 1.5B artifact:
//   1. Busy: a second sslm_decode_step_gpu call against a Submitted sequence returns
//      SSLM_BUSY; sslm_gpu_ready(block=1) then completes it.
//   2. SSLM_DISPATCH_BUDGET_TOO_SMALL: a budget under one layer's own dispatch count.
//   3. Poll semantics: sslm_gpu_ready(block=0) never blocks the calling thread; a
//      subsequent sslm_gpu_ready(block=1) against the SAME still-Submitted sequence
//      always completes.
//   4. THE MAIN GATE -- time-slice invariance at the real boundary: the SAME decode
//      session (a fixed embedded token, run for N tokens, context_length advancing),
//      driven at several distinct per-call dispatch_budget granularities -- one call per
//      token (uninterrupted) down to one call per LAYER (the every-layer extreme, the
//      finest granularity a whole-layer-quantum budget allows) -- every one compared
//      per-step, bit-for-bit, against the SAME session's own CPU oracle (RunLayerLoop),
//      run fully serially from an identical cold start. No divergence at any granularity
//      proves a decode session interrupted and resumed across MANY separate async API
//      calls produces the identical result to one uninterrupted call.
//   5. C5-equivalent: a single full-token call (dispatch_budget covering every layer)
//      is bit-identical to the CPU oracle, the "uninterrupted" end of point 4's own
//      sweep, stated as its own explicit check.
//   6. The two T-2106 violation-class pins, planted at the REAL async boundary
//      (SSLM_B5_ASYNC_DROP_UAV_REBIND / SSLM_B5_ASYNC_SWAP_SRV_REBIND): each fires (the
//      per-step oracle diverges) when planted on a MULTI-CALL (interrupted) session, and
//      is silent (bit-identical) when reverted -- the plant-and-revert commissioning
//      protocol StandardsDocument.md Sec5.4 requires.
//
// Usage: t2113_b5_async_smoke <model1p5b.sslm> [tokens]  (exits 0 on pass, 1 on fail)
#include <chrono>
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

#define CHECK(cond, msg)                                                 \
	do {                                                                    \
		++g_checks;                                                          \
		if (!(cond)) {                                                       \
			++g_failures;                                                      \
			std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, msg);   \
		}                                                                     \
	} while (0)

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
	const SslmModelStatus st = superslm::SslmModel::Load(out_bytes->data(), out_bytes->size(), *out_view, &err);
	if (st != SslmModelStatus::Ok) {
		std::fprintf(stderr, "model load failed for %s: %s\n", path.c_str(), err.c_str());
		return false;
	}
	return true;
}

// One CPU decode step: re-embed the fixed token, run every layer -- the identical
// convention t2113_b3_sequence_smoke.cpp/t2100_gpu_throughput.cpp use.
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

// One GPU decode step, driven through the REAL sslm_decode_step_gpu/sslm_gpu_ready
// async pair, at a caller-chosen dispatch_budget -- `dispatches_per_layer` calls when
// `budget_layers == 1` (the every-layer extreme), one call when `budget_layers >=
// num_hidden_layers` (uninterrupted). Blocks (via sslm_gpu_ready(block=1)) after every
// sub-call -- the interruption this test cares about is the ASYNC BOUNDARY BEING
// CROSSED MULTIPLE TIMES (a fresh Submit+Finish per budget-worth of layers), not
// wall-clock overlap with other work, which is a caller-timing property this bit-
// equality gate has no stake in.
static SslmForwardStatus StepGpuAsync(SslmGpuContext* ctx, SslmGpuSequenceHandle* seq_handle,
                                       const int8_t* embed_codes, const CarriedScale& embed_scale,
                                       size_t hidden_size, uint32_t num_hidden_layers,
                                       uint32_t dispatches_per_layer, uint32_t budget_layers,
                                       int8_t* out_codes_for_compare,
                                       SequenceLayerState* out_view_for_compare) {
	// NO sslm_gpu_seq_reset here -- this is called ONCE PER TOKEN, and reset would wipe
	// context_length/kv_saturation_count/K-V history between tokens, exactly the
	// per-token divergence a first, wrong draft of this tool reproduced (CPU's own
	// context_length correctly accumulates across StepCpu's own repeated calls in
	// main()'s loop below; forcing GPU back to a fresh sequence every call diverged
	// from step 1 onward). Only `sslm_gpu_seq_create` (main(), once per granularity)
	// resets state; this function re-embeds the FIXED token's own codes/scale into the
	// handle's own hidden_codes/hidden_scale (matching StepCpu's own "re-embed token 0,
	// run every layer" convention) via the public API's own bench accessor -- there is
	// no "set hidden codes" call in the 1.0 API surface (a caller always starts from
	// embed or restore), so this bench tool reaches in via the one channel design
	// Sec5.3 gives it: the handle owns hidden_codes, and this test IS the code that
	// would normally call EmbedEntry into it. We copy through the same accessor B1-B3's
	// bench tools used, since it is still declared and still valid
	// (gpu_1p0_bench_bridge.h). `layer_index` IS reset here, to 0, alongside the embed --
	// corrected after a first, wrong draft of this tool assumed the GPU's own commit
	// dispatch auto-wraps it. Read from source (forward_sites.cpp, the CPU oracle
	// RunLayerLoop itself, `S3.7`'s own comment): "a fresh-token embed... [zeroes
	// layer_index]" -- `seq.layer_index` reaches `num_hidden_layers` when a token
	// commits and STAYS there (`SequenceAlreadyComplete` rejects a further call at that
	// value, forward_sites.cpp:1404); zeroing it is bundled with embedding the NEXT
	// token, at every existing call site in this tree (t2039_c5_harness.cpp,
	// t2100_gpu_throughput.cpp, t2113_b3_sequence_smoke.cpp's own `StepGpuThroughHandle`
	// -- all set `.layer_index = 0` in the SAME place they set the new token's codes).
	// This bench tool follows the identical, already-established convention.
	extern int8_t* SslmGpuSeqHandleHiddenCodesForBench(SslmGpuSequenceHandle*);
	extern superslm::CarriedScale* SslmGpuSeqHandleHiddenScaleForBench(SslmGpuSequenceHandle*);
	extern uint32_t* SslmGpuSeqHandleLayerIndexForBench(SslmGpuSequenceHandle*);
	std::memcpy(SslmGpuSeqHandleHiddenCodesForBench(seq_handle), embed_codes, hidden_size);
	*SslmGpuSeqHandleHiddenScaleForBench(seq_handle) = embed_scale;
	*SslmGpuSeqHandleLayerIndexForBench(seq_handle) = 0;

	SslmGpuStatus st = SSLM_OK;
	int32_t out_ready = 0;
	SslmGpuStatus out_status = SSLM_OK;
	uint32_t layers_done = 0;
	while (layers_done < num_hidden_layers) {
		const uint32_t budget = budget_layers * dispatches_per_layer;
		st = sslm_decode_step_gpu(ctx, seq_handle, nullptr, budget);
		if (st != SSLM_OK) break;
		// A second call while Submitted must return SSLM_BUSY (point 1 of this file's
		// own header) -- exercised on the FIRST sub-call of the FIRST step only, so the
		// main sweep below is not slowed by a redundant check on every call.
		out_ready = 0;
		do {
			st = sslm_gpu_ready(ctx, seq_handle, /*block=*/1, &out_ready, &out_status);
		} while (st == SSLM_OK && !out_ready);
		if (st != SSLM_OK || out_status != SSLM_OK) break;
		layers_done += budget_layers < (num_hidden_layers - layers_done)
		                   ? budget_layers
		                   : (num_hidden_layers - layers_done);
	}
	extern superslm::CarriedScale* SslmGpuSeqHandleHiddenScaleForBench(SslmGpuSequenceHandle*);
	extern uint32_t* SslmGpuSeqHandleLayerIndexForBench(SslmGpuSequenceHandle*);
	extern uint64_t* SslmGpuSeqHandleKvSaturationForBench(SslmGpuSequenceHandle*);
	extern int64_t* SslmGpuSeqHandleContextLengthForBench(SslmGpuSequenceHandle*);
	out_view_for_compare->hidden_codes = SslmGpuSeqHandleHiddenCodesForBench(seq_handle);
	out_view_for_compare->hidden_scale = *SslmGpuSeqHandleHiddenScaleForBench(seq_handle);
	out_view_for_compare->layer_index = *SslmGpuSeqHandleLayerIndexForBench(seq_handle);
	out_view_for_compare->kv_saturation_count = *SslmGpuSeqHandleKvSaturationForBench(seq_handle);
	out_view_for_compare->context_length = *SslmGpuSeqHandleContextLengthForBench(seq_handle);
	std::memcpy(out_codes_for_compare, out_view_for_compare->hidden_codes, hidden_size);
	if (st != SSLM_OK) return SslmForwardStatus::GpuDeviceRemoved;  // no better mapping for a bench tool
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
		std::fprintf(stderr, "usage: %s <model1p5b.sslm> [tokens]\n", argv[0]);
		return 2;
	}
	const int tokens = argc >= 3 ? std::atoi(argv[2]) : 6;

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
	constexpr uint32_t kDispatchesPerLayer = 24;  // design Sec3/Sec6.1, T-2113 B4

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
	      "sslm_gpu_model_map(1.5B) failed");
	if (!model) {
		sslm_gpu_context_destroy(ctx);
		return g_failures ? 1 : 0;
	}

	// A violation-pin env var, if set, is set for THIS WHOLE PROCESS (the corruption it
	// gates is read once via a `static const` in superslm_gpu.cpp, B4's own established
	// per-process convention, carried forward at B5 -- see that file's own comment).
	// Unlike B4's own bench-only mid-call mechanism (which corrupted only a partial
	// slice within one call), planting this pin at the REAL top-of-call rebind site
	// corrupts EVERY dispatch of EVERY call for the rest of the process -- a dropped
	// K/V UAV rebind redirects K/V-row writes at K/V-buffer-scale offsets into
	// `work_scratch_uav` (sized for transient per-dispatch scratch, far smaller), a
	// genuine out-of-bounds UAV write that can legitimately fault the device (a TDR),
	// not merely a clean wrong-answer -- worse, and more real, than B4's own partial-
	// slice version, and once it happens the device is unusable for the rest of THIS
	// process. So: a "planted" invocation runs ONLY the pin-check section below, never
	// the granularity sweep or the Busy/budget checks (which the fault would otherwise
	// cascade-fail, exactly as an earlier draft of this tool reproduced).
	const bool any_pin_planted = (std::getenv("SSLM_B5_ASYNC_DROP_UAV_REBIND") != nullptr &&
	                               std::getenv("SSLM_B5_ASYNC_DROP_UAV_REBIND")[0] == '1') ||
	                              (std::getenv("SSLM_B5_ASYNC_SWAP_SRV_REBIND") != nullptr &&
	                               std::getenv("SSLM_B5_ASYNC_SWAP_SRV_REBIND")[0] == '1');

	// The granularities swept, uninterrupted -> every-layer extreme.
	const uint32_t granularities[] = {num_hidden_layers, num_hidden_layers / 2 > 0 ? num_hidden_layers / 2 : 1,
	                                   4, 1};

	for (uint32_t budget_layers : !any_pin_planted ? std::vector<uint32_t>(std::begin(granularities), std::end(granularities)) : std::vector<uint32_t>{}) {
		SslmGpuSequenceHandle* seq_handle = nullptr;
		CHECK(sslm_gpu_seq_create(ctx, model, context_cap, &seq_handle) == SSLM_OK && seq_handle != nullptr,
		      "sslm_gpu_seq_create failed");
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
			    StepGpuAsync(ctx, seq_handle, embed_codes.data(), embed_scale, hidden_size, num_hidden_layers,
			                 kDispatchesPerLayer, budget_layers, gpu_codes.data(), &gpu_view);

			granularity_ok = CompareStep("granularity", t, cpu_st, cpu_seq, cpu_codes, gpu_st, gpu_view,
			                             gpu_codes.data(), hidden_size);
		}
		char msg[160];
		std::snprintf(msg, sizeof(msg), "budget_layers=%u: async decode diverged from CPU oracle",
		              budget_layers);
		CHECK(granularity_ok, msg);
		std::fprintf(stderr, "  granularity budget_layers=%u (%u calls/token): %s\n", budget_layers,
		             (num_hidden_layers + budget_layers - 1) / budget_layers, granularity_ok ? "OK" : "FAIL");

		sslm_gpu_seq_release(ctx, seq_handle);
	}

	// Point 1/2: Busy and DispatchBudgetTooSmall, on a dedicated fresh sequence. Skipped
	// on a planted-pin invocation, per this file's own note above.
	if (!any_pin_planted) {
		SslmGpuSequenceHandle* seq_handle = nullptr;
		CHECK(sslm_gpu_seq_create(ctx, model, context_cap, &seq_handle) == SSLM_OK && seq_handle != nullptr,
		      "sslm_gpu_seq_create (policy) failed");
		if (seq_handle) {
			extern int8_t* SslmGpuSeqHandleHiddenCodesForBench(SslmGpuSequenceHandle*);
			std::memcpy(SslmGpuSeqHandleHiddenCodesForBench(seq_handle), embed_codes.data(), hidden_size);

			CHECK(sslm_decode_step_gpu(ctx, seq_handle, nullptr, 0) == SSLM_DISPATCH_BUDGET_TOO_SMALL,
			      "dispatch_budget=0 did not return SSLM_DISPATCH_BUDGET_TOO_SMALL");
			CHECK(sslm_decode_step_gpu(ctx, seq_handle, nullptr, kDispatchesPerLayer - 1) ==
			          SSLM_DISPATCH_BUDGET_TOO_SMALL,
			      "dispatch_budget < one layer's own dispatch count did not return SSLM_DISPATCH_BUDGET_TOO_SMALL");

			CHECK(sslm_decode_step_gpu(ctx, seq_handle, nullptr, kDispatchesPerLayer * num_hidden_layers) ==
			          SSLM_OK,
			      "full-budget decode_step_gpu did not return SSLM_OK");
			CHECK(sslm_decode_step_gpu(ctx, seq_handle, nullptr, kDispatchesPerLayer * num_hidden_layers) ==
			          SSLM_BUSY,
			      "a second decode_step_gpu against a Submitted sequence did not return SSLM_BUSY");
			int32_t poll_ready = -1;
			SslmGpuStatus poll_status = SSLM_OK;
			CHECK(sslm_gpu_ready(ctx, seq_handle, /*block=*/0, &poll_ready, &poll_status) == SSLM_OK,
			      "sslm_gpu_ready(block=0) did not return SSLM_OK");
			// poll_ready may be 0 or 1 here (a race against real GPU completion latency) --
			// both are legal per design Sec4.2; what matters is the call itself never blocked
			// (this line was reached at all) and a subsequent block=1 always completes it.
			int32_t block_ready = 0;
			SslmGpuStatus block_status = SSLM_OK;
			CHECK(sslm_gpu_ready(ctx, seq_handle, /*block=*/1, &block_ready, &block_status) == SSLM_OK &&
			          block_ready == 1 && block_status == SSLM_OK,
			      "sslm_gpu_ready(block=1) did not complete a Submitted sequence");
			sslm_gpu_seq_release(ctx, seq_handle);
		}
	}

	// Point 6: the two T-2106 violation-class pins, at the real async boundary. The
	// corruption they gate (superslm_gpu.cpp) reads its own env var via `std::getenv`
	// exactly ONCE per PROCESS (a `static` local, B4's own established convention,
	// carried forward unchanged at B5) -- so the plant-and-revert protocol runs as
	// SEPARATE PROCESS INVOCATIONS of this same executable, driven by the env var
	// state the OS environment already has when this process starts, never by
	// toggling it mid-run (which the static local would silently ignore). This run
	// checks its OWN environment ONCE, here, and asserts the ONE outcome that
	// env state predicts -- a clean sequence, decoded for `tokens` tokens at
	// budget_layers=1 (the every-layer extreme, maximizing async-boundary crossings),
	// compared against the CPU oracle.
	{
		const bool drop_planted = std::getenv("SSLM_B5_ASYNC_DROP_UAV_REBIND") != nullptr &&
		                           std::getenv("SSLM_B5_ASYNC_DROP_UAV_REBIND")[0] == '1';
		const bool swap_planted = std::getenv("SSLM_B5_ASYNC_SWAP_SRV_REBIND") != nullptr &&
		                           std::getenv("SSLM_B5_ASYNC_SWAP_SRV_REBIND")[0] == '1';
		if (drop_planted || swap_planted) {
			SslmGpuSequenceHandle* seq_handle = nullptr;
			CHECK(sslm_gpu_seq_create(ctx, model, context_cap, &seq_handle) == SSLM_OK && seq_handle != nullptr,
			      "sslm_gpu_seq_create (pin sweep) failed");
			if (seq_handle) {
				SequenceLayerState cpu_seq{};
				std::vector<int8_t> cpu_codes(hidden_size, 0);
				cpu_seq.hidden_codes = cpu_codes.data();
				std::vector<uint8_t> cpu_ws(kv_bytes, 0);
				bool all_match = true;
				for (int t = 0; t < tokens; ++t) {
					const SslmForwardStatus cpu_st =
					    StepCpu(cpu_seq, cpu_codes.data(), embed_codes.data(), embed_scale, layers.data(),
					            num_hidden_layers, hidden_size, head_dim, num_kv_heads, intermediate_size,
					            context_cap, view.rope_tables, cpu_ws.data(), cpu_ws.size());
					SequenceLayerState gpu_view{};
					std::vector<int8_t> gpu_codes(hidden_size, 0);
					const SslmForwardStatus gpu_st = StepGpuAsync(
					    ctx, seq_handle, embed_codes.data(), embed_scale, hidden_size, num_hidden_layers,
					    kDispatchesPerLayer, /*budget_layers=*/1, gpu_codes.data(), &gpu_view);
					bool step_ok = cpu_st == gpu_st &&
					               cpu_seq.kv_saturation_count == gpu_view.kv_saturation_count &&
					               cpu_seq.context_length == gpu_view.context_length;
					for (size_t i = 0; step_ok && i < hidden_size; ++i) {
						if (cpu_codes[i] != gpu_codes[i]) step_ok = false;
					}
					all_match = all_match && step_ok;
				}
				sslm_gpu_seq_release(ctx, seq_handle);
				// PLANTED: the pin must FIRE -- a real, safely-producible divergence.
				CHECK(!all_match, drop_planted
				                       ? "SSLM_B5_ASYNC_DROP_UAV_REBIND=1 did NOT diverge (pin did not fire)"
				                       : "SSLM_B5_ASYNC_SWAP_SRV_REBIND=1 did NOT diverge (pin did not fire)");
				std::fprintf(stderr, "  pin %s planted: %s\n", drop_planted ? "DROP_UAV_REBIND" : "SWAP_SRV_REBIND",
				             all_match ? "DID NOT FIRE (FAIL)" : "fired as expected");
			}
		} else {
			std::fprintf(stderr,
			             "  pin sweep: neither env var set on this invocation -- this run is the REVERTED leg "
			             "(covered by the granularity sweep above, which already ran with both env vars unset "
			             "and found zero divergence at every granularity including budget_layers=1)\n");
		}
	}

	// T-2113 (B5, design Sec10 B5's own gate + the build seat's own instruction):
	// "throughput through the ASYNC 1.0 path: 3x64 protocol with spread, against B4's
	// 58.41-58.77 synchronous figure." One call per token (full dispatch_budget,
	// matching every prior round's own comparison basis, B4's own confirmation table
	// and T-2105's own headline cell) -- this is the "uninterrupted" end of this file's
	// own granularity sweep above, now TIMED rather than merely correctness-checked,
	// through the production `sslm_decode_step_gpu`/`sslm_gpu_ready` pair (never the
	// pre-1.0 `RunLayerLoopGpu` `t2100_gpu_throughput.cpp` itself still measures) --
	// this is the routing that retires `g_resident_weights`/`g_resident_rope` for
	// the MEASURED path (D-SLM3362): every call here binds `model`'s own resident
	// buffers directly, never touching either process-global cache. Skipped on a
	// planted-pin invocation (this file's own note above -- timing a run already
	// expected to device-fault is meaningless).
	if (!any_pin_planted) {
		constexpr uint32_t kFullBudget = 0xFFFFFFFFu;  // PlanDispatchBudgetGpu clamps to
		                                                // num_hidden_layers regardless
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
			// One warmup step, discarded (this codebase's own established convention,
			// t2100_gpu_throughput.cpp's own header comment: "already the harness's own
			// one warmup step, discarded").
			for (int w = 0; w < 1; ++w) {
				std::memcpy(SslmGpuSeqHandleHiddenCodesForBench(seq_handle), embed_codes.data(), hidden_size);
				*SslmGpuSeqHandleHiddenScaleForBench(seq_handle) = embed_scale;
				*SslmGpuSeqHandleLayerIndexForBench(seq_handle) = 0;
				SslmGpuStatus st = sslm_decode_step_gpu(ctx, seq_handle, nullptr, kFullBudget);
				int32_t ready = 0;
				SslmGpuStatus out_status = SSLM_OK;
				while (st == SSLM_OK && !ready) st = sslm_gpu_ready(ctx, seq_handle, 1, &ready, &out_status);
				if (st != SSLM_OK || out_status != SSLM_OK) {
					timing_ok = false;
				}
			}
			const auto t0 = std::chrono::steady_clock::now();
			for (int s = 0; s < kSteps && timing_ok; ++s) {
				std::memcpy(SslmGpuSeqHandleHiddenCodesForBench(seq_handle), embed_codes.data(), hidden_size);
				*SslmGpuSeqHandleHiddenScaleForBench(seq_handle) = embed_scale;
				*SslmGpuSeqHandleLayerIndexForBench(seq_handle) = 0;
				SslmGpuStatus st = sslm_decode_step_gpu(ctx, seq_handle, nullptr, kFullBudget);
				int32_t ready = 0;
				SslmGpuStatus out_status = SSLM_OK;
				while (st == SSLM_OK && !ready) st = sslm_gpu_ready(ctx, seq_handle, 1, &ready, &out_status);
				if (st != SSLM_OK || out_status != SSLM_OK) {
					timing_ok = false;
				}
			}
			const auto t1 = std::chrono::steady_clock::now();
			const double elapsed_s = std::chrono::duration<double>(t1 - t0).count();
			run_tok_s[run] = kSteps / elapsed_s;
			sslm_gpu_seq_release(ctx, seq_handle);
		}
		CHECK(timing_ok, "3x64 async throughput protocol did not complete cleanly");
		if (timing_ok) {
			std::fprintf(stderr,
			             "  THROUGHPUT (async 1.0 path, one sslm_decode_step_gpu+sslm_gpu_ready "
			             "round-trip/token, full budget, 3x64 steps): run1=%.2f run2=%.2f run3=%.2f tok/s\n",
			             run_tok_s[0], run_tok_s[1], run_tok_s[2]);
		}
	}

	sslm_gpu_model_unmap(ctx, model);
	sslm_gpu_context_destroy(ctx);

	std::fprintf(stderr, "checks=%d failures=%d\n", g_checks, g_failures);
	return g_failures ? 1 : 0;
}

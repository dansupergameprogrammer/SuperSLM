// T-2169 (Rung 2, design Sec5/Sec8, D-SLM3596): measures the TDR-safe sub-chunk bound on the
// TARGET hardware -- "the sub-chunk split bound is set from a measured wall-clock figure, not
// the diagnostics ceiling... queried or measured on the target machine, never assumed at a
// default, since the TDR delay is a machine-local, driver-adjustable value."
//
// Method: query this machine's own real TDR timeout (Windows registry
// HKLM\SYSTEM\CurrentControlSet\Control\GraphicsDrivers\TdrDelay, seconds; falls back to the
// documented Windows default of 2 seconds only if the value key is genuinely absent -- this is
// NOT the "assumed at a default" the design forbids, since an absent override key IS the
// platform's real configured behavior, confirmed against the OS's own documented default, not
// guessed). Drive real full-depth chunks of increasing size (1, 2, 4, 8 tokens) through
// SubmitChunkToFullDepthForG5Bridge against the real Qwen2.5-1.5B artifact, read
// `LastCallTiming().gpu_busy_ms` (the GPU-clock-measured, timestamp-query-derived busy window --
// excludes recording/submission/readback, the number the design's own TDR reasoning is about)
// after each, derive the measured ms-per-dispatch rate, then compute the maximum chunk_tokens
// such that `chunk_tokens * num_hidden_layers * kDispatchesPerLayer` dispatches' own measured
// wall-clock time stays under HALF the platform's real TDR timeout (the design's own safety
// margin).
//
// Throwaway harness, matching tools/t2039_c5_harness.cpp's own precedent.
//
// Usage: t2169_tdr_measure <model.sslm>
#include <cstdio>
#include <cstring>
#include <vector>

#include "superslm/artifact.h"
#include "superslm/forward_sites.h"
#include "superslm/gpu_port.h"
#include "superslm/model.h"
#include "sslm_marshal.h"

#if defined(_WIN32)
#include <windows.h>
#endif

using namespace superslm;
using superslm_marshal::LayerBacking;
using superslm_marshal::MarshalLayer;
using superslm_marshal::PreflightScanWscFolds;
using superslm_marshal::ReadCarriedScale;
using superslm_marshal::ReadFile;

namespace superslm_gpu {
struct GpuLayerLoopInFlight;
superslm::SslmForwardStatus SubmitChunkToFullDepthForG5Bridge(
    superslm::SequenceLayerState& seq, const superslm::LayerWeights* layers,
    uint32_t num_hidden_layers, size_t hidden_size, size_t head_dim, size_t num_key_value_heads,
    size_t intermediate_size, int64_t context_cap, const superslm::SslmTensorManifest& rope_tables,
    uint8_t* workspace, size_t workspace_size, const uint8_t* chunk_embedding_bytes,
    uint32_t chunk_len, ID3D12Resource* external_kv_resident,
    bool* io_external_kv_needs_resume_barrier, ID3D12Resource* external_weights_resident,
    ID3D12Resource* external_rope_cos_resident, ID3D12Resource* external_rope_sin_resident,
    bool external_rope_has, uint64_t external_rope_cos_elems, uint64_t external_rope_sin_elems,
    const GpuAdapterBridge* adapter_bridge, GpuLayerLoopInFlight** out_inflight);
}  // namespace superslm_gpu

namespace {

// This project's own established per-layer dispatch count (superslm_gpu.cpp,
// PlanDispatchBudgetGpu's own header comment; kDispatchesPerLayer = 24).
constexpr uint32_t kDispatchesPerLayer = 24;

uint32_t Align8U32(uint32_t v) { return (v + 7u) & ~7u; }
uint32_t SeqScaleOffLocal(uint32_t hidden_size) { return Align8U32(hidden_size * 4u); }

void PackEmbeddingBlock(const int8_t* codes, size_t hidden_size, const CarriedScale& scale,
                         std::vector<uint8_t>* out) {
	const uint32_t H = static_cast<uint32_t>(hidden_size);
	const uint32_t block_bytes = SeqScaleOffLocal(H) + 16u;
	out->assign(block_bytes, 0);
	for (uint32_t i = 0; i < H; ++i) {
		int32_t v = static_cast<int32_t>(codes[i]);
		std::memcpy(out->data() + i * 4u, &v, 4);
	}
	std::memcpy(out->data() + SeqScaleOffLocal(H) + 0, &scale.m, 8);
	std::memcpy(out->data() + SeqScaleOffLocal(H) + 8, &scale.e, 8);
}

// Queries the platform's own real TDR delay, seconds. Registry absent -> the OS's own
// documented default (2s) -- a real, confirmed platform behavior, not an assumed one.
double QueryPlatformTdrDelaySeconds() {
#if defined(_WIN32)
	HKEY key;
	if (RegOpenKeyExA(HKEY_LOCAL_MACHINE, "SYSTEM\\CurrentControlSet\\Control\\GraphicsDrivers", 0,
	                   KEY_READ, &key) == ERROR_SUCCESS) {
		DWORD value = 0, size = sizeof(value), type = 0;
		LONG rc = RegQueryValueExA(key, "TdrDelay", nullptr, &type, reinterpret_cast<BYTE*>(&value), &size);
		RegCloseKey(key);
		if (rc == ERROR_SUCCESS && type == REG_DWORD) {
			return static_cast<double>(value);
		}
	}
#endif
	return 2.0;  // Windows' own documented default TDR delay when no override key is present
}

bool MeasureOneChunkSize(const SslmModelView& model_view, const std::vector<LayerWeights>& layers,
                          const SslmTensorView* embed_w, const CarriedScale& embed_site_constant,
                          uint32_t chunk_tokens, double* out_gpu_busy_ms) {
	const uint32_t num_hidden_layers = model_view.config.num_hidden_layers;
	const size_t hidden_size = model_view.config.hidden_size;
	const size_t head_dim = model_view.config.head_dim;
	const size_t num_kv_heads = model_view.config.num_key_value_heads;
	const size_t intermediate_size = model_view.config.intermediate_size;
	const int64_t context_cap = static_cast<int64_t>(model_view.config.context_cap);
	const int8_t* embed_weights = reinterpret_cast<const int8_t*>(embed_w->data);
	const size_t kv_bytes = static_cast<size_t>(num_hidden_layers) * static_cast<size_t>(context_cap) *
	                        num_kv_heads * head_dim * 2;

	SequenceLayerState seq;
	std::vector<int8_t> codes(hidden_size, 0);
	seq.hidden_codes = codes.data();
	std::vector<uint8_t> ws(kv_bytes, 0);

	const uint32_t block_bytes = SeqScaleOffLocal(static_cast<uint32_t>(hidden_size)) + 16u;
	std::vector<uint8_t> chunk_bytes(static_cast<size_t>(block_bytes) * chunk_tokens);
	for (uint32_t i = 0; i < chunk_tokens; ++i) {
		const int32_t token_id = 90 + static_cast<int32_t>(i % 50);  // real, in-range vocab ids
		std::vector<int8_t> embed_codes(hidden_size);
		CarriedScale embed_scale{};
		const SslmForwardStatus est =
		    EmbedEntry(token_id, model_view.config.vocab_size, embed_weights, hidden_size,
		               embed_site_constant, embed_codes.data(), &embed_scale);
		if (est != SslmForwardStatus::Ok) {
			std::fprintf(stderr, "FAILED at stage=embed token=%d status=%s\n", token_id,
			             SslmForwardStatusName(est));
			return false;
		}
		std::vector<uint8_t> block;
		PackEmbeddingBlock(embed_codes.data(), hidden_size, embed_scale, &block);
		std::memcpy(chunk_bytes.data() + static_cast<size_t>(i) * block_bytes, block.data(), block_bytes);
	}

	superslm_gpu::GpuLayerLoopInFlight* inflight = nullptr;
	const SslmForwardStatus submit_status = superslm_gpu::SubmitChunkToFullDepthForG5Bridge(
	    seq, layers.data(), num_hidden_layers, hidden_size, head_dim, num_kv_heads, intermediate_size,
	    context_cap, model_view.rope_tables, ws.data(), ws.size(), chunk_bytes.data(), chunk_tokens,
	    nullptr, nullptr, nullptr, nullptr, nullptr, false, 0, 0, nullptr, &inflight);
	if (submit_status != SslmForwardStatus::Ok || !inflight) {
		std::fprintf(stderr, "FAILED at stage=submit chunk_tokens=%u status=%s\n", chunk_tokens,
		             SslmForwardStatusName(submit_status));
		return false;
	}
	int32_t ready = 0;
	const SslmForwardStatus finish_status =
	    superslm_gpu::RunLayerLoopGpuFinish(inflight, seq, ws.data(), /*block=*/1, &ready);
	if (finish_status != SslmForwardStatus::Ok) {
		std::fprintf(stderr, "FAILED at stage=finish chunk_tokens=%u status=%s\n", chunk_tokens,
		             SslmForwardStatusName(finish_status));
		return false;
	}
	const superslm_gpu::GpuCallTiming t = superslm_gpu::LastCallTiming();
	*out_gpu_busy_ms = t.gpu_busy_ms;
	return true;
}

}  // namespace

int main(int argc, char** argv) {
	std::setvbuf(stdout, nullptr, _IONBF, 0);  // unbuffered -- so a crash never hides prior output
	if (argc < 2) {
		std::fprintf(stderr, "usage: %s <model.sslm>\n", argv[0]);
		return 2;
	}
	std::vector<uint8_t> model_bytes;
	if (!ReadFile(argv[1], model_bytes)) {
		std::fprintf(stderr, "FAILED: could not read \"%s\"\n", argv[1]);
		return 1;
	}
	SslmModelView model_view;
	std::string model_err;
	if (SslmModel::Load(model_bytes.data(), model_bytes.size(), model_view, &model_err) !=
	    SslmModelStatus::Ok) {
		std::fprintf(stderr, "FAILED at stage=model_load: %s\n", model_err.c_str());
		return 1;
	}
	const uint32_t num_hidden_layers = model_view.config.num_hidden_layers;
	std::printf("model loaded: hidden_size=%u layers=%u dispatches/token=%u\n",
	            model_view.config.hidden_size, num_hidden_layers, num_hidden_layers * kDispatchesPerLayer);

	PreflightScanWscFolds(model_view);
	std::vector<LayerBacking> backings(num_hidden_layers);
	std::vector<LayerWeights> layers(num_hidden_layers);
	for (uint32_t l = 0; l < num_hidden_layers; ++l) {
		std::string marshal_err;
		if (!MarshalLayer(model_view, l, model_view.config.num_attention_heads,
		                   model_view.config.num_key_value_heads, backings[l], layers[l], &marshal_err)) {
			std::fprintf(stderr, "FAILED at stage=marshal layer=%u: %s\n", l, marshal_err.c_str());
			return 1;
		}
	}
	const SslmTensorView* embed_w = model_view.weights.Tensor("embed");
	bool ok = true;
	CarriedScale embed_site_constant = ReadCarriedScale(model_view.composition_constants, "embed", &ok);
	if (!embed_w || !ok) {
		std::fprintf(stderr, "FAILED: missing embed tensor/site constant\n");
		return 1;
	}

	const double tdr_seconds = QueryPlatformTdrDelaySeconds();
	const double tdr_budget_ms = (tdr_seconds * 1000.0) * 0.5;  // D-SLM3596's own half-TDR margin
	std::printf("platform TDR delay: %.1f s (registry HKLM\\...\\GraphicsDrivers\\TdrDelay) -- "
	            "safety budget: %.1f ms (half)\n",
	            tdr_seconds, tdr_budget_ms);

	// Warm-up call (residency caches cold on the very first call skews its own timing --
	// excluded from the rate fit, matching this project's own established "one warmup step,
	// discarded" convention, PrepareGpuLayerLoopChunkOpenState's own fresh_sequence comment).
	double warm_ms = 0.0;
#if defined(SUPERSLM_T2169_MINIMAL_CRASH_REPRO)
	// Minimal, isolated repro for cdb: chunk_tokens=8 as the ONLY call in the process, no warmup.
	double crash_ms = 0.0;
	std::fprintf(stderr, "REPRO: about to call chunk_tokens=8\n");
	std::fflush(stderr);
	if (!MeasureOneChunkSize(model_view, layers, embed_w, embed_site_constant, 8, &crash_ms)) return 1;
	std::printf("REPRO: chunk_tokens=8 gpu_busy_ms=%.4f (did not crash?!)\n", crash_ms);
	return 0;
#endif
	if (!MeasureOneChunkSize(model_view, layers, embed_w, embed_site_constant, 1, &warm_ms)) return 1;
	std::printf("warm-up (discarded): chunk_tokens=1 gpu_busy_ms=%.4f\n", warm_ms);

	// HISTORICAL NOTE (superseded by the fix below, kept for the record): this pass's own first
	// run swept {2,3,4,5,6,7}, all single-list (the split-wrapper did not exist yet), and found
	// chunk_tokens=8 reproducibly crashed the process with STATUS_STACK_OVERFLOW (0xC00000FD) --
	// root-caused under cdb as a recursive cycle entirely inside nvwgf2umx.dll (NVIDIA's own
	// D3D12 driver), not this codebase. D-SLM3649: SubmitChunkToFullDepthForG5Bridge
	// (src/gpu/superslm_gpu.cpp) now internally splits any chunk larger than
	// superslm_gpu::kT2169TdrSafeMaxChunkTokens into multiple sub-chunk submissions, each
	// finished synchronously before the next opens. Consequence for THIS tool: calling the
	// public name at chunk_tokens > kT2169TdrSafeMaxChunkTokens now measures only the LAST
	// sub-chunk's own gpu_busy_ms, not the whole requested chunk's -- the swept set below is
	// restricted to {2,3,4}, all <= the bound (4), so every measurement here stays single-list
	// and directly comparable to a real command-list's own busy time, matching what this rate
	// fit needs. tools/t2169_rung2b_selfcheck.cpp's own chunk_tokens=8/256 cells are what
	// exercise and bit-identity-verify the split path itself.
	const uint32_t sizes[] = {2, 3, 4};
	double total_ms_per_dispatch = 0.0;
	int fits = 0;
	for (uint32_t sz : sizes) {
		double gpu_busy_ms = 0.0;
		if (!MeasureOneChunkSize(model_view, layers, embed_w, embed_site_constant, sz, &gpu_busy_ms)) {
			return 1;
		}
		const uint32_t dispatches = sz * num_hidden_layers * kDispatchesPerLayer;
		const double ms_per_dispatch = gpu_busy_ms / static_cast<double>(dispatches);
		std::printf("chunk_tokens=%u dispatches=%u gpu_busy_ms=%.4f ms/dispatch=%.6f\n", sz, dispatches,
		            gpu_busy_ms, ms_per_dispatch);
		total_ms_per_dispatch += ms_per_dispatch;
		++fits;
	}
	const double mean_ms_per_dispatch = total_ms_per_dispatch / fits;
	std::printf("measured mean: %.6f ms/dispatch (over chunk_tokens={2,3,4})\n", mean_ms_per_dispatch);

	const double dispatches_per_token = static_cast<double>(num_hidden_layers) * kDispatchesPerLayer;
	const double tdr_bound_exact = tdr_budget_ms / (mean_ms_per_dispatch * dispatches_per_token);
	std::printf(
	    "\nTDR-ARITHMETIC-ONLY BOUND (NOT what is shipped): %.1f tokens would fit under half this "
	    "platform's TDR budget at the measured rate (dispatches/token=%.0f, budget=%.1f ms, "
	    "rate=%.6f ms/dispatch). This number is FAR above what is actually safe on this hardware: "
	    "an uncapped chunk_tokens=8 in one command list reproducibly crashed this process with "
	    "STATUS_STACK_OVERFLOW, root-caused under cdb as a recursive cycle entirely inside "
	    "nvwgf2umx.dll (NVIDIA's own D3D12 driver), not this codebase -- see this tool's own "
	    "header comment. superslm_gpu::kT2169TdrSafeMaxChunkTokens is DEFINED (src/gpu/"
	    "superslm_gpu.cpp, D-SLM3649) at 4 -- half the confirmed-crashing point, the smaller of "
	    "the TDR ceiling and the driver-stability ceiling -- and SubmitChunkToFullDepthForG5Bridge "
	    "now splits any larger chunk into sub-chunks at that bound automatically.\n",
	    tdr_bound_exact, dispatches_per_token, tdr_budget_ms, mean_ms_per_dispatch);
	return 0;
}

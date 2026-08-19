// T-2180 (Rung 6, Brunel build-seat verification tool -- not part of the committed T-2178 red
// suite, matching tools/t2169_rung2b_selfcheck.cpp's own precedent for a builder's own
// bench-proof harness): the product cell -- tok/s on a long forced span (jump-forward), across
// THREE arms, on this device. This is 1.1's headline GPU number (design Sec10's own "a
// round-trip-count reduction... should show a multi-fold reduction in forced-span wall time...
// dominated by removed fence-wait latency").
//
// T-2184 remedy C1 (Brunel fix round 1, Claude/Poirot/efeb9ba-t2184-t2169-gpu-batched-prefill-
// review.md; D-SLM3662): the review's own C1 finding -- this tool's prior "shipped-per-token" arm
// (N separate single-token calls into `SslmGpuSeqPrefillPromptForG5Bridge`) silently became the
// batched primitive at `chunk_len=1` once Rungs 3/4 rewrote that function's body, so the prior
// headline was chunk-of-1 vs chunk-of-N, never shipped-vs-batched. Three arms now, so both real
// levers this ticket delivers are visible separately:
//
//   1. genuinely-shipped: `SslmGpuSeqPrefillPromptPreBatchingBenchOnly` (gpu_1p0_bench_bridge.h),
//      the real `e35edc1` per-token composition -- one submit-and-fence round-trip PER LAYER (28
//      for Qwen2.5-1.5B, dispatch_budget=24), issued once per token.
//   2. chunk-of-1: N separate single-token calls into the PUBLIC bridge
//      (SslmGpuSeqPrefillPromptForG5Bridge) -- one submit-and-fence round-trip per TOKEN, the
//      residual per-token fence cost the chunk primitive still pays at chunk_len=1.
//   3. chunk-of-N: ONE public-bridge call carrying the whole forced span -- one submit-and-fence
//      round-trip per TDR-safe sub-chunk (kT2169TdrSafeMaxChunkTokens tokens each).
//
// The headline (1 vs 3) replaces every 1.10x-1.14x figure this ticket previously reported. The
// decomposition (1->2, 2->3) separates the two levers: the 28-round-trip-to-1-round-trip-per-token
// collapse Rungs 2-4 actually delivered (1->2), and the residual chunk-of-1-to-chunk-of-N gain
// from batching multiple tokens per command list (2->3).
//
// Usage: t2180_rung6_tokps <model.sslm> [span_len]
#include <chrono>
#include <cstdio>
#include <vector>

#include "superslm/gpu_1p0.h"
#include "superslm/gpu_1p0_bench_bridge.h"
#include "superslm/model.h"

using namespace superslm;

namespace {
constexpr uint32_t kDispatchBudget = 24;

std::vector<int32_t> MakeTokens(int32_t start, size_t n) {
	std::vector<int32_t> v(n);
	for (size_t i = 0; i < n; ++i) v[i] = start + static_cast<int32_t>(i);
	return v;
}
}  // namespace

int main(int argc, char** argv) {
	if (argc < 2) {
		std::fprintf(stderr, "usage: %s <model.sslm> [span_len]\n", argv[0]);
		return 2;
	}
	const size_t span_len = argc >= 3 ? static_cast<size_t>(std::atoi(argv[2])) : 128;

	std::vector<uint8_t> bytes;
	SslmModelView view{};
	std::string err;
	{
		std::FILE* f = std::fopen(argv[1], "rb");
		if (!f) { std::fprintf(stderr, "cannot open %s\n", argv[1]); return 2; }
		std::fseek(f, 0, SEEK_END);
		const long sz = std::ftell(f);
		std::fseek(f, 0, SEEK_SET);
		bytes.resize(static_cast<size_t>(sz));
		std::fread(bytes.data(), 1, bytes.size(), f);
		std::fclose(f);
	}
	if (SslmModel::Load(bytes.data(), bytes.size(), view, &err) != SslmModelStatus::Ok) {
		std::fprintf(stderr, "load failed: %s\n", err.c_str());
		return 2;
	}

	SslmGpuContext* ctx = nullptr;
	if (sslm_gpu_context_create(GpuContextConfig{}, &ctx) != SSLM_OK || !ctx) {
		std::fprintf(stderr, "context create failed\n");
		return 2;
	}
	SslmGpuModelHandle* model = nullptr;
	if (sslm_gpu_model_map(ctx, &view, GpuResidencyConfig{}, &model) != SSLM_OK || !model) {
		std::fprintf(stderr, "model map failed\n");
		return 2;
	}

	const int64_t context_cap = static_cast<int64_t>(span_len) + 32;
	const std::vector<int32_t> prime = {11, 22, 33};
	const std::vector<int32_t> span = MakeTokens(100, span_len);

	// Warm-up pass (residency/PSO caches, page faults) -- not timed, matching this project's own
	// established convention (tools/t2100_gpu_throughput.cpp). Warms all three arms' own code
	// paths: the bench-only pre-batching path, and the batched public bridge (chunk-of-1 and
	// chunk-of-N share the same underlying primitive post-Rungs-3/4).
	{
		SslmGpuSequenceHandle* seq = nullptr;
		sslm_gpu_seq_create(ctx, model, context_cap, &seq);
		SslmGpuSeqPrefillPromptPreBatchingBenchOnly(ctx, seq, prime.data(),
		                                             static_cast<int32_t>(prime.size()), kDispatchBudget);
		SslmGpuSeqPrefillPromptPreBatchingBenchOnly(ctx, seq, span.data(),
		                                             static_cast<int32_t>(span.size()), kDispatchBudget);
		sslm_gpu_seq_release(ctx, seq);
	}
	{
		SslmGpuSequenceHandle* seq = nullptr;
		sslm_gpu_seq_create(ctx, model, context_cap, &seq);
		SslmGpuSeqPrefillPromptForG5Bridge(ctx, seq, prime.data(), static_cast<int32_t>(prime.size()),
		                                    kDispatchBudget);
		SslmGpuSeqPrefillPromptForG5Bridge(ctx, seq, span.data(), static_cast<int32_t>(span.size()),
		                                    kDispatchBudget);
		sslm_gpu_seq_release(ctx, seq);
	}

	// Arm 1 -- genuinely-shipped: the real e35edc1 per-token composition, one submit-and-fence
	// round-trip PER LAYER, issued once per token.
	double shipped_ms = 0.0;
	{
		SslmGpuSequenceHandle* seq = nullptr;
		sslm_gpu_seq_create(ctx, model, context_cap, &seq);
		SslmGpuSeqPrefillPromptPreBatchingBenchOnly(ctx, seq, prime.data(),
		                                             static_cast<int32_t>(prime.size()), kDispatchBudget);
		const auto t0 = std::chrono::steady_clock::now();
		SslmGpuSeqPrefillPromptPreBatchingBenchOnly(ctx, seq, span.data(),
		                                             static_cast<int32_t>(span.size()), kDispatchBudget);
		const auto t1 = std::chrono::steady_clock::now();
		shipped_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
		sslm_gpu_seq_release(ctx, seq);
	}

	// Arm 2 -- chunk-of-1: span_len separate single-token calls into the PUBLIC bridge (the batched
	// primitive at chunk_len=1 -- one submit-and-fence round-trip per TOKEN).
	double chunk_of_1_ms = 0.0;
	{
		SslmGpuSequenceHandle* seq = nullptr;
		sslm_gpu_seq_create(ctx, model, context_cap, &seq);
		SslmGpuSeqPrefillPromptForG5Bridge(ctx, seq, prime.data(), static_cast<int32_t>(prime.size()),
		                                    kDispatchBudget);
		const auto t0 = std::chrono::steady_clock::now();
		for (int32_t t : span) {
			SslmGpuSeqPrefillPromptForG5Bridge(ctx, seq, &t, 1, kDispatchBudget);
		}
		const auto t1 = std::chrono::steady_clock::now();
		chunk_of_1_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
		sslm_gpu_seq_release(ctx, seq);
	}

	// Arm 3 -- chunk-of-N: ONE public-bridge call carrying the whole forced span (one
	// submit-and-fence round-trip per TDR-safe sub-chunk).
	double chunk_of_n_ms = 0.0;
	{
		SslmGpuSequenceHandle* seq = nullptr;
		sslm_gpu_seq_create(ctx, model, context_cap, &seq);
		SslmGpuSeqPrefillPromptForG5Bridge(ctx, seq, prime.data(), static_cast<int32_t>(prime.size()),
		                                    kDispatchBudget);
		const auto t0 = std::chrono::steady_clock::now();
		SslmGpuSeqPrefillPromptForG5Bridge(ctx, seq, span.data(), static_cast<int32_t>(span.size()),
		                                    kDispatchBudget);
		const auto t1 = std::chrono::steady_clock::now();
		chunk_of_n_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
		sslm_gpu_seq_release(ctx, seq);
	}

	sslm_gpu_model_unmap(ctx, model);
	sslm_gpu_context_destroy(ctx);

	const double shipped_tokps = span_len / (shipped_ms / 1000.0);
	const double chunk_of_1_tokps = span_len / (chunk_of_1_ms / 1000.0);
	const double chunk_of_n_tokps = span_len / (chunk_of_n_ms / 1000.0);
	std::printf("=== T-2180 Rung 6 product cell (T-2184 C1 remedy): span_len=%zu, RTX 2080S ===\n",
	            span_len);
	std::printf("1. genuinely-shipped (pre-batching, bench-only): %.3f ms total, %.2f ms/token, "
	            "%.2f tok/s\n",
	            shipped_ms, shipped_ms / span_len, shipped_tokps);
	std::printf("2. chunk-of-1 (public bridge, span_len single-token calls): %.3f ms total, "
	            "%.2f ms/token, %.2f tok/s\n",
	            chunk_of_1_ms, chunk_of_1_ms / span_len, chunk_of_1_tokps);
	std::printf("3. chunk-of-N (public bridge, one call carrying the whole span): %.3f ms total, "
	            "%.2f ms/token, %.2f tok/s\n",
	            chunk_of_n_ms, chunk_of_n_ms / span_len, chunk_of_n_tokps);
	std::printf("headline speedup (1 -> 3, genuinely-shipped vs chunked): %.2fx\n",
	            shipped_ms / chunk_of_n_ms);
	std::printf("decomposition: shipped -> chunk-of-1 (28-round-trip collapse): %.2fx\n",
	            shipped_ms / chunk_of_1_ms);
	std::printf("decomposition: chunk-of-1 -> chunk-of-N (residual batching gain): %.2fx\n",
	            chunk_of_1_ms / chunk_of_n_ms);
	return 0;
}

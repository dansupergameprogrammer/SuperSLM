// T-2180 (Rung 6, Brunel build-seat verification tool -- not part of the committed T-2178 red
// suite, matching tools/t2169_rung2b_selfcheck.cpp's own precedent for a builder's own
// bench-proof harness): the product cell -- tok/s on a long forced span (jump-forward), chunked
// (one public-bridge call carrying every forced token) vs shipped-per-token (N separate
// single-token public-bridge calls, the same call shape that shipped before this ticket), on
// this device, through the PUBLIC bridge (SslmGpuSeqPrefillPromptForG5Bridge). This is 1.1's
// headline GPU number (design Sec10's own "a round-trip-count reduction... should show a
// multi-fold reduction in forced-span wall time... dominated by removed fence-wait latency").
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
	// established convention (tools/t2100_gpu_throughput.cpp).
	{
		SslmGpuSequenceHandle* seq = nullptr;
		sslm_gpu_seq_create(ctx, model, context_cap, &seq);
		SslmGpuSeqPrefillPromptForG5Bridge(ctx, seq, prime.data(), static_cast<int32_t>(prime.size()),
		                                    kDispatchBudget);
		SslmGpuSeqPrefillPromptForG5Bridge(ctx, seq, span.data(), static_cast<int32_t>(span.size()),
		                                    kDispatchBudget);
		sslm_gpu_seq_release(ctx, seq);
	}

	// Shipped-per-token shape: span_len separate single-token public-bridge calls.
	double per_token_ms = 0.0;
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
		per_token_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
		sslm_gpu_seq_release(ctx, seq);
	}

	// Chunked shape: ONE public-bridge call carrying the whole forced span.
	double chunked_ms = 0.0;
	{
		SslmGpuSequenceHandle* seq = nullptr;
		sslm_gpu_seq_create(ctx, model, context_cap, &seq);
		SslmGpuSeqPrefillPromptForG5Bridge(ctx, seq, prime.data(), static_cast<int32_t>(prime.size()),
		                                    kDispatchBudget);
		const auto t0 = std::chrono::steady_clock::now();
		SslmGpuSeqPrefillPromptForG5Bridge(ctx, seq, span.data(), static_cast<int32_t>(span.size()),
		                                    kDispatchBudget);
		const auto t1 = std::chrono::steady_clock::now();
		chunked_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
		sslm_gpu_seq_release(ctx, seq);
	}

	sslm_gpu_model_unmap(ctx, model);
	sslm_gpu_context_destroy(ctx);

	const double per_token_tokps = span_len / (per_token_ms / 1000.0);
	const double chunked_tokps = span_len / (chunked_ms / 1000.0);
	std::printf("=== T-2180 Rung 6 product cell: span_len=%zu, RTX 2080S ===\n", span_len);
	std::printf("shipped-per-token shape: %.3f ms total, %.2f ms/token, %.2f tok/s\n", per_token_ms,
	            per_token_ms / span_len, per_token_tokps);
	std::printf("chunked (public bridge, one call) shape: %.3f ms total, %.2f ms/token, %.2f tok/s\n",
	            chunked_ms, chunked_ms / span_len, chunked_tokps);
	std::printf("speedup: %.2fx\n", per_token_ms / chunked_ms);
	return 0;
}

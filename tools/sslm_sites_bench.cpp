// sslm_sites_bench -- the attention and per-row sites plan's engine bench (rev 3.1, cell 10.1).
//
// One source, built against either library (the base's or the candidate's), so a ratio between two
// builds' readings is the slice's own effect. Reports only; nothing here gates a slice.
//
//   sslm_sites_bench kernel [--repeat=R]
//       Each per-row site at the Qwen2.5-0.5B geometry, through its public entry point with realistic
//       constants: RmsNormSite at 896, MlpActSite at 4,864, ResidualReconcileSite at 896. Prints the
//       best-of-R microseconds per call (each rep times a batch of calls over 16 distinct rows).
//   sslm_sites_bench pv [--repeat=R]
//       Slice S2: GemmProbQ15Accumulate at head_dim 64 (the 0.5B's) on realistic probability rows (every p
//       formed as the softmax forms it, Sum p <= 2^15) over one KV head's value rows. Prints best-of-R
//       microseconds per call at single widths, then the per-token cost at the 0.5B's full depth (24
//       layers x 14 query heads, one call per head per token): prefill of T tokens sums the calls at
//       widths 1..T and divides by T; decode at context C is one call at width C + 1 per head.
//   sslm_sites_bench prefill <artifact.sslm> <T> [--layers=L] [--repeat=R]
//       One sslm_prefill of T token ids at chunk_budget = T; best-of-R ms per prompt token.
//   sslm_sites_bench decode <artifact.sslm> <context> <D> [--layers=L] [--repeat=R]
//       Prefill `context` token ids, then D greedy sslm_decode_step calls; best-of-R ms per decode token.
//
// Token ids are 1 + (i mod 200), inside every synthetic artifact's vocabulary (tools/t2147's "ids:N").
// Slice S1's method (docs/attention-rowsites/s1/bench.md): kernel mode's per-call savings times the
// sites' per-token counts at full depth, checked against prefill and decode on reduced-layer artifacts.

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include "superslm/checked_chain_funnel.h"
#include "superslm/forward_sites.h"
#include "superslm/matmul.h"
#include "superslm/silu_lut_canonical.h"
#include "superslm/sslm_abi.h"

namespace {

using Clock = std::chrono::steady_clock;

struct Rng {
	uint64_t s;
	explicit Rng(uint64_t seed) : s(seed) {}
	uint64_t Next() {
		s += 0x9e3779b97f4a7c15ULL;
		uint64_t z = s;
		z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
		z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
		return z ^ (z >> 31);
	}
	int64_t InRange(int64_t lo, int64_t hi) {
		return lo + static_cast<int64_t>(Next() % (static_cast<uint64_t>(hi - lo) + 1ULL));
	}
};

volatile int64_t g_sink = 0;

template <class F>
double BestMicrosPerCall(int repeat, int calls, F&& f) {
	double best = 1e300;
	for (int r = 0; r < repeat; ++r) {
		const auto t0 = Clock::now();
		for (int c = 0; c < calls; ++c) f(c);
		const auto t1 = Clock::now();
		best = std::min(best, std::chrono::duration<double, std::micro>(t1 - t0).count() / calls);
	}
	return best;
}

int KernelMode(int repeat) {
	using superslm::CarriedScale;
	constexpr int kRows = 16;
	Rng rng(0x5155454E424E4348ULL);
	const size_t hidden = 896, inter = 4864;
	std::vector<std::vector<int8_t>> h(kRows, std::vector<int8_t>(hidden)), gate(kRows, std::vector<int8_t>(inter)),
	    up(kRows, std::vector<int8_t>(inter)), branch(kRows, std::vector<int8_t>(hidden)),
	    stream(kRows, std::vector<int8_t>(hidden));
	for (int r = 0; r < kRows; ++r) {
		for (auto& v : h[r]) v = static_cast<int8_t>(rng.InRange(-127, 127));
		for (auto& v : gate[r]) v = static_cast<int8_t>(rng.InRange(-127, 127));
		for (auto& v : up[r]) v = static_cast<int8_t>(rng.InRange(-127, 127));
		for (auto& v : branch[r]) v = static_cast<int8_t>(rng.InRange(-127, 127));
		for (auto& v : stream[r]) v = static_cast<int8_t>(rng.InRange(-127, 127));
	}
	std::vector<int32_t> g(hidden);
	for (auto& v : g) v = static_cast<int32_t>(rng.InRange(-300, 300));
	const CarriedScale unit{INT64_C(1073741824), 0};
	std::vector<int8_t> out(inter);
	CarriedScale scale{};
	int bad = 0;

	const double norm = BestMicrosPerCall(repeat, 256, [&](int c) {
		bad += superslm::RmsNormSite(h[c % kRows].data(), g.data(), hidden, CarriedScale{}, unit, out.data(), &scale) !=
		       superslm::SslmForwardStatus::Ok;
	});
	const double silu = BestMicrosPerCall(repeat, 64, [&](int c) {
		bad += superslm::MlpActSite(gate[c % kRows].data(), {INT64_C(1073741824), -34}, up[c % kRows].data(),
		                            {INT64_C(1340958474), -18}, inter, superslm::kSiluLutCanonicalTable, unit,
		                            out.data(), &scale) != superslm::SslmForwardStatus::Ok;
	});
	const double residual = BestMicrosPerCall(repeat, 256, [&](int c) {
		bad += superslm::ResidualReconcileSite(branch[c % kRows].data(), {INT64_C(1234567890), -37},
		                                       stream[c % kRows].data(), {INT64_C(1987654321), -44}, hidden, unit,
		                                       out.data(), &scale) != superslm::SslmForwardStatus::Ok;
	});
	g_sink = g_sink + out[0];
	std::printf("kernel best-of-%d us/call: rmsnorm_896 %.3f  mlp_act_4864 %.3f  residual_896 %.3f  (refused %d)\n",
	            repeat, norm, silu, residual, bad);
	return bad == 0 ? 0 : 1;
}

// Slice S2's bench (see the header): the prob·V call at the 0.5B's head_dim over realistic rows.
int PvMode(int repeat) {
	constexpr size_t kHeadDim = 64, kMaxWidth = 1025, kLayers = 24, kHeads = 14;
	Rng rng(0x5332505642454E43ULL);
	std::vector<int8_t> values(kMaxWidth * kHeadDim);
	for (auto& v : values) v = static_cast<int8_t>(rng.InRange(-128, 127));
	// probs[w] is a realistic row of width w: weights 2^(0..12) with jitter, p = floor(e * 2^15 / Sum e).
	std::vector<std::vector<int64_t>> probs(kMaxWidth + 1);
	for (size_t w = 1; w <= kMaxWidth; ++w) {
		std::vector<int64_t> e(w);
		int64_t total = 0;
		for (auto& x : e) {
			const int sh = static_cast<int>(rng.InRange(0, 12));
			x = (INT64_C(1) << sh) + rng.InRange(0, (INT64_C(1) << sh) - 1);
			total += x;
		}
		probs[w].resize(w);
		for (size_t k = 0; k < w; ++k) probs[w][k] = (e[k] << 15) / total;
	}
	std::vector<int64_t> out(kHeadDim);
	const auto call = [&](size_t w) {
		superslm::GemmProbQ15Accumulate(probs[w].data(), values.data(), w, kHeadDim, out.data());
		g_sink = g_sink + out[0];
	};
	// Single widths.
	std::printf("pv best-of-%d us/call at head_dim 64:", repeat);
	for (size_t w : {size_t{1}, size_t{128}, size_t{301}, size_t{512}, size_t{601}, size_t{1024}}) {
		const int calls = w < 64 ? 4096 : 256;
		std::printf("  w=%zu %.4f", w, BestMicrosPerCall(repeat, calls, [&](int) { call(w); }));
	}
	std::printf("\n");
	// Prefill: every width 1..T once (one head of one layer), best of R, scaled to 24 layers x 14 heads / T.
	for (size_t T : {size_t{128}, size_t{512}, size_t{1024}}) {
		const double us = BestMicrosPerCall(repeat, 1, [&](int) {
			for (size_t w = 1; w <= T; ++w) call(w);
		});
		std::printf("pv prefill T=%zu: %.4f ms/token at 24 layers x 14 heads\n", T,
		            us * kLayers * kHeads / static_cast<double>(T) / 1000.0);
	}
	// Decode at context C: one call at width C + 1 per head per layer.
	for (size_t C : {size_t{300}, size_t{600}}) {
		const double us = BestMicrosPerCall(repeat, 256, [&](int) { call(C + 1); });
		std::printf("pv decode ctx=%zu: %.4f ms/token at 24 layers x 14 heads\n", C, us * kLayers * kHeads / 1000.0);
	}
	return 0;
}

bool ReadFile(const char* path, std::vector<uint8_t>& out) {
	std::ifstream f(path, std::ios::binary);
	if (!f) return false;
	out.assign(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
	return !out.empty();
}

[[noreturn]] void Fail(const char* what, int st) {
	std::fprintf(stderr, "sslm_sites_bench: %s failed (status %d)\n", what, st);
	std::exit(2);
}

// One fresh model, pool, workspace and sequence; prefill T ids in one call, then D decode steps.
// Returns the prefill and decode wall times in ms.
void RunOnce(const std::vector<uint8_t>& bytes, int32_t layers, int32_t T, int32_t D, double* prefill_ms,
             double* decode_ms) {
	sslm_model model = nullptr;
	sslm_status st = sslm_model_map(bytes.data(), bytes.size(), &model);
	if (st != SSLM_OK) Fail("sslm_model_map", st);
	const size_t kv_required = sslm_kv_block_size(model) + sslm_kv_pool_overhead_size(model, 1);
	std::vector<uint8_t> pool_raw(kv_required + SSLM_ABI_ALIGNMENT_BYTES);
	void* pa = pool_raw.data();
	size_t ps = pool_raw.size();
	std::align(SSLM_ABI_ALIGNMENT_BYTES, kv_required, pa, ps);
	sslm_kv_pool pool = nullptr;
	if ((st = sslm_kv_pool_create(model, pa, kv_required, 1, &pool)) != SSLM_OK) Fail("sslm_kv_pool_create", st);
	sslm_config cfg{};
	cfg.max_batch = 1;
	cfg.max_chunk_budget = T;
	cfg.max_layer_budget = layers;
	const size_t wsb = sslm_workspace_size(model, &cfg);
	std::vector<uint8_t> ws_raw(wsb + SSLM_ABI_ALIGNMENT_BYTES);
	void* wa = ws_raw.data();
	size_t wsz = ws_raw.size();
	std::align(SSLM_ABI_ALIGNMENT_BYTES, wsb, wa, wsz);
	sslm_workspace ws = nullptr;
	if ((st = sslm_workspace_create(model, &cfg, wa, wsb, &ws)) != SSLM_OK) Fail("sslm_workspace_create", st);
	sslm_seq seq = nullptr;
	if ((st = sslm_seq_create(model, &pool, &seq)) != SSLM_OK) Fail("sslm_seq_create", st);
	std::vector<int32_t> ids(static_cast<size_t>(T));
	for (int32_t i = 0; i < T; ++i) ids[static_cast<size_t>(i)] = 1 + (i % 200);
	const auto t0 = Clock::now();
	int32_t consumed = 0;
	if ((st = sslm_prefill(model, seq, ids.data(), T, T, SSLM_SPAN_PROMPT, ws, &consumed)) != SSLM_OK || consumed != T)
		Fail("sslm_prefill", st);
	const auto t1 = Clock::now();
	if (D > 0) {
		sslm_decode_params params{};
		if ((st = sslm_decode_params_init(model, SSLM_DECODE_MODE_GREEDY, layers, &params)) != SSLM_OK)
			Fail("sslm_decode_params_init", st);
		for (int32_t d = 0; d < D; ++d) {
			int32_t next = -1;
			if ((st = sslm_decode_step(model, &seq, 1, &params, ws, &next)) != SSLM_OK) Fail("sslm_decode_step", st);
		}
	}
	const auto t2 = Clock::now();
	*prefill_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
	*decode_ms = std::chrono::duration<double, std::milli>(t2 - t1).count();
	sslm_seq_release(seq);
	sslm_workspace_destroy(ws);
	sslm_kv_pool_destroy(pool);
	sslm_model_unmap(model);
}

}  // namespace

int main(int argc, char** argv) {
	if (argc < 2) {
		std::fprintf(stderr,
		             "usage: %s kernel [--repeat=R] | pv [--repeat=R] | prefill <artifact> <T> [--layers=L] [--repeat=R] | "
		             "decode <artifact> <context> <D> [--layers=L] [--repeat=R]\n",
		             argv[0]);
		return 2;
	}
	int repeat = 7;
	int32_t layers = 1;
	for (int i = 2; i < argc; ++i) {
		if (std::strncmp(argv[i], "--repeat=", 9) == 0) repeat = std::max(1, std::atoi(argv[i] + 9));
		if (std::strncmp(argv[i], "--layers=", 9) == 0) layers = std::max(1, std::atoi(argv[i] + 9));
	}
	const std::string mode = argv[1];
	if (mode == "kernel") return KernelMode(repeat);
	if (mode == "pv") return PvMode(repeat);
	if ((mode == "prefill" && argc >= 4) || (mode == "decode" && argc >= 5)) {
		std::vector<uint8_t> bytes;
		if (!ReadFile(argv[2], bytes)) Fail("reading the artifact", 0);
		const int32_t T = std::atoi(argv[3]);
		const int32_t D = mode == "decode" ? std::atoi(argv[4]) : 0;
		double best = 1e300;
		for (int r = 0; r < repeat; ++r) {
			double p = 0, d = 0;
			RunOnce(bytes, layers, T, D, &p, &d);
			best = std::min(best, mode == "prefill" ? p / T : d / D);
		}
		std::printf("%s best-of-%d ms/token: %.4f (T=%d D=%d layers=%d)\n", mode.c_str(), repeat, best, T, D, layers);
		return 0;
	}
	std::fprintf(stderr, "bad arguments\n");
	return 2;
}

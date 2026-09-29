// Adopted in plan rev 3 as the seed of cell 10.0 (consumer reach). Verbatim copy of
// reviews/adversary-strike-round1-plugin_chunk_probe.cpp; the cell's spec is plan.md §9 10.0.
// Adversary probe, 2026-09-25: which num_tokens does the SuperSLM Unreal plugin's CPU prefill
// actually hand GemmInt8Accumulate, and how many of those calls would slice 1's tiled path
// (num_tokens >= kTiledMinTokens = 8) take?
//
// Part A (pure): PlanPrefillChunk / PrefillChunkCostMs / PromptChunkPositionSum copied VERBATIM
// from Wizard develop@5c1a93d Plugins/SuperSLMUnreal/Source/SuperSLMUnreal/Private/
// SuperSLMSubsystem.cpp:473-503 (Unreal int32/int64 -> int32_t/int64_t), swept over TickBudgetMs
// and start depth at the plugin's shipped defaults (SuperSLMRuntimeConfig.h: BudgetHeadroom 0.7,
// PromptTokenCostMs 36.0, PromptTokenCostPerPositionMs 0.021).
//
// Part B (end to end): SuperSLM main@1ca2803 libsuperslm.a (Release), linked with
// -Wl,--wrap=_ZN8superslm18GemmInt8AccumulateEPKaS1_mmmPl so every call the real forward pass
// makes is counted by num_tokens. A 64-token prompt is prefilled through the public ABI exactly as
// DispatchSequencePrefill (SuperSLMSubsystem.cpp:3444-3496) drives it: each call's chunk_budget is
// PlanPrefillChunk(min(MaxPrefillChunkBudget, Left), ContextUsed, TickBudgetMs * 0.7, ...), with
// count = Left. Controls: TickBudgetMs = 1000 (the value the plugin's own test fixtures use) must
// produce tiled-eligible calls; a whole-prompt call (chunk_budget = count, the SuperEmbedder shape)
// must too.
//
// Build: g++ -std=c++20 -O2 -I<SuperSLM>/include plugin_chunk_probe.cpp <build>/libsuperslm.a \
//        -Wl,--wrap=_ZN8superslm18GemmInt8AccumulateEPKaS1_mmmPl -pthread -o plugin_chunk_probe
// Run:   ./plugin_chunk_probe <artifact.sslm> <num_hidden_layers>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <map>
#include <memory>
#include <vector>

#include "superslm/sslm_abi.h"

typedef int32_t int32;
typedef int64_t int64;

// ---- verbatim from SuperSLMSubsystem.cpp:473-503 ----
int64 PromptChunkPositionSum(int32 Chunk, int64 StartDepth)
{
	return static_cast<int64>(Chunk) * StartDepth + static_cast<int64>(Chunk) * (Chunk - 1) / 2;
}
double PrefillChunkCostMs(int32 Chunk, int64 StartDepth, double PromptTokenCostMs, double PromptTokenCostPerPositionMs)
{
	return Chunk * PromptTokenCostMs + PromptTokenCostPerPositionMs * static_cast<double>(PromptChunkPositionSum(Chunk, StartDepth));
}
int32 PlanPrefillChunk(int32 MaxChunk, int64 StartDepth, double ShareMs, double PromptTokenCostMs, double PromptTokenCostPerPositionMs)
{
	int32 Chunk = 1;
	for (int32 C = 2; C <= MaxChunk; ++C)
	{
		if (PrefillChunkCostMs(C, StartDepth, PromptTokenCostMs, PromptTokenCostPerPositionMs) <= ShareMs)
		{
			Chunk = C;
		}
		else
		{
			break;
		}
	}
	return Chunk;
}
// ---- end verbatim ----

constexpr double kHeadroom = 0.7, kPromptMs = 36.0, kPromptPerPosMs = 0.021;  // shipped defaults
constexpr size_t kTiledMinTokens = 8;                                           // plan §4.1

std::map<size_t, long> g_hist;  // num_tokens -> GemmInt8Accumulate calls
extern "C" void __real__ZN8superslm18GemmInt8AccumulateEPKaS1_mmmPl(const int8_t*, const int8_t*, size_t,
                                                                  size_t, size_t, int64_t*);
extern "C" void __wrap__ZN8superslm18GemmInt8AccumulateEPKaS1_mmmPl(const int8_t* a, const int8_t* w,
                                                                  size_t m, size_t k, size_t n, int64_t* o) {
	++g_hist[m];
	__real__ZN8superslm18GemmInt8AccumulateEPKaS1_mmmPl(a, w, m, k, n, o);
}

static void Die(const char* what, int st) { std::fprintf(stderr, "FATAL %s status=%d\n", what, st); std::exit(1); }

// Prefill `count` tokens the way DispatchSequencePrefill does (tick_ms > 0), or in one whole call
// (tick_ms == 0, the SuperEmbedder shape). Returns the chunk sequence used.
static std::vector<int32_t> Run(const std::vector<uint8_t>& bytes, int32_t layers, int32_t count,
                                int32_t max_chunk, double tick_ms) {
	sslm_model model = nullptr;
	sslm_status st = sslm_model_map(bytes.data(), bytes.size(), &model);
	if (st != SSLM_OK) Die("model_map", st);
	const size_t kvb = sslm_kv_block_size(model), kvo = sslm_kv_pool_overhead_size(model, 1);
	std::vector<uint8_t> pool_raw(kvb + kvo + SSLM_ABI_ALIGNMENT_BYTES);
	void* pa = pool_raw.data(); size_t ps = pool_raw.size();
	std::align(SSLM_ABI_ALIGNMENT_BYTES, kvb + kvo, pa, ps);
	sslm_kv_pool pool = nullptr;
	if ((st = sslm_kv_pool_create(model, pa, kvb + kvo, 1, &pool)) != SSLM_OK) Die("pool", st);
	sslm_config cfg{}; cfg.max_batch = 1; cfg.max_chunk_budget = max_chunk; cfg.max_layer_budget = layers;
	const size_t wsb = sslm_workspace_size(model, &cfg);
	std::vector<uint8_t> ws_raw(wsb + SSLM_ABI_ALIGNMENT_BYTES);
	void* wa = ws_raw.data(); size_t wsz = ws_raw.size();
	std::align(SSLM_ABI_ALIGNMENT_BYTES, wsb, wa, wsz);
	sslm_workspace ws = nullptr;
	if ((st = sslm_workspace_create(model, &cfg, wa, wsb, &ws)) != SSLM_OK) Die("ws", st);
	sslm_seq seq = nullptr;
	if ((st = sslm_seq_create(model, &pool, &seq)) != SSLM_OK) Die("seq", st);
	std::vector<int32_t> toks(count);
	for (int32_t i = 0; i < count; ++i) toks[i] = 1 + (i % 200);
	std::vector<int32_t> chunks;
	int32_t done = 0; int64_t depth = 0;
	while (done < count) {
		const int32_t left = count - done;
		int32_t chunk = left;
		if (tick_ms > 0) chunk = PlanPrefillChunk(left < max_chunk ? left : max_chunk, depth, tick_ms * kHeadroom,
		                                          kPromptMs, kPromptPerPosMs);
		int32_t consumed = 0;
		st = sslm_prefill(model, seq, toks.data() + done, left, chunk, SSLM_SPAN_PROMPT, ws, &consumed);
		if (st != SSLM_OK || consumed != chunk) Die("prefill", st);
		chunks.push_back(chunk);
		done += consumed; depth += consumed;
	}
	sslm_seq_release(seq); sslm_workspace_destroy(ws); sslm_kv_pool_destroy(pool); sslm_model_unmap(model);
	return chunks;
}

int main(int argc, char** argv) {
	std::printf("== Part A: PlanPrefillChunk (verbatim) at shipped defaults, MaxPrefillChunkBudget 64 ==\n");
	std::printf("%12s |", "TickBudgetMs");
	const int64 depths[] = {0, 64, 512, 2048, 4095};
	for (int64 d : depths) std::printf(" d=%-5lld", (long long)d);
	std::printf("\n");
	const double ticks[] = {6.94, 8.33, 11.1, 16.6, 33.3, 50.0, 100.0, 250.0, 411.0, 412.0, 1000.0};
	for (double t : ticks) {
		std::printf("%12.2f |", t);
		for (int64 d : depths) std::printf(" %-7d", PlanPrefillChunk(64, d, t * kHeadroom, kPromptMs, kPromptPerPosMs));
		std::printf("\n");
	}
	double lo = 0, hi = 100000;
	for (int i = 0; i < 200; ++i) { double mid = (lo + hi) / 2; (PlanPrefillChunk(64, 0, mid * kHeadroom, kPromptMs, kPromptPerPosMs) >= 8 ? hi : lo) = mid; }
	std::printf("smallest TickBudgetMs giving chunk >= 8 at depth 0: %.3f ms (%.2f Hz)\n\n", hi, 1000.0 / hi);

	if (argc < 3) return 0;
	std::vector<uint8_t> bytes;
	{ std::ifstream f(argv[1], std::ios::binary); bytes.assign(std::istreambuf_iterator<char>(f), {}); }
	const int32_t layers = std::atoi(argv[2]);
	std::printf("== Part B: real forward pass, 64-token prompt, artifact %s ==\n", argv[1]);
	struct Case { const char* name; double tick; } cases[] = {
	    {"plugin, 120 Hz (8.33 ms)", 8.33}, {"plugin, 60 Hz (16.6 ms)", 16.6}, {"plugin, 30 Hz (33.3 ms)", 33.3},
	    {"plugin, 100 ms", 100.0}, {"CONTROL plugin, 1000 ms (test fixture value)", 1000.0},
	    {"CONTROL whole prompt (embedder shape)", 0.0}};
	for (const Case& c : cases) {
		g_hist.clear();
		std::vector<int32_t> ch = Run(bytes, layers, 64, 64, c.tick);
		long total = 0, tiled = 0;
		for (auto& [m, n] : g_hist) { total += n; if (m >= kTiledMinTokens) tiled += n; }
		std::printf("%-46s calls=%zu chunks[0..]=", c.name, ch.size());
		for (size_t i = 0; i < ch.size() && i < 6; ++i) std::printf("%d ", ch[i]);
		std::printf("| GEMM calls %ld, M>=8 (tiled-eligible) %ld | M hist:", total, tiled);
		for (auto& [m, n] : g_hist) std::printf(" M=%zu:%ld", m, n);
		std::printf("\n");
	}
	return 0;
}

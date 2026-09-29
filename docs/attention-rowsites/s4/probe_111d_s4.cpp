// S4 re-derivation of 11.1(d)'s softmax data term (docs/attention-rowsites/s4/softmax-data-terms.txt). A copy of
// probes/rev3/probe_111d.cpp (plan of record's probe directory) with a --wrap shim; build:
//   g++ -std=c++20 -O2 -I<base>/include -I<base>/tools probe_111d_s4.cpp <base>/build/libsuperslm.a \
//     -Wl,--wrap=_ZN8superslm13SoftmaxRowQ15EPKlmlllPl -Wl,--wrap=_ZN8superslm21GemmProbQ15AccumulateEPKlPKammPl
//   ./a.out <p05_l1.sslm> direct
// Probe for plan rev 3, cell 11.1(d): counts, per window, what each §3.6 counter would count,
// on the shipped c3e0004 forward (hooks added only at function entries in a scratch copy).
// Mode "direct": RunLayerLoopChunkBatched over 128 tokens, then 32 x RunLayerLoop, one
// workspace, trace hook installed, no final norm, embeds inside the windows with the same hook.
// Mode "abi": sslm_prefill(128) then 32 x sslm_decode_step (greedy), as the strike describes.
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "superslm/forward_sites.h"
#include "superslm/model.h"
#include "superslm/sslm_abi.h"
#include "superslm/trace_hook.h"
#include "sslm_marshal.h"

using namespace superslm;
using superslm_marshal::LayerBacking;
using superslm_marshal::MarshalLayer;
using superslm_marshal::PreflightScanWscFolds;
using superslm_marshal::ReadCarriedScale;
using superslm_marshal::ReadFile;

struct Counts {
	uint64_t softmax = 0, softmax_guard_fail = 0, pv = 0, pv_fail = 0, pv_hd16_fail = 0;
	uint64_t requant = 0, norm_taken = 0, norm_skip = 0, silu_taken = 0, silu_skip = 0, land_taken = 0,
	         land_skip = 0, chain_records = 0, kv_records = 0;
	std::map<std::string, uint64_t> by_site;
};
static Counts g;

extern "C" void sslm_probe_softmax(const int64_t* s, size_t w, int64_t q_ln2, int64_t q_b, int64_t q_c) {
	if (w == 0) return;
	++g.softmax;
	bool ok = w <= (size_t(1) << 14) && q_ln2 >= 1 && q_c >= 0;
	if (ok) {
		const __int128 M = (__int128)q_b * q_b + q_c;
		ok = M >= 1 && M <= ((__int128)1 << 47) && q_ln2 <= 2 * q_b + 1;
	}
	if (ok)
		for (size_t i = 0; i < w; ++i)
			if (s[i] > (int64_t(1) << 61) || s[i] < -(int64_t(1) << 61)) ok = false;
	if (!ok) ++g.softmax_guard_fail;
}
extern "C" void sslm_probe_pv(const int64_t* p, size_t w, size_t hd) {
	++g.pv;
	if (hd % 16 != 0) ++g.pv_hd16_fail;
	int64_t sum = 0;
	bool ok = true;
	for (size_t k = 0; k < w; ++k) {
		if (p[k] < 0 || p[k] > 32767) ok = false;
		sum += p[k];
	}
	if (sum > 32768) ok = false;
	if (!ok) { ++g.pv_fail; if (w > 1) { int64_t mx = 0; size_t at = 0, nz = 0; for (size_t k = 0; k < w; ++k) { if (p[k] > mx) { mx = p[k]; at = k; } if (p[k]) ++nz; } std::printf("pv fail: width=%zu sum=%lld max=%lld at=%zu nonzero=%zu (call %llu)\n", w, (long long)sum, (long long)mx, at, nz, (unsigned long long)g.pv); } }
}
extern "C" void sslm_probe_requant(size_t) { ++g.requant; }
extern "C" void sslm_probe_site(int kind, size_t n) {
	const bool taken = n >= 512;
	if (kind == 0) (taken ? g.norm_taken : g.norm_skip)++;
	if (kind == 1) (taken ? g.silu_taken : g.silu_skip)++;
	if (kind == 2) (taken ? g.land_taken : g.land_skip)++;
}

// S4 re-derivation on the S3 head: SoftmaxRowQ15 and GemmProbQ15Accumulate reached through -Wl,--wrap from
// forward_sites.o (no source edit of the base), each row classified by the plan's guard copy above.
extern "C" {
bool __real__ZN8superslm13SoftmaxRowQ15EPKlmlllPl(const int64_t*, size_t, int64_t, int64_t, int64_t, int64_t*);
}
struct CStat { int64_t qln2_min = INT64_MAX, qln2_max = 0, qb_min = INT64_MAX, qb_max = 0, qc_min = INT64_MAX, qc_max = 0;
  __int128 M_min = ((__int128)1) << 100, M_max = 0; int64_t spread_max = 0; uint64_t clipped = 0, elems = 0; };
static CStat gc;
extern "C" bool __wrap__ZN8superslm13SoftmaxRowQ15EPKlmlllPl(const int64_t* s, size_t w, int64_t ql, int64_t qb, int64_t qc, int64_t* out) {
  sslm_probe_softmax(s, w, ql, qb, qc);
  if (w) { gc.qln2_min = std::min(gc.qln2_min, ql); gc.qln2_max = std::max(gc.qln2_max, ql); gc.qb_min = std::min(gc.qb_min, qb);
    gc.qb_max = std::max(gc.qb_max, qb); gc.qc_min = std::min(gc.qc_min, qc); gc.qc_max = std::max(gc.qc_max, qc);
    __int128 M = (__int128)qb * qb + qc; if (M < gc.M_min) gc.M_min = M; if (M > gc.M_max) gc.M_max = M;
    int64_t mx = s[0], mn = s[0]; for (size_t k = 0; k < w; ++k) { mx = std::max(mx, s[k]); mn = std::min(mn, s[k]); }
    gc.spread_max = std::max(gc.spread_max, mx - mn);
    for (size_t k = 0; k < w; ++k) { ++gc.elems; if (mx - s[k] >= 30 * ql) ++gc.clipped; } }
  return __real__ZN8superslm13SoftmaxRowQ15EPKlmlllPl(s, w, ql, qb, qc, out);
}
extern "C" {
void __real__ZN8superslm21GemmProbQ15AccumulateEPKlPKammPl(const int64_t*, const int8_t*, size_t, size_t, int64_t*);
}
extern "C" void __wrap__ZN8superslm21GemmProbQ15AccumulateEPKlPKammPl(const int64_t* p, const int8_t* v, size_t w, size_t hd, int64_t* o) {
  sslm_probe_pv(p, w, hd);
  __real__ZN8superslm21GemmProbQ15AccumulateEPKlPKammPl(p, v, w, hd, o);
}
static void PrintC() { std::printf("softmax constants over all rows: q_ln2 [%lld, %lld] q_b [%lld, %lld] q_c [%lld, %lld] M [%lld, %lld] "
  "max score spread %lld; elements %llu, clipped (z = 30) %llu\n", (long long)gc.qln2_min, (long long)gc.qln2_max, (long long)gc.qb_min,
  (long long)gc.qb_max, (long long)gc.qc_min, (long long)gc.qc_max, (long long)gc.M_min, (long long)gc.M_max, (long long)gc.spread_max,
  (unsigned long long)gc.elems, (unsigned long long)gc.clipped); }

static void Hook(const SslmChainTraceRecord* c, const SslmKvLandingTraceRecord* kv, void*) {
	if (c) {
		++g.chain_records;
		std::string s(c->site);
		const size_t dot = s.rfind('.');
		std::string key = s;
		if (s.rfind("layer", 0) == 0) key = s.substr(s.find('.') + 1);
		++g.by_site[key];
	}
	if (kv) ++g.kv_records;
}
static void Print(const char* name, const Counts& a, const Counts& b) {
	std::printf("[%s] softmax=%llu guard_fail=%llu pv=%llu pv_int16_fail=%llu pv_hd16_fail=%llu requant=%llu "
	            "chain_records=%llu kv_records=%llu norm_taken=%llu norm_skip=%llu silu_taken=%llu silu_skip=%llu "
	            "land_taken=%llu land_skip=%llu\n",
	            name, (unsigned long long)(b.softmax - a.softmax),
	            (unsigned long long)(b.softmax_guard_fail - a.softmax_guard_fail), (unsigned long long)(b.pv - a.pv),
	            (unsigned long long)(b.pv_fail - a.pv_fail), (unsigned long long)(b.pv_hd16_fail - a.pv_hd16_fail),
	            (unsigned long long)(b.requant - a.requant), (unsigned long long)(b.chain_records - a.chain_records),
	            (unsigned long long)(b.kv_records - a.kv_records), (unsigned long long)(b.norm_taken - a.norm_taken),
	            (unsigned long long)(b.norm_skip - a.norm_skip), (unsigned long long)(b.silu_taken - a.silu_taken),
	            (unsigned long long)(b.silu_skip - a.silu_skip), (unsigned long long)(b.land_taken - a.land_taken),
	            (unsigned long long)(b.land_skip - a.land_skip));
	std::printf("[%s] records by site:", name);
	for (auto& [k, v] : b.by_site) {
		auto it = a.by_site.find(k);
		const uint64_t d = v - (it == a.by_site.end() ? 0 : it->second);
		if (d) std::printf(" %s=%llu", k.c_str(), (unsigned long long)d);
	}
	std::printf("\n");
}

static int32_t Tok(size_t i) { return static_cast<int32_t>((i * 37 + 11) % 256); }

int main(int argc, char** argv) {
	if (argc < 3) return std::fprintf(stderr, "usage: probe <model.sslm> direct|abi [T] [D]\n"), 2;
	const size_t T = argc > 3 ? std::stoul(argv[3]) : 128, D = argc > 4 ? std::stoul(argv[4]) : 32;
	std::vector<uint8_t> bytes;
	if (!ReadFile(argv[1], bytes)) return 1;
	const std::string mode = argv[2];
	if (mode == "abi") {
		sslm_model model = nullptr;
		if (sslm_model_map(bytes.data(), bytes.size(), &model) != SSLM_OK) return 3;
		const size_t kvb = sslm_kv_block_size(model), req = kvb + sslm_kv_pool_overhead_size(model, 1);
		std::vector<uint8_t> praw(req + SSLM_ABI_ALIGNMENT_BYTES);
		void* pa = praw.data();
		size_t ps = praw.size();
		std::align(SSLM_ABI_ALIGNMENT_BYTES, req, pa, ps);
		sslm_kv_pool pool = nullptr;
		if (sslm_kv_pool_create(model, pa, req, 1, &pool) != SSLM_OK) return 4;
		sslm_config cfg{};
		cfg.max_batch = 1;
		cfg.max_chunk_budget = static_cast<int32_t>(T);
		cfg.max_layer_budget = 1;
		const size_t wsb = sslm_workspace_size(model, &cfg);
		std::vector<uint8_t> wraw(wsb + SSLM_ABI_ALIGNMENT_BYTES);
		void* wa = wraw.data();
		size_t wsz = wraw.size();
		std::align(SSLM_ABI_ALIGNMENT_BYTES, wsb, wa, wsz);
		sslm_workspace ws = nullptr;
		if (sslm_workspace_create(model, &cfg, wa, wsb, &ws) != SSLM_OK) return 5;
		sslm_seq seq = nullptr;
		if (sslm_seq_create(model, &pool, &seq) != SSLM_OK) return 6;
		std::vector<int32_t> toks(T);
		for (size_t i = 0; i < T; ++i) toks[i] = Tok(i);
		Counts c0 = g;
		int32_t consumed = 0;
		if (sslm_prefill(model, seq, toks.data(), (int32_t)T, (int32_t)T, SSLM_SPAN_PROMPT, ws, &consumed) != SSLM_OK ||
		    consumed != (int32_t)T)
			return 7;
		Counts c1 = g;
		sslm_decode_params params{};
		if (sslm_decode_params_init(model, SSLM_DECODE_MODE_GREEDY, 1, &params) != SSLM_OK) return 8;
		for (size_t i = 0; i < D; ++i) {
			int32_t out = 0;
			if (sslm_decode_step(model, &seq, 1, &params, ws, &out) != SSLM_OK || out < 0) return 9;
		}
		Counts c2 = g;
		Print("abi prefill", c0, c1);
		Print("abi decode", c1, c2);
		return 0;
	}
	SslmModelView model;
	std::string error;
	if (SslmModel::Load(bytes.data(), bytes.size(), model, &error) != SslmModelStatus::Ok) return 3;
	const uint32_t L = model.config.num_hidden_layers;
	const size_t hidden = model.config.hidden_size, head_dim = model.config.head_dim,
	             kv = model.config.num_key_value_heads, H = model.config.num_attention_heads;
	const int64_t cap = model.config.context_cap;
	std::printf("L=%u H=%zu KV=%zu head_dim=%zu hidden=%zu inter=%u cap=%lld\n", L, H, kv, head_dim, hidden,
	            model.config.intermediate_size, (long long)cap);
	PreflightScanWscFolds(model);
	std::vector<LayerBacking> backing(L);
	std::vector<LayerWeights> layers(L);
	for (uint32_t l = 0; l < L; ++l)
		if (!MarshalLayer(model, l, (uint32_t)H, (uint32_t)kv, backing[l], layers[l], &error)) return 4;
	const int8_t* embed = reinterpret_cast<const int8_t*>(model.weights.Tensor("embed")->data);
	bool ok = true;
	const CarriedScale embed_scale = ReadCarriedScale(model.composition_constants, "embed", &ok);
	std::vector<uint8_t> workspace(size_t(L) * size_t(cap) * kv * head_dim * 2);
	SslmSetTraceHook(model.trace_hook, Hook, nullptr);
	const OptionGKLandingMode k_mode =
	    model.option_g_fused_k_landing ? OptionGKLandingMode::kFused : OptionGKLandingMode::kLegacy;

	Counts c0 = g;
	std::vector<int8_t> chunk(T * hidden);
	std::vector<CarriedScale> scales(T);
	for (size_t i = 0; i < T; ++i)
		if (EmbedEntry(Tok(i), (int32_t)model.config.vocab_size, embed, hidden, embed_scale, chunk.data() + i * hidden,
		               &scales[i], "embed", i, &model.trace_hook) != SslmForwardStatus::Ok)
			return 5;
	SequenceLayerState seq;
	std::vector<int8_t> hc(hidden);
	seq.hidden_codes = hc.data();
	SslmForwardStatus st = RunLayerLoopChunkBatched(
	    chunk.data(), scales.data(), T, layers.data(), L, hidden, head_dim, kv, model.config.intermediate_size, cap, 0,
	    model.rope_tables, workspace.data(), workspace.size(), model.option_g_fused_k_landing, &seq.kv_saturation_count,
	    {}, &model.trace_hook, H * head_dim);
	if (st != SslmForwardStatus::Ok) return std::fprintf(stderr, "chunk: %s\n", SslmForwardStatusName(st)), 6;
	seq.context_length = (int64_t)T;
	Counts c1 = g;
	for (size_t i = 0; i < D; ++i) {
		CarriedScale sc{};
		if (EmbedEntry(Tok(T + i), (int32_t)model.config.vocab_size, embed, hidden, embed_scale, hc.data(), &sc, "embed",
		               T + i, &model.trace_hook) != SslmForwardStatus::Ok)
			return 7;
		seq.hidden_scale = sc;
		seq.layer_index = 0;
		st = RunLayerLoop(seq, layers.data(), L, L, hidden, head_dim, kv, model.config.intermediate_size, cap,
		                  model.rope_tables, workspace.data(), workspace.size(), k_mode, {}, T + i, &model.trace_hook,
		                  H * head_dim);
		if (st != SslmForwardStatus::Ok) return std::fprintf(stderr, "decode: %s\n", SslmForwardStatusName(st)), 8;
	}
	Counts c2 = g;
	std::printf("context_length after decode = %lld\n", (long long)seq.context_length);
	Print("direct prefill", c0, c1);
	Print("direct decode", c1, c2);
	PrintC();
	return 0;
}

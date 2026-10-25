// Multi-row GPU plan (rev 1.3.2), slice MR2: shared helpers for the MR2 red suite's executables (mr2_cells.cpp,
// mr2_fixture_probe.cpp). Public verbs, the bench bridge's live-state accessor (for counters only), and the MR2
// entry/seams of mr2_entry.h. Header-only.
#pragma once
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "superslm/forward_sites.h"
#include "superslm/gpu_1p0.h"
#include "superslm/gpu_1p0_bench_bridge.h"
#include "superslm/gpu_port.h"
#include "superslm/model.h"

namespace mr2 {

using S = ::SslmGpuStatus;

inline const char* Name(S s) {
	static char buf[16];
	std::snprintf(buf, sizeof buf, "%d", static_cast<int>(s));
	return buf;
}

inline bool ReadFileBytes(const std::string& path, std::vector<uint8_t>* out) {
	std::FILE* f = std::fopen(path.c_str(), "rb");
	if (!f) return false;
	std::fseek(f, 0, SEEK_END);
	const long sz = std::ftell(f);
	std::fseek(f, 0, SEEK_SET);
	out->resize(sz > 0 ? static_cast<size_t>(sz) : 0);
	const size_t n = sz > 0 ? std::fread(out->data(), 1, static_cast<size_t>(sz), f) : 0;
	std::fclose(f);
	return n == out->size();
}

inline uint64_t Fnv(const std::vector<uint8_t>& b) {
	uint64_t h = 1469598103934665603ull;
	for (uint8_t c : b) h = (h ^ c) * 1099511628211ull;
	return h;
}

// The suite's token streams: deterministic, inside the vocabulary, different per `stream`.
inline int32_t StreamToken(int i, int32_t vocab, uint32_t stream = 0) {
	return static_cast<int32_t>((1103515245u * static_cast<uint32_t>(i + 7) + 12345u + 7919u * stream) %
	                            static_cast<uint32_t>(vocab));
}

struct Model {
	std::string path;
	std::vector<uint8_t> bytes;
	superslm::SslmModelView view{};
	bool qk = false;
	bool Load(const std::string& p) {
		path = p;
		std::string err;
		if (!ReadFileBytes(p, &bytes) ||
		    superslm::SslmModel::Load(bytes.data(), bytes.size(), view, &err) != superslm::SslmModelStatus::Ok) {
			std::printf("FAIL cannot load model '%s' %s\n", p.c_str(), err.c_str());
			return false;
		}
		for (uint32_t l = 0; l < view.config.num_hidden_layers; ++l)
			if (view.weights.Tensor("layer" + std::to_string(l) + ".q_norm.gain") != nullptr) qk = true;
		return true;
	}
	uint32_t dpl() const { return superslm_gpu::DispatchesPerLayer(qk); }
	uint32_t layers() const { return view.config.num_hidden_layers; }
	uint32_t full_budget() const { return dpl() * layers(); }
	int32_t vocab() const { return static_cast<int32_t>(view.config.vocab_size); }
	uint32_t nqh() const { return view.config.num_attention_heads; }
	std::string base() const {
		std::string n = path.substr(path.find_last_of("/\\") + 1);
		return n.substr(0, n.find('.'));
	}
};

// One context, one mapped model, and the handles a case makes; everything is released in reverse order.
struct World {
	SslmGpuContext* ctx = nullptr;
	SslmGpuModelHandle* model = nullptr;
	std::vector<SslmGpuKvPoolHandle*> pools;
	std::vector<SslmGpuSequenceHandle*> seqs;
	std::vector<SslmGpuAdapterHandle*> adapters;
	bool ok = false;
	explicit World(const Model& m, const char* shader_dir = nullptr) {
		GpuContextConfig cfg{};
		cfg.shader_dir = shader_dir;
		S st = sslm_gpu_context_create(cfg, &ctx);
		if (st != S::SSLM_OK) { std::printf("FAIL context create %s\n", Name(st)); return; }
		if ((st = sslm_gpu_model_map(ctx, &m.view, GpuResidencyConfig{}, &model)) != S::SSLM_OK) {
			std::printf("FAIL model map %s\n", Name(st));
			return;
		}
		ok = true;
	}
	World(const World&) = delete;
	~World() {
		for (auto it = seqs.rbegin(); it != seqs.rend(); ++it)
			if (*it) sslm_gpu_seq_release(ctx, *it);
		for (auto* a : adapters) sslm_gpu_adapter_unmap(ctx, a);
		for (auto* p : pools) sslm_gpu_kv_pool_destroy(ctx, p);
		if (model) sslm_gpu_model_unmap(ctx, model);
		if (ctx) sslm_gpu_context_destroy(ctx);
	}
	SslmGpuSequenceHandle* Dedicated(int32_t cap) {
		SslmGpuSequenceHandle* s = nullptr;
		const S st = sslm_gpu_seq_create(ctx, model, cap, &s);
		if (st != S::SSLM_OK) { std::printf("FAIL seq create %s\n", Name(st)); return nullptr; }
		seqs.push_back(s);
		return s;
	}
	SslmGpuKvPoolHandle* Pool(uint32_t pages) {
		SslmGpuKvPoolHandle* p = nullptr;
		const S st = sslm_gpu_kv_pool_create(ctx, model, pages, &p);
		if (st != S::SSLM_OK) { std::printf("FAIL pool create %s\n", Name(st)); return nullptr; }
		pools.push_back(p);
		return p;
	}
	SslmGpuSequenceHandle* Pooled(SslmGpuKvPoolHandle* p, int32_t budget) {
		SslmGpuSequenceHandle* s = nullptr;
		const S st = sslm_gpu_seq_create_budgeted(ctx, model, p, budget, &s);
		if (st != S::SSLM_OK) { std::printf("FAIL budgeted create %s\n", Name(st)); return nullptr; }
		seqs.push_back(s);
		return s;
	}
	void Release(SslmGpuSequenceHandle* s) {
		for (auto& x : seqs)
			if (x == s) {
				sslm_gpu_seq_release(ctx, s);
				x = nullptr;
			}
	}
};

// Drains `s` and returns the slice's status (sslm_gpu_ready's out-status), or the ready call's own failure.
inline S Ready(World& w, SslmGpuSequenceHandle* s) {
	int32_t ready = 0;
	S out = S::SSLM_OK;
	const S st = sslm_gpu_ready(w.ctx, s, 1, &ready, &out);
	if (st != S::SSLM_OK) return st;
	if (!ready) return S::SSLM_BUSY;
	return out;
}
// One sslm_decode_step_gpu of `budget` dispatches, drained.
inline S Slice(World& w, SslmGpuSequenceHandle* s, uint32_t budget, SslmGpuAdapterHandle* adapter = nullptr) {
	const S st = sslm_decode_step_gpu(w.ctx, s, adapter, budget);
	if (st != S::SSLM_OK) return st;
	return Ready(w, s);
}
// The one-row twin's token: embed, then slices of `budget` dispatches each until the row is at full depth
// (layer_index == L). Returns the first non-OK status (a refusal's SSLM_SEQUENCE_REJECTED included).
inline S FeedRow(World& w, SslmGpuSequenceHandle* s, int32_t tok, uint32_t budget, uint32_t layers,
                 SslmGpuAdapterHandle* adapter = nullptr) {
	S st = sslm_gpu_seq_embed_token(w.ctx, s, tok);
	if (st != S::SSLM_OK) return st;
	for (uint32_t guard = 0; guard <= layers; ++guard) {
		st = Slice(w, s, budget, adapter);
		if (st != S::SSLM_OK) return st;
		if (*SslmGpuSeqHandleLayerIndexForBench(s) >= layers) return S::SSLM_OK;
	}
	return S::SSLM_DEVICE_LOST;  // never reached full depth: a broken layer seam
}

inline bool Save(World& w, SslmGpuSequenceHandle* s, std::vector<uint8_t>* b, S* status = nullptr) {
	size_t need = 0;
	sslm_gpu_seq_save(w.ctx, s, nullptr, &need);
	b->assign(need ? need : 1, 0);
	size_t sz = b->size();
	const S st = sslm_gpu_seq_save(w.ctx, s, b->data(), &sz);
	if (status) *status = st;
	if (st != S::SSLM_OK) return false;
	b->resize(sz);
	return true;
}

// The five counters plan E27 names, read from the handle's live state.
struct Counters {
	uint64_t v[5] = {};
	Counters operator-(const Counters& o) const {
		Counters r;
		for (int i = 0; i < 5; ++i) r.v[i] = v[i] - o.v[i];
		return r;
	}
	bool operator==(const Counters& o) const { return std::memcmp(v, o.v, sizeof v) == 0; }
	std::string Str() const {
		char b[160];
		std::snprintf(b, sizeof b, "kv_sat %llu kv_landing %llu k_channel %llu rope_q %llu rope_k %llu",
		              (unsigned long long)v[0], (unsigned long long)v[1], (unsigned long long)v[2],
		              (unsigned long long)v[3], (unsigned long long)v[4]);
		return b;
	}
};
inline Counters ReadCounters(SslmGpuSequenceHandle* s) {
	const superslm::SequenceLayerState* ls = SslmGpuSeqHandleLiveStateForBench(s);
	Counters c;
	c.v[0] = ls->kv_saturation_count;
	c.v[1] = ls->kv_landing_saturation_count;
	c.v[2] = ls->k_channel_landing_saturation_count;
	c.v[3] = ls->rope_q_saturation_count;
	c.v[4] = ls->rope_k_saturation_count;
	return c;
}

// First differing byte, or -1.
inline long FirstDiff(const std::vector<uint8_t>& a, const std::vector<uint8_t>& b) {
	const size_t n = a.size() < b.size() ? a.size() : b.size();
	for (size_t i = 0; i < n; ++i)
		if (a[i] != b[i]) return static_cast<long>(i);
	return a.size() == b.size() ? -1 : static_cast<long>(n);
}

}  // namespace mr2

// Paged-KV plan (rev 16.2), step G1, group g1b: the shared harness of this group's GPU red cells.
//
// Cells (plan §7, with the owning step of each part): 2.6 [G3 pool; G5 blob], 2.7 [G4; G5 re-save],
// 2.11 [G3 create; G4 prefix; G5 restore-into-pool], 5.7 [G3], 5.8 [G4], 5.9 [G5], 9.13's A15 row
// [G4] and 11.4 [G3, with its ceiling create, bisection and fault-seam legs]. One translation unit
// per owning step (g1b_g3_cells.cpp, g1b_g3_pool_ceiling.cpp, g1b_g4_cells.cpp, g1b_g5_cells.cpp),
// so a per-step executable links exactly the verbs its step defines, as tests/paged-kv's per-step
// executables do.
//
// HOW THESE CELLS ARE RED NOW. Every verb of the page-pool block of include/superslm/gpu_1p0.h is
// declared and has no definition (the G1 header layer, commit 63e7b0e). Each cell therefore
// COMPILES and fails to LINK (LNK2019 on the verbs it calls) until its step defines them; that is
// t2791's documented convention. Two test seams the plan requires and does not name are declared
// below under SUPERSLM_GPU_ALLOC_FAULT_INJECTION with the names this group chose (11.4's R12-G3
// legs): they are link-red too, until G3's builder defines them (or renames them here).
//
// Grading (plan §7, "Three rules", class C). A verdict reads only: GPU statuses counted against a
// pool size this file computes from the §3.4 formula; tokens, compared with an unsaved twin; raw
// bytes read through sslm_gpu_test_only_peek_pool_mirror_bytes; blob bytes; the status of a public
// verb called afterwards. sslm_gpu_kv_pool_stats is never read. Every page-count verdict is the
// GPU fill probe of §8 (GpuFillProbe below), taken after sslm_gpu_ready has drained every
// sequence of the scene.
//
// Fixtures (plan §8; tests/paged-kv's generator, tools/gen_paged_kv_fixture.py):
//   --tiny=PATH   pkv_def.sslm (cap 4096, kv_block_size 16, so B = 16; 2 layers, 2 KV heads of
//                 48, int8: 384 B per token, page_bytes 6,144; the S8 schema). Every cell but
//                 11.4's ceiling legs. "Another model" is a second mapping of the same bytes on
//                 the same context (a distinct model handle with the same geometry: the strictest
//                 form, since only identity differs), as tests/paged-kv/c5_dim9_persistence.cpp's
//                 OtherModel does. "Another context" is a second context with its own mapping.
//   --m05=PATH    the Qwen2.5-0.5B artifact (11.4: page_bytes 98,304, ceiling 43,690)
//   --m15=PATH    the Qwen2.5-1.5B artifact (11.4: page_bytes 229,376, ceiling 18,724)
//   --shaders=DIR optional absolute shader directory (GpuContextConfig::shader_dir)
// A missing artifact FAILS the cell that needs it (tests/paged-kv's rule): nothing skips.
#ifndef SSLM_G1B_HELPERS_H
#define SSLM_G1B_HELPERS_H

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include "pool_cap_helpers.h"
#include <windows.h>
#include <fcntl.h>
#include <io.h>
#include <sys/stat.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <string>
#include <vector>

#include "pkvg_slm6_layout.h"
#include "superslm/gpu_1p0.h"
#include "superslm/gpu_1p0_bench_bridge.h"
#include "superslm/gpu_1p0_g5_bridge.h"
#include "superslm/gpu_port.h"  // superslm_gpu::DispatchesPerLayer; the T-2851 fault seams
#include "superslm/model.h"
#include "superslm/schema_masks.h"

// ---------------------------------------------------------------------------------------------
// Test seams 11.4 needs that no step has declared yet (plan §7 11.4 and the carrier's R12-G3 row).
// Names are this group's; G3's builder defines them in gpu_1p0.cpp beside the T-2851 seams, or
// renames them here. Until then every 11.4 seam leg is red by link on them.
// ---------------------------------------------------------------------------------------------
#if defined(SUPERSLM_GPU_ALLOC_FAULT_INJECTION)
// R12-G3 leg (2), "with the seam recording each call's size": the k-th Device::TryMakeBuffer call
// since the last SslmGpuAllocCounterResetForTest (1-based): its requested byte size and its
// D3D12_HEAP_TYPE. False when fewer than k calls were counted.
bool SslmGpuAllocRecordForTest(uint32_t k, uint64_t* out_bytes, uint32_t* out_heap_type) noexcept;
// R12-G3 leg (3), "with close_failures scoped to the zero-fill": the next `consecutive_failures`
// Close attempts made inside the pool create's ZeroFillResidentUavBufferSync fail without closing
// the list (ArmGpuDeviceLogitsCloseFaultForTest's shape, scoped to the zero-fill). 0 disarms.
void ArmGpuPoolZeroFillCloseFaultForTest(uint32_t consecutive_failures) noexcept;
#endif

namespace g1b {

using S = ::SslmGpuStatus;

// ---- registry and checks (tests/paged-kv/pkv_common.h's shape) -------------------------------

struct CellEntry {
	const char* id;    // "<§7 id>/<step>", e.g. "2.11/G4"
	const char* step;  // G3, G4 or G5
	void (*fn)();
};
inline std::vector<CellEntry>& Registry() {
	static std::vector<CellEntry> r;
	return r;
}
struct Registrar {
	Registrar(const char* id, const char* step, void (*fn)()) { Registry().push_back({id, step, fn}); }
};
#define G1B_CAT2(a, b) a##b
#define G1B_CAT(a, b) G1B_CAT2(a, b)
#define G1B_CELL(id, step, fn) static ::g1b::Registrar G1B_CAT(g1b_registrar_, __LINE__)(id, step, fn)

struct RunState {
	const char* cell = "(none)";
	int checks = 0, failures = 0, cell_failures = 0;
};
inline RunState& State() {
	static RunState s;
	return s;
}
inline void Fail(const char* file, int line, const char* what) {
	++State().failures;
	++State().cell_failures;
	std::printf("FAIL [%s] %s:%d: %s\n", State().cell, file, line, what);
}
#define G1B_CHECK(cond)                                                     \
	do {                                                                    \
		++::g1b::State().checks;                                            \
		if (!(cond)) ::g1b::Fail(__FILE__, __LINE__, #cond);                \
	} while (0)
#define G1B_CHECK_MSG(cond, ...)                                            \
	do {                                                                    \
		++::g1b::State().checks;                                            \
		if (!(cond)) {                                                      \
			char g1b_buf_[512];                                             \
			std::snprintf(g1b_buf_, sizeof(g1b_buf_), __VA_ARGS__);         \
			::g1b::Fail(__FILE__, __LINE__, g1b_buf_);                      \
		}                                                                   \
	} while (0)

inline const char* StatusName(S s) {
	static const char* const kNames[] = {
	    "SSLM_OK", "SSLM_DISPATCH_BUDGET_TOO_SMALL", "SSLM_BUSY", "SSLM_CONTEXT_HAS_LIVE_HANDLES",
	    "SSLM_MODEL_HAS_LIVE_SEQUENCES", "SSLM_ADAPTER_MODEL_MISMATCH", "SSLM_ADAPTER_BASE_HASH_MISMATCH",
	    "SSLM_SEQUENCE_KV_BUFFER_MISMATCH", "SSLM_DEVICE_LOST", "SSLM_BATCH_BUDGET_EXHAUSTED",
	    "SSLM_TOKEN_ID_OUT_OF_RANGE", "SSLM_SEQUENCE_REJECTED", "SSLM_RESTORE_MODEL_MISMATCH",
	    "SSLM_MODEL_HAS_LIVE_ADAPTERS", "SSLM_ADAPTER_HAS_BOUND_SEQUENCES", "SSLM_GPU_SHADER_BINARY_STALE",
	    "SSLM_GPU_ALLOCATION_FAILED", "SSLM_OUTPUT_BUFFER_TOO_SMALL", "SSLM_PREFILL_HIDDEN_UNAVAILABLE",
	    "SSLM_GPU_SHADER_DIR_INVALID", "SSLM_GPU_SHADER_DIR_CONFLICT", "SSLM_GPU_PARALLEL_FOR_INVALID",
	    "SSLM_GPU_PARALLEL_FOR_INCOMPLETE", "SSLM_GPU_RESIDENCY_FLAGS_INVALID", "SSLM_GPU_KV_BUDGET_EXCEEDED",
	    "SSLM_GPU_KV_POOL_EXHAUSTED", "SSLM_GPU_KV_POOL_TOO_LARGE", "SSLM_GPU_HOLDER_MODE_MISMATCH",
	    "SSLM_GPU_PREFIX_REJECTED", "SSLM_GPU_PREFIX_SCHEMA_MISMATCH", "SSLM_RESTORE_PRODUCER_MISMATCH",
	    "SSLM_GPU_FORCED_TABLE_ABSENT", "SSLM_GPU_CONTEXT_CAP_EXCEEDED"};
	const uint32_t i = static_cast<uint32_t>(s);
	return i < sizeof(kNames) / sizeof(kNames[0]) ? kNames[i] : "UNKNOWN_STATUS";
}
#define G1B_CHECK_STATUS(got, want, ...)                                                       \
	do {                                                                                       \
		const ::g1b::S g1b_got_ = (got), g1b_want_ = (want);                                   \
		++::g1b::State().checks;                                                               \
		if (g1b_got_ != g1b_want_) {                                                           \
			char g1b_ctx_[256];                                                                \
			std::snprintf(g1b_ctx_, sizeof(g1b_ctx_), __VA_ARGS__);                            \
			char g1b_buf_[512];                                                                \
			std::snprintf(g1b_buf_, sizeof(g1b_buf_), "%s: got %s, want %s", g1b_ctx_,         \
			              ::g1b::StatusName(g1b_got_), ::g1b::StatusName(g1b_want_));          \
			::g1b::Fail(__FILE__, __LINE__, g1b_buf_);                                         \
		}                                                                                      \
	} while (0)

// ---- arguments --------------------------------------------------------------------------------

struct Args {
	std::string tiny, m05, m15, shaders;
};
inline Args& GArgs() {
	static Args a;
	return a;
}
// Consumes the flags this harness knows and returns every other argument (cell selectors).
inline std::vector<std::string> ParseArgs(int argc, char** argv) {
	std::vector<std::string> rest;
	for (int i = 1; i < argc; ++i) {
		const std::string a = argv[i];
		auto take = [&](const char* flag, std::string* dst) {
			const size_t n = std::strlen(flag);
			if (a.compare(0, n, flag) != 0) return false;
			*dst = a.substr(n);
			return true;
		};
		if (take("--tiny=", &GArgs().tiny) || take("--m05=", &GArgs().m05) || take("--m15=", &GArgs().m15) ||
		    take("--shaders=", &GArgs().shaders)) {
			continue;
		}
		rest.push_back(a);
	}
	return rest;
}

// ---- page geometry, written here from the plan (§3.1, §3.4), never read from the engine -----------

struct Geometry {
	uint32_t layers = 0, kv_heads = 0, head_dim = 0, elem = 0;
	int64_t cap = 0, B = 0;
	uint64_t page_bytes = 0;  // L * 2 * H_kv * B * D * elem (§3.1)
	uint32_t cap_pages = 0;   // ceil(cap / B)
	int32_t vocab = 0;
	uint32_t dispatches_per_layer = 0;
	uint32_t decode_budget = 0;  // enough dispatches for one token at full depth

	// R(budget) = min(ceil(budget / B) + 1, ceil(cap / B)) (§3.4).
	uint32_t R(int64_t budget) const {
		const int64_t a = (budget + B - 1) / B + 1;
		return static_cast<uint32_t>(std::min<int64_t>(a, cap_pages));
	}
	// The pages a frozen prefix of `len` positions keeps: ceil(len / B) (§3.4, freeze).
	uint32_t Frozen(int64_t len) const { return static_cast<uint32_t>((len + B - 1) / B); }
	// E = min(floor(origin / B), ceil(cap / B) - R(budget)) (§3.4, restore).
	uint32_t E(int64_t origin, int64_t budget) const {
		return static_cast<uint32_t>(std::min<int64_t>(origin / B, static_cast<int64_t>(cap_pages) - R(budget)));
	}
	// The largest page count whose byte size is at most the adapter's pool cap (§4 item 3; 11.4;
	// pool_cap_helpers.h).
	uint32_t CeilingPages() const { return static_cast<uint32_t>(pool_cap::TestPoolCapBytes() / page_bytes); }
};

inline Geometry GeometryOf(const superslm::SslmModelView& v) {
	Geometry g;
	const auto& c = v.config;
	g.layers = c.num_hidden_layers;
	g.kv_heads = c.num_key_value_heads;
	g.head_dim = c.head_dim;
	g.cap = c.context_cap;
	g.elem = c.kv_precision == superslm::SslmKvPrecision::Int16 ? 2u : 1u;
	// B = kv_block_size when kv_block_size <= cap and cap % kv_block_size == 0; otherwise B = cap.
	g.B = (c.kv_block_size != 0 && c.kv_block_size <= c.context_cap && c.context_cap % c.kv_block_size == 0)
	          ? static_cast<int64_t>(c.kv_block_size)
	          : g.cap;
	g.page_bytes = uint64_t{g.layers} * 2u * g.kv_heads * static_cast<uint64_t>(g.B) * g.head_dim * g.elem;
	g.cap_pages = static_cast<uint32_t>((g.cap + g.B - 1) / g.B);
	g.vocab = static_cast<int32_t>(c.vocab_size);
	bool qk = false;
	for (uint32_t l = 0; l < g.layers; ++l) {
		if (v.weights.Tensor("layer" + std::to_string(l) + ".q_norm.gain") != nullptr) qk = true;
	}
	g.dispatches_per_layer = superslm_gpu::DispatchesPerLayer(qk);
	g.decode_budget = g.dispatches_per_layer * g.layers;
	return g;
}

// ---- artifacts and contexts -------------------------------------------------------------------

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

struct Artifact {
	std::string path;
	std::vector<uint8_t> bytes;
	superslm::SslmModelView view{};
	Geometry geo;
	bool ok = false;
	bool Load(const std::string& p) {
		path = p;
		std::string err;
		if (p.empty() || !ReadFileBytes(p, &bytes)) {
			std::printf("g1b: cannot read artifact '%s'\n", p.c_str());
			return false;
		}
		if (superslm::SslmModel::Load(bytes.data(), bytes.size(), view, &err) != superslm::SslmModelStatus::Ok) {
			std::printf("g1b: load %s failed: %s\n", p.c_str(), err.c_str());
			return false;
		}
		geo = GeometryOf(view);
		ok = true;
		return true;
	}
};

inline SslmGpuContext* NewContext() {
	GpuContextConfig cfg{};
	cfg.shader_dir = GArgs().shaders.empty() ? nullptr : GArgs().shaders.c_str();
	SslmGpuContext* ctx = nullptr;
	const S st = sslm_gpu_context_create(cfg, &ctx);
	if (st != S::SSLM_OK || !ctx) {
		std::printf("g1b: sslm_gpu_context_create -> %s\n", StatusName(st));
		return nullptr;
	}
	return ctx;
}
inline SslmGpuModelHandle* MapModel(SslmGpuContext* ctx, const Artifact& a) {
	SslmGpuModelHandle* m = nullptr;
	const S st = sslm_gpu_model_map(ctx, &a.view, GpuResidencyConfig{}, &m);
	if (st != S::SSLM_OK || !m) {
		std::printf("g1b: sslm_gpu_model_map(%s) -> %s\n", a.path.c_str(), StatusName(st));
		return nullptr;
	}
	return m;
}

// The tiny-fixture environment every cell but 11.4's ceiling legs runs in. ctx1 carries m1 (the
// model under test) and m1b (a second mapping of the same bytes: "another model"); ctx2 carries m2
// (the same bytes on another context: "another context"). Built once per process; contexts are
// never destroyed while a cell runs.
struct Env {
	bool ok = false;
	Artifact tiny;
	Geometry geo;
	SslmGpuContext* ctx1 = nullptr;
	SslmGpuContext* ctx2 = nullptr;
	SslmGpuModelHandle* m1 = nullptr;
	SslmGpuModelHandle* m1b = nullptr;
	SslmGpuModelHandle* m2 = nullptr;
};
inline Env& GetEnv() {
	static Env e;
	static bool tried = false;
	if (!tried) {
		tried = true;
		if (GArgs().tiny.empty()) {
			std::printf("g1b: --tiny=<pkv_def.sslm> is required (tools/gen_paged_kv_fixture.py OUT_DIR)\n");
		} else if (e.tiny.Load(GArgs().tiny)) {
			e.geo = e.tiny.geo;
			e.ctx1 = NewContext();
			e.ctx2 = NewContext();
			if (e.ctx1) e.m1 = MapModel(e.ctx1, e.tiny);
			if (e.ctx1) e.m1b = MapModel(e.ctx1, e.tiny);
			if (e.ctx2) e.m2 = MapModel(e.ctx2, e.tiny);
			e.ok = e.ctx1 && e.ctx2 && e.m1 && e.m1b && e.m2;
			std::printf("g1b: tiny %s cap=%lld B=%lld page_bytes=%llu cap_pages=%u vocab=%d dispatches/layer=%u\n",
			            e.tiny.path.c_str(), static_cast<long long>(e.geo.cap), static_cast<long long>(e.geo.B),
			            static_cast<unsigned long long>(e.geo.page_bytes), e.geo.cap_pages, e.geo.vocab,
			            e.geo.dispatches_per_layer);
		}
	}
	G1B_CHECK_MSG(e.ok, "the tiny-fixture environment did not come up (see the g1b: lines above)");
	return e;
}

// ---- a scene: every handle a construction makes, released in reverse order ---------------------

struct Scene {
	struct Seq { SslmGpuContext* ctx; SslmGpuSequenceHandle* h; };
	// A prefix's release is called through the pointer stored when it was added (AddPrefix), so a
	// [G3] translation unit that makes no prefix never references the [G4] verb.
	using PrefixRelease = S (*)(SslmGpuContext*, SslmGpuPrefixHandle*) noexcept;
	struct Pfx { SslmGpuContext* ctx; SslmGpuPrefixHandle* h; PrefixRelease release; };
	struct Pool { SslmGpuContext* ctx; SslmGpuKvPoolHandle* h; };
	std::vector<Seq> seqs;
	std::vector<Pfx> prefixes;
	std::vector<Pool> pools;

	void Forget(SslmGpuSequenceHandle* h) {
		seqs.erase(std::remove_if(seqs.begin(), seqs.end(), [&](const Seq& s) { return s.h == h; }), seqs.end());
	}
	void Forget(SslmGpuPrefixHandle* h) {
		prefixes.erase(std::remove_if(prefixes.begin(), prefixes.end(), [&](const Pfx& p) { return p.h == h; }),
		               prefixes.end());
	}
	void Release(SslmGpuSequenceHandle* h) {
		for (const Seq& s : seqs) {
			if (s.h == h) {
				DrainOne(s.ctx, h);
				sslm_gpu_seq_release(s.ctx, h);
			}
		}
		Forget(h);
	}
	void AddPrefix(SslmGpuContext* ctx, SslmGpuPrefixHandle* h) { prefixes.push_back({ctx, h, &sslm_gpu_prefix_release}); }
	void Release(SslmGpuPrefixHandle* h) {
		for (const Pfx& p : prefixes) {
			if (p.h == h) p.release(p.ctx, h);
		}
		Forget(h);
	}
	static void DrainOne(SslmGpuContext* ctx, SslmGpuSequenceHandle* h) {
		int32_t ready = 0;
		S out = S::SSLM_OK;
		for (int i = 0; i < 1000000 && !ready; ++i) {
			if (sslm_gpu_ready(ctx, h, 1, &ready, &out) != S::SSLM_OK) break;
		}
	}
	// "A GPU fill-probe reading is taken after sslm_gpu_ready has drained every in-flight sequence"
	// (§7, grading rule). Note: sslm_gpu_ready clears a sequence's bind eligibility, so a cell that
	// grades SslmGpuSeqSetSchemaForG5Bridge calls it before draining.
	void DrainAll() {
		for (const Seq& s : seqs) DrainOne(s.ctx, s.h);
	}
	~Scene() {
		for (auto it = seqs.rbegin(); it != seqs.rend(); ++it) {
			DrainOne(it->ctx, it->h);
			sslm_gpu_seq_release(it->ctx, it->h);
		}
		for (auto it = prefixes.rbegin(); it != prefixes.rend(); ++it) it->release(it->ctx, it->h);
		for (auto it = pools.rbegin(); it != pools.rend(); ++it) sslm_gpu_kv_pool_destroy(it->ctx, it->h);
	}
};

// ---- verb wrappers -------------------------------------------------------------------------------

inline SslmGpuKvPoolHandle* NewPool(Scene& sc, SslmGpuContext* ctx, SslmGpuModelHandle* model, uint32_t pages) {
	SslmGpuKvPoolHandle* p = nullptr;
	const S st = sslm_gpu_kv_pool_create(ctx, model, pages, &p);
	G1B_CHECK_STATUS(st, S::SSLM_OK, "sslm_gpu_kv_pool_create(%u pages)", pages);
	if (st == S::SSLM_OK && p) sc.pools.push_back({ctx, p});
	return st == S::SSLM_OK ? p : nullptr;
}
inline SslmGpuSequenceHandle* NewBudgetSeq(Scene& sc, SslmGpuContext* ctx, SslmGpuModelHandle* model,
                                           SslmGpuKvPoolHandle* pool, int32_t budget) {
	SslmGpuSequenceHandle* s = nullptr;
	const S st = sslm_gpu_seq_create_budgeted(ctx, model, pool, budget, &s);
	G1B_CHECK_STATUS(st, S::SSLM_OK, "sslm_gpu_seq_create_budgeted(budget %d)", budget);
	if (st == S::SSLM_OK && s) sc.seqs.push_back({ctx, s});
	return st == S::SSLM_OK ? s : nullptr;
}
inline SslmGpuSequenceHandle* NewDedicatedSeq(Scene& sc, SslmGpuContext* ctx, SslmGpuModelHandle* model, int64_t cap) {
	SslmGpuSequenceHandle* s = nullptr;
	const S st = sslm_gpu_seq_create(ctx, model, cap, &s);
	G1B_CHECK_STATUS(st, S::SSLM_OK, "sslm_gpu_seq_create(cap %lld)", static_cast<long long>(cap));
	if (st == S::SSLM_OK && s) sc.seqs.push_back({ctx, s});
	return st == S::SSLM_OK ? s : nullptr;
}

// A deterministic token stream valid for the vocabulary (tests/paged-kv's Stream(seed, n, vocab)
// shape: distinct seeds give distinct streams).
inline std::vector<int32_t> Stream(int32_t seed, int32_t n, int32_t vocab) {
	std::vector<int32_t> t(static_cast<size_t>(n > 0 ? n : 0));
	uint32_t x = 2654435761u * static_cast<uint32_t>(seed + 1);
	for (int32_t i = 0; i < n; ++i) {
		x = x * 1664525u + 1013904223u;
		t[static_cast<size_t>(i)] = static_cast<int32_t>((x >> 8) % static_cast<uint32_t>(vocab - 1)) + 1;
	}
	return t;
}

// Every token of `t` through sslm_gpu_prefix_prefill in calls of at most `per_call` dispatches'
// worth; `*total` counts the admitted tokens (the bridges' partial consumption).
inline S PrefixPrefillAll(const Geometry& g, SslmGpuContext* ctx, SslmGpuPrefixHandle* p, const std::vector<int32_t>& t,
                          int32_t kind = 0, int32_t* total = nullptr) {
	int32_t done = 0;
	S st = S::SSLM_OK;
	const uint32_t whole = g.decode_budget * static_cast<uint32_t>(t.size() + 1);
	while (done < static_cast<int32_t>(t.size())) {
		int32_t consumed = 0;
		st = sslm_gpu_prefix_prefill(ctx, p, t.data() + done, static_cast<int32_t>(t.size()) - done, kind, whole,
		                             &consumed);
		if (consumed > 0) done += consumed;
		if (st != S::SSLM_OK || consumed <= 0) break;
	}
	if (total) *total = done;
	return st;
}
inline SslmGpuPrefixHandle* NewPrefix(Scene& sc, SslmGpuContext* ctx, SslmGpuModelHandle* model, SslmGpuKvPoolHandle* pool,
                                      int32_t budget) {
	SslmGpuPrefixHandle* p = nullptr;
	const S st = sslm_gpu_prefix_begin(ctx, model, pool, budget, &p);
	G1B_CHECK_STATUS(st, S::SSLM_OK, "sslm_gpu_prefix_begin(budget %d)", budget);
	if (st == S::SSLM_OK && p) sc.AddPrefix(ctx, p);
	return st == S::SSLM_OK ? p : nullptr;
}
// A prefix of exactly `t` (budget = its length), prefilled through the prompt kind and frozen.
inline SslmGpuPrefixHandle* FrozenPrefix(Scene& sc, const Geometry& g, SslmGpuContext* ctx, SslmGpuModelHandle* model,
                                         SslmGpuKvPoolHandle* pool, const std::vector<int32_t>& t) {
	SslmGpuPrefixHandle* p = NewPrefix(sc, ctx, model, pool, static_cast<int32_t>(t.size()));
	if (!p) return nullptr;
	int32_t n = 0;
	G1B_CHECK_STATUS(PrefixPrefillAll(g, ctx, p, t, 0, &n), S::SSLM_OK, "prefix prefill of %zu tokens", t.size());
	G1B_CHECK_STATUS(sslm_gpu_prefix_freeze(ctx, p), S::SSLM_OK, "sslm_gpu_prefix_freeze");
	return p;
}

inline S PrefillPrompt(const Geometry& g, SslmGpuContext* ctx, SslmGpuSequenceHandle* s, const std::vector<int32_t>& t) {
	return SslmGpuSeqPrefillPromptForG5Bridge(ctx, s, t.data(), static_cast<int32_t>(t.size()), g.decode_budget);
}

// One decode step through the recommended composition (embed-if-needed, layer loop to depth,
// finish). `*tok` is the produced token, or -1 when the step was refused.
inline S DecodeOne(const Geometry& g, SslmGpuContext* ctx, SslmGpuSequenceHandle* s, int32_t embed, int32_t* tok) {
	int32_t out = -1;
	const S st = SslmGpuSeqDecodeStepForG5Bridge(ctx, s, embed, g.decode_budget, &out);
	*tok = st == S::SSLM_OK ? out : -1;
	return st;
}
// Decodes until a step is refused (or `max` tokens), feeding each token back. Returns the number
// of tokens produced; `*refusal` is the refusing status (SSLM_OK if `max` was reached).
inline int64_t TokensUntilRefusal(const Geometry& g, SslmGpuContext* ctx, SslmGpuSequenceHandle* s, S* refusal,
                                  int64_t max = 8192, std::vector<int32_t>* tokens = nullptr) {
	int32_t last = 1;
	int64_t n = 0;
	*refusal = S::SSLM_OK;
	while (n < max) {
		int32_t tok = -1;
		const S st = DecodeOne(g, ctx, s, last, &tok);
		if (st != S::SSLM_OK) {
			*refusal = st;
			break;
		}
		if (tokens) tokens->push_back(tok);
		last = tok >= 0 ? tok : 1;
		++n;
	}
	return n;
}
inline std::vector<int32_t> DecodeN(const Geometry& g, SslmGpuContext* ctx, SslmGpuSequenceHandle* s, int64_t n,
                                    S* first_refusal = nullptr) {
	std::vector<int32_t> t;
	S r = S::SSLM_OK;
	TokensUntilRefusal(g, ctx, s, &r, n, &t);
	if (first_refusal) *first_refusal = r;
	return t;
}

// ---- the GPU mirror peek (§4 item 9) -----------------------------------------------------------

// The whole host mirror of a pool of `pages` pages, page by page through the peek seam.
inline std::vector<uint8_t> MirrorSnapshot(const Geometry& g, SslmGpuKvPoolHandle* pool, uint32_t pages) {
	std::vector<uint8_t> all(static_cast<size_t>(g.page_bytes) * pages, 0xEE);
	for (uint32_t p = 0; p < pages; ++p) {
		const S st = sslm_gpu_test_only_peek_pool_mirror_bytes(pool, p, 0, static_cast<size_t>(g.page_bytes),
		                                                       all.data() + static_cast<size_t>(g.page_bytes) * p);
		if (st != S::SSLM_OK) {
			G1B_CHECK_STATUS(st, S::SSLM_OK, "mirror peek of page %u", p);
			break;
		}
	}
	return all;
}

// ---- blobs ----------------------------------------------------------------------------------------

inline std::vector<uint8_t> SaveBlob(SslmGpuContext* ctx, const SslmGpuSequenceHandle* s, S* status = nullptr) {
	size_t need = 0;
	sslm_gpu_seq_save(ctx, s, nullptr, &need);
	std::vector<uint8_t> b(need);
	size_t sz = need;
	const S st = need ? sslm_gpu_seq_save(ctx, s, b.data(), &sz) : S::SSLM_SEQUENCE_KV_BUFFER_MISMATCH;
	if (status) *status = st;
	if (st != S::SSLM_OK) b.clear();
	else b.resize(sz);
	return b;
}

template <typename T>
inline T ReadAt(const std::vector<uint8_t>& b, size_t off) {
	T v{};
	if (off + sizeof(T) <= b.size()) std::memcpy(&v, b.data() + off, sizeof(T));
	return v;
}
template <typename T>
inline void WriteAt(std::vector<uint8_t>* b, size_t off, T v) {
	if (off + sizeof(T) <= b->size()) std::memcpy(b->data() + off, &v, sizeof(T));
}

// The 'SLM6' layout this suite edits (plan §4 item 10: "SLM5 fields, plus kv_mode, budget, origin
// and the canonical [0, L') K/V gather of §3.7"). The plan fixes the fields, not their offsets:
// these are 'SLM5''s header (superslm_gpu.cpp GpuSeqBlobHeader, 120 bytes) and its 12-byte tail,
// then the three new fields in §3.7's order, then the residual, then kv_positions (u64) and the
// rows. THIS IS A PIN AGAINST G5's WRITER: Slm6LayoutMatches checks it against a real save of a
// state the test knows, and a cell whose check fails is red with "SLM6 layout pin stale" instead
// of editing the wrong bytes. G5's builder re-anchors the offsets here, in one place.
namespace slm6 {
constexpr uint32_t kMagic = 0x364D4C53u;    // "SLM6" little-endian
constexpr uint32_t kMagicV5 = 0x354D4C53u;  // "SLM5"
constexpr uint32_t kMagicV4 = 0x344D4C53u;  // "SLM4"
constexpr size_t kOffMagic = 0;
constexpr size_t kOffLayerIndex = 4;
constexpr size_t kOffContextLength = 32;
constexpr size_t kOffHiddenCodesSize = 40;
// pkvg unification: the offsets are the suite's one SLM6 pin, pkvg_slm6_layout.h (shared with g1e).
constexpr size_t kV5HeaderBytes = pkvg_slm6::kSlm5HeaderBytes;
constexpr size_t kV5TailBytes = pkvg_slm6::kSlm5TailBytes;
constexpr size_t kOffKvMode = pkvg_slm6::kOffKvMode;    // 132, u32: 0 whole_reserve, 1 budget
constexpr size_t kOffBudget = pkvg_slm6::kOffBudget;    // 136, i32
constexpr size_t kOffOrigin = pkvg_slm6::kOffOrigin;    // 140, i64
constexpr size_t kFixedBytes = pkvg_slm6::kFixedBytes;  // 148
inline size_t OffKvPositions(const std::vector<uint8_t>& b) {
	return pkvg_slm6::OffKvPositions(ReadAt<uint64_t>(b, kOffHiddenCodesSize));
}
inline size_t RowBytes(const Geometry& g) { return static_cast<size_t>(g.layers) * 2u * g.kv_heads * g.head_dim * g.elem; }
}  // namespace slm6

// The pin check: `b` is the 'SLM6' save of a pooled sequence the test built with this budget,
// origin and context length, resting at layer 0. Every field is read at the pinned offset and the
// size is checked against §3.7's formula (fixed bytes, residual, kv_positions, L' rows).
inline bool Slm6LayoutMatches(const Geometry& g, const std::vector<uint8_t>& b, int32_t budget, int64_t origin,
                              int64_t context_length) {
	if (b.size() < slm6::kFixedBytes + 8) return false;
	const uint64_t kvpos = ReadAt<uint64_t>(b, slm6::OffKvPositions(b));
	const bool ok = ReadAt<uint32_t>(b, slm6::kOffMagic) == slm6::kMagic && ReadAt<uint32_t>(b, slm6::kOffKvMode) == 1u &&
	                ReadAt<int32_t>(b, slm6::kOffBudget) == budget && ReadAt<int64_t>(b, slm6::kOffOrigin) == origin &&
	                ReadAt<int64_t>(b, slm6::kOffContextLength) == context_length &&
	                ReadAt<uint32_t>(b, slm6::kOffLayerIndex) == 0u && kvpos == static_cast<uint64_t>(context_length) &&
	                b.size() == slm6::OffKvPositions(b) + pkvg_slm6::kKvPositionsBytes + static_cast<size_t>(kvpos) * slm6::RowBytes(g);
	if (!ok) {
		std::printf("g1b: SLM6 layout pin check: size=%zu magic=0x%08x kv_mode=%u budget=%d origin=%lld ctx=%lld kvpos=%llu\n",
		            b.size(), ReadAt<uint32_t>(b, slm6::kOffMagic), ReadAt<uint32_t>(b, slm6::kOffKvMode),
		            ReadAt<int32_t>(b, slm6::kOffBudget), static_cast<long long>(ReadAt<int64_t>(b, slm6::kOffOrigin)),
		            static_cast<long long>(ReadAt<int64_t>(b, slm6::kOffContextLength)),
		            static_cast<unsigned long long>(kvpos));
	}
	return ok;
}

// ---- the GPU fill probe (§8, two-sided; through sslm_gpu_seq_create_budgeted, G3) ----------------

// One pool a construction exposes for probing, with the context and model whose creates draw it.
struct ProbeTarget {
	SslmGpuContext* ctx = nullptr;
	SslmGpuModelHandle* model = nullptr;
	SslmGpuKvPoolHandle* pool = nullptr;
};
// Builds the state under probe in `sc` (fresh handles every call: the refusal leg is a rebuild)
// and returns the pools a cell probes. `leg` is 0 for the admission leg and 1 for the refusal leg.
using Builder = std::function<std::vector<ProbeTarget>(Scene& sc, int leg)>;

// k pages as creates of r pages each, 2 <= r <= ceil(cap / B) (§8: "no create reserves fewer
// than 2"). Requires k >= 2.
inline std::vector<uint32_t> SplitPages(const Geometry& g, uint32_t k) {
	std::vector<uint32_t> parts;
	const uint32_t maxr = g.cap_pages;
	while (k > 0) {
		uint32_t r = std::min(k, maxr);
		if (k - r == 1) r -= 1;  // never leave a remainder of 1
		parts.push_back(r);
		k -= r;
	}
	return parts;
}
// Creates totalling `k` pages on `t`; true iff all are admitted. Every created sequence is released
// before returning (they live in `sc`).
inline bool CreatesAdmitted(Scene& sc, const Geometry& g, const ProbeTarget& t, uint32_t k) {
	bool all = true;
	for (uint32_t r : SplitPages(g, k)) {
		SslmGpuSequenceHandle* s = nullptr;
		const int32_t budget = static_cast<int32_t>((r - 1) * g.B);  // R((r - 1) * B) = r (§8)
		const S st = sslm_gpu_seq_create_budgeted(t.ctx, t.model, t.pool, budget, &s);
		if (st == S::SSLM_OK && s) sc.seqs.push_back({t.ctx, s});
		else all = false;
	}
	return all;
}
// "Exactly k pages are free" on target `index` of the construction: creates totalling k are all
// admitted; on a fresh rebuild, creates totalling k + 1 are not. Each leg drains every sequence of
// the construction first.
inline bool GpuFillProbe(const Geometry& g, const Builder& build, size_t index, uint32_t k, const char* label) {
	if (k < 2) {
		G1B_CHECK_MSG(false, "%s: fill probe asked for k = %u (< 2): the cell must size its pool larger", label, k);
		return false;
	}
	bool admitted = false, refused = false;
	{
		Scene sc;
		std::vector<ProbeTarget> t = build(sc, 0);
		if (index < t.size() && t[index].pool) {
			sc.DrainAll();
			admitted = CreatesAdmitted(sc, g, t[index], k);
		}
	}
	{
		Scene sc;
		std::vector<ProbeTarget> t = build(sc, 1);
		if (index < t.size() && t[index].pool) {
			sc.DrainAll();
			refused = !CreatesAdmitted(sc, g, t[index], k + 1);
		}
	}
	G1B_CHECK_MSG(admitted && refused, "%s: GPU fill probe: exactly %u pages free? admission leg %s, refusal leg (k + 1) %s",
	              label, k, admitted ? "admitted" : "REFUSED", refused ? "refused" : "ADMITTED");
	return admitted && refused;
}

// ---- stderr capture (11.4: each refusal's HRESULT and failing call, as the library logs them) ---

// Runs `fn` with this process's stderr (fd 2) redirected to a temporary file and returns what was
// written. superslm_gpu's SSLM_GPU_HR prints "superslm_gpu: HR FAIL 0x%08lx at <file>:<line>".
inline std::string CaptureStderr(const std::function<void()>& fn) {
	char dir[MAX_PATH] = {}, path[MAX_PATH] = {};
	GetTempPathA(MAX_PATH, dir);
	GetTempFileNameA(dir, "g1b", 0, path);
	std::fflush(stderr);
	const int saved = _dup(2);
	const int fd = _open(path, _O_CREAT | _O_TRUNC | _O_WRONLY | _O_BINARY, _S_IREAD | _S_IWRITE);
	if (saved < 0 || fd < 0) {
		fn();
		return std::string();
	}
	_dup2(fd, 2);
	fn();
	std::fflush(stderr);
	_dup2(saved, 2);
	_close(fd);
	_close(saved);
	std::vector<uint8_t> bytes;
	ReadFileBytes(path, &bytes);
	DeleteFileA(path);
	return std::string(bytes.begin(), bytes.end());
}

// ---- an independent schema walk (tests/t2130-g5-red-suite's technique): legal schema-content tokens
// built from the artifact's bytes, not from the GPU model handle ------------------------------------

struct SchemaWalk {
	superslm::SchemaMasksTable table;
	const superslm::SchemaEntry* entry = nullptr;
	bool Build(const Artifact& a, size_t index) {
		std::string err;
		const superslm::SslmSectionView* s = a.view.Section(superslm::SslmSectionType::SchemaMasks);
		if (!s || !superslm::SchemaMasksTable::Parse(s->data, s->byte_size, a.view.config.vocab_size, table, &err)) return false;
		entry = table.ByIndex(index);
		vocab = a.view.config.vocab_size;
		return entry != nullptr;
	}
	// `n` tokens of a legal walk from state 0 (the lowest legal token at each state); stops early
	// at a state with no legal token. `*end_state` is the state reached.
	std::vector<int32_t> LegalTokens(int32_t n, uint32_t* end_state) const {
		std::vector<int32_t> out;
		uint32_t state = 0;
		for (int32_t i = 0; i < n; ++i) {
			bool found = false;
			for (uint32_t t = 0; t < table_vocab() && !found; ++t) {
				uint32_t next = 0;
				if (table.Transition(*entry, state, t, &next)) {
					out.push_back(static_cast<int32_t>(t));
					state = next;
					found = true;
				}
			}
			if (!found) break;
		}
		if (end_state) *end_state = state;
		return out;
	}
	uint32_t vocab = 0;
	uint32_t table_vocab() const { return vocab; }
};

}  // namespace g1b

#endif  // SSLM_G1B_HELPERS_H

// Paged-KV plan (rev 16.2), step G1, group g1c: the shared harness for cells 3.6, 3.6b, 3.6c and
// 3.6d of plan §7 (the GPU pool's interleaving and §4.4's entry guard).
//
// WHAT IS UNDER TEST. §4.4 (item 4 of the GPU design): `EntrySubmitGuard` at the head of every
// "guarded" row of §4.4's derived table, its three scope conditions (any sequence Submitted for the
// arc verbs and for every existing entry point called on a pooled holder; a pooled sequence
// Submitted for every existing entry point called on a dedicated sequence, a model or an adapter;
// the singleton's ledger unobserved for every singleton-reaching row), its status precedence, the
// batch's two channels, the wait seam `sslm_gpu_test_only_wait_in_flight`, and the bench
// configuration build. None of it exists before G3, and none of the arc verbs these cells call has a
// definition before its step, so every cell here is RED BY LINK today (the suite convention:
// tests/t2791-gpu-prefill-read-red-suite/fixture_common.h documents it): unresolved externals on
// the verbs include/superslm/gpu_1p0.h declares at 63e7b0e with no definition.
//
// GRADING (plan §7, "Three rules for every cell"). Every verdict below reads only:
//   - ABI statuses, and out-handles a refusal must leave null;
//   - tokens, compared with an UNSAVED TWIN: the same construction, on fresh handles, with the
//     refused call left out (so the twin reaches the call only after the drain, where it succeeds);
//   - raw bytes read through the GPU mirror peek (`sslm_gpu_test_only_peek_pool_mirror_bytes`), every
//     page of every pool the construction made, `page_bytes` computed here from the §3.1 formula;
//   - blob bytes (the G5 re-save, compared with the blob saved before the refused call);
//   - the debug-layer message count, read by the test from the device's info queue (3.6, the scope
//     control of 3.6d);
//   - the status of a public verb called afterwards on the state the cell built (SetSchema, the
//     prefix's set_schema, the follow-on bridge decode);
//   - the GPU FILL PROBE of §8, built here from `sslm_gpu_seq_create_budgeted` and the §3.4 formula
//     written here, read after `sslm_gpu_ready` has drained every in-flight sequence.
// `sslm_gpu_kv_pool_stats` and `*out_shared_pages` are diagnostics: printed, never graded.
//
// The fill probe and the GPU mirror peek are G3 instruments; until G3 commissions them (§8) their
// readings are quarantined, which here means: the cell records them and is not cited as evidence
// before the commissioning is on file.
#ifndef SSLM_G1C_HELPERS_H
#define SSLM_G1C_HELPERS_H

// pkvg unification: PKVG_STEP (3, 4 or 5; set by build_paged_kv_gpu.bat / _mingw.sh) keeps only the cell
// parts (and the teardown calls) whose bracket is at or before that step, so a step's executable links
// only the verbs that step defines. Unset, every part is compiled (the group's original build).
#ifndef PKVG_STEP
#define PKVG_STEP 5
#endif

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <iterator>
#include <string>
#include <vector>

#include "superslm/gpu_1p0.h"
#include "superslm/gpu_1p0_bench_bridge.h"
#include "superslm/gpu_1p0_g5_bridge.h"
#include "superslm/gpu_port.h"  // superslm_gpu::DispatchesPerLayer
#include "superslm/model.h"

using enum SslmGpuStatus;

namespace g1c {

// =============================================================================================
// Registry, checks and the run state (the shape of tests/paged-kv/pkv_common.h's PKV_CELL).
// =============================================================================================
struct RunState {
	int checks = 0;
	int failures = 0;
	int skips = 0;
	int cell_failures = 0;
	const char* cell = "";
};
inline RunState& State() {
	static RunState s;
	return s;
}

struct CellEntry {
	const char* id;    // "3.6d/G3:bridge-decode-not-ready"
	const char* step;  // the part's owning step: "G3", "G4" or "G5"
	void (*fn)();
};
inline std::vector<CellEntry>& Registry() {
	static std::vector<CellEntry> r;
	return r;
}
struct Registrar {
	Registrar(const char* id, const char* step, void (*fn)()) { Registry().push_back({id, step, fn}); }
};
#define G1C_CAT2(a, b) a##b
#define G1C_CAT(a, b) G1C_CAT2(a, b)
#define G1C_CELL(id, step, fn) static ::g1c::Registrar G1C_CAT(g1c_registrar_, __LINE__)(id, step, fn)

#define G1C_CHECK(cond, ...)                                                                       \
	do {                                                                                           \
		++::g1c::State().checks;                                                                   \
		if (!(cond)) {                                                                             \
			++::g1c::State().failures;                                                             \
			++::g1c::State().cell_failures;                                                        \
			std::printf("FAIL [%s] %s:%d: %s -- ", ::g1c::State().cell, __FILE__, __LINE__, #cond); \
			std::printf(__VA_ARGS__);                                                              \
			std::printf("\n");                                                                     \
			std::fflush(stdout);                                                                   \
		}                                                                                          \
	} while (0)

// A skip fails the run unless --allow-skip (tests/t2791's rule: a run that left an artifact out
// is not a green run).
#define G1C_SKIP(...)                                                \
	do {                                                             \
		++::g1c::State().skips;                                      \
		std::printf("SKIP [%s] %s:%d -- ", ::g1c::State().cell, __FILE__, __LINE__); \
		std::printf(__VA_ARGS__);                                    \
		std::printf("\n");                                           \
	} while (0)

#define G1C_NOTE(...)                              \
	do {                                           \
		std::printf("  note [%s]: ", ::g1c::State().cell); \
		std::printf(__VA_ARGS__);                  \
		std::printf("\n");                         \
	} while (0)

inline const char* St(SslmGpuStatus s) {
	switch (static_cast<uint32_t>(s)) {
		case 0: return "SSLM_OK";
		case 1: return "SSLM_DISPATCH_BUDGET_TOO_SMALL";
		case 2: return "SSLM_BUSY";
		case 3: return "SSLM_CONTEXT_HAS_LIVE_HANDLES";
		case 4: return "SSLM_MODEL_HAS_LIVE_SEQUENCES";
		case 5: return "SSLM_ADAPTER_MODEL_MISMATCH";
		case 6: return "SSLM_ADAPTER_BASE_HASH_MISMATCH";
		case 7: return "SSLM_SEQUENCE_KV_BUFFER_MISMATCH";
		case 8: return "SSLM_DEVICE_LOST";
		case 9: return "SSLM_BATCH_BUDGET_EXHAUSTED";
		case 10: return "SSLM_TOKEN_ID_OUT_OF_RANGE";
		case 11: return "SSLM_SEQUENCE_REJECTED";
		case 12: return "SSLM_RESTORE_MODEL_MISMATCH";
		case 13: return "SSLM_MODEL_HAS_LIVE_ADAPTERS";
		case 14: return "SSLM_ADAPTER_HAS_BOUND_SEQUENCES";
		case 15: return "SSLM_GPU_SHADER_BINARY_STALE";
		case 16: return "SSLM_GPU_ALLOCATION_FAILED";
		case 17: return "SSLM_OUTPUT_BUFFER_TOO_SMALL";
		case 18: return "SSLM_PREFILL_HIDDEN_UNAVAILABLE";
		case 19: return "SSLM_GPU_SHADER_DIR_INVALID";
		case 20: return "SSLM_GPU_SHADER_DIR_CONFLICT";
		case 21: return "SSLM_GPU_PARALLEL_FOR_INVALID";
		case 22: return "SSLM_GPU_PARALLEL_FOR_INCOMPLETE";
		case 23: return "SSLM_GPU_RESIDENCY_FLAGS_INVALID";
		case 24: return "SSLM_GPU_KV_BUDGET_EXCEEDED";
		case 25: return "SSLM_GPU_KV_POOL_EXHAUSTED";
		case 26: return "SSLM_GPU_KV_POOL_TOO_LARGE";
		case 27: return "SSLM_GPU_HOLDER_MODE_MISMATCH";
		case 28: return "SSLM_GPU_PREFIX_REJECTED";
		case 29: return "SSLM_GPU_PREFIX_SCHEMA_MISMATCH";
		default: return "UNKNOWN_STATUS";
	}
}

// =============================================================================================
// Options (argv). Every artifact path is supplied, never hardcoded (tests/t2112's rule).
//   --model=PATH               the model under test. Default: $SUPERSLM_PAGED_KV_FIXTURE_DIR/pkv_def.sslm
//                              (tools/gen_paged_kv_fixture.py). It must carry a schema (--schema) for
//                              the schema-content runs, and a LoRA adapter of its base (--adapter) for
//                              adapter map's runs.
//   --adapter=PATH             an adapter artifact whose base is --model (3.6d's adapter-map run).
//   --adapter-other-base=PATH  an adapter artifact of a DIFFERENT base (3.6d's adapter-map precedence
//                              run: SSLM_ADAPTER_BASE_HASH_MISMATCH with nothing Submitted).
//   --schema=NAME              the schema the schema-bound runs bind (default g5_minimal_one_field, pkv_def's).
//   --pool-pages=N             the default pool size (default 256).
//   --allow-skip               a deliberate partial run: skips do not fail it (never an acceptance run).
// Positional arguments select cells by id prefix ("=ID" exactly), as tests/paged-kv/pkv_main.cpp.
// =============================================================================================
struct Options {
	std::string model;
	std::string adapter;
	std::string adapter_other_base;
	std::string schema = "g5_minimal_one_field";
	uint32_t pool_pages = 256;
	bool allow_skip = false;
	std::vector<std::string> selectors;
};
inline Options& Opt() {
	static Options o;
	return o;
}
inline void ParseOptions(int argc, char** argv) {
	Options& o = Opt();
	if (const char* d = std::getenv("SUPERSLM_PAGED_KV_FIXTURE_DIR")) {
		if (*d) o.model = std::string(d) + "/pkv_def.sslm";
	}
	for (int i = 1; i < argc; ++i) {
		const std::string a = argv[i];
		auto take = [&](const char* flag) -> const char* {
			const size_t n = std::strlen(flag);
			return a.compare(0, n, flag) == 0 ? a.c_str() + n : nullptr;
		};
		if (const char* v = take("--model=")) o.model = v;
		else if (const char* v = take("--adapter-other-base=")) o.adapter_other_base = v;
		else if (const char* v = take("--adapter=")) o.adapter = v;
		else if (const char* v = take("--schema=")) o.schema = v;
		else if (const char* v = take("--pool-pages=")) o.pool_pages = static_cast<uint32_t>(std::atoi(v));
		else if (a == "--allow-skip") o.allow_skip = true;
		else if (a.rfind("--", 0) == 0) {
			std::printf("unknown flag %s\n", a.c_str());
			++State().failures;
		} else {
			o.selectors.push_back(a);
		}
	}
}

inline bool ReadAll(const std::string& path, std::vector<uint8_t>* out) {
	std::ifstream f(path, std::ios::binary);
	if (!f) return false;
	out->assign(std::istreambuf_iterator<char>(f), {});
	return !out->empty();
}

// =============================================================================================
// One artifact, parsed here, with the page geometry WRITTEN HERE from plan §3.1 and §3.4 -- never
// read back from the code under test.
// =============================================================================================
struct Artifact {
	std::string path;
	std::vector<uint8_t> bytes;
	superslm::SslmModelView view{};
	bool ok = false;

	bool Load(const std::string& p) {
		path = p;
		std::string err;
		if (!ReadAll(p, &bytes)) return false;
		ok = superslm::SslmModel::Load(bytes.data(), bytes.size(), view, &err) == superslm::SslmModelStatus::Ok;
		if (!ok) std::printf("artifact %s: load failed: %s\n", p.c_str(), err.c_str());
		return ok;
	}
	uint32_t Layers() const { return view.config.num_hidden_layers; }
	uint32_t KvHeads() const { return view.config.num_key_value_heads; }
	uint32_t HeadDim() const { return view.config.head_dim; }
	int64_t Cap() const { return static_cast<int64_t>(view.config.context_cap); }
	int32_t Vocab() const { return static_cast<int32_t>(view.config.vocab_size); }
	size_t Elem() const { return view.config.kv_precision == superslm::SslmKvPrecision::Int16 ? 2u : 1u; }
	// §3.1: B = kv_block_size when it divides context_cap (and is <= it), else context_cap.
	int64_t B() const {
		const int64_t kb = view.config.kv_block_size;
		return (kb > 0 && kb <= Cap() && Cap() % kb == 0) ? kb : Cap();
	}
	int64_t CapPages() const { return (Cap() + B() - 1) / B(); }
	// §3.4: R(budget) = min(ceil(budget / B) + 1, ceil(cap / B)).
	int64_t R(int64_t budget) const {
		const int64_t r = (budget + B() - 1) / B() + 1;
		return r < CapPages() ? r : CapPages();
	}
	// §8: a create of budget (r - 1) * B reserves exactly r pages, 2 <= r <= ceil(cap / B).
	int32_t BudgetForPages(int64_t r) const { return static_cast<int32_t>((r - 1) * B()); }
	// §3.1: page_bytes = L * 2 * H_kv * B * D * elem.
	size_t PageBytes() const {
		return size_t{Layers()} * 2u * KvHeads() * static_cast<size_t>(B()) * HeadDim() * Elem();
	}
	bool HasQkNorm() const {
		for (uint32_t l = 0; l < Layers(); ++l) {
			if (view.weights.Tensor("layer" + std::to_string(l) + ".q_norm.gain") != nullptr) return true;
		}
		return false;
	}
};

// `n` valid tokens starting at `base` (deterministic, the t2791 Run() shape).
inline std::vector<int32_t> Toks(const Artifact& a, size_t n, int32_t base) {
	std::vector<int32_t> t;
	for (size_t i = 0; i < n; ++i) t.push_back(static_cast<int32_t>((base + 37 * static_cast<int32_t>(i)) % a.Vocab()));
	return t;
}

// =============================================================================================
// The process's stack: one context, the model mapped twice -- host logits (flags 0) and device
// logits (SSLM_GPU_RESIDENCY_HEAD_ON_DEVICE) -- and the schema index on each. Opened once.
// =============================================================================================
struct Stack {
	Artifact art;
	SslmGpuContext* ctx = nullptr;
	SslmGpuModelHandle* host = nullptr;  // host-logits model
	SslmGpuModelHandle* dev = nullptr;   // device-logits model
	int32_t schema_host = -1;
	int32_t schema_dev = -1;
	uint32_t one_layer = 0;   // superslm_gpu::DispatchesPerLayer: the smallest admitted decode budget
	uint32_t full_depth = 0;  // one token to full depth in one decode call
	bool ok = false;

	bool Open() {
		const Options& o = Opt();
		if (o.model.empty()) {
			G1C_SKIP("no --model and SUPERSLM_PAGED_KV_FIXTURE_DIR unset (tools/gen_paged_kv_fixture.py DIR)");
			return false;
		}
		if (!art.Load(o.model)) {
			G1C_CHECK(false, "cannot load --model=%s", o.model.c_str());
			return false;
		}
		GpuContextConfig cfg{};
		const SslmGpuStatus c = sslm_gpu_context_create(cfg, &ctx);
		G1C_CHECK(c == SSLM_OK && ctx, "sslm_gpu_context_create -> %s", St(c));
		if (!ctx) return false;
		GpuResidencyConfig hc{};
		const SslmGpuStatus mh = sslm_gpu_model_map(ctx, &art.view, hc, &host);
		GpuResidencyConfig dc{};
		dc.flags = SSLM_GPU_RESIDENCY_HEAD_ON_DEVICE;
		const SslmGpuStatus md = sslm_gpu_model_map(ctx, &art.view, dc, &dev);
		G1C_CHECK(mh == SSLM_OK && host, "model map (host logits) -> %s", St(mh));
		G1C_CHECK(md == SSLM_OK && dev, "model map (device logits) -> %s", St(md));
		if (!host || !dev) return false;
		if (SslmGpuModelHasSchemasForG5Bridge(host)) {
			schema_host = SslmGpuSchemaLookupForG5Bridge(host, o.schema.c_str());
			schema_dev = SslmGpuSchemaLookupForG5Bridge(dev, o.schema.c_str());
		}
		one_layer = superslm_gpu::DispatchesPerLayer(art.HasQkNorm());
		full_depth = one_layer * art.Layers();
		std::printf("stack: %s layers=%u kv_heads=%u head_dim=%u cap=%lld B=%lld page_bytes=%zu "
		            "one_layer=%u schema(%s)=%d\n",
		            o.model.c_str(), art.Layers(), art.KvHeads(), art.HeadDim(), static_cast<long long>(art.Cap()),
		            static_cast<long long>(art.B()), art.PageBytes(), one_layer, o.schema.c_str(), schema_host);
		ok = true;
		return true;
	}
};
// The stack is opened on first use and kept for the process (each executable is one process).
inline Stack* GetStack() {
	static Stack s;
	static bool tried = false;
	if (!tried) {
		tried = true;
		s.Open();
	}
	return s.ok ? &s : nullptr;
}

// =============================================================================================
// Primitives.
// =============================================================================================
// Drains `seq` with blocking polls and returns the drained call's out-status.
inline SslmGpuStatus Ready(SslmGpuContext* ctx, SslmGpuSequenceHandle* seq) {
	int32_t ready = 0;
	SslmGpuStatus out = SSLM_DEVICE_LOST;
	for (int guard = 0; !ready && guard < 1000000; ++guard) {
		const SslmGpuStatus st = sslm_gpu_ready(ctx, seq, /*block=*/1, &ready, &out);
		if (st != SSLM_OK) return st;
	}
	return ready ? out : SSLM_DEVICE_LOST;
}

inline std::vector<uint8_t> Save(SslmGpuContext* ctx, SslmGpuSequenceHandle* seq, SslmGpuStatus* st = nullptr) {
	size_t n = 0;
	(void)sslm_gpu_seq_save(ctx, seq, nullptr, &n);
	std::vector<uint8_t> b(n);
	const SslmGpuStatus s = n ? sslm_gpu_seq_save(ctx, seq, b.data(), &n) : SSLM_SEQUENCE_KV_BUFFER_MISMATCH;
	if (st) *st = s;
	if (s != SSLM_OK) b.clear();
	return b;
}

// `n` tokens through the bridge decode, feeding each produced token back (the first embed is
// `first`, ignored when the sequence is ready for logits). The statuses are part of the output.
inline std::vector<int64_t> DecodeRun(Stack& s, SslmGpuSequenceHandle* seq, size_t n, int32_t first) {
	std::vector<int64_t> out;
	int32_t tok = first;
	for (size_t i = 0; i < n; ++i) {
		int32_t got = -1;
		const SslmGpuStatus st = SslmGpuSeqDecodeStepForG5Bridge(s.ctx, seq, tok, s.full_depth, &got);
		out.push_back(static_cast<int64_t>(st));
		out.push_back(got);
		if (st != SSLM_OK) break;
		tok = got >= 0 ? got : first;
	}
	return out;
}

// Every page of `pool` through the GPU mirror peek, `pages` pages of `page_bytes` each (§3.1,
// written here). A refused peek is a failed check and leaves a sentinel run.
inline std::vector<uint8_t> MirrorBytes(SslmGpuKvPoolHandle* pool, uint32_t pages, size_t page_bytes) {
	std::vector<uint8_t> all(size_t{pages} * page_bytes, 0xEE);
	for (uint32_t p = 0; p < pages; ++p) {
		const SslmGpuStatus st =
		    sslm_gpu_test_only_peek_pool_mirror_bytes(pool, p, 0, page_bytes, all.data() + size_t{p} * page_bytes);
		if (st != SSLM_OK) {
			G1C_CHECK(false, "mirror peek page %u -> %s", p, St(st));
			break;
		}
	}
	return all;
}

inline size_t FirstDiff(const std::vector<uint8_t>& a, const std::vector<uint8_t>& b) {
	const size_t n = std::min(a.size(), b.size());
	for (size_t i = 0; i < n; ++i) {
		if (a[i] != b[i]) return i;
	}
	return a.size() == b.size() ? SIZE_MAX : n;
}

// The debug layer's WARNING-or-worse message count since the last read (g1c_debug_layer.cpp: the
// one translation unit that includes the engine's d3d12_harness.h). `*enabled` is false when the
// run was not started with SSLM_GPU_ENABLE_DEBUG_LAYER=1 or the info queue is unavailable; a cell
// grading "the debug layer is clean" fails on that, because a count of 0 from no queue is vacuous.
size_t DebugLayerReportable(bool* enabled);

// =============================================================================================
// A World: the handles one construction makes, torn down in its destructor (every Submitted
// sequence drained first, so no later cell inherits a raised count).
// =============================================================================================
struct World {
	Stack& s;
	struct PoolRec {
		SslmGpuKvPoolHandle* h = nullptr;
		SslmGpuModelHandle* model = nullptr;
		uint32_t pages = 0;
		int64_t drawn = 0;  // pages the construction drew, by the §3.4 formula written here
	};
	std::vector<PoolRec> pools;
	std::vector<SslmGpuSequenceHandle*> seqs;
	std::vector<SslmGpuSequenceHandle*> submitted;
	std::vector<SslmGpuPrefixHandle*> prefixes;
	std::vector<SslmGpuModelHandle*> models;
	std::vector<SslmGpuAdapterHandle*> adapters;
	std::vector<SslmGpuKvPoolHandle*> loose_pools;  // pools a verb under test made (no accounting)
	bool ok = true;

	// Slots the constructions fill and the verbs under test write.
	SslmGpuSequenceHandle* A = nullptr;  // the submitter
	SslmGpuSequenceHandle* B = nullptr;  // the holder that calls the guarded verb
	SslmGpuSequenceHandle* C = nullptr;  // a second holder (the batch)
	SslmGpuPrefixHandle* P = nullptr;    // a frozen prefix, or the parent of begin_from
	SslmGpuPrefixHandle* Q = nullptr;    // a fresh prefix (prefix prefill)
	int32_t binding = -1;                // B's schema binding, as SetSchema re-states it
	std::vector<int32_t> toks;           // the tokens the guarded call consumes
	std::vector<uint8_t> blob;           // a saved blob the guarded call restores
	std::vector<SslmGpuStatus> slots;    // the batch's out_statuses
	int32_t i32 = -1;                    // an out token, or a consumed count
	uint32_t u32 = 0;                    // *out_shared_pages (a diagnostic)
	SslmGpuSequenceHandle* out_seq = nullptr;
	SslmGpuPrefixHandle* out_prefix = nullptr;
	SslmGpuKvPoolHandle* out_pool = nullptr;
	SslmGpuModelHandle* out_model = nullptr;
	SslmGpuAdapterHandle* out_adapter = nullptr;
	// Moves every out-handle a verb made into the teardown lists (a refused call under a mutant
	// may make one too).
	void TakeOuts() {
		if (out_seq) seqs.push_back(out_seq);
		if (out_prefix) prefixes.push_back(out_prefix);
		if (out_pool) loose_pools.push_back(out_pool);
		if (out_model) models.push_back(out_model);
		if (out_adapter) adapters.push_back(out_adapter);
		out_seq = nullptr;
		out_prefix = nullptr;
		out_pool = nullptr;
		out_model = nullptr;
		out_adapter = nullptr;
	}

	explicit World(Stack& st) : s(st) {}
	World(const World&) = delete;
	World& operator=(const World&) = delete;

	~World() {
		TakeOuts();
		for (SslmGpuSequenceHandle* q : submitted) (void)Ready(s.ctx, q);
		submitted.clear();
		for (SslmGpuSequenceHandle* q : seqs) (void)sslm_gpu_seq_release(s.ctx, q);
#if PKVG_STEP >= 4  // G4 verb; before G4 no part makes a prefix
		for (SslmGpuPrefixHandle* p : prefixes) (void)sslm_gpu_prefix_release(s.ctx, p);
#endif
		for (SslmGpuAdapterHandle* a : adapters) (void)sslm_gpu_adapter_unmap(s.ctx, a);
		for (SslmGpuModelHandle* m : models) (void)sslm_gpu_model_unmap(s.ctx, m);
		for (PoolRec& p : pools) {
			if (p.h) (void)sslm_gpu_kv_pool_destroy(s.ctx, p.h);
		}
		for (SslmGpuKvPoolHandle* p : loose_pools) (void)sslm_gpu_kv_pool_destroy(s.ctx, p);
	}

	SslmGpuModelHandle* M(bool device_logits) const { return device_logits ? s.dev : s.host; }
	int32_t Schema(bool device_logits) const { return device_logits ? s.schema_dev : s.schema_host; }

	// A pool of `pages` pages on the host-logits or device-logits model; returns its index.
	int Pool(uint32_t pages, bool device_logits = false) {
		PoolRec r;
		r.model = M(device_logits);
		r.pages = pages;
		const SslmGpuStatus st = sslm_gpu_kv_pool_create(s.ctx, r.model, pages, &r.h);
		G1C_CHECK(st == SSLM_OK && r.h, "sslm_gpu_kv_pool_create(%u) -> %s", pages, St(st));
		ok = ok && r.h;
		pools.push_back(r);
		return static_cast<int>(pools.size()) - 1;
	}
	// A budgeted pooled sequence on pool `pi`; its R(budget) pages are added to the pool's drawn count.
	SslmGpuSequenceHandle* Pooled(int pi, int32_t budget) {
		SslmGpuSequenceHandle* q = nullptr;
		if (pi < 0 || pi >= static_cast<int>(pools.size()) || !pools[pi].h) {
			ok = false;
			return nullptr;
		}
		const SslmGpuStatus st = sslm_gpu_seq_create_budgeted(s.ctx, pools[pi].model, pools[pi].h, budget, &q);
		G1C_CHECK(st == SSLM_OK && q, "sslm_gpu_seq_create_budgeted(budget %d) -> %s", budget, St(st));
		if (q) {
			seqs.push_back(q);
			pools[pi].drawn += s.art.R(budget);
		}
		ok = ok && q;
		return q;
	}
	SslmGpuSequenceHandle* Dedicated(bool device_logits = false, int64_t cap = 64) {
		SslmGpuSequenceHandle* q = nullptr;
		const SslmGpuStatus st = sslm_gpu_seq_create(s.ctx, M(device_logits), cap, &q);
		G1C_CHECK(st == SSLM_OK && q, "sslm_gpu_seq_create(%lld) -> %s", static_cast<long long>(cap), St(st));
		if (q) seqs.push_back(q);
		ok = ok && q;
		return q;
	}
	// A budgeted prefix on pool `pi`, prefilled with `toks` through the prompt kind (0) and frozen if
	// asked. Its R(budget) pages are drawn; freeze's reserve return (§3.5) gives back the pages past
	// R(length), which the accounting below applies when `freeze` is set.
	SslmGpuPrefixHandle* Prefix(int pi, int32_t budget, const std::vector<int32_t>& toks, bool freeze) {
		SslmGpuPrefixHandle* p = nullptr;
		const SslmGpuStatus st = sslm_gpu_prefix_begin(s.ctx, pools[pi].model, pools[pi].h, budget, &p);
		G1C_CHECK(st == SSLM_OK && p, "sslm_gpu_prefix_begin(budget %d) -> %s", budget, St(st));
		if (!p) {
			ok = false;
			return nullptr;
		}
		prefixes.push_back(p);
		pools[pi].drawn += s.art.R(budget);
		if (!toks.empty()) {
			// §7 5.8: the verb's dispatch_budget may stop mid-span and the call resumes, so the build loops
			// on *consumed; each call passes one token's cost (full depth), so each admits at least one.
			int32_t consumed = 0;
			SslmGpuStatus pf = SSLM_OK;
			const int32_t n = static_cast<int32_t>(toks.size());
			while (consumed < n) {
				int32_t c = -1;
				pf = sslm_gpu_prefix_prefill(s.ctx, p, toks.data() + consumed, n - consumed, 0, s.full_depth, &c);
				if (c > 0) consumed += c;
				if (pf != SSLM_OK || c <= 0) break;
			}
			G1C_CHECK(pf == SSLM_OK && consumed == n, "sslm_gpu_prefix_prefill(%zu) -> %s consumed %d", toks.size(),
			          St(pf), consumed);
			ok = ok && pf == SSLM_OK && consumed == n;
		}
		if (freeze) {
			const SslmGpuStatus fz = sslm_gpu_prefix_freeze(s.ctx, p);
			G1C_CHECK(fz == SSLM_OK, "sslm_gpu_prefix_freeze -> %s", St(fz));
			ok = ok && fz == SSLM_OK;
			// §3.5: freeze returns the unused tail of the reserve -- the prefix keeps ceil(L / B) pages.
			const int64_t keep = (static_cast<int64_t>(toks.size()) + s.art.B() - 1) / s.art.B();
			pools[pi].drawn -= s.art.R(budget) - keep;
		}
		return p;
	}
	// Makes `q` Submitted: embed a token, then one decode call of one layer. Recorded for teardown.
	bool Submit(SslmGpuSequenceHandle* q, int32_t token) {
		const SslmGpuStatus e = sslm_gpu_seq_embed_token(s.ctx, q, token);
		const SslmGpuStatus d = e == SSLM_OK ? sslm_decode_step_gpu(s.ctx, q, nullptr, s.one_layer) : e;
		G1C_CHECK(e == SSLM_OK && d == SSLM_OK, "SETUP: making a sequence Submitted: embed %s decode %s", St(e), St(d));
		if (d == SSLM_OK) submitted.push_back(q);
		ok = ok && d == SSLM_OK;
		return d == SSLM_OK;
	}
	SslmGpuStatus Drain(SslmGpuSequenceHandle* q) {
		const SslmGpuStatus st = Ready(s.ctx, q);
		submitted.erase(std::remove(submitted.begin(), submitted.end(), q), submitted.end());
		return st;
	}
	void DrainAll() {
		std::vector<SslmGpuSequenceHandle*> v = submitted;
		for (SslmGpuSequenceHandle* q : v) (void)Drain(q);
	}
	// Every pool's whole mirror, concatenated, through the GPU mirror peek.
	std::vector<uint8_t> Mirrors() const {
		std::vector<uint8_t> all;
		for (const PoolRec& p : pools) {
			if (!p.h) continue;
			const std::vector<uint8_t> m = MirrorBytes(p.h, p.pages, s.art.PageBytes());
			all.insert(all.end(), m.begin(), m.end());
		}
		return all;
	}
	int64_t FreePages(int pi) const { return static_cast<int64_t>(pools[pi].pages) - pools[pi].drawn; }
};

// =============================================================================================
// The GPU fill probe (plan §8), two-sided: "exactly k pages are free" on pool `pi`.
//   admission leg: creates totalling k, each of 2..ceil(cap/B) pages, all admitted;
//   refusal leg:   on a FRESH REBUILD of the same state, creates totalling k + 1, split the same way,
//                  not all admitted -- and a refusal counts only when it is
//                  SSLM_GPU_KV_POOL_EXHAUSTED (the R0 instrument fix, d601af8): any other status is a
//                  failed check that names it.
// =============================================================================================
inline std::vector<int64_t> SplitPages(const Artifact& a, int64_t total) {
	std::vector<int64_t> parts;
	int64_t rem = total;
	while (rem > 0) {
		int64_t take = std::min<int64_t>(a.CapPages(), rem);
		if (rem - take == 1 && take > 2) --take;  // never leave a 1-page remainder (no create draws 1)
		if (take < 2) take = 2;                   // the last part of an odd total rounds up (k >= 2 only)
		parts.push_back(take);
		rem -= take;
	}
	return parts;
}

// One leg's creates on pool `pi`. Returns the number admitted; releases them (pooled release touches
// only the holder's own table and reserve, §4 item 6b) so the construction can continue.
struct ProbeLeg {
	int admitted = 0;
	int exhausted = 0;
	int other = 0;
	int parts = 0;
};
inline ProbeLeg ProbeCreates(World& w, int pi, int64_t total) {
	ProbeLeg r;
	std::vector<SslmGpuSequenceHandle*> made;
	for (int64_t pages : SplitPages(w.s.art, total)) {
		++r.parts;
		SslmGpuSequenceHandle* q = nullptr;
		const SslmGpuStatus st =
		    sslm_gpu_seq_create_budgeted(w.s.ctx, w.pools[pi].model, w.pools[pi].h, w.s.art.BudgetForPages(pages), &q);
		if (st == SSLM_OK && q) {
			++r.admitted;
			made.push_back(q);
		} else if (st == SSLM_GPU_KV_POOL_EXHAUSTED) {
			++r.exhausted;
		} else {
			++r.other;
			G1C_CHECK(false, "fill probe: a %lld-page create returned %s (only SSLM_GPU_KV_POOL_EXHAUSTED "
			                 "is a refusal)", static_cast<long long>(pages), St(st));
		}
	}
	for (SslmGpuSequenceHandle* q : made) (void)sslm_gpu_seq_release(w.s.ctx, q);
	return r;
}
inline bool AdmissionLeg(World& w, int pi, int64_t k, const char* what) {
	const ProbeLeg r = ProbeCreates(w, pi, k);
	const bool pass = r.admitted == r.parts && r.other == 0;
	G1C_CHECK(pass, "%s: fill probe admission leg: creates totalling %lld pages, %d of %d admitted", what,
	          static_cast<long long>(k), r.admitted, r.parts);
	return pass;
}
inline bool RefusalLeg(World& w, int pi, int64_t k, const char* what) {
	const ProbeLeg r = ProbeCreates(w, pi, k + 1);
	const bool pass = r.exhausted > 0 && r.other == 0;
	G1C_CHECK(pass, "%s: fill probe refusal leg: creates totalling %lld pages, %d of %d admitted, %d exhausted",
	          what, static_cast<long long>(k + 1), r.admitted, r.parts, r.exhausted);
	return pass;
}

}  // namespace g1c

#endif  // SSLM_G1C_HELPERS_H

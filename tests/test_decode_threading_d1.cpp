// Decode-threading plan (rev 1.2), step D1: the cells of the plan's coverage model (§8) that thread
// the one-row (M = 1) projections, on the generated CI fixtures (§3.9). Cell numbers are the plan's.
//
// Own translation unit because it calls the C ABI (test_slm18x_saturation_census.cpp explains why
// test_main.cpp cannot include sslm_abi.h on Windows). test_main.cpp calls RunDecodeThreadingD1Cells;
// the scalar-forced standalone executable (decode_threading_standalone_main.cpp) calls it too.
//
// The fixtures come from tools/gen_decode_threading_fixture.py, run by every CI leg that runs a suite,
// into the directory SUPERSLM_DECODE_THREADING_FIXTURE_DIR names. A missing fixture fails every cell
// that needs it; nothing here skips.
//
// "Reference" below is the same workload run with no hook (the serial path), or DotRowScalarRef; never
// the build under test's split. Every threaded cell first asserts that `run` was called with two or
// more tasks (the §8 preamble), and every guard cell asserts the F1 preamble before its restore check.

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <new>
#include <string>
#include <thread>
#include <vector>

#include "superslm/forward_sites.h"
#include "superslm/layer_marshal.h"
#include "superslm/matmul.h"
#include "superslm/model.h"
#include "superslm/parallel_for.h"
#include "superslm/sslm_abi.h"
#include "superslm/trace_hook.h"
#include "../src/forward/parallel_split.h"
#include "../docs/parallel_for_reference.hpp"

#if !defined(SUPERSLM_ENABLE_MATVEC_TEST_SEAMS)
#error "test_decode_threading_d1.cpp needs a library built with SUPERSLM_ENABLE_MATVEC_TEST_SEAMS"
#endif

extern "C" superslm::SequenceLayerState* SslmSeqLiveStateForTest(sslm_seq);

static int GChecks = 0;
static int GFailures = 0;

#define CHECK_MSG(cond, ...) \
	do { \
		++GChecks; \
		if (!(cond)) { \
			++GFailures; \
			std::printf("FAIL %s:%d: %s -- ", __FILE__, __LINE__, #cond); \
			std::printf(__VA_ARGS__); \
			std::printf("\n"); \
		} \
	} while (0)

namespace {

using superslm::ColumnSplit;
using superslm::GemmThreading;
using superslm::SaturationCounters;
using superslm::SslmForwardStatus;

constexpr size_t kSeam8K = 8 * 1024;
constexpr size_t kSeamBoundary = 18432;  // K·N / 2 for q, o and k + v on F-DEF
constexpr size_t kSeamAbove = 18433;
constexpr int32_t kBadToken = 100000;    // out of range for every fixture (vocab 256)

bool operator==(const SaturationCounters& a, const SaturationCounters& b) {
	return a.kv == b.kv && a.kv_landing == b.kv_landing && a.k_channel_landing == b.k_channel_landing &&
	       a.rope_q == b.rope_q && a.rope_k == b.rope_k;
}
bool operator!=(const SaturationCounters& a, const SaturationCounters& b) { return !(a == b); }

SaturationCounters Minus(const SaturationCounters& a, const SaturationCounters& b) {
	return SaturationCounters{a.kv - b.kv, a.kv_landing - b.kv_landing,
	                          a.k_channel_landing - b.k_channel_landing, a.rope_q - b.rope_q,
	                          a.rope_k - b.rope_k};
}

std::string Str(const SaturationCounters& c) {
	char buf[160];
	std::snprintf(buf, sizeof(buf), "{kv %llu, landing %llu, k_channel %llu, rope_q %llu, rope_k %llu}",
	              (unsigned long long)c.kv, (unsigned long long)c.kv_landing,
	              (unsigned long long)c.k_channel_landing, (unsigned long long)c.rope_q,
	              (unsigned long long)c.rope_k);
	return buf;
}

std::string Str(const std::vector<int32_t>& v) {
	std::string s = "{";
	for (size_t i = 0; i < v.size(); ++i) s += (i ? "," : "") + std::to_string(v[i]);
	return s + "}";
}

std::vector<int32_t> Repeat(const std::vector<int32_t>& v, int times) {
	std::vector<int32_t> out;
	for (int i = 0; i < times; ++i) out.insert(out.end(), v.begin(), v.end());
	return out;
}

std::vector<int32_t> Concat(std::vector<int32_t> a, const std::vector<int32_t>& b) {
	a.insert(a.end(), b.begin(), b.end());
	return a;
}

// The minimum-work seam, set for one scope (§3.9).
struct SeamScope {
	explicit SeamScope(size_t bytes) { superslm::test::SetMinRowBytesPerTask(bytes); }
	~SeamScope() { superslm::test::ResetMinRowBytesPerTask(); }
	SeamScope(const SeamScope&) = delete;
	SeamScope& operator=(const SeamScope&) = delete;
};

// ---- the fixtures ----------------------------------------------------------------------------------

struct Fixture {
	std::string name;
	bool ok = false;
	uint8_t* bytes = nullptr;
	size_t size = 0;
	sslm_model model = nullptr;
	superslm::SslmModelView view;
	std::vector<superslm_marshal::LayerBacking> backings;
	std::vector<superslm::LayerWeights> layers;
	const int8_t* embed = nullptr;
	superslm::CarriedScale embed_site_constant{};
	bool cpp_ok = false;
	~Fixture() {
		if (model) sslm_model_unmap(model);
		if (bytes) ::operator delete(bytes, std::align_val_t(SSLM_ABI_ALIGNMENT_BYTES));
	}
	uint32_t Layers() const { return view.config.num_hidden_layers; }
	size_t Hidden() const { return view.config.hidden_size; }
	size_t QWidth() const { return size_t{view.config.num_attention_heads} * view.config.head_dim; }
	size_t KvWidth() const { return size_t{view.config.num_key_value_heads} * view.config.head_dim; }
	size_t Inter() const { return view.config.intermediate_size; }
};

std::map<std::string, std::unique_ptr<Fixture>>& FixtureCache() {
	static std::map<std::string, std::unique_ptr<Fixture>> cache;
	return cache;
}

// Loads (once) a generated fixture. A missing directory variable or file is a failed check, so every
// cell that needs the fixture is red, never skipped.
const Fixture& GetFixture(const char* file) {
	auto& cache = FixtureCache();
	auto it = cache.find(file);
	if (it != cache.end()) {
		CHECK_MSG(it->second->ok, "fixture %s did not load (see the first failure)", file);
		return *it->second;
	}
	auto f = std::make_unique<Fixture>();
	f->name = file;
	const char* dir = std::getenv("SUPERSLM_DECODE_THREADING_FIXTURE_DIR");
	CHECK_MSG(dir != nullptr && *dir != '\0',
	          "SUPERSLM_DECODE_THREADING_FIXTURE_DIR is not set; run tools/gen_decode_threading_fixture.py "
	          "and point it at the output (a missing fixture fails the cell, §3.9)");
	if (dir != nullptr && *dir != '\0') {
		const std::string path = std::string(dir) + "/" + file;
		std::vector<uint8_t> raw;
		const bool read = superslm_marshal::ReadFile(path.c_str(), raw) && !raw.empty();
		CHECK_MSG(read, "fixture %s not found or empty", path.c_str());
		if (read) {
			f->size = raw.size();
			f->bytes = static_cast<uint8_t*>(
			    ::operator new(raw.size(), std::align_val_t(SSLM_ABI_ALIGNMENT_BYTES)));
			std::memcpy(f->bytes, raw.data(), raw.size());
			const sslm_status ms = sslm_model_map(f->bytes, f->size, &f->model);
			CHECK_MSG(ms == SSLM_OK, "sslm_model_map(%s) == %d", file, static_cast<int>(ms));
			std::string err;
			const bool loaded = superslm::SslmModel::Load(f->bytes, f->size, f->view, &err) ==
			                    superslm::SslmModelStatus::Ok;
			CHECK_MSG(loaded, "SslmModel::Load(%s): %s", file, err.c_str());
			f->ok = ms == SSLM_OK && loaded;
			if (f->ok) {
				// The C++ layer loop's inputs, as tools/sslm_layer_trace.cpp builds them.
				const uint32_t n = f->Layers();
				f->backings.resize(n);
				f->layers.resize(n);
				bool cpp_ok = true;
				for (uint32_t l = 0; l < n && cpp_ok; ++l) {
					std::string merr;
					cpp_ok = superslm_marshal::MarshalLayer(f->view, l, f->view.config.num_attention_heads,
					                                        f->view.config.num_key_value_heads, f->backings[l],
					                                        f->layers[l], &merr);
					CHECK_MSG(cpp_ok, "MarshalLayer(%s, %u): %s", file, l, merr.c_str());
				}
				const superslm::SslmTensorView* embed = f->view.weights.Tensor("embed");
				bool sc_ok = true;
				f->embed_site_constant =
				    superslm_marshal::ReadCarriedScale(f->view.composition_constants, "embed", &sc_ok);
				CHECK_MSG(embed != nullptr && sc_ok, "fixture %s: no embed tensor or site constant", file);
				f->embed = embed ? reinterpret_cast<const int8_t*>(embed->data) : nullptr;
				f->cpp_ok = cpp_ok && embed != nullptr && sc_ok;
			}
		}
	}
	const Fixture& ref = *f;
	cache[file] = std::move(f);
	return ref;
}

// ---- the test hook ---------------------------------------------------------------------------------

enum class RunMode { kInline, kReverse, kThreads8 };
enum class Hostile { kNone, kDuplicate, kOmit, kMinusOne, kTaskCount };

const char* HostileName(Hostile h) {
	switch (h) {
	case Hostile::kNone: return "none";
	case Hostile::kDuplicate: return "duplicate";
	case Hostile::kOmit: return "omit";
	case Hostile::kMinusOne: return "-1";
	case Hostile::kTaskCount: return "task_count";
	}
	return "?";
}

// Counts every `run` call and the task count it carried, in call order. Optionally breaks
// exactly-once on one call (0-based, counted from the last Clear), in one of four ways.
struct TestHook {
	RunMode mode = RunMode::kInline;
	Hostile hostile = Hostile::kNone;
	int hostile_call = -1;
	int32_t max_tasks = 4;
	std::vector<int32_t> counts;

	sslm_parallel_for Pf(uint32_t bits) { return sslm_parallel_for{&Run, this, max_tasks, bits}; }
	void Clear() { counts.clear(); }
	bool AllThreaded() const {
		for (int32_t c : counts)
			if (c < 2) return false;
		return !counts.empty();
	}

	static void Run(void* host_ctx, int32_t task_count, sslm_task_fn task, void* task_ctx) {
		TestHook& h = *static_cast<TestHook*>(host_ctx);
		const int call = static_cast<int>(h.counts.size());
		h.counts.push_back(task_count);
		if (h.hostile != Hostile::kNone && call == h.hostile_call) {
			switch (h.hostile) {
			case Hostile::kDuplicate:
				for (int32_t i = 0; i < task_count; ++i) task(task_ctx, i);
				task(task_ctx, 0);
				return;
			case Hostile::kOmit:
				for (int32_t i = 0; i + 1 < task_count; ++i) task(task_ctx, i);
				return;
			case Hostile::kMinusOne:
				task(task_ctx, -1);
				for (int32_t i = 0; i < task_count; ++i) task(task_ctx, i);
				return;
			case Hostile::kTaskCount:
				for (int32_t i = 0; i < task_count; ++i) task(task_ctx, i);
				task(task_ctx, task_count);
				return;
			case Hostile::kNone:
				break;
			}
		}
		switch (h.mode) {
		case RunMode::kInline:
			for (int32_t i = 0; i < task_count; ++i) task(task_ctx, i);
			break;
		case RunMode::kReverse:
			for (int32_t i = task_count - 1; i >= 0; --i) task(task_ctx, i);
			break;
		case RunMode::kThreads8: {
			std::vector<std::thread> threads;
			for (int32_t t = 0; t < 8; ++t) {
				threads.emplace_back([=] {
					for (int32_t i = t; i < task_count; i += 8) task(task_ctx, i);
				});
			}
			for (std::thread& th : threads) th.join();
			break;
		}
		}
	}
};

// ---- the ABI rig -----------------------------------------------------------------------------------

struct Rig {
	const Fixture& f;
	void* pool_buf = nullptr;
	sslm_kv_pool pool = nullptr;
	std::vector<void*> ws_bufs;
	std::vector<sslm_workspace> ws;
	std::vector<sslm_seq> seqs;
	bool ok = false;

	explicit Rig(const Fixture& fixture, int workspaces = 1, uint32_t blocks = 8) : f(fixture) {
		if (!f.ok) return;
		const size_t size = blocks * sslm_kv_block_size(f.model) + sslm_kv_pool_overhead_size(f.model, blocks);
		pool_buf = ::operator new(size, std::align_val_t(SSLM_ABI_ALIGNMENT_BYTES));
		const sslm_status ps = sslm_kv_pool_create(f.model, pool_buf, size, blocks, &pool);
		CHECK_MSG(ps == SSLM_OK, "sslm_kv_pool_create == %d", static_cast<int>(ps));
		ok = ps == SSLM_OK;
		const sslm_config cfg{2, 64, static_cast<int32_t>(f.Layers()), 0u};
		for (int i = 0; i < workspaces && ok; ++i) {
			const size_t wsz = sslm_workspace_size(f.model, &cfg);
			void* buf = ::operator new(wsz, std::align_val_t(SSLM_ABI_ALIGNMENT_BYTES));
			ws_bufs.push_back(buf);
			sslm_workspace w = nullptr;
			const sslm_status wst = sslm_workspace_create(f.model, &cfg, buf, wsz, &w);
			CHECK_MSG(wst == SSLM_OK, "sslm_workspace_create == %d", static_cast<int>(wst));
			ok = wst == SSLM_OK;
			ws.push_back(w);
		}
	}
	~Rig() {
		for (sslm_seq s : seqs) sslm_seq_release(s);
		for (sslm_workspace w : ws)
			if (w) sslm_workspace_destroy(w);
		for (void* b : ws_bufs) ::operator delete(b, std::align_val_t(SSLM_ABI_ALIGNMENT_BYTES));
		if (pool) sslm_kv_pool_destroy(pool);
		if (pool_buf) ::operator delete(pool_buf, std::align_val_t(SSLM_ABI_ALIGNMENT_BYTES));
	}
	sslm_seq NewSeq() {
		sslm_seq s = nullptr;
		const sslm_status st = sslm_seq_create(f.model, &pool, &s);
		CHECK_MSG(st == SSLM_OK, "sslm_seq_create == %d", static_cast<int>(st));
		if (s) seqs.push_back(s);
		return s;
	}
	void Release(sslm_seq s) {
		seqs.erase(std::remove(seqs.begin(), seqs.end(), s), seqs.end());
		sslm_seq_release(s);
	}
	void Install(TestHook* hook, uint32_t bits, int w = 0) {
		if (hook == nullptr) {
			CHECK_MSG(sslm_workspace_set_parallel_for(ws[w], nullptr) == SSLM_OK, "clearing the hook");
			return;
		}
		const sslm_parallel_for pf = hook->Pf(bits);
		const sslm_status st = sslm_workspace_set_parallel_for(ws[w], &pf);
		CHECK_MSG(st == SSLM_OK, "installing a hook with reserved %#x, max_tasks %d == %d", bits, pf.max_tasks,
		          static_cast<int>(st));
	}
	sslm_status Prefill(sslm_seq s, const int32_t* tokens, int32_t count, int32_t chunk, int32_t* consumed,
	                    int w = 0) {
		int32_t c = 0;
		const sslm_status st = sslm_prefill(f.model, s, tokens, count, chunk, SSLM_SPAN_PROMPT, ws[w], &c);
		if (consumed) *consumed = c;
		return st;
	}
	// Whole-prompt prefill in chunks of up to `chunk` tokens; checks every call.
	void PrefillAll(sslm_seq s, const std::vector<int32_t>& prompt, int32_t chunk, int w = 0) {
		for (size_t at = 0; at < prompt.size();) {
			const int32_t n = static_cast<int32_t>(std::min<size_t>(chunk, prompt.size() - at));
			int32_t consumed = 0;
			const sslm_status st = Prefill(s, prompt.data() + at, n, n, &consumed, w);
			CHECK_MSG(st == SSLM_OK && consumed == n, "prefill at %zu: status %d, consumed %d of %d", at,
			          static_cast<int>(st), consumed, n);
			if (st != SSLM_OK || consumed != n) return;
			at += static_cast<size_t>(n);
		}
	}
	sslm_status Decode(sslm_seq* s, int32_t n, int32_t layer_budget, int32_t* out, int w = 0,
	                   int32_t mode = SSLM_DECODE_MODE_GREEDY) {
		sslm_decode_params p{};
		const sslm_status ist = sslm_decode_params_init(f.model, mode, layer_budget, &p);
		if (ist != SSLM_OK) return ist;
		return sslm_decode_step_v2(f.model, s, n, &p, ws[w], out);
	}
	// `n` full decode steps at the full layer budget, returning the tokens; stops at a failure.
	std::vector<int32_t> DecodeN(sslm_seq s, int n, int w = 0, int32_t mode = SSLM_DECODE_MODE_GREEDY,
	                             std::vector<sslm_status>* statuses = nullptr) {
		std::vector<int32_t> toks;
		for (int i = 0; i < n; ++i) {
			int32_t t = -2;
			const sslm_status st = Decode(&s, 1, static_cast<int32_t>(f.Layers()), &t, w, mode);
			if (statuses) {
				statuses->push_back(st);
			} else {
				CHECK_MSG(st == SSLM_OK, "decode step %d == %d", i, static_cast<int>(st));
			}
			if (st != SSLM_OK) break;
			toks.push_back(t);
		}
		return toks;
	}
	std::vector<uint8_t> Save(sslm_seq s) {
		std::vector<uint8_t> blob(sslm_seq_state_size(f.model));
		size_t n = blob.size();
		const sslm_status st = sslm_seq_save(s, blob.data(), &n);
		CHECK_MSG(st == SSLM_OK, "sslm_seq_save == %d", static_cast<int>(st));
		blob.resize(st == SSLM_OK ? n : 0);
		return blob;
	}
};

superslm::SequenceLayerState& Live(sslm_seq s) { return *SslmSeqLiveStateForTest(s); }
SaturationCounters Ctr(sslm_seq s) { return SaturationCounters::Snapshot(Live(s)); }
std::vector<int8_t> Hidden(sslm_seq s, size_t hidden) {
	const superslm::SequenceLayerState& st = Live(s);
	return std::vector<int8_t>(st.hidden_codes, st.hidden_codes + hidden);
}

std::vector<int32_t> Prompt(size_t n, int mul = 37, int add = 11) {
	std::vector<int32_t> p;
	for (size_t i = 0; i < n; ++i) p.push_back(static_cast<int32_t>((i * mul + add) % 256));
	return p;
}

// ---- expected `run` sequences (§3.9 tables, max_tasks 4, alignment 64) ---------------------------------

// One decode token on F-DEF: five groups per layer (q, k + v, o, gate + up, down), then the logits.
std::vector<int32_t> DecodeTokenCounts(size_t seam) {
	std::vector<int32_t> layer;
	if (seam == kSeam8K) layer = {3, 3, 3, 4, 3};
	else if (seam == kSeamBoundary) layer = {2, 2, 2, 4, 3};
	else if (seam == kSeamAbove) layer = {4, 3};
	return Concat(Repeat(layer, 2), {4});
}
// One one-token prefill call on F-DEF: q, k, v, o, gate, up, down, each alone; no finish.
std::vector<int32_t> PrefillTokenCounts(size_t seam) {
	std::vector<int32_t> layer;
	if (seam == kSeam8K) layer = {3, 2, 2, 3, 4, 4, 3};
	else if (seam == kSeamBoundary) layer = {2, 2, 4, 4, 3};
	else if (seam == kSeamAbove) layer = {4, 4, 3};
	return Repeat(layer, 2);
}

// ---- a whole-workload outcome ----------------------------------------------------------------------

struct Outcome {
	std::vector<int32_t> tokens;
	std::vector<uint8_t> blob;
	SaturationCounters ctr;
	std::vector<int32_t> counts;
	bool operator==(const Outcome& o) const { return tokens == o.tokens && blob == o.blob && ctr == o.ctr; }
};

// Prefill `prompt` in one chunk with no hook, then install `hook` (null: none) and decode `n` tokens.
Outcome DecodeWorkload(const Fixture& f, const std::vector<int32_t>& prompt, int n, TestHook* hook,
                       uint32_t bits) {
	Outcome o;
	Rig rig(f);
	if (!rig.ok) return o;
	sslm_seq s = rig.NewSeq();
	if (!s) return o;
	rig.PrefillAll(s, prompt, 64);
	rig.Install(hook, bits);
	if (hook) hook->Clear();
	o.tokens = rig.DecodeN(s, n);
	o.blob = rig.Save(s);
	o.ctr = Ctr(s);
	if (hook) o.counts = hook->counts;
	return o;
}

// ---- 4.2: the rule, per site (M2, F4; rev 1.2's prefill oracle) ------------------------------------------

void TestRuleCell42() {
	const Fixture& f = GetFixture("fdef.sslm");
	if (!f.ok) return;
	const std::vector<int32_t> prompt = Prompt(16);
	const std::vector<int32_t> tail = Prompt(4, 53, 7);
	for (size_t seam : {kSeam8K, kSeamBoundary, kSeamAbove, superslm::kMinRowBytesPerTask}) {
		SeamScope scope(seam);
		// Decode: the first step after a prefill is the finish alone (the ready-for-logits path).
		{
			Rig rig(f);
			if (!rig.ok) return;
			sslm_seq s = rig.NewSeq();
			TestHook hook;
			rig.Install(&hook, SSLM_PARALLEL_FOR_MATVEC);
			rig.PrefillAll(s, prompt, 16);
			CHECK_MSG(hook.counts.empty(), "4.2 seam %zu: a 16-token prefill made %zu run calls, want 0", seam,
			          hook.counts.size());
			hook.Clear();
			rig.DecodeN(s, 1);
			CHECK_MSG(hook.counts == std::vector<int32_t>{4}, "4.2 seam %zu: ready-for-logits step made %s, want {4}",
			          seam, Str(hook.counts).c_str());
			for (int t = 0; t < 2; ++t) {
				hook.Clear();
				rig.DecodeN(s, 1);
				const std::vector<int32_t> want = DecodeTokenCounts(seam);
				CHECK_MSG(hook.counts == want, "4.2 decode seam %zu token %d: run task counts %s, want %s (%zu calls)",
				          seam, t, Str(hook.counts).c_str(), Str(want).c_str(), want.size());
			}
		}
		// One-token prefill: seven ungrouped sites per layer, no finish.
		{
			Rig rig(f);
			if (!rig.ok) return;
			sslm_seq s = rig.NewSeq();
			rig.PrefillAll(s, prompt, 16);
			TestHook hook;
			rig.Install(&hook, SSLM_PARALLEL_FOR_MATVEC);
			for (int32_t tok : tail) {
				hook.Clear();
				int32_t consumed = 0;
				const sslm_status st = rig.Prefill(s, &tok, 1, 1, &consumed);
				const std::vector<int32_t> want = PrefillTokenCounts(seam);
				CHECK_MSG(st == SSLM_OK && consumed == 1 && hook.counts == want,
				          "4.2 one-token prefill seam %zu: status %d, run task counts %s, want %s (%zu calls)", seam,
				          static_cast<int>(st), Str(hook.counts).c_str(), Str(want).c_str(), want.size());
			}
		}
	}
	// Bit 1 clear, and max_tasks 1 with bit 1: no matvec calls (the logits call only, or none).
	{
		SeamScope scope(kSeam8K);
		for (int variant = 0; variant < 2; ++variant) {
			Rig rig(f);
			if (!rig.ok) return;
			sslm_seq s = rig.NewSeq();
			rig.PrefillAll(s, prompt, 16);
			TestHook hook;
			hook.max_tasks = variant == 0 ? 4 : 1;
			rig.Install(&hook, variant == 0 ? 0u : SSLM_PARALLEL_FOR_MATVEC);
			rig.DecodeN(s, 3);
			int32_t tok = tail[0], consumed = 0;
			rig.Prefill(s, &tok, 1, 1, &consumed);
			const std::vector<int32_t> want = variant == 0 ? std::vector<int32_t>{4, 4, 4} : std::vector<int32_t>{};
			CHECK_MSG(hook.counts == want, "4.2 %s: run task counts %s, want %s",
			          variant == 0 ? "bit 1 clear" : "max_tasks 1", Str(hook.counts).c_str(), Str(want).c_str());
		}
	}
	std::printf("decode-threading 4.2: rule counts checked at 8 KiB, 18,432, 18,433 and 256 KiB\n");
}

// ---- 1.3: the seam moved between calls ---------------------------------------------------------------

void TestSeamBetweenCallsCell13() {
	const Fixture& f = GetFixture("fdef.sslm");
	if (!f.ok) return;
	const std::vector<int32_t> prompt = Prompt(16);
	const Outcome ref = DecodeWorkload(f, prompt, 4, nullptr, 0);
	Rig rig(f);
	if (!rig.ok) return;
	sslm_seq s = rig.NewSeq();
	rig.PrefillAll(s, prompt, 16);
	TestHook hook;
	rig.Install(&hook, SSLM_PARALLEL_FOR_MATVEC);
	std::vector<int32_t> toks = rig.DecodeN(s, 1);  // the finish of the prefilled prompt
	for (size_t seam : {kSeam8K, kSeamAbove, kSeam8K}) {
		SeamScope scope(seam);
		hook.Clear();
		const std::vector<int32_t> t = rig.DecodeN(s, 1);
		toks.insert(toks.end(), t.begin(), t.end());
		CHECK_MSG(hook.counts == DecodeTokenCounts(seam), "1.3 seam %zu: counts %s, want %s", seam,
		          Str(hook.counts).c_str(), Str(DecodeTokenCounts(seam)).c_str());
	}
	CHECK_MSG(toks == ref.tokens && rig.Save(s) == ref.blob, "1.3: tokens %s, reference %s (or the blob differs)",
	          Str(toks).c_str(), Str(ref.tokens).c_str());
}

// ---- 6.1: every projection site and each group, row by row against DotRowScalarRef --------------------

void TestAccAgainstScalarRefCell61() {
	SeamScope scope(kSeam8K);
	long long rows = 0, bad = 0, calls = 0;
	for (const char* file : {"fdef.sslm", "fqk.sslm"}) {
		const Fixture& f = GetFixture(file);
		if (!f.ok || !f.cpp_ok) continue;
		const size_t h = f.Hidden(), q = f.QWidth(), kv = f.KvWidth(), inter = f.Inter();
		for (uint32_t l = 0; l < f.Layers(); ++l) {
			const superslm::LayerWeights& lw = f.layers[l];
			struct Site {
				const char* name;
				std::vector<superslm::MatvecPart> parts;
				size_t k;
				std::vector<std::vector<int64_t>> outs;
			};
			std::vector<Site> sites;
			auto add = [&](const char* name, size_t k, std::vector<std::pair<const int8_t*, size_t>> mats) {
				Site s{name, {}, k, {}};
				for (auto& m : mats) s.outs.emplace_back(m.second, INT64_MIN);
				for (size_t i = 0; i < mats.size(); ++i) s.parts.push_back({mats[i].first, mats[i].second, nullptr});
				sites.push_back(std::move(s));
			};
			add("q", h, {{lw.q_weight, q}});
			add("k", h, {{lw.k_weight, kv}});
			add("v", h, {{lw.v_weight, kv}});
			add("o", q, {{lw.o_weight, h}});
			add("gate", h, {{lw.gate_weight, inter}});
			add("up", h, {{lw.up_weight, inter}});
			add("down", inter, {{lw.down_weight, h}});
			add("k+v", h, {{lw.k_weight, kv}, {lw.v_weight, kv}});
			add("gate+up", h, {{lw.gate_weight, inter}, {lw.up_weight, inter}});
			for (Site& site : sites) {
				std::vector<int8_t> x(site.k);
				for (size_t i = 0; i < site.k; ++i)
					x[i] = static_cast<int8_t>(static_cast<int>((i * 97 + l * 13 + site.k) % 255) - 127);
				size_t total = 0;
				for (auto& p : site.parts) total += p.rows;
				for (int32_t mt : {2, 3, 4, 7, 8, 256}) {
					TestHook hook;
					hook.max_tasks = mt;
					const sslm_parallel_for pf = hook.Pf(SSLM_PARALLEL_FOR_MATVEC);
					for (size_t i = 0; i < site.parts.size(); ++i) {
						std::fill(site.outs[i].begin(), site.outs[i].end(), INT64_MIN);
						site.parts[i].out = site.outs[i].data();
					}
					SslmForwardStatus st;
					if (site.parts.size() == 1) {
						// The single-matrix sites go through the one dispatch point at M = 1.
						const GemmThreading th{&pf, nullptr};
						st = superslm::GemmDispatch(th, x.data(), site.parts[0].weight, 1, site.k, total,
						                            site.outs[0].data());
					} else {
						const ColumnSplit split = superslm::MatvecGroupSplit(&pf, site.k, total);
						st = superslm::MatvecGroupParallel(pf, split, x.data(), site.k, site.parts.data(),
						                                   site.parts.size());
					}
					++calls;
					CHECK_MSG(st == SslmForwardStatus::Ok && hook.AllThreaded() && hook.counts.size() == 1,
					          "6.1 %s layer %u %s max_tasks %d: status %s, run counts %s (preamble: one call, >= 2 "
					          "tasks)",
					          file, l, site.name, mt, superslm::SslmForwardStatusName(st), Str(hook.counts).c_str());
					for (size_t i = 0; i < site.parts.size(); ++i) {
						for (size_t r = 0; r < site.parts[i].rows; ++r) {
							++rows;
							const int64_t want =
							    superslm::DotRowScalarRef(x.data(), site.parts[i].weight + r * site.k, site.k);
							if (site.outs[i][r] != want && ++bad <= 5)
								std::printf("  6.1 %s layer %u %s max_tasks %d part %zu row %zu: %lld, want %lld\n",
								            file, l, site.name, mt, i, r, (long long)site.outs[i][r], (long long)want);
						}
					}
				}
			}
		}
	}
	CHECK_MSG(bad == 0 && rows > 0, "6.1: %lld of %lld rows differ from DotRowScalarRef", bad, rows);
	std::printf("decode-threading 6.1: %lld group calls, %lld rows checked against DotRowScalarRef\n", calls, rows);
}

// ---- 4.3: group straddles (F8) --------------------------------------------------------------------------

void TestStraddlesCell43() {
	const Fixture& f = GetFixture("fdef.sslm");
	if (!f.ok || !f.cpp_ok) return;
	SeamScope scope(kSeam8K);
	const superslm::LayerWeights& lw = f.layers[1];
	struct Case {
		const char* name;
		int32_t max_tasks;
		const int8_t* w0;
		const int8_t* w1;
		size_t rows_each;
		size_t k;
		ColumnSplit want;
	};
	const Case cases[] = {
	    {"gate+up", 3, lw.gate_weight, lw.up_weight, f.Inter(), f.Hidden(), {384, 3}},
	    {"k+v", 3, lw.k_weight, lw.v_weight, f.KvWidth(), f.Hidden(), {64, 3}},
	    {"k+v", 2, lw.k_weight, lw.v_weight, f.KvWidth(), f.Hidden(), {128, 2}},
	};
	for (const Case& c : cases) {
		TestHook hook;
		hook.max_tasks = c.max_tasks;
		const sslm_parallel_for pf = hook.Pf(SSLM_PARALLEL_FOR_MATVEC);
		const ColumnSplit split = superslm::MatvecGroupSplit(&pf, c.k, 2 * c.rows_each);
		CHECK_MSG(split.rows_per_task == c.want.rows_per_task && split.task_count == c.want.task_count,
		          "4.3 %s max_tasks %d: split {%zu, %zu}, want {%zu, %zu}", c.name, c.max_tasks, split.rows_per_task,
		          split.task_count, c.want.rows_per_task, c.want.task_count);
		std::vector<int8_t> x(c.k);
		for (size_t i = 0; i < c.k; ++i) x[i] = static_cast<int8_t>(static_cast<int>((i * 71 + 5) % 255) - 127);
		std::vector<int64_t> a(c.rows_each, INT64_MIN), b(c.rows_each, INT64_MIN), ra(c.rows_each), rb(c.rows_each);
		const superslm::MatvecPart parts[2] = {{c.w0, c.rows_each, a.data()}, {c.w1, c.rows_each, b.data()}};
		const SslmForwardStatus st = superslm::MatvecGroupParallel(pf, split, x.data(), c.k, parts, 2);
		superslm::GemmInt8AccumulateRow(x.data(), c.w0, c.k, c.rows_each, ra.data());
		superslm::GemmInt8AccumulateRow(x.data(), c.w1, c.k, c.rows_each, rb.data());
		CHECK_MSG(st == SslmForwardStatus::Ok && hook.AllThreaded() && a == ra && b == rb,
		          "4.3 %s max_tasks %d: per-part acc bytes differ from the serial run (status %s)", c.name,
		          c.max_tasks, superslm::SslmForwardStatusName(st));
	}
	// Whole decode through the straddling splits.
	const std::vector<int32_t> prompt = Prompt(16);
	const Outcome ref = DecodeWorkload(f, prompt, 12, nullptr, 0);
	for (int32_t mt : {3, 2}) {
		TestHook hook;
		hook.max_tasks = mt;
		const Outcome o = DecodeWorkload(f, prompt, 12, &hook, SSLM_PARALLEL_FOR_MATVEC);
		CHECK_MSG(o.tokens.size() == 12 && o == ref && hook.AllThreaded(),
		          "4.3 decode at max_tasks %d differs from the no-hook run", mt);
	}
}

// ---- the C++ layer loop with a trace hook (6.2's trace half, 8.1's LoRA and Option-G halves) -------

struct TraceLog {
	std::vector<std::string> records;
};

void AppendTrace(const superslm::SslmChainTraceRecord* chain, const superslm::SslmKvLandingTraceRecord* kv,
                 void* user) {
	TraceLog& log = *static_cast<TraceLog*>(user);
	std::string r;
	char head[96];
	if (chain) {
		std::snprintf(head, sizeof(head), "|t%zu|m%lld|e%lld|", chain->token_index, (long long)chain->m_out,
		              (long long)chain->e_out);
		r = std::string("C|") + std::string(chain->site) + head;
		r.append(reinterpret_cast<const char*>(chain->codes.data()), chain->codes.size());
		r.append(reinterpret_cast<const char*>(chain->x_int.data()), chain->x_int.size_bytes());
	}
	if (kv) {
		std::snprintf(head, sizeof(head), "|t%zu|h%u|m%lld|e%lld|", kv->token_index, kv->head, (long long)kv->m_out,
		              (long long)kv->e_out);
		r += std::string("K|") + std::string(kv->site) + head;
		r.append(reinterpret_cast<const char*>(kv->codes.data()), kv->codes.size());
	}
	log.records.push_back(std::move(r));
}

struct CppRun {
	SslmForwardStatus status = SslmForwardStatus::Ok;
	std::vector<std::string> trace;
	std::vector<uint8_t> kv;
	std::vector<int8_t> hidden;
	superslm::CarriedScale scale{};
	SaturationCounters ctr;
};

// Runs `tokens` through EmbedEntry and the threaded RunLayerLoop overload, one whole token each, with a
// trace hook installed. `layers` defaults to the fixture's own.
CppRun RunCpp(const Fixture& f, const std::vector<int32_t>& tokens, const sslm_parallel_for* row, bool option_g,
              const superslm::LayerWeights* layers = nullptr) {
	CppRun out;
	const auto& c = f.view.config;
	const size_t kv_bytes =
	    size_t{c.num_hidden_layers} * c.context_cap * c.num_key_value_heads * c.head_dim * 2;
	out.kv.assign(kv_bytes, 0);
	out.hidden.assign(c.hidden_size, 0);
	superslm::SequenceLayerState seq;
	seq.hidden_codes = out.hidden.data();
	TraceLog log;
	superslm::SslmTraceHookState trace;
	superslm::SslmSetTraceHook(trace, &AppendTrace, &log);
	const GemmThreading th{row, nullptr};
	for (size_t i = 0; i < tokens.size(); ++i) {
		superslm::CarriedScale es{};
		out.status = superslm::EmbedEntry(tokens[i], static_cast<int32_t>(c.vocab_size), f.embed, c.hidden_size,
		                                  f.embed_site_constant, seq.hidden_codes, &es);
		if (out.status != SslmForwardStatus::Ok) break;
		seq.hidden_scale = es;
		seq.layer_index = 0;
		const superslm::LayerWeights* lw = layers ? layers : f.layers.data();
		if (option_g) {
			out.status = superslm::RunLayerLoop(
			    seq, lw, c.num_hidden_layers, c.num_hidden_layers, c.hidden_size, c.head_dim, c.num_key_value_heads,
			    c.intermediate_size, c.context_cap, f.view.rope_tables, out.kv.data(), out.kv.size(),
			    superslm::OptionGKLandingMode::kFused, "dt", i, &trace, f.QWidth(), th);
		} else {
			out.status = superslm::RunLayerLoop(seq, lw, c.num_hidden_layers, c.num_hidden_layers, c.hidden_size,
			                                    c.head_dim, c.num_key_value_heads, c.intermediate_size, c.context_cap,
			                                    f.view.rope_tables, out.kv.data(), out.kv.size(), "dt", i, &trace,
			                                    f.QWidth(), th);
		}
		if (out.status != SslmForwardStatus::Ok) break;
	}
	out.trace = std::move(log.records);
	out.scale = seq.hidden_scale;
	out.ctr = SaturationCounters::Snapshot(seq);
	return out;
}

bool SameRun(const CppRun& a, const CppRun& b) {
	return a.status == b.status && a.trace == b.trace && a.kv == b.kv && a.hidden == b.hidden &&
	       a.scale.m == b.scale.m && a.scale.e == b.scale.e && a.ctr == b.ctr;
}

// ---- 6.2: whole decode, tokens, trace records and K/V bytes ---------------------------------------------

void TestWholeDecodeCell62() {
	SeamScope scope(kSeam8K);
	for (const char* file : {"fdef.sslm", "fqk.sslm"}) {
		const Fixture& f = GetFixture(file);
		if (!f.ok) continue;
		const std::vector<int32_t> prompt = Prompt(16);
		const Outcome ref = DecodeWorkload(f, prompt, 64, nullptr, 0);
		TestHook hook;
		const Outcome o = DecodeWorkload(f, prompt, 64, &hook, SSLM_PARALLEL_FOR_MATVEC);
		CHECK_MSG(hook.AllThreaded() && hook.counts.size() > 64, "6.2 %s preamble: %zu run calls", file,
		          hook.counts.size());
		CHECK_MSG(o.tokens.size() == 64 && o.tokens == ref.tokens, "6.2 %s: 64 tokens differ from the no-hook run",
		          file);
		CHECK_MSG(o.blob == ref.blob && o.ctr == ref.ctr, "6.2 %s: final SSB5 blob (K/V bytes, counters) differs",
		          file);
		// The per-site trace records, teacher-forced over the same token stream through the C++ loop.
		if (!f.cpp_ok) continue;
		const std::vector<int32_t> stream = Concat(prompt, ref.tokens);
		const CppRun serial = RunCpp(f, stream, nullptr, false);
		TestHook th;
		const sslm_parallel_for pf = th.Pf(SSLM_PARALLEL_FOR_MATVEC);
		const CppRun threaded = RunCpp(f, stream, &pf, false);
		CHECK_MSG(serial.status == SslmForwardStatus::Ok && !serial.trace.empty(), "6.2 %s: serial C++ run %s", file,
		          superslm::SslmForwardStatusName(serial.status));
		CHECK_MSG(th.AllThreaded() && th.counts.size() == stream.size() * 10,
		          "6.2 %s trace preamble: %zu run calls for %zu tokens", file, th.counts.size(), stream.size());
		size_t first_diff = 0;
		while (first_diff < std::min(serial.trace.size(), threaded.trace.size()) &&
		       serial.trace[first_diff] == threaded.trace[first_diff])
			++first_diff;
		CHECK_MSG(SameRun(serial, threaded),
		          "6.2 %s: threaded trace/K/V differs from serial (%zu vs %zu records, first difference at %zu)", file,
		          threaded.trace.size(), serial.trace.size(), first_diff);
	}
}

// ---- 6.3: one-token prefill of a 64-token prompt ----------------------------------------------------------

void TestOneTokenPrefillCell63() {
	SeamScope scope(kSeam8K);
	const Fixture& f = GetFixture("fdef.sslm");
	if (!f.ok) return;
	const std::vector<int32_t> prompt = Prompt(64, 29, 3);
	std::vector<uint8_t> blobs[2];
	SaturationCounters ctr[2];
	for (int threaded = 0; threaded < 2; ++threaded) {
		Rig rig(f);
		if (!rig.ok) return;
		sslm_seq s = rig.NewSeq();
		TestHook hook;
		rig.Install(threaded ? &hook : nullptr, SSLM_PARALLEL_FOR_MATVEC);
		rig.PrefillAll(s, prompt, 1);
		if (threaded)
			CHECK_MSG(hook.AllThreaded() && hook.counts.size() == 64 * 14, "6.3 preamble: %zu run calls, want %d",
			          hook.counts.size(), 64 * 14);
		blobs[threaded] = rig.Save(s);
		ctr[threaded] = Ctr(s);
	}
	CHECK_MSG(!blobs[0].empty() && blobs[0] == blobs[1], "6.3: one-token-prefill SSB5 blob differs from chunk-1 no-hook");
	CHECK_MSG(ctr[0].kv > 0, "6.3: the prompt never saturates, so the blob's counters are vacuous");
}

// ---- 7.2: opt-in, zero `run` calls outside the finish with reserved = 0 ---------------------------------

void TestOptInCell72() {
	SeamScope scope(kSeam8K);
	for (const char* file : {"fdef.sslm", "fqk.sslm"}) {
		const Fixture& f = GetFixture(file);
		if (!f.ok) continue;
		Rig rig(f);
		if (!rig.ok) return;
		TestHook hook;
		rig.Install(&hook, 0u);
		sslm_seq s = rig.NewSeq();
		const std::vector<int32_t> prompt = Prompt(16);
		rig.PrefillAll(s, prompt, 1);                // one-token prefill
		CHECK_MSG(hook.counts.empty(), "7.2 %s: one-token prefill with reserved 0 made %zu run calls", file,
		          hook.counts.size());
		rig.DecodeN(s, 8);                           // decode: the finish only
		CHECK_MSG(sslm_seq_reset(s) == SSLM_OK, "reset");
		rig.PrefillAll(s, prompt, 8);
		rig.DecodeN(s, 4);
		const std::vector<int32_t> want(12, 4);
		CHECK_MSG(hook.counts == want, "7.2 %s: reserved 0 run calls %s, want the 12 finishes only %s", file,
		          Str(hook.counts).c_str(), Str(want).c_str());
	}
}

// ---- 1.1: lifetime and reuse ----------------------------------------------------------------------------

Outcome LifecycleWorkload(const Fixture& f, TestHook* hook) {
	Outcome o;
	Rig rig(f);
	if (!rig.ok) return o;
	rig.Install(hook, SSLM_PARALLEL_FOR_MATVEC);
	sslm_seq s = rig.NewSeq();
	const std::vector<int32_t> prompt = Prompt(16), prefix_prompt = Prompt(12, 41, 9);
	rig.PrefillAll(s, prompt, 1);
	o.tokens = rig.DecodeN(s, 32);
	CHECK_MSG(sslm_seq_reset(s) == SSLM_OK, "1.1 reset");
	sslm_prefix prefix = nullptr;
	CHECK_MSG(sslm_prefix_begin(f.model, &rig.pool, &prefix) == SSLM_OK, "1.1 prefix_begin");
	if (!prefix) return o;
	for (int32_t tok : prefix_prompt) {
		int32_t consumed = 0;
		CHECK_MSG(sslm_prefix_prefill(f.model, prefix, &tok, 1, 1, SSLM_SPAN_PROMPT, rig.ws[0], &consumed) ==
		                  SSLM_OK &&
		              consumed == 1,
		          "1.1 one-token prefix prefill");
	}
	CHECK_MSG(sslm_prefix_freeze(prefix) == SSLM_OK, "1.1 freeze");
	CHECK_MSG(sslm_seq_adopt_prefix(s, prefix) == SSLM_OK, "1.1 adopt");
	const std::vector<int32_t> t2 = rig.DecodeN(s, 32);
	o.tokens.insert(o.tokens.end(), t2.begin(), t2.end());
	const std::vector<uint8_t> saved = rig.Save(s);
	sslm_seq r = nullptr;
	CHECK_MSG(sslm_seq_restore(f.model, &rig.pool, saved.data(), saved.size(), &r) == SSLM_OK, "1.1 restore");
	if (r) rig.seqs.push_back(r);
	const std::vector<int32_t> t3 = r ? rig.DecodeN(r, 8) : std::vector<int32_t>{};
	o.tokens.insert(o.tokens.end(), t3.begin(), t3.end());
	o.blob = r ? rig.Save(r) : std::vector<uint8_t>{};
	o.ctr = r ? Ctr(r) : SaturationCounters{};
	sslm_prefix_release(prefix);
	if (hook) o.counts = hook->counts;
	return o;
}

void TestLifecycleCell11() {
	SeamScope scope(kSeam8K);
	const Fixture& f = GetFixture("fdef.sslm");
	if (!f.ok) return;
	const Outcome ref = LifecycleWorkload(f, nullptr);
	TestHook hook;
	const Outcome o = LifecycleWorkload(f, &hook);
	CHECK_MSG(hook.AllThreaded(), "1.1 preamble: run counts include a call below 2 tasks");
	CHECK_MSG(o.tokens.size() == 72 && o == ref, "1.1: tokens/counters/SSB5 differ from the no-hook run (%zu tokens)",
	          o.tokens.size());
}

// ---- 1.2: install, clear and reinstall between decode calls ----------------------------------------------

void TestInstallClearCell12() {
	SeamScope scope(kSeam8K);
	const Fixture& f = GetFixture("fdef.sslm");
	if (!f.ok) return;
	const std::vector<int32_t> prompt = Prompt(16);
	const Outcome ref = DecodeWorkload(f, prompt, 9, nullptr, 0);
	Rig rig(f);
	if (!rig.ok) return;
	sslm_seq s = rig.NewSeq();
	rig.PrefillAll(s, prompt, 16);
	TestHook hook;
	std::vector<int32_t> toks;
	for (int round = 0; round < 3; ++round) {
		rig.Install(&hook, SSLM_PARALLEL_FOR_MATVEC);
		hook.Clear();
		std::vector<int32_t> t = rig.DecodeN(s, 2);
		toks.insert(toks.end(), t.begin(), t.end());
		CHECK_MSG(!hook.counts.empty() && hook.counts.back() == 4, "1.2 round %d: installed hook not called", round);
		rig.Install(nullptr, 0);
		hook.Clear();
		t = rig.DecodeN(s, 1);
		toks.insert(toks.end(), t.begin(), t.end());
		CHECK_MSG(hook.counts.empty(), "1.2 round %d: %zu run calls while the hook was cleared", round,
		          hook.counts.size());
	}
	CHECK_MSG(toks == ref.tokens && rig.Save(s) == ref.blob, "1.2: tokens %s, want %s", Str(toks).c_str(),
	          Str(ref.tokens).c_str());
}

// ---- 2.2: the setters (S1, F3); the GPU halves are cell_setters.cpp (WIN32) -----------------------------

void TestSettersCell22() {
	const Fixture& f = GetFixture("fdef.sslm");
	if (!f.ok) return;
	Rig rig(f);
	if (!rig.ok) return;
	TestHook hook;
	struct Case {
		uint32_t reserved;
		int32_t max_tasks;
		bool accept;
	};
	const Case cases[] = {
	    {0u, 4, true},           {2u, 4, true},           {1u, 4, false},  // bit 0 rejected on a D1-only engine
	    {3u, 4, false},          {4u, 4, false},          {0x80000000u, 4, false}, {0xFFFFFFFFu, 4, false},
	    {2u, -1, false},         {2u, 257, false},        {2u, 0, true},   {2u, 1, true},
	    {2u, 256, true},
	};
	for (const Case& c : cases) {
		// Install a known-good hook first, so a rejection can be seen to change nothing.
		hook.max_tasks = 4;
		rig.Install(&hook, 0u);
		sslm_parallel_for pf{&TestHook::Run, &hook, c.max_tasks, c.reserved};
		const sslm_status st = sslm_workspace_set_parallel_for(rig.ws[0], &pf);
		CHECK_MSG((st == SSLM_OK) == c.accept, "2.2 reserved %#x max_tasks %d: status %d, want %s", c.reserved,
		          c.max_tasks, static_cast<int>(st), c.accept ? "accepted" : "SSLM_INVALID_ARGUMENT");
		if (!c.accept) CHECK_MSG(st == SSLM_INVALID_ARGUMENT, "2.2: rejection status %d", static_cast<int>(st));
	}
	// max_tasks 0 and 1 with bit 1: serial, zero `run` calls.
	SeamScope scope(kSeam8K);
	for (int32_t mt : {0, 1}) {
		sslm_seq s = rig.NewSeq();
		hook.max_tasks = mt;
		hook.Clear();
		rig.Install(&hook, SSLM_PARALLEL_FOR_MATVEC);
		rig.PrefillAll(s, Prompt(4), 1);
		rig.DecodeN(s, 3);
		CHECK_MSG(hook.counts.empty(), "2.2 max_tasks %d: %zu run calls, want 0", mt, hook.counts.size());
		rig.Release(s);
	}
	// A null `run` with max_tasks > 1 is rejected (cell_setters.cpp:87-90 through the shared function).
	const sslm_parallel_for null_run{nullptr, nullptr, 2, SSLM_PARALLEL_FOR_MATVEC};
	CHECK_MSG(sslm_workspace_set_parallel_for(rig.ws[0], &null_run) == SSLM_INVALID_ARGUMENT,
	          "2.2: a null run with max_tasks 2 must be rejected");
}

// ---- 2.3: run modes -------------------------------------------------------------------------------------

void TestRunModesCell23() {
	SeamScope scope(kSeam8K);
	const Fixture& f = GetFixture("fdef.sslm");
	if (!f.ok) return;
	const std::vector<int32_t> prompt = Prompt(16);
	const Outcome ref = DecodeWorkload(f, prompt, 16, nullptr, 0);
	for (RunMode m : {RunMode::kInline, RunMode::kReverse, RunMode::kThreads8}) {
		TestHook hook;
		hook.mode = m;
		const Outcome o = DecodeWorkload(f, prompt, 16, &hook, SSLM_PARALLEL_FOR_MATVEC);
		CHECK_MSG(hook.AllThreaded() && o.tokens.size() == 16 && o == ref, "2.3 mode %d differs from the no-hook run",
		          static_cast<int>(m));
	}
}

// ---- 3.1 / 3.2: one shared thread-safe hook on two workspaces, two sequences on two threads ---------------

void TestSharedHookCell31() {
	SeamScope scope(kSeam8K);
	const Fixture& f = GetFixture("fdef.sslm");
	if (!f.ok) return;
	const std::vector<int32_t> prompts[2] = {Prompt(16), Prompt(16, 59, 2)};
	Outcome ref[2];
	for (int i = 0; i < 2; ++i) ref[i] = DecodeWorkload(f, prompts[i], 24, nullptr, 0);
	Rig rig(f, /*workspaces=*/2);
	if (!rig.ok) return;
	ReferenceParallelFor pool(3);
	sslm_parallel_for pf = pool.Hook(4);
	pf.reserved = SSLM_PARALLEL_FOR_MATVEC;
	sslm_seq s[2] = {rig.NewSeq(), rig.NewSeq()};
	for (int i = 0; i < 2; ++i) {
		CHECK_MSG(sslm_workspace_set_parallel_for(rig.ws[i], &pf) == SSLM_OK, "3.1 install on workspace %d", i);
		rig.PrefillAll(s[i], prompts[i], 16, i);
	}
	// Each thread records its own tokens and statuses; checks run on this thread afterwards.
	std::vector<int32_t> toks[2];
	std::vector<sslm_status> sts[2];
	std::thread threads[2];
	for (int i = 0; i < 2; ++i) {
		threads[i] = std::thread([&, i] {
			for (int k = 0; k < 24; ++k) {
				int32_t t = -2;
				const sslm_status st = rig.Decode(&s[i], 1, static_cast<int32_t>(f.Layers()), &t, i);
				sts[i].push_back(st);
				if (st != SSLM_OK) break;
				toks[i].push_back(t);
			}
		});
	}
	for (std::thread& t : threads) t.join();
	for (int i = 0; i < 2; ++i) {
		CHECK_MSG(sts[i].size() == 24 && sts[i].back() == SSLM_OK && toks[i] == ref[i].tokens &&
		              rig.Save(s[i]) == ref[i].blob,
		          "3.1 sequence %d under the shared hook differs from its serial run", i);
		sslm_workspace_set_parallel_for(rig.ws[i], nullptr);
	}
}

// ---- 4.4: M is the admitted count (F5) ---------------------------------------------------------------

void TestAdmittedCountCell44() {
	SeamScope scope(kSeam8K);
	const Fixture& f = GetFixture("fdef.sslm");
	if (!f.ok) return;
	Rig rig(f);
	if (!rig.ok) return;
	TestHook hook;
	rig.Install(&hook, SSLM_PARALLEL_FOR_MATVEC);
	sslm_seq s = rig.NewSeq();
	const std::vector<int32_t> eight = Prompt(8);
	int32_t consumed = 0;
	sslm_status st = rig.Prefill(s, eight.data(), 8, 1, &consumed);
	CHECK_MSG(st == SSLM_OK && consumed == 1 && hook.counts == PrefillTokenCounts(kSeam8K),
	          "4.4 count 8, chunk_budget 1: status %d, consumed %d, counts %s", static_cast<int>(st), consumed,
	          Str(hook.counts).c_str());
	hook.Clear();
	const int32_t two_bad[2] = {eight[1], kBadToken};
	st = rig.Prefill(s, two_bad, 2, 2, &consumed);
	CHECK_MSG(st == SSLM_TOKEN_ID_OUT_OF_RANGE && consumed == 1 && hook.counts == PrefillTokenCounts(kSeam8K),
	          "4.4 count 2 with tokens[1] out of range: status %d, consumed %d, counts %s", static_cast<int>(st),
	          consumed, Str(hook.counts).c_str());
	for (int32_t n : {2, 7}) {
		hook.Clear();
		st = rig.Prefill(s, eight.data(), n, n, &consumed);
		CHECK_MSG(st == SSLM_OK && consumed == n && hook.counts.empty(),
		          "4.4 a %d-token admitted chunk: status %d, %zu run calls, want 0", n, static_cast<int>(st),
		          hook.counts.size());
	}
}

// ---- the F1 preamble and the guard cells (2.1, 5.1-5.5) --------------------------------------------------

// Which counters a fixture's guard cells check (F1): each must rise in the failed layer before the
// failure point, and rope_q and rope_k must rise by different amounts.
bool F1Holds(const SaturationCounters& d, bool qk) {
	const bool each = qk ? (d.kv > 0 && d.k_channel_landing > 0 && d.rope_q > 0)
	                     : (d.kv > 0 && d.kv_landing > 0 && d.rope_q > 0 && d.rope_k > 0);
	return each && d.rope_q != d.rope_k;
}

// The never-failed reference around one decode step, taken at layer budget 1 so the state between the
// two layers can be read. `step` counts full-token decode steps after the ready-for-logits one.
struct StepRef {
	bool ok = false;
	int step = -1;
	SaturationCounters before, after_l0, after;
	std::vector<int8_t> hidden_l0;
	superslm::CarriedScale scale_l0{};
	int64_t context_before = 0;
	int32_t token = -1;
	std::vector<uint8_t> blob_after;
};

std::vector<int32_t> GuardPrompt() { return Prompt(16); }

// Brings `s` to the start of decode step `step`: prompt prefilled, the ready-for-logits finish, then
// `step` full tokens. No hook is installed while it runs.
void PrepareAt(Rig& rig, sslm_seq s, int step, int w = 0) {
	rig.PrefillAll(s, GuardPrompt(), 16, w);
	rig.DecodeN(s, 1 + step, w);
}

// Scans decode steps 0..max_step (no hook) for the first whose layer-1 delta meets F1 and whose layer-0
// delta moved kv; returns that step's reference. With `must_find` false it only reports.
StepRef FindF1Step(const Fixture& f, bool qk, int max_step = 48, bool* any_found = nullptr) {
	StepRef r;
	Rig rig(f);
	if (!rig.ok) return r;
	sslm_seq s = rig.NewSeq();
	rig.PrefillAll(s, GuardPrompt(), 16);
	rig.DecodeN(s, 1);
	for (int step = 0; step <= max_step; ++step) {
		const SaturationCounters before = Ctr(s);
		const int64_t ctx = Live(s).context_length;
		int32_t tok = -2;
		sslm_status st = rig.Decode(&s, 1, 1, &tok);  // layer 0
		if (st != SSLM_OK || tok != -1 || Live(s).layer_index != 1) break;
		const SaturationCounters l0 = Ctr(s);
		const std::vector<int8_t> h0 = Hidden(s, f.Hidden());
		const superslm::CarriedScale sc0 = Live(s).hidden_scale;
		st = rig.Decode(&s, 1, 1, &tok);  // layer 1 and the finish
		if (st != SSLM_OK || tok < 0) break;
		const SaturationCounters after = Ctr(s);
		if (Minus(l0, before).kv > 0 && F1Holds(Minus(after, l0), qk)) {
			r.ok = true;
			r.step = step;
			r.before = before;
			r.after_l0 = l0;
			r.after = after;
			r.hidden_l0 = h0;
			r.scale_l0 = sc0;
			r.context_before = ctx;
			r.token = tok;
			r.blob_after = rig.Save(s);
			break;
		}
	}
	if (any_found) *any_found = r.ok;
	return r;
}

std::map<std::string, StepRef>& StepRefCache() {
	static std::map<std::string, StepRef> cache;
	return cache;
}
const StepRef& GetStepRef(const Fixture& f, bool qk) {
	auto& cache = StepRefCache();
	auto it = cache.find(f.name);
	if (it == cache.end()) it = cache.emplace(f.name, FindF1Step(f, qk)).first;
	CHECK_MSG(it->second.ok,
	          "F1 preamble: no decode step of %s moves every checked counter in layer 1 (and kv in layer 0) with "
	          "rope_q != rope_k; raise the generator's activation scale (§8)",
	          f.name.c_str());
	return it->second;
}

// Decode-call indices at 8 KiB, max_tasks 4: five groups per layer, then the finish.
enum DecodeSite { kSiteQ = 0, kSiteKV = 1, kSiteO = 2, kSiteGateUp = 3, kSiteDown = 4 };
int DecodeCallIndex(int layer, DecodeSite site) { return layer * 5 + site; }
constexpr int kFinishCall = 10;

// Checks `s` rests at the start of layer 1 of the reference step, with layer 0 kept.
void CheckRestsAtLayer1(const char* what, const Fixture& f, sslm_seq s, const StepRef& ref) {
	const superslm::SequenceLayerState& st = Live(s);
	CHECK_MSG(st.layer_index == 1 && Hidden(s, f.Hidden()) == ref.hidden_l0 && st.hidden_scale.m == ref.scale_l0.m &&
	              st.hidden_scale.e == ref.scale_l0.e && st.context_length == ref.context_before,
	          "%s: sequence does not rest at layer 1 of the step (layer_index %u, context %lld want %lld)", what,
	          st.layer_index, (long long)st.context_length, (long long)ref.context_before);
	CHECK_MSG(Ctr(s) == ref.after_l0, "%s: counters %s, want layer 0's kept and layer 1's restored %s", what,
	          Str(Ctr(s)).c_str(), Str(ref.after_l0).c_str());
}

// Retries the step with a correct hook and checks it equals the never-failed run.
void CheckRetry(const char* what, Rig& rig, sslm_seq s, TestHook& hook, const StepRef& ref) {
	hook.hostile = Hostile::kNone;
	int32_t tok = -2;
	const sslm_status st = rig.Decode(&s, 1, static_cast<int32_t>(rig.f.Layers()), &tok);
	CHECK_MSG(st == SSLM_OK && tok == ref.token, "%s retry: status %d token %d, want token %d", what,
	          static_cast<int>(st), tok, ref.token);
	CHECK_MSG(Ctr(s) == ref.after && rig.Save(s) == ref.blob_after,
	          "%s retry: counters %s (want %s) or the SSB5 blob differ from the never-failed run", what,
	          Str(Ctr(s)).c_str(), Str(ref.after).c_str());
}

// 2.1: a hostile `run` on the Nth call, landing on q, k + v, gate + up, down (layer 1) and the finish.
// Also on o: the coverage replica found o's failure arm (its `fail_layer`) taken by no cell, and §8 2.1
// names every other threaded site. Added after green, outside the plan's list (a deviation, recorded).
void TestHostileRunCell21() {
	SeamScope scope(kSeam8K);
	const Fixture& f = GetFixture("fdef.sslm");
	if (!f.ok) return;
	const StepRef& ref = GetStepRef(f, false);
	if (!ref.ok) return;
	struct Site {
		const char* name;
		int call;
	};
	const Site sites[] = {{"q", DecodeCallIndex(1, kSiteQ)},
	                      {"k+v", DecodeCallIndex(1, kSiteKV)},
	                      {"o", DecodeCallIndex(1, kSiteO)},
	                      {"gate+up", DecodeCallIndex(1, kSiteGateUp)},
	                      {"down", DecodeCallIndex(1, kSiteDown)},
	                      {"finish", kFinishCall}};
	int cases = 0;
	for (const Site& site : sites) {
		for (Hostile h : {Hostile::kDuplicate, Hostile::kOmit, Hostile::kMinusOne, Hostile::kTaskCount}) {
			char what[64];
			std::snprintf(what, sizeof(what), "2.1 %s at %s", HostileName(h), site.name);
			Rig rig(f);
			if (!rig.ok) return;
			sslm_seq s = rig.NewSeq();
			PrepareAt(rig, s, ref.step);
			CHECK_MSG(Ctr(s) == ref.before, "%s: prepared state differs from the reference", what);
			TestHook hook;
			hook.hostile = h;
			hook.hostile_call = site.call;
			rig.Install(&hook, SSLM_PARALLEL_FOR_MATVEC);
			int32_t tok = -2;
			const sslm_status st = rig.Decode(&s, 1, static_cast<int32_t>(f.Layers()), &tok);
			CHECK_MSG(st == SSLM_INVALID_ARGUMENT && static_cast<int>(hook.counts.size()) == site.call + 1,
			          "%s: status %d after %zu calls, want SSLM_INVALID_ARGUMENT on call %d", what,
			          static_cast<int>(st), hook.counts.size(), site.call);
			CHECK_MSG(hook.AllThreaded(), "%s preamble: a run call below 2 tasks", what);
			if (site.call == kFinishCall) {
				CHECK_MSG(Live(s).layer_index == 0 && Ctr(s) == ref.after &&
				              Live(s).context_length == ref.context_before + 1,
				          "%s: the finish's failure must keep every layer's counts", what);
			} else {
				CheckRestsAtLayer1(what, f, s, ref);
			}
			CheckRetry(what, rig, s, hook, ref);
			++cases;
		}
	}
	std::printf("decode-threading 2.1: %d hostile cases at decode step %d\n", cases, ref.step);
}

// 5.1: ParallelForIncomplete at down of layer 1 after layer 0 committed in the same call.
void TestDecodePfiMidLayerCell51() {
	SeamScope scope(kSeam8K);
	for (const char* file : {"fdef.sslm", "fqk.sslm"}) {
		const bool qk = std::strcmp(file, "fqk.sslm") == 0;
		const Fixture& f = GetFixture(file);
		if (!f.ok) continue;
		const StepRef& ref = GetStepRef(f, qk);
		if (!ref.ok) continue;
		char what[48];
		std::snprintf(what, sizeof(what), "5.1 %s", file);
		CHECK_MSG(F1Holds(Minus(ref.after, ref.after_l0), qk) && ref.after_l0 != ref.before,
		          "%s F1 preamble: layer 1 delta %s", what, Str(Minus(ref.after, ref.after_l0)).c_str());
		Rig rig(f);
		if (!rig.ok) return;
		sslm_seq s = rig.NewSeq();
		PrepareAt(rig, s, ref.step);
		TestHook hook;
		hook.hostile = Hostile::kOmit;
		hook.hostile_call = DecodeCallIndex(1, kSiteDown);
		rig.Install(&hook, SSLM_PARALLEL_FOR_MATVEC);
		int32_t tok = -2;
		const sslm_status st = rig.Decode(&s, 1, static_cast<int32_t>(f.Layers()), &tok);
		CHECK_MSG(st == SSLM_INVALID_ARGUMENT && hook.AllThreaded(), "%s: status %d", what, static_cast<int>(st));
		CheckRestsAtLayer1(what, f, s, ref);
		// The next call resumes at layer 1: layer 1's five groups, then the finish.
		hook.Clear();
		CheckRetry(what, rig, s, hook, ref);
		const std::vector<int32_t> want = std::vector<int32_t>(hook.counts.begin(), hook.counts.end());
		CHECK_MSG(hook.counts.size() == 6, "%s: the retry made %zu run calls, want 6 (layer 1 and the finish): %s",
		          what, hook.counts.size(), Str(want).c_str());
	}
}

// 5.2: a non-PFI rejection after landing (F2) keeps v1.9.0's counts.
void TestDecodeNonPfiCell52() {
	SeamScope scope(kSeam8K);
	const Fixture& f = GetFixture("fdef.sslm");
	if (!f.ok) return;
	const StepRef& ref = GetStepRef(f, false);
	if (!ref.ok) return;
	CHECK_MSG(F1Holds(Minus(ref.after, ref.after_l0), false), "5.2 F1 preamble");
	Rig rig(f);
	if (!rig.ok) return;
	sslm_seq s = rig.NewSeq();
	PrepareAt(rig, s, ref.step);
	TestHook hook;
	rig.Install(&hook, SSLM_PARALLEL_FOR_MATVEC);
	superslm::test::ArmLayerSiteFault(superslm::MatvecFaultSite::kDown, 1, superslm::MatvecFaultKind::kNonPfiStatus);
	int32_t tok = -2;
	const sslm_status st = rig.Decode(&s, 1, static_cast<int32_t>(f.Layers()), &tok);
	superslm::test::DisarmLayerSiteFault();
	CHECK_MSG(st == SSLM_ARTIFACT_REJECTED && hook.AllThreaded(), "5.2: status %d, want SSLM_ARTIFACT_REJECTED",
	          static_cast<int>(st));
	// Every counter moves before o and the MLP moves none, so a run in which layer 1 succeeds has these.
	CHECK_MSG(Ctr(s) == ref.after, "5.2: counters %s, want the unrestored %s", Str(Ctr(s)).c_str(),
	          Str(ref.after).c_str());
	CHECK_MSG(Live(s).layer_index == 1, "5.2: layer_index %u, want 1", Live(s).layer_index);
}

// 5.3: a one-token prefill that commits nothing restores the counters (the PFI and the exception arms).
void TestPrefillNothingCommittedCell53() {
	SeamScope scope(kSeam8K);
	const Fixture& f = GetFixture("fdef.sslm");
	if (!f.ok) return;
	const std::vector<int32_t> prompt = GuardPrompt();
	// Find a prompt position whose one-token prefill moves every checked counter (F1).
	int pos = -1;
	SaturationCounters before{}, after{};
	std::vector<uint8_t> blob_after;
	{
		Rig rig(f);
		if (!rig.ok) return;
		sslm_seq s = rig.NewSeq();
		for (size_t i = 0; i < prompt.size(); ++i) {
			const SaturationCounters b = Ctr(s);
			int32_t consumed = 0;
			rig.Prefill(s, &prompt[i], 1, 1, &consumed);
			if (i >= 1 && F1Holds(Minus(Ctr(s), b), false)) {
				pos = static_cast<int>(i);
				before = b;
				after = Ctr(s);
				blob_after = rig.Save(s);
				break;
			}
		}
	}
	CHECK_MSG(pos >= 0, "5.3 F1 preamble: no one-token prefill of the prompt moves every checked counter");
	if (pos < 0) return;
	struct Case {
		const char* name;
		bool pfi;
		uint32_t bits;
		sslm_status want;
	};
	const Case cases[] = {{"ParallelForIncomplete", true, SSLM_PARALLEL_FOR_MATVEC, SSLM_INVALID_ARGUMENT},
	                      {"bad_alloc, bit 1", false, SSLM_PARALLEL_FOR_MATVEC, SSLM_ALLOCATION_FAILED},
	                      {"bad_alloc, reserved 0", false, 0u, SSLM_ALLOCATION_FAILED}};
	for (const Case& c : cases) {
		Rig rig(f);
		if (!rig.ok) return;
		sslm_seq s = rig.NewSeq();
		const std::vector<int32_t> head(prompt.begin(), prompt.begin() + pos);
		rig.PrefillAll(s, head, 1);
		CHECK_MSG(Ctr(s) == before, "5.3 %s: prepared counters differ", c.name);
		TestHook hook;
		rig.Install(&hook, c.bits);
		if (c.pfi) {
			hook.hostile = Hostile::kDuplicate;
			hook.hostile_call = 13;  // down of layer 1: seven sites per layer
		} else {
			superslm::test::ArmLayerSiteFault(superslm::MatvecFaultSite::kDown, 1,
			                                  superslm::MatvecFaultKind::kBadAlloc);
		}
		const int64_t ctx = Live(s).context_length;
		int32_t consumed = -1;
		const sslm_status st = rig.Prefill(s, &prompt[pos], 1, 1, &consumed);
		superslm::test::DisarmLayerSiteFault();
		CHECK_MSG(st == c.want && consumed == 0, "5.3 %s: status %d consumed %d, want %d", c.name,
		          static_cast<int>(st), consumed, static_cast<int>(c.want));
		if (c.pfi) CHECK_MSG(hook.AllThreaded() && hook.counts.size() == 14, "5.3 %s preamble: counts %s", c.name,
		                     Str(hook.counts).c_str());
		CHECK_MSG(Ctr(s) == before && Live(s).context_length == ctx,
		          "5.3 %s: counters %s context %lld, want the pre-call %s context %lld", c.name, Str(Ctr(s)).c_str(),
		          (long long)Live(s).context_length, Str(before).c_str(), (long long)ctx);
		hook.hostile = Hostile::kNone;
		const sslm_status rst = rig.Prefill(s, &prompt[pos], 1, 1, &consumed);
		CHECK_MSG(rst == SSLM_OK && consumed == 1 && Ctr(s) == after && rig.Save(s) == blob_after,
		          "5.3 %s: retry differs from the never-failed run", c.name);
	}
}

// 5.4: a committed partial admission keeps its counts (F6(a)), bit 1 off and on.
void TestPrefillPartialAdmissionCell54() {
	SeamScope scope(kSeam8K);
	const Fixture& f = GetFixture("fdef.sslm");
	if (!f.ok) return;
	const std::vector<int32_t> prompt = GuardPrompt();
	const std::vector<int32_t> head(prompt.begin(), prompt.begin() + 8);
	struct Case {
		int32_t sent;
		int32_t admitted;
	};
	for (const Case c : {Case{2, 1}, Case{8, 3}}) {
		std::vector<int32_t> call(prompt.begin() + 8, prompt.begin() + 8 + c.admitted);
		while (static_cast<int32_t>(call.size()) < c.sent) call.push_back(kBadToken);
		// The clean prefill of the admitted prefix alone (v1.9.0's value).
		Outcome clean;
		SaturationCounters pre{};
		{
			Rig rig(f);
			if (!rig.ok) return;
			sslm_seq s = rig.NewSeq();
			rig.PrefillAll(s, head, 8);
			pre = Ctr(s);
			int32_t consumed = 0;
			rig.Prefill(s, call.data(), c.admitted, c.admitted, &consumed);
			clean.blob = rig.Save(s);
			clean.ctr = Ctr(s);
		}
		CHECK_MSG(clean.ctr.kv > pre.kv, "5.4 %d/%d preamble: the admitted prefix does not move the counters", c.sent,
		          c.admitted);
		for (uint32_t bits : {0u, SSLM_PARALLEL_FOR_MATVEC}) {
			Rig rig(f);
			if (!rig.ok) return;
			sslm_seq s = rig.NewSeq();
			rig.PrefillAll(s, head, 8);
			TestHook hook;
			rig.Install(&hook, bits);
			int32_t consumed = 0;
			const sslm_status st = rig.Prefill(s, call.data(), c.sent, c.sent, &consumed);
			CHECK_MSG(st == SSLM_TOKEN_ID_OUT_OF_RANGE && consumed == c.admitted,
			          "5.4 %d sent: status %d consumed %d", c.sent, static_cast<int>(st), consumed);
			if (bits && c.admitted == 1)
				CHECK_MSG(hook.AllThreaded() && hook.counts.size() == 14, "5.4 preamble: one-token call not threaded");
			CHECK_MSG(Ctr(s) == clean.ctr && rig.Save(s) == clean.blob,
			          "5.4 %d sent %d admitted bits %u: counters %s, want the clean prefix's %s", c.sent, c.admitted,
			          bits, Str(Ctr(s)).c_str(), Str(clean.ctr).c_str());
		}
	}
}

// 5.5: a multi-sequence decode call whose second sequence fails.
void TestMultiSequenceCell55() {
	SeamScope scope(kSeam8K);
	const Fixture& f = GetFixture("fdef.sslm");
	if (!f.ok) return;
	const StepRef& ref = GetStepRef(f, false);
	if (!ref.ok) return;
	Rig rig(f);
	if (!rig.ok) return;
	sslm_seq s[2] = {rig.NewSeq(), rig.NewSeq()};
	PrepareAt(rig, s[0], ref.step);
	PrepareAt(rig, s[1], ref.step);
	TestHook hook;
	hook.hostile = Hostile::kOmit;
	hook.hostile_call = 11 + DecodeCallIndex(1, kSiteDown);  // the second sequence's down of layer 1
	rig.Install(&hook, SSLM_PARALLEL_FOR_MATVEC);
	int32_t toks[2] = {-2, -2};
	const sslm_status st = rig.Decode(s, 2, static_cast<int32_t>(f.Layers()), toks);
	CHECK_MSG(st == SSLM_INVALID_ARGUMENT && static_cast<int>(hook.counts.size()) == hook.hostile_call + 1,
	          "5.5: status %d after %zu calls", static_cast<int>(st), hook.counts.size());
	CHECK_MSG(Ctr(s[0]) == ref.after && rig.Save(s[0]) == ref.blob_after,
	          "5.5: the first sequence must keep its advance");
	CheckRestsAtLayer1("5.5 second sequence", f, s[1], ref);
	CheckRetry("5.5 second sequence", rig, s[1], hook, ref);
}

// ---- 8.1: composition -------------------------------------------------------------------------------------

void TestCompositionCell81() {
	SeamScope scope(kSeam8K);
	const Fixture& f = GetFixture("fdef.sslm");
	if (!f.ok) return;
	const std::vector<int32_t> prompt = Prompt(16);
	// (a) A LoRA adapter bound (a synthetic rank-1 adapter on q, o, gate, up and down, identity folds),
	// through the C++ layer loop: the delta GEMMs stay serial inside the funnel.
	if (f.cpp_ok) {
		const size_t h = f.Hidden(), q = f.QWidth(), inter = f.Inter();
		static const int32_t kFold[3] = {1, 0, 0};
		std::vector<int32_t> delta_triples(3 * std::max(q, inter), 0);
		for (size_t i = 0; i < delta_triples.size(); i += 3) delta_triples[i] = 1;
		const superslm::SslmAmplifyingFoldEntry delta_q{{}, reinterpret_cast<const uint8_t*>(delta_triples.data()), q};
		const superslm::SslmAmplifyingFoldEntry delta_h{{}, reinterpret_cast<const uint8_t*>(delta_triples.data()), h};
		const superslm::SslmAmplifyingFoldEntry delta_i{
		    {}, reinterpret_cast<const uint8_t*>(delta_triples.data()), inter};
		const superslm::SslmAmplifyingFoldEntry u_entry{{}, reinterpret_cast<const uint8_t*>(kFold), 1};
		auto pattern = [](size_t n, int mul) {
			std::vector<int8_t> v(n);
			for (size_t i = 0; i < n; ++i) v[i] = static_cast<int8_t>(static_cast<int>((i * mul + 3) % 61) - 30);
			return v;
		};
		const std::vector<int8_t> a_h = pattern(h, 7), a_q = pattern(q, 11), a_i = pattern(inter, 13);
		const std::vector<int8_t> b_q = pattern(q, 5), b_h = pattern(h, 17), b_i = pattern(inter, 19);
		superslm::LayerAdapter adapter;
		adapter.rank = 1;
		adapter.q = {a_h.data(), b_q.data(), &delta_q, &u_entry};
		adapter.o = {a_q.data(), b_h.data(), &delta_h, &u_entry};
		adapter.gate = {a_h.data(), b_i.data(), &delta_i, &u_entry};
		adapter.up = {a_h.data(), b_i.data(), &delta_i, &u_entry};
		adapter.down = {a_i.data(), b_h.data(), &delta_h, &u_entry};
		std::vector<superslm::LayerWeights> adapted = f.layers;
		for (auto& lw : adapted) lw.adapter = &adapter;
		const std::vector<int32_t> stream = Concat(prompt, Prompt(8, 13, 1));
		const CppRun base = RunCpp(f, stream, nullptr, false);
		const CppRun serial = RunCpp(f, stream, nullptr, false, adapted.data());
		TestHook hook;
		const sslm_parallel_for pf = hook.Pf(SSLM_PARALLEL_FOR_MATVEC);
		const CppRun threaded = RunCpp(f, stream, &pf, false, adapted.data());
		CHECK_MSG(serial.status == SslmForwardStatus::Ok && serial.trace != base.trace,
		          "8.1 LoRA: the adapter must run (status %s) and change the trace",
		          superslm::SslmForwardStatusName(serial.status));
		CHECK_MSG(hook.AllThreaded() && SameRun(serial, threaded), "8.1 LoRA: threaded differs from serial");
	}
	// (b) Damped-greedy mode, alone and with the fixture's schema bound.
	for (int schema = 0; schema < 2; ++schema) {
		Outcome o[2];
		std::vector<sslm_status> sts[2];
		for (int threaded = 0; threaded < 2; ++threaded) {
			Rig rig(f);
			if (!rig.ok) return;
			TestHook hook;
			sslm_seq s = rig.NewSeq();
			if (schema) {
				sslm_schema sc = nullptr;
				CHECK_MSG(sslm_schema_lookup(f.model, "g5_minimal_one_field", &sc) == SSLM_OK &&
				              sslm_seq_set_schema(s, sc) == SSLM_OK,
				          "8.1: binding the fixture's schema");
			}
			rig.PrefillAll(s, prompt, 16);
			rig.Install(threaded ? &hook : nullptr, SSLM_PARALLEL_FOR_MATVEC);
			o[threaded].tokens = rig.DecodeN(s, schema ? 4 : 16, 0, SSLM_DECODE_MODE_DAMPED_GREEDY, &sts[threaded]);
			o[threaded].blob = rig.Save(s);
			o[threaded].ctr = Ctr(s);
			if (threaded) CHECK_MSG(hook.AllThreaded() && hook.counts.size() > 1, "8.1 damped preamble");
		}
		CHECK_MSG(!sts[0].empty() && sts[0][0] == SSLM_OK && sts[0] == sts[1] && o[0] == o[1],
		          "8.1 damped greedy%s: threaded differs from no-hook (tokens %s vs %s)", schema ? " + schema" : "",
		          Str(o[1].tokens).c_str(), Str(o[0].tokens).c_str());
	}
	// (c) Option-G fused K landing, through the C++ Option-G overload.
	if (f.cpp_ok) {
		const std::vector<int32_t> stream = Concat(prompt, Prompt(8, 13, 1));
		const CppRun serial = RunCpp(f, stream, nullptr, true);
		TestHook hook;
		const sslm_parallel_for pf = hook.Pf(SSLM_PARALLEL_FOR_MATVEC);
		const CppRun threaded = RunCpp(f, stream, &pf, true);
		CHECK_MSG(serial.status == SslmForwardStatus::Ok, "8.1 Option-G: serial status %s",
		          superslm::SslmForwardStatusName(serial.status));
		CHECK_MSG(hook.AllThreaded() && SameRun(serial, threaded), "8.1 Option-G: threaded differs from serial");
	}
	// (d) A prefix adopted: 1.1's workload adopts one; here decode straight after adoption.
	{
		Outcome o[2];
		for (int threaded = 0; threaded < 2; ++threaded) {
			Rig rig(f);
			if (!rig.ok) return;
			TestHook hook;
			rig.Install(threaded ? &hook : nullptr, SSLM_PARALLEL_FOR_MATVEC);
			sslm_prefix prefix = nullptr;
			CHECK_MSG(sslm_prefix_begin(f.model, &rig.pool, &prefix) == SSLM_OK, "8.1 prefix_begin");
			if (!prefix) return;
			int32_t consumed = 0;
			sslm_prefix_prefill(f.model, prefix, prompt.data(), 16, 16, SSLM_SPAN_PROMPT, rig.ws[0], &consumed);
			sslm_prefix_freeze(prefix);
			sslm_seq s = rig.NewSeq();
			CHECK_MSG(sslm_seq_adopt_prefix(s, prefix) == SSLM_OK, "8.1 adopt");
			const int32_t more[1] = {prompt[3]};
			rig.Prefill(s, more, 1, 1, &consumed);
			o[threaded].tokens = rig.DecodeN(s, 12);
			o[threaded].blob = rig.Save(s);
			o[threaded].ctr = Ctr(s);
			sslm_prefix_release(prefix);
			if (threaded) CHECK_MSG(hook.AllThreaded() && hook.counts.size() > 12, "8.1 prefix preamble");
		}
		CHECK_MSG(o[0].tokens.size() == 12 && o[0] == o[1], "8.1 prefix adopted: threaded differs from no-hook");
	}
}

// ---- 8.2: each bit alone (F7), the D1-only half ---------------------------------------------------------

void TestEachBitAloneCell82() {
	SeamScope scope(kSeam8K);
	const Fixture& f = GetFixture("fdef.sslm");
	if (!f.ok) return;
	Rig rig(f);
	if (!rig.ok) return;
	TestHook hook;
	sslm_parallel_for pf = hook.Pf(1u);
	CHECK_MSG(sslm_workspace_set_parallel_for(rig.ws[0], &pf) == SSLM_INVALID_ARGUMENT,
	          "8.2: bit 0 must be rejected on a D1-only engine");
	rig.Install(&hook, SSLM_PARALLEL_FOR_MATVEC);
	sslm_seq s = rig.NewSeq();
	const std::vector<int32_t> eight = Prompt(8);
	rig.PrefillAll(s, eight, 8);
	CHECK_MSG(hook.counts.empty(), "8.2 bit 1 alone: an 8-token prefill made %zu run calls", hook.counts.size());
	int32_t tok = 77, consumed = 0;
	rig.Prefill(s, &tok, 1, 1, &consumed);
	CHECK_MSG(hook.counts == PrefillTokenCounts(kSeam8K), "8.2 bit 1 alone: a one-token call made %s",
	          Str(hook.counts).c_str());
}

// ---- 11.1: guard vitality -------------------------------------------------------------------------------

void TestVitalityCell111() {
	// The F1 preamble fires: on the no-clamp variant no decode step can satisfy it, so 5.1's
	// precondition would turn that cell red rather than let it pass on counters that never moved.
	const Fixture& f = GetFixture("fnoclamp.sslm");
	if (!f.ok) return;
	bool found = true;
	const StepRef r = FindF1Step(f, false, 48, &found);
	CHECK_MSG(!found, "11.1: the F1 preamble holds on fnoclamp at step %d; it cannot detect a fixture that never "
	          "clamps", r.step);
	// And the counting hook counts: the same hook reads the expected nonzero counts on a threaded workload.
	const Fixture& d = GetFixture("fdef.sslm");
	if (!d.ok) return;
	SeamScope scope(kSeam8K);
	TestHook hook;
	DecodeWorkload(d, Prompt(16), 2, &hook, SSLM_PARALLEL_FOR_MATVEC);
	CHECK_MSG(hook.counts == Concat({4}, DecodeTokenCounts(kSeam8K)), "11.1: the counting hook read %s",
	          Str(hook.counts).c_str());
}

}  // namespace

void RunDecodeThreadingD1Cells(int& checks, int& failures) {
	GChecks = 0;
	GFailures = 0;
	TestRuleCell42();
	TestSeamBetweenCallsCell13();
	TestAccAgainstScalarRefCell61();
	TestStraddlesCell43();
	TestWholeDecodeCell62();
	TestOneTokenPrefillCell63();
	TestOptInCell72();
	TestLifecycleCell11();
	TestInstallClearCell12();
	TestSettersCell22();
	TestRunModesCell23();
	TestSharedHookCell31();
	TestAdmittedCountCell44();
	TestHostileRunCell21();
	TestDecodePfiMidLayerCell51();
	TestDecodeNonPfiCell52();
	TestPrefillNothingCommittedCell53();
	TestPrefillPartialAdmissionCell54();
	TestMultiSequenceCell55();
	TestCompositionCell81();
	TestEachBitAloneCell82();
	TestVitalityCell111();
	FixtureCache().clear();
	std::printf("decode-threading D1 cells: %d checks, %d failures\n", GChecks, GFailures);
	checks += GChecks;
	failures += GFailures;
}

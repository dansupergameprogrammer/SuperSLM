// Tiled-matmul plan (rev 11.5), step S2-A: the cells of the plan's coverage model (§9) that thread the
// batched (M >= 2) projections across the host's parallel-for hook, bit 0 (SSLM_PARALLEL_FOR_PREFILL).
// Cell numbers are the plan's where one exists.
//
// Own translation unit because it calls the C ABI (test_slm18x_saturation_census.cpp explains why
// test_main.cpp cannot include sslm_abi.h on Windows). test_main.cpp and the scalar-forced standalone
// executable (decode_threading_standalone_main.cpp) call RunBatchedThreadingCells.
//
// The fixtures are the decode-threading plan's (tools/gen_decode_threading_fixture.py, into the
// directory SUPERSLM_DECODE_THREADING_FIXTURE_DIR names). F-DEF's projections are far below the
// production minimum work per task, so every ABI cell runs under the minimum-MACs seam; the pure rule
// cells read the production constant. A missing fixture fails every cell that needs it.
//
// "Reference" below is the same workload with no hook (the serial path) or GemmInt8Accumulate, never
// the build under test's split. Every threaded cell first asserts that `run` was called with two or
// more tasks.

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <new>
#include <string>
#include <thread>
#include <vector>

#include "superslm/forward_sites.h"
#include "superslm/layer_marshal.h"
#include "superslm/matmul.h"
#include "superslm/parallel_for.h"
#include "superslm/sslm_abi.h"
#include "../src/forward/parallel_split.h"
#include "support/bad_alloc_injection.h"

#if !defined(SUPERSLM_ENABLE_MATVEC_TEST_SEAMS)
#error "test_batched_threading.cpp needs a library built with SUPERSLM_ENABLE_MATVEC_TEST_SEAMS"
#endif

extern "C" superslm::SequenceLayerState* SslmSeqLiveStateForTest(sslm_seq);

// Atomic: cell 3.2 checks from two threads at once.
static std::atomic<int> GChecks{0};
static std::atomic<int> GFailures{0};

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

bool SameCounters(const SaturationCounters& a, const SaturationCounters& b) {
	return a.kv == b.kv && a.kv_landing == b.kv_landing && a.k_channel_landing == b.k_channel_landing &&
	       a.rope_q == b.rope_q && a.rope_k == b.rope_k;
}

std::string Str(const std::vector<int32_t>& v) {
	std::string s = "{";
	for (size_t i = 0; i < v.size(); ++i) s += (i ? "," : "") + std::to_string(v[i]);
	return s + "}";
}

std::vector<int32_t> Prompt(size_t n, int mul = 37, int add = 11) {
	std::vector<int32_t> p;
	for (size_t i = 0; i < n; ++i) p.push_back(static_cast<int32_t>((i * mul + add) % 256));
	return p;
}

// The minimum-MACs seam, set for one scope.
struct MacsScope {
	explicit MacsScope(size_t macs) { superslm::test::SetMinMacsPerTask(macs); }
	~MacsScope() { superslm::test::ResetMinMacsPerTask(); }
	MacsScope(const MacsScope&) = delete;
	MacsScope& operator=(const MacsScope&) = delete;
};

// ---- the fixture -----------------------------------------------------------------------------------

struct Fixture {
	bool ok = false;
	uint8_t* bytes = nullptr;
	size_t size = 0;
	sslm_model model = nullptr;
	uint32_t layers = 0;
	~Fixture() {
		if (model) sslm_model_unmap(model);
		if (bytes) ::operator delete(bytes, std::align_val_t(SSLM_ABI_ALIGNMENT_BYTES));
	}
};

std::unique_ptr<Fixture>& FixtureSlot() {
	static std::unique_ptr<Fixture> f;
	return f;
}

const Fixture& GetFdef() {
	std::unique_ptr<Fixture>& slot = FixtureSlot();
	if (slot) {
		CHECK_MSG(slot->ok, "fdef.sslm did not load (see the first failure)");
		return *slot;
	}
	slot = std::make_unique<Fixture>();
	Fixture& f = *slot;
	const char* dir = std::getenv("SUPERSLM_DECODE_THREADING_FIXTURE_DIR");
	CHECK_MSG(dir != nullptr && *dir != '\0',
	          "SUPERSLM_DECODE_THREADING_FIXTURE_DIR is not set; run tools/gen_decode_threading_fixture.py");
	if (dir == nullptr || *dir == '\0') return f;
	const std::string path = std::string(dir) + "/fdef.sslm";
	std::vector<uint8_t> raw;
	const bool read = superslm_marshal::ReadFile(path.c_str(), raw) && !raw.empty();
	CHECK_MSG(read, "fixture %s not found or empty", path.c_str());
	if (!read) return f;
	f.size = raw.size();
	f.bytes = static_cast<uint8_t*>(::operator new(raw.size(), std::align_val_t(SSLM_ABI_ALIGNMENT_BYTES)));
	std::memcpy(f.bytes, raw.data(), raw.size());
	const sslm_status ms = sslm_model_map(f.bytes, f.size, &f.model);
	CHECK_MSG(ms == SSLM_OK, "sslm_model_map(fdef) == %d", static_cast<int>(ms));
	f.layers = 2;  // F-DEF (tools/gen_decode_threading_fixture.py)
	f.ok = ms == SSLM_OK;
	return f;
}

// ---- the test hook ---------------------------------------------------------------------------------

enum class RunMode { kInline, kReverse, kThreads };
enum class Hostile { kNone, kDuplicate, kOmit, kMinusOne, kTaskCount };

// Counts every `run` call and the task count it carried, in call order; optionally breaks exactly-once
// on one call (0-based since the last Clear).
struct TestHook {
	RunMode mode = RunMode::kInline;
	int threads = 4;
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
		case RunMode::kThreads: {
			// The calling thread takes a share too (parallel_for.h allows it).
			std::vector<std::thread> pool;
			const int n = h.threads;
			for (int t = 1; t < n; ++t) {
				pool.emplace_back([=] {
					for (int32_t i = t; i < task_count; i += n) task(task_ctx, i);
				});
			}
			for (int32_t i = 0; i < task_count; i += n) task(task_ctx, i);
			for (std::thread& th : pool) th.join();
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
	void* ws_buf = nullptr;
	sslm_workspace ws = nullptr;
	std::vector<sslm_seq> seqs;
	bool ok = false;

	explicit Rig(const Fixture& fixture, uint32_t blocks = 4) : f(fixture) {
		if (!f.ok) return;
		const size_t size = blocks * sslm_kv_block_size(f.model) + sslm_kv_pool_overhead_size(f.model, blocks);
		pool_buf = ::operator new(size, std::align_val_t(SSLM_ABI_ALIGNMENT_BYTES));
		const sslm_status ps = sslm_kv_pool_create(f.model, pool_buf, size, blocks, &pool);
		CHECK_MSG(ps == SSLM_OK, "sslm_kv_pool_create == %d", static_cast<int>(ps));
		const sslm_config cfg{2, 64, static_cast<int32_t>(f.layers), 0u};
		const size_t wsz = sslm_workspace_size(f.model, &cfg);
		ws_buf = ::operator new(wsz, std::align_val_t(SSLM_ABI_ALIGNMENT_BYTES));
		const sslm_status wst = sslm_workspace_create(f.model, &cfg, ws_buf, wsz, &ws);
		CHECK_MSG(wst == SSLM_OK, "sslm_workspace_create == %d", static_cast<int>(wst));
		ok = ps == SSLM_OK && wst == SSLM_OK;
	}
	~Rig() {
		for (sslm_seq s : seqs) sslm_seq_release(s);
		if (ws) sslm_workspace_destroy(ws);
		if (ws_buf) ::operator delete(ws_buf, std::align_val_t(SSLM_ABI_ALIGNMENT_BYTES));
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
	void Install(TestHook* hook, uint32_t bits) {
		if (hook == nullptr) {
			CHECK_MSG(sslm_workspace_set_parallel_for(ws, nullptr) == SSLM_OK, "clearing the hook");
			return;
		}
		const sslm_parallel_for pf = hook->Pf(bits);
		const sslm_status st = sslm_workspace_set_parallel_for(ws, &pf);
		CHECK_MSG(st == SSLM_OK, "installing a hook with reserved %#x, max_tasks %d == %d", bits, pf.max_tasks,
		          static_cast<int>(st));
	}
	sslm_status Prefill(sslm_seq s, const int32_t* tokens, int32_t count, int32_t* consumed) {
		int32_t c = 0;
		const sslm_status st = sslm_prefill(f.model, s, tokens, count, count, SSLM_SPAN_PROMPT, ws, &c);
		if (consumed) *consumed = c;
		return st;
	}
	// Whole-prompt prefill in calls of up to `chunk` tokens; checks every call.
	void PrefillAll(sslm_seq s, const std::vector<int32_t>& prompt, size_t chunk) {
		for (size_t at = 0; at < prompt.size();) {
			const int32_t n = static_cast<int32_t>(std::min(chunk, prompt.size() - at));
			int32_t consumed = 0;
			const sslm_status st = Prefill(s, prompt.data() + at, n, &consumed);
			CHECK_MSG(st == SSLM_OK && consumed == n, "prefill at %zu: status %d, consumed %d of %d", at,
			          static_cast<int>(st), consumed, n);
			if (st != SSLM_OK || consumed != n) return;
			at += static_cast<size_t>(n);
		}
	}
	std::vector<int32_t> DecodeN(sslm_seq s, int n) {
		std::vector<int32_t> toks;
		sslm_decode_params p{};
		if (sslm_decode_params_init(f.model, SSLM_DECODE_MODE_GREEDY, static_cast<int32_t>(f.layers), &p) != SSLM_OK)
			return toks;
		for (int i = 0; i < n; ++i) {
			int32_t t = -2;
			const sslm_status st = sslm_decode_step_v2(f.model, &s, 1, &p, ws, &t);
			CHECK_MSG(st == SSLM_OK, "decode step %d == %d", i, static_cast<int>(st));
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

SaturationCounters Ctr(sslm_seq s) { return SaturationCounters::Snapshot(*SslmSeqLiveStateForTest(s)); }
int64_t ContextLength(sslm_seq s) { return SslmSeqLiveStateForTest(s)->context_length; }

// One prefill pass of two or more rows on F-DEF at max_tasks 4 with every projection above the minimum
// work: q, k, v, o, gate, up, down per layer. T = min(4, ceil(N / 32)) (§5.2), and the 32-aligned block
// of ceil(N / T) columns then sets the count: N = 192 (q, o, down) gives blocks of 64, so 3 tasks;
// N = 96 (k, v) blocks of 32, 3 tasks; N = 512 (gate, up) blocks of 128, 4 tasks.
std::vector<int32_t> BatchedPassCounts(uint32_t layers) {
	const std::vector<int32_t> layer = {3, 3, 3, 3, 4, 4, 3};
	std::vector<int32_t> out;
	for (uint32_t l = 0; l < layers; ++l) out.insert(out.end(), layer.begin(), layer.end());
	return out;
}

// What a prompt prefilled in `chunk`-token calls, then `decode` greedy tokens, leaves: the blob after the
// prefill, the decoded tokens, and the blob after them.
struct Workload {
	std::vector<uint8_t> blob_prefill;
	std::vector<int32_t> tokens;
	std::vector<uint8_t> blob_end;
	bool operator==(const Workload& o) const {
		return blob_prefill == o.blob_prefill && tokens == o.tokens && blob_end == o.blob_end;
	}
};

Workload RunWorkload(const Fixture& f, const std::vector<int32_t>& prompt, size_t chunk, int decode,
                     TestHook* hook, uint32_t bits) {
	Workload w;
	Rig rig(f);
	if (!rig.ok) return w;
	rig.Install(hook, bits);
	sslm_seq s = rig.NewSeq();
	if (!s) return w;
	rig.PrefillAll(s, prompt, chunk);
	w.blob_prefill = rig.Save(s);
	w.tokens = rig.DecodeN(s, decode);
	w.blob_end = rig.Save(s);
	return w;
}

// ---- 2.3: the §5.2 rule, as a pure function ---------------------------------------------------------

void TestRuleCell23() {
	TestHook hook;
	sslm_parallel_for pf = hook.Pf(SSLM_PARALLEL_FOR_PREFILL);
	CHECK_MSG(superslm::MinMacsPerTask() == superslm::kMinMacsPerTask && superslm::kMinMacsPerTask == size_t{4} << 20,
	          "2.3: the production minimum (4 Mi MACs) is not in effect");
	// A-EX (plan §5.4's table) at M = 8, max_tasks 4: q and o serial, k and v serial, gate, up, down 4.
	struct Case {
		size_t m, k, n;
		int32_t max_tasks;
		size_t want;
	};
	// `want` below 2 means serial (any count below 2); otherwise the exact task count.
	const Case cases[] = {
	    {8, 896, 896, 4, 0},     {8, 896, 128, 4, 0},    {8, 896, 4864, 4, 4},  {8, 4864, 896, 4, 4},
	    {2, 896, 4864, 4, 2},    {2, 4864, 896, 4, 2},   {2, 896, 896, 4, 0},   // the exact check pass's 2 rows
	    {1, 896, 4864, 4, 0},                                                    // one row: never this rule
	    {64, 896, 896, 4, 4},    {64, 896, 896, 256, 10},  // T = 12 by the work floor, blocks of 96 -> 10
	    {64, 896, 896, 1, 0},    {64, 896, 896, 0, 0},
	    {64, 8960, 32, 4, 0},    // N <= 32: one block, serial whatever the work
	    {64, 8960, 33, 4, 2},    // two blocks of 32
	};
	for (const Case& c : cases) {
		pf.max_tasks = c.max_tasks;
		const ColumnSplit s = superslm::BatchedGemmSplit(&pf, c.m, c.k, c.n);
		const bool ok = c.want >= 2 ? s.task_count == c.want : s.task_count < 2;
		CHECK_MSG(ok, "2.3 m %zu k %zu n %zu max_tasks %d: %zu tasks, want %s%zu", c.m, c.k, c.n, c.max_tasks,
		          s.task_count, c.want >= 2 ? "" : "serial, not ", c.want >= 2 ? c.want : size_t{2});
		if (s.task_count >= 2) {
			CHECK_MSG(s.rows_per_task % 32 == 0, "2.3: blocks of %zu columns are not 32-aligned", s.rows_per_task);
		}
	}
	// No hook, no `run`, max_tasks < 2: serial.
	CHECK_MSG(superslm::BatchedGemmSplit(nullptr, 64, 896, 4864).task_count < 2, "2.3: null hook threads");
	sslm_parallel_for no_run{nullptr, nullptr, 4, SSLM_PARALLEL_FOR_PREFILL};
	CHECK_MSG(superslm::BatchedGemmSplit(&no_run, 64, 896, 4864).task_count < 2, "2.3: null run threads");
	// GemmDispatch with only `row` set never takes the column split at M >= 2.
	{
		MacsScope scope(1);
		hook.Clear();
		pf.max_tasks = 4;
		std::vector<int8_t> a(8 * 64, 3), w(128 * 64, 5);
		std::vector<int64_t> out(8 * 128);
		GemmThreading rowonly{&pf, nullptr};
		CHECK_MSG(superslm::GemmDispatch(rowonly, a.data(), w.data(), 8, 64, 128, out.data()) == SslmForwardStatus::Ok &&
		              hook.counts.empty(),
		          "2.3: a row-only threading made %zu run calls at M = 8", hook.counts.size());
		GemmThreading batched{nullptr, &pf};
		CHECK_MSG(superslm::GemmDispatch(batched, a.data(), w.data(), 1, 64, 128, out.data()) == SslmForwardStatus::Ok &&
		              hook.counts.empty(),
		          "2.3: a batched-only threading made %zu run calls at M = 1", hook.counts.size());
		CHECK_MSG(superslm::GemmDispatch(batched, a.data(), w.data(), 2, 64, 128, out.data()) == SslmForwardStatus::Ok &&
		              hook.counts == std::vector<int32_t>{4},
		          "2.3: a batched threading at M = 2 made %s, want {4}", Str(hook.counts).c_str());
	}
}

// ---- 3.4: the split covers every column once, in 32-aligned blocks -----------------------------------

void TestSplitCell34() {
	size_t bad = 0;
	for (size_t n = 1; n <= 600; ++n) {
		for (size_t tasks : {2u, 3u, 4u, 7u, 16u, 256u}) {
			for (size_t work : {size_t{0}, n * 8, n * 1000000}) {
				const ColumnSplit s = superslm::SplitColumns(n, work, tasks, 32, 1000);
				if (s.task_count == 0 || s.rows_per_task % 32 != 0 || s.task_count > tasks ||
				    (s.task_count - 1) * s.rows_per_task >= n || s.task_count * s.rows_per_task < n) {
					++bad;
				}
				const ColumnSplit again = superslm::SplitColumns(n, work, tasks, 32, 1000);
				if (again.task_count != s.task_count || again.rows_per_task != s.rows_per_task) ++bad;
			}
		}
	}
	CHECK_MSG(bad == 0, "3.4: %zu splits do not cover [0, N) once in 32-aligned non-empty blocks", bad);
}

// ---- 4.5 / 6.x: the threaded GEMM against GemmInt8Accumulate ------------------------------------------

void TestGemmIdentityCell45() {
	MacsScope scope(1);
	uint32_t seed = 12345u;
	auto next = [&] {
		seed = seed * 1664525u + 1013904223u;
		return static_cast<int8_t>(static_cast<int32_t>(seed >> 24) - 128);
	};
	size_t mismatches = 0, unthreaded = 0, cases = 0;
	for (size_t m : {2u, 3u, 7u, 8u, 9u, 17u, 64u}) {
		for (size_t k : {1u, 5u, 64u, 191u, 896u}) {
			for (size_t n : {33u, 64u, 100u, 512u}) {
				std::vector<int8_t> a(m * k), w(n * k);
				for (auto& x : a) x = next();
				for (auto& x : w) x = next();
				// Saturate some cells: the int8 extremes the flush bound is stated for.
				a[0] = -128;
				w[0] = -128;
				std::vector<int64_t> want(m * n);
				superslm::GemmInt8Accumulate(a.data(), w.data(), m, k, n, want.data());
				for (RunMode mode : {RunMode::kInline, RunMode::kReverse, RunMode::kThreads}) {
					for (int32_t tasks : {2, 3, 7, 32}) {
						TestHook hook;
						hook.mode = mode;
						hook.max_tasks = tasks;
						const sslm_parallel_for pf = hook.Pf(SSLM_PARALLEL_FOR_PREFILL);
						std::vector<int64_t> got(m * n, int64_t{0x5A5A5A5A});
						const SslmForwardStatus st = superslm::GemmDispatch(GemmThreading{nullptr, &pf}, a.data(),
						                                                    w.data(), m, k, n, got.data());
						++cases;
						if (st != SslmForwardStatus::Ok || got != want) ++mismatches;
						if (!hook.AllThreaded() || hook.counts.size() != 1) ++unthreaded;
					}
				}
			}
		}
	}
	CHECK_MSG(unthreaded == 0, "4.5 preamble: %zu of %zu cases did not make one threaded run call", unthreaded,
	          cases);
	CHECK_MSG(mismatches == 0, "4.5: %zu of %zu threaded GEMMs differ from GemmInt8Accumulate", mismatches, cases);
}

// ---- 2.1: a hostile `run` at the GEMM ------------------------------------------------------------------

void TestHostileGemmCell21() {
	MacsScope scope(1);
	std::vector<int8_t> a(8 * 64, 7), w(128 * 64, -3);
	for (Hostile h : {Hostile::kDuplicate, Hostile::kOmit, Hostile::kMinusOne, Hostile::kTaskCount}) {
		TestHook hook;
		hook.hostile = h;
		hook.hostile_call = 0;
		const sslm_parallel_for pf = hook.Pf(SSLM_PARALLEL_FOR_PREFILL);
		std::vector<int64_t> out(8 * 128);
		const SslmForwardStatus st =
		    superslm::GemmDispatch(GemmThreading{nullptr, &pf}, a.data(), w.data(), 8, 64, 128, out.data());
		CHECK_MSG(st == SslmForwardStatus::ParallelForIncomplete, "2.1 hostile %d at the GEMM: status %d",
		          static_cast<int>(h), static_cast<int>(st));
	}
}

// ---- 2.4: the setters --------------------------------------------------------------------------------

void TestSettersCell24() {
	const Fixture& f = GetFdef();
	if (!f.ok) return;
	Rig rig(f);
	if (!rig.ok) return;
	TestHook hook;
	struct Case {
		uint32_t reserved;
		bool accept;
	};
	const Case cases[] = {{0u, true},  {1u, true},  {2u, true},          {3u, true},
	                      {4u, false}, {5u, false}, {0x80000001u, false}, {0xFFFFFFFFu, false}};
	for (const Case& c : cases) {
		sslm_parallel_for pf{&TestHook::Run, &hook, 4, c.reserved};
		const sslm_status st = sslm_workspace_set_parallel_for(rig.ws, &pf);
		CHECK_MSG(c.accept ? st == SSLM_OK : st == SSLM_INVALID_ARGUMENT, "2.4 reserved %#x: status %d, want %s",
		          c.reserved, static_cast<int>(st), c.accept ? "accepted" : "SSLM_INVALID_ARGUMENT");
	}
}

// ---- 1.2 / 9.1b: whole prefill and decode, threaded against no hook -----------------------------------

void TestWholePrefillCell12() {
	MacsScope scope(1);
	const Fixture& f = GetFdef();
	if (!f.ok) return;
	const std::vector<int32_t> prompt = Prompt(40);
	for (size_t chunk : {2u, 3u, 8u, 13u, 40u}) {
		const Workload ref = RunWorkload(f, prompt, chunk, 6, nullptr, 0u);
		CHECK_MSG(!ref.blob_prefill.empty() && ref.tokens.size() == 6, "1.2 chunk %zu: the reference did not run",
		          chunk);
		for (uint32_t bits : {SSLM_PARALLEL_FOR_PREFILL, SSLM_PARALLEL_FOR_PREFILL | SSLM_PARALLEL_FOR_MATVEC}) {
			for (RunMode mode : {RunMode::kInline, RunMode::kReverse, RunMode::kThreads}) {
				TestHook hook;
				hook.mode = mode;
				const Workload got = RunWorkload(f, prompt, chunk, 6, &hook, bits);
				CHECK_MSG(hook.AllThreaded(), "1.2 chunk %zu bits %u mode %d: preamble, run calls %s", chunk, bits,
				          static_cast<int>(mode), Str(hook.counts).c_str());
				CHECK_MSG(got == ref, "1.2 chunk %zu bits %u mode %d: blobs or tokens differ from no hook (prefill "
				          "blob %s, tokens %s / %s)", chunk, bits, static_cast<int>(mode),
				          got.blob_prefill == ref.blob_prefill ? "same" : "differs", Str(got.tokens).c_str(),
				          Str(ref.tokens).c_str());
			}
		}
	}
	// The exact `run` sequence of one 8-token call under bit 0 alone: one call per projection.
	{
		Rig rig(f);
		if (!rig.ok) return;
		TestHook hook;
		rig.Install(&hook, SSLM_PARALLEL_FOR_PREFILL);
		sslm_seq s = rig.NewSeq();
		superslm::test::ResetBatchedRunCalls();
		rig.PrefillAll(s, Prompt(8), 8);
		CHECK_MSG(hook.counts == BatchedPassCounts(f.layers), "1.2: one 8-token call made %s, want %s",
		          Str(hook.counts).c_str(), Str(BatchedPassCounts(f.layers)).c_str());
		CHECK_MSG(superslm::test::BatchedRunCalls() == hook.counts.size(),
		          "10.0 H seam: the engine counted %zu batched run calls, the hook %zu",
		          superslm::test::BatchedRunCalls(), hook.counts.size());
		// Bit 0 alone does not thread a one-token call (that is bit 1's rule).
		hook.Clear();
		int32_t tok = 77, consumed = 0;
		CHECK_MSG(rig.Prefill(s, &tok, 1, &consumed) == SSLM_OK && consumed == 1 && hook.counts.empty(),
		          "8.2 bit 0 alone: a one-token call made %s", Str(hook.counts).c_str());
	}
	// The production minimum on F-DEF: every projection is below it, so nothing threads (§5.2).
	{
		superslm::test::ResetMinMacsPerTask();
		TestHook hook;
		const Workload got = RunWorkload(f, prompt, 40, 2, &hook, SSLM_PARALLEL_FOR_PREFILL);
		CHECK_MSG(got.blob_prefill == RunWorkload(f, prompt, 40, 2, nullptr, 0u).blob_prefill,
		          "1.2 production minimum: blob differs");
		// Only the token finish's logits step calls `run` (two decode steps).
		CHECK_MSG(hook.counts.size() == 2, "1.2 production minimum: %zu run calls, want the 2 finishes",
		          hook.counts.size());
		superslm::test::SetMinMacsPerTask(1);
	}
}

// ---- 2.1 / 5.2: a hostile `run` inside a batched prefill: nothing committed, counters restored, retry ----

void TestHostilePrefillCell21() {
	MacsScope scope(1);
	const Fixture& f = GetFdef();
	if (!f.ok) return;
	const std::vector<int32_t> head = Prompt(16);
	const std::vector<int32_t> tail = Prompt(24, 53, 5);
	const std::vector<int32_t> whole = [&] {
		std::vector<int32_t> v = head;
		v.insert(v.end(), tail.begin(), tail.end());
		return v;
	}();
	const Workload ref = RunWorkload(f, whole, 16, 4, nullptr, 0u);
	const int last = static_cast<int>(BatchedPassCounts(f.layers).size()) - 1;
	for (Hostile h : {Hostile::kDuplicate, Hostile::kOmit, Hostile::kMinusOne, Hostile::kTaskCount}) {
		for (int call : {0, 5, last}) {
			Rig rig(f);
			if (!rig.ok) return;
			TestHook hook;
			rig.Install(&hook, SSLM_PARALLEL_FOR_PREFILL);
			sslm_seq s = rig.NewSeq();
			rig.PrefillAll(s, head, 16);
			const SaturationCounters before = Ctr(s);
			const int64_t len = ContextLength(s);
			const std::vector<uint8_t> blob_before = rig.Save(s);
			hook.Clear();
			hook.hostile = h;
			hook.hostile_call = call;
			int32_t consumed = -1;
			const sslm_status st = rig.Prefill(s, tail.data(), static_cast<int32_t>(tail.size()), &consumed);
			CHECK_MSG(st == SSLM_INVALID_ARGUMENT && consumed == 0,
			          "2.1 hostile %d at call %d: status %d consumed %d, want SSLM_INVALID_ARGUMENT and 0",
			          static_cast<int>(h), call, static_cast<int>(st), consumed);
			CHECK_MSG(ContextLength(s) == len && SameCounters(Ctr(s), before) && rig.Save(s) == blob_before,
			          "5.2 hostile %d at call %d: the failed call moved the sequence", static_cast<int>(h), call);
			// Retry with a good hook: the whole workload equals the no-hook reference.
			hook.hostile = Hostile::kNone;
			rig.PrefillAll(s, tail, 16);
			Workload got;
			got.blob_prefill = rig.Save(s);
			got.tokens = rig.DecodeN(s, 4);
			got.blob_end = rig.Save(s);
			CHECK_MSG(got == ref, "5.2 hostile %d at call %d: the retry differs from the no-hook run",
			          static_cast<int>(h), call);
		}
	}
}

// ---- 5.4: an allocation failure on the calling thread before `run` --------------------------------------

void TestAllocFaultCell54() {
	MacsScope scope(1);
	const Fixture& f = GetFdef();
	if (!f.ok) return;
	if (superslm::detail::DispatchGemmPath(superslm::detail::ActiveGemmTier(), 8) !=
	    superslm::detail::GemmPath::kTiled) {
		// The loop path allocates nothing per task and has no slot to arm; the cell's subject is absent.
		std::printf("5.4: this tier runs no tiled kernel; the tiled allocation cell is not applicable here\n");
		return;
	}
	const std::vector<int32_t> head = Prompt(8);
	const std::vector<int32_t> tail = Prompt(16, 41, 3);
	std::vector<int32_t> whole = head;
	whole.insert(whole.end(), tail.begin(), tail.end());
	const Workload ref = RunWorkload(f, whole, 8, 3, nullptr, 0u);
	Rig rig(f);
	if (!rig.ok) return;
	TestHook hook;
	rig.Install(&hook, SSLM_PARALLEL_FOR_PREFILL);
	sslm_seq s = rig.NewSeq();
	rig.PrefillAll(s, head, 8);
	const SaturationCounters before = Ctr(s);
	const std::vector<uint8_t> blob_before = rig.Save(s);
	hook.Clear();
	superslm_test::g_inject_tiled_gemm_alloc_failure = true;
	int32_t consumed = -1;
	const sslm_status st = rig.Prefill(s, tail.data(), 8, &consumed);
	superslm_test::g_inject_tiled_gemm_alloc_failure = false;
	CHECK_MSG(st == SSLM_ALLOCATION_FAILED && consumed == 0, "5.4: status %d consumed %d", static_cast<int>(st),
	          consumed);
	CHECK_MSG(hook.counts.empty(), "5.4: the fault fired after %zu run calls; it must fire before the first",
	          hook.counts.size());
	CHECK_MSG(SameCounters(Ctr(s), before) && rig.Save(s) == blob_before, "5.4: the failed call moved the sequence");
	rig.PrefillAll(s, tail, 8);
	Workload got;
	got.blob_prefill = rig.Save(s);
	got.tokens = rig.DecodeN(s, 3);
	got.blob_end = rig.Save(s);
	CHECK_MSG(got == ref, "5.4: the retry differs from the no-hook run");
}

// ---- 3.2: two workspaces prefilling concurrently, each with its own threaded hook ------------------------

void TestConcurrentWorkspacesCell32() {
	MacsScope scope(1);
	const Fixture& f = GetFdef();
	if (!f.ok) return;
	const std::vector<int32_t> p1 = Prompt(48), p2 = Prompt(48, 29, 7);
	const Workload r1 = RunWorkload(f, p1, 16, 4, nullptr, 0u), r2 = RunWorkload(f, p2, 16, 4, nullptr, 0u);
	Workload g1, g2;
	TestHook h1, h2;
	h1.mode = h2.mode = RunMode::kThreads;
	std::thread t1([&] { g1 = RunWorkload(f, p1, 16, 4, &h1, SSLM_PARALLEL_FOR_PREFILL | SSLM_PARALLEL_FOR_MATVEC); });
	std::thread t2([&] { g2 = RunWorkload(f, p2, 16, 4, &h2, SSLM_PARALLEL_FOR_PREFILL); });
	t1.join();
	t2.join();
	CHECK_MSG(h1.AllThreaded() && h2.AllThreaded(), "3.2 preamble: a hook did not thread");
	CHECK_MSG(g1 == r1 && g2 == r2, "3.2: a concurrent threaded workload differs from its no-hook run");
}

}  // namespace

void RunBatchedThreadingCells(int& checks, int& failures) {
	GChecks = 0;
	GFailures = 0;
	TestRuleCell23();
	TestSplitCell34();
	TestGemmIdentityCell45();
	TestHostileGemmCell21();
	TestSettersCell24();
	TestWholePrefillCell12();
	TestHostilePrefillCell21();
	TestAllocFaultCell54();
	TestConcurrentWorkspacesCell32();
	FixtureSlot().reset();
	std::printf("batched-threading (tiled slice 2) cells: %d checks, %d failures\n", GChecks.load(), GFailures.load());
	checks += GChecks.load();
	failures += GFailures.load();
}

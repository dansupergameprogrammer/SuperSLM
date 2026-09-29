// Decode-threading plan (rev 1.2): the cells of the plan's coverage model (§8) that steps D0 and
// D1 own, in their own translation unit. test_main.cpp calls RunDecodeThreadingCells and adds this
// unit's check and failure counts to its own totals, the pattern test_tiled_gemm.cpp set. Cell
// numbers are the plan's §8 numbering.
//
// D0 (§6): the shared split (4.1), the logits step's partition and one-task call pinned to v1.9.0
// (7.1), and the saturation-counter snapshot.
//
// "Reference" below is v1.9.0's formula restated test-side, or the serial path with no hook; never
// the build under test's split.

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "superslm/forward_sites.h"
#include "superslm/matmul.h"
#include "superslm/parallel_for.h"
#include "../src/forward/parallel_split.h"

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
using superslm::SslmForwardStatus;
using superslm::SslmForwardStatusName;

// ---- D0: 4.1, the shared split ---------------------------------------------------------------------

// The test-side T of §3.2, written from the plan's formula.
size_t ExpectedT(size_t n, size_t work, size_t max_tasks, size_t align, size_t min_work) {
	size_t t = std::min<size_t>(max_tasks, 256);
	t = std::min(t, (n + align - 1) / align);
	if (min_work != 0) t = std::min(t, work / min_work);
	return t;
}

void TestSplitColumnsCell41() {
	static const size_t kN[] = {1,   31,  32,   33,   63,   64,   65,    127,
	                            128, 192, 896, 1024, 4864, 9728, 151936};
	static const size_t kMinWork[] = {0, 1, 256 * 1024};
	long long bad = 0, cases = 0, threaded = 0;
	for (size_t align : {size_t{32}, size_t{64}}) {
		for (size_t n : kN) {
			for (size_t min_work : kMinWork) {
				for (size_t mt = 1; mt <= 256; ++mt) {
					const size_t work = n * 896;  // K·N at the real hidden width
					const ColumnSplit s = superslm::SplitColumns(n, work, mt, align, min_work);
					const size_t t = ExpectedT(n, work, mt, align, min_work);
					++cases;
					// Blocks contiguous, covering [0, N) exactly once, the last ending at N.
					bool ok = s.rows_per_task > 0 && s.task_count > 0 && s.rows_per_task % align == 0;
					if (ok) {
						size_t covered = 0;
						for (size_t i = 0; i < s.task_count; ++i) {
							const size_t b = i * s.rows_per_task;
							const size_t e = std::min(n, b + s.rows_per_task);
							if (b != covered || e <= b) ok = false;
							covered = e;
						}
						if (covered != n) ok = false;
					}
					// Threaded exactly when T >= 2, and then with a task count in [2, T].
					const bool is_threaded = s.task_count >= 2;
					if (is_threaded != (t >= 2)) ok = false;
					if (is_threaded && (s.task_count < 2 || s.task_count > t)) ok = false;
					if (is_threaded) ++threaded;
					if (!ok) {
						if (++bad <= 5)
							std::printf("  4.1: align %zu N %zu min_work %zu max_tasks %zu -> rows %zu tasks %zu (T %zu)\n",
							            align, n, min_work, mt, s.rows_per_task, s.task_count, t);
					}
				}
			}
		}
	}
	CHECK_MSG(bad == 0, "4.1: %lld of %lld splits break the §3.2 contract", bad, cases);
	CHECK_MSG(threaded > 0 && threaded < cases, "4.1: the grid must hold both threaded (%lld) and serial splits",
	          threaded);
	const ColumnSplit zero = superslm::SplitColumns(0, 0, 4, 64, 0);
	CHECK_MSG(zero.rows_per_task == 0 && zero.task_count == 0, "4.1: N = 0 gives {%zu, %zu}", zero.rows_per_task,
	          zero.task_count);
	std::printf("decode-threading 4.1: %lld splits checked, %lld threaded\n", cases, threaded);
}

// ---- D0: 7.1, the logits step unchanged (M1) -------------------------------------------------------

// v1.9.0's LogitsSiteParallel partition, restated: rows = roundup(ceil(V / max_tasks), 64).
ColumnSplit V190LogitsSplit(size_t v, size_t max_tasks) {
	const size_t mt = std::min<size_t>(max_tasks, 256);
	const size_t per_task = (v + mt - 1) / mt;
	const size_t rows = ((per_task + 63) / 64) * 64;
	return ColumnSplit{rows, (v + rows - 1) / rows};
}

// A hook that records each `run` and sniffs the partition: it runs task 0 alone first, reads how many
// poisoned wide-logit entries that task wrote (its block is [0, rows_per_task)), then runs the rest.
struct SniffHook {
	int64_t* wide = nullptr;
	size_t vocab = 0;
	int calls = 0;
	int32_t last_task_count = 0;
	size_t sniffed_rows = 0;
};
constexpr int64_t kPoisonWide = INT64_MIN + 12345;
void SniffRun(void* host_ctx, int32_t task_count, sslm_task_fn task, void* task_ctx) {
	SniffHook& h = *static_cast<SniffHook*>(host_ctx);
	++h.calls;
	h.last_task_count = task_count;
	for (size_t i = 0; i < h.vocab; ++i) h.wide[i] = kPoisonWide;
	task(task_ctx, 0);
	size_t rows = 0;
	while (rows < h.vocab && h.wide[rows] != kPoisonWide) ++rows;
	h.sniffed_rows = rows;
	for (int32_t i = 1; i < task_count; ++i) task(task_ctx, i);
}

void TestLogitsPartitionCell71() {
	// (a) The shared split called as the logits step calls it, over the whole grid.
	long long bad = 0, cases = 0;
	for (size_t v = 1; v <= 2999; ++v) {
		for (size_t mt = 2; mt <= 256; ++mt) {
			const ColumnSplit a = superslm::SplitColumns(v, 0, mt, 64, 0);
			const ColumnSplit b = V190LogitsSplit(v, mt);
			++cases;
			if (a.rows_per_task != b.rows_per_task || a.task_count != b.task_count) {
				if (++bad <= 5)
					std::printf("  7.1: V %zu max_tasks %zu: split {%zu, %zu}, v1.9.0 {%zu, %zu}\n", v, mt,
					            a.rows_per_task, a.task_count, b.rows_per_task, b.task_count);
			}
		}
	}
	for (size_t mt = 2; mt <= 256; ++mt) {
		const ColumnSplit a = superslm::SplitColumns(151936, 0, mt, 64, 0);
		const ColumnSplit b = V190LogitsSplit(151936, mt);
		++cases;
		if (a.rows_per_task != b.rows_per_task || a.task_count != b.task_count) ++bad;
	}
	CHECK_MSG(bad == 0, "7.1(a): %lld of %lld (V, max_tasks) splits differ from v1.9.0's", bad, cases);

	// (b) Through LogitsSiteParallel itself: the task count `run` receives, the block size task 0 writes,
	// whether `run` is called at all (a one-task call included, V <= 64), and the row, against LogitsSite.
	std::vector<size_t> vs;
	for (size_t v = 1; v <= 130; ++v) vs.push_back(v);
	for (size_t v : {191, 192, 193, 255, 256, 257, 1000, 2999}) vs.push_back(v);
	long long bad_b = 0, calls_b = 0, one_task_calls = 0;
	for (size_t v : vs) {
		std::vector<int8_t> head(v), codes(1, 3);
		for (size_t i = 0; i < v; ++i) head[i] = static_cast<int8_t>(static_cast<int>(i * 37 % 255) - 127);
		std::vector<int64_t> wide_ref(v), wide(v);
		std::vector<int32_t> out_ref(v), out(v);
		const SslmForwardStatus rs = superslm::LogitsSite(codes.data(), 1, head.data(), v, wide_ref.data(),
		                                                  out_ref.data());
		for (size_t mt = 2; mt <= 256; ++mt) {
			SniffHook h;
			h.wide = wide.data();
			h.vocab = v;
			const sslm_parallel_for pf{&SniffRun, &h, static_cast<int32_t>(mt), 0u};
			const SslmForwardStatus st =
			    superslm::LogitsSiteParallel(codes.data(), 1, head.data(), v, wide.data(), out.data(), &pf);
			const ColumnSplit want = V190LogitsSplit(v, mt);
			++calls_b;
			const bool ok = st == rs && h.calls == 1 && static_cast<size_t>(h.last_task_count) == want.task_count &&
			                h.sniffed_rows == std::min(v, want.rows_per_task) && out == out_ref;
			if (h.last_task_count == 1) ++one_task_calls;
			if (!ok && ++bad_b <= 5)
				std::printf("  7.1(b): V %zu max_tasks %zu: status %s, %d calls, %d tasks (want %zu), block %zu (want %zu)\n",
				            v, mt, SslmForwardStatusName(st), h.calls, h.last_task_count, want.task_count,
				            h.sniffed_rows, std::min(v, want.rows_per_task));
		}
	}
	{
		// V = 151,936 at a spread of max_tasks, hidden width 1.
		const size_t v = 151936;
		std::vector<int8_t> head(v), codes(1, -5);
		for (size_t i = 0; i < v; ++i) head[i] = static_cast<int8_t>(static_cast<int>(i * 101 % 255) - 127);
		std::vector<int64_t> wide_ref(v), wide(v);
		std::vector<int32_t> out_ref(v), out(v);
		superslm::LogitsSite(codes.data(), 1, head.data(), v, wide_ref.data(), out_ref.data());
		for (size_t mt : {2, 3, 4, 7, 8, 16, 64, 255, 256}) {
			SniffHook h;
			h.wide = wide.data();
			h.vocab = v;
			const sslm_parallel_for pf{&SniffRun, &h, static_cast<int32_t>(mt), 0u};
			const SslmForwardStatus st =
			    superslm::LogitsSiteParallel(codes.data(), 1, head.data(), v, wide.data(), out.data(), &pf);
			const ColumnSplit want = V190LogitsSplit(v, mt);
			++calls_b;
			if (!(st == SslmForwardStatus::Ok && h.calls == 1 &&
			      static_cast<size_t>(h.last_task_count) == want.task_count && h.sniffed_rows == want.rows_per_task &&
			      out == out_ref) &&
			    ++bad_b <= 5)
				std::printf("  7.1(b): V %zu max_tasks %zu: %d tasks (want %zu), block %zu (want %zu)\n", v, mt,
				            h.last_task_count, want.task_count, h.sniffed_rows, want.rows_per_task);
		}
	}
	CHECK_MSG(bad_b == 0, "7.1(b): %lld of %lld logits calls differ from v1.9.0 in task count, block or row", bad_b,
	          calls_b);
	// 64 vocabularies (V = 1..64) x 255 max_tasks values call `run` with one task.
	CHECK_MSG(one_task_calls == 64 * 255, "7.1(b): %lld one-task `run` calls, want %d (V <= 64 still calls run)",
	          one_task_calls, 64 * 255);
	std::printf("decode-threading 7.1: %lld splits and %lld logits calls checked, %lld one-task calls\n", cases,
	            calls_b, one_task_calls);
}

// ---- D0: the saturation-counter snapshot -------------------------------------------------------------

void TestSaturationCountersRoundTrip() {
	superslm::SequenceLayerState s;
	s.kv_saturation_count = 11;
	s.kv_landing_saturation_count = 3;
	s.k_channel_landing_saturation_count = 4;
	s.rope_q_saturation_count = 1;
	s.rope_k_saturation_count = 3;
	s.context_length = 7;
	const superslm::SaturationCounters snap = superslm::SaturationCounters::Snapshot(s);
	s.kv_saturation_count = 99;
	s.kv_landing_saturation_count = 98;
	s.k_channel_landing_saturation_count = 97;
	s.rope_q_saturation_count = 96;
	s.rope_k_saturation_count = 95;
	s.context_length = 8;
	snap.RestoreTo(s);
	CHECK_MSG(s.kv_saturation_count == 11 && s.kv_landing_saturation_count == 3 &&
	              s.k_channel_landing_saturation_count == 4 && s.rope_q_saturation_count == 1 &&
	              s.rope_k_saturation_count == 3 && s.context_length == 8,
	          "SaturationCounters: restore must put back exactly the five counters and nothing else");
}

}  // namespace

void RunDecodeThreadingCells(int& checks, int& failures) {
	TestSplitColumnsCell41();
	TestLogitsPartitionCell71();
	TestSaturationCountersRoundTrip();
	std::printf("decode-threading cells: %d checks, %d failures\n", GChecks, GFailures);
	checks += GChecks;
	failures += GFailures;
}

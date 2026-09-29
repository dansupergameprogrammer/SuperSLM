// The shared split and exactly-once runner (decode-threading plan rev 1.2, §3.2). See
// parallel_split.h for the contracts.
#include "parallel_split.h"

#include <algorithm>
#include <atomic>
#include <new>

#include "superslm/matmul.h"  // GemmInt8AccumulateRow, GemmInt8Accumulate

namespace superslm {

ColumnSplit SplitColumns(size_t n, size_t work, size_t max_tasks, size_t align,
                         size_t min_work) noexcept {
	if (n == 0 || align == 0) return ColumnSplit{};
	size_t t = std::min<size_t>(max_tasks, SSLM_PARALLEL_FOR_MAX_TASKS);
	t = std::min(t, (n + align - 1) / align);
	if (min_work != 0) t = std::min(t, work / min_work);
	if (t < 1) t = 1;
	const size_t per_task = (n + t - 1) / t;
	const size_t rows_per_task = ((per_task + align - 1) / align) * align;
	return ColumnSplit{rows_per_task, (n + rows_per_task - 1) / rows_per_task};
}

namespace {

// Call-local state one RunExactlyOnce call hands to the host's `run` as `task_ctx`. It lives on
// that call's stack and ends when the call returns (parallel_for.h's precondition).
struct ExactlyOnceCtx {
	ExactlyOnceBody body;
	void* body_ctx;
	int32_t task_count;
	std::atomic<uint8_t>* state;  // [task_count]: 0 = not started, 1 = admitted, 2 = done
	std::atomic<bool>* violation;
};

void ExactlyOnceTask(void* task_ctx, int32_t task_index) {
	ExactlyOnceCtx& c = *static_cast<ExactlyOnceCtx*>(task_ctx);
	if (task_index < 0 || task_index >= c.task_count) {
		c.violation->store(true, std::memory_order_release);
		return;
	}
	// Admission: exactly one invocation of an index can take 0 -> 1, so exactly one can reach the
	// body. A later or concurrent second invocation fails the exchange and runs nothing.
	uint8_t expected = 0;
	if (!c.state[task_index].compare_exchange_strong(expected, uint8_t{1},
	                                                 std::memory_order_acq_rel)) {
		c.violation->store(true, std::memory_order_release);
		return;
	}
	c.body(c.body_ctx, static_cast<size_t>(task_index));
	// Release: publishes this task's writes to the acquire scan after `run` returns.
	c.state[task_index].store(uint8_t{2}, std::memory_order_release);
}

}  // namespace

SslmForwardStatus RunExactlyOnce(const sslm_parallel_for& pf, size_t task_count,
                                 ExactlyOnceBody body, void* ctx) {
	// Value-initialized (C++20): every entry starts at 0 = not started.
	std::atomic<uint8_t> state[SSLM_PARALLEL_FOR_MAX_TASKS];
	std::atomic<bool> violation{false};
	ExactlyOnceCtx c{body, ctx, static_cast<int32_t>(task_count), state, &violation};
	pf.run(pf.host_ctx, static_cast<int32_t>(task_count), &ExactlyOnceTask, &c);

	bool complete = !violation.load(std::memory_order_acquire);
	for (size_t i = 0; complete && i < task_count; ++i) {
		complete = state[i].load(std::memory_order_acquire) == 2;
	}
	return complete ? SslmForwardStatus::Ok : SslmForwardStatus::ParallelForIncomplete;
}

// ---- one-row matvec groups -------------------------------------------------------------------------

#if defined(SUPERSLM_ENABLE_MATVEC_TEST_SEAMS)
namespace {
std::atomic<size_t> g_min_row_bytes_per_task{kMinRowBytesPerTask};
}  // namespace
size_t MinRowBytesPerTask() noexcept {
	return g_min_row_bytes_per_task.load(std::memory_order_relaxed);
}
#else
size_t MinRowBytesPerTask() noexcept { return kMinRowBytesPerTask; }
#endif

ColumnSplit MatvecGroupSplit(const sslm_parallel_for* pf, size_t k, size_t total_rows) noexcept {
	// RED: the one-row split is not implemented yet; every group runs serially.
	(void)pf;
	(void)k;
	(void)total_rows;
	return ColumnSplit{};
}

namespace {

// Call-local state one MatvecGroupParallel call hands RunExactlyOnce; on that call's stack.
struct MatvecTaskCtx {
	const int8_t* x;
	size_t k;
	const MatvecPart* parts;
	size_t part_count;
	size_t total_rows;
	size_t rows_per_task;
};

void MatvecTask(void* ctx, size_t task_index) noexcept {
	const MatvecTaskCtx& c = *static_cast<const MatvecTaskCtx*>(ctx);
	const size_t begin = task_index * c.rows_per_task;
	const size_t end = std::min(c.total_rows, begin + c.rows_per_task);
	// Map [begin, end) of the concatenated index space onto each part it covers.
	size_t part_start = 0;
	for (size_t p = 0; p < c.part_count; ++p) {
		const MatvecPart& part = c.parts[p];
		const size_t part_end = part_start + part.rows;
		const size_t lo = std::max(begin, part_start);
		const size_t hi = std::min(end, part_end);
		if (lo < hi) {
			const size_t r = lo - part_start;
			GemmInt8AccumulateRow(c.x, part.weight + r * c.k, c.k, hi - lo, part.out + r);
		}
		part_start = part_end;
	}
}

}  // namespace

SslmForwardStatus MatvecGroupParallel(const sslm_parallel_for& pf, const ColumnSplit& split,
                                      const int8_t* x, size_t k, const MatvecPart* parts,
                                      size_t part_count) {
	size_t total = 0;
	for (size_t p = 0; p < part_count; ++p) total += parts[p].rows;
	MatvecTaskCtx ctx{x, k, parts, part_count, total, split.rows_per_task};
	return RunExactlyOnce(pf, split.task_count, &MatvecTask, &ctx);
}

SslmForwardStatus GemmDispatch(const GemmThreading& threading, const int8_t* activations,
                               const int8_t* weights, size_t m, size_t k, size_t n, int64_t* out) {
	if (m == 1 && threading.row != nullptr) {
		const ColumnSplit split = MatvecGroupSplit(threading.row, k, n);
		if (split.task_count >= 2) {
			const MatvecPart part{weights, n, out};
			return MatvecGroupParallel(*threading.row, split, activations, k, &part, 1);
		}
	}
	GemmInt8Accumulate(activations, weights, m, k, n, out);
	return SslmForwardStatus::Ok;
}

// ---- test seams ------------------------------------------------------------------------------------

#if defined(SUPERSLM_ENABLE_MATVEC_TEST_SEAMS)
namespace {
struct ArmedFault {
	bool armed = false;
	MatvecFaultSite site = MatvecFaultSite::kQ;
	uint32_t layer = 0;
	MatvecFaultKind kind = MatvecFaultKind::kNonPfiStatus;
};
std::atomic<bool> g_fault_armed{false};
ArmedFault g_fault;  // written only while disarmed, by the single test thread that arms it
}  // namespace

namespace test {
void SetMinRowBytesPerTask(size_t bytes) noexcept {
	g_min_row_bytes_per_task.store(bytes, std::memory_order_relaxed);
}
void ResetMinRowBytesPerTask() noexcept {
	g_min_row_bytes_per_task.store(kMinRowBytesPerTask, std::memory_order_relaxed);
}
void ArmLayerSiteFault(MatvecFaultSite site, uint32_t layer, MatvecFaultKind kind) noexcept {
	g_fault.site = site;
	g_fault.layer = layer;
	g_fault.kind = kind;
	g_fault_armed.store(true, std::memory_order_release);
}
void DisarmLayerSiteFault() noexcept { g_fault_armed.store(false, std::memory_order_release); }
}  // namespace test

SslmForwardStatus MaybeLayerSiteFault(MatvecFaultSite site, uint32_t layer) {
	if (!g_fault_armed.load(std::memory_order_acquire)) return SslmForwardStatus::Ok;
	if (g_fault.site != site || g_fault.layer != layer) return SslmForwardStatus::Ok;
	bool expected = true;
	if (!g_fault_armed.compare_exchange_strong(expected, false, std::memory_order_acq_rel)) {
		return SslmForwardStatus::Ok;
	}
	if (g_fault.kind == MatvecFaultKind::kBadAlloc) throw std::bad_alloc{};
	return SslmForwardStatus::CarriedScaleMantissaOutOfDomain;
}
#endif

}  // namespace superslm

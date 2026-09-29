// The shared split and exactly-once runner (decode-threading plan rev 1.2, §3.2). See
// parallel_split.h for the contracts.
#include "parallel_split.h"

#include <algorithm>
#include <atomic>

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

}  // namespace superslm

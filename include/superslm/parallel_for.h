#ifndef SUPERSLM_PARALLEL_FOR_H
#define SUPERSLM_PARALLEL_FOR_H
/* A caller-supplied parallel-for hook. SuperSLM never creates a thread: a host that wants the
 * token finish's logits rows computed on several threads installs one of these, and the library
 * hands its row blocks to the host's `run`. With no hook installed the finish runs serially on
 * the calling thread, exactly as in releases before 1.7.0.
 *
 * Installed with sslm_workspace_set_parallel_for (CPU backend, sslm_abi.h) or
 * sslm_gpu_context_set_host_parallel_for (GPU backend, gpu_1p0.h). Only the token finish's
 * logits step reads it; prefill and every other call ignore it.
 *
 * THE CONTRACT `run` MUST MEET:
 * - Invoke `task(task_ctx, i)` exactly once for every `i` in [0, task_count), on any threads,
 *   including the calling one, in any order, concurrently or not.
 * - A duplicate, an omitted or an out-of-range index, invoked by a `run` that still waits for
 *   all of its invocations, is detected and fails the SuperSLM call (SSLM_INVALID_ARGUMENT on the
 *   CPU backend, SSLM_GPU_PARALLEL_FOR_INCOMPLETE on the GPU backend). It never corrupts the row
 *   or returns a token, and the sequence is left ready to retry.
 * - Return only after every invocation has returned, and never touch `task` or `task_ctx` after
 *   that. This is a hard precondition, not a detected error: `task_ctx` and the storage it points
 *   into belong to the SuperSLM call and end when it returns. A `run` that returns while an
 *   invocation is still executing has undefined behaviour.
 * - Tasks never block on one another, never call into SuperSLM, and never throw.
 * - `task_count` is in [1, max_tasks].
 * - `run` is called only from inside a SuperSLM call, on the thread that made that call. A host
 *   whose job system can wait on work from inside a job may call SuperSLM from a job.
 * - Running every task on the calling thread is a valid `run` and produces identical output.
 *
 * The logits rows are split into contiguous blocks whose size is a function of the vocabulary
 * size and `max_tasks` only. Every row's value is an exact integer sum, so the tokens produced
 * are identical with any hook, any `max_tasks`, and no hook.
 *
 * docs/api.md carries a reference `run` over std::thread for hosts with no job system. */

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef void (*sslm_task_fn)(void* task_ctx, int32_t task_index);
typedef void (*sslm_run_tasks_fn)(void* host_ctx, int32_t task_count, sslm_task_fn task,
                                  void* task_ctx);

typedef struct sslm_parallel_for {
	sslm_run_tasks_fn run; /* NULL: serial */
	void* host_ctx;        /* passed back to run unchanged; must outlive the installation */
	int32_t max_tasks;     /* most tasks one run may carry; <= 1: serial;
	                          at most SSLM_PARALLEL_FOR_MAX_TASKS */
	uint32_t reserved;     /* must be 0 */
} sslm_parallel_for;

/* The most tasks one `run` may carry. The exactly-once check keeps one byte of call-local state
 * per task, so a bounded count keeps the decode path free of heap allocation. */
#define SSLM_PARALLEL_FOR_MAX_TASKS 256

#ifdef __cplusplus
}
#endif

#endif /* SUPERSLM_PARALLEL_FOR_H */

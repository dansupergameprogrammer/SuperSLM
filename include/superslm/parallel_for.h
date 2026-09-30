#ifndef SUPERSLM_PARALLEL_FOR_H
#define SUPERSLM_PARALLEL_FOR_H
/* A caller-supplied parallel-for hook. SuperSLM never creates a thread: a host that wants the
 * token finish's logits rows computed on several threads installs one of these, and the library
 * hands its row blocks to the host's `run`. With no hook installed the finish runs serially on
 * the calling thread, as in releases before 1.7.0, and its tokens are identical to 1.6.0.
 *
 * Installed with sslm_workspace_set_parallel_for (CPU backend, sslm_abi.h) or
 * sslm_gpu_context_set_host_parallel_for (GPU backend, gpu_1p0.h). Which steps read it:
 * - The token finish's logits step, always.
 * - With SSLM_PARALLEL_FOR_MATVEC set in `reserved` (CPU backend), every one-row (M = 1)
 *   projection too: each decode layer's, and a prefill call's that admits exactly one token. A
 *   projection group (k and v together, gate and up together in decode; every other projection
 *   alone) calls `run` only when it splits into two or more tasks; otherwise it runs on the
 *   calling thread. A fixed minimum of weight bytes caps the task count by the group's total
 *   weight bytes; it is not a floor on each task, and the last task can be smaller (docs/api.md
 *   gives the rule).
 * - Every other step, and every step of a prefill call that admits more than one token, ignores it.
 * A setter rejects any `reserved` bit this library does not implement, so a host that asks for a
 * step this build lacks is told so rather than silently ignored. The GPU backend accepts the
 * implemented bits and reads its hook only for the token finish.
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
 * Rows are split into contiguous blocks whose size is a function of the matrix shape,
 * `max_tasks` and the library's named minimum-work constant only. Every row's value is an exact
 * integer sum computed on one thread, so the tokens produced are identical with any hook, any
 * `max_tasks`, any `reserved` bits, and no hook.
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
	uint32_t reserved;     /* opt-in bits: 0 or SSLM_PARALLEL_FOR_MATVEC; any other bit is
	                          rejected by the setters */
} sslm_parallel_for;

/* Bit 1 of `reserved`: also split every CPU one-row (M = 1) projection across `run` (above).
 * Opt-in, so a host that installs a hook for the logits step alone keeps exactly that. Bit 0 is
 * reserved for a later batched-prefill split and is rejected by this build. The macro's presence
 * is the compile-time test for the feature; a library without it rejects the bit at the setter. */
#define SSLM_PARALLEL_FOR_MATVEC 0x2u

/* The most tasks one `run` may carry. The exactly-once check keeps one byte of call-local state
 * per task, so a bounded count keeps the decode path free of heap allocation. */
#define SSLM_PARALLEL_FOR_MAX_TASKS 256

#ifdef __cplusplus
}
#endif

#endif /* SUPERSLM_PARALLEL_FOR_H */

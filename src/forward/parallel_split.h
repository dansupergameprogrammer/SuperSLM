// The pieces every site that splits rows across the host's parallel-for hook shares (decode-
// threading plan rev 1.2, §3.2 and §12): the split, the exactly-once runner, the saturation-counter
// snapshot and the setters' validation. Internal to the library: nothing here is SUPERSLM_API, and
// the installed headers do not include this file.
//
// Users: LogitsSiteParallel (the token finish's logits step), and the one-row matvec groups of the
// decode and one-token-prefill layer loops (MatvecGroupParallel, below).
#ifndef SUPERSLM_SRC_FORWARD_PARALLEL_SPLIT_H
#define SUPERSLM_SRC_FORWARD_PARALLEL_SPLIT_H

#include <cstddef>
#include <cstdint>

#include "superslm/checked_chain_funnel.h"  // SslmForwardStatus
#include "superslm/forward_sites.h"         // SequenceLayerState
#include "superslm/parallel_for.h"

namespace superslm {

// ---- the split -----------------------------------------------------------------------------------

struct ColumnSplit {
	size_t rows_per_task = 0;
	size_t task_count = 0;
};

// Splits N output rows into contiguous blocks of `align`-rounded size (§3.2):
//   T = min(max_tasks, SSLM_PARALLEL_FOR_MAX_TASKS, ceil(N / align), floor(work / min_work)),
// the last term omitted when `min_work` is 0 and T raised to 1 when it would be 0;
//   rows_per_task = roundup(ceil(N / T), align), task_count = ceil(N / rows_per_task).
// N = 0 gives {0, 0}. Task i covers [i * rows_per_task, min(N, (i + 1) * rows_per_task)).
// Whether to call `run` is the caller's rule, not this function's: the logits step calls it for
// any split once the hook has max_tasks >= 2 (one task included); the matvec groups only when
// task_count >= 2. `align` must be at least 1.
ColumnSplit SplitColumns(size_t n, size_t work, size_t max_tasks, size_t align,
                         size_t min_work) noexcept;

// ---- the exactly-once runner ---------------------------------------------------------------------

// One task's work: task `task_index` of the split the caller made. Must not throw, block on
// another task, call into the hook or allocate (parallel_for.h's task rules, and C10).
using ExactlyOnceBody = void (*)(void* ctx, size_t task_index) noexcept;

// Hands `task_count` tasks (1 <= task_count <= SSLM_PARALLEL_FOR_MAX_TASKS) to `pf.run`, admitting
// each index once by compare-exchange on call-local stack state before `body` runs for it, and
// scanning that state after `run` returns. Returns Ok when every index ran exactly once, and
// ParallelForIncomplete on a duplicate, an omitted or an out-of-range index (a body that was
// admitted has run; nothing else has). Allocates nothing. `pf.run` must be non-null. Not noexcept:
// what a throwing `run` does is the host's, and it propagates exactly as it did from v1.9.0's
// LogitsSiteParallel.
SslmForwardStatus RunExactlyOnce(const sslm_parallel_for& pf, size_t task_count,
                                 ExactlyOnceBody body, void* ctx);

// ---- the saturation-counter snapshot (§3.6) ------------------------------------------------------

// The five per-sequence saturation counters a layer's K/V landing, QK-norm and RoPE steps move
// before that layer commits. A retry guard snapshots them and restores them when the attempt it
// guards fails in a way the guard is keyed on.
struct SaturationCounters {
	uint64_t kv = 0;
	uint64_t kv_landing = 0;
	uint64_t k_channel_landing = 0;
	uint64_t rope_q = 0;
	uint64_t rope_k = 0;

	static SaturationCounters Snapshot(const SequenceLayerState& s) noexcept {
		return SaturationCounters{s.kv_saturation_count, s.kv_landing_saturation_count,
		                          s.k_channel_landing_saturation_count,
		                          s.rope_q_saturation_count, s.rope_k_saturation_count};
	}
	void RestoreTo(SequenceLayerState& s) const noexcept {
		s.kv_saturation_count = kv;
		s.kv_landing_saturation_count = kv_landing;
		s.k_channel_landing_saturation_count = k_channel_landing;
		s.rope_q_saturation_count = rope_q;
		s.rope_k_saturation_count = rope_k;
	}
};

// ---- the setters' shared validation (§3.7) -------------------------------------------------------

// The `reserved` bits this build implements. A setter rejects any other bit, so a host that asks
// for a feature this library lacks is told so rather than silently ignored.
inline constexpr uint32_t kImplementedParallelForBits = 0u;

// The one field-domain check both hook setters apply (sslm_workspace_set_parallel_for and
// sslm_gpu_context_set_host_parallel_for). `pf` is non-null.
inline bool ParallelForHookValid(const sslm_parallel_for& pf) noexcept {
	return (pf.reserved & ~kImplementedParallelForBits) == 0u && pf.max_tasks >= 0 &&
	       pf.max_tasks <= SSLM_PARALLEL_FOR_MAX_TASKS &&
	       !(pf.run == nullptr && pf.max_tasks > 1);
}

}  // namespace superslm

#endif  // SUPERSLM_SRC_FORWARD_PARALLEL_SPLIT_H

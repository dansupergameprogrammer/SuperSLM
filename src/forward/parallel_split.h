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

// ---- one-row (M = 1) matvec groups (§3.1, §3.3) --------------------------------------------------

// The least weight bytes (K·N) one task of a one-row matvec group must stream (§3.1). Set on the box
// (plan §7 B1); 256 KiB until then. The production library reads this constant and nothing else;
// a test build moves it only through superslm::test::SetMinRowBytesPerTask (below).
inline constexpr size_t kMinRowBytesPerTask = size_t{256} * 1024;
size_t MinRowBytesPerTask() noexcept;

// One matrix of a group: `rows` output rows of `weight` (row-major, K int8 per row), accumulated
// into `out`. A group is one or two matrices that read the same input row (decode's k + v and
// gate + up); its output index space is its parts laid end to end.
struct MatvecPart {
	const int8_t* weight;
	size_t rows;
	int64_t* out;
};

// The §3.1 rule for one group of `total_rows` rows over input width `k`: the split it runs with,
// or a task_count below 2 when it runs serially (no hook, max_tasks < 2, or too little work for
// two tasks at alignment 64 and MinRowBytesPerTask()). Reads nothing but the hook's `run` and
// `max_tasks`; the caller has already decided the hook applies (GemmThreading::row).
ColumnSplit MatvecGroupSplit(const sslm_parallel_for* pf, size_t k, size_t total_rows) noexcept;

// Runs a group whose split has task_count >= 2 through RunExactlyOnce: task i computes rows
// [i * rows_per_task, min(N, (i + 1) * rows_per_task)) of the concatenated parts, each row one
// GemmInt8AccumulateRow on its own weight row, so every row's value is the serial path's. A block
// that straddles a part boundary writes the tail of one part and the head of the next. Returns Ok
// or ParallelForIncomplete. Allocates nothing.
SslmForwardStatus MatvecGroupParallel(const sslm_parallel_for& pf, const ColumnSplit& split,
                                      const int8_t* x, size_t k, const MatvecPart* parts,
                                      size_t part_count);

// The one GEMM dispatch point of the chunk-batched layer loop (§3.4, §12): M = 1 with
// `threading.row` set and a split of two or more tasks takes the row split above; everything else
// (M >= 8 with `threading.batched` is tiled slice 2's, not in this build) is the serial
// GemmInt8Accumulate call made before this plan. Returns Ok or ParallelForIncomplete.
SslmForwardStatus GemmDispatch(const GemmThreading& threading, const int8_t* activations,
                               const int8_t* weights, size_t m, size_t k, size_t n, int64_t* out);

// ---- test seams (§3.9), compiled only into the test-injection libraries -------------------------

enum class MatvecFaultSite : uint8_t { kQ, kO, kGateUp, kDown };
enum class MatvecFaultKind : uint8_t { kNonPfiStatus, kBadAlloc };

#if defined(SUPERSLM_ENABLE_MATVEC_TEST_SEAMS)
namespace test {
// Moves the minimum work per task the matvec rule reads (default kMinRowBytesPerTask).
void SetMinRowBytesPerTask(size_t bytes) noexcept;
void ResetMinRowBytesPerTask() noexcept;
// Single-shot: the next time either layer loop reaches `site` of layer `layer`, it fails there,
// before that site's GEMM (so after K/V landing for kO, kGateUp and kDown), with a real domain
// rejection (kNonPfiStatus: CarriedScaleMantissaOutOfDomain) or a thrown std::bad_alloc, and the
// seam disarms. kGateUp names the gate projection in the chunk-batched loop.
void ArmLayerSiteFault(MatvecFaultSite site, uint32_t layer, MatvecFaultKind kind) noexcept;
void DisarmLayerSiteFault() noexcept;
}  // namespace test
// The layer loops' call into the fault seam: Ok unless armed for this site and layer.
SslmForwardStatus MaybeLayerSiteFault(MatvecFaultSite site, uint32_t layer);
#else
inline SslmForwardStatus MaybeLayerSiteFault(MatvecFaultSite, uint32_t) noexcept {
	return SslmForwardStatus::Ok;
}
#endif

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
// for a feature this library lacks is told so rather than silently ignored. D1 adds bit 1; the
// batched-prefill bit (bit 0) is added by the step that implements it, not before.
inline constexpr uint32_t kImplementedParallelForBits = SSLM_PARALLEL_FOR_MATVEC;

// The one field-domain check both hook setters apply (sslm_workspace_set_parallel_for and
// sslm_gpu_context_set_host_parallel_for). `pf` is non-null.
inline bool ParallelForHookValid(const sslm_parallel_for& pf) noexcept {
	return (pf.reserved & ~kImplementedParallelForBits) == 0u && pf.max_tasks >= 0 &&
	       pf.max_tasks <= SSLM_PARALLEL_FOR_MAX_TASKS &&
	       !(pf.run == nullptr && pf.max_tasks > 1);
}

}  // namespace superslm

#endif  // SUPERSLM_SRC_FORWARD_PARALLEL_SPLIT_H

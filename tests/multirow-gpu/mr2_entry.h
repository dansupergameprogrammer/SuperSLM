// Multi-row GPU plan (rev 1.3.2), slice MR2 (the recorder): the entry and seams the MR2 red suite drives.
//
// WHAT THIS IS. MR2's "internal entry" (plan §9) is the recorder reached without MR3's public header. The suite
// drives it through exactly the shapes plan §4.1 gives the public verbs, so the cells carry over to MR3 unchanged:
// MR3 publishes the same functions in include/superslm/gpu_1p0.h, and this header then stops declaring them (the
// SSLM_GPU_MAX_ROWS guard below). Only the PROMPT kind exists at MR2. Advancing is the existing
// sslm_decode_step_gpu (dispatch_budget counts DISPATCHES: every open row advances by
// floor(budget / DispatchesPerLayer) whole layers, capped at the chunk's end; below one layer's dispatches it
// returns SSLM_DISPATCH_BUDGET_TOO_SMALL with the chunk unchanged, plan §4.1 / E12), and the close runs in the
// sslm_gpu_ready that finishes the last layer (§3.5).
//
// The seams are plan §8's instruments that MR2's cells read and that no public verb exposes. They are test-only
// (built under SUPERSLM_MULTIROW_WIDE, as MR1's smoke entry is) and decide nothing on their own: every verdict is
// a status, a byte comparison against the one-row twin, or an anchor outside the branch (plan §7 grading rule).
// Each seam is commissioned by its cell's must-reject (a mutant that the seam must see).
//
// CONTRACT FOR THE BUILDER. Implement these names (or tell the test author the names you chose; one edit here).
// Until they exist, the runner links mr2_stub.cpp and compiles the cells with -DMR2_STUB: every cell then reports
// red-unimplemented, and the suite is red.
#pragma once
#include <cstddef>
#include <cstdint>
#include <vector>

#include "superslm/gpu_1p0.h"

#ifndef SSLM_GPU_MAX_ROWS
// Plan §4.1, verbatim (MR3 publishes these; MR2 implements them internally).
#define SSLM_GPU_MAX_ROWS 8
typedef enum {
	SSLM_GPU_ROWS_PROMPT = 0,
	SSLM_GPU_ROWS_SCHEMA_FORCED = 1,
	SSLM_GPU_ROWS_EXACT_DRAFT = 2 /* MR5 */
} SslmGpuRowsKind;
#define SSLM_GPU_ROWS_FIRST_PREWALKED 0x1u
typedef struct SslmGpuRowsSpec {
	uint32_t struct_size;
	int32_t kind;
	const int32_t* tokens;
	int32_t count;
	uint32_t flags;
} SslmGpuRowsSpec;
SslmGpuStatus sslm_gpu_seq_embed_rows(SslmGpuContext*, SslmGpuSequenceHandle*, const SslmGpuRowsSpec*,
                                      int32_t* out_admitted) noexcept;
SslmGpuStatus sslm_gpu_seq_rows_abandon(SslmGpuContext*, SslmGpuSequenceHandle*) noexcept;
SslmGpuStatus sslm_gpu_seq_rows_open(const SslmGpuSequenceHandle*, int32_t* out_rows) noexcept;
SslmGpuStatus sslm_gpu_seq_last_rows_committed(const SslmGpuSequenceHandle*, int32_t* out_rows) noexcept;
#endif

// Plan §8's seams that MR2 reads. Counters are cumulative since context create; a cell reads them before and
// after and grades the difference.
struct SslmGpuMr2Seams {
	uint32_t struct_size = sizeof(SslmGpuMr2Seams);
	uint64_t submissions = 0;             // command lists submitted on this context (submission seam)
	uint64_t widened_lists = 0;           // lists that recorded widened (multi-row) dispatches
	uint32_t last_widened_list_dispatches = 0;  // dispatches in the most recent widened list (dispatches-per-list)
	uint32_t max_widened_list_dispatches = 0;   // the largest widened list since create
	uint64_t allocations = 0;             // device + host allocations the recorder made (allocation seam, 1.2)
	uint32_t row_buffer_sets = 0;         // H-row buffer sets the context owns (free + in use)
	uint32_t row_buffer_free = 0;         // of those, on the free list (row-buffer free-list seam, 1.2)
	uint32_t chunks_open = 0;             // chunks open on this context now
	uint32_t chunks_open_peak = 0;        // the most chunks open at once since create
};
SslmGpuStatus sslm_gpu_test_only_mr2_seams(SslmGpuContext* ctx, SslmGpuMr2Seams* out) noexcept;

// The layer seam for an open chunk: rows M, variant H (1 for M = 1, which stays on the one-row body), the layer
// the rows have reached, and p0. With no chunk open: SSLM_OK with *out zeroed.
struct SslmGpuMr2ChunkState {
	uint32_t struct_size = sizeof(SslmGpuMr2ChunkState);
	uint32_t rows = 0, variant_h = 0, layer = 0;
	int64_t p0 = 0;
};
SslmGpuStatus sslm_gpu_test_only_mr2_chunk_state(const SslmGpuSequenceHandle* seq, SslmGpuMr2ChunkState* out) noexcept;

// The pool-wide DEVICE read-back (4.6b): the whole device K/V buffer of `pool`, read after the context's fences,
// so a write the host mirror never received (a padded row's) is visible. Dedicated sequences: `seq` non-null and
// `pool` null reads that sequence's device K/V buffer.
SslmGpuStatus sslm_gpu_test_only_mr2_device_kv(SslmGpuContext* ctx, SslmGpuKvPoolHandle* pool,
                                               SslmGpuSequenceHandle* seq, std::vector<uint8_t>* out) noexcept;

// 5.3: the next widened slice whose index within its chunk is `slice_k` (0 = the slice that runs layer 0) fails
// its allocation as a live device does (SSLM_GPU_ALLOCATION_FAILED), once. -1 disarms.
SslmGpuStatus sslm_gpu_test_only_mr2_arm_slice_alloc_failure(SslmGpuContext* ctx, int32_t slice_k) noexcept;

// 1.1r (MX8): the number of pages `seq`'s holder maps now (0 for a dedicated sequence), chunk open or not. A rollback
// must leave it where the chunk's entry found it: pool counts cannot show it (a budget reserves its pages up front)
// and HolderUnmapTail returns a page to the holder's reserve, so the next map draws the same page.
SslmGpuStatus sslm_gpu_test_only_mr2_holder_mapped(const SslmGpuSequenceHandle* seq, uint32_t* out_pages) noexcept;

// Multi-row GPU plan (rev 1.3.2), slice MR3 (the public prompt verb): the contract the MR3 red suite drives.
//
// WHAT MR3 PUBLISHES (plan §4.1-§4.6, §3.7). The four verbs MR2 built as its internal entry (sslm_gpu_seq_embed_rows,
// sslm_gpu_seq_rows_abandon, sslm_gpu_seq_rows_open, sslm_gpu_seq_last_rows_committed), with SslmGpuRowsSpec,
// SslmGpuRowsKind, SSLM_GPU_MAX_ROWS and SSLM_GPU_ROWS_FIRST_PREWALKED, declared in include/superslm/gpu_1p0.h.
// (sslm_gpu_seq_rows_produced is MR5's; 2.2 grades it there.) The batch form admits chunk-open members (5.4).
//
// STATUSES ON THIS BASE (plan §3.7). Since R-17 the branch sits on 1.17 (claude/sfd-gpu-1p17 until 1.17.0 is
// tagged), so the cap refusal is 1.17's SSLM_GPU_CONTEXT_CAP_EXCEEDED; 1.20's SSLM_GPU_INVALID_ARGUMENT exists on no
// ref this branch sits on. Until R-20 the verbs return:
//   - a bad spec, a null out pointer, a malformed or foreign handle: SSLM_SEQUENCE_KV_BUFFER_MISMATCH (main's
//     argument-fault status; becomes SSLM_GPU_INVALID_ARGUMENT at R-20);
//   - an adapter change on a later slice (mid-token): SSLM_BUSY (becomes 1.20's mid-token status at R-20);
//   - zero rows admitted on a DEDICATED sequence (no headroom): 1.17's SSLM_GPU_CONTEXT_CAP_EXCEEDED (32), re-pinned
//     at R-17 from the branch-provisional SSLM_GPU_ROWS_CAP_EXCEEDED;
//   - zero rows admitted on a POOLED sequence at its budget limit: SSLM_GPU_KV_BUDGET_EXCEEDED, the one-row pooled
//     path's own limit status (§3.7 makes only the dedicated case provisional; the test author's reading);
//   - save while a chunk is open: SSLM_GPU_ROWS_CHUNK_OPEN, branch-provisional, appended after 1.17's 32 (the name
//     the plan gives; it takes the next free ordinal at R-20).
// The provisional enumerator must be > 32 and marked provisional on its line in the header (the runner's contract
// leg checks the header text; the cells check the values the verbs return).
//
// If the builder chooses another name for either provisional status, it is one edit each below.
//
// Until gpu_1p0.h publishes the verbs and both statuses, the runner compiles the cells with -DMR3_UNPUBLISHED: the
// declarations then come from mr2_entry.h (MR2's internal entry, which the engine still defines under
// SUPERSLM_MULTIROW_WIDE), and the two provisional statuses are sentinels no verb can return, so every check of
// them is red while every behaviour that MR2's entry already has is graded as it is.
#pragma once
#include <cstdint>

#include "mr2_entry.h"
#include "superslm/gpu_1p0.h"

#if defined(MR3_UNPUBLISHED)
#define MR3_CAP_STATUS (static_cast<SslmGpuStatus>(0xFFFF0001u))
#define MR3_CHUNK_OPEN_STATUS (static_cast<SslmGpuStatus>(0xFFFF0002u))
#else
#define MR3_CAP_STATUS (SslmGpuStatus::SSLM_GPU_CONTEXT_CAP_EXCEEDED)
#define MR3_CHUNK_OPEN_STATUS (SslmGpuStatus::SSLM_GPU_ROWS_CHUNK_OPEN)
static_assert(static_cast<uint32_t>(MR3_CAP_STATUS) == 32u, "plan §3.7, R-17: the cap status is 1.17's ordinal 32");
static_assert(static_cast<uint32_t>(MR3_CHUNK_OPEN_STATUS) > 32u, "plan §3.7: CHUNK_OPEN is appended after 1.17's 32");
#endif
// The statuses this base keeps for the cases §3.7 does not make provisional.
#define MR3_ARG_STATUS (SslmGpuStatus::SSLM_SEQUENCE_KV_BUFFER_MISMATCH)
#define MR3_MIDTOKEN_ADAPTER_STATUS (SslmGpuStatus::SSLM_BUSY)
#define MR3_POOLED_LIMIT_STATUS (SslmGpuStatus::SSLM_GPU_KV_BUDGET_EXCEEDED)

// The engine's claim seam (gpu_1p0.cpp, SUPERSLM_GPU_ALLOC_FAULT_INJECTION builds; the 1.13.0 same-handle race
// suite's): the hook runs on the thread that takes a handle's per-handle claim, the moment it takes it. 3.1 parks
// the winner there, so every loser runs while the winner holds the claim (the overlap is certain, not sampled).
void SslmGpuSeqClaimHookForTest(void (*hook)(SslmGpuSequenceHandle* seq, void* user), void* user) noexcept;

// 1.1's device-bytes seam (PROPOSED; the test author's name, one edit here): the bytes of device resources the harness
// holds now, incremented at the counted allocation primitive (harness::Device::TryMakeBuffer and the descriptor heaps)
// and decremented when the resource is freed (e.g. through a private-data IUnknown whose Release subtracts). Test builds
// only (SUPERSLM_GPU_ALLOC_FAULT_INJECTION). The runner defines MR3_HAVE_LIVE_BYTES when src/ defines it.
uint64_t SslmGpuLiveDeviceBytesForTest() noexcept;

// The harness fault seams 5.6 composes (include/superslm/gpu_port.h): the next removed-device query the fault
// classifier makes reports removed, so the armed slice fault (mr2_entry.h) classifies as SSLM_DEVICE_LOST.
void ArmGpuMapDeviceRemovedQueryInjection() noexcept;

// Multi-row GPU plan (rev 1.3.2), slice MR5 (G1-X, Exact on the GPU): the surface the MR5 red suite drives, and the
// seams it needs that the engine does not have yet. Red-first: every name below that the library lacks is supplied
// by mr5_red_stubs.cpp (no behaviour), so the suite builds and runs today and each cell that needs the name is red.
// The runner (mr5_wine.sh) greps src/ and include/ for each name and passes -DMR5_HAVE_<NAME> for those present, so a
// stub is never linked next to the real definition.
//
// THE PUBLIC VERB (plan §4.1, §4.6 rev 1.3 F6). Declared in include/superslm/gpu_1p0.h by MR5, with the other
// multi-row verbs:
//   SslmGpuStatus sslm_gpu_seq_rows_produced(const SslmGpuSequenceHandle* seq, int32_t* out_token) noexcept;
// A null sequence or `out_token`: SSLM_SEQUENCE_KV_BUFFER_MISMATCH. A chunk open: SSLM_BUSY. After an Exact close:
// SSLM_OK with the produced token in *out_token, or -2 at a schema dead end; at a NarrowRowChecked failure (§4.7 step
// 5, cell 7.19) it returns Off's mapped status (MapDecodedStatusToGpuStatus) with *out_token untouched. After any
// other close, or none (a fresh, reset or restored handle, an abandon): SSLM_SEQUENCE_REJECTED, *out_token untouched.
// No claim.
//
// EXACT_DRAFT (kind 2; ruling 26 Q1): row 0 is the pending token, which its finish already walked (after a prompt's
// finish, an Off token's, or an Exact close's produced token), and rows 1.. are the drafts. `flags` stays 0 (bit 0x1,
// the removed FIRST_PREWALKED, is a bad spec). Admission: the longest prefix legal under the walk from the state
// after row 0 (row 0 is never re-checked), capped by headroom and a pooled budget. On a model mapped without a
// device-resident logits head (no SSLM_GPU_RESIDENCY_HEAD_ON_DEVICE): SSLM_SEQUENCE_REJECTED, nothing changed
// (ruling 26 Q2). The close: CommitPrefix(a, a - 1); the produced token is row a - 1's masked argmax.
//
// THE SEAMS (test builds only: SUPERSLM_GPU_TEST_SEAMS). The test author's names; one edit here if the builder names
// them otherwise. Each is extern "C", single-shot, and fires on the Off path and on the Exact close alike, so a cell
// can arm the same fault on the one-row twin and on the chunk.
//   S-GSATROW (plan SFD §11.0; 7.10, GX.2, XG6, MX19, MX23). The GPU twin of the CPU's S-SATROW (ArmSatRow):
//     the next row computed at index `row` of the next multi-row pass (an embed_rows chunk or a jump pass), or row 0
//     of the next one-row token when `row` is 0, adds one to each of its five saturation counters' deltas (kv_sat,
//     kv_landing, k_channel, rope_q, rope_k), as a real saturation at that row would. Consumed when it fires.
//       extern "C" void ArmGpuSatRowForTest(int32_t row) noexcept;          // -1 disarms
//       extern "C" int32_t GpuSatRowArmedForTest() noexcept;                // 1 while armed, 0 once consumed
//   S-GNARROW (7.19). The n-th finish row from now (0 = the next one) has its NarrowRowChecked fail with
//     SslmForwardStatus::LogitNarrowingOverflow, before any flag write, as a real out-of-domain row would. A finish
//     row is one Off finish (the decode bridge's or SslmGpuSeqFinishTokenForG5Bridge's), or one row of an Exact
//     close's per-row finish, in row order. On a model whose bound finish scores legal rows on the host
//     (LogitsSiteLegalRows), the fault fires at that path's narrowing.
//       extern "C" void ArmGpuFinishNarrowFaultForTest(int32_t occurrence) noexcept;  // -1 disarms
//   S-GREADBACK (GX.8 (b)). A fault in the read-back of row `row` of the next jump pass after the first (the second
//     pass is 1): the pass fails as a device fault there.
//       extern "C" void ArmGpuExactReadbackFaultForTest(int32_t pass, int32_t row) noexcept;  // pass < 0 disarms
//   S-GVERIFY (GX.11). While on, the Exact check treats every draft row as matching (the CPU's S-VERIFY "always
//     equal"), on the jump verb and on an EXACT_DRAFT close.
//       extern "C" void SetGpuExactVerifyAlwaysEqualForTest(int32_t on) noexcept;
//   S-GEXACTPASSES (GX.13, 7.10). Counts the Exact passes the jump verb ran since the last reset, and the rows
//     the largest of them computed.
//       extern "C" void GpuExactPassLogResetForTest() noexcept;
//       extern "C" int32_t GpuExactPassCountForTest() noexcept;
#pragma once
#include <cstdint>

#include "superslm/gpu_1p0.h"

#if !defined(MR5_HAVE_ROWS_PRODUCED)
SslmGpuStatus sslm_gpu_seq_rows_produced(const SslmGpuSequenceHandle* seq, int32_t* out_token) noexcept;
#endif
// The stub's status: no verb returns it, so every check of rows_produced is red until the verb exists.
#define MR5_STUB_STATUS (static_cast<SslmGpuStatus>(0xFFFF0005u))

extern "C" void ArmGpuSatRowForTest(int32_t row) noexcept;
extern "C" int32_t GpuSatRowArmedForTest() noexcept;
extern "C" void ArmGpuFinishNarrowFaultForTest(int32_t occurrence) noexcept;
extern "C" void ArmGpuExactReadbackFaultForTest(int32_t pass, int32_t row) noexcept;
extern "C" void SetGpuExactVerifyAlwaysEqualForTest(int32_t on) noexcept;
extern "C" void GpuExactPassLogResetForTest() noexcept;
extern "C" int32_t GpuExactPassCountForTest() noexcept;
// 1 when the seam above is the stub (mr5_red_stubs.cpp): the cell that needs it reports "unimplemented" and is red.
extern "C" int32_t Mr5StubbedSeamsForTest() noexcept;

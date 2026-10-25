// Multi-row GPU plan (rev 1.3.2), slice MR1: the test-only host side of the row-widened kernels, declared
// for the one-layer smoke (MR1's gate: "Wine one-layer smoke equal on pkv_def and pkv_qk"). Everything here
// exists only in a build that defines SUPERSLM_MULTIROW_WIDE; no shipped target defines it. The shipped library
// is not byte-identical to the base: MR1 adds behaviour-neutral hooks to it (FillComposedRootParams, the bound-
// resource fields of GpuLayerLoopChunkOpenState, and PackGpuSeqStateBytes, a new external symbol), and none of
// the widened code. The widened .cso are built and staged only under the CMake option
// SUPERSLM_MULTIROW_WIDE_SHADERS (default OFF), so the default install tree is the base's. The recorder, the chunk state and the internal entry are MR2's (plan §3.3 to
// §3.6), not this file's.
//
// Definitions: superslm_gpu.cpp (PackGpuSeqStateBytes), multirow_wide.inl (included at the end of
// superslm_gpu.cpp) and gpu_1p0.cpp (the smoke entry, which reaches the handles' internals).
#pragma once

#if defined(SUPERSLM_MULTIROW_WIDE)

#include <cstddef>
#include <cstdint>
#include <vector>

#include "superslm/gpu_1p0.h"
#include "superslm/gpu_port.h"

namespace superslm_gpu {

// The SeqState bytes a one-row submission uploads for `seq` (superslm_gpu.cpp).
std::vector<uint8_t> PackGpuSeqStateBytes(const superslm::SequenceLayerState& seq, uint32_t H);

// What one widened submission leaves: every row's SeqState as it entered (`entry_rows`) and as the fence
// left it (`rows`), variant_h rows of row_bytes each, padded rows included, and the whole device K/V buffer
// the submission bound, read back after its fence.
struct GpuWideRowsReadback {
	size_t row_bytes = 0;
	std::vector<uint8_t> entry_rows;
	std::vector<uint8_t> rows;
	std::vector<uint8_t> kv;
};

// Records layers [rows[0].layer_index, + layers) for `real_rows` consecutive rows of one sequence as ONE
// submission of the widened kernels at variant `variant_h` (2, 4 or 8; real_rows <= variant_h), waits for
// it, and reads back. Row r enters as rows[r] (its own SeqState, packed as a one-row submission packs it;
// row r's position is rows[0].context_length + r) and rows past real_rows are padded rows (sticky 99). The
// binding arguments are RunLayerLoopGpuSubmit's, and the guard ladder, residency and bindings are
// PrepareGpuLayerLoopChunkOpenState's, run once for row 0 with coverage for real_rows rows. The sequence's
// host state is not touched: the caller compares and discards. Synchronous; test-only.
superslm::SslmForwardStatus RunMultirowWideLayersForSmoke(
    const superslm::SequenceLayerState* rows, uint32_t real_rows, uint32_t variant_h, uint32_t layers,
    uint32_t num_hidden_layers, size_t hidden_size, size_t head_dim, size_t num_key_value_heads,
    size_t intermediate_size, int64_t context_cap, uint8_t* workspace, size_t workspace_size,
    ID3D12Resource* external_kv_resident, bool* io_external_kv_needs_resume_barrier,
    ID3D12Resource* external_weights_resident, ID3D12Resource* external_rope_cos_resident,
    ID3D12Resource* external_rope_sin_resident, bool external_rope_has, uint64_t external_rope_cos_elems,
    uint64_t external_rope_sin_elems, const GpuAdapterBridge* adapter_bridge, size_t q_width,
    uint64_t model_generation, bool model_has_qk_norm, const GpuPagedKvBinding* paged_kv,
    GpuWideRowsReadback* out);

}  // namespace superslm_gpu

// The MR1 one-layer smoke's result (gpu_1p0.cpp).
struct SslmGpuMr1WideSmokeReport {
	uint32_t real_rows = 0, variant_h = 0;
	int64_t p0 = 0;
	uint32_t rows_equal = 0;          // real rows whose widened SeqState equals one-row's, byte for byte
	uint32_t padded_rows_inert = 0;   // padded rows whose SeqState left the submission as it entered
	uint64_t kv_row_bytes_compared = 0;
	uint64_t kv_row_bytes_differing = 0;  // in the real rows' K/V rows, widened against one-row
	uint64_t kv_stray_bytes = 0;          // K/V bytes outside the real rows' rows the widened run changed
	char detail[512] = {};
};

// MR1 one-layer smoke (test-only). `ref` and `wide` are two Idle sequences of one model at the same
// context length p0 with identical K/V history (the caller fed both the same tokens to full depth), each at
// a token boundary. For each of the `real_rows` tokens, `ref` runs one-row: embed, the first layer alone
// (one sslm_decode_step_gpu at DispatchesPerLayer, drained), then the rest of the token. Each row's entry
// state and its state after the first layer are kept. `wide` then runs the first layer for all rows as one
// widened submission at `variant_h` (RunMultirowWideLayersForSmoke), and the report compares every real
// row's SeqState, every real row's first-layer K/V rows, every padded row, and every other K/V byte. With
// `adapter` non-null both sequences decode with it. SSLM_OK when every call succeeded, whatever the
// comparison found; the report says whether the two agree. `wide` is left unusable for further decoding
// (its device K/V holds rows its host state does not): release it.
SslmGpuStatus sslm_gpu_test_only_mr1_wide_one_layer_smoke(SslmGpuContext* ctx, SslmGpuSequenceHandle* ref,
                                                          SslmGpuSequenceHandle* wide,
                                                          const SslmGpuAdapterHandle* adapter,
                                                          const int32_t* row_tokens, uint32_t real_rows,
                                                          uint32_t variant_h,
                                                          SslmGpuMr1WideSmokeReport* out) noexcept;

#endif  // SUPERSLM_MULTIROW_WIDE

// Multi-row GPU plan (rev 1.3.2): the host side of the row-widened kernels.
//
// MR2 (plan §3.3 to §3.6) built the recorder (one widened submission per layer slice), the per-chunk row buffers
// and their per-context free list, the close (CommitPrefix), and the entry snapshot and rollback. MR3 (§4.1 to
// §4.6) publishes the verbs that reach it (include/superslm/gpu_1p0.h: sslm_gpu_seq_embed_rows,
// sslm_gpu_seq_rows_abandon, sslm_gpu_seq_rows_open, sslm_gpu_seq_last_rows_committed) and builds the widened
// .cso beside the one-row shaders, so the recorder is part of the shipped library. Two things stay test-only,
// under SUPERSLM_MULTIROW_WIDE (the multi-row test-seam build, which no shipped target defines): MR1's one-layer
// smoke and plan §8's seams.
//
// Definitions: superslm_gpu.cpp (PackGpuSeqStateBytes), multirow_wide.inl (included at the end of
// superslm_gpu.cpp: the widened submissions and their finish) and gpu_1p0.cpp (the chunk on the handle, the
// close, rollback and the verbs, which reach the handles' internals; and the test-only smoke and seams).
#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <vector>

#include <wrl/client.h>

#include "multirow_rows_host.h"
#include "superslm/gpu_1p0.h"
#include "superslm/gpu_port.h"

namespace superslm_gpu {

// The SeqState bytes a one-row submission uploads for `seq` (superslm_gpu.cpp).
std::vector<uint8_t> PackGpuSeqStateBytes(const superslm::SequenceLayerState& seq, uint32_t H);

#if defined(SUPERSLM_MULTIROW_WIDE)
// MR1 (test-only): what one widened submission leaves: every row's SeqState as it entered (`entry_rows`) and as the fence
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
#endif  // SUPERSLM_MULTIROW_WIDE


// ---- MR2: the recorder (plan §3.3 to §3.5) ----

// The largest chunk (plan §4.1: SSLM_GPU_MAX_ROWS).
constexpr uint32_t kGpuRowsMaxRows = 8;

// The variant a chunk of `rows` real rows runs at (plan §3.1): 1 for M = 1 (today's one-row body), 2, 4 or 8
// otherwise, the next one up.
constexpr uint32_t GpuRowsVariantFor(uint32_t rows) { return rows <= 1 ? 1u : rows <= 2 ? 2u : rows <= 4 ? 4u : 8u; }

// What PrepareGpuLayerLoopChunkOpenState allocates on every call, kept for the widened slices (cell 1.2: no
// per-slice allocation after warm-up). The five constant uploads and the one-page table upload are reused while
// their bytes are unchanged; the one-row SeqState, LayerScratch and WorkScratch buffers, which no widened shader
// reads (BindWideRoot binds them only so every root descriptor is a valid address), are reused while their size
// is unchanged. Held by the row set, so it moves with it through the context's free list.
struct GpuWidePrepareReuse {
	struct Upload {
		Microsoft::WRL::ComPtr<ID3D12Resource> buf;
		std::vector<uint8_t> bytes;
	};
	Upload layout, rope, model_const, silu, scratch_layout, page_table;
	Microsoft::WRL::ComPtr<ID3D12Resource> seq_uav, scratch_uav, work_scratch_uav;
};

// One chunk's row buffers (plan §3.4, SF3): kGpuRowsMaxRows rows of each per-row buffer, so one set serves every
// variant; the shader-visible table of their views; the read-backs; and the upload window the first slice
// writes the rows' entry states through. Taken from the context's free list when a chunk's first slice is
// recorded, returned to it when the chunk closes, so after warm-up no slice allocates (cell 1.2): `prepare`
// holds what Prepare would otherwise allocate on each slice.
struct GpuRowSet {
	Microsoft::WRL::ComPtr<ID3D12Resource> seq_rows, scratch_rows, work_rows, seq_readback, kv_readback, init_upload;
	Microsoft::WRL::ComPtr<ID3D12DescriptorHeap> heap;
	uint64_t seq_bytes = 0, scratch_bytes = 0, work_bytes = 0, kv_readback_bytes = 0;
	uint64_t seq_stride = 0, scratch_stride = 0, work_stride = 0;
	// Where seq_rows rests between submissions (COPY_DEST when new, COPY_SOURCE after every slice's read-back).
	D3D12_RESOURCE_STATES seq_rows_state = D3D12_RESOURCE_STATE_COPY_DEST;
	GpuWidePrepareReuse prepare;
	bool Fits(uint64_t seq, uint64_t scratch, uint64_t work, uint64_t kv) const noexcept {
		return seq_bytes >= seq && scratch_bytes >= scratch && work_bytes >= work && kv_readback_bytes >= kv;
	}
};
// The device and host allocations one GpuRowSet takes (six buffers and one descriptor heap): the allocation
// seam's unit.
constexpr uint64_t kGpuRowSetAllocations = 7;

// The five per-row counter deltas a row's SeqState carries (plan §3.5, E16: every row starts at 0).
enum GpuRowsCounter : uint32_t {
	kGpuRowsKvSaturation = 0,
	kGpuRowsKvLandingSaturation,
	kGpuRowsKChannelLandingSaturation,
	kGpuRowsRopeQSaturation,
	kGpuRowsRopeKSaturation,
	kGpuRowsCounterCount
};

// One real row's SeqState as a slice's read-back left it.
struct GpuRowsRowState {
	std::vector<int8_t> hidden_codes;
	superslm::CarriedScale hidden_scale{};
	uint32_t layer_index = 0;
	int64_t context_length = 0;
	uint64_t counters[kGpuRowsCounterCount] = {};
	int64_t sticky = 0;
};

// One layer slice of a chunk: layers [start_layer, start_layer + layers) for real_rows rows at variant_h.
struct GpuRowsSlice {
	// The first slice only: the real rows' entry states (row r at position p0 + r, counters 0).
	const superslm::SequenceLayerState* entry_rows = nullptr;
	uint32_t real_rows = 0, variant_h = 0, start_layer = 0, layers = 0;
	int64_t p0 = 0;
	bool first_slice = false;
	// The slice that reaches the last layer: it also reads back every real row's K/V rows at every layer, for
	// the close to choose from (plan §3.5 step 1).
	bool close_slice = false;
	// Cell 5.3's seam: this slice fails its allocation as a live device does.
	bool inject_allocation_failure = false;
};

// What recording a slice did, for the context's seams (plan §8).
struct GpuRowsSliceRecord {
	uint32_t dispatches = 0;      // dispatches the slice's one list recorded
	uint64_t allocations = 0;     // allocations the recorder made for it (a new or regrown row set)
	bool row_set_created = false; // *io_row_set was null and now holds a new set
};

// Records one slice as ONE submission of the widened kernels and submits it without waiting (the one-row
// path's Submit half). The guard ladder, residency and bindings are PrepareGpuLayerLoopChunkOpenState's, run
// once for row 0 with coverage for real_rows rows. *io_row_set is the chunk's row set: created when null,
// regrown when too small for this model. On Ok, *out_inflight is the submission; the rows live on in the row
// set between slices. Nothing of the sequence's host state is touched.
superslm::SslmForwardStatus RunRowsSliceGpuSubmit(
    const GpuRowsSlice& slice, std::unique_ptr<GpuRowSet>* io_row_set, uint32_t num_hidden_layers,
    size_t hidden_size, size_t head_dim, size_t num_key_value_heads, size_t intermediate_size, int64_t context_cap,
    uint8_t* workspace, size_t workspace_size, ID3D12Resource* external_kv_resident,
    bool* io_external_kv_needs_resume_barrier, ID3D12Resource* external_weights_resident,
    ID3D12Resource* external_rope_cos_resident, ID3D12Resource* external_rope_sin_resident, bool external_rope_has,
    uint64_t external_rope_cos_elems, uint64_t external_rope_sin_elems, const GpuAdapterBridge* adapter_bridge,
    size_t q_width, uint64_t model_generation, bool model_has_qk_norm, const GpuPagedKvBinding* paged_kv,
    GpuInFlightToken* out_inflight, GpuRowsSliceRecord* out_record);

// The Finish half: waits (or, with block 0, returns Ok and *out_ready 0 while the fence is pending), publishes
// the call's timing, and reads back every real row's SeqState into *out_rows. A close slice's K/V rows come
// back in *out_kv_rows, head_dim bytes each, with *out_kv_offsets their offsets in the host mirror the
// submission's view addresses, ordered row, layer, K/V head, then K before V. Touches no host state: the close
// decides what to copy (plan §3.5). Ok, or the fault classification's status.
superslm::SslmForwardStatus RunRowsSliceGpuFinish(GpuInFlightToken&& inflight, const GpuRowSet& row_set,
                                                  uint32_t real_rows, int32_t block, int32_t* out_ready,
                                                  std::vector<GpuRowsRowState>* out_rows,
                                                  std::vector<size_t>* out_kv_offsets,
                                                  std::vector<uint8_t>* out_kv_rows);

// The sticky tag's status (superslm_gpu.cpp's DecodeStickyTag, which the one-row finish returns).
superslm::SslmForwardStatus DecodeRowsStickyTag(int64_t tag);

// The context's recorder state (plan §3.4, §8): the row-set free list and open-chunk count (the portable
// rows_host::RowSetPool, multirow_rows_host.h, which cell 3.3 runs under TSan), and the seams' counters under `mu`.
struct GpuRowsContextState {
	rows_host::RowSetPool<GpuRowSet> pool;
	std::mutex mu;  // the counters below
	uint64_t widened_lists = 0, allocations = 0;
	uint32_t last_widened_list_dispatches = 0, max_widened_list_dispatches = 0;
	std::atomic<int32_t> armed_failure_slice{-1};  // set only through the test-only seam (cell 5.3)
};

// A chunk open on a sequence handle (plan §3.4): the rows, the entry snapshot (§3.6), the adapter fixed at the
// slice that runs layer 0 (§4.4), the slices recorded so far, and the row set while one is held.
struct GpuRowsEntrySnapshot {
	std::vector<int8_t> hidden_codes;
	superslm::CarriedScale hidden_scale{};
	uint32_t layer_index = 0;
	int64_t context_length = 0;
	uint64_t counters[kGpuRowsCounterCount] = {};
	int32_t bound_schema_index = -1;
	uint32_t dfa_walk_state = 0;
	bool ready_for_logits = false;
	bool bind_eligible = false;
	bool table_dirty = false;
	uint32_t holder_mapped = 0;
	// M = 1 only (the one-row body writes the mirror at each slice's finish): row p0's mirror bytes at every
	// layer, as (offset, head_dim bytes) pairs, taken before the first slice.
	std::vector<size_t> mirror_offsets;
	std::vector<uint8_t> mirror_bytes;
	bool mirror_taken = false;
};
struct GpuRowsChunk {
	uint32_t rows = 0, variant_h = 0;
	int32_t kind = 0;
	uint32_t flags = 0;
	int64_t p0 = 0;
	std::vector<int32_t> tokens;
	std::vector<std::vector<int8_t>> row_codes;
	std::vector<superslm::CarriedScale> row_scales;
	// SCHEMA_FORCED (MR4): the walk state after row r is committed (row 0 under FIRST_PREWALKED leaves the entry's);
	// the close sets the walk to walk_after[k - 1] for k committed rows. Empty for PROMPT.
	std::vector<uint32_t> walk_after;
	// Exact (MR5, plan §4.7): an internal jump pass's check tokens, row r against verify_next[r] (-1: the jump's
	// last row, not checked), and whether its read-back is to fail (S-GREADBACK). Empty / false otherwise.
	std::vector<int32_t> verify_next;
	bool readback_fault = false;
	// Exact (MR5): the chunk's real rows' device K/V rows at the first slice (offsets in the mirror's layout, row
	// major as RowsMirrorOffsetsAt orders one row's), so the close puts the rejected rows back.
	std::vector<size_t> device_kv_offsets;
	Microsoft::WRL::ComPtr<ID3D12Resource> device_kv_snap;
	GpuRowsEntrySnapshot entry;
	const SslmGpuAdapterHandle* adapter = nullptr;
	bool adapter_fixed = false;
	uint32_t layers_done = 0;
	uint32_t slice_layers_in_flight = 0;
	bool slice_in_flight = false;
	std::vector<uint32_t> slice_ends;  // the layer each finished slice ended at, in order
	std::unique_ptr<GpuRowSet> row_set;
};

}  // namespace superslm_gpu

#if defined(SUPERSLM_MULTIROW_WIDE)
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

// ---- Plan §8's seams (test-only; the MR2/MR3 cells read them) ----
//
// The seams (the test author's contract, tests/multirow-gpu/mr2_entry.h; the layouts must stay identical).
struct SslmGpuMr2Seams {
	uint32_t struct_size = sizeof(SslmGpuMr2Seams);
	uint64_t submissions = 0;
	uint64_t widened_lists = 0;
	uint32_t last_widened_list_dispatches = 0;
	uint32_t max_widened_list_dispatches = 0;
	uint64_t allocations = 0;
	uint32_t row_buffer_sets = 0;
	uint32_t row_buffer_free = 0;
	uint32_t chunks_open = 0;
	uint32_t chunks_open_peak = 0;
};
// `submissions` counts the process's submission device (every context submits through it), so a cell grades
// a difference; the other counters are this context's since create.
SslmGpuStatus sslm_gpu_test_only_mr2_seams(SslmGpuContext* ctx, SslmGpuMr2Seams* out) noexcept;

struct SslmGpuMr2ChunkState {
	uint32_t struct_size = sizeof(SslmGpuMr2ChunkState);
	uint32_t rows = 0, variant_h = 0, layer = 0;
	int64_t p0 = 0;
};
SslmGpuStatus sslm_gpu_test_only_mr2_chunk_state(const SslmGpuSequenceHandle* seq, SslmGpuMr2ChunkState* out) noexcept;

SslmGpuStatus sslm_gpu_test_only_mr2_device_kv(SslmGpuContext* ctx, SslmGpuKvPoolHandle* pool,
                                               SslmGpuSequenceHandle* seq, std::vector<uint8_t>* out) noexcept;

SslmGpuStatus sslm_gpu_test_only_mr2_arm_slice_alloc_failure(SslmGpuContext* ctx, int32_t slice_k) noexcept;

// 1.1r (MX8): the pages `seq`'s holder has mapped now (kv_pages::HolderMapped); 0 for a dedicated sequence.
// A rollback that skips HolderUnmapTail leaves this above the entry's count while pool stats stay equal.
SslmGpuStatus sslm_gpu_test_only_mr2_holder_mapped(const SslmGpuSequenceHandle* seq, uint32_t* out_pages) noexcept;

#endif  // SUPERSLM_MULTIROW_WIDE

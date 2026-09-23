#pragma once
// SuperSLM GPU-serial port — the lower-level GPU submission and layer-loop surface that
// include/superslm/gpu_1p0.h's public API is built on top of: dispatch-budget accounting,
// asynchronous per-sequence lifecycle, and the layer-loop guard ladder that maps a rejected
// numeric/structural check to the correct status without conflating it with a real device
// failure.
//
// Every function this header declares has its definition in src/gpu/superslm_gpu.cpp.
//
// Function groups below are named by the build step whose contract they discharge.

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "superslm/checked_chain_funnel.h"
#include "superslm/forward_sites.h"

// Forward-declared at GLOBAL scope (matching where d3d12.h itself declares
// the real COM interface) rather than pulling <d3d12.h> into every translation unit that
// includes this header (many are CPU-only test/tool files) -- an incomplete COM interface
// type is sufficient for a pointer parameter; the .cpp definitions that actually
// dereference it already include d3d12_harness.h, which itself includes <d3d12.h> before
// this header, so this forward declaration and the real one refer to the identical type.
struct ID3D12Resource;

namespace superslm_gpu {

// --- B1 (Sec7.1, Sec11 B1): per-primitive int64/128-bit battery. One GPU dispatch
// per call, reading back the primitive's own output type. Oracle (in the red suite):
// the real, shipped CPU function of the same name, called directly. ---
int64_t DynamicScaleReciprocalGpu(int64_t dn);
// Returns the RequantTokenCodeWide narrowed int8 code, widened back to int64_t for
// a uniform comparison surface across every primitive in this battery.
int64_t RequantTokenCodeWideGpu(int64_t x_i, int64_t r, int s);
bool BiasReconcileWideGpu(int64_t b, int64_t q_b, int64_t r_a, int64_t e_a, int64_t* out);
superslm::CarriedScale CombineCarriedScaleGpu(superslm::CarriedScale a, superslm::CarriedScale b);

// --- B1 (Sec7.2, Sec11 B1): standalone signed-right-shift battery. The three named
// sites (int32 arithmetic shift, int64 arithmetic shift, the 128-bit SShrFull
// cross-word shift) are internal to intmath.cpp with no public symbol this header
// can cite without inventing one the CPU side does not have -- grep of this tree at
// main@727e63e finds no public `SShrFull` (Curie casebook, Sec2). This battery's own
// negative-operand residual (O-3, Sec7.2: DynamicScaleReciprocal's seed `>> 31`) is
// realized as a fixture on `DynamicScaleReciprocalGpu` above rather than a separate
// symbol -- named explicitly in the casebook rather than silently dropped.

// --- B5 (Sec5.5, Sec11 B5): the two-schedule int64 abs-max reduction (SCHEME 0
// sequential, SCHEME 1 shared-memory binary tree) -- SCHEME 2/3 are not built for
// this operator/width (Sec5.5's own ruling) and carry no symbol here. Oracle: the
// real, shipped MaxAbsReduceWide. ---
int64_t MaxAbsReduceWideGpuScheme0(const int64_t* x, size_t n);
int64_t MaxAbsReduceWideGpuScheme1(const int64_t* x, size_t n);

// --- B2 (Sec6, Sec11 B2): the guard-path port, isolated tier -- the per-dispatch
// status word a standalone guard kernel would write, before any site integrates it.
// Covers the three genuine rejecting guards (RoundingDivideByPot's composed-exponent
// check, RequantChainChecked's mantissa/d_prime checks) and ApplyBiasReconcileRow's
// own compound predicate (Sec11 B2 (3)), for k_bias and v_bias
// independently. ---
superslm::SslmForwardStatus CheckRoundingDivideByPotExponentDomainGpu(int64_t q_B, int64_t e_a);
superslm::SslmForwardStatus CheckBiasAccumulateMagnitudeDomainGpu(int64_t acc_i, int64_t b,
                                                                   int64_t q_b, int64_t r_a,
                                                                   int64_t e_a);
superslm::ChainResult RequantChainCheckedGpu(const int64_t* wide_row, size_t n,
                                              const superslm::CarriedScale* incoming,
                                              size_t n_incoming,
                                              superslm::CarriedScale site_constant);

// --- B7/B11 (Sec5.6, Sec11 B7/B11): the composed, device-resident, multi-layer
// forward -- the sequence-level sticky word, kv_proj's fused guard-combinator and
// split-word saturation counter, and RoPE's staged K commit, all included. Mirrors
// RunLayerLoop's own signature and SequenceLayerState contract exactly (Sec10 dim 7:
// "the CPU path remains the normative oracle" the GPU port is checked against, never
// the reverse) -- a caller supplies the identical inputs RunLayerLoop takes and reads
// back the identical struct shape. This is also B8's own per-token driver (a decode
// session is N calls at layer_budget == num_hidden_layers) and B4's per-site proof
// composes through it once every site lands. ---
// (design Sec5.3/Sec10 B3): the trailing pair below is ADDITIVE -- every
// existing caller (~40 call sites across tests/test_main.cpp, tools/t2039_c5_harness.cpp,
// tools/t2100_gpu_throughput.cpp) passes neither argument and defaults them to nullptr,
// which reproduces this function's PRE-B3 behavior byte-for-byte (the process-global
// g_resident_kv single-slot cache, unchanged). `external_kv_resident`, when non-null,
// is a caller-owned, ALREADY-ALLOCATED DEFAULT-heap UAV buffer (an SslmGpuSequenceHandle's
// own dedicated K/V residency, design Sec5.3) this call binds directly as kv_uav --
// bypassing g_resident_kv and its pointer/size fast-hit check entirely for this call, so
// two sequences each passing their OWN buffer here never share, alias, or evict each
// other's K/V state, closing this class for the K/V cache by construction
// (the same closure B2 already gave weight residency). `io_external_kv_needs_resume_barrier`
// is the caller-owned per-sequence latch this function reads/writes to track whether ITS
// buffer was left in COPY_SOURCE by the PRIOR call on this same handle (mirroring
// kv_fast_hit's own resume-barrier shape below, now scoped to one handle instead of one
// global slot): false on a sequence's first call (the buffer was created directly in
// UNORDERED_ACCESS state, design Sec5.3/Sec10 B3, no resume needed), set true at the end
// of every call that used this path (this call's own targeted readback always leaves the
// buffer in COPY_SOURCE), read (and consumed) at the top of the NEXT call on the same
// handle. `workspace`/`workspace_size` remain required in the external-buffer path too --
// still validated by this function's own existing size guard, and the targeted per-call
// readback still scatters into `workspace` exactly as it does today, so a caller in this
// mode gets the identical CPU-oracle-comparable host mirror the pre-existing path always
// produced (never merely a GPU self-consistency proof).
// (T-2577, D-SLM6278; T-2578 correction): the trailing `model_generation` parameter is the
// caller's own model identity for the read-only weight/RoPE caches and one conjunct of the
// mutable K/V cache key (`g_resident_weights`/`g_resident_kv`/`g_resident_rope`,
// `superslm_gpu.cpp`) -- an opaque value that MUST change whenever the model
// backing `layers`/`rope_tables`/`workspace` changes and MUST NOT change across sequences of the
// same still-live model (a per-load counter, or the artifact's own integrity hash reduced to 64
// bits, are both sound). Defaults to `0`, the "not supplied" sentinel: every pre-T-2577 caller
// compiles and behaves unchanged (`!fresh_sequence` alone gates the fast path, exactly as before
// this ticket). A caller that supplies a nonzero value gets the cheaper, identity-keyed path
// instead -- a fresh sequence of the SAME model is then a legitimate weight/RoPE cache hit,
// closing the ~19 ms/sequence cost the external review's Significant 1 measured
// (`Claude/Poirot/5fafd98-t2573-trackb-external-fold-review.md`) without reopening the recycled-
// host-address hazard T-2576 closed (a fresh sequence of a DIFFERENT model, even one that
// recycled the same address, still misses, because the identity does not match). K/V still
// requires `!fresh_sequence`: model identity cannot identify mutable sequence history.
superslm::SslmForwardStatus RunLayerLoopGpu(superslm::SequenceLayerState& seq,
                                             const superslm::LayerWeights* layers,
                                             uint32_t num_hidden_layers, uint32_t layer_budget,
                                             size_t hidden_size, size_t head_dim,
                                             size_t num_key_value_heads, size_t intermediate_size,
                                             int64_t context_cap,
                                             const superslm::SslmTensorManifest& rope_tables,
                                             uint8_t* workspace, size_t workspace_size,
                                             ID3D12Resource* external_kv_resident = nullptr,
                                             bool* io_external_kv_needs_resume_barrier = nullptr,
                                             uint64_t model_generation = 0,
                                             bool model_has_qk_norm = false);

// (design Sec4.2/Sec4.3/Sec6.2/Sec10 B5): the async submission boundary.
// `RunLayerLoopGpu` above is UNCHANGED -- every one of its existing ~40 callers still
// gets the identical synchronous, blocking call it always did. It is now IMPLEMENTED as
// `RunLayerLoopGpuSubmit` immediately followed by a BLOCKING `RunLayerLoopGpuFinish` --
// the split below reorganizes WHERE the fence-wait happens without moving a single GPU
// instruction relative to where it already ran. The 1.0 API's own `sslm_decode_step_gpu`
// calls only Submit; `sslm_gpu_ready` calls only Finish, with `block=0` (poll) or
// `block=1` (the one case design Sec4.2 permits a fence-wait outside this pair -- the
// caller asked for it).
//
// GpuLayerLoopInFlight is opaque here (fully defined in superslm_gpu.cpp, the only TU
// that touches its fields) -- the same handle-opacity convention gpu_1p0.cpp already
// uses for SslmGpuContext/SslmGpuModelHandle/SslmGpuSequenceHandle. It owns exactly the
// state the FINISH half needs that only exists once the SUBMIT half has recorded and
// closed the command list: the two already-allocated GPU readback buffers (valid to
// READ only once the fence this token names has signaled), the K/V row-offset table the
// scatter reads, and the small dimension/count fields the scatter needs. A caller owns
// the token from the moment Submit sets `*out_inflight` until the SAME caller's own
// Finish call consumes and frees it, exactly once -- Finish always deletes it on the
// path that actually performs the readback (fence signaled, or `block=1`); a `block=0`
// poll against an unsignaled fence returns the token UNCHANGED (still owned by the
// caller, pollable again).
struct GpuLayerLoopInFlight;

// SUBMIT: identical work RunLayerLoopGpu always did, through ExecuteCommandLists/Signal
// -- NEVER a fence-wait (design Sec6.2: "records ... submits ... and returns without
// waiting for the fence"). Two outcomes:
//   - a guard rejects, or an exception is thrown during recording (before Signal is
//     ever reached): the rejecting SslmForwardStatus is returned directly and
//     `*out_inflight` is left null -- nothing was submitted, `seq`/`workspace`
//     untouched, exactly RunLayerLoopGpu's own existing contract for the identical
//     inputs.
//   - recording+submission succeeds: returns `superslm::SslmForwardStatus::Ok` (the
//     CALL-LEVEL "did this proceed" signal, design Sec4.3's own two-channel split --
//     NOT yet the decoded per-call result, which exists only once the fence the
//     returned `*out_inflight` names has signaled) and `*out_inflight` is set to a
//     freshly heap-allocated token the caller passes to RunLayerLoopGpuFinish exactly
//     once.
//
// The trailing six parameters are the model/RoPE analogue of the existing
// `external_kv_resident` pair above (B3), added here (B5) so a caller routed
// through the 1.0 API's own model handle (design Sec5.1, `SslmGpuModelHandle`) never
// touches the pre-1.0 `g_resident_weights`/`g_resident_rope` process-global caches at
// all -- this is what retires `g_resident_rope` for the handle-routed path: when `external_weights_resident` is non-null, `layers` is NEVER
// dereferenced (the caller's own resident buffer is bound directly, no re-pack, no
// re-upload, no g_resident_weights lookup); when BOTH `external_rope_cos_resident`/
// `external_rope_sin_resident` are non-null, `rope_tables` is never read for the two
// big tables (only `external_rope_has`/`external_rope_cos_elems`/
// `external_rope_sin_elems` decide the small per-call RopeInfo presence/size fields a
// caller with no live `SslmTensorManifest` of its own can still supply directly) and
// `g_resident_rope` is never touched. Every existing caller passes none of the six and
// observes byte-for-byte pre-B5 behavior.
// (design Sec5.2/Sec8/Sec10 B6): one covered (layer, projection) slot of a
// mapped adapter's own device residency -- byte offsets into the adapter handle's
// `lora_ab_buf` (a_offset: lora_A bytes, rank*in_channels int8; b_offset: lora_B bytes,
// out_channels*rank int8) and `fold_buf` (fold_offset: DeltaFoldScales rows
// (out_channels*12 bytes: identity/mult/exponent int32 triples) immediately followed by
// UFoldScales rows (rank*12 bytes, same triple shape) -- `SslmDeltaFoldScaleView`/
// `SslmUFoldScaleView`'s own on-disk row layout, `model.cpp`'s
// `SslmAmplifyingFoldScaleView<Kind>::Identity/Mult/Exponent`). `present=false` is the
// identical NULL-per-projection shape `LayerAdapterProjection::a_weight==nullptr`
// already documents on the CPU path (forward_sites.h) -- the (layer, projection) this
// adapter does not cover. Defined here (not gpu_1p0.cpp, where `sslm_gpu_adapter_map`
// builds it) so `RunLayerLoopGpuSubmit`'s own dispatch-recording code
// (superslm_gpu.cpp) can read it through `GpuAdapterBridge` below without either TU
// depending on the other's opaque handle type -- the same handle-opacity convention
// `external_kv_resident`/`external_weights_resident` above already established.
struct AdapterProjSlot {
	bool present = false;
	uint64_t a_offset = 0;
	uint64_t b_offset = 0;
	uint64_t a_bytes = 0;
	uint64_t b_bytes = 0;
	uint64_t fold_offset = 0;
	uint32_t rank = 0;
	uint32_t out_channels = 0;
};

// (design Sec8): the adapter-delta dispatch bridge. `gpu_1p0.cpp`'s own
// `SslmGpuAdapterHandle` is opaque to this TU (mirroring B3/B5's `external_kv_resident`/
// `external_weights_resident` bridges above) -- this POD struct is what
// `RunLayerLoopGpuSubmit` actually reads, per (layer, projection), to record the
// GEMM-site delta-application dispatches design Sec8 specifies. `slots` is a flat
// `[layer * 7 + projection]` array (q=0, o=1, gate=2, up=3, down=4, k=5, v=6 -- the
// SAME `LayerAdapter` field-order-by-position `sslm_gpu_adapter_map` already uses),
// sized `num_hidden_layers * 7`, owned by the caller (the adapter handle's own `slots`
// vector, whose `std::vector<std::array<AdapterProjSlot, 7>>` shape is already
// contiguous in exactly this flat layout) for at least the duration of this call.
// Null (`adapter_bridge == nullptr`, the default at every pre-B6b caller) records the
// IDENTICAL base-only dispatch chain B5 always recorded -- zero adapter-delta
// dispatches issued, not a guarded no-op branch inside one (design Sec8: "a sequence
// with adapter_or_null == nullptr records the identical dispatch chain a base-only
// sequence always did").
struct GpuAdapterBridge {
	ID3D12Resource* lora_ab_resident = nullptr;
	ID3D12Resource* fold_resident = nullptr;
	uint32_t rank = 0;
	const AdapterProjSlot* slots = nullptr;  // flat [layer*7+projection], see above
	uint32_t num_hidden_layers = 0;
};

superslm::SslmForwardStatus RunLayerLoopGpuSubmit(
    superslm::SequenceLayerState& seq, const superslm::LayerWeights* layers,
    uint32_t num_hidden_layers, uint32_t layer_budget, size_t hidden_size, size_t head_dim,
    size_t num_key_value_heads, size_t intermediate_size, int64_t context_cap,
    const superslm::SslmTensorManifest& rope_tables, uint8_t* workspace, size_t workspace_size,
    ID3D12Resource* external_kv_resident, bool* io_external_kv_needs_resume_barrier,
    GpuLayerLoopInFlight** out_inflight, ID3D12Resource* external_weights_resident = nullptr,
    ID3D12Resource* external_rope_cos_resident = nullptr,
    ID3D12Resource* external_rope_sin_resident = nullptr, bool external_rope_has = false,
    uint64_t external_rope_cos_elems = 0, uint64_t external_rope_sin_elems = 0,
    const GpuAdapterBridge* adapter_bridge = nullptr,
    // T-2432 (Track A step 2/3): `q_width`, identical convention as `RunLayerLoop`'s own
    // (forward_sites.h) -- default `0` means "not supplied, derive as hidden_size," so
    // `RunLayerLoopGpu`'s own ~40 existing callers (all square fixtures) are unaffected.
    size_t q_width = 0,
    // T-2441 (Poirot 327ee29-t2438-ask5-tracka-review.md, Significant 1, D-SLM5434/D-SLM5435):
    // an optional direct readback of the GPU's own intermediate q_codes LayerScratch region --
    // the design's own first-preference remedy (§6 Track A), closing the acceptance evidence
    // gap without relying on end-to-end propagation through attention/o_proj/the MLP, which an
    // executed probe found unreliable (a wrong-but-uniform attention weighting can survive
    // int8 requantization undetected). `out_q_codes` non-null with `out_q_codes_capacity >=
    // q_width` (or `>= hidden_size` when `q_width == 0`) records the copy at Submit time;
    // `RunLayerLoopGpuFinish`'s own matching parameter performs the actual host-visible copy
    // after the fence wait. Both default to "no readback" so every existing caller (~40 sites)
    // is unaffected.
    uint8_t* out_q_codes = nullptr, size_t out_q_codes_capacity = 0,
    // (T-2577, D-SLM6278): mirrors `RunLayerLoopGpu`'s own trailing `model_generation` --
    // see that declaration's own header comment for the full contract. Defaults to `0` so every
    // existing caller (~40 sites) is unaffected.
    uint64_t model_generation = 0,
    bool model_has_qk_norm = false);

// FINISH: `block == 0` and the fence has not yet signaled: `*out_ready = 0`, returns
// `superslm::SslmForwardStatus::Ok` (design Sec4.2: "the call itself succeeded; nothing
// is ready yet"), `inflight` is left UNCHANGED and still owned by the caller (pollable
// again -- a poll never consumes the token). Otherwise (`block == 1`, or `block == 0`
// and the fence has already signaled): waits on the fence if it has not signaled yet
// and `block == 1`; then performs the identical readback + SeqState-decode +
// `DecodeStickyTag(...)` work RunLayerLoopGpu's own synchronous tail always performed,
// sets `*out_ready = 1`, deletes `inflight` (consumed, exactly once), and returns the
// DECODED per-call result -- the channel design Sec4.2 names `sslm_gpu_ready`'s own
// `out_status`.
// T-2441: `out_q_codes` mirrors `RunLayerLoopGpuSubmit`'s own new parameter -- if Submit
// recorded a q_codes readback copy (non-null `out_q_codes` there), Finish copies the actual
// bytes into THIS call's own `out_q_codes` buffer after the fence wait, up to the capacity
// Submit was given. Null (the default) skips the copy -- every existing caller unaffected.
superslm::SslmForwardStatus RunLayerLoopGpuFinish(GpuLayerLoopInFlight* inflight,
                                                    superslm::SequenceLayerState& seq,
                                                    uint8_t* workspace, int32_t block,
                                                    int32_t* out_ready,
                                                    uint8_t* out_q_codes = nullptr);

// T-2169 (Rung 2b/3/4, design Sec5/Sec5.1/Sec8, D-SLM3595/D-SLM3611/D-SLM3612/D-SLM3631/D-SLM3634/
// D-SLM3641): the chunk-submission entry point -- declared here (not exported via the public C
// ABI, gpu_1p0.h) so `gpu_1p0.cpp`'s own two G5 bridge functions (Rung 3/4) can call it after
// their own pre-scan admission decision, on the identical footing `RunLayerLoopGpuSubmit`'s own
// cross-TU declaration above already establishes. Splits `chunk_len` admitted tokens into
// TDR-safe/driver-stable sub-chunks automatically (D-SLM3641); `chunk_embedding_bytes` is the
// caller's own pre-packed, per-token `[0, SeqScaleOff(hidden_size)+16)` embedding buffer (Sec5
// 2b) -- the caller (the pre-scan) computes it via `EmbedEntry`, this function performs no
// embedding arithmetic and no admission decision of its own. Async, matching
// `RunLayerLoopGpuSubmit`'s own contract: `*out_inflight` is null on an immediate rejection
// (nothing to close), otherwise owns a token the caller drains via `RunLayerLoopGpuFinish`.
// T-2432 (Track A step 2/3): `q_width`, identical convention as `RunLayerLoopGpuSubmit`'s own
// new parameter (immediately above) -- default `0` means "not supplied, derive as
// hidden_size," so every pre-T-2432 caller of this function is unaffected.
superslm::SslmForwardStatus SubmitChunkToFullDepthForG5Bridge(
    superslm::SequenceLayerState& seq, const superslm::LayerWeights* layers,
    uint32_t num_hidden_layers, size_t hidden_size, size_t head_dim, size_t num_key_value_heads,
    size_t intermediate_size, int64_t context_cap, const superslm::SslmTensorManifest& rope_tables,
    uint8_t* workspace, size_t workspace_size, const uint8_t* chunk_embedding_bytes,
    uint32_t chunk_len, ID3D12Resource* external_kv_resident,
    bool* io_external_kv_needs_resume_barrier, ID3D12Resource* external_weights_resident,
    ID3D12Resource* external_rope_cos_resident, ID3D12Resource* external_rope_sin_resident,
    bool external_rope_has, uint64_t external_rope_cos_elems, uint64_t external_rope_sin_elems,
    const GpuAdapterBridge* adapter_bridge, GpuLayerLoopInFlight** out_inflight,
    size_t q_width = 0,
    // (T-2577 round 2, D-SLM6278): mirrors `RunLayerLoopGpuSubmit`'s own trailing
    // `model_generation` -- see that declaration's own header comment for the full contract.
    // Threaded through to `PrepareGpuLayerLoopChunkOpenState`'s own read-only residency-cache
    // predicates and the mutable K/V predicate's model-identity conjunct. Defaults to `0` so
    // every pre-existing caller is unaffected.
    uint64_t model_generation = 0,
    bool model_has_qk_norm = false,
    // Set true exactly when this call returns because a NON-FINAL sub-chunk's
    // `RunLayerLoopGpuFinish` returned a status other than `Ok`, `GpuDeviceRemoved` or
    // `GpuAllocationFailed` -- by construction a device-side guard's sticky-tag decode. False on
    // every other return, including a submission failure in any sub-chunk. The final sub-chunk's
    // finish is the caller's, so its classification is the caller's too. Defaults to null so
    // every caller that does not classify is unaffected.
    bool* out_readback_guard_rejected = nullptr);

// T-2169 (Rung 2, design Sec5, D-SLM3596/D-SLM3641): the measured, driver-stability-bounded
// maximum sub-chunk size, in tokens -- see its own definition (src/gpu/superslm_gpu.cpp) for the
// full derivation. Exposed here so a caller building an admitted chunk (Rung 3/4's own pre-scan)
// can size its own embedding-byte buffer without needing to know the bound is enforced
// internally -- the buffer must still cover the FULL `admit_count`, since
// `SubmitChunkToFullDepthForG5Bridge` does its own internal splitting.
extern const uint32_t kT2169TdrSafeMaxChunkTokens;

// T-2169 (Rung 2, D-SLM3595): the SeqState embedding-byte block size for one token, `[0,
// SeqScaleOff(hidden_size)+16)` (Sec5 2b) -- exposed so a caller can size and index its own
// per-chunk embedding buffer without re-deriving `SeqScaleOff`'s own alignment arithmetic.
uint32_t T2169SeqEmbeddingBlockBytes(uint32_t hidden_size);

// (M1's
// own remedy): the structural closure for `RunLayerLoopGpu`'s own host-side
// guard ladder -- generated from `gpu_layer_loop_guards.def`, the single
// source of truth for how many guards, in what order, to what CPU-matching
// status the ladder implements. `GpuLayerLoopGuard::kCount` is a compile-time
// constant a test can assert against directly (`static_cast<int>(
// GpuLayerLoopGuard::kCount) == 9`); the pin round's own table-walk cell
// (Curie's work, not built here -- Brunel does not author tests) constructs
// one malformed-input fixture per named guard and asserts CPU's
// `RunLayerLoop` and GPU's `RunLayerLoopGpu` agree, status for status, so a
// guard added to CPU without a matching `.def` row -- and a matching `return`
// in `RunLayerLoopGpu`'s own ladder -- fails that walk loudly rather than
// waiting for a fourth hand-count to miss it.
//
// CORRECTED 2026-08-14 (superseding this comment's own prior claim):
// this comment's own last sentence overclaims. `kCount` and both
// `static_assert`s below compare a literal to `gpu_layer_loop_guards.def`'s
// own row count -- none of it, and no pin cell matched to that file's row
// order by eye, ever reads `forward_sites.cpp`, so a guard added to BOTH
// ladders with no `.def` row (proven by execution) leaves every one of these
// green. The class-level guarantee this paragraph describes is
// `tests/ci/check_gpu_guard_status_parity.py`'s job, not this
// enum's or that cell's -- it derives CPU's own guard-status set from
// `forward_sites.cpp` directly and compares it against this file's
// generated set. `GpuLayerLoopGuard`/`kCount` remain exactly what they were:
// the compile-time row count this header's `.def` include generates, useful
// for the table-walk cell's own loop bound, not a guarantee that the count
// is CPU-correct on their own.
enum class GpuLayerLoopGuard : int {
#define SSLM_GPU_LAYER_LOOP_GUARD(enum_name, status_name, cpp_citation) enum_name,
#include "superslm/gpu_layer_loop_guards.def"
	kCount
};

// True iff the most recent RunLayerLoopGpu call skipped a weight upload because its packed row
// already matched the resident cache (a cache hit); false on a cache miss. The
// device-observable residency cache exists because content correctness alone (a byte-for-byte
// match against the cached row) cannot distinguish "read from the resident DEFAULT-heap buffer"
// from "read from a plain upload-heap buffer holding the identical bytes" -- both would pass a
// value-comparison pin identically. Set internally by RunLayerLoopGpu's own weight-residency
// decision; read back by the caller after RunLayerLoopGpu returns.
//
// Every return path that resolves the call BEFORE that decision runs reads false, by
// construction (the function-entry reset, never overwritten on that path): the nine-guard
// ladder, the two device-capability rejections, and every return inside the recording window's
// own catch, twenty-nine paths in all -- T-2568 added one new catch clause to each of
// RunLayerLoopGpuSubmit and SubmitOneSubChunkToFullDepthForG5Bridge
// (GpuLayerWeightsContractError's own, PackLayerWeightsBytes' required-pointer refusal), two more
// than the twenty-five this paragraph named before; T-2577 (D-SLM6279, GpuShaderBinaryStaleError's
// own, ShaderPath's stale-binary refusal) added a second new catch clause to each of the same two
// functions, two more again. Every path that resolves the call AFTER the
// decision reads exactly what the decision decided (true on a cache hit, false on a miss),
// whether the call's own final status is Ok or one of DecodeStickyTag's fifteen rejecting
// statuses, including ResidualReconciliationScaleOutOfDomain -- the twenty-nine before them,
// alike, thirty-five paths' own destination in total.
//
// Both counts are derived structurally from source, not restated by hand, by
// tests/ci/check_gpu_guard_status_parity.py (derive_lwuws_before_decision_count/
// derive_lwuws_after_decision_count) -- that check is what a future reader trusts if this
// paragraph and the real code ever disagree, and it is what must be re-run, not this comment
// hand-edited, whenever RunLayerLoopGpuSubmit/RunLayerLoopGpuFinish/
// SubmitOneSubChunkToFullDepthForG5Bridge's own return-path shape changes.
bool LastWeightUploadWasSkipped();

// (T-2577, D-SLM6278): the RoPE-table residency cache's own observable, mirroring
// `LastWeightUploadWasSkipped`'s contract exactly -- true iff the most recent call served
// `g_resident_rope`'s cached cos/sin tables rather than repacking and re-uploading them. Set
// internally by the shared residency preparation reached from `RunLayerLoopGpu`,
// `RunLayerLoopGpuSubmit`, and `SubmitChunkToFullDepthForG5Bridge`; read back by the caller after
// that call (or its finish half) returns. This is the instrument the S1 must-accept cell reads: a second
// fresh sequence of the SAME model (the caller's own `model_generation` unchanged) is required
// to report `true` here -- a pack/upload COUNT of zero, not a timing.
bool LastRopeUploadWasSkipped();

// (T-2578 confirmation remedy): true iff the most recent pre-1.0 call reused
// `g_resident_kv` instead of uploading the caller's workspace. A fresh sequence always reports
// false, even when both workspace address and model_generation match the preceding sequence.
// This is mutable sequence history; unlike weights/RoPE, model identity cannot make it reusable.
bool LastKvUploadWasSkipped();

// A per-call timing
// breakdown for the most recent `RunLayerLoopGpu` call that reached command-list recording (a call
// rejected by the guard ladder before that point leaves every field at 0.0 -- there is nothing to
// time). Four numbers, each in milliseconds, each covering a DISJOINT phase of the call so they sum
// to (approximately) the call's own wall-clock cost:
//
// - `record_ms`      -- CPU time building the command list (weight/K-V pack-or-skip decision,
//                        every Upload()/MakeBuffer() call, every bind_and_dispatch()'s own
//                        SetComputeRoot*/Dispatch/ResourceBarrier recording): from the command
//                        list's own Reset() to its Close().
// - `submit_wait_ms` -- CPU time from ExecuteCommandLists to the fence signaling complete. This is
//                        submission overhead PLUS actual GPU execution PLUS driver/OS scheduling --
//                        it is not a GPU-only number; `gpu_busy_ms` below is.
// - `gpu_busy_ms`    -- GPU-measured time between a timestamp query placed immediately before the
//                        call's own first dispatch and one placed immediately after its last,
//                        resolved via the command queue's own timestamp frequency. This is the
//                        number the roofline comparison is judged against -- it excludes recording,
//                        submission, and readback entirely, on the GPU's own clock, not a CPU
//                        estimate of it.
// - `readback_ms`    -- CPU time mapping the small readback buffers and copying their contents into
//                        the caller's own `seq`/`workspace` arguments.
struct GpuCallTiming {
	double record_ms = 0.0;
	double submit_wait_ms = 0.0;
	double gpu_busy_ms = 0.0;
	double readback_ms = 0.0;
};
GpuCallTiming LastCallTiming();

// The ceiling-division formula
// `ComputeGpuGemmSiteGroupPlan` (below) uses to turn an output width into a group count --
// `ceil(out_channels / threads_per_group)`, i.e. `(out_channels + threads_per_group - 1) /
// threads_per_group`. The property this formula must hold, for any caller: the returned group
// count `g` satisfies `g * threads_per_group >= out_channels` (every channel has a thread) and,
// when `g > 0`, `(g - 1) * threads_per_group < out_channels` (no wasted group).
uint32_t ComputeGpuGemmGroupCount(uint32_t out_channels, uint32_t threads_per_group);

// The sites whose GEMM step runs as its own multi-group dispatch
// (`down_proj_gemm_site.hlsl` and the siblings named there).
// (design Sec3/Sec6.1): `KvProj` added -- kv_proj's own GEMM step is now
// its own multi-group dispatch (`kv_proj_gemm_site.hlsl`), as NEW production work (this
// enumerator never existed in the pre-1.0 substrate; `kv_proj_site.hlsl`'s own header comment
// named the second-dispatch remedy this enumerator realizes).
enum class GpuGemmSplitSite { QProj, OProj, GateProj, UpProj, DownProj, KvProj };

struct GpuGemmSiteGroupPlan {
	uint32_t out_channels = 0;
	uint32_t threads_per_group = 0;
	uint32_t groups = 0;
	// How the 256 threads of a group are split -- `lanes` threads cooperate on
	// one output channel via the transposed GEMM partition (`GemmCoalescedGpu`,
	// site_common.hlsli), so a group covers `256 / lanes` channels. lanes == 1 degenerates to
	// the legacy one-thread-per-channel partition with no reduction. The SAME value is passed
	// to the shader as the 11th root constant, so the group count and the shader's own
	// channel indexing cannot drift apart.
	uint32_t lanes = 1;
	uint32_t channels_per_group = 256;
};

// The ONE source `RunLayerLoopGpu`'s own dispatch call for `site` reads its
// `Dispatch(groups, 1, 1)` grid size from -- no local variable sits between this function's own
// return value and the `bind_and_dispatch` call, so there is nothing at the call site left to
// hand-edit independently of this function. `tests/test_main.cpp`'s own
// `TestT2101_ComputeGpuGemmSiteGroupPlan_RealDimensions` calls this SAME function directly, no
// device required, at the real model's own dimensions (hidden_size=1536, intermediate_size=8960)
// -- a mutation of this function's own body (the confirmation pass's own specified re-falsification
// method) is therefore observable by the pinned suite, not only by a manual C5/throughput run
// against real hardware.
//
// `kv_out_channels` added -- `KvProj`'s own dispatch covers BOTH K's and V's
// channels packed into one grid (`kv_proj_gemm_site.hlsl`'s own header comment), so its own
// out_channels is 2*kv_hidden_size, passed in already doubled by the caller (the ONLY site
// that reads this parameter; every other site ignores it).
// T-2432 (Track A step 9, design §2.5 GS-14/§6 Track A step 9, D-SLM5248): `q_width`, read
// ONLY by the `QProj` case (`plan.out_channels`) -- every other site ignores it, on the
// identical "the only site that reads this parameter" footing `kv_out_channels` already
// documents above. Default `UINT32_MAX` means "not supplied, derive as hidden_size" (the
// pre-widening identity, matching every other new q_width-shaped parameter this ask adds).
GpuGemmSiteGroupPlan ComputeGpuGemmSiteGroupPlan(GpuGemmSplitSite site, uint32_t hidden_size,
                                                  uint32_t kv_out_channels,
                                                  uint32_t intermediate_size,
                                                  uint32_t q_width = UINT32_MAX);

// One GPU-measured millisecond figure per dispatch the most recent
// `RunLayerLoopGpu` call issued, in dispatch order (empty if the call was rejected before
// recording, or timed by fewer than one dispatch). Dispatch `d` within this call corresponds to
// layer `d / 24` and that layer's own `d % 24`-th call in the fixed, unconditional per-layer order
// `RunLayerLoopGpu` issues them: attn_norm, q_proj_gemm, q_proj, kv_proj_gemm, kv_proj,
// rope_guard, rope_commit, attention_score, softmax, context_accumulate, ctx_fold, o_proj_gemm,
// o_proj, attn_residual, mlp_norm, gate_proj_gemm, gate_proj, up_proj_gemm, up_proj, mlp_act,
// down_proj_gemm, down_proj, mlp_residual, commit (24 sites/layer -- re-derived (design
// Sec3/Sec6.1) from the 22-site count previously shipped: `kv_proj_gemm` is a NEW
// dispatch (kv_proj no longer fused-and-single-dispatch, `kv_proj_gemm_site.hlsl`'s own header
// comment states why the split is now sound) and RoPE's own commit phase is a second, separate
// dispatch (`rope_commit_site.hlsl`) rather than a group-barrier-separated phase inside one).
// Summing every `d` with the same `d % 24` across however many layers a call processed gives
// that SITE's own total GPU time for the call -- `tools/t2100_gpu_throughput.cpp`'s own
// per-site table is exactly this sum.
std::vector<double> LastCallPerDispatchTimingsMs();

// An earlier always-declared LINK-RED form of this instrument (`ArmO11AllocationFailure
// Injection`/`ClearO11AllocationInjection`, undefined) was UNGATED, so merging it broke the
// arc's own executing acceptance gate -- `tests/test_main.cpp` is not a pre-build red suite the
// way an earlier LINK-RED precedent was (this file's own :9-18 banner describes the ORIGINAL,
// whole-suite situation, not this one): most of this suite is already built and green, and a
// SINGLE undefined symbol anywhere in the single translation unit `tests/test_main.cpp` compiles
// into fails the WHOLE binary's link, taking the 33879-check baseline down with it. Corrected:
// O11's own instrument is held behind a compile-time gate, off by default, mirroring this
// project's own established precedent for exactly this shape (`src/bad_alloc_wrap.h`'s
// `SUPERSLM_ENABLE_BAD_ALLOC_INJECTION`, defined only for the test-injection build target). Not
// defined by `build.bat` -- the default build never sees these two declarations at all. The
// build seat's own trigger to arm this pin: land the real bodies AND define
// `SUPERSLM_O11_ALLOC_INJECTION` (e.g. via `build.bat`'s own `/D` list, beside
// `SUPERSLM_ENABLE_BAD_ALLOC_INJECTION`) in the SAME round -- defining the macro without the
// real bodies reproduces this ticket's own LINK-RED proof, on purpose, as the gate's own
// self-check.
//
// CORRECTED 2026-08-14: "Not defined by `build.bat`" and "the build seat's own trigger to arm this
// pin" (naming a future condition) were true for a while and false since -- `build.bat`'s own
// test-binary compile line
// has defined `SUPERSLM_O11_ALLOC_INJECTION` alongside `SUPERSLM_ENABLE_BAD_ALLOC_INJECTION` on
// the default test-binary compile line since the real bodies landed, five rounds ago. The
// trigger this paragraph describes as pending fired several commits ago; nothing in this file was
// edited to say so, because an intervening rename touched the `SUPERSLM_O11_
// ALLOC_INJECTION` mention one paragraph below
// this sentence without re-reading the paragraph it sits in. Believing
// this sentence and removing the flag drops the O11 gate silently: gate-on and gate-off both then
// report the SAME 3-failure count (16 checks and both discrimination cells vanish with no test
// failure of any kind, an executed measurement) -- the real, structural fix is the
// pinned assertion in `tests/ci/check_gpu_guard_status_parity.py`
// (`check_build_bat_defines_o11_gate`), not this sentence alone.
//
// CORRECTED 2026-08-14: the block above's own header attributed a rename it
// describes to "this same round" -- the rename was a PREVIOUS round's; "this
// same round" made the two the same round, which they were not. Corrected to the accurate
// wording ("the rename this same arc made in an earlier round").
//
// Mirrors B12's own arm/inject naming (ArmAllocationFailureInjection/ClearAllocationInjection):
// arms injection so the NEXT allocation call AT THE NAMED SITE in RunLayerLoopGpu fails with a
// synthetic allocation error; the call after clears it.
//
// CORRECTED 2026-08-14 (superseding the note below -- left standing, not rewritten, per this
// tree's own append-only discipline): a prior fix MOVED the arm site from the weight DEFAULT-heap
// allocation to `work_scratch_uav`, which made the M-b half of `TestT2063_S1Mb_...` live for the
// first time -- and, measured by this review, made an earlier remedy (the weight allocation's
// throw reaching the shared outer catch rather than a site-local one) permanently unpinnable: that
// allocation is gated behind `!weights_resident` and a hit never reaches it, so with a single arm
// point the two remedies could never both be live at once. Fixed by taking the index-parameterized
// shape this instrument declined when it was single-site (B12's own `ArmAllocationFailureInjection
// (uint32_t)` convention, matched here): `ArmO11AllocationFailureInjection` now takes a `site`
// selector, one of the two named constants below, and the injected throw fires only when the ARMED
// site matches the call site currently executing -- both the weight DEFAULT-heap allocation and
// `work_scratch_uav`'s own allocation carry the check now, so either remedy can be pinned,
// independently, by arming the site it lives at.
constexpr uint32_t kO11AllocInjectionSiteWeightDefaultHeap = 0;
constexpr uint32_t kO11AllocInjectionSiteWorkScratchUav = 1;
// A third site, inside
// `RestoreGpuSequenceState`'s own device round-trip (superslm_gpu.cpp) -- pins that
// `sslm_gpu_seq_restore`, defined in gpu_1p0.cpp, now catches an exception thrown from that
// call and returns SSLM_DEVICE_LOST instead of letting it escape the status-returning API
// boundary (the same boundary B5 already closed for `sslm_gpu_ready`/`RunLayerLoopGpuFinish`).
constexpr uint32_t kO11AllocInjectionSiteSeqRestore = 2;
//
// Note: the definitions built
// targeted `work_scratch_uav`'s own allocation, not the weight DEFAULT-heap buffer this
// comment originally specified -- a later fix moved the arm site there so an armed call
// reaches the throw on a cache HIT too, which the weight buffer's own allocation (gated behind
// `!weights_resident`) cannot. The function names below are unchanged; only the site they arm
// moved. `TestT2063_S1Mb_WorkScratchUavAllocationThrow_ReturnsGpuAllocationFailed_SkippedFalse`
// (tests/test_main.cpp) is gated identically -- it does not compile into the default build at all,
// so it cannot be the thing that fails the default build's own link.
//
// CLARIFIED 2026-08-14: "the default build" above means the CMake target (`superslm_tests`,
// `CMakeLists.txt`) -- the ONLY build configuration where `SUPERSLM_O11_ALLOC_INJECTION` is ever
// undefined, since `build.bat`'s own default test-binary line has defined it since the real bodies
// landed (this same file's own dated correction below). Not a correction to the CLAIM (both builds
// genuinely never compile this cell without the macro) -- a disambiguation of "default," the exact
// word S1 found already misleading readers about a DIFFERENT sentence in this same paragraph
// group. See `derive_execution_scope`/`check_execution_scope_waivers` (`check_gpu_guard_status_parity.py`)
// for the machine-derived, standing-waived statement of which build compiles what.
//
// CORRECTED 2026-08-14: the CLARIFIED block above is itself false -- `build.bat`'s own C5-harness `cl`
// invocation (the second of its own two, `t2039_c5_harness.cpp`'s own build line) ALSO compiles
// `superslm_gpu.cpp` without ever defining `SUPERSLM_O11_ALLOC_INJECTION`; it does not compile
// `test_main.cpp`, so it is not "the default build" in the sense this paragraph means, but "the
// ONLY build configuration where the macro is ever undefined" was false the day it was written --
// `derive_execution_scope` (`check_gpu_guard_status_parity.py`) has always reported two such
// configurations, not one. Corrected to the property the sentence actually needs: the CMake target
// is the only configuration compiling `test_main.cpp` without the macro.
#ifdef SUPERSLM_O11_ALLOC_INJECTION
void ArmO11AllocationFailureInjection(uint32_t site);
void ClearO11AllocationInjection();
#endif  // SUPERSLM_O11_ALLOC_INJECTION

// T-2180/T-2183 (D-SLM3655/D-SLM3660): the tenth-failure-origin fault-injection seam the design's
// own Sec9 Coverage Model names (`Claude/Vitruvius/t2169-gpu-batched-prefill-design-2026-08-18.md`
// Sec5.1/Sec9, D-SLM3634) and the test author's own casebook (T-2183) specified as owed to the
// builder, not authorable against the pre-existing O11 seam (that seam's only two call sites fire
// at chunk-OPEN, strictly before this window). Forces `GpuGemmGroupArithmeticError` from INSIDE
// `SubmitOneSubChunkToFullDepthForG5Bridge`'s own per-token loop (`superslm_gpu.cpp`), immediately
// after `RecordOneTokenFullDepthDispatchBody` returns for admitted token index `t` and before the
// loop advances to `t+1` -- letting a cell arm "throw after recording admitted token index N" and
// observe the design's own ruled divergence (whole-(sub-)chunk discard, not token-scoped) through
// the real code path rather than reasoning about it.
//
// Mirrors `SUPERSLM_O11_ALLOC_INJECTION`'s Arm/fire idiom exactly (a single-shot armed flag plus an
// index match, cleared on fire so a re-armed-forever flag never re-fires on a LATER, unrelated
// call in the same process) but is its OWN macro, not a third named site under O11's: O11 arms one
// of a small, fixed enumeration of allocation call sites (`kO11AllocInjectionSite*`); this seam
// arms an arbitrary 0-based TOKEN INDEX within whatever chunk the next call submits -- a different
// parameter shape (an index into a per-call sequence, not a selector over a fixed site set) that
// does not fit the existing constants' own convention. `after_token_index` is 0-based, within the
// current (sub-)chunk: arming 1 against a chunk with >= 2 admitted tokens fires immediately after
// token index 1's own dispatches are recorded (both tokens 0 and 1 are in the never-submitted
// command list at that point), reproducing D-SLM3634's own worked example (an admitted-token index
// `i > 0`).
#ifdef SUPERSLM_T2169_CHUNK_RECORDING_FAULT_INJECTION
void ArmT2169ChunkRecordingFaultInjection(uint32_t after_token_index);
void ClearT2169ChunkRecordingFaultInjection();

// T-2186 remedy P1's own pin (Brunel fix round 3, D-SLM3682, confirmation
// `Claude/Poirot/8642652-t2186-t2169-fix2-confirmation.md`): the seam above only reaches the
// try-covered per-token recording loop, whose throw is caught by
// `SubmitOneSubChunkToFullDepthForG5Bridge`'s own two catch clauses. Pinning P1 (the widened
// `Submitted` window is not exception-safe against a throw from that function's own UNCOVERED
// TAIL -- `dev.list->Close()`, `dev.queue->Signal()`, `new GpuLayerLoopInFlight()`, all outside
// either catch) requires a throw from exactly that region. Single-shot, no token index -- the
// tail runs once per (sub-)chunk submission, after every admitted token's dispatches are already
// recorded. Arm it, drive one chunk through the public bridge, and the call throws
// `std::runtime_error` uncaught -- exactly the class `SSLM_GPU_HR` throws on a real `Close()`/
// `Signal()` failure -- letting a cell catch it and assert `SubmitAdmittedChunkForG5Bridge`'s own
// RAII window guard (`gpu_1p0.cpp`) left `model->submitted_sequences` and `seq->state` clean
// rather than wedged.
//
// T-2195 pertoken pin (Curie, D-SLM3702 arc round 5 S1/S2 fold): the identical pre-Close fire
// point ALSO exists in `RunLayerLoopGpuSubmit`'s own tail (superslm_gpu.cpp) since that
// function's containment landed -- the primitive the PER-TOKEN drive (`sslm_decode_step_gpu` via
// `SubmitOneSequenceDecode`, reached from the public surface through
// `SslmGpuSeqDecodeStepForG5Bridge`'s embed-then-drive branch) calls directly, never through
// `SubmitOneSubChunkToFullDepthForG5Bridge`. This single Arm/Clear pair now fires at BOTH sites;
// which one a given call reaches depends only on which public entry point drove it.
void ArmT2169ChunkRecordingTailFaultInjection();
void ClearT2169ChunkRecordingTailFaultInjection();

// T-2192 (D-SLM3695's own confirmation round, finding M3): the pin above (and its cell,
// `TestGuard_ContextReusableAfterCaughtTailFault`) only exercises the pre-`Close()` failure point
// -- the ONE of the tail's three failure points the single existing seam fires at. The other two
// -- a `Signal()` failure after `ExecuteCommandLists` has already queued the work, and a
// `std::bad_alloc` from the `new GpuLayerLoopInFlight()` allocation, both handled by their own
// dedicated catch clauses in `SubmitOneSubChunkToFullDepthForG5Bridge` (`superslm_gpu.cpp`) --
// had no seam that could reach them. These two fire at those exact points, mirroring the
// pre-`Close()` seam's own single-shot Arm/Clear idiom.
void ArmT2169ChunkRecordingTailSignalFaultInjection();
void ClearT2169ChunkRecordingTailSignalFaultInjection();
void ArmT2169ChunkRecordingTailBadAllocFaultInjection();
void ClearT2169ChunkRecordingTailBadAllocFaultInjection();

// The prompt twin (`SslmGpuSeqPrefillPromptForG5Bridge`, gpu_1p0.cpp) reports a device-side guard
// refusal as `SSLM_SEQUENCE_REJECTED` only when `GetDeviceRemovedReason()` on the device the chunk
// ran on returns `S_OK` at classification time. A real removal between the readback and that query
// cannot be constructed deterministically, so this single-shot seam makes the NEXT such query
// report a removed device, and is consumed by it. Defined in gpu_1p0.cpp.
void ArmPrefillGuardDeviceRemovedQueryInjection();
#endif  // SUPERSLM_T2169_CHUNK_RECORDING_FAULT_INJECTION

// T-2866's own `ArmGpuFinishDegenerateLogitRowInjection` (gpu_1p0.cpp, plan Sec3.10.1/Sec3.10.2)
// is declared `extern "C"` at global scope, not here: it matches the test author's own
// reservation (`tests/t2899-schema-deadend-red-suite/cell_gpu_cell2_degenerate.cpp`) and the CPU
// twin's own identical placement (`ArmCpuFinishDegenerateLogitRowInjection`, src/sslm_abi.cpp) --
// the same header-free, locally-`extern`-declared pattern `SslmSeqLiveStateForTest` already uses,
// deliberately not folded into this namespace.

// Read back the device-resident K/V cache in the SAME layout and argument order
// superslm::KeyRow/ValueRow already define (forward_sites.h) -- the GPU port's
// `workspace` is the device-resident twin of the identical buffer RunLayerLoop uses.
const int8_t* KeyRowGpu(const uint8_t* workspace, uint32_t layer, int64_t context_cap,
                         size_t num_kv_heads, size_t head_dim, size_t kv_head,
                         int64_t position);
const int8_t* ValueRowGpu(const uint8_t* workspace, uint32_t layer, int64_t context_cap,
                           size_t num_kv_heads, size_t head_dim, size_t kv_head,
                           int64_t position);

// --- B7 (Sec5.8, Sec11 B7): the dispatch_budget contract, whole-
// layer quanta. This is the ARITHMETIC/policy half of `sslm_decode_step_gpu`'s own
// contract -- how many layers a call at a given budget will record, and what status
// it returns -- and is testable without a device. Legacy models retain 24 dispatches
// per layer. A model carrying QK norm adds qk_norm_site.hlsl between kv_proj_site and
// rope_guard_site and therefore uses 25. The caller passes the model-derived divisor;
// floor division means a budget one below that divisor makes no progress, and a
// partial layer is never recorded. ---
enum class SslmGpuStatus { Ok, DispatchBudgetTooSmall, Busy };

// The one source for both model shapes' real per-layer dispatch counts. Model mapping
// records the appropriate value and both planning and batch budget spending consume it.
constexpr uint32_t kLegacyDispatchesPerLayer = 24;
constexpr uint32_t kQkNormDispatchesPerLayer = 25;
// Compatibility name for callers whose model is already known to carry QK norm.
constexpr uint32_t kDispatchesPerLayer = kQkNormDispatchesPerLayer;
constexpr uint32_t DispatchesPerLayer(bool has_qk_norm) {
	return has_qk_norm ? kQkNormDispatchesPerLayer : kLegacyDispatchesPerLayer;
}

// T-2240/O3 (SuperSLM 1.2.1, plan Sec10 Phase 2 O3): the ADAPTER_U region's own byte size,
// extracted from the work_total site (superslm_gpu.cpp, `work_adapter_u_off + ...`) into
// this namespace-scope pure helper -- the same one-source discipline
// kDispatchesPerLayer above establishes, so the sizing expression and its test read one
// definition. The region keeps its 4-byte floor (a D3D12 buffer needs to be a legal
// resource even at rank 0 / no bound adapter) and is ROUNDED UP TO A MULTIPLE OF 4 for
// every rank: a raw non-multiple-of-4 size (rank 5, 6, 7, ...) made the total an exact
// multiple of 4 only by luck of the offsets before it; the rounding closes the theoretical
// over-read at the CAS's tail (the CAS was already well-formed in practice -- this is a
// structural guarantee, not a runtime behavior change).
constexpr uint64_t AdapterURegionBytes(uint64_t adapter_rank) {
	const uint64_t floored = adapter_rank < uint64_t{4} ? uint64_t{4} : adapter_rank;
	return (floored + uint64_t{3}) & ~uint64_t{3};
}

SslmGpuStatus PlanDispatchBudgetGpu(uint32_t dispatch_budget, uint32_t num_hidden_layers,
                                     uint32_t current_layer_position,
                                     uint32_t* out_layers_to_issue,
                                     uint32_t dispatches_per_layer = kQkNormDispatchesPerLayer);

// (design Sec10 B2):
// GpuLayerLayout/ComputeLayerLayout/PackLayerWeightsBytes, promoted from
// src/gpu/superslm_gpu.cpp's own internal implementation (unchanged bodies)
// to this shared header so BOTH the pre-1.0 substrate's own RunLayerLoopGpu (superslm_gpu.cpp,
// call site unchanged by this move) and the 1.0 API's own model-handle upload path
// (src/gpu/gpu_1p0.cpp, sslm_gpu_model_map) compute weight-packing bytes from exactly ONE
// implementation -- never two copies that could silently drift, the same hazard
// ResidentWeights' own header comment (superslm_gpu.cpp) already names for a pointer-only
// cache key. Only the DECLARATION moved; every function body is byte-for-byte the loop it
// replaces at its old call site. `off[56]`/`stride` match every `*_site.hlsl`'s own
// `Layout.Load<uint>(N*4)` index
// order exactly (superslm_gpu.cpp's own original header comment on this struct, carried here
// unchanged).
// (design §4/§6 Track B step 1/3, T-2551): `off[56..61]` are six NEW per-layer field offsets --
// q_norm_present, q_norm_gain, q_norm_site_constant, k_norm_present, k_norm_gain,
// k_norm_site_constant -- appended AFTER every pre-existing field (`ComputeLayerLayout`'s own
// `cur` accumulation places them last, so `off[0..55]` keep byte-identical values to before this
// ask). Serialized into the `Layout` GPU buffer at buffer POSITIONS 57-62, one past the stride
// slot at position 56 -- never at off[]'s own array indices 56-61, which would collide with
// stride's existing serialization position. Every pre-existing shader reads only positions 0-56
// and is therefore unaffected; only `qk_norm_site.hlsl` reads 57-62 (superslm_gpu.cpp's own
// `layout_bytes` construction, PackLayerWeightsBytes below).
// `off[62]` carries K's wide source scale and `off[64]` the dense QKC1
// (r_t, e_t, ratio) channel image. `off[63]` remains reserved.
struct GpuLayerLayout {
	uint32_t off[65]{};
	uint32_t stride = 0;
};

// T-2432 (Track A step 5/7, design §2.1 items 4/5, §6 Track A steps 5/7, GS-07/GS-10/GS-11):
// `q_width` (num_attention_heads * head_dim) is Q's own real output width / O's own real
// input width, independent of `hidden_size` once R1 no longer holds -- appended LAST, with a
// default of `UINT32_MAX` meaning "not supplied, derive as `hidden_size`" (the pre-widening
// identity), so this function's two pre-T-2432 callers (both updated in the same change to
// pass the real value) and any future caller that genuinely wants the square-geometry
// behavior are both served by the identical convention `RunLayerLoop`'s own `q_width`
// parameter uses (forward_sites.h).
GpuLayerLayout ComputeLayerLayout(uint32_t hidden_size, uint32_t kv_hidden_size,
                                   uint32_t num_kv_heads, uint32_t num_attention_heads,
                                   uint32_t intermediate_size, uint32_t q_width = UINT32_MAX);

// Packs `N` LayerWeights entries into one contiguous byte buffer at `layout`'s own
// stride/offsets -- the exact byte-for-byte transformation RunLayerLoopGpu's own
// weight-residency miss path has always performed, extracted verbatim (B2) so
// it has exactly one implementation. `H`=hidden_size, `KV`=num_kv_heads*head_dim,
// `NH`=num_kv_heads, `NQH`=num_attention_heads, `I`=intermediate_size -- the same five
// dimension values `ComputeLayerLayout` above must be called with to produce `layout`.
// T-2432 (Track A step 5): `q_width`, identical convention as `ComputeLayerLayout`'s own new
// parameter above -- `q_weight`'s and `o_weight`'s real byte extents (GS-10/GS-11) need it
// independently of `layout` (which already encodes it into `layout.off[]`), because this
// function's own copy loops read `q_width` directly to bound their `for` loops.
std::vector<uint8_t> PackLayerWeightsBytes(const superslm::LayerWeights* layers, uint32_t N,
                                            const GpuLayerLayout& layout, uint32_t H, uint32_t KV,
                                            uint32_t NH, uint32_t NQH, uint32_t I,
                                            uint32_t q_width = UINT32_MAX);

// --- Sec5.9: the asynchronous sequence lifecycle, Idle ->
// Submitted -> Completed -> Idle. Each `CallProceedsOrBusy_*` function is the POLICY
// half of its named ABI call -- given whether the sequence (or, for unmap, the model)
// currently has Submitted work outstanding, does the call proceed or return
// SSLM_BUSY synchronously -- testable without a device for the identical reason
// PlanDispatchBudgetGpu is: no actual submission is needed to answer a policy
// question, only the state predicate the call is deciding on. ---
enum class SslmSequenceGpuState { Idle, Submitted, Completed };

// The five calls Sec5.9 names as returning SSLM_BUSY against a Submitted sequence.
bool CallProceedsOrBusy_DecodeStepGpu(bool sequence_is_submitted);
bool CallProceedsOrBusy_SeqSave(bool sequence_is_submitted);
bool CallProceedsOrBusy_SeqReset(bool sequence_is_submitted);
bool CallProceedsOrBusy_SeqRelease(bool sequence_is_submitted);
// Model-wide: true iff ANY sequence created against the model handle is
// Submitted, not only a single targeted one.
bool CallProceedsOrBusy_ModelUnmap(bool any_sequence_submitted);

// `sslm_gpu_ready`'s dual role: exempt from SSLM_BUSY by construction (the
// one call legal against a Submitted sequence); collapses Submitted -> Completed ->
// Idle in one call once the device fence signals. `fence_signaled` is the test's own
// simulated fence state; `*out_ready` mirrors the real ABI's own out-parameter
// (0 = still Submitted, ordinary polling; 1 = collapsed to Idle, aggregated status in
// `*out_status`).
bool GpuReadySignalsCompletion(bool fence_signaled, int32_t* out_ready,
                                superslm::SslmForwardStatus* out_status);

// --- B8 (Sec11 B8): device-to-host / host-to-device round-trip of the
// residual-stream/KV/kv_saturation_count/context_length state (Sec10 dim 9),
// mechanism-level. SuperSLM_Plan.md Sec12's `sslm_seq_save`/`sslm_seq_restore` CPU-side
// C-ABI wrapper is now built (src/sslm_abi.cpp) -- this pair below is the GPU-resident
// mechanism that wrapper's own restore path calls into for a sequence with device-resident
// state, gated on the same device-resident round-trip mechanism the design commits to,
// not a specific ABI symbol name. **Restore-time
// device allocation and the host-to-device upload happen inside
// `RestoreGpuSequenceState` itself (Sec5.9) -- never inside a model-map-
// level call** -- `out_seq`/`out_workspace` are caller-owned, sized identically to
// the save call's own `seq`/`workspace`, and this function's own definition is where
// the build seat's restored-sequence device allocation is sited. ---
// `hidden_codes_size`
// gates whether the residual stream round-trips through the blob, exactly like
// `workspace_size` already gates the K/V bytes -- 0 means the caller does not want
// hidden_codes carried (the pre-1.0 test_main.cpp cells that predate the 1.0 handle owning
// hidden_codes pass 0 and keep their own established contract); the 1.0
// `sslm_gpu_seq_restore` path passes the model's real hidden_size and gets the full
// round-trip. Save reads from `seq.hidden_codes`; Restore writes into `out_seq->hidden_codes`
// (both already-set pointers, never allocated here).
//
// (design Sec4.2/Sec22): `model_content_hash` is the SAVING model's own 32-byte
// `SslmModelView::RawIntegrityHash()` (design Sec5.1) -- written into the v4 blob's v3-prefix
// field so a later restore against a DIFFERENT model can detect the mismatch the N1
// size-admissibility widening left open (a foreign blob whose derived context_cap happens to be
// admissible against a same-shape-but-different model used to restore silently). The v4
// append-only tail carries all four per-site K/V saturation counters, preserving their provenance
// alongside `kv_saturation_count` across save/restore. Restore's own
// identity check is NOT threaded through this function -- it runs in the caller
// (`sslm_gpu_seq_restore`, gpu_1p0.cpp) via `PeekGpuSeqBlobModelHash` below, before this function (or
// the fresh handle it restores into) is ever reached, matching the peek-then-validate-then-allocate
// ordering the size-derivation ladder already established.
bool SaveGpuSequenceState(const superslm::SequenceLayerState& seq, size_t hidden_codes_size,
                           const uint8_t* workspace, size_t workspace_size,
                           const std::array<uint8_t, superslm::kIntegrityHashBytes>& model_content_hash,
                           int32_t bound_schema_index, uint32_t dfa_walk_state, bool ready_for_logits,
                           void* out_blob, size_t* out_blob_size);
bool RestoreGpuSequenceState(const void* blob, size_t blob_size, superslm::SequenceLayerState* out_seq,
                              size_t hidden_codes_size, uint8_t* out_workspace, size_t workspace_size,
                              int32_t* out_bound_schema_index, uint32_t* out_dfa_walk_state,
                              bool* out_ready_for_logits);

// (design Sec4.2/Sec21): reads only the blob's header (magic + `workspace_size`) -- never the body, never a device
// call -- so `sslm_gpu_seq_restore` can derive the blob's own `context_cap` (Sec5.1's K/V sizing
// formula) BEFORE it allocates the fresh handle, instead of always sizing that handle from
// `model->context_cap`. Returns false for a blob too small to hold the header or carrying the wrong
// magic (a v1 `'SSLM'` or v2 `'SLM2'` blob, or corrupt/foreign data) -- the same "malformed blob"
// disposition the caller already assigns a rejecting status to; `*out_workspace_size` is untouched on
// a false return.
bool PeekGpuSeqBlobWorkspaceSize(const void* blob, size_t blob_size, uint64_t* out_workspace_size);

// (design Sec4.2/Sec22): the identity twin of `PeekGpuSeqBlobWorkspaceSize` above -- reads only the v3
// blob's own header (magic + `model_content_hash`), never the body, never a device call, so
// `sslm_gpu_seq_restore` can compare the blob's own recorded model identity against the TARGET model
// handle's `RawIntegrityHash()` before any device work, per design Sec4.2's own "checked after the
// size-derivation ladder... and before any device work" ordering. Returns false under the identical
// "malformed blob" disposition `PeekGpuSeqBlobWorkspaceSize` uses (too small to hold the header, or
// wrong magic -- a v1/v2 blob predates this field entirely); `*out_hash` is untouched on a false
// return.
bool PeekGpuSeqBlobModelHash(const void* blob, size_t blob_size,
                              std::array<uint8_t, superslm::kIntegrityHashBytes>* out_hash);

// --- B3 (Sec5.1, Sec11 B3): descriptor-table binding substrate + int8-native
// packing. Maps a synthetic multi-array fixture (sized to a real tier's layer count
// x 42-array-pointer shape, Sec4) through the table-based binder and reads back
// every bound array's known-pattern content, including negative int8 values
// (Sec5.2's corrected sign-extension). ---
bool BindDescriptorTableAndReadback(const int8_t* const* array_pointers,
                                     const size_t* array_element_counts, size_t n_arrays,
                                     int8_t* out_readback_concat, size_t out_readback_capacity);
// Two-handle cell (Sec11 B3, added Sec14 Fold G1): map, write a distinguishing
// pattern, release; map a second, different handle into the freed descriptor-heap
// region; report whether the second handle's own shader read shows residue from the
// first.
bool DescriptorHeapRegionIsCleanAfterHandleRelease();

// Added Sec21 Fold (amendment 5): Resource Binding Tier 1/2
// feature-query fallback. `ArmResourceBindingTierMock(tier)` (1 or 2) makes the next
// `MapModelGpuResidencyTierCheck()` call observe a mocked
// `D3D12_FEATURE_DATA_D3D12_OPTIONS::ResourceBindingTier` result below the Tier-3
// this design's binding architecture requires -- every device this
// project's probes have run on reports Tier 3, so this is the only way to exercise
// the sub-Tier-3 path. Returns true iff the GPU path proceeds (Tier 3 or unmocked);
// false iff it returns the defined "GPU path unsupported on this device" status
// having built no descriptor table, no UAV table, and no device allocation.
void ArmResourceBindingTierMock(int tier);
void ClearResourceBindingTierMock();
bool MapModelGpuResidencyTierCheck();

// --- B12 (Sec10 dim 5, Sec11 B12, redesigned Sec21 Fold, amendment
// 4): deterministic allocation-failure injection, not real VRAM exhaustion (which is
// non-deterministic and therefore not a valid gate, per this design's own
// reasoning). This retires the original real-VRAM-exhaustion construction (a single
// `GpuResidencyMapReturnsDefinedFailureOnOverBudgetRequest` symbol) as this step's
// correctness gate -- superseded, not merely extended, since a gate that sometimes
// passes for the wrong reason is not a gate; a real memory-pressure configuration
// remains a permitted OPTIONAL SMOKE (this design's own allowance), not authored as a
// symbol here since this suite does not run it. `sslm_model_map`'s GPU-residency
// setup makes a fixed, enumerable sequence of `kGpuResidencyAllocationCallCount`
// ("N") mock allocation calls. ---
// This comment previously stated N's own provenance as "Sec5.1's read-resource
// and write-resource tables" -- a later remedy corrected that claim at
// its definition site and left this declaring header still stating it, a
// claim surviving in exactly one file for three consecutive rounds.
// N's SINGLE definition, with its own current and honestly-
// stated provenance, is `src/gpu/superslm_gpu.cpp`'s
// `kGpuResidencyAllocationCallCount` -- this header does not restate that
// provenance a second time, structurally, so it cannot go stale here again:
// read the definition site, not this comment, for what N counts and why.
extern const uint32_t kGpuResidencyAllocationCallCount;
// Arms injection: the next `MapModelGpuResidencyWithInjection` call fails exactly
// allocation call `index` (0-based, in [0, N)) with a synthetic out-of-memory result,
// succeeding on every other call in the sequence.
void ArmAllocationFailureInjection(uint32_t index);
// Arms the budget preflight mock: the next `MapModelGpuResidencyWithInjection` call
// observes `IDXGIAdapter3::QueryVideoMemoryInfo`'s reported budget as `mocked_budget_bytes`
// rather than the real device's own value.
void ArmLowBudgetInjection(uint64_t mocked_budget_bytes);
void ClearAllocationInjection();
// The mock allocator's own live-allocation count -- must read 0 after a failed-and-
// cleaned-up map (transactional-cleanup verification).
uint32_t LiveAllocationCount();
// How many of the N allocation calls were actually attempted on the most recent
// `MapModelGpuResidencyWithInjection` call -- must read 0 when the low-budget
// preflight fires (it returns before attempting any of the N calls).
uint32_t AllocationCallsAttempted();
// Returns true iff the GPU-residency map succeeds cleanly (Ok); false iff it returns
// the defined failure status this step names (unchanged by the redesign).
bool MapModelGpuResidencyWithInjection(uint64_t required_bytes);

}  // namespace superslm_gpu

// T-2851 test-build seams (SUPERSLM_GPU_ALLOC_FAULT_INJECTION only; T-2851 design Sec4.6 items 2
// and 4, Sec4.9), defined in gpu_1p0.cpp. Global scope with C++ linkage, not in superslm_gpu,
// because the red suite (tests/t2956-token-finish-red-suite/) declares them there itself. None is
// in a product build. `hr` is an HRESULT (Windows' `long`), spelled `long` so this header needs no
// <windows.h>.
#if defined(SUPERSLM_GPU_ALLOC_FAULT_INJECTION)
struct SslmGpuContext;
struct SslmGpuModelHandle;
enum class SslmGpuStatus : uint32_t;

// Occurrence-indexed allocation faults. Every device allocation passes through one counted
// primitive (harness::Device::TryMakeBuffer).
//   SslmGpuAllocCounterResetForTest: zero the count and the in-bundle record.
//   SslmGpuAllocCountForTest: allocations since the last reset.
//   ArmGpuAllocFaultAtOccurrence: the k-th allocation after arming (1-based) returns `hr` without
//     calling D3D12, once; k = 0 disarms.
//   SslmGpuAllocInBundleForTest: whether the k-th allocation since the last reset ran inside the
//     device logits bundle's creation (a test's expected-status choice only; it never selects
//     what is faulted).
void SslmGpuAllocCounterResetForTest() noexcept;
uint32_t SslmGpuAllocCountForTest() noexcept;
void ArmGpuAllocFaultAtOccurrence(uint32_t k, long hr) noexcept;
bool SslmGpuAllocInBundleForTest(uint32_t k) noexcept;
// The map-time twin of ArmPrefillGuardDeviceRemovedQueryInjection: the next removed-device query
// made while classifying a map-time allocation failure reports removed. Single-shot.
void ArmGpuMapDeviceRemovedQueryInjection() noexcept;
// The next device logits read-back row has element `row` replaced by `value` before it is
// returned. Single-shot.
void ArmGpuDeviceLogitsReadbackOverride(int32_t row, int64_t value) noexcept;
// Phase-B Close faults: the next `consecutive_failures` Close attempts made inside the device
// logits bundle's creation or its run fail without closing the list. 1: the retry succeeds, the
// call returns SSLM_DEVICE_LOST and the context stays usable. 2: both attempts fail and the
// context is left in the documented terminal state. 0 disarms.
void ArmGpuDeviceLogitsCloseFaultForTest(uint32_t consecutive_failures) noexcept;
// The model handle's host lm_head copy, in bytes: 0 for a tied head or a device-resident head,
// vocab_size x hidden_size for an untied head mapped without the flag.
size_t SslmGpuHeadWeightCopySizeForTest(const SslmGpuModelHandle* model) noexcept;

// One device logits bundle driven directly, through the model map's own creation path
// (CreateDeviceLogitsBuffers) and the finish's own run (RunDeviceLogits).
struct SslmGpuDeviceLogitsBundleForTest;
SslmGpuStatus SslmGpuDeviceLogitsBundleCreateForTest(
    SslmGpuContext* ctx, const int8_t* head_rows, uint32_t vocab_size, uint32_t hidden_size,
    SslmGpuDeviceLogitsBundleForTest** out_bundle) noexcept;
SslmGpuStatus SslmGpuDeviceLogitsRunForTest(SslmGpuContext* ctx,
                                            SslmGpuDeviceLogitsBundleForTest* bundle,
                                            const int8_t* x_codes, int64_t* wide_out) noexcept;
// GPU virtual addresses of the bundle's head, x staging, x, output row and readback buffers, in
// that order: identical before and after a run shows the run reused the bundle's buffers.
void SslmGpuDeviceLogitsBundleResourcesForTest(const SslmGpuDeviceLogitsBundleForTest* bundle,
                                               uint64_t out_gpu_vas[5]) noexcept;
void SslmGpuDeviceLogitsBundleDestroyForTest(SslmGpuDeviceLogitsBundleForTest* bundle) noexcept;
#endif  // SUPERSLM_GPU_ALLOC_FAULT_INJECTION

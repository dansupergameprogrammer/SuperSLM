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
superslm::SslmForwardStatus RunLayerLoopGpu(superslm::SequenceLayerState& seq,
                                             const superslm::LayerWeights* layers,
                                             uint32_t num_hidden_layers, uint32_t layer_budget,
                                             size_t hidden_size, size_t head_dim,
                                             size_t num_key_value_heads, size_t intermediate_size,
                                             int64_t context_cap,
                                             const superslm::SslmTensorManifest& rope_tables,
                                             uint8_t* workspace, size_t workspace_size,
                                             ID3D12Resource* external_kv_resident = nullptr,
                                             bool* io_external_kv_needs_resume_barrier = nullptr);

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
    const GpuAdapterBridge* adapter_bridge = nullptr);

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
superslm::SslmForwardStatus RunLayerLoopGpuFinish(GpuLayerLoopInFlight* inflight,
                                                    superslm::SequenceLayerState& seq,
                                                    uint8_t* workspace, int32_t block,
                                                    int32_t* out_ready);

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
superslm::SslmForwardStatus SubmitChunkToFullDepthForG5Bridge(
    superslm::SequenceLayerState& seq, const superslm::LayerWeights* layers,
    uint32_t num_hidden_layers, size_t hidden_size, size_t head_dim, size_t num_key_value_heads,
    size_t intermediate_size, int64_t context_cap, const superslm::SslmTensorManifest& rope_tables,
    uint8_t* workspace, size_t workspace_size, const uint8_t* chunk_embedding_bytes,
    uint32_t chunk_len, ID3D12Resource* external_kv_resident,
    bool* io_external_kv_needs_resume_barrier, ID3D12Resource* external_weights_resident,
    ID3D12Resource* external_rope_cos_resident, ID3D12Resource* external_rope_sin_resident,
    bool external_rope_has, uint64_t external_rope_cos_elems, uint64_t external_rope_sin_elems,
    const GpuAdapterBridge* adapter_bridge, GpuLayerLoopInFlight** out_inflight);

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

// The device-observable residency cache was missing. Content
// correctness alone (a byte-for-byte match against the cached row) cannot
// distinguish "read from the resident DEFAULT-heap buffer" from "read from a
// plain upload-heap buffer holding the identical bytes" -- both would pass a
// value-comparison pin identically. Set internally by `RunLayerLoopGpu`'s own
// weight-residency decision, every call: `true` iff that call's own packed
// row matched `g_resident_weights`' cached content and the DEFAULT-heap
// upload/copy/transition sequence was skipped entirely (a cache hit); `false`
// on a cache miss (a fresh upload+copy ran, whether because the content
// changed or because this is the first call). Read back by the caller AFTER
// `RunLayerLoopGpu` returns -- this is the pin round's own observable to
// consume, not built as a test cell here (Brunel does not author tests).
//
// "every call" above is now true of a REJECTING call too, corrected
// from a code defect (not a doc defect) this round found and fixed --
// `RunLayerLoopGpu` used to leave this flag holding the PREVIOUS call's
// value across any of its eleven rejecting return paths, reproduced by
// execution reading back a stale, sometimes-wrong `skipped` value after a
// guard-rejected call. A rejecting call makes no weight-residency decision
// at all, so it now reads `false` ("no upload was skipped") on every one of
// those eleven paths -- set at function entry, before the first guard.
//
// CORRECTED 2026-08-14: "eleven"
// above undercounted by one -- the recording-window catch (`superslm_gpu.cpp`,
// added the same round this paragraph was written) is a TWELFTH
// rejecting return path, and it was not among the eleven this paragraph's
// own fix touched: a cache-hit call that throws inside the try left this
// flag reading the call's own stale `true` even though the catch had just
// invalidated the cache it describes. Now set `false` at the catch site too,
// beside the cache invalidation -- every one of the now-TWELVE rejecting
// paths reads `false`.
//
// CORRECTED 2026-08-14 (line citations
// refreshed twice, across several same-day commits that each shifted the lines
// they pointed at; CONVERTED, from line-number pointers to SYMBOL/ANCHOR
// references -- a line number drifts on any unrelated edit that shifts source;
// a symbol name or a named anchor comment does not drift until the symbol
// itself is renamed, which `check_symbol_integrity`
// (`tests/ci/check_gpu_guard_status_parity.py`) already catches. The CLAIM
// below is unchanged from its own original statement, only HOW it points at
// `superslm_gpu.cpp` has changed twice, first refreshed, now removed as a
// class): "twelve" above is
// also wrong -- the correction above counted the function's rejecting
// RETURNS (its guard ladder), not its rejecting return PATHS, and never
// re-derived the number from the function itself. Enumerated fresh, at
// source, every rejecting `return superslm::SslmForwardStatus::` (or
// return-via-ternary) `RunLayerLoopGpu` has: eleven in the nine-guard
// ladder (`InvalidLayerBudget` (`superslm_gpu.cpp`), `InvalidContextCap`
// (`superslm_gpu.cpp`), `HeadDimGeometryMismatch` (`superslm_gpu.cpp`),
// `KvHeadGeometryMismatch` (`superslm_gpu.cpp`), `WorkspaceTooSmall`
// (`superslm_gpu.cpp`), `InvalidHiddenCodes` (`superslm_gpu.cpp`),
// `SequenceAlreadyComplete` (`superslm_gpu.cpp`), `PositionOverCap`
// (`superslm_gpu.cpp`), `KvCapacityExhausted` (`superslm_gpu.cpp`) --
// eleven RETURN STATEMENTS realizing nine distinct guards, two
// of them, `InvalidContextCap`/`WorkspaceTooSmall`, each with two return
// sites); two device-capability rejections below the ladder (the
// `dev.available` (`superslm_gpu.cpp`) check, the sub-Tier-3
// `MapModelGpuResidencyTierCheck` (`superslm_gpu.cpp`)
// check, both `KvPrecisionUnsupported`); and the recording-window catch
// itself (`device_removed_reason` (`superslm_gpu.cpp`), one return statement, a ternary choosing between
// `GpuDeviceRemoved`/`GpuAllocationFailed`). **Fourteen**, not twelve --
// CORRECTED AGAIN below: fourteen is also short by one.
// `g_last_weight_upload_was_skipped`'s own FOUR write sites (corrected from
// "three" in the same sentence that listed four citations --
// the static init (`g_last_weight_upload_was_skipped` (`superslm_gpu.cpp`)),
// the function-entry write anchored
// `lwuws_write_function_entry` (`superslm_gpu.cpp`), the residency-decision
// write (`weights_resident` (`superslm_gpu.cpp`)), the catch's own write
// anchored `lwuws_write_catch` (`superslm_gpu.cpp`)) still cover all fourteen of THIS
// paragraph's own paths correctly -- the residency-decision write is the only conditional write
// and it sits below both device-capability rejections, so those two paths read the
// entry-set `false` unchanged; this correction is to the COUNT, not to the
// code, which was already right as far as this paragraph's own scope went.
// The `!dev.available`/Tier-3 half of this claim is derived by inspection
// of the write sites, not executed -- forcing either condition is outside
// what this project's own test harness can do on real hardware.
//
// CORRECTED 2026-08-14: "fourteen" above
// is short by one, and -- unlike the three corrections before it -- the
// property being counted is FALSE, not merely mis-numbered, so this
// correction does not just renumber it. `RunLayerLoopGpu`'s own TERMINAL
// statement, `return DecodeStickyTag(sticky_tag);` (`DecodeStickyTag`
// (`superslm_gpu.cpp`)),
// is neither a ladder return nor the catch's ternary -- it is a FUNCTION
// CALL whose result is returned directly, and `DecodeStickyTag` (`superslm_gpu.cpp`)
// maps the device's own sticky tag to fourteen statuses, THIRTEEN of them
// rejecting (`ChainInputOutOfDomain`, `RopeTableTensorMissing`, and eleven
// more -- only tag 0, `Ok`, is non-rejecting). That is a FIFTEENTH rejecting
// return path, it is not hypothetical, and this suite already drives it:
// `TestT2053_Item3_LastWeightUploadWasSkipped`'s own call 3 rejects through
// exactly this path (`ChainInputOutOfDomain`, a real content-mutation
// fixture, not an injected fault) -- **executed and measured, that call
// reads `LastWeightUploadWasSkipped() == true`.**
//
// This falsifies the property every version of this paragraph has stated
// previously: "every rejecting path reads `false`." On the sticky-tag
// path the upload genuinely WAS skipped (the call was a real cache hit;
// the rejection happens deep inside the per-layer dispatch, after the
// weight-residency decision already ran and read `true`), so `true` is the
// HONEST answer and the accessor's own primary contract --
// `"true iff this call's own upload was skipped"` -- is satisfied on this
// path. Making this path read `false` would require lying about a skipped
// upload; that is not available as a fix. What changes is the SENTENCE, to
// the smaller promise that is actually true, per `StandardsDocument.md`
// §5.6:
//
// CORRECTED 2026-08-19 (T-2184, Claude/Poirot/efeb9ba-t2184-t2169-gpu-batched-prefill-review.md,
// S3; D-SLM3662): T-2169's own chunk-submission primitive, `SubmitOneSubChunkToFullDepthForG5
// Bridge` (`superslm_gpu.cpp`), calls `PrepareGpuLayerLoopChunkOpenState` for its own chunk-open
// (the SAME ladder/device-capability region the "before" count already reads, not duplicated) and
// then carries its OWN two catch clauses -- `GpuGemmGroupArithmeticError`'s own literal return,
// and the generic `std::runtime_error` one's own ternary -- each calling the identical shared
// `InvalidateResidencyCachesOnThrow()` (T-2184's own S3 remedy factored the lambda this paragraph
// already named into one file-scope helper both `RunLayerLoopGpuSubmit` and this primitive call,
// so "before every one of their own returns" below is now a compile-time guarantee, not a
// re-derived fact) before every one of their own returns -- two more paths that never reached a
// residency decision. Its own terminal `return superslm::SslmForwardStatus::Ok;` (handing the
// caller an in-flight token, the identical async-submission-succeeded shape `RunLayerLoopGpuSubmit`
// already contributes one of) is one more path that DOES read whatever the residency decision
// already decided. The two counts below are updated to name three functions, not two.
//
// **The true contract, stated precisely rather than as a path count:**
// `LastWeightUploadWasSkipped()` reflects THIS CALL's own weight-residency
// decision. It reads `false` on every path that returns BEFORE that
// decision runs (the `weights_resident` write above) -- the nine-guard ladder, the two
// device-capability rejections, and the recording-window catch, seventeen
// paths in all (the recording window now carries FOUR catch clauses across two functions --
// `RunLayerLoopGpuSubmit`'s own `GpuGemmGroupArithmeticError`/`std::runtime_error` pair, and
// `SubmitOneSubChunkToFullDepthForG5Bridge`'s own identical pair -- all four counted, since all
// four call the shared cache-invalidation helper before every one of their own returns), none of
// which ever reached a residency decision to report.
// It reads exactly `weights_resident` (`true` on a cache hit, `false` on a
// miss) on every path that returns AFTER the decision -- **re-derived after
// this file's own (B5) split of the single function this paragraph originally
// described into `RunLayerLoopGpuSubmit` (the guard ladder, the residency
// decision, and everything above) and `RunLayerLoopGpuFinish` (the fence-
// wait and the readback), then again (T-2184) for `SubmitOneSubChunkToFullDepthForG5Bridge`'s own
// terminal success return -- the "after" population now has six members
// instead of one, not because any NEW decision-making was added, but
// because the single old "keep going toward the terminal decode" fall-
// through is now six separate real return statements across three
// functions: `RunLayerLoopGpuSubmit`'s own async-submission-succeeded
// return (`return ...::Ok;`, handing the caller an in-flight token),
// `SubmitOneSubChunkToFullDepthForG5Bridge`'s own identical terminal `Ok`
// return, and `RunLayerLoopGpuFinish`'s own four -- the non-blocking-poll-not-ready
// return, the caller-misuse null-token rejection, the terminal
// `return DecodeStickyTag(sticky_tag);` the sticky-tag-decoded path always
// ended on, and `RunLayerLoopGpuFinish`'s own catch clause (added the same
// section this paragraph documents, after a real device fault discovered
// DURING the wait/readback -- not during Submit's own recording -- was
// found escaping this function as an uncaught exception instead of the
// defined `GpuDeviceRemoved`/`GpuAllocationFailed` channel design Sec9
// promises; StandardsDocument.md Sec5.4, reproduced by execution before
// being fixed). None of these six re-decides or re-writes the flag --
// each reads whatever the residency decision already decided -- the six,
// and the seventeen before them, alike, twenty-three paths' own destination in
// total across the three functions, whether the decoded
// status is `Ok` or one of `DecodeStickyTag`'s thirteen rejecting statuses.**
// A caller that wants "did THIS call's upload run" reads this accessor for
// exactly that, on every path, correctly; a caller that reads it as "did
// this call SUCCEED" is reading a different question than the one it
// answers, on the sticky-tag paths specifically -- named here so a future
// reader does not have to re-derive it from the code a fourth time.
bool LastWeightUploadWasSkipped();

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
GpuGemmSiteGroupPlan ComputeGpuGemmSiteGroupPlan(GpuGemmSplitSite site, uint32_t hidden_size,
                                                  uint32_t kv_out_channels,
                                                  uint32_t intermediate_size);

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
#endif  // SUPERSLM_T2169_CHUNK_RECORDING_FAULT_INJECTION

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
// it returns -- and is testable without a device (no dispatch is actually issued to
// answer this question, only planned). `complete_layers = min(dispatch_budget / 24,
// num_hidden_layers - current_layer_position)`, floor division, 24 = the real per-layer
// dispatch count this design's own geometry ships (Sec3/Sec6.1 -- re-derived
// from the 17 = 16 sites + 1 commit figure this constant carried before B4 ported the
// dispatch chain onto its own production geometry). Returns
// SslmGpuStatus::DispatchBudgetTooSmall, `*out_layers_to_issue = 0`, for any
// `dispatch_budget` in [0, 23] -- floor division by 24 is uniformly zero there. Never
// records a partial layer. ---
enum class SslmGpuStatus { Ok, DispatchBudgetTooSmall, Busy };

// The ONE source for
// the real per-layer dispatch count -- `PlanDispatchBudgetGpu`'s own body (superslm_gpu.cpp)
// and `sslm_decode_step_batch_gpu`'s own budget-spend line (gpu_1p0.cpp,
// `remaining_budget -= layers_to_issue * kDispatchesPerLayer`) both read this constant rather
// than each carrying its own `24u` literal. Before this fix the two agreed only because no one
// had changed either copy since B4; a future change to one and not the other would have made
// the batch call's own unsigned subtraction wrap (an effectively unlimited budget for every
// later sequence in the same call), silently.
constexpr uint32_t kDispatchesPerLayer = 24;  // the real per-layer dispatch count

SslmGpuStatus PlanDispatchBudgetGpu(uint32_t dispatch_budget, uint32_t num_hidden_layers,
                                     uint32_t current_layer_position,
                                     uint32_t* out_layers_to_issue);

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
struct GpuLayerLayout {
	uint32_t off[56]{};
	uint32_t stride = 0;
};

GpuLayerLayout ComputeLayerLayout(uint32_t hidden_size, uint32_t kv_hidden_size,
                                   uint32_t num_kv_heads, uint32_t num_attention_heads,
                                   uint32_t intermediate_size);

// Packs `N` LayerWeights entries into one contiguous byte buffer at `layout`'s own
// stride/offsets -- the exact byte-for-byte transformation RunLayerLoopGpu's own
// weight-residency miss path has always performed, extracted verbatim (B2) so
// it has exactly one implementation. `H`=hidden_size, `KV`=num_kv_heads*head_dim,
// `NH`=num_kv_heads, `NQH`=num_attention_heads, `I`=intermediate_size -- the same five
// dimension values `ComputeLayerLayout` above must be called with to produce `layout`.
std::vector<uint8_t> PackLayerWeightsBytes(const superslm::LayerWeights* layers, uint32_t N,
                                            const GpuLayerLayout& layout, uint32_t H, uint32_t KV,
                                            uint32_t NH, uint32_t NQH, uint32_t I);

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
// `SslmModelView::RawIntegrityHash()` (design Sec5.1) -- written into the v3 blob's own new,
// append-only trailing field so a later restore against a DIFFERENT model can detect the mismatch
// the N1 size-admissibility widening left open (a foreign blob whose derived context_cap happens to
// be admissible against a same-shape-but-different model used to restore silently). Restore's own
// identity check is NOT threaded through this function -- it runs in the caller
// (`sslm_gpu_seq_restore`, gpu_1p0.cpp) via `PeekGpuSeqBlobModelHash` below, before this function (or
// the fresh handle it restores into) is ever reached, matching the peek-then-validate-then-allocate
// ordering the size-derivation ladder already established.
bool SaveGpuSequenceState(const superslm::SequenceLayerState& seq, size_t hidden_codes_size,
                           const uint8_t* workspace, size_t workspace_size,
                           const std::array<uint8_t, superslm::kIntegrityHashBytes>& model_content_hash,
                           void* out_blob, size_t* out_blob_size);
bool RestoreGpuSequenceState(const void* blob, size_t blob_size, superslm::SequenceLayerState* out_seq,
                              size_t hidden_codes_size, uint8_t* out_workspace, size_t workspace_size);

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

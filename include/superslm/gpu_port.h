#pragma once
// T-1986 GPU-serial port (Claude/Vitruvius/t1986-gpu-serial-port-design-2026-08-13.md,
// commit 2de2e388a6 -- Sec10 Coverage Model, Sec11 B1-B12 build decomposition, Sec5.8
// dispatch_budget contract, Sec5.9 asynchronous sequence lifecycle): the production
// entry-point surface T-2024's re-derived red suite (Claude/Curie/
// t2019-gpu-serial-red-suite-2026-08-13.md, dated section "T-2024 re-derivation")
// gates the build against.
//
// THIS BANNER IS HISTORICAL, NOT CURRENT (corrected 2026-08-14, T-2055,
// Claude/Poirot/db73b22-gpu-serial-port-final-confirmation-review.md, Minor
// 3, closing an M5/O10 finding this file carried across at least the
// 82cfca7, 36b9327, and db73b22 reviews without ever being routed for a
// fix): at this header's OWN origin (Sec11's B1-B12 decomposition, before any
// build round landed) every symbol below really was declared and not
// defined, exactly as the paragraph below states. It is false of the tree
// today -- every function this header declares has its real definition in
// `src/gpu/superslm_gpu.cpp` (T-2024 through T-2052 landed them one B-step at
// a time), and this round added a construct, `enum class GpuLayerLoopGuard`
// directly below this banner, that this banner's own claim cannot even be
// true OF: an enum has no separate declaration/definition split to be
// pending. Left below for its historical value (the precedent citation, and
// what this file's role was AT ITS ORIGIN) -- read it as describing how this
// header started, not its current link-completeness, which build.bat's own
// green link proves every time it runs:
//
// EVERY SYMBOL BELOW IS DECLARED, NOT DEFINED. This header exists so the red suite
// compiles; it links to nothing until the build seat (Brunel, B1-B12) provides a
// definition. This is test-authoring infrastructure -- the call SHAPE the suite
// needs -- not the shipped design: the build seat's own dispatch recording,
// descriptor-table binding, and D3D12 plumbing are unconstrained by this header
// beyond the input/output contract each red-suite test asserts against. Mirrors the
// precedent this project already used for the identical situation (T-1899's red
// suite for T-1894 Option-G, `artifact.h`/`checked_chain_funnel.h`/`forward_sites.h`,
// commit 7f55b9d): "contract extensions, declared not defined... link fails on
// exactly these N symbols."
//
// Function groups below are named by the B-step (Sec11) whose gate they discharge.
// Each one's own test cell(s) are filed in tests/test_main.cpp (search "T-2019") and
// derived in full in the Curie casebook cited above.

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "superslm/checked_chain_funnel.h"
#include "superslm/forward_sites.h"

// T-2113 (B3): forward-declared at GLOBAL scope (matching where d3d12.h itself declares
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
// own compound predicate (Sec11 B2 (3), D-SLM3053), for k_bias and v_bias
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
// T-2113 (B3, design Sec5.3/Sec10 B3): the trailing pair below is ADDITIVE -- every
// existing caller (enumerated at the B3 section of Claude/Brunel/t2113-1p0-core-build-
// 2026-08-15.md: ~40 call sites across tests/test_main.cpp, tools/t2039_c5_harness.cpp,
// tools/t2100_gpu_throughput.cpp) passes neither argument and defaults them to nullptr,
// which reproduces this function's PRE-B3 behavior byte-for-byte (the process-global
// g_resident_kv single-slot cache, unchanged). `external_kv_resident`, when non-null,
// is a caller-owned, ALREADY-ALLOCATED DEFAULT-heap UAV buffer (an SslmGpuSequenceHandle's
// own dedicated K/V residency, design Sec5.3) this call binds directly as kv_uav --
// bypassing g_resident_kv and its pointer/size fast-hit check entirely for this call, so
// two sequences each passing their OWN buffer here never share, alias, or evict each
// other's K/V state, closing D-SLM3311's own class for the K/V cache by construction
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

// T-2113 (B5, design Sec4.2/Sec4.3/Sec6.2/Sec10 B5): the async submission boundary.
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
// `external_kv_resident` pair above (T-2113 B3), added here (B5) so a caller routed
// through the 1.0 API's own model handle (design Sec5.1, `SslmGpuModelHandle`) never
// touches the pre-1.0 `g_resident_weights`/`g_resident_rope` process-global caches at
// all -- this is what retires `g_resident_rope` for the handle-routed path (D-SLM3362's
// own dated deferral): when `external_weights_resident` is non-null, `layers` is NEVER
// dereferenced (the caller's own resident buffer is bound directly, no re-pack, no
// re-upload, no g_resident_weights lookup); when BOTH `external_rope_cos_resident`/
// `external_rope_sin_resident` are non-null, `rope_tables` is never read for the two
// big tables (only `external_rope_has`/`external_rope_cos_elems`/
// `external_rope_sin_elems` decide the small per-call RopeInfo presence/size fields a
// caller with no live `SslmTensorManifest` of its own can still supply directly) and
// `g_resident_rope` is never touched. Every existing caller passes none of the six and
// observes byte-for-byte pre-B5 behavior.
// T-2113 (B6b, design Sec5.2/Sec8/Sec10 B6): one covered (layer, projection) slot of a
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

// T-2113 (B6b, design Sec8): the adapter-delta dispatch bridge. `gpu_1p0.cpp`'s own
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

// T-2052 (Claude/Poirot/36b9327-gpu-serial-port-reconfirmation-review.md, M1's
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
// CORRECTED 2026-08-14 (T-2055, Claude/Poirot/db73b22-gpu-serial-port-final-
// confirmation-review.md, P1; D-SLM3183, superseding D-SLM3182's own claim):
// this comment's own last sentence overclaims. `kCount` and both
// `static_assert`s below compare a literal to `gpu_layer_loop_guards.def`'s
// own row count -- none of it, and no pin cell matched to that file's row
// order by eye, ever reads `forward_sites.cpp`, so a guard added to BOTH
// ladders with no `.def` row (proven by execution) leaves every one of these
// green. The class-level guarantee this paragraph describes is
// `tests/ci/check_gpu_guard_status_parity.py` (T-2055)'s job, not this
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

// T-2052 (item 3, Claude/Curie/t2019-gpu-serial-red-suite-2026-08-13.md §13.2):
// the device-observable N3's own residency cache was missing. Content
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
// T-2055 (Claude/Poirot/db73b22-gpu-serial-port-final-confirmation-review.md,
// P2): "every call" above is now true of a REJECTING call too, corrected
// from a code defect (not a doc defect) this round found and fixed --
// `RunLayerLoopGpu` used to leave this flag holding the PREVIOUS call's
// value across any of its eleven rejecting return paths, reproduced by
// execution reading back a stale, sometimes-wrong `skipped` value after a
// guard-rejected call. A rejecting call makes no weight-residency decision
// at all, so it now reads `false` ("no upload was skipped") on every one of
// those eleven paths -- set at function entry, before the first guard.
//
// CORRECTED 2026-08-14 (T-2062, Claude/Poirot/
// a3d44e7-gpu-serial-port-ship-confirmation-review.md, M-b): "eleven"
// above undercounted by one -- the recording-window catch (`superslm_gpu.cpp`,
// added by T-2055 the same round this paragraph was written) is a TWELFTH
// rejecting return path, and it was not among the eleven this paragraph's
// own fix touched: a cache-hit call that throws inside the try left this
// flag reading the call's own stale `true` even though the catch had just
// invalidated the cache it describes. Now set `false` at the catch site too,
// beside the cache invalidation -- every one of the now-TWELVE rejecting
// paths reads `false`.
//
// CORRECTED 2026-08-14 (T-2069, Claude/Poirot/
// b543abe-gpu-serial-port-ship-reverdict-review.md, M1; line citations
// refreshed 2026-08-14, T-2075, per that round's own M1 finding that all
// but one had gone stale across two same-day commits; refreshed AGAIN
// 2026-08-14, T-2084, S2, Claude/Poirot/42ecf79-gpu-serial-port-round9-
// review.md, D-SLM3247, after S2's own restoration of superslm_gpu.cpp's
// deleted T-2071 paragraph shifted every line below it; CONVERTED 2026-08-14,
// T-2094, Claude/Poirot/2a937f5-gpu-serial-port-class-closure-review.md
// (S1/S3), from line-number pointers to SYMBOL/ANCHOR references -- a line
// number drifts on any unrelated edit that shifts source; a symbol name or a
// named anchor comment does not drift until the symbol itself is renamed,
// which `check_symbol_integrity` (`tests/ci/check_gpu_guard_status_parity.py`)
// already catches. The CLAIM below is unchanged from what T-2069 wrote, only
// HOW it points at `superslm_gpu.cpp` has changed twice, first refreshed,
// now removed as a class): "twelve" above is
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
// CORRECTED AGAIN below (T-2075, S2): fourteen is also short by one.
// `g_last_weight_upload_was_skipped`'s own FOUR write sites (corrected from
// "three" in the same sentence that listed four citations, T-2075, M2 --
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
// CORRECTED 2026-08-14 (T-2075, Claude/Poirot/
// 72a9b0d-gpu-serial-port-final-review.md, S2; D-SLM3228): "fourteen" above
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
// since T-2055: "every rejecting path reads `false`." On the sticky-tag
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
// **The true contract, stated precisely rather than as a path count:**
// `LastWeightUploadWasSkipped()` reflects THIS CALL's own weight-residency
// decision. It reads `false` on every path that returns BEFORE that
// decision runs (the `weights_resident` write above) -- the nine-guard ladder, the two
// device-capability rejections, and the recording-window catch, fifteen
// paths in all (T-2101: the recording window now carries TWO catch clauses --
// `GpuGemmGroupArithmeticError`'s own, and the generic `std::runtime_error`
// one -- both counted, since both call the shared cache-invalidation lambda
// before every one of their own returns), none of which ever reached a
// residency decision to report.
// It reads exactly `weights_resident` (`true` on a cache hit, `false` on a
// miss) on every path that returns AFTER the decision -- **re-derived at
// T-2113 (B5), which split the single function this paragraph originally
// described into `RunLayerLoopGpuSubmit` (the guard ladder, the residency
// decision, and everything above) and `RunLayerLoopGpuFinish` (the fence-
// wait and the readback) -- the "after" population now has five members
// instead of one, not because any NEW decision-making was added, but
// because the single old "keep going toward the terminal decode" fall-
// through is now five separate real return statements across two
// functions: `RunLayerLoopGpuSubmit`'s own async-submission-succeeded
// return (`return ...::Ok;`, handing the caller an in-flight token), and
// `RunLayerLoopGpuFinish`'s own four -- the non-blocking-poll-not-ready
// return, the caller-misuse null-token rejection, the terminal
// `return DecodeStickyTag(sticky_tag);` the sticky-tag-decoded path always
// ended on, and `RunLayerLoopGpuFinish`'s own catch clause (added the same
// section this paragraph documents, after a real device fault discovered
// DURING the wait/readback -- not during Submit's own recording -- was
// found escaping this function as an uncaught exception instead of the
// defined `GpuDeviceRemoved`/`GpuAllocationFailed` channel design Sec9
// promises; StandardsDocument.md Sec5.4, reproduced by execution before
// being fixed). None of these five re-decides or re-writes the flag --
// each reads whatever `RunLayerLoopGpuSubmit` already decided -- the four,
// and the fifteen before them, alike, twenty paths' own destination in
// total across the two functions, whether the decoded
// status is `Ok` or one of `DecodeStickyTag`'s thirteen rejecting statuses.**
// A caller that wants "did THIS call's upload run" reads this accessor for
// exactly that, on every path, correctly; a caller that reads it as "did
// this call SUCCEED" is reading a different question than the one it
// answers, on the sticky-tag paths specifically -- named here so a future
// reader does not have to re-derive it from the code a fourth time.
bool LastWeightUploadWasSkipped();

// T-2101 (dispatch-overhead decomposition, follow-up to D-SLM3302/D-SLM3304): a per-call timing
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

// T-2101 (S3, code review 6d9e04e-t2101-gpu-throughput-review.md): the ceiling-division formula
// `ComputeGpuGemmSiteGroupPlan` (below) uses to turn an output width into a group count --
// `ceil(out_channels / threads_per_group)`, i.e. `(out_channels + threads_per_group - 1) /
// threads_per_group`. The property this formula must hold, for any caller: the returned group
// count `g` satisfies `g * threads_per_group >= out_channels` (every channel has a thread) and,
// when `g > 0`, `(g - 1) * threads_per_group < out_channels` (no wasted group).
uint32_t ComputeGpuGemmGroupCount(uint32_t out_channels, uint32_t threads_per_group);

// T-2101 (S3-prime, code review 6d9e04e-t2101-gpu-throughput-review.md, confirmation pass @
// f7026db): the sites whose GEMM step runs as its own multi-group dispatch
// (`down_proj_gemm_site.hlsl` and the siblings named there).
// T-2113 (B4, design Sec3/Sec6.1, D-SLM3341): `KvProj` added -- kv_proj's own GEMM step is now
// its own multi-group dispatch (`kv_proj_gemm_site.hlsl`), re-derived from Claude/Laplace/
// t2105-gpu-speed-ceiling-2026-08-14.md as NEW production work (this enumerator never existed
// in the pre-1.0 substrate; `kv_proj_site.hlsl`'s own header comment named the second-dispatch
// remedy this enumerator realizes).
enum class GpuGemmSplitSite { QProj, OProj, GateProj, UpProj, DownProj, KvProj };

struct GpuGemmSiteGroupPlan {
	uint32_t out_channels = 0;
	uint32_t threads_per_group = 0;
	uint32_t groups = 0;
	// T-2113 (B4): how the 256 threads of a group are split -- `lanes` threads cooperate on
	// one output channel via the transposed GEMM partition (`GemmCoalescedGpu`,
	// site_common.hlsli), so a group covers `256 / lanes` channels. lanes == 1 degenerates to
	// the legacy one-thread-per-channel partition with no reduction. The SAME value is passed
	// to the shader as the 11th root constant, so the group count and the shader's own
	// channel indexing cannot drift apart.
	uint32_t lanes = 1;
	uint32_t channels_per_group = 256;
};

// T-2101 (S3-prime): the ONE source `RunLayerLoopGpu`'s own dispatch call for `site` reads its
// `Dispatch(groups, 1, 1)` grid size from -- no local variable sits between this function's own
// return value and the `bind_and_dispatch` call, so there is nothing at the call site left to
// hand-edit independently of this function. `tests/test_main.cpp`'s own
// `TestT2101_ComputeGpuGemmSiteGroupPlan_RealDimensions` calls this SAME function directly, no
// device required, at the real model's own dimensions (hidden_size=1536, intermediate_size=8960)
// -- a mutation of this function's own body (the confirmation pass's own specified re-falsification
// method) is therefore observable by the pinned suite, not only by a manual C5/throughput run
// against real hardware.
//
// T-2113 (B4): `kv_out_channels` added -- `KvProj`'s own dispatch covers BOTH K's and V's
// channels packed into one grid (`kv_proj_gemm_site.hlsl`'s own header comment), so its own
// out_channels is 2*kv_hidden_size, passed in already doubled by the caller (the ONLY site
// that reads this parameter; every other site ignores it).
GpuGemmSiteGroupPlan ComputeGpuGemmSiteGroupPlan(GpuGemmSplitSite site, uint32_t hidden_size,
                                                  uint32_t kv_out_channels,
                                                  uint32_t intermediate_size);

// T-2101 (per-site decomposition, follow-up to D-SLM3312; per-dispatch parallelism, D-SLM3313's
// own follow-up): one GPU-measured millisecond figure per dispatch the most recent
// `RunLayerLoopGpu` call issued, in dispatch order (empty if the call was rejected before
// recording, or timed by fewer than one dispatch). Dispatch `d` within this call corresponds to
// layer `d / 24` and that layer's own `d % 24`-th call in the fixed, unconditional per-layer order
// `RunLayerLoopGpu` issues them: attn_norm, q_proj_gemm, q_proj, kv_proj_gemm, kv_proj,
// rope_guard, rope_commit, attention_score, softmax, context_accumulate, ctx_fold, o_proj_gemm,
// o_proj, attn_residual, mlp_norm, gate_proj_gemm, gate_proj, up_proj_gemm, up_proj, mlp_act,
// down_proj_gemm, down_proj, mlp_residual, commit (24 sites/layer -- T-2113 (B4, design
// Sec3/Sec6.1) re-derived from the 22-site count T-2101 shipped: `kv_proj_gemm` is a NEW
// dispatch (kv_proj no longer fused-and-single-dispatch, `kv_proj_gemm_site.hlsl`'s own header
// comment states why the split is now sound) and RoPE's own commit phase is a second, separate
// dispatch (`rope_commit_site.hlsl`) rather than a group-barrier-separated phase inside one).
// Summing every `d` with the same `d % 24` across however many layers a call processed gives
// that SITE's own total GPU time for the call -- `tools/t2100_gpu_throughput.cpp`'s own
// per-site table is exactly this sum.
std::vector<double> LastCallPerDispatchTimingsMs();

// T-2070 (D-SLM3215, S4, Claude/Poirot/b543abe-gpu-serial-port-ship-reverdict-review.md):
// T-2063's own always-declared LINK-RED form of this instrument (`ArmO11AllocationFailure
// Injection`/`ClearO11AllocationInjection`, undefined) is UNGATED, so merging it broke the
// arc's own executing acceptance gate -- `tests/test_main.cpp` is not a pre-build red suite the
// way T-1899's own LINK-RED precedent was (this file's own :9-18 banner describes the ORIGINAL,
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
// real bodies reproduces this ticket's own LINK-RED proof (`Claude/Curie/t2019-gpu-serial-
// red-suite-2026-08-13.md` S16.2/S17), on purpose, as the gate's own self-check.
//
// CORRECTED 2026-08-14 (T-2088, Claude/Poirot/8420005-gpu-serial-port-round11-review.md, S1;
// D-SLM3261): "Not defined by `build.bat`" and "the build seat's own trigger to arm this pin"
// (naming a future condition) were true from T-2070 to T-2071 and false since -- `build.bat`'s own
// test-binary compile line
// has defined `SUPERSLM_O11_ALLOC_INJECTION` alongside `SUPERSLM_ENABLE_BAD_ALLOC_INJECTION` on
// the default test-binary compile line since T-2071 landed the real bodies, five rounds ago. The
// trigger this paragraph describes as pending fired at `72a9b0d`; nothing in this file was edited
// to say so, because the rename T-2084 made (M2, the previous round) touched the `SUPERSLM_O11_
// ALLOC_INJECTION` mention one paragraph below
// this sentence without re-reading the paragraph it sits in (`StandardsDocument.md` §7). Believing
// this sentence and removing the flag drops the O11 gate silently: gate-on and gate-off both then
// report the SAME 3-failure count (16 checks and both discrimination cells vanish with no test
// failure of any kind, D-SLM3261 S1's own executed measurement) -- the real, structural fix is the
// pinned assertion in `tests/ci/check_gpu_guard_status_parity.py`
// (`check_build_bat_defines_o11_gate`), not this sentence alone.
//
// CORRECTED 2026-08-14 (T-2091, Claude/Poirot/2aceac3-gpu-serial-port-ship-candidate-review.md,
// M1; build log §27): the block above's own header said T-2088 while attributing the rename it
// describes to "this same round" -- the rename was T-2084's own M2, the PREVIOUS round; "this
// same round" made the two the same round, which they were not. Corrected to the wording
// D-SLM3262 already had right ("the rename this same arc made at T-2084 (M2)").
//
// Mirrors B12's own arm/inject naming (ArmAllocationFailureInjection/ClearAllocationInjection):
// arms injection so the NEXT allocation call AT THE NAMED SITE in RunLayerLoopGpu fails with a
// synthetic allocation error; the call after clears it.
//
// CORRECTED 2026-08-14 (T-2080, Claude/Poirot/94ebee3-gpu-serial-port-closing-review.md, S1;
// D-SLM3241, superseding the T-2076 note below -- left standing, not rewritten, per this tree's
// own append-only discipline): T-2075's own fix MOVED the arm site from the weight DEFAULT-heap
// allocation to `work_scratch_uav`, which made the M-b half of `TestT2063_S1Mb_...` live for the
// first time -- and, measured by this review, made T-2062's OWN S1 remedy (the weight allocation's
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
// T-2114 (S4, Claude/Poirot/50f3d5d-t2113-1p0-gpu-core-build-review.md): a third site, inside
// `RestoreGpuSequenceState`'s own device round-trip (superslm_gpu.cpp) -- pins that
// `sslm_gpu_seq_restore`, defined in gpu_1p0.cpp, now catches an exception thrown from that
// call and returns SSLM_DEVICE_LOST instead of letting it escape the status-returning API
// boundary (the same boundary B5 already closed for `sslm_gpu_ready`/`RunLayerLoopGpuFinish`).
constexpr uint32_t kO11AllocInjectionSiteSeqRestore = 2;
//
// T-2076 note (Claude/Curie/t2019-gpu-serial-red-suite-2026-08-13.md): the definitions built by
// T-2071 targeted `work_scratch_uav`'s own allocation, not the weight DEFAULT-heap buffer this
// comment originally specified -- T-2075's own S1 fix moved the arm site there so an armed call
// reaches the throw on a cache HIT too, which the weight buffer's own allocation (gated behind
// `!weights_resident`) cannot. The function names below are unchanged; only the site they arm
// moved. `TestT2063_S1Mb_WorkScratchUavAllocationThrow_ReturnsGpuAllocationFailed_SkippedFalse`
// (tests/test_main.cpp) is gated identically -- it does not compile into the default build at all,
// so it cannot be the thing that fails the default build's own link.
//
// CLARIFIED 2026-08-14 (T-2091, Claude/Poirot/2aceac3-gpu-serial-port-ship-candidate-review.md's
// own class sweep; build log §27): "the default build" above means the CMake target (`superslm_tests`,
// `CMakeLists.txt`) -- the ONLY build configuration where `SUPERSLM_O11_ALLOC_INJECTION` is ever
// undefined, since `build.bat`'s own default test-binary line has defined it since T-2071 (S1,
// this same file's own dated correction below). Not a correction to the CLAIM (both builds
// genuinely never compile this cell without the macro) -- a disambiguation of "default," the exact
// word S1 found already misleading readers about a DIFFERENT sentence in this same paragraph
// group. See `derive_execution_scope`/`check_execution_scope_waivers` (`check_gpu_guard_status_parity.py`)
// for the machine-derived, standing-waived statement of which build compiles what.
//
// CORRECTED 2026-08-14 (T-2094, Claude/Poirot/2a937f5-gpu-serial-port-class-closure-review.md, M2;
// build log §28): the CLARIFIED block above is itself false -- `build.bat`'s own C5-harness `cl`
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

// Read back the device-resident K/V cache in the SAME layout and argument order
// superslm::KeyRow/ValueRow already define (forward_sites.h) -- the GPU port's
// `workspace` is the device-resident twin of the identical buffer RunLayerLoop uses.
const int8_t* KeyRowGpu(const uint8_t* workspace, uint32_t layer, int64_t context_cap,
                         size_t num_kv_heads, size_t head_dim, size_t kv_head,
                         int64_t position);
const int8_t* ValueRowGpu(const uint8_t* workspace, uint32_t layer, int64_t context_cap,
                           size_t num_kv_heads, size_t head_dim, size_t kv_head,
                           int64_t position);

// --- B7 (Sec5.8, Sec11 B7, D-SLM3069/3070/3072): the dispatch_budget contract, whole-
// layer quanta. This is the ARITHMETIC/policy half of `sslm_decode_step_gpu`'s own
// contract -- how many layers a call at a given budget will record, and what status
// it returns -- and is testable without a device (no dispatch is actually issued to
// answer this question, only planned). `complete_layers = min(dispatch_budget / 24,
// num_hidden_layers - current_layer_position)`, floor division, 24 = the real per-layer
// dispatch count this design's own geometry ships (Sec3/Sec6.1, T-2113 B4 -- re-derived
// from the 17 = 16 sites + 1 commit figure this constant carried before B4 ported the
// dispatch chain onto its own production geometry). Returns
// SslmGpuStatus::DispatchBudgetTooSmall, `*out_layers_to_issue = 0`, for any
// `dispatch_budget` in [0, 23] -- floor division by 24 is uniformly zero there. Never
// records a partial layer. ---
enum class SslmGpuStatus { Ok, DispatchBudgetTooSmall, Busy };

// T-2114 (M1, Claude/Poirot/50f3d5d-t2113-1p0-gpu-core-build-review.md): the ONE source for
// the real per-layer dispatch count -- `PlanDispatchBudgetGpu`'s own body (superslm_gpu.cpp)
// and `sslm_decode_step_batch_gpu`'s own budget-spend line (gpu_1p0.cpp,
// `remaining_budget -= layers_to_issue * kDispatchesPerLayer`) both read this constant rather
// than each carrying its own `24u` literal. Before this fix the two agreed only because no one
// had changed either copy since B4; a future change to one and not the other would have made
// the batch call's own unsigned subtraction wrap (an effectively unlimited budget for every
// later sequence in the same call), silently.
constexpr uint32_t kDispatchesPerLayer = 24;  // T-2113 (B4): the real per-layer dispatch count

SslmGpuStatus PlanDispatchBudgetGpu(uint32_t dispatch_budget, uint32_t num_hidden_layers,
                                     uint32_t current_layer_position,
                                     uint32_t* out_layers_to_issue);

// T-2113 (B2, design Claude/Vitruvius/t2107-gpu-core-1p0-design-2026-08-14.md Sec10 B2):
// GpuLayerLayout/ComputeLayerLayout/PackLayerWeightsBytes, promoted from
// src/gpu/superslm_gpu.cpp's own internal implementation (T-2035/T-2039, unchanged bodies)
// to this shared header so BOTH the pre-1.0 substrate's own RunLayerLoopGpu (superslm_gpu.cpp,
// call site unchanged by this move) and the 1.0 API's own model-handle upload path
// (src/gpu/gpu_1p0.cpp, sslm_gpu_model_map) compute weight-packing bytes from exactly ONE
// implementation -- never two copies that could silently drift, the same hazard
// ResidentWeights' own header comment (superslm_gpu.cpp) already names for a pointer-only
// cache key. Only the DECLARATION moved; every function body is byte-for-byte the loop it
// replaces at its old call site -- see the B2 section of
// Claude/Brunel/t2113-1p0-core-build-2026-08-15.md for the extraction's own before/after
// citation. `off[56]`/`stride` match every `*_site.hlsl`'s own `Layout.Load<uint>(N*4)` index
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
// weight-residency miss path has performed since T-2035, extracted verbatim (T-2113 B2) so
// it has exactly one implementation. `H`=hidden_size, `KV`=num_kv_heads*head_dim,
// `NH`=num_kv_heads, `NQH`=num_attention_heads, `I`=intermediate_size -- the same five
// dimension values `ComputeLayerLayout` above must be called with to produce `layout`.
std::vector<uint8_t> PackLayerWeightsBytes(const superslm::LayerWeights* layers, uint32_t N,
                                            const GpuLayerLayout& layout, uint32_t H, uint32_t KV,
                                            uint32_t NH, uint32_t NQH, uint32_t I);

// --- Sec5.9 (D-SLM3076/3077/3078/3079): the asynchronous sequence lifecycle, Idle ->
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
// Model-wide (D-SLM3079): true iff ANY sequence created against the model handle is
// Submitted, not only a single targeted one.
bool CallProceedsOrBusy_ModelUnmap(bool any_sequence_submitted);

// `sslm_gpu_ready`'s dual role (D-SLM3078): exempt from SSLM_BUSY by construction (the
// one call legal against a Submitted sequence); collapses Submitted -> Completed ->
// Idle in one call once the device fence signals. `fence_signaled` is the test's own
// simulated fence state; `*out_ready` mirrors the real ABI's own out-parameter
// (0 = still Submitted, ordinary polling; 1 = collapsed to Idle, aggregated status in
// `*out_status`).
bool GpuReadySignalsCompletion(bool fence_signaled, int32_t* out_ready,
                                superslm::SslmForwardStatus* out_status);

// --- B8 (Sec11 B8): device-to-host / host-to-device round-trip of the
// residual-stream/KV/kv_saturation_count/context_length state (Sec10 dim 9,
// D-SLM3034/D-SLM3080), mechanism-level -- SuperSLM_Plan.md Sec12's `sslm_seq_save`/
// `restore` C-ABI wrapper is not yet built on ANY backend (grep of this tree at
// main@727e63e finds no such symbol), so this red suite gates the device-resident
// round-trip mechanism the design commits to, not a specific ABI symbol name; named
// explicitly in the Curie casebook rather than silently assumed. **Restore-time
// device allocation and the host-to-device upload happen inside
// `RestoreGpuSequenceState` itself (D-SLM3080, Sec5.9) -- never inside a model-map-
// level call** -- `out_seq`/`out_workspace` are caller-owned, sized identically to
// the save call's own `seq`/`workspace`, and this function's own definition is where
// the build seat's restored-sequence device allocation is sited. ---
// T-2114 (C1, Claude/Poirot/50f3d5d-t2113-1p0-gpu-core-build-review.md): `hidden_codes_size`
// gates whether the residual stream round-trips through the blob, exactly like
// `workspace_size` already gates the K/V bytes -- 0 means the caller does not want
// hidden_codes carried (the pre-1.0 test_main.cpp cells that predate the 1.0 handle owning
// hidden_codes pass 0 and keep their own established contract); the 1.0
// `sslm_gpu_seq_restore` path passes the model's real hidden_size and gets the full
// round-trip. Save reads from `seq.hidden_codes`; Restore writes into `out_seq->hidden_codes`
// (both already-set pointers, never allocated here).
//
// T-2113 (P2, design Sec4.2/Sec22, routed `Claude/Poirot/50f3d5d-t2113-1p0-gpu-core-build-review.md`
// Sec15, D-SLM3415): `model_content_hash` is the SAVING model's own 32-byte
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

// T-2113 (N1, design Sec4.2/Sec21, routed `Claude/Poirot/50f3d5d-t2113-1p0-gpu-core-build-review.md`
// Sec8): reads only the blob's header (magic + `workspace_size`) -- never the body, never a device
// call -- so `sslm_gpu_seq_restore` can derive the blob's own `context_cap` (Sec5.1's K/V sizing
// formula) BEFORE it allocates the fresh handle, instead of always sizing that handle from
// `model->context_cap`. Returns false for a blob too small to hold the header or carrying the wrong
// magic (a v1 `'SSLM'` or v2 `'SLM2'` blob, or corrupt/foreign data) -- the same "malformed blob"
// disposition the caller already assigns a rejecting status to; `*out_workspace_size` is untouched on
// a false return.
bool PeekGpuSeqBlobWorkspaceSize(const void* blob, size_t blob_size, uint64_t* out_workspace_size);

// T-2113 (P2, design Sec4.2/Sec22, routed `Claude/Poirot/50f3d5d-t2113-1p0-gpu-core-build-review.md`
// Sec15, D-SLM3415): the identity twin of `PeekGpuSeqBlobWorkspaceSize` above -- reads only the v3
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

// Added Sec21 Fold, D-SLM3082 (Dan's amendment 5): Resource Binding Tier 1/2
// feature-query fallback. `ArmResourceBindingTierMock(tier)` (1 or 2) makes the next
// `MapModelGpuResidencyTierCheck()` call observe a mocked
// `D3D12_FEATURE_DATA_D3D12_OPTIONS::ResourceBindingTier` result below the Tier-3
// this design's binding architecture requires (D-SLM3000) -- every device this
// project's probes have run on reports Tier 3, so this is the only way to exercise
// the sub-Tier-3 path. Returns true iff the GPU path proceeds (Tier 3 or unmocked);
// false iff it returns the defined "GPU path unsupported on this device" status
// having built no descriptor table, no UAV table, and no device allocation.
void ArmResourceBindingTierMock(int tier);
void ClearResourceBindingTierMock();
bool MapModelGpuResidencyTierCheck();

// --- B12 (Sec10 dim 5, Sec11 B12, redesigned Sec21 Fold D-SLM3081, Dan's amendment
// 4): deterministic allocation-failure injection, not real VRAM exhaustion (which is
// non-deterministic and therefore not a valid gate, per D-SLM3067 item 4's own
// reasoning). This retires the original real-VRAM-exhaustion construction (a single
// `GpuResidencyMapReturnsDefinedFailureOnOverBudgetRequest` symbol) as this step's
// correctness gate -- superseded, not merely extended, since a gate that sometimes
// passes for the wrong reason is not a gate; a real memory-pressure configuration
// remains a permitted OPTIONAL SMOKE (D-SLM3067's own allowance), not authored as a
// symbol here since this suite does not run it. `sslm_model_map`'s GPU-residency
// setup makes a fixed, enumerable sequence of `kGpuResidencyAllocationCallCount`
// ("N") mock allocation calls. ---
// T-2052 (M3, Claude/Poirot/36b9327-gpu-serial-port-reconfirmation-review.md):
// this comment previously stated N's own provenance as "Sec5.1's read-resource
// and write-resource tables" -- T-2049's own N4 remedy corrected that claim at
// its definition site and left this declaring header still stating it, a
// claim surviving in exactly one file for the third consecutive round (M3's
// own finding). N's SINGLE definition, with its own current and honestly-
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
// cleaned-up map (transactional-cleanup verification, D-SLM3081 item (2)).
uint32_t LiveAllocationCount();
// How many of the N allocation calls were actually attempted on the most recent
// `MapModelGpuResidencyWithInjection` call -- must read 0 when the low-budget
// preflight fires (it returns before attempting any of the N calls).
uint32_t AllocationCallsAttempted();
// Returns true iff the GPU-residency map succeeds cleanly (Ok); false iff it returns
// the defined failure status this step names (unchanged by the redesign).
bool MapModelGpuResidencyWithInjection(uint64_t required_bytes);

}  // namespace superslm_gpu

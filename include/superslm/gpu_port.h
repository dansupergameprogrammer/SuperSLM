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

#include <cstddef>
#include <cstdint>

#include "superslm/checked_chain_funnel.h"
#include "superslm/forward_sites.h"

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
superslm::SslmForwardStatus RunLayerLoopGpu(superslm::SequenceLayerState& seq,
                                             const superslm::LayerWeights* layers,
                                             uint32_t num_hidden_layers, uint32_t layer_budget,
                                             size_t hidden_size, size_t head_dim,
                                             size_t num_key_value_heads, size_t intermediate_size,
                                             int64_t context_cap,
                                             const superslm::SslmTensorManifest& rope_tables,
                                             uint8_t* workspace, size_t workspace_size);

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
// but one had gone stale across two same-day commits -- the CLAIM below is
// unchanged from what T-2069 wrote, only the `superslm_gpu.cpp:<line>`
// pointers are corrected to resolve against current HEAD): "twelve" above is
// also wrong -- the correction above counted the function's rejecting
// RETURNS (its guard ladder), not its rejecting return PATHS, and never
// re-derived the number from the function itself. Enumerated fresh, at
// source, every rejecting `return superslm::SslmForwardStatus::` (or
// return-via-ternary) `RunLayerLoopGpu` has: eleven in the nine-guard
// ladder (`superslm_gpu.cpp:640, 641, 650, 654, 669, 673, 674, 684, 687,
// 693, 695` -- eleven RETURN STATEMENTS realizing nine distinct guards, two
// of them, `InvalidContextCap`/`WorkspaceTooSmall`, each with two return
// sites); two device-capability rejections below the ladder (`:702`
// `!dev.available`, `:714` the sub-Tier-3 `MapModelGpuResidencyTierCheck`
// check, both `KvPrecisionUnsupported`); and the recording-window catch
// itself (`:1368`, one return statement, a ternary choosing between
// `GpuDeviceRemoved`/`GpuAllocationFailed`). **Fourteen**, not twelve --
// CORRECTED AGAIN below (T-2075, S2): fourteen is also short by one.
// `g_last_weight_upload_was_skipped`'s own FOUR write sites (corrected from
// "three" in the same sentence that listed four citations, T-2075, M2 --
// `superslm_gpu.cpp:298` static init, `:600` function entry, `:889` the
// residency decision, `:1358` the catch) still cover all fourteen of THIS
// paragraph's own paths correctly -- `:889` is the only conditional write
// and it sits below both `:702` and `:714`, so those two paths read the
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
// statement, `return DecodeStickyTag(sticky_tag);` (`superslm_gpu.cpp:1419`),
// is neither a ladder return nor the catch's ternary -- it is a FUNCTION
// CALL whose result is returned directly, and `DecodeStickyTag` (`:521-539`)
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
// decision runs (`:889` above) -- the nine-guard ladder, the two
// device-capability rejections, and the recording-window catch, fourteen
// paths in all, none of which ever reached a residency decision to report.
// It reads exactly `weights_resident` (`true` on a cache hit, `false` on a
// miss) on every path that returns AFTER the decision -- the success path
// and the sticky-tag-decoded path alike, fifteen paths' own destination in
// total across the function, whether the decoded status is `Ok` or one of
// `DecodeStickyTag`'s thirteen rejecting statuses. A caller that wants "did
// THIS call's upload run" reads this accessor for exactly that, on every
// path, correctly; a caller that reads it as "did this call SUCCEED" is
// reading a different question than the one it answers, on the sticky-tag
// paths specifically -- named here so a future reader does not have to
// re-derive it from the code a fourth time.
bool LastWeightUploadWasSkipped();

// T-2070 (D-SLM3215, S4, Claude/Poirot/b543abe-gpu-serial-port-ship-reverdict-review.md):
// T-2063's own always-declared LINK-RED form of this instrument (`ArmWeightAllocationFailure
// Injection`/`ClearWeightAllocationInjection`, undefined) is UNGATED, so merging it broke the
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
// `SUPERSLM_O11_WEIGHT_ALLOC_INJECTION` (e.g. via `build.bat`'s own `/D` list, beside
// `SUPERSLM_ENABLE_BAD_ALLOC_INJECTION`) in the SAME round -- defining the macro without the
// real bodies reproduces this ticket's own LINK-RED proof (`Claude/Curie/t2019-gpu-serial-
// red-suite-2026-08-13.md` S16.2/S17), on purpose, as the gate's own self-check.
//
// Mirrors B12's own arm/inject naming (ArmAllocationFailureInjection/ClearAllocationInjection):
// arms injection so the NEXT armed allocation call in RunLayerLoopGpu fails with a synthetic
// allocation error; the call after clears it. T-2076 note: the definitions built by T-2071 target
// `work_scratch_uav`'s own allocation, not the weight DEFAULT-heap buffer this comment originally
// specified -- T-2075's own S1 fix moved the arm site there so an armed call reaches the throw on
// a cache HIT too, which the weight buffer's own allocation (gated behind `!weights_resident`)
// cannot. The function names below are unchanged; only the site they arm moved. `TestT2063_S1Mb_
// WorkScratchUavAllocationThrow_ReturnsGpuAllocationFailed_SkippedFalse` (tests/test_main.cpp) is
// gated identically -- it does not compile into the default build at all, so it cannot be the
// thing that fails the default build's own link.
#ifdef SUPERSLM_O11_WEIGHT_ALLOC_INJECTION
void ArmWeightAllocationFailureInjection();
void ClearWeightAllocationInjection();
#endif  // SUPERSLM_O11_WEIGHT_ALLOC_INJECTION

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
// answer this question, only planned). `complete_layers = min(dispatch_budget / 17,
// num_hidden_layers - current_layer_position)`, floor division, 17 = 16 sites + 1
// commit dispatch (Sec5.4/Sec5.6). Returns SslmGpuStatus::DispatchBudgetTooSmall,
// `*out_layers_to_issue = 0`, for any `dispatch_budget` in [0, 16] -- floor division by
// 17 is uniformly zero there. Never records a partial layer. ---
enum class SslmGpuStatus { Ok, DispatchBudgetTooSmall, Busy };

SslmGpuStatus PlanDispatchBudgetGpu(uint32_t dispatch_budget, uint32_t num_hidden_layers,
                                     uint32_t current_layer_position,
                                     uint32_t* out_layers_to_issue);

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
bool SaveGpuSequenceState(const superslm::SequenceLayerState& seq, const uint8_t* workspace,
                           size_t workspace_size, void* out_blob, size_t* out_blob_size);
bool RestoreGpuSequenceState(const void* blob, size_t blob_size, superslm::SequenceLayerState* out_seq,
                              uint8_t* out_workspace, size_t workspace_size);

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

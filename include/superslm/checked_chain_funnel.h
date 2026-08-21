// SuperSLM checked chain funnel.
//
// The single checked entry surface into the per-token dynamic-scale requant chain
// (intmath.h) at the composition's wide (int64) row width: RequantChainChecked and
// NarrowRowChecked are the only two functions in the forward pass permitted to call the
// underlying max-abs-reduction, scale-normalization, reciprocal, and requant primitives
// directly — every other site routes through one of these two, enforced structurally by a CI
// source check that bans those primitives from every other forward translation unit.
//
// Also declares the funnel's status vocabulary (SslmForwardStatus) and the domain predicates
// for runtime-derived operands that have no domain check of their own: the SwiGLU site's
// runtime gate scale, the bias-reconciliation site's operand pair, and the softmax row's
// numerator/sum width. CheckSoftmaxRowWidthDomain's own bound check runs at 128-bit width to
// avoid overflow when re-deriving it. This file also declares CheckPositionOverCap, a
// standalone predicate for the position-cap gate that forward_sites.h's RopeApplySite calls as
// its first act, before any rotary-table read.
//
// The RMSNorm site, the weight-scale fold-apply, the bias-reconciliation compute, and the
// embed entry are separate site-composition functions declared in
// include/superslm/forward_sites.h instead of here.
//
// The declarations below are the approved API surface; every body is real in
// src/forward/checked_chain_funnel.cpp.
#ifndef SUPERSLM_CHECKED_CHAIN_FUNNEL_H
#define SUPERSLM_CHECKED_CHAIN_FUNNEL_H

#include <cstddef>
#include <cstdint>
#include <span>
#include <string_view>

#include "superslm/intmath.h"  // kProbFracBits (kSoftmaxRowMaxSafeExponent's derivation)
#include "superslm/trace_hook.h"

namespace superslm {

// A carried (mantissa, exponent) dynamic-scale pair: value == m * 2^e. This is the
// shape every KVC1 keyed-constant entry and every runtime-derived per-token/per-query
// scale in the S3a composition carries (model.h:46 — "every value, including the
// exponent e, is a little-endian int64", which is why both fields are int64 here
// rather than a narrower type). Canonical form (artifact-carried) constrains
// m in [2^30, 2^31); a runtime-derived carried_scale_product (C26) is not required to
// hold that tighter range mid-composition.
//
// **It IS required to fit int32_t's own representable range**
// ([INT32_MIN, INT32_MAX]) (ac34677 S5): `CombineCarriedScale`
// (checked_chain_funnel.cpp) folds two operands through a single int32 high-mul, and
// an operand outside int32's range is silently truncated by the unconditional
// narrowing cast — executed at m = 2^31 (one past INT32_MAX), which produced a
// negative mantissa with no diagnostic. `RequantChainChecked` checks this
// precondition on `incoming` and `site_constant` before folding either into the
// running product, AND on the running product itself after every fold step
// (380b75f review N1: `running` is the fold's own left operand on every combine
// after the first, and the entry check alone never sees it), returning
// `CarriedScaleMantissaOutOfDomain` rather than truncating silently.
struct CarriedScale {
	int64_t m = 0;
	int64_t e = 0;
};

// Every way the S3a forward composition can reject, plus Ok. This is the funnel's
// (§7) own status vocabulary — distinct from SslmStatus (artifact-container
// rejection, artifact.h) and SslmModelStatus (tensor-manifest / KVC1 parse rejection,
// model.h), neither of which this design touches. §13 dim 5 names most of these
// members as owed by S3.1 or the sub-slots immediately after it; several are declared
// here for completeness ahead of the sub-slot that first produces them, per that
// section's own note (§4.6).
enum class SslmForwardStatus {
	Ok = 0,
	ChainInputOutOfDomain,                   // C29 (§7.2 step 3): D' > 2^31 in RequantChainChecked
	LogitNarrowingOverflow,                  // C35 (§5.5): NarrowRowChecked's asymmetric int32 domain check
	IExpConstantsOutOfDomain,                // C30 (§7.2): IExpConstantsInDomain rejects the derived i-exp constants
	CarriedScaleMantissaOutOfDomain,         // ac34677 S5 / 380b75f N1: an incoming/site/running CarriedScale.m does not fit int32_t
	SiluCompositionScaleOutOfDomain,         // C34 (§5.4): CheckSiluCompositionScaleDomain rejects the derived (m,e)
	RoundingDivideByPotExponentOutOfDomain,  // C28 (§7.2, §4.4) — S3.2's own build
	SoftmaxRowWidthOutOfDomain,              // C32 (§7.2 second limb) — S3.3's own build, §11 S3.3 §6.2
	TokenIdOutOfRange,                       // owed by S3.6 (§9.1) — declared here for completeness
	PositionOverCap,                         // CheckPositionOverCap (this file) declares the predicate, S3.3;
	                                          // RopeApplySite (forward_sites.h) wires it in as its first act --
	                                          // S3.3's own job, not S3.6's
	WorkspaceTooSmall,                       // owed by a later sub-slot
	KvCapacityExhausted,                     // owed by S3.7 (§9.4) — defined and resumable
	KvPrecisionUnsupported,                  // owed by S3.7 (§14.4)
	InvalidLayerBudget,                      // owed by S3.5 (§9.3, §11 S3.5 "C26, §9.3" --
	                                          // this comment previously read "owed by S3.6";
	                                          // §9.3's own two budget-axis contracts are decided
	                                          // inside S3.5's own sub-slot header ("### S3.5 --
	                                          // Residual reconciliation and the layer loop (C26,
	                                          // §9.3)"), and §11 S3.5's own gate line ("budget
	                                          // invariance green at every enumerated budget")
	                                          // names this enumerator directly, while §11 S3.6's
	                                          // sub-slot text (C16, §9.1) never mentions the
	                                          // budget axis at all -- corrected by the S3.5
	                                          // test-design pass
	// --- Poirot fa3189a-s3.3-rope-site-and-c32-softmax-review-2026-07-28.md ---
	RopeTableTensorMissing,                  // Critical 1: RopeApplySite's ROP1 manifest carries no
	                                          // "cos" or no "sin" tensor -- SslmTensorManifest::Tensor
	                                          // returns nullptr for an absent name (model.h), and
	                                          // nothing at load time requires either name to be present.
	RopeTableExtentExceeded,                 // Critical 2: `position`/`head_dim` address a row or
	                                          // column past the "cos"/"sin" tensors' own validated
	                                          // extent -- the caller's `context_cap` is a fact about
	                                          // CFG1, not about the ROP1 tensors' real shape, and
	                                          // nothing joins the two at load time.
	// --- Poirot e4b398c-s3.4-s3.5-mlp-act-and-layer-loop-review-2026-07-29.md ---
	InvalidContextCap,                       // Critical 2/3: RunLayerLoop's `context_cap <= 0` --
	                                          // zero or negative -- both make the required K/V
	                                          // workspace size computation unsound (zero clears the
	                                          // size guard entirely; negative wraps the same size_t
	                                          // product mod 2^64), checked before either size is
	                                          // formed.
	HeadDimGeometryMismatch,                  // Significant 1: `hidden_size` is not an exact multiple
	                                          // of `head_dim` (or `head_dim == 0`) -- a CFG1 geometry
	                                          // fact, never a fact about the caller-supplied workspace.
	                                          // `WorkspaceTooSmall` used to be returned here, which
	                                          // sends a host that enlarges its buffer into an infinite
	                                          // retry against a size no buffer satisfies.
	KvHeadGeometryMismatch,                   // num_key_value_heads == 0, or num_key_value_heads >
	                                          // num_heads, or num_heads % num_key_value_heads != 0 --
	                                          // a CFG1 geometry fact, mirroring HeadDimGeometryMismatch's
	                                          // own class; checked immediately after it, on the query
	                                          // head count HeadDimGeometryMismatch's own check has just
	                                          // proven in-domain (S3.8a).
	SequenceAlreadyComplete,                  // Significant 6: `seq.layer_index >= num_hidden_layers`
	                                          // at entry -- there are no more layers to advance
	                                          // through for this token, whether because the sequence
	                                          // already ran to completion or because
	                                          // `num_hidden_layers == 0`. Checked before the loop body
	                                          // runs, on the same "reject-over-silently-degrade" law
	                                          // §9.3 already applies to `layer_budget == 0`: a step
	                                          // that consumes a call and advances nothing must not
	                                          // return `Ok`, whichever of the two reasons produced it.
	SoftmaxKernelRefusedAfterGateAccepted,    // Minor A: `SoftmaxRowQ15` returned false having already
	                                          // passed `CheckSoftmaxRowWidthDomain`'s own gate -- a
	                                          // distinct outcome from the gate's own rejection, which
	                                          // this loop used to report instead, sending a host
	                                          // debugging `SoftmaxRowWidthOutOfDomain` to inspect a
	                                          // width that is already in domain.
	// --- (SuperSLM_S3a_WalkingSkeleton_Plan.md Sec7.2b, Sec14.14) ---
	ResidualReconciliationMagnitudeOutOfDomain,  // C26's residual-reconciliation site
	                                          // (ResidualReconcileSite, forward_sites.h/.cpp): the
	                                          // true 128-bit magnitude `LandingRescale` computes for
	                                          // at least one element of the row does not fit int64
	                                          // (`out_magnitude_exceeded_int64`, LandingRescale's own
	                                          // second, distinct out-parameter) -- a runtime check on
	                                          // an already-computed loss signal, not a static margin
	                                          // (Sec7.2b's derivation; the exponent `CarriedScale.e`
	                                          // carries no domain check anywhere in this tree, so no
	                                          // closed-form threshold independent of layer count
	                                          // exists). Distinct from the existing
	                                          // `out_saturation_count`/[-127,127] clamp signal, which
	                                          // is C27's own and stays coupled to that caller's clamp
	                                          // range.
	// --- Poirot cd2e75a-t1585-t1587-t1588-confirmation-2026-07-31.md ---
	InvalidHiddenCodes,                       // RunLayerLoop's `seq.hidden_codes == nullptr`
	                                          // -- a caller-restored `SequenceLayerState` carries no
	                                          // provenance guarantee (§13 dim 9's own addressable-as-
	                                          // a-unit save/restore), the same argument that motivated
	                                          // the `context_length` guards below, applied to the one
	                                          // other field this loop dereferences unconditionally.
	                                          // `hidden_codes` is a caller-owned pointer with no length
	                                          // carried in the struct, so a too-short buffer is not a
	                                          // domain this check (or any check) can detect; `nullptr`
	                                          // is the one value that is both detectable and the
	                                          // struct's own default member initializer, and was
	                                          // previously dereferenced unconditionally at this loop's
	                                          // first read of it.
	// --- design superslm-t1655-t1656-iexp-and-bias-design ---
	IExpScaleDerivationOutOfDomain,           // §4.6: C30's derivation site
	                                          // (RunLayerLoop's per-kv_head IExpScaleConstants call)
	                                          // found NO triple could be formed at all --
	                                          // IExpScaleConstants returned other than kOk. Distinct
	                                          // from IExpConstantsOutOfDomain, which answers "a
	                                          // FORMED triple fails IExpConstantsInDomain": this
	                                          // status answers "no triple could be formed", C30's own
	                                          // upstream construction domain
	                                          // (kBadCoefficient/kNegativeShift/kNotRepresentable).
	BiasReconcileProductOutOfDomain,          // the
	                                          // magnitude-domain guard at the C28 bias-reconciliation
	                                          // call sites (ProjectAndFunnel's q_proj insertion and
	                                          // the k/v landing path's identical insertion) found
	                                          // either BiasReconcile's own rounded, divided RESULT
	                                          // (intmath.h's BiasReconcileWide) does not fit
	                                          // int64_t, or -- a separate finding -- that
	                                          // result, though itself representable, does not fit
	                                          // int64_t once added to the call site's own running
	                                          // acc[i], the composed quantity the site actually
	                                          // forms (CheckBiasAccumulateMagnitudeDomain below).
	                                          // Both conditions are checked before either loop body
	                                          // in ApplyBiasReconcileRow applies anything, so a
	                                          // rejection for either reason leaves acc untouched.
	// --- (design Sec31.2.2/Sec12 "Option G's own coverage") ---
	OptionGWideRopeMagnitudeOutOfDomain,      // design Sec31.2's own wide-RoPE-pair
	                                          // primitive (RopeApplyPairWide, forward_sites.h,
	                                          // this fold): refuse, not wrap, whenever either
	                                          // rotated component's ROUNDED value does not fit
	                                          // int64_t (checked
	                                          // on the value AFTER C3 rounding, never an unrounded
	                                          // "true" value the primitive never materializes).
	                                          // Mirrors the spike's own gate G2
	                                          // (option_g_spike.h), now production-named.
	OptionGFusedLandingExponentOutOfDomain,   // design Sec31.2.2 (the round-3 repair): `LandingRescale`'s
	                                          // own `out_magnitude_exceeded_int64` output, checked
	                                          // UNCONDITIONALLY at both fused K-landing call sites,
	                                          // on EVERY element, with no skip condition on any
	                                          // static exponent. Distinct from
	                                          // OptionGWideRopeMagnitudeOutOfDomain above: that
	                                          // status answers "did the ROTATION overflow int64";
	                                          // this one answers "did the subsequent LANDING (the
	                                          // already-shipped LandingRescale, called on the
	                                          // rotation's own in-domain output) lose magnitude" --
	                                          // two different arithmetic stages, two different
	                                          // guards, per the design's own repaired text.
	// --- (design §22): the GPU-serial port's recording-window catch
	// (superslm_gpu.cpp, RunLayerLoopGpu) used to reuse KvPrecisionUnsupported
	// for "an allocation inside the command-list recording window threw" --
	// already carrying two other meanings on that leg (no device at all;
	// sub-Tier-3 hardware), both permanent-hardware conditions a caller
	// should stop retrying against. These two are host-observed AT THE
	// CATCH SITE via ID3D12Device::GetDeviceRemovedReason(), never derived
	// from SSLM_GPU_HR's own thrown std::runtime_error (which carries no
	// HRESULT payload) and never encoded into the GPU-side sticky-tag space
	// DecodeStickyTag decodes (superslm_gpu.cpp) -- appending here is
	// ABI-safe by every constraint this tree enforces or has planned
	// (§22.2: no fixed underlying values beyond Ok=0, never
	// serialized, no shipped sslm_status C-ABI type exists yet to freeze). ---
	GpuAllocationFailed,                      // a device allocation, or any
	                                          // other command-recording-window D3D12 operation,
	                                          // failed while the device is CONFIRMED STILL PRESENT
	                                          // (GetDeviceRemovedReason() == S_OK). Transient and
	                                          // size-dependent -- the correct host response is to
	                                          // retry at a smaller configuration, never to fall
	                                          // back off this GPU permanently (the same class of
	                                          // host-facing confusion this enum's own
	                                          // HeadDimGeometryMismatch/WorkspaceTooSmall history
	                                          // documents as a defect when the wrong status sends
	                                          // the host down the wrong recovery path).
	GpuDeviceRemoved,                         // the GPU device itself was
	                                          // removed, reset, or hung
	                                          // (GetDeviceRemovedReason() != S_OK) during a
	                                          // recording-window operation. The device object is no
	                                          // longer usable for any further call -- the host must
	                                          // recreate it, not retry against the same handle.
	                                          // Distinct from GpuAllocationFailed (device alive,
	                                          // this one call failed) and from
	                                          // KvPrecisionUnsupported's "no device"/"tier too low"
	                                          // meanings (this device WAS usable a moment ago and
	                                          // may be again after recreation -- neither a
	                                          // permanent-hardware nor a size-dependent condition).
	                                          // Named residual: GetDeviceRemovedReason()
	                                          // answers "is the device gone right now," not "did
	                                          // removal cause THIS call's throw" -- an allocation
	                                          // failure racing an unrelated, asynchronous device
	                                          // removal reads as GpuDeviceRemoved, the conservative
	                                          // direction (an unneeded device-recreation costs less
	                                          // than an unneeded retry against a device that is
	                                          // actually gone).
	GpuGemmGroupArithmeticInvalid,             // the
	                                          // multi-group GEMM dispatch's own standing guard
	                                          // (`ComputeGpuGemmSiteGroupPlan`, superslm_gpu.cpp) found
	                                          // a group count that does not cover its own output
	                                          // dimension -- a PERMANENT coding-arithmetic defect,
	                                          // never a transient or size-dependent one. Distinct from
	                                          // `GpuAllocationFailed` on purpose: that status's own
	                                          // documented recovery ("retry smaller") is actively wrong
	                                          // advice for this condition, which no retry at any size
	                                          // fixes -- the confirmation pass found the original guard
	                                          // routed through the SAME generic catch as
	                                          // `GpuAllocationFailed`, discarding the diagnostic string
	                                          // and reporting a permanent bug as a recoverable one.
	InvalidDecodeParams,                     // T-2199 Phase D review fix S3
	                                          // (Claude/Poirot/7a3b10a-t2199-phaseD-review.md):
	                                          // RunGreedyOrDampedGreedyDecodeLoop's own
	                                          // ValidateDampedGreedyParams failure used to return
	                                          // TokenIdOutOfRange -- a real status, but the wrong
	                                          // one (no token id was ever examined), and
	                                          // disagreeing with sslm_decode_stepImpl's own
	                                          // identical-invalid-set rejection (SSLM_INVALID_
	                                          // ARGUMENT directly, plan Sec9 dim2's "validation-
	                                          // symmetry" cell). MapForwardStatus (sslm_abi.cpp)
	                                          // maps this one member to SSLM_INVALID_ARGUMENT,
	                                          // giving the two entry points the same outcome for
	                                          // the same bad input.
};

// Human-readable name, for diagnostics and test messages (mirrors SslmStatusName,
// artifact.h, and SslmModelStatusName's convention).
const char* SslmForwardStatusName(SslmForwardStatus s) noexcept;

// A checked chain call's outcome. Only the status rides in the return value; the
// codes and the carried output scale are written through `out_codes`/`out_scale`
// (comment retained from §7.2's own pseudocode: "codes, carried (m,e)" travel out of
// band, not in this struct).
struct ChainResult {
	SslmForwardStatus status = SslmForwardStatus::Ok;
};

// The funnel's first entry point (§7.2): the single checked path into the C19-C22
// per-token dynamic-scale chain at int64 (wide) row width. Performs, in this order
// and no other:
//   0. Every CarriedScale in `incoming`, then `site_constant`, checked against
//      CombineCarriedScale's own precondition (its mantissa fits int32_t's range) —
//      else return {CarriedScaleMantissaOutOfDomain} before step 1 runs (ac34677 S5)
//   1. MaxAbsReduceWide(wide_row, n) -> D
//   2. D' = max(D, 1)                                                    (C20's guard)
//   3. C29's domain check: D' <= 2^31, else return {ChainInputOutOfDomain} — this is
//      C21's own domain check for NormalizeScale, entitling step 4 and nothing else
//   4. NormalizeScale(D') -> DynamicScaleReciprocal — both already int64-domain
//   5. carried_scale_product, in C26's pinned LEFT-ASSOCIATED order — the fold's own
//      running product is checked against CombineCarriedScale's precondition after
//      every combine (380b75f review N1), else return {CarriedScaleMantissaOutOfDomain}.
//      Computed here, ahead of step 6's per-element write, because the two steps do
//      not depend on each other's output and a fold rejection must leave out_codes
//      untouched, which is only true if the fold runs first
//   6. RequantTokenCodeWide(x_i, r, s) per element, directly on the int64 row — the
//      row is NEVER narrowed to int32 (NarrowAccumulatorToI32 does not
//      appear in this list)
// `wide_row` has `n` elements. `incoming` is C26's left-associated product inputs so
// far; `site_constant` is the artifact's KVC1 entry for this site. On Ok, `out_codes`
// (n elements) and `*out_scale` are written; on any rejection, neither is touched.
//
// `site` and `token_index` (§11 S3.1a; trace_hook.h) are the SslmChainTraceRecord
// fields this call cannot derive from its own arguments -- the caller names its
// site and the token it is processing. They default to an empty site and index 0
// so every existing call compiles unchanged; a caller that wants trace coverage
// passes its real site name and token index.
//
// `trace_hook_state` is the caller's own model
// handle's trace-hook state (SslmModelView::trace_hook, model.h) -- never a
// process-wide static. It defaults to nullptr so every existing call compiles
// unchanged and gets no tracing; a caller that wants trace coverage passes
// `&model_view.trace_hook`. Whenever `trace_hook_state` is non-null AND it
// carries an installed hook (SslmTraceHookInstalled), and only then,
// RequantChainChecked builds one SslmChainTraceRecord from the values it has
// already computed -- `wide_row`, `n`, the D'/Dn/s/R intermediates,
// `out_codes`, and the folded `*out_scale` -- and emits it through
// SslmEmitChainTrace on that same state, strictly after every write above.
// This block reads outputs already finalized above; it writes none of them, and
// it does not run at all when no hook is installed. That ordering is what makes
// installing the hook produce the identical ChainResult, out_codes, and
// *out_scale as not installing it (§10.3's instrumentation axis). Because the
// state lives on the model handle the caller passes in, two callers driving two
// different handles never observe or disturb each other's hook.
// The ONE door a site may use to obtain C19's reciprocal of a carried mantissa.
// §7.3's invariant is that no site re-derives the chain's
// own arithmetic, enforced by banning eight leaf names outside this file's own
// translation unit. That invariant was written when every site-level reciprocal
// was either artifact-carried (C27's landing takes `r_t` as a parameter) or
// internal to the funnel; C26's residual reconciliation (§6.2 step 8) is the
// first site that must derive one at runtime, from the STREAM's own mantissa,
// and so could satisfy neither form.
//
// This forwards to DynamicScaleReciprocal and does nothing else. It opens
// exactly ONE of the eight leaves, through a named and auditable door, rather
// than exempting a whole file: the other seven -- the max-abs reductions, the
// row bounds, NormalizeScale, both requant primitives and the int32 narrowing
// -- remain reachable only from inside this translation unit, which is the
// property that stops a site from reassembling the chain. Neither relocating
// the site into this TU nor allowlisting its call site preserves that property:
// the first converts a function-level rule into a file-level exemption, and the
// second removes the wall at one point permanently.
//
// **`DynamicScaleReciprocal`'s `(2^31, 2^32]` ceiling holds only for a canonical
// `Dn ∈ [2^30, 2^31)` -- the range `NormalizeScale` produces.**
// This door forwards to it on whatever mantissa the caller supplies, which is not
// required to be canonical (`CombineCarriedScale`'s own renormalization does not
// guarantee it). A caller reasoning about the reciprocal's maximum on a
// mid-composition operand must not assume `2^32`; treat the return value as an
// arbitrary `int64_t` and use a widened, checked composition downstream
// (`BiasReconcileWide`, `intmath.h`, is one such consumer) rather than a bound on
// this function's own output. **operand: the return value of this call --
// canonical: no, unguarded by this door itself; guarded only by whatever the
// caller does with it downstream.**
int64_t CarriedScaleReciprocal(int64_t m);

// §4.3: the second door this design opens, onto C26's own carried-scale
// combine step (`carried_scale_product` of exactly two factors). `CombineCarriedScale`
// itself is unchanged (its body stays in checked_chain_funnel.cpp, only its linkage
// changes from anonymous-namespace-private to this header's declared surface) -- this is
// the smallest sound way to make it reachable from forward_sites.cpp without a second
// derivation of the same domain (the drift class F10 already recorded once). Precondition
// (unchecked by this function itself; the caller checks it, per the
// CarriedScaleMantissaOutOfDomain convention `RequantChainChecked` already uses): both
// `a.m` and `b.m` fit int32_t's own range.
CarriedScale CombineCarriedScale(CarriedScale a, CarriedScale b);

ChainResult RequantChainChecked(const int64_t* wide_row, size_t n,
                                 std::span<const CarriedScale> incoming,
                                 CarriedScale site_constant, int8_t* out_codes,
                                 CarriedScale* out_scale,
                                 std::string_view site = {}, size_t token_index = 0,
                                 SslmTraceHookState* trace_hook_state = nullptr);

// The funnel's second entry point (§7.2): the one narrowing genuinely owed —
// the head's int32 logits (C16's tie-break, §10.1's digest format). Performs, in this
// order and no other:
//   1. RowBoundsWide(wide_row, n) -> the row's signed max and min. MaxAbsReduceWide is
//      NOT reused here (§5.5) — narrowing depends on the row's signed extremes, not
//      its magnitude, and a magnitude bound cannot express an asymmetric target range
//   2. C35's domain check (§5.5): max <= INT32_MAX and min >= INT32_MIN, else return
//      LogitNarrowingOverflow — a different check from step 3 above, on a different
//      quantity
//   3. NarrowAccumulatorToI32 — genuinely sound: every element has just been proven,
//      individually and by its own sign, to lie in int32's representable range
// On Ok, `out_i32` (n elements) is written; on rejection it is not touched.
//
// No production path in this build emits a trace record from
// NarrowRowChecked, so it carries no `trace_hook_state` parameter (380b75f
// review N4: a parameter a caller can set with no effect and no diagnostic is
// worse than no parameter at all). When S3.3 gives this entry point a trace
// emission, the parameter is added then, threaded through the caller's own
// model handle the same way RequantChainChecked's already is --
// not before, and not as a knob that does nothing in the meantime.
SslmForwardStatus NarrowRowChecked(const int64_t* wide_row, size_t n, int32_t* out_i32);

// `[[nodiscard]]` added here and on
// CheckSiluCompositionScaleDomain, CheckRoundingDivideByPotExponentDomain,
// CheckSoftmaxRowWidthDomain, and CheckPositionOverCap below -- the five `Check*`
// siblings in this file that had not yet been swept when an earlier pass
// fixed only the two bias predicates it was flagged against. `intmath.h`
// states the doctrine this sweep completes: "[[nodiscard]] is load-bearing, not
// decoration". Every `Check*` predicate declared in this file now carries it.
//
// C30's derived-operand predicate (§7.2 second limb). The not-yet-built C30
// derivation site (S3.3) forms (q_ln2, q_b, q_c) from a per-query carried scale via
// C30's own formula and calls this on the result. THIS FUNCTION ENCODES NO THRESHOLD
// OF ITS OWN — it calls the already-shipped IExpConstantsInDomain (intmath.h:427)
// on the four values and maps a false result to IExpConstantsOutOfDomain, nothing
// else.
//
// **This (2026-07-28) settled the apparent disagreement between the design's
// textual domain rule (§7.2's table, "e >= -60, or e = -61 with m >= 1,268,234,713")
// and `IExpConstantsInDomain`, which disagree at 79 of 198 swept points (§5): the shipped
// predicate is correct, and the design's prose over-generalized the underflow tail
// (`e in [-31, +8]`) where the predicate's genuine divide-by-zero rejection reads as
// an extra restriction but is not one, verified against an arbitrary-precision
// decomposition oracle at every swept point. `IExpConstantsInDomain` is the total,
// sole domain authority; this function implements that ruling rather than
// paraphrasing it.**
[[nodiscard]] SslmForwardStatus CheckIExpConstantsDomain(int64_t q, int64_t q_ln2, int64_t q_b,
                                             int64_t q_c);

// C34's derived-operand predicate (§7.2 second limb, §5.4). The SwiGLU activation
// site (S3.4, forward_sites.h's MlpActSite) forms
// the per-token gate scale (m, e) at runtime — never
// artifact-carried, so no load-time gate stands behind it — and calls this before
// `SiluSigmoidQ15` (silu_lut.h:61). Encodes the runtime no-UB domain directly:
// `|m|` must stay within the same symmetric bound `SiluSigmoidQ15`'s
// `term = code * m` needs to stay int64-exact, and `e` must keep both of that
// function's shift placements in range. Its upper branch names `kSiluLutTermLeftShiftOverflowExponent`
// (silu_lut.h) and its lower branch names `kRoundingDivideByPotExponentMaxI64`
// (intmath.h) — the same two constants the loader's own static_asserts
// (model.cpp) use, so neither side can drift from the primitive without failing a
// compile.
//
// **The relation to S-HARDEN-1's load-time descriptor is CONTAINMENT, not
// equality, and is deliberate**: every (m, e) the loader accepts is accepted here, and
// this predicate's domain is strictly wider on the upper branch — `e = 8` is the one
// point of difference (§5.4, executed), accepted here and rejected at load time. The
// mirroring static_assert next to this predicate's definition
// (src/forward/checked_chain_funnel.cpp) proves that ordering at compile time: the
// load-time ceiling never exceeds this predicate's own ceiling.
[[nodiscard]] SslmForwardStatus CheckSiluCompositionScaleDomain(int64_t m, int64_t e);

// C28's derived-operand pair predicate (§7.2 second limb, §4.4; S3.2). The
// C28 bias-reconciliation site (forward_sites.h's BiasReconcile) checks this
// before calling RoundingDivideByPOT(int64_t, int) with the composed exponent
// q_B + 62 + e_a: 0 <= q_B + 62 + e_a <= 63, else
// RoundingDivideByPotExponentOutOfDomain. Names
// kRoundingDivideByPotExponentMinI64 / kRoundingDivideByPotExponentMaxI64
// (intmath.h:59-60) rather than the literals 0 and 63, so neither side can drift
// from RoundingDivideByPOT's own domain without failing a compile.
//
// Re-staged unchanged from its original declaration at commit 32aca0c (removed the
// same day, f98eee9, as belonging to S3.2 rather than S3.1) -- see the file header
// comment above. The real
// 0 <= q_B + 62 + e_a <= 63 comparison now lives in
// RoundingDivideByPotComposedExponentInDomain (src/intmath.cpp); this function
// (src/forward/checked_chain_funnel.cpp, S3.2's green phase) only delegates to it.
[[nodiscard]] SslmForwardStatus CheckRoundingDivideByPotExponentDomain(int64_t q_B, int64_t e_a);

// C28's magnitude domain predicate. Exactly
// `BiasReconcileWide(b, q_b, r_a, e_a, &unused) == true` -- it validates the SAME
// domain `BiasReconcile` itself computes over, never a second derivation of it
// (the `IExpConstantsInDomain` precedent this predicate follows exactly:
// intmath.h, "this predicate is exactly IExpConstruct(...) == kOk"). Reuses the
// existing `BiasReconcileProductOutOfDomain` status -- what changed is
// the CONDITION that produces it, not the status's own meaning: it now answers
// whether the rounded, divided C28 RESULT fits int64_t, not whether the raw
// product does. That is a strictly weaker, strictly more permissive condition
// every input the retired `BiasReconcileProductFitsInt64` guard
// accepted is accepted here too, and some inputs whose raw product overflows
// int64_t but whose rounded result does not -- which the retired guard wrongly
// rejected -- are accepted here as well. Because this predicate is exactly `BiasReconcileWide(...) == true`, and
// that function now checks the composed exponent domain itself before it checks the
// magnitude, this predicate ALSO rejects an out-of-domain exponent -- but
// reports it as `BiasReconcileProductOutOfDomain`, the identical status an
// in-domain-exponent magnitude failure produces. It does not distinguish the two.
// `CheckRoundingDivideByPotExponentDomain` is unchanged and still required first, at
// the call site, for a diagnosis that names the right mechanism -- a caller that
// skips it and reads this predicate's status alone will attribute an
// exponent-domain rejection to the magnitude.
//
// `[[nodiscard]]`, restored. The retired
// `BiasReconcileProductFitsInt64` carried the sole `[[nodiscard]]` in this
// header at the time; this predicate matched its `Check*` siblings -- at the time,
// none of which had it -- rather than the guard it replaced, silently propagating
// the weaker convention onto a guard predicate whose return value being dropped is a
// silent security failure. Matching an unaudited sibling's convention is the
// `StandardsDocument` §7 sibling-pinning trap this repeats: the fix states the
// property directly rather than
// copying a neighbor that was never checked for it.
[[nodiscard]] SslmForwardStatus CheckBiasReconcileMagnitudeDomain(int64_t b, int64_t q_b,
                                                                   int64_t r_a, int64_t e_a);

// C28's accumulate domain predicate. The line
// this guards -- forward_sites.cpp's ApplyBiasReconcileRow, `acc[i] += BiasReconcile(...)`
// -- forms `acc[i] + result`, not `result` alone.
// `CheckBiasReconcileMagnitudeDomain` proves the second term of that sum representable;
// it proves nothing about the sum, which is a different property (a value 1 away from
// int64_t's boundary is representable on its own and overflows against any nonzero
// `acc[i]` of the same sign). This predicate proves the SAME expression the call site
// forms: `acc_i + BiasReconcile(b, q_b, r_a, e_a)` representable in int64_t, over BOTH
// terms of the sum -- the existing per-term magnitude domain (unchanged, computed here
// via the same BiasReconcileWide call `CheckBiasReconcileMagnitudeDomain` uses) AND the
// addition against the caller's own accumulator. Reuses `BiasReconcileProductOutOfDomain`
// for both failure conditions, since the status already means "the C28 site's
// own composed quantity does not fit int64_t" and the accumulate is that quantity's next
// term, not a new one. This predicate calls
// `BiasReconcileWide` too, so it inherits the same conflation `CheckBiasReconcileMagnitudeDomain`
// carries -- an out-of-domain exponent is ALSO rejected here, reported as the same
// `BiasReconcileProductOutOfDomain` an in-domain-exponent magnitude or accumulate
// failure produces. `CheckRoundingDivideByPotExponentDomain` is unchanged and still
// required first, at the call site, exactly as it was for `CheckBiasReconcileMagnitudeDomain`
// -- skipping it and reading this predicate's status alone misattributes the rejection.
// `[[nodiscard]]` for the same reason as that predicate's own.
[[nodiscard]] SslmForwardStatus CheckBiasAccumulateMagnitudeDomain(int64_t acc_i, int64_t b,
                                                                    int64_t q_b, int64_t r_a,
                                                                    int64_t e_a);

// C32's own numerator ceiling (§7.2 second limb; §14.1; §11 S3.3 §6.2,
// §3). This derives a softmax row's largest i-exp
// value in closed form as `M = q_b*q_b + q_c` (the value at `q = 0`, where
// `ShiftByMax` puts the row maximum); this finding shows the shipped
// `IExpConstantsInDomain` does not cover the numerator/sum widths this needs.
//
// **RULED BY DAN (2026-07-28): `2^47` on every path** —
// `(2**62) >> PROB_FRAC_BITS`, the value `Tools/superslm_spike/pipeline.py`'s
// own `_guard_probability_width` already enforced (option C over the closed form's
// own `INT64_MAX >> PROB_FRAC_BITS` = 2^48-1, and over shipping 2^47 without
// touching the Python reference). Both Python paths now refuse against one
// named constant, `pipeline.PROB_WIDTH_CEILING`; this is the C++ side's own
// name for the same quantity. **No `static_assert` can tie the two across the
// C++/Python language boundary** — the tie this tree's `static_assert`
// convention (kRoundingDivideByPotExponentMaxI64, kSiluLutTermLeftShiftOverflow
// Exponent) uses is unavailable here, so the tie is BY NAME AND CITATION only:
// this constant's own name, kept in step with
// `pipeline.PROB_WIDTH_CEILING` by hand. A future edit to either side is not
// caught by this build; it is caught only by re-reading this comment.
// `kProbFracBits` (intmath.h) is PROB_FRAC_BITS itself, so the derivation
// below stays tied to the same shift width C32's own composition uses,
// verified equal to 2^47 at authoring (2^62 >> 15 == 2^47, exactly).
inline constexpr int64_t kSoftmaxRowMaxSafeExponent = (int64_t{1} << 62) >> kProbFracBits;

// C32's own derived-operand predicate (§7.2 second limb; §11 S3.3
// §6.2). The built C32 softmax row kernel (SoftmaxRowQ15, intmath.h) calls
// this before evaluating a row: `M = q_b*q_b + q_c` (the closed form,
// the row's own i-exp value at the shifted-max element) must satisfy
// `q_c >= 0`, `M <= kSoftmaxRowMaxSafeExponent` (the numerator), AND
// `width * M <= INT64_MAX` (the sum). `M` is formed and judged at 128-bit
// width (Poirot 2026-07-28 finding 1) rather than in int64, which is the
// exact re-derivation intmath.h:391-395 names as unsafe.
//
// **`q_c >= 0` (Popper 2026-07-28 Null 2, both bullets), and it is
// necessary but not sufficient.** With `q_c` non-negative, `M = q_b*q_b + q_c`
// can never be driven to zero or negative by a `q_c` that cancels against
// `q_b*q_b` -- the mechanism both witnesses in Null 2 use (`q_b=10, q_c=-100`
// gives `M=0` while a real row element reaches ~9x10^16 past the ceiling) --
// and it makes `m` provably non-negative before the sum check below runs, so
// the sum check's own `INT64_MAX / m` can never see a negative divisor (the
// signed-to-`size_t` cast Null 2's second bullet exploited). It is NOT
// sufficient on its own: `M` bounding every OTHER row element is a claim that
// holds only under the ratio `2*q_b >= q_ln2 - 1` (intmath.h:414), and this
// predicate has no `q_ln2` parameter with which to check that ratio (Popper
// Null 1) -- a `q_c >= 0` row can still be off that ratio. `SoftmaxRowQ15`
// (intmath.cpp) closes that remaining gap: it independently recomputes this
// same `M` from the call's own `q_b`/`q_c` and enforces it as a real
// per-element ceiling against the row's ACTUAL evaluated values, which this
// closed-form predicate cannot observe.
//
// The sum check runs after the `q_c >= 0` and ceiling tests, without
// overflowing itself, on the now-known-non-negative int64 `m`
// (`m == 0 || width <= static_cast<size_t>(INT64_MAX / m)`). Returns
// SoftmaxRowWidthOutOfDomain on any failure, Ok otherwise.
//
// **`width == 0` is rejected outright.** `ShiftByMax` (intmath.h) documents its own `n >= 1`
// precondition ("Undefined on an empty sequence (n >= 1)"), and
// `SoftmaxRowQ15`'s width-gated compute path calls it under that contract --
// so this gate, which exists to certify a width safe for that compute path,
// cannot answer Ok at a width the compute path is not defined for.
// (`SoftmaxRowQ15` also guards `width == 0` inside the kernel itself;
// this gate's rejection is independently required under
// reject-over-degrade and is unchanged by that guard.)
[[nodiscard]] SslmForwardStatus CheckSoftmaxRowWidthDomain(int64_t q_b, int64_t q_c, size_t width);

// C33's own position-cap guard (§11 S3.3's own gate line: "a position ==
// context_cap is rejected before a table read"). `position` is
// a host/runtime-supplied sequence position; `context_cap` is the artifact's
// own config field (model.h). Rejects when `position` is outside
// `[0, context_cap)` -- the cap is an EXCLUSIVE upper bound, matching the
// plan's own "position == context_cap is rejected" wording (equality with
// the cap is already one past the last valid slot). Declared here, in this
// file's own §7.2 second-limb predicate family, so a follow-up Curie pass
// can attach a red cell against a real, callable symbol rather than the bare
// `PositionOverCap` enumerator this tree carried with no function behind it.
// Wiring this into an actual forward call site is S3.3's own job, not S3.6's
// (this exact paragraph was found, a day after the ruling overturned it, still
// routing the wiring to S3.6, and is
// corrected here as part of the site's own build rather than as a separate
// pass): forward_sites.h's RopeApplySite calls this predicate first, before
// any ROP1 table read, exactly the ordering §11 S3.3's own gate line names.
[[nodiscard]] SslmForwardStatus CheckPositionOverCap(int64_t position, int64_t context_cap);

}  // namespace superslm

#endif  // SUPERSLM_CHECKED_CHAIN_FUNNEL_H

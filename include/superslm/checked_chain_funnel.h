// SuperSLM checked chain funnel (SuperSLM_S3a_WalkingSkeleton_Plan.md §7, closing
// T-200 and §17.3 cell 5; S3.1).
//
// The single checked entry surface into the C19-C22 per-token dynamic-scale chain
// (intmath.h) at the composition's wide (int64) row width: RequantChainChecked and
// NarrowRowChecked are the only two functions in the whole S3a forward permitted to
// call MaxAbsReduce/MaxAbsReduceWide/RowBoundsWide/NormalizeScale/
// DynamicScaleReciprocal/RequantTokenCode/RequantTokenCodeWide/NarrowAccumulatorToI32
// directly — every other site routes through one of these two (§7.3's CI source check
// enforces this structurally, banning the leaves from every other forward TU).
//
// Also declares the funnel's status vocabulary (SslmForwardStatus) and the
// derived-operand predicates §7.2's second limb names for the runtime-derived
// operands with no domain predicate of their own: C30's (a call to the already-shipped
// IExpConstantsInDomain, never an encoded threshold), C34's (the SwiGLU site's
// runtime (m, e) gate scale, §5.4), and C28's (the (q_B, e_a) pair check at the
// bias-reconciliation site, §4.4). C28's predicate was declared once during S3.1's
// own header-contract commit (32aca0c) and deliberately removed the same day
// (f98eee9) as belonging to S3.2 instead; it is RE-STAGED here, unchanged from its
// original declaration, as part of S3.2's own header contract (Claude/Curie/
// superslm-s3.2-weightless-and-projection-sites-test-design-2026-07-28.md §9 item 4).
// The RMSNorm site, the WSC1 identity/near-identity fold-apply, the bias-
// reconciliation compute, and the embed entry are S3.2's own site-composition
// functions and are declared in include/superslm/forward_sites.h instead — not here
// — because this file's own translation unit is the sole entry the §7.3 CI source
// check's default glob (tests/ci/check_no_forward_leaf_calls.py) currently expects
// to exist under src/forward/**, pinned by that check's own exact-population
// assertion. Adding a second real file under src/forward/ would fail that assertion
// without a matching update to the check itself, which is outside this campaign's
// writable scope (tests/ is read-only here); forward_sites.cpp is therefore placed
// at src/forward_sites.cpp, a sibling of src/model.cpp, until a future pass updates
// the check's own glob/expectation to cover it.
//
// The declarations below are the approved API surface (§7.2, §5.5, §7.2's second
// limb). Bodies are real constructions in src/forward/checked_chain_funnel.cpp, green
// in the standing suite (Claude/Curie/superslm-s3.1-checked-chain-funnel-test-design-
// 2026-07-28.md §4/§8).
#ifndef SUPERSLM_CHECKED_CHAIN_FUNNEL_H
#define SUPERSLM_CHECKED_CHAIN_FUNNEL_H

#include <cstddef>
#include <cstdint>
#include <span>
#include <string_view>

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
// section's own note (Claude/Curie/superslm-s3.1-checked-chain-funnel-test-design-
// 2026-07-28.md §4.6).
enum class SslmForwardStatus {
	Ok = 0,
	ChainInputOutOfDomain,                   // C29 (§7.2 step 3): D' > 2^31 in RequantChainChecked
	LogitNarrowingOverflow,                  // C35 (§5.5, T-1254): NarrowRowChecked's asymmetric int32 domain check
	IExpConstantsOutOfDomain,                // C30 (§7.2): IExpConstantsInDomain rejects the derived i-exp constants
	CarriedScaleMantissaOutOfDomain,         // ac34677 S5 / 380b75f N1: an incoming/site/running CarriedScale.m does not fit int32_t
	SiluCompositionScaleOutOfDomain,         // C34 (§5.4): CheckSiluCompositionScaleDomain rejects the derived (m,e)
	RoundingDivideByPotExponentOutOfDomain,  // C28 (§7.2, §4.4) — owed by S3.2; declared for completeness
	TokenIdOutOfRange,                       // owed by S3.6 (§9.1) — declared here for completeness
	PositionOverCap,                         // owed by S3.6 (§9.1)
	WorkspaceTooSmall,                       // owed by a later sub-slot
	KvCapacityExhausted,                     // owed by S3.7 (§9.4) — defined and resumable
	KvPrecisionUnsupported,                  // owed by S3.7 (§14.4)
	InvalidLayerBudget,                      // owed by S3.6 (§9.3)
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
//      row is NEVER narrowed to int32 (T-1254's fold; NarrowAccumulatorToI32 does not
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
// `trace_hook_state` (D-SLM353's corrected storage) is the caller's own model
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
ChainResult RequantChainChecked(const int64_t* wide_row, size_t n,
                                 std::span<const CarriedScale> incoming,
                                 CarriedScale site_constant, int8_t* out_codes,
                                 CarriedScale* out_scale,
                                 std::string_view site = {}, size_t token_index = 0,
                                 SslmTraceHookState* trace_hook_state = nullptr);

// The funnel's second entry point (§7.2, T-1254): the one narrowing genuinely owed —
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
// model handle the same way RequantChainChecked's already is (D-SLM353) --
// not before, and not as a knob that does nothing in the meantime.
SslmForwardStatus NarrowRowChecked(const int64_t* wide_row, size_t n, int32_t* out_i32);

// C30's derived-operand predicate (§7.2 second limb). The not-yet-built C30
// derivation site (S3.3) forms (q_ln2, q_b, q_c) from a per-query carried scale via
// C30's own formula and calls this on the result. THIS FUNCTION ENCODES NO THRESHOLD
// OF ITS OWN — it calls the already-shipped IExpConstantsInDomain (intmath.h:427)
// on the four values and maps a false result to IExpConstantsOutOfDomain, nothing
// else.
//
// **D-SLM348 (2026-07-28) settled the apparent disagreement between the design's
// textual domain rule (§7.2's table, "e >= -60, or e = -61 with m >= 1,268,234,713")
// and `IExpConstantsInDomain`, which disagree at 79 of 198 swept points (Claude/Curie/
// superslm-s3.1-checked-chain-funnel-test-design-2026-07-28.md §5): the shipped
// predicate is correct, and the design's prose over-generalized the underflow tail
// (`e in [-31, +8]`) where the predicate's genuine divide-by-zero rejection reads as
// an extra restriction but is not one, verified against an arbitrary-precision
// decomposition oracle at every swept point. `IExpConstantsInDomain` is the total,
// sole domain authority; this function implements that ruling rather than
// paraphrasing it.**
SslmForwardStatus CheckIExpConstantsDomain(int64_t q, int64_t q_ln2, int64_t q_b,
                                             int64_t q_c);

// C34's derived-operand predicate (§7.2 second limb, §5.4). The not-yet-built SwiGLU
// activation site (S3.4) forms the per-token gate scale (m, e) at runtime — never
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
// **The relation to S-HARDEN-1's load-time descriptor (D-SLM142) is CONTAINMENT, not
// equality, and is deliberate**: every (m, e) the loader accepts is accepted here, and
// this predicate's domain is strictly wider on the upper branch — `e = 8` is the one
// point of difference (§5.4, executed), accepted here and rejected at load time. The
// mirroring static_assert next to this predicate's definition
// (src/forward/checked_chain_funnel.cpp) proves that ordering at compile time: the
// load-time ceiling never exceeds this predicate's own ceiling.
SslmForwardStatus CheckSiluCompositionScaleDomain(int64_t m, int64_t e);

// C28's derived-operand pair predicate (§7.2 second limb, §4.4; S3.2). The
// not-yet-built C28 bias-reconciliation site (forward_sites.h's BiasReconcile,
// S3.2) checks this before calling RoundingDivideByPOT(int64_t, int) with the
// composed exponent q_B + 62 + e_a: 0 <= q_B + 62 + e_a <= 63, else
// RoundingDivideByPotExponentOutOfDomain. Names
// kRoundingDivideByPotExponentMinI64 / kRoundingDivideByPotExponentMaxI64
// (intmath.h:59-60) rather than the literals 0 and 63, so neither side can drift
// from RoundingDivideByPOT's own domain without failing a compile.
//
// Re-staged unchanged from its original declaration at commit 32aca0c (removed the
// same day, f98eee9, as belonging to S3.2 rather than S3.1) -- see the file header
// comment above. THIS FUNCTION IS A STUB in src/forward/checked_chain_funnel.cpp:
// it returns a fixed, deliberately wrong status regardless of its arguments; the
// real 0 <= q_B + 62 + e_a <= 63 comparison is written in S3.2's green phase.
SslmForwardStatus CheckRoundingDivideByPotExponentDomain(int64_t q_B, int64_t e_a);

}  // namespace superslm

#endif  // SUPERSLM_CHECKED_CHAIN_FUNNEL_H

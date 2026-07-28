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
// Also declares the funnel's status vocabulary (SslmForwardStatus) and the two
// derived-operand predicates §7.2's second limb names for the runtime-derived
// operands with no domain predicate of their own: C30's (a call to the already-shipped
// IExpConstantsInDomain, never an encoded threshold) and C28's (the (q_B, e_a) pair
// check at the bias-reconciliation site). A third such predicate, C34's, is realized
// as a test-side oracle standing in for a not-yet-built production wrapper
// (Claude/Curie/superslm-s3.1-checked-chain-funnel-test-design-2026-07-28.md §3.3)
// and is not declared here — it is not part of S3.1's own two (§11 S3.1: "the two
// derived-operand predicates, C30's being a call to IExpConstantsInDomain rather than
// a threshold").
//
// THIS FILE IS THE S3.1 CONTRACT STEP (red-first TDD): the declarations below are the
// approved API surface (§7.2, §5.5, §7.2's second limb). Bodies in
// src/forward/checked_chain_funnel.cpp are STUBS -- they compile and link but return
// a fixed, deliberately wrong value when called. Curie's red suite is authored against
// this contract in a follow-up pass (Claude/Curie/superslm-s3.1-checked-chain-funnel-
// test-design-2026-07-28.md §4/§8); the real construction lands at the next build step
// (Brunel green).
#ifndef SUPERSLM_CHECKED_CHAIN_FUNNEL_H
#define SUPERSLM_CHECKED_CHAIN_FUNNEL_H

#include <cstddef>
#include <cstdint>
#include <span>

namespace superslm {

// A carried (mantissa, exponent) dynamic-scale pair: value == m * 2^e. This is the
// shape every KVC1 keyed-constant entry and every runtime-derived per-token/per-query
// scale in the S3a composition carries (model.h:46 — "every value, including the
// exponent e, is a little-endian int64", which is why both fields are int64 here
// rather than a narrower type). Canonical form (artifact-carried) constrains
// m in [2^30, 2^31); a runtime-derived carried_scale_product (C26) is not required to
// be canonical mid-composition.
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
	RoundingDivideByPotExponentOutOfDomain,  // C28 (§7.2, §4.4): the (q_B, e_a) pair check fails
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
//   1. MaxAbsReduceWide(wide_row, n) -> D
//   2. D' = max(D, 1)                                                    (C20's guard)
//   3. C29's domain check: D' <= 2^31, else return {ChainInputOutOfDomain} — this is
//      C21's own domain check for NormalizeScale, entitling step 4 and nothing else
//   4. NormalizeScale(D') -> DynamicScaleReciprocal — both already int64-domain
//   5. RequantTokenCodeWide(x_i, r, s) per element, directly on the int64 row — the
//      row is NEVER narrowed to int32 (T-1254's fold; NarrowAccumulatorToI32 does not
//      appear in this list)
//   6. carried_scale_product, in C26's pinned LEFT-ASSOCIATED order
// `wide_row` has `n` elements. `incoming` is C26's left-associated product inputs so
// far; `site_constant` is the artifact's KVC1 entry for this site. On Ok, `out_codes`
// (n elements) and `*out_scale` are written; on any rejection, neither is touched.
ChainResult RequantChainChecked(const int64_t* wide_row, size_t n,
                                 std::span<const CarriedScale> incoming,
                                 CarriedScale site_constant, int8_t* out_codes,
                                 CarriedScale* out_scale);

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
SslmForwardStatus NarrowRowChecked(const int64_t* wide_row, size_t n, int32_t* out_i32);

// C30's derived-operand predicate (§7.2 second limb). The not-yet-built C30
// derivation site (S3.3) forms (q_ln2, q_b, q_c) from a per-query carried scale via
// C30's own formula and calls this on the result. THIS FUNCTION ENCODES NO THRESHOLD
// OF ITS OWN — it calls the already-shipped IExpConstantsInDomain (intmath.h:362)
// on the four values and maps a false result to IExpConstantsOutOfDomain, nothing
// else. The design's own textual domain rule (§7.2's table, "e >= -60, or e = -61
// with m >= 1,268,234,713") and IExpConstantsInDomain are currently known to disagree
// at 79 of 198 swept points (Claude/Curie/superslm-s3.1-checked-chain-funnel-test-
// design-2026-07-28.md §5, D-SLM318) — which of the two governs S3.3's derivation
// site is a planner ruling pending as of this writing. This declaration and its stub
// body take neither side; they exist so the contract is complete.
SslmForwardStatus CheckIExpConstantsDomain(int64_t q, int64_t q_ln2, int64_t q_b,
                                             int64_t q_c);

// C28's derived-operand pair predicate (§7.2 second limb, §4.4). The not-yet-built C28
// bias-reconciliation site (S3.2) checks this before calling
// RoundingDivideByPOT(int64_t, int) with the composed exponent q_B + 62 + e_a:
// 0 <= q_B + 62 + e_a <= 63, else RoundingDivideByPotExponentOutOfDomain. Names
// kRoundingDivideByPotExponentMinI64 / kRoundingDivideByPotExponentMaxI64
// (intmath.h:59-60) rather than the literals 0 and 63, so neither side can drift from
// RoundingDivideByPOT's own domain without failing a compile.
SslmForwardStatus CheckRoundingDivideByPotExponentDomain(int64_t q_B, int64_t e_a);

}  // namespace superslm

#endif  // SUPERSLM_CHECKED_CHAIN_FUNNEL_H

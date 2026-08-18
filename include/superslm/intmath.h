// SuperSLM integer-arithmetic kernels — Layer-1 requant primitives and the per-token
// dynamic-scale chain.
//
// This is a compiled C++ port of a pinned offline reference implementation. Every function
// here is a bit-for-bit reimplementation of that reference: conformance is output exactness
// against it over the whole domain, never a blessing of any particular decomposition. Standard
// library only; no float on any reproducible path — floats appear only as offline constants
// folded to integers by the converter, never here.
//
// The two families:
//   * Fixed-point requant primitives (`SaturatingRoundingDoublingHighMul`,
//     `RoundingDivideByPOT`) and their composition. Tie directions are pinned and opposite by
//     design: the doubling high-mul rounds toward +infinity, the divide-by-POT rounds away
//     from zero.
//   * The per-token dynamic quantizer: max-abs reduction, scale normalization, a fixed-point
//     reciprocal (corrected Newton), and the per-token requant composition (the "127 scale
//     wrapper").
//
// Cost determinism: every op count here is data-independent. The reciprocal runs exactly 3
// Newton iterations plus exactly 2 fixed-iteration-count correction steps; normalization runs
// one bit-scan plus one shift plus one unconditional select. No early exit, no data-dependent
// trip count.
//
// Wide intermediates: the reciprocal's Newton products and the requant composition's widest
// product (magnitude up to ~2^70) exceed int64. The port carries them in a portable 128-bit
// intermediate (no reliance on a compiler `__int128`, which MSVC lacks). Conformance is by
// output exactness against the pinned formula, not by the decomposition chosen.
#ifndef SUPERSLM_INTMATH_H
#define SUPERSLM_INTMATH_H

#include <cstddef>
#include <cstdint>

namespace superslm {

inline constexpr int32_t kInt32Min = -2147483647 - 1;  // -2^31
inline constexpr int32_t kInt32Max = 2147483647;       //  2^31 - 1

// --- §6.2 requant primitives (C1/C2/C3) --------------------------------------

// C2 — gemmlowp `SaturatingRoundingDoublingHighMul`: (2·a·b) / 2^32, rounded, ties
// toward +infinity (the whole primitive is `(a*b + (1<<30)) >> 31`, both signs).
// The product `a·b` is formed widened to int64; the arithmetic right shift is the
// C++20-guaranteed floor shift. `(INT32_MIN, INT32_MIN)` is the only int32 pair whose
// exact result exceeds INT32_MAX, and it saturates to INT32_MAX.
int32_t SaturatingRoundingDoublingHighMul(int32_t a, int32_t b);

// RoundingDivideByPOT's documented exponent domain, named (S-HARDEN-1, D-SLM142) so a
// caller-side bound — e.g. the load-time schema-value gate's WSC1 `shift` domain and the
// KVC1 `e` no-UB floor's right-branch reach, both in src/model.cpp — can be pinned to this
// primitive's actual domain by `static_assert` rather than by a duplicated numeric literal
// that can drift out of step with it silently.
inline constexpr int kRoundingDivideByPotExponentMinI32 = 0;
inline constexpr int kRoundingDivideByPotExponentMaxI32 = 31;   // int32 overload's exponent ceiling
inline constexpr int kRoundingDivideByPotExponentMinI64 = 0;
inline constexpr int kRoundingDivideByPotExponentMaxI64 = 63;   // int64 overload's exponent ceiling

// C1/C3 — gemmlowp `RoundingDivideByPOT`: x / 2^exponent, ties AWAY FROM ZERO.
// Defined for exponent in [kRoundingDivideByPotExponentMinI32, kRoundingDivideByPotExponentMaxI32];
// exponent 0 is the identity. The `+1` carried on the negative branch of the threshold is what
// turns the floor shift's half-up behaviour into half-away-from-zero.
int32_t RoundingDivideByPOT(int32_t x, int exponent);

// C3 — the int64-domain sibling of RoundingDivideByPOT: x / 2^exponent, ties AWAY FROM
// ZERO, for exponent in [kRoundingDivideByPotExponentMinI64, kRoundingDivideByPotExponentMaxI64].
// Delegates to the SAME width-generic rounding template S2.3
// already instantiates at int64 for RopeApplyPair (S23-1's unification) — one tie rule, no
// duplication. Exposed publicly so S2.4's SiLU-LUT index derivation can round its int64
// sub-node position (SuperSLM_S2.4_SiLU_LUT_Design §5, §9).
int64_t RoundingDivideByPOT(int64_t x, int exponent);

// C1 — the two primitives composed in order, both roundings applied:
// `RoundingDivideByPOT(SaturatingRoundingDoublingHighMul(x, m), shift)`.
int32_t MultiplyByQuantizedMultiplier(int32_t x, int32_t quantized_multiplier, int shift);

// T-2021/T-2029 (design Sec4, D-SLM2915): `x << shift`, widened to int64_t before the shift
// (never itself unsafe -- `x` is 32-bit-bounded and `shift` is checked in [0,31] by the caller's
// own domain contract below, so the widened shift cannot overflow int64_t), then saturated into
// int32_t range. A five-line, TOTAL function -- no precondition to violate, unlike a raw
// `x << shift` on a 32-bit type. This is the ONE new arithmetic primitive
// `ApplyAmplifyingWeightScaleFold` (forward_sites.h) introduces beyond the two gemmlowp
// primitives above, which it reuses unmodified for its own `exponent >= 0` branch (bit-identical
// to `ApplyWeightScaleFold` on that branch, design's own text). `shift <= 0` returns `x`
// unchanged (a caller-side domain error, not a precondition this function enforces -- `shift < 0`
// is meaningless for a left shift and `shift == 0` is the identity either way).
int32_t SaturatingLeftShift32(int32_t x, int shift);

// --- §6.2 rung-1 per-token dynamic-scale chain (C19/C20/C21/C22) --------------

// C21 helper — count leading zeros over 64 bits. Defined for n in [1, 2^64 - 1]; the
// bit-scan is pinned to scalar-reference behaviour (§18 GPU-semantics row). n == 0 is
// out of contract (undefined leading-zero count) and never reached: its only caller,
// NormalizeScale, receives D' >= 1 by C20's guard.
int Clz64(uint64_t n);

// C20 — max-abs reduction: D = max_i |x_i| over the token's int32 activation vector,
// each element WIDENED to int64 before the abs (so x_i = INT32_MIN yields |x_i| = 2^31
// exactly rather than int32 UB), then the all-zero-token guard D' = max(D, 1). Returns
// D' in [1, 2^31]. Order-free (max applies no mid-stream rounding); every element is
// visited unconditionally. `n == 0` yields D' = 1 (the empty reduction), matching the
// reference; the runtime never sees a zero-length vector — hidden dims are architectural
// constants >= 1, rejected at artifact validation.
int64_t MaxAbsReduce(const int32_t* x, size_t n);

// C21 — scale normalization. Result of NormalizeScale: Dn in [2^30, 2^31) and the shift
// exponent s that rides into the C22 composition (1/D' = R · 2^s / 2^62).
struct NormalizedScale {
	int64_t dn;  // normalized denominator in [2^30, 2^31)
	int s;       // shift exponent, s = 30 - (63 - clz64(D'))
};

// C21 — map D' in [1, 2^31] into Dn in [2^30, 2^31) by a pure shift:
// p = 63 - clz64(D'), s = 30 - p, Dn = s >= 0 ? D' << s : D' >> 1. The only negative-s
// case is D' = 2^31 (even), whose single right-shift loses no bits (Dn = 2^30). Exact,
// no rounding. Exactly one bit-scan + one shift + one unconditional select.
NormalizedScale NormalizeScale(int64_t d_prime);

// C19 — the reciprocal's pinned, data-INDEPENDENT op counts (§14): exactly 3 Newton
// iterations then exactly 2 fixed-iteration-count correction steps. Bare fixed-count Newton is
// deterministically wrong on 54–62% of the domain; the corrections close it.
inline constexpr int DYNAMIC_RECIPROCAL_NEWTON_ITERATIONS = 3;
inline constexpr int DYNAMIC_RECIPROCAL_CORRECTION_STEPS = 2;

// C19 — fixed-point reciprocal: R = round_half_up(2^62 / Dn) for Dn in [2^30, 2^31),
// computed DIVISION-FREE via corrected Newton — exactly 3 Newton iterations then exactly
// 2 unconditional, fixed-iteration-count correction steps. Bare fixed-count Newton is deterministically
// wrong on 54–62% of the domain (error biased high → activation clipping); the correction
// closes it to correctly-rounded over the entire normalized domain. Returns R in
// (2^31, 2^32] (R = 2^32 exactly at Dn = 2^30). The seed is not pinned — any construction
// reaching output-exactness is conformant.
int64_t DynamicScaleReciprocal(int64_t dn);

// C22 — per-token requant composition, the "127 scale wrapper" (D-SLM52), R-composed:
// q_i = clamp(round_half_away_from_zero((x_i · 127 · R) / 2^(62 − s)), −127, 127),
// with R from C19 and s from C21. Ties away from zero (C3's direction — the composite IS
// a rounding divide by a power of two). `|x_i · 127 · R|` reaches ~2^70 (a token containing
// INT32_MIN forces R = 2^32 with x_i = −2^31 present), carried in a 128-bit intermediate.
// Returns the int8 code in [−127, 127].
int8_t RequantTokenCode(int32_t x_i, int64_t r, int s);

// C28's widened bias-reconciliation core (T-1657, D-SLM621/641/645/676): forms B*R_a and
// divides by 2^exponent in the SAME portable 128-bit facility RequantTokenCode/
// IExpEvaluate already use internally. TOTAL over the full int64_t domain of b and
// r_a, and over exponent in [kRoundingDivideByPotExponentMinI64,
// kRoundingDivideByPotExponentMaxI64] -- unchanged from RoundingDivideByPOT's own
// existing int64 domain; this function widens the NUMERATOR, not the exponent's own
// range. This function checks the composed exponent q_B + 62 + e_a itself
// (RoundingDivideByPotComposedExponentInDomain, D-SLM676) before it forms 2^exponent
// to divide by. CheckBiasReconcileMagnitudeDomain (checked_chain_funnel.h) is exactly
// `BiasReconcileWide(...) == true`. CheckBiasAccumulateMagnitudeDomain calls
// BiasReconcileWide FIRST and inherits the same exponent rejection, but is NOT equivalent
// to it: on the path where BiasReconcileWide returns true it additionally proves
// `acc_i + result` representable in int64_t, and rejects when that sum overflows. The two
// therefore disagree on real inputs -- pinned by execution at b = INT64_MAX, q_b = -62,
// r_a = 1, e_a = 0, acc_i = 1, where the first returns Ok and the second rejects
// (tests/test_main.cpp, T-1678; the claim that they are identical was written and
// corrected four times before a cell was made to hold it). Both report an exponent-domain
// failure as BiasReconcileProductOutOfDomain -- the identical status an
// in-domain-exponent magnitude failure produces (T-1657 Poirot N-9, confirmation pass
// 3f37ba2, N-17 at 53de03e). CheckRoundingDivideByPotExponentDomain
// (checked_chain_funnel.h) is unchanged and still required at the call site for a
// diagnosis that names the right mechanism -- a caller that skips it and reads either
// magnitude predicate's status alone will attribute an exponent-domain rejection to the
// magnitude. This function has three outcomes, not two:
//   * Exponent out of domain: returns false with *out = 0. There is no wide result to
//     narrow on this path -- the divide never runs.
//   * Exponent in domain, magnitude out of range: returns false with *out set to the
//     low 64 bits of the true wide result, matching this file's own established "may be
//     numerically wrong outside its domain, never UB" convention for an
//     out-of-domain-but-still-total function (IExpEvaluate's own kNotRepresentable
//     outcome above is the precedent this follows).
//   * Exponent in domain, magnitude in range: writes the correctly-rounded (C3, ties
//     away from zero) result to *out and returns true.
bool BiasReconcileWide(int64_t b, int64_t q_b, int64_t r_a, int64_t e_a, int64_t* out);

// T-1657 Poirot Significant 3 (D-SLM676): the ONE derivation of C28's composed exponent
// domain, `0 <= q_B + 62 + e_a <= 63`, that both `BiasReconcileWide` above and
// `CheckRoundingDivideByPotExponentDomain` (checked_chain_funnel.h) call, rather than
// each computing `q_B + 62 + e_a` as plain int64_t arithmetic independently. Plain
// int64_t addition is undefined behavior on overflow, and the overflow is reachable
// even when the TRUE mathematical exponent is inside [0,63]: `q_B = INT64_MAX,
// e_a = -INT64_MAX` sums to 62, but forming `q_B + 62` first overflows before `e_a` is
// added. This function computes the sum in the same portable 128-bit facility
// `BiasReconcileWide`'s own product uses, so no ordering of the three terms can
// overflow. Total over the full int64_t domain of both `q_B` and `e_a`: returns
// whether the true sum lies in `[kRoundingDivideByPotExponentMinI64,
// kRoundingDivideByPotExponentMaxI64]`, and writes the sum's low 64 bits to
// `*out_exponent` unconditionally (meaningful only when the return is true — the same
// "narrow the wide result, meaningful only in-domain" convention `BiasReconcileWide`
// itself uses).
bool RoundingDivideByPotComposedExponentInDomain(int64_t q_B, int64_t e_a, int64_t* out_exponent);

// --- F-S3-7 / §7.2 wide-row (int64 input width) siblings ---------------------
//
// The declarations below are the funnel's approved API surface (§7.2, §5.5, §4.7's
// correction). Bodies are real, bit-exact ports in intmath.cpp, green in the
// standing suite (Claude/Curie/superslm-s3.1-checked-chain-funnel-test-design-
// 2026-07-28.md §4/§8).
//
// These three, plus RequantChainChecked/NarrowRowChecked (checked_chain_funnel.h),
// are the funnel's leaf set: §7.3's CI source check bans every call to them outside
// the funnel's own file and the leaf certification TUs.

// C20/F-S3-7 — MaxAbsReduce's construction at int64 (wide composition row) input
// width: D = max_i |x_i| over the row, then the all-zero-row guard D' = max(D, 1).
// Order-free; every element visited unconditionally. Same contract shape as
// MaxAbsReduce above, at the wider input width the composition's wide rows carry
// (matmul.h's int64 accumulator rows).
//
// **Returns D' in [1, INT64_MAX], NOT [1, 2^31]** (corrected, ac34677 finding S2:
// the row {2^40, 3} returns 1099511627776, which the prior text claimed could not
// happen — directly contradicting the reason the caller exists, since
// RequantChainChecked's own C29 step rejects precisely when D' > 2^31). A caller
// that wants an in-domain guarantee reads the return through C29
// (checked_chain_funnel.cpp), not through this header's own range claim.
//
// At int64 there is no wider built-in type to widen into before the abs, unlike
// the int32 sibling above — the fix (ac34677 finding S1) takes the row's UNSIGNED
// magnitude instead (defined for every int64_t, including INT64_MIN, whose
// magnitude 2^63 has no int64_t representation but has an exact uint64_t one),
// then narrows back to int64_t with a saturating clamp at INT64_MAX so a row
// whose true magnitude exceeds int64_t's positive range still reports a value
// unambiguously > 2^31 rather than wrapping into one the C29 domain check could
// misread as in-domain.
int64_t MaxAbsReduceWide(const int64_t* x, size_t n);

// New (§5.5, T-1254) — the row's signed max and min, order-free (max and min are each
// associative and commutative). Distinct from MaxAbsReduceWide: int32 narrowing
// (NarrowRowChecked, checked_chain_funnel.h) depends on the row's SIGNED extremes,
// not its magnitude — a magnitude bound cannot express int32's asymmetric
// representable range ([-2^31, 2^31-1]), which is exactly why C29's `D' <= 2^31`
// check cannot gate this narrowing (§5.5).
//
// **`n == 0` is in-contract and defined** (ac34677 finding S3): both outputs are
// written 0 and neither `x[0]` nor any other element is read -- the convention
// this wide-row primitive family shares at `n == 0` is "no element is read",
// not "returns 0" (380b75f review N5: MaxAbsReduceWide's own `n == 0` result is
// 1, from its all-zero-row guard, not 0). Neither this primitive nor
// NarrowRowChecked (checked_chain_funnel.h) documented an `n >= 1`
// precondition, and the prior implementation read `x[0]` before testing `n`,
// which crashed on a null `x` with `n == 0`.
void RowBoundsWide(const int64_t* x, size_t n, int64_t* out_max, int64_t* out_min);

// C22/F-S3-7 — RequantTokenCode's already-pinned formula, unchanged, at int64 input
// width (§4.7's correction): q_i = clamp(round_half_away_from_zero((x_i * 127 * R) /
// 2^(62 - s)), -127, 127). Same 128-bit-intermediate construction as the int32-input
// overload above; only the input width changes, because the funnel (§7.2's fold,
// T-1254) never narrows the composition's wide row to int32 before this step.
//
// **Total over its whole documented parameter space: every `int64_t x_i` is safe**
// (ac34677 O1) — the 128-bit composite `|x_i| * 127 * R` overflows into the `> 127`
// clamp rather than wrapping, traced through the UMul/UMulWide/UShrToU64 chain
// (intmath.cpp) at the worst case `|x_i| = 2^63`, `R <= 2^32`. `r` and `s` are not
// independently swept here; they arrive from this same chain's own
// DynamicScaleReciprocal/NormalizeScale, whose own contracts bound them.
int8_t RequantTokenCodeWide(int64_t x_i, int64_t r, int s);

// --- §6.3 nonlinear scalar primitives (i-sqrt C4/C5/C6, i-exp C7/C8/C9) -------
//
// The reproducible-path integer cores only. The float-taking offline derivations
// (`intmath.py`'s `i_exp(q, scale)`, `iexp_scale_constants` / C30, `i_exp_ln2_quantum`)
// are NOT ported here — those live in the converter (offline) or ride the site-composition
// slot (C23–C30). Here: `IExpConstruct`/`IExpEvaluate` consume constants already derived to
// integers.

// C5 — one restoring digit-recurrence iteration per base-4 digit of an int64 radicand: the
// count follows from the type, so it CANNOT be data-dependent (§14, §17 "the cell determinism
// is blind to"). Exactly 32, unconditional; no `while bit > n` prologue (that would make the
// op count a function of the radicand). C6 — the first digit's weight is 1 << 62.
inline constexpr int I_SQRT_ITERATIONS = 32;

// C8 — the i-exp exponent decomposition's clip: I-BERT clamps the shifted logit at
// -I_EXP_CLIP_N · q_ln2, making the far tail total (the clip point's result, not a divergent
// shift). 30 is the I-BERT construction's value.
inline constexpr int I_EXP_CLIP_N = 30;

// C4/C5/C6 — floor(sqrt(n)) over [0, 2^63 − 1] by restoring shift-and-subtract (compare,
// subtract, shift — no division, so i-sqrt stays off §18's int64-division GPU-semantics row).
// Domain n >= 0; n < 0 is out of contract (i-sqrt is defined on non-negative sums). n == 0
// needs no guard (every digit decision takes the else branch, root stays 0).
int64_t ISqrt(int64_t n);

// C4/C5 — the root after each of the exactly-I_SQRT_ITERATIONS digit steps, written into
// `out_iterates` (which must hold I_SQRT_ITERATIONS entries). The trace makes the pinned
// op count falsifiable at runtime: it is I_SQRT_ITERATIONS for every input, so a
// data-dependent trip count is a test failure rather than a reading of the source (§17).
void ISqrtTrace(int64_t n, int64_t out_iterates[I_SQRT_ITERATIONS]);

// C9 — softmax max-subtraction: out[i] = logits[i] − max_j(logits[j]), an integer op that
// puts the maximum at 0 (so every result is <= 0, the domain i-exp requires). Inputs widened
// to int64 so the difference cannot overflow. Undefined on an empty sequence (n >= 1).
void ShiftByMax(const int64_t* logits, size_t n, int64_t* out);

// C7/C8 — the I-BERT second-order integer polynomial core, exp(q·scale) in fixed point, from
// PRE-DERIVED integer constants (q_ln2, q_b, q_c) — the caller supplies them (offline or C30).
// Same decomposition as intmath.py's `i_exp_from_constants`:
//   clipped = max(q, −I_EXP_CLIP_N·q_ln2);  z = −clipped / q_ln2;  q_p = clipped + z·q_ln2;
//   value   = ((q_p + q_b)^2 + q_c) >> z.
//
// **The operation is TWO calls, and that is deliberate (S-HARDEN-0).** `IExpConstruct` below
// validates and produces an `IExpConstruction`; `IExpEvaluate` turns that construction into the
// value. There is no single-call form, because a single-call form must answer a
// contract-violating call with *some* `int64_t`, and every available answer is a legitimate
// in-domain result — `0` included, which `((base² + q_c) >> z)` yields whenever `base² + q_c`
// lands in `[0, 2^z)`. A broken call would be indistinguishable from a real one. Splitting the
// operation removes the question instead of answering it badly: the invalid call cannot be
// written.
//
// **Preconditions are now CHECKED rather than caller-ensured**, and the two are not the same
// promise. `q <= 0` (a positive q has no valid decomposition — beyond `q_ln2` it drives z
// negative) and `q_ln2 >= 1` (q_ln2 == 0 would divide by zero) were previously documented
// preconditions that a release build did not enforce. `IExpConstruct` tests them and reports
// `kBadQ`/`kBadQLn2`, so the caller learns rather than guesses. `out_scale` is never computed at
// runtime (C30: the nonlinear consumers are same-scale ratios and it cancels).
//
// **The coefficients are NOT constrained to be positive.** C7 N2-5's positivity is a property of
// how C30 and the offline derivation produce these constants, not an invariant of this primitive
// — the parameter types admit all of `int64_t`, and `q_b` near `INT64_MIN` was an executed
// overflow (F21). `q_b` is therefore checked for *representability* of `q_p + q_b`, not for sign.
//
// **What changed for callers that were outside the contract (S-HARDEN-0, measured).** Two input
// strips executed no undefined behaviour before this slot and produced values, by accident rather
// than by promise: `0 < q < q_ln2`, where `(−q)/q_ln2` truncates to `z == 0`; and `q_ln2 <= −1`,
// where the clip bound goes positive and `z` lands on `I_EXP_CLIP_N`. Both are now refused. The
// preserved property is the one D-SLM80 actually states — **behaviour-preserving INSIDE the
// domain** — and inside the domain every value is bit-identical, including the wrapped results of
// `kNotRepresentable`. Outside it, a documented precondition that a release build silently
// tolerated is now enforced; nothing depended on the tolerance.

// C7/C8 — the i-exp decomposition, as ONE checked construction/evaluation entry point
// (S-HARDEN-0; findings F9 and F21). Every consumer of the decomposition routes through
// here: the evaluator `IExpEvaluate` and the predicate `IExpConstantsInDomain` are both
// defined in terms of it, so neither can form the decomposition before the checks that
// make it safe have run.
//
// **Why a funnel rather than two accessors plus a documented rule.** The prior API exposed
// `IExpShift`/`IExpBase` so a caller could evaluate the domain instead of re-deriving it,
// and both projected through the clip/divide unguarded. Two orderings then went wrong the
// same way: the evaluator called them BEFORE asserting the domain, so the guard added to
// prevent `−I_EXP_CLIP_N · q_ln2` from overflowing ran after the overflow (F9); and the
// predicate itself formed `q_p + q_b` on a `q_b` that carries no lower bound, so the guard
// callers were told to invoke was undefined on the input class it exists to screen (F21).
// A rule about call order is a rule someone must remember; a funnel is a path that does not
// exist. `StandardsDocument` §4: where a rule can be made structural, make it structural.
//
// **`IExpConstruct` is TOTAL.** It is defined on all four `int64_t` arguments, asserts
// nothing about its own domain, and executes no undefined behaviour on any input — including
// the inputs it rejects. That totality is the fix: a guard that is UB on hostile input is not
// a guard. It is therefore safe to call speculatively, from any caller, on unvalidated
// constants.
enum class IExpDomain : int {
	kOk = 0,            // decomposition formed AND `(base² + q_c) >> z` fits int64_t
	kNotRepresentable,  // decomposition formed; the result does not fit int64_t
	kBadQ,              // `q > 0` — no valid decomposition (it would drive `z` negative)
	kBadQLn2,           // `q_ln2 < 1`, or above the clip ceiling `INT64_MAX / I_EXP_CLIP_N`
	kBadQB,             // `q_p + q_b` is not representable
};

// Declared ahead of the class so the friend declaration below names THIS function rather than
// introducing a new one that unqualified lookup would never find.
class IExpConstruction;

// `[[nodiscard]]` is load-bearing, not decoration. Without it the outcome can be dropped and
// the untouched `*out` read anyway — which reproduces, one level up, the exact defect that
// splitting this operation was meant to remove: a caller gets a plausible value (`0` from a
// default construction, or the PREVIOUS call's values from a reused one) with no diagnostic at
// `-Wall -Wextra`. The split moved the indistinguishability from a returned `int64_t` to a
// discarded enum; this attribute is what actually closes it.
[[nodiscard]] IExpDomain IExpConstruct(int64_t q, int64_t q_ln2, int64_t q_b, int64_t q_c,
                                       IExpConstruction* out);

// A validated decomposition: `z` is the number of ln2 steps the clip/divide yields, `base` is
// `q_p + q_b`, and `q_c` is the constant the representability judgement was made against.
//
// **The fields are private, and that is the load-bearing part of this type.** `IExpEvaluate`
// narrows with `>> z`, so a `z` outside `[0, I_EXP_CLIP_N]` is an unchecked shift and undefined
// behaviour. A public `{z, base}` aggregate would admit `z = 999` from any caller — which is F21's
// own lesson one level up: a *type* that admits values the contract forbids is the defect, whether
// the values arrive as parameters or as fields. Only `IExpConstruct` can populate one, so the two
// reachable origins are a default-constructed `{0, 0, 0}` (safe: `z = 0` is a legal shift) and a
// construction this header validated.
//
// **The exact reach of that claim, stated narrowly on purpose.** It covers every route the
// language offers through the type system — aggregate initialisation, direct field assignment,
// derived types, copy and assignment, value-initialisation. It does **not** cover deliberate
// object-representation punning: `std::bit_cast` or `memcpy` over these three `int64_t`s can
// fabricate any `z`, and `IExpEvaluate` would then shift by it. That is out of contract and is
// not defended against, because nothing at this layer can defend against it. The claim is
// "cannot be written by accident", never "cannot be forged".
//
// **`q_c` is carried rather than passed to the evaluator, and that is not tidiness.** A separate
// `q_c` argument on `IExpEvaluate` would let a caller validate against one constant and evaluate
// against another, silently voiding the representability guarantee — a second derivation of one
// domain, which is the drift finding F10 already recorded happening between the S2.6 design and
// this predicate. Carrying it makes the mismatch unrepresentable.
class IExpConstruction {
public:
	int64_t z() const { return z_; }
	int64_t base() const { return base_; }
	int64_t q_c() const { return q_c_; }

private:
	friend IExpDomain IExpConstruct(int64_t, int64_t, int64_t, int64_t, IExpConstruction*);
	int64_t z_ = 0;
	int64_t base_ = 0;
	int64_t q_c_ = 0;
};

// Forms the decomposition and judges the result, in that order, each step guarded before
// the arithmetic that could overflow it.
//
// **`*out` is filled whenever the decomposition is WELL-FORMED — for `kOk` and for
// `kNotRepresentable` alike — and is left untouched for the three `kBad*` outcomes.** The
// distinction is load-bearing and is not a convenience: `kNotRepresentable` means the
// decomposition is sound and only the narrowing overflows, which is a defined (if wrong)
// result the evaluator still returns, so the caller that wants it needs `z` and `base`.
// The `kBad*` outcomes mean no decomposition exists to report. `out` may be null when only
// the domain answer is wanted.
//
// (Declared above the class; this is the documentation site.)

// C7/C8 — the value, from a construction this header produced: `(base² + q_c) >> z`, carried at
// 128-bit width and narrowed by an arithmetic (floor) shift to match the reference's big-integer
// `>>`.
//
// **TOTAL, and it needs no preconditions of its own.** Every `IExpConstruction` that can exist
// holds a `z` in `[0, I_EXP_CLIP_N]` and a `base` this header formed without overflow, so there is
// no input to this function that is out of contract. That is the whole reason the operation is
// split: the check happens where the caller can act on it, and the arithmetic happens where
// nothing can be wrong.
//
// Evaluating a construction from a `kNotRepresentable` outcome is **legal and defined**: the low 64
// bits are kept and the result can be NEGATIVE. That is the exact value the pre-S-HARDEN-0 code
// returned for the same constants and a committed golden pins it (D-SLM80). A caller that does not
// want it was told so by the outcome.
int64_t IExpEvaluate(const IExpConstruction& c);

// C7/C8 — **the domain predicate, for a caller that wants the yes/no and not the
// construction.** Returns whether `(base² + q_c) >> z` — the value `IExpEvaluate` produces —
// is representable in `int64_t`. A caller that is going to evaluate anyway should call
// `IExpConstruct` and read the outcome instead; this asks the same question and throws the
// construction away.
//
// **Why this exists as a function rather than a documented inequality.** Every other
// precondition this header states is a LOWER bound; there was no upper bound on `q_b` or
// `q_c`, and the evaluator narrows its 128-bit intermediate with an unchecked shift. A blind
// adversary strike produced contract-legal constants (`q = 0`, `q_ln2 = 887904998`,
// `q_b = 1733160715`, `q_c = 2^63−1`) for which the value is a NEGATIVE exponential:
// `base² + q_c` exceeds `INT64_MAX` and the low 64 bits are kept. With `q = 0` the shift is
// 0, so nothing shifts the overflow away.
//
// The test is performed in the 128-bit domain internally **because a caller cannot safely
// perform it**: the obvious check, `base² + q_c <= INT64_MAX`, squares `base` in int64 and
// itself overflows once `q_b` exceeds ~3.04e9 — reproducing the defect in the guard
// (D-SLM81). Callers therefore use this predicate; they do not re-derive it.
//
// **This predicate is TOTAL (S-HARDEN-0, F21).** It is exactly
// `IExpConstruct(q, q_ln2, q_b, q_c, nullptr) == IExpDomain::kOk`, so it validates `q`,
// `q_ln2`, and `q_b`'s range itself rather than inheriting them as caller-ensures
// preconditions, and it executes no undefined behaviour on any `int64_t` input. It
// previously asserted `q <= 0 && q_ln2 >= 1` and formed `q_p + q_b` unguarded, which made
// the guard undefined on `q_b = INT64_MIN` — the input class it exists to screen. Its
// **Its answer is unchanged INSIDE the contract, and changed on two strips outside it** —
// `0 < q < q_ln2` and `q_ln2 <= -1`, where the old predicate answered `true` without executing
// undefined behaviour and this one answers `false`. Those are the same two strips §157 records;
// stating "unchanged wherever the old predicate was defined" here would contradict that, and did.
// Use `IExpConstruct` directly when the reason for a rejection matters.
//
// **Cost note (D-SLM83) — read the condition, it is load-bearing.** `base = q_p + q_b`
// varies with `q` even at a fixed `z`, so this predicate is **not** `q`-invariant in
// general: at `q_ln2 = 4000000000, q_b = 1, q_c = 0`, `q = 0` answers `true` while
// `q = −3999999999` — same triple, same `z = 0` — answers `false`.
//
// The one-call-per-triple shortcut holds **iff `2·q_b >= q_ln2 − 1`.** `q_p` ranges over
// `(−q_ln2, 0]`, so `|base| = |q_p + q_b|` peaks at one of the two ends: `q_b` at `q_p = 0`,
// or `q_ln2 − 1 − q_b` at `q_p = −(q_ln2 − 1)`. The first dominates exactly under that
// inequality, and it is also the element where the bound `2^63·2^z − 1` is tightest (`z = 0`
// there), so largest numerator and tightest bound coincide and one `q = 0` call discharges
// the row. C30's derivation satisfies it comfortably — it fixes `q_b / q_ln2` at ≈1.952 —
// so a C30-fed caller always gets the shortcut. This is not a per-element runtime guard.
//
// **A caller whose constants fail `2·q_b >= q_ln2 − 1` must check BOTH ends** — `q_p = 0`
// and `q_p = −(q_ln2 − 1)` — or simply call this per element. Checking only the far end is a
// false all-clear: at `q_ln2 = 3e9, q_b = 1.8e9, q_c = 7783372039254775806` the element at
// `q_p = −(q_ln2 − 1)` answers `true` while `q_p = 0` answers `false`, and `q_p = 0` is the
// one that makes the parent return a wrapped negative under `NDEBUG`.
[[nodiscard]] bool IExpConstantsInDomain(int64_t q, int64_t q_ln2, int64_t q_b, int64_t q_c);

// C30 — the offline derivation `iexp_scale_constants`, now ported (T-1655, D-SLM620): forms
// the per-query (q_ln2, q_b, q_c) triple `IExpConstruct`/`IExpEvaluate` consume, from a
// canonical carried scale `(m, e)` and the format-30 coefficient triple. TOTAL over every
// int64_t argument, mirroring IExpConstruct's own standard (S-HARDEN-0): every step is
// checked before the arithmetic it protects, so no input drives this function into
// undefined behaviour, including the inputs it rejects.
enum class IExpScaleDomain : int {
	kOk = 0,             // all three constants formed and int64-representable
	kBadCoefficient,      // ln2_q <= 0, b_q <= 0, or ca_q <= 0 (C7 N2-5's positivity precondition)
	kNegativeShift,       // min(shift_ln2, shift_b, shift_c) < 0 -- C30's fine-scale rejection
	kNotRepresentable,    // a well-formed derivation shift produced a value outside int64_t
};

// `*out_q_ln2`/`*out_q_b`/`*out_q_c` are written iff the outcome is kOk; left untouched
// otherwise. `out_q_ln2`/`out_q_b`/`out_q_c` may be null when only the domain answer is
// wanted, matching IExpConstruct's own convention for `out`.
[[nodiscard]] IExpScaleDomain IExpScaleConstants(
    int64_t m, int64_t e,
    int64_t ln2_q, int ln2_fmt, int64_t b_q, int b_fmt, int64_t ca_q, int ca_fmt,
    int64_t* out_q_ln2, int64_t* out_q_b, int64_t* out_q_c);

// C30's pinned, per-artifact-independent format-30 coefficient triple (T-1655,
// D-SLM620): the SAME (ln2_q, b_q, ca_q) `iexp_scale_constants`/
// `tests/gen_s3_1_c30_iexp_domain_sweep_fixtures.py` derive from the pinned
// reference's own `_POLY_A`/`_POLY_B`/`_POLY_C` and the pinned ln2 double
// (`tests/reference/superslm_spike/rope_tables_pinned.json`, "ln2":
// 0x1.62e42fefa39efp-1) — computed HERE at compile time from the SAME pinned bit
// pattern, never by hand-typing the resulting integers (`StandardsDocument` §5.4:
// verified at source or by execution, never by construction). `IExpScaleConstants`'s
// production call site (RunLayerLoop, forward_sites.cpp) passes these three, format
// 30 in each, as its own `ln2_q`/`b_q`/`ca_q` arguments.
inline constexpr double kIExpPinnedLn2 = 0x1.62e42fefa39efp-1;
inline constexpr double kIExpPolyA = 0.3585;
inline constexpr double kIExpPolyB = 1.353;
inline constexpr double kIExpPolyC = 0.344;
inline constexpr int64_t kIExpLn2Q =
    static_cast<int64_t>(kIExpPinnedLn2 * static_cast<double>(int64_t{1} << 30));
inline constexpr int64_t kIExpBQ =
    static_cast<int64_t>(kIExpPolyB * static_cast<double>(int64_t{1} << 30));
inline constexpr int64_t kIExpCaQ = static_cast<int64_t>(
    (kIExpPolyC / kIExpPolyA) * static_cast<double>(int64_t{1} << 30));

// --- §6.4 RoPE rotation (C11/C12/C13) -----------------------------------------
//
// Runtime primitive only. The Q2.30 sin/cos TABLES (C12) are generated OFFLINE in double
// precision — the one place double runs (§6.4) — by the converter, and carried in the
// artifact's ROP1 section (S2.0a). They are NOT ported here; the runtime reads those integer
// tables and applies the rotation with a single pinned rounding.
inline constexpr int ROPE_FRAC_BITS = 30;                 // Q2.30 fixed point
inline constexpr int32_t ROPE_ONE = 1 << ROPE_FRAC_BITS;  // 1.0 in Q2.30 (2^30)

// C28's bias Q-format (T-1656, D-SLM622, §5.2): the format's bias codes are stored at a
// fixed q_B = 30 (docs/sslm_format.md, D-SLM621's own text) -- not an artifact-carried,
// per-tensor value, the same convention this file's other Q-format constants already
// use. If a future artifact revision carries a per-tensor q_B, this constant is the
// single edit point.
inline constexpr int64_t kBiasQFormat = 30;

// The rotated pair, EXACT and UNCLAMPED (see RopeApplyPair). int64 because a rotation can reach
// ~sqrt(2)·|input|, which exceeds int32 for wide inputs.
struct RopePair {
	int64_t x;
	int64_t y;
};

// C11/C13 — rotate one pair: (x·cos − y·sin, x·sin + y·cos). Each component is combined at full
// int64 width, then rounded ONCE by the §6.2 RoundingDivideByPOT primitive at ROPE_FRAC_BITS
// (C13 — ties away from zero, C3; §6.4 pins "exactly one rounding" and defers the scheme to the
// §6.2 primitive rather than a second rounding). `cos_q30`/`sin_q30` are Q2.30 table entries
// (|·| ≤ ROPE_ONE). The result is the exact single-rounded rotation, **UNCLAMPED** — clamping to
// the activation format is the caller's (site composition), exactly as the reference forward does
// it (`dynamic_engine.py`: rotate at width, then clamp to int8).
RopePair RopeApplyPair(int32_t x, int32_t y, int32_t cos_q30, int32_t sin_q30);

// --- §6.2/§5.2 C32 softmax row (arm A) ----------------------------------------
//
// The runtime primitive only. C30's derivation site (the not-yet-built call
// forming (q_ln2, q_b, q_c) from a per-query carried scale) and C32/D-SLM366's
// own width predicate (CheckSoftmaxRowWidthDomain, checked_chain_funnel.h) are
// declared elsewhere; this is the row kernel the funnel's own predicate gates.

// C32's own probability fixed-point width (SuperSLM_S3a_WalkingSkeleton_Plan.md
// §5.2: "PROB_FRAC_BITS = 15"). Named so kSoftmaxRowMaxSafeExponent
// (checked_chain_funnel.h) and SoftmaxRowQ15 below derive from the same
// quantity rather than each restating 15 as a bare literal.
inline constexpr int kProbFracBits = 15;

// C32 — arm-A softmax row normalization (§5.2, §11 S3.3 §6.2 step 5):
//   s[k]   = ShiftByMax(scores)                                    (above)
//   e[k]   = IExpEvaluate(IExpConstruct(s[k], q_ln2, q_b, q_c))     (C7/C8/C9)
//   total  = Sum_k e[k]                                             (int64)
//   p[k]   = (e[k] << kProbFracBits) / max(total, 1)                (both operands
//            non-negative: floor == truncate)
// `max(total, 1)` guards the all-clipped row exactly (every e[k] is then 0, so
// every p[k] is 0). The caller gates this kernel with
// CheckSoftmaxRowWidthDomain(q_b, q_c, width) (checked_chain_funnel.h) BEFORE
// calling it — this function guards `width == 0` itself (below) but performs
// no width *upper-bound* check of its own: no compute in this tree checks a
// width upper bound of its own. `scores`/`out_probs` each have `width`
// elements.
//
// **`width == 0` is guarded in the kernel itself, returning `true`** (D-SLM497;
// T-1411, whole-tree review b9dcbe0 Significant 1;
// Claude/Poirot/9b0f938-t1411-t1415-t1416-t1386-t1388-confirmation-2026-07-31.md
// Significant 1). Below, `ShiftByMax`'s own `n >= 1` precondition (above: "Undefined
// on an empty sequence (n >= 1)") is honoured by never calling it at `width == 0`:
// this kernel returns before touching `scores` or `out_probs` -- vacuous
// well-formedness over zero row elements, matching this function's own "whether
// every row element's i-exp construction was well-formed" contract, which holds
// trivially when there are no elements. `CheckSoftmaxRowWidthDomain`
// (checked_chain_funnel.h) independently rejects `width == 0` under
// reject-over-degrade, so a caller following the documented gate-before-kernel
// contract never reaches this guard; it exists for a caller who invokes this
// public-header kernel directly, without that gate.
//
// **Returns whether every row element's i-exp construction was WELL-FORMED**
// (Poirot 2026-07-28 finding 3): CheckSoftmaxRowWidthDomain bounds q_b and
// q_c but takes no q_ln2 parameter, so it is not a sufficient precondition
// for this kernel -- IExpConstruct can still answer kBadQ, kBadQLn2, or
// kBadQB for an accepted (q_b, q_c) pair (150 reachable (m, e) points
// degenerate to q_ln2 == 0, the coarse-scale underflow tail). The prior body
// discarded IExpConstruct's own [[nodiscard]] outcome ((void)IExpConstruct)
// and evaluated the untouched default construction anyway, silently
// emitting an all-zero probability row where the reference raises. This
// function now reads that outcome per element: a kBad* element contributes
// 0 to exps/total (the reference's own raise has no well-defined numeric
// substitute here, so the row's other elements are still computed) and the
// return value is false. kOk and kNotRepresentable are both well-formed
// decompositions (this header's own IExpConstruct doc: "*out is filled ...
// for kOk and for kNotRepresentable alike") and are evaluated exactly as
// before. **`[[nodiscard]]`, matching this header's own stated doctrine**
// (IExpConstruct's own declaration above: "load-bearing, not decoration.
// Without it the outcome can be dropped and the untouched `*out` read
// anyway"). The same shape applies here one level up: a caller who drops
// this return value gets a plausible-looking `out_probs` row with no
// diagnostic that its own inputs were rejected. A caller that genuinely
// does not need to know whether a row's constants were trustworthy discards
// the outcome explicitly, `(void)SoftmaxRowQ15(...)`, rather than silently.
//
// **Also enforces the row's real per-element peak against `M = q_b*q_b + q_c`
// (Popper 2026-07-28 Null 1, fixed here rather than only at the gate).**
// `CheckSoftmaxRowWidthDomain`'s own contract states `M` bounds every row
// element only under the ratio `2*q_b >= q_ln2 - 1`, and that predicate has
// no `q_ln2` parameter with which to check it -- so a `q_c >= 0`, in-ceiling
// `M` the gate accepts can still be an unsound stand-in for a row that is off
// that ratio (the gate cannot observe this; only the row's own evaluated
// values can). This kernel recomputes `M` from its own `q_b`/`q_c` arguments,
// at the same 128-bit width the gate uses, and treats any element whose real
// evaluated value falls outside `[0, M]` as untrustworthy exactly like a
// kBad* construction -- it contributes 0 to `total` and the return value is
// false. A row whose closed-form `M` is not itself usable (not representable
// in int64_t, `<= 0`, or exceeding `kSoftmaxRowMaxSafeExponent`) makes every
// element of that row untrustworthy, since there is then no bound left to
// check against. **The third condition is enforced by this kernel itself**
// (D-SLM409, plan Sec14.1), not merely relied upon from the gate, so the
// `exps[k] << kProbFracBits` numerator shift below cannot overflow
// UNCONDITIONALLY -- for any element this kernel keeps, `M <=
// kSoftmaxRowMaxSafeExponent` gives `exps[k] << kProbFracBits <= 2^47 << 15
// == 2^62`, safely inside `int64_t`, whether or not the caller invoked
// `CheckSoftmaxRowWidthDomain` first. `total`'s width/sum bound is a
// separate claim and remains conditional: it is provably bounded only when
// the caller followed the documented contract (gate before kernel) -- the
// gate's own sum check proves `width * M <= INT64_MAX` for the same `M`, so
// `total` cannot overflow `int64_t` under that contract. A caller that skips
// the gate and calls this kernel directly still gets the per-element bound
// (enforced unconditionally, above), but not the width bound -- `total` can
// still overflow on an ungated, sufficiently wide row: this kernel guards
// `width == 0` (above) but performs no width upper-bound check of its own.
[[nodiscard]] bool SoftmaxRowQ15(const int64_t* scores, size_t width, int64_t q_ln2, int64_t q_b,
                                  int64_t q_c, int64_t* out_probs);

}  // namespace superslm

#endif  // SUPERSLM_INTMATH_H

// SuperSLM integer-arithmetic kernels — Layer-1 requant primitives and the
// per-token dynamic-scale chain (SuperSLM_Plan.md §6.2, §6.8 C1/C2/C3 + C19–C22).
//
// This is the compiled C++ port of the offline reference `Tools/superslm_spike/
// intmath.py`. Every function here is a bit-for-bit reimplementation of the pinned
// reference: conformance is OUTPUT EXACTNESS against that reference over the whole
// domain, never a blessing of any particular decomposition. Standard library only
// (`<bit>` for the pinned scalar count-leading-zeros); no float on any reproducible
// path — floats appear only as offline constants folded to integers by the converter,
// never here.
//
// The two families:
//   * C1/C2/C3 — gemmlowp's fixed-point requant primitives
//     (`SaturatingRoundingDoublingHighMul`, `RoundingDivideByPOT`) and their
//     composition. Tie directions are pinned and opposite by design: the doubling
//     high-mul rounds toward +infinity (C2), the divide-by-POT rounds away from zero
//     (C3). §15 / D-SLM36.
//   * C19/C20/C21/C22 — the rung-1 per-token dynamic quantizer: max-abs reduction
//     (C20), scale normalization (C21), fixed-point reciprocal (C19, corrected Newton),
//     and the per-token requant composition — the "127 scale wrapper" (C22, D-SLM52).
//
// Cost determinism (§14): every op count here is data-INDEPENDENT. The reciprocal runs
// exactly 3 Newton iterations plus exactly 2 branch-free correction steps; normalization
// runs one bit-scan plus one shift plus one select. No early exit, no data-dependent
// trip count. This property is a source obligation Poirot verifies by reading the code,
// on top of the exhaustive value certification.
//
// Wide intermediates: C19's reciprocal Newton products and C22's `x_i · 127 · R`
// (magnitude up to ~2^70) exceed int64. The port carries them in a portable 128-bit
// intermediate (no reliance on a compiler `__int128`, which MSVC lacks). Conformance is
// by output exactness against the pinned formula, not by the decomposition chosen.
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

// C1/C3 — gemmlowp `RoundingDivideByPOT`: x / 2^exponent, ties AWAY FROM ZERO.
// Defined for exponent in [0, 31]; exponent 0 is the identity. The `+1` carried on the
// negative branch of the threshold is what turns the floor shift's half-up behaviour
// into half-away-from-zero.
int32_t RoundingDivideByPOT(int32_t x, int exponent);

// C3 — the int64-domain sibling of RoundingDivideByPOT: x / 2^exponent, ties AWAY FROM
// ZERO, for exponent in [0, 63]. Delegates to the SAME width-generic rounding template S2.3
// already instantiates at int64 for RopeApplyPair (S23-1's unification) — one tie rule, no
// duplication. Exposed publicly so S2.4's SiLU-LUT index derivation can round its int64
// sub-node position (SuperSLM_S2.4_SiLU_LUT_Design §5, §9).
int64_t RoundingDivideByPOT(int64_t x, int exponent);

// C1 — the two primitives composed in order, both roundings applied:
// `RoundingDivideByPOT(SaturatingRoundingDoublingHighMul(x, m), shift)`.
int32_t MultiplyByQuantizedMultiplier(int32_t x, int32_t quantized_multiplier, int shift);

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
// no rounding. Exactly one bit-scan + one shift + one branch-free select.
NormalizedScale NormalizeScale(int64_t d_prime);

// C19 — the reciprocal's pinned, data-INDEPENDENT op counts (§14): exactly 3 Newton
// iterations then exactly 2 branch-free correction steps. Bare fixed-count Newton is
// deterministically wrong on 54–62% of the domain; the corrections close it.
inline constexpr int DYNAMIC_RECIPROCAL_NEWTON_ITERATIONS = 3;
inline constexpr int DYNAMIC_RECIPROCAL_CORRECTION_STEPS = 2;

// C19 — fixed-point reciprocal: R = round_half_up(2^62 / Dn) for Dn in [2^30, 2^31),
// computed DIVISION-FREE via corrected Newton — exactly 3 Newton iterations then exactly
// 2 unconditional branch-free correction steps. Bare fixed-count Newton is deterministically
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

// --- §6.3 nonlinear scalar primitives (i-sqrt C4/C5/C6, i-exp C7/C8/C9) -------
//
// The reproducible-path integer cores only. The float-taking offline derivations
// (`intmath.py`'s `i_exp(q, scale)`, `iexp_scale_constants` / C30, `i_exp_ln2_quantum`)
// are NOT ported here — those live in the converter (offline) or ride the site-composition
// slot (C23–C30). Here: `IExpFromConstants` consumes constants already derived to integers.

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
// PRE-DERIVED positive integer constants (q_ln2, q_b, q_c) — the caller supplies them (offline
// or C30). Same decomposition as intmath.py's `i_exp_from_constants`:
//   clipped = max(q, −I_EXP_CLIP_N·q_ln2);  z = −clipped / q_ln2;  q_p = clipped + z·q_ln2;
//   return ((q_p + q_b)^2 + q_c) >> z.
// **Preconditions the caller ensures** (structurally upstream — `ShiftByMax` gives q <= 0, and
// C30's fine-scale rejection gives q_ln2 >= 1): `q` is a max-shifted logit (q <= 0; a positive q
// has no valid decomposition — it would drive z negative) and q_ln2 >= 1 (a coarser scale has no
// decomposition to state, and q_ln2 == 0 would divide by zero). The reference `i_exp_from_constants`
// RAISES on either violation; this primitive asserts them (the no-exceptions runtime equivalent) and
// is otherwise UB out of domain — the same caller-ensures convention as `MaxAbsReduce`/`ShiftByMax`,
// not a runtime rejection. The coefficient integers are positive (C7 N2-5). `out_scale` is never
// computed at runtime (C30: the nonlinear consumers are same-scale ratios and it cancels).
int64_t IExpFromConstants(int64_t q, int64_t q_ln2, int64_t q_b, int64_t q_c);

// C7/C8 — `IExpFromConstants`'s internal decomposition, exposed so a caller can EVALUATE
// the domain requirement below instead of re-deriving it (D-SLM79/D-SLM81). Same
// caller-ensures preconditions as the parent (`q <= 0`, `q_ln2 >= 1`); the parent calls
// these, so they are the same values it uses, not a parallel derivation.
//
// `IExpShift` returns `z` — the number of ln2 steps the clip/divide yields. `IExpBase`
// returns `q_p + q_b`, the value the parent squares.
//
// **`z` lies in `[0, I_EXP_CLIP_N]` only while `q_ln2 <= INT64_MAX / I_EXP_CLIP_N`.** Above
// that the clip bound `−I_EXP_CLIP_N · q_ln2` overflows int64 and the decomposition is
// meaningless — the executed witness is `z = −29` at a contract-legal `q_ln2`. `q_ln2` has
// no documented upper bound, so this is a real reachable region; `IExpConstantsInDomain`
// answers `false` throughout it, which is the guard a caller should be relying on.
int64_t IExpShift(int64_t q, int64_t q_ln2);
int64_t IExpBase(int64_t q, int64_t q_ln2, int64_t q_b);

// C7/C8 — **the domain predicate. Call this before `IExpFromConstants` on any constants
// not already proven in range.** Returns whether `(base² + q_c) >> z` — the value the
// parent returns — is representable in `int64_t`.
//
// **Why this exists as a function rather than a documented inequality.** Every other
// precondition this header states is a LOWER bound; there was no upper bound on `q_b` or
// `q_c`, and `IExpFromConstants` narrows its 128-bit intermediate with an unchecked shift.
// A blind adversary strike produced contract-legal constants (`q = 0`, `q_ln2 = 887904998`,
// `q_b = 1733160715`, `q_c = 2^63−1`) for which the parent returns a NEGATIVE exponential:
// `base² + q_c` exceeds `INT64_MAX` and the low 64 bits are kept. With `q = 0` the shift is
// 0, so nothing shifts the overflow away.
//
// The test is performed in the 128-bit domain internally **because a caller cannot safely
// perform it**: the obvious check, `IExpBase(...)² + q_c <= INT64_MAX`, squares `base` in
// int64 and itself overflows once `q_b` exceeds ~3.04e9 — reproducing the defect in the
// guard (D-SLM81). Callers therefore use this predicate; they do not re-derive it.
//
// Same caller-ensures preconditions as the parent (`q <= 0`, `q_ln2 >= 1`) — this predicate
// answers the width question only, and does not validate those.
//
// **Cost note (D-SLM83) — read the condition, it is load-bearing.** `base = q_p + q_b`
// varies with `q` even at a fixed `z`, so this predicate is **not** `q`-invariant in
// general: at `q_ln2 = 4000000000, q_b = 1, q_c = 0`, `q = 0` answers `true` while
// `q = −3999999999` — same triple, same `z = 0` — answers `false`.
//
// The one-call-per-triple shortcut holds **only where C30 derives the constants**, because
// there `q_b / q_ln2` is fixed at ≈1.952, which keeps `base` positive across the row and
// maximal at `q = 0`. A caller in that regime discharges the whole row's obligation with a
// single `q = 0` call, and this is not a per-element runtime guard. **A caller supplying
// constants from anywhere else does not get that shortcut** and must either establish the
// same property for its own constants or check the row's extremes.
bool IExpConstantsInDomain(int64_t q, int64_t q_ln2, int64_t q_b, int64_t q_c);

// --- §6.4 RoPE rotation (C11/C12/C13) -----------------------------------------
//
// Runtime primitive only. The Q2.30 sin/cos TABLES (C12) are generated OFFLINE in double
// precision — the one place double runs (§6.4) — by the converter, and carried in the
// artifact's ROP1 section (S2.0a). They are NOT ported here; the runtime reads those integer
// tables and applies the rotation with a single pinned rounding.
inline constexpr int ROPE_FRAC_BITS = 30;                 // Q2.30 fixed point
inline constexpr int32_t ROPE_ONE = 1 << ROPE_FRAC_BITS;  // 1.0 in Q2.30 (2^30)

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

}  // namespace superslm

#endif  // SUPERSLM_INTMATH_H

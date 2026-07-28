// SuperSLM integer-arithmetic kernels — the bit-exact C++ port of the pinned
// reference Tools/superslm_spike/intmath.py (D-SLM52). See include/superslm/intmath.h
// for the per-function contract. Conformance is OUTPUT EXACTNESS against the reference
// over the whole domain (the standing suite samples it; the exhaustive certifier proves
// it once per device tier — SuperSLM_Plan.md §6.8 Popper boundary).
//
// Wide intermediates. Two operations exceed int64:
//   * C19 Newton `y·delta` (up to ~2^63) and the correction's `2·(2^62 − y·Dn)` (~±2^63),
//   * C22 `x_i · 127 · R` (magnitude up to ~2^70).
// They are carried in a portable 128-bit facility below. Where the compiler offers a
// native 128-bit integer (gcc/clang `__int128`) it is used directly; MSVC (which has no
// such type) gets a two's-complement {lo, hi} struct with hand-rolled multiply / add /
// subtract / arithmetic-shift. The kernels are written ONCE against a common set of
// helper names (`SMul`, `SSub`, `SShrToI64`, `UMul`, …) so there is a single algorithm,
// not two. All struct-path arithmetic is unsigned two's-complement (no signed overflow),
// so the UBSan CI leg is clean.
#include "superslm/intmath.h"

#include <bit>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <vector>

namespace superslm {
namespace {

// --- Portable 128-bit facility ------------------------------------------------

#if defined(__SIZEOF_INT128__)

using U128 = unsigned __int128;
using S128 = __int128;

inline U128 UMul(uint64_t a, uint64_t b) { return static_cast<U128>(a) * b; }
inline U128 UMulWide(U128 a, uint64_t b) { return a * b; }
inline U128 UTwice(U128 a) { return a << 1; }
inline U128 UAdd64(U128 a, uint64_t b) { return a + b; }
// Logical right shift of an unsigned 128-bit value; result assumed to fit in u64.
inline uint64_t UShrToU64(U128 v, int k) { return static_cast<uint64_t>(v >> k); }

inline S128 SFromI64(int64_t v) { return static_cast<S128>(v); }
inline S128 SMul(int64_t a, int64_t b) { return static_cast<S128>(a) * static_cast<S128>(b); }
inline S128 SAdd(S128 a, S128 b) { return a + b; }
inline S128 SSub(S128 a, S128 b) { return a - b; }
inline S128 STwice(S128 a) { return a << 1; }
inline bool SGe(S128 a, S128 b) { return a >= b; }
inline bool SLt(S128 a, S128 b) { return a < b; }
// Arithmetic (floor) right shift; result assumed to fit in i64.
inline int64_t SShrToI64(S128 v, int k) { return static_cast<int64_t>(v >> k); }

#else  // MSVC and any other toolchain without a native 128-bit integer.

struct U128 { uint64_t lo, hi; };
struct S128 { uint64_t lo, hi; };  // two's-complement

// 64x64 -> 128 unsigned (schoolbook, exact).
inline U128 UMul(uint64_t a, uint64_t b) {
	uint64_t ll = (a & 0xFFFFFFFFu) * (b & 0xFFFFFFFFu);
	uint64_t lh = (a & 0xFFFFFFFFu) * (b >> 32);
	uint64_t hl = (a >> 32) * (b & 0xFFFFFFFFu);
	uint64_t hh = (a >> 32) * (b >> 32);
	uint64_t mid = (ll >> 32) + (lh & 0xFFFFFFFFu) + (hl & 0xFFFFFFFFu);
	uint64_t lo = (ll & 0xFFFFFFFFu) | (mid << 32);
	uint64_t hi = hh + (lh >> 32) + (hl >> 32) + (mid >> 32);
	return U128{lo, hi};
}

// 128 * (small u64) -> 128; caller guarantees the product fits 128 bits (it does:
// the C22 use is (<=2^63) * 127).
inline U128 UMulWide(U128 a, uint64_t b) {
	U128 lo = UMul(a.lo, b);
	uint64_t hi = a.hi * b;  // a.hi is small here; no overflow of the retained 128 bits
	return U128{lo.lo, lo.hi + hi};
}

inline U128 UAdd64(U128 a, uint64_t b) {
	uint64_t lo = a.lo + b;
	uint64_t carry = (lo < a.lo) ? 1u : 0u;
	return U128{lo, a.hi + carry};
}

inline U128 UTwice(U128 a) {
	return U128{a.lo << 1, (a.hi << 1) | (a.lo >> 63)};
}

inline uint64_t UShrToU64(U128 v, int k) {
	if (k == 0) return v.lo;
	if (k >= 64) return v.hi >> (k - 64);
	return (v.lo >> k) | (v.hi << (64 - k));
}

inline S128 SFromI64(int64_t v) {
	return S128{static_cast<uint64_t>(v), v < 0 ? ~0ull : 0ull};
}

inline S128 SMul(int64_t a, int64_t b) {
	uint64_t ua = a < 0 ? (~static_cast<uint64_t>(a) + 1u) : static_cast<uint64_t>(a);
	uint64_t ub = b < 0 ? (~static_cast<uint64_t>(b) + 1u) : static_cast<uint64_t>(b);
	U128 m = UMul(ua, ub);
	S128 r{m.lo, m.hi};
	if ((a < 0) ^ (b < 0)) {  // two's-complement negate
		r.lo = ~r.lo;
		r.hi = ~r.hi;
		if (++r.lo == 0) ++r.hi;
	}
	return r;
}

inline S128 SAdd(S128 a, S128 b) {
	uint64_t lo = a.lo + b.lo;
	uint64_t carry = (lo < a.lo) ? 1u : 0u;
	return S128{lo, a.hi + b.hi + carry};
}

inline S128 SSub(S128 a, S128 b) {
	uint64_t lo = a.lo - b.lo;
	uint64_t borrow = (a.lo < b.lo) ? 1u : 0u;
	return S128{lo, a.hi - b.hi - borrow};
}

inline S128 STwice(S128 a) { return SAdd(a, a); }

inline bool SGe(S128 a, S128 b) {  // signed a >= b
	int64_t ah = static_cast<int64_t>(a.hi), bh = static_cast<int64_t>(b.hi);
	if (ah != bh) return ah > bh;
	return a.lo >= b.lo;
}

inline bool SLt(S128 a, S128 b) { return !SGe(a, b); }

// Arithmetic (floor) right shift; result assumed to fit in i64 (we return its low 64
// bits, which for an in-range result carry the correct sign).
inline int64_t SShrToI64(S128 v, int k) {
	if (k == 0) return static_cast<int64_t>(v.lo);
	uint64_t lo = (v.lo >> k) | (v.hi << (64 - k));
	return static_cast<int64_t>(lo);
}

#endif  // 128-bit facility

}  // namespace

// --- §6.2 requant primitives (C1/C2/C3) --------------------------------------

int32_t SaturatingRoundingDoublingHighMul(int32_t a, int32_t b) {
	// (a*b + 2^30) >> 31, arithmetic (floor) shift → ties toward +infinity; only
	// (INT32_MIN, INT32_MIN) exceeds INT32_MAX, and it saturates.
	int64_t ab = static_cast<int64_t>(a) * static_cast<int64_t>(b);
	int64_t result = (ab + (int64_t{1} << 30)) >> 31;
	return result > kInt32Max ? kInt32Max : static_cast<int32_t>(result);
}

namespace {
// The single source of the C1/C3 RoundingDivideByPOT rule (ties away from zero), width-generic
// so the int32 requant primitive and RoPE's int64 rotation intermediate share ONE tie rule — a
// future change to C3 touches exactly one place. x / 2^exponent; the `+1` on the negative branch
// of the threshold turns the floor shift's half-up into half-away-from-zero. exponent in
// [0, bits(T)-1]; exponent 0 → mask 0 (identity).
template <typename T>
T RoundingDivideByPOTImpl(T x, int exponent) {
	using U = std::make_unsigned_t<T>;
	U mask = (static_cast<U>(1) << exponent) - 1u;
	U remainder = static_cast<U>(x) & mask;
	U threshold = (mask >> 1) + (x < 0 ? U{1} : U{0});
	return (x >> exponent) + (remainder > threshold ? 1 : 0);
}
}  // namespace

int32_t RoundingDivideByPOT(int32_t x, int exponent) {
	return RoundingDivideByPOTImpl<int32_t>(x, exponent);
}

int64_t RoundingDivideByPOT(int64_t x, int exponent) {
	// Same tie rule (C3, away from zero), int64 width — the sibling S2.4's SiLU-LUT index
	// derivation needs. Delegates to the one width-generic template (S23-1) that RopeApplyPair
	// already instantiates at int64, so C3 lives in exactly one place across both widths.
	return RoundingDivideByPOTImpl<int64_t>(x, exponent);
}

int32_t MultiplyByQuantizedMultiplier(int32_t x, int32_t quantized_multiplier, int shift) {
	return RoundingDivideByPOT(
	    SaturatingRoundingDoublingHighMul(x, quantized_multiplier), shift);
}

// --- §6.2 rung-1 per-token dynamic-scale chain (C19/C20/C21/C22) --------------

int Clz64(uint64_t n) {
	// Pinned to scalar-reference behaviour (§18). Domain [1, 2^64-1]; n == 0 is out of
	// contract and never reached (NormalizeScale's input D' >= 1).
	return std::countl_zero(n);
}

int64_t MaxAbsReduce(const int32_t* x, size_t n) {
	int64_t d = 0;
	for (size_t i = 0; i < n; ++i) {
		int64_t xi = x[i];  // widen BEFORE abs so INT32_MIN yields 2^31 (not int32 UB)
		int64_t a = xi < 0 ? -xi : xi;
		if (a > d) d = a;
	}
	return d < 1 ? 1 : d;  // all-zero-token guard, exact on that class
}

NormalizedScale NormalizeScale(int64_t d_prime) {
	int p = 63 - Clz64(static_cast<uint64_t>(d_prime));
	int s = 30 - p;
	int64_t dn = s >= 0 ? (d_prime << s) : (d_prime >> 1);
	return NormalizedScale{dn, s};
}

int64_t DynamicScaleReciprocal(int64_t dn) {
	// R = round_half_up(2^62 / Dn), division-free: 3 Newton iterations + 2 branch-free
	// correction steps, all unconditional (cost-determinism, §14). The seed and Newton
	// carry the wide product `y·delta` in 128-bit; the correction's `2·(2^62 − y·Dn)`
	// likewise. Every op count here is data-independent.
	constexpr int64_t kC32 = (2 * (int64_t{48} << 31) + 17) / 34;    // round_half_up(48·2^31/17)
	constexpr int64_t kC32_2 = (2 * (int64_t{32} << 31) + 17) / 34;  // round_half_up(32·2^31/17)

	// Seed y0 ≈ 1/d, d = Dn/2^31, from the minimax line 48/17 − (32/17)·d in Q31. The
	// product kC32_2·Dn is < 2^63, so the seed itself needs no 128-bit intermediate.
	int64_t y = kC32 - ((kC32_2 * dn) >> 31);

	// Newton: y ← y·(2 − d·y) = (y·(2^32 − ((Dn·y) >> 31))) >> 31.
	for (int i = 0; i < DYNAMIC_RECIPROCAL_NEWTON_ITERATIONS; ++i) {
		int64_t dn_y = SShrToI64(SMul(dn, y), 31);
		int64_t delta = (int64_t{1} << 32) - dn_y;
		y = SShrToI64(SMul(y, delta), 31);
	}

	// Correction: q correctly rounded iff −Dn ≤ 2·(2^62 − q·Dn) < Dn. Two unconditional
	// branch-free adjustments close bare Newton's high-biased error.
	for (int i = 0; i < DYNAMIC_RECIPROCAL_CORRECTION_STEPS; ++i) {
		S128 residual_2x = STwice(SSub(SFromI64(int64_t{1} << 62), SMul(y, dn)));
		if (SGe(residual_2x, SFromI64(dn))) {
			y += 1;
		} else if (SLt(residual_2x, SFromI64(-dn))) {
			y -= 1;
		}
	}
	return y;
}

int8_t RequantTokenCode(int32_t x_i, int64_t r, int s) {
	// q_i = clamp(round_half_away_from_zero((x_i·127·R) / 2^(62−s)), −127, 127).
	// R > 0 and 127 > 0, so the composite's sign is x_i's; work in magnitude, apply the
	// away-from-zero rule (C3), then the sign and clamp. |x_i·127·R| reaches ~2^70.
	// No x_i==0 early exit: the general path yields 0 for x_i==0 (magnitude 0 → q 0), so
	// the op count stays data-independent (§14) rather than branching on the input.
	int exponent = 62 - s;  // in [32, 63]
	uint64_t abs_x = x_i < 0 ? (~static_cast<uint64_t>(x_i) + 1u) : static_cast<uint64_t>(x_i);

	U128 prod = UMulWide(UMul(abs_x, static_cast<uint64_t>(r)), 127u);  // |x_i|·127·R
	// magnitude = (2·|prod| + 2^exponent) / 2^(exponent+1), floor (all positive).
	U128 numerator = UAdd64(UTwice(prod), uint64_t{1} << exponent);
	uint64_t magnitude = UShrToU64(numerator, exponent + 1);

	if (magnitude > 127u) magnitude = 127u;  // clamp before sign; symmetric range [−127,127]
	int32_t q = static_cast<int32_t>(magnitude);
	return static_cast<int8_t>(x_i < 0 ? -q : q);
}

// --- F-S3-7 / §7.2 wide-row (int64 input width) siblings ---------------------
//
// Bit-exact wide-width ports of the int32-input siblings above: same construction,
// each element already the wide type (no widen-before-abs step needed for
// MaxAbsReduceWide, since the row already is int64), same 128-bit-intermediate
// requant composite for RequantTokenCodeWide.

int64_t MaxAbsReduceWide(const int64_t* x, size_t n) {
	// Widen to the UNSIGNED magnitude before comparing, not `-x[i]` (ac34677 S1):
	// negating INT64_MIN is signed-integer overflow (UB), and on this toolchain it
	// wraps back to INT64_MIN, which fails the `a > d` test and silently drops the
	// row's largest-magnitude element from the reduction. At int64 there is no
	// wider built-in type to widen into the way the int32 sibling above does; the
	// unsigned two's-complement negate (`~x + 1`, the same form RequantTokenCodeWide
	// already uses below) is defined for every int64_t, including INT64_MIN, whose
	// magnitude 2^63 has no int64_t representation but has an exact uint64_t one.
	uint64_t d = 0;
	for (size_t i = 0; i < n; ++i) {
		const uint64_t a = x[i] < 0 ? (~static_cast<uint64_t>(x[i]) + 1u) : static_cast<uint64_t>(x[i]);
		if (a > d) d = a;
	}
	if (d < 1) d = 1;  // all-zero-row guard, exact on that class
	// Saturating narrow back to int64_t: every row this composition actually
	// produces is far below this ceiling (RequantChainChecked's own C29 domain
	// check rejects anything past 2^31), so this clamp only fires on a magnitude
	// that does not fit int64_t's positive range at all (reachable only from an
	// INT64_MIN element, whose true magnitude is 2^63). Clamping to INT64_MAX
	// keeps the result unambiguously > 2^31 rather than wrapping into a value C29
	// could misread as in-domain.
	return d > static_cast<uint64_t>(INT64_MAX) ? INT64_MAX : static_cast<int64_t>(d);
}

void RowBoundsWide(const int64_t* x, size_t n, int64_t* out_max, int64_t* out_min) {
	// n == 0 is in-contract (ac34677 S3): neither this primitive's header nor
	// NarrowRowChecked's contract documents an n >= 1 precondition, and reading
	// x[0] unconditionally is a null-pointer dereference on an empty row. Defined
	// as (0, 0) -- the shared wide-row convention at n == 0 is "no element is
	// read", not "returns 0" (380b75f review N5: MaxAbsReduceWide's own n == 0
	// result is 1, from its all-zero-row guard). NarrowRowChecked's C35 check
	// accepts (0, 0) trivially and its subsequent NarrowAccumulatorToI32 call is
	// a no-op loop over zero elements.
	if (n == 0) {
		*out_max = 0;
		*out_min = 0;
		return;
	}
	int64_t mx = x[0];
	int64_t mn = x[0];
	for (size_t i = 1; i < n; ++i) {
		if (x[i] > mx) mx = x[i];
		if (x[i] < mn) mn = x[i];
	}
	*out_max = mx;
	*out_min = mn;
}

int8_t RequantTokenCodeWide(int64_t x_i, int64_t r, int s) {
	// Same composite as RequantTokenCode above (C22's formula, unchanged) at
	// int64 input width: q_i = clamp(round_half_away_from_zero((x_i·127·R) /
	// 2^(62−s)), −127, 127).
	int exponent = 62 - s;  // in [32, 63]
	uint64_t abs_x = x_i < 0 ? (~static_cast<uint64_t>(x_i) + 1u) : static_cast<uint64_t>(x_i);

	U128 prod = UMulWide(UMul(abs_x, static_cast<uint64_t>(r)), 127u);  // |x_i|·127·R
	U128 numerator = UAdd64(UTwice(prod), uint64_t{1} << exponent);
	uint64_t magnitude = UShrToU64(numerator, exponent + 1);

	if (magnitude > 127u) magnitude = 127u;
	int32_t q = static_cast<int32_t>(magnitude);
	return static_cast<int8_t>(x_i < 0 ? -q : q);
}

// --- §6.3 nonlinear scalar primitives (i-sqrt C4/C5/C6, i-exp C7/C8/C9) --------

namespace {
// The root after each of the exactly-I_SQRT_ITERATIONS digit steps (restoring
// shift-and-subtract). Compare / subtract / shift — no division (keeps i-sqrt off §18's
// int64-division GPU-semantics row). All 32 iterations run UNCONDITIONALLY: skipping leading
// zero digits (the `while bit > n` prologue) would make the op count a function of the
// radicand, which §14 forbids — below the leading digit the comparison simply takes the else
// branch. No overflow: remainder <= n < 2^63; root reaches 2^62 and bit 2^62, so
// trial = root + bit reaches ~2^62 + 2^60 < 2^63 (the leading iterations dominate).
void ISqrtIterates(int64_t n, int64_t out[I_SQRT_ITERATIONS]) {
	int64_t remainder = n;
	int64_t root = 0;
	int64_t bit = int64_t{1} << 62;  // C6: the first digit's weight, 4^31
	for (int i = 0; i < I_SQRT_ITERATIONS; ++i) {
		int64_t trial = root + bit;
		if (remainder >= trial) {
			remainder -= trial;
			root = (root >> 1) + bit;
		} else {
			root >>= 1;
		}
		bit >>= 2;
		out[i] = root;
	}
}
}  // namespace

int64_t ISqrt(int64_t n) {
	int64_t it[I_SQRT_ITERATIONS];
	ISqrtIterates(n, it);
	return it[I_SQRT_ITERATIONS - 1];
}

void ISqrtTrace(int64_t n, int64_t out_iterates[I_SQRT_ITERATIONS]) {
	ISqrtIterates(n, out_iterates);
}

void ShiftByMax(const int64_t* logits, size_t n, int64_t* out) {
	// Puts the maximum at 0 (so every result is <= 0, i-exp's domain). Inputs are int64 so
	// the difference cannot overflow for int32-range logits widened by the caller. n >= 1.
	int64_t peak = logits[0];
	for (size_t i = 1; i < n; ++i) {
		if (logits[i] > peak) peak = logits[i];
	}
	for (size_t i = 0; i < n; ++i) out[i] = logits[i] - peak;
}

int64_t IExpEvaluate(const IExpConstruction& c) {
	// S-HARDEN-0 (F9, F21). This function has NO preconditions and asserts nothing, because
	// there is no input to it that can be out of contract: an IExpConstruction's fields are
	// private and only IExpConstruct can populate one, so every value reaching here was either
	// validated by that function or is the default {0, 0, 0}. `z` is therefore in
	// [0, I_EXP_CLIP_N] and `base` was formed without overflow.
	//
	// That is the point of splitting construct from evaluate. The prior single-call shape had
	// to answer a contract-violating call with some int64_t, and every candidate answer -- 0
	// included -- is a legitimate in-domain result, so a broken call was indistinguishable from
	// a real one. Here the check happens where the caller can act on it and the arithmetic
	// happens where nothing can be wrong.
	//
	//   value = (base^2 + q_c) >> z
	//
	// base^2 reaches ~2^62 and q_c ~2^62 in the realistic domain (near 0.5*INT64_MAX), and
	// C30's per-token constants may push wider -- so carry it wide and take an arithmetic
	// (floor) shift to match the reference's big-integer `>>`.
	//
	// For a construction from a kNotRepresentable outcome the narrowing keeps the low 64 bits
	// and the result can be NEGATIVE; a blind strike produced contract-legal constants that did
	// exactly that (D-SLM78). That value is bit-identical to what the pre-S-HARDEN-0 code
	// returned for the same constants, which is the behaviour-preservation property D-SLM80
	// records and a committed golden pins. The caller was told by the outcome; it is not this
	// function's job to withhold the value.
	const S128 v = SAdd(SMul(c.base(), c.base()), SFromI64(c.q_c()));
	return SShrToI64(v, static_cast<int>(c.z()));  // z in [0, I_EXP_CLIP_N] (=30), fits int
}

namespace {

// `clip_lo` is `−I_EXP_CLIP_N · q_ln2`, which overflows int64 once q_ln2 exceeds this.
// Beyond it the decomposition is meaningless — the executed witness is z = −29 at a
// contract-legal q_ln2 — so IExpConstruct refuses there rather than blessing it. This is a
// SECOND axis from the width question: q_ln2 has no documented upper bound either, the
// same gap that produced the original strike on q_c.
constexpr int64_t kIExpMaxQLn2 = INT64_MAX / static_cast<int64_t>(I_EXP_CLIP_N);

}  // namespace

IExpDomain IExpConstruct(int64_t q, int64_t q_ln2, int64_t q_b, int64_t q_c,
                         IExpConstruction* out) {
	// S-HARDEN-0 (F9, F21). The checks below run in DEPENDENCY ORDER: each one is what
	// makes the next line's arithmetic well defined, so no check may be moved after the
	// operation it protects. That ordering is the whole fix, and it is why this is one
	// function rather than a predicate callers are asked to remember to call first — the
	// previous shape put the ceiling test after the overflow it existed to prevent (F9),
	// and left the predicate itself undefined on the input class it screens (F21).
	//
	// This function is TOTAL: every path below is defined for all four int64 arguments,
	// including the ones it rejects. A guard that is UB on hostile input is not a guard.

	// A positive q has no valid decomposition — it drives z negative, and the parent's
	// final shift with it. (Found by execution during this slot's test authoring; a third
	// UB class, distinct from F9 and F21, and closed here by the same funnel.)
	if (q > 0) return IExpDomain::kBadQ;

	// q_ln2 == 0 divides by zero below and a negative q_ln2 has no decomposition to state.
	// Above kIExpMaxQLn2 the clip bound overflows — this is F9's own ceiling, now tested
	// BEFORE clip_lo is formed rather than after.
	if (q_ln2 < 1 || q_ln2 > kIExpMaxQLn2) return IExpDomain::kBadQLn2;

	// Safe now, and only now: |clip_lo| <= I_EXP_CLIP_N · (INT64_MAX / I_EXP_CLIP_N) <=
	// INT64_MAX, so clip_lo > INT64_MIN and the negation of `clipped` cannot overflow
	// either. z <= I_EXP_CLIP_N follows from clipped >= clip_lo, so z · q_ln2 fits.
	const int64_t clip_lo = -static_cast<int64_t>(I_EXP_CLIP_N) * q_ln2;
	const int64_t clipped = q < clip_lo ? clip_lo : q;
	const int64_t z = (-clipped) / q_ln2;     // clipped <= 0, q_ln2 >= 1 → non-negative
	const int64_t q_p = clipped + z * q_ln2;  // q_p in (−q_ln2, 0]

	// F21: q_b carries no lower bound anywhere. The public header states coefficient
	// positivity as a fact about how C7/C30 derive the constants, not as a precondition,
	// and the parameter type admits all of int64 — so `q_p + q_b` is checked, not assumed.
	// Only underflow is possible because q_p <= 0, and `INT64_MIN - q_p` cannot itself
	// overflow for q_p <= 0.
	if (q_b < INT64_MIN - q_p) return IExpDomain::kBadQB;
	const int64_t base = q_p + q_b;

	// The decomposition is well formed from here, so it is reported even when the result
	// below is not representable — that outcome is a defined (wrapped) return the parent
	// still produces, and a caller reproducing it needs these two values.
	if (out != nullptr) {
		out->z_ = z;
		out->base_ = base;
		// Carried, not re-supplied at evaluation: a caller that could validate against one q_c
		// and evaluate against another would hold a representability guarantee about a value it
		// never computed. That is a second derivation of one domain, which F10 already recorded
		// drifting between the S2.6 design and this predicate.
		out->q_c_ = q_c;
	}

	const S128 v = SAdd(SMul(base, base), SFromI64(q_c));

	// `(v >> z)` is an arithmetic (floor) shift, so it is representable in int64 exactly
	// when `-2^63 · 2^z <= v <= 2^63 · 2^z - 1`. Both bounds are built by doubling rather
	// than by a 128-bit left shift so this uses only the wide facility both toolchain paths
	// already provide (no `__int128` dependency on MSVC). `z <= I_EXP_CLIP_N` (30), so the
	// bound peaks at 2^93 — far inside S128.
	S128 limit = STwice(SFromI64(INT64_C(1) << 62));  // 2^63
	for (int64_t i = 0; i < z; ++i) limit = STwice(limit);

	const S128 upper = SSub(limit, SFromI64(1));      //  2^63·2^z − 1
	const S128 lower = SSub(SFromI64(0), limit);      // −2^63·2^z

	// The lower test is UNREACHABLE-AS-FALSE for this signature, and that is deliberate.
	// `base²` is non-negative and `q_c >= INT64_MIN` by its own type, so the smallest `v`
	// any int64 pair can produce is `INT64_MIN` — which is exactly `lower` at `z = 0` and
	// strictly inside it at every `z > 0`. No representable input can make it reject, so no
	// test can kill it by deletion (Poirot's M4 mutation is executed-identical to this code;
	// his M1, `lower = 0`, IS caught by the committed cells).
	//
	// It stays because it states the predicate the function CLAIMS — "representable in
	// int64" — rather than the narrower one today's argument types happen to guarantee.
	// Delete it and the function is correct only by accident of the signature: widen `q_c`,
	// or admit a negative `base²` term, and the guarantee silently disappears with nothing
	// failing. Keeping it costs one comparison and is the difference between correct by
	// construction and correct by coincidence.
	//
	// The paragraph above rests on `q_c` being 64-bit — and names widening `q_c` as the very
	// event that would make the branch live. So it is asserted rather than merely written
	// down: widen the parameter and this fails at compile time instead of leaving a comment
	// that quietly became false.
	static_assert(sizeof(q_c) == 8,
	              "IExpConstruct's lower bound is documented as unreachable BECAUSE "
	              "q_c is 64-bit (so v >= INT64_MIN == lower at z=0). Widening q_c makes the "
	              "branch reachable and that comment wrong — re-derive it, and give the "
	              "lower bound test coverage, before changing this signature.");
	return (SGe(upper, v) && SGe(v, lower)) ? IExpDomain::kOk : IExpDomain::kNotRepresentable;
}

bool IExpConstantsInDomain(int64_t q, int64_t q_ln2, int64_t q_b, int64_t q_c) {
	// The predicate IS the entry point, asked for one bit of its answer. Defining it any
	// other way is how the two derivations of one domain drift apart — which this file has
	// already recorded happening once (evaluation record F10), and which the S2.6 design
	// and the shipped predicate did to each other independently.
	return IExpConstruct(q, q_ln2, q_b, q_c, nullptr) == IExpDomain::kOk;
}

// --- §6.4 RoPE rotation (C11/C12/C13) -----------------------------------------

RopePair RopeApplyPair(int32_t x, int32_t y, int32_t cos_q30, int32_t sin_q30) {
	// (x·cos − y·sin, x·sin + y·cos) combined at full int64 width (each product ≤ 2^61, each
	// sum ≤ 2^62 — no int64 overflow), then rounded ONCE at ROPE_FRAC_BITS through the SAME C1/C3
	// rounding rule as the int32 requant primitive (RoundingDivideByPOTImpl<int64_t>, so the
	// ~2^62 intermediate does not truncate). Unclamped: the exact single-rounded rotation, which
	// can reach ~sqrt(2)·|input| (2^32 at the corners); the caller clamps to the activation format.
	int64_t xr = static_cast<int64_t>(x) * cos_q30 - static_cast<int64_t>(y) * sin_q30;
	int64_t yr = static_cast<int64_t>(x) * sin_q30 + static_cast<int64_t>(y) * cos_q30;
	return RopePair{RoundingDivideByPOTImpl<int64_t>(xr, ROPE_FRAC_BITS),
	                RoundingDivideByPOTImpl<int64_t>(yr, ROPE_FRAC_BITS)};
}

// --- §6.2/§5.2 C32 softmax row (arm A) ----------------------------------------

bool SoftmaxRowQ15(const int64_t* scores, size_t width, int64_t q_ln2, int64_t q_b, int64_t q_c,
                    int64_t* out_probs) {
	// C32 (§5.2, §11 S3.3 §6.2 step 5): ShiftByMax -> per-element
	// IExpConstruct/IExpEvaluate -> sum -> Q15 divide. The caller gates this
	// kernel with CheckSoftmaxRowWidthDomain(q_b, q_c, width) before calling
	// it; this function performs no width check of its own, matching every
	// other funnel-adjacent compute in this tree (the domain check and the
	// compute are separate calls).
	std::vector<int64_t> shifted(width);
	ShiftByMax(scores, width, shifted.data());

	std::vector<int64_t> exps(width);
	int64_t total = 0;
	bool all_well_formed = true;
	for (size_t k = 0; k < width; ++k) {
		IExpConstruction construction;
		// Poirot 2026-07-28 finding 3: the outcome is [[nodiscard]] and is
		// now read, not thrown away. kOk and kNotRepresentable both leave a
		// well-formed construction (IExpConstruct's own doc) and are
		// evaluated exactly as before; the three kBad* outcomes leave the
		// default-constructed {0,0,0} untouched, and evaluating that (the
		// prior behaviour) silently produced 0 rather than surfacing the
		// refusal -- this element instead contributes 0 to the sum directly
		// and marks the row untrustworthy, without evaluating a construction
		// IExpConstruct never validated.
		const IExpDomain d = IExpConstruct(shifted[k], q_ln2, q_b, q_c, &construction);
		if (d == IExpDomain::kBadQ || d == IExpDomain::kBadQLn2 || d == IExpDomain::kBadQB) {
			all_well_formed = false;
			exps[k] = 0;
			continue;
		}
		exps[k] = IExpEvaluate(construction);
		total += exps[k];
	}
	const int64_t denom = total > 1 ? total : 1;  // guards the all-clipped row exactly
	for (size_t k = 0; k < width; ++k) {
		out_probs[k] = (exps[k] << kProbFracBits) / denom;  // both operands non-negative
	}
	return all_well_formed;
}

}  // namespace superslm

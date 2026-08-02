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

#include "superslm/checked_chain_funnel.h"  // kSoftmaxRowMaxSafeExponent (D-SLM409, plan Sec14.1)

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
// T-1655/D-SLM620 (IExpScaleConstants): logical right shift keeping the FULL 128-bit
// result, for k in [0, 127] -- k >= 128 saturates to 0 (never asked of the operator
// form above, which is only well-defined for k < 128).
inline U128 UShrFull(U128 v, int k) {
	if (k >= 128) return static_cast<U128>(0);
	if (k <= 0) return v;
	return v >> k;
}
// Representable in int64_t: non-negative (by construction of every caller here) and
// not past INT64_MAX.
inline bool UFitsI64(U128 v) { return v <= static_cast<U128>(INT64_MAX); }
inline U128 U128Zero() { return static_cast<U128>(0); }

inline S128 SFromI64(int64_t v) { return static_cast<S128>(v); }
inline S128 SMul(int64_t a, int64_t b) { return static_cast<S128>(a) * static_cast<S128>(b); }
inline S128 SAdd(S128 a, S128 b) { return a + b; }
inline S128 SSub(S128 a, S128 b) { return a - b; }
inline S128 STwice(S128 a) { return a << 1; }
inline bool SGe(S128 a, S128 b) { return a >= b; }
inline bool SLt(S128 a, S128 b) { return a < b; }
// Arithmetic (floor) right shift; result assumed to fit in i64.
inline int64_t SShrToI64(S128 v, int k) { return static_cast<int64_t>(v >> k); }
// T-1657/D-SLM621/D-SLM675: full-width arithmetic (floor) right shift, unlike
// SShrToI64 above which assumes (and narrows to) only the low 64 bits. Contract is
// k in [0,63] -- RoundingDivideByPOTWide's own only caller is BiasReconcileWide, which
// calls it only after BiasReconcileWide's own internal
// RoundingDivideByPotComposedExponentInDomain check (D-SLM675) confirms the composed
// exponent is in that range; no caller-side gate is what enforces this. T-1657 Poirot Minor 3: narrowed from a documented [0,127] (this
// path's own MSVC struct sibling below had two branches, k>=64 and k>=128, that no
// caller of THIS function ever drove) to the range the one real caller actually
// needs; the native __int128 shift below is well-defined over the narrower range
// exactly as it was over the wider one, so this is a contract narrowing, not a
// behavior change. A future caller needing [0,127] gets it from UShrFull's own wider
// contract (that function DOES have a caller needing the full width, IExpScaleConstants)
// rather than reviving an unexercised branch here.
inline S128 SShrFull(S128 v, int k) { return v >> k; }
// Representable in int64_t: the high bits are exactly the sign-extension of the low
// 64 bits (IExpEvaluate's own narrowing comment, made an explicit, checked test here).
inline bool SFitsI64(S128 v) { return v == static_cast<S128>(static_cast<int64_t>(v)); }
inline int64_t SLow64(S128 v) { return static_cast<int64_t>(v); }

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

// T-1655/D-SLM620 (IExpScaleConstants): logical right shift keeping the FULL 128-bit
// result, for k in [0, 127] -- k >= 128 saturates to 0. The operator-shift form above
// (UShrToU64) is undefined once k >= 128 (`v.hi >> (k - 64)` becomes an out-of-range
// shift once `k - 64 >= 64`); this function is never called with k outside [0, 127].
inline U128 UShrFull(U128 v, int k) {
	if (k <= 0) return v;
	if (k >= 128) return U128{0, 0};
	if (k >= 64) return U128{v.hi >> (k - 64), 0};
	return U128{(v.lo >> k) | (v.hi << (64 - k)), v.hi >> k};
}

// Representable in int64_t: non-negative (by construction of every caller here) and
// not past INT64_MAX.
inline bool UFitsI64(U128 v) { return v.hi == 0 && v.lo <= static_cast<uint64_t>(INT64_MAX); }
inline U128 U128Zero() { return U128{0, 0}; }

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

// T-1657/D-SLM621/D-SLM675: full-width arithmetic (floor) right shift, unlike
// SShrToI64 above which assumes (and narrows to) only the low 64 bits. Contract is
// k in [0,63] -- this function's own only caller, RoundingDivideByPOTWide, is itself
// only ever called by BiasReconcileWide, which calls it only after BiasReconcileWide's
// own internal RoundingDivideByPotComposedExponentInDomain check (D-SLM675) confirms
// the composed exponent is in that range; no caller-side gate is what enforces this.
// T-1657 Poirot Minor 3: this
// used to document and implement k in [0,127] (mirroring UShrFull's own [0,127]
// convention above, generalized to a SIGNED, sign-extending shift) with two branches,
// k >= 64 and k >= 128, that no caller of THIS function ever drove -- correct on
// inspection, but relied on by nothing the suite executes. Narrowed to the range the
// one real caller actually needs rather than left standing for a future caller to
// rely on unverified; a future k in [64,127] need gets it from UShrFull's own wider
// contract (that function DOES have a caller needing the full width,
// IExpScaleConstants), not from reviving branches here. Two's-complement {lo, hi}:
// v = hi*2^64 + lo with hi read as signed -- shifting right by k in [1,63] combines
// bits from both words exactly as UShrToU64 does, but with hi's OWN right shift kept
// arithmetic (sign-extending) rather than logical; k == 0 is the identity (the
// combined-word formula below is itself UB at k == 0, `hi << 64`).
inline S128 SShrFull(S128 v, int k) {
	if (k <= 0) return v;
	const uint64_t lo = (v.lo >> k) | (v.hi << (64 - k));
	const int64_t hi = static_cast<int64_t>(v.hi) >> k;
	return S128{lo, static_cast<uint64_t>(hi)};
}

// Representable in int64_t: the high word is exactly the sign-extension of the low
// word's own sign bit (IExpEvaluate's own narrowing comment, made an explicit,
// checked test here rather than assumed).
inline bool SFitsI64(S128 v) {
	const uint64_t sign_ext = (static_cast<int64_t>(v.lo) < 0) ? ~0ull : 0ull;
	return v.hi == sign_ext;
}
inline int64_t SLow64(S128 v) { return static_cast<int64_t>(v.lo); }

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

namespace {
// C3's tie-away-from-zero rule (RoundingDivideByPOTImpl<T> above), generalized to a
// 128-bit numerator divided by 2^exponent (T-1657, D-SLM621/641/645) -- the SAME rule,
// generalized only in the numerator's width, never the exponent's own domain or the
// rounding rule itself. Because exponent <= 63 always (this function's own only
// caller, BiasReconcileWide, returns before this divide runs when the composed
// exponent is outside [0,63] -- D-SLM675 -- the contract is enforced upstream, not
// assumed), the rounding remainder
// and threshold are exactly the low 64 bits of `x`'s own two's-complement pattern
// masked to `exponent` bits -- no 128-bit remainder arithmetic is needed, only a
// 128-bit arithmetic shift for the quotient itself.
S128 RoundingDivideByPOTWide(S128 x, int exponent) {
	const uint64_t mask = (uint64_t{1} << exponent) - 1u;
#if defined(__SIZEOF_INT128__)
	const uint64_t x_lo = static_cast<uint64_t>(x);
	const bool x_negative = x < 0;
#else
	const uint64_t x_lo = x.lo;
	const bool x_negative = static_cast<int64_t>(x.hi) < 0;
#endif
	const uint64_t remainder = x_lo & mask;
	const uint64_t threshold = (mask >> 1) + (x_negative ? 1u : 0u);
	const S128 shifted = SShrFull(x, exponent);
	return remainder > threshold ? SAdd(shifted, SFromI64(1)) : shifted;
}
}  // namespace

bool RoundingDivideByPotComposedExponentInDomain(int64_t q_B, int64_t e_a, int64_t* out_exponent) {
	// T-1657 Poirot Significant 3 (D-SLM676): q_B + 62 + e_a formed in the SAME
	// portable 128-bit facility BiasReconcileWide's own product uses, never as plain
	// int64_t arithmetic (see intmath.h for why: the naive sum can overflow -- signed
	// UB -- even when the true mathematical exponent is inside [0,63]). SFromI64 is an
	// exact int64_t -> S128 widen (always representable, no overflow possible); SAdd on
	// two such widened terms cannot itself overflow S128 (each term's magnitude is
	// under 2^63, and three of them sum to under 2^65, far inside S128's ~2^127 range).
	const S128 sum = SAdd(SAdd(SFromI64(q_B), SFromI64(62)), SFromI64(e_a));
	const bool in_domain =
	    SGe(sum, SFromI64(kRoundingDivideByPotExponentMinI64)) &&
	    SLt(sum, SFromI64(int64_t{kRoundingDivideByPotExponentMaxI64} + 1));
	*out_exponent = SLow64(sum);  // meaningful only when in_domain is true
	return in_domain;
}

bool BiasReconcileWide(int64_t b, int64_t q_b, int64_t r_a, int64_t e_a, int64_t* out) {
	// C28's widened bias-reconciliation core (T-1657, D-SLM621/641/645) -- see
	// intmath.h for the full contract.
	int64_t exponent = 0;
	if (!RoundingDivideByPotComposedExponentInDomain(q_b, e_a, &exponent)) {
		// T-1657 Poirot Significant 3: the composed exponent itself is out of
		// [0,63] (or would only appear in range by wrapping int64_t, which is not
		// this function's story to tell) -- this file's own established
		// out-of-domain-but-total convention (IExpEvaluate's precedent): report
		// non-representable, write a defined-but-not-meaningful value, never UB.
		// `RoundingDivideByPOTWide`'s own [0,63] contract is never invoked with an
		// out-of-contract shift.
		*out = 0;
		return false;
	}
	const S128 wide = SMul(b, r_a);  // exact, total over the full int64_t domain of
	                                  // both operands (SMul's own contract, this file).
	const S128 rounded = RoundingDivideByPOTWide(wide, static_cast<int>(exponent));
	const bool fits = SFitsI64(rounded);
	*out = SLow64(rounded);
	return fits;
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

namespace {

// T-1655/D-SLM620: the same saturating-add technique CombineCarriedScale
// (checked_chain_funnel.cpp) uses for its own exponent fold, reproduced here rather than
// shared across a translation-unit boundary -- this file's own convention (see this TU's
// portable-128-bit facility above) is a small, self-contained helper set per TU rather
// than a cross-TU dependency for one function. Two operands of the SAME sign overflow
// exactly when the wrapped result's sign does not match theirs.
inline bool AddOverflows64ForIExpScale(int64_t a, int64_t b, int64_t* out) {
	const uint64_t ua = static_cast<uint64_t>(a);
	const uint64_t ub = static_cast<uint64_t>(b);
	*out = static_cast<int64_t>(ua + ub);  // well-defined: unsigned wraparound, then C++20's
	                                        // mandated two's-complement narrowing (P0907R4)
	return (a >= 0) == (b >= 0) && (*out >= 0) != (a >= 0);
}

inline int64_t SaturatingAdd64ForIExpScale(int64_t a, int64_t b) {
	int64_t out;
	if (!AddOverflows64ForIExpScale(a, b, &out)) return out;
	return (a >= 0) ? INT64_MAX : INT64_MIN;
}

}  // namespace

IExpScaleDomain IExpScaleConstants(int64_t m, int64_t e, int64_t ln2_q, int ln2_fmt,
                                    int64_t b_q, int b_fmt, int64_t ca_q, int ca_fmt,
                                    int64_t* out_q_ln2, int64_t* out_q_b, int64_t* out_q_c) {
	// C30's derivation (design §4.2), TOTAL over its full int64_t domain in every
	// argument -- the same standard IExpConstruct (S-HARDEN-0) holds itself to.

	// Step 1: C7 N2-5's positivity precondition (mirrors the Python ValueError).
	if (ln2_q <= 0 || b_q <= 0 || ca_q <= 0) return IExpScaleDomain::kBadCoefficient;

	// Step 2: the three derivation shifts, each through the SAME saturating-add
	// technique CombineCarriedScale already uses, for the identical reason -- `e` is an
	// unconstrained int64_t and a plain `+`/`2*` here would reproduce the already-fixed
	// T-1596 defect class at a new call site.
	const int64_t shift_ln2 = SaturatingAdd64ForIExpScale(
	    SaturatingAdd64ForIExpScale(static_cast<int64_t>(ln2_fmt), 62), e);
	const int64_t shift_b = SaturatingAdd64ForIExpScale(
	    SaturatingAdd64ForIExpScale(static_cast<int64_t>(b_fmt), 62), e);
	const int64_t two_e = SaturatingAdd64ForIExpScale(e, e);
	const int64_t shift_c = SaturatingAdd64ForIExpScale(
	    SaturatingAdd64ForIExpScale(static_cast<int64_t>(ca_fmt), 124), two_e);

	int64_t min_shift = shift_ln2;
	if (shift_b < min_shift) min_shift = shift_b;
	if (shift_c < min_shift) min_shift = shift_c;
	if (min_shift < 0) return IExpScaleDomain::kNegativeShift;

	// Step 3: C19's reciprocal over the canonical/mid-composition mantissa. Same-TU
	// call, no funnel-ban issue (this function lives in the leaf-owning translation
	// unit).
	const int64_t r_m = DynamicScaleReciprocal(m);
	// r_m is non-negative over this function's full reachable domain -- established by
	// execution (design §4.2 step 6, the standalone DynamicScaleReciprocal-replica
	// sweep), not by "any product fits 128 bits". Its unsigned reinterpretation
	// therefore equals its true value exactly, which is what makes the UMul calls
	// below exact rather than merely in-range.
	const uint64_t ur_m = static_cast<uint64_t>(r_m);

	// Step 4 (corrected, D-SLM643): ln2_q*r_m and b_q*r_m are formed WIDE, never as a
	// plain int64_t multiply -- r_m is not bounded by 2^32 the way the original
	// derivation assumed, and a plain multiply here is genuine signed-overflow UB at a
	// reachable, non-canonical mid-composition m (CombineCarriedScale's own
	// renormalization does not guarantee canonicality, checked_chain_funnel.h's own
	// CarriedScale doc). Both operands are non-negative (ln2_q/b_q > 0, checked above;
	// r_m >= 0), so the unsigned product equals the true product exactly.
	const U128 num_ln2 = UMul(static_cast<uint64_t>(ln2_q), ur_m);
	const U128 num_b = UMul(static_cast<uint64_t>(b_q), ur_m);

	// Step 6: q_c's wide intermediate, ca_q * r_m^2 -- true magnitude can exceed the
	// original ~2^94 estimate (step 4's premise no longer holds) but never 2^128 (r_m is
	// int64_t-domain by construction, so r_m^2 < 2^126, and ca_q < 2^30).
	const U128 rm2 = UMul(ur_m, ur_m);
	const U128 num_c = UMulWide(rm2, static_cast<uint64_t>(ca_q));

	// Step 7 (corrected shift-width guard, generalized to all three per step 5): a
	// shift >= 128 has a floor-shift result of exactly 0 for any value that fits U128
	// (num_ln2/num_b/num_c all do, by construction above) -- so UShrToU64/UShrFull are
	// never asked to shift outside their own proven [0, 127] domain.
	const uint64_t u_q_ln2 = (shift_ln2 >= 128) ? 0 : UShrToU64(num_ln2, static_cast<int>(shift_ln2));
	const uint64_t u_q_b = (shift_b >= 128) ? 0 : UShrToU64(num_b, static_cast<int>(shift_b));
	const uint64_t u_q_c = (shift_c >= 128) ? 0 : UShrToU64(num_c, static_cast<int>(shift_c));

	// Step 8: representability, checked against the FULL shifted width -- UShrToU64
	// assumes/returns only the low 64 bits of the true shifted value, and a genuinely
	// too-large shifted value can still carry nonzero bits above bit 63 that narrowing
	// alone would silently drop.
	const U128 full_ln2 = (shift_ln2 >= 128) ? U128Zero() : UShrFull(num_ln2, static_cast<int>(shift_ln2));
	const U128 full_b = (shift_b >= 128) ? U128Zero() : UShrFull(num_b, static_cast<int>(shift_b));
	const U128 full_c = (shift_c >= 128) ? U128Zero() : UShrFull(num_c, static_cast<int>(shift_c));
	if (!UFitsI64(full_ln2) || !UFitsI64(full_b) || !UFitsI64(full_c)) {
		return IExpScaleDomain::kNotRepresentable;
	}

	if (out_q_ln2 != nullptr) *out_q_ln2 = static_cast<int64_t>(u_q_ln2);
	if (out_q_b != nullptr) *out_q_b = static_cast<int64_t>(u_q_b);
	if (out_q_c != nullptr) *out_q_c = static_cast<int64_t>(u_q_c);
	return IExpScaleDomain::kOk;
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
	// D-SLM497: `width == 0` is guarded here, before `scores` or `out_probs`
	// is touched. `ShiftByMax` below documents its own `n >= 1` precondition
	// and this kernel's own contract is "whether every row element's i-exp
	// construction was well-formed" -- vacuously true over zero elements, so
	// `true` is the answer that keeps that contract meaningful rather than
	// silently degrading. `CheckSoftmaxRowWidthDomain` (checked_chain_funnel.h)
	// independently rejects `width == 0` under reject-over-degrade; this
	// guard exists for the caller who invokes this public-header kernel
	// directly, without that gate.
	if (width == 0) {
		return true;
	}
	// C32 (§5.2, §11 S3.3 §6.2 step 5): ShiftByMax -> per-element
	// IExpConstruct/IExpEvaluate -> sum -> Q15 divide. The caller gates this
	// kernel with CheckSoftmaxRowWidthDomain(q_b, q_c, width) before calling
	// it; this function guards `width == 0` above but performs no width
	// upper-bound check of its own: no compute in this tree checks a width
	// upper bound of its own. It DOES independently enforce the per-element
	// peak below (Popper 2026-07-28 Null 1) -- that is not a width check, it
	// is the one thing only this function can observe: the row's real,
	// computed values.
	//
	// M = q_b*q_b + q_c, the same closed form CheckSoftmaxRowWidthDomain
	// (checked_chain_funnel.cpp) forms and judges, recomputed here at the
	// same 128-bit width from this call's own q_b/q_c so this kernel does
	// not trust that the caller invoked -- or correctly implemented -- that
	// gate. The gate's own contract states M bounds every row element other
	// than the shifted-max one only under the ratio 2*q_b >= q_ln2 - 1
	// (intmath.h:414), a precondition the gate has no q_ln2 parameter to
	// check; this kernel does not attempt that ratio either. What it does
	// instead is refuse to accumulate any element whose REAL evaluated
	// value exceeds M, which needs no ratio precondition because it is a
	// direct observation of the data rather than a claim about it.
	const S128 m128 = SAdd(SMul(q_b, q_b), SFromI64(q_c));
	// Usable as a per-element ceiling only if it is representable in
	// int64_t, strictly positive, AND within the ratified numerator ceiling
	// (D-SLM409, plan Sec14.1) -- a non-positive or unrepresentable M is not
	// a real row peak (no legitimate i-exp magnitude is <= 0), and an M past
	// kSoftmaxRowMaxSafeExponent is exactly what the prior two-conjunct form
	// of this test admitted: an element could evaluate to exactly M, pass
	// the `value > m` check below unchanged, and then overflow
	// `exps[k] << kProbFracBits`, wrapping to a wrong (often zero)
	// probability returned under `well_formed == true` (executed at
	// q_b=0, q_c=2^60, width=1: well_formed=true, out_probs[0]=0 where a
	// single-element row must be full scale). The third conjunct is the
	// SAME named ceiling CheckSoftmaxRowWidthDomain enforces
	// (checked_chain_funnel.h), so this kernel's per-element bound now
	// holds unconditionally -- independent of whether the caller invoked
	// that gate first. A row whose closed-form M fails any of the three
	// tests gets no per-element bound to check against and every element of
	// it is untrustworthy.
	const bool m_usable = SGe(SFromI64(INT64_MAX), m128) && SGe(m128, SFromI64(1)) &&
	                       SGe(SFromI64(kSoftmaxRowMaxSafeExponent), m128);
	const int64_t m = m_usable ? SShrToI64(m128, 0) : 0;

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
		const int64_t value = IExpEvaluate(construction);
		// Popper 2026-07-28 Null 1, closed here: a well-formed construction
		// can still evaluate far past M off the gate's unstated ratio
		// (kSoftmaxRowOffRatioWitness reaches ~9x10^16 past the D-SLM367
		// ceiling with M == 0). Refusing any element outside [0, M] is the
		// direct, precondition-free check the closed-form gate cannot make.
		if (!m_usable || value < 0 || value > m) {
			all_well_formed = false;
			exps[k] = 0;
			continue;
		}
		exps[k] = value;
		total += exps[k];
	}
	// `denom` can no longer silently absorb a wrapped-negative `total`: every
	// summed term above is proven `<= m` (a value CheckSoftmaxRowWidthDomain
	// itself bounds to kSoftmaxRowMaxSafeExponent, 2^47), so `total` stays
	// within int64_t whenever the caller followed the documented gate-before-
	// kernel contract, and `max(total, 1)` guards only the genuine
	// all-clipped/all-refused row (every e[k] == 0) it was always meant to.
	const int64_t denom = total > 1 ? total : 1;
	for (size_t k = 0; k < width; ++k) {
		out_probs[k] = (exps[k] << kProbFracBits) / denom;  // both operands non-negative
	}
	return all_well_formed;
}

}  // namespace superslm

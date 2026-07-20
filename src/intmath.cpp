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
#include <cstddef>
#include <cstdint>

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

int32_t RoundingDivideByPOT(int32_t x, int exponent) {
	// x / 2^exponent, ties away from zero. exponent in [0, 31]; the +1 on the negative
	// branch of the threshold turns the floor shift's half-up into half-away-from-zero.
	uint32_t mask = (uint32_t{1} << exponent) - 1u;  // exponent 0 → mask 0 (identity)
	uint32_t remainder = static_cast<uint32_t>(x) & mask;
	uint32_t threshold = (mask >> 1) + (x < 0 ? 1u : 0u);
	return (x >> exponent) + (remainder > threshold ? 1 : 0);
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

// --- §6.3 nonlinear scalar primitives — STUB (S2.2 red-phase) -----------------
// Deliberately-wrong sentinel bodies so Curie's S2.2 red suite compiles+links+fails.
// Brunel replaces each with the bit-exact port in the S2.2 green phase.

int64_t ISqrt(int64_t) {
	return -1;  // stub
}

void ISqrtTrace(int64_t, int64_t out_iterates[I_SQRT_ITERATIONS]) {
	for (int i = 0; i < I_SQRT_ITERATIONS; ++i) out_iterates[i] = -1;  // stub
}

void ShiftByMax(const int64_t*, size_t n, int64_t* out) {
	for (size_t i = 0; i < n; ++i) out[i] = -1;  // stub
}

int64_t IExpFromConstants(int64_t, int64_t, int64_t, int64_t) {
	return -1;  // stub
}

}  // namespace superslm

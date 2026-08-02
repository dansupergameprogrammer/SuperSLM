// SuperSLM S3.2/S3.3/S3.4 site compositions.
//
// See include/superslm/forward_sites.h for the contract
// (SuperSLM_S3a_WalkingSkeleton_Plan.md §11 S3.2, §11 S3.3, §11 S3.4; C31,
// C24/C25, C28, F-S3-8, C27, C33, C34). The S3.2 bodies (FloorDivI64, RmsNormSite,
// ApplyWeightScaleFold, BiasReconcile, EmbedEntry) and the S3.3 bodies
// (LandingRescale, ClampRopeCode, RopeApplySite) are the real green
// construction against the red suite authored in tests/test_main.cpp
// (Claude/Curie/superslm-s3.2-weightless-and-projection-sites-test-design-
// 2026-07-28.md §11; superslm-s3.3-attention-interior-test-design-2026-07-28.md
// §11/§12; superslm-s3.3-rope-application-site-test-design-2026-07-28.md §6).
// RopeApplySite's own real three-step composition (CheckPositionOverCap
// first, then the ROP1 table read, then RopeApplyPair+ClampRopeCode per
// pair) replaces the prior red-first STUB named in D-SLM376, D-SLM383,
// D-SLM384, D-SLM386, per this build's own record
// (Claude/Brunel/superslm-s3.3-rope-application-site-body-build-2026-07-28.md).
// MlpActSite (S3.4, C34, T-1345) is likewise a real green construction here,
// replacing its own declare-and-stub against
// Claude/Curie/superslm-s3.4-mlp-act-site-test-design-2026-07-29.md's red
// suite (Claude/Brunel/superslm-s3.4-mlp-act-site-body-build-2026-07-29.md).
#include "superslm/forward_sites.h"

#include <string>
#include <vector>

#include "superslm/intmath.h"
#include "superslm/silu_lut.h"  // SiluSigmoidQ15 (C34's LUT construction, MlpActSite step 2)
#include "superslm/silu_lut_canonical.h"  // kSiluLutCanonicalTable (RunLayerLoop's MlpActSite call)
#include "superslm/matmul.h"  // GemmInt8AccumulateRow / GemmProbQ15Accumulate (RunLayerLoop)

namespace superslm {

namespace {

// pipeline.py:190: NORM_FRAC_BITS = 16 (SuperSLM_S3a_WalkingSkeleton_Plan.md
// §5.1's own pinned integer — the normalization divide and the per-element
// wide-row divide both shift by 2*NORM_FRAC_BITS).
constexpr int kNormFracBits = 16;

// Little-endian byte-assembly read of one int64 element from a ROP1 tensor's
// stored bytes — the same discipline the loader itself uses for this exact
// section (src/model.cpp's RdI64/ValidateRopeTablesDomain) and for every
// other untrusted-alignment array in this tree (model.h's own
// SslmKeyedConstants::Value comment: "read them with the byte-assembly
// reader, never a cast (the array is not guaranteed aligned for an int64
// load)"). `base` is a tensor's `data` pointer; `index` is a flat element
// index into that tensor's row-major [context_cap, head_dim/2] layout.
int64_t ReadRopeTableEntryI64(const uint8_t* base, uint64_t index) {
	const uint8_t* p = base + index * 8;
	uint64_t v = 0;
	for (int i = 0; i < 8; ++i) v |= static_cast<uint64_t>(p[i]) << (8 * i);
	return static_cast<int64_t>(v);
}

// --- LandingRescale's own portable 128-bit facility ---------------------------
//
// C27's landing composite (§8.1) forms `branch_code * m_a * r_t`, a magnitude
// derived (Claude/Brunel build log, this campaign) as reaching ~2^90-2^91 at
// this site's own realistic operand ranges (m_a canonical in [2^30, 2^31),
// r_t in [2^31+1, 2^32], branch_code up to the §4.7 projection-accumulator
// bound ~2^27) — and int64 is left almost immediately: at the DOMAIN'S OWN
// minimum operand values, |branch_code| >= 4 already overflows int64's
// representable range. A genuine, never-narrowed wide intermediate is
// therefore not a conservative choice here, it is the only correct one. This
// is a small, self-contained unsigned-magnitude facility (matching
// intmath.cpp's own portable-128-bit precedent, not the same instance —
// that one lives in an anonymous namespace private to that translation
// unit): 91 bits of headroom below the 128-bit ceiling comfortably covers
// this site's own derived magnitude.
struct U128 {
	uint64_t lo = 0, hi = 0;
};

inline U128 U128Mul64(uint64_t a, uint64_t b) {
	const uint64_t ll = (a & 0xFFFFFFFFull) * (b & 0xFFFFFFFFull);
	const uint64_t lh = (a & 0xFFFFFFFFull) * (b >> 32);
	const uint64_t hl = (a >> 32) * (b & 0xFFFFFFFFull);
	const uint64_t hh = (a >> 32) * (b >> 32);
	const uint64_t mid = (ll >> 32) + (lh & 0xFFFFFFFFull) + (hl & 0xFFFFFFFFull);
	const uint64_t lo = (ll & 0xFFFFFFFFull) | (mid << 32);
	const uint64_t hi = hh + (lh >> 32) + (hl >> 32) + (mid >> 32);
	return U128{lo, hi};
}

// 128 * small-u64 -> 128; caller guarantees the true product fits 128 bits
// (it does here: at most ~2^91, far below the 2^128 ceiling this widening
// assumes, so `a.hi * b`'s own overflow is unreachable for any operand this
// site ever forms).
inline U128 U128MulSmall(U128 a, uint64_t b) {
	const U128 lo_part = U128Mul64(a.lo, b);
	const uint64_t hi_part = a.hi * b;
	return U128{lo_part.lo, lo_part.hi + hi_part};
}

inline U128 U128Add(U128 a, U128 b) {
	const uint64_t lo = a.lo + b.lo;
	const uint64_t carry = (lo < a.lo) ? 1u : 0u;
	return U128{lo, a.hi + b.hi + carry};
}

// Logical left shift by k in [0, 127]; k outside that range saturates to 0
// (the identically-shaped convention U128ShrToU64 below uses), since no
// caller here ever needs a k this large to be meaningful.
inline U128 U128Shl(U128 v, int k) {
	if (k <= 0) return v;
	if (k >= 128) return U128{0, 0};
	if (k >= 64) return U128{0, v.lo << (k - 64)};
	return U128{v.lo << k, (v.hi << k) | (v.lo >> (64 - k))};
}

inline U128 U128OneShl(int k) { return U128Shl(U128{1, 0}, k); }

// Logical right shift by k in [0, 127]; the result is assumed (by every
// caller here) to fit in 64 bits.
inline uint64_t U128ShrToU64(U128 v, int k) {
	if (k <= 0) return v.lo;
	if (k >= 128) return 0;
	if (k >= 64) return v.hi >> (k - 64);
	return (v.lo >> k) | (v.hi << (64 - k));
}

// Logical right shift by k in [0, 127], keeping the full 128-bit result
// (LandingRescale's finding-3 remedy: shifting a value left then right by
// the same amount and comparing to the original is how a lost-bits left
// shift is DETECTED without needing wider-than-128-bit arithmetic).
inline U128 U128Shr(U128 v, int k) {
	if (k <= 0) return v;
	if (k >= 128) return U128{0, 0};
	if (k >= 64) return U128{v.hi >> (k - 64), 0};
	return U128{(v.lo >> k) | (v.hi << (64 - k)), v.hi >> k};
}

// --- LandingRescale's composed-exponent arithmetic, made overflow-safe as a class (T-1596) ---
//
// `k = 62 - (e_a - e_t)` is two chained signed subtractions, and neither
// `e_a` nor `e_t` carries a domain check on every path that reaches this
// site: `ResidualReconcileSite`'s `e_t` is `hidden_scale.e`, a
// caller-restorable `SequenceLayerState` field the tree has never bounded
// (forward_sites.cpp's own `hidden_scale` exemption comment, below), and
// `RunLayerLoop`'s direct K/V-landing call (two call sites below in this
// file) passes `KvLandingReciprocals`' word 1 as `e_t`, which
// `model.cpp::ValidateKvLandingReciprocalsDomain` checks against only a
// FLOOR (`kKvLandingExponentMin = -60`), never a ceiling. Computing either
// subtraction directly in signed `int64_t` executes signed-integer-overflow
// UB whenever that step's true (infinite-precision) result falls outside
// `int64_t`'s range -- reached directly through the public `RunLayerLoop`
// (Poirot 8f63577-t1602 Significant 1's own witness: `hidden_scale = {1,
// INT64_MAX}` drives the FIRST subtraction over, `e_a=-2, e_t=INT64_MAX`,
// reported verbatim as "-2 - 9223372036854775807").
//
// `SubOverflows64` computes `a - b` with unsigned wraparound (well-defined)
// and the standard two's-complement overflow test -- operand signs differ
// AND the wrapped result's sign does not match the minuend's `a`. The
// wrapped value itself is NOT a usable fallback: for this exact witness it
// wraps to `INT64_MAX`, the OPPOSITE sign from the true (very negative)
// difference, verified by execution below. `SaturatingSub64` instead
// substitutes `INT64_MAX`/`INT64_MIN` on overflow, keyed off `a`'s own sign
// (the correct direction: `a - b` overflows toward `+infinity` exactly when
// `a >= 0` and `b < 0`, and toward `-infinity` exactly when `a < 0` and
// `b >= 0`).
//
// `ComposedExponent` applies this twice -- `e_a - e_t`, then `62 -` that
// result -- and clamps the outcome to a small magnitude before either
// branch below ever narrows `k` (or `-k`) to `int`. Neither branch needs an
// exact `k` once its magnitude reaches the 128-bit shift ceiling
// `U128Shl`/`U128Shr` already saturate at (the `shift >= 128` special case
// below, and the round-divide branch's own "floors correctly to 0 for
// arbitrarily large k" comment): both branches' own decision is unchanged
// by clamping any k whose magnitude already exceeds that ceiling to a
// smaller one that still exceeds it. The clamp is not merely cosmetic: an
// unclamped `int64_t` k of astronomical magnitude, narrowed straight to
// `int` (well-defined per C++20, but not the intended value), can land on
// an ARBITRARY small `int` rather than one that reads as "huge" -- verified
// by execution: this witness's own naive (overflowing) computation, if the
// overflow were merely left to wrap rather than fixed, truncates to a
// `shift` of `-63`, a small negative number that would misroute the
// left-shift branch's own `shift >= 128` dispatch entirely. Clamping to a
// bound safely inside `int` and safely past 128 removes both hazards at
// once, and changes no result this function's own realistic operand range
// (Popper's derived `e_a` range `[-80, 39]`, `e_t`'s load-time floor `-60`)
// ever produced, because that range never approached the clamp.
inline bool SubOverflows64(int64_t a, int64_t b, int64_t* out) {
	const uint64_t ua = static_cast<uint64_t>(a);
	const uint64_t ub = static_cast<uint64_t>(b);
	*out = static_cast<int64_t>(ua - ub);  // well-defined: unsigned wraparound, then C++20's
	                                        // mandated two's-complement narrowing (P0907R4)
	return (a >= 0) != (b >= 0) && (*out >= 0) != (a >= 0);
}

inline int64_t SaturatingSub64(int64_t a, int64_t b) {
	int64_t out;
	if (!SubOverflows64(a, b, &out)) return out;
	return (a >= 0) ? INT64_MAX : INT64_MIN;
}

// Comfortably past the 128-bit shift ceiling either branch below treats
// specially, and comfortably inside `int`'s own range -- see the comment
// above `SubOverflows64`.
constexpr int64_t kComposedExponentClamp = 4096;

inline int64_t ComposedExponent(int64_t e_a, int64_t e_t) {
	const int64_t diff = SaturatingSub64(e_a, e_t);
	int64_t k = SaturatingSub64(int64_t{62}, diff);
	if (k > kComposedExponentClamp) k = kComposedExponentClamp;
	if (k < -kComposedExponentClamp) k = -kComposedExponentClamp;
	return k;
}

}  // namespace

int64_t FloorDivI64(int64_t a, int64_t b) {
	// C31 (§5.1): the greatest integer q with q*b <= a, for b > 0 (caller-
	// ensures). C++'s own `/` truncates toward zero; floor division differs
	// from it exactly when the divide is inexact and the numerator is negative
	// (F-S3-2) -- correct for that one case by decrementing the truncated
	// quotient.
	const int64_t q = a / b;
	const int64_t r = a % b;
	return (r != 0 && r < 0) ? q - 1 : q;
}

SslmForwardStatus RmsNormSite(const int8_t* h, const int32_t* g, size_t hidden_size,
                               CarriedScale /*incoming_scale*/, CarriedScale site_constant,
                               int8_t* out_codes, CarriedScale* out_scale,
                               std::string_view site, size_t token_index,
                               SslmTraceHookState* trace_hook_state) {
	// C31 (§5.1, §6.2 step 1/9): sumsq -> ISqrt(FloorDivI64(...)) ->
	// max(root,1) -> per-element FloorDivI64(h[i]<<2*NORM_FRAC_BITS, root)*g[i]
	// -> the funnel, with the incoming span EMPTY. `incoming_scale` is accepted
	// per the header's own contract but never folded in: the norm's carried
	// output scale is C23's gain-derived (scale-killing) form alone (Coverage
	// Model dim 7).
	int64_t sumsq = 0;
	for (size_t i = 0; i < hidden_size; ++i) {
		const int64_t hi = static_cast<int64_t>(h[i]);
		sumsq += hi * hi;
	}
	int64_t root =
	    ISqrt(FloorDivI64(sumsq << (2 * kNormFracBits), static_cast<int64_t>(hidden_size)));
	root = root > 1 ? root : 1;

	std::vector<int64_t> wide(hidden_size);
	for (size_t i = 0; i < hidden_size; ++i) {
		const int64_t hi = static_cast<int64_t>(h[i]);
		wide[i] = FloorDivI64(hi << (2 * kNormFracBits), root) * static_cast<int64_t>(g[i]);
	}

	// §11 S3.1a (D-SLM362): `site`/`token_index`/`trace_hook_state` are
	// forwarded unchanged -- this site never fixes its own name (the same
	// composition serves every RMSNorm instance in the per-layer forward), so
	// the caller's own site string, token index, and model-handle hook state
	// are exactly what reaches the funnel's own emission seam.
	const ChainResult result = RequantChainChecked(wide.data(), hidden_size,
	                                                std::span<const CarriedScale>{}, site_constant,
	                                                out_codes, out_scale, site, token_index,
	                                                trace_hook_state);
	return result.status;
}

int64_t ApplyWeightScaleFold(int64_t acc, int32_t identity, int32_t mult, int32_t shift) {
	// C24/C25 (§4.3, §6.2 step 2/6/10/12): identity==1 is the true pass-through
	// (acc unchanged, no multiply, no shift); identity==0 applies the already-
	// shipped MultiplyByQuantizedMultiplier. `identity` is load-time validated
	// to {0,1} (WeightScaleIdentityNotBool, model.cpp), so no third value ever
	// reaches this dispatch.
	if (identity != 0) return acc;
	return static_cast<int64_t>(
	    MultiplyByQuantizedMultiplier(static_cast<int32_t>(acc), mult, shift));
}

int64_t BiasReconcile(int64_t b, int64_t q_b, int64_t r_a, int64_t e_a) {
	// C28 (§4.4, §6.2 step 2): round_half_away_from_zero(B * R_a /
	// 2^(q_B + 62 + e_a)). RoundingDivideByPOT's int64 overload already ties
	// away from zero (C3), which is load-bearing here because B is signed
	// (§4.4). The composed exponent's domain is the caller's own check
	// (CheckRoundingDivideByPotExponentDomain, checked_chain_funnel.h) before
	// this function is ever invoked; it performs no domain check of its own.
	const int64_t exponent = q_b + 62 + e_a;
	return RoundingDivideByPOT(b * r_a, static_cast<int>(exponent));
}

int64_t LandingRescale(int64_t branch_code, int64_t m_a, int64_t r_t, int64_t e_a, int64_t e_t,
                        uint64_t* out_saturation_count, bool* out_magnitude_exceeded_int64) {
	// C27's residual_reconcile (§8.1, dynamic_engine.py-vendored formula):
	//   round_half_away_from_zero((branch_code * m_a * r_t) / 2^(62 - (e_a - e_t)))
	// with a negative composite exponent an EXACT left shift (no rounding).
	// `r_t` is positive by construction (KvLandingReciprocals' offline
	// reciprocal, checked at load time -- ValidateKvLandingReciprocalsDomain,
	// model.cpp). `m_a` is NOT (Popper 2026-07-28 §3.1, finding 2): a
	// mid-composition carried mantissa need only fit int32_t's own range
	// (checked_chain_funnel.h's CarriedScale doc), no sign, and a negative one
	// is reachable through the already-wired RmsNormSite/RequantChainChecked
	// path from an artifact-legal CompositionConstants entry -- the prior
	// unconditional `static_cast<uint64_t>(m_a)` treated it as always positive
	// and was wrong by ~13 orders of magnitude on a negative witness, while
	// also falsely incrementing the saturation counter. The product's sign is
	// `branch_code`'s XOR `m_a`'s -- work in magnitude on both (matching C22's
	// own requant composite convention, intmath.cpp), apply C3's
	// away-from-zero tie rule, then reapply the combined sign.
	const bool branch_negative = branch_code < 0;
	const bool m_a_negative = m_a < 0;
	const bool negative = branch_negative != m_a_negative;
	const uint64_t abs_branch = branch_negative ? (~static_cast<uint64_t>(branch_code) + 1u)
	                                             : static_cast<uint64_t>(branch_code);
	const uint64_t abs_m_a =
	    m_a_negative ? (~static_cast<uint64_t>(m_a) + 1u) : static_cast<uint64_t>(m_a);
	const U128 magnitude = U128MulSmall(U128Mul64(abs_branch, abs_m_a), static_cast<uint64_t>(r_t));

	// T-1596: computed by `ComposedExponent` (this file's own anonymous
	// namespace, above), not by the direct `62 - (e_a - e_t)` this line used
	// to read -- neither subtraction is safe in plain `int64_t` over the
	// domain `e_a`/`e_t` actually carry (see that function's own comment).
	// The RESULT this function returns for every input the direct
	// computation already computed without overflow is unchanged -- `k`
	// itself is clamped once its magnitude passes the 128-bit ceiling
	// `U128Shl`/`U128Shr` already saturate at, at which point neither
	// branch's decision below depends on `k`'s own value; what changes is
	// that an input which used to execute signed-overflow UB here now
	// resolves to a clamped, defined k that reaches the same saturating
	// path (below) an in-range but merely-large k already reaches.
	const int64_t k = ComposedExponent(e_a, e_t);
	int64_t raw;
	// Popper 2026-07-28 §3.2 / finding 3: neither e_t nor e_a carries a domain
	// check anywhere in this tree, and an extreme composed exponent drives
	// the negative-k branch's shift amount past the 128-bit carry's own
	// width. The prior code narrowed `U128Shl`'s own saturating-to-zero
	// result straight into `raw`, returning a silently wrong 0 with the
	// saturation counter untouched -- exactly the class the counter exists to
	// catch (D-SLM201). `magnitude_exceeds_clamp` records that loss so the
	// counter is not fooled by it, independent of what `raw` narrows to.
	bool magnitude_exceeds_clamp = false;
	// T-1377 / D-SLM457 (§7.2b, §14.14): the SECOND, narrower loss signal --
	// "the true 128-bit magnitude does not fit int64" -- computed alongside
	// `magnitude_exceeds_clamp` above but never OR'd with the `> 127` clamp
	// test: a wide row at this site legitimately carries values outside
	// `+/-127` (§6.2 step 8 composes no clamp here), so that test alone would
	// flag every ordinary large-but-correct result. `kInt64MaxU` is
	// `INT64_MAX` widened to `uint64_t`, the same threshold §7.2b's own
	// derivation names ("the divisor 2^(k+1) ... must bring that magnitude
	// below 2^63").
	constexpr uint64_t kInt64MaxU = static_cast<uint64_t>(INT64_MAX);
	bool magnitude_exceeds_int64 = false;
	if (k >= 0) {
		// round_half_away_from_zero(magnitude / 2^k) == floor((2*magnitude + 2^k) / 2^(k+1)),
		// both terms carried in the same 128-bit space as `magnitude` itself
		// (a plain `uint64_t{1} << k` is undefined behaviour once k >= 64,
		// which this site's own operand ranges reach routinely). An oversized
		// divisor floors the QUOTIENT to 0 correctly (2^k already exceeding
		// the 128-bit magnitude means the true quotient is 0), but a SMALL k
		// gives no such guarantee: this site's own documented magnitude
		// (~2^90-2^91, the U128 comment above) divided by a small 2^(k+1)
		// leaves a true quotient that itself does not fit int64 -- and the
		// prior code narrowed that quotient straight into `raw` via
		// `U128ShrToU64`, which silently drops any bits above position 63
		// rather than detecting them. The pinned witness (branch_code=100,
		// m_a=-2147483647, r_t=2147483649, e_a=2, e_t=-60) is exactly this:
		// a 69-bit true quotient narrows to an in-band, wrong-sign `raw=100`
		// with the saturation counter silent (Poirot 2026-07-28 finding 1).
		// Fixed the same way the negative-k branch below already detects
		// loss: keep the full 128-bit quotient (`U128Shr`, not
		// `U128ShrToU64`) and flag whenever its high word is nonzero -- that
		// is the true "does not fit int64" condition, independent of what
		// `raw`'s narrowed low word happens to wrap to.
		const U128 doubled = U128Add(magnitude, magnitude);
		const U128 rounded = U128Add(doubled, U128OneShl(static_cast<int>(k)));
		const U128 quotient = U128Shr(rounded, static_cast<int>(k) + 1);
		magnitude_exceeds_clamp = (quotient.hi != 0) || quotient.lo > 127;
		magnitude_exceeds_int64 = (quotient.hi != 0) || quotient.lo > kInt64MaxU;
		raw = static_cast<int64_t>(quotient.lo);
	} else {
		// A negative composite exponent is an exact left shift -- no
		// rounding. Detect bit loss by shifting back and comparing to the
		// pre-shift magnitude: a mismatch (or a shift amount that itself
		// reaches or exceeds the 128-bit width, for a nonzero magnitude)
		// means the true, left-shifted magnitude no longer fits the 128-bit
		// carry -- which, at this site's own realistic magnitude (~2^90-2^91,
		// this file's own U128 comment above), is already far past the
		// [-127, 127] clamp before it is even shifted further left. `raw`
		// itself stays the (possibly wrapped) low 64 bits of whatever the
		// shift produced, matching this function's own caller-ensures
		// convention for out-of-domain magnitude (forward_sites.h: "correct
		// whenever the true result fits int64").
		const int shift = static_cast<int>(-k);
		const U128 shifted = U128Shl(magnitude, shift);
		if (shift >= 128) {
			magnitude_exceeds_clamp = (magnitude.lo != 0 || magnitude.hi != 0);
			magnitude_exceeds_int64 = magnitude_exceeds_clamp;
		} else {
			const U128 verify = U128Shr(shifted, shift);
			const bool shift_lost_bits = (verify.lo != magnitude.lo || verify.hi != magnitude.hi);
			magnitude_exceeds_clamp = shift_lost_bits || shifted.hi != 0 || shifted.lo > 127;
			magnitude_exceeds_int64 = shift_lost_bits || shifted.hi != 0 || shifted.lo > kInt64MaxU;
		}
		raw = static_cast<int64_t>(shifted.lo);
	}
	// T-1382 (Poirot, `5af6ab5-t1377-t1378-review-2026-07-31.md`, Significant 3):
	// `raw` may be exactly INT64_MIN here -- reached one call level up on a
	// fully load-legal witness (branch_code=-2, m_a=2^30, r_t=2^32, e_a=62,
	// e_t=0) -- and `-raw` on INT64_MIN is signed-integer-overflow UB, not a
	// wrap the standard promises. Negating in `uint64_t` and casting back is
	// well-defined (C++20 mandates two's-complement conversion, P0907R4) and
	// produces the identical bit pattern a wrapping negation would.
	if (negative) raw = static_cast<int64_t>(0u - static_cast<uint64_t>(raw));

	// T-518 / D-SLM201 option 2, §8.2: the predicated-increment half. The
	// clamp comparison the caller's own `clamp(LandingRescale(...), -127, 127)`
	// performs is evaluated here, once, against this function's own
	// about-to-be-returned raw value, OR-ed with the internal loss detection
	// above -- and increments `*out_saturation_count` by exactly one when
	// either fires. This has no effect whatsoever on `raw`.
	if (out_saturation_count != nullptr && (magnitude_exceeds_clamp || raw < -127 || raw > 127)) {
		*out_saturation_count += 1;
	}
	// T-1377 / D-SLM457: the SECOND, distinct signal -- written exactly once,
	// never accumulated (unlike `out_saturation_count`'s own per-sequence
	// accumulation) -- so `ResidualReconcileSite` can check it per element,
	// immediately, this call alone. Has no effect whatsoever on `raw`.
	if (out_magnitude_exceeded_int64 != nullptr) {
		*out_magnitude_exceeded_int64 = magnitude_exceeds_int64;
	}
	return raw;
}

int64_t ClampRopeCode(int64_t raw) {
	// C33 (§5.3): clamp to the pinned CODE range [-127, 127] -- NOT the int8
	// storage range [-128, 127]; the two differ at exactly the value -128.
	if (raw > 127) return 127;
	if (raw < -127) return -127;
	return raw;
}

SslmForwardStatus RopeApplySite(const int8_t* row, size_t head_dim, int64_t position,
                                 int64_t context_cap, const SslmTensorManifest& rope_tables,
                                 int8_t* out_row) {
	// §6.2 step 3 / §11 S3.3's own gate line (D-SLM376): CheckPositionOverCap
	// is the site's documented FIRST ACT, and no ROP1 tensor is read before
	// it returns. On rejection, `out_row` stays exactly as the caller left
	// it and neither "cos" nor "sin" is touched -- "never a table read".
	const SslmForwardStatus cap_status = CheckPositionOverCap(position, context_cap);
	if (cap_status != SslmForwardStatus::Ok) {
		return cap_status;
	}

	// forward_sites.h's own 5-step contract, step 2: resolve
	// `rope_tables.Tensor("cos")` / `Tensor("sin")`. `rope_tables` is the
	// loaded ROP1 view -- its "cos"/"sin" tensors are the declaration's own
	// stated contract (forward_sites.h: "carrying the 'cos'/'sin' tensors
	// this site reads by row"). Every element already cleared
	// ValidateRopeTablesDomain's |v| <= 2^30 bound at load time
	// (src/model.cpp), which is RopeApplyPair's own safety precondition
	// (intmath.cpp) -- but that bound is over whatever elements the tensor
	// actually carries, and neither the tensor's presence nor its shape is a
	// caller-ensures precondition here: this function is the one that
	// performs the `Tensor("cos")`/`Tensor("sin")` lookup, so no caller can
	// discharge it (Poirot fa3189a review, Critical 1 and Critical 2).
	const SslmTensorView* cos = rope_tables.Tensor("cos");
	const SslmTensorView* sin = rope_tables.Tensor("sin");
	if (cos == nullptr || sin == nullptr) {
		// Critical 1: a ROP1 manifest with zero tensors, or with tensors
		// named anything other than "cos"/"sin", loads Ok -- ParseImpl bounds
		// tensor_count only above kMaxTensors, tensor names are constrained
		// only to non-empty/in-blob/unique, and ValidateRopeTablesDomain
		// walks whatever tensors are present with no name requirement.
		// Tensor() returns nullptr for an absent name (model.h:206-207); a
		// dereference here with no check is a real, reachable null-pointer
		// fault, not a caller-ensures violation.
		return SslmForwardStatus::RopeTableTensorMissing;
	}

	const size_t pairs = head_dim / 2;
	// forward_sites.h's own 5-step contract, step 3: bound `position` and
	// `head_dim / 2` against `cos`/`sin`'s own validated `elem_count`.
	// Critical 2: `context_cap` (already cleared above by
	// CheckPositionOverCap) is a fact the caller supplies about CFG1, and
	// nothing at load time joins it to the ROP1 tensors' own real shape --
	// no cross-section check exists in ValidateSectionValues for ROP1
	// (model.cpp), so `position` and `pairs` are bounded here, directly
	// against `cos`/`sin`'s own validated `elem_count`, before the row is
	// touched. `pairs != 0` is guaranteed by CFG1's own head_dim parity and
	// nonzero checks (ParseConfigImpl, Significant 5's remedy) reaching this
	// call through a real load, but a direct caller (as this suite's own
	// crash-probe and unit cells are) is not assumed to have gone through
	// SslmModel::Load, so the degenerate `pairs == 0` case is rejected here
	// too rather than divided by. This is defense-in-depth against the
	// degenerate zero case only, not a re-derivation of the loader's own
	// parity check: an odd `head_dim >= 3` (`pairs` odd but nonzero) is NOT
	// rejected here, and a direct caller supplying one gets a silent
	// under-write (the trailing unpaired byte of `out_row` is never touched)
	// under `Ok`. Plan Sec6.2 step 3 places the general parity rejection at
	// load time ("head_dim odd is a load-time rejection"), which
	// ParseConfigImpl now performs for every artifact reaching this site
	// through SslmModel::Load (Significant 5's remedy) -- re-deriving that
	// same parity check here would duplicate the loader's own validation at
	// every call site, which this codebase's own doctrine on re-derived
	// predicates (D-SLM81) argues against. `head_dim` parity for a direct,
	// off-Load caller is therefore a caller-ensures precondition this
	// function does not enforce beyond the zero case, matching how
	// `context_cap`'s own load-derived bound (step 1, above) is trusted
	// rather than re-validated here.
	if (pairs == 0 || pairs > cos->elem_count || pairs > sin->elem_count) {
		return SslmForwardStatus::RopeTableExtentExceeded;
	}
	const uint64_t upos = static_cast<uint64_t>(position);
	const uint64_t cos_rows = cos->elem_count / static_cast<uint64_t>(pairs);
	const uint64_t sin_rows = sin->elem_count / static_cast<uint64_t>(pairs);
	if (upos >= cos_rows || upos >= sin_rows) {
		return SslmForwardStatus::RopeTableExtentExceeded;
	}
	const uint64_t row_offset = upos * static_cast<uint64_t>(pairs);

	// forward_sites.h's own 5-step contract, steps 4 and 5, merged into one
	// loop: step 4 reads each pair's "cos"/"sin" row entry inline
	// (ReadRopeTableEntryI64 below) rather than as a separate bulk read,
	// immediately followed by step 5's RopeApplyPair(row[2i], row[2i+1],
	// cos_row[i], sin_row[i]) for that same pair i in [0, head_dim/2) --
	// interleaved even/odd pairing, matching the reference's own
	// _rotate_rows, not a first-half/second-half split -- then ClampRopeCode
	// on each component, written to out_row[2i]/out_row[2i+1].
	for (size_t i = 0; i < pairs; ++i) {
		const int32_t x = static_cast<int32_t>(row[2 * i]);
		const int32_t y = static_cast<int32_t>(row[2 * i + 1]);
		const int32_t cos_q30 = static_cast<int32_t>(ReadRopeTableEntryI64(cos->data, row_offset + i));
		const int32_t sin_q30 = static_cast<int32_t>(ReadRopeTableEntryI64(sin->data, row_offset + i));
		const RopePair rotated = RopeApplyPair(x, y, cos_q30, sin_q30);
		out_row[2 * i] = static_cast<int8_t>(ClampRopeCode(rotated.x));
		out_row[2 * i + 1] = static_cast<int8_t>(ClampRopeCode(rotated.y));
	}

	return SslmForwardStatus::Ok;
}

SslmForwardStatus EmbedEntry(int32_t token_id, int32_t vocab_size, const int8_t* embed_weights,
                              size_t hidden_size, CarriedScale site_constant, int8_t* out_codes,
                              CarriedScale* out_scale, std::string_view site, size_t token_index,
                              SslmTraceHookState* trace_hook_state) {
	// F-S3-8 (§4.8, §6.1): validate token_id against [0, vocab_size) BEFORE any
	// row of embed_weights is read -- a host-supplied id, never sanitized
	// upstream.
	if (token_id < 0 || token_id >= vocab_size) {
		return SslmForwardStatus::TokenIdOutOfRange;
	}
	const int8_t* row = embed_weights + static_cast<size_t>(token_id) * hidden_size;
	std::vector<int64_t> wide(hidden_size);
	for (size_t i = 0; i < hidden_size; ++i) {
		wide[i] = static_cast<int64_t>(row[i]);
	}
	// §11 S3.1a (D-SLM362): forwarded unchanged, same convention as RmsNormSite
	// above -- this entry has exactly one instance in the forward, but the
	// caller still supplies its site string ("embed") rather than this
	// function fixing it, matching RmsNormSite rather than special-casing.
	const ChainResult result = RequantChainChecked(wide.data(), hidden_size,
	                                                std::span<const CarriedScale>{}, site_constant,
	                                                out_codes, out_scale, site, token_index,
	                                                trace_hook_state);
	return result.status;
}

// T-1345: declared by the test-design pass authoring the red suite
// (Claude/Curie/superslm-s3.4-mlp-act-site-test-design-2026-07-29.md),
// following the RoPE application site's own declare-and-stub sequence
// (D-SLM384/385/386). Built by T-1345/T-1346: the real four-step
// composition the header's own doc comment states, driven against the real
// LUT and the real funnel and proven by the site's own feature oracle and
// negative controls (superslm-s3.4-mlp-act-site-body-build-2026-07-29.md).
SslmForwardStatus MlpActSite(const int8_t* gate_code, CarriedScale gate_scale,
                              const int8_t* up_code, CarriedScale up_scale, size_t n,
                              const int32_t* sigmoid_lut_table, CarriedScale site_constant,
                              int8_t* out_codes, CarriedScale* out_scale,
                              std::string_view site, size_t token_index,
                              SslmTraceHookState* trace_hook_state) {
	// C34 (§5.4, §6.3 step 11; T-1345), in the header's stated order and no
	// other. Step 1 is the site's FIRST ACT: the per-token gate scale is
	// derived at runtime and no load-time gate stands behind it (§7.2 second
	// limb), so the domain check precedes the primitive rather than trusting
	// it — the same predicate-before-primitive ordering RopeApplySite's
	// CheckPositionOverCap establishes as this tree's house convention. On
	// rejection nothing is written and SiluSigmoidQ15 never evaluates.
	const SslmForwardStatus domain =
	    CheckSiluCompositionScaleDomain(gate_scale.m, gate_scale.e);
	if (domain != SslmForwardStatus::Ok) return domain;

	// The predicate above has already bounded `e` to its own accepted range
	// (kSiluCompositionRuntimeMinE = -80 at the low end -- the RUNTIME
	// predicate CheckSiluCompositionScaleDomain actually tests against
	// (checked_chain_funnel.cpp), distinct from the load-time descriptor
	// bound kCompositionScaleMinE this comment used to cite; both are -80
	// today, but the static_asserts at checked_chain_funnel.cpp:340-344 exist
	// precisely because the two are allowed to differ, and already do at the
	// upper end: 7 versus 8 -- and an upper branch pinned against
	// kSiluLutTermLeftShiftOverflowExponent, §5.4's executed e = 8 ceiling).
	// SiluSigmoidQ15 takes `int e`, and every value that reaches this
	// narrowing is inside [-80, 8] by that check — so the conversion is exact
	// here BECAUSE of step 1's ordering, not by assumption about the caller.
	const int gate_e = static_cast<int>(gate_scale.e);

	std::vector<int64_t> wide(n);
	for (size_t i = 0; i < n; ++i) {
		// Step 2: C10's fixed-point LUT construction (silu_lut.h), never the
		// i-exp-sigmoid construction F-S3-1 found the reference computing
		// before S3.0's reconciliation. Substituting i-exp-sigmoid here
		// diverges from this construction on real activation codes and, at
		// least once, on the requantized int8 code downstream — executed, and
		// pinned as this slot's own negative control (T-1345).
		const int32_t sig =
		    SiluSigmoidQ15(sigmoid_lut_table, gate_code[i], gate_scale.m, gate_e);
		// Step 3: the product's bound is 127 * 2^15 * 127 < 2^29 (§5.4) — far
		// inside int64, so this is an exact widening product with no
		// narrowing anywhere on the path.
		wide[i] = static_cast<int64_t>(gate_code[i]) * static_cast<int64_t>(sig) *
		          static_cast<int64_t>(up_code[i]);
	}

	// Step 4: BOTH carried scales fold into the funnel's incoming span, gate
	// then up — the reference's own list literal at this site
	// (Tools/superslm_spike/dynamic_engine.py's `mlp_act` chain record passes
	// [gate_scale[t], up_scale[t]]). Neither scale is dropped -- the order is
	// gate then up, matching the reference's own list literal, but the fold
	// itself (carried_scale_product, D-SLM57) is permutation-invariant for a
	// two-element span: mantissa product and exponent sum both commute, so no
	// implementation can diverge on which of the two is listed first (Poirot
	// e4b398c review, Significant 7 -- corrected from a prior claim here that
	// the order was "not free", which does not hold for two operands). What
	// the span DOES pin is that both scales are present; dropping one changes
	// `out_scale`. `site`/`token_index`/`trace_hook_state` are forwarded unchanged
	// (§11 S3.1a, D-SLM362) — this site never fixes its own name, so the
	// caller's own layer-qualified string is exactly what reaches the
	// emission seam.
	const CarriedScale incoming[2] = {gate_scale, up_scale};
	const ChainResult result = RequantChainChecked(
	    wide.data(), n, std::span<const CarriedScale>{incoming, 2}, site_constant, out_codes,
	    out_scale, site, token_index, trace_hook_state);
	return result.status;
}

// T-1347: declared by the test-design pass authoring S3.5's red suite
// (Claude/Curie/superslm-s3.5-residual-and-layer-loop-test-design-
// 2026-07-29.md), following the same declare-and-stub sequence RopeApplySite
// and MlpActSite used (D-SLM384/385/386; T-1345). Built by T-1347/T-1358:
// the real four-step composition the header's own doc comment states.
SslmForwardStatus ResidualReconcileSite(const int8_t* branch_code, CarriedScale branch_scale,
                                          const int8_t* stream_code, CarriedScale stream_scale,
                                          size_t hidden_size, CarriedScale site_constant,
                                          int8_t* out_codes, CarriedScale* out_scale,
                                          std::string_view site, size_t token_index,
                                          SslmTraceHookState* trace_hook_state) {
	// Step 0 (Poirot 76a9776-t1599, Significant 2; the claim below corrected
	// by Poirot 8f63577-t1602, Significant 2, and T-1604): `stream_scale.m`
	// must fit int32_t's own range before it reaches the reciprocal below --
	// `CarriedScaleReciprocal` (checked_chain_funnel.h's own exported door,
	// T-1357/D-SLM433) forwards straight to the funnel's own C19 leaf --
	// deliberately unnamed here, matching this file's own convention below
	// (§7.3's CI source check matches text, not calls) -- whose seed
	// computation is signed-overflow UB once its operand's magnitude exceeds
	// 2281701375 (`INT64_MAX / kC32_2`, intmath.cpp's own reciprocal seed),
	// a bound 2^27 values per sign WIDER than int32_t's own range. This check
	// does not track that wider UB boundary -- it enforces the funnel's own
	// narrower int32 precondition at the door instead, the SAME test as
	// `CarriedScaleMantissaFitsInt32` (checked_chain_funnel.cpp, TU-local),
	// deliberately: `RequantChainChecked`'s own step-0 rejection
	// (`CarriedScaleMantissaOutOfDomain`) runs on `incoming`/`site_constant`
	// only AFTER this call, at step 5 (the exemption at this loop's own
	// guard block below claims the mantissa "reaches only pure arithmetic
	// ... confirmed at source", true of the call but not of the domain it is
	// applied to), so this hoists that SAME funnel precondition to the door
	// instead of leaving it to be enforced two steps later.
	//
	// That hoist is NOT status-neutral on the whole domain it covers.
	// `stream_scale.m` in `(kInt32Max, 2281701375]` per sign -- 2^27 values,
	// the gap between int32's range and the reciprocal's own UB boundary --
	// executed no undefined behaviour before this check existed (verified
	// under this project's own UBSan instrument at interior points) and
	// returned whatever the loop's per-element magnitude predicate
	// (`LandingRescale`'s `out_magnitude_exceeded_int64`, T-1377/D-SLM457) or
	// `RequantChainChecked`'s own step 0 returned, two steps later. Poirot
	// 8f63577-t1602 (Significant 2) swept 1,408 inputs to this exported
	// function and measured 481 that changed returned status because of
	// this check alone; an independently constructed 1,408-input sweep of a
	// different shape (T-1604 build log, `Claude/Brunel/`) reproduces the
	// same class of change (220 of 1,408 on that sweep's own grid). This
	// check is the funnel's own precondition, returned earlier and at the
	// door -- the better place for it to live -- not the no-op the prior
	// version of this comment claimed.
	if (stream_scale.m < static_cast<int64_t>(kInt32Min) ||
	    stream_scale.m > static_cast<int64_t>(kInt32Max)) {
		return SslmForwardStatus::CarriedScaleMantissaOutOfDomain;
	}

	// C26 (§6.2 step 8 / §6.3 step 13), the header's four steps in order.
	//
	// Step 1: C19 over the STREAM's own mantissa, never the branch's -- the
	// branch is what gets rescaled INTO the stream's scale, so the stream's is
	// the target. It is taken through the funnel's own exported door
	// (CarriedScaleReciprocal, checked_chain_funnel.h) rather than the leaf
	// itself: §7.3's check fires on any site that names one of the eight leaves,
	// and this was the first site that had to derive a reciprocal at runtime
	// rather than receive one. T-1357 / D-SLM433 is the ruling, and the door is
	// its resolution. The leaf's own name is deliberately absent from this
	// comment too -- the check matches text, not calls, and a site has no
	// business naming it in either.
	const int64_t r_h = CarriedScaleReciprocal(stream_scale.m);

	std::vector<int64_t> wide(hidden_size);
	// T-1377 / D-SLM457 (§7.2b, §14.14): checked across the WHOLE row before
	// step 4's funnel call runs, on the funnel's own "reject leaves output
	// untouched" convention -- a rejection midway through the loop must not
	// leave `out_codes`/`*out_scale` partially written, and neither is touched
	// by this loop regardless (only the local `wide` buffer is).
	bool any_magnitude_out_of_domain = false;
	for (size_t i = 0; i < hidden_size; ++i) {
		// Step 2: the SAME primitive C27's landing composite uses, at a
		// different call site -- there the reciprocal is the static offline
		// one, here it is derived from the stream at runtime. Unclamped in the
		// saturation-counting sense: T-518's counter belongs to C27's landing
		// site, not to this one, so no counter is passed. The second,
		// distinct out-parameter IS checked here -- this site's own derived-
		// operand predicate (§7.2's second limb, fourth bullet).
		bool magnitude_exceeded = false;
		const int64_t reconciled = LandingRescale(
		    static_cast<int64_t>(branch_code[i]), branch_scale.m, r_h, branch_scale.e,
		    stream_scale.e, /*out_saturation_count=*/nullptr, &magnitude_exceeded);
		if (magnitude_exceeded) {
			// T-1382 (Poirot, `5af6ab5-t1377-t1378-review-2026-07-31.md`,
			// Significant 3): the row is rejected below regardless, so
			// `wide[i]` is never read -- but `reconciled` is a value this
			// call has already declared does not fit int64, and adding to it
			// is not a computation this site is entitled to perform (the
			// prior "no observable effect" comment was true only in the
			// sense that `wide` is discarded, not in the sense that the add
			// itself is well-defined). The loop still runs every index,
			// matching the funnel's own step-5-before-step-6 ordering (the
			// check governs write-out, not loop completion) -- it just does
			// not compute on an element it has already flagged.
			any_magnitude_out_of_domain = true;
			continue;
		}
		// Step 3: the wide add, both operands now nominally at the stream's
		// own scale -- only reached for an element this call has NOT
		// flagged. `wide` is a local buffer never exposed to the caller
		// regardless of which branch runs.
		wide[i] = reconciled + static_cast<int64_t>(stream_code[i]);
	}
	if (any_magnitude_out_of_domain) {
		return SslmForwardStatus::ResidualReconciliationMagnitudeOutOfDomain;
	}

	// Step 4: the funnel's own left-associated fold (D-SLM57). The incoming
	// span is the stream's scale alone -- the branch's scale was already
	// consumed by the rescale in step 2 and folding it again here would double-
	// count it. This call IS the association-order pin at this site: a
	// right-associated fold, or one that pre-combines stream_scale with
	// site_constant outside the funnel, diverges from what the funnel's own
	// proven order produces.
	const CarriedScale incoming[1] = {stream_scale};
	const ChainResult result = RequantChainChecked(
	    wide.data(), hidden_size, std::span<const CarriedScale>{incoming, 1}, site_constant,
	    out_codes, out_scale, site, token_index, trace_hook_state);
	return result.status;
}

// T-1347: same declare-and-stub provenance as ResidualReconcileSite above.
// Built by T-1347/T-1358/T-1375/T-1376: `RunLayerLoop`, below, is the real
// per-layer composition the header's own doc comment states -- every
// rejection path leaves `seq`/`workspace` untouched exactly as the red
// suite's own budget=0 and resume/workspace-poisoning cells require, now
// because the real body honours that contract, not because a stub does.
namespace {

// T-1656/D-SLM642, §5.3: the per-element product-magnitude guard shared by every
// C28 bias-reconciliation insertion in this file (ProjectAndFunnel's q_proj call and
// the k/v landing path below) -- two passes over `out_channels` so a rejection
// anywhere in the row leaves `acc` untouched, matching this file's own "reject leaves
// output untouched" convention (ResidualReconcileSite's own two-pass construction).
// Returns Ok having applied every element's bias into `acc`, or
// BiasReconcileProductOutOfDomain having applied none of them -- `bias` is assumed
// non-null by every caller (the caller checks first).
SslmForwardStatus ApplyBiasReconcileRow(int64_t* acc, size_t out_channels, const int64_t* bias,
                                         int64_t in_scale_m, int64_t in_scale_e) {
	const SslmForwardStatus gate = CheckRoundingDivideByPotExponentDomain(kBiasQFormat, in_scale_e);
	if (gate != SslmForwardStatus::Ok) return gate;
	const int64_t r_a = CarriedScaleReciprocal(in_scale_m);  // loop-invariant, computed once
	bool any_out_of_domain = false;
	for (size_t i = 0; i < out_channels; ++i) {
		if (!BiasReconcileProductFitsInt64(bias[i], r_a)) {
			any_out_of_domain = true;
		}
	}
	if (any_out_of_domain) {
		return SslmForwardStatus::BiasReconcileProductOutOfDomain;
	}
	for (size_t i = 0; i < out_channels; ++i) {
		acc[i] += BiasReconcile(bias[i], kBiasQFormat, r_a, in_scale_e);  // now proven not to overflow
	}
	return SslmForwardStatus::Ok;
}

// One projection: GemmInt8AccumulateRow -> the shared WSC1 fold -> C28's optional bias
// reconciliation -> the funnel. §6.2 step 2's own shape, and q_proj/o_proj/gate_proj/
// up_proj/down_proj all have it -- they differ only in weights, incoming scale, site
// constant, and (T-1656) bias, never in construction. `bias` defaults to nullptr so
// every pre-existing call (o/gate/up/down) compiles unchanged and gets no bias term,
// matching today's behaviour exactly; the q_proj call site passes `lw.q_bias`.
SslmForwardStatus ProjectAndFunnel(const int8_t* in_codes, CarriedScale in_scale,
                                    const int8_t* weight, size_t in_channels, size_t out_channels,
                                    int32_t identity, int32_t mult, int32_t shift,
                                    CarriedScale site_constant, const int64_t* bias,
                                    int8_t* out_codes, CarriedScale* out_scale,
                                    std::string_view site, size_t token_index,
                                    SslmTraceHookState* trace_hook_state) {
	std::vector<int64_t> acc(out_channels);
	GemmInt8AccumulateRow(in_codes, weight, in_channels, out_channels, acc.data());
	for (size_t i = 0; i < out_channels; ++i) {
		acc[i] = ApplyWeightScaleFold(acc[i], identity, mult, shift);
	}
	// T-1656/D-SLM642, §5.3: inserted between the WSC1 fold loop above and the funnel
	// call below -- the exact composition slot the reference's `biased_fold_row`
	// occupies between `_fold_rows` and `_chain_record_vec`.
	if (bias != nullptr) {
		const SslmForwardStatus bias_status =
		    ApplyBiasReconcileRow(acc.data(), out_channels, bias, in_scale.m, in_scale.e);
		if (bias_status != SslmForwardStatus::Ok) return bias_status;
	}
	const CarriedScale incoming[1] = {in_scale};
	const ChainResult result = RequantChainChecked(
	    acc.data(), out_channels, std::span<const CarriedScale>{incoming, 1}, site_constant,
	    out_codes, out_scale, site, token_index, trace_hook_state);
	return result.status;
}

// Per-layer site names ("layer3.attn_norm"), built here because only the loop
// knows which layer it is on; the caller's own outer qualifier is preserved
// ahead of it.
std::string LayerSite(std::string_view site_prefix, uint32_t layer, const char* suffix) {
	std::string s;
	if (!site_prefix.empty()) {
		s.append(site_prefix);
		s.push_back('.');
	}
	s.append("layer");
	s.append(std::to_string(layer));
	s.push_back('.');
	s.append(suffix);
	return s;
}

}  // namespace

namespace {
// S3.7 (§9.4, §11 S3.7 "The K/V store's real layout, and the accessor"): the
// one offset formula every KeyRow/ValueRow accessor below shares --
// per-(layer, head)-major, position-minor.
inline size_t KvHalfOffset(uint32_t layer, int64_t context_cap, size_t num_kv_heads,
                            size_t head_dim) {
	return (static_cast<size_t>(layer) * static_cast<size_t>(context_cap) * num_kv_heads *
	        head_dim) * 2u;
}
inline size_t KvRowOffsetWithinHalf(int64_t context_cap, size_t head_dim, size_t kv_head,
                                     int64_t position) {
	return static_cast<size_t>(kv_head) * static_cast<size_t>(context_cap) * head_dim +
	       static_cast<size_t>(position) * head_dim;
}
}  // namespace

const int8_t* KeyRow(const uint8_t* workspace, uint32_t layer, int64_t context_cap,
                      size_t num_kv_heads, size_t head_dim, size_t kv_head,
                      int64_t position) noexcept {
	const int8_t* const kv_base = reinterpret_cast<const int8_t*>(workspace);
	const int8_t* const k_store = kv_base + KvHalfOffset(layer, context_cap, num_kv_heads, head_dim);
	return k_store + KvRowOffsetWithinHalf(context_cap, head_dim, kv_head, position);
}

const int8_t* ValueRow(const uint8_t* workspace, uint32_t layer, int64_t context_cap,
                        size_t num_kv_heads, size_t head_dim, size_t kv_head,
                        int64_t position) noexcept {
	const int8_t* const kv_base = reinterpret_cast<const int8_t*>(workspace);
	const int8_t* const k_store = kv_base + KvHalfOffset(layer, context_cap, num_kv_heads, head_dim);
	const int8_t* const v_store =
	    k_store + static_cast<size_t>(context_cap) * num_kv_heads * head_dim;
	return v_store + KvRowOffsetWithinHalf(context_cap, head_dim, kv_head, position);
}

int8_t* MutableKeyRow(uint8_t* workspace, uint32_t layer, int64_t context_cap,
                       size_t num_kv_heads, size_t head_dim, size_t kv_head,
                       int64_t position) noexcept {
	return const_cast<int8_t*>(
	    KeyRow(workspace, layer, context_cap, num_kv_heads, head_dim, kv_head, position));
}

int8_t* MutableValueRow(uint8_t* workspace, uint32_t layer, int64_t context_cap,
                         size_t num_kv_heads, size_t head_dim, size_t kv_head,
                         int64_t position) noexcept {
	return const_cast<int8_t*>(
	    ValueRow(workspace, layer, context_cap, num_kv_heads, head_dim, kv_head, position));
}

SslmForwardStatus RunLayerLoop(SequenceLayerState& seq, const LayerWeights* layers,
                                 uint32_t num_hidden_layers, uint32_t layer_budget,
                                 size_t hidden_size, size_t head_dim, size_t num_key_value_heads,
                                 size_t intermediate_size, int64_t context_cap,
                                 const SslmTensorManifest& rope_tables, uint8_t* workspace,
                                 size_t workspace_size, std::string_view site_prefix,
                                 size_t token_index, SslmTraceHookState* trace_hook_state) {
	// §9.3's first decided contract, checked BEFORE anything is read or
	// written: a budget of 0 consumes a call, advances nothing, and would
	// return "pending" -- a host-visible livelock. `seq` is left bit-identical,
	// which is exactly what the cell poison-fills to assert.
	if (layer_budget == 0) return SslmForwardStatus::InvalidLayerBudget;

	// `context_cap` sizes the K/V workspace below; zero or negative is
	// invalid for the identical reason (neither is a legitimate count of
	// cache positions), so one check covers both rather than two separate
	// guards for the same domain requirement. Checked before the size
	// product is formed at all: a zero `context_cap` would otherwise zero
	// that product and clear the workspace-size guard entirely, and a
	// negative one would wrap the same `static_cast<size_t>` product mod
	// 2^64 to the same effect (Poirot e4b398c review, Critical 2/3).
	if (context_cap < 1) return SslmForwardStatus::InvalidContextCap;

	// §9.4: the K/V store is caller-supplied memory "sized from
	// config.context_cap, num_key_value_heads, head_dim, and kv_precision".
	// That is what `workspace` is here and the only thing this loop takes from
	// it -- per-site scratch heaps through std::vector, exactly as RmsNormSite
	// and MlpActSite already do in this same translation unit. The required
	// size is therefore computed from the parameters rather than chosen here.
	//
	// Significant 1 (Poirot e4b398c review): this is a fact about the CFG1
	// geometry the caller supplied (`hidden_size` is not an exact multiple of
	// `head_dim`, or `head_dim == 0`), never a fact about `workspace` --
	// `WorkspaceTooSmall` used to be returned here, which sends a host that
	// enlarges its buffer into an infinite retry against a size no buffer
	// satisfies, because the guard below it is never reached.
	const size_t num_heads = head_dim == 0 ? 0 : hidden_size / head_dim;
	if (num_heads == 0 || num_heads * head_dim != hidden_size) {
		return SslmForwardStatus::HeadDimGeometryMismatch;
	}
	// T-1654 (S3.8a): `num_key_value_heads` gets the identical treatment as
	// `num_heads` above, for the identical reason -- a caller-suppliable
	// scalar is untrusted at this function's own boundary regardless of what
	// any other caller (the loader's `ValidateConfigGeometryJoin`) already
	// checked. Checked immediately after `HeadDimGeometryMismatch`, on the
	// query head count that check has just proven in-domain, and before
	// `kv_bytes_needed` is formed below (design record §4).
	if (num_key_value_heads == 0 || num_key_value_heads > num_heads ||
	    num_heads % num_key_value_heads != 0) {
		return SslmForwardStatus::KvHeadGeometryMismatch;
	}
	// Loop-invariant across every layer and every token, exactly like
	// `num_heads` itself. Integer division is exact by construction: the
	// guard immediately above has just proven `num_heads % num_key_value_heads
	// == 0`.
	const size_t group = num_heads / num_key_value_heads;
	// The size product itself, overflow-guarded factor by factor (the same
	// `product > SIZE_MAX / factor` idiom model.cpp's tensor-shape check
	// already uses) -- `context_cap` is now known positive, but the product
	// of four caller-supplied dimensions can still overflow size_t for a
	// sufficiently large one, and an overflowed product would silently
	// under-size the same guard C2/C3 exist to keep sound.
	//
	// T-1654 (S3.8a): the third factor is `num_key_value_heads`, not
	// `num_heads` -- the K/V store holds one row per KV head per position
	// per layer per half (K or V), not one row per query head (§9.4's own
	// design intent, matching `KeyRow`/`ValueRow`'s own already-correct
	// addressing). At `num_key_value_heads == num_heads` (every existing
	// fixture), the two factors are the same number, so this substitution
	// changes nothing the pre-existing suite already computes.
	size_t kv_bytes_needed = static_cast<size_t>(num_hidden_layers);
	const size_t kv_factors[] = {static_cast<size_t>(context_cap), num_key_value_heads,
	                             head_dim, 2u};
	for (size_t factor : kv_factors) {
		if (factor != 0 && kv_bytes_needed > SIZE_MAX / factor) {
			return SslmForwardStatus::InvalidContextCap;
		}
		kv_bytes_needed *= factor;
	}
	if (workspace == nullptr) return SslmForwardStatus::WorkspaceTooSmall;
	if (workspace_size < kv_bytes_needed) return SslmForwardStatus::WorkspaceTooSmall;

	// T-1590 (Poirot cd2e75a review, Critical 1): every caller-settable field
	// of `seq` is enumerated and validated here, before any of them is read
	// or written -- the same "addressable as a unit" invitation (§13 dim 9)
	// that motivated the KvCapacityExhausted guard below to run on every
	// call, applied to the whole struct rather than to `context_length`
	// alone (that guard's own remedy closed one side of `context_length`'s
	// domain and left `hidden_codes` unprobed; both are closed here). Per
	// field:
	//
	//   - `hidden_codes`: a caller-owned pointer with no length carried in
	//     the struct (`hidden_size` is this call's own parameter, never a
	//     struct member), so a too-short buffer is not a domain any check
	//     here can detect -- the same unavoidable gap as any C API taking a
	//     pointer and a separate length. `nullptr` specifically IS
	//     detectable, is exactly what the struct's own default member
	//     initializer produces, and used to be dereferenced unconditionally
	//     at RmsNormSite's very first read of it a few lines below --
	//     rejected here instead of left to crash the process.
	//   - `layer_index`: already fully bounded by the very next check below
	//     (`>= num_hidden_layers`) -- unsigned, so "negative" is not a
	//     domain it can occupy, and every later use is as an array index the
	//     `while` loop's own condition re-proves `< num_hidden_layers` on
	//     every iteration. No further guard needed.
	//   - `kv_saturation_count`: a monotonically-incremented `uint64_t`
	//     diagnostic counter (T-518 / D-SLM201 option 2) -- never read back
	//     as a size, offset, or index anywhere in this loop or its callees,
	//     so an arbitrary restored value, including one that wraps to 0 on
	//     its very next increment, changes no memory this call touches.
	//     Probed, not merely reasoned: T-1590's kv_saturation_count cell
	//     restores it at UINT64_MAX and confirms the call's outcome and the
	//     workspace are identical to the same call at 0. No guard needed.
	//   - `context_length`: validated `>= 0` immediately below, beside the
	//     pre-existing `>= context_cap` check it sits next to -- the exact
	//     remedy this ticket's own review prescribed.
	//   - `hidden_scale`: reaches only pure arithmetic
	//     (`CarriedScaleReciprocal`, `LandingRescale`), never a size, count,
	//     or offset. `ResidualReconcileSite`'s own Step 0 (Poirot
	//     76a9776-t1599, Significant 2) rejects an out-of-int32 MANTISSA
	//     (`CarriedScaleMantissaOutOfDomain`) BEFORE the reciprocal call --
	//     not `RequantChainChecked`'s step 0, which runs two steps later and
	//     was reached only after the reciprocal had already executed
	//     signed-overflow UB on the out-of-domain mantissa (measured;
	//     `intmath.cpp:224`, the funnel's own C19 leaf's seed computation --
	//     deliberately unnamed here, matching this file's own convention).
	//     The EXPONENT half carries no analogous rejection -- an extreme `e`
	//     is never rejected, it is COMPUTED THROUGH. What changed (T-1596,
	//     closing three prior rounds' own open finding at this exact site,
	//     Poirot 8f63577-t1602 Significant 1) is that the arithmetic it is
	//     computed through no longer executes UB doing so: `LandingRescale`'s
	//     own composed exponent (`ComposedExponent`, this file's anonymous
	//     namespace, beside the `U128` facility) and `CombineCarriedScale`'s
	//     own exponent fold (`checked_chain_funnel.cpp`, `SaturatingAdd64`)
	//     both compute with unsigned-wraparound overflow detection and
	//     saturate rather than overflow -- closing not only the subtraction
	//     this file's own `62 - (e_a - e_t)` used to execute directly, but a
	//     SECOND site the same witness reaches one call later
	//     (`CombineCarriedScale`'s `a.e + b.e + 31`, in `RequantChainChecked`'s
	//     own fold), found by execution when closing the first let the
	//     project's hard-abort ASan+UBSan instrument run far enough into the
	//     same call to reach it. Probed, not merely reasoned: T-1590's
	//     hidden_scale cell runs TWO sub-cases -- an out-of-domain mantissa
	//     `m` (rejected before the reciprocal runs), and, per T-1596, the
	//     field's own TRUE extreme exponent, `e = INT64_MAX` (the value this
	//     tree has always documented as carrying no domain check anywhere,
	//     run through to whatever status it actually returns -- `Ok`,
	//     measured, and the committed `seq.hidden_scale.e` this call produces
	//     pinned to its own deterministic saturated value) -- against
	//     workspace and `hidden_codes` buffers each ringed with a canary
	//     immediately before AND after their declared region, confirming all
	//     four canaries untouched regardless of which status either sub-case
	//     returns, and confirmed under this project's own MSVC-ABI
	//     ASan+UBSan instrument to produce zero UBSan reports. No REJECTING
	//     guard exists for the exponent, and none is needed: the arithmetic
	//     it reaches is itself now defined over the field's whole `int64_t`
	//     domain.
	//     How that claim was established, and what would reopen it: the two
	//     sites named above (`ComposedExponent`, `SaturatingAdd64`) are the
	//     two the project's own hard-abort ASan+UBSan instrument reported,
	//     one at a time -- closing the first and re-running let the same
	//     witness reach the second (T-1596). That witness drives one call
	//     path through `ResidualReconcileSite` -> `RequantChainChecked`;
	//     `-fno-sanitize-recover=all` reports the first UB site it reaches
	//     on a path and stops there, so "no reports remain" means no
	//     reports remain ON THE PATHS THIS WITNESS DRIVES, not that the
	//     reachable call graph's membership has been enumerated or is
	//     closed. Nothing in this tree pins these two sites as the whole of
	//     the exponent-arithmetic call graph's membership: a third
	//     exponent-composition site added later anywhere under this same
	//     call graph would make this comment silently false, with no build
	//     failure, until someone re-runs the instrument and finds it.
	if (seq.hidden_codes == nullptr) return SslmForwardStatus::InvalidHiddenCodes;

	// Significant 6 (Poirot e4b398c review): the SAME livelock the
	// `layer_budget == 0` check above exists to reject also arises when
	// there is nothing left to advance through for a REASON OTHER than the
	// budget -- `seq.layer_index` already at `num_hidden_layers` (this
	// token's sequence already ran to completion), or `num_hidden_layers ==
	// 0` (which makes `layer_index >= num_hidden_layers` true trivially at
	// `layer_index == 0`, so one check covers both). Without this, the
	// `while` below never enters its body and falls through to `return Ok`
	// at the bottom having advanced nothing -- indistinguishable, from the
	// return value alone, from "advanced N layers." §9.3's
	// reject-over-silently-degrade law does not depend on which of the two
	// reasons produced zero progress. Checked after every domain guard
	// above rather than before them: an invalid `context_cap` or geometry is
	// still the more specific rejection when both are true of the same call,
	// and every guard above it leaves `seq` untouched on its own rejection,
	// so ordering relative to them changes no cell's observable contract.
	if (seq.layer_index >= num_hidden_layers) return SslmForwardStatus::SequenceAlreadyComplete;

	// S3.7 (§11 S3.7 "Fail fast on a full cache"): a full cache is rejected
	// before any layer runs, `seq` left untouched. Checked on EVERY call,
	// not only at `layer_index == 0`: `context_length` is stable across a
	// resumed mid-token call only for states this loop itself produced, and
	// this check runs against `seq` as the caller hands it in, which is not
	// restricted to that provenance -- `SequenceLayerState` documents only
	// `0 <= layer_index <= num_hidden_layers`, no joint constraint against
	// `context_length >= context_cap`, and §13 dim 9 pins the struct as
	// "addressable as a unit", an explicit invitation for a caller to save
	// and restore one. A caller-restored state at a non-zero `layer_index`
	// with a full cache used to reach the landing write below -- past the
	// deeper `RopeApplySite`/`CheckPositionOverCap` check that would
	// otherwise catch it -- and write through `MutableKeyRow`/
	// `MutableValueRow` up to `head_dim` bytes past a `workspace_size` the
	// call had already validated as sufficient (Poirot 0d64462 review,
	// Critical 1). Re-checking every call costs nothing extra on the
	// provenance this comment used to rely on: for a state this loop itself
	// produced, `context_length` did not change since the last check, so
	// the comparison repeats the same true/false it already computed.
	//
	// T-1590 (Poirot cd2e75a review, Critical 1): the identical provenance
	// argument above bounds `context_length` on the upper side only.
	// `SequenceLayerState` states no domain at all for `context_length`, and
	// a caller-restored state with a NEGATIVE one used to reach
	// `KvRowOffsetWithinHalf`'s unguarded `static_cast<size_t>` (this file,
	// the S3.7 accessor block above `RunLayerLoop`), which turns a negative
	// `position` into a near-`SIZE_MAX` quantity and lands the K/V write
	// `head_dim * |context_length|` bytes BELOW the workspace base -- after
	// this call's own size check above had already certified the workspace
	// sufficient. `CheckPositionOverCap` already rejects `position < 0` with
	// exactly this status; this hoists that existing rejection ahead of the
	// write instead of inventing a new one, so no caller-observable status
	// changes for any input this guard was not already going to reject.
	if (seq.context_length < 0) return SslmForwardStatus::PositionOverCap;
	if (seq.context_length >= context_cap) {
		return SslmForwardStatus::KvCapacityExhausted;
	}

	// S3.7 (§11 S3.7 "The mechanism"): the current token attends to every
	// already-committed position (`seq.context_length` of them) plus its own
	// just-landed K/V at `position` -- ordinary causal self-attention, never
	// 0 by this construction (`context_length >= 0`, so `width >= 1`; the
	// minimum is `context_length == 0` on a sequence's first token, giving
	// `width == 1`, the exact case S3a already builds and gates). Constant
	// across every layer of this token, since `context_length` only advances
	// at the token's last layer (§13 dim 8's own resume/width-stability
	// cell).
	const int64_t position = seq.context_length;
	const size_t width = static_cast<size_t>(seq.context_length) + 1;

	std::vector<int8_t> normed(hidden_size), q_codes(hidden_size), o_codes(hidden_size);
	std::vector<int8_t> q_rot(hidden_size), k_rot(hidden_size), ctx_codes(hidden_size);
	std::vector<int8_t> gate_codes(intermediate_size), up_codes(intermediate_size);
	std::vector<int8_t> act_codes(intermediate_size), down_codes(hidden_size);
	std::vector<int8_t> stream_next(hidden_size);
	// §9.3/Critical 4: the attention-residual output is staged here rather
	// than committed into `seq.hidden_codes` mid-layer -- the MLP half reads
	// it as its own input, and it is committed into `seq` only once, together
	// with the MLP residual and the resume marker, at the bottom of the loop
	// body. This is the one extra hidden-size buffer atomicity costs.
	std::vector<int8_t> attn_stream(hidden_size);

	uint32_t advanced = 0;
	while (advanced < layer_budget && seq.layer_index < num_hidden_layers) {
		const uint32_t l = seq.layer_index;
		const LayerWeights& lw = layers[l];
		CarriedScale normed_scale{}, q_scale{}, ctx_scale{}, o_scale{};
		CarriedScale mlp_normed_scale{}, gate_scale{}, up_scale{}, act_scale{}, down_scale{};
		CarriedScale stream_scale{}, attn_stream_scale{};
		SslmForwardStatus st;

		// --- attention half (§6.2) --------------------------------------------
		st = RmsNormSite(seq.hidden_codes, lw.attn_norm_gain, hidden_size, seq.hidden_scale,
		                 lw.attn_norm_site_constant, normed.data(), &normed_scale,
		                 LayerSite(site_prefix, l, "attn_norm"), token_index, trace_hook_state);
		if (st != SslmForwardStatus::Ok) return st;

		st = ProjectAndFunnel(normed.data(), normed_scale, lw.q_weight, hidden_size, hidden_size,
		                      lw.proj_identity, lw.proj_mult, lw.proj_shift, lw.q_site_constant,
		                      lw.q_bias, q_codes.data(), &q_scale,
		                      LayerSite(site_prefix, l, "q_proj.requant"),
		                      token_index, trace_hook_state);
		if (st != SslmForwardStatus::Ok) return st;

		// k_proj / v_proj do NOT funnel: they land at the static per-head scale
		// through LandingRescale (§8.1), writing straight into the K/V store,
		// through the S3.7 accessor -- one row per head, at THIS token's own
		// `position`, never a whole-hidden_size flat write (§9.4's real
		// per-(layer, head)-major, position-minor layout).
		{
			// T-1654 (S3.8a): the K/V landing write's loop bound narrows from
			// `num_heads` to `num_key_value_heads` -- this is a resize, not a
			// substitution, because `kacc`/`vacc` only have `kv_hidden_size`
			// elements once the GEMM calls below are corrected; there is no `h`
			// beyond `num_key_value_heads` to index.
			const size_t kv_hidden_size = num_key_value_heads * head_dim;
			std::vector<int64_t> kacc(kv_hidden_size), vacc(kv_hidden_size);
			GemmInt8AccumulateRow(normed.data(), lw.k_weight, hidden_size, kv_hidden_size,
			                      kacc.data());
			GemmInt8AccumulateRow(normed.data(), lw.v_weight, hidden_size, kv_hidden_size,
			                      vacc.data());
			for (size_t i = 0; i < kv_hidden_size; ++i) {
				kacc[i] = ApplyWeightScaleFold(kacc[i], lw.proj_identity, lw.proj_mult, lw.proj_shift);
				vacc[i] = ApplyWeightScaleFold(vacc[i], lw.proj_identity, lw.proj_mult, lw.proj_shift);
			}
			// T-1656/D-SLM642, §5.3: the identical bias insertion ProjectAndFunnel's
			// q_proj call site carries, written a second time here -- a separate
			// location that does not call ProjectAndFunnel, keyed on
			// `normed_scale.e`/`normed_scale.m` rather than `in_scale.e`/`in_scale.m`.
			// Applied to the whole folded row before any element lands, so a
			// rejection here leaves `workspace`/`seq` untouched (kacc/vacc are local
			// temporaries; no landing write has happened yet).
			if (lw.k_bias != nullptr) {
				const SslmForwardStatus bias_status = ApplyBiasReconcileRow(
				    kacc.data(), kv_hidden_size, lw.k_bias, normed_scale.m, normed_scale.e);
				if (bias_status != SslmForwardStatus::Ok) return bias_status;
			}
			if (lw.v_bias != nullptr) {
				const SslmForwardStatus bias_status = ApplyBiasReconcileRow(
				    vacc.data(), kv_hidden_size, lw.v_bias, normed_scale.m, normed_scale.e);
				if (bias_status != SslmForwardStatus::Ok) return bias_status;
			}
			for (size_t h = 0; h < num_key_value_heads; ++h) {
				int8_t* const k_row = MutableKeyRow(workspace, l, context_cap,
				                                    num_key_value_heads, head_dim, h, position);
				int8_t* const v_row = MutableValueRow(workspace, l, context_cap,
				                                      num_key_value_heads, head_dim, h, position);
				for (size_t d = 0; d < head_dim; ++d) {
					const size_t i = h * head_dim + d;
					// §8.1's clamp is this call site's own (LandingRescale's
					// header states the clamp is the caller's); reuses
					// ClampRopeCode for the pinned [-127, 127] code range it
					// already implements. T-518's saturation counter (§8.2) is
					// wired into seq's own per-sequence accumulator, the one
					// call in this tree that composes the landing.
					k_row[d] = static_cast<int8_t>(ClampRopeCode(LandingRescale(
					    kacc[i], normed_scale.m, lw.kv_landing_r_t_k[h], normed_scale.e,
					    lw.kv_landing_e_t_k[h], &seq.kv_saturation_count)));
					v_row[d] = static_cast<int8_t>(ClampRopeCode(LandingRescale(
					    vacc[i], normed_scale.m, lw.kv_landing_r_t_v[h], normed_scale.e,
					    lw.kv_landing_e_t_v[h], &seq.kv_saturation_count)));
				}
			}
		}

		// RoPE on q and on the just-landed k, per head (§6.2 step 3). k is
		// read/written through the S3.7 accessor at THIS token's own
		// `position` -- never a flat `hidden_size` offset, which under the
		// real per-head layout would read/write the wrong head once
		// `context_cap > 1`.
		for (size_t h = 0; h < num_heads; ++h) {
			st = RopeApplySite(q_codes.data() + h * head_dim, head_dim, position, context_cap,
			                   rope_tables, q_rot.data() + h * head_dim);
			if (st != SslmForwardStatus::Ok) return st;
			// T-1654 (S3.8a): the accessor index is `h / group`, not `h` -- the
			// reference's own grouping (`dynamic_engine.py:410`,
			// `kv_head = head // group`). This loop's own bound stays `num_heads`
			// (`q_rot` is query-head-sized); only the K accessor's index and its
			// own size argument change.
			const size_t kv_head = h / group;
			const int8_t* const k_row_before_rotate =
			    KeyRow(workspace, l, context_cap, num_key_value_heads, head_dim, kv_head, position);
			st = RopeApplySite(k_row_before_rotate, head_dim, position, context_cap, rope_tables,
			                   k_rot.data() + h * head_dim);
			if (st != SslmForwardStatus::Ok) return st;
		}
		// S3.7 (§11 S3.7 "The mechanism", the RoPE write-back correction): each
		// head's row is written back individually, through `MutableKeyRow` at
		// THIS token's own `position` -- the old `for (i<hidden_size)
		// k_store[i] = k_rot[i]` copy was only valid at `context_cap == 1`
		// (the whole per-token K region WAS exactly `hidden_size` contiguous
		// bytes at position 0); under the real layout it would overwrite
		// EVERY committed position's row with this token's own rotation.
		for (size_t h = 0; h < num_heads; ++h) {
			// T-1654 (S3.8a): `h / group`, matching the read loop above -- the
			// same KV row is written once per query head sharing it (redundant
			// but sound, design record §6.2: every read above happens before
			// any write here, so no partially-rotated store is ever observed).
			const size_t kv_head = h / group;
			int8_t* const k_row = MutableKeyRow(workspace, l, context_cap, num_key_value_heads,
			                                    head_dim, kv_head, position);
			for (size_t d = 0; d < head_dim; ++d) k_row[d] = k_rot[h * head_dim + d];
		}

		// Attention proper (§6.2 step 5). No named site for this composition
		// exists anywhere in this tree; this is where it is first composed.
		{
			// T-1655/D-SLM620, §4.5: C30's per-query i-exp derivation, once per
			// distinct kv_head (never once per layer, and never once per query
			// head) -- q_scale (the q_proj carried scale, in scope from this
			// layer's own ProjectAndFunnel call above) is per-token and shared
			// across every head; only softmax_khead[kv_head] varies by KV head.
			// Query heads sharing one kv_head are visited in non-decreasing
			// order (`h / group` for `h = 0..num_heads-1`), so deriving on first
			// sight of each kv_head costs at most num_key_value_heads
			// derivations per token per layer, never num_heads.
			std::vector<int64_t> khead_q_ln2(num_key_value_heads), khead_q_b(num_key_value_heads),
			    khead_q_c(num_key_value_heads);
			std::vector<bool> khead_derived(num_key_value_heads, false);
			std::vector<int64_t> ctx_wide(hidden_size);
			for (size_t h = 0; h < num_heads; ++h) {
				std::vector<int64_t> scores(width), probs(width), ctx_acc(head_dim);
				// T-1654 (S3.8a): `h / group`, the reference's own grouping
				// (`dynamic_engine.py:410-419`) -- this loop's own bound stays
				// `num_heads` (`ctx_wide`/`ctx_fold_identity` are query-head-sized).
				const size_t kv_head = h / group;

				if (!khead_derived[kv_head]) {
					// §4.5 step 2a: both operands' mantissas checked against
					// CombineCarriedScale's own precondition before the combine
					// runs (ac34677 S5's convention, applied at this new call
					// site).
					const int64_t sm_khead_m = lw.iexp_softmax_khead_m[kv_head];
					const int64_t sm_khead_e = lw.iexp_softmax_khead_e[kv_head];
					const bool q_scale_in_domain =
					    q_scale.m >= static_cast<int64_t>(kInt32Min) &&
					    q_scale.m <= static_cast<int64_t>(kInt32Max);
					const bool khead_in_domain =
					    sm_khead_m >= static_cast<int64_t>(kInt32Min) &&
					    sm_khead_m <= static_cast<int64_t>(kInt32Max);
					if (!q_scale_in_domain || !khead_in_domain) {
						return SslmForwardStatus::CarriedScaleMantissaOutOfDomain;
					}
					// §4.5 step 2b: carried_scale_product([q_scale, softmax_khead]),
					// through the funnel's own exposed combine door -- then the
					// post-combine check RequantChainChecked's own fold already
					// applies, at this new call site.
					const CarriedScale sm =
					    CombineCarriedScale(q_scale, CarriedScale{sm_khead_m, sm_khead_e});
					if (sm.m < static_cast<int64_t>(kInt32Min) ||
					    sm.m > static_cast<int64_t>(kInt32Max)) {
						return SslmForwardStatus::CarriedScaleMantissaOutOfDomain;
					}
					// §4.5 step 2c: C30's derivation itself. Any outcome other
					// than kOk means no triple exists to hand to SoftmaxRowQ15
					// at all -- caught here, before SoftmaxRowQ15 (or the width
					// gate below) is ever called for this kv_head.
					int64_t derived_q_ln2 = 0, derived_q_b = 0, derived_q_c = 0;
					const IExpScaleDomain scale_domain = IExpScaleConstants(
					    sm.m, sm.e, kIExpLn2Q, 30, kIExpBQ, 30, kIExpCaQ, 30, &derived_q_ln2,
					    &derived_q_b, &derived_q_c);
					if (scale_domain != IExpScaleDomain::kOk) {
						return SslmForwardStatus::IExpScaleDerivationOutOfDomain;
					}
					khead_q_ln2[kv_head] = derived_q_ln2;
					khead_q_b[kv_head] = derived_q_b;
					khead_q_c[kv_head] = derived_q_c;
					khead_derived[kv_head] = true;
				}

				// §4.5 step 3: CheckSoftmaxRowWidthDomain moves from once-per-
				// layer to once-per-distinct-kv_head, still hoisted above the
				// per-head kernel call for that kv_head -- the same gate-
				// before-kernel ordering the prior once-per-layer call already
				// established (Minor B, Poirot e4b398c review), now scoped to
				// the per-kv_head triple this design derives.
				st = CheckSoftmaxRowWidthDomain(khead_q_b[kv_head], khead_q_c[kv_head], width);
				if (st != SslmForwardStatus::Ok) return st;

				// S3.7: the score row reads `width` contiguous rows of this
				// KV head's own K store, starting at position 0 (the store's
				// position-minor layout makes positions 0..width-1 for one
				// head contiguous) -- q·K, never q·V (D-SLM516/D-SLM503's A3
				// mutant, which this closes the K side of by construction:
				// reading the wrong store here is exactly what a k_weight
				// mutation cell (§3 Cell 1) would fail to distinguish).
				const int8_t* const k_rows_base =
				    KeyRow(workspace, l, context_cap, num_key_value_heads, head_dim, kv_head, 0);
				GemmInt8AccumulateRow(q_rot.data() + h * head_dim, k_rows_base, head_dim, width,
				                      scores.data());
				if (!SoftmaxRowQ15(scores.data(), width, khead_q_ln2[kv_head], khead_q_b[kv_head],
				                   khead_q_c[kv_head], probs.data())) {
					// Minor A (Poirot e4b398c review): the kernel refused after
					// its own gate already accepted -- a distinct outcome from
					// the gate's own rejection, now named rather than reported
					// as the gate's own status (which used to send a host
					// debugging `SoftmaxRowWidthOutOfDomain` to inspect a width
					// already in domain).
					return SslmForwardStatus::SoftmaxKernelRefusedAfterGateAccepted;
				}
				const int8_t* const v_rows_base =
				    ValueRow(workspace, l, context_cap, num_key_value_heads, head_dim, kv_head, 0);
				GemmProbQ15Accumulate(probs.data(), v_rows_base, width, head_dim,
				                      ctx_acc.data());
				for (size_t d = 0; d < head_dim; ++d) {
					// D-SLM57's per-head dispatch (§6.2 step 6): WSC1's
					// `layer{L}.ctx_fold` row for THIS head, not one triple
					// shared across every head.
					ctx_wide[h * head_dim + d] = ApplyWeightScaleFold(
					    ctx_acc[d], lw.ctx_fold_identity[h], lw.ctx_fold_mult[h],
					    lw.ctx_fold_shift[h]);
				}
			}
			// §6.2 step 6: the context funnel takes an EMPTY incoming span --
			// the per-head static scale was already consumed at the landing.
			const ChainResult ctx_result = RequantChainChecked(
			    ctx_wide.data(), hidden_size, std::span<const CarriedScale>{},
			    lw.ctx_fold_site_constant, ctx_codes.data(), &ctx_scale,
			    LayerSite(site_prefix, l, "attn_ctx"), token_index, trace_hook_state);
			if (ctx_result.status != SslmForwardStatus::Ok) return ctx_result.status;
		}

		st = ProjectAndFunnel(ctx_codes.data(), ctx_scale, lw.o_weight, hidden_size, hidden_size,
		                      lw.proj_identity, lw.proj_mult, lw.proj_shift, lw.o_site_constant,
		                      /*bias=*/nullptr, o_codes.data(), &o_scale,
		                      LayerSite(site_prefix, l, "o_proj.requant"),
		                      token_index, trace_hook_state);
		if (st != SslmForwardStatus::Ok) return st;

		// §9.3/Critical 4: staged into `attn_stream`, NOT committed into
		// `seq.hidden_codes` here. Committing mid-layer would leave `seq`
		// carrying this layer's attention residual while `layer_index` still
		// names the layer as not-yet-run; a rejection anywhere in the MLP
		// half below would then leave that inconsistent state visible to a
		// resume, which §9.3's "the ONLY resume point is a layer boundary"
		// forbids.
		st = ResidualReconcileSite(o_codes.data(), o_scale, seq.hidden_codes, seq.hidden_scale,
		                           hidden_size, lw.attn_residual_site_constant, attn_stream.data(),
		                           &attn_stream_scale, LayerSite(site_prefix, l, "attn_residual"),
		                           token_index, trace_hook_state);
		if (st != SslmForwardStatus::Ok) return st;

		// --- MLP half (§6.3) ---------------------------------------------------
		// Reads the STAGED attention-residual output, not `seq`, which is
		// still the layer's pre-attention state until the commit at the
		// bottom of this loop body.
		st = RmsNormSite(attn_stream.data(), lw.mlp_norm_gain, hidden_size, attn_stream_scale,
		                 lw.mlp_norm_site_constant, normed.data(), &mlp_normed_scale,
		                 LayerSite(site_prefix, l, "mlp_norm"), token_index, trace_hook_state);
		if (st != SslmForwardStatus::Ok) return st;

		st = ProjectAndFunnel(normed.data(), mlp_normed_scale, lw.gate_weight, hidden_size,
		                      intermediate_size, lw.proj_identity, lw.proj_mult, lw.proj_shift,
		                      lw.gate_site_constant, /*bias=*/nullptr, gate_codes.data(), &gate_scale,
		                      LayerSite(site_prefix, l, "gate_proj.requant"), token_index,
		                      trace_hook_state);
		if (st != SslmForwardStatus::Ok) return st;
		st = ProjectAndFunnel(normed.data(), mlp_normed_scale, lw.up_weight, hidden_size,
		                      intermediate_size, lw.proj_identity, lw.proj_mult, lw.proj_shift,
		                      lw.up_site_constant, /*bias=*/nullptr, up_codes.data(), &up_scale,
		                      LayerSite(site_prefix, l, "up_proj.requant"), token_index,
		                      trace_hook_state);
		if (st != SslmForwardStatus::Ok) return st;

		// Minor G (Poirot e4b398c review): ROP1 (`rope_tables`, above) comes
		// from the loaded artifact; SIL1 here comes from a compiled constant
		// instead, an asymmetry that reads as an oversight and is not one --
		// `SslmModel::Load` (model.cpp) already validates any loaded SIL1
		// element-by-element against this same `kSiluLutCanonicalTable`, so
		// the two cannot differ for an artifact that loads at all.
		st = MlpActSite(gate_codes.data(), gate_scale, up_codes.data(), up_scale, intermediate_size,
		                kSiluLutCanonicalTable, lw.mlp_act_site_constant, act_codes.data(),
		                &act_scale, LayerSite(site_prefix, l, "mlp_act"), token_index,
		                trace_hook_state);
		if (st != SslmForwardStatus::Ok) return st;

		st = ProjectAndFunnel(act_codes.data(), act_scale, lw.down_weight, intermediate_size,
		                      hidden_size, lw.proj_identity, lw.proj_mult, lw.proj_shift,
		                      lw.down_site_constant, /*bias=*/nullptr, down_codes.data(), &down_scale,
		                      LayerSite(site_prefix, l, "down_proj.requant"), token_index,
		                      trace_hook_state);
		if (st != SslmForwardStatus::Ok) return st;

		// Reconciles against the STAGED attention-residual output (the
		// current stream this layer is still composing), not `seq` --
		// matching the attn_residual call's own reasoning above.
		st = ResidualReconcileSite(down_codes.data(), down_scale, attn_stream.data(),
		                           attn_stream_scale, hidden_size, lw.mlp_residual_site_constant,
		                           stream_next.data(), &stream_scale,
		                           LayerSite(site_prefix, l, "mlp_residual"), token_index,
		                           trace_hook_state);
		if (st != SslmForwardStatus::Ok) return st;

		// §9.3/Critical 4: the ONE commit point for the whole layer.
		// `hidden_codes`, `hidden_scale`, AND `layer_index` all move together
		// here, only once every checked call in the layer has returned Ok --
		// the same atomicity the `layer_budget == 0` cell already requires
		// one level up, extended across the whole layer body. A rejection
		// anywhere above this line returns before any of the three is
		// touched, so `seq` is left exactly as it was before this layer's
		// attempt started, ready to re-attempt the WHOLE layer on resume.
		for (size_t i = 0; i < hidden_size; ++i) seq.hidden_codes[i] = stream_next[i];
		seq.hidden_scale = stream_scale;
		seq.layer_index = l + 1;
		// S3.7 (§11 S3.7 "The mechanism"): `context_length` advances at this
		// SAME commit point, once, only when the layer that just committed is
		// the token's last -- reusing the loop's existing atomicity rather
		// than adding a second commit point (§9.3/Critical 4's law). No other
		// call site resets or touches it; unlike `layer_index`, a fresh-token
		// embed never zeroes it, because it counts committed cache slots, not
		// progress through the token in flight.
		if (seq.layer_index == num_hidden_layers) {
			seq.context_length += 1;
		}
		++advanced;
	}

	return SslmForwardStatus::Ok;
}

// Master plan §6.4 steps 14-15's real two-step composition (T-1389; built
// against Claude/Curie/superslm-s3.6-head-and-greedy-decode-test-design-
// 2026-07-31.md's red suite, replacing the WorkspaceTooSmall stub the
// test-design pass landed): the real GEMM into the caller-owned wide row,
// then the funnel's second entry point, which performs C35's own domain
// check and narrows only once every element has been individually proven to
// fit int32 (checked_chain_funnel.h). Neither step is reordered, and no
// third step is inserted -- `out_logits` is exactly `NarrowRowChecked`'s own
// output, never independently touched by this function.
SslmForwardStatus LogitsSite(const int8_t* final_codes, size_t hidden_size,
                              const int8_t* head_weights, size_t vocab_size,
                              int64_t* wide_logits, int32_t* out_logits) {
	GemmInt8AccumulateRow(final_codes, head_weights, hidden_size, vocab_size, wide_logits);
	return NarrowRowChecked(wide_logits, vocab_size, out_logits);
}

// C16's pinned tie-break (master plan §6.8 row C16, D-SLM35, T-1389): scans
// left to right and keeps the running maximum's FIRST index -- a later
// element strictly greater replaces it; a later element merely EQUAL to the
// running maximum never does, which is what makes this lowest-index rather
// than last-write-wins. Caller-ensures `n >= 1` (this file's existing
// caller-ensures convention, matching FloorDivI64/ShiftByMax's own shape): a
// conformant artifact's `vocab_size` is load-time rejected at 0, so no
// production call ever passes `n == 0`.
int32_t ArgmaxLowestIndexTieBreak(const int32_t* logits, size_t n) {
	int32_t best_index = 0;
	int32_t best_value = logits[0];
	for (size_t i = 1; i < n; ++i) {
		if (logits[i] > best_value) {
			best_value = logits[i];
			best_index = static_cast<int32_t>(i);
		}
	}
	return best_index;
}

// §9.1's real prefill-then-repeat composition (T-1389; built against
// Claude/Curie/superslm-s3.6-head-and-greedy-decode-test-design-2026-07-31.md's
// red suite, replacing the WorkspaceTooSmall stub the test-design pass
// landed).
//
// Every host-supplied id is validated FIRST, in two whole passes with no
// embedding or state mutation in between -- every stop id, then every prompt
// token id, both against `[0, vocab_size)`, both before `seq` or any output
// parameter is touched at all. This is what makes "a bad prompt id past a
// good prefix still leaves everything untouched" true: a partial, already-
// valid prefix of `prompt_tokens` is never embedded before a later bad id in
// the same array is discovered, because no id is embedded until every id in
// both arrays has already been proven in range.
//
// Once every id validates, one whole token -- embed it fresh (§9.3: a fresh
// token resets the layer-position marker to 0 and the residual becomes this
// token's own embed output), then run every layer -- is the one composition
// both the prefill loop and the generation loop below share. Prefill runs it
// for every prompt token EXCEPT the last; the generation loop's own first
// iteration runs it for the last prompt token, which is exactly §9.1's own
// text: "embed the last produced (OR LAST PROMPT) token." Neither prefill's
// own calls, nor the generation loop's whole-token step, differ in this
// composition -- only the generation loop additionally runs final_norm,
// LogitsSite, and the tie-break afterward, because only a PRODUCED token
// needs a logit row computed for it.
//
// Every produced token and logit row is accumulated in LOCAL storage, never
// written through `out_tokens`/`out_logit_rows` directly, and `seq` is
// mutated only through calls whose own atomic-layer contract already governs
// what a mid-call rejection leaves behind (RunLayerLoop, §11 S3.5). The
// caller's own output parameters are committed -- `out_tokens`,
// `out_logit_rows`, `*out_tokens_produced`, `*out_stop_reason` -- only once a
// stop condition is actually reached with no internal call ever returning
// anything but Ok; any rejection along the way returns immediately, before
// that commit, leaving every one of the four exactly as the caller passed it
// in.
SslmForwardStatus RunGreedyDecodeLoop(
    SequenceLayerState& seq, const LayerWeights* layers, uint32_t num_hidden_layers,
    size_t hidden_size, size_t head_dim, size_t num_key_value_heads, size_t intermediate_size,
    int64_t context_cap, const SslmTensorManifest& rope_tables, const int32_t* prompt_tokens,
    size_t prompt_len,
    const int8_t* embed_weights, CarriedScale embed_site_constant, const int32_t* final_norm_gain,
    CarriedScale final_norm_site_constant, const int8_t* head_weights, int32_t vocab_size,
    const int32_t* stop_ids, size_t stop_count, size_t max_new_tokens, uint8_t* workspace,
    size_t workspace_size, int32_t* out_tokens, int32_t* out_logit_rows,
    size_t out_tokens_capacity, size_t* out_tokens_produced,
    SslmDecodeStopReason* out_stop_reason, SslmKvPrecision kv_precision) {
	// S3.7 (§14.4): checked FIRST, before `seq`, `workspace`, or any output
	// parameter is touched, and before any token is embedded -- an artifact
	// carrying `kv_precision = Int16` loads today (CFG1's own domain check
	// only rejects a value outside {0,1}), but S3a builds int8 only.
	if (kv_precision == SslmKvPrecision::Int16) {
		return SslmForwardStatus::KvPrecisionUnsupported;
	}

	// T-1597 (Poirot e24b971 review, Critical 1): checked immediately after
	// `kv_precision`, before `seq` is touched a second way -- this loop's own
	// RunWholeToken step (below) writes `hidden_size` bytes through
	// `seq.hidden_codes` directly, one call frame ahead of ever entering
	// RunLayerLoop, so RunLayerLoop's own `hidden_codes == nullptr` guard
	// (this file, RunLayerLoop's top-of-function guard block) never runs on
	// this entry point. A default-constructed or zero-restored
	// `SequenceLayerState` -- `hidden_codes`'s own default member initializer
	// -- used to terminate the process here with no status ever formed
	// (measured: 0xC0000005) rather than return one.
	if (seq.hidden_codes == nullptr) {
		return SslmForwardStatus::InvalidHiddenCodes;
	}

	// Caller-ensures `out_tokens_capacity >= max_new_tokens` (header comment,
	// forward_sites.h) -- a workspace-sizing question this call does not
	// scope, matching this file's existing caller-ensures convention for
	// buffer sizes throughout (not runtime-checked, like GemmInt8AccumulateRow's
	// own `out_acc` sizing).
	(void)out_tokens_capacity;

	// §9.1: every stop id validated against [0, vocab_size) BEFORE the loop
	// starts -- checked here, ahead of every prompt token, so neither array
	// can leave any partial state on a rejection from the other.
	for (size_t i = 0; i < stop_count; ++i) {
		if (stop_ids[i] < 0 || stop_ids[i] >= vocab_size) {
			return SslmForwardStatus::TokenIdOutOfRange;
		}
	}
	// Every prompt token id validated too, in encounter order -- a pure
	// bounds pass, still touching neither `seq` nor any output.
	for (size_t i = 0; i < prompt_len; ++i) {
		if (prompt_tokens[i] < 0 || prompt_tokens[i] >= vocab_size) {
			return SslmForwardStatus::TokenIdOutOfRange;
		}
	}
	// Caller-ensures `prompt_len >= 1` (this file's existing caller-ensures
	// convention for an unstated domain precondition, matching
	// ArgmaxLowestIndexTieBreak's own `n >= 1`): "the last prompt token",
	// below, presupposes one exists.

	std::vector<int8_t> embed_codes(hidden_size);
	std::vector<int8_t> final_codes(hidden_size);
	const size_t vocab_size_z = static_cast<size_t>(vocab_size);
	std::vector<int64_t> wide_logits(vocab_size_z);
	std::vector<int32_t> logit_row(vocab_size_z);
	CarriedScale embed_scale{};

	// One whole token: embed it fresh, then run every layer. Shared by both
	// the prefill loop and the generation loop's own first act.
	auto RunWholeToken = [&](int32_t token) -> SslmForwardStatus {
		const SslmForwardStatus est = EmbedEntry(token, vocab_size, embed_weights, hidden_size,
		                                          embed_site_constant, embed_codes.data(),
		                                          &embed_scale);
		if (est != SslmForwardStatus::Ok) return est;
		for (size_t i = 0; i < hidden_size; ++i) seq.hidden_codes[i] = embed_codes[i];
		seq.hidden_scale = embed_scale;
		seq.layer_index = 0;
		return RunLayerLoop(seq, layers, num_hidden_layers, /*layer_budget=*/num_hidden_layers,
		                     hidden_size, head_dim, num_key_value_heads, intermediate_size,
		                     context_cap, rope_tables, workspace, workspace_size);
	};

	// Prefill: every prompt token except the last (the last is folded into
	// the generation loop's own first iteration below).
	for (size_t i = 0; i + 1 < prompt_len; ++i) {
		const SslmForwardStatus st = RunWholeToken(prompt_tokens[i]);
		if (st != SslmForwardStatus::Ok) return st;
	}

	std::vector<int32_t> produced_tokens;
	std::vector<int32_t> produced_logit_rows;
	produced_tokens.reserve(max_new_tokens);
	produced_logit_rows.reserve(max_new_tokens * vocab_size_z);

	int32_t current_token = prompt_tokens[prompt_len - 1];
	SslmDecodeStopReason stop_reason = SslmDecodeStopReason::MaxTokensReached;

	while (produced_tokens.size() < max_new_tokens) {
		SslmForwardStatus st = RunWholeToken(current_token);
		if (st != SslmForwardStatus::Ok) return st;

		CarriedScale final_scale{};
		st = RmsNormSite(seq.hidden_codes, final_norm_gain, hidden_size, seq.hidden_scale,
		                  final_norm_site_constant, final_codes.data(), &final_scale,
		                  "final_norm");
		if (st != SslmForwardStatus::Ok) return st;

		st = LogitsSite(final_codes.data(), hidden_size, head_weights, vocab_size,
		                wide_logits.data(), logit_row.data());
		if (st != SslmForwardStatus::Ok) return st;

		const int32_t token = ArgmaxLowestIndexTieBreak(logit_row.data(), vocab_size_z);

		// §9.1: the produced token is appended BEFORE the stop-id test, so a
		// matched stop token is present in the output and in both digests.
		produced_tokens.push_back(token);
		produced_logit_rows.insert(produced_logit_rows.end(), logit_row.begin(), logit_row.end());

		bool matched = false;
		for (size_t i = 0; i < stop_count; ++i) {
			if (stop_ids[i] == token) {
				matched = true;
				break;
			}
		}
		if (matched) {
			stop_reason = SslmDecodeStopReason::StopTokenMatched;
			break;
		}
		if (produced_tokens.size() >= max_new_tokens) {
			stop_reason = SslmDecodeStopReason::MaxTokensReached;
			break;
		}
		current_token = token;
	}

	// The commit: every internal call above returned Ok, so the caller's own
	// output buffers are touched now, for the first time.
	for (size_t i = 0; i < produced_tokens.size(); ++i) out_tokens[i] = produced_tokens[i];
	for (size_t i = 0; i < produced_logit_rows.size(); ++i) {
		out_logit_rows[i] = produced_logit_rows[i];
	}
	*out_tokens_produced = produced_tokens.size();
	*out_stop_reason = stop_reason;
	return SslmForwardStatus::Ok;
}

}  // namespace superslm

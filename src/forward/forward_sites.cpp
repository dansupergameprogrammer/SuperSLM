// SuperSLM S3.2/S3.3 site compositions.
//
// See include/superslm/forward_sites.h for the contract
// (SuperSLM_S3a_WalkingSkeleton_Plan.md §11 S3.2, §11 S3.3; C31, C24/C25, C28,
// F-S3-8, C27, C33). The S3.2 bodies (FloorDivI64, RmsNormSite,
// ApplyWeightScaleFold, BiasReconcile, EmbedEntry) and the S3.3 bodies below
// (LandingRescale, ClampRopeCode) are the real green construction against the
// red suite authored in tests/test_main.cpp (Claude/Curie/
// superslm-s3.2-weightless-and-projection-sites-test-design-2026-07-28.md
// §11; superslm-s3.3-attention-interior-test-design-2026-07-28.md §11/§12).
#include "superslm/forward_sites.h"

#include <vector>

#include "superslm/intmath.h"

namespace superslm {

namespace {

// pipeline.py:190: NORM_FRAC_BITS = 16 (SuperSLM_S3a_WalkingSkeleton_Plan.md
// §5.1's own pinned integer — the normalization divide and the per-element
// wide-row divide both shift by 2*NORM_FRAC_BITS).
constexpr int kNormFracBits = 16;

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
	// this function is ever invoked; it performs no check of its own, matching
	// every other funnel-adjacent compute in this tree.
	const int64_t exponent = q_b + 62 + e_a;
	return RoundingDivideByPOT(b * r_a, static_cast<int>(exponent));
}

int64_t LandingRescale(int64_t branch_code, int64_t m_a, int64_t r_t, int64_t e_a, int64_t e_t,
                        uint64_t* out_saturation_count) {
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

	const int64_t k = 62 - (e_a - e_t);
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
		} else {
			const U128 verify = U128Shr(shifted, shift);
			magnitude_exceeds_clamp = (verify.lo != magnitude.lo || verify.hi != magnitude.hi) ||
			                           shifted.hi != 0 || shifted.lo > 127;
		}
		raw = static_cast<int64_t>(shifted.lo);
	}
	if (negative) raw = -raw;

	// T-518 / D-SLM201 option 2, §8.2: the predicated-increment half. The
	// clamp comparison the caller's own `clamp(LandingRescale(...), -127, 127)`
	// performs is evaluated here, once, against this function's own
	// about-to-be-returned raw value, OR-ed with the internal loss detection
	// above -- and increments `*out_saturation_count` by exactly one when
	// either fires. This has no effect whatsoever on `raw`.
	if (out_saturation_count != nullptr && (magnitude_exceeds_clamp || raw < -127 || raw > 127)) {
		*out_saturation_count += 1;
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

}  // namespace superslm

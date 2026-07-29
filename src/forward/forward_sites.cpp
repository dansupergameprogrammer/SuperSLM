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

// STUB (T-1345): declared and stubbed by the test-design pass authoring the
// red suite (Claude/Curie/superslm-s3.4-mlp-act-site-test-design-2026-07-29.md),
// following the RoPE application site's own declare-and-stub sequence
// (D-SLM384/385/386). Returns WorkspaceTooSmall unconditionally -- a status
// none of this site's real outcomes (Ok, SiluCompositionScaleOutOfDomain)
// ever is, matching the established stub convention (RopeApplySite's own
// prior stub, this file's git history). Calls nothing, reads nothing, and
// writes nothing to out_codes/out_scale. The build seat replaces this body
// with the real four-step composition the header's own doc comment states.
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
	// (kCompositionScaleMinE = -80 at the low end, and an upper branch pinned
	// against kSiluLutTermLeftShiftOverflowExponent, §5.4's executed
	// e = 8 ceiling). SiluSigmoidQ15 takes `int e`, and every value that
	// reaches this narrowing is inside [-80, 8] by that check — so the
	// conversion is exact here BECAUSE of step 1's ordering, not by
	// assumption about the caller.
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
	// [gate_scale[t], up_scale[t]]). Neither scale is dropped and the order is
	// not free: the span is what the funnel composes into the outgoing carried
	// scale. `site`/`token_index`/`trace_hook_state` are forwarded unchanged
	// (§11 S3.1a, D-SLM362) — this site never fixes its own name, so the
	// caller's own layer-qualified string is exactly what reaches the
	// emission seam.
	const CarriedScale incoming[2] = {gate_scale, up_scale};
	const ChainResult result = RequantChainChecked(
	    wide.data(), n, std::span<const CarriedScale>{incoming, 2}, site_constant, out_codes,
	    out_scale, site, token_index, trace_hook_state);
	return result.status;
}

// STUB (T-1347): declared and stubbed by the test-design pass authoring
// S3.5's red suite (Claude/Curie/superslm-s3.5-residual-and-layer-loop-
// test-design-2026-07-29.md), following the same declare-and-stub sequence
// RopeApplySite and MlpActSite used (D-SLM384/385/386; T-1345). Returns
// WorkspaceTooSmall unconditionally -- a status none of this site's real
// outcomes ever is, matching the established stub convention. Calls
// nothing, reads nothing, and writes nothing to out_codes/out_scale. The
// build seat replaces this body with the real four-step composition the
// header's own doc comment states.
SslmForwardStatus ResidualReconcileSite(const int8_t* branch_code, CarriedScale branch_scale,
                                          const int8_t* stream_code, CarriedScale stream_scale,
                                          size_t hidden_size, CarriedScale site_constant,
                                          int8_t* out_codes, CarriedScale* out_scale,
                                          std::string_view site, size_t token_index,
                                          SslmTraceHookState* trace_hook_state) {
	// C26 (§6.2 step 8 / §6.3 step 13), the header's four steps in order.
	// Step 1: the reciprocal is C19 over the STREAM's own mantissa, never the
	// branch's -- the branch is what gets rescaled INTO the stream's scale, so
	// the stream's is the target.
	const int64_t r_h = DynamicScaleReciprocal(stream_scale.m);

	std::vector<int64_t> wide(hidden_size);
	for (size_t i = 0; i < hidden_size; ++i) {
		// Step 2: the SAME primitive C27's landing composite uses, at a
		// different call site -- there the reciprocal is the static offline
		// one, here it is derived from the stream at runtime. Unclamped in the
		// saturation-counting sense: T-518's counter belongs to C27's landing
		// site, not to this one, so no counter is passed.
		const int64_t reconciled =
		    LandingRescale(static_cast<int64_t>(branch_code[i]), branch_scale.m, r_h,
		                   branch_scale.e, stream_scale.e, /*out_saturation_count=*/nullptr);
		// Step 3: the wide add, both operands now nominally at the stream's
		// own scale.
		wide[i] = reconciled + static_cast<int64_t>(stream_code[i]);
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

// STUB (T-1347): same convention as ResidualReconcileSite above. Returns
// WorkspaceTooSmall unconditionally, touching neither `seq` nor `workspace`
// -- this is deliberate, not an oversight: the red suite's own budget=0 and
// resume/workspace cells depend on the STUB leaving every out-param
// untouched, exactly as every other declare-and-stub site in this file
// does on rejection. The build seat replaces this body with the real
// per-layer composition the header's own doc comment states.
namespace {

// One projection: GemmInt8AccumulateRow -> the shared WSC1 fold -> the funnel.
// §6.2 step 2's own shape, and q_proj/o_proj/gate_proj/up_proj/down_proj all
// have it -- they differ only in weights, incoming scale, and site constant,
// never in construction.
SslmForwardStatus ProjectAndFunnel(const int8_t* in_codes, CarriedScale in_scale,
                                    const int8_t* weight, size_t in_channels, size_t out_channels,
                                    int32_t identity, int32_t mult, int32_t shift,
                                    CarriedScale site_constant, int8_t* out_codes,
                                    CarriedScale* out_scale, std::string_view site,
                                    size_t token_index, SslmTraceHookState* trace_hook_state) {
	std::vector<int64_t> acc(out_channels);
	GemmInt8AccumulateRow(in_codes, weight, in_channels, out_channels, acc.data());
	for (size_t i = 0; i < out_channels; ++i) {
		acc[i] = ApplyWeightScaleFold(acc[i], identity, mult, shift);
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

SslmForwardStatus RunLayerLoop(SequenceLayerState& seq, const LayerWeights* layers,
                                 uint32_t num_hidden_layers, uint32_t layer_budget,
                                 size_t hidden_size, size_t head_dim, size_t intermediate_size,
                                 int64_t context_cap, const SslmTensorManifest& rope_tables,
                                 uint8_t* workspace, size_t workspace_size,
                                 std::string_view site_prefix, size_t token_index,
                                 SslmTraceHookState* trace_hook_state) {
	// §9.3's first decided contract, checked BEFORE anything is read or
	// written: a budget of 0 consumes a call, advances nothing, and would
	// return "pending" -- a host-visible livelock. `seq` is left bit-identical,
	// which is exactly what the cell poison-fills to assert.
	if (layer_budget == 0) return SslmForwardStatus::InvalidLayerBudget;

	// §9.4: the K/V store is caller-supplied memory "sized from
	// config.context_cap, num_key_value_heads, head_dim, and kv_precision".
	// That is what `workspace` is here and the only thing this loop takes from
	// it -- per-site scratch heaps through std::vector, exactly as RmsNormSite
	// and MlpActSite already do in this same translation unit. The required
	// size is therefore computed from the parameters rather than chosen here.
	//
	// The signature carries no `num_key_value_heads`, so the MHA-degenerate
	// case is not a fixture choice this body is free to make -- it is forced by
	// the declared surface, and a GQA loop needs a parameter that does not
	// exist yet.
	const size_t num_heads = head_dim == 0 ? 0 : hidden_size / head_dim;
	if (num_heads == 0 || num_heads * head_dim != hidden_size) {
		return SslmForwardStatus::WorkspaceTooSmall;
	}
	const size_t kv_bytes_needed = static_cast<size_t>(num_hidden_layers) *
	                                static_cast<size_t>(context_cap) * num_heads * head_dim * 2u;
	if (workspace_size < kv_bytes_needed) return SslmForwardStatus::WorkspaceTooSmall;

	// This sub-slot's declared scope is a single position. §9.4's multi-
	// position accessor (`KeyRow(layer, kv_head, position)`) is S3.7's and does
	// not exist; the signature carries no position parameter, so position 0
	// with a one-key attention row is the only reading the declared surface
	// admits.
	const size_t width = 1;
	const int64_t position = 0;

	int8_t* const kv_base = reinterpret_cast<int8_t*>(workspace);

	std::vector<int8_t> normed(hidden_size), q_codes(hidden_size), o_codes(hidden_size);
	std::vector<int8_t> q_rot(hidden_size), k_rot(hidden_size), ctx_codes(hidden_size);
	std::vector<int8_t> gate_codes(intermediate_size), up_codes(intermediate_size);
	std::vector<int8_t> act_codes(intermediate_size), down_codes(hidden_size);
	std::vector<int8_t> stream_next(hidden_size);

	uint32_t advanced = 0;
	while (advanced < layer_budget && seq.layer_index < num_hidden_layers) {
		const uint32_t l = seq.layer_index;
		const LayerWeights& lw = layers[l];
		CarriedScale normed_scale{}, q_scale{}, ctx_scale{}, o_scale{};
		CarriedScale mlp_normed_scale{}, gate_scale{}, up_scale{}, act_scale{}, down_scale{};
		CarriedScale stream_scale{};
		SslmForwardStatus st;

		// --- attention half (§6.2) --------------------------------------------
		st = RmsNormSite(seq.hidden_codes, lw.attn_norm_gain, hidden_size, seq.hidden_scale,
		                 lw.attn_norm_site_constant, normed.data(), &normed_scale,
		                 LayerSite(site_prefix, l, "attn_norm"), token_index, trace_hook_state);
		if (st != SslmForwardStatus::Ok) return st;

		st = ProjectAndFunnel(normed.data(), normed_scale, lw.q_weight, hidden_size, hidden_size,
		                      lw.proj_identity, lw.proj_mult, lw.proj_shift, lw.q_site_constant,
		                      q_codes.data(), &q_scale, LayerSite(site_prefix, l, "q_proj.requant"),
		                      token_index, trace_hook_state);
		if (st != SslmForwardStatus::Ok) return st;

		// k_proj / v_proj do NOT funnel: they land at the static per-head scale
		// through LandingRescale (§8.1), writing straight into the K/V store.
		int8_t* const k_store =
		    kv_base + (static_cast<size_t>(l) * static_cast<size_t>(context_cap) * num_heads *
		               head_dim) * 2u;
		int8_t* const v_store = k_store + static_cast<size_t>(context_cap) * num_heads * head_dim;
		{
			std::vector<int64_t> kacc(hidden_size), vacc(hidden_size);
			GemmInt8AccumulateRow(normed.data(), lw.k_weight, hidden_size, hidden_size,
			                      kacc.data());
			GemmInt8AccumulateRow(normed.data(), lw.v_weight, hidden_size, hidden_size,
			                      vacc.data());
			for (size_t i = 0; i < hidden_size; ++i) {
				const int64_t kf =
				    ApplyWeightScaleFold(kacc[i], lw.proj_identity, lw.proj_mult, lw.proj_shift);
				const int64_t vf =
				    ApplyWeightScaleFold(vacc[i], lw.proj_identity, lw.proj_mult, lw.proj_shift);
				k_store[i] = static_cast<int8_t>(LandingRescale(
				    kf, normed_scale.m, lw.kv_landing_r_t, normed_scale.e, lw.kv_landing_e_t));
				v_store[i] = static_cast<int8_t>(LandingRescale(
				    vf, normed_scale.m, lw.kv_landing_r_t, normed_scale.e, lw.kv_landing_e_t));
			}
		}

		// RoPE on q and on the just-landed k, per head (§6.2 step 3).
		for (size_t h = 0; h < num_heads; ++h) {
			st = RopeApplySite(q_codes.data() + h * head_dim, head_dim, position, context_cap,
			                   rope_tables, q_rot.data() + h * head_dim);
			if (st != SslmForwardStatus::Ok) return st;
			st = RopeApplySite(k_store + h * head_dim, head_dim, position, context_cap, rope_tables,
			                   k_rot.data() + h * head_dim);
			if (st != SslmForwardStatus::Ok) return st;
		}
		for (size_t i = 0; i < hidden_size; ++i) k_store[i] = k_rot[i];

		// Attention proper (§6.2 step 5). No named site for this composition
		// exists anywhere in this tree; this is where it is first composed.
		{
			std::vector<int64_t> ctx_wide(hidden_size);
			for (size_t h = 0; h < num_heads; ++h) {
				std::vector<int64_t> scores(width), probs(width), ctx_acc(head_dim);
				GemmInt8AccumulateRow(q_rot.data() + h * head_dim, k_store + h * head_dim, head_dim,
				                      width, scores.data());
				// The gate before the kernel, in that order -- the kernel's own
				// contract states it performs no width check and that `total`'s
				// bound holds only under this ordering.
				st = CheckSoftmaxRowWidthDomain(lw.q_b_iexp, lw.q_c_iexp, width);
				if (st != SslmForwardStatus::Ok) return st;
				if (!SoftmaxRowQ15(scores.data(), width, lw.q_ln2, lw.q_b_iexp, lw.q_c_iexp,
				                   probs.data())) {
					// The kernel refused after its own gate accepted. No status
					// distinguishes that from the gate's own rejection, so the
					// gate's is reported rather than inventing an enumerator --
					// named as an owed distinction in this slot's build record.
					return SslmForwardStatus::SoftmaxRowWidthOutOfDomain;
				}
				GemmProbQ15Accumulate(probs.data(), v_store + h * head_dim, width, head_dim,
				                      ctx_acc.data());
				for (size_t d = 0; d < head_dim; ++d) {
					ctx_wide[h * head_dim + d] = ApplyWeightScaleFold(
					    ctx_acc[d], lw.ctx_fold_identity, lw.ctx_fold_mult, lw.ctx_fold_shift);
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
		                      o_codes.data(), &o_scale, LayerSite(site_prefix, l, "o_proj.requant"),
		                      token_index, trace_hook_state);
		if (st != SslmForwardStatus::Ok) return st;

		st = ResidualReconcileSite(o_codes.data(), o_scale, seq.hidden_codes, seq.hidden_scale,
		                           hidden_size, lw.attn_residual_site_constant, stream_next.data(),
		                           &stream_scale, LayerSite(site_prefix, l, "attn_residual"),
		                           token_index, trace_hook_state);
		if (st != SslmForwardStatus::Ok) return st;
		for (size_t i = 0; i < hidden_size; ++i) seq.hidden_codes[i] = stream_next[i];
		seq.hidden_scale = stream_scale;

		// --- MLP half (§6.3) ---------------------------------------------------
		st = RmsNormSite(seq.hidden_codes, lw.mlp_norm_gain, hidden_size, seq.hidden_scale,
		                 lw.mlp_norm_site_constant, normed.data(), &mlp_normed_scale,
		                 LayerSite(site_prefix, l, "mlp_norm"), token_index, trace_hook_state);
		if (st != SslmForwardStatus::Ok) return st;

		st = ProjectAndFunnel(normed.data(), mlp_normed_scale, lw.gate_weight, hidden_size,
		                      intermediate_size, lw.proj_identity, lw.proj_mult, lw.proj_shift,
		                      lw.gate_site_constant, gate_codes.data(), &gate_scale,
		                      LayerSite(site_prefix, l, "gate_proj.requant"), token_index,
		                      trace_hook_state);
		if (st != SslmForwardStatus::Ok) return st;
		st = ProjectAndFunnel(normed.data(), mlp_normed_scale, lw.up_weight, hidden_size,
		                      intermediate_size, lw.proj_identity, lw.proj_mult, lw.proj_shift,
		                      lw.up_site_constant, up_codes.data(), &up_scale,
		                      LayerSite(site_prefix, l, "up_proj.requant"), token_index,
		                      trace_hook_state);
		if (st != SslmForwardStatus::Ok) return st;

		st = MlpActSite(gate_codes.data(), gate_scale, up_codes.data(), up_scale, intermediate_size,
		                kSiluLutCanonicalTable, lw.mlp_act_site_constant, act_codes.data(),
		                &act_scale, LayerSite(site_prefix, l, "mlp_act"), token_index,
		                trace_hook_state);
		if (st != SslmForwardStatus::Ok) return st;

		st = ProjectAndFunnel(act_codes.data(), act_scale, lw.down_weight, intermediate_size,
		                      hidden_size, lw.proj_identity, lw.proj_mult, lw.proj_shift,
		                      lw.down_site_constant, down_codes.data(), &down_scale,
		                      LayerSite(site_prefix, l, "down_proj.requant"), token_index,
		                      trace_hook_state);
		if (st != SslmForwardStatus::Ok) return st;

		st = ResidualReconcileSite(down_codes.data(), down_scale, seq.hidden_codes,
		                           seq.hidden_scale, hidden_size, lw.mlp_residual_site_constant,
		                           stream_next.data(), &stream_scale,
		                           LayerSite(site_prefix, l, "mlp_residual"), token_index,
		                           trace_hook_state);
		if (st != SslmForwardStatus::Ok) return st;
		for (size_t i = 0; i < hidden_size; ++i) seq.hidden_codes[i] = stream_next[i];
		seq.hidden_scale = stream_scale;

		// §9.3: the resume point, and the ONLY one -- advanced after the MLP
		// residual, which is where the reference yields.
		seq.layer_index = l + 1;
		++advanced;
	}

	return SslmForwardStatus::Ok;
}

}  // namespace superslm

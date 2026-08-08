// T-1822 activation-scale remedy — Stage A primitive-tier implementation (T-1833,
// Brunel). See include/superslm/t1822_activation_scale_remedy.h for the per-function
// contract, and Claude/Vitruvius/t1822-activation-scale-remedy-design-2026-08-07.md
// §4-§6 for the mechanisms these functions implement at the VALUE level.
//
// T-1834 O3 (D-SLM1897): this file is a SCANNED-AND-ALLOWLISTED leaf caller of
// tests/ci/check_no_forward_leaf_calls.py's forward-leaf ban -- named in both
// _DEFAULT_FORWARD_GLOBS and _DEFAULT_ALLOWLIST, rather than sitting outside
// the scan root the way it did before this fold. The primitive tier calls
// superslm::RequantTokenCodeWide directly by design (ComputeGroupedCode's own
// contract: "not a re-implementation of C22 — a call to it"), which the
// forward-leaf ban would otherwise refuse; the allowlist entry certifies that
// call explicitly instead of the file going dark to the check by living
// outside its glob root, the exact past scar that check's own module
// docstring records.
#include "superslm/t1822_activation_scale_remedy.h"

#include <vector>

#include "superslm/intmath.h"

namespace superslm_t1822 {
namespace {

// --- Portable 128-bit facility, self-contained in this TU -----------------------
// Mirrors src/intmath.cpp's own dual-path convention (native __int128 where the
// compiler offers one; a hand-rolled {lo, hi} struct for MSVC, which has none) --
// duplicated here rather than shared, matching the house style already established
// per-TU (src/forward/forward_sites.cpp carries its own copy of the same shape).
// This facility backs BOTH ComputePeeledCode and ReferencePeeledCode below; the two
// functions compose it into the M2 step-3 composite via genuinely different
// constructions (a doubled-numerator divide vs. a remainder/threshold divide) so
// the differential cell (§12 "Differential vs. the independent reference" (ii))
// exercises two independently-shaped rounding paths, not one path called twice.

#if defined(__SIZEOF_INT128__)

using Wide128 = unsigned __int128;

inline Wide128 WideMul64(uint64_t a, uint64_t b) {
	return static_cast<Wide128>(a) * b;
}
// 128 * (small u64) -> 128; caller guarantees the true product fits 128 bits (it
// does here: at most a ~2^95 operand times 127).
inline Wide128 WideMulSmall(Wide128 a, uint64_t b) { return a * b; }
inline Wide128 WideShl1(Wide128 a) { return a << 1; }
inline Wide128 WideAddSmall(Wide128 a, uint64_t b) { return a + b; }
inline Wide128 WideIncrement(Wide128 a) { return a + 1; }
// Full logical right shift, k in [0, 127].
inline Wide128 WideShrFull(Wide128 v, int k) {
	if (k <= 0) return v;
	if (k >= 128) return static_cast<Wide128>(0);
	return v >> k;
}
// Full logical left shift, k in [0, 127] -- the F2/F6 rewrites' own mirror of
// WideShrFull above, needed to compose `group_max_abs << k` and `2^r_cap << r_cap`
// at full width rather than in the plain int64 arithmetic F1-F3's casebook shows
// wraps (T-1834, §6.8).
//
// T-1844 N1 (Poirot, 5899b8f): the two-arg form is total over its own 128-bit
// width -- it never invokes undefined behavior -- but it is NOT total against
// wrapping: a shift whose true result needs more than 128 bits silently drops
// the high bits, same as any fixed-width shift. Every caller that infers
// "did this overflow 64 bits" from testing the RESULT's high word (WideHi(...)
// != 0) is trusting that the shift itself did not ALSO wrap the 128-bit
// container, which is false once `k` plus v's own bit-length reaches 128 --
// exactly the F6 defect this finding is closing. `out_truncated`, when
// non-null, is set whenever any set bit of `v` was shifted out of the 128-bit
// result: computed by shifting the result back down by the same `k` (via
// WideShrFull, already total the identical way) and comparing to `v` -- if
// shifting back does not recover the original value, bits were lost going
// forward. This reuses the two already-verified total primitives rather than
// re-deriving a bit-length formula, and it is correct at every boundary
// WideShrFull/WideShlFull themselves handle specially (k <= 0, k >= 128).
inline Wide128 WideShlFull(Wide128 v, int k, bool* out_truncated = nullptr) {
	Wide128 result;
	if (k <= 0) {
		result = v;
	} else if (k >= 128) {
		result = static_cast<Wide128>(0);
	} else {
		result = v << k;
	}
	if (out_truncated) *out_truncated = (WideShrFull(result, k) != v);
	return result;
}
// 128-bit + 128-bit addition, exact whenever the true sum fits 128 bits (every
// call site below bounds both operands well under that before adding).
inline Wide128 WideAddWide(Wide128 a, Wide128 b) { return a + b; }
inline uint64_t WideHi(Wide128 v) { return static_cast<uint64_t>(v >> 64); }
inline uint64_t WideLo(Wide128 v) { return static_cast<uint64_t>(v); }
inline bool WideLessEqual(Wide128 a, Wide128 b) { return a <= b; }

#else  // MSVC and any other toolchain without a native 128-bit integer.

struct Wide128 {
	uint64_t lo;
	uint64_t hi;
};

inline Wide128 WideMul64(uint64_t a, uint64_t b) {
	// 64x64 -> 128 unsigned, schoolbook.
	uint64_t ll = (a & 0xFFFFFFFFu) * (b & 0xFFFFFFFFu);
	uint64_t lh = (a & 0xFFFFFFFFu) * (b >> 32);
	uint64_t hl = (a >> 32) * (b & 0xFFFFFFFFu);
	uint64_t hh = (a >> 32) * (b >> 32);
	uint64_t mid = (ll >> 32) + (lh & 0xFFFFFFFFu) + (hl & 0xFFFFFFFFu);
	uint64_t lo = (ll & 0xFFFFFFFFu) | (mid << 32);
	uint64_t hi = hh + (lh >> 32) + (hl >> 32) + (mid >> 32);
	return Wide128{lo, hi};
}

inline Wide128 WideMulSmall(Wide128 a, uint64_t b) {
	Wide128 lo_part = WideMul64(a.lo, b);
	uint64_t hi_part = a.hi * b;  // a.hi is small here; no overflow of the retained 128 bits
	return Wide128{lo_part.lo, lo_part.hi + hi_part};
}

inline Wide128 WideShl1(Wide128 a) {
	return Wide128{a.lo << 1, (a.hi << 1) | (a.lo >> 63)};
}

inline Wide128 WideAddSmall(Wide128 a, uint64_t b) {
	uint64_t lo = a.lo + b;
	uint64_t carry = (lo < a.lo) ? 1u : 0u;
	return Wide128{lo, a.hi + carry};
}

inline Wide128 WideIncrement(Wide128 a) { return WideAddSmall(a, 1u); }

inline Wide128 WideShrFull(Wide128 v, int k) {
	if (k <= 0) return v;
	if (k >= 128) return Wide128{0, 0};
	if (k >= 64) return Wide128{v.hi >> (k - 64), 0};
	return Wide128{(v.lo >> k) | (v.hi << (64 - k)), v.hi >> k};
}

// Full logical left shift, k in [0, 127] -- the struct-path mirror of
// WideShrFull above (F2/F6 rewrites, §6.8).
//
// T-1844 N1 (Poirot, 5899b8f): see the native-path WideShlFull's comment for
// the truncation-report rationale -- identical here, computed by shifting the
// result back down (WideShrFull) and comparing field-for-field, since Wide128
// has no operator== on this path.
inline Wide128 WideShlFull(Wide128 v, int k, bool* out_truncated = nullptr) {
	Wide128 result;
	if (k <= 0) {
		result = v;
	} else if (k >= 128) {
		result = Wide128{0, 0};
	} else if (k >= 64) {
		result = Wide128{0, v.lo << (k - 64)};
	} else {
		result = Wide128{v.lo << k, (v.hi << k) | (v.lo >> (64 - k))};
	}
	if (out_truncated) {
		const Wide128 back = WideShrFull(result, k);
		*out_truncated = (back.lo != v.lo) || (back.hi != v.hi);
	}
	return result;
}

inline Wide128 WideAddWide(Wide128 a, Wide128 b) {
	const uint64_t lo = a.lo + b.lo;
	const uint64_t carry = (lo < a.lo) ? 1u : 0u;
	return Wide128{lo, a.hi + b.hi + carry};
}

inline uint64_t WideHi(Wide128 v) { return v.hi; }
inline uint64_t WideLo(Wide128 v) { return v.lo; }
inline bool WideLessEqual(Wide128 a, Wide128 b) {
	if (a.hi != b.hi) return a.hi < b.hi;
	return a.lo <= b.lo;
}

#endif

// Unsigned two's-complement magnitude of a signed int64 -- defined for every
// int64_t, including INT64_MIN, whose magnitude 2^63 has no int64_t representation
// but has an exact uint64_t one (the same convention intmath.cpp and
// forward_sites.cpp already use).
inline uint64_t AbsMagnitude(int64_t v) {
	return v < 0 ? (~static_cast<uint64_t>(v) + 1u) : static_cast<uint64_t>(v);
}

// T-1844 N5 (Poirot, 5899b8f): a sign/magnitude "is lhs <= rhs" comparison,
// where each side is carried as (negative flag, unsigned magnitude in Wide128)
// rather than as a signed value the Wide128 facility has no representation
// for. §6.8 specifies ComputeRefinementExponent and
// ComputeRefinementExponentRopeSafe both "correct over the whole int64_t
// domain ... unconditionally" -- the F1 pair gets this for free from plain
// signed int64 comparison; the F2 pair, whose predicate needs 128-bit width,
// cannot use a signed comparison directly because Wide128 is unsigned-only, so
// this reconstructs the same signed ordering from magnitude + sign explicitly.
// `lhs_overflowed` marks a magnitude that did not fit 128 bits (WideShlFull's
// own truncation report): a true magnitude that large exceeds every magnitude
// this file ever forms on the other side of a comparison (bounded to roughly
// 90 * 2^63 < 2^71), so it is treated as "larger than any representable
// magnitude" on whichever side of the sign split it falls -- never as the
// wrapped, silently-small value the shift itself produced.
inline bool WideLessEqualSigned(bool lhs_negative, bool lhs_overflowed, Wide128 lhs_mag,
                                 bool rhs_negative, Wide128 rhs_mag) {
	if (lhs_negative && rhs_negative) {
		// Both negative: lhs <= rhs iff |lhs| >= |rhs|. An overflowed |lhs| is
		// unrepresentably large, so it is >= any representable |rhs| by construction.
		return lhs_overflowed || WideLessEqual(rhs_mag, lhs_mag);
	}
	if (lhs_negative) return true;    // negative <= any non-negative rhs, always.
	if (rhs_negative) return false;   // non-negative lhs > any negative rhs, always.
	// Both non-negative: lhs <= rhs iff |lhs| <= |rhs|. An overflowed |lhs| exceeds
	// any representable |rhs|, so the predicate is false.
	return !lhs_overflowed && WideLessEqual(lhs_mag, rhs_mag);
}

}  // namespace

// ---------------------------------------------------------------------------
// M1 — per-group power-of-two scales (§4.1 steps 2-3)
// ---------------------------------------------------------------------------

int ComputeRefinementExponent(int64_t group_max_abs, int64_t row_max_abs, int k_cap) {
	// §4.1 step 2, T-1834 F1 remedy (§6.8, D-SLM1893): the largest k in
	// [0, k_cap] with (group_max_abs << k) <= row_max_abs, rewritten as
	// group_max_abs <= (row_max_abs >> k) -- the identical predicate for every
	// non-negative group_max_abs/row_max_abs pair (exact integer floor division
	// by 2^k on the right instead of a left shift that can wrap on the left),
	// with no reachable overflow anywhere in the whole int64_t domain
	// (T-1838's exhaustive-plus-random strike, 264,000+200,000 cases, zero
	// mismatches). Constructed as a DOWNWARD search (largest-first) --
	// deliberately the opposite direction from ReferenceRefinementExponent's own
	// upward while-loop below, so the two are independent constructions of the
	// same monotonic predicate rather than one loop direction copied into two
	// functions.
	for (int k = k_cap; k > 0; --k) {
		if (group_max_abs <= (row_max_abs >> k)) return k;
	}
	return 0;
}

int ComputeRefinementExponentRopeSafe(int64_t group_max_abs, int64_t row_max_abs, int k_cap) {
	// §6.2, D-SLM1816: k = 0 is always admissible; otherwise the largest k in
	// [0, k_cap] with 127*(group_max_abs << k) <= 90*row_max_abs. T-1834 F2
	// remedy (§6.8, D-SLM1893): both constant multiplies are computed in the
	// file's own Wide128 facility rather than plain int64_t, which overflows
	// (UB) at both the shift and the x127 well inside this design's domain.
	// Same downward-search shape as the standard path above, independent
	// predicate.
	//
	// T-1844 N5 (Poirot, 5899b8f): the magnitude is formed from AbsMagnitude
	// (correct for every int64_t, including INT64_MIN) rather than a raw
	// uint64_t cast (which reinterpreted a negative group_max_abs as a huge
	// positive one), and the comparison is reconstructed as signed via
	// WideLessEqualSigned -- matching the standard path above, which gets
	// whole-domain correctness for free from plain signed comparison. For
	// every caller in this design's actual domain (group_max_abs >= 1, C20's
	// guard) AbsMagnitude(x) == x, so this is bit-identical to the prior
	// behavior; it only changes the answer for the negative operands §6.8
	// claims as in-domain and the prior cast silently mishandled.
	//
	// T-1844 N1 (Poirot, 5899b8f): WideShlFull's truncation report also closes
	// this function's exposure to the same silent-wrap mis-admit F6 had --
	// this is the "second caller" the casebook names as safe today only
	// because k_cap is small.
	for (int k = k_cap; k > 0; --k) {
		bool lhs_overflowed = false;
		const Wide128 lhs_mag =
		    WideShlFull(WideMul64(AbsMagnitude(group_max_abs), 127u), k, &lhs_overflowed);
		const Wide128 rhs_mag = WideMul64(AbsMagnitude(row_max_abs), 90u);
		if (WideLessEqualSigned(group_max_abs < 0, lhs_overflowed, lhs_mag, row_max_abs < 0,
		                         rhs_mag)) {
			return k;
		}
	}
	return 0;
}

int8_t ComputeGroupedCode(int64_t wide_value, int k_g, int64_t r, int s) {
	// §4.1 step 3, D-SLM1663's kinship framing: a CALL to C22
	// (superslm::RequantTokenCodeWide) on the pre-shifted operand, not a
	// re-implementation of it. T-1834 F4 remedy (§6.8, D-SLM1893): the
	// pre-shift is applied in plain int64_t and can wrap before the call ever
	// sees it, breaking RequantTokenCodeWide's own documented totality
	// (intmath.h:242-247). Saturate BEFORE shifting rather than after: when
	// |wide_value| > (INT64_MAX >> k_g) the exact pre-shift value would not fit
	// int64_t, so the clamp code is returned directly -- 127/-127/0 by sign,
	// mirroring C22's own overflow-into-clamp semantics at the one point
	// upstream of it where the shift itself is what could wrap. Otherwise the
	// shift is exact (no overflow reachable, by the guard) and C22 is called
	// exactly as before.
	//
	// T-1844 N3 (Poirot, 5899b8f): the old `shift = k_g > 0 ? k_g : 0` clamp
	// protected only the `limit` computation below, not the actual data-path
	// shift (`wide_value << k_g`, which still used the raw, unclamped k_g) --
	// so a negative k_g passed the guard (limit == INT64_MAX, everything but
	// INT64_MIN admitted) and then hit `wide_value << k_g` with a negative
	// shift count, undefined behavior. Re-derived independently rather than
	// taking the casebook's suggested `k_g > 62` bound (StandardsDocument.md
	// §7): the shift-count rule in [expr.shift] is undefined only for a count
	// that is negative or >= the operand's width (64 for int64_t/uint64_t), so
	// k_g = 63 is well-defined on both lines below (`INT64_MAX >> 63` is `0`,
	// legal; a shift count of 63 on a 64-bit operand is legal) and only
	// k_g >= 64 is undefined -- the correct domain is [0, 63], not [0, 62].
	// Refuse outside that domain up front, per this file's own explicit-
	// domain-rejection convention (F5, F7), rather than clamping only half the
	// domain the way the old code did.
	if (k_g < 0 || k_g > 63) return 0;
	const uint64_t abs_v = AbsMagnitude(wide_value);
	const uint64_t limit = static_cast<uint64_t>(INT64_MAX) >> k_g;
	if (abs_v > limit) {
		if (wide_value > 0) return 127;
		if (wide_value < 0) return -127;
		return 0;
	}
	return superslm::RequantTokenCodeWide(wide_value << k_g, r, s);
}

int ReferenceRefinementExponent(int64_t group_max_abs, int64_t row_max_abs, int k_cap) {
	// §11's A6-independent differential reference (D-SLM1614(i)): authored without
	// reading ComputeRefinementExponent's own body. T-1834 F1 remedy (§6.8,
	// D-SLM1893): "both sides of the differential were wrong together, and a fix
	// on only one side would make the cell fail where it should now pass" --
	// this side takes the identical right-shift rewrite, kept in its own upward
	// while-loop direction so the two functions remain independent constructions
	// of the same predicate rather than one direction copied into two.
	int k = 0;
	while (k < k_cap && group_max_abs <= (row_max_abs >> (k + 1))) ++k;
	return k;
}

int ReferenceRefinementExponentRopeSafe(int64_t group_max_abs, int64_t row_max_abs, int k_cap) {
	// Mendeleev F1 (D-SLM1881): authored without reading
	// ComputeRefinementExponentRopeSafe's own body. T-1834 F2 remedy (§6.8,
	// D-SLM1893): both constant multiplies computed in Wide128, same as the
	// primitive above, kept in this function's own upward while-loop direction.
	//
	// T-1844 N5/N1 (Poirot, 5899b8f): same AbsMagnitude + WideLessEqualSigned
	// rewrite as ComputeRefinementExponentRopeSafe above, for the identical
	// reason -- kept as an independent construction of the same predicate
	// (this function's own contract), not a call into the primitive.
	int k = 0;
	while (k < k_cap) {
		bool lhs_overflowed = false;
		const Wide128 lhs_mag =
		    WideShlFull(WideMul64(AbsMagnitude(group_max_abs), 127u), k + 1, &lhs_overflowed);
		const Wide128 rhs_mag = WideMul64(AbsMagnitude(row_max_abs), 90u);
		if (!WideLessEqualSigned(group_max_abs < 0, lhs_overflowed, lhs_mag, row_max_abs < 0,
		                          rhs_mag)) {
			break;
		}
		++k;
	}
	return k;
}

bool IsGroupSizeAdmissible(int64_t row_width, int64_t group_size) {
	if (group_size <= 0 || row_width <= 0) return false;
	const bool is_power_of_two = (group_size & (group_size - 1)) == 0;
	if (!is_power_of_two) return false;
	return (row_width % group_size) == 0;
}

bool IsNormConsumerKCapAdmissible(int k_cap) {
	// §6.3b: k_cap above 3 is refused at a norm consumer.
	return k_cap >= 0 && k_cap <= 3;
}

bool IsGroupingAdmissibleAtSite(int site_id, int k_cap_requested) {
	// T-1834 F7 (D-SLM1896): site_id outside §6.6's [1, 18] numbering is a
	// configuration error, and the predicate whose entire job is to refuse
	// inadmissible configurations must reject it rather than fall through to
	// `true`.
	if (site_id < 1 || site_id > 18) return false;
	// §8.5, D-SLM1672: site 1 (embed, committed-state) is refused unconditionally --
	// it has no M1 role at all, so even k_cap_requested == 0 is refused.
	if (site_id == 1) return false;
	// §6.3c, D-SLM1662: sites 11 and 18 (peeled-residual) admit only K = 0 -- the
	// consistent-grid combined bound.
	if (site_id == 11 || site_id == 18) return k_cap_requested == 0;
	return true;
}

bool IsG1AdmissibleAtSite(int site_id, int64_t group_size) {
	// T-1834 F7 (D-SLM1896): same out-of-range refusal as
	// IsGroupingAdmissibleAtSite above -- site_id outside [1, 18] is refused
	// rather than falling through to `site_id != 3` (which evaluates true for
	// every out-of-range id).
	if (site_id < 1 || site_id > 18) return false;
	// §6.2, D-SLM1679: G = 1 is refused only at site 3 (the RoPE pair-in-group
	// floor is 2); admissible everywhere else, and the G == 1 restriction has
	// nothing to say about any other group size.
	if (group_size != 1) return true;
	return site_id != 3;
}

// ---------------------------------------------------------------------------
// M2 — fixed-P outlier peel (§5 steps 1-3)
// ---------------------------------------------------------------------------

size_t SelectPeelIndices(const int64_t* wide_row, size_t n, int p, PeelRecord* out_records) {
	// §5 step 1: the min(p, n) largest-magnitude channels, lowest-index tie-break.
	// Re-derives the set fresh from THIS row on every call -- no static/cached
	// state, which is what audit F9's discrimination cell exercises.
	//
	// T-1834 F5 (D-SLM1895): PeelRecord::index is uint16_t; a row wider than
	// UINT16_MAX would have its true argmax index silently truncated by the
	// narrowing cast below. Reachable by configuration (a wider intermediate),
	// not at this design's pinned n = 1536, but this file's own house
	// convention is explicit domain rejection over a silent narrow -- applied
	// here at the one place it was missing.
	if (n > static_cast<size_t>(UINT16_MAX)) return 0;
	if (p <= 0 || n == 0) return 0;
	const size_t count = static_cast<size_t>(p) < n ? static_cast<size_t>(p) : n;

	// Repeatedly pick the remaining argmax. count <= p <= a fixed small peel budget
	// (this design's swept range tops out at P = 7, §6.3c), so the O(count * n)
	// cost is not a concern regardless of the row width n.
	std::vector<bool> used(n, false);
	for (size_t sel = 0; sel < count; ++sel) {
		size_t best_idx = 0;
		uint64_t best_abs = 0;
		bool found = false;
		for (size_t i = 0; i < n; ++i) {
			if (used[i]) continue;
			const uint64_t a = AbsMagnitude(wide_row[i]);
			if (!found || a > best_abs) {
				best_abs = a;
				best_idx = i;
				found = true;
			}
		}
		used[best_idx] = true;
		out_records[sel].index = static_cast<uint16_t>(best_idx);
		out_records[sel].c_star = wide_row[best_idx];
	}
	return count;
}

int64_t CeilDivPow2(int64_t x, int r_cap) {
	// T-1834 F3 remedy (§6.8, D-SLM1892/1893): the biased-add-then-shift form
	// overflows (UB) for x > INT64_MAX - bias, and does so at exactly the
	// magnitude MaxAbsReduceWide's own deliberate saturation produces
	// (intmath.h:202-216). Rewritten as the remainder form -- exact for the
	// whole non-negative int64_t domain, including INT64_MAX, with no addition
	// that can overflow: (x >> r_cap) + (1 if the low r_cap bits are nonzero
	// else 0). Valid for x >= 0, the only domain unpeeled_max_abs and
	// full_row_max_abs occupy (C20's >= 1 guard).
	const int64_t bias_mask = (int64_t{1} << r_cap) - 1;
	return (x >> r_cap) + ((x & bias_mask) != 0 ? 1 : 0);
}

int64_t ComputePeelGrid(int64_t unpeeled_max_abs, int64_t full_row_max_abs, int r_cap) {
	// §5 step 2: D'_grid = max(unpeeled_max_abs, CeilDivPow2(full_row_max_abs, r_cap)).
	const int64_t capped_term = CeilDivPow2(full_row_max_abs, r_cap);
	return unpeeled_max_abs > capped_term ? unpeeled_max_abs : capped_term;
}

RemedyStatus ComputePeeledCode(int64_t wide_value, int64_t r, int s, int64_t* out_c_star) {
	// §5 step 3: C22's composite (identical grid, tie rule, 128-bit intermediate) —
	// the SAME construction as RequantTokenCodeWide (intmath.cpp): magnitude =
	// floor((2*prod + 2^exponent) / 2^(exponent+1)), the doubled-numerator form of
	// round-half-away-from-zero — but kept at full 128-bit width rather than
	// narrowed and clamped to int8. §12's failure-path bullet: *out_c_star is left
	// untouched on rejection.
	const int exponent = 62 - s;
	const uint64_t abs_x = AbsMagnitude(wide_value);

	const Wide128 xr = WideMul64(abs_x, static_cast<uint64_t>(r));
	const Wide128 prod = WideMulSmall(xr, 127u);  // |wide_value| * 127 * R
	const Wide128 numerator = WideAddSmall(WideShl1(prod), uint64_t{1} << exponent);
	const Wide128 magnitude = WideShrFull(numerator, exponent + 1);

	// §12 boundary bullet / D-SLM1633/1663: evaluated on the UN-NARROWED 128-bit
	// magnitude, before any truncation to int64. The high word is tested FIRST —
	// the truncation-corner cell's own target — so a magnitude whose low 64 bits
	// happen to sit inside C_max is never accepted just because the high word was
	// never examined.
	if (WideHi(magnitude) != 0) return RemedyStatus::PeeledCodeMagnitudeOutOfDomain;
	const uint64_t lo = WideLo(magnitude);
	if (lo > static_cast<uint64_t>(kPeelCMax)) return RemedyStatus::PeeledCodeMagnitudeOutOfDomain;

	const int64_t signed_magnitude = static_cast<int64_t>(lo);
	*out_c_star = wide_value < 0 ? -signed_magnitude : signed_magnitude;
	return RemedyStatus::Ok;
}

RemedyStatus ReferencePeeledCode(int64_t wide_value, int64_t r, int s, int64_t* out_c_star) {
	// §11 A6-independent reference for step 3 (D-SLM1633): the same composite,
	// authored independently of ComputePeeledCode's own body. Where that function
	// rounds via a doubled numerator, this one rounds via an explicit
	// quotient/remainder/threshold split (the same technique RoundingDivideByPOTWide
	// already uses elsewhere in this codebase for an unrelated primitive) — a
	// genuinely different construction of the identical round-half-away-from-zero
	// rule, not the same code path called twice.
	const int exponent = 62 - s;
	const uint64_t abs_x = AbsMagnitude(wide_value);

	const Wide128 xr = WideMul64(abs_x, static_cast<uint64_t>(r));
	const Wide128 prod = WideMulSmall(xr, 127u);  // |wide_value| * 127 * R

	const Wide128 quotient = WideShrFull(prod, exponent);  // floor(prod / 2^exponent)
	// exponent is always < 64 in this design's domain (s comes from NormalizeScale,
	// exponent = 62 - s), so every remainder bit lives in prod's own low word.
	const uint64_t mask = (exponent >= 64) ? ~uint64_t{0} : ((uint64_t{1} << exponent) - 1u);
	const uint64_t remainder = WideLo(prod) & mask;
	const bool round_up = exponent > 0 && remainder >= ((mask >> 1) + 1u);
	const Wide128 magnitude = round_up ? WideIncrement(quotient) : quotient;

	if (WideHi(magnitude) != 0) return RemedyStatus::PeeledCodeMagnitudeOutOfDomain;
	const uint64_t lo = WideLo(magnitude);
	if (lo > static_cast<uint64_t>(kPeelCMax)) return RemedyStatus::PeeledCodeMagnitudeOutOfDomain;

	const int64_t signed_magnitude = static_cast<int64_t>(lo);
	*out_c_star = wide_value < 0 ? -signed_magnitude : signed_magnitude;
	return RemedyStatus::Ok;
}

bool IsPeelParamsAdmissible(int p, int r_cap) {
	// §6.3c / E13: P <= 7 at r_cap = 7 is admissible in this design's swept range;
	// P = 8 exceeds the residual consumer's executed headroom bound. This is the
	// swept-range RECTANGLE, sound only at the pinned n = 1536 (T-1834 F6) --
	// IsPeelParamsAdmissibleAtN below is the coupled, n-dependent region.
	return p >= 0 && p <= 7 && r_cap >= 0 && r_cap <= 7;
}

bool IsPeelParamsAdmissibleAtN(int p, int r_cap, int64_t n) {
	// T-1834 F6 remedy (§6.3c, D-SLM1896/1898): the coupled, n-DEPENDENT region
	// -- P*C_max^2 + (n-P)*127^2 <= 2^31 - 1, with C_max = 2^r_cap * 2^7 (the
	// checked peeled-code bound at that r_cap). IsPeelParamsAdmissible above is
	// a rectangle that never over-admits at the pinned n = 1536 this design
	// sweeps at, but its name promises this region and its two-argument
	// signature cannot express n's own effect on the bound (over-refuses at low
	// r_cap, under-refuses at n above the pin). Evaluated directly here in the
	// file's own Wide128 facility, so a large P or r_cap cannot silently wrap
	// the way this fold's other primitives did before their own remedy.
	if (p < 0 || r_cap < 0 || n < static_cast<int64_t>(p)) return false;

	const uint64_t remainder_channels = static_cast<uint64_t>(n) - static_cast<uint64_t>(p);
	const Wide128 remainder_term = WideMul64(remainder_channels, uint64_t{127u * 127u});

	Wide128 p_term{};  // zero-initialized; stays zero when p == 0 regardless of C_max
	if (p > 0) {
		// C_max = 2^r_cap * 2^7, formed via WideShlFull.
		//
		// T-1844 N1 (Poirot, 5899b8f): the claim this comment used to make --
		// "an r_cap large enough to push C_max (or C_max^2) past 64 bits
		// refuses here instead of silently wrapping" -- was false at
		// r_cap >= 121: WideShlFull is total against undefined behavior over
		// its full declared width (never invokes UB, for any k), but that is
		// NOT the same property as total against wrapping. At r_cap = 121,
		// `2^7 << 121` needs 128 bits and does not fit; the shift silently
		// drops the high bits, WideHi(c_max) reads zero on the WRAPPED value,
		// and both guards below pass an overflowed C_max as if it were small.
		// `WideShlFull`'s `out_truncated` parameter closes this at the root --
		// it reports whenever the shift itself lost bits, not merely whenever
		// the shifted RESULT'S high word happens to be non-zero, so this now
		// refuses for every r_cap that overflows the shift, all the way to
		// INT_MAX, not just up to whatever threshold a swept range happened
		// to check.
		bool c_max_truncated = false;
		const Wide128 c_max = WideShlFull(WideMul64(1u, 128u), r_cap, &c_max_truncated);
		if (c_max_truncated || WideHi(c_max) != 0) return false;
		const Wide128 c_max_sq = WideMul64(WideLo(c_max), WideLo(c_max));
		if (WideHi(c_max_sq) != 0) return false;
		p_term = WideMul64(WideLo(c_max_sq), static_cast<uint64_t>(p));
	}

	const Wide128 sum = WideAddWide(p_term, remainder_term);
	if (WideHi(sum) != 0) return false;
	return WideLo(sum) <= static_cast<uint64_t>((int64_t{1} << 31) - 1);
}

void ApplyPeelRankPFixup(int64_t* acc, size_t out_channels, const PeelRecord* records,
                          size_t record_count, const int8_t* weight, size_t in_channels) {
	// §5 step 4: for every output channel j, acc[j] += sum over the P peel records
	// of records[p].c_star * weight[records[p].index * out_channels + j] -- weight
	// is ROW-MAJOR [in_channels x out_channels]. acc is read-modify-write: it
	// already holds the bulk accumulate on entry.
	//
	// T-1834 F8 (D-SLM1896): in_channels is the one bound this function holds
	// alongside the index it guards; a record whose index is >= in_channels is
	// skipped rather than read out of the weight matrix. Given F5 can produce
	// exactly such an index at a caller that bypasses SelectPeelIndices' own
	// guard (a test, or a future direct PeelRecord construction), this closes
	// the hole F5's fix would otherwise still leave open here.
	for (size_t j = 0; j < out_channels; ++j) {
		int64_t sum = 0;
		for (size_t p = 0; p < record_count; ++p) {
			const size_t idx = records[p].index;
			if (idx >= in_channels) continue;
			sum += records[p].c_star * static_cast<int64_t>(weight[idx * out_channels + j]);
		}
		acc[j] += sum;
	}
}

}  // namespace superslm_t1822

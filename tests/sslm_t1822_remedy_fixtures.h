// Fixtures for the T-1822 activation-scale remedy red suite (T-1832, Curie).
//
// Every boundary constant below was derived by independently executing the exact
// composite formula RequantTokenCodeWide/ComputePeeledCode share (C22's
// (2*|x|*127*r + 2^exponent) >> (exponent+1), exponent = 62-s) in Python's arbitrary-
// precision integers — the same 128-bit-exact arithmetic, computed once outside the
// implementation under test, per StandardsDocument.md §5.4 (exactness verified at
// source or by execution, never by construction). The derivation is not re-stated
// per constant here; it is filed in
// Claude/Curie/t1832-activation-scale-remedy-red-suite-test-design-2026-08-08.md §6.
#ifndef SUPERSLM_TEST_T1822_REMEDY_FIXTURES_H
#define SUPERSLM_TEST_T1822_REMEDY_FIXTURES_H

#include <cstdint>

namespace superslm_test::t1822 {

// A representative "canonical" (r, s) pair: D'_grid = 1 (the all-peeled / D'=1 corner),
// so NormalizeScale(1) = {dn = 2^30, s = 30} and DynamicScaleReciprocal(2^30) = 2^32
// exactly (2^62 / 2^30 = 2^32, no rounding). Used by every boundary fixture below so
// each is a single-variable perturbation of the others.
inline constexpr int64_t kCanonicalR = 4294967296LL;  // 2^32
inline constexpr int kCanonicalS = 30;

// --- M2 peeled-code boundary fixtures (§12 boundary bullet, §5 step 3) -----------

// magnitude(x_i, r=1, s=30) == 16384 exactly (C_max, the accept-side edge). Uses r = 1
// (not kCanonicalR) because kCanonicalR's step size per unit x_i (~127) skips most
// integers near this magnitude — r = 1 is still a totally in-contract int64 reciprocal
// input for this boundary-only fixture, chosen purely so the boundary lands exactly.
inline constexpr int64_t kPeelAtCMaxXi = 554067690505LL;
inline constexpr int64_t kPeelAtCMaxR = 1;
inline constexpr int kPeelAtCMaxS = 30;
inline constexpr int64_t kPeelAtCMaxExpectedMagnitude = 16384;  // == kPeelCMax

// magnitude(x_i, r=1, s=30) == 16385 exactly (C_max + 1, the reject-side edge).
inline constexpr int64_t kPeelAtCMaxPlus1Xi = 554101509145LL;
inline constexpr int64_t kPeelAtCMaxPlus1R = 1;
inline constexpr int kPeelAtCMaxPlus1S = 30;
inline constexpr int64_t kPeelAtCMaxPlus1ExpectedMagnitude = 16385;

// The 64-bit totality extreme (§12: "a composite near 2^38 — the C29-boundary corner
// D' = 2^31 at D'_grid = 1"): x_i = 2^31 (C29's own legal maximum), (r, s) = the D'=1
// grid pair. True magnitude = 272,730,423,296 (~2^38.0), high 64 bits zero, low word
// far past kPeelCMax — rejected via the LOW-WORD test.
inline constexpr int64_t kPeelTotalityExtremeXi = 2147483648LL;  // 2^31
inline constexpr int64_t kPeelTotalityExtremeR = kCanonicalR;
inline constexpr int kPeelTotalityExtremeS = kCanonicalS;
inline constexpr int64_t kPeelTotalityExtremeExpectedMagnitude = 272730423296LL;

// The truncation corner (§12: "an int64-legal input whose 128-bit composite is ≈ 2^65
// and whose truncated low word would be 123, inside C_max"). x_i, (r, s) = the D'=1
// grid pair. True 128-bit magnitude = 2^64 + 125 — high 64 bits are 1 (non-zero), low
// word is 125 (< kPeelCMax). A guard that tests only the narrowed-to-int64 low word
// (dropping the high 64 bits) sees 125, "inside C_max", and wrongly accepts; the
// correct guard tests the un-narrowed 128-bit magnitude's high word first and rejects.
// x_i itself is 58 bits wide, well inside int64 ("int64-legal").
inline constexpr int64_t kPeelTruncationCornerXi = 145249953336295683LL;
inline constexpr int64_t kPeelTruncationCornerR = kCanonicalR;
inline constexpr int kPeelTruncationCornerS = kCanonicalS;
inline constexpr uint64_t kPeelTruncationCornerExpectedHi64 = 1;
inline constexpr uint64_t kPeelTruncationCornerExpectedLo64 = 125;

// --- Small M1/M2 rows for the differential and degenerate cells -----------------

// A 32-channel row (G = 32 divides both 1536 and 8,960's... actually not 8,960; this
// fixture is width-agnostic — it is one group's own data, not a full row) with an
// ordinary, non-corner spread: no channel at 0, no tie, row max held by index 5.
inline constexpr int64_t kOrdinaryGroup[32] = {
	1000, 800, -1200, 300, 50, -2000000, 700, -650, 900, 1100,
	200, -300, 400, -900, 1234, -55, 66, 77, -88, 99,
	111, -222, 333, -444, 555, -666, 777, -888, 999, -1000,
	123, -321,
};
inline constexpr size_t kOrdinaryGroupN = 32;
inline constexpr int64_t kOrdinaryGroupMaxAbs = 2000000;  // index 5

// A small group calibrated to leave REAL headroom under the int8 rail across every
// k_g in [0, 3] against a row max of 800 -- unlike kOrdinaryGroup at kCanonicalR/
// kCanonicalS (the D'_grid=1 scale, deliberately the FINEST grid, for the C_max/
// totality/truncation boundary fixtures above), this group's own scale is derived
// from ITS OWN row max, matching how a real funnel would set it. Independently
// verified (Python, exact integer arithmetic, transcribing the same composite
// formula as §6): with (r, s) = NormalizeScale(800)/DynamicScaleReciprocal, at
// k_g in {0,1,2,3} no channel's magnitude reaches +-127 (codes at k_g=3:
// [114, 38, -76, 13]); at k_g=4 -- one past the k_cap=3 this group's own ratio
// (800/90 ~ 8.9) admits -- two of the four channels legitimately overshoot and are
// clamped to +-127 exactly (codes: [127, 76, -127, 25]). This is the fixture the
// retained-clamp-engagement cell (§12 contract bullet, D-SLM1615) and the
// C22-kinship-agreement cell (D-SLM1663) both need: a scale where "does the clamp
// engage" is actually discriminating, rather than either always-true (every
// nonzero value saturates) or always-false (nothing ever approaches the rail).
inline constexpr int64_t kClampMarginGroup[4] = {90, 30, -60, 10};
inline constexpr size_t kClampMarginGroupN = 4;
inline constexpr int64_t kClampMarginGroupMaxAbs = 90;
inline constexpr int64_t kClampMarginRowMax = 800;

// An all-zero group (D'_g's own >= 1 guard fires; k_g's admissibility predicate must
// still be well-defined at D'_g == 1).
inline constexpr int64_t kAllZeroGroup[8] = {0, 0, 0, 0, 0, 0, 0, 0};
inline constexpr size_t kAllZeroGroupN = 8;
inline constexpr int64_t kAllZeroGroupMaxAbs = 1;  // C20's guard: max(reduce, 1)

// A row where two channels tie at the same magnitude, opposite sign, for the peel
// selection's lowest-index tie-break cell.
inline constexpr int64_t kPeelTieRow[6] = {10, -500, 500, 20, -30, 40};
inline constexpr size_t kPeelTieRowN = 6;
// index 1 (-500) and index 2 (500) tie at magnitude 500; lowest-index tie-break
// selects index 1 first.

// Two rows with different outlier identity, same shape otherwise — the discrimination
// cell (audit F9): the emitted peel index sets must differ between them.
inline constexpr int64_t kDiscriminationRowA[6] = {5000, 10, 20, 30, 40, 50};
inline constexpr int64_t kDiscriminationRowB[6] = {10, 20, 30, 40, 50, 5000};
inline constexpr size_t kDiscriminationRowN = 6;

}  // namespace superslm_test::t1822

#endif  // SUPERSLM_TEST_T1822_REMEDY_FIXTURES_H

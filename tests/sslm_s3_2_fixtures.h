// GENERATED FILE. Do not hand-edit.
//
// Produced by tests/gen_s3_2_fixtures.py. Every witness below is derived by
// EXECUTING the vendored reference (tests/reference/superslm_spike/intmath.py)
// or by arithmetic checked against it -- never copied from a probe, never
// hand-computed.
//
// T-1657 Poirot review, Observation 1 (corrected): this file's own preamble
// used to state that S3.2's forward-composition entry points (the RMSNorm
// site, the WSC1 fold-apply, the C28 bias-reconciliation site, BIA1's
// value-domain descriptor) "do not exist in D:\\SuperSLM" -- true only at the
// S3.2 red-phase authoring of this generator, and false since the green
// phase (src/forward/forward_sites.cpp, src/forward/checked_chain_funnel.cpp).
// The witnesses below are consumed by tests/test_main.cpp's own S3.2/T-1657
// cells against those real bodies, not staged for a future build.
//
// Re-running this script must reproduce this file byte-for-byte.
//
// Test-design record:
// Claude/Curie/superslm-s3.2-weightless-and-projection-sites-test-design-2026-07-28.md
#ifndef SUPERSLM_TESTS_SSLM_S3_2_FIXTURES_H
#define SUPERSLM_TESTS_SSLM_S3_2_FIXTURES_H

#include <cstddef>

namespace superslm_test {

// --- C31 (Sec5.1) unit cell: F-S3-2's own three-row witness table ---------
struct C31UnitCase {
	long long hidden;
	long long roots;
	long long floor_q;   // the pinned (correct) floor-division quotient
	long long trunc_q;   // the naive-C++ (wrong) truncating quotient -- negative control
	bool differs;        // floor_q != trunc_q
};

inline constexpr C31UnitCase kC31UnitCases[] = {
	{ /*hidden=*/-5LL, /*roots=*/7LL, /*floor_q=*/-3067833783LL, /*trunc_q=*/-3067833782LL, /*differs=*/true },
	{ /*hidden=*/-127LL, /*roots=*/3LL, /*floor_q=*/-181820282198LL, /*trunc_q=*/-181820282197LL, /*differs=*/true },
	{ /*hidden=*/-1LL, /*roots=*/2LL, /*floor_q=*/-2147483648LL, /*trunc_q=*/-2147483648LL, /*differs=*/false },
};

inline constexpr size_t kC31UnitCasesCount = sizeof(kC31UnitCases) / sizeof(kC31UnitCases[0]);

// --- C31 (Sec5.1) site cell: a reachable (h, root) witness at H=1536 -------
struct C31SiteElement {
	int index;
	int h;
	int g;
	long long floor_wide;  // the pinned (correct) floor-based wide value
	long long trunc_wide;  // the naive-C++ (wrong) truncating wide value -- negative control
	bool diverges;
};

inline constexpr int kC31SiteHiddenSize = 1536;
inline constexpr long long kC31SiteRoot = 237956LL;
inline constexpr long long kC31SiteSumSq = 20250LL;
inline constexpr C31SiteElement kC31SiteElements[] = {
	{ /*index=*/0, /*h=*/-5, /*g=*/100, /*floor_wide=*/-9024800LL, /*trunc_wide=*/-9024700LL, /*diverges=*/true },
	{ /*index=*/1, /*h=*/64, /*g=*/-30, /*floor_wide=*/-34654860LL, /*trunc_wide=*/-34654860LL, /*diverges=*/false },
	{ /*index=*/17, /*h=*/-127, /*g=*/5, /*floor_wide=*/-11461385LL, /*trunc_wide=*/-11461380LL, /*diverges=*/true },
};

inline constexpr size_t kC31SiteElementsCount = sizeof(kC31SiteElements) / sizeof(kC31SiteElements[0]);

// --- C24 identity/near-identity discrimination (the plan's own two-element row) ---
inline constexpr long long kC24RefChannelPassThrough = 1073741825LL;
inline constexpr long long kC24RefChannelNearIdentity = 1073741824LL;  // wrong: off by one
inline constexpr long long kC24SharedElementX = 215593831LL;
inline constexpr int kC24CodeRefPassThrough = 127;
inline constexpr int kC24CodeXPassThrough = 25;
inline constexpr int kC24CodeRefNearIdentity = 127;
inline constexpr int kC24CodeXNearIdentity = 26;  // differs from kC24CodeXPassThrough via the shared D'

// --- BIA1 (Sec7.2a) magnitude descriptor -----------------------------------
inline constexpr long long kBia1HostileValue = 2147483648LL;  // one past the FORMER bound -- must now accept
inline constexpr long long kBia1HostileValueNegated = -2147483648LL;
inline constexpr long long kBia1AcceptBoundaryValue = 2147483647LL;  // == INT32_MAX, the former bound itself -- must accept

// --- C28 (Sec7.2 2nd limb) (q_B, e_a) domain boundary ----------------------
inline constexpr long long kC28BoundaryBelowMinEA = -93LL;
inline constexpr int kC28BoundaryBelowMinK = -1;
inline constexpr bool kC28BoundaryBelowMinInDomain = false;
inline constexpr long long kC28BoundaryAtMinEA = -92LL;
inline constexpr int kC28BoundaryAtMinK = 0;
inline constexpr bool kC28BoundaryAtMinInDomain = true;
inline constexpr long long kC28BoundaryAtMaxEA = -29LL;
inline constexpr int kC28BoundaryAtMaxK = 63;
inline constexpr bool kC28BoundaryAtMaxInDomain = true;
inline constexpr long long kC28BoundaryAboveMaxEA = -28LL;
inline constexpr int kC28BoundaryAboveMaxK = 64;
inline constexpr bool kC28BoundaryAboveMaxInDomain = false;

// --- C28 (Sec4.4) bias-reconciliation sign-inverted tie witness ------------
inline constexpr long long kC28TieB = 1LL;
inline constexpr long long kC28TieQB = 30LL;
inline constexpr long long kC28TieRA = 2147483649LL;
inline constexpr long long kC28TieEA = -91LL;
inline constexpr int kC28TieK = 1;
inline constexpr long long kC28TieCorrectPos = 1073741825LL;
inline constexpr long long kC28TieCorrectNeg = -1073741825LL;
inline constexpr long long kC28TieWrongPos = 1073741825LL;  // wrong candidate; must equal kC28TieCorrectPos
inline constexpr long long kC28TieWrongNeg = -1073741824LL;  // wrong candidate; must NOT equal kC28TieCorrectNeg

// --- BIA1 widened core (T-1657, Sec10 cell 1): bit-exact at a magnitude
// --- the deleted load-time bound used to reject -----------------------------
inline constexpr long long kBia1WideB = 99381983436726LL;
inline constexpr long long kBia1WideQB = 30LL;
inline constexpr long long kBia1WideRA = 4294967296LL;
inline constexpr long long kBia1WideEA = -52LL;
inline constexpr int kBia1WideExponent = 40;
inline constexpr long long kBia1WideExpected = 388210872800LL;

// --- BIA1 widened core (T-1657, Sec10 cell 2): representability boundary ---
inline constexpr long long kBia1WideBoundaryQB = 30LL;
inline constexpr long long kBia1WideBoundaryEA = -92LL;
inline constexpr int kBia1WideBoundaryExponent = 0;
inline constexpr long long kBia1WideBoundaryFitsB = 9223372036854775807LL;
inline constexpr long long kBia1WideBoundaryFitsRA = 1LL;
inline constexpr long long kBia1WideBoundaryFitsExpected = 9223372036854775807LL;  // == INT64_MAX
inline constexpr long long kBia1WideBoundaryOverB = 4611686018427387904LL;
inline constexpr long long kBia1WideBoundaryOverRA = 2LL;
// kBia1WideBoundaryOverExpected == INT64_MAX + 1, one past int64_t -- not representable as a literal

// --- BIA1 widened core (T-1657, Sec10 cell 3): the strike's own raw-product-
// --- overflow, result-in-range discriminating witness -----------------------
inline constexpr long long kBia1WideStrikeB = 2147483647LL;  // == kBia1AcceptBoundaryValue (INT32_MAX)
inline constexpr long long kBia1WideStrikeQB = 30LL;
inline constexpr long long kBia1WideStrikeRA = 48507865471LL;
inline constexpr long long kBia1WideStrikeEA = -30LL;
inline constexpr int kBia1WideStrikeExponent = 62;
inline constexpr long long kBia1WideStrikeExpected = 23LL;

// --- BIA1 widened core (T-1657, Sec10 cell 8): a genuine C3 tie forced onto
// --- the 128-bit path, both signs --------------------------------------------
inline constexpr long long kBia1WideTieB = 1099511628032LL;
inline constexpr long long kBia1WideTieQB = 30LL;
inline constexpr long long kBia1WideTieRA = 2147483649LL;
inline constexpr long long kBia1WideTieEA = -83LL;
inline constexpr int kBia1WideTieK = 9;
inline constexpr long long kBia1WideTieCorrectPos = 4611686021648613377LL;
inline constexpr long long kBia1WideTieCorrectNeg = -4611686021648613377LL;
inline constexpr long long kBia1WideTieWrongPos = 4611686021648613377LL;  // wrong candidate; must equal kBia1WideTieCorrectPos
inline constexpr long long kBia1WideTieWrongNeg = -4611686021648613376LL;  // wrong candidate; must NOT equal kBia1WideTieCorrectNeg

}  // namespace superslm_test

#endif  // SUPERSLM_TESTS_SSLM_S3_2_FIXTURES_H

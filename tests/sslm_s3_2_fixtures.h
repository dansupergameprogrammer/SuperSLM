// GENERATED FILE. Do not hand-edit.
//
// Produced by tests/gen_s3_2_fixtures.py. Every witness below is derived by
// EXECUTING the vendored reference (tests/reference/superslm_spike/intmath.py)
// or by arithmetic checked against it -- never copied from a probe, never
// hand-computed. None of these structs has a production consumer yet: S3.2's
// forward-composition entry points (the RMSNorm site, the WSC1 fold-apply, the
// C28 bias-reconciliation site, BIA1's value-domain descriptor) do not exist in
// D:\SuperSLM (verified by grep, recorded in the test-design record). This
// header exists so the moment Brunel's header contract lands, the S3.2 red
// cells drop in directly against these witnesses -- no further derivation owed.
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
inline constexpr long long kBia1RaMax = 4294967296LL;
inline constexpr long long kBia1MagnitudeBound = 2147483647LL;  // == INT32_MAX
inline constexpr long long kBia1HostileValue = 2147483648LL;  // one past the bound -- must reject
inline constexpr long long kBia1HostileValueNegated = -2147483648LL;
inline constexpr long long kBia1AcceptBoundaryValue = 2147483647LL;  // exactly at the bound -- must accept

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

}  // namespace superslm_test

#endif  // SUPERSLM_TESTS_SSLM_S3_2_FIXTURES_H

// T-2296 (Curie) -- Coverage Model dimension 4 (Shape and platform matrices), cross-reference
// stub. Design of record: Claude/Vitruvius/t2265-superslm-fp-free-open-design-2026-08-24.md
// Sec7 dim 4's own closing paragraph (fold round 22, coverage audit round 6 Finding G13):
// "The CI matrix's own toolchain axis does not vary MSVC point release... The two cells this
// creates -- one must-ship, one not-yet-applicable -- are stated in full at dimension 6 below
// (Sec4.2/Sec4.3's own home), cross-referenced here rather than duplicated."
//
// This file exists so the suite's own dimension-by-dimension accounting (this campaign's test-
// design record, Claude/Curie/t2296-fp-free-open-red-suite-test-design-2026-08-26.md) has one
// file per Coverage Model dimension to point at, matching this repo's own established suite
// layout (tests/t2138-abi-red-suite, tests/t2130-g5-red-suite). It authors no test of its own --
// the cells it would otherwise carry are dim6_determinism_red.cpp's Cell A/Cell B.
//
// Dimension 4's OTHER cells -- sites 1-7's zero/one/many-element artifact-count legs, the
// n==0/Clz64-domain mutation-proof cell, and the toolchain/ISA matrix sweep over BucketCountFor's
// bit-shift sizing -- are NOT authored in this session; see this campaign's test-design record
// for the explicit scope decision and reason (they require the real replacement types built and,
// for the toolchain/ISA sweep, the 29-job CI matrix itself, neither of which exists yet).

#include "fixture_common.h"

int main(int, char**) {
	std::printf("=== dim4_shape_red: Coverage Model dimension 4 ===\n");
	SKIP_MSG(
	    "dimension 4's toolchain-axis gap is cross-referenced to dim6_determinism_red.cpp's own "
	    "Cell A/Cell B per the design's own text (Sec7 dim 4) -- no separate cell authored here, "
	    "matching the design's own cross-reference-rather-than-duplicate convention.");
	SKIP_MSG(
	    "dimension 4's own zero/one/many-element artifact-count legs and the n==0/Clz64-domain "
	    "mutation-proof cell (BucketCountFor's own debug assert(need >= 2)): NOT authored this "
	    "session -- requires the real FixedIntMap/FixedIntSet types and a debug-assert-enabled "
	    "build; routed as a follow-on in this campaign's test-design record.");
	std::printf("\n%d checks, %d failures, %d skips.\n", GChecks, GFailures, GSkips);
	return GFailures == 0 ? 0 : 1;
}

// T-2296 (Curie) -- Coverage Model dimension 6 (Numerical edges and determinism), Cell A:
// the float-representability-boundary construction. Design of record:
// Claude/Vitruvius/t2265-superslm-fp-free-open-design-2026-08-24.md Sec4.2 leg (b) / Sec7 dim 6
// Cell A (fold round 22, coverage audit round 6 Finding G13).
//
// THE CLAIM UNDER TEST: on the shipped pre-remedy std::unordered_map/std::unordered_set types,
// reserve(n) for an n NOT exactly representable as a 32-bit float (e.g. 16,777,217 = 2^24 + 1)
// TRAPS under the masks-cleared state on every toolchain -- the boundary is a language-standard
// fact (IEEE-754 float representability), not an implementation choice, so it is guaranteed to
// force the pre-remedy trap regardless of which MSVC point release or non-MSVC standard library
// a given CI job runs (contrast dimension 6 Cell B, deliberately NOT built here -- see the
// disposition note at the bottom of this file). Once Sec3's replacement lands, the identical
// numeric value routed through FixedIntMap::Init/FixedIntSet::Init -- BucketCountFor's own
// bit-shift sizing, integer-only, Sec3.1 -- must NOT trap, because there is no floating-point
// divide left on the path for representability to matter to.
//
// CONTROL LEG, this file's own addition beyond the design's literal text: reserve(16,777,216 =
// 2^24, EXACTLY representable) on the same pre-remedy types -- proving the boundary really is
// representability, not "any seven-digit reserve() traps." Without this leg, a regression that
// made EVERY reserve() trap (a much coarser defect) would still pass a trap-only assertion.
//
// EXECUTED FINDING (this suite's own authoring session, not previously recorded at this exact
// cell): the control leg's own expected outcome is ITSELF toolchain-family-conditioned, exactly
// the dimension 4/dimension 6 gap the design names ("the CI matrix's own toolchain axis does not
// vary MSVC point release"). Re-running Claude/Loki/t2284-probe/threshold.cpp VERBATIM on this
// same machine, through the VS2022 Community install's own VsDevCmd.bat (MSVC 19.33.31631,
// confirmed via `cl` banner) rather than the BuildTools install (19.44.35214) every prior probe
// in this design's fold history used, found EVERY tested n from 1 to 100,000,000 traps --
// matching fold round 21's own "categorical, not a shifted threshold" 19.33 finding exactly
// (Sec1's own text, out-threshold-1933.txt), not the representability-boundary-only 19.44
// finding this file's control leg originally assumed unconditionally. The split below is
// _MSC_VER-conditioned rather than hard-coded to one toolchain, so this cell reports its actual
// finding on whichever toolchain builds it instead of silently asserting the wrong expectation.
//
// EXECUTION MODEL: every leg below runs in a CHILD process (fixture_common.h's RunChildMode) --
// a genuine SEH/hardware trap ends the child, never the test binary itself, mirroring
// Claude/Loki/t2284-probe/threshold.cpp's own parent/child construction (attribution stated,
// StandardsDocument.md Sec4/Sec7 -- this file's own child-mode dispatch and CHECK harness are an
// independent construction, not a copy of that probe's text).
//
// SELF-CHECK PERFORMED, this authoring session: the design's exact printed Sec3.1 text was
// assembled into a throwaway scratch header, never touching the engine tree, and this file was
// compiled and run against it unmodified -- all 6 CHECKs passed (0 failures), including the two
// post-remedy legs (FixedIntMap::Init/FixedIntSet::Init at the non-representable boundary,
// confirmed NOT to trap). This validates the test's own logic against the design's own
// construction before handing it to Brunel; it is not a substitute for the build-time re-run
// against the real src/detail/int_hash.h, which remains owed.
//
// RED STATE, as authored (nothing built): the __has_include guard below routes every
// post-remedy leg to SKIP with an explicit "NOT YET BUILDABLE" message -- src/detail/int_hash.h
// does not exist yet (Sec3.1, Sec10: "Nothing here is built"). The pre-remedy legs (both TRAP
// and the representable-boundary NO-TRAP control) compile, link, and run against the REAL
// shipped std::unordered_map/std::unordered_set today and are the suite's genuine red evidence:
// this cell fails as a whole (SUITE STATUS: RED) until the post-remedy leg exists and passes.

#include "fixture_common.h"

#include <unordered_map>
#include <unordered_set>

#if __has_include("detail/int_hash.h")
#define T2296_HAVE_INT_HASH_H 1
#include "detail/int_hash.h"
#else
#define T2296_HAVE_INT_HASH_H 0
#endif

namespace {

// n = 2^24 + 1 -- the smallest integer NOT exactly representable as a 32-bit float, T-2284's own
// pinned boundary (Claude/Loki/t2284-probe/out-threshold.txt, re-confirmed fold round 21).
constexpr unsigned long long kNonRepresentableN = 16777217ull;
// n = 2^24 -- the largest power-of-two-adjacent integer that IS exactly representable, the
// control leg's own value (one less than the boundary above).
constexpr unsigned long long kRepresentableN = 16777216ull;

int RunChild(const std::string& mode) {
	if (mode == "pre_map_trap") {
		ClearAllMxcsrExceptionMasks();
		std::unordered_map<uint64_t, uint64_t> m;
		m.reserve(static_cast<size_t>(kNonRepresentableN));
		return 0;  // reaching here means it did NOT trap
	}
	if (mode == "pre_set_trap") {
		ClearAllMxcsrExceptionMasks();
		std::unordered_set<uint64_t> s;
		s.reserve(static_cast<size_t>(kNonRepresentableN));
		return 0;
	}
	if (mode == "pre_map_control_notrap") {
		ClearAllMxcsrExceptionMasks();
		std::unordered_map<uint64_t, uint64_t> m;
		m.reserve(static_cast<size_t>(kRepresentableN));
		return 0;
	}
	if (mode == "pre_set_control_notrap") {
		ClearAllMxcsrExceptionMasks();
		std::unordered_set<uint64_t> s;
		s.reserve(static_cast<size_t>(kRepresentableN));
		return 0;
	}
#if T2296_HAVE_INT_HASH_H
	if (mode == "post_map_notrap") {
		ClearAllMxcsrExceptionMasks();
		superslm::detail::FixedIntMap<uint64_t, uint64_t, superslm::detail::MixKey64> m;
		m.Init(kNonRepresentableN);
		return 0;
	}
	if (mode == "post_set_notrap") {
		ClearAllMxcsrExceptionMasks();
		superslm::detail::FixedIntSet<uint64_t, superslm::detail::MixKey64> s;
		s.Init(kNonRepresentableN);
		return 0;
	}
#endif
	std::fprintf(stderr, "unknown child mode: %s\n", mode.c_str());
	return 2;
}

}  // namespace

int main(int argc, char** argv) {
	std::string mode;
	if (ParseChildMode(argc, argv, &mode)) {
		return RunChild(mode);
	}
	SetSelfPathFromModule();

	std::printf("=== dim6_determinism_red: Coverage Model dimension 6 Cell A ===\n");

	// --- Pre-remedy: the shipped std::unordered_map/std::unordered_set boundary trap. Real,
	//     executable, RED-CONFIRMING today (this is the defect the design exists to remove). ---
	{
		ChildResult r = RunChildMode("pre_map_trap");
		CHECK_MSG(r == ChildResult::kCrashed,
		          "pre-remedy std::unordered_map::reserve(%llu) [not float-representable] under "
		          "masks-cleared: expected a hardware trap (the defect this design removes), got %s",
		          kNonRepresentableN, ChildResultName(r));
	}
	{
		ChildResult r = RunChildMode("pre_set_trap");
		CHECK_MSG(r == ChildResult::kCrashed,
		          "pre-remedy std::unordered_set::reserve(%llu) [not float-representable] under "
		          "masks-cleared: expected a hardware trap, got %s",
		          kNonRepresentableN, ChildResultName(r));
	}

	// --- Control: the SAME operation at the exactly-representable neighbour. Expected outcome is
	//     ITSELF toolchain-family-conditioned (this file's own header comment, executed finding):
	//     the 19.3x family traps categorically (every n); 19.40+ traps only at the
	//     non-representable boundary. _MSC_VER >= 1940 is this cell's own split point, matching
	//     the two families the design's fold round 21 measured (19.33 vs 19.44). ---
#if defined(_MSC_VER) && _MSC_VER >= 1940
	std::printf("(compiled with _MSC_VER=%d -- expecting the 19.40+ family's boundary-only "
	            "trap behaviour for the control leg)\n", _MSC_VER);
	{
		ChildResult r = RunChildMode("pre_map_control_notrap");
		CHECK_MSG(r == ChildResult::kExitedZero,
		          "control: std::unordered_map::reserve(%llu) [exactly float-representable] under "
		          "masks-cleared, _MSC_VER=%d (19.40+ family): expected normal return, got %s -- "
		          "if this traps too, the boundary is not representability on this toolchain and "
		          "Cell A's own premise needs re-deriving for it",
		          kRepresentableN, _MSC_VER, ChildResultName(r));
	}
	{
		ChildResult r = RunChildMode("pre_set_control_notrap");
		CHECK_MSG(r == ChildResult::kExitedZero,
		          "control: std::unordered_set::reserve(%llu) [exactly float-representable] under "
		          "masks-cleared, _MSC_VER=%d (19.40+ family): expected normal return, got %s",
		          kRepresentableN, _MSC_VER, ChildResultName(r));
	}
#elif defined(_MSC_VER)
	std::printf("(compiled with _MSC_VER=%d -- expecting the pre-19.40 family's categorical "
	            "trap behaviour for the control leg: EVERY reserve() traps, this executed "
	            "session's own finding re-running Claude/Loki/t2284-probe/threshold.cpp verbatim "
	            "on MSVC 19.33.31631)\n", _MSC_VER);
	{
		ChildResult r = RunChildMode("pre_map_control_notrap");
		CHECK_MSG(r == ChildResult::kCrashed,
		          "control: std::unordered_map::reserve(%llu) [exactly float-representable] under "
		          "masks-cleared, _MSC_VER=%d (pre-19.40 family): expected a hardware trap (this "
		          "family traps categorically, T-2284/fold round 21), got %s",
		          kRepresentableN, _MSC_VER, ChildResultName(r));
	}
	{
		ChildResult r = RunChildMode("pre_set_control_notrap");
		CHECK_MSG(r == ChildResult::kCrashed,
		          "control: std::unordered_set::reserve(%llu) [exactly float-representable] under "
		          "masks-cleared, _MSC_VER=%d (pre-19.40 family): expected a hardware trap, got %s",
		          kRepresentableN, _MSC_VER, ChildResultName(r));
	}
#else
	SKIP_MSG("control leg needs an _MSC_VER-bearing compiler to know which trap family to "
	         "expect; this build defines no _MSC_VER.");
#endif

	// --- Post-remedy: FixedIntMap/FixedIntSet must NOT trap at the identical n, because
	//     BucketCountFor is integer-only end to end (Sec3.1) -- representability of n as a float
	//     is irrelevant once no float divide exists on the path. THIS is the leg that must ship
	//     green in CI (Sec4.2: "the one leg that ships in CI"); it is red-unimplemented today. ---
#if T2296_HAVE_INT_HASH_H
	{
		ChildResult r = RunChildMode("post_map_notrap");
		CHECK_MSG(r == ChildResult::kExitedZero,
		          "post-remedy superslm::detail::FixedIntMap::Init(%llu) under masks-cleared: "
		          "expected normal return (integer-only sizing, Sec3.1), got %s",
		          kNonRepresentableN, ChildResultName(r));
	}
	{
		ChildResult r = RunChildMode("post_set_notrap");
		CHECK_MSG(r == ChildResult::kExitedZero,
		          "post-remedy superslm::detail::FixedIntSet::Init(%llu) under masks-cleared: "
		          "expected normal return, got %s",
		          kNonRepresentableN, ChildResultName(r));
	}
#else
	SKIP_MSG(
	    "post-remedy leg NOT YET BUILDABLE -- src/detail/int_hash.h does not exist "
	    "(design Sec3.1, Sec10: \"Nothing here is built\"). This is the red-unimplemented half "
	    "of dimension 6 Cell A: once Brunel adds src/detail/int_hash.h with FixedIntMap/"
	    "FixedIntSet as printed in the design, this __has_include gate flips on and the two "
	    "post_map_notrap/post_set_notrap CHECKs above become load-bearing without editing this "
	    "file.");
	++GFailures;  // a cell that cannot even be attempted is not a pass -- count it red, loudly,
	              // not silently absorbed into the SKIP tally alone.
#endif

	// --- Dimension 6 Cell B disposition (NOT a test authored here) -----------------------------
	// Design Sec7 dim 6 Cell B: the engine's own REAL declared types and realistic
	// artifact-driven populations (merges at 151,936; the three seen_names sites at
	// 400/400/64), trapping pre-remedy ONLY on an MSVC 19.3x-class toolset -- "Not applicable to
	// any CI job today, because no pinned job in the 29-job matrix (Sec2.5, Sec5.5) builds on
	// that toolset family; becomes applicable the day a CI job's toolchain is pinned to an MSVC
	// 19.3x-class release." This suite carries no test for Cell B, per that disposition, stated
	// by the design itself -- authoring one now would exercise a toolchain no CI job runs, which
	// StandardsDocument.md Sec5.4's commissioning rule names directly: "a construction exercising
	// a rendering the production path cannot produce establishes nothing about the readings the
	// instrument will actually emit." Routed back to the CI-image-pinning owner, not this suite's
	// authority to build.

	std::printf("\n%d checks, %d failures, %d skips.\n", GChecks, GFailures, GSkips);
	return GFailures == 0 ? 0 : 1;
}

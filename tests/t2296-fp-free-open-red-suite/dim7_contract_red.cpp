// T-2296 (Curie) -- Coverage Model dimension 7 (Contract claims), fifth cell-group: the
// pre-Init/double-Init misuse and capacity/population-desync must-reject contract. Design of
// record: Claude/Vitruvius/t2265-superslm-fp-free-open-design-2026-08-24.md Sec7 dim 7's fifth
// cell-group (fold round 23, coverage audit round 7 Finding G14), cross-referenced from
// dimension 11 (Guard vitality) per the design's own text.
//
// THE GUARDS UNDER TEST (Sec3.1, printed C++ text): FixedIntMap/FixedIntSet's `ready_` one-shot
// guard (`if (!ready_) std::abort();` on every mutating/read method before Init() has run;
// `if (ready_) std::abort();` inside Init() itself on a second call) and the bounded-probe
// exhaustion guard every one of the five replacement types carries (`std::abort()` when a
// capacity_hint/population desync makes the linear probe exhaust every slot without finding a
// match or an empty slot). "Nothing has ever proven any of these guards fires" -- the finding
// this cell-group exists to close (Sec7 dim 7 fifth cell-group's own header line).
//
// CELL A -- must-reject, pre-Init/double-Init misuse. The design's own text: "For each of
// FixedIntMap/FixedIntSet (sites 1-7, the only two types with a ready_ guard): construct a
// default instance and call each of Insert/Find/InsertOrAssign (map) or InsertUnique (set)
// before any Init() call, asserting the process aborts; separately, call Init() twice on one
// instance and assert the second call aborts." That is SIX constructions as literally described
// (three methods x pre-Init for FixedIntMap, one method x pre-Init for FixedIntSet, one
// Init-twice case per type) -- the design's own count ("Eight constructions total (four methods
// x pre-Init, one Init-twice case per type)") does not arithmetically match its own prose (3+1
// pre-Init cases plus 2 Init-twice cases = 6, not 8). Filed as a minor documentation-arithmetic
// finding in this suite's own test-design record (Claude/Curie/t2296-fp-free-open-red-suite-test-
// design-2026-08-26.md) rather than silently matched by inventing two more constructions the
// prose does not describe or silently "corrected" in the design itself, which this seat does not
// own (StandardsDocument.md Sec6.4). All six constructions the prose DOES describe are built and
// asserted below.
//
// CELL B -- must-reject, capacity/population desync, mutation-proof. The design names "each of
// the five replacement types." Built here for the TWO fixed-capacity types (FixedIntMap,
// FixedIntSet) only: Init() with a small capacity_hint fixes the slot count forever, so
// inserting more DISTINCT keys than that capacity allows is reachable through the PUBLIC API
// alone (Init(1) sizes to 2 slots via BucketCountFor; a 3rd distinct key exhausts the bounded
// probe). The three GROWABLE types (GrowableIntSet, GrowableIntMap, GrowableContextMap) check
// their own load factor and call Grow() BEFORE every insert (Sec3.5/Sec3.6's own printed
// `if ((live_[+tombstones_]+1)*2 > slots_.size()) Grow();"), which keeps the table at <=50%
// occupied after every public-API insert -- there is no sequence of InsertOrReclaim/operator[]/
// FindOrEmplace calls that leaves the table 100% full, so the bounded-probe-exhaustion state
// these three types' own Contains/Erase/Find/operator[] guard against is NOT reachable through
// the public surface the design specifies. This is a SPECIFICATION GAP, not a cell this suite
// declines to build: the design names all five types for Cell B but states no construction
// (public-API sequence, test-only hook, or otherwise) that reaches the state for the three
// growable types, and reaching into their private slots_/live_/tombstones_/mask_ fields from a
// test file would require adding a friend/backdoor to the production header -- writing
// implementation, which this seat does not do (Curie.md, "Does not implement"). Routed back to
// the design (Claude/Vitruvius) rather than papered over with a reflective/UB construction that
// would look like coverage without being a genuine test of the public contract.
//
// MUTATION-PROOF STANDARD (Sec7 dim 7's own text, StandardsDocument.md's "pin the documented
// claim"): "the cell is only load-bearing if a version of the code with the std::abort() calls
// reverted to their pre-fold-16/pre-fold-20 behavior (silent return false, or an infinite loop
// for the pre-fold-1 shape) demonstrably fails this cell while the current text passes it."
//
// SELF-CHECK PERFORMED, this authoring session (not a substitute for the build-time re-run
// below -- a throwaway, uncommitted verification that this file's own test LOGIC is sound,
// built entirely outside src/, never touching the engine tree): the design's exact printed
// Sec3.1 text was assembled into a scratch header, this file was compiled and run against it
// unmodified, and every CHECK above PASSED (12 checks, 0 failures) -- confirming the six Cell A
// constructions and the two Cell B constructions this file builds are internally consistent
// with the design's own printed C++. Two mutated copies were then built and run the same way:
// (1) the two probe-exhaustion `std::abort()` calls (Insert/InsertUnique's own trailing guard)
// reverted to the pre-fold-16 silent-miss shape -- both Cell B CHECKs (fim_capacity_desync,
// fis_capacity_desync) correctly flipped from PASS to FAIL, confirming genuine mutation-proof
// discrimination for Cell B. (2) the `ready_` one-shot guard removed entirely (all six pre-Init/
// double-Init sites) -- only the two double-Init CHECKs flipped to FAIL; the four pre-Init-misuse
// CHECKs (Insert/Find/InsertOrAssign/InsertUnique on a default-constructed, mask_=0,
// zero-length-slots_ instance) did NOT reliably crash, because calling those methods without the
// guard is undefined behaviour (an out-of-bounds std::vector element access on an empty vector),
// not a well-defined alternate code path -- on this build it happened to return without visibly
// crashing. This is not a defect in the four CHECKs (kCrashed IS the correct, and in this run
// the actually-observed, outcome when the guard IS present) -- it is a genuine, disclosed limit
// on how strongly the pre-Init half of Cell A's own mutation-proof claim can be trusted from a
// SINGLE build/toolchain/optimization-level combination, since UB's own defining property is
// that a different compiler, flag set, or allocator could observe the crash, the corruption, or
// neither. Filed here as the honest finding rather than a stronger claim this session's own
// evidence does not support. The full, build-time re-run against the real src/detail/int_hash.h
// (once it exists) remains owed and is the load-bearing evidence for the shipped suite -- this
// paragraph records that the test's own construction was checked for soundness before being
// handed to Brunel, not that the production header has been proven to satisfy it.
//
// EXECUTION MODEL, RELEASE-SAFE proof specifically: every cell below runs its child in a build
// compiled WITH /DNDEBUG (see build_link_red.bat) so the debug-only `assert(size_ < slots_.size())`
// lines inside Insert/InsertOrAssign/InsertUnique compile to nothing -- what fires, if anything
// does, is the standalone `std::abort()` the design states is "release-safe (independent of
// NDEBUG)," not a plain assert that would fire earlier and mask whether the dedicated guard
// exists at all.

#include "fixture_common.h"

#if __has_include("detail/int_hash.h")
#define T2296_HAVE_INT_HASH_H 1
#include "detail/int_hash.h"
#else
#define T2296_HAVE_INT_HASH_H 0
#endif

#if T2296_HAVE_INT_HASH_H
namespace {

using FIM = superslm::detail::FixedIntMap<uint64_t, uint64_t, superslm::detail::MixKey64>;
using FIS = superslm::detail::FixedIntSet<uint64_t, superslm::detail::MixKey64>;

// --- Cell A: pre-Init misuse, one child mode per method -------------------------------------
void ChildFimInsertPreInit() {
	FIM m;  // default-constructed: ready_ == false
	m.Insert(1, 1);
}
void ChildFimFindPreInit() {
	FIM m;
	(void)m.Find(1);
}
void ChildFimInsertOrAssignPreInit() {
	FIM m;
	m.InsertOrAssign(1, 1);
}
void ChildFisInsertUniquePreInit() {
	FIS s;
	(void)s.InsertUnique(1);
}

// --- Cell A: double-Init misuse ---------------------------------------------------------------
void ChildFimDoubleInit() {
	FIM m;
	m.Init(4);
	m.Init(4);  // second call must abort
}
void ChildFisDoubleInit() {
	FIS s;
	s.Init(4);
	s.Init(4);
}

// --- Cell A: must-accept control -- correct order never aborts --------------------------------
void ChildFimCorrectOrder() {
	FIM m;
	m.Init(4);
	m.Insert(1, 1);
	(void)m.Find(1);
	m.InsertOrAssign(1, 2);
}
void ChildFisCorrectOrder() {
	FIS s;
	s.Init(4);
	(void)s.InsertUnique(1);
}

// --- Cell B: capacity/population desync, fixed-capacity types only ----------------------------
// Init(1) -> BucketCountFor(1) = 2 (need=2, bits=1, 1<<1=2) -> mask_=1, 2 slots. Two DISTINCT
// keys fill both slots; a third distinct key's probe visits both occupied, non-matching slots
// and exhausts (steps 0..mask_ == 0..1, i.e. exactly 2 iterations) without finding a match or an
// empty slot -- the exact desync the trailing std::abort() exists to catch. Built with /DNDEBUG
// (build_link_red.bat) so the leading `assert(size_ < slots_.size())` does not fire first and
// mask whether the probe-exhaustion abort() itself is what fires.
void ChildFimCapacityDesync() {
	FIM m;
	m.Init(1);
	m.Insert(100, 1);
	m.Insert(200, 2);
	m.Insert(300, 3);  // third DISTINCT key into a 2-slot table -- must abort, not loop/return
}
void ChildFisCapacityDesync() {
	FIS s;
	s.Init(1);
	(void)s.InsertUnique(100);
	(void)s.InsertUnique(200);
	(void)s.InsertUnique(300);
}

// --- Cell B: must-accept control -- population within capacity never desyncs ------------------
void ChildFimCapacityWithinBounds() {
	FIM m;
	m.Init(4);  // BucketCountFor(4): need=8, bits=3, 1<<3=8 -> 8 slots
	for (uint64_t k = 0; k < 4; ++k) m.Insert(k, k);
}
void ChildFisCapacityWithinBounds() {
	FIS s;
	s.Init(4);
	for (uint64_t k = 0; k < 4; ++k) (void)s.InsertUnique(k);
}

}  // namespace
#endif  // T2296_HAVE_INT_HASH_H

namespace {

int RunChild(const std::string& mode) {
#if T2296_HAVE_INT_HASH_H
	if (mode == "fim_insert_preinit") { ChildFimInsertPreInit(); return 0; }
	if (mode == "fim_find_preinit") { ChildFimFindPreInit(); return 0; }
	if (mode == "fim_insertorassign_preinit") { ChildFimInsertOrAssignPreInit(); return 0; }
	if (mode == "fis_insertunique_preinit") { ChildFisInsertUniquePreInit(); return 0; }
	if (mode == "fim_double_init") { ChildFimDoubleInit(); return 0; }
	if (mode == "fis_double_init") { ChildFisDoubleInit(); return 0; }
	if (mode == "fim_correct_order") { ChildFimCorrectOrder(); return 0; }
	if (mode == "fis_correct_order") { ChildFisCorrectOrder(); return 0; }
	if (mode == "fim_capacity_desync") { ChildFimCapacityDesync(); return 0; }
	if (mode == "fis_capacity_desync") { ChildFisCapacityDesync(); return 0; }
	if (mode == "fim_capacity_within_bounds") { ChildFimCapacityWithinBounds(); return 0; }
	if (mode == "fis_capacity_within_bounds") { ChildFisCapacityWithinBounds(); return 0; }
#endif
	std::fprintf(stderr, "unknown or unbuildable child mode: %s\n", mode.c_str());
	return 2;
}

// Asserts `mode` aborts the child (SIGABRT/0xC0000409-class exit -- ChildResult::kCrashed),
// never a clean return and never a hang.
void ExpectAbort(const char* label, const std::string& mode) {
	ChildResult r = RunChildMode(mode);
	CHECK_MSG(r == ChildResult::kCrashed,
	          "%s: expected the guard to abort the process, got %s", label, ChildResultName(r));
}

// Asserts `mode` returns normally -- the must-accept control half of each pair above.
void ExpectCleanExit(const char* label, const std::string& mode) {
	ChildResult r = RunChildMode(mode);
	CHECK_MSG(r == ChildResult::kExitedZero,
	          "%s: expected a normal return (must-accept control), got %s", label,
	          ChildResultName(r));
}

}  // namespace

int main(int argc, char** argv) {
	std::string mode;
	if (ParseChildMode(argc, argv, &mode)) {
		return RunChild(mode);
	}
	SetSelfPathFromModule();

	std::printf("=== dim7_contract_red: Coverage Model dimension 7 fifth cell-group ===\n");

#if T2296_HAVE_INT_HASH_H
	std::printf("--- Cell A: pre-Init/double-Init misuse (six constructions) ---\n");
	ExpectAbort("FixedIntMap::Insert before Init()", "fim_insert_preinit");
	ExpectAbort("FixedIntMap::Find before Init()", "fim_find_preinit");
	ExpectAbort("FixedIntMap::InsertOrAssign before Init()", "fim_insertorassign_preinit");
	ExpectAbort("FixedIntSet::InsertUnique before Init()", "fis_insertunique_preinit");
	ExpectAbort("FixedIntMap::Init() called twice", "fim_double_init");
	ExpectAbort("FixedIntSet::Init() called twice", "fis_double_init");
	ExpectCleanExit("FixedIntMap correct Init-then-use order", "fim_correct_order");
	ExpectCleanExit("FixedIntSet correct Init-then-use order", "fis_correct_order");

	std::printf(
	    "--- Cell B: capacity/population desync (fixed-capacity types; growable types: "
	    "specification gap, see this file's header) ---\n");
	ExpectAbort("FixedIntMap capacity_hint=1, 3 distinct Insert()s", "fim_capacity_desync");
	ExpectAbort("FixedIntSet capacity_hint=1, 3 distinct InsertUnique()s", "fis_capacity_desync");
	ExpectCleanExit("FixedIntMap population within its own capacity_hint",
	                 "fim_capacity_within_bounds");
	ExpectCleanExit("FixedIntSet population within its own capacity_hint",
	                 "fis_capacity_within_bounds");

	SKIP_MSG(
	    "Cell B for GrowableIntSet/GrowableIntMap/GrowableContextMap: SPECIFICATION GAP, not "
	    "executed -- see this file's header comment. The design names all five replacement "
	    "types for Cell B but the three growable types' own preemptive Grow()-before-insert "
	    "discipline (Sec3.5/Sec3.6) makes the probe-exhaustion state unreachable through the "
	    "public API alone; routed back to Claude/Vitruvius rather than built with a private-"
	    "state backdoor this seat does not own adding.");

	SKIP_MSG(
	    "Mutation-proof demonstration against the PRODUCTION src/detail/int_hash.h is OWED AT "
	    "BUILD TIME, once it exists -- see this file's header comment. A throwaway self-check "
	    "against the design's own printed text was already performed this authoring session "
	    "(this file's header comment records the result, including a disclosed UB-related limit "
	    "on the pre-Init half of Cell A) but that check never touched src/ and is not a "
	    "substitute for the build-time re-run against what Brunel actually ships.");
#else
	SKIP_MSG(
	    "dimension 7 fifth cell-group NOT YET BUILDABLE -- src/detail/int_hash.h does not exist "
	    "(design Sec3.1, Sec10: \"Nothing here is built\"). This whole cell-group -- Cell A's six "
	    "pre-Init/double-Init constructions and Cell B's two fixed-capacity-type desync "
	    "constructions -- is red-unimplemented: once Brunel adds the header with FixedIntMap/"
	    "FixedIntSet as printed in the design (including the ready_ guard and the bounded-probe "
	    "std::abort()), this __has_include gate flips on and every CHECK above becomes "
	    "load-bearing without editing this file.");
	++GFailures;  // whole cell-group unattemptable -- counted red, not silently absorbed by SKIP.
#endif

	std::printf("\n%d checks, %d failures, %d skips.\n", GChecks, GFailures, GSkips);
	return GFailures == 0 ? 0 : 1;
}

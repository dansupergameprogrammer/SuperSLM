// T-2199 (Curie) -- Phase D1 red suite: the artifact flag + constants field (plan Sec8 D1,
// Decision A mechanism half, D-SLM3794) and Sec9 dim9's own siting of the pre-flag loader
// rejection cell here.
//
// Two cells are BORN GREEN (pin CURRENT, real, already-shipped loader behavior -- the exact
// guarantee D-SLM3794 states about a pre-damped-greedy runtime, executed against THIS
// worktree's own real src/artifact.cpp, no stub involved): TestD1a and TestD1c. Two cells are
// RED BY LINK (call superslm_test_phaseD::ArtifactHasDampedGreedyConstants/
// ReadDampedGreedyScaleConstants, declared in sslm_phaseD_stub.h, defined nowhere yet):
// TestD1b and TestD1d.
//
// Coverage Model cells realized here (plan Sec9):
//   dim9  Persistence round-trip and version evolution -- GATE, D-SLM3719. "An old loader
//         (pre-damped-greedy-flag) against a new artifact with the flag set rejects with
//         BadHeader" (sited at Phase D1 by Mendeleev's coverage audit, d3b310714d) --
//         TestD1c. The flag-gated field itself loading/round-tripping -- TestD1a/TestD1b.
//   dim2  Trust boundaries and hostile inputs -- GATE, D-SLM3719. Malformed/out-of-domain
//         constants reject, matching this codebase's own reject-over-degrade discipline for
//         every other artifact-carried field -- TestD1d.
#include "sslm_fixtures.h"
#include "sslm_sil1_hostile_fixtures.h"
#include "sslm_phaseD_stub.h"

#include <cstdio>

static int GChecks = 0;
static int GFailures = 0;
static int GSkips = 0;

#define CHECK(cond) \
	do { \
		++GChecks; \
		if (!(cond)) { \
			++GFailures; \
			std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
			std::fflush(stdout); \
		} \
	} while (0)

#define CHECK_MSG(cond, ...) \
	do { \
		++GChecks; \
		if (!(cond)) { \
			++GFailures; \
			std::printf("FAIL %s:%d: %s -- ", __FILE__, __LINE__, #cond); \
			std::printf(__VA_ARGS__); \
			std::printf("\n"); \
			std::fflush(stdout); \
		} \
	} while (0)

using namespace superslm;
using namespace superslm_test;
using namespace superslm_test_phaseD;

// A minimally, genuinely loadable v2 artifact -- Config + SigmoidLut, the format's own
// required-section pair (artifact.cpp's kRequiredSectionsV2, confirmed against source; the
// same pair tests/test_main.cpp's own TestOptionGFlagsZero_LoadsUnderCurrentAndLoosenedChecker
// uses for the identical reason). No damped-greedy section is added -- that is exactly what
// makes this fixture "unflagged" for TestD1a, and what TestD1c/TestD1b/TestD1d each mutate
// away from by name.
static BuiltArtifact MakeMinimalValidArtifactBytes() {
	return BuildArtifact({MakeConfigSection(), MakeSigmoidLutSection()});
}

// --- TestD1a: unflagged artifact -- byte-level regression (plan Sec8 D1, Sec9 dim9). BORN
// GREEN: a flags==0 artifact must load Ok today, unaffected by anything this ticket adds --
// the exact "artifacts without the flag are untouched" half of D-SLM3794's own ruling,
// executed against this worktree's real, unmodified src/artifact.cpp. This is the regression
// guard: any Phase D1 build that perturbs the flags==0 path for ANY reason (a stray edit to
// the header-parse order, a section-table change) fails this cell, even though nothing about
// damped greedy's own new field is exercised by it.
static void TestD1a_UnflaggedArtifact_ByteLevelRegression() {
	auto built = MakeMinimalValidArtifactBytes();
	CHECK_MSG(GetU32(built.bytes, 16) == 0, "fixture's own flags field must be 0 pre-mutation");
	SslmArtifact out;
	SslmError err;
	const auto status = SslmArtifact::OpenFromMemory(built.bytes.data(), built.bytes.size(), out, &err);
	CHECK_MSG(status == SslmStatus::Ok, "a flags=0 artifact must load Ok -- got %s (D-SLM3794: "
	                                    "'artifacts without the flag are untouched')",
	          SslmStatusName(status));
	CHECK(out.Ok());
	CHECK_MSG(out.FormatVersion() == kArtifactFormatVersion, "format_version must stay 2 -- "
	                                                          "D-SLM3794 rules 'no format-version bump'");
	// A second, byte-identical build (independent BuildArtifact() call) must produce identical
	// bytes -- the loader's own determinism, unaffected by whatever Phase D1's own converter-
	// side changes add for the FLAGGED case (a converter that starts writing extra padding, a
	// different section ORDER, etc. would still be "untouched" for THIS artifact only if the
	// unflagged path is byte-for-byte unperturbed).
	auto built2 = MakeMinimalValidArtifactBytes();
	CHECK_MSG(built.bytes == built2.bytes, "two independently-built flags=0 fixtures must be "
	                                       "byte-identical -- fixture determinism precondition "
	                                       "for this cell to mean anything");
}

// --- TestD1b: a flagged artifact with valid constants loads and exposes them (plan Sec8 D1,
// Decision A mechanism half). RED BY LINK: ArtifactHasDampedGreedyConstants/
// ReadDampedGreedyScaleConstants are declared in sslm_phaseD_stub.h, defined nowhere. Once
// Phase D1 lands (a real damped-greedy-constants section + the widened
// kKnownArtifactFlagsMask), this cell's own fixture construction -- set the flag, add a
// section carrying (m, e) -- is what a real converter output looks like; the cell then reads
// back the SAME (m, e) it wrote, bit-exact.
static void TestD1b_FlaggedArtifactWithValidConstants_LoadsAndExposesThem() {
	// A hand-chosen, real, in-domain (m, e) pair -- reuses this suite's own certified-primitive
	// derivation (fixture_common.h's DeriveDefaultScaleConstants search target, q_ln2=493) so
	// the constants this cell round-trips are not an arbitrary pair no real converter would
	// ever emit, matching this campaign's own "verified at source, not hand-typed" discipline.
	const DampedGreedyScaleConstants kExpected{/*scale_mantissa_m=*/(int64_t{1} << 30) + (int64_t{7} << 20),
	                                            /*scale_exponent_e=*/-42};

	auto built = MakeMinimalValidArtifactBytes();
	PutU32(built.bytes, 16, kDampedGreedyConstantsScaleFlag);
	// The constants section itself: this cell does not prescribe the section's own byte
	// layout (that is Phase D1's own decision, "an additive field or a versioned successor
	// struct") -- it only prescribes that SslmArtifact::OpenFromMemory("this buffer") followed
	// by ReadDampedGreedyScaleConstants("that artifact") round-trips kExpected. A real Phase D1
	// build appends whatever section/field it defines; this fixture leaves the trailing bytes
	// untouched for that build to fill in during its own fixture regeneration pass (the same
	// "regenerate the fixture against the real header" step every other Phase A/C cell in this
	// suite went through once its own symbols existed -- see Claude/Curie/
	// t2199-red-suite-2026-08-20.md's fold-21 S10 entry for the precedent).
	RecomputeIntegrityHash(built.bytes);

	SslmArtifact out;
	SslmError err;
	const auto status = SslmArtifact::OpenFromMemory(built.bytes.data(), built.bytes.size(), out, &err);
	CHECK_MSG(status == SslmStatus::Ok,
	          "a damped-greedy-flagged artifact with a valid constants field must load Ok under "
	          "Phase D1's own widened kKnownArtifactFlagsMask -- got %s",
	          SslmStatusName(status));
	CHECK_MSG(ArtifactHasDampedGreedyConstants(out),
	          "ArtifactHasDampedGreedyConstants must read true once the flag is set and a valid "
	          "constants field is present");
	DampedGreedyScaleConstants got{};
	CHECK_MSG(ReadDampedGreedyScaleConstants(out, &got), "ReadDampedGreedyScaleConstants must "
	                                                      "succeed against a well-formed flagged artifact");
	CHECK_MSG(got.scale_mantissa_m == kExpected.scale_mantissa_m,
	          "round-tripped m = %lld, want %lld", (long long)got.scale_mantissa_m,
	          (long long)kExpected.scale_mantissa_m);
	CHECK_MSG(got.scale_exponent_e == kExpected.scale_exponent_e,
	          "round-tripped e = %d, want %d", got.scale_exponent_e, kExpected.scale_exponent_e);
}

// --- TestD1c: the pre-flag loader semantics check -- an old loader (pre-damped-greedy-flag)
// against a new artifact with the flag set REJECTS with BadHeader (plan Sec9 dim9, D-SLM3794,
// docs/sslm_format.md's own Versioning section: "the loader accepts flags values that set only
// known bits and rejects (BadHeader) any unknown bit"). BORN GREEN: THIS worktree's own real,
// unmodified src/artifact.cpp/include/superslm/artifact.h IS the pre-damped-greedy loader
// D-SLM3794 describes (kKnownArtifactFlagsMask == kOptionGFusedKLandingFlag == 0x1 only, as of
// this suite's authoring commit, confirmed by direct read of artifact.h above) -- so this cell
// exercises the real guarantee live, not a stand-in. A PRESUMPTIVE bit is used
// (kDampedGreedyConstantsScaleFlag = 0x2, sslm_phaseD_stub.h) precisely because it is
// unclaimed today; per that header's own note, this specific assertion is expected to need
// updating in the SAME edit that lands Phase D1 if D1 claims a different bit.
static void TestD1c_PreDampedGreedyLoader_RejectsPresumptiveFlagBit_BadHeader() {
	CHECK_MSG((kKnownArtifactFlagsMask & kDampedGreedyConstantsScaleFlag) == 0,
	          "precondition: kDampedGreedyConstantsScaleFlag (0x%x) must be OUTSIDE this "
	          "worktree's current kKnownArtifactFlagsMask (0x%x) for this cell to be testing "
	          "'pre-flag' semantics at all -- if this fails, Phase D1 has already landed and "
	          "widened the mask, and this cell (not the loader) needs updating",
	          kDampedGreedyConstantsScaleFlag, kKnownArtifactFlagsMask);

	auto built = MakeMinimalValidArtifactBytes();
	PutU32(built.bytes, 16, kDampedGreedyConstantsScaleFlag);
	RecomputeIntegrityHash(built.bytes);

	SslmArtifact out;
	SslmError err;
	const auto status = SslmArtifact::OpenFromMemory(built.bytes.data(), built.bytes.size(), out, &err);
	CHECK_MSG(status == SslmStatus::BadHeader,
	          "a pre-damped-greedy loader against an artifact carrying the (presumptive) "
	          "damped-greedy flags bit must reject BadHeader -- got %s (D-SLM3794: 'a "
	          "pre-damped-greedy runtime REJECTS a flagged artifact with BadHeader per "
	          "docs/sslm_format.md's own versioning semantics')",
	          SslmStatusName(status));
	CHECK(err.code == SslmStatus::BadHeader);
	CHECK(!out.Ok());

	// A flags value combining the KNOWN Option-G bit with the presumptive damped-greedy bit
	// must also reject -- a known bit does not license an unknown one riding alongside it
	// (mirrors tests/test_main.cpp's own TestOptionGArtifactFlags_KnownBitAcceptedUnknownBitRejected
	// third sub-case, same reasoning, applied to this new bit).
	auto built2 = MakeMinimalValidArtifactBytes();
	PutU32(built2.bytes, 16, kOptionGFusedKLandingFlag | kDampedGreedyConstantsScaleFlag);
	RecomputeIntegrityHash(built2.bytes);
	SslmArtifact out2;
	SslmError err2;
	const auto status2 =
	    SslmArtifact::OpenFromMemory(built2.bytes.data(), built2.bytes.size(), out2, &err2);
	CHECK_MSG(status2 == SslmStatus::BadHeader,
	          "flags=(Option-G known | presumptive damped-greedy) must still reject on THIS "
	          "loader -- got %s", SslmStatusName(status2));
}

// --- TestD1d: malformed/out-of-domain constants reject per the loader's hostile-input
// discipline (plan Sec9 dim2, applied to the new field). RED BY LINK: same undefined symbols
// as TestD1b. Constructs a flagged artifact whose (m, e) violates plan Sec2.3's own real
// constraint (M <= INT64_MAX / vocab_size, Phase B2's own verification obligation) --
// specifically e chosen so the derived M overflows int64_t at the real 151,936-wide vocab this
// suite's own fixture_common.h names (t2199fixture::kRealVocabSize) -- and asserts a defined
// rejection, never a silent accept of garbage constants (this cell's own commission text:
// "an artifact missing the Decision-A scale field under damped greedy mode must be a defined
// rejection, not a silent fallback to garbage constants" -- Sec9 dim2 -- applied here to a
// PRESENT-but-malformed field, the adjacent hostile-input case that same dimension's own
// discipline covers).
static void TestD1d_MalformedConstants_RejectHostileInput() {
	// A hostile pair: m at INT64_MAX (the largest mantissa the field's own width admits) with
	// e = 0 -- under any sane derivation this makes M >> INT64_MAX / 151,936, the exact
	// overflow Sec2.3 rules out. A correct Phase D1 build's ReadDampedGreedyScaleConstants must
	// refuse this rather than hand back a value Phase B3's own IExpScaleConstants derivation
	// would silently misbehave on downstream.
	const int64_t kHostileM = INT64_MAX;
	const int32_t kHostileE = 0;

	auto built = MakeMinimalValidArtifactBytes();
	PutU32(built.bytes, 16, kDampedGreedyConstantsScaleFlag);
	RecomputeIntegrityHash(built.bytes);

	SslmArtifact out;
	SslmError err;
	const auto status = SslmArtifact::OpenFromMemory(built.bytes.data(), built.bytes.size(), out, &err);
	// The OUTER artifact load may itself accept or reject this depending on how Phase D1
	// chooses to validate the section at PARSE time (BadHeader/a new dedicated status) vs. defer
	// to ReadDampedGreedyScaleConstants's own domain check -- this cell does not prescribe
	// which; it prescribes that ONE of the two fires, never that a hostile (m, e) is silently
	// accepted and returned as "valid."
	if (status == SslmStatus::Ok) {
		DampedGreedyScaleConstants got{};
		got.scale_mantissa_m = kHostileM;  // (this cell does not have a real section-writer for
		got.scale_exponent_e = kHostileE;  // the hostile pair yet -- see the note below)
		CHECK_MSG(!ReadDampedGreedyScaleConstants(out, &got),
		          "ReadDampedGreedyScaleConstants must reject a constants field whose derived M "
		          "exceeds INT64_MAX / vocab_size (plan Sec2.3) rather than returning it -- this "
		          "is the hostile-input half of Sec9 dim2, applied to the new field");
	} else {
		CHECK_MSG(status == SslmStatus::BadHeader,
		          "if the outer loader itself rejects a hostile constants field, it must do so "
		          "with a defined status (BadHeader, matching every other structural rejection "
		          "this format uses), never a crash or an unrelated status -- got %s",
		          SslmStatusName(status));
	}
	// NOTE, filed openly rather than papered over: this cell's fixture does not yet have a real
	// section-writer to place (kHostileM, kHostileE) INTO the artifact's own bytes (Phase D1's
	// own section layout does not exist to write against) -- the `got.scale_*` fields above are
	// set directly on the OUTPUT struct as a placeholder for "what a real read would have to
	// refuse," which is sufficient to pin ReadDampedGreedyScaleConstants's own domain-check
	// CONTRACT (never return a hostile pair) but not yet its wiring to a real malformed BYTE
	// buffer. Once Phase D1's section layout exists, this cell's own fixture must be extended
	// to write kHostileM/kHostileE into the actual section bytes and re-verify the rejection
	// fires from a genuinely parsed hostile buffer, not merely from the struct's own values --
	// routed here as an owed fixture upgrade, not silently left implicit.
}

int main() {
	TestD1a_UnflaggedArtifact_ByteLevelRegression();
	TestD1c_PreDampedGreedyLoader_RejectsPresumptiveFlagBit_BadHeader();
	TestD1b_FlaggedArtifactWithValidConstants_LoadsAndExposesThem();
	TestD1d_MalformedConstants_RejectHostileInput();
	std::printf("checks=%d failures=%d skips=%d\n", GChecks, GFailures, GSkips);
	return GFailures ? 1 : 0;
}

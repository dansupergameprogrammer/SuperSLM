// T-2199 (Curie) -- Phase D1 red suite: the artifact flag + constants field (plan Sec8 D1,
// Decision A mechanism half, D-SLM3794) and Sec9 dim9's own siting of the pre-flag loader
// rejection cell here.
//
// REVISED 2026-08-20 (conductor's dispute-resolution commission, after brunel/t2199-phaseD@
// 5ced7a6 landed): TestD1a stays BORN GREEN, unaffected. TestD1b's fixture now writes a REAL
// DampedGreedyConstants section via the production MakeDampedGreedyConstantsSection (disputed
// 1 of 5 -- the fixture never wrote real bytes before; fixed, not worked around). TestD1c is
// RE-SCOPED (disputed 1 of 5, ruling below) -- its literal pre-build assertion is now
// definitionally false the moment D1 lands (flags=0x2 is KNOWN in this build), which is not a
// build defect, it is exactly what the cell's own original comment predicted. TestD1d is
// unchanged (not disputed, already GREEN).
//
// RULING on the TestD1b/TestD1c contradiction (conductor's dispute 1). Two readings were
// possible: (a) treat TestD1c as permanently broken and delete it, losing D-SLM3794's own
// "pre-damped-greedy runtime REJECTS a flagged artifact" guarantee as a live, re-checkable
// claim; or (b) re-scope TestD1c to what is STILL true and STILL checkable post-build --
// (b) is the correct reading: D-SLM3794's real, general promise is the FORMAT's own
// unknown-flag-bit-rejection mechanism (docs/sslm_format.md's Versioning section), not the
// specific historical fact that 0x2 itself was once unknown -- that fact is real, was true,
// and is preserved as dated evidence (below), but a cell asserting it AS A LIVE, RE-RUNNABLE
// CHECK is structurally impossible once the very build under test is the one that made 0x2
// known -- no bit choice fixes this (the builder's own evidence: picking a different literal
// for "the real flag" only moves the contradiction, since TestD1b's own fixture would then need
// the SAME new bit to be known, and TestD1c would need it to be unknown, in the identical
// build). TestD1c is renamed TestD1_UnknownFlagBit_RejectsWithBadHeader_FormatSemantics and now
// probes a bit that is STILL genuinely unclaimed after D1 (0x4 -- the next bit past both
// Option-G's 0x1 and damped greedy's own 0x2), pinning the GENERAL mechanism against the
// CURRENT, post-D1 loader -- an evergreen regression guard, not a time-scoped historical claim.
// The specific historical fact (0x2 rejected pre-D1) is preserved as a dated citation to this
// suite's own already-executed evidence (Claude/Curie/t2199-phaseD-red-2026-08-20.md's own
// "Born-green verification, isolated" section: checks=10 failures=0 against commit 2d7d381,
// before D1 landed) rather than reasserted as a cell that cannot both exist and pass.
//
// Coverage Model cells realized here (plan Sec9):
//   dim9  Persistence round-trip and version evolution -- GATE, D-SLM3719. The flag-gated
//         field itself loading/round-tripping -- TestD1a/TestD1b. The FORMAT's own general
//         unknown-bit-rejection mechanism -- TestD1_UnknownFlagBit_RejectsWithBadHeader_
//         FormatSemantics (re-scoped from the original TestD1c, see ruling above; the
//         ORIGINAL, time-scoped claim -- specifically that 0x2 was unknown pre-D1 -- is a
//         dated historical fact, cited above, not a live cell in a post-D1 tree).
//   dim2  Trust boundaries and hostile inputs -- GATE, D-SLM3719. Malformed/out-of-domain
//         constants reject, matching this codebase's own reject-over-degrade discipline for
//         every other artifact-carried field -- TestD1d.
#include "sslm_fixtures.h"
#include "sslm_sil1_hostile_fixtures.h"
#include "sslm_phaseD_stub.h"
#include "sslm_phaseD_fixture.h"

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
// Decision A mechanism half). FIXED 2026-08-20 (conductor's dispute-resolution commission,
// dispute 1): the original fixture set the flags word and stopped, never writing a real
// DampedGreedyConstants section -- no implementation could make ArtifactHasDampedGreedyConstants
// return true against bytes that were never written. Now uses the REAL production
// MakeDampedGreedyConstantsSection (src/damped_greedy_phaseD.cpp, landed with Phase D1) to
// append a genuine DGC1 section, and a genuinely in-domain (m, e) pair derived via this
// suite's own shared DeriveDefaultScaleConstantsSourceME (sslm_phaseD_fixture.h) rather than a
// hand-picked pair never verified against ReadDampedGreedyScaleConstants's own domain check
// (ScaleConstantsInDomain, src/damped_greedy_phaseD.cpp) -- a hand-picked pair that happened to
// fail that check would have made this cell fail for a REASON UNRELATED to what it tests.
static void TestD1b_FlaggedArtifactWithValidConstants_LoadsAndExposesThem() {
	int64_t src_m = 0;
	int32_t src_e = 0;
	CHECK_MSG(t2199phaseD::DeriveDefaultScaleConstantsSourceME(&src_m, &src_e),
	          "could not derive an in-domain (m, e) source pair for q_ln2=493 -- fixture "
	          "precondition, not the cell under test");
	const DampedGreedyScaleConstants kExpected{/*scale_mantissa_m=*/src_m, /*scale_exponent_e=*/src_e};

	auto raw_section = MakeDampedGreedyConstantsSection(kExpected);
	FixtureSection dgc_section;
	dgc_section.type = raw_section.type;
	dgc_section.dtype = raw_section.dtype;
	dgc_section.data = raw_section.data;
	dgc_section.alignment = raw_section.alignment;

	auto built = BuildArtifact({MakeConfigSection(), MakeSigmoidLutSection(), dgc_section});
	PutU32(built.bytes, 16, kDampedGreedyArtifactConstantsFlag);
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

// --- TestD1_UnknownFlagBit_RejectsWithBadHeader_FormatSemantics: RE-SCOPED from the original
// TestD1c (conductor's dispute-resolution commission, dispute 1 -- see this file's own header
// comment for the full ruling). D-SLM3794's real, GENERAL promise is the format's own
// unknown-flag-bit-rejection mechanism (docs/sslm_format.md's Versioning section: "the loader
// accepts flags values that set only known bits and rejects (BadHeader) any unknown bit") --
// this cell pins that mechanism against a bit that is STILL genuinely unclaimed in THIS,
// post-D1 tree (0x4, the next bit past Option-G's 0x1 and damped greedy's own 0x2, confirmed
// below), so it stays a live, re-runnable regression guard rather than a claim that becomes
// definitionally false the moment any future flag (D1's own damped-greedy bit included) lands.
//
// The ORIGINAL, time-scoped claim this cell used to make -- that 0x2 SPECIFICALLY was unknown
// and rejected BEFORE Phase D1 landed -- is real, was independently verified by execution, and
// is preserved as dated evidence rather than reasserted here: Claude/Curie/
// t2199-phaseD-red-2026-08-20.md's own "Born-green verification, isolated" section records
// `checks=10 failures=0 skips=0` for that exact construction (flags=0x2, and flags=(0x1|0x2))
// against this repo's real loader at commit 2d7d381 (SuperSLM repo), before Phase D1's own
// commit 5ced7a6 widened kKnownArtifactFlagsMask to include 0x2. That evidence stands on its
// own; it is not re-asserted as a live cell because the build that would run it is definitionally
// the one that makes the assertion false (the builder's own dispute evidence, § this file's
// header comment).
static void TestD1_UnknownFlagBit_RejectsWithBadHeader_FormatSemantics() {
	// kProbeUnknownFlag must stay OUTSIDE kKnownArtifactFlagsMask for this cell to test
	// "unknown bit" semantics at all -- checked live so a FUTURE flag landing at 0x4 fails this
	// PRECONDITION loudly (routing to Curie for a bit bump) rather than this cell silently
	// testing a bit that is no longer unknown.
	constexpr uint32_t kProbeUnknownFlag = 0x4u;
	CHECK_MSG((kKnownArtifactFlagsMask & kProbeUnknownFlag) == 0,
	          "precondition: the probe bit (0x%x) must be OUTSIDE this worktree's current "
	          "kKnownArtifactFlagsMask (0x%x) -- if this fails, a THIRD flag has landed at this "
	          "bit and this cell needs a fresh unclaimed probe, not a claim about the loader",
	          kProbeUnknownFlag, kKnownArtifactFlagsMask);
	// Sanity check on the ruling's own premise: kDampedGreedyArtifactConstantsFlag (D1's real
	// bit) IS now known -- confirms this tree is genuinely post-D1, i.e. that TestD1c's own
	// original assertion really would be false here (not merely reasoned to be).
	CHECK_MSG((kKnownArtifactFlagsMask & kDampedGreedyArtifactConstantsFlag) != 0,
	          "sanity: kDampedGreedyArtifactConstantsFlag must be KNOWN in this tree -- if this "
	          "fails, Phase D1 has NOT actually landed and the original TestD1c should be "
	          "restored instead of this re-scoped cell");

	auto built = MakeMinimalValidArtifactBytes();
	PutU32(built.bytes, 16, kProbeUnknownFlag);
	RecomputeIntegrityHash(built.bytes);

	SslmArtifact out;
	SslmError err;
	const auto status = SslmArtifact::OpenFromMemory(built.bytes.data(), built.bytes.size(), out, &err);
	CHECK_MSG(status == SslmStatus::BadHeader,
	          "an artifact carrying a genuinely unclaimed flags bit must reject BadHeader -- got "
	          "%s (docs/sslm_format.md's Versioning section: 'rejects (BadHeader) any unknown "
	          "bit')", SslmStatusName(status));
	CHECK(err.code == SslmStatus::BadHeader);
	CHECK(!out.Ok());

	// A flags value combining BOTH now-known bits (Option-G, damped greedy) with the unknown
	// probe must also reject -- known bits do not license an unknown one riding alongside them
	// (mirrors tests/test_main.cpp's own
	// TestOptionGArtifactFlags_KnownBitAcceptedUnknownBitRejected third sub-case).
	auto built2 = MakeMinimalValidArtifactBytes();
	PutU32(built2.bytes, 16,
	       kOptionGFusedKLandingFlag | kDampedGreedyArtifactConstantsFlag | kProbeUnknownFlag);
	RecomputeIntegrityHash(built2.bytes);
	SslmArtifact out2;
	SslmError err2;
	const auto status2 =
	    SslmArtifact::OpenFromMemory(built2.bytes.data(), built2.bytes.size(), out2, &err2);
	CHECK_MSG(status2 == SslmStatus::BadHeader,
	          "flags=(every known bit | one unknown probe bit) must still reject -- got %s",
	          SslmStatusName(status2));
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
	PutU32(built.bytes, 16, kDampedGreedyArtifactConstantsFlag);
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
	TestD1_UnknownFlagBit_RejectsWithBadHeader_FormatSemantics();
	TestD1b_FlaggedArtifactWithValidConstants_LoadsAndExposesThem();
	TestD1d_MalformedConstants_RejectHostileInput();
	std::printf("checks=%d failures=%d skips=%d\n", GChecks, GFailures, GSkips);
	return GFailures ? 1 : 0;
}

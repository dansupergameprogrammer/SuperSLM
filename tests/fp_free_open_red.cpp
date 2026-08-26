// T-2296 (Curie) -- Sec4.2 "Liveness control (i) -- the masks-cleared trap-return regression
// cell." Design of record: Claude/Vitruvius/t2265-superslm-fp-free-open-design-2026-08-24.md
// Sec4.2, filed at this exact path per the design's own text ("A CI-resident test,
// tests/fp_free_open_red.cpp (new)...").
//
// THE FULL CELL, AS SPECIFIED (Sec4.2): opens a tokenizer AND a model, calls sslm_seq_create,
// calls sslm_seq_restore on a saved handle carrying a nonempty anti-LM history, runs a forward,
// under every combination of the six MXCSR exception masks cleared, four IEEE rounding modes,
// and both denormal states -- asserting the process returns (leg (a)/(b), Sec4.2/Sec7 dim 6)
// and the emitted bytes are byte-identical across all 16 non-crashing combinations.
//
// SCOPE AUTHORED THIS SESSION: the tokenizer-open leg only -- TokenizerViewAccess::OpenImpl,
// the first of the design's own four scan roots (Sec2.6), covering sites 1-4 (merges/ccc_map/
// decomp/compose) at REAL artifact-driven population (this repo's own committed
// tests/fixtures/qwen2.5-1.5b.tok.sslm, the real Qwen2.5-1.5B tokenizer -- 151,936-element
// merges table, matching T-2284/fold round 21's own measured real-site population exactly).
// NOT authored this session, and why: SslmModel::Load, sslm_seq_create, and sslm_seq_restore
// (covering sites 5-10 and the design's remaining three scan roots) need a REAL full model
// artifact with a bound anti-LM/damped-greedy configuration -- no such artifact is committed to
// this repo (only the tokenizer-only .tok.sslm fixture is; every other suite in this tree that
// needs a full model takes one via --model=/--modeltok= argv and SKIPs when not supplied,
// tests/t2138-abi-red-suite/fixture_common.h's own convention, reused below). The 16-combination
// masks/rounding-mode/denormal-state byte-identity oracle is ALSO not authored this session: it
// is only informative once the masks-cleared leg stops trapping (Sec4.2's own text: this oracle
// compares bytes across "all 16 combinations," and a trapping leg produces no bytes to compare)
// -- pre-remedy, on a toolchain where the masks-cleared leg traps (this session's own executed
// finding: MSVC 19.33.31631 traps categorically, see this suite's own dim6_determinism_red.cpp),
// there is nothing yet to run the oracle over. Both gaps are named in this campaign's test-design
// record (Claude/Curie/t2296-fp-free-open-red-suite-test-design-2026-08-26.md) as owed follow-on
// work, not silently dropped.
//
// RED-FIRST LAW (StandardsDocument.md; Curie.md "Red before green"): the CHECK below asserts the
// POST-REMEDY target directly -- TokenizerView::Open returns normally under masks-cleared -- not
// today's known-broken behaviour. A test that asserts the current defect as the correct outcome
// is not a red test; it inherits the defect's own authority instead of constraining it away.
//
// EXECUTED FINDING, this session: running TokenizerView::Open on the real fixture under
// masks-cleared on THIS machine's default toolchain (MSVC 19.33.31631, the Community install's
// VsDevCmd.bat -- confirmed via `cl` banner, this suite's own dim6_determinism_red.cpp header
// comment) TRAPS -- the real, artifact-driven, four-site population reproduces TE-19's own
// original finding directly through the production API, not merely through a reconstructed
// standalone container (contrast Claude/Loki/t2284-probe/tensites.cpp, which replicates each
// site's declared type/idiom shape rather than calling TokenizerView::Open itself). This is the
// genuine RED state Sec4.2 commissions this cell to prove exists pre-remedy: on this machine's
// default toolchain, the CHECK below fails today, for exactly this reason. Sec4.2's own text
// disclaims that this leg's outcome is toolchain-conditioned (T-2284/fold round 21: 0 of 10 real
// sites trap on MSVC 19.44) -- on a 19.40+ toolchain this cell MAY already read green pre-remedy,
// a disclosed property of testing a real population rather than a constructed boundary, not a
// defect in this test; dim6_determinism_red.cpp's own Cell A construction is the leg the design
// names as guaranteed red on every toolchain (Sec7 dim 6 Cell A, "the one leg that ships in CI").

#include "t2296-fp-free-open-red-suite/fixture_common.h"

#include "superslm/artifact.h"
#include "superslm/tokenizer.h"

using superslm::SslmArtifact;
using superslm::SslmStatus;
using superslm::SslmStatusName;
using superslm::TokenizerView;

namespace {

std::string ResolveTokFixture() {
	return ResolveFixturePath("qwen2.5-1.5b.tok.sslm");
}

int RunChild(const std::string& mode) {
	if (mode == "open_masks_cleared") {
		std::string path = ResolveTokFixture();
		if (path.empty()) {
			std::fprintf(stderr, "fixture not found\n");
			return 3;
		}
		superslm::SslmError aerr;
		SslmArtifact artifact;
		auto status = SslmArtifact::OpenFromFile(path.c_str(), artifact, &aerr);
		if (status != SslmStatus::Ok) {
			std::fprintf(stderr, "artifact open failed before the leg under test: %s: %s\n",
			             SslmStatusName(status), aerr.message.c_str());
			return 4;
		}
		// The leg under test: MXCSR is cleared immediately before the call this design's own
		// four sites live inside (TokenizerViewAccess::OpenImpl, Sec2.6 root 1) -- reading the
		// artifact itself (above) is not part of the claim and runs under default masks.
		ClearAllMxcsrExceptionMasks();
		TokenizerView view;
		std::string terr;
		bool ok = TokenizerView::Open(artifact, view, &terr);
		// Reaching here means Open() returned -- true or false, it did NOT trap. A false return
		// (a parse rejection) would still be a real bug worth CHECKing on the must-accept control
		// below, but it is not this cell's own claim (Sec4.2 is about trapping, not parsing).
		return ok ? 0 : 1;
	}
	if (mode == "open_masks_default") {
		std::string path = ResolveTokFixture();
		if (path.empty()) return 3;
		superslm::SslmError aerr;
		SslmArtifact artifact;
		auto status = SslmArtifact::OpenFromFile(path.c_str(), artifact, &aerr);
		if (status != SslmStatus::Ok) return 4;
		TokenizerView view;
		std::string terr;
		bool ok = TokenizerView::Open(artifact, view, &terr);
		return ok ? 0 : 1;
	}
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

	std::printf("=== fp_free_open_red: Sec4.2 liveness control (i), tokenizer-open leg ===\n");

	if (ResolveTokFixture().empty()) {
		SKIP_MSG(
		    "tests/fixtures/qwen2.5-1.5b.tok.sslm not found -- cannot run this cell from the "
		    "current working directory. Run from the repo root or a directory tests/fixtures is "
		    "reachable from (matching every other fixture-driven suite in this repo).");
		std::printf("\n%d checks, %d failures, %d skips.\n", GChecks, GFailures, GSkips);
		return GFailures == 0 ? 0 : 1;
	}

	// --- Must-accept control: the SAME open, under default masks, must succeed and parse the
	//     real Qwen2.5-1.5B tokenizer artifact correctly -- proves a failure on the
	//     masks-cleared leg (below) is about the trap, not about the fixture being broken. ---
	{
		ChildResult r = RunChildMode("open_masks_default");
		CHECK_MSG(r == ChildResult::kExitedZero,
		          "control: TokenizerView::Open under default masks: expected a clean, "
		          "successful open of the real fixture, got %s", ChildResultName(r));
	}

	// --- The claim under test (Sec4.2, Sec1's own motivating sentence): with all six MXCSR
	//     exception masks cleared, the real, artifact-driven, four-site population (site 1
	//     `merges` at 151,936 elements from this exact fixture) traps -- reproducing TE-19's own
	//     finding through the actual production TokenizerView::Open call, on whichever toolchain
	//     builds this cell. This is a TOOLCHAIN-CONDITIONED claim (Sec1's own re-pricing,
	//     D-SLM4591): it traps on a 19.3x-class MSVC family and returns normally on 19.44 at
	//     these real populations (T-2284/fold round 21, both re-confirmed against the raw STL
	//     boundary by this suite's own dim6_determinism_red.cpp on this exact machine). Reported,
	//     not asserted as a hard pass/fail either way, on a 19.40+ toolchain -- the toolchain-
	//     conditioned reading this design's own §1/§9 residual carries per-toolchain, not a
	//     universal one this single cell can certify (Sec4.2's own text: "What it cannot decide:
	//     whether the pre-remedy engine traps on any specific consumer's toolchain and
	//     population"). ---
	// RED-FIRST LAW (StandardsDocument.md; Curie.md "Red before green"): the CHECK below asserts
	// the POST-REMEDY target -- the process does NOT trap -- never the current, known-broken
	// behaviour. A test asserting today's defect as correct is not a red test; it is the defect
	// wearing a passing suite (Curie.md's own phrasing). The assertion is unconditional; what
	// varies by toolchain is only whether it is EXPECTED to be red right now, printed as context.
#if defined(_MSC_VER) && _MSC_VER < 1940
	std::printf("(compiled with _MSC_VER=%d, a pre-19.40 family: this session's own "
	            "dim6_determinism_red.cpp finding is that the real, artifact-driven population "
	            "traps here today -- expect RED below, for the right reason, until Sec3 lands)\n",
	            _MSC_VER);
#else
	std::printf("(compiled with a 19.40+ or non-MSVC toolchain: Sec4.2's own text and T-2284's "
	            "own execution found the real, artifact-driven population does NOT trap here "
	            "even pre-remedy -- this cell may already read green on this toolchain, a "
	            "disclosed limitation of testing a real population rather than a constructed "
	            "boundary; Sec7 dim 6 Cell A's own construction, dim6_determinism_red.cpp, is "
	            "the toolchain-independent leg that is guaranteed red pre-remedy on every "
	            "toolchain)\n");
#endif
	{
		ChildResult r = RunChildMode("open_masks_cleared");
		CHECK_MSG(r == ChildResult::kExitedZero,
		          "TokenizerView::Open(real qwen2.5-1.5b tokenizer, real 151,936-element merges "
		          "population) under masks-cleared: expected a normal return (the post-remedy "
		          "target, Sec4.2), got %s -- if this is a trap, that is the pre-remedy defect "
		          "this design removes, reproduced through the real production call sequence",
		          ChildResultName(r));
	}

	// --- Not authored this session (see this file's own header comment): SslmModel::Load,
	//     sslm_seq_create, sslm_seq_restore (sites 5-10, scan roots 2-4) -- no committed full-
	//     model fixture; and the 16-combination rounding-mode/denormal-state byte-identity
	//     oracle -- informationally vacuous while the masks-cleared leg still traps. ---
	SKIP_MSG(
	    "SslmModel::Load / sslm_seq_create / sslm_seq_restore legs (sites 5-10, scan roots 2-4): "
	    "NOT authored this session -- no committed full-model artifact with a bound anti-LM "
	    "history; routed as a follow-on in this campaign's test-design record.");
	SKIP_MSG(
	    "the 16-combination (masks-cleared x 4 rounding modes x 2 denormal states) byte-identity "
	    "oracle: NOT authored this session -- informationally vacuous while the masks-cleared "
	    "leg still traps pre-remedy on the toolchain family that traps; owed once FixedIntMap/"
	    "FixedIntSet exist and the leg returns bytes to compare.");

	std::printf("\n%d checks, %d failures, %d skips.\n", GChecks, GFailures, GSkips);
	return GFailures == 0 ? 0 : 1;
}

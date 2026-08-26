// T-2296 (Curie) -- Sec4.3 "Liveness control (ii) -- the post-swap sticky-flag readback."
// Design of record: Claude/Vitruvius/t2265-superslm-fp-free-open-design-2026-08-24.md Sec4.3,
// filed at this exact path per the design's own text ("tests/fp_free_sticky_readback.cpp (new,
// or an added leg on Sec4.2's harness)").
//
// WHAT TE-19 MEASURED, THE BASELINE THIS CONTROL MOVES (Sec4.3's own text): "TE-19 read MXCSR's
// sticky exception flags back across a full tokenizer open... and found exactly one exception
// class raised: PE (precision), from the bucket-sizing divide in reserve()." This cell's own
// construction: clear MXCSR's STICKY FLAGS (bits 0-5) before the call -- masks stay at their
// DEFAULT (masked/suppressed) state, so the divide does not trap, it merely raises and leaves
// its own sticky bit set -- run the real sequence, read the sticky flags back, and assert clean.
// Distinct from Sec4.2's own masks-CLEARED trap-return cell (fp_free_open_red.cpp): that cell
// clears exception MASKS so the divide traps; this cell leaves masks default so the divide
// completes and its sticky flag is inspectable afterward, matching TE-19's own original reading.
//
// SCOPE, identical to fp_free_open_red.cpp's own (see that file's header comment for the full
// reasoning): the tokenizer-open leg only (sites 1-4, scan root 1), against the real committed
// tests/fixtures/qwen2.5-1.5b.tok.sslm fixture. SslmModel::Load / sslm_seq_create /
// sslm_seq_restore (sites 5-10, scan roots 2-4) are NOT authored this session -- no committed
// full-model fixture; routed as a follow-on in this campaign's test-design record
// (Claude/Curie/t2296-fp-free-open-red-suite-test-design-2026-08-26.md).
//
// RED STATE, this session's own executed finding: running the real TokenizerView::Open on the
// real fixture with sticky flags cleared and masks at default leaves PE (precision) set --
// reproducing TE-19's own finding through the production API at the real, artifact-driven
// population, unconditionally of toolchain family (unlike Sec4.2's own trap-return cell, a
// divide that raises PE and continues under DEFAULT masks is not toolchain-version-conditioned
// the way TRAPPING under cleared masks is -- both 19.33 and 19.44 raise PE on an inexact divide,
// per T-2284's own D-SLM4593/D-SLM4594 findings that the DIVERGENCE between toolset families is
// specifically about whether the trap fires, not about whether the divide is inexact).

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

constexpr unsigned kStickyFlagsMask = 0x3Fu;  // IE DE ZE OE UE PE, MXCSR bits 0-5

// Clears MXCSR's sticky exception flags (bits 0-5) WITHOUT touching the exception masks
// (bits 7-12) -- distinct from fixture_common.h's ClearAllMxcsrExceptionMasks, which clears the
// masks instead so an exception traps. This cell wants the opposite: masks stay default (masked,
// non-trapping) so an exception raised by the call under test completes and leaves its own
// sticky bit inspectable afterward, matching TE-19's own original reading (Sec4.3's own text).
void ClearStickyFlagsOnly() {
	_mm_setcsr(_mm_getcsr() & ~kStickyFlagsMask);
}

int RunChild(const std::string& mode) {
	if (mode != "sticky_readback") {
		std::fprintf(stderr, "unknown child mode: %s\n", mode.c_str());
		return 2;
	}
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
	// The leg under test: sticky flags cleared, masks left at default (this process's own
	// startup state -- masked/non-trapping, the ordinary CRT default), immediately before the
	// call this design's own four sites live inside.
	ClearStickyFlagsOnly();
	TokenizerView view;
	std::string terr;
	bool ok = TokenizerView::Open(artifact, view, &terr);
	if (!ok) {
		std::fprintf(stderr, "TokenizerView::Open failed (not this cell's own claim): %s\n",
		             terr.c_str());
		return 5;
	}
	unsigned sticky = _mm_getcsr() & kStickyFlagsMask;
	// Encode the six sticky bits directly in the exit code (values 0-63) so the parent can
	// read exactly which exception class(es) fired without needing a shared file or pipe --
	// this process either traps (Sec4.2's own concern, not reachable here since masks stay
	// default) or returns, and its exit code is the one channel guaranteed to survive either
	// way.
	return static_cast<int>(sticky);
}

}  // namespace

int main(int argc, char** argv) {
	std::string mode;
	if (ParseChildMode(argc, argv, &mode)) {
		return RunChild(mode);
	}
	SetSelfPathFromModule();

	std::printf("=== fp_free_sticky_readback: Sec4.3 liveness control (ii), tokenizer-open leg ===\n");

	if (ResolveTokFixture().empty()) {
		SKIP_MSG(
		    "tests/fixtures/qwen2.5-1.5b.tok.sslm not found -- cannot run this cell from the "
		    "current working directory. Run from the repo root or a directory tests/fixtures is "
		    "reachable from (matching every other fixture-driven suite in this repo).");
		std::printf("\n%d checks, %d failures, %d skips.\n", GChecks, GFailures, GSkips);
		return GFailures == 0 ? 0 : 1;
	}

	// The child's own real exit code IS the sticky-flag bitmask (0-63) on a clean run
	// (RunChild's own "return static_cast<int>(sticky);" above) -- RunChildModeRaw is the one
	// spawn that recovers both the classification AND the actual code value, so this cell needs
	// no second process spawn to read it.
	DWORD raw_exit = 0;
	ChildResult r = RunChildModeRaw("sticky_readback", &raw_exit);
	if (r == ChildResult::kSpawnFailed || r == ChildResult::kTimedOut ||
	    r == ChildResult::kCrashed) {
		CHECK_MSG(false,
		          "sticky_readback child did not exit with a decodable sticky-flag exit code: %s "
		          "(a crash here would be Sec4.2's own concern, not expected under default "
		          "masks)",
		          ChildResultName(r));
	} else if (raw_exit >= 3 && raw_exit <= 5) {
		CHECK_MSG(false, "sticky-readback child reported a fixture/open failure (exit %lu), not "
		                  "a sticky-flag reading -- see the child's own stderr if re-run "
		                  "directly with --child-mode=sticky_readback",
		          raw_exit);
	} else {
		const bool pe_set = (raw_exit & 0x20u) != 0;  // PE, MXCSR bit 5
		const bool any_sticky_set = (raw_exit & kStickyFlagsMask) != 0;
		std::printf("sticky flags after tokenizer-open (raw=0x%02lX): IE=%d DE=%d ZE=%d OE=%d "
		            "UE=%d PE=%d\n",
		            raw_exit, (raw_exit & 0x01) != 0, (raw_exit & 0x02) != 0,
		            (raw_exit & 0x04) != 0, (raw_exit & 0x08) != 0, (raw_exit & 0x10) != 0,
		            pe_set);
		// RED-FIRST LAW (StandardsDocument.md; Curie.md "Red before green"): the CHECK below is
		// the design's own POST-REMEDY target -- "asserting all six of IE DE ZE OE UE PE clear"
		// (Sec4.3's own text) -- asserted directly, not the current defect's own presence. A
		// cell that CHECKs "PE is set" today would be asserting the defect as correct, which is
		// not a red test (Curie.md).
		//
		// EXECUTED FINDING, this session: PE is set today (pe_set == true, printed above), at
		// the real fixture's real population, matching TE-19's own original reading exactly --
		// this is WHY the CHECK below fails right now, not a separate fact reported alongside a
		// weaker assertion. Unlike Sec4.2's own trap-return leg, this reading is NOT
		// toolchain-family-conditioned in the way trapping is: a divide that is inexact for a
		// given operand set raises PE and completes under DEFAULT (masked) exception handling on
		// EITHER MSVC family (T-2284's own D-SLM4593/D-SLM4594: the 19.33/19.44 divergence is
		// specifically about whether the operation TRAPS, never about whether it is inexact) --
		// so this cell is expected red on both toolchain families this design has measured,
		// unlike fp_free_open_red.cpp's own leg.
		CHECK_MSG(!any_sticky_set,
		          "expected all six sticky flags (IE DE ZE OE UE PE) clear after tokenizer-open "
		          "(Sec4.3's own post-remedy target) -- got raw=0x%02lX (PE=%d). This is the "
		          "pre-remedy defect TE-19 originally recorded, reproduced through the real "
		          "production TokenizerView::Open call at the real fixture's real population.",
		          raw_exit, pe_set);
	}

	SKIP_MSG(
	    "SslmModel::Load / sslm_seq_create / sslm_seq_restore legs (sites 5-10, scan roots 2-4): "
	    "NOT authored this session -- no committed full-model artifact with a bound anti-LM "
	    "history; routed as a follow-on in this campaign's test-design record.");

	std::printf("\n%d checks, %d failures, %d skips.\n", GChecks, GFailures, GSkips);
	return GFailures == 0 ? 0 : 1;
}

// tests/areas/bad_alloc_contract.cpp -- T-1574 test suite split, Stage 2.
// Area #10 (Claude/Plans/SuperSLM_TestSuiteSplit_Plan.md §4): the cross-
// cutting "throws only std::bad_alloc" exception-safety contract, all
// nineteen sites, spanning artifact.cpp/sha256.cpp/tokenizer.cpp/model.cpp/
// proof_manifest.cpp's public entry points. Extracted verbatim from
// tests/test_main.cpp (order keys 314-335); no local fixtures owned.
// FixtureTokenizer/OpenFixtureTokenizer and BuildFullyValidV2ArtifactForLoad
// (plus its own MakeKvc1CompositionSection/MakeWsc1Section/MakeRop1Section
// ingredients) are consumed from tests/support/shared_fixtures.h -- promoted
// there by this stage per the plan's standing cross-area-fixture rule (§4):
// both are also used by candidate areas #2 (tokenizer.cpp) and #4
// (model_load.cpp), neither yet extracted.

#include "superslm/artifact.h"
#include "superslm/model.h"
#include "superslm/proof_manifest.h"
#include "superslm/sha256.h"
#include "superslm/tokenizer.h"
#include "sslm_tokenizer_fixtures.h"

#include "support/bad_alloc_injection.h"
#include "support/shared_fixtures.h"
#include "support/test_harness.h"
#include "support/test_registry.h"

#include <cstdint>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

using namespace superslm;
using namespace superslm_test;

// ---------------------------------------------------------------------------
// Curie's S-HARDEN-7 suite (Claude/Vitruvius/SuperSLM_SHARDEN678_Bundle_
// Design-2026-07-23.md §3; T-411): the "throws only std::bad_alloc" contract.
// Every cell below is authored against the corrected, four-condition
// membership rule's population, NINETEEN sites as of T-1475 (JsonEscape's
// promotion out of its anonymous namespace and into proof_manifest.h made
// it the 19th derived member; design §3.1's table itself still states
// eighteen and is owed a matching amendment outside this suite's writable
// surface) -- not the twelve, ten, six, or seven a hand enumeration found
// across earlier passes of the design.
//
// The rename-and-wrap has landed: every site's public entry point is now a
// thin wrapper (src/bad_alloc_wrap.h's WrapBadAllocContract) around an
// internal *Impl body, and each *Impl body consults
// superslm_test::MaybeThrowInjectedFault() (tests/support/bad_alloc_injection.h)
// via internal::MaybeThrowInjectedBadAllocFault() -- a no-op in a release
// build (SUPERSLM_ENABLE_BAD_ALLOC_INJECTION is defined only for the
// superslm_test_injection target the test binary links, CMakeLists.txt).
// Every cell below arms the seam, calls the site's real public entry point
// with a minimal, safe argument, and asserts the injected fault crossed as
// std::bad_alloc.
// ---------------------------------------------------------------------------

namespace {

// One CHECK per site: the caller must observe exactly std::bad_alloc for an
// injected std::length_error, and nothing else. `call_site` invokes the real
// production entry point with the seam armed.
template <typename Callable>
void CheckBadAllocContractSite(const char* site_name, Callable&& call_site) {
	using namespace superslm_test;
	ArmInjectedFault(InjectThrowKind::kLengthError);
	bool threw_bad_alloc = false;
	bool threw_other = false;
	bool threw_nothing = false;
	try {
		call_site();
		threw_nothing = true;
	} catch (const std::bad_alloc&) {
		threw_bad_alloc = true;
	} catch (...) {
		threw_other = true;
	}
	DisarmInjectedFault();
	CHECK_MSG(threw_bad_alloc,
	          "%s must convert an injected std::length_error into std::bad_alloc "
	          "(S-HARDEN-7 design §3.1) -- observed %s",
	          site_name,
	          threw_nothing ? "no exception at all (the seam was not consulted)"
	                        : (threw_other ? "a non-bad_alloc exception (unconverted)"
	                                       : "bad_alloc"));
}

}  // namespace

// --- Site 1/19: SslmArtifact::OpenFromMemory (artifact.h:149) ---
SSLM_TEST(TestBadAllocContractOpenFromMemory, 314) {
	uint8_t data[4] = {'S', 'S', 'L', 'M'};
	SslmArtifact out;
	SslmError err;
	CheckBadAllocContractSite("SslmArtifact::OpenFromMemory", [&] {
		SslmArtifact::OpenFromMemory(data, sizeof(data), out, &err);
	});
}

// --- Site 1/19, representative marker cell (design §3.2 "New cell -- force a
//     genuine std::bad_alloc through the wrap"): the shared wrap helper must
//     take the catch(const std::bad_alloc&){throw;} clause specifically, not
//     merely produce an observably-equal std::bad_alloc via the general
//     catch(const std::exception&) clause. One representative site stands for
//     the mechanism per §17 dimension 11's usual population-validation shape
//     (the shared helper makes this true for all nineteen sites by
//     construction). ---
SSLM_TEST(TestBadAllocContractOpenFromMemoryPassthroughClauseIsSpecific, 315) {
	using namespace superslm_test;
	ArmInjectedFault(InjectThrowKind::kBadAlloc);
	bool threw_bad_alloc = false;
	uint8_t data[4] = {'S', 'S', 'L', 'M'};
	SslmArtifact out;
	SslmError err;
	try {
		SslmArtifact::OpenFromMemory(data, sizeof(data), out, &err);
	} catch (const std::bad_alloc&) {
		threw_bad_alloc = true;
	} catch (...) {
	}
	LastWrapClause clause = g_last_wrap_clause;
	DisarmInjectedFault();
	CHECK_MSG(threw_bad_alloc,
	          "OpenFromMemory did not propagate an injected std::bad_alloc unchanged");
	CHECK_MSG(clause == LastWrapClause::kBadAllocClause,
	          "OpenFromMemory's wrap must take the catch(const std::bad_alloc&){throw;} "
	          "clause specifically -- marker read %s",
	          clause == LastWrapClause::kNone ? "kNone (the wrap helper's marker was never set)"
	          : clause == LastWrapClause::kGeneralClause ? "kGeneralClause (wrong branch)"
	                                                      : "kBadAllocClause");
}

// --- Site 2/19: SslmArtifact::OpenFromFile (artifact.h:155) ---
SSLM_TEST(TestBadAllocContractOpenFromFile, 316) {
	SslmArtifact out;
	SslmError err;
	CheckBadAllocContractSite("SslmArtifact::OpenFromFile", [&] {
		SslmArtifact::OpenFromFile("this/path/does/not/exist.sslm", out, &err);
	});
}

// --- Site 3/19 (this fold's addition, condition 4(b)): SslmArtifact::
//     FingerprintHex (artifact.h:162) ---
SSLM_TEST(TestBadAllocContractFingerprintHex, 317) {
	SslmArtifact out;
	CheckBadAllocContractSite("SslmArtifact::FingerprintHex", [&] {
		(void)out.FingerprintHex();
	});
}

// --- Site 4/19: SslmTensorManifest::Parse (model.h:178), the direct-call
//     path ---
SSLM_TEST(TestBadAllocContractTensorManifestParseDirect, 318) {
	SslmSectionView section{};
	SslmTensorManifest out;
	std::string err;
	CheckBadAllocContractSite("SslmTensorManifest::Parse (direct)", [&] {
		SslmTensorManifest::Parse(section, out, &err);
	});
}

// --- Site 4/19, the Load-mediated path: the S-HARDEN-7 design's own defect
//     class (design §3.1: "Load's wrap ... is not a substitute for [a
//     site's] own independent wrap") -- confirms the seam is consulted by the
//     *Impl body itself, not only by whatever calls it directly, using a
//     fully valid artifact whose Weights section routes through
//     SslmTensorManifest::Parse from inside SslmModel::Load. ---
SSLM_TEST(TestBadAllocContractTensorManifestParseViaLoad, 319) {
	auto built = BuildFullyValidV2ArtifactForLoad();
	SslmModelView view;
	std::string err;
	CheckBadAllocContractSite("SslmTensorManifest::Parse (via SslmModel::Load)", [&] {
		SslmModel::Load(built.bytes.data(), built.bytes.size(), view, &err);
	});
}

// --- Site 5/19: SslmKeyedConstants::Parse (model.h:219) ---
SSLM_TEST(TestBadAllocContractKeyedConstantsParse, 320) {
	SslmSectionView section{};
	SslmKeyedConstants out;
	std::string err;
	CheckBadAllocContractSite("SslmKeyedConstants::Parse", [&] {
		SslmKeyedConstants::Parse(section, out, &err);
	});
}

// --- Site 6/19: ParseConfig (model.h:239) ---
SSLM_TEST(TestBadAllocContractParseConfig, 321) {
	SslmSectionView section{};
	SslmModelConfig out{};
	std::string err;
	CheckBadAllocContractSite("ParseConfig", [&] {
		ParseConfig(section, out, &err);
	});
}

// --- Site 7/19: ParseSigmoidLut (model.h:261) ---
SSLM_TEST(TestBadAllocContractParseSigmoidLut, 322) {
	SslmSectionView section{};
	SslmSigmoidLut out{};
	std::string err;
	CheckBadAllocContractSite("ParseSigmoidLut", [&] {
		ParseSigmoidLut(section, out, &err);
	});
}

// --- Site 8/19: SslmModel::Load (model.h:416), its own unwrapped surface
//     (model.cpp:711,788's string concatenations) -- exercised here via the
//     null-data path, which reaches Load's own body before any sub-parser
//     runs. ---
SSLM_TEST(TestBadAllocContractLoad, 323) {
	SslmModelView out;
	std::string err;
	CheckBadAllocContractSite("SslmModel::Load", [&] {
		SslmModel::Load(nullptr, 0, out, &err);
	});
}

// --- Site 9/19: TokenizerView::Open (tokenizer.h:35), the direct-call path
//     (tools/tok_verify.cpp's bypass shape) ---
SSLM_TEST(TestBadAllocContractTokenizerOpenDirect, 324) {
	SslmArtifact artifact;  // default: no sections, Ok() == false
	TokenizerView out;
	std::string err;
	CheckBadAllocContractSite("TokenizerView::Open (direct)", [&] {
		TokenizerView::Open(artifact, out, &err);
	});
}

// --- Site 9/19, the Load-mediated path: TokenizerView::Open called from
//     inside SslmModel::Load when a Tokenizer section is present -- the exact
//     bypass shape this fold's own §3.1 documents as previously missed. Uses
//     the real Qwen2.5-1.5B fixture artifact (the only fixture in this suite
//     that carries a genuine Tokenizer + UnicodeTables pair), read as raw
//     bytes and driven through SslmModel::Load directly rather than through
//     SslmArtifact::OpenFromFile. ---
SSLM_TEST(TestBadAllocContractTokenizerOpenViaLoad, 325) {
	std::string path = ResolveFixturePath("qwen2.5-1.5b.tok.sslm");
	if (path.empty()) {
		CHECK_MSG(false,
		          "fixture qwen2.5-1.5b.tok.sslm not found under tests/fixtures -- cannot "
		          "exercise TokenizerView::Open's Load-mediated path");
		return;
	}
	std::ifstream f(path, std::ios::binary);
	std::vector<uint8_t> bytes((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
	CHECK_MSG(!bytes.empty(), "fixture qwen2.5-1.5b.tok.sslm read as zero bytes from %s",
	          path.c_str());
	if (bytes.empty()) return;

	SslmModelView view;
	std::string err;
	CheckBadAllocContractSite("TokenizerView::Open (via SslmModel::Load)", [&] {
		SslmModel::Load(bytes.data(), bytes.size(), view, &err);
	});
}

// --- Site 10/19 (this fold's addition, condition 4(b)): TokenizerView::Encode
//     (tokenizer.h:44) ---
SSLM_TEST(TestBadAllocContractTokenizerEncode, 326) {
	auto ft = OpenFixtureTokenizer();
	CHECK_MSG(ft.view_ok, "TokenizerView::Open failed for the Encode injection fixture: %s",
	          ft.view_error.c_str());
	if (!ft.view_ok) return;
	CheckBadAllocContractSite("TokenizerView::Encode", [&] {
		(void)ft.view.Encode("the quick brown fox");
	});
}

// --- Site 11/19 (this fold's addition, condition 4(b)): TokenizerView::Decode
//     (tokenizer.h:48) ---
SSLM_TEST(TestBadAllocContractTokenizerDecode, 327) {
	auto ft = OpenFixtureTokenizer();
	CHECK_MSG(ft.view_ok, "TokenizerView::Open failed for the Decode injection fixture: %s",
	          ft.view_error.c_str());
	if (!ft.view_ok) return;
	std::vector<int32_t> ids = {1, 2, 3};
	CheckBadAllocContractSite("TokenizerView::Decode", [&] {
		(void)ft.view.Decode(ids);
	});
}

// --- Site 12/19 (this fold's addition, condition 4(b)): ComputeTensorEvidence
//     (proof_manifest.h:84) ---
SSLM_TEST(TestBadAllocContractComputeTensorEvidence, 328) {
	SslmTensorManifest manifest;
	CheckBadAllocContractSite("ComputeTensorEvidence", [&] {
		(void)ComputeTensorEvidence(manifest, SslmDtype::Int8);
	});
}

// --- Site 13/19 (this fold's addition, condition 4(b)):
//     ComputeWeightScaleEvidence (proof_manifest.h:99) ---
SSLM_TEST(TestBadAllocContractComputeWeightScaleEvidence, 329) {
	SslmTensorManifest manifest;
	CheckBadAllocContractSite("ComputeWeightScaleEvidence", [&] {
		(void)ComputeWeightScaleEvidence(manifest);
	});
}

// --- Site 14/19: HashSectionHex (proof_manifest.h:106) ---
SSLM_TEST(TestBadAllocContractHashSectionHex, 330) {
	SslmSectionView section{};
	CheckBadAllocContractSite("HashSectionHex", [&] {
		(void)HashSectionHex(section);
	});
}

// --- Site 15/19: BuildProofManifestJson (proof_manifest.h:128) ---
SSLM_TEST(TestBadAllocContractBuildProofManifestJson, 331) {
	SslmArtifact artifact;
	CheckBadAllocContractSite("BuildProofManifestJson", [&] {
		(void)BuildProofManifestJson(artifact);
	});
}

// --- Site 16/19: Sha256::Update (sha256.h:19). Cannot currently throw
//     anything (src/sha256.cpp operates on fixed-size stack buffers only,
//     design §3.1) -- this cell proves the injection seam itself fires
//     rather than a pre-existing real-world leak, the same shape every other
//     site's cell uses, applied to a site whose current implementation
//     happens not to need it yet. ---
SSLM_TEST(TestBadAllocContractSha256Update, 332) {
	Sha256 h;
	const uint8_t byte = 'x';
	CheckBadAllocContractSite("Sha256::Update", [&] {
		h.Update(&byte, 1);
	});
}

// --- Site 17/19: Sha256Hash (sha256.h:32). Same "cannot currently throw"
//     shape as site 16. ---
SSLM_TEST(TestBadAllocContractSha256HashFreeFunction, 333) {
	const uint8_t byte = 'x';
	uint8_t digest[32];
	CheckBadAllocContractSite("Sha256Hash", [&] {
		Sha256Hash(&byte, 1, digest);
	});
}

// --- Site 18/19 (this fold's addition, condition 4(b)): superslm::ToHex
//     (sha256.h:35). A FREE FUNCTION in namespace superslm, not a Sha256
//     member -- the design text writes "Sha256::ToHex" throughout §3.1/§3.2/
//     §3.3, but sha256.h:35 declares it outside the Sha256 class (Weak
//     finding, fourth-pass temper). This cell references the correct symbol,
//     superslm::ToHex; the design's prose is corrected by the planner, not by
//     this test. Cannot practically reach std::length_error given a fixed
//     64-character output (design §3.1) -- same "seam-fires, not a real
//     leak" shape as sites 16-17. ---
SSLM_TEST(TestBadAllocContractToHex, 334) {
	uint8_t digest[32] = {};
	CheckBadAllocContractSite("superslm::ToHex", [&] {
		(void)ToHex(digest);
	});
}

// --- Site 19/19 (T-1475, T-1495): JsonEscape (proof_manifest.h:52).
//     JsonEscapeImpl was promoted out of proof_manifest.cpp's anonymous
//     namespace and into the header, making it a derived member of the
//     contract's population -- until this cell, that promotion had a wrap
//     (WrapBadAllocContract) and a text-match production gate but no
//     executed fault-injection cell of its own. tools/ci/
//     check_bad_alloc_contract.py's _is_wrapped requires both the literal
//     "WrapBadAllocContract" and a `JsonEscapeImpl(` call to appear anywhere
//     in the definition's body -- it is a text match, not a structural one,
//     so a body that still NAMES the helper (e.g. left behind in a comment)
//     while no longer applying it reads as wrapped: exit 0, "OK -- every
//     derived member is wrapped," on a genuinely unwrapped member. Deleting
//     the wrap outright, as opposed to merely commenting it out, does not
//     pass the gate -- removing "WrapBadAllocContract" from the body fails
//     the first half of the same check, exit 1, naming JsonEscape -- so this
//     cell's fault-injection is not what catches THAT mutation; it is what
//     catches the gate's actual blind spot, a body that keeps the helper's
//     name in scope (e.g. in a comment) without calling it, which the
//     production gate reports OK on but this cell fails on both assertions.
//     Did not fail this suite's own check count (23,115 at this commit and
//     at its parent alike, before this cell). Also asserts the wrap's clause
//     marker, which only one other cell in this file does --
//     TestBadAllocContractOpenFromMemoryPassthroughClauseIsSpecific
//     (site 1/19, above) -- and here for a different reason: JsonEscape's
//     own public entry point wraps JsonEscapeImpl directly (one layer), so
//     an injected std::length_error must take the general
//     catch(const std::exception&) clause, unlike the nested call inside
//     BuildProofManifestJsonImpl
//     (proof_manifest.cpp), where JsonEscape's own inner wrap already
//     narrows to std::bad_alloc before the outer wrap's
//     catch(const std::bad_alloc&){throw;} clause re-catches it -- a path
//     this cell does not exercise, since it calls the public JsonEscape
//     directly. ---
SSLM_TEST(TestBadAllocContractJsonEscape, 335) {
	using namespace superslm_test;
	CheckBadAllocContractSite("JsonEscape", [&] {
		(void)JsonEscape("test");
	});
	CHECK_MSG(g_last_wrap_clause == LastWrapClause::kGeneralClause,
	          "JsonEscape's own wrap must take the catch(const std::exception&) general "
	          "clause for a converted std::length_error -- marker read %s",
	          g_last_wrap_clause == LastWrapClause::kNone
	              ? "kNone (the wrap helper's marker was never set)"
	          : g_last_wrap_clause == LastWrapClause::kBadAllocClause
	              ? "kBadAllocClause (wrong branch)"
	              : "kGeneralClause");
}

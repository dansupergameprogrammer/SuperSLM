// tests/areas/tokenizer.cpp -- T-1574 test suite split, Stage 4.
// Area #2 (Claude/Plans/SuperSLM_TestSuiteSplit_Plan.md §4): tests calling
// through src/tokenizer.cpp's public contract (TokenizerView::Open/Encode/
// Decode). Extracted verbatim from tests/test_main.cpp (order keys 37-110).
// SingleVocabTokenizer is owned locally (single-area use, confirmed by grep).
// FixtureTokenizer/OpenFixtureTokenizer are consumed from
// tests/support/shared_fixtures.h -- promoted there at Stage 2 because
// candidate area #10 (bad_alloc_contract.cpp) also needed them; this area's
// own note on the crossing is left in place below, where it was originally
// written.

#include "superslm/artifact.h"
#include "superslm/tokenizer.h"
#include "sslm_fixtures.h"
#include "sslm_tokenizer_fixtures.h"
#include "sslm_tokenizer_hostile_fixtures.h"

#include "support/shared_fixtures.h"
#include "support/test_harness.h"
#include "support/test_registry.h"

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

using namespace superslm;
using namespace superslm_test;

// ---------------------------------------------------------------------------
// Curie's S1 tokenizer red suite (SuperSLM_Plan.md §10; Claude/Curie/
// SuperSLM_S1_Tokenizer_TestDesign-2026-07-19.md). The runtime byte-level BPE
// algorithm is already proven bit-for-bit against the upstream HF tokenizer by
// the Python reference (tools/convert_tokenizer.py, 0 mismatch over 2000+
// adversarial+multilingual lines) — this suite is the C++ gate that proves the
// ported TokenizerView reproduces those upstream ids. TokenizerView is fully
// built (src/tokenizer.cpp, shipped at S-HARDEN-2); every cell below is green
// against it.
// ---------------------------------------------------------------------------

// FixtureTokenizer/OpenFixtureTokenizer moved to tests/support/shared_fixtures.h
// (T-1574 Stage 2, §4's standing promotion rule: a third crossing discovered
// mid-migration -- used by this area and by candidate area #10
// bad_alloc_contract.cpp). Already visible here via test_main.cpp's own
// #include "support/shared_fixtures.h".

// --- The golden gate (dim 10 — the load-bearing feature oracle): Encode()
//     executed against the real Qwen2.5-1.5B artifact must reproduce the
//     upstream HF tokenizer's ids, not merely agree with itself. ---

SSLM_TEST(TestTokenizerGoldenEncodeMatchesUpstreamIds, 38) {
	auto ft = OpenFixtureTokenizer();
	CHECK_MSG(ft.artifact_ok, "fixture artifact failed to load: %s", ft.artifact_error.c_str());
	CHECK_MSG(ft.view_ok, "TokenizerView::Open failed: %s", ft.view_error.c_str());

	std::string golden_path = ResolveFixturePath("qwen_tok_golden.gld");
	CHECK_MSG(!golden_path.empty(), "golden pack qwen_tok_golden.gld not found under tests/fixtures");
	if (golden_path.empty()) return;
	GoldenPack golden = LoadGoldenPack(golden_path);
	CHECK_MSG(golden.ok, "golden pack failed to parse: %s", golden.error.c_str());
	if (!golden.ok) return;
	CHECK(golden.record_count == golden.records.size());

	for (size_t i = 0; i < golden.records.size(); ++i) {
		const auto& rec = golden.records[i];
		std::vector<int32_t> got = ft.view.Encode(rec.text);
		CHECK_MSG(got == rec.ids,
		          "record %zu \"%s\": Encode produced %zu ids, golden (upstream HF) has %zu",
		          i, rec.text.c_str(), got.size(), rec.ids.size());
	}
}

SSLM_TEST(TestTokenizerGoldenIdsHashMatchesConverter, 39) {
	auto ft = OpenFixtureTokenizer();
	CHECK_MSG(ft.artifact_ok, "fixture artifact failed to load: %s", ft.artifact_error.c_str());
	CHECK_MSG(ft.view_ok, "TokenizerView::Open failed: %s", ft.view_error.c_str());

	std::string golden_path = ResolveFixturePath("qwen_tok_golden.gld");
	CHECK_MSG(!golden_path.empty(), "golden pack qwen_tok_golden.gld not found under tests/fixtures");
	if (golden_path.empty()) return;
	GoldenPack golden = LoadGoldenPack(golden_path);
	CHECK_MSG(golden.ok, "golden pack failed to parse: %s", golden.error.c_str());
	if (!golden.ok) return;

	// Recomputes tools/convert_tokenizer.py's emit_golden() hash, but over THIS
	// tokenizer's Encode() output rather than the golden file's own stored ids —
	// a feature oracle (does Encode() reproduce the hash an independent, correct
	// upstream run produced), not a self-consistency check against the fixture's
	// own bytes.
	Sha256 hasher;
	for (const auto& rec : golden.records) {
		std::vector<int32_t> ids = ft.view.Encode(rec.text);
		hasher.Update(reinterpret_cast<const uint8_t*>(rec.text.data()), rec.text.size());
		const uint8_t zero = 0;
		hasher.Update(&zero, 1);
		for (int32_t id : ids) {
			const uint32_t u = static_cast<uint32_t>(id);
			const uint8_t le[4] = {static_cast<uint8_t>(u & 0xFF), static_cast<uint8_t>((u >> 8) & 0xFF),
			                       static_cast<uint8_t>((u >> 16) & 0xFF), static_cast<uint8_t>((u >> 24) & 0xFF)};
			hasher.Update(le, 4);
		}
	}
	uint8_t digest[32];
	hasher.Final(digest);
	CHECK_MSG(std::memcmp(digest, golden.ids_hash.data(), 32) == 0,
	          "Encode()-derived ids_hash %s does not match the converter's stored ids_hash %s",
	          ToHex(digest).c_str(), ToHex(golden.ids_hash.data()).c_str());
}

// --- Decode round-trip: Encode NFC-normalizes, so Decode(golden.ids) equals the
//     NFC form of the line. ASCII lines assert the exact match (NFC is a no-op);
//     the rest assert the idempotent Decode(Encode(t)) == Decode(golden.ids)
//     round-trip, avoiding the need for an NFC oracle in C++. ---

SSLM_TEST(TestTokenizerDecodeRoundTrip, 40) {
	auto ft = OpenFixtureTokenizer();
	CHECK_MSG(ft.artifact_ok, "fixture artifact failed to load: %s", ft.artifact_error.c_str());
	CHECK_MSG(ft.view_ok, "TokenizerView::Open failed: %s", ft.view_error.c_str());

	std::string golden_path = ResolveFixturePath("qwen_tok_golden.gld");
	CHECK_MSG(!golden_path.empty(), "golden pack qwen_tok_golden.gld not found under tests/fixtures");
	if (golden_path.empty()) return;
	GoldenPack golden = LoadGoldenPack(golden_path);
	CHECK_MSG(golden.ok, "golden pack failed to parse: %s", golden.error.c_str());
	if (!golden.ok) return;

	for (size_t i = 0; i < golden.records.size(); ++i) {
		const auto& rec = golden.records[i];
		std::string decoded_golden = ft.view.Decode(rec.ids);
		if (IsAsciiOnly(rec.text)) {
			CHECK_MSG(decoded_golden == rec.text,
			          "record %zu (ASCII) \"%s\": Decode(golden.ids) == \"%s\", want an exact match",
			          i, rec.text.c_str(), decoded_golden.c_str());
		} else {
			std::string decoded_roundtrip = ft.view.Decode(ft.view.Encode(rec.text));
			CHECK_MSG(decoded_roundtrip == decoded_golden,
			          "record %zu (non-ASCII) \"%s\": Decode(Encode(t)) != Decode(golden.ids)", i,
			          rec.text.c_str());
		}
	}
}

// --- Targeted edges: no HF reference needed. ---

SSLM_TEST(TestTokenizerOpensFixtureArtifact, 37) {
	auto ft = OpenFixtureTokenizer();
	CHECK_MSG(ft.artifact_ok, "fixture artifact failed to load: %s", ft.artifact_error.c_str());
	CHECK_MSG(ft.view_ok, "TokenizerView::Open failed: %s", ft.view_error.c_str());
	CHECK(ft.view.Ok());
	CHECK(ft.view.VocabSize() > 0);
}

SSLM_TEST(TestTokenizerEncodeEmptyStringYieldsEmptyIds, 41) {
	auto ft = OpenFixtureTokenizer();
	CHECK_MSG(ft.artifact_ok, "fixture artifact failed to load: %s", ft.artifact_error.c_str());
	CHECK_MSG(ft.view_ok, "TokenizerView::Open failed: %s", ft.view_error.c_str());
	CHECK(ft.view.Encode("").empty());
}

SSLM_TEST(TestTokenizerSpecialTokenIdMatchesArtifactDeclaration, 42) {
	auto ft = OpenFixtureTokenizer();
	CHECK_MSG(ft.artifact_ok, "fixture artifact failed to load: %s", ft.artifact_error.c_str());
	if (!ft.artifact_ok) return;

	const SslmSectionView* tok_section = ft.artifact.Section(SslmSectionType::Tokenizer);
	CHECK_MSG(tok_section != nullptr, "artifact has no Tokenizer section");
	if (!tok_section) return;
	TokBlobInfo blob = ParseTokenizerBlob(tok_section->data, static_cast<size_t>(tok_section->byte_size));
	CHECK_MSG(blob.ok, "TOK1 blob failed to parse: %s", blob.error.c_str());
	if (!blob.ok) return;

	const char* kSpecial = "<|im_start|>";
	const TokBlobSpecial* declared = nullptr;
	for (const auto& sp : blob.specials) {
		if (sp.content == kSpecial) {
			declared = &sp;
			break;
		}
	}
	CHECK_MSG(declared != nullptr, "artifact's Tokenizer section does not declare %s among its %u specials",
	          kSpecial, blob.special_count);
	// Independently confirmed against this exact fixture file (Claude/Curie/
	// SuperSLM_S1_Tokenizer_TestDesign-2026-07-19.md §2): guards a bug in
	// ParseTokenizerBlob from masquerading as a tokenizer defect below.
	if (declared) CHECK(declared->id == 151644u);

	CHECK_MSG(ft.view_ok, "TokenizerView::Open failed: %s", ft.view_error.c_str());
	std::vector<int32_t> ids = ft.view.Encode(kSpecial);
	CHECK_MSG(ids.size() == 1, "Encode(%s) produced %zu ids, want exactly 1", kSpecial, ids.size());
	if (declared && ids.size() == 1) {
		CHECK_MSG(ids[0] == static_cast<int32_t>(declared->id),
		          "Encode(%s) == %d, artifact declares id %u", kSpecial, ids[0], declared->id);
	}
	CHECK(ft.view.Decode(ids) == kSpecial);
}

SSLM_TEST(TestTokenizerVocabSizeMatchesArtifactDeclaration, 43) {
	auto ft = OpenFixtureTokenizer();
	CHECK_MSG(ft.artifact_ok, "fixture artifact failed to load: %s", ft.artifact_error.c_str());
	if (!ft.artifact_ok) return;

	const SslmSectionView* tok_section = ft.artifact.Section(SslmSectionType::Tokenizer);
	CHECK_MSG(tok_section != nullptr, "artifact has no Tokenizer section");
	if (!tok_section) return;
	TokBlobInfo blob = ParseTokenizerBlob(tok_section->data, static_cast<size_t>(tok_section->byte_size));
	CHECK_MSG(blob.ok, "TOK1 blob failed to parse: %s", blob.error.c_str());
	if (!blob.ok) return;

	// Independently confirmed against this exact fixture file (Claude/Curie/
	// SuperSLM_S1_Tokenizer_TestDesign-2026-07-19.md §2): guards a bug in
	// ParseTokenizerBlob from masquerading as a tokenizer defect below.
	CHECK_MSG(blob.vocab_count == 151665u,
	          "TOK1 blob parse produced vocab_count %u, this fixture is known to declare 151665",
	          blob.vocab_count);

	CHECK_MSG(ft.view_ok, "TokenizerView::Open failed: %s", ft.view_error.c_str());
	CHECK_MSG(ft.view.VocabSize() == static_cast<int32_t>(blob.vocab_count),
	          "VocabSize() == %d, artifact declares vocab_count %u", ft.view.VocabSize(), blob.vocab_count);
}

SSLM_TEST(TestTokenizerAsciiStringRoundTrips, 44) {
	auto ft = OpenFixtureTokenizer();
	CHECK_MSG(ft.artifact_ok, "fixture artifact failed to load: %s", ft.artifact_error.c_str());
	CHECK_MSG(ft.view_ok, "TokenizerView::Open failed: %s", ft.view_error.c_str());
	const std::string text = "The quick brown fox jumps over 42 lazy dogs!";
	CHECK(ft.view.Decode(ft.view.Encode(text)) == text);
}

// S-HARDEN-2 (F15's "every raw byte mapping" class): the GPT-2-style
// byte-level BPE base vocabulary is a bijection over every possible byte
// value 0-255 (bytes_to_unicode(), tools/convert_tokenizer.py) -- every one
// of the 256 entries in the artifact's byte_to_id table must name an id
// whose OWN raw vocabulary bytes are exactly that one byte. Checked directly
// against the artifact's bytes (ParseTokenizerBlob's byte_to_id and
// id_to_bytes, independent of src/tokenizer.cpp), never through
// Decode() -- a single extended-range byte (>= 0x80) decoded ALONE is
// correctly not valid standalone UTF-8 under F7's strict policy, which is a
// different claim from "the base vocabulary has one slot per byte value";
// testing Decode() on an isolated out-of-context id would conflate the two
// and assert something F7 correctly makes false.
SSLM_TEST(TestTokenizerByteToIdBijectsOntoEveryRawByteValue, 45) {
	auto ft = OpenFixtureTokenizer();
	CHECK_MSG(ft.artifact_ok, "fixture artifact failed to load: %s", ft.artifact_error.c_str());
	if (!ft.artifact_ok) return;

	const SslmSectionView* tok_section = ft.artifact.Section(SslmSectionType::Tokenizer);
	CHECK_MSG(tok_section != nullptr, "artifact has no Tokenizer section");
	if (!tok_section) return;
	TokBlobInfo blob = ParseTokenizerBlob(tok_section->data, static_cast<size_t>(tok_section->byte_size));
	CHECK_MSG(blob.ok, "TOK1 blob failed to parse: %s", blob.error.c_str());
	if (!blob.ok) return;
	CHECK_MSG(blob.byte_to_id.size() == 256, "byte_to_id table has %zu entries, want 256",
	          blob.byte_to_id.size());

	for (int b = 0; b < 256 && b < static_cast<int>(blob.byte_to_id.size()); ++b) {
		const uint32_t id = blob.byte_to_id[b];
		CHECK_MSG(id < blob.id_to_bytes.size(), "byte_to_id[0x%02x] == %u, outside id_to_bytes (size %zu)", b, id,
		          blob.id_to_bytes.size());
		if (id >= blob.id_to_bytes.size()) continue;
		const std::string& raw = blob.id_to_bytes[id];
		const std::string want(1, static_cast<char>(static_cast<unsigned char>(b)));
		CHECK_MSG(raw == want, "byte_to_id[0x%02x] == id %u, whose own vocabulary bytes are %zu byte(s), want "
		          "exactly the one byte 0x%02x", b, id, raw.size(), b);
	}
}

// ---------------------------------------------------------------------------
// Curie's T-129 TOK1/UNI1 sub-parse hostile-input suite (DecisionLog D-SLM62).
// TokenizerView::Open parses the TOK1 (Tokenizer section) and UNI1 (UnicodeTables
// section) blobs AFTER the loader (SslmArtifact::OpenFromMemory) has verified
// whole-file integrity — a malicious artifact can carry a valid self-hash re-
// stamped over a tampered sub-blob, so this sub-parse is its own hostile-input
// trust boundary, independent of the outer loader's. This suite is the systematic
// sweep of that boundary: every cell below starts from the minimal valid TOK1/UNI1
// blobs in sslm_tokenizer_hostile_fixtures.h, mutates exactly one field, rebuilds
// the artifact (which re-stamps the whole-file integrity hash over the mutated
// bytes, so the OUTER loader always still accepts it), and asserts
// TokenizerView::Open returns false, Ok() is false, and nothing crashes.
//
// Poirot found two OOB defects in this parse (unchecked vocab offsets; a u32
// length-math overflow) that Brunel fixed at 1bb19a6. A cell already defended by
// that fix passes GREEN here — that is the certification this suite exists to
// produce. A cell that still accepts the malformed input, or crashes, fails RED —
// a finding for Brunel.
// ---------------------------------------------------------------------------

// --- The feature oracle: the minimal fixture this suite mutates from is itself
//     spec-faithful — it Opens and Encode/Decode round-trip correctly. Every
//     hostile cell below attributes a rejection to its one named mutation; this
//     cell is what proves the baseline isn't rejecting (or wrongly accepting) for
//     some unrelated reason of its own. ---

SSLM_TEST(TestMinimalTokenizerArtifactOpensAndRoundTrips, 58) {
	auto tok1 = MakeMinimalValidTok1();
	auto uni1 = MakeMinimalValidUni1();
	auto built = BuildTokenizerArtifact(tok1.bytes, uni1.bytes);

	SslmArtifact artifact;
	SslmError aerr;
	auto status = SslmArtifact::OpenFromMemory(built.bytes.data(), built.bytes.size(), artifact, &aerr);
	CHECK_MSG(status == SslmStatus::Ok, "minimal fixture's outer artifact failed to load: got %s: %s",
	          SslmStatusName(status), aerr.message.c_str());
	if (status != SslmStatus::Ok) return;

	TokenizerView view;
	std::string terr;
	bool opened = TokenizerView::Open(artifact, view, &terr);
	CHECK_MSG(opened, "TokenizerView::Open failed on the minimal valid fixture: %s", terr.c_str());
	if (!opened) return;
	CHECK(view.Ok());
	// S-HARDEN-2 (F18): 5 entries now — "c","a","t","ca","<eos>" — the fixture's
	// special token has its own real vocabulary slot (id 4) rather than the prior
	// out-of-vocabulary id 1000 a committed test used to require.
	CHECK(view.VocabSize() == 5);

	// "cat": pretokenizes as one word piece (all ASCII letters); byte-level ids
	// [byte_to_id['c'],['a'],['t']] = [0,1,2]; the one merge (0,1)->3 fires once at
	// the head -> [3,2]. Decode: id3 -> "ca", id2 -> "t" -> "cat".
	std::vector<int32_t> ids = view.Encode("cat");
	std::vector<int32_t> want_ids = {3, 2};
	CHECK_MSG(ids == want_ids, "Encode(\"cat\") produced %zu id(s), want [3,2]", ids.size());
	CHECK(view.Decode(ids) == "cat");

	// The special token matches the whole input and emits its declared id, not a
	// byte-level encoding of its text. S-HARDEN-2 (F18): id 4 is now an id INSIDE
	// the vocabulary (unlike the prior fixture's 1000), and it decodes back to the
	// special's own content via the vocabulary entry, not a raw-index escape hatch.
	std::vector<int32_t> special_ids = view.Encode("<eos>");
	CHECK_MSG(special_ids.size() == 1 && special_ids[0] == 4,
	          "Encode(\"<eos>\") produced %zu id(s), want exactly [4]", special_ids.size());
	CHECK(view.Decode(special_ids) == "<eos>");
}

// ---------------------------------------------------------------------------
// S-HARDEN-2 (F7): Decode's documented malformed-UTF-8 policy
// (include/superslm/tokenizer.h: "invalid sequences pass through as
// replacement chars") through the ONE strict decoder shared by encode and
// decode. Each cell builds a tiny TOK1 whose vocabulary entries carry the
// exact raw bytes under test — independent of MakeMinimalValidTok1 (which is
// all-ASCII) — and asserts Decode()'s output, never Encode() (encode's input
// is a std::string_view over already-decoded text; this suite is about bytes
// arriving FROM the vocabulary, the concatenated-token-bytes path F7 names).
// ---------------------------------------------------------------------------

namespace {

// TokenizerView::Open's contract (include/superslm/tokenizer.h) is explicit:
// "The artifact must outlive the view — table pointers reference its bytes."
// OpenTokenizerWithSingleVocabEntry's fixtures must therefore keep the
// artifact and the view together, exactly like FixtureTokenizer above — a
// helper that opened a local SslmArtifact and returned only the TokenizerView
// left the artifact's owned byte buffer destroyed at return while the
// returned view's Decode() still read through pointers into it.
struct SingleVocabTokenizer {
	SslmArtifact artifact;
	TokenizerView view;
};

// A single-entry TOK1 whose one vocabulary slot (id 0) carries exactly
// `raw_bytes` — enough to exercise Decode({0, ...}) against those bytes, with
// no merges/specials in the way.
SingleVocabTokenizer OpenTokenizerWithSingleVocabEntry(const std::string& raw_bytes, std::string* out_err = nullptr) {
	SingleVocabTokenizer result;
	std::array<uint32_t, 256> byte_to_id{};
	std::vector<TokVocabEntry> vocab = {{raw_bytes}};
	auto tok1 = BuildTok1(byte_to_id, vocab, {}, {});
	auto uni1 = MakeMinimalValidUni1();
	auto built = BuildTokenizerArtifact(tok1.bytes, uni1.bytes);

	SslmError aerr;
	auto status = SslmArtifact::OpenFromMemory(built.bytes.data(), built.bytes.size(), result.artifact, &aerr);
	if (status != SslmStatus::Ok) {
		if (out_err) *out_err = std::string("outer artifact rejected: ") + SslmStatusName(status);
		return result;
	}
	std::string terr;
	bool opened = TokenizerView::Open(result.artifact, result.view, &terr);
	if (!opened && out_err) *out_err = terr;
	return result;
}

const std::string kFffdUtf8 = "\xEF\xBF\xBD";  // U+FFFD, the documented replacement char

}  // namespace

SSLM_TEST(TestDecodeSubstitutesReplacementCharForOverlongTwoByteSequence, 46) {
	// 0xC0 0x80: the canonical two-byte encoding of NUL — always overlong (NUL is
	// representable in one byte); C0/C1 can never start a well-formed sequence.
	std::string err;
	auto t = OpenTokenizerWithSingleVocabEntry(std::string("\xC0\x80", 2), &err);
	CHECK_MSG(t.view.Ok(), "setup: %s", err.c_str());
	CHECK(t.view.Decode({0}) == kFffdUtf8 + kFffdUtf8);  // both bytes are individually invalid leads
}

SSLM_TEST(TestDecodeSubstitutesReplacementCharForOverlongThreeByteSequence, 47) {
	// 0xE0 0x80 0x80: an overlong 3-byte encoding (E0's first continuation must be
	// >= 0xA0; here it is 0x80). Per the Unicode Standard's "maximal subpart of an
	// ill-formed subsequence" algorithm, a lead whose FIRST continuation byte
	// fails the range check forms a one-byte maximal subpart (E0 alone) — the
	// failing continuation byte is NOT consumed with it, so it is re-scanned at
	// the top of the loop; a bare 0x80 byte with nothing before it is itself an
	// ill-formed one-byte subpart. Three independently-invalid bytes -> three
	// U+FFFD, not one — this is the textbook worked example for this policy.
	std::string err;
	auto t = OpenTokenizerWithSingleVocabEntry(std::string("\xE0\x80\x80", 3), &err);
	CHECK_MSG(t.view.Ok(), "setup: %s", err.c_str());
	CHECK(t.view.Decode({0}) == kFffdUtf8 + kFffdUtf8 + kFffdUtf8);
}

SSLM_TEST(TestDecodeSubstitutesReplacementCharForSurrogateCodepoint, 48) {
	// 0xED 0xA0 0x80: encodes U+D800, a UTF-16 surrogate half — UTF-8 must never
	// encode U+D800-U+DFFF (ED's first continuation must be <= 0x9F; here 0xA0).
	// Same maximal-subpart shape as the overlong-3-byte cell above: ED's first
	// continuation fails the range check, so ED alone is the one-byte subpart,
	// and the un-consumed 0xA0 and 0x80 are each their own ill-formed subpart —
	// three U+FFFD.
	std::string err;
	auto t = OpenTokenizerWithSingleVocabEntry(std::string("\xED\xA0\x80", 3), &err);
	CHECK_MSG(t.view.Ok(), "setup: %s", err.c_str());
	CHECK(t.view.Decode({0}) == kFffdUtf8 + kFffdUtf8 + kFffdUtf8);
}

SSLM_TEST(TestDecodeSubstitutesReplacementCharForCodepointPastU10FFFF, 49) {
	// 0xF5 alone: any lead >= 0xF5 can only encode a codepoint past U+10FFFF.
	std::string err;
	auto t = OpenTokenizerWithSingleVocabEntry(std::string("\xF5\x80\x80\x80", 4), &err);
	CHECK_MSG(t.view.Ok(), "setup: %s", err.c_str());
	// F5 itself is rejected outright (one U+FFFD, one byte consumed); the three
	// trailing 0x80 bytes are then each a lone continuation byte (one U+FFFD
	// each) — four total.
	CHECK(t.view.Decode({0}) == kFffdUtf8 + kFffdUtf8 + kFffdUtf8 + kFffdUtf8);
}

SSLM_TEST(TestDecodeSubstitutesReplacementCharForF4WithContinuationPastMax, 50) {
	// 0xF4 0x90 0x80 0x80: F4's first continuation must be <= 0x8F (else the
	// codepoint exceeds U+10FFFF) — 0x90 is one past that. Same maximal-subpart
	// shape as the two three-byte cells above, one byte longer: F4 alone is the
	// one-byte subpart, and the three un-consumed trailing bytes (0x90, 0x80,
	// 0x80) are each their own ill-formed one-byte subpart — four U+FFFD.
	std::string err;
	auto t = OpenTokenizerWithSingleVocabEntry(std::string("\xF4\x90\x80\x80", 4), &err);
	CHECK_MSG(t.view.Ok(), "setup: %s", err.c_str());
	CHECK(t.view.Decode({0}) == kFffdUtf8 + kFffdUtf8 + kFffdUtf8 + kFffdUtf8);
}

SSLM_TEST(TestDecodeSubstitutesReplacementCharForOverlongFourByteSequence, 51) {
	// 0xF0 0x80 0x80 0x80: an overlong 4-byte encoding (F0's first continuation
	// must be >= 0x90; here it is 0x80). Same maximal-subpart shape as the
	// overlong-3-byte and surrogate cells above, one byte longer: F0's first
	// continuation fails the range check, so F0 alone is the one-byte subpart,
	// and the three un-consumed trailing bytes are each their own ill-formed
	// one-byte subpart — four U+FFFD, not one.
	std::string err;
	auto t = OpenTokenizerWithSingleVocabEntry(std::string("\xF0\x80\x80\x80", 4), &err);
	CHECK_MSG(t.view.Ok(), "setup: %s", err.c_str());
	CHECK(t.view.Decode({0}) == kFffdUtf8 + kFffdUtf8 + kFffdUtf8 + kFffdUtf8);
}

SSLM_TEST(TestDecodeSubstitutesReplacementCharForTruncatedFourByteSequence, 52) {
	// 0xF0 0x90 0x80: the first three bytes of the 4-byte sequence for U+10000,
	// missing its final continuation byte. F0's first continuation (0x90) and
	// second continuation (0x80) are both individually valid, so the decoder
	// consumes all three bytes as one truncated unit and substitutes exactly one
	// U+FFFD — the 4-byte analogue of the already-tested 3-byte truncation cell.
	std::string err;
	auto t = OpenTokenizerWithSingleVocabEntry(std::string("\xF0\x90\x80", 3), &err);
	CHECK_MSG(t.view.Ok(), "setup: %s", err.c_str());
	CHECK(t.view.Decode({0}) == kFffdUtf8);
}

SSLM_TEST(TestDecodeSubstitutesReplacementCharForLoneContinuationByte, 53) {
	// 0x80 alone: a continuation byte with no lead byte before it.
	std::string err;
	auto t = OpenTokenizerWithSingleVocabEntry(std::string("\x80", 1), &err);
	CHECK_MSG(t.view.Ok(), "setup: %s", err.c_str());
	CHECK(t.view.Decode({0}) == kFffdUtf8);
}

SSLM_TEST(TestDecodeSubstitutesReplacementCharForTruncatedSequenceAtEnd, 54) {
	// 0xE2 0x82: the first two bytes of the 3-byte sequence for U+20AC ('€'),
	// missing its final continuation byte.
	std::string err;
	auto t = OpenTokenizerWithSingleVocabEntry(std::string("\xE2\x82", 2), &err);
	CHECK_MSG(t.view.Ok(), "setup: %s", err.c_str());
	CHECK(t.view.Decode({0}) == kFffdUtf8);
}

// Regression guard, 2026-07-22: OpenTokenizerWithSingleVocabEntry used to open a
// local SslmArtifact and return only the TokenizerView built against it. The
// artifact -- and the byte buffer TokenizerView::Open's contract requires it to
// keep alive (include/superslm/tokenizer.h: "The artifact must outlive the
// view") -- was destroyed at the helper's return, so every Decode() call after
// it read through a dangling pointer (CI's ASan leg caught this as a
// heap-use-after-free in Rd32 at tokenizer.cpp:13, freed via ~vector, allocated
// via vector::_M_assign_aux -- SslmArtifact::OpenFromMemory's
// `bytes_.assign(data, data + size)`). The fix pairs the artifact with the view
// in one returned struct (SingleVocabTokenizer), the same lifetime idiom
// FixtureTokenizer already used above for exactly this reason. This cell forces
// a burst of unrelated heap allocation and deallocation between the helper's
// return and the Decode() call -- the shape most likely to overwrite a freed
// buffer with different bytes -- and asserts the exact expected output, so a
// reintroduced bare-view return is likely to surface here as corrupted bytes on
// a normal (non-ASan) build, not only under CI's sanitizer leg.
SSLM_TEST(TestSingleVocabTokenizerSurvivesHeapChurnBetweenOpenAndDecode, 55) {
	std::string err;
	auto t = OpenTokenizerWithSingleVocabEntry(std::string("\xC0\x80", 2), &err);
	CHECK_MSG(t.view.Ok(), "setup: %s", err.c_str());
	{
		std::vector<std::vector<uint8_t>> churn;
		churn.reserve(64);
		for (int i = 0; i < 64; ++i) churn.emplace_back(256, uint8_t(0xAB + i));
	}
	CHECK(t.view.Decode({0}) == kFffdUtf8 + kFffdUtf8);
}

SSLM_TEST(TestDecodeReconstructsSequenceSplitAcrossTokenBoundary, 56) {
	// U+00E9 ('é') encodes as 0xC3 0xA9. Split it across two vocabulary entries —
	// id 0 carries only the lead byte 0xC3, id 1 carries only the continuation
	// byte 0xA9 — so NEITHER token's bytes are individually valid UTF-8, but
	// concatenated in order they are. F7's own scope: "invalid bytes split across
	// token boundaries" — this proves the split reassembles rather than each
	// half independently substituting U+FFFD.
	std::array<uint32_t, 256> byte_to_id{};
	std::vector<TokVocabEntry> vocab = {{std::string("\xC3", 1)}, {std::string("\xA9", 1)}};
	auto tok1 = BuildTok1(byte_to_id, vocab, {}, {});
	auto uni1 = MakeMinimalValidUni1();
	auto built = BuildTokenizerArtifact(tok1.bytes, uni1.bytes);

	SslmArtifact artifact;
	SslmError aerr;
	auto status = SslmArtifact::OpenFromMemory(built.bytes.data(), built.bytes.size(), artifact, &aerr);
	CHECK_MSG(status == SslmStatus::Ok, "outer artifact failed to load: %s", SslmStatusName(status));
	if (status != SslmStatus::Ok) return;
	TokenizerView view;
	std::string terr;
	bool opened = TokenizerView::Open(artifact, view, &terr);
	CHECK_MSG(opened, "TokenizerView::Open failed: %s", terr.c_str());
	if (!opened) return;

	CHECK(view.Decode({0, 1}) == "\xC3\xA9");  // reassembled: valid 'é', no U+FFFD
	// Each half decoded ALONE, by contrast, is genuinely malformed on its own —
	// confirms the two single-id decodes are not accidentally also valid.
	CHECK(view.Decode({0}) == kFffdUtf8);
	CHECK(view.Decode({1}) == kFffdUtf8);
}

SSLM_TEST(TestDecodeAndEncodeShareOneStrictDecoderOnWellFormedMultibyteText, 57) {
	// A well-formed non-ASCII round trip through the SAME decoder Encode's input
	// path uses (Utf8Decode -> NFC -> pretokenize -> BPE), confirming the shared
	// strict decoder does not regress ordinary valid text: 'é' (U+00E9, already
	// NFC-composed) byte-BPE-encodes to its own byte-level id and decodes back
	// unchanged.
	std::array<uint32_t, 256> byte_to_id{};
	// Map both raw bytes of 'é' to distinct ids so Decode's byte-level path is
	// exercised without relying on NFC composing them into one vocabulary entry.
	byte_to_id[0xC3] = 0;
	byte_to_id[0xA9] = 1;
	std::vector<TokVocabEntry> vocab = {{std::string("\xC3", 1)}, {std::string("\xA9", 1)}};
	auto tok1 = BuildTok1(byte_to_id, vocab, {}, {});
	auto uni1 = MakeMinimalValidUni1();  // already declares NFC-relevant tables for U+00E9
	auto built = BuildTokenizerArtifact(tok1.bytes, uni1.bytes);

	SslmArtifact artifact;
	SslmError aerr;
	auto status = SslmArtifact::OpenFromMemory(built.bytes.data(), built.bytes.size(), artifact, &aerr);
	CHECK_MSG(status == SslmStatus::Ok, "outer artifact failed to load: %s", SslmStatusName(status));
	if (status != SslmStatus::Ok) return;
	TokenizerView view;
	std::string terr;
	bool opened = TokenizerView::Open(artifact, view, &terr);
	CHECK_MSG(opened, "TokenizerView::Open failed: %s", terr.c_str());
	if (!opened) return;

	std::vector<int32_t> ids = view.Encode("\xC3\xA9");  // "é"
	CHECK(view.Decode(ids) == "\xC3\xA9");
}

// --- Structural: TokenizerView::Open requires both sections outright. ---

SSLM_TEST(TestOpenRejectsArtifactMissingTokenizerSection, 59) {
	auto uni1 = MakeMinimalValidUni1();
	auto built = BuildArtifactMissingTokenizer(uni1.bytes);

	SslmArtifact artifact;
	SslmError aerr;
	auto status = SslmArtifact::OpenFromMemory(built.bytes.data(), built.bytes.size(), artifact, &aerr);
	CHECK_MSG(status == SslmStatus::Ok, "outer artifact (no Tokenizer section) must still load: got %s",
	          SslmStatusName(status));
	if (status != SslmStatus::Ok) return;

	TokenizerView view;
	std::string terr;
	bool opened = TokenizerView::Open(artifact, view, &terr);
	CHECK_MSG(!opened, "TokenizerView::Open accepted an artifact with no Tokenizer section");
	CHECK(!view.Ok());
}

SSLM_TEST(TestOpenRejectsArtifactMissingUnicodeTablesSection, 60) {
	auto tok1 = MakeMinimalValidTok1();
	auto built = BuildArtifactMissingUnicodeTables(tok1.bytes);

	SslmArtifact artifact;
	SslmError aerr;
	auto status = SslmArtifact::OpenFromMemory(built.bytes.data(), built.bytes.size(), artifact, &aerr);
	CHECK_MSG(status == SslmStatus::Ok, "outer artifact (no UnicodeTables section) must still load: got %s",
	          SslmStatusName(status));
	if (status != SslmStatus::Ok) return;

	TokenizerView view;
	std::string terr;
	bool opened = TokenizerView::Open(artifact, view, &terr);
	CHECK_MSG(!opened, "TokenizerView::Open accepted an artifact with no UnicodeTables section");
	CHECK(!view.Ok());
}

namespace {

// Shared assertion for every TOK1/UNI1 hostile cell below: the mutation lives
// entirely inside the named sub-blob, so BuildTokenizerArtifact's fresh integrity
// stamp always makes the OUTER artifact load Ok — a cell where it does not is a
// bug in the cell, not a finding — and then TokenizerView::Open on that outer
// artifact must return false, with Ok() left false.
//
// `why` is the literal substring of ParseTok/ParseUni's fail() message that the
// TARGETED guard emits (verified against src/tokenizer.cpp), not merely a
// description of the mutation. Asserting `terr` contains it closes the shadowing
// class Poirot's review found (e448fb8..e79ca03 S-1): `Open` returning false
// alone does not say WHICH check fired, so a mutated fixture can be caught by an
// unrelated downstream guard (a shadowing check) while the guard the cell targets
// is silently dead — reverting a targeted guard now surfaces as `terr` carrying
// the shadowing guard's own (different) message, which fails this assertion.
void AssertTok1Rejected(const std::vector<uint8_t>& mutated_tok1, const char* why) {
	auto uni1 = MakeMinimalValidUni1();
	auto built = BuildTokenizerArtifact(mutated_tok1, uni1.bytes);

	SslmArtifact artifact;
	SslmError aerr;
	auto status = SslmArtifact::OpenFromMemory(built.bytes.data(), built.bytes.size(), artifact, &aerr);
	CHECK_MSG(status == SslmStatus::Ok, "%s: outer artifact failed to load (mutation should be TOK1-internal) — got %s",
	          why, SslmStatusName(status));
	if (status != SslmStatus::Ok) return;

	TokenizerView view;
	std::string terr;
	bool opened = TokenizerView::Open(artifact, view, &terr);
	CHECK_MSG(!opened, "%s: TokenizerView::Open ACCEPTED a malformed TOK1 blob (gap)", why);
	CHECK(!view.Ok());
	if (opened) return;
	CHECK_MSG(terr.find(why) != std::string::npos,
	          "%s: rejected, but for the WRONG reason (a shadowing check fired instead) — got \"%s\"",
	          why, terr.c_str());
}

void AssertUni1Rejected(const std::vector<uint8_t>& mutated_uni1, const char* why) {
	auto tok1 = MakeMinimalValidTok1();
	auto built = BuildTokenizerArtifact(tok1.bytes, mutated_uni1);

	SslmArtifact artifact;
	SslmError aerr;
	auto status = SslmArtifact::OpenFromMemory(built.bytes.data(), built.bytes.size(), artifact, &aerr);
	CHECK_MSG(status == SslmStatus::Ok, "%s: outer artifact failed to load (mutation should be UNI1-internal) — got %s",
	          why, SslmStatusName(status));
	if (status != SslmStatus::Ok) return;

	TokenizerView view;
	std::string terr;
	bool opened = TokenizerView::Open(artifact, view, &terr);
	CHECK_MSG(!opened, "%s: TokenizerView::Open ACCEPTED a malformed UNI1 blob (gap)", why);
	CHECK(!view.Ok());
	if (opened) return;
	CHECK_MSG(terr.find(why) != std::string::npos,
	          "%s: rejected, but for the WRONG reason (a shadowing check fired instead) — got \"%s\"",
	          why, terr.c_str());
}

}  // namespace

// --- TOK1 cells. Each isolates exactly one deviation from docs/sslm_format.md's
//     "Tokenizer blob — TOK1" layout in the minimal valid blob. ---

SSLM_TEST(TestTok1RejectsBadMagic, 61) {
	auto tok1 = MakeMinimalValidTok1();
	tok1.bytes[0] = 'X';  // was 'T' of "TOK1", offset 0
	AssertTok1Rejected(tok1.bytes, "Tokenizer: bad TOK1 header");
}

SSLM_TEST(TestTok1RejectsTruncatedHeader, 62) {
	auto tok1 = MakeMinimalValidTok1();
	tok1.bytes.resize(10);  // shorter than the 24-byte fixed TOK1 header
	AssertTok1Rejected(tok1.bytes, "Tokenizer: bad TOK1 header");
}

SSLM_TEST(TestTok1RejectsVocabCountOverflow, 63) {
	auto tok1 = MakeMinimalValidTok1();
	PutU32(tok1.bytes, tok1.layout.vocab_count_off, 0xFFFFFFFFu);
	AssertTok1Rejected(tok1.bytes, "Tokenizer: vocab_count exceeds INT32_MAX");
}

SSLM_TEST(TestTok1RejectsMergeCountOverflow, 64) {
	auto tok1 = MakeMinimalValidTok1();
	PutU32(tok1.bytes, tok1.layout.merge_count_off, 0xFFFFFFFFu);
	AssertTok1Rejected(tok1.bytes, "Tokenizer: truncated merges");
}

SSLM_TEST(TestTok1RejectsSpecialCountOverflow, 65) {
	auto tok1 = MakeMinimalValidTok1();
	PutU32(tok1.bytes, tok1.layout.special_count_off, 0xFFFFFFFFu);
	AssertTok1Rejected(tok1.bytes, "Tokenizer: truncated special ids");
}

SSLM_TEST(TestTok1RejectsTruncatedByteToId, 66) {
	auto tok1 = MakeMinimalValidTok1();
	tok1.bytes.resize(tok1.layout.byte_to_id_off + 100);  // 100 of the required 1024 bytes
	AssertTok1Rejected(tok1.bytes, "Tokenizer: truncated byte_to_id");
}

SSLM_TEST(TestTok1RejectsTruncatedVocabOffsets, 67) {
	auto tok1 = MakeMinimalValidTok1();
	tok1.bytes.resize(tok1.layout.vocab_offsets_off + 4);  // 1 of the required vocab_count+1=5 entries
	AssertTok1Rejected(tok1.bytes, "Tokenizer: truncated vocab offsets");
}

SSLM_TEST(TestTok1RejectsTruncatedVocabBlobLen, 68) {
	auto tok1 = MakeMinimalValidTok1();
	tok1.bytes.resize(tok1.layout.vocab_blob_len_off + 2);  // half of the 4-byte length field
	AssertTok1Rejected(tok1.bytes, "Tokenizer: truncated vocab blob_len");
}

SSLM_TEST(TestTok1RejectsTruncatedVocabBlob, 69) {
	auto tok1 = MakeMinimalValidTok1();
	tok1.bytes.resize(tok1.layout.vocab_blob_off + tok1.layout.vocab_blob_len - 1);  // one byte short
	AssertTok1Rejected(tok1.bytes, "Tokenizer: truncated vocab blob");
}

SSLM_TEST(TestTok1RejectsVocabOffsetNonMonotonic, 70) {
	auto tok1 = MakeMinimalValidTok1();
	// Baseline vocab_offsets [0,1,2,3,5,10] (vblob=10, S-HARDEN-2's 5-entry
	// fixture). Bump index 2 from 2 to 4 (still <= vblob): index 3's value (3) is
	// now smaller than its predecessor (4) — a pure non-monotonic violation, no
	// offset exceeds vblob.
	PutU32(tok1.bytes, tok1.layout.vocab_offsets_off + 2 * 4, 4);
	AssertTok1Rejected(tok1.bytes, "Tokenizer: vocab offset out of range");
}

SSLM_TEST(TestTok1RejectsLastVocabOffsetExceedsBlob, 71) {
	auto tok1 = MakeMinimalValidTok1();
	// Index `vocab_count` is the terminal offset (derived, not a hardcoded literal,
	// per S-HARDEN-2's fixture replacement note above); baseline value == vblob.
	PutU32(tok1.bytes, tok1.layout.vocab_offsets_off + tok1.layout.vocab_count * 4,
	       tok1.layout.vocab_blob_len + 50);
	AssertTok1Rejected(tok1.bytes, "Tokenizer: vocab offset out of range");
}

SSLM_TEST(TestTok1RejectsMiddleVocabOffsetExceedsBlob, 72) {
	auto tok1 = MakeMinimalValidTok1();
	// Index 2 is not the terminal offset (index `vocab_count` is); baseline value 2.
	PutU32(tok1.bytes, tok1.layout.vocab_offsets_off + 2 * 4, tok1.layout.vocab_blob_len + 50);
	AssertTok1Rejected(tok1.bytes, "Tokenizer: vocab offset out of range");
}

SSLM_TEST(TestTok1RejectsTruncatedMerges, 73) {
	auto tok1 = MakeMinimalValidTok1();
	tok1.bytes.resize(tok1.layout.merges_off + 6);  // half of the one 12-byte merge record
	AssertTok1Rejected(tok1.bytes, "Tokenizer: truncated merges");
}

SSLM_TEST(TestTok1RejectsTruncatedSpecialIds, 74) {
	auto tok1 = MakeMinimalValidTok1();
	tok1.bytes.resize(tok1.layout.special_ids_off + 2);  // half of the one 4-byte special id
	AssertTok1Rejected(tok1.bytes, "Tokenizer: truncated special ids");
}

SSLM_TEST(TestTok1RejectsTruncatedSpecialOffsets, 75) {
	auto tok1 = MakeMinimalValidTok1();
	tok1.bytes.resize(tok1.layout.special_offsets_off + 4);  // 1 of the required special_count+1=2 entries
	AssertTok1Rejected(tok1.bytes, "Tokenizer: truncated special offsets");
}

SSLM_TEST(TestTok1RejectsTruncatedSpecialBlobLen, 76) {
	auto tok1 = MakeMinimalValidTok1();
	tok1.bytes.resize(tok1.layout.special_blob_len_off + 2);  // half of the 4-byte length field
	AssertTok1Rejected(tok1.bytes, "Tokenizer: truncated special blob_len");
}

SSLM_TEST(TestTok1RejectsTruncatedSpecialBlob, 77) {
	auto tok1 = MakeMinimalValidTok1();
	tok1.bytes.resize(tok1.layout.special_blob_off + tok1.layout.special_blob_len - 1);  // one byte short
	AssertTok1Rejected(tok1.bytes, "Tokenizer: truncated special blob");
}

SSLM_TEST(TestTok1RejectsSpecialOffsetNonMonotonic, 78) {
	auto tok1 = MakeMinimalValidTok1();
	// Baseline special_offsets [0,5] (sblob=5). Set index 0 to 6 (> index 1's 5) —
	// non-monotonic; index 1 (5) does not itself exceed sblob (5), so the range
	// branch cannot also fire.
	PutU32(tok1.bytes, tok1.layout.special_offsets_off + 0 * 4, 6);
	AssertTok1Rejected(tok1.bytes, "Tokenizer: bad special offset");
}

SSLM_TEST(TestTok1RejectsSpecialOffsetOutOfRange, 79) {
	auto tok1 = MakeMinimalValidTok1();
	// Leave the (monotonic) offsets [0,5] untouched; lie about special_blob_len
	// instead (5 -> 3), so the terminal offset (5) now exceeds the declared length
	// — a pure range violation, isolated from the monotonic check.
	PutU32(tok1.bytes, tok1.layout.special_blob_len_off, tok1.layout.special_blob_len - 2);
	AssertTok1Rejected(tok1.bytes, "Tokenizer: bad special offset");
}

// --- S-HARDEN-2 (F6): TOK1's declared version and reserved fields (offset 4 and
//     offset 20 respectively, docs/sslm_format.md "Tokenizer blob — TOK1") were
//     never read by any code path before this slot — a guard-vitality gap (§17
//     dim 11): the format declares both and nothing enforced either. ---

SSLM_TEST(TestTok1RejectsUnsupportedVersion, 80) {
	auto tok1 = MakeMinimalValidTok1();
	PutU32(tok1.bytes, tok1.layout.version_off, 2);  // only 1 is the declared TOK1 version
	AssertTok1Rejected(tok1.bytes, "Tokenizer: unsupported TOK1 version");
}

SSLM_TEST(TestTok1RejectsNonzeroReserved, 81) {
	auto tok1 = MakeMinimalValidTok1();
	PutU32(tok1.bytes, tok1.layout.reserved_off, 1);
	AssertTok1Rejected(tok1.bytes, "Tokenizer: TOK1 reserved field != 0");
}

// --- S-HARDEN-2 (F18): vocab_count's own declared domain — zero encodes no byte
//     and indexes no token; every id this parser stores is narrowed to int32_t
//     downstream, so a vocab_count past INT32_MAX must be rejected before that
//     narrowing (or the bound checks below, which compare ids against vocab_count
//     as read) can be silently wrong. ---

SSLM_TEST(TestTok1RejectsVocabCountZero, 82) {
	auto tok1 = MakeMinimalValidTok1();
	PutU32(tok1.bytes, tok1.layout.vocab_count_off, 0);
	AssertTok1Rejected(tok1.bytes, "Tokenizer: vocab_count == 0");
}

SSLM_TEST(TestTok1RejectsVocabCountExceedsInt32Max, 83) {
	auto tok1 = MakeMinimalValidTok1();
	// The exact boundary — INT32_MAX + 1 — not merely a very large value (the
	// pre-existing TestTok1RejectsVocabCountOverflow uses 0xFFFFFFFF, which this
	// slot's new INT32_MAX check also rejects, but that cell was written before
	// this bound existed and does not isolate it).
	PutU32(tok1.bytes, tok1.layout.vocab_count_off, 0x80000000u);
	AssertTok1Rejected(tok1.bytes, "Tokenizer: vocab_count exceeds INT32_MAX");
}

// --- S-HARDEN-2 (F18): vocabulary-bound rejection. Every id this parser stores
//     for later emission (byte_to_id, merge operands/results, special ids) must
//     index a real vocabulary slot — the exact defect the finding named: a
//     committed fixture used to declare a special id (1000) with no matching
//     vocabulary entry (4 declared), and TokenizerView::Open accepted it. Each
//     cell below isolates one of the four id classes the parser stores. ---

SSLM_TEST(TestTok1RejectsByteToIdEntryAtOrAboveVocabCount, 84) {
	auto tok1 = MakeMinimalValidTok1();
	// byte_to_id['z'] defaults to 0 (a valid id, below vocab_count=5); mutate it to
	// vocab_count itself — one past the last real vocabulary slot.
	PutU32(tok1.bytes, tok1.layout.byte_to_id_off + static_cast<unsigned char>('z') * 4, tok1.layout.vocab_count);
	AssertTok1Rejected(tok1.bytes, "Tokenizer: byte_to_id entry >= vocab_count");
}

SSLM_TEST(TestTok1RejectsMergeOperandAAtOrAboveVocabCount, 85) {
	auto tok1 = MakeMinimalValidTok1();
	// The one merge record is (a=0, b=1, merged=3) at merges_off; field `a` is the
	// first u32.
	PutU32(tok1.bytes, tok1.layout.merges_off + 0, tok1.layout.vocab_count);
	AssertTok1Rejected(tok1.bytes, "Tokenizer: merge operand or result >= vocab_count");
}

SSLM_TEST(TestTok1RejectsMergeOperandBAtOrAboveVocabCount, 86) {
	auto tok1 = MakeMinimalValidTok1();
	PutU32(tok1.bytes, tok1.layout.merges_off + 4, tok1.layout.vocab_count);
	AssertTok1Rejected(tok1.bytes, "Tokenizer: merge operand or result >= vocab_count");
}

SSLM_TEST(TestTok1RejectsMergeResultAtOrAboveVocabCount, 87) {
	auto tok1 = MakeMinimalValidTok1();
	PutU32(tok1.bytes, tok1.layout.merges_off + 8, tok1.layout.vocab_count);
	AssertTok1Rejected(tok1.bytes, "Tokenizer: merge operand or result >= vocab_count");
}

SSLM_TEST(TestTok1RejectsSpecialIdAtOrAboveVocabCount, 88) {
	auto tok1 = MakeMinimalValidTok1();
	// This is F18's own fixture pattern reproduced as a hostile mutation cell now
	// that the baseline fixture itself is valid: the special's declared id (4)
	// pushed one past the vocabulary (5).
	PutU32(tok1.bytes, tok1.layout.special_ids_off, tok1.layout.vocab_count);
	AssertTok1Rejected(tok1.bytes, "Tokenizer: special id >= vocab_count");
}

// --- UNI1 cells. Each isolates exactly one deviation from docs/sslm_format.md's
//     "UnicodeTables blob — UNI1" layout in the minimal valid blob. letter/
//     number/space share one ReadRanges() implementation in src/tokenizer.cpp;
//     the count-field-truncation sub-case is exercised once (on letter, the first
//     call) rather than duplicated three times — the other two axes (range-data
//     truncation, count overflow) are exercised per table, since those are the
//     axes a per-call regression could plausibly differ on. ---

SSLM_TEST(TestUni1RejectsBadMagic, 89) {
	auto uni1 = MakeMinimalValidUni1();
	uni1.bytes[0] = 'X';  // was 'U' of "UNI1", offset 0
	AssertUni1Rejected(uni1.bytes, "UnicodeTables: bad UNI1 header");
}

SSLM_TEST(TestUni1RejectsTruncatedHeader, 90) {
	auto uni1 = MakeMinimalValidUni1();
	uni1.bytes.resize(5);  // shorter than the 8-byte magic+version header
	AssertUni1Rejected(uni1.bytes, "UnicodeTables: bad UNI1 header");
}

// S-HARDEN-2 (F6): offset 4's version field was skipped past (`pos = 8`) but
// never actually read and compared — the same guard-vitality gap as TOK1's
// version/reserved fields.
SSLM_TEST(TestUni1RejectsUnsupportedVersion, 91) {
	auto uni1 = MakeMinimalValidUni1();
	PutU32(uni1.bytes, uni1.layout.version_off, 2);  // only 1 is the declared UNI1 version
	AssertUni1Rejected(uni1.bytes, "UnicodeTables: unsupported UNI1 version");
}

SSLM_TEST(TestUni1RejectsLetterCountFieldTruncated, 92) {
	auto uni1 = MakeMinimalValidUni1();
	uni1.bytes.resize(uni1.layout.letter_count_off + 2);  // half of the 4-byte count field
	AssertUni1Rejected(uni1.bytes, "UnicodeTables: truncated range count");
}

SSLM_TEST(TestUni1RejectsLetterRangesTruncated, 93) {
	auto uni1 = MakeMinimalValidUni1();
	uni1.bytes.resize(uni1.layout.letter_data_off + 4);  // 4 of the required 16 bytes (2 ranges)
	AssertUni1Rejected(uni1.bytes, "UnicodeTables: truncated ranges");
}

SSLM_TEST(TestUni1RejectsLetterCountOverflow, 94) {
	auto uni1 = MakeMinimalValidUni1();
	PutU32(uni1.bytes, uni1.layout.letter_count_off, 0xFFFFFFFFu);
	AssertUni1Rejected(uni1.bytes, "UnicodeTables: truncated ranges");
}

SSLM_TEST(TestUni1RejectsNumberRangesTruncated, 95) {
	auto uni1 = MakeMinimalValidUni1();
	uni1.bytes.resize(uni1.layout.number_data_off + 4);  // 4 of the required 8 bytes (1 range)
	AssertUni1Rejected(uni1.bytes, "UnicodeTables: truncated ranges");
}

SSLM_TEST(TestUni1RejectsNumberCountOverflow, 96) {
	auto uni1 = MakeMinimalValidUni1();
	PutU32(uni1.bytes, uni1.layout.number_count_off, 0xFFFFFFFFu);
	AssertUni1Rejected(uni1.bytes, "UnicodeTables: truncated ranges");
}

SSLM_TEST(TestUni1RejectsSpaceRangesTruncated, 97) {
	auto uni1 = MakeMinimalValidUni1();
	uni1.bytes.resize(uni1.layout.space_data_off + 8);  // 8 of the required 16 bytes (2 ranges)
	AssertUni1Rejected(uni1.bytes, "UnicodeTables: truncated ranges");
}

SSLM_TEST(TestUni1RejectsSpaceCountOverflow, 98) {
	auto uni1 = MakeMinimalValidUni1();
	PutU32(uni1.bytes, uni1.layout.space_count_off, 0xFFFFFFFFu);
	AssertUni1Rejected(uni1.bytes, "UnicodeTables: truncated ranges");
}

SSLM_TEST(TestUni1RejectsCccTruncated, 99) {
	auto uni1 = MakeMinimalValidUni1();
	uni1.bytes.resize(uni1.layout.ccc_data_off + 4);  // 4 of the required 8 bytes (1 entry)
	AssertUni1Rejected(uni1.bytes, "UnicodeTables: truncated ccc");
}

SSLM_TEST(TestUni1RejectsCccCountOverflow, 100) {
	auto uni1 = MakeMinimalValidUni1();
	PutU32(uni1.bytes, uni1.layout.ccc_count_off, 0xFFFFFFFFu);
	AssertUni1Rejected(uni1.bytes, "UnicodeTables: truncated ccc");
}

SSLM_TEST(TestUni1RejectsDecompCpsTruncated, 101) {
	auto uni1 = MakeMinimalValidUni1();
	uni1.bytes.resize(uni1.layout.decomp_cps_off + 2);  // half of the required 4 bytes (1 cp)
	AssertUni1Rejected(uni1.bytes, "UnicodeTables: truncated decomp cps");
}

SSLM_TEST(TestUni1RejectsDecompCountOverflow, 102) {
	auto uni1 = MakeMinimalValidUni1();
	PutU32(uni1.bytes, uni1.layout.decomp_count_off, 0xFFFFFFFFu);
	AssertUni1Rejected(uni1.bytes, "UnicodeTables: truncated decomp cps");
}

SSLM_TEST(TestUni1RejectsDecompOffsetsTruncated, 103) {
	auto uni1 = MakeMinimalValidUni1();
	uni1.bytes.resize(uni1.layout.decomp_offsets_off + 4);  // 1 of the required decomp_count+1=2 entries
	AssertUni1Rejected(uni1.bytes, "UnicodeTables: truncated decomp offsets");
}

SSLM_TEST(TestUni1RejectsDecompOffsetOutOfRange, 104) {
	auto uni1 = MakeMinimalValidUni1();
	// Baseline decomp offsets [0,2] (seq_len=2). Bump the terminal offset (index 1)
	// past seq_len while keeping it >= its predecessor — a pure range violation.
	PutU32(uni1.bytes, uni1.layout.decomp_offsets_off + 1 * 4, uni1.layout.seq_len + 97);
	AssertUni1Rejected(uni1.bytes, "UnicodeTables: bad decomp offset");
}

SSLM_TEST(TestUni1RejectsDecompOffsetNonMonotonic, 105) {
	auto uni1 = MakeMinimalValidUni1();
	// Baseline decomp offsets [0,2]. Set index 0 to 3 (> index 1's 2) — non-
	// monotonic; index 1 (2) does not itself exceed seq_len (2).
	PutU32(uni1.bytes, uni1.layout.decomp_offsets_off + 0 * 4, 3);
	AssertUni1Rejected(uni1.bytes, "UnicodeTables: bad decomp offset");
}

SSLM_TEST(TestUni1RejectsDecompSeqTruncated, 106) {
	auto uni1 = MakeMinimalValidUni1();
	uni1.bytes.resize(uni1.layout.decomp_seq_off + 4);  // half of the required 8 bytes (seq_len=2)
	AssertUni1Rejected(uni1.bytes, "UnicodeTables: truncated decomp seq");
}

SSLM_TEST(TestUni1RejectsComposeTruncated, 107) {
	auto uni1 = MakeMinimalValidUni1();
	uni1.bytes.resize(uni1.layout.compose_data_off + 6);  // half of the required 12 bytes (1 entry)
	AssertUni1Rejected(uni1.bytes, "UnicodeTables: truncated compose");
}

SSLM_TEST(TestUni1RejectsComposeCountOverflow, 108) {
	auto uni1 = MakeMinimalValidUni1();
	PutU32(uni1.bytes, uni1.layout.compose_count_off, 0xFFFFFFFFu);
	AssertUni1Rejected(uni1.bytes, "UnicodeTables: truncated compose");
}

// T-1416 (whole-tree review b9dcbe0, Minor 4): letter/number/space are binary-
// searched by InRanges (tokenizer.cpp), which is unsound over an unsorted or
// overlapping table -- deterministic misclassification with no diagnostic,
// the exact trust-boundary gap this section's header comment names. The two
// cells below isolate the two ways ReadRanges' new monotonicity check can
// fail, mirroring TestUni1RejectsDecompOffsetOutOfRange/NonMonotonic's
// two-cell shape for the same class of defect on a sibling table.
SSLM_TEST(TestUni1RejectsLetterRangeWithLoGreaterThanHi, 109) {
	auto uni1 = MakeMinimalValidUni1();
	// Baseline letter range 0 is ('A'=65, 'Z'=90). Swap to (90, 65) -- a pure
	// lo > hi violation; range 1 ('a'-'z') is untouched and still valid on its
	// own, so this isolates the malformed range.
	PutU32(uni1.bytes, uni1.layout.letter_data_off + 0 * 8, 90);
	PutU32(uni1.bytes, uni1.layout.letter_data_off + 0 * 8 + 4, 65);
	AssertUni1Rejected(uni1.bytes, "UnicodeTables: range lo > hi");
}

SSLM_TEST(TestUni1RejectsLetterRangesOverlapping, 110) {
	auto uni1 = MakeMinimalValidUni1();
	// Baseline letter ranges are ('A'-'Z'=65-90), ('a'-'z'=97-122) -- sorted,
	// non-overlapping. Pull range 1's lo down to 70, inside range 0's [65,90],
	// so InRanges' own sortedness precondition breaks while each range is
	// individually well-formed (lo <= hi on both).
	PutU32(uni1.bytes, uni1.layout.letter_data_off + 1 * 8, 70);
	AssertUni1Rejected(uni1.bytes, "UnicodeTables: ranges not sorted or overlapping");
}


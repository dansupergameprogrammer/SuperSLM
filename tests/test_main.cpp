// SuperSLM test harness. Standard library only; no third-party test framework.
// Mirrors the SuperFAISS harness convention: CHECK/CHECK_MSG macros, global
// check/failure counters, plain Test*() functions called from main, a summary
// line, exit code 0 iff every check passed.
//
// S0 skeleton baseline: the SHA-256 known-answer and contract-lookup tests below
// prove the toolchain, CMake/build.bat, and the integrity primitive the loader
// depends on. The artifact LOADER suite is authored red-first by Curie
// (SuperSLM_Plan.md §15; §17 Coverage Model) and appended here.

#include "superslm/artifact.h"
#include "superslm/sha256.h"
#include "superslm/tokenizer.h"
#include "sslm_fixtures.h"
#include "sslm_tokenizer_fixtures.h"
#include "sslm_tokenizer_hostile_fixtures.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

using namespace superslm;
using namespace superslm_test;

static int GChecks = 0;
static int GFailures = 0;

#define CHECK(cond) \
	do { \
		++GChecks; \
		if (!(cond)) { \
			++GFailures; \
			std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
		} \
	} while (0)

#define CHECK_MSG(cond, ...) \
	do { \
		++GChecks; \
		if (!(cond)) { \
			++GFailures; \
			std::printf("FAIL %s:%d: %s — ", __FILE__, __LINE__, #cond); \
			std::printf(__VA_ARGS__); \
			std::printf("\n"); \
		} \
	} while (0)

// --- S0 skeleton baseline -------------------------------------------------

static void TestSha256KnownVectors() {
	// FIPS 180-4 / NIST known-answer vectors.
	uint8_t d[32];
	Sha256Hash(reinterpret_cast<const uint8_t*>(""), 0, d);
	CHECK(ToHex(d) ==
	      "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");

	const char* abc = "abc";
	Sha256Hash(reinterpret_cast<const uint8_t*>(abc), 3, d);
	CHECK(ToHex(d) ==
	      "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");

	// A message spanning a block boundary (56 bytes → two-block padding path).
	std::string long_msg(
	    "abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq");
	Sha256Hash(reinterpret_cast<const uint8_t*>(long_msg.data()), long_msg.size(), d);
	CHECK(ToHex(d) ==
	      "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1");
}

static void TestDtypeSizes() {
	CHECK(DtypeSize(static_cast<uint32_t>(SslmDtype::Raw)) == 1);
	CHECK(DtypeSize(static_cast<uint32_t>(SslmDtype::Int8)) == 1);
	CHECK(DtypeSize(static_cast<uint32_t>(SslmDtype::Int32)) == 4);
	CHECK(DtypeSize(static_cast<uint32_t>(SslmDtype::Int64)) == 8);
	CHECK(DtypeSize(static_cast<uint32_t>(SslmDtype::Float32)) == 4);
	CHECK(DtypeSize(static_cast<uint32_t>(SslmDtype::Float64)) == 8);
	CHECK(DtypeSize(9999u) == 0); // unknown dtype
}

static void TestKnownSectionTypes() {
	CHECK(IsKnownSectionType(static_cast<uint32_t>(SslmSectionType::Config)));
	CHECK(IsKnownSectionType(static_cast<uint32_t>(SslmSectionType::Weights)));
	CHECK(IsKnownSectionType(static_cast<uint32_t>(SslmSectionType::GoldenHashes)));
	CHECK(!IsKnownSectionType(12u));   // gap in the enum
	CHECK(!IsKnownSectionType(9999u)); // outside the set
}

// ---------------------------------------------------------------------------
// Curie's S0 loader red suite (SuperSLM_Plan.md §15; §17 Coverage Model dim 2 —
// the artifact loader is the named trust boundary). Every cell below builds a
// `.sslm` byte buffer per docs/sslm_format.md via sslm_fixtures.h, independent of
// SslmArtifact::OpenFromMemory, and calls the loader under test. At S0 the loader
// is the unbuilt stub in src/artifact.cpp (always returns IoError, no field
// examined) — every cell below fails red for that one reason: the loader has not
// validated. Once Brunel builds the loader, each cell fails red for its own
// documented reason until the corresponding check is implemented, then goes
// green.
// ---------------------------------------------------------------------------

// --- Rejection cells: one per SslmStatus the header/table/section checks can
//     produce, each on a buffer that is structurally valid except the one named
//     defect. ---

static void TestRejectsBadMagic() {
	auto built = BuildArtifact({MakeConfigSection()});
	built.bytes[0] = 'X';  // was 'S' — docs/sslm_format.md header offset 0
	RecomputeIntegrityHash(built.bytes);

	SslmArtifact out;
	SslmError err;
	auto status = SslmArtifact::OpenFromMemory(built.bytes.data(), built.bytes.size(), out, &err);
	CHECK_MSG(status == SslmStatus::BadMagic, "got %s", SslmStatusName(status));
	CHECK(err.code == SslmStatus::BadMagic);
	CHECK(err.section_index == kNoSection);
	CHECK(!out.Ok());
}

static void TestRejectsUnsupportedVersion() {
	auto built = BuildArtifact({MakeConfigSection()});
	PutU32(built.bytes, 4, kArtifactFormatVersion + 1);  // format_version, offset 4
	RecomputeIntegrityHash(built.bytes);

	SslmArtifact out;
	SslmError err;
	auto status = SslmArtifact::OpenFromMemory(built.bytes.data(), built.bytes.size(), out, &err);
	CHECK_MSG(status == SslmStatus::UnsupportedVersion, "got %s", SslmStatusName(status));
	CHECK(err.code == SslmStatus::UnsupportedVersion);
	CHECK(err.section_index == kNoSection);
}

static void TestRejectsHeaderBytesMismatch() {
	auto built = BuildArtifact({MakeConfigSection()});
	PutU32(built.bytes, 8, kHeaderBytes - 1);  // header_bytes, offset 8, must == 64
	RecomputeIntegrityHash(built.bytes);

	SslmArtifact out;
	SslmError err;
	auto status = SslmArtifact::OpenFromMemory(built.bytes.data(), built.bytes.size(), out, &err);
	CHECK_MSG(status == SslmStatus::BadHeader, "got %s", SslmStatusName(status));
	CHECK(err.code == SslmStatus::BadHeader);
	CHECK(err.section_index == kNoSection);
}

static void TestRejectsNonzeroFlags() {
	auto built = BuildArtifact({MakeConfigSection()});
	PutU32(built.bytes, 16, 1);  // flags, offset 16, reserved == 0
	RecomputeIntegrityHash(built.bytes);

	SslmArtifact out;
	SslmError err;
	auto status = SslmArtifact::OpenFromMemory(built.bytes.data(), built.bytes.size(), out, &err);
	CHECK_MSG(status == SslmStatus::BadHeader, "got %s", SslmStatusName(status));
	CHECK(err.code == SslmStatus::BadHeader);
	CHECK(err.section_index == kNoSection);
}

static void TestRejectsNonzeroReserved0() {
	auto built = BuildArtifact({MakeConfigSection()});
	PutU32(built.bytes, 20, 1);  // reserved0, offset 20, == 0
	RecomputeIntegrityHash(built.bytes);

	SslmArtifact out;
	SslmError err;
	auto status = SslmArtifact::OpenFromMemory(built.bytes.data(), built.bytes.size(), out, &err);
	CHECK_MSG(status == SslmStatus::BadHeader, "got %s", SslmStatusName(status));
	CHECK(err.code == SslmStatus::BadHeader);
	CHECK(err.section_index == kNoSection);
}

static void TestRejectsTruncatedHeader() {
	auto built = BuildArtifact({MakeConfigSection()});
	built.bytes.resize(32);  // shorter than the 64-byte header itself

	SslmArtifact out;
	SslmError err;
	auto status = SslmArtifact::OpenFromMemory(built.bytes.data(), built.bytes.size(), out, &err);
	CHECK_MSG(status == SslmStatus::Truncated, "got %s", SslmStatusName(status));
	CHECK(err.code == SslmStatus::Truncated);
	CHECK(err.section_index == kNoSection);
}

static void TestRejectsTruncatedSectionTable() {
	// Two sections declared (a 144-byte table+header region) but the buffer is cut
	// to 100 bytes — the header is intact but the declared table does not fit.
	auto built = BuildArtifact({MakeConfigSection(),
	                             MakeSection(SslmSectionType::Provenance, SslmDtype::Raw, {1, 2, 3, 4})});
	CHECK(built.bytes.size() > 100);  // sanity: the valid baseline is longer than the cut
	built.bytes.resize(100);
	PutU64(built.bytes, 24, 100);  // file_bytes now matches the actual (truncated) length,
	                                // so FileSizeMismatch cannot also fire — isolates Truncated
	                                // to "shorter than the declared section table."
	RecomputeIntegrityHash(built.bytes);

	SslmArtifact out;
	SslmError err;
	auto status = SslmArtifact::OpenFromMemory(built.bytes.data(), built.bytes.size(), out, &err);
	CHECK_MSG(status == SslmStatus::Truncated, "got %s", SslmStatusName(status));
	CHECK(err.code == SslmStatus::Truncated);
	CHECK(err.section_index == kNoSection);
}

static void TestRejectsTooManySections() {
	// Structural count check: header.section_count > kMaxSections must reject
	// before any per-row validation runs. This cell's 4097 table rows are left
	// zeroed garbage (type 0 duplicated 4097x, alignment 0) because v1's known
	// section-type set has too few members (15) to build 4097 structurally-valid,
	// non-duplicate rows — isolating this cell from every other defect is not
	// achievable by construction. The cell therefore assumes the loader bounds
	// section_count against kMaxSections immediately after the header, before
	// descending into row-by-row validation (the natural reading of the header
	// table's own "section_count <= 4096" constraint, listed as a header field,
	// not a per-row one) — documented here as a routed finding, not asserted
	// elsewhere in the suite.
	uint32_t section_count = kMaxSections + 1;  // 4097
	uint64_t table_end = kHeaderBytes + static_cast<uint64_t>(section_count) * kSectionDescBytes;
	std::vector<uint8_t> bytes(static_cast<size_t>(table_end), 0);
	std::memcpy(bytes.data(), kMagic, 4);
	PutU32(bytes, 4, kArtifactFormatVersion);
	PutU32(bytes, 8, kHeaderBytes);
	PutU32(bytes, 12, section_count);
	PutU32(bytes, 16, 0);
	PutU32(bytes, 20, 0);
	PutU64(bytes, 24, table_end);
	RecomputeIntegrityHash(bytes);

	SslmArtifact out;
	SslmError err;
	auto status = SslmArtifact::OpenFromMemory(bytes.data(), bytes.size(), out, &err);
	CHECK_MSG(status == SslmStatus::TooManySections, "got %s", SslmStatusName(status));
	CHECK(err.code == SslmStatus::TooManySections);
	CHECK(err.section_index == kNoSection);
}

static void TestRejectsFileSizeMismatch() {
	auto built = BuildArtifact({MakeConfigSection()});
	// file_bytes (header) keeps declaring the ORIGINAL length; the buffer grows.
	built.bytes.insert(built.bytes.end(), 8, 0xAB);
	RecomputeIntegrityHash(built.bytes);  // hash matches the actual (now longer) content —
	                                       // isolates FileSizeMismatch from IntegrityMismatch.

	SslmArtifact out;
	SslmError err;
	auto status = SslmArtifact::OpenFromMemory(built.bytes.data(), built.bytes.size(), out, &err);
	CHECK_MSG(status == SslmStatus::FileSizeMismatch, "got %s", SslmStatusName(status));
	CHECK(err.code == SslmStatus::FileSizeMismatch);
	CHECK(err.section_index == kNoSection);
}

static void TestRejectsAlignmentNotPowerOfTwo() {
	auto built = BuildArtifact({MakeConfigSection(),
	                             MakeSection(SslmSectionType::Provenance, SslmDtype::Raw, {1, 2, 3, 4},
	                                         /*alignment=*/24)});
	SslmArtifact out;
	SslmError err;
	auto status = SslmArtifact::OpenFromMemory(built.bytes.data(), built.bytes.size(), out, &err);
	CHECK_MSG(status == SslmStatus::BadAlignment, "got %s", SslmStatusName(status));
	CHECK(err.code == SslmStatus::BadAlignment);
	CHECK(err.section_index == 1);
}

static void TestRejectsAlignmentBelowMinimum() {
	auto built = BuildArtifact({MakeConfigSection(),
	                             MakeSection(SslmSectionType::Provenance, SslmDtype::Raw, {1, 2, 3, 4},
	                                         /*alignment=*/4)});  // power of two, but < 8
	SslmArtifact out;
	SslmError err;
	auto status = SslmArtifact::OpenFromMemory(built.bytes.data(), built.bytes.size(), out, &err);
	CHECK_MSG(status == SslmStatus::BadAlignment, "got %s", SslmStatusName(status));
	CHECK(err.code == SslmStatus::BadAlignment);
	CHECK(err.section_index == 1);
}

static void TestRejectsAlignmentAboveMaximum() {
	auto built = BuildArtifact({MakeConfigSection(),
	                             MakeSection(SslmSectionType::Provenance, SslmDtype::Raw, {1, 2, 3, 4},
	                                         /*alignment=*/8192)});  // power of two, but > 4096
	SslmArtifact out;
	SslmError err;
	auto status = SslmArtifact::OpenFromMemory(built.bytes.data(), built.bytes.size(), out, &err);
	CHECK_MSG(status == SslmStatus::BadAlignment, "got %s", SslmStatusName(status));
	CHECK(err.code == SslmStatus::BadAlignment);
	CHECK(err.section_index == 1);
}

static void TestRejectsMisalignedOffset() {
	FixtureSection provenance =
	    MakeSection(SslmSectionType::Provenance, SslmDtype::Raw, {1, 2, 3, 4}, /*alignment=*/64);
	provenance.offset_override = 200;  // valid alignment (64), but 200 % 64 != 0
	auto built = BuildArtifact({MakeConfigSection(), provenance});

	SslmArtifact out;
	SslmError err;
	auto status = SslmArtifact::OpenFromMemory(built.bytes.data(), built.bytes.size(), out, &err);
	CHECK_MSG(status == SslmStatus::Misaligned, "got %s", SslmStatusName(status));
	CHECK(err.code == SslmStatus::Misaligned);
	CHECK(err.section_index == 1);
}

static void TestRejectsSectionOutOfBoundsPastEof() {
	auto built = BuildArtifact({MakeConfigSection(),
	                             MakeSection(SslmSectionType::Provenance, SslmDtype::Raw, {1, 2, 3, 4})});
	const size_t row = kHeaderBytes + 1 * kSectionDescBytes;  // section index 1
	const uint64_t lying_size = built.bytes.size() + 1000000;
	PutU64(built.bytes, row + 16, lying_size);  // byte_size — declared range now exceeds the file
	PutU64(built.bytes, row + 24, lying_size);  // elem_count kept consistent (dtype Raw, size 1)
	                                             // so SizeMismatch cannot also fire.
	RecomputeIntegrityHash(built.bytes);         // hash covers the unchanged physical buffer.

	SslmArtifact out;
	SslmError err;
	auto status = SslmArtifact::OpenFromMemory(built.bytes.data(), built.bytes.size(), out, &err);
	CHECK_MSG(status == SslmStatus::SectionOutOfBounds, "got %s", SslmStatusName(status));
	CHECK(err.code == SslmStatus::SectionOutOfBounds);
	CHECK(err.section_index == 1);
}

static void TestRejectsSectionOffsetOverflow() {
	auto built = BuildArtifact({MakeConfigSection(),
	                             MakeSection(SslmSectionType::Provenance, SslmDtype::Raw, {1, 2, 3, 4},
	                                         /*alignment=*/16)});
	const size_t row = kHeaderBytes + 1 * kSectionDescBytes;
	const uint64_t overflowing_offset = 0xFFFFFFFFFFFFFFF0ULL;  // multiple of 16; offset+size wraps
	PutU64(built.bytes, row + 8, overflowing_offset);           // offset
	PutU64(built.bytes, row + 16, 32);                          // byte_size — sum overflows uint64
	PutU64(built.bytes, row + 24, 32);                          // elem_count (dtype Raw, size 1)
	RecomputeIntegrityHash(built.bytes);

	SslmArtifact out;
	SslmError err;
	auto status = SslmArtifact::OpenFromMemory(built.bytes.data(), built.bytes.size(), out, &err);
	CHECK_MSG(status == SslmStatus::SectionOutOfBounds, "got %s", SslmStatusName(status));
	CHECK(err.code == SslmStatus::SectionOutOfBounds);
	CHECK(err.section_index == 1);
}

static void TestRejectsSectionOverlapWithHeader() {
	// Empty (byte_size 0) so BuildArtifact writes no bytes at offset 0 — a non-empty
	// section here would memcpy over the magic and the loader would (correctly)
	// reject BadMagic first, not reach the overlap check at all (F-3, coordinator
	// 2026-07-19). The loader's overlap-with-header/table check is unconditional on
	// the declared offset alone (fires even for a zero-byte section), so this cell
	// isolates it without corrupting anything else.
	FixtureSection provenance =
	    MakeSection(SslmSectionType::Provenance, SslmDtype::Raw, {}, /*alignment=*/8);
	provenance.offset_override = 0;  // inside the header; 0 % 8 == 0, alignment 8 is valid
	auto built = BuildArtifact({MakeConfigSection(), provenance});

	SslmArtifact out;
	SslmError err;
	auto status = SslmArtifact::OpenFromMemory(built.bytes.data(), built.bytes.size(), out, &err);
	CHECK_MSG(status == SslmStatus::SectionOverlap, "got %s", SslmStatusName(status));
	CHECK(err.code == SslmStatus::SectionOverlap);
	CHECK(err.section_index == 1);
}

static void TestRejectsSectionOverlapWithSection() {
	FixtureSection config = MakeConfigSection();
	config.offset_override = 192;  // 192 % 64 == 0

	FixtureSection provenance =
	    MakeSection(SslmSectionType::Provenance, SslmDtype::Raw, {1, 2, 3, 4}, /*alignment=*/64);
	provenance.offset_override = 256;  // 256 % 64 == 0

	FixtureSection weights =
	    MakeSection(SslmSectionType::Weights, SslmDtype::Int8, EncodeInt8({1, 2, 3, 4, 5, 6, 7, 8}),
	                /*alignment=*/64);
	weights.offset_override = 256;  // deliberately the same range as `provenance`

	auto built = BuildArtifact({config, provenance, weights});

	SslmArtifact out;
	SslmError err;
	auto status = SslmArtifact::OpenFromMemory(built.bytes.data(), built.bytes.size(), out, &err);
	CHECK_MSG(status == SslmStatus::SectionOverlap, "got %s", SslmStatusName(status));
	CHECK(err.code == SslmStatus::SectionOverlap);
	// Either overlapping row is a defensible diagnostic; the spec does not pin
	// which of two mutually-overlapping sections is named.
	CHECK(err.section_index == 1 || err.section_index == 2);
}

static void TestRejectsBadDtype() {
	FixtureSection bad;
	bad.type = static_cast<uint32_t>(SslmSectionType::Provenance);
	bad.dtype = 9999;  // not a known SslmDtype
	bad.alignment = 64;
	// byte_size/elem_count auto-derive to 0 (DtypeSize(9999) == 0), so SizeMismatch
	// cannot also fire (0 == elem_count * 0 holds for any elem_count).
	auto built = BuildArtifact({MakeConfigSection(), bad});

	SslmArtifact out;
	SslmError err;
	auto status = SslmArtifact::OpenFromMemory(built.bytes.data(), built.bytes.size(), out, &err);
	CHECK_MSG(status == SslmStatus::BadDtype, "got %s", SslmStatusName(status));
	CHECK(err.code == SslmStatus::BadDtype);
	CHECK(err.section_index == 1);
}

static void TestRejectsSectionDtypeMismatch() {
	// docs/sslm_format.md load-bearing choice 6: Weights requires Int8. Declare it
	// Int32 instead — a known dtype, just not the one Weights requires. byte_size /
	// elem_count stay consistent for the DECLARED (wrong) dtype (4 elements @ size
	// 4 = 16 bytes) so SizeMismatch cannot also fire and mask this cell; BadDtype
	// cannot fire either, since Int32 is itself a known SslmDtype value. The only
	// defect is the type/dtype pairing itself.
	FixtureSection weights =
	    MakeSection(SslmSectionType::Weights, SslmDtype::Int32, EncodeInt32LE({1, 2, 3, 4}));
	auto built = BuildArtifact({MakeConfigSection(), weights});

	SslmArtifact out;
	SslmError err;
	auto status = SslmArtifact::OpenFromMemory(built.bytes.data(), built.bytes.size(), out, &err);
	CHECK_MSG(status == SslmStatus::SectionDtypeMismatch, "got %s", SslmStatusName(status));
	CHECK(err.code == SslmStatus::SectionDtypeMismatch);
	CHECK(err.section_index == 1);
}

static void TestRejectsSizeMismatch() {
	FixtureSection biases =
	    MakeSection(SslmSectionType::Biases, SslmDtype::Int32, EncodeInt32LE({1, 2, 3, 4}));  // 16 bytes, 4 elems
	biases.elem_count_override = 3;  // 16 != 3 * 4
	auto built = BuildArtifact({MakeConfigSection(), biases});

	SslmArtifact out;
	SslmError err;
	auto status = SslmArtifact::OpenFromMemory(built.bytes.data(), built.bytes.size(), out, &err);
	CHECK_MSG(status == SslmStatus::SizeMismatch, "got %s", SslmStatusName(status));
	CHECK(err.code == SslmStatus::SizeMismatch);
	CHECK(err.section_index == 1);
}

static void TestRejectsUnknownSectionType() {
	FixtureSection bad;
	bad.type = 999;  // outside the v1 set (docs/sslm_format.md "Section types")
	bad.dtype = static_cast<uint32_t>(SslmDtype::Raw);
	bad.data = {1, 2, 3, 4};
	bad.alignment = 64;
	auto built = BuildArtifact({MakeConfigSection(), bad});

	SslmArtifact out;
	SslmError err;
	auto status = SslmArtifact::OpenFromMemory(built.bytes.data(), built.bytes.size(), out, &err);
	CHECK_MSG(status == SslmStatus::UnknownSection, "got %s", SslmStatusName(status));
	CHECK(err.code == SslmStatus::UnknownSection);
	CHECK(err.section_index == 1);
}

static void TestRejectsDuplicateSection() {
	auto built = BuildArtifact({MakeConfigSection(),
	                             MakeSection(SslmSectionType::Config, SslmDtype::Raw, {'{', ' ', '}'})});
	SslmArtifact out;
	SslmError err;
	auto status = SslmArtifact::OpenFromMemory(built.bytes.data(), built.bytes.size(), out, &err);
	CHECK_MSG(status == SslmStatus::DuplicateSection, "got %s", SslmStatusName(status));
	CHECK(err.code == SslmStatus::DuplicateSection);
	// The spec does not pin which of the two same-typed rows is named.
	CHECK(err.section_index == 0 || err.section_index == 1);
}

static void TestRejectsMissingConfigSection() {
	auto built = BuildArtifact(
	    {MakeSection(SslmSectionType::Weights, SslmDtype::Int8, EncodeInt8({1, 2, 3, 4}))});  // no Config
	SslmArtifact out;
	SslmError err;
	auto status = SslmArtifact::OpenFromMemory(built.bytes.data(), built.bytes.size(), out, &err);
	CHECK_MSG(status == SslmStatus::MissingSection, "got %s", SslmStatusName(status));
	CHECK(err.code == SslmStatus::MissingSection);
	CHECK(err.section_index == kNoSection);
}

static void TestRejectsIntegrityMismatch() {
	auto built = BuildArtifact(
	    {MakeConfigSection(), MakeSection(SslmSectionType::Weights, SslmDtype::Int8, EncodeInt8({1, 2, 3, 4}))});
	const uint64_t weights_offset = built.placed[1].offset;
	built.bytes[static_cast<size_t>(weights_offset)] ^= 0xFF;  // flip a data byte; hash left stale

	SslmArtifact out;
	SslmError err;
	auto status = SslmArtifact::OpenFromMemory(built.bytes.data(), built.bytes.size(), out, &err);
	CHECK_MSG(status == SslmStatus::IntegrityMismatch, "got %s", SslmStatusName(status));
	CHECK(err.code == SslmStatus::IntegrityMismatch);
	CHECK(err.section_index == kNoSection);
}

// --- Boundary richness: shapes the rejection roster does not exercise. ---

static void TestAcceptsEmptySection() {
	FixtureSection empty = MakeSection(SslmSectionType::Provenance, SslmDtype::Raw, {});  // byte_size 0
	auto built = BuildArtifact({MakeConfigSection(), empty});

	SslmArtifact out;
	SslmError err;
	auto status = SslmArtifact::OpenFromMemory(built.bytes.data(), built.bytes.size(), out, &err);
	CHECK_MSG(status == SslmStatus::Ok, "got %s", SslmStatusName(status));
	CHECK(out.Ok());
	const SslmSectionView* view = out.Section(SslmSectionType::Provenance);
	CHECK(view != nullptr);
	if (view) {
		CHECK(view->byte_size == 0);
		CHECK(view->elem_count == 0);
	}
}

static void TestAcceptsMaximumAlignment() {
	auto built = BuildArtifact({MakeConfigSection(),
	                             MakeSection(SslmSectionType::Provenance, SslmDtype::Raw, {1, 2, 3, 4},
	                                         /*alignment=*/4096)});
	SslmArtifact out;
	SslmError err;
	auto status = SslmArtifact::OpenFromMemory(built.bytes.data(), built.bytes.size(), out, &err);
	CHECK_MSG(status == SslmStatus::Ok, "got %s", SslmStatusName(status));
	CHECK(out.Ok());
	const SslmSectionView* view = out.Section(SslmSectionType::Provenance);
	CHECK(view != nullptr);
	if (view) CHECK(view->alignment == 4096);
}

static void TestAcceptsSectionsInNonAscendingOffsetOrder() {
	// Table row 0 (Config) is placed at the HIGHER byte offset; table row 1
	// (Provenance) is placed at the LOWER one — docs/sslm_format.md: "Their order
	// in the table is not constrained; their byte ranges are."
	FixtureSection config = MakeConfigSection();
	config.data.resize(32, '{');
	config.alignment = 16;
	config.offset_override = 160;  // 160 % 16 == 0

	FixtureSection provenance =
	    MakeSection(SslmSectionType::Provenance, SslmDtype::Raw, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
	                /*alignment=*/16);
	provenance.offset_override = 144;  // 144 % 16 == 0; ends exactly at 160, no overlap

	auto built = BuildArtifact({config, provenance});

	SslmArtifact out;
	SslmError err;
	auto status = SslmArtifact::OpenFromMemory(built.bytes.data(), built.bytes.size(), out, &err);
	CHECK_MSG(status == SslmStatus::Ok, "got %s", SslmStatusName(status));
	CHECK(out.Ok());
	const SslmSectionView* cfg_view = out.Section(SslmSectionType::Config);
	const SslmSectionView* prov_view = out.Section(SslmSectionType::Provenance);
	CHECK(cfg_view != nullptr);
	CHECK(prov_view != nullptr);
	if (cfg_view) {
		CHECK(cfg_view->byte_size == config.data.size());
		CHECK(cfg_view->data != nullptr);
		if (cfg_view->data) CHECK(std::memcmp(cfg_view->data, config.data.data(), config.data.size()) == 0);
	}
	if (prov_view) {
		CHECK(prov_view->byte_size == provenance.data.size());
		CHECK(prov_view->data != nullptr);
		if (prov_view->data) {
			CHECK(std::memcmp(prov_view->data, provenance.data.data(), provenance.data.size()) == 0);
		}
	}
}

static void TestAcceptsReservedSectionTypeStructurally() {
	auto built = BuildArtifact(
	    {MakeConfigSection(), MakeSection(SslmSectionType::Tokenizer, SslmDtype::Raw, {9, 9, 9})});
	SslmArtifact out;
	SslmError err;
	auto status = SslmArtifact::OpenFromMemory(built.bytes.data(), built.bytes.size(), out, &err);
	CHECK_MSG(status == SslmStatus::Ok, "got %s", SslmStatusName(status));
	CHECK(out.Ok());
	const SslmSectionView* view = out.Section(SslmSectionType::Tokenizer);
	CHECK(view != nullptr);
	if (view) {
		CHECK(view->byte_size == 3);
		CHECK(view->data != nullptr);
		if (view->data) {
			const uint8_t expected[3] = {9, 9, 9};
			CHECK(std::memcmp(view->data, expected, 3) == 0);
		}
	}
}

static void TestOpenFromFileLoadsValidArtifact() {
	auto built = BuildArtifact(
	    {MakeConfigSection(), MakeSection(SslmSectionType::Weights, SslmDtype::Int8, EncodeInt8({1, 2, 3, 4}))});

	std::filesystem::path path =
	    std::filesystem::temp_directory_path() / "superslm_test_valid_artifact_37c1.sslm";
	{
		std::ofstream f(path, std::ios::binary | std::ios::trunc);
		f.write(reinterpret_cast<const char*>(built.bytes.data()), static_cast<std::streamsize>(built.bytes.size()));
	}

	SslmArtifact out;
	SslmError err;
	auto status = SslmArtifact::OpenFromFile(path.string().c_str(), out, &err);
	std::error_code ec;
	std::filesystem::remove(path, ec);  // best-effort cleanup

	CHECK_MSG(status == SslmStatus::Ok, "got %s", SslmStatusName(status));
	CHECK(out.Ok());
	CHECK(out.FormatVersion() == kArtifactFormatVersion);
	CHECK(out.FileBytes() == built.bytes.size());
	const SslmSectionView* view = out.Section(SslmSectionType::Weights);
	CHECK(view != nullptr);
	if (view) CHECK(view->byte_size == 4);
}

static void TestOpenFromFileMissingFileReturnsIoError() {
	std::filesystem::path path =
	    std::filesystem::temp_directory_path() / "superslm_test_nonexistent_9f3c2a.sslm";
	std::error_code ec;
	std::filesystem::remove(path, ec);  // ensure it genuinely does not exist

	SslmArtifact out;
	SslmError err;
	auto status = SslmArtifact::OpenFromFile(path.string().c_str(), out, &err);
	CHECK_MSG(status == SslmStatus::IoError, "got %s", SslmStatusName(status));
	CHECK(err.code == SslmStatus::IoError);
	CHECK(!out.Ok());
	// The IoError status alone coincides with the S0 stub's unconditional return
	// (it never reads a byte), so it is not yet red. artifact.h pins SslmError's
	// message as carrying "the offending values" — for a missing file, that is the
	// path — so this cell also requires the diagnostic to name it, which the stub's
	// fixed placeholder text does not. Forces this cell red until OpenFromFile
	// genuinely attempts the read and reports what it tried.
	CHECK_MSG(err.message.find(path.string()) != std::string::npos,
	          "message does not carry the missing path: \"%s\"", err.message.c_str());
}

// --- The feature oracle (dim 10): a valid multi-section artifact loads to the
//     exact end-state the bytes were built to carry — not merely "the loader
//     agrees with itself." ---

static void TestValidArtifactLoadsToExpectedEndState() {
	FixtureSection config = MakeConfigSection();
	FixtureSection weights =
	    MakeSection(SslmSectionType::Weights, SslmDtype::Int8,
	                EncodeInt8({-5, 0, 7, 127, -128, 3, 3, 3}), /*alignment=*/64);
	FixtureSection biases =
	    MakeSection(SslmSectionType::Biases, SslmDtype::Int32,
	                EncodeInt32LE({1000, -2000, 0, 123456}), /*alignment=*/64);
	FixtureSection rope =
	    MakeSection(SslmSectionType::RopeTables, SslmDtype::Int64,
	                EncodeInt64LE({1, -1, 123456789012LL, 0}), /*alignment=*/64);

	auto built = BuildArtifact({config, weights, biases, rope});

	SslmArtifact out;
	SslmError err;
	auto status = SslmArtifact::OpenFromMemory(built.bytes.data(), built.bytes.size(), out, &err);
	CHECK_MSG(status == SslmStatus::Ok, "got %s", SslmStatusName(status));
	CHECK(out.Ok());
	CHECK(out.FormatVersion() == kArtifactFormatVersion);
	CHECK(out.FileBytes() == built.bytes.size());
	CHECK(out.Sections().size() == 4);

	// Independent fingerprint oracle: SHA-256 over the exact bytes handed to
	// OpenFromMemory, hash field zeroed — the spec's definition, computed fresh
	// here rather than by re-invoking the fixture builder's own hashing helper a
	// second time on the loaded artifact.
	{
		std::vector<uint8_t> tmp = built.bytes;
		std::memset(tmp.data() + kIntegrityHashOffset, 0, kIntegrityHashBytes);
		uint8_t digest[32];
		Sha256Hash(tmp.data(), tmp.size(), digest);
		CHECK(out.FingerprintHex() == ToHex(digest));
	}

	struct Expected {
		SslmSectionType type;
		SslmDtype dtype;
		const std::vector<uint8_t>* data;
		uint64_t elem_count;
		uint32_t alignment;
	};
	const Expected expected[] = {
	    {SslmSectionType::Config, SslmDtype::Raw, &config.data, config.data.size(), 64},
	    {SslmSectionType::Weights, SslmDtype::Int8, &weights.data, 8, 64},
	    {SslmSectionType::Biases, SslmDtype::Int32, &biases.data, 4, 64},
	    {SslmSectionType::RopeTables, SslmDtype::Int64, &rope.data, 4, 64},
	};
	for (const auto& e : expected) {
		const SslmSectionView* view = out.Section(e.type);
		CHECK(view != nullptr);
		if (!view) continue;
		CHECK(view->dtype == e.dtype);
		CHECK(view->alignment == e.alignment);
		CHECK(view->byte_size == e.data->size());
		CHECK(view->elem_count == e.elem_count);
		CHECK(view->data != nullptr);
		if (view->data) CHECK(std::memcmp(view->data, e.data->data(), e.data->size()) == 0);
	}
}

// ---------------------------------------------------------------------------
// Curie's S1 tokenizer red suite (SuperSLM_Plan.md §10; Claude/Curie/
// SuperSLM_S1_Tokenizer_TestDesign-2026-07-19.md). The runtime byte-level BPE
// algorithm is already proven bit-for-bit against the upstream HF tokenizer by
// the Python reference (tools/convert_tokenizer.py, 0 mismatch over 2000+
// adversarial+multilingual lines) — this suite is the C++ gate that proves the
// ported TokenizerView reproduces those upstream ids. At S1.3 TokenizerView is
// an unbuilt stub (src/tokenizer.cpp): Open always returns false, Encode/Decode
// always return empty — every cell below fails red for that one reason: the
// golden ids never match empty output, and every cell that depends on a
// successfully-opened view fails its explicit `view_ok` assertion. Once Brunel
// builds the tokenizer, each cell fails red for its own reason (if any) until
// implemented correctly, then goes green.
// ---------------------------------------------------------------------------

namespace {

struct FixtureTokenizer {
	SslmArtifact artifact;
	TokenizerView view;
	bool artifact_ok = false;
	bool view_ok = false;
	std::string artifact_error;
	std::string view_error;
};

FixtureTokenizer OpenFixtureTokenizer() {
	FixtureTokenizer ft;
	std::string path = ResolveFixturePath("qwen2.5-1.5b.tok.sslm");
	if (path.empty()) {
		ft.artifact_error =
		    "fixture qwen2.5-1.5b.tok.sslm not found under tests/fixtures (searched CWD, .., ../..)";
		return ft;
	}
	SslmError aerr;
	auto status = SslmArtifact::OpenFromFile(path.c_str(), ft.artifact, &aerr);
	ft.artifact_ok = (status == SslmStatus::Ok);
	if (!ft.artifact_ok) {
		ft.artifact_error = std::string(SslmStatusName(status)) + ": " + aerr.message;
		return ft;
	}
	std::string terr;
	ft.view_ok = TokenizerView::Open(ft.artifact, ft.view, &terr);
	ft.view_error = terr;
	return ft;
}

}  // namespace

// --- The golden gate (dim 10 — the load-bearing feature oracle): Encode()
//     executed against the real Qwen2.5-1.5B artifact must reproduce the
//     upstream HF tokenizer's ids, not merely agree with itself. ---

static void TestTokenizerGoldenEncodeMatchesUpstreamIds() {
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

static void TestTokenizerGoldenIdsHashMatchesConverter() {
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

static void TestTokenizerDecodeRoundTrip() {
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

static void TestTokenizerOpensFixtureArtifact() {
	auto ft = OpenFixtureTokenizer();
	CHECK_MSG(ft.artifact_ok, "fixture artifact failed to load: %s", ft.artifact_error.c_str());
	CHECK_MSG(ft.view_ok, "TokenizerView::Open failed: %s", ft.view_error.c_str());
	CHECK(ft.view.Ok());
	CHECK(ft.view.VocabSize() > 0);
}

static void TestTokenizerEncodeEmptyStringYieldsEmptyIds() {
	auto ft = OpenFixtureTokenizer();
	CHECK_MSG(ft.artifact_ok, "fixture artifact failed to load: %s", ft.artifact_error.c_str());
	CHECK_MSG(ft.view_ok, "TokenizerView::Open failed: %s", ft.view_error.c_str());
	CHECK(ft.view.Encode("").empty());
}

static void TestTokenizerSpecialTokenIdMatchesArtifactDeclaration() {
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

static void TestTokenizerVocabSizeMatchesArtifactDeclaration() {
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

static void TestTokenizerAsciiStringRoundTrips() {
	auto ft = OpenFixtureTokenizer();
	CHECK_MSG(ft.artifact_ok, "fixture artifact failed to load: %s", ft.artifact_error.c_str());
	CHECK_MSG(ft.view_ok, "TokenizerView::Open failed: %s", ft.view_error.c_str());
	const std::string text = "The quick brown fox jumps over 42 lazy dogs!";
	CHECK(ft.view.Decode(ft.view.Encode(text)) == text);
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

static void TestMinimalTokenizerArtifactOpensAndRoundTrips() {
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
	CHECK(view.VocabSize() == 4);

	// "cat": pretokenizes as one word piece (all ASCII letters); byte-level ids
	// [byte_to_id['c'],['a'],['t']] = [0,1,2]; the one merge (0,1)->3 fires once at
	// the head -> [3,2]. Decode: id3 -> "ca", id2 -> "t" -> "cat".
	std::vector<int32_t> ids = view.Encode("cat");
	std::vector<int32_t> want_ids = {3, 2};
	CHECK_MSG(ids == want_ids, "Encode(\"cat\") produced %zu id(s), want [3,2]", ids.size());
	CHECK(view.Decode(ids) == "cat");

	// The special token matches the whole input and emits its declared id, not a
	// byte-level encoding of its text.
	std::vector<int32_t> special_ids = view.Encode("<eos>");
	CHECK_MSG(special_ids.size() == 1 && special_ids[0] == 1000,
	          "Encode(\"<eos>\") produced %zu id(s), want exactly [1000]", special_ids.size());
}

// --- Structural: TokenizerView::Open requires both sections outright. ---

static void TestOpenRejectsArtifactMissingTokenizerSection() {
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

static void TestOpenRejectsArtifactMissingUnicodeTablesSection() {
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
}

}  // namespace

// --- TOK1 cells. Each isolates exactly one deviation from docs/sslm_format.md's
//     "Tokenizer blob — TOK1" layout in the minimal valid blob. ---

static void TestTok1RejectsBadMagic() {
	auto tok1 = MakeMinimalValidTok1();
	tok1.bytes[0] = 'X';  // was 'T' of "TOK1", offset 0
	AssertTok1Rejected(tok1.bytes, "TOK1 bad magic");
}

static void TestTok1RejectsTruncatedHeader() {
	auto tok1 = MakeMinimalValidTok1();
	tok1.bytes.resize(10);  // shorter than the 24-byte fixed TOK1 header
	AssertTok1Rejected(tok1.bytes, "TOK1 truncated header");
}

static void TestTok1RejectsVocabCountOverflow() {
	auto tok1 = MakeMinimalValidTok1();
	PutU32(tok1.bytes, tok1.layout.vocab_count_off, 0xFFFFFFFFu);
	AssertTok1Rejected(tok1.bytes, "TOK1 vocab_count == 0xFFFFFFFF");
}

static void TestTok1RejectsMergeCountOverflow() {
	auto tok1 = MakeMinimalValidTok1();
	PutU32(tok1.bytes, tok1.layout.merge_count_off, 0xFFFFFFFFu);
	AssertTok1Rejected(tok1.bytes, "TOK1 merge_count == 0xFFFFFFFF");
}

static void TestTok1RejectsSpecialCountOverflow() {
	auto tok1 = MakeMinimalValidTok1();
	PutU32(tok1.bytes, tok1.layout.special_count_off, 0xFFFFFFFFu);
	AssertTok1Rejected(tok1.bytes, "TOK1 special_count == 0xFFFFFFFF");
}

static void TestTok1RejectsTruncatedByteToId() {
	auto tok1 = MakeMinimalValidTok1();
	tok1.bytes.resize(tok1.layout.byte_to_id_off + 100);  // 100 of the required 1024 bytes
	AssertTok1Rejected(tok1.bytes, "TOK1 truncated byte_to_id");
}

static void TestTok1RejectsTruncatedVocabOffsets() {
	auto tok1 = MakeMinimalValidTok1();
	tok1.bytes.resize(tok1.layout.vocab_offsets_off + 4);  // 1 of the required vocab_count+1=5 entries
	AssertTok1Rejected(tok1.bytes, "TOK1 truncated vocab_offsets");
}

static void TestTok1RejectsTruncatedVocabBlobLen() {
	auto tok1 = MakeMinimalValidTok1();
	tok1.bytes.resize(tok1.layout.vocab_blob_len_off + 2);  // half of the 4-byte length field
	AssertTok1Rejected(tok1.bytes, "TOK1 truncated vocab_blob_len");
}

static void TestTok1RejectsTruncatedVocabBlob() {
	auto tok1 = MakeMinimalValidTok1();
	tok1.bytes.resize(tok1.layout.vocab_blob_off + tok1.layout.vocab_blob_len - 1);  // one byte short
	AssertTok1Rejected(tok1.bytes, "TOK1 truncated vocab_blob");
}

static void TestTok1RejectsVocabOffsetNonMonotonic() {
	auto tok1 = MakeMinimalValidTok1();
	// Baseline vocab_offsets [0,1,2,3,5] (vblob=5). Bump index 2 from 2 to 4 (still
	// <= vblob): index 3's value (3) is now smaller than its predecessor (4) — a
	// pure non-monotonic violation, no offset exceeds vblob.
	PutU32(tok1.bytes, tok1.layout.vocab_offsets_off + 2 * 4, 4);
	AssertTok1Rejected(tok1.bytes, "TOK1 vocab offset non-monotonic");
}

static void TestTok1RejectsLastVocabOffsetExceedsBlob() {
	auto tok1 = MakeMinimalValidTok1();
	// Index 4 (vocab_count) is the terminal offset; baseline value 5 == vblob.
	PutU32(tok1.bytes, tok1.layout.vocab_offsets_off + 4 * 4, tok1.layout.vocab_blob_len + 50);
	AssertTok1Rejected(tok1.bytes, "TOK1 last vocab offset > vblob");
}

static void TestTok1RejectsMiddleVocabOffsetExceedsBlob() {
	auto tok1 = MakeMinimalValidTok1();
	// Index 2 is not the terminal offset (index 4 is); baseline value 2.
	PutU32(tok1.bytes, tok1.layout.vocab_offsets_off + 2 * 4, tok1.layout.vocab_blob_len + 50);
	AssertTok1Rejected(tok1.bytes, "TOK1 middle (non-terminal) vocab offset > vblob");
}

static void TestTok1RejectsTruncatedMerges() {
	auto tok1 = MakeMinimalValidTok1();
	tok1.bytes.resize(tok1.layout.merges_off + 6);  // half of the one 12-byte merge record
	AssertTok1Rejected(tok1.bytes, "TOK1 truncated merges");
}

static void TestTok1RejectsTruncatedSpecialIds() {
	auto tok1 = MakeMinimalValidTok1();
	tok1.bytes.resize(tok1.layout.special_ids_off + 2);  // half of the one 4-byte special id
	AssertTok1Rejected(tok1.bytes, "TOK1 truncated special-id table");
}

static void TestTok1RejectsTruncatedSpecialOffsets() {
	auto tok1 = MakeMinimalValidTok1();
	tok1.bytes.resize(tok1.layout.special_offsets_off + 4);  // 1 of the required special_count+1=2 entries
	AssertTok1Rejected(tok1.bytes, "TOK1 truncated special-offset table");
}

static void TestTok1RejectsTruncatedSpecialBlobLen() {
	auto tok1 = MakeMinimalValidTok1();
	tok1.bytes.resize(tok1.layout.special_blob_len_off + 2);  // half of the 4-byte length field
	AssertTok1Rejected(tok1.bytes, "TOK1 truncated special_blob_len");
}

static void TestTok1RejectsTruncatedSpecialBlob() {
	auto tok1 = MakeMinimalValidTok1();
	tok1.bytes.resize(tok1.layout.special_blob_off + tok1.layout.special_blob_len - 1);  // one byte short
	AssertTok1Rejected(tok1.bytes, "TOK1 truncated special_blob");
}

static void TestTok1RejectsSpecialOffsetNonMonotonic() {
	auto tok1 = MakeMinimalValidTok1();
	// Baseline special_offsets [0,5] (sblob=5). Set index 0 to 6 (> index 1's 5) —
	// non-monotonic; index 1 (5) does not itself exceed sblob (5), so the range
	// branch cannot also fire.
	PutU32(tok1.bytes, tok1.layout.special_offsets_off + 0 * 4, 6);
	AssertTok1Rejected(tok1.bytes, "TOK1 special offset non-monotonic");
}

static void TestTok1RejectsSpecialOffsetOutOfRange() {
	auto tok1 = MakeMinimalValidTok1();
	// Leave the (monotonic) offsets [0,5] untouched; lie about special_blob_len
	// instead (5 -> 3), so the terminal offset (5) now exceeds the declared length
	// — a pure range violation, isolated from the monotonic check.
	PutU32(tok1.bytes, tok1.layout.special_blob_len_off, tok1.layout.special_blob_len - 2);
	AssertTok1Rejected(tok1.bytes, "TOK1 special offset out of range (special_blob_len understated)");
}

// --- UNI1 cells. Each isolates exactly one deviation from docs/sslm_format.md's
//     "UnicodeTables blob — UNI1" layout in the minimal valid blob. letter/
//     number/space share one ReadRanges() implementation in src/tokenizer.cpp;
//     the count-field-truncation sub-case is exercised once (on letter, the first
//     call) rather than duplicated three times — the other two axes (range-data
//     truncation, count overflow) are exercised per table, since those are the
//     axes a per-call regression could plausibly differ on. ---

static void TestUni1RejectsBadMagic() {
	auto uni1 = MakeMinimalValidUni1();
	uni1.bytes[0] = 'X';  // was 'U' of "UNI1", offset 0
	AssertUni1Rejected(uni1.bytes, "UNI1 bad magic");
}

static void TestUni1RejectsTruncatedHeader() {
	auto uni1 = MakeMinimalValidUni1();
	uni1.bytes.resize(5);  // shorter than the 8-byte magic+version header
	AssertUni1Rejected(uni1.bytes, "UNI1 truncated header");
}

static void TestUni1RejectsLetterCountFieldTruncated() {
	auto uni1 = MakeMinimalValidUni1();
	uni1.bytes.resize(uni1.layout.letter_count_off + 2);  // half of the 4-byte count field
	AssertUni1Rejected(uni1.bytes, "UNI1 letter range-table count field truncated");
}

static void TestUni1RejectsLetterRangesTruncated() {
	auto uni1 = MakeMinimalValidUni1();
	uni1.bytes.resize(uni1.layout.letter_data_off + 4);  // 4 of the required 16 bytes (2 ranges)
	AssertUni1Rejected(uni1.bytes, "UNI1 letter ranges truncated");
}

static void TestUni1RejectsLetterCountOverflow() {
	auto uni1 = MakeMinimalValidUni1();
	PutU32(uni1.bytes, uni1.layout.letter_count_off, 0xFFFFFFFFu);
	AssertUni1Rejected(uni1.bytes, "UNI1 letter range count == 0xFFFFFFFF");
}

static void TestUni1RejectsNumberRangesTruncated() {
	auto uni1 = MakeMinimalValidUni1();
	uni1.bytes.resize(uni1.layout.number_data_off + 4);  // 4 of the required 8 bytes (1 range)
	AssertUni1Rejected(uni1.bytes, "UNI1 number ranges truncated");
}

static void TestUni1RejectsNumberCountOverflow() {
	auto uni1 = MakeMinimalValidUni1();
	PutU32(uni1.bytes, uni1.layout.number_count_off, 0xFFFFFFFFu);
	AssertUni1Rejected(uni1.bytes, "UNI1 number range count == 0xFFFFFFFF");
}

static void TestUni1RejectsSpaceRangesTruncated() {
	auto uni1 = MakeMinimalValidUni1();
	uni1.bytes.resize(uni1.layout.space_data_off + 8);  // 8 of the required 16 bytes (2 ranges)
	AssertUni1Rejected(uni1.bytes, "UNI1 space ranges truncated");
}

static void TestUni1RejectsSpaceCountOverflow() {
	auto uni1 = MakeMinimalValidUni1();
	PutU32(uni1.bytes, uni1.layout.space_count_off, 0xFFFFFFFFu);
	AssertUni1Rejected(uni1.bytes, "UNI1 space range count == 0xFFFFFFFF");
}

static void TestUni1RejectsCccTruncated() {
	auto uni1 = MakeMinimalValidUni1();
	uni1.bytes.resize(uni1.layout.ccc_data_off + 4);  // 4 of the required 8 bytes (1 entry)
	AssertUni1Rejected(uni1.bytes, "UNI1 ccc table truncated");
}

static void TestUni1RejectsCccCountOverflow() {
	auto uni1 = MakeMinimalValidUni1();
	PutU32(uni1.bytes, uni1.layout.ccc_count_off, 0xFFFFFFFFu);
	AssertUni1Rejected(uni1.bytes, "UNI1 ccc count == 0xFFFFFFFF");
}

static void TestUni1RejectsDecompCpsTruncated() {
	auto uni1 = MakeMinimalValidUni1();
	uni1.bytes.resize(uni1.layout.decomp_cps_off + 2);  // half of the required 4 bytes (1 cp)
	AssertUni1Rejected(uni1.bytes, "UNI1 decomp cps truncated");
}

static void TestUni1RejectsDecompCountOverflow() {
	auto uni1 = MakeMinimalValidUni1();
	PutU32(uni1.bytes, uni1.layout.decomp_count_off, 0xFFFFFFFFu);
	AssertUni1Rejected(uni1.bytes, "UNI1 decomp count == 0xFFFFFFFF");
}

static void TestUni1RejectsDecompOffsetsTruncated() {
	auto uni1 = MakeMinimalValidUni1();
	uni1.bytes.resize(uni1.layout.decomp_offsets_off + 4);  // 1 of the required decomp_count+1=2 entries
	AssertUni1Rejected(uni1.bytes, "UNI1 decomp offsets truncated");
}

static void TestUni1RejectsDecompOffsetOutOfRange() {
	auto uni1 = MakeMinimalValidUni1();
	// Baseline decomp offsets [0,2] (seq_len=2). Bump the terminal offset (index 1)
	// past seq_len while keeping it >= its predecessor — a pure range violation.
	PutU32(uni1.bytes, uni1.layout.decomp_offsets_off + 1 * 4, uni1.layout.seq_len + 97);
	AssertUni1Rejected(uni1.bytes, "UNI1 decomp offset out of range");
}

static void TestUni1RejectsDecompOffsetNonMonotonic() {
	auto uni1 = MakeMinimalValidUni1();
	// Baseline decomp offsets [0,2]. Set index 0 to 3 (> index 1's 2) — non-
	// monotonic; index 1 (2) does not itself exceed seq_len (2).
	PutU32(uni1.bytes, uni1.layout.decomp_offsets_off + 0 * 4, 3);
	AssertUni1Rejected(uni1.bytes, "UNI1 decomp offset non-monotonic");
}

static void TestUni1RejectsDecompSeqTruncated() {
	auto uni1 = MakeMinimalValidUni1();
	uni1.bytes.resize(uni1.layout.decomp_seq_off + 4);  // half of the required 8 bytes (seq_len=2)
	AssertUni1Rejected(uni1.bytes, "UNI1 decomp seq truncated");
}

static void TestUni1RejectsComposeTruncated() {
	auto uni1 = MakeMinimalValidUni1();
	uni1.bytes.resize(uni1.layout.compose_data_off + 6);  // half of the required 12 bytes (1 entry)
	AssertUni1Rejected(uni1.bytes, "UNI1 compose table truncated");
}

static void TestUni1RejectsComposeCountOverflow() {
	auto uni1 = MakeMinimalValidUni1();
	PutU32(uni1.bytes, uni1.layout.compose_count_off, 0xFFFFFFFFu);
	AssertUni1Rejected(uni1.bytes, "UNI1 compose count == 0xFFFFFFFF");
}

int main() {
	TestSha256KnownVectors();
	TestDtypeSizes();
	TestKnownSectionTypes();

	// --- Curie's S0 loader red suite (red-first). ---
	TestRejectsBadMagic();
	TestRejectsUnsupportedVersion();
	TestRejectsHeaderBytesMismatch();
	TestRejectsNonzeroFlags();
	TestRejectsNonzeroReserved0();
	TestRejectsTruncatedHeader();
	TestRejectsTruncatedSectionTable();
	TestRejectsTooManySections();
	TestRejectsFileSizeMismatch();
	TestRejectsAlignmentNotPowerOfTwo();
	TestRejectsAlignmentBelowMinimum();
	TestRejectsAlignmentAboveMaximum();
	TestRejectsMisalignedOffset();
	TestRejectsSectionOutOfBoundsPastEof();
	TestRejectsSectionOffsetOverflow();
	TestRejectsSectionOverlapWithHeader();
	TestRejectsSectionOverlapWithSection();
	TestRejectsBadDtype();
	TestRejectsSectionDtypeMismatch();
	TestRejectsSizeMismatch();
	TestRejectsUnknownSectionType();
	TestRejectsDuplicateSection();
	TestRejectsMissingConfigSection();
	TestRejectsIntegrityMismatch();
	TestAcceptsEmptySection();
	TestAcceptsMaximumAlignment();
	TestAcceptsSectionsInNonAscendingOffsetOrder();
	TestAcceptsReservedSectionTypeStructurally();
	TestOpenFromFileLoadsValidArtifact();
	TestOpenFromFileMissingFileReturnsIoError();
	TestValidArtifactLoadsToExpectedEndState();

	// --- Curie's S1 tokenizer red suite (red-first). ---
	TestTokenizerOpensFixtureArtifact();
	TestTokenizerGoldenEncodeMatchesUpstreamIds();
	TestTokenizerGoldenIdsHashMatchesConverter();
	TestTokenizerDecodeRoundTrip();
	TestTokenizerEncodeEmptyStringYieldsEmptyIds();
	TestTokenizerSpecialTokenIdMatchesArtifactDeclaration();
	TestTokenizerVocabSizeMatchesArtifactDeclaration();
	TestTokenizerAsciiStringRoundTrips();

	// --- Curie's T-129 TOK1/UNI1 sub-parse hostile-input suite (red-first). ---
	TestMinimalTokenizerArtifactOpensAndRoundTrips();
	TestOpenRejectsArtifactMissingTokenizerSection();
	TestOpenRejectsArtifactMissingUnicodeTablesSection();
	TestTok1RejectsBadMagic();
	TestTok1RejectsTruncatedHeader();
	TestTok1RejectsVocabCountOverflow();
	TestTok1RejectsMergeCountOverflow();
	TestTok1RejectsSpecialCountOverflow();
	TestTok1RejectsTruncatedByteToId();
	TestTok1RejectsTruncatedVocabOffsets();
	TestTok1RejectsTruncatedVocabBlobLen();
	TestTok1RejectsTruncatedVocabBlob();
	TestTok1RejectsVocabOffsetNonMonotonic();
	TestTok1RejectsLastVocabOffsetExceedsBlob();
	TestTok1RejectsMiddleVocabOffsetExceedsBlob();
	TestTok1RejectsTruncatedMerges();
	TestTok1RejectsTruncatedSpecialIds();
	TestTok1RejectsTruncatedSpecialOffsets();
	TestTok1RejectsTruncatedSpecialBlobLen();
	TestTok1RejectsTruncatedSpecialBlob();
	TestTok1RejectsSpecialOffsetNonMonotonic();
	TestTok1RejectsSpecialOffsetOutOfRange();
	TestUni1RejectsBadMagic();
	TestUni1RejectsTruncatedHeader();
	TestUni1RejectsLetterCountFieldTruncated();
	TestUni1RejectsLetterRangesTruncated();
	TestUni1RejectsLetterCountOverflow();
	TestUni1RejectsNumberRangesTruncated();
	TestUni1RejectsNumberCountOverflow();
	TestUni1RejectsSpaceRangesTruncated();
	TestUni1RejectsSpaceCountOverflow();
	TestUni1RejectsCccTruncated();
	TestUni1RejectsCccCountOverflow();
	TestUni1RejectsDecompCpsTruncated();
	TestUni1RejectsDecompCountOverflow();
	TestUni1RejectsDecompOffsetsTruncated();
	TestUni1RejectsDecompOffsetOutOfRange();
	TestUni1RejectsDecompOffsetNonMonotonic();
	TestUni1RejectsDecompSeqTruncated();
	TestUni1RejectsComposeTruncated();
	TestUni1RejectsComposeCountOverflow();

	std::printf("superslm tests: %d checks, %d failures\n", GChecks, GFailures);
	return GFailures == 0 ? 0 : 1;
}

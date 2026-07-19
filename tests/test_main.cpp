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
#include "sslm_fixtures.h"

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
	FixtureSection provenance =
	    MakeSection(SslmSectionType::Provenance, SslmDtype::Raw, {1, 2, 3, 4}, /*alignment=*/8);
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

	std::printf("superslm tests: %d checks, %d failures\n", GChecks, GFailures);
	return GFailures == 0 ? 0 : 1;
}

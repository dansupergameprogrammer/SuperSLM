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
#include "superslm/intmath.h"
#include "superslm/matmul.h"
#include "superslm/model.h"
#include "superslm/sha256.h"
#include "superslm/silu_lut.h"
#include "superslm/tokenizer.h"
#include "sslm_cfg1_hostile_fixtures.h"
#include "sslm_fixtures.h"
#include "sslm_intmath_fixtures.h"
#include "sslm_kvc1_hostile_fixtures.h"
#include "sslm_matmul_fixtures.h"
#include "sslm_model_hostile_fixtures.h"
#include "sslm_sil1_hostile_fixtures.h"
#include "sslm_silu_lut_real_vectors_fixtures.h"
#include "silu_lut_golden_table.h"
#include "sslm_tokenizer_fixtures.h"
#include "sslm_tokenizer_hostile_fixtures.h"

#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <thread>
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
	CHECK(!IsKnownSectionType(13u));   // gap in the enum (12 is now SigmoidLut, v2)
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
	    MakeSection(SslmSectionType::Biases, SslmDtype::Int64, EncodeInt64LE({1, 2, 3, 4}));  // 32 bytes, 4 elems
	biases.elem_count_override = 3;  // 32 != 3 * 8
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
	    MakeSection(SslmSectionType::Biases, SslmDtype::Int64,
	                EncodeInt64LE({1000, -2000, 0, 123456}), /*alignment=*/64);
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
	    {SslmSectionType::Biases, SslmDtype::Int64, &biases.data, 4, 64},
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

// ---------------------------------------------------------------------------
// Curie's S2.0a WGT1/BIA1/ROP1 tensor-manifest hostile-input red suite
// (SuperSLM_Plan.md S2.0a; docs/sslm_format.md "Model sub-formats").
// SslmTensorManifest::Parse parses one array section's self-contained tensor
// manifest AFTER SslmArtifact has verified whole-file structure and integrity
// — a crafted integrity-valid artifact can still carry a malformed manifest
// inside a validated section, so this sub-parse is its own hostile-input
// trust boundary, held to the same T-129 bar. src/model.cpp is currently a
// RED-FIRST STUB: Parse() leaves `out` empty and reports Ok unconditionally,
// so every cell below is red until the parse is built.
//
// Every cell starts from a small spec-faithful manifest (sslm_model_hostile_
// fixtures.h), mutates exactly one descriptor or header field, and asserts
// Parse returns the ONE SslmModelStatus its mutation should trigger (not
// merely "some non-Ok status") — every non-Ok SslmModelStatus value is
// covered by at least one cell. WGT1 (int8, element_size 1) carries every
// magic-agnostic structural/per-descriptor cell; BIA1 (int32) and ROP1
// (int64) carry the element-size-DEPENDENT cells (TensorMisaligned; the
// elem_count * element_size 32-bit-overflow class), which are unreachable at
// int8's element_size of 1.
// ---------------------------------------------------------------------------

namespace {

// Shared assertion for every WGT1/BIA1/ROP1 hostile cell below: Parse must
// reject with the cell's one named status, leave `out` empty, and not crash.
void AssertManifestRejected(const std::vector<uint8_t>& mutated_bytes, SslmSectionType type, SslmDtype dtype,
                             SslmModelStatus want, const char* why) {
	SslmSectionView view = MakeManifestSectionView(type, dtype, mutated_bytes);
	SslmTensorManifest out;
	std::string err;
	SslmModelStatus status = SslmTensorManifest::Parse(view, out, &err);
	CHECK_MSG(status == want, "%s: got %s, want %s", why, SslmModelStatusName(status), SslmModelStatusName(want));
	CHECK_MSG(out.Tensors().empty(), "%s: manifest left %zu tensor(s) on a rejected parse", why,
	          out.Tensors().size());
}

}  // namespace

// --- The feature oracles: the minimal fixture each magic's hostile cells
//     mutate from is itself spec-faithful — it parses to Ok and every tensor's
//     name/rank/shape/elem_count/dtype/data matches what was declared. Every
//     hostile cell below attributes a rejection to its one named mutation;
//     these three cells (one per magic) are what prove the baseline isn't
//     rejecting -- or silently misreading -- for some unrelated reason. ---

static void TestWgtMinimalManifestParsesAndRoundTrips() {
	auto m = MakeMinimalValidManifest(kWeightsMagic, /*element_size=*/1);
	SslmSectionView view = MakeManifestSectionView(SslmSectionType::Weights, SslmDtype::Int8, m.bytes);

	SslmTensorManifest manifest;
	std::string err;
	SslmModelStatus status = SslmTensorManifest::Parse(view, manifest, &err);
	CHECK_MSG(status == SslmModelStatus::Ok, "WGT1 minimal manifest failed to parse: got %s: %s",
	          SslmModelStatusName(status), err.c_str());
	if (status != SslmModelStatus::Ok) return;

	CHECK(manifest.Tensors().size() == 4);

	const SslmTensorView* t1 = manifest.Tensor("t1");
	CHECK_MSG(t1 != nullptr, "Tensor(\"t1\") missing");
	if (t1) {
		CHECK(t1->name == "t1");
		CHECK(t1->dtype == SslmDtype::Int8);
		CHECK(t1->rank == 1);
		CHECK(t1->shape[0] == 3);
		CHECK(t1->elem_count == 3);
		CHECK_MSG(t1->data == view.data + m.tensor_data_off[0], "t1.data does not point at its declared data_off");
		CHECK_MSG(t1->data[0] == static_cast<uint8_t>((0 * 31 + 0 * 7 + 11) & 0xFF),
		          "t1's first data byte does not match the fixture's deterministic pattern");
	}

	const SslmTensorView* t2 = manifest.Tensor("t2");
	CHECK_MSG(t2 != nullptr, "Tensor(\"t2\") missing");
	if (t2) {
		CHECK(t2->rank == 2);
		CHECK(t2->shape[0] == 2 && t2->shape[1] == 2);
		CHECK(t2->elem_count == 4);
	}

	const SslmTensorView* t3 = manifest.Tensor("t3");
	CHECK_MSG(t3 != nullptr, "Tensor(\"t3\") missing");
	if (t3) {
		CHECK(t3->rank == 3);
		CHECK(t3->shape[0] == 2 && t3->shape[1] == 1 && t3->shape[2] == 2);
		CHECK(t3->elem_count == 4);
	}

	const SslmTensorView* t4 = manifest.Tensor("t4");
	CHECK_MSG(t4 != nullptr, "Tensor(\"t4\") missing");
	if (t4) {
		CHECK(t4->rank == 4);
		CHECK(t4->shape[0] == 1 && t4->shape[1] == 1 && t4->shape[2] == 1 && t4->shape[3] == 2);
		CHECK(t4->elem_count == 2);
		CHECK_MSG(t4->data[1] == static_cast<uint8_t>((3 * 31 + 1 * 7 + 11) & 0xFF),
		          "t4's second data byte does not match the fixture's deterministic pattern");
	}

	CHECK_MSG(manifest.Tensor("does-not-exist") == nullptr, "Tensor() lookup miss did not return nullptr");
}

static void TestBiaMinimalManifestParsesAndRoundTrips() {
	auto m = MakeMinimalValidManifest(kBiasesMagic, /*element_size=*/4);
	SslmSectionView view = MakeManifestSectionView(SslmSectionType::Biases, SslmDtype::Int32, m.bytes);

	SslmTensorManifest manifest;
	std::string err;
	SslmModelStatus status = SslmTensorManifest::Parse(view, manifest, &err);
	CHECK_MSG(status == SslmModelStatus::Ok, "BIA1 minimal manifest failed to parse: got %s: %s",
	          SslmModelStatusName(status), err.c_str());
	if (status != SslmModelStatus::Ok) return;

	CHECK(manifest.Tensors().size() == 4);

	const SslmTensorView* t1 = manifest.Tensor("t1");
	CHECK_MSG(t1 != nullptr, "Tensor(\"t1\") missing");
	if (t1) {
		CHECK(t1->dtype == SslmDtype::Int32);
		CHECK(t1->rank == 1);
		CHECK(t1->shape[0] == 3);
		CHECK(t1->elem_count == 3);
		CHECK_MSG(t1->data == view.data + m.tensor_data_off[0], "t1.data does not point at its declared data_off");
		CHECK_MSG(t1->data[0] == static_cast<uint8_t>((0 * 31 + 0 * 7 + 11) & 0xFF),
		          "t1's first data byte does not match the fixture's deterministic pattern");
	}

	const SslmTensorView* t4 = manifest.Tensor("t4");
	CHECK_MSG(t4 != nullptr, "Tensor(\"t4\") missing");
	if (t4) {
		CHECK(t4->rank == 4);
		CHECK(t4->shape[0] == 1 && t4->shape[1] == 1 && t4->shape[2] == 1 && t4->shape[3] == 2);
		CHECK(t4->elem_count == 2);
		CHECK_MSG(t4->data == view.data + m.tensor_data_off[3], "t4.data does not point at its declared data_off");
	}

	CHECK_MSG(manifest.Tensor("does-not-exist") == nullptr, "Tensor() lookup miss did not return nullptr");
}

static void TestRopMinimalManifestParsesAndRoundTrips() {
	auto m = MakeMinimalValidManifest(kRopeMagic, /*element_size=*/8);
	SslmSectionView view = MakeManifestSectionView(SslmSectionType::RopeTables, SslmDtype::Int64, m.bytes);

	SslmTensorManifest manifest;
	std::string err;
	SslmModelStatus status = SslmTensorManifest::Parse(view, manifest, &err);
	CHECK_MSG(status == SslmModelStatus::Ok, "ROP1 minimal manifest failed to parse: got %s: %s",
	          SslmModelStatusName(status), err.c_str());
	if (status != SslmModelStatus::Ok) return;

	CHECK(manifest.Tensors().size() == 4);

	const SslmTensorView* t1 = manifest.Tensor("t1");
	CHECK_MSG(t1 != nullptr, "Tensor(\"t1\") missing");
	if (t1) {
		CHECK(t1->dtype == SslmDtype::Int64);
		CHECK(t1->rank == 1);
		CHECK(t1->shape[0] == 3);
		CHECK(t1->elem_count == 3);
		CHECK_MSG(t1->data == view.data + m.tensor_data_off[0], "t1.data does not point at its declared data_off");
		CHECK_MSG(t1->data[0] == static_cast<uint8_t>((0 * 31 + 0 * 7 + 11) & 0xFF),
		          "t1's first data byte does not match the fixture's deterministic pattern");
	}

	const SslmTensorView* t4 = manifest.Tensor("t4");
	CHECK_MSG(t4 != nullptr, "Tensor(\"t4\") missing");
	if (t4) {
		CHECK(t4->rank == 4);
		CHECK(t4->shape[0] == 1 && t4->shape[1] == 1 && t4->shape[2] == 1 && t4->shape[3] == 2);
		CHECK(t4->elem_count == 2);
		CHECK_MSG(t4->data == view.data + m.tensor_data_off[3], "t4.data does not point at its declared data_off");
	}

	CHECK_MSG(manifest.Tensor("does-not-exist") == nullptr, "Tensor() lookup miss did not return nullptr");
}

// --- Section-level / header cells. ---

static void TestManifestRejectsSectionTooShort() {
	std::vector<uint8_t> bytes(10, 0);  // < kManifestHeaderBytes (16)
	bytes[0] = 'W';
	bytes[1] = 'G';
	bytes[2] = 'T';
	bytes[3] = '1';
	AssertManifestRejected(bytes, SslmSectionType::Weights, SslmDtype::Int8, SslmModelStatus::SectionTooShort,
	                        "WGT1 section too short (10 bytes < 16)");
}

static void TestManifestRejectsBadMagicWgt() {
	auto m = MakeMinimalValidManifest(kWeightsMagic, 1);
	m.bytes[0] = 'X';  // was 'W' of "WGT1"
	AssertManifestRejected(m.bytes, SslmSectionType::Weights, SslmDtype::Int8, SslmModelStatus::BadManifestMagic,
	                        "WGT1 bad magic");
}

static void TestManifestRejectsBadMagicBia() {
	auto m = MakeMinimalValidManifest(kBiasesMagic, 4);
	m.bytes[0] = 'X';  // was 'B' of "BIA1"
	AssertManifestRejected(m.bytes, SslmSectionType::Biases, SslmDtype::Int32, SslmModelStatus::BadManifestMagic,
	                        "BIA1 bad magic");
}

static void TestManifestRejectsBadMagicRop() {
	auto m = MakeMinimalValidManifest(kRopeMagic, 8);
	m.bytes[0] = 'X';  // was 'R' of "ROP1"
	AssertManifestRejected(m.bytes, SslmSectionType::RopeTables, SslmDtype::Int64, SslmModelStatus::BadManifestMagic,
	                        "ROP1 bad magic");
}

static void TestManifestRejectsUnsupportedVersion() {
	auto m = MakeMinimalValidManifest(kWeightsMagic, 1);
	PutU32(m.bytes, kManifestVersionOff, 2);
	AssertManifestRejected(m.bytes, SslmSectionType::Weights, SslmDtype::Int8,
	                        SslmModelStatus::UnsupportedManifestVersion, "WGT1 version == 2");
}

static void TestManifestRejectsTooManyTensors() {
	// A complete, valid 16-byte header declaring a tensor_count far beyond
	// kMaxTensors (65536) and no following bytes -- TooManyTensors must be
	// checked (and must reject) before any attempt to read a descriptor table
	// this large, so no real descriptor/name/data bytes are needed here.
	std::vector<uint8_t> bytes;
	bytes.insert(bytes.end(), kWeightsMagic, kWeightsMagic + 4);
	WriteU32LE(bytes, kManifestVersion);
	WriteU32LE(bytes, 0xFFFFFFFFu);  // tensor_count
	WriteU32LE(bytes, 0);            // name_blob_len
	AssertManifestRejected(bytes, SslmSectionType::Weights, SslmDtype::Int8, SslmModelStatus::TooManyTensors,
	                        "WGT1 tensor_count == 0xFFFFFFFF");
}

static void TestManifestRejectsManifestOutOfBoundsTruncatedDescriptors() {
	auto m = MakeMinimalValidManifest(kWeightsMagic, 1);
	// Cut the buffer to strictly inside the descriptor table: one full
	// descriptor plus half of a second, while tensor_count (4) still declares
	// a full four-descriptor table the truncated buffer no longer holds.
	m.bytes.resize(kManifestHeaderBytes + kTensorDescBytes + kTensorDescBytes / 2);
	AssertManifestRejected(m.bytes, SslmSectionType::Weights, SslmDtype::Int8, SslmModelStatus::ManifestOutOfBounds,
	                        "WGT1 truncated mid-descriptor-table");
}

static void TestManifestRejectsManifestOutOfBoundsTruncatedNameBlob() {
	auto m = MakeMinimalValidManifest(kWeightsMagic, 1);
	// The full descriptor table is present, but not all of the declared
	// name_blob_len bytes after it -- cut one byte short of the name blob.
	m.bytes.resize(m.name_blob_off + m.name_blob_len - 1);
	AssertManifestRejected(m.bytes, SslmSectionType::Weights, SslmDtype::Int8, SslmModelStatus::ManifestOutOfBounds,
	                        "WGT1 truncated mid-name-blob");
}

// --- Per-descriptor cells. Each isolates exactly one deviation from
//     docs/sslm_format.md's TensorDesc field table in the minimal valid
//     manifest, tested once (on WGT1) for every magic-agnostic field --
//     magic/element-size only change which magic reads the manifest, never
//     how a descriptor field is validated. ---

static void TestManifestRejectsBadTensorNameOutOfRange() {
	auto m = MakeMinimalValidManifest(kWeightsMagic, 1);
	PutU32(m.bytes, ManifestDescNameLenOff(0), m.name_blob_len + 50);  // t1's name now runs past the blob
	AssertManifestRejected(m.bytes, SslmSectionType::Weights, SslmDtype::Int8, SslmModelStatus::BadTensorName,
	                        "WGT1 tensor[0] (t1) name range exceeds name_blob_len");
}

static void TestManifestRejectsEmptyTensorName() {
	auto m = MakeMinimalValidManifest(kWeightsMagic, 1);
	PutU32(m.bytes, ManifestDescNameLenOff(0), 0);
	AssertManifestRejected(m.bytes, SslmSectionType::Weights, SslmDtype::Int8, SslmModelStatus::EmptyTensorName,
	                        "WGT1 tensor[0] (t1) name_len == 0");
}

static void TestManifestRejectsDuplicateTensorName() {
	auto m = MakeMinimalValidManifest(kWeightsMagic, 1);
	// Point tensor[1] ("t2") at the exact same name-blob range as tensor[0]
	// ("t1") -- both descriptors now name the same tensor.
	const uint32_t t1_name_off = GetU32(m.bytes, ManifestDescNameOffOff(0));
	const uint32_t t1_name_len = GetU32(m.bytes, ManifestDescNameLenOff(0));
	PutU32(m.bytes, ManifestDescNameOffOff(1), t1_name_off);
	PutU32(m.bytes, ManifestDescNameLenOff(1), t1_name_len);
	AssertManifestRejected(m.bytes, SslmSectionType::Weights, SslmDtype::Int8, SslmModelStatus::DuplicateTensorName,
	                        "WGT1 tensor[1] (t2) name duplicates tensor[0]'s (t1)");
}

static void TestManifestRejectsBadTensorRankZero() {
	auto m = MakeMinimalValidManifest(kWeightsMagic, 1);
	PutU32(m.bytes, ManifestDescRankOff(0), 0);
	AssertManifestRejected(m.bytes, SslmSectionType::Weights, SslmDtype::Int8, SslmModelStatus::BadTensorRank,
	                        "WGT1 tensor[0] (t1) rank == 0");
}

static void TestManifestRejectsBadTensorRankTooLarge() {
	auto m = MakeMinimalValidManifest(kWeightsMagic, 1);
	PutU32(m.bytes, ManifestDescRankOff(0), kMaxTensorRank + 1);  // 5
	AssertManifestRejected(m.bytes, SslmSectionType::Weights, SslmDtype::Int8, SslmModelStatus::BadTensorRank,
	                        "WGT1 tensor[0] (t1) rank == kMaxTensorRank+1 (5)");
}

static void TestManifestRejectsBadTensorShapeZeroDim() {
	auto m = MakeMinimalValidManifest(kWeightsMagic, 1);
	// t2 (tensor[1]) is rank 2, shape [2,2]; zero its first dimension -- still
	// "declared" by rank (2) but no longer > 0 as required for i < rank. The
	// elem_count field is patched to match the now-invalid product (0) too, so
	// this cell isolates BadTensorShape alone, not a coincidental
	// ShapeCountMismatch riding along with it.
	PutU32(m.bytes, ManifestDescShapeOff(1, 0), 0);
	PutU64(m.bytes, ManifestDescElemCountOff(1), 0);
	AssertManifestRejected(m.bytes, SslmSectionType::Weights, SslmDtype::Int8, SslmModelStatus::BadTensorShape,
	                        "WGT1 tensor[1] (t2) shape[0] == 0 within rank");
}

static void TestManifestRejectsBadTensorShapeNonzeroPastRank() {
	auto m = MakeMinimalValidManifest(kWeightsMagic, 1);
	// t1 (tensor[0]) is rank 1, shape [3,0,0,0]; set shape[1] (past rank)
	// nonzero. elem_count (3, the product over shape[0..rank)) is unaffected.
	PutU32(m.bytes, ManifestDescShapeOff(0, 1), 5);
	AssertManifestRejected(m.bytes, SslmSectionType::Weights, SslmDtype::Int8, SslmModelStatus::BadTensorShape,
	                        "WGT1 tensor[0] (t1) shape[1] != 0 past rank 1");
}

static void TestManifestRejectsShapeCountMismatch() {
	auto m = MakeMinimalValidManifest(kWeightsMagic, 1);
	// t3 (tensor[2]) is rank 3, shape [2,1,2] -> product 4; declare 999.
	PutU64(m.bytes, ManifestDescElemCountOff(2), 999);
	AssertManifestRejected(m.bytes, SslmSectionType::Weights, SslmDtype::Int8, SslmModelStatus::ShapeCountMismatch,
	                        "WGT1 tensor[2] (t3) elem_count 999 != product(shape) 4");
}

static void TestManifestRejectsTensorOutOfBoundsDataExceedsSection() {
	auto m = MakeMinimalValidManifest(kWeightsMagic, 1);
	// Push t4's (tensor[3]) data_off to one byte before the end of the buffer;
	// its declared elem_count (2) then needs 2 bytes there, exceeding
	// byte_size by 1. t4 is the last tensor packed, so this cannot also
	// overlap t1/t2/t3's earlier ranges.
	PutU64(m.bytes, ManifestDescDataOffOff(3), m.bytes.size() - 1);
	AssertManifestRejected(m.bytes, SslmSectionType::Weights, SslmDtype::Int8, SslmModelStatus::TensorOutOfBounds,
	                        "WGT1 tensor[3] (t4) data range exceeds byte_size by 1");
}

static void TestManifestRejectsTensorOverlap() {
	auto m = MakeMinimalValidManifest(kWeightsMagic, 1);
	// Point t2's (tensor[1]) data_off at t1's (tensor[0]) -- both then claim
	// overlapping byte ranges, while the claimed range still fits comfortably
	// inside byte_size (so TensorOutOfBounds cannot also fire).
	PutU64(m.bytes, ManifestDescDataOffOff(1), m.tensor_data_off[0]);
	AssertManifestRejected(m.bytes, SslmSectionType::Weights, SslmDtype::Int8, SslmModelStatus::TensorOverlap,
	                        "WGT1 tensor[1] (t2) data_off == tensor[0]'s (t1)");
}

static void TestManifestRejectsBadDescriptorReserved() {
	auto m = MakeMinimalValidManifest(kWeightsMagic, 1);
	PutU32(m.bytes, ManifestDescReservedOff(0), 1);
	AssertManifestRejected(m.bytes, SslmSectionType::Weights, SslmDtype::Int8, SslmModelStatus::BadDescriptorReserved,
	                        "WGT1 tensor[0] (t1) reserved == 1");
}

static void TestManifestRejectsDataOffBelowDataRegion() {
	// docs/sslm_format.md's TensorDesc table states data_off must be "multiple
	// of the element size; >= end of the name blob; in bounds" -- three
	// sub-constraints on one field. This cell violates only the middle one:
	// data_off points at the manifest's own magic bytes (well inside the
	// header/descriptor/name-blob region, not past the end of the buffer, and
	// -- at element_size 1 -- trivially "aligned"). There is no dedicated
	// SslmModelStatus for this sub-constraint; TensorOutOfBounds ("a tensor's
	// data range exceeds byte_size or overflows") is the only code whose text
	// plausibly covers a data_off that has strayed outside its VALID region in
	// either direction, so that is what this cell asserts -- flagged in the
	// test-design record as an inference, not a literal spec mapping, for
	// Brunel/Dan to confirm against the implementation.
	auto m = MakeSingleTensorManifest(kWeightsMagic, 1, {3});
	PutU64(m.bytes, ManifestDescDataOffOff(0), 0);
	AssertManifestRejected(m.bytes, SslmSectionType::Weights, SslmDtype::Int8, SslmModelStatus::TensorOutOfBounds,
	                        "WGT1 tensor[0] (t0) data_off == 0, below the data region (into header/descriptors)");
}

// --- Element-size-DEPENDENT cells. TensorMisaligned and the elem_count *
//     element_size 32-bit-overflow class cannot be reached at WGT1's
//     element_size of 1 (any offset is trivially "aligned"; elem_count * 1
//     never differs from elem_count itself) -- these are only reachable at
//     BIA1 (int32, element_size 4) and ROP1 (int64, element_size 8). Each
//     cell uses a single-tensor manifest so an alignment/offset shift on the
//     one tensor present cannot also read as TensorOverlap. ---

static void TestManifestRejectsTensorMisalignedBia() {
	auto m = MakeSingleTensorManifest(kBiasesMagic, 4, {4});
	m.bytes.resize(m.bytes.size() + 64, 0);  // headroom: isolate misalignment from bounds
	PutU64(m.bytes, ManifestDescDataOffOff(0), m.tensor_data_off[0] + 1);  // +1: not a multiple of 4
	AssertManifestRejected(m.bytes, SslmSectionType::Biases, SslmDtype::Int32, SslmModelStatus::TensorMisaligned,
	                        "BIA1 tensor[0] (t0) data_off + 1 (not a multiple of element_size 4)");
}

static void TestManifestRejectsTensorMisalignedRop() {
	auto m = MakeSingleTensorManifest(kRopeMagic, 8, {4});
	m.bytes.resize(m.bytes.size() + 64, 0);  // headroom: isolate misalignment from bounds
	PutU64(m.bytes, ManifestDescDataOffOff(0), m.tensor_data_off[0] + 4);  // +4: not a multiple of 8
	AssertManifestRejected(m.bytes, SslmSectionType::RopeTables, SslmDtype::Int64, SslmModelStatus::TensorMisaligned,
	                        "ROP1 tensor[0] (t0) data_off + 4 (not a multiple of element_size 8)");
}

static void TestManifestRejectsElemCountTimesElementSizeOverflows32BitBia() {
	auto m = MakeSingleTensorManifest(kBiasesMagic, 4, {1});
	// A single-dim shape/elem_count of 1,200,000,000 is well under 2^32
	// (4,294,967,295), so the elem_count field itself does not overflow --
	// but elem_count * element_size (4) == 4,800,000,000 overflows a 32-bit
	// product while fitting trivially in 64-bit. The section's real byte_size
	// is tiny (no data was actually materialized for 1.2B elements), so a
	// correct 64-bit bounds check must reject with TensorOutOfBounds; a
	// 32-bit-wrapped computation could instead land on a small, in-bounds-
	// looking range and wrongly ACCEPT -- the T-129 defect class, one
	// multiplication further in than the section-table-size check it was
	// originally found in.
	PutU32(m.bytes, ManifestDescShapeOff(0, 0), 1200000000u);
	PutU64(m.bytes, ManifestDescElemCountOff(0), 1200000000ull);
	AssertManifestRejected(m.bytes, SslmSectionType::Biases, SslmDtype::Int32, SslmModelStatus::TensorOutOfBounds,
	                        "BIA1 tensor[0] (t0) elem_count(1.2B) * element_size(4) overflows 32 bits");
}

static void TestManifestRejectsElemCountTimesElementSizeOverflows32BitRop() {
	auto m = MakeSingleTensorManifest(kRopeMagic, 8, {1});
	// element_size 8 needs a smaller elem_count to overflow 32 bits at the
	// second multiplication: 600,000,000 * 8 == 4,800,000,000.
	PutU32(m.bytes, ManifestDescShapeOff(0, 0), 600000000u);
	PutU64(m.bytes, ManifestDescElemCountOff(0), 600000000ull);
	AssertManifestRejected(m.bytes, SslmSectionType::RopeTables, SslmDtype::Int64, SslmModelStatus::TensorOutOfBounds,
	                        "ROP1 tensor[0] (t0) elem_count(600M) * element_size(8) overflows 32 bits");
}

static void TestManifestRejectsShapeProductOverflows32BitTensorOutOfBounds() {
	auto m = MakeSingleTensorManifest(kWeightsMagic, 1, {1, 1});
	// shape [70000,70000]: product 4,900,000,000 overflows a 32-bit
	// multiplication (2^32 == 4,294,967,296) but is exactly representable and
	// computable in 64-bit. BOTH the shape and elem_count are set to the
	// correct 64-bit product, so ShapeCountMismatch must NOT fire -- only the
	// bounds check against the section's real (tiny) byte_size should, proving
	// the product itself is computed 64-bit-safe rather than silently
	// wrapping to a smaller value a naive comparison could accept. This is
	// the element-size-independent sibling of the two cells above: it isolates
	// the FIRST multiplication (the shape product itself), tested at WGT1
	// (element_size 1) where the second multiplication (elem_count *
	// element_size) can never itself be the cause.
	PutU32(m.bytes, ManifestDescShapeOff(0, 0), 70000u);
	PutU32(m.bytes, ManifestDescShapeOff(0, 1), 70000u);
	PutU64(m.bytes, ManifestDescElemCountOff(0), 4900000000ull);
	AssertManifestRejected(m.bytes, SslmSectionType::Weights, SslmDtype::Int8, SslmModelStatus::TensorOutOfBounds,
	                        "WGT1 tensor[0] (t0) shape [70000,70000] product 4.9B overflows 32 bits, correct in 64-bit");
}

// ---------------------------------------------------------------------------
// Curie's S2.0b KVC1 keyed-constant sub-parse hostile-input red suite
// (SuperSLM_Plan.md S2.0b; docs/sslm_format.md "Keyed numeric-constant blob —
// KVC1"). SslmKeyedConstants::Parse parses one keyed-constant section's
// self-contained KVC1 table AFTER SslmArtifact has verified whole-file
// structure and integrity — a crafted integrity-valid artifact can still
// carry a malformed KVC1 blob inside a validated section, so this sub-parse
// is its own hostile-input trust boundary, held to the same T-129 bar as
// TOK1/UNI1 and WGT1/BIA1/ROP1. src/model.cpp is currently a RED-FIRST STUB:
// Parse() leaves `out` empty and reports Ok unconditionally, so every cell
// below is red until the parse is built.
//
// Every cell starts from a small spec-faithful KVC1 blob (sslm_kvc1_hostile_
// fixtures.h), mutates exactly one header or descriptor field, and asserts
// Parse returns the ONE SslmModelStatus its mutation should trigger — every
// non-Ok KVC1-specific SslmModelStatus value is covered by at least one
// cell. CompositionConstants (value_words 2) carries every magic-agnostic
// structural/entry-level cell; KvLandingScales and KvLandingReciprocals
// each carry one BadValueWords cell so all three section types are
// exercised (docs/sslm_format.md's KVC1 table).
// ---------------------------------------------------------------------------

namespace {

// Shared assertion for every KVC1 hostile cell below: Parse must reject with
// the cell's one named status, leave `out` empty, and not crash.
void AssertKvc1Rejected(const std::vector<uint8_t>& mutated_bytes, SslmSectionType type, SslmModelStatus want,
                         const char* why) {
	SslmSectionView view = MakeConstantsSectionView(type, mutated_bytes);
	SslmKeyedConstants out;
	std::string err;
	SslmModelStatus status = SslmKeyedConstants::Parse(view, out, &err);
	CHECK_MSG(status == want, "%s: got %s, want %s", why, SslmModelStatusName(status), SslmModelStatusName(want));
	CHECK_MSG(out.Entries().empty(), "%s: KVC1 table left %zu entr(y/ies) on a rejected parse", why,
	          out.Entries().size());
}

}  // namespace

// --- The feature oracles: the minimal fixture each value_words shape's
//     hostile cells mutate from is itself spec-faithful — it parses to Ok
//     and every entry's name/value_words/values matches what was declared,
//     including a negative value and a large-magnitude value (pins the
//     little-endian SIGNED int64 read: a naive unsigned or truncated read
//     would corrupt either). These two cells (one per value_words shape)
//     prove the baseline isn't rejecting -- or silently misreading -- for
//     some unrelated reason before any hostile cell attributes a rejection
//     to its one named mutation. ---

static void TestCompositionConstantsMinimalKvc1ParsesAndRoundTrips() {
	auto m = MakeMinimalValidKvc1(/*value_words=*/2);
	SslmSectionView view = MakeConstantsSectionView(SslmSectionType::CompositionConstants, m.bytes);

	SslmKeyedConstants table;
	std::string err;
	SslmModelStatus status = SslmKeyedConstants::Parse(view, table, &err);
	CHECK_MSG(status == SslmModelStatus::Ok, "CompositionConstants minimal KVC1 failed to parse: got %s: %s",
	          SslmModelStatusName(status), err.c_str());
	if (status != SslmModelStatus::Ok) return;

	CHECK(table.Entries().size() == 3);

	const SslmConstantEntry* alpha = table.Entry("alpha");
	CHECK_MSG(alpha != nullptr, "Entry(\"alpha\") missing");
	if (alpha) {
		CHECK(alpha->name == "alpha");
		CHECK(alpha->value_words == 2);
		CHECK_MSG(SslmKeyedConstants::Value(*alpha, 0) == 7, "alpha[0] != 7");
		CHECK_MSG(SslmKeyedConstants::Value(*alpha, 1) == -3, "alpha[1] != -3");
	}

	const SslmConstantEntry* beta = table.Entry("beta");
	CHECK_MSG(beta != nullptr, "Entry(\"beta\") missing");
	if (beta) {
		CHECK(beta->value_words == 2);
		CHECK_MSG(SslmKeyedConstants::Value(*beta, 0) == -1000000, "beta[0] != -1000000");
		CHECK_MSG(SslmKeyedConstants::Value(*beta, 1) == 42, "beta[1] != 42");
	}

	const SslmConstantEntry* gamma = table.Entry("gamma");
	CHECK_MSG(gamma != nullptr, "Entry(\"gamma\") missing");
	if (gamma) {
		CHECK(gamma->value_words == 2);
		CHECK_MSG(SslmKeyedConstants::Value(*gamma, 0) == 4611686018427387903LL,
		          "gamma[0] != large positive magnitude");
		CHECK_MSG(SslmKeyedConstants::Value(*gamma, 1) == -4611686018427387904LL,
		          "gamma[1] != large negative magnitude");
	}

	CHECK_MSG(table.Entry("does-not-exist") == nullptr, "Entry() lookup miss did not return nullptr");
}

static void TestKvLandingReciprocalsMinimalKvc1ParsesAndRoundTrips() {
	auto m = MakeMinimalValidKvc1(/*value_words=*/3);
	SslmSectionView view = MakeConstantsSectionView(SslmSectionType::KvLandingReciprocals, m.bytes);

	SslmKeyedConstants table;
	std::string err;
	SslmModelStatus status = SslmKeyedConstants::Parse(view, table, &err);
	CHECK_MSG(status == SslmModelStatus::Ok, "KvLandingReciprocals minimal KVC1 failed to parse: got %s: %s",
	          SslmModelStatusName(status), err.c_str());
	if (status != SslmModelStatus::Ok) return;

	CHECK(table.Entries().size() == 3);

	const SslmConstantEntry* alpha = table.Entry("alpha");
	CHECK_MSG(alpha != nullptr, "Entry(\"alpha\") missing");
	if (alpha) {
		CHECK(alpha->value_words == 3);
		CHECK_MSG(SslmKeyedConstants::Value(*alpha, 0) == 7, "alpha[0] != 7");
		CHECK_MSG(SslmKeyedConstants::Value(*alpha, 1) == -3, "alpha[1] != -3");
		CHECK_MSG(SslmKeyedConstants::Value(*alpha, 2) == 1000, "alpha[2] != 1000");
	}

	const SslmConstantEntry* gamma = table.Entry("gamma");
	CHECK_MSG(gamma != nullptr, "Entry(\"gamma\") missing");
	if (gamma) {
		CHECK(gamma->value_words == 3);
		CHECK_MSG(SslmKeyedConstants::Value(*gamma, 0) == 4611686018427387903LL,
		          "gamma[0] != large positive magnitude");
		CHECK_MSG(SslmKeyedConstants::Value(*gamma, 1) == -4611686018427387904LL,
		          "gamma[1] != large negative magnitude");
		CHECK_MSG(SslmKeyedConstants::Value(*gamma, 2) == 9223372036854775807LL, "gamma[2] != INT64_MAX");
	}

	CHECK_MSG(table.Entry("does-not-exist") == nullptr, "Entry() lookup miss did not return nullptr");
}

// --- Section-level / header cells. CompositionConstants (value_words 2) is
//     used throughout: KVC1's magic/version/reserved/entry_count fields are
//     validated identically across all three section types (unlike
//     WGT1/BIA1/ROP1, KVC1 shares ONE magic across every type, so there is
//     no per-type BadConstantsMagic split to make). ---

static void TestKvc1RejectsSectionTooShort() {
	std::vector<uint8_t> bytes(20, 0);  // < kConstantHeaderBytes (24)
	bytes[0] = 'K';
	bytes[1] = 'V';
	bytes[2] = 'C';
	bytes[3] = '1';
	AssertKvc1Rejected(bytes, SslmSectionType::CompositionConstants, SslmModelStatus::SectionTooShort,
	                    "KVC1 section too short (20 bytes < 24)");
}

static void TestKvc1RejectsBadMagic() {
	auto m = MakeMinimalValidKvc1(2);
	m.bytes[0] = 'X';  // was 'K' of "KVC1"
	AssertKvc1Rejected(m.bytes, SslmSectionType::CompositionConstants, SslmModelStatus::BadConstantsMagic,
	                    "KVC1 bad magic");
}

static void TestKvc1RejectsUnsupportedVersion() {
	auto m = MakeMinimalValidKvc1(2);
	PutU32(m.bytes, kConstantsVersionOff, 2);
	AssertKvc1Rejected(m.bytes, SslmSectionType::CompositionConstants, SslmModelStatus::UnsupportedConstantsVersion,
	                    "KVC1 version == 2");
}

static void TestKvc1RejectsTooManyEntries() {
	// A complete, valid 24-byte header declaring an entry_count far beyond
	// kMaxConstantEntries (1048576) and no following bytes -- TooManyConstant
	// Entries must be checked (and must reject) before any attempt to read a
	// descriptor table this large, so no real descriptor/value/name bytes are
	// needed here (mirrors WGT1's TestManifestRejectsTooManyTensors).
	std::vector<uint8_t> bytes;
	bytes.insert(bytes.end(), superslm::kConstantsMagic, superslm::kConstantsMagic + 4);
	WriteU32LE(bytes, kManifestVersion);
	WriteU32LE(bytes, kMaxConstantEntries + 1);  // entry_count
	WriteU32LE(bytes, 2);                        // value_words
	WriteU32LE(bytes, 0);                        // name_blob_len
	WriteU32LE(bytes, 0);                         // reserved
	AssertKvc1Rejected(bytes, SslmSectionType::CompositionConstants, SslmModelStatus::TooManyConstantEntries,
	                    "KVC1 entry_count == kMaxConstantEntries+1");
}

static void TestKvc1RejectsBadReserved() {
	auto m = MakeMinimalValidKvc1(2);
	PutU32(m.bytes, kConstantsReservedOff, 1);
	AssertKvc1Rejected(m.bytes, SslmSectionType::CompositionConstants, SslmModelStatus::BadConstantsReserved,
	                    "KVC1 header reserved == 1");
}

// --- ConstantsOutOfBounds cells. KVC1 has three variable-length regions
//     after the fixed header (descriptors, values, name_blob, in that
//     order) -- one truncation cell per region, plus a fourth that exploits
//     the format's one genuinely unbounded field (name_blob_len, a u32 not
//     capped by kMaxConstantEntries the way entry_count is) to force the
//     header+descriptors+values+name_blob SUM past 2^32, proving the sum is
//     computed 64-bit-safe. See the test-design record §4.4 for why the
//     commission's literally-described "entry_count * value_words * 8
//     overflows 32-bit" scenario is unreachable given kMaxConstantEntries,
//     and why this cell is the faithful adaptation. ---

static void TestKvc1RejectsOutOfBoundsTruncatedDescriptors() {
	auto m = MakeMinimalValidKvc1(2);  // 3 entries
	// Cut the buffer to strictly inside the descriptor table: one full
	// descriptor plus half of a second, while entry_count (3) still declares
	// a full three-descriptor table the truncated buffer no longer holds.
	m.bytes.resize(m.descriptors_off + superslm::kConstantDescBytes + superslm::kConstantDescBytes / 2);
	AssertKvc1Rejected(m.bytes, SslmSectionType::CompositionConstants, SslmModelStatus::ConstantsOutOfBounds,
	                    "KVC1 truncated mid-descriptor-table");
}

static void TestKvc1RejectsOutOfBoundsTruncatedValues() {
	auto m = MakeMinimalValidKvc1(2);  // 3 entries * 2 words * 8 bytes == 48 value bytes
	// The full descriptor table is present, but not all of the declared
	// entry_count*value_words int64s after it -- cut one byte short of the
	// value array.
	m.bytes.resize(m.values_off + static_cast<size_t>(m.entry_count) * m.value_words * 8 - 1);
	AssertKvc1Rejected(m.bytes, SslmSectionType::CompositionConstants, SslmModelStatus::ConstantsOutOfBounds,
	                    "KVC1 truncated mid-value-array");
}

static void TestKvc1RejectsOutOfBoundsTruncatedNameBlob() {
	auto m = MakeMinimalValidKvc1(2);
	// The full descriptor table and value array are present, but not all of
	// the declared name_blob_len bytes after them -- cut one byte short of
	// the name blob.
	m.bytes.resize(m.name_blob_off + m.name_blob_len - 1);
	AssertKvc1Rejected(m.bytes, SslmSectionType::CompositionConstants, SslmModelStatus::ConstantsOutOfBounds,
	                    "KVC1 truncated mid-name-blob");
}

static void TestKvc1RejectsNameBlobLenSumOverflow32Bit() {
	// entry_count is capped at kMaxConstantEntries (1048576), so entry_count *
	// value_words * 8 tops out around 25M -- nowhere near overflowing a
	// 32-bit product on its own (see the test-design record §4.4). name_blob_
	// len, however, is an independent u32 with no such cap: declaring it near
	// UINT32_MAX pushes the header+descriptors+values+name_blob SUM itself
	// past 2^32. A 32-bit-wrapped sum would land on a small value (here, 56)
	// that a tiny real byte_size (74) would wrongly satisfy as "in bounds";
	// only a 64-bit-safe sum correctly sees the true (multi-gigabyte) required
	// size exceeds the section's real, tiny byte_size and rejects.
	std::vector<Kvc1EntrySpec> entries = {{"a", {1, 2}}, {"b", {3, 4}}};
	auto m = BuildKvc1(2, entries);
	CHECK_MSG(m.bytes.size() == 74, "fixture precondition: expected a 74-byte real buffer, got %zu",
	          m.bytes.size());
	PutU32(m.bytes, kConstantsNameBlobLenOff, 0xFFFFFFF0u);  // 4,294,967,280
	AssertKvc1Rejected(m.bytes, SslmSectionType::CompositionConstants, SslmModelStatus::ConstantsOutOfBounds,
	                    "KVC1 name_blob_len == 0xFFFFFFF0, header+descriptors+values+name_blob sum overflows 32-bit");
}

// --- Per-entry cells. Each isolates exactly one deviation from
//     docs/sslm_format.md's EntryDesc field table in the minimal valid
//     blob. ---

static void TestKvc1RejectsBadEntryNameOutOfRange() {
	auto m = MakeMinimalValidKvc1(2);
	PutU32(m.bytes, ConstantsDescNameLenOff(0), m.name_blob_len + 50);  // alpha's name now runs past the blob
	AssertKvc1Rejected(m.bytes, SslmSectionType::CompositionConstants, SslmModelStatus::BadEntryName,
	                    "KVC1 entry[0] (alpha) name range exceeds name_blob_len");
}

static void TestKvc1RejectsEmptyEntryName() {
	auto m = MakeMinimalValidKvc1(2);
	PutU32(m.bytes, ConstantsDescNameLenOff(0), 0);
	AssertKvc1Rejected(m.bytes, SslmSectionType::CompositionConstants, SslmModelStatus::EmptyEntryName,
	                    "KVC1 entry[0] (alpha) name_len == 0");
}

static void TestKvc1RejectsDuplicateEntryName() {
	auto m = MakeMinimalValidKvc1(2);
	// Point entry[1] ("beta") at the exact same name-blob range as entry[0]
	// ("alpha") -- both descriptors now name the same entry.
	const uint32_t alpha_name_off = GetU32(m.bytes, ConstantsDescNameOffOff(0));
	const uint32_t alpha_name_len = GetU32(m.bytes, ConstantsDescNameLenOff(0));
	PutU32(m.bytes, ConstantsDescNameOffOff(1), alpha_name_off);
	PutU32(m.bytes, ConstantsDescNameLenOff(1), alpha_name_len);
	AssertKvc1Rejected(m.bytes, SslmSectionType::CompositionConstants, SslmModelStatus::DuplicateEntryName,
	                    "KVC1 entry[1] (beta) name duplicates entry[0]'s (alpha)");
}

// --- BadValueWords cells: (a) a value not in {2,3} at all, and (b) a value
//     in range but wrong for the section type -- tested in both directions
//     (2-required-but-declares-3, and 3-required-but-declares-2) so all
//     three KVC1 section types are exercised somewhere in this suite (the
//     two feature oracles above cover CompositionConstants and
//     KvLandingReciprocals; these three cells add KvLandingScales and
//     re-exercise the other two from the opposite direction). ---

static void TestKvc1RejectsValueWordsOutOfRange() {
	auto m = MakeMinimalValidKvc1(2);  // KvLandingScales also requires 2
	PutU32(m.bytes, kConstantsValueWordsOff, 5);
	AssertKvc1Rejected(m.bytes, SslmSectionType::KvLandingScales, SslmModelStatus::BadValueWords,
	                    "KVC1 (KvLandingScales) value_words == 5, not in {2,3}");
}

static void TestKvc1RejectsValueWordsWrongForTypeCompositionDeclaresThree() {
	auto m = MakeMinimalValidKvc1(2);  // CompositionConstants requires 2
	PutU32(m.bytes, kConstantsValueWordsOff, 3);
	AssertKvc1Rejected(m.bytes, SslmSectionType::CompositionConstants, SslmModelStatus::BadValueWords,
	                    "KVC1 (CompositionConstants, requires 2) value_words == 3");
}

static void TestKvc1RejectsValueWordsWrongForTypeReciprocalsDeclaresTwo() {
	auto m = MakeMinimalValidKvc1(3);  // KvLandingReciprocals requires 3
	PutU32(m.bytes, kConstantsValueWordsOff, 2);
	AssertKvc1Rejected(m.bytes, SslmSectionType::KvLandingReciprocals, SslmModelStatus::BadValueWords,
	                    "KVC1 (KvLandingReciprocals, requires 3) value_words == 2");
}

// ---------------------------------------------------------------------------
// Curie's S2.0b CFG1 config sub-parse hostile-input red suite (SuperSLM_Plan.md
// S2.0b; docs/sslm_format.md "Config blob — CFG1"). `ParseConfig` parses the
// `Config` section's self-contained fixed 84-byte `CFG1` struct AFTER
// `SslmArtifact::OpenFromMemory` has verified whole-file structure and
// integrity — a crafted integrity-valid artifact can still carry a malformed
// or degraded Config section, so this sub-parse is its own hostile-input
// trust boundary AND the §11 reject-over-degrade gate applied to config: a
// zero dimension or a defaulted field is "a model that loads, runs, generates
// fluent text, and is not the source model" (§6.8 C15) and must be rejected,
// never repaired. src/model.cpp's ParseConfig is currently a RED-FIRST STUB:
// it leaves `out` at SslmModelConfig{} defaults and reports Ok unconditionally,
// so every hostile cell below is red until the parse is built.
//
// Every cell asserts the SPECIFIC SslmModelStatus its one named mutation
// should trigger. CFG1 is a single fixed-layout struct (no variable-length
// regions), so BadConfigSize is the entire bounds surface (both directions —
// too short AND too long, since the parse must reject any byte_size != 84,
// not merely byte_size < 84); the eight dimension fields each get their own
// cell so a parse that only guards some of them is caught.
// ---------------------------------------------------------------------------

namespace {

// Shared assertion for every CFG1 hostile cell below: ParseConfig must reject
// with the cell's one named status and leave `out` at SslmModelConfig{}
// defaults (never a partial or repaired config) — the reject-over-degrade law.
void AssertCfg1Rejected(const std::vector<uint8_t>& mutated_bytes, SslmModelStatus want, const char* why) {
	SslmSectionView view = MakeConfigSectionView(mutated_bytes);
	SslmModelConfig out;
	std::string err;
	SslmModelStatus status = ParseConfig(view, out, &err);
	CHECK_MSG(status == want, "%s: got %s, want %s", why, SslmModelStatusName(status), SslmModelStatusName(want));
	CHECK_MSG(out.hidden_size == 0 && out.num_hidden_layers == 0 && out.num_attention_heads == 0 &&
	              out.num_key_value_heads == 0 && out.head_dim == 0 && out.intermediate_size == 0 &&
	              out.vocab_size == 0 && out.context_cap == 0 && out.tie_word_embeddings == false &&
	              out.kv_precision == SslmKvPrecision::Int8 && out.kv_block_size == 0 && out.unicode_major == 0 &&
	              out.unicode_minor == 0 && out.unicode_patch == 0 && out.rope_theta == 0.0 &&
	              out.rms_norm_eps == 0.0,
	          "%s: SslmModelConfig not left at defaults on a rejected parse", why);
}

}  // namespace

// --- The feature oracle: the minimal fixture every hostile cell mutates from
//     is itself spec-faithful — it parses to Ok and every typed field equals
//     what was baked in, including the true-case bool, the Int16 kv_precision
//     (pinning the enum mapping, not just the zero default), and two
//     distinctive non-round f64s (pinning the little-endian f64 read). This
//     proves the baseline isn't rejecting -- or silently misreading -- before
//     any hostile cell attributes a rejection to its one named mutation. ---

static void TestMinimalCfg1ParsesAndMatchesEveryField() {
	Cfg1Spec spec;  // defaults: see sslm_cfg1_hostile_fixtures.h
	auto bytes = BuildCfg1(spec);
	SslmSectionView view = MakeConfigSectionView(bytes);

	SslmModelConfig out;
	std::string err;
	SslmModelStatus status = ParseConfig(view, out, &err);
	CHECK_MSG(status == SslmModelStatus::Ok, "minimal CFG1 failed to parse: got %s: %s", SslmModelStatusName(status),
	          err.c_str());
	if (status != SslmModelStatus::Ok) return;

	CHECK(out.hidden_size == spec.hidden_size);
	CHECK(out.num_hidden_layers == spec.num_hidden_layers);
	CHECK(out.num_attention_heads == spec.num_attention_heads);
	CHECK(out.num_key_value_heads == spec.num_key_value_heads);
	CHECK(out.head_dim == spec.head_dim);
	CHECK(out.intermediate_size == spec.intermediate_size);
	CHECK(out.vocab_size == spec.vocab_size);
	CHECK(out.context_cap == spec.context_cap);
	CHECK_MSG(out.tie_word_embeddings == true, "tie_word_embeddings (baked in as 1) did not read back true");
	CHECK_MSG(out.kv_precision == SslmKvPrecision::Int16,
	          "kv_precision (baked in as 1) did not read back SslmKvPrecision::Int16");
	CHECK(out.kv_block_size == spec.kv_block_size);
	CHECK(out.unicode_major == spec.unicode_major);
	CHECK(out.unicode_minor == spec.unicode_minor);
	CHECK(out.unicode_patch == spec.unicode_patch);
	CHECK_MSG(out.rope_theta == spec.rope_theta, "rope_theta round-trip mismatch (little-endian f64 read)");
	CHECK_MSG(out.rms_norm_eps == spec.rms_norm_eps, "rms_norm_eps round-trip mismatch (little-endian f64 read)");
}

// --- BadConfigSize — both directions, since the parse must reject ANY
//     byte_size != 84, not merely a too-short buffer. ---

static void TestCfg1RejectsSizeTooShort() {
	auto bytes = MakeMinimalValidCfg1();
	bytes.pop_back();  // 83 bytes, < kConfigBytes (84)
	AssertCfg1Rejected(bytes, SslmModelStatus::BadConfigSize, "CFG1 byte_size == 83 (< 84)");
}

static void TestCfg1RejectsSizeTooLong() {
	auto bytes = MakeMinimalValidCfg1();
	bytes.push_back(0);  // 85 bytes, > kConfigBytes (84)
	AssertCfg1Rejected(bytes, SslmModelStatus::BadConfigSize, "CFG1 byte_size == 85 (> 84)");
}

static void TestCfg1RejectsBadMagic() {
	auto bytes = MakeMinimalValidCfg1();
	bytes[0] = 'X';  // was 'C' of "CFG1"
	AssertCfg1Rejected(bytes, SslmModelStatus::BadConfigMagic, "CFG1 bad magic");
}

static void TestCfg1RejectsUnsupportedVersion() {
	auto bytes = MakeMinimalValidCfg1();
	PutU32(bytes, kCfg1VersionOff, 2);
	AssertCfg1Rejected(bytes, SslmModelStatus::UnsupportedConfigVersion, "CFG1 version == 2");
}

// --- BadConfigDim — all eight dimension fields, each its own cell (a parse
//     that wrongly guards only some of them is caught). ---

static void TestCfg1RejectsZeroHiddenSize() {
	auto bytes = MakeMinimalValidCfg1();
	PutU32(bytes, kCfg1HiddenSizeOff, 0);
	AssertCfg1Rejected(bytes, SslmModelStatus::BadConfigDim, "CFG1 hidden_size == 0");
}

static void TestCfg1RejectsZeroNumHiddenLayers() {
	auto bytes = MakeMinimalValidCfg1();
	PutU32(bytes, kCfg1NumHiddenLayersOff, 0);
	AssertCfg1Rejected(bytes, SslmModelStatus::BadConfigDim, "CFG1 num_hidden_layers == 0");
}

static void TestCfg1RejectsZeroNumAttentionHeads() {
	auto bytes = MakeMinimalValidCfg1();
	PutU32(bytes, kCfg1NumAttentionHeadsOff, 0);
	AssertCfg1Rejected(bytes, SslmModelStatus::BadConfigDim, "CFG1 num_attention_heads == 0");
}

static void TestCfg1RejectsZeroNumKeyValueHeads() {
	auto bytes = MakeMinimalValidCfg1();
	PutU32(bytes, kCfg1NumKeyValueHeadsOff, 0);
	AssertCfg1Rejected(bytes, SslmModelStatus::BadConfigDim, "CFG1 num_key_value_heads == 0");
}

static void TestCfg1RejectsZeroHeadDim() {
	auto bytes = MakeMinimalValidCfg1();
	PutU32(bytes, kCfg1HeadDimOff, 0);
	AssertCfg1Rejected(bytes, SslmModelStatus::BadConfigDim, "CFG1 head_dim == 0");
}

static void TestCfg1RejectsZeroIntermediateSize() {
	auto bytes = MakeMinimalValidCfg1();
	PutU32(bytes, kCfg1IntermediateSizeOff, 0);
	AssertCfg1Rejected(bytes, SslmModelStatus::BadConfigDim, "CFG1 intermediate_size == 0");
}

static void TestCfg1RejectsZeroVocabSize() {
	auto bytes = MakeMinimalValidCfg1();
	PutU32(bytes, kCfg1VocabSizeOff, 0);
	AssertCfg1Rejected(bytes, SslmModelStatus::BadConfigDim, "CFG1 vocab_size == 0");
}

static void TestCfg1RejectsZeroContextCap() {
	auto bytes = MakeMinimalValidCfg1();
	PutU32(bytes, kCfg1ContextCapOff, 0);
	AssertCfg1Rejected(bytes, SslmModelStatus::BadConfigDim, "CFG1 context_cap == 0");
}

// kv_block_size is a required-nonzero field too (docs "Config blob"; Poirot C-1).
static void TestCfg1RejectsZeroKvBlockSize() {
	auto bytes = MakeMinimalValidCfg1();
	PutU32(bytes, kCfg1KvBlockSizeOff, 0);
	AssertCfg1Rejected(bytes, SslmModelStatus::BadConfigDim, "CFG1 kv_block_size == 0");
}

static void TestCfg1RejectsBadKvPrecision() {
	auto bytes = MakeMinimalValidCfg1();
	PutU32(bytes, kCfg1KvPrecisionOff, 2);  // only 0 (Int8) or 1 (Int16) are valid
	AssertCfg1Rejected(bytes, SslmModelStatus::BadKvPrecision, "CFG1 kv_precision == 2");
}

static void TestCfg1RejectsBadConfigBool() {
	auto bytes = MakeMinimalValidCfg1();
	PutU32(bytes, kCfg1TieWordEmbeddingsOff, 2);  // only 0 or 1 are valid
	AssertCfg1Rejected(bytes, SslmModelStatus::BadConfigBool, "CFG1 tie_word_embeddings == 2");
}

static void TestCfg1RejectsBadConfigReserved() {
	auto bytes = MakeMinimalValidCfg1();
	PutU32(bytes, kCfg1ReservedOff, 1);
	AssertCfg1Rejected(bytes, SslmModelStatus::BadConfigReserved, "CFG1 reserved == 1");
}

// ---------------------------------------------------------------------------
// Curie's S2.0b WeightScales manifest-reuse oracle (SuperSLM_Plan.md S2.0b;
// docs/sslm_format.md "Weight-scale fold blob — WSC1"). `WeightScales` (type 6,
// dtype Int32) is now an int32 WSC1 tensor manifest parsed by the ALREADY-
// CERTIFIED `SslmTensorManifest::Parse` (S2.0a) — no new parse code, only the
// magic (kWeightScalesMagic, 'WSC1') and the int32 element type per the
// tensor-manifest rules. This is a confirmation that the new magic + dtype
// wiring routes a WeightScales section through the shipped manifest parse
// correctly, NOT a re-sweep of the manifest parse itself (already hostile-
// swept in S2.0a's WGT1/BIA1/ROP1 suite above) — a small oracle plus one
// wiring-discrimination negative cell suffices.
//
// Unlike every CFG1 cell above, this feature oracle is expected to PASS at
// authoring time: it exercises SslmTensorManifest::Parse, which S2.0a already
// built to green, not src/model.cpp's still-stubbed ParseConfig.
// ---------------------------------------------------------------------------

static void TestWeightScalesMinimalManifestParsesAndRoundTrips() {
	// Column 0 = identity flag (0/1), column 1 = mult, column 2 = shift — one
	// C24/C25 fold-op triple per output channel (docs/sslm_format.md
	// "Weight-scale fold blob — WSC1"). Two channels ("layer0.q_proj",
	// "layer0.k_proj"), shape [2,3], distinctive int32 values per column so a
	// column-order bug (e.g. mult/shift swapped) would be caught by the
	// spot-check below.
	auto m = MakeMinimalValidManifest(kWeightScalesMagic, /*element_size=*/4);
	SslmSectionView view = MakeManifestSectionView(SslmSectionType::WeightScales, SslmDtype::Int32, m.bytes);

	SslmTensorManifest manifest;
	std::string err;
	SslmModelStatus status = SslmTensorManifest::Parse(view, manifest, &err);
	CHECK_MSG(status == SslmModelStatus::Ok, "WSC1 minimal manifest (via WeightScales wiring) failed to parse: got %s: %s",
	          SslmModelStatusName(status), err.c_str());
	if (status != SslmModelStatus::Ok) return;

	CHECK(manifest.Tensors().size() == 4);  // MakeMinimalValidManifest's fixed t1..t4 fixture

	const SslmTensorView* t2 = manifest.Tensor("t2");
	CHECK_MSG(t2 != nullptr, "Tensor(\"t2\") missing (WeightScales-routed WSC1 manifest)");
	if (t2) {
		CHECK(t2->name == "t2");
		CHECK(t2->dtype == SslmDtype::Int32);
		CHECK(t2->rank == 2);
		CHECK(t2->shape[0] == 2 && t2->shape[1] == 2);
		CHECK(t2->elem_count == 4);
		CHECK_MSG(t2->data == view.data + m.tensor_data_off[1],
		          "t2.data does not point at its declared data_off (WeightScales wiring)");
		CHECK_MSG(t2->data[0] == static_cast<uint8_t>((1 * 31 + 0 * 7 + 11) & 0xFF),
		          "t2's first data byte does not match the fixture's deterministic pattern (WeightScales wiring)");
	}
}

static void TestWeightScalesRejectsWrongMagicDiscriminatesPerType() {
	// A WGT1-magic'd blob handed to a WeightScales section: confirms the
	// per-type magic actually discriminates — WeightScales requires 'WSC1',
	// not 'WGT1', even though both are int8/int32-element tensor manifests of
	// the identical structural shape.
	auto m = MakeMinimalValidManifest(kWeightsMagic, /*element_size=*/4);
	SslmSectionView view = MakeManifestSectionView(SslmSectionType::WeightScales, SslmDtype::Int32, m.bytes);

	SslmTensorManifest manifest;
	std::string err;
	SslmModelStatus status = SslmTensorManifest::Parse(view, manifest, &err);
	CHECK_MSG(status == SslmModelStatus::BadManifestMagic,
	          "WGT1-magic'd blob handed to a WeightScales section: got %s, want BadManifestMagic",
	          SslmModelStatusName(status));
	CHECK_MSG(manifest.Tensors().empty(), "manifest left %zu tensor(s) on a rejected parse",
	          manifest.Tensors().size());
}

// ---------------------------------------------------------------------------
// Curie's S2.1 intmath red suite (SuperSLM_Plan.md S2.1; §6.8 C1/C2/C3 +
// C19-C22; Claude/Curie/superslm-s2.1-intmath-test-design-2026-07-19.md).
// Every golden value in sslm_intmath_fixtures.h is computed by CALLING
// Tools/superslm_spike/intmath.py (D-SLM52, the pinned reference) via
// tests/gen_intmath_fixtures.py — never hand-computed. src/intmath.cpp is
// currently a red-first stub (deliberately-wrong sentinel bodies): every
// cell below fails red against those sentinels for its own documented
// reason. Once Brunel ports the real bodies, each cell either passes or
// fails for its own reason until the port is bit-exact against the pinned
// reference, then goes green.
// ---------------------------------------------------------------------------

static void TestC2SaturatingRoundingDoublingHighMul() {
	using namespace superslm_test;
	for (size_t i = 0; i < kC2CasesCount; ++i) {
		const C2Case& c = kC2Cases[i];
		int32_t got = superslm::SaturatingRoundingDoublingHighMul(c.a, c.b);
		CHECK_MSG(got == c.expected, "%s: SaturatingRoundingDoublingHighMul(%d, %d) == %d, want %d",
		          c.label, c.a, c.b, got, c.expected);
	}
}

static void TestC1C3RoundingDivideByPOT() {
	using namespace superslm_test;
	for (size_t i = 0; i < kC1C3CasesCount; ++i) {
		const C1C3Case& c = kC1C3Cases[i];
		int32_t got = superslm::RoundingDivideByPOT(c.x, c.exponent);
		CHECK_MSG(got == c.expected, "%s: RoundingDivideByPOT(%d, %d) == %d, want %d",
		          c.label, c.x, c.exponent, got, c.expected);
	}
}

static void TestMultiplyByQuantizedMultiplier() {
	using namespace superslm_test;
	for (size_t i = 0; i < kMbqmCasesCount; ++i) {
		const MbqmCase& c = kMbqmCases[i];
		int32_t got = superslm::MultiplyByQuantizedMultiplier(c.x, c.multiplier, c.shift);
		CHECK_MSG(got == c.expected, "%s: MultiplyByQuantizedMultiplier(%d, %d, %d) == %d, want %d",
		          c.label, c.x, c.multiplier, c.shift, got, c.expected);
	}
}

static void TestClz64() {
	using namespace superslm_test;
	for (size_t i = 0; i < kClz64CasesCount; ++i) {
		const Clz64Case& c = kClz64Cases[i];
		int got = superslm::Clz64(c.n);
		CHECK_MSG(got == c.expected, "%s: Clz64(%llu) == %d, want %d", c.label,
		          static_cast<unsigned long long>(c.n), got, c.expected);
	}
}

static void TestMaxAbsReduce() {
	using namespace superslm_test;
	for (size_t i = 0; i < kMaxAbsCasesCount; ++i) {
		const MaxAbsCase& c = kMaxAbsCases[i];
		int64_t got = superslm::MaxAbsReduce(c.data, c.n);
		CHECK_MSG(got == c.expected, "%s: MaxAbsReduce(n=%zu) == %lld, want %lld", c.label, c.n,
		          static_cast<long long>(got), static_cast<long long>(c.expected));
	}

	// Order-independence, checked directly rather than only via matching
	// expected values: the three "order_perm_*" fixture cases are the same
	// multiset under three different orderings, so their live results must
	// agree with each other, not merely each with its own precomputed golden.
	int64_t perm_a = superslm::MaxAbsReduce(kMaxAbsData8, 5);
	int64_t perm_b = superslm::MaxAbsReduce(kMaxAbsData9, 5);
	int64_t perm_c = superslm::MaxAbsReduce(kMaxAbsData10, 5);
	CHECK_MSG(perm_a == perm_b && perm_b == perm_c,
	          "MaxAbsReduce is not order-independent: perm_a=%lld perm_b=%lld perm_c=%lld",
	          static_cast<long long>(perm_a), static_cast<long long>(perm_b),
	          static_cast<long long>(perm_c));
}

static void TestNormalizeScale() {
	using namespace superslm_test;
	for (size_t i = 0; i < kNormalizeScaleCasesCount; ++i) {
		const NormalizeScaleCase& c = kNormalizeScaleCases[i];
		superslm::NormalizedScale got = superslm::NormalizeScale(c.d_prime);
		CHECK_MSG(got.dn == c.expected_dn, "%s: NormalizeScale(%lld).dn == %lld, want %lld", c.label,
		          static_cast<long long>(c.d_prime), static_cast<long long>(got.dn),
		          static_cast<long long>(c.expected_dn));
		CHECK_MSG(got.s == c.expected_s, "%s: NormalizeScale(%lld).s == %d, want %d", c.label,
		          static_cast<long long>(c.d_prime), got.s, c.expected_s);
		// Postcondition (C21): 2^30 <= Dn < 2^31 must hold on every case,
		// independent of whether it matches the golden — a bug that produces
		// a wrong-but-still-in-range Dn should not also silently violate the
		// documented contract.
		CHECK_MSG(got.dn >= (INT64_C(1) << 30) && got.dn < (INT64_C(1) << 31),
		          "%s: NormalizeScale(%lld).dn == %lld violates the C21 postcondition [2^30, 2^31)",
		          c.label, static_cast<long long>(c.d_prime), static_cast<long long>(got.dn));
	}
}

static void TestDynamicScaleReciprocalNamed() {
	using namespace superslm_test;
	for (size_t i = 0; i < kDynRecipNamedCasesCount; ++i) {
		const DynRecipNamedCase& c = kDynRecipNamedCases[i];
		int64_t got = superslm::DynamicScaleReciprocal(c.dn);
		CHECK_MSG(got == c.expected_r, "%s: DynamicScaleReciprocal(%lld) == %lld, want %lld", c.label,
		          static_cast<long long>(c.dn), static_cast<long long>(got),
		          static_cast<long long>(c.expected_r));
	}
}

static void TestDynamicScaleReciprocalDenseSample() {
	using namespace superslm_test;
	size_t mismatches = 0;
	for (size_t i = 0; i < kDynRecipDenseCasesCount; ++i) {
		const DynRecipDenseCase& c = kDynRecipDenseCases[i];
		int64_t got = superslm::DynamicScaleReciprocal(c.dn);
		++GChecks;
		if (got != c.expected_r) {
			++GFailures;
			++mismatches;
			if (mismatches <= 20) {
				std::printf("FAIL %s:%d: DynamicScaleReciprocal(%lld) == %lld, want %lld (dense sample)\n",
				            __FILE__, __LINE__, static_cast<long long>(c.dn), static_cast<long long>(got),
				            static_cast<long long>(c.expected_r));
			}
		}
	}
	if (mismatches > 20) {
		std::printf("... %zu additional DynamicScaleReciprocal dense-sample mismatches suppressed\n",
		            mismatches - 20);
	}
}

static void TestRequantTokenCode() {
	using namespace superslm_test;
	for (size_t i = 0; i < kRequantCasesCount; ++i) {
		const RequantCase& c = kRequantCases[i];
		int8_t got = superslm::RequantTokenCode(c.x_i, c.r, c.s);
		CHECK_MSG(got == c.expected, "%s: RequantTokenCode(%d, %lld, %d) == %d, want %d", c.label,
		          c.x_i, static_cast<long long>(c.r), c.s, static_cast<int>(got),
		          static_cast<int>(c.expected));
		CHECK_MSG(got >= -127 && got <= 127, "%s: RequantTokenCode(%d, %lld, %d) == %d, out of [-127, 127]",
		          c.label, c.x_i, static_cast<long long>(c.r), c.s, static_cast<int>(got));
	}
}

static void TestIntmathPipelineComposition() {
	// The feature oracle (dim 10) for the C19-C22 chain: MaxAbsReduce ->
	// NormalizeScale -> DynamicScaleReciprocal -> RequantTokenCode composed
	// exactly as the runtime rung-1 per-token quantizer composes them,
	// asserted against intmath.py's own end-to-end composition rather than
	// against each primitive's isolated golden alone — proves the four
	// C++ ports agree with each other's outputs in sequence, not merely
	// each in isolation.
	using namespace superslm_test;
	for (size_t i = 0; i < kPipelineCasesCount; ++i) {
		const PipelineCase& c = kPipelineCases[i];
		int64_t d_prime = superslm::MaxAbsReduce(c.xs, c.n);
		CHECK_MSG(d_prime == c.expected_d_prime, "%s: pipeline MaxAbsReduce == %lld, want %lld", c.label,
		          static_cast<long long>(d_prime), static_cast<long long>(c.expected_d_prime));

		superslm::NormalizedScale ns = superslm::NormalizeScale(d_prime);
		CHECK_MSG(ns.dn == c.expected_dn, "%s: pipeline NormalizeScale(D').dn == %lld, want %lld", c.label,
		          static_cast<long long>(ns.dn), static_cast<long long>(c.expected_dn));
		CHECK_MSG(ns.s == c.expected_s, "%s: pipeline NormalizeScale(D').s == %d, want %d", c.label, ns.s,
		          c.expected_s);

		int64_t r = superslm::DynamicScaleReciprocal(ns.dn);
		CHECK_MSG(r == c.expected_r, "%s: pipeline DynamicScaleReciprocal(Dn) == %lld, want %lld", c.label,
		          static_cast<long long>(r), static_cast<long long>(c.expected_r));

		for (size_t j = 0; j < c.n; ++j) {
			int8_t code = superslm::RequantTokenCode(c.xs[j], r, ns.s);
			CHECK_MSG(code == c.expected_codes[j],
			          "%s: pipeline RequantTokenCode(x[%zu]=%d, R, s) == %d, want %d", c.label, j, c.xs[j],
			          static_cast<int>(code), static_cast<int>(c.expected_codes[j]));
		}
	}
}

// ---------------------------------------------------------------------------
// Curie's S2.2 nonlinear scalar primitives red suite (red-first; src/intmath.cpp's
// ISqrt/ISqrtTrace/ShiftByMax/IExpFromConstants bodies are currently
// deliberately-wrong stub sentinels — every cell below fails red for that one
// reason until Brunel's port lands. Test-design record: Claude/Curie/
// superslm-s2.2-nonlinear-test-design-2026-07-19.md.
// ---------------------------------------------------------------------------

static void TestISqrt() {
	using namespace superslm_test;
	for (size_t i = 0; i < kISqrtCasesCount; ++i) {
		const ISqrtCase& c = kISqrtCases[i];
		int64_t got = superslm::ISqrt(c.n);
		CHECK_MSG(got == c.expected_root, "%s: ISqrt(%lld) == %lld, want %lld", c.label,
		          static_cast<long long>(c.n), static_cast<long long>(got),
		          static_cast<long long>(c.expected_root));
	}
}

static void TestISqrtTrace() {
	// The §17 blind cell: the fixed 32-iteration recurrence. Every case asserts
	// BOTH that the trace is exactly I_SQRT_ITERATIONS entries long AND that
	// EVERY iterate matches the reference's i_sqrt_trace(n)[k] individually —
	// not only the final root — so a data-dependent early exit (the `while bit
	// > n` prologue intmath.py's docstring explicitly forbids) is caught even
	// if it happens to still land on the correct final root.
	using namespace superslm_test;
	for (size_t i = 0; i < kISqrtCasesCount; ++i) {
		const ISqrtCase& c = kISqrtCases[i];
		int64_t got_iterates[superslm::I_SQRT_ITERATIONS];
		// Poison the buffer so an implementation that writes fewer than
		// I_SQRT_ITERATIONS entries is caught by the per-iterate comparison
		// below rather than silently reading stale/uninitialized data as a
		// coincidental pass.
		for (int k = 0; k < superslm::I_SQRT_ITERATIONS; ++k) got_iterates[k] = INT64_C(-777777777);
		superslm::ISqrtTrace(c.n, got_iterates);
		for (int k = 0; k < superslm::I_SQRT_ITERATIONS; ++k) {
			CHECK_MSG(got_iterates[k] == c.expected_iterates[k],
			          "%s: ISqrtTrace(%lld)[%d] == %lld, want %lld (i_sqrt_trace reference iterate)",
			          c.label, static_cast<long long>(c.n), k, static_cast<long long>(got_iterates[k]),
			          static_cast<long long>(c.expected_iterates[k]));
		}
		CHECK_MSG(got_iterates[superslm::I_SQRT_ITERATIONS - 1] == c.expected_root,
		          "%s: ISqrtTrace(%lld)'s last iterate == %lld, want the ISqrt root %lld", c.label,
		          static_cast<long long>(c.n), static_cast<long long>(got_iterates[superslm::I_SQRT_ITERATIONS - 1]),
		          static_cast<long long>(c.expected_root));
	}
}

static void TestShiftByMax() {
	using namespace superslm_test;
	for (size_t i = 0; i < kShiftByMaxCasesCount; ++i) {
		const ShiftByMaxCase& c = kShiftByMaxCases[i];
		std::vector<int64_t> got(c.n, INT64_C(-777777777));
		superslm::ShiftByMax(c.logits, c.n, got.data());
		for (size_t j = 0; j < c.n; ++j) {
			CHECK_MSG(got[j] == c.expected[j], "%s: ShiftByMax(...)[%zu] == %lld, want %lld", c.label, j,
			          static_cast<long long>(got[j]), static_cast<long long>(c.expected[j]));
		}
		// The feature claim itself (C9): the maximum is at 0 and every result
		// is <= 0 — i-exp's domain requirement — checked live against this
		// call's own output, not only against the precomputed golden array.
		int64_t max_out = got.empty() ? 0 : got[0];
		for (size_t j = 0; j < c.n; ++j) {
			if (got[j] > max_out) max_out = got[j];
			CHECK_MSG(got[j] <= 0, "%s: ShiftByMax(...)[%zu] == %lld, want <= 0", c.label, j,
			          static_cast<long long>(got[j]));
		}
		CHECK_MSG(max_out == 0, "%s: ShiftByMax(...)'s maximum output == %lld, want exactly 0", c.label,
		          static_cast<long long>(max_out));
	}
}

static void TestIExpFromConstants() {
	using namespace superslm_test;
	for (size_t i = 0; i < kIExpCasesCount; ++i) {
		const IExpCase& c = kIExpCases[i];
		int64_t got = superslm::IExpFromConstants(c.q, c.q_ln2, c.q_b, c.q_c);
		CHECK_MSG(got == c.expected, "%s: IExpFromConstants(q=%lld, q_ln2=%lld, q_b=%lld, q_c=%lld) == %lld, want %lld",
		          c.label, static_cast<long long>(c.q), static_cast<long long>(c.q_ln2),
		          static_cast<long long>(c.q_b), static_cast<long long>(c.q_c), static_cast<long long>(got),
		          static_cast<long long>(c.expected));
	}
}

static void TestIExpFromConstantsClipClampsIdenticallyAcrossFamily() {
	// C8's clip claim, checked live across every "realistic_s*"/"qln2_min"
	// family in the fixture set: the clip-boundary input and the beyond-clip
	// input for the SAME (q_ln2, q_b, q_c) triple must produce the SAME
	// result, executed here rather than only asserted equal inside the
	// generator that produced the golden values.
	using namespace superslm_test;
	size_t pairs_checked = 0;
	for (size_t i = 0; i + 1 < kIExpCasesCount; ++i) {
		const IExpCase& clip = kIExpCases[i];
		const IExpCase& beyond = kIExpCases[i + 1];
		std::string clip_label(clip.label);
		std::string beyond_label(beyond.label);
		bool is_clip_pair = clip_label.find("clip_boundary") != std::string::npos &&
		                     beyond_label.find("beyond_clip") != std::string::npos &&
		                     clip.q_ln2 == beyond.q_ln2 && clip.q_b == beyond.q_b && clip.q_c == beyond.q_c;
		if (!is_clip_pair) continue;
		++pairs_checked;
		int64_t got_clip = superslm::IExpFromConstants(clip.q, clip.q_ln2, clip.q_b, clip.q_c);
		int64_t got_beyond = superslm::IExpFromConstants(beyond.q, beyond.q_ln2, beyond.q_b, beyond.q_c);
		CHECK_MSG(got_clip == got_beyond,
		          "%s/%s: clip-boundary result %lld != beyond-clip result %lld for the same constants",
		          clip.label, beyond.label, static_cast<long long>(got_clip), static_cast<long long>(got_beyond));
	}
	CHECK_MSG(pairs_checked == 6, "expected 6 clip/beyond-clip fixture pairs (5 realistic + 1 qln2_min), found %zu",
	          pairs_checked);
}

// ---------------------------------------------------------------------------
// Curie's S2.3 RopeApplyPair red suite (red-first; src/intmath.cpp's
// RopeApplyPair body is currently the deliberately-wrong stub sentinel
// {-1, -1} — every cell below fails red for that one reason until Brunel's
// port lands. Test-design record: Claude/Curie/
// superslm-s2.3-rope-test-design-2026-07-19.md.
// ---------------------------------------------------------------------------

static void TestRopeApplyPair() {
	using namespace superslm_test;
	for (size_t i = 0; i < kRopeCasesCount; ++i) {
		const RopeCase& c = kRopeCases[i];
		superslm::RopePair got = superslm::RopeApplyPair(c.x, c.y, c.cos_q30, c.sin_q30);
		CHECK_MSG(got.x == c.expected_x,
		          "%s: RopeApplyPair(x=%d, y=%d, cos=%d, sin=%d).x == %lld, want %lld", c.label, c.x, c.y,
		          c.cos_q30, c.sin_q30, static_cast<long long>(got.x), static_cast<long long>(c.expected_x));
		CHECK_MSG(got.y == c.expected_y,
		          "%s: RopeApplyPair(x=%d, y=%d, cos=%d, sin=%d).y == %lld, want %lld", c.label, c.x, c.y,
		          c.cos_q30, c.sin_q30, static_cast<long long>(got.y), static_cast<long long>(c.expected_y));
	}
}

static void TestRopeApplyPairIdentityIsExact() {
	// The identity/passthrough claim (dimension 10, feature oracle): rotating by
	// angle 0 (cos=ROPE_ONE, sin=0) must reproduce (x, y) EXACTLY — the
	// x*ROPE_ONE / ROPE_ONE division has zero remainder, so no rounding may
	// perturb the result. Checked live against the executed call, not only
	// against the precomputed golden in kRopeCases.
	using namespace superslm_test;
	const int32_t kOne = superslm::ROPE_ONE;
	const int32_t xs[] = {0, 12345, -12345, superslm::kInt32Max, superslm::kInt32Min};
	for (int32_t x : xs) {
		superslm::RopePair got = superslm::RopeApplyPair(x, x, kOne, 0);
		CHECK_MSG(got.x == static_cast<int64_t>(x), "identity x=%d: RopeApplyPair(...).x == %lld, want %d", x,
		          static_cast<long long>(got.x), x);
		CHECK_MSG(got.y == static_cast<int64_t>(x), "identity x=%d: RopeApplyPair(...).y == %lld, want %d", x,
		          static_cast<long long>(got.y), x);
	}
}

static void TestRopeApplyPairQuarterTurnIsExact() {
	// The quarter-turn claim, executed live: cos=0 isolates the pure-sin
	// rotation, and (-y, x) / (y, -x) must hold exactly for every (x, y) in
	// this suite's kRopeCases quarter-turn cells (found by label prefix).
	using namespace superslm_test;
	size_t checked = 0;
	for (size_t i = 0; i < kRopeCasesCount; ++i) {
		const RopeCase& c = kRopeCases[i];
		std::string label(c.label);
		if (label.rfind("quarter_pos_sin_", 0) == 0) {
			superslm::RopePair got = superslm::RopeApplyPair(c.x, c.y, 0, superslm::ROPE_ONE);
			CHECK_MSG(got.x == -static_cast<int64_t>(c.y) && got.y == static_cast<int64_t>(c.x),
			          "%s: cos=0,sin=ROPE_ONE must give (-y, x) == (%lld, %lld), got (%lld, %lld)", c.label,
			          static_cast<long long>(-c.y), static_cast<long long>(c.x), static_cast<long long>(got.x),
			          static_cast<long long>(got.y));
			++checked;
		} else if (label.rfind("quarter_neg_sin_", 0) == 0) {
			superslm::RopePair got = superslm::RopeApplyPair(c.x, c.y, 0, -superslm::ROPE_ONE);
			CHECK_MSG(got.x == static_cast<int64_t>(c.y) && got.y == -static_cast<int64_t>(c.x),
			          "%s: cos=0,sin=-ROPE_ONE must give (y, -x) == (%lld, %lld), got (%lld, %lld)", c.label,
			          static_cast<long long>(c.y), static_cast<long long>(-c.x), static_cast<long long>(got.x),
			          static_cast<long long>(got.y));
			++checked;
		}
	}
	CHECK_MSG(checked == 8, "expected 8 quarter-turn fixture cells (4 pos-sin + 4 neg-sin), found %zu", checked);
}

static void TestRopeApplyPairWideInputExceedsInt32Range() {
	// The int64-return-type claim (dimension 10, load-bearing): at least one
	// wide-input cell's result must exceed INT32_MAX in magnitude, checked live
	// against this suite's own executed calls — not only asserted true of the
	// precomputed golden inside the generator. A RopePair whose fields were
	// silently truncated to int32 would fail this even if the C++ under test
	// otherwise computed the right 64-bit value internally.
	using namespace superslm_test;
	int wide_exceeding = 0;
	int64_t largest_magnitude = 0;
	for (size_t i = 0; i < kRopeCasesCount; ++i) {
		const RopeCase& c = kRopeCases[i];
		std::string label(c.label);
		if (label.rfind("wide_", 0) != 0) continue;
		superslm::RopePair got = superslm::RopeApplyPair(c.x, c.y, c.cos_q30, c.sin_q30);
		int64_t mag_x = got.x < 0 ? -got.x : got.x;
		int64_t mag_y = got.y < 0 ? -got.y : got.y;
		if (mag_x > largest_magnitude) largest_magnitude = mag_x;
		if (mag_y > largest_magnitude) largest_magnitude = mag_y;
		if (mag_x > superslm::kInt32Max || mag_y > superslm::kInt32Max) ++wide_exceeding;
	}
	CHECK_MSG(wide_exceeding >= 1,
	          "at least one wide-input case must produce a result exceeding INT32_MAX in magnitude "
	          "(the int64 return type is load-bearing); found %d",
	          wide_exceeding);
	CHECK_MSG(largest_magnitude > superslm::kInt32Max,
	          "largest constructed magnitude %lld must exceed INT32_MAX (%d)",
	          static_cast<long long>(largest_magnitude), superslm::kInt32Max);
}

static void TestRopeApplyPairTieRoundsAwayFromZero() {
	// The tie-inheritance claim, executed live: RopeApplyPair's rounding must
	// move strictly off zero on an exact-half input, on both signs, for the
	// x-component alone, the y-component alone, and both simultaneously —
	// proving RoundingDivideByPOT's away-from-zero rule (C3) is inherited
	// correctly rather than merely matching a precomputed golden that happened
	// to be produced the same (wrong) way.
	using namespace superslm_test;
	const char* kTieLabels[] = {"tie_x_pos",  "tie_x_neg",   "tie_y_pos",
	                             "tie_y_neg",  "tie_both_pos", "tie_both_neg"};
	int checked = 0;
	for (const char* want_label : kTieLabels) {
		bool found = false;
		for (size_t i = 0; i < kRopeCasesCount; ++i) {
			const RopeCase& c = kRopeCases[i];
			if (std::strcmp(c.label, want_label) != 0) continue;
			found = true;
			++checked;
			superslm::RopePair got = superslm::RopeApplyPair(c.x, c.y, c.cos_q30, c.sin_q30);
			CHECK_MSG(got.x == c.expected_x, "%s: RopeApplyPair(...).x == %lld, want %lld (away-from-zero tie)",
			          c.label, static_cast<long long>(got.x), static_cast<long long>(c.expected_x));
			CHECK_MSG(got.y == c.expected_y, "%s: RopeApplyPair(...).y == %lld, want %lld (away-from-zero tie)",
			          c.label, static_cast<long long>(got.y), static_cast<long long>(c.expected_y));
			bool moved_off_zero = (got.x != 0) || (got.y != 0);
			CHECK_MSG(moved_off_zero, "%s: an exact-half tie must round strictly off zero", c.label);
		}
		CHECK_MSG(found, "expected a fixture cell labeled \"%s\"", want_label);
	}
	CHECK_MSG(checked == 6, "expected 6 tie fixture cells, found %d", checked);
}

// ---------------------------------------------------------------------------
// Curie's S2.4 SiLU sigmoid-LUT red suite (SuperSLM_S2.4_SiLU_LUT_Design;
// SuperSLM_Plan.md S2.4). Two code-under-test surfaces, both red-first stubs:
//   - `superslm::ParseSigmoidLut` (src/model.cpp) — the SIL1 fixed-layout
//     hostile sub-parse, currently accepts everything and exposes nothing
//     (§8, §12 dim 2/5/9).
//   - `superslm::SiluSigmoidQ15` (src/silu_lut.cpp) — the runtime index
//     derivation + interpolation, currently returns the sentinel INT32_MIN
//     unconditionally (§5, §6, §12 dim 4/6/7/10).
//
// Every cell states, in its own comment, which §12 Coverage Model dimension /
// §10 acceptance item it proves and its oracle kind. Per §12's own discipline:
// the op-level parity cell is a CONSISTENCY oracle (LUT vs an independently
// computed float reference) — blind to a value that is wrong but deterministic
// — so the saturation, interior-interpolation, and downstream int8-agreement
// cells are the REFERENCE/EXACT-VALUE oracles that catch what parity alone
// cannot. Every expected numeric constant below is derived from scratch
// against the §4/§5/§6 formulas (independently re-derived and cross-checked
// against a standalone Python re-implementation of the same formulas before
// authoring), never by calling `SiluSigmoidQ15` itself.
// ---------------------------------------------------------------------------

namespace {

// Shared assertion for every SIL1 hostile cell below: ParseSigmoidLut must
// reject with the cell's one named status and leave `out` at
// SslmSigmoidLut{} defaults (never a partial or repaired view) — the
// reject-over-degrade law, mirroring AssertCfg1Rejected.
void AssertSil1Rejected(const std::vector<uint8_t>& mutated_bytes, SslmModelStatus want, const char* why) {
	SslmSectionView view = MakeSigmoidLutSectionView(mutated_bytes);
	SslmSigmoidLut out;
	std::string err;
	SslmModelStatus status = ParseSigmoidLut(view, out, &err);
	CHECK_MSG(status == want, "%s: got %s, want %s", why, SslmModelStatusName(status), SslmModelStatusName(want));
	CHECK_MSG(out.values == nullptr && out.entry_count == 0,
	          "%s: SslmSigmoidLut not left at defaults on a rejected parse", why);
}

}  // namespace

// --- SIL1 sub-parse: the feature oracle (dim 10) + lifetime/reuse (dim 1) +
//     the payload round-trip (dim 9's third clause). The minimal fixture every
//     hostile cell mutates from is itself spec-faithful: it parses to Ok and
//     every one of the 1025 nodes reads back exactly what the independent
//     reference table computed (BuildReferenceSigmoidLutQ15 — a from-scratch
//     double-precision computation, not a read of any code under test),
//     including the two extreme nodes (table[0], table[N]) whose exact values
//     the domain-clamp cells below also depend on. ---

static void TestMinimalSil1ParsesAndReadsBackAllNodes() {
	using namespace superslm_test;
	std::vector<int32_t> ref = BuildReferenceSigmoidLutQ15();
	CHECK_MSG(ref.size() == kSigmoidLutEntries, "reference table has %zu entries, want %u", ref.size(),
	          kSigmoidLutEntries);

	auto bytes = MakeMinimalValidSil1();
	SslmSectionView view = MakeSigmoidLutSectionView(bytes);
	SslmSigmoidLut out;
	std::string err;
	SslmModelStatus status = ParseSigmoidLut(view, out, &err);
	CHECK_MSG(status == SslmModelStatus::Ok, "minimal SIL1 failed to parse: got %s: %s",
	          SslmModelStatusName(status), err.c_str());
	if (status != SslmModelStatus::Ok) return;

	CHECK_MSG(out.entry_count == kSigmoidLutEntries, "entry_count == %u, want %u", out.entry_count,
	          kSigmoidLutEntries);
	// The red-first stub reports Ok while leaving `out` at SslmSigmoidLut{}
	// defaults (values == nullptr): SigmoidLutValue's contract is undefined
	// for i >= entry_count, so guard against reading through a null/empty
	// view rather than let the stub's false "Ok" crash this cell.
	CHECK_MSG(out.values != nullptr, "out.values is null on a status==Ok parse (red-first stub)");
	if (out.entry_count != kSigmoidLutEntries || out.values == nullptr) return;
	int mismatches = 0;
	for (uint32_t i = 0; i < kSigmoidLutEntries; ++i) {
		int32_t got = SigmoidLutValue(out, i);
		if (got != ref[i]) ++mismatches;
	}
	CHECK_MSG(mismatches == 0, "%d of %u nodes did not read back the reference table's value", mismatches,
	          kSigmoidLutEntries);
	// The two extreme nodes, named explicitly: the domain-clamp saturation
	// cells below assert against these same two values.
	CHECK_MSG(SigmoidLutValue(out, 0) == ref[0], "table[0] == %d, want %d", SigmoidLutValue(out, 0), ref[0]);
	CHECK_MSG(SigmoidLutValue(out, kSiluLutN) == ref[static_cast<size_t>(kSiluLutN)], "table[N] == %d, want %d",
	          SigmoidLutValue(out, kSiluLutN), ref[static_cast<size_t>(kSiluLutN)]);
}

// dim 1 — lifetime and reuse: the same parsed SslmSigmoidLut, read many times
// (standing in for "across many tokens"), must never drift — every repeated
// read of the same node returns the identical value a fresh read would.
static void TestSil1WarmObjectRepeatedReadsShowNoDrift() {
	using namespace superslm_test;
	auto bytes = MakeMinimalValidSil1();
	SslmSectionView view = MakeSigmoidLutSectionView(bytes);
	SslmSigmoidLut out;
	std::string err;
	SslmModelStatus status = ParseSigmoidLut(view, out, &err);
	CHECK_MSG(status == SslmModelStatus::Ok, "warm-object fixture failed to parse: got %s",
	          SslmModelStatusName(status));
	if (status != SslmModelStatus::Ok) return;
	// Guard against the red-first stub's false "Ok" over a null/empty view
	// (see TestMinimalSil1ParsesAndReadsBackAllNodes).
	CHECK_MSG(out.values != nullptr && out.entry_count == kSigmoidLutEntries,
	          "out is not a populated view on a status==Ok parse (red-first stub)");
	if (out.values == nullptr || out.entry_count != kSigmoidLutEntries) return;

	const int32_t first_read_node0 = SigmoidLutValue(out, 0);
	const int32_t first_read_mid = SigmoidLutValue(out, kSiluLutN / 2);
	const int32_t first_read_n = SigmoidLutValue(out, kSiluLutN);
	int drift = 0;
	for (int pass = 0; pass < 1000; ++pass) {
		if (SigmoidLutValue(out, 0) != first_read_node0) ++drift;
		if (SigmoidLutValue(out, kSiluLutN / 2) != first_read_mid) ++drift;
		if (SigmoidLutValue(out, kSiluLutN) != first_read_n) ++drift;
	}
	CHECK_MSG(drift == 0, "%d of 3000 repeated reads drifted from the first read (no re-derivation expected)",
	          drift);
}

// dim 9 (third clause) — SIL1's payload plus its internal magic+version pair
// round-trips bit-exactly after parse + read-back + re-encode.
static void TestSil1RoundTripReencodeMatchesOriginalBytes() {
	using namespace superslm_test;
	auto original = MakeMinimalValidSil1();
	SslmSectionView view = MakeSigmoidLutSectionView(original);
	SslmSigmoidLut out;
	std::string err;
	SslmModelStatus status = ParseSigmoidLut(view, out, &err);
	CHECK_MSG(status == SslmModelStatus::Ok, "round-trip fixture failed to parse: got %s",
	          SslmModelStatusName(status));
	if (status != SslmModelStatus::Ok) return;
	// Guard against the red-first stub's false "Ok" over a null/empty view
	// (see TestMinimalSil1ParsesAndReadsBackAllNodes).
	CHECK_MSG(out.values != nullptr && out.entry_count == kSigmoidLutEntries,
	          "out is not a populated view on a status==Ok parse (red-first stub)");
	if (out.values == nullptr || out.entry_count != kSigmoidLutEntries) return;

	std::vector<int32_t> readback(kSigmoidLutEntries);
	for (uint32_t i = 0; i < kSigmoidLutEntries; ++i) readback[i] = SigmoidLutValue(out, i);
	auto reencoded = BuildSil1(readback, kManifestVersion, kSigmoidLutEntries, /*reserved=*/0);
	CHECK_MSG(reencoded.size() == original.size(), "re-encoded size %zu != original size %zu", reencoded.size(),
	          original.size());
	CHECK_MSG(reencoded == original, "re-encoded SIL1 bytes (magic+version+entry_count+reserved+payload) "
	                                  "do not match the original bytes exactly");
}

// --- SIL1 hostile rejection cells (dim 2/5). SIL1 is a single FIXED-layout
//     struct — no variable-length region — so BadSigmoidLutSize (byte_size !=
//     4116) is the entire bounds surface, both directions. ---

static void TestSil1RejectsSizeTooShort() {
	using namespace superslm_test;
	auto bytes = MakeMinimalValidSil1();
	bytes.pop_back();  // 4115 bytes, < kSigmoidLutBytes (4116)
	AssertSil1Rejected(bytes, SslmModelStatus::BadSigmoidLutSize, "SIL1 byte_size == 4115 (< 4116)");
}

static void TestSil1RejectsSizeTooLong() {
	using namespace superslm_test;
	auto bytes = MakeMinimalValidSil1();
	bytes.push_back(0);  // 4117 bytes, > kSigmoidLutBytes (4116)
	AssertSil1Rejected(bytes, SslmModelStatus::BadSigmoidLutSize, "SIL1 byte_size == 4117 (> 4116)");
}

static void TestSil1RejectsBadMagic() {
	using namespace superslm_test;
	auto bytes = MakeMinimalValidSil1();
	bytes[kSil1MagicOff] = 'X';  // was 'S' of "SIL1"
	AssertSil1Rejected(bytes, SslmModelStatus::BadSigmoidLutMagic, "SIL1 bad magic");
}

static void TestSil1RejectsUnsupportedVersion() {
	using namespace superslm_test;
	auto bytes = MakeMinimalValidSil1();
	PutU32(bytes, kSil1VersionOff, 2);  // only 1 is valid (kManifestVersion)
	AssertSil1Rejected(bytes, SslmModelStatus::UnsupportedSigmoidLutVersion, "SIL1 version == 2");
}

// entry_count is a distinct field from byte_size: this cell holds byte_size at
// exactly 4116 (so BadSigmoidLutSize's check would pass) and mutates only the
// header's declared entry_count, isolating BadSigmoidLutCount from the size
// check above.
static void TestSil1RejectsBadEntryCount() {
	using namespace superslm_test;
	auto bytes = MakeMinimalValidSil1();
	PutU32(bytes, kSil1EntryCountOff, kSigmoidLutEntries - 1);  // 1024, byte_size unchanged at 4116
	AssertSil1Rejected(bytes, SslmModelStatus::BadSigmoidLutCount, "SIL1 entry_count == 1024 (!= 1025)");
}

static void TestSil1RejectsBadReserved() {
	using namespace superslm_test;
	auto bytes = MakeMinimalValidSil1();
	PutU32(bytes, kSil1ReservedOff, 1);
	AssertSil1Rejected(bytes, SslmModelStatus::BadSigmoidLutReserved, "SIL1 reserved == 1");
}

// ---------------------------------------------------------------------------
// dim 9 — version evolution / mutual rejection. Only the CURRENT (v2) loader
// is compiled into this binary, so only one of the design's two named
// directions is executable here: a new-version loader rejecting a
// pre-SigmoidLut (v1-shaped) artifact under the old format_version. The
// reverse direction (an old-version loader rejecting a v2 artifact) is the
// SAME single-field check (format_version != the reader's own compiled-in
// constant) applied symmetrically, and has no independent v1-loader binary to
// execute against in this repository — named here rather than silently
// assumed, mirroring TestRejectsUnsupportedVersion's own one-direction scope
// for the "too new" case (test_main.cpp line ~128).
// ---------------------------------------------------------------------------

static void TestArtifactRejectsPreSigmoidLutV1FormatUnderCurrentLoader() {
	using namespace superslm_test;
	// A v1-shaped artifact: Config only, no SigmoidLut section, format_version
	// one less than the compiled-in v2 constant — the exact "pre-SIL1 artifact
	// under the old format_version" scenario §12 dim 9 names.
	auto built = BuildArtifact({MakeConfigSection()});
	PutU32(built.bytes, 4, kArtifactFormatVersion - 1);  // format_version, offset 4
	RecomputeIntegrityHash(built.bytes);

	SslmArtifact out;
	SslmError err;
	auto status = SslmArtifact::OpenFromMemory(built.bytes.data(), built.bytes.size(), out, &err);
	CHECK_MSG(status == SslmStatus::UnsupportedVersion, "got %s, want UnsupportedVersion",
	          SslmStatusName(status));
	CHECK(err.code == SslmStatus::UnsupportedVersion);
	CHECK(!out.Ok());
}

static void TestArtifactAcceptsV2ArtifactCarryingValidSigmoidLutSection() {
	using namespace superslm_test;
	std::vector<int32_t> ref = BuildReferenceSigmoidLutQ15();
	FixtureSection config = MakeConfigSection();
	FixtureSection sigmoid_lut =
	    MakeSection(SslmSectionType::SigmoidLut, SslmDtype::Int32, BuildSil1(ref), /*alignment=*/64);
	auto built = BuildArtifact({config, sigmoid_lut});  // format_version defaults to kArtifactFormatVersion (2)

	SslmArtifact out;
	SslmError err;
	auto status = SslmArtifact::OpenFromMemory(built.bytes.data(), built.bytes.size(), out, &err);
	CHECK_MSG(status == SslmStatus::Ok, "got %s, want Ok", SslmStatusName(status));
	if (status != SslmStatus::Ok) return;
	CHECK(out.FormatVersion() == kArtifactFormatVersion);

	const SslmSectionView* section = out.Section(SslmSectionType::SigmoidLut);
	CHECK_MSG(section != nullptr, "loaded v2 artifact has no SigmoidLut section");
	if (section == nullptr) return;
	CHECK(section->dtype == SslmDtype::Int32);
	CHECK(section->byte_size == kSigmoidLutBytes);
	// The OUTER artifact-level elem_count is byte_size / dtype_size (the
	// artifact loader's generic per-section bookkeeping — it has no notion of
	// SIL1's internal 16-byte header), NOT the SIL1 sub-parse's own
	// kSigmoidLutEntries (1025 nodes): 4116 bytes / 4-byte Int32 = 1029.
	CHECK(section->elem_count == kSigmoidLutBytes / 4);

	SslmSigmoidLut lut;
	std::string parse_err;
	SslmModelStatus pstatus = ParseSigmoidLut(*section, lut, &parse_err);
	CHECK_MSG(pstatus == SslmModelStatus::Ok, "ParseSigmoidLut on the loaded section: got %s: %s",
	          SslmModelStatusName(pstatus), parse_err.c_str());
	if (pstatus != SslmModelStatus::Ok) return;
	// Guard against the red-first stub's false "Ok" over a null/empty view
	// (see TestMinimalSil1ParsesAndReadsBackAllNodes).
	CHECK_MSG(lut.values != nullptr && lut.entry_count == kSigmoidLutEntries,
	          "lut is not a populated view on a status==Ok parse (red-first stub)");
	if (lut.values == nullptr || lut.entry_count != kSigmoidLutEntries) return;
	CHECK(SigmoidLutValue(lut, 0) == ref[0]);
	CHECK(SigmoidLutValue(lut, kSiluLutN) == ref[static_cast<size_t>(kSiluLutN)]);
}

// ---------------------------------------------------------------------------
// SiluSigmoidQ15 domain-clamp saturation (§12 dim 4a, §10 item 8). Every cell
// forces `pos_fixed` to its clamped extreme (N<<Q_idx or 0) and asserts the
// EXACT extreme node — table[N] or table[0] — never the off-by-one
// table[N-1] the pre-correction defect produced (SuperSLM_S2.4_SiLU_LUT_Design
// §14 amendment 4). Oracle: reference-implementation/exact-value (the
// independent reference table), never a self-consistency check against
// SiluSigmoidQ15's own output. Both the shift<0 (realistic) and shift>=0
// (structurally reachable but never hit by the current calibrated artifact)
// branches are crossed, at both domain extremes, plus a real-corpus deep
// -saturation pair (row from Claude/Laplace/harness/silu_lut/real_rows.npz,
// m=1898583166, e=-32 — the corpus's largest measured realscale, ~0.442,
// which drives x to ~±56 at code=±127, four times past the ±16 table domain).
// ---------------------------------------------------------------------------

static void TestSiluSigmoidQ15SaturatesHighDomainShiftNegativeBranch() {
	using namespace superslm_test;
	std::vector<int32_t> ref = BuildReferenceSigmoidLutQ15();
	// code=127, m=1500000000, e=-30 -> shift = e+k+Q_idx = -30+5+12 = -13 < 0
	// (the RoundingDivideByPOT branch). realscale = m*2^e ~= 1.396, x ~= 177,
	// eleven times past +X=16 -- deep into saturation.
	int32_t got = SiluSigmoidQ15(ref.data(), /*code=*/127, /*m=*/1500000000, /*e=*/-30);
	CHECK_MSG(got == ref[static_cast<size_t>(kSiluLutN)],
	          "SiluSigmoidQ15 saturated-high (shift<0) == %d, want table[N] == %d", got,
	          ref[static_cast<size_t>(kSiluLutN)]);
}

static void TestSiluSigmoidQ15SaturatesLowDomainShiftNegativeBranch() {
	using namespace superslm_test;
	std::vector<int32_t> ref = BuildReferenceSigmoidLutQ15();
	int32_t got = SiluSigmoidQ15(ref.data(), /*code=*/-127, /*m=*/1500000000, /*e=*/-30);
	CHECK_MSG(got == ref[0], "SiluSigmoidQ15 saturated-low (shift<0) == %d, want table[0] == %d", got, ref[0]);
}

static void TestSiluSigmoidQ15SaturatesHighDomainShiftNonNegativeBranch() {
	using namespace superslm_test;
	std::vector<int32_t> ref = BuildReferenceSigmoidLutQ15();
	// e=-17 -> shift = -17+5+12 = 0 exactly (the left-shift branch, term<<0):
	// this is the shift>=0 branch the current calibrated artifact never
	// reaches, forced here so it is not silently dead code. m at its
	// canonical-format floor (2^30) still saturates at any nonzero code,
	// which is structurally forced (§14's own worked derivation): with
	// shift=0 the position contribution equals `code*m` directly, and
	// |code*m| >= 2^30 for any nonzero code given m's floor -- vastly beyond
	// the ~2^22 table domain -- so the shift>=0 branch is exercised here at
	// the one code magnitude (extremes) it can ever produce a defined
	// (saturated) result for; an interior, non-saturated shift>=0 cell is not
	// constructible under the canonical scale format's own m >= 2^30 floor.
	int32_t got = SiluSigmoidQ15(ref.data(), /*code=*/127, /*m=*/1073741824 /*2^30*/, /*e=*/-17);
	CHECK_MSG(got == ref[static_cast<size_t>(kSiluLutN)],
	          "SiluSigmoidQ15 saturated-high (shift==0) == %d, want table[N] == %d", got,
	          ref[static_cast<size_t>(kSiluLutN)]);
}

static void TestSiluSigmoidQ15SaturatesLowDomainShiftNonNegativeBranch() {
	using namespace superslm_test;
	std::vector<int32_t> ref = BuildReferenceSigmoidLutQ15();
	int32_t got = SiluSigmoidQ15(ref.data(), /*code=*/-127, /*m=*/1073741824, /*e=*/-17);
	CHECK_MSG(got == ref[0], "SiluSigmoidQ15 saturated-low (shift==0) == %d, want table[0] == %d", got, ref[0]);
}

static void TestSiluSigmoidQ15RealCorpusDeepSaturationHighCode() {
	using namespace superslm_test;
	std::vector<int32_t> ref = BuildReferenceSigmoidLutQ15();
	// The largest realscale in the measurement corpus (Claude/Laplace/harness/
	// silu_lut/real_rows.npz, row 456): m=1898583166, e=-32, realscale ~=
	// 0.442. At code=127, x ~= 56.1 -- 3.5x past the domain -- a regime the
	// calibrated artifact actually reaches at code extremes, not only a
	// synthetic construction.
	int32_t got = SiluSigmoidQ15(ref.data(), /*code=*/127, /*m=*/1898583166, /*e=*/-32);
	CHECK_MSG(got == ref[static_cast<size_t>(kSiluLutN)],
	          "SiluSigmoidQ15 real-corpus deep saturation (high) == %d, want table[N] == %d", got,
	          ref[static_cast<size_t>(kSiluLutN)]);
}

static void TestSiluSigmoidQ15RealCorpusDeepSaturationLowCode() {
	using namespace superslm_test;
	std::vector<int32_t> ref = BuildReferenceSigmoidLutQ15();
	int32_t got = SiluSigmoidQ15(ref.data(), /*code=*/-127, /*m=*/1898583166, /*e=*/-32);
	CHECK_MSG(got == ref[0], "SiluSigmoidQ15 real-corpus deep saturation (low) == %d, want table[0] == %d", got,
	          ref[0]);
}

// ---------------------------------------------------------------------------
// SiluSigmoidQ15 interior sub-node interpolation (§12 dim 4b), away from the
// boundary, at i0=600 (well interior of [0, N-1]). Both cells use the
// shift<0 branch (e=-29, shift=-12 -- the realistic regime every real
// calibrated (m,e) pair in the corpus lands in). `m` is chosen so the
// RoundingDivideByPOT division is EXACT (no tie ambiguity): m is constructed
// as an exact multiple of 2^12 so `code*m >> 12` (rounded) equals the target
// contribution precisely, isolating the interpolation arithmetic from any
// rounding-direction question. Oracle: reference/exact-value, against the
// independent reference table's table[600]/table[601] and the interpolation
// formula worked by hand (comments below), never SiluSigmoidQ15 called twice.
// ---------------------------------------------------------------------------

static void TestSiluSigmoidQ15InteriorInterpolationFracZero() {
	using namespace superslm_test;
	std::vector<int32_t> ref = BuildReferenceSigmoidLutQ15();
	// code=1, m=1,476,395,008, e=-29 -> shift=-12; term=1*m=1,476,395,008,
	// which is an exact multiple of 4096 (m = 360,448 * 2^12), so
	// RoundingDivideByPOT(term, 12) = 360,448 exactly (no rounding applied).
	// pos_fixed = 360,448 + (N<<Q_idx)/2 = 360,448 + 2,097,152 = 2,457,600
	//           = 600 << 12 exactly -> i0=600, frac=0.
	int32_t got = SiluSigmoidQ15(ref.data(), /*code=*/1, /*m=*/1476395008, /*e=*/-29);
	CHECK_MSG(got == ref[600], "SiluSigmoidQ15 interior frac=0 (i0=600) == %d, want table[600] == %d", got,
	          ref[600]);
}

static void TestSiluSigmoidQ15InteriorInterpolationFracMax() {
	using namespace superslm_test;
	std::vector<int32_t> ref = BuildReferenceSigmoidLutQ15();
	// Same i0=600, same shift=-12; m=1,493,168,128 = 364,543 * 2^12 (also an
	// exact multiple of 4096), so RoundingDivideByPOT(term,12) = 364,543
	// exactly. pos_fixed = 364,543 + 2,097,152 = 2,461,695 = (600<<12) + 4095
	// -> i0=600, frac=4095 (2^Q_idx - 1, the ordinary-case ceiling, distinct
	// from the saturated frac=2^Q_idx of the domain-clamp cells above).
	// diff = table[601]-table[600] = 30856-30799 = 57; product=4095*57=
	// 233,415; RoundingDivideByPOT(233415,12): 233415/4096 ~= 56.986, rounds
	// to 57 (not a tie) -> expected = table[600] + 57 = table[601] exactly.
	int32_t got = SiluSigmoidQ15(ref.data(), /*code=*/1, /*m=*/1493168128, /*e=*/-29);
	CHECK_MSG(got == ref[601],
	          "SiluSigmoidQ15 interior frac=4095 (i0=600) == %d, want table[600]+round(57*4095/4096) == "
	          "table[601] == %d",
	          got, ref[601]);
}

// ---------------------------------------------------------------------------
// SiluSigmoidQ15 code=0 extreme, crossed with the shift>=0 branch at two
// distinct shift magnitudes (dim 4b's "cross code extremes ... and the scale
// range's both branches"): with code=0, term=0 regardless of m or the shift
// value, so pos_fixed lands EXACTLY at the table's own midpoint constant
// (N<<Q_idx)/2 = 512<<12, i.e. i0=512, frac=0 -> table[512] -- proving the
// additive midpoint offset and the clamp/index arithmetic are correct along
// the shift>=0 branch specifically (distinct from the code=+-127 saturation
// cells above, which also use shift>=0 but only exercise the clamped path).
// ---------------------------------------------------------------------------

static void TestSiluSigmoidQ15ShiftZeroBranchCodeZeroReachesMidpointExactly() {
	using namespace superslm_test;
	std::vector<int32_t> ref = BuildReferenceSigmoidLutQ15();
	// e=-17 -> shift = -17+5+12 = 0 exactly.
	int32_t got = SiluSigmoidQ15(ref.data(), /*code=*/0, /*m=*/1500000000, /*e=*/-17);
	CHECK_MSG(got == ref[512], "SiluSigmoidQ15 code=0, shift==0 == %d, want table[512] == %d", got, ref[512]);
}

static void TestSiluSigmoidQ15PositiveShiftBranchCodeZeroReachesMidpointExactly() {
	using namespace superslm_test;
	std::vector<int32_t> ref = BuildReferenceSigmoidLutQ15();
	// e=0 -> shift = 0+5+12 = 17 (strictly positive, the left-shift branch at
	// a nontrivial shift magnitude, not merely the shift==0 identity case).
	int32_t got = SiluSigmoidQ15(ref.data(), /*code=*/0, /*m=*/1500000000, /*e=*/0);
	CHECK_MSG(got == ref[512], "SiluSigmoidQ15 code=0, shift==17 == %d, want table[512] == %d", got, ref[512]);
}

// ---------------------------------------------------------------------------
// Op-level parity vs an independent float reference, within 1 ULP (§10 item 4,
// §12 dim 7/10). CONSISTENCY oracle by construction (LUT vs a reference
// computation of the SAME defined quantity) -- named explicitly per §12's own
// discipline, because a consistency oracle alone cannot see a value that is
// wrong but deterministic; the saturation and interior-interpolation cells
// above, and the downstream int8-agreement cell below, are what catch that
// class of error here. Swept over the full real-vector fixture's (m,e) pairs
// crossed with every code in [-127,127] (224 rows x 255 codes = 57,120
// comparisons), not merely the synthetic single-point cells above.
// ---------------------------------------------------------------------------

static void TestSiluSigmoidQ15OpLevelParityWithinOneUlpOnRealVectors() {
	using namespace superslm_test;
	std::string path = ResolveFixturePath("silu_lut_real_vectors.bin");
	CHECK_MSG(!path.empty(), "silu_lut_real_vectors.bin not found under tests/fixtures");
	if (path.empty()) return;
	SiluLutRealVectors vecs = LoadSiluLutRealVectors(path);
	CHECK_MSG(vecs.ok, "silu_lut_real_vectors.bin failed to parse: %s", vecs.error.c_str());
	if (!vecs.ok) return;

	std::vector<int32_t> table = BuildReferenceSigmoidLutQ15();
	int over_bound = 0;
	int64_t worst = 0;
	for (uint32_t r = 0; r < vecs.row_count; ++r) {
		const double realscale = static_cast<double>(vecs.m[r]) * std::pow(2.0, vecs.e[r]);
		for (int code = -127; code <= 127; ++code) {
			const int32_t lut = SiluSigmoidQ15(table.data(), static_cast<int8_t>(code), vecs.m[r], vecs.e[r]);
			// Fresh independent computation grounded in the definition
			// (sigmoid(x)*2^15), NOT a read of the LUT and NOT a call into
			// any code under test.
			const double x = static_cast<double>(code) * realscale;
			const double xc = x < -static_cast<double>(kSiluLutX)   ? -static_cast<double>(kSiluLutX)
			                   : x > static_cast<double>(kSiluLutX) ? static_cast<double>(kSiluLutX)
			                                                        : x;
			const double sig = 1.0 / (1.0 + std::exp(-xc));
			const int32_t ref_q15 = static_cast<int32_t>(std::nearbyint(sig * 32768.0));
			// Widen to int64 before subtracting: `lut` can be the stub's
			// INT32_MIN sentinel, and `ref_q15 - lut` in 32-bit arithmetic
			// would itself overflow (signed UB) rather than report a large
			// delta.
			const int64_t delta = lut > ref_q15 ? static_cast<int64_t>(lut) - ref_q15
			                                     : static_cast<int64_t>(ref_q15) - lut;
			if (delta > 1) ++over_bound;
			if (delta > worst) worst = delta;
		}
	}
	CHECK_MSG(over_bound == 0, "%d of %u comparisons exceeded the 1-ULP bound (worst delta = %lld)", over_bound,
	          vecs.row_count * 255u, static_cast<long long>(worst));
}

// ---------------------------------------------------------------------------
// Downstream int8-code-agreement (§9 step 5, §10 item 5, §12 dim 7/10 -- the
// feature oracle's intermediate achievement claim). Runs the IDENTICAL subset
// (Claude/Laplace/harness/silu_lut/experiment.py's "Q2" selection: first 8
// tokens of every layer, 28 layers x 8 = 224 rows x 8960 elements, committed
// at tests/fixtures/silu_lut_real_vectors.bin) through the SAME
// already-shipped C19-C22 requant primitives (S2.1) the harness's Python
// port of intmath.py also calls.
//
// The primary acceptance is BAND reproduction, not the harness's exact diff
// count, per the design's own boundary: §10 item 2 states the harness's
// float64 model "is not the golden source for this comparison -- it
// establishes quality, not bits"; §14 amendment 3 / §10 item 4 name the root
// cause directly -- the harness's B_sigmoid TRUNCATES the sub-node position
// (`experiment.py`'s `frac_bits` port), while this construction's runtime
// (§5) ROUNDS it (C3, RoundingDivideByPOT) -- a different decomposition that
// can flip which borderline int8 codes land on which side of a boundary,
// changing the exact diff count while preserving the statistic. §10 item 5
// states the acceptance explicitly: the re-measurement "reproduces its
// 0.016-0.021% band ... not a new number" -- band, not count. Oracle kind:
// REFERENCE/EXACT-VALUE against the band + the max-delta bound (both from
// the design's own acceptance criteria), not a self-consistency check.
//
// (2026-07-19 correction, folded from Brunel's flag on greening this cell:
// the initial authoring over-specified the oracle by asserting the harness's
// own exact count (319) as bit-golden for a construction the design itself
// says produces a different decomposition. Brunel's build measured 338 diffs
// (0.0168%), max |code delta| = 1 -- inside the cited band, consistent with
// the op-level <=1-ULP parity cell (§3.7) passing at 0 over-bound on the same
// vectors -- so this is an oracle fix, not a defect in §5/§6. The exact count
// is now an informational comment only, not asserted.)
// ---------------------------------------------------------------------------

static void TestSiluSigmoidQ15DownstreamInt8AgreementReproducesLaplaceBand() {
	using namespace superslm_test;
	std::string path = ResolveFixturePath("silu_lut_real_vectors.bin");
	CHECK_MSG(!path.empty(), "silu_lut_real_vectors.bin not found under tests/fixtures");
	if (path.empty()) return;
	SiluLutRealVectors vecs = LoadSiluLutRealVectors(path);
	CHECK_MSG(vecs.ok, "silu_lut_real_vectors.bin failed to parse: %s", vecs.error.c_str());
	if (!vecs.ok) return;
	CHECK_MSG(vecs.row_count == 224, "fixture row_count == %u, want 224 (the Q2 subset)", vecs.row_count);
	CHECK_MSG(vecs.width == 8960, "fixture width == %u, want 8960", vecs.width);

	std::vector<int32_t> table = BuildReferenceSigmoidLutQ15();
	uint64_t total_codes = 0;
	uint64_t diff_count = 0;
	int32_t max_delta = 0;

	std::vector<int32_t> wide_lut(vecs.width);
	std::vector<int32_t> wide_ref(vecs.width);

	for (uint32_t r = 0; r < vecs.row_count; ++r) {
		const double realscale = static_cast<double>(vecs.m[r]) * std::pow(2.0, vecs.e[r]);
		// Sigmoid(code) depends only on the code value and this row's scale,
		// not on element position -- computed once per distinct code,
		// applied across the row (matching the construction's own contract:
		// SiluSigmoidQ15 is a pure function of (code, m, e)).
		int32_t lut_by_code[255];    // index [code+127]
		int32_t ref_by_code[255];
		for (int code = -127; code <= 127; ++code) {
			lut_by_code[code + 127] =
			    SiluSigmoidQ15(table.data(), static_cast<int8_t>(code), vecs.m[r], vecs.e[r]);
			const double x = static_cast<double>(code) * realscale;
			const double xc = x < -static_cast<double>(kSiluLutX)   ? -static_cast<double>(kSiluLutX)
			                   : x > static_cast<double>(kSiluLutX) ? static_cast<double>(kSiluLutX)
			                                                        : x;
			const double sig = 1.0 / (1.0 + std::exp(-xc));
			ref_by_code[code + 127] = static_cast<int32_t>(std::nearbyint(sig * 32768.0));
		}

		const size_t base = static_cast<size_t>(r) * vecs.width;
		for (uint32_t i = 0; i < vecs.width; ++i) {
			const int g = vecs.g_codes[base + i];
			const int u = vecs.u_codes[base + i];
			wide_lut[i] = g * lut_by_code[g + 127] * u;
			wide_ref[i] = g * ref_by_code[g + 127] * u;
		}

		// The exact C19-C22 per-token requant chain (S2.1, already shipped
		// and certified) -- one dynamic scale derived per row, exactly as
		// experiment.py's requant_row_int8 does.
		const int64_t d_lut = MaxAbsReduce(wide_lut.data(), vecs.width);
		const NormalizedScale n_lut = NormalizeScale(d_lut);
		const int64_t r_lut = DynamicScaleReciprocal(n_lut.dn);
		const int64_t d_ref = MaxAbsReduce(wide_ref.data(), vecs.width);
		const NormalizedScale n_ref = NormalizeScale(d_ref);
		const int64_t r_ref = DynamicScaleReciprocal(n_ref.dn);

		for (uint32_t i = 0; i < vecs.width; ++i) {
			const int8_t code_lut = RequantTokenCode(wide_lut[i], r_lut, n_lut.s);
			const int8_t code_ref = RequantTokenCode(wide_ref[i], r_ref, n_ref.s);
			++total_codes;
			if (code_lut != code_ref) {
				++diff_count;
				const int32_t delta = code_lut > code_ref ? code_lut - code_ref : code_ref - code_lut;
				if (delta > max_delta) max_delta = delta;
			}
		}
	}

	CHECK_MSG(total_codes == 2007040, "total_codes == %llu, want 2007040",
	          static_cast<unsigned long long>(total_codes));
	// Informational only, not asserted: the Laplace C10 solve's own harness
	// measured 319 diffs of 2,007,040 (0.0159%) on this exact subset
	// (result_packet.json "downstream" key, B_champ N1024_X16_Q12idx) using
	// its truncating sub-node index port. This construction rounds instead
	// (C3), so its exact count is expected to differ -- see the comment
	// above this function. `diff_count` and the derived `pct` are printed via
	// CHECK_MSG's failure path only if the band check below fails, so a
	// regression names its own actual numbers.
	const double pct = 100.0 * static_cast<double>(diff_count) / static_cast<double>(total_codes);
	CHECK_MSG(pct >= 0.016 && pct <= 0.021,
	          "downstream int8-agreement %.4f%% (%llu/%llu diffs) outside the design's cited 0.016-0.021%% band",
	          pct, static_cast<unsigned long long>(diff_count), static_cast<unsigned long long>(total_codes));
	CHECK_MSG(max_delta <= 1, "max |code delta| == %d, want <= 1", max_delta);
}

// ---------------------------------------------------------------------------
// Concurrency (§12 dim 3): 8 reader threads x >=10,000 lookups each, spanning
// index values across [0, N] (interior and both saturated regions), every
// threaded read asserted byte-identical to a single-threaded read of the same
// (code, m, e). The table has no mutable state once loaded -- a structural
// argument -- backed here by the executed stress this dimension's catalog
// entry requires. Runs as a plain stress under the local MSVC build
// (build.bat); the project's ThreadSanitizer leg is the separate CI clang
// build, not this binary -- no sanitizer is invoked here. CHECK/CHECK_MSG
// touch non-atomic global counters (GChecks/GFailures) and are NOT
// thread-safe, so every thread body accumulates its own local mismatch count
// and only the joining main thread calls CHECK.
// ---------------------------------------------------------------------------

static void TestSiluSigmoidQ15ConcurrentReadsMatchSingleThreaded() {
	using namespace superslm_test;
	std::vector<int32_t> table = BuildReferenceSigmoidLutQ15();

	// One realistic interior/saturating scale (m=1,500,000,000, e=-32;
	// realscale ~= 0.349, so code in [-127,127] maps x across
	// [-44.3, 44.3] -- comfortably spanning the interior AND both saturated
	// ends of the [-16,16] domain), crossed with every code, repeated 40x:
	// 255 * 40 = 10,200 lookups per thread, each an independent call.
	constexpr int64_t kM = 1500000000;
	constexpr int kE = -32;
	constexpr int kSweeps = 40;
	constexpr int kCodesPerSweep = 255;
	constexpr int kLookupsPerThread = kSweeps * kCodesPerSweep;  // 10,200 >= 10,000
	constexpr int kThreads = 8;

	std::vector<int32_t> baseline(kLookupsPerThread);
	for (int sweep = 0; sweep < kSweeps; ++sweep) {
		for (int code = -127; code <= 127; ++code) {
			baseline[static_cast<size_t>(sweep * kCodesPerSweep + (code + 127))] =
			    SiluSigmoidQ15(table.data(), static_cast<int8_t>(code), kM, kE);
		}
	}

	std::vector<int> per_thread_mismatches(kThreads, 0);
	std::vector<std::thread> threads;
	threads.reserve(kThreads);
	for (int t = 0; t < kThreads; ++t) {
		threads.emplace_back([&table, &baseline, &per_thread_mismatches, t]() {
			int mismatches = 0;
			for (int sweep = 0; sweep < kSweeps; ++sweep) {
				for (int code = -127; code <= 127; ++code) {
					const int32_t got = SiluSigmoidQ15(table.data(), static_cast<int8_t>(code), kM, kE);
					const size_t idx = static_cast<size_t>(sweep * kCodesPerSweep + (code + 127));
					if (got != baseline[idx]) ++mismatches;
				}
			}
			per_thread_mismatches[static_cast<size_t>(t)] = mismatches;
		});
	}
	for (auto& th : threads) th.join();

	int total_mismatches = 0;
	for (int m : per_thread_mismatches) total_mismatches += m;
	CHECK_MSG(total_mismatches == 0,
	          "%d of %d concurrent reads (8 threads x %d lookups) did not match the single-threaded baseline",
	          total_mismatches, kThreads * kLookupsPerThread, kLookupsPerThread);
}

// ---------------------------------------------------------------------------
// Division-free hot path (§12 dim 7, low priority per the commission brief).
// Not an executed probe: SiluSigmoidQ15's contract (include/superslm/
// silu_lut.h) states the runtime path is division-free, composed only of
// RoundingDivideByPOT (a shift-and-round primitive, not `/`), multiply, add,
// and clamp/min. A source-text scan for `/` or `%` would be a brittle,
// easily-defeated proxy (a comment or an unrelated helper could trip it, or a
// real division could be hidden behind a macro), so this claim is left a
// structural, source-inspection claim rather than a fabricated executed cell
// -- named here, not silently assumed, per the brief's explicit "low
// priority, do not block" scoping. Poirot's code review is the structural
// check that reads src/silu_lut.cpp against this claim once Brunel greens it.
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// S2.4 golden hash (§10 item 6): a pinned SHA-256 over SiluSigmoidQ15's outputs
// across a canonical, safe-window input set. This is the cross-ISA/toolchain
// determinism gate — every platform in the CI matrix (and, via the run-only GPU
// package, every vendor) must reproduce this exact hash bit-for-bit. The table is
// the PINNED converter table (silu_lut_golden_table.h), never a std::exp
// regeneration, so the hash isolates the integer runtime path from libm. Every
// (code, m, e) below lies inside the width-sweep-proven safe window (shift = e+17
// in [-23, 23]; no int64 overflow), so the golden exercises the construction,
// never UB. e crosses both branches (shift>=0 and <0) and the saturation regime;
// m spans the [2^30, 2^31) mantissa domain incl. both ends and the real-corpus max.
// ---------------------------------------------------------------------------
static void TestSiluSigmoidQ15GoldenHashCrossPlatform() {
	using superslm::SiluSigmoidQ15;
	static constexpr int kEs[] = {6, 0, -15, -17, -19, -25, -32, -40};  // shift = e+17
	static constexpr int64_t kMs[] = {1073741824LL, 1073741831LL, 1500000000LL, 1898583166LL, 2147483647LL};

	std::vector<uint8_t> bytes;
	bytes.reserve(8u * 5u * 255u * 4u);
	for (int e : kEs) {
		for (int64_t m : kMs) {
			for (int code = -127; code <= 127; ++code) {
				const int32_t out = SiluSigmoidQ15(superslm_test::kSiluLutGoldenTable,
				                                   static_cast<int8_t>(code), m, e);
				const uint32_t u = static_cast<uint32_t>(out);  // little-endian, host-independent
				bytes.push_back(static_cast<uint8_t>(u & 0xFFu));
				bytes.push_back(static_cast<uint8_t>((u >> 8) & 0xFFu));
				bytes.push_back(static_cast<uint8_t>((u >> 16) & 0xFFu));
				bytes.push_back(static_cast<uint8_t>((u >> 24) & 0xFFu));
			}
		}
	}
	uint8_t digest[32];
	superslm::Sha256Hash(bytes.data(), bytes.size(), digest);
	const std::string hex = superslm::ToHex(digest);

	// PINNED golden. A mismatch is a cross-platform determinism break — OR an intended
	// construction change, in which case regenerate this constant deliberately.
	static const char* const kSiluLutGoldenHash =
	    "587576aba105a73a74b0dc75763259fb3e24ba170977caaf511440513b1fa5c6";  // pinned 2026-07-20 (MSVC x64)
	std::printf("S2.4 SiLU-LUT golden hash: %s (%zu inputs, %zu bytes)\n",
	            hex.c_str(), bytes.size() / 4, bytes.size());
	CHECK_MSG(hex == kSiluLutGoldenHash,
	          "SiluSigmoidQ15 golden hash %s != pinned %s (cross-platform determinism break, or the pin "
	          "needs regenerating for an intended construction change)",
	          hex.c_str(), kSiluLutGoldenHash);
}

// ---------------------------------------------------------------------------
// S2.5 matmul red suite (Claude/Curie/superslm-s2.5-matmul-test-design-2026-07-20.md;
// design: SuperSLM_matmul_subslot_design-2026-07-20.md). src/matmul.cpp is currently
// a red-phase stub: every one of GemmInt8AccumulateRow / GemmInt8Accumulate /
// NarrowAccumulatorToI32 unconditionally asserts(false) on entry
// (include/superslm/matmul.h), before examining any argument. Every cell below
// therefore fails red for the SAME reason today -- the implementation is absent --
// which is verified structurally (the stub body never reaches its arguments) and
// empirically (build+run; see the test-design record's red-confirmation section).
// Once Brunel builds the scalar reference (design S5), each cell fails red for its
// own reason (if any) until implemented correctly, then goes green.
//
// Goldens: tests/gen_matmul_fixtures.py, an arbitrary-precision (Python native int)
// oracle independent of any C++ implementation (design S5). Composition-regression
// goldens additionally cross the pinned Python intmath.py reference used by
// gen_intmath_fixtures.py.
// ---------------------------------------------------------------------------

static std::string GSelfPath;  // argv[0], captured in main() for the death-test probe below

// --- S12 dim 5 / the design's debug-assert-fire smoke check: a caller-contract
//     violation (in_channels == 0 -- below the architectural floor of 1, design S12
//     dim 4) must abort a debug build (the caller-ensures convention MaxAbsReduce/
//     IExpFromConstants already use). assert()'s abort() would take down this
//     ENTIRE process -- and every check after it -- if the violating call ran
//     in-process, so it runs in an isolated child process instead; the parent only
//     observes whether the child terminated abnormally. This is new infrastructure:
//     no death-test convention existed anywhere in this suite before S2.5 (confirmed
//     by inspection before authoring this cell) -- documented in the test-design
//     record, not silently introduced. Verified empirically (see the record) that
//     an MSVC assert() failure under build.bat's flags exits promptly with a
//     nonzero abnormal-termination code and does NOT block on a dialog. ---

static bool RunsCrashProbeAndCrashes(const char* probe_name, std::string* out_tail) {
	std::filesystem::path out_path =
	    std::filesystem::temp_directory_path() /
	    (std::string("superslm_crash_probe_") + probe_name + ".txt");
	std::error_code rm_ec;
	std::filesystem::remove(out_path, rm_ec);

	std::string cmd = "\"" + GSelfPath + "\" --crash-probe=" + probe_name +
	                   " > \"" + out_path.string() + "\" 2>&1";
	// system() on Windows invokes `cmd.exe /c <cmd>`; when <cmd> itself begins with a
	// quoted executable path, cmd.exe's first/last-quote-stripping parser misreads the
	// nested quotes (a well-known cmd.exe quirk) unless the WHOLE string is wrapped in
	// one more outer quote pair -- that outer pair is what cmd strips, leaving the
	// interior correctly quoted.
	std::string wrapped_cmd = "\"" + cmd + "\"";
	int rc = std::system(wrapped_cmd.c_str());

	std::string content;
	{
		std::ifstream f(out_path, std::ios::binary);
		content.assign(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
	}
	if (out_tail) *out_tail = content;
	std::filesystem::remove(out_path, rm_ec);

	// A clean, non-crashing exit (the probe function returned normally and printed
	// "PROBE DID NOT CRASH") is the one outcome that reads as "did not crash" --
	// every other outcome (SIGABRT/abort, an unhandled SEH exception, the CRT
	// assert handler's own nonzero exit) reads as crashed.
	return rc != 0;
}

// Dispatched from main() when argv[1] == "--crash-probe=<name>": runs exactly one
// contract-violating call in isolation. Returns normally (exit 0) ONLY if the call
// failed to crash -- a suite defect this cell exists to catch, not the expected
// outcome.
static int RunCrashProbe(const std::string& name) {
	if (name == "matmul_zero_in_channels") {
		int8_t act[1] = {0};
		int8_t wgt[1] = {0};
		int64_t out_acc[1] = {0};
		std::printf("crash-probe matmul_zero_in_channels: calling GemmInt8AccumulateRow "
		            "with in_channels=0 (below the architectural floor, design S12 dim 4)\n");
		std::fflush(stdout);
		GemmInt8AccumulateRow(act, wgt, /*in_channels=*/0, /*out_channels=*/1, out_acc);
		std::printf("PROBE DID NOT CRASH\n");
		return 0;
	}
	std::printf("PROBE DID NOT CRASH (unknown probe name: %s)\n", name.c_str());
	return 0;
}

static void TestGemmInt8AccumulateRowAssertsOnZeroInChannelsContractViolation() {
	std::string tail;
	bool crashed = RunsCrashProbeAndCrashes("matmul_zero_in_channels", &tail);
	CHECK_MSG(crashed,
	          "GemmInt8AccumulateRow(in_channels=0) must abort a debug build (contract "
	          "violation, design S12 dim 4/5) -- child output was: %s",
	          tail.c_str());
}

// --- S12 dim 1/6, S11 item 1: exactness against the arbitrary-precision oracle over
//     small vectors, int8 extremes on both operands, and the architectural floor
//     (in_channels == out_channels == 1). ---

static void TestGemmInt8AccumulateRowMatchesOracleAcrossRowCases() {
	using namespace superslm_test;
	for (size_t i = 0; i < kRowCasesCount; ++i) {
		const RowCase& c = kRowCases[i];
		std::vector<int64_t> out_acc(c.out_channels, 0);
		GemmInt8AccumulateRow(c.activations, c.weights, c.in_channels, c.out_channels,
		                       out_acc.data());
		for (size_t j = 0; j < c.out_channels; ++j) {
			CHECK_MSG(out_acc[j] == c.expected[j],
			          "%s: out_acc[%zu] == %lld, oracle expects %lld", c.label, j,
			          static_cast<long long>(out_acc[j]), static_cast<long long>(c.expected[j]));
		}
	}
}

// --- S12 dim 4: int32-safe regime on real S16-candidate hidden_size (1536, 960);
//     tail-length shape matrix -- real intermediate_size (8960, 2560, defensive,
//     block-aligned) AND deliberately non-block-aligned synthetic lengths (777,
//     4095 -- the load-bearing tail forcer). Also proves NarrowAccumulatorToI32
//     bit-exact against the independently-constructed golden int32 row. ---

static void TestGemmInt8AccumulateRowInt32SafeAndTailLengthCases() {
	using namespace superslm_test;
	for (size_t i = 0; i < kCompositionCasesCount; ++i) {
		const CompositionCase& c = kCompositionCases[i];
		std::vector<int64_t> wide(c.out_channels, 0);
		GemmInt8AccumulateRow(c.activations, c.weights, c.in_channels, c.out_channels,
		                       wide.data());
		for (size_t j = 0; j < c.out_channels; ++j) {
			CHECK_MSG(wide[j] == static_cast<int64_t>(c.golden_i32[j]),
			          "%s: wide[%zu] == %lld, oracle expects %d", c.label, j,
			          static_cast<long long>(wide[j]), c.golden_i32[j]);
		}
		std::vector<int32_t> narrowed(c.out_channels, 0);
		NarrowAccumulatorToI32(wide.data(), c.out_channels, narrowed.data());
		for (size_t j = 0; j < c.out_channels; ++j) {
			CHECK_MSG(narrowed[j] == c.golden_i32[j],
			          "%s: NarrowAccumulatorToI32 row[%zu] == %d, oracle expects %d", c.label,
			          j, narrowed[j], c.golden_i32[j]);
		}
	}
}

// --- S8/S11 item 2/S12 dim 4/6: the accumulator-overflow boundary, forced at the
//     ATTAINABLE transition (132,104 last-safe / 132,105 first-overflow) with the
//     jointly-attainable worst-case input, both sign directions -- NOT at the
//     conservative width-selection number 131,072 (which does not wrap under valid
//     inputs and would be silently too weak, design S11 item 2). Also carries the
//     deep-int64 and SIMD-saturation-hazard uniform cases (same case shape). For
//     every case that DOES overflow int32, the wrapped-int32 value is asserted to
//     DIFFER from the true sum (proving the boundary genuinely forces a wrap, not
//     merely a large-but-safe sum); for the width-selection documentation case and
//     every safe case, wrapped == true (proving no wrap there). The kernel itself is
//     asserted against the TRUE unwrapped int64 value only -- the wrapped value is
//     never a golden. ---

static void TestGemmInt8AccumulateRowUniformCasesExactAgainstOracle() {
	using namespace superslm_test;
	for (size_t i = 0; i < kUniformCasesCount; ++i) {
		const UniformCase& c = kUniformCases[i];
		std::vector<int8_t> acts(c.in_channels, c.act_value);
		std::vector<int8_t> wgts(c.in_channels, c.wgt_value);
		int64_t out_acc = 0;
		GemmInt8AccumulateRow(acts.data(), wgts.data(), c.in_channels, /*out_channels=*/1,
		                       &out_acc);
		CHECK_MSG(out_acc == c.expected,
		          "%s (in_channels=%zu): out_acc == %lld, oracle expects %lld (the TRUE, "
		          "unwrapped int64 sum)",
		          c.label, c.in_channels, static_cast<long long>(out_acc),
		          static_cast<long long>(c.expected));
		if (c.overflows_i32) {
			CHECK_MSG(c.wrapped_i32 != c.expected,
			          "%s: this case is marked overflows_i32 but its wrapped-int32 value "
			          "equals the true sum -- it does not actually force a wrap and is "
			          "silently too weak (the exact hole design S11 item 2 names)",
			          c.label);
		} else {
			CHECK_MSG(static_cast<int64_t>(c.wrapped_i32) == c.expected,
			          "%s: this case is marked int32-safe but a naive int32 accumulator "
			          "would produce a DIFFERENT value than the true sum -- the fixture's "
			          "own overflow classification is wrong",
			          c.label);
		}
	}
}

// --- S12 dim 4: SIMD alignment hazard -- distinct from tail length. Buffers
//     deliberately started at addresses not aligned to common SIMD widths
//     (offsets 1/3/5/7/15/31/63 bytes into a padded allocation), reusing an
//     already-exact composition case's data. ---

static void TestGemmInt8AccumulateRowUnalignedBufferPointers() {
	using namespace superslm_test;
	const CompositionCase& c = kCompositionCases[0];  // hidden_size_1536_qwen2_5_1_5b
	static constexpr size_t kOffsets[] = {1, 3, 5, 7, 15, 31, 63};
	for (size_t off : kOffsets) {
		std::vector<int8_t> act_buf(off + c.in_channels, 0);
		std::vector<int8_t> wgt_buf(off + c.out_channels * c.in_channels, 0);
		std::memcpy(act_buf.data() + off, c.activations, c.in_channels);
		std::memcpy(wgt_buf.data() + off, c.weights, c.out_channels * c.in_channels);

		std::vector<int64_t> out_acc(c.out_channels, 0);
		GemmInt8AccumulateRow(act_buf.data() + off, wgt_buf.data() + off, c.in_channels,
		                       c.out_channels, out_acc.data());
		for (size_t j = 0; j < c.out_channels; ++j) {
			CHECK_MSG(out_acc[j] == static_cast<int64_t>(c.golden_i32[j]),
			          "unaligned pointer offset %zu: out_acc[%zu] == %lld, oracle expects %d",
			          off, j, static_cast<long long>(out_acc[j]), c.golden_i32[j]);
		}
	}
}

// --- S12 dim 7 contract claim: "the reduction is exactly associative -- any lane
//     order/regrouping is safe" (design S4). Re-labels k alongside its paired
//     weight (a genuine reordering of the same dot product, not a different
//     computation) and asserts the bit-identical sum. ---

static void TestGemmInt8AccumulateRowOrderLaneRegroupingAssociativity() {
	using namespace superslm_test;
	int64_t original = 0;
	GemmInt8AccumulateRow(kPermActs, kPermWgts, kPermInChannels, /*out_channels=*/1, &original);
	CHECK_MSG(original == kPermExpected, "original order: out_acc == %lld, oracle expects %lld",
	          static_cast<long long>(original), static_cast<long long>(kPermExpected));

	std::vector<int8_t> reordered_acts(kPermInChannels);
	std::vector<int8_t> reordered_wgts(kPermInChannels);
	for (size_t k = 0; k < kPermInChannels; ++k) {
		reordered_acts[k] = kPermActs[kPermIndex[k]];
		reordered_wgts[k] = kPermWgts[kPermIndex[k]];
	}
	int64_t reordered = 0;
	GemmInt8AccumulateRow(reordered_acts.data(), reordered_wgts.data(), kPermInChannels,
	                       /*out_channels=*/1, &reordered);
	CHECK_MSG(reordered == kPermExpected,
	          "lane-regrouped order: out_acc == %lld, oracle expects %lld (same value as the "
	          "original order -- the reduction must be order-independent)",
	          static_cast<long long>(reordered), static_cast<long long>(kPermExpected));
}

// --- S12 dim 7: "no cross-row reduction -- rows are independent" (design S3), the
//     discrimination cell (Mendeleev gap 4). Mutating row 2's activations must leave
//     every OTHER row's output byte-unchanged; GemmInt8Accumulate over num_tokens
//     rows must be bit-identical to num_tokens independent GemmInt8AccumulateRow
//     calls stacked row-major [num_tokens, out_channels] (design S3). ---

static void TestGemmInt8AccumulateRowIndependenceAndMultiRowStackingEquivalence() {
	using namespace superslm_test;
	const MultiRowCase& base = kMultiRowCases[0];      // row_independence_base
	const MultiRowCase& mutated = kMultiRowCases[1];    // row_independence_row2_mutated
	CHECK(base.num_tokens == mutated.num_tokens);
	CHECK(base.out_channels == mutated.out_channels);

	std::vector<int64_t> out_base(base.num_tokens * base.out_channels, 0);
	GemmInt8Accumulate(base.activations, base.weights, base.num_tokens, base.in_channels,
	                    base.out_channels, out_base.data());
	for (size_t i = 0; i < base.num_tokens * base.out_channels; ++i) {
		CHECK_MSG(out_base[i] == base.expected[i], "%s: out_acc[%zu] == %lld, oracle expects %lld",
		          base.label, i, static_cast<long long>(out_base[i]),
		          static_cast<long long>(base.expected[i]));
	}

	std::vector<int64_t> out_mutated(mutated.num_tokens * mutated.out_channels, 0);
	GemmInt8Accumulate(mutated.activations, mutated.weights, mutated.num_tokens,
	                    mutated.in_channels, mutated.out_channels, out_mutated.data());
	for (size_t i = 0; i < mutated.num_tokens * mutated.out_channels; ++i) {
		CHECK_MSG(out_mutated[i] == mutated.expected[i],
		          "%s: out_acc[%zu] == %lld, oracle expects %lld", mutated.label, i,
		          static_cast<long long>(out_mutated[i]), static_cast<long long>(mutated.expected[i]));
	}

	// Discrimination: rows 0,1,3 must be byte-identical between base and mutated
	// (only row 2's activations differ); row 2 must differ.
	for (size_t t = 0; t < base.num_tokens; ++t) {
		bool row_matches = true;
		for (size_t j = 0; j < base.out_channels; ++j) {
			if (out_base[t * base.out_channels + j] != out_mutated[t * base.out_channels + j]) {
				row_matches = false;
				break;
			}
		}
		if (t == 2) {
			CHECK_MSG(!row_matches,
			          "row 2 (the mutated row) must differ between base and mutated calls -- "
			          "identical output would mean the mutation had no effect on the row it "
			          "targeted");
		} else {
			CHECK_MSG(row_matches,
			          "row %zu must be byte-identical between base and mutated calls -- a "
			          "difference here means row 2's mutation leaked into another row (cross-row "
			          "state)",
			          t);
		}
	}

	// Stacking equivalence: GemmInt8Accumulate(num_tokens) == num_tokens independent
	// GemmInt8AccumulateRow calls, row-major [num_tokens, out_channels] (design S3).
	std::vector<int64_t> stacked(base.num_tokens * base.out_channels, 0);
	for (size_t t = 0; t < base.num_tokens; ++t) {
		GemmInt8AccumulateRow(base.activations + t * base.in_channels, base.weights,
		                       base.in_channels, base.out_channels,
		                       stacked.data() + t * base.out_channels);
	}
	for (size_t i = 0; i < base.num_tokens * base.out_channels; ++i) {
		CHECK_MSG(stacked[i] == out_base[i],
		          "%s: stacked single-row call [%zu] == %lld, GemmInt8Accumulate produced %lld "
		          "-- multi-row must be bit-identical to independently-stacked single-row calls",
		          base.label, i, static_cast<long long>(stacked[i]), static_cast<long long>(out_base[i]));
	}
}

// --- S12 dim 1: warm-object -- the SAME loaded weight buffer read correctly across
//     many independent activation rows (simulating many tokens/resets against a
//     loaded artifact), not only a fresh-load test. ---

static void TestGemmInt8AccumulateRowWarmObjectManyTokensAgainstSameWeights() {
	using namespace superslm_test;
	for (size_t r = 0; r < kWarmObjectRowCount; ++r) {
		std::vector<int64_t> out_acc(kWarmObjectOutChannels, 0);
		GemmInt8AccumulateRow(kWarmObjectActRows[r], kWarmObjectWeights, kWarmObjectInChannels,
		                       kWarmObjectOutChannels, out_acc.data());
		for (size_t j = 0; j < kWarmObjectOutChannels; ++j) {
			CHECK_MSG(out_acc[j] == kWarmObjectExpectedRows[r][j],
			          "warm-object row %zu: out_acc[%zu] == %lld, oracle expects %lld", r, j,
			          static_cast<long long>(out_acc[j]),
			          static_cast<long long>(kWarmObjectExpectedRows[r][j]));
		}
	}
}

// --- S12 dim 1: scratch-buffer reuse across a shape change (prefill then decode, or
//     decode after a shape change) -- the out_acc buffer is reused WITHOUT clearing
//     between calls, and the used portion of a smaller subsequent call must be fully
//     overwritten with fresh data, never stale bytes from a larger prior call (the
//     v2.1-v2.2 workspace-reuse stride bug the catalog names directly). ---

static void TestGemmInt8AccumulateScratchBufferNoStaleByteCarryoverAcrossShapeChange() {
	using namespace superslm_test;
	const MultiRowCase& call1 = kMultiRowCases[2];  // scratch_reuse_call1_num_tokens_6
	const MultiRowCase& call2 = kMultiRowCases[3];  // scratch_reuse_call2_num_tokens_2
	CHECK(call1.in_channels == call2.in_channels);
	CHECK(call1.out_channels == call2.out_channels);
	CHECK(call1.num_tokens > call2.num_tokens);

	// One buffer sized for the LARGER call, reused (not reallocated, not cleared)
	// for the smaller call.
	std::vector<int64_t> out_acc(call1.num_tokens * call1.out_channels, 0);

	GemmInt8Accumulate(call1.activations, call1.weights, call1.num_tokens, call1.in_channels,
	                    call1.out_channels, out_acc.data());
	for (size_t i = 0; i < call1.num_tokens * call1.out_channels; ++i) {
		CHECK_MSG(out_acc[i] == call1.expected[i], "%s: out_acc[%zu] == %lld, oracle expects %lld",
		          call1.label, i, static_cast<long long>(out_acc[i]),
		          static_cast<long long>(call1.expected[i]));
	}

	// Reuse the SAME buffer for the smaller call -- no clear, no reallocation.
	GemmInt8Accumulate(call2.activations, call2.weights, call2.num_tokens, call2.in_channels,
	                    call2.out_channels, out_acc.data());
	const size_t used = call2.num_tokens * call2.out_channels;
	for (size_t i = 0; i < used; ++i) {
		CHECK_MSG(out_acc[i] == call2.expected[i],
		          "%s: out_acc[%zu] == %lld after reuse, oracle expects %lld -- a mismatch here "
		          "(especially one equal to call1's earlier value) is stale-byte carryover from "
		          "the larger prior call",
		          call2.label, i, static_cast<long long>(out_acc[i]),
		          static_cast<long long>(call2.expected[i]));
		CHECK_MSG(out_acc[i] != call1.expected[i] || call1.expected[i] == call2.expected[i],
		          "%s: out_acc[%zu] == %lld still equals call1's value for this slot -- this is "
		          "the specific stale-byte-carryover signature (fixtures were constructed so "
		          "call1 and call2 differ at rows 0/1)",
		          call2.label, i, static_cast<long long>(out_acc[i]));
	}
}

// --- S12 dim 3: 8 reader threads x >=10,000 calls each against the SAME loaded
//     weight tensor, every threaded result asserted byte-identical to a
//     single-threaded baseline (mirrors TestSiluSigmoidQ15ConcurrentReadsMatchSingleThreaded's
//     shape exactly). Runs as a plain stress under build.bat; the project's
//     ThreadSanitizer leg is the separate CI clang build, not this binary -- no
//     sanitizer is invoked here. CHECK/CHECK_MSG are not thread-safe, so each thread
//     accumulates a local mismatch count and only the joining main thread calls
//     CHECK. ---

static void TestGemmInt8AccumulateRowConcurrentReadsMatchSingleThreaded() {
	using namespace superslm_test;
	constexpr int kThreads = 8;
	constexpr size_t kSweeps = 501;  // 501 * 20 rows == 10,020 calls/thread >= 10,000
	const size_t rows_per_sweep = kWarmObjectRowCount;
	const size_t calls_per_thread = kSweeps * rows_per_sweep;

	std::vector<std::vector<int64_t>> baseline(rows_per_sweep);
	for (size_t r = 0; r < rows_per_sweep; ++r) {
		baseline[r].assign(kWarmObjectOutChannels, 0);
		GemmInt8AccumulateRow(kWarmObjectActRows[r], kWarmObjectWeights, kWarmObjectInChannels,
		                       kWarmObjectOutChannels, baseline[r].data());
	}

	std::vector<int> per_thread_mismatches(kThreads, 0);
	std::vector<std::thread> threads;
	threads.reserve(kThreads);
	for (int t = 0; t < kThreads; ++t) {
		threads.emplace_back([&baseline, &per_thread_mismatches, t, rows_per_sweep]() {
			int mismatches = 0;
			for (size_t sweep = 0; sweep < kSweeps; ++sweep) {
				for (size_t r = 0; r < rows_per_sweep; ++r) {
					std::vector<int64_t> out_acc(kWarmObjectOutChannels, 0);
					GemmInt8AccumulateRow(kWarmObjectActRows[r], kWarmObjectWeights,
					                       kWarmObjectInChannels, kWarmObjectOutChannels,
					                       out_acc.data());
					if (out_acc != baseline[r]) ++mismatches;
				}
			}
			per_thread_mismatches[static_cast<size_t>(t)] = mismatches;
		});
	}
	for (auto& th : threads) th.join();

	int total_mismatches = 0;
	for (int m : per_thread_mismatches) total_mismatches += m;
	CHECK_MSG(total_mismatches == 0,
	          "%d of %d concurrent reads (8 threads x %zu calls) did not match the "
	          "single-threaded baseline",
	          total_mismatches, kThreads * static_cast<int>(calls_per_thread), calls_per_thread);
}

// --- S12 dim 8, S11 item 3: composition regression. NarrowAccumulatorToI32's output,
//     fed into the already-shipped MaxAbsReduce/NormalizeScale/DynamicScaleReciprocal/
//     RequantTokenCode chain, must be bit-identical to feeding an INDEPENDENTLY-
//     CONSTRUCTED equivalent int32_t[] row through the SAME already-certified chain
//     directly -- proving the matmul kernel introduces no divergence at the seam.
//     Cross-checked a second way against the pinned Python intmath.py pipeline
//     oracle (the same reference gen_intmath_fixtures.py's pipeline cases use). ---

static void TestGemmInt8AccumulateComposesWithShippedRequantChain() {
	using namespace superslm_test;
	for (size_t i = 0; i < kCompositionCasesCount; ++i) {
		const CompositionCase& c = kCompositionCases[i];

		// Path A: real matmul -> narrow -> chain.
		std::vector<int64_t> wide(c.out_channels, 0);
		GemmInt8AccumulateRow(c.activations, c.weights, c.in_channels, c.out_channels,
		                       wide.data());
		std::vector<int32_t> narrowed(c.out_channels, 0);
		NarrowAccumulatorToI32(wide.data(), c.out_channels, narrowed.data());

		int64_t d_prime_a = MaxAbsReduce(narrowed.data(), c.out_channels);
		NormalizedScale ns_a = NormalizeScale(d_prime_a);
		int64_t r_a = DynamicScaleReciprocal(ns_a.dn);
		std::vector<int8_t> codes_a(c.out_channels);
		for (size_t j = 0; j < c.out_channels; ++j) {
			codes_a[j] = RequantTokenCode(narrowed[j], r_a, ns_a.s);
		}

		// Path B: the SAME chain, fed the independently-constructed golden int32 row
		// directly -- never derived from calling NarrowAccumulatorToI32.
		int64_t d_prime_b = MaxAbsReduce(c.golden_i32, c.out_channels);
		NormalizedScale ns_b = NormalizeScale(d_prime_b);
		int64_t r_b = DynamicScaleReciprocal(ns_b.dn);
		std::vector<int8_t> codes_b(c.out_channels);
		for (size_t j = 0; j < c.out_channels; ++j) {
			codes_b[j] = RequantTokenCode(c.golden_i32[j], r_b, ns_b.s);
		}

		CHECK_MSG(d_prime_a == d_prime_b && d_prime_a == c.expected_d_prime,
		          "%s: composition seam d_prime mismatch (matmul-path %lld, golden-path %lld, "
		          "oracle %lld)",
		          c.label, static_cast<long long>(d_prime_a), static_cast<long long>(d_prime_b),
		          static_cast<long long>(c.expected_d_prime));
		CHECK_MSG(ns_a.dn == ns_b.dn && ns_a.dn == c.expected_dn && ns_a.s == ns_b.s &&
		              ns_a.s == c.expected_s,
		          "%s: composition seam NormalizeScale mismatch", c.label);
		CHECK_MSG(r_a == r_b && r_a == c.expected_r,
		          "%s: composition seam DynamicScaleReciprocal mismatch (matmul-path %lld, "
		          "golden-path %lld, oracle %lld)",
		          c.label, static_cast<long long>(r_a), static_cast<long long>(r_b),
		          static_cast<long long>(c.expected_r));
		for (size_t j = 0; j < c.out_channels; ++j) {
			CHECK_MSG(codes_a[j] == codes_b[j],
			          "%s: composition seam codes[%zu] diverge -- matmul-path %d, golden-path %d "
			          "(the matmul kernel perturbs the already-certified C19-C22 chain)",
			          c.label, j, codes_a[j], codes_b[j]);
			CHECK_MSG(codes_a[j] == c.expected_codes[j],
			          "%s: codes[%zu] == %d, independent Python intmath.py pipeline oracle "
			          "expects %d",
			          c.label, j, codes_a[j], c.expected_codes[j]);
		}
	}
}

// --- S12 dim 10, S11 item 5: op-level parity -- dequantized int8 output vs a
//     float32 reference matmul of the dequantized inputs (NOT raw-accumulator vs
//     float32; the int64 accumulate is exact ground truth, proven by item 1). The
//     reference is the UNSCALED, code-level matmul (the C24/C25 weight-scale fold is
//     out of scope, design S9): both operands are used as their raw int8 numeric
//     value in float32, matching this component's own scope boundary. The
//     TOLERANCE is an owed build-time measurement (design S11 item 5) -- this
//     harness computes and prints the error statistics; the CHECK below is a
//     generous smoke bound catching a gross divergence, not the acceptance
//     tolerance, which is Brunel's to measure and pin. ---

static void TestGemmInt8AccumulateOpLevelDequantParityVsFloat32Reference() {
	using namespace superslm_test;
	const CompositionCase& c = kCompositionCases[0];  // hidden_size_1536_qwen2_5_1_5b

	std::vector<int64_t> wide(c.out_channels, 0);
	GemmInt8AccumulateRow(c.activations, c.weights, c.in_channels, c.out_channels, wide.data());
	std::vector<int32_t> narrowed(c.out_channels, 0);
	NarrowAccumulatorToI32(wide.data(), c.out_channels, narrowed.data());

	int64_t d_prime = MaxAbsReduce(narrowed.data(), c.out_channels);
	NormalizedScale ns = NormalizeScale(d_prime);
	int64_t r = DynamicScaleReciprocal(ns.dn);
	std::vector<int8_t> codes(c.out_channels);
	for (size_t j = 0; j < c.out_channels; ++j) codes[j] = RequantTokenCode(narrowed[j], r, ns.s);

	const float output_scale = static_cast<float>(d_prime) / 127.0f;
	double max_abs_error = 0.0;
	double sum_abs_error = 0.0;
	for (size_t j = 0; j < c.out_channels; ++j) {
		float float_ref = 0.0f;  // unscaled, code-level float32 matmul (design S9/S11 item 5)
		for (size_t k = 0; k < c.in_channels; ++k) {
			float_ref += static_cast<float>(c.activations[k]) *
			              static_cast<float>(c.weights[j * c.in_channels + k]);
		}
		const float dequant_output = static_cast<float>(codes[j]) * output_scale;
		const double err = std::fabs(static_cast<double>(dequant_output) - static_cast<double>(float_ref));
		max_abs_error = std::max(max_abs_error, err);
		sum_abs_error += err;
	}
	const double mean_abs_error = sum_abs_error / static_cast<double>(c.out_channels);
	std::printf(
	    "S2.5 op-level dequant parity (%s): max |error| = %.6f, mean |error| = %.6f, "
	    "output_scale = %.6f (D'=%lld) -- tolerance is an owed build-time measurement, not "
	    "asserted here (design S11 item 5)\n",
	    c.label, max_abs_error, mean_abs_error, static_cast<double>(output_scale),
	    static_cast<long long>(d_prime));
	// Provisional smoke bound only (catches a gross divergence, e.g. a scale or sign
	// error): the true magnitude a per-token dynamic-scale requant can be off by is
	// bounded by roughly one quantization step, output_scale/2, times some small
	// constant for cross-channel accumulation slack -- generous by design.
	const double smoke_bound = static_cast<double>(output_scale) * 4.0 + 1.0;
	CHECK_MSG(max_abs_error <= smoke_bound,
	          "%s: max |dequant - float32 reference| = %.6f exceeds the provisional smoke bound "
	          "%.6f -- this is not the acceptance tolerance (owed, build-time measured, design "
	          "S11 item 5), but this magnitude suggests a gross defect (scale or sign error), "
	          "not ordinary quantization error",
	          c.label, max_abs_error, smoke_bound);
}

// --- S11 item 4 addendum -- closes Poirot's S2.5 review finding: the normative
//     scalar reference (design S5) had zero committed exercise on x64 (DotRow
//     dispatches unconditionally to SSE2 there, so DotRowScalarRef was reachable
//     only from uncommitted scratch). This cell drives DotRowScalarRef and the
//     shipping SSE2 path (GemmInt8AccumulateRow, which resolves to DotRowSse2 on
//     this platform) as two INDEPENDENT calls against the SAME inputs, both
//     asserted bit-for-bit equal to the arbitrary-precision oracle -- discharging
//     design S11 item 4's "SIMD == scalar bit-equal" claim directly, not only
//     transitively through the row/composition/uniform cells above. Spread: the
//     full existing row/composition/uniform fixture corpus (small random, int8
//     extremes, the architectural floor, real hidden_size/intermediate_size
//     shapes, the non-block-aligned lengths 777/4095, the overflow boundary,
//     deep int64 exactness, the SIMD-saturation hazard) PLUS a dedicated
//     in_channels=1..80 tail-length sweep with both sign extremes on both
//     operands at every length (tests/gen_matmul_fixtures.py, kTailSweepCases). ---

static void TestDotRowScalarRefMatchesShippingSse2PathAndOracle() {
	using namespace superslm_test;

	auto check_one = [](const char* label, const int8_t* acts, const int8_t* wgts,
	                     size_t in_channels, int64_t oracle) {
		const int64_t scalar = DotRowScalarRef(acts, wgts, in_channels);
		int64_t simd = 0;
		GemmInt8AccumulateRow(acts, wgts, in_channels, /*out_channels=*/1, &simd);
		CHECK_MSG(scalar == oracle,
		          "%s (in_channels=%zu): DotRowScalarRef == %lld, oracle expects %lld", label,
		          in_channels, static_cast<long long>(scalar), static_cast<long long>(oracle));
		CHECK_MSG(simd == oracle,
		          "%s (in_channels=%zu): shipping SSE2 path (GemmInt8AccumulateRow) == %lld, "
		          "oracle expects %lld",
		          label, in_channels, static_cast<long long>(simd), static_cast<long long>(oracle));
		CHECK_MSG(scalar == simd,
		          "%s (in_channels=%zu): scalar reference (%lld) != shipping SSE2 path (%lld) -- "
		          "design S11 item 4's SIMD == scalar bit-equal claim violated",
		          label, in_channels, static_cast<long long>(scalar), static_cast<long long>(simd));
	};

	// Existing row corpus -- small random, int8 extremes crossed, architectural floor.
	for (size_t i = 0; i < kRowCasesCount; ++i) {
		const RowCase& c = kRowCases[i];
		for (size_t j = 0; j < c.out_channels; ++j) {
			check_one(c.label, c.activations, c.weights + j * c.in_channels, c.in_channels,
			          c.expected[j]);
		}
	}

	// Existing composition corpus -- real hidden_size/intermediate_size shapes,
	// including the load-bearing non-block-aligned lengths 777 and 4095.
	for (size_t i = 0; i < kCompositionCasesCount; ++i) {
		const CompositionCase& c = kCompositionCases[i];
		for (size_t j = 0; j < c.out_channels; ++j) {
			check_one(c.label, c.activations, c.weights + j * c.in_channels, c.in_channels,
			          static_cast<int64_t>(c.golden_i32[j]));
		}
	}

	// Existing uniform corpus -- overflow boundary (both attainable-transition
	// lengths, both sign directions), deep int64 exactness, SIMD-saturation hazard.
	for (size_t i = 0; i < kUniformCasesCount; ++i) {
		const UniformCase& c = kUniformCases[i];
		std::vector<int8_t> acts(c.in_channels, c.act_value);
		std::vector<int8_t> wgts(c.in_channels, c.wgt_value);
		check_one(c.label, acts.data(), wgts.data(), c.in_channels, c.expected);
	}

	// Dedicated tail-length sweep, in_channels 1..80, both sign extremes on both
	// operands at every length (the fixture this cell adds).
	for (size_t i = 0; i < kTailSweepCasesCount; ++i) {
		const RowCase& c = kTailSweepCases[i];
		CHECK(c.out_channels == 1);
		check_one(c.label, c.activations, c.weights, c.in_channels, c.expected[0]);
	}
}

int main(int argc, char** argv) {
	GSelfPath = (argc > 0 && argv[0] != nullptr) ? argv[0] : "superslm_tests";
	if (argc > 1) {
		const std::string arg1 = argv[1];
		const std::string prefix = "--crash-probe=";
		if (arg1.rfind(prefix, 0) == 0) {
			return RunCrashProbe(arg1.substr(prefix.size()));
		}
	}
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

	// --- Curie's S2.0a WGT1/BIA1/ROP1 tensor-manifest hostile-input suite
	//     (red-first; src/model.cpp is currently a stub). ---
	TestWgtMinimalManifestParsesAndRoundTrips();
	TestBiaMinimalManifestParsesAndRoundTrips();
	TestRopMinimalManifestParsesAndRoundTrips();
	TestManifestRejectsSectionTooShort();
	TestManifestRejectsBadMagicWgt();
	TestManifestRejectsBadMagicBia();
	TestManifestRejectsBadMagicRop();
	TestManifestRejectsUnsupportedVersion();
	TestManifestRejectsTooManyTensors();
	TestManifestRejectsManifestOutOfBoundsTruncatedDescriptors();
	TestManifestRejectsManifestOutOfBoundsTruncatedNameBlob();
	TestManifestRejectsBadTensorNameOutOfRange();
	TestManifestRejectsEmptyTensorName();
	TestManifestRejectsDuplicateTensorName();
	TestManifestRejectsBadTensorRankZero();
	TestManifestRejectsBadTensorRankTooLarge();
	TestManifestRejectsBadTensorShapeZeroDim();
	TestManifestRejectsBadTensorShapeNonzeroPastRank();
	TestManifestRejectsShapeCountMismatch();
	TestManifestRejectsTensorOutOfBoundsDataExceedsSection();
	TestManifestRejectsTensorOverlap();
	TestManifestRejectsBadDescriptorReserved();
	TestManifestRejectsDataOffBelowDataRegion();
	TestManifestRejectsTensorMisalignedBia();
	TestManifestRejectsTensorMisalignedRop();
	TestManifestRejectsElemCountTimesElementSizeOverflows32BitBia();
	TestManifestRejectsElemCountTimesElementSizeOverflows32BitRop();
	TestManifestRejectsShapeProductOverflows32BitTensorOutOfBounds();

	// --- Curie's S2.0b KVC1 keyed-constant sub-parse hostile-input suite
	//     (red-first; src/model.cpp's SslmKeyedConstants::Parse is currently a
	//     stub). ---
	TestCompositionConstantsMinimalKvc1ParsesAndRoundTrips();
	TestKvLandingReciprocalsMinimalKvc1ParsesAndRoundTrips();
	TestKvc1RejectsSectionTooShort();
	TestKvc1RejectsBadMagic();
	TestKvc1RejectsUnsupportedVersion();
	TestKvc1RejectsTooManyEntries();
	TestKvc1RejectsBadReserved();
	TestKvc1RejectsOutOfBoundsTruncatedDescriptors();
	TestKvc1RejectsOutOfBoundsTruncatedValues();
	TestKvc1RejectsOutOfBoundsTruncatedNameBlob();
	TestKvc1RejectsNameBlobLenSumOverflow32Bit();
	TestKvc1RejectsBadEntryNameOutOfRange();
	TestKvc1RejectsEmptyEntryName();
	TestKvc1RejectsDuplicateEntryName();
	TestKvc1RejectsValueWordsOutOfRange();
	TestKvc1RejectsValueWordsWrongForTypeCompositionDeclaresThree();
	TestKvc1RejectsValueWordsWrongForTypeReciprocalsDeclaresTwo();

	// --- Curie's S2.0b CFG1 config sub-parse hostile-input suite (red-first;
	//     src/model.cpp's ParseConfig is currently a stub). ---
	TestMinimalCfg1ParsesAndMatchesEveryField();
	TestCfg1RejectsSizeTooShort();
	TestCfg1RejectsSizeTooLong();
	TestCfg1RejectsBadMagic();
	TestCfg1RejectsUnsupportedVersion();
	TestCfg1RejectsZeroHiddenSize();
	TestCfg1RejectsZeroNumHiddenLayers();
	TestCfg1RejectsZeroNumAttentionHeads();
	TestCfg1RejectsZeroNumKeyValueHeads();
	TestCfg1RejectsZeroHeadDim();
	TestCfg1RejectsZeroIntermediateSize();
	TestCfg1RejectsZeroVocabSize();
	TestCfg1RejectsZeroContextCap();
	TestCfg1RejectsZeroKvBlockSize();
	TestCfg1RejectsBadKvPrecision();
	TestCfg1RejectsBadConfigBool();
	TestCfg1RejectsBadConfigReserved();

	// --- Curie's S2.0b WeightScales manifest-reuse oracle (WSC1 routes through
	//     the already-certified S2.0a SslmTensorManifest::Parse — expected
	//     green at authoring time). ---
	TestWeightScalesMinimalManifestParsesAndRoundTrips();
	TestWeightScalesRejectsWrongMagicDiscriminatesPerType();

	// --- Curie's S2.1 intmath red suite (red-first; src/intmath.cpp is
	//     currently deliberately-wrong stub bodies). ---
	TestC2SaturatingRoundingDoublingHighMul();
	TestC1C3RoundingDivideByPOT();
	TestMultiplyByQuantizedMultiplier();
	TestClz64();
	TestMaxAbsReduce();
	TestNormalizeScale();
	TestDynamicScaleReciprocalNamed();
	TestDynamicScaleReciprocalDenseSample();
	TestRequantTokenCode();
	TestIntmathPipelineComposition();

	// --- Curie's S2.2 nonlinear scalar primitives red suite (red-first;
	//     src/intmath.cpp's ISqrt/ISqrtTrace/ShiftByMax/IExpFromConstants
	//     bodies are currently deliberately-wrong stub sentinels). ---
	TestISqrt();
	TestISqrtTrace();
	TestShiftByMax();
	TestIExpFromConstants();
	TestIExpFromConstantsClipClampsIdenticallyAcrossFamily();

	// --- Curie's S2.3 RopeApplyPair red suite (red-first; src/intmath.cpp's
	//     RopeApplyPair body is currently the deliberately-wrong stub sentinel
	//     {-1, -1}). ---
	TestRopeApplyPair();
	TestRopeApplyPairIdentityIsExact();
	TestRopeApplyPairQuarterTurnIsExact();
	TestRopeApplyPairWideInputExceedsInt32Range();
	TestRopeApplyPairTieRoundsAwayFromZero();

	// --- Curie's S2.4 SiLU sigmoid-LUT red suite (red-first; src/model.cpp's
	//     ParseSigmoidLut and src/silu_lut.cpp's SiluSigmoidQ15 are currently
	//     deliberately-wrong stubs). ---
	TestMinimalSil1ParsesAndReadsBackAllNodes();
	TestSil1WarmObjectRepeatedReadsShowNoDrift();
	TestSil1RoundTripReencodeMatchesOriginalBytes();
	TestSil1RejectsSizeTooShort();
	TestSil1RejectsSizeTooLong();
	TestSil1RejectsBadMagic();
	TestSil1RejectsUnsupportedVersion();
	TestSil1RejectsBadEntryCount();
	TestSil1RejectsBadReserved();
	TestArtifactRejectsPreSigmoidLutV1FormatUnderCurrentLoader();
	TestArtifactAcceptsV2ArtifactCarryingValidSigmoidLutSection();
	TestSiluSigmoidQ15SaturatesHighDomainShiftNegativeBranch();
	TestSiluSigmoidQ15SaturatesLowDomainShiftNegativeBranch();
	TestSiluSigmoidQ15SaturatesHighDomainShiftNonNegativeBranch();
	TestSiluSigmoidQ15SaturatesLowDomainShiftNonNegativeBranch();
	TestSiluSigmoidQ15RealCorpusDeepSaturationHighCode();
	TestSiluSigmoidQ15RealCorpusDeepSaturationLowCode();
	TestSiluSigmoidQ15InteriorInterpolationFracZero();
	TestSiluSigmoidQ15InteriorInterpolationFracMax();
	TestSiluSigmoidQ15ShiftZeroBranchCodeZeroReachesMidpointExactly();
	TestSiluSigmoidQ15PositiveShiftBranchCodeZeroReachesMidpointExactly();
	TestSiluSigmoidQ15OpLevelParityWithinOneUlpOnRealVectors();
	TestSiluSigmoidQ15DownstreamInt8AgreementReproducesLaplaceBand();
	TestSiluSigmoidQ15ConcurrentReadsMatchSingleThreaded();
	TestSiluSigmoidQ15GoldenHashCrossPlatform();

	// --- Curie's S2.5 matmul red suite (red-first; src/matmul.cpp's
	//     GemmInt8AccumulateRow/GemmInt8Accumulate/NarrowAccumulatorToI32 are
	//     currently stubs that unconditionally assert(false) on entry). Every
	//     cell below that calls into matmul.cpp directly will abort this process
	//     until Brunel implements the scalar reference -- this is the expected,
	//     documented RED state (see the test-design record), not a suite defect. ---
	TestGemmInt8AccumulateRowAssertsOnZeroInChannelsContractViolation();
	TestGemmInt8AccumulateRowMatchesOracleAcrossRowCases();
	TestGemmInt8AccumulateRowInt32SafeAndTailLengthCases();
	TestGemmInt8AccumulateRowUniformCasesExactAgainstOracle();
	TestGemmInt8AccumulateRowUnalignedBufferPointers();
	TestGemmInt8AccumulateRowOrderLaneRegroupingAssociativity();
	TestGemmInt8AccumulateRowIndependenceAndMultiRowStackingEquivalence();
	TestGemmInt8AccumulateRowWarmObjectManyTokensAgainstSameWeights();
	TestGemmInt8AccumulateScratchBufferNoStaleByteCarryoverAcrossShapeChange();
	TestGemmInt8AccumulateRowConcurrentReadsMatchSingleThreaded();
	TestGemmInt8AccumulateComposesWithShippedRequantChain();
	TestGemmInt8AccumulateOpLevelDequantParityVsFloat32Reference();
	TestDotRowScalarRefMatchesShippingSse2PathAndOracle();

	std::printf("superslm tests: %d checks, %d failures\n", GChecks, GFailures);
	return GFailures == 0 ? 0 : 1;
}

// tests/areas/loader.cpp -- T-1574 test suite split, Stage 3, amended during
// Stage 6 prep.
// Area #1 (Claude/Plans/SuperSLM_TestSuiteSplit_Plan.md §4): tests calling
// through src/artifact.cpp's and src/sha256.cpp's public contract
// (SslmArtifact::OpenFromMemory/OpenFromFile, DtypeSize, IsKnownSectionType,
// Sha256Hash, ToHex). Extracted verbatim from tests/test_main.cpp (order
// keys 0-36, S0 skeleton baseline + Curie's S0 loader red suite; order key
// 336, the S-HARDEN-8 generic section-descriptor-reserved-field cell; and
// order key 238, added by this amendment).
//
// **Amendment (order key 238).** `TestArtifactRejectsConfigOnlyV2MissingSigmoidLut`
// sits physically beside the SIL1 sub-parse block (tests/areas/tensor_manifest.cpp's
// own Stage-6-prep amendment, see that file's header comment) but calls only
// `SslmArtifact::OpenFromMemory` -- no `ParseSigmoidLut` call at all -- so
// §3.1's rule places it here, in the loader's own area, not with the SIL1
// sub-parse content it happens to sit beside. Missed at Stage 3's own commit
// for the same physical-proximity reason the SIL1 block was missed at
// Stage 5; closed together.
//
// The plan's own §4 area table lists `FoundSection` as this area's owned
// local fixture. Reading the actual source at this stage finds no such
// symbol used by any test in this area -- the only `FoundSection` in the
// tree is `independent_reader::FoundSection`, a from-scratch reimplementation
// used exclusively by `TestIndependentReaderRecoversBia1AndBiasQbMatchingModelView`/
// `TestIndependentReaderFlagsHandCorruptedBia1TensorWithoutGoingThroughLoad`
// (both drive `SslmModel::Load`, candidate area #4 model_load.cpp, not this
// one). Per §3.1's rule (area is the contract actually called, not a
// starting-map label) and §4's own text ("a test filed in the wrong area is
// a readability defect, never a correctness one"), this area is extracted
// with no owned fixture; `independent_reader::FoundSection` is left where it
// is for now and will be routed to model_load.cpp at Stage 10, noted there.

#include "superslm/artifact.h"
#include "superslm/sha256.h"
#include "sslm_fixtures.h"
#include "sslm_sil1_hostile_fixtures.h"

#include "support/test_harness.h"
#include "support/test_registry.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

using namespace superslm;
using namespace superslm_test;

// --- S0 skeleton baseline -------------------------------------------------

SSLM_TEST(TestSha256KnownVectors, 0) {
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

SSLM_TEST(TestDtypeSizes, 1) {
	CHECK(DtypeSize(static_cast<uint32_t>(SslmDtype::Raw)) == 1);
	CHECK(DtypeSize(static_cast<uint32_t>(SslmDtype::Int8)) == 1);
	CHECK(DtypeSize(static_cast<uint32_t>(SslmDtype::Int32)) == 4);
	CHECK(DtypeSize(static_cast<uint32_t>(SslmDtype::Int64)) == 8);
	CHECK(DtypeSize(static_cast<uint32_t>(SslmDtype::Float32)) == 4);
	CHECK(DtypeSize(static_cast<uint32_t>(SslmDtype::Float64)) == 8);
	CHECK(DtypeSize(9999u) == 0); // unknown dtype
}

SSLM_TEST(TestKnownSectionTypes, 2) {
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
// SslmArtifact::OpenFromMemory, and calls the loader under test. The loader in
// src/artifact.cpp is fully built (shipped at S-HARDEN-1); every cell below is
// green against it.
// ---------------------------------------------------------------------------

// --- Rejection cells: one per SslmStatus the header/table/section checks can
//     produce, each on a buffer that is structurally valid except the one named
//     defect. ---

SSLM_TEST(TestRejectsBadMagic, 3) {
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

SSLM_TEST(TestRejectsUnsupportedVersion, 4) {
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

SSLM_TEST(TestRejectsHeaderBytesMismatch, 5) {
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

SSLM_TEST(TestRejectsNonzeroFlags, 6) {
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

SSLM_TEST(TestRejectsNonzeroReserved0, 7) {
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

SSLM_TEST(TestRejectsTruncatedHeader, 8) {
	auto built = BuildArtifact({MakeConfigSection()});
	built.bytes.resize(32);  // shorter than the 64-byte header itself

	SslmArtifact out;
	SslmError err;
	auto status = SslmArtifact::OpenFromMemory(built.bytes.data(), built.bytes.size(), out, &err);
	CHECK_MSG(status == SslmStatus::Truncated, "got %s", SslmStatusName(status));
	CHECK(err.code == SslmStatus::Truncated);
	CHECK(err.section_index == kNoSection);
}

// F14 (S-HARDEN-1): a null buffer with a nonzero size must be REJECTED with an
// explicit diagnostic, never dereferenced. Before this fix, OpenFromMemory's
// only length check (`size < kHeaderBytes`) lets a 64+-byte nonzero `size`
// through to `std::memcmp(data, kMagic, 4)`, which faults on `data == nullptr`
// (the external review's ASan probe: abort on `OpenFromMemory(nullptr, 64, ...)`).
// This cell asserts the DEFINED-REJECTION contract, not merely "does not crash"
// — a null check that returns Ok would pass a crash-only assertion and still be
// wrong, so the assertion is the specific status and diagnostic.
SSLM_TEST(TestRejectsNullDataNonzeroSize, 9) {
	SslmArtifact out;
	SslmError err;
	auto status = SslmArtifact::OpenFromMemory(nullptr, 64, out, &err);
	CHECK_MSG(status == SslmStatus::NullData, "got %s, want NullData", SslmStatusName(status));
	CHECK(err.code == SslmStatus::NullData);
	CHECK(err.section_index == kNoSection);
	CHECK(!out.Ok());
}

// The null-buffer rejection does not depend on `size` happening to clear the
// header-length bound — a null pointer with a SMALL nonzero size (which the
// unfixed code's `size < kHeaderBytes` check would have caught first, masking
// whether the null check exists at all) must reach the same explicit status.
SSLM_TEST(TestRejectsNullDataSmallNonzeroSize, 10) {
	SslmArtifact out;
	SslmError err;
	auto status = SslmArtifact::OpenFromMemory(nullptr, 4, out, &err);
	CHECK_MSG(status == SslmStatus::NullData, "got %s, want NullData", SslmStatusName(status));
	CHECK(err.code == SslmStatus::NullData);
	CHECK(!out.Ok());
}

// F14 (S-HARDEN-1): a null path must be rejected explicitly, never handed to
// the stream constructor (whose behavior on a null `const char*` is itself
// undefined — std::ifstream's path constructor requires a valid null-terminated
// string).
SSLM_TEST(TestRejectsNullPath, 11) {
	SslmArtifact out;
	SslmError err;
	auto status = SslmArtifact::OpenFromFile(nullptr, out, &err);
	CHECK_MSG(status == SslmStatus::NullPath, "got %s, want NullPath", SslmStatusName(status));
	CHECK(err.code == SslmStatus::NullPath);
	CHECK(!out.Ok());
}

SSLM_TEST(TestRejectsTruncatedSectionTable, 12) {
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

SSLM_TEST(TestRejectsTooManySections, 13) {
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

SSLM_TEST(TestRejectsFileSizeMismatch, 14) {
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

SSLM_TEST(TestRejectsAlignmentNotPowerOfTwo, 15) {
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

SSLM_TEST(TestRejectsAlignmentBelowMinimum, 16) {
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

SSLM_TEST(TestRejectsAlignmentAboveMaximum, 17) {
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

SSLM_TEST(TestRejectsMisalignedOffset, 18) {
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

SSLM_TEST(TestRejectsSectionOutOfBoundsPastEof, 19) {
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

SSLM_TEST(TestRejectsSectionOffsetOverflow, 20) {
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

SSLM_TEST(TestRejectsSectionOverlapWithHeader, 21) {
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

SSLM_TEST(TestRejectsSectionOverlapWithSection, 22) {
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

SSLM_TEST(TestRejectsBadDtype, 23) {
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

SSLM_TEST(TestRejectsSectionDtypeMismatch, 24) {
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

SSLM_TEST(TestRejectsSizeMismatch, 25) {
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

SSLM_TEST(TestRejectsUnknownSectionType, 26) {
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

SSLM_TEST(TestRejectsDuplicateSection, 27) {
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

SSLM_TEST(TestRejectsMissingConfigSection, 28) {
	auto built = BuildArtifact(
	    {MakeSection(SslmSectionType::Weights, SslmDtype::Int8, EncodeInt8({1, 2, 3, 4}))});  // no Config
	SslmArtifact out;
	SslmError err;
	auto status = SslmArtifact::OpenFromMemory(built.bytes.data(), built.bytes.size(), out, &err);
	CHECK_MSG(status == SslmStatus::MissingSection, "got %s", SslmStatusName(status));
	CHECK(err.code == SslmStatus::MissingSection);
	CHECK(err.section_index == kNoSection);
}

SSLM_TEST(TestRejectsIntegrityMismatch, 29) {
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

SSLM_TEST(TestAcceptsEmptySection, 30) {
	FixtureSection empty = MakeSection(SslmSectionType::Provenance, SslmDtype::Raw, {});  // byte_size 0
	auto built = BuildArtifact({MakeConfigSection(), MakeSigmoidLutSection(), empty});  // SigmoidLut required from v2 (F1)

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

SSLM_TEST(TestAcceptsMaximumAlignment, 31) {
	auto built = BuildArtifact({MakeConfigSection(), MakeSigmoidLutSection(),  // required from v2 (F1)
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

SSLM_TEST(TestAcceptsSectionsInNonAscendingOffsetOrder, 32) {
	// Table row 0 (Config) is placed at the HIGHER byte offset; table row 1
	// (Provenance) is placed at the LOWER one — docs/sslm_format.md: "Their order
	// in the table is not constrained; their byte ranges are."
	//
	// Three rows now (SigmoidLut required from v2, F1): the header+table region
	// is 64 + 3*40 = 184 bytes, so every explicit offset below must clear that —
	// the two-row table this cell predates only needed to clear 144.
	FixtureSection config = MakeConfigSection();
	config.data.resize(32, '{');
	config.alignment = 16;
	config.offset_override = 224;  // 224 % 16 == 0; >= table_end (184) and >= provenance's end (208)

	FixtureSection provenance =
	    MakeSection(SslmSectionType::Provenance, SslmDtype::Raw, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
	                /*alignment=*/16);
	provenance.offset_override = 192;  // 192 % 16 == 0; >= table_end (184); ends at 208, no overlap with config

	// required from v2 (F1) — explicit offset well past both of the above.
	FixtureSection sigmoid_lut = MakeSigmoidLutSection();
	sigmoid_lut.offset_override = 320;  // 320 % 64 == 0; >= config's end (256)

	auto built = BuildArtifact({config, provenance, sigmoid_lut});

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

SSLM_TEST(TestAcceptsReservedSectionTypeStructurally, 33) {
	auto built = BuildArtifact({MakeConfigSection(), MakeSigmoidLutSection(),  // required from v2 (F1)
	                            MakeSection(SslmSectionType::Tokenizer, SslmDtype::Raw, {9, 9, 9})});
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

SSLM_TEST(TestOpenFromFileLoadsValidArtifact, 34) {
	auto built = BuildArtifact({MakeConfigSection(), MakeSigmoidLutSection(),  // required from v2 (F1)
	                            MakeSection(SslmSectionType::Weights, SslmDtype::Int8, EncodeInt8({1, 2, 3, 4}))});

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

SSLM_TEST(TestOpenFromFileMissingFileReturnsIoError, 35) {
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
	// artifact.h pins SslmError's message as carrying "the offending values" —
	// for a missing file, that is the path — so this cell also requires the
	// diagnostic to name it, not merely to report IoError.
	CHECK_MSG(err.message.find(path.string()) != std::string::npos,
	          "message does not carry the missing path: \"%s\"", err.message.c_str());
}

// --- The feature oracle (dim 10): a valid multi-section artifact loads to the
//     exact end-state the bytes were built to carry — not merely "the loader
//     agrees with itself." ---

SSLM_TEST(TestValidArtifactLoadsToExpectedEndState, 36) {
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
	FixtureSection sigmoid_lut = MakeSigmoidLutSection();  // required from v2 (F1)

	auto built = BuildArtifact({config, weights, biases, rope, sigmoid_lut});

	SslmArtifact out;
	SslmError err;
	auto status = SslmArtifact::OpenFromMemory(built.bytes.data(), built.bytes.size(), out, &err);
	CHECK_MSG(status == SslmStatus::Ok, "got %s", SslmStatusName(status));
	CHECK(out.Ok());
	CHECK(out.FormatVersion() == kArtifactFormatVersion);
	CHECK(out.FileBytes() == built.bytes.size());
	CHECK(out.Sections().size() == 5);

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
	    {SslmSectionType::SigmoidLut, SslmDtype::Int32, &sigmoid_lut.data, kSigmoidLutBytes / 4, 64},
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
// Curie's S-HARDEN-8 coverage cell (design §4.2/§4.3; T-412): the generic
// per-section-descriptor-row `reserved` field (artifact.cpp:275-280) is
// rejected by production code today, but no existing test exercised this
// specific field before this cell -- the four existing "reserved == 1"-style
// tests (test_main.cpp:2152,2425,2730,3604) each target a structurally
// distinct PAYLOAD-level reserved field (WGT1/KVC1/CFG1/SIL1), parsed inside
// a typed section by model.cpp, never the generic per-section-descriptor row
// every section carries regardless of type. This cell is expected to PASS
// immediately: the rejection already exists in production (S-HARDEN-1); its
// role is closing the coverage hole the branch-coverage instrument (design
// §4.1) would otherwise have to rediscover, not gating unbuilt behavior.
// ---------------------------------------------------------------------------

SSLM_TEST(TestRejectsNonZeroSectionDescriptorReservedField, 336) {
	auto built = BuildArtifact({MakeConfigSection()});
	const size_t row = kHeaderBytes;  // section 0's descriptor row
	built.bytes[row + 36] = 7;        // the generic reserved field -- distinct from any
	                                  // payload-level reserved field
	RecomputeIntegrityHash(built.bytes);

	SslmArtifact out;
	SslmError err;
	auto status = SslmArtifact::OpenFromMemory(built.bytes.data(), built.bytes.size(), out, &err);
	CHECK_MSG(status == SslmStatus::BadHeader,
	          "a nonzero generic section-descriptor reserved field must reject BadHeader: "
	          "got %s (%s)",
	          SslmStatusName(status), err.message.c_str());
	CHECK(err.code == SslmStatus::BadHeader);
	CHECK(err.section_index == 0);
	CHECK(err.message.find("reserved") != std::string::npos);
}

// The F1 gate's own literal red cell, stated in S-HARDEN-1's plan text: "a
// correctly-hashed config-only v2 artifact expecting a specific
// missing-SigmoidLut diagnostic." Distinct from TestRejectsMissingConfigSection
// (which is missing Config, not SigmoidLut) — this is the mirror case the
// version-indexed schema exists to catch.
SSLM_TEST(TestArtifactRejectsConfigOnlyV2MissingSigmoidLut, 238) {
	auto built = BuildArtifact({MakeConfigSection()});  // Config only; no SigmoidLut

	SslmArtifact out;
	SslmError err;
	auto status = SslmArtifact::OpenFromMemory(built.bytes.data(), built.bytes.size(), out, &err);
	CHECK_MSG(status == SslmStatus::MissingSection, "got %s, want MissingSection", SslmStatusName(status));
	CHECK(err.code == SslmStatus::MissingSection);
	CHECK(err.section_index == kNoSection);
	CHECK_MSG(err.message.find("SigmoidLut") != std::string::npos,
	          "diagnostic does not name SigmoidLut specifically: \"%s\"", err.message.c_str());
	CHECK(!out.Ok());
}

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
#include "superslm/checked_chain_funnel.h"
#include "superslm/intmath.h"
#include "superslm/matmul.h"
#include "superslm/model.h"
#include "superslm/proof_manifest.h"
#include "superslm/sha256.h"
#include "superslm/silu_lut.h"
#include "superslm/silu_lut_canonical.h"
#include "superslm/tokenizer.h"
#include "superslm/trace_hook.h"
#include "sslm_cfg1_hostile_fixtures.h"
#include "sslm_fixtures.h"
#include "sslm_iexp_domain_fixtures.h"
#include "sslm_intmath_fixtures.h"
#include "sslm_kvc1_hostile_fixtures.h"
#include "sslm_matmul_fixtures.h"
#include "sslm_model_hostile_fixtures.h"
#include "sslm_sil1_hostile_fixtures.h"
#include "sslm_silu_lut_real_vectors_fixtures.h"
#include "sslm_s3_1_c30_iexp_domain_sweep_fixtures.h"
#include "sslm_s3_1_wide_intmath_fixtures.h"
#include "silu_lut_golden_table.h"
#include "matmul_golden_pin.h"
#include "sslm_tokenizer_fixtures.h"
#include "sslm_tokenizer_hostile_fixtures.h"
#include "support/bad_alloc_injection.h"

#include <algorithm>
#include <atomic>
#include <stdexcept>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <thread>
#include <type_traits>
#include <vector>

#ifdef _WIN32
#include <process.h>  // _getpid
#else
#include <unistd.h>  // getpid
#endif

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
// SslmArtifact::OpenFromMemory, and calls the loader under test. The loader in
// src/artifact.cpp is fully built (shipped at S-HARDEN-1); every cell below is
// green against it.
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

// F14 (S-HARDEN-1): a null buffer with a nonzero size must be REJECTED with an
// explicit diagnostic, never dereferenced. Before this fix, OpenFromMemory's
// only length check (`size < kHeaderBytes`) lets a 64+-byte nonzero `size`
// through to `std::memcmp(data, kMagic, 4)`, which faults on `data == nullptr`
// (the external review's ASan probe: abort on `OpenFromMemory(nullptr, 64, ...)`).
// This cell asserts the DEFINED-REJECTION contract, not merely "does not crash"
// — a null check that returns Ok would pass a crash-only assertion and still be
// wrong, so the assertion is the specific status and diagnostic.
static void TestRejectsNullDataNonzeroSize() {
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
static void TestRejectsNullDataSmallNonzeroSize() {
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
static void TestRejectsNullPath() {
	SslmArtifact out;
	SslmError err;
	auto status = SslmArtifact::OpenFromFile(nullptr, out, &err);
	CHECK_MSG(status == SslmStatus::NullPath, "got %s, want NullPath", SslmStatusName(status));
	CHECK(err.code == SslmStatus::NullPath);
	CHECK(!out.Ok());
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

static void TestAcceptsMaximumAlignment() {
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

static void TestAcceptsSectionsInNonAscendingOffsetOrder() {
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

static void TestAcceptsReservedSectionTypeStructurally() {
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

static void TestOpenFromFileLoadsValidArtifact() {
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
	// artifact.h pins SslmError's message as carrying "the offending values" —
	// for a missing file, that is the path — so this cell also requires the
	// diagnostic to name it, not merely to report IoError.
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
// Curie's S1 tokenizer red suite (SuperSLM_Plan.md §10; Claude/Curie/
// SuperSLM_S1_Tokenizer_TestDesign-2026-07-19.md). The runtime byte-level BPE
// algorithm is already proven bit-for-bit against the upstream HF tokenizer by
// the Python reference (tools/convert_tokenizer.py, 0 mismatch over 2000+
// adversarial+multilingual lines) — this suite is the C++ gate that proves the
// ported TokenizerView reproduces those upstream ids. TokenizerView is fully
// built (src/tokenizer.cpp, shipped at S-HARDEN-2); every cell below is green
// against it.
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
static void TestTokenizerByteToIdBijectsOntoEveryRawByteValue() {
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

static void TestDecodeSubstitutesReplacementCharForOverlongTwoByteSequence() {
	// 0xC0 0x80: the canonical two-byte encoding of NUL — always overlong (NUL is
	// representable in one byte); C0/C1 can never start a well-formed sequence.
	std::string err;
	auto t = OpenTokenizerWithSingleVocabEntry(std::string("\xC0\x80", 2), &err);
	CHECK_MSG(t.view.Ok(), "setup: %s", err.c_str());
	CHECK(t.view.Decode({0}) == kFffdUtf8 + kFffdUtf8);  // both bytes are individually invalid leads
}

static void TestDecodeSubstitutesReplacementCharForOverlongThreeByteSequence() {
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

static void TestDecodeSubstitutesReplacementCharForSurrogateCodepoint() {
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

static void TestDecodeSubstitutesReplacementCharForCodepointPastU10FFFF() {
	// 0xF5 alone: any lead >= 0xF5 can only encode a codepoint past U+10FFFF.
	std::string err;
	auto t = OpenTokenizerWithSingleVocabEntry(std::string("\xF5\x80\x80\x80", 4), &err);
	CHECK_MSG(t.view.Ok(), "setup: %s", err.c_str());
	// F5 itself is rejected outright (one U+FFFD, one byte consumed); the three
	// trailing 0x80 bytes are then each a lone continuation byte (one U+FFFD
	// each) — four total.
	CHECK(t.view.Decode({0}) == kFffdUtf8 + kFffdUtf8 + kFffdUtf8 + kFffdUtf8);
}

static void TestDecodeSubstitutesReplacementCharForF4WithContinuationPastMax() {
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

static void TestDecodeSubstitutesReplacementCharForOverlongFourByteSequence() {
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

static void TestDecodeSubstitutesReplacementCharForTruncatedFourByteSequence() {
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

static void TestDecodeSubstitutesReplacementCharForLoneContinuationByte() {
	// 0x80 alone: a continuation byte with no lead byte before it.
	std::string err;
	auto t = OpenTokenizerWithSingleVocabEntry(std::string("\x80", 1), &err);
	CHECK_MSG(t.view.Ok(), "setup: %s", err.c_str());
	CHECK(t.view.Decode({0}) == kFffdUtf8);
}

static void TestDecodeSubstitutesReplacementCharForTruncatedSequenceAtEnd() {
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
static void TestSingleVocabTokenizerSurvivesHeapChurnBetweenOpenAndDecode() {
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

static void TestDecodeReconstructsSequenceSplitAcrossTokenBoundary() {
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

static void TestDecodeAndEncodeShareOneStrictDecoderOnWellFormedMultibyteText() {
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

static void TestTok1RejectsBadMagic() {
	auto tok1 = MakeMinimalValidTok1();
	tok1.bytes[0] = 'X';  // was 'T' of "TOK1", offset 0
	AssertTok1Rejected(tok1.bytes, "Tokenizer: bad TOK1 header");
}

static void TestTok1RejectsTruncatedHeader() {
	auto tok1 = MakeMinimalValidTok1();
	tok1.bytes.resize(10);  // shorter than the 24-byte fixed TOK1 header
	AssertTok1Rejected(tok1.bytes, "Tokenizer: bad TOK1 header");
}

static void TestTok1RejectsVocabCountOverflow() {
	auto tok1 = MakeMinimalValidTok1();
	PutU32(tok1.bytes, tok1.layout.vocab_count_off, 0xFFFFFFFFu);
	AssertTok1Rejected(tok1.bytes, "Tokenizer: vocab_count exceeds INT32_MAX");
}

static void TestTok1RejectsMergeCountOverflow() {
	auto tok1 = MakeMinimalValidTok1();
	PutU32(tok1.bytes, tok1.layout.merge_count_off, 0xFFFFFFFFu);
	AssertTok1Rejected(tok1.bytes, "Tokenizer: truncated merges");
}

static void TestTok1RejectsSpecialCountOverflow() {
	auto tok1 = MakeMinimalValidTok1();
	PutU32(tok1.bytes, tok1.layout.special_count_off, 0xFFFFFFFFu);
	AssertTok1Rejected(tok1.bytes, "Tokenizer: truncated special ids");
}

static void TestTok1RejectsTruncatedByteToId() {
	auto tok1 = MakeMinimalValidTok1();
	tok1.bytes.resize(tok1.layout.byte_to_id_off + 100);  // 100 of the required 1024 bytes
	AssertTok1Rejected(tok1.bytes, "Tokenizer: truncated byte_to_id");
}

static void TestTok1RejectsTruncatedVocabOffsets() {
	auto tok1 = MakeMinimalValidTok1();
	tok1.bytes.resize(tok1.layout.vocab_offsets_off + 4);  // 1 of the required vocab_count+1=5 entries
	AssertTok1Rejected(tok1.bytes, "Tokenizer: truncated vocab offsets");
}

static void TestTok1RejectsTruncatedVocabBlobLen() {
	auto tok1 = MakeMinimalValidTok1();
	tok1.bytes.resize(tok1.layout.vocab_blob_len_off + 2);  // half of the 4-byte length field
	AssertTok1Rejected(tok1.bytes, "Tokenizer: truncated vocab blob_len");
}

static void TestTok1RejectsTruncatedVocabBlob() {
	auto tok1 = MakeMinimalValidTok1();
	tok1.bytes.resize(tok1.layout.vocab_blob_off + tok1.layout.vocab_blob_len - 1);  // one byte short
	AssertTok1Rejected(tok1.bytes, "Tokenizer: truncated vocab blob");
}

static void TestTok1RejectsVocabOffsetNonMonotonic() {
	auto tok1 = MakeMinimalValidTok1();
	// Baseline vocab_offsets [0,1,2,3,5,10] (vblob=10, S-HARDEN-2's 5-entry
	// fixture). Bump index 2 from 2 to 4 (still <= vblob): index 3's value (3) is
	// now smaller than its predecessor (4) — a pure non-monotonic violation, no
	// offset exceeds vblob.
	PutU32(tok1.bytes, tok1.layout.vocab_offsets_off + 2 * 4, 4);
	AssertTok1Rejected(tok1.bytes, "Tokenizer: vocab offset out of range");
}

static void TestTok1RejectsLastVocabOffsetExceedsBlob() {
	auto tok1 = MakeMinimalValidTok1();
	// Index `vocab_count` is the terminal offset (derived, not a hardcoded literal,
	// per S-HARDEN-2's fixture replacement note above); baseline value == vblob.
	PutU32(tok1.bytes, tok1.layout.vocab_offsets_off + tok1.layout.vocab_count * 4,
	       tok1.layout.vocab_blob_len + 50);
	AssertTok1Rejected(tok1.bytes, "Tokenizer: vocab offset out of range");
}

static void TestTok1RejectsMiddleVocabOffsetExceedsBlob() {
	auto tok1 = MakeMinimalValidTok1();
	// Index 2 is not the terminal offset (index `vocab_count` is); baseline value 2.
	PutU32(tok1.bytes, tok1.layout.vocab_offsets_off + 2 * 4, tok1.layout.vocab_blob_len + 50);
	AssertTok1Rejected(tok1.bytes, "Tokenizer: vocab offset out of range");
}

static void TestTok1RejectsTruncatedMerges() {
	auto tok1 = MakeMinimalValidTok1();
	tok1.bytes.resize(tok1.layout.merges_off + 6);  // half of the one 12-byte merge record
	AssertTok1Rejected(tok1.bytes, "Tokenizer: truncated merges");
}

static void TestTok1RejectsTruncatedSpecialIds() {
	auto tok1 = MakeMinimalValidTok1();
	tok1.bytes.resize(tok1.layout.special_ids_off + 2);  // half of the one 4-byte special id
	AssertTok1Rejected(tok1.bytes, "Tokenizer: truncated special ids");
}

static void TestTok1RejectsTruncatedSpecialOffsets() {
	auto tok1 = MakeMinimalValidTok1();
	tok1.bytes.resize(tok1.layout.special_offsets_off + 4);  // 1 of the required special_count+1=2 entries
	AssertTok1Rejected(tok1.bytes, "Tokenizer: truncated special offsets");
}

static void TestTok1RejectsTruncatedSpecialBlobLen() {
	auto tok1 = MakeMinimalValidTok1();
	tok1.bytes.resize(tok1.layout.special_blob_len_off + 2);  // half of the 4-byte length field
	AssertTok1Rejected(tok1.bytes, "Tokenizer: truncated special blob_len");
}

static void TestTok1RejectsTruncatedSpecialBlob() {
	auto tok1 = MakeMinimalValidTok1();
	tok1.bytes.resize(tok1.layout.special_blob_off + tok1.layout.special_blob_len - 1);  // one byte short
	AssertTok1Rejected(tok1.bytes, "Tokenizer: truncated special blob");
}

static void TestTok1RejectsSpecialOffsetNonMonotonic() {
	auto tok1 = MakeMinimalValidTok1();
	// Baseline special_offsets [0,5] (sblob=5). Set index 0 to 6 (> index 1's 5) —
	// non-monotonic; index 1 (5) does not itself exceed sblob (5), so the range
	// branch cannot also fire.
	PutU32(tok1.bytes, tok1.layout.special_offsets_off + 0 * 4, 6);
	AssertTok1Rejected(tok1.bytes, "Tokenizer: bad special offset");
}

static void TestTok1RejectsSpecialOffsetOutOfRange() {
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

static void TestTok1RejectsUnsupportedVersion() {
	auto tok1 = MakeMinimalValidTok1();
	PutU32(tok1.bytes, tok1.layout.version_off, 2);  // only 1 is the declared TOK1 version
	AssertTok1Rejected(tok1.bytes, "Tokenizer: unsupported TOK1 version");
}

static void TestTok1RejectsNonzeroReserved() {
	auto tok1 = MakeMinimalValidTok1();
	PutU32(tok1.bytes, tok1.layout.reserved_off, 1);
	AssertTok1Rejected(tok1.bytes, "Tokenizer: TOK1 reserved field != 0");
}

// --- S-HARDEN-2 (F18): vocab_count's own declared domain — zero encodes no byte
//     and indexes no token; every id this parser stores is narrowed to int32_t
//     downstream, so a vocab_count past INT32_MAX must be rejected before that
//     narrowing (or the bound checks below, which compare ids against vocab_count
//     as read) can be silently wrong. ---

static void TestTok1RejectsVocabCountZero() {
	auto tok1 = MakeMinimalValidTok1();
	PutU32(tok1.bytes, tok1.layout.vocab_count_off, 0);
	AssertTok1Rejected(tok1.bytes, "Tokenizer: vocab_count == 0");
}

static void TestTok1RejectsVocabCountExceedsInt32Max() {
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

static void TestTok1RejectsByteToIdEntryAtOrAboveVocabCount() {
	auto tok1 = MakeMinimalValidTok1();
	// byte_to_id['z'] defaults to 0 (a valid id, below vocab_count=5); mutate it to
	// vocab_count itself — one past the last real vocabulary slot.
	PutU32(tok1.bytes, tok1.layout.byte_to_id_off + static_cast<unsigned char>('z') * 4, tok1.layout.vocab_count);
	AssertTok1Rejected(tok1.bytes, "Tokenizer: byte_to_id entry >= vocab_count");
}

static void TestTok1RejectsMergeOperandAAtOrAboveVocabCount() {
	auto tok1 = MakeMinimalValidTok1();
	// The one merge record is (a=0, b=1, merged=3) at merges_off; field `a` is the
	// first u32.
	PutU32(tok1.bytes, tok1.layout.merges_off + 0, tok1.layout.vocab_count);
	AssertTok1Rejected(tok1.bytes, "Tokenizer: merge operand or result >= vocab_count");
}

static void TestTok1RejectsMergeOperandBAtOrAboveVocabCount() {
	auto tok1 = MakeMinimalValidTok1();
	PutU32(tok1.bytes, tok1.layout.merges_off + 4, tok1.layout.vocab_count);
	AssertTok1Rejected(tok1.bytes, "Tokenizer: merge operand or result >= vocab_count");
}

static void TestTok1RejectsMergeResultAtOrAboveVocabCount() {
	auto tok1 = MakeMinimalValidTok1();
	PutU32(tok1.bytes, tok1.layout.merges_off + 8, tok1.layout.vocab_count);
	AssertTok1Rejected(tok1.bytes, "Tokenizer: merge operand or result >= vocab_count");
}

static void TestTok1RejectsSpecialIdAtOrAboveVocabCount() {
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

static void TestUni1RejectsBadMagic() {
	auto uni1 = MakeMinimalValidUni1();
	uni1.bytes[0] = 'X';  // was 'U' of "UNI1", offset 0
	AssertUni1Rejected(uni1.bytes, "UnicodeTables: bad UNI1 header");
}

static void TestUni1RejectsTruncatedHeader() {
	auto uni1 = MakeMinimalValidUni1();
	uni1.bytes.resize(5);  // shorter than the 8-byte magic+version header
	AssertUni1Rejected(uni1.bytes, "UnicodeTables: bad UNI1 header");
}

// S-HARDEN-2 (F6): offset 4's version field was skipped past (`pos = 8`) but
// never actually read and compared — the same guard-vitality gap as TOK1's
// version/reserved fields.
static void TestUni1RejectsUnsupportedVersion() {
	auto uni1 = MakeMinimalValidUni1();
	PutU32(uni1.bytes, uni1.layout.version_off, 2);  // only 1 is the declared UNI1 version
	AssertUni1Rejected(uni1.bytes, "UnicodeTables: unsupported UNI1 version");
}

static void TestUni1RejectsLetterCountFieldTruncated() {
	auto uni1 = MakeMinimalValidUni1();
	uni1.bytes.resize(uni1.layout.letter_count_off + 2);  // half of the 4-byte count field
	AssertUni1Rejected(uni1.bytes, "UnicodeTables: truncated range count");
}

static void TestUni1RejectsLetterRangesTruncated() {
	auto uni1 = MakeMinimalValidUni1();
	uni1.bytes.resize(uni1.layout.letter_data_off + 4);  // 4 of the required 16 bytes (2 ranges)
	AssertUni1Rejected(uni1.bytes, "UnicodeTables: truncated ranges");
}

static void TestUni1RejectsLetterCountOverflow() {
	auto uni1 = MakeMinimalValidUni1();
	PutU32(uni1.bytes, uni1.layout.letter_count_off, 0xFFFFFFFFu);
	AssertUni1Rejected(uni1.bytes, "UnicodeTables: truncated ranges");
}

static void TestUni1RejectsNumberRangesTruncated() {
	auto uni1 = MakeMinimalValidUni1();
	uni1.bytes.resize(uni1.layout.number_data_off + 4);  // 4 of the required 8 bytes (1 range)
	AssertUni1Rejected(uni1.bytes, "UnicodeTables: truncated ranges");
}

static void TestUni1RejectsNumberCountOverflow() {
	auto uni1 = MakeMinimalValidUni1();
	PutU32(uni1.bytes, uni1.layout.number_count_off, 0xFFFFFFFFu);
	AssertUni1Rejected(uni1.bytes, "UnicodeTables: truncated ranges");
}

static void TestUni1RejectsSpaceRangesTruncated() {
	auto uni1 = MakeMinimalValidUni1();
	uni1.bytes.resize(uni1.layout.space_data_off + 8);  // 8 of the required 16 bytes (2 ranges)
	AssertUni1Rejected(uni1.bytes, "UnicodeTables: truncated ranges");
}

static void TestUni1RejectsSpaceCountOverflow() {
	auto uni1 = MakeMinimalValidUni1();
	PutU32(uni1.bytes, uni1.layout.space_count_off, 0xFFFFFFFFu);
	AssertUni1Rejected(uni1.bytes, "UnicodeTables: truncated ranges");
}

static void TestUni1RejectsCccTruncated() {
	auto uni1 = MakeMinimalValidUni1();
	uni1.bytes.resize(uni1.layout.ccc_data_off + 4);  // 4 of the required 8 bytes (1 entry)
	AssertUni1Rejected(uni1.bytes, "UnicodeTables: truncated ccc");
}

static void TestUni1RejectsCccCountOverflow() {
	auto uni1 = MakeMinimalValidUni1();
	PutU32(uni1.bytes, uni1.layout.ccc_count_off, 0xFFFFFFFFu);
	AssertUni1Rejected(uni1.bytes, "UnicodeTables: truncated ccc");
}

static void TestUni1RejectsDecompCpsTruncated() {
	auto uni1 = MakeMinimalValidUni1();
	uni1.bytes.resize(uni1.layout.decomp_cps_off + 2);  // half of the required 4 bytes (1 cp)
	AssertUni1Rejected(uni1.bytes, "UnicodeTables: truncated decomp cps");
}

static void TestUni1RejectsDecompCountOverflow() {
	auto uni1 = MakeMinimalValidUni1();
	PutU32(uni1.bytes, uni1.layout.decomp_count_off, 0xFFFFFFFFu);
	AssertUni1Rejected(uni1.bytes, "UnicodeTables: truncated decomp cps");
}

static void TestUni1RejectsDecompOffsetsTruncated() {
	auto uni1 = MakeMinimalValidUni1();
	uni1.bytes.resize(uni1.layout.decomp_offsets_off + 4);  // 1 of the required decomp_count+1=2 entries
	AssertUni1Rejected(uni1.bytes, "UnicodeTables: truncated decomp offsets");
}

static void TestUni1RejectsDecompOffsetOutOfRange() {
	auto uni1 = MakeMinimalValidUni1();
	// Baseline decomp offsets [0,2] (seq_len=2). Bump the terminal offset (index 1)
	// past seq_len while keeping it >= its predecessor — a pure range violation.
	PutU32(uni1.bytes, uni1.layout.decomp_offsets_off + 1 * 4, uni1.layout.seq_len + 97);
	AssertUni1Rejected(uni1.bytes, "UnicodeTables: bad decomp offset");
}

static void TestUni1RejectsDecompOffsetNonMonotonic() {
	auto uni1 = MakeMinimalValidUni1();
	// Baseline decomp offsets [0,2]. Set index 0 to 3 (> index 1's 2) — non-
	// monotonic; index 1 (2) does not itself exceed seq_len (2).
	PutU32(uni1.bytes, uni1.layout.decomp_offsets_off + 0 * 4, 3);
	AssertUni1Rejected(uni1.bytes, "UnicodeTables: bad decomp offset");
}

static void TestUni1RejectsDecompSeqTruncated() {
	auto uni1 = MakeMinimalValidUni1();
	uni1.bytes.resize(uni1.layout.decomp_seq_off + 4);  // half of the required 8 bytes (seq_len=2)
	AssertUni1Rejected(uni1.bytes, "UnicodeTables: truncated decomp seq");
}

static void TestUni1RejectsComposeTruncated() {
	auto uni1 = MakeMinimalValidUni1();
	uni1.bytes.resize(uni1.layout.compose_data_off + 6);  // half of the required 12 bytes (1 entry)
	AssertUni1Rejected(uni1.bytes, "UnicodeTables: truncated compose");
}

static void TestUni1RejectsComposeCountOverflow() {
	auto uni1 = MakeMinimalValidUni1();
	PutU32(uni1.bytes, uni1.layout.compose_count_off, 0xFFFFFFFFu);
	AssertUni1Rejected(uni1.bytes, "UnicodeTables: truncated compose");
}

// ---------------------------------------------------------------------------
// Curie's S2.0a WGT1/BIA1/ROP1 tensor-manifest hostile-input red suite
// (SuperSLM_Plan.md S2.0a; docs/sslm_format.md "Model sub-formats").
// SslmTensorManifest::Parse parses one array section's self-contained tensor
// manifest AFTER SslmArtifact has verified whole-file structure and integrity
// — a crafted integrity-valid artifact can still carry a malformed manifest
// inside a validated section, so this sub-parse is its own hostile-input
// trust boundary, held to the same T-129 bar. SslmTensorManifest::Parse is
// fully built (src/model.cpp, shipped at S-HARDEN-1); every cell below is
// green against it.
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
// TOK1/UNI1 and WGT1/BIA1/ROP1. SslmKeyedConstants::Parse is fully built
// (src/model.cpp, shipped at S-HARDEN-1); every cell below is green against
// it.
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
// never repaired. ParseConfig is fully built (src/model.cpp, shipped at
// S-HARDEN-1); every hostile cell below is green against it.
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
// Unlike every CFG1 cell above, this feature oracle exercises
// SslmTensorManifest::Parse (S2.0a), not ParseConfig.
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
// fully ported and bit-exact against the pinned reference (shipped at
// S-HARDEN-1); every cell below is green against it.
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

// --- F-S3-7 / Sec11 S3.1: wide-row (int64 input width) siblings --------------
//
// MaxAbsReduceWide / RowBoundsWide / RequantTokenCodeWide have real, shipped
// bodies (src/intmath.cpp:270-281 and neighbouring lines), fixed against the
// INT64_MIN signed-overflow and n==0 read-before-check defects Poirot's
// ac34677 review found (S1, S3) -- the unsigned-magnitude form and the
// defined-empty-reduction early return are both in place. Every cell below is
// specified completely against
// Claude/Curie/superslm-s3.1-checked-chain-funnel-test-design-2026-07-28.md
// Sec4.1/Sec4.2/Sec4.3.

static void TestMaxAbsReduceWide() {
	using namespace superslm_test;
	for (size_t i = 0; i < kMaxAbsWideCasesCount; ++i) {
		const MaxAbsWideCase& c = kMaxAbsWideCases[i];
		int64_t got = superslm::MaxAbsReduceWide(c.data, c.n);
		CHECK_MSG(got == c.expected, "%s: MaxAbsReduceWide(n=%zu) == %lld, want %lld", c.label, c.n,
		          static_cast<long long>(got), static_cast<long long>(c.expected));
	}

	// Order-independence, checked directly rather than only via matching
	// expected values -- mirrors TestMaxAbsReduce's own discipline at the
	// narrow width, widened.
	int64_t perm_a = superslm::MaxAbsReduceWide(kMaxAbsWideData8, 5);
	int64_t perm_b = superslm::MaxAbsReduceWide(kMaxAbsWideData9, 5);
	int64_t perm_c = superslm::MaxAbsReduceWide(kMaxAbsWideData10, 5);
	CHECK_MSG(perm_a == perm_b && perm_b == perm_c,
	          "MaxAbsReduceWide is not order-independent: perm_a=%lld perm_b=%lld perm_c=%lld",
	          static_cast<long long>(perm_a), static_cast<long long>(perm_b),
	          static_cast<long long>(perm_c));
}

static void TestRowBoundsWide() {
	using namespace superslm_test;
	for (size_t i = 0; i < kRowBoundsWideCasesCount; ++i) {
		const RowBoundsWideCase& c = kRowBoundsWideCases[i];
		int64_t got_max = INT64_C(0xdeadbeef);
		int64_t got_min = INT64_C(0xdeadbeef);
		superslm::RowBoundsWide(c.data, c.n, &got_max, &got_min);
		CHECK_MSG(got_max == c.expected_max, "%s: RowBoundsWide(n=%zu).out_max == %lld, want %lld",
		          c.label, c.n, static_cast<long long>(got_max), static_cast<long long>(c.expected_max));
		CHECK_MSG(got_min == c.expected_min, "%s: RowBoundsWide(n=%zu).out_min == %lld, want %lld",
		          c.label, c.n, static_cast<long long>(got_min), static_cast<long long>(c.expected_min));
	}

	// Order-independence on the two permuted in-range rows: both permutations
	// of the same multiset must report the identical (max, min) pair.
	int64_t max_a = 0, min_a = 0, max_b = 0, min_b = 0;
	superslm::RowBoundsWide(kRowBoundsInRangeRowPermA, 4, &max_a, &min_a);
	superslm::RowBoundsWide(kRowBoundsInRangeRowPermB, 4, &max_b, &min_b);
	CHECK_MSG(max_a == max_b && min_a == min_b,
	          "RowBoundsWide is not order-independent: (max_a=%lld,min_a=%lld) vs (max_b=%lld,min_b=%lld)",
	          static_cast<long long>(max_a), static_cast<long long>(min_a), static_cast<long long>(max_b),
	          static_cast<long long>(min_b));
}

static void TestRequantTokenCodeWide() {
	using namespace superslm_test;
	for (size_t i = 0; i < kRequantWideCasesCount; ++i) {
		const RequantWideCase& c = kRequantWideCases[i];
		int8_t got = superslm::RequantTokenCodeWide(c.x_i, c.r, c.s);
		CHECK_MSG(got == c.expected, "%s: RequantTokenCodeWide(%lld, %lld, %d) == %d, want %d", c.label,
		          static_cast<long long>(c.x_i), static_cast<long long>(c.r), c.s, static_cast<int>(got),
		          static_cast<int>(c.expected));
		CHECK_MSG(got >= -127 && got <= 127,
		          "%s: RequantTokenCodeWide(%lld, %lld, %d) == %d, out of [-127, 127]", c.label,
		          static_cast<long long>(c.x_i), static_cast<long long>(c.r), c.s, static_cast<int>(got));
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
// Curie's S2.2 nonlinear scalar primitives red suite. src/intmath.cpp's
// ISqrt/ISqrtTrace/ShiftByMax/IExpFromConstants bodies are fully ported
// (shipped at S-HARDEN-1); every cell below is green against them.
// Test-design record: Claude/Curie/
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

// PORTED (this session, S-HARDEN-0 final API): IExpFromConstants is REMOVED --
// evaluation is now construct-then-evaluate. Every one of these 36 goldens is
// documented in-domain (each already produced a value that fits int64_t, by
// construction of gen_intmath_fixtures.py's own width-probe search), so
// IExpConstruct is asserted to return kOk before IExpEvaluate is ever called;
// the expected numeric value is UNCHANGED from the original suite (per the
// commission: if any of these 36 values had moved, that would be a real
// regression, not a port detail -- none did, confirmed by this port compiling
// and passing against the unedited kIExpCases table).
static void TestIExpConstructAndEvaluateMatchGoldenCasesAcrossKIExpCases() {
	using namespace superslm_test;
	for (size_t i = 0; i < kIExpCasesCount; ++i) {
		const IExpCase& c = kIExpCases[i];
		superslm::IExpConstruction out;
		superslm::IExpDomain d = superslm::IExpConstruct(c.q, c.q_ln2, c.q_b, c.q_c, &out);
		CHECK_MSG(d == superslm::IExpDomain::kOk,
		          "%s: IExpConstruct(q=%lld, q_ln2=%lld, q_b=%lld, q_c=%lld) returned domain %d, want kOk "
		          "-- every kIExpCases golden is documented in-domain",
		          c.label, static_cast<long long>(c.q), static_cast<long long>(c.q_ln2),
		          static_cast<long long>(c.q_b), static_cast<long long>(c.q_c), static_cast<int>(d));
		if (d != superslm::IExpDomain::kOk) continue;
		int64_t got = superslm::IExpEvaluate(out);
		CHECK_MSG(got == c.expected,
		          "%s: IExpEvaluate(IExpConstruct(q=%lld, q_ln2=%lld, q_b=%lld, q_c=%lld)) == %lld, want %lld",
		          c.label, static_cast<long long>(c.q), static_cast<long long>(c.q_ln2),
		          static_cast<long long>(c.q_b), static_cast<long long>(c.q_c), static_cast<long long>(got),
		          static_cast<long long>(c.expected));
	}
}

// PORTED (this session): construct-then-evaluate, same claim.
static void TestIExpConstructAndEvaluateClipClampsIdenticallyAcrossFamily() {
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
		superslm::IExpConstruction clip_out, beyond_out;
		superslm::IExpDomain clip_d = superslm::IExpConstruct(clip.q, clip.q_ln2, clip.q_b, clip.q_c, &clip_out);
		superslm::IExpDomain beyond_d =
		    superslm::IExpConstruct(beyond.q, beyond.q_ln2, beyond.q_b, beyond.q_c, &beyond_out);
		CHECK_MSG(clip_d == superslm::IExpDomain::kOk && beyond_d == superslm::IExpDomain::kOk,
		          "%s/%s: both are documented in-domain fixtures; IExpConstruct returned %d/%d, want kOk/kOk",
		          clip.label, beyond.label, static_cast<int>(clip_d), static_cast<int>(beyond_d));
		if (clip_d != superslm::IExpDomain::kOk || beyond_d != superslm::IExpDomain::kOk) continue;
		int64_t got_clip = superslm::IExpEvaluate(clip_out);
		int64_t got_beyond = superslm::IExpEvaluate(beyond_out);
		CHECK_MSG(got_clip == got_beyond,
		          "%s/%s: clip-boundary result %lld != beyond-clip result %lld for the same constants",
		          clip.label, beyond.label, static_cast<long long>(got_clip), static_cast<long long>(got_beyond));
	}
	CHECK_MSG(pairs_checked == 6, "expected 6 clip/beyond-clip fixture pairs (5 realistic + 1 qln2_min), found %zu",
	          pairs_checked);
}

// ---------------------------------------------------------------------------
// Curie's S2.6-amendment suite for IExpConstantsInDomain (D-SLM78/79/81;
// Claude/Loki/softmax-s2.6-strike-2026-07-21.md; Claude/Curie/
// superslm-s2.6-softmax-iexp-domain-test-design-2026-07-21.md). This primitive
// is declared and shipped in src/intmath.cpp / include/superslm/intmath.h.
//
// The two accessor cells originally here (TestIExpShiftMatchesIndependentlyDerivedZ,
// TestIExpBaseMatchesIndependentlyDerivedBase, against IExpShift/IExpBase) were
// PORTED to TestIExpConstructMatchesAccessorCasesZAndBase above (Brunel, mid-build,
// 2026-07-21): IExpShift/IExpBase are removed from the public header by S-HARDEN-0,
// so every cell using them had to move to IExpConstruct's IExpConstruction output.
// kIExpAccessorCases itself (36 rows) is unchanged -- its values do not depend on
// which function reads them.
//
// Every expected value in sslm_iexp_domain_fixtures.h is computed by
// tests/gen_iexp_domain_fixtures.py, transcribing IExpFromConstants's documented
// five-line decomposition directly in Python arbitrary precision -- never by
// calling the primitives under test and never by re-deriving the bound in
// fixed-width int64 (the exact shape of the D-SLM81 defect this amendment
// fixes).
// ---------------------------------------------------------------------------

static void TestIExpConstantsInDomainAcrossCorpus() {
	// The full domain-predicate corpus: the strike's exact input, both the z=0 and
	// the z>=1 in-domain/out-of-domain boundaries forced EXACTLY on both sides (a
	// boundary is only authored where a valid int64_t q_c actually sits on the
	// transition -- see gen_iexp_domain_fixtures.py's note on the rejected
	// 1733160715-base z=1/z=30 construction, which forces nothing because the
	// theoretical boundary q_c there does not fit int64_t), the q_b-alone overflow
	// axis (large q_b, q_c=1, distinct from the strike's q_c-dominated overflow),
	// a realistic operating-scale positive case, and every one of the 36 fixtures
	// already shipped in sslm_intmath_fixtures.h's kIExpCases (each must be
	// in-domain, since each already produced a golden that fits int64_t).
	using namespace superslm_test;
	for (size_t i = 0; i < kIExpDomainCasesCount; ++i) {
		const IExpDomainCase& c = kIExpDomainCases[i];
		bool got = superslm::IExpConstantsInDomain(c.q, c.q_ln2, c.q_b, c.q_c);
		CHECK_MSG(got == c.expected_in_domain,
		          "%s: IExpConstantsInDomain(q=%lld, q_ln2=%lld, q_b=%lld, q_c=%lld) == %s, want %s", c.label,
		          static_cast<long long>(c.q), static_cast<long long>(c.q_ln2), static_cast<long long>(c.q_b),
		          static_cast<long long>(c.q_c), got ? "true" : "false", c.expected_in_domain ? "true" : "false");
	}
}

static void TestIExpConstantsInDomainRejectsStrikeExactInput() {
	// Standalone, individually diagnosable regression cell for the defect itself
	// (Claude/Loki/softmax-s2.6-strike-2026-07-21.md): this exact call is
	// contract-legal under every documented LOWER-bound precondition (q<=0,
	// q_ln2>=1) and yet base^2+q_c = 12,227,218,100,874,087,032 exceeds INT64_MAX.
	// Redundant with one row of TestIExpConstantsInDomainAcrossCorpus by design --
	// this is the one cell this whole amendment exists to force, and it must be
	// able to fail on its own without scanning a table's output to find it.
	bool in_domain = superslm::IExpConstantsInDomain(INT64_C(0), INT64_C(887904998), INT64_C(1733160715),
	                                                  INT64_C(9223372036854775807));
	CHECK_MSG(!in_domain,
	          "IExpConstantsInDomain(0, 887904998, 1733160715, 2^63-1) == true, want false -- this is the "
	          "strike's exact contract-legal input for which IExpFromConstants returns a NEGATIVE value "
	          "(Claude/Loki/softmax-s2.6-strike-2026-07-21.md)");
}

namespace {

// The header's claimed sufficient-AND-necessary condition for
// IExpConstantsInDomain's one-call-per-triple shortcut (include/superslm/
// intmath.h, corrected at c33843d: "The one-call-per-triple shortcut holds
// iff 2*q_b >= q_ln2 - 1"). This is prose about how a CALLER may use the
// primitive, not behavior the primitive itself performs, so nothing in
// src/intmath.cpp encodes it; TestIExpConstantsInDomainShortcutConditionMatchesHeaderClaim
// below is the only place that does, and is therefore the only place a wrong
// version of this claim (Claude/Poirot/93622d3-s2.2-iexp-amendment-close-
// round-review-2026-07-21.md, finding N7) can be caught by the suite rather
// than by a reviewer re-deriving it by hand.
bool IExpShortcutHolds(int64_t q_ln2, int64_t q_b) { return 2 * q_b >= q_ln2 - 1; }

}  // namespace

static void TestIExpConstantsInDomainShortcutConditionMatchesHeaderClaim() {
	// Four scenarios authored in tests/gen_iexp_domain_fixtures.py at
	// q_ln2 = 3,000,000,000 (labels "shortcut_*_q0" / "shortcut_*_far_end" in
	// kIExpDomainCases), each placing q_c so one end of the row -- q=0 or the
	// far end q=-(q_ln2-1) -- sits exactly at its own in-domain boundary:
	//
	//   A (q_b=1,800,000,000): condition HOLDS.       q0=false, far=true.
	//   B (q_b=1,000,000,000): condition FAILS.       q0=true,  far=false.
	//   C (q_b=1,500,000,000, the least satisfying):  condition HOLDS.  q0=true,  far=true.
	//   D (q_b=1,499,999,999, one less than C):       condition FAILS.  q0=true,  far=false.
	//
	// For every scenario this test (a) pins IExpShortcutHolds's own return
	// value against the scenario's independently-known intent, and (b) checks
	// that the REAL primitive's behavior at q=0 and at the row's other
	// extreme is consistent with that claim: where the shortcut holds, a
	// caller who discharges the row at q=0 alone can never get a false
	// all-clear (q0=true implies far=true); where it fails, this suite
	// constructed an actual witness of exactly that false all-clear
	// (q0=true, far=false) -- proving the condition is not merely sufficient
	// but tight, i.e. genuinely necessary at these constants.
	using namespace superslm_test;

	struct Scenario {
		const char* q0_label;
		const char* far_label;
		int64_t q_ln2;
		int64_t q_b;
		bool expected_holds;
	};
	static const Scenario kScenarios[] = {
	    {"shortcut_holds_interior_A_q0", "shortcut_holds_interior_A_far_end", INT64_C(3000000000),
	     INT64_C(1800000000), true},
	    {"shortcut_fails_interior_B_q0", "shortcut_fails_interior_B_far_end", INT64_C(3000000000),
	     INT64_C(1000000000), false},
	    {"shortcut_boundary_holds_C_q0", "shortcut_boundary_holds_C_far_end", INT64_C(3000000000),
	     INT64_C(1500000000), true},
	    {"shortcut_boundary_fails_D_q0", "shortcut_boundary_fails_D_far_end", INT64_C(3000000000),
	     INT64_C(1499999999), false},
	};

	for (const Scenario& s : kScenarios) {
		bool computed_holds = IExpShortcutHolds(s.q_ln2, s.q_b);
		CHECK_MSG(computed_holds == s.expected_holds,
		          "IExpShortcutHolds(q_ln2=%lld, q_b=%lld) == %s, want %s (%s) -- the header's condition "
		          "no longer matches this scenario's known intent",
		          static_cast<long long>(s.q_ln2), static_cast<long long>(s.q_b), computed_holds ? "true" : "false",
		          s.expected_holds ? "true" : "false", s.q0_label);

		const IExpDomainCase* q0_case = nullptr;
		const IExpDomainCase* far_case = nullptr;
		for (size_t i = 0; i < kIExpDomainCasesCount; ++i) {
			const IExpDomainCase& c = kIExpDomainCases[i];
			if (std::strcmp(c.label, s.q0_label) == 0) q0_case = &c;
			if (std::strcmp(c.label, s.far_label) == 0) far_case = &c;
		}
		CHECK_MSG(q0_case != nullptr, "fixture label not found: %s", s.q0_label);
		CHECK_MSG(far_case != nullptr, "fixture label not found: %s", s.far_label);
		if (q0_case == nullptr || far_case == nullptr) continue;

		bool got_q0 = superslm::IExpConstantsInDomain(q0_case->q, q0_case->q_ln2, q0_case->q_b, q0_case->q_c);
		bool got_far = superslm::IExpConstantsInDomain(far_case->q, far_case->q_ln2, far_case->q_b, far_case->q_c);

		if (computed_holds) {
			CHECK_MSG(!(got_q0 && !got_far),
			          "%s: shortcut claimed to hold (q_ln2=%lld, q_b=%lld) but q=0 answered true while the far "
			          "end answered false -- discharging at q=0 alone is unsound here, contradicting the header's "
			          "claim",
			          s.q0_label, static_cast<long long>(s.q_ln2), static_cast<long long>(s.q_b));
		} else {
			CHECK_MSG(got_q0 && !got_far,
			          "%s: this scenario's witness (q_ln2=%lld, q_b=%lld) no longer demonstrates q=0-alone "
			          "unsoundness (got_q0=%s, got_far=%s) -- the constructed q_c must produce true at q=0 and "
			          "false at the far end to prove the shortcut genuinely fails here",
			          s.q0_label, static_cast<long long>(s.q_ln2), static_cast<long long>(s.q_b),
			          got_q0 ? "true" : "false", got_far ? "true" : "false");
		}
	}
}

// D-SLM79 part 2's internal domain assert on IExpFromConstants no longer has a
// subject under the final API (IExpFromConstants is removed; IExpConstruct/
// IExpEvaluate assert nothing, in any build configuration). The wrapped-value
// golden it used to pin survives as
// TestIExpConstructAndEvaluateProducesKnownWrappedValueForOutOfDomainConstants,
// defined further below -- see that function's own comment for the full account.

// ---------------------------------------------------------------------------
// Curie's S2.3 RopeApplyPair red suite. src/intmath.cpp's RopeApplyPair
// body is fully ported (shipped at S-HARDEN-1); every cell below is green
// against it. Test-design record: Claude/Curie/
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
// SuperSLM_Plan.md S2.4). Two code-under-test surfaces, both fully built
// (shipped at S-HARDEN-1):
//   - `superslm::ParseSigmoidLut` (src/model.cpp) — the SIL1 fixed-layout
//     hostile sub-parse (§8, §12 dim 2/5/9).
//   - `superslm::SiluSigmoidQ15` (src/silu_lut.cpp) — the runtime index
//     derivation + interpolation (§5, §6, §12 dim 4/6/7/10).
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
	// SigmoidLutValue's contract is undefined for i >= entry_count, so guard
	// against reading through a null/empty view rather than let a defect
	// that reports Ok with an empty view crash this cell.
	CHECK_MSG(out.values != nullptr, "out.values is null on a status==Ok parse");
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
	// Guard against a defect that reports Ok over a null/empty view (see
	// TestMinimalSil1ParsesAndReadsBackAllNodes).
	CHECK_MSG(out.values != nullptr && out.entry_count == kSigmoidLutEntries,
	          "out is not a populated view on a status==Ok parse");
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
	// Guard against a defect that reports Ok over a null/empty view (see
	// TestMinimalSil1ParsesAndReadsBackAllNodes).
	CHECK_MSG(out.values != nullptr && out.entry_count == kSigmoidLutEntries,
	          "out is not a populated view on a status==Ok parse");
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
// S-HARDEN-1 (F20/F22): pinned canonical content. SIL1 is a universal
// construction the spec fixes entirely, not model-specific learned data — so a
// section that passes every STRUCTURAL check (size/magic/version/count/
// reserved) but carries node values that are not byte-for-byte the canonical
// table must still be rejected. This is the gap the external review's strike
// exploited: F20's original fix validated node RANGES/monotonicity: this cell
// forces a case that is well-formed under a range check yet is not the
// canonical table, which only an exact-content comparison catches. §17.3
// cell 6's oracle ("pinned canonical content... a section hash or
// byte-for-byte comparison").
// ---------------------------------------------------------------------------

static void TestSil1RejectsSingleNodeContentMismatch() {
	using namespace superslm_test;
	auto bytes = MakeMinimalValidSil1();
	// Node 512 (the table's midpoint, sigmoid(0)*2^15 == 16384): off by exactly
	// one from canonical — a range/monotonicity check would not catch this, an
	// exact-content check must.
	const uint32_t mutated = static_cast<uint32_t>(superslm::kSiluLutCanonicalTable[512] + 1);
	PutU32(bytes, kSil1NodesOff + 512u * 4u, mutated);
	AssertSil1Rejected(bytes, SslmModelStatus::BadSigmoidLutContent,
	                    "SIL1 node[512] == canonical+1 (single off-by-one, still monotone and in-range)");
}

// The exact shape the external review's strike used against F20's original
// range-only fix: two ADJACENT nodes at INT32_MIN and INT32_MAX. Both are
// individually "in range" for an int32 field and the pair is even monotone
// non-decreasing is not required by a range check — this is precisely the
// operand that reached SiluSigmoidQ15's `diff = hi - lo` and produced UB
// (F20's remedy text). The content-pinning check must reject this
// independently of any node-range/monotonicity check that may or may not
// exist, because content-pinning is what this slot commits to as the actual
// gate (defence-in-depth, never the sole claimed protection).
static void TestSil1RejectsHostileExtremeAdjacentNodes() {
	using namespace superslm_test;
	auto bytes = MakeMinimalValidSil1();
	PutU32(bytes, kSil1NodesOff + 512u * 4u, static_cast<uint32_t>(INT32_MIN));
	PutU32(bytes, kSil1NodesOff + 513u * 4u, static_cast<uint32_t>(INT32_MAX));
	AssertSil1Rejected(bytes, SslmModelStatus::BadSigmoidLutContent,
	                    "SIL1 node[512]=INT32_MIN, node[513]=INT32_MAX (the F20/F22 strike's exact operand)");
}

// The gate cell, run through the REAL artifact/parser path end to end
// (OpenFromMemory, THEN ParseSigmoidLut on the section it returns) so parser
// safety and construction identity cannot drift apart (S-HARDEN-1's own gate
// text). A correctly-hashed, structurally valid v2 artifact carrying the
// F20/F22 hostile operand must load Ok at the OUTER layer (the outer loader
// has no notion of SIL1's internal content) and then be rejected at the SIL1
// sub-parse specifically — never reach SiluSigmoidQ15.
static void TestArtifactRejectsHostileSigmoidLutContentThroughRealPath() {
	using namespace superslm_test;
	auto hostile_sil1 = MakeMinimalValidSil1();
	PutU32(hostile_sil1, kSil1NodesOff + 512u * 4u, static_cast<uint32_t>(INT32_MIN));
	PutU32(hostile_sil1, kSil1NodesOff + 513u * 4u, static_cast<uint32_t>(INT32_MAX));
	FixtureSection sigmoid_lut =
	    MakeSection(SslmSectionType::SigmoidLut, SslmDtype::Int32, hostile_sil1, /*alignment=*/64);
	auto built = BuildArtifact({MakeConfigSection(), sigmoid_lut});

	SslmArtifact out;
	SslmError aerr;
	auto status = SslmArtifact::OpenFromMemory(built.bytes.data(), built.bytes.size(), out, &aerr);
	CHECK_MSG(status == SslmStatus::Ok,
	          "outer artifact structurally valid (hostile bytes are a content matter, not structure): got %s",
	          SslmStatusName(status));
	if (status != SslmStatus::Ok) return;

	const SslmSectionView* section = out.Section(SslmSectionType::SigmoidLut);
	CHECK(section != nullptr);
	if (section == nullptr) return;

	SslmSigmoidLut lut;
	std::string perr;
	SslmModelStatus pstatus = ParseSigmoidLut(*section, lut, &perr);
	CHECK_MSG(pstatus == SslmModelStatus::BadSigmoidLutContent,
	          "ParseSigmoidLut on a loaded hostile SIL1 section: got %s, want BadSigmoidLutContent",
	          SslmModelStatusName(pstatus));
	CHECK_MSG(lut.values == nullptr && lut.entry_count == 0,
	          "hostile SIL1 not left at defaults on a rejected parse — a view MUST NOT be exposed");
}

// The F1 gate's own literal red cell, stated in S-HARDEN-1's plan text: "a
// correctly-hashed config-only v2 artifact expecting a specific
// missing-SigmoidLut diagnostic." Distinct from TestRejectsMissingConfigSection
// (which is missing Config, not SigmoidLut) — this is the mirror case the
// version-indexed schema exists to catch.
static void TestArtifactRejectsConfigOnlyV2MissingSigmoidLut() {
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

// ---------------------------------------------------------------------------
// S-HARDEN-1's schema-value gate (F22/F23/F24) — the parser-vs-consumer
// question RULED (D-SLM141, 2026-07-22): value validation lives in neither
// the two structural parsers nor the forward consumers — it lives in a single
// load-time schema-value gate, hosted by the new `SslmModel::Load` entry
// point, driven by a per-section domain-descriptor table (D-SLM142). The
// three structural parsers stay value-blind by design; each cell below
// therefore drives a complete, otherwise-valid v2 artifact (required Config +
// SigmoidLut, plus the one hostile section) through `SslmModel::Load` and
// asserts the load-time rejection, never the bare sub-parser (D-SLM143).
//
// None of these cells invoke the downstream kernel with the hostile operand
// (SiluSigmoidQ15, RoundingDivideByPOT, RopeApplyPair) — doing so would
// execute the actual UB the finding names, which is unsafe to run inside an
// automated suite (Debug/UBSan/ASan behavior on genuine UB is by definition
// unspecified — it could abort the whole binary rather than fail one check).
// The assertion is the contract's rejection half — `SslmModel::Load` returns
// the finding's specific `SslmModelStatus` domain code and exposes no view —
// which is what stops the hostile operand from ever reaching the kernel.
// ---------------------------------------------------------------------------

// F22: the exact operand a strike drove through SiluSigmoidQ15's three-site UB
// chain (Claude/Loki/superslm-reviewfold-strike-2026-07-21.md, Case 1 —
// "hostile mantissa", FRACTURE): m = 2^60, e = -17. Both are individually
// in-range int64 values (no field-level bound is violated — the KVC1 parser's
// own checks are all structural: magic/version/count/value_words/bounds), so
// only the load-time (m, e) domain check catches this.
static void TestKvc1RejectsHostileCompositionConstantsScale() {
	using namespace superslm_test;
	const int64_t kHostileM = INT64_C(1) << 60;
	const int64_t kHostileE = -17;
	auto kvc1 = BuildKvc1(2, {{"hostile_scale", {kHostileM, kHostileE}}});
	FixtureSection composition_constants =
	    MakeSection(SslmSectionType::CompositionConstants, SslmDtype::Raw, kvc1.bytes, /*alignment=*/64);
	auto built = BuildArtifact({MakeValidConfigSection(), MakeSigmoidLutSection(), composition_constants});

	SslmModelView view;
	std::string err;
	SslmModelStatus status = SslmModel::Load(built.bytes.data(), built.bytes.size(), view, &err);
	CHECK_MSG(status == SslmModelStatus::CompositionScaleOutOfDomain,
	          "F22 (S-HARDEN-1, D-SLM141/142): hostile CompositionConstants scale (m=2^60, e=-17) through "
	          "SslmModel::Load: got %s, want CompositionScaleOutOfDomain (%s)",
	          SslmModelStatusName(status), err.c_str());
	CHECK_MSG(!view.has_composition_constants,
	          "hostile CompositionConstants view exposed on a rejected Load — a view MUST NOT be exposed");
}

// F23: WSC1's shift column, documented [0,31] bound is a header comment only
// (docs/sslm_format.md; `RoundingDivideByPOT`'s exponent parameter). shift=32
// is the exact one-past-the-end value the finding names as UB.
static void TestWsc1RejectsHostileShiftOutOfDocumentedBound() {
	using namespace superslm_test;
	auto manifest = MakeSingleTensorManifest(superslm::kWeightScalesMagic, /*element_size=*/4, /*shape=*/{3});
	// Overwrite the one tensor's (identity, mult, shift) int32 triple directly —
	// BuildManifest's auto-filled data pattern is not a meaningful fold op, so
	// every field is set explicitly rather than patching one byte of the pattern.
	const size_t data_off = static_cast<size_t>(manifest.tensor_data_off[0]);
	PutU32(manifest.bytes, data_off + 0, 0);   // identity = 0 (not a pass-through fold)
	PutU32(manifest.bytes, data_off + 4, 1);   // mult = 1
	PutU32(manifest.bytes, data_off + 8, 32);  // shift = 32 — one past the documented [0,31] bound

	FixtureSection weight_scales =
	    MakeSection(SslmSectionType::WeightScales, SslmDtype::Int32, manifest.bytes, /*alignment=*/64);
	auto built = BuildArtifact({MakeValidConfigSection(), MakeSigmoidLutSection(), weight_scales});

	SslmModelView view;
	std::string err;
	SslmModelStatus status = SslmModel::Load(built.bytes.data(), built.bytes.size(), view, &err);
	CHECK_MSG(status == SslmModelStatus::WeightScaleShiftOutOfDomain,
	          "F23 (S-HARDEN-1, D-SLM141/142): WSC1 shift=32 (documented bound [0,31] is a header comment) "
	          "through SslmModel::Load: got %s, want WeightScaleShiftOutOfDomain (%s)",
	          SslmModelStatusName(status), err.c_str());
	CHECK_MSG(!view.has_weight_scales,
	          "hostile WeightScales view exposed on a rejected Load — a view MUST NOT be exposed");
}

// F24: ROP1's yr addition overflows on hostile cos/sin (the sweep's own
// by-reading prediction said this join was safe; execution refuted it —
// "the method lesson" per the finding record). cos = sin = INT32_MIN is the
// named hostile pair; ROP1's declared dtype is Int64 (ExpectedDtype), so the
// hostile values are carried as the int64 bit pattern of INT32_MIN.
static void TestRop1RejectsHostileCosSinPair() {
	using namespace superslm_test;
	std::vector<ManifestTensorSpec> tensors = {{"cos", {1}}, {"sin", {1}}};
	auto manifest = BuildManifest(superslm::kRopeMagic, /*element_size=*/8, tensors);
	PutU64(manifest.bytes, static_cast<size_t>(manifest.tensor_data_off[0]), static_cast<uint64_t>(int64_t{INT32_MIN}));
	PutU64(manifest.bytes, static_cast<size_t>(manifest.tensor_data_off[1]), static_cast<uint64_t>(int64_t{INT32_MIN}));

	FixtureSection rope_tables =
	    MakeSection(SslmSectionType::RopeTables, SslmDtype::Int64, manifest.bytes, /*alignment=*/64);
	auto built = BuildArtifact({MakeValidConfigSection(), MakeSigmoidLutSection(), rope_tables});

	SslmModelView view;
	std::string err;
	SslmModelStatus status = SslmModel::Load(built.bytes.data(), built.bytes.size(), view, &err);
	CHECK_MSG(status == SslmModelStatus::RopeTableEntryOutOfDomain,
	          "F24 (S-HARDEN-1, D-SLM141/142): ROP1 cos=sin=INT32_MIN through SslmModel::Load: got %s, want "
	          "RopeTableEntryOutOfDomain (%s)",
	          SslmModelStatusName(status), err.c_str());
	CHECK_MSG(!view.has_rope_tables,
	          "hostile RopeTables view exposed on a rejected Load — a view MUST NOT be exposed");
}

// ---------------------------------------------------------------------------
// Mendeleev's 2026-07-22 coverage audit of this gate (Claude/Mendeleev/
// superslm-s-harden-1-valuegate-coverage-audit-2026-07-22.md) found the three
// cells above prove only the exact adversarial operand each finding names —
// one guard sub-branch per domain, one side of each bound. The cells below
// close the audit's routed gaps: the other guard sub-branch on each domain
// check (§3.1, dims 2/4/5/11), an accept-at-bound cell per bound (dim 4, so a
// future off-by-one is caught on the side that currently passes silently),
// `Load`'s own `ArtifactRejected` container-reject branch, `Load`'s
// fail-closed reset on a STRUCTURAL sub-parse failure as opposed to a value
// rejection (§3.2, dims 7/11), `Load`'s own success path (§3.3, dim 10,
// Structural), and the dim-1 warm-object obligation for the newly validated
// `SslmModelView` (§4, T-164). Every cell drives a complete v2 artifact
// through `SslmModel::Load`, never the bare sub-parsers, per D-SLM143.
// ---------------------------------------------------------------------------

// Builds a CompositionConstants (KVC1) section with one entry named "scale"
// carrying (m, e) — the same construction as
// TestKvc1RejectsHostileCompositionConstantsScale above, factored out so the
// boundary matrix below does not repeat it once per clause.
static FixtureSection MakeKvc1CompositionSection(int64_t m, int64_t e) {
	using namespace superslm_test;
	auto kvc1 = BuildKvc1(2, {{"scale", {m, e}}});
	return MakeSection(SslmSectionType::CompositionConstants, SslmDtype::Raw, kvc1.bytes, /*alignment=*/64);
}

// Builds a WeightScales (WSC1) section with one (identity, mult, shift) row —
// the same construction as TestWsc1RejectsHostileShiftOutOfDocumentedBound
// above, factored out for the boundary matrix below.
static FixtureSection MakeWsc1Section(int32_t identity, int32_t mult, int32_t shift) {
	using namespace superslm_test;
	auto manifest = MakeSingleTensorManifest(superslm::kWeightScalesMagic, /*element_size=*/4, /*shape=*/{3});
	const size_t data_off = static_cast<size_t>(manifest.tensor_data_off[0]);
	PutU32(manifest.bytes, data_off + 0, static_cast<uint32_t>(identity));
	PutU32(manifest.bytes, data_off + 4, static_cast<uint32_t>(mult));
	PutU32(manifest.bytes, data_off + 8, static_cast<uint32_t>(shift));
	return MakeSection(SslmSectionType::WeightScales, SslmDtype::Int32, manifest.bytes, /*alignment=*/64);
}

// Builds a RopeTables (ROP1) section with "cos"/"sin" single-element tensors —
// the same construction as TestRop1RejectsHostileCosSinPair above, factored
// out for the boundary matrix below.
static FixtureSection MakeRop1Section(int64_t cos_v, int64_t sin_v) {
	using namespace superslm_test;
	std::vector<ManifestTensorSpec> tensors = {{"cos", {1}}, {"sin", {1}}};
	auto manifest = BuildManifest(superslm::kRopeMagic, /*element_size=*/8, tensors);
	PutU64(manifest.bytes, static_cast<size_t>(manifest.tensor_data_off[0]), static_cast<uint64_t>(cos_v));
	PutU64(manifest.bytes, static_cast<size_t>(manifest.tensor_data_off[1]), static_cast<uint64_t>(sin_v));
	return MakeSection(SslmSectionType::RopeTables, SslmDtype::Int64, manifest.bytes, /*alignment=*/64);
}

// Little-endian byte assemblers over a raw section pointer (never a cast over
// unaligned/untrusted bytes) — the same discipline model.cpp's own RdI32/RdI64
// use, reimplemented independently here so a §3.3/§4 feature-oracle assertion
// is never checked against the code under test's own reader.
static int32_t ReadRawI32LE(const uint8_t* p) {
	uint32_t v = uint32_t(p[0]) | (uint32_t(p[1]) << 8) | (uint32_t(p[2]) << 16) | (uint32_t(p[3]) << 24);
	return static_cast<int32_t>(v);
}
static int64_t ReadRawI64LE(const uint8_t* p) {
	uint64_t v = 0;
	for (int i = 0; i < 8; ++i) v |= static_cast<uint64_t>(p[i]) << (8 * i);
	return static_cast<int64_t>(v);
}

// A complete, fully in-domain v2 artifact: Config, SigmoidLut, and one each of
// CompositionConstants/WeightScales/RopeTables with every gated field inside
// its stated domain — the §3.3 success-path fixture and the §4 warm-object
// base case (Mendeleev's routing: "should be authored first so §4's cell can
// build on it rather than duplicate it"). Values are arbitrary but
// deliberately non-trivial (not 0, not a bound) so the feature-oracle checks
// below cannot pass on a coincidentally-zeroed read.
static superslm_test::BuiltArtifact BuildFullyValidV2ArtifactForLoad() {
	using namespace superslm_test;
	FixtureSection composition_constants = MakeKvc1CompositionSection(/*m=*/1000000, /*e=*/0);
	FixtureSection weight_scales = MakeWsc1Section(/*identity=*/1, /*mult=*/12345, /*shift=*/10);
	FixtureSection rope_tables = MakeRop1Section(/*cos=*/500000000, /*sin=*/-500000000);
	return BuildArtifact({MakeValidConfigSection(), MakeSigmoidLutSection(), composition_constants,
	                      weight_scales, rope_tables});
}

// --- §3.1 boundary/guard matrix: the bare reject-just-past clauses (Mendeleev
//     finding 2). Each is the ONE value past the stated bound on the side no
//     existing cell reaches; the companion field is held safely in-domain so
//     the assertion isolates exactly the one clause named. ---

static void TestKvc1RejectsCompositionScaleMUnderNoUbFloor() {
	using namespace superslm_test;
	auto built = BuildArtifact({MakeValidConfigSection(), MakeSigmoidLutSection(),
	                            MakeKvc1CompositionSection(/*m=*/INT64_C(-2147483648), /*e=*/0)});
	SslmModelView view;
	std::string err;
	SslmModelStatus status = SslmModel::Load(built.bytes.data(), built.bytes.size(), view, &err);
	CHECK_MSG(status == SslmModelStatus::CompositionScaleOutOfDomain,
	          "KVC1 m=-2147483648 (one past the no-UB floor's lower bound -2147483647): got %s, want "
	          "CompositionScaleOutOfDomain (%s)",
	          SslmModelStatusName(status), err.c_str());
	CHECK(!view.has_composition_constants);
}

static void TestKvc1RejectsCompositionScaleEUnderNoUbFloor() {
	using namespace superslm_test;
	auto built = BuildArtifact({MakeValidConfigSection(), MakeSigmoidLutSection(),
	                            MakeKvc1CompositionSection(/*m=*/0, /*e=*/-81)});
	SslmModelView view;
	std::string err;
	SslmModelStatus status = SslmModel::Load(built.bytes.data(), built.bytes.size(), view, &err);
	CHECK_MSG(status == SslmModelStatus::CompositionScaleOutOfDomain,
	          "KVC1 e=-81 (one past the no-UB floor's lower bound -80): got %s, want "
	          "CompositionScaleOutOfDomain (%s)",
	          SslmModelStatusName(status), err.c_str());
	CHECK(!view.has_composition_constants);
}

static void TestKvc1RejectsCompositionScaleEOverNoUbFloor() {
	using namespace superslm_test;
	auto built = BuildArtifact({MakeValidConfigSection(), MakeSigmoidLutSection(),
	                            MakeKvc1CompositionSection(/*m=*/0, /*e=*/8)});
	SslmModelView view;
	std::string err;
	SslmModelStatus status = SslmModel::Load(built.bytes.data(), built.bytes.size(), view, &err);
	CHECK_MSG(status == SslmModelStatus::CompositionScaleOutOfDomain,
	          "KVC1 e=8 (one past the no-UB floor's upper bound 7): got %s, want "
	          "CompositionScaleOutOfDomain (%s)",
	          SslmModelStatusName(status), err.c_str());
	CHECK(!view.has_composition_constants);
}

static void TestWsc1RejectsIdentityUnderDocumentedBound() {
	using namespace superslm_test;
	auto built = BuildArtifact({MakeValidConfigSection(), MakeSigmoidLutSection(),
	                            MakeWsc1Section(/*identity=*/-1, /*mult=*/1, /*shift=*/0)});
	SslmModelView view;
	std::string err;
	SslmModelStatus status = SslmModel::Load(built.bytes.data(), built.bytes.size(), view, &err);
	CHECK_MSG(status == SslmModelStatus::WeightScaleIdentityNotBool,
	          "WSC1 identity=-1 (below the documented {0,1} bound): got %s, want "
	          "WeightScaleIdentityNotBool (%s)",
	          SslmModelStatusName(status), err.c_str());
	CHECK(!view.has_weight_scales);
}

static void TestWsc1RejectsIdentityOverDocumentedBound() {
	using namespace superslm_test;
	auto built = BuildArtifact({MakeValidConfigSection(), MakeSigmoidLutSection(),
	                            MakeWsc1Section(/*identity=*/2, /*mult=*/1, /*shift=*/0)});
	SslmModelView view;
	std::string err;
	SslmModelStatus status = SslmModel::Load(built.bytes.data(), built.bytes.size(), view, &err);
	CHECK_MSG(status == SslmModelStatus::WeightScaleIdentityNotBool,
	          "WSC1 identity=2 (above the documented {0,1} bound): got %s, want "
	          "WeightScaleIdentityNotBool (%s)",
	          SslmModelStatusName(status), err.c_str());
	CHECK(!view.has_weight_scales);
}

static void TestWsc1RejectsShiftUnderDocumentedBound() {
	using namespace superslm_test;
	auto built = BuildArtifact({MakeValidConfigSection(), MakeSigmoidLutSection(),
	                            MakeWsc1Section(/*identity=*/0, /*mult=*/1, /*shift=*/-1)});
	SslmModelView view;
	std::string err;
	SslmModelStatus status = SslmModel::Load(built.bytes.data(), built.bytes.size(), view, &err);
	CHECK_MSG(status == SslmModelStatus::WeightScaleShiftOutOfDomain,
	          "WSC1 shift=-1 (one past the documented [0,31] bound's lower side): got %s, want "
	          "WeightScaleShiftOutOfDomain (%s)",
	          SslmModelStatusName(status), err.c_str());
	CHECK(!view.has_weight_scales);
}

static void TestRop1RejectsElementOverDocumentedBound() {
	using namespace superslm_test;
	auto built = BuildArtifact({MakeValidConfigSection(), MakeSigmoidLutSection(),
	                            MakeRop1Section(/*cos_v=*/INT64_C(1073741825), /*sin_v=*/0)});
	SslmModelView view;
	std::string err;
	SslmModelStatus status = SslmModel::Load(built.bytes.data(), built.bytes.size(), view, &err);
	CHECK_MSG(status == SslmModelStatus::RopeTableEntryOutOfDomain,
	          "ROP1 element=1073741825 (one past the [-2^30,2^30] bound's upper side): got %s, want "
	          "RopeTableEntryOutOfDomain (%s)",
	          SslmModelStatusName(status), err.c_str());
	CHECK(!view.has_rope_tables);
}

// --- §3.1 boundary/guard matrix: the accept-at-bound companions (Mendeleev
//     finding 3, dim 4). Each is the value EXACTLY at the stated bound —
//     distinct from the reject-just-past cell above it by exactly one — so a
//     future off-by-one in either direction is caught on the side that
//     currently passes silently. Reference/exact-value oracle: status must be
//     Ok and the section's own view must be exposed. ---

static void TestKvc1AcceptsCompositionScaleMAtLowerNoUbFloor() {
	using namespace superslm_test;
	auto built = BuildArtifact({MakeValidConfigSection(), MakeSigmoidLutSection(),
	                            MakeKvc1CompositionSection(/*m=*/INT64_C(-2147483647), /*e=*/0)});
	SslmModelView view;
	std::string err;
	SslmModelStatus status = SslmModel::Load(built.bytes.data(), built.bytes.size(), view, &err);
	CHECK_MSG(status == SslmModelStatus::Ok, "KVC1 m=-2147483647 (exactly the no-UB floor's lower "
	          "bound) must be accepted: got %s (%s)", SslmModelStatusName(status), err.c_str());
	CHECK(view.has_composition_constants);
}

static void TestKvc1AcceptsCompositionScaleEAtLowerNoUbFloor() {
	using namespace superslm_test;
	auto built = BuildArtifact({MakeValidConfigSection(), MakeSigmoidLutSection(),
	                            MakeKvc1CompositionSection(/*m=*/0, /*e=*/-80)});
	SslmModelView view;
	std::string err;
	SslmModelStatus status = SslmModel::Load(built.bytes.data(), built.bytes.size(), view, &err);
	CHECK_MSG(status == SslmModelStatus::Ok, "KVC1 e=-80 (exactly the no-UB floor's lower bound) "
	          "must be accepted: got %s (%s)", SslmModelStatusName(status), err.c_str());
	CHECK(view.has_composition_constants);
}

static void TestKvc1AcceptsCompositionScaleEAtUpperNoUbFloor() {
	using namespace superslm_test;
	auto built = BuildArtifact({MakeValidConfigSection(), MakeSigmoidLutSection(),
	                            MakeKvc1CompositionSection(/*m=*/0, /*e=*/7)});
	SslmModelView view;
	std::string err;
	SslmModelStatus status = SslmModel::Load(built.bytes.data(), built.bytes.size(), view, &err);
	CHECK_MSG(status == SslmModelStatus::Ok, "KVC1 e=7 (exactly the no-UB floor's upper bound) must "
	          "be accepted: got %s (%s)", SslmModelStatusName(status), err.c_str());
	CHECK(view.has_composition_constants);
}

static void TestWsc1AcceptsIdentityAtZero() {
	using namespace superslm_test;
	auto built = BuildArtifact({MakeValidConfigSection(), MakeSigmoidLutSection(),
	                            MakeWsc1Section(/*identity=*/0, /*mult=*/1, /*shift=*/0)});
	SslmModelView view;
	std::string err;
	SslmModelStatus status = SslmModel::Load(built.bytes.data(), built.bytes.size(), view, &err);
	CHECK_MSG(status == SslmModelStatus::Ok, "WSC1 identity=0 must be accepted: got %s (%s)",
	          SslmModelStatusName(status), err.c_str());
	CHECK(view.has_weight_scales);
}

static void TestWsc1AcceptsIdentityAtOne() {
	using namespace superslm_test;
	auto built = BuildArtifact({MakeValidConfigSection(), MakeSigmoidLutSection(),
	                            MakeWsc1Section(/*identity=*/1, /*mult=*/1, /*shift=*/0)});
	SslmModelView view;
	std::string err;
	SslmModelStatus status = SslmModel::Load(built.bytes.data(), built.bytes.size(), view, &err);
	CHECK_MSG(status == SslmModelStatus::Ok, "WSC1 identity=1 must be accepted: got %s (%s)",
	          SslmModelStatusName(status), err.c_str());
	CHECK(view.has_weight_scales);
}

static void TestWsc1AcceptsShiftAtLowerBound() {
	using namespace superslm_test;
	auto built = BuildArtifact({MakeValidConfigSection(), MakeSigmoidLutSection(),
	                            MakeWsc1Section(/*identity=*/0, /*mult=*/1, /*shift=*/0)});
	SslmModelView view;
	std::string err;
	SslmModelStatus status = SslmModel::Load(built.bytes.data(), built.bytes.size(), view, &err);
	CHECK_MSG(status == SslmModelStatus::Ok, "WSC1 shift=0 (exactly the documented lower bound) "
	          "must be accepted: got %s (%s)", SslmModelStatusName(status), err.c_str());
	CHECK(view.has_weight_scales);
}

static void TestWsc1AcceptsShiftAtUpperBound() {
	using namespace superslm_test;
	auto built = BuildArtifact({MakeValidConfigSection(), MakeSigmoidLutSection(),
	                            MakeWsc1Section(/*identity=*/0, /*mult=*/1, /*shift=*/31)});
	SslmModelView view;
	std::string err;
	SslmModelStatus status = SslmModel::Load(built.bytes.data(), built.bytes.size(), view, &err);
	CHECK_MSG(status == SslmModelStatus::Ok, "WSC1 shift=31 (exactly the documented upper bound) "
	          "must be accepted: got %s (%s)", SslmModelStatusName(status), err.c_str());
	CHECK(view.has_weight_scales);
}

static void TestRop1AcceptsElementAtPositiveBound() {
	using namespace superslm_test;
	auto built = BuildArtifact({MakeValidConfigSection(), MakeSigmoidLutSection(),
	                            MakeRop1Section(/*cos_v=*/INT64_C(1073741824), /*sin_v=*/0)});
	SslmModelView view;
	std::string err;
	SslmModelStatus status = SslmModel::Load(built.bytes.data(), built.bytes.size(), view, &err);
	CHECK_MSG(status == SslmModelStatus::Ok, "ROP1 element=2^30 (exactly the upper bound) must be "
	          "accepted: got %s (%s)", SslmModelStatusName(status), err.c_str());
	CHECK(view.has_rope_tables);
}

static void TestRop1AcceptsElementAtNegativeBound() {
	using namespace superslm_test;
	auto built = BuildArtifact({MakeValidConfigSection(), MakeSigmoidLutSection(),
	                            MakeRop1Section(/*cos_v=*/INT64_C(-1073741824), /*sin_v=*/0)});
	SslmModelView view;
	std::string err;
	SslmModelStatus status = SslmModel::Load(built.bytes.data(), built.bytes.size(), view, &err);
	CHECK_MSG(status == SslmModelStatus::Ok, "ROP1 element=-2^30 (exactly the lower bound) must be "
	          "accepted: got %s (%s)", SslmModelStatusName(status), err.c_str());
	CHECK(view.has_rope_tables);
}

// --- §3.1: Load's own container-reject branch (Mendeleev row 11). No cell
//     called SslmModel::Load on a structurally invalid artifact before this
//     one — every existing structural-rejection cell calls
//     SslmArtifact::OpenFromMemory directly. ---

static void TestLoadRejectsStructurallyInvalidArtifactAsArtifactRejected() {
	using namespace superslm_test;
	auto built = BuildArtifact({MakeValidConfigSection(), MakeSigmoidLutSection()});
	built.bytes[0] = 'X';  // was 'S' — corrupts the header magic, a container-level defect
	RecomputeIntegrityHash(built.bytes);

	SslmModelView view;
	std::string err;
	SslmModelStatus status = SslmModel::Load(built.bytes.data(), built.bytes.size(), view, &err);
	CHECK_MSG(status == SslmModelStatus::ArtifactRejected,
	          "a structurally invalid artifact through SslmModel::Load: got %s, want ArtifactRejected (%s)",
	          SslmModelStatusName(status), err.c_str());
	CHECK(!view.has_config);
	CHECK(!view.has_sigmoid_lut);
	CHECK(!view.has_composition_constants);
	CHECK(!view.has_weight_scales);
	CHECK(!view.has_rope_tables);
}

// --- §3.2: Load's fail-closed reset on a STRUCTURAL sub-parse failure, as
//     opposed to a value-domain rejection (dims 7/11). The three existing
//     "!view.has_*" assertions above are all on the value-rejection path;
//     none drives a structurally-hostile section through Load. A WeightScales
//     tensor with a corrupted manifest magic is reused here (the same defect
//     class TestManifestRejectsBadMagicWgt/Bia/Rop already prove at the bare
//     sub-parser) so the fixture's own values stay in-domain — the rejection
//     this cell targets is the STRUCTURAL one, never the value gate's. ---

static void TestLoadFailsClosedOnStructuralSubParseFailureBeforeValueGate() {
	using namespace superslm_test;
	// An otherwise-valid WeightScales manifest (in-domain identity/mult/shift)
	// with its own magic corrupted, so SslmTensorManifest::Parse itself rejects
	// it (BadManifestMagic) before ValidateSectionValues ever runs.
	auto manifest = MakeSingleTensorManifest(superslm::kWeightScalesMagic, /*element_size=*/4, /*shape=*/{3});
	const size_t data_off = static_cast<size_t>(manifest.tensor_data_off[0]);
	PutU32(manifest.bytes, data_off + 0, 0);  // identity — in-domain
	PutU32(manifest.bytes, data_off + 4, 1);  // mult
	PutU32(manifest.bytes, data_off + 8, 0);  // shift — in-domain
	manifest.bytes[0] = 'X';                  // was 'W' of "WSC1" — corrupts the manifest's own magic

	FixtureSection weight_scales =
	    MakeSection(SslmSectionType::WeightScales, SslmDtype::Int32, manifest.bytes, /*alignment=*/64);
	auto built = BuildArtifact({MakeValidConfigSection(), MakeSigmoidLutSection(), weight_scales});

	SslmModelView view;
	std::string err;
	SslmModelStatus status = SslmModel::Load(built.bytes.data(), built.bytes.size(), view, &err);
	CHECK_MSG(status == SslmModelStatus::BadManifestMagic,
	          "a WeightScales section whose OWN manifest magic is corrupt (a structural defect, not a "
	          "value-domain one) through SslmModel::Load: got %s, want BadManifestMagic (the structural "
	          "sub-parse's own status, never a value-domain code and never ArtifactRejected) (%s)",
	          SslmModelStatusName(status), err.c_str());
	// Load's loop resets `out` unconditionally on ANY section's structural
	// sub-parse failure — Config and SigmoidLut, which parsed fine before
	// WeightScales was reached, must be un-exposed too (fail-closed, never a
	// partial view).
	CHECK_MSG(!view.has_config, "structural failure on a LATER section must still un-expose an EARLIER "
	          "section's already-successful sub-parse (Load's fail-closed reset)");
	CHECK(!view.has_sigmoid_lut);
	CHECK(!view.has_weight_scales);
}

// --- §3.3: Load's own success path (dim 10, Structural). Every existing call
//     to SslmModel::Load in this suite asserts a rejection status; none
//     asserts Ok with a populated view — the entry point's own achievement
//     claim ("exposes the composed ModelView only on full success", D-SLM141)
//     had zero proving test before this cell. Feature oracle: the exposed
//     values on every populated view are checked against what the fixture
//     wrote, not against themselves. ---

static void TestLoadSucceedsAndExposesFullyPopulatedViewOnValidArtifact() {
	using namespace superslm_test;
	auto built = BuildFullyValidV2ArtifactForLoad();

	SslmModelView view;
	std::string err;
	SslmModelStatus status = SslmModel::Load(built.bytes.data(), built.bytes.size(), view, &err);
	CHECK_MSG(status == SslmModelStatus::Ok,
	          "a fully valid v2 artifact (every gated field inside its domain) through SslmModel::Load: "
	          "got %s, want Ok (%s)", SslmModelStatusName(status), err.c_str());
	if (status != SslmModelStatus::Ok) return;

	CHECK(view.has_config);
	CHECK(view.has_sigmoid_lut);
	CHECK(view.has_composition_constants);
	CHECK(view.has_weight_scales);
	CHECK(view.has_rope_tables);

	// Config — spot-check against MakeValidConfigSection's Cfg1Spec{} defaults.
	CHECK(view.config.hidden_size == 4096);
	CHECK(view.config.num_hidden_layers == 32);
	CHECK(view.config.vocab_size == 32001);
	CHECK(view.config.tie_word_embeddings == true);
	CHECK(view.config.kv_precision == SslmKvPrecision::Int16);
	CHECK(view.config.rope_theta == 1000000.0);

	// SigmoidLut — the pinned canonical table (MakeSigmoidLutSection's content).
	CHECK(view.sigmoid_lut.entry_count == kSigmoidLutEntries);
	CHECK(SigmoidLutValue(view.sigmoid_lut, 0) == kSiluLutGoldenTable[0]);
	CHECK(SigmoidLutValue(view.sigmoid_lut, kSiluLutN) ==
	      kSiluLutGoldenTable[static_cast<size_t>(kSiluLutN)]);

	// CompositionConstants — the "scale" entry's (m, e) written by
	// BuildFullyValidV2ArtifactForLoad.
	const SslmConstantEntry* scale = view.composition_constants.Entry("scale");
	CHECK_MSG(scale != nullptr, "the CompositionConstants view exposes no \"scale\" entry");
	if (scale != nullptr) {
		CHECK(SslmKeyedConstants::Value(*scale, 0) == 1000000);
		CHECK(SslmKeyedConstants::Value(*scale, 1) == 0);
	}

	// WeightScales — the "t0" tensor's (identity, mult, shift) triple.
	const SslmTensorView* wsc = view.weight_scales.Tensor("t0");
	CHECK_MSG(wsc != nullptr, "the WeightScales view exposes no \"t0\" tensor");
	if (wsc != nullptr) {
		CHECK(ReadRawI32LE(wsc->data + 0) == 1);
		CHECK(ReadRawI32LE(wsc->data + 4) == 12345);
		CHECK(ReadRawI32LE(wsc->data + 8) == 10);
	}

	// RopeTables — the "cos"/"sin" tensors' single elements.
	const SslmTensorView* cos = view.rope_tables.Tensor("cos");
	const SslmTensorView* sin = view.rope_tables.Tensor("sin");
	CHECK_MSG(cos != nullptr && sin != nullptr, "the RopeTables view is missing \"cos\" or \"sin\"");
	if (cos != nullptr) CHECK(ReadRawI64LE(cos->data) == 500000000);
	if (sin != nullptr) CHECK(ReadRawI64LE(sin->data) == -500000000);
}

// --- §4: the dim-1 warm-object obligation for the newly validated
//     SslmModelView (T-164), on the existing TestSil1WarmObjectRepeatedReadsShowNoDrift
//     precedent. Two clauses: repeated reads never drift (this cell), and a
//     second Load on the identical bytes reproduces the first call's result
//     bit-for-bit (the next cell — re-open idempotence, which the SIL1
//     precedent does not itself cover since it never re-parses). The third
//     clause ("a view built from a rejected artifact is never exposed") is
//     already discharged by every "!view.has_*" assertion above. ---

static void TestLoadComposedViewRepeatedReadsShowNoDrift() {
	using namespace superslm_test;
	auto built = BuildFullyValidV2ArtifactForLoad();

	SslmModelView view;
	std::string err;
	SslmModelStatus status = SslmModel::Load(built.bytes.data(), built.bytes.size(), view, &err);
	CHECK_MSG(status == SslmModelStatus::Ok, "warm-object fixture failed to load: got %s (%s)",
	          SslmModelStatusName(status), err.c_str());
	if (status != SslmModelStatus::Ok) return;

	const SslmConstantEntry* scale = view.composition_constants.Entry("scale");
	const SslmTensorView* wsc = view.weight_scales.Tensor("t0");
	const SslmTensorView* cos = view.rope_tables.Tensor("cos");
	CHECK_MSG(scale != nullptr && wsc != nullptr && cos != nullptr,
	          "warm-object fixture's view is missing an expected entry/tensor");
	if (scale == nullptr || wsc == nullptr || cos == nullptr) return;

	const int32_t first_hidden_size = static_cast<int32_t>(view.config.hidden_size);
	const int32_t first_lut_node0 = SigmoidLutValue(view.sigmoid_lut, 0);
	const int64_t first_scale_m = SslmKeyedConstants::Value(*scale, 0);
	const int32_t first_wsc_shift = ReadRawI32LE(wsc->data + 8);
	const int64_t first_cos = ReadRawI64LE(cos->data);

	int drift = 0;
	for (int pass = 0; pass < 1000; ++pass) {
		if (static_cast<int32_t>(view.config.hidden_size) != first_hidden_size) ++drift;
		if (SigmoidLutValue(view.sigmoid_lut, 0) != first_lut_node0) ++drift;
		if (SslmKeyedConstants::Value(*scale, 0) != first_scale_m) ++drift;
		if (ReadRawI32LE(wsc->data + 8) != first_wsc_shift) ++drift;
		if (ReadRawI64LE(cos->data) != first_cos) ++drift;
	}
	CHECK_MSG(drift == 0,
	          "%d of 5000 repeated reads across the Load-composed view's five section kinds drifted "
	          "from the first read (no re-derivation expected)",
	          drift);
}

static void TestLoadComposedViewReopenIsIdempotent() {
	using namespace superslm_test;
	auto built = BuildFullyValidV2ArtifactForLoad();

	SslmModelView view1;
	std::string err1;
	SslmModelStatus status1 = SslmModel::Load(built.bytes.data(), built.bytes.size(), view1, &err1);
	CHECK_MSG(status1 == SslmModelStatus::Ok, "first Load of the warm-object fixture failed: got %s (%s)",
	          SslmModelStatusName(status1), err1.c_str());

	// A second Load on the IDENTICAL byte buffer (no re-encode, no mutation) —
	// re-open idempotence, distinct from repeated reads of one already-parsed
	// view: this re-runs the whole parse-and-validate pass a second time.
	SslmModelView view2;
	std::string err2;
	SslmModelStatus status2 = SslmModel::Load(built.bytes.data(), built.bytes.size(), view2, &err2);
	CHECK_MSG(status2 == SslmModelStatus::Ok, "second Load of the identical bytes failed: got %s (%s)",
	          SslmModelStatusName(status2), err2.c_str());
	if (status1 != SslmModelStatus::Ok || status2 != SslmModelStatus::Ok) return;

	CHECK(view1.config.hidden_size == view2.config.hidden_size);
	CHECK(view1.config.vocab_size == view2.config.vocab_size);
	CHECK(SigmoidLutValue(view1.sigmoid_lut, 0) == SigmoidLutValue(view2.sigmoid_lut, 0));
	CHECK(SigmoidLutValue(view1.sigmoid_lut, kSiluLutN) == SigmoidLutValue(view2.sigmoid_lut, kSiluLutN));

	const SslmConstantEntry* scale1 = view1.composition_constants.Entry("scale");
	const SslmConstantEntry* scale2 = view2.composition_constants.Entry("scale");
	CHECK_MSG(scale1 != nullptr && scale2 != nullptr, "one of the two Loads exposes no \"scale\" entry");
	if (scale1 != nullptr && scale2 != nullptr) {
		CHECK(SslmKeyedConstants::Value(*scale1, 0) == SslmKeyedConstants::Value(*scale2, 0));
		CHECK(SslmKeyedConstants::Value(*scale1, 1) == SslmKeyedConstants::Value(*scale2, 1));
	}

	const SslmTensorView* wsc1 = view1.weight_scales.Tensor("t0");
	const SslmTensorView* wsc2 = view2.weight_scales.Tensor("t0");
	CHECK_MSG(wsc1 != nullptr && wsc2 != nullptr, "one of the two Loads exposes no \"t0\" tensor");
	if (wsc1 != nullptr && wsc2 != nullptr) {
		CHECK(ReadRawI32LE(wsc1->data + 0) == ReadRawI32LE(wsc2->data + 0));
		CHECK(ReadRawI32LE(wsc1->data + 4) == ReadRawI32LE(wsc2->data + 4));
		CHECK(ReadRawI32LE(wsc1->data + 8) == ReadRawI32LE(wsc2->data + 8));
	}

	const SslmTensorView* cos1 = view1.rope_tables.Tensor("cos");
	const SslmTensorView* cos2 = view2.rope_tables.Tensor("cos");
	CHECK_MSG(cos1 != nullptr && cos2 != nullptr, "one of the two Loads exposes no \"cos\" tensor");
	if (cos1 != nullptr && cos2 != nullptr) {
		CHECK(ReadRawI64LE(cos1->data) == ReadRawI64LE(cos2->data));
	}
}

// T-402 (Poirot Observation 2): TestLoadComposedViewReopenIsIdempotent above loads into two
// FRESH views. This pins the other half — Load onto an ALREADY-POPULATED view object. The
// second Load must fully RESET the target to the freshly-parsed state; an append-without-clear
// would leave the first Load's entries in place (doubling the constants container), which a
// value-only check on Entry("scale") would not catch because the lookup would still resolve.
static void TestLoadOntoPopulatedViewResetsEntries() {
	using namespace superslm_test;
	auto built = BuildFullyValidV2ArtifactForLoad();

	// A fresh-view Load of the identical bytes: the reference the populated-target Load must match.
	SslmModelView fresh;
	std::string errf;
	SslmModelStatus sf = SslmModel::Load(built.bytes.data(), built.bytes.size(), fresh, &errf);
	CHECK_MSG(sf == SslmModelStatus::Ok, "reference fresh Load failed: got %s (%s)",
	          SslmModelStatusName(sf), errf.c_str());

	// ONE view, loaded twice: the second Load's target is already populated.
	SslmModelView view;
	std::string err;
	SslmModelStatus s1 = SslmModel::Load(built.bytes.data(), built.bytes.size(), view, &err);
	CHECK_MSG(s1 == SslmModelStatus::Ok, "first Load into the reused view failed: got %s (%s)",
	          SslmModelStatusName(s1), err.c_str());
	SslmModelStatus s2 = SslmModel::Load(built.bytes.data(), built.bytes.size(), view, &err);
	CHECK_MSG(s2 == SslmModelStatus::Ok, "re-Load onto a populated view failed: got %s (%s)",
	          SslmModelStatusName(s2), err.c_str());
	if (sf != SslmModelStatus::Ok || s1 != SslmModelStatus::Ok || s2 != SslmModelStatus::Ok) return;

	// Entry COUNT, not just lookup: a missing reset would double the populated container.
	CHECK_MSG(view.composition_constants.Entries().size() == fresh.composition_constants.Entries().size(),
	          "re-Load onto a populated view did not reset composition_constants: %zu entries vs a fresh %zu",
	          view.composition_constants.Entries().size(), fresh.composition_constants.Entries().size());

	// And the values still match a fresh parse (reset produced the right state, not an empty one).
	CHECK(view.config.hidden_size == fresh.config.hidden_size);
	CHECK(view.config.vocab_size == fresh.config.vocab_size);
	const SslmConstantEntry* sp = view.composition_constants.Entry("scale");
	const SslmConstantEntry* sr = fresh.composition_constants.Entry("scale");
	CHECK_MSG(sp != nullptr && sr != nullptr, "populated-target re-Load lost the \"scale\" entry");
	if (sp != nullptr && sr != nullptr) {
		CHECK(SslmKeyedConstants::Value(*sp, 0) == SslmKeyedConstants::Value(*sr, 0));
		CHECK(SslmKeyedConstants::Value(*sp, 1) == SslmKeyedConstants::Value(*sr, 1));
	}
}

// ---------------------------------------------------------------------------
// T-403 regression PIN (SslmModelView lifetime-contract fix, the design at
// Claude/Vitruvius/SuperSLM_Load_UAF_LifetimeFix_Design-2026-07-22.md §7 plus
// Charpy's 2026-07-22 temper §6 amendment). `SslmModel::Load` previously
// constructed a function-local SslmArtifact and populated the returned view
// with pointers into it, freed the instant Load returned. Every existing
// happy-path cell above reads a Load-returned view immediately, with no
// intervening heap allocation, so none of them can expose a freed backing
// store. These cells close that gap: a read after deliberate intervening
// allocation (§7.2), and the moved-from object's inertness (the amendment's
// §6 cell) — both against an independent, fixture-known oracle, never a
// self-comparison.
// ---------------------------------------------------------------------------

// SslmModelView must never be copyable: its backing store (like SslmArtifact's
// own) is move-only, so a copyable view of borrowed bytes would be a lie. True
// on both the pre-fix and post-fix type (TokenizerView's own deleted copy
// constructor already forced this — Charpy's 2026-07-22 temper, Note finding);
// stated here as a standing structural guarantee the rest of this PIN assumes.
static_assert(!std::is_copy_constructible_v<SslmModelView>,
              "SslmModelView must stay non-copyable — its backing store is move-only");

static superslm_test::BuiltArtifact BuildFullyValidV2ArtifactForLoadWithTokenizer() {
	using namespace superslm_test;
	auto tok1 = MakeMinimalValidTok1();  // vocab_count == 5; Encode("cat") == {3, 2} (pinned above, line ~1096)
	auto uni1 = MakeMinimalValidUni1();
	Cfg1Spec cfg;
	cfg.vocab_size = tok1.layout.vocab_count;  // join CFG1.vocab_size to TOK1.vocab_count (F18)
	FixtureSection composition_constants = MakeKvc1CompositionSection(/*m=*/1000000, /*e=*/0);
	FixtureSection weight_scales = MakeWsc1Section(/*identity=*/1, /*mult=*/12345, /*shift=*/10);
	FixtureSection rope_tables = MakeRop1Section(/*cos=*/500000000, /*sin=*/-500000000);
	return BuildArtifact({MakeSection(SslmSectionType::Config, SslmDtype::Raw, BuildCfg1(cfg)),
	                      MakeSigmoidLutSection(), composition_constants, weight_scales, rope_tables,
	                      MakeSection(SslmSectionType::Tokenizer, SslmDtype::Raw, tok1.bytes),
	                      MakeSection(SslmSectionType::UnicodeTables, SslmDtype::Raw, uni1.bytes)});
}

// §7.2 — the behavioral layer. RED on the pre-fix Load (freed function-local
// SslmArtifact; the intervening allocation below reclaims and poisons the
// freed block, reproducing the sslm_verify garble witness deterministically);
// GREEN once SslmModelView owns its backing store.
static void TestLoadReturnedViewSurvivesInterveningHeapAllocationBeforeRead() {
	using namespace superslm_test;
	auto built = BuildFullyValidV2ArtifactForLoadWithTokenizer();

	SslmModelView view;
	std::string err;
	SslmModelStatus status = SslmModel::Load(built.bytes.data(), built.bytes.size(), view, &err);
	CHECK_MSG(status == SslmModelStatus::Ok,
	          "the PIN's fully valid v2+tokenizer artifact through SslmModel::Load: got %s (%s)",
	          SslmModelStatusName(status), err.c_str());
	if (status != SslmModelStatus::Ok) return;

	// Intervening heap churn, sized to reclaim and overwrite Load's freed
	// function-local backing store under the pre-fix code: many blocks the same
	// size as the artifact's own byte length, retained so the allocator actually
	// hands the freed block back and the runtime stamps it with 0xA5. Under the
	// fix, `view`'s own owned backing store is untouched by any of this churn.
	std::vector<std::vector<uint8_t>> churn;
	churn.reserve(4096);
	for (int i = 0; i < 4096; ++i) {
		churn.emplace_back(built.bytes.size(), static_cast<uint8_t>(0xA5));
	}

	// Every pointer-bearing field, read AFTER the churn above, asserted against
	// the FIXTURE's own known written values — an independent oracle, never a
	// self-comparison.
	const SslmTensorView* wsc = view.weight_scales.Tensor("t0");
	CHECK_MSG(wsc != nullptr, "the WeightScales view exposes no \"t0\" tensor after intervening allocation");
	if (wsc != nullptr) {
		CHECK(ReadRawI32LE(wsc->data + 0) == 1);
		CHECK(ReadRawI32LE(wsc->data + 4) == 12345);
		CHECK(ReadRawI32LE(wsc->data + 8) == 10);
	}

	const SslmTensorView* cos = view.rope_tables.Tensor("cos");
	CHECK_MSG(cos != nullptr, "the RopeTables view exposes no \"cos\" tensor after intervening allocation");
	if (cos != nullptr) CHECK(ReadRawI64LE(cos->data) == 500000000);

	const SslmConstantEntry* scale = view.composition_constants.Entry("scale");
	CHECK_MSG(scale != nullptr,
	          "the CompositionConstants view exposes no \"scale\" entry after intervening allocation");
	if (scale != nullptr) {
		CHECK(SslmKeyedConstants::Value(*scale, 0) == 1000000);
		CHECK(SslmKeyedConstants::Value(*scale, 1) == 0);
	}

	CHECK(SigmoidLutValue(view.sigmoid_lut, 0) == kSiluLutGoldenTable[0]);
	CHECK(SigmoidLutValue(view.sigmoid_lut, kSiluLutN) == kSiluLutGoldenTable[static_cast<size_t>(kSiluLutN)]);

	// The strongest exposure: TokenizerView::Impl::decode() is the one tokenizer
	// path that dereferences vocab_offsets/vocab_blob straight out of the backing
	// bytes (byte_to_id and the merge table are copied into Impl at Open() time,
	// so Encode()/VocabSize() alone do NOT exercise the dangling pointer — Decode()
	// is required to reach it). Fixture-known (pinned at test_main.cpp line ~1096
	// for the same MakeMinimalValidTok1 fixture): Encode("cat") == {3, 2} and
	// Decode({3, 2}) == "cat".
	CHECK_MSG(view.has_tokenizer, "the view exposes no tokenizer after intervening allocation");
	if (view.has_tokenizer) {
		CHECK(view.tokenizer.VocabSize() == static_cast<int32_t>(view.config.vocab_size));
		std::vector<int32_t> ids = view.tokenizer.Encode("cat");
		std::vector<int32_t> want_ids = {3, 2};
		CHECK_MSG(ids == want_ids,
		          "Encode(\"cat\") after intervening allocation produced %zu id(s), want [3,2]", ids.size());
		std::string decoded = view.tokenizer.Decode(want_ids);
		CHECK_MSG(decoded == "cat",
		          "Decode({3, 2}) after intervening allocation produced \"%s\", want \"cat\" — this is the "
		          "path that actually dereferences vocab_offsets/vocab_blob into the backing bytes",
		          decoded.c_str());
	}
}

// The amendment's cell (Charpy temper §6): the MOVED-FROM object, not only the
// moved-to one. RED against a member-wise default move (the pre-amendment
// type: sigmoid_lut is a bare pointer+count with no container to clear it, so
// a default move copies its bits onto the destination and leaves the source's
// copy — and its has_sigmoid_lut flag — untouched); GREEN only once the move
// constructor/assignment are hand-written to clear the source explicitly.
static void TestLoadMovedFromViewIsInertAndCarriesNoDanglingSigmoidLut() {
	using namespace superslm_test;
	auto built = BuildFullyValidV2ArtifactForLoad();

	SslmModelView view;
	std::string err;
	SslmModelStatus status = SslmModel::Load(built.bytes.data(), built.bytes.size(), view, &err);
	CHECK_MSG(status == SslmModelStatus::Ok, "fixture failed to load: got %s (%s)",
	          SslmModelStatusName(status), err.c_str());
	if (status != SslmModelStatus::Ok) return;

	SslmModelView moved = std::move(view);

	// The moved-TO object is fully populated — confirms the move actually ran.
	CHECK(moved.has_sigmoid_lut);
	CHECK(SigmoidLutValue(moved.sigmoid_lut, 0) == kSiluLutGoldenTable[0]);
	CHECK(SigmoidLutValue(moved.sigmoid_lut, kSiluLutN) == kSiluLutGoldenTable[static_cast<size_t>(kSiluLutN)]);

	// The moved-FROM object must be inert: every has_* flag reads false, and
	// sigmoid_lut carries no pointer into memory the moved-to object now
	// exclusively owns.
	CHECK(!view.has_config);
	CHECK(!view.has_sigmoid_lut);
	CHECK(!view.has_weights);
	CHECK(!view.has_biases);
	CHECK(!view.has_rope_tables);
	CHECK(!view.has_weight_scales);
	CHECK(!view.has_composition_constants);
	CHECK(!view.has_kv_landing_scales);
	CHECK(!view.has_kv_landing_reciprocals);
	CHECK(!view.has_tokenizer);
	CHECK_MSG(view.sigmoid_lut.values == nullptr,
	          "moved-from view's sigmoid_lut.values must be cleared to nullptr, not left pointing into "
	          "memory the moved-to object now exclusively owns");
	CHECK(view.sigmoid_lut.entry_count == 0);
}

// ---------------------------------------------------------------------------
// S-HARDEN-2's tokenizer join (F18, §17.3 cell 3): TOK1.vocab_count and
// CFG1.vocab_size, "enforced at a named API" — SslmModel::Load, per the
// S-HARDEN-1 boundary-gate pattern this slot follows rather than inventing a
// new mechanism (Claude/Vitruvius/SuperSLM_S-HARDEN-1_ParserVsConsumer_
// Decision-2026-07-22.md §4). Every cell drives a complete, otherwise-valid v2
// artifact (Config + SigmoidLut + Tokenizer + UnicodeTables) through
// SslmModel::Load, never the bare TokenizerView::Open/ParseConfig alone — a
// bare-sub-parser assertion could not detect a MISSING join by construction.
// ---------------------------------------------------------------------------

static void TestLoadRejectsTokenizerVocabCountVsConfigVocabSizeMismatch() {
	using namespace superslm_test;
	auto tok1 = MakeMinimalValidTok1();       // vocab_count == 5
	auto uni1 = MakeMinimalValidUni1();
	auto built = BuildTokenizerArtifactForLoad(tok1.bytes, uni1.bytes, /*vocab_size=*/999);  // deliberately != 5

	SslmModelView view;
	std::string err;
	SslmModelStatus status = SslmModel::Load(built.bytes.data(), built.bytes.size(), view, &err);
	CHECK_MSG(status == SslmModelStatus::TokenizerVocabSizeMismatch,
	          "F18 join (S-HARDEN-2): TOK1.vocab_count=5, CFG1.vocab_size=999, through SslmModel::Load: "
	          "got %s, want TokenizerVocabSizeMismatch (%s)",
	          SslmModelStatusName(status), err.c_str());
	CHECK_MSG(!view.has_tokenizer, "a mismatched-vocab-size artifact must not expose a tokenizer view");
	CHECK_MSG(err.find("5") != std::string::npos && err.find("999") != std::string::npos,
	          "diagnostic does not name both declared sizes: \"%s\"", err.c_str());
}

static void TestLoadAcceptsMatchingTokenizerVocabCountAndConfigVocabSize() {
	using namespace superslm_test;
	auto tok1 = MakeMinimalValidTok1();  // vocab_count == 5
	auto uni1 = MakeMinimalValidUni1();
	auto built = BuildTokenizerArtifactForLoad(tok1.bytes, uni1.bytes, /*vocab_size=*/5);  // matches

	SslmModelView view;
	std::string err;
	SslmModelStatus status = SslmModel::Load(built.bytes.data(), built.bytes.size(), view, &err);
	CHECK_MSG(status == SslmModelStatus::Ok, "matching TOK1.vocab_count/CFG1.vocab_size (both 5): got %s (%s)",
	          SslmModelStatusName(status), err.c_str());
	if (status != SslmModelStatus::Ok) return;
	CHECK(view.has_tokenizer);
	CHECK(view.tokenizer.Ok());
	CHECK(view.tokenizer.VocabSize() == 5);
	CHECK(view.config.vocab_size == 5u);
	// The exposed view is a working tokenizer, not merely a bookkeeping flag.
	std::vector<int32_t> ids = view.tokenizer.Encode("cat");
	CHECK(view.tokenizer.Decode(ids) == "cat");
}

static void TestLoadRejectsArtifactWithTokenizerSectionButNoUnicodeTables() {
	using namespace superslm_test;
	auto tok1 = MakeMinimalValidTok1();
	Cfg1Spec cfg;
	cfg.vocab_size = tok1.layout.vocab_count;
	auto built = BuildArtifact({MakeSection(SslmSectionType::Config, SslmDtype::Raw, BuildCfg1(cfg)),
	                            MakeSigmoidLutSection(),
	                            MakeSection(SslmSectionType::Tokenizer, SslmDtype::Raw, tok1.bytes)});

	SslmModelView view;
	std::string err;
	SslmModelStatus status = SslmModel::Load(built.bytes.data(), built.bytes.size(), view, &err);
	CHECK_MSG(status == SslmModelStatus::TokenizerRejected,
	          "Tokenizer section present, UnicodeTables absent, through SslmModel::Load: got %s, want "
	          "TokenizerRejected (%s)",
	          SslmModelStatusName(status), err.c_str());
	CHECK(!view.has_tokenizer);
}

static void TestLoadRejectsArtifactWithUnicodeTablesSectionButNoTokenizer() {
	using namespace superslm_test;
	auto uni1 = MakeMinimalValidUni1();
	auto built = BuildArtifact({MakeValidConfigSection(), MakeSigmoidLutSection(),
	                            MakeSection(SslmSectionType::UnicodeTables, SslmDtype::Raw, uni1.bytes)});

	SslmModelView view;
	std::string err;
	SslmModelStatus status = SslmModel::Load(built.bytes.data(), built.bytes.size(), view, &err);
	CHECK_MSG(status == SslmModelStatus::TokenizerRejected,
	          "UnicodeTables section present, Tokenizer absent, through SslmModel::Load: got %s, want "
	          "TokenizerRejected (%s)",
	          SslmModelStatusName(status), err.c_str());
	CHECK(!view.has_tokenizer);
}

static void TestLoadAcceptsArtifactWithNeitherTokenizerSection() {
	// A model-weights-only artifact (no Tokenizer, no UnicodeTables) is a valid
	// artifact shape (tokenizer and model weights are emitted by separate
	// converter invocations, tools/convert_tokenizer.py vs tools/convert_model.py)
	// — the tokenizer join must not turn a legitimately tokenizer-less model
	// artifact into a rejection.
	using namespace superslm_test;
	auto built = BuildArtifact({MakeValidConfigSection(), MakeSigmoidLutSection()});

	SslmModelView view;
	std::string err;
	SslmModelStatus status = SslmModel::Load(built.bytes.data(), built.bytes.size(), view, &err);
	CHECK_MSG(status == SslmModelStatus::Ok, "a Config+SigmoidLut-only (no tokenizer) artifact: got %s (%s)",
	          SslmModelStatusName(status), err.c_str());
	CHECK(!view.has_tokenizer);
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
	// Guard against a defect that reports Ok over a null/empty view (see
	// TestMinimalSil1ParsesAndReadsBackAllNodes).
	CHECK_MSG(lut.values != nullptr && lut.entry_count == kSigmoidLutEntries,
	          "lut is not a populated view on a status==Ok parse");
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
			// Widen to int64 before subtracting: a defective `lut` read
			// could be as far off as INT32_MIN, and `ref_q15 - lut` in
			// 32-bit arithmetic would itself overflow (signed UB) rather
			// than report a large delta.
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
// design: SuperSLM_matmul_subslot_design-2026-07-20.md). src/matmul.cpp's
// GemmInt8AccumulateRow / GemmInt8Accumulate / NarrowAccumulatorToI32 are fully
// built (shipped at S2.5); every cell below is green against them.
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
//     ShiftByMax already use -- i-exp moved from caller-ensures to checked at
//     S-HARDEN-0, so it is no longer this convention's example). assert()'s abort() would take down this
//     ENTIRE process -- and every check after it -- if the violating call ran
//     in-process, so it runs in an isolated child process instead; the parent only
//     observes whether the child terminated abnormally. This is new infrastructure:
//     no death-test convention existed anywhere in this suite before S2.5 (confirmed
//     by inspection before authoring this cell) -- documented in the test-design
//     record, not silently introduced. Verified empirically (see the record) that
//     an MSVC assert() failure under build.bat's flags exits promptly with a
//     nonzero abnormal-termination code and does NOT block on a dialog. ---

// Printed by RunCrashProbe to the child's stdout, immediately before the
// contract-violating call, so the parent can prove the named probe was actually
// dispatched and reached the call -- not merely that the child exited some way.
// The marker is probe-name-qualified so a stale or mismatched name cannot be
// mistaken for the one under test.
static std::string CrashProbeBeganMarker(const std::string& probe_name) {
	return "CRASH_PROBE_BEGAN:" + probe_name;
}

// Environment variable set by RunsCrashProbeAndCrashes before spawning the child,
// and inherited by it. Its presence lets main() recognize "this process is a
// crash-probe child" independently of argv parsing, so a future change that
// breaks the "--crash-probe=<name>" prefix match cannot make the child silently
// fall through to running the full suite -- which would itself spawn another
// crash-probe child, recursively (design/finding: pre-existing fork-bomb risk,
// Claude/Poirot/2fdaf49-s2.5-golden-hash-review-2026-07-20.md finding 8).
static const char* kCrashProbeChildEnvVar = "SUPERSLM_CRASH_PROBE_CHILD";

// std::getenv is the portable read; MSVC's /W4 flags it as deprecated in favor of
// _dupenv_s. This process only checks presence, never reads a value into a fixed
// buffer, so the deprecation does not apply -- silenced locally rather than
// project-wide.
static bool EnvVarIsSet(const char* name) {
#if defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable : 4996)
#endif
	return std::getenv(name) != nullptr;
#if defined(_MSC_VER)
#pragma warning(pop)
#endif
}

// Portable current-process-id read, used only to keep each crash-probe child's capture
// file distinct from every other configuration's (see kCrashProbeChildEnvVar's sibling
// note below): two configurations of this binary (e.g. a debug and an NDEBUG build) run
// concurrently on one machine and, before this, wrote and read the same fixed path in the
// shared system temp directory -- a race that turned a lost capture file from a degraded
// diagnostic message into an outright cell failure once the began-marker check started
// depending on that file's contents.
static long CurrentProcessId() {
#ifdef _WIN32
	return static_cast<long>(_getpid());
#else
	return static_cast<long>(getpid());
#endif
}

// The single verdict RunsCrashProbeAndCrashes returns. A plain bool cannot represent
// "the probe never ran" without collapsing it onto one of the other two answers -- which
// is exactly the shape of the finding this type exists to close (Claude/Poirot/
// 7511117-s2.5-golden-crash-probe-reverify-2026-07-20.md, finding 3, and the residual it
// names in the same entry). kDidNotRun is a third state a caller must handle explicitly;
// there is no bool-shaped shortcut back to "crashed" or "did not crash" for it.
enum class CrashProbeOutcome {
	kDidNotRun,      // the began-marker for the requested probe never appeared in the
	                 // child's captured output -- the contract-violating call was never
	                 // dispatched (probe name mismatch, broken dispatch, the child exited
	                 // before reaching it, or an unrecognized probe name). Neither
	                 // "crashed" nor "did not crash" is a meaningful answer for this
	                 // outcome.
	kRanNoCrash,     // the marker is present (the probe genuinely ran) and the child
	                 // completed without abnormal termination.
	kRanAndCrashed,  // the marker is present (the probe genuinely ran) and the child
	                 // terminated abnormally.
};

static const char* CrashProbeOutcomeName(CrashProbeOutcome outcome) {
	switch (outcome) {
		case CrashProbeOutcome::kDidNotRun:
			return "did-not-run";
		case CrashProbeOutcome::kRanNoCrash:
			return "ran-no-crash";
		case CrashProbeOutcome::kRanAndCrashed:
			return "ran-and-crashed";
	}
	return "(unknown CrashProbeOutcome)";
}

// Runs the named crash probe in a child process and returns a single verdict that already
// accounts for whether the probe genuinely ran -- a caller cannot obtain kRanNoCrash or
// kRanAndCrashed without the began-marker check below having passed first, because the
// CHECK_MSG that enforces it, and the early return past it, both live here rather than at
// each call site. A future second death-test cell built on this helper inherits that
// enforcement by construction: there is no code path through this function that hands back
// a crashed/did-not-crash answer for a probe that never ran, so there is nothing for a new
// caller to forget to repeat.
static CrashProbeOutcome RunsCrashProbeAndCrashes(const char* probe_name, std::string* out_tail) {
	std::filesystem::path out_path =
	    std::filesystem::temp_directory_path() /
	    (std::string("superslm_crash_probe_") + probe_name + "_" +
	     std::to_string(CurrentProcessId()) + ".txt");
	std::error_code rm_ec;
	std::filesystem::remove(out_path, rm_ec);

	std::string cmd = "\"" + GSelfPath + "\" --crash-probe=" + probe_name +
	                   " > \"" + out_path.string() + "\" 2>&1";
#ifdef _WIN32
	// system() on Windows invokes `cmd.exe /c <cmd>`; when <cmd> itself begins with a
	// quoted executable path, cmd.exe's first/last-quote-stripping parser misreads the
	// nested quotes (a well-known cmd.exe quirk) unless the WHOLE string is wrapped in
	// one more outer quote pair -- that outer pair is what cmd strips, leaving the
	// interior correctly quoted. This wrap is a cmd.exe-only workaround: std::system on
	// POSIX invokes `/bin/sh -c <cmd>`, which has no equivalent quote-stripping step, so
	// the same outer wrap there collapses the whole command into one (nonexistent)
	// command word -- verified by execution: sh exits 127 and the child never runs, which
	// a naive `rc != 0` read misreports as "crashed" (Claude/Brunel/
	// superslm-s2.5-finding-for-curie-crash-probe-2026-07-20.md).
	std::string wrapped_cmd = "\"" + cmd + "\"";
	_putenv_s(kCrashProbeChildEnvVar, "1");
#else
	const std::string& wrapped_cmd = cmd;
	setenv(kCrashProbeChildEnvVar, "1", /*overwrite=*/1);
#endif
	int rc = std::system(wrapped_cmd.c_str());
#ifdef _WIN32
	// _putenv_s with an empty value removes the variable (documented behavior). Set only
	// for the duration of spawning this one child; a variable left set in the parent's own
	// environment would be inherited by any later child this process spawns, which would
	// misread as "I am a crash-probe child" and refuse to run (Claude/Poirot/
	// 7511117-s2.5-golden-crash-probe-reverify-2026-07-20.md, finding 4).
	_putenv_s(kCrashProbeChildEnvVar, "");
#else
	unsetenv(kCrashProbeChildEnvVar);
#endif

	std::string content;
	{
		std::ifstream f(out_path, std::ios::binary);
		content.assign(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
	}
	if (out_tail) *out_tail = content;
	std::filesystem::remove(out_path, rm_ec);

	// The began-marker check that used to live in the caller (Claude/Poirot/
	// 7511117-s2.5-golden-crash-probe-reverify-2026-07-20.md, finding 3): the contract-
	// violating call must have been genuinely dispatched before "crashed" or "did not
	// crash" means anything. Folded in here, this CHECK_MSG fires for every caller of this
	// helper, present and future, not only for the one call site that remembers to repeat
	// it.
	bool began = content.find(CrashProbeBeganMarker(probe_name)) != std::string::npos;
	CHECK_MSG(began,
	          "the child's captured output never contains the began-marker for probe "
	          "'%s' -- the contract-violating call was never dispatched (probe name "
	          "mismatch, broken dispatch, or the child exited before reaching it), so no "
	          "crashed/did-not-crash outcome can be trusted for it -- child output was: %s",
	          probe_name, content.c_str());
	if (!began) return CrashProbeOutcome::kDidNotRun;

	// Reachable only once the marker has proven the probe genuinely ran. RunCrashProbe's
	// unrecognized-name sentinel (return code 2) cannot occur on this path: it returns 2
	// only on the branch that never prints the marker, so `began` would have been false
	// and the function would already have returned above. rc's only remaining meanings are
	// therefore "completed normally" (0) or "terminated abnormally" (nonzero) -- the
	// residual the reviewer named (RunCrashProbe's code 2 satisfying `rc != 0` for a probe
	// that never ran) closes by construction, not by exclusion, because the sentinel and a
	// present marker cannot occur together.
	return (rc != 0) ? CrashProbeOutcome::kRanAndCrashed : CrashProbeOutcome::kRanNoCrash;
}

// Dispatched from main() when argv[1] == "--crash-probe=<name>": runs exactly one
// contract-violating call in isolation. Prints CrashProbeBeganMarker(name)
// immediately before making the call and flushes it, so the marker reaches the
// parent's captured output even if the call crashes the process outright.
// Returns 0 ONLY if the named, recognized call ran to completion without
// crashing -- a suite defect this cell exists to catch, not the expected
// outcome. Returns 2, without printing the began-marker for `name`, if `name`
// is not a recognized probe -- this must read as "the probe did not run" to
// every caller, never as "the probe ran and did not crash".
static int RunCrashProbe(const std::string& name) {
	// RETIRED 2026-07-22 (S-HARDEN-0 final API): the "iexp_out_of_domain_constants" and
	// "iexp_guard_order:<fn>:..." probes that used to live here called IExpFromConstants
	// directly with contract-violating raw (q, q_ln2, q_b, q_c) -- a call this API no
	// longer allows anyone to write. IExpFromConstants is removed; the only entry point
	// that takes raw constants is IExpConstruct, and it is TOTAL and asserts nothing, in
	// every build configuration, so there is no longer a contract-violating call to
	// isolate in a crash-probe child at all. See
	// TestIExpConstructAndEvaluateProducesKnownWrappedValueForOutOfDomainConstants and the
	// comment at the retired TestIExpGuardOrderCasesNeverExecuteUBAndAgreeWithIndependentOracle
	// call site below for the full account and where each witness value now lives.
	if (name == "matmul_zero_in_channels") {
		int8_t act[1] = {0};
		int8_t wgt[1] = {0};
		int64_t out_acc[1] = {0};
		std::printf("%s\n", CrashProbeBeganMarker(name).c_str());
		std::printf("crash-probe matmul_zero_in_channels: calling GemmInt8AccumulateRow "
		            "with in_channels=0 (below the architectural floor, design S12 dim 4)\n");
		std::fflush(stdout);
		GemmInt8AccumulateRow(act, wgt, /*in_channels=*/0, /*out_channels=*/1, out_acc);
		std::printf("PROBE DID NOT CRASH\n");
		return 0;
	}
	// S3 (Poirot review ac34677, 2026-07-28): RowBoundsWide (src/intmath.cpp:
	// 279-280) reads x[0] before testing n at all -- with a null data pointer
	// and n == 0 this is a null-pointer dereference, and the reviewer's own
	// probe process terminated at the call (P7). Unlike matmul_zero_in_
	// channels above, this is a genuine memory access, not an assert(): it is
	// not compiled out under NDEBUG, so it must not crash in EITHER build
	// configuration once fixed.
	if (name == "row_bounds_wide_zero_len_null_ptr") {
		int64_t out_max = 0;
		int64_t out_min = 0;
		std::printf("%s\n", CrashProbeBeganMarker(name).c_str());
		std::printf("crash-probe row_bounds_wide_zero_len_null_ptr: calling RowBoundsWide "
		            "with a null data pointer and n=0 (no n >= 1 precondition is documented "
		            "on this primitive or on NarrowRowChecked, S3)\n");
		std::fflush(stdout);
		superslm::RowBoundsWide(nullptr, /*n=*/0, &out_max, &out_min);
		std::printf("PROBE DID NOT CRASH (out_max=%lld out_min=%lld)\n",
		            static_cast<long long>(out_max), static_cast<long long>(out_min));
		return 0;
	}
	std::printf("PROBE DID NOT CRASH (unknown probe name: %s)\n", name.c_str());
	return 2;
}

static void TestGemmInt8AccumulateRowAssertsOnZeroInChannelsContractViolation() {
	static const char* kProbeName = "matmul_zero_in_channels";
	std::string tail;
	// The began-marker check is enforced inside RunsCrashProbeAndCrashes itself (it fires
	// its own CHECK_MSG and returns kDidNotRun) -- this call site does not repeat it. That
	// is the fix to Claude/Poirot/7511117-s2.5-golden-crash-probe-reverify-2026-07-20.md
	// finding 3: the marker check used to live only here, so a second cell built on the
	// same helper without also remembering to check it would have reinherited the finding.
	CrashProbeOutcome outcome = RunsCrashProbeAndCrashes(kProbeName, &tail);
#ifdef NDEBUG
	// assert() is a no-op whenever NDEBUG is defined, which every current CI job
	// defines (windows-x64/linux-x64 build Release, linux-x64-asan builds
	// RelWithDebInfo, macos-arm64 builds Release -- all four map NDEBUG onto the
	// child, since it is the same binary re-invoked). The design's caller-ensures
	// convention (S12 dim 2/5) makes this contract violation UB in a configuration
	// where the assert is compiled out, not a runtime-rejected hostile input -- so no
	// abort is the correct, documented outcome here, and the cell's claim in this
	// configuration is exactly that: the child completes without abnormal
	// termination. The abort-on-violation claim itself (design S12 dim 5, "must abort
	// a debug build") is proved by the non-NDEBUG default build (build.bat), where
	// the assert is compiled in -- see the #else branch below. A probe that never ran
	// (kDidNotRun) satisfies neither this check nor the debug-branch one below --
	// there is no bool-shaped path by which it could pass either.
	CHECK_MSG(outcome == CrashProbeOutcome::kRanNoCrash,
	          "GemmInt8AccumulateRow(in_channels=0) under NDEBUG: assert is compiled "
	          "out, so the child must complete without abnormal termination (caller- "
	          "ensures UB in this configuration, design S12 dim 2/5) -- outcome was "
	          "%s, child output was: %s",
	          CrashProbeOutcomeName(outcome), tail.c_str());
#else
	CHECK_MSG(outcome == CrashProbeOutcome::kRanAndCrashed,
	          "GemmInt8AccumulateRow(in_channels=0) must abort a debug build (contract "
	          "violation, design S12 dim 4/5) -- outcome was %s, child output was: %s",
	          CrashProbeOutcomeName(outcome), tail.c_str());
#endif
}

// S3 (Poirot review ac34677, 2026-07-28; SuperSLM_S3a_WalkingSkeleton_Plan.md
// Sec11 S3.1, F-S3-7): RowBoundsWide reads x[0] before testing n at all
// (src/intmath.cpp:279-280); with a null data pointer and n == 0 -- a
// degenerate but in-contract input (neither RowBoundsWide's own header nor
// NarrowRowChecked's contract documents an n >= 1 precondition) -- this is a
// null-pointer dereference. Unlike the assert()-gated contract violation
// above, a null-pointer dereference is NOT compiled out under NDEBUG (it is
// a genuine memory access, not a debug check), so this cell asserts
// kRanNoCrash unconditionally, with no #ifdef NDEBUG branch: the sibling
// primitive MaxAbsReduceWide already treats n == 0 as an in-contract,
// defined case (D' == 1, "the empty reduction", intmath.h:150-152), so n ==
// 0 is not a caller-ensures contract violation this primitive is entitled
// to leave undefined either.
static void TestRowBoundsWideZeroLenNullPtrDoesNotCrash() {
	static const char* kProbeName = "row_bounds_wide_zero_len_null_ptr";
	std::string tail;
	CrashProbeOutcome outcome = RunsCrashProbeAndCrashes(kProbeName, &tail);
	CHECK_MSG(outcome == CrashProbeOutcome::kRanNoCrash,
	          "RowBoundsWide(nullptr, 0, &out_max, &out_min) must not crash in any build "
	          "configuration -- outcome was %s, child output was: %s",
	          CrashProbeOutcomeName(outcome), tail.c_str());
}

// REWORKED 2026-07-22 (S-HARDEN-0 final API port). This cell used to pin TWO things
// under the old two-function API: a debug-build abort (IExpFromConstants's internal
// "asserts success" contract), and the exact NDEBUG wrapped value
// (-6219525972835464584) for IExpFromConstants(0, 887904998, 1733160715, INT64_MAX).
//
// Under the final API there is no contract-violating call left to abort on.
// IExpConstruct(0, 887904998, 1733160715, INT64_MAX, &out) does not violate any
// contract -- q<=0 and 1<=q_ln2<=ceiling both hold, and q_p+q_b is representable --
// it returns kNotRepresentable, a documented, non-erroneous outcome that FILLS *out,
// and IExpEvaluate(out) is TOTAL on that construction in EVERY build configuration
// (no assert anywhere in IExpConstruct or IExpEvaluate -- confirmed by reading both
// bodies in src/intmath.cpp, S-HARDEN-0). There is no debug-vs-release split left to
// assert, because there is no code path in this call that ever asserts, in any
// configuration. The assert half of this cell's original claim therefore has no
// surviving subject, and is not silently dropped -- it is named here as retired,
// with the reason, per this project's "a gap is a finding, not a silent omission"
// discipline.
//
// The wrapped-value golden MUST survive, and does: this is exactly
// kIExpConstructCases' "notrepresentable_strike_witness" row (D-SLM78's original
// strike input), so this cell is also a standalone, individually diagnosable
// regression check for that one row, checked directly rather than by scanning a
// table (the same rationale TestIExpConstantsInDomainRejectsStrikeExactInput already
// uses for the predicate side of the same input).
static void TestIExpConstructAndEvaluateProducesKnownWrappedValueForOutOfDomainConstants() {
	superslm::IExpConstruction out;
	superslm::IExpDomain d = superslm::IExpConstruct(INT64_C(0), INT64_C(887904998), INT64_C(1733160715),
	                                                  INT64_C(9223372036854775807), &out);
	CHECK_MSG(d == superslm::IExpDomain::kNotRepresentable,
	          "IExpConstruct(0, 887904998, 1733160715, 2^63-1) returned domain %d, want "
	          "kNotRepresentable -- this is the strike's exact contract-legal input for which the "
	          "decomposition is well-formed but base^2+q_c does not fit int64 "
	          "(Claude/Loki/softmax-s2.6-strike-2026-07-21.md)",
	          static_cast<int>(d));
	if (d != superslm::IExpDomain::kNotRepresentable) return;
	CHECK_MSG(out.z() == 0 && out.base() == INT64_C(1733160715) && out.q_c() == INT64_C(9223372036854775807),
	          "IExpConstruct(...) is kNotRepresentable and must FILL *out with the well-formed "
	          "decomposition -- got z=%lld base=%lld q_c=%lld, want z=0 base=1733160715 q_c=%lld",
	          static_cast<long long>(out.z()), static_cast<long long>(out.base()),
	          static_cast<long long>(out.q_c()), static_cast<long long>(INT64_C(9223372036854775807)));
	int64_t got = superslm::IExpEvaluate(out);
	CHECK_MSG(got == INT64_C(-6219525972835464584),
	          "IExpEvaluate(construction from the strike's out-of-domain constants) == %lld, want "
	          "-6219525972835464584 (the exact wrapped value the strike observed, "
	          "Claude/Loki/softmax-s2.6-strike-2026-07-21.md, D-SLM80 behaviour-preservation) -- this "
	          "value must be identical in EVERY build configuration: IExpEvaluate has no assert and "
	          "no NDEBUG split, unlike the retired IExpFromConstants this cell used to pin",
	          static_cast<long long>(got));
}

// ---------------------------------------------------------------------------
// RETIRED 2026-07-22 (S-HARDEN-0 final API port). Curie's S-HARDEN-0 population
// suite, LAYER A, used to live here: a crash-probe population over the OLD
// two-function API (IExpFromConstants / IExpConstantsInDomain), distinguishing an
// "eval" call path (F9's finding: the evaluator consults its accessors, which
// overflow, before ever asserting the domain) from a "pred" call path (F21's
// finding: the guard's own internal accessor call overflows on an unbounded q_b
// before the guard can answer). Both required child-process isolation because,
// at f078403 and through this branch's a1d7986 revision, the un-fixed call really
// could crash the ENTIRE test process.
//
// The final API collapses the distinction this population existed to test.
// IExpFromConstants is removed; the only function that takes raw (q, q_ln2, q_b,
// q_c) is IExpConstruct, and it is TOTAL -- asserts nothing, executes no UB, in
// every build configuration (read: both IExpConstruct's and IExpEvaluate's bodies
// in src/intmath.cpp, S-HARDEN-0; confirmed further by Poirot's a1d7986 review,
// 5,684,354 executed quadruples under UBSan, 0 diagnostics). There is no longer a
// second call path for "eval" to name, and no longer a crash to isolate a child
// process against -- an evaluation is now only ever reachable through a
// construction IExpConstruct itself validated (or the safe default {0,0,0}), which
// TestIExpConstructionDefaultIsSafeToEvaluate and the two IExpEvaluate-totality
// static_asserts below prove structurally rather than by population.
//
// No witness value is dropped. Every input this population's ten rows drove is a
// row of kIExpConstructCases (tests/sslm_iexp_domain_fixtures.h) now: F9's four
// ceiling witnesses are badqln2_last_valid_ceiling / badqln2_first_invalid_ceiling
// / badqln2_interior_invalid_ceiling / badqln2_f9_witness_int64_max; F21's four
// q_b witnesses are badqb_f21_witness_qb_int64_min / badqb_boundary_first_unsafe /
// badqb_interior_unsafe / badqb_representable_boundary_square_not_representable
// (the last one carries a corrected label: it IS the old "boundary_last_safe"
// witness, exact same (q, q_ln2, q_b) -- accurate about q_p+q_b being
// representable, but the old label implied overall domain membership, which this
// input does not have once base=INT64_MIN is squared; see that row's own comment
// in gen_iexp_domain_fixtures.py). The two "_eval_" duplicates named no witness
// value the "pred" rows above did not already carry, so nothing beyond those eight
// is owed. TestIExpConstructMatchesIndependentOracleAcrossCases (below) sweeps all
// of kIExpConstructCases, including these rows, in-process, under whichever
// sanitizer configuration the suite is built with -- the same population, proven
// the same way execution always proved it, without a probe.
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Curie's S-HARDEN-0 population suite (LAYER B): the checked `IExpConstruct` /
// `IExpConstruction` / `IExpDomain` / `IExpEvaluate` entry point, final API
// (SuperSLM_Plan.md's S-HARDEN-0 sub-slot; Claude/Curie/
// superslm-s-harden-0-test-design-2026-07-21.md). `IExpConstruction`'s fields are
// PRIVATE (`z()`/`base()`/`q_c()` accessors only, populated solely by
// `IExpConstruct`'s friend access) and `IExpEvaluate` takes only the construction --
// this section's own two structural static_asserts, immediately below, pin both of
// those properties at compile time, and TestIExpConstructionDefaultIsSafeToEvaluate
// pins that the only OTHER reachable origin (a default-constructed object) is safe.
// ---------------------------------------------------------------------------

// Property: "IExpEvaluate takes only a construction; q_c is carried, never passed
// separately." A separate q_c parameter would let a caller validate against one
// constant and evaluate against another -- the exact two-derivations-drift defect
// this slot's own header comment names (F10). The final API makes that call
// unwritable: IExpEvaluate's signature has no q_c parameter at all. This is a
// property OF THE SIGNATURE, so it is pinned at compile time -- a future edit that
// re-introduces a second q_c argument fails to compile this file, immediately,
// rather than waiting for a test to notice a caller passing mismatched values.
static_assert(std::is_same<decltype(&superslm::IExpEvaluate), int64_t (*)(const superslm::IExpConstruction&)>::value,
              "IExpEvaluate must take ONLY a construction -- q_c must be carried by it, not accepted "
              "as a separate argument (S-HARDEN-0: a separate q_c would let a caller validate "
              "against one constant and evaluate against another, the two-derivations-drift defect "
              "F10 already recorded)");

// Property: "IExpConstruction's fields are private, and only IExpConstruct can
// populate one." A type with private non-static data members is never an
// aggregate (C++ [dcl.init.aggr]), so this is equivalent to "no caller can
// brace-initialize or field-assign an IExpConstruction into a state IExpConstruct
// never produced" -- the exact defect this slot's header comment names: "a public
// {z, base} aggregate would admit z = 999 from any caller," which is an unchecked
// shift and undefined behaviour in IExpEvaluate. Pinned at compile time for the
// same reason as the assert above: this is a property of the TYPE, not of any one
// call, so a test that only ever calls IExpConstruct correctly could never observe
// a regression here -- only the type system can.
static_assert(!std::is_aggregate<superslm::IExpConstruction>::value,
              "IExpConstruction must not be an aggregate -- z_/base_/q_c_ must stay private and "
              "unsettable directly by any caller (S-HARDEN-0: a public aggregate would admit an "
              "unchecked z outside [0, I_EXP_CLIP_N], which IExpEvaluate's shift cannot survive)");

// Property: the OTHER reachable origin of an IExpConstruction -- besides one
// IExpConstruct itself validated -- is a default-constructed object, and the header
// documents it as safe ("a default-constructed one is {0, 0, 0}, which is safe: z =
// 0 is a legal shift"). Checked directly rather than assumed: default-construct,
// read the three accessors, and evaluate it, confirming the documented zero state
// and that evaluating it executes no UB (run under ASan+UBSan, this cell would trap
// if z were anything other than a legal shift amount) and returns the value the
// formula predicts ((0^2 + 0) >> 0 == 0).
static void TestIExpConstructionDefaultIsSafeToEvaluate() {
	superslm::IExpConstruction c;
	CHECK_MSG(c.z() == 0 && c.base() == 0 && c.q_c() == 0,
	          "a default-constructed IExpConstruction must be {z=0, base=0, q_c=0} -- got "
	          "z=%lld base=%lld q_c=%lld",
	          static_cast<long long>(c.z()), static_cast<long long>(c.base()), static_cast<long long>(c.q_c()));
	int64_t got = superslm::IExpEvaluate(c);
	CHECK_MSG(got == 0,
	          "IExpEvaluate(default-constructed IExpConstruction) == %lld, want 0 -- (0^2+0)>>0 == 0, "
	          "and z=0 is documented as a legal (no-op) shift",
	          static_cast<long long>(got));
}

// Stringifies IExpDomain for CHECK_MSG output and for comparison against the
// fixture's expected_domain field (a string, kept free of a compile-time dependency
// on IExpDomain's exact enum values -- see sslm_iexp_domain_fixtures.h's own
// comment). Mirrors this file's existing CrashProbeOutcomeName pattern.
static const char* IExpDomainName(IExpDomain d) {
	switch (d) {
		case IExpDomain::kOk:
			return "kOk";
		case IExpDomain::kNotRepresentable:
			return "kNotRepresentable";
		case IExpDomain::kBadQ:
			return "kBadQ";
		case IExpDomain::kBadQLn2:
			return "kBadQLn2";
		case IExpDomain::kBadQB:
			return "kBadQB";
	}
	return "(unknown IExpDomain)";
}

// "*out is FILLED whenever the decomposition is well-formed -- that is, for kOk AND
// kNotRepresentable, and IExpEvaluate is TOTAL over both." True for exactly those
// two outcome names.
static bool IExpDomainNameIsWellFormed(const char* name) {
	return std::strcmp(name, "kOk") == 0 || std::strcmp(name, "kNotRepresentable") == 0;
}

// The primary sweep: every row's returned IExpDomain against the independent
// oracle, and -- for the two well-formed outcomes -- out.z()/out.base()/out.q_c()
// AND IExpEvaluate(out) against the same oracle. This is the cell that proves
// IExpEvaluate's totality (dimension: "IExpEvaluate is total on any construction it
// can be given, including one from a kNotRepresentable outcome, which still yields
// the exact wrapped value") across the FULL population, not only the single
// standalone golden TestIExpConstructAndEvaluateProducesKnownWrappedValueForOutOfDomainConstants
// pins. Does not poison *out with a sentinel (that is the dedicated concern of
// TestIExpConstructOutContractPerOutcome below); this function is the one place a
// reader confirms "does the entry point classify, decompose, and evaluate
// correctly" without also having to reason about the untouched-vs-filled contract
// at the same time.
static void TestIExpConstructMatchesIndependentOracleAcrossCases() {
	using namespace superslm_test;
	for (size_t i = 0; i < kIExpConstructCasesCount; ++i) {
		const IExpConstructCase& c = kIExpConstructCases[i];
		IExpConstruction out;
		IExpDomain got = IExpConstruct(c.q, c.q_ln2, c.q_b, c.q_c, &out);
		CHECK_MSG(std::strcmp(IExpDomainName(got), c.expected_domain) == 0,
		          "%s: IExpConstruct(%lld, %lld, %lld, %lld) returned %s, independently-derived "
		          "oracle expects %s",
		          c.label, static_cast<long long>(c.q), static_cast<long long>(c.q_ln2),
		          static_cast<long long>(c.q_b), static_cast<long long>(c.q_c), IExpDomainName(got),
		          c.expected_domain);
		if (std::strcmp(IExpDomainName(got), c.expected_domain) != 0) continue;
		if (IExpDomainNameIsWellFormed(c.expected_domain)) {
			CHECK_MSG(out.z() == c.expected_z,
			          "%s: IExpConstruct(%lld, %lld, %lld, %lld) [%s].z() == %lld, oracle expects %lld",
			          c.label, static_cast<long long>(c.q), static_cast<long long>(c.q_ln2),
			          static_cast<long long>(c.q_b), static_cast<long long>(c.q_c), c.expected_domain,
			          static_cast<long long>(out.z()), static_cast<long long>(c.expected_z));
			CHECK_MSG(out.base() == c.expected_base,
			          "%s: IExpConstruct(%lld, %lld, %lld, %lld) [%s].base() == %lld, oracle expects %lld",
			          c.label, static_cast<long long>(c.q), static_cast<long long>(c.q_ln2),
			          static_cast<long long>(c.q_b), static_cast<long long>(c.q_c), c.expected_domain,
			          static_cast<long long>(out.base()), static_cast<long long>(c.expected_base));
			CHECK_MSG(out.q_c() == c.q_c,
			          "%s: IExpConstruct(%lld, %lld, %lld, %lld) [%s].q_c() == %lld, want the SAME "
			          "q_c passed to IExpConstruct (%lld) -- q_c is carried by the construction, not "
			          "re-derived",
			          c.label, static_cast<long long>(c.q), static_cast<long long>(c.q_ln2),
			          static_cast<long long>(c.q_b), static_cast<long long>(c.q_c), c.expected_domain,
			          static_cast<long long>(out.q_c()), static_cast<long long>(c.q_c));
			int64_t evaluated = IExpEvaluate(out);
			CHECK_MSG(evaluated == c.expected_value,
			          "%s: IExpEvaluate(IExpConstruct(%lld, %lld, %lld, %lld)) == %lld, "
			          "independently-derived oracle expects %lld -- IExpEvaluate must be TOTAL over "
			          "this outcome (%s)",
			          c.label, static_cast<long long>(c.q), static_cast<long long>(c.q_ln2),
			          static_cast<long long>(c.q_b), static_cast<long long>(c.q_c),
			          static_cast<long long>(evaluated), static_cast<long long>(c.expected_value),
			          c.expected_domain);
		}
	}
}

// The IExpConstruction contract, per outcome: filled for kOk and kNotRepresentable,
// untouched for kBadQ/kBadQLn2/kBadQB. IExpConstruction's fields are private with no
// setters (pinned above by the !is_aggregate static_assert), so this cell cannot
// poison *out with a hand-fabricated sentinel the way the prior (public-struct)
// revision did -- the only way to put a NON-default value into an IExpConstruction
// is a real IExpConstruct call. So it primes *out with a real, known-good
// construction first (kPriming*, chosen so its z/base/q_c collide with no row's
// expected_z/expected_base/q_c below -- confirmed: every well-formed row here has
// z==0, and the priming call's z==1), records the priming values, then calls
// IExpConstruct for the case under test into the SAME out: for kBad* outcomes,
// *out must still read back exactly the PRIMING values (untouched); for kOk/
// kNotRepresentable, *out must now read back the CASE's own independently-derived
// values (overwritten). This is a stronger test than a sentinel, not a weaker one:
// it proves untouched-ness against a real object a caller could actually be
// holding across two calls, not against an artificial poison value.
static void TestIExpConstructOutContractPerOutcome() {
	using namespace superslm_test;
	const int64_t kPrimingQ = INT64_C(-777);
	const int64_t kPrimingQLn2 = INT64_C(777);
	const int64_t kPrimingQB = INT64_C(777);
	const int64_t kPrimingQC = INT64_C(777);
	for (size_t i = 0; i < kIExpConstructCasesCount; ++i) {
		const IExpConstructCase& c = kIExpConstructCases[i];
		IExpConstruction out;
		IExpDomain priming_domain = IExpConstruct(kPrimingQ, kPrimingQLn2, kPrimingQB, kPrimingQC, &out);
		CHECK_MSG(priming_domain == IExpDomain::kOk,
		          "priming call IExpConstruct(-777, 777, 777, 777) returned %s, want kOk -- this "
		          "cell's priming construction is expected in-domain; if this fails, the priming "
		          "constants themselves need revisiting, not the row under test",
		          IExpDomainName(priming_domain));
		const int64_t priming_z = out.z();
		const int64_t priming_base = out.base();
		const int64_t priming_qc = out.q_c();
		IExpDomain got = IExpConstruct(c.q, c.q_ln2, c.q_b, c.q_c, &out);
		CHECK_MSG(std::strcmp(IExpDomainName(got), c.expected_domain) == 0,
		          "%s: IExpConstruct(%lld, %lld, %lld, %lld) returned %s, independently-derived "
		          "oracle expects %s",
		          c.label, static_cast<long long>(c.q), static_cast<long long>(c.q_ln2),
		          static_cast<long long>(c.q_b), static_cast<long long>(c.q_c), IExpDomainName(got),
		          c.expected_domain);
		if (IExpDomainNameIsWellFormed(c.expected_domain)) {
			CHECK_MSG(out.z() == c.expected_z && out.base() == c.expected_base && out.q_c() == c.q_c,
			          "%s: IExpConstruct(...) is %s and must FILL *out -- got z=%lld base=%lld "
			          "q_c=%lld, oracle expects z=%lld base=%lld q_c=%lld",
			          c.label, c.expected_domain, static_cast<long long>(out.z()),
			          static_cast<long long>(out.base()), static_cast<long long>(out.q_c()),
			          static_cast<long long>(c.expected_z), static_cast<long long>(c.expected_base),
			          static_cast<long long>(c.q_c));
		} else {
			CHECK_MSG(out.z() == priming_z && out.base() == priming_base && out.q_c() == priming_qc,
			          "%s: IExpConstruct(...) is %s and must leave *out UNTOUCHED -- got z=%lld "
			          "base=%lld q_c=%lld, priming construction was z=%lld base=%lld q_c=%lld",
			          c.label, c.expected_domain, static_cast<long long>(out.z()),
			          static_cast<long long>(out.base()), static_cast<long long>(out.q_c()),
			          static_cast<long long>(priming_z), static_cast<long long>(priming_base),
			          static_cast<long long>(priming_qc));
		}
	}
}

// "out may be null when only the predicate answer is wanted."
static void TestIExpConstructAcceptsNullOutForPredicateOnlyUse() {
	using namespace superslm_test;
	for (size_t i = 0; i < kIExpConstructCasesCount; ++i) {
		const IExpConstructCase& c = kIExpConstructCases[i];
		IExpDomain got = IExpConstruct(c.q, c.q_ln2, c.q_b, c.q_c, nullptr);
		CHECK_MSG(std::strcmp(IExpDomainName(got), c.expected_domain) == 0,
		          "%s: IExpConstruct(%lld, %lld, %lld, %lld, nullptr) returned %s, "
		          "independently-derived oracle expects %s",
		          c.label, static_cast<long long>(c.q), static_cast<long long>(c.q_ln2),
		          static_cast<long long>(c.q_b), static_cast<long long>(c.q_c), IExpDomainName(got),
		          c.expected_domain);
	}
}

// "IExpConstantsInDomain(q, q_ln2, q_b, q_c) is exactly IExpConstruct(q, q_ln2, q_b,
// q_c, nullptr) == IExpDomain::kOk -- same signature, same bool, unchanged answers on
// every input that was previously defined" (S-HARDEN-0 sub-slot, Brunel's revision).
// Proves the equivalence directly rather than assuming the two calls agree because
// both matched the same oracle independently.
static void TestIExpConstantsInDomainEquivalentToIExpConstructEqualsKOk() {
	using namespace superslm_test;
	for (size_t i = 0; i < kIExpConstructCasesCount; ++i) {
		const IExpConstructCase& c = kIExpConstructCases[i];
		bool pred = IExpConstantsInDomain(c.q, c.q_ln2, c.q_b, c.q_c);
		bool construct_is_ok = IExpConstruct(c.q, c.q_ln2, c.q_b, c.q_c, nullptr) == IExpDomain::kOk;
		bool expected_ok = std::strcmp(c.expected_domain, "kOk") == 0;
		CHECK_MSG(pred == construct_is_ok,
		          "%s: IExpConstantsInDomain(%lld, %lld, %lld, %lld) == %s but "
		          "(IExpConstruct(..., nullptr) == kOk) == %s on the identical arguments",
		          c.label, static_cast<long long>(c.q), static_cast<long long>(c.q_ln2),
		          static_cast<long long>(c.q_b), static_cast<long long>(c.q_c),
		          pred ? "true" : "false", construct_is_ok ? "true" : "false");
		CHECK_MSG(pred == expected_ok,
		          "%s: IExpConstantsInDomain(%lld, %lld, %lld, %lld) == %s, independently-derived "
		          "oracle expects %s",
		          c.label, static_cast<long long>(c.q), static_cast<long long>(c.q_ln2),
		          static_cast<long long>(c.q_b), static_cast<long long>(c.q_c),
		          pred ? "true" : "false", expected_ok ? "true" : "false");
	}
}

// ---------------------------------------------------------------------------
// PORTED (Brunel, mid-build 2026-07-21; accessor methods 2026-07-22): the
// pre-existing S2.6-amendment accessor cells (Claude/Curie/
// superslm-s2.6-softmax-iexp-domain-test-design-2026-07-21.md) called
// `superslm::IExpShift`/`superslm::IExpBase` directly; both are removed from the
// public header by this slot. `kIExpAccessorCases` (tests/
// sslm_iexp_domain_fixtures.h, unchanged -- its 36 rows' values are unaffected by
// the API change) is reused as-is; only the two functions that consumed it are
// ported, merged into one below since both now come from a single IExpConstruct
// call rather than two separate accessor calls. Confirmed (this session, before
// porting): none of the 36 rows' q_ln2 exceeds kIExpMaxQLn2 (max is 887904998
// against a ceiling of 307445734561825860), so none needs to become a rejection
// cell -- the caution in Brunel's message ("cells that pin decomposition values for
// inputs ABOVE the q_ln2 ceiling... cannot port as value pins") does not apply to
// any row here, verified rather than assumed. Every row is expected kOk (each
// already produced a golden that fits int64_t, per the original suite's own
// docstring), so this function also checks the returned IExpDomain.
// ---------------------------------------------------------------------------

static void TestIExpConstructMatchesAccessorCasesZAndBase() {
	using namespace superslm_test;
	for (size_t i = 0; i < kIExpAccessorCasesCount; ++i) {
		const IExpAccessorCase& c = kIExpAccessorCases[i];
		IExpConstruction out;
		IExpDomain got = IExpConstruct(c.q, c.q_ln2, c.q_b, /*q_c=*/0, &out);
		// q_c=0 here: kIExpAccessorCases carries no q_c field (IExpShift/IExpBase
		// never took one -- z/base depend only on q, q_ln2, q_b) and this function
		// checks only z/base, not the final (base^2+q_c)>>z step, so no real q_c is
		// needed. This does NOT justify asserting kOk specifically: q_c=0 is a
		// fabricated probe value, not this row's real q_c, so its actual domain
		// (kOk vs kNotRepresentable) at q_c=0 is not a claim about the row as
		// originally fixtured. What IS true regardless of q_c, confirmed for every
        // one of these 36 rows before porting (all have q<=0, 1<=q_ln2<=ceiling --
		// verified above -- and |q_b| at most ~1.7e9, far below where q_p+q_b could
		// overflow int64): the decomposition is well-formed, so *out is filled and
		// the domain is one of {kOk, kNotRepresentable} -- never one of the three
		// kBad* outcomes, which would leave *out untouched and make the z/base
		// comparisons below meaningless.
		CHECK_MSG(got == IExpDomain::kOk || got == IExpDomain::kNotRepresentable,
		          "%s: IExpConstruct(%lld, %lld, %lld, 0) returned %s, want kOk or "
		          "kNotRepresentable (this row's decomposition is well-formed regardless of q_c)",
		          c.label, static_cast<long long>(c.q), static_cast<long long>(c.q_ln2),
		          static_cast<long long>(c.q_b), IExpDomainName(got));
		if (got != IExpDomain::kOk && got != IExpDomain::kNotRepresentable) continue;
		CHECK_MSG(out.z() == c.expected_z,
		          "%s: IExpConstruct(q=%lld, q_ln2=%lld, q_b=%lld).z() == %lld, want %lld "
		          "(independently derived)",
		          c.label, static_cast<long long>(c.q), static_cast<long long>(c.q_ln2),
		          static_cast<long long>(c.q_b), static_cast<long long>(out.z()),
		          static_cast<long long>(c.expected_z));
		CHECK_MSG(out.base() == c.expected_base,
		          "%s: IExpConstruct(q=%lld, q_ln2=%lld, q_b=%lld).base() == %lld, want %lld "
		          "(independently derived)",
		          c.label, static_cast<long long>(c.q), static_cast<long long>(c.q_ln2),
		          static_cast<long long>(c.q_b), static_cast<long long>(out.base()),
		          static_cast<long long>(c.expected_base));
		CHECK_MSG(out.q_c() == 0,
		          "%s: IExpConstruct(q=%lld, q_ln2=%lld, q_b=%lld, q_c=0).q_c() == %lld, want 0 -- "
		          "q_c is carried by the construction unchanged from what was passed in",
		          c.label, static_cast<long long>(c.q), static_cast<long long>(c.q_ln2),
		          static_cast<long long>(c.q_b), static_cast<long long>(out.q_c()));
		// IExpShift's own documented postcondition (z in [0, I_EXP_CLIP_N]),
		// checked live -- ported unchanged from TestIExpShiftMatchesIndependentlyDerivedZ.
		CHECK_MSG(out.z() >= 0 && out.z() <= I_EXP_CLIP_N,
		          "%s: IExpConstruct(q=%lld, q_ln2=%lld, q_b=%lld).z() == %lld, outside documented "
		          "[0, %d]",
		          c.label, static_cast<long long>(c.q), static_cast<long long>(c.q_ln2),
		          static_cast<long long>(c.q_b), static_cast<long long>(out.z()), I_EXP_CLIP_N);
	}
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

// --- S-HARDEN-4 (F2, SuperSLM_Plan.md §17.3 cell 8): the S2.5 acceptance gate, not a
//     smoke bound. The S2.5 closeout record pinned max_abs_error <= 0.5 * output_scale
//     from a scratch measurement across all six committed composition cases (worst
//     observed ratio 0.3287-0.4990 x output_scale) -- but the committed suite exercised
//     only kCompositionCases[0], against a 4x+1 smoke bound roughly eight times looser,
//     and printed the true tolerance as "owed, not asserted". This test is the missing
//     committed fixture: every one of the six cases is measured and gated against the
//     pinned 0.5x coefficient, and the six-case aggregate maximum and mean are pinned
//     as regression constants below. A regression that would have passed the old 4x+1
//     smoke bound now fails here (mutation-proved; see the build log).
//
//     Reference is the UNSCALED, code-level float32 matmul (the C24/C25 weight-scale
//     fold is out of scope, design S9): both operands are used as their raw int8
//     numeric value in float32, matching this component's own scope boundary.
//
//     UPDATE PROTOCOL for kPinnedAggregateMaxAbsError / kPinnedAggregateMeanAbsError:
//     these two constants may be revised only when (1) an intentional, reviewed change
//     to the S2.5 matmul/requant pipeline (C17-C22) or this op-level dequant reference
//     changes the measurement's true error characteristics -- never to silence a
//     failing assertion; (2) the new values are re-measured by running this exact test
//     against the changed implementation over the same six committed composition
//     cases; (3) the new literals land in the same commit as the implementation change
//     that moved them, with the decision log recording why and citing the measurement;
//     and (4) the per-case 0.5x bound (kPerCaseBoundCoefficient) is not weakened as
//     part of the same change -- a wider aggregate pin does not license a wider
//     per-case bound, and vice versa. ---

static void TestS2Point5SixCaseAcceptanceGateMeasurement() {
	using namespace superslm_test;

	// The pinned per-case acceptance coefficient (F2): the closeout record's measured
	// worst-case ratio across all six cases was 0.3287-0.4990 x output_scale; 0.5x is
	// the pinned acceptance bound, not the smoke bound's 4x+1.
	const double kPerCaseBoundCoefficient = 0.5;

	// Six-case aggregate regression constants (see UPDATE PROTOCOL above before
	// changing either). Measured by running this exact test against
	// D:\SuperSLM @ ee77e8a (the S2.5 kernel and requant chain are unchanged by
	// S-HARDEN-4 -- this slot adds the missing assertion, not a kernel fix).
	const double kPinnedAggregateMaxAbsError = 2111.34375;
	const double kPinnedAggregateMeanAbsError = 626.114286634657;
	// Absorbs float32 summation-order noise across in_channels up to 8960 under
	// different compilers' auto-vectorization of the reference accumulation loop
	// (-ffp-contract=off / /fp:precise forbid FMA contraction project-wide, but not
	// reassociation); at these magnitudes a real regression is the whole point of the
	// smoke-bound gap this test closes (roughly 8x), several orders of magnitude
	// above this tolerance.
	const double kAggregateTolerance = 0.5;

	double aggregate_max_abs_error = 0.0;
	double sum_of_case_means = 0.0;

	for (size_t i = 0; i < kCompositionCasesCount; ++i) {
		const CompositionCase& c = kCompositionCases[i];

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
		const double per_case_bound = kPerCaseBoundCoefficient * static_cast<double>(output_scale);

		std::printf(
		    "S2.5 acceptance gate (%s): max |error| = %.6f, mean |error| = %.6f, "
		    "output_scale = %.6f (D'=%lld), per-case bound (0.5x) = %.6f -- asserted (F2, "
		    "S-HARDEN-4)\n",
		    c.label, max_abs_error, mean_abs_error, static_cast<double>(output_scale),
		    static_cast<long long>(d_prime), per_case_bound);

		CHECK_MSG(max_abs_error <= per_case_bound,
		          "%s: max |dequant - float32 reference| = %.6f exceeds the pinned S2.5 "
		          "acceptance bound 0.5 * output_scale = %.6f (F2, S-HARDEN-4)",
		          c.label, max_abs_error, per_case_bound);

		aggregate_max_abs_error = std::max(aggregate_max_abs_error, max_abs_error);
		sum_of_case_means += mean_abs_error;
	}

	const double aggregate_mean_abs_error =
	    sum_of_case_means / static_cast<double>(kCompositionCasesCount);

	CHECK_MSG(std::fabs(aggregate_max_abs_error - kPinnedAggregateMaxAbsError) <= kAggregateTolerance,
	          "S2.5 six-case aggregate max |error| = %.6f moved past the pinned regression "
	          "constant %.6f +/- %.6f -- see this test's UPDATE PROTOCOL before changing the pin",
	          aggregate_max_abs_error, kPinnedAggregateMaxAbsError, kAggregateTolerance);
	CHECK_MSG(std::fabs(aggregate_mean_abs_error - kPinnedAggregateMeanAbsError) <= kAggregateTolerance,
	          "S2.5 six-case aggregate mean |error| = %.6f moved past the pinned regression "
	          "constant %.6f +/- %.6f -- see this test's UPDATE PROTOCOL before changing the pin",
	          aggregate_mean_abs_error, kPinnedAggregateMeanAbsError, kAggregateTolerance);
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

// ---------------------------------------------------------------------------
// S2.5 golden hash (design §10 step 5 / §11 item 6): a pinned SHA-256 over the
// matmul kernel's outputs across a canonical input set. This is the cross-
// platform/toolchain/ISA determinism gate — every platform in the CI matrix must
// reproduce this exact hash bit-for-bit. macos-arm64 is the load-bearing member:
// src/matmul.cpp's SSE2 specialization is x64-only, so that runner exercises the
// scalar reference while the others exercise SSE2, and a matching hash across the
// matrix IS the scalar ≡ SIMD cross-ISA proof rather than a repeated x64 result.
//
// The pinned hash is computed by tools/gen_matmul_golden.py in arbitrary-precision
// Python integer arithmetic reproducing the design's §5 scalar reference — not by
// recording what a C++ build printed — so agreement here is agreement with an
// independent computation. The inputs are produced by a pinned 64-bit LCG (pure
// uint64 arithmetic, exactly defined on every toolchain, calling nothing), mirrored
// bit-for-bit from that generator; kMatmulGoldenLcgProbe pins the LCG's own first
// bytes so a drift in this mirror fails as its own named check instead of
// masquerading as a kernel determinism break. See the generator's header comment for
// why matmul generates its inputs where S2.4 pinned its LUT table as data.
// ---------------------------------------------------------------------------
namespace {

// Mirror of tools/gen_matmul_golden.py's Lcg. PCG's multiplier/increment; bytes taken
// from the HIGH end of the state, since an LCG's low bits have short periods.
struct MatmulGoldenLcg {
	uint64_t x;
	explicit MatmulGoldenLcg(uint64_t seed) : x(seed) {}
	uint8_t NextByte() {
		x = x * 6364136223846793005ull + 1442695040888963407ull;
		return static_cast<uint8_t>((x >> 56) & 0xFFu);
	}
};

// int8 activation code in [-127, 127] — C22's RequantTokenCode never emits -128 and
// matmul.h's contract states that range, so the golden must not feed a code the
// runtime cannot produce. Weights carry no such exclusion and use full [-128, 127].
inline int8_t MatmulGoldenActCode(uint8_t b) {
	return static_cast<int8_t>(static_cast<int>(b % 255u) - 127);
}
inline int8_t MatmulGoldenWgtCode(uint8_t b) {
	return static_cast<int8_t>(static_cast<int>(b) - 128);
}

inline void AppendI64Le(std::vector<uint8_t>& out, int64_t v) {
	const uint64_t u = static_cast<uint64_t>(v);
	for (int i = 0; i < 8; ++i) out.push_back(static_cast<uint8_t>((u >> (8 * i)) & 0xFFu));
}
inline void AppendI32Le(std::vector<uint8_t>& out, int32_t v) {
	const uint32_t u = static_cast<uint32_t>(v);
	for (int i = 0; i < 4; ++i) out.push_back(static_cast<uint8_t>((u >> (8 * i)) & 0xFFu));
}

}  // namespace

static void TestMatmulGoldenHashCrossPlatform() {
	using namespace superslm_test;
	using superslm::DotRowScalarRef;
	using superslm::GemmInt8Accumulate;
	using superslm::NarrowAccumulatorToI32;

	// The LCG mirror itself, checked before anything depends on it.
	{
		MatmulGoldenLcg probe(1ull);
		for (size_t i = 0; i < sizeof(kMatmulGoldenLcgProbe); ++i) {
			const uint8_t got = probe.NextByte();
			CHECK_MSG(got == kMatmulGoldenLcgProbe[i],
			          "matmul golden LCG mirror diverged at byte %zu: produced %u, generator "
			          "pinned %u -- the C++ mirror no longer matches tools/gen_matmul_golden.py "
			          "(this is a generator-mirror drift, NOT a kernel determinism break)",
			          i, static_cast<unsigned>(got),
			          static_cast<unsigned>(kMatmulGoldenLcgProbe[i]));
		}
	}

	std::vector<uint8_t> bytes;
	bytes.reserve(kMatmulGoldenTotalBytes);

	for (const MatmulGoldenCase& c : kMatmulGoldenCases) {
		std::vector<int8_t> acts(c.num_tokens * c.in_channels);
		std::vector<int8_t> wgts(c.out_channels * c.in_channels);
		if (c.kind == 1) {  // alternating int8-extremes pattern
			for (size_t t = 0; t < c.num_tokens; ++t) {
				for (size_t k = 0; k < c.in_channels; ++k) {
					acts[t * c.in_channels + k] = static_cast<int8_t>((k % 2 == 0) ? 127 : -127);
				}
			}
			for (size_t j = 0; j < c.out_channels; ++j) {
				for (size_t k = 0; k < c.in_channels; ++k) {
					wgts[j * c.in_channels + k] = static_cast<int8_t>(((j + k) % 2 == 0) ? 127 : -128);
				}
			}
		} else if (c.kind == 2 || c.kind == 3) {
			// Single-signed attainable extremes: every product at magnitude 16,256
			// (±127 against -128) and all one sign, so the sum grows monotonically and
			// leaves int32 at the deep lengths. The LCG fills cannot reach here — their
			// mixed signs cancel — so these are the only cases that put the int64
			// accumulate itself inside the golden.
			const int8_t a = static_cast<int8_t>(c.kind == 2 ? 127 : -127);
			for (size_t i = 0; i < acts.size(); ++i) acts[i] = a;
			for (size_t i = 0; i < wgts.size(); ++i) wgts[i] = static_cast<int8_t>(-128);
		} else {  // pinned-LCG fill; activation rows first, then weight rows, one stream
			MatmulGoldenLcg g(c.seed);
			for (size_t i = 0; i < acts.size(); ++i) acts[i] = MatmulGoldenActCode(g.NextByte());
			for (size_t i = 0; i < wgts.size(); ++i) wgts[i] = MatmulGoldenWgtCode(g.NextByte());
		}

		// 1. Every int64 accumulator the shipping dispatch produces, row-major
		//    [num_tokens, out_channels] -- SSE2 on x64, the scalar reference on arm64.
		std::vector<int64_t> wide(c.num_tokens * c.out_channels);
		GemmInt8Accumulate(acts.data(), wgts.data(), c.num_tokens, c.in_channels, c.out_channels,
		                   wide.data());
		for (int64_t v : wide) AppendI64Le(bytes, v);

		// 2. The normative §5 scalar construction on row 0 / output channel 0, so it is
		//    inside the golden even on x64, where DotRow never dispatches to it.
		AppendI64Le(bytes, DotRowScalarRef(acts.data(), wgts.data(), c.in_channels));

		// 3. The narrowed int32 row 0, for the cases §8's 131,071 bound declares Int32.
		//    The deep case is deliberately excluded: at 132,105 the declared width is
		//    Int64, and narrowing there would be the caller-contract UB matmul.h names.
		if (c.int32_safe) {
			std::vector<int32_t> narrowed(c.out_channels);
			NarrowAccumulatorToI32(wide.data(), c.out_channels, narrowed.data());
			for (int32_t v : narrowed) AppendI32Le(bytes, v);
		}
	}

	CHECK_MSG(bytes.size() == kMatmulGoldenTotalBytes,
	          "matmul golden byte stream is %zu bytes, generator pinned %zu -- the C++ and Python "
	          "canonical sets have diverged in shape, so the hash comparison below would be "
	          "meaningless",
	          bytes.size(), kMatmulGoldenTotalBytes);

	uint8_t digest[32];
	superslm::Sha256Hash(bytes.data(), bytes.size(), digest);
	const std::string hex = superslm::ToHex(digest);
	std::printf("S2.5 matmul golden hash: %s (%zu cases, %zu bytes)\n", hex.c_str(),
	            sizeof(kMatmulGoldenCases) / sizeof(kMatmulGoldenCases[0]), bytes.size());
	CHECK_MSG(hex == std::string(kMatmulGoldenHash),
	          "matmul golden hash %s != pinned %s (cross-platform/toolchain/ISA determinism break, "
	          "or the pin needs regenerating for an intended construction change -- re-run "
	          "tools/gen_matmul_golden.py deliberately)",
	          hex.c_str(), kMatmulGoldenHash);
}

// --- S-HARDEN-3 (F13, §13 item 7, §17.3 cell 4): the independent converter
//     verifier's core -- config geometry x tensor shapes, per-tensor evidence,
//     and the proof-manifest document itself. Curie's red suite (red-first;
//     the whole superslm::proof_manifest translation unit did not exist before
//     this slot -- every symbol below is new). Mendeleev's 2026-07-21 coverage
//     audit §3.3 names the zero-boundary cells explicitly: kv_heads == 0 and,
//     separately, heads == 0, must each produce a DEFINED rejection rather than
//     a fault in the `heads % kv_heads` modulus -- the two cells directly below
//     are that specification, unmodified. ---

static void TestConfigGeometryRejectsZeroAttentionHeads() {
	const auto r = CheckConfigGeometry(/*hidden_size=*/4096, /*heads=*/0, /*kv_heads=*/8, /*head_dim=*/128);
	CHECK_MSG(r.status == ConfigGeometryStatus::ZeroAttentionHeads,
	          "num_attention_heads == 0 must be a DEFINED rejection (ZeroAttentionHeads), checked before "
	          "any modulus, not a crash: got %s",
	          ConfigGeometryStatusName(r.status));
}

static void TestConfigGeometryRejectsZeroKeyValueHeadsBeforeModulusFaults() {
	// The coverage audit's own specification (§3.3): "heads % kv_heads faults at
	// the validation site itself when kv_heads == 0, so the check crashes before
	// any rejection can fire." This call reaching a CHECK at all (rather than a
	// SIGFPE from an integer division by zero) is itself part of what this cell
	// proves -- the zero must be caught before the modulus in CheckConfigGeometry's
	// own body ever executes.
	const auto r = CheckConfigGeometry(/*hidden_size=*/4096, /*heads=*/32, /*kv_heads=*/0, /*head_dim=*/128);
	CHECK_MSG(r.status == ConfigGeometryStatus::ZeroKeyValueHeads,
	          "num_key_value_heads == 0 must be a DEFINED rejection (ZeroKeyValueHeads), checked before "
	          "`heads %% kv_heads` is ever evaluated: got %s",
	          ConfigGeometryStatusName(r.status));
}

static void TestConfigGeometryRejectsKvHeadsExceedsHeads() {
	const auto r = CheckConfigGeometry(/*hidden_size=*/4096, /*heads=*/8, /*kv_heads=*/16, /*head_dim=*/512);
	CHECK_MSG(r.status == ConfigGeometryStatus::KvHeadsExceedsHeads,
	          "num_key_value_heads (16) > num_attention_heads (8): got %s", ConfigGeometryStatusName(r.status));
}

static void TestConfigGeometryRejectsHeadsNotDivisibleByKv() {
	// 10 heads, 3 kv_heads: 10 % 3 != 0. kv_heads <= heads holds, so this cell
	// isolates the divisibility relation from the ordering relation above it.
	const auto r = CheckConfigGeometry(/*hidden_size=*/4096, /*heads=*/10, /*kv_heads=*/3, /*head_dim=*/128);
	CHECK_MSG(r.status == ConfigGeometryStatus::HeadsNotDivisibleByKv,
	          "num_attention_heads (10) %% num_key_value_heads (3) != 0: got %s",
	          ConfigGeometryStatusName(r.status));
}

static void TestConfigGeometryRejectsHiddenSizeMismatch() {
	// heads=32, head_dim=128 -> 4096, but hidden_size declares 4097: the GQA
	// relations both hold, isolating the hidden_size x heads*head_dim relation.
	const auto r = CheckConfigGeometry(/*hidden_size=*/4097, /*heads=*/32, /*kv_heads=*/8, /*head_dim=*/128);
	CHECK_MSG(r.status == ConfigGeometryStatus::HiddenSizeGeometryMismatch,
	          "hidden_size (4097) != num_attention_heads * head_dim (32*128=4096): got %s",
	          ConfigGeometryStatusName(r.status));
}

static void TestConfigGeometryAcceptsGqaShape() {
	const auto r = CheckConfigGeometry(/*hidden_size=*/4096, /*heads=*/32, /*kv_heads=*/8, /*head_dim=*/128);
	CHECK_MSG(r.status == ConfigGeometryStatus::Ok,
	          "a coherent GQA shape (32 heads, 8 kv_heads, head_dim=128, hidden_size=4096) must be Ok: got "
	          "%s (%s)",
	          ConfigGeometryStatusName(r.status), r.diagnostic.c_str());
}

static void TestConfigGeometryAcceptsMhaShape() {
	// kv_heads == heads (plain multi-head attention, no grouping) is the
	// degenerate-but-valid case of the same relations, not a special case.
	const auto r = CheckConfigGeometry(/*hidden_size=*/2048, /*heads=*/16, /*kv_heads=*/16, /*head_dim=*/128);
	CHECK_MSG(r.status == ConfigGeometryStatus::Ok, "kv_heads == heads (plain MHA) must be Ok: got %s (%s)",
	          ConfigGeometryStatusName(r.status), r.diagnostic.c_str());
}

// --- Per-tensor evidence: a feature oracle grounded in known planted values,
//     not a recode of ComputeTensorEvidence's own arithmetic. A single WGT1
//     tensor's four bytes are patched directly to -128, -128, 127, 5 after the
//     spec-faithful manifest builder lays out the section, so the assertion
//     below is checked against values this test wrote, not against whatever
//     the implementation happens to compute. ---

static void TestComputeTensorEvidenceReportsExtremaAndSaturationBoundary() {
	using namespace superslm_test;
	auto manifest = MakeSingleTensorManifest(superslm::kWeightsMagic, /*element_size=*/1, /*shape=*/{4});
	const size_t data_off = static_cast<size_t>(manifest.tensor_data_off[0]);
	manifest.bytes[data_off + 0] = static_cast<uint8_t>(int8_t(-128));  // dtype minimum
	manifest.bytes[data_off + 1] = static_cast<uint8_t>(int8_t(-128));  // dtype minimum, second hit
	manifest.bytes[data_off + 2] = static_cast<uint8_t>(int8_t(127));   // dtype maximum
	manifest.bytes[data_off + 3] = static_cast<uint8_t>(int8_t(5));     // interior, not a boundary

	SslmSectionView view = MakeManifestSectionView(SslmSectionType::Weights, SslmDtype::Int8, manifest.bytes);
	SslmTensorManifest parsed;
	std::string err;
	SslmModelStatus status = SslmTensorManifest::Parse(view, parsed, &err);
	CHECK_MSG(status == SslmModelStatus::Ok, "fixture manifest failed to parse: %s (%s)",
	          SslmModelStatusName(status), err.c_str());
	if (status != SslmModelStatus::Ok) return;

	auto evidence = ComputeTensorEvidence(parsed, SslmDtype::Int8);
	CHECK_MSG(evidence.size() == 1, "expected 1 tensor of evidence, got %zu", evidence.size());
	if (evidence.size() != 1) return;
	CHECK(evidence[0].name == "t0");
	CHECK(evidence[0].elem_count == 4);
	CHECK_MSG(evidence[0].min_value == -128, "min_value: got %lld, want -128",
	          (long long)evidence[0].min_value);
	CHECK_MSG(evidence[0].max_value == 127, "max_value: got %lld, want 127", (long long)evidence[0].max_value);
	CHECK_MSG(evidence[0].saturation_lo_count == 2, "saturation_lo_count: got %llu, want 2",
	          (unsigned long long)evidence[0].saturation_lo_count);
	CHECK_MSG(evidence[0].saturation_hi_count == 1, "saturation_hi_count: got %llu, want 1",
	          (unsigned long long)evidence[0].saturation_hi_count);
}

static void TestComputeWeightScaleEvidenceReportsShiftRangeAndIdentityCount() {
	using namespace superslm_test;
	// Two rows: (identity=1, mult=0, shift=0) and (identity=0, mult=99, shift=31)
	// -- shift_min/max isolate the fold triple's shift column; identity_count
	// isolates the {0,1} flag column, independent of mult (deliberately
	// unbounded per D-SLM142, so mult=99 must not affect either reported field).
	auto manifest = MakeSingleTensorManifest(superslm::kWeightScalesMagic, /*element_size=*/4, /*shape=*/{2, 3});
	const size_t data_off = static_cast<size_t>(manifest.tensor_data_off[0]);
	PutU32(manifest.bytes, data_off + 0, 1);   // row0 identity
	PutU32(manifest.bytes, data_off + 4, 0);   // row0 mult
	PutU32(manifest.bytes, data_off + 8, 0);   // row0 shift
	PutU32(manifest.bytes, data_off + 12, 0);  // row1 identity
	PutU32(manifest.bytes, data_off + 16, 99); // row1 mult
	PutU32(manifest.bytes, data_off + 20, 31); // row1 shift

	SslmSectionView view = MakeManifestSectionView(SslmSectionType::WeightScales, SslmDtype::Int32, manifest.bytes);
	SslmTensorManifest parsed;
	std::string err;
	SslmModelStatus status = SslmTensorManifest::Parse(view, parsed, &err);
	CHECK_MSG(status == SslmModelStatus::Ok, "fixture manifest failed to parse: %s (%s)",
	          SslmModelStatusName(status), err.c_str());
	if (status != SslmModelStatus::Ok) return;

	auto evidence = ComputeWeightScaleEvidence(parsed);
	CHECK_MSG(evidence.size() == 1, "expected 1 tensor of evidence, got %zu", evidence.size());
	if (evidence.size() != 1) return;
	CHECK(evidence[0].row_count == 2);
	CHECK_MSG(evidence[0].shift_min == 0, "shift_min: got %d, want 0", evidence[0].shift_min);
	CHECK_MSG(evidence[0].shift_max == 31, "shift_max: got %d, want 31", evidence[0].shift_max);
	CHECK_MSG(evidence[0].identity_count == 1, "identity_count: got %llu, want 1",
	          (unsigned long long)evidence[0].identity_count);
}

static void TestHashSectionHexMatchesIndependentSha256() {
	using namespace superslm_test;
	auto manifest = MakeMinimalValidManifest(superslm::kWeightsMagic, /*element_size=*/1);
	SslmSectionView view = MakeManifestSectionView(SslmSectionType::Weights, SslmDtype::Int8, manifest.bytes);

	uint8_t digest[32];
	Sha256Hash(view.data, static_cast<size_t>(view.byte_size), digest);
	const std::string want = ToHex(digest);

	const std::string got = HashSectionHex(view);
	CHECK_MSG(got == want, "HashSectionHex diverged from a direct Sha256Hash call over the same bytes: got "
	          "%s, want %s",
	          got.c_str(), want.c_str());
}

// --- The proof manifest document, driven through a real Load()-accepted view
//     -- both the geometry-coherent and geometry-incoherent cases, so the
//     manifest's own "config_geometry" field is proven to actually reflect the
//     independent check rather than a hardcoded "ok". ---

static void TestBuildProofManifestJsonReportsGeometryOkOnCoherentArtifact() {
	using namespace superslm_test;
	Cfg1Spec coherent;
	coherent.num_attention_heads = 32;
	coherent.num_key_value_heads = 8;
	coherent.head_dim = 128;
	coherent.hidden_size = 32 * 128;  // = 4096, coherent with heads*head_dim
	FixtureSection cfg = MakeSection(SslmSectionType::Config, SslmDtype::Raw, BuildCfg1(coherent));
	auto built = BuildArtifact({cfg, MakeSigmoidLutSection()});

	SslmModelView view;
	std::string err;
	SslmModelStatus status = SslmModel::Load(built.bytes.data(), built.bytes.size(), view, &err);
	CHECK_MSG(status == SslmModelStatus::Ok, "coherent-geometry fixture failed to load: got %s (%s)",
	          SslmModelStatusName(status), err.c_str());
	if (status != SslmModelStatus::Ok) return;

	SslmArtifact artifact;
	SslmError aerr;
	SslmArtifact::OpenFromMemory(built.bytes.data(), built.bytes.size(), artifact, &aerr);
	const std::string manifest = BuildProofManifestJson(artifact);
	CHECK_MSG(manifest.find("\"ok\": true") != std::string::npos,
	          "proof manifest for a geometry-coherent artifact must report config_geometry.ok == true; "
	          "manifest:\n%s",
	          manifest.c_str());
	CHECK(manifest.find(artifact.FingerprintHex()) != std::string::npos);
}

static void TestBuildProofManifestJsonReportsGeometryMismatchOnIncoherentArtifact() {
	using namespace superslm_test;
	// Cfg1Spec{}'s own defaults: 24 heads * 128 head_dim = 3072 != hidden_size
	// 4096 -- an incoherent shape by construction, unrelated to this slot; every
	// existing CFG1 fixture in this suite uses these defaults, which is exactly
	// why the geometry check is NOT wired into SslmModel::Load (it would break
	// every one of them for a relation no runtime kernel yet consumes).
	auto built = BuildArtifact({MakeValidConfigSection(), MakeSigmoidLutSection()});

	SslmModelView view;
	std::string err;
	SslmModelStatus status = SslmModel::Load(built.bytes.data(), built.bytes.size(), view, &err);
	CHECK_MSG(status == SslmModelStatus::Ok, "incoherent-geometry-but-otherwise-valid fixture failed to "
	          "load: got %s (%s)",
	          SslmModelStatusName(status), err.c_str());
	if (status != SslmModelStatus::Ok) return;

	SslmArtifact artifact;
	SslmError aerr;
	SslmArtifact::OpenFromMemory(built.bytes.data(), built.bytes.size(), artifact, &aerr);
	const std::string manifest = BuildProofManifestJson(artifact);
	CHECK_MSG(manifest.find("\"ok\": false") != std::string::npos,
	          "proof manifest for Cfg1Spec{}'s incoherent default shape (24*128=3072 != hidden_size 4096) "
	          "must report config_geometry.ok == false; manifest:\n%s",
	          manifest.c_str());
	CHECK(manifest.find("HiddenSizeGeometryMismatch") != std::string::npos);
}

// ---------------------------------------------------------------------------
// Curie's S-HARDEN-7 suite (Claude/Vitruvius/SuperSLM_SHARDEN678_Bundle_
// Design-2026-07-23.md §3; T-411): the "throws only std::bad_alloc" contract.
// Every cell below is authored against the corrected, four-condition
// membership rule's population of EIGHTEEN sites (§3.1's table) -- not the
// twelve, ten, six, or seven a hand enumeration found across earlier passes
// of the design.
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

// --- Site 1/18: SslmArtifact::OpenFromMemory (artifact.h:149) ---
static void TestBadAllocContractOpenFromMemory() {
	uint8_t data[4] = {'S', 'S', 'L', 'M'};
	SslmArtifact out;
	SslmError err;
	CheckBadAllocContractSite("SslmArtifact::OpenFromMemory", [&] {
		SslmArtifact::OpenFromMemory(data, sizeof(data), out, &err);
	});
}

// --- Site 1/18, representative marker cell (design §3.2 "New cell -- force a
//     genuine std::bad_alloc through the wrap"): the shared wrap helper must
//     take the catch(const std::bad_alloc&){throw;} clause specifically, not
//     merely produce an observably-equal std::bad_alloc via the general
//     catch(const std::exception&) clause. One representative site stands for
//     the mechanism per §17 dimension 11's usual population-validation shape
//     (the shared helper makes this true for all eighteen sites by
//     construction). ---
static void TestBadAllocContractOpenFromMemoryPassthroughClauseIsSpecific() {
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

// --- Site 2/18: SslmArtifact::OpenFromFile (artifact.h:155) ---
static void TestBadAllocContractOpenFromFile() {
	SslmArtifact out;
	SslmError err;
	CheckBadAllocContractSite("SslmArtifact::OpenFromFile", [&] {
		SslmArtifact::OpenFromFile("this/path/does/not/exist.sslm", out, &err);
	});
}

// --- Site 3/18 (this fold's addition, condition 4(b)): SslmArtifact::
//     FingerprintHex (artifact.h:162) ---
static void TestBadAllocContractFingerprintHex() {
	SslmArtifact out;
	CheckBadAllocContractSite("SslmArtifact::FingerprintHex", [&] {
		(void)out.FingerprintHex();
	});
}

// --- Site 4/18: SslmTensorManifest::Parse (model.h:178), the direct-call
//     path ---
static void TestBadAllocContractTensorManifestParseDirect() {
	SslmSectionView section{};
	SslmTensorManifest out;
	std::string err;
	CheckBadAllocContractSite("SslmTensorManifest::Parse (direct)", [&] {
		SslmTensorManifest::Parse(section, out, &err);
	});
}

// --- Site 4/18, the Load-mediated path: the S-HARDEN-7 design's own defect
//     class (design §3.1: "Load's wrap ... is not a substitute for [a
//     site's] own independent wrap") -- confirms the seam is consulted by the
//     *Impl body itself, not only by whatever calls it directly, using a
//     fully valid artifact whose Weights section routes through
//     SslmTensorManifest::Parse from inside SslmModel::Load. ---
static void TestBadAllocContractTensorManifestParseViaLoad() {
	auto built = BuildFullyValidV2ArtifactForLoad();
	SslmModelView view;
	std::string err;
	CheckBadAllocContractSite("SslmTensorManifest::Parse (via SslmModel::Load)", [&] {
		SslmModel::Load(built.bytes.data(), built.bytes.size(), view, &err);
	});
}

// --- Site 5/18: SslmKeyedConstants::Parse (model.h:219) ---
static void TestBadAllocContractKeyedConstantsParse() {
	SslmSectionView section{};
	SslmKeyedConstants out;
	std::string err;
	CheckBadAllocContractSite("SslmKeyedConstants::Parse", [&] {
		SslmKeyedConstants::Parse(section, out, &err);
	});
}

// --- Site 6/18: ParseConfig (model.h:239) ---
static void TestBadAllocContractParseConfig() {
	SslmSectionView section{};
	SslmModelConfig out{};
	std::string err;
	CheckBadAllocContractSite("ParseConfig", [&] {
		ParseConfig(section, out, &err);
	});
}

// --- Site 7/18: ParseSigmoidLut (model.h:261) ---
static void TestBadAllocContractParseSigmoidLut() {
	SslmSectionView section{};
	SslmSigmoidLut out{};
	std::string err;
	CheckBadAllocContractSite("ParseSigmoidLut", [&] {
		ParseSigmoidLut(section, out, &err);
	});
}

// --- Site 8/18: SslmModel::Load (model.h:416), its own unwrapped surface
//     (model.cpp:711,788's string concatenations) -- exercised here via the
//     null-data path, which reaches Load's own body before any sub-parser
//     runs. ---
static void TestBadAllocContractLoad() {
	SslmModelView out;
	std::string err;
	CheckBadAllocContractSite("SslmModel::Load", [&] {
		SslmModel::Load(nullptr, 0, out, &err);
	});
}

// --- Site 9/18: TokenizerView::Open (tokenizer.h:35), the direct-call path
//     (tools/tok_verify.cpp's bypass shape) ---
static void TestBadAllocContractTokenizerOpenDirect() {
	SslmArtifact artifact;  // default: no sections, Ok() == false
	TokenizerView out;
	std::string err;
	CheckBadAllocContractSite("TokenizerView::Open (direct)", [&] {
		TokenizerView::Open(artifact, out, &err);
	});
}

// --- Site 9/18, the Load-mediated path: TokenizerView::Open called from
//     inside SslmModel::Load when a Tokenizer section is present -- the exact
//     bypass shape this fold's own §3.1 documents as previously missed. Uses
//     the real Qwen2.5-1.5B fixture artifact (the only fixture in this suite
//     that carries a genuine Tokenizer + UnicodeTables pair), read as raw
//     bytes and driven through SslmModel::Load directly rather than through
//     SslmArtifact::OpenFromFile. ---
static void TestBadAllocContractTokenizerOpenViaLoad() {
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

// --- Site 10/18 (this fold's addition, condition 4(b)): TokenizerView::Encode
//     (tokenizer.h:44) ---
static void TestBadAllocContractTokenizerEncode() {
	auto ft = OpenFixtureTokenizer();
	CHECK_MSG(ft.view_ok, "TokenizerView::Open failed for the Encode injection fixture: %s",
	          ft.view_error.c_str());
	if (!ft.view_ok) return;
	CheckBadAllocContractSite("TokenizerView::Encode", [&] {
		(void)ft.view.Encode("the quick brown fox");
	});
}

// --- Site 11/18 (this fold's addition, condition 4(b)): TokenizerView::Decode
//     (tokenizer.h:48) ---
static void TestBadAllocContractTokenizerDecode() {
	auto ft = OpenFixtureTokenizer();
	CHECK_MSG(ft.view_ok, "TokenizerView::Open failed for the Decode injection fixture: %s",
	          ft.view_error.c_str());
	if (!ft.view_ok) return;
	std::vector<int32_t> ids = {1, 2, 3};
	CheckBadAllocContractSite("TokenizerView::Decode", [&] {
		(void)ft.view.Decode(ids);
	});
}

// --- Site 12/18 (this fold's addition, condition 4(b)): ComputeTensorEvidence
//     (proof_manifest.h:84) ---
static void TestBadAllocContractComputeTensorEvidence() {
	SslmTensorManifest manifest;
	CheckBadAllocContractSite("ComputeTensorEvidence", [&] {
		(void)ComputeTensorEvidence(manifest, SslmDtype::Int8);
	});
}

// --- Site 13/18 (this fold's addition, condition 4(b)):
//     ComputeWeightScaleEvidence (proof_manifest.h:99) ---
static void TestBadAllocContractComputeWeightScaleEvidence() {
	SslmTensorManifest manifest;
	CheckBadAllocContractSite("ComputeWeightScaleEvidence", [&] {
		(void)ComputeWeightScaleEvidence(manifest);
	});
}

// --- Site 14/18: HashSectionHex (proof_manifest.h:106) ---
static void TestBadAllocContractHashSectionHex() {
	SslmSectionView section{};
	CheckBadAllocContractSite("HashSectionHex", [&] {
		(void)HashSectionHex(section);
	});
}

// --- Site 15/18: BuildProofManifestJson (proof_manifest.h:128) ---
static void TestBadAllocContractBuildProofManifestJson() {
	SslmArtifact artifact;
	CheckBadAllocContractSite("BuildProofManifestJson", [&] {
		(void)BuildProofManifestJson(artifact);
	});
}

// --- Site 16/18: Sha256::Update (sha256.h:19). Cannot currently throw
//     anything (src/sha256.cpp operates on fixed-size stack buffers only,
//     design §3.1) -- this cell proves the injection seam itself fires
//     rather than a pre-existing real-world leak, the same shape every other
//     site's cell uses, applied to a site whose current implementation
//     happens not to need it yet. ---
static void TestBadAllocContractSha256Update() {
	Sha256 h;
	const uint8_t byte = 'x';
	CheckBadAllocContractSite("Sha256::Update", [&] {
		h.Update(&byte, 1);
	});
}

// --- Site 17/18: Sha256Hash (sha256.h:32). Same "cannot currently throw"
//     shape as site 16. ---
static void TestBadAllocContractSha256HashFreeFunction() {
	const uint8_t byte = 'x';
	uint8_t digest[32];
	CheckBadAllocContractSite("Sha256Hash", [&] {
		Sha256Hash(&byte, 1, digest);
	});
}

// --- Site 18/18 (this fold's addition, condition 4(b)): superslm::ToHex
//     (sha256.h:35). A FREE FUNCTION in namespace superslm, not a Sha256
//     member -- the design text writes "Sha256::ToHex" throughout §3.1/§3.2/
//     §3.3, but sha256.h:35 declares it outside the Sha256 class (Weak
//     finding, fourth-pass temper). This cell references the correct symbol,
//     superslm::ToHex; the design's prose is corrected by the planner, not by
//     this test. Cannot practically reach std::length_error given a fixed
//     64-character output (design §3.1) -- same "seam-fires, not a real
//     leak" shape as sites 16-17. ---
static void TestBadAllocContractToHex() {
	uint8_t digest[32] = {};
	CheckBadAllocContractSite("superslm::ToHex", [&] {
		(void)ToHex(digest);
	});
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

static void TestRejectsNonZeroSectionDescriptorReservedField() {
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

// --- SuperSLM_S3a_WalkingSkeleton_Plan.md Sec11 S3.1 (T-200, board T-132, Sec7.2's
//     C30 derived-operand predicate; D-SLM318): the differential cell the plan asks
//     for ("the design's C30 site and IExpConstantsInDomain agree at every point,
//     e = -62, e = -61 and e = -60 included explicitly") is realized here against the
//     ALREADY-SHIPPED IExpConstantsInDomain -- the not-yet-built production wrapper
//     ("the design's C30 site") that would call it from the attention interior is
//     S3.3's build (Claude/Curie/superslm-s3.1-checked-chain-funnel-test-design-
//     2026-07-28.md Sec3.2 names this substitution explicitly, and names that once
//     the wrapper exists a thin delegation cell is still owed). The (q_ln2, q_b, q_c)
//     triples are C30's own derivation (iexp_scale_constants), called from the
//     vendored reference by tests/gen_s3_1_c30_iexp_domain_sweep_fixtures.py -- never
//     re-derived in this file. ---
static void TestIExpConstantsInDomainAgreesWithC30DerivedConstantsAcrossTheSweep() {
	using namespace superslm_test;
	int checked = 0;
	for (size_t i = 0; i < kC30IExpDomainSweepCasesCount; ++i) {
		const C30IExpDomainSweepCase& c = kC30IExpDomainSweepCases[i];
		if (!c.derivation_ok) continue;  // C30's OWN construction-domain rejection (a
		                                  // different, upstream guard than the one under
		                                  // test here) -- nothing to call the predicate with.
		// The q=0 representative is valid only where the shortcut's own precondition
		// holds (intmath.h: "2*q_b >= q_ln2 - 1 ... The first dominates"); verify it
		// rather than assume it, per every row the generator emitted.
		CHECK_MSG(c.shortcut_condition_holds,
		          "m=%lld e=%d: the q=0 shortcut's own precondition (2*q_b >= q_ln2-1) does "
		          "not hold for this row -- q=0 alone cannot stand in for the full per-element "
		          "sweep IExpConstantsInDomain's cost note requires here",
		          c.m, c.e);
		bool actual = IExpConstantsInDomain(0, c.q_ln2, c.q_b, c.q_c);
		CHECK_MSG(actual == c.expected_in_domain,
		          "m=%lld e=%d q_ln2=%lld q_b=%lld q_c=%lld: IExpConstantsInDomain(0, ...) "
		          "returned %s, want %s (arbitrary-precision oracle)",
		          c.m, c.e, c.q_ln2, c.q_b, c.q_c,
		          actual ? "true" : "false", c.expected_in_domain ? "true" : "false");
		++checked;
	}
	CHECK_MSG(checked > 0, "no sweep row had a valid C30 derivation to check -- fixture regressed");
}

// The three named disagreement points (D-SLM318's table), asserted explicitly and by
// name rather than only swept generically above -- T-1254's own discipline extended
// to this cell (a required-green witness stated in the open, not just implied by a
// loop). Both mantissa extremes at each e, so the mantissa-conditional branch at
// e=-61 is pinned on both sides of its own m >= 1,268,234,713 threshold.
static void TestC30DomainDisagreementPointsAreExplicitlyPinned() {
	using namespace superslm_test;
	constexpr int64_t kMLow = INT64_C(1073741824);   // 2^30
	constexpr int64_t kMHigh = INT64_C(2147483647);  // 2^31 - 1
	auto find_case = [](int64_t m, int e) -> const C30IExpDomainSweepCase& {
		for (size_t i = 0; i < kC30IExpDomainSweepCasesCount; ++i) {
			if (kC30IExpDomainSweepCases[i].m == m && kC30IExpDomainSweepCases[i].e == e) {
				return kC30IExpDomainSweepCases[i];
			}
		}
		std::abort();  // fixture regressed -- a named point must be present
	};

	// e = -62: OUT of domain at both mantissas (D-SLM318: e >= -77 is the wrong,
	// permissive threshold this strip wrongly admits), but by TWO DIFFERENT
	// mechanisms -- found by execution, not assumed. At the high mantissa the derived
	// (q_ln2, q_b, q_c) fit int64 and IExpConstantsInDomain itself rejects on
	// representability. At the LOW mantissa, q_c itself overflows int64 during C30's
	// own derivation (the generator's derivation_ok guard) -- there is no valid int64
	// triple to call the predicate with at all, so "OUT of domain" here rests on the
	// derivation site's own upstream construction guard (S3.3's, not yet built), not
	// on IExpConstantsInDomain. Both are asserted for what they are.
	{
		const auto& hi = find_case(kMHigh, -62);
		CHECK(hi.derivation_ok);
		CHECK_MSG(!IExpConstantsInDomain(0, hi.q_ln2, hi.q_b, hi.q_c),
		          "m=2^31-1 e=-62 must be OUT of domain via IExpConstantsInDomain's own "
		          "representability check");

		const auto& lo = find_case(kMLow, -62);
		CHECK_MSG(!lo.derivation_ok,
		          "m=2^30 e=-62: expected the derivation itself to be unable to form a "
		          "valid int64 (q_ln2,q_b,q_c) triple at this point (q_c overflows int64 "
		          "during C30's own construction) -- if this now succeeds, the fixture's "
		          "int64-fit boundary moved and this cell's routed finding is stale");
		CHECK_MSG(!lo.expected_in_domain,
		          "m=2^30 e=-62 must still read OUT of domain overall, via the "
		          "construction-domain rejection rather than IExpConstantsInDomain");
	}
	// e = -61: the mantissa-conditional case. Low mantissa (2^30 < 1,268,234,713) is
	// OUT; high mantissa (2^31-1 >= 1,268,234,713) is IN. A scalar threshold (D-SLM77's
	// e >= -61) cannot express this split; only the shipped predicate does.
	{
		const auto& lo = find_case(kMLow, -61);
		const auto& hi = find_case(kMHigh, -61);
		CHECK(lo.derivation_ok && hi.derivation_ok);
		CHECK_MSG(!IExpConstantsInDomain(0, lo.q_ln2, lo.q_b, lo.q_c),
		          "m=2^30 e=-61 must be OUT of domain (below the mantissa threshold)");
		CHECK_MSG(IExpConstantsInDomain(0, hi.q_ln2, hi.q_b, hi.q_c),
		          "m=2^31-1 e=-61 must be IN domain (at/above the mantissa threshold "
		          "1,268,234,713) -- the case no scalar e-only threshold can express");
	}
	// e = -60: IN domain at both mantissas (the corrected floor's own boundary).
	for (int64_t m : {kMLow, kMHigh}) {
		const auto& c = find_case(m, -60);
		CHECK(c.derivation_ok);
		CHECK_MSG(IExpConstantsInDomain(0, c.q_ln2, c.q_b, c.q_c),
		          "m=%lld e=-60 must be IN domain (the current-truth floor)", m);
	}
}

// --- SuperSLM_S3a_WalkingSkeleton_Plan.md Sec5.4, Sec7.2 (C34's derived-operand
//     predicate), Sec11 S3.1: the CONTAINMENT relation between the load-time
//     descriptor (already shipped, S-HARDEN-1) and the runtime no-UB domain the
//     not-yet-built C34 predicate would encode. There is no production entry point
//     for "the runtime predicate" itself (verified: no function of that shape is
//     declared anywhere under include/superslm/ or src/) -- S3.2/S3.3's build. What
//     IS real today is the load-time half (SslmModel::Load, driven below through the
//     shipped MakeKvc1CompositionSection/BuildArtifact fixtures already exercising
//     e=7/-80 accept and e=8/-81 reject elsewhere in this file) and the two NAMED
//     constants the runtime predicate would check against
//     (kSiluLutTermLeftShiftOverflowExponent, kRoundingDivideByPotExponentMaxI64,
//     both already shipped, both public). This test computes the runtime no-UB
//     domain's own oracle from those two real constants -- the exact formula Sec5.4
//     specifies (shift = e + kSiluLutLog2K + kSiluLutQIdx on the upper branch,
//     -e - kSiluLutLog2K - kSiluLutQIdx on the lower) -- and asserts CONTAINMENT
//     against the real, executed load-time verdict at all four named boundary
//     points plus their immediate neighbours, exactly as Sec5.4's own executed probe
//     does. The oracle function below is TEST CODE ONLY: it stands in for the
//     not-yet-built production predicate and is not a claim that any production
//     entry point exists. ---
static bool C34RuntimeNoUbDomainOracle(int e) {
	using namespace superslm;
	const int shift_upper = e + kSiluLutLog2K + kSiluLutQIdx;
	const int shift_lower = -e - kSiluLutLog2K - kSiluLutQIdx;
	return shift_upper < kSiluLutTermLeftShiftOverflowExponent &&
	       shift_lower <= kRoundingDivideByPotExponentMaxI64;
}

static bool C34LoadTimeAcceptsE(int64_t e) {
	using namespace superslm_test;
	auto built = BuildArtifact({MakeValidConfigSection(), MakeSigmoidLutSection(),
	                             MakeKvc1CompositionSection(/*m=*/0, e)});
	SslmModelView view;
	std::string err;
	SslmModelStatus status = SslmModel::Load(built.bytes.data(), built.bytes.size(), view, &err);
	return status == SslmModelStatus::Ok;
}

static void TestC34RuntimeDomainOracleContainsTheShippedLoadTimeDescriptor() {
	// Containment, swept: every e the real load-time gate accepts, the runtime no-UB
	// oracle also accepts. Never asserted the other way (the runtime domain is
	// deliberately wider -- Sec5.4's whole point).
	for (int e = -85; e <= 15; ++e) {
		if (C34LoadTimeAcceptsE(e)) {
			CHECK_MSG(C34RuntimeNoUbDomainOracle(e),
			          "e=%d: load-time ACCEPTS but the runtime no-UB oracle rejects -- "
			          "containment violated (Sec5.4)",
			          e);
		}
	}

	// The four named boundary points, pinned by their EXECUTED disposition (Sec5.4):
	// e=7 both accept; e=8 the one point of difference (runtime accepts, load
	// rejects); e=9 both reject (the overflow point itself, shift==26); e=-80 both
	// accept (the shared exact lower endpoint), e=-81 both reject.
	CHECK_MSG(C34LoadTimeAcceptsE(7) && C34RuntimeNoUbDomainOracle(7),
	          "e=7: both must accept (the load-time ceiling, shift 24)");
	CHECK_MSG(!C34LoadTimeAcceptsE(8) && C34RuntimeNoUbDomainOracle(8),
	          "e=8: load-time must reject and the runtime oracle must accept -- the one "
	          "point of difference Sec5.4 executes");
	CHECK_MSG(!C34LoadTimeAcceptsE(9) && !C34RuntimeNoUbDomainOracle(9),
	          "e=9: both must reject (shift==26, the overflow point itself)");
	CHECK_MSG(C34LoadTimeAcceptsE(-80) && C34RuntimeNoUbDomainOracle(-80),
	          "e=-80: both must accept (the shared exact lower endpoint)");
	CHECK_MSG(!C34LoadTimeAcceptsE(-81) && !C34RuntimeNoUbDomainOracle(-81),
	          "e=-81: both must reject");

	// The load-time ceiling's own static_assert names the overflow point exactly:
	// shift(7) = 7+17 = 24 < 26 (the overflow exponent) -- confirmed live against the
	// real named constants rather than restated as a literal.
	CHECK(7 + superslm::kSiluLutLog2K + superslm::kSiluLutQIdx <
	      superslm::kSiluLutTermLeftShiftOverflowExponent);
	CHECK(9 + superslm::kSiluLutLog2K + superslm::kSiluLutQIdx ==
	      superslm::kSiluLutTermLeftShiftOverflowExponent);
}

// ---------------------------------------------------------------------------
// SuperSLM_S3a_WalkingSkeleton_Plan.md Sec11 S3.1 -- the funnel's two entry
// points (RequantChainChecked, NarrowRowChecked), shipped in
// src/forward/checked_chain_funnel.cpp against the header contract
// (include/superslm/checked_chain_funnel.h). Every cell below drives the real
// construction, not a stub. Fixtures:
// Claude/Curie/superslm-s3.1-checked-chain-funnel-test-design-2026-07-28.md
// Sec4.4/Sec4.5, tests/sslm_s3_1_wide_intmath_fixtures.h.
// ---------------------------------------------------------------------------

static void TestRequantChainCheckedT1254Witness() {
	using namespace superslm_test;
	using superslm::CarriedScale;
	using superslm::ChainResult;
	using superslm::SslmForwardStatus;
	// Canonical (m in [2^30, 2^31)), neutral site constant -- not itself under
	// test here; only the wide row and the resulting codes are pinned by
	// T-1254.
	const CarriedScale site_constant{/*m=*/INT64_C(1073741824), /*e=*/0};
	for (size_t i = 0; i < kFunnelWitnessRowsCount; ++i) {
		const FunnelWitnessRow& c = kFunnelWitnessRows[i];
		int8_t out_codes[4] = {INT8_C(-99), INT8_C(-99), INT8_C(-99), INT8_C(-99)};  // poison
		CarriedScale out_scale{/*m=*/INT64_C(-99), /*e=*/INT64_C(-99)};              // poison
		ChainResult result = superslm::RequantChainChecked(c.row, c.n, std::span<const CarriedScale>{},
		                                                     site_constant, out_codes, &out_scale);
		CHECK_MSG(result.status == SslmForwardStatus::Ok,
		          "%s: RequantChainChecked status == %s, want Ok (T-1254 required-green witness)",
		          c.label, superslm::SslmForwardStatusName(result.status));
		for (size_t j = 0; j < c.n && j < 4; ++j) {
			CHECK_MSG(out_codes[j] == c.expected_codes[j],
			          "%s: RequantChainChecked out_codes[%zu] == %d, want %d (T-1254 witness, "
			          "matches _requant_row_int64/intmath.requant_token_code)",
			          c.label, j, static_cast<int>(out_codes[j]), static_cast<int>(c.expected_codes[j]));
		}
		CHECK_MSG(out_scale.m != INT64_C(-99) || out_scale.e != INT64_C(-99),
		          "%s: RequantChainChecked must write *out_scale on Ok (still the poison value)", c.label);
	}
}

static void TestRequantChainCheckedRejectsOverC29Domain() {
	using namespace superslm_test;
	using superslm::CarriedScale;
	// Sec11 S3.1's own named red cell: "a wide row at D' = 2^31 + 1 returns
	// ChainInputOutOfDomain ... and computes nothing."
	const CarriedScale site_constant{INT64_C(1073741824), 0};
	int8_t out_codes[1] = {INT8_C(-99)};             // poison
	CarriedScale out_scale{INT64_C(-99), INT64_C(-99)};  // poison
	auto result = superslm::RequantChainChecked(kWideOverC29DomainRow, kWideOverC29DomainRowLen,
	                                              std::span<const CarriedScale>{}, site_constant, out_codes,
	                                              &out_scale);
	CHECK_MSG(result.status == superslm::SslmForwardStatus::ChainInputOutOfDomain,
	          "D' = 2^31+1: RequantChainChecked status == %s, want ChainInputOutOfDomain",
	          superslm::SslmForwardStatusName(result.status));
	CHECK_MSG(out_codes[0] == INT8_C(-99),
	          "D' = 2^31+1: out_codes must be untouched on rejection (\"computes nothing\")");
	CHECK_MSG(out_scale.m == INT64_C(-99) && out_scale.e == INT64_C(-99),
	          "D' = 2^31+1: *out_scale must be untouched on rejection (\"computes nothing\")");
}

static void TestNarrowRowCheckedT1254Witness() {
	using namespace superslm_test;
	// Positive-extreme row: max element is 2^31, one past INT32_MAX -- C35
	// must reject.
	{
		int32_t out_i32[4] = {INT32_C(-99), INT32_C(-99), INT32_C(-99), INT32_C(-99)};  // poison
		auto status =
		    superslm::NarrowRowChecked(kWideT1254WitnessPositiveRow, kWideT1254WitnessRowLen, out_i32);
		CHECK_MSG(status == superslm::SslmForwardStatus::LogitNarrowingOverflow,
		          "positive-extreme witness row: NarrowRowChecked status == %s, want "
		          "LogitNarrowingOverflow (T-1254)",
		          superslm::SslmForwardStatusName(status));
		CHECK_MSG(out_i32[0] == INT32_C(-99),
		          "positive-extreme witness row: out_i32 must be untouched on rejection");
	}
	// Negated-extreme row: min element is exactly -2^31 == INT32_MIN, max is
	// 2^30 -- both within [INT32_MIN, INT32_MAX] -- C35 must accept, and every
	// element narrows to its exact value, unchanged (NarrowAccumulatorToI32's
	// own soundness, once C35 has proven every element in range).
	{
		int32_t out_i32[4] = {0, 0, 0, 0};
		auto status =
		    superslm::NarrowRowChecked(kWideT1254WitnessNegatedRow, kWideT1254WitnessRowLen, out_i32);
		CHECK_MSG(status == superslm::SslmForwardStatus::Ok,
		          "negated-extreme witness row: NarrowRowChecked status == %s, want Ok (T-1254)",
		          superslm::SslmForwardStatusName(status));
		CHECK_MSG(out_i32[0] == INT32_C(-2147483648),
		          "negated-extreme witness row: out_i32[0] == %d, want INT32_MIN (T-1254)", out_i32[0]);
		for (size_t j = 0; j < kWideT1254WitnessRowLen; ++j) {
			CHECK_MSG(out_i32[j] == static_cast<int32_t>(kWideT1254WitnessNegatedRow[j]),
			          "negated-extreme witness row: out_i32[%zu] == %d, want %d (exact narrowing)", j,
			          out_i32[j], static_cast<int32_t>(kWideT1254WitnessNegatedRow[j]));
		}
	}
}

// The plan's own named negative control (Sec5.5, Sec11 S3.1): "C35 replaced
// by C29's magnitude check must fail this cell by accepting the positive
// row." Computed directly from the row's own values, never via
// MaxAbsReduceWide, so this control does not depend on the very primitive
// under test to make its point -- a negative control that called the
// primitive it substitutes for would no longer be independent of it.
// There is exactly one production NarrowRowChecked to call (Sec7.3's funnel
// discipline), so the "replaced" predicate is realized as a test-side
// computation standing in for it, mirroring TestC34RuntimeDomainOracle...'s
// own oracle-substitution pattern (Sec3.3).
static bool C29StyleMagnitudeCheckWouldAcceptRow(const int64_t* row, size_t n) {
	int64_t d = 0;
	for (size_t i = 0; i < n; ++i) {
		const int64_t v = row[i];
		const int64_t av = v < 0 ? -v : v;  // safe: no fixture row here carries INT64_MIN
		if (av > d) d = av;
	}
	const int64_t d_prime = d > 1 ? d : 1;
	return d_prime <= INT64_C(2147483648);  // C29's own D' <= 2^31 magnitude bound
}

static void TestNarrowRowCheckedC35VsC29NegativeControl() {
	using namespace superslm_test;
	const bool c29_would_accept =
	    C29StyleMagnitudeCheckWouldAcceptRow(kWideT1254WitnessPositiveRow, kWideT1254WitnessRowLen);
	CHECK_MSG(c29_would_accept,
	          "C29's own D' <= 2^31 magnitude check must ACCEPT the positive-extreme witness row "
	          "(D' == 2^31 exactly) -- if this is false, the negative control's own premise (a "
	          "magnitude bound cannot see this row's overflow) no longer holds and must be "
	          "re-derived before this cell means anything");

	int64_t max_element = kWideT1254WitnessPositiveRow[0];
	for (size_t i = 1; i < kWideT1254WitnessRowLen; ++i) {
		if (kWideT1254WitnessPositiveRow[i] > max_element) max_element = kWideT1254WitnessPositiveRow[i];
	}
	CHECK_MSG(max_element > superslm::kInt32Max,
	          "the witness row's max element (%lld) must exceed INT32_MAX for this negative control "
	          "to demonstrate anything -- this is C35's real rejection reason, proven live by "
	          "TestNarrowRowCheckedT1254Witness's own LogitNarrowingOverflow expectation on this "
	          "same row",
	          static_cast<long long>(max_element));
	// The discrimination itself: a magnitude-only substitute for C35 says
	// ACCEPT on this row (c29_would_accept, above); the real, asymmetric C35
	// check must say REJECT on the same row, proven live elsewhere in this
	// file. A predicate that agreed with the magnitude check here would prove
	// nothing beyond what C29 already proves -- this is the exact case
	// Sec5.5 names as inexpressible by any scalar magnitude bound.
	CHECK_MSG(c29_would_accept && max_element > superslm::kInt32Max,
	          "the negative control's two halves must both hold simultaneously: a magnitude bound "
	          "accepts this row while the row is genuinely out of C35's asymmetric range");
}

// ---------------------------------------------------------------------------
// Sec7.2's second limb, C30: the derivation-site predicate wrapper. Ruled
// 2026-07-28 (Claude/Vitruvius/SuperSLM_S3.1_C30DomainRule_Ruling-2026-07-28.md):
// IExpConstantsInDomain is the total, sole domain authority at every e; the
// closed-form "e >= -60" clause is a corollary valid only in the overflow
// tail and does not extend to the underflow tail (e roughly [-31,+8]), where
// IExpConstantsInDomain correctly rejects via its own q_ln2 < 1 guard.
// CheckIExpConstantsDomain (src/forward/checked_chain_funnel.cpp) implements
// the ruled contract: it calls IExpConstantsInDomain and encodes no threshold
// of its own. These cells assert exactly that, against the already-generated
// sweep fixture whose expected_in_domain field is the ruled oracle throughout
// (never the textual corollary).
// ---------------------------------------------------------------------------

static void TestCheckIExpConstantsDomainWrapsIExpConstantsInDomainAcrossTheSweep() {
	using namespace superslm_test;
	int checked = 0;
	for (size_t i = 0; i < kC30IExpDomainSweepCasesCount; ++i) {
		const C30IExpDomainSweepCase& c = kC30IExpDomainSweepCases[i];
		if (!c.derivation_ok) continue;  // C30's own construction-domain rejection -- no valid
		                                  // int64 triple exists to call the wrapper with.
		CHECK_MSG(c.shortcut_condition_holds,
		          "m=%lld e=%d: the q=0 shortcut's own precondition does not hold for this row",
		          c.m, c.e);
		const superslm::SslmForwardStatus expected = c.expected_in_domain
		                                                  ? superslm::SslmForwardStatus::Ok
		                                                  : superslm::SslmForwardStatus::IExpConstantsOutOfDomain;
		const superslm::SslmForwardStatus got =
		    superslm::CheckIExpConstantsDomain(0, c.q_ln2, c.q_b, c.q_c);
		CHECK_MSG(got == expected,
		          "m=%lld e=%d q_ln2=%lld q_b=%lld q_c=%lld: CheckIExpConstantsDomain(0, ...) == %s, "
		          "want %s (D-SLM348's ruling: IExpConstantsInDomain is the total domain authority, "
		          "never the textual e>=-60 corollary outside its overflow-tail scope)",
		          c.m, c.e, c.q_ln2, c.q_b, c.q_c, superslm::SslmForwardStatusName(got),
		          superslm::SslmForwardStatusName(expected));
		++checked;
	}
	CHECK_MSG(checked > 0, "no sweep row had a valid C30 derivation to check -- fixture regressed");
}

// The three named disagreement points (D-SLM318's table), through the wrapper
// rather than the bare primitive -- mirrors
// TestC30DomainDisagreementPointsAreExplicitlyPinned's own direct-call cell,
// realized here against the funnel's actual entry point now that it exists.
static void TestCheckIExpConstantsDomainDisagreementPointsAreExplicitlyPinned() {
	using namespace superslm_test;
	constexpr int64_t kMLow = INT64_C(1073741824);   // 2^30
	constexpr int64_t kMHigh = INT64_C(2147483647);  // 2^31 - 1
	auto find_case = [](int64_t m, int e) -> const C30IExpDomainSweepCase& {
		for (size_t i = 0; i < kC30IExpDomainSweepCasesCount; ++i) {
			if (kC30IExpDomainSweepCases[i].m == m && kC30IExpDomainSweepCases[i].e == e) {
				return kC30IExpDomainSweepCases[i];
			}
		}
		std::abort();  // fixture regressed -- a named point must be present
	};

	// e = -62: the high-mantissa case is representable and must reject via
	// the wrapper. The low-mantissa case has no valid int64 triple to call
	// the wrapper with at all (S3.3's own upstream construction guard, not
	// this wrapper's) and is pinned directly against the oracle in
	// TestC30DomainDisagreementPointsAreExplicitlyPinned rather than
	// re-asserted here.
	{
		const auto& hi = find_case(kMHigh, -62);
		CHECK(hi.derivation_ok);
		CHECK_MSG(superslm::CheckIExpConstantsDomain(0, hi.q_ln2, hi.q_b, hi.q_c) ==
		              superslm::SslmForwardStatus::IExpConstantsOutOfDomain,
		          "m=2^31-1 e=-62 must be IExpConstantsOutOfDomain via the wrapper");
	}
	// e = -61: the mantissa-conditional case -- no scalar e-only threshold
	// can express this split; only the shipped predicate, called by the
	// wrapper, does.
	{
		const auto& lo = find_case(kMLow, -61);
		const auto& hi = find_case(kMHigh, -61);
		CHECK(lo.derivation_ok && hi.derivation_ok);
		CHECK_MSG(superslm::CheckIExpConstantsDomain(0, lo.q_ln2, lo.q_b, lo.q_c) ==
		              superslm::SslmForwardStatus::IExpConstantsOutOfDomain,
		          "m=2^30 e=-61 must be IExpConstantsOutOfDomain via the wrapper (below the "
		          "mantissa threshold)");
		CHECK_MSG(superslm::CheckIExpConstantsDomain(0, hi.q_ln2, hi.q_b, hi.q_c) ==
		              superslm::SslmForwardStatus::Ok,
		          "m=2^31-1 e=-61 must be Ok via the wrapper (at/above the mantissa threshold "
		          "1,268,234,713)");
	}
	// e = -60: Ok at both mantissas (the corrected floor's own boundary).
	for (int64_t m : {kMLow, kMHigh}) {
		const auto& c = find_case(m, -60);
		CHECK(c.derivation_ok);
		CHECK_MSG(superslm::CheckIExpConstantsDomain(0, c.q_ln2, c.q_b, c.q_c) ==
		              superslm::SslmForwardStatus::Ok,
		          "m=%lld e=-60 must be Ok via the wrapper (the current-truth floor)", m);
	}
}

// ---------------------------------------------------------------------------
// Poirot review ac34677 (2026-07-28) of S3.1/S3.1a: five test-coverage
// findings, each pinned as a cell in its own right (Claude/Curie/
// superslm-s3.1-checked-chain-funnel-test-design-2026-07-28.md Sec10;
// superslm-s3.1a-trace-hook-test-design-2026-07-28.md Sec10). S1, S3, and S4
// below fail today, each for its own reason; the two S9 cells are already
// green (the reviewer executed both, P5/P6) and are committed here as suite
// members rather than left as review-only probes.
// ---------------------------------------------------------------------------

// S1: MaxAbsReduceWide's own abs computation (`x[i] < 0 ? -x[i] : x[i]`,
// src/intmath.cpp:272) is signed-integer overflow at x[i] == INT64_MIN --
// negating INT64_MIN has no int64_t representation. On this toolchain it
// wraps back to INT64_MIN, which fails the `a > d` comparison and silently
// excludes the row's largest-magnitude element from the reduction: executed
// (Poirot P1/P2), {INT64_MIN,5,-5,0} reports D' == 5 and RequantChainChecked
// returns Ok, bypassing the very guard (C29) whose purpose is to reject a
// row the chain cannot handle.
//
// What is pinned here, and why not an exact numeric D': the row's TRUE
// max-abs magnitude is |INT64_MIN| == 2^63 -- C20's own definition (D =
// max_i |x_i|) makes this unambiguous -- but 2^63 has no representation as a
// positive int64_t (int64_t's own range tops out at 2^63-1), so no exact
// equality claim on MaxAbsReduceWide's return is well-posed at this element,
// unlike the int32 sibling MaxAbsReduce, whose own widen-before-abs fix
// (src/intmath.cpp:197, "widen BEFORE abs so INT32_MIN yields 2^31 (not
// int32 UB)") lands INT32_MIN's magnitude at exactly 2^31 -- comfortably
// representable in the WIDER int64_t return type that primitive uses. This
// cell derives its claim from the sibling's own contract (magnitude is
// widened before the abs, and the guard the composition relies on is
// C29's own D' > 2^31 domain check, checked_chain_funnel.cpp:78) rather than
// from a specific fix construction: a row whose true magnitude is 2^63 is
// unambiguously out of C29's domain (2^63 > 2^31), so MaxAbsReduceWide's own
// report for this row must exceed 2^31, and RequantChainChecked must reject
// it -- whatever internal representation the primitive eventually uses to
// get there. Today it does neither.
static void TestMaxAbsReduceWideInt64MinElementReportsOutOfC29Domain() {
	using superslm::CarriedScale;
	using superslm::SslmForwardStatus;

	constexpr int64_t kRow[] = {INT64_MIN, INT64_C(5), INT64_C(-5), INT64_C(0)};
	constexpr size_t kRowLen = 4;

	const int64_t d_prime = superslm::MaxAbsReduceWide(kRow, kRowLen);
	CHECK_MSG(d_prime > (int64_t{1} << 31),
	          "MaxAbsReduceWide({INT64_MIN,5,-5,0}) == %lld, want a value > 2^31 (the "
	          "row's true max-abs magnitude is |INT64_MIN| == 2^63, definitionally out "
	          "of C29's <= 2^31 domain regardless of how the implementation represents "
	          "it)",
	          static_cast<long long>(d_prime));

	const CarriedScale site_constant{INT64_C(1073741824), 0};
	int8_t out_codes[kRowLen] = {INT8_C(-99), INT8_C(-99), INT8_C(-99), INT8_C(-99)};
	CarriedScale out_scale{INT64_C(-99), INT64_C(-99)};
	auto result = superslm::RequantChainChecked(kRow, kRowLen, std::span<const CarriedScale>{},
	                                              site_constant, out_codes, &out_scale);
	CHECK_MSG(result.status == SslmForwardStatus::ChainInputOutOfDomain,
	          "RequantChainChecked({INT64_MIN,5,-5,0}) status == %s, want "
	          "ChainInputOutOfDomain -- C29's guard exists precisely to reject a row "
	          "the chain cannot handle, and INT64_MIN's magnitude is the largest a "
	          "wide row can carry",
	          superslm::SslmForwardStatusName(result.status));
	CHECK_MSG(out_codes[0] == INT8_C(-99),
	          "RequantChainChecked({INT64_MIN,5,-5,0}): out_codes must be untouched on "
	          "rejection (\"computes nothing\")");
	CHECK_MSG(out_scale.m == INT64_C(-99) && out_scale.e == INT64_C(-99),
	          "RequantChainChecked({INT64_MIN,5,-5,0}): *out_scale must be untouched on "
	          "rejection (\"computes nothing\")");
}

// S4: step 6 of RequantChainChecked -- C26's carried-scale product,
// checked_chain_funnel.cpp:92-112 -- has no assertion of any kind anywhere
// in the suite. The only prior statement about *out_scale
// (TestRequantChainCheckedT1254Witness) asserts merely that it differs from
// its poison value; two mutants defeat that (Poirot M3, M5): replacing the
// whole computed value with a constant, and reversing the fold to
// right-associated. This cell pins the EXACT value, derived independently
// of src/forward/checked_chain_funnel.cpp -- from the vendored reference's
// own carried_scale_product (tests/reference/superslm_spike/intmath.py:
// 410-428), the same left-associated mechanism C26 pins and this design's
// own pinned mechanism (Sec7.2 step 6), executed directly in Python rather
// than read back from the C++ under test:
//
//   python -c "
//   import sys; sys.path.insert(0, 'tests/reference/superslm_spike')
//   sys.path.insert(0, 'tests/reference'); import intmath as im
//   incoming = [(1441784252, -5), (2032538395, 4)]
//   site_constant = (1935848999, -3)
//   d_prime_factor = (1073741824, 1)  # this row's own (Dn, -s), Dn=2^30, s=-1
//   print(im.carried_scale_product(incoming + [site_constant, d_prime_factor]))"
//   # -> (1230129356, 89)
//
// A multi-factor `incoming` span is required: a single-factor fold cannot
// distinguish left- from right-association at all (there is only one
// combination to order). The three canonical factors above were found by
// direct execution of the vendored reference across a random search over
// canonical (m,e) triples (not constructed to force a result) to diverge
// between left- and right-associated composition: right-associated
// computes {m=1230129357, e=89} for this identical input -- a
// one-ULP-in-mantissa difference this cell's exact-equality assertion
// catches and the prior poison-value-difference check does not.
static void TestRequantChainCheckedOutScaleLeftAssociatedFoldPinnedAgainstVendoredReference() {
	using namespace superslm_test;
	using superslm::CarriedScale;
	using superslm::SslmForwardStatus;

	const CarriedScale incoming[] = {
	    CarriedScale{/*m=*/INT64_C(1441784252), /*e=*/-5},
	    CarriedScale{/*m=*/INT64_C(2032538395), /*e=*/4},
	};
	const CarriedScale site_constant{/*m=*/INT64_C(1935848999), /*e=*/-3};

	// kWideT1254WitnessPositiveRow's own D' is exactly 2^31 (TestMaxAbsReduceWide),
	// giving NormalizeScale's (Dn, s) = (2^30, -1) -- verified independently by
	// Poirot's own P5 probe on this identical row ("D'=2^31, Dn=2^30, s=-1") -- and
	// therefore the D'-factor CarriedScale{2^30, 1} used in the derivation above.
	int8_t out_codes[4] = {0, 0, 0, 0};
	CarriedScale out_scale{};
	auto result = superslm::RequantChainChecked(
	    kWideT1254WitnessPositiveRow, kWideT1254WitnessRowLen,
	    std::span<const CarriedScale>(incoming, 2), site_constant, out_codes, &out_scale);
	CHECK_MSG(result.status == SslmForwardStatus::Ok, "RequantChainChecked status == %s, want Ok",
	          superslm::SslmForwardStatusName(result.status));
	CHECK_MSG(out_scale.m == INT64_C(1230129356) && out_scale.e == INT64_C(89),
	          "*out_scale == {%lld,%lld}, want {1230129356,89} (C26's left-associated "
	          "carried_scale_product, executed independently via the vendored reference "
	          "-- see the derivation command above this test)",
	          static_cast<long long>(out_scale.m), static_cast<long long>(out_scale.e));
}

// S9: the trace instrument's REDUCED funnel-level form (Sec11 S3.1a's own
// Cell 1 and Cell 3, narrowed to the one production call site the hook is
// wired to today -- RequantChainChecked itself, per the routing option
// named and not adopted in superslm-s3.1a-trace-hook-test-design-2026-07-28.
// md Sec6, since adopted). Both cells were already GREEN when the reviewer
// executed them (P5, P6); they are committed here as suite members. This is
// the REDUCED form, not Sec11 S3.1a's own full-forward Cell 1 (needs S3.6's
// digests) or Cell 2 (needs the full forward and the reference prompt pack,
// Sec14.3) -- both remain open, unchanged by this pass.
namespace {

struct ChainTraceSinkRecord {
	std::string site;
	size_t token_index = 0;
	std::vector<int64_t> x_int;
	int64_t d_prime = 0;
	int64_t dn = 0;
	int32_t s = 0;
	int64_t r = 0;
	std::vector<int8_t> codes;
	int64_t m_out = 0;
	int64_t e_out = 0;
};

void ChainTraceSinkHookFn(const superslm::SslmChainTraceRecord* chain,
                           const superslm::SslmKvLandingTraceRecord* kv, void* user) {
	// trace_hook.h's own contract: exactly one of the two record pointers is
	// non-null per call, never both, never neither.
	CHECK_MSG((chain != nullptr) != (kv != nullptr),
	          "a trace hook call must carry exactly one non-null record pointer, never "
	          "both and never neither");
	if (chain == nullptr) return;
	auto* sink = static_cast<std::vector<ChainTraceSinkRecord>*>(user);
	ChainTraceSinkRecord rec;
	rec.site.assign(chain->site.begin(), chain->site.end());
	rec.token_index = chain->token_index;
	rec.x_int.assign(chain->x_int.begin(), chain->x_int.end());
	rec.d_prime = chain->d_prime;
	rec.dn = chain->dn;
	rec.s = chain->s;
	rec.r = chain->r;
	rec.codes.assign(chain->codes.begin(), chain->codes.end());
	rec.m_out = chain->m_out;
	rec.e_out = chain->e_out;
	sink->push_back(std::move(rec));
}

}  // namespace

static void TestRequantChainCheckedHookInstalledProducesIdenticalOutputs() {
	using namespace superslm_test;
	using superslm::CarriedScale;

	const CarriedScale site_constant{/*m=*/INT64_C(1073741824), /*e=*/0};

	int8_t codes_no_hook[4] = {0, 0, 0, 0};
	CarriedScale scale_no_hook{};
	auto result_no_hook = superslm::RequantChainChecked(
	    kWideT1254WitnessPositiveRow, kWideT1254WitnessRowLen, std::span<const CarriedScale>{},
	    site_constant, codes_no_hook, &scale_no_hook, "s3.1a_reduced_probe", /*token_index=*/7);
	CHECK_MSG(!superslm::SslmTraceHookInstalled(),
	          "no hook installed yet -- SslmTraceHookInstalled() must read false");

	std::vector<ChainTraceSinkRecord> sink;
	superslm::SslmSetTraceHook(&ChainTraceSinkHookFn, &sink);
	CHECK_MSG(superslm::SslmTraceHookInstalled(), "SslmSetTraceHook must install the hook");

	int8_t codes_with_hook[4] = {0, 0, 0, 0};
	CarriedScale scale_with_hook{};
	auto result_with_hook = superslm::RequantChainChecked(
	    kWideT1254WitnessPositiveRow, kWideT1254WitnessRowLen, std::span<const CarriedScale>{},
	    site_constant, codes_with_hook, &scale_with_hook, "s3.1a_reduced_probe",
	    /*token_index=*/7);

	superslm::SslmSetTraceHook(nullptr, nullptr);
	CHECK_MSG(!superslm::SslmTraceHookInstalled(), "SslmSetTraceHook(nullptr,...) must uninstall");

	CHECK_MSG(result_with_hook.status == result_no_hook.status,
	          "installing the hook must not change RequantChainChecked's status: %s (hook) "
	          "vs %s (no hook)",
	          superslm::SslmForwardStatusName(result_with_hook.status),
	          superslm::SslmForwardStatusName(result_no_hook.status));
	for (int i = 0; i < 4; ++i) {
		CHECK_MSG(codes_with_hook[i] == codes_no_hook[i],
		          "installing the hook must not change out_codes[%d]: %d (hook) vs %d (no "
		          "hook)",
		          i, static_cast<int>(codes_with_hook[i]), static_cast<int>(codes_no_hook[i]));
	}
	CHECK_MSG(scale_with_hook.m == scale_no_hook.m && scale_with_hook.e == scale_no_hook.e,
	          "installing the hook must not change *out_scale: {%lld,%lld} (hook) vs "
	          "{%lld,%lld} (no hook)",
	          static_cast<long long>(scale_with_hook.m), static_cast<long long>(scale_with_hook.e),
	          static_cast<long long>(scale_no_hook.m), static_cast<long long>(scale_no_hook.e));

	// Exactly one record emitted, and its fields match the call's own already-
	// computed internals -- the transparency half of Sec10.3's axis (Poirot P5:
	// "the emitted record's fields matched the funnel's own internals").
	CHECK_MSG(sink.size() == 1, "exactly one chain trace record must be emitted per call; got %zu",
	          sink.size());
	if (sink.size() == 1) {
		const ChainTraceSinkRecord& rec = sink[0];
		CHECK_MSG(rec.site == "s3.1a_reduced_probe",
		          "emitted record's site == \"%s\", want the caller's own site name",
		          rec.site.c_str());
		CHECK_MSG(rec.token_index == 7, "emitted record's token_index == %zu, want 7",
		          rec.token_index);
		CHECK_MSG(rec.d_prime == kWideT1254WitnessDPrime,
		          "emitted record's d_prime == %lld, want %lld (this row's own D')",
		          static_cast<long long>(rec.d_prime), static_cast<long long>(kWideT1254WitnessDPrime));
		CHECK_MSG(rec.m_out == scale_with_hook.m && rec.e_out == scale_with_hook.e,
		          "emitted record's (m_out,e_out) == (%lld,%lld) must equal the call's own "
		          "*out_scale (%lld,%lld)",
		          static_cast<long long>(rec.m_out), static_cast<long long>(rec.e_out),
		          static_cast<long long>(scale_with_hook.m), static_cast<long long>(scale_with_hook.e));
		CHECK_MSG(rec.codes.size() == 4 &&
		              std::equal(rec.codes.begin(), rec.codes.end(), codes_with_hook),
		          "emitted record's codes must equal the call's own out_codes");
		CHECK_MSG(rec.x_int.size() == kWideT1254WitnessRowLen &&
		              std::equal(rec.x_int.begin(), rec.x_int.end(), kWideT1254WitnessPositiveRow),
		          "emitted record's x_int must equal the row passed in");
	}
}

static void TestRequantChainCheckedRejectedCallEmitsNoTraceRecordsEvenWithHookInstalled() {
	using namespace superslm_test;
	using superslm::CarriedScale;

	std::vector<ChainTraceSinkRecord> sink;
	superslm::SslmSetTraceHook(&ChainTraceSinkHookFn, &sink);

	const CarriedScale site_constant{INT64_C(1073741824), 0};
	int8_t out_codes[1] = {INT8_C(-99)};
	CarriedScale out_scale{INT64_C(-99), INT64_C(-99)};
	auto result = superslm::RequantChainChecked(kWideOverC29DomainRow, kWideOverC29DomainRowLen,
	                                              std::span<const CarriedScale>{}, site_constant,
	                                              out_codes, &out_scale, "s3.1a_reduced_probe_reject",
	                                              /*token_index=*/3);

	superslm::SslmSetTraceHook(nullptr, nullptr);

	CHECK_MSG(result.status == superslm::SslmForwardStatus::ChainInputOutOfDomain,
	          "a D'=2^31+1 row must still be rejected with a hook installed: got %s",
	          superslm::SslmForwardStatusName(result.status));
	CHECK_MSG(sink.empty(),
	          "a rejected call must emit zero trace records even with a hook installed; got %zu",
	          sink.size());
}

// D-SLM353 / Claude/Vitruvius/SuperSLM_S3.1a_TraceHookGlobal_Ruling-2026-07-28.md
// Sec6: the cross-model isolation cell the ruling names as what is required
// before Sec13 dimension 3's not-applicable reason can be re-asserted --
// "each model handle owns its own trace-hook state; no state is shared across
// model handles."
//
// EXPECTED RED against today's implementation. src/trace_hook.cpp:13-14 holds
// exactly one process-wide (fn, user) pair, so installing a hook for one
// model handle silently redirects every OTHER handle's already-installed
// hook to the newest installation. Today's SslmSetTraceHook/
// SslmTraceHookInstalled/SslmEmitChainTrace/SslmEmitKvLandingTrace take no
// model parameter at all (Poirot ac34677 review, finding S8) -- that omission
// is exactly the defect this cell exists to catch: "handle A" and "handle B"
// below are represented by two independent (hook fn, sink) pairs, standing in
// for the per-model trace state the ruling's corrected storage requires
// (Sec3: "hook state is owned by the model handle... never through a
// process-wide static"). Once that storage lands, a hook installed through
// handle A's own accessor must not be reachable, or disturbable, from any
// call made through handle B, and vice versa; this cell must then be updated
// to thread an actual model handle through each half instead of standing in
// with two sinks.
static void TestTraceHookCrossModelHandleIsolation() {
	using namespace superslm_test;
	using superslm::CarriedScale;

	const CarriedScale site_constant{/*m=*/INT64_C(1073741824), /*e=*/0};

	std::vector<ChainTraceSinkRecord> sink_a;
	std::vector<ChainTraceSinkRecord> sink_b;

	// Install "on handle A" and drive one emitting call attributed to it.
	superslm::SslmSetTraceHook(&ChainTraceSinkHookFn, &sink_a);
	{
		int8_t out_codes[4] = {0, 0, 0, 0};
		CarriedScale out_scale{};
		superslm::RequantChainChecked(kWideT1254WitnessPositiveRow, kWideT1254WitnessRowLen,
		                                std::span<const CarriedScale>{}, site_constant, out_codes,
		                                &out_scale, "isolation_probe_handle_a", /*token_index=*/1);
	}
	CHECK_MSG(sink_a.size() == 1,
	          "handle A's own call must land exactly one record in handle A's own sink; got %zu",
	          sink_a.size());
	CHECK_MSG(sink_b.empty(),
	          "handle A's own call must not emit into handle B's sink before handle B's hook is "
	          "ever installed; got %zu",
	          sink_b.size());

	// Install "on handle B", independently of handle A -- under a model-scoped
	// design this must not touch handle A's own hook state at all.
	superslm::SslmSetTraceHook(&ChainTraceSinkHookFn, &sink_b);
	{
		int8_t out_codes[4] = {0, 0, 0, 0};
		CarriedScale out_scale{};
		superslm::RequantChainChecked(kWideT1254WitnessPositiveRow, kWideT1254WitnessRowLen,
		                                std::span<const CarriedScale>{}, site_constant, out_codes,
		                                &out_scale, "isolation_probe_handle_b", /*token_index=*/2);
	}
	CHECK_MSG(sink_b.size() == 1,
	          "handle B's own call must land exactly one record in handle B's own sink; got %zu",
	          sink_b.size());
	CHECK_MSG(sink_a.size() == 1,
	          "installing a hook on handle B must not emit into handle A's sink, and must not "
	          "retroactively change what handle A already recorded; got %zu",
	          sink_a.size());

	// The cell's crux: drive a SECOND call attributed to handle A, without
	// touching either hook again. Handle A's own hook must still be the one
	// that fires -- installing on B must not have redirected A's emissions.
	{
		int8_t out_codes[4] = {0, 0, 0, 0};
		CarriedScale out_scale{};
		superslm::RequantChainChecked(kWideT1254WitnessPositiveRow, kWideT1254WitnessRowLen,
		                                std::span<const CarriedScale>{}, site_constant, out_codes,
		                                &out_scale, "isolation_probe_handle_a_second", /*token_index=*/3);
	}
	superslm::SslmSetTraceHook(nullptr, nullptr);

	CHECK_MSG(sink_a.size() == 2,
	          "cross-model isolation (D-SLM353): handle A's SECOND call must land in handle A's "
	          "own sink (size 2), not handle B's -- got sink_a.size()=%zu. A count that stayed "
	          "at 1 means installing handle B's hook silently redirected handle A's own, "
	          "already-installed hook -- exactly the process-global leak Sec13 dimension 3's "
	          "corrected reason requires be false.",
	          sink_a.size());
	CHECK_MSG(sink_b.size() == 1,
	          "cross-model isolation (D-SLM353): handle B's sink must not grow from handle A's "
	          "second call -- got sink_b.size()=%zu, want 1. A count of 2 means handle A's call "
	          "was silently emitted through handle B's hook instead of its own.",
	          sink_b.size());
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
	// Defense-in-depth against unbounded recursive self-spawning: a crash-probe
	// child is marked by kCrashProbeChildEnvVar (set by the parent, inherited
	// automatically). If this process IS such a child but its argv did not match
	// the "--crash-probe=<name>" dispatch above -- which should never happen
	// given RunsCrashProbeAndCrashes always constructs that exact argument, but
	// would happen if a future change to either side broke the match -- falling
	// through to the full suite below would run
	// TestGemmInt8AccumulateRowAssertsOnZeroInChannelsContractViolation again,
	// which spawns another mismatched child, recursively, without bound. Refuse
	// instead of recursing (pre-existing risk at commit 464c522; Claude/Poirot/
	// 2fdaf49-s2.5-golden-hash-review-2026-07-20.md finding 8).
	if (EnvVarIsSet(kCrashProbeChildEnvVar)) {
		std::fprintf(stderr,
		             "superslm_tests: this process was spawned as a crash-probe child "
		             "(%s is set) but argv did not match \"--crash-probe=<name>\" -- "
		             "refusing to run the full suite to avoid recursive self-spawning. "
		             "argv[1] was: %s\n",
		             kCrashProbeChildEnvVar, argc > 1 ? argv[1] : "(none)");
		return 3;
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

	// --- S-HARDEN-1 (F14): null-buffer / null-path rejection (red-first). ---
	TestRejectsNullDataNonzeroSize();
	TestRejectsNullDataSmallNonzeroSize();
	TestRejectsNullPath();

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
	TestTokenizerByteToIdBijectsOntoEveryRawByteValue();

	// --- S-HARDEN-2 (F7, red-first): Decode's documented malformed-UTF-8 policy
	//     through the one strict decoder shared by encode and decode. ---
	TestDecodeSubstitutesReplacementCharForOverlongTwoByteSequence();
	TestDecodeSubstitutesReplacementCharForOverlongThreeByteSequence();
	TestDecodeSubstitutesReplacementCharForSurrogateCodepoint();
	TestDecodeSubstitutesReplacementCharForCodepointPastU10FFFF();
	TestDecodeSubstitutesReplacementCharForF4WithContinuationPastMax();
	TestDecodeSubstitutesReplacementCharForOverlongFourByteSequence();
	TestDecodeSubstitutesReplacementCharForTruncatedFourByteSequence();
	TestDecodeSubstitutesReplacementCharForLoneContinuationByte();
	TestDecodeSubstitutesReplacementCharForTruncatedSequenceAtEnd();
	TestSingleVocabTokenizerSurvivesHeapChurnBetweenOpenAndDecode();
	TestDecodeReconstructsSequenceSplitAcrossTokenBoundary();
	TestDecodeAndEncodeShareOneStrictDecoderOnWellFormedMultibyteText();

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

	// --- S-HARDEN-2 (F6/F18, red-first): TOK1's version/reserved fields and its
	//     vocabulary-bound rejection (byte_to_id, merge operands/results, special
	//     ids, and vocab_count's own domain). ---
	TestTok1RejectsUnsupportedVersion();
	TestTok1RejectsNonzeroReserved();
	TestTok1RejectsVocabCountZero();
	TestTok1RejectsVocabCountExceedsInt32Max();
	TestTok1RejectsByteToIdEntryAtOrAboveVocabCount();
	TestTok1RejectsMergeOperandAAtOrAboveVocabCount();
	TestTok1RejectsMergeOperandBAtOrAboveVocabCount();
	TestTok1RejectsMergeResultAtOrAboveVocabCount();
	TestTok1RejectsSpecialIdAtOrAboveVocabCount();

	TestUni1RejectsBadMagic();
	TestUni1RejectsTruncatedHeader();
	TestUni1RejectsUnsupportedVersion();
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

	// --- Curie's S2.0a WGT1/BIA1/ROP1 tensor-manifest hostile-input suite. ---
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

	// --- Curie's S2.0b KVC1 keyed-constant sub-parse hostile-input suite. ---
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

	// --- Curie's S2.0b CFG1 config sub-parse hostile-input suite. ---
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

	// --- Curie's S2.1 intmath red suite. ---
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

	// --- Curie's S3.1 wide-row (int64 input width) red suite (F-S3-7). ---
	TestMaxAbsReduceWide();
	TestRowBoundsWide();
	TestRequantTokenCodeWide();

	// --- Curie's S3.1 checked-chain-funnel red suite (T-200, T-1254). ---
	TestRequantChainCheckedT1254Witness();
	TestRequantChainCheckedRejectsOverC29Domain();
	TestNarrowRowCheckedT1254Witness();
	TestNarrowRowCheckedC35VsC29NegativeControl();
	TestCheckIExpConstantsDomainWrapsIExpConstantsInDomainAcrossTheSweep();
	TestCheckIExpConstantsDomainDisagreementPointsAreExplicitlyPinned();

	// --- Poirot review ac34677 (2026-07-28) test-coverage findings: S1, S3,
	//     S4 red; the two S9 cells already green at review time. ---
	TestMaxAbsReduceWideInt64MinElementReportsOutOfC29Domain();
	TestRowBoundsWideZeroLenNullPtrDoesNotCrash();
	TestRequantChainCheckedOutScaleLeftAssociatedFoldPinnedAgainstVendoredReference();
	TestRequantChainCheckedHookInstalledProducesIdenticalOutputs();
	TestRequantChainCheckedRejectedCallEmitsNoTraceRecordsEvenWithHookInstalled();

	// --- D-SLM353 / SuperSLM_S3.1a_TraceHookGlobal_Ruling-2026-07-28.md Sec6:
	//     the cross-model isolation cell. EXPECTED RED against the current
	//     process-global trace hook storage. ---
	TestTraceHookCrossModelHandleIsolation();

	// --- Curie's S2.2 nonlinear scalar primitives red suite. ---
	TestISqrt();
	TestISqrtTrace();
	TestShiftByMax();
	// PORTED (S-HARDEN-0 final API): IExpFromConstants is removed; evaluation is
	// now construct-then-evaluate. Expected values unchanged from the original 36
	// kIExpCases goldens.
	TestIExpConstructAndEvaluateMatchGoldenCasesAcrossKIExpCases();
	TestIExpConstructAndEvaluateClipClampsIdenticallyAcrossFamily();

	// --- Curie's S2.6-amendment red suite for IExpConstantsInDomain (D-SLM78/79/81;
	//     unchanged by the S-HARDEN-0 final API -- IExpConstantsInDomain's own
	//     signature and behaviour are untouched). ---
	TestIExpConstantsInDomainAcrossCorpus();
	TestIExpConstantsInDomainRejectsStrikeExactInput();
	TestIExpConstantsInDomainShortcutConditionMatchesHeaderClaim();
	// REWORKED (S-HARDEN-0 final API): no debug-build assert survives to pin (see
	// the function's own comment); the wrapped-value golden is preserved.
	TestIExpConstructAndEvaluateProducesKnownWrappedValueForOutOfDomainConstants();

	// --- Curie's S-HARDEN-0 population suite, final API (F9, F21, Poirot's
	// a1d7986 code-review Finding 1; SuperSLM_Plan.md's S-HARDEN preamble;
	// Claude/Curie/superslm-s-harden-0-test-design-2026-07-21.md). Layer A's
	// separate crash-probe population is RETIRED -- the API collapse removed its
	// subject; see the comment at its former call site (above, in this file) for
	// the full account and where every witness value now lives (kIExpConstructCases
	// below). IExpEvaluate's totality and IExpConstruction's structural safety are
	// pinned first (default-construction safety, and the two static_asserts
	// immediately preceding it), then the full construct/evaluate population. ---
	TestIExpConstructionDefaultIsSafeToEvaluate();
	TestIExpConstructMatchesIndependentOracleAcrossCases();
	TestIExpConstructOutContractPerOutcome();
	TestIExpConstructAcceptsNullOutForPredicateOnlyUse();
	TestIExpConstantsInDomainEquivalentToIExpConstructEqualsKOk();
	TestIExpConstructMatchesAccessorCasesZAndBase();

	// --- Curie's S2.3 RopeApplyPair red suite. ---
	TestRopeApplyPair();
	TestRopeApplyPairIdentityIsExact();
	TestRopeApplyPairQuarterTurnIsExact();
	TestRopeApplyPairWideInputExceedsInt32Range();
	TestRopeApplyPairTieRoundsAwayFromZero();

	// --- Curie's S2.4 SiLU sigmoid-LUT red suite. ---
	TestMinimalSil1ParsesAndReadsBackAllNodes();
	TestSil1WarmObjectRepeatedReadsShowNoDrift();
	TestSil1RoundTripReencodeMatchesOriginalBytes();
	TestSil1RejectsSizeTooShort();
	TestSil1RejectsSizeTooLong();
	TestSil1RejectsBadMagic();
	TestSil1RejectsUnsupportedVersion();
	TestSil1RejectsBadEntryCount();
	TestSil1RejectsBadReserved();

	// --- S-HARDEN-1 (F20/F22): pinned canonical content (red-first). ---
	TestSil1RejectsSingleNodeContentMismatch();
	TestSil1RejectsHostileExtremeAdjacentNodes();
	TestArtifactRejectsHostileSigmoidLutContentThroughRealPath();
	TestArtifactRejectsConfigOnlyV2MissingSigmoidLut();

	// --- S-HARDEN-1's schema-value gate (F22/F23/F24, D-SLM141/D-SLM142) — the
	//     parser-vs-consumer question is RULED: value validation lives at the
	//     new SslmModel::Load entry point, not at the two structural parsers. ---
	TestKvc1RejectsHostileCompositionConstantsScale();
	TestWsc1RejectsHostileShiftOutOfDocumentedBound();
	TestRop1RejectsHostileCosSinPair();

	// --- Mendeleev's 2026-07-22 coverage audit of the gate above: the
	//     boundary/guard matrix (§3.1), Load's structural fail-closed reset
	//     (§3.2), Load's own success path (§3.3), and the dim-1 warm-object
	//     cell (§4, T-164). ---
	TestKvc1RejectsCompositionScaleMUnderNoUbFloor();
	TestKvc1RejectsCompositionScaleEUnderNoUbFloor();
	TestKvc1RejectsCompositionScaleEOverNoUbFloor();
	TestWsc1RejectsIdentityUnderDocumentedBound();
	TestWsc1RejectsIdentityOverDocumentedBound();
	TestWsc1RejectsShiftUnderDocumentedBound();
	TestRop1RejectsElementOverDocumentedBound();
	TestKvc1AcceptsCompositionScaleMAtLowerNoUbFloor();
	TestKvc1AcceptsCompositionScaleEAtLowerNoUbFloor();
	TestKvc1AcceptsCompositionScaleEAtUpperNoUbFloor();
	TestWsc1AcceptsIdentityAtZero();
	TestWsc1AcceptsIdentityAtOne();
	TestWsc1AcceptsShiftAtLowerBound();
	TestWsc1AcceptsShiftAtUpperBound();
	TestRop1AcceptsElementAtPositiveBound();
	TestRop1AcceptsElementAtNegativeBound();
	TestLoadRejectsStructurallyInvalidArtifactAsArtifactRejected();
	TestLoadFailsClosedOnStructuralSubParseFailureBeforeValueGate();
	TestLoadSucceedsAndExposesFullyPopulatedViewOnValidArtifact();
	TestLoadComposedViewRepeatedReadsShowNoDrift();
	TestLoadComposedViewReopenIsIdempotent();
	TestLoadOntoPopulatedViewResetsEntries();

	// --- T-403 regression PIN: SslmModelView lifetime-contract fix (the
	//     design's §7 behavioral cell plus Charpy's 2026-07-22 temper §6
	//     amendment's moved-from-inertness cell). ---
	TestLoadReturnedViewSurvivesInterveningHeapAllocationBeforeRead();
	TestLoadMovedFromViewIsInertAndCarriesNoDanglingSigmoidLut();

	// --- S-HARDEN-2's tokenizer join (F18, §17.3 cell 3): TOK1.vocab_count x
	//     CFG1.vocab_size enforced at SslmModel::Load, following the S-HARDEN-1
	//     boundary-gate pattern. ---
	TestLoadRejectsTokenizerVocabCountVsConfigVocabSizeMismatch();
	TestLoadAcceptsMatchingTokenizerVocabCountAndConfigVocabSize();
	TestLoadRejectsArtifactWithTokenizerSectionButNoUnicodeTables();
	TestLoadRejectsArtifactWithUnicodeTablesSectionButNoTokenizer();
	TestLoadAcceptsArtifactWithNeitherTokenizerSection();

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

	// --- Curie's S2.5 matmul red suite. ---
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
	TestS2Point5SixCaseAcceptanceGateMeasurement();
	TestDotRowScalarRefMatchesShippingSse2PathAndOracle();
	TestMatmulGoldenHashCrossPlatform();

	// --- S-HARDEN-3 (F13, §13 item 7, §17.3 cell 4): the independent converter
	//     verifier's core (red-first; superslm/proof_manifest.h is new this slot). ---
	TestConfigGeometryRejectsZeroAttentionHeads();
	TestConfigGeometryRejectsZeroKeyValueHeadsBeforeModulusFaults();
	TestConfigGeometryRejectsKvHeadsExceedsHeads();
	TestConfigGeometryRejectsHeadsNotDivisibleByKv();
	TestConfigGeometryRejectsHiddenSizeMismatch();
	TestConfigGeometryAcceptsGqaShape();
	TestConfigGeometryAcceptsMhaShape();
	TestComputeTensorEvidenceReportsExtremaAndSaturationBoundary();
	TestComputeWeightScaleEvidenceReportsShiftRangeAndIdentityCount();
	TestHashSectionHexMatchesIndependentSha256();
	TestBuildProofManifestJsonReportsGeometryOkOnCoherentArtifact();
	TestBuildProofManifestJsonReportsGeometryMismatchOnIncoherentArtifact();

	// --- S-HARDEN-7 (F5, §3, T-411): the "throws only std::bad_alloc" contract,
	//     all eighteen sites of the corrected, four-condition membership rule's
	//     derived population (design §3.1's table). The rename-and-wrap has
	//     landed; each site's *Impl body consults the test-only injection seam
	//     (tests/support/bad_alloc_injection.h). ---
	TestBadAllocContractOpenFromMemory();
	TestBadAllocContractOpenFromMemoryPassthroughClauseIsSpecific();
	TestBadAllocContractOpenFromFile();
	TestBadAllocContractFingerprintHex();
	TestBadAllocContractTensorManifestParseDirect();
	TestBadAllocContractTensorManifestParseViaLoad();
	TestBadAllocContractKeyedConstantsParse();
	TestBadAllocContractParseConfig();
	TestBadAllocContractParseSigmoidLut();
	TestBadAllocContractLoad();
	TestBadAllocContractTokenizerOpenDirect();
	TestBadAllocContractTokenizerOpenViaLoad();
	TestBadAllocContractTokenizerEncode();
	TestBadAllocContractTokenizerDecode();
	TestBadAllocContractComputeTensorEvidence();
	TestBadAllocContractComputeWeightScaleEvidence();
	TestBadAllocContractHashSectionHex();
	TestBadAllocContractBuildProofManifestJson();
	TestBadAllocContractSha256Update();
	TestBadAllocContractSha256HashFreeFunction();
	TestBadAllocContractToHex();

	// --- S-HARDEN-8 (F12, §4.2/§4.3, T-412): the generic section-descriptor
	//     `reserved` field, untested until this cell. ---
	TestRejectsNonZeroSectionDescriptorReservedField();

	// --- S3.1 (T-200, board T-132, §7.2, D-SLM318): the C30 derived-operand
	//     predicate's oracle-pinning cell. The full "design's C30 site" differential
	//     cell against a production wrapper is blocked (no such wrapper exists yet,
	//     S3.3's build) -- see Claude/Curie/superslm-s3.1-checked-chain-funnel-test-
	//     design-2026-07-28.md Sec3.2 for the routed finding. ---
	TestIExpConstantsInDomainAgreesWithC30DerivedConstantsAcrossTheSweep();
	TestC30DomainDisagreementPointsAreExplicitlyPinned();

	// --- S3.1 (§5.4, §7.2, D-SLM318): the C34 derived-operand predicate's
	//     containment cell, realized against the real load-time gate and the two
	//     real named constants -- see the test-design record for what remains
	//     blocked on the not-yet-built runtime predicate itself. ---
	TestC34RuntimeDomainOracleContainsTheShippedLoadTimeDescriptor();

	std::printf("superslm tests: %d checks, %d failures\n", GChecks, GFailures);
	return GFailures == 0 ? 0 : 1;
}

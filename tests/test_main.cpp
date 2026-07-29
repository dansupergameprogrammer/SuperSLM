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
#include "superslm/forward_sites.h"
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
#include "sslm_s3_2_fixtures.h"
#include "sslm_s3_3_fixtures.h"
#include "sslm_s3_3_red_regression_fixtures.h"
#include "sslm_s3_3_rope_application_site_fixtures.h"
#include "sslm_c32_softmax_row_width_gate_fixtures.h"
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
	// Every element carries the hostile value (not just one): SslmModel::Load's ROP1<->CFG1 shape
	// join (S3.3 §13.1 cell 4, D-SLM420-D-SLM423) now rejects a "cos"/"sin" tensor whose elem_count
	// disagrees with the default Config's context_cap * (head_dim/2) before the magnitude check this
	// cell targets ever runs, so the tensor must be sized to kCfg1DefaultRopeElemCount.
	std::vector<ManifestTensorSpec> tensors = {{"cos", {kCfg1DefaultRopeElemCount}}, {"sin", {kCfg1DefaultRopeElemCount}}};
	auto manifest = BuildManifest(superslm::kRopeMagic, /*element_size=*/8, tensors);
	for (uint32_t i = 0; i < kCfg1DefaultRopeElemCount; ++i) {
		PutU64(manifest.bytes, static_cast<size_t>(manifest.tensor_data_off[0]) + static_cast<size_t>(i) * 8,
		       static_cast<uint64_t>(int64_t{INT32_MIN}));
		PutU64(manifest.bytes, static_cast<size_t>(manifest.tensor_data_off[1]) + static_cast<size_t>(i) * 8,
		       static_cast<uint64_t>(int64_t{INT32_MIN}));
	}

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

// Builds a RopeTables (ROP1) section whose "cos"/"sin" tensors each carry
// kCfg1DefaultRopeElemCount elements (every element set to the one given
// value) — the same construction as TestRop1RejectsHostileCosSinPair above,
// factored out for the boundary matrix below. Sized to satisfy R4 against
// MakeValidConfigSection()'s default context_cap/head_dim (S3.3 §13.1 cell 4,
// D-SLM420-D-SLM423): a magnitude/domain cell (this helper's only concern)
// needs every element in a defined, uniform state, not a single element —
// SslmModel::Load's ROP1<->CFG1 shape join now rejects a "cos"/"sin" tensor
// whose elem_count disagrees with context_cap * (head_dim/2) before any
// magnitude check runs, so a single-element tensor paired with the default
// Config can no longer reach the magnitude check this helper exists to drive.
static FixtureSection MakeRop1Section(int64_t cos_v, int64_t sin_v) {
	using namespace superslm_test;
	std::vector<ManifestTensorSpec> tensors = {{"cos", {kCfg1DefaultRopeElemCount}}, {"sin", {kCfg1DefaultRopeElemCount}}};
	auto manifest = BuildManifest(superslm::kRopeMagic, /*element_size=*/8, tensors);
	for (uint32_t i = 0; i < kCfg1DefaultRopeElemCount; ++i) {
		PutU64(manifest.bytes, static_cast<size_t>(manifest.tensor_data_off[0]) + static_cast<size_t>(i) * 8,
		       static_cast<uint64_t>(cos_v));
		PutU64(manifest.bytes, static_cast<size_t>(manifest.tensor_data_off[1]) + static_cast<size_t>(i) * 8,
		       static_cast<uint64_t>(sin_v));
	}
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
	CHECK(view.config.num_hidden_layers == 40);
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
	// N5, second half (Poirot 380b75f review): the S3 crash probe above drives
	// RowBoundsWide directly; nothing drove NarrowRowChecked(nullptr, 0, out) --
	// the actual caller the S3 finding was raised against, and the one path a
	// null-row call reaches in production. NarrowRowChecked calls RowBoundsWide
	// internally (checked_chain_funnel.cpp), so this probe also exercises S3's
	// own fix at one further remove.
	if (name == "narrow_row_checked_zero_len_null_ptr") {
		int32_t out_i32[1] = {0};
		std::printf("%s\n", CrashProbeBeganMarker(name).c_str());
		std::printf("crash-probe narrow_row_checked_zero_len_null_ptr: calling "
		            "NarrowRowChecked with a null wide-row pointer and n=0 (no n >= 1 "
		            "precondition is documented on this entry point, N5)\n");
		std::fflush(stdout);
		superslm::NarrowRowChecked(nullptr, /*n=*/0, out_i32);
		std::printf("PROBE DID NOT CRASH\n");
		return 0;
	}
	// Critical 1 (closed; Poirot fa3189a-s3.3-rope-site-and-c32-softmax-review-
	// 2026-07-28.md, confirmed closed by Claude/Poirot/72b0c7f-s3.3-rope-site-and-
	// c32-softmax-confirmation-2026-07-28.md): RopeApplySite now guards its "cos"/
	// "sin" tensor lookup and returns RopeTableTensorMissing before any read, rather
	// than dereferencing a null pointer. SslmTensorManifest::Tensor returns nullptr
	// when the name is absent (model.h's own contract), and a zero-tensor ROP1
	// manifest loads Ok (ParseImpl bounds tensor_count only ABOVE kMaxTensors;
	// ValidateRopeTablesDomain walks whatever tensors are present and requires no
	// particular name) -- so a real, successfully loaded model can carry
	// has_rope_tables == true with Tensor("cos") == nullptr, and this probe proves
	// the guard intercepts that case rather than faulting. Still isolated here
	// because a regression that removed the guard would again fault the process --
	// a cell that crashes the runner is not a usable red (this suite's own
	// established death-test convention, S2.5).
	if (name == "rope_apply_site_null_cos_tensor_deref") {
		using namespace superslm_test;
		Cfg1Spec spec{};
		spec.head_dim = 8;
		spec.context_cap = 1;
		// hidden_size must track the overridden head_dim to satisfy R1
		// (SslmModel::Load's config-geometry join, S3.3 §13.1 cell 4,
		// D-SLM420-D-SLM423) -- Cfg1Spec{}'s own default (4096) only pairs
		// with the default head_dim (128).
		spec.hidden_size = spec.num_attention_heads * spec.head_dim;
		FixtureSection config = MakeSection(SslmSectionType::Config, SslmDtype::Raw, BuildCfg1(spec));
		auto manifest = BuildManifest(superslm::kRopeMagic, /*element_size=*/8, /*tensors=*/{});
		FixtureSection rope_tables =
		    MakeSection(SslmSectionType::RopeTables, SslmDtype::Int64, manifest.bytes, /*alignment=*/64);
		auto built = BuildArtifact({config, MakeSigmoidLutSection(), rope_tables});

		SslmModelView view;
		std::string err;
		SslmModelStatus status = SslmModel::Load(built.bytes.data(), built.bytes.size(), view, &err);
		if (status != SslmModelStatus::Ok || !view.has_rope_tables ||
		    view.rope_tables.Tensor("cos") != nullptr) {
			std::printf("PROBE SETUP FAILED (Load status=%s has_rope_tables=%d cos_tensor_is_null=%d) "
			            "-- the fixture no longer reaches the state this probe requires\n",
			            SslmModelStatusName(status), static_cast<int>(view.has_rope_tables),
			            static_cast<int>(view.has_rope_tables && view.rope_tables.Tensor("cos") == nullptr));
			return 3;
		}
		std::printf("%s\n", CrashProbeBeganMarker(name).c_str());
		std::printf("crash-probe rope_apply_site_null_cos_tensor_deref: calling RopeApplySite against "
		            "a loaded, zero-tensor ROP1 manifest (Tensor(\"cos\") == nullptr) -- Critical 1, "
		            "src/forward/forward_sites.cpp:329-330,342-343\n");
		std::fflush(stdout);
		int8_t row[8] = {0, 0, 0, 0, 0, 0, 0, 0};
		int8_t out_row[8] = {0, 0, 0, 0, 0, 0, 0, 0};
		const auto forward_status =
		    superslm::RopeApplySite(row, 8, /*position=*/0, /*context_cap=*/1, view.rope_tables, out_row);
		std::printf("PROBE DID NOT CRASH (forward_status=%s)\n",
		            superslm::SslmForwardStatusName(forward_status));
		return 0;
	}
	// Critical 2 (closed; Poirot fa3189a review, confirmed closed by the 72b0c7f
	// confirmation pass): the position guard now bounds `position` against the
	// ROP1 tensors' own extent, not merely against the caller's `context_cap` --
	// `row_offset` is checked against `cos->elem_count`/`sin->elem_count` before
	// either tensor is read. A `position` the caller's `context_cap` legally
	// admits, but far past a real, loaded tensor's actual row count, would
	// previously have read unmapped heap memory. This probe mirrors the review's
	// own executed reproduction exactly (position=268435456,
	// context_cap=2147483647, a genuine CFG1-admissible u32 value, against a real
	// 1-row-times-4-pair ROP1 tensor parsed by the shipped loader) and proves the
	// extent guard intercepts it. Still isolated here because a regression that
	// removed the guard would again fault the process.
	if (name == "rope_apply_site_position_far_past_tensor_extent") {
		using namespace superslm_test;
		Cfg1Spec spec{};
		spec.head_dim = 8;
		// context_cap matches the real tensor's own row count (1) below, and
		// hidden_size tracks the overridden head_dim -- both needed to satisfy
		// SslmModel::Load's config-geometry/ROP1 join (R1, R4; S3.3 §13.1 cell
		// 4, D-SLM420-D-SLM423). The probe's own direct RopeApplySite call
		// below still passes context_cap=2147483647 as an explicit argument,
		// decoupled from what Load validated -- that decoupling is the point
		// of this probe and is unaffected by this fixture's own config.
		spec.context_cap = 1;
		spec.hidden_size = spec.num_attention_heads * spec.head_dim;
		FixtureSection config = MakeSection(SslmSectionType::Config, SslmDtype::Raw, BuildCfg1(spec));
		std::vector<ManifestTensorSpec> tensors = {{"cos", {1, 4}}, {"sin", {1, 4}}};
		auto manifest = BuildManifest(superslm::kRopeMagic, /*element_size=*/8, tensors);
		for (size_t i = 0; i < 4; ++i) {
			PutU64(manifest.bytes, static_cast<size_t>(manifest.tensor_data_off[0]) + i * 8,
			       static_cast<uint64_t>(INT64_C(1073741824)));  // identity cos row
			PutU64(manifest.bytes, static_cast<size_t>(manifest.tensor_data_off[1]) + i * 8, 0);  // sin row
		}
		FixtureSection rope_tables =
		    MakeSection(SslmSectionType::RopeTables, SslmDtype::Int64, manifest.bytes, /*alignment=*/64);
		auto built = BuildArtifact({config, MakeSigmoidLutSection(), rope_tables});

		SslmModelView view;
		std::string err;
		SslmModelStatus status = SslmModel::Load(built.bytes.data(), built.bytes.size(), view, &err);
		if (status != SslmModelStatus::Ok || !view.has_rope_tables) {
			std::printf("PROBE SETUP FAILED (Load status=%s has_rope_tables=%d) -- the fixture no "
			            "longer reaches the state this probe requires\n",
			            SslmModelStatusName(status), static_cast<int>(view.has_rope_tables));
			return 3;
		}
		std::printf("%s\n", CrashProbeBeganMarker(name).c_str());
		std::printf("crash-probe rope_apply_site_position_far_past_tensor_extent: calling "
		            "RopeApplySite(head_dim=8, position=268435456, context_cap=2147483647) against a "
		            "real, loaded 1-row ROP1 tensor -- Critical 2, src/forward/forward_sites.cpp:"
		            "316-343\n");
		std::fflush(stdout);
		int8_t row[8] = {0, 0, 0, 0, 0, 0, 0, 0};
		int8_t out_row[8] = {0, 0, 0, 0, 0, 0, 0, 0};
		const auto forward_status = superslm::RopeApplySite(row, 8, /*position=*/268435456,
		                                                     /*context_cap=*/2147483647, view.rope_tables,
		                                                     out_row);
		std::printf("PROBE DID NOT CRASH (forward_status=%s)\n",
		            superslm::SslmForwardStatusName(forward_status));
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

// N5, second half (Poirot 380b75f review): "the crash probe covers RowBoundsWide
// directly; nothing drives NarrowRowChecked(nullptr, 0, out), which is the caller
// the finding was raised against." This cell drives that exact call.
// NarrowRowChecked(nullptr, 0, out_i32) resolves through three steps today: (1)
// RowBoundsWide(nullptr, 0, ...) -- the S3 cell above's own case, already fixed
// to return (0,0) without touching x; (2) the (0,0) result is trivially within
// [INT32_MIN, INT32_MAX], so C35's own check accepts; (3) NarrowAccumulatorToI32
// loops zero times over a null row, writing nothing. None of the three steps
// dereferences a null pointer at n == 0, so this cell asserts kRanNoCrash
// unconditionally, mirroring the sibling cell's own reasoning and the same
// "n == 0 is in-contract, not a caller-ensures violation" convention.
static void TestNarrowRowCheckedZeroLenNullPtrDoesNotCrash() {
	static const char* kProbeName = "narrow_row_checked_zero_len_null_ptr";
	std::string tail;
	CrashProbeOutcome outcome = RunsCrashProbeAndCrashes(kProbeName, &tail);
	CHECK_MSG(outcome == CrashProbeOutcome::kRanNoCrash,
	          "NarrowRowChecked(nullptr, 0, out_i32) must not crash in any build configuration "
	          "-- outcome was %s, child output was: %s",
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
	// A deliberately incoherent shape, unrelated to this slot: 24 heads * 128
	// head_dim = 3072 != hidden_size 4096. Built explicitly rather than via
	// Cfg1Spec{}'s own defaults -- since SslmModel::Load's config-geometry join
	// landed (S3.3 §13.1 cell 4, D-SLM420-D-SLM423), Cfg1Spec{}'s defaults are
	// themselves R1-coherent (32*128 == 4096, the sibling coherent-case test's
	// own explicit values above), so this cell can no longer borrow the shared
	// default's incoherence and states its own.
	Cfg1Spec incoherent;
	incoherent.num_attention_heads = 24;
	incoherent.head_dim = 128;  // hidden_size stays the default 4096: 24*128=3072 != 4096
	auto built = BuildArtifact(
	    {MakeSection(SslmSectionType::Config, SslmDtype::Raw, BuildCfg1(incoherent)), MakeSigmoidLutSection()});

	// This artifact is deliberately R1-incoherent, so SslmModel::Load itself now
	// rejects it too (ConfigGeometryHiddenSizeMismatch) -- that is the correct,
	// intended outcome of cell 4's own obligation, not a regression, and is
	// exercised by cell 4's own red suite above. This cell's purpose is
	// different: proving BuildProofManifestJson's "config_geometry" field
	// reflects CheckConfigGeometry's own independent verdict rather than a
	// hardcoded "ok" -- which needs only that the artifact is otherwise
	// structurally valid (OpenFromMemory succeeds), not that SslmModel::Load
	// accepts it.
	SslmArtifact artifact;
	SslmError aerr;
	const SslmStatus open_status = SslmArtifact::OpenFromMemory(built.bytes.data(), built.bytes.size(), artifact, &aerr);
	CHECK_MSG(open_status == SslmStatus::Ok,
	          "the incoherent-geometry-but-otherwise-valid fixture failed to open (a defect unrelated to "
	          "this cell's own target): got status %d",
	          static_cast<int>(open_status));
	if (open_status != SslmStatus::Ok) return;

	const std::string manifest = BuildProofManifestJson(artifact);
	CHECK_MSG(manifest.find("\"ok\": false") != std::string::npos,
	          "proof manifest for an incoherent shape (24*128=3072 != hidden_size 4096) must report "
	          "config_geometry.ok == false; manifest:\n%s",
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
//     descriptor (already shipped, S-HARDEN-1) and the runtime no-UB domain C34's
//     production predicate encodes.
//
//     CheckSiluCompositionScaleDomain (declared in checked_chain_funnel.h, defined in
//     src/forward/checked_chain_funnel.cpp -- cited by symbol, not line: three prior
//     rounds' line citations were each invalidated by a later commit in the same
//     range, Poirot 7c74636 confirmation review NEW-4) is real and is what every
//     cell below calls -- CLOSING the prior round's own finding (Poirot 380b75f
//     review, N3):
//     the containment cell used to call a test-local reimplementation instead of the
//     production predicate, which let the two diverge with no failure. The formula
//     below (IndependentSiluCompositionEDomainFormula) is retained ONLY as an
//     independently-derived cross-check, computed from the same two named public
//     constants the predicate's own shift-placement reasoning uses
//     (kSiluLutTermLeftShiftOverflowExponent, kRoundingDivideByPotExponentMaxI64) --
//     it is never itself the thing under test; every assertion below is against
//     CheckSiluCompositionScaleDomain's own return value. ---
static bool IndependentSiluCompositionEDomainFormula(int e) {
	using namespace superslm;
	const int shift_upper = e + kSiluLutLog2K + kSiluLutQIdx;
	const int shift_lower = -e - kSiluLutLog2K - kSiluLutQIdx;
	return shift_upper < kSiluLutTermLeftShiftOverflowExponent &&
	       shift_lower <= kRoundingDivideByPotExponentMaxI64;
}

// The |m| axis's own independent formula (CheckSiluCompositionScaleDomain's own
// documented bound, checked_chain_funnel.h -- cited by symbol per NEW-4 above):
// the same symmetric kCompositionScaleMaxAbsM ceiling
// SiluSigmoidQ15's term = code*m needs, a real public constant (silu_lut.h),
// pinned equal to kInt32Max by model.cpp's own static_assert.
static bool IndependentSiluCompositionMDomainFormula(int64_t m) {
	return m >= -superslm::kCompositionScaleMaxAbsM && m <= superslm::kCompositionScaleMaxAbsM;
}

static bool C34LoadTimeAcceptsE(int64_t m, int64_t e) {
	using namespace superslm_test;
	auto built = BuildArtifact(
	    {MakeValidConfigSection(), MakeSigmoidLutSection(), MakeKvc1CompositionSection(m, e)});
	SslmModelView view;
	std::string err;
	SslmModelStatus status = SslmModel::Load(built.bytes.data(), built.bytes.size(), view, &err);
	return status == SslmModelStatus::Ok;
}

// NEW-5 (Poirot 7c74636 confirmation review): this cell used to sweep e alone
// with m fixed at 0 -- a value in neither of Sec5.4's own two named mantissas
// (2^30, 2^31-1) -- so the containment relation the design actually specifies
// (swept over BOTH m in {2^30, 2^31-1} and e) was measured on one axis only.
// Fixed by sweeping the outer loop over both named mantissas; both are within
// kCompositionScaleMaxAbsM (2^30 comfortably, 2^31-1 exactly at the boundary),
// so the |m| branch never fires here -- this cell's own scope is the e-axis
// containment relation at each mantissa, not the |m| branch (that is
// TestCheckSiluCompositionScaleDomainRejectsMOutsideNoUbAbsBound's own scope).
static void TestCheckSiluCompositionScaleDomainContainsTheShippedLoadTimeDescriptor() {
	using superslm::CheckSiluCompositionScaleDomain;
	using superslm::SslmForwardStatus;
	constexpr int64_t kMLow = INT64_C(1073741824);   // 2^30
	constexpr int64_t kMHigh = INT64_C(2147483647);  // 2^31 - 1
	for (int64_t m : {kMLow, kMHigh}) {
		// Containment, swept: every e the real load-time gate accepts at this
		// mantissa, the real production predicate also accepts. Never asserted
		// the other way (the runtime domain is deliberately wider -- Sec5.4's
		// whole point).
		for (int e = -85; e <= 15; ++e) {
			if (C34LoadTimeAcceptsE(m, e)) {
				CHECK_MSG(CheckSiluCompositionScaleDomain(m, e) == SslmForwardStatus::Ok,
				          "m=%lld e=%d: load-time ACCEPTS but CheckSiluCompositionScaleDomain(m, e) "
				          "== %s -- containment violated (Sec5.4)",
				          static_cast<long long>(m), e,
				          superslm::SslmForwardStatusName(CheckSiluCompositionScaleDomain(m, e)));
			}
		}

		// The four named boundary points, pinned by their EXECUTED disposition
		// (Sec5.4), at this mantissa: e=7 both accept; e=8 the one point of
		// difference (runtime accepts, load rejects); e=9 both reject (the
		// overflow point itself, shift==26); e=-80 both accept (the shared exact
		// lower endpoint), e=-81 both reject.
		CHECK_MSG(
		    C34LoadTimeAcceptsE(m, 7) && CheckSiluCompositionScaleDomain(m, 7) == SslmForwardStatus::Ok,
		    "m=%lld e=7: both must accept (the load-time ceiling, shift 24)", static_cast<long long>(m));
		CHECK_MSG(!C34LoadTimeAcceptsE(m, 8) &&
		              CheckSiluCompositionScaleDomain(m, 8) == SslmForwardStatus::Ok,
		          "m=%lld e=8: load-time must reject and the runtime predicate must accept -- the "
		          "one point of difference Sec5.4 executes",
		          static_cast<long long>(m));
		CHECK_MSG(!C34LoadTimeAcceptsE(m, 9) &&
		              CheckSiluCompositionScaleDomain(m, 9) == SslmForwardStatus::SiluCompositionScaleOutOfDomain,
		          "m=%lld e=9: both must reject (shift==26, the overflow point itself)",
		          static_cast<long long>(m));
		CHECK_MSG(C34LoadTimeAcceptsE(m, -80) &&
		              CheckSiluCompositionScaleDomain(m, -80) == SslmForwardStatus::Ok,
		          "m=%lld e=-80: both must accept (the shared exact lower endpoint)",
		          static_cast<long long>(m));
		CHECK_MSG(!C34LoadTimeAcceptsE(m, -81) &&
		              CheckSiluCompositionScaleDomain(m, -81) == SslmForwardStatus::SiluCompositionScaleOutOfDomain,
		          "m=%lld e=-81: both must reject", static_cast<long long>(m));
	}

	// The load-time ceiling's own static_assert names the overflow point exactly:
	// shift(7) = 7+17 = 24 < 26 (the overflow exponent) -- confirmed live against the
	// real named constants rather than restated as a literal.
	CHECK(7 + superslm::kSiluLutLog2K + superslm::kSiluLutQIdx <
	      superslm::kSiluLutTermLeftShiftOverflowExponent);
	CHECK(9 + superslm::kSiluLutLog2K + superslm::kSiluLutQIdx ==
	      superslm::kSiluLutTermLeftShiftOverflowExponent);
}

// N3 (Poirot 380b75f review): "the design's own sweep is realized on one of its two
// axes" -- Sec5.4 specifies the differential cell swept over BOTH m in {2^30,
// 2^31-1} and e in [-90, +12], both branch selections; the shipped cell (above)
// swept e alone at m fixed to 0, a value in neither of the design's two named
// mantissas, and never varied m at all. This cell realizes the missing axis:
// CheckSiluCompositionScaleDomain(m, e) against the independently-derived formula
// (both axes), across the design's own named sweep. Both named mantissas are
// within kCompositionScaleMaxAbsM (2^30 comfortably, 2^31-1 exactly at the
// boundary), so the |m| branch never fires within this sweep -- expected agreement
// reduces to the e-formula alone for both mantissas, which this cell confirms
// rather than assumes, closing the possibility of an m/e coupling bug the
// design's own two-axis sweep exists to catch.
static void TestCheckSiluCompositionScaleDomainAgreesWithIndependentFormulaAcrossMESweep() {
	using superslm::CheckSiluCompositionScaleDomain;
	using superslm::SslmForwardStatus;
	constexpr int64_t kMLow = INT64_C(1073741824);   // 2^30
	constexpr int64_t kMHigh = INT64_C(2147483647);  // 2^31 - 1, == kCompositionScaleMaxAbsM
	int checked = 0;
	for (int64_t m : {kMLow, kMHigh}) {
		CHECK(IndependentSiluCompositionMDomainFormula(m));  // both named mantissas are in-domain on the |m| axis
		for (int e = -90; e <= 12; ++e) {
			const bool expected_ok = IndependentSiluCompositionEDomainFormula(e);
			const auto status = CheckSiluCompositionScaleDomain(m, e);
			CHECK_MSG((status == SslmForwardStatus::Ok) == expected_ok,
			          "m=%lld e=%d: CheckSiluCompositionScaleDomain == %s, want %s (independent "
			          "e-domain formula, Sec5.4's own design sweep)",
			          static_cast<long long>(m), e, superslm::SslmForwardStatusName(status),
			          expected_ok ? "Ok" : "SiluCompositionScaleOutOfDomain");
			++checked;
		}
	}
	CHECK_MSG(checked == 2 * 103, "sweep must cover both named mantissas across e in [-90,12] "
	                              "(103 values each) -- got %d checks, fixture regressed",
	          checked);
}

// N3's other half: the predicate's entire |m| branch
// (m < -kCompositionScaleMaxAbsM || m > kCompositionScaleMaxAbsM) had no cell of any
// kind (380b75f review); Sec11 S3.1's own red-cell list names "a (m,e) outside C34's
// no-UB domain is rejected" as required. e is held at 0 (comfortably in-domain)
// throughout, isolating the assertion to the |m| axis alone.
static void TestCheckSiluCompositionScaleDomainRejectsMOutsideNoUbAbsBound() {
	using superslm::CheckSiluCompositionScaleDomain;
	using superslm::SslmForwardStatus;
	using superslm::kCompositionScaleMaxAbsM;

	CHECK_MSG(CheckSiluCompositionScaleDomain(kCompositionScaleMaxAbsM, 0) == SslmForwardStatus::Ok,
	          "m == kCompositionScaleMaxAbsM (the positive boundary, in-domain): must accept");
	CHECK_MSG(CheckSiluCompositionScaleDomain(kCompositionScaleMaxAbsM + 1, 0) ==
	              SslmForwardStatus::SiluCompositionScaleOutOfDomain,
	          "m == kCompositionScaleMaxAbsM + 1: must reject SiluCompositionScaleOutOfDomain "
	          "(the |m| branch this cell exists to cover)");
	CHECK_MSG(CheckSiluCompositionScaleDomain(-kCompositionScaleMaxAbsM, 0) == SslmForwardStatus::Ok,
	          "m == -kCompositionScaleMaxAbsM (the negative boundary, in-domain): must accept");
	// The bound is symmetric on kCompositionScaleMaxAbsM (== kInt32Max), NOT on
	// kInt32Min -- INT32_MIN is one past the negative boundary and must reject,
	// the asymmetric case a naive int32-range check would miss.
	CHECK_MSG(CheckSiluCompositionScaleDomain(superslm::kInt32Min, 0) ==
	              SslmForwardStatus::SiluCompositionScaleOutOfDomain,
	          "m == INT32_MIN (one past -kCompositionScaleMaxAbsM, since the bound is symmetric "
	          "on kInt32Max rather than kInt32Min): must reject SiluCompositionScaleOutOfDomain");
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
// C29's own D' > 2^31 domain check, RequantChainChecked's own step 3 in
// src/forward/checked_chain_funnel.cpp -- cited by symbol, not line, per NEW-4)
// rather than from a specific fix construction: a row whose true magnitude is 2^63 is
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

// S4: step 5 of RequantChainChecked (renumbered from step 6 by commit 11650a8,
// which reconciled the header's own step ordering against the fold-before-
// write execution order) -- C26's carried-scale product, computed in
// src/forward/checked_chain_funnel.cpp (cited by symbol, not line, per
// NEW-4) -- HAD no assertion of any kind anywhere in the suite at review time
// (Poirot ac34677 review); pinned by this cell below since. The only prior
// statement about *out_scale (TestRequantChainCheckedT1254Witness) asserted
// merely that it differs from its poison value; two mutants defeated that
// (Poirot M3, M5): replacing the whole computed value with a constant, and
// reversing the fold to
// right-associated. This cell pins the EXACT value, derived independently
// of src/forward/checked_chain_funnel.cpp -- from the vendored reference's
// own carried_scale_product (tests/reference/superslm_spike/intmath.py:
// 410-428), the same left-associated mechanism C26 pins and this design's
// own pinned mechanism (Sec7.2 step 5, post-11650a8 numbering), executed directly in Python rather
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

// N1 (Poirot 380b75f confirmation review): step 0 (RequantChainChecked's own
// entry check, src/forward/checked_chain_funnel.cpp -- cited by symbol, not
// line, per the 7c74636 confirmation review's NEW-4) checks every `incoming`
// factor and `site_constant` against
// CarriedScaleMantissaFitsInt32 before the fold runs, but never checks `running` --
// the fold's own left operand on every combine after the first, and exactly what
// CombineCarriedScale receives. CombineCarriedScale's renormalization
// (`if (m < (int64_t{1} << 30)) { m <<= 1; ... }`) is a SIGNED comparison, so a
// negative high-mul result is doubled rather than renormalized, and the next fold's
// unconditional `static_cast<int32_t>(a.m)` silently truncates whatever that
// produced. This cell pins the header's own documented promise
// (checked_chain_funnel.h:42-51: "returning CarriedScaleMantissaOutOfDomain rather
// than truncating silently") against the two in-domain endpoint factors the
// reviewer's own probe used (P1/P2): both accepted by step 0, both exactly at the
// endpoints the precondition names, and their fold's own high-mul result is
// {m=-4294967294} -- outside int32 -- which the next fold's cast turns into {m=2}.
static void TestRequantChainCheckedRunningProductOutOfInt32DomainIsRejectedNotTruncated() {
	using namespace superslm_test;
	using superslm::CarriedScale;
	using superslm::SslmForwardStatus;

	// Reviewer's own executed witness (380b75f review Sec4, P1/P2): both factors
	// are individually in-domain (each fits int32_t exactly, at its own endpoint),
	// so step 0 accepts both; only their FOLDED PRODUCT escapes int32.
	const CarriedScale incoming[] = {
	    CarriedScale{/*m=*/INT64_C(-2147483648), /*e=*/0},  // INT32_MIN, endpoint-in-domain
	    CarriedScale{/*m=*/INT64_C(2147483647), /*e=*/0},   // INT32_MAX, endpoint-in-domain
	};
	const CarriedScale site_constant{/*m=*/INT64_C(1073741824), /*e=*/0};  // canonical, neutral

	int8_t out_codes[kWideT1254WitnessRowLen] = {INT8_C(-99), INT8_C(-99), INT8_C(-99), INT8_C(-99)};
	CarriedScale out_scale{INT64_C(-99), INT64_C(-99)};  // poison
	auto result = superslm::RequantChainChecked(kWideT1254WitnessPositiveRow, kWideT1254WitnessRowLen,
	                                              std::span<const CarriedScale>(incoming, 2),
	                                              site_constant, out_codes, &out_scale);

	CHECK_MSG(result.status == SslmForwardStatus::CarriedScaleMantissaOutOfDomain,
	          "RequantChainChecked({INT32_MIN,0},{INT32_MAX,0}) status == %s, want "
	          "CarriedScaleMantissaOutOfDomain (checked_chain_funnel.h:42-51's own documented "
	          "promise) -- both factors are individually in-domain, so this can only be caught "
	          "by checking the FOLDED running product, which step 0 does not do today; a "
	          "status of Ok here means the fold's own out-of-int32 result silently truncated "
	          "through CombineCarriedScale's unconditional cast instead of being rejected "
	          "(380b75f review Sec4, P1/P2: folds to out_scale={m=2, e=91})",
	          superslm::SslmForwardStatusName(result.status));
	CHECK_MSG(out_codes[0] == INT8_C(-99),
	          "on the documented CarriedScaleMantissaOutOfDomain rejection, out_codes must be "
	          "untouched (\"computes nothing\", same contract every other rejection path holds)");
	CHECK_MSG(out_scale.m == INT64_C(-99) && out_scale.e == INT64_C(-99),
	          "on the documented CarriedScaleMantissaOutOfDomain rejection, *out_scale must be "
	          "untouched -- got {m=%lld, e=%lld} (the review's own witness: an unguarded fold "
	          "silently produces {m=2, e=91} here instead of leaving the poison value alone)",
	          static_cast<long long>(out_scale.m), static_cast<long long>(out_scale.e));
}

// N1's compounding half: CarriedScaleMantissaOutOfDomain appeared zero times in
// tests/ (380b75f review Sec4) -- an entirely new status shipped with no cell of
// any kind. CORRECTED (Poirot 7c74636 confirmation review, NEW-1): the comment
// this cell carried claimed it "exercises step 0's existing, already-correct
// per-operand check, and is expected to pass today" -- checked by mutation, that
// claim is false. On the row used here (kWideT1254WitnessPositiveRow, D' == 2^31,
// within C29's domain), an out-of-domain `incoming` or `site_constant` factor
// becomes the FIRST value `running` takes in the fold (`fold_in`'s own
// `running = have_running ? Combine(...) : next` branch assigns the raw operand,
// uncombined, on the first factor), so the fold's own running-product check
// (N1's remedy) rejects it with the identical CarriedScaleMantissaOutOfDomain
// status even with step 0 deleted in full -- confirmed by mutation: 0 of this
// cell's checks fail under the step-0-deleted mutant. What this cell actually
// pins is narrower than its old comment claimed: an out-of-domain operand is
// rejected with CarriedScaleMantissaOutOfDomain on an otherwise C29-in-domain
// row, by WHICHEVER mechanism enforces it (step 0 today; the fold's own check
// would enforce it alone if step 0 were ever removed). The cell that isolates
// step 0 specifically -- a row that ALSO violates C29, so only step 0's
// ahead-of-C29 ordering produces the documented status -- is
// TestRequantChainCheckedStepZeroPrecedesC29OnARowViolatingBoth, immediately
// below.
static void TestRequantChainCheckedRejectsIncomingFactorMantissaOutOfInt32Domain() {
	using namespace superslm_test;
	using superslm::CarriedScale;
	using superslm::SslmForwardStatus;

	const CarriedScale site_constant{/*m=*/INT64_C(1073741824), /*e=*/0};

	// A single `incoming` factor one past INT32_MAX, on a row within C29's own
	// domain -- rejected today by step 0's per-operand check, and (per the
	// comment above) also by the fold's own running-product check alone.
	{
		const CarriedScale incoming[] = {CarriedScale{/*m=*/INT64_C(2147483648), /*e=*/0}};
		int8_t out_codes[kWideT1254WitnessRowLen] = {INT8_C(-99), INT8_C(-99), INT8_C(-99), INT8_C(-99)};
		CarriedScale out_scale{INT64_C(-99), INT64_C(-99)};
		auto result = superslm::RequantChainChecked(
		    kWideT1254WitnessPositiveRow, kWideT1254WitnessRowLen,
		    std::span<const CarriedScale>(incoming, 1), site_constant, out_codes, &out_scale);
		CHECK_MSG(result.status == SslmForwardStatus::CarriedScaleMantissaOutOfDomain,
		          "incoming[0].m == INT32_MAX+1: RequantChainChecked status == %s, want "
		          "CarriedScaleMantissaOutOfDomain (enforced today by step 0, and equally by the "
		          "fold's own running-product check alone -- this cell does not discriminate "
		          "between the two on this row)",
		          superslm::SslmForwardStatusName(result.status));
		CHECK_MSG(out_codes[0] == INT8_C(-99) && out_scale.m == INT64_C(-99) && out_scale.e == INT64_C(-99),
		          "incoming[0].m == INT32_MAX+1: out_codes and *out_scale must be untouched on "
		          "rejection");
	}
	// site_constant one past INT32_MIN on the negative side -- the sibling operand
	// step 0 also checks, same non-discrimination as above.
	{
		const CarriedScale site_out_of_domain{/*m=*/INT64_C(-2147483649), /*e=*/0};
		int8_t out_codes[kWideT1254WitnessRowLen] = {INT8_C(-99), INT8_C(-99), INT8_C(-99), INT8_C(-99)};
		CarriedScale out_scale{INT64_C(-99), INT64_C(-99)};
		auto result = superslm::RequantChainChecked(kWideT1254WitnessPositiveRow, kWideT1254WitnessRowLen,
		                                              std::span<const CarriedScale>{}, site_out_of_domain,
		                                              out_codes, &out_scale);
		CHECK_MSG(result.status == SslmForwardStatus::CarriedScaleMantissaOutOfDomain,
		          "site_constant.m == INT32_MIN-1: RequantChainChecked status == %s, want "
		          "CarriedScaleMantissaOutOfDomain (enforced today by step 0, and equally by the "
		          "fold's own running-product check alone -- this cell does not discriminate "
		          "between the two on this row)",
		          superslm::SslmForwardStatusName(result.status));
		CHECK_MSG(out_codes[0] == INT8_C(-99) && out_scale.m == INT64_C(-99) && out_scale.e == INT64_C(-99),
		          "site_constant.m == INT32_MIN-1: out_codes and *out_scale must be untouched on "
		          "rejection");
	}
}

// NEW-1 (Poirot 7c74636 confirmation review, the Significant): step 0 (the
// per-operand check above) is almost entirely redundant now that the fold's
// own running-product check (N1's remedy, immediately above) exists -- any
// operand step 0 would reject is also the first value `running` takes, so the
// fold rejects it with the identical status (proven by the cell above, and by
// mutation: deleting step 0 leaves the whole 9,710-check suite green). The one
// residue is STATUS PRECEDENCE, and it is real: on a row that ALSO violates
// C29 (step 3's own D' > 2^31 check) while carrying an out-of-domain operand,
// step 0 rejects with CarriedScaleMantissaOutOfDomain BEFORE step 3 ever runs;
// delete step 0, and the identical call instead reaches step 3 first and
// returns ChainInputOutOfDomain. Which status a call returns is load-bearing
// (Sec10.4's diagnostic table names the two statuses to different site
// classes), so this cell pins the shipped precedence explicitly and is the
// one cell in this suite that dies when step 0 is deleted (mutation-verified).
static void TestRequantChainCheckedStepZeroPrecedesC29OnARowViolatingBoth() {
	using namespace superslm_test;
	using superslm::CarriedScale;
	using superslm::SslmForwardStatus;

	const CarriedScale site_constant{/*m=*/INT64_C(1073741824), /*e=*/0};
	// kWideOverC29DomainRow (D' == 2^31+1) violates C29 on its own
	// (TestRequantChainCheckedRejectsOverC29Domain, no out-of-domain operand
	// involved); incoming[0].m one past INT32_MAX violates step 0's own
	// per-operand check on its own (the cell above, on an otherwise
	// C29-in-domain row). Combined on ONE call, only step 0's ahead-of-C29
	// ordering can produce CarriedScaleMantissaOutOfDomain here -- with step 0
	// deleted, step 3's C29 check fires first and the fold never runs.
	const CarriedScale incoming[] = {CarriedScale{/*m=*/INT64_C(2147483648), /*e=*/0}};
	int8_t out_codes[kWideOverC29DomainRowLen] = {INT8_C(-99)};
	CarriedScale out_scale{INT64_C(-99), INT64_C(-99)};
	auto result = superslm::RequantChainChecked(
	    kWideOverC29DomainRow, kWideOverC29DomainRowLen, std::span<const CarriedScale>(incoming, 1),
	    site_constant, out_codes, &out_scale);
	CHECK_MSG(result.status == SslmForwardStatus::CarriedScaleMantissaOutOfDomain,
	          "D'=2^31+1 row, incoming[0].m == INT32_MAX+1: RequantChainChecked status == %s, want "
	          "CarriedScaleMantissaOutOfDomain (step 0's own rejection must fire ahead of C29's "
	          "step-3 check on a row that violates both -- deleting step 0 makes this return "
	          "ChainInputOutOfDomain instead, per Sec10.4's diagnostic table)",
	          superslm::SslmForwardStatusName(result.status));
	CHECK_MSG(out_codes[0] == INT8_C(-99) && out_scale.m == INT64_C(-99) && out_scale.e == INT64_C(-99),
	          "D'=2^31+1 row, incoming[0].m == INT32_MAX+1: out_codes and *out_scale must be "
	          "untouched on rejection");
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

	superslm::SslmTraceHookState hook_state;

	int8_t codes_no_hook[4] = {0, 0, 0, 0};
	CarriedScale scale_no_hook{};
	auto result_no_hook = superslm::RequantChainChecked(
	    kWideT1254WitnessPositiveRow, kWideT1254WitnessRowLen, std::span<const CarriedScale>{},
	    site_constant, codes_no_hook, &scale_no_hook, "s3.1a_reduced_probe", /*token_index=*/7,
	    &hook_state);
	CHECK_MSG(!superslm::SslmTraceHookInstalled(hook_state),
	          "no hook installed yet -- SslmTraceHookInstalled(state) must read false");

	std::vector<ChainTraceSinkRecord> sink;
	superslm::SslmSetTraceHook(hook_state, &ChainTraceSinkHookFn, &sink);
	CHECK_MSG(superslm::SslmTraceHookInstalled(hook_state), "SslmSetTraceHook must install the hook");

	int8_t codes_with_hook[4] = {0, 0, 0, 0};
	CarriedScale scale_with_hook{};
	auto result_with_hook = superslm::RequantChainChecked(
	    kWideT1254WitnessPositiveRow, kWideT1254WitnessRowLen, std::span<const CarriedScale>{},
	    site_constant, codes_with_hook, &scale_with_hook, "s3.1a_reduced_probe",
	    /*token_index=*/7, &hook_state);

	superslm::SslmSetTraceHook(hook_state, nullptr, nullptr);
	CHECK_MSG(!superslm::SslmTraceHookInstalled(hook_state),
	          "SslmSetTraceHook(state, nullptr,...) must uninstall");

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

	superslm::SslmTraceHookState hook_state;
	std::vector<ChainTraceSinkRecord> sink;
	superslm::SslmSetTraceHook(hook_state, &ChainTraceSinkHookFn, &sink);

	const CarriedScale site_constant{INT64_C(1073741824), 0};
	int8_t out_codes[1] = {INT8_C(-99)};
	CarriedScale out_scale{INT64_C(-99), INT64_C(-99)};
	auto result = superslm::RequantChainChecked(kWideOverC29DomainRow, kWideOverC29DomainRowLen,
	                                              std::span<const CarriedScale>{}, site_constant,
	                                              out_codes, &out_scale, "s3.1a_reduced_probe_reject",
	                                              /*token_index=*/3, &hook_state);

	superslm::SslmSetTraceHook(hook_state, nullptr, nullptr);

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
// GREEN as of commit 380b75f: the trace hook's storage was converted from a
// process-wide static to per-handle state (SslmModelView::trace_hook, model.h;
// SslmTraceHookState, trace_hook.h), and SslmSetTraceHook/SslmTraceHookInstalled/
// SslmEmitChainTrace/SslmEmitKvLandingTrace all take that state as an explicit
// parameter rather than touching a file static. "handle A" and "handle B" below
// are represented by two independent (hook fn, sink) pairs standing in for the
// per-model trace state each SslmModelView::trace_hook now carries; a hook
// installed through handle A's own accessor is not reachable, or disturbable,
// from any call made through handle B, and vice versa. This cell is
// mutation-checked, not merely observed passing: reverting
// src/trace_hook.cpp to the prior process-global (fn, user) pair while keeping
// this cell's own two-state-instance call shape makes it fail at exactly its
// crux assertions (Poirot 380b75f review, M6) -- confirming the cell would
// catch a regression back to shared storage, not just that it passes today.
static void TestTraceHookCrossModelHandleIsolation() {
	using namespace superslm_test;
	using superslm::CarriedScale;

	const CarriedScale site_constant{/*m=*/INT64_C(1073741824), /*e=*/0};

	// Two independent SslmTraceHookState instances, standing in for two model
	// handles' own SslmModelView::trace_hook fields (D-SLM353's corrected
	// storage) -- the actual model handle threading the comment above calls
	// for, without constructing a full SslmModelView for this reduced,
	// funnel-level cell.
	superslm::SslmTraceHookState hook_state_a;
	superslm::SslmTraceHookState hook_state_b;

	std::vector<ChainTraceSinkRecord> sink_a;
	std::vector<ChainTraceSinkRecord> sink_b;

	// Install "on handle A" and drive one emitting call attributed to it.
	superslm::SslmSetTraceHook(hook_state_a, &ChainTraceSinkHookFn, &sink_a);
	{
		int8_t out_codes[4] = {0, 0, 0, 0};
		CarriedScale out_scale{};
		superslm::RequantChainChecked(kWideT1254WitnessPositiveRow, kWideT1254WitnessRowLen,
		                                std::span<const CarriedScale>{}, site_constant, out_codes,
		                                &out_scale, "isolation_probe_handle_a", /*token_index=*/1,
		                                &hook_state_a);
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
	superslm::SslmSetTraceHook(hook_state_b, &ChainTraceSinkHookFn, &sink_b);
	{
		int8_t out_codes[4] = {0, 0, 0, 0};
		CarriedScale out_scale{};
		superslm::RequantChainChecked(kWideT1254WitnessPositiveRow, kWideT1254WitnessRowLen,
		                                std::span<const CarriedScale>{}, site_constant, out_codes,
		                                &out_scale, "isolation_probe_handle_b", /*token_index=*/2,
		                                &hook_state_b);
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
		                                &out_scale, "isolation_probe_handle_a_second", /*token_index=*/3,
		                                &hook_state_a);
	}
	superslm::SslmSetTraceHook(hook_state_a, nullptr, nullptr);
	superslm::SslmSetTraceHook(hook_state_b, nullptr, nullptr);

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

// N6 (Poirot 380b75f review): SslmModelAccess::LoadImpl (src/model.cpp:764) opens
// with `out = SslmModelView{}`, whose move-assign runs MoveFrom and copies a
// default-constructed (no-hook) trace_hook over whatever `out` already carried.
// Loading a new artifact onto a view that already has a hook installed therefore
// uninstalls it, with no return value and no diagnostic; model.h:359-370 says only
// "Not populated by SslmModel::Load", which reads as "left alone" rather than
// "cleared."
//
// PINNED HERE: clearing is the correct behaviour, and is pinned rather than
// changed. Every other rejection path in this same function resets `out` to a
// fresh SslmModelView with the identical comment "fail closed -- never a partial
// view" (src/model.cpp); the unconditional reset at Load's own entry is the same
// discipline applied to the success path -- a caller of Load always receives a
// wholly fresh view, artifact state and any caller-attached instrumentation both
// reset, never a view that is part fresh (the new artifact's sections) and part
// stale (a hook installed against whatever was loaded before). Preserving the
// hook across a reload would require Load to special-case exactly one field
// during its own reset, which nothing in the design or the plan asks for. A
// caller that wants trace coverage across a reload reinstalls the hook after
// Load returns, the same way it would attach the hook the first time.
static void TestSslmModelLoadClearsPreviouslyInstalledTraceHook() {
	using namespace superslm_test;
	using superslm::SslmChainTraceRecord;
	using superslm::SslmKvLandingTraceRecord;

	auto built = BuildArtifact({MakeValidConfigSection(), MakeSigmoidLutSection()});
	SslmModelView view;
	std::string err;
	SslmModelStatus status = SslmModel::Load(built.bytes.data(), built.bytes.size(), view, &err);
	CHECK_MSG(status == SslmModelStatus::Ok, "first Load must succeed: got %s (%s)",
	          SslmModelStatusName(status), err.c_str());

	int dummy_user = 0;
	superslm::SslmSetTraceHook(
	    view.trace_hook,
	    +[](const SslmChainTraceRecord*, const SslmKvLandingTraceRecord*, void*) {}, &dummy_user);
	CHECK_MSG(superslm::SslmTraceHookInstalled(view.trace_hook),
	          "SslmSetTraceHook must install the hook before the reload under test");

	// Re-Load a second, independently-built valid artifact onto the SAME view.
	auto built2 = BuildArtifact({MakeValidConfigSection(), MakeSigmoidLutSection()});
	status = SslmModel::Load(built2.bytes.data(), built2.bytes.size(), view, &err);
	CHECK_MSG(status == SslmModelStatus::Ok, "second Load (same view) must succeed: got %s (%s)",
	          SslmModelStatusName(status), err.c_str());

	CHECK_MSG(!superslm::SslmTraceHookInstalled(view.trace_hook),
	          "SslmModel::Load must clear a previously-installed trace hook on the view it loads "
	          "into (pinned as the correct behaviour -- Load always produces a wholly fresh view, "
	          "matching the \"fail closed, never a partial view\" reset every other path in this "
	          "function already uses) -- got the hook still installed after a second Load");
}

// ---------------------------------------------------------------------------
// SuperSLM_S3a_WalkingSkeleton_Plan.md Sec11 S3.2 -- the weightless and
// projection sites (C31's RMSNorm site, C24/C25's WSC1 identity/near-identity
// fold-apply, C28's bias reconciliation, F-S3-8's embed entry, BIA1's
// load-time value-domain descriptor). Every symbol under test here is a STUB
// as of the S3.2 header-contract build (commit a594dd2:
// include/superslm/forward_sites.h / src/forward_sites.cpp;
// CheckRoundingDivideByPotExponentDomain in checked_chain_funnel.h/.cpp;
// BiasCodeOutOfDomain/ValidateBiasesDomain in model.h/src/model.cpp) -- every
// cell below is RED-UNIMPLEMENTED today, authored against the header contract
// ahead of Brunel's green phase, per Claude/Curie/superslm-s3.2-weightless-
// and-projection-sites-test-design-2026-07-28.md Sec4. Fixtures:
// tests/sslm_s3_2_fixtures.h (Curie's own derived-and-mutation-verified
// witnesses, Sec3/Sec6 of that record).
//
// NOT authored here, and why: Sec4.5's composition-level guard-vitality cell
// for C28's (q_B, e_a) predicate needs a RUNNING FORWARD reaching the C28 site
// with a hostile pair -- out of reach until the embed entry, the RMSNorm site,
// the projection sites, and the C28 site itself are all wired into one
// forward, the same class of gap S3.1's own Sec4.7 named for its own
// composition-level guard-vitality cell. It remains a routed, fully specified
// finding (test-design record Sec4.5), not invented coverage of a kind this
// pass cannot yet realize as real, compiling C++.
// ---------------------------------------------------------------------------

// Sec4.1 (C31, F-S3-2, Sec5.1): FloorDivI64's own three-row unit witness.
// The negative control is folded into this same cell rather than a second
// function: asserting FloorDivI64 equals floor_q on every row already fails
// against a truncating implementation on rows 0/1 (where floor_q != trunc_q);
// row 2 is an exact divide, where floor and truncation cannot diverge for any
// rounding-mode pair, so it has no discriminating power on its own and is not
// expected to.
//
// Mutation-verification (StandardsDocument Sec4/D-SLM357): this cell's own
// premise -- that a native C++ truncating divide reproduces the fixture's
// trunc_q, derived in Python by gen_s3_2_fixtures.py's own `_trunc_div` -- is
// checked here BY EXECUTING C++'s own `/` directly on every row, not assumed
// to survive the Python/C++ language boundary unchanged.
static void TestFloorDivI64C31UnitWitnessDivergesFromNativeTruncatingDivision() {
	using namespace superslm_test;
	for (size_t i = 0; i < kC31UnitCasesCount; ++i) {
		const C31UnitCase& c = kC31UnitCases[i];
		const int64_t numerator = c.hidden << 32;  // NORM_FRAC_BITS=16, 2*16=32 (Sec5.1)

		const int64_t native_trunc = numerator / c.roots;
		CHECK_MSG(native_trunc == c.trunc_q,
		          "row %zu (hidden=%lld, roots=%lld): native C++ (hidden<<32)/roots == %lld, "
		          "want the fixture's own trunc_q %lld -- this cell's negative-control premise "
		          "(a truncating divide reproduces trunc_q) does not hold in real C++",
		          i, static_cast<long long>(c.hidden), static_cast<long long>(c.roots),
		          static_cast<long long>(native_trunc), static_cast<long long>(c.trunc_q));
		if (c.differs) {
			CHECK_MSG(c.floor_q != c.trunc_q,
			          "row %zu: fixture claims differs=true but floor_q == trunc_q -- the "
			          "fixture itself has no discriminating power on this row",
			          i);
		}

		const int64_t got = superslm::FloorDivI64(numerator, c.roots);
		CHECK_MSG(got == c.floor_q,
		          "row %zu (hidden=%lld, roots=%lld): FloorDivI64(hidden<<32, roots) == %lld, "
		          "want %lld (F-S3-2's own floor-division witness)",
		          i, static_cast<long long>(c.hidden), static_cast<long long>(c.roots),
		          static_cast<long long>(got), static_cast<long long>(c.floor_q));
	}
}

// T-1267 (D-SLM360): pins RmsNormSite's own USE of FloorDivI64 -- a mechanism
// claim the very next test's own comment (below) proves CANNOT be discharged
// by any assertion on out_codes or *out_scale: floor-vs-truncation divergence
// in FloorDivI64 is unobservable at this site's int8 resolution (closed-form
// bound -- minimum reachable |floor_q| 516 against a needed magnitude below
// 254 -- and independently a 500,000-trial randomized search, zero hits).
// D-SLM360 files the residual as a checkable mechanism claim instead: mutate
// FloorDivI64 to truncate and confirm SOME cell that exercises RmsNormSite
// fails. This is that cell -- built the moment that requirement could not be
// met by strengthening the existing output-level assertion, which the
// preceding correction already proved is not possible at this resolution.
//
// RmsNormSite's own wide-row construction
// (`FloorDivI64(h[i]<<2*NORM_FRAC_BITS, root) * g[i]`) is never exposed by its
// public signature -- only out_codes/*out_scale escape, and both are proven
// invariant to the primitive's own correctness (see the correction below).
// This cell recovers the wide row from the emitted trace record instead
// (S3.1a, trace_hook.h; D-SLM362, wired into RmsNormSite/EmbedEntry per
// Claude/Brunel/superslm-s3.2-forward-move-and-trace-emission-build-2026-07-28.md
// Job 2): RmsNormSite forwards its trailing site/token_index/trace_hook_state
// parameters, unchanged, into its own internal RequantChainChecked call, and
// that call's own emission block (checked_chain_funnel.cpp:205,
// `record.x_int = std::span<const int64_t>(wide_row, n)`) builds the trace
// record's `x_int` from the SAME `wide_row` pointer RmsNormSite passed in --
// full int64 precision, strictly after every write RequantChainChecked itself
// performs (checked_chain_funnel.h's own documented emission-ordering
// contract) and never altering it. This is the real, library-linked
// ::superslm::RmsNormSite -- no shadow recompile of production source, no
// spy substituted for RequantChainChecked. The hook installed below only
// observes what the site already computed; `x_int` is copied into owned
// storage before the call returns, since the span is a non-owning view valid
// only for the duration of the hook call (trace_hook.h's own documented
// lifetime).
namespace test_t1267_trace_capture {

struct CapturedRecord {
	std::vector<int64_t> x_int;
	bool called = false;
};
inline CapturedRecord g_captured;

// Matches SslmTraceHookFn exactly (trace_hook.h). Exactly one of the two
// record pointers is non-null per call -- for RmsNormSite's own chain-funnel
// emission, the chain record, with the K/V-landing pointer null
// (trace_hook.h's own per-call contract) -- and this pin only reads that one.
inline void CaptureChainTrace(const superslm::SslmChainTraceRecord* chain,
                               const superslm::SslmKvLandingTraceRecord* /*kv*/, void* user) {
	(void)user;
	if (chain != nullptr) {
		g_captured.x_int.assign(chain->x_int.begin(), chain->x_int.end());
		g_captured.called = true;
	}
}

}  // namespace test_t1267_trace_capture

// Sec4.2's mechanism residual (D-SLM360, filed T-1267): RmsNormSite's own use
// of FloorDivI64, pinned at the wide-row level (before quantization) via the
// emitted trace record above, against the SAME kC31SiteElements witness the
// next test uses -- its `floor_wide` values are independently derived
// (test-design record Sec3.2/Sec6: computed by executing the vendored Python
// reference, never by calling FloorDivI64 itself), so a mutated FloorDivI64
// cannot drag both sides of this comparison the same way it drags a
// comparison that calls FloorDivI64 on both ends.
static void TestRmsNormSiteC31UsesFloorDivisionMechanismPin() {
	using superslm::CarriedScale;
	using superslm::SslmForwardStatus;
	using superslm::SslmTraceHookState;

	std::vector<int8_t> h(kC31SiteHiddenSize, 0);
	std::vector<int32_t> g(kC31SiteHiddenSize, 0);
	for (size_t i = 0; i < kC31SiteElementsCount; ++i) {
		const C31SiteElement& e = kC31SiteElements[i];
		h[static_cast<size_t>(e.index)] = static_cast<int8_t>(e.h);
		g[static_cast<size_t>(e.index)] = e.g;
	}

	const CarriedScale site_constant{/*m=*/INT64_C(1073741824), /*e=*/0};
	const CarriedScale incoming_scale{/*m=*/INT64_C(1073741824), /*e=*/0};

	test_t1267_trace_capture::g_captured = test_t1267_trace_capture::CapturedRecord{};
	SslmTraceHookState trace_hook_state{};
	superslm::SslmSetTraceHook(trace_hook_state, &test_t1267_trace_capture::CaptureChainTrace,
	                            /*user=*/nullptr);

	std::vector<int8_t> out_codes(kC31SiteHiddenSize, INT8_C(-99));  // poison
	CarriedScale out_scale{INT64_C(-99), INT64_C(-99)};              // poison
	auto result = superslm::RmsNormSite(h.data(), g.data(), kC31SiteHiddenSize, incoming_scale,
	                                      site_constant, out_codes.data(), &out_scale,
	                                      /*site=*/"t1267.rmsnorm_c31", /*token_index=*/0,
	                                      &trace_hook_state);
	CHECK_MSG(result == SslmForwardStatus::Ok,
	          "RmsNormSite (T-1267, traced) status == %s, want Ok",
	          superslm::SslmForwardStatusName(result));
	CHECK_MSG(test_t1267_trace_capture::g_captured.called,
	          "RmsNormSite must reach RequantChainChecked exactly once per call and emit a "
	          "chain trace record through the installed hook -- the trace-record capture "
	          "(T-1267) was never invoked");
	if (result == SslmForwardStatus::Ok && test_t1267_trace_capture::g_captured.called) {
		const std::vector<int64_t>& wide = test_t1267_trace_capture::g_captured.x_int;
		CHECK_MSG(wide.size() == static_cast<size_t>(kC31SiteHiddenSize),
		          "captured trace record's x_int size == %zu, want %d", wide.size(),
		          kC31SiteHiddenSize);
		if (wide.size() == static_cast<size_t>(kC31SiteHiddenSize)) {
			for (size_t i = 0; i < kC31SiteElementsCount; ++i) {
				const C31SiteElement& e = kC31SiteElements[i];
				const size_t idx = static_cast<size_t>(e.index);
				CHECK_MSG(wide[idx] == e.floor_wide,
				          "RmsNormSite's own internal wide_row[%d], read from the emitted "
				          "trace record's x_int, == %lld, want %lld (the independently-"
				          "derived floor-based value, diverges=%s -- T-1267's mechanism pin "
				          "on the site's actual USE of FloorDivI64, checked before "
				          "RequantChainChecked's own quantization ever runs)",
				          e.index, static_cast<long long>(wide[idx]), e.floor_wide,
				          e.diverges ? "true" : "false");
			}
		}
	}
}

// Sec4.2 (C31, Sec5.1): the RMSNorm site's own composition, against a
// reachable (h, root) witness at H=1536. The oracle is the ALREADY-SHIPPED
// funnel (RequantChainChecked, S3.1), fed the floor-based wide row Curie's
// record derived and mutation-verified at the WIDE level (test-design record
// Sec3.2/Sec6) -- never a recount of RmsNormSite's own internal steps.
//
// CORRECTED against this pass's own test-design record Sec4.2, by execution
// (StandardsDocument Sec5.6: "a ruling contradicted by a measurement is
// re-opened, not defended"): Sec4.2's own text additionally promised that
// out_codes would DIVERGE from a truncating implementation's codes at indices
// 0 and 17. Checked by running both the vendored Python reference and this
// exact C++ funnel over the floor-based and truncation-based rows: they agree
// at every index every time (empirically, 500,000 randomized (h, g, root)
// trials found zero cases where a non-dominant element's floor/trunc
// divergence survived into a different requantized code). This is not
// implementation flakiness; it is a closed-form bound on this composition's
// own numeric ranges -- floor and trunc quotients differ by exactly 1 unit
// when they diverge, so a non-dominant element's code can only shift if its
// own quotient magnitude is under roughly 254 (half the ~D'/127 quantization
// step, expressed as a ratio, which cancels the row's actual scale); the
// smallest quotient magnitude reachable by ANY int8 h at this composition's
// own root ceiling (H=1536, all elements at |h|=127, root=8,323,072) is 516,
// already above that threshold. Floor-vs-truncation discrimination for C31 is
// therefore fully and only provable at the unit-cell level (Sec4.1, above,
// which retains full discriminating power on FloorDivI64 directly) and at the
// wide-value level (kC31SiteElements' own `diverges` field, already
// mutation-verified in gen_s3_2_fixtures.py) -- not restatable as an
// out_codes-level claim at this site's own output resolution. What THIS cell
// still proves, and correctly: the site's whole composition (sumsq -> ISqrt
// o FloorDivI64 -> max(root,1) -> per-element FloorDivI64*g[i] -> the funnel)
// produces exactly the codes the already-shipped funnel computes from the
// correct, floor-based wide row -- a real feature oracle on the composition's
// wiring, which a wrong hidden_size, a wrong shift, a wrong per-element
// formula, or a dropped max(root,1) guard would all still fail.
static void TestRmsNormSiteC31FloorDivisionWitnessAgainstTheRealFunnel() {
	using superslm::CarriedScale;
	using superslm::SslmForwardStatus;

	std::vector<int8_t> h(kC31SiteHiddenSize, 0);
	std::vector<int32_t> g(kC31SiteHiddenSize, 0);
	for (size_t i = 0; i < kC31SiteElementsCount; ++i) {
		const C31SiteElement& e = kC31SiteElements[i];
		h[static_cast<size_t>(e.index)] = static_cast<int8_t>(e.h);
		g[static_cast<size_t>(e.index)] = e.g;
	}

	const CarriedScale site_constant{/*m=*/INT64_C(1073741824), /*e=*/0};
	const CarriedScale incoming_scale{/*m=*/INT64_C(1073741824), /*e=*/0};

	std::vector<int64_t> floor_wide_row(kC31SiteHiddenSize, 0);
	for (size_t i = 0; i < kC31SiteElementsCount; ++i) {
		const C31SiteElement& e = kC31SiteElements[i];
		floor_wide_row[static_cast<size_t>(e.index)] = e.floor_wide;
	}

	std::vector<int8_t> expected_codes(kC31SiteHiddenSize, 0);
	CarriedScale expected_scale{};
	auto expected_result = superslm::RequantChainChecked(
	    floor_wide_row.data(), kC31SiteHiddenSize, std::span<const CarriedScale>{}, site_constant,
	    expected_codes.data(), &expected_scale);
	CHECK_MSG(expected_result.status == SslmForwardStatus::Ok,
	          "the floor-based oracle row itself must be accepted by the already-shipped "
	          "funnel: got %s", superslm::SslmForwardStatusName(expected_result.status));

	std::vector<int8_t> out_codes(kC31SiteHiddenSize, INT8_C(-99));  // poison
	CarriedScale out_scale{INT64_C(-99), INT64_C(-99)};              // poison
	auto result = superslm::RmsNormSite(h.data(), g.data(), kC31SiteHiddenSize, incoming_scale,
	                                      site_constant, out_codes.data(), &out_scale);
	CHECK_MSG(result == SslmForwardStatus::Ok,
	          "RmsNormSite(C31 site witness) status == %s, want Ok (red-unimplemented until "
	          "Brunel's green phase)",
	          superslm::SslmForwardStatusName(result));
	if (result == SslmForwardStatus::Ok) {
		for (size_t i = 0; i < kC31SiteElementsCount; ++i) {
			const C31SiteElement& e = kC31SiteElements[i];
			const size_t idx = static_cast<size_t>(e.index);
			CHECK_MSG(out_codes[idx] == expected_codes[idx],
			          "RmsNormSite out_codes[%d] == %d, want %d (the floor-division oracle's "
			          "own code, via the already-shipped funnel)",
			          e.index, static_cast<int>(out_codes[idx]), static_cast<int>(expected_codes[idx]));
		}
	}
}

// Sec4.3 (C24/C25, Sec11 S3.2's own two-element-row cell): the WSC1
// identity/near-identity fold-apply dispatch, against the real funnel.
static void TestApplyWeightScaleFoldC24IdentityVsNearIdentityAgainstTheRealFunnel() {
	using superslm::CarriedScale;
	using superslm::SslmForwardStatus;

	// Mutation-verification: MultiplyByQuantizedMultiplier is ALREADY SHIPPED
	// (S2.1) -- confirm by direct execution that it reproduces the plan's own
	// "off by one" near-identity fold on the reference channel's raw value,
	// rather than assuming the Python-derived fixture's claim survives into the
	// real C++ kernel unchanged.
	const int32_t near_identity_direct = superslm::MultiplyByQuantizedMultiplier(
	    static_cast<int32_t>(kC24RefChannelPassThrough), /*quantized_multiplier=*/INT32_MAX,
	    /*shift=*/0);
	CHECK_MSG(near_identity_direct == static_cast<int32_t>(kC24RefChannelNearIdentity),
	          "MultiplyByQuantizedMultiplier(%lld, INT32_MAX, 0) == %d, want %lld (the plan's "
	          "own 'off by one' near-identity fold, S2.1's already-shipped kernel)",
	          static_cast<long long>(kC24RefChannelPassThrough), near_identity_direct,
	          static_cast<long long>(kC24RefChannelNearIdentity));

	const CarriedScale site_constant{/*m=*/INT64_C(1073741824), /*e=*/0};

	// Correct: the reference channel's true pass-through (identity=1).
	const int64_t pass_through_ref = superslm::ApplyWeightScaleFold(
	    kC24RefChannelPassThrough, /*identity=*/1, /*mult=*/0, /*shift=*/0);
	CHECK_MSG(pass_through_ref == kC24RefChannelPassThrough,
	          "ApplyWeightScaleFold(identity=1) == %lld, want %lld unchanged (true pass-through)",
	          static_cast<long long>(pass_through_ref), static_cast<long long>(kC24RefChannelPassThrough));

	int64_t pass_row[2] = {pass_through_ref, kC24SharedElementX};
	int8_t pass_codes[2] = {INT8_C(-99), INT8_C(-99)};
	CarriedScale pass_scale{INT64_C(-99), INT64_C(-99)};
	auto pass_result = superslm::RequantChainChecked(pass_row, 2, std::span<const CarriedScale>{},
	                                                    site_constant, pass_codes, &pass_scale);
	CHECK_MSG(pass_result.status == SslmForwardStatus::Ok,
	          "pass-through row through the real funnel: status == %s, want Ok",
	          superslm::SslmForwardStatusName(pass_result.status));
	CHECK_MSG(pass_codes[0] == kC24CodeRefPassThrough && pass_codes[1] == kC24CodeXPassThrough,
	          "pass-through row codes == {%d, %d}, want {%d, %d} (the plan's own stated row)",
	          static_cast<int>(pass_codes[0]), static_cast<int>(pass_codes[1]), kC24CodeRefPassThrough,
	          kC24CodeXPassThrough);

	// The cell's whole point: the same row with the reference channel WRONGLY
	// dispatched as identity=0 (the near-identity fold applied where a true
	// pass-through is owed). ApplyWeightScaleFold is exercised with these real
	// (wrong-for-this-row) arguments -- not a fake stand-in -- to prove the
	// one-unit difference this produces changes the OTHER element's own
	// requantized code through the shared D', even though X's own raw value
	// never changed between the two runs (test-design record Sec3.3).
	const int64_t near_identity_ref = superslm::ApplyWeightScaleFold(
	    kC24RefChannelPassThrough, /*identity=*/0, /*mult=*/INT32_MAX, /*shift=*/0);
	CHECK_MSG(near_identity_ref == kC24RefChannelNearIdentity,
	          "ApplyWeightScaleFold(identity=0, mult=2^31-1, shift=0) == %lld, want %lld (the "
	          "near-identity fold wrongly applied to the reference channel)",
	          static_cast<long long>(near_identity_ref), static_cast<long long>(kC24RefChannelNearIdentity));

	int64_t near_identity_row[2] = {near_identity_ref, kC24SharedElementX};
	int8_t near_identity_codes[2] = {INT8_C(-99), INT8_C(-99)};
	CarriedScale near_identity_scale{INT64_C(-99), INT64_C(-99)};
	auto near_identity_result = superslm::RequantChainChecked(
	    near_identity_row, 2, std::span<const CarriedScale>{}, site_constant, near_identity_codes,
	    &near_identity_scale);
	CHECK_MSG(near_identity_result.status == SslmForwardStatus::Ok,
	          "near-identity row through the real funnel: status == %s, want Ok",
	          superslm::SslmForwardStatusName(near_identity_result.status));
	CHECK_MSG(near_identity_codes[0] == kC24CodeRefNearIdentity &&
	              near_identity_codes[1] == kC24CodeXNearIdentity,
	          "near-identity row codes == {%d, %d}, want {%d, %d} -- the SECOND element's code "
	          "must shift even though its own raw value (kC24SharedElementX) never changed (the "
	          "divergence is token-wide through the shared D', SuperSLM_Plan.md:2194-2197)",
	          static_cast<int>(near_identity_codes[0]), static_cast<int>(near_identity_codes[1]),
	          kC24CodeRefNearIdentity, kC24CodeXNearIdentity);

	CHECK_MSG(pass_codes[1] != near_identity_codes[1],
	          "the shared element X's code must differ between the pass-through and "
	          "near-identity runs (%d vs %d) -- otherwise this cell has no discriminating power "
	          "on the exact claim it exists to prove",
	          static_cast<int>(pass_codes[1]), static_cast<int>(near_identity_codes[1]));
}

// Sec4.4 (C28, Sec7.2 second limb, Sec4.4): the (q_B, e_a) domain-boundary
// half. All four boundary points at q_B=30 (kC28TieQB, F-S3-4's own verified
// constant), derived from k = q_B + 62 + e_a rather than guessed.
static void TestCheckRoundingDivideByPotExponentDomainC28BoundaryMatrix() {
	using superslm::SslmForwardStatus;
	struct Point {
		int64_t e_a;
		bool in_domain;
		const char* label;
	};
	const Point points[] = {
	    {kC28BoundaryBelowMinEA, kC28BoundaryBelowMinInDomain, "below_min (k=-1)"},
	    {kC28BoundaryAtMinEA, kC28BoundaryAtMinInDomain, "at_min (k=0)"},
	    {kC28BoundaryAtMaxEA, kC28BoundaryAtMaxInDomain, "at_max (k=63)"},
	    {kC28BoundaryAboveMaxEA, kC28BoundaryAboveMaxInDomain, "above_max (k=64)"},
	};
	for (const Point& p : points) {
		auto status = superslm::CheckRoundingDivideByPotExponentDomain(/*q_B=*/kC28TieQB, p.e_a);
		if (p.in_domain) {
			CHECK_MSG(status == SslmForwardStatus::Ok,
			          "%s: CheckRoundingDivideByPotExponentDomain(q_B=30, e_a=%lld) status == "
			          "%s, want Ok",
			          p.label, static_cast<long long>(p.e_a), superslm::SslmForwardStatusName(status));
		} else {
			CHECK_MSG(status == SslmForwardStatus::RoundingDivideByPotExponentOutOfDomain,
			          "%s: CheckRoundingDivideByPotExponentDomain(q_B=30, e_a=%lld) status == "
			          "%s, want RoundingDivideByPotExponentOutOfDomain",
			          p.label, static_cast<long long>(p.e_a), superslm::SslmForwardStatusName(status));
		}
	}
}

// Sec4.6 (C28, Sec4.4): the bias-reconciliation compute's sign-inverted
// negative control -- away-from-zero (C3, the correct rule, load-bearing
// because B is signed) vs. round-half-up (C2, the wrong rule, used elsewhere
// in this same composition) agree on the positive tie and disagree only on
// the negative one. See the test-design record Sec5 for why this is the one
// reading "C28's sign-inverted negative control" supports.
static void TestBiasReconcileC28SignInvertedNegativeControl() {
	// Pin the fixture's own claim (a comment naming a mechanism is a claim
	// mutation-verified before filing): the wrong (round-half-up) candidate
	// agrees with the correct (away-from-zero) result on the positive tie and
	// disagrees on the negative one -- verified by execution in
	// gen_s3_2_fixtures.py's own mutation log (test-design record Sec6,
	// mutation 3). Pinned again here at compile time so a hand-edit to the
	// generated header cannot silently drift this claim.
	static_assert(kC28TieWrongPos == kC28TieCorrectPos,
	              "the fixture's own negative-control premise (agreement on the positive tie) no "
	              "longer holds");
	static_assert(kC28TieWrongNeg != kC28TieCorrectNeg,
	              "the fixture's own negative-control premise (disagreement on the negative tie) "
	              "no longer holds -- the control would have no discriminating power");

	const int64_t pos = superslm::BiasReconcile(kC28TieB, kC28TieQB, kC28TieRA, kC28TieEA);
	CHECK_MSG(pos == kC28TieCorrectPos, "BiasReconcile(+B) == %lld, want %lld (C3, away-from-zero)",
	          static_cast<long long>(pos), static_cast<long long>(kC28TieCorrectPos));

	const int64_t neg = superslm::BiasReconcile(-kC28TieB, kC28TieQB, kC28TieRA, kC28TieEA);
	CHECK_MSG(neg == kC28TieCorrectNeg,
	          "BiasReconcile(-B) == %lld, want %lld (C3, away-from-zero) -- a wrong "
	          "round-half-up implementation would return %lld here instead, agreeing with the "
	          "correct result only on +B",
	          static_cast<long long>(neg), static_cast<long long>(kC28TieCorrectNeg),
	          static_cast<long long>(kC28TieWrongNeg));
}

// Sec4.7/Sec4.8 (BIA1, Sec7.2a third limb): the load-time magnitude descriptor.
// Builds a single-tensor BIA1 (int64) manifest with one element set to the
// value under test -- the same "one otherwise-valid v2 artifact, one hostile
// section" pattern as TestKvc1RejectsHostileCompositionConstantsScale et al.
static FixtureSection MakeBia1Section(int64_t value) {
	auto manifest = BuildManifest(superslm::kBiasesMagic, /*element_size=*/8, {{"b0", {1}}});
	PutU64(manifest.bytes, static_cast<size_t>(manifest.tensor_data_off[0]), static_cast<uint64_t>(value));
	return MakeSection(SslmSectionType::Biases, SslmDtype::Int64, manifest.bytes, /*alignment=*/64);
}

static void TestBia1RejectsHostileMagnitudeBothSignsAndAcceptsTheBoundary() {
	// Reject: one past the derived bound, both signs (Sec3.4/Sec4.7 -- the
	// domain is symmetric, so both signs are named rather than assumed to fail
	// identically).
	{
		auto built = BuildArtifact(
		    {MakeValidConfigSection(), MakeSigmoidLutSection(), MakeBia1Section(kBia1HostileValue)});
		SslmModelView view;
		std::string err;
		SslmModelStatus status = SslmModel::Load(built.bytes.data(), built.bytes.size(), view, &err);
		CHECK_MSG(status == SslmModelStatus::BiasCodeOutOfDomain,
		          "BIA1 B[0]=2^31 (one past INT32_MAX): SslmModel::Load status == %s, want "
		          "BiasCodeOutOfDomain (%s)",
		          SslmModelStatusName(status), err.c_str());
		CHECK_MSG(!view.has_biases,
		          "hostile Biases view exposed on a rejected Load — a view MUST NOT be exposed "
		          "(Sec4.8's vitality proof: a view never exposed cannot be read by any "
		          "downstream C28 site, so B[j] categorically cannot reach B[j]*R_a)");
	}
	{
		auto built = BuildArtifact({MakeValidConfigSection(), MakeSigmoidLutSection(),
		                            MakeBia1Section(kBia1HostileValueNegated)});
		SslmModelView view;
		std::string err;
		SslmModelStatus status = SslmModel::Load(built.bytes.data(), built.bytes.size(), view, &err);
		CHECK_MSG(status == SslmModelStatus::BiasCodeOutOfDomain,
		          "BIA1 B[0]=-2^31 (one past the bound, negative side): SslmModel::Load status "
		          "== %s, want BiasCodeOutOfDomain (%s)",
		          SslmModelStatusName(status), err.c_str());
		CHECK_MSG(!view.has_biases,
		          "hostile Biases view exposed on a rejected Load — a view MUST NOT be exposed");
	}
	// Accept-at-bound: the off-by-one control every S-HARDEN-1 boundary matrix
	// carries (Sec13 dim 4).
	{
		auto built = BuildArtifact({MakeValidConfigSection(), MakeSigmoidLutSection(),
		                            MakeBia1Section(kBia1AcceptBoundaryValue)});
		SslmModelView view;
		std::string err;
		SslmModelStatus status = SslmModel::Load(built.bytes.data(), built.bytes.size(), view, &err);
		CHECK_MSG(status == SslmModelStatus::Ok,
		          "BIA1 B[0]==INT32_MAX (exactly at the bound): SslmModel::Load status == %s, "
		          "want Ok (%s)",
		          SslmModelStatusName(status), err.c_str());
		CHECK_MSG(view.has_biases, "an in-domain Biases view must be exposed on a successful Load");
	}
}

// Sec4.9 (F-S3-8, Sec4.8, Sec13 dim 2): the embed entry's token-id validation.
static void TestEmbedEntryRejectsHostileTokenIdBeforeAnyReadAndAcceptsTheBoundary() {
	using superslm::CarriedScale;
	using superslm::SslmForwardStatus;

	constexpr int32_t kVocabSize = 8;
	constexpr size_t kHiddenSize = 4;
	const CarriedScale site_constant{/*m=*/INT64_C(1073741824), /*e=*/0};

	// Hostile ids: embed_weights is passed as nullptr. If EmbedEntry validates
	// token_id BEFORE reading any row (the contract's own stated order,
	// forward_sites.h), a null embed_weights is never dereferenced and the call
	// returns cleanly; an implementation that reads first would dereference a
	// pointer computed from a null base and a hostile id and crash -- a real,
	// unambiguous vitality proof for a `const` input buffer, in the same
	// non-crash idiom this suite already uses
	// (TestNarrowRowCheckedZeroLenNullPtrDoesNotCrash) rather than a
	// poison-fill, which cannot distinguish "read" from "not read" on a buffer
	// the callee never writes to.
	const int32_t hostile_ids[] = {-1, kVocabSize, kVocabSize + 1};
	const char* labels[] = {"token_id=-1", "token_id==vocab_size", "token_id==vocab_size+1"};
	for (size_t i = 0; i < 3; ++i) {
		int8_t out_codes[kHiddenSize] = {INT8_C(-99), INT8_C(-99), INT8_C(-99), INT8_C(-99)};
		CarriedScale out_scale{INT64_C(-99), INT64_C(-99)};
		auto status = superslm::EmbedEntry(hostile_ids[i], kVocabSize, /*embed_weights=*/nullptr,
		                                     kHiddenSize, site_constant, out_codes, &out_scale);
		CHECK_MSG(status == SslmForwardStatus::TokenIdOutOfRange,
		          "%s: EmbedEntry status == %s, want TokenIdOutOfRange (embed_weights == "
		          "nullptr -- a validate-before-read implementation never dereferences it)",
		          labels[i], superslm::SslmForwardStatusName(status));
		CHECK_MSG(out_codes[0] == INT8_C(-99) && out_scale.m == INT64_C(-99) && out_scale.e == INT64_C(-99),
		          "%s: out_codes/*out_scale must be untouched on rejection (\"computes nothing\")",
		          labels[i]);
	}

	// Accept boundary: token_id == vocab_size - 1 (the last valid id) must
	// proceed to read the embedding row and produce the funnel's own codes --
	// an independent feature oracle against the ALREADY-SHIPPED funnel, over a
	// row widened from real embed_weights bytes, never a recount of
	// EmbedEntry's own steps.
	const int8_t embed_row[kHiddenSize] = {5, -3, 127, -100};
	std::vector<int8_t> embed_weights(static_cast<size_t>(kVocabSize) * kHiddenSize, 0);
	const int32_t accept_id = kVocabSize - 1;
	std::memcpy(embed_weights.data() + static_cast<size_t>(accept_id) * kHiddenSize, embed_row,
	            kHiddenSize);

	int64_t expected_wide[kHiddenSize];
	for (size_t j = 0; j < kHiddenSize; ++j) expected_wide[j] = static_cast<int64_t>(embed_row[j]);
	int8_t expected_codes[kHiddenSize] = {0, 0, 0, 0};
	CarriedScale expected_scale{};
	auto expected_result = superslm::RequantChainChecked(expected_wide, kHiddenSize,
	                                                        std::span<const CarriedScale>{}, site_constant,
	                                                        expected_codes, &expected_scale);
	CHECK_MSG(expected_result.status == SslmForwardStatus::Ok,
	          "the accept-boundary oracle row itself must be accepted by the already-shipped "
	          "funnel: got %s",
	          superslm::SslmForwardStatusName(expected_result.status));

	int8_t out_codes[kHiddenSize] = {INT8_C(-99), INT8_C(-99), INT8_C(-99), INT8_C(-99)};
	CarriedScale out_scale{INT64_C(-99), INT64_C(-99)};
	auto status = superslm::EmbedEntry(accept_id, kVocabSize, embed_weights.data(), kHiddenSize,
	                                     site_constant, out_codes, &out_scale);
	CHECK_MSG(status == SslmForwardStatus::Ok,
	          "token_id==vocab_size-1 (last valid id): EmbedEntry status == %s, want Ok "
	          "(red-unimplemented until Brunel's green phase)",
	          superslm::SslmForwardStatusName(status));
	if (status == SslmForwardStatus::Ok) {
		for (size_t j = 0; j < kHiddenSize; ++j) {
			CHECK_MSG(out_codes[j] == expected_codes[j],
			          "EmbedEntry out_codes[%zu] == %d, want %d (the already-shipped funnel's "
			          "own code for this row)",
			          j, static_cast<int>(out_codes[j]), static_cast<int>(expected_codes[j]));
		}
	}
}

// Sec4.10 (Sec11 S3.2's own gate line; Coverage Model dim 7): "the norm's
// carried scale is gain-derived and an implementation forwarding the incoming
// scale fails." A differential cell against the site's own two invocations --
// identical hidden row, gain row, and site constant, two different incoming
// carried scales -- no separate reference oracle is owed beyond the site's
// own gain-derived formula (C23), which is already pinned prose, not a
// quantity this pass derives.
static void TestRmsNormSiteCarriedScaleIsGainDerivedNotIncomingScale() {
	using superslm::CarriedScale;
	using superslm::SslmForwardStatus;

	std::vector<int8_t> h(kC31SiteHiddenSize, 0);
	std::vector<int32_t> g(kC31SiteHiddenSize, 0);
	for (size_t i = 0; i < kC31SiteElementsCount; ++i) {
		const C31SiteElement& e = kC31SiteElements[i];
		h[static_cast<size_t>(e.index)] = static_cast<int8_t>(e.h);
		g[static_cast<size_t>(e.index)] = e.g;
	}
	const CarriedScale site_constant{/*m=*/INT64_C(1073741824), /*e=*/0};
	const CarriedScale incoming_a{/*m=*/INT64_C(1073741824), /*e=*/0};
	const CarriedScale incoming_b{/*m=*/INT64_C(2000000000), /*e=*/-7};

	std::vector<int8_t> codes_a(kC31SiteHiddenSize, INT8_C(-99));
	CarriedScale scale_a{INT64_C(-99), INT64_C(-99)};
	auto result_a = superslm::RmsNormSite(h.data(), g.data(), kC31SiteHiddenSize, incoming_a,
	                                        site_constant, codes_a.data(), &scale_a);
	CHECK_MSG(result_a == SslmForwardStatus::Ok, "RmsNormSite (incoming scale A) status == %s, want Ok",
	          superslm::SslmForwardStatusName(result_a));

	std::vector<int8_t> codes_b(kC31SiteHiddenSize, INT8_C(-99));
	CarriedScale scale_b{INT64_C(-99), INT64_C(-99)};
	auto result_b = superslm::RmsNormSite(h.data(), g.data(), kC31SiteHiddenSize, incoming_b,
	                                        site_constant, codes_b.data(), &scale_b);
	CHECK_MSG(result_b == SslmForwardStatus::Ok, "RmsNormSite (incoming scale B) status == %s, want Ok",
	          superslm::SslmForwardStatusName(result_b));

	if (result_a == SslmForwardStatus::Ok && result_b == SslmForwardStatus::Ok) {
		for (size_t i = 0; i < kC31SiteHiddenSize; ++i) {
			CHECK_MSG(codes_a[i] == codes_b[i],
			          "out_codes[%zu] must be identical across two incoming carried scales (%d "
			          "vs %d) -- the incoming scale must be annihilated, never folded in",
			          i, static_cast<int>(codes_a[i]), static_cast<int>(codes_b[i]));
		}
		CHECK_MSG(scale_a.m == scale_b.m && scale_a.e == scale_b.e,
		          "*out_scale must be identical across two incoming carried scales -- got "
		          "{%lld,%lld} vs {%lld,%lld}",
		          static_cast<long long>(scale_a.m), static_cast<long long>(scale_a.e),
		          static_cast<long long>(scale_b.m), static_cast<long long>(scale_b.e));
	}
}

// ---------------------------------------------------------------------------
// SuperSLM_S3a_WalkingSkeleton_Plan.md Sec13.1 cells 2 and 3 (D-SLM417, board
// T-1336): two cross-component join cells the dimension-ownership sweep found
// enumerated with no owning sub-slot, assigned to Sec11 S3.2 (already the
// first consumer of both sections) but never authored while that sub-slot's
// own header contract landed -- owed against S3.2's already-green production
// code (Claude/Curie/superslm-s3.2-weightless-and-projection-sites-test-
// design-2026-07-28.md; Claude/Brunel/superslm-s3.2-weightless-and-
// projection-sites-green-build-2026-07-28.md).
// ---------------------------------------------------------------------------

// Cell 2: converter -> forward, independently parsed. A minimal standalone
// reader, written directly from docs/sslm_format.md's "Tensor-manifest blob"
// and "Keyed numeric-constant blob -- KVC1" field tables, sharing no code
// with SslmTensorManifest::Parse, SslmKeyedConstants::Parse, or
// SslmModel::Load -- the correlated-oracle risk F-S3-3 names for the context
// fold, applying identically here (Sec11 S3.2's own text). Reads neither
// through the C++ loader nor through the converter's/loader's own shared
// section-writer/reader helpers (BuildManifest/BuildKvc1 below are Curie's
// own from-spec fixture WRITERS, already independent of src/model.cpp per
// their own file headers, but they do not parse bytes back -- the read side
// below is new).
namespace independent_reader {

inline uint32_t ReadU32(const uint8_t* p) {
	return static_cast<uint32_t>(p[0]) | (static_cast<uint32_t>(p[1]) << 8) |
	       (static_cast<uint32_t>(p[2]) << 16) | (static_cast<uint32_t>(p[3]) << 24);
}
inline uint64_t ReadU64(const uint8_t* p) {
	const uint64_t lo = ReadU32(p);
	const uint64_t hi = ReadU32(p + 4);
	return lo | (hi << 32);
}

// Locates a section by SslmSectionType within the artifact's own 64-byte
// header and section table (docs/sslm_format.md "Byte layout"), independent
// of SslmArtifact::OpenFromMemory.
struct FoundSection {
	uint64_t offset = 0;
	uint64_t byte_size = 0;
	bool found = false;
};

inline FoundSection FindSectionByType(const uint8_t* file, size_t file_len, uint32_t want_type) {
	FoundSection r;
	if (file_len < 64) return r;
	const uint32_t section_count = ReadU32(file + 12);
	for (uint32_t i = 0; i < section_count; ++i) {
		const size_t row_off = 64 + static_cast<size_t>(i) * 40;
		if (row_off + 40 > file_len) break;
		const uint8_t* row = file + row_off;
		if (ReadU32(row + 0) == want_type) {
			r.offset = ReadU64(row + 8);
			r.byte_size = ReadU64(row + 16);
			r.found = true;
			return r;
		}
	}
	return r;
}

// Independently parses one named int64 tensor out of a WGT1/BIA1/ROP1-shaped
// manifest section (docs/sslm_format.md "Tensor-manifest blob": 16-byte
// header, 48-byte descriptors at offset 16, then the name blob, then the
// data region). `*ok` is false and the return value empty on any structural
// problem -- the negative-control path a hostile/corrupted tensor takes
// (this cell's own negative control), never a crash and never a silent wrong
// answer.
inline std::vector<int64_t> ReadInt64TensorByName(const uint8_t* section, uint64_t section_len,
                                                   const uint8_t expected_magic[4],
                                                   std::string_view tensor_name, bool* ok) {
	*ok = false;
	std::vector<int64_t> out;
	if (section_len < 16) return out;
	if (std::memcmp(section, expected_magic, 4) != 0) return out;
	const uint32_t version = ReadU32(section + 4);
	if (version != 1) return out;
	const uint32_t tensor_count = ReadU32(section + 8);
	const uint32_t name_blob_len = ReadU32(section + 12);
	const uint64_t desc_off = 16;
	const uint64_t name_blob_off = desc_off + static_cast<uint64_t>(tensor_count) * 48;
	if (name_blob_off + name_blob_len > section_len) return out;
	for (uint32_t i = 0; i < tensor_count; ++i) {
		const uint8_t* d = section + desc_off + static_cast<uint64_t>(i) * 48;
		const uint32_t name_off = ReadU32(d + 0);
		const uint32_t name_len = ReadU32(d + 4);
		const uint64_t data_off = ReadU64(d + 28);
		const uint64_t elem_count = ReadU64(d + 36);
		if (static_cast<uint64_t>(name_off) + name_len > name_blob_len) continue;
		const std::string_view name(reinterpret_cast<const char*>(section + name_blob_off + name_off),
		                             name_len);
		if (name != tensor_name) continue;
		if (data_off < name_blob_off + name_blob_len) return out;  // overlaps the name blob -- reject
		if (elem_count > (UINT64_MAX / 8)) return out;             // would overflow the bounds check
		if (data_off + elem_count * 8 > section_len) return out;   // out of bounds -- reject
		out.resize(static_cast<size_t>(elem_count));
		for (uint64_t e = 0; e < elem_count; ++e) {
			out[static_cast<size_t>(e)] = static_cast<int64_t>(ReadU64(section + data_off + e * 8));
		}
		*ok = true;
		return out;
	}
	return out;  // name not found
}

// Independently parses one named entry's int64 tuple out of a KVC1 keyed-
// constant section (docs/sslm_format.md "Keyed numeric-constant blob": magic,
// version, entry_count, value_words, name_blob_len, reserved, descriptors,
// values, name_blob).
inline std::vector<int64_t> ReadKvc1EntryByName(const uint8_t* section, uint64_t section_len,
                                                 std::string_view key, bool* ok) {
	*ok = false;
	std::vector<int64_t> out;
	if (section_len < 24) return out;
	static constexpr uint8_t kKvc1Magic[4] = {'K', 'V', 'C', '1'};
	if (std::memcmp(section, kKvc1Magic, 4) != 0) return out;
	const uint32_t version = ReadU32(section + 4);
	if (version != 1) return out;
	const uint32_t entry_count = ReadU32(section + 8);
	const uint32_t value_words = ReadU32(section + 12);
	const uint32_t name_blob_len = ReadU32(section + 16);
	if (value_words != 2 && value_words != 3) return out;
	const uint64_t desc_off = 24;
	const uint64_t values_off = desc_off + static_cast<uint64_t>(entry_count) * 8;
	const uint64_t name_blob_off =
	    values_off + static_cast<uint64_t>(entry_count) * static_cast<uint64_t>(value_words) * 8;
	if (name_blob_off + name_blob_len > section_len) return out;
	for (uint32_t i = 0; i < entry_count; ++i) {
		const uint8_t* d = section + desc_off + static_cast<uint64_t>(i) * 8;
		const uint32_t name_off = ReadU32(d + 0);
		const uint32_t name_len = ReadU32(d + 4);
		if (static_cast<uint64_t>(name_off) + name_len > name_blob_len) continue;
		const std::string_view name(reinterpret_cast<const char*>(section + name_blob_off + name_off),
		                             name_len);
		if (name != key) continue;
		out.resize(value_words);
		for (uint32_t w = 0; w < value_words; ++w) {
			const uint64_t off = values_off + (static_cast<uint64_t>(i) * value_words + w) * 8;
			out[w] = static_cast<int64_t>(ReadU64(section + off));
		}
		*ok = true;
		return out;
	}
	return out;  // key not found
}

}  // namespace independent_reader

// Builds a BIA1 tensor and a CompositionConstants/bias.q_b KVC1 entry the way
// tools/convert_model.py emits them (BIA1: a per-layer int64 bias tensor;
// bias.q_b: the (30, 0) shift-count pair, Sec6.8/Sec14.2), loads the result
// through the REAL loader for cross-check, then confirms the independent
// reader above recovers the same values from the same bytes.
static void TestIndependentReaderRecoversBia1AndBiasQbMatchingModelView() {
	auto bia1_manifest = BuildManifest(superslm::kBiasesMagic, /*element_size=*/8, {{"layer0.bias", {2}}});
	const size_t bo = static_cast<size_t>(bia1_manifest.tensor_data_off[0]);
	PutU64(bia1_manifest.bytes, bo + 0, static_cast<uint64_t>(INT64_C(123456789)));
	PutU64(bia1_manifest.bytes, bo + 8, static_cast<uint64_t>(INT64_C(-987654321)));

	auto kvc1 = BuildKvc1(/*declared_value_words=*/2, {{"bias.q_b", {30, 0}}});

	auto built =
	    BuildArtifact({MakeValidConfigSection(), MakeSigmoidLutSection(),
	                   MakeSection(SslmSectionType::Biases, SslmDtype::Int64, bia1_manifest.bytes, 64),
	                   MakeSection(SslmSectionType::CompositionConstants, SslmDtype::Raw, kvc1.bytes, 64)});

	// Ground truth via the real loader -- what this reader's recovered values
	// are checked AGAINST, never the mechanism being verified.
	SslmModelView view;
	std::string err;
	auto status = SslmModel::Load(built.bytes.data(), built.bytes.size(), view, &err);
	CHECK_MSG(status == SslmModelStatus::Ok, "converter-representative fixture failed to load: %s (%s)",
	          SslmModelStatusName(status), err.c_str());
	CHECK(view.has_biases);
	CHECK(view.has_composition_constants);

	// The independent parse.
	auto bia1_section = independent_reader::FindSectionByType(
	    built.bytes.data(), built.bytes.size(), static_cast<uint32_t>(SslmSectionType::Biases));
	CHECK_MSG(bia1_section.found, "independent reader could not locate the Biases section");
	bool bia1_ok = false;
	auto bia1_values =
	    independent_reader::ReadInt64TensorByName(built.bytes.data() + bia1_section.offset, bia1_section.byte_size,
	                                               superslm::kBiasesMagic, "layer0.bias", &bia1_ok);
	CHECK_MSG(bia1_ok, "independent reader failed to parse a structurally valid BIA1 tensor");
	CHECK(bia1_values.size() == 2);
	if (bia1_values.size() == 2) {
		CHECK(bia1_values[0] == INT64_C(123456789));
		CHECK(bia1_values[1] == INT64_C(-987654321));
	}

	auto cc_section = independent_reader::FindSectionByType(
	    built.bytes.data(), built.bytes.size(), static_cast<uint32_t>(SslmSectionType::CompositionConstants));
	CHECK_MSG(cc_section.found, "independent reader could not locate the CompositionConstants section");
	bool qb_ok = false;
	auto qb_values = independent_reader::ReadKvc1EntryByName(built.bytes.data() + cc_section.offset,
	                                                          cc_section.byte_size, "bias.q_b", &qb_ok);
	CHECK_MSG(qb_ok, "independent reader failed to parse bias.q_b out of a structurally valid KVC1 blob");
	CHECK(qb_values.size() == 2);
	if (qb_values.size() == 2) {
		CHECK(qb_values[0] == INT64_C(30));
		CHECK(qb_values[1] == INT64_C(0));
	}

	// Cross-check: what the independent reader recovers matches what
	// SslmModelView (the C++ loader under test) exposes for the SAME bytes.
	const superslm::SslmTensorView* loader_tensor = view.biases.Tensor("layer0.bias");
	CHECK_MSG(loader_tensor != nullptr, "loader's own view does not expose layer0.bias");
	if (loader_tensor != nullptr && bia1_values.size() == loader_tensor->elem_count) {
		for (size_t i = 0; i < bia1_values.size(); ++i) {
			const int64_t loader_value =
			    static_cast<int64_t>(independent_reader::ReadU64(loader_tensor->data + i * 8));
			CHECK_MSG(loader_value == bia1_values[i],
			          "independent reader's BIA1[%zu] == %lld disagrees with SslmModelView's own == %lld", i,
			          static_cast<long long>(bia1_values[i]), static_cast<long long>(loader_value));
		}
	}

	const superslm::SslmConstantEntry* loader_entry = view.composition_constants.Entry("bias.q_b");
	CHECK_MSG(loader_entry != nullptr, "loader's own view does not expose bias.q_b");
	if (loader_entry != nullptr && qb_values.size() == loader_entry->value_words) {
		for (uint32_t w = 0; w < loader_entry->value_words; ++w) {
			const int64_t loader_value = superslm::SslmKeyedConstants::Value(*loader_entry, w);
			CHECK_MSG(loader_value == qb_values[w],
			          "independent reader's bias.q_b[%u] == %lld disagrees with SslmModelView's own == %lld",
			          w, static_cast<long long>(qb_values[w]), static_cast<long long>(loader_value));
		}
	}
}

// Negative control (this cell's own, Sec11 S3.2's text): a hand-corrupted
// BIA1 tensor, run through THIS cell's independent reader alone -- never
// through SslmModel::Load.
static void TestIndependentReaderFlagsHandCorruptedBia1TensorWithoutGoingThroughLoad() {
	auto bia1_manifest = BuildManifest(superslm::kBiasesMagic, /*element_size=*/8, {{"layer0.bias", {2}}});
	// Corrupt the tensor's own data_off descriptor field (descriptor 0 starts
	// at byte 16; data_off is at descriptor offset 28) to point past the end
	// of the section -- the same defect class ManifestOutOfBounds/
	// TensorOutOfBounds name in model.h, caught here by the independent reader
	// alone, not by SslmModel::Load.
	const size_t desc0_data_off_field = 16 + 28;
	const uint64_t corrupted = static_cast<uint64_t>(bia1_manifest.bytes.size()) + 1000;
	PutU64(bia1_manifest.bytes, desc0_data_off_field, corrupted);

	bool ok = false;
	auto values = independent_reader::ReadInt64TensorByName(
	    bia1_manifest.bytes.data(), bia1_manifest.bytes.size(), superslm::kBiasesMagic, "layer0.bias", &ok);
	CHECK_MSG(!ok, "independent reader accepted a BIA1 tensor whose data_off runs past the section");
	CHECK(values.empty());
}

// Cell 3: the tokenizer-drive half of the TOK1 x CFG1 join. The load-time
// rejection half (a mismatched CFG1.vocab_size/TOK1.vocab_count pair,
// TestLoadRejectsTokenizerVocabCountVsConfigVocabSizeMismatch above,
// S-HARDEN-2) is already closed pre-S3a -- this cell owes only the live
// half: on a conformant artifact, every id the tokenizer's OWN Encode() can
// emit is driven through the embed site (EmbedEntry) and asserted to
// address its OWN, correct embedding row -- never TokenIdOutOfRange, and
// matching the already-shipped funnel's own codes for that specific row
// (not merely a non-error status), so a wiring defect that reads the wrong
// row is caught, not only one that rejects outright.
static void TestTokenizerDriveEveryEncodedIdAddressesItsOwnValidEmbeddingRow() {
	using superslm::CarriedScale;
	using superslm::SslmForwardStatus;
	using superslm::TokenizerView;

	auto tok1 = MakeMinimalValidTok1();  // vocab_count == 5
	auto uni1 = MakeMinimalValidUni1();
	auto built = BuildTokenizerArtifact(tok1.bytes, uni1.bytes);

	SslmArtifact artifact;
	SslmError aerr;
	auto astatus = SslmArtifact::OpenFromMemory(built.bytes.data(), built.bytes.size(), artifact, &aerr);
	CHECK_MSG(astatus == SslmStatus::Ok, "conformant tokenizer fixture's outer artifact failed to load: %s: %s",
	          SslmStatusName(astatus), aerr.message.c_str());
	if (astatus != SslmStatus::Ok) return;

	TokenizerView view;
	std::string terr;
	const bool opened = TokenizerView::Open(artifact, view, &terr);
	CHECK_MSG(opened, "TokenizerView::Open failed on the conformant tokenizer fixture: %s", terr.c_str());
	if (!opened) return;

	// A small reference prompt set reaching every declared id: the three base
	// bytes (ids 0,1,2), the one BPE merge (id 3), and the one special token
	// (id 4) -- MakeMinimalValidTok1's own declared vocabulary.
	constexpr int32_t kVocabSize = 5;  // conformant: == TOK1.vocab_count
	const char* prompts[] = {"c", "a", "t", "ca", "<eos>", "cat"};
	bool seen[kVocabSize] = {false, false, false, false, false};
	for (const char* p : prompts) {
		for (int32_t id : view.Encode(p)) {
			if (id >= 0 && id < kVocabSize) seen[id] = true;
		}
	}
	for (int32_t id = 0; id < kVocabSize; ++id) {
		CHECK_MSG(seen[id],
		          "reference prompt set does not reach id %d -- every declared id must be reachable "
		          "for this cell to cover the whole vocabulary",
		          id);
	}

	constexpr size_t kHiddenSize = 3;
	const CarriedScale site_constant{/*m=*/INT64_C(1073741824), /*e=*/0};

	// One distinct embed row per id so a wrong-row read is observable, not
	// only a wrong status -- the achievement oracle this cell exists to
	// provide.
	std::vector<int8_t> embed_weights(static_cast<size_t>(kVocabSize) * kHiddenSize, 0);
	for (int32_t id = 0; id < kVocabSize; ++id) {
		for (size_t j = 0; j < kHiddenSize; ++j) {
			embed_weights[static_cast<size_t>(id) * kHiddenSize + j] =
			    static_cast<int8_t>(10 * (id + 1) + static_cast<int>(j));
		}
	}

	for (int32_t id = 0; id < kVocabSize; ++id) {
		if (!seen[id]) continue;

		int64_t expected_wide[kHiddenSize];
		for (size_t j = 0; j < kHiddenSize; ++j) {
			expected_wide[j] = static_cast<int64_t>(embed_weights[static_cast<size_t>(id) * kHiddenSize + j]);
		}
		int8_t expected_codes[kHiddenSize] = {0, 0, 0};
		CarriedScale expected_scale{};
		auto expected_result = superslm::RequantChainChecked(
		    expected_wide, kHiddenSize, std::span<const CarriedScale>{}, site_constant, expected_codes, &expected_scale);
		CHECK_MSG(expected_result.status == SslmForwardStatus::Ok,
		          "id %d's own oracle row must be accepted by the already-shipped funnel: got %s", id,
		          superslm::SslmForwardStatusName(expected_result.status));

		int8_t out_codes[kHiddenSize] = {INT8_C(-99), INT8_C(-99), INT8_C(-99)};
		CarriedScale out_scale{INT64_C(-99), INT64_C(-99)};
		auto status = superslm::EmbedEntry(id, kVocabSize, embed_weights.data(), kHiddenSize, site_constant,
		                                    out_codes, &out_scale);
		CHECK_MSG(status != SslmForwardStatus::TokenIdOutOfRange,
		          "tokenizer-emitted id %d (from the reference prompt set) was rejected as out of "
		          "range by EmbedEntry against a conformant vocab_size=%d -- the TOK1 x CFG1 join is "
		          "broken",
		          id, kVocabSize);
		CHECK_MSG(status == SslmForwardStatus::Ok, "id %d: EmbedEntry status == %s, want Ok", id,
		          superslm::SslmForwardStatusName(status));
		if (status == SslmForwardStatus::Ok) {
			for (size_t j = 0; j < kHiddenSize; ++j) {
				CHECK_MSG(out_codes[j] == expected_codes[j],
				          "id %d: EmbedEntry out_codes[%zu] == %d, want %d (this id's OWN row -- the "
				          "join's whole point is that every id Encode() emits addresses its own row, "
				          "not another id's)",
				          id, j, static_cast<int>(out_codes[j]), static_cast<int>(expected_codes[j]));
			}
		}
	}
}

// ---------------------------------------------------------------------------
// SuperSLM_S3a_WalkingSkeleton_Plan.md Sec11 S3.3 -- the attention interior
// (C27's K/V landing composite and the D-SLM57 per-head ctx_fold, C32's
// softmax row plus D-SLM366's owed width predicate, C33's post-rotation
// clamp, GemmProbQ15Accumulate, KvLandingScales'/KvLandingReciprocals' owed
// value-domain descriptors, the saturation counter). Per Claude/Curie/
// superslm-s3.3-attention-interior-test-design-2026-07-28.md Sec2: EVERY
// production entry point this sub-slot's own construction needs (the C32
// kernel and its width predicate, the landing-composite site, the saturation
// counter, GemmProbQ15Accumulate, the two new KVC1 value-domain checks, the
// clamp call site itself) is undeclared anywhere under include/ or src/ as of
// this pass -- verified by direct search, not assumed. Every cell that would
// need one of those is therefore fully specified in that record's Sec4
// ("blocked cells"), NOT authored as compiling C++ here: this suite is one
// translation unit, and a reference to an undeclared symbol would fail the
// WHOLE file to compile rather than fail one cell for its own reason, which
// StandardsDocument Sec4/this campaign's own exit condition forbids.
//
// What IS real, compiling, and meaningful today: three already-shipped
// primitives this sub-slot's own witnesses depend on (ApplyWeightScaleFold,
// S3.2; RopeApplyPair and DynamicScaleReciprocal, both pre-S3a) are exercised
// directly against the derived witnesses in sslm_s3_3_fixtures.h, proving
// each witness against the REAL C++ primitive it will be consumed by/compared
// to, not only against the vendored Python mirror the generator calls. This
// is the same "fixture genuinely holds against the real kernel" discipline
// S3.2's own suite used for its native-truncating-divide and
// MultiplyByQuantizedMultiplier mutation checks.
// ---------------------------------------------------------------------------

// Sec4 (record's Sec4.1/Sec7): the D-SLM57 ctx_fold join's realizable half.
// Builds a hand-built WSC1 "layer0.ctx_fold" [2,3] int32 tensor -- head 0
// identity (1,0,0), head 1 non-identity, using the INDEPENDENTLY-derived
// (mult, shift) from kCtxFoldJoinCase (gen_s3_3_fixtures.py Sec4 -- a
// gemmlowp QuantizeMultiplier decomposition written from the published
// algorithm, never from Tools/superslm_spike/pipeline.py's own source, so
// this is not the correlated-oracle shape F-S3-3 warns against). Loads it
// through the REAL, already-shipped SslmModel::Load (WSC1's identity/shift
// domains already enforced there) and drives the REAL, already-shipped
// ApplyWeightScaleFold on both heads.
static FixtureSection MakeCtxFoldSection(int32_t mult1, int32_t shift1) {
	using namespace superslm_test;
	auto manifest = BuildManifest(superslm::kWeightScalesMagic, /*element_size=*/4,
	                               {{"layer0.ctx_fold", {2, 3}}});
	const size_t off = static_cast<size_t>(manifest.tensor_data_off[0]);
	PutU32(manifest.bytes, off + 0, static_cast<uint32_t>(1));   // head0.identity
	PutU32(manifest.bytes, off + 4, static_cast<uint32_t>(0));   // head0.mult (unused, identity)
	PutU32(manifest.bytes, off + 8, static_cast<uint32_t>(0));   // head0.shift
	PutU32(manifest.bytes, off + 12, static_cast<uint32_t>(0));  // head1.identity
	PutU32(manifest.bytes, off + 16, static_cast<uint32_t>(mult1));
	PutU32(manifest.bytes, off + 20, static_cast<uint32_t>(shift1));
	return MakeSection(SslmSectionType::WeightScales, SslmDtype::Int32, manifest.bytes, /*alignment=*/64);
}

static void TestCtxFoldJoinIdentityVsIndependentlyDecomposedNonIdentityOnHandBuiltWsc1() {
	using namespace superslm_test;
	auto built = BuildArtifact(
	    {MakeValidConfigSection(), MakeSigmoidLutSection(), MakeCtxFoldSection(kCtxFoldJoinCase.mult, kCtxFoldJoinCase.shift)});
	SslmModelView view;
	std::string err;
	SslmModelStatus status = SslmModel::Load(built.bytes.data(), built.bytes.size(), view, &err);
	CHECK_MSG(status == SslmModelStatus::Ok,
	          "hand-built layer0.ctx_fold WSC1 section: SslmModel::Load status == %s, want Ok (%s)",
	          SslmModelStatusName(status), err.c_str());
	if (status != SslmModelStatus::Ok) return;

	const SslmTensorView* ctx_fold = view.weight_scales.Tensor("layer0.ctx_fold");
	CHECK_MSG(ctx_fold != nullptr, "the WeightScales view exposes no \"layer0.ctx_fold\" tensor");
	if (ctx_fold == nullptr) return;
	CHECK_MSG(ctx_fold->elem_count == 6, "layer0.ctx_fold elem_count == %llu, want 6 (2 heads x 3)",
	          static_cast<unsigned long long>(ctx_fold->elem_count));

	// Read the two rows back the same way a real forward would (a raw
	// little-endian read over the validated tensor view, never a cast).
	const int32_t h0_identity = ReadRawI32LE(ctx_fold->data + 0);
	const int32_t h0_mult = ReadRawI32LE(ctx_fold->data + 4);
	const int32_t h0_shift = ReadRawI32LE(ctx_fold->data + 8);
	const int32_t h1_identity = ReadRawI32LE(ctx_fold->data + 12);
	const int32_t h1_mult = ReadRawI32LE(ctx_fold->data + 16);
	const int32_t h1_shift = ReadRawI32LE(ctx_fold->data + 20);
	CHECK(h0_identity == 1 && h1_identity == 0);
	CHECK(h1_mult == kCtxFoldJoinCase.mult && h1_shift == kCtxFoldJoinCase.shift);

	const int64_t out0 = superslm::ApplyWeightScaleFold(kCtxFoldJoinCase.ctx0, h0_identity, h0_mult, h0_shift);
	CHECK_MSG(out0 == kCtxFoldJoinCase.expected0,
	          "ApplyWeightScaleFold(identity head, from the artifact's own row) == %lld, want %lld "
	          "(the true pass-through -- ctx0 unchanged)",
	          static_cast<long long>(out0), static_cast<long long>(kCtxFoldJoinCase.expected0));

	const int64_t out1 = superslm::ApplyWeightScaleFold(kCtxFoldJoinCase.ctx1, h1_identity, h1_mult, h1_shift);
	CHECK_MSG(out1 == kCtxFoldJoinCase.expected1,
	          "ApplyWeightScaleFold(non-identity head, from the artifact's own row) == %lld, want %lld "
	          "(the independently-derived gemmlowp decomposition's own oracle -- Sec13.1 cell 9's "
	          "realizable half: what the artifact carries, applied through the real shipped kernel, "
	          "matches an oracle built without reading pipeline.py's own quantize_multiplier source)",
	          static_cast<long long>(out1), static_cast<long long>(kCtxFoldJoinCase.expected1));

	CHECK_MSG(out0 != out1 || kCtxFoldJoinCase.ctx0 != kCtxFoldJoinCase.ctx1,
	          "sanity: the two heads' own dispatch must actually differ for this cell to discriminate "
	          "identity from non-identity");
}

// Sec1 (record's Sec4.2/Sec7): C33's clamp witnesses, pinned against the
// REAL, already-shipped RopeApplyPair (not only the vendored Python mirror
// gen_s3_3_fixtures.py calls to derive them) -- the clamp SITE itself is
// blocked (no per-layer forward exists to call it from), but the witnesses
// that will feed it the moment it lands are proven genuine here, against the
// real primitive, so the C33 red cell (record Sec4.2) is a drop-in the day
// the site exists.
static void TestC33ClampWitnessesGenuinelyExceedTheirTargetRangesAgainstTheRealRopeApplyPair() {
	using superslm::RopeApplyPair;

	const auto& c = kC33BothSignsCase;
	const auto pos = RopeApplyPair(c.x, c.y, c.cos_q30, c.sin_q30);
	CHECK_MSG(pos.x == c.raw_x && pos.y == c.raw_y,
	          "RopeApplyPair(x=%d,y=%d,cos=%d,sin=%d) == {%lld,%lld}, want the fixture's own "
	          "{%lld,%lld} (recomputed via the vendored reference)",
	          c.x, c.y, c.cos_q30, c.sin_q30, static_cast<long long>(pos.x), static_cast<long long>(pos.y),
	          static_cast<long long>(c.raw_x), static_cast<long long>(c.raw_y));
	CHECK_MSG(pos.x > 127, "the positive-side witness's real RopeApplyPair output (%lld) must exceed +127",
	          static_cast<long long>(pos.x));

	const auto neg = RopeApplyPair(c.x_neg, c.y_neg, c.cos_q30, c.sin_q30);
	CHECK_MSG(neg.x == c.raw_x_neg && neg.y == c.raw_y_neg,
	          "RopeApplyPair on the negated input must reproduce the fixture's own negated witness "
	          "(RopeApplyPair is linear) -- got {%lld,%lld}, want {%lld,%lld}",
	          static_cast<long long>(neg.x), static_cast<long long>(neg.y),
	          static_cast<long long>(c.raw_x_neg), static_cast<long long>(c.raw_y_neg));
	CHECK_MSG(neg.x < -127, "the negative-side witness's real RopeApplyPair output (%lld) must exceed -127",
	          static_cast<long long>(neg.x));

	const auto& c2 = kC33Int32OverflowCase;
	const auto wide = RopeApplyPair(c2.x, c2.y, c2.cos_q30, c2.sin_q30);
	CHECK_MSG(wide.x == c2.raw_x, "the int32-overflow witness's real RopeApplyPair.x == %lld, want %lld",
	          static_cast<long long>(wide.x), static_cast<long long>(c2.raw_x));
	CHECK_MSG(wide.x > INT32_MAX || wide.x < INT32_MIN,
	          "the int32-overflow witness's real RopeApplyPair output (%lld) must exceed int32's own "
	          "representable range -- otherwise this fixture cannot pin that the future clamp site "
	          "reads the int64 RopePair member rather than a narrowed copy",
	          static_cast<long long>(wide.x));

	// The two negative controls the future clamp cell must fail (record
	// Sec4.2): an unclamped implementation, and clamp-at-[-128,127] (the int8
	// STORAGE range, not the pinned CODE range). Verified here that the
	// fixture's own witness actually discriminates the second control at the
	// exact point the plan calls load-bearing (Sec5.3): a raw value of
	// exactly -128 is admitted by [-128,127] and rejected by [-127,127].
	const int64_t clamp127 = std::clamp<int64_t>(neg.x, -127, 127);
	const int64_t clamp128 = std::clamp<int64_t>(neg.x, -128, 127);
	CHECK_MSG(clamp127 == -127, "clamp(%lld, -127, 127) == %lld, want -127", static_cast<long long>(neg.x),
	          static_cast<long long>(clamp127));
	CHECK_MSG(clamp128 != clamp127 || neg.x >= -127,
	          "the [-128,127] negative control has no discriminating power on this witness unless the "
	          "raw value is below -127 but not exactly -128 -- got raw=%lld", static_cast<long long>(neg.x));
}

// Sec5 (record's Sec4.4/Sec7): KvLandingScales'/KvLandingReciprocals' owed
// joint-domain bound (Plan Sec7.2a), pinned against the REAL, already-shipped
// DynamicScaleReciprocal at its own domain's two endpoints -- the new
// SslmModelStatus/ValidateKvLandingXDomain checks that will USE this bound
// are blocked (undeclared), but the bound itself is proven against the real
// primitive it is defined in terms of ("R_t IS C19's reciprocal", Plan
// Sec8.1), not only against the vendored Python mirror.
static void TestKvLandingReciprocalBoundMatchesTheRealDynamicScaleReciprocalDomainEndpoints() {
	using superslm::DynamicScaleReciprocal;

	const int64_t r_at_min_dn = DynamicScaleReciprocal(INT64_C(1) << 30);
	CHECK_MSG(r_at_min_dn == kKvLandingReciprocalMax,
	          "DynamicScaleReciprocal(2^30) == %lld, want %lld (kKvLandingReciprocalMax, the derived "
	          "upper bound on any legitimate KvLandingReciprocals entry)",
	          static_cast<long long>(r_at_min_dn), static_cast<long long>(kKvLandingReciprocalMax));
	CHECK(r_at_min_dn == (INT64_C(1) << 32));

	const int64_t r_at_max_dn = DynamicScaleReciprocal((INT64_C(1) << 31) - 1);
	CHECK_MSG(r_at_max_dn == kKvLandingReciprocalMin,
	          "DynamicScaleReciprocal(2^31-1) == %lld, want %lld (kKvLandingReciprocalMin, the derived "
	          "lower bound on any legitimate KvLandingReciprocals entry)",
	          static_cast<long long>(r_at_max_dn), static_cast<long long>(kKvLandingReciprocalMin));
	CHECK(r_at_max_dn == (INT64_C(1) << 31) + 1);

	// MakeMinimalValidKvc1's own shared "gamma" third word (a near-INT64_MAX
	// witness chosen for the KVC1 SUB-PARSE's little-endian-signed-read
	// cell, sslm_kvc1_hostile_fixtures.h) is nowhere near this derived
	// domain -- confirmed here rather than asserted, so the record's routed
	// finding (the not-yet-existing ValidateKvLandingReciprocalsDomain check
	// must reject it) is grounded in an executed comparison, not a claim
	// about a literal nobody re-checked.
	constexpr int64_t kGammaThirdWord = INT64_C(9223372036854775807);  // sslm_kvc1_hostile_fixtures.h's own value
	CHECK_MSG(kGammaThirdWord > kKvLandingReciprocalMax,
	          "MakeMinimalValidKvc1's gamma R-word (%lld) must be found out-of-domain once the owed "
	          "KvLandingReciprocals check lands (record Sec4.7's routed cell) -- got a value inside "
	          "[%lld, %lld] instead, which would make that routed finding wrong",
	          static_cast<long long>(kGammaThirdWord), static_cast<long long>(kKvLandingReciprocalMin),
	          static_cast<long long>(kKvLandingReciprocalMax));
}

// ---------------------------------------------------------------------------
// The header contract landed (D:\SuperSLM@169265d/c4ee594). Every cell below
// was fully specified in the test-design record's Sec6-Sec8 as blocked on an
// undeclared symbol; each now compiles, links, and fails against the
// deliberately-wrong stub it is authored against (LandingRescale/
// ClampRopeCode return 0 unconditionally and never touch an output
// parameter; CheckSoftmaxRowWidthDomain always returns WorkspaceTooSmall;
// SoftmaxRowQ15/GemmProbQ15Accumulate write nothing;
// ValidateKvLandingScalesDomain/ValidateKvLandingReciprocalsDomain always
// return Ok) -- each for its own reason, verified individually below rather
// than assumed from the stub's own doc comment.
//
// Dan's ruling (T-1304/D-SLM367): the C32 numerator ceiling is 2^47 on every
// path. This suite reads that ruling through kSoftmaxRowMaxSafeExponent
// (checked_chain_funnel.h) -- via kC32WidthDomainCases' own
// `ok_under_2pow47` field, which IS that ruling -- never as a re-typed
// literal. The band witness (record Sec4.2/Sec3) is now asserted REJECTED,
// per the coordinator's explicit correction to this suite's own prior draft.
// ---------------------------------------------------------------------------

// Record Sec6.1: C27's K/V landing composite, feature oracle against the
// same `kLandingCompositeCases` witnesses Sec4.3 derived (the regeneration
// gate already proves both named negative controls -- per-key-token-
// reciprocal, two-rounding scale-product -- diverge from `correct` on the
// two discriminating rows; asserting LandingRescale equals `correct`
// therefore already has full discriminating power against either wrong
// construction, without re-implementing them a second time in C++).
static void TestLandingRescaleFeatureOracleAgainstResidualReconcileWitnesses() {
	using namespace superslm_test;
	for (size_t i = 0; i < kLandingCompositeCasesCount; ++i) {
		const LandingCompositeCase& c = kLandingCompositeCases[i];
		const int64_t raw = superslm::LandingRescale(c.branch_code, c.m_b, c.r_h, c.e_b, c.e_h);
		CHECK_MSG(raw == c.correct_raw,
		          "%s: LandingRescale(branch_code=%lld, m_a=%lld, r_t=%lld, e_a=%d, e_t=%d) == %lld, "
		          "want %lld (residual_reconcile's own pinned formula, C3 away-from-zero)",
		          c.label, static_cast<long long>(c.branch_code), static_cast<long long>(c.m_b),
		          static_cast<long long>(c.r_h), c.e_b, c.e_h, static_cast<long long>(raw),
		          static_cast<long long>(c.correct_raw));
		const int64_t clamped = std::clamp<int64_t>(raw, -127, 127);
		CHECK_MSG(clamped == c.correct,
		          "%s: clamp(LandingRescale(...), -127, 127) == %lld, want %lld (Sec8.1's own "
		          "composition: the clamp is the caller's, matching C33's)",
		          c.label, static_cast<long long>(clamped), static_cast<long long>(c.correct));
	}
}

// Record Sec7 (T-518/D-SLM201 option 2, Sec8.2; coordinator ruling 3): three
// named cells on `LandingRescale`'s own `out_saturation_count` half, plus the
// non-negotiable invariance property. Every witness below uses
// `m_a=2^30, r_t=2^32, e_a=e_t=0`, at which `residual_reconcile`'s divide is
// EXACT (numerator == branch_code * 2^62, denominator == 2^62, no rounding)
// -- confirmed by direct execution of the vendored reference at generation
// time (gen_s3_3_fixtures.py's own record, Sec4.3's derivation script run
// interactively) -- so `LandingRescale`'s raw return is exactly
// `branch_code` and the clamp fires iff `|branch_code| > 127`, giving exact,
// hand-verifiable saturation counts rather than a derived-and-hoped-for one.
namespace {
constexpr int64_t kSatCounterMA = INT64_C(1) << 30;
constexpr int64_t kSatCounterRT = INT64_C(1) << 32;
}  // namespace

// Sec7's first cell: a hand-constructed landing input with a KNOWN number of
// clamped elements across a KNOWN set of (head, projection) sites -- here,
// five sequential calls sharing one accumulator, standing in for five sites
// of one landing composite. Two saturate (128, -128 -- one past each bound);
// three do not (127, -127 -- the in-band boundary controls; 50, comfortably
// interior). The reported VALUE is asserted, not merely that it fires
// (Sec8.2: "the counter's reported value is asserted, not only its firing").
static void TestLandingRescaleSaturationCounterExactValueOnKnownClampedElements() {
	struct Site {
		int64_t branch_code;
		bool expect_clamp;
	};
	const Site sites[] = {
	    {127, false},   // exactly at the pinned code bound -- in-band control
	    {128, true},    // one past the bound -- must fire
	    {-127, false},  // exactly at the negative bound -- in-band control
	    {-128, true},   // one past the negative bound -- must fire
	    {50, false},    // comfortably interior
	};
	uint64_t count = 0;
	for (const Site& s : sites) {
		const int64_t raw = superslm::LandingRescale(s.branch_code, kSatCounterMA, kSatCounterRT, 0, 0, &count);
		CHECK_MSG(raw == s.branch_code,
		          "LandingRescale(%lld, m_a=2^30, r_t=2^32, e_a=e_t=0) == %lld, want %lld exactly "
		          "(this witness's own divide is exact, no rounding)",
		          static_cast<long long>(s.branch_code), static_cast<long long>(raw),
		          static_cast<long long>(s.branch_code));
	}
	CHECK_MSG(count == 2,
	          "*out_saturation_count after 5 sites (2 saturating: 128, -128) == %llu, want 2 -- the "
	          "reported VALUE, not merely that the counter fired",
	          static_cast<unsigned long long>(count));
}

// Sec7's second cell: multi-token monotone accumulation -- the accumulated
// count after N tokens equals the sum of the per-token counts, and no
// intermediate checkpoint reports a count for a site not yet executed
// (Sec13 dim 8's "counter x layer budget" crossing, realized without a
// layer-budget mechanism by simply checking the running total at each
// checkpoint rather than only at the end).
static void TestLandingRescaleSaturationCounterMonotoneAcrossTokens() {
	uint64_t count = 0;

	// "Token" 1: one site, saturates. Running total after: 1.
	superslm::LandingRescale(200, kSatCounterMA, kSatCounterRT, 0, 0, &count);
	CHECK_MSG(count == 1, "after token 1 (1 saturating site): count == %llu, want 1",
	          static_cast<unsigned long long>(count));

	// "Token" 2: three sites, two saturate. Running total after: 1 + 2 == 3.
	superslm::LandingRescale(300, kSatCounterMA, kSatCounterRT, 0, 0, &count);
	superslm::LandingRescale(-300, kSatCounterMA, kSatCounterRT, 0, 0, &count);
	superslm::LandingRescale(10, kSatCounterMA, kSatCounterRT, 0, 0, &count);
	CHECK_MSG(count == 3, "after token 2 (cumulative +2 saturating): count == %llu, want 3 (1 + 2)",
	          static_cast<unsigned long long>(count));

	// "Token" 3: two sites, neither saturates. Running total unchanged: 3.
	superslm::LandingRescale(20, kSatCounterMA, kSatCounterRT, 0, 0, &count);
	superslm::LandingRescale(-20, kSatCounterMA, kSatCounterRT, 0, 0, &count);
	CHECK_MSG(count == 3,
	          "after token 3 (0 saturating sites, monotone -- must not decrease or reset): count == "
	          "%llu, want 3 (the accumulated total is per-sequence, never per-token)",
	          static_cast<unsigned long long>(count));
}

// Sec7's third cell: a whole row where EVERY element saturates, so an
// off-by-one in the predicate's placement (e.g. incrementing before the
// comparison, or skipping the row's last element) is visible as a count that
// is not exactly the row length.
static void TestLandingRescaleSaturationCounterWholeRowClamp() {
	uint64_t count = 0;
	const int64_t row[] = {500, -500, 200, -1000};  // every element saturates
	for (int64_t branch_code : row) {
		superslm::LandingRescale(branch_code, kSatCounterMA, kSatCounterRT, 0, 0, &count);
	}
	CHECK_MSG(count == 4,
	          "a 4-element row where every element saturates: count == %llu, want exactly 4 -- an "
	          "off-by-one in the predicate's placement would report 3 or 5",
	          static_cast<unsigned long long>(count));
}

// Sec8.2's own non-negotiable property, asserted directly: the saturation
// count has NO EFFECT WHATSOEVER on the return value. Same input, once with
// `out_saturation_count == nullptr` and once with a real accumulator; the
// two raw results must be bit-identical, on both a saturating and a
// non-saturating branch_code.
static void TestLandingRescaleSaturationCountHasNoEffectOnReturnValue() {
	const int64_t no_counter_interior = superslm::LandingRescale(50, kSatCounterMA, kSatCounterRT, 0, 0, nullptr);
	uint64_t count = 0;
	const int64_t with_counter_interior =
	    superslm::LandingRescale(50, kSatCounterMA, kSatCounterRT, 0, 0, &count);
	CHECK_MSG(no_counter_interior == with_counter_interior,
	          "LandingRescale(50, ...) == %lld with no counter, %lld with one -- the counter must have "
	          "no effect whatsoever on the return value (Sec8.2)",
	          static_cast<long long>(no_counter_interior), static_cast<long long>(with_counter_interior));

	const int64_t no_counter_saturating =
	    superslm::LandingRescale(500, kSatCounterMA, kSatCounterRT, 0, 0, nullptr);
	uint64_t count2 = 0;
	const int64_t with_counter_saturating =
	    superslm::LandingRescale(500, kSatCounterMA, kSatCounterRT, 0, 0, &count2);
	CHECK_MSG(no_counter_saturating == with_counter_saturating,
	          "LandingRescale(500, ...) == %lld with no counter, %lld with one -- must agree even on a "
	          "SATURATING branch_code, where the counter is actually active",
	          static_cast<long long>(no_counter_saturating), static_cast<long long>(with_counter_saturating));
	CHECK_MSG(count2 == 1, "the counting call above must still have counted (count2 == %llu, want 1)",
	          static_cast<unsigned long long>(count2));
}

// Record Sec6.3: C33's post-rotation clamp. `ClampRopeCode` is now the real
// caller-side site; asserted against the same witnesses Sec4.1 already
// pinned genuine against the real RopeApplyPair, above.
static void TestClampRopeCodeAgainstC33Witnesses() {
	using superslm::ClampRopeCode;

	const auto& c = kC33BothSignsCase;
	CHECK_MSG(ClampRopeCode(c.raw_x) == 127, "ClampRopeCode(%lld) == %lld, want 127",
	          static_cast<long long>(c.raw_x), static_cast<long long>(ClampRopeCode(c.raw_x)));
	CHECK_MSG(ClampRopeCode(c.raw_x_neg) == -127,
	          "ClampRopeCode(%lld) == %lld, want -127 (not -128 -- the pinned CODE range, not the int8 "
	          "STORAGE range, Sec5.3's own load-bearing distinction)",
	          static_cast<long long>(c.raw_x_neg), static_cast<long long>(ClampRopeCode(c.raw_x_neg)));

	const auto& c2 = kC33Int32OverflowCase;
	CHECK_MSG(ClampRopeCode(c2.raw_x) == 127,
	          "ClampRopeCode(%lld) == %lld, want 127 -- a implementation that narrows to int32 before "
	          "clamping would wrap this value negative first and return -127 instead",
	          static_cast<long long>(c2.raw_x), static_cast<long long>(ClampRopeCode(c2.raw_x)));
}

// Record Sec6.2: C32/D-SLM366's width predicate, now read against
// `kSoftmaxRowMaxSafeExponent` (via each case's own `ok_under_2pow47` field
// -- Dan's ruling, T-1304/D-SLM367) rather than either superseded candidate.
static bool ExpectedSoftmaxRowWidthOk(const superslm_test::C32WidthDomainCase& c) {
	if (!c.ok_under_2pow47) return false;
	if (c.row_max <= 0) return true;
	return static_cast<uint64_t>(c.width) <=
	       static_cast<uint64_t>(INT64_MAX) / static_cast<uint64_t>(c.row_max);
}

static void TestCheckSoftmaxRowWidthDomainAgainstDerivedCasesAndTheRoutedBandCase() {
	using namespace superslm_test;
	using superslm::CheckSoftmaxRowWidthDomain;
	using superslm::SslmForwardStatus;

	for (size_t i = 0; i < kC32WidthDomainCasesCount; ++i) {
		const C32WidthDomainCase& c = kC32WidthDomainCases[i];
		const auto status = CheckSoftmaxRowWidthDomain(c.q_b, c.q_c, c.width);
		const bool expect_ok = ExpectedSoftmaxRowWidthOk(c);
		CHECK_MSG(expect_ok ? status == SslmForwardStatus::Ok
		                    : status == SslmForwardStatus::SoftmaxRowWidthOutOfDomain,
		          "%s: CheckSoftmaxRowWidthDomain(q_b=%lld, q_c=%lld, width=%zu) status == %s, want %s "
		          "(row_max=%lld against kSoftmaxRowMaxSafeExponent, D-SLM367's ratified 2^47)",
		          c.label, static_cast<long long>(c.q_b), static_cast<long long>(c.q_c), c.width,
		          superslm::SslmForwardStatusName(status),
		          expect_ok ? "Ok" : "SoftmaxRowWidthOutOfDomain", static_cast<long long>(c.row_max));
	}

	// The routed band case (record Sec3/Sec4.2): the coordinator's own
	// ruling directs this suite to assert REJECTION now that 2^47 is the
	// shipped threshold on every path -- the case is `ok_under_2pow48m1` but
	// NOT `ok_under_2pow47`, so `kC32BandCase.ok_under_2pow47` alone already
	// carries the correct verdict.
	CHECK_MSG(!kC32BandCase.ok_under_2pow47,
	          "the band witness's own fixture field must read ok_under_2pow47==false under the "
	          "ratified threshold, or this cell's own premise is stale");
	const auto band_status =
	    CheckSoftmaxRowWidthDomain(kC32BandCase.q_b, kC32BandCase.q_c, kC32BandCase.width);
	CHECK_MSG(band_status == SslmForwardStatus::SoftmaxRowWidthOutOfDomain,
	          "the band witness (row_max=%lld, between the two superseded candidate thresholds): "
	          "CheckSoftmaxRowWidthDomain status == %s, want SoftmaxRowWidthOutOfDomain -- REJECTED, "
	          "per D-SLM367's ruling that 2^47 is the shipped ceiling on every path",
	          static_cast<long long>(kC32BandCase.row_max), superslm::SslmForwardStatusName(band_status));
}

// Record Sec6.2: the softmax row kernel itself. The oracle is composed from
// the SAME already-certified primitives the kernel's own pinned pseudocode
// names (ShiftByMax -> IExpConstruct/IExpEvaluate per element -> sum ->
// Q15 divide) -- C32's pinned formula IS this composition, so driving the
// real primitives directly computes the statistic's definition, not a
// recode of SoftmaxRowQ15's own internals (matching S3.2's RmsNormSite
// precedent, which used the already-shipped funnel as its own site's
// oracle).
static void TestSoftmaxRowQ15AgainstComposedShippedPrimitivesOracle() {
	using namespace superslm;
	using namespace superslm_test;

	const C32WidthDomainCase* accept_case = nullptr;
	for (size_t i = 0; i < kC32WidthDomainCasesCount; ++i) {
		if (std::strcmp(kC32WidthDomainCases[i].label, "accept_realistic_width") == 0) {
			accept_case = &kC32WidthDomainCases[i];
			break;
		}
	}
	CHECK_MSG(accept_case != nullptr, "kC32WidthDomainCases must carry an 'accept_realistic_width' case");
	if (accept_case == nullptr) return;

	constexpr size_t kWidth = 3;
	const int64_t raw_scores[kWidth] = {10, 5, 0};

	int64_t shifted[kWidth];
	ShiftByMax(raw_scores, kWidth, shifted);

	int64_t expected_e[kWidth];
	int64_t expected_total = 0;
	for (size_t i = 0; i < kWidth; ++i) {
		IExpConstruction construction;
		const IExpDomain d =
		    IExpConstruct(shifted[i], accept_case->q_ln2, accept_case->q_b, accept_case->q_c, &construction);
		CHECK_MSG(d == IExpDomain::kOk, "IExpConstruct(shifted[%zu]=%lld) domain == %d, want kOk", i,
		          static_cast<long long>(shifted[i]), static_cast<int>(d));
		expected_e[i] = IExpEvaluate(construction);
		expected_total += expected_e[i];
	}
	int64_t expected_probs[kWidth];
	for (size_t i = 0; i < kWidth; ++i) {
		expected_probs[i] = (expected_e[i] << kProbFracBits) / std::max<int64_t>(expected_total, 1);
	}

	int64_t out_probs[kWidth] = {INT64_C(-99), INT64_C(-99), INT64_C(-99)};  // poison
	SoftmaxRowQ15(raw_scores, kWidth, accept_case->q_ln2, accept_case->q_b, accept_case->q_c, out_probs);

	for (size_t i = 0; i < kWidth; ++i) {
		CHECK_MSG(out_probs[i] == expected_probs[i],
		          "SoftmaxRowQ15(...)[%zu] == %lld, want %lld (ShiftByMax -> IExpConstruct/IExpEvaluate "
		          "-> sum -> Q15 divide, composed from the already-certified primitives directly)",
		          i, static_cast<long long>(out_probs[i]), static_cast<long long>(expected_probs[i]));
	}
}

// Record Sec6.4: `GemmProbQ15Accumulate`'s feature oracle plus its
// order-freedom certification (Sec4.6 of the plan: "the same accumulate
// under permuted summation orders ... bit-identical"), plus the two named
// width extremes (a row whose mass is entirely on one key; a row spread
// across many keys).
static void TestGemmProbQ15AccumulateFeatureOracleAndOrderFreedomCertification() {
	using superslm::GemmProbQ15Accumulate;

	// Feature oracle: width=2, head_dim=2. key0=[3,-3], key1=[7,1].
	{
		const int64_t probs[2] = {100, 200};
		const int8_t values[4] = {3, -3, 7, 1};
		int64_t out_ctx[2] = {INT64_C(-99), INT64_C(-99)};  // poison
		GemmProbQ15Accumulate(probs, values, 2, 2, out_ctx);
		CHECK_MSG(out_ctx[0] == 1700, "out_ctx[0] == %lld, want 1700 (100*3 + 200*7)",
		          static_cast<long long>(out_ctx[0]));
		CHECK_MSG(out_ctx[1] == -100, "out_ctx[1] == %lld, want -100 (100*-3 + 200*1)",
		          static_cast<long long>(out_ctx[1]));
	}

	// Order-freedom certification: the SAME two (prob, value-row) pairs, key
	// order reversed. Exact int64 addition is commutative and associative,
	// so the result must be bit-identical to the un-reversed call above.
	{
		const int64_t probs_rev[2] = {200, 100};
		const int8_t values_rev[4] = {7, 1, 3, -3};
		int64_t out_ctx_rev[2] = {INT64_C(-99), INT64_C(-99)};  // poison
		GemmProbQ15Accumulate(probs_rev, values_rev, 2, 2, out_ctx_rev);
		CHECK_MSG(out_ctx_rev[0] == 1700,
		          "permuted-order out_ctx[0] == %lld, want 1700 (bit-identical to the un-permuted call)",
		          static_cast<long long>(out_ctx_rev[0]));
		CHECK_MSG(out_ctx_rev[1] == -100,
		          "permuted-order out_ctx[1] == %lld, want -100 (bit-identical to the un-permuted call)",
		          static_cast<long long>(out_ctx_rev[1]));
	}

	// Width extreme 1: mass entirely on one key (width=8, head_dim=1).
	{
		const int64_t probs[8] = {INT64_C(32768), 0, 0, 0, 0, 0, 0, 0};
		const int8_t values[8] = {7, -1, -1, -1, -1, -1, -1, -1};  // only key 0's value matters
		int64_t out_ctx[1] = {INT64_C(-99)};  // poison
		GemmProbQ15Accumulate(probs, values, 8, 1, out_ctx);
		CHECK_MSG(out_ctx[0] == INT64_C(229376), "mass-on-one-key out_ctx[0] == %lld, want 229376 (32768*7)",
		          static_cast<long long>(out_ctx[0]));
	}

	// Width extreme 2: spread across many keys (width=64, head_dim=1),
	// probabilities summing to exactly 2^15 (C32's own row-sum bound).
	{
		constexpr size_t kWidth = 64;
		int64_t probs[kWidth];
		int8_t values[kWidth];
		int64_t expected = 0;
		for (size_t k = 0; k < kWidth; ++k) {
			probs[k] = 512;  // 64 * 512 == 32768 == 2^15
			values[k] = static_cast<int8_t>(static_cast<int>(k % 3) - 1);  // -1, 0, 1 repeating
			expected += probs[k] * static_cast<int64_t>(values[k]);
		}
		int64_t out_ctx[1] = {INT64_C(-99)};  // poison
		GemmProbQ15Accumulate(probs, values, kWidth, 1, out_ctx);
		CHECK_MSG(out_ctx[0] == expected,
		          "spread-across-%zu-keys out_ctx[0] == %lld, want %lld (exact int64 sum over the whole "
		          "row, C27's own stated bound |Sum p_k*v_k| <= 2^15*127 << int64)",
		          kWidth, static_cast<long long>(out_ctx[0]), static_cast<long long>(expected));
	}
}

// Record Sec6.5: KvLandingScales'/KvLandingReciprocals' owed value-domain
// descriptors (Plan Sec7.2a's third limb). Section builders on the exact
// "one otherwise-valid v2 artifact, one hostile section" pattern
// `MakeBia1Section` (S3.2, above) already established, reusing `BuildKvc1`
// (sslm_kvc1_hostile_fixtures.h) since both section types ARE KVC1 tables.
static FixtureSection MakeKvLandingScalesSection(int64_t m_t, int64_t e_t) {
	using namespace superslm_test;
	auto kvc1 = BuildKvc1(/*declared_value_words=*/2, {{"layer0.k_head0", {m_t, e_t}}});
	return MakeSection(SslmSectionType::KvLandingScales, SslmDtype::Raw, kvc1.bytes, /*alignment=*/64);
}

static FixtureSection MakeKvLandingReciprocalsSection(int64_t m_t, int64_t e_t, int64_t r_t) {
	using namespace superslm_test;
	auto kvc1 = BuildKvc1(/*declared_value_words=*/3, {{"layer0.k_head0", {m_t, e_t, r_t}}});
	return MakeSection(SslmSectionType::KvLandingReciprocals, SslmDtype::Raw, kvc1.bytes, /*alignment=*/64);
}

static void TestKvLandingScalesRejectsHostileMantissaBothSignsAndAcceptsTheBoundary() {
	using namespace superslm_test;
	// Reject: one past each bound.
	{
		auto built = BuildArtifact({MakeValidConfigSection(), MakeSigmoidLutSection(),
		                            MakeKvLandingScalesSection(kKvLandingScaleMantissaMin - 1, 0)});
		SslmModelView view;
		std::string err;
		auto status = SslmModel::Load(built.bytes.data(), built.bytes.size(), view, &err);
		CHECK_MSG(status == SslmModelStatus::KvLandingScaleOutOfDomain,
		          "m_t=%lld (one below the canonical minimum): SslmModel::Load status == %s, want "
		          "KvLandingScaleOutOfDomain (%s)",
		          static_cast<long long>(kKvLandingScaleMantissaMin - 1), SslmModelStatusName(status),
		          err.c_str());
		CHECK_MSG(!view.has_kv_landing_scales,
		          "hostile KvLandingScales view exposed on a rejected Load — a view MUST NOT be exposed");
	}
	{
		auto built = BuildArtifact({MakeValidConfigSection(), MakeSigmoidLutSection(),
		                            MakeKvLandingScalesSection(kKvLandingScaleMantissaMax + 1, 0)});
		SslmModelView view;
		std::string err;
		auto status = SslmModel::Load(built.bytes.data(), built.bytes.size(), view, &err);
		CHECK_MSG(status == SslmModelStatus::KvLandingScaleOutOfDomain,
		          "m_t=%lld (one above the canonical maximum): SslmModel::Load status == %s, want "
		          "KvLandingScaleOutOfDomain (%s)",
		          static_cast<long long>(kKvLandingScaleMantissaMax + 1), SslmModelStatusName(status),
		          err.c_str());
		CHECK_MSG(!view.has_kv_landing_scales,
		          "hostile KvLandingScales view exposed on a rejected Load — a view MUST NOT be exposed");
	}
	// Accept-at-bound, both endpoints — the off-by-one control every
	// S-HARDEN-1 boundary matrix carries (record Sec13 dim 4).
	{
		auto built = BuildArtifact({MakeValidConfigSection(), MakeSigmoidLutSection(),
		                            MakeKvLandingScalesSection(kKvLandingScaleMantissaMin, 0)});
		SslmModelView view;
		std::string err;
		auto status = SslmModel::Load(built.bytes.data(), built.bytes.size(), view, &err);
		CHECK_MSG(status == SslmModelStatus::Ok, "m_t at the canonical minimum: status == %s, want Ok (%s)",
		          SslmModelStatusName(status), err.c_str());
		CHECK(view.has_kv_landing_scales);
	}
	{
		auto built = BuildArtifact({MakeValidConfigSection(), MakeSigmoidLutSection(),
		                            MakeKvLandingScalesSection(kKvLandingScaleMantissaMax, 0)});
		SslmModelView view;
		std::string err;
		auto status = SslmModel::Load(built.bytes.data(), built.bytes.size(), view, &err);
		CHECK_MSG(status == SslmModelStatus::Ok, "m_t at the canonical maximum: status == %s, want Ok (%s)",
		          SslmModelStatusName(status), err.c_str());
		CHECK(view.has_kv_landing_scales);
	}
}

static void TestKvLandingReciprocalsRejectsHostileMagnitudeBothSignsAndAcceptsTheBoundary() {
	using namespace superslm_test;
	// Reject: one past each bound. m_t/e_t held at a safe, in-domain value
	// throughout (only R_t, word 2, is under test here).
	{
		auto built = BuildArtifact(
		    {MakeValidConfigSection(), MakeSigmoidLutSection(),
		     MakeKvLandingReciprocalsSection(kKvLandingScaleMantissaMin, 0, kKvLandingReciprocalMin - 1)});
		SslmModelView view;
		std::string err;
		auto status = SslmModel::Load(built.bytes.data(), built.bytes.size(), view, &err);
		CHECK_MSG(status == SslmModelStatus::KvLandingReciprocalOutOfDomain,
		          "R_t=%lld (one below the derived minimum): SslmModel::Load status == %s, want "
		          "KvLandingReciprocalOutOfDomain (%s)",
		          static_cast<long long>(kKvLandingReciprocalMin - 1), SslmModelStatusName(status),
		          err.c_str());
		CHECK_MSG(!view.has_kv_landing_reciprocals,
		          "hostile KvLandingReciprocals view exposed on a rejected Load — a view MUST NOT be "
		          "exposed");
	}
	{
		auto built = BuildArtifact(
		    {MakeValidConfigSection(), MakeSigmoidLutSection(),
		     MakeKvLandingReciprocalsSection(kKvLandingScaleMantissaMin, 0, kKvLandingReciprocalMax + 1)});
		SslmModelView view;
		std::string err;
		auto status = SslmModel::Load(built.bytes.data(), built.bytes.size(), view, &err);
		CHECK_MSG(status == SslmModelStatus::KvLandingReciprocalOutOfDomain,
		          "R_t=%lld (one above the derived maximum): SslmModel::Load status == %s, want "
		          "KvLandingReciprocalOutOfDomain (%s)",
		          static_cast<long long>(kKvLandingReciprocalMax + 1), SslmModelStatusName(status),
		          err.c_str());
		CHECK_MSG(!view.has_kv_landing_reciprocals,
		          "hostile KvLandingReciprocals view exposed on a rejected Load — a view MUST NOT be "
		          "exposed");
	}
	// Accept-at-bound, both endpoints.
	{
		auto built = BuildArtifact(
		    {MakeValidConfigSection(), MakeSigmoidLutSection(),
		     MakeKvLandingReciprocalsSection(kKvLandingScaleMantissaMin, 0, kKvLandingReciprocalMin)});
		SslmModelView view;
		std::string err;
		auto status = SslmModel::Load(built.bytes.data(), built.bytes.size(), view, &err);
		CHECK_MSG(status == SslmModelStatus::Ok, "R_t at the derived minimum: status == %s, want Ok (%s)",
		          SslmModelStatusName(status), err.c_str());
		CHECK(view.has_kv_landing_reciprocals);
	}
	{
		auto built = BuildArtifact(
		    {MakeValidConfigSection(), MakeSigmoidLutSection(),
		     MakeKvLandingReciprocalsSection(kKvLandingScaleMantissaMin, 0, kKvLandingReciprocalMax)});
		SslmModelView view;
		std::string err;
		auto status = SslmModel::Load(built.bytes.data(), built.bytes.size(), view, &err);
		CHECK_MSG(status == SslmModelStatus::Ok, "R_t at the derived maximum: status == %s, want Ok (%s)",
		          SslmModelStatusName(status), err.c_str());
		CHECK(view.has_kv_landing_reciprocals);
	}
}

// D-SLM393 / audit F3 (Claude/Mendeleev/superslm-s3.3-attention-interior-
// coverage-audit-2026-07-28.md Sec2 row 9, Sec4 F3): kKvLandingExponentMin
// (src/model.cpp:806, the e_t floor derived there, -60) had no boundary-
// exact accept/reject pair -- only the extreme witness e_t=-1000 above
// (TestKvLandingReciprocalsLoadRejectsAnExtremeUncheckedExponentRegardlessOfRT)
// proved the guard is reachable and fires, which a floor placed at any of
// many wrong values would also pass. This cell is the accept-at-bound /
// reject-one-past-bound pair every other S-HARDEN-1 boundary in this file
// already carries -- the direct template is the R_t cell immediately above
// (TestKvLandingReciprocalsRejectsHostileMagnitudeBothSignsAndAcceptsTheBoundary).
// m_t and R_t are held at canonical in-domain values throughout; only e_t
// (word 1) is under test. The literal -60 is not exposed as a test fixture
// constant (unlike kKvLandingReciprocalMin/Max, which are generated into
// tests/sslm_s3_3_fixtures.h) -- it is asserted directly against the
// production derivation's own comment at src/model.cpp:790-806, per this
// cell's audit specification.
static void TestKvLandingReciprocalsExponentFloorAcceptsAtBoundRejectsOnePast() {
	using namespace superslm_test;
	constexpr int64_t kExponentFloor = -60;  // src/model.cpp:806, kKvLandingExponentMin.

	// Reject: one past the floor (e_t == -61).
	{
		auto built = BuildArtifact({MakeValidConfigSection(), MakeSigmoidLutSection(),
		                            MakeKvLandingReciprocalsSection(kKvLandingScaleMantissaMin,
		                                                             kExponentFloor - 1,
		                                                             kKvLandingReciprocalMin)});
		SslmModelView view;
		std::string err;
		auto status = SslmModel::Load(built.bytes.data(), built.bytes.size(), view, &err);
		CHECK_MSG(status == SslmModelStatus::KvLandingReciprocalOutOfDomain,
		          "e_t=%lld (one below the derived floor -60), R_t=%lld (canonical): "
		          "SslmModel::Load status == %s, want KvLandingReciprocalOutOfDomain (%s)",
		          static_cast<long long>(kExponentFloor - 1),
		          static_cast<long long>(kKvLandingReciprocalMin), SslmModelStatusName(status),
		          err.c_str());
		CHECK_MSG(!view.has_kv_landing_reciprocals,
		          "hostile KvLandingReciprocals view exposed on a rejected Load — a view MUST NOT be "
		          "exposed");
	}
	// Accept-at-bound (e_t == -60 exactly).
	{
		auto built = BuildArtifact({MakeValidConfigSection(), MakeSigmoidLutSection(),
		                            MakeKvLandingReciprocalsSection(kKvLandingScaleMantissaMin,
		                                                             kExponentFloor,
		                                                             kKvLandingReciprocalMin)});
		SslmModelView view;
		std::string err;
		auto status = SslmModel::Load(built.bytes.data(), built.bytes.size(), view, &err);
		CHECK_MSG(status == SslmModelStatus::Ok,
		          "e_t=%lld (at the derived floor, exactly), R_t=%lld (canonical): "
		          "SslmModel::Load status == %s, want Ok (%s)",
		          static_cast<long long>(kExponentFloor), static_cast<long long>(kKvLandingReciprocalMin),
		          SslmModelStatusName(status), err.c_str());
		CHECK(view.has_kv_landing_reciprocals);
	}
}

// Record Sec4.5/Sec6.5: "the correction of MakeMinimalValidKvc1's near-
// INT64_MAX acceptance" (plan Sec7.2a's own text), realized as a cell rather
// than an edit to the shared fixture (editing it would break unrelated,
// already-green KVC1 sub-parse cells that hardcode its exact literal
// values, test-design record Sec4.5). `MakeMinimalValidKvc1`'s own gamma
// row -- reused here UNMODIFIED, wrapped as a KvLandingScales/
// KvLandingReciprocals section instead of CompositionConstants -- is
// definitively out of domain on both section types (gamma's mantissa is
// 2^62-1, and even alpha/beta's own mantissas are out of the canonical
// [2^30,2^31) range; gamma's own R-word is INT64_MAX), so this cell is
// overdetermined to fail once the real check lands -- exactly the
// correction owed.
static void TestKvLandingScalesAndReciprocalsRejectMakeMinimalValidKvc1sOwnGammaRow() {
	using namespace superslm_test;
	{
		auto m = MakeMinimalValidKvc1(/*value_words=*/2);
		auto built = BuildArtifact({MakeValidConfigSection(), MakeSigmoidLutSection(),
		                            MakeSection(SslmSectionType::KvLandingScales, SslmDtype::Raw, m.bytes,
		                                        /*alignment=*/64)});
		SslmModelView view;
		std::string err;
		auto status = SslmModel::Load(built.bytes.data(), built.bytes.size(), view, &err);
		CHECK_MSG(status == SslmModelStatus::KvLandingScaleOutOfDomain,
		          "MakeMinimalValidKvc1(2)'s own alpha/beta/gamma rows, wrapped as KvLandingScales: "
		          "SslmModel::Load status == %s, want KvLandingScaleOutOfDomain (%s) -- the correction "
		          "of this fixture's near-INT64_MAX acceptance",
		          SslmModelStatusName(status), err.c_str());
		CHECK(!view.has_kv_landing_scales);
	}
	{
		auto m = MakeMinimalValidKvc1(/*value_words=*/3);
		auto built = BuildArtifact({MakeValidConfigSection(), MakeSigmoidLutSection(),
		                            MakeSection(SslmSectionType::KvLandingReciprocals, SslmDtype::Raw,
		                                        m.bytes, /*alignment=*/64)});
		SslmModelView view;
		std::string err;
		auto status = SslmModel::Load(built.bytes.data(), built.bytes.size(), view, &err);
		CHECK_MSG(status == SslmModelStatus::KvLandingReciprocalOutOfDomain,
		          "MakeMinimalValidKvc1(3)'s own alpha/beta/gamma rows, wrapped as KvLandingReciprocals: "
		          "SslmModel::Load status == %s, want KvLandingReciprocalOutOfDomain (%s) -- the "
		          "correction of this fixture's near-INT64_MAX acceptance",
		          SslmModelStatusName(status), err.c_str());
		CHECK(!view.has_kv_landing_reciprocals);
	}
}

// ---------------------------------------------------------------------------
// S3.3 red-regression suite (Curie, 2026-07-28). The suite above is green at
// c314a64 and wrong: two independent seats confirmed six defects BY
// EXECUTION that the suite above never samples --
// Claude/Poirot/c314a64-s3.3-attention-interior-review-2026-07-28.md and
// Claude/Popper/superslm-c27-kv-landing-domain-bounds-debunk-2026-07-28.md.
// Every cell below fails against the shipped c314a64 code for its own
// reason and asserts the vendored reference's own value
// (tests/reference/superslm_spike/intmath.py, run directly by
// gen_s3_3_red_regression_fixtures.py -- never re-implemented) where a
// reference value exists.
//
// Test-design record:
// Claude/Curie/superslm-s3.3-attention-interior-red-regression-2026-07-28.md
// ---------------------------------------------------------------------------

// Finding 1 (CRITICAL, Poirot #1): CheckSoftmaxRowWidthDomain forms
// `q_b*q_b + q_c` in int64 (checked_chain_funnel.cpp:326) -- the exact
// computation intmath.h:391-395 documents as unsafe for a caller to perform
// ("the obvious check ... squares base in int64 and itself overflows ...
// Callers therefore use this predicate; they do not re-derive it",
// D-SLM81). The witness (m=2^30, e=-61) is one of 108 points in the
// canonical mantissa domain where (q_b, q_c) is fully int64-representable
// but q_b*q_b + q_c is not -- the predicate's own threshold computation
// overflows, and the row it exists to refuse is accepted.
static void TestCheckSoftmaxRowWidthDomainRejectsAWitnessWhoseOwnThresholdOverflowsInt64() {
	using namespace superslm_test;
	const auto& w = kSoftmaxRowOverflowWitness;
	const auto status = superslm::CheckSoftmaxRowWidthDomain(w.q_b, w.q_c, w.width);
	CHECK_MSG(status == superslm::SslmForwardStatus::SoftmaxRowWidthOutOfDomain,
	          "CheckSoftmaxRowWidthDomain(q_b=%lld, q_c=%lld, width=%zu) == %s, want "
	          "SoftmaxRowWidthOutOfDomain -- q_b*q_b + q_c (arbitrary-precision) exceeds both "
	          "INT64_MAX and kSoftmaxRowMaxSafeExponent (2^47), but the predicate forms this sum in "
	          "int64 and overflows, wrapping into a value no width exceeds "
	          "(Poirot 2026-07-28 finding 1, D-SLM81)",
	          static_cast<long long>(w.q_b), static_cast<long long>(w.q_c), w.width,
	          superslm::SslmForwardStatusName(status));
}

// Finding 2 (Popper Sec3.1): LandingRescale casts m_a to uint64_t
// unconditionally (forward_sites.cpp:172-181) on a comment claiming it is
// "positive by construction"; a mid-composition carried mantissa need only
// fit int32_t's range (checked_chain_funnel.h), so a negative m_a is
// reachable through the already-wired RmsNormSite/RequantChainChecked path
// from an artifact-legal CompositionConstants entry. The mutation this cell
// targets: the SAME operands with m_a's sign flipped must flip
// residual_reconcile's own result (it is odd-symmetric in m_a) -- the
// shipped sign handling does not, and falsely reports a clamp event that
// never happened.
static void TestLandingRescaleIsOddSymmetricInMAAgainstResidualReconcile() {
	using namespace superslm_test;
	const auto& w = kLandingNegativeMaWitness;

	const int64_t raw_pos = superslm::LandingRescale(w.branch_code, w.m_a_pos, w.r_t, w.e_a, w.e_t);
	CHECK_MSG(raw_pos == w.correct_raw_pos,
	          "positive-sign control: LandingRescale(branch_code=%lld, m_a=%lld, r_t=%lld, e_a=%d, "
	          "e_t=%d) == %lld, want %lld (residual_reconcile)",
	          static_cast<long long>(w.branch_code), static_cast<long long>(w.m_a_pos),
	          static_cast<long long>(w.r_t), w.e_a, w.e_t, static_cast<long long>(raw_pos),
	          static_cast<long long>(w.correct_raw_pos));

	uint64_t count = 0;
	const int64_t raw_neg =
	    superslm::LandingRescale(w.branch_code, w.m_a_neg, w.r_t, w.e_a, w.e_t, &count);
	CHECK_MSG(raw_neg == w.correct_raw_neg,
	          "LandingRescale(branch_code=%lld, m_a=%lld, r_t=%lld, e_a=%d, e_t=%d) == %lld, want "
	          "%lld (residual_reconcile) -- the shipped uint64_t cast of m_a on a false 'positive by "
	          "construction' precondition (Popper 2026-07-28 Sec3.1)",
	          static_cast<long long>(w.branch_code), static_cast<long long>(w.m_a_neg),
	          static_cast<long long>(w.r_t), w.e_a, w.e_t, static_cast<long long>(raw_neg),
	          static_cast<long long>(w.correct_raw_neg));
	CHECK_MSG(count == 0,
	          "the correct result (%lld) does not saturate the [-127, 127] clamp, so the saturation "
	          "counter must not fire -- count == %llu, want 0 (the shipped, wrongly-magnituded raw "
	          "value DOES exceed the clamp range, so it falsely increments the counter -- a clamp "
	          "event that never happened is reported as one)",
	          static_cast<long long>(w.correct_raw_neg), static_cast<unsigned long long>(count));
}

// Finding 3 (Popper Sec3.2): neither KVC1 landing exponent word (e_t, e_a)
// has any domain check anywhere in the tree -- artifact-carried or
// runtime-derived. An extreme e_t drives U128Shl's left-shift saturation
// (k = -938 at this witness), which silently returns 0 rather than the
// true, astronomically large magnitude -- and because `raw` comes back as
// 0, the saturation check (raw < -127 || raw > 127) never fires. This is
// the exact class D-SLM201's counter exists to catch, defeated by the same
// bug that produces the wrong answer.
static void TestLandingRescaleSaturationCounterFiresOnAnExtremeUncheckedExponent() {
	using namespace superslm_test;
	const auto& w = kLandingExtremeExponentWitness;

	uint64_t count = 0;
	const int64_t raw = superslm::LandingRescale(w.branch_code, w.m_a, w.r_t, w.e_a, w.e_t, &count);
	CHECK_MSG(count == 1,
	          "LandingRescale(branch_code=%lld, m_a=%lld, r_t=%lld, e_a=%d, e_t=%d): "
	          "*out_saturation_count == %llu, want 1 -- the true residual_reconcile result at this "
	          "operand set is a %d-bit %s integer, whose magnitude the [-127, 127] clamp must "
	          "saturate, so the counter must fire (D-SLM201, Popper 2026-07-28 Sec3.2). The shipped "
	          "result was raw=%lld -- a silently wrong 0, indistinguishable from a legitimately "
	          "near-zero branch value, because U128Shl saturates the left shift to {0,0} once the "
	          "shift count exceeds 127",
	          static_cast<long long>(w.branch_code), static_cast<long long>(w.m_a),
	          static_cast<long long>(w.r_t), w.e_a, w.e_t, static_cast<unsigned long long>(count),
	          w.reference_bit_length, w.reference_positive ? "positive" : "negative",
	          static_cast<long long>(raw));
}

// Finding 5 (Poirot #3), re-authored 2026-07-28 against the builder's own
// remediation (6eb1b76): SoftmaxRowQ15 now READS IExpConstruct's
// [[nodiscard]] outcome per element instead of discarding it, and returns
// `bool` -- false whenever any element's construction is kBadQ/kBadQLn2/
// kBadQB (intmath.h:480-498's own documented contract). This is the fix the
// finding asked for, at the KERNEL: CheckSoftmaxRowWidthDomain deliberately
// still does not take q_ln2 (it bounds q_b/q_c only; the header comment
// beside SoftmaxRowQ15's declaration states this is not a sufficient
// precondition BY DESIGN, and the kernel itself is the caller's own signal).
// The suite's prior cell asserted the pre-fix SYMPTOM directly
// (IExpConstruct(...) == kOk) -- an assertion no correct implementation can
// satisfy, since q_ln2 == 0 is genuinely invalid (intmath.h: "q_ln2 == 0
// divides by zero below"). Re-authored to assert the PROPERTY instead: on
// this witness, the composed path must REFUSE, observably, at the kernel.
//
// Witness: m=2^30, e=-10, one of 150 reachable (m, e) points where the real
// iexp_scale_constants degenerates to (q_ln2, q_b, q_c) = (0, 0, 0) -- the
// coarse-scale underflow tail.
//
// Vitality (executed 2026-07-28, disposable git worktree at 1b0bd10, the
// commit immediately before 6eb1b76): the pre-fix SoftmaxRowQ15 returned
// `void` and discarded IExpConstruct's outcome, so this witness produced
// `out_probs == {0, 0, 0}` with NO observable refusal signal at all -- a
// silently-plausible zero row, not a rejection. A cell asserting an
// explicit refusal genuinely could not have been expressed against that
// code, let alone pass; the bool return this cell requires is exactly what
// 6eb1b76 added.
static void TestSoftmaxRowQ15RefusesATripleWhoseIExpConstructionIsInvalid() {
	using namespace superslm_test;
	const auto& w = kSoftmaxQLn2ZeroWitness;

	// Grounding, not the pin: CheckSoftmaxRowWidthDomain is documented to
	// deliberately not cover q_ln2 (intmath.h:480-498), so it still accepts
	// this triple -- the kernel is where the refusal is now observable.
	const auto width_status = superslm::CheckSoftmaxRowWidthDomain(w.q_b, w.q_c, w.width);
	CHECK_MSG(width_status == superslm::SslmForwardStatus::Ok,
	          "premise: CheckSoftmaxRowWidthDomain(q_b=%lld, q_c=%lld, width=%zu) == %s, want Ok "
	          "(this predicate is documented to not cover q_ln2 by design -- if this no longer "
	          "holds, either the design changed or this witness needs re-deriving)",
	          static_cast<long long>(w.q_b), static_cast<long long>(w.q_c), w.width,
	          superslm::SslmForwardStatusName(width_status));

	std::vector<int64_t> scores(w.width, 0);
	std::vector<int64_t> out_probs(w.width, INT64_C(-99));  // poison
	const bool well_formed =
	    superslm::SoftmaxRowQ15(scores.data(), w.width, w.q_ln2, w.q_b, w.q_c, out_probs.data());
	CHECK_MSG(!well_formed,
	          "SoftmaxRowQ15(width=%zu, q_ln2=%lld, q_b=%lld, q_c=%lld) returned true (well-formed), "
	          "want false -- every element's IExpConstruct(q=0, q_ln2=0, ...) == kBadQLn2 (q_ln2 == 0 "
	          "has no valid decomposition), so the kernel must report the row untrustworthy rather "
	          "than silently succeeding (Poirot 2026-07-28 finding 3; fixed at intmath.cpp by 6eb1b76)",
	          w.width, static_cast<long long>(w.q_ln2), static_cast<long long>(w.q_b),
	          static_cast<long long>(w.q_c));
}

// ---------------------------------------------------------------------------
// Remediation-confirmation red suite (Curie, 2026-07-28). Claude/Poirot/
// ad6bd09-s3.3-remediation-confirmation-review-2026-07-28.md confirmed by
// execution that the pass which closed the six findings above introduced two
// more, in the same shape: the operand that is checked is the one that was
// easy to reach, and the operand that determines the answer is the one left
// open. Every cell below fails against the shipped ad6bd09 code for its own
// reason and asserts the vendored reference's own value
// (tests/reference/superslm_spike/intmath.py, run directly by
// gen_s3_3_red_regression_fixtures.py -- never re-implemented) where a
// reference value exists.
//
// Test-design record:
// Claude/Curie/superslm-s3.3-remediation-confirmation-red-regression-
// 2026-07-28.md
// ---------------------------------------------------------------------------

// New finding A (CRITICAL, Poirot ad6bd09 review): LandingRescale's
// round-divide branch (k >= 0, forward_sites.cpp:216-227) narrows a >64-bit
// true quotient through U128ShrToU64 with NO loss detection of its own --
// the identical class the left-shift branch's own fix (commit 3bbb11e)
// closed, left open here. That fix's own comment claims the opposite: "this
// branch needs no loss detection of its own." The witness below is one of
// 111 rows (Poirot's probe, this pass) where the true residual_reconcile
// result is a 69-bit integer and the shipped narrowing's low 64 bits land
// INSIDE [-127, 127] -- an ordinary-looking activation code with the WRONG
// SIGN, which is exactly why the symptom alone cannot catch this class:
// only the saturation counter can, and today it does not fire.
static void TestLandingRescaleSaturationCounterFiresOnRoundDivideBranchPrecisionLoss() {
	using namespace superslm_test;
	const auto& w = kLandingRoundDivideInBandWitness;

	uint64_t count = 0;
	const int64_t raw =
	    superslm::LandingRescale(w.branch_code, w.m_a, w.r_t, w.e_a, w.e_t, &count);
	CHECK_MSG(count == 1,
	          "LandingRescale(branch_code=%lld, m_a=%lld, r_t=%lld, e_a=%d, e_t=%d): "
	          "*out_saturation_count == %llu, want 1 -- the true residual_reconcile result at this "
	          "operand set is a %d-bit %s integer, far outside the [-127, 127] clamp, so the counter "
	          "must fire (D-SLM201). The shipped result was raw=%lld -- an IN-BAND code "
	          "indistinguishable from an ordinary activation (and, at this exact witness, the WRONG "
	          "SIGN too), because the round-divide branch (k=%lld >= 0) narrows the true >64-bit "
	          "quotient through U128ShrToU64 with no loss detection, unlike its sibling left-shift "
	          "branch (Poirot 2026-07-28 remediation-confirmation review, finding A)",
	          static_cast<long long>(w.branch_code), static_cast<long long>(w.m_a),
	          static_cast<long long>(w.r_t), w.e_a, w.e_t, static_cast<unsigned long long>(count),
	          w.reference_bit_length, w.reference_positive ? "positive" : "negative",
	          static_cast<long long>(raw), static_cast<long long>(62 - (w.e_a - w.e_t)));
}

// New finding B (Significant, Poirot ad6bd09 review): LandingRescale reads
// its `e_t` argument from KvLandingReciprocals word 1
// (Tools/superslm_spike/dynamic_engine.py:374-378: `_m_t, e_t, r_t =
// model.kv_landing_reciprocals[...]`), never from KvLandingScales. Commit
// 1b0bd10's load-time floor landed on KvLandingScales word 1 instead --
// finding 4's fix, verified above, closes that section, but the field the
// composite actually consumes, KvLandingReciprocals' own e_t, carries no
// domain check anywhere in the tree. An artifact whose R_t (the section's
// only checked word) is canonical but whose e_t is the same extreme
// witness this suite already proved silently wrong
// (kLandingExtremeExponentWitness.e_t) must be REJECTED at load time; today
// it is accepted regardless of what e_t carries, because no check anywhere
// reads word 1 of this section.
static void TestKvLandingReciprocalsLoadRejectsAnExtremeUncheckedExponentRegardlessOfRT() {
	using namespace superslm_test;
	const int64_t extreme_e_t = static_cast<int64_t>(kLandingExtremeExponentWitness.e_t);  // -1000

	for (int64_t r_t : {kKvLandingReciprocalMin, kKvLandingReciprocalMax}) {
		auto built = BuildArtifact(
		    {MakeValidConfigSection(), MakeSigmoidLutSection(),
		     MakeKvLandingReciprocalsSection(kKvLandingScaleMantissaMin, extreme_e_t, r_t)});
		SslmModelView view;
		std::string err;
		auto status = SslmModel::Load(built.bytes.data(), built.bytes.size(), view, &err);
		CHECK_MSG(status != SslmModelStatus::Ok,
		          "m_t=%lld (canonical, unchecked either way), e_t=%lld (extreme, unchecked "
		          "anywhere in the tree), R_t=%lld (canonical, the section's only checked word): "
		          "SslmModel::Load status == Ok, want a rejection -- LandingRescale's own e_t "
		          "operand is read from KvLandingReciprocals word 1, not KvLandingScales "
		          "(Tools/superslm_spike/dynamic_engine.py:374-378), and no check anywhere in this "
		          "tree validates it there (Poirot 2026-07-28 remediation-confirmation review, "
		          "finding B) -- R_t's own value must not change this artifact's fate",
		          static_cast<long long>(kKvLandingScaleMantissaMin), static_cast<long long>(extreme_e_t),
		          static_cast<long long>(r_t));
	}
}

// ---------------------------------------------------------------------------
// C32 softmax-row width-gate red suite (Curie, 2026-07-28). Popper's attack
// (Claude/Popper/superslm-c32-softmax-denominator-2026-07-28.md) killed the
// claim that `total` is safe and that the shipped
// `CheckSoftmaxRowWidthDomain(q_b, q_c, width)` correctly gates it, by three
// independent, executed mechanisms, each producing a triple the gate accepts
// (`Ok`) that is unsound: (1) the closed-form proxy `M = q_b*q_b + q_c` is
// not a valid upper bound on the row's real i-exp values outside an
// undocumented ratio the gate has no parameter to check; (2) `M == 0` skips
// the sum-width check entirely, at any width; (3) `M < 0` defeats the sum
// check through a signed-to-size_t cast, at any width. Every cell below
// asserts the PROPERTY -- this row must be refused, and no row may be
// reported well-formed while carrying an out-of-range probability -- not one
// particular internal formula, so a correct re-shaping of the predicate
// (e.g. taking q_ln2 and checking the shortcut ratio, or gating on the row's
// real computed peak the way tests/reference/superslm_spike's own
// _vec_softmax does) greens every cell here and a second proxy does not.
//
// Test-design record:
// Claude/Curie/superslm-c32-softmax-row-width-gate-test-design-2026-07-28.md
// ---------------------------------------------------------------------------

// Defect 1 (Popper Null 1): CheckSoftmaxRowWidthDomain must not accept a
// triple whose row REALLY carries an element this many orders of magnitude
// past the D-SLM367 ceiling (2^47) -- the closed form M it actually checks
// (== 0 on this witness) is not a valid stand-in for that real peak off the
// undocumented shortcut ratio `2*q_b >= q_ln2 - 1`, and the predicate has no
// q_ln2 parameter with which to check that ratio. This cell targets the gate
// alone, before any kernel call, so a fix that rejects here (rather than
// only downstream at the kernel) is recognized as sound.
static void TestCheckSoftmaxRowWidthDomainAcceptsAWitnessWhoseRealPeakVastlyExceedsTheCeiling() {
	using namespace superslm_test;
	const auto& w = kSoftmaxRowOffRatioWitness;

	const auto status = superslm::CheckSoftmaxRowWidthDomain(w.q_b, w.q_c, w.width);
	CHECK_MSG(status != superslm::SslmForwardStatus::Ok,
	          "CheckSoftmaxRowWidthDomain(q_b=%lld, q_c=%lld, width=%zu) == %s, want a rejection -- "
	          "this row's real per-element i-exp peak (computed by the vendored reference's "
	          "i_exp_from_constants) exceeds PROB_WIDTH_CEILING (2^47) by roughly 9x10^16, but the "
	          "gate's own closed-form proxy M = q_b*q_b + q_c == 0 on this witness and is not a valid "
	          "stand-in for that peak off the undocumented shortcut ratio the gate cannot check "
	          "(Popper 2026-07-28 Null 1)",
	          static_cast<long long>(w.q_b), static_cast<long long>(w.q_c), w.width,
	          superslm::SslmForwardStatusName(status));
}

// Defect 1 + the end-state consequence (Popper Null 1/Null 3), pinned as one
// composed property so a fix is free to catch this witness at either stage:
// the row must never reach a state where SoftmaxRowQ15 reports it
// well-formed while ANY reported probability falls outside the format's own
// [0, 32768] range (kProbFracBits == 15). `total` is independently
// replayed here from the same shipped IExpConstruct/IExpEvaluate primitives
// SoftmaxRowQ15 itself calls (SoftmaxRowQ15 does not expose its internal
// accumulator) and bit-compared against the vendored reference's
// exact-precision sum, wrapped to int64_t -- grounding "genuinely overflows"
// in execution, not assertion.
static void TestSoftmaxRowQ15NeverReportsWellFormedWithAnOutOfRangeProbability() {
	using namespace superslm_test;
	using superslm::IExpConstruct;
	using superslm::IExpConstruction;
	using superslm::IExpDomain;
	using superslm::IExpEvaluate;
	const auto& w = kSoftmaxRowOffRatioWitness;

	// Grounding: replay the same accumulation SoftmaxRowQ15's own loop
	// performs, bit-compared against the vendored reference's exact-precision
	// sum wrapped to int64_t (Popper Null 3).
	int64_t replayed_total = 0;
	for (size_t k = 0; k < w.width; ++k) {
		IExpConstruction c;
		const IExpDomain d = IExpConstruct(w.scores[k], w.q_ln2, w.q_b, w.q_c, &c);
		CHECK_MSG(d == IExpDomain::kOk || d == IExpDomain::kNotRepresentable,
		          "grounding: IExpConstruct(scores[%zu]=%lld, q_ln2=%lld, q_b=%lld, q_c=%lld) domain "
		          "== %d, want kOk or kNotRepresentable -- this witness's own premise (every element "
		          "individually constructs) no longer holds",
		          k, static_cast<long long>(w.scores[k]), static_cast<long long>(w.q_ln2),
		          static_cast<long long>(w.q_b), static_cast<long long>(w.q_c), static_cast<int>(d));
		replayed_total += IExpEvaluate(c);
	}
	CHECK_MSG(replayed_total == w.exact_total_wrapped_i64,
	          "grounding: replayed int64 total (via the real shipped IExpConstruct/IExpEvaluate) == "
	          "%lld, want %lld (the vendored reference's exact-precision sum, two's-complement "
	          "wrapped) -- SoftmaxRowQ15's own accumulation must genuinely overflow int64 on this "
	          "witness, not merely be asserted to",
	          static_cast<long long>(replayed_total), static_cast<long long>(w.exact_total_wrapped_i64));

	// The property: gate, then kernel -- if the gate accepts, the kernel must
	// never report well-formed alongside an out-of-range probability. Any of
	// three outcomes is sound: the gate rejects; the kernel reports
	// !well_formed; or the kernel reports well_formed with every probability
	// inside [0, 32768]. Only "gate Ok, kernel well_formed, an out-of-range
	// probability" is the defect.
	const auto gate_status = superslm::CheckSoftmaxRowWidthDomain(w.q_b, w.q_c, w.width);
	std::vector<int64_t> out_probs(w.width, INT64_C(-99));  // poison
	bool well_formed = false;
	if (gate_status == superslm::SslmForwardStatus::Ok) {
		well_formed =
		    superslm::SoftmaxRowQ15(w.scores, w.width, w.q_ln2, w.q_b, w.q_c, out_probs.data());
	}
	bool any_out_of_range = false;
	for (size_t k = 0; k < w.width; ++k) {
		if (out_probs[k] < 0 || out_probs[k] > (INT64_C(1) << superslm::kProbFracBits)) {
			any_out_of_range = true;
		}
	}
	const bool sound = gate_status != superslm::SslmForwardStatus::Ok || !well_formed || !any_out_of_range;
	CHECK_MSG(sound,
	          "CheckSoftmaxRowWidthDomain(q_b=%lld, q_c=%lld, width=%zu) == Ok, then "
	          "SoftmaxRowQ15(...) returned well_formed=%d with out_probs={%lld, %lld, %lld} -- a "
	          "probability outside [0, %lld] reported alongside well_formed==true must be impossible "
	          "(Popper 2026-07-28 Null 3: denom collapses to 1 because the wrapped negative total is "
	          "indistinguishable from a genuine all-clipped row, then exps[k] << kProbFracBits "
	          "overflows a second, independent time)",
	          static_cast<long long>(w.q_b), static_cast<long long>(w.q_c), w.width, (int)well_formed,
	          static_cast<long long>(out_probs[0]), static_cast<long long>(out_probs[1]),
	          static_cast<long long>(out_probs[2]), static_cast<long long>(INT64_C(1) << superslm::kProbFracBits));
}

// Defect 2 (Popper Null 2, first bullet): the m == 0 short-circuit
// (`m != 0 && width > ...`) that skips the sum-width check entirely must not
// be a width=3 coincidence -- the same (q_b, q_c) pair must still be refused
// at a width near the artifact format's own admitted context_cap ceiling
// (uint32_t, model.h + model.cpp:463's non-zero-only load-time validation).
static void TestCheckSoftmaxRowWidthDomainMZeroBypassIsIndependentOfWidth() {
	using namespace superslm_test;
	const auto& w = kSoftmaxRowMZeroWideWitness;

	const auto status = superslm::CheckSoftmaxRowWidthDomain(w.q_b, w.q_c, w.width);
	CHECK_MSG(status != superslm::SslmForwardStatus::Ok,
	          "CheckSoftmaxRowWidthDomain(q_b=%lld, q_c=%lld, width=%zu) == %s, want a rejection -- "
	          "M = q_b*q_b + q_c == 0 for this (q_b, q_c) pair (kSoftmaxRowOffRatioWitness's own "
	          "unsound premise), and the `m != 0` guard skips the sum-width check unconditionally, "
	          "at ANY width up to the artifact format's own admitted context_cap ceiling "
	          "(Popper 2026-07-28 Null 2)",
	          static_cast<long long>(w.q_b), static_cast<long long>(w.q_c), w.width,
	          superslm::SslmForwardStatusName(status));
}

// Defect 3 (Popper Null 2, second bullet): the sum limb must not become MORE
// permissive when M's sign flips negative at fixed magnitude. A positive M
// of magnitude m_magnitude (the largest value representable under the
// D-SLM367 ceiling, 2^47 - 1) is genuinely rejected by the sum limb at
// `width` (independently confirmed below, not assumed) -- a negative M of
// the SAME magnitude must not be treated as safer than its positive
// counterpart, since a computed "peak" can never legitimately be negative
// and a negative M carries no more assurance than a positive one of equal
// size.
static void TestCheckSoftmaxRowWidthDomainMustNotBeMorePermissiveForNegativeMThanPositiveOfEqualMagnitude() {
	using namespace superslm_test;
	const auto& w = kSoftmaxRowSignAsymmetryWitness;

	// q_b = 0 isolates M = q_c, so the same magnitude is realized on both
	// signs by flipping q_c's own sign alone.
	const auto positive_status =
	    superslm::CheckSoftmaxRowWidthDomain(/*q_b=*/0, /*q_c=*/w.m_magnitude, w.width);
	CHECK_MSG(positive_status == superslm::SslmForwardStatus::SoftmaxRowWidthOutOfDomain,
	          "control: CheckSoftmaxRowWidthDomain(q_b=0, q_c=%lld, width=%zu) == %s, want "
	          "SoftmaxRowWidthOutOfDomain -- this positive-M control must itself be genuinely "
	          "rejected by the sum limb (width was derived from INT64_MAX // m_magnitude), or the "
	          "asymmetry this cell targets is not actually demonstrated",
	          static_cast<long long>(w.m_magnitude), w.width, superslm::SslmForwardStatusName(positive_status));

	const auto negative_status =
	    superslm::CheckSoftmaxRowWidthDomain(/*q_b=*/0, /*q_c=*/-w.m_magnitude, w.width);
	CHECK_MSG(negative_status == superslm::SslmForwardStatus::SoftmaxRowWidthOutOfDomain,
	          "CheckSoftmaxRowWidthDomain(q_b=0, q_c=%lld, width=%zu) == %s, want "
	          "SoftmaxRowWidthOutOfDomain (matching the positive-M control of the SAME magnitude at "
	          "the SAME width, which is genuinely rejected) -- the negative-M sum check "
	          "(`INT64_MAX / m` with m < 0, cast to size_t) wraps to a huge unsigned value and passes "
	          "`width > (huge value)` trivially, defeating the check the positive-sign case correctly "
	          "enforces (Popper 2026-07-28 Null 2, second bullet)",
	          static_cast<long long>(-w.m_magnitude), w.width, superslm::SslmForwardStatusName(negative_status));
}

// Significant 3 (Poirot fa3189a-s3.3-rope-site-and-c32-softmax-review-2026-07-28.md,
// closed by this cell): the kernel half of the C32 remediation -- SoftmaxRowQ15's own
// per-element ceiling check, refusing any element whose REAL evaluated value exceeds
// M -- was pinned by no cell before this one. Every OTHER witness in this suite
// carries q_c <= 0, so CheckSoftmaxRowWidthDomain's own `q_c < 0` rejection
// (checked_chain_funnel.cpp) satisfies TestSoftmaxRowQ15NeverReportsWellFormedWithAn
// OutOfRangeProbability unconditionally via a GATE rejection, never reaching the
// kernel's per-element check. THIS cell is the missing off-ratio witness WITH
// q_c >= 0: the gate accepts it (M = q_b*q_b + q_c is trivially inside the ratified
// ceiling), so only the kernel's own check stands between this row and a reported
// well-formed result carrying a value ~9e16 past the D-SLM367 ceiling.
//
// This is not a red-before-green cell -- the kernel check already exists and is
// already correct (D-SLM380). This cell is a coverage pin: it passes today because
// the shipped check does its job, and it is mutation-proved by deleting
// src/intmath.cpp:598-602 in a scratch copy outside the repository and confirming
// this cell (and no other) newly fails.
static void TestSoftmaxRowQ15RejectsOffRatioWitnessWithNonnegativeQcThatPassesTheGate() {
	using namespace superslm_test;
	using superslm::IExpConstruct;
	using superslm::IExpConstruction;
	using superslm::IExpDomain;
	using superslm::IExpEvaluate;
	const auto& w = kSoftmaxRowOffRatioNonnegativeQcWitness;

	// Grounding: the gate must accept this witness, or the kernel is never reached and
	// this cell proves nothing about it.
	const auto gate_status = superslm::CheckSoftmaxRowWidthDomain(w.q_b, w.q_c, w.width);
	CHECK_MSG(gate_status == superslm::SslmForwardStatus::Ok,
	          "grounding: CheckSoftmaxRowWidthDomain(q_b=%lld, q_c=%lld, width=%zu) == %s, want Ok -- "
	          "this witness's own premise (q_c >= 0 clears the gate's own rejection, so only the "
	          "kernel's per-element check stands between this row and a reported well-formed result) "
	          "no longer holds",
	          static_cast<long long>(w.q_b), static_cast<long long>(w.q_c), w.width,
	          superslm::SslmForwardStatusName(gate_status));

	// Grounding: element k=1 must genuinely evaluate past M through the real shipped
	// IExpConstruct/IExpEvaluate -- computed by calling them, not asserted.
	IExpConstruction construction;
	const IExpDomain d = IExpConstruct(w.scores[1], w.q_ln2, w.q_b, w.q_c, &construction);
	CHECK_MSG(d == IExpDomain::kOk || d == IExpDomain::kNotRepresentable,
	          "grounding: IExpConstruct(scores[1]=%lld, q_ln2=%lld, q_b=%lld, q_c=%lld) domain == %d, "
	          "want kOk or kNotRepresentable -- this witness's own premise (element k=1 individually "
	          "constructs) no longer holds",
	          static_cast<long long>(w.scores[1]), static_cast<long long>(w.q_ln2),
	          static_cast<long long>(w.q_b), static_cast<long long>(w.q_c), static_cast<int>(d));
	const int64_t real_value = IExpEvaluate(construction);
	const int64_t m = w.q_b * w.q_b + w.q_c;
	CHECK_MSG(real_value > m,
	          "grounding: the real evaluated i-exp value at scores[1] == %lld, want > M(=%lld) -- this "
	          "witness's own off-ratio premise no longer holds",
	          static_cast<long long>(real_value), static_cast<long long>(m));

	// The property this cell exists to pin: SoftmaxRowQ15 must not report this row
	// well-formed. Deleting src/intmath.cpp:598-602 removes the only check that can
	// refuse element k=1 here (the gate already accepted; IExpConstruct already
	// constructed) -- so this assertion is exactly what that deletion flips.
	std::vector<int64_t> out_probs(w.width, INT64_C(-99));  // poison
	const bool well_formed =
	    superslm::SoftmaxRowQ15(w.scores, w.width, w.q_ln2, w.q_b, w.q_c, out_probs.data());
	CHECK_MSG(!well_formed,
	          "SoftmaxRowQ15(q_b=%lld, q_c=%lld, q_ln2=%lld) reported well_formed=true for a row whose "
	          "real evaluated element at k=1 (%lld) exceeds M (%lld) -- the kernel's own per-element "
	          "ceiling check (src/intmath.cpp:598-602) must refuse this row",
	          static_cast<long long>(w.q_b), static_cast<long long>(w.q_c),
	          static_cast<long long>(w.q_ln2), static_cast<long long>(real_value),
	          static_cast<long long>(m));
}

// T-1324 (BLOCKING; D-SLM409; Poirot 72b0c7f-s3.3-rope-site-and-c32-softmax-
// confirmation-2026-07-28.md, Significant 4): `m_usable` (src/intmath.cpp:566)
// admits any M representable in int64_t and >= 1 -- it does not bound M against
// `kSoftmaxRowMaxSafeExponent` (2**47), the ceiling intmath.h:521-528's own
// contract attributes the `exps[k] << kProbFracBits` shift's safety to. Plan
// Sec14.1 owes exactly this bound, "on `e[k] << PROB_FRAC_BITS`", before the
// kernel greens; the kernel has greened and the bound is neither derived nor
// enforced.
//
// A width-gate rejection already exists for this M (CheckSoftmaxRowWidthDomain,
// checked_chain_funnel.cpp:409) -- this cell reaches the kernel the way the
// header's own documented "ungated caller" case does: calling SoftmaxRowQ15
// directly, bypassing the gate. The gate's own rejection is grounded first, so
// the cell cannot be satisfied by accident of the gate being reachable here too.
static void TestSoftmaxRowQ15MustNotReportWellFormedWhenShiftedMaxElementExceedsTheSafeShiftCeiling() {
	using namespace superslm_test;
	using superslm::IExpConstruct;
	using superslm::IExpConstruction;
	using superslm::IExpDomain;
	using superslm::IExpEvaluate;
	const auto& w = kSoftmaxRowUngatedShiftOverflowWitness;

	// Grounding 1: this M is genuinely outside the ratified ceiling, and the width gate
	// genuinely rejects it -- so the only way to reach the kernel with this M is to call it
	// directly, ungated, exactly as the header's own "caller that skips the gate" case
	// describes. If the gate stopped rejecting this M, the witness would no longer isolate
	// the ungated-caller path this cell targets.
	CHECK_MSG(w.m > superslm::kSoftmaxRowMaxSafeExponent,
	          "grounding: witness M (%lld) is not greater than kSoftmaxRowMaxSafeExponent (%lld) -- "
	          "this witness's own premise (M exceeds the ratified safe ceiling) no longer holds",
	          static_cast<long long>(w.m), static_cast<long long>(superslm::kSoftmaxRowMaxSafeExponent));
	const auto gate_status = superslm::CheckSoftmaxRowWidthDomain(w.q_b, w.q_c, w.width);
	CHECK_MSG(gate_status == superslm::SslmForwardStatus::SoftmaxRowWidthOutOfDomain,
	          "grounding: CheckSoftmaxRowWidthDomain(q_b=%lld, q_c=%lld, width=%zu) == %s, want "
	          "SoftmaxRowWidthOutOfDomain -- this witness's own premise (only an ungated direct call "
	          "can reach the kernel with this M) no longer holds",
	          static_cast<long long>(w.q_b), static_cast<long long>(w.q_c), w.width,
	          superslm::SslmForwardStatusName(gate_status));

	// Grounding 2: the row's single element genuinely constructs and its real evaluated
	// value genuinely equals M exactly -- computed by calling the real shipped
	// IExpConstruct/IExpEvaluate, not asserted. This isolates the missing bound: the
	// element itself is not "off-ratio" or otherwise malformed, it is precisely the row's
	// own claimed peak, which is the exact value the shift is supposed to be safe up to.
	IExpConstruction construction;
	const IExpDomain d = IExpConstruct(w.scores[0], w.q_ln2, w.q_b, w.q_c, &construction);
	CHECK_MSG(d == IExpDomain::kOk || d == IExpDomain::kNotRepresentable,
	          "grounding: IExpConstruct(scores[0]=%lld, q_ln2=%lld, q_b=%lld, q_c=%lld) domain == %d, "
	          "want kOk or kNotRepresentable -- this witness's own premise (the shifted-max element "
	          "individually constructs) no longer holds",
	          static_cast<long long>(w.scores[0]), static_cast<long long>(w.q_ln2),
	          static_cast<long long>(w.q_b), static_cast<long long>(w.q_c), static_cast<int>(d));
	const int64_t real_value = IExpEvaluate(construction);
	CHECK_MSG(real_value == w.m,
	          "grounding: the real evaluated i-exp value at scores[0] == %lld, want exactly M (%lld) -- "
	          "this witness's own premise (the shifted-max element evaluates to exactly M, isolating "
	          "the shift from the row's other arithmetic) no longer holds",
	          static_cast<long long>(real_value), static_cast<long long>(w.m));

	// The property this cell exists to pin: a row whose only element's real value is this
	// far past kSoftmaxRowMaxSafeExponent must not be reported well-formed. `m_usable`'s
	// existing two conjuncts (representable in int64_t, and >= 1) both pass for this M --
	// only a third conjunct bounding M against kSoftmaxRowMaxSafeExponent closes this. The
	// buggy value this defect actually produces (out_probs[0] == 0 for a single-element row,
	// which must be exactly full scale) is named in the failure message as the concrete
	// consequence, not asserted directly -- the property this cell pins is the well-formed
	// flag's own honesty, which is what the documented fix (adding the third conjunct)
	// closes.
	std::vector<int64_t> out_probs(w.width, INT64_C(-99));  // poison
	const bool well_formed =
	    superslm::SoftmaxRowQ15(w.scores, w.width, w.q_ln2, w.q_b, w.q_c, out_probs.data());
	const int64_t full_scale = INT64_C(1) << superslm::kProbFracBits;
	CHECK_MSG(well_formed == false,
	          "SoftmaxRowQ15(q_b=%lld, q_c=%lld, q_ln2=%lld, width=1) reported well_formed=true for a "
	          "row whose only element's real value (%lld) equals M (%lld), which exceeds "
	          "kSoftmaxRowMaxSafeExponent (%lld) -- `m_usable` (src/intmath.cpp:566) must also bound M "
	          "against this ceiling, or the reported out_probs[0] (%lld) is a wrong answer (a "
	          "single-element row's probability must be exactly full scale, %lld) delivered under a "
	          "well-formed flag asserting the answer is trustworthy (plan Sec14.1; D-SLM409)",
	          static_cast<long long>(w.q_b), static_cast<long long>(w.q_c),
	          static_cast<long long>(w.q_ln2), static_cast<long long>(real_value),
	          static_cast<long long>(w.m), static_cast<long long>(superslm::kSoftmaxRowMaxSafeExponent),
	          static_cast<long long>(out_probs[0]), static_cast<long long>(full_scale));
}

// ---------------------------------------------------------------------------
// S3.3 -- the RoPE APPLICATION SITE (Claude/Plans/SuperSLM_S3a_WalkingSkeleton_
// Plan.md Sec6.2 step 3, Sec11 S3.3's own gate line; D-SLM376, D-SLM383).
//
// SUPERSEDED as of D-SLM387 (2026-07-28), retained only to explain why the
// cells below are ordered as they are. This block described the state at
// D-SLM376's authoring, when the tree shipped `RopeApplyPair`, `ClampRopeCode`
// and `CheckPositionOverCap` separately and NO production symbol composed
// them. That is no longer true: `RopeApplySite` (forward_sites.h, body in
// src/forward/forward_sites.cpp) is built and green, with
// CheckPositionOverCap as its first act, and the site's own feature-oracle,
// "never a table read" ordering, and guard-vitality (ASan) cells ARE authored
// -- see the block at the next section boundary below. A reader arriving here
// first should not conclude the site is unbuilt.
//
// What is authored immediately below, real and executed against already-shipped
// primitives (the Sec5 pattern this campaign's prior passes already
// established): CheckPositionOverCap's own boundary matrix (built, correct,
// and until this pass exercised by zero C++ cells -- only a Python text-scan
// proving its header comment no longer calls it a "stub"); RopeApplyPair +
// ClampRopeCode composed, pairwise, exactly as the site's own contract
// states (Sec6.2 step 3) and exactly as the reference's `_rotate_rows`
// composes them (dynamic_engine.py:243-252), at position 0 and position
// (context_cap - 1) -- Sec13 dim 6's own named cell, "RoPE at position 0 and
// context_cap-1"; and the ROP1 artifact round-trip, proving the table the
// site will read is carried byte-for-byte through the real loader.
// ---------------------------------------------------------------------------

// Sec6.2 step 3 / Sec11 S3.3's gate line / Sec13 dim 5,11: `CheckPositionOverCap`
// rejects with PositionOverCap for every position outside `[0, context_cap)` --
// an EXCLUSIVE upper bound (position == context_cap is one past the last valid
// slot, matching the plan's own wording). Real, already-shipped, already-correct
// (Poirot ad6bd09-s3.3-remediation-confirmation-review-2026-07-28.md finding D)
// -- this cell is the first C++ exercise of it directly, closing the gap that
// only a Python source-text-scan (tests/ci/check_checked_chain_funnel_position_
// cap_not_a_stub.py) had ever driven it before this pass.
static void TestCheckPositionOverCapBoundaryMatrixAcrossMultipleCaps() {
	using superslm::SslmForwardStatus;
	struct Point {
		int64_t position;
		int64_t context_cap;
		bool in_domain;
		const char* label;
	};
	const Point points[] = {
	    // context_cap == 1: the minimal legal cap -- only position 0 is valid.
	    {0, 1, true, "cap=1, position=0 (only valid slot)"},
	    {1, 1, false, "cap=1, position=1 (== cap, one past the last valid slot)"},
	    {-1, 1, false, "cap=1, position=-1"},
	    // A realistic cap (the pinned RoPE table's own context_cap, 128).
	    {0, superslm_test::kRopeSitePinnedContextCap, true, "cap=128, position=0"},
	    {superslm_test::kRopeSitePinnedContextCap - 1, superslm_test::kRopeSitePinnedContextCap, true,
	     "cap=128, position=127 (context_cap - 1, the last valid slot)"},
	    {superslm_test::kRopeSitePinnedContextCap, superslm_test::kRopeSitePinnedContextCap, false,
	     "cap=128, position=128 (== cap, one past the last valid slot)"},
	    {superslm_test::kRopeSitePinnedContextCap + 1, superslm_test::kRopeSitePinnedContextCap, false,
	     "cap=128, position=129 (> cap)"},
	    {-1, superslm_test::kRopeSitePinnedContextCap, false, "cap=128, position=-1"},
	    // Full int64 extremes, at a realistic cap -- a host/runtime-supplied
	    // position is untrusted input, and the check must not special-case a
	    // magnitude anywhere near overflow.
	    {INT64_MAX, superslm_test::kRopeSitePinnedContextCap, false, "cap=128, position=INT64_MAX"},
	    {INT64_MIN, superslm_test::kRopeSitePinnedContextCap, false, "cap=128, position=INT64_MIN"},
	};
	for (const Point& p : points) {
		const auto status = superslm::CheckPositionOverCap(p.position, p.context_cap);
		if (p.in_domain) {
			CHECK_MSG(status == SslmForwardStatus::Ok,
			          "%s: CheckPositionOverCap(position=%lld, context_cap=%lld) == %s, want Ok",
			          p.label, static_cast<long long>(p.position), static_cast<long long>(p.context_cap),
			          superslm::SslmForwardStatusName(status));
		} else {
			CHECK_MSG(status == SslmForwardStatus::PositionOverCap,
			          "%s: CheckPositionOverCap(position=%lld, context_cap=%lld) == %s, want PositionOverCap",
			          p.label, static_cast<long long>(p.position), static_cast<long long>(p.context_cap),
			          superslm::SslmForwardStatusName(status));
		}
	}
}

// Sec6.2 step 3 / Sec13 dim 6 ("RoPE at position 0 and context_cap-1"): composes
// the real, already-shipped `RopeApplyPair` + `ClampRopeCode` pairwise over one
// fixture case, exactly as the (not-yet-built) site's own contract states and
// exactly as the reference's `_rotate_rows` composes them (dynamic_engine.py:
// 243-252, read at source): pairs are (row[2*i], row[2*i+1]) -- interleaved
// even/odd, never a first-half/second-half split.
static void CheckRopeSitePositionCase(const superslm_test::RopeSitePositionCase& c) {
	using superslm::RopeApplyPair;
	using superslm::ClampRopeCode;
	int64_t got[128];
	for (int i = 0; i < superslm_test::kRopeSitePinnedPairs; ++i) {
		const int32_t x = static_cast<int32_t>(c.row[2 * i]);
		const int32_t y = static_cast<int32_t>(c.row[2 * i + 1]);
		const int32_t cos_q30 = static_cast<int32_t>(c.cos_row[i]);
		const int32_t sin_q30 = static_cast<int32_t>(c.sin_row[i]);
		const auto rotated = RopeApplyPair(x, y, cos_q30, sin_q30);
		got[2 * i] = ClampRopeCode(rotated.x);
		got[2 * i + 1] = ClampRopeCode(rotated.y);
	}
	for (int i = 0; i < superslm_test::kRopeSitePinnedHeadDim; ++i) {
		CHECK_MSG(got[i] == c.expected_out[i],
		          "%s: pair-composed RopeApplyPair+ClampRopeCode at element %d == %lld, want %lld "
		          "(reference: tests/reference/superslm_spike/rope.py's rope_apply_pair, via "
		          "tests/gen_s3_3_rope_application_site_fixtures.py)",
		          c.label, i, static_cast<long long>(got[i]), static_cast<long long>(c.expected_out[i]));
	}
}

static void TestRopeSitePositionZeroAndCapMinusOneAgainstRealPrimitives() {
	CheckRopeSitePositionCase(superslm_test::kRopeSitePositionZeroCase);
	CheckRopeSitePositionCase(superslm_test::kRopeSitePositionCapMinusOneCase);
}

// Sec13.1 cell 9's sibling join for the RoPE table itself: the real, pinned
// cos/sin rows the site will eventually read are carried byte-for-byte
// through the real `SslmModel::Load`, at every row a small (context_cap=4)
// artifact declares -- proving the artifact-carries-the-real-table half of
// the site's own join independently of the site itself (which does not yet
// exist to be driven end to end).
static superslm_test::FixtureSection MakeRop1SectionMultiRow(int32_t context_cap, int32_t pairs,
                                                               const int64_t* cos_flat, const int64_t* sin_flat) {
	using namespace superslm_test;
	std::vector<ManifestTensorSpec> tensors = {
	    {"cos", {static_cast<uint32_t>(context_cap), static_cast<uint32_t>(pairs)}},
	    {"sin", {static_cast<uint32_t>(context_cap), static_cast<uint32_t>(pairs)}},
	};
	auto manifest = BuildManifest(superslm::kRopeMagic, /*element_size=*/8, tensors);
	const size_t n = static_cast<size_t>(context_cap) * static_cast<size_t>(pairs);
	for (size_t i = 0; i < n; ++i) {
		PutU64(manifest.bytes, static_cast<size_t>(manifest.tensor_data_off[0]) + i * 8,
		       static_cast<uint64_t>(cos_flat[i]));
		PutU64(manifest.bytes, static_cast<size_t>(manifest.tensor_data_off[1]) + i * 8,
		       static_cast<uint64_t>(sin_flat[i]));
	}
	return MakeSection(SslmSectionType::RopeTables, SslmDtype::Int64, manifest.bytes, /*alignment=*/64);
}

static void TestRopeTableSectionRoundTripsThroughRealLoadAtEveryPinnedRow() {
	using namespace superslm_test;
	const int32_t cap = kRopeSiteRoundTripContextCap;
	const int32_t pairs = kRopeSiteRoundTripPairs;

	Cfg1Spec spec{};
	spec.context_cap = static_cast<uint32_t>(cap);
	spec.head_dim = static_cast<uint32_t>(kRopeSiteRoundTripHeadDim);
	FixtureSection config = MakeSection(SslmSectionType::Config, SslmDtype::Raw, BuildCfg1(spec));

	FixtureSection rope_tables =
	    MakeRop1SectionMultiRow(cap, pairs, kRopeSiteRoundTripCosFlat, kRopeSiteRoundTripSinFlat);
	auto built = BuildArtifact({config, MakeSigmoidLutSection(), rope_tables});

	SslmModelView view;
	std::string err;
	SslmModelStatus status = SslmModel::Load(built.bytes.data(), built.bytes.size(), view, &err);
	CHECK_MSG(status == SslmModelStatus::Ok,
	          "the round-trip fixture (real pinned RoPE table rows, context_cap=%d, head_dim=%d) failed to "
	          "load: got %s (%s)",
	          cap, kRopeSiteRoundTripHeadDim, SslmModelStatusName(status), err.c_str());
	if (status != SslmModelStatus::Ok) return;
	CHECK_MSG(view.has_rope_tables, "a successfully loaded artifact must expose its RopeTables view");
	if (!view.has_rope_tables) return;

	const SslmTensorView* cos = view.rope_tables.Tensor("cos");
	const SslmTensorView* sin = view.rope_tables.Tensor("sin");
	CHECK_MSG(cos != nullptr && sin != nullptr, "the loaded RopeTables view is missing \"cos\" or \"sin\"");
	if (cos == nullptr || sin == nullptr) return;
	CHECK(cos->elem_count == static_cast<uint64_t>(cap) * static_cast<uint64_t>(pairs));
	CHECK(sin->elem_count == static_cast<uint64_t>(cap) * static_cast<uint64_t>(pairs));

	for (int64_t position = 0; position < cap; ++position) {
		for (int32_t col = 0; col < pairs; ++col) {
			const size_t idx = static_cast<size_t>(position) * static_cast<size_t>(pairs) + static_cast<size_t>(col);
			const int64_t got_cos = ReadRawI64LE(cos->data + idx * 8);
			const int64_t got_sin = ReadRawI64LE(sin->data + idx * 8);
			CHECK_MSG(got_cos == kRopeSiteRoundTripCosFlat[idx],
			          "position=%lld col=%d: loaded ROP1 cos element == %lld, want %lld (the real pinned "
			          "reference table's own value)",
			          static_cast<long long>(position), col, static_cast<long long>(got_cos),
			          static_cast<long long>(kRopeSiteRoundTripCosFlat[idx]));
			CHECK_MSG(got_sin == kRopeSiteRoundTripSinFlat[idx],
			          "position=%lld col=%d: loaded ROP1 sin element == %lld, want %lld (the real pinned "
			          "reference table's own value)",
			          static_cast<long long>(position), col, static_cast<long long>(got_sin),
			          static_cast<long long>(kRopeSiteRoundTripSinFlat[idx]));
		}
	}
}

// ---------------------------------------------------------------------------
// §13.1 cell 4 (D-SLM410, D-SLM420-D-SLM423, board T-1333): the config-geometry
// x tensor-shape join at SslmModel::Load. R1-R3 (config-geometry) and R4
// (ROP1<->CFG1) each get their own hostile cell below, plus the positive
// control the plan names. Every symbol asserted against
// (ConfigGeometryKvHeadsExceedsHeads/ConfigGeometryHeadsNotDivisibleByKv/
// ConfigGeometryHiddenSizeMismatch/RopeTablesShapeMismatchConfig,
// ValidateConfigGeometryJoin/ValidateRopeTablesShapeAgainstConfig wired into
// ValidateSectionValues) was declared and stubbed at commit 6bb6b92
// (Claude/Brunel/superslm-s3.3-configgeometry-join-cell4-symbols-stub-build-
// 2026-07-28.md), then given its real body at commit b2a3a91
// (Claude/Brunel/superslm-s3.3-configgeometry-join-cell4-r1r4-body-build-
// 2026-07-28.md): ValidateConfigGeometryJoin wraps the existing, unit-tested
// CheckConfigGeometry, and ValidateRopeTablesShapeAgainstConfig resolves
// "cos"/"sin" and bounds each present tensor's elem_count independently. Every
// hostile cell below is green against that real logic. R5 (WGT1) is
// not-applicable (§13.2, D-SLM422/D-SLM423) and owes no cell.
//
// Every cell shares one coherent base config (hidden_size=4096,
// num_attention_heads=32, num_key_value_heads=8, head_dim=128, context_cap=4)
// under which R1, R2, R3, and R4 all independently hold, and each hostile
// cell mutates exactly one field or tensor away from it -- isolating the one
// relation the cell names, per Curie's mutation-pin discipline
// (StandardsDocument §"Pin the documented claim"). head_dim=128 and
// context_cap=4 are chosen to match kRopeSiteRoundTripHeadDim/
// -ContextCap/-Pairs/-CosFlat/-SinFlat exactly, so the ROP1 tensors are the
// same real, already-committed reference table the round-trip test above
// carries, not fabricated data.
// ---------------------------------------------------------------------------

// The shared base: R1 (4096 == 32*128), R2 (32 % 8 == 0), R3 (8 <= 32), and R4
// (context_cap=4, head_dim=128 -> pairs=64, matching
// kRopeSiteRoundTripContextCap/-Pairs) all hold. Only context_cap is moved off
// Cfg1Spec{}'s own default (2); num_attention_heads=32 and head_dim=128 are
// already Cfg1Spec{}'s own defaults (the explicit num_attention_heads
// assignment below is redundant with the default but kept so this function
// states its coherent base's geometry explicitly rather than by omission).
static Cfg1Spec MakeCell4CoherentCfg1Spec() {
	Cfg1Spec spec{};
	spec.num_attention_heads = 32;
	spec.context_cap = static_cast<uint32_t>(kRopeSiteRoundTripContextCap);
	return spec;
}

// A ROP1 (RopeTables) section whose "cos" and "sin" tensors carry
// independently chosen element counts -- MakeRop1SectionMultiRow above always
// gives both tensors the same shape, which cannot express "one tensor wrong,
// the other correct," the case D-SLM421's independent-per-tensor ruling
// requires a cell to prove.
static superslm_test::FixtureSection MakeRop1SectionAsymmetric(uint32_t cos_elem_count, uint32_t sin_elem_count,
                                                                 const int64_t* cos_flat, const int64_t* sin_flat) {
	using namespace superslm_test;
	std::vector<ManifestTensorSpec> tensors = {
	    {"cos", {cos_elem_count}},
	    {"sin", {sin_elem_count}},
	};
	auto manifest = BuildManifest(superslm::kRopeMagic, /*element_size=*/8, tensors);
	for (uint32_t i = 0; i < cos_elem_count; ++i) {
		PutU64(manifest.bytes, static_cast<size_t>(manifest.tensor_data_off[0]) + static_cast<size_t>(i) * 8,
		       static_cast<uint64_t>(cos_flat[i]));
	}
	for (uint32_t i = 0; i < sin_elem_count; ++i) {
		PutU64(manifest.bytes, static_cast<size_t>(manifest.tensor_data_off[1]) + static_cast<size_t>(i) * 8,
		       static_cast<uint64_t>(sin_flat[i]));
	}
	return MakeSection(SslmSectionType::RopeTables, SslmDtype::Int64, manifest.bytes, /*alignment=*/64);
}

// Positive control: every relation holds, Load accepts the artifact, and the
// one forward capability S3.3 itself ships today -- RopeApplySite, since
// S3.5's layer loop does not yet exist to run a whole-model forward -- runs
// against the loaded view. This is what stops a real R1-R4 implementation
// from rejecting a conformant artifact; it is not itself a hostile cell and
// is expected to stay green both before and after the real logic lands.
static void TestCell4LoadAcceptsFullyConformantConfigGeometryAndRopeShapeJoin() {
	using namespace superslm_test;
	Cfg1Spec spec = MakeCell4CoherentCfg1Spec();
	FixtureSection config = MakeSection(SslmSectionType::Config, SslmDtype::Raw, BuildCfg1(spec));
	FixtureSection rope_tables = MakeRop1SectionMultiRow(kRopeSiteRoundTripContextCap, kRopeSiteRoundTripPairs,
	                                                      kRopeSiteRoundTripCosFlat, kRopeSiteRoundTripSinFlat);
	auto built = BuildArtifact({config, MakeSigmoidLutSection(), rope_tables});

	SslmModelView view;
	std::string err;
	SslmModelStatus status = SslmModel::Load(built.bytes.data(), built.bytes.size(), view, &err);
	CHECK_MSG(status == SslmModelStatus::Ok,
	          "§13.1 cell 4 positive control: hidden_size=4096, heads=32, kv_heads=8, head_dim=128 (R1/R2/R3 "
	          "all hold) and ROP1 \"cos\"/\"sin\" each at context_cap*(head_dim/2)=256 elements (R4 holds) must "
	          "load Ok: got %s (%s)",
	          SslmModelStatusName(status), err.c_str());
	if (status != SslmModelStatus::Ok) return;
	CHECK(view.has_config);
	CHECK(view.has_rope_tables);

	int8_t row[kRopeSiteRoundTripHeadDim];
	for (int i = 0; i < kRopeSiteRoundTripHeadDim; ++i) row[i] = static_cast<int8_t>((i % 7) - 3);
	int8_t out_row[kRopeSiteRoundTripHeadDim];
	SslmForwardStatus fwd = RopeApplySite(row, static_cast<size_t>(kRopeSiteRoundTripHeadDim), /*position=*/0,
	                                       static_cast<int64_t>(kRopeSiteRoundTripContextCap), view.rope_tables,
	                                       out_row);
	CHECK_MSG(fwd == SslmForwardStatus::Ok,
	          "§13.1 cell 4 positive control, \"and the forward runs\" (plan §11 S3.3): RopeApplySite at "
	          "position 0 against the loaded, conformant ROP1 view must succeed: got %s",
	          SslmForwardStatusName(fwd));
}

// R1: hidden_size (4097) != num_attention_heads * head_dim (32 * 128 = 4096),
// one past the exact product, with R2/R3/R4 all held.
static void TestCell4LoadRejectsHiddenSizeMismatchAgainstHeadsTimesHeadDim() {
	using namespace superslm_test;
	Cfg1Spec spec = MakeCell4CoherentCfg1Spec();
	spec.hidden_size = spec.num_attention_heads * spec.head_dim + 1;  // 4097
	FixtureSection config = MakeSection(SslmSectionType::Config, SslmDtype::Raw, BuildCfg1(spec));
	FixtureSection rope_tables = MakeRop1SectionMultiRow(kRopeSiteRoundTripContextCap, kRopeSiteRoundTripPairs,
	                                                      kRopeSiteRoundTripCosFlat, kRopeSiteRoundTripSinFlat);
	auto built = BuildArtifact({config, MakeSigmoidLutSection(), rope_tables});

	SslmModelView view;
	std::string err;
	SslmModelStatus status = SslmModel::Load(built.bytes.data(), built.bytes.size(), view, &err);
	CHECK_MSG(status == SslmModelStatus::ConfigGeometryHiddenSizeMismatch,
	          "§13.1 cell 4, R1: hidden_size=4097 != num_attention_heads(32)*head_dim(128)=4096, R2/R3/R4 all "
	          "held: got %s, want ConfigGeometryHiddenSizeMismatch (%s)",
	          SslmModelStatusName(status), err.c_str());
	CHECK_MSG(err.find("4097") != std::string::npos && err.find("4096") != std::string::npos,
	          "diagnostic does not name both the declared hidden_size (4097) and the expected product (4096): "
	          "\"%s\"",
	          err.c_str());
}

// R2: num_attention_heads (32) % num_key_value_heads (7) != 0, with kv_heads
// still <= heads (R3 held) and hidden_size still == heads*head_dim (R1 held)
// and ROP1 still conformant (R4 held).
static void TestCell4LoadRejectsHeadsNotDivisibleByKvHeads() {
	using namespace superslm_test;
	Cfg1Spec spec = MakeCell4CoherentCfg1Spec();
	spec.num_key_value_heads = 7;  // 32 % 7 == 4 != 0; 7 <= 32 still holds
	FixtureSection config = MakeSection(SslmSectionType::Config, SslmDtype::Raw, BuildCfg1(spec));
	FixtureSection rope_tables = MakeRop1SectionMultiRow(kRopeSiteRoundTripContextCap, kRopeSiteRoundTripPairs,
	                                                      kRopeSiteRoundTripCosFlat, kRopeSiteRoundTripSinFlat);
	auto built = BuildArtifact({config, MakeSigmoidLutSection(), rope_tables});

	SslmModelView view;
	std::string err;
	SslmModelStatus status = SslmModel::Load(built.bytes.data(), built.bytes.size(), view, &err);
	CHECK_MSG(status == SslmModelStatus::ConfigGeometryHeadsNotDivisibleByKv,
	          "§13.1 cell 4, R2: num_attention_heads(32) %% num_key_value_heads(7) == 4 != 0, R1/R3/R4 all "
	          "held: got %s, want ConfigGeometryHeadsNotDivisibleByKv (%s)",
	          SslmModelStatusName(status), err.c_str());
	CHECK_MSG(err.find("32") != std::string::npos && err.find("7") != std::string::npos,
	          "diagnostic does not name both num_attention_heads (32) and num_key_value_heads (7): \"%s\"",
	          err.c_str());
}

// R3: num_key_value_heads (33) > num_attention_heads (32). CheckConfigGeometry
// checks KvHeadsExceedsHeads before HeadsNotDivisibleByKv
// (src/proof_manifest.cpp), so this cell also leaves R2 (32 % 33 != 0)
// failing at the same time -- the same precedent the existing
// TestConfigGeometryRejectsKvHeadsExceedsHeads pure-function cell already
// establishes (heads=8, kv_heads=16). R1 and R4 are held.
static void TestCell4LoadRejectsKvHeadsExceedsHeads() {
	using namespace superslm_test;
	Cfg1Spec spec = MakeCell4CoherentCfg1Spec();
	spec.num_key_value_heads = 33;  // > 32
	FixtureSection config = MakeSection(SslmSectionType::Config, SslmDtype::Raw, BuildCfg1(spec));
	FixtureSection rope_tables = MakeRop1SectionMultiRow(kRopeSiteRoundTripContextCap, kRopeSiteRoundTripPairs,
	                                                      kRopeSiteRoundTripCosFlat, kRopeSiteRoundTripSinFlat);
	auto built = BuildArtifact({config, MakeSigmoidLutSection(), rope_tables});

	SslmModelView view;
	std::string err;
	SslmModelStatus status = SslmModel::Load(built.bytes.data(), built.bytes.size(), view, &err);
	CHECK_MSG(status == SslmModelStatus::ConfigGeometryKvHeadsExceedsHeads,
	          "§13.1 cell 4, R3: num_key_value_heads(33) > num_attention_heads(32), R1/R4 held: got %s, want "
	          "ConfigGeometryKvHeadsExceedsHeads (%s)",
	          SslmModelStatusName(status), err.c_str());
	CHECK_MSG(err.find("33") != std::string::npos && err.find("32") != std::string::npos,
	          "diagnostic does not name both num_key_value_heads (33) and num_attention_heads (32): \"%s\"",
	          err.c_str());
}

// R4a: "cos" carries 512 elements -- context_cap(4)*head_dim(128), exactly the
// plan's own pre-D-SLM421 wrong bound (the factor-of-two error the fold
// corrected) -- while "sin" carries the correct 256 (context_cap*(head_dim/2))
// via the real, pinned kRopeSiteRoundTripSinFlat table. R1/R2/R3 held.
static void TestCell4LoadRejectsRopeCosShapeMismatchWithSinCorrect() {
	using namespace superslm_test;
	Cfg1Spec spec = MakeCell4CoherentCfg1Spec();
	FixtureSection config = MakeSection(SslmSectionType::Config, SslmDtype::Raw, BuildCfg1(spec));

	const uint32_t wrong_cos_elems =
	    static_cast<uint32_t>(kRopeSiteRoundTripContextCap) * static_cast<uint32_t>(kRopeSiteRoundTripHeadDim);
	const uint32_t correct_sin_elems =
	    static_cast<uint32_t>(kRopeSiteRoundTripContextCap) * static_cast<uint32_t>(kRopeSiteRoundTripPairs);
	std::vector<int64_t> cos_wrong(wrong_cos_elems, INT64_C(1073741824));  // in-domain filler; the length is the defect
	FixtureSection rope_tables =
	    MakeRop1SectionAsymmetric(wrong_cos_elems, correct_sin_elems, cos_wrong.data(), kRopeSiteRoundTripSinFlat);
	auto built = BuildArtifact({config, MakeSigmoidLutSection(), rope_tables});

	SslmModelView view;
	std::string err;
	SslmModelStatus status = SslmModel::Load(built.bytes.data(), built.bytes.size(), view, &err);
	CHECK_MSG(status == SslmModelStatus::RopeTablesShapeMismatchConfig,
	          "§13.1 cell 4, R4a: \"cos\" elem_count=512 (context_cap*head_dim, the plan's pre-D-SLM421 wrong "
	          "bound) against context_cap=4/head_dim=128 (expected context_cap*(head_dim/2)=256), \"sin\" "
	          "correct at 256: got %s, want RopeTablesShapeMismatchConfig (%s)",
	          SslmModelStatusName(status), err.c_str());
	CHECK_MSG(err.find("cos") != std::string::npos,
	          "diagnostic does not name the offending tensor \"cos\": \"%s\"", err.c_str());
}

// R4b: the mirror of R4a -- "sin" carries the wrong 512, "cos" carries the
// correct 256 via kRopeSiteRoundTripCosFlat. Proves the two tensors are
// checked independently (D-SLM421): a correct "cos" does not mask a
// malformed "sin".
static void TestCell4LoadRejectsRopeSinShapeMismatchWithCosCorrect() {
	using namespace superslm_test;
	Cfg1Spec spec = MakeCell4CoherentCfg1Spec();
	FixtureSection config = MakeSection(SslmSectionType::Config, SslmDtype::Raw, BuildCfg1(spec));

	const uint32_t correct_cos_elems =
	    static_cast<uint32_t>(kRopeSiteRoundTripContextCap) * static_cast<uint32_t>(kRopeSiteRoundTripPairs);
	const uint32_t wrong_sin_elems =
	    static_cast<uint32_t>(kRopeSiteRoundTripContextCap) * static_cast<uint32_t>(kRopeSiteRoundTripHeadDim);
	std::vector<int64_t> sin_wrong(wrong_sin_elems, INT64_C(1073741824));  // in-domain filler; the length is the defect
	FixtureSection rope_tables =
	    MakeRop1SectionAsymmetric(correct_cos_elems, wrong_sin_elems, kRopeSiteRoundTripCosFlat, sin_wrong.data());
	auto built = BuildArtifact({config, MakeSigmoidLutSection(), rope_tables});

	SslmModelView view;
	std::string err;
	SslmModelStatus status = SslmModel::Load(built.bytes.data(), built.bytes.size(), view, &err);
	CHECK_MSG(status == SslmModelStatus::RopeTablesShapeMismatchConfig,
	          "§13.1 cell 4, R4b: \"sin\" elem_count=512 (context_cap*head_dim, the plan's pre-D-SLM421 wrong "
	          "bound) against context_cap=4/head_dim=128 (expected context_cap*(head_dim/2)=256), \"cos\" "
	          "correct at 256: got %s, want RopeTablesShapeMismatchConfig (%s)",
	          SslmModelStatusName(status), err.c_str());
	CHECK_MSG(err.find("sin") != std::string::npos,
	          "diagnostic does not name the offending tensor \"sin\": \"%s\"", err.c_str());
}

// ---------------------------------------------------------------------------
// S3.3 -- the RoPE application site's own red suite, against the real,
// callable RopeApplySite symbol declared and stubbed at commit 13dfcfd
// (Claude/Brunel/superslm-s3.3-rope-application-site-header-contract-build-
// 2026-07-28.md). The stub unconditionally returns WorkspaceTooSmall --
// calling nothing, reading nothing, writing nothing to out_row -- so every
// cell below fails against that status or against out_row's poison pattern,
// never against a compile error or a fixture/setup failure: the three
// blocked cells the test-design record's Sec6 fully specified
// (Claude/Curie/superslm-s3.3-rope-application-site-test-design-2026-07-28.md
// Sec6.2/6.3/6.4) are landed here, red, against the real symbol.
// ---------------------------------------------------------------------------

// Sec6.2's own feature-oracle cell: a purpose-built 128-row ROP1 section
// carrying real, pinned data at rows 0 and (context_cap-1) only -- the two
// positions kRopeSitePositionZeroCase/kRopeSitePositionCapMinusOneCase
// exercise -- with every other row zero-filled. ValidateRopeTablesDomain's
// own check is magnitude-only (|cos|,|sin| <= 2^30, Sec2 of the test-design
// record) and has no shape cross-check against CFG1.context_cap, so a
// zero-filled row is accepted; this is the "purpose-built 128-row ROP1
// section" option the test-design record's own Sec6.2 names as a mechanical
// extension of the already-green round-trip cell's own MakeRop1SectionMultiRow
// helper.
static superslm_test::FixtureSection MakeRop1SectionSparse128At(
    const superslm_test::RopeSitePositionCase& zero_case,
    const superslm_test::RopeSitePositionCase& cap_minus_one_case) {
	using namespace superslm_test;
	const int32_t cap = kRopeSitePinnedContextCap;
	const int32_t pairs = kRopeSitePinnedPairs;
	std::vector<int64_t> cos_flat(static_cast<size_t>(cap) * static_cast<size_t>(pairs), 0);
	std::vector<int64_t> sin_flat(static_cast<size_t>(cap) * static_cast<size_t>(pairs), 0);
	const size_t zero_row_off = static_cast<size_t>(zero_case.position) * static_cast<size_t>(pairs);
	const size_t last_row_off = static_cast<size_t>(cap_minus_one_case.position) * static_cast<size_t>(pairs);
	for (int32_t i = 0; i < pairs; ++i) {
		cos_flat[zero_row_off + static_cast<size_t>(i)] = zero_case.cos_row[i];
		sin_flat[zero_row_off + static_cast<size_t>(i)] = zero_case.sin_row[i];
		cos_flat[last_row_off + static_cast<size_t>(i)] = cap_minus_one_case.cos_row[i];
		sin_flat[last_row_off + static_cast<size_t>(i)] = cap_minus_one_case.sin_row[i];
	}
	return MakeRop1SectionMultiRow(cap, pairs, cos_flat.data(), sin_flat.data());
}

// Sec6.2 (test-design record §6.2): drives the real RopeApplySite with each
// Sec4.1 case's row/position/context_cap (= kRopeSitePinnedContextCap)
// against a real loaded model view carrying the purpose-built table above,
// and asserts out_row == c.expected_out exactly -- the site's own feature
// oracle, at position 0 (the identity) and position context_cap-1 (a
// genuine, clamp-engaging rotation), per Sec13 dim 6's own named cell.
static void TestRopeApplySiteFeatureOracleAtPositionZeroAndCapMinusOne() {
	using namespace superslm_test;

	Cfg1Spec spec{};
	spec.context_cap = static_cast<uint32_t>(kRopeSitePinnedContextCap);
	spec.head_dim = static_cast<uint32_t>(kRopeSitePinnedHeadDim);
	FixtureSection config = MakeSection(SslmSectionType::Config, SslmDtype::Raw, BuildCfg1(spec));
	FixtureSection rope_tables =
	    MakeRop1SectionSparse128At(kRopeSitePositionZeroCase, kRopeSitePositionCapMinusOneCase);
	auto built = BuildArtifact({config, MakeSigmoidLutSection(), rope_tables});

	SslmModelView view;
	std::string err;
	SslmModelStatus status = SslmModel::Load(built.bytes.data(), built.bytes.size(), view, &err);
	CHECK_MSG(status == SslmModelStatus::Ok,
	          "the feature-oracle fixture (sparse 128-row RoPE table) failed to load: got %s (%s)",
	          SslmModelStatusName(status), err.c_str());
	if (status != SslmModelStatus::Ok) return;
	CHECK_MSG(view.has_rope_tables, "a successfully loaded artifact must expose its RopeTables view");
	if (!view.has_rope_tables) return;

	const RopeSitePositionCase* cases[] = {&kRopeSitePositionZeroCase, &kRopeSitePositionCapMinusOneCase};
	for (const RopeSitePositionCase* c : cases) {
		int8_t row[128];
		int8_t out_row[128];
		for (int i = 0; i < kRopeSitePinnedHeadDim; ++i) {
			row[i] = static_cast<int8_t>(c->row[i]);
			out_row[i] = INT8_C(-99);  // poison
		}
		const auto forward_status =
		    RopeApplySite(row, static_cast<size_t>(kRopeSitePinnedHeadDim), c->position,
		                  static_cast<int64_t>(kRopeSitePinnedContextCap), view.rope_tables, out_row);
		CHECK_MSG(forward_status == SslmForwardStatus::Ok,
		          "%s: RopeApplySite(position=%lld) status == %s, want Ok (Sec6.2's feature oracle: "
		          "CheckPositionOverCap -> table read -> RopeApplyPair -> ClampRopeCode, in order, to "
		          "completion)",
		          c->label, static_cast<long long>(c->position), SslmForwardStatusName(forward_status));
		bool matches = true;
		for (int i = 0; i < kRopeSitePinnedHeadDim; ++i) {
			if (out_row[i] != static_cast<int8_t>(c->expected_out[i])) {
				matches = false;
				break;
			}
		}
		CHECK_MSG(matches,
		          "%s: RopeApplySite(position=%lld) out_row does not match the reference-derived "
		          "expected rotated-and-clamped row (tests/gen_s3_3_rope_application_site_fixtures.py, "
		          "calling the vendored rope.py's own rope_apply_pair)",
		          c->label, static_cast<long long>(c->position));
	}
}

// Sec6.3 (test-design record §6.3): "a position == context_cap is rejected
// before a table read" -- Sec11 S3.3's own gate line, Sec6.2 step 3's own
// text. Poison-fills out_row before each call (this suite's own poison-fill
// convention, e.g. TestRequantChainCheckedRejectsOverC29Domain above), calls
// RopeApplySite at position == context_cap and position == context_cap + 1
// against a real loaded ROP1 tensor whose row count is exactly context_cap
// (kRopeSiteRoundTripContextCap = 4, so row 4 is one past the tensor's last
// valid row), and asserts PositionOverCap AND out_row byte-for-byte
// unchanged from the poison pattern -- proving no write, and therefore no
// read of the out-of-bounds row, occurred (the strongest ordering proof
// constructible without instrumenting the tensor accessor itself).
static void TestRopeApplySiteRejectsPositionAtOrAboveCapBeforeAnyTableRead() {
	using namespace superslm_test;

	Cfg1Spec spec{};
	spec.context_cap = static_cast<uint32_t>(kRopeSiteRoundTripContextCap);
	spec.head_dim = static_cast<uint32_t>(kRopeSiteRoundTripHeadDim);
	FixtureSection config = MakeSection(SslmSectionType::Config, SslmDtype::Raw, BuildCfg1(spec));
	FixtureSection rope_tables = MakeRop1SectionMultiRow(kRopeSiteRoundTripContextCap, kRopeSiteRoundTripPairs,
	                                                      kRopeSiteRoundTripCosFlat, kRopeSiteRoundTripSinFlat);
	auto built = BuildArtifact({config, MakeSigmoidLutSection(), rope_tables});

	SslmModelView view;
	std::string err;
	SslmModelStatus status = SslmModel::Load(built.bytes.data(), built.bytes.size(), view, &err);
	CHECK_MSG(status == SslmModelStatus::Ok,
	          "the ordering-cell fixture (context_cap=%d RoPE table) failed to load: got %s (%s)",
	          kRopeSiteRoundTripContextCap, SslmModelStatusName(status), err.c_str());
	if (status != SslmModelStatus::Ok) return;
	CHECK_MSG(view.has_rope_tables, "a successfully loaded artifact must expose its RopeTables view");
	if (!view.has_rope_tables) return;

	int8_t row[128];
	for (int i = 0; i < kRopeSiteRoundTripHeadDim; ++i) row[i] = static_cast<int8_t>((i * 3) - 5);

	const int64_t over_positions[] = {
	    static_cast<int64_t>(kRopeSiteRoundTripContextCap),      // == context_cap: one past the last valid slot
	    static_cast<int64_t>(kRopeSiteRoundTripContextCap) + 1,  // > context_cap
	};
	for (int64_t position : over_positions) {
		int8_t out_row[128];
		for (int i = 0; i < kRopeSiteRoundTripHeadDim; ++i) out_row[i] = INT8_C(-99);  // poison

		const auto forward_status =
		    RopeApplySite(row, static_cast<size_t>(kRopeSiteRoundTripHeadDim), position,
		                  static_cast<int64_t>(kRopeSiteRoundTripContextCap), view.rope_tables, out_row);
		CHECK_MSG(forward_status == SslmForwardStatus::PositionOverCap,
		          "position=%lld (context_cap=%d): RopeApplySite status == %s, want PositionOverCap",
		          static_cast<long long>(position), kRopeSiteRoundTripContextCap,
		          SslmForwardStatusName(forward_status));
		bool untouched = true;
		for (int i = 0; i < kRopeSiteRoundTripHeadDim; ++i) {
			if (out_row[i] != INT8_C(-99)) {
				untouched = false;
				break;
			}
		}
		CHECK_MSG(untouched,
		          "position=%lld: out_row must be byte-for-byte unchanged from the poison pattern on "
		          "rejection -- \"never a table read\" (Sec11 S3.3's own gate line) -- a real read would "
		          "have overwritten at least one poisoned byte",
		          static_cast<long long>(position));
	}
}

// Sec6.4 (test-design record §6.4): the guard-vitality (ASan) cell, Coverage
// Model dim 11 realized as a build-configuration-gated property (plan
// acceptance criterion 8) rather than a plain-build assertion alone. The
// ROP1 tensor here has exactly ONE row (context_cap = 1) -- the tightest
// allocation this suite can construct -- so a real read of row 1 (one past
// the tensor's only valid row) has the shortest possible distance to travel
// before landing in unallocated memory, maximizing the chance an ASan
// redzone traps it under the project's sanitizer CI leg. This cell also runs
// and is checked in a plain (non-ASan) build: it does not require the
// sanitizer to assert PositionOverCap, and the sanitizer is what turns a
// would-be silent OOB read into a hard failure on top of that assertion --
// exactly the "vitality" this dimension names, and distinct from Sec6.3's
// cell above by using the tightest possible tensor rather than a
// general-purpose one.
static void TestRopeApplySiteGuardFiresBeforeOutOfBoundsTensorReadUnderAsan() {
	using namespace superslm_test;

	const int32_t cap = 1;
	const int32_t pairs = kRopeSitePinnedPairs;
	std::vector<int64_t> cos_flat(static_cast<size_t>(pairs), INT64_C(1073741824));  // identity row (2^30)
	std::vector<int64_t> sin_flat(static_cast<size_t>(pairs), 0);

	Cfg1Spec spec{};
	spec.context_cap = static_cast<uint32_t>(cap);
	spec.head_dim = static_cast<uint32_t>(kRopeSitePinnedHeadDim);
	FixtureSection config = MakeSection(SslmSectionType::Config, SslmDtype::Raw, BuildCfg1(spec));
	FixtureSection rope_tables = MakeRop1SectionMultiRow(cap, pairs, cos_flat.data(), sin_flat.data());
	auto built = BuildArtifact({config, MakeSigmoidLutSection(), rope_tables});

	SslmModelView view;
	std::string err;
	SslmModelStatus status = SslmModel::Load(built.bytes.data(), built.bytes.size(), view, &err);
	CHECK_MSG(status == SslmModelStatus::Ok,
	          "the guard-vitality fixture (context_cap=1 RoPE table, the tightest allocation this suite "
	          "constructs) failed to load: got %s (%s)",
	          SslmModelStatusName(status), err.c_str());
	if (status != SslmModelStatus::Ok) return;
	CHECK_MSG(view.has_rope_tables, "a successfully loaded artifact must expose its RopeTables view");
	if (!view.has_rope_tables) return;

	int8_t row[128];
	for (int i = 0; i < kRopeSitePinnedHeadDim; ++i) row[i] = static_cast<int8_t>((i * 7) - 3);
	int8_t out_row[128];
	for (int i = 0; i < kRopeSitePinnedHeadDim; ++i) out_row[i] = INT8_C(-99);  // poison

	const auto forward_status =
	    RopeApplySite(row, static_cast<size_t>(kRopeSitePinnedHeadDim), /*position=*/static_cast<int64_t>(cap),
	                  static_cast<int64_t>(cap), view.rope_tables, out_row);
	CHECK_MSG(forward_status == SslmForwardStatus::PositionOverCap,
	          "context_cap=1, position=1 (one past the tensor's single valid row): RopeApplySite status == "
	          "%s, want PositionOverCap -- the guard must fire before any read of row 1, which does not "
	          "exist in this tensor",
	          SslmForwardStatusName(forward_status));
}

// ---------------------------------------------------------------------------
// Poirot fa3189a-s3.3-rope-site-and-c32-softmax-review-2026-07-28.md -- the
// remediation red suite for Critical 1, Critical 2, and Significant 5. The
// C32 kernel-half pin (Significant 3) lives above, in the C32 red suite
// section, next to its own gate-half siblings.
// Claude/Curie/fa3189a-s3.3-rope-site-and-c32-softmax-remediation-test-design-
// 2026-07-28.md.
// ---------------------------------------------------------------------------

// Critical 1 (closed; confirmed by Claude/Poirot/72b0c7f-s3.3-rope-site-and-c32-
// softmax-confirmation-2026-07-28.md): RopeApplySite now guards its "cos"/"sin"
// tensor lookup and returns RopeTableTensorMissing before the read that
// previously dereferenced null on this input. Routed through the crash-probe
// child (this suite's established death-test convention, S2.5) because a
// regression here would again fault the process -- a cell that crashes the
// runner is not a usable red.
static void TestRopeApplySiteRejectsMissingCosSinTensorsInsteadOfDereferencingNull() {
	static const char* kProbeName = "rope_apply_site_null_cos_tensor_deref";
	std::string tail;
	CrashProbeOutcome outcome = RunsCrashProbeAndCrashes(kProbeName, &tail);
	CHECK_MSG(outcome == CrashProbeOutcome::kRanNoCrash,
	          "RopeApplySite against a loaded ROP1 manifest carrying no \"cos\"/\"sin\" tensor must "
	          "return a defined status, not fault the process (Critical 1) -- a null Tensor(\"cos\")/"
	          "Tensor(\"sin\") is a real, reachable load-time outcome (SslmTensorManifest::Tensor "
	          "returns nullptr for an absent name, model.h:206-207) -- outcome was %s, child output "
	          "was: %s",
	          CrashProbeOutcomeName(outcome), tail.c_str());
	if (outcome != CrashProbeOutcome::kRanNoCrash) return;
	// Significant B (Poirot 72b0c7f-s3.3-rope-site-and-c32-softmax-confirmation-2026-07-28.md,
	// plan Sec13 dim 5): "did not fault" alone does not pin WHICH defined status came back --
	// the probe's own RunCrashProbe branch prints it (superslm::SslmForwardStatusName(forward_status))
	// into the captured tail, so the exact enumerator is asserted here by name rather than
	// discarded. A future edit that routes this rejection through any other status (e.g.
	// RopeTableExtentExceeded) would leave the crash-probe assertion above green while
	// silently breaking the contract this status name states.
	CHECK_MSG(tail.find("forward_status=RopeTableTensorMissing") != std::string::npos,
	          "the probe's captured output does not name RopeTableTensorMissing -- child output "
	          "was: %s",
	          tail.c_str());
}

// Critical 2, the fault half (closed; Poirot fa3189a-s3.3-rope-site-and-c32-
// softmax-review-2026-07-28.md, confirmed closed by Claude/Poirot/72b0c7f-
// s3.3-rope-site-and-c32-softmax-confirmation-2026-07-28.md): a `position`
// the caller's `context_cap` legally admits, but far past a real, loaded
// tensor's actual row count, previously read unmapped heap memory before the
// extent guard existed -- reproduced exactly as the review's own executed
// probe (position=268435456, context_cap=2147483647, a genuine
// CFG1-admissible u32 value). Routed through the crash-probe child for the
// same reason as the cell above.
static void TestRopeApplySiteRejectsPositionFarPastTensorExtentInsteadOfReadingUnmappedMemory() {
	static const char* kProbeName = "rope_apply_site_position_far_past_tensor_extent";
	std::string tail;
	CrashProbeOutcome outcome = RunsCrashProbeAndCrashes(kProbeName, &tail);
	CHECK_MSG(outcome == CrashProbeOutcome::kRanNoCrash,
	          "RopeApplySite at a position the caller-supplied context_cap admits, but the loaded ROP1 "
	          "tensor's own row count does not, must return a defined status, not fault the process "
	          "(Critical 2) -- outcome was %s, child output was: %s",
	          CrashProbeOutcomeName(outcome), tail.c_str());
	if (outcome != CrashProbeOutcome::kRanNoCrash) return;
	// Significant B: the exact status, not merely "did not fault" -- see the sibling cell
	// above for the full rationale. The probe's own captured tail already carries
	// SslmForwardStatusName(forward_status); asserted by name rather than discarded.
	CHECK_MSG(tail.find("forward_status=RopeTableExtentExceeded") != std::string::npos,
	          "the probe's captured output does not name RopeTableExtentExceeded -- child output "
	          "was: %s",
	          tail.c_str());
}

// Critical 2, the quieter near-miss half: a `position` just ONE row past the
// tensor's real row count, still inside a legitimate `context_cap`, is a small,
// memory-SAFE offset (still inside the same manifest allocation SslmModel::Load
// itself produced -- deliberately not the unmapped-memory case the two crash-
// probe cells above pin, so this cell needs no isolation). Before the T-1322
// remedy, RopeApplySite had no way to know it had read past the tensor's own
// validated extent and returned Ok: the caller-ensures safety argument ("every
// element already cleared ValidateRopeTablesDomain's bound at load time") does
// not survive a read outside the tensor the loader actually validated (Poirot
// fa3189a review, Critical 2). The site now bounds position/pairs against each
// tensor's own elem_count and returns RopeTableExtentExceeded, which is what
// this cell asserts.
//
// This is the sibling the review's finding names directly: the existing
// TestRopeApplySiteGuardFiresBeforeOutOfBoundsTensorReadUnderAsan cell above
// cannot fail for this defect class, because its own fixture sets
// context_cap == the tensor's row count (1), so CheckPositionOverCap's own
// bound intercepts position=1 before any tensor read is attempted -- it proves
// a different, also-true property (the position-cap guard fires before a table
// read when the two bounds happen to agree), not this one. This cell sets
// context_cap ABOVE the tensor's row count, so CheckPositionOverCap does NOT
// intercept, and the read past the tensor's own extent actually happens.
static void TestRopeApplySiteReturnsOkWhenReadingPastTensorExtentWithinContextCap() {
	using namespace superslm_test;

	const int32_t tensor_rows = 1;
	const int32_t pairs = 4;
	const int32_t head_dim = pairs * 2;
	std::vector<int64_t> cos_flat(static_cast<size_t>(pairs), INT64_C(1073741824));  // identity row (2^30)
	std::vector<int64_t> sin_flat(static_cast<size_t>(pairs), 0);

	Cfg1Spec spec{};
	spec.head_dim = static_cast<uint32_t>(head_dim);
	// spec.context_cap is set to the tensor's own real row count (1) and
	// spec.hidden_size tracks the overridden head_dim, so SslmModel::Load's
	// config-geometry/ROP1 join (R1, R4; S3.3 §13.1 cell 4, D-SLM420-D-SLM423)
	// accepts this fixture -- unrelated to and does not narrow the local
	// context_cap=2 this cell passes directly to RopeApplySite below, which
	// stays decoupled from what Load validated (the property this cell proves).
	spec.context_cap = static_cast<uint32_t>(tensor_rows);
	spec.hidden_size = spec.num_attention_heads * spec.head_dim;
	FixtureSection config = MakeSection(SslmSectionType::Config, SslmDtype::Raw, BuildCfg1(spec));
	FixtureSection rope_tables = MakeRop1SectionMultiRow(tensor_rows, pairs, cos_flat.data(), sin_flat.data());
	auto built = BuildArtifact({config, MakeSigmoidLutSection(), rope_tables});

	SslmModelView view;
	std::string err;
	SslmModelStatus status = SslmModel::Load(built.bytes.data(), built.bytes.size(), view, &err);
	CHECK_MSG(status == SslmModelStatus::Ok,
	          "the near-miss fixture (a real 1-row ROP1 tensor) failed to load: got %s (%s)",
	          SslmModelStatusName(status), err.c_str());
	if (status != SslmModelStatus::Ok) return;
	CHECK_MSG(view.has_rope_tables, "a successfully loaded artifact must expose its RopeTables view");
	if (!view.has_rope_tables) return;

	// context_cap decoupled from the tensor's real row count (1) -- exactly the join
	// the loader does not make (Critical 2's own text).
	const int64_t context_cap = 2;
	const int64_t position = 1;  // < context_cap, but one row past the tensor's real extent

	int8_t row[8];
	for (int i = 0; i < head_dim; ++i) row[i] = static_cast<int8_t>((i * 5) - 7);
	int8_t out_row[8];
	for (int i = 0; i < head_dim; ++i) out_row[i] = INT8_C(-99);  // poison

	const auto forward_status = RopeApplySite(row, static_cast<size_t>(head_dim), position, context_cap,
	                                           view.rope_tables, out_row);
	// Significant B (plan Sec13 dim 5): the exact status the site's own header names for this
	// path, not merely "not Ok" -- a future edit that routed this rejection through a
	// different status (e.g. RopeTableTensorMissing) would satisfy a weaker `!= Ok` check
	// while breaking the contract this specific enumerator states.
	CHECK_MSG(forward_status == SslmForwardStatus::RopeTableExtentExceeded,
	          "context_cap=%lld (a legitimate value), position=%lld (one row past the tensor's real row "
	          "count, %d): RopeApplySite status == %s, want RopeTableExtentExceeded -- Critical 2 "
	          "(closed; Poirot fa3189a review, confirmed closed by the 72b0c7f confirmation pass): "
	          "before the fix the site bounded position only against context_cap, never against the "
	          "tensors' own extent, so it silently read whatever bytes lay past the validated tensor "
	          "and reported Ok",
	          static_cast<long long>(context_cap), static_cast<long long>(position), tensor_rows,
	          SslmForwardStatusName(forward_status));

	// Significant B, second half: the plan's dim 5 also owes "leaves the sequence in the
	// state the contract names" -- forward_sites.cpp's own comment above CheckPositionOverCap
	// states out_row stays exactly as the caller left it on rejection ("never a table read");
	// no cell asserted the poison survives this specific rejection path until now.
	for (int i = 0; i < head_dim; ++i) {
		CHECK_MSG(out_row[i] == INT8_C(-99),
		          "out_row[%d] == %d after a RopeTableExtentExceeded rejection, want the poison "
		          "value -99 untouched -- the site's own contract states out_row is left exactly "
		          "as the caller supplied it on rejection",
		          i, static_cast<int>(out_row[i]));
	}
}

// Significant 5 (closed; confirmed by Claude/Poirot/72b0c7f-s3.3-rope-site-and-
// c32-softmax-confirmation-2026-07-28.md): forward_sites.h asserts a load-time
// rejection of odd `head_dim` ("load-time-rejected otherwise, per Sec6.2 step
// 3's own 'head_dim odd is a load-time rejection'"), and ParseConfigImpl now
// performs that parity check, returning BadConfigHeadDimParity. Per
// StandardsDocument Sec5.6, a document claiming what the code does not deliver
// is a code bug; this cell pins the remedy at the boundary the header's own
// text names.
static void TestParseConfigRejectsOddHeadDimAtLoadTime() {
	using namespace superslm_test;

	Cfg1Spec spec{};
	spec.head_dim = 1;  // odd -- the rotation pairs elements two at a time
	FixtureSection config = MakeSection(SslmSectionType::Config, SslmDtype::Raw, BuildCfg1(spec));
	auto built = BuildArtifact({config, MakeSigmoidLutSection()});

	SslmModelView view;
	std::string err;
	SslmModelStatus status = SslmModel::Load(built.bytes.data(), built.bytes.size(), view, &err);
	// Significant B (plan Sec13 dim 5): the exact status, not merely "not Ok" -- a future
	// edit that folded the odd-head_dim rejection into the zero-dimension check (BadConfigDim)
	// would satisfy a weaker `!= Ok` check while breaking the contract this specific
	// enumerator states.
	CHECK_MSG(status == SslmModelStatus::BadConfigHeadDimParity,
	          "SslmModel::Load with CFG1 head_dim=1 (odd) == %s, want BadConfigHeadDimParity -- "
	          "forward_sites.h's own text claims odd head_dim is load-time-rejected (Sec6.2 step 3), "
	          "and ParseConfigImpl (model.cpp) is required to enforce that parity check (Significant "
	          "5, closed by Claude/Poirot/72b0c7f-s3.3-rope-site-and-c32-softmax-confirmation-"
	          "2026-07-28.md) -- this failure means that check regressed",
	          SslmModelStatusName(status));
}

// ---------------------------------------------------------------------------
// S3.4 -- the SwiGLU activation site (C34, §5.4, §6.3 step 11; T-1345;
// Claude/Curie/superslm-s3.4-mlp-act-site-test-design-2026-07-29.md). The
// three cells below are the first C++ exercise of a real, declared,
// deliberately-wrong MlpActSite (WorkspaceTooSmall stub, this file's own
// declare-and-stub convention, matching the RoPE application site's
// precedent D-SLM384/385/386).
// ---------------------------------------------------------------------------

// §3's feature-oracle cell (test-design record §3, §5): the site's whole
// composition -- CheckSiluCompositionScaleDomain, SiluSigmoidQ15 per element,
// the wide product, then the funnel folding BOTH gate_scale and up_scale --
// against the already-shipped LUT (kSiluLutGoldenTable) and the
// already-shipped funnel (RequantChainChecked), never a recode of the site's
// own steps. Mutation-verified by execution (test-design record §5,
// `mlp_act_fold_mutation_proof.py`, out-of-repo scratch, vendored intmath.py
// port of the identical C++ funnel math): dropping up_scale from the fold
// leaves out_codes IDENTICAL (0/64 differ) but changes out_scale from
// (2^30, 8) to (2^30, -4) -- this cell's own out_scale assertion below kills
// that mutation. Evaluating the sigmoid on up_code instead of gate_code
// changes out_codes at 54/64 elements -- this cell's own out_codes assertion
// kills that mutation. Changing site_constant from (2^30, 0) to
// (1234567890, 3) leaves out_codes unchanged but moves out_scale to
// (1234567890, 11) -- again caught by the out_scale assertion. Every
// discriminating property this cell needs is therefore proven to fail on a
// real, executed mutant, not asserted by argument.
static void TestMlpActSiteC34FeatureOracleAgainstTheRealFunnelAndLut() {
	using superslm::CarriedScale;
	using superslm::SiluSigmoidQ15;
	using superslm::SslmForwardStatus;
	using namespace superslm_test;

	constexpr size_t kN = 64;
	int8_t gate_code[kN];
	int8_t up_code[kN];
	for (size_t i = 0; i < kN; ++i) {
		gate_code[i] = static_cast<int8_t>(((static_cast<int>(i) * 41 + 13) % 255) - 127);
		up_code[i] = static_cast<int8_t>(((static_cast<int>(i) * 17 + 5) % 255) - 127);
	}

	// (m, e) = (2^30, -34): silu_lut.cpp's own header comment names "-shift
	// measured in [15,19] over the full calibrated corpus", and
	// shift = e + kSiluLutLog2K(5) + kSiluLutQIdx(12) = e + 17, so
	// -shift in [15,19] <=> e in [-36,-32]; -34 is the midpoint of that real,
	// executed corpus range.
	const CarriedScale gate_scale{/*m=*/INT64_C(1073741824), /*e=*/-34};
	const CarriedScale up_scale{/*m=*/INT64_C(1073741824), /*e=*/-18};
	const CarriedScale site_constant{/*m=*/INT64_C(1073741824), /*e=*/0};

	// Expected: the real, already-shipped C10 LUT composed through the real,
	// already-shipped funnel, folding BOTH carried scales (Tools/superslm_spike/
	// dynamic_engine.py:546-548's own `_chain_record_vec(..., wide,
	// [gate_scale[t], up_scale[t]], trace)` -- gate then up, neither dropped).
	std::vector<int64_t> wide(kN);
	for (size_t i = 0; i < kN; ++i) {
		const int32_t sig =
		    SiluSigmoidQ15(kSiluLutGoldenTable, gate_code[i], gate_scale.m, gate_scale.e);
		wide[i] = static_cast<int64_t>(gate_code[i]) * static_cast<int64_t>(sig) *
		          static_cast<int64_t>(up_code[i]);
	}
	const CarriedScale incoming[] = {gate_scale, up_scale};
	std::vector<int8_t> expected_codes(kN, INT8_C(-99));
	CarriedScale expected_scale{INT64_C(-99), INT64_C(-99)};
	auto expected_result =
	    superslm::RequantChainChecked(wide.data(), kN, std::span<const CarriedScale>{incoming, 2},
	                                    site_constant, expected_codes.data(), &expected_scale);
	CHECK_MSG(expected_result.status == SslmForwardStatus::Ok,
	          "the LUT-based oracle row itself must be accepted by the already-shipped funnel: got %s",
	          SslmForwardStatusName(expected_result.status));

	std::vector<int8_t> out_codes(kN, INT8_C(-99));
	CarriedScale out_scale{INT64_C(-99), INT64_C(-99)};
	auto result = superslm::MlpActSite(gate_code, gate_scale, up_code, up_scale, kN,
	                                     kSiluLutGoldenTable, site_constant, out_codes.data(),
	                                     &out_scale);
	CHECK_MSG(result == SslmForwardStatus::Ok,
	          "MlpActSite status == %s, want Ok (red-unimplemented until Brunel's green phase)",
	          SslmForwardStatusName(result));
	if (result == SslmForwardStatus::Ok) {
		for (size_t i = 0; i < kN; ++i) {
			CHECK_MSG(out_codes[i] == expected_codes[i],
			          "MlpActSite out_codes[%zu] == %d, want %d (the LUT-and-funnel oracle's own code)",
			          i, static_cast<int>(out_codes[i]), static_cast<int>(expected_codes[i]));
		}
		CHECK_MSG(out_scale.m == expected_scale.m && out_scale.e == expected_scale.e,
		          "MlpActSite out_scale == (%lld, %lld), want (%lld, %lld) (both gate_scale and "
		          "up_scale must fold into the funnel's incoming span)",
		          static_cast<long long>(out_scale.m), static_cast<long long>(out_scale.e),
		          static_cast<long long>(expected_scale.m), static_cast<long long>(expected_scale.e));
	}
}

// §3/§7.2's ordering-and-vitality cell: CheckSiluCompositionScaleDomain must
// fire BEFORE SiluSigmoidQ15 ever evaluates, and reject rather than compute
// (the S3.3 §6.3 ordering-cell precedent, test-design record §3). e=9 is
// §5.4's own fourth executed boundary point -- both the load-time descriptor
// and the runtime predicate reject it, AND it is exactly the overflow point
// of SiluSigmoidQ15's own left-shift branch (shift = e+17 = 26 ==
// kSiluLutTermLeftShiftOverflowExponent, silu_lut.h). `gate_scale.m` is
// pinned to kCompositionScaleMaxAbsM (2^31-1, the domain's own worst-case
// mantissa) rather than an arbitrary in-domain value, because the overflow
// claim is magnitude-dependent: executed (`mlp_act_ordering_mutation_proof.py`,
// out-of-repo scratch), at code=127 and this exact m, `term << 26` computes
// 18302628877110870016 against INT64_MAX's 9223372036854775807 -- genuinely
// exceeds int64 and wraps (two's complement) to -144115196598681600 rather
// than trapping. A reads-before-rejecting mutant of this exact fixture would
// therefore not merely fail an abstract UB argument; it would silently
// compute a negative garbage `pos_fixed` and feed it onward, which is
// precisely the failure this ordering guard exists to prevent.
static void TestMlpActSiteC34RejectsOutOfDomainGateScaleBeforeComputingSigmoid() {
	using superslm::CarriedScale;
	using superslm::SslmForwardStatus;
	using namespace superslm_test;

	constexpr size_t kN = 8;
	int8_t gate_code[kN];
	int8_t up_code[kN];
	for (size_t i = 0; i < kN; ++i) {
		gate_code[i] = static_cast<int8_t>(((static_cast<int>(i) * 41 + 13) % 255) - 127);
		up_code[i] = static_cast<int8_t>(((static_cast<int>(i) * 17 + 5) % 255) - 127);
	}
	const CarriedScale gate_scale{/*m=*/INT64_C(2147483647) /*2^31-1, kCompositionScaleMaxAbsM*/,
	                              /*e=*/9};
	const CarriedScale up_scale{/*m=*/INT64_C(1073741824), /*e=*/-18};
	const CarriedScale site_constant{/*m=*/INT64_C(1073741824), /*e=*/0};

	std::vector<int8_t> out_codes(kN, INT8_C(-99));
	CarriedScale out_scale{INT64_C(-99), INT64_C(-99)};
	auto result = superslm::MlpActSite(gate_code, gate_scale, up_code, up_scale, kN,
	                                     kSiluLutGoldenTable, site_constant, out_codes.data(),
	                                     &out_scale);
	CHECK_MSG(result == SslmForwardStatus::SiluCompositionScaleOutOfDomain,
	          "MlpActSite(gate_scale.e=9) status == %s, want SiluCompositionScaleOutOfDomain -- the "
	          "site must call CheckSiluCompositionScaleDomain BEFORE SiluSigmoidQ15 ever evaluates, "
	          "and reject rather than compute",
	          SslmForwardStatusName(result));
	for (size_t i = 0; i < kN; ++i) {
		CHECK_MSG(out_codes[i] == INT8_C(-99),
		          "out_codes[%zu] == %d after rejection, want the poison value -99 untouched -- a "
		          "site that computed SiluSigmoidQ15 before checking the domain would have written "
		          "through the funnel first",
		          i, static_cast<int>(out_codes[i]));
	}
	CHECK_MSG(out_scale.m == INT64_C(-99) && out_scale.e == INT64_C(-99),
	          "out_scale == (%lld, %lld) after rejection, want the poison pair (-99, -99) untouched",
	          static_cast<long long>(out_scale.m), static_cast<long long>(out_scale.e));
}

// §4.1 (F-S3-1)'s mandated red cell: "an i-exp-sigmoid implementation fails
// the site's fixture -- the negative control that makes F-S3-1 unable to
// recur silently." The witness below is pinned from an out-of-repo scratch
// execution (test-design record §5, `mlp_act_iexp_mutation_proof.py`):
// gate_code[60]=60's own sigmoid value differs between the C10 LUT (already
// proven, per Tools/superslm_spike/tests/test_silu_lut_reference_
// reconciliation.py Group C, to disagree with the excluded i-exp-sigmoid
// construction on real activations) and the i-exp-sigmoid construction
// (computed by the ALREADY-VENDORED, ALREADY-SHIPPED reference
// `pipeline._vec_sigmoid`/`pipeline.vec_i_exp`, never hand re-derived), and
// that divergence survives the SAME already-shipped C++ funnel
// (RequantChainChecked) into a DIFFERENT requantized int8 code at index 60:
// -64 (LUT) vs -65 (i-exp-sigmoid). Combined with the feature-oracle cell
// above (which asserts out_codes[60] == -64, the LUT value), this proves
// -- by execution, not by argument -- that an implementation of MlpActSite
// substituting i-exp-sigmoid for the LUT at this fixture would fail that
// cell's own out_codes[60] assertion.
static void TestMlpActSiteC34IExpSigmoidWitnessDivergesFromTheLutAtIndex60() {
	using superslm::CarriedScale;
	using superslm::SslmForwardStatus;
	using namespace superslm_test;

	constexpr size_t kN = 64;
	int8_t gate_code[kN];
	int8_t up_code[kN];
	for (size_t i = 0; i < kN; ++i) {
		gate_code[i] = static_cast<int8_t>(((static_cast<int>(i) * 41 + 13) % 255) - 127);
		up_code[i] = static_cast<int8_t>(((static_cast<int>(i) * 17 + 5) % 255) - 127);
	}

	// Pinned by `mlp_act_iexp_mutation_proof.py` (out-of-repo scratch,
	// executed 2026-07-29): pipeline._vec_sigmoid(gate_code, realscale=
	// 2^30 * 2^-34) at each of the 64 gate codes above -- the i-exp-sigmoid
	// construction F-S3-1 cites, called through the vendored reference, never
	// hand re-derived.
	static const int32_t kIexpSigmoidWitness[kN] = {
	    0,     284,   3837,  20911, 31439, 32673, 0,     142,   2267,  16384, 30501, 32626, 32768,
	    95,    1329,  11857, 28931, 32484, 32768, 47,    746,   7956,  26540, 32205, 32768, 0,
	    424,   5050,  23175, 31795, 32721, 0,     237,   3073,  18938, 31092, 32673, 0,     142,
	    1805,  14342, 29893, 32531, 32768, 47,    1018,  10039, 27994, 32391, 32768, 0,     563,
	    6538,  25173, 32067, 32721, 0,     330,   4059,  21368, 31528, 32673, 0,     189};

	const CarriedScale gate_scale{/*m=*/INT64_C(1073741824), /*e=*/-34};
	const CarriedScale up_scale{/*m=*/INT64_C(1073741824), /*e=*/-18};
	const CarriedScale site_constant{/*m=*/INT64_C(1073741824), /*e=*/0};
	const CarriedScale incoming[] = {gate_scale, up_scale};

	std::vector<int64_t> wide_iexp(kN);
	for (size_t i = 0; i < kN; ++i) {
		wide_iexp[i] = static_cast<int64_t>(gate_code[i]) *
		               static_cast<int64_t>(kIexpSigmoidWitness[i]) *
		               static_cast<int64_t>(up_code[i]);
	}
	std::vector<int8_t> codes_iexp(kN, INT8_C(-99));
	CarriedScale scale_iexp{INT64_C(-99), INT64_C(-99)};
	auto result_iexp =
	    superslm::RequantChainChecked(wide_iexp.data(), kN, std::span<const CarriedScale>{incoming, 2},
	                                    site_constant, codes_iexp.data(), &scale_iexp);
	CHECK_MSG(result_iexp.status == SslmForwardStatus::Ok,
	          "the i-exp-sigmoid witness row must itself be accepted by the already-shipped funnel "
	          "(the divergence this cell proves is a VALUE divergence, not a domain rejection): got %s",
	          SslmForwardStatusName(result_iexp.status));

	CHECK_MSG(codes_iexp[60] == INT8_C(-65),
	          "the i-exp-sigmoid witness's own requantized code at index 60 == %d, want -65 -- this "
	          "pins the executed divergence `mlp_act_iexp_mutation_proof.py` found (LUT: -64, "
	          "i-exp-sigmoid: -65); if this fails, the pinned witness or the funnel's own behavior "
	          "has drifted and this cell's discriminating claim needs re-deriving",
	          static_cast<int>(codes_iexp[60]));

	// The feature-oracle cell above computes the SAME fixture's LUT-based
	// out_codes[60] independently and asserts it equals the real MlpActSite's
	// output. Recomputed here too, so this cell stands on its own (not
	// dependent on execution order) and pins the LUT side of the divergence
	// directly: the real, already-shipped SiluSigmoidQ15/kSiluLutGoldenTable
	// at gate_code[60], through the SAME funnel, must NOT equal codes_iexp[60].
	using superslm::SiluSigmoidQ15;
	std::vector<int64_t> wide_lut(kN);
	for (size_t i = 0; i < kN; ++i) {
		const int32_t sig =
		    SiluSigmoidQ15(kSiluLutGoldenTable, gate_code[i], gate_scale.m, gate_scale.e);
		wide_lut[i] = static_cast<int64_t>(gate_code[i]) * static_cast<int64_t>(sig) *
		              static_cast<int64_t>(up_code[i]);
	}
	std::vector<int8_t> codes_lut(kN, INT8_C(-99));
	CarriedScale scale_lut{INT64_C(-99), INT64_C(-99)};
	auto result_lut =
	    superslm::RequantChainChecked(wide_lut.data(), kN, std::span<const CarriedScale>{incoming, 2},
	                                    site_constant, codes_lut.data(), &scale_lut);
	CHECK_MSG(result_lut.status == SslmForwardStatus::Ok, "the LUT row must be accepted: got %s",
	          SslmForwardStatusName(result_lut.status));
	CHECK_MSG(codes_lut[60] == INT8_C(-64),
	          "the LUT-based row's own requantized code at index 60 == %d, want -64 (§4.1's own "
	          "citation of the excluded construction, executed against this fixture)",
	          static_cast<int>(codes_lut[60]));
	CHECK_MSG(codes_lut[60] != codes_iexp[60],
	          "codes_lut[60] (%d) must differ from codes_iexp[60] (%d) -- this is the executed "
	          "divergence that makes F-S3-1 unable to recur silently: an implementation computing "
	          "i-exp-sigmoid at this site would produce -65 here, not the LUT's -64, and would "
	          "therefore fail the feature-oracle cell's own out_codes[60] assertion",
	          static_cast<int>(codes_lut[60]), static_cast<int>(codes_iexp[60]));
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
	//     the cross-model isolation cell -- storage is model-scoped and green
	//     (Poirot 380b75f review O4; mutation-checked against a reversion to
	//     process-global storage by M6, which fails this cell's own two crux
	//     assertions). ---
	TestTraceHookCrossModelHandleIsolation();

	// --- Poirot review 380b75f (2026-07-28) test-coverage findings: N1 (the
	//     funnel's own running product, RED against today's unguarded fold);
	//     N5's second half (a null-row cell for NarrowRowChecked); N6 (Load
	//     clears a previously-installed trace hook, pinned as the correct
	//     behaviour). Curie's own record: Claude/Curie/superslm-s3.1-checked-
	//     chain-funnel-test-design-2026-07-28.md Sec12. ---
	TestRequantChainCheckedRunningProductOutOfInt32DomainIsRejectedNotTruncated();
	TestRequantChainCheckedRejectsIncomingFactorMantissaOutOfInt32Domain();
	TestRequantChainCheckedStepZeroPrecedesC29OnARowViolatingBoth();
	TestNarrowRowCheckedZeroLenNullPtrDoesNotCrash();
	TestSslmModelLoadClearsPreviouslyInstalledTraceHook();

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

	// --- S3.1 (§5.4, §7.2, D-SLM318): the C34 derived-operand predicate
	//     (CheckSiluCompositionScaleDomain, checked_chain_funnel.h/.cpp) is real
	//     and every cell below calls it directly -- N3 (Poirot 380b75f review)
	//     closed: the containment cell used to call a test-local
	//     reimplementation instead. ---
	TestCheckSiluCompositionScaleDomainContainsTheShippedLoadTimeDescriptor();
	TestCheckSiluCompositionScaleDomainAgreesWithIndependentFormulaAcrossMESweep();
	TestCheckSiluCompositionScaleDomainRejectsMOutsideNoUbAbsBound();

	// --- S3.2 (Sec11, C31/C24/C25/C28/F-S3-8/BIA1): the weightless and
	//     projection sites' red suite, authored against the header contract
	//     (commit a594dd2). Every one of these is RED-UNIMPLEMENTED today --
	//     Claude/Curie/superslm-s3.2-weightless-and-projection-sites-test-
	//     design-2026-07-28.md Sec4/Sec9. ---
	TestFloorDivI64C31UnitWitnessDivergesFromNativeTruncatingDivision();
	TestRmsNormSiteC31UsesFloorDivisionMechanismPin();
	TestRmsNormSiteC31FloorDivisionWitnessAgainstTheRealFunnel();
	TestApplyWeightScaleFoldC24IdentityVsNearIdentityAgainstTheRealFunnel();
	TestCheckRoundingDivideByPotExponentDomainC28BoundaryMatrix();
	TestBiasReconcileC28SignInvertedNegativeControl();
	TestBia1RejectsHostileMagnitudeBothSignsAndAcceptsTheBoundary();
	TestEmbedEntryRejectsHostileTokenIdBeforeAnyReadAndAcceptsTheBoundary();
	TestRmsNormSiteCarriedScaleIsGainDerivedNotIncomingScale();

	// --- Sec13.1 cells 2 and 3 (D-SLM417, board T-1336): owed against S3.2's
	//     already-green production code. ---
	TestIndependentReaderRecoversBia1AndBiasQbMatchingModelView();
	TestIndependentReaderFlagsHandCorruptedBia1TensorWithoutGoingThroughLoad();
	TestTokenizerDriveEveryEncodedIdAddressesItsOwnValidEmbeddingRow();

	// S3.3 -- the attention interior (Claude/Curie/
	// superslm-s3.3-attention-interior-test-design-2026-07-28.md).
	TestCtxFoldJoinIdentityVsIndependentlyDecomposedNonIdentityOnHandBuiltWsc1();
	TestC33ClampWitnessesGenuinelyExceedTheirTargetRangesAgainstTheRealRopeApplyPair();
	TestKvLandingReciprocalBoundMatchesTheRealDynamicScaleReciprocalDomainEndpoints();

	// S3.3, second pass -- the header contract landed; the blocked cells are
	// now authored (record Sec6-Sec8).
	TestLandingRescaleFeatureOracleAgainstResidualReconcileWitnesses();
	TestLandingRescaleSaturationCounterExactValueOnKnownClampedElements();
	TestLandingRescaleSaturationCounterMonotoneAcrossTokens();
	TestLandingRescaleSaturationCounterWholeRowClamp();
	TestLandingRescaleSaturationCountHasNoEffectOnReturnValue();
	TestClampRopeCodeAgainstC33Witnesses();
	TestCheckSoftmaxRowWidthDomainAgainstDerivedCasesAndTheRoutedBandCase();
	TestSoftmaxRowQ15AgainstComposedShippedPrimitivesOracle();
	TestGemmProbQ15AccumulateFeatureOracleAndOrderFreedomCertification();
	TestKvLandingScalesRejectsHostileMantissaBothSignsAndAcceptsTheBoundary();
	TestKvLandingReciprocalsRejectsHostileMagnitudeBothSignsAndAcceptsTheBoundary();
	TestKvLandingReciprocalsExponentFloorAcceptsAtBoundRejectsOnePast();
	TestKvLandingScalesAndReciprocalsRejectMakeMinimalValidKvc1sOwnGammaRow();

	// S3.3 red-regression suite (Curie, 2026-07-28) -- Claude/Curie/
	// superslm-s3.3-attention-interior-red-regression-2026-07-28.md.
	TestCheckSoftmaxRowWidthDomainRejectsAWitnessWhoseOwnThresholdOverflowsInt64();
	TestLandingRescaleIsOddSymmetricInMAAgainstResidualReconcile();
	TestLandingRescaleSaturationCounterFiresOnAnExtremeUncheckedExponent();
	TestSoftmaxRowQ15RefusesATripleWhoseIExpConstructionIsInvalid();

	// Remediation-confirmation red suite (Curie, 2026-07-28) -- Claude/Poirot/
	// ad6bd09-s3.3-remediation-confirmation-review-2026-07-28.md.
	TestLandingRescaleSaturationCounterFiresOnRoundDivideBranchPrecisionLoss();
	TestKvLandingReciprocalsLoadRejectsAnExtremeUncheckedExponentRegardlessOfRT();

	// C32 softmax-row width-gate red suite (Curie, 2026-07-28) -- Claude/Popper/
	// superslm-c32-softmax-denominator-2026-07-28.md.
	TestCheckSoftmaxRowWidthDomainAcceptsAWitnessWhoseRealPeakVastlyExceedsTheCeiling();
	TestSoftmaxRowQ15NeverReportsWellFormedWithAnOutOfRangeProbability();
	TestCheckSoftmaxRowWidthDomainMZeroBypassIsIndependentOfWidth();
	TestCheckSoftmaxRowWidthDomainMustNotBeMorePermissiveForNegativeMThanPositiveOfEqualMagnitude();
	TestSoftmaxRowQ15RejectsOffRatioWitnessWithNonnegativeQcThatPassesTheGate();

	// T-1324 (BLOCKING; D-SLM409) -- Claude/Curie/72b0c7f-s3.3-rope-site-and-
	// c32-softmax-confirmation-test-design-2026-07-28.md.
	TestSoftmaxRowQ15MustNotReportWellFormedWhenShiftedMaxElementExceedsTheSafeShiftCeiling();

	// S3.3 -- the RoPE application site (D-SLM376, D-SLM383, D-SLM384,
	// D-SLM385; Claude/Curie/superslm-s3.3-rope-application-site-test-design-
	// 2026-07-28.md). The three cells below are the first C++ exercise of a
	// real, declared, deliberately-wrong RopeApplySite (commit 13dfcfd's
	// stub, unconditionally WorkspaceTooSmall) -- the fully specified Sec6
	// cells, now landed red against the real symbol:
	TestCheckPositionOverCapBoundaryMatrixAcrossMultipleCaps();
	TestRopeSitePositionZeroAndCapMinusOneAgainstRealPrimitives();
	TestRopeTableSectionRoundTripsThroughRealLoadAtEveryPinnedRow();

	// §13.1 cell 4 -- the config-geometry x tensor-shape join at
	// SslmModel::Load (D-SLM410, D-SLM420-D-SLM423, board T-1333). Symbols
	// declared and stubbed at commit 6bb6b92; every hostile cell below is red
	// against the real symbols' unconditional-Ok stub bodies.
	TestCell4LoadAcceptsFullyConformantConfigGeometryAndRopeShapeJoin();
	TestCell4LoadRejectsHiddenSizeMismatchAgainstHeadsTimesHeadDim();
	TestCell4LoadRejectsHeadsNotDivisibleByKvHeads();
	TestCell4LoadRejectsKvHeadsExceedsHeads();
	TestCell4LoadRejectsRopeCosShapeMismatchWithSinCorrect();
	TestCell4LoadRejectsRopeSinShapeMismatchWithCosCorrect();

	TestRopeApplySiteFeatureOracleAtPositionZeroAndCapMinusOne();
	TestRopeApplySiteRejectsPositionAtOrAboveCapBeforeAnyTableRead();
	TestRopeApplySiteGuardFiresBeforeOutOfBoundsTensorReadUnderAsan();

	// Poirot fa3189a-s3.3-rope-site-and-c32-softmax-review-2026-07-28.md remediation
	// red suite (Curie, 2026-07-28).
	TestRopeApplySiteRejectsMissingCosSinTensorsInsteadOfDereferencingNull();
	TestRopeApplySiteRejectsPositionFarPastTensorExtentInsteadOfReadingUnmappedMemory();
	TestRopeApplySiteReturnsOkWhenReadingPastTensorExtentWithinContextCap();
	TestParseConfigRejectsOddHeadDimAtLoadTime();

	// S3.4 -- the SwiGLU activation site (C34, §5.4, §6.3 step 11; T-1345;
	// Claude/Curie/superslm-s3.4-mlp-act-site-test-design-2026-07-29.md). The
	// three cells below are the first C++ exercise of a real, declared,
	// deliberately-wrong MlpActSite (WorkspaceTooSmall stub, this pass's own
	// declare-and-stub commit).
	TestMlpActSiteC34FeatureOracleAgainstTheRealFunnelAndLut();
	TestMlpActSiteC34RejectsOutOfDomainGateScaleBeforeComputingSigmoid();
	TestMlpActSiteC34IExpSigmoidWitnessDivergesFromTheLutAtIndex60();

	std::printf("superslm tests: %d checks, %d failures\n", GChecks, GFailures);
	return GFailures == 0 ? 0 : 1;
}

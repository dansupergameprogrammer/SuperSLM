// tests/areas/tensor_manifest.cpp -- T-1574 test suite split, Stage 5,
// amended during Stage 6 prep.
// Area #3 (Claude/Plans/SuperSLM_TestSuiteSplit_Plan.md §4): tests calling
// through src/model.cpp's structural sub-parsers (SslmTensorManifest::Parse
// for WGT1/BIA1/ROP1/WeightScales; SslmKeyedConstants::Parse for KVC1;
// ParseConfig for CFG1; ParseSigmoidLut for SIL1). Extracted verbatim from
// tests/test_main.cpp (order keys 111-174, then 226-237); no local fixtures
// owned.
//
// **Amendment (order keys 226-237, added after Stage 5's own commit).** The
// design record's superseded legacy-content column filed these under a
// "S2.4 SiLU sigmoid-LUT" campaign slot alongside `SiluSigmoidQ15` (silu_lut.h,
// candidate area #7); the design record's own header comment on this slot
// (originally at test_main.cpp, "Curie's S2.4 SiLU sigmoid-LUT red suite")
// names ParseSigmoidLut (src/model.cpp) and SiluSigmoidQ15 (src/silu_lut.cpp)
// as two DISTINCT code-under-test surfaces sharing one campaign banner. The
// plan's own §4 table settles which area each belongs to by production TU,
// not campaign slot: area #3's row names `model.cpp`; area #7's row names
// only `silu_lut.cpp`. Every test in this amendment calls ParseSigmoidLut
// (model.cpp) -- the structural SIL1 sub-parse, exactly the same shape as
// this area's WGT1/BIA1/ROP1/KVC1/CFG1/WeightScales cells -- never
// SiluSigmoidQ15, so §3.1's rule (area is the contract actually called)
// places them here, not in silu_lut.cpp. This was missed at Stage 5's own
// commit (the SIL1 block was not adjacent to the rest of this area's content
// and read, at a glance, as belonging to the upcoming SiLU slot) and is
// closed here, before Stage 8 (silu_lut.cpp) could extract this content
// under the wrong area by physical proximity. `TestArtifactRejectsConfigOnly
// V2MissingSigmoidLut` (order 238, physically adjacent to this block) calls
// only `SslmArtifact::OpenFromMemory` with no `ParseSigmoidLut` call at all --
// area #1 loader.cpp's own contract -- and is routed there as the same
// amendment's second half, not here.

#include "superslm/artifact.h"
#include "superslm/model.h"
#include "superslm/silu_lut_canonical.h"
#include "sslm_cfg1_hostile_fixtures.h"
#include "sslm_fixtures.h"
#include "sslm_kvc1_hostile_fixtures.h"
#include "sslm_model_hostile_fixtures.h"
#include "sslm_sil1_hostile_fixtures.h"

#include "support/test_harness.h"
#include "support/test_registry.h"

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

using namespace superslm;
using namespace superslm_test;

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

SSLM_TEST(TestWgtMinimalManifestParsesAndRoundTrips, 111) {
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

SSLM_TEST(TestBiaMinimalManifestParsesAndRoundTrips, 112) {
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

SSLM_TEST(TestRopMinimalManifestParsesAndRoundTrips, 113) {
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

SSLM_TEST(TestManifestRejectsSectionTooShort, 114) {
	std::vector<uint8_t> bytes(10, 0);  // < kManifestHeaderBytes (16)
	bytes[0] = 'W';
	bytes[1] = 'G';
	bytes[2] = 'T';
	bytes[3] = '1';
	AssertManifestRejected(bytes, SslmSectionType::Weights, SslmDtype::Int8, SslmModelStatus::SectionTooShort,
	                        "WGT1 section too short (10 bytes < 16)");
}

SSLM_TEST(TestManifestRejectsBadMagicWgt, 115) {
	auto m = MakeMinimalValidManifest(kWeightsMagic, 1);
	m.bytes[0] = 'X';  // was 'W' of "WGT1"
	AssertManifestRejected(m.bytes, SslmSectionType::Weights, SslmDtype::Int8, SslmModelStatus::BadManifestMagic,
	                        "WGT1 bad magic");
}

SSLM_TEST(TestManifestRejectsBadMagicBia, 116) {
	auto m = MakeMinimalValidManifest(kBiasesMagic, 4);
	m.bytes[0] = 'X';  // was 'B' of "BIA1"
	AssertManifestRejected(m.bytes, SslmSectionType::Biases, SslmDtype::Int32, SslmModelStatus::BadManifestMagic,
	                        "BIA1 bad magic");
}

SSLM_TEST(TestManifestRejectsBadMagicRop, 117) {
	auto m = MakeMinimalValidManifest(kRopeMagic, 8);
	m.bytes[0] = 'X';  // was 'R' of "ROP1"
	AssertManifestRejected(m.bytes, SslmSectionType::RopeTables, SslmDtype::Int64, SslmModelStatus::BadManifestMagic,
	                        "ROP1 bad magic");
}

SSLM_TEST(TestManifestRejectsUnsupportedVersion, 118) {
	auto m = MakeMinimalValidManifest(kWeightsMagic, 1);
	PutU32(m.bytes, kManifestVersionOff, 2);
	AssertManifestRejected(m.bytes, SslmSectionType::Weights, SslmDtype::Int8,
	                        SslmModelStatus::UnsupportedManifestVersion, "WGT1 version == 2");
}

SSLM_TEST(TestManifestRejectsTooManyTensors, 119) {
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

SSLM_TEST(TestManifestRejectsManifestOutOfBoundsTruncatedDescriptors, 120) {
	auto m = MakeMinimalValidManifest(kWeightsMagic, 1);
	// Cut the buffer to strictly inside the descriptor table: one full
	// descriptor plus half of a second, while tensor_count (4) still declares
	// a full four-descriptor table the truncated buffer no longer holds.
	m.bytes.resize(kManifestHeaderBytes + kTensorDescBytes + kTensorDescBytes / 2);
	AssertManifestRejected(m.bytes, SslmSectionType::Weights, SslmDtype::Int8, SslmModelStatus::ManifestOutOfBounds,
	                        "WGT1 truncated mid-descriptor-table");
}

SSLM_TEST(TestManifestRejectsManifestOutOfBoundsTruncatedNameBlob, 121) {
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

SSLM_TEST(TestManifestRejectsBadTensorNameOutOfRange, 122) {
	auto m = MakeMinimalValidManifest(kWeightsMagic, 1);
	PutU32(m.bytes, ManifestDescNameLenOff(0), m.name_blob_len + 50);  // t1's name now runs past the blob
	AssertManifestRejected(m.bytes, SslmSectionType::Weights, SslmDtype::Int8, SslmModelStatus::BadTensorName,
	                        "WGT1 tensor[0] (t1) name range exceeds name_blob_len");
}

SSLM_TEST(TestManifestRejectsEmptyTensorName, 123) {
	auto m = MakeMinimalValidManifest(kWeightsMagic, 1);
	PutU32(m.bytes, ManifestDescNameLenOff(0), 0);
	AssertManifestRejected(m.bytes, SslmSectionType::Weights, SslmDtype::Int8, SslmModelStatus::EmptyTensorName,
	                        "WGT1 tensor[0] (t1) name_len == 0");
}

SSLM_TEST(TestManifestRejectsDuplicateTensorName, 124) {
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

SSLM_TEST(TestManifestRejectsBadTensorRankZero, 125) {
	auto m = MakeMinimalValidManifest(kWeightsMagic, 1);
	PutU32(m.bytes, ManifestDescRankOff(0), 0);
	AssertManifestRejected(m.bytes, SslmSectionType::Weights, SslmDtype::Int8, SslmModelStatus::BadTensorRank,
	                        "WGT1 tensor[0] (t1) rank == 0");
}

SSLM_TEST(TestManifestRejectsBadTensorRankTooLarge, 126) {
	auto m = MakeMinimalValidManifest(kWeightsMagic, 1);
	PutU32(m.bytes, ManifestDescRankOff(0), kMaxTensorRank + 1);  // 5
	AssertManifestRejected(m.bytes, SslmSectionType::Weights, SslmDtype::Int8, SslmModelStatus::BadTensorRank,
	                        "WGT1 tensor[0] (t1) rank == kMaxTensorRank+1 (5)");
}

SSLM_TEST(TestManifestRejectsBadTensorShapeZeroDim, 127) {
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

SSLM_TEST(TestManifestRejectsBadTensorShapeNonzeroPastRank, 128) {
	auto m = MakeMinimalValidManifest(kWeightsMagic, 1);
	// t1 (tensor[0]) is rank 1, shape [3,0,0,0]; set shape[1] (past rank)
	// nonzero. elem_count (3, the product over shape[0..rank)) is unaffected.
	PutU32(m.bytes, ManifestDescShapeOff(0, 1), 5);
	AssertManifestRejected(m.bytes, SslmSectionType::Weights, SslmDtype::Int8, SslmModelStatus::BadTensorShape,
	                        "WGT1 tensor[0] (t1) shape[1] != 0 past rank 1");
}

SSLM_TEST(TestManifestRejectsShapeCountMismatch, 129) {
	auto m = MakeMinimalValidManifest(kWeightsMagic, 1);
	// t3 (tensor[2]) is rank 3, shape [2,1,2] -> product 4; declare 999.
	PutU64(m.bytes, ManifestDescElemCountOff(2), 999);
	AssertManifestRejected(m.bytes, SslmSectionType::Weights, SslmDtype::Int8, SslmModelStatus::ShapeCountMismatch,
	                        "WGT1 tensor[2] (t3) elem_count 999 != product(shape) 4");
}

SSLM_TEST(TestManifestRejectsTensorOutOfBoundsDataExceedsSection, 130) {
	auto m = MakeMinimalValidManifest(kWeightsMagic, 1);
	// Push t4's (tensor[3]) data_off to one byte before the end of the buffer;
	// its declared elem_count (2) then needs 2 bytes there, exceeding
	// byte_size by 1. t4 is the last tensor packed, so this cannot also
	// overlap t1/t2/t3's earlier ranges.
	PutU64(m.bytes, ManifestDescDataOffOff(3), m.bytes.size() - 1);
	AssertManifestRejected(m.bytes, SslmSectionType::Weights, SslmDtype::Int8, SslmModelStatus::TensorOutOfBounds,
	                        "WGT1 tensor[3] (t4) data range exceeds byte_size by 1");
}

SSLM_TEST(TestManifestRejectsTensorOverlap, 131) {
	auto m = MakeMinimalValidManifest(kWeightsMagic, 1);
	// Point t2's (tensor[1]) data_off at t1's (tensor[0]) -- both then claim
	// overlapping byte ranges, while the claimed range still fits comfortably
	// inside byte_size (so TensorOutOfBounds cannot also fire).
	PutU64(m.bytes, ManifestDescDataOffOff(1), m.tensor_data_off[0]);
	AssertManifestRejected(m.bytes, SslmSectionType::Weights, SslmDtype::Int8, SslmModelStatus::TensorOverlap,
	                        "WGT1 tensor[1] (t2) data_off == tensor[0]'s (t1)");
}

SSLM_TEST(TestManifestRejectsBadDescriptorReserved, 132) {
	auto m = MakeMinimalValidManifest(kWeightsMagic, 1);
	PutU32(m.bytes, ManifestDescReservedOff(0), 1);
	AssertManifestRejected(m.bytes, SslmSectionType::Weights, SslmDtype::Int8, SslmModelStatus::BadDescriptorReserved,
	                        "WGT1 tensor[0] (t1) reserved == 1");
}

SSLM_TEST(TestManifestRejectsDataOffBelowDataRegion, 133) {
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

SSLM_TEST(TestManifestRejectsTensorMisalignedBia, 134) {
	auto m = MakeSingleTensorManifest(kBiasesMagic, 4, {4});
	m.bytes.resize(m.bytes.size() + 64, 0);  // headroom: isolate misalignment from bounds
	PutU64(m.bytes, ManifestDescDataOffOff(0), m.tensor_data_off[0] + 1);  // +1: not a multiple of 4
	AssertManifestRejected(m.bytes, SslmSectionType::Biases, SslmDtype::Int32, SslmModelStatus::TensorMisaligned,
	                        "BIA1 tensor[0] (t0) data_off + 1 (not a multiple of element_size 4)");
}

SSLM_TEST(TestManifestRejectsTensorMisalignedRop, 135) {
	auto m = MakeSingleTensorManifest(kRopeMagic, 8, {4});
	m.bytes.resize(m.bytes.size() + 64, 0);  // headroom: isolate misalignment from bounds
	PutU64(m.bytes, ManifestDescDataOffOff(0), m.tensor_data_off[0] + 4);  // +4: not a multiple of 8
	AssertManifestRejected(m.bytes, SslmSectionType::RopeTables, SslmDtype::Int64, SslmModelStatus::TensorMisaligned,
	                        "ROP1 tensor[0] (t0) data_off + 4 (not a multiple of element_size 8)");
}

SSLM_TEST(TestManifestRejectsElemCountTimesElementSizeOverflows32BitBia, 136) {
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

SSLM_TEST(TestManifestRejectsElemCountTimesElementSizeOverflows32BitRop, 137) {
	auto m = MakeSingleTensorManifest(kRopeMagic, 8, {1});
	// element_size 8 needs a smaller elem_count to overflow 32 bits at the
	// second multiplication: 600,000,000 * 8 == 4,800,000,000.
	PutU32(m.bytes, ManifestDescShapeOff(0, 0), 600000000u);
	PutU64(m.bytes, ManifestDescElemCountOff(0), 600000000ull);
	AssertManifestRejected(m.bytes, SslmSectionType::RopeTables, SslmDtype::Int64, SslmModelStatus::TensorOutOfBounds,
	                        "ROP1 tensor[0] (t0) elem_count(600M) * element_size(8) overflows 32 bits");
}

SSLM_TEST(TestManifestRejectsShapeProductOverflows32BitTensorOutOfBounds, 138) {
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

SSLM_TEST(TestCompositionConstantsMinimalKvc1ParsesAndRoundTrips, 139) {
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

SSLM_TEST(TestKvLandingReciprocalsMinimalKvc1ParsesAndRoundTrips, 140) {
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

SSLM_TEST(TestKvc1RejectsSectionTooShort, 141) {
	std::vector<uint8_t> bytes(20, 0);  // < kConstantHeaderBytes (24)
	bytes[0] = 'K';
	bytes[1] = 'V';
	bytes[2] = 'C';
	bytes[3] = '1';
	AssertKvc1Rejected(bytes, SslmSectionType::CompositionConstants, SslmModelStatus::SectionTooShort,
	                    "KVC1 section too short (20 bytes < 24)");
}

SSLM_TEST(TestKvc1RejectsBadMagic, 142) {
	auto m = MakeMinimalValidKvc1(2);
	m.bytes[0] = 'X';  // was 'K' of "KVC1"
	AssertKvc1Rejected(m.bytes, SslmSectionType::CompositionConstants, SslmModelStatus::BadConstantsMagic,
	                    "KVC1 bad magic");
}

SSLM_TEST(TestKvc1RejectsUnsupportedVersion, 143) {
	auto m = MakeMinimalValidKvc1(2);
	PutU32(m.bytes, kConstantsVersionOff, 2);
	AssertKvc1Rejected(m.bytes, SslmSectionType::CompositionConstants, SslmModelStatus::UnsupportedConstantsVersion,
	                    "KVC1 version == 2");
}

SSLM_TEST(TestKvc1RejectsTooManyEntries, 144) {
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

SSLM_TEST(TestKvc1RejectsBadReserved, 145) {
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

SSLM_TEST(TestKvc1RejectsOutOfBoundsTruncatedDescriptors, 146) {
	auto m = MakeMinimalValidKvc1(2);  // 3 entries
	// Cut the buffer to strictly inside the descriptor table: one full
	// descriptor plus half of a second, while entry_count (3) still declares
	// a full three-descriptor table the truncated buffer no longer holds.
	m.bytes.resize(m.descriptors_off + superslm::kConstantDescBytes + superslm::kConstantDescBytes / 2);
	AssertKvc1Rejected(m.bytes, SslmSectionType::CompositionConstants, SslmModelStatus::ConstantsOutOfBounds,
	                    "KVC1 truncated mid-descriptor-table");
}

SSLM_TEST(TestKvc1RejectsOutOfBoundsTruncatedValues, 147) {
	auto m = MakeMinimalValidKvc1(2);  // 3 entries * 2 words * 8 bytes == 48 value bytes
	// The full descriptor table is present, but not all of the declared
	// entry_count*value_words int64s after it -- cut one byte short of the
	// value array.
	m.bytes.resize(m.values_off + static_cast<size_t>(m.entry_count) * m.value_words * 8 - 1);
	AssertKvc1Rejected(m.bytes, SslmSectionType::CompositionConstants, SslmModelStatus::ConstantsOutOfBounds,
	                    "KVC1 truncated mid-value-array");
}

SSLM_TEST(TestKvc1RejectsOutOfBoundsTruncatedNameBlob, 148) {
	auto m = MakeMinimalValidKvc1(2);
	// The full descriptor table and value array are present, but not all of
	// the declared name_blob_len bytes after them -- cut one byte short of
	// the name blob.
	m.bytes.resize(m.name_blob_off + m.name_blob_len - 1);
	AssertKvc1Rejected(m.bytes, SslmSectionType::CompositionConstants, SslmModelStatus::ConstantsOutOfBounds,
	                    "KVC1 truncated mid-name-blob");
}

SSLM_TEST(TestKvc1RejectsNameBlobLenSumOverflow32Bit, 149) {
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

SSLM_TEST(TestKvc1RejectsBadEntryNameOutOfRange, 150) {
	auto m = MakeMinimalValidKvc1(2);
	PutU32(m.bytes, ConstantsDescNameLenOff(0), m.name_blob_len + 50);  // alpha's name now runs past the blob
	AssertKvc1Rejected(m.bytes, SslmSectionType::CompositionConstants, SslmModelStatus::BadEntryName,
	                    "KVC1 entry[0] (alpha) name range exceeds name_blob_len");
}

SSLM_TEST(TestKvc1RejectsEmptyEntryName, 151) {
	auto m = MakeMinimalValidKvc1(2);
	PutU32(m.bytes, ConstantsDescNameLenOff(0), 0);
	AssertKvc1Rejected(m.bytes, SslmSectionType::CompositionConstants, SslmModelStatus::EmptyEntryName,
	                    "KVC1 entry[0] (alpha) name_len == 0");
}

SSLM_TEST(TestKvc1RejectsDuplicateEntryName, 152) {
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

SSLM_TEST(TestKvc1RejectsValueWordsOutOfRange, 153) {
	auto m = MakeMinimalValidKvc1(2);  // KvLandingScales also requires 2
	PutU32(m.bytes, kConstantsValueWordsOff, 5);
	AssertKvc1Rejected(m.bytes, SslmSectionType::KvLandingScales, SslmModelStatus::BadValueWords,
	                    "KVC1 (KvLandingScales) value_words == 5, not in {2,3}");
}

SSLM_TEST(TestKvc1RejectsValueWordsWrongForTypeCompositionDeclaresThree, 154) {
	auto m = MakeMinimalValidKvc1(2);  // CompositionConstants requires 2
	PutU32(m.bytes, kConstantsValueWordsOff, 3);
	AssertKvc1Rejected(m.bytes, SslmSectionType::CompositionConstants, SslmModelStatus::BadValueWords,
	                    "KVC1 (CompositionConstants, requires 2) value_words == 3");
}

SSLM_TEST(TestKvc1RejectsValueWordsWrongForTypeReciprocalsDeclaresTwo, 155) {
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

SSLM_TEST(TestMinimalCfg1ParsesAndMatchesEveryField, 156) {
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

SSLM_TEST(TestCfg1RejectsSizeTooShort, 157) {
	auto bytes = MakeMinimalValidCfg1();
	bytes.pop_back();  // 83 bytes, < kConfigBytes (84)
	AssertCfg1Rejected(bytes, SslmModelStatus::BadConfigSize, "CFG1 byte_size == 83 (< 84)");
}

SSLM_TEST(TestCfg1RejectsSizeTooLong, 158) {
	auto bytes = MakeMinimalValidCfg1();
	bytes.push_back(0);  // 85 bytes, > kConfigBytes (84)
	AssertCfg1Rejected(bytes, SslmModelStatus::BadConfigSize, "CFG1 byte_size == 85 (> 84)");
}

SSLM_TEST(TestCfg1RejectsBadMagic, 159) {
	auto bytes = MakeMinimalValidCfg1();
	bytes[0] = 'X';  // was 'C' of "CFG1"
	AssertCfg1Rejected(bytes, SslmModelStatus::BadConfigMagic, "CFG1 bad magic");
}

SSLM_TEST(TestCfg1RejectsUnsupportedVersion, 160) {
	auto bytes = MakeMinimalValidCfg1();
	PutU32(bytes, kCfg1VersionOff, 2);
	AssertCfg1Rejected(bytes, SslmModelStatus::UnsupportedConfigVersion, "CFG1 version == 2");
}

// --- BadConfigDim — all eight dimension fields, each its own cell (a parse
//     that wrongly guards only some of them is caught). ---

SSLM_TEST(TestCfg1RejectsZeroHiddenSize, 161) {
	auto bytes = MakeMinimalValidCfg1();
	PutU32(bytes, kCfg1HiddenSizeOff, 0);
	AssertCfg1Rejected(bytes, SslmModelStatus::BadConfigDim, "CFG1 hidden_size == 0");
}

SSLM_TEST(TestCfg1RejectsZeroNumHiddenLayers, 162) {
	auto bytes = MakeMinimalValidCfg1();
	PutU32(bytes, kCfg1NumHiddenLayersOff, 0);
	AssertCfg1Rejected(bytes, SslmModelStatus::BadConfigDim, "CFG1 num_hidden_layers == 0");
}

SSLM_TEST(TestCfg1RejectsZeroNumAttentionHeads, 163) {
	auto bytes = MakeMinimalValidCfg1();
	PutU32(bytes, kCfg1NumAttentionHeadsOff, 0);
	AssertCfg1Rejected(bytes, SslmModelStatus::BadConfigDim, "CFG1 num_attention_heads == 0");
}

SSLM_TEST(TestCfg1RejectsZeroNumKeyValueHeads, 164) {
	auto bytes = MakeMinimalValidCfg1();
	PutU32(bytes, kCfg1NumKeyValueHeadsOff, 0);
	AssertCfg1Rejected(bytes, SslmModelStatus::BadConfigDim, "CFG1 num_key_value_heads == 0");
}

SSLM_TEST(TestCfg1RejectsZeroHeadDim, 165) {
	auto bytes = MakeMinimalValidCfg1();
	PutU32(bytes, kCfg1HeadDimOff, 0);
	AssertCfg1Rejected(bytes, SslmModelStatus::BadConfigDim, "CFG1 head_dim == 0");
}

SSLM_TEST(TestCfg1RejectsZeroIntermediateSize, 166) {
	auto bytes = MakeMinimalValidCfg1();
	PutU32(bytes, kCfg1IntermediateSizeOff, 0);
	AssertCfg1Rejected(bytes, SslmModelStatus::BadConfigDim, "CFG1 intermediate_size == 0");
}

SSLM_TEST(TestCfg1RejectsZeroVocabSize, 167) {
	auto bytes = MakeMinimalValidCfg1();
	PutU32(bytes, kCfg1VocabSizeOff, 0);
	AssertCfg1Rejected(bytes, SslmModelStatus::BadConfigDim, "CFG1 vocab_size == 0");
}

SSLM_TEST(TestCfg1RejectsZeroContextCap, 168) {
	auto bytes = MakeMinimalValidCfg1();
	PutU32(bytes, kCfg1ContextCapOff, 0);
	AssertCfg1Rejected(bytes, SslmModelStatus::BadConfigDim, "CFG1 context_cap == 0");
}

// kv_block_size is a required-nonzero field too (docs "Config blob"; Poirot C-1).
SSLM_TEST(TestCfg1RejectsZeroKvBlockSize, 169) {
	auto bytes = MakeMinimalValidCfg1();
	PutU32(bytes, kCfg1KvBlockSizeOff, 0);
	AssertCfg1Rejected(bytes, SslmModelStatus::BadConfigDim, "CFG1 kv_block_size == 0");
}

SSLM_TEST(TestCfg1RejectsBadKvPrecision, 170) {
	auto bytes = MakeMinimalValidCfg1();
	PutU32(bytes, kCfg1KvPrecisionOff, 2);  // only 0 (Int8) or 1 (Int16) are valid
	AssertCfg1Rejected(bytes, SslmModelStatus::BadKvPrecision, "CFG1 kv_precision == 2");
}

SSLM_TEST(TestCfg1RejectsBadConfigBool, 171) {
	auto bytes = MakeMinimalValidCfg1();
	PutU32(bytes, kCfg1TieWordEmbeddingsOff, 2);  // only 0 or 1 are valid
	AssertCfg1Rejected(bytes, SslmModelStatus::BadConfigBool, "CFG1 tie_word_embeddings == 2");
}

SSLM_TEST(TestCfg1RejectsBadConfigReserved, 172) {
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

SSLM_TEST(TestWeightScalesMinimalManifestParsesAndRoundTrips, 173) {
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

SSLM_TEST(TestWeightScalesRejectsWrongMagicDiscriminatesPerType, 174) {
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

SSLM_TEST(TestMinimalSil1ParsesAndReadsBackAllNodes, 226) {
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
SSLM_TEST(TestSil1WarmObjectRepeatedReadsShowNoDrift, 227) {
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
SSLM_TEST(TestSil1RoundTripReencodeMatchesOriginalBytes, 228) {
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

SSLM_TEST(TestSil1RejectsSizeTooShort, 229) {
	using namespace superslm_test;
	auto bytes = MakeMinimalValidSil1();
	bytes.pop_back();  // 4115 bytes, < kSigmoidLutBytes (4116)
	AssertSil1Rejected(bytes, SslmModelStatus::BadSigmoidLutSize, "SIL1 byte_size == 4115 (< 4116)");
}

SSLM_TEST(TestSil1RejectsSizeTooLong, 230) {
	using namespace superslm_test;
	auto bytes = MakeMinimalValidSil1();
	bytes.push_back(0);  // 4117 bytes, > kSigmoidLutBytes (4116)
	AssertSil1Rejected(bytes, SslmModelStatus::BadSigmoidLutSize, "SIL1 byte_size == 4117 (> 4116)");
}

SSLM_TEST(TestSil1RejectsBadMagic, 231) {
	using namespace superslm_test;
	auto bytes = MakeMinimalValidSil1();
	bytes[kSil1MagicOff] = 'X';  // was 'S' of "SIL1"
	AssertSil1Rejected(bytes, SslmModelStatus::BadSigmoidLutMagic, "SIL1 bad magic");
}

SSLM_TEST(TestSil1RejectsUnsupportedVersion, 232) {
	using namespace superslm_test;
	auto bytes = MakeMinimalValidSil1();
	PutU32(bytes, kSil1VersionOff, 2);  // only 1 is valid (kManifestVersion)
	AssertSil1Rejected(bytes, SslmModelStatus::UnsupportedSigmoidLutVersion, "SIL1 version == 2");
}

// entry_count is a distinct field from byte_size: this cell holds byte_size at
// exactly 4116 (so BadSigmoidLutSize's check would pass) and mutates only the
// header's declared entry_count, isolating BadSigmoidLutCount from the size
// check above.
SSLM_TEST(TestSil1RejectsBadEntryCount, 233) {
	using namespace superslm_test;
	auto bytes = MakeMinimalValidSil1();
	PutU32(bytes, kSil1EntryCountOff, kSigmoidLutEntries - 1);  // 1024, byte_size unchanged at 4116
	AssertSil1Rejected(bytes, SslmModelStatus::BadSigmoidLutCount, "SIL1 entry_count == 1024 (!= 1025)");
}

SSLM_TEST(TestSil1RejectsBadReserved, 234) {
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

SSLM_TEST(TestSil1RejectsSingleNodeContentMismatch, 235) {
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
SSLM_TEST(TestSil1RejectsHostileExtremeAdjacentNodes, 236) {
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
SSLM_TEST(TestArtifactRejectsHostileSigmoidLutContentThroughRealPath, 237) {
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

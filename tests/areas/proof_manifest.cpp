// tests/areas/proof_manifest.cpp -- T-1574 test suite split, Stage 1.
// Area #9 (Claude/Plans/SuperSLM_TestSuiteSplit_Plan.md §4): tests calling
// through src/proof_manifest.cpp's public contract (CheckConfigGeometry,
// ComputeTensorEvidence, ComputeWeightScaleEvidence, HashSectionHex,
// BuildProofManifestJson). Extracted verbatim from tests/test_main.cpp
// (order keys 302-313); no local fixtures owned. The four
// TestBadAllocContract* cells that also call through this file's functions
// stay in tests/test_main.cpp -- §4's table routes all 19 bad-alloc-contract
// sites to the cross-cutting bad_alloc_contract.cpp area (#10, Stage 2), not
// to each function's own "natural" area.

#include "superslm/artifact.h"
#include "superslm/model.h"
#include "superslm/proof_manifest.h"
#include "superslm/sha256.h"
#include "sslm_cfg1_hostile_fixtures.h"
#include "sslm_fixtures.h"
#include "sslm_model_hostile_fixtures.h"
#include "sslm_sil1_hostile_fixtures.h"

#include "support/test_harness.h"
#include "support/test_registry.h"

#include <cstdint>
#include <string>

using namespace superslm;
using namespace superslm_test;

// --- S-HARDEN-3 (F13, §13 item 7, §17.3 cell 4): the independent converter
//     verifier's core -- config geometry x tensor shapes, per-tensor evidence,
//     and the proof-manifest document itself. Curie's red suite (red-first;
//     the whole superslm::proof_manifest translation unit did not exist before
//     this slot -- every symbol below is new). Mendeleev's 2026-07-21 coverage
//     audit §3.3 names the zero-boundary cells explicitly: kv_heads == 0 and,
//     separately, heads == 0, must each produce a DEFINED rejection rather than
//     a fault in the `heads % kv_heads` modulus -- the two cells directly below
//     are that specification, unmodified. ---

SSLM_TEST(TestConfigGeometryRejectsZeroAttentionHeads, 302) {
	const auto r = CheckConfigGeometry(/*hidden_size=*/4096, /*heads=*/0, /*kv_heads=*/8, /*head_dim=*/128);
	CHECK_MSG(r.status == ConfigGeometryStatus::ZeroAttentionHeads,
	          "num_attention_heads == 0 must be a DEFINED rejection (ZeroAttentionHeads), checked before "
	          "any modulus, not a crash: got %s",
	          ConfigGeometryStatusName(r.status));
}

SSLM_TEST(TestConfigGeometryRejectsZeroKeyValueHeadsBeforeModulusFaults, 303) {
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

SSLM_TEST(TestConfigGeometryRejectsKvHeadsExceedsHeads, 304) {
	const auto r = CheckConfigGeometry(/*hidden_size=*/4096, /*heads=*/8, /*kv_heads=*/16, /*head_dim=*/512);
	CHECK_MSG(r.status == ConfigGeometryStatus::KvHeadsExceedsHeads,
	          "num_key_value_heads (16) > num_attention_heads (8): got %s", ConfigGeometryStatusName(r.status));
}

SSLM_TEST(TestConfigGeometryRejectsHeadsNotDivisibleByKv, 305) {
	// 10 heads, 3 kv_heads: 10 % 3 != 0. kv_heads <= heads holds, so this cell
	// isolates the divisibility relation from the ordering relation above it.
	const auto r = CheckConfigGeometry(/*hidden_size=*/4096, /*heads=*/10, /*kv_heads=*/3, /*head_dim=*/128);
	CHECK_MSG(r.status == ConfigGeometryStatus::HeadsNotDivisibleByKv,
	          "num_attention_heads (10) %% num_key_value_heads (3) != 0: got %s",
	          ConfigGeometryStatusName(r.status));
}

SSLM_TEST(TestConfigGeometryRejectsHiddenSizeMismatch, 306) {
	// heads=32, head_dim=128 -> 4096, but hidden_size declares 4097: the GQA
	// relations both hold, isolating the hidden_size x heads*head_dim relation.
	const auto r = CheckConfigGeometry(/*hidden_size=*/4097, /*heads=*/32, /*kv_heads=*/8, /*head_dim=*/128);
	CHECK_MSG(r.status == ConfigGeometryStatus::HiddenSizeGeometryMismatch,
	          "hidden_size (4097) != num_attention_heads * head_dim (32*128=4096): got %s",
	          ConfigGeometryStatusName(r.status));
}

SSLM_TEST(TestConfigGeometryAcceptsGqaShape, 307) {
	const auto r = CheckConfigGeometry(/*hidden_size=*/4096, /*heads=*/32, /*kv_heads=*/8, /*head_dim=*/128);
	CHECK_MSG(r.status == ConfigGeometryStatus::Ok,
	          "a coherent GQA shape (32 heads, 8 kv_heads, head_dim=128, hidden_size=4096) must be Ok: got "
	          "%s (%s)",
	          ConfigGeometryStatusName(r.status), r.diagnostic.c_str());
}

SSLM_TEST(TestConfigGeometryAcceptsMhaShape, 308) {
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

SSLM_TEST(TestComputeTensorEvidenceReportsExtremaAndSaturationBoundary, 309) {
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

SSLM_TEST(TestComputeWeightScaleEvidenceReportsShiftRangeAndIdentityCount, 310) {
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

SSLM_TEST(TestHashSectionHexMatchesIndependentSha256, 311) {
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

SSLM_TEST(TestBuildProofManifestJsonReportsGeometryOkOnCoherentArtifact, 312) {
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

SSLM_TEST(TestBuildProofManifestJsonReportsGeometryMismatchOnIncoherentArtifact, 313) {
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

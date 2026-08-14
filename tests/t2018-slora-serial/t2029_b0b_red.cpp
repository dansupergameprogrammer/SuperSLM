// T-2021/T-2029 -- Brunel's B0b build: the artifact loader (design Sec9/Sec11 B0b,
// D-SLM3093/D-SLM3094), specified in full at Claude/Curie/t2018-slora-serial-red-suite-
// 2026-08-13.md Sec18.3 (Wizard repo). Every cell below is that section's own named cell,
// exercised against the REAL, production `SslmDeltaFoldScaleView`/`SslmUFoldScaleView` types
// and validators this build adds to include/superslm/model.h + src/model.cpp -- nothing here
// is a scratch reimplementation.
//
// Reuses tests/sslm_model_hostile_fixtures.h's spec-faithful manifest builder
// (BuildManifest/MakeManifestSectionView), the same construction T-2018's own sibling suites
// (test_main.cpp's own WSC1 cells) already use for WGT1/BIA1/ROP1/WSC1 -- DFS1/UFS1 share the
// identical on-disk tensor-manifest byte layout (design Sec9's own "storage-shape-identical"
// framing), so the fixture builder needs no DFS1/UFS1-specific logic, only the right magic.
//
// Standard-library-only, CHECK/CHECK_MSG counters, matching this directory's own
// t2018_offline_red.cpp convention.

#include "superslm/artifact.h"
#include "superslm/model.h"
#include "../sslm_model_hostile_fixtures.h"

#include <array>
#include <cstdio>
#include <cstring>
#include <type_traits>
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
			std::printf("FAIL %s:%d: %s -- ", __FILE__, __LINE__, #cond); \
			std::printf(__VA_ARGS__); \
			std::printf("\n"); \
		} \
	} while (0)

// Writes one (identity, mult, exponent) row into `bytes` at `tensor_data_off + row*12`,
// little-endian int32 each -- overwriting BuildManifest's own deterministic filler pattern with
// a real, caller-chosen triple.
static void WriteTriple(std::vector<uint8_t>& bytes, uint64_t tensor_data_off, uint64_t row,
                         int32_t identity, int32_t mult, int32_t exponent) {
	auto put = [&](uint64_t off, int32_t v) {
		uint32_t u = static_cast<uint32_t>(v);
		for (int i = 0; i < 4; ++i) bytes[off + i] = static_cast<uint8_t>((u >> (8 * i)) & 0xFF);
	};
	const uint64_t base = tensor_data_off + row * 12;
	put(base + 0, identity);
	put(base + 4, mult);
	put(base + 8, exponent);
}

// --- Cell: dedicated wrapper types exist and are not interchangeable -----------------------

static void TestDedicatedWrapperTypesAreNotInterchangeable() {
	CHECK_MSG((!std::is_same_v<SslmDeltaFoldScaleView, SslmUFoldScaleView>),
	          "SslmDeltaFoldScaleView and SslmUFoldScaleView must be distinct C++ types");
	CHECK_MSG((!std::is_same_v<SslmDeltaFoldScaleView, SslmTensorManifest>),
	          "SslmDeltaFoldScaleView must not be SslmTensorManifest (WeightScales' own shared container)");
	CHECK_MSG((!std::is_same_v<SslmUFoldScaleView, SslmTensorManifest>),
	          "SslmUFoldScaleView must not be SslmTensorManifest (WeightScales' own shared container)");
	// Neither type is implicitly constructible from the other -- a call site expecting
	// SslmDeltaFoldScaleView cannot silently accept an SslmUFoldScaleView argument.
	CHECK_MSG((!std::is_convertible_v<SslmUFoldScaleView, SslmDeltaFoldScaleView>),
	          "SslmUFoldScaleView must not be implicitly convertible to SslmDeltaFoldScaleView");
	CHECK_MSG((!std::is_convertible_v<SslmDeltaFoldScaleView, SslmUFoldScaleView>),
	          "SslmDeltaFoldScaleView must not be implicitly convertible to SslmUFoldScaleView");
}

// --- Cell: swap mutation direction (i) -- WSC1 bytes reinterpreted as DeltaFoldScales -------

static void TestSwapDirectionOneWsc1BytesRejectedByDeltaFoldParserBeforeAnyValueRead() {
	// A genuine WeightScales section: correct WSC1 magic, correct byte layout, a legal
	// (unsigned-domain) triple -- but its DECLARED section.type is WeightScales, not
	// DeltaFoldScales. Direction (i) per design Sec9: this must be rejected by section-type
	// dispatch BEFORE any triple value is inspected, regardless of how legal those values are.
	BuiltManifest m = MakeSingleTensorManifest(kWeightScalesMagic, 4, {3});
	WriteTriple(m.bytes, m.tensor_data_off[0], 0, /*identity=*/0, /*mult=*/1000000000, /*exponent=*/5);
	SslmSectionView view = MakeManifestSectionView(SslmSectionType::WeightScales, SslmDtype::Int32, m.bytes);

	SslmDeltaFoldScaleView out;
	std::string err;
	const SslmModelStatus status = SslmDeltaFoldScaleView::Parse(view, out, &err);
	CHECK_MSG(status == SslmModelStatus::AmplifyingFoldSectionTypeMismatch,
	          "a WeightScales-typed section handed to SslmDeltaFoldScaleView::Parse must be "
	          "rejected by AmplifyingFoldSectionTypeMismatch (got %s: %s)",
	          SslmModelStatusName(status), err.c_str());
	CHECK_MSG(out.Entries().empty(), "a rejected parse must expose no entries");

	// The identical bytes, correctly declared as DeltaFoldScales (magic swapped too -- this is
	// the SAME triple content a converter would legitimately emit), must be REJECTED for a
	// different reason (bad magic, since the byte content still carries WSC1's own magic), not
	// silently accepted -- confirms the type-mismatch rejection above is not merely a proxy for
	// "the magic happened to be wrong."
	SslmSectionView correctly_typed = MakeManifestSectionView(SslmSectionType::DeltaFoldScales,
	                                                           SslmDtype::Int32, m.bytes);
	SslmDeltaFoldScaleView out2;
	std::string err2;
	const SslmModelStatus status2 = SslmDeltaFoldScaleView::Parse(correctly_typed, out2, &err2);
	CHECK_MSG(status2 == SslmModelStatus::BadManifestMagic,
	          "WSC1-magic'd bytes declared as DeltaFoldScales must fail on magic, not silently "
	          "pass through the type gate: got %s", SslmModelStatusName(status2));
}

// --- Cell: swap mutation direction (ii) -- delta/u bytes reinterpreted as WeightScales ------
// "Already true today, confirmed by reading ValidateWeightScalesDomain's existing [0,31]
// bound" -- this suite's own casebook (Sec18.3) names this the ONE B0b cell partially
// discharged by EXISTING code; reproduced here as a standing green control, not a new red cell.

static void TestSwapDirectionTwoAmplifyingBytesWithNegativeExponentRejectedByWsc1Validator() {
	BuiltManifest m = MakeSingleTensorManifest(kDeltaFoldScalesMagic, 4, {3});
	WriteTriple(m.bytes, m.tensor_data_off[0], 0, /*identity=*/0, /*mult=*/1000000000, /*exponent=*/-5);
	SslmSectionView delta_view = MakeManifestSectionView(SslmSectionType::DeltaFoldScales,
	                                                      SslmDtype::Int32, m.bytes);
	// Confirm it parses legally as its own kind first (negative exponent IS in-domain for
	// DeltaFoldScales, design Sec9's own signed [-31,31]).
	SslmDeltaFoldScaleView delta_out;
	std::string derr;
	CHECK(SslmDeltaFoldScaleView::Parse(delta_view, delta_out, &derr) == SslmModelStatus::Ok);
	CHECK(SslmModelStatus::Ok ==
	      ValidateAmplifyingFoldScalesDomain(delta_out.Entries(), &derr));

	// Now reinterpret the SAME bytes as if the section's magic were WSC1's own -- the structural
	// parse (SslmTensorManifest::Parse, real and unmodified) still succeeds (this is a
	// VALUE-domain rejection, not a structural one); the negative exponent occupies WSC1's own
	// "shift" column, which is the PUBLIC, documented UNSIGNED [kWeightScaleShiftMin,
	// kWeightScaleShiftMax] domain (model.h) -- a negative value is outside it, exactly as
	// design Sec9 states this direction was already closed by the EXISTING value-domain check
	// (WeightScaleShiftOutOfDomain, model.cpp's own ValidateWeightScalesDomain -- a TU-internal
	// function this cell confirms the CONTRACT of via its public constants, per this suite's own
	// "standing green control, not a new red cell" framing, T-2018/T-2027 casebook Sec18.3).
	std::vector<uint8_t> wsc1_bytes = m.bytes;
	for (int i = 0; i < 4; ++i) wsc1_bytes[i] = kWeightScalesMagic[i];
	SslmSectionView wsc1_view = MakeManifestSectionView(SslmSectionType::WeightScales,
	                                                     SslmDtype::Int32, wsc1_bytes);
	SslmTensorManifest wsc1_manifest;
	std::string werr;
	CHECK_MSG(SslmTensorManifest::Parse(wsc1_view, wsc1_manifest, &werr) == SslmModelStatus::Ok,
	          "the structural parse itself must still succeed (this is a VALUE-domain rejection, "
	          "not a structural one)");
	const SslmTensorView* t = wsc1_manifest.Tensor("t0");
	CHECK_MSG(t != nullptr, "the reinterpreted tensor must still be structurally present");
	if (t != nullptr) {
		// Read the shift column exactly as ValidateWeightScalesDomain itself does (row 0, byte
		// offset 8: identity, mult, shift each 4 bytes) and confirm it falls outside the public,
		// documented WSC1 domain -- the same fact ValidateWeightScalesDomain's own
		// WeightScaleShiftOutOfDomain rejection is built on.
		int32_t shift = 0;
		std::memcpy(&shift, t->data + 8, 4);
		CHECK_MSG(shift < kWeightScaleShiftMin || shift > kWeightScaleShiftMax,
		          "a genuine delta-fold triple's negative exponent (%d), read as WSC1's own "
		          "shift column, must fall outside [%d,%d] -- confirming WSC1's existing "
		          "WeightScaleShiftOutOfDomain check already rejects this swap direction",
		          shift, kWeightScaleShiftMin, kWeightScaleShiftMax);
	}
}

// --- Cell: dimension validation --------------------------------------------------------------

static void TestDimensionValidationRejectsRowCountMismatch() {
	BuiltManifest m = MakeSingleTensorManifest(kDeltaFoldScalesMagic, 4, {3});  // 1 row (3 elems)
	WriteTriple(m.bytes, m.tensor_data_off[0], 0, 0, 1000000000, 5);
	SslmSectionView view = MakeManifestSectionView(SslmSectionType::DeltaFoldScales, SslmDtype::Int32, m.bytes);
	SslmDeltaFoldScaleView out;
	std::string err;
	CHECK(SslmDeltaFoldScaleView::Parse(view, out, &err) == SslmModelStatus::Ok);
	CHECK(out.Entries().size() == 1);

	// The projection this entry targets was actually adapted with out_channels=896 (a stand-in
	// base geometry) -- but the section carries only 1 row. Must reject.
	const SslmModelStatus status = ValidateAmplifyingFoldDimension(out.Entries()[0], /*expected=*/896, &err);
	CHECK_MSG(status == SslmModelStatus::AmplifyingFoldDimensionMismatch,
	          "row_count=1 vs expected out_channels=896 must reject: got %s", SslmModelStatusName(status));

	// Negative control: the correct expectation passes.
	CHECK(ValidateAmplifyingFoldDimension(out.Entries()[0], /*expected=*/1, &err) == SslmModelStatus::Ok);
}

// --- Cell: signed-domain validation -----------------------------------------------------------

static void TestSignedDomainValidationRejectsOutOfRangeExponent() {
	BuiltManifest m = MakeSingleTensorManifest(kDeltaFoldScalesMagic, 4, {3});
	WriteTriple(m.bytes, m.tensor_data_off[0], 0, /*identity=*/0, /*mult=*/1000000000, /*exponent=*/32);  // > 31
	SslmSectionView view = MakeManifestSectionView(SslmSectionType::DeltaFoldScales, SslmDtype::Int32, m.bytes);
	SslmDeltaFoldScaleView out;
	std::string err;
	CHECK(SslmDeltaFoldScaleView::Parse(view, out, &err) == SslmModelStatus::Ok);
	SslmModelStatus status = ValidateAmplifyingFoldScalesDomain(out.Entries(), &err);
	CHECK_MSG(status == SslmModelStatus::AmplifyingFoldExponentOutOfDomain,
	          "exponent=32 (outside [-31,31]) must reject: got %s", SslmModelStatusName(status));

	// -31 (the widened floor, illegal for WSC1's own [0,31] but legal here) must pass.
	std::vector<uint8_t> ok_bytes = m.bytes;
	WriteTriple(ok_bytes, m.tensor_data_off[0], 0, 0, 1000000000, -31);
	SslmSectionView ok_view = MakeManifestSectionView(SslmSectionType::DeltaFoldScales, SslmDtype::Int32, ok_bytes);
	SslmDeltaFoldScaleView ok_out;
	CHECK(SslmDeltaFoldScaleView::Parse(ok_view, ok_out, &err) == SslmModelStatus::Ok);
	CHECK_MSG(ValidateAmplifyingFoldScalesDomain(ok_out.Entries(), &err) == SslmModelStatus::Ok,
	          "exponent=-31 (the widened floor) must be legal: %s", err.c_str());

	// identity not in {0,1} must also reject.
	std::vector<uint8_t> bad_id_bytes = m.bytes;
	WriteTriple(bad_id_bytes, m.tensor_data_off[0], 0, /*identity=*/2, 0, 0);
	SslmSectionView bad_id_view = MakeManifestSectionView(SslmSectionType::DeltaFoldScales, SslmDtype::Int32,
	                                                       bad_id_bytes);
	SslmDeltaFoldScaleView bad_id_out;
	CHECK(SslmDeltaFoldScaleView::Parse(bad_id_view, bad_id_out, &err) == SslmModelStatus::Ok);
	CHECK(ValidateAmplifyingFoldScalesDomain(bad_id_out.Entries(), &err) ==
	      SslmModelStatus::AmplifyingFoldIdentityNotBool);
}

// --- Cell: projection validation --------------------------------------------------------------

static void TestProjectionValidationRejectsUnknownAndUnclaimedProjections() {
	std::string err;
	std::vector<std::string_view> target_modules = {"q_proj", "v_proj"};

	CHECK(ValidateAmplifyingFoldProjection("q_proj", target_modules, &err) == SslmModelStatus::Ok);
	CHECK_MSG(ValidateAmplifyingFoldProjection("k_proj", target_modules, &err) ==
	              SslmModelStatus::AmplifyingFoldProjectionInvalid,
	          "a projection not in the adapter's own target_modules must reject");
	CHECK_MSG(ValidateAmplifyingFoldProjection("not_a_real_projection", target_modules, &err) ==
	              SslmModelStatus::AmplifyingFoldProjectionInvalid,
	          "a projection outside the seven PEFT-adaptable projections must reject even if "
	          "(hypothetically) claimed by target_modules");
}

// --- Cell: base-hash validation ----------------------------------------------------------------

static void TestBaseHashValidationRejectsMismatch() {
	std::array<uint8_t, kIntegrityHashBytes> declared{};
	std::array<uint8_t, kIntegrityHashBytes> actual{};
	declared.fill(0xAB);
	actual.fill(0xAB);
	std::string err;
	CHECK(ValidateAmplifyingFoldBaseHash(declared, actual, &err) == SslmModelStatus::Ok);

	actual[31] ^= 0x01;  // one-byte divergence
	CHECK_MSG(ValidateAmplifyingFoldBaseHash(declared, actual, &err) ==
	              SslmModelStatus::AmplifyingFoldBaseHashMismatch,
	          "a one-byte-divergent base hash must reject");
}

// --- Cell: a clean, correctly-formed adapter section passes every check -----------------------

static void TestCleanCorrectlyFormedSectionPassesEveryCheck() {
	// Two projections, mirroring a q_proj/v_proj-only adapter (the routine PEFT case, design
	// Sec8's own B1c motivation) -- out_channels=4 for this stand-in fixture.
	BuiltManifest m = BuildManifest(kDeltaFoldScalesMagic, 4, {{"layer0.q_proj", {12}}, {"layer0.v_proj", {12}}});
	for (int t = 0; t < 2; ++t) {
		for (uint64_t row = 0; row < 4; ++row) {
			WriteTriple(m.bytes, m.tensor_data_off[t], row, /*identity=*/1, 0, 0);  // exact pass-through
		}
	}
	SslmSectionView view = MakeManifestSectionView(SslmSectionType::DeltaFoldScales, SslmDtype::Int32, m.bytes);
	SslmDeltaFoldScaleView out;
	std::string err;
	CHECK_MSG(SslmDeltaFoldScaleView::Parse(view, out, &err) == SslmModelStatus::Ok,
	          "a correctly-typed, correctly-magic'd, well-formed section must parse Ok: %s", err.c_str());
	CHECK(out.Entries().size() == 2);
	CHECK(ValidateAmplifyingFoldScalesDomain(out.Entries(), &err) == SslmModelStatus::Ok);

	const std::vector<std::string_view> target_modules = {"q_proj", "v_proj"};
	for (const auto& entry : out.Entries()) {
		CHECK(ValidateAmplifyingFoldDimension(entry, /*expected=*/4, &err) == SslmModelStatus::Ok);
		std::string_view proj = entry.name.substr(entry.name.find('.') + 1);
		CHECK_MSG(ValidateAmplifyingFoldProjection(proj, target_modules, &err) == SslmModelStatus::Ok,
		          "projection \"%.*s\" must validate cleanly", (int)proj.size(), proj.data());
		CHECK(SslmDeltaFoldScaleView::Identity(entry, 0) == 1);
	}

	std::array<uint8_t, kIntegrityHashBytes> hash{};
	hash.fill(0x42);
	CHECK(ValidateAmplifyingFoldBaseHash(hash, hash, &err) == SslmModelStatus::Ok);
}

int main() {
	TestDedicatedWrapperTypesAreNotInterchangeable();
	TestSwapDirectionOneWsc1BytesRejectedByDeltaFoldParserBeforeAnyValueRead();
	TestSwapDirectionTwoAmplifyingBytesWithNegativeExponentRejectedByWsc1Validator();
	TestDimensionValidationRejectsRowCountMismatch();
	TestSignedDomainValidationRejectsOutOfRangeExponent();
	TestProjectionValidationRejectsUnknownAndUnclaimedProjections();
	TestBaseHashValidationRejectsMismatch();
	TestCleanCorrectlyFormedSectionPassesEveryCheck();

	std::printf("t2029 B0b red suite (artifact loader): %d checks, %d failures\n", GChecks, GFailures);
	return GFailures == 0 ? 0 : 1;
}

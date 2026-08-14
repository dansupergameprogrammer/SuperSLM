// SuperSLM model view — tensor-manifest sub-parse (WGT1/BIA1/ROP1).
//
// Parses one array section's self-contained tensor manifest after SslmArtifact has
// verified whole-file structure and integrity. A crafted integrity-valid artifact can
// still carry a malformed manifest inside a validated section, so this sub-parse is its
// own hostile-input trust boundary (docs/sslm_format.md "Model sub-formats"; the T-129
// bar): every field is validated against the section's own bounds, in 64-bit arithmetic
// with explicit overflow guards (a 32-bit length product is the T-129 defect class),
// before any tensor byte is exposed. Deviation is a rejection with a status, never a
// silent partial view. Standard library only (D-SLM13).
#include "superslm/model.h"
#include "superslm/intmath.h"       // S-HARDEN-1 (D-SLM142): pin the value gate's bounds to their source
#include "superslm/proof_manifest.h"  // S3.3 §13.1 cell 4 R1-R3 (T-1333): CheckConfigGeometry/ConfigGeometryStatus
#include "superslm/silu_lut.h"      // kSiluLutLog2K/kSiluLutQIdx/kSiluLutTermLeftShiftOverflowExponent
#include "superslm/silu_lut_canonical.h"  // kSiluLutCanonicalTable — S-HARDEN-1 (F20/F22)
#include "superslm/tokenizer.h"     // S-HARDEN-2 (F18/F6/F7/F15): the tokenizer join

#include "bad_alloc_wrap.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <string>
#include <unordered_set>

namespace superslm {

namespace {

// Little-endian field reads, byte-by-byte — never a struct cast over untrusted bytes
// (docs/sslm_format.md choice 1), matching the loader's discipline in artifact.cpp.
uint32_t RdU32(const uint8_t* p) noexcept {
	return uint32_t(p[0]) | (uint32_t(p[1]) << 8) | (uint32_t(p[2]) << 16) | (uint32_t(p[3]) << 24);
}
uint64_t RdU64(const uint8_t* p) noexcept {
	uint64_t v = 0;
	for (int i = 0; i < 8; ++i) v |= uint64_t(p[i]) << (8 * i);
	return v;
}
double RdF64(const uint8_t* p) noexcept {
	uint64_t bits = RdU64(p);
	double v;
	std::memcpy(&v, &bits, sizeof(v));
	return v;
}

SslmModelStatus Reject(SslmModelStatus code, std::string* err, const char* msg) {
	if (err) *err = msg;
	return code;
}

// Header field offsets (docs/sslm_format.md "Tensor-manifest blob").
constexpr uint32_t kOffVersion = 4;
constexpr uint32_t kOffTensorCount = 8;
constexpr uint32_t kOffNameBlobLen = 12;
// TensorDesc field offsets, relative to the descriptor start.
constexpr uint32_t kDescNameOff = 0;
constexpr uint32_t kDescNameLen = 4;
constexpr uint32_t kDescRank = 8;
constexpr uint32_t kDescShape = 12;   // shape[4]
constexpr uint32_t kDescDataOff = 28;
constexpr uint32_t kDescElemCount = 36;
constexpr uint32_t kDescReserved = 44;

} // namespace

const char* SslmModelStatusName(SslmModelStatus s) noexcept {
	switch (s) {
		case SslmModelStatus::Ok: return "Ok";
		case SslmModelStatus::SectionTooShort: return "SectionTooShort";
		case SslmModelStatus::BadManifestMagic: return "BadManifestMagic";
		case SslmModelStatus::UnsupportedManifestVersion: return "UnsupportedManifestVersion";
		case SslmModelStatus::TooManyTensors: return "TooManyTensors";
		case SslmModelStatus::ManifestOutOfBounds: return "ManifestOutOfBounds";
		case SslmModelStatus::BadTensorName: return "BadTensorName";
		case SslmModelStatus::EmptyTensorName: return "EmptyTensorName";
		case SslmModelStatus::DuplicateTensorName: return "DuplicateTensorName";
		case SslmModelStatus::BadTensorRank: return "BadTensorRank";
		case SslmModelStatus::BadTensorShape: return "BadTensorShape";
		case SslmModelStatus::ShapeCountMismatch: return "ShapeCountMismatch";
		case SslmModelStatus::TensorMisaligned: return "TensorMisaligned";
		case SslmModelStatus::TensorOutOfBounds: return "TensorOutOfBounds";
		case SslmModelStatus::TensorOverlap: return "TensorOverlap";
		case SslmModelStatus::BadDescriptorReserved: return "BadDescriptorReserved";
		case SslmModelStatus::BadConstantsMagic: return "BadConstantsMagic";
		case SslmModelStatus::UnsupportedConstantsVersion: return "UnsupportedConstantsVersion";
		case SslmModelStatus::TooManyConstantEntries: return "TooManyConstantEntries";
		case SslmModelStatus::BadValueWords: return "BadValueWords";
		case SslmModelStatus::ConstantsOutOfBounds: return "ConstantsOutOfBounds";
		case SslmModelStatus::BadEntryName: return "BadEntryName";
		case SslmModelStatus::EmptyEntryName: return "EmptyEntryName";
		case SslmModelStatus::DuplicateEntryName: return "DuplicateEntryName";
		case SslmModelStatus::BadConstantsReserved: return "BadConstantsReserved";
		case SslmModelStatus::BadConfigSize: return "BadConfigSize";
		case SslmModelStatus::BadConfigMagic: return "BadConfigMagic";
		case SslmModelStatus::UnsupportedConfigVersion: return "UnsupportedConfigVersion";
		case SslmModelStatus::BadConfigDim: return "BadConfigDim";
		case SslmModelStatus::BadKvPrecision: return "BadKvPrecision";
		case SslmModelStatus::BadConfigBool: return "BadConfigBool";
		case SslmModelStatus::BadConfigReserved: return "BadConfigReserved";
		case SslmModelStatus::BadSigmoidLutSize: return "BadSigmoidLutSize";
		case SslmModelStatus::BadSigmoidLutMagic: return "BadSigmoidLutMagic";
		case SslmModelStatus::UnsupportedSigmoidLutVersion: return "UnsupportedSigmoidLutVersion";
		case SslmModelStatus::BadSigmoidLutCount: return "BadSigmoidLutCount";
		case SslmModelStatus::BadSigmoidLutReserved: return "BadSigmoidLutReserved";
		case SslmModelStatus::BadSigmoidLutContent: return "BadSigmoidLutContent";
		case SslmModelStatus::ArtifactRejected: return "ArtifactRejected";
		case SslmModelStatus::CompositionScaleOutOfDomain: return "CompositionScaleOutOfDomain";
		case SslmModelStatus::WeightScaleShiftOutOfDomain: return "WeightScaleShiftOutOfDomain";
		case SslmModelStatus::WeightScaleIdentityNotBool: return "WeightScaleIdentityNotBool";
		case SslmModelStatus::WeightScaleTripleCountInvalid: return "WeightScaleTripleCountInvalid";
		case SslmModelStatus::RopeTableEntryOutOfDomain: return "RopeTableEntryOutOfDomain";
		case SslmModelStatus::KvLandingScaleOutOfDomain: return "KvLandingScaleOutOfDomain";
		case SslmModelStatus::KvLandingReciprocalOutOfDomain: return "KvLandingReciprocalOutOfDomain";
		case SslmModelStatus::TokenizerRejected: return "TokenizerRejected";
		case SslmModelStatus::TokenizerVocabSizeMismatch: return "TokenizerVocabSizeMismatch";
		case SslmModelStatus::BadConfigHeadDimParity: return "BadConfigHeadDimParity";
		case SslmModelStatus::ConfigGeometryKvHeadsExceedsHeads: return "ConfigGeometryKvHeadsExceedsHeads";
		case SslmModelStatus::ConfigGeometryHeadsNotDivisibleByKv: return "ConfigGeometryHeadsNotDivisibleByKv";
		case SslmModelStatus::ConfigGeometryHiddenSizeMismatch: return "ConfigGeometryHiddenSizeMismatch";
		case SslmModelStatus::RopeTablesShapeMismatchConfig: return "RopeTablesShapeMismatchConfig";
		case SslmModelStatus::CalibrationBandOutOfDomain: return "CalibrationBandOutOfDomain";
		case SslmModelStatus::AmplifyingFoldSectionTypeMismatch: return "AmplifyingFoldSectionTypeMismatch";
		case SslmModelStatus::AmplifyingFoldTripleCountInvalid: return "AmplifyingFoldTripleCountInvalid";
		case SslmModelStatus::AmplifyingFoldIdentityNotBool: return "AmplifyingFoldIdentityNotBool";
		case SslmModelStatus::AmplifyingFoldExponentOutOfDomain: return "AmplifyingFoldExponentOutOfDomain";
		case SslmModelStatus::AmplifyingFoldDimensionMismatch: return "AmplifyingFoldDimensionMismatch";
		case SslmModelStatus::AmplifyingFoldProjectionInvalid: return "AmplifyingFoldProjectionInvalid";
		case SslmModelStatus::AmplifyingFoldBaseHashMismatch: return "AmplifyingFoldBaseHashMismatch";
	}
	return "Unknown";
}

const uint8_t* ConstantsMagicFor(SslmSectionType type) noexcept {
	switch (type) {
		case SslmSectionType::CompositionConstants:
		case SslmSectionType::KvLandingScales:
		case SslmSectionType::KvLandingReciprocals:
		case SslmSectionType::CalibrationBand:
			return kConstantsMagic;
		default:
			return nullptr;
	}
}

uint32_t ExpectedValueWords(SslmSectionType type) noexcept {
	switch (type) {
		case SslmSectionType::CompositionConstants: return 2;  // (m, e)
		case SslmSectionType::KvLandingScales: return 2;       // (m, e)
		case SslmSectionType::KvLandingReciprocals: return 3;  // (m, e, R)
		case SslmSectionType::CalibrationBand: return 2;       // (min, max)
		default: return 0;
	}
}

const uint8_t* ManifestMagicFor(SslmSectionType type) noexcept {
	switch (type) {
		case SslmSectionType::Weights: return kWeightsMagic;
		case SslmSectionType::Biases: return kBiasesMagic;
		case SslmSectionType::RopeTables: return kRopeMagic;
		case SslmSectionType::WeightScales: return kWeightScalesMagic;
		default: return nullptr;
	}
}

// T-2021/T-2029 B0b (design Sec9, D-SLM3093): DeltaFoldScales/UFoldScales, deliberately NOT
// added to ManifestMagicFor/SslmTensorManifest above -- that function and class are WSC1's own
// (and Weights/Biases/RopeTables', which share its generic container), and design Sec9's own
// text requires the two new arrays never share WeightScales' type or magic. Their own
// magic/section-type mapping is a separate, single-source pair of switches.
SslmSectionType AmplifyingFoldSectionTypeFor(SslmAmplifyingFoldKind kind) noexcept {
	switch (kind) {
		case SslmAmplifyingFoldKind::Delta: return SslmSectionType::DeltaFoldScales;
		case SslmAmplifyingFoldKind::U: return SslmSectionType::UFoldScales;
	}
	return SslmSectionType::DeltaFoldScales; // unreachable -- Kind is a 2-value enum class
}

const uint8_t* AmplifyingFoldMagicFor(SslmAmplifyingFoldKind kind) noexcept {
	switch (kind) {
		case SslmAmplifyingFoldKind::Delta: return kDeltaFoldScalesMagic;
		case SslmAmplifyingFoldKind::U: return kUFoldScalesMagic;
	}
	return kDeltaFoldScalesMagic; // unreachable
}

// SslmAmplifyingFoldScaleView<Kind>::Parse -- mirrors SslmTensorManifest::ParseImpl's own
// header/descriptor walk (byte-for-byte offset convention, same kManifestHeaderBytes/
// kTensorDescBytes geometry, so the two array kinds stay a drop-in-compatible ON-DISK shape
// with WSC1 even though their C++ TYPES are never interconvertible -- design Sec9's own
// "storage-shape-identical... but domain-incompatible" framing), with the ONE load-bearing
// difference this design's own B0b build cell requires: the section's DECLARED type is checked
// against THIS Kind's own expected type FIRST, before the magic bytes or any descriptor is
// read -- so a genuine WeightScales section (or the other Kind's own section) is rejected by
// AmplifyingFoldSectionTypeMismatch at the very first line, never reaching a byte comparison
// that a crafted WSC1 payload might otherwise pass.
template <SslmAmplifyingFoldKind Kind>
SslmModelStatus SslmAmplifyingFoldScaleView<Kind>::Parse(const SslmSectionView& section,
                                                         SslmAmplifyingFoldScaleView& out,
                                                         std::string* err) {
	internal::MaybeThrowInjectedBadAllocFault();
	out.entries_.clear();
	if (err) err->clear();

	// --- Section-type gate: FIRST, before any byte is read (design Sec9's own requirement). ---
	if (section.type != AmplifyingFoldSectionTypeFor(Kind)) {
		return Reject(SslmModelStatus::AmplifyingFoldSectionTypeMismatch, err,
		              "section's declared type does not match this wrapper's own kind "
		              "(DeltaFoldScales vs UFoldScales vs WeightScales are never interchangeable)");
	}

	const uint8_t* base = section.data;
	const uint64_t size = section.byte_size;

	if (base == nullptr || size < kManifestHeaderBytes)
		return Reject(SslmModelStatus::SectionTooShort, err, "section smaller than the manifest header");

	const uint8_t* magic = AmplifyingFoldMagicFor(Kind);
	for (int i = 0; i < 4; ++i) {
		if (base[i] != magic[i])
			return Reject(SslmModelStatus::BadManifestMagic, err, "manifest magic mismatch");
	}

	if (RdU32(base + kOffVersion) != kManifestVersion)
		return Reject(SslmModelStatus::UnsupportedManifestVersion, err, "unsupported manifest version");

	const uint32_t tensor_count = RdU32(base + kOffTensorCount);
	if (tensor_count > kMaxTensors)
		return Reject(SslmModelStatus::TooManyTensors, err, "tensor_count exceeds kMaxTensors");

	const uint32_t name_blob_len = RdU32(base + kOffNameBlobLen);
	const uint64_t name_blob_off = uint64_t(kManifestHeaderBytes) + uint64_t(tensor_count) * kTensorDescBytes;
	const uint64_t data_region_floor = name_blob_off + name_blob_len;
	if (data_region_floor > size)
		return Reject(SslmModelStatus::ManifestOutOfBounds, err,
		              "header + descriptors + name blob exceed the section");

	// This section's own dtype is always Int32 (ExpectedDtype, artifact.cpp) -- 4 bytes/elem.
	constexpr uint32_t kElementSize = 4;

	std::vector<SslmAmplifyingFoldEntry> entries;
	entries.reserve(tensor_count);
	std::vector<std::pair<uint64_t, uint64_t>> ranges;
	ranges.reserve(tensor_count);
	std::unordered_set<std::string_view> seen_names;
	seen_names.reserve(tensor_count);

	for (uint32_t i = 0; i < tensor_count; ++i) {
		const uint8_t* d = base + kManifestHeaderBytes + uint64_t(i) * kTensorDescBytes;
		const uint32_t name_off = RdU32(d + kDescNameOff);
		const uint32_t name_len = RdU32(d + kDescNameLen);
		const uint32_t rank = RdU32(d + kDescRank);
		uint32_t shape[kMaxTensorRank];
		for (uint32_t k = 0; k < kMaxTensorRank; ++k) shape[k] = RdU32(d + kDescShape + k * 4);
		const uint64_t data_off = RdU64(d + kDescDataOff);
		const uint64_t elem_count = RdU64(d + kDescElemCount);
		const uint32_t reserved = RdU32(d + kDescReserved);

		if (name_len == 0)
			return Reject(SslmModelStatus::EmptyTensorName, err, "entry name is empty");
		if (uint64_t(name_off) + name_len > name_blob_len)
			return Reject(SslmModelStatus::BadTensorName, err, "entry name range outside the name blob");
		std::string_view name(reinterpret_cast<const char*>(base + name_blob_off + name_off), name_len);
		if (!seen_names.insert(name).second)
			return Reject(SslmModelStatus::DuplicateTensorName, err, "duplicate entry name");

		if (rank == 0 || rank > kMaxTensorRank)
			return Reject(SslmModelStatus::BadTensorRank, err, "entry rank out of range");
		uint64_t product = 1;
		for (uint32_t k = 0; k < kMaxTensorRank; ++k) {
			if (k < rank) {
				if (shape[k] == 0)
					return Reject(SslmModelStatus::BadTensorShape, err, "shape entry zero within rank");
				if (product > UINT64_MAX / shape[k])
					return Reject(SslmModelStatus::ShapeCountMismatch, err, "shape product overflows 64-bit");
				product *= shape[k];
			} else if (shape[k] != 0) {
				return Reject(SslmModelStatus::BadTensorShape, err, "shape entry nonzero past rank");
			}
		}
		if (elem_count != product)
			return Reject(SslmModelStatus::ShapeCountMismatch, err, "elem_count != product(shape)");
		if (reserved != 0)
			return Reject(SslmModelStatus::BadDescriptorReserved, err, "descriptor reserved field != 0");

		// design Sec9 item (a)/(b) precursor: elem_count must be a multiple of 3 (one
		// (identity,mult,exponent) triple per row) -- mirrors WeightScaleTripleCountInvalid,
		// but under THIS kind's own status code, per model.h's own documented rationale.
		if (elem_count % 3 != 0)
			return Reject(SslmModelStatus::AmplifyingFoldTripleCountInvalid, err,
			              "entry elem_count is not a multiple of 3");

		if (data_off % kElementSize != 0)
			return Reject(SslmModelStatus::TensorMisaligned, err, "data_off not a multiple of the element size");
		if (data_off < data_region_floor)
			return Reject(SslmModelStatus::TensorOutOfBounds, err, "data_off below the data region");
		if (elem_count > UINT64_MAX / kElementSize)
			return Reject(SslmModelStatus::TensorOutOfBounds, err, "elem_count * element_size overflows 64-bit");
		const uint64_t nbytes = elem_count * kElementSize;
		if (data_off > size || nbytes > size - data_off)
			return Reject(SslmModelStatus::TensorOutOfBounds, err, "entry data range exceeds the section");

		const uint64_t range_start = data_off, range_end = data_off + nbytes;
		for (const auto& r : ranges) {
			if (range_start < r.second && r.first < range_end)
				return Reject(SslmModelStatus::TensorOverlap, err, "entry data ranges overlap");
		}
		ranges.emplace_back(range_start, range_end);

		SslmAmplifyingFoldEntry entry;
		entry.name = name;
		entry.data = base + data_off;
		entry.row_count = elem_count / 3;
		entries.push_back(entry);
	}

	out.entries_ = std::move(entries);
	return SslmModelStatus::Ok;
}

template <SslmAmplifyingFoldKind Kind>
const SslmAmplifyingFoldEntry* SslmAmplifyingFoldScaleView<Kind>::Entry(std::string_view name) const noexcept {
	for (const auto& e : entries_) {
		if (e.name == name) return &e;
	}
	return nullptr;
}

template <SslmAmplifyingFoldKind Kind>
int32_t SslmAmplifyingFoldScaleView<Kind>::Identity(const SslmAmplifyingFoldEntry& entry, uint64_t row) noexcept {
	return static_cast<int32_t>(RdU32(entry.data + (row * 3 + 0) * 4));
}
template <SslmAmplifyingFoldKind Kind>
int32_t SslmAmplifyingFoldScaleView<Kind>::Mult(const SslmAmplifyingFoldEntry& entry, uint64_t row) noexcept {
	return static_cast<int32_t>(RdU32(entry.data + (row * 3 + 1) * 4));
}
template <SslmAmplifyingFoldKind Kind>
int32_t SslmAmplifyingFoldScaleView<Kind>::Exponent(const SslmAmplifyingFoldEntry& entry, uint64_t row) noexcept {
	return static_cast<int32_t>(RdU32(entry.data + (row * 3 + 2) * 4));
}

// Explicit instantiation -- the only two Kind values this design specifies (Sec9); keeps the
// template's body in this .cpp, matching this file's own "no *Impl body in the header" convention.
template class SslmAmplifyingFoldScaleView<SslmAmplifyingFoldKind::Delta>;
template class SslmAmplifyingFoldScaleView<SslmAmplifyingFoldKind::U>;

// design Sec9 item (b): signed domain, run through this array kind's own dedicated validator
// (not WSC1's ValidateWeightScalesDomain) -- identity in {0,1}, exponent in
// [kAmplifyingScaleExponentMin, kAmplifyingScaleExponentMax] = [-31,31] (widened relative to
// WSC1's own unsigned [0,31]).
SslmModelStatus ValidateAmplifyingFoldScalesDomain(const std::vector<SslmAmplifyingFoldEntry>& entries,
                                                    std::string* err) {
	for (const SslmAmplifyingFoldEntry& e : entries) {
		for (uint64_t row = 0; row < e.row_count; ++row) {
			const int32_t identity = static_cast<int32_t>(RdU32(e.data + (row * 3 + 0) * 4));
			const int32_t exponent = static_cast<int32_t>(RdU32(e.data + (row * 3 + 2) * 4));
			if (identity < 0 || identity > 1) {
				if (err) {
					*err = "AmplifyingFold entry \"" + std::string(e.name) + "\" row " + std::to_string(row) +
					       " identity=" + std::to_string(identity) + " not in {0,1}";
				}
				return SslmModelStatus::AmplifyingFoldIdentityNotBool;
			}
			if (exponent < kAmplifyingScaleExponentMin || exponent > kAmplifyingScaleExponentMax) {
				if (err) {
					*err = "AmplifyingFold entry \"" + std::string(e.name) + "\" row " + std::to_string(row) +
					       " exponent=" + std::to_string(exponent) + " outside [" +
					       std::to_string(kAmplifyingScaleExponentMin) + "," +
					       std::to_string(kAmplifyingScaleExponentMax) + "]";
				}
				return SslmModelStatus::AmplifyingFoldExponentOutOfDomain;
			}
		}
	}
	return SslmModelStatus::Ok;
}

// design Sec9 item (a): dimension.
SslmModelStatus ValidateAmplifyingFoldDimension(const SslmAmplifyingFoldEntry& entry,
                                                 uint64_t expected_row_count, std::string* err) {
	if (entry.row_count != expected_row_count) {
		if (err) {
			*err = "AmplifyingFold entry \"" + std::string(entry.name) + "\" row_count=" +
			       std::to_string(entry.row_count) + " != expected " + std::to_string(expected_row_count);
		}
		return SslmModelStatus::AmplifyingFoldDimensionMismatch;
	}
	return SslmModelStatus::Ok;
}

// design Sec9 item (c): projection. `declared_projection` must be one of the seven
// PEFT-adaptable projections AND must be claimed by the adapter's own target_modules.
SslmModelStatus ValidateAmplifyingFoldProjection(std::string_view declared_projection,
                                                  const std::vector<std::string_view>& adapter_target_modules,
                                                  std::string* err) {
	bool is_peft_adaptable = false;
	for (std::string_view p : kPeftAdaptableProjections) {
		if (p == declared_projection) { is_peft_adaptable = true; break; }
	}
	if (!is_peft_adaptable) {
		if (err) *err = "AmplifyingFold declared projection \"" + std::string(declared_projection) +
		                "\" is not one of the seven PEFT-adaptable projections";
		return SslmModelStatus::AmplifyingFoldProjectionInvalid;
	}
	bool claimed = false;
	for (std::string_view t : adapter_target_modules) {
		if (t == declared_projection) { claimed = true; break; }
	}
	if (!claimed) {
		if (err) *err = "AmplifyingFold declared projection \"" + std::string(declared_projection) +
		                "\" is not claimed by the adapter's own target_modules";
		return SslmModelStatus::AmplifyingFoldProjectionInvalid;
	}
	return SslmModelStatus::Ok;
}

// design Sec9 item (d): base-hash.
SslmModelStatus ValidateAmplifyingFoldBaseHash(const std::array<uint8_t, kIntegrityHashBytes>& declared,
                                                const std::array<uint8_t, kIntegrityHashBytes>& actual,
                                                std::string* err) {
	if (declared != actual) {
		if (err) *err = "AmplifyingFold entry's declared base-artifact integrity hash does not match "
		                "the actually-mapped base's own hash";
		return SslmModelStatus::AmplifyingFoldBaseHashMismatch;
	}
	return SslmModelStatus::Ok;
}


// S-HARDEN-7 (design Sec3.1): the *Impl bodies below need private access to
// their class (out.tensors_, out.entries_, out.backing_), which a free
// function cannot have. Each Access struct is the sole friend of its class
// (model.h) -- declared and defined only here, never in the header, so the
// membership-check AST walk never sees it. See artifact.cpp's identical
// SslmArtifactAccess comment for the full reasoning.
struct SslmTensorManifestAccess {
	static SslmModelStatus ParseImpl(const SslmSectionView& section, SslmTensorManifest& out,
	                                 std::string* err);
};
struct SslmKeyedConstantsAccess {
	static SslmModelStatus ParseImpl(const SslmSectionView& section, SslmKeyedConstants& out,
	                                 std::string* err);
};
struct SslmModelAccess {
	static SslmModelStatus LoadImpl(const uint8_t* data, size_t size, SslmModelView& out,
	                                std::string* err);
};

SslmModelStatus SslmTensorManifest::Parse(const SslmSectionView& section,
                                          SslmTensorManifest& out, std::string* err) {
	return internal::WrapBadAllocContract(
	    [&] { return SslmTensorManifestAccess::ParseImpl(section, out, err); });
}

SslmModelStatus SslmTensorManifestAccess::ParseImpl(const SslmSectionView& section,
                                                     SslmTensorManifest& out, std::string* err) {
	internal::MaybeThrowInjectedBadAllocFault();
	out.tensors_.clear();
	if (err) err->clear();

	const uint8_t* base = section.data;
	const uint64_t size = section.byte_size;

	// --- Header ---
	if (base == nullptr || size < kManifestHeaderBytes)
		return Reject(SslmModelStatus::SectionTooShort, err, "section smaller than the manifest header");

	const uint8_t* magic = ManifestMagicFor(section.type);
	if (magic == nullptr)
		return Reject(SslmModelStatus::BadManifestMagic, err, "section type carries no tensor manifest");
	for (int i = 0; i < 4; ++i) {
		if (base[i] != magic[i])
			return Reject(SslmModelStatus::BadManifestMagic, err, "manifest magic mismatch");
	}

	if (RdU32(base + kOffVersion) != kManifestVersion)
		return Reject(SslmModelStatus::UnsupportedManifestVersion, err, "unsupported manifest version");

	const uint32_t tensor_count = RdU32(base + kOffTensorCount);
	if (tensor_count > kMaxTensors)
		return Reject(SslmModelStatus::TooManyTensors, err, "tensor_count exceeds kMaxTensors");

	const uint32_t name_blob_len = RdU32(base + kOffNameBlobLen);

	// header + descriptor table + name blob must fit the section (64-bit; the fields
	// above are u32 so these sums cannot overflow u64).
	const uint64_t name_blob_off = uint64_t(kManifestHeaderBytes) + uint64_t(tensor_count) * kTensorDescBytes;
	const uint64_t data_region_floor = name_blob_off + name_blob_len;
	if (data_region_floor > size)
		return Reject(SslmModelStatus::ManifestOutOfBounds, err,
		              "header + descriptors + name blob exceed the section");

	const uint32_t element_size = DtypeSize(static_cast<uint32_t>(section.dtype));
	if (element_size == 0)
		return Reject(SslmModelStatus::BadManifestMagic, err, "section carries an unknown element dtype");

	// --- Descriptors (all descriptor bytes are now known in-bounds) ---
	std::vector<SslmTensorView> tensors;
	tensors.reserve(tensor_count);
	// Accepted tensors' byte ranges [start, end), for the post-loop overlap check.
	std::vector<std::pair<uint64_t, uint64_t>> ranges;
	ranges.reserve(tensor_count);
	// Names seen so far, for O(1) duplicate detection (S2a-1: was an O(n^2) scan).
	std::unordered_set<std::string_view> seen_names;
	seen_names.reserve(tensor_count);

	for (uint32_t i = 0; i < tensor_count; ++i) {
		const uint8_t* d = base + kManifestHeaderBytes + uint64_t(i) * kTensorDescBytes;
		const uint32_t name_off = RdU32(d + kDescNameOff);
		const uint32_t name_len = RdU32(d + kDescNameLen);
		const uint32_t rank = RdU32(d + kDescRank);
		uint32_t shape[kMaxTensorRank];
		for (uint32_t k = 0; k < kMaxTensorRank; ++k) shape[k] = RdU32(d + kDescShape + k * 4);
		const uint64_t data_off = RdU64(d + kDescDataOff);
		const uint64_t elem_count = RdU64(d + kDescElemCount);
		const uint32_t reserved = RdU32(d + kDescReserved);

		// Name: nonempty and within the name blob.
		if (name_len == 0)
			return Reject(SslmModelStatus::EmptyTensorName, err, "tensor name is empty");
		if (uint64_t(name_off) + name_len > name_blob_len)
			return Reject(SslmModelStatus::BadTensorName, err, "tensor name range outside the name blob");
		std::string_view name(reinterpret_cast<const char*>(base + name_blob_off + name_off), name_len);
		if (!seen_names.insert(name).second)
			return Reject(SslmModelStatus::DuplicateTensorName, err, "duplicate tensor name");

		// Rank.
		if (rank == 0 || rank > kMaxTensorRank)
			return Reject(SslmModelStatus::BadTensorRank, err, "tensor rank out of range");

		// Shape: nonzero within rank, zero past it; product held in 64-bit with an
		// explicit overflow guard (a u64 product of four u32 dims can overflow).
		uint64_t product = 1;
		for (uint32_t k = 0; k < kMaxTensorRank; ++k) {
			if (k < rank) {
				if (shape[k] == 0)
					return Reject(SslmModelStatus::BadTensorShape, err, "shape entry zero within rank");
				if (product > UINT64_MAX / shape[k])
					return Reject(SslmModelStatus::ShapeCountMismatch, err, "shape product overflows 64-bit");
				product *= shape[k];
			} else if (shape[k] != 0) {
				return Reject(SslmModelStatus::BadTensorShape, err, "shape entry nonzero past rank");
			}
		}
		if (elem_count != product)
			return Reject(SslmModelStatus::ShapeCountMismatch, err, "elem_count != product(shape)");

		if (reserved != 0)
			return Reject(SslmModelStatus::BadDescriptorReserved, err, "descriptor reserved field != 0");

		// Data placement: element-aligned, at or past the data region, in bounds, with
		// elem_count * element_size and data_off + bytes both overflow-guarded.
		if (data_off % element_size != 0)
			return Reject(SslmModelStatus::TensorMisaligned, err, "data_off not a multiple of the element size");
		if (data_off < data_region_floor)
			return Reject(SslmModelStatus::TensorOutOfBounds, err, "data_off below the data region");
		if (elem_count > UINT64_MAX / element_size)
			return Reject(SslmModelStatus::TensorOutOfBounds, err, "elem_count * element_size overflows 64-bit");
		const uint64_t nbytes = elem_count * element_size;
		if (data_off > size || nbytes > size - data_off)
			return Reject(SslmModelStatus::TensorOutOfBounds, err, "tensor data range exceeds the section");

		// Record this tensor's byte range for the post-loop overlap check (half-open;
		// nbytes >= 1 always, since rank >= 1 forces product >= 1 and element_size >= 1).
		ranges.emplace_back(data_off, data_off + nbytes);

		SslmTensorView tv;
		tv.name = name;
		tv.dtype = section.dtype;
		tv.rank = rank;
		for (uint32_t k = 0; k < kMaxTensorRank; ++k) tv.shape[k] = shape[k];
		tv.data = base + data_off;
		tv.elem_count = elem_count;
		tensors.push_back(tv);
	}

	// No two tensors' data ranges may overlap. Sorting by start reduces this from an
	// O(n^2) pairwise scan to O(n log n): once sorted, a range can only overlap its
	// immediate predecessor, so a single adjacent pass detects any overlap (S2a-1).
	std::sort(ranges.begin(), ranges.end());
	for (size_t i = 1; i < ranges.size(); ++i) {
		if (ranges[i].first < ranges[i - 1].second)
			return Reject(SslmModelStatus::TensorOverlap, err, "tensor data range overlaps another tensor");
	}

	out.tensors_ = std::move(tensors);
	return SslmModelStatus::Ok;
}

const SslmTensorView* SslmTensorManifest::Tensor(std::string_view name) const noexcept {
	for (const auto& t : tensors_) {
		if (t.name == name) return &t;
	}
	return nullptr;
}

SslmModelStatus SslmKeyedConstants::Parse(const SslmSectionView& section,
                                          SslmKeyedConstants& out, std::string* err) {
	return internal::WrapBadAllocContract(
	    [&] { return SslmKeyedConstantsAccess::ParseImpl(section, out, err); });
}

SslmModelStatus SslmKeyedConstantsAccess::ParseImpl(const SslmSectionView& section,
                                                     SslmKeyedConstants& out, std::string* err) {
	internal::MaybeThrowInjectedBadAllocFault();
	out.entries_.clear();
	if (err) err->clear();

	const uint8_t* base = section.data;
	const uint64_t size = section.byte_size;

	// --- Header ---
	if (base == nullptr || size < kConstantHeaderBytes)
		return Reject(SslmModelStatus::SectionTooShort, err, "section smaller than the KVC1 header");

	const uint8_t* magic = ConstantsMagicFor(section.type);
	if (magic == nullptr)
		return Reject(SslmModelStatus::BadConstantsMagic, err, "section type carries no keyed-constant table");
	for (int i = 0; i < 4; ++i) {
		if (base[i] != magic[i])
			return Reject(SslmModelStatus::BadConstantsMagic, err, "KVC1 magic mismatch");
	}

	if (RdU32(base + 4) != kManifestVersion)
		return Reject(SslmModelStatus::UnsupportedConstantsVersion, err, "unsupported KVC1 version");

	const uint32_t entry_count = RdU32(base + 8);
	if (entry_count > kMaxConstantEntries)
		return Reject(SslmModelStatus::TooManyConstantEntries, err, "entry_count exceeds kMaxConstantEntries");

	// value_words is validated before it is used in the size computation below.
	const uint32_t value_words = RdU32(base + 12);
	if (value_words != ExpectedValueWords(section.type))
		return Reject(SslmModelStatus::BadValueWords, err, "value_words is not the section type's required count");

	if (RdU32(base + 20) != 0)
		return Reject(SslmModelStatus::BadConstantsReserved, err, "KVC1 header reserved field != 0");

	const uint32_t name_blob_len = RdU32(base + 16);

	// Total size in 64-bit: name_blob_len is an uncapped u32, so header + descriptors +
	// values + name blob can exceed 2^32 (a 32-bit sum would wrap and wrongly accept).
	// entry_count <= kMaxConstantEntries (2^20) and value_words <= 3, so the descriptor
	// and value products cannot themselves overflow u64.
	const uint64_t values_off = uint64_t(kConstantHeaderBytes) + uint64_t(entry_count) * kConstantDescBytes;
	const uint64_t name_blob_off = values_off + uint64_t(entry_count) * value_words * 8;
	const uint64_t total = name_blob_off + name_blob_len;
	if (total > size)
		return Reject(SslmModelStatus::ConstantsOutOfBounds, err,
		              "header + descriptors + values + name blob exceed the section");

	// --- Entries (all descriptor/value/name bytes are now known in-bounds) ---
	std::vector<SslmConstantEntry> entries;
	entries.reserve(entry_count);
	std::unordered_set<std::string_view> seen_names;
	seen_names.reserve(entry_count);

	for (uint32_t i = 0; i < entry_count; ++i) {
		const uint8_t* d = base + kConstantHeaderBytes + uint64_t(i) * kConstantDescBytes;
		const uint32_t name_off = RdU32(d + 0);
		const uint32_t name_len = RdU32(d + 4);

		if (name_len == 0)
			return Reject(SslmModelStatus::EmptyEntryName, err, "entry name is empty");
		if (uint64_t(name_off) + name_len > name_blob_len)
			return Reject(SslmModelStatus::BadEntryName, err, "entry name range outside the name blob");
		std::string_view name(reinterpret_cast<const char*>(base + name_blob_off + name_off), name_len);
		if (!seen_names.insert(name).second)
			return Reject(SslmModelStatus::DuplicateEntryName, err, "duplicate entry name");

		SslmConstantEntry e;
		e.name = name;
		e.values = base + values_off + uint64_t(i) * value_words * 8;
		e.value_words = value_words;
		entries.push_back(e);
	}

	out.entries_ = std::move(entries);
	return SslmModelStatus::Ok;
}

const SslmConstantEntry* SslmKeyedConstants::Entry(std::string_view name) const noexcept {
	for (const auto& e : entries_) {
		if (e.name == name) return &e;
	}
	return nullptr;
}

int64_t SslmKeyedConstants::Value(const SslmConstantEntry& entry, uint32_t w) noexcept {
	const uint8_t* p = entry.values + static_cast<size_t>(w) * 8;
	uint64_t v = 0;
	for (int i = 0; i < 8; ++i) v |= uint64_t(p[i]) << (8 * i);
	return static_cast<int64_t>(v);
}

namespace {
SslmModelStatus ParseConfigImpl(const SslmSectionView& section, SslmModelConfig& out,
                                std::string* err) {
	internal::MaybeThrowInjectedBadAllocFault();
	out = SslmModelConfig{};
	if (err) err->clear();

	const uint8_t* base = section.data;

	// A fixed 84-byte struct: the exact-size check is the entire bounds surface and gates
	// every field read below.
	if (base == nullptr || section.byte_size != kConfigBytes)
		return Reject(SslmModelStatus::BadConfigSize, err, "Config section is not exactly kConfigBytes");

	for (int i = 0; i < 4; ++i) {
		if (base[i] != kConfigMagic[i])
			return Reject(SslmModelStatus::BadConfigMagic, err, "Config magic is not 'CFG1'");
	}
	if (RdU32(base + 4) != kManifestVersion)
		return Reject(SslmModelStatus::UnsupportedConfigVersion, err, "unsupported CFG1 version");

	SslmModelConfig c;
	c.hidden_size = RdU32(base + 8);
	c.num_hidden_layers = RdU32(base + 12);
	c.num_attention_heads = RdU32(base + 16);
	c.num_key_value_heads = RdU32(base + 20);
	c.head_dim = RdU32(base + 24);
	c.intermediate_size = RdU32(base + 28);
	c.vocab_size = RdU32(base + 32);
	c.context_cap = RdU32(base + 36);
	const uint32_t tie = RdU32(base + 40);
	const uint32_t kvp = RdU32(base + 44);
	c.kv_block_size = RdU32(base + 48);
	c.unicode_major = RdU32(base + 52);
	c.unicode_minor = RdU32(base + 56);
	c.unicode_patch = RdU32(base + 60);
	const uint32_t reserved = RdU32(base + 64);
	c.rope_theta = RdF64(base + 68);
	c.rms_norm_eps = RdF64(base + 76);

	if (reserved != 0)
		return Reject(SslmModelStatus::BadConfigReserved, err, "CFG1 reserved field != 0");

	// §11 reject-over-degrade: a zero dimension produces a model that loads, runs, and is
	// not the source model. Every dimension must be present.
	if (c.hidden_size == 0 || c.num_hidden_layers == 0 || c.num_attention_heads == 0 ||
	    c.num_key_value_heads == 0 || c.head_dim == 0 || c.intermediate_size == 0 ||
	    c.vocab_size == 0 || c.context_cap == 0 || c.kv_block_size == 0)
		return Reject(SslmModelStatus::BadConfigDim, err, "a required dimension field is 0");

	// Poirot fa3189a-s3.3-rope-site-and-c32-softmax-review-2026-07-28.md,
	// Significant 5: RoPE pairs elements two at a time (Sec6.2 step 3), and
	// forward_sites.h's own contract states "head_dim odd is a load-time
	// rejection" -- discharged here rather than left as an unchecked claim.
	if ((c.head_dim % 2) != 0)
		return Reject(SslmModelStatus::BadConfigHeadDimParity, err, "head_dim is odd -- RoPE pairs elements");

	if (tie > 1)
		return Reject(SslmModelStatus::BadConfigBool, err, "tie_word_embeddings not in {0,1}");
	if (kvp > 1)
		return Reject(SslmModelStatus::BadKvPrecision, err, "kv_precision not in {0,1}");

	c.tie_word_embeddings = (tie == 1);
	c.kv_precision = (kvp == 1) ? SslmKvPrecision::Int16 : SslmKvPrecision::Int8;

	out = c;
	return SslmModelStatus::Ok;
}
}  // namespace

// S-HARDEN-7: today's body, renamed to ParseConfigImpl above; wraps it with
// the shared catch-and-rethrow helper (src/bad_alloc_wrap.h).
SslmModelStatus ParseConfig(const SslmSectionView& section, SslmModelConfig& out,
                            std::string* err) {
	return internal::WrapBadAllocContract([&] { return ParseConfigImpl(section, out, err); });
}

// --- SIL1 sigmoid-LUT sub-parse -----------------------------------------------
// A fixed-layout section (like CFG1): the exact-size check is the entire bounds surface and
// gates every field read below. Every deviation is a rejection with a status, never a repaired
// or partial view (§11 reject-over-degrade; docs/sslm_format.md "Sigmoid-LUT blob — SIL1").
// Throws only std::bad_alloc (S-HARDEN-7, F5).
namespace {
SslmModelStatus ParseSigmoidLutImpl(const SslmSectionView& section, SslmSigmoidLut& out,
                                    std::string* err) {
	internal::MaybeThrowInjectedBadAllocFault();
	out = SslmSigmoidLut{};
	if (err) err->clear();

	const uint8_t* base = section.data;

	if (base == nullptr || section.byte_size != kSigmoidLutBytes)
		return Reject(SslmModelStatus::BadSigmoidLutSize, err, "SIL1 section is not exactly kSigmoidLutBytes");

	for (int i = 0; i < 4; ++i) {
		if (base[i] != kSigmoidLutMagic[i])
			return Reject(SslmModelStatus::BadSigmoidLutMagic, err, "SIL1 magic is not 'SIL1'");
	}
	if (RdU32(base + 4) != kManifestVersion)
		return Reject(SslmModelStatus::UnsupportedSigmoidLutVersion, err, "unsupported SIL1 version");
	if (RdU32(base + 8) != kSigmoidLutEntries)
		return Reject(SslmModelStatus::BadSigmoidLutCount, err, "SIL1 entry_count != kSigmoidLutEntries");
	if (RdU32(base + 12) != 0)
		return Reject(SslmModelStatus::BadSigmoidLutReserved, err, "SIL1 reserved field != 0");

	// S-HARDEN-1 (F20/F22): SIL1 is a universal construction the spec fixes
	// entirely (not model-specific learned data), so every node is validated
	// against the pinned canonical table — a byte-for-byte comparison, not a
	// per-node range/monotonicity predicate. A structurally valid section whose
	// content differs from canonical is rejected here, BEFORE any node value is
	// exposed through `out` — this is what stops a hostile-but-structurally-
	// valid table (adjacent nodes at INT32_MIN/INT32_MAX, the exact operand a
	// strike drove through this parser once already) from ever reaching
	// SiluSigmoidQ15's interpolation. Read via the same little-endian
	// byte-assembly discipline as every other field in this file — never a
	// reinterpret_cast over the section's bytes.
	const uint8_t* nodes = base + kSigmoidLutHeaderBytes;
	for (uint32_t i = 0; i < kSigmoidLutEntries; ++i) {
		const int32_t got = static_cast<int32_t>(RdU32(nodes + static_cast<size_t>(i) * 4));
		if (got != kSiluLutCanonicalTable[i]) {
			if (err) {
				*err = "SIL1 node " + std::to_string(i) + " (" + std::to_string(got) +
				       ") does not match the pinned canonical table (" +
				       std::to_string(kSiluLutCanonicalTable[i]) + ")";
			}
			return SslmModelStatus::BadSigmoidLutContent;
		}
	}

	out.values = nodes;  // the 1025 int32 Q15 nodes, read via SigmoidLutValue
	out.entry_count = kSigmoidLutEntries;
	return SslmModelStatus::Ok;
}
}  // namespace

// S-HARDEN-7: today's body, renamed to ParseSigmoidLutImpl above; wraps it
// with the shared catch-and-rethrow helper (src/bad_alloc_wrap.h).
SslmModelStatus ParseSigmoidLut(const SslmSectionView& section, SslmSigmoidLut& out,
                                std::string* err) {
	return internal::WrapBadAllocContract([&] { return ParseSigmoidLutImpl(section, out, err); });
}

int32_t SigmoidLutValue(const SslmSigmoidLut& lut, uint32_t i) noexcept {
	// Real accessor (not the construction): little-endian int32 byte assembly, matching the
	// loader's read discipline (never a cast over unaligned/untrusted bytes).
	const uint8_t* p = lut.values + static_cast<size_t>(i) * 4;
	uint32_t v = uint32_t(p[0]) | (uint32_t(p[1]) << 8) | (uint32_t(p[2]) << 16) | (uint32_t(p[3]) << 24);
	return static_cast<int32_t>(v);
}

// --- S-HARDEN-1: the load-time orchestration entry point and the
//     schema-value gate (D-SLM141/D-SLM142) ---------------------------------
//
// SslmModel::Load is the one call site that composes SslmArtifact::
// OpenFromMemory, every present section's sub-parser, and ValidateSectionValues
// into a single pass. No such orchestrator existed before this slot — every
// sub-parser above was invoked one at a time, by hand, by tests and tools.
namespace {

int32_t RdI32(const uint8_t* p) noexcept { return static_cast<int32_t>(RdU32(p)); }
int64_t RdI64(const uint8_t* p) noexcept { return static_cast<int64_t>(RdU64(p)); }

// S-HARDEN-1's domain-descriptor constants (D-SLM141/D-SLM142), carried as
// data and cited to their source rather than re-derived per call site — one
// pass over these closes F22/F23/F24 and every future site the same section
// value would otherwise reach unchecked.
//
// CompositionConstants (KVC1) SiLU (m, e): the mandatory no-UB floor
// (kCompositionScaleMaxAbsM/MinE/MaxE, silu_lut.h — moved there from this
// anonymous namespace, ac34677 S11, so the runtime no-UB domain predicate in
// checked_chain_funnel.cpp can pin its own ceiling against the same public
// constants rather than a duplicated literal). Narrowing the floor further is a
// follow-on data edit gated on a recorded S2.4 §10 sweep (Charpy Finding 2,
// D-SLM142).

// WeightScales (WSC1): shift is UB-derived and exact (intmath.h:53,
// RoundingDivideByPOT's documented exponent domain; shift==32 and any
// negative shift are shift-count UB in RoundingDivideByPOTImpl). identity is
// a documented format invariant (sslm_format.md:315) with no in-tree consumer
// yet — enforced early at the gate, not because it is UB-linked.
// T-2021/T-2029 B0b: kWeightScaleShiftMin/Max moved to model.h as PUBLIC constants (design
// Sec9's B0b build cell needs to state this existing, unchanged bound from outside this TU) --
// this file now reads the single public copy rather than shadowing it with a second, TU-local
// one of the same name and value.

// RopeTables (ROP1): RopeApplyPair's safety argument (intmath.cpp:451) holds
// iff |cos|,|sin| <= ROPE_ONE (2^30, intmath.h:360) — the exact bound that
// restores its "each product <= 2^61, each sum <= 2^62" premise.
constexpr int64_t kRopeEntryAbsMax = INT64_C(1) << 30;

// S-HARDEN-1 (T-400, Poirot Finding 2): pin every domain constant above to the source it is
// cited to, so an edit to either side — this gate's own literal, or the kernel arithmetic it
// protects — fails the build instead of drifting past a reviewer's read. A comment citing a
// source line is a claim nobody checks; these are the same claims, checked at compile time.
static_assert(kCompositionScaleMaxAbsM == kInt32Max,
              "KVC1 m's no-UB floor must equal intmath.h's kInt32Max: silu_lut.cpp's |term| bound "
              "(|term| < 127*2^31 < 2^38) assumes |m| <= INT32_MAX — if this drifts, F22's no-UB "
              "floor is no longer the exact bound the comment claims");
static_assert(kCompositionScaleMaxE + kSiluLutLog2K + kSiluLutQIdx < kSiluLutTermLeftShiftOverflowExponent,
              "KVC1 e's upper no-UB floor must keep SiluSigmoidQ15's left-branch shift "
              "(e + kSiluLutLog2K + kSiluLutQIdx) below silu_lut.cpp's overflow point — if this "
              "drifts, F22's left branch (term << shift) can overflow int64");
static_assert(-kCompositionScaleMinE - kSiluLutLog2K - kSiluLutQIdx <= kRoundingDivideByPotExponentMaxI64,
              "KVC1 e's lower no-UB floor must keep SiluSigmoidQ15's right-branch -shift inside "
              "RoundingDivideByPOT's documented int64 exponent domain — if this drifts, F22's right "
              "branch calls RoundingDivideByPOT out of its domain");
static_assert(kWeightScaleShiftMin == kRoundingDivideByPotExponentMinI32 &&
                  kWeightScaleShiftMax == kRoundingDivideByPotExponentMaxI32,
              "WSC1 shift's domain must equal RoundingDivideByPOT's documented int32 exponent "
              "domain (intmath.h) exactly — if this drifts, F23's floor no longer matches the "
              "primitive it protects");
static_assert(kRopeEntryAbsMax == ROPE_ONE,
              "ROP1 element's domain must equal intmath.h's ROPE_ONE — RopeApplyPair's safety "
              "argument (intmath.cpp:451, '|cos|,|sin| <= ROPE_ONE') is exactly this bound; if it "
              "drifts, F24's floor no longer restores that premise");

// F22 — CompositionConstants (m, e): one check per keyed entry closes all
// three of SiluSigmoidQ15's UB sites (silu_lut.cpp:20 the code*m product,
// :35 the shift placement, :39 the accumulate) at once, because the mantissa
// and exponent both feed all three from the same two values.
SslmModelStatus ValidateCompositionConstantsDomain(const SslmKeyedConstants& kvc, std::string* err) {
	for (const SslmConstantEntry& e : kvc.Entries()) {
		const int64_t m = SslmKeyedConstants::Value(e, 0);
		const int64_t x_e = SslmKeyedConstants::Value(e, 1);
		if (m < -kCompositionScaleMaxAbsM || m > kCompositionScaleMaxAbsM) {
			if (err) {
				*err = "CompositionConstants entry \"" + std::string(e.name) + "\" m=" + std::to_string(m) +
				       " outside the no-UB floor |m| <= " + std::to_string(kCompositionScaleMaxAbsM);
			}
			return SslmModelStatus::CompositionScaleOutOfDomain;
		}
		if (x_e < kCompositionScaleMinE || x_e > kCompositionScaleMaxE) {
			if (err) {
				*err = "CompositionConstants entry \"" + std::string(e.name) + "\" e=" + std::to_string(x_e) +
				       " outside the no-UB floor [" + std::to_string(kCompositionScaleMinE) + "," +
				       std::to_string(kCompositionScaleMaxE) + "]";
			}
			return SslmModelStatus::CompositionScaleOutOfDomain;
		}
	}
	return SslmModelStatus::Ok;
}

// F23 — WeightScales: docs/sslm_format.md "Weight-scale fold blob" describes
// each row as an (identity, mult, shift) int32 triple, column-major within
// the row; walked here as flat triples over each tensor's elements (row i's
// triple at [3i, 3i+3)) rather than assuming a specific rank, so both a
// production [num_channels,3] tensor and a single-row rank-1 [3] tensor use
// the same walk. `mult` (column 1) is intentionally unchecked — any int32 is
// safe, per D-SLM142 (SaturatingRoundingDoublingHighMul saturates the sole
// overflow pair, intmath.cpp:146-152).
SslmModelStatus ValidateWeightScalesDomain(const SslmTensorManifest& wsc, std::string* err) {
	for (const SslmTensorView& t : wsc.Tensors()) {
		// T-1415 (whole-tree review b9dcbe0, Minor 3): the section stores rows of
		// (identity, mult, shift) int32 triples (docs/sslm_format.md "Weight-scale
		// fold blob"); a tensor whose elem_count is not a multiple of 3 has a
		// trailing partial triple the prior elem_count/3 walk below never
		// reached, so its tail elements loaded Ok with no validation and nothing
		// else in this artifact rejects the malformed geometry (the structural
		// parse checks shape-product consistency, not triple-ness). Rejected
		// outright rather than walked partially.
		if (t.elem_count % 3 != 0) {
			if (err) {
				*err = "WeightScales tensor \"" + std::string(t.name) + "\" elem_count=" +
				       std::to_string(t.elem_count) + " is not a multiple of 3";
			}
			return SslmModelStatus::WeightScaleTripleCountInvalid;
		}
		const uint64_t rows = t.elem_count / 3;
		for (uint64_t r = 0; r < rows; ++r) {
			const int32_t identity = RdI32(t.data + (r * 3 + 0) * 4);
			const int32_t shift = RdI32(t.data + (r * 3 + 2) * 4);
			if (identity < 0 || identity > 1) {
				if (err) {
					*err = "WeightScales tensor \"" + std::string(t.name) + "\" row " + std::to_string(r) +
					       " identity=" + std::to_string(identity) + " not in {0,1}";
				}
				return SslmModelStatus::WeightScaleIdentityNotBool;
			}
			if (shift < kWeightScaleShiftMin || shift > kWeightScaleShiftMax) {
				if (err) {
					*err = "WeightScales tensor \"" + std::string(t.name) + "\" row " + std::to_string(r) +
					       " shift=" + std::to_string(shift) + " outside [" +
					       std::to_string(kWeightScaleShiftMin) + "," + std::to_string(kWeightScaleShiftMax) + "]";
				}
				return SslmModelStatus::WeightScaleShiftOutOfDomain;
			}
		}
	}
	return SslmModelStatus::Ok;
}

// F24 — RopeTables: every stored element (cos and sin tables alike) must
// clear RopeApplyPair's safety bound before any element is used as an
// operand. The section stores int64 while the kernel takes int32, so the
// bound is checked on the stored int64 before any narrowing.
SslmModelStatus ValidateRopeTablesDomain(const SslmTensorManifest& rop, std::string* err) {
	for (const SslmTensorView& t : rop.Tensors()) {
		for (uint64_t i = 0; i < t.elem_count; ++i) {
			const int64_t v = RdI64(t.data + i * 8);
			if (v < -kRopeEntryAbsMax || v > kRopeEntryAbsMax) {
				if (err) {
					*err = "RopeTables tensor \"" + std::string(t.name) + "\" element " + std::to_string(i) +
					       "=" + std::to_string(v) + " outside [-" + std::to_string(kRopeEntryAbsMax) + "," +
					       std::to_string(kRopeEntryAbsMax) + "]";
				}
				return SslmModelStatus::RopeTableEntryOutOfDomain;
			}
		}
	}
	return SslmModelStatus::Ok;
}

// KvLandingScales'/KvLandingReciprocals' load-time value-domain descriptors
// (SuperSLM_S3a_WalkingSkeleton_Plan.md §7.2a third limb, §8.1; S3.3;
// Claude/Curie/superslm-s3.3-attention-interior-test-design-2026-07-28.md
// §4.5/§6.5). `R_t` IS, by the plan's own text, "C19's reciprocal" of the
// target mantissa (DynamicScaleReciprocal, intmath.h -- C19, not C27; C27
// names LandingRescale's own composite, which CONSUMES this reciprocal
// rather than being it) -- so any artifact-carried value outside what
// `DynamicScaleReciprocal` can ever produce is definitionally not a
// reciprocal of anything. Curie's record derives the bound by execution,
// calling the real primitive at its own domain's two endpoints:
//   DynamicScaleReciprocal(2^30)     == 4294967296  (== 2^32, the max)
//   DynamicScaleReciprocal(2^31 - 1) == 2147483649  (== 2^31 + 1, the min)
// `kKvLandingScaleMantissaMin/Max` is the canonical carried-mantissa range
// every other KVC1 scale mantissa in this tree already uses (the funnel's own
// CombineCarriedScale precondition; the format's canonical-scale convention).
constexpr int64_t kKvLandingScaleMantissaMin = int64_t{1} << 30;              // 2^30
constexpr int64_t kKvLandingScaleMantissaMax = (int64_t{1} << 31) - 1;        // 2^31 - 1
constexpr int64_t kKvLandingReciprocalMin = (int64_t{1} << 31) + 1;           // 2^31 + 1
constexpr int64_t kKvLandingReciprocalMax = int64_t{1} << 32;                 // 2^32

// e_t (word 1 of KvLandingReciprocals) domain -- §7.2a's own joint bound
// (Poirot 2026-07-28 finding 3 / Popper §3.2/§3.3), checked on
// KvLandingReciprocals' own word 1 (ValidateKvLandingReciprocalsDomain
// below). Commit 1b0bd10 checked KvLandingScales' word 1 instead; the
// confirmation review (D-SLM372, correcting D-SLM370(c)'s wrong section
// name; Poirot's remediation-confirmation review finding B) found that the
// composite (`LandingRescale`, forward_sites.cpp), called as
// `landing_rescale_vec(seg, m_a, r_t, e_a, e_t)` (dynamic_engine.py's own
// unpack: `_m_t, e_t, r_t = model.kv_landing_reciprocals[...]`), reads
// `e_t` and `r_t` BOTH from KvLandingReciprocals at runtime -- KvLandingScales'
// own words (`m_target, e_target`) are the composite's landed TARGET scale,
// never an argument to LandingRescale at all (forward_sites.h's own header
// comment named both sections in one breath for `(r_t, e_t)`, which is the
// ambiguity that put commit 1b0bd10's check on the wrong one). A second
// remediation round (Poirot a6d6728 second confirmation review, finding 1)
// found that a floor derived about this RATIO exponent does not bound
// KvLandingScales' word 1 at all -- that word is an ABSOLUTE per-head scale
// exponent (`canonical_scale(S_kh)`), a different quantity appearing in no
// `k` anywhere in the tree, and no artifact carrying KvLandingScales alone
// can even reach `LandingRescale` (it supplies neither `r_t` nor `e_t`).
// KvLandingScales' `e_target` therefore carries no domain check here and is
// restored to pending-consumer status (D-SLM142) until a C++ consumer of it
// exists.
// `e_t` drives the composed shift exponent `k = 62 - (e_a - e_t)` and had no
// domain check anywhere before 1b0bd10, artifact-carried or runtime-derived.
// `LandingRescale`'s own finding-3 remedy (forward_sites.cpp) now DETECTS an
// extreme composed exponent at runtime rather than silently narrowing it, so
// this load-time floor is a second, independent line of defense -- it keeps
// a hostile artifact from reaching that runtime path in the first place, for
// the exponent range this composite can still answer without needing the
// runtime detection at all.
//
// Derived, not guessed: `e_a` (the incoming carried-scale exponent shared by
// the K/V branches) is produced ONLY by the currently-wired path that can
// feed this composite -- RmsNormSite's own fold (forward_sites.cpp) --
// bounded by that fold's own arithmetic: `running.e = site_constant.e +
// d_prime_factor.e + 31` (CombineCarriedScale), possibly one further -1 on
// renormalization. `site_constant.e` is CompositionConstants' own checked
// exponent word, `[kCompositionScaleMinE, kCompositionScaleMaxE]` =
// `[-80, 7]` (ValidateCompositionConstantsDomain, this file).
// `d_prime_factor.e = -ns.s`, and `NormalizeScale`'s own contract bounds `s`
// in `[-1, 30]` (intmath.h), so `d_prime_factor.e` is in `[-30, 1]`. Summed:
// `e_a` in `[-80 - 30 + 31 - 1, 7 + 1 + 31]` = `[-80, 39]`.
//
// The left-shift branch of `LandingRescale` (k < 0, i.e. `e_a - e_t > 62`)
// carries its magnitude (this site's own documented worst case, ~2^90-2^91,
// forward_sites.cpp's own U128 comment) in a 128-bit intermediate whose
// remaining headroom is `128 - 91 = 37` bits. At `e_a`'s own worst case
// (39), the shift stays within that headroom exactly down to
// `e_t = 39 - 62 - 37 = -60`; below it, the left shift can lose bits the
// same way the reachable witness (`e_t = -1000`) does. `e_t` therefore
// carries a FLOOR only -- there is no equivalent risk on the round-divide
// branch (k >= 0), which floors correctly to 0 for arbitrarily large k
// (LandingRescale's own comment), so no upper bound is derived here.
constexpr int64_t kKvLandingExponentMin = -60;

// KvLandingScales' m_t (word 0) checked against the canonical carried-
// mantissa range every other KVC1 scale mantissa in this tree already uses
// (§7.2a third limb; S3.3). Its word 1 (`e_target`, the section's own
// landed TARGET scale exponent -- an ABSOLUTE `canonical_scale(S_kh)`, not
// the ratio exponent `LandingRescale` consumes) carries no domain check
// here: no artifact carrying KvLandingScales alone can reach
// `LandingRescale` at all (it supplies neither that composite's `r_t` nor
// its `e_t`), and no C++ consumer of `e_target` exists yet, so it is
// pending-consumer per D-SLM142 (Poirot a6d6728 second confirmation review,
// finding 1, correcting the `a62de8c`..`a6d6728` retention of a floor
// derived for a different word).
SslmModelStatus ValidateKvLandingScalesDomain(const SslmKeyedConstants& kv_landing_scales,
                                               std::string* err) {
	for (const SslmConstantEntry& e : kv_landing_scales.Entries()) {
		const int64_t m_t = SslmKeyedConstants::Value(e, 0);
		if (m_t < kKvLandingScaleMantissaMin || m_t > kKvLandingScaleMantissaMax) {
			if (err) {
				*err = "KvLandingScales entry \"" + std::string(e.name) + "\" m_t=" +
				       std::to_string(m_t) + " outside [" +
				       std::to_string(kKvLandingScaleMantissaMin) + "," +
				       std::to_string(kKvLandingScaleMantissaMax) + "]";
			}
			return SslmModelStatus::KvLandingScaleOutOfDomain;
		}
	}
	return SslmModelStatus::Ok;
}

// KvLandingReciprocals' R_t (word 2) checked against the exact domain
// `DynamicScaleReciprocal` can ever produce -- the same per-element load-time
// walk ValidateRopeTablesDomain above uses, applied identically here: walk
// every element of every entry, checked before any narrowing. e_t (word 1)
// ALSO checked against the joint-bound floor derived above -- LandingRescale's own call
// (`landing_rescale_vec(seg, m_a, r_t, e_a, e_t)`, dynamic_engine.py) reads
// BOTH r_t and e_t from THIS section at runtime, never from KvLandingScales
// (confirmation review D-SLM372, correcting D-SLM370(c)'s wrong section
// name, and Poirot's remediation-confirmation review finding B) -- so this
// is the field the composite actually consumes, checked independently of
// whether a KvLandingScales section is even present in the same artifact.
SslmModelStatus ValidateKvLandingReciprocalsDomain(const SslmKeyedConstants& kv_landing_reciprocals,
                                                    std::string* err) {
	for (const SslmConstantEntry& e : kv_landing_reciprocals.Entries()) {
		const int64_t e_t = SslmKeyedConstants::Value(e, 1);
		if (e_t < kKvLandingExponentMin) {
			if (err) {
				*err = "KvLandingReciprocals entry \"" + std::string(e.name) + "\" e_t=" +
				       std::to_string(e_t) + " below the joint-bound floor " +
				       std::to_string(kKvLandingExponentMin);
			}
			return SslmModelStatus::KvLandingReciprocalOutOfDomain;
		}
		const int64_t r_t = SslmKeyedConstants::Value(e, 2);
		if (r_t < kKvLandingReciprocalMin || r_t > kKvLandingReciprocalMax) {
			if (err) {
				*err = "KvLandingReciprocals entry \"" + std::string(e.name) + "\" R_t=" +
				       std::to_string(r_t) + " outside [" +
				       std::to_string(kKvLandingReciprocalMin) + "," +
				       std::to_string(kKvLandingReciprocalMax) + "]";
			}
			return SslmModelStatus::KvLandingReciprocalOutOfDomain;
		}
	}
	return SslmModelStatus::Ok;
}

// S3.7 (§8.3): the calibration band's own hostile-value gate -- "hostile
// bands min > max and max == 0 are load rejections" (test design §3 Cell
// 13), walked over every entry the same way ValidateKvLandingScalesDomain
// walks its own section's entries, before any entry is exposed to a
// classification call.
//
// T-1579 (Poirot b8ecff5 review, Minor 2): a band's (min, max) is "the
// calibration corpus's observed extremes" in tokens (§8.3) -- a negative
// extreme is the same species of nonsense as a zero one, and §8.3's own two
// named hostile cases (min > max, max == 0) left it admitted. `max_v == 0`
// is subsumed by `max_v <= 0` below; `min_v < 0` is the added rejection.
// Without it, `(min=-10, max=-1)` loaded `Ok` and classified a 5-token
// sequence `AboveBand` against a band whose maximum is -1.
SslmModelStatus ValidateCalibrationBandDomain(const SslmKeyedConstants& calibration_band,
                                               std::string* err) {
	for (const SslmConstantEntry& e : calibration_band.Entries()) {
		const int64_t min_v = SslmKeyedConstants::Value(e, 0);
		const int64_t max_v = SslmKeyedConstants::Value(e, 1);
		if (min_v > max_v || min_v < 0 || max_v <= 0) {
			if (err) {
				*err = "CalibrationBand entry \"" + std::string(e.name) + "\" (min=" +
				       std::to_string(min_v) + ", max=" + std::to_string(max_v) +
				       ") violates min <= max, min >= 0, and max > 0";
			}
			return SslmModelStatus::CalibrationBandOutOfDomain;
		}
	}
	return SslmModelStatus::Ok;
}

// S-HARDEN-2 (F18, join cell §17.3-3): TOK1.vocab_count x CFG1.vocab_size,
// "enforced at a named API" -- this is that API. The two blobs are parsed by
// entirely independent sub-parsers (TokenizerView::Open, ParseConfig) that
// never see each other's bytes; nothing before this slot joined them, while
// the forward path indexes the token embedding with whatever Encode() returns
// and sizes that embedding from CFG1.vocab_size. Both views are already
// populated and individually valid by the time this runs (Load's section
// loop / tokenizer-open step), so this check is a pure comparison, not a
// re-parse.
SslmModelStatus ValidateTokenizerVocabSizeJoin(const SslmModelView& view, std::string* err) {
	if (!view.has_tokenizer || !view.has_config) return SslmModelStatus::Ok;
	const int32_t tok_vocab = view.tokenizer.VocabSize();
	const int64_t cfg_vocab = int64_t(view.config.vocab_size);
	if (int64_t(tok_vocab) != cfg_vocab) {
		if (err) {
			*err = "TOK1.vocab_count (" + std::to_string(tok_vocab) + ") != CFG1.vocab_size (" +
			       std::to_string(view.config.vocab_size) + ")";
		}
		return SslmModelStatus::TokenizerVocabSizeMismatch;
	}
	return SslmModelStatus::Ok;
}

// S3.3, Sec11 S3.3 / Sec13.1 cell 4, R1-R3 (D-SLM410, D-SLM423, board T-1333).
// A thin wrapper over the existing, already-unit-tested CheckConfigGeometry
// (proof_manifest.h/.cpp), mapping ConfigGeometryResult::status onto
// ConfigGeometryKvHeadsExceedsHeads/ConfigGeometryHeadsNotDivisibleByKv/
// ConfigGeometryHiddenSizeMismatch and forwarding `.diagnostic` into `*err`,
// per the fold that declared this symbol (Claude/Vitruvius/superslm-s3.3-
// configgeometry-join-cell4-gaps-fold-2026-07-28.md Sec4). CheckConfigGeometry's
// own Zero* arms are unreachable through this call site -- ParseConfigImpl's
// BadConfigDim already rejects a zero num_attention_heads/num_key_value_heads
// upstream of ValidateSectionValues -- and are left structurally unreachable
// here rather than given a defensive mapping, per the fold's own note that
// either choice satisfies the plan's zero-boundary requirement.
SslmModelStatus ValidateConfigGeometryJoin(const SslmModelConfig& config, std::string* err) {
	const ConfigGeometryResult result = CheckConfigGeometry(
	    config.hidden_size, config.num_attention_heads, config.num_key_value_heads, config.head_dim);
	switch (result.status) {
		case ConfigGeometryStatus::Ok:
			if (err) err->clear();
			return SslmModelStatus::Ok;
		case ConfigGeometryStatus::KvHeadsExceedsHeads:
			if (err) *err = result.diagnostic;
			return SslmModelStatus::ConfigGeometryKvHeadsExceedsHeads;
		case ConfigGeometryStatus::HeadsNotDivisibleByKv:
			if (err) *err = result.diagnostic;
			return SslmModelStatus::ConfigGeometryHeadsNotDivisibleByKv;
		case ConfigGeometryStatus::HiddenSizeGeometryMismatch:
			if (err) *err = result.diagnostic;
			return SslmModelStatus::ConfigGeometryHiddenSizeMismatch;
		case ConfigGeometryStatus::ZeroAttentionHeads:
		case ConfigGeometryStatus::ZeroKeyValueHeads:
			// Unreachable through SslmModel::Load: ParseConfigImpl's BadConfigDim
			// already rejects a zero num_attention_heads/num_key_value_heads
			// before ValidateSectionValues runs. A direct, off-Load caller of
			// CheckConfigGeometry can still reach these arms; mapped to the same
			// diagnostic-carrying rejection as every other non-Ok status rather
			// than left to fall through.
			if (err) *err = result.diagnostic;
			return SslmModelStatus::ConfigGeometryHiddenSizeMismatch;
	}
	if (err) *err = "ValidateConfigGeometryJoin: unrecognized ConfigGeometryStatus";
	return SslmModelStatus::ConfigGeometryHiddenSizeMismatch;
}

// S3.3, Sec11 S3.3 / Sec13.1 cell 4, R4 -- the ROP1<->CFG1 join itself
// (D-SLM410, D-SLM421, D-SLM423, board T-1333). Resolves `rope_tables.
// Tensor("cos")` / `Tensor("sin")` the same way RopeApplySite does
// (forward_sites.cpp), then for each tensor that is present, checks
// `elem_count == uint64_t(context_cap) * (head_dim / 2)` independently,
// returning RopeTablesShapeMismatchConfig with a diagnostic naming the
// tensor, its actual elem_count, and the expected value on mismatch. An
// absent tensor is not this check's concern -- RopeApplySite's own
// RopeTableTensorMissing covers it at forward time (D-SLM81/D-SLM413); this
// function bounds the shape of whichever tensor is present, it does not
// assert presence.
SslmModelStatus ValidateRopeTablesShapeAgainstConfig(const SslmTensorManifest& rope_tables,
                                                      uint32_t context_cap, uint32_t head_dim,
                                                      std::string* err) {
	const uint64_t expected = static_cast<uint64_t>(context_cap) * static_cast<uint64_t>(head_dim / 2);
	const SslmTensorView* cos = rope_tables.Tensor("cos");
	if (cos != nullptr && cos->elem_count != expected) {
		if (err) {
			*err = "RopeTables \"cos\" elem_count (" + std::to_string(cos->elem_count) +
			       ") != context_cap * (head_dim / 2) (" + std::to_string(context_cap) + " * (" +
			       std::to_string(head_dim) + " / 2) = " + std::to_string(expected) + ")";
		}
		return SslmModelStatus::RopeTablesShapeMismatchConfig;
	}
	const SslmTensorView* sin = rope_tables.Tensor("sin");
	if (sin != nullptr && sin->elem_count != expected) {
		if (err) {
			*err = "RopeTables \"sin\" elem_count (" + std::to_string(sin->elem_count) +
			       ") != context_cap * (head_dim / 2) (" + std::to_string(context_cap) + " * (" +
			       std::to_string(head_dim) + " / 2) = " + std::to_string(expected) + ")";
		}
		return SslmModelStatus::RopeTablesShapeMismatchConfig;
	}
	if (err) err->clear();
	return SslmModelStatus::Ok;
}

// S-HARDEN-1's schema-value gate (D-SLM141): one pass over the already
// sub-parsed views, applying each present section's domain descriptor. Runs
// AFTER every present section's structural sub-parse has already succeeded
// (Load's section loop) and BEFORE `out` is exposed to the caller — the
// boundary where an artifact value enters and can still be refused.
// KvLandingScales/KvLandingReciprocals are now wired below
// (ValidateKvLandingScalesDomain/ValidateKvLandingReciprocalsDomain), S3.3's
// green-phase construction -- D-SLM142's "pending-consumer" status is
// superseded (S3.3 is the C27 consumer).
SslmModelStatus ValidateSectionValues(const SslmModelView& view, std::string* err) {
	if (view.has_composition_constants) {
		const SslmModelStatus s = ValidateCompositionConstantsDomain(view.composition_constants, err);
		if (s != SslmModelStatus::Ok) return s;
	}
	if (view.has_weight_scales) {
		const SslmModelStatus s = ValidateWeightScalesDomain(view.weight_scales, err);
		if (s != SslmModelStatus::Ok) return s;
	}
	if (view.has_rope_tables) {
		const SslmModelStatus s = ValidateRopeTablesDomain(view.rope_tables, err);
		if (s != SslmModelStatus::Ok) return s;
	}
	// BIA1's bias codes carry no load-time value-domain check (T-1657, D-SLM621/641).
	// Unlike every other section this function validates, B[j]'s true constraint is
	// joint with a runtime-derived reciprocal and exponent (R_a, e_a) that do not
	// exist until the C28 call site -- the same shape already ruled for bias.q_b
	// (SuperSLM_S3a_WalkingSkeleton_Plan.md §4.4/§7.2). The check lives at
	// CheckBiasAccumulateMagnitudeDomain (checked_chain_funnel.h, T-1657 Poirot
	// Critical C-1, D-SLM674), not here. (Its sibling CheckBiasReconcileMagnitudeDomain
	// checks only BiasReconcile's own per-element magnitude and has no production
	// caller; the per-element gate ApplyBiasReconcileRow (forward_sites.cpp) actually
	// runs is the accumulate predicate, which additionally proves the running
	// accumulator sum representable.)
	if (view.has_kv_landing_scales) {
		const SslmModelStatus s = ValidateKvLandingScalesDomain(view.kv_landing_scales, err);
		if (s != SslmModelStatus::Ok) return s;
	}
	if (view.has_kv_landing_reciprocals) {
		const SslmModelStatus s = ValidateKvLandingReciprocalsDomain(view.kv_landing_reciprocals, err);
		if (s != SslmModelStatus::Ok) return s;
	}
	if (view.has_calibration_band) {
		const SslmModelStatus s = ValidateCalibrationBandDomain(view.calibration_band, err);
		if (s != SslmModelStatus::Ok) return s;
	}
	// T-2041 (Poirot c81e48c review, Significant 2): DeltaFoldScales/UFoldScales's own
	// value-domain check -- an out-of-domain exponent (outside the signed
	// [kAmplifyingScaleExponentMin, kAmplifyingScaleExponentMax] design Sec4/Sec9 specifies) or a
	// non-boolean identity now REJECTS at load, the same two-phase (structural parse, then value
	// validation) gate every other typed section here already gets.
	if (view.has_delta_fold_scales) {
		const SslmModelStatus s =
		    ValidateAmplifyingFoldScalesDomain(view.delta_fold_scales.Entries(), err);
		if (s != SslmModelStatus::Ok) return s;
	}
	if (view.has_u_fold_scales) {
		const SslmModelStatus s = ValidateAmplifyingFoldScalesDomain(view.u_fold_scales.Entries(), err);
		if (s != SslmModelStatus::Ok) return s;
	}
	// S-HARDEN-2 (F18): the tokenizer's own cross-section join.
	{
		const SslmModelStatus s = ValidateTokenizerVocabSizeJoin(view, err);
		if (s != SslmModelStatus::Ok) return s;
	}
	// S3.3, §13.1 cell 4 (D-SLM410, D-SLM421, D-SLM423, T-1333): the
	// config-geometry x tensor-shape join. The double-guard on the second
	// call matches ValidateTokenizerVocabSizeJoin's own pattern above: both
	// sections must be present and individually valid before their
	// cross-section relation is meaningful.
	if (view.has_config) {
		const SslmModelStatus s = ValidateConfigGeometryJoin(view.config, err);
		if (s != SslmModelStatus::Ok) return s;
	}
	if (view.has_rope_tables && view.has_config) {
		const SslmModelStatus s = ValidateRopeTablesShapeAgainstConfig(
		    view.rope_tables, view.config.context_cap, view.config.head_dim, err);
		if (s != SslmModelStatus::Ok) return s;
	}
	return SslmModelStatus::Ok;
}

}  // namespace

SslmModelStatus SslmModel::Load(const uint8_t* data, size_t size, SslmModelView& out, std::string* err) {
	return internal::WrapBadAllocContract(
	    [&] { return SslmModelAccess::LoadImpl(data, size, out, err); });
}

SslmModelStatus SslmModelAccess::LoadImpl(const uint8_t* data, size_t size, SslmModelView& out,
                                          std::string* err) {
	internal::MaybeThrowInjectedBadAllocFault();
	out = SslmModelView{};
	if (err) err->clear();

	SslmError aerr;
	const SslmStatus astatus = SslmArtifact::OpenFromMemory(data, size, out.backing_, &aerr);
	if (astatus != SslmStatus::Ok) {
		if (err) {
			*err = std::string("artifact rejected (") + SslmStatusName(astatus) + "): " + aerr.message;
		}
		return SslmModelStatus::ArtifactRejected;
	}

	// T-1894 (design Sec31.2.1, round 4/D-SLM2423): the header `flags` bit is
	// a property of the artifact itself, set once here, before the
	// per-section loop -- it is not a section and has no `has_*` flag. On any
	// LATER rejection in this function, `out` is reset to `SslmModelView{}`
	// by every existing return path below, which restores this field to its
	// own default (`false`) exactly like every other member already is -- no
	// special-casing added.
	out.option_g_fused_k_landing = out.backing_.OptionGFusedKLandingEnabled();

	for (const SslmSectionView& section : out.backing_.Sections()) {
		SslmModelStatus s = SslmModelStatus::Ok;
		switch (section.type) {
			case SslmSectionType::Config:
				s = ParseConfig(section, out.config, err);
				out.has_config = (s == SslmModelStatus::Ok);
				break;
			case SslmSectionType::SigmoidLut:
				s = ParseSigmoidLut(section, out.sigmoid_lut, err);
				out.has_sigmoid_lut = (s == SslmModelStatus::Ok);
				break;
			case SslmSectionType::Weights:
				s = SslmTensorManifest::Parse(section, out.weights, err);
				out.has_weights = (s == SslmModelStatus::Ok);
				break;
			case SslmSectionType::Biases:
				s = SslmTensorManifest::Parse(section, out.biases, err);
				out.has_biases = (s == SslmModelStatus::Ok);
				break;
			case SslmSectionType::RopeTables:
				s = SslmTensorManifest::Parse(section, out.rope_tables, err);
				out.has_rope_tables = (s == SslmModelStatus::Ok);
				break;
			case SslmSectionType::WeightScales:
				s = SslmTensorManifest::Parse(section, out.weight_scales, err);
				out.has_weight_scales = (s == SslmModelStatus::Ok);
				break;
			case SslmSectionType::CompositionConstants:
				s = SslmKeyedConstants::Parse(section, out.composition_constants, err);
				out.has_composition_constants = (s == SslmModelStatus::Ok);
				break;
			case SslmSectionType::KvLandingScales:
				s = SslmKeyedConstants::Parse(section, out.kv_landing_scales, err);
				out.has_kv_landing_scales = (s == SslmModelStatus::Ok);
				break;
			case SslmSectionType::KvLandingReciprocals:
				s = SslmKeyedConstants::Parse(section, out.kv_landing_reciprocals, err);
				out.has_kv_landing_reciprocals = (s == SslmModelStatus::Ok);
				break;
			case SslmSectionType::CalibrationBand:
				s = SslmKeyedConstants::Parse(section, out.calibration_band, err);
				out.has_calibration_band = (s == SslmModelStatus::Ok);
				break;
			case SslmSectionType::DeltaFoldScales:
				// T-2041 (Poirot c81e48c review, Significant 2): DeltaFoldScales/UFoldScales now
				// reach a real sub-parser -- previously fell through `default: continue`,
				// meaning they parsed structurally (via SslmArtifact::OpenFromMemory) but were
				// never registered in SslmModelView and never reached ValidateSectionValues, so
				// an out-of-domain triple loaded clean. `SslmDeltaFoldScaleView::Parse` itself
				// enforces the section-type gate (Sec9's own "constructed only by parsing a
				// section whose declared type matches the wrapper's own kind") before any byte
				// is read; the VALUE-domain check runs below, in ValidateSectionValues, the same
				// two-phase split every other typed section here already uses.
				s = SslmDeltaFoldScaleView::Parse(section, out.delta_fold_scales, err);
				out.has_delta_fold_scales = (s == SslmModelStatus::Ok);
				break;
			case SslmSectionType::UFoldScales:
				s = SslmUFoldScaleView::Parse(section, out.u_fold_scales, err);
				out.has_u_fold_scales = (s == SslmModelStatus::Ok);
				break;
			default:
				// No sub-parser owns this section type here (Provenance, Scales,
				// Calibration, GoldenHashes, Tokenizer, ChatTemplate, UnicodeTables,
				// SchemaMasks) — the S1 sections have their own TokenizerView entry
				// point, and the rest carry no typed sub-parse in this tree. Already
				// structurally validated by SslmArtifact::OpenFromMemory.
				continue;
		}
		if (s != SslmModelStatus::Ok) {
			out = SslmModelView{};  // fail closed — never a partial view
			return s;
		}
	}

	// S-HARDEN-2 (F18/F6/F7/F15): the tokenizer's own entry point, driven from the
	// same `SslmModel::Load` boundary as every other section — TokenizerView::Open
	// takes the whole artifact (it needs both the Tokenizer and UnicodeTables
	// sections together), so it is not a per-section-type case in the loop above.
	// Neither section present is a valid tokenizer-less model artifact; exactly one
	// present, or a structurally malformed TOK1/UNI1, is a rejection here — never a
	// partial tokenizer view.
	{
		const SslmSectionView* tok_sec = out.backing_.Section(SslmSectionType::Tokenizer);
		const SslmSectionView* uni_sec = out.backing_.Section(SslmSectionType::UnicodeTables);
		if (tok_sec != nullptr || uni_sec != nullptr) {
			if (tok_sec == nullptr || uni_sec == nullptr) {
				out = SslmModelView{};
				return Reject(SslmModelStatus::TokenizerRejected, err,
				              "artifact carries one of Tokenizer/UnicodeTables without the other");
			}
			std::string terr;
			if (!TokenizerView::Open(out.backing_, out.tokenizer, &terr)) {
				out = SslmModelView{};
				if (err) *err = "Tokenizer rejected: " + terr;
				return SslmModelStatus::TokenizerRejected;
			}
			out.has_tokenizer = true;
		}
	}

	const SslmModelStatus vstatus = ValidateSectionValues(out, err);
	if (vstatus != SslmModelStatus::Ok) {
		out = SslmModelView{};
		return vstatus;
	}

	return SslmModelStatus::Ok;
}

// S3.7 (§8.3): looks up the "token_length" entry by name -- the fixture
// family's own naming convention (§8.3's costed table, "the Recommended
// row") -- rather than assuming the section carries exactly one entry
// unconditionally. `ValidateCalibrationBandDomain` (above) already rejects a
// hostile (min, max) at load time, so any entry reached here is well-formed.
SslmCalibrationBandVerdict ClassifyCalibrationBand(const SslmModelView& view,
                                                    int64_t token_length) noexcept {
	if (!view.has_calibration_band) return SslmCalibrationBandVerdict::BandUnknown;
	const SslmConstantEntry* entry = view.calibration_band.Entry("token_length");
	if (entry == nullptr) return SslmCalibrationBandVerdict::BandUnknown;
	const int64_t min_v = SslmKeyedConstants::Value(*entry, 0);
	const int64_t max_v = SslmKeyedConstants::Value(*entry, 1);
	if (token_length < min_v) return SslmCalibrationBandVerdict::BelowBand;
	if (token_length > max_v) return SslmCalibrationBandVerdict::AboveBand;
	return SslmCalibrationBandVerdict::InBand;
}

} // namespace superslm

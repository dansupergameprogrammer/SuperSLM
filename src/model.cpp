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

#include <algorithm>
#include <cstdint>
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
	}
	return "Unknown";
}

const uint8_t* ManifestMagicFor(SslmSectionType type) noexcept {
	switch (type) {
		case SslmSectionType::Weights: return kWeightsMagic;
		case SslmSectionType::Biases: return kBiasesMagic;
		case SslmSectionType::RopeTables: return kRopeMagic;
		default: return nullptr;
	}
}

SslmModelStatus SslmTensorManifest::Parse(const SslmSectionView& section,
                                          SslmTensorManifest& out, std::string* err) {
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

} // namespace superslm

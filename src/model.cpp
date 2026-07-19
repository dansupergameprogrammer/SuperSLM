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
#include <cstring>
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
	}
	return "Unknown";
}

const uint8_t* ConstantsMagicFor(SslmSectionType type) noexcept {
	switch (type) {
		case SslmSectionType::CompositionConstants:
		case SslmSectionType::KvLandingScales:
		case SslmSectionType::KvLandingReciprocals:
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

SslmModelStatus SslmKeyedConstants::Parse(const SslmSectionView& section,
                                          SslmKeyedConstants& out, std::string* err) {
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

SslmModelStatus ParseConfig(const SslmSectionView& section, SslmModelConfig& out,
                            std::string* err) {
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

	if (tie > 1)
		return Reject(SslmModelStatus::BadConfigBool, err, "tie_word_embeddings not in {0,1}");
	if (kvp > 1)
		return Reject(SslmModelStatus::BadKvPrecision, err, "kv_precision not in {0,1}");

	c.tie_word_embeddings = (tie == 1);
	c.kv_precision = (kvp == 1) ? SslmKvPrecision::Int16 : SslmKvPrecision::Int8;

	out = c;
	return SslmModelStatus::Ok;
}

} // namespace superslm

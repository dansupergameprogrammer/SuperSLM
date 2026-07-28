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
		case SslmModelStatus::RopeTableEntryOutOfDomain: return "RopeTableEntryOutOfDomain";
		case SslmModelStatus::BiasCodeOutOfDomain: return "BiasCodeOutOfDomain";
		case SslmModelStatus::TokenizerRejected: return "TokenizerRejected";
		case SslmModelStatus::TokenizerVocabSizeMismatch: return "TokenizerVocabSizeMismatch";
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
constexpr int32_t kWeightScaleShiftMin = 0;
constexpr int32_t kWeightScaleShiftMax = 31;

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

// BIA1's load-time value-domain descriptor (SuperSLM_S3a_WalkingSkeleton_Plan.md
// §7.2a third limb, §4.4; S3.2; Claude/Curie/superslm-s3.2-weightless-and-
// projection-sites-test-design-2026-07-28.md §3.4/§4.7). Curie's record derives
// the bound by execution: R_a's maximum over the C19 reciprocal's own domain is
// 2^32, so keeping B[j]*R_a inside int64 requires |B[j]| <= floor((2^63-1) / 2^32)
// == INT32_MAX, verified tight (one past it does not fit).
//
// THIS IS A STUB (S3.2 red-phase): it does not compare any element against that
// bound and always accepts. It is written this way, rather than as a fixed-wrong
// rejection (the sentinel style every other stub in this campaign uses), because
// this function is wired into ValidateSectionValues below, which every existing
// SslmModel::Load call already exercises on every artifact carrying a Biases
// section -- an unconditionally-rejecting stub would fail every currently-green
// fixture that loads one, not only the not-yet-authored hostile-value cell this
// stub is staged for. A permissive stub changes nothing about today's passing
// suite and still fails unmistakably the moment Curie's real hostile-value cell
// (an artifact with one bias code at 2^31, one past the derived bound) asserts
// SslmModel::Load returns BiasCodeOutOfDomain and gets Ok instead. S3.2's green
// phase replaces this body with the real |B[j]| <= INT32_MAX comparison over
// every element of every tensor in `biases`.
SslmModelStatus ValidateBiasesDomain(const SslmTensorManifest& /*biases*/, std::string* /*err*/) {
	return SslmModelStatus::Ok;  // stub -- deliberately does not check the bound
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

// S-HARDEN-1's schema-value gate (D-SLM141): one pass over the already
// sub-parsed views, applying each present section's domain descriptor. Runs
// AFTER every present section's structural sub-parse has already succeeded
// (Load's section loop) and BEFORE `out` is exposed to the caller — the
// boundary where an artifact value enters and can still be refused.
// KvLandingScales/KvLandingReciprocals are deliberately not checked here
// (D-SLM142: pending-consumer rows — no consumer exists in the tree yet, so
// no true domain exists to declare).
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
	if (view.has_biases) {
		const SslmModelStatus s = ValidateBiasesDomain(view.biases, err);
		if (s != SslmModelStatus::Ok) return s;
	}
	// S-HARDEN-2 (F18): the tokenizer's own cross-section join.
	{
		const SslmModelStatus s = ValidateTokenizerVocabSizeJoin(view, err);
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

} // namespace superslm

// SuperSLM model view (`.sslm` model sections) — version 1.
//
// The runtime typed view over a converted, quantized model's array sections. It sits
// on top of SslmArtifact (include/superslm/artifact.h): SslmArtifact validates the
// file's structure and whole-file integrity; SslmTensorManifest then parses one array
// section's internal tensor manifest (WGT1/BIA1/ROP1, docs/sslm_format.md "Model
// sub-formats"). Standard library only — Layer 1 is independently embeddable, no
// third-party runtime dependency (SuperSLM_Plan.md §11, D-SLM13).
//
// Like TokenizerView, the manifest sub-parse is a trust boundary held to the T-129 bar
// (§17 dim 2): the artifact's integrity hash proves the bytes are intact, but a crafted
// integrity-valid artifact can still carry a malformed manifest, so every descriptor
// field is validated against the section's own bounds before any tensor byte is exposed.
// Deviation is rejection with a status, never a silent partial view.
#ifndef SUPERSLM_MODEL_H
#define SUPERSLM_MODEL_H

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include "superslm/artifact.h"

namespace superslm {

// Manifest geometry (v1). See docs/sslm_format.md "Model sub-formats".
inline constexpr uint32_t kMaxTensors = 65536;
inline constexpr uint32_t kMaxTensorRank = 4;
inline constexpr uint32_t kTensorDescBytes = 48;    // one TensorDesc
inline constexpr uint32_t kManifestHeaderBytes = 16; // magic + version + count + name_blob_len
inline constexpr uint32_t kManifestVersion = 1;
inline constexpr uint8_t kWeightsMagic[4] = {'W', 'G', 'T', '1'};
inline constexpr uint8_t kBiasesMagic[4] = {'B', 'I', 'A', '1'};
inline constexpr uint8_t kRopeMagic[4] = {'R', 'O', 'P', '1'};

// Keyed numeric-constant geometry (v1). See docs/sslm_format.md "Keyed numeric-constant
// blob — KVC1". The three keyed integer-constant sections (CompositionConstants,
// KvLandingScales, KvLandingReciprocals) share this layout; the section type fixes
// value_words (2 or 3), and every value — including the exponent e — is a little-endian int64.
inline constexpr uint32_t kMaxConstantEntries = 1048576;
inline constexpr uint32_t kConstantHeaderBytes = 24;  // magic+version+count+value_words+name_blob_len+reserved
inline constexpr uint32_t kConstantDescBytes = 8;     // name_off + name_len
inline constexpr uint8_t kConstantsMagic[4] = {'K', 'V', 'C', '1'};

// A validated view of one tensor packed in an array section's manifest. `data` points
// into the artifact's owned buffer and is valid for the artifact's lifetime; `name`
// points into the same section's name blob.
struct SslmTensorView {
	std::string_view name;
	SslmDtype dtype{};                       // the section's element dtype
	uint32_t rank = 0;                       // in [1, kMaxTensorRank]
	uint32_t shape[kMaxTensorRank] = {};     // shape[i], i < rank; 0 at/after rank
	const uint8_t* data = nullptr;           // element_size * elem_count bytes
	uint64_t elem_count = 0;                 // == product(shape[0..rank))
};

// Every way a tensor manifest can be rejected. `Ok` is the only non-error value. The
// codes mirror the rejection list in docs/sslm_format.md "Model sub-formats".
enum class SslmModelStatus {
	Ok = 0,
	SectionTooShort,            // section byte_size < kManifestHeaderBytes
	BadManifestMagic,           // first four bytes are not the section's magic
	UnsupportedManifestVersion, // manifest version != kManifestVersion
	TooManyTensors,             // tensor_count > kMaxTensors
	ManifestOutOfBounds,        // header + descriptors + name blob exceeds byte_size
	BadTensorName,              // a descriptor's name range is outside the name blob
	EmptyTensorName,            // name_len == 0
	DuplicateTensorName,        // two descriptors carry the same name
	BadTensorRank,              // rank == 0 or rank > kMaxTensorRank
	BadTensorShape,             // shape nonzero at/after rank, or zero before it
	ShapeCountMismatch,         // elem_count != product(shape), or product overflows
	TensorMisaligned,           // data_off not a multiple of the element size
	TensorOutOfBounds,          // a tensor's data range exceeds byte_size or overflows
	TensorOverlap,              // two tensors' data ranges overlap
	BadDescriptorReserved,      // a descriptor's reserved field != 0
	// --- KVC1 keyed-constant sub-parse ---
	BadConstantsMagic,          // first four bytes are not 'KVC1'
	UnsupportedConstantsVersion,// KVC1 version != kManifestVersion
	TooManyConstantEntries,     // entry_count > kMaxConstantEntries
	BadValueWords,              // value_words not in {2,3}, or not the section type's required count
	ConstantsOutOfBounds,       // header + descriptors + values + name blob exceeds byte_size
	BadEntryName,               // an entry's name range is outside the name blob
	EmptyEntryName,             // an entry's name_len == 0
	DuplicateEntryName,         // two entries carry the same name
	BadConstantsReserved,       // the KVC1 header reserved field != 0
};

// Human-readable name for a status, for diagnostics and test messages.
const char* SslmModelStatusName(SslmModelStatus s) noexcept;

// The magic a section type's manifest must carry, or nullptr if the type has no
// tensor manifest. Only Weights/Biases/RopeTables carry one.
const uint8_t* ManifestMagicFor(SslmSectionType type) noexcept;

// A parsed, fully validated tensor manifest for one array section. Constructed only
// through Parse; a default instance is empty.
class SslmTensorManifest {
public:
	SslmTensorManifest() = default;

	// Parse the tensor manifest in `section`. The section must already be validated by
	// SslmArtifact (structure + integrity); its type fixes the expected magic and the
	// tensors' element dtype. On Ok, `out` owns the validated tensor views. On any
	// error, `out` is left empty and `err` (if non-null) carries a diagnostic. Never
	// throws; never exposes a tensor byte before every descriptor passes validation.
	static SslmModelStatus Parse(const SslmSectionView& section, SslmTensorManifest& out,
	                             std::string* err);

	const std::vector<SslmTensorView>& Tensors() const noexcept { return tensors_; }

	// The tensor with the given name, or nullptr if absent.
	const SslmTensorView* Tensor(std::string_view name) const noexcept;

private:
	std::vector<SslmTensorView> tensors_;
};

// The KVC1 magic a section type carries, or nullptr if the type has no keyed-constant
// table. Only CompositionConstants/KvLandingScales/KvLandingReciprocals carry one.
const uint8_t* ConstantsMagicFor(SslmSectionType type) noexcept;

// The value_words a keyed-constant section type requires (2 or 3), or 0 if the type has
// no keyed-constant table.
uint32_t ExpectedValueWords(SslmSectionType type) noexcept;

// A validated view of one keyed-constant entry. `name` points into the section's name
// blob; `values` points at `value_words` little-endian int64s in the section's value
// array — read them with the byte-assembly reader, never a cast (the array is not
// guaranteed aligned for an int64 load).
struct SslmConstantEntry {
	std::string_view name;
	const uint8_t* values = nullptr;   // value_words * 8 bytes, little-endian int64 each
	uint32_t value_words = 0;
};

// A parsed, fully validated keyed numeric-constant table (KVC1) for one section.
// Constructed only through Parse; a default instance is empty.
class SslmKeyedConstants {
public:
	SslmKeyedConstants() = default;

	// Parse the KVC1 table in `section` (whose type fixes the expected value_words). The
	// section must already be validated by SslmArtifact. The parse is a hostile-input
	// trust boundary: it fails closed on any malformed field, leaving `out` empty. On Ok,
	// `out` owns the validated entry views. Never throws; never exposes an entry before
	// every descriptor passes validation.
	static SslmModelStatus Parse(const SslmSectionView& section, SslmKeyedConstants& out,
	                             std::string* err);

	const std::vector<SslmConstantEntry>& Entries() const noexcept { return entries_; }

	// The entry with the given name, or nullptr if absent.
	const SslmConstantEntry* Entry(std::string_view name) const noexcept;

	// Read value index `w` (< value_words) of `entry` as a signed int64 (little-endian
	// byte assembly). Behavior is undefined if `w >= entry.value_words`.
	static int64_t Value(const SslmConstantEntry& entry, uint32_t w) noexcept;

private:
	std::vector<SslmConstantEntry> entries_;
};

} // namespace superslm

#endif // SUPERSLM_MODEL_H

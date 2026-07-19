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

} // namespace superslm

#endif // SUPERSLM_MODEL_H

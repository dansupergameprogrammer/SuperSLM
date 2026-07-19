// SuperSLM model view — tensor-manifest sub-parse (WGT1/BIA1/ROP1).
//
// RED-FIRST STUB (S2.0a). The contract in include/superslm/model.h is declared; the
// manifest parse is not yet implemented. Parse() accepts and produces an empty view,
// so the hostile-input and feature-oracle red suite fails until the parse is built.
// Do not treat this file as a working parser.
#include "superslm/model.h"

namespace superslm {

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

// STUB: not implemented. Leaves `out` empty and reports Ok so the red suite fails.
SslmModelStatus SslmTensorManifest::Parse(const SslmSectionView& /*section*/,
                                          SslmTensorManifest& out, std::string* /*err*/) {
	out.tensors_.clear();
	return SslmModelStatus::Ok;
}

const SslmTensorView* SslmTensorManifest::Tensor(std::string_view name) const noexcept {
	for (const auto& t : tensors_) {
		if (t.name == name) return &t;
	}
	return nullptr;
}

} // namespace superslm

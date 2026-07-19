#include "superslm/artifact.h"

#include "superslm/sha256.h"

namespace superslm {

uint32_t DtypeSize(uint32_t dtype) noexcept {
	switch (static_cast<SslmDtype>(dtype)) {
		case SslmDtype::Raw: return 1;
		case SslmDtype::Int8: return 1;
		case SslmDtype::Int32: return 4;
		case SslmDtype::Int64: return 8;
		case SslmDtype::Uint8: return 1;
		case SslmDtype::Float32: return 4;
		case SslmDtype::Float64: return 8;
	}
	return 0; // unknown
}

bool IsKnownSectionType(uint32_t type) noexcept {
	switch (static_cast<SslmSectionType>(type)) {
		case SslmSectionType::Config:
		case SslmSectionType::Provenance:
		case SslmSectionType::Weights:
		case SslmSectionType::Biases:
		case SslmSectionType::RopeTables:
		case SslmSectionType::Scales:
		case SslmSectionType::WeightScales:
		case SslmSectionType::CompositionConstants:
		case SslmSectionType::KvLandingScales:
		case SslmSectionType::KvLandingReciprocals:
		case SslmSectionType::Calibration:
		case SslmSectionType::GoldenHashes:
		case SslmSectionType::Tokenizer:
		case SslmSectionType::ChatTemplate:
		case SslmSectionType::SchemaMasks:
			return true;
	}
	return false;
}

const char* SslmStatusName(SslmStatus s) noexcept {
	switch (s) {
		case SslmStatus::Ok: return "Ok";
		case SslmStatus::Truncated: return "Truncated";
		case SslmStatus::BadMagic: return "BadMagic";
		case SslmStatus::UnsupportedVersion: return "UnsupportedVersion";
		case SslmStatus::BadHeader: return "BadHeader";
		case SslmStatus::TooManySections: return "TooManySections";
		case SslmStatus::FileSizeMismatch: return "FileSizeMismatch";
		case SslmStatus::BadAlignment: return "BadAlignment";
		case SslmStatus::Misaligned: return "Misaligned";
		case SslmStatus::SectionOutOfBounds: return "SectionOutOfBounds";
		case SslmStatus::SectionOverlap: return "SectionOverlap";
		case SslmStatus::BadDtype: return "BadDtype";
		case SslmStatus::SizeMismatch: return "SizeMismatch";
		case SslmStatus::UnknownSection: return "UnknownSection";
		case SslmStatus::DuplicateSection: return "DuplicateSection";
		case SslmStatus::MissingSection: return "MissingSection";
		case SslmStatus::IntegrityMismatch: return "IntegrityMismatch";
		case SslmStatus::IoError: return "IoError";
	}
	return "?";
}

// ---------------------------------------------------------------------------
// S0 RED-FIRST STUB. The validating loader is authored test-first: Curie's red
// suite is written against this contract and must fail red before Brunel builds
// the body. Until then Open* reports the loader is unbuilt and yields no sections.
// (SuperSLM_Plan.md §15 red-first TDD law; Curie owns the S0 loader red suite.)
// ---------------------------------------------------------------------------

SslmStatus SslmArtifact::OpenFromMemory(const uint8_t* /*data*/, size_t /*size*/,
                                        SslmArtifact& out, SslmError* err) {
	out = SslmArtifact{};
	if (err) {
		err->code = SslmStatus::IoError;
		err->section_index = kNoSection;
		err->message = "OpenFromMemory: loader not implemented (S0 red-first stub)";
	}
	return SslmStatus::IoError;
}

SslmStatus SslmArtifact::OpenFromFile(const char* /*path*/, SslmArtifact& out,
                                      SslmError* err) {
	out = SslmArtifact{};
	if (err) {
		err->code = SslmStatus::IoError;
		err->section_index = kNoSection;
		err->message = "OpenFromFile: loader not implemented (S0 red-first stub)";
	}
	return SslmStatus::IoError;
}

std::string SslmArtifact::FingerprintHex() const {
	return ToHex(integrity_);
}

const SslmSectionView* SslmArtifact::Section(SslmSectionType type) const noexcept {
	for (const auto& s : sections_) {
		if (s.type == type) return &s;
	}
	return nullptr;
}

} // namespace superslm

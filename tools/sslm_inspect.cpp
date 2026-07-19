// sslm_inspect — load a `.sslm` with the runtime loader and print what it found.
// Build-time debug tool and the cross-check that a Python-emitted artifact loads in
// C++ with a matching fingerprint. Not part of the shipped library.
//
// Build (from repo root): see tools/build_inspect.bat, or:
//   cl /std:c++20 /EHsc /Iinclude src/artifact.cpp src/sha256.cpp tools/sslm_inspect.cpp
#include "superslm/artifact.h"

#include <cstdio>

using namespace superslm;

static const char* TypeName(SslmSectionType t) {
	switch (t) {
		case SslmSectionType::Config: return "Config";
		case SslmSectionType::Provenance: return "Provenance";
		case SslmSectionType::Weights: return "Weights";
		case SslmSectionType::Biases: return "Biases";
		case SslmSectionType::RopeTables: return "RopeTables";
		case SslmSectionType::Scales: return "Scales";
		case SslmSectionType::WeightScales: return "WeightScales";
		case SslmSectionType::CompositionConstants: return "CompositionConstants";
		case SslmSectionType::KvLandingScales: return "KvLandingScales";
		case SslmSectionType::KvLandingReciprocals: return "KvLandingReciprocals";
		case SslmSectionType::Calibration: return "Calibration";
		case SslmSectionType::GoldenHashes: return "GoldenHashes";
		case SslmSectionType::Tokenizer: return "Tokenizer";
		case SslmSectionType::ChatTemplate: return "ChatTemplate";
		case SslmSectionType::UnicodeTables: return "UnicodeTables";
		case SslmSectionType::SchemaMasks: return "SchemaMasks";
	}
	return "?";
}

int main(int argc, char** argv) {
	if (argc < 2) {
		std::printf("usage: sslm_inspect <artifact.sslm>\n");
		return 2;
	}
	SslmArtifact art;
	SslmError err;
	SslmStatus st = SslmArtifact::OpenFromFile(argv[1], art, &err);
	if (st != SslmStatus::Ok) {
		std::printf("REJECTED: %s (section %u) — %s\n", SslmStatusName(st),
		            err.section_index, err.message.c_str());
		return 1;
	}
	std::printf("OK  format_version=%u  file_bytes=%llu\n", art.FormatVersion(),
	            (unsigned long long)art.FileBytes());
	std::printf("fingerprint %s\n", art.FingerprintHex().c_str());
	std::printf("%zu sections:\n", art.Sections().size());
	for (const auto& s : art.Sections()) {
		std::printf("  %-22s dtype=%u  bytes=%llu  elems=%llu  align=%u\n",
		            TypeName(s.type), (unsigned)s.dtype,
		            (unsigned long long)s.byte_size, (unsigned long long)s.elem_count,
		            s.alignment);
	}
	return 0;
}

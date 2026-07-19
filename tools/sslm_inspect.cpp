// sslm_inspect — load a `.sslm` with the runtime loader and print what it found.
// Build-time debug tool and the cross-check that a Python-emitted artifact loads in
// C++ with a matching fingerprint. Not part of the shipped library.
//
// Build (from repo root): see tools/build_inspect.bat, or:
//   cl /std:c++20 /EHsc /Iinclude src/artifact.cpp src/sha256.cpp tools/sslm_inspect.cpp
#include "superslm/artifact.h"
#include "superslm/model.h"

#include <cstdio>

using namespace superslm;

// Parse a model section through the ModelView and print what it read back — the
// round-trip proof that a Python-emitted section is not just structurally valid but
// interpretable (tensor names/shapes, constant tuples, config dims recovered).
static bool InspectModelSection(const SslmSectionView& s) {
	std::string err;
	if (ManifestMagicFor(s.type) != nullptr) {
		SslmTensorManifest m;
		SslmModelStatus st = SslmTensorManifest::Parse(s, m, &err);
		if (st != SslmModelStatus::Ok) {
			std::printf("      ModelView REJECTED: %s — %s\n", SslmModelStatusName(st), err.c_str());
			return false;
		}
		std::printf("      -> %zu tensors", m.Tensors().size());
		if (!m.Tensors().empty()) {
			const auto& t = m.Tensors().front();
			std::printf("; first \"%.*s\" rank=%u shape=[%u,%u,%u,%u]",
			            (int)t.name.size(), t.name.data(), t.rank,
			            t.shape[0], t.shape[1], t.shape[2], t.shape[3]);
		}
		std::printf("\n");
		return true;
	}
	if (ConstantsMagicFor(s.type) != nullptr) {
		SslmKeyedConstants k;
		SslmModelStatus st = SslmKeyedConstants::Parse(s, k, &err);
		if (st != SslmModelStatus::Ok) {
			std::printf("      ModelView REJECTED: %s — %s\n", SslmModelStatusName(st), err.c_str());
			return false;
		}
		std::printf("      -> %zu entries (value_words=%u)", k.Entries().size(), ExpectedValueWords(s.type));
		if (!k.Entries().empty()) {
			const auto& e = k.Entries().front();
			std::printf("; first \"%.*s\" = (%lld, %lld",
			            (int)e.name.size(), e.name.data(),
			            (long long)SslmKeyedConstants::Value(e, 0), (long long)SslmKeyedConstants::Value(e, 1));
			if (e.value_words == 3) std::printf(", %lld", (long long)SslmKeyedConstants::Value(e, 2));
			std::printf(")");
		}
		std::printf("\n");
		return true;
	}
	if (s.type == SslmSectionType::Config) {
		SslmModelConfig c;
		SslmModelStatus st = ParseConfig(s, c, &err);
		if (st != SslmModelStatus::Ok) {
			std::printf("      ModelView REJECTED: %s — %s\n", SslmModelStatusName(st), err.c_str());
			return false;
		}
		std::printf("      -> hidden=%u layers=%u heads=%u/%u head_dim=%u inter=%u vocab=%u ctx=%u tie=%d kvp=%u theta=%g\n",
		            c.hidden_size, c.num_hidden_layers, c.num_attention_heads, c.num_key_value_heads,
		            c.head_dim, c.intermediate_size, c.vocab_size, c.context_cap,
		            (int)c.tie_word_embeddings, (unsigned)c.kv_precision, c.rope_theta);
		return true;
	}
	return true;  // not a model section this tool interprets
}

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
	bool all_ok = true;
	for (const auto& s : art.Sections()) {
		std::printf("  %-22s dtype=%u  bytes=%llu  elems=%llu  align=%u\n",
		            TypeName(s.type), (unsigned)s.dtype,
		            (unsigned long long)s.byte_size, (unsigned long long)s.elem_count,
		            s.alignment);
		if (!InspectModelSection(s)) all_ok = false;
	}
	std::printf(all_ok ? "MODELVIEW OK — every model section parsed\n"
	                   : "MODELVIEW FAILED — a section did not parse\n");
	return all_ok ? 0 : 1;
}

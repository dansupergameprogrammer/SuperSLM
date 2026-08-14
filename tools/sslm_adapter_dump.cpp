// sslm_adapter_dump -- T-2046 round-trip proof tool (design §24.2 D-SLM3162's own write->load
// round-trip contract, closing Poirot c81e48c review Significant 6).
//
// Loads an adapter `.sslm` artifact through the REAL engine loader (the same
// SslmDeltaFoldScaleView::Parse/SslmUFoldScaleView::Parse/SslmTensorManifest::Parse B0b's own
// unit-tested typed path already exercises, T-2021/T-2029) and dumps every DeltaFoldScales/
// UFoldScales entry's (identity,mult,exponent) rows and every Weights tensor's int8 codes as
// JSON, so the Python writer's own derivation can be compared against what the loader ACTUALLY
// accepted -- bit-for-bit, by execution, not by re-reading the writer's own intent.
//
// Usage: sslm_adapter_dump <artifact.sslm> <out.json>
//
// Follows sslm_verify.cpp's own documented, discovered-the-hard-way discipline: SslmModel::Load's
// returned SslmModelView has pointer fields that dangle the instant Load returns (Load
// constructs and destroys its own internal SslmArtifact) -- Load is invoked here ONLY to prove/
// report its Ok status, in its own scoped block, and every section this tool actually dumps is
// re-parsed directly from a separate, long-lived SslmArtifact object (SslmArtifact::OpenFromFile),
// exactly as sslm_verify.cpp's own BuildProofManifestJson does.

#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>

#include "superslm/artifact.h"
#include "superslm/model.h"

using namespace superslm;

namespace {

std::string JsonEscape(std::string_view s) {
	std::string out;
	out.reserve(s.size());
	for (char c : s) {
		if (c == '"' || c == '\\') out += '\\';
		out += c;
	}
	return out;
}

// Dumps one DeltaFoldScales/UFoldScales-shaped view's every entry as a JSON object:
// {"layer0.q_proj": [[identity,mult,exponent], ...], ...}
template <typename View>
void DumpFoldEntries(std::ostringstream& out, const View& view) {
	out << "{";
	bool first_entry = true;
	for (const auto& entry : view.Entries()) {
		if (!first_entry) out << ",";
		first_entry = false;
		out << "\n    \"" << JsonEscape(entry.name) << "\": [";
		for (uint64_t row = 0; row < entry.row_count; ++row) {
			if (row != 0) out << ", ";
			out << "[" << View::Identity(entry, row) << ", " << View::Mult(entry, row) << ", "
			    << View::Exponent(entry, row) << "]";
		}
		out << "]";
	}
	out << (first_entry ? "}" : "\n  }");
}

// Dumps a Weights-shaped tensor manifest's every int8 tensor as a JSON object:
// {"layer0.q_proj.lora_A": [1, -2, ...], ...}
void DumpWeightTensors(std::ostringstream& out, const SslmTensorManifest& manifest) {
	out << "{";
	bool first_entry = true;
	for (const auto& tensor : manifest.Tensors()) {
		if (!first_entry) out << ",";
		first_entry = false;
		out << "\n    \"" << JsonEscape(tensor.name) << "\": [";
		const auto* codes = reinterpret_cast<const int8_t*>(tensor.data);
		for (uint64_t i = 0; i < tensor.elem_count; ++i) {
			if (i != 0) out << ", ";
			out << static_cast<int>(codes[i]);
		}
		out << "]";
	}
	out << (first_entry ? "}" : "\n  }");
}

}  // namespace

int main(int argc, char** argv) {
	if (argc != 3) {
		std::fprintf(stderr, "usage: sslm_adapter_dump <artifact.sslm> <out.json>\n");
		return 2;
	}
	const char* artifact_path = argv[1];
	const char* out_path = argv[2];

	// The long-lived artifact object: every section dumped below is re-parsed from THIS object,
	// never from a Load() call's own (separately constructed, separately destroyed) internal one.
	SslmArtifact artifact;
	SslmError aerr;
	if (SslmArtifact::OpenFromFile(artifact_path, artifact, &aerr) != SslmStatus::Ok) {
		std::fprintf(stderr, "REJECTED (container): %s -- %s\n", SslmStatusName(aerr.code), aerr.message.c_str());
		return 1;
	}

	// Discharge the "must load Ok" contract by invoking the REAL SslmModel::Load -- only its
	// STATUS is read; its view's pointer fields are never touched (see this file's own header).
	SslmModelStatus load_status;
	std::string load_err;
	{
		std::ifstream f(artifact_path, std::ios::binary);
		std::string bytes((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
		SslmModelView view;
		load_status = SslmModel::Load(reinterpret_cast<const uint8_t*>(bytes.data()), bytes.size(), view, &load_err);
	}
	if (load_status != SslmModelStatus::Ok) {
		std::fprintf(stderr, "REJECTED (model): %s -- %s\n", SslmModelStatusName(load_status), load_err.c_str());
		return 1;
	}

	std::ostringstream out;
	out << "{\n  \"load_status\": \"Ok\",\n";

	const SslmSectionView* dfs1_section = artifact.Section(SslmSectionType::DeltaFoldScales);
	SslmDeltaFoldScaleView dfs1_view;
	std::string dfs1_err;
	const bool has_delta_fold_scales =
	    dfs1_section != nullptr && SslmDeltaFoldScaleView::Parse(*dfs1_section, dfs1_view, &dfs1_err) == SslmModelStatus::Ok;
	out << "  \"has_delta_fold_scales\": " << (has_delta_fold_scales ? "true" : "false") << ",\n";
	out << "  \"delta_fold_scales\": ";
	if (has_delta_fold_scales) DumpFoldEntries(out, dfs1_view); else out << "{}";
	out << ",\n";

	const SslmSectionView* ufs1_section = artifact.Section(SslmSectionType::UFoldScales);
	SslmUFoldScaleView ufs1_view;
	std::string ufs1_err;
	const bool has_u_fold_scales =
	    ufs1_section != nullptr && SslmUFoldScaleView::Parse(*ufs1_section, ufs1_view, &ufs1_err) == SslmModelStatus::Ok;
	out << "  \"has_u_fold_scales\": " << (has_u_fold_scales ? "true" : "false") << ",\n";
	out << "  \"u_fold_scales\": ";
	if (has_u_fold_scales) DumpFoldEntries(out, ufs1_view); else out << "{}";
	out << ",\n";

	const SslmSectionView* weights_section = artifact.Section(SslmSectionType::Weights);
	SslmTensorManifest weights_manifest;
	std::string weights_err;
	const bool has_weights =
	    weights_section != nullptr && SslmTensorManifest::Parse(*weights_section, weights_manifest, &weights_err) == SslmModelStatus::Ok;
	out << "  \"has_weights\": " << (has_weights ? "true" : "false") << ",\n";
	out << "  \"weights\": ";
	if (has_weights) DumpWeightTensors(out, weights_manifest); else out << "{}";
	out << "\n}\n";

	std::ofstream f(out_path, std::ios::binary);
	f << out.str();
	f.close();
	if (!f) {
		std::fprintf(stderr, "REJECTED: failed to write %s\n", out_path);
		return 1;
	}

	std::printf("OK  artifact=%s  dump=%s  fingerprint=%s  has_delta_fold_scales=%d "
	            "has_u_fold_scales=%d has_weights=%d\n",
	            artifact_path, out_path, artifact.FingerprintHex().c_str(),
	            has_delta_fold_scales, has_u_fold_scales, has_weights);
	return 0;
}

// sslm_verify -- the §13 item 7 independent converter verifier (S-HARDEN-3, F13).
//
// Loads a `.sslm` artifact through SslmModel::Load (the same entry point every
// runtime consumer uses -- proving the writer's "must load Ok" contract by
// INVOKING the loader, not by asserting it) and writes a proof manifest built
// entirely from what the loader accepted. Exits nonzero on any rejection --
// container-level (SslmArtifact) or the schema-value gate (D-SLM141) -- or if
// the independent config-geometry cross-check (§17.3 cell 4) disagrees with the
// artifact's own declared Config.
//
// Usage: sslm_verify <artifact.sslm> <manifest_out.json>
//
// convert_model.py invokes this after every write and merges the resulting JSON
// with the calibration-side fields only Python can supply (source hashes,
// converter/reference commits) into the combined proof manifest. This binary
// never sees the calibration pipeline's float arrays -- it reads only the bytes
// that were actually written, which is the whole point of the independence.
#include <cstdio>
#include <fstream>
#include <string>

#include "superslm/artifact.h"
#include "superslm/model.h"
#include "superslm/proof_manifest.h"

using namespace superslm;

int main(int argc, char** argv) {
	if (argc < 3) {
		std::fprintf(stderr, "usage: sslm_verify <artifact.sslm> <manifest_out.json>\n");
		return 2;
	}
	const char* artifact_path = argv[1];
	const char* manifest_path = argv[2];

	SslmModelView view;
	std::string err;
	const SslmModelStatus status = [&] {
		SslmArtifact art;
		SslmError aerr;
		if (SslmArtifact::OpenFromFile(artifact_path, art, &aerr) != SslmStatus::Ok) {
			err = std::string("artifact rejected before Load (") + SslmStatusName(aerr.code) + "): " + aerr.message;
			return SslmModelStatus::ArtifactRejected;
		}
		// Re-open via Load's own entry point (it re-parses from bytes rather than
		// reusing `art`, matching how every real consumer calls it) using the raw
		// file bytes so Load's OpenFromMemory path is what is actually exercised.
		std::ifstream f(artifact_path, std::ios::binary);
		std::string bytes((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
		return SslmModel::Load(reinterpret_cast<const uint8_t*>(bytes.data()), bytes.size(), view, &err);
	}();

	if (status != SslmModelStatus::Ok) {
		std::ofstream out(manifest_path, std::ios::binary);
		out << "{\n  \"schema\": \"sslm_proof_manifest_v1\",\n  \"status\": \"REJECTED\",\n  \"reject_status\": \""
		    << SslmModelStatusName(status) << "\",\n  \"diagnostic\": \"" << err << "\"\n}\n";
		std::fprintf(stderr, "REJECTED: %s -- %s\n", SslmModelStatusName(status), err.c_str());
		return 1;
	}

	SslmArtifact artifact;
	SslmError aerr;
	SslmArtifact::OpenFromFile(artifact_path, artifact, &aerr);  // already proven Ok above

	const std::string manifest = BuildProofManifestJson(artifact, view);

	// The independent geometry cross-check is also this tool's own pass/fail
	// verdict, not only a manifest field -- a manifest nobody's exit code
	// depends on is the correlated-oracle failure this whole family exists to
	// close (§17.3's own preamble).
	bool geometry_ok = true;
	if (view.has_config) {
		const auto g = CheckConfigGeometry(view.config.hidden_size, view.config.num_attention_heads,
		                                   view.config.num_key_value_heads, view.config.head_dim);
		geometry_ok = (g.status == ConfigGeometryStatus::Ok);
		if (!geometry_ok) {
			std::fprintf(stderr, "GEOMETRY REJECTED: %s -- %s\n", ConfigGeometryStatusName(g.status),
			             g.diagnostic.c_str());
		}
	}

	std::ofstream out(manifest_path, std::ios::binary);
	out << manifest;
	out.close();

	std::printf("OK  artifact=%s  manifest=%s  fingerprint=%s\n", artifact_path, manifest_path,
	            artifact.FingerprintHex().c_str());
	return geometry_ok ? 0 : 1;
}

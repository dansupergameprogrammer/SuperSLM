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
//
// Load's returned SslmModelView is used ONLY for its Ok/rejected STATUS below, never for a
// pointer-bearing field (SslmTensorView::data, SslmConstantEntry::values, etc.). CORRECTED (T-2104,
// Poirot 8e07d0c review, Significant 2): the claim this comment used to make here -- that Load's own
// view dangles the instant Load returns because it "constructs and destroys its own internal
// SslmArtifact" -- is false at source. `SslmModelView`'s own `backing_` member OWNS the file bytes
// every pointer in the view points into (include/superslm/model.h); the view stays valid for its
// own lifetime, exactly as long as any other owning object. This tool still re-parses from its own
// long-lived `artifact` object below, but the reason is simply that `SslmModel::Load` takes raw
// bytes rather than an existing `SslmArtifact` and this tool wants ONE artifact object serving both
// `BuildProofManifestJson` and the config-geometry cross-check above -- never "the view would
// dangle." Every manifest field below is derived by BuildProofManifestJson reading from THIS
// function's own long-lived `artifact` object, a structural choice, not a lifetime requirement.
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

	// The long-lived artifact object: everything BuildProofManifestJson
	// reports below is re-parsed from THIS object, not from Load's
	// (separately constructed, separately destroyed) internal one.
	SslmArtifact artifact;
	SslmError aerr;
	if (SslmArtifact::OpenFromFile(artifact_path, artifact, &aerr) != SslmStatus::Ok) {
		std::ofstream out(manifest_path, std::ios::binary);
		out << "{\n  \"schema\": \"sslm_proof_manifest_v1\",\n  \"status\": \"REJECTED\",\n"
		    << "  \"reject_status\": \"ArtifactRejected\",\n  \"diagnostic\": \""
		    << JsonEscape(SslmStatusName(aerr.code)) << ": " << JsonEscape(aerr.message) << "\"\n}\n";
		std::fprintf(stderr, "REJECTED: %s -- %s\n", SslmStatusName(aerr.code), aerr.message.c_str());
		return 1;
	}

	// Discharge the "must load Ok" contract by invoking Load -- the returned
	// `view`'s STATUS is what matters here; its pointer fields are never
	// touched (see this file's header comment).
	{
		std::ifstream f(artifact_path, std::ios::binary);
		std::string bytes((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
		SslmModelView view;
		std::string load_err;
		const SslmModelStatus status =
		    SslmModel::Load(reinterpret_cast<const uint8_t*>(bytes.data()), bytes.size(), view, &load_err);
		if (status != SslmModelStatus::Ok) {
			std::ofstream out(manifest_path, std::ios::binary);
			out << "{\n  \"schema\": \"sslm_proof_manifest_v1\",\n  \"status\": \"REJECTED\",\n  \"reject_status\": \""
			    << JsonEscape(SslmModelStatusName(status)) << "\",\n  \"diagnostic\": \"" << JsonEscape(load_err)
			    << "\"\n}\n";
			std::fprintf(stderr, "REJECTED: %s -- %s\n", SslmModelStatusName(status), load_err.c_str());
			return 1;
		}
	}

	const std::string manifest = BuildProofManifestJson(artifact);

	// The independent geometry cross-check is also this tool's own pass/fail
	// verdict, not only a manifest field -- a manifest nobody's exit code
	// depends on is the correlated-oracle failure this whole family exists to
	// close (§17.3's own preamble). Re-parsed fresh here too, for the same
	// reason BuildProofManifestJson does: never trust a pointer-bearing (or,
	// here, even a plain-value) field of a view returned by a Load call whose
	// internal artifact has already been destroyed.
	bool geometry_ok = true;
	const SslmSectionView* config_section = artifact.Section(SslmSectionType::Config);
	if (config_section != nullptr) {
		SslmModelConfig cfg;
		std::string cfg_err;
		if (ParseConfig(*config_section, cfg, &cfg_err) == SslmModelStatus::Ok) {
			const auto g = CheckConfigGeometry(cfg.hidden_size, cfg.num_attention_heads,
			                                   cfg.num_key_value_heads, cfg.head_dim);
			geometry_ok = (g.status == ConfigGeometryStatus::Ok);
			if (!geometry_ok) {
				std::fprintf(stderr, "GEOMETRY REJECTED: %s -- %s\n", ConfigGeometryStatusName(g.status),
				             g.diagnostic.c_str());
			}
		}
	}

	std::ofstream out(manifest_path, std::ios::binary);
	out << manifest;
	out.close();
	// T-1457: the manifest is this tool's product on the OK path (the caller,
	// sslm_convert_manifest.py's run_verifier, reads it back and json.load()s
	// it) -- a failed write (bad path, full disk) must not still print OK and
	// exit 0, which is what happened here before this check existed.
	if (!out) {
		std::fprintf(stderr, "REJECTED: failed to write manifest to %s\n", manifest_path);
		return 1;
	}

	std::printf("OK  artifact=%s  manifest=%s  fingerprint=%s\n", artifact_path, manifest_path,
	            artifact.FingerprintHex().c_str());
	return geometry_ok ? 0 : 1;
}

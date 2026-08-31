#include "superslm/proof_manifest.h"

#include <algorithm>
#include <cstdio>

#include "superslm/sha256.h"

#include "bad_alloc_wrap.h"

namespace superslm {

namespace {

// JSON string escaping -- the manifest carries tensor and section names, which
// are UTF-8 but otherwise untrusted; escape the ASCII control set and the two
// structural characters so the emitted document is valid JSON regardless of
// what a hostile-but-Load-accepted name contains. Declared in the header (not
// file-local) so tools/sslm_verify.cpp's hand-assembled REJECTED-path
// manifests route through this one implementation too (T-1449). Renamed to
// *Impl and wrapped below (S-HARDEN-7, F5, T-1475) -- promoting it out of
// this anonymous namespace into the header made it a member of the "throws
// only std::bad_alloc" contract's derived population (it allocates via
// `out.reserve`/`out +=`), so it follows the same rename-and-wrap shape as
// every other member in this file.
std::string JsonEscapeImpl(std::string_view s) {
	internal::MaybeThrowInjectedBadAllocFault();
	std::string out;
	out.reserve(s.size() + 2);
	for (unsigned char c : s) {
		switch (c) {
			case '"': out += "\\\""; break;
			case '\\': out += "\\\\"; break;
			case '\n': out += "\\n"; break;
			case '\r': out += "\\r"; break;
			case '\t': out += "\\t"; break;
			default:
				if (c < 0x20) {
					char buf[8];
					std::snprintf(buf, sizeof(buf), "\\u%04x", c);
					out += buf;
				} else {
					out += static_cast<char>(c);
				}
		}
	}
	return out;
}

}  // namespace

// S-HARDEN-7: wraps JsonEscapeImpl above with the shared catch-and-rethrow
// helper (src/bad_alloc_wrap.h), matching every other member of this file's
// derived population (T-1475).
std::string JsonEscape(std::string_view s) {
	return internal::WrapBadAllocContract([&] { return JsonEscapeImpl(s); });
}

namespace {

int32_t RdI32(const uint8_t* p) noexcept {
	uint32_t v = 0;
	for (int i = 0; i < 4; ++i) v |= static_cast<uint32_t>(p[i]) << (8 * i);
	return static_cast<int32_t>(v);
}

int64_t RdI64(const uint8_t* p) noexcept {
	uint64_t v = 0;
	for (int i = 0; i < 8; ++i) v |= static_cast<uint64_t>(p[i]) << (8 * i);
	return static_cast<int64_t>(v);
}

}  // namespace

const char* ConfigGeometryStatusName(ConfigGeometryStatus s) noexcept {
	// T-2367 (Brunel), D-SLM4359: restructured from a switch to direct
	// conditional branches -- one of the seven switch-jump-table symbols
	// D-SLM4359 rules restructured (D-SLM4988's own measured "seven", this
	// ticket's own build). Behaviourally identical: same six named statuses
	// map to their own string, every other value falls through to "?".
	if (s == ConfigGeometryStatus::Ok) return "Ok";
	if (s == ConfigGeometryStatus::ZeroAttentionHeads) return "ZeroAttentionHeads";
	if (s == ConfigGeometryStatus::ZeroKeyValueHeads) return "ZeroKeyValueHeads";
	if (s == ConfigGeometryStatus::KvHeadsExceedsHeads) return "KvHeadsExceedsHeads";
	if (s == ConfigGeometryStatus::HeadsNotDivisibleByKv) return "HeadsNotDivisibleByKv";
	if (s == ConfigGeometryStatus::HiddenSizeGeometryMismatch) return "HiddenSizeGeometryMismatch";
	return "?";
}

ConfigGeometryResult CheckConfigGeometry(uint32_t hidden_size, uint32_t num_attention_heads,
                                         uint32_t num_key_value_heads, uint32_t head_dim) noexcept {
	ConfigGeometryResult r;

	// Zero-boundary FIRST, and each checked independently -- neither may be
	// inferred from the other. This is the exact ordering the coverage audit's
	// §3.3 zero-boundary note requires: `heads % kv_heads` below would fault
	// (division by zero) if reached with kv_heads == 0, so both zero cases are
	// rejected with a named diagnostic before any modulus is evaluated.
	if (num_attention_heads == 0) {
		r.status = ConfigGeometryStatus::ZeroAttentionHeads;
		r.diagnostic = "num_attention_heads == 0";
		return r;
	}
	if (num_key_value_heads == 0) {
		r.status = ConfigGeometryStatus::ZeroKeyValueHeads;
		r.diagnostic = "num_key_value_heads == 0";
		return r;
	}

	if (num_key_value_heads > num_attention_heads) {
		r.status = ConfigGeometryStatus::KvHeadsExceedsHeads;
		r.diagnostic = "num_key_value_heads (" + std::to_string(num_key_value_heads) +
		               ") > num_attention_heads (" + std::to_string(num_attention_heads) + ")";
		return r;
	}
	if (num_attention_heads % num_key_value_heads != 0) {
		r.status = ConfigGeometryStatus::HeadsNotDivisibleByKv;
		r.diagnostic = "num_attention_heads (" + std::to_string(num_attention_heads) +
		               ") % num_key_value_heads (" + std::to_string(num_key_value_heads) + ") != 0";
		return r;
	}

	// SSLM-GEOMETRY-SITE: GS-01
	// T-2432 (Track A step 1, design §6 Track A step 1, D-SLM5273): the R1 identity
	// hidden_size == num_attention_heads * head_dim is no longer enforced here -- removed,
	// not loosened. q_width (num_attention_heads * head_dim) is threaded independently of
	// hidden_size once the forward path decouples Q/O's own width (§2.1). hidden_size and
	// head_dim are retained as parameters (unused past this point) so this function's
	// signature is unchanged, per the design's own §2.1 framing of this as a widening of
	// the existing check, not a new one. Confirmed correct by execution against the real
	// Qwen3-Embedding-0.6B candidate, T-2423's spike (D-SLM5291).
	(void)hidden_size;
	(void)head_dim;

	r.status = ConfigGeometryStatus::Ok;
	return r;
}

namespace {
std::vector<TensorEvidence> ComputeTensorEvidenceImpl(const SslmTensorManifest& manifest, SslmDtype dtype) {
	internal::MaybeThrowInjectedBadAllocFault();
	std::vector<TensorEvidence> out;
	out.reserve(manifest.Tensors().size());

	int64_t dtype_min = 0, dtype_max = 0;
	int elem_size = 1;
	switch (dtype) {
		case SslmDtype::Int8: dtype_min = -128; dtype_max = 127; elem_size = 1; break;
		case SslmDtype::Int32: dtype_min = INT32_MIN; dtype_max = INT32_MAX; elem_size = 4; break;
		case SslmDtype::Int64: dtype_min = INT64_MIN; dtype_max = INT64_MAX; elem_size = 8; break;
		default: elem_size = 1; break;  // unsupported dtype for this evidence kind: emits an empty read
	}

	for (const SslmTensorView& t : manifest.Tensors()) {
		TensorEvidence e;
		e.name = std::string(t.name);
		e.elem_count = t.elem_count;
		if (t.elem_count == 0 || t.data == nullptr) {
			out.push_back(e);
			continue;
		}
		int64_t mn = INT64_MAX, mx = INT64_MIN;
		uint64_t lo_count = 0, hi_count = 0;
		for (uint64_t i = 0; i < t.elem_count; ++i) {
			int64_t v = 0;
			switch (elem_size) {
				case 1: v = static_cast<int8_t>(t.data[i]); break;
				case 4: v = RdI32(t.data + i * 4); break;
				case 8: v = RdI64(t.data + i * 8); break;
			}
			mn = std::min(mn, v);
			mx = std::max(mx, v);
			if (v == dtype_min) ++lo_count;
			if (v == dtype_max) ++hi_count;
		}
		e.min_value = mn;
		e.max_value = mx;
		e.saturation_lo_count = lo_count;
		e.saturation_hi_count = hi_count;
		out.push_back(e);
	}
	return out;
}

std::vector<WeightScaleEvidence> ComputeWeightScaleEvidenceImpl(const SslmTensorManifest& weight_scales) {
	internal::MaybeThrowInjectedBadAllocFault();
	std::vector<WeightScaleEvidence> out;
	out.reserve(weight_scales.Tensors().size());
	for (const SslmTensorView& t : weight_scales.Tensors()) {
		WeightScaleEvidence e;
		e.tensor_name = std::string(t.name);
		const uint64_t rows = t.elem_count / 3;
		e.row_count = rows;
		if (rows == 0 || t.data == nullptr) {
			out.push_back(e);
			continue;
		}
		int32_t mn = INT32_MAX, mx = INT32_MIN;
		uint64_t identity_count = 0;
		for (uint64_t r = 0; r < rows; ++r) {
			const int32_t identity = RdI32(t.data + (r * 3 + 0) * 4);
			const int32_t shift = RdI32(t.data + (r * 3 + 2) * 4);
			mn = std::min(mn, shift);
			mx = std::max(mx, shift);
			if (identity == 1) ++identity_count;
		}
		e.shift_min = mn;
		e.shift_max = mx;
		e.identity_count = identity_count;
		out.push_back(e);
	}
	return out;
}

std::string HashSectionHexImpl(const SslmSectionView& section) {
	internal::MaybeThrowInjectedBadAllocFault();
	uint8_t digest[32];
	Sha256Hash(section.data, static_cast<size_t>(section.byte_size), digest);
	return ToHex(digest);
}
}  // namespace

// S-HARDEN-7: today's bodies, renamed to *Impl above; each wraps its Impl
// with the shared catch-and-rethrow helper (src/bad_alloc_wrap.h).
std::vector<TensorEvidence> ComputeTensorEvidence(const SslmTensorManifest& manifest, SslmDtype dtype) {
	return internal::WrapBadAllocContract([&] { return ComputeTensorEvidenceImpl(manifest, dtype); });
}

std::vector<WeightScaleEvidence> ComputeWeightScaleEvidence(const SslmTensorManifest& weight_scales) {
	return internal::WrapBadAllocContract([&] { return ComputeWeightScaleEvidenceImpl(weight_scales); });
}

std::string HashSectionHex(const SslmSectionView& section) {
	return internal::WrapBadAllocContract([&] { return HashSectionHexImpl(section); });
}

namespace {

const char* SectionTypeName(SslmSectionType t) {
	// T-2367 (Brunel), D-SLM4359: restructured from a switch to direct
	// conditional branches. This function is called from exactly one site
	// (BuildProofManifestJsonImpl's own section loop, below) and MSVC
	// inlines it there at /O2 -- its own switch's jump-table dispatch was
	// compiled directly into BuildProofManifestJsonImpl's own extent, which
	// is why THAT symbol (never this one) carried the indirect jmp check
	// (C) rejected (D-SLM4982, confirmed by disassembly this ticket's own
	// session).
	//
	// T-2371 (Brunel), D-SLM5017/D-SLM5021: at the time this switch was
	// authored, SslmSectionType had exactly the seventeen enumerators named
	// below and this was genuinely exhaustive (no default, every
	// enumerator a case label). Four enumerators were added to the header
	// afterward (CalibrationBand, DeltaFoldScales, UFoldScales,
	// DampedGreedyConstants) without a matching arm here, which the
	// original switch's own `-Wswitch` guard should have caught and did
	// not catch before this restructuring ever touched the function --
	// this was already a real gap, not one the restructure introduced, and
	// it meant a proof manifest naming one of those four legitimate
	// section types printed "Unknown" for it. Closed here by naming all
	// twenty-one: same seventeen types return the string they always did,
	// the four that used to fall through now return their own name, and
	// only a value outside the enum entirely still falls through to
	// "Unknown".
	if (t == SslmSectionType::Config) return "Config";
	if (t == SslmSectionType::Provenance) return "Provenance";
	if (t == SslmSectionType::Weights) return "Weights";
	if (t == SslmSectionType::Biases) return "Biases";
	if (t == SslmSectionType::RopeTables) return "RopeTables";
	if (t == SslmSectionType::Scales) return "Scales";
	if (t == SslmSectionType::WeightScales) return "WeightScales";
	if (t == SslmSectionType::CompositionConstants) return "CompositionConstants";
	if (t == SslmSectionType::KvLandingScales) return "KvLandingScales";
	if (t == SslmSectionType::KvLandingReciprocals) return "KvLandingReciprocals";
	if (t == SslmSectionType::Calibration) return "Calibration";
	if (t == SslmSectionType::GoldenHashes) return "GoldenHashes";
	if (t == SslmSectionType::SigmoidLut) return "SigmoidLut";
	if (t == SslmSectionType::Tokenizer) return "Tokenizer";
	if (t == SslmSectionType::ChatTemplate) return "ChatTemplate";
	if (t == SslmSectionType::UnicodeTables) return "UnicodeTables";
	if (t == SslmSectionType::SchemaMasks) return "SchemaMasks";
	if (t == SslmSectionType::CalibrationBand) return "CalibrationBand";
	if (t == SslmSectionType::DeltaFoldScales) return "DeltaFoldScales";
	if (t == SslmSectionType::UFoldScales) return "UFoldScales";
	if (t == SslmSectionType::DampedGreedyConstants) return "DampedGreedyConstants";
	return "Unknown";
}

void AppendTensorEvidenceArray(std::string& out, const std::vector<TensorEvidence>& evs) {
	out += "[";
	for (size_t i = 0; i < evs.size(); ++i) {
		if (i) out += ",";
		const auto& e = evs[i];
		out += "{\"name\":\"" + JsonEscape(e.name) + "\",";
		out += "\"elem_count\":" + std::to_string(e.elem_count) + ",";
		out += "\"min\":" + std::to_string(e.min_value) + ",";
		out += "\"max\":" + std::to_string(e.max_value) + ",";
		out += "\"saturation_lo_count\":" + std::to_string(e.saturation_lo_count) + ",";
		out += "\"saturation_hi_count\":" + std::to_string(e.saturation_hi_count) + "}";
	}
	out += "]";
}

}  // namespace

namespace {

// Re-parses `type`'s section fresh, from `artifact` (long-lived for the
// duration of BuildProofManifestJson's call), into `out`. Returns false
// (leaving `out` empty) if the section is absent or fails its own structural
// sub-parse -- both are reported as an empty evidence array, matching how
// the rest of this manifest degrades gracefully on an absent section rather
// than treating it as an error (the caller has already confirmed the whole
// artifact loads Ok via a separate SslmModel::Load call before this
// function is ever invoked).
bool TryParseTensorManifest(const SslmArtifact& artifact, SslmSectionType type, SslmTensorManifest& out) {
	const SslmSectionView* section = artifact.Section(type);
	if (section == nullptr) return false;
	std::string err;
	return SslmTensorManifest::Parse(*section, out, &err) == SslmModelStatus::Ok;
}

}  // namespace

namespace {
std::string BuildProofManifestJsonImpl(const SslmArtifact& artifact) {
	internal::MaybeThrowInjectedBadAllocFault();
	std::string out;
	out += "{\n";
	out += "  \"schema\": \"sslm_proof_manifest_v1\",\n";
	out += "  \"format_version\": " + std::to_string(artifact.FormatVersion()) + ",\n";
	out += "  \"file_bytes\": " + std::to_string(artifact.FileBytes()) + ",\n";
	out += "  \"artifact_hash\": \"" + artifact.FingerprintHex() + "\",\n";

	// The geometry cross-check (§17.3 cell 4) -- Config is re-parsed HERE, fresh, from `artifact`
	// (long-lived). CORRECTED (T-2104, Poirot 8e07d0c7/7a0b6426 review, Significant 5): this
	// function's own signature (above) takes only an `SslmArtifact&`, so there IS no previously-
	// populated `SslmModelView` in scope to read Config from in the first place -- the re-parse is
	// not routing around a lifetime hazard (see this function's own header comment for the full
	// correction: `SslmModelView` does not dangle; nothing in `SslmModel::Load` needed routing
	// around). It is kept for an independent, real reason: every field this function reports is
	// sourced the same way, fresh from `artifact`, never from a second object of different
	// provenance -- worth the one cheap 84-byte re-parse on its own.
	const SslmSectionView* config_section = artifact.Section(SslmSectionType::Config);
	if (config_section != nullptr) {
		SslmModelConfig cfg;
		std::string err;
		if (ParseConfig(*config_section, cfg, &err) == SslmModelStatus::Ok) {
			const auto g = CheckConfigGeometry(cfg.hidden_size, cfg.num_attention_heads,
			                                   cfg.num_key_value_heads, cfg.head_dim);
			out += "  \"config_geometry\": {\"ok\": " +
			       std::string(g.status == ConfigGeometryStatus::Ok ? "true" : "false") + ", \"status\": \"" +
			       ConfigGeometryStatusName(g.status) + "\", \"diagnostic\": \"" + JsonEscape(g.diagnostic) +
			       "\"},\n";
		} else {
			out += "  \"config_geometry\": null,\n";
		}
	} else {
		out += "  \"config_geometry\": null,\n";
	}

	out += "  \"sections\": [\n";
	for (size_t i = 0; i < artifact.Sections().size(); ++i) {
		const auto& s = artifact.Sections()[i];
		out += "    {\"type\": \"" + std::string(SectionTypeName(s.type)) + "\", \"byte_size\": " +
		       std::to_string(s.byte_size) + ", \"elem_count\": " + std::to_string(s.elem_count) +
		       ", \"sha256\": \"" + HashSectionHex(s) + "\"}";
		out += (i + 1 < artifact.Sections().size()) ? ",\n" : "\n";
	}
	out += "  ],\n";

	SslmTensorManifest weights, biases, rope_tables, weight_scales;
	const bool has_weights = TryParseTensorManifest(artifact, SslmSectionType::Weights, weights);
	const bool has_biases = TryParseTensorManifest(artifact, SslmSectionType::Biases, biases);
	const bool has_rope = TryParseTensorManifest(artifact, SslmSectionType::RopeTables, rope_tables);
	const bool has_wsc = TryParseTensorManifest(artifact, SslmSectionType::WeightScales, weight_scales);

	out += "  \"weights_evidence\": ";
	AppendTensorEvidenceArray(out, has_weights ? ComputeTensorEvidence(weights, SslmDtype::Int8)
	                                           : std::vector<TensorEvidence>{});
	out += ",\n";

	out += "  \"biases_evidence\": ";
	AppendTensorEvidenceArray(out, has_biases ? ComputeTensorEvidence(biases, SslmDtype::Int64)
	                                          : std::vector<TensorEvidence>{});
	out += ",\n";

	out += "  \"rope_tables_evidence\": ";
	AppendTensorEvidenceArray(out, has_rope ? ComputeTensorEvidence(rope_tables, SslmDtype::Int64)
	                                        : std::vector<TensorEvidence>{});
	out += ",\n";

	out += "  \"weight_scales_evidence\": [";
	if (has_wsc) {
		const auto wse = ComputeWeightScaleEvidence(weight_scales);
		for (size_t i = 0; i < wse.size(); ++i) {
			if (i) out += ",";
			const auto& e = wse[i];
			out += "{\"tensor_name\":\"" + JsonEscape(e.tensor_name) + "\",";
			out += "\"row_count\":" + std::to_string(e.row_count) + ",";
			out += "\"shift_min\":" + std::to_string(e.shift_min) + ",";
			out += "\"shift_max\":" + std::to_string(e.shift_max) + ",";
			out += "\"identity_count\":" + std::to_string(e.identity_count) + "}";
		}
	}
	out += "]\n";

	out += "}\n";
	return out;
}
}  // namespace

// S-HARDEN-7: today's body, renamed to BuildProofManifestJsonImpl above;
// wraps it with the shared catch-and-rethrow helper (src/bad_alloc_wrap.h).
std::string BuildProofManifestJson(const SslmArtifact& artifact) {
	return internal::WrapBadAllocContract([&] { return BuildProofManifestJsonImpl(artifact); });
}

}  // namespace superslm

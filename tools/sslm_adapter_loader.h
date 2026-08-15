// sslm_adapter_loader.h -- T-2102 (RESUME-2026-08-14-gpu-throughput.md §4, D-SLM3304): the
// missing piece model.h:409 names out loud ("a future adapter-loader's own section-table walk").
// The converter (tools/sslm_convert_adapter.py) writes a standalone adapter `.sslm` carrying
// Config/SigmoidLut/Provenance(ADP1)/Weights(lora_A/lora_B)/DeltaFoldScales(DFS1)/
// UFoldScales(UFS1); `RunLayerLoop` already applies `LayerWeights::adapter` if a caller sets it
// (design Sec8/Sec11 B1a-B1c, T-2021/T-2029); nothing before this file ever set it from a loaded
// artifact. This is that section-table walk.
//
// Contract (T-2104, Poirot 8e07d0c review, Significant 2 -- corrected; the ORIGINAL text here
// claimed "SslmModel::Load returns a view whose pointers dangle the instant Load returns," which is
// false at source: SslmModelView's last member, `backing_`, OWNS the file bytes every pointer in the
// view points into (model.h), and the view stays valid for its own lifetime -- both this file's own
// production callers, sslm_generate.cpp and sslm_lora_switch_demo.cpp, read tensors off a model view
// long after Load returns and generate from them, which is only sound because the claim was wrong.
// The claim was inherited from tools/sslm_adapter_dump.cpp, corrected there too). `LoadAdapterArtifact`
// calls `SslmModel::Load` exactly ONCE and KEEPS the resulting view (`AdapterHandle::view_`) for the
// handle's own lifetime -- every pointer this loader hands back into LayerAdapter/LayerAdapterProjection
// is read directly from that one already-parsed view (its `weights`/`delta_fold_scales`/
// `u_fold_scales` members, and its own `Section()`/`RawIntegrityHash()` forwarders for the raw
// Provenance section and the base-hash check) rather than a second, independent re-parse.
//
// What this file adds beyond SslmModel::Load's existing gate -- design Sec9's four explicit B0b
// validations model.h declares and documents as "the checks the [as yet unbuilt] adapter-loading
// ABI will call" (AmplifyingFoldDimensionMismatch, AmplifyingFoldProjectionInvalid,
// AmplifyingFoldBaseHashMismatch, plus this file's own ADP1 payload parse and per-entry
// "layer<N>.<proj>" name resolution): this is the first caller that wires
// ValidateAmplifyingFoldDimension/ValidateAmplifyingFoldProjection/ValidateAmplifyingFoldBaseHash
// for real, against a converted artifact's own DeltaFoldScales/UFoldScales entries.
//
// Tools-only, matching sslm_marshal.h's own convention (never included from src/ or include/):
// the ABI-shape question ("sslm_adapter_map"/opaque handles, SuperSLM_Plan.md §12/§19) is a
// separate, later obligation this file does not build -- this is the section-table walk that ABI
// would call, exposed here as ordinary C++ so sslm_generate.cpp (and this suite's own tests) can
// use it today.
#ifndef SUPERSLM_TOOLS_SSLM_ADAPTER_LOADER_H
#define SUPERSLM_TOOLS_SSLM_ADAPTER_LOADER_H

#include <array>
#include <cstdint>
#include <cstring>
#include <iterator>
#include <string>
#include <string_view>
#include <vector>

#include "superslm/artifact.h"
#include "superslm/forward_sites.h"
#include "superslm/model.h"
#include "sslm_marshal.h"

namespace superslm_adapter {

using superslm_marshal::ReadFile;

// ADP1's 68-byte fixed prefix (tools/sslm_convert_adapter.py's write_adp1, design §24.2
// D-SLM3158): magic(4)+version(4)+rank(4)+target_modules_mask(4)+base_artifact_hash(32)+
// lora_alpha(8,f64)+use_rslora(4)+reserved(4)+source_adapter_name_len(4), then the name bytes.
inline constexpr size_t kAdp1FixedBytes = 68;
inline constexpr uint8_t kAdp1Magic[4] = {'A', 'D', 'P', '1'};
inline constexpr uint32_t kAdp1Version = 1;

// One parsed ADP1 payload.
struct AdapterMeta {
	uint32_t rank = 0;
	uint32_t target_modules_mask = 0;
	std::array<uint8_t, superslm::kIntegrityHashBytes> base_artifact_hash{};
	double lora_alpha = 0.0;
	bool use_rslora = false;
	std::string source_adapter_name;
};

// Every way LoadAdapterArtifact can fail. `Ok` is the only non-error value. Mirrors this
// project's established "reject with a specific, named code" convention (SslmStatus/
// SslmModelStatus) rather than a bare bool -- a caller (sslm_generate.cpp, this file's own
// tests) can distinguish "no such file" from "wrong base" from "artifact structurally broken"
// without parsing the diagnostic string.
enum class AdapterLoadStatus {
	Ok = 0,
	FileReadFailed,           // the adapter path could not be read
	ArtifactRejected,         // SslmModel::Load's own container-level rejection
	                          // (SslmModelStatus::ArtifactRejected -- the outer container failed
	                          // before any section was even parsed: bad magic, unsupported
	                          // version, truncation, integrity mismatch, etc.)
	ModelRejected,            // SslmModel::Load rejection deeper than the container (structural/
	                          // domain: WGT1/DFS1/UFS1/CFG1/SIL1)
	MissingProvenanceSection, // no Provenance section at all
	BadAdp1Size,              // Provenance section shorter than the 68-byte fixed prefix, or the
	                          // declared name length overruns it
	BadAdp1Magic,             // Provenance section's first 4 bytes are not "ADP1"
	BadAdp1Version,           // ADP1 version field != kAdp1Version
	BadAdp1Reserved,          // ADP1's reserved field != 0
	BaseHashMismatch,         // ADP1's declared base_artifact_hash != the base actually mapped
	                          // (ValidateAmplifyingFoldBaseHash)
	MissingWeightsSection,    // no Weights section (needed for lora_A/lora_B)
	MissingDeltaFoldSection,  // no DeltaFoldScales section
	MissingUFoldSection,      // no UFoldScales section
	UnparsableEntryName,      // a DeltaFoldScales/UFoldScales entry name is not "layer<N>.<proj>"
	LayerIndexOutOfRange,     // parsed layer index >= the base model's own num_hidden_layers
	ProjectionInvalid,        // not one of the seven PEFT-adaptable projections, or not claimed by
	                          // ADP1's own target_modules_mask (ValidateAmplifyingFoldProjection)
	DeltaDimensionMismatch,   // a DeltaFoldScales entry's row_count != this projection's out_channels
	                          // (ValidateAmplifyingFoldDimension)
	MissingUFoldEntry,        // DeltaFoldScales names a pair with no matching UFoldScales entry
	UFoldDimensionMismatch,   // a UFoldScales entry's row_count != ADP1's own declared rank
	MissingWeightTensor,      // a DeltaFoldScales-named pair has no matching "<name>.lora_A"/".lora_B"
	                          // Weights tensor
	InvalidTargetModulesMask, // ADP1's target_modules_mask sets a bit at or past
	                          // kPeftAdaptableProjections' own length -- an unknown target module
	                          // (T-2104 M2: silently discarding it would load an adapter as though
	                          // the unknown module had never been declared)
	LoraAWeightShapeMismatch, // T-2104 S1: a "<name>.lora_A" tensor's own rank/shape does not match
	                          // [ADP1's declared rank, this projection's in_channels] -- the FIRST
	                          // of the two pointers AddAmplifyingLoraDelta actually walks that this
	                          // loader had never checked
	LoraBWeightShapeMismatch, // T-2104 S1: a "<name>.lora_B" tensor's own rank/shape does not match
	                          // [this projection's out_channels, ADP1's declared rank] -- the SECOND
};

inline const char* AdapterLoadStatusName(AdapterLoadStatus s) noexcept {
	switch (s) {
		case AdapterLoadStatus::Ok: return "Ok";
		case AdapterLoadStatus::FileReadFailed: return "FileReadFailed";
		case AdapterLoadStatus::ArtifactRejected: return "ArtifactRejected";
		case AdapterLoadStatus::ModelRejected: return "ModelRejected";
		case AdapterLoadStatus::MissingProvenanceSection: return "MissingProvenanceSection";
		case AdapterLoadStatus::BadAdp1Size: return "BadAdp1Size";
		case AdapterLoadStatus::BadAdp1Magic: return "BadAdp1Magic";
		case AdapterLoadStatus::BadAdp1Version: return "BadAdp1Version";
		case AdapterLoadStatus::BadAdp1Reserved: return "BadAdp1Reserved";
		case AdapterLoadStatus::BaseHashMismatch: return "BaseHashMismatch";
		case AdapterLoadStatus::MissingWeightsSection: return "MissingWeightsSection";
		case AdapterLoadStatus::MissingDeltaFoldSection: return "MissingDeltaFoldSection";
		case AdapterLoadStatus::MissingUFoldSection: return "MissingUFoldSection";
		case AdapterLoadStatus::UnparsableEntryName: return "UnparsableEntryName";
		case AdapterLoadStatus::LayerIndexOutOfRange: return "LayerIndexOutOfRange";
		case AdapterLoadStatus::ProjectionInvalid: return "ProjectionInvalid";
		case AdapterLoadStatus::DeltaDimensionMismatch: return "DeltaDimensionMismatch";
		case AdapterLoadStatus::MissingUFoldEntry: return "MissingUFoldEntry";
		case AdapterLoadStatus::UFoldDimensionMismatch: return "UFoldDimensionMismatch";
		case AdapterLoadStatus::MissingWeightTensor: return "MissingWeightTensor";
		case AdapterLoadStatus::InvalidTargetModulesMask: return "InvalidTargetModulesMask";
		case AdapterLoadStatus::LoraAWeightShapeMismatch: return "LoraAWeightShapeMismatch";
		case AdapterLoadStatus::LoraBWeightShapeMismatch: return "LoraBWeightShapeMismatch";
	}
	return "Unknown";
}

// model.h:433's kPeftAdaptableProjections, in ADP1's OWN bit order (tools/sslm_convert_adapter.py
// PEFT_ADAPTABLE_PROJECTIONS, "the ONLY source of ADP1's target_modules_mask bit order") -- a
// second, independent copy of the ARRAY would risk drifting from model.h's, so this file reads
// model.h's own array directly. T-2104 (Poirot 8e07d0c review, Minor 2): the array's own LENGTH is
// still read via `std::size`, never restated as a literal -- the original `for (i < 7)` was exactly
// the restatement the header comment above claimed to avoid.
inline std::vector<std::string_view> TargetModulesFromMask(uint32_t mask) {
	std::vector<std::string_view> out;
	for (size_t i = 0; i < std::size(superslm::kPeftAdaptableProjections); ++i) {
		if (mask & (1u << i)) out.push_back(superslm::kPeftAdaptableProjections[i]);
	}
	return out;
}

// T-2104 (Poirot 8e07d0c review, Minor 2): true iff `mask` sets any bit at or past
// kPeftAdaptableProjections' own length -- an ADP1 declaring a target module this build does not
// know about. `TargetModulesFromMask` above silently drops such a bit (it can only ever push a
// name for a bit it recognizes), which -- unchecked -- would load an adapter exactly as though the
// unknown module had never been declared, rather than refusing it.
inline bool TargetModulesMaskHasUnknownBit(uint32_t mask) noexcept {
	constexpr size_t kKnown = std::size(superslm::kPeftAdaptableProjections);
	const uint32_t known_mask = kKnown >= 32 ? 0xFFFFFFFFu : ((1u << kKnown) - 1u);
	return (mask & ~known_mask) != 0;
}

// Parses "layer<N>.<proj>" (the converter's own naming convention, tools/sslm_convert_adapter.py
// `name = f"layer{layer}.{proj}"`) into a numeric layer index and the projection substring.
// Returns false (leaving `layer`/`proj` unspecified) if the name does not match that shape --
// no leading "layer", no digits, no ".", or a proj substring containing another '.'.
inline bool ParseLayerProjName(std::string_view name, uint32_t& layer, std::string& proj) {
	constexpr std::string_view kPrefix = "layer";
	if (name.size() <= kPrefix.size() || name.substr(0, kPrefix.size()) != kPrefix) return false;
	size_t i = kPrefix.size();
	size_t digits_start = i;
	while (i < name.size() && name[i] >= '0' && name[i] <= '9') ++i;
	if (i == digits_start) return false;  // no digits at all
	if (i >= name.size() || name[i] != '.') return false;
	std::string_view digits = name.substr(digits_start, i - digits_start);
	uint64_t v = 0;
	for (char c : digits) {
		v = v * 10 + static_cast<uint64_t>(c - '0');
		if (v > 0xFFFFFFFFull) return false;  // overflow guard -- never a plausible layer index anyway
	}
	std::string_view rest = name.substr(i + 1);
	if (rest.empty() || rest.find('.') != std::string_view::npos) return false;
	layer = static_cast<uint32_t>(v);
	proj = std::string(rest);
	return true;
}

// This projection's own out_channels, the SAME per-projection channel-count convention
// sslm_marshal.h's MarshalProjectionFold already applies to the base model's own WSC1 fold
// (hidden_size for q/o/down_proj, num_key_value_heads*head_dim for k/v_proj, intermediate_size
// for gate/up_proj) -- design Sec9 item (a)'s own "the caller reads the expectation from the base
// artifact's own geometry."
inline uint64_t AdapterOutChannelsFor(const std::string& proj, uint64_t hidden_size,
                                       uint64_t intermediate_size, uint64_t kv_hidden_size) {
	if (proj == "q_proj" || proj == "o_proj" || proj == "down_proj") return hidden_size;
	if (proj == "k_proj" || proj == "v_proj") return kv_hidden_size;
	if (proj == "gate_proj" || proj == "up_proj") return intermediate_size;
	return 0;  // unreachable once ValidateAmplifyingFoldProjection has already accepted `proj`
}

// T-2104 (Poirot 8e07d0c review, Significant 1): this projection's own IN-channels -- the sibling
// `AdapterOutChannelsFor` above never had. Read from the SAME call sites `AddAmplifyingLoraDelta`
// itself is invoked from (src/forward/forward_sites.cpp): `hidden_size` feeds q/o/gate/up/k/v_proj's
// own delta-add (the q/o/gate/up calls each pass `hidden_size` as `in_channels`; the k/v calls pass
// `normed.data(), hidden_size`), `intermediate_size` feeds down_proj's own (the down_proj call passes
// `act_codes.data(), intermediate_size`). This is what `lora_A`'s own declared shape must match --
// `GemmInt8AccumulateRow(in_codes, adapter->a_weight, in_channels, rank, ...)` reads
// `in_channels * rank` bytes from `a_weight` with a length that comes from THIS function's own
// return value and `LayerAdapter::rank`, never from the tensor actually indexed.
inline uint64_t AdapterInChannelsFor(const std::string& proj, uint64_t hidden_size,
                                      uint64_t intermediate_size) {
	if (proj == "down_proj") return intermediate_size;
	if (proj == "q_proj" || proj == "o_proj" || proj == "gate_proj" || proj == "up_proj" ||
	    proj == "k_proj" || proj == "v_proj") {
		return hidden_size;
	}
	return 0;  // unreachable once ValidateAmplifyingFoldProjection has already accepted `proj`
}

// Owns everything a bound adapter needs to stay alive: the ONE loaded, kept-alive model view
// (`view_` -- T-2104 S2: no second, redundant parse; every typed field this loader reads,
// `weights`/`delta_fold_scales`/`u_fold_scales`, plus the raw Provenance section and the base hash
// via `view_`'s own `Section()`/`RawIntegrityHash()` forwarders, comes from this ONE call to
// `SslmModel::Load`), and one `LayerAdapter` per BASE-MODEL layer (`layer_adapters`, index l ==
// base layer l's adapter state; default-constructed -- every LayerAdapterProjection::a_weight ==
// nullptr -- for a layer/projection this adapter does not cover, exactly LayerWeights' own
// documented B1c "adapter bound, but this projection is unadapted" case).
//
// Movable, non-copyable: every pointer this handle exposes (through layer_adapters) points into
// view_'s own owned byte buffer (its `backing_`), which only a move (not a copy) preserves the
// address of -- the SAME ownership convention SslmArtifact/SslmModelView already document and rely
// on elsewhere in this tree.
class AdapterHandle {
public:
	AdapterHandle() = default;
	AdapterHandle(const AdapterHandle&) = delete;
	AdapterHandle& operator=(const AdapterHandle&) = delete;
	AdapterHandle(AdapterHandle&&) = default;
	AdapterHandle& operator=(AdapterHandle&&) = default;

	AdapterMeta meta;
	std::vector<superslm::LayerAdapter> layer_adapters;  // size == base num_hidden_layers

	superslm::SslmModelView view_;
};

// The base model's geometry, exactly what AdapterOutChannelsFor/dimension validation and the
// base-hash check need -- separated from a full SslmModelView so a caller need not keep the
// base's (transient-pointer) view alive to load an adapter against it, matching how
// sslm_generate.cpp already threads config fields around independent of the view's lifetime.
struct BaseModelGeometry {
	uint32_t num_hidden_layers = 0;
	uint64_t hidden_size = 0;
	uint64_t intermediate_size = 0;
	uint64_t kv_hidden_size = 0;  // num_key_value_heads * head_dim
	std::array<uint8_t, superslm::kIntegrityHashBytes> base_artifact_hash{};
};

// Loads the adapter `.sslm` at `path`, validates it against `base` (the base-hash check, design
// Sec9 item (d)), and populates `out` -- one `LayerAdapter` per base layer, wired for exactly the
// (layer, projection) pairs the adapter's own DeltaFoldScales/UFoldScales/Weights sections cover.
// Returns Ok and a fully populated `out` on success; on ANY rejection returns the specific status
// and `out` is left in a caller-should-not-use state (partially populated in general -- the
// §11 reject-over-degrade convention this codebase already applies everywhere else: callers gate
// on the returned status, never on `out`'s contents, exactly like SslmModel::Load's own `out`).
inline AdapterLoadStatus LoadAdapterArtifact(const std::string& path, const BaseModelGeometry& base,
                                              AdapterHandle& out, std::string* err) {
	std::vector<uint8_t> bytes;
	if (!ReadFile(path.c_str(), bytes)) {
		if (err) *err = "could not read adapter file: " + path;
		return AdapterLoadStatus::FileReadFailed;
	}

	// T-2104 (Poirot 8e07d0c review, Significant 2): ONE parse. `SslmModel::Load` opens the
	// container, validates it, and parses every present section (Config/SigmoidLut/Weights/
	// DeltaFoldScales/UFoldScales) into `out.view_`, which this handle keeps for its own lifetime --
	// there is no second `SslmArtifact::OpenFromMemory` and no re-parse of any section this loader
	// already gets typed access to through the view's own public members.
	{
		std::string model_err;
		const superslm::SslmModelStatus st =
		    superslm::SslmModel::Load(bytes.data(), bytes.size(), out.view_, &model_err);
		if (st != superslm::SslmModelStatus::Ok) {
			if (err) *err = std::string("adapter model rejected: ") +
			                superslm::SslmModelStatusName(st) + " -- " + model_err;
			// T-2104 (Poirot 7a0b6426 review, Minor 9): SslmModel::Load's own ArtifactRejected
			// (model.h: "the outer SslmArtifact::OpenFromMemory rejected the container") is the
			// SAME container-level rejection AdapterLoadStatus::ArtifactRejected was declared to
			// name, back when this loader still opened its own separate SslmArtifact. That second
			// open is gone (Significant 2, this same round), but the distinction it was declared
			// for -- "no such file" vs "wrong base" vs "artifact structurally broken," this enum's
			// own header promise -- is still worth keeping, so it is re-wired here rather than left
			// dead: a container-level rejection maps to AdapterLoadStatus::ArtifactRejected, every
			// other SslmModelStatus rejection (a structural/domain rejection deeper than the
			// container -- WGT1/DFS1/UFS1/CFG1/SIL1) to ModelRejected.
			return st == superslm::SslmModelStatus::ArtifactRejected ? AdapterLoadStatus::ArtifactRejected
			                                                          : AdapterLoadStatus::ModelRejected;
		}
	}

	// --- ADP1 (Provenance) -- via view_'s own Section() forwarder, the SAME long-lived artifact
	// Load already opened (no second open). ---------------------------------
	const superslm::SslmSectionView* prov = out.view_.Section(superslm::SslmSectionType::Provenance);
	if (!prov) {
		if (err) *err = "adapter carries no Provenance section";
		return AdapterLoadStatus::MissingProvenanceSection;
	}
	if (prov->byte_size < kAdp1FixedBytes) {
		if (err) *err = "Provenance section (" + std::to_string(prov->byte_size) +
		                 " bytes) is shorter than ADP1's 68-byte fixed prefix";
		return AdapterLoadStatus::BadAdp1Size;
	}
	if (std::memcmp(prov->data, kAdp1Magic, 4) != 0) {
		if (err) *err = "Provenance section's first 4 bytes are not \"ADP1\"";
		return AdapterLoadStatus::BadAdp1Magic;
	}
	const uint32_t adp1_version = static_cast<uint32_t>(superslm_marshal::RdI32(prov->data + 4));
	if (adp1_version != kAdp1Version) {
		if (err) *err = "unsupported ADP1 version " + std::to_string(adp1_version);
		return AdapterLoadStatus::BadAdp1Version;
	}
	out.meta.rank = static_cast<uint32_t>(superslm_marshal::RdI32(prov->data + 8));
	out.meta.target_modules_mask = static_cast<uint32_t>(superslm_marshal::RdI32(prov->data + 12));
	std::memcpy(out.meta.base_artifact_hash.data(), prov->data + 16, superslm::kIntegrityHashBytes);
	{
		uint64_t bits = 0;
		for (int i = 0; i < 8; ++i) bits |= static_cast<uint64_t>(prov->data[48 + i]) << (8 * i);
		double v = 0.0;
		std::memcpy(&v, &bits, 8);
		out.meta.lora_alpha = v;
	}
	out.meta.use_rslora = superslm_marshal::RdI32(prov->data + 56) != 0;
	if (superslm_marshal::RdI32(prov->data + 60) != 0) {
		if (err) *err = "ADP1 reserved field != 0";
		return AdapterLoadStatus::BadAdp1Reserved;
	}
	const uint32_t name_len = static_cast<uint32_t>(superslm_marshal::RdI32(prov->data + 64));
	if (prov->byte_size < static_cast<uint64_t>(kAdp1FixedBytes) + name_len) {
		if (err) *err = "ADP1's declared source_adapter_name_len overruns the Provenance section";
		return AdapterLoadStatus::BadAdp1Size;
	}
	out.meta.source_adapter_name.assign(reinterpret_cast<const char*>(prov->data + kAdp1FixedBytes),
	                                     name_len);

	// T-2104 (Poirot 8e07d0c review, Minor 2): reject a mask with any bit at or past
	// kPeftAdaptableProjections' own length -- an unknown target module must never load clean and
	// be silently treated as though it had not been declared.
	if (TargetModulesMaskHasUnknownBit(out.meta.target_modules_mask)) {
		if (err) *err = "ADP1 target_modules_mask " + std::to_string(out.meta.target_modules_mask) +
		                 " sets a bit at or past the known projection count (" +
		                 std::to_string(std::size(superslm::kPeftAdaptableProjections)) + ")";
		return AdapterLoadStatus::InvalidTargetModulesMask;
	}

	// --- base-hash (design Sec9 item (d), ValidateAmplifyingFoldBaseHash) --
	std::string hash_err;
	if (superslm::ValidateAmplifyingFoldBaseHash(out.meta.base_artifact_hash, base.base_artifact_hash,
	                                              &hash_err) != superslm::SslmModelStatus::Ok) {
		if (err) *err = "adapter base-hash mismatch: " + hash_err;
		return AdapterLoadStatus::BaseHashMismatch;
	}

	// --- weights / DFS1 / UFS1 -- read directly from out.view_'s own already-parsed members. ---
	if (!out.view_.has_weights) {
		if (err) *err = "adapter carries no Weights section";
		return AdapterLoadStatus::MissingWeightsSection;
	}
	if (!out.view_.has_delta_fold_scales) {
		if (err) *err = "adapter carries no DeltaFoldScales section";
		return AdapterLoadStatus::MissingDeltaFoldSection;
	}
	if (!out.view_.has_u_fold_scales) {
		if (err) *err = "adapter carries no UFoldScales section";
		return AdapterLoadStatus::MissingUFoldSection;
	}

	out.layer_adapters.assign(base.num_hidden_layers, superslm::LayerAdapter{});
	for (auto& la : out.layer_adapters) la.rank = out.meta.rank;

	const std::vector<std::string_view> claimed = TargetModulesFromMask(out.meta.target_modules_mask);

	for (const auto& dentry : out.view_.delta_fold_scales.Entries()) {
		uint32_t layer = 0;
		std::string proj;
		if (!ParseLayerProjName(dentry.name, layer, proj)) {
			if (err) *err = "unparsable DeltaFoldScales entry name: " + std::string(dentry.name);
			return AdapterLoadStatus::UnparsableEntryName;
		}
		if (layer >= base.num_hidden_layers) {
			if (err) *err = "DeltaFoldScales entry \"" + std::string(dentry.name) + "\" names layer " +
			                 std::to_string(layer) + " >= base model's own " +
			                 std::to_string(base.num_hidden_layers) + " layers";
			return AdapterLoadStatus::LayerIndexOutOfRange;
		}

		std::string proj_err;
		if (superslm::ValidateAmplifyingFoldProjection(proj, claimed, &proj_err) !=
		    superslm::SslmModelStatus::Ok) {
			if (err) *err = "adapter projection invalid for \"" + std::string(dentry.name) +
			                 "\": " + proj_err;
			return AdapterLoadStatus::ProjectionInvalid;
		}

		const uint64_t expected_out =
		    AdapterOutChannelsFor(proj, base.hidden_size, base.intermediate_size, base.kv_hidden_size);
		std::string dim_err;
		if (superslm::ValidateAmplifyingFoldDimension(dentry, expected_out, &dim_err) !=
		    superslm::SslmModelStatus::Ok) {
			if (err) *err = "DeltaFoldScales dimension mismatch for \"" + std::string(dentry.name) +
			                 "\": " + dim_err;
			return AdapterLoadStatus::DeltaDimensionMismatch;
		}

		const superslm::SslmAmplifyingFoldEntry* uentry = out.view_.u_fold_scales.Entry(dentry.name);
		if (!uentry) {
			if (err) *err = "UFoldScales carries no entry matching DeltaFoldScales' own \"" +
			                 std::string(dentry.name) + "\"";
			return AdapterLoadStatus::MissingUFoldEntry;
		}
		std::string udim_err;
		if (superslm::ValidateAmplifyingFoldDimension(*uentry, out.meta.rank, &udim_err) !=
		    superslm::SslmModelStatus::Ok) {
			if (err) *err = "UFoldScales dimension mismatch for \"" + std::string(dentry.name) +
			                 "\": " + udim_err;
			return AdapterLoadStatus::UFoldDimensionMismatch;
		}

		const superslm::SslmTensorView* a_t =
		    out.view_.weights.Tensor(std::string(dentry.name) + ".lora_A");
		const superslm::SslmTensorView* b_t =
		    out.view_.weights.Tensor(std::string(dentry.name) + ".lora_B");
		if (!a_t || !b_t) {
			if (err) *err = "adapter Weights section carries no lora_A/lora_B tensor for \"" +
			                 std::string(dentry.name) + "\"";
			return AdapterLoadStatus::MissingWeightTensor;
		}

		// T-2104 (Poirot 8e07d0c review, Significant 1): the two pointers `AddAmplifyingLoraDelta`
		// actually walks (forward_sites.cpp) -- `a_weight` read for `in_channels * rank` bytes,
		// `b_weight` for `rank * out_channels` -- validated against the SAME [rank, in_channels] /
		// [out_channels, rank] shape PEFT's own orientation declares (forward_sites.h's own
		// LayerAdapterProjection comment: "a_weight is [rank, in_channels]... b_weight is
		// [out_channels, rank]"), for the FIRST time. Everything else about this pair was already
		// checked by name (rejected magic, version, reserved, projection, entry name, layer index,
		// DFS1/UFS1 dimension) -- these two were taken on trust and become raw pointers the engine
		// dereferences at a length neither tensor's own declared shape had ever been read against.
		const uint64_t in_channels =
		    AdapterInChannelsFor(proj, base.hidden_size, base.intermediate_size);
		if (a_t->rank != 2 || a_t->shape[0] != out.meta.rank || a_t->shape[1] != in_channels) {
			if (err) {
				*err = "\"" + std::string(dentry.name) + ".lora_A\" shape mismatch: expected rank=2 "
				       "shape=[" + std::to_string(out.meta.rank) + "," + std::to_string(in_channels) +
				       "], got rank=" + std::to_string(a_t->rank) + " shape=[" +
				       (a_t->rank >= 1 ? std::to_string(a_t->shape[0]) : "?") + "," +
				       (a_t->rank >= 2 ? std::to_string(a_t->shape[1]) : "?") + "]";
			}
			return AdapterLoadStatus::LoraAWeightShapeMismatch;
		}
		if (b_t->rank != 2 || b_t->shape[0] != expected_out || b_t->shape[1] != out.meta.rank) {
			if (err) {
				*err = "\"" + std::string(dentry.name) + ".lora_B\" shape mismatch: expected rank=2 "
				       "shape=[" + std::to_string(expected_out) + "," + std::to_string(out.meta.rank) +
				       "], got rank=" + std::to_string(b_t->rank) + " shape=[" +
				       (b_t->rank >= 1 ? std::to_string(b_t->shape[0]) : "?") + "," +
				       (b_t->rank >= 2 ? std::to_string(b_t->shape[1]) : "?") + "]";
			}
			return AdapterLoadStatus::LoraBWeightShapeMismatch;
		}

		superslm::LayerAdapterProjection lap;
		lap.a_weight = reinterpret_cast<const int8_t*>(a_t->data);
		lap.b_weight = reinterpret_cast<const int8_t*>(b_t->data);
		lap.delta_fold_entry = out.view_.delta_fold_scales.Entry(dentry.name);
		lap.u_fold_entry = uentry;

		superslm::LayerAdapter& la = out.layer_adapters[layer];
		if (proj == "q_proj") la.q = lap;
		else if (proj == "o_proj") la.o = lap;
		else if (proj == "gate_proj") la.gate = lap;
		else if (proj == "up_proj") la.up = lap;
		else if (proj == "down_proj") la.down = lap;
		else if (proj == "k_proj") la.k = lap;
		else if (proj == "v_proj") la.v = lap;
	}

	return AdapterLoadStatus::Ok;
}

// Points every one of `num_hidden_layers` entries in `layers` at `handle`'s own per-layer
// `LayerAdapter` (handle != nullptr -- the "adapter bound" case), or clears every entry to
// nullptr (handle == nullptr -- LayerWeights' own documented B1b NULL-adapter case, "the
// caller sets this to nullptr on EVERY layer's LayerWeights when the sequence has no bound
// adapter"). This is the whole of "unload" (call with nullptr) and "swap" (load a second
// AdapterHandle, call again with its address -- every LayerWeights::adapter pointer is
// REPLACED, never merged; the caller then drops the first handle, and its destructor frees
// every byte the old adapter's pointers referenced). `handle->layer_adapters.size()` must be
// >= `num_hidden_layers` (guaranteed by LoadAdapterArtifact, which sizes it from the same
// `base.num_hidden_layers` the caller passes here).
inline void ApplyAdapterToLayers(const AdapterHandle* handle, superslm::LayerWeights* layers,
                                  uint32_t num_hidden_layers) {
	for (uint32_t l = 0; l < num_hidden_layers; ++l) {
		layers[l].adapter = handle ? &handle->layer_adapters[l] : nullptr;
	}
}

}  // namespace superslm_adapter

#endif  // SUPERSLM_TOOLS_SSLM_ADAPTER_LOADER_H

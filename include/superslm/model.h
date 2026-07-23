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
// WeightScales is a tensor manifest too (int32 (identity,mult,shift) fold ops); its magic.
inline constexpr uint8_t kWeightScalesMagic[4] = {'W', 'S', 'C', '1'};

// Keyed numeric-constant geometry (v1). See docs/sslm_format.md "Keyed numeric-constant
// blob — KVC1". The three keyed integer-constant sections (CompositionConstants,
// KvLandingScales, KvLandingReciprocals) share this layout; the section type fixes
// value_words (2 or 3), and every value — including the exponent e — is a little-endian int64.
inline constexpr uint32_t kMaxConstantEntries = 1048576;
inline constexpr uint32_t kConstantHeaderBytes = 24;  // magic+version+count+value_words+name_blob_len+reserved
inline constexpr uint32_t kConstantDescBytes = 8;     // name_off + name_len
inline constexpr uint8_t kConstantsMagic[4] = {'K', 'V', 'C', '1'};

// Config blob (v1). A fixed 84-byte CFG1 struct. See docs/sslm_format.md "Config blob".
inline constexpr uint32_t kConfigBytes = 84;
inline constexpr uint8_t kConfigMagic[4] = {'C', 'F', 'G', '1'};

// Sigmoid-LUT blob (SIL1, v2). A fixed-layout section (like CFG1 — the table geometry is a
// pinned build-time constant, not per-artifact): a 16-byte header then kSigmoidLutEntries
// little-endian int32 Q15 nodes. N, X, Q_idx are compile-time constants of the construction,
// NOT carried in the section. See docs/sslm_format.md "Sigmoid-LUT blob — SIL1" and
// SuperSLM_S2.4_SiLU_LUT_Design §4, §8.
inline constexpr uint32_t kSigmoidLutNodes = 1024;                        // N (x in [-X, X])
inline constexpr uint32_t kSigmoidLutEntries = kSigmoidLutNodes + 1;      // N+1 = 1025 table entries
inline constexpr uint32_t kSigmoidLutHeaderBytes = 16;                    // magic+version+entry_count+reserved
inline constexpr uint32_t kSigmoidLutBytes = kSigmoidLutHeaderBytes + kSigmoidLutEntries * 4;  // int32 payload
inline constexpr uint8_t kSigmoidLutMagic[4] = {'S', 'I', 'L', '1'};

// KV quantization width (Config.kv_precision).
enum class SslmKvPrecision : uint32_t {
	Int8 = 0,
	Int16 = 1,
};

// The model's architecture, parsed from the CFG1 Config section. The eight dimension
// fields are guaranteed nonzero; enum/bool fields are guaranteed in range.
struct SslmModelConfig {
	uint32_t hidden_size = 0;
	uint32_t num_hidden_layers = 0;
	uint32_t num_attention_heads = 0;
	uint32_t num_key_value_heads = 0;
	uint32_t head_dim = 0;
	uint32_t intermediate_size = 0;
	uint32_t vocab_size = 0;
	uint32_t context_cap = 0;
	bool tie_word_embeddings = false;
	SslmKvPrecision kv_precision = SslmKvPrecision::Int8;
	uint32_t kv_block_size = 0;
	uint32_t unicode_major = 0;
	uint32_t unicode_minor = 0;
	uint32_t unicode_patch = 0;
	double rope_theta = 0.0;   // recorded; not read by a kernel (offline RoPE-table input)
	double rms_norm_eps = 0.0; // recorded; the integer RMSNorm carries no eps term
};

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
	// --- KVC1 keyed-constant sub-parse ---
	BadConstantsMagic,          // first four bytes are not 'KVC1'
	UnsupportedConstantsVersion,// KVC1 version != kManifestVersion
	TooManyConstantEntries,     // entry_count > kMaxConstantEntries
	BadValueWords,              // value_words not in {2,3}, or not the section type's required count
	ConstantsOutOfBounds,       // header + descriptors + values + name blob exceeds byte_size
	BadEntryName,               // an entry's name range is outside the name blob
	EmptyEntryName,             // an entry's name_len == 0
	DuplicateEntryName,         // two entries carry the same name
	BadConstantsReserved,       // the KVC1 header reserved field != 0
	// --- CFG1 config sub-parse ---
	BadConfigSize,              // Config section byte_size != kConfigBytes
	BadConfigMagic,             // first four bytes are not 'CFG1'
	UnsupportedConfigVersion,   // CFG1 version != kManifestVersion
	BadConfigDim,               // a required dimension field is 0
	BadKvPrecision,             // kv_precision not in {0,1}
	BadConfigBool,              // tie_word_embeddings not in {0,1}
	BadConfigReserved,          // the CFG1 reserved field != 0
	// --- SIL1 sigmoid-LUT sub-parse ---
	BadSigmoidLutSize,          // SIL1 section byte_size != kSigmoidLutBytes
	BadSigmoidLutMagic,         // first four bytes are not 'SIL1'
	UnsupportedSigmoidLutVersion, // SIL1 version != kManifestVersion
	BadSigmoidLutCount,         // entry_count != kSigmoidLutEntries
	BadSigmoidLutReserved,      // the SIL1 reserved field != 0
	BadSigmoidLutContent,       // a node does not match the pinned canonical table (S-HARDEN-1, F20/F22)
	// --- S-HARDEN-1 load-time schema-value gate (D-SLM141/D-SLM142) ---
	ArtifactRejected,            // SslmModel::Load: the outer SslmArtifact::OpenFromMemory rejected the
	                              // container; the underlying SslmStatus is named in the diagnostic
	CompositionScaleOutOfDomain, // KVC1 CompositionConstants (m,e) outside SiluSigmoidQ15's no-UB floor
	WeightScaleShiftOutOfDomain, // WSC1 shift column outside [0,31] (RoundingDivideByPOT's exponent domain)
	WeightScaleIdentityNotBool,  // WSC1 identity column not in {0,1}
	RopeTableEntryOutOfDomain,   // ROP1 element outside [-2^30, 2^30] (RopeApplyPair's yr-addition safety bound)
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

// The KVC1 magic a section type carries, or nullptr if the type has no keyed-constant
// table. Only CompositionConstants/KvLandingScales/KvLandingReciprocals carry one.
const uint8_t* ConstantsMagicFor(SslmSectionType type) noexcept;

// The value_words a keyed-constant section type requires (2 or 3), or 0 if the type has
// no keyed-constant table.
uint32_t ExpectedValueWords(SslmSectionType type) noexcept;

// A validated view of one keyed-constant entry. `name` points into the section's name
// blob; `values` points at `value_words` little-endian int64s in the section's value
// array — read them with the byte-assembly reader, never a cast (the array is not
// guaranteed aligned for an int64 load).
struct SslmConstantEntry {
	std::string_view name;
	const uint8_t* values = nullptr;   // value_words * 8 bytes, little-endian int64 each
	uint32_t value_words = 0;
};

// A parsed, fully validated keyed numeric-constant table (KVC1) for one section.
// Constructed only through Parse; a default instance is empty.
class SslmKeyedConstants {
public:
	SslmKeyedConstants() = default;

	// Parse the KVC1 table in `section` (whose type fixes the expected value_words). The
	// section must already be validated by SslmArtifact. The parse is a hostile-input
	// trust boundary: it fails closed on any malformed field, leaving `out` empty. On Ok,
	// `out` owns the validated entry views. Never throws; never exposes an entry before
	// every descriptor passes validation.
	static SslmModelStatus Parse(const SslmSectionView& section, SslmKeyedConstants& out,
	                             std::string* err);

	const std::vector<SslmConstantEntry>& Entries() const noexcept { return entries_; }

	// The entry with the given name, or nullptr if absent.
	const SslmConstantEntry* Entry(std::string_view name) const noexcept;

	// Read value index `w` (< value_words) of `entry` as a signed int64 (little-endian
	// byte assembly). Behavior is undefined if `w >= entry.value_words`.
	static int64_t Value(const SslmConstantEntry& entry, uint32_t w) noexcept;

private:
	std::vector<SslmConstantEntry> entries_;
};

// Parse the CFG1 Config section into `out`. The section must already be validated by
// SslmArtifact. Rejects (fails closed, `out` left default) on a wrong size/magic/version,
// a zero dimension, an out-of-range enum/bool, or a nonzero reserved field — the §11
// reject-over-degrade law for config. Never throws.
SslmModelStatus ParseConfig(const SslmSectionView& section, SslmModelConfig& out, std::string* err);

// A validated view of the SIL1 sigmoid LUT: kSigmoidLutEntries Q15 nodes. `values` points
// into the artifact's owned buffer (valid for its lifetime). Read a node with SigmoidLutValue
// (little-endian byte assembly — the payload is not guaranteed int32-aligned).
struct SslmSigmoidLut {
	const uint8_t* values = nullptr;   // kSigmoidLutEntries * 4 bytes, little-endian int32 Q15
	uint32_t entry_count = 0;          // == kSigmoidLutEntries on Ok
};

// Parse the SIL1 Sigmoid-LUT section (whose geometry is fixed by kSigmoidLut* constants). The
// section must already be validated by SslmArtifact. Fixed-layout like CFG1: the exact-size
// check gates every read. Rejects (fails closed, `out` left default) on a wrong size/magic/
// version/entry_count or a nonzero reserved field — the §11 reject-over-degrade law. Never throws.
//
// S-HARDEN-1 (F20/F22): SIL1 is a universal construction the spec fixes entirely, not
// model-specific learned data, so every node is ALSO validated against the pinned
// canonical table (include/superslm/silu_lut_canonical.h) — a structurally valid
// section whose content is not byte-for-byte the canonical table is rejected with
// BadSigmoidLutContent. This is what stops a hostile-but-structurally-valid table
// (e.g. adjacent nodes at INT32_MIN/INT32_MAX) from ever reaching SiluSigmoidQ15's
// interpolation, which relies on the canonical invariant |table[i+1]-table[i]| bounded.
SslmModelStatus ParseSigmoidLut(const SslmSectionView& section, SslmSigmoidLut& out, std::string* err);

// Read Q15 node `i` (< entry_count) of `lut` as a signed int32 (little-endian byte assembly).
// Behavior is undefined if `i >= entry_count`.
int32_t SigmoidLutValue(const SslmSigmoidLut& lut, uint32_t i) noexcept;

// A fully composed, value-validated view of one `.sslm` artifact's model
// sections, built only by SslmModel::Load. S-HARDEN-1 (D-SLM141): this is the
// one place that opens an artifact, drives every present section's
// sub-parser, and then validates every section's carried VALUES against its
// consumer's domain — the schema-value gate. A default instance is empty;
// Load leaves it at defaults on any rejection (fail closed, never a partial
// view).
struct SslmModelView {
	SslmModelConfig config;
	bool has_config = false;

	SslmSigmoidLut sigmoid_lut;
	bool has_sigmoid_lut = false;

	SslmTensorManifest weights;
	bool has_weights = false;

	SslmTensorManifest biases;
	bool has_biases = false;

	SslmTensorManifest rope_tables;
	bool has_rope_tables = false;

	SslmTensorManifest weight_scales;
	bool has_weight_scales = false;

	SslmKeyedConstants composition_constants;
	bool has_composition_constants = false;

	// KvLandingScales/KvLandingReciprocals (C27): parsed structurally like
	// every other keyed-constant section, but their VALUE domain is not yet
	// declared — no consumer exists in the tree for either (S-HARDEN-1,
	// D-SLM142). SslmModel::Load leaves their carried values unchecked; a
	// future slot adds their domain descriptor when the C27 consumer lands.
	SslmKeyedConstants kv_landing_scales;
	bool has_kv_landing_scales = false;

	SslmKeyedConstants kv_landing_reciprocals;
	bool has_kv_landing_reciprocals = false;
};

// The load-time orchestration entry point (S-HARDEN-1, D-SLM141): the one
// call site that composes SslmArtifact::OpenFromMemory, every present
// section's sub-parser, and the schema-value gate into a single pass. Never
// throws; `out` is left at SslmModelView{} defaults on ANY rejection —
// container-level, structural sub-parse, or value-domain — and is populated
// only on full success (§11 reject-over-degrade). `err` (if non-null) carries
// a diagnostic naming what was rejected.
//
// A container-level rejection (SslmArtifact::OpenFromMemory failing) is
// reported as SslmModelStatus::ArtifactRejected with the underlying
// SslmStatus named in `err`; every other rejection is the specific
// SslmModelStatus code of the section sub-parse or value-domain check that
// failed.
class SslmModel {
public:
	static SslmModelStatus Load(const uint8_t* data, size_t size, SslmModelView& out, std::string* err);
};

} // namespace superslm

#endif // SUPERSLM_MODEL_H

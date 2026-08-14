// SuperSLM model view (`.sslm` model sections). Current format version is
// superslm::kArtifactFormatVersion (artifact.h), the single source of
// truth -- not restated as a number here (S-HARDEN-8; see artifact.h's
// identical header-comment note).
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

#include <array>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include "superslm/artifact.h"
#include "superslm/tokenizer.h"
#include "superslm/trace_hook.h"

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
// T-2021/T-2029 B0b (design Sec9, D-SLM3093): the two new runtime-additive-LoRA adapter arrays
// get their OWN magics, distinct from WSC1's -- never reused or aliased.
inline constexpr uint8_t kDeltaFoldScalesMagic[4] = {'D', 'F', 'S', '1'};
inline constexpr uint8_t kUFoldScalesMagic[4] = {'U', 'F', 'S', '1'};

// WSC1's own unsigned shift domain (RoundingDivideByPOT's documented int32 exponent domain).
// T-2041 (Poirot c81e48c review, Minor 1): this is now the ONLY definition of
// kWeightScaleShiftMin/Max -- a prior TU-internal copy in model.cpp was removed this build, so
// there is nothing left for a static_assert to pin these two constants AGAINST each other; the
// static_assert in model.cpp (src/model.cpp, "WSC1 shift's domain must equal RoundingDivideByPOT's
// documented int32 exponent domain") instead pins this single copy against
// kRoundingDivideByPotExponentMinI32/Max (intmath.h) -- the primitive whose domain these
// constants actually describe. Exposed here because design Sec9's B0b build cell
// ("swap direction (ii): a genuine delta-fold/u-fold triple's negative exponent, read as WSC1's
// own unsigned shift column, is already caught by the EXISTING check") needs a way to state that
// existing, unchanged bound without duplicating WSC1's own internal validator's logic.
inline constexpr int32_t kWeightScaleShiftMin = 0;
inline constexpr int32_t kWeightScaleShiftMax = 31;

// The amplifying-fold triple's own signed domain (design Sec4/Sec9, D-SLM2917) --
// ApplyAmplifyingWeightScaleFold's consumption shape, widened relative to WSC1's own unsigned
// [kWeightScaleShiftMin=0, kWeightScaleShiftMax=31] (ApplyWeightScaleFold's domain, unchanged).
inline constexpr int32_t kAmplifyingScaleExponentMin = -31;
inline constexpr int32_t kAmplifyingScaleExponentMax = 31;

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
	KvLandingScaleOutOfDomain,   // KvLandingScales entry's m_t (word 0) outside [kKvLandingScaleMantissaMin,
	                              // kKvLandingScaleMantissaMax] (Sec7.2a third limb, S3.3). Its word 1
	                              // (e_target) carries no check here -- pending-consumer per D-SLM142
	                              // (Poirot a6d6728 second confirmation review, finding 1). Enforced by
	                              // ValidateKvLandingScalesDomain (model.cpp), wired into
	                              // ValidateSectionValues.
	KvLandingReciprocalOutOfDomain, // KvLandingReciprocals entry's e_t (word 1) below kKvLandingExponentMin,
	                              // OR its R_t (word 2) outside [kKvLandingReciprocalMin,
	                              // kKvLandingReciprocalMax] (Sec7.2a third limb, S3.3; Poirot 2026-07-28
	                              // finding 3; confirmation review D-SLM372, correcting D-SLM370(c)'s wrong
	                              // section name -- e_t is the field LandingRescale's composite actually
	                              // reads from THIS section at runtime). Enforced by
	                              // ValidateKvLandingReciprocalsDomain (model.cpp), wired into
	                              // ValidateSectionValues.
	// --- S-HARDEN-2 tokenizer joins (F18, F6, F7, F15) ---
	TokenizerRejected,           // SslmModel::Load: TOK1/UnicodeTables present but TokenizerView::Open rejected
	                              // them (structurally, or exactly one of the two sections is present)
	TokenizerVocabSizeMismatch,  // TOK1.vocab_count != CFG1.vocab_size -- the two blobs' declared sizes disagree
	// --- Poirot fa3189a-s3.3-rope-site-and-c32-softmax-review-2026-07-28.md, Significant 5 ---
	BadConfigHeadDimParity,      // CFG1 head_dim is odd -- RoPE pairs elements two at a time
	                              // (forward_sites.h's own "head_dim odd is a load-time rejection",
	                              // Sec6.2 step 3); ParseConfigImpl performed no parity check before
	                              // this addition.
	// --- S3.3, Sec11 S3.3 / Sec13.1 cell 4: config-geometry x tensor-shape join
	// (D-SLM410, D-SLM421, D-SLM423, board T-1333). ValidateConfigGeometryJoin
	// and ValidateRopeTablesShapeAgainstConfig (model.cpp), wired into
	// ValidateSectionValues, reject a hostile artifact for each of the four
	// relations below.
	ConfigGeometryKvHeadsExceedsHeads,   // R3: num_key_value_heads > num_attention_heads
	ConfigGeometryHeadsNotDivisibleByKv, // R2: num_attention_heads % num_key_value_heads != 0
	ConfigGeometryHiddenSizeMismatch,    // R1: hidden_size != num_attention_heads * head_dim
	RopeTablesShapeMismatchConfig,       // R4: a present ROP1 "cos"/"sin" tensor's elem_count !=
	                                      // context_cap * (head_dim / 2), checked independently
	                                      // per tensor -- the ROP1<->CFG1 join itself
	// --- T-1415 (whole-tree review b9dcbe0, Minor 3) ---
	WeightScaleTripleCountInvalid,       // WSC1 tensor's elem_count % 3 != 0 -- the section stores
	                                      // rows of (identity, mult, shift) int32 triples
	                                      // (docs/sslm_format.md), and ValidateWeightScalesDomain's
	                                      // prior walk of elem_count / 3 triples left a trailing
	                                      // partial triple's elements unvalidated. Rejected outright
	                                      // rather than walked partially.
	// --- S3.7 (§8.3): the calibration band's own hostile-value gate ---
	CalibrationBandOutOfDomain,          // a CalibrationBand entry's (min, max) violates min <= max,
	                                      // min >= 0, or max > 0 -- ValidateCalibrationBandDomain
	                                      // (model.cpp), wired into ValidateSectionValues.
	// --- T-2021/T-2029 B0b (design Sec4/Sec9/Sec11 B0b, D-SLM3093/D-SLM3094): the two new
	// runtime-additive-LoRA adapter arrays (DeltaFoldScales/DFS1, UFoldScales/UFS1) ---
	AmplifyingFoldSectionTypeMismatch,   // a section's DECLARED type does not match the wrapper
	                                      // kind that attempted to parse it (design's own
	                                      // "constructed only by parsing a section whose declared
	                                      // type matches the wrapper's own kind") -- checked
	                                      // BEFORE any byte is read; the direction-(i) swap
	                                      // mutation (a genuine WeightScales section handed to
	                                      // the DeltaFoldScales/UFoldScales parser) is caught here.
	AmplifyingFoldTripleCountInvalid,    // an entry's elem_count is not a multiple of 3 (mirrors
	                                      // WeightScaleTripleCountInvalid)
	AmplifyingFoldIdentityNotBool,       // identity column not in {0,1}
	AmplifyingFoldExponentOutOfDomain,   // exponent column outside the SIGNED
	                                      // [kAmplifyingScaleExponentMin, kAmplifyingScaleExponentMax]
	                                      // = [-31,31] domain (design Sec4/Sec9, D-SLM2917;
	                                      // WIDENED relative to WSC1's own unsigned [0,31] -- the
	                                      // direction-(ii) swap mutation, a genuine delta/u-fold
	                                      // section's negative exponent handed to WSC1's OWN
	                                      // ValidateWeightScalesDomain, is already caught by the
	                                      // EXISTING WeightScaleShiftOutOfDomain check, retained
	                                      // as defense-in-depth per design Sec9)
	AmplifyingFoldDimensionMismatch,     // a DeltaFoldScales entry's row_count != the adapted
	                                      // projection's own declared out_channels, or a
	                                      // UFoldScales entry's row_count != the adapter's own
	                                      // declared rank r (design Sec9 item (a))
	AmplifyingFoldProjectionInvalid,     // an entry's declared target projection is not one of
	                                      // the seven PEFT-adaptable projections, or is not
	                                      // claimed by the adapter's own target_modules (design
	                                      // Sec9 item (c))
	AmplifyingFoldBaseHashMismatch,      // an entry's declared base-artifact integrity hash does
	                                      // not match the actually-mapped base's own hash (design
	                                      // Sec9 item (d))
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
	// error, `out` is left empty and `err` (if non-null) carries a diagnostic. Throws
	// only std::bad_alloc (S-HARDEN-7, F5); never exposes a tensor byte before every
	// descriptor passes validation.
	static SslmModelStatus Parse(const SslmSectionView& section, SslmTensorManifest& out,
	                             std::string* err);

	const std::vector<SslmTensorView>& Tensors() const noexcept { return tensors_; }

	// The tensor with the given name, or nullptr if absent.
	const SslmTensorView* Tensor(std::string_view name) const noexcept;

private:
	// S-HARDEN-7 (design Sec3.1): grants src/model.cpp's
	// SslmTensorManifestAccess (defined only there) access to tensors_, so
	// Parse's *Impl body can live entirely in the .cpp rather than as a
	// private member declaration here — a private member declaration would
	// itself be picked up by the membership-check AST walk, which does not
	// see access specifiers. See artifact.h's identical SslmArtifactAccess
	// comment for the full reasoning.
	friend struct SslmTensorManifestAccess;

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
	// `out` owns the validated entry views. Throws only std::bad_alloc (S-HARDEN-7, F5);
	// never exposes an entry before every descriptor passes validation.
	static SslmModelStatus Parse(const SslmSectionView& section, SslmKeyedConstants& out,
	                             std::string* err);

	const std::vector<SslmConstantEntry>& Entries() const noexcept { return entries_; }

	// The entry with the given name, or nullptr if absent.
	const SslmConstantEntry* Entry(std::string_view name) const noexcept;

	// Read value index `w` (< value_words) of `entry` as a signed int64 (little-endian
	// byte assembly). Behavior is undefined if `w >= entry.value_words`.
	static int64_t Value(const SslmConstantEntry& entry, uint32_t w) noexcept;

private:
	// S-HARDEN-7 (design Sec3.1): see SslmTensorManifest's identical
	// SslmTensorManifestAccess comment above.
	friend struct SslmKeyedConstantsAccess;

	std::vector<SslmConstantEntry> entries_;
};

// T-2021/T-2029 B0b (design Sec4/Sec9/Sec11 B0b, D-SLM3093/D-SLM3094): the two new
// runtime-additive-LoRA adapter arrays this design's build adds -- `DeltaFoldScales`/`DFS1`
// (design Sec4's own `delta_identity`/`delta_mult`/`delta_exponent`, one triple per adapted
// output channel) and `UFoldScales`/`UFS1` (design Sec4's extension `u_identity`/`u_mult`/
// `u_exponent`, one triple per rank index). NEVER `int32_t*`, NEVER `WeightScales`'s own
// `SslmTensorManifest` (a genuinely distinct C++ type, section-type-gated at parse time, per
// design Sec9's own "never implicitly convertible... to each other" requirement) -- `Kind`
// makes `SslmAmplifyingFoldScaleView<Delta>` and `SslmAmplifyingFoldScaleView<U>` two distinct
// template instantiations with no implicit conversion between them.
enum class SslmAmplifyingFoldKind : uint32_t { Delta, U };

// One entry's raw view: `name` is the adapted (layer, projection) this triple set targets
// (e.g. "layer3.q_proj", the same per-projection naming convention WeightScales' own tensors
// already use); `data` points at `row_count * 3` little-endian int32 values (identity, mult,
// exponent per row); `row_count` is out_channels (DeltaFoldScales) or rank r (UFoldScales) --
// checked against the caller's own expectation by `ValidateAmplifyingFoldDimension` (design
// Sec9 item (a)), not asserted here.
struct SslmAmplifyingFoldEntry {
	std::string_view name;
	const uint8_t* data = nullptr;
	uint64_t row_count = 0;
};

template <SslmAmplifyingFoldKind Kind>
class SslmAmplifyingFoldScaleView {
public:
	SslmAmplifyingFoldScaleView() = default;

	// Parses `section` ONLY if `section.type` matches THIS Kind's own SslmSectionType
	// (`DeltaFoldScales` for `Kind::Delta`, `UFoldScales` for `Kind::U`) -- checked FIRST,
	// before any manifest byte is read (design Sec9's own "constructed only by parsing a
	// section whose declared type matches the wrapper's own kind"). A section declared as the
	// OTHER kind, or as `WeightScales`, is rejected with `AmplifyingFoldSectionTypeMismatch`
	// regardless of its byte contents -- the swap-mutation direction (i) this design's own B0b
	// build cell exercises (design Sec11 B0b). On any rejection, `out` is left empty. Throws
	// only std::bad_alloc (S-HARDEN-7 convention, matching every other sub-parse in this file).
	static SslmModelStatus Parse(const SslmSectionView& section, SslmAmplifyingFoldScaleView& out,
	                             std::string* err);

	const std::vector<SslmAmplifyingFoldEntry>& Entries() const noexcept { return entries_; }
	const SslmAmplifyingFoldEntry* Entry(std::string_view name) const noexcept;

	// Typed per-row accessors over the SIGNED [kAmplifyingScaleExponentMin,
	// kAmplifyingScaleExponentMax] domain `ApplyAmplifyingWeightScaleFold` consumes --
	// structurally distinct from WSC1's own unsigned-domain `ApplyWeightScaleFold` reads.
	// Behavior is undefined if `row >= entry.row_count`.
	static int32_t Identity(const SslmAmplifyingFoldEntry& entry, uint64_t row) noexcept;
	static int32_t Mult(const SslmAmplifyingFoldEntry& entry, uint64_t row) noexcept;
	static int32_t Exponent(const SslmAmplifyingFoldEntry& entry, uint64_t row) noexcept;

private:
	std::vector<SslmAmplifyingFoldEntry> entries_;
};

using SslmDeltaFoldScaleView = SslmAmplifyingFoldScaleView<SslmAmplifyingFoldKind::Delta>;
using SslmUFoldScaleView = SslmAmplifyingFoldScaleView<SslmAmplifyingFoldKind::U>;

// The section type and on-disk magic each Kind requires -- the single source both Parse's own
// type-gate and a future adapter-loader's own section-table walk read, so the two can never
// independently drift.
SslmSectionType AmplifyingFoldSectionTypeFor(SslmAmplifyingFoldKind kind) noexcept;
const uint8_t* AmplifyingFoldMagicFor(SslmAmplifyingFoldKind kind) noexcept;

// Design Sec9's four explicit B0b validations, as free functions (the adapter-loading ABI that
// calls them, `sslm_adapter_map`, is outside this design's own B0-B9 scope, design Sec11's own
// stated build boundary -- these are the checks that ABI will call, specified and tested here).

// (b) signed domain (D-SLM2917, unchanged in value, run through this array kind's OWN dedicated
// validator rather than WSC1's `ValidateWeightScalesDomain`): identity in {0,1}, exponent in
// [kAmplifyingScaleExponentMin, kAmplifyingScaleExponentMax].
SslmModelStatus ValidateAmplifyingFoldScalesDomain(const std::vector<SslmAmplifyingFoldEntry>& entries,
                                                    std::string* err);

// (a) dimension: one entry's row_count against the caller-supplied expectation (out_channels
// for DeltaFoldScales, rank r for UFoldScales -- the caller reads the expectation from the base
// artifact's own geometry or the adapter's own declared rank, per design Sec9).
SslmModelStatus ValidateAmplifyingFoldDimension(const SslmAmplifyingFoldEntry& entry,
                                                 uint64_t expected_row_count, std::string* err);

// (c) projection: the entry's own declared target projection (parsed from its `name`, by the
// caller's own convention) must be one of the seven PEFT-adaptable projections AND must be
// claimed by the adapter's own target_modules.
inline constexpr std::string_view kPeftAdaptableProjections[7] = {
    "q_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "k_proj", "v_proj",
};
SslmModelStatus ValidateAmplifyingFoldProjection(std::string_view declared_projection,
                                                  const std::vector<std::string_view>& adapter_target_modules,
                                                  std::string* err);

// (d) base-hash: the entry's own declared base-artifact integrity hash (32 bytes, the same
// value `SslmArtifact::RawIntegrityHash()` exposes) must match the ACTUALLY-mapped base's own
// hash, extending `sslm_adapter_map`'s existing mismatch rejection (SuperSLM_Plan.md Sec12) to
// these two sections specifically.
SslmModelStatus ValidateAmplifyingFoldBaseHash(const std::array<uint8_t, kIntegrityHashBytes>& declared,
                                                const std::array<uint8_t, kIntegrityHashBytes>& actual,
                                                std::string* err);

// Parse the CFG1 Config section into `out`. The section must already be validated by
// SslmArtifact. Rejects (fails closed, `out` left default) on a wrong size/magic/version,
// a zero dimension, an out-of-range enum/bool, or a nonzero reserved field — the §11
// reject-over-degrade law for config. Throws only std::bad_alloc (S-HARDEN-7, F5).
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
// version/entry_count or a nonzero reserved field — the §11 reject-over-degrade law. Throws
// only std::bad_alloc (S-HARDEN-7, F5).
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
//
// Ownership (T-403 lifetime-contract fix, 2026-07-22): the view is
// self-contained. It owns its backing store (`backing_`, populated only by
// SslmModel::Load) and every pointer/`string_view` it exposes points into
// storage the view itself keeps alive for exactly as long as the view lives
// — holding the view is sufficient; there is no second object to keep alive.
// The view is movable (a move carries the backing store — `std::vector`'s
// move-preserves-buffer-address guarantee, the same invariant SslmArtifact
// already ships, artifact.h — so every exposed pointer stays valid) and
// non-copyable (its backing store is move-only; a copyable view of borrowed
// bytes was never sound). Move construction/assignment are HAND-WRITTEN, not
// `= default`: every pointer-bearing member except `sigmoid_lut` clears
// itself on an ordinary container move, but `sigmoid_lut` is a bare
// pointer+count with no container of its own, so a member-wise default move
// would copy its pointer bits onto the destination while leaving the
// source's copy — and its `has_sigmoid_lut` flag — untouched, a dangling
// pointer on the moved-from object indistinguishable from a populated one.
// The hand-written move clears `sigmoid_lut` and every `has_*` flag on the
// source explicitly, so destroying — or moving from — the view invalidates
// every view it exposed, uniformly across every pointer-bearing member.
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
	// every other keyed-constant section. S3.3 is the C27 consumer D-SLM142
	// named ("a future slot adds their domain descriptor when the C27
	// consumer lands") and closes it: their domain descriptors
	// (KvLandingScaleOutOfDomain/KvLandingReciprocalOutOfDomain,
	// ValidateKvLandingScalesDomain/ValidateKvLandingReciprocalsDomain,
	// model.cpp) are wired into SslmModel::Load, which now rejects a carried
	// value outside either derived bound.
	SslmKeyedConstants kv_landing_scales;
	bool has_kv_landing_scales = false;

	SslmKeyedConstants kv_landing_reciprocals;
	bool has_kv_landing_reciprocals = false;

	// S3.7 (§8.3): the calibration band -- a KVC1 keyed blob, parsed
	// structurally like CompositionConstants/KvLandingScales above. OPTIONAL:
	// `has_calibration_band == false` is a valid artifact (every artifact the
	// tree emits today), never a rejection on its own; `ClassifyCalibrationBand`
	// below reports `BandUnknown` for it.
	SslmKeyedConstants calibration_band;
	bool has_calibration_band = false;

	// T-2041 (Poirot c81e48c review, Significant 2): DeltaFoldScales/UFoldScales -- B0b's own
	// section types (design Sec9), parsed structurally AND value-domain-validated at load time,
	// the same two-phase gate every other typed section here already gets (WeightScales'
	// ValidateWeightScalesDomain is the sibling). OPTIONAL: `has_* == false` is a valid artifact
	// (a base-model-only artifact carries neither; an adapter artifact carries both) -- absence
	// is not a rejection on its own.
	SslmDeltaFoldScaleView delta_fold_scales;
	bool has_delta_fold_scales = false;
	SslmUFoldScaleView u_fold_scales;
	bool has_u_fold_scales = false;

	// The tokenizer join (S-HARDEN-2, F18/F6/F7/F15): present iff the artifact
	// carries BOTH the Tokenizer and UnicodeTables sections and TokenizerView::Open
	// accepted them structurally. An artifact carrying neither is a valid
	// tokenizer-less model artifact (has_tokenizer stays false, no rejection);
	// carrying exactly one of the two, or a structurally malformed TOK1/UNI1, is a
	// Load-time rejection (TokenizerRejected) -- the same fail-closed posture as
	// every other section here. When present, its VocabSize() is additionally
	// cross-checked against `config.vocab_size` (TokenizerVocabSizeMismatch on
	// disagreement) -- the TOK1 x CFG1 join two independently-parsed blobs never
	// enforced before this slot.
	TokenizerView tokenizer;
	bool has_tokenizer = false;

	// T-1894 (T-1822 design Sec31.2.1, round 4/D-SLM2423): the Option-G
	// selection bit, a property of the artifact HEADER (not a section, so no
	// `has_*` flag -- present, meaningfully, on every valid artifact, unlike
	// a section which may be legitimately absent). Set by `LoadImpl`
	// immediately after `SslmArtifact::OpenFromMemory` succeeds, from
	// `backing_.OptionGFusedKLandingEnabled()`. A consumer already holding a
	// `SslmModelView` (`tools/sslm_generate.cpp`, this build) reads this
	// field directly rather than reaching into the private `backing_` only
	// `SslmModelAccess` may touch.
	bool option_g_fused_k_landing = false;

	// The numeric-record trace hook's own state (D-SLM353): owned here, per
	// model handle, instead of a process-wide static -- the corrected reading
	// of SuperSLM_S3a_WalkingSkeleton_Plan.md §3's Layer-1-wide no-global-state
	// law applied to the mechanism trace_hook.h builds (Claude/Vitruvius/
	// SuperSLM_S3.1a_TraceHookGlobal_Ruling-2026-07-28.md). Not populated by
	// SslmModel::Load and not gated by a `has_*` flag -- a default-constructed
	// SslmTraceHookState{} is already a fully valid "no hook installed" state,
	// not a partial-load state like the sections above.
	//
	// **Load CLEARS whatever hook state this field already carries, on every
	// call, success or rejection alike.** `SslmModelAccess::LoadImpl`
	// (src/model.cpp) resets `out` -- this field included -- to
	// `SslmModelView{}` defaults before it opens the artifact, and again on
	// any rejection path; nothing re-installs the hook afterward. A caller
	// who installs a hook on a handle and then re-Loads a new artifact into
	// that same handle (to swap models) loses trace coverage silently -- the
	// call succeeds, the reload is fail-closed and correct (§11
	// reject-over-degrade: never a partial view), but the hook is gone with
	// no diagnostic of its own, because it is caller-installed state rather
	// than artifact-parsed state and Load's fail-closed reset does not
	// distinguish the two. A caller that wants tracing to survive a reload
	// re-installs the hook after `Load` returns.
	//
	// A caller reaches this field through the view it already holds and
	// passes it to RequantChainChecked (checked_chain_funnel.h) to get trace
	// emission scoped to this one handle; two model views never share
	// fn/user state. NarrowRowChecked (checked_chain_funnel.h) takes no
	// trace_hook_state parameter -- no production path emits a trace record
	// from it yet.
	SslmTraceHookState trace_hook;

	SslmModelView() = default;
	SslmModelView(SslmModelView&& other) noexcept { MoveFrom(other); }
	SslmModelView& operator=(SslmModelView&& other) noexcept {
		if (this != &other) MoveFrom(other);
		return *this;
	}
	SslmModelView(const SslmModelView&) = delete;
	SslmModelView& operator=(const SslmModelView&) = delete;

private:
	// S-HARDEN-7 (design Sec3.1): SslmModel::Load's *Impl body (which needs
	// backing_) lives in src/model.cpp's SslmModelAccess, not as a private
	// member of SslmModel — see SslmArtifact's identical SslmArtifactAccess
	// comment (artifact.h) for why. SslmModelAccess is the sole friend here;
	// SslmModel itself needs no friend access, since its only method (Load)
	// is public and never touches backing_ directly.
	friend struct SslmModelAccess;

	// The one place that carries every member across a move and then clears
	// the source. See the struct's ownership comment above for why this is
	// hand-written rather than `= default`.
	void MoveFrom(SslmModelView& other) noexcept {
		config = other.config;
		has_config = other.has_config;
		sigmoid_lut = other.sigmoid_lut;
		has_sigmoid_lut = other.has_sigmoid_lut;
		weights = std::move(other.weights);
		has_weights = other.has_weights;
		biases = std::move(other.biases);
		has_biases = other.has_biases;
		rope_tables = std::move(other.rope_tables);
		has_rope_tables = other.has_rope_tables;
		weight_scales = std::move(other.weight_scales);
		has_weight_scales = other.has_weight_scales;
		composition_constants = std::move(other.composition_constants);
		has_composition_constants = other.has_composition_constants;
		kv_landing_scales = std::move(other.kv_landing_scales);
		has_kv_landing_scales = other.has_kv_landing_scales;
		kv_landing_reciprocals = std::move(other.kv_landing_reciprocals);
		has_kv_landing_reciprocals = other.has_kv_landing_reciprocals;
		calibration_band = std::move(other.calibration_band);
		has_calibration_band = other.has_calibration_band;
		delta_fold_scales = std::move(other.delta_fold_scales);
		has_delta_fold_scales = other.has_delta_fold_scales;
		u_fold_scales = std::move(other.u_fold_scales);
		has_u_fold_scales = other.has_u_fold_scales;
		tokenizer = std::move(other.tokenizer);
		has_tokenizer = other.has_tokenizer;
		option_g_fused_k_landing = other.option_g_fused_k_landing;
		trace_hook = other.trace_hook;
		backing_ = std::move(other.backing_);

		// Clear the source. Every member above except `sigmoid_lut` is a
		// vector/unique_ptr-backed container whose ordinary move already left
		// `other` empty/null; `sigmoid_lut` has no container to do that for
		// it, so it is reset explicitly here, alongside every `has_*` flag —
		// otherwise `other.has_sigmoid_lut` would still read true over a
		// pointer `this` now exclusively owns.
		other.config = SslmModelConfig{};
		other.has_config = false;
		other.sigmoid_lut = SslmSigmoidLut{};
		other.has_sigmoid_lut = false;
		other.has_weights = false;
		other.has_biases = false;
		other.has_rope_tables = false;
		other.has_weight_scales = false;
		other.has_composition_constants = false;
		other.has_kv_landing_scales = false;
		other.has_kv_landing_reciprocals = false;
		other.has_calibration_band = false;
		other.has_delta_fold_scales = false;
		other.has_u_fold_scales = false;
		other.has_tokenizer = false;
		other.option_g_fused_k_landing = false;
		other.trace_hook = SslmTraceHookState{};
	}

	SslmArtifact backing_;  // owns the file bytes every view above points into
};

// The load-time orchestration entry point (S-HARDEN-1, D-SLM141): the one
// call site that composes SslmArtifact::OpenFromMemory, every present
// section's sub-parser, and the schema-value gate into a single pass. Throws
// only std::bad_alloc (S-HARDEN-7, F5); `out` is left at SslmModelView{}
// defaults on ANY rejection — container-level, structural sub-parse, or
// value-domain — and is populated only on full success (§11
// reject-over-degrade). `err` (if non-null) carries a diagnostic naming what
// was rejected.
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

// S3.7 (§8.3, §8.4): the calibration band's own verdict. `InBand` and the
// two out-of-band values are achievement claims (a token count is compared
// against the artifact-carried (min, max)); `BandUnknown` is the section-
// absent case (§13 dim 9's own version-stability cell), never conflated with
// an in-band or out-of-band verdict.
enum class SslmCalibrationBandVerdict : uint32_t {
	InBand = 0,
	AboveBand = 1,
	BelowBand = 2,
	BandUnknown = 3,
};

// Classifies `token_length` against `view`'s own CalibrationBand entry
// (named "token_length", the KVC1-shaped fixture family §8.3's costed table
// specifies). Returns `BandUnknown` when the section is absent
// (`!view.has_calibration_band`) -- the band is optional at the current
// container version -- **or when the section is present but carries no entry
// of that name.** That second case is a deliberate, tested policy, not a
// rejection: a present-but-misnamed CalibrationBand section is
// indistinguishable from an absent one at this call (D-SLM563,
// `TestCalibrationBandMisnamedEntryLoadsOkAndReportsUnknown`,
// `tests/test_main.cpp`); §8.3 does not define behaviour for that case, and
// the D-SLM143 pattern every other section's hostile-value handling follows
// would instead make it a load-time rejection. `min`/`max` are read as word
// 0/word 1 of the entry (§8.3's inclusive-at-both-endpoints statement:
// `token_length < min` is `BelowBand`, `token_length > max` is `AboveBand`,
// and `token_length == min` or `token_length == max` is `InBand`).
// `SslmModel::Load`'s own `ValidateCalibrationBandDomain` (model.cpp)
// already rejects a hostile band (`min > max`, `min < 0`, or `max <= 0`) at
// load time, so a successfully loaded `view` with `has_calibration_band ==
// true` always carries a well-formed, non-negative band under a
// `token_length` entry, if one is present under that name.
SslmCalibrationBandVerdict ClassifyCalibrationBand(const SslmModelView& view,
                                                    int64_t token_length) noexcept;

// S3.7 (§8.4): the decode-step status this sub-slot's mechanism declares --
// "one struct, returned by every decode step, carrying: the produced token
// (or 'pending' under a partial layer budget), the per-sequence saturation
// count, and the band verdict" (plan §8.4). Every field is excluded from
// every hash and digest per §10.2's rule -- none of the three is part of the
// decode's own numeric output. Not yet wired into `RunGreedyDecodeLoop` -- no
// cell in this sub-slot's own suite asserts a wiring that does not exist,
// matching this file's existing declared-scope convention for a struct whose
// consumer is a separate, later obligation (LayerWeights' own header comment
// states the same pattern).
struct SslmDecodeStepStatus {
	// The token this step produced, or the sentinel `-1` when the step left
	// the sequence mid-token under a partial layer budget ("pending" per
	// §8.4) -- `-1` is never a producible token id, since every host-facing
	// token id is validated non-negative against `config.vocab_size` (§9.1,
	// F-S3-8) before it can reach an output slot.
	int32_t produced_token = -1;
	// The per-sequence K/V landing saturation count (§8.2, `SequenceLayerState::kv_saturation_count`)
	// at the moment this step returned.
	uint64_t saturation_count = 0;
	SslmCalibrationBandVerdict calibration_band_verdict = SslmCalibrationBandVerdict::BandUnknown;
};

} // namespace superslm

#endif // SUPERSLM_MODEL_H

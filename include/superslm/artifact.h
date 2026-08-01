// SuperSLM artifact format (`.sslm`). Current format version is
// kArtifactFormatVersion below, the single source of truth -- not restated
// as a number here, so this comment cannot go stale the next time the
// format bumps (S-HARDEN-8; this file previously said "version 1" after the
// real format version had already moved to 2 at S-HARDEN-1).
//
// The runtime C++ loader for a converted, quantized model. This header is the
// machine-readable contract for the format specified in docs/sslm_format.md; the
// two must agree byte-for-byte. Standard library only — Layer 1 is independently
// embeddable, no third-party runtime dependency (DecisionLog D-SLM13).
//
// The loader is a trust boundary (SuperSLM_Plan.md §17 dim 2): every field of the
// file is treated as hostile input and validated against declared bounds before any
// section byte is read. Deviation is rejection with a versioned diagnostic, never a
// silent partial load (§11, reject-over-degrade).
#ifndef SUPERSLM_ARTIFACT_H
#define SUPERSLM_ARTIFACT_H

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace superslm {

// Bumped on any field-layout change, new required section, or integrity-hash change.
// v2 (S2.4): adds the required SigmoidLut (SIL1) section — the forward has no i-exp-sigmoid
// fallback once C10 is the LUT (D-SLM68), so a v1 artifact lacks a required section and a v2
// loader rejects it (UnsupportedVersion), never a silent degrade (docs/sslm_format.md Versioning).
inline constexpr uint32_t kArtifactFormatVersion = 2;

// Header/table geometry (v1). See docs/sslm_format.md "Byte layout".
inline constexpr uint32_t kHeaderBytes = 64;
inline constexpr uint32_t kSectionDescBytes = 40;
inline constexpr uint32_t kMaxSections = 4096;
inline constexpr uint32_t kIntegrityHashOffset = 32; // where integrity_sha256 begins
inline constexpr uint32_t kIntegrityHashBytes = 32;
inline constexpr uint8_t kMagic[4] = {'S', 'S', 'L', 'M'};

enum class SslmSectionType : uint32_t {
	Config = 0,
	Provenance = 1,
	Weights = 2,
	Biases = 3,
	RopeTables = 4,
	Scales = 5,
	WeightScales = 6,
	CompositionConstants = 7,
	KvLandingScales = 8,
	KvLandingReciprocals = 9,
	Calibration = 10,
	GoldenHashes = 11,
	SigmoidLut = 12,    // S2: fixed-point SiLU sigmoid LUT (SIL1); required from v2 (C10, D-SLM68)
	Tokenizer = 20,     // S1: byte-BPE vocab + merges + special tokens
	ChatTemplate = 21,  // S1: chat template + special-token metadata (JSON)
	UnicodeTables = 22, // S1: pinned NFC + \p{L}/\p{N}/\s tables
	// Reserved — introduced at a later slot.
	SchemaMasks = 30,   // S5
	// S3.7 (§8.3): the token-length calibration band -- a KVC1 keyed blob,
	// like CompositionConstants/KvLandingScales/KvLandingReciprocals above,
	// carrying one entry's (min, max) token-count pair. OPTIONAL at the
	// current container version, not a CFG1 extension and not a version
	// bump -- absence loads exactly as before this type existed.
	CalibrationBand = 31,
};

enum class SslmDtype : uint32_t {
	Raw = 0,     // byte stream (JSON sections use this)
	Int8 = 1,
	Int32 = 2,
	Int64 = 3,
	Uint8 = 4,
	Float32 = 5,
	Float64 = 6,
};

// Byte width of a dtype element; 0 for an unknown dtype value.
uint32_t DtypeSize(uint32_t dtype) noexcept;

// True if the value names a section type the v1 loader recognizes.
bool IsKnownSectionType(uint32_t type) noexcept;

// The dtype every section of a given known type must carry (docs/sslm_format.md
// section-types table — normative). Only meaningful once IsKnownSectionType(type)
// holds; returns SslmDtype::Raw for anything else.
SslmDtype ExpectedDtype(uint32_t type) noexcept;

// Every way a `.sslm` can be rejected. `Ok` is the only non-error value.
enum class SslmStatus {
	Ok = 0,
	NullData,            // OpenFromMemory: data == nullptr
	NullPath,            // OpenFromFile: path == nullptr
	Truncated,           // file smaller than the header/table it declares
	BadMagic,            // first four bytes are not "SSLM"
	UnsupportedVersion,  // format_version != kArtifactFormatVersion
	BadHeader,           // header_bytes wrong, or a reserved field nonzero
	TooManySections,     // section_count > kMaxSections
	FileSizeMismatch,    // header file_bytes != actual size
	BadAlignment,        // a section alignment is not a power of two in [8,4096]
	Misaligned,          // a section offset is not a multiple of its alignment
	SectionOutOfBounds,  // a section range exceeds the file (or overflows)
	SectionOverlap,      // a section overlaps the header/table or another section
	BadDtype,            // a section dtype is unknown
	SectionDtypeMismatch, // a known dtype, but not the one this section type requires
	SizeMismatch,        // byte_size != elem_count * dtype_size
	UnknownSection,      // a section type outside the v1 set
	DuplicateSection,    // the same section type appears twice
	MissingSection,      // a required section for this format_version is absent
	IntegrityMismatch,   // recomputed SHA-256 != stored integrity hash
	IoError,             // the file could not be read (OpenFromFile only)
};

// Human-readable name for a status, for diagnostics and test messages.
const char* SslmStatusName(SslmStatus s) noexcept;

inline constexpr uint32_t kNoSection = 0xFFFFFFFFu;

// A rejection: the code, which section tripped it (or kNoSection), and a message
// carrying the offending values.
struct SslmError {
	SslmStatus code = SslmStatus::Ok;
	uint32_t section_index = kNoSection;
	std::string message;
};

// A validated, in-place view of one section's bytes. `data` points into the
// artifact's owned buffer and is valid for the artifact's lifetime.
struct SslmSectionView {
	SslmSectionType type{};
	SslmDtype dtype{};
	const uint8_t* data = nullptr;
	uint64_t byte_size = 0;
	uint64_t elem_count = 0;
	uint32_t alignment = 0;
};

// A loaded, fully validated `.sslm`. Constructed only through Open*; a default
// instance is empty and Ok()==false.
class SslmArtifact {
public:
	SslmArtifact() = default;

	// Owns a copy of the artifact bytes; section views point into that buffer. A
	// copy would deep-copy the bytes to a new address while the views still pointed
	// at the source, so copy is deleted. Move transfers the vector's buffer (its
	// address is preserved), so the views stay valid across a move.
	SslmArtifact(const SslmArtifact&) = delete;
	SslmArtifact& operator=(const SslmArtifact&) = delete;
	SslmArtifact(SslmArtifact&&) = default;
	SslmArtifact& operator=(SslmArtifact&&) = default;

	// Validate `size` bytes at `data` as a v1 `.sslm`. On Ok, `out` owns a copy of
	// the bytes and exposes the header + sections. On any error, `out` is left empty
	// and `err` (if non-null) carries the diagnostic. Throws only std::bad_alloc
	// (S-HARDEN-7, F5); never reads a section byte before the file passes every
	// structural check. `data == nullptr` is rejected explicitly (NullData) before
	// any other check, regardless of `size` — the caller's null pointer is never
	// dereferenced (F14).
	static SslmStatus OpenFromMemory(const uint8_t* data, size_t size,
	                                 SslmArtifact& out, SslmError* err);

	// Read the file at `path`, then OpenFromMemory. `path == nullptr` is rejected
	// explicitly (NullPath) before any file-system call. IoError if the file is
	// unreadable. Throws only std::bad_alloc (S-HARDEN-7, F5).
	static SslmStatus OpenFromFile(const char* path, SslmArtifact& out, SslmError* err);

	bool Ok() const noexcept { return ok_; }
	uint32_t FormatVersion() const noexcept { return format_version_; }
	uint64_t FileBytes() const noexcept { return file_bytes_; }

	// Lowercase hex of the stored integrity hash (the artifact's fingerprint).
	// Throws only std::bad_alloc (S-HARDEN-7, F5).
	std::string FingerprintHex() const;

	const std::vector<SslmSectionView>& Sections() const noexcept { return sections_; }

	// The section of the given type, or nullptr if absent.
	const SslmSectionView* Section(SslmSectionType type) const noexcept;

private:
	// S-HARDEN-7: grants src/artifact.cpp's SslmArtifactAccess (defined only
	// there, never declared here) access to the private members below, so
	// this class's *Impl bodies can live entirely in the .cpp rather than as
	// private member declarations in this header. A private member
	// declaration here would itself be a public-C++-surface function the
	// membership-check AST walk (S-HARDEN-7, design Sec3.1) would derive as
	// a member of its own population — access specifiers are invisible to
	// that walk, which is why the *Impl bodies must not be declared here at
	// all, not merely marked private. A friend `struct` declaration is a
	// FriendDecl in the AST, not a FunctionDecl/CXXMethodDecl, so it can
	// never be picked up by that walk regardless of the rule's conditions.
	friend struct SslmArtifactAccess;

	bool ok_ = false;
	uint32_t format_version_ = 0;
	uint64_t file_bytes_ = 0;
	uint8_t integrity_[kIntegrityHashBytes] = {};
	std::vector<uint8_t> bytes_;              // owned copy of the whole file
	std::vector<SslmSectionView> sections_;   // views into bytes_
};

} // namespace superslm

#endif // SUPERSLM_ARTIFACT_H

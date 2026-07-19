#include "superslm/artifact.h"

#include "superslm/sha256.h"

#include <cstring>
#include <fstream>
#include <string>
#include <vector>

namespace superslm {

uint32_t DtypeSize(uint32_t dtype) noexcept {
	switch (static_cast<SslmDtype>(dtype)) {
		case SslmDtype::Raw: return 1;
		case SslmDtype::Int8: return 1;
		case SslmDtype::Int32: return 4;
		case SslmDtype::Int64: return 8;
		case SslmDtype::Uint8: return 1;
		case SslmDtype::Float32: return 4;
		case SslmDtype::Float64: return 8;
	}
	return 0; // unknown
}

bool IsKnownSectionType(uint32_t type) noexcept {
	switch (static_cast<SslmSectionType>(type)) {
		case SslmSectionType::Config:
		case SslmSectionType::Provenance:
		case SslmSectionType::Weights:
		case SslmSectionType::Biases:
		case SslmSectionType::RopeTables:
		case SslmSectionType::Scales:
		case SslmSectionType::WeightScales:
		case SslmSectionType::CompositionConstants:
		case SslmSectionType::KvLandingScales:
		case SslmSectionType::KvLandingReciprocals:
		case SslmSectionType::Calibration:
		case SslmSectionType::GoldenHashes:
		case SslmSectionType::Tokenizer:
		case SslmSectionType::ChatTemplate:
		case SslmSectionType::SchemaMasks:
			return true;
	}
	return false;
}

SslmDtype ExpectedDtype(uint32_t type) noexcept {
	switch (static_cast<SslmSectionType>(type)) {
		case SslmSectionType::Weights: return SslmDtype::Int8;
		case SslmSectionType::Biases: return SslmDtype::Int32;
		case SslmSectionType::RopeTables: return SslmDtype::Int64;
		// All JSON / opaque-byte sections are Raw.
		case SslmSectionType::Config:
		case SslmSectionType::Provenance:
		case SslmSectionType::Scales:
		case SslmSectionType::WeightScales:
		case SslmSectionType::CompositionConstants:
		case SslmSectionType::KvLandingScales:
		case SslmSectionType::KvLandingReciprocals:
		case SslmSectionType::Calibration:
		case SslmSectionType::GoldenHashes:
		case SslmSectionType::Tokenizer:
		case SslmSectionType::ChatTemplate:
		case SslmSectionType::SchemaMasks:
			return SslmDtype::Raw;
	}
	return SslmDtype::Raw; // unknown type — callers gate on IsKnownSectionType first
}

const char* SslmStatusName(SslmStatus s) noexcept {
	switch (s) {
		case SslmStatus::Ok: return "Ok";
		case SslmStatus::Truncated: return "Truncated";
		case SslmStatus::BadMagic: return "BadMagic";
		case SslmStatus::UnsupportedVersion: return "UnsupportedVersion";
		case SslmStatus::BadHeader: return "BadHeader";
		case SslmStatus::TooManySections: return "TooManySections";
		case SslmStatus::FileSizeMismatch: return "FileSizeMismatch";
		case SslmStatus::BadAlignment: return "BadAlignment";
		case SslmStatus::Misaligned: return "Misaligned";
		case SslmStatus::SectionOutOfBounds: return "SectionOutOfBounds";
		case SslmStatus::SectionOverlap: return "SectionOverlap";
		case SslmStatus::BadDtype: return "BadDtype";
		case SslmStatus::SectionDtypeMismatch: return "SectionDtypeMismatch";
		case SslmStatus::SizeMismatch: return "SizeMismatch";
		case SslmStatus::UnknownSection: return "UnknownSection";
		case SslmStatus::DuplicateSection: return "DuplicateSection";
		case SslmStatus::MissingSection: return "MissingSection";
		case SslmStatus::IntegrityMismatch: return "IntegrityMismatch";
		case SslmStatus::IoError: return "IoError";
	}
	return "?";
}

namespace {

// Little-endian field reads, matching docs/sslm_format.md choice 1. The loader
// never reinterpret_casts a struct over untrusted bytes — it assembles each field
// byte-by-byte, so host endianness and padding cannot leak into a trust boundary.
uint32_t RdU32(const uint8_t* p) noexcept {
	return uint32_t(p[0]) | (uint32_t(p[1]) << 8) | (uint32_t(p[2]) << 16) | (uint32_t(p[3]) << 24);
}
uint64_t RdU64(const uint8_t* p) noexcept {
	uint64_t v = 0;
	for (int i = 0; i < 8; ++i) v |= uint64_t(p[i]) << (8 * i);
	return v;
}

bool IsPowerOfTwo(uint32_t v) noexcept { return v != 0 && (v & (v - 1)) == 0; }

SslmStatus Reject(SslmError* err, SslmStatus code, uint32_t section, std::string msg) {
	if (err) {
		err->code = code;
		err->section_index = section;
		err->message = std::move(msg);
	}
	return code;
}

struct Placed {
	uint32_t type;
	uint32_t dtype;
	uint64_t offset;
	uint64_t byte_size;
	uint64_t elem_count;
	uint32_t alignment;
};

} // namespace

SslmStatus SslmArtifact::OpenFromMemory(const uint8_t* data, size_t size,
                                        SslmArtifact& out, SslmError* err) {
	out = SslmArtifact{};

	// --- Header (all checks before a single section byte is examined) ---
	if (size < kHeaderBytes) {
		return Reject(err, SslmStatus::Truncated, kNoSection,
		              "file " + std::to_string(size) + " bytes < " +
		                  std::to_string(kHeaderBytes) + "-byte header");
	}
	if (std::memcmp(data, kMagic, 4) != 0) {
		return Reject(err, SslmStatus::BadMagic, kNoSection, "first four bytes are not 'SSLM'");
	}
	const uint32_t version = RdU32(data + 4);
	if (version != kArtifactFormatVersion) {
		return Reject(err, SslmStatus::UnsupportedVersion, kNoSection,
		              "format_version " + std::to_string(version) + " != " +
		                  std::to_string(kArtifactFormatVersion));
	}
	const uint32_t header_bytes = RdU32(data + 8);
	if (header_bytes != kHeaderBytes) {
		return Reject(err, SslmStatus::BadHeader, kNoSection,
		              "header_bytes " + std::to_string(header_bytes) + " != " +
		                  std::to_string(kHeaderBytes));
	}
	const uint32_t section_count = RdU32(data + 12);
	const uint32_t flags = RdU32(data + 16);
	const uint32_t reserved0 = RdU32(data + 20);
	if (flags != 0) {
		return Reject(err, SslmStatus::BadHeader, kNoSection,
		              "flags must be 0, got " + std::to_string(flags));
	}
	if (reserved0 != 0) {
		return Reject(err, SslmStatus::BadHeader, kNoSection,
		              "reserved0 must be 0, got " + std::to_string(reserved0));
	}
	// F-1: bound section_count before any per-row work, so a hostile count cannot
	// drive the table-fit arithmetic or a huge loop.
	if (section_count > kMaxSections) {
		return Reject(err, SslmStatus::TooManySections, kNoSection,
		              "section_count " + std::to_string(section_count) + " > " +
		                  std::to_string(kMaxSections));
	}
	const uint64_t table_end =
	    uint64_t(kHeaderBytes) + uint64_t(section_count) * kSectionDescBytes;
	if (table_end > size) {
		return Reject(err, SslmStatus::Truncated, kNoSection,
		              "section table ends at " + std::to_string(table_end) +
		                  " > file size " + std::to_string(size));
	}
	const uint64_t file_bytes = RdU64(data + 24);
	if (file_bytes != size) {
		return Reject(err, SslmStatus::FileSizeMismatch, kNoSection,
		              "header file_bytes " + std::to_string(file_bytes) +
		                  " != actual " + std::to_string(size));
	}

	// --- Integrity: the whole file, the 32 hash bytes treated as zero. Verified
	//     before any section is interpreted (the bytes are already length-sane). ---
	{
		Sha256 h;
		const uint8_t zero[kIntegrityHashBytes] = {};
		h.Update(data, kIntegrityHashOffset);
		h.Update(zero, kIntegrityHashBytes);
		h.Update(data + kIntegrityHashOffset + kIntegrityHashBytes,
		         size - (kIntegrityHashOffset + kIntegrityHashBytes));
		uint8_t digest[32];
		h.Final(digest);
		if (std::memcmp(digest, data + kIntegrityHashOffset, kIntegrityHashBytes) != 0) {
			return Reject(err, SslmStatus::IntegrityMismatch, kNoSection,
			              "integrity SHA-256 does not match stored hash");
		}
	}

	// --- Per-section descriptors ---
	std::vector<Placed> placed;
	placed.reserve(section_count);
	for (uint32_t i = 0; i < section_count; ++i) {
		const uint8_t* row = data + kHeaderBytes + size_t(i) * kSectionDescBytes;
		const uint32_t type = RdU32(row + 0);
		const uint32_t dtype = RdU32(row + 4);
		const uint64_t offset = RdU64(row + 8);
		const uint64_t byte_size = RdU64(row + 16);
		const uint64_t elem_count = RdU64(row + 24);
		const uint32_t alignment = RdU32(row + 32);
		const uint32_t reserved = RdU32(row + 36);

		if (reserved != 0) {
			return Reject(err, SslmStatus::BadHeader, i,
			              "section reserved field must be 0, got " + std::to_string(reserved));
		}
		if (!IsPowerOfTwo(alignment) || alignment < 8 || alignment > 4096) {
			return Reject(err, SslmStatus::BadAlignment, i,
			              "alignment " + std::to_string(alignment) +
			                  " is not a power of two in [8,4096]");
		}
		if (offset % alignment != 0) {
			return Reject(err, SslmStatus::Misaligned, i,
			              "offset " + std::to_string(offset) + " not a multiple of alignment " +
			                  std::to_string(alignment));
		}
		const uint32_t dsize = DtypeSize(dtype);
		if (dsize == 0) {
			return Reject(err, SslmStatus::BadDtype, i, "unknown dtype " + std::to_string(dtype));
		}
		// byte_size == elem_count * dsize, guarding the multiply against overflow.
		if (elem_count > UINT64_MAX / dsize || elem_count * dsize != byte_size) {
			return Reject(err, SslmStatus::SizeMismatch, i,
			              "byte_size " + std::to_string(byte_size) + " != elem_count " +
			                  std::to_string(elem_count) + " * dtype_size " + std::to_string(dsize));
		}
		if (offset > UINT64_MAX - byte_size || offset + byte_size > size) {
			return Reject(err, SslmStatus::SectionOutOfBounds, i,
			              "section [" + std::to_string(offset) + "," +
			                  std::to_string(offset) + "+" + std::to_string(byte_size) +
			                  ") exceeds file size " + std::to_string(size));
		}
		if (offset < table_end) {
			return Reject(err, SslmStatus::SectionOverlap, i,
			              "offset " + std::to_string(offset) +
			                  " overlaps the header/table (ends at " + std::to_string(table_end) + ")");
		}
		if (!IsKnownSectionType(type)) {
			return Reject(err, SslmStatus::UnknownSection, i, "unknown section type " + std::to_string(type));
		}
		// type and dtype are each known here; the pairing is normative (choice 6).
		if (static_cast<SslmDtype>(dtype) != ExpectedDtype(type)) {
			return Reject(err, SslmStatus::SectionDtypeMismatch, i,
			              "section type " + std::to_string(type) + " requires dtype " +
			                  std::to_string(static_cast<uint32_t>(ExpectedDtype(type))) +
			                  ", got " + std::to_string(dtype));
		}
		placed.push_back(Placed{type, dtype, offset, byte_size, elem_count, alignment});
	}

	// --- Cross-section: duplicates, then overlap, then the required section. ---
	for (size_t a = 0; a < placed.size(); ++a) {
		for (size_t b = a + 1; b < placed.size(); ++b) {
			if (placed[a].type == placed[b].type) {
				return Reject(err, SslmStatus::DuplicateSection, static_cast<uint32_t>(a),
				              "section type " + std::to_string(placed[a].type) +
				                  " appears at rows " + std::to_string(a) + " and " + std::to_string(b));
			}
		}
	}
	for (size_t a = 0; a < placed.size(); ++a) {
		for (size_t b = a + 1; b < placed.size(); ++b) {
			const Placed& x = placed[a];
			const Placed& y = placed[b];
			if (x.byte_size == 0 || y.byte_size == 0) continue; // an empty section occupies no bytes
			if (x.offset < y.offset + y.byte_size && y.offset < x.offset + x.byte_size) {
				return Reject(err, SslmStatus::SectionOverlap, static_cast<uint32_t>(b),
				              "section rows " + std::to_string(a) + " and " + std::to_string(b) +
				                  " occupy overlapping byte ranges");
			}
		}
	}
	bool has_config = false;
	for (const Placed& p : placed) {
		if (static_cast<SslmSectionType>(p.type) == SslmSectionType::Config) has_config = true;
	}
	if (!has_config) {
		return Reject(err, SslmStatus::MissingSection, kNoSection,
		              "required Config section is absent");
	}

	// --- Accepted: take ownership of the bytes and build views into them. bytes_
	//     is assigned once and never resized, so the view pointers stay valid. ---
	out.bytes_.assign(data, data + size);
	out.format_version_ = version;
	out.file_bytes_ = file_bytes;
	std::memcpy(out.integrity_, data + kIntegrityHashOffset, kIntegrityHashBytes);
	out.sections_.reserve(placed.size());
	for (const Placed& p : placed) {
		SslmSectionView v;
		v.type = static_cast<SslmSectionType>(p.type);
		v.dtype = static_cast<SslmDtype>(p.dtype);
		v.data = out.bytes_.data() + p.offset;
		v.byte_size = p.byte_size;
		v.elem_count = p.elem_count;
		v.alignment = p.alignment;
		out.sections_.push_back(v);
	}
	out.ok_ = true;
	if (err) {
		err->code = SslmStatus::Ok;
		err->section_index = kNoSection;
		err->message.clear();
	}
	return SslmStatus::Ok;
}

SslmStatus SslmArtifact::OpenFromFile(const char* path, SslmArtifact& out, SslmError* err) {
	out = SslmArtifact{};
	std::ifstream f(path, std::ios::binary | std::ios::ate);
	if (!f) {
		return Reject(err, SslmStatus::IoError, kNoSection,
		              std::string("cannot open file: ") + path);
	}
	const std::streamoff end = f.tellg();
	if (end < 0) {
		return Reject(err, SslmStatus::IoError, kNoSection,
		              std::string("cannot size file: ") + path);
	}
	std::vector<uint8_t> buf(static_cast<size_t>(end));
	f.seekg(0);
	if (end > 0 && !f.read(reinterpret_cast<char*>(buf.data()), end)) {
		return Reject(err, SslmStatus::IoError, kNoSection,
		              std::string("cannot read file: ") + path);
	}
	return OpenFromMemory(buf.data(), buf.size(), out, err);
}

std::string SslmArtifact::FingerprintHex() const {
	return ToHex(integrity_);
}

const SslmSectionView* SslmArtifact::Section(SslmSectionType type) const noexcept {
	for (const auto& s : sections_) {
		if (s.type == type) return &s;
	}
	return nullptr;
}

} // namespace superslm

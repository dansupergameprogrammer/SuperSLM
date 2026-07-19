// Curie's S0 loader red suite — spec-faithful `.sslm` fixture builder.
//
// Builds byte buffers per docs/sslm_format.md "Byte layout" directly (explicit
// little-endian field writes; the SHA-256 integrity hash computed with the hash
// field zeroed, per the format's load-bearing choice 2), independent of
// superslm::SslmArtifact::OpenFromMemory — the loader under test. A rejection
// cell takes a structurally valid buffer from BuildArtifact() and mutates exactly
// one field, so the byte buffer differs from a valid artifact in exactly the one
// way the cell names. Standard library only, matching the project's Layer-1 rule.
#ifndef SUPERSLM_TESTS_SSLM_FIXTURES_H
#define SUPERSLM_TESTS_SSLM_FIXTURES_H

#include "superslm/artifact.h"
#include "superslm/sha256.h"

#include <cstdint>
#include <cstring>
#include <vector>

namespace superslm_test {

// --- little-endian field access, matching docs/sslm_format.md exactly ---

inline void PutU32(std::vector<uint8_t>& b, size_t off, uint32_t v) {
	for (int i = 0; i < 4; ++i) b[off + i] = static_cast<uint8_t>((v >> (8 * i)) & 0xFF);
}
inline void PutU64(std::vector<uint8_t>& b, size_t off, uint64_t v) {
	for (int i = 0; i < 8; ++i) b[off + i] = static_cast<uint8_t>((v >> (8 * i)) & 0xFF);
}
inline uint32_t GetU32(const std::vector<uint8_t>& b, size_t off) {
	uint32_t v = 0;
	for (int i = 0; i < 4; ++i) v |= static_cast<uint32_t>(b[off + i]) << (8 * i);
	return v;
}
inline uint64_t GetU64(const std::vector<uint8_t>& b, size_t off) {
	uint64_t v = 0;
	for (int i = 0; i < 8; ++i) v |= static_cast<uint64_t>(b[off + i]) << (8 * i);
	return v;
}

// Recomputes the SHA-256 integrity hash over `bytes` (the hash field zeroed while
// hashing, docs/sslm_format.md load-bearing choice 2) and writes it back. Called
// after every deliberate mutation EXCEPT the IntegrityMismatch cell itself (which
// must leave a stale hash), so every other cell isolates its one named defect.
inline void RecomputeIntegrityHash(std::vector<uint8_t>& bytes) {
	std::vector<uint8_t> tmp = bytes;
	std::memset(tmp.data() + superslm::kIntegrityHashOffset, 0, superslm::kIntegrityHashBytes);
	uint8_t digest[32];
	superslm::Sha256Hash(tmp.data(), tmp.size(), digest);
	std::memcpy(bytes.data() + superslm::kIntegrityHashOffset, digest, superslm::kIntegrityHashBytes);
}

inline uint64_t AlignUp(uint64_t v, uint64_t a) {
	return a == 0 ? v : (v + (a - 1)) / a * a;
}

// Sentinels meaning "derive this field the spec-normal way" on a FixtureSection.
inline constexpr uint64_t kAuto64 = UINT64_MAX;

// One section to bake into a fixture artifact. `type`/`dtype` are raw values (not
// the enum) so a cell can name a value outside the known set. The three overrides
// let a cell state a byte_size / elem_count / offset that disagrees with the
// spec-normal derivation from `data`, to target exactly one field; left at
// kAuto64, each derives the way a correctly-converted artifact would.
struct FixtureSection {
	uint32_t type;
	uint32_t dtype;
	std::vector<uint8_t> data;
	uint32_t alignment = 64;
	uint64_t byte_size_override = kAuto64;
	uint64_t elem_count_override = kAuto64;
	uint64_t offset_override = kAuto64;
};

struct PlacedSection {
	uint32_t type;
	uint32_t dtype;
	uint64_t offset;
	uint64_t byte_size;
	uint64_t elem_count;
	uint32_t alignment;
};

struct BuiltArtifact {
	std::vector<uint8_t> bytes;
	std::vector<PlacedSection> placed;  // parallel to the `sections` passed in, table order
};

// Assembles a structurally valid v1 `.sslm`: a 64-byte header, a
// `sections.size()`-row section table (rows written in the given order), then
// each section's bytes. A section with no offset_override is auto-placed
// ascending, aligned to its own alignment, immediately after the previous
// section (the first after the table). file_bytes and the integrity hash are set
// correctly for the bytes actually produced — the returned buffer loads Ok
// against a correct loader unless a caller mutates it afterward.
inline BuiltArtifact BuildArtifact(const std::vector<FixtureSection>& sections) {
	BuiltArtifact out;
	const uint32_t section_count = static_cast<uint32_t>(sections.size());
	const uint64_t table_end =
	    superslm::kHeaderBytes + static_cast<uint64_t>(section_count) * superslm::kSectionDescBytes;

	std::vector<uint64_t> offsets(section_count), byte_sizes(section_count), elem_counts(section_count);
	uint64_t cursor = table_end;
	uint64_t max_end = table_end;
	for (uint32_t i = 0; i < section_count; ++i) {
		const auto& s = sections[i];
		const uint64_t bsize = s.byte_size_override == kAuto64 ? s.data.size() : s.byte_size_override;
		const uint32_t dsize = superslm::DtypeSize(s.dtype);
		const uint64_t ecount = s.elem_count_override == kAuto64
		                             ? (dsize > 0 ? bsize / dsize : 0)
		                             : s.elem_count_override;
		const uint64_t off = s.offset_override == kAuto64 ? AlignUp(cursor, s.alignment) : s.offset_override;
		offsets[i] = off;
		byte_sizes[i] = bsize;
		elem_counts[i] = ecount;
		if (s.offset_override == kAuto64) cursor = off + bsize;
		if (off + bsize > max_end) max_end = off + bsize;
	}

	out.bytes.assign(static_cast<size_t>(max_end), 0);

	std::memcpy(out.bytes.data() + 0, superslm::kMagic, 4);
	PutU32(out.bytes, 4, superslm::kArtifactFormatVersion);
	PutU32(out.bytes, 8, superslm::kHeaderBytes);
	PutU32(out.bytes, 12, section_count);
	PutU32(out.bytes, 16, 0);  // flags — reserved, == 0
	PutU32(out.bytes, 20, 0);  // reserved0 — == 0
	PutU64(out.bytes, 24, max_end);  // file_bytes

	for (uint32_t i = 0; i < section_count; ++i) {
		const size_t row = superslm::kHeaderBytes + static_cast<size_t>(i) * superslm::kSectionDescBytes;
		PutU32(out.bytes, row + 0, sections[i].type);
		PutU32(out.bytes, row + 4, sections[i].dtype);
		PutU64(out.bytes, row + 8, offsets[i]);
		PutU64(out.bytes, row + 16, byte_sizes[i]);
		PutU64(out.bytes, row + 24, elem_counts[i]);
		PutU32(out.bytes, row + 32, sections[i].alignment);
		PutU32(out.bytes, row + 36, 0);  // reserved — == 0
		if (!sections[i].data.empty()) {
			std::memcpy(out.bytes.data() + offsets[i], sections[i].data.data(), sections[i].data.size());
		}
		out.placed.push_back(PlacedSection{sections[i].type, sections[i].dtype, offsets[i], byte_sizes[i],
		                                    elem_counts[i], sections[i].alignment});
	}

	RecomputeIntegrityHash(out.bytes);
	return out;
}

inline FixtureSection MakeSection(superslm::SslmSectionType type, superslm::SslmDtype dtype,
                                   std::vector<uint8_t> data, uint32_t alignment = 64) {
	FixtureSection s;
	s.type = static_cast<uint32_t>(type);
	s.dtype = static_cast<uint32_t>(dtype);
	s.data = std::move(data);
	s.alignment = alignment;
	return s;
}

// A minimal valid Config section. Content is opaque Raw (UTF-8 JSON) bytes — no
// cell asserts its meaning, only its presence, dtype, and byte range, matching
// docs/sslm_format.md ("Json sections use dtype Raw").
inline FixtureSection MakeConfigSection() {
	return MakeSection(superslm::SslmSectionType::Config, superslm::SslmDtype::Raw, {'{', '}'});
}

// Little-endian element encoders — a converter never emits host-endian bytes, so
// fixtures built for the loader must not either (docs/sslm_format.md choice 1).
inline std::vector<uint8_t> EncodeInt8(const std::vector<int8_t>& v) {
	std::vector<uint8_t> out(v.size());
	for (size_t i = 0; i < v.size(); ++i) out[i] = static_cast<uint8_t>(v[i]);
	return out;
}
inline std::vector<uint8_t> EncodeInt32LE(const std::vector<int32_t>& v) {
	std::vector<uint8_t> out(v.size() * 4);
	for (size_t i = 0; i < v.size(); ++i) {
		const uint32_t u = static_cast<uint32_t>(v[i]);
		for (int b = 0; b < 4; ++b) out[i * 4 + static_cast<size_t>(b)] = static_cast<uint8_t>((u >> (8 * b)) & 0xFF);
	}
	return out;
}
inline std::vector<uint8_t> EncodeInt64LE(const std::vector<int64_t>& v) {
	std::vector<uint8_t> out(v.size() * 8);
	for (size_t i = 0; i < v.size(); ++i) {
		const uint64_t u = static_cast<uint64_t>(v[i]);
		for (int b = 0; b < 8; ++b) out[i * 8 + static_cast<size_t>(b)] = static_cast<uint8_t>((u >> (8 * b)) & 0xFF);
	}
	return out;
}

}  // namespace superslm_test

#endif  // SUPERSLM_TESTS_SSLM_FIXTURES_H

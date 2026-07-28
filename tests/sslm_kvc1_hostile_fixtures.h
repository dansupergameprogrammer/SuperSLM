// Curie's S2.0b red suite — spec-faithful KVC1 keyed-constant-blob builder
// (SuperSLM_Plan.md S2.0b; docs/sslm_format.md "Keyed numeric-constant blob —
// KVC1"). `SslmKeyedConstants::Parse` parses one keyed-constant section's
// self-contained KVC1 table AFTER SslmArtifact has verified whole-file
// structure and integrity — like WGT1/BIA1/ROP1 (sslm_model_hostile_
// fixtures.h) and TOK1/UNI1 before it, a crafted integrity-valid artifact can
// still carry a malformed KVC1 blob inside a validated section, so this
// sub-parse is its own hostile-input trust boundary, held to the same T-129
// bar.
//
// The three keyed-constant section types (CompositionConstants,
// KvLandingScales, KvLandingReciprocals) share one KVC1 layout; only the
// section type's REQUIRED value_words (2 or 3, ExpectedValueWords in
// model.h) differs. `BuildKvc1` is written directly against the spec's field
// table, independent of `src/model.cpp` (the code under test) — derived from
// the spec, never from reading the parser's own implementation. A rejection
// cell takes a small spec-faithful KVC1 blob and mutates exactly one header or
// descriptor field via `ConstantsDesc*Off`/`PutU32`/`PutU64` (from
// sslm_fixtures.h), so the byte buffer differs from a valid blob in exactly
// the one way the cell names.
//
// KVC1's byte layout order is header, descriptors, VALUES, then name_blob —
// note this differs from WGT1/BIA1/ROP1's header/descriptors/name_blob/data
// order (docs/sslm_format.md's KVC1 field table lists `values` before
// `name_blob`); BuildKvc1 follows KVC1's own order, not the tensor
// manifest's.
#ifndef SUPERSLM_TESTS_SSLM_KVC1_HOSTILE_FIXTURES_H
#define SUPERSLM_TESTS_SSLM_KVC1_HOSTILE_FIXTURES_H

#include "sslm_fixtures.h"
#include "sslm_model_hostile_fixtures.h"  // WriteU32LE/WriteU64LE/WriteBytes
#include "superslm/artifact.h"
#include "superslm/model.h"

#include <cstdint>
#include <string>
#include <vector>

namespace superslm_test {

// --- Fixed-formula byte offsets from docs/sslm_format.md's KVC1 field table
//     — not derived from the parser under test. Header fields are constants;
//     descriptor fields are a function of the entry's index `i` (each
//     EntryDesc is a fixed kConstantDescBytes==8 bytes, starting at
//     kConstantHeaderBytes==24). ---

inline constexpr size_t kConstantsMagicOff = 0;
inline constexpr size_t kConstantsVersionOff = 4;
inline constexpr size_t kConstantsEntryCountOff = 8;
inline constexpr size_t kConstantsValueWordsOff = 12;
inline constexpr size_t kConstantsNameBlobLenOff = 16;
inline constexpr size_t kConstantsReservedOff = 20;

inline size_t ConstantsDescOff(size_t i) {
	return superslm::kConstantHeaderBytes + i * superslm::kConstantDescBytes;
}
inline size_t ConstantsDescNameOffOff(size_t i) { return ConstantsDescOff(i) + 0; }
inline size_t ConstantsDescNameLenOff(size_t i) { return ConstantsDescOff(i) + 4; }

// One entry to bake into a KVC1 blob: a key and its `value_words`-length
// int64 tuple. `values.size()` must equal the blob's `value_words`.
struct Kvc1EntrySpec {
	std::string name;
	std::vector<int64_t> values;
};

struct BuiltKvc1 {
	std::vector<uint8_t> bytes;
	uint32_t entry_count = 0;
	uint32_t value_words = 0;
	uint32_t name_blob_len = 0;
	size_t descriptors_off = 0;
	size_t values_off = 0;
	size_t name_blob_off = 0;

	// Byte offset of entry `i`'s value `w` (< value_words) in `bytes` — the
	// int64 at that offset is little-endian (WriteU64LE/GetU64).
	size_t ValueOff(size_t i, size_t w) const { return values_off + (i * value_words + w) * 8; }
};

// Builds a spec-faithful KVC1 blob (docs/sslm_format.md "Keyed numeric-
// constant blob — KVC1"): 24-byte header, `entries.size()` fixed 8-byte
// descriptors, the `entry_count * value_words` int64 value array (row-major,
// entry i's tuple at `[i*value_words, (i+1)*value_words)`), then the
// concatenated name blob. `declared_value_words` is written into the header
// field verbatim — normally `entries[i].values.size()` for every i, but a
// hostile cell may pass a `declared_value_words` that disagrees (the caller
// is then responsible for whether the value array itself still matches; the
// BadValueWords cells build entries whose `values.size()` already equals
// `declared_value_words`, so only the header field's *meaning* — right count,
// wrong section type — is what's being tested, not a structural mismatch).
inline BuiltKvc1 BuildKvc1(uint32_t declared_value_words, const std::vector<Kvc1EntrySpec>& entries) {
	BuiltKvc1 out;
	std::vector<uint8_t>& b = out.bytes;

	b.insert(b.end(), superslm::kConstantsMagic, superslm::kConstantsMagic + 4);
	WriteU32LE(b, superslm::kManifestVersion);  // KVC1 version field, == 1, same constant as WGT1/BIA1/ROP1's
	const uint32_t entry_count = static_cast<uint32_t>(entries.size());
	WriteU32LE(b, entry_count);
	out.entry_count = entry_count;
	WriteU32LE(b, declared_value_words);
	out.value_words = declared_value_words;

	// Name blob content and per-entry name offsets, computed before the
	// descriptors are written (a descriptor needs its own name_off).
	std::vector<uint32_t> name_offs(entries.size()), name_lens(entries.size());
	std::string name_blob;
	for (size_t i = 0; i < entries.size(); ++i) {
		name_offs[i] = static_cast<uint32_t>(name_blob.size());
		name_lens[i] = static_cast<uint32_t>(entries[i].name.size());
		name_blob += entries[i].name;
	}
	const uint32_t name_blob_len = static_cast<uint32_t>(name_blob.size());
	WriteU32LE(b, name_blob_len);
	out.name_blob_len = name_blob_len;

	WriteU32LE(b, 0);  // reserved

	out.descriptors_off = b.size();
	for (size_t i = 0; i < entries.size(); ++i) {
		WriteU32LE(b, name_offs[i]);
		WriteU32LE(b, name_lens[i]);
	}

	out.values_off = b.size();
	for (size_t i = 0; i < entries.size(); ++i) {
		for (int64_t v : entries[i].values) {
			WriteU64LE(b, static_cast<uint64_t>(v));  // two's-complement bit pattern, little-endian
		}
	}

	out.name_blob_off = b.size();
	WriteBytes(b, name_blob);

	return out;
}

// The minimal valid KVC1 blob every structural/per-entry hostile cell for a
// given `value_words` mutates from: three entries with distinctive,
// hand-picked int64 tuples — a small positive value, a negative value (pins
// the little-endian signed read: a naive unsigned read would corrupt it),
// and a large-magnitude value near the int64 range — so the feature oracle's
// spot-check cannot pass on a coincidentally-zeroed or truncated read.
// `value_words` must be 2 or 3.
inline BuiltKvc1 MakeMinimalValidKvc1(uint32_t value_words) {
	std::vector<Kvc1EntrySpec> entries;
	if (value_words == 2) {
		entries = {
		    {"alpha", {7, -3}},
		    {"beta", {-1000000, 42}},
		    {"gamma", {4611686018427387903LL, -4611686018427387904LL}},  // near INT64_MAX/MIN quarter-range
		};
	} else {
		entries = {
		    {"alpha", {7, -3, 1000}},
		    {"beta", {-1000000, 42, -55}},
		    {"gamma", {4611686018427387903LL, -4611686018427387904LL, 9223372036854775807LL}},
		};
	}
	return BuildKvc1(value_words, entries);
}

// A validated SslmSectionView directly over `bytes`, of the given keyed-
// constant section type — Parse's actual input. No outer SslmArtifact is
// built: the KVC1 sub-parse never reads back through the artifact once it
// holds a validated section (the S0 suite already covers the outer loader's
// own hostile-input sweep). dtype is always Raw — KVC1 sections carry dtype
// Raw (the "Json" label in the section-types table; docs/sslm_format.md
// "Json sections use dtype Raw").
inline superslm::SslmSectionView MakeConstantsSectionView(superslm::SslmSectionType type,
                                                            const std::vector<uint8_t>& bytes) {
	superslm::SslmSectionView v;
	v.type = type;
	v.dtype = superslm::SslmDtype::Raw;
	v.data = bytes.data();
	v.byte_size = bytes.size();
	v.elem_count = bytes.size();
	v.alignment = 64;
	return v;
}

}  // namespace superslm_test

#endif  // SUPERSLM_TESTS_SSLM_KVC1_HOSTILE_FIXTURES_H

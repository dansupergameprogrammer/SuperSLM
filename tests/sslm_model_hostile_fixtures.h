// Curie's S2.0a red suite — spec-faithful WGT1/BIA1/ROP1 tensor-manifest
// builder (SuperSLM_Plan.md S2.0a; docs/sslm_format.md "Model sub-formats").
// `SslmTensorManifest::Parse` parses one array section's self-contained tensor
// manifest AFTER SslmArtifact has verified whole-file structure and integrity
// — like the TOK1/UNI1 sub-parse (Claude/Curie/SuperSLM_T129_SubParse_
// Hostile_TestDesign-2026-07-19.md), a crafted integrity-valid artifact can
// still carry a malformed manifest inside a validated section, so the manifest
// sub-parse is its own hostile-input trust boundary held to the same bar.
//
// Unlike the tokenizer suite, this suite calls `Parse` directly against a
// hand-built `SslmSectionView` (Parse's actual contract — it takes a section,
// not a whole artifact) rather than routing every cell through
// `SslmArtifact::OpenFromMemory`: the outer loader's own hostile-input sweep
// is already covered by the S0 suite, and the manifest sub-parse never
// consults the artifact once it holds a validated section, so building the
// outer artifact wrapper would add ceremony without isolating anything new.
//
// The three array sections (Weights/WGT1/int8, Biases/BIA1/int32,
// RopeTables/ROP1/int64) share one manifest layout; only the magic and the
// element size differ. `BuildManifest` is written directly against the spec's
// field table (docs/sslm_format.md "Tensor-manifest blob"), independent of
// `src/model.cpp` (the code under test — currently a red-first stub that
// leaves `out` empty and reports Ok). A rejection cell takes a small
// spec-faithful manifest and mutates exactly one descriptor or header field
// via `ManifestDesc*Off`/`Put32`/`Put64` (from sslm_fixtures.h), so the byte
// buffer differs from a valid manifest in exactly the one way the cell names.
#ifndef SUPERSLM_TESTS_SSLM_MODEL_HOSTILE_FIXTURES_H
#define SUPERSLM_TESTS_SSLM_MODEL_HOSTILE_FIXTURES_H

#include "sslm_fixtures.h"
#include "superslm/artifact.h"
#include "superslm/model.h"

#include <cstdint>
#include <string>
#include <vector>

namespace superslm_test {

// --- append-style little-endian writers, distinct names from sslm_fixtures.h's
//     in-place PutU32/PutU64 (which write at an existing offset) and from
//     sslm_tokenizer_hostile_fixtures.h's AppendU32LE/AppendBytes (a sibling
//     header included in the same translation unit — same names would
//     redefine, not overload). ---

inline void WriteU32LE(std::vector<uint8_t>& b, uint32_t v) {
	for (int i = 0; i < 4; ++i) b.push_back(static_cast<uint8_t>((v >> (8 * i)) & 0xFF));
}
inline void WriteU64LE(std::vector<uint8_t>& b, uint64_t v) {
	for (int i = 0; i < 8; ++i) b.push_back(static_cast<uint8_t>((v >> (8 * i)) & 0xFF));
}
inline void WriteBytes(std::vector<uint8_t>& b, const std::string& s) {
	b.insert(b.end(), s.begin(), s.end());
}

// One tensor to bake into a manifest. `shape.size()` is the rank (1..4);
// elem_count is always the correct product(shape) — a cell that wants a
// mismatch patches the elem_count field afterward via ManifestDescElemCountOff.
struct ManifestTensorSpec {
	std::string name;
	std::vector<uint32_t> shape;
};

// Fixed-formula byte offsets from docs/sslm_format.md's field tables — not
// derived from the parser under test, mirroring sslm_fixtures.h's
// `row = kHeaderBytes + i * kSectionDescBytes` convention for the outer
// section table. Header fields (0/4/8/12) are constants; descriptor fields
// are a function of the tensor's index `i` (each TensorDesc is a fixed
// kTensorDescBytes==48 bytes, starting at kManifestHeaderBytes==16).
inline constexpr size_t kManifestMagicOff = 0;
inline constexpr size_t kManifestVersionOff = 4;
inline constexpr size_t kManifestTensorCountOff = 8;
inline constexpr size_t kManifestNameBlobLenOff = 12;

inline size_t ManifestDescOff(size_t i) {
	return superslm::kManifestHeaderBytes + i * superslm::kTensorDescBytes;
}
inline size_t ManifestDescNameOffOff(size_t i) { return ManifestDescOff(i) + 0; }
inline size_t ManifestDescNameLenOff(size_t i) { return ManifestDescOff(i) + 4; }
inline size_t ManifestDescRankOff(size_t i) { return ManifestDescOff(i) + 8; }
inline size_t ManifestDescShapeOff(size_t i, size_t dim) { return ManifestDescOff(i) + 12 + dim * 4; }
inline size_t ManifestDescDataOffOff(size_t i) { return ManifestDescOff(i) + 28; }
inline size_t ManifestDescElemCountOff(size_t i) { return ManifestDescOff(i) + 36; }
inline size_t ManifestDescReservedOff(size_t i) { return ManifestDescOff(i) + 44; }

struct BuiltManifest {
	std::vector<uint8_t> bytes;
	uint32_t tensor_count = 0;
	uint32_t name_blob_len = 0;
	size_t name_blob_off = 0;
	size_t data_region_off = 0;
	std::vector<uint64_t> tensor_data_off;    // per tensor, in descriptor order
	std::vector<uint64_t> tensor_elem_count;  // per tensor, == product(shape)
};

// Builds a spec-faithful WGT1/BIA1/ROP1 blob (docs/sslm_format.md "Tensor-
// manifest blob"): 16-byte header, `tensors.size()` fixed 48-byte descriptors,
// the concatenated name blob, then each tensor's data packed contiguously in
// descriptor order starting at the data region (name-blob end, rounded up to
// `element_size`) — the same non-overlapping, ascending placement the spec
// says the converter produces. Every field an ordinary converter would emit
// correctly is correct here; a hostile cell mutates exactly one afterward.
// `element_size` is 1/4/8 for WGT1/BIA1/ROP1 (int8/int32/int64) respectively.
// Tensor data bytes are a deterministic, non-zero pattern per tensor so an
// oracle test can spot-check a value.
inline BuiltManifest BuildManifest(const uint8_t magic[4], uint32_t element_size,
                                    const std::vector<ManifestTensorSpec>& tensors) {
	BuiltManifest out;
	std::vector<uint8_t>& b = out.bytes;

	b.insert(b.end(), magic, magic + 4);
	WriteU32LE(b, superslm::kManifestVersion);
	const uint32_t tensor_count = static_cast<uint32_t>(tensors.size());
	WriteU32LE(b, tensor_count);
	out.tensor_count = tensor_count;

	// Name blob content and per-tensor name offsets, computed before the
	// descriptors are written (a descriptor needs its own name_off).
	std::vector<uint32_t> name_offs(tensors.size()), name_lens(tensors.size());
	std::string name_blob;
	for (size_t i = 0; i < tensors.size(); ++i) {
		name_offs[i] = static_cast<uint32_t>(name_blob.size());
		name_lens[i] = static_cast<uint32_t>(tensors[i].name.size());
		name_blob += tensors[i].name;
	}
	const uint32_t name_blob_len = static_cast<uint32_t>(name_blob.size());
	WriteU32LE(b, name_blob_len);
	out.name_blob_len = name_blob_len;

	// elem_count (== product(shape), 64-bit) and each tensor's data_off,
	// packed contiguously starting at the data region — computed before the
	// descriptors are written, since a descriptor carries its own data_off.
	std::vector<uint64_t> elem_counts(tensors.size());
	for (size_t i = 0; i < tensors.size(); ++i) {
		uint64_t p = 1;
		for (uint32_t d : tensors[i].shape) p *= d;
		elem_counts[i] = p;
	}
	const size_t header_and_desc = superslm::kManifestHeaderBytes +
	                                tensors.size() * superslm::kTensorDescBytes;
	const size_t name_blob_start = header_and_desc;
	size_t data_region_start = name_blob_start + name_blob_len;
	data_region_start = static_cast<size_t>(
	    AlignUp(static_cast<uint64_t>(data_region_start), element_size == 0 ? 1 : element_size));
	out.name_blob_off = name_blob_start;
	out.data_region_off = data_region_start;

	std::vector<uint64_t> data_offs(tensors.size());
	uint64_t cursor = data_region_start;
	for (size_t i = 0; i < tensors.size(); ++i) {
		data_offs[i] = cursor;
		cursor += elem_counts[i] * element_size;
	}
	out.tensor_data_off = data_offs;
	out.tensor_elem_count = elem_counts;

	for (size_t i = 0; i < tensors.size(); ++i) {
		WriteU32LE(b, name_offs[i]);
		WriteU32LE(b, name_lens[i]);
		WriteU32LE(b, static_cast<uint32_t>(tensors[i].shape.size()));  // rank
		for (uint32_t dim = 0; dim < superslm::kMaxTensorRank; ++dim) {
			WriteU32LE(b, dim < tensors[i].shape.size() ? tensors[i].shape[dim] : 0u);
		}
		WriteU64LE(b, data_offs[i]);
		WriteU64LE(b, elem_counts[i]);
		WriteU32LE(b, 0);  // reserved
	}

	WriteBytes(b, name_blob);
	b.resize(data_region_start, 0);  // pad up to the (possibly rounded-up) data region

	for (size_t i = 0; i < tensors.size(); ++i) {
		const uint64_t nbytes = elem_counts[i] * element_size;
		for (uint64_t k = 0; k < nbytes; ++k) {
			// Deterministic, tensor-distinguishing, non-zero pattern for the
			// feature oracle's spot-check.
			b.push_back(static_cast<uint8_t>((i * 31 + k * 7 + 11) & 0xFF));
		}
	}
	return out;
}

// The minimal valid manifest every structural/per-descriptor hostile cell for
// a given magic mutates from: four small tensors spanning rank 1 through 4
// ("t1" rank1 shape[3]; "t2" rank2 shape[2,2]; "t3" rank3 shape[2,1,2]; "t4"
// rank4 shape[1,1,1,2]) — every shape entry genuinely nonzero where declared,
// matching docs/sslm_format.md's per-descriptor constraints exactly, not a
// placeholder. Reused directly as the feature-oracle fixture (Ok() case).
inline BuiltManifest MakeMinimalValidManifest(const uint8_t magic[4], uint32_t element_size) {
	std::vector<ManifestTensorSpec> tensors = {
	    {"t1", {3}},
	    {"t2", {2, 2}},
	    {"t3", {2, 1, 2}},
	    {"t4", {1, 1, 1, 2}},
	};
	return BuildManifest(magic, element_size, tensors);
}

// A single-tensor manifest, name "t0", for cells where a second tensor would
// make the mutated field's effect ambiguous with TensorOverlap (an alignment
// or data_off shift on one tensor of a contiguously-packed multi-tensor
// manifest can also shove its range into the next tensor's — a single tensor
// has nothing to overlap). `shape` need not be small; the overflow cells pass
// a shape of `{1}` or `{1,1}` and then patch the descriptor's shape/elem_count
// fields directly to declare a hostile value without ever materializing that
// many actual data bytes (the manifest's true `byte_size` stays tiny, which
// is exactly what should make the parse reject the claim).
inline BuiltManifest MakeSingleTensorManifest(const uint8_t magic[4], uint32_t element_size,
                                               const std::vector<uint32_t>& shape) {
	return BuildManifest(magic, element_size, {{"t0", shape}});
}

// A validated SslmSectionView directly over `bytes` — Parse's actual input.
// No outer SslmArtifact is built: the manifest sub-parse never reads back
// through the artifact once it holds a validated section (the S0 suite
// already covers the outer loader's own hostile-input sweep).
inline superslm::SslmSectionView MakeManifestSectionView(superslm::SslmSectionType type,
                                                          superslm::SslmDtype dtype,
                                                          const std::vector<uint8_t>& bytes) {
	superslm::SslmSectionView v;
	v.type = type;
	v.dtype = dtype;
	v.data = bytes.data();
	v.byte_size = bytes.size();
	const uint32_t dsize = superslm::DtypeSize(static_cast<uint32_t>(dtype));
	v.elem_count = dsize > 0 ? bytes.size() / dsize : 0;
	v.alignment = 64;
	return v;
}

}  // namespace superslm_test

#endif  // SUPERSLM_TESTS_SSLM_MODEL_HOSTILE_FIXTURES_H

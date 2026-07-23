// Curie's S1 tokenizer red suite — fixture helpers.
//
// Three helpers, independent of `superslm::TokenizerView` (the code under test):
//   - `ResolveFixturePath` — locates a committed fixture under `tests/fixtures/`
//     whether the test binary's CWD is the repo root (build.bat) or a build/out
//     subdirectory (CMake/ctest), by searching upward.
//   - `LoadGoldenPack` — parses `qwen_tok_golden.gld`, the binary golden emitted by
//     `tools/convert_tokenizer.py`'s `emit_golden()`: 'GLD1', u32 record_count,
//     u8[32] ids_hash, then per record: u32 text_len, text UTF-8 bytes, u32
//     id_count, i32[id_count] ids. These ids are the upstream HF tokenizer's
//     output for each corpus line (no special tokens auto-added) — the golden
//     gate's oracle.
//   - `ParseTokenizerBlob` — parses the artifact's raw `Tokenizer` section bytes
//     (the `TOK1` blob `tools/convert_tokenizer.py`'s `serialize_tokenizer()`
//     writes) directly, independent of `TokenizerView::Open`, so a cell can assert
//     `VocabSize()` and a special token's id against the artifact's own declared
//     bytes rather than against `TokenizerView`'s own state.
//
// This mirrors `sslm_fixtures.h`'s convention for the S0 loader suite: fixture
// parsing is spec-faithful and independent of the code under test, so a defect in
// the code under test cannot also corrupt the oracle that catches it.
#ifndef SUPERSLM_TESTS_SSLM_TOKENIZER_FIXTURES_H
#define SUPERSLM_TESTS_SSLM_TOKENIZER_FIXTURES_H

#include <array>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace superslm_test {

// Searches, in order, "tests/fixtures/<name>", "../tests/fixtures/<name>",
// "../../tests/fixtures/<name>" for an existing file and returns the first hit's
// path. Returns an empty string if none exist — the caller CHECKs non-empty so a
// missing fixture fails with a named reason, not a crash.
inline std::string ResolveFixturePath(const std::string& filename) {
	static const char* kPrefixes[] = {"tests/fixtures/", "../tests/fixtures/",
	                                   "../../tests/fixtures/"};
	for (const char* prefix : kPrefixes) {
		std::string candidate = std::string(prefix) + filename;
		std::error_code ec;
		if (std::filesystem::exists(candidate, ec)) return candidate;
	}
	return {};
}

// --- qwen_tok_golden.gld -----------------------------------------------------

struct GoldenRecord {
	std::string text;
	std::vector<int32_t> ids;
};

struct GoldenPack {
	bool ok = false;
	std::string error;  // set iff !ok
	uint32_t record_count = 0;
	std::array<uint8_t, 32> ids_hash{};
	std::vector<GoldenRecord> records;
};

inline uint32_t ReadU32LE(const std::vector<uint8_t>& b, size_t off) {
	uint32_t v = 0;
	for (int i = 0; i < 4; ++i) v |= static_cast<uint32_t>(b[off + i]) << (8 * i);
	return v;
}
inline int32_t ReadI32LE(const std::vector<uint8_t>& b, size_t off) {
	return static_cast<int32_t>(ReadU32LE(b, off));
}

// Parses the binary golden pack. Fails closed (`ok == false`, `error` named) on a
// bad magic, a truncated buffer, or a length that overruns the file — this parser
// is fixture-reading scaffolding, not a hostile-input trust boundary, but a
// silently wrong parse would corrupt the oracle every tokenizer cell depends on,
// so a malformed file is a named failure, never a best-effort read.
inline GoldenPack LoadGoldenPack(const std::string& path) {
	GoldenPack out;
	std::ifstream f(path, std::ios::binary);
	if (!f) {
		out.error = "cannot open golden pack: " + path;
		return out;
	}
	std::vector<uint8_t> data((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());

	auto need = [&](size_t off, size_t len) {
		return off + len <= data.size();
	};

	if (!need(0, 8)) {
		out.error = "golden pack truncated before header: " + path;
		return out;
	}
	if (std::memcmp(data.data(), "GLD1", 4) != 0) {
		out.error = "golden pack bad magic (want GLD1): " + path;
		return out;
	}
	out.record_count = ReadU32LE(data, 4);
	if (!need(8, 32)) {
		out.error = "golden pack truncated before ids_hash: " + path;
		return out;
	}
	std::memcpy(out.ids_hash.data(), data.data() + 8, 32);

	size_t pos = 40;
	out.records.reserve(out.record_count);
	for (uint32_t r = 0; r < out.record_count; ++r) {
		if (!need(pos, 4)) {
			out.error = "golden pack truncated at record " + std::to_string(r) + " text_len: " + path;
			return out;
		}
		uint32_t text_len = ReadU32LE(data, pos);
		pos += 4;
		if (!need(pos, text_len)) {
			out.error = "golden pack truncated at record " + std::to_string(r) + " text: " + path;
			return out;
		}
		GoldenRecord rec;
		rec.text.assign(reinterpret_cast<const char*>(data.data() + pos), text_len);
		pos += text_len;
		if (!need(pos, 4)) {
			out.error = "golden pack truncated at record " + std::to_string(r) + " id_count: " + path;
			return out;
		}
		uint32_t id_count = ReadU32LE(data, pos);
		pos += 4;
		if (!need(pos, static_cast<size_t>(id_count) * 4)) {
			out.error = "golden pack truncated at record " + std::to_string(r) + " ids: " + path;
			return out;
		}
		rec.ids.reserve(id_count);
		for (uint32_t i = 0; i < id_count; ++i) {
			rec.ids.push_back(ReadI32LE(data, pos));
			pos += 4;
		}
		out.records.push_back(std::move(rec));
	}
	out.ok = true;
	return out;
}

// True iff every byte of `s` is 7-bit ASCII (< 0x80) — the "pure-ASCII" split the
// decode round-trip cell uses (an ASCII line's NFC form is itself, so Decode of
// its golden ids is asserted exactly; a non-ASCII line's NFC form is not computed
// in C++, so it gets the idempotent Decode(Encode(t)) == Decode(golden.ids) form).
inline bool IsAsciiOnly(const std::string& s) {
	for (unsigned char c : s) {
		if (c >= 0x80) return false;
	}
	return true;
}

// --- The artifact's raw Tokenizer section ("TOK1" blob) ----------------------

struct TokBlobSpecial {
	std::string content;
	uint32_t id = 0;
};

struct TokBlobInfo {
	bool ok = false;
	std::string error;  // set iff !ok
	uint32_t version = 0;
	uint32_t vocab_count = 0;
	uint32_t merge_count = 0;
	uint32_t special_count = 0;
	std::vector<TokBlobSpecial> specials;
	// S-HARDEN-2 (F15's "every raw byte mapping" class): the 256-entry byte->id
	// base table and the id->raw-bytes vocabulary blob, retained (not merely
	// walked) so a cell can check the base vocabulary's bijection onto raw
	// bytes directly from the artifact's own bytes, independent of
	// TokenizerView::Open/Decode (whose Decode() now applies UTF-8 validation
	// over the CONCATENATION of a request's ids — a single extended-range byte
	// token decoded ALONE is correctly not valid standalone UTF-8, which is a
	// different claim from "the vocabulary has one slot per byte value").
	std::vector<uint32_t> byte_to_id;
	std::vector<std::string> id_to_bytes;  // index == token id, size == vocab_count
};

// Parses the TOK1 blob exactly as `tools/convert_tokenizer.py`'s
// `serialize_tokenizer()` writes it:
//   'TOK1', u32 version, u32 vocab_count, u32 merge_count, u32 special_count, u32 reserved,
//   u32[256] byte_to_id,
//   u32[vocab_count+1] id_to_bytes offsets, u32 blob_len, byte[blob_len] blob,
//   (u32 a, u32 b, u32 merged) * merge_count,
//   u32[special_count] special ids,
//   u32[special_count+1] special-content offsets, u32 s_blob_len, byte[s_blob_len] s_blob.
// Only `version`/`vocab_count`/`merge_count`/`special_count`/`specials` are kept —
// `byte_to_id` and the merge table are walked (to advance the cursor correctly)
// but not retained, since no cell asserts against them. Independent of
// `TokenizerView::Open`, so `VocabSize()` and a special token's declared id can be
// checked against the artifact's own bytes, not against the view's own state.
inline TokBlobInfo ParseTokenizerBlob(const uint8_t* data, size_t size) {
	TokBlobInfo out;
	std::vector<uint8_t> b(data, data + size);  // reuse the ReadU32LE/ReadI32LE helpers above
	auto need = [&](size_t off, size_t len) {
		return off + len <= b.size();
	};

	if (!need(0, 24)) {
		out.error = "TOK1 blob truncated before header";
		return out;
	}
	if (std::memcmp(b.data(), "TOK1", 4) != 0) {
		out.error = "TOK1 blob bad magic";
		return out;
	}
	out.version = ReadU32LE(b, 4);
	out.vocab_count = ReadU32LE(b, 8);
	out.merge_count = ReadU32LE(b, 12);
	out.special_count = ReadU32LE(b, 16);
	// b[20..24) is the reserved u32 — not asserted here.

	size_t pos = 24;
	out.byte_to_id.reserve(256);
	for (uint32_t k = 0; k < 256; ++k) out.byte_to_id.push_back(ReadU32LE(b, pos + k * 4u));
	pos += 256u * 4u;  // byte_to_id[256]
	if (!need(pos, static_cast<size_t>(out.vocab_count + 1) * 4)) {
		out.error = "TOK1 blob truncated in id_to_bytes offsets";
		return out;
	}
	std::vector<uint32_t> vocab_offs(out.vocab_count + 1);
	for (uint32_t i = 0; i <= out.vocab_count; ++i) vocab_offs[i] = ReadU32LE(b, pos + static_cast<size_t>(i) * 4);
	pos += static_cast<size_t>(out.vocab_count + 1) * 4;  // id_to_bytes offsets
	if (!need(pos, 4)) {
		out.error = "TOK1 blob truncated before id_to_bytes blob_len";
		return out;
	}
	uint32_t id_bytes_blob_len = ReadU32LE(b, pos);
	pos += 4;
	if (!need(pos, id_bytes_blob_len)) {
		out.error = "TOK1 blob truncated in id_to_bytes blob";
		return out;
	}
	out.id_to_bytes.reserve(out.vocab_count);
	for (uint32_t i = 0; i < out.vocab_count; ++i) {
		const uint32_t o0 = vocab_offs[i], o1 = vocab_offs[i + 1];
		if (o0 > o1 || o1 > id_bytes_blob_len) {
			out.error = "TOK1 blob: id_to_bytes offset out of range";
			return out;
		}
		out.id_to_bytes.emplace_back(reinterpret_cast<const char*>(b.data() + pos + o0), o1 - o0);
	}
	pos += id_bytes_blob_len;

	if (!need(pos, static_cast<size_t>(out.merge_count) * 12)) {
		out.error = "TOK1 blob truncated in merges";
		return out;
	}
	pos += static_cast<size_t>(out.merge_count) * 12;  // (a,b,merged) * merge_count

	if (!need(pos, static_cast<size_t>(out.special_count) * 4)) {
		out.error = "TOK1 blob truncated in special ids";
		return out;
	}
	std::vector<uint32_t> special_ids(out.special_count);
	for (uint32_t i = 0; i < out.special_count; ++i) {
		special_ids[i] = ReadU32LE(b, pos);
		pos += 4;
	}

	if (!need(pos, static_cast<size_t>(out.special_count + 1) * 4)) {
		out.error = "TOK1 blob truncated in special-content offsets";
		return out;
	}
	std::vector<uint32_t> s_offs(out.special_count + 1);
	for (uint32_t i = 0; i <= out.special_count; ++i) {
		s_offs[i] = ReadU32LE(b, pos);
		pos += 4;
	}

	if (!need(pos, 4)) {
		out.error = "TOK1 blob truncated before special-content blob_len";
		return out;
	}
	uint32_t s_blob_len = ReadU32LE(b, pos);
	pos += 4;
	if (!need(pos, s_blob_len)) {
		out.error = "TOK1 blob truncated in special-content blob";
		return out;
	}
	const uint8_t* s_blob = b.data() + pos;

	out.specials.reserve(out.special_count);
	for (uint32_t i = 0; i < out.special_count; ++i) {
		if (s_offs[i] > s_blob_len || s_offs[i + 1] > s_blob_len || s_offs[i] > s_offs[i + 1]) {
			out.error = "TOK1 blob special-content offsets out of range";
			return out;
		}
		TokBlobSpecial sp;
		sp.id = special_ids[i];
		sp.content.assign(reinterpret_cast<const char*>(s_blob) + s_offs[i], s_offs[i + 1] - s_offs[i]);
		out.specials.push_back(std::move(sp));
	}

	out.ok = true;
	return out;
}

}  // namespace superslm_test

#endif  // SUPERSLM_TESTS_SSLM_TOKENIZER_FIXTURES_H

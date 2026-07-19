// Curie's T-129 red suite — spec-faithful TOK1/UNI1 sub-blob builders (DecisionLog
// D-SLM62). `TokenizerView::Open` parses these two self-contained blobs (docs/
// sslm_format.md "Tokenizer sub-formats") AFTER the loader has verified whole-file
// integrity, so a malicious artifact can carry a valid self-hash over a hostile
// sub-blob — the sub-parse is its own trust boundary. This header builds each blob
// by explicit little-endian field writes per the spec, independent of
// src/tokenizer.cpp's ParseTok/ParseUni (the code under test), mirroring
// sslm_fixtures.h's convention for the S0 loader suite: a rejection cell starts
// from a structurally valid blob and mutates exactly one field, so the byte buffer
// differs from a valid one in exactly the way the cell names.
//
// The two `Layout` structs record every field's byte offset within its blob as it
// is written, so a cell can target a field by name (`layout.vocab_count_off`,
// `layout.decomp_offsets_off`, ...) instead of a hardcoded magic number — the
// layout is unaffected by unrelated changes elsewhere in the minimal fixture.
#ifndef SUPERSLM_TESTS_SSLM_TOKENIZER_HOSTILE_FIXTURES_H
#define SUPERSLM_TESTS_SSLM_TOKENIZER_HOSTILE_FIXTURES_H

#include "sslm_fixtures.h"
#include "superslm/artifact.h"

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace superslm_test {

inline void AppendU32LE(std::vector<uint8_t>& b, uint32_t v) {
	for (int i = 0; i < 4; ++i) b.push_back(static_cast<uint8_t>((v >> (8 * i)) & 0xFF));
}
inline void AppendBytes(std::vector<uint8_t>& b, const std::string& s) {
	b.insert(b.end(), s.begin(), s.end());
}

// ============================== TOK1 ======================================

struct TokVocabEntry {
	std::string bytes;  // the UTF-8 bytes this id decodes to
};
struct TokMerge {
	uint32_t a, b, merged;
};
struct TokSpecial {
	std::string content;
	uint32_t id;
};

// Byte offsets of every TOK1 field, populated by BuildTok1 as it writes. A cell
// mutates the blob by patching at `layout.<field>_off` (PutU32, from
// sslm_fixtures.h) or resizing the buffer to a point strictly inside a field to
// truncate it — never by hand-computing an offset.
struct Tok1Layout {
	size_t magic_off = 0;
	size_t version_off = 4;
	size_t vocab_count_off = 8;
	size_t merge_count_off = 12;
	size_t special_count_off = 16;
	size_t reserved_off = 20;
	size_t byte_to_id_off = 24;
	size_t vocab_offsets_off = 0;
	size_t vocab_blob_len_off = 0;
	size_t vocab_blob_off = 0;
	size_t merges_off = 0;
	size_t special_ids_off = 0;
	size_t special_offsets_off = 0;
	size_t special_blob_len_off = 0;
	size_t special_blob_off = 0;
	uint32_t vocab_count = 0;
	uint32_t merge_count = 0;
	uint32_t special_count = 0;
	uint32_t vocab_blob_len = 0;
	uint32_t special_blob_len = 0;
};

struct BuiltTok1 {
	std::vector<uint8_t> bytes;
	Tok1Layout layout;
};

// Builds a spec-faithful TOK1 blob (docs/sslm_format.md "Tokenizer blob — TOK1").
// Vocab ids are assigned 0..vocab.size()-1 in array order; `vocab[i].bytes` is the
// UTF-8 byte string id i decodes to. `byte_to_id` must have exactly 256 entries.
inline BuiltTok1 BuildTok1(const std::array<uint32_t, 256>& byte_to_id,
                           const std::vector<TokVocabEntry>& vocab,
                           const std::vector<TokMerge>& merges,
                           const std::vector<TokSpecial>& specials) {
	BuiltTok1 out;
	std::vector<uint8_t>& b = out.bytes;
	Tok1Layout& L = out.layout;

	AppendBytes(b, "TOK1");
	AppendU32LE(b, 1);  // version
	const uint32_t vocab_count = static_cast<uint32_t>(vocab.size());
	const uint32_t merge_count = static_cast<uint32_t>(merges.size());
	const uint32_t special_count = static_cast<uint32_t>(specials.size());
	AppendU32LE(b, vocab_count);
	AppendU32LE(b, merge_count);
	AppendU32LE(b, special_count);
	AppendU32LE(b, 0);  // reserved
	L.vocab_count = vocab_count;
	L.merge_count = merge_count;
	L.special_count = special_count;

	for (uint32_t v : byte_to_id) AppendU32LE(b, v);

	L.vocab_offsets_off = b.size();
	uint32_t running = 0;
	std::vector<uint32_t> voffs;
	voffs.push_back(0);
	for (const auto& e : vocab) {
		running += static_cast<uint32_t>(e.bytes.size());
		voffs.push_back(running);
	}
	for (uint32_t o : voffs) AppendU32LE(b, o);

	L.vocab_blob_len_off = b.size();
	AppendU32LE(b, running);
	L.vocab_blob_len = running;
	L.vocab_blob_off = b.size();
	for (const auto& e : vocab) AppendBytes(b, e.bytes);

	L.merges_off = b.size();
	for (const auto& m : merges) {
		AppendU32LE(b, m.a);
		AppendU32LE(b, m.b);
		AppendU32LE(b, m.merged);
	}

	L.special_ids_off = b.size();
	for (const auto& s : specials) AppendU32LE(b, s.id);

	L.special_offsets_off = b.size();
	uint32_t srunning = 0;
	std::vector<uint32_t> soffs;
	soffs.push_back(0);
	for (const auto& s : specials) {
		srunning += static_cast<uint32_t>(s.content.size());
		soffs.push_back(srunning);
	}
	for (uint32_t o : soffs) AppendU32LE(b, o);

	L.special_blob_len_off = b.size();
	AppendU32LE(b, srunning);
	L.special_blob_len = srunning;
	L.special_blob_off = b.size();
	for (const auto& s : specials) AppendBytes(b, s.content);

	return out;
}

// The minimal valid TOK1 blob every TOK1 hostile cell mutates from: 4 vocab
// entries ("c","a","t","ca"), one merge ((c,a) -> "ca"), one special ("<eos>").
// byte_to_id maps only 'c'/'a'/'t' (the bytes the round-trip cell actually
// exercises) — every other byte value is left at 0, which is harmless because no
// cell decodes an id produced from an unmapped byte.
inline BuiltTok1 MakeMinimalValidTok1() {
	std::array<uint32_t, 256> byte_to_id{};
	byte_to_id[static_cast<unsigned char>('c')] = 0;
	byte_to_id[static_cast<unsigned char>('a')] = 1;
	byte_to_id[static_cast<unsigned char>('t')] = 2;
	std::vector<TokVocabEntry> vocab = {{"c"}, {"a"}, {"t"}, {"ca"}};
	std::vector<TokMerge> merges = {{0, 1, 3}};
	std::vector<TokSpecial> specials = {{"<eos>", 1000}};
	return BuildTok1(byte_to_id, vocab, merges, specials);
}

// =============================== UNI1 ======================================

struct UniRange {
	uint32_t lo, hi;
};
struct UniCcc {
	uint32_t cp, klass;
};
struct UniDecomp {
	uint32_t cp;
	std::vector<uint32_t> seq;
};
struct UniCompose {
	uint32_t a, b, c;
};

struct Uni1Layout {
	size_t magic_off = 0;
	size_t version_off = 4;
	size_t letter_count_off = 0;
	size_t letter_data_off = 0;
	size_t number_count_off = 0;
	size_t number_data_off = 0;
	size_t space_count_off = 0;
	size_t space_data_off = 0;
	size_t ccc_count_off = 0;
	size_t ccc_data_off = 0;
	size_t decomp_count_off = 0;
	size_t decomp_cps_off = 0;
	size_t decomp_offsets_off = 0;
	size_t decomp_seqlen_off = 0;
	size_t decomp_seq_off = 0;
	size_t compose_count_off = 0;
	size_t compose_data_off = 0;
	uint32_t letter_count = 0, number_count = 0, space_count = 0;
	uint32_t ccc_count = 0, decomp_count = 0, seq_len = 0, compose_count = 0;
};

struct BuiltUni1 {
	std::vector<uint8_t> bytes;
	Uni1Layout layout;
};

// Builds a spec-faithful UNI1 blob (docs/sslm_format.md "Tokenizer sub-formats" —
// UnicodeTables blob — UNI1). `letters`/`numbers`/`spaces` must each already be in
// ascending, non-overlapping order (TokenizerView::Impl::InRanges binary-searches
// them) — the minimal fixture below satisfies this; a cell that wants to test
// range-table validation targets truncation/overflow, not ordering (the format has
// no declared ordering constraint to violate).
inline BuiltUni1 BuildUni1(const std::vector<UniRange>& letters, const std::vector<UniRange>& numbers,
                           const std::vector<UniRange>& spaces, const std::vector<UniCcc>& cccs,
                           const std::vector<UniDecomp>& decomps, const std::vector<UniCompose>& composes) {
	BuiltUni1 out;
	std::vector<uint8_t>& b = out.bytes;
	Uni1Layout& L = out.layout;

	AppendBytes(b, "UNI1");
	AppendU32LE(b, 1);  // version

	L.letter_count_off = b.size();
	AppendU32LE(b, static_cast<uint32_t>(letters.size()));
	L.letter_data_off = b.size();
	for (const auto& r : letters) {
		AppendU32LE(b, r.lo);
		AppendU32LE(b, r.hi);
	}
	L.letter_count = static_cast<uint32_t>(letters.size());

	L.number_count_off = b.size();
	AppendU32LE(b, static_cast<uint32_t>(numbers.size()));
	L.number_data_off = b.size();
	for (const auto& r : numbers) {
		AppendU32LE(b, r.lo);
		AppendU32LE(b, r.hi);
	}
	L.number_count = static_cast<uint32_t>(numbers.size());

	L.space_count_off = b.size();
	AppendU32LE(b, static_cast<uint32_t>(spaces.size()));
	L.space_data_off = b.size();
	for (const auto& r : spaces) {
		AppendU32LE(b, r.lo);
		AppendU32LE(b, r.hi);
	}
	L.space_count = static_cast<uint32_t>(spaces.size());

	L.ccc_count_off = b.size();
	AppendU32LE(b, static_cast<uint32_t>(cccs.size()));
	L.ccc_data_off = b.size();
	for (const auto& c : cccs) {
		AppendU32LE(b, c.cp);
		AppendU32LE(b, c.klass);
	}
	L.ccc_count = static_cast<uint32_t>(cccs.size());

	L.decomp_count_off = b.size();
	const uint32_t dn = static_cast<uint32_t>(decomps.size());
	AppendU32LE(b, dn);
	L.decomp_count = dn;
	L.decomp_cps_off = b.size();
	for (const auto& d : decomps) AppendU32LE(b, d.cp);

	L.decomp_offsets_off = b.size();
	uint32_t running = 0;
	std::vector<uint32_t> doffs;
	doffs.push_back(0);
	for (const auto& d : decomps) {
		running += static_cast<uint32_t>(d.seq.size());
		doffs.push_back(running);
	}
	for (uint32_t o : doffs) AppendU32LE(b, o);

	L.decomp_seqlen_off = b.size();
	AppendU32LE(b, running);
	L.seq_len = running;
	L.decomp_seq_off = b.size();
	for (const auto& d : decomps)
		for (uint32_t cp : d.seq) AppendU32LE(b, cp);

	L.compose_count_off = b.size();
	AppendU32LE(b, static_cast<uint32_t>(composes.size()));
	L.compose_count = static_cast<uint32_t>(composes.size());
	L.compose_data_off = b.size();
	for (const auto& c : composes) {
		AppendU32LE(b, c.a);
		AppendU32LE(b, c.b);
		AppendU32LE(b, c.c);
	}

	return out;
}

// The minimal valid UNI1 blob every UNI1 hostile cell mutates from: letter ranges
// covering ASCII 'A'-'Z'/'a'-'z' (what the round-trip cell's "cat" needs), a number
// range and two space ranges present for structural completeness, and one real
// canonical decomposition/composition pair (U+00E9 'é' <-> 'e' U+0065 + combining
// acute U+0301, ccc 230) — correct per Unicode, exercising all five UNI1 fields
// with genuine (not placeholder) data.
inline BuiltUni1 MakeMinimalValidUni1() {
	std::vector<UniRange> letters = {{'A', 'Z'}, {'a', 'z'}};
	std::vector<UniRange> numbers = {{'0', '9'}};
	std::vector<UniRange> spaces = {{0x09, 0x0D}, {0x20, 0x20}};
	std::vector<UniCcc> cccs = {{0x0301, 230}};
	std::vector<UniDecomp> decomps = {{0x00E9, {0x0065, 0x0301}}};
	std::vector<UniCompose> composes = {{0x0065, 0x0301, 0x00E9}};
	return BuildUni1(letters, numbers, spaces, cccs, decomps, composes);
}

// ========================= artifact assembly ================================

// Assembles {Config, Tokenizer(tok1), UnicodeTables(uni1)} via sslm_fixtures.h's
// BuildArtifact, which stamps file_bytes and the whole-file integrity hash fresh
// over whatever bytes it is given — so a cell that mutates `tok1`/`uni1` before
// calling this always gets a self-consistent, integrity-valid outer artifact: the
// loader accepts it, and only TokenizerView::Open's sub-parse can reject.
inline BuiltArtifact BuildTokenizerArtifact(const std::vector<uint8_t>& tok1, const std::vector<uint8_t>& uni1) {
	return BuildArtifact({MakeConfigSection(),
	                       MakeSection(superslm::SslmSectionType::Tokenizer, superslm::SslmDtype::Raw, tok1),
	                       MakeSection(superslm::SslmSectionType::UnicodeTables, superslm::SslmDtype::Raw, uni1)});
}

// Config + Tokenizer only — no UnicodeTables section.
inline BuiltArtifact BuildArtifactMissingUnicodeTables(const std::vector<uint8_t>& tok1) {
	return BuildArtifact(
	    {MakeConfigSection(), MakeSection(superslm::SslmSectionType::Tokenizer, superslm::SslmDtype::Raw, tok1)});
}

// Config + UnicodeTables only — no Tokenizer section.
inline BuiltArtifact BuildArtifactMissingTokenizer(const std::vector<uint8_t>& uni1) {
	return BuildArtifact(
	    {MakeConfigSection(), MakeSection(superslm::SslmSectionType::UnicodeTables, superslm::SslmDtype::Raw, uni1)});
}

}  // namespace superslm_test

#endif  // SUPERSLM_TESTS_SSLM_TOKENIZER_HOSTILE_FIXTURES_H

#include "superslm/tokenizer.h"

#include <algorithm>
#include <cstring>
#include <unordered_map>
#include <utility>
#include <vector>

namespace superslm {
namespace {

uint32_t Rd32(const uint8_t* p) {
	return uint32_t(p[0]) | (uint32_t(p[1]) << 8) | (uint32_t(p[2]) << 16) | (uint32_t(p[3]) << 24);
}

// The tokenizer's TOK1/UNI1 blob format version (docs/sslm_format.md "Tokenizer
// blob -- TOK1" / "UnicodeTables blob -- UNI1"): both fields are documented `u32 = 1`
// today. S-HARDEN-2 (F6): this is a real, checked precondition, not a magic literal
// repeated at each call site.
constexpr uint32_t kTokBlobVersion = 1;
constexpr uint32_t kUniBlobVersion = 1;
// TOK1.vocab_count's declared domain (S-HARDEN-2, F18): zero is a vocabulary that can
// encode no byte and index no token, and every id this parser stores is narrowed to
// int32_t downstream (Encode/Decode/VocabSize), so a vocab_count that cannot be
// represented as a non-negative int32 is rejected here rather than silently wrapping
// at the narrowing site.
constexpr uint32_t kTokVocabCountMax = 0x7FFFFFFFu;  // INT32_MAX

// The ONE strict UTF-8 decoder shared by Encode (over raw input text) and Decode
// (over the concatenated raw bytes of every emitted token id) -- S-HARDEN-2 (F7).
// `include/superslm/tokenizer.h`'s documented policy is "invalid sequences pass
// through as replacement chars"; this is the one place that policy is implemented,
// so Encode and Decode cannot silently diverge on what counts as malformed. Rejects
// (substitutes exactly one U+FFFD for) every ill-formed subsequence per the Unicode
// Standard's "maximal subpart of an ill-formed subsequence" algorithm: overlong
// encodings (a non-canonical byte length for the encoded codepoint), UTF-16
// surrogate codepoints (U+D800-U+DFFF, which UTF-8 must never encode), codepoints
// past U+10FFFF, a continuation byte with no valid lead, and a sequence truncated at
// end-of-input. Because this runs over the FULLY CONCATENATED byte string (all of
// Encode's input span, or all of Decode's ids' bytes joined in order) rather than
// per-token, a multi-byte sequence whose bytes are split across two adjacent
// tokens' vocabulary entries reassembles correctly before this ever sees it.
void Utf8Decode(std::string_view s, std::vector<uint32_t>& out) {
	const size_t n = s.size();
	size_t i = 0;
	// The continuation byte (10xxxxxx) at `idx`'s low 6 bits, or -1 if `idx` is past
	// the end or the byte there is not a valid continuation byte.
	auto cont = [&](size_t idx) -> int {
		if (idx >= n) return -1;
		uint8_t c = uint8_t(s[idx]);
		if ((c & 0xC0) != 0x80) return -1;
		return int(c & 0x3F);
	};
	while (i < n) {
		const uint8_t b0 = uint8_t(s[i]);
		if (b0 < 0x80) { out.push_back(b0); ++i; continue; }
		if (b0 < 0xC2) {
			// 0x80-0xBF: a lone/unexpected continuation byte, no lead preceded it.
			// 0xC0-0xC1: the only two-byte lead bytes that can only ever encode an
			// overlong form (codepoints < 0x80, already representable in one byte).
			out.push_back(0xFFFD); ++i; continue;
		}
		if (b0 < 0xE0) {
			// Two-byte: C2-DF. Every remaining lead in this range is canonical for
			// some codepoint in [0x80, 0x7FF]; no separate overlong check needed.
			const int c1 = cont(i + 1);
			if (c1 < 0) { out.push_back(0xFFFD); ++i; continue; }
			out.push_back((uint32_t(b0 & 0x1F) << 6) | uint32_t(c1));
			i += 2; continue;
		}
		if (b0 < 0xF0) {
			// Three-byte: E0-EF. E0's first continuation must be >= 0xA0 (else the
			// codepoint is overlong, representable in two bytes); ED's first
			// continuation must be <= 0x9F (else the codepoint is a UTF-16
			// surrogate, U+D800-U+DFFF, which UTF-8 must never encode).
			const int c1 = cont(i + 1);
			bool c1_ok = c1 >= 0;
			if (c1_ok) {
				const uint8_t c1b = uint8_t(s[i + 1]);
				if (b0 == 0xE0 && c1b < 0xA0) c1_ok = false;
				if (b0 == 0xED && c1b > 0x9F) c1_ok = false;
			}
			if (!c1_ok) { out.push_back(0xFFFD); ++i; continue; }
			const int c2 = cont(i + 2);
			if (c2 < 0) { out.push_back(0xFFFD); i += 2; continue; }  // truncated: consume lead + 1 valid cont
			out.push_back((uint32_t(b0 & 0x0F) << 12) | (uint32_t(c1) << 6) | uint32_t(c2));
			i += 3; continue;
		}
		if (b0 < 0xF5) {
			// Four-byte: F0-F4. F0's first continuation must be >= 0x90 (else
			// overlong, representable in three bytes); F4's first continuation must
			// be <= 0x8F (else the codepoint exceeds U+10FFFF).
			const int c1 = cont(i + 1);
			bool c1_ok = c1 >= 0;
			if (c1_ok) {
				const uint8_t c1b = uint8_t(s[i + 1]);
				if (b0 == 0xF0 && c1b < 0x90) c1_ok = false;
				if (b0 == 0xF4 && c1b > 0x8F) c1_ok = false;
			}
			if (!c1_ok) { out.push_back(0xFFFD); ++i; continue; }
			const int c2 = cont(i + 2);
			if (c2 < 0) { out.push_back(0xFFFD); i += 2; continue; }
			const int c3 = cont(i + 3);
			if (c3 < 0) { out.push_back(0xFFFD); i += 3; continue; }
			out.push_back((uint32_t(b0 & 0x07) << 18) | (uint32_t(c1) << 12) | (uint32_t(c2) << 6) | uint32_t(c3));
			i += 4; continue;
		}
		// 0xF5-0xFF: every codepoint a lead this large could encode exceeds
		// U+10FFFF outright.
		out.push_back(0xFFFD); ++i;
	}
}

void Utf8Append(uint32_t cp, std::string& s) {
	if (cp < 0x80) {
		s += char(cp);
	} else if (cp < 0x800) {
		s += char(0xC0 | (cp >> 6));
		s += char(0x80 | (cp & 0x3F));
	} else if (cp < 0x10000) {
		s += char(0xE0 | (cp >> 12));
		s += char(0x80 | ((cp >> 6) & 0x3F));
		s += char(0x80 | (cp & 0x3F));
	} else {
		s += char(0xF0 | (cp >> 18));
		s += char(0x80 | ((cp >> 12) & 0x3F));
		s += char(0x80 | ((cp >> 6) & 0x3F));
		s += char(0x80 | (cp & 0x3F));
	}
}

// Hangul (UAX #15), same constants as tools/unicode_tables.py.
constexpr uint32_t SBASE = 0xAC00, LBASE = 0x1100, VBASE = 0x1161, TBASE = 0x11A7;
constexpr uint32_t LCOUNT = 19, VCOUNT = 21, TCOUNT = 28;
constexpr uint32_t NCOUNT = VCOUNT * TCOUNT, SCOUNT = LCOUNT * NCOUNT;

using Range = std::pair<uint32_t, uint32_t>;

bool InRanges(uint32_t cp, const std::vector<Range>& r) {
	size_t lo = 0, hi = r.size();
	while (lo < hi) {
		size_t m = (lo + hi) / 2;
		if (cp < r[m].first) hi = m;
		else if (cp > r[m].second) lo = m + 1;
		else return true;
	}
	return false;
}

} // namespace

struct TokenizerView::Impl {
	// Tokenizer tables (pointers into the artifact's Tokenizer section; the artifact
	// outlives the view).
	uint32_t byte_to_id[256] = {};
	const uint8_t* vocab_offsets = nullptr; // u32[vocab_count+1], LE
	const uint8_t* vocab_blob = nullptr;
	uint32_t vocab_count = 0;
	// (a<<32|b) -> (rank, merged_id)
	std::unordered_map<uint64_t, std::pair<int32_t, int32_t>> merges;
	std::vector<std::pair<std::string, int32_t>> specials; // longest-content-first

	// Unicode (copied from the UnicodeTables section; ~55 KB).
	std::vector<Range> letter, number, space;
	std::unordered_map<uint32_t, uint8_t> ccc_map;
	std::unordered_map<uint32_t, std::vector<uint32_t>> decomp;
	std::unordered_map<uint64_t, uint32_t> compose;
	bool ok = false;

	bool is_letter(uint32_t cp) const { return InRanges(cp, letter); }
	bool is_number(uint32_t cp) const { return InRanges(cp, number); }
	bool is_space(uint32_t cp) const { return InRanges(cp, space); }
	uint8_t ccc(uint32_t cp) const {
		auto it = ccc_map.find(cp);
		return it == ccc_map.end() ? 0 : it->second;
	}

	// --- NFC (decompose -> canonical order -> compose), mirroring unicode_tables.py ---
	void decompose(const std::vector<uint32_t>& in, std::vector<uint32_t>& out) const {
		for (uint32_t cp : in) {
			if (cp >= SBASE && cp < SBASE + SCOUNT) {
				uint32_t s = cp - SBASE;
				out.push_back(LBASE + s / NCOUNT);
				out.push_back(VBASE + (s % NCOUNT) / TCOUNT);
				uint32_t t = s % TCOUNT;
				if (t) out.push_back(TBASE + t);
			} else {
				auto it = decomp.find(cp);
				if (it == decomp.end()) out.push_back(cp);
				else out.insert(out.end(), it->second.begin(), it->second.end());
			}
		}
	}

	void canonical_order(std::vector<uint32_t>& cps) const {
		for (size_t i = 1; i < cps.size(); ++i) {
			uint8_t cc = ccc(cps[i]);
			if (cc == 0) continue;
			size_t j = i;
			while (j > 0) {
				uint8_t pc = ccc(cps[j - 1]);
				if (pc > 0 && pc > cc) {
					std::swap(cps[j - 1], cps[j]);
					--j;
				} else {
					break;
				}
			}
		}
	}

	void compose_pass(std::vector<uint32_t>& cps) const {
		if (cps.empty()) return;
		std::vector<uint32_t> result;
		result.push_back(cps[0]);
		size_t last_starter = (ccc(cps[0]) == 0) ? 0 : SIZE_MAX;
		uint8_t prev_cc = ccc(cps[0]);
		for (size_t i = 1; i < cps.size(); ++i) {
			uint32_t cp = cps[i];
			uint8_t cc = ccc(cp);
			if (last_starter != SIZE_MAX && (prev_cc == 0 || prev_cc < cc)) {
				uint32_t s = result[last_starter];
				uint32_t comp = 0;
				bool have = false;
				if (s >= LBASE && s < LBASE + LCOUNT && cp >= VBASE && cp < VBASE + VCOUNT) {
					comp = SBASE + ((s - LBASE) * VCOUNT + (cp - VBASE)) * TCOUNT;
					have = true;
				} else if (s >= SBASE && s < SBASE + SCOUNT && (s - SBASE) % TCOUNT == 0 &&
				           cp > TBASE && cp < TBASE + TCOUNT) {
					comp = s + (cp - TBASE);
					have = true;
				} else {
					auto it = compose.find((uint64_t(s) << 32) | cp);
					if (it != compose.end()) { comp = it->second; have = true; }
				}
				if (have) {
					result[last_starter] = comp;
					continue;
				}
			}
			result.push_back(cp);
			if (cc == 0) last_starter = result.size() - 1;
			prev_cc = cc;
		}
		cps.swap(result);
	}

	void nfc(const std::vector<uint32_t>& in, std::vector<uint32_t>& out) const {
		out.clear();
		decompose(in, out);
		canonical_order(out);
		compose_pass(out);
	}

	// --- pre-tokenization: the fixed Qwen/GPT pattern (mirrors _pretokenize) ------
	// Emits [start,end) codepoint ranges into `pieces`.
	void pretokenize(const std::vector<uint32_t>& t,
	                 std::vector<std::pair<size_t, size_t>>& pieces) const {
		static const char* kContr[] = {"'s", "'t", "'re", "'ve", "'m", "'ll", "'d"};
		size_t i = 0, n = t.size();
		auto is_nl = [](uint32_t c) { return c == '\r' || c == '\n'; };
		while (i < n) {
			uint32_t ch = t[i];
			// 1. (?i:'s|'t|'re|'ve|'m|'ll|'d)
			if (ch == '\'') {
				char low[3] = {0, 0, 0};
				for (int k = 0; k < 3 && i + size_t(k) < n; ++k) {
					uint32_t c = t[i + k];
					low[k] = char((c >= 'A' && c <= 'Z') ? c + 32 : (c < 128 ? c : 0));
				}
				size_t best = 0;
				for (const char* c : kContr) {
					size_t len = std::strlen(c);
					if (i + len <= n && std::strncmp(low, c, len) == 0 && len > best) best = len;
				}
				if (best) { pieces.emplace_back(i, i + best); i += best; continue; }
			}
			// 2. [^\r\n\p{L}\p{N}]? \p{L}+
			size_t j = i;
			if (!is_nl(ch) && !is_letter(ch) && !is_number(ch)) {
				if (j + 1 < n && is_letter(t[j + 1])) ++j;
			}
			if (j < n && is_letter(t[j])) {
				size_t k = j;
				while (k < n && is_letter(t[k])) ++k;
				pieces.emplace_back(i, k); i = k; continue;
			}
			// 3. \p{N}  (single)
			if (is_number(ch)) { pieces.emplace_back(i, i + 1); ++i; continue; }
			// 4.  ?[^\s\p{L}\p{N}]+[\r\n]*
			j = i;
			if (ch == ' ') j = i + 1;
			if (j < n && !is_space(t[j]) && !is_letter(t[j]) && !is_number(t[j])) {
				size_t k = j;
				while (k < n && !is_space(t[k]) && !is_letter(t[k]) && !is_number(t[k])) ++k;
				while (k < n && is_nl(t[k])) ++k;
				pieces.emplace_back(i, k); i = k; continue;
			}
			// 5. \s*[\r\n]+   6. \s+(?!\S)   7. \s+
			if (is_space(ch)) {
				size_t k = i;
				while (k < n && is_space(t[k])) ++k;
				size_t last_nl = SIZE_MAX;
				for (size_t p = i; p < k; ++p) if (is_nl(t[p])) last_nl = p;
				if (last_nl != SIZE_MAX) { pieces.emplace_back(i, last_nl + 1); i = last_nl + 1; continue; }
				if (k < n) {
					if (k - i > 1) { pieces.emplace_back(i, k - 1); i = k - 1; continue; }
					pieces.emplace_back(i, k); i = k; continue;
				}
				pieces.emplace_back(i, k); i = k; continue;
			}
			// fallback (unreachable for well-formed input)
			pieces.emplace_back(i, i + 1); ++i;
		}
	}

	// --- byte-level BPE on a piece's UTF-8 bytes (mirrors _bpe) --------------------
	void bpe(const std::string& piece, std::vector<int32_t>& out) const {
		std::vector<int32_t> ids;
		ids.reserve(piece.size());
		for (unsigned char b : piece) ids.push_back(int32_t(byte_to_id[b]));
		while (ids.size() >= 2) {
			int32_t best_rank = -1;
			size_t best_pos = 0;
			for (size_t p = 0; p + 1 < ids.size(); ++p) {
				auto it = merges.find((uint64_t(uint32_t(ids[p])) << 32) | uint32_t(ids[p + 1]));
				if (it != merges.end() && (best_rank < 0 || it->second.first < best_rank)) {
					best_rank = it->second.first;
					best_pos = p;
				}
			}
			if (best_rank < 0) break;
			auto it = merges.find((uint64_t(uint32_t(ids[best_pos])) << 32) | uint32_t(ids[best_pos + 1]));
			ids[best_pos] = it->second.second;
			ids.erase(ids.begin() + best_pos + 1);
		}
		out.insert(out.end(), ids.begin(), ids.end());
	}

	void encode(std::string_view text, std::vector<int32_t>& out) const {
		const std::string s(text);
		size_t i = 0, n = s.size();
		auto special_at = [&](size_t pos) -> const std::pair<std::string, int32_t>* {
			for (const auto& sp : specials) {  // longest-first
				if (!sp.first.empty() && pos + sp.first.size() <= n &&
				    std::memcmp(s.data() + pos, sp.first.data(), sp.first.size()) == 0)
					return &sp;
			}
			return nullptr;
		};
		while (i < n) {
			if (const auto* sp = special_at(i)) {
				out.push_back(sp->second);
				i += sp->first.size();
				continue;
			}
			size_t j = i;
			while (j < n && special_at(j) == nullptr) ++j;
			// text span [i,j): NFC -> pretokenize -> byte-level BPE
			std::vector<uint32_t> cps, norm;
			Utf8Decode(std::string_view(s.data() + i, j - i), cps);
			nfc(cps, norm);
			std::vector<std::pair<size_t, size_t>> pieces;
			pretokenize(norm, pieces);
			std::string piece;
			for (const auto& pr : pieces) {
				piece.clear();
				for (size_t k = pr.first; k < pr.second; ++k) Utf8Append(norm[k], piece);
				bpe(piece, out);
			}
			i = j;
		}
	}

	// S-HARDEN-2 (F7): concatenate every id's raw vocabulary bytes FIRST -- so a
	// multi-byte UTF-8 sequence whose bytes are split across two adjacent tokens
	// reassembles correctly -- then run the one strict decoder (Utf8Decode, shared
	// with encode()) and re-encode, so a malformed byte sequence anywhere in the
	// concatenation surfaces as U+FFFD per the documented policy
	// (include/superslm/tokenizer.h) rather than passing through as raw bytes.
	void decode(const std::vector<int32_t>& ids, std::string& out) const {
		std::string raw;
		for (int32_t id : ids) {
			if (id < 0 || uint32_t(id) >= vocab_count) continue;
			uint32_t off = Rd32(vocab_offsets + size_t(id) * 4);
			uint32_t end = Rd32(vocab_offsets + (size_t(id) + 1) * 4);
			raw.append(reinterpret_cast<const char*>(vocab_blob) + off, end - off);
		}
		std::vector<uint32_t> cps;
		Utf8Decode(raw, cps);
		out.reserve(out.size() + raw.size());
		for (uint32_t cp : cps) Utf8Append(cp, out);
	}
};

namespace {

// Parse the TOK1 blob (tools/convert_tokenizer.py serialize_tokenizer). Integrity is
// already verified by the loader, so this trusts the bytes and only fails on a bad
// magic or a length that would overrun the section.
bool ParseTok(const uint8_t* d, size_t sz, TokenizerView::Impl& im, std::string* err) {
	auto fail = [&](const char* m) { if (err) *err = m; return false; };
	// 64-bit length math throughout: counts are attacker-controlled u32, so a u32
	// `count + 1` or `count * k` must not wrap before the bound check (Poirot S1-2).
	auto need = [&](uint64_t off, uint64_t len) { return off + len <= uint64_t(sz); };
	if (!need(0, 24) || std::memcmp(d, "TOK1", 4) != 0) return fail("Tokenizer: bad TOK1 header");
	// S-HARDEN-2 (F6): offset 4 (version) and offset 20 (reserved) are declared by
	// the format (docs/sslm_format.md "Tokenizer blob -- TOK1") and were never read
	// by any code path before this -- a guard-vitality gap (§17 dim 11): the format
	// claims both are enforced and nothing enforced either.
	if (Rd32(d + 4) != kTokBlobVersion) return fail("Tokenizer: unsupported TOK1 version");
	uint32_t vocab = Rd32(d + 8), merge = Rd32(d + 12), special = Rd32(d + 16);
	if (Rd32(d + 20) != 0) return fail("Tokenizer: TOK1 reserved field != 0");
	// S-HARDEN-2 (F18): vocab_count == 0 encodes no byte and indexes no token; every
	// id this parser stores is narrowed to int32_t downstream, so a vocab_count that
	// cannot be represented as a non-negative int32 is rejected before it can make
	// that narrowing (or the byte/merge/special bound checks below, which compare
	// against `vocab` as read) silently wrong.
	if (vocab == 0) return fail("Tokenizer: vocab_count == 0");
	if (vocab > kTokVocabCountMax) return fail("Tokenizer: vocab_count exceeds INT32_MAX");
	im.vocab_count = vocab;
	size_t pos = 24;
	if (!need(pos, 256 * 4)) return fail("Tokenizer: truncated byte_to_id");
	for (int b = 0; b < 256; ++b) {
		const uint32_t id = Rd32(d + pos + size_t(b) * 4);
		// S-HARDEN-2 (F18): every one of the 256 byte->id entries must index a real
		// vocabulary slot -- an id here that Encode() can emit and Decode()/the
		// forward path's token-embedding lookup then cannot bound is exactly the
		// defect a committed test used to require (MakeMinimalValidTok1's <eos> at
		// id 1000 against a 4-entry vocab).
		if (id >= vocab) return fail("Tokenizer: byte_to_id entry >= vocab_count");
		im.byte_to_id[b] = id;
	}
	pos += 256 * 4;
	if (!need(pos, (uint64_t(vocab) + 1) * 4)) return fail("Tokenizer: truncated vocab offsets");
	im.vocab_offsets = d + pos;
	pos += (uint64_t(vocab) + 1) * 4;
	if (!need(pos, 4)) return fail("Tokenizer: truncated vocab blob_len");
	uint32_t vblob = Rd32(d + pos); pos += 4;
	if (!need(pos, vblob)) return fail("Tokenizer: truncated vocab blob");
	im.vocab_blob = d + pos;
	pos += vblob;
	// Validate the vocab offset table (Poirot S1-1): it indexes vocab_blob and is read
	// unchecked by Decode, so — like the special and decomp tables — every offset must
	// be non-decreasing and within the blob, or a crafted artifact makes Decode read
	// out of bounds. Monotonic + last <= vblob ⇒ every [off[i], off[i+1]) is in range.
	{
		uint32_t prev = 0;
		for (uint64_t i = 0; i <= vocab; ++i) {
			uint32_t o = Rd32(im.vocab_offsets + i * 4);
			if (o < prev || o > vblob) return fail("Tokenizer: vocab offset out of range");
			prev = o;
		}
	}
	if (!need(pos, size_t(merge) * 12)) return fail("Tokenizer: truncated merges");
	im.merges.reserve(merge);
	for (uint32_t r = 0; r < merge; ++r) {
		uint32_t a = Rd32(d + pos), b = Rd32(d + pos + 4), m = Rd32(d + pos + 8);
		// S-HARDEN-2 (F18): a merge's operands and result all become ids Encode()
		// can emit (`bpe`'s `ids[best_pos] = it->second.second`); each must index
		// the vocabulary the same way byte_to_id's entries must.
		if (a >= vocab || b >= vocab || m >= vocab)
			return fail("Tokenizer: merge operand or result >= vocab_count");
		im.merges.emplace((uint64_t(a) << 32) | b, std::make_pair(int32_t(r), int32_t(m)));
		pos += 12;
	}
	if (!need(pos, uint64_t(special) * 4)) return fail("Tokenizer: truncated special ids");
	std::vector<uint32_t> sids(special);
	for (uint32_t i = 0; i < special; ++i) {
		const uint32_t sid = Rd32(d + pos);
		// S-HARDEN-2 (F18): the exact defect the review's finding named -- a special
		// id outside the vocabulary is emitted verbatim by encode()'s special-token
		// path and then indexes past the token embedding once a forward kernel
		// exists (F18's "out-of-range index into the token embedding").
		if (sid >= vocab) return fail("Tokenizer: special id >= vocab_count");
		sids[i] = sid; pos += 4;
	}
	if (!need(pos, (uint64_t(special) + 1) * 4)) return fail("Tokenizer: truncated special offsets");
	std::vector<uint32_t> soff(special + 1);
	for (uint32_t i = 0; i <= special; ++i) { soff[i] = Rd32(d + pos); pos += 4; }
	if (!need(pos, 4)) return fail("Tokenizer: truncated special blob_len");
	uint32_t sblob = Rd32(d + pos); pos += 4;
	if (!need(pos, sblob)) return fail("Tokenizer: truncated special blob");
	const uint8_t* sb = d + pos;
	im.specials.reserve(special);
	for (uint32_t i = 0; i < special; ++i) {
		if (soff[i] > soff[i + 1] || soff[i + 1] > sblob) return fail("Tokenizer: bad special offset");
		im.specials.emplace_back(std::string(reinterpret_cast<const char*>(sb) + soff[i],
		                                      soff[i + 1] - soff[i]), int32_t(sids[i]));
	}
	return true;
}

bool ReadRanges(const uint8_t* d, size_t sz, size_t& pos, std::vector<Range>& r, std::string* err) {
	if (pos + 4 > sz) { if (err) *err = "UnicodeTables: truncated range count"; return false; }
	uint32_t cnt = Rd32(d + pos); pos += 4;
	if (pos + size_t(cnt) * 8 > sz) { if (err) *err = "UnicodeTables: truncated ranges"; return false; }
	r.reserve(cnt);
	for (uint32_t i = 0; i < cnt; ++i) { r.emplace_back(Rd32(d + pos), Rd32(d + pos + 4)); pos += 8; }
	return true;
}

// Parse the UNI1 blob (tools/unicode_tables.py Unicode.serialize).
bool ParseUni(const uint8_t* d, size_t sz, TokenizerView::Impl& im, std::string* err) {
	auto fail = [&](const char* m) { if (err) *err = m; return false; };
	if (sz < 8 || std::memcmp(d, "UNI1", 4) != 0) return fail("UnicodeTables: bad UNI1 header");
	// S-HARDEN-2 (F6): the version field at offset 4 was skipped past (`pos = 8`)
	// but never actually read and compared -- the same guard-vitality gap as TOK1's
	// version/reserved fields (§17 dim 11): the format declares `u32 = 1` and no
	// code path could ever reject a different value.
	if (Rd32(d + 4) != kUniBlobVersion) return fail("UnicodeTables: unsupported UNI1 version");
	size_t pos = 8; // magic + blob version
	if (!ReadRanges(d, sz, pos, im.letter, err)) return false;
	if (!ReadRanges(d, sz, pos, im.number, err)) return false;
	if (!ReadRanges(d, sz, pos, im.space, err)) return false;
	if (pos + 4 > sz) return fail("UnicodeTables: truncated ccc count");
	uint32_t ccc_n = Rd32(d + pos); pos += 4;
	if (pos + size_t(ccc_n) * 8 > sz) return fail("UnicodeTables: truncated ccc");
	for (uint32_t i = 0; i < ccc_n; ++i) {
		im.ccc_map[Rd32(d + pos)] = uint8_t(Rd32(d + pos + 4)); pos += 8;
	}
	if (pos + 4 > sz) return fail("UnicodeTables: truncated decomp count");
	uint32_t dn = Rd32(d + pos); pos += 4;
	if (pos + size_t(dn) * 4 > sz) return fail("UnicodeTables: truncated decomp cps");
	std::vector<uint32_t> dcps(dn);
	for (uint32_t i = 0; i < dn; ++i) { dcps[i] = Rd32(d + pos); pos += 4; }
	if (uint64_t(pos) + (uint64_t(dn) + 1) * 4 > uint64_t(sz)) return fail("UnicodeTables: truncated decomp offsets");
	std::vector<uint32_t> doff(dn + 1);
	for (uint32_t i = 0; i <= dn; ++i) { doff[i] = Rd32(d + pos); pos += 4; }
	if (pos + 4 > sz) return fail("UnicodeTables: truncated decomp seq_len");
	uint32_t seq_len = Rd32(d + pos); pos += 4;
	if (pos + size_t(seq_len) * 4 > sz) return fail("UnicodeTables: truncated decomp seq");
	const uint8_t* seq = d + pos;
	for (uint32_t i = 0; i < dn; ++i) {
		if (doff[i] > doff[i + 1] || doff[i + 1] > seq_len) return fail("UnicodeTables: bad decomp offset");
		std::vector<uint32_t> s;
		for (uint32_t k = doff[i]; k < doff[i + 1]; ++k) s.push_back(Rd32(seq + size_t(k) * 4));
		im.decomp.emplace(dcps[i], std::move(s));
	}
	pos += size_t(seq_len) * 4;
	if (pos + 4 > sz) return fail("UnicodeTables: truncated compose count");
	uint32_t cn = Rd32(d + pos); pos += 4;
	if (pos + size_t(cn) * 12 > sz) return fail("UnicodeTables: truncated compose");
	for (uint32_t i = 0; i < cn; ++i) {
		uint32_t a = Rd32(d + pos), b = Rd32(d + pos + 4), c = Rd32(d + pos + 8);
		im.compose[(uint64_t(a) << 32) | b] = c;
		pos += 12;
	}
	return true;
}

} // namespace

TokenizerView::TokenizerView() = default;
TokenizerView::~TokenizerView() = default;
TokenizerView::TokenizerView(TokenizerView&&) noexcept = default;
TokenizerView& TokenizerView::operator=(TokenizerView&&) noexcept = default;

bool TokenizerView::Open(const SslmArtifact& artifact, TokenizerView& out, std::string* err) {
	out = TokenizerView{};
	const SslmSectionView* tok = artifact.Section(SslmSectionType::Tokenizer);
	const SslmSectionView* uni = artifact.Section(SslmSectionType::UnicodeTables);
	if (!tok) { if (err) *err = "artifact has no Tokenizer section"; return false; }
	if (!uni) { if (err) *err = "artifact has no UnicodeTables section"; return false; }
	auto im = std::make_unique<Impl>();
	if (!ParseTok(tok->data, size_t(tok->byte_size), *im, err)) return false;
	if (!ParseUni(uni->data, size_t(uni->byte_size), *im, err)) return false;
	im->ok = true;
	out.impl_ = std::move(im);
	return true;
}

bool TokenizerView::Ok() const noexcept { return impl_ && impl_->ok; }

int32_t TokenizerView::VocabSize() const noexcept {
	return impl_ ? int32_t(impl_->vocab_count) : 0;
}

std::vector<int32_t> TokenizerView::Encode(std::string_view text) const {
	std::vector<int32_t> out;
	if (impl_ && impl_->ok) impl_->encode(text, out);
	return out;
}

std::string TokenizerView::Decode(const std::vector<int32_t>& ids) const {
	std::string out;
	if (impl_ && impl_->ok) impl_->decode(ids, out);
	return out;
}

} // namespace superslm

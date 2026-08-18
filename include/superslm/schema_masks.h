// schema_masks.h -- the SchemaMasks (SCM1) artifact-section reader for schema-constrained
// decoding: the fixed header, the per-schema descriptor table, the name blob, and the four
// per-schema data blocks (mask pages, CSR state-offset array, accepting-state array, flattened
// transition array), plus every load-time structural and cross-check rejection the format
// defines.
//
// Header-only: every function below is `inline`, defined here with no companion .cpp, so this
// file compiles directly into whichever translation unit includes it.
//
// Every SchemaMasks structural or cross-check rejection is enforced -- `Parse` returns false
// and a diagnostic string on any violation, never a silent partial parse. A caller mapping a
// model (`sslm_model_map`) sees any SchemaMasks rejection surfaced as the existing
// `SSLM_ARTIFACT_REJECTED` status, the same family used for "this artifact's content cannot be
// used as requested, no more specific status applies" -- not a dedicated per-reason status.
// The rejection's own diagnostic string still names the specific reason for a caller that logs
// or displays it.
#ifndef SUPERSLM_SCHEMA_MASKS_H
#define SUPERSLM_SCHEMA_MASKS_H

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace superslm {

namespace schema_masks_detail {

inline uint32_t ReadLE32(const uint8_t* p) {
	return static_cast<uint32_t>(p[0]) | (static_cast<uint32_t>(p[1]) << 8) |
	       (static_cast<uint32_t>(p[2]) << 16) | (static_cast<uint32_t>(p[3]) << 24);
}

inline bool CheckedAddU64(uint64_t a, uint64_t b, uint64_t* out) {
	if (a > UINT64_MAX - b) return false;
	*out = a + b;
	return true;
}
inline bool CheckedMulU64(uint64_t a, uint64_t b, uint64_t* out) {
	if (a != 0 && b > UINT64_MAX / a) return false;
	*out = a * b;
	return true;
}

struct Span {
	uint64_t off;
	uint64_t len;
	uint64_t End() const { return off + len; }
};

inline bool Overlaps(const Span& a, const Span& b) {
	return a.off < b.End() && b.off < a.End();
}

}  // namespace schema_masks_detail

// Sec13.5's named bounds, generous headroom over the real reference case (~594 states, the
// rung-5 strike's own cited number) rather than a tight fit -- reproduced here (not
// model.h, per this file's own scoping note above) since this is the one place they gate.
inline constexpr uint32_t kMaxSchemas = 65536;
inline constexpr uint32_t kMaxSchemaStates = 1048576;
inline constexpr uint32_t kMaxSchemaTransitions = 16777216;

// FNV-1a 64, the project's own house name-hash convention (Sec13.1/13.4:
// tools/sslm_layer_trace.cpp's Fnv1a64, standard offset basis/prime) -- reproduced here as the
// C++-side computation the save-blob's schema_name_hash field and restore-time resolution both
// need. A pure, parameterless, standard algorithm -- not project-specific logic duplicated
// across TUs.
inline uint64_t Fnv1a64(const void* data, size_t len) noexcept {
	const uint8_t* p = static_cast<const uint8_t*>(data);
	uint64_t h = 14695981039346656037ull;
	for (size_t i = 0; i < len; ++i) {
		h ^= p[i];
		h *= 1099511628211ull;
	}
	return h;
}
inline uint64_t Fnv1a64(std::string_view s) noexcept { return Fnv1a64(s.data(), s.size()); }

// One compiled schema's own view into the section's bytes -- every pointer/span below points
// directly into the caller-supplied `section_data` (SslmModelView::Section(SchemaMasks)'s own
// bytes, which live for the owning sslm_model_s's whole lifetime, exactly the same
// artifact-lifetime-bound convention SslmSectionView already establishes). No per-entry copy;
// multi-byte fields are decoded on read (ReadLE32/64), matching every other sub-format in this
// codebase's own "explicit little-endian byte assembly, never a struct cast" rule.
struct SchemaEntry {
	std::string_view name;       // points into the section's own name blob
	uint64_t name_hash = 0;      // Fnv1a64(name) -- computed once at parse time
	uint32_t state_count = 0;
	uint32_t accepting_count = 0;
	uint32_t transition_count = 0;
	const uint8_t* mask_pages = nullptr;       // state_count * mask_page_bytes bytes
	const uint8_t* state_offsets_le = nullptr; // (state_count+1) LE u32
	const uint8_t* accepting_le = nullptr;     // accepting_count LE u32, strictly ascending
	const uint8_t* transitions_le = nullptr;   // transition_count * (u32 token_id, u32 next_state) LE,
	                                            // state-major, token_id ascending per row
};

// A parsed, fully validated SCM1 section -- one per sslm_model_s, built once at
// sslm_model_map time (Sec13.1: "an artifact with no compiled schemas carries no SchemaMasks
// section at all" -- a default-constructed, empty table is exactly that case, Count()==0).
class SchemaMasksTable {
public:
	SchemaMasksTable() = default;

	// Parses `section_data[0, section_size)` (a SchemaMasks/type-30 section's own bytes,
	// section-relative offsets throughout, per Sec13.2) against `config_vocab_size` (the
	// semantic cross-check, Sec13.3's own "SchemaMasks.vocab_size != Config.vocab_size"
	// rejection). On success, `out`'s entries point INTO `section_data` -- the caller (
	// sslm_model_map) must keep the artifact's own owned bytes alive for as long as `out` is
	// used, exactly as every other SslmModelView-derived view already requires. On any
	// structural/cross-check violation, returns false and `err` (if non-null) carries a
	// diagnostic naming the Sec13.3 rejection family; `out` is left empty.
	static bool Parse(const uint8_t* section_data, uint64_t section_size,
	                   uint32_t config_vocab_size, SchemaMasksTable& out, std::string* err);

	size_t Count() const noexcept { return entries_.size(); }
	uint32_t VocabSize() const noexcept { return vocab_size_; }
	uint32_t MaskPageBytes() const noexcept { return mask_page_bytes_; }

	const SchemaEntry* ByIndex(size_t index) const noexcept {
		return index < entries_.size() ? &entries_[index] : nullptr;
	}
	// Byte-exact, case-sensitive match (Sec13.2's own ABI-binding text). Returns nullptr and
	// leaves *out_index untouched on no match.
	const SchemaEntry* ByName(std::string_view name, size_t* out_index = nullptr) const noexcept {
		for (size_t i = 0; i < entries_.size(); ++i) {
			if (entries_[i].name == name) {
				if (out_index) *out_index = i;
				return &entries_[i];
			}
		}
		return nullptr;
	}
	// Resolves by Fnv1a64 name hash (the save-blob restore path, Sec13.4) -- a 64-bit collision
	// between two distinct registered names is not specially handled, per Sec13.4's own text.
	const SchemaEntry* ByNameHash(uint64_t hash, size_t* out_index = nullptr) const noexcept {
		for (size_t i = 0; i < entries_.size(); ++i) {
			if (entries_[i].name_hash == hash) {
				if (out_index) *out_index = i;
				return &entries_[i];
			}
		}
		return nullptr;
	}

	// True iff bit `token_id` of `entry`'s own page(state) is set. `state` and `token_id` are
	// trusted to already be in range (callers check state_count/vocab_size themselves, matching
	// the runtime's own "table lookup, no bounds re-check on the hot path" convention every
	// other decode-time primitive in this codebase already follows).
	bool MaskBit(const SchemaEntry& entry, uint32_t state, uint32_t token_id) const noexcept {
		const uint8_t* page = entry.mask_pages + static_cast<size_t>(state) * mask_page_bytes_;
		return (page[token_id >> 3] >> (token_id & 7)) & 1u;
	}

	// Binary search of `entry`'s own CSR row for `state` -> `token_id`'s transition. Returns
	// true and sets *next_state on a hit (token_id IS a legal continuation at state); false
	// (no *next_state write) on a miss -- the "reachability" query G5-4's template-fill check
	// and the DFA-walk advance both consult, per Sec13.2's own CSR-row construction (strictly
	// ascending token_id per row, by the compiler's own sorted(targets) construction).
	bool Transition(const SchemaEntry& entry, uint32_t state, uint32_t token_id,
	                 uint32_t* next_state) const noexcept {
		using schema_masks_detail::ReadLE32;
		const uint32_t row_begin = ReadLE32(entry.state_offsets_le + static_cast<size_t>(state) * 4);
		const uint32_t row_end =
		    ReadLE32(entry.state_offsets_le + static_cast<size_t>(state + 1) * 4);
		uint32_t lo = row_begin, hi = row_end;
		while (lo < hi) {
			const uint32_t mid = lo + (hi - lo) / 2;
			const uint32_t tid = ReadLE32(entry.transitions_le + static_cast<size_t>(mid) * 8);
			if (tid == token_id) {
				if (next_state) {
					*next_state = ReadLE32(entry.transitions_le + static_cast<size_t>(mid) * 8 + 4);
				}
				return true;
			}
			if (tid < token_id) lo = mid + 1; else hi = mid;
		}
		return false;
	}

private:
	std::vector<SchemaEntry> entries_;
	uint32_t vocab_size_ = 0;
	uint32_t mask_page_bytes_ = 0;
};

inline bool SchemaMasksTable::Parse(const uint8_t* data, uint64_t size, uint32_t config_vocab_size,
                                     SchemaMasksTable& out, std::string* err) {
	using namespace schema_masks_detail;
	out = SchemaMasksTable{};
	auto fail = [&](const char* msg) {
		if (err) *err = msg;
		return false;
	};

	if (size < 24) return fail("SchemaMasksTruncated: section shorter than the 24-byte fixed header");
	if (std::memcmp(data, "SCM1", 4) != 0) return fail("BadSchemaMasksMagic");
	const uint32_t version = ReadLE32(data + 4);
	if (version != 1) return fail("UnsupportedSchemaMasksVersion");
	const uint32_t schema_count = ReadLE32(data + 8);
	const uint32_t vocab_size = ReadLE32(data + 12);
	const uint32_t name_blob_len = ReadLE32(data + 16);
	const uint32_t reserved = ReadLE32(data + 20);
	if (reserved != 0) return fail("BadSchemaMasksReserved");
	if (schema_count == 0 || schema_count > kMaxSchemas) return fail("TooManySchemas");
	if (vocab_size == 0) return fail("BadSchemaVocabSize");
	// Semantic cross-check (Sec13.3): checked here, before the per-schema structural walk below,
	// since it is cheap and would otherwise waste work parsing a section that can never bind
	// correctly against this artifact's own Config anyway.
	if (vocab_size != config_vocab_size) return fail("SchemaMasksVocabSizeMismatch");

	uint64_t manifest_end = 0;
	{
		uint64_t desc_bytes = 0, t1 = 0;
		if (!CheckedMulU64(static_cast<uint64_t>(schema_count), 56, &desc_bytes)) {
			return fail("SchemaMasksManifestOutOfBounds: descriptor table overflow");
		}
		if (!CheckedAddU64(24, desc_bytes, &t1) ||
		    !CheckedAddU64(t1, static_cast<uint64_t>(name_blob_len), &manifest_end) ||
		    manifest_end > size) {
			return fail("SchemaMasksManifestOutOfBounds");
		}
	}
	const uint64_t name_blob_off = manifest_end - name_blob_len;
	// M3 (Claude/Poirot/9bc9ec6-t2132-g5-arc-review.md): widened to uint64_t before the `+7`--
	// `vocab_size` is untrusted `u32` input (only checked `!= 0` above; this is a hostile-input
	// section, design Sec7 dim2) and `vocab_size + 7u` in uint32_t arithmetic wraps for any
	// `vocab_size > 0xFFFFFFF8`, silently truncating `mask_page_bytes` to a small, wrong value
	// that would then pass the CheckedMulU64 bounds checks below as if the artifact declared a
	// tiny vocabulary. `mask_page_bytes` itself is still representable in `uint32_t` for any
	// `vocab_size` this format can otherwise support (`ceil(2^32/8)` fits in 32 bits with room to
	// spare), so only the intermediate `+7` needs the wider type.
	const uint64_t mask_page_bytes64 = (static_cast<uint64_t>(vocab_size) + 7u) / 8u;
	const uint32_t mask_page_bytes = static_cast<uint32_t>(mask_page_bytes64);

	std::vector<SchemaEntry> entries;
	entries.reserve(schema_count);
	std::vector<std::string_view> seen_names;
	seen_names.reserve(schema_count);
	std::vector<Span> occupied;  // every data block placed so far, for overlap checking

	for (uint32_t i = 0; i < schema_count; ++i) {
		const uint8_t* d = data + 24 + static_cast<uint64_t>(i) * 56;
		const uint32_t name_off = ReadLE32(d + 0);
		const uint32_t name_len = ReadLE32(d + 4);
		const uint32_t state_count = ReadLE32(d + 8);
		const uint32_t accepting_count = ReadLE32(d + 12);
		const uint32_t transition_count = ReadLE32(d + 16);
		const uint32_t desc_reserved = ReadLE32(d + 20);
		// M1 (Claude/Poirot/9bc9ec6-t2132-g5-arc-review.md): design Sec13.2's descriptor table
		// DOES constrain this field to `== 0` (the same row as the fixed header's own `reserved`
		// field, checked above) -- the comment previously here claimed the opposite. Corrected;
		// the check itself was simply missing.
		const uint64_t mask_pages_off = ReadLE32(d + 24) | (static_cast<uint64_t>(ReadLE32(d + 28)) << 32);
		const uint64_t state_offsets_off = ReadLE32(d + 32) | (static_cast<uint64_t>(ReadLE32(d + 36)) << 32);
		const uint64_t accepting_off = ReadLE32(d + 40) | (static_cast<uint64_t>(ReadLE32(d + 44)) << 32);
		const uint64_t transitions_off = ReadLE32(d + 48) | (static_cast<uint64_t>(ReadLE32(d + 52)) << 32);
		if (desc_reserved != 0) return fail("BadSchemaMasksReserved");

		if (name_len == 0 || static_cast<uint64_t>(name_off) + name_len > name_blob_len) {
			return fail("BadSchemaName");
		}
		std::string_view name(reinterpret_cast<const char*>(data + name_blob_off + name_off), name_len);
		for (auto other : seen_names) {
			if (other == name) return fail("DuplicateSchemaName");
		}
		seen_names.push_back(name);

		if (state_count == 0 || state_count > kMaxSchemaStates) return fail("BadSchemaStateCount");
		if (accepting_count > state_count) return fail("BadSchemaAcceptingSet");
		if (transition_count > kMaxSchemaTransitions) return fail("TooManySchemaTransitions");

		uint64_t mask_pages_len = 0, state_offsets_len = 0, accepting_len = 0, transitions_len = 0;
		if (!CheckedMulU64(static_cast<uint64_t>(state_count), mask_page_bytes, &mask_pages_len) ||
		    !CheckedMulU64(static_cast<uint64_t>(state_count) + 1, 4, &state_offsets_len) ||
		    !CheckedMulU64(static_cast<uint64_t>(accepting_count), 4, &accepting_len) ||
		    !CheckedMulU64(static_cast<uint64_t>(transition_count), 8, &transitions_len)) {
			return fail("SchemaMasksOutOfBounds: block size overflow");
		}
		const Span blocks[4] = {
		    {mask_pages_off, mask_pages_len},
		    {state_offsets_off, state_offsets_len},
		    {accepting_off, accepting_len},
		    {transitions_off, transitions_len},
		};
		for (const Span& b : blocks) {
			uint64_t end = 0;
			if (!CheckedAddU64(b.off, b.len, &end) || b.off < manifest_end || end > size) {
				return fail("SchemaMasksOutOfBounds");
			}
		}
		for (const Span& b : blocks) {
			for (const Span& prior : occupied) {
				if (Overlaps(b, prior)) return fail("SchemaMasksBlockOverlap");
			}
		}
		for (int a = 0; a < 4; ++a) {
			for (int b = a + 1; b < 4; ++b) {
				if (Overlaps(blocks[a], blocks[b])) return fail("SchemaMasksBlockOverlap");
			}
		}
		for (const Span& b : blocks) occupied.push_back(b);

		const uint8_t* state_offsets_le = data + state_offsets_off;
		if (ReadLE32(state_offsets_le) != 0) return fail("BadSchemaStateOffsets");
		if (ReadLE32(state_offsets_le + static_cast<size_t>(state_count) * 4) != transition_count) {
			return fail("BadSchemaStateOffsets");
		}
		for (uint32_t s = 0; s < state_count; ++s) {
			if (ReadLE32(state_offsets_le + static_cast<size_t>(s) * 4) >
			    ReadLE32(state_offsets_le + static_cast<size_t>(s + 1) * 4)) {
				return fail("BadSchemaStateOffsets");
			}
		}

		const uint8_t* accepting_le = data + accepting_off;
		uint32_t prev_accept = 0;
		for (uint32_t a = 0; a < accepting_count; ++a) {
			const uint32_t v = ReadLE32(accepting_le + static_cast<size_t>(a) * 4);
			if (v >= state_count) return fail("BadSchemaAcceptingSet");
			if (a > 0 && v <= prev_accept) return fail("BadSchemaAcceptingSet");
			prev_accept = v;
		}

		const uint8_t* transitions_le = data + transitions_off;
		for (uint32_t s = 0; s < state_count; ++s) {
			const uint32_t row_begin = ReadLE32(state_offsets_le + static_cast<size_t>(s) * 4);
			const uint32_t row_end = ReadLE32(state_offsets_le + static_cast<size_t>(s + 1) * 4);
			uint32_t prev_tok = 0;
			for (uint32_t k = row_begin; k < row_end; ++k) {
				const uint32_t tok = ReadLE32(transitions_le + static_cast<size_t>(k) * 8);
				const uint32_t next = ReadLE32(transitions_le + static_cast<size_t>(k) * 8 + 4);
				if (tok >= vocab_size || next >= state_count) return fail("BadSchemaTransitionTable");
				if (k > row_begin && tok <= prev_tok) return fail("BadSchemaTransitionTable");
				prev_tok = tok;
			}
		}

		// Cross-check (Sec13.3, the one rejection this format adds beyond direct precedent):
		// mask bit `token_id` at state `i` set IFF `token_id` appears in state i's CSR row.
		const uint8_t* mask_pages = data + mask_pages_off;
		for (uint32_t s = 0; s < state_count; ++s) {
			const uint8_t* page = mask_pages + static_cast<size_t>(s) * mask_page_bytes;
			const uint32_t row_begin = ReadLE32(state_offsets_le + static_cast<size_t>(s) * 4);
			const uint32_t row_end = ReadLE32(state_offsets_le + static_cast<size_t>(s + 1) * 4);
			uint32_t row_cursor = row_begin;
			for (uint32_t t = 0; t < vocab_size; ++t) {
				const bool bit_set = (page[t >> 3] >> (t & 7)) & 1u;
				bool in_row = false;
				if (row_cursor < row_end) {
					const uint32_t row_tok = ReadLE32(transitions_le + static_cast<size_t>(row_cursor) * 8);
					if (row_tok == t) {
						in_row = true;
						++row_cursor;
					}
				}
				if (bit_set != in_row) return fail("SchemaMasksMaskTransitionMismatch");
			}
		}

		SchemaEntry entry;
		entry.name = name;
		entry.name_hash = Fnv1a64(name);
		entry.state_count = state_count;
		entry.accepting_count = accepting_count;
		entry.transition_count = transition_count;
		entry.mask_pages = mask_pages;
		entry.state_offsets_le = state_offsets_le;
		entry.accepting_le = accepting_le;
		entry.transitions_le = transitions_le;
		entries.push_back(entry);
	}

	out.entries_ = std::move(entries);
	out.vocab_size_ = vocab_size;
	out.mask_page_bytes_ = mask_page_bytes;
	return true;
}

}  // namespace superslm

#endif  // SUPERSLM_SCHEMA_MASKS_H

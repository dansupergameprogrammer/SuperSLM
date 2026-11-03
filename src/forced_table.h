// Internal (not installed, not exported). Schema fast-decode plan rev 1.4.4 §7.3, slice G1-F: the forced-next
// table builder, lifted verbatim out of src/sslm_abi.cpp (J1) so both maps build the one table. sslm_model_map (CPU) and
// sslm_gpu_model_map (GPU) call BuildForcedTables over the model's parsed SCM1 table and its TOK1 view; the CPU
// output is unchanged (cell 6.6's golden hashes pin it), and the GPU's equals it (cell G12.4).
//
// §5.1's rule, per schema and state: next_forced(s) (-1 when s is unforced) and |F(s)| (0 when unforced). With
// it come the per-token `draftable` flags (J1-X) and, for slice M1/G1-M, each schema's CSR legal-id list.
#ifndef SUPERSLM_SRC_FORCED_TABLE_H
#define SUPERSLM_SRC_FORCED_TABLE_H

// The shared source is this header: its functions have internal linkage (an unnamed namespace), so
// src/sslm_abi.cpp and src/gpu/gpu_1p0.cpp each compile their own copy from the same text. A header, not a new
// .cpp, because every engine source list in the tree (CMake, build.bat, the box and cloud GPU builds, the red
// suites' scripts) and the archive-member census of the fp-free gate would otherwise change with it; the
// builder's text is the same either way.
#include <cstdint>
#include <cstring>
#include <string>
#include <string_view>
#include <vector>

#include "superslm/schema_masks.h"
#include "superslm/tokenizer.h"
#include "bad_alloc_wrap.h"
#include "tokenizer_tables.h"

namespace superslm {
namespace forced_table {

struct ForcedEntry {
	int32_t next_forced = -1;
	uint32_t f_len = 0;
};

struct ForcedTables {
	// False: no table (no schema, or no tokenizer); every query and verb refuses.
	bool present = false;
	// Per schema, per state.
	std::vector<std::vector<ForcedEntry>> table;
	// Per token id: 1 when an exact-mode draft may carry it (not a special id, and it has bytes).
	std::vector<uint8_t> draftable;
};

// The functions, in order:
//  - IsAcceptingState: true iff `state` is in the schema's accepting set (SCM1 stores it strictly ascending).
//  - BuildLegalIds: schema i's CSR token ids, contiguous, in CSR order. Throws only std::bad_alloc.
//  - BuildForcedTables: every schema's table, from SCM1 and TOK1; `present` stays false when the tokenizer's
//    tables cannot be read. Consults the table-build bad-alloc seam (bad_alloc_wrap.h) first. Throws only
//    std::bad_alloc; the caller maps it to its map's allocation status.
//  - ForcedChain: the forced chain from `state`: next_forced(w), then w := the state that token leads to,
//    until an unforced state. Writes the first min(cap, length) ids and returns the full length; bounded by the
//    state count. A token with no transition at w (only a corrupted table holds one) is part of the chain and
//    ends it.
namespace {

[[maybe_unused]] bool IsAcceptingState(const SchemaEntry& entry, uint32_t state) {
	using schema_masks_detail::ReadLE32;
	uint32_t lo = 0, hi = entry.accepting_count;
	while (lo < hi) {
		const uint32_t mid = lo + (hi - lo) / 2;
		const uint32_t v = ReadLE32(entry.accepting_le + static_cast<size_t>(mid) * 4);
		if (v == state) return true;
		if (v < state) lo = mid + 1; else hi = mid;
	}
	return false;
}

[[maybe_unused]] void BuildLegalIds(const SchemaMasksTable& schemas, std::vector<std::vector<uint32_t>>* out) {
	using schema_masks_detail::ReadLE32;
	out->resize(schemas.Count());
	for (size_t i = 0; i < schemas.Count(); ++i) {
		const SchemaEntry* e = schemas.ByIndex(i);
		std::vector<uint32_t>& ids = (*out)[i];
		ids.resize(e->transition_count);
		for (uint32_t t = 0; t < e->transition_count; ++t) {
			ids[t] = ReadLE32(e->transitions_le + static_cast<size_t>(t) * 8);
		}
	}
}

// A token's bytes in TOK1; false for an id with none (at or past TOK1's vocab count, or empty).
[[maybe_unused]] bool TokenBytes(const internal::TokenizerTables& tt, uint32_t id, const uint8_t** p, size_t* len) {
	using schema_masks_detail::ReadLE32;
	if (id >= tt.vocab_count) return false;
	const uint32_t b = ReadLE32(tt.vocab_offsets_le + static_cast<size_t>(id) * 4);
	const uint32_t e = ReadLE32(tt.vocab_offsets_le + static_cast<size_t>(id + 1) * 4);
	if (e <= b) return false;
	*p = tt.vocab_blob + b;
	*len = e - b;
	return true;
}

// §5.1, slice J1: the forced-next table of one schema, from its token DFA and TOK1 alone.
// - Forced test, per state: non-accepting; a non-empty row whose every token has bytes and is not a
//   special id; one first byte b shared by the whole row; the single-byte token of b legal there.
//   step[s] = b and succ[s] = the state that byte token leads to.
// - Forced cycles: a state whose step chain returns to itself is on a cycle; every such state is
//   unforced (step cleared) before F is computed, so a state that leads into a cycle keeps the text
//   up to the cycle and every chain ends within the state count.
// - F(s) = step[s] + F(succ[s]); next_forced(s) = Encode(F)[0] when that token is legal at s and a
//   byte prefix of F, otherwise the longest legal token that is a byte prefix of F (the byte token
//   always qualifies). Encode works on F's bytes as they are (a non-UTF-8 F encodes U+FFFD first,
//   which is never a byte prefix of F, so the fallback carries that state).
// Throws only std::bad_alloc.
[[maybe_unused]] void BuildForcedTableForSchema(const SchemaMasksTable& schemas, const TokenizerView& tokenizer,
                               const SchemaEntry& e,
                               const internal::TokenizerTables& tt, const std::vector<uint8_t>& is_special,
                               std::vector<ForcedEntry>* out) {
	using schema_masks_detail::ReadLE32;
	const uint32_t n = e.state_count;
	out->assign(n, ForcedEntry{});
	std::vector<int16_t> step(n, -1);
	std::vector<uint32_t> succ(n, 0);
	for (uint32_t s = 0; s < n; ++s) {
		const uint32_t rb = ReadLE32(e.state_offsets_le + static_cast<size_t>(s) * 4);
		const uint32_t re = ReadLE32(e.state_offsets_le + static_cast<size_t>(s + 1) * 4);
		if (rb == re || IsAcceptingState(e, s)) continue;
		int first = -1;
		bool ok = true;
		for (uint32_t t = rb; t < re && ok; ++t) {
			const uint32_t id = ReadLE32(e.transitions_le + static_cast<size_t>(t) * 8);
			const uint8_t* p = nullptr;
			size_t len = 0;
			if (id < is_special.size() && is_special[id]) ok = false;
			else if (!TokenBytes(tt, id, &p, &len)) ok = false;
			else if (first < 0) first = p[0];
			else if (first != p[0]) ok = false;
		}
		if (!ok || first < 0) continue;
		uint32_t nx = 0;
		if (!schemas.Transition(e, s, tt.byte_to_id[first], &nx)) continue;
		step[s] = static_cast<int16_t>(first);
		succ[s] = nx;
	}
	// Forced cycles (the step graph is functional): walk each unvisited chain, marking the states on
	// the current walk; reaching one of them again closes a cycle from it to the walk's end.
	{
		std::vector<uint8_t> color(n, 0);  // 0 unvisited, 1 on the current walk, 2 done
		std::vector<uint32_t> pos(n, 0);
		std::vector<uint32_t> walk;
		for (uint32_t s0 = 0; s0 < n; ++s0) {
			if (color[s0] != 0 || step[s0] < 0) continue;
			walk.clear();
			uint32_t w = s0;
			while (step[w] >= 0 && color[w] == 0) {
				color[w] = 1;
				pos[w] = static_cast<uint32_t>(walk.size());
				walk.push_back(w);
				w = succ[w];
			}
			if (step[w] >= 0 && color[w] == 1) {
				for (size_t k = pos[w]; k < walk.size(); ++k) step[walk[k]] = -1;
			}
			for (uint32_t v : walk) color[v] = 2;
		}
	}
	// |F| per state, memoized along each chain (no cycles remain).
	std::vector<uint32_t> flen(n, UINT32_MAX);
	{
		std::vector<uint32_t> walk;
		for (uint32_t s0 = 0; s0 < n; ++s0) {
			walk.clear();
			uint32_t w = s0;
			while (flen[w] == UINT32_MAX && step[w] >= 0) {
				walk.push_back(w);
				w = succ[w];
			}
			uint32_t len = flen[w] == UINT32_MAX ? 0u : flen[w];
			if (step[w] < 0) flen[w] = 0;
			for (size_t k = walk.size(); k-- > 0;) flen[walk[k]] = ++len;
		}
	}
	std::string f;
	for (uint32_t s = 0; s < n; ++s) {
		if (step[s] < 0) continue;
		f.clear();
		for (uint32_t w = s; step[w] >= 0; w = succ[w]) f.push_back(static_cast<char>(step[w]));
		const std::vector<int32_t> ids = tokenizer.Encode(std::string_view(f));
		int32_t chosen = -1;
		const uint8_t* p = nullptr;
		size_t len = 0;
		if (!ids.empty() && ids[0] >= 0 && schemas.Transition(e, s, static_cast<uint32_t>(ids[0]), nullptr) &&
		    TokenBytes(tt, static_cast<uint32_t>(ids[0]), &p, &len) && len <= f.size() &&
		    std::memcmp(p, f.data(), len) == 0) {
			chosen = ids[0];
		} else {
			const uint32_t rb = ReadLE32(e.state_offsets_le + static_cast<size_t>(s) * 4);
			const uint32_t re = ReadLE32(e.state_offsets_le + static_cast<size_t>(s + 1) * 4);
			size_t best = 0;
			for (uint32_t t = rb; t < re; ++t) {
				const uint32_t id = ReadLE32(e.transitions_le + static_cast<size_t>(t) * 8);
				if (TokenBytes(tt, id, &p, &len) && len > best && len <= f.size() &&
				    std::memcmp(p, f.data(), len) == 0) {
					best = len;
					chosen = static_cast<int32_t>(id);
				}
			}
		}
		(*out)[s].next_forced = chosen;
		(*out)[s].f_len = flen[s];
	}
}

// §5.1: every schema's table. An allocation failure propagates as std::bad_alloc, so a model never silently
// lacks the table: the CPU map maps it to SSLM_ALLOCATION_FAILED, the GPU map to SSLM_GPU_ALLOCATION_FAILED.
[[maybe_unused]] void BuildForcedTables(const SchemaMasksTable& schemas, const TokenizerView& tokenizer, ForcedTables* out) {
	internal::MaybeThrowInjectedForcedTableBuildAllocFault();
	internal::TokenizerTables tt;
	if (!internal::ReadTokenizerTables(tokenizer, &tt)) return;
	std::vector<uint8_t> is_special(tt.vocab_count, 0);
	for (const auto& sp : *tt.specials) {
		if (sp.second >= 0 && static_cast<uint32_t>(sp.second) < tt.vocab_count) {
			is_special[static_cast<size_t>(sp.second)] = 1;
		}
	}
	out->draftable.assign(tt.vocab_count, 0);
	for (uint32_t id = 0; id < tt.vocab_count; ++id) {
		const uint8_t* p = nullptr;
		size_t len = 0;
		out->draftable[id] = !is_special[id] && TokenBytes(tt, id, &p, &len) ? 1 : 0;
	}
	out->table.resize(schemas.Count());
	for (size_t i = 0; i < schemas.Count(); ++i) {
		BuildForcedTableForSchema(schemas, tokenizer, *schemas.ByIndex(i), tt, is_special, &out->table[i]);
	}
	out->present = true;
}

[[maybe_unused]] int32_t ForcedChain(const SchemaMasksTable& schemas, const std::vector<ForcedEntry>& table, uint32_t schema_index,
                    uint32_t state, int32_t* out, int32_t cap) {
	const SchemaEntry* e = schemas.ByIndex(schema_index);
	int32_t count = 0;
	uint32_t w = state;
	for (size_t guard = 0; guard < table.size() && w < table.size(); ++guard) {
		const int32_t tok = table[w].next_forced;
		if (tok < 0) break;
		if (count < cap) out[count] = tok;
		++count;
		uint32_t nx = 0;
		if (!schemas.Transition(*e, w, static_cast<uint32_t>(tok), &nx)) break;
		w = nx;
	}
	return count;
}

}  // namespace

}  // namespace forced_table
}  // namespace superslm

#endif  // SUPERSLM_SRC_FORCED_TABLE_H

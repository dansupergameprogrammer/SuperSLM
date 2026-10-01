// Paged-KV plan (rev 16.1) step C1: helpers the legacy-holder cells of dimensions 1-4 share
// (c4_dim1_lifetime.cpp, c4_dim2_hostile.cpp, c4_dim3_concurrency.cpp, c4_dim4_shape.cpp).
//
//   - page reads through the C4 seams: a whole page through the page-index peek, a holder's table
//     through the table-entry read, and the K/V rows [0, L) gathered from the pages by §3.1's
//     address formula written here (never by calling the engine), in pkv_scenarios.h's
//     position-major order so a digest compares with a reference record's rows_sha;
//   - a B = 1 twin of a fixture (§3.1, G36): the same artifact with CFG1's kv_block_size patched
//     to 1 and the integrity hash recomputed, so every position is its own page;
//   - the pre-1.9.0 blob shapes (SSB3, SSB2) cut from a pinned v1.8.1 SSB4 blob, the way
//     tests/t2138-abi-red-suite/dim9_persistence_red.cpp builds them from a live blob.
//
// Every helper that calls a seam is inline and emitted only where a cell calls it (pkv_common.h's
// convention), so each executable links against exactly the symbols its own cells use.

#ifndef SUPERSLM_TESTS_PKV_LEGACY_A_HELPERS_H
#define SUPERSLM_TESTS_PKV_LEGACY_A_HELPERS_H

#include "pkv_common.h"

#include "superslm/artifact.h"
#include "superslm/sha256.h"

#include <algorithm>
#include <set>

namespace pkv {
namespace legacy_a {

// The value an out-handle is preset to before a verb whose refusal a cell checks by a null out:
// non-null, so a verb that never writes *out leaves it standing and the check fails. Never released
// or dereferenced -- release a handle only when it is neither null nor this.
template <class H>
inline H OutSentinel() {
	return reinterpret_cast<H>(uintptr_t{0x1});
}

inline bool AllBytesAre(const std::vector<uint8_t>& b, uint8_t v) {
	for (uint8_t x : b)
		if (x != v) return false;
	return !b.empty();
}

// Page `page` of `pool`, whole, through the page-index peek (C4 seam). Empty on a refusal.
inline std::vector<uint8_t> PeekPage(sslm_kv_pool pool, uint32_t page, size_t page_bytes) {
	std::vector<uint8_t> out(page_bytes, 0);
	if (sslm_pkv_test_only_peek_page_bytes(pool, page, out.data(), out.size()) != SSLM_OK) out.clear();
	return out;
}

// A sequence's mapped table entries [0, mapped), through the table-entry read (C4 seam). False
// when the seam refuses an entry below the `mapped` it reported.
inline bool SeqTable(sslm_seq s, std::vector<uint32_t>* table) {
	table->clear();
	uint32_t page = 0, mapped = 0;
	if (sslm_pkv_test_only_seq_table_entry(s, 0, &page, &mapped) != SSLM_OK) return false;
	for (uint32_t i = 0; i < mapped; ++i) {
		if (sslm_pkv_test_only_seq_table_entry(s, i, &page, nullptr) != SSLM_OK) return false;
		table->push_back(page);
	}
	return true;
}

// The K/V rows of positions [0, L) of a live sequence, read from the pool's raw page bytes and
// addressed by §3.1's formula written here: for position p, entry table[p / B], and within the
// page ((l * 2 + half) * H_kv + h) * B * D + (p % B) * D. `B` is the test's own §3.1 value for the
// model, passed in (a B = 1 twin's differs from the fixture's). False when a seam refuses or the
// table does not cover L.
inline bool RowsFromPages(sslm_kv_pool pool, sslm_seq s, const Geometry& g, int64_t B, int64_t L,
                          std::vector<uint8_t>* rows) {
	std::vector<uint32_t> table;
	if (!SeqTable(s, &table)) return false;
	if (static_cast<int64_t>(table.size()) * B < L) return false;
	const size_t D = g.head_dim;
	const size_t page_bytes = size_t{g.layers} * 2u * g.kv_heads * static_cast<size_t>(B) * D;
	std::map<uint32_t, std::vector<uint8_t>> pages;
	rows->clear();
	for (int64_t p = 0; p < L; ++p) {
		const uint32_t page = table[static_cast<size_t>(p / B)];
		auto it = pages.find(page);
		if (it == pages.end()) {
			it = pages.emplace(page, PeekPage(pool, page, page_bytes)).first;
			if (it->second.empty()) return false;
		}
		for (uint32_t l = 0; l < g.layers; ++l)
			for (uint32_t half = 0; half < 2; ++half)
				for (uint32_t h = 0; h < g.kv_heads; ++h) {
					const size_t at = ((size_t{l} * 2 + half) * g.kv_heads + h) * static_cast<size_t>(B) * D +
					                  static_cast<size_t>(p % B) * D;
					rows->insert(rows->end(), it->second.begin() + static_cast<std::ptrdiff_t>(at),
					             it->second.begin() + static_cast<std::ptrdiff_t>(at + D));
				}
	}
	return true;
}

// A copy of `fx`'s artifact with CFG1's kv_block_size (config + 48, model.cpp) set to `kv_block_size`
// and the whole-file integrity hash recomputed (artifact.cpp: SHA-256 of the file with the hash
// bytes zeroed). The weights, cap and calibration are unchanged, so tokens and K/V rows equal the
// fixture's own; only the model identity (and so the blobs' model hash) differs. Mapped into `out`.
inline bool PatchedBlockSizeTwin(const Fixture& fx, uint32_t kv_block_size, Fixture* out) {
	if (!fx.ok) return false;
	std::vector<uint8_t> b(fx.bytes.p, fx.bytes.p + fx.size);
	uint32_t count = 0;
	std::memcpy(&count, b.data() + 12, 4);
	bool patched = false;
	for (uint32_t i = 0; i < count && !patched; ++i) {
		const uint8_t* row = b.data() + superslm::kHeaderBytes + size_t{i} * superslm::kSectionDescBytes;
		uint32_t type = 0;
		uint64_t offset = 0;
		std::memcpy(&type, row, 4);
		std::memcpy(&offset, row + 8, 8);
		if (type != static_cast<uint32_t>(superslm::SslmSectionType::Config) || offset + 52 > b.size()) continue;
		std::memcpy(b.data() + offset + 48, &kv_block_size, 4);
		patched = true;
	}
	if (!patched) return false;
	std::memset(b.data() + superslm::kIntegrityHashOffset, 0, superslm::kIntegrityHashBytes);
	uint8_t digest[32];
	superslm::Sha256Hash(b.data(), b.size(), digest);
	std::memcpy(b.data() + superslm::kIntegrityHashOffset, digest, 32);
	out->stem = fx.stem + "_kvbs" + std::to_string(kv_block_size);
	out->geo = fx.geo;
	out->size = b.size();
	out->bytes.Reset(b.size());
	std::memcpy(out->bytes.p, b.data(), b.size());
	out->sha = Sha(b.data(), b.size());
	out->ok = sslm_model_map(out->bytes.p, out->size, &out->model) == SSLM_OK;
	return out->ok;
}

// The pre-1.9.0 shapes, cut from a real SSB4 blob (the v1.8.1 pins) as dim9_persistence_red.cpp
// cuts them from a live one. SSB4: a 124-byte fixed header ([4, 108) shared with SSB2, the anti-LM
// order and history count at 108/112 shared with SSB3, ready_for_logits at 120), the residual
// (hidden_size bytes, always written by SSB4), the anti-LM history, kv_block_count and the block.
// SSB3 = magic + [4, 120) + residual only when mid-token + history + tail; SSB2 = magic + [4, 108)
// + residual only when mid-token + tail, and carries no history (so the source must have none).
// Empty when the source is not an SSB4 blob of this model's block size, or the SSB2 source has a
// history.
constexpr size_t kSsb4Header = 124;

inline std::vector<uint8_t> LegacyShapeFromSsb4(const std::vector<uint8_t>& ssb4, const Fixture& fx, char version) {
	std::vector<uint8_t> out;
	const size_t block = sslm_kv_block_size(fx.model);
	const size_t hidden = HiddenSize(fx);
	if (!IsMagic(ssb4, "SSB4") || ssb4.size() < kSsb4Header + hidden + 4 + block) return out;  // size before subtraction
	const uint64_t history = Le64(ssb4, 112);
	const bool mid_token = Le32(ssb4, 68) != 0;
	const size_t history_at = kSsb4Header + hidden;
	const size_t tail_at = ssb4.size() - 4 - block;  // kv_block_count + the block
	if (tail_at != history_at + 4 * history) return out;
	if (version == '2' && history != 0) return out;
	const size_t shared_end = version == '3' ? 120 : 108;
	out = {'S', 'S', 'B', static_cast<uint8_t>(version)};
	out.insert(out.end(), ssb4.begin() + 4, ssb4.begin() + static_cast<std::ptrdiff_t>(shared_end));
	if (mid_token)
		out.insert(out.end(), ssb4.begin() + kSsb4Header, ssb4.begin() + static_cast<std::ptrdiff_t>(history_at));
	if (version == '3')
		out.insert(out.end(), ssb4.begin() + static_cast<std::ptrdiff_t>(history_at),
		           ssb4.begin() + static_cast<std::ptrdiff_t>(tail_at));
	out.insert(out.end(), ssb4.begin() + static_cast<std::ptrdiff_t>(tail_at), ssb4.end());
	return out;
}

// `n` greedy tokens, one layer per call (pkv_common.h's NextToken); -1 entries mark a refusal.
inline std::vector<int32_t> NextTokens(sslm_model model, sslm_seq s, int n) {
	std::vector<int32_t> t;
	for (int i = 0; i < n; ++i) t.push_back(NextToken(model, s));
	return t;
}

}  // namespace legacy_a
}  // namespace pkv

#endif  // SUPERSLM_TESTS_PKV_LEGACY_A_HELPERS_H

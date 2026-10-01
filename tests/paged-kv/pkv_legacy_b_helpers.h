// Paged-KV plan (rev 16.1) step C1: helpers the legacy-holder cells of dimensions 6, 7, 9 and 11
// share (c4_dim6_determinism.cpp, c4_dim7_contract.cpp, c4_dim9_persistence.cpp,
// c4_dim11_guards.cpp).
//
// Everything here calls shipped verbs or the C4 seams only, so a C4 file that includes it links
// once C4's builder has defined the seams. The arithmetic (pages per cap, page bytes, the §3.1
// address) is pkv_common.h's, written in the test; nothing here reads the code under test's own
// stats.

#ifndef SUPERSLM_TESTS_PKV_LEGACY_B_HELPERS_H
#define SUPERSLM_TESTS_PKV_LEGACY_B_HELPERS_H

#include "pkv_common.h"

#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

namespace pkv {
namespace legacy_b {

// The decode params every cell here uses unless it says otherwise: greedy, one layer per call, so
// a single call can leave a sequence mid-token (layer_index > 0).
inline sslm_decode_params OneLayerGreedy() {
	sslm_decode_params p{};
	p.layer_budget = 1;
	return p;
}

// One decode_step call of one layer. Returns the status; *tok receives the emitted token or -1
// when the call ended mid-token.
inline sslm_status StepOneLayer(sslm_model model, sslm_seq s, int32_t* tok) {
	sslm_decode_params p = OneLayerGreedy();
	sslm_seq b[1] = {s};
	*tok = -1;
	return sslm_decode_step(model, b, 1, &p, nullptr, tok);
}

// Leaves `s` mid-token: emits the ready token if there is one, then runs exactly one layer of the
// next token, so the blob's layer_index is 1 and L' = context_length + 1. Returns false (with a
// failed check) if the state could not be reached.
inline bool EnterMidToken(sslm_model model, sslm_seq s, std::vector<int32_t>* emitted = nullptr) {
	int32_t tok = -1;
	sslm_status st = StepOneLayer(model, s, &tok);
	if (st == SSLM_OK && tok >= 0) {
		if (emitted) emitted->push_back(tok);
		st = StepOneLayer(model, s, &tok);
	}
	PKV_CHECK_MSG(st == SSLM_OK && tok < 0, "entering mid-token: status %d, token %d", static_cast<int>(st), tok);
	return st == SSLM_OK && tok < 0;
}

// The "next turn": tokens emitted until the first refusal, which is returned in *refusal. A
// legacy holder's limit is the cap (§3.4), so a holder resting ready at context_length c emits
// exactly 1 + (cap - c) tokens and then SSLM_CONTEXT_CAP_EXCEEDED. `max_tokens` bounds the loop.
inline std::vector<int32_t> RunToRefusal(sslm_model model, sslm_seq s, sslm_status* refusal, int64_t max_tokens) {
	std::vector<int32_t> out;
	sslm_status st = SSLM_OK;
	while (static_cast<int64_t>(out.size()) <= max_tokens) {
		const int32_t t = NextToken(model, s, &st);
		if (st != SSLM_OK || t < 0) break;
		out.push_back(t);
	}
	if (refusal) *refusal = st;
	return out;
}

inline std::vector<int32_t> NextTokens(sslm_model model, sslm_seq s, int n) {
	std::vector<int32_t> out;
	for (int i = 0; i < n; ++i) out.push_back(NextToken(model, s));
	return out;
}

inline std::string BlobSha(const std::vector<uint8_t>& b) { return Sha(b.data(), b.size()); }

inline sslm_status Restore(const Fixture& fx, sslm_kv_pool* pool, const std::vector<uint8_t>& blob, sslm_seq* out) {
	*out = nullptr;
	return sslm_seq_restore(fx.model, pool, blob.data(), blob.size(), out);
}

// A hand-built SSB6 whole_reserve blob carrying `Lp` canonical rows (normally L' of the source;
// a guard cell passes another count to make kv_positions disagree with the header while the size
// still matches it). Rows past the source's cap are impossible; rows the source never wrote come
// from its block, which 1.9.0's SSB5 zero-fills past L'. Built from the SSB5 bytes alone.
inline std::vector<uint8_t> Ssb6FromSsb5(const std::vector<uint8_t>& ssb5, const Fixture& fx, int64_t origin,
                                         int64_t Lp) {
	const size_t hidden = HiddenSize(fx);
	const BlobView v = ParseBlobWithHidden(ssb5, hidden);
	std::vector<uint8_t> out;
	if (!v.ok || !IsMagic(ssb5, "SSB5") || Lp < 0 || Lp > fx.geo.context_cap) return out;
	out.assign(ssb5.begin(), ssb5.begin() + kSsb5Header);
	std::memcpy(out.data(), "SSB6", 4);
	out.resize(kSsb6Header);
	PutLe32(out, 156, 0);
	PutLe32(out, 160, static_cast<uint32_t>(fx.geo.context_cap));
	PutLe64(out, 164, static_cast<uint64_t>(origin));
	const size_t tail = hidden + 4 * static_cast<size_t>(v.history_count);
	out.insert(out.end(), ssb5.begin() + kSsb5Header, ssb5.begin() + kSsb5Header + static_cast<std::ptrdiff_t>(tail));
	out.resize(out.size() + 8);
	PutLe64(out, out.size() - 8, static_cast<uint64_t>(Lp));
	const uint8_t* block = ssb5.data() + v.kv_offset;
	const size_t D = fx.geo.head_dim, cap = static_cast<size_t>(fx.geo.context_cap);
	for (uint32_t l = 0; l < fx.geo.layers; ++l)
		for (uint32_t half = 0; half < 2; ++half)
			for (uint32_t h = 0; h < fx.geo.kv_heads; ++h) {
				const uint8_t* src = block + ((size_t{l} * 2 + half) * fx.geo.kv_heads + h) * cap * D;
				out.insert(out.end(), src, src + static_cast<size_t>(Lp) * D);
			}
	return out;
}

// L' of a blob, read from its fixed header (§3.7): context_length + (layer_index > 0 ? 1 : 0).
inline int64_t BlobLPrime(const std::vector<uint8_t>& b) {
	return static_cast<int64_t>(Le64(b, 60)) + (Le32(b, 68) > 0 ? 1 : 0);
}

// Where SSB6's kv_positions field sits in a blob built by Ssb6FromSsb5.
inline size_t Ssb6KvPositionsOffset(const std::vector<uint8_t>& b, const Fixture& fx) {
	return kSsb6Header + HiddenSize(fx) + 4 * static_cast<size_t>(Le64(b, 112));
}

// 1.9.0's SSB5 size (the R0 re-anchor note, read from v1.11.0's sslm_seq_save):
// 156 + residual + 4 * history + 4 + block_size, the whole block whatever L' is.
inline size_t Ssb5Size(const Fixture& fx, uint64_t history) {
	return kSsb5Header + HiddenSize(fx) + 4 * static_cast<size_t>(history) + 4 + sslm_kv_block_size(fx.model);
}

// The C4 page-count instrument at the two pool sizes of §8 (rev 6): `P` is a legacy pool of
// `blocks` blocks (blocks * ceil(cap/B) pages); `P - 1` is the same pool with one page held by a
// frozen one-token legacy prefix, the only C4 construction that holds a page count that is not a
// multiple of ceil(cap/B). Legacy freeze keeps exactly ceil(len/B) = 1 page (§3.4); 1.13's legacy
// half grades that return, so these two sizes rest on it.
struct SizedLegacyPool {
	std::unique_ptr<LegacyPool> pool;
	sslm_prefix eater = nullptr;
	bool ok = false;
	SizedLegacyPool(const Fixture& fx, uint32_t blocks, bool minus_one) {
		pool = std::make_unique<LegacyPool>(fx.model, blocks);
		ok = pool->status == SSLM_OK;
		if (ok && minus_one) {
			const int32_t tok = 1;
			int32_t consumed = 0;
			ok = sslm_prefix_begin(fx.model, &pool->pool, &eater) == SSLM_OK &&
			     sslm_prefix_prefill(fx.model, eater, &tok, 1, 1, SSLM_SPAN_PROMPT, nullptr, &consumed) == SSLM_OK &&
			     consumed == 1 && sslm_prefix_freeze(eater) == SSLM_OK;
		}
		PKV_CHECK_MSG(ok, "sized legacy pool (%u blocks%s) did not build", blocks, minus_one ? ", minus one page" : "");
	}
	sslm_kv_pool* Pool() { return &pool->pool; }
	~SizedLegacyPool() {
		if (eater) sslm_prefix_release(eater);
	}
};

// §3.1's address, written here: the byte offset of (layer, half, kv_head, pos % B) inside a page.
inline size_t InPageOffset(const Fixture& fx, uint32_t layer, uint32_t half, uint32_t h, int64_t pos) {
	const size_t B = static_cast<size_t>(fx.B()), D = fx.geo.head_dim;
	return ((size_t{layer} * 2 + half) * fx.geo.kv_heads + h) * B * D + static_cast<size_t>(pos % fx.B()) * D;
}

// Rows [0, L) of a live legacy sequence, read through the page-index peek and the table-entry
// seam and addressed by the §3.1 formula above, in pkv_scenarios.h's position-major order, so the
// digest compares with a reference record's rows_sha. Never calls the engine or the save path.
inline bool PeekRows(const Fixture& fx, sslm_kv_pool pool, sslm_seq s, int64_t L, std::vector<uint8_t>* rows) {
	rows->clear();
	const size_t page_bytes = fx.PageBytes();
	std::vector<uint8_t> page(page_bytes);
	int64_t loaded = -1;
	for (int64_t p = 0; p < L; ++p) {
		const int64_t idx = p / fx.B();
		if (idx != loaded) {
			uint32_t pg = 0, mapped = 0;
			if (sslm_pkv_test_only_seq_table_entry(s, static_cast<uint32_t>(idx), &pg, &mapped) != SSLM_OK) return false;
			if (sslm_pkv_test_only_peek_page_bytes(pool, pg, page.data(), page_bytes) != SSLM_OK) return false;
			loaded = idx;
		}
		for (uint32_t l = 0; l < fx.geo.layers; ++l)
			for (uint32_t half = 0; half < 2; ++half)
				for (uint32_t h = 0; h < fx.geo.kv_heads; ++h) {
					const uint8_t* r = page.data() + InPageOffset(fx, l, half, h, p);
					rows->insert(rows->end(), r, r + fx.geo.head_dim);
				}
	}
	return true;
}

}  // namespace legacy_b
}  // namespace pkv

#endif  // SUPERSLM_TESTS_PKV_LEGACY_B_HELPERS_H

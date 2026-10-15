// Paged-KV plan (rev 16.2), step G1, group g1b: the construction of cell 2.7's hostile population,
// shared by its [G4] part (g1b_g4_cells.cpp: statuses, mirror, SetSchema, fill probe) and its [G5]
// part (g1b_g5_cells.cpp: the sequence's and the prefix's re-saved 'SLM6' blobs).
//
// Each case: pool A (kPoolA pages) with the frozen prefix X (kPrefixLen = 1,000 tokens, mid-page) and
// a fresh budgeted sequence S (budget 64), plus the case's own handles; then the one refused call.
#ifndef SSLM_G1B_HOSTILE_PREFIX_H
#define SSLM_G1B_HOSTILE_PREFIX_H

#include "g1b_helpers.h"

namespace g1b {
namespace hostile {


constexpr int32_t kPrefixLen = 1000;  // mid-page (1000 % 16 = 8), so an adopt's tail copy would show
constexpr int32_t kSeqBudget = 64;    // R(64) = 5
constexpr uint32_t kPoolA = 200;
constexpr uint32_t kPoolB = 100;

struct HostileCase {
	const char* name;
	S want;
	enum Kind {
		kAdoptUnfrozen,
		kAdoptOtherPool,
		kPrefillAfterFreeze,
		kSetSchemaAfterContent,
		kBeginFromUnfrozen,
		kAdoptSchemaMismatch,
		kAdoptOtherModel,
		kAdoptOtherContext,
		kBeginFromOtherCtx
	} kind;
};
inline const HostileCase kHostile[] = {
    {"adopt of an unfrozen prefix", S::SSLM_GPU_PREFIX_REJECTED, HostileCase::kAdoptUnfrozen},
    {"adopt across pools", S::SSLM_GPU_PREFIX_REJECTED, HostileCase::kAdoptOtherPool},
    {"prefix prefill after freeze", S::SSLM_GPU_PREFIX_REJECTED, HostileCase::kPrefillAfterFreeze},
    {"set_schema after content", S::SSLM_GPU_PREFIX_REJECTED, HostileCase::kSetSchemaAfterContent},
    {"begin_from on an unfrozen parent", S::SSLM_GPU_PREFIX_REJECTED, HostileCase::kBeginFromUnfrozen},
    {"adopt with a schema mismatch", S::SSLM_GPU_PREFIX_SCHEMA_MISMATCH, HostileCase::kAdoptSchemaMismatch},
    {"adopt across models", S::SSLM_SEQUENCE_KV_BUFFER_MISMATCH, HostileCase::kAdoptOtherModel},
    {"adopt across contexts", S::SSLM_SEQUENCE_KV_BUFFER_MISMATCH, HostileCase::kAdoptOtherContext},
    {"begin_from with a ctx other than the parent's", S::SSLM_SEQUENCE_KV_BUFFER_MISMATCH, HostileCase::kBeginFromOtherCtx},
};

// The construction of one hostile case, before its refused call: pool A with the frozen mid-page
// prefix X and the fresh budgeted sequence S, plus the case's own handles. `probes` lists every pool
// the case involves with the free count §3.4 gives for it.
struct HostileScene {
	SslmGpuKvPoolHandle* pool_a = nullptr;
	SslmGpuPrefixHandle* x = nullptr;
	SslmGpuSequenceHandle* s = nullptr;
	SslmGpuPrefixHandle* other = nullptr;  // the case's foreign or unfrozen prefix, when it has one
	SslmGpuContext* other_ctx = nullptr;   // ... and the context, model and pool it was made on
	SslmGpuModelHandle* other_model = nullptr;
	SslmGpuKvPoolHandle* other_pool = nullptr;
	std::vector<ProbeTarget> probes;
	std::vector<uint32_t> free_pages;
	std::vector<uint32_t> pool_pages;
	bool ok = false;
};

// A frozen prefix of kPrefixLen tokens with real schema-content progress: bound to schema 0 before
// content, kPrefixLen - n prompt tokens, then n legal schema-content tokens (an independent walk of
// the artifact's own SCM1 bytes), so its walk state is past 0 (the CPU rule's "real progress").
inline SslmGpuPrefixHandle* SchemaProgressPrefix(Scene& sc, Env& e, SslmGpuKvPoolHandle* pool) {
	const Geometry& g = e.geo;
	SchemaWalk w;
	if (!w.Build(e.tiny, 0)) {
		G1B_CHECK_MSG(false, "2.7: the tiny fixture carries no schema (pkv_def has S8)");
		return nullptr;
	}
	uint32_t end_state = 0;
	const std::vector<int32_t> legal = w.LegalTokens(4, &end_state);
	G1B_CHECK_MSG(!legal.empty() && end_state != 0, "2.7: no legal schema-content walk past state 0");
	SslmGpuPrefixHandle* z = NewPrefix(sc, e.ctx1, e.m1, pool, kPrefixLen);
	if (!z || legal.empty()) return nullptr;
	G1B_CHECK_STATUS(sslm_gpu_prefix_set_schema(e.ctx1, z, 0), S::SSLM_OK, "2.7: schema bind on a fresh prefix");
	const std::vector<int32_t> prompt = Stream(77, kPrefixLen - static_cast<int32_t>(legal.size()), g.vocab);
	G1B_CHECK_STATUS(PrefixPrefillAll(g, e.ctx1, z, prompt, 0), S::SSLM_OK, "2.7: schema prefix, prompt span");
	G1B_CHECK_STATUS(PrefixPrefillAll(g, e.ctx1, z, legal, 1), S::SSLM_OK, "2.7: schema prefix, schema-content span");
	G1B_CHECK_STATUS(sslm_gpu_prefix_freeze(e.ctx1, z), S::SSLM_OK, "2.7: schema prefix freeze");
	return z;
}

inline HostileScene BuildHostile(Scene& sc, Env& e, const HostileCase& c) {
	const Geometry& g = e.geo;
	HostileScene h;
	h.pool_a = NewPool(sc, e.ctx1, e.m1, kPoolA);
	if (!h.pool_a) return h;
	const std::vector<int32_t> xt = Stream(51, kPrefixLen, g.vocab);
	h.x = FrozenPrefix(sc, g, e.ctx1, e.m1, h.pool_a, xt);
	h.s = NewBudgetSeq(sc, e.ctx1, e.m1, h.pool_a, kSeqBudget);
	if (!h.x || !h.s) return h;
	uint32_t free_a = kPoolA - g.Frozen(kPrefixLen) - g.R(kSeqBudget);
	auto add_probe = [&](SslmGpuContext* ctx, SslmGpuModelHandle* m, SslmGpuKvPoolHandle* p, uint32_t pages, uint32_t free) {
		h.probes.push_back({ctx, m, p});
		h.pool_pages.push_back(pages);
		h.free_pages.push_back(free);
	};
	switch (c.kind) {
		case HostileCase::kAdoptUnfrozen:
		case HostileCase::kSetSchemaAfterContent:
		case HostileCase::kBeginFromUnfrozen: {
			// An unfrozen prefix with content (20 tokens) on pool A: R(64) = 5 pages.
			h.other = NewPrefix(sc, e.ctx1, e.m1, h.pool_a, kSeqBudget);
			if (!h.other) return h;
			h.other_ctx = e.ctx1, h.other_model = e.m1, h.other_pool = h.pool_a;
			G1B_CHECK_STATUS(PrefixPrefillAll(g, e.ctx1, h.other, Stream(52, 20, g.vocab)), S::SSLM_OK, "2.7: unfrozen prefix content");
			free_a -= g.R(kSeqBudget);
			add_probe(e.ctx1, e.m1, h.pool_a, kPoolA, free_a);
			break;
		}
		case HostileCase::kAdoptOtherPool: {
			SslmGpuKvPoolHandle* pb = NewPool(sc, e.ctx1, e.m1, kPoolB);
			h.other = pb ? FrozenPrefix(sc, g, e.ctx1, e.m1, pb, xt) : nullptr;
			if (!h.other) return h;
			h.other_ctx = e.ctx1, h.other_model = e.m1, h.other_pool = pb;
			add_probe(e.ctx1, e.m1, h.pool_a, kPoolA, free_a);
			add_probe(e.ctx1, e.m1, pb, kPoolB, kPoolB - g.Frozen(kPrefixLen));
			break;
		}
		case HostileCase::kAdoptOtherModel: {
			SslmGpuKvPoolHandle* pm = NewPool(sc, e.ctx1, e.m1b, kPoolB);
			h.other = pm ? FrozenPrefix(sc, g, e.ctx1, e.m1b, pm, xt) : nullptr;
			if (!h.other) return h;
			h.other_ctx = e.ctx1, h.other_model = e.m1b, h.other_pool = pm;
			add_probe(e.ctx1, e.m1, h.pool_a, kPoolA, free_a);
			add_probe(e.ctx1, e.m1b, pm, kPoolB, kPoolB - g.Frozen(kPrefixLen));
			break;
		}
		case HostileCase::kAdoptOtherContext: {
			SslmGpuKvPoolHandle* pc = NewPool(sc, e.ctx2, e.m2, kPoolB);
			h.other = pc ? FrozenPrefix(sc, g, e.ctx2, e.m2, pc, xt) : nullptr;
			if (!h.other) return h;
			h.other_ctx = e.ctx2, h.other_model = e.m2, h.other_pool = pc;
			add_probe(e.ctx1, e.m1, h.pool_a, kPoolA, free_a);
			add_probe(e.ctx2, e.m2, pc, kPoolB, kPoolB - g.Frozen(kPrefixLen));
			break;
		}
		case HostileCase::kAdoptSchemaMismatch: {
			h.other = SchemaProgressPrefix(sc, e, h.pool_a);
			if (!h.other) return h;
			h.other_ctx = e.ctx1, h.other_model = e.m1, h.other_pool = h.pool_a;
			free_a -= g.Frozen(kPrefixLen);
			add_probe(e.ctx1, e.m1, h.pool_a, kPoolA, free_a);
			break;
		}
		case HostileCase::kPrefillAfterFreeze:
		case HostileCase::kBeginFromOtherCtx:
			add_probe(e.ctx1, e.m1, h.pool_a, kPoolA, free_a);
			break;
	}
	h.ok = true;
	return h;
}

// The refused call itself. A begin_from that is (wrongly) admitted is kept in the scene so it is
// released, and the probe then reads its draw.
inline S RunHostile(Scene& sc, Env& e, const HostileCase& c, HostileScene& h) {
	const Geometry& g = e.geo;
	SslmGpuPrefixHandle* out = reinterpret_cast<SslmGpuPrefixHandle*>(uintptr_t{0x1});
	S st = S::SSLM_OK;
	switch (c.kind) {
		case HostileCase::kAdoptUnfrozen:
		case HostileCase::kAdoptOtherPool:
		case HostileCase::kAdoptSchemaMismatch:
		case HostileCase::kAdoptOtherModel:
		case HostileCase::kAdoptOtherContext:
			return sslm_gpu_seq_adopt_prefix(e.ctx1, h.s, h.other);
		case HostileCase::kPrefillAfterFreeze: {
			const std::vector<int32_t> t = Stream(53, 8, g.vocab);
			int32_t consumed = -1;
			st = sslm_gpu_prefix_prefill(e.ctx1, h.x, t.data(), 8, 0, g.decode_budget * 9, &consumed);
			G1B_CHECK_MSG(st == S::SSLM_OK || consumed <= 0, "2.7 %s: refused but consumed %d", c.name, consumed);
			return st;
		}
		case HostileCase::kSetSchemaAfterContent:
			return sslm_gpu_prefix_set_schema(e.ctx1, h.other, 0);
		case HostileCase::kBeginFromUnfrozen:
			st = sslm_gpu_prefix_begin_from(e.ctx1, h.other, kSeqBudget, &out);
			break;
		case HostileCase::kBeginFromOtherCtx:
			st = sslm_gpu_prefix_begin_from(e.ctx2, h.x, kSeqBudget, &out);
			break;
	}
	G1B_CHECK_MSG(out == nullptr || st == S::SSLM_OK, "2.7 %s: begin_from refused but out-handle not nulled", c.name);
	if (st == S::SSLM_OK && out) sc.AddPrefix(c.kind == HostileCase::kBeginFromOtherCtx ? e.ctx2 : e.ctx1, out);
	return st;
}

}  // namespace hostile
}  // namespace g1b

#endif  // SSLM_G1B_HOSTILE_PREFIX_H

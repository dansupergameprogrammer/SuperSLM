// sslm_abi.cpp -- T-2139 (Brunel): C1 (sizing/construction) + C2 (model lifecycle) of the
// Layer-1 CPU-side sslm_* consumer C ABI, built to Claude/Vitruvius/
// t2133-layer1-c-abi-design-2026-08-16.md (Wizard repo). See include/superslm/sslm_abi.h's own
// header comment for the surface this file discharges and what remains undefined.
//
// SCOPE (StandardsDocument Sec5.5): this file defines exactly the ten verbs C1/C2 name --
// sslm_workspace_size, sslm_kv_block_size, sslm_kv_pool_overhead_size, sslm_seq_state_size,
// sslm_workspace_create, sslm_workspace_destroy, sslm_kv_pool_create, sslm_kv_pool_destroy,
// sslm_model_map, sslm_model_unmap -- and nothing past them. C3-C7's verbs are declared in the
// header (for Gate A's own whole-header coverage) but not defined here.
//
// PROVISIONAL SIZING FORMULAS, stated plainly rather than presented as final (Claude/Brunel/
// t2139-abi-build-2026-08-16.md Sec4, the buffer-mapping ruling, design commit fab235c1c6):
// sslm_workspace_size's exact byte count and sslm_workspace_create's alignment requirement are
// this design's own "opaque to the caller, self-describing internally" layout (design Sec7.1) --
// no consumer of the ACTUAL bytes exists until C4 wires this buffer into the real per-sequence
// call. The formulas below are a real, overflow-checked, monotonic sizing function of `config`/
// `model` that satisfies every C1 gate obligation (hostile-config sentinel/rejection parity
// between sslm_workspace_size and sslm_workspace_create, buffer-too-small/misaligned detection,
// construct/destroy symmetry) -- they are NOT yet grounded in a real per-call scratch layout,
// because C4 has not been built. Revisited, not re-derived from scratch, once C4 lands: the
// DOMAIN CHECK (which configs are hostile) is the design's own Sec7.1 statement and does not
// change; only the BYTE COUNT a valid config maps to may grow or shrink, which is exactly the
// "the caller only ever sees sslm_workspace_size's return value grow or shrink" contract design
// Sec7.1 states this internal layout is free to do.
#include "superslm/sslm_abi.h"

#include <atomic>
#include <cstdint>
#include <cstring>
#include <new>
#include <string>

#include "superslm/model.h"

// -----------------------------------------------------------------------------------------
// Opaque handle bodies (design Sec8: sslm_model_s/sslm_kv_pool_s/sslm_workspace_s are declared
// as incomplete struct tags in the public header; defined here, matching Sec5's own
// "opaque handles passed by value" convention -- gpu_1p0.cpp's own SslmGpuContext/
// SslmGpuModelHandle definitions are the house precedent this mirrors). Defined before the
// anonymous-namespace helpers below, several of which (ConfigDomainOk) dereference
// sslm_model_s and therefore need it complete, not merely declared.
// -----------------------------------------------------------------------------------------

// C2. Owns the loaded model view. `live_refs` is the D-SLM32 lifecycle-guard bookkeeping design
// Sec9's own C2 gate names ("unmap-while-live is rejected") -- nothing built in C1/C2 ever
// increments it (no sslm_seq/sslm_prefix/sslm_adapter exists yet), so the guard cannot yet be
// exercised end-to-end; stated here rather than left implicit, matching design Sec10 dim11's own
// "shown able to fire" standard, which this arc does not yet meet for this guard. C3 (sequences/
// prefixes) and C6 (adapters) are the callers that will increment/decrement it.
struct sslm_model_s {
	superslm::SslmModelView view;
	std::atomic<uint32_t> live_refs{0};
};

// C1. A caller-owned compute-scratch region, wrapped for construct/destroy symmetry and
// buffer-size/alignment validation. `buf`/`buf_size` are the caller's own memory (never freed by
// sslm_workspace_destroy, design Sec7.1) -- this handle's own heap allocation is bookkeeping
// only, matching sslm_model_s's own "the handle is heap-owned, the artifact bytes/caller buffer
// are not" split.
struct sslm_workspace_s {
	void* buf = nullptr;
	size_t buf_size = 0;
};

// C1. A caller-owned KV region, sized for `block_count` blocks (design Sec7.2). `live_refs` is
// the SSLM_POOL_HAS_LIVE_HANDLES guard's own bookkeeping (design Sec6) -- like
// sslm_model_s::live_refs, unexercised until C3 (sslm_seq_create/sslm_prefix_begin) exists to
// increment it; C1's own obligation is that the field exists and destroy checks it, not that a
// real caller can yet drive it nonzero.
struct sslm_kv_pool_s {
	void* buf = nullptr;
	size_t buf_size = 0;
	uint32_t block_count = 0;
	std::atomic<uint32_t> live_refs{0};
};

namespace {

// -----------------------------------------------------------------------------------------
// Shared arithmetic: overflow-safe (checked/saturating) size_t compose, per design Sec10 dim2's
// own "asserts either SSLM_INVALID_ARGUMENT before the multiplication is trusted, or a proven
// overflow-checked (saturating/checked) computation whose reported size is never smaller than
// the true requirement at the boundary" -- applied uniformly to every sizing formula below, not
// only the one cell dim2 names by name. No native 128-bit integer on this toolchain
// (forward_sites.h's own established note); the standard "a != 0 && b > MAX/a" technique is used
// throughout instead.
// -----------------------------------------------------------------------------------------

constexpr size_t kSizeMax = static_cast<size_t>(-1);

// true on success (*out is the exact product); false and *out left at kSizeMax (saturated) on
// overflow.
bool CheckedMulSizeT(size_t a, size_t b, size_t* out) {
	if (a != 0 && b > kSizeMax / a) {
		*out = kSizeMax;
		return false;
	}
	*out = a * b;
	return true;
}

bool CheckedAddSizeT(size_t a, size_t b, size_t* out) {
	if (a > kSizeMax - b) {
		*out = kSizeMax;
		return false;
	}
	*out = a + b;
	return true;
}

// Saturating compose of an arbitrary sequence of (multiply, then add) steps -- every sizing
// function below is a sum of a small number of products, and a single "did anything overflow"
// flag threaded through is simpler and no less exact than re-deriving the check inline at each
// call site.
struct SaturatingAccumulator {
	size_t value = 0;
	bool overflowed = false;

	void AddProduct(size_t a, size_t b) {
		size_t product = 0;
		if (!CheckedMulSizeT(a, b, &product)) overflowed = true;
		size_t sum = 0;
		if (!CheckedAddSizeT(value, product, &sum)) overflowed = true;
		value = overflowed ? kSizeMax : sum;
	}
};

// -----------------------------------------------------------------------------------------
// sslm_config domain (design Sec7.1, verbatim): max_batch >= 1, max_chunk_budget >= 1,
// 1 <= max_layer_budget <= num_hidden_layers, reserved == 0. A null model or null config is
// ALSO treated as hostile here (the design's own hostile-config cell is scoped per-field on a
// non-null config against a real model; a null pointer is the same "caller's call shape is
// wrong" class Sec6's SSLM_INVALID_ARGUMENT family already names for every other verb).
// -----------------------------------------------------------------------------------------

bool ConfigDomainOk(const sslm_model_s* model, const sslm_config* config) {
	if (model == nullptr || config == nullptr) return false;
	if (config->max_batch < 1) return false;
	if (config->max_chunk_budget < 1) return false;
	if (config->max_layer_budget < 1) return false;
	if (static_cast<uint32_t>(config->max_layer_budget) > model->view.config.num_hidden_layers) {
		return false;
	}
	if (config->reserved != 0) return false;
	return true;
}

// KV precision's element width in bytes -- Int8 = 1, Int16 = 2 (design Sec7.2/Sec7.3;
// SslmKvPrecision, model.h). Stated as its own function because every KV-sizing formula below
// reads it, and RunGreedyDecodeLoop's own KvPrecisionUnsupported rejection (forward_sites.h)
// means Int16 has no real decode path yet -- the SIZING formula is still well-defined for it
// (a caller sizing a pool for a model declaring Int16 gets a real, correct byte count; whether
// anything can decode against it is C4's own, already-documented, unchanged restriction).
size_t KvElementBytes(superslm::SslmKvPrecision p) {
	return p == superslm::SslmKvPrecision::Int16 ? 2 : 1;
}

}  // namespace

// -----------------------------------------------------------------------------------------
// C1 -- sizing (design Sec7): pure functions of an already-loaded sslm_model. Every function
// below returns 0 on a null/invalid model -- 0 is never a legitimate positive sizing answer for
// any real artifact (every dimension a formula below reads is load-time-rejected at 0 by
// SslmModel::Load, model.cpp's own CFG1 domain gate), so it is a safe, unambiguous sentinel
// alongside sslm_workspace_size's own hostile-config sentinel (design Sec7.1).
// -----------------------------------------------------------------------------------------

extern "C" size_t sslm_workspace_size(sslm_model model, const sslm_config* config) {
	if (!ConfigDomainOk(model, config)) return 0;

	// Provisional layout (this file's own header comment): per-call scratch sized from the
	// call-shape declaration alone, monotonic in every sslm_config field and in the model's own
	// geometry, saturating rather than wrapping on an adversarial (but domain-valid) combination.
	const superslm::SslmModelConfig& c = model->view.config;
	SaturatingAccumulator acc;
	// Attention-interior + matmul accumulator scratch: one wide (int64) row per token in the
	// widest single call this workspace must serve, sized to the widest of hidden_size/
	// intermediate_size/vocab_size (the three row widths any one funnel call in this tree
	// composes over, forward_sites.h) so a single conservative term covers all three rather than
	// summing three independently-justified guesses.
	uint64_t widest_row = c.hidden_size;
	if (c.intermediate_size > widest_row) widest_row = c.intermediate_size;
	if (c.vocab_size > widest_row) widest_row = c.vocab_size;
	size_t call_shape = 0;
	CheckedMulSizeT(static_cast<size_t>(config->max_batch),
	                 static_cast<size_t>(config->max_chunk_budget), &call_shape);
	acc.AddProduct(call_shape, static_cast<size_t>(widest_row));
	acc.AddProduct(acc.value, sizeof(int64_t));
	// Per-head softmax-row scratch for the widest layer_budget this workspace must serve.
	SaturatingAccumulator heads;
	heads.AddProduct(static_cast<size_t>(config->max_layer_budget),
	                  static_cast<size_t>(c.num_attention_heads));
	heads.AddProduct(heads.value, static_cast<size_t>(c.context_cap));
	heads.AddProduct(heads.value, sizeof(int32_t));
	size_t total = 0;
	CheckedAddSizeT(acc.value, heads.value, &total);
	// A fixed internal bookkeeping header (the version-pinned offset table design Sec7.1 names).
	size_t final_size = 0;
	CheckedAddSizeT(total, 64, &final_size);
	return final_size;
}

extern "C" size_t sslm_kv_block_size(sslm_model model) {
	if (!model) return 0;
	const superslm::SslmModelConfig& c = model->view.config;
	if (c.kv_block_size == 0) return 0;
	// One block's bytes = (tokens per block) * num_hidden_layers * num_key_value_heads *
	// head_dim * 2 (K and V halves) * per-element width (design Sec7.2/Sec7.3; the K/V store
	// layout this composes from is forward_sites.h's own KeyRow/ValueRow addressing,
	// per-(layer,head)-major, position-minor).
	SaturatingAccumulator acc;
	acc.AddProduct(static_cast<size_t>(c.kv_block_size), static_cast<size_t>(c.num_hidden_layers));
	acc.AddProduct(acc.value, static_cast<size_t>(c.num_key_value_heads));
	acc.AddProduct(acc.value, static_cast<size_t>(c.head_dim));
	acc.AddProduct(acc.value, 2);
	acc.AddProduct(acc.value, KvElementBytes(c.kv_precision));
	return acc.value;
}

extern "C" size_t sslm_kv_pool_overhead_size(sslm_model model, uint32_t block_count) {
	if (!model) return 0;
	// A free-list next-index slot (uint32_t) + a per-block refcount slot (uint32_t) per block,
	// plus a small fixed header -- design Sec7.2's own "a free-list and per-block refcount array,
	// sized O(block_count)". Saturates (never wraps to a too-small value) on an adversarial
	// block_count, per design Sec10 dim2's own overflow-safety cell -- this function has no
	// status channel, so saturation to SIZE_MAX is its only way to signal "this cannot be
	// satisfied by any real buffer" without under-reporting.
	SaturatingAccumulator acc;
	acc.AddProduct(static_cast<size_t>(block_count), sizeof(uint32_t) * 2);
	size_t total = 0;
	CheckedAddSizeT(acc.value, 32, &total);
	return acc.overflowed ? kSizeMax : total;
}

extern "C" size_t sslm_seq_state_size(sslm_model model) {
	if (!model) return 0;
	const superslm::SslmModelConfig& c = model->view.config;
	const size_t block_size = sslm_kv_block_size(model);
	if (block_size == 0) return 0;
	// Design Sec7.3's field list: fixed-size header fields, the residual (hidden_size *
	// activation_bytes, worst case -- a mid-token residual, layer_index != 0), then the whole
	// KV store a single sequence can carry at its own context_cap. Fixed fields:
	// magic(4) + model_hash(32) + kv_precision(4) + schema_name_hash(8) + dfa_walk_state(4) +
	// adapter_binding_id(8) + context_length(8) + layer_index(4) + hidden_scale(16, CarriedScale
	// as two int64) + kv_saturation_count(8) + kv_block_count(4) = 100 bytes.
	constexpr size_t kFixedHeaderBytes = 100;
	SaturatingAccumulator acc;
	acc.value = kFixedHeaderBytes;
	acc.AddProduct(1, static_cast<size_t>(c.hidden_size));  // residual_bytes, int8 codes
	// Worst-case KV block count for one sequence at its own context_cap: ceil(context_cap /
	// tokens_per_block).
	const uint32_t tokens_per_block = c.kv_block_size;
	uint64_t worst_blocks = (static_cast<uint64_t>(c.context_cap) + tokens_per_block - 1) /
	                         tokens_per_block;
	acc.AddProduct(static_cast<size_t>(worst_blocks), block_size);
	return acc.value;
}

// -----------------------------------------------------------------------------------------
// C1 -- construction over caller memory (design Sec7, NEW verbs). A fixed, documented alignment
// (64 bytes -- a common cache-line/SIMD width in this codebase's own kernels, matmul.h) stands
// in for "the artifact's declared alignment requirement" (design Sec7.1's own phrase): no CFG1
// field carries a distinct declared alignment (SslmModelConfig, model.h, grep-confirmed), so a
// fixed, generous, always-sufficient constant is what "the artifact's declared alignment
// requirement" resolves to until/unless a future artifact format revision adds a real per-model
// value -- stated here rather than silently assumed, per this file's own top-of-file disposition
// on provisional layout choices.
// -----------------------------------------------------------------------------------------

namespace {
constexpr size_t kAbiAlignmentBytes = 64;

bool IsAligned(const void* p) {
	return (reinterpret_cast<uintptr_t>(p) % kAbiAlignmentBytes) == 0;
}
}  // namespace

extern "C" sslm_status sslm_workspace_create(sslm_model model, const sslm_config* config,
                                              void* buf, size_t buf_size, sslm_workspace* out) {
	if (!out) return SSLM_INVALID_ARGUMENT;
	*out = nullptr;
	if (!ConfigDomainOk(model, config)) return SSLM_INVALID_ARGUMENT;
	if (!buf) return SSLM_INVALID_ARGUMENT;
	const size_t required = sslm_workspace_size(model, config);
	if (buf_size < required) return SSLM_BUFFER_TOO_SMALL;
	if (!IsAligned(buf)) return SSLM_MISALIGNED_BUFFER;
	sslm_workspace_s* h = new (std::nothrow) sslm_workspace_s();
	if (!h) throw std::bad_alloc();  // house convention: propagates unchanged, bad_alloc_wrap.h
	h->buf = buf;
	h->buf_size = buf_size;
	*out = h;
	return SSLM_OK;
}

extern "C" sslm_status sslm_workspace_destroy(sslm_workspace ws) {
	if (!ws) return SSLM_INVALID_ARGUMENT;
	// Never frees ws->buf (caller-owned, design Sec7.1) -- only the handle's own bookkeeping.
	delete ws;
	return SSLM_OK;
}

extern "C" sslm_status sslm_kv_pool_create(sslm_model model, void* buf, size_t buf_size,
                                            uint32_t block_count, sslm_kv_pool* out) {
	if (!out) return SSLM_INVALID_ARGUMENT;
	*out = nullptr;
	if (!model || !buf) return SSLM_INVALID_ARGUMENT;
	if (block_count == 0) return SSLM_INVALID_ARGUMENT;
	const size_t block_size = sslm_kv_block_size(model);
	if (block_size == 0) return SSLM_INVALID_ARGUMENT;
	const size_t overhead = sslm_kv_pool_overhead_size(model, block_count);
	if (overhead == kSizeMax) return SSLM_INVALID_ARGUMENT;  // saturated: no real buffer suffices
	size_t blocks_bytes = 0;
	if (!CheckedMulSizeT(static_cast<size_t>(block_count), block_size, &blocks_bytes)) {
		return SSLM_INVALID_ARGUMENT;  // overflow before the multiplication is trusted (dim2)
	}
	size_t required = 0;
	if (!CheckedAddSizeT(blocks_bytes, overhead, &required)) {
		return SSLM_INVALID_ARGUMENT;
	}
	if (buf_size < required) return SSLM_BUFFER_TOO_SMALL;
	sslm_kv_pool_s* h = new (std::nothrow) sslm_kv_pool_s();
	if (!h) throw std::bad_alloc();
	h->buf = buf;
	h->buf_size = buf_size;
	h->block_count = block_count;
	*out = h;
	return SSLM_OK;
}

extern "C" sslm_status sslm_kv_pool_destroy(sslm_kv_pool pool) {
	if (!pool) return SSLM_INVALID_ARGUMENT;
	if (pool->live_refs.load(std::memory_order_acquire) != 0) return SSLM_POOL_HAS_LIVE_HANDLES;
	delete pool;
	return SSLM_OK;
}

// -----------------------------------------------------------------------------------------
// C2 -- model lifecycle (design Sec8/Sec9 C2): thin wrappers over the already-shipped
// SslmModel::Load, plus the D-SLM32 lifecycle-symmetry guard.
// -----------------------------------------------------------------------------------------

extern "C" sslm_status sslm_model_map(const void* data, size_t size, sslm_model* out) {
	if (!out) return SSLM_INVALID_ARGUMENT;
	*out = nullptr;
	if (!data) return SSLM_INVALID_ARGUMENT;

	superslm::SslmModelView view;
	std::string err;
	// SslmModel::Load "throws only std::bad_alloc" (model.h's own contract, S-HARDEN-7) -- this
	// call site does not catch it, matching src/bad_alloc_wrap.h's own house convention: a
	// bad_alloc crosses this ABI boundary unchanged rather than being encoded into sslm_status,
	// which (design Sec6, 17 enumerators, closed set) carries no resource-exhaustion member.
	// `view` is a local, stack-owned RAII object, so no handle allocation happens before Load
	// returns -- there is nothing for a thrown bad_alloc to leak here.
	const superslm::SslmModelStatus st = superslm::SslmModel::Load(
	    static_cast<const uint8_t*>(data), size, view, &err);
	if (st != superslm::SslmModelStatus::Ok) {
		// design Sec6: every SslmModel::Load rejection maps to SSLM_ARTIFACT_REJECTED; the
		// specific SslmModelStatus (`err`/`st`) has no exposed channel on this signature
		// (sslm_g5.h's own 3-argument sslm_model_map shape, which this design's Sec7.4
		// derivation method requires this design match verbatim, carries no diagnostic
		// out-parameter) -- discarded here, not silently invented a channel for.
		return SSLM_ARTIFACT_REJECTED;
	}

	sslm_model_s* h = new (std::nothrow) sslm_model_s{std::move(view)};
	if (!h) throw std::bad_alloc();
	*out = h;
	return SSLM_OK;
}

extern "C" sslm_status sslm_model_unmap(sslm_model model) {
	if (!model) return SSLM_INVALID_ARGUMENT;
	if (model->live_refs.load(std::memory_order_acquire) != 0) {
		return SSLM_MODEL_HAS_LIVE_SEQUENCES;
	}
	delete model;
	return SSLM_OK;
}

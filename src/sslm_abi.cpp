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
// SIZING FORMULAS (Claude/Brunel/t2139-abi-build-2026-08-16.md Sec4, the buffer-mapping ruling,
// design commit fab235c1c6). sslm_workspace_size's exact byte count and sslm_workspace_create's
// alignment requirement are this design's own "opaque to the caller, self-describing internally"
// layout (design Sec7.1).
//
// S7, ROUND 2 (coordinator's closing-round brief, following Claude/Poirot/
// 2c18dab-t2139-abi-build-review.md's own S7 finding and Curie's real dim7 red pin,
// tests/t2138-abi-red-suite/dim7_contract_red.cpp TestDim7_C1, curie/t2138-abi-red-suite@f229449):
// EVERY allocation this file's own code makes on the sslm_prefill/sslm_prefix_prefill/
// sslm_decode_step call graph is now carved from a caller-supplied, correctly-sized workspace
// when one is provided -- sslm_decode_step's own per-token scratch (embed codes, final-norm
// codes, wide logits, the narrowed logit row) AND PrefillWholeTokens' own embed_codes (shared by
// sslm_prefill/sslm_prefix_prefill, the SAME workspace region decode_step uses, since one
// workspace handle is never read by two calls concurrently -- design Sec8.3/dim8 M1's own
// "successive, not concurrent" sharing contract). `layers_scratch` (both call sites) stays
// unthreaded -- it is only allocated at all when an adapter is bound (the common no-adapter case
// makes zero allocation there already, ResolveLayers' own early return), and a
// std::vector<superslm::LayerWeights> is not POD bytes this byte-oriented workspace layout can
// safely describe without a much larger, riskier placement-new scheme; disclosed, not fixed, this
// round.
//
// S7, ROUND 3 (arc-final closing round, design commit 959336ad64's own recalibration; curie/
// t2138-abi-red-suite@11e7182's own recalibrated dim7 C1a/C1b cells). The ruled contract narrowed
// to what THIS ABI layer controls (true engine-wide zero-allocation is D-SLM3457's own later,
// separate inventory) -- and one real ABI-layer gap remained inside that narrowed contract:
// sslm_decode_step's own ready_for_logits path (the one call shape that skips RunLayerLoop
// entirely -- no engine-internal-scratch exception applies there at all) measured 1 allocation
// against a ruled 0. Traced with a WITH-vs-WITHOUT-workspace differential (the technique C1a's own
// second half uses) to RmsNormSite's own internal `wide` scratch for the final_norm call this file
// makes directly. FIXED FOR REAL: RmsNormSite (forward_sites.h/.cpp) gained a trailing, defaulted
// `external_wide_scratch` parameter -- nullptr (every pre-existing call site, unchanged behavior)
// keeps the original internal allocation; this file's own final_norm call in sslm_decode_step now
// passes a `WorkspaceLayout::rms_wide` region carved from the caller's workspace, eliminating the
// allocation for real rather than reporting it as engine-internal-and-out-of-scope -- unlike
// RunLayerLoop's own much larger, deeply nested per-layer scratch (attention-interior temporaries,
// matmul accumulators, softmax rows -- dozens of std::vector locals per layer × num_hidden_layers
// × tokens, still explicitly out of scope per the design's own RunLayerLoop/RunGreedyDecodeLoop
// signature note: "separately, internally heap-allocated ... never a caller-supplied buffer of
// any kind"), RmsNormSite is a single, small, already-directly-called site this file owns the
// call to -- adding one optional parameter closes the gap without touching RunLayerLoop's own,
// much deeper, shared per-layer machinery.
//
// PrefillWholeTokens' own embed_codes is threaded the same way (round 2); `layers_scratch` (both
// call sites) stays unthreaded -- only allocated when an adapter is bound (the common no-adapter
// case makes zero allocation there already), and a std::vector<superslm::LayerWeights> is not POD
// bytes this byte-oriented workspace layout can safely describe without a much larger, riskier
// placement-new scheme; disclosed, not fixed.
//
// sslm_prefill's own remaining ~9716 allocations (EmbedEntry's own internal `wide` scratch, called
// once per prefilled token, PLUS RunLayerLoop's own internal per-layer scratch) are NOT reachable
// from this file's own workspace layout, and are NOT part of the recalibrated contract (C1b's own
// stable-count cell states the engine's own scratch cost as disclosed and data-independent, never
// a zero requirement) -- correctly, permanently out of this ABI layer's own scope per the design's
// own text quoted above.
#include "superslm/sslm_abi.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <new>
#include <string>
#include <vector>

#include "superslm/adapter_marshal.h"
#include "superslm/forward_sites.h"
#include "superslm/layer_marshal.h"
#include "superslm/model.h"

#include "bad_alloc_wrap.h"  // N3 pin (Claude/Poirot/2c18dab-t2139-abi-build-review.md Sec6.3):
                              // superslm::internal::MaybeThrowInjectedBadAllocFault(), the same
                              // test-only fault-injection seam S-HARDEN-7's own population already
                              // uses -- consulted once, at sslm_model_map's own Load call, so a
                              // test can force a genuine bad_alloc through this file's own new
                              // catch-and-return path without needing an actual OOM condition.

// -----------------------------------------------------------------------------------------
// Opaque handle bodies (design Sec8: sslm_model_s/sslm_kv_pool_s/sslm_workspace_s are declared
// as incomplete struct tags in the public header; defined here, matching Sec5's own
// "opaque handles passed by value" convention -- gpu_1p0.cpp's own SslmGpuContext/
// SslmGpuModelHandle definitions are the house precedent this mirrors). Defined before the
// anonymous-namespace helpers below, several of which (ConfigDomainOk) dereference
// sslm_model_s and therefore need it complete, not merely declared.
// -----------------------------------------------------------------------------------------

// C4's own resolved-once engine surface (design Sec9 C4: "resolving a loaded SslmModelView's
// WGT1/BIA1/ROP1/WSC1 sections into the LayerWeights[] array RunLayerLoop/RunGreedyDecodeLoop
// already consume" -- reuses include/superslm/layer_marshal.h, already production code since
// T-1684/T-2113, rather than re-deriving it (Claude/Brunel/t2139-abi-build-2026-08-16.md Sec3's
// own de-risking finding). Built ONCE, eagerly, at sslm_model_map time (matching
// sslm_gpu_model_map's own convention, src/gpu/gpu_1p0.cpp) rather than lazily on first use --
// lazy-on-first-use under Sec8.3's "sslm_model read-only concurrent" contract would be a real
// data race between two threads' first calls; building it once, before the handle is ever
// returned to a caller, has no such window. `layers[l]` is the BASE (no-adapter) LayerWeights
// for layer l; a per-sequence adapter (C6, not built) is composed by copying this base array and
// patching each layer's own `.adapter` field per call, never by rebuilding the marshaled arrays.
struct sslm_model_engine_cache {
	bool ok = false;
	std::string err;
	std::vector<superslm_marshal::LayerBacking> backings;
	std::vector<superslm::LayerWeights> layers;
	const int8_t* embed_weights = nullptr;
	superslm::CarriedScale embed_site_constant{};
	std::vector<int32_t> final_norm_gain;
	superslm::CarriedScale final_norm_site_constant{};
	const int8_t* head_weights = nullptr;
};

// C2. Owns the loaded model view and (C4) the resolved engine cache. `live_refs` is the D-SLM32
// lifecycle-guard bookkeeping design Sec9's own C2 gate names ("unmap-while-live is rejected") --
// incremented/decremented by C3's sslm_seq_create/_release, sslm_prefix_begin/_release, and (C6,
// not built) sslm_adapter_map.
struct sslm_model_s {
	superslm::SslmModelView view;
	std::atomic<uint32_t> live_refs{0};
	sslm_model_engine_cache engine;
};

// C6. A mapped LoRA adapter (design Sec9 C6, S-LoRA-serial's own outstanding ABI debt): wraps
// the already-proven V5 delta kernel and converter adapter mode via
// include/superslm/adapter_marshal.h's own AdapterHandle/PopulateAdapterFromView/
// ApplyAdapterToLayers -- reused, not re-derived, the same one-real-implementation discipline
// this file's own engine cache already follows for layer_marshal.h. `live_refs` is the
// SSLM_ADAPTER_HAS_LIVE_SEQUENCES guard's own bookkeeping (design Sec6), incremented/decremented
// by sslm_seq_set_adapter's own bind/unbind. `artifact_bytes` is this handle's own reported
// residency (sslm_adapter_residency) -- the mapped artifact's own byte size, a real footprint
// answer rather than an estimate.
struct sslm_adapter_s {
	sslm_model_s* base = nullptr;
	superslm_adapter::AdapterHandle handle;
	size_t artifact_bytes = 0;
	std::atomic<uint32_t> live_refs{0};
};

// C1. A caller-owned batch-orchestration scratch region (design Sec7.1, RULED design commit
// fab235c1c6 -- see this file's own top-of-file comment): a per-sequence status array across
// sslm_decode_step's own seqs[]/out_tokens, sized by max_batch, plus chunk-staging state for
// sslm_prefill's own chunk_budget-bounded internal loop. `buf`/`buf_size` are the caller's own
// memory (never freed by sslm_workspace_destroy, design Sec7.1) -- this handle's own heap
// allocation is bookkeeping only, matching sslm_model_s's own "the handle is heap-owned, the
// artifact bytes/caller buffer are not" split. `config` is cached from construction time so C4's
// own use of this workspace can re-derive the same layout sslm_workspace_size computed, without
// threading `sslm_config` through every call that takes a workspace.
struct sslm_workspace_s {
	void* buf = nullptr;
	size_t buf_size = 0;
	sslm_config config{};
};

// C1/C3. A caller-owned KV region, sized for `block_count` SEQUENCES (design Sec7.2, RULED --
// a block is one whole sequence's entire KV footprint, never a sub-sequence page). `free_list`
// holds the indices of currently-unclaimed blocks; `mutex` protects it against the concurrent
// sslm_seq_create/_release race design Sec10 dim3/dim8 names explicitly. `live_refs` is the
// SSLM_POOL_HAS_LIVE_HANDLES guard's own bookkeeping (design Sec6), incremented/decremented by
// C3's own draw/return.
struct sslm_kv_pool_s {
	// C2 (Claude/Poirot/2c18dab-t2139-abi-build-review.md): the owning model, bound at
	// sslm_kv_pool_create time. Design Sec7.2 states "a pool is bound to exactly one model for
	// its whole lifetime -- mixing models against one pool is structurally impossible" as settled
	// fact; recording `block_size` alone (below) does not enforce that, since two different
	// models can produce equal block sizes and a well-formed blob from one model can still
	// overrun a pool sized for another. This field, checked at every draw site (sslm_prefix_begin,
	// sslm_seq_create, sslm_seq_restore) and at sslm_seq_adopt_prefix, is what makes the design's
	// claim true rather than merely asserted.
	sslm_model_s* model = nullptr;
	void* buf = nullptr;
	size_t buf_size = 0;
	uint32_t block_count = 0;
	size_t block_size = 0;
	std::mutex mutex;
	std::vector<uint32_t> free_list;
	std::atomic<uint32_t> live_refs{0};
};

// C3. One prefix-under-construction / frozen prefix (design Sec7.2/Sec8). Owns its own
// hidden_codes scratch (SequenceLayerState's own residual buffer, ABI-internal bookkeeping
// allocated once at sslm_prefix_begin time -- not a hot-path allocation, matching
// sslm_model_engine_cache's own "built once, not per call" convention) and points its
// SequenceLayerState at that scratch and at its own drawn block.
struct sslm_prefix_s {
	sslm_model_s* model = nullptr;  // owning model -- SSLM_MODEL_HAS_LIVE_SEQUENCES guard (Sec6)
	sslm_kv_pool_s* pool = nullptr;
	uint32_t block_index = 0;
	uint8_t* kv_block = nullptr;
	size_t block_size = 0;
	std::vector<int8_t> hidden_codes_storage;
	superslm::SequenceLayerState state;
	int32_t current_token = -1;  // the prefix's own last-prefilled token id (mirrors
	                              // sslm_seq_s::current_token) -- carried to the adopting
	                              // sequence so decode can resume immediately after adoption,
	                              // without a redundant re-prefill of the prefix's own last token.
	bool frozen = false;
	// NO `released` flag (S5, Claude/Poirot/2c18dab-t2139-abi-build-review.md): a flag written
	// immediately before `delete` protects nothing -- there is no window on which a LIVE handle
	// carries it true, and every read of it on an already-released handle is itself a read of
	// freed memory (undefined behaviour), never a safe reject. Double-release and any other use
	// of an already-released handle is caller UB, documented here rather than pretend-guarded --
	// the same honest position sslm_workspace_destroy/sslm_kv_pool_destroy already take (neither
	// keeps a poisoned handle alive to detect misuse either).
};

// C3. One live decode sequence. `current_token` is the ABI's own tracked "last embedded or
// produced token id" (design's sslm_decode_step signature carries no token-input parameter --
// C4 reads/writes this field to resume generation across calls, since the token to embed next
// is otherwise unrecoverable from SequenceLayerState alone, which carries only the IN-PROGRESS
// residual, never the token id that produced it). `adapter` is C6's own field, unbuilt here --
// nullptr means "no adapter bound," the same convention LayerWeights::adapter already uses.
struct sslm_seq_s {
	sslm_model_s* model = nullptr;  // owning model -- SSLM_MODEL_HAS_LIVE_SEQUENCES guard (Sec6)
	sslm_kv_pool_s* pool = nullptr;
	uint32_t block_index = 0;
	uint8_t* kv_block = nullptr;
	size_t block_size = 0;
	std::vector<int8_t> hidden_codes_storage;
	superslm::SequenceLayerState state;
	int32_t current_token = -1;
	// C4 (Brunel T-2139, correctness finding during oracle verification): true immediately
	// after a completed sslm_prefill (its last token's own RunLayerLoop call already ran EVERY
	// layer, matching RunGreedyDecodeLoop's own RunWholeToken convention -- Sec8.1's own "every
	// layer" prefill semantics) -- state.hidden_codes ALREADY holds that token's fully-computed
	// final-layer residual, ready for final_norm+logits+argmax with no further RunLayerLoop
	// work. Without this flag, sslm_decode_step's own "layer_index == 0 means embed a fresh
	// token" rule would re-embed and re-run the JUST-PREFILLED last token a second time,
	// double-committing its KV position -- caught by this ticket's own bit-for-bit oracle
	// comparison against RunGreedyDecodeLoop (tools/t2139_c4_oracle.cpp), not by construction.
	// Cleared the instant sslm_decode_step consumes it (one call, no RunLayerLoop cost, since
	// the layer work already happened during prefill).
	bool ready_for_logits = false;
	sslm_adapter adapter_handle = nullptr;
	// NO `released` flag -- see sslm_prefix_s's own comment on why (S5): double-release and any
	// other use of an already-released handle is caller UB, not a state this ABI pretend-guards.
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

// N2 (Claude/Poirot/2c18dab-t2139-abi-build-review.md Sec6.3): rounds `offset` up to the next
// multiple of `alignment` (`alignment` a power of two, always SSLM_ABI_ALIGNMENT_BYTES here --
// the same constant sslm_workspace_create/sslm_kv_pool_create already enforce on the WHOLE
// buffer's own base address, design Sec7.1). Every WorkspaceLayout region carve below rounds its
// own start offset through this first: `ws->buf` is 64-byte aligned by construction, but a raw
// byte-size accumulation of PRECEDING regions is not guaranteed to land the NEXT region's start on
// any particular boundary -- `4*max_chunk_budget` at an odd max_chunk_budget is exactly the
// counter-example N2 found (a 4-mod-8 residue reaching an int64_t* region, UB on any platform
// where an 8-byte load must be 8-byte-aligned). Overflow-checked the same way every other
// composition in this function already is.
bool RoundUpToAlignment(size_t offset, size_t alignment, size_t* out) {
	size_t mask = alignment - 1;  // alignment is always a power of two (SSLM_ABI_ALIGNMENT_BYTES)
	size_t padded = 0;
	if (!CheckedAddSizeT(offset, mask, &padded)) {
		*out = kSizeMax;
		return false;
	}
	*out = padded & ~mask;
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

// C4's own artifact-resolution obligation (design Sec9 C3/Sec3's grounding), discharged by
// reusing include/superslm/layer_marshal.h -- MarshalLayer per layer, then the embed/final_norm/
// head resolution tools/sslm_generate.cpp's own "embed/final_norm/head marshaling" block already
// establishes (mirrored here verbatim rather than re-derived, per StandardsDocument Sec6.6's own
// one-real-implementation discipline -- this is a second call site of an already-proven pattern,
// not a second, independently-authored copy of its logic). Called once, at sslm_model_map time.
bool BuildEngineCache(sslm_model_s* h) {
	auto& view = h->view;
	auto& cache = h->engine;
	const uint32_t num_layers = view.config.num_hidden_layers;
	cache.backings.resize(num_layers);
	cache.layers.resize(num_layers);
	for (uint32_t l = 0; l < num_layers; ++l) {
		std::string layer_err;
		if (!superslm_marshal::MarshalLayer(view, l, view.config.num_attention_heads,
		                                     view.config.num_key_value_heads, cache.backings[l],
		                                     cache.layers[l], &layer_err)) {
			cache.err = layer_err;
			return false;
		}
	}
	const superslm::SslmTensorView* embed_w = view.weights.Tensor("embed");
	const superslm::SslmTensorView* final_gain_w = view.weights.Tensor("final_norm.gain");
	if (!embed_w || !final_gain_w) {
		cache.err = "missing embed or final_norm.gain WGT1 tensor";
		return false;
	}
	cache.final_norm_gain = superslm_marshal::WidenGainToInt32(*final_gain_w);
	bool ok = true;
	cache.embed_site_constant =
	    superslm_marshal::ReadCarriedScale(view.composition_constants, "embed", &ok);
	cache.final_norm_site_constant =
	    superslm_marshal::ReadCarriedScale(view.composition_constants, "final_norm", &ok);
	if (!ok) {
		cache.err = "missing embed/final_norm composition_constants site entry";
		return false;
	}
	cache.embed_weights = reinterpret_cast<const int8_t*>(embed_w->data);
	if (view.config.tie_word_embeddings) {
		cache.head_weights = cache.embed_weights;
	} else {
		const superslm::SslmTensorView* lm_head_w = view.weights.Tensor("lm_head");
		if (!lm_head_w) {
			cache.err = "tie_word_embeddings=0 but no \"lm_head\" WGT1 tensor is present";
			return false;
		}
		cache.head_weights = reinterpret_cast<const int8_t*>(lm_head_w->data);
	}
	cache.ok = true;
	return true;
}

// -----------------------------------------------------------------------------------------
// sslm_workspace's REAL layout (design Sec7.1, RULED design commit fab235c1c6): ABI-level
// batch-orchestration scratch, not per-layer engine scratch (RunLayerLoop/RunGreedyDecodeLoop
// take no second caller-owned buffer at all -- see the ruling). C4 is the real consumer;
// computed here, in the same anonymous namespace both sslm_workspace_size and C4's own
// implementation share, so the two can never independently drift.
// -----------------------------------------------------------------------------------------

// One entry per sslm_decode_step batch slot -- design Sec7.1's own "a per-sequence status/
// completion array across the n sequences one sslm_decode_step call drives".
struct DecodeSeqStatus {
	int32_t produced_token;  // -1 = pending (partial layer_budget this call), matching
	                          // SslmDecodeStepStatus's own sentinel convention (model.h)
	int32_t status;          // this sequence's own sslm_status for this call, as int32_t
};

struct WorkspaceLayout {
	size_t header_bytes = 0;
	size_t status_array_offset = 0;
	size_t status_array_bytes = 0;
	size_t staged_tokens_offset = 0;
	size_t staged_tokens_bytes = 0;
	// S7 (Claude/Poirot/2c18dab-t2139-abi-build-review.md): sslm_decode_step's own per-call
	// scratch, carved from THIS buffer when the caller supplies one, instead of the
	// std::vector<...> heap allocations the design's "never mallocs on the hot path" law (Sec2/
	// Sec10 dim7) names by name. Sized once here, per token, since sslm_decode_step processes its
	// `n` sequences one at a time within a single call and reuses the same scratch for each --
	// these do NOT scale with max_batch, only with the model's own hidden_size/vocab_size.
	size_t embed_codes_offset = 0;
	size_t embed_codes_bytes = 0;
	size_t final_codes_offset = 0;
	size_t final_codes_bytes = 0;
	size_t wide_logits_offset = 0;
	size_t wide_logits_bytes = 0;
	size_t logit_row_offset = 0;
	size_t logit_row_bytes = 0;
	// T-2139 closing round (curie/t2138-abi-red-suite@11e7182's own recalibrated dim7 C1a cell,
	// design commit 959336ad64): sslm_decode_step's own ready_for_logits path was ruled at
	// EXACTLY ZERO allocations (it skips RunLayerLoop entirely -- no engine-internal-scratch
	// exception applies), and measured at 1: RmsNormSite's own internal `wide` scratch for the
	// final_norm call. `int64_t[hidden_size]`, forwarded to RmsNormSite's own new
	// `external_wide_scratch` trailing parameter (forward_sites.h) -- eliminates that allocation
	// for real rather than reporting it as engine-internal-and-out-of-scope, since RmsNormSite
	// (unlike RunLayerLoop's own much larger per-layer scratch) is a single, small, ABI-callable
	// site this file already calls directly.
	size_t rms_wide_offset = 0;
	size_t rms_wide_bytes = 0;
	size_t total_bytes = 0;
	bool overflowed = false;
};

WorkspaceLayout ComputeWorkspaceLayout(const sslm_model_s* model, const sslm_config& config) {
	WorkspaceLayout L;
	bool of = false;

	// N2 (Claude/Poirot/2c18dab-t2139-abi-build-review.md Sec6.3): EVERY region's own start
	// offset is rounded up to SSLM_ABI_ALIGNMENT_BYTES before it is assigned -- not only the
	// int64_t-typed regions the finding measured, so this layout never depends on which C++ type
	// a future region happens to carve as. `cursor` is always the next free, ALREADY-ALIGNED
	// offset; each region advances it by its own byte size (never itself required to be a
	// multiple of the alignment) and the NEXT region rounds up again before claiming it.
	size_t cursor = 0;
	size_t aligned = 0;
	if (!RoundUpToAlignment(cursor, SSLM_ABI_ALIGNMENT_BYTES, &aligned)) of = true;
	L.header_bytes = 64;
	cursor = aligned;
	size_t after_header = 0;
	if (!CheckedAddSizeT(cursor, L.header_bytes, &after_header)) of = true;
	cursor = after_header;

	if (!RoundUpToAlignment(cursor, SSLM_ABI_ALIGNMENT_BYTES, &aligned)) of = true;
	L.status_array_offset = aligned;
	size_t status_bytes = 0;
	if (!CheckedMulSizeT(static_cast<size_t>(config.max_batch), sizeof(DecodeSeqStatus),
	                      &status_bytes)) {
		of = true;
	}
	L.status_array_bytes = status_bytes;
	if (!CheckedAddSizeT(L.status_array_offset, L.status_array_bytes, &cursor)) of = true;

	// sslm_prefill's own chunk_budget-bounded internal loop stages up to max_chunk_budget token
	// ids at a time (design Sec7.1's own "staged output-token bookkeeping across a sslm_prefill
	// call's own chunk_budget-bounded internal loop"). THE region N2 named: max_chunk_budget is
	// caller-supplied and odd on the real S-FREEZE prompt (5 tokens) -- its own byte size
	// (4*max_chunk_budget) is what left every region after it on a 4-mod-8 residue pre-fix.
	if (!RoundUpToAlignment(cursor, SSLM_ABI_ALIGNMENT_BYTES, &aligned)) of = true;
	L.staged_tokens_offset = aligned;
	size_t staged_bytes = 0;
	if (!CheckedMulSizeT(static_cast<size_t>(config.max_chunk_budget), sizeof(int32_t),
	                      &staged_bytes)) {
		of = true;
	}
	L.staged_tokens_bytes = staged_bytes;
	if (!CheckedAddSizeT(L.staged_tokens_offset, L.staged_tokens_bytes, &cursor)) of = true;

	// S7's own four regions -- present only when `model` is supplied (sslm_workspace_size always
	// passes one; kept optional in signature-shape only so this function's own internal callers
	// never need a fake model to size the header/status/staged regions alone).
	if (model != nullptr) {
		const superslm::SslmModelConfig& c = model->view.config;

		if (!RoundUpToAlignment(cursor, SSLM_ABI_ALIGNMENT_BYTES, &aligned)) of = true;
		L.embed_codes_offset = aligned;
		L.embed_codes_bytes = static_cast<size_t>(c.hidden_size);  // int8 codes
		if (!CheckedAddSizeT(L.embed_codes_offset, L.embed_codes_bytes, &cursor)) of = true;

		if (!RoundUpToAlignment(cursor, SSLM_ABI_ALIGNMENT_BYTES, &aligned)) of = true;
		L.final_codes_offset = aligned;
		L.final_codes_bytes = static_cast<size_t>(c.hidden_size);
		if (!CheckedAddSizeT(L.final_codes_offset, L.final_codes_bytes, &cursor)) of = true;

		if (!RoundUpToAlignment(cursor, SSLM_ABI_ALIGNMENT_BYTES, &aligned)) of = true;
		L.wide_logits_offset = aligned;  // int64_t* -- the region N2 measured landing at
		                                 // 4-mod-8 whenever max_chunk_budget was odd.
		if (!CheckedMulSizeT(static_cast<size_t>(c.vocab_size), sizeof(int64_t),
		                      &L.wide_logits_bytes)) {
			of = true;
		}
		if (!CheckedAddSizeT(L.wide_logits_offset, L.wide_logits_bytes, &cursor)) of = true;

		if (!RoundUpToAlignment(cursor, SSLM_ABI_ALIGNMENT_BYTES, &aligned)) of = true;
		L.logit_row_offset = aligned;
		if (!CheckedMulSizeT(static_cast<size_t>(c.vocab_size), sizeof(int32_t),
		                      &L.logit_row_bytes)) {
			of = true;
		}
		if (!CheckedAddSizeT(L.logit_row_offset, L.logit_row_bytes, &cursor)) of = true;

		if (!RoundUpToAlignment(cursor, SSLM_ABI_ALIGNMENT_BYTES, &aligned)) of = true;
		L.rms_wide_offset = aligned;  // int64_t* -- the other region N2 measured misaligned.
		if (!CheckedMulSizeT(static_cast<size_t>(c.hidden_size), sizeof(int64_t),
		                      &L.rms_wide_bytes)) {
			of = true;
		}
		if (!CheckedAddSizeT(L.rms_wide_offset, L.rms_wide_bytes, &cursor)) of = true;
	}

	L.total_bytes = of ? kSizeMax : cursor;
	L.overflowed = of;
	return L;
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
	const WorkspaceLayout L = ComputeWorkspaceLayout(model, *config);
	return L.overflowed ? 0 : L.total_bytes;
}

// CORRECTED to the ruled unit (design Sec7.2, Brunel T-2139 Sec4, design commit fab235c1c6): a
// "block" is one WHOLE sequence's entire KV footprint across every layer -- exactly the byte
// count RunLayerLoop's own workspace/workspace_size parameter needs for one sequence -- never a
// sub-sequence PagedAttention page. CFG1's own `kv_block_size` field (SslmModelConfig, a
// tokens-per-page count) plays NO role in this formula under the ruling; this function's own
// name is now a slight misnomer relative to that unrelated CFG1 field, which the ruling itself
// notes is model metadata this ABI's block unit does not consume.
extern "C" size_t sslm_kv_block_size(sslm_model model) {
	if (!model) return 0;
	const superslm::SslmModelConfig& c = model->view.config;
	// num_hidden_layers * context_cap * num_key_value_heads * head_dim * 2 (K+V) *
	// kv_precision_width -- the exact formula the KeyRow/ValueRow accessor comment
	// (forward_sites.h) states for one sequence's own contiguous KV span. A pure product chain
	// (not a sum-of-products), so chained CheckedMulSizeT rather than SaturatingAccumulator
	// (which composes sum-of-products, not a running product).
	size_t v = static_cast<size_t>(c.num_hidden_layers);
	bool overflowed = false;
	auto Mul = [&](size_t factor) {
		size_t out = 0;
		if (!CheckedMulSizeT(v, factor, &out)) overflowed = true;
		v = overflowed ? kSizeMax : out;
	};
	Mul(static_cast<size_t>(c.context_cap));
	Mul(static_cast<size_t>(c.num_key_value_heads));
	Mul(static_cast<size_t>(c.head_dim));
	Mul(2);
	Mul(KvElementBytes(c.kv_precision));
	return v;
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
	// KV store a single sequence carries -- CORRECTED to the ruled block unit (Sec4 above): a
	// sequence draws exactly ONE block (its own whole-sequence KV footprint), so
	// kv_block_count is always 1 for a real saved sequence, not a ceil(context_cap/page) count
	// under the now-retired PagedAttention reading. Fixed fields: magic(4) + model_hash(32) +
	// kv_precision(4) + schema_name_hash(8) + dfa_walk_state(4) + adapter_binding_id(8) +
	// context_length(8) + layer_index(4) + current_token(4, design commit 9e2995f4e7's own
	// amendment -- see the C5 block's own top comment) + hidden_scale(16, CarriedScale as two
	// int64) + kv_saturation_count(8) + kv_block_count(4) = 104 bytes.
	constexpr size_t kFixedHeaderBytes = 104;
	SaturatingAccumulator acc;
	acc.value = kFixedHeaderBytes;
	acc.AddProduct(1, static_cast<size_t>(c.hidden_size));  // residual_bytes, int8 codes
	acc.AddProduct(1, block_size);  // kv_blocks: exactly one block (Sec7.2's ruled unit)
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
// S3 (Claude/Poirot/2c18dab-t2139-abi-build-review.md): the ONE real definition is now the
// public header's SSLM_ABI_ALIGNMENT_BYTES macro (sslm_abi.h) -- this constant reads it rather
// than restating "64" a second time, closing the exact drift class the finding named (the
// example previously had to hardcode this number by reading THIS file, which the S-FREEZE bar
// says a consumer never does).
constexpr size_t kAbiAlignmentBytes = SSLM_ABI_ALIGNMENT_BYTES;

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
	// N3 (Claude/Poirot/2c18dab-t2139-abi-build-review.md Sec6.3): never throws across this
	// extern "C" boundary -- under /EHc (this file's own build flag, build.bat/CMakeLists.txt)
	// a throwing extern "C" function's behavior is undefined (MSVC's own C4297 diagnostic, fired
	// 63 times a build against the pre-fix `throw std::bad_alloc()` at all seven construction
	// verbs). SSLM_ALLOCATION_FAILED (sslm_abi.h, ordinal 25) reports the cause through the
	// closed status taxonomy instead.
	if (!h) return SSLM_ALLOCATION_FAILED;
	h->buf = buf;
	h->buf_size = buf_size;
	h->config = *config;
	// Zero the status/staged-token region so a workspace's first real use (C4) never reads
	// uninitialized bytes -- construction is not on the hot path (design Sec7.1), so this one
	// memset here is sound.
	std::memset(buf, 0, required);
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
	// S2 (Claude/Poirot/2c18dab-t2139-abi-build-review.md): design Sec6 names "Sec7.1's
	// workspace/pool construction verbs" as the ones that check alignment explicitly, and
	// sslm_workspace_create (above) already does. Checked AFTER the size check, matching
	// sslm_workspace_create's own ordering (buf_size before alignment) -- so a buffer that is
	// BOTH too small and misaligned still reports SSLM_BUFFER_TOO_SMALL first, consistent with
	// the sibling verb and with this suite's own too-small-buffer cells, which do not separately
	// construct an aligned-but-too-small buffer.
	if (!IsAligned(buf)) return SSLM_MISALIGNED_BUFFER;
	sslm_kv_pool_s* h = new (std::nothrow) sslm_kv_pool_s();
	// N3 -- see sslm_workspace_create's own identical comment, above.
	if (!h) return SSLM_ALLOCATION_FAILED;
	h->model = model;
	h->buf = buf;
	h->buf_size = buf_size;
	h->block_count = block_count;
	h->block_size = block_size;
	h->free_list.reserve(block_count);
	for (uint32_t i = 0; i < block_count; ++i) h->free_list.push_back(i);
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
	// N3 (Claude/Poirot/2c18dab-t2139-abi-build-review.md Sec6.3): SslmModel::Load "throws only
	// std::bad_alloc" (model.h's own contract, S-HARDEN-7) -- previously left uncaught here on
	// the theory that "a bad_alloc crosses this ABI boundary unchanged," which is not a guarantee
	// /EHc gives an extern "C" function (MSVC's own C4297: undefined behavior, not propagation).
	// Caught explicitly now, mapped to SSLM_ALLOCATION_FAILED (sslm_abi.h, ordinal 25) through
	// the closed status taxonomy instead of crossing the boundary as an exception.
	superslm::SslmModelStatus st;
	try {
		// N3 pin: a no-op outside the test-injection build (bad_alloc_wrap.h's own guard) --
		// armed, this throws std::bad_alloc right here, exercising the catch below for real.
		superslm::internal::MaybeThrowInjectedBadAllocFault();
		st = superslm::SslmModel::Load(static_cast<const uint8_t*>(data), size, view, &err);
	} catch (const std::bad_alloc&) {
		return SSLM_ALLOCATION_FAILED;
	}
	if (st != superslm::SslmModelStatus::Ok) {
		// design Sec6: every SslmModel::Load rejection maps to SSLM_ARTIFACT_REJECTED; the
		// specific SslmModelStatus (`err`/`st`) has no exposed channel on this signature
		// (sslm_g5.h's own 3-argument sslm_model_map shape, which this design's Sec7.4
		// derivation method requires this design match verbatim, carries no diagnostic
		// out-parameter) -- discarded here, not silently invented a channel for.
		return SSLM_ARTIFACT_REJECTED;
	}

	sslm_model_s* h = new (std::nothrow) sslm_model_s{std::move(view)};
	// N3 -- see sslm_workspace_create's own identical comment.
	if (!h) return SSLM_ALLOCATION_FAILED;
	// C3/C4's own dependency: resolve every layer's LayerWeights[] plus embed/final_norm/head
	// once, now, while the handle is not yet visible to any caller (Sec8.3's "sslm_model
	// read-only concurrent" contract, honored by building this BEFORE *out is set, never lazily
	// on first use under concurrent access). An artifact that loads structurally and passes
	// every value-domain check (SslmModel::Load, above) but is missing a WGT1 tensor this ABI
	// needs to actually run it (embed/final_norm.gain/lm_head, or a per-layer projection) is
	// correspondingly SSLM_ARTIFACT_REJECTED here too -- the same "this artifact cannot be used"
	// class SslmModel::Load's own rejection already maps to, extended to this ABI's own
	// resolution step rather than left to fail unrecoverably at first real use.
	if (!BuildEngineCache(h)) {
		delete h;
		return SSLM_ARTIFACT_REJECTED;
	}
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

// -----------------------------------------------------------------------------------------
// C3 -- prefix and sequence lifecycle over a real pool (design Sec7.2/Sec8/Sec9 C3): the
// block-allocation/free-list machinery against a real sslm_kv_pool (C1), plus the real engine
// call for sslm_prefix_prefill (reusing C4's own resolved-once engine cache, sslm_model_
// engine_cache, built at sslm_model_map time -- see this file's own top-of-file comment on why
// C3 and C4 share this infrastructure: sslm_prefix_prefill cannot be built for real without it).
// -----------------------------------------------------------------------------------------

namespace {

// Draws one free block from `pool`. SSLM_KV_POOL_EXHAUSTED if none remain -- design Sec7.2's
// ruled exhaustion timing: fires at draw time (sslm_seq_create/sslm_prefix_begin), never
// mid-prefill (a block, once drawn, is never re-drawn mid-construction).
sslm_status DrawBlock(sslm_kv_pool_s* pool, uint32_t* out_block_index) {
	std::lock_guard<std::mutex> lock(pool->mutex);
	if (pool->free_list.empty()) return SSLM_KV_POOL_EXHAUSTED;
	*out_block_index = pool->free_list.back();
	pool->free_list.pop_back();
	pool->live_refs.fetch_add(1, std::memory_order_acq_rel);
	return SSLM_OK;
}

// Poison-fills and returns `block_index` to `pool`'s free list -- design Sec7.2's own
// "poison-filled" leak-check obligation (Sec17 dim 1: no leaked content crosses a
// release/create boundary).
void ReturnBlock(sslm_kv_pool_s* pool, uint32_t block_index, uint8_t* kv_block, size_t block_size) {
	std::memset(kv_block, 0xCD, block_size);
	std::lock_guard<std::mutex> lock(pool->mutex);
	pool->free_list.push_back(block_index);
	pool->live_refs.fetch_sub(1, std::memory_order_acq_rel);
}

// EmbedEntry+RunLayerLoop's own real, distinguishable domain rejections (checked_chain_funnel.h/
// forward_sites.h) have no counterpart in this ABI's own closed, 17-member Sec6 taxonomy -- none
// is named for this call shape, since a well-formed, load-time-value-domain-checked artifact
// (S-HARDEN-1's own schema-value gate, already run by SslmModel::Load) is not expected to
// produce one on any real, in-domain call. Mapped to SSLM_ARTIFACT_REJECTED here (the closest
// existing "this artifact's own content cannot be used as requested" family, Sec6) rather than
// invented a new status this design never named -- flagged as a modeling choice, not a design
// citation, since Sec6's own text does not name this mapping. Never observed on any real
// artifact this build tested against (see this ticket's own build log).
sslm_status MapForwardStatus(superslm::SslmForwardStatus st) {
	return st == superslm::SslmForwardStatus::Ok ? SSLM_OK : SSLM_ARTIFACT_REJECTED;
}

// C6: resolves the LayerWeights[] array a real RunLayerLoop call should use -- the model's own
// cached BASE array directly when no adapter is bound (the common, zero-copy case; Sec8.3's
// "sslm_model read-only concurrent" contract means this shared array is never mutated in
// place), or a per-call COPY with every layer's own `.adapter` field patched via
// adapter_marshal.h's ApplyAdapterToLayers when one is (`*scratch` is the copy's own backing
// storage, owned by the caller for the call's duration). Two sequences decoding concurrently
// against different adapters (or one adapter, one none) never see each other's `.adapter`
// patch, because each gets its own `*scratch` -- the shared cache itself is never touched.
const superslm::LayerWeights* ResolveLayers(sslm_model_s* model, sslm_adapter_s* adapter,
                                             std::vector<superslm::LayerWeights>* scratch) {
	if (!adapter) return model->engine.layers.data();
	*scratch = model->engine.layers;
	superslm_adapter::ApplyAdapterToLayers(&adapter->handle, scratch->data(),
	                                        model->view.config.num_hidden_layers);
	return scratch->data();
}

// Shared whole-token prefill loop -- design Sec8.1's "every layer" prefill semantics, the SAME
// EmbedEntry+RunLayerLoop composition RunGreedyDecodeLoop's own RunWholeToken lambda uses for
// its own prefill phase (forward_sites.cpp), reused here as ONE implementation rather than two
// independently-authored copies for sslm_prefix_prefill and sslm_prefill (StandardsDocument
// Sec6.6's own one-real-implementation discipline -- the shape both callers need is identical:
// a state/kv-block pair, a token array, a chunk_budget). Processes up to min(count,
// chunk_budget) tokens, stopping early (leaving state/*consumed resumable) on the first
// rejection, exactly as sslm_prefix_prefill's own already-tested contract does. `adapter` is
// nullptr for every sslm_prefix_prefill call (adapters bind to sequences, never prefixes,
// design Sec12) and the sequence's own bound adapter (possibly nullptr) for sslm_prefill.
sslm_status PrefillWholeTokens(sslm_model_s* model, superslm::SequenceLayerState& state,
                                uint8_t* kv_block, size_t block_size, const int32_t* tokens,
                                int32_t count, int32_t chunk_budget, int32_t* consumed,
                                int32_t* out_last_token, sslm_adapter_s* adapter,
                                sslm_workspace_s* ws) {
	const superslm::SslmModelConfig& c = model->view.config;
	const int32_t n = count < chunk_budget ? count : chunk_budget;
	// S7 fix round 2 (coordinator's own closing-round brief): thread this function's own
	// embed_codes scratch through a caller-supplied, correctly-sized workspace, exactly the
	// carving sslm_decode_step already does -- the SAME embed_codes region (design Sec7.1's own
	// workspace is never used by two calls concurrently on one handle; sequential reuse across
	// sslm_prefill/sslm_decode_step calls is the whole point of a shared, caller-owned scratch
	// region, design Sec8.3/dim8 M1). Falls back to a heap-allocated vector when `ws` is null or
	// undersized for this model, so no existing nullptr-`ws` caller regresses.
	const WorkspaceLayout layout = ws ? ComputeWorkspaceLayout(model, ws->config) : WorkspaceLayout{};
	const bool ws_usable = ws && !layout.overflowed && ws->buf_size >= layout.total_bytes;
	std::vector<int8_t> embed_codes_fallback;
	int8_t* embed_codes;
	if (ws_usable) {
		embed_codes = reinterpret_cast<int8_t*>(static_cast<uint8_t*>(ws->buf) +
		                                         layout.embed_codes_offset);
	} else {
		embed_codes_fallback.assign(c.hidden_size, 0);
		embed_codes = embed_codes_fallback.data();
	}
	std::vector<superslm::LayerWeights> layers_scratch;
	const superslm::LayerWeights* layers = ResolveLayers(model, adapter, &layers_scratch);
	for (int32_t i = 0; i < n; ++i) {
		const int32_t token = tokens[i];
		if (token < 0 || static_cast<uint32_t>(token) >= c.vocab_size) {
			return SSLM_TOKEN_ID_OUT_OF_RANGE;
		}
		if (state.context_length >= static_cast<int64_t>(c.context_cap)) {
			return SSLM_CONTEXT_CAP_EXCEEDED;
		}

		superslm::CarriedScale embed_scale{};
		const superslm::SslmForwardStatus est = superslm::EmbedEntry(
		    token, static_cast<int32_t>(c.vocab_size), model->engine.embed_weights, c.hidden_size,
		    model->engine.embed_site_constant, embed_codes, &embed_scale);
		if (est != superslm::SslmForwardStatus::Ok) return MapForwardStatus(est);
		for (uint32_t k = 0; k < c.hidden_size; ++k) state.hidden_codes[k] = embed_codes[k];
		state.hidden_scale = embed_scale;
		state.layer_index = 0;

		const superslm::SslmForwardStatus st = superslm::RunLayerLoop(
		    state, layers, c.num_hidden_layers,
		    /*layer_budget=*/c.num_hidden_layers, c.hidden_size, c.head_dim,
		    c.num_key_value_heads, c.intermediate_size, c.context_cap, model->view.rope_tables,
		    kv_block, block_size, /*site_prefix=*/{}, /*token_index=*/0, nullptr);
		if (st != superslm::SslmForwardStatus::Ok) return MapForwardStatus(st);
		// forward_sites.h: "a sequence resting between whole tokens carries a marker at layer
		// 0" -- RunLayerLoop, given the full layer_budget above, leaves layer_index at
		// num_hidden_layers on success; reset to the resting convention before the next token.
		state.layer_index = 0;
		++(*consumed);
		if (out_last_token) *out_last_token = token;
	}
	return SSLM_OK;
}

}  // namespace

extern "C" sslm_status sslm_prefix_begin(sslm_model model, sslm_kv_pool* pool, sslm_prefix* out) {
	if (!out) return SSLM_INVALID_ARGUMENT;
	*out = nullptr;
	if (!model || !pool || !*pool) return SSLM_INVALID_ARGUMENT;
	sslm_kv_pool_s* p = *pool;
	// C2: reject a pool bound to a different model before drawing from it (design Sec7.2's
	// "structurally impossible" claim, enforced rather than assumed).
	if (p->model != model) return SSLM_INVALID_ARGUMENT;
	uint32_t block_index = 0;
	const sslm_status draw_st = DrawBlock(p, &block_index);
	if (draw_st != SSLM_OK) return draw_st;

	// M4 (Claude/Poirot/2c18dab-t2139-abi-build-review.md): return the block before reporting
	// allocation failure, or the pool permanently loses it (and sslm_kv_pool_destroy would then
	// reject forever with SSLM_POOL_HAS_LIVE_HANDLES for a block nobody can ever release). N3
	// (same casebook, Sec6.3): return SSLM_ALLOCATION_FAILED rather than throw across this
	// extern "C" boundary -- see sslm_workspace_create's own identical comment.
	auto* h = new (std::nothrow) sslm_prefix_s();
	if (!h) {
		ReturnBlock(p, block_index,
		            static_cast<uint8_t*>(p->buf) + static_cast<size_t>(block_index) * p->block_size,
		            p->block_size);
		return SSLM_ALLOCATION_FAILED;
	}
	h->model = model;
	h->pool = p;
	h->block_index = block_index;
	h->kv_block = static_cast<uint8_t*>(p->buf) + static_cast<size_t>(block_index) * p->block_size;
	h->block_size = p->block_size;
	h->hidden_codes_storage.assign(model->view.config.hidden_size, 0);
	h->state.hidden_codes = h->hidden_codes_storage.data();
	model->live_refs.fetch_add(1, std::memory_order_acq_rel);
	*out = h;
	return SSLM_OK;
}

// `kind` (design Sec4/Sec12): behaviorally inert on this base-only, SSLM_SPAN_PROMPT-throughout
// build -- G5's schema-content differentiation is not built (declared-but-deferred, this
// design's own Sec12 scope statement); every call is driven identically regardless of `kind`.
extern "C" sslm_status sslm_prefix_prefill(sslm_model model, sslm_prefix prefix,
                                            const int32_t* tokens, int32_t count,
                                            int32_t chunk_budget, sslm_span_kind kind,
                                            sslm_workspace ws, int32_t* consumed) {
	(void)kind;
	// S7 fix round 2: ws is now genuinely read (see PrefillWholeTokens' own comment) -- no longer
	// merely accepted for call-shape parity.
	if (!consumed) return SSLM_INVALID_ARGUMENT;
	*consumed = 0;
	if (!model || !prefix) return SSLM_INVALID_ARGUMENT;
	if (count < 0 || chunk_budget < 1 || (count > 0 && !tokens)) return SSLM_INVALID_ARGUMENT;
	if (prefix->frozen) return SSLM_PREFIX_FROZEN_REJECTED;
	if (!model->engine.ok) return SSLM_ARTIFACT_REJECTED;

	return PrefillWholeTokens(model, prefix->state, prefix->kv_block, prefix->block_size, tokens,
	                           count, chunk_budget, consumed, &prefix->current_token, nullptr, ws);
}

extern "C" sslm_status sslm_prefix_freeze(sslm_prefix prefix) {
	if (!prefix) return SSLM_INVALID_ARGUMENT;
	prefix->frozen = true;
	return SSLM_OK;
}

extern "C" sslm_status sslm_prefix_release(sslm_prefix prefix) {
	if (!prefix) return SSLM_INVALID_ARGUMENT;
	ReturnBlock(prefix->pool, prefix->block_index, prefix->kv_block, prefix->block_size);
	prefix->model->live_refs.fetch_sub(1, std::memory_order_acq_rel);
	delete prefix;
	return SSLM_OK;
}

extern "C" sslm_status sslm_seq_create(sslm_model model, sslm_kv_pool* pool, sslm_seq* out) {
	if (!out) return SSLM_INVALID_ARGUMENT;
	*out = nullptr;
	if (!model || !pool || !*pool) return SSLM_INVALID_ARGUMENT;
	sslm_kv_pool_s* p = *pool;
	// C2: see sslm_prefix_begin's own identical check.
	if (p->model != model) return SSLM_INVALID_ARGUMENT;
	uint32_t block_index = 0;
	const sslm_status draw_st = DrawBlock(p, &block_index);
	if (draw_st != SSLM_OK) return draw_st;

	// M4/N3 (Claude/Poirot/2c18dab-t2139-abi-build-review.md): see sslm_prefix_begin's own
	// identical return-before-throw (M4) and no-throw-across-the-boundary (N3) fixes.
	auto* h = new (std::nothrow) sslm_seq_s();
	if (!h) {
		ReturnBlock(p, block_index,
		            static_cast<uint8_t*>(p->buf) + static_cast<size_t>(block_index) * p->block_size,
		            p->block_size);
		return SSLM_ALLOCATION_FAILED;
	}
	h->model = model;
	h->pool = p;
	h->block_index = block_index;
	h->kv_block = static_cast<uint8_t*>(p->buf) + static_cast<size_t>(block_index) * p->block_size;
	h->block_size = p->block_size;
	h->hidden_codes_storage.assign(model->view.config.hidden_size, 0);
	h->state.hidden_codes = h->hidden_codes_storage.data();
	h->current_token = -1;
	model->live_refs.fetch_add(1, std::memory_order_acq_rel);
	*out = h;
	return SSLM_OK;
}

extern "C" sslm_status sslm_seq_release(sslm_seq seq) {
	if (!seq) return SSLM_INVALID_ARGUMENT;
	if (seq->adapter_handle) {
		// A still-bound adapter's own live_refs must drop too, or sslm_adapter_release would
		// wrongly see a live reference from a sequence that no longer exists.
		seq->adapter_handle->live_refs.fetch_sub(1, std::memory_order_acq_rel);
	}
	ReturnBlock(seq->pool, seq->block_index, seq->kv_block, seq->block_size);
	seq->model->live_refs.fetch_sub(1, std::memory_order_acq_rel);
	delete seq;
	return SSLM_OK;
}

extern "C" sslm_status sslm_seq_reset(sslm_seq seq) {
	if (!seq) return SSLM_INVALID_ARGUMENT;
	if (seq->state.layer_index != 0) return SSLM_SEQ_RESET_MIDTOKEN_REJECTED;
	std::memset(seq->kv_block, 0xCD, seq->block_size);
	seq->state.context_length = 0;
	seq->state.kv_saturation_count = 0;
	seq->state.layer_index = 0;
	seq->state.hidden_scale = superslm::CarriedScale{};
	std::fill(seq->hidden_codes_storage.begin(), seq->hidden_codes_storage.end(), int8_t{0});
	seq->current_token = -1;
	// S1 (Claude/Poirot/2c18dab-t2139-abi-build-review.md): reset-as-restart (design Sec12,
	// D-SLM32) must leave the handle indistinguishable from a freshly created one, and a fresh
	// handle's ready_for_logits is false. Left uncleared, a reset sequence carried layer_index==0,
	// current_token==-1, context_length==0 AND ready_for_logits==true -- sslm_decode_step's own
	// ready_for_logits branch would then skip straight to final_norm/logits/argmax over the
	// just-zeroed residual, emitting a token from an empty sequence with no history.
	seq->ready_for_logits = false;
	return SSLM_OK;
}

// RULED, copy-on-adopt (design Sec7.2, design commit fab235c1c6): an eager, whole-block copy of
// the frozen prefix's own occupied bytes into the adopting sequence's own already-drawn block,
// never physical sharing (the real block-table indirection true sharing needs is committed
// post-1.0 engine work, D-SLM3457). Copies the WHOLE block (not merely the first
// context_length-worth of bytes): KeyRow/ValueRow's own per-(layer,head)-major layout means
// occupied positions are interleaved across the whole block, one span per layer, not a single
// contiguous prefix of the buffer -- copying the entire block is the simplest construction that
// is unconditionally correct (every layer's own occupied span lands intact; the few bytes past
// context_length within each layer's own span are copied too, but nothing ever reads them,
// since context_length gates what RunLayerLoop treats as valid history).
extern "C" sslm_status sslm_seq_adopt_prefix(sslm_seq seq, sslm_prefix prefix) {
	if (!seq) return SSLM_INVALID_ARGUMENT;
	if (!prefix) return SSLM_INVALID_ARGUMENT;
	// M2 (Claude/Poirot/2c18dab-t2139-abi-build-review.md): SSLM_PREFIX_FROZEN_REJECTED's design
	// Sec6 meaning is "sslm_prefix_prefill called on a prefix PAST freeze" -- i.e. the call is
	// rejected BECAUSE the prefix is frozen. Adoption is the opposite condition (rejected because
	// the prefix is NOT YET frozen); the design's closed 18-enumerator Sec6 taxonomy names no
	// status for it, so SSLM_INVALID_ARGUMENT (the general call-shape-wrong family) is used here
	// rather than reusing an enumerator whose own name asserts the opposite of what happened.
	if (!prefix->frozen) return SSLM_INVALID_ARGUMENT;
	// C2 (Claude/Poirot/2c18dab-t2139-abi-build-review.md): the real granularity is the MODEL,
	// not the block size two different-but-same-geometry models could still share -- checked
	// first, model identity, then block_size as defense-in-depth (a model mismatch that somehow
	// carried equal block_size would still be a mismatch worth catching, though it cannot arise
	// once sslm_kv_pool_s is itself bound to one model, below).
	if (seq->model != prefix->model) return SSLM_INVALID_ARGUMENT;  // different models
	if (seq->block_size != prefix->block_size) return SSLM_INVALID_ARGUMENT;
	std::memcpy(seq->kv_block, prefix->kv_block, seq->block_size);
	std::copy(prefix->hidden_codes_storage.begin(), prefix->hidden_codes_storage.end(),
	          seq->hidden_codes_storage.begin());
	seq->state.hidden_scale = prefix->state.hidden_scale;
	seq->state.layer_index = prefix->state.layer_index;
	seq->state.kv_saturation_count = prefix->state.kv_saturation_count;
	seq->state.context_length = prefix->state.context_length;
	seq->current_token = prefix->current_token;  // carries the prefix's own last-prefilled
	                                              // token, so decode can resume immediately
	                                              // after adoption with no redundant re-prefill.
	// The copied hidden state is the prefix's own last-prefilled token's fully-computed final
	// hidden state (identical reasoning to sslm_prefill's own ready_for_logits set, above) --
	// ready for decode's first logits computation with no re-embed or further RunLayerLoop work.
	seq->ready_for_logits = (prefix->current_token >= 0);
	return SSLM_OK;
}

// -----------------------------------------------------------------------------------------
// C4 -- prefill and decode over the internal engine (design Sec9 C4): wraps
// RunLayerLoop/RunGreedyDecodeLoop, never reimplements them (design's own C4 law). sslm_prefill
// reuses C3's own PrefillWholeTokens verbatim (a real sequence's own state/kv-block, in place of
// a prefix's). sslm_decode_step is new: a genuinely bounded, resumable single micro-step over
// RunLayerLoop directly (never RunGreedyDecodeLoop, which always runs a whole token to
// completion in one call and manages its own prompt+generation loop together) -- the
// composition (embed at a token boundary, RunLayerLoop bounded to layer_budget, finish with
// final_norm+logits+argmax when the budget completes a token) is the SAME one
// RunGreedyDecodeLoop's own RunWholeToken lambda performs per whole token
// (forward_sites.cpp), here split at an externally-observable, resumable boundary
// (params->layer_budget) instead of always running to completion internally -- verified
// bit-for-bit against a direct RunGreedyDecodeLoop call by this ticket's own oracle tool
// (tools/t2139_c4_oracle.cpp).
// -----------------------------------------------------------------------------------------

extern "C" sslm_status sslm_prefill(sslm_model model, sslm_seq seq, const int32_t* tokens,
                                     int32_t count, int32_t chunk_budget, sslm_span_kind kind,
                                     sslm_workspace ws, int32_t* consumed) {
	(void)kind;  // behaviorally inert, same disposition as sslm_prefix_prefill's own (design
	             // Sec4/Sec12)
	// S7 fix round 2: ws is now genuinely read (see PrefillWholeTokens' own comment).
	if (!consumed) return SSLM_INVALID_ARGUMENT;
	*consumed = 0;
	if (!model || !seq) return SSLM_INVALID_ARGUMENT;
	if (count < 0 || chunk_budget < 1 || (count > 0 && !tokens)) return SSLM_INVALID_ARGUMENT;
	if (!model->engine.ok) return SSLM_ARTIFACT_REJECTED;

	const sslm_status st = PrefillWholeTokens(model, seq->state, seq->kv_block, seq->block_size,
	                                           tokens, count, chunk_budget, consumed,
	                                           &seq->current_token, seq->adapter_handle, ws);
	if (*consumed > 0) {
		// The last token this call processed already ran every layer (PrefillWholeTokens'
		// own full-layer_budget convention) -- its final hidden state is ready for logits with
		// no further RunLayerLoop work (see sslm_seq_s::ready_for_logits's own comment).
		seq->ready_for_logits = true;
	}
	return st;
}

extern "C" sslm_status sslm_decode_step(sslm_model model, sslm_seq* seqs, int32_t n,
                                         const sslm_decode_params* params, sslm_workspace ws,
                                         int32_t* out_tokens) {
	if (!model || !seqs || !params || !out_tokens) return SSLM_INVALID_ARGUMENT;
	if (n < 1) return SSLM_INVALID_ARGUMENT;
	if (params->layer_budget < 1 ||
	    static_cast<uint32_t>(params->layer_budget) > model->view.config.num_hidden_layers) {
		return SSLM_INVALID_ARGUMENT;
	}
	if (!model->engine.ok) return SSLM_ARTIFACT_REJECTED;
	// Every sequence validated BEFORE any is touched -- a malformed entry anywhere in the batch
	// leaves every sequence's state exactly as it was (this call's own "reject leaves state
	// unperturbed" contract, matching every other lifecycle guard in this design).
	for (int32_t i = 0; i < n; ++i) {
		if (!seqs[i]) return SSLM_INVALID_ARGUMENT;
		if (seqs[i]->state.layer_index == 0 && seqs[i]->current_token < 0 &&
		    !seqs[i]->ready_for_logits) {
			// No prompt/prior token to resume from, AND no already-computed final hidden state
			// waiting for logits -- a sequence reaching decode_step in neither shape has nothing
			// to produce a token from. (Bug found by execution, S-FREEZE-EXAMPLE's own restore-
			// then-decode step: this check originally omitted the ready_for_logits exemption,
			// rejecting every legitimately-restored resting sequence outright.)
			return SSLM_INVALID_ARGUMENT;
		}
	}

	const superslm::SslmModelConfig& c = model->view.config;

	// S7 (Claude/Poirot/2c18dab-t2139-abi-build-review.md): "never mallocs on the hot path"
	// (design Sec2/Sec10 dim7) -- when the caller supplies a workspace sized against THIS model
	// (the ordinary case: sslm_workspace_size(model, &config) then sslm_workspace_create), this
	// call's own per-token scratch is carved from it instead of heap-allocated per call. `ws` is
	// still optional on this signature (unchanged from before this fix, and already exercised
	// that way throughout this build's own smokes and the red suite) -- a null or
	// too-small-for-this-model workspace falls back to the pre-fix heap-allocating path, so no
	// existing caller regresses; it is the SUPPLIED-and-sized case that now actually honors the
	// caller-owned-memory contract instead of silently discarding the buffer.
	const WorkspaceLayout layout = ws ? ComputeWorkspaceLayout(model, ws->config) : WorkspaceLayout{};
	const bool ws_usable = ws && !layout.overflowed && ws->buf_size >= layout.total_bytes;
	uint8_t* const ws_base = ws_usable ? static_cast<uint8_t*>(ws->buf) : nullptr;

	// Fallback storage -- only actually allocated (non-empty) when ws_usable is false, so the
	// caller-supplied-workspace path performs no heap allocation here at all.
	std::vector<int8_t> embed_codes_fallback;
	std::vector<int8_t> final_codes_fallback;
	std::vector<int64_t> wide_logits_fallback;
	std::vector<int32_t> logit_row_fallback;
	int8_t* embed_codes;
	int8_t* final_codes;
	int64_t* wide_logits;
	int32_t* logit_row;
	// rms_wide: nullptr on the fallback path is CORRECT, not an oversight -- RmsNormSite's own
	// external_wide_scratch parameter (forward_sites.h) already defaults to nullptr, meaning "fall
	// back to your own internal allocation," exactly this path's pre-existing behavior.
	int64_t* rms_wide = nullptr;
	if (ws_usable) {
		embed_codes = reinterpret_cast<int8_t*>(ws_base + layout.embed_codes_offset);
		final_codes = reinterpret_cast<int8_t*>(ws_base + layout.final_codes_offset);
		wide_logits = reinterpret_cast<int64_t*>(ws_base + layout.wide_logits_offset);
		logit_row = reinterpret_cast<int32_t*>(ws_base + layout.logit_row_offset);
		rms_wide = reinterpret_cast<int64_t*>(ws_base + layout.rms_wide_offset);
		// N2 (Claude/Poirot/2c18dab-t2139-abi-build-review.md Sec6.3): a real, compiled-in check
		// at the exact point of use, not merely trusted from ComputeWorkspaceLayout's own rounding
		// -- these two are the int64_t*-typed regions the finding measured landing on a 4-mod-8
		// address whenever max_chunk_budget was odd (wide_logits/rms_wide; embed_codes/final_codes
		// are int8_t*, logit_row is int32_t* and 8-byte alignment already implies its own
		// 4-byte requirement). Fires immediately, at the call that would otherwise silently read/
		// write through a misaligned int64_t* on any platform that faults on it.
		assert(reinterpret_cast<uintptr_t>(wide_logits) % alignof(int64_t) == 0);
		assert(reinterpret_cast<uintptr_t>(rms_wide) % alignof(int64_t) == 0);
	} else {
		embed_codes_fallback.assign(c.hidden_size, 0);
		final_codes_fallback.assign(c.hidden_size, 0);
		wide_logits_fallback.assign(static_cast<size_t>(c.vocab_size), 0);
		logit_row_fallback.assign(static_cast<size_t>(c.vocab_size), 0);
		embed_codes = embed_codes_fallback.data();
		final_codes = final_codes_fallback.data();
		wide_logits = wide_logits_fallback.data();
		logit_row = logit_row_fallback.data();
	}

	for (int32_t i = 0; i < n; ++i) {
		sslm_seq_s* seq = seqs[i];

		if (seq->ready_for_logits) {
			// This sequence's state.hidden_codes already holds a fully-computed final hidden
			// state (from sslm_prefill or sslm_seq_adopt_prefix, see sslm_seq_s's own comment)
			// -- no embed, no RunLayerLoop this call; jump straight to finishing the token
			// below. This is the ONE call that costs no layer work, matching
			// RunGreedyDecodeLoop's own fold of the prompt's last token into its generation
			// loop's first iteration (forward_sites.cpp).
			seq->ready_for_logits = false;
		} else {
			if (seq->state.layer_index == 0) {
				// A fresh token boundary -- embed the last produced/prior token first (design's
				// own decode_step signature carries no token-input parameter; this ABI's own
				// current_token tracking, sslm_seq_s's own header comment, is what supplies it),
				// exactly RunGreedyDecodeLoop's own RunWholeToken lambda's first act.
				if (seq->state.context_length >= static_cast<int64_t>(c.context_cap)) {
					out_tokens[i] = -1;
					return SSLM_CONTEXT_CAP_EXCEEDED;
				}
				superslm::CarriedScale embed_scale{};
				const superslm::SslmForwardStatus est = superslm::EmbedEntry(
				    seq->current_token, static_cast<int32_t>(c.vocab_size),
				    model->engine.embed_weights, c.hidden_size, model->engine.embed_site_constant,
				    embed_codes, &embed_scale);
				if (est != superslm::SslmForwardStatus::Ok) return MapForwardStatus(est);
				for (uint32_t k = 0; k < c.hidden_size; ++k) {
					seq->state.hidden_codes[k] = embed_codes[k];
				}
				seq->state.hidden_scale = embed_scale;
				seq->state.layer_index = 0;
			}

			std::vector<superslm::LayerWeights> layers_scratch;
			const superslm::LayerWeights* layers =
			    ResolveLayers(model, seq->adapter_handle, &layers_scratch);
			const superslm::SslmForwardStatus st = superslm::RunLayerLoop(
			    seq->state, layers, c.num_hidden_layers,
			    static_cast<uint32_t>(params->layer_budget), c.hidden_size, c.head_dim,
			    c.num_key_value_heads, c.intermediate_size, c.context_cap, model->view.rope_tables,
			    seq->kv_block, seq->block_size, /*site_prefix=*/{}, /*token_index=*/0, nullptr);
			if (st != superslm::SslmForwardStatus::Ok) return MapForwardStatus(st);

			if (seq->state.layer_index < c.num_hidden_layers) {
				// The requested layer_budget did not reach the end of this token -- pending,
				// resumable on the next call with layer_index carried forward (design's own
				// -1 "pending" sentinel, SslmDecodeStepStatus::produced_token, model.h).
				out_tokens[i] = -1;
				continue;
			}
		}

		// This call's own layer_budget completed the token (or the state was already complete,
		// ready_for_logits) -- finish it: final_norm, logits,
		// argmax, exactly RunGreedyDecodeLoop's own generation-loop body (forward_sites.cpp).
		superslm::CarriedScale final_scale{};
		// T-2139 closing round: rms_wide (carved from the workspace when supplied, nullptr
		// otherwise -- see this function's own carving block above) eliminates the ready_for_logits
		// path's own last disclosed allocation (RmsNormSite's internal `wide` scratch, design
		// commit 959336ad64's own recalibrated zero-allocation ruling for this specific path).
		superslm::SslmForwardStatus fst =
		    superslm::RmsNormSite(seq->state.hidden_codes, model->engine.final_norm_gain.data(),
		                           c.hidden_size, seq->state.hidden_scale,
		                           model->engine.final_norm_site_constant, final_codes,
		                           &final_scale, "final_norm", /*token_index=*/0,
		                           /*trace_hook_state=*/nullptr, rms_wide);
		if (fst != superslm::SslmForwardStatus::Ok) return MapForwardStatus(fst);

		fst = superslm::LogitsSite(final_codes, c.hidden_size, model->engine.head_weights,
		                            static_cast<int32_t>(c.vocab_size), wide_logits,
		                            logit_row);
		if (fst != superslm::SslmForwardStatus::Ok) return MapForwardStatus(fst);

		const int32_t produced =
		    superslm::ArgmaxLowestIndexTieBreak(logit_row, static_cast<size_t>(c.vocab_size));
		out_tokens[i] = produced;
		seq->current_token = produced;
		// forward_sites.h: "a sequence resting between whole tokens carries a marker at layer
		// 0" -- reset to the resting convention now that this token is complete.
		seq->state.layer_index = 0;
	}
	return SSLM_OK;
}

extern "C" sslm_status sslm_stats(sslm_model model, sslm_seq seq, sslm_stats_out* out) {
	if (!out) return SSLM_INVALID_ARGUMENT;
	if (!model || !seq) return SSLM_INVALID_ARGUMENT;
	// design Sec12 ("ceiling + actual work, forced-token count, cache state"): this arc builds
	// no schema/constrained-decoding surface (G5's own scope, declared-but-deferred here), so
	// forced_token_count is always 0 -- a real, correct answer for a sequence with no schema
	// bound (G5's own forcing mechanism has never run), not a placeholder. decode_step_ceiling/
	// decode_step_actual are both the model's own num_hidden_layers: this arc's own
	// sslm_decode_step always processes exactly one call's own layer_budget against the SAME
	// worst-case ceiling (a full token, num_hidden_layers), since no partial-work-skipping
	// mechanism (e.g. early-exit) exists in this arc's own scope to make ceiling and actual
	// diverge -- reported honestly as equal, not fabricated apart.
	out->decode_step_ceiling = static_cast<int64_t>(model->view.config.num_hidden_layers);
	out->decode_step_actual = static_cast<int64_t>(model->view.config.num_hidden_layers);
	out->forced_token_count = 0;
	out->kv_blocks_resident = 1;  // this sequence's own single, whole-sequence block (Sec7.2)
	return SSLM_OK;
}

// -----------------------------------------------------------------------------------------
// C5 -- save/restore (design Sec7.3): the blob format at that section's own field order,
// magic-per-version hard-reject (design Sec7.3's own corrected version-evolution strategy,
// GpuSeqBlobHeader's precedent). kv_block_count is always 1 (Sec7.2's ruled one-block-per-
// sequence unit) -- no per-blob variability there, unlike the pre-ruling page-count reading.
//
// `current_token` -- AMENDED IN (design commit 9e2995f4e7, the blob-amendment ruling). This
// build's own S-FREEZE run measured, by execution, that a sequence resting BETWEEN
// sslm_decode_step calls (produced a token, not yet embedded the next one) cannot have its
// pending-embed token id re-derived from layer_index/context_length alone -- restore's own
// ready_for_logits path recomputed the SAME prediction already produced before saving instead
// of the live sequence's true one-token-further continuation (measured: live continues with
// 315, the pre-amendment restore produced 0). One field closes it: `current_token` (int32_t,
// `-1` sentinel = not applicable -- every OTHER resting state, fresh-post-prefill or mid-token,
// is still correctly determined from layer_index/context_length alone, so no second field is
// needed). See sslm_seq_save/sslm_seq_restore below for the exact field position and
// sslm_seq_restore's own comment for how the sentinel and the ready_for_logits fallback compose.
//
// MAGIC NOTE (design commit 9e2995f4e7's own one-time exception, cited here so the exception is
// visible at the point it is taken, not only in the design doc): 'SSB1' is amended IN PLACE for
// this one field, not bumped to a new magic. This is a stated, one-time, pre-first-ship
// exception to the standing magic-per-version law immediately below -- no 'SSB1' blob has ever
// been committed or persisted anywhere in this tree (every one produced so far is a same-run,
// uncommitted smoke/example artifact, T-2139's own build log §15), so there is no already-
// shipped consumer's stored bytes this amendment could misread. The standing law resumes with
// no further exceptions the instant this corrected 'SSB1' ships -- every subsequent field-layout
// change, including one this small, gets a new, distinct magic.
// -----------------------------------------------------------------------------------------

namespace {

constexpr uint8_t kSeqBlobMagic[4] = {'S', 'S', 'B', '1'};
constexpr uint32_t kDfaWalkStateUnused = 0xFFFFFFFFu;
constexpr int32_t kSeqBlobNoCurrentToken = -1;

void WriteLE32(uint8_t* p, uint32_t v) {
	p[0] = static_cast<uint8_t>(v);
	p[1] = static_cast<uint8_t>(v >> 8);
	p[2] = static_cast<uint8_t>(v >> 16);
	p[3] = static_cast<uint8_t>(v >> 24);
}
void WriteLE64(uint8_t* p, uint64_t v) {
	for (int i = 0; i < 8; ++i) p[i] = static_cast<uint8_t>(v >> (8 * i));
}
uint32_t ReadLE32(const uint8_t* p) {
	return static_cast<uint32_t>(p[0]) | (static_cast<uint32_t>(p[1]) << 8) |
	       (static_cast<uint32_t>(p[2]) << 16) | (static_cast<uint32_t>(p[3]) << 24);
}
uint64_t ReadLE64(const uint8_t* p) {
	uint64_t v = 0;
	for (int i = 0; i < 8; ++i) v |= static_cast<uint64_t>(p[i]) << (8 * i);
	return v;
}

// Fixed-size prefix, magic through kv_saturation_count -- design Sec7.3's own field order
// (current_token AMENDED IN, design commit 9e2995f4e7 -- see this block's own top comment):
// magic(4) + model_hash(32) + kv_precision(4) + schema_name_hash(8) + dfa_walk_state(4) +
// adapter_binding_id(8) + context_length(8) + layer_index(4) + current_token(4) +
// hidden_scale(16) + kv_saturation_count(8) = 100 bytes, followed by the variable-length
// residual, kv_block_count(4), then kv_blocks.
constexpr size_t kSeqBlobFixedHeaderBytes = 100;

}  // namespace

extern "C" sslm_status sslm_seq_save(sslm_seq seq, void* buf, size_t* n) {
	if (!n) return SSLM_INVALID_ARGUMENT;
	if (!seq) return SSLM_INVALID_ARGUMENT;
	sslm_model_s* model = seq->model;
	const superslm::SslmModelConfig& c = model->view.config;
	// design Sec7.3: "residual_bytes ... zero-length when layer_index == 0" (§7's own "a
	// sequence resting between whole tokens carries a zero-length residual").
	const bool mid_token = seq->state.layer_index != 0;
	const size_t residual_len = mid_token ? static_cast<size_t>(c.hidden_size) : 0;
	const size_t required = kSeqBlobFixedHeaderBytes + residual_len + 4 + seq->block_size;
	if (!buf || *n < required) {
		// design Sec7.3: "the same two-call sizing convention sslm_seq_state_size already
		// establishes" -- *n set to the required size on this specific rejection.
		*n = required;
		return SSLM_BUFFER_TOO_SMALL;
	}

	uint8_t* p = static_cast<uint8_t*>(buf);
	size_t off = 0;
	std::memcpy(p + off, kSeqBlobMagic, 4);
	off += 4;
	const std::array<uint8_t, 32> hash = model->view.RawIntegrityHash();
	std::memcpy(p + off, hash.data(), 32);
	off += 32;
	WriteLE32(p + off, static_cast<uint32_t>(c.kv_precision));
	off += 4;
	WriteLE64(p + off, 0);  // schema_name_hash -- G5 scope, not built
	off += 8;
	WriteLE32(p + off, kDfaWalkStateUnused);  // dfa_walk_state -- G5 scope, not built
	off += 4;
	WriteLE64(p + off, 0);  // adapter_binding_id -- C6 scope, not built (no adapter can be bound)
	off += 8;
	WriteLE64(p + off, static_cast<uint64_t>(seq->state.context_length));
	off += 8;
	WriteLE32(p + off, seq->state.layer_index);
	off += 4;
	// current_token (design commit 9e2995f4e7): the pending-embed token id, written ONLY for the
	// one resting shape that actually needs it -- resting BETWEEN sslm_decode_step calls
	// (layer_index == 0, ready_for_logits == false, current_token >= 0: the exact state a live
	// sequence carries right after producing a token and before embedding the next one). The
	// sentinel (-1) is written for every other case, per the ruling's own list: fresh-post-
	// prefill/adopt (layer_index == 0, ready_for_logits == true -- no pending-embed id exists
	// yet) and mid-token (layer_index != 0 -- current_token plays no role in that path at all,
	// so its live value is not this field's concern).
	const bool has_pending_embed = (seq->state.layer_index == 0) && !seq->ready_for_logits;
	WriteLE32(p + off, has_pending_embed ? seq->current_token : kSeqBlobNoCurrentToken);
	off += 4;
	WriteLE64(p + off, static_cast<uint64_t>(seq->state.hidden_scale.m));
	off += 8;
	WriteLE64(p + off, static_cast<uint64_t>(seq->state.hidden_scale.e));
	off += 8;
	WriteLE64(p + off, seq->state.kv_saturation_count);
	off += 8;
	if (residual_len > 0) {
		std::memcpy(p + off, seq->hidden_codes_storage.data(), residual_len);
	}
	off += residual_len;
	WriteLE32(p + off, 1);  // kv_block_count -- always 1 (Sec7.2's ruled unit)
	off += 4;
	std::memcpy(p + off, seq->kv_block, seq->block_size);
	off += seq->block_size;

	*n = off;
	return SSLM_OK;
}

extern "C" sslm_status sslm_seq_restore(sslm_model model, sslm_kv_pool* pool, const void* buf,
                                         size_t n, sslm_seq* out) {
	if (!out) return SSLM_INVALID_ARGUMENT;
	*out = nullptr;
	if (!model || !pool || !*pool || !buf) return SSLM_INVALID_ARGUMENT;
	if (n < kSeqBlobFixedHeaderBytes + 4) return SSLM_INVALID_ARGUMENT;  // too small to hold even
	                                                                    // the fixed header + block
	                                                                    // count

	const uint8_t* p = static_cast<const uint8_t*>(buf);
	// design Sec7.3, corrected (Mendeleev audit 4.5): magic-per-version HARD REJECT -- checked
	// first, before any other field is trusted, exactly GpuSeqBlobHeader's own precedent
	// (superslm_gpu.cpp). A well-formed GPU-format ('SLM3') blob is rejected here on the magic
	// check alone, never mis-parsed as a CPU blob (design Sec10 dim7/dim9). M1 (Claude/Poirot/
	// 2c18dab-t2139-abi-build-review.md): a bad magic is a malformed/foreign-format BLOB, not a
	// model mismatch -- SSLM_INVALID_ARGUMENT states the actual cause (the blob itself is
	// unusable) rather than sending the caller to re-check which MODEL it bound.
	if (std::memcmp(p, kSeqBlobMagic, 4) != 0) return SSLM_INVALID_ARGUMENT;

	std::array<uint8_t, 32> saved_hash{};
	std::memcpy(saved_hash.data(), p + 4, 32);
	const std::array<uint8_t, 32> current_hash = model->view.RawIntegrityHash();
	if (saved_hash != current_hash) return SSLM_RESTORE_MODEL_MISMATCH;

	const uint32_t saved_kv_precision = ReadLE32(p + 36);
	if (saved_kv_precision != static_cast<uint32_t>(model->view.config.kv_precision)) {
		return SSLM_RESTORE_KV_MISMATCH;
	}

	const int64_t context_length = static_cast<int64_t>(ReadLE64(p + 60));
	const uint32_t layer_index = ReadLE32(p + 68);
	const int32_t saved_current_token = static_cast<int32_t>(ReadLE32(p + 72));  // design commit
	                                                                              // 9e2995f4e7
	const int64_t hidden_scale_m = static_cast<int64_t>(ReadLE64(p + 76));
	const int64_t hidden_scale_e = static_cast<int64_t>(ReadLE64(p + 84));
	const uint64_t kv_saturation_count = ReadLE64(p + 92);

	const superslm::SslmModelConfig& c = model->view.config;
	const bool mid_token = layer_index != 0;
	const size_t residual_len = mid_token ? static_cast<size_t>(c.hidden_size) : 0;
	if (n < kSeqBlobFixedHeaderBytes + residual_len + 4) return SSLM_INVALID_ARGUMENT;
	const uint8_t* residual_ptr = p + kSeqBlobFixedHeaderBytes;
	const uint32_t kv_block_count = ReadLE32(residual_ptr + residual_len);
	if (kv_block_count != 1) {
		// design Sec7.2's ruled unit: a saved sequence always carries exactly one block. A
		// value other than 1 is a structurally malformed (or pre-ruling-format) blob.
		return SSLM_INVALID_ARGUMENT;
	}
	const size_t block_size = sslm_kv_block_size(model);
	if (block_size == 0) return SSLM_ARTIFACT_REJECTED;
	if (n < kSeqBlobFixedHeaderBytes + residual_len + 4 + block_size) {
		return SSLM_INVALID_ARGUMENT;
	}
	const uint8_t* kv_blocks_ptr = residual_ptr + residual_len + 4;

	sslm_kv_pool_s* pool_ptr = *pool;
	// C2 (Claude/Poirot/2c18dab-t2139-abi-build-review.md): THE actual overflow site -- checked
	// BEFORE DrawBlock, so a mismatched pool is rejected before any block is even drawn (never
	// mind copied into). Without this, `block_size` above (from sslm_kv_block_size(model), the
	// MODEL's own footprint) and `pool_ptr->block_size` (the POOL's own per-block stride) can
	// differ -- a well-formed blob/model pair against a pool built for a smaller-footprint model
	// then memcpy's the model's block_size into a destination sized by the pool's own smaller
	// one, overrunning the pool's buffer by the difference (measured, pre-fix: 268,435,456 bytes
	// restoring a 1.5B sequence into a 0.5B-sized pool). Checking `pool_ptr->model != model` here
	// is exactly equivalent to checking `block_size != pool_ptr->block_size` (both are pure,
	// deterministic functions of the model) but states the real invariant directly instead of by
	// coincidence.
	if (pool_ptr->model != model) return SSLM_INVALID_ARGUMENT;
	uint32_t block_index = 0;
	const sslm_status draw_st = DrawBlock(pool_ptr, &block_index);
	if (draw_st != SSLM_OK) return draw_st;

	// M4/N3 (Claude/Poirot/2c18dab-t2139-abi-build-review.md): the handle is allocated BEFORE the
	// block is copied into and BEFORE any further work, so an early return path never needs to
	// unwind a drawn block -- the allocation-failure path (immediately below) is the only way out
	// before the block is committed to `h`, and it now returns the block first (M4) and reports
	// SSLM_ALLOCATION_FAILED rather than throwing across this extern "C" boundary (N3).
	auto* h = new (std::nothrow) sslm_seq_s();
	if (!h) {
		ReturnBlock(pool_ptr, block_index,
		            static_cast<uint8_t*>(pool_ptr->buf) +
		                static_cast<size_t>(block_index) * pool_ptr->block_size,
		            pool_ptr->block_size);
		return SSLM_ALLOCATION_FAILED;
	}
	h->model = model;
	h->pool = pool_ptr;
	h->block_index = block_index;
	h->kv_block =
	    static_cast<uint8_t*>(pool_ptr->buf) + static_cast<size_t>(block_index) * pool_ptr->block_size;
	h->block_size = pool_ptr->block_size;
	std::memcpy(h->kv_block, kv_blocks_ptr, block_size);
	h->hidden_codes_storage.assign(c.hidden_size, 0);
	if (residual_len > 0) {
		std::memcpy(h->hidden_codes_storage.data(), residual_ptr, residual_len);
	}
	h->state.hidden_codes = h->hidden_codes_storage.data();
	h->state.hidden_scale = superslm::CarriedScale{hidden_scale_m, hidden_scale_e};
	h->state.layer_index = layer_index;
	h->state.kv_saturation_count = kv_saturation_count;
	h->state.context_length = context_length;
	// current_token/ready_for_logits reconstruction -- CLOSED (design commit 9e2995f4e7, the
	// blob-amendment ruling; see this whole C5 block's own top comment for the finding this
	// resolves). `saved_current_token` (read above, `-1` sentinel = not applicable) is now the
	// blob's own explicit source of truth, not a re-derivation:
	//   - saved_current_token >= 0 (the resting-BETWEEN-decode-steps case): the residual was
	//     ALREADY consumed once by the live sequence to produce this exact id -- restore it
	//     verbatim as current_token, ready_for_logits = false, so the next sslm_decode_step call
	//     takes the ordinary layer_index == 0 embed-current_token path, exactly matching what
	//     the live sequence itself would do next. This is the case that used to diverge (measured:
	//     live continues with 315, the pre-amendment restore produced 0) and is what design
	//     Sec10 dim 9's new cell (tools/t2139_dim9_current_token_pin.cpp) proves closed.
	//   - saved_current_token == -1, layer_index == 0, context_length > 0 (fresh-post-prefill/
	//     adopt): the residual has never been consumed -- ready_for_logits = true, unchanged
	//     from before this fold.
	//   - saved_current_token == -1 otherwise (mid-token, or a genuinely fresh/empty sequence):
	//     current_token stays -1, ready_for_logits stays false -- unchanged from before this
	//     fold (mid-token resumes via layer_index alone; a fresh/empty sequence has nothing to
	//     decode from until prefilled, exactly as sslm_decode_step's own validation already
	//     requires).
	if (saved_current_token >= 0) {
		h->current_token = saved_current_token;
		h->ready_for_logits = false;
	} else {
		h->current_token = -1;
		h->ready_for_logits = (layer_index == 0 && context_length > 0);
	}
	model->live_refs.fetch_add(1, std::memory_order_acq_rel);
	*out = h;
	return SSLM_OK;
}

// -----------------------------------------------------------------------------------------
// C6 -- adapter lifecycle (design Sec9 C6, Sec4: "S-LoRA-serial's own outstanding ABI debt"):
// wraps the already-proven V5 delta kernel and converter adapter mode via
// include/superslm/adapter_marshal.h, never reimplements it.
// -----------------------------------------------------------------------------------------

extern "C" sslm_status sslm_adapter_map(const void* data, size_t size, sslm_model base,
                                         sslm_adapter* out) {
	if (!out) return SSLM_INVALID_ARGUMENT;
	*out = nullptr;
	if (!data || !base) return SSLM_INVALID_ARGUMENT;

	superslm::SslmModelView adapter_view;
	std::string err;
	// N3 -- see sslm_model_map's own identical comment (Claude/Poirot/
	// 2c18dab-t2139-abi-build-review.md Sec6.3): SslmModel::Load caught explicitly rather than
	// left to cross this extern "C" boundary as an exception.
	superslm::SslmModelStatus load_st;
	try {
		load_st = superslm::SslmModel::Load(static_cast<const uint8_t*>(data), size, adapter_view,
		                                     &err);
	} catch (const std::bad_alloc&) {
		return SSLM_ALLOCATION_FAILED;
	}
	if (load_st != superslm::SslmModelStatus::Ok) return SSLM_ARTIFACT_REJECTED;

	superslm_adapter::BaseModelGeometry base_geom;
	const superslm::SslmModelConfig& bc = base->view.config;
	base_geom.num_hidden_layers = bc.num_hidden_layers;
	base_geom.hidden_size = bc.hidden_size;
	base_geom.intermediate_size = bc.intermediate_size;
	base_geom.kv_hidden_size = static_cast<uint64_t>(bc.num_key_value_heads) * bc.head_dim;
	base_geom.base_artifact_hash = base->view.RawIntegrityHash();

	auto* h = new (std::nothrow) sslm_adapter_s();
	// N3 -- see sslm_workspace_create's own identical comment.
	if (!h) return SSLM_ALLOCATION_FAILED;
	h->base = base;
	h->artifact_bytes = size;
	// AdapterHandle::view_ is the long-lived owner of the adapter's own tensor bytes --
	// populated BEFORE PopulateAdapterFromView runs against it, so layer_adapters' own pointers
	// (adapter_marshal.h) point into this handle's own member, never the temporary local
	// `adapter_view` above (which would leave them dangling the instant this function returns).
	h->handle.view_ = std::move(adapter_view);
	// M3 (Claude/Poirot/2c18dab-t2139-abi-build-review.md): no separate local -- populates
	// h->handle.layer_adapters directly, matching the comment above.
	const superslm_adapter::AdapterLoadStatus populate_st = superslm_adapter::PopulateAdapterFromView(
	    h->handle.view_, base_geom, h->handle.meta, h->handle.layer_adapters, &err);
	if (populate_st == superslm_adapter::AdapterLoadStatus::BaseHashMismatch) {
		delete h;
		return SSLM_ADAPTER_MODEL_MISMATCH;
	}
	if (populate_st != superslm_adapter::AdapterLoadStatus::Ok) {
		// Every other rejection (malformed ADP1, missing DeltaFoldScales/UFoldScales, a
		// dimension/shape mismatch) is a structurally-broken-artifact class this design's own
		// Sec6 taxonomy assigns no dedicated status to -- the same "no more specific status"
		// disposition sslm_gpu_adapter_map already uses for the identical rejection family
		// (src/gpu/gpu_1p0.cpp), substituting this ABI's own closest artifact/content rejection.
		delete h;
		return SSLM_ARTIFACT_REJECTED;
	}

	base->live_refs.fetch_add(1, std::memory_order_acq_rel);
	*out = h;
	return SSLM_OK;
}

extern "C" sslm_status sslm_adapter_release(sslm_adapter adapter) {
	if (!adapter) return SSLM_INVALID_ARGUMENT;
	if (adapter->live_refs.load(std::memory_order_acquire) != 0) {
		return SSLM_ADAPTER_HAS_LIVE_SEQUENCES;
	}
	adapter->base->live_refs.fetch_sub(1, std::memory_order_acq_rel);
	delete adapter;
	return SSLM_OK;
}

extern "C" size_t sslm_adapter_residency(sslm_adapter adapter) {
	if (!adapter) return 0;
	return adapter->artifact_bytes;
}

// design Sec6: sslm_seq_set_adapter against a non-zero mid-token residual marker is
// SSLM_ADAPTER_SWAP_MIDTOKEN_REJECTED -- swapping the delta kernel mid-layer-loop would compose
// two different adapters' deltas across one token's own layer range, an undefined mixture this
// design forecloses by rejecting outright rather than defining. `adapter == nullptr` unbinds
// (LayerWeights::adapter's own existing NULL-adapter convention, forward_sites.h).
extern "C" sslm_status sslm_seq_set_adapter(sslm_seq seq, sslm_adapter adapter) {
	if (!seq) return SSLM_INVALID_ARGUMENT;
	if (seq->state.layer_index != 0) return SSLM_ADAPTER_SWAP_MIDTOKEN_REJECTED;
	if (adapter && adapter->base != seq->model) return SSLM_ADAPTER_MODEL_MISMATCH;

	if (seq->adapter_handle == adapter) return SSLM_OK;  // idempotent no-op
	if (seq->adapter_handle) {
		seq->adapter_handle->live_refs.fetch_sub(1, std::memory_order_acq_rel);
	}
	if (adapter) {
		adapter->live_refs.fetch_add(1, std::memory_order_acq_rel);
	}
	seq->adapter_handle = adapter;
	return SSLM_OK;
}

// -----------------------------------------------------------------------------------------
// C7 -- text I/O (design Sec9 C7, settled build-now by D-SLM3452): wraps
// TokenizerView::Encode/Decode (already-shipped, whole-buffer C++ calls, include/superslm/
// tokenizer.h). sslm_detokenize_stream adds the incremental-safety state Forge W4 names, not
// present in TokenizerView itself.
// -----------------------------------------------------------------------------------------

extern "C" sslm_status sslm_tokenize(sslm_model model, const char* utf8, int32_t* tokens,
                                      int32_t* n) {
	if (!n) return SSLM_INVALID_ARGUMENT;
	if (!model || !utf8) {
		if (n) *n = 0;
		return SSLM_INVALID_ARGUMENT;
	}
	if (!model->view.has_tokenizer) {
		*n = 0;
		return SSLM_ARTIFACT_REJECTED;
	}
	// TokenizerView::Encode throws only std::bad_alloc (tokenizer.h's own contract) -- this call
	// site does not catch it, matching sslm_model_map's own house-convention disposition
	// (bad_alloc_wrap.h) for the identical reason: sslm_status carries no resource-exhaustion
	// member.
	const std::vector<int32_t> ids = model->view.tokenizer.Encode(std::string_view(utf8));
	if (!tokens || *n < static_cast<int32_t>(ids.size())) {
		// design's own two-call sizing convention (Sec7.3's own precedent, extended here):
		// *n set to the required count on this specific rejection.
		*n = static_cast<int32_t>(ids.size());
		return SSLM_BUFFER_TOO_SMALL;
	}
	std::copy(ids.begin(), ids.end(), tokens);
	*n = static_cast<int32_t>(ids.size());
	return SSLM_OK;
}

namespace {

// Forge W4's own incremental-safety obligation: how many of `s`'s own TRAILING bytes form an
// INCOMPLETE UTF-8 multi-byte sequence (0..3) -- these are held back (sslm_detok_state's own
// pending_bytes/pending_count, design Sec7.4) rather than emitted, so a caller never sees a
// multi-byte codepoint split across two sslm_detokenize_stream calls. A malformed (not merely
// incomplete) lead byte is NOT a hold-back case -- it is the tokenizer's own already-established
// "invalid sequences pass through as replacement chars" policy (design Sec10, inherited
// unchanged), which TokenizerView::Decode has already applied by the time this function sees the
// bytes; this function's own job is narrower: distinguish "this tail is genuinely incomplete, not
// yet decodable" from "this tail is already a complete (possibly-replacement-char) sequence."
size_t CountIncompleteTrailingUtf8(const std::string& s) {
	const size_t len = s.size();
	if (len == 0) return 0;
	size_t lead_pos = len;
	size_t back = 0;
	while (back < 3 && back < len) {
		const uint8_t b = static_cast<uint8_t>(s[len - 1 - back]);
		if ((b & 0xC0) == 0x80) {
			++back;
			continue;
		}
		lead_pos = len - 1 - back;
		break;
	}
	if (lead_pos == len) return 0;  // 3 continuation bytes scanned with no lead in range, or
	                                 // len <= 3 continuation bytes -- treat as complete.
	const uint8_t lead = static_cast<uint8_t>(s[lead_pos]);
	size_t expected_len;
	if ((lead & 0x80) == 0x00) {
		expected_len = 1;
	} else if ((lead & 0xE0) == 0xC0) {
		expected_len = 2;
	} else if ((lead & 0xF0) == 0xE0) {
		expected_len = 3;
	} else if ((lead & 0xF8) == 0xF0) {
		expected_len = 4;
	} else {
		return 0;  // not a valid multi-byte lead -- the tokenizer's own replacement-char policy,
		           // not a hold-back case.
	}
	const size_t have_len = len - lead_pos;
	return have_len < expected_len ? have_len : 0;
}

}  // namespace

extern "C" sslm_status sslm_detokenize_stream(sslm_model model, sslm_detok_state* state,
                                               const int32_t* tokens, int32_t n, char* utf8,
                                               int32_t* out_n) {
	if (!out_n) return SSLM_INVALID_ARGUMENT;
	if (!model || !state || n < 0 || (n > 0 && !tokens)) {
		if (out_n) *out_n = 0;
		return SSLM_INVALID_ARGUMENT;
	}
	// S4 (Claude/Poirot/2c18dab-t2139-abi-build-review.md): `*out_n` is the caller's buffer
	// capacity on entry -- a negative value must be rejected here, before the later
	// `static_cast<size_t>(*out_n) < emit_len` guard, which casts a negative capacity to
	// SIZE_MAX and lets it silently pass, exactly the way sslm_tokenize's own `*n < 0` guard
	// (below, its sibling verb) already does for its own capacity parameter.
	if (*out_n < 0) {
		*out_n = 0;
		return SSLM_INVALID_ARGUMENT;
	}
	if (state->pending_count > 3) {
		// design Sec7.4: pending_count is in [0, 3] by this struct's own invariant -- a caller-
		// supplied value outside that domain is hostile input, not a state this function can
		// resume from.
		*out_n = 0;
		return SSLM_INVALID_ARGUMENT;
	}
	if (!model->view.has_tokenizer) {
		*out_n = 0;
		return SSLM_ARTIFACT_REJECTED;
	}

	// design commit 212de7742c (the padded-vocabulary ruling, companion rule, Sec6/Sec7.4):
	// every input id checked against tok_vocab BEFORE any text is looked up or emitted for it,
	// and before `state` is touched -- the whole call is atomic, not per-id, matching every
	// other rejection in this design's own "reject leaves output/state unperturbed" contract.
	// tok_vocab <= cfg_vocab always holds for an artifact that reached this call (the loosened
	// ValidateTokenizerVocabSizeJoin, src/model.cpp, already enforced it at sslm_model_map time)
	// -- three disjoint bands: id < tok_vocab (real tokenizer entry, proceeds normally); id in
	// [tok_vocab, cfg_vocab) (a legal decode-OUTPUT id with no tokenizer text -- padding row,
	// SSLM_TOKEN_ID_UNMAPPED); id < 0 or id >= cfg_vocab (never a legal decode output at all --
	// plain SSLM_INVALID_ARGUMENT, distinct from both the malformed-UTF-8 policy and the
	// unmapped-padding case).
	const int32_t tok_vocab = model->view.tokenizer.VocabSize();
	const int64_t cfg_vocab = static_cast<int64_t>(model->view.config.vocab_size);
	for (int32_t i = 0; i < n; ++i) {
		const int32_t id = tokens[i];
		if (id < 0 || static_cast<int64_t>(id) >= cfg_vocab) {
			*out_n = 0;
			return SSLM_INVALID_ARGUMENT;
		}
		if (id >= tok_vocab) {
			*out_n = 0;
			return SSLM_TOKEN_ID_UNMAPPED;
		}
	}

	const std::vector<int32_t> ids(tokens, tokens + n);
	const std::string decoded = model->view.tokenizer.Decode(ids);

	std::string combined;
	combined.reserve(static_cast<size_t>(state->pending_count) + decoded.size());
	combined.append(reinterpret_cast<const char*>(state->pending_bytes), state->pending_count);
	combined.append(decoded);

	const size_t hold = CountIncompleteTrailingUtf8(combined);
	const size_t emit_len = combined.size() - hold;

	if (!utf8 || static_cast<size_t>(*out_n) < emit_len) {
		*out_n = static_cast<int32_t>(emit_len);
		return SSLM_BUFFER_TOO_SMALL;
	}
	if (emit_len > 0) std::memcpy(utf8, combined.data(), emit_len);
	*out_n = static_cast<int32_t>(emit_len);

	state->pending_count = static_cast<uint8_t>(hold);
	for (size_t i = 0; i < hold; ++i) {
		state->pending_bytes[i] = static_cast<uint8_t>(combined[emit_len + i]);
	}
	return SSLM_OK;
}

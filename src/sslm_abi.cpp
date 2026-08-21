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
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

#include "superslm/adapter_marshal.h"
#include "superslm/forward_sites.h"
#include "superslm/layer_marshal.h"
#include "superslm/model.h"
#include "superslm/schema_masks.h"
#include "superslm/sslm_damped_greedy.h"  // T-2199 Phase A/C: AntiLmState, DampedGreedyScoreAndArgmax
#include "superslm/sslm_phaseD.h"         // T-2199 Phase D: ValidateDampedGreedyParams

#include "bad_alloc_wrap.h"  // N3 pin (Claude/Poirot/2c18dab-t2139-abi-build-review.md Sec6.3):
                              // superslm::internal::MaybeThrowInjectedBadAllocFault(), the same
                              // test-only fault-injection seam S-HARDEN-7's own population already
                              // uses -- consulted once, at sslm_model_map's own Load call, so a
                              // test can force a genuine bad_alloc through this file's own new
                              // catch-and-return path without needing an actual OOM condition.

// G5 (design Sec5/Sec13.4, T-2132): the DFA-walk-state "no schema bound" sentinel -- moved up
// here (out of the save/restore-local anonymous namespace it originated in, below) so it is
// visible to every G5 verb, not only save/restore. `kMaxSchemaStates` (schema_masks.h) is fixed
// well below this value, so it is never a legitimate state id (design Sec13.4).
namespace {
constexpr uint32_t kDfaWalkStateUnused = 0xFFFFFFFFu;
}  // namespace

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

// G5 (T-2119 design Sec5, T-2132 build): one entry of sslm_model_s::schema_handles below. No
// map/release verb (design Sec5) -- these live exactly as long as the owning model, allocated
// once at sslm_model_map time, never individually created/destroyed. `index` is this schema's
// own row in the model's SchemaMasksTable (superslm::schema_masks.h).
struct sslm_schema_s {
	sslm_model_s* model = nullptr;
	uint32_t index = 0;
};

// C2. Owns the loaded model view and (C4) the resolved engine cache. `live_refs` is the D-SLM32
// lifecycle-guard bookkeeping design Sec9's own C2 gate names ("unmap-while-live is rejected") --
// incremented/decremented by C3's sslm_seq_create/_release, sslm_prefix_begin/_release, and (C6,
// not built) sslm_adapter_map.
//
// G5-2 (T-2132): `schemas`/`schema_handles` -- built once at sslm_model_map time (below), from
// this model's own SchemaMasks/SCM1 section if present (design Sec13.1: absence is a valid,
// unconstrained-only artifact, exactly today's pre-G5 behavior -- `schemas.Count() == 0`).
// `schema_handles[i]` is the stable, never-reallocated `sslm_schema` handle
// `sslm_schema_lookup`/`sslm_schema_name` hand out for `schemas.ByIndex(i)`/`ByName(...)`; both
// vectors are sized once, at construction, and never resized afterward, so every handle this
// model ever returns stays valid (and stably addressed) for the model's own whole lifetime.
struct sslm_model_s {
	superslm::SslmModelView view;
	std::atomic<uint32_t> live_refs{0};
	sslm_model_engine_cache engine;
	superslm::SchemaMasksTable schemas;
	std::vector<sslm_schema_s> schema_handles;
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
	// G5 (design Sec5, T-2132): "prefix construction inherits the same discriminator" -- a
	// prefix's own recorded schema binding and DFA-walk-state, set by sslm_prefix_set_schema
	// and advanced by sslm_prefix_prefill's own SSLM_SPAN_SCHEMA_CONTENT path, consulted by
	// sslm_seq_adopt_prefix's own schema-compatibility rule. kDfaWalkStateUnused == no schema
	// ever bound; 0 == a schema IS bound but the walk is still at its start state (no
	// SSLM_SPAN_SCHEMA_CONTENT content prefilled yet). Both read as "no real schema-content
	// progress" -- "compatible with any [adopting sequence's] binding" per design Sec5's own
	// "prompt-only prefix" framing -- only a walk-state that is neither of these two values is
	// real, transferable progress (the `prefix_has_real_progress` check inline in
	// sslm_seq_adopt_prefix, below, is where this reasoning is applied).
	sslm_schema bound_schema = nullptr;
	uint32_t dfa_walk_state = kDfaWalkStateUnused;
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
	// G5 (design Sec5, T-2132): see sslm_prefix_s's own identical fields for the shared
	// semantics (kDfaWalkStateUnused == no schema bound; 0 == bound, at start). Reset by
	// sslm_seq_reset (walk-state only, binding preserved) and round-tripped through
	// sslm_seq_save/restore (Sec13.4). `forced_token_count` (design Sec6 G5-3/Sec7 dim7):
	// sslm_stats' own "actual forced-position count, not the ceiling" -- incremented once per
	// token admitted through a SSLM_SPAN_SCHEMA_CONTENT prefill call (this build's own G5-3
	// scope: the host/runtime issues the forced span explicitly; no autonomous jump-forward
	// scheduling heuristic is built, design Sec6 G5-3's own "no separate forced-chain-detection
	// mechanism is required" text).
	sslm_schema bound_schema = nullptr;
	uint32_t dfa_walk_state = kDfaWalkStateUnused;
	int64_t forced_token_count = 0;
	// T-2199 Phase D2: this sequence's own damped-greedy anti-LM state (plan Sec7.2, "a
	// warm-object class, not fresh-per-call") -- created lazily on the first decode_step call
	// made with mode=SSLM_DECODE_MODE_DAMPED_GREEDY (sslm_decode_stepImpl, below), destroyed in
	// sslm_seq_release. Recreated (destroy + fresh AntiLmCreate) if a later call names a
	// DIFFERENT anti_lm_max_order than the instance already live -- a defined, if unusual,
	// caller behavior (switching n mid-generation), never a silent reuse of a mismatched order.
	superslm::AntiLmState* damped_greedy_antilm = nullptr;
	int32_t damped_greedy_antilm_order = 0;
	// T-2199 Phase D3 fix (Sec9 dim3, GATE, D-SLM3719 -- conductor's routed finding, real crash
	// 0xC0000005 reproduced 5/5 against a real-checkpoint fixture): `sslm_seq_release` frees this
	// WHOLE object (`delete seq`, below) with no coordination against a batched
	// `sslm_decode_stepImpl` call that may still be mid-flight touching the SAME sequence pointer
	// (the exact "teardown races an in-flight step over this sequence's own state" construction
	// `TestD3_TeardownDuringFlight...` commissions). Before Phase D, per-sequence processing
	// touched only a handful of already-owned fields per call, so this race window existed but was
	// too narrow to reproduce in practice; damped-greedy mode's own lazy `AntiLmCreate`/`AntiLmDestroy`
	// (a heap allocation/deallocation inside the per-sequence critical section) widens it enough
	// to fire close to every trial. `lifecycle_mutex` closes it as an acquire-then-release BARRIER,
	// not a lock held across the object's own lifetime: `sslm_decode_stepImpl` holds it for the
	// FULL per-sequence body (every mode, not only damped-greedy -- the hazard is general, the
	// widened window is what made it observable), and `sslm_seq_release` acquires and immediately
	// releases it before `delete seq` -- if a decode step is mid-flight on this sequence,
	// `sslm_seq_release` blocks until that step's own per-sequence section completes, THEN frees
	// the object; if no step is in flight, the acquire is uncontended and costs one uncontended
	// lock/unlock pair.
	mutable std::mutex lifecycle_mutex;
	// NO `released` flag -- see sslm_prefix_s's own comment on why (S5): double-release and any
	// other use of an already-released handle is caller UB, not a state this ABI pretend-guards.
	// `lifecycle_mutex` above is a narrower, additive guarantee against ONE specific race (a
	// release concurrent with an in-flight step naming the SAME live sequence), not a general
	// double-release/use-after-release guard -- that remains caller UB exactly as documented here.
};

// T-2199 Phase D3 fix, corrected a SECOND time (Sec9 dim3, GATE, D-SLM3719): the first two
// corrections (moving the batch lock earlier, then earlier still) both STILL crashed under
// repro against a real checkpoint. Root cause, found via an SEH-wrapped/instrumented repro,
// 2026-08-20: `lifecycle_mutex` embedded IN `sslm_seq_s` cannot, by itself, guard against the
// object's OWN deletion, because locking it requires the object to still exist. If
// `sslm_seq_release` runs to completion (including `delete seq`) before `sslm_decode_stepImpl`'s
// own lock-acquisition loop ever reaches that pointer -- entirely possible, since the release
// thread does no forward-pass work and can win the race outright -- that loop's own
// `s->lifecycle_mutex` dereferences already-freed memory. A registry OUTSIDE any individual
// sequence, alive for the whole process, closes this: every touch of a sequence pointer that
// could race a concurrent release -- both the decode side's liveness-check-and-lock and the
// release side's remove-then-barrier-then-delete -- goes through `g_seq_registry_mutex` first,
// so the two operations that must never interleave are themselves mutually exclusive, and a
// decode call PROVES a sequence is still live (a lookup, not a dereference) before it ever
// touches it. `static`, not a class member: process-wide is a deliberately coarser scope than
// per-model (this file's only registry of live `sslm_seq_s*` handles), acceptable because this
// section is held only for a lookup plus a small number of already-cheap mutex acquisitions, not
// for a call's actual forward-pass work.
static std::mutex g_seq_registry_mutex;
static std::unordered_set<sslm_seq_s*> g_live_seqs;

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

// P1 (Claude/Poirot/2c18dab-t2139-abi-build-review.md Sec7.3, the third confirmation pass):
// N3's own remedy converted every place the COMPILER could see an explicit `throw
// std::bad_alloc()`; it did not convert every place one can actually happen -- std::vector/
// std::string operations (`.assign`, `.resize`, `.reserve`, construction, `TokenizerView::
// Encode`/`Decode`'s own "throws only std::bad_alloc" contract) can all throw without any
// `throw` keyword visible in THIS file, so C4297 (the diagnostic that found the class) has
// nothing to fire on and stayed silent at exactly the sites this finding names.
//
// SWEEP METHOD (stated per the coordinator's own instruction): every `extern "C"` verb in this
// file (29 total, `tools/t2139_count_abi_verbs.sh`'s own count) was read in full, and every
// operation in its own body AND in every anonymous-namespace helper it calls directly (one level
// -- `PrefillWholeTokens`, `BuildEngineCache`, `ResolveLayers`) was classified: (a)
// `new (std::nothrow)` -- never throws, already returns nullptr on failure, no wrap needed; (b)
// POD field copies, `std::memcpy`/`std::memset`/`std::fill` over an ALREADY-sized destination,
// arithmetic, `noexcept`-declared calls (`RawIntegrityHash`) -- cannot throw, no wrap needed; (c)
// `std::vector`/`std::string` construction, `.assign`/`.resize`/`.reserve`/`.push_back`, or a
// call into another TU with a documented "throws only std::bad_alloc" contract
// (`TokenizerView::Encode`/`Decode`, `tokenizer.h`; `SslmModel::Load`, already caught) -- CAN
// throw, needs a boundary catch. Verbs found in class (c), beyond the three the finding named by
// name: `sslm_kv_pool_create` (`free_list.reserve`/`push_back`), `sslm_prefix_begin`/
// `sslm_seq_create`/`sslm_seq_restore` (`hidden_codes_storage.assign`), `PrefillWholeTokens`
// (shared by `sslm_prefix_prefill`/`sslm_prefill` -- its own `embed_codes` fallback `.assign`
// and `ResolveLayers`' own per-call `layers_scratch` vector copy when an adapter is bound),
// `sslm_decode_step` (the identical fallback-vector class, plus its own `layers_scratch` copy),
// `sslm_adapter_map` (`PopulateAdapterFromView`, `adapter_marshal.h` -- allocates the adapter's
// own per-layer delta arrays). Every one of these, plus the three named sites
// (`sslm_tokenize`/`sslm_detokenize_stream`/`sslm_model_map`'s own `BuildEngineCache` call), is
// now wrapped below. Verbs NOT wrapped (class (a)/(b) only, verified by the same reading):
// `sslm_workspace_size`/`sslm_kv_block_size`/`sslm_kv_pool_overhead_size`/`sslm_seq_state_size`
// (pure arithmetic), `sslm_workspace_create`/`sslm_workspace_destroy`/`sslm_kv_pool_destroy`/
// `sslm_model_unmap`/`sslm_prefix_freeze`/`sslm_prefix_release`/`sslm_seq_release`/
// `sslm_seq_reset`/`sslm_seq_adopt_prefix`/`sslm_stats`/`sslm_seq_save`/`sslm_adapter_release`/
// `sslm_adapter_residency`/`sslm_seq_set_adapter` (no vector mutation, no documented-throwing
// callee).
//
// One general helper, not sixteen ad hoc try/catch blocks with the same three lines each: `fn`'s
// own return value passes through unchanged on success. This IS the extern "C" boundary that
// must never let a C++ exception cross it (undefined behavior under this codebase's own /EHc
// build flags, MSVC's own C4297 diagnostic) -- so every exception is caught here, never rethrown,
// but NOT every exception narrows to the same status.
//
// FOLD RULING on the third confirmation pass's F2 (Claude/Vitruvius/
// t2133-layer1-c-abi-design-2026-08-16.md Sec6, design commit dated 2026-08-17; D-SLM3462):
// this comment previously claimed `catch (const std::exception&)` meant "nothing past this point
// may propagate as a C++ exception regardless of its concrete type" -- false in both directions,
// proven by a compiled probe (F2): a throw NOT derived from `std::exception` (a custom type, or a
// non-class throw like `throw int`) crossed the boundary as real, observed undefined behavior,
// while every `std::exception` subtype that is NOT an allocation failure (`std::out_of_range`,
// `std::logic_error`, ...) was mis-attributed as SSLM_ALLOCATION_FAILED -- contradicting design
// Sec6's own ratified meaning for that status ("one dedicated cause (process-level exhaustion),
// never a wrapper for causes that already have their own status") and the "What this design does
// NOT add" text ("a SSLM_DEVICE_LOST-shaped catch-all"). The ruling narrows the shape to satisfy
// both constraints with one helper:
//   - `catch (const std::bad_alloc&)` and `catch (const std::length_error&)` -- the two standard
//     exceptions that mean "the requested allocation/size cannot be honored," the only causes
//     this family is defined to cover -- return SSLM_ALLOCATION_FAILED, unchanged.
//   - A final `catch (...)` -- the true, unconditional boundary, catching every remaining
//     exception type, INCLUDING non-`std::exception`-derived ones and non-class throws -- returns
//     SSLM_ARTIFACT_REJECTED. No new ordinal is minted: SSLM_ARTIFACT_REJECTED is the existing
//     family for "this call cannot be honored for an internal-rejection cause with no dedicated
//     status of its own," the same mapping MapForwardStatus (below, this file) already
//     establishes for every SslmForwardStatus cause this ABI's own closed taxonomy does not
//     separately enumerate, extended here to the identical `catch (...)`-boundary shape rather
//     than inventing a second mechanism for the same decision.
//
// SCOPE (FOLD RULING, second pass, D-SLM3464): the two rules above hold for THIS helper --
// `CatchAllocationFailure` itself, and every call site that goes through it (`BuildEngineCache`'s
// own wrap inside `sslm_model_map`, and every sizing/construction verb's own wrap). They do NOT
// describe `sslm_model_map`'s or `sslm_adapter_map`'s own bare `try { SslmModel::Load(...); }
// catch (...) { ... }` blocks (this file, their own definitions) -- those two are fronted by
// `SslmModel::Load`'s public entry, `internal::WrapBadAllocContract` (src/bad_alloc_wrap.h),
// which narrows every non-bad_alloc `std::exception` to `std::bad_alloc` before either of THOSE
// try/catch blocks ever runs. Their own delivered contract is narrower and documented at their
// own definitions, not here: any `std::exception`-derived internal cause -> `SSLM_ALLOCATION_FAILED`,
// `catch (...)` -> `SSLM_ARTIFACT_REJECTED` live only for non-`std::exception`-derived throws.
// `WrapBadAllocContract` is independently-governed S-HARDEN-7 machinery this ABI does not own and
// is ruled out of scope for a rewrite (D-SLM3464) -- see the two call sites' own comments.
template <typename Fn>
sslm_status CatchAllocationFailure(Fn&& fn) {
	try {
		return fn();
	} catch (const std::bad_alloc&) {
		return SSLM_ALLOCATION_FAILED;
	} catch (const std::length_error&) {
		return SSLM_ALLOCATION_FAILED;
	} catch (...) {
		return SSLM_ARTIFACT_REJECTED;
	}
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
	// P1 (Claude/Poirot/2c18dab-t2139-abi-build-review.md Sec7.3, third confirmation pass),
	// CLOSED (D-SLM3466's owed pin, Claude/Poirot/3bcbe43-t2139-fourth-confirmation-review.md S2/
	// S3): consulted at THIS function's own entry, matching S-HARDEN-7's own *Impl convention --
	// but now through the SITE-SPECIFIC post-load-region slot
	// (MaybeThrowInjectedBadAllocFaultPostLoadRegion(), src/bad_alloc_wrap.h /
	// tests/support/bad_alloc_injection.h), not the plain shared slot SslmModel::Load's own *Impl
	// consults. The two slots are independent thread_locals -- arming the post-load-region slot
	// and calling sslm_model_map is NOT consumed by Load's own consultation (which only ever reads
	// the plain slot), so an armed fault reaches THIS consultation point for real
	// (tools/t2139_d3466_postload_region_pin.cpp). Real, zero-cost (outside the test-injection
	// build, bad_alloc_wrap.h's own guard) defense-in-depth otherwise: it fires for real if this
	// function is ever reached by a future call path that does not go through Load first.
	superslm::internal::MaybeThrowInjectedBadAllocFaultPostLoadRegion();
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
		// T-2147 (design §15.3, D-SLM3483): GROWN from `hidden_size` (one token) to
		// `max_chunk_budget * hidden_size` -- PrefillWholeTokensImpl's chunk-batched path
		// (RunLayerLoopChunkBatched) needs a [chunk_tokens, hidden_size] activation region to
		// embed and carry the whole admitted chunk across layers, not one token's worth. This
		// is an ABI-VISIBLE change to `sslm_workspace_size`'s own returned byte count -- correct
		// and larger for any config with `max_chunk_budget > 1`, identical to before for
		// `max_chunk_budget == 1` (the region's byte count is unchanged at exactly `hidden_size`)
		// -- legitimate now, before S-FREEZE locks the ABI surface, by the identical reasoning
		// already applied once to `sslm_prefill`'s own `sslm_span_kind` parameter (§5 there,
		// "Pre-freeze migration note"). `sslm_decode_step`'s own single-token consumer is
		// unaffected: it always uses the first `hidden_size` bytes of whatever the (now larger)
		// region provides, exactly as `sslm_workspace`'s own reuse law (T-2133 §7.1) already
		// requires of every other region. `final_codes` (below) is UNCHANGED at `hidden_size` --
		// design §15.3 rules only `embed_codes_bytes` grows; the exact remaining scratch geometry
		// (kacc_all/vacc_all/normed/etc.) is chunk-batched, function-local `std::vector` scratch,
		// the same convention RunLayerLoopImpl's own per-token locals already use, never
		// ABI-workspace-carved.
		if (!CheckedMulSizeT(static_cast<size_t>(config.max_chunk_budget),
		                      static_cast<size_t>(c.hidden_size), &L.embed_codes_bytes)) {
			of = true;
		}
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
	// int64) + kv_saturation_count(8) + forced_token_count(8, design Sec7.3, D-SLM3486, 'SSB2' --
	// see the C5 block's own top comment) + kv_block_count(4) = 112 bytes. THIS CONSTANT IS
	// INDEPENDENT of the save/restore block's own kSeqBlobFixedHeaderBytes (108, which excludes
	// kv_block_count, added separately at each of that block's own call sites) -- both name the
	// same design Sec7.3 field list and must be kept in step by hand; there is no single shared
	// constant between this function and sslm_seq_save/sslm_seq_restore (S4/M4 sweep,
	// Claude/Poirot/9bc9ec6-t2132-g5-arc-review.md and the coordinator's own M4 follow-up brief).
	constexpr size_t kFixedHeaderBytes = 112;
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
	// P1 (Claude/Poirot/2c18dab-t2139-abi-build-review.md Sec7.3, third confirmation pass): the
	// sweep above found `free_list.reserve`/`push_back` can throw `std::bad_alloc` past the
	// `new (std::nothrow)` check above -- caught here (see CatchAllocationFailure's own comment),
	// with `h` deleted first so a mid-construction throw never leaks the handle.
	const sslm_status fill_st = CatchAllocationFailure([&]() -> sslm_status {
		h->free_list.reserve(block_count);
		for (uint32_t i = 0; i < block_count; ++i) h->free_list.push_back(i);
		return SSLM_OK;
	});
	if (fill_st != SSLM_OK) {
		delete h;
		return fill_st;
	}
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
		// P1 (Claude/Poirot/2c18dab-t2139-abi-build-review.md Sec7.3, third confirmation pass):
		// the injection seam's own consultation point moved from here to BuildEngineCache's own
		// entry (below) -- the seam is single-shot and consulted synchronously, so one armed
		// fault can only ever reach the FIRST consultation point a call makes; having a
		// consultation here would make BuildEngineCache's own path structurally unreachable by
		// the pin, exactly the gap P1 named. This try/catch's own correctness is unchanged and
		// still real (any std::bad_alloc SslmModel::Load's own documented contract produces is
		// still caught here); the pin's own injected-fault coverage is now spent proving the
		// specific gap the finding named, not duplicating coverage of an already-proven catch.
		st = superslm::SslmModel::Load(static_cast<const uint8_t*>(data), size, view, &err);
	} catch (const std::bad_alloc&) {
		// FOLD RULING on F2 (design Sec6, design commit dated 2026-08-17; D-SLM3462) -- see
		// CatchAllocationFailure's own comment for the full reasoning: this is one of the two
		// exception types this family is defined to cover.
		return SSLM_ALLOCATION_FAILED;
	} catch (const std::length_error&) {
		// FOLD RULING, second pass, on D-SLM3464 (design Sec6, design commit dated 2026-08-17):
		// this clause is written for symmetry with CatchAllocationFailure's own three-way split,
		// but on THIS call path it is dead in practice -- `SslmModel::Load`'s own public entry is
		// `internal::WrapBadAllocContract` (src/bad_alloc_wrap.h), whose own
		// `catch (const std::exception&) { throw std::bad_alloc{}; }` narrows every non-bad_alloc
		// `std::exception` -- `std::length_error` included -- to `std::bad_alloc` BEFORE it ever
		// reaches this try/catch. A real `std::length_error` from inside `Load` is therefore
		// caught by the `bad_alloc` clause above, not this one; this clause exists only in case a
		// future caller of THIS try/catch (were one ever added) throws `length_error` directly,
		// bypassing `Load`. Kept for the same reason `MapForwardStatus`-style completeness is kept
		// elsewhere: the shape stays exhaustive over `sslm_status`'s own resource-exhaustion
		// family even where one arm is currently unreachable through the one real caller.
		return SSLM_ALLOCATION_FAILED;
	} catch (...) {
		// The true, unconditional boundary -- every exception type NOT caught above.
		//
		// FOLD RULING, second pass, on D-SLM3464: on THIS specific call path (fronted by
		// `SslmModel::Load`'s own `WrapBadAllocContract`), this arm is NARROWER than
		// `CatchAllocationFailure`'s own general `catch (...)` -> `SSLM_ARTIFACT_REJECTED` rule.
		// `WrapBadAllocContract` narrows every `std::exception`-derived throw (bad_alloc,
		// length_error, runtime_error, logic_error, every other standard or user-defined
		// std::exception subtype) to `std::bad_alloc`, so none of them ever reach this line --
		// they are all caught by the `bad_alloc` clause above instead, and this verb's own
		// DELIVERED contract for any std::exception-derived internal cause is
		// `SSLM_ALLOCATION_FAILED`, not `SSLM_ARTIFACT_REJECTED` (proven by real fault injection,
		// `tools/t2139_f2_length_error_pin.cpp`'s own `kRuntimeError` cell). This arm stays live
		// and correct for exactly the one class `WrapBadAllocContract` does not narrow: a throw
		// NOT derived from `std::exception` at all (a foreign type, or a non-class throw like
		// `throw int`), which matches neither of `WrapBadAllocContract`'s own two catch clauses
		// and propagates through it unmolested, reaching this boundary exactly as it does on
		// every other verb. No new ordinal minted; `SSLM_ARTIFACT_REJECTED` is the existing
		// "internal rejection, no dedicated status" family (`MapForwardStatus`, below, this file,
		// is the existing precedent). Do not "fix" this by rewriting `WrapBadAllocContract` --
		// D-SLM3464 rules against that (src/bad_alloc_wrap.h is independently-governed S-HARDEN-7
		// machinery, a 20-site house-wide contract this ABI does not own).
		return SSLM_ARTIFACT_REJECTED;
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
	//
	// P1 (Claude/Poirot/2c18dab-t2139-abi-build-review.md Sec7.3, third confirmation pass): THE
	// named finding -- BuildEngineCache sat OUTSIDE the N3 fix round's own try, allocating freely
	// (backings.resize/layers.resize/WidenGainToInt32's own returned vector/cache.err's string
	// assignment) with nothing catching it, which is also exactly why the pre-P1 N3 pin could not
	// reach it (that pin's own injection point was inside sslm_model_map's earlier try, around
	// SslmModel::Load, which ran to completion and returned before BuildEngineCache was ever
	// called). Wrapped now, with `h` deleted on an allocation failure exactly as it already is on
	// BuildEngineCache's own ordinary rejection, so both failure paths leave no leaked handle.
	//
	// F3 (Claude/Poirot/4466666-t2139-third-confirmation-review.md): this comment previously
	// claimed the injection seam's own consultation point moving INTO BuildEngineCache itself
	// (its own top comment) meant tools/t2139_n3_bad_alloc_pin.cpp's own armed call now reaches
	// THIS exact wrap -- false at the time (the seam was single-shot and shared, and
	// SslmModel::Load consulted it first). CLOSED (D-SLM3466's owed pin, Claude/Poirot/
	// 3bcbe43-t2139-fourth-confirmation-review.md S2/S3): BuildEngineCache's own consultation
	// (above) now reads a SEPARATE, independent post-load-region slot
	// (tests/support/bad_alloc_injection.h) that Load's own *Impl never touches, so arming that
	// slot specifically and calling sslm_model_map DOES reach this exact wrap, isolated from
	// Load's own catch. Proven by tools/t2139_d3466_postload_region_pin.cpp. The N3 pin
	// (tools/t2139_n3_bad_alloc_pin.cpp) is unchanged and still documents the ORIGINAL, still-true
	// fact about the PLAIN shared slot it arms: that one still always trips Load's own
	// consultation first, by design -- the two pins now exercise the two independent slots.
	const sslm_status cache_st = CatchAllocationFailure([&]() -> sslm_status {
		return BuildEngineCache(h) ? SSLM_OK : SSLM_ARTIFACT_REJECTED;
	});
	if (cache_st != SSLM_OK) {
		delete h;
		return cache_st;
	}

	// G5-2 (T-2132, design Sec13): parse this artifact's own SchemaMasks/SCM1 section, if
	// present -- absence is a valid, unconstrained-only artifact (design Sec13.1), exactly
	// today's pre-G5 behavior; `h->schemas` stays default-constructed/empty in that case, and
	// `sslm_schema_lookup` correctly reports SSLM_SCHEMA_NOT_FOUND for every name against it.
	// Runs AFTER BuildEngineCache (h->engine.ok already true) but BEFORE *out is published, the
	// same "not visible to any caller until fully built" discipline BuildEngineCache's own
	// comment states for C4's resolution step. Every structural/cross-check rejection Sec13.3
	// names is enforced by SchemaMasksTable::Parse (schema_masks.h); this call site maps ANY
	// such rejection to SSLM_ARTIFACT_REJECTED -- see schema_masks.h's own header comment for
	// why this build deliberately does not mint per-rejection SslmModelStatus enumerators.
	const superslm::SslmSectionView* schema_section =
	    h->view.Section(superslm::SslmSectionType::SchemaMasks);
	if (schema_section) {
		std::string schema_err;
		if (!superslm::SchemaMasksTable::Parse(schema_section->data, schema_section->byte_size,
		                                        h->view.config.vocab_size, h->schemas,
		                                        &schema_err)) {
			delete h;
			return SSLM_ARTIFACT_REJECTED;
		}
		// P1-style throw safety: schema_handles.resize can throw (bad_alloc/length_error) past
		// every check above it, matching every other vector-mutation site this file already
		// wraps (see this file's own P1 sweep comment, above).
		const sslm_status handles_st = CatchAllocationFailure([&]() -> sslm_status {
			h->schema_handles.resize(h->schemas.Count());
			for (size_t i = 0; i < h->schema_handles.size(); ++i) {
				h->schema_handles[i].model = h;
				h->schema_handles[i].index = static_cast<uint32_t>(i);
			}
			return SSLM_OK;
		});
		if (handles_st != SSLM_OK) {
			delete h;
			return handles_st;
		}
	}

	*out = h;
	return SSLM_OK;
}

// G5 (design Sec5, T-2132): schema lookup/enumeration -- read-only, no map/release verb, the
// schema is already resident wherever the model is.
extern "C" sslm_status sslm_schema_lookup(sslm_model model, const char* name, sslm_schema* out) {
	if (!out) return SSLM_INVALID_ARGUMENT;
	*out = nullptr;
	if (!model || !name) return SSLM_INVALID_ARGUMENT;
	size_t index = 0;
	if (!model->schemas.ByName(name, &index)) return SSLM_SCHEMA_NOT_FOUND;
	*out = &model->schema_handles[index];
	return SSLM_OK;
}

extern "C" size_t sslm_schema_count(sslm_model model) {
	if (!model) return 0;
	return model->schemas.Count();
}

extern "C" sslm_status sslm_schema_name(sslm_model model, int32_t index, char* buf, size_t* n) {
	if (!n) return SSLM_INVALID_ARGUMENT;
	if (!model || index < 0 || static_cast<size_t>(index) >= model->schemas.Count()) {
		return SSLM_INVALID_ARGUMENT;
	}
	const superslm::SchemaEntry* entry = model->schemas.ByIndex(static_cast<size_t>(index));
	const size_t required = entry->name.size();
	if (!buf || *n < required) {
		*n = required;
		return SSLM_BUFFER_TOO_SMALL;
	}
	std::memcpy(buf, entry->name.data(), required);
	*n = required;
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
//
// Zero-fills the drawn block before handing it back (T-2132 M2 fix, Claude/Curie/
// t2130-g5-red-suite-composition-joins-2026-08-17.md): a block's own not-yet-written region
// (padding within a per-token K/V span, or any capacity a given write path never touches) was
// previously left holding whatever the block last contained -- either ReturnBlock's own 0xCD
// poison (a block that was drawn, used, and released before) or the pool's own untouched
// creation-time content (0x00 for a block never drawn before this call, in every configuration
// this build observed; not itself a documented guarantee of the caller-owned buf, which is why
// this call defines the state explicitly rather than relying on it). `sslm_seq_save` (below)
// memcpy's the ENTIRE block, so an unwritten byte is as observable to a caller as a written
// one -- content-identical sequences must therefore start from a content-identical block,
// regardless of the pool's own draw/return history. Zeroing here (at the single point every
// draw site funnels through) rather than the whole pool at `sslm_kv_pool_create` time means the
// cost is paid once per live block, not once per pool byte, and a block a caller immediately
// overwrites in full (the common case) pays nothing extra beyond the memset itself. This does
// not weaken ReturnBlock's own poison-fill leak-check (Sec17 dim1): that obligation is "no
// content from the PRIOR sequence survives release," which a defined post-draw zero state
// still proves -- if prior content leaked through, the drawn block would read non-zero at a
// position no write touched, exactly as detectable as it was against the poison pattern.
sslm_status DrawBlock(sslm_kv_pool_s* pool, uint32_t* out_block_index) {
	uint32_t block_index;
	{
		std::lock_guard<std::mutex> lock(pool->mutex);
		if (pool->free_list.empty()) return SSLM_KV_POOL_EXHAUSTED;
		block_index = pool->free_list.back();
		pool->free_list.pop_back();
		pool->live_refs.fetch_add(1, std::memory_order_acq_rel);
	}
	// S3 (Claude/Poirot/9bc9ec6-t2132-g5-arc-review.md): the memset moved OUTSIDE the lock,
	// mirroring ReturnBlock's own sibling shape (fill outside, publish under lock) below.
	// Reasoning: once `block_index` is popped off `free_list` inside the critical section
	// above, no OTHER thread can draw that same index again -- `free_list` is the pool's only
	// publication mechanism for "this block is available," and this block is no longer on it.
	// The memset therefore touches memory no concurrent DrawBlock/ReturnBlock call can reach,
	// so it needs no lock at all; the draw is still atomic (the free-list pop/live_refs bump
	// happen together, under the lock, before any caller sees `*out_block_index`), only the
	// ~448 MB fill (the 1.5B fixture's own block_size) no longer serializes every concurrent
	// sslm_seq_create/sslm_prefix_begin/sslm_seq_restore behind one pool-wide mutex.
	std::memset(static_cast<uint8_t*>(pool->buf) + static_cast<size_t>(block_index) * pool->block_size,
	            0, pool->block_size);
	*out_block_index = block_index;
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
// P1 (Claude/Poirot/2c18dab-t2139-abi-build-review.md Sec7.3, third confirmation pass): renamed
// to *Impl and wrapped (the same rename-and-wrap convention src/bad_alloc_wrap.h's own S-HARDEN-7
// population already uses) -- this function's own `embed_codes_fallback.assign` (ws-absent path)
// and `ResolveLayers`' own per-call `layers_scratch` copy (adapter-bound path) can both throw,
// and this helper is shared by two extern "C" verbs (sslm_prefix_prefill, sslm_prefill), so
// wrapping it once here closes both call sites at once rather than twice, separately.
// G5 (design Sec5/Sec10.2, T-2132): `kind`/`bound_schema`/`walk_state` thread the repaired
// span-kind discriminator through the shared prefill loop -- SSLM_SPAN_PROMPT never advances
// `*walk_state` (design's own "never advance, whatever schema is bound"); SSLM_SPAN_SCHEMA_CONTENT
// advances it token by token, checking each token's reachability against `bound_schema`'s own
// compiled DFA FIRST (before that token's forward pass runs) -- a span that leaves the DFA's
// language returns SSLM_SCHEMA_SPAN_UNREACHABLE with the walk-state left at its last-good value
// (design Sec6 G5-4: "the sequence's own walk-state is unmoved by the rejected call"). Callers
// (sslm_prefill/sslm_prefix_prefill) reject SSLM_SCHEMA_SPAN_UNBOUND against SSLM_SCHEMA_NONE
// BEFORE calling this -- `bound_schema` is only ever non-null here on a SCHEMA_CONTENT call, so
// this function never itself needs to re-check for the unbound case. `forced_token_count`
// (nullptr for prefix construction, design Sec12's own "no dedicated stats call for a prefix")
// is incremented once per token admitted under SCHEMA_CONTENT (design Sec6 G5-3/Sec7 dim7).
sslm_status PrefillWholeTokensImpl(sslm_model_s* model, superslm::SequenceLayerState& state,
                                    uint8_t* kv_block, size_t block_size, const int32_t* tokens,
                                    int32_t count, int32_t chunk_budget, int32_t* consumed,
                                    int32_t* out_last_token, sslm_adapter_s* adapter,
                                    sslm_workspace_s* ws, sslm_span_kind kind,
                                    sslm_schema bound_schema, uint32_t* walk_state,
                                    int64_t* forced_token_count) {
	const superslm::SslmModelConfig& c = model->view.config;
	const int32_t n = count < chunk_budget ? count : chunk_budget;
	const superslm::SchemaEntry* schema_entry =
	    (kind == SSLM_SPAN_SCHEMA_CONTENT && bound_schema)
	        ? model->schemas.ByIndex(bound_schema->index)
	        : nullptr;
	// S7 fix round 2 (coordinator's own closing-round brief): thread this function's own
	// embed_codes scratch through a caller-supplied, correctly-sized workspace, exactly the
	// carving sslm_decode_step already does -- the SAME embed_codes region (design Sec7.1's own
	// workspace is never used by two calls concurrently on one handle; sequential reuse across
	// sslm_prefill/sslm_decode_step calls is the whole point of a shared, caller-owned scratch
	// region, design Sec8.3/dim8 M1). Falls back to a heap-allocated vector when `ws` is null or
	// undersized for this model, so no existing nullptr-`ws` caller regresses.
	//
	// T-2147 (design §15.3, D-SLM3483): `embed_codes` is now `max_chunk_budget * hidden_size`
	// bytes when workspace-carved (ComputeWorkspaceLayout, above) -- this function uses it as the
	// chunk-batched forward pass's own [chunk_tokens, hidden_size] activation buffer, staging
	// every admitted token's embedding before the first layer runs, then letting
	// RunLayerLoopChunkBatched mutate it in place, layer by layer, for the whole chunk at once.
	const WorkspaceLayout layout = ws ? ComputeWorkspaceLayout(model, ws->config) : WorkspaceLayout{};
	const bool ws_usable = ws && !layout.overflowed && ws->buf_size >= layout.total_bytes &&
	                        layout.embed_codes_bytes >=
	                            static_cast<size_t>(n) * static_cast<size_t>(c.hidden_size);
	std::vector<int8_t> embed_codes_fallback;
	int8_t* embed_codes;
	if (ws_usable) {
		embed_codes = reinterpret_cast<int8_t*>(static_cast<uint8_t*>(ws->buf) +
		                                         layout.embed_codes_offset);
	} else {
		embed_codes_fallback.assign(static_cast<size_t>(n) * static_cast<size_t>(c.hidden_size), 0);
		embed_codes = embed_codes_fallback.data();
	}
	std::vector<superslm::LayerWeights> layers_scratch;
	const superslm::LayerWeights* layers = ResolveLayers(model, adapter, &layers_scratch);

	if (n == 0) return SSLM_OK;

	// --- T-2147 (design §15.3): admission pre-scan, before any forward-pass compute --------
	// Three independent stopping conditions, checked in the SAME priority order the pre-fold
	// per-token loop checked them in (token-id domain, then context-cap headroom, then -- only
	// over the range both of those already admit -- the DFA reachability walk): each names the
	// first index it would itself reject at; the smallest of the three is the admitted prefix
	// length, and which one produced it is the status this call returns when it is less than
	// `n`. This reproduces PrefillWholeTokensImpl's own pre-existing per-token, first-failure-
	// wins behaviour exactly, computed up front so the (expensive) batched forward pass below
	// runs ONLY over tokens already known to be admitted (§15.3's "a rejected token and
	// everything after it costs zero forward-pass compute" contract, extended from status/
	// consumption to compute cost).
	int32_t admit_id = n;
	for (int32_t i = 0; i < n; ++i) {
		const int32_t token = tokens[i];
		if (token < 0 || static_cast<uint32_t>(token) >= c.vocab_size) {
			admit_id = i;
			break;
		}
	}
	const int64_t cap_headroom = static_cast<int64_t>(c.context_cap) - state.context_length;
	const int32_t admit_cap = cap_headroom <= 0
	                               ? 0
	                               : (cap_headroom >= n ? n : static_cast<int32_t>(cap_headroom));
	const int32_t admit_idcap = admit_id < admit_cap ? admit_id : admit_cap;

	// G5-4 (design Sec6/Sec14.3, D-SLM3478): the reachability walk is a cheap, sequential,
	// O(1)-per-token CSR-table lookup (design §15.3) -- run here, over exactly the
	// already-id/cap-admitted prefix, BEFORE the GEMM-heavy batched forward pass below ever
	// issues. `dim5_failure_red.cpp`'s own TestDim5_M5 cell (partial-consumption against the
	// actual post-rejection state of a multi-token span) is unaffected: every token strictly
	// before the rejected one is still fully, permanently admitted by this same call.
	int32_t admit_schema = admit_idcap;
	uint32_t schema_walk_state_after = walk_state ? *walk_state : 0;
	if (schema_entry) {
		uint32_t w = *walk_state;
		for (int32_t i = 0; i < admit_idcap; ++i) {
			uint32_t next_w = 0;
			if (!model->schemas.Transition(*schema_entry, w, static_cast<uint32_t>(tokens[i]),
			                                &next_w)) {
				admit_schema = i;
				break;
			}
			w = next_w;
		}
		schema_walk_state_after = w;
	}
	const int32_t admit_count = admit_schema;  // <= admit_idcap <= admit_id, admit_cap

	sslm_status stop_status = SSLM_OK;
	if (admit_count < n) {
		if (admit_schema < admit_idcap) {
			stop_status = SSLM_SCHEMA_SPAN_UNREACHABLE;
		} else if (admit_id <= admit_cap) {
			stop_status = SSLM_TOKEN_ID_OUT_OF_RANGE;
		} else {
			stop_status = SSLM_CONTEXT_CAP_EXCEEDED;
		}
	}

	// --- T-2147 (design §15.1): the chunk-batched forward pass, over exactly `admit_count` ---
	// admitted tokens -- zero forward-pass compute for anything past the admitted prefix,
	// matching §15.3's own extension of the partial-consumption contract to compute cost.
	if (admit_count > 0) {
		std::vector<superslm::CarriedScale> hidden_scales(static_cast<size_t>(admit_count));
		for (int32_t i = 0; i < admit_count; ++i) {
			superslm::CarriedScale embed_scale{};
			const superslm::SslmForwardStatus est = superslm::EmbedEntry(
			    tokens[i], static_cast<int32_t>(c.vocab_size), model->engine.embed_weights,
			    c.hidden_size, model->engine.embed_site_constant,
			    embed_codes + static_cast<size_t>(i) * c.hidden_size, &embed_scale);
			if (est != superslm::SslmForwardStatus::Ok) return MapForwardStatus(est);
			hidden_scales[static_cast<size_t>(i)] = embed_scale;
		}

		const superslm::SslmForwardStatus st = superslm::RunLayerLoopChunkBatched(
		    embed_codes, hidden_scales.data(), static_cast<size_t>(admit_count), layers,
		    c.num_hidden_layers, c.hidden_size, c.head_dim, c.num_key_value_heads,
		    c.intermediate_size, c.context_cap, state.context_length, model->view.rope_tables,
		    kv_block, block_size, /*option_g_fused_k_landing=*/false, &state.kv_saturation_count,
		    /*site_prefix=*/{}, nullptr);
		if (st != superslm::SslmForwardStatus::Ok) return MapForwardStatus(st);

		// forward_sites.h: "a sequence resting between whole tokens carries a marker at layer
		// 0" -- the batched call above leaves every admitted token's own layer stack fully run;
		// `state` rests at the LAST admitted token's own final hidden state, exactly what the
		// pre-existing per-token loop left there after its own last iteration (the per-token
		// intermediate hidden states for tokens before the last are never independently
		// observed -- each token started fresh from its own EmbedEntry, never from the previous
		// token's hidden_codes; only the K/V store carries state between tokens).
		for (uint32_t k = 0; k < c.hidden_size; ++k) {
			state.hidden_codes[k] =
			    embed_codes[static_cast<size_t>(admit_count - 1) * c.hidden_size + k];
		}
		state.hidden_scale = hidden_scales[static_cast<size_t>(admit_count - 1)];
		state.layer_index = 0;
		state.context_length += admit_count;

		if (schema_entry) {
			*walk_state = schema_walk_state_after;
			if (forced_token_count) *forced_token_count += admit_count;
		}
		*consumed += admit_count;
		if (out_last_token) *out_last_token = tokens[admit_count - 1];
	}

	return stop_status;
}

sslm_status PrefillWholeTokens(sslm_model_s* model, superslm::SequenceLayerState& state,
                                uint8_t* kv_block, size_t block_size, const int32_t* tokens,
                                int32_t count, int32_t chunk_budget, int32_t* consumed,
                                int32_t* out_last_token, sslm_adapter_s* adapter,
                                sslm_workspace_s* ws, sslm_span_kind kind, sslm_schema bound_schema,
                                uint32_t* walk_state, int64_t* forced_token_count) {
	return CatchAllocationFailure([&]() -> sslm_status {
		return PrefillWholeTokensImpl(model, state, kv_block, block_size, tokens, count,
		                               chunk_budget, consumed, out_last_token, adapter, ws, kind,
		                               bound_schema, walk_state, forced_token_count);
	});
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
	// P1 (Claude/Poirot/2c18dab-t2139-abi-build-review.md Sec7.3): `.assign` can throw past the
	// `new (std::nothrow)` check above -- caught, with both the handle AND its own drawn block
	// released on failure (matching M4's own leak discipline).
	const sslm_status assign_st = CatchAllocationFailure([&]() -> sslm_status {
		h->hidden_codes_storage.assign(model->view.config.hidden_size, 0);
		return SSLM_OK;
	});
	if (assign_st != SSLM_OK) {
		ReturnBlock(p, block_index, h->kv_block, p->block_size);
		delete h;
		return assign_st;
	}
	h->state.hidden_codes = h->hidden_codes_storage.data();
	model->live_refs.fetch_add(1, std::memory_order_acq_rel);
	*out = h;
	return SSLM_OK;
}

// G5 (design Sec5, T-2132): "Valid only before any content has been prefilled into the prefix
// under construction" -- prefix->current_token == -1 (its own default, and its own
// never-touched sentinel; PrefillWholeTokensImpl only ever writes a token id >= 0 to it) is
// exactly that condition. No dedicated rejection status is named for this case in the design's
// closed Sec6 taxonomy (unlike sslm_seq_set_schema's own SSLM_SCHEMA_BIND_REJECTED) --
// SSLM_INVALID_ARGUMENT is the general call-shape-wrong family, matching this file's own
// identical precedent at sslm_seq_adopt_prefix's "M2" comment for an analogous naming gap.
extern "C" sslm_status sslm_prefix_set_schema(sslm_prefix prefix, sslm_schema schema) {
	if (!prefix) return SSLM_INVALID_ARGUMENT;
	if (prefix->frozen || prefix->current_token != -1) return SSLM_INVALID_ARGUMENT;
	// C2 (Claude/Poirot/9bc9ec6-t2132-g5-arc-review.md): `sslm_schema_s::model` (written once at
	// sslm_model_map, src/sslm_abi.cpp:1054) is read here for exactly the cross-model bind this
	// field exists to prevent -- mirrors sslm_seq_set_schema's own identical fix below, and the
	// GPU twin (`SslmGpuSeqSetSchemaForG5Bridge`) already rejects an out-of-range schema index
	// the same way. Uses this file's own existing cross-handle-mismatch family
	// (SSLM_INVALID_ARGUMENT, the same status sslm_prefix_begin/sslm_seq_create/
	// sslm_seq_adopt_prefix/sslm_seq_restore already return for a pool/model/prefix identity
	// mismatch, src/sslm_abi.cpp:1336/1436/1562/2121) rather than minting a new schema-specific
	// ordinal -- no design ruling names one, and Sec6's registry is closed without a fold.
	if (schema && schema->model != prefix->model) return SSLM_INVALID_ARGUMENT;
	prefix->bound_schema = schema;
	prefix->dfa_walk_state = schema ? 0u : kDfaWalkStateUnused;
	return SSLM_OK;
}

// G5 (design Sec5's "prefix construction inherits the same discriminator", T-2132, wired for
// real this build): SSLM_SPAN_SCHEMA_CONTENT against a prefix with no schema bound rejects
// (SSLM_SCHEMA_SPAN_UNBOUND, symmetric with sslm_prefill's own identical rule); otherwise the
// span kind threads through PrefillWholeTokens exactly as sslm_prefill's own call does.
extern "C" sslm_status sslm_prefix_prefill(sslm_model model, sslm_prefix prefix,
                                            const int32_t* tokens, int32_t count,
                                            int32_t chunk_budget, sslm_span_kind kind,
                                            sslm_workspace ws, int32_t* consumed) {
	// S7 fix round 2: ws is now genuinely read (see PrefillWholeTokens' own comment) -- no longer
	// merely accepted for call-shape parity.
	if (!consumed) return SSLM_INVALID_ARGUMENT;
	*consumed = 0;
	if (!model || !prefix) return SSLM_INVALID_ARGUMENT;
	if (count < 0 || chunk_budget < 1 || (count > 0 && !tokens)) return SSLM_INVALID_ARGUMENT;
	if (prefix->frozen) return SSLM_PREFIX_FROZEN_REJECTED;
	if (!model->engine.ok) return SSLM_ARTIFACT_REJECTED;
	if (kind == SSLM_SPAN_SCHEMA_CONTENT && !prefix->bound_schema) {
		return SSLM_SCHEMA_SPAN_UNBOUND;
	}

	return PrefillWholeTokens(model, prefix->state, prefix->kv_block, prefix->block_size, tokens,
	                           count, chunk_budget, consumed, &prefix->current_token, nullptr, ws,
	                           kind, prefix->bound_schema, &prefix->dfa_walk_state, nullptr);
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
	// P1 -- see sslm_prefix_begin's own identical fix.
	const sslm_status assign_st = CatchAllocationFailure([&]() -> sslm_status {
		h->hidden_codes_storage.assign(model->view.config.hidden_size, 0);
		return SSLM_OK;
	});
	if (assign_st != SSLM_OK) {
		ReturnBlock(p, block_index, h->kv_block, p->block_size);
		delete h;
		return assign_st;
	}
	h->state.hidden_codes = h->hidden_codes_storage.data();
	h->current_token = -1;
	model->live_refs.fetch_add(1, std::memory_order_acq_rel);
	// T-2199 Phase D3 fix (Sec9 dim3, GATE, D-SLM3719): registered as live BEFORE the handle is
	// ever handed to the caller -- no concurrent decode_step call can name this pointer before
	// `*out = h` below makes it observable to any other thread, so there is no ordering hazard
	// registering it here (this store happens-before any possible use of the handle).
	{
		std::lock_guard<std::mutex> registry_lock(g_seq_registry_mutex);
		g_live_seqs.insert(h);
	}
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
	// T-2199 Phase D3 fix, corrected a SECOND time (Sec9 dim3, GATE, D-SLM3719): remove this
	// sequence from the liveness registry FIRST, under `g_seq_registry_mutex` -- the SAME lock
	// `sslm_decode_stepImpl`'s own liveness-check-and-lock step holds while it decides whether
	// `seq` is even safe to touch (sslm_seq_s's own comment, above `g_seq_registry_mutex`'s
	// declaration, has the full reasoning). Mutually exclusive with that check: either this erase
	// runs first (a not-yet-started or not-yet-arrived decode call will find `seq` absent and
	// never dereference it at all), or a decode call's own registry-locked section already ran
	// first (proved `seq` live and locked its `lifecycle_mutex` before this erase could begin) --
	// there is no third interleaving.
	{
		std::lock_guard<std::mutex> registry_lock(g_seq_registry_mutex);
		g_live_seqs.erase(seq);
	}
	// Acquire-then-release `lifecycle_mutex` as a BARRIER before touching anything
	// `sslm_decode_stepImpl` might still be mid-flight on for this SAME sequence: if a batched
	// decode step already proved liveness (above) and is currently inside its own per-sequence
	// critical section, this blocks here until that section completes -- THEN it is safe to
	// destroy the anti-LM state and free the object; if no step is in flight (or none reached
	// this sequence before the registry erase above), this is one uncontended lock/unlock pair.
	{
		std::lock_guard<std::mutex> lock(seq->lifecycle_mutex);
	}
	// T-2199 Phase D3 (plan Sec9 dim3, teardown-during-flight): this sequence's own anti-LM
	// state, if it was ever created, is destroyed here -- before ReturnBlock/delete, matching
	// this function's own established teardown order (release owned resources, then the pool
	// block, then the handle itself).
	if (seq->damped_greedy_antilm) superslm::AntiLmDestroy(seq->damped_greedy_antilm);
	ReturnBlock(seq->pool, seq->block_index, seq->kv_block, seq->block_size);
	seq->model->live_refs.fetch_sub(1, std::memory_order_acq_rel);
	delete seq;
	return SSLM_OK;
}

// G5 (design Sec5, T-2132): "valid ONLY when the sequence's DFA-walk state is at its start (a
// fresh sslm_seq_create, or immediately after sslm_seq_reset)". Both of those states leave
// dfa_walk_state at kDfaWalkStateUnused (never bound) or 0 (bound, unadvanced) -- see
// sslm_seq_s's own field comment. Any other value means the walk has genuinely advanced (some
// SSLM_SPAN_SCHEMA_CONTENT content was admitted), so re-binding -- even to a different schema --
// rejects: schema re-binding mid-generation is out of 1.0 scope (design Sec11).
extern "C" sslm_status sslm_seq_set_schema(sslm_seq seq, sslm_schema schema) {
	if (!seq) return SSLM_INVALID_ARGUMENT;
	// S7 (Claude/Poirot/9bc9ec6-t2132-g5-arc-review.md): the shipped header's own condition is
	// "valid ONLY when the sequence's DFA-walk state is at its start (a fresh sslm_seq_create, or
	// immediately after sslm_seq_reset)" (design Sec5, ABI surface). `dfa_walk_state` alone does
	// not detect that: an unbound sequence's walk-state stays at kDfaWalkStateUnused forever, no
	// matter how much UNCONSTRAINED content has already been prefilled/decoded on it (nothing
	// touches dfa_walk_state while bound_schema is null), so the pre-fix guard would accept a
	// first-time bind on a sequence that is neither freshly created nor freshly reset -- not "at
	// its start" by the header's own two named examples. `current_token == -1` is exactly what
	// both of those examples share (sslm_seq_create's own default; sslm_seq_reset's own explicit
	// reset, src/sslm_abi.cpp) and what changes the instant either a prompt or schema-content
	// token is prefilled/decoded -- the same discriminator sslm_prefix_set_schema already uses
	// for the mirrored precondition on a prefix (above), which this fix now makes symmetric
	// across both handle types.
	if (seq->current_token != -1) {
		return SSLM_SCHEMA_BIND_REJECTED;
	}
	if (seq->dfa_walk_state != kDfaWalkStateUnused && seq->dfa_walk_state != 0) {
		return SSLM_SCHEMA_BIND_REJECTED;
	}
	// C2 (Claude/Poirot/9bc9ec6-t2132-g5-arc-review.md): see sslm_prefix_set_schema's own
	// identical fix and comment, above.
	if (schema && schema->model != seq->model) return SSLM_INVALID_ARGUMENT;
	seq->bound_schema = schema;
	seq->dfa_walk_state = schema ? 0u : kDfaWalkStateUnused;
	return SSLM_OK;
}

extern "C" sslm_status sslm_seq_reset(sslm_seq seq) {
	if (!seq) return SSLM_INVALID_ARGUMENT;
	if (seq->state.layer_index != 0) return SSLM_SEQ_RESET_MIDTOKEN_REJECTED;
	// T-2132 M2 fix (same class as DrawBlock's own zero-fill, above): reset re-exposes this
	// block's not-yet-written region to the NEXT generation exactly the way a fresh draw does --
	// left at 0xCD (the poison this call used before this fix), a reset-and-reused sequence's
	// save-blob would disagree with a content-identical freshly-drawn sequence's, at whatever
	// bytes neither generation's own writes touch, breaking the same determinism law DrawBlock's
	// comment states. Zero (DrawBlock's own defined post-draw state) is the correct target here,
	// not the pre-fix poison value: the leak this memset guards against -- the PRIOR generation's
	// real K/V content surviving into the next one -- is caught exactly as well by a defined zero
	// as by 0xCD (either one is trivially distinct from real, non-degenerate model output).
	std::memset(seq->kv_block, 0, seq->block_size);
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
	// G5 (design Sec5, T-2132): "clears the DFA-walk-state back to the bound schema's start
	// state... but preserves the schema binding itself" -- bound_schema is untouched;
	// dfa_walk_state returns to 0 (bound) or stays kDfaWalkStateUnused (unbound). No leak from
	// the prior generation's walk into the next one (design Sec7 dim1, the poison-fill
	// discipline this dimension already applies to KV-block recycling, applied here).
	seq->dfa_walk_state = seq->bound_schema ? 0u : kDfaWalkStateUnused;
	seq->forced_token_count = 0;
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

	// G5 (design Sec5, T-2132): the prefix's own recorded walk-state transfers IFF the
	// adopting sequence's bound schema matches the prefix's recorded origin schema, OR the
	// prefix carries no real schema-content progress (unbound, or bound-but-still-at-start --
	// "a prompt-only prefix... compatible with any sequence's schema binding"). EVERY OTHER
	// CASE is SSLM_PREFIX_SCHEMA_MISMATCH -- stated exhaustively (design Sec5): a mismatched
	// bound schema, AND an unbound sequence adopting real progress, both reject. Checked BEFORE
	// any copy below, so a rejection leaves the sequence's own KV/state untouched (this file's
	// own "reject leaves state unperturbed" convention).
	const bool prefix_has_real_progress =
	    prefix->bound_schema != nullptr && prefix->dfa_walk_state != kDfaWalkStateUnused &&
	    prefix->dfa_walk_state != 0;
	if (prefix_has_real_progress && seq->bound_schema != prefix->bound_schema) {
		return SSLM_PREFIX_SCHEMA_MISMATCH;
	}
	if (prefix_has_real_progress) {
		seq->dfa_walk_state = prefix->dfa_walk_state;
	}
	// else: prefix is prompt-only (or bound-but-unadvanced) -- the adopting sequence's own
	// walk-state (and binding) is left exactly as it already was, per design Sec5.

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

// G5 (design Sec5/Sec10.2, T-2132): `kind` is wired for real -- SSLM_SPAN_PROMPT never
// advances the walk (whatever schema is bound); SSLM_SPAN_SCHEMA_CONTENT always advances it,
// rejecting SSLM_SCHEMA_SPAN_UNBOUND against an unbound sequence up front (the repair's own
// rejection, design Sec5/Sec10.2). This is THE discriminator the rung-5 strike's pigeonhole
// proof (design Sec10.1) forced -- see PrefillWholeTokensImpl's own comment for the mechanism.
extern "C" sslm_status sslm_prefill(sslm_model model, sslm_seq seq, const int32_t* tokens,
                                     int32_t count, int32_t chunk_budget, sslm_span_kind kind,
                                     sslm_workspace ws, int32_t* consumed) {
	// S7 fix round 2: ws is now genuinely read (see PrefillWholeTokens' own comment).
	if (!consumed) return SSLM_INVALID_ARGUMENT;
	*consumed = 0;
	if (!model || !seq) return SSLM_INVALID_ARGUMENT;
	if (count < 0 || chunk_budget < 1 || (count > 0 && !tokens)) return SSLM_INVALID_ARGUMENT;
	if (!model->engine.ok) return SSLM_ARTIFACT_REJECTED;
	if (kind == SSLM_SPAN_SCHEMA_CONTENT && !seq->bound_schema) {
		return SSLM_SCHEMA_SPAN_UNBOUND;
	}

	const sslm_status st = PrefillWholeTokens(model, seq->state, seq->kv_block, seq->block_size,
	                                           tokens, count, chunk_budget, consumed,
	                                           &seq->current_token, seq->adapter_handle, ws, kind,
	                                           seq->bound_schema, &seq->dfa_walk_state,
	                                           &seq->forced_token_count);
	if (*consumed > 0) {
		// The last token this call processed already ran every layer (PrefillWholeTokens'
		// own full-layer_budget convention) -- its final hidden state is ready for logits with
		// no further RunLayerLoop work (see sslm_seq_s::ready_for_logits's own comment).
		seq->ready_for_logits = true;
	}
	return st;
}

// G5-2/G5-5 (design Sec4/Sec6, T-2132): the mask-application primitive both sslm_decode_step
// (below) and the suite's own test-only guard-vitality hook (sslm_g5_test_only_apply_mask_and_
// argmax, this file's own bottom) call -- RELOCATED this fold (G5-5, T-2132) to
// superslm::ApplyMaskAndArgmax (forward_sites.h/.cpp), unchanged bits, so a second TU
// (src/gpu/gpu_1p0.cpp) can call the identical implementation instead of a parallel
// reimplementation -- ONE implementation, never two, exactly the discipline this file's own
// prior comment already named, now enforced across TUs rather than only within this one.

// P1 (Claude/Poirot/2c18dab-t2139-abi-build-review.md Sec7.3, third confirmation pass): renamed
// to *Impl and wrapped (same rename-and-wrap convention as PrefillWholeTokens/*Impl, above) --
// the fallback vectors' own `.assign` calls (ws-absent/undersized path) and `ResolveLayers`' own
// per-call `layers_scratch` copy (adapter-bound path, inside the loop below) can both throw.
static sslm_status sslm_decode_stepImpl(sslm_model model, sslm_seq* seqs, int32_t n,
                                         const sslm_decode_params* params, sslm_workspace ws,
                                         int32_t* out_tokens) {
	if (!model || !seqs || !params || !out_tokens) return SSLM_INVALID_ARGUMENT;
	if (n < 1) return SSLM_INVALID_ARGUMENT;
	if (params->layer_budget < 1 ||
	    static_cast<uint32_t>(params->layer_budget) > model->view.config.num_hidden_layers) {
		return SSLM_INVALID_ARGUMENT;
	}
	// T-2199 Phase D2 (plan Sec9 dim2): the SAME shared validation D3/D4 route through
	// (superslm_test_phaseD::ValidateDampedGreedyParams) -- checked before any sequence is
	// touched, matching this function's own "reject leaves state unperturbed" discipline. A
	// no-op (returns true unconditionally) when mode is greedy.
	{
		const superslm_test_phaseD::DampedGreedyValidationParams vp{
		    static_cast<superslm_test_phaseD::DampedGreedyMode>(params->mode), params->alpha_q15,
		    params->anti_lm_max_order, params->top_k};
		if (!superslm_test_phaseD::ValidateDampedGreedyParams(
		        vp, static_cast<int32_t>(model->view.config.vocab_size))) {
			return SSLM_INVALID_ARGUMENT;
		}
	}
	if (!model->engine.ok) return SSLM_ARTIFACT_REJECTED;
	// Pointer-only null check FIRST -- no dereference yet, so this is safe to do before any lock
	// is held (a null entry cannot be racing a concurrent release, there is nothing to free).
	for (int32_t i = 0; i < n; ++i) {
		if (!seqs[i]) return SSLM_INVALID_ARGUMENT;
	}

	// T-2199 Phase D3 fix, corrected a SECOND time (Sec9 dim3, GATE, D-SLM3719): the first two
	// corrections both still crashed (root-caused via SEH-wrapped/instrumented repro against a
	// real checkpoint, 2026-08-20): a lock embedded IN `sslm_seq_s` cannot protect against the
	// object's OWN deletion, because locking it requires the object to still exist -- if a
	// concurrent `sslm_seq_release` runs to completion (including `delete seq`) before this
	// function's own lock loop ever reaches that pointer, locking it is a use-after-free. Every
	// sequence in this batch is therefore first PROVED live via `g_live_seqs` (a lookup, never a
	// dereference of the sequence itself) under `g_seq_registry_mutex` -- the SAME lock
	// `sslm_seq_release` acquires to remove a sequence before it may proceed to its own barrier
	// and `delete` (see that function's and `g_seq_registry_mutex`'s own comments for the full
	// mutual-exclusion argument). `live[i]` records this batch's own per-INDEX result (not just
	// the deduped unique-pointer set locked below) so every later loop can skip an entry this
	// step proved dead WITHOUT ever dereferencing it -- and, per this cell's own construction, a
	// sequence found dead here does not abort sibling sequences in the same batch: the surviving
	// sequence's own decode must succeed regardless of what a DIFFERENT sequence's concurrent
	// release did. Locked in ASCENDING POINTER ORDER (not batch order) so two concurrent batched
	// calls naming overlapping sequences in different orders cannot deadlock against each other.
	std::vector<bool> live(static_cast<size_t>(n), false);
	std::vector<std::unique_lock<std::mutex>> seq_locks;
	{
		std::lock_guard<std::mutex> registry_lock(g_seq_registry_mutex);
		std::vector<sslm_seq_s*> lock_order(seqs, seqs + n);
		std::sort(lock_order.begin(), lock_order.end());
		// A duplicate sequence pointer named twice in one batch is not otherwise validated by
		// this function (pre-existing) -- deduped here specifically so this fix does not turn an
		// already-unvalidated input shape into a NEW self-deadlock (locking the same
		// non-recursive mutex twice from one thread).
		lock_order.erase(std::unique(lock_order.begin(), lock_order.end()), lock_order.end());
		seq_locks.reserve(lock_order.size());
		for (sslm_seq_s* s : lock_order) {
			if (g_live_seqs.count(s)) seq_locks.emplace_back(s->lifecycle_mutex);
		}
		for (int32_t i = 0; i < n; ++i) {
			live[i] = g_live_seqs.count(seqs[i]) != 0;
		}
	}

	// Every LIVE sequence validated before any is touched -- a malformed entry anywhere in the
	// batch leaves every sequence's state exactly as it was (this call's own "reject leaves state
	// unperturbed" contract, matching every other lifecycle guard in this design). A dead entry
	// (proved absent above, under the registry lock) is never dereferenced here at all -- it
	// contributes neither a rejection nor a touch, matching this cell's own "a concurrently
	// released sequence must never abort a surviving sibling's own call" construction.
	for (int32_t i = 0; i < n; ++i) {
		if (!live[i]) continue;
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
		// Lifecycle-safety note: `seq` is already covered by `seq_locks` above (acquired for
		// every LIVE sequence in this batch before this loop began), so no per-iteration lock is
		// taken here -- see that block's own comment for why locking only once this loop
		// reaches a sequence is insufficient. A dead entry (proved absent under the registry
		// lock, above) is skipped here BEFORE `seqs[i]` is ever dereferenced -- `seqs[i]` itself
		// (a pointer VALUE sitting in the caller's own array) is always safe to read; it is
		// dereferencing the sequence it points to (`seq->...`) that would be a use-after-free.
		if (!live[i]) {
			out_tokens[i] = -1;
			continue;
		}
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

		// T-2199 Phase D2 (plan Sec8 D2, Sec6 "mask-first"): damped-greedy mode operates on the
		// SAME masked row the schema branch below produces -- the mask is resolved FIRST (the
		// schema's own page, or an all-ones free-text page, per Sec6), narrowed into `logit_row`
		// in place exactly as ApplyMaskAndArgmax's own masking half does, THEN
		// DampedGreedyScoreAndArgmax scores it. This branch never reaches the plain
		// ArgmaxLowestIndexTieBreak/ApplyMaskAndArgmax calls below it -- those two remain
		// byte-for-byte unchanged for mode=SSLM_DECODE_MODE_GREEDY (Poirot-style no-regression
		// discipline, this ticket's own build log).
		if (params->mode == SSLM_DECODE_MODE_DAMPED_GREEDY) {
			const size_t mask_bytes = (static_cast<size_t>(c.vocab_size) + 7) / 8;
			std::vector<uint8_t> free_text_mask;
			const uint8_t* mask_bits;
			if (seq->bound_schema) {
				const superslm::SchemaEntry* entry = model->schemas.ByIndex(seq->bound_schema->index);
				mask_bits = entry->mask_pages +
				            static_cast<size_t>(seq->dfa_walk_state) * model->schemas.MaskPageBytes();
			} else {
				// Sec6: free-text mode's mask is all-ones by construction -- built once per call,
				// per sequence (no persistent buffer needed; this is not the hot O(V) cost this
				// design's own Sec2.4 measured, it is a memset-shaped O(V/8) fill).
				free_text_mask.assign(mask_bytes, 0xFF);
				mask_bits = free_text_mask.data();
			}
			// Mask-first: narrow masked positions to INT32_MIN in place, matching
			// ApplyMaskAndArgmax's own masking half exactly (superslm::ApplyMaskAndArgmax,
			// forward_sites.cpp) -- damped greedy never sees a row this step has not already
			// applied.
			for (int32_t t = 0; t < c.vocab_size; ++t) {
				if (!((mask_bits[static_cast<size_t>(t) >> 3] >> (t & 7)) & 1u)) {
					logit_row[t] = INT32_MIN;
				}
			}
			if (!seq->damped_greedy_antilm ||
			    seq->damped_greedy_antilm_order != params->anti_lm_max_order) {
				if (seq->damped_greedy_antilm) superslm::AntiLmDestroy(seq->damped_greedy_antilm);
				seq->damped_greedy_antilm = superslm::AntiLmCreate(params->anti_lm_max_order);
				seq->damped_greedy_antilm_order = params->anti_lm_max_order;
				if (!seq->damped_greedy_antilm) return SSLM_ALLOCATION_FAILED;
			}
			int32_t produced_dg = -1;
			bool refused = false;
			const bool ok = superslm::DampedGreedyScoreAndArgmax(
			    logit_row, mask_bits, static_cast<int32_t>(c.vocab_size), params->top_k,
			    seq->damped_greedy_antilm, params->alpha_q15, params->q_ln2, params->q_b,
			    params->q_c, &produced_dg, &refused);
			if (!ok) return SSLM_INVALID_ARGUMENT;  // domain rejection (k/vocab_size), plan Sec8 D1
			if (refused) {
				// Plan Sec7.5's own adopted refusal policy: abort this sequence's generation for
				// this call rather than fall back to plain argmax, matching the shipped attention
				// call site's SoftmaxKernelRefusedAfterGateAccepted precedent.
				return MapForwardStatus(superslm::SslmForwardStatus::SoftmaxKernelRefusedAfterGateAccepted);
			}
			if (seq->bound_schema) {
				const superslm::SchemaEntry* entry = model->schemas.ByIndex(seq->bound_schema->index);
				uint32_t next_state = seq->dfa_walk_state;
				const bool has_transition = model->schemas.Transition(
				    *entry, seq->dfa_walk_state, static_cast<uint32_t>(produced_dg), &next_state);
				if (!has_transition) {
					out_tokens[i] = -2;
					continue;
				}
				seq->dfa_walk_state = next_state;
			}
			// Sec7.7 Feedback: the anti-LM's update runs once per emitted token, after selection.
			superslm::AntiLmUpdate(seq->damped_greedy_antilm, produced_dg);
			out_tokens[i] = produced_dg;
			seq->current_token = produced_dg;
			seq->state.layer_index = 0;
			continue;
		}

		// G5-2 (design Sec4, T-2132): masking applies to int32 logits BEFORE argmax, indexed by
		// the sequence's own DFA-walk-state (design Sec4's architecture table) -- SSLM_SCHEMA_NONE
		// (bound_schema == nullptr) is byte-for-byte unchanged from pre-G5 output (no masking, the
		// existing ArgmaxLowestIndexTieBreak call, unmodified).
		int32_t produced;
		if (seq->bound_schema) {
			const superslm::SchemaEntry* entry = model->schemas.ByIndex(seq->bound_schema->index);
			const uint8_t* page = entry->mask_pages +
			                       static_cast<size_t>(seq->dfa_walk_state) * model->schemas.MaskPageBytes();
			superslm::ApplyMaskAndArgmax(logit_row, page, static_cast<int32_t>(c.vocab_size), &produced);
			// S2/D-SLM3476 (design Sec14.1, Claude/Poirot/9bc9ec6-t2132-g5-arc-review.md): the walk
			// advances to whatever state `produced` reaches ONLY when a matching CSR transition
			// entry actually exists. It is NOT guaranteed to: Sec13.2 permits an accepting state's
			// own mask page to be legitimately all-zero (the compiler's own non-empty-valid-set
			// proof, G-7a, only covers reachable NON-accepting states), and at such a state masked
			// argmax forces every logit to INT32_MIN and the lowest-index tie-break returns token
			// 0 -- for which no CSR row entry exists, because the row is empty. Before this fix
			// that failure was silently discarded and the walk froze in place, emitting token 0
			// forever at SSLM_OK. Ruled (Sec14.1): this is a genuine, defined dead end, not a
			// caller error -- `out_tokens[i]` reserves -2 for it (alongside the existing -1
			// "pending" sentinel), no new sslm_status ordinal. Nothing schema-related advances for
			// this sequence when it fires: `seq->dfa_walk_state` stays exactly at the state that
			// had no legal continuation, and neither `current_token` nor `state.layer_index` are
			// touched, so a caller that calls sslm_decode_step on this sequence again gets -2
			// again, deterministically -- a defined, resumable stop, never a torn state (design
			// Sec7 dim5's own "reject leaves state unperturbed" contract, applied to a per-sequence
			// outcome rather than a call-level fault, mirroring how -1/pending already coexists
			// with an overall SSLM_OK call).
			uint32_t next_state = seq->dfa_walk_state;
			const bool has_transition = model->schemas.Transition(
			    *entry, seq->dfa_walk_state, static_cast<uint32_t>(produced), &next_state);
			if (!has_transition) {
				out_tokens[i] = -2;
				continue;
			}
			seq->dfa_walk_state = next_state;
		} else {
			produced = superslm::ArgmaxLowestIndexTieBreak(logit_row, static_cast<size_t>(c.vocab_size));
		}
		out_tokens[i] = produced;
		seq->current_token = produced;
		// forward_sites.h: "a sequence resting between whole tokens carries a marker at layer
		// 0" -- reset to the resting convention now that this token is complete.
		seq->state.layer_index = 0;
	}
	return SSLM_OK;
}

extern "C" sslm_status sslm_decode_step(sslm_model model, sslm_seq* seqs, int32_t n,
                                         const sslm_decode_params* params, sslm_workspace ws,
                                         int32_t* out_tokens) {
	return CatchAllocationFailure([&]() -> sslm_status {
		return sslm_decode_stepImpl(model, seqs, n, params, ws, out_tokens);
	});
}

namespace {
// D-SLM3476 (design Sec14.1, Claude/Poirot/9bc9ec6-t2132-g5-arc-review.md S2): true iff `state`
// is a member of `entry`'s own accept set (accepting_le, strictly ascending -- SchemaMasksTable::
// Parse's own validated invariant, schema_masks.h). Binary search, matching Transition's own
// CSR-row lookup style; never reads decode's own internal state, only the schema's own compiled
// accept set.
bool IsAcceptingState(const superslm::SchemaEntry& entry, uint32_t state) {
	using superslm::schema_masks_detail::ReadLE32;
	uint32_t lo = 0, hi = entry.accepting_count;
	while (lo < hi) {
		const uint32_t mid = lo + (hi - lo) / 2;
		const uint32_t v = ReadLE32(entry.accepting_le + static_cast<size_t>(mid) * 4);
		if (v == state) return true;
		if (v < state) lo = mid + 1; else hi = mid;
	}
	return false;
}
}  // namespace

extern "C" sslm_status sslm_stats(sslm_model model, sslm_seq seq, sslm_stats_out* out) {
	if (!out) return SSLM_INVALID_ARGUMENT;
	if (!model || !seq) return SSLM_INVALID_ARGUMENT;
	// design Sec12 ("ceiling + actual work, forced-token count, cache state"). decode_step_ceiling/
	// decode_step_actual are both the model's own num_hidden_layers: this arc's own
	// sslm_decode_step always processes exactly one call's own layer_budget against the SAME
	// worst-case ceiling (a full token, num_hidden_layers), since no partial-work-skipping
	// mechanism (e.g. early-exit) exists in this arc's own scope to make ceiling and actual
	// diverge -- reported honestly as equal, not fabricated apart.
	out->decode_step_ceiling = static_cast<int64_t>(model->view.config.num_hidden_layers);
	out->decode_step_actual = static_cast<int64_t>(model->view.config.num_hidden_layers);
	// G5-3 (design Sec6/Sec7 dim7, T-2132): the actual forced-position count, not the ceiling --
	// seq->forced_token_count is incremented once per token admitted through a
	// SSLM_SPAN_SCHEMA_CONTENT prefill call (PrefillWholeTokensImpl, above), never fabricated
	// from the schema binding alone. A sequence with no schema bound (or one bound but never
	// driven through a forced/fixed span) correctly reports 0.
	out->forced_token_count = seq->forced_token_count;
	out->kv_blocks_resident = 1;  // this sequence's own single, whole-sequence block (Sec7.2)
	// D-SLM3476 (design Sec14.1): 0 when no schema is bound; otherwise 1 iff dfa_walk_state is
	// currently a member of the bound schema's own accept set.
	out->schema_accepting = 0;
	if (seq->bound_schema) {
		const superslm::SchemaEntry* entry = model->schemas.ByIndex(seq->bound_schema->index);
		if (entry && IsAcceptingState(*entry, seq->dfa_walk_state)) {
			out->schema_accepting = 1;
		}
	}
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
//
// 'SSB2' (M4, D-SLM3486, design Sec7.3 -- the G5 blockers round's own M4, ruled): the standing
// law above IS exercised here, not re-excepted. `forced_token_count` (int64_t, `sslm_stats_out`'s
// own field, above) joins the blob -- it did not survive save/restore before this fold, silently
// resetting to 0 on every restore, which the M4 finding named and correctly declined to fix
// in-place (the current_token ruling's own closing sentence forecloses a second 'SSB1' amendment).
// Placed after `kv_saturation_count`, grouped with the other fixed-width per-sequence scalar
// counters it belongs with by kind -- Sec7.3 states plainly there is no safe-insertion-point
// constraint to respect for a genuinely new format. Restore-side validation is the blob's own
// existing size-sufficiency check only (no domain constraint narrower than "any non-negative
// int64_t", matching `current_token`'s own precedent) -- see sslm_seq_restore's own magic check,
// below, for the "reject 'SSB1' outright, never default the field" half of this ruling.
// -----------------------------------------------------------------------------------------

namespace {

// D-SLM3486 (design Sec7.3, M4): magic bumped 'SSB1' -> 'SSB2' for forced_token_count joining the
// blob -- the standing magic-per-version law, not a second in-place amendment. A 'SSB1'-magic blob
// (or any other non-'SSB2' magic) is rejected outright on the memcmp check in sslm_seq_restore,
// below, never parsed as SSB2 and never left to default the new field to 0.
constexpr uint8_t kSeqBlobMagic[4] = {'S', 'S', 'B', '2'};
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

// Fixed-size prefix, magic through forced_token_count -- design Sec7.3's own field order
// ('SSB2', D-SLM3486 -- see this block's own top comment; current_token AMENDED IN under 'SSB1',
// design commit 9e2995f4e7):
// magic(4) + model_hash(32) + kv_precision(4) + schema_name_hash(8) + dfa_walk_state(4) +
// adapter_binding_id(8) + context_length(8) + layer_index(4) + current_token(4) +
// hidden_scale(16) + kv_saturation_count(8) + forced_token_count(8) = 108 bytes, followed by the
// variable-length residual, kv_block_count(4), then kv_blocks.
constexpr size_t kSeqBlobFixedHeaderBytes = 108;

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
	// G5 (design Sec13.4, T-2132): schema_name_hash is Fnv1a64(bound schema's own name), or 0
	// when SSLM_SCHEMA_NONE is bound -- 0 is never a legitimate Fnv1a64 output for this
	// project's own name-hash convention with any realistic probability, and restore's own
	// resolution (below) treats a stored 0 as "no schema" without a name-blob scan, matching
	// this file's own "0 == not applicable" sentinel convention (kSeqBlobNoCurrentToken).
	const uint64_t schema_name_hash =
	    seq->bound_schema ? model->schemas.ByIndex(seq->bound_schema->index)->name_hash : 0;
	WriteLE64(p + off, schema_name_hash);
	off += 8;
	WriteLE32(p + off, seq->dfa_walk_state);  // dfa_walk_state -- design Sec13.4
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
	// forced_token_count (design Sec7.3, D-SLM3486, 'SSB2'): sslm_stats_out's own field,
	// byte-identical semantics -- written verbatim, no domain narrowing (any non-negative int64_t
	// is a legal value, matching current_token's own precedent).
	WriteLE64(p + off, static_cast<uint64_t>(seq->forced_token_count));
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
	//
	// D-SLM3486 (design Sec7.3, M4, 'SSB2'): this check is ALSO now the 'SSB1'-rejection --
	// `kSeqBlobMagic` is 'SSB2', so a well-formed 'SSB1' blob (forced_token_count-less, 100-byte
	// fixed header) fails this memcmp and is rejected right here, on the magic check alone, never
	// parsed as SSB2 and never left to silently default forced_token_count to 0 while accepting
	// the rest of an old-format blob. No 'SSB1'-to-'SSB2' upgrade path exists (design Sec7.3's own
	// grounding: no genuinely-shipped 'SSB1' consumer exists to need one).
	if (std::memcmp(p, kSeqBlobMagic, 4) != 0) return SSLM_INVALID_ARGUMENT;

	std::array<uint8_t, 32> saved_hash{};
	std::memcpy(saved_hash.data(), p + 4, 32);
	const std::array<uint8_t, 32> current_hash = model->view.RawIntegrityHash();
	if (saved_hash != current_hash) return SSLM_RESTORE_MODEL_MISMATCH;

	const uint32_t saved_kv_precision = ReadLE32(p + 36);
	if (saved_kv_precision != static_cast<uint32_t>(model->view.config.kv_precision)) {
		return SSLM_RESTORE_KV_MISMATCH;
	}

	// G5 (design Sec5/Sec13.4, T-2132): resolve the save-blob's own schema binding AFTER the
	// model/kv-precision validation above and BEFORE any device work -- specifically, before
	// DrawBlock draws a block from the pool, below, so a rejection here never claims (and never
	// needs to return) pool resources. schema_name_hash == 0 means SSLM_SCHEMA_NONE was bound
	// (this file's own save-side sentinel, above) -- restores as unbound, no name-blob scan. A
	// nonzero hash that does not resolve against this model's OWN schema set -- including a
	// model with no SchemaMasks section at all -- is SSLM_RESTORE_SCHEMA_MISMATCH, symmetric
	// with SSLM_RESTORE_MODEL_MISMATCH's own precedent (design Sec5).
	const uint64_t saved_schema_name_hash = ReadLE64(p + 40);
	const uint32_t saved_dfa_walk_state = ReadLE32(p + 48);
	sslm_schema resolved_schema = nullptr;
	if (saved_schema_name_hash != 0) {
		size_t idx = 0;
		if (!model->schemas.ByNameHash(saved_schema_name_hash, &idx)) {
			return SSLM_RESTORE_SCHEMA_MISMATCH;
		}
		resolved_schema = &model->schema_handles[idx];
		// C1 (Claude/Poirot/9bc9ec6-t2132-g5-arc-review.md): the blob is untrusted input --
		// every other field on this path is checked, and this is the one that was not. Bound
		// the restored walk-state against the RESOLVED schema's own state_count before it is
		// ever used to index mask_pages/state_offsets_le (sslm_decode_step,
		// src/sslm_abi.cpp:1816-1826). kDfaWalkStateUnused (0xFFFFFFFF) is a real reject here,
		// not a pass-through: a bound sequence's walk-state is always a real state id in
		// [0, state_count) (see the kDfaWalkStateUnused-as-"no schema" comments above), so the
		// sentinel arriving on a BOUND blob is exactly the malformed-blob case this check
		// exists for, not the unbound case (that one is handled in the branch below, where
		// resolved_schema stays null and the sentinel is the only value accepted). Rejected as
		// SSLM_RESTORE_SCHEMA_MISMATCH -- the taxonomy's existing status for "the blob's schema
		// binding cannot be honoured against this model", already minted, no new ordinal.
		const superslm::SchemaEntry* resolved_entry = model->schemas.ByIndex(idx);
		if (!resolved_entry || saved_dfa_walk_state >= resolved_entry->state_count) {
			return SSLM_RESTORE_SCHEMA_MISMATCH;
		}
	} else if (saved_dfa_walk_state != kDfaWalkStateUnused) {
		// C1 / O3 (Claude/Poirot/9bc9ec6-t2132-g5-arc-review.md): schema_name_hash == 0 means
		// "no schema bound" (this file's own save-side sentinel). A non-sentinel walk-state on
		// an unbound blob is symmetric malformation -- reject it here rather than admit a
		// restored sequence that holds an unbound binding with a non-sentinel walk-state, which
		// used to reject its own later sslm_seq_set_schema call (O3) instead of being rejected
		// at the point the bad blob was actually read.
		return SSLM_RESTORE_SCHEMA_MISMATCH;
	}

	const int64_t context_length = static_cast<int64_t>(ReadLE64(p + 60));
	const uint32_t layer_index = ReadLE32(p + 68);
	const int32_t saved_current_token = static_cast<int32_t>(ReadLE32(p + 72));  // design commit
	                                                                              // 9e2995f4e7
	const int64_t hidden_scale_m = static_cast<int64_t>(ReadLE64(p + 76));
	const int64_t hidden_scale_e = static_cast<int64_t>(ReadLE64(p + 84));
	const uint64_t kv_saturation_count = ReadLE64(p + 92);
	// forced_token_count (design Sec7.3, D-SLM3486, 'SSB2'): read verbatim -- no domain
	// constraint beyond the blob's own existing size-sufficiency check (below), matching
	// current_token's own precedent (see this function's own magic-check comment, above, for the
	// 'SSB1'-rejection half of this ruling).
	const int64_t saved_forced_token_count = static_cast<int64_t>(ReadLE64(p + 100));

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
	// P1 -- see sslm_prefix_begin's own identical fix.
	const sslm_status assign_st = CatchAllocationFailure([&]() -> sslm_status {
		h->hidden_codes_storage.assign(c.hidden_size, 0);
		return SSLM_OK;
	});
	if (assign_st != SSLM_OK) {
		ReturnBlock(pool_ptr, block_index, h->kv_block, pool_ptr->block_size);
		delete h;
		return assign_st;
	}
	if (residual_len > 0) {
		std::memcpy(h->hidden_codes_storage.data(), residual_ptr, residual_len);
	}
	h->state.hidden_codes = h->hidden_codes_storage.data();
	h->state.hidden_scale = superslm::CarriedScale{hidden_scale_m, hidden_scale_e};
	h->state.layer_index = layer_index;
	h->state.kv_saturation_count = kv_saturation_count;
	h->state.context_length = context_length;
	// forced_token_count (design Sec7.3, D-SLM3486, 'SSB2'): restored verbatim, bit-equal --
	// closes M4 (Claude/Brunel/t2132-g5-build-2026-08-16.md), which found this counter silently
	// reset to 0 on every restore. Proven by tools/t2132_m4_forced_token_count_pin.cpp.
	h->forced_token_count = saved_forced_token_count;
	// G5 (design Sec5/Sec13.4, T-2132): the binding/walk-state resolved above -- restored
	// verbatim, bit-equal (design Sec7 dim9's own round-trip cell).
	h->bound_schema = resolved_schema;
	h->dfa_walk_state = saved_dfa_walk_state;
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
		// FOLD RULING on F2 -- see sslm_model_map's own identical comment and
		// CatchAllocationFailure's own comment for the full reasoning.
		return SSLM_ALLOCATION_FAILED;
	} catch (const std::length_error&) {
		// FOLD RULING, second pass, D-SLM3464 -- see sslm_model_map's own identical comment: dead
		// in practice on this call path, same `WrapBadAllocContract` narrowing reason (this verb
		// is also fronted by `SslmModel::Load`'s own public entry).
		return SSLM_ALLOCATION_FAILED;
	} catch (...) {
		// FOLD RULING, second pass, D-SLM3464 -- see sslm_model_map's own identical comment: this
		// verb's own delivered contract for any std::exception-derived internal cause is
		// SSLM_ALLOCATION_FAILED (WrapBadAllocContract's narrowing, proven by real fault injection
		// against this exact call path), narrower than CatchAllocationFailure's general rule; this
		// arm stays live only for throws not derived from std::exception at all.
		return SSLM_ARTIFACT_REJECTED;
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
	//
	// P1 (Sec7.3, third confirmation pass): the sweep found PopulateAdapterFromView
	// (adapter_marshal.h) allocates the adapter's own per-layer delta arrays -- can throw past
	// every check above it. Wrapped; `h` is deleted on an allocation failure exactly as it
	// already is on either of PopulateAdapterFromView's own ordinary rejections below.
	superslm_adapter::AdapterLoadStatus populate_st = superslm_adapter::AdapterLoadStatus::Ok;
	const sslm_status populate_alloc_st = CatchAllocationFailure([&]() -> sslm_status {
		// D-SLM3466's owed pin (Claude/Poirot/3bcbe43-t2139-fourth-confirmation-review.md S2/S3):
		// consulted at THIS step's own entry, matching BuildEngineCache's own identical convention
		// (above, this file) -- the SITE-SPECIFIC post-load-region slot, independent of the plain
		// slot this verb's own SslmModel::Load call (above) consults. Proven by
		// tools/t2139_d3466_postload_region_pin.cpp.
		superslm::internal::MaybeThrowInjectedBadAllocFaultPostLoadRegion();
		populate_st = superslm_adapter::PopulateAdapterFromView(
		    h->handle.view_, base_geom, h->handle.meta, h->handle.layer_adapters, &err);
		return SSLM_OK;
	});
	if (populate_alloc_st != SSLM_OK) {
		delete h;
		return populate_alloc_st;
	}
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
	// P1 (Claude/Poirot/2c18dab-t2139-abi-build-review.md Sec7.3, third confirmation pass): THIS
	// COMMENT PREVIOUSLY ARGUED THE EXACT RATIONALE N3 REFUTED, on this same file, in this same
	// commit's own earlier round -- "does not catch it, matching sslm_model_map's own house-
	// convention disposition ... sslm_status carries no resource-exhaustion member" was already
	// false the moment N3 gave sslm_model_map a try/catch and sslm_status a resource-exhaustion
	// member (SSLM_ALLOCATION_FAILED, ordinal 25, ratified design commit 9f84d9e4ca). Fixed for
	// real now, not merely re-worded: TokenizerView::Encode "throws only std::bad_alloc"
	// (tokenizer.h's own contract) is caught here, same as every other documented-throwing call
	// this sweep found.
	std::vector<int32_t> ids;
	const sslm_status encode_st = CatchAllocationFailure([&]() -> sslm_status {
		ids = model->view.tokenizer.Encode(std::string_view(utf8));
		return SSLM_OK;
	});
	if (encode_st != SSLM_OK) {
		*n = 0;
		return encode_st;
	}
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

	// P1 (Claude/Poirot/2c18dab-t2139-abi-build-review.md Sec7.3, third confirmation pass): named
	// finding -- `tokenizer.Decode(ids)` ("throws only std::bad_alloc", tokenizer.h's own
	// contract, same as Encode above) plus this function's own `std::vector`/`std::string`
	// construction were all previously uncaught. Wrapped as one block: `combined`/`hold`/
	// `emit_len` need to stay visible below (the buffer-size check and the state update both
	// read them), so they are declared outside the lambda and assigned inside it, matching the
	// same outer-declared/inner-assigned shape sslm_tokenize's own `ids` fix uses above.
	std::string combined;
	size_t hold = 0;
	size_t emit_len = 0;
	const sslm_status decode_st = CatchAllocationFailure([&]() -> sslm_status {
		const std::vector<int32_t> ids(tokens, tokens + n);
		const std::string decoded = model->view.tokenizer.Decode(ids);
		combined.reserve(static_cast<size_t>(state->pending_count) + decoded.size());
		combined.append(reinterpret_cast<const char*>(state->pending_bytes), state->pending_count);
		combined.append(decoded);
		hold = CountIncompleteTrailingUtf8(combined);
		emit_len = combined.size() - hold;
		return SSLM_OK;
	});
	if (decode_st != SSLM_OK) {
		*out_n = 0;
		return decode_st;
	}

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

// -----------------------------------------------------------------------------------------
// G5-2's own test-only guard-vitality hook (design Sec11.2, BLESSED with a structural
// ship-boundary -- Claude/Vitruvius/t2119-rung7-fold-2026-08-16.md, Wizard repo, ruling 2).
// Declared ONLY in tests/t2130-g5-red-suite/sslm_g5.h (this suite's own private header), NEVER
// in include/superslm/ -- deliberately absent from sslm_abi.h/sslm_abi_functions*.inc, so it
// carries no counterpart in this repo's own install/export rule set at all (an entry it is
// never added to, not an entry removed later, per the ruling's own "install-list curation"
// remedy). Its entire body is a call to superslm::ApplyMaskAndArgmax (forward_sites.h, relocated
// this fold from this file's own file-local ApplyMaskAndArgmaxImpl, G5-5/T-2132) -- the SAME
// primitive sslm_decode_step's masked-argmax step (above) AND the GPU-1.0 parity bridge
// (src/gpu/gpu_1p0.cpp) call -- never a parallel reimplementation, per the ruling's own contract:
// a passing negative control here proves something about the production guard only because this
// hook exercises the production guard's own code, not a lookalike.
// -----------------------------------------------------------------------------------------
extern "C" sslm_status sslm_g5_test_only_apply_mask_and_argmax(const int32_t* logits,
                                                                 const uint8_t* mask,
                                                                 int32_t vocab_size,
                                                                 int32_t* out_token_id) {
	if (!logits || !mask || vocab_size < 1 || !out_token_id) return SSLM_INVALID_ARGUMENT;
	// ApplyMaskAndArgmax mutates `logits` in place (masked-out positions become INT32_MIN) --
	// this hook's own caller (dim11's negative-control cell) does not rely on `logits` surviving
	// the call, matching the production decode_step path's own identical use of its scratch
	// `logit_row` (never read again after this step there either).
	superslm::ApplyMaskAndArgmax(const_cast<int32_t*>(logits), mask, vocab_size, out_token_id);
	return SSLM_OK;
}

// -----------------------------------------------------------------------------------------
// S4 (Claude/Poirot/9bc9ec6-t2132-g5-arc-review.md): a test-only pool-level peek hook,
// following the EXACT precedent this file's own sslm_g5_test_only_apply_mask_and_argmax already
// established one function above (design Sec11.2's own "test-only guard-vitality hook, BLESSED
// with a structural ship-boundary") -- not declared anywhere in include/superslm/, no
// install/export entry, callable only by a tool that declares this exact extern "C" signature
// itself (this file's own compiled .obj is the only definition; nothing routes a real host to
// it).
//
// WHY THIS EXISTS: T-2132's own zero-fill fix (DrawBlock, above) made ReturnBlock's poison-fill
// unobservable through the PRODUCTION read path -- every live sequence's own KV block is reached
// via DrawBlock, which now zero-fills unconditionally, so ANY dimension-1 leak-check cell that
// reads a block through a live sequence handle passes whether or not ReturnBlock ever poisoned
// anything at all (Poirot's own finding: "the guard is now one that cannot fail"). The leak
// obligation itself (Sec7 dim1: "no content from the PRIOR sequence survives release") did not
// get weaker -- it got STRONGER, structurally guaranteed by the zero-fill rather than merely
// detectable against a poison pattern -- but a guard that structurally cannot fail is also a
// guard nothing can exercise, and Poirot's own remedy (S4) asks for a discriminating mechanism to
// be restored or the claim to be honestly retired.
//
// THE MECHANISM: reads `n` raw bytes directly from `pool`'s own backing buffer at `block_index`,
// BYPASSING DrawBlock entirely -- the one read path in this whole file that does NOT run through
// the zero-fill. A test can therefore: draw a block, write real content into it (a real decode),
// release it (ReturnBlock's own poison-fill runs), then peek the SAME block_index through THIS
// hook BEFORE drawing it again -- proving the poison-fill actually ran (0xCD observed) rather
// than merely trusting the source. This restores exactly the discrimination S4 asks for: DELETE
// ReturnBlock's own `std::memset(kv_block, 0xCD, block_size)` line and this peek would read
// whatever ReturnBlock's caller left behind instead of 0xCD -- a real, demonstrated failure mode
// for the pin built against this hook (tools/t2132_s4_leak_guard_mutation_pin.cpp), not an
// assertion that cannot be tripped. Bounds-checked against `pool->buf_size`/`block_count` --
// hostile-input-safe like every other verb in this file, even though only a test author is
// expected to call it.
extern "C" sslm_status sslm_g5_test_only_peek_kv_block_bytes(sslm_kv_pool pool,
                                                               uint32_t block_index, uint8_t* out_buf,
                                                               size_t n) {
	if (!pool || !out_buf) return SSLM_INVALID_ARGUMENT;
	if (block_index >= pool->block_count) return SSLM_INVALID_ARGUMENT;
	if (n > pool->block_size) return SSLM_INVALID_ARGUMENT;
	const uint8_t* src =
	    static_cast<const uint8_t*>(pool->buf) + static_cast<size_t>(block_index) * pool->block_size;
	std::memcpy(out_buf, src, n);
	return SSLM_OK;
}

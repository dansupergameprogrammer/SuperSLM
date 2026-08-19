// T-2113 (B1/B2/B3): production implementation of the 1.0 GPU API's context, model,
// and sequence lifecycles (Claude/Vitruvius/t2107-gpu-core-1p0-design-2026-08-14.md
// Sec4.1.1/Sec5.1/Sec5.3, Sec10 B1/B2/B3).
//
// B1's own gate (design Sec10): "existing suite green, unchanged behavior when
// exactly one context is created (a same-process, same-thread equivalence check
// against the substrate's own current single-call behavior)". Concretely: this file
// adds a NEW, additive construction (SslmGpuContext, its own owned harness::Device
// and PSO/root-sig caches) that touches none of the pre-1.0 substrate's own process-
// global statics (`harness::GetDevice()`, `harness::GetOrBuildPipeline`,
// `harness::GetOrBuildComposedPipeline`, all still defined in d3d12_harness.h and
// still used unchanged by `RunLayerLoopGpu` and every other pre-1.0 entry point in
// superslm_gpu.cpp) -- so the existing test suite (tests\test_main.cpp,
// out\superslm_tests.exe) observes zero behavior change. The migration that actually
// removes the pre-1.0 singleton usage from RunLayerLoopGpu's own dispatch-recording
// path is B2/B4's job (design Sec10: B2 folds the T-2105 residency construction onto
// the model handle's own upload path; B4 ports the dispatch geometry onto the new
// handle shapes) -- B1 only builds the object the later sections will route weights,
// PSOs, and dispatches through instead of the singleton, per D-SLM3294's own
// "no process-global device state in the 1.0 backend" requirement (design Sec1).
//
// Symbols DEFINED here: sslm_gpu_context_create, sslm_gpu_context_destroy.
// Every other declaration in include/superslm/gpu_1p0.h stays declared-only until
// its own B-section lands (B2 through B9); this file is the single production TU
// the whole 1.0 API grows in, one B-section at a time -- calling out its own
// section boundary in comments as later sections extend it, mirroring
// src/gpu/superslm_gpu.cpp's own established convention for the pre-1.0 surface.

#include "superslm/gpu_1p0.h"
#include "superslm/gpu_1p0_bench_bridge.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdio>
#include <cstring>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "d3d12_harness.h"
#include "superslm/adapter_marshal.h"  // T-2113 B6: LoadAdapterArtifact (relocated, tools/sslm_adapter_loader.h)
#include "superslm/gpu_1p0_g5_bridge.h"  // G5-5 (T-2132): the build-seat-owned schema/parity bridge
#include "superslm/gpu_port.h"       // T-2113 B2: GpuLayerLayout/ComputeLayerLayout/PackLayerWeightsBytes
#include "superslm/layer_marshal.h"  // T-2113 B2: MarshalLayer/LayerBacking (relocated, tools/sslm_marshal.h)
#include "superslm/model.h"          // T-2113 B2: the REAL superslm::SslmModelView this TU needs by value
#include "superslm/schema_masks.h"   // G5-5 (T-2132): SchemaMasksTable -- the SAME SCM1 reader the CPU
                                      // ABI (sslm_abi.cpp) already uses, reused here, not re-derived.

// The real SslmGpuContext this handle type opaquely names to every 1.0 API caller.
// Owns everything B1's own gate requires be OWNED rather than reached through a
// process-global name: its own harness::Device (never harness::GetDevice()'s
// static), and its own PSO/root-signature caches, keyed exactly the way the
// pre-1.0 singleton caches were (by shader base name) so the later B-sections that
// port dispatch-recording code onto this object can reuse that code's own lookup
// shape unchanged -- only the cache's OWNER moves.
// T-2113 (B8, design Sec5.4/Sec10 B8): thread-safety contract, stated here because this is
// where every thread-shared object this design names actually lives.
//
//   - The PSO/root-signature caches above and `device` are immutable-after-populate for the
//     RESIDENCY half of this object's life (map/create time) and are never touched by the
//     decode dispatch path at all (see the next bullet) -- reading them concurrently is safe.
//   - `SslmGpuModelHandle`/`SslmGpuAdapterHandle`: read-only for the whole of their lifetime
//     after `map` returns (design Sec5.4) -- safe for any number of threads to read (bind,
//     dispatch against, or call `sslm_gpu_seq_embed_token` through) concurrently, by
//     construction, since nothing ever writes to resident weight/adapter/embed buffers again.
//   - `SslmGpuSequenceHandle`: owns state private to itself. Two threads each driving a
//     DIFFERENT sequence handle through `sslm_decode_step_gpu`/`sslm_gpu_ready`/
//     `sslm_gpu_seq_embed_token` touch disjoint per-sequence memory -- this is "thread-safe
//     execution over disjoint sequences" (D-SLM3294), true by construction once the handle's
//     own ownership split holds. Two threads driving the SAME sequence handle concurrently is
//     a caller error this design does not guard against structurally (proven, not assumed: the
//     T-2112 suite's own dim3 M2 cell reproduces a real D3D12 device fault doing exactly this,
//     `Claude/Brunel/t2113-1p0-core-build-2026-08-15.md` Sec11.4) -- named as a residual, never
//     silently treated as safe.
//   - **The one thing this design deliberately does NOT make safe without caller help**: the
//     actual GPU submission every `sslm_decode_step_gpu`/`_batch_gpu`/`sslm_gpu_ready(block=1)`
//     call performs. Design Sec5.4: "the command queue and descriptor-heap allocator are not
//     safe to submit-from or allocate-from concurrently without external synchronization; this
//     design does not build an internal queue-level lock (Sec13's own deferral: 'speculative
//     concurrency infrastructure the product does not yet need') -- a caller driving two
//     threads' decode calls through one context's queue concurrently must serialize the submit
//     call itself." **Grounded, not merely quoted from the design**: as of B5-B7, the decode
//     dispatch path (`RunLayerLoopGpuSubmit`/`RunLayerLoopGpuFinish`, superslm_gpu.cpp) does not
//     even route through THIS context's own `device` above -- it still calls the pre-1.0
//     substrate's process-WIDE `harness::GetDevice()` singleton (`SubmitOneSequenceDecode`'s own
//     `(void)ctx;` -- ctx is not read for dispatch purposes at all, only for residency lookups
//     upstream of it). So the serialization requirement is, today, PROCESS-wide, not merely
//     per-context: any two threads anywhere in the process that call a decode-touching function
//     concurrently, on any context, must be externally serialized at that call boundary, or the
//     shared allocator's `Reset()` races exactly the way the undrained-batch hang (D-SLM3384,
//     `Claude/Brunel/t2113-1p0-core-build-2026-08-15.md` Sec12.2) already proved by execution --
//     that hang is this same hazard's sequential-but-undrained shape; two genuinely concurrent
//     threads racing the identical `Reset()` is the stronger, race-condition form of it. B8's own
//     bench tool (`tools/t2113_b8_thread_smoke.cpp`) is built to this documented discipline: a
//     mutex external to this API wraps exactly the decode-submit-and-drain call pair, nothing
//     more -- proving the design's own "disjoint sequences, externally serialized only at the
//     queue-submit boundary" claim holds, without this file adding the internal lock the design
//     explicitly declines to build (Sec13's own deferral row, dated, routed: "if evidence arrives
//     that a caller needs concurrent submission through one shared queue... add the lock... as
//     its own scoped follow-on, not retrofitted here"). Migrating the decode dispatch path onto
//     this context's OWN `device` (making the requirement genuinely per-context rather than
//     process-wide) is out of B8's own scope -- named here, dated, the same class of routed item
//     `g_resident_rope`'s own retirement note already established (Sec6.3 of the build log).
struct SslmGpuContext {
	superslm_gpu::harness::Device device;

	// Mirrors harness::GetOrBuildPipeline's cache shape (one-SRV-one-UAV B1
	// primitive-battery pipelines), now owned per-context instead of a function-
	// local static. Unused until a later B-section routes a call through it;
	// present now so B1 delivers the object the design's own gate names ("PSO/
	// root-sig cache moved off harness::GetDevice()'s static onto the context
	// object") as a real, populatable member, not a placeholder added later.
	std::map<std::string, superslm_gpu::harness::CachedPipeline> simple_pipeline_cache;

	// Mirrors harness::GetOrBuildComposedPipeline's cache shape (the 14-shader
	// composed per-layer-dispatch pipeline family) plus its own single shared
	// root signature, owned per-context for the identical reason.
	std::map<std::string, superslm_gpu::harness::CachedPipeline> composed_pipeline_cache;
	Microsoft::WRL::ComPtr<ID3D12RootSignature> composed_root_sig;

	// ContextHasLiveHandles guard (design Sec4.1.1, Sec9): incremented by every
	// model/adapter/sequence handle this context issues (B2/B3/B6, not yet wired
	// -- no such handle type is constructible yet), decremented on release.
	// destroy() refuses while this is nonzero.
	int64_t live_handles = 0;

	// Set once destroy() has freed this object's own device/GPU state, so a
	// caller who somehow retains a dangling pointer past a successful destroy
	// cannot be silently handed a half-torn-down object by a second call through
	// the same pointer (destroy is not re-entrant against its own prior success;
	// the object itself is deleted at the end of a successful destroy, so this
	// flag is defensive documentation of that invariant rather than a runtime
	// branch anything reads today).
	bool destroyed = false;
};

// T-2113 (B2, design Sec5.1/Sec10 B2): the real SslmGpuModelHandle. Owns the model's own
// weight residency and the T-2105 RoPE cos/sin table residency (Claude/Laplace/
// t2105-gpu-speed-ceiling-2026-08-14.md Sec2 change 1, re-derived here per the design's
// own "re-derived, never cherry-picked" instruction, Sec1 constraints paragraph) --
// uploaded ONCE, inside map(), into DEFAULT-heap buffers this handle owns for its whole
// lifetime (design Sec5.1: "weight buffers upload once... and stay resident for the mapped
// model handle's entire lifetime"). Keyed by the artifact's own content hash
// (SslmModelView::RawIntegrityHash(), design Sec5.1) rather than by the caller's host
// pointer -- this is the structural fix D-SLM3311 names (Sec2.1): identity is "which
// handle," never "which address the allocator happened to hand back." No process-global
// weight/RoPE cache exists anywhere in this file (D-SLM3294) -- every sslm_gpu_model_map
// call uploads its OWN dedicated residency, independent of any other live handle, matching
// design Sec5.1's own "multiple SslmGpuModelHandles may be live at once... nothing in this
// design imposes a one-model-at-a-time limit."
//
// The T-2105 "root-binding-hoist" half of this section's own delivery line (design Sec10
// B2) is NOT built here: hoisting the twelve per-dispatch SetComputeRoot* calls out of a
// dispatch-recording loop has nothing to hoist out of until B4 builds that loop -- this is
// named explicitly, not silently dropped, per StandardsDocument.md Sec5.6 (a deferral is
// surfaced loudly). B4's own gate (design Sec10) is where the composed dispatch chain --
// and therefore the rebind site this handle's own resident buffers get bound at -- first
// exists; see the B2 section of Claude/Brunel/t2113-1p0-core-build-2026-08-15.md.
struct SslmGpuModelHandle {
	SslmGpuContext* ctx = nullptr;
	std::array<uint8_t, superslm::kIntegrityHashBytes> content_hash{};

	// Resident, DEFAULT-heap, read-only for this handle's whole lifetime (design Sec5.4:
	// "read-only for the whole of their lifetime after map returns... safe for any number
	// of threads to read concurrently, by construction").
	Microsoft::WRL::ComPtr<ID3D12Resource> weights_buf;
	Microsoft::WRL::ComPtr<ID3D12Resource> rope_cos_buf;
	Microsoft::WRL::ComPtr<ID3D12Resource> rope_sin_buf;
	bool has_rope_tables = false;  // true iff the artifact's own rope_tables.Tensor("cos"/"sin") existed
	// T-2113 (B5): the RoPE presence/element-count metadata `RunLayerLoopGpuSubmit`'s own
	// external-rope bridge needs at every decode call -- this handle keeps no live
	// SslmTensorManifest past map() returning (map()'s own locals, above, go out of scope),
	// so these three small facts are captured here, once, instead of re-deriving them from a
	// manifest sslm_decode_step_gpu has no access to.
	uint64_t rope_cos_elem_count = 0;
	uint64_t rope_sin_elem_count = 0;

	// The layout/dimension set this handle's own weights_buf was packed at (superslm_gpu::
	// GpuLayerLayout, T-2113 B2's own shared PackLayerWeightsBytes) -- B4 reads these to
	// bind the composed dispatch chain's own per-layer stride/offsets against this handle's
	// buffer without re-deriving them from the model view a second time.
	superslm_gpu::GpuLayerLayout layout{};
	uint32_t num_hidden_layers = 0;
	uint32_t hidden_size = 0;
	uint32_t head_dim = 0;
	uint32_t num_key_value_heads = 0;
	uint32_t num_attention_heads = 0;
	uint32_t intermediate_size = 0;
	int64_t context_cap = 0;

	// ModelHasLiveSequences guard (design Sec9, Sec4.2's own model_unmap row): T-2113 (B3)
	// incremented by sslm_gpu_seq_create, decremented on release, below.
	int64_t live_sequences = 0;

	// T-2113 (design Sec4.2/Sec9, routed `Claude/Curie/t2112-1p0-red-suite-2026-08-15.md` Sec7.2
	// item 2, D-SLM3417): design Sec9's own Busy row states `sslm_gpu_model_unmap` returns Busy
	// "against a model with any Submitted sequence bound... which takes precedence" over
	// ModelHasLiveSequences' own "any state, not only Submitted" caveat -- `live_sequences` alone
	// cannot distinguish a Submitted-bound model from an Idle-bound one. Incremented in
	// SubmitOneSequenceDecode (below) the instant a bound sequence transitions Idle -> Submitted;
	// decremented in sslm_gpu_ready the instant that SAME sequence collapses back to Idle. Every
	// sequence that ever transitions to Submitted is, by construction, still bound to exactly the
	// model it was created against (a sequence's own `model` pointer never changes across its
	// lifetime) -- so this count is always a count of THIS model's own Submitted sequences.
	//
	// T-2124 (D-SLM3446 S2, Claude/Poirot/435f730-t2124-adapter-uaf-review.md): `std::atomic`, not
	// a plain `int64_t` -- design Sec5.4 blesses two threads each driving a DIFFERENT sequence
	// handle concurrently, and Sec6.2's own Submit/Finish split is built precisely so the submit
	// (writer: SubmitOneSequenceDecode) and the drain (writer: sslm_gpu_ready) can happen on
	// different threads. A plain `+= 1`/`-= 1` is a non-atomic read-modify-write; two sequences
	// sharing one model, driven from two threads, can race it -- a lost increment reopens the
	// model-handle analogue of this ticket's own adapter UAF class (the guard below reads 0 while
	// a sequence is genuinely Submitted), a lost decrement makes the model permanently un-unmappable.
	// `seq_cst` (the type's own default for every operator used below: `+=`, `-=`, `> 0`, `load()`)
	// is kept deliberately rather than relaxed to `acquire`/`release`: this counter gates a
	// correctness precondition (whether it is safe to free device memory the GPU may still read),
	// not a performance-sensitive hot path -- one Submit/Finish pair per decode step, not per
	// dispatch -- so there is no measured cost this file's own conventions (Sec12, performance
	// claims anchored to execution) would accept trading away the simplest-to-reason-about ordering
	// for. A weaker order would need its own argued happens-before edge to the buffer-free this
	// counter gates; none is on record, so none is assumed.
	std::atomic<int64_t> submitted_sequences{0};

	// T-2113 (B3.5, design Sec5.1's own amendment / Sec5.3a): host-side metadata
	// sslm_gpu_seq_embed_token needs for its whole lifetime -- NOT new GPU residency (the
	// design's own explicit "not new residency" clause, Sec5.3a): the embedding matrix
	// itself has no dispatch site and is never bound to a command list, so it stays a
	// plain host-side copy, taken once at map() time (mirroring the packed weight bytes'
	// own "upload once, inside map" discipline, but never uploaded to a device buffer
	// since nothing ever reads it from a shader). Copied rather than pointed at `base`
	// (the SslmModelView passed to sslm_gpu_model_map) because that view's own lifetime is
	// caller-owned and this handle must serve sslm_gpu_seq_embed_token for its OWN whole
	// lifetime, which can outlive the caller's view -- the identical reasoning that
	// already governs why LayerWeights/lw_bytes above are marshaled and packed at map()
	// time rather than read from `base` lazily per call.
	std::vector<int8_t> embed_weights;
	superslm::CarriedScale embed_site_constant{};
	int32_t vocab_size = 0;

	// G5-5 (T-2132, Brunel): mask-page residency on device (design Sec6 G5-5's own "Builds"
	// list, VRAM, alongside weights/scales per the existing GPU-tenancy model, Sec8.4) --
	// uploaded, read-only, once here, the same UploadResidentBufferSync three-step shape
	// weights_buf/rope_cos_buf/rope_sin_buf already use above. Empty/null when this model's
	// artifact carries no SchemaMasks section (an unconstrained-only artifact, design Sec13.1) --
	// the same "absence loads exactly as before this section type existed" disposition the CPU
	// ABI's own sslm_model_map already applies (src/sslm_abi.cpp).
	Microsoft::WRL::ComPtr<ID3D12Resource> mask_pages_buf;
	// Host-owned copy of the SchemaMasks section's own bytes -- `schemas`'s own SchemaEntry
	// pointers point INTO this vector (schema_masks.h's own "entries point into section_data,
	// the caller must keep the bytes alive" contract), so this handle owns them for its whole
	// lifetime, exactly the "copied rather than pointed at `base`" reasoning this file's own
	// embed_weights capture (B3.5, above) already documents for the identical caller-view-
	// lifetime hazard.
	std::vector<uint8_t> schema_section_bytes;
	// The SAME SchemaMasksTable reader the CPU ABI parses this model's own SchemaMasks section
	// with (schema_masks.h) -- reused, not re-derived, so a schema's mask pages/CSR transitions
	// mean the identical bits on both paths. Count()==0 when this model carries no schema.
	superslm::SchemaMasksTable schemas;
	// G5-5's own final_norm+logits capture (design's "no new arithmetic" claim: the GPU-1.0
	// substrate's job is the layer loop only, design Sec4/Sec12's own established scope -- the
	// finish-token bridge below needs these three host-resident facts to run the SAME
	// RmsNormSite/LogitsSite calls sslm_decode_step's own finishing block already runs, on this
	// model's own GPU-derived hidden state). Mirrors embed_weights/embed_site_constant's own
	// capture immediately above -- read from `*base` once, at map time, never re-read from the
	// caller's view afterward (the identical caller-view-lifetime reasoning).
	std::vector<int32_t> final_norm_gain;
	superslm::CarriedScale final_norm_site_constant{};
	std::vector<int8_t> head_weights;

	// T-2114 (S2, Claude/Poirot/50f3d5d-t2113-1p0-gpu-core-build-review.md): set at the START
	// of unmap(), before `delete this`, matching `SslmGpuContext::destroyed`'s own established
	// disposition (this file, above) -- defensive documentation of "this object's own release
	// has begun," not a runtime branch anything reads. It CANNOT be, structurally: every call
	// site that used to test `->destroyed` on this handle read it AFTER `delete`, which is a
	// use-after-free (undefined behavior, not a check) on a released handle, and is always
	// false -- vacuously -- on a live one, since the flag is set only the line before delete.
	// The twelve call sites this used to gate (across model/adapter/sequence handles) had their
	// own `->destroyed` reads removed in the same fix round: a caller that passes an already-
	// released handle to any 1.0 entry point again invokes undefined behavior, exactly as
	// passing a freed pointer to any C API does -- this was already true before S2 (the guard
	// never protected against it), S2 only removes the code that pretended otherwise.
	bool destroyed = false;
};

// T-2113 (B6, design Sec5.2/Sec10 B6): the real SslmGpuAdapterHandle. Owns the adapter's own
// device residency, independent of any sequence (design Sec5.2: "resident and shared read-only
// by every sequence that later binds this handle"). Mirrors the CPU-side `LoadAdapterArtifact`/
// `AdapterHandle` shape (include/superslm/adapter_marshal.h, relocated from tools/
// sslm_adapter_loader.h this section, D-SLM3368) exactly, on the device side, per design Sec5.2's
// own instruction. `model` is retained (not merely its content hash) so `sslm_decode_step_gpu`'s
// own AdapterModelMismatch check (design Sec5.2: "a pointer-equality check against the handle
// each side already carries, not a re-hash") is a real pointer compare, never a re-hash.
//
// Residency shape this section builds: every (layer, projection) pair the adapter's own
// DeltaFoldScales section covers gets its lora_A/lora_B bytes and DeltaFoldScales/UFoldScales
// (identity, mult, shift) triples packed into three DEFAULT-heap resident buffers (mirroring the
// model handle's own three-buffer split, weights/rope_cos/rope_sin) -- uploaded ONCE, here, never
// per decode call. `AdapterProjSlot` records each covered slot's own byte offset into those
// buffers plus its rank, indexed [layer][projection] host-side, so a later section's dispatch-
// recording code can bind the right sub-range without re-deriving the packing.
//
// NOT built this section, named per StandardsDocument.md Sec5.6 (a deferral surfaced loudly, not
// silently folded into "adapter support"): nothing in the dispatch-recording path
// (RunLayerLoopGpuSubmit, src/gpu/superslm_gpu.cpp) reads these buffers yet -- no GEMM site's own
// HLSL applies the delta this handle makes resident. `sslm_decode_step_gpu` accepts a bound
// adapter, validates it, and records the identical base-only dispatch chain regardless. See
// Claude/Brunel/t2113-1p0-core-build-2026-08-15.md Sec9 for the full scope statement.
// T-2113 (B6b): AdapterProjSlot now lives in gpu_port.h (superslm_gpu::AdapterProjSlot)
// so RunLayerLoopGpuSubmit's own dispatch-recording code (superslm_gpu.cpp) can read it
// through GpuAdapterBridge without depending on this TU's opaque handle types -- the
// same relocation shape D-SLM3351/D-SLM3368 already used for the marshal headers.
// Aliased here so every existing reference in this file (`AdapterProjSlot`, unqualified)
// keeps compiling unchanged.
using AdapterProjSlot = superslm_gpu::AdapterProjSlot;

struct SslmGpuAdapterHandle {
	SslmGpuContext* ctx = nullptr;
	SslmGpuModelHandle* model = nullptr;  // AdapterModelMismatch check, design Sec5.2: pointer-equality

	Microsoft::WRL::ComPtr<ID3D12Resource> lora_ab_buf;  // every covered slot's lora_A+lora_B bytes
	Microsoft::WRL::ComPtr<ID3D12Resource> fold_buf;     // every covered slot's DFS+UFS (identity,mult,shift) triples

	// [layer][7 projections: q,o,gate,up,down,k,v] -- AdapterProjSlot::present false for a
	// (layer, projection) this adapter does not cover, the identical NULL-adapter-per-projection
	// shape LayerAdapterProjection::a_weight==nullptr already documents on the CPU path
	// (forward_sites.h).
	std::vector<std::array<AdapterProjSlot, 7>> slots;
	uint32_t rank = 0;

	// T-2124 (D-SLM3446 P0-3): count of currently-Submitted sequences whose in-flight decode call
	// bound this adapter -- the adapter-handle analogue of SslmGpuModelHandle::submitted_sequences
	// above (identical Busy-precedence shape, D-SLM3417). `mutable`: written through the const
	// SslmGpuAdapterHandle* every decode-step call receives (design Sec5.2's own "checked at every
	// call" contract keeps that pointer const at the public boundary), unlike the model handle's
	// own field, which SubmitOneSequenceDecode already holds through a non-const `model` local.
	// See sslm_gpu_adapter_unmap's own header comment for why this exists: the command list a
	// decode-step call records reads this handle's own lora_ab_buf/fold_buf via root descriptors
	// and returns before the fence signals, so deleting this handle while any count above zero is
	// outstanding would free a buffer the GPU may still be reading from.
	//
	// `mutable` is a DELIBERATE weakening of the published `const SslmGpuAdapterHandle*` contract
	// every public decode signature (include/superslm/gpu_1p0.h) presents to a caller -- not merely
	// a consequence of that pointer's constness. A caller reasoning from the header's `const` sees a
	// handle safe to share read-only across threads (design Sec5.4's own claim); this field is the
	// one piece of state that call actually writes through that same const pointer. Named here
	// explicitly (T-2124 S2, Claude/Poirot/435f730-t2124-adapter-uaf-review.md) because the prior
	// wording of this comment described only the mechanism and not the contract it weakens, which is
	// what let the concurrency consequence (S2, below) go unexamined at authoring time. `std::atomic`
	// (not a plain `int64_t`): see SslmGpuModelHandle::submitted_sequences' own comment, above, for
	// the race this closes and the memory-order justification -- identical reasoning, restated there
	// rather than duplicated here since this field mirrors that one's shape exactly.
	mutable std::atomic<int64_t> submitted_sequences{0};

	// T-2114 (S2): see SslmGpuModelHandle's own comment on its `destroyed` field, above --
	// identical disposition here.
	bool destroyed = false;
};

// T-2113 (B3, design Sec5.3/Sec10 B3): the real SslmGpuSequenceHandle. Owns a
// **dedicated** DEFAULT-heap K/V buffer, sized to this handle's own `context_cap` at
// create() time -- this is the direct replacement for `g_resident_kv`'s single
// pointer-keyed slot (superslm_gpu.cpp): every sequence gets its own buffer, so two
// concurrently-live sequences never share, alias, or evict each other's K/V state, and
// nothing about a second sequence's existence can invalidate the first's cache the way
// a colliding freed address could under the substrate's own keying (D-SLM3311, design
// Sec2.1/Sec5.3). Also owns the host-mirrored SequenceLayerState fields the design
// names (Sec5.3: "hidden_codes, hidden_scale, layer_index, kv_saturation_count,
// context_length") -- a caller decodes through this handle without maintaining a
// separate SequenceLayerState of its own. `kv_needs_resume_barrier` is this handle's
// own half of the RunLayerLoopGpu bridge (gpu_port.h, T-2113 B3's own additive
// trailing-parameter pair): true once a call has left this handle's own kv_buf in
// COPY_SOURCE state (the targeted per-call readback always does this), read and
// consumed by the NEXT call through this handle.
struct SslmGpuSequenceHandle {
	SslmGpuContext* ctx = nullptr;
	SslmGpuModelHandle* model = nullptr;

	Microsoft::WRL::ComPtr<ID3D12Resource> kv_buf;
	bool kv_needs_resume_barrier = false;
	size_t kv_bytes = 0;

	int64_t context_cap = 0;
	superslm_gpu::SslmSequenceGpuState state = superslm_gpu::SslmSequenceGpuState::Idle;

	// Host-mirrored SequenceLayerState surface (design Sec5.3) -- `hidden_codes` is
	// owned here (a std::vector, sized to the model's own hidden_size) rather than
	// caller-supplied, since the 1.0 API's own sequence handle is the thing that owns
	// this state now, not a struct the caller constructs by hand (design Sec4.2's own
	// "no SequenceLayerState a test constructs by hand" framing, generalized to every
	// caller of the 1.0 surface).
	std::vector<int8_t> hidden_codes;
	superslm::CarriedScale hidden_scale{};
	uint32_t layer_index = 0;
	uint64_t kv_saturation_count = 0;
	int64_t context_length = 0;

	// T-2113 (B5, design Sec4.2/Sec6.2/Sec10 B5): the async decode lifecycle's own state.
	// `live_state.hidden_codes` is set ONCE, in sslm_gpu_seq_create, to alias
	// `hidden_codes.data()` directly -- `RunLayerLoopGpuSubmit`/`Finish` read/write through
	// this pointer, so the handle's own `hidden_codes` vector is always the live content, no
	// separate copy. The four scalar fields are synced FROM this handle's own
	// hidden_scale/layer_index/kv_saturation_count/context_length immediately before every
	// sslm_decode_step_gpu call, and copied back INTO them once sslm_gpu_ready's own Finish
	// call completes (design Sec4.2: "collapses Submitted -> Completed -> Idle").
	superslm::SequenceLayerState live_state{};
	// The host mirror `RunLayerLoopGpuSubmit`'s external-K/V readback scatters each call's
	// own newly-written rows into (design Sec5.3's own "still gets the identical CPU-oracle-
	// comparable host mirror" contract) -- sized identically to `kv_bytes` (the same buffer
	// `sslm_gpu_seq_create` already sizes `kv_buf` to), allocated once, accumulated across
	// every decode step this handle ever runs, exactly like the pre-1.0 substrate's own
	// per-session `workspace` vector (tools/t2100_gpu_throughput.cpp's own `ws`).
	std::vector<uint8_t> host_kv_mirror;
	// The in-flight token between a Submitted sslm_decode_step_gpu call and the
	// sslm_gpu_ready call that finishes it -- null whenever this handle is Idle/Completed.
	superslm_gpu::GpuLayerLoopInFlight* in_flight = nullptr;

	// T-2124 (D-SLM3446 P0-3): the adapter bound at this handle's own currently-in-flight
	// submission, null when no adapter was bound or nothing is in flight -- set in
	// SubmitOneSequenceDecode alongside `in_flight` above, cleared and used to decrement the
	// adapter's own `submitted_sequences` count in sslm_gpu_ready the instant this sequence
	// collapses back to Idle. An adapter is a per-CALL argument (sslm_decode_step_gpu's own
	// `adapter_or_null`), never state a sequence handle otherwise retains, so this is the one
	// place a submitted call's own adapter binding needs remembering between Submit and Finish.
	const SslmGpuAdapterHandle* in_flight_adapter = nullptr;

	// G5-5 (T-2132, Brunel): the GPU-1.0 twin of `sslm_seq_s::bound_schema`/`dfa_walk_state`
	// (src/sslm_abi.cpp) -- a sibling scalar pair on this wrapper handle, NEVER folded into
	// `live_state`/`SequenceLayerState` above, mirroring the CPU ABI's own established shape
	// exactly (SequenceLayerState is `forward_sites.h`'s own pinned "closed unit," §13 dim 9 --
	// design Sec6 G5-5's own "the sequence's own state, resident wherever the sequence's other
	// GPU-side state already lives" is satisfied by this handle, not by widening the shared
	// layer-loop struct every RunLayerLoop call already guards by exact field count). `-1` ==
	// no schema bound (this bridge's own int32_t index convention,
	// gpu_1p0_g5_bridge.h); `dfa_walk_state` stays kSslmGpuDfaWalkStateUnused until a schema
	// binds, matching `kDfaWalkStateUnused`'s identical CPU-side semantics.
	int32_t bound_schema_index = -1;
	uint32_t dfa_walk_state = kSslmGpuDfaWalkStateUnused;

	// G5-5 session 3 fix (T-2132, Brunel, Claude/Brunel/t2132-g5-build-2026-08-16.md session 3):
	// the GPU-1.0 twin of `sslm_seq_s::ready_for_logits` (src/sslm_abi.cpp) -- session 3's own
	// diagnosis found the parity harness's own prompt-to-decode transition re-embedding the
	// prompt's last token (no GPU-side shortcut existed), committing a duplicate KV row and
	// running `context_length` +1 from decode step 0 onward, silently, until it flipped an
	// argmax pick 19 steps later. This field, set by `SslmGpuSeqPrefillPromptForG5Bridge`/
	// `SslmGpuSeqPrefillSchemaContentForG5Bridge` on success and consumed (cleared) by
	// `SslmGpuSeqDecodeStepForG5Bridge`, is the real fix: it gives the bridge -- not each
	// caller's own ad hoc composition -- the authority to skip re-embedding a token whose final
	// residual a prefill call already computed, mirroring `sslm_decode_step`'s own
	// `ready_for_logits` branch (src/sslm_abi.cpp:1708) exactly. `sslm_gpu_seq_reset` clears it,
	// mirroring `sslm_seq_reset`'s own identical clear (src/sslm_abi.cpp:1488).
	bool ready_for_logits = false;

	// T-2114 (S2): see SslmGpuModelHandle's own comment on its `destroyed` field
	// (gpu_1p0.cpp) -- identical disposition here.
	bool destroyed = false;
};

namespace {

// T-2113 (B2): synchronous single-buffer resident upload -- UPLOAD-heap temporary -> a
// fresh DEFAULT-heap copy, one command list, one submit, one fence wait. This is the same
// three-step shape RunLayerLoopGpu's own weight-residency miss path uses
// (src/gpu/superslm_gpu.cpp, the `!weights_resident` branch: Upload, MakeBuffer DEFAULT +
// CopyResource + barrier), performed here as its own self-contained call because
// sslm_gpu_model_map has no shared decode-step command list to append to -- it is not part
// of any `sslm_decode_step_gpu` recording window (design Sec5.1: "upload happens once,
// inside map, before the handle is returned to the caller"). Throws std::runtime_error via
// SSLM_GPU_HR on any D3D12 failure, exactly like the substrate's own upload path; the
// caller (sslm_gpu_model_map) catches this and translates it to SSLM_DEVICE_LOST, mirroring
// sslm_gpu_context_create's own device-acquisition-failure disposition (design Sec5.1: "on
// an upload/allocation failure, *out_model=nullptr and the call returns DeviceLost -- the
// same status sslm_gpu_context_create uses for the analogous failure").
Microsoft::WRL::ComPtr<ID3D12Resource> UploadResidentBufferSyncTo(
    superslm_gpu::harness::Device& dev, const void* data, size_t bytes,
    D3D12_RESOURCE_FLAGS resource_flags, D3D12_RESOURCE_STATES final_state) {
	SSLM_GPU_HR(dev.alloc->Reset());
	SSLM_GPU_HR(dev.list->Reset(dev.alloc.Get(), nullptr));
	Microsoft::WRL::ComPtr<ID3D12Resource> upload = dev.Upload(data, bytes);
	Microsoft::WRL::ComPtr<ID3D12Resource> resident =
	    dev.MakeBuffer(bytes, D3D12_HEAP_TYPE_DEFAULT, resource_flags, D3D12_RESOURCE_STATE_COPY_DEST);
	dev.list->CopyResource(resident.Get(), upload.Get());
	D3D12_RESOURCE_BARRIER barrier{};
	barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
	barrier.Transition.pResource = resident.Get();
	barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST;
	barrier.Transition.StateAfter = final_state;
	barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
	dev.list->ResourceBarrier(1, &barrier);
	SSLM_GPU_HR(dev.list->Close());
	ID3D12CommandList* lists[] = {dev.list.Get()};
	dev.queue->ExecuteCommandLists(1, lists);
	SSLM_GPU_HR(dev.queue->Signal(dev.fence.Get(), ++dev.fence_val));
	if (dev.fence->GetCompletedValue() < dev.fence_val) {
		SSLM_GPU_HR(dev.fence->SetEventOnCompletion(dev.fence_val, dev.fence_event));
		WaitForSingleObject(dev.fence_event, INFINITE);
	}
	return resident;
}

// T-2113 (B2): synchronous single-buffer resident upload -- UPLOAD-heap temporary -> a
// fresh DEFAULT-heap copy, one command list, one submit, one fence wait. This is the same
// three-step shape RunLayerLoopGpu's own weight-residency miss path uses
// (src/gpu/superslm_gpu.cpp, the `!weights_resident` branch: Upload, MakeBuffer DEFAULT +
// CopyResource + barrier), performed here as its own self-contained call because
// sslm_gpu_model_map has no shared decode-step command list to append to -- it is not part
// of any `sslm_decode_step_gpu` recording window (design Sec5.1: "upload happens once,
// inside map, before the handle is returned to the caller"). Throws std::runtime_error via
// SSLM_GPU_HR on any D3D12 failure, exactly like the substrate's own upload path; the
// caller (sslm_gpu_model_map) catches this and translates it to SSLM_DEVICE_LOST, mirroring
// sslm_gpu_context_create's own device-acquisition-failure disposition (design Sec5.1: "on
// an upload/allocation failure, *out_model=nullptr and the call returns DeviceLost -- the
// same status sslm_gpu_context_create uses for the analogous failure"). Read-only for its
// whole lifetime (SRV state), matching `weights_buf`/`rope_cos_buf`/`rope_sin_buf`'s own
// contract (design Sec5.4).
Microsoft::WRL::ComPtr<ID3D12Resource> UploadResidentBufferSync(superslm_gpu::harness::Device& dev,
                                                                 const void* data, size_t bytes) {
	return UploadResidentBufferSyncTo(dev, data, bytes, D3D12_RESOURCE_FLAG_NONE,
	                                   D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
}

// T-2113 (B3, design Sec5.3): the K/V-buffer analogue of UploadResidentBufferSync above,
// differing in exactly the two respects a UAV a dispatch chain WRITES to (not merely
// reads) requires: the D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS resource flag, and a
// final state of UNORDERED_ACCESS rather than NON_PIXEL_SHADER_RESOURCE -- matching
// `MakeInitializedUav`'s own target state (superslm_gpu.cpp, internal linkage, not
// reachable from this TU), which is exactly the state RunLayerLoopGpu's own external-
// buffer path (gpu_port.h, T-2113 B3's own trailing-parameter pair) requires a freshly
// created sequence handle's kv_buf to already be in before its first call (no resume
// barrier needed on that first call, design Sec5.3/Sec10 B3).
Microsoft::WRL::ComPtr<ID3D12Resource> UploadResidentUavBufferSync(superslm_gpu::harness::Device& dev,
                                                                    const void* data, size_t bytes) {
	return UploadResidentBufferSyncTo(dev, data, bytes, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
	                                   D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
}

}  // namespace

// Design Sec5.1: "uploads the model's weights, fold-scale tables, and RoPE cos/sin tables
// ... into DEFAULT-heap buffers owned by the returned handle, keyed by the artifact's own
// content hash." Marshals `*base` into a real LayerWeights[] (superslm_marshal::MarshalLayer,
// T-2113 B2's own relocation of tools/sslm_marshal.h -> include/superslm/layer_marshal.h --
// the first production call site for this logic), packs it via the shared
// PackLayerWeightsBytes (T-2113 B2, extracted from RunLayerLoopGpu's own miss path so both
// the pre-1.0 substrate and this handle's upload path compute identical bytes from one
// implementation), and uploads weights + RoPE cos/sin tables as three independent resident
// buffers -- no process-global cache anywhere in this path (D-SLM3294).
SslmGpuStatus sslm_gpu_model_map(SslmGpuContext* ctx, const SslmModelView* base,
                                  GpuResidencyConfig cfg, SslmGpuModelHandle** out_model) {
	(void)cfg;  // GpuResidencyConfig carries no fields the design assigns yet (Sec5.1).
	if (!out_model) {
		return SSLM_DEVICE_LOST;  // same "no live object to report through" reasoning as
		                          // sslm_gpu_context_create's own null-out-parameter case.
	}
	*out_model = nullptr;
	if (!ctx || !base) {
		return SSLM_DEVICE_LOST;
	}

	const uint32_t num_heads = base->config.num_attention_heads;
	const uint32_t num_kv_heads = base->config.num_key_value_heads;
	const uint32_t num_hidden_layers = base->config.num_hidden_layers;
	const uint32_t H = static_cast<uint32_t>(base->config.hidden_size);
	const uint32_t HD = static_cast<uint32_t>(base->config.head_dim);
	const uint32_t KV = num_kv_heads * HD;
	const uint32_t I = static_cast<uint32_t>(base->config.intermediate_size);
	const uint32_t NQH = num_heads;

	// Marshal every layer's LayerWeights (superslm_marshal::MarshalLayer, the identical
	// call tools/t2100_gpu_throughput.cpp's own T-2100 harness makes before calling
	// RunLayerLoopGpu). A malformed artifact (a missing required tensor) fails here; Sec9's
	// own table assigns model-map failure only DeviceLost (the upload/allocation
	// disposition) -- there is no separate status for "the artifact's own content could not
	// be marshaled," so a marshal failure takes that same disposition rather than this
	// build inventing a status the design's own Sec9 table does not name.
	std::vector<superslm_marshal::LayerBacking> backings(num_hidden_layers);
	std::vector<superslm::LayerWeights> layers(num_hidden_layers);
	for (uint32_t l = 0; l < num_hidden_layers; ++l) {
		std::string marshal_err;
		if (!superslm_marshal::MarshalLayer(*base, l, num_heads, num_kv_heads, backings[l], layers[l],
		                                     &marshal_err)) {
			return SSLM_DEVICE_LOST;
		}
	}

	const superslm_gpu::GpuLayerLayout layout =
	    superslm_gpu::ComputeLayerLayout(H, KV, num_kv_heads, NQH, I);
	const std::vector<uint8_t> lw_bytes = superslm_gpu::PackLayerWeightsBytes(
	    layers.data(), num_hidden_layers, layout, H, KV, num_kv_heads, NQH, I);

	// T-2105's own RoPE cos/sin residency construction (Claude/Laplace/
	// t2105-gpu-speed-ceiling-2026-08-14.md Sec2 change 1), re-derived here per design Sec1's
	// own "re-derived, never cherry-picked" instruction: model-wide constants, read directly
	// from the artifact's own rope_tables manifest, exactly as RunLayerLoopGpu's own
	// `cos_t`/`sin_t` lookup does (superslm_gpu.cpp).
	const superslm::SslmTensorView* cos_t = base->rope_tables.Tensor("cos");
	const superslm::SslmTensorView* sin_t = base->rope_tables.Tensor("sin");
	const bool has_rope = cos_t != nullptr && sin_t != nullptr;
	const uint64_t cos_need = cos_t != nullptr ? static_cast<uint64_t>(cos_t->elem_count) * 8u : 8u;
	const uint64_t sin_need = sin_t != nullptr ? static_cast<uint64_t>(sin_t->elem_count) * 8u : 8u;
	std::vector<uint8_t> cos_bytes(static_cast<size_t>(cos_need), 0);
	if (cos_t != nullptr) std::memcpy(cos_bytes.data(), cos_t->data, cos_bytes.size());
	std::vector<uint8_t> sin_bytes(static_cast<size_t>(sin_need), 0);
	if (sin_t != nullptr) std::memcpy(sin_bytes.data(), sin_t->data, sin_bytes.size());

	// T-2113 (B3.5, design Sec5.1's own amendment / Sec5.3a): the embedding matrix, its
	// site constant, and vocab_size -- read from `*base` exactly where sslm_generate.cpp's
	// own precedent reads them (`model_view.weights.Tensor("embed")`,
	// `ReadCarriedScale(model_view.composition_constants, "embed", &ok)`,
	// `model_view.config.vocab_size`). A malformed artifact missing either takes the same
	// "no more specific status than DeviceLost" disposition every other marshal failure in
	// this function already uses (the missing-tensor case above).
	const superslm::SslmTensorView* embed_w = base->weights.Tensor("embed");
	if (!embed_w) {
		return SSLM_DEVICE_LOST;
	}
	bool embed_scale_ok = true;
	const superslm::CarriedScale embed_site_constant =
	    superslm_marshal::ReadCarriedScale(base->composition_constants, "embed", &embed_scale_ok);
	if (!embed_scale_ok) {
		return SSLM_DEVICE_LOST;
	}
	const int32_t vocab_size = base->config.vocab_size;
	const size_t embed_bytes_needed = static_cast<size_t>(vocab_size) * static_cast<size_t>(H);
	// T-2114 (S3, Claude/Poirot/50f3d5d-t2113-1p0-gpu-core-build-review.md): bound the copy
	// below against the tensor's own declared extent -- the SAME check `cos_need`/`sin_need`
	// above already apply to the RoPE tables (`cos_t->elem_count`/`sin_t->elem_count`), which
	// this one used to skip. `embed_w->data` is `elem_count` int8 bytes (element_size 1 for
	// this dtype); a short `embed` section previously read `embed_bytes_needed` bytes from it
	// regardless, a multi-megabyte out-of-bounds read at map time on a malformed artifact --
	// the CPU precedent this line is derived from (sslm_generate.cpp:287) never takes this
	// risk because it reads one row at a time, bounded by `token_id < vocab_size`, never the
	// whole matrix eagerly.
	if (embed_w->elem_count < static_cast<uint64_t>(embed_bytes_needed)) {
		return SSLM_DEVICE_LOST;  // same "no more specific status than DeviceLost" disposition
		                          // every other marshal failure in this function already uses.
	}

	// G5-5 (T-2132, Brunel): final_norm.gain + head (lm_head or tied embed) -- the SAME two
	// tensor lookups sslm_abi.cpp's own BuildEngineCache already runs (src/sslm_abi.cpp), read
	// here directly from `*base` since this TU has no engine-cache object of its own. Required
	// for the finish-token bridge's own final_norm+logits step; a malformed artifact missing
	// either takes the same "no more specific status than DeviceLost" disposition every other
	// marshal failure in this function already uses.
	const superslm::SslmTensorView* final_gain_w = base->weights.Tensor("final_norm.gain");
	if (!final_gain_w) {
		return SSLM_DEVICE_LOST;
	}
	bool final_norm_scale_ok = true;
	const superslm::CarriedScale final_norm_site_constant = superslm_marshal::ReadCarriedScale(
	    base->composition_constants, "final_norm", &final_norm_scale_ok);
	if (!final_norm_scale_ok) {
		return SSLM_DEVICE_LOST;
	}
	const superslm::SslmTensorView* head_w = nullptr;
	if (base->config.tie_word_embeddings) {
		head_w = embed_w;  // tied embeddings: the head IS the embedding matrix, CPU precedent.
	} else {
		head_w = base->weights.Tensor("lm_head");
		if (!head_w) {
			return SSLM_DEVICE_LOST;
		}
	}
	const size_t head_bytes_needed = static_cast<size_t>(vocab_size) * static_cast<size_t>(H);
	if (head_w->elem_count < static_cast<uint64_t>(head_bytes_needed)) {
		return SSLM_DEVICE_LOST;  // same bounded-copy discipline the embed check above applies.
	}

	// G5-5 (T-2132, Brunel): this artifact's own SchemaMasks/SCM1 section, if present -- absence
	// is a valid, unconstrained-only artifact (design Sec13.1), the SAME disposition the CPU
	// ABI's own sslm_model_map already applies (src/sslm_abi.cpp). Parsed against a HOST-OWNED
	// copy of the section bytes (schema_section_bytes, below) rather than `base`'s own pointer,
	// since `base`'s own lifetime is caller-owned and this handle must serve every schema-bound
	// call for its OWN whole lifetime -- the identical caller-view-lifetime reasoning B3.5's own
	// embed_weights capture (this function, above) already documents.
	const superslm::SslmSectionView* schema_section =
	    base->Section(superslm::SslmSectionType::SchemaMasks);
	std::vector<uint8_t> schema_section_bytes;
	if (schema_section) {
		schema_section_bytes.assign(schema_section->data,
		                             schema_section->data + schema_section->byte_size);
	}

	std::unique_ptr<SslmGpuModelHandle> h(new SslmGpuModelHandle());
	try {
		h->weights_buf = UploadResidentBufferSync(ctx->device, lw_bytes.data(), lw_bytes.size());
		h->rope_cos_buf = UploadResidentBufferSync(ctx->device, cos_bytes.data(), cos_bytes.size());
		h->rope_sin_buf = UploadResidentBufferSync(ctx->device, sin_bytes.data(), sin_bytes.size());
		if (!schema_section_bytes.empty()) {
			// G5-5's own "Builds: mask-page residency on device" (design Sec6 G5-5, VRAM,
			// alongside weights/scales per the existing GPU-tenancy model, Sec8.4) -- the whole
			// SCM1 section, uploaded read-only, the SAME UploadResidentBufferSync three-step
			// shape weights/rope use immediately above.
			h->mask_pages_buf = UploadResidentBufferSync(ctx->device, schema_section_bytes.data(),
			                                              schema_section_bytes.size());
		}
	} catch (const std::exception&) {
		return SSLM_DEVICE_LOST;
	}

	// Host-only copy (Sec5.3a: "not new residency, no additional GPU upload") -- taken
	// after the three real device uploads above have already succeeded, so a failure here
	// never leaves a half-uploaded device handle: this is a plain host memcpy that cannot
	// itself throw a D3D12 exception, so it is outside the try/catch above by construction.
	h->embed_weights.assign(reinterpret_cast<const int8_t*>(embed_w->data),
	                         reinterpret_cast<const int8_t*>(embed_w->data) + embed_bytes_needed);
	h->embed_site_constant = embed_site_constant;
	h->vocab_size = vocab_size;
	h->final_norm_gain = superslm_marshal::WidenGainToInt32(*final_gain_w);
	h->final_norm_site_constant = final_norm_site_constant;
	h->head_weights.assign(reinterpret_cast<const int8_t*>(head_w->data),
	                        reinterpret_cast<const int8_t*>(head_w->data) + head_bytes_needed);

	// G5-5: parse the host-owned copy (schema_section_bytes, moved into the handle here) --
	// `schemas`'s own SchemaEntry pointers point INTO `h->schema_section_bytes`, never the local
	// `schema_section_bytes`/`base`'s own view, so they stay valid for this handle's whole
	// lifetime. A structurally-invalid SchemaMasks section is a malformed artifact -- same
	// "no more specific status than DeviceLost" disposition every other marshal failure in this
	// function already uses (mirrors sslm_abi.cpp's own SSLM_ARTIFACT_REJECTED-for-any-Sec13.3-
	// violation collapse, translated to this file's own DeviceLost convention).
	if (!schema_section_bytes.empty()) {
		h->schema_section_bytes = std::move(schema_section_bytes);
		std::string schema_err;
		if (!superslm::SchemaMasksTable::Parse(h->schema_section_bytes.data(),
		                                        h->schema_section_bytes.size(),
		                                        static_cast<uint32_t>(vocab_size), h->schemas,
		                                        &schema_err)) {
			return SSLM_DEVICE_LOST;
		}
	}

	h->ctx = ctx;
	h->content_hash = base->RawIntegrityHash();
	h->has_rope_tables = has_rope;
	h->rope_cos_elem_count = cos_t != nullptr ? cos_t->elem_count : 0;
	h->rope_sin_elem_count = sin_t != nullptr ? sin_t->elem_count : 0;
	h->layout = layout;
	h->num_hidden_layers = num_hidden_layers;
	h->hidden_size = H;
	h->head_dim = HD;
	h->num_key_value_heads = num_kv_heads;
	h->num_attention_heads = NQH;
	h->intermediate_size = I;
	h->context_cap = static_cast<int64_t>(base->config.context_cap);

	ctx->live_handles += 1;
	*out_model = h.release();
	return SSLM_OK;
}

// Design Sec5.1: "frees the residency and returns Ok. Freeing a model handle with live
// sequences bound to it is a caller error: returns ModelHasLiveSequences ... and does NOT
// free the residency." T-2113 (B3): `live_sequences` is now incremented/decremented by
// real `SslmGpuSequenceHandle`s (sslm_gpu_seq_create/release, below) -- this guard reads
// live values as of this section.
//
// T-2113 (design Sec4.2/Sec9, routed `Claude/Curie/t2112-1p0-red-suite-2026-08-15.md` Sec7.2
// item 2 / Sec7.3, D-SLM3417): design Sec9's own Busy row states this call returns `Busy`
// "against a model with any Submitted sequence bound... which takes precedence" over
// `ModelHasLiveSequences`' own "any state, not only Submitted" caveat -- the guard below used
// to check only `live_sequences` and could not distinguish a Submitted-bound model from an
// Idle-bound one, always returning `ModelHasLiveSequences` for both. Fixed: `submitted_sequences`
// (SslmGpuModelHandle's own field, above) is checked FIRST, so a model with at least one
// Submitted sequence bound returns `Busy`; only once that count is zero does the broader
// `live_sequences` (any state) check run.
SslmGpuStatus sslm_gpu_model_unmap(SslmGpuContext* ctx, SslmGpuModelHandle* model) {
	if (!model) {
		return SSLM_OK;  // same null-is-a-no-op reasoning as sslm_gpu_context_destroy(nullptr).
	}
	// T-2124 (D-SLM3446 P1-4): validate this handle was actually mapped against `ctx` -- the
	// external-review gap: this function used to ignore `ctx` entirely (a bare `(void)ctx`, now
	// removed since the parameter is read below and by the decrement further down) and act purely
	// off `model->ctx`, so a caller passing a DIFFERENT (but live) context here was never rejected
	// -- a sequence could be created on one context against a model mapped on another, binding
	// resources from different D3D12 devices (see sslm_gpu_seq_create's own identical fix for the
	// sequence-creation half of this same gap).
	if (model->ctx != ctx) {
		return SSLM_DEVICE_LOST;  // same "no more specific status" disposition sslm_gpu_model_map
		                          // already uses for a malformed/foreign-context argument.
		                          // P1-6 (external review, status-ABI collapse to DEVICE_LOST):
		                          // this rejects a PERMANENT caller-contract violation (a foreign
		                          // handle never becomes valid by retrying) through a status §9
		                          // classifies transient/recoverable -- routing left as-is per
		                          // Claude/Poirot/435f730-t2124-adapter-uaf-review.md M2 (a new
		                          // enumerator is out of this ticket's scope), flagged here so the
		                          // P1-6 taxonomy pass (S-FREEZE) finds this site.
	}
	if (model->submitted_sequences > 0) {
		return SSLM_BUSY;  // design Sec9's own Busy-precedence, D-SLM3417.
	}
	if (model->live_sequences > 0) {
		return SSLM_MODEL_HAS_LIVE_SEQUENCES;
	}
	// T-2114 (M3, Claude/Poirot/50f3d5d-t2113-1p0-gpu-core-build-review.md): decrement the
	// handle's OWN stored context (`model->ctx`, set once at map() time) -- now provably equal to
	// `ctx` by the check above too, kept as `model->ctx` unchanged since every handle already
	// carries the context it was created against for exactly this reason (design Sec5.1's own
	// per-handle ownership).
	if (model->ctx->live_handles > 0) {
		model->ctx->live_handles -= 1;
	}
	model->destroyed = true;
	delete model;
	return SSLM_OK;
}

// Design Sec5.2/Sec10 B6: "mirrors the CPU-side LoadAdapterArtifact/AdapterHandle shape exactly,
// on the device side: validates the adapter's base-hash against model's own content hash ...
// and uploads the per-layer, per-projection lora_A/lora_B weights plus DeltaFoldScales/
// UFoldScales tables into their own DEFAULT-heap buffers." Reuses
// superslm_adapter::PopulateAdapterFromView (include/superslm/adapter_marshal.h -- the body of
// LoadAdapterArtifact that runs after SslmModel::Load, extracted this section, D-SLM3368, since
// this call already holds a parsed view and must not re-read/re-parse the artifact's own bytes
// from a path a view does not retain) for the parse/validate half -- the identical B0b checks
// (ValidateAmplifyingFoldBaseHash/Dimension/Projection) the CPU path already runs, run here for
// real against a real converted artifact for the first time on the GPU path.
SslmGpuStatus sslm_gpu_adapter_map(SslmGpuContext* ctx, SslmGpuModelHandle* model,
                                    const SslmModelView* adapter_artifact,
                                    SslmGpuAdapterHandle** out_adapter) {
	if (!out_adapter) {
		return SSLM_DEVICE_LOST;  // same "no live object to report through" reasoning as
		                          // sslm_gpu_model_map's own null-out-parameter case.
	}
	*out_adapter = nullptr;
	if (!ctx || !model || !adapter_artifact) {  // T-2114 (S2): the ->destroyed read this line
	                                             // used to carry was dead on a live handle and
	                                             // UB on a released one -- removed, see
	                                             // SslmGpuModelHandle's own field comment.
		return SSLM_DEVICE_LOST;
	}
	if (model->ctx != ctx) {  // T-2124 (D-SLM3446 P1-4): cross-context validation -- a model
	                          // handle mapped against a different context names a different
	                          // D3D12 device; same disposition as the malformed-argument case above.
	                          // P1-6 (external review, status-ABI collapse to DEVICE_LOST): this
	                          // rejects a PERMANENT caller-contract violation through a status §9
	                          // classifies transient/recoverable -- routing left as-is per
	                          // Claude/Poirot/435f730-t2124-adapter-uaf-review.md M2, flagged for
	                          // the P1-6 taxonomy pass (S-FREEZE).
		return SSLM_DEVICE_LOST;
	}

	superslm_adapter::BaseModelGeometry base_geom;
	base_geom.num_hidden_layers = model->num_hidden_layers;
	base_geom.hidden_size = model->hidden_size;
	base_geom.intermediate_size = model->intermediate_size;
	base_geom.kv_hidden_size = static_cast<uint64_t>(model->num_key_value_heads) * model->head_dim;
	base_geom.base_artifact_hash = model->content_hash;

	// T-2113 (B6, D-SLM3368): PopulateAdapterFromView is the extracted body of
	// LoadAdapterArtifact (include/superslm/adapter_marshal.h) that runs AFTER SslmModel::Load --
	// this call site already HAS a parsed view (`*adapter_artifact`, design Sec5.2's own
	// parameter), so it validates/populates against that view directly rather than re-reading
	// and re-parsing the artifact's own bytes from a file path a view does not retain.
	superslm_adapter::AdapterMeta meta;
	std::vector<superslm::LayerAdapter> layer_adapters;
	std::string load_err;
	const superslm_adapter::AdapterLoadStatus load_st = superslm_adapter::PopulateAdapterFromView(
	    *adapter_artifact, base_geom, meta, layer_adapters, &load_err);
	if (load_st == superslm_adapter::AdapterLoadStatus::BaseHashMismatch) {
		return SSLM_ADAPTER_BASE_HASH_MISMATCH;
	}
	if (load_st != superslm_adapter::AdapterLoadStatus::Ok) {
		// Every other rejection (malformed ADP1, missing sections, a dimension/shape
		// mismatch) is a structurally-broken-artifact class design Sec9's own table assigns
		// no dedicated status to -- same "no more specific status" disposition B2/B3 already
		// use for the analogous marshal-failure case (sslm_gpu_model_map's own comment above).
		return SSLM_DEVICE_LOST;
	}

	// q,o,gate,up,down,k,v -- LayerAdapter's own field order (forward_sites.h), matched to
	// AdapterProjSlot's own [7] index by position.
	auto proj_at = [](const superslm::LayerAdapter& la, int p) -> const superslm::LayerAdapterProjection& {
		switch (p) {
			case 0: return la.q;
			case 1: return la.o;
			case 2: return la.gate;
			case 3: return la.up;
			case 4: return la.down;
			case 5: return la.k;
			default: return la.v;
		}
	};
	// T-2114 (M4, Claude/Poirot/50f3d5d-t2113-1p0-gpu-core-build-review.md): the SAME
	// positional order `proj_at` above uses, named -- feeds `AdapterInChannelsFor`
	// (adapter_marshal.h:225, the one function that owns the projection-kind-to-in_channels
	// rule) below instead of this loop re-deriving it from a bare `p == 4` positional check.
	static const char* const kProjNamesByIndex[7] = {"q_proj",   "o_proj", "gate_proj", "up_proj",
	                                                  "down_proj", "k_proj", "v_proj"};

	std::vector<uint8_t> ab_bytes;
	std::vector<uint8_t> fold_bytes;
	std::vector<std::array<AdapterProjSlot, 7>> slots(model->num_hidden_layers);

	for (uint32_t l = 0; l < model->num_hidden_layers; ++l) {
		const superslm::LayerAdapter& la = layer_adapters[l];
		for (int p = 0; p < 7; ++p) {
			const superslm::LayerAdapterProjection& proj = proj_at(la, p);
			if (!proj.a_weight || !proj.b_weight || !proj.delta_fold_entry || !proj.u_fold_entry) {
				continue;  // this (layer, projection) is not covered by this adapter -- present
				           // stays false, the identical NULL-per-projection shape LayerWeights'
				           // own B1c documents on the CPU path (forward_sites.h).
			}
			AdapterProjSlot& slot = slots[l][static_cast<size_t>(p)];
			slot.present = true;
			slot.rank = la.rank;
			slot.out_channels = static_cast<uint32_t>(proj.delta_fold_entry->row_count);
			// a_weight is [rank, in_channels]: in_channels isn't independently recorded on
			// LayerAdapterProjection -- reconstructed here by calling the ONE function that owns
			// the projection-kind-to-in_channels rule (AdapterInChannelsFor,
			// include/superslm/adapter_marshal.h), not a second copy of its own logic.
			const uint64_t in_channels = superslm_adapter::AdapterInChannelsFor(
			    kProjNamesByIndex[p], model->hidden_size, model->intermediate_size);
			slot.a_bytes = static_cast<uint64_t>(la.rank) * in_channels;
			slot.b_bytes = static_cast<uint64_t>(slot.out_channels) * la.rank;

			slot.a_offset = ab_bytes.size();
			ab_bytes.insert(ab_bytes.end(), proj.a_weight, proj.a_weight + slot.a_bytes);
			slot.b_offset = ab_bytes.size();
			ab_bytes.insert(ab_bytes.end(), proj.b_weight, proj.b_weight + slot.b_bytes);

			slot.fold_offset = fold_bytes.size();
			const uint64_t dfs_bytes = static_cast<uint64_t>(proj.delta_fold_entry->row_count) * 12u;
			const uint64_t ufs_bytes = static_cast<uint64_t>(proj.u_fold_entry->row_count) * 12u;
			fold_bytes.insert(fold_bytes.end(), proj.delta_fold_entry->data,
			                   proj.delta_fold_entry->data + dfs_bytes);
			fold_bytes.insert(fold_bytes.end(), proj.u_fold_entry->data,
			                   proj.u_fold_entry->data + ufs_bytes);
		}
	}
	// A degenerate adapter (every slot uncovered -- e.g. a target_modules_mask of zero) still
	// maps cleanly; MakeBuffer/Upload need at least one byte, matching sslm_gpu_model_map's own
	// cos_need/sin_need "no rope tables" fallback (>= 8 bytes) above.
	if (ab_bytes.empty()) ab_bytes.resize(1, 0);
	if (fold_bytes.empty()) fold_bytes.resize(1, 0);

	std::unique_ptr<SslmGpuAdapterHandle> h(new SslmGpuAdapterHandle());
	try {
		h->lora_ab_buf = UploadResidentBufferSync(ctx->device, ab_bytes.data(), ab_bytes.size());
		h->fold_buf = UploadResidentBufferSync(ctx->device, fold_bytes.data(), fold_bytes.size());
	} catch (const std::exception&) {
		return SSLM_DEVICE_LOST;
	}

	h->ctx = ctx;
	h->model = model;
	h->slots = std::move(slots);
	h->rank = meta.rank;

	ctx->live_handles += 1;
	*out_adapter = h.release();
	return SSLM_OK;
}

// Design Sec5.2 states "releases the adapter's own residency and returns Ok ... carries no Busy
// precondition of its own (unlike model/sequence release) because an adapter handle holds no
// in-flight GPU work." T-2124 (D-SLM3446 P0-3, external review): that premise is FALSE on the
// async submit/finish split (design Sec6.2/B5). `sslm_decode_step_gpu`'s own submission path
// (SubmitOneSequenceDecode, this file) passes this handle's `lora_ab_buf`/`fold_buf` into the
// recorded command list via root descriptors (`adapter_bridge.lora_ab_resident`/`fold_resident`)
// and returns BEFORE the fence signals -- unlike every OTHER per-call GPU resource
// (`GpuLayerLoopInFlight`, superslm_gpu.cpp), nothing kept these two buffers alive past that
// return. Deleting this handle here while such a submission is still in flight frees them out
// from under a command list the GPU may still be executing: a real use-after-free/device-removal
// hazard, the exact class `GpuLayerLoopInFlight`'s own header comment already documents being
// reproduced by execution once (T-2039) for the other resident buffers this same split shares.
//
// Fixed the same way `sslm_gpu_model_unmap` already fixes the identical class of gap for MODEL
// handles (design Sec9's own Busy-precedence row, D-SLM3417): `submitted_sequences` (this
// handle's own field, above) is checked before any release proceeds -- an adapter bound to at
// least one Submitted sequence returns Busy, exactly like a model does. Chosen over retaining
// ComPtr copies inside `GpuLayerLoopInFlight` (the alternative this design's own nearest
// precedent, T-2039's fix, would suggest) because Busy-accounting mirrors the model-unmap
// precedent exactly and keeps this call's contract legible at the call site (a caller sees WHY
// release did not happen), rather than adding a second, adapter-shaped retention mechanism next
// to the per-call-resource one `GpuLayerLoopInFlight` already carries.
//
// A sequence still bound to `adapter` in the IDLE state (no submission currently in flight)
// continues decoding against whatever its own buffers still contain until that sequence's next
// call rebinds or releases -- the documented residual design Sec5.2 already names explicitly,
// unaffected by this fix, not a guarded case this call adds one for.
SslmGpuStatus sslm_gpu_adapter_unmap(SslmGpuContext* ctx, SslmGpuAdapterHandle* adapter) {
	if (!adapter) {
		return SSLM_OK;  // same null-is-a-no-op reasoning as sslm_gpu_model_unmap(nullptr).
	}
	// T-2124 (D-SLM3446 P1-4): validate this handle was actually mapped against `ctx` -- see
	// sslm_gpu_model_unmap's own identical check and comment.
	if (adapter->ctx != ctx) {
		return SSLM_DEVICE_LOST;  // same "no more specific status" disposition sslm_gpu_adapter_map
		                          // already uses for a malformed/foreign-context argument.
		                          // P1-6 (external review, status-ABI collapse to DEVICE_LOST): this
		                          // rejects a PERMANENT caller-contract violation through a status
		                          // §9 classifies transient/recoverable -- routing left as-is per
		                          // Claude/Poirot/435f730-t2124-adapter-uaf-review.md M2, flagged
		                          // for the P1-6 taxonomy pass (S-FREEZE).
	}
	if (adapter->submitted_sequences > 0) {
		return SSLM_BUSY;  // T-2124 (D-SLM3446 P0-3): Busy-precedence, mirroring
		                    // sslm_gpu_model_unmap's own D-SLM3417 fix above.
	}
	if (adapter->ctx->live_handles > 0) {
		adapter->ctx->live_handles -= 1;
	}
	adapter->destroyed = true;
	delete adapter;
	return SSLM_OK;
}

// Design Sec4.1.1: "on device-acquisition failure, *out_ctx is set to nullptr and
// the call returns DeviceLost (Sec9) -- the same status a later mid-decode
// device-removal produces". harness::Device::Init() never throws past its own
// try/catch inside harness::GetDevice()'s lambda for the singleton path; here, since
// this context owns its Device value directly (no lambda-wrapped static), the
// equivalent guard is applied at the call site instead.
SslmGpuStatus sslm_gpu_context_create(GpuContextConfig cfg, SslmGpuContext** out_ctx) {
	(void)cfg;  // GpuContextConfig carries no fields the design assigns yet (Sec4.1.1
	            // does not name any -- the parameter exists for forward compatibility
	            // with a future field, per the design's own declared surface).
	if (!out_ctx) {
		// Not a case the design's own Sec9 table assigns a status to (a null
		// out-parameter is a caller contract violation the design does not model
		// as a runtime-recoverable condition anywhere else in Sec4/Sec5 either);
		// there is no live object to report through, so there is nothing this
		// call can do except decline to dereference a null pointer. Documented
		// here rather than silently invoking undefined behavior.
		return SSLM_DEVICE_LOST;
	}

	SslmGpuContext* ctx = new SslmGpuContext();
	try {
		ctx->device.Init();
	} catch (const std::exception&) {
		ctx->device.available = false;
	}
	if (!ctx->device.available) {
		delete ctx;
		*out_ctx = nullptr;
		return SSLM_DEVICE_LOST;
	}

	*out_ctx = ctx;
	return SSLM_OK;
}

// Design Sec4.1.1: "asserts (debug) or fails loudly (release: returns
// ContextHasLiveHandles ... and does NOT destroy the context, so the caller may
// release the offending handles and retry) if any model, adapter, or sequence
// handle created from this context has not been released first; returns Ok and
// destroys the context otherwise." No handle type above SslmGpuContext itself is
// constructible yet (B2/B3/B6 land those), so `live_handles` is always 0 today --
// the guard is real code, exercised honestly (it reads the same field B2+ will
// increment), not a stand-in that merely returns Ok unconditionally.
SslmGpuStatus sslm_gpu_context_destroy(SslmGpuContext* ctx) {
	if (!ctx) {
		// Destroying a null context is a caller no-op under the same "no status
		// exists for a null-pointer contract violation" reasoning as create()
		// above; nothing to free, nothing to reject.
		return SSLM_OK;
	}
	if (ctx->live_handles > 0) {
		return SSLM_CONTEXT_HAS_LIVE_HANDLES;
	}
	ctx->destroyed = true;
	delete ctx;
	return SSLM_OK;
}

// T-2113 (B3, design Sec5.3/Sec10 B3): "allocates a REAL, dedicated DEFAULT-heap K/V
// buffer sized to context_cap for this sequence ... plus the host-mirrored
// SequenceLayerState fields ... the handle owns." Sizing uses the identical overflow-
// guarded product RunLayerLoopGpu's own guard ladder computes (superslm_gpu.cpp,
// `kv_bytes_needed` = num_hidden_layers * context_cap * num_key_value_heads * head_dim
// * 2) -- the same formula, so a buffer this call sizes and the buffer RunLayerLoopGpu's
// own guard would separately compute for the identical inputs can never disagree.
SslmGpuStatus sslm_gpu_seq_create(SslmGpuContext* ctx, SslmGpuModelHandle* model,
                                   int64_t context_cap, SslmGpuSequenceHandle** out_seq) {
	if (!out_seq) {
		return SSLM_SEQUENCE_KV_BUFFER_MISMATCH;  // no live object to report through,
		                                           // same reasoning as the null-out-
		                                           // parameter cases in B1/B2 above.
	}
	*out_seq = nullptr;
	if (!ctx || !model || model->ctx != ctx) {  // T-2114 (S2): ->destroyed read removed, see model
	                                             // handle's own comment. T-2124 (D-SLM3446 P1-4):
	                                             // `model->ctx != ctx` -- a model mapped on a
	                                             // different context names a different D3D12
	                                             // device; a sequence created here would bind
	                                             // resources across two contexts.
		return SSLM_SEQUENCE_KV_BUFFER_MISMATCH;
	}
	if (context_cap < 1) {
		// Mirrors RunLayerLoopGpu's own InvalidContextCap guard (superslm_gpu.cpp:1038,
		// "context_cap < 1") -- design Sec9 assigns this call's own sizing failures to
		// SequenceKvBufferMismatch, so a non-positive context_cap (which the sizing
		// formula below cannot produce a sane buffer from either) takes that status
		// here rather than the CPU/GPU forward-status enum this call never returns.
		return SSLM_SEQUENCE_KV_BUFFER_MISMATCH;
	}

	// Design Sec5.3: "a real, dedicated DEFAULT-heap K/V buffer sized to context_cap" --
	// the identical overflow-guarded kv_bytes_needed product RunLayerLoopGpu's own guard
	// ladder computes (superslm_gpu.cpp, the WorkspaceSizeOrOverflow guard), so this
	// handle's own buffer and what a decode call through it will expect can never
	// silently disagree on size.
	size_t kv_bytes_needed = static_cast<size_t>(model->num_hidden_layers);
	const size_t kv_factors[] = {static_cast<size_t>(context_cap), model->num_key_value_heads,
	                              static_cast<size_t>(model->head_dim), 2u};
	for (size_t factor : kv_factors) {
		if (factor != 0 && kv_bytes_needed > SIZE_MAX / factor) {
			return SSLM_SEQUENCE_KV_BUFFER_MISMATCH;
		}
		kv_bytes_needed *= factor;
	}

	std::unique_ptr<SslmGpuSequenceHandle> h(new SslmGpuSequenceHandle());
	// Zero-initialized: a fresh sequence (context_length == 0) never reads a K/V row
	// before this same call's own commit writes it (RunLayerLoopGpu's own
	// `fresh_sequence` reasoning, superslm_gpu.cpp, applies identically here), so the
	// buffer's initial content is never observed by a real forward pass -- zeroed
	// anyway (rather than left as MakeBuffer's own undefined DEFAULT-heap content) so a
	// hostile or buggy caller reading past `context_length` sees a defined value, never
	// uninitialized VRAM.
	std::vector<uint8_t> zero_kv(kv_bytes_needed, 0);
	try {
		h->kv_buf = UploadResidentUavBufferSync(ctx->device, zero_kv.data(), zero_kv.size());
	} catch (const std::exception&) {
		return SSLM_DEVICE_LOST;  // mirrors sslm_gpu_model_map's own upload-failure
		                          // disposition (design Sec5.1/Sec5.3 symmetry).
	}
	// Structural self-check (design Sec9: "context_cap inconsistent with the K/V buffer
	// this call itself just sized -- an internal invariant that should be unreachable,
	// guarded anyway per the substrate's own GpuGemmSplitSite default-case precedent").
	// By construction (the buffer was allocated at exactly `kv_bytes_needed` bytes,
	// immediately above), this cannot fail from any caller input -- real, exercised
	// guard code proving its own invariant true on every call, not a stand-in, matching
	// B1's `ContextHasLiveHandles`/B2's `ModelHasLiveSequences` precedent of a guard
	// that reads a real field even where no reachable input drives it false.
	if (h->kv_buf == nullptr || zero_kv.size() != kv_bytes_needed) {
		return SSLM_SEQUENCE_KV_BUFFER_MISMATCH;
	}

	h->ctx = ctx;
	h->model = model;
	h->context_cap = context_cap;
	h->kv_bytes = kv_bytes_needed;
	h->kv_needs_resume_barrier = false;  // freshly created in UNORDERED_ACCESS state
	                                      // (UploadResidentUavBufferSync above) -- the
	                                      // first call through this handle needs no
	                                      // resume barrier (gpu_port.h's own B3 comment).
	h->hidden_codes.assign(model->hidden_size, 0);
	h->hidden_scale = superslm::CarriedScale{};
	h->layer_index = 0;
	h->kv_saturation_count = 0;
	h->context_length = 0;
	h->state = superslm_gpu::SslmSequenceGpuState::Idle;
	// T-2113 (B5): `live_state.hidden_codes` aliases `hidden_codes.data()` for this handle's
	// whole lifetime -- `hidden_codes` never reallocates after this construction (fixed size,
	// never resized), so the pointer stays valid until sslm_gpu_seq_release destroys the
	// handle. `host_kv_mirror` starts zeroed, identically to `zero_kv` above (a fresh
	// sequence's own workspace mirror has never had a row written into it either).
	h->live_state.hidden_codes = h->hidden_codes.data();
	h->host_kv_mirror.assign(kv_bytes_needed, 0);

	model->live_sequences += 1;
	ctx->live_handles += 1;
	*out_seq = h.release();
	return SSLM_OK;
}

// Design Sec5.3/Sec4.2's own state table: "frees the buffer and host state ...
// CallProceedsOrBusy_SeqRelease's existing policy, unchanged: returns Busy against
// Submitted; otherwise ... returns Ok." No `sslm_decode_step_gpu` exists yet (B5 lands
// it), so no `SslmGpuSequenceHandle` this section can construct is ever reachable in
// the `Submitted` state -- `CallProceedsOrBusy_SeqRelease` is reused verbatim anyway
// (not reimplemented as "always proceed"), so B5's own async lifecycle needs no change
// here when it starts setting `state = Submitted` for real.
SslmGpuStatus sslm_gpu_seq_release(SslmGpuContext* ctx, SslmGpuSequenceHandle* seq) {
	if (!seq) {
		return SSLM_OK;  // same null-is-a-no-op reasoning as sslm_gpu_context_destroy/
		                  // sslm_gpu_model_unmap(nullptr) above.
	}
	// T-2124 (D-SLM3446 P1-4): validate this handle was actually created against `ctx` -- see
	// sslm_gpu_model_unmap's own identical check and comment. `(void)ctx` (previously here) is
	// removed since the parameter is now read both here and by the decrement further down.
	if (seq->ctx != ctx) {
		return SSLM_SEQUENCE_KV_BUFFER_MISMATCH;
	}
	const bool is_submitted = seq->state == superslm_gpu::SslmSequenceGpuState::Submitted;
	if (!superslm_gpu::CallProceedsOrBusy_SeqRelease(is_submitted)) {
		return SSLM_BUSY;
	}
	if (seq->model && seq->model->live_sequences > 0) {
		seq->model->live_sequences -= 1;
	}
	if (seq->ctx->live_handles > 0) {
		seq->ctx->live_handles -= 1;
	}
	seq->destroyed = true;
	delete seq;
	return SSLM_OK;
}

// T-2113 (B3.5, design Sec5.3a/Sec10 B3.5, mini-fold 2026-08-15 routing D-SLM3367): the
// production token-feed entry point. Host-only -- no dispatch, no GPU state touched, no
// interaction with the async lifecycle or the dispatch geometry (design's own "issues no
// dispatch" framing is why this call needs neither B4 nor B5). Reuses the CPU path's own
// `EmbedEntry` primitive verbatim (forward_sites.h) against this handle's own model's
// host-retained embed_weights/embed_site_constant/vocab_size (B2's own amendment, above)
// -- the identical arithmetic RunWholeToken's own closure calls (forward_sites.cpp),
// called with the same trailing defaults (site="", token_index=0, no trace hook) so a
// caller driving this call pair reproduces that closure's own output byte-for-byte.
SslmGpuStatus sslm_gpu_seq_embed_token(SslmGpuContext* ctx, SslmGpuSequenceHandle* seq,
                                        int32_t token_id) {
	if (!ctx || !seq || !seq->model || seq->ctx != ctx) {  // T-2114 (S2): ->destroyed reads
	                                     // removed, see the sequence/model handles' own field
	                                     // comments. T-2124 (D-SLM3446 P1-4): `seq->ctx != ctx`.
		return SSLM_SEQUENCE_KV_BUFFER_MISMATCH;  // no channel exists for a malformed handle
		                                           // at this call, same "no more specific
		                                           // status" disposition every other
		                                           // malformed-handle site in this file uses.
	}
	// Design Sec5.3a: "seq state Idle (never Submitted -- returns Busy, the same convention
	// every other host-mutating call on a sequence handle already follows)."
	const bool is_submitted = seq->state == superslm_gpu::SslmSequenceGpuState::Submitted;
	if (!superslm_gpu::CallProceedsOrBusy_SeqReset(is_submitted)) {
		return SSLM_BUSY;  // CallProceedsOrBusy_SeqReset is the correct policy predicate to
		                    // reuse here: it is the SAME "proceed iff Idle/Completed, Busy
		                    // iff Submitted" rule design Sec5.9 assigns every host-mutating
		                    // sequence call (seq_save/seq_reset/seq_release all reuse the
		                    // identical predicate against their own distinct calls today).
	}

	SslmGpuModelHandle* model = seq->model;
	// F-S3-8 (forward_sites.cpp): validate token_id against [0, vocab_size) BEFORE any row
	// of embed_weights is read -- the identical bounds check EmbedEntry itself performs,
	// checked HERE too (not merely relying on EmbedEntry's own internal check) so this
	// call's own documented contract ("leaves seq's own state untouched" on rejection) is
	// satisfied by construction: nothing below this check ever writes to `seq`.
	if (token_id < 0 || token_id >= model->vocab_size) {
		return SSLM_TOKEN_ID_OUT_OF_RANGE;
	}

	std::vector<int8_t> embed_codes(model->hidden_size);
	superslm::CarriedScale embed_scale{};
	const superslm::SslmForwardStatus est = superslm::EmbedEntry(
	    token_id, model->vocab_size, model->embed_weights.data(),
	    static_cast<size_t>(model->hidden_size), model->embed_site_constant, embed_codes.data(),
	    &embed_scale);
	if (est != superslm::SslmForwardStatus::Ok) {
		// EmbedEntry's own only rejection is the identical token_id bounds check already
		// performed above (forward_sites.cpp:757) -- unreachable in practice given the
		// guard above, handled anyway per this file's own "an unhandled path throws loudly
		// rather than silently miscomputing" precedent (SslmGpuSequenceHandle's own
		// SequenceKvBufferMismatch structural self-check, sslm_gpu_seq_create, above).
		return SSLM_TOKEN_ID_OUT_OF_RANGE;
	}

	// Design Sec5.3a: "overwrites seq's own host-mirrored hidden_codes/hidden_scale ...
	// and resets layer_index to 0" -- matching RunWholeToken's own reset before a fresh
	// layer pass (forward_sites.cpp). `seq->live_state.hidden_codes` already aliases
	// `seq->hidden_codes.data()` (set once at sslm_gpu_seq_create) -- writing through
	// `seq->hidden_codes` keeps both in sync without a second copy.
	std::copy(embed_codes.begin(), embed_codes.end(), seq->hidden_codes.begin());
	seq->hidden_scale = embed_scale;
	seq->layer_index = 0;
	seq->live_state.hidden_scale = seq->hidden_scale;
	seq->live_state.layer_index = 0;
	return SSLM_OK;
}

namespace {
// T-2113 (B5): `RunLayerLoopGpuSubmit`'s own guard ladder (superslm_gpu.cpp, unchanged
// from the pre-1.0 substrate) rejects for reasons the 1.0 API's own construction already
// makes unreachable in ordinary use for MOST guards -- context_cap/hidden_size/head_dim
// are validated once, at sslm_gpu_seq_create/sslm_gpu_model_map time, never re-supplied
// per decode call. Numeric-domain guards (ChainInputOutOfDomain,
// SiluCompositionScaleOutOfDomain, SoftmaxRowWidthOutOfDomain, and the rest of
// superslm::SslmForwardStatus's own CPU-domain family) ARE reachable through this API --
// a poisoned adapter delta or a pathological activation value reaches them exactly the
// way the CPU path always could.
//
// T-2114 (S1, Claude/Poirot/50f3d5d-t2113-1p0-gpu-core-build-review.md): `st` is
// whatever `RunLayerLoopGpuSubmit` RETURNED when `inflight == nullptr` -- confirmed by
// execution (not assumed) to include TWO distinct classes, not one: the ordinary
// CPU-domain guard ladder (ChainInputOutOfDomain and the rest of the family, checked
// BEFORE any device work begins), AND `GpuDeviceRemoved`/`GpuAllocationFailed`, which
// `RunLayerLoopGpuSubmit`'s own internal `catch (const std::runtime_error&)`
// (superslm_gpu.cpp) produces when a REAL device/HR failure happens mid-recording and
// converts it to a normal return rather than letting it escape as an exception. An
// earlier version of this function (and this comment) assumed only the first class
// could reach here -- refuted by running this suite's own dim9 cells, which surfaced a
// real `GpuAllocationFailed` through this exact path. Design Sec9 still assigns no
// per-guard 1.0 status to the CPU-domain family (SSLM_SEQUENCE_REJECTED covers all of
// it, undifferentiated, on purpose), but the two device-derived statuses must map to
// SSLM_DEVICE_LOST -- the SAME distinction MapDecodedStatusToGpuStatus below draws for
// the identical two statuses arriving through the FINISH path instead.
SslmGpuStatus MapSubmitRejectionToGpuStatus(superslm::SslmForwardStatus st) {
	if (st == superslm::SslmForwardStatus::GpuDeviceRemoved ||
	    st == superslm::SslmForwardStatus::GpuAllocationFailed) {
		return SSLM_DEVICE_LOST;
	}
	return SSLM_SEQUENCE_REJECTED;
}
// T-2113 (B5): the FINAL, GPU-decoded per-call result (DecodeStickyTag's own return,
// design Sec4.2's own `sslm_gpu_ready`'s `out_status` channel) -- `Ok` is the only
// non-rejecting value.
//
// T-2114 (S1): `RunLayerLoopGpuFinish`'s own catch block (superslm_gpu.cpp) is the ONE
// path that can deliver a genuine device-level failure here -- it returns
// `GpuDeviceRemoved` (device confirmed gone) or `GpuAllocationFailed` (a transient/
// size-dependent failure at the readback, device still alive) when a real HR exception
// unwound the finish call. Every OTHER non-Ok value `DecodeStickyTag` can produce is a
// CPU-domain-equivalent guard rejection the GPU dispatch chain itself found (mirroring
// the CPU oracle's own sticky-tag family) -- a healthy-device, per-sequence fact, not a
// device loss. Only the two device-derived statuses map to SSLM_DEVICE_LOST; every other
// rejecting value maps to SSLM_SEQUENCE_REJECTED, the same status
// MapSubmitRejectionToGpuStatus uses for the identical class of fact discovered before
// submission instead of after.
SslmGpuStatus MapDecodedStatusToGpuStatus(superslm::SslmForwardStatus st) {
	if (st == superslm::SslmForwardStatus::Ok) return SSLM_OK;
	if (st == superslm::SslmForwardStatus::GpuDeviceRemoved ||
	    st == superslm::SslmForwardStatus::GpuAllocationFailed) {
		return SSLM_DEVICE_LOST;
	}
	return SSLM_SEQUENCE_REJECTED;
}

// T-2113 (B7, design Sec4.3/Sec7/Sec10 B7): the actual submission logic shared, byte-for-byte,
// between the single-sequence call (`sslm_decode_step_gpu`) and every per-sequence slot of the
// batch call (`sslm_decode_step_batch_gpu`) below -- extracted so the two entry points can never
// silently drift on what "submit one sequence's own dispatch chain" means (StandardsDocument.md
// Sec5.4's own "one implementation, not two copies that could drift" discipline, the identical
// reasoning D-SLM3352 already applied to weight packing). Takes an ALREADY-DECIDED
// `layers_to_issue` (design Sec7: "each sequence taking whole layers only... a sequence reached
// after the budget is exhausted" is a decision the CALLER makes against whichever budget
// (single-call: the call's own `dispatch_budget`; batch: the running batch-wide remainder) --
// this function has no opinion on where the number came from, only on submitting exactly that
// many layers for exactly this one sequence. Caller-ensures `ctx`/`seq`/`seq->model` are non-null,
// non-destroyed, and `seq` is not `Submitted` (every guard above this call's own callers already
// check, per-sequence, before reaching here) -- this function re-validates none of that, matching
// `RunLayerLoopGpuSubmit`'s own "guards are the caller's problem below this layer" convention.
SslmGpuStatus SubmitOneSequenceDecode(SslmGpuContext* ctx, SslmGpuSequenceHandle* seq,
                                       const SslmGpuAdapterHandle* adapter_or_null,
                                       uint32_t layers_to_issue) {
	(void)ctx;  // not read directly -- every resource this call touches is reached through `seq`/
	            // `seq->model`/`adapter_or_null`'s own already-resident buffers (B2/B3/B6), the
	            // identical shape `sslm_decode_step_gpu`'s own pre-B7 body already had.
	SslmGpuModelHandle* model = seq->model;

	// Sync the handle's own canonical scalar fields into `live_state` (its own `hidden_codes`
	// pointer was set once, at sslm_gpu_seq_create, and needs no per-call refresh).
	seq->live_state.hidden_scale = seq->hidden_scale;
	seq->live_state.layer_index = seq->layer_index;
	seq->live_state.kv_saturation_count = seq->kv_saturation_count;
	seq->live_state.context_length = seq->context_length;

	static const superslm::SslmTensorManifest kEmptyManifest;

	superslm_gpu::GpuAdapterBridge adapter_bridge;
	const superslm_gpu::GpuAdapterBridge* adapter_bridge_ptr = nullptr;
	if (adapter_or_null != nullptr) {
		adapter_bridge.lora_ab_resident = adapter_or_null->lora_ab_buf.Get();
		adapter_bridge.fold_resident = adapter_or_null->fold_buf.Get();
		adapter_bridge.rank = adapter_or_null->rank;
		adapter_bridge.slots = &adapter_or_null->slots[0][0];
		adapter_bridge.num_hidden_layers = model->num_hidden_layers;
		adapter_bridge_ptr = &adapter_bridge;
	}

	superslm_gpu::GpuLayerLoopInFlight* inflight = nullptr;
	const superslm::SslmForwardStatus submit_status = superslm_gpu::RunLayerLoopGpuSubmit(
	    seq->live_state, /*layers=*/nullptr, model->num_hidden_layers, layers_to_issue,
	    model->hidden_size, model->head_dim, model->num_key_value_heads, model->intermediate_size,
	    seq->context_cap, kEmptyManifest, seq->host_kv_mirror.data(), seq->host_kv_mirror.size(),
	    seq->kv_buf.Get(), &seq->kv_needs_resume_barrier, &inflight, model->weights_buf.Get(),
	    model->rope_cos_buf.Get(), model->rope_sin_buf.Get(), model->has_rope_tables,
	    model->rope_cos_elem_count, model->rope_sin_elem_count, adapter_bridge_ptr);

	if (!inflight) {
		// Rejected before submission (a guard, or an exception) -- seq/host state untouched,
		// still Idle, exactly RunLayerLoopGpu's own existing contract.
		return MapSubmitRejectionToGpuStatus(submit_status);
	}
	seq->in_flight = inflight;
	seq->state = superslm_gpu::SslmSequenceGpuState::Submitted;
	// T-2113 (design Sec4.2/Sec9, D-SLM3417): model's own Busy-precedence count -- see
	// SslmGpuModelHandle::submitted_sequences' own field comment, above.
	model->submitted_sequences += 1;
	// T-2124 (D-SLM3446 P0-3): the adapter-handle analogue of the line above -- see
	// SslmGpuAdapterHandle::submitted_sequences' own field comment. `seq->in_flight_adapter`
	// remembers which adapter (if any) this in-flight submission bound, so sslm_gpu_ready knows
	// which adapter's count to decrement once this sequence collapses back to Idle -- the adapter
	// binding is a per-call argument, not state this function's own caller otherwise retains on
	// `seq`.
	seq->in_flight_adapter = adapter_or_null;
	if (adapter_or_null != nullptr) {
		adapter_or_null->submitted_sequences += 1;
	}
	return SSLM_OK;
}
}  // namespace

// Design Sec4.3/Sec6.2 (T-2113 B5): "records up to dispatch_budget dispatches ... into
// ONE command list, closes it, submits it, and returns without waiting for the fence."
// Routes weights/RoPE through `model`'s own resident buffers (B2, `weights_buf`/
// `rope_cos_buf`/`rope_sin_buf` -- the external-weights/external-rope bridge B5 adds to
// `RunLayerLoopGpuSubmit`) and K/V through `seq`'s own dedicated buffer (B3, the existing
// external-KV bridge) -- this is the production measurement path that retires
// `g_resident_rope`/`g_resident_weights` for every call that reaches here (D-SLM3362):
// neither process-global cache is ever read or written on this path.
SslmGpuStatus sslm_decode_step_gpu(SslmGpuContext* ctx, SslmGpuSequenceHandle* seq,
                                    const SslmGpuAdapterHandle* adapter_or_null,
                                    uint32_t dispatch_budget) {
	// Design Sec8/Sec10 B6: per-sequence adapter binding is a call argument, checked at every
	// call (design Sec5.2: "checked at every sslm_decode_step_gpu/_batch_gpu call... a
	// pointer-equality check against the handle each side already carries, not a re-hash").
	// T-2113 (B6b, design Sec8): the residency and guard checks below are real, AND the GPU
	// dispatch-recording path this call submits through (RunLayerLoopGpuSubmit) now reads the
	// adapter's own resident buffers when `adapter_or_null` validates -- the GEMM-site
	// delta-application dispatches, via the GpuAdapterBridge built below.
	if (!ctx || !seq || !seq->model || seq->ctx != ctx) {  // T-2114 (S2): ->destroyed reads
	                                     // removed, see the sequence/model handles' own field
	                                     // comments. T-2124 (D-SLM3446 P1-4): `seq->ctx != ctx`.
		return SSLM_SEQUENCE_KV_BUFFER_MISMATCH;  // no channel exists for a malformed handle
		                                           // at this call besides the structural-
		                                           // invariant status (same "no more specific
		                                           // status" disposition as the helpers above).
	}
	if (adapter_or_null && adapter_or_null->model != seq->model) {  // T-2114 (S2): ->destroyed
	                                                                  // read removed.
		return SSLM_ADAPTER_MODEL_MISMATCH;  // design Sec5.2/Sec9: pointer-equality against the
		                                      // model handle each side already carries.
	}
	if (adapter_or_null && adapter_or_null->ctx != ctx) {  // T-2124 (D-SLM3446 P1-4): a bound
	                                                        // adapter mapped against a different
	                                                        // context names a different device.
		return SSLM_SEQUENCE_KV_BUFFER_MISMATCH;  // same "no channel exists for a malformed
		                                           // handle" disposition as the seq check above.
	}
	const bool is_submitted = seq->state == superslm_gpu::SslmSequenceGpuState::Submitted;
	if (!superslm_gpu::CallProceedsOrBusy_DecodeStepGpu(is_submitted)) {
		return SSLM_BUSY;  // design Sec4.2's own state table, unchanged policy.
	}

	SslmGpuModelHandle* model = seq->model;
	uint32_t layers_to_issue = 0;
	const superslm_gpu::SslmGpuStatus plan = superslm_gpu::PlanDispatchBudgetGpu(
	    dispatch_budget, model->num_hidden_layers, seq->layer_index, &layers_to_issue);
	if (plan == superslm_gpu::SslmGpuStatus::DispatchBudgetTooSmall) {
		return SSLM_DISPATCH_BUDGET_TOO_SMALL;
	}

	// T-2113 (B7): submission itself is shared, byte-for-byte, with every per-sequence slot
	// of sslm_decode_step_batch_gpu (below) -- see SubmitOneSequenceDecode's own header
	// comment. This call has already decided layers_to_issue against its OWN full
	// dispatch_budget (immediately above); the batch call decides the identical quantity
	// against a running batch-wide remainder instead -- the only difference between the two
	// call sites, and it lives entirely in the budget arithmetic surrounding this shared call,
	// never inside it.
	return SubmitOneSequenceDecode(ctx, seq, adapter_or_null, layers_to_issue);
}

// Design Sec4.3/Sec7/Sec10 B7 (T-2113 B7, repaired signature per the T-2111 fold, D-SLM3344):
// "records n sequences' own, independent, per-sequence dispatch chains ... the batch call
// changes submission grouping, never any sequence's own numerics." `dispatch_budget` is
// BATCH-WIDE (design Sec7): consumed by sequences strictly in `seqs[]`'s own array order, each
// sequence taking whole layers only from whatever remains when its own turn comes. A guard
// rejection for `seqs[i]` (a malformed handle, an adapter/model mismatch, a `Busy` sequence)
// does not abort the batch -- `out_statuses[i]` carries that sequence's own outcome and
// recording continues with `seqs[i+1]`. A sequence reached once the running remainder cannot
// cover even one whole layer records zero dispatches and reads `SSLM_BATCH_BUDGET_EXHAUSTED`
// (never `SSLM_DISPATCH_BUDGET_TOO_SMALL`, which design Sec9's own channel table reserves for
// the single-sequence call) -- its own state stays `Idle`, exactly as if it had never been
// passed to this call at all. The function's own RETURN VALUE is call-level only (design
// Sec4.3): whether the call proceeded at all (`Ok`) -- never a reduction of `out_statuses`.
//
// **Named limitation, stated per Brunel's own "stop at genuine limits, honest stage report"
// discipline, not silently absorbed**: this implementation submits each recorded sequence
// through the IDENTICAL `RunLayerLoopGpuSubmit` call the single-sequence path uses --
// `SubmitOneSequenceDecode`, shared verbatim -- which means every sequence in the batch still
// gets its OWN `ExecuteCommandLists`/`Signal` pair underneath, not the ONE-command-list fusion
// design Sec7's own text describes ("records n sequences ... into ONE command list and
// submits/fences them together"). The CORRECTNESS contract this section's own gate names
// (design Sec10 B7: "per sequence, output bit-identical to that same sequence decoded alone")
// is satisfied by construction -- reusing the exact same, already-proven submission call for
// every sequence makes bit-identity to the serial case a property of the code being literally
// the same code, not a separately-proven claim -- and the per-sequence `out_statuses`/
// batch-wide-budget/mixed-adapter semantics are fully and correctly implemented. What is NOT
// realized is the submission-side throughput amortization design Sec7 names as the batching
// mechanism's OWN rationale (a real, but explicitly out-of-scope-for-this-gate, cost: design
// Sec13 states the batch call's own throughput is "UNDERIVED... an empirical question," never
// a design-time claim, and Sec10 B7's own gate text is itself "a correctness proof, not a
// throughput proof"). §14's own throughput measurement, below, reports this honestly rather
// than assuming batching helps.
//
// **A REAL device hang, found at the bench, is why this is not merely a missed optimization**:
// every 1.0-API decode call -- single or batched -- ultimately submits through the process's ONE
// shared `harness::GetDevice()` singleton (every `SslmGpuContext::device`, B1, is used only for
// RESIDENCY uploads at map/create time, never for the decode dispatch chain itself, which the
// pre-1.0 substrate's own `RunLayerLoopGpuSubmit` still routes through the singleton).
// `ID3D12CommandAllocator::Reset()` is undefined behavior while a command list allocated from it
// is still executing on the GPU -- calling `SubmitOneSequenceDecode` for `seqs[1]` before
// `seqs[0]`'s own fence has signaled resets that SAME shared allocator out from under a submission
// that may still be executing. A first draft of this function left every recorded sequence
// Submitted (undrained) until the whole loop finished, matching the design's own async-batch
// framing literally -- and hung the test process indefinitely on the very first multi-sequence
// batch it ever ran (`t2113_b7_batch_smoke.cpp`'s own Gate 1a, reproduced by execution: 12.12
// CPU-seconds, motionless, for several minutes -- a real device-level stall, not a slow
// computation). Fixed below: each sequence is drained via the public `sslm_gpu_ready(block=1)`
// the INSTANT its own submission succeeds, before the loop's next iteration can touch the shared
// allocator again -- at most one sequence is ever Submitted against the device at once. This is
// the same constraint that makes the "ONE command list" fusion design Sec7 describes a genuine
// correctness requirement for a from-scratch implementation, not merely a throughput nicety --
// this implementation avoids needing that fusion by never having two sequences in flight
// simultaneously in the first place, at the cost named above (no cross-sequence submission
// overlap, so no submission-side amortization either).
SslmGpuStatus sslm_decode_step_batch_gpu(SslmGpuContext* ctx, SslmGpuSequenceHandle* const* seqs,
                                          const SslmGpuAdapterHandle* const* adapters_or_null,
                                          uint32_t n_sequences, uint32_t dispatch_budget,
                                          SslmGpuStatus* out_statuses) {
	if (!ctx) {
		return SSLM_SEQUENCE_KV_BUFFER_MISMATCH;  // call-level: no live object to report through,
		                                           // same disposition every other null-ctx site
		                                           // in this file already uses.
	}
	if (n_sequences > 0 && (!seqs || !out_statuses)) {
		// A non-zero count with a null seqs/out_statuses array is a caller contract violation
		// this call cannot record anything against -- no per-sequence channel exists to report
		// through (out_statuses is itself null), so this is call-level, matching the "no live
		// object to report through" disposition every other malformed-argument site uses.
		return SSLM_SEQUENCE_KV_BUFFER_MISMATCH;
	}
	// n_sequences == 0: a legal, vacuous call (design Sec4.3 states no floor on n_sequences) --
	// nothing to record, nothing to report, proceeds and returns Ok.

	uint32_t remaining_budget = dispatch_budget;
	for (uint32_t i = 0; i < n_sequences; ++i) {
		SslmGpuSequenceHandle* seq = seqs[i];
		const SslmGpuAdapterHandle* adapter_or_null = adapters_or_null ? adapters_or_null[i] : nullptr;

		// Per-sequence guard ladder -- IDENTICAL checks and IDENTICAL statuses to the
		// single-sequence call (design Sec9: AdapterModelMismatch's own channel table entry
		// names `out_statuses[i]` explicitly as the batch call's own per-sequence channel for
		// exactly this check). A guard rejection for seqs[i] does NOT abort the batch (design
		// Sec7's own "records every sequence's chain up to... that sequence's own guard
		// failure... skips that sequence's own remaining dispatches, and continues recording
		// the next sequence") -- `continue` to seqs[i+1], remaining_budget UNCHANGED (a
		// rejected sequence consumes no budget, since nothing was recorded for it).
		if (!seq || !seq->model || seq->ctx != ctx) {  // T-2114 (S2): ->destroyed reads removed,
		                            // see the sequence/model handles' own field comments.
		                            // T-2124 (D-SLM3446 P1-4): `seq->ctx != ctx`.
			out_statuses[i] = SSLM_SEQUENCE_KV_BUFFER_MISMATCH;
			continue;
		}
		if (adapter_or_null && adapter_or_null->model != seq->model) {  // T-2114 (S2): ->destroyed
		                                                                 // read removed.
			out_statuses[i] = SSLM_ADAPTER_MODEL_MISMATCH;
			continue;
		}
		if (adapter_or_null && adapter_or_null->ctx != ctx) {  // T-2124 (D-SLM3446 P1-4): a bound
		                                                        // adapter mapped against a
		                                                        // different context.
			out_statuses[i] = SSLM_SEQUENCE_KV_BUFFER_MISMATCH;
			continue;
		}
		const bool is_submitted = seq->state == superslm_gpu::SslmSequenceGpuState::Submitted;
		if (!superslm_gpu::CallProceedsOrBusy_DecodeStepGpu(is_submitted)) {
			out_statuses[i] = SSLM_BUSY;
			continue;
		}

		// Design Sec7: "dispatch_budget is consumed by sequences strictly in the recording
		// order... each sequence taking whole layers only... from whatever remains" -- planned
		// against the RUNNING remainder, never the call's own original dispatch_budget past
		// the first sequence. Zero layers available (whether because the ORIGINAL budget never
		// covered even seqs[0]'s own first layer, or because earlier sequences in this same
		// call already consumed it) is uniformly SSLM_BATCH_BUDGET_EXHAUSTED at the batch call
		// -- design Sec9's own table reserves SSLM_DISPATCH_BUDGET_TOO_SMALL for the
		// single-sequence call alone, never naming it as a batch-call channel.
		uint32_t layers_to_issue = 0;
		const superslm_gpu::SslmGpuStatus plan = superslm_gpu::PlanDispatchBudgetGpu(
		    remaining_budget, seq->model->num_hidden_layers, seq->layer_index, &layers_to_issue);
		if (plan == superslm_gpu::SslmGpuStatus::DispatchBudgetTooSmall) {
			out_statuses[i] = SSLM_BATCH_BUDGET_EXHAUSTED;  // seq->state stays Idle -- nothing
			                                                  // recorded, nothing submitted.
			continue;
		}

		out_statuses[i] = SubmitOneSequenceDecode(ctx, seq, adapter_or_null, layers_to_issue);
		if (out_statuses[i] == SSLM_OK) {
			// Design Sec7's own budget arithmetic: exactly this sequence's own consumed
			// dispatches (layers_to_issue whole layers, 24 dispatches/layer, T-2113 B4) leave
			// the running remainder for seqs[i+1..]. A submission REJECTED after planning
			// succeeded (e.g. a device-lost mid-batch, design Sec9's own DeviceLost row: "every
			// not-yet-reached out_statuses[i] reading DeviceLost") consumes no budget either --
			// only a call that actually reached SSLM_OK genuinely used GPU dispatch slots.
			// T-2114 (M1): kDispatchesPerLayer (gpu_port.h) is now the one source this line and
			// PlanDispatchBudgetGpu's own body (superslm_gpu.cpp) both read -- a bare `24u` here
			// used to be a second copy that could silently drift from PlanDispatchBudgetGpu's
			// own constant and wrap this unsigned subtraction.
			remaining_budget -= layers_to_issue * superslm_gpu::kDispatchesPerLayer;

			// T-2113 (B7): a REAL device hang found at the bench, fixed here, not merely
			// worked around -- see this function's own header comment for the fuller account.
			// `RunLayerLoopGpuSubmit` always operates on the process's ONE shared
			// `harness::GetDevice()` singleton's own command allocator/list/queue (every 1.0
			// context's own `SslmGpuContext::device` is used only for RESIDENCY uploads, B1/B2,
			// never for the decode dispatch chain itself). `ID3D12CommandAllocator::Reset()` is
			// undefined behavior while a command list allocated from it is still executing on
			// the GPU (D3D12's own documented contract) -- submitting seqs[i+1]'s own dispatches
			// (which calls `dev.alloc->Reset()` again, unconditionally, at the top of the NEXT
			// `RunLayerLoopGpuSubmit`) BEFORE seqs[i]'s own fence has signaled is exactly that:
			// two submissions against the SAME shared allocator with no fence boundary between
			// them. Reproduced by execution: `t2113_b7_batch_smoke.cpp`'s own Gate 1a (a
			// batch-of-3, back-to-back submissions with no drain between them) hung the process
			// indefinitely (12.12 CPU-seconds, unmoving, for several minutes -- a genuine device-
			// level stall, not a slow computation) the first time this loop left more than one
			// sequence Submitted at once. Fixed: each sequence is drained (via the EXISTING
			// public `sslm_gpu_ready(block=1)`, never a second, parallel implementation) the
			// MOMENT its own submission succeeds, before the loop's next iteration ever calls
			// `dev.alloc->Reset()` again -- at most one sequence is ever Submitted against the
			// shared device at a time. This is also the concrete, now-necessary (not merely
			// named-as-a-cost) form of this function's own "no one-command-list fusion" limit:
			// the batch call is observably synchronous per sequence internally, though `seq`'s
			// own state still transitions through `Submitted` correctly, so a caller polling via
			// `sslm_gpu_ready` afterward always finds `ready=1` immediately -- a legal realization
			// of the async contract (design Sec4.2 never requires a call to still be pending).
			int32_t drained_ready = 0;
			SslmGpuStatus drained_status = SSLM_OK;
			const SslmGpuStatus ready_call_status =
			    sslm_gpu_ready(ctx, seq, /*block=*/1, &drained_ready, &drained_status);
			out_statuses[i] = (ready_call_status == SSLM_OK) ? drained_status : ready_call_status;
		}
		if (out_statuses[i] == SSLM_DEVICE_LOST) {
			// Design Sec9's own DeviceLost row for the batch call: "every not-yet-reached
			// out_statuses[i] reading DeviceLost" -- a device removal discovered mid-batch must
			// not let the REST of the batch's own not-yet-recorded sequences silently proceed
			// as if the device were still healthy (their own dispatches, if recorded, would
			// execute against a lost device with undefined results). Every remaining slot is
			// marked DeviceLost and recording stops -- this is the one case design Sec7's own
			// per-sequence-independence text does not apply to, because the SHARED device
			// itself, not any one sequence's own state, is what failed.
			for (uint32_t j = i + 1; j < n_sequences; ++j) out_statuses[j] = SSLM_DEVICE_LOST;
			break;
		}
	}
	return SSLM_OK;  // call-level: recording proceeded (design Sec4.3) -- per-sequence outcomes
	                  // are exclusively in out_statuses, never folded into this return value.
}

// Design Sec4.2 (T-2113 B5): "block(=0): a pure poll -- if the fence has not yet
// signaled, *out_ready=0, no state change ... block(=1): blocks the calling thread on
// the fence rather than returning *out_ready=0 -- the only call this design allows to
// legitimately fence-wait." `Idle`/`Completed` (nothing outstanding): `*out_ready=1`
// immediately, `*out_status=Ok`.
SslmGpuStatus sslm_gpu_ready(SslmGpuContext* ctx, SslmGpuSequenceHandle* seq, int32_t block,
                              int32_t* out_ready, SslmGpuStatus* out_status) {
	if (out_ready) *out_ready = 0;
	if (out_status) *out_status = SSLM_OK;
	if (!ctx || !seq || seq->ctx != ctx) {  // T-2114 (S2): ->destroyed read removed, see sequence
	                                         // handle's own field comment. T-2124 (D-SLM3446
	                                         // P1-4): `seq->ctx != ctx`.
		return SSLM_SEQUENCE_KV_BUFFER_MISMATCH;  // no channel exists for a malformed handle
	}
	if (seq->state != superslm_gpu::SslmSequenceGpuState::Submitted) {
		// design Sec4.2: "Idle/Completed: *out_ready=1 immediately, *out_status=Ok
		// (nothing outstanding)."
		if (out_ready) *out_ready = 1;
		return SSLM_OK;
	}

	int32_t ready = 0;
	const superslm::SslmForwardStatus decoded = superslm_gpu::RunLayerLoopGpuFinish(
	    seq->in_flight, seq->live_state, seq->host_kv_mirror.data(), block, &ready);
	if (!ready) {
		// Still Submitted -- a non-blocking poll against an unsignaled fence, design
		// Sec4.2's own "no state change" case. `seq->in_flight` is untouched by Finish
		// on this path (still owned by this handle, still pollable next call).
		return SSLM_OK;
	}

	// Fence signaled (or block=1 waited for it) -- collapse Submitted -> Idle in one
	// call, design Sec4.2's own contract, and copy the decoded result back into this
	// handle's own canonical fields.
	seq->in_flight = nullptr;  // RunLayerLoopGpuFinish already freed the token on this path
	seq->state = superslm_gpu::SslmSequenceGpuState::Idle;
	// T-2113 (design Sec4.2/Sec9, D-SLM3417): the model's own Busy-precedence count -- the
	// symmetric twin of SubmitOneSequenceDecode's own increment, above.
	if (seq->model && seq->model->submitted_sequences > 0) {
		seq->model->submitted_sequences -= 1;
	}
	// T-2124 (D-SLM3446 P0-3): the adapter-handle analogue -- the symmetric twin of
	// SubmitOneSequenceDecode's own `adapter_or_null->submitted_sequences += 1` above. The fence
	// has now signaled (this is only reached once `ready` is true), so the GPU is provably done
	// reading the adapter's own lora_ab_buf/fold_buf for this submission -- safe for
	// sslm_gpu_adapter_unmap to proceed once this count reaches zero.
	if (seq->in_flight_adapter && seq->in_flight_adapter->submitted_sequences > 0) {
		seq->in_flight_adapter->submitted_sequences -= 1;
	}
	seq->in_flight_adapter = nullptr;
	seq->hidden_scale = seq->live_state.hidden_scale;
	seq->layer_index = seq->live_state.layer_index;
	seq->kv_saturation_count = seq->live_state.kv_saturation_count;
	seq->context_length = seq->live_state.context_length;
	if (out_ready) *out_ready = 1;
	if (out_status) *out_status = MapDecodedStatusToGpuStatus(decoded);
	return SSLM_OK;
}

// Design Sec4.2/Sec5.3 (T-2113 B5): the device-resident save/restore/reset trio,
// declared for B3/B5 (gpu_1p0.h's own header comment) -- reusing
// `SaveGpuSequenceState`/`RestoreGpuSequenceState` (gpu_port.h, the substrate's own
// existing blob mechanism, D-SLM3080) rather than a second implementation. T-2114 (C1)
// extended both functions to also round-trip `hidden_codes` -- see gpu_port.h's own
// header comment on the pair for the current contract.
SslmGpuStatus sslm_gpu_seq_save(SslmGpuContext* ctx, const SslmGpuSequenceHandle* seq,
                                 void* out_blob, size_t* out_blob_size) {
	if (!ctx || !seq || seq->ctx != ctx) {  // T-2114 (S2): ->destroyed read removed, see sequence
	                                         // handle's own field comment. T-2124 (D-SLM3446
	                                         // P1-4): `seq->ctx != ctx`.
		return SSLM_SEQUENCE_KV_BUFFER_MISMATCH;
	}
	const bool is_submitted = seq->state == superslm_gpu::SslmSequenceGpuState::Submitted;
	if (!superslm_gpu::CallProceedsOrBusy_SeqSave(is_submitted)) {
		return SSLM_BUSY;  // design Sec4.2's own state table, unchanged policy.
	}
	// T-2114 (C1): hidden_codes_size = seq->hidden_codes.size() -- the 1.0 handle owns this
	// vector (design Sec5.3), so the residual stream now round-trips through the blob
	// alongside the K/V workspace bytes, closing the class of loss a restored, non-zero
	// hidden_scale paired with a zeroed residual used to produce silently.
	// T-2113 (P2, design Sec4.2/Sec22, D-SLM3415): `seq->model->content_hash` -- the model this
	// sequence was created against, already retained (design Sec5.1) -- is written into the v3
	// blob's own trailing `model_content_hash` field so a later restore can detect a mismatch
	// against a DIFFERENT target model.
	const bool ok = superslm_gpu::SaveGpuSequenceState(seq->live_state, seq->hidden_codes.size(),
	                                                    seq->host_kv_mirror.data(),
	                                                    seq->host_kv_mirror.size(),
	                                                    seq->model->content_hash, out_blob,
	                                                    out_blob_size);
	// SaveGpuSequenceState's own contract (gpu_port.h): a too-small out_blob reports the
	// required size via *out_blob_size and returns false -- not a device/guard failure,
	// no dedicated 1.0 status exists for it either, same disposition as the guards above.
	return ok ? SSLM_OK : SSLM_DEVICE_LOST;
}

SslmGpuStatus sslm_gpu_seq_restore(SslmGpuContext* ctx, SslmGpuModelHandle* model,
                                    const void* blob, size_t blob_size,
                                    SslmGpuSequenceHandle** out_seq) {
	if (!out_seq) {
		return SSLM_SEQUENCE_KV_BUFFER_MISMATCH;
	}
	*out_seq = nullptr;
	if (!ctx || !model || !blob || model->ctx != ctx) {  // T-2114 (S2): ->destroyed read removed,
	                                                       // see model handle's own field comment.
	                                                       // T-2124 (D-SLM3446 P1-4):
	                                                       // `model->ctx != ctx`.
		return SSLM_SEQUENCE_KV_BUFFER_MISMATCH;
	}
	// Design Sec4.2/Sec21 (T-2114 N1, corrected 2026-08-15, routing
	// `Claude/Poirot/50f3d5d-t2113-1p0-gpu-core-build-review.md` Sec6/Sec8): "sslm_gpu_seq_restore
	// allocates a fresh SslmGpuSequenceHandle... never reusing an existing handle's" -- sized to the
	// BLOB's own recorded context_cap, not the model's. Reading the header first (never the body,
	// never a device call) lets this derive that cap before anything is allocated: `workspace_size`
	// solves `workspace_size = num_hidden_layers * context_cap * num_key_value_heads * head_dim * 2`
	// (Sec5.1/D-SLM3310's own K/V sizing formula) for context_cap, with every other term already
	// known through `model`. A blob too small/wrong-magic to hold a header, a division with a
	// nonzero remainder, or a derived context_cap <= 0 is malformed; a derived context_cap that
	// EXCEEDS model->context_cap is inadmissible (this model cannot represent a sequence that large)
	// -- both rejected under the same disposition every other structural refusal in this function
	// uses. A derived context_cap strictly less than the model's own maximum is the ordinary case
	// this correction exists to admit, not an error: the prior always-model-cap sizing made restoring
	// a short, cheaply-capped sequence -- the common case, since context_cap is a sslm_gpu_seq_create
	// parameter for exactly the reason not every sequence needs the model's full capacity -- the one
	// case that always failed the size check below, loudly but wrongly scoped.
	uint64_t blob_workspace_size = 0;
	if (!superslm_gpu::PeekGpuSeqBlobWorkspaceSize(blob, blob_size, &blob_workspace_size)) {
		return SSLM_SEQUENCE_KV_BUFFER_MISMATCH;
	}
	// Identical per-context_cap-unit byte count to sslm_gpu_seq_create's own kv_bytes_needed
	// computation (this file, above), factored out of context_cap so dividing workspace_size by it
	// recovers context_cap -- the same overflow-guarded product, computed the same way.
	size_t per_cap_unit_bytes = static_cast<size_t>(model->num_hidden_layers);
	const size_t per_cap_unit_factors[] = {model->num_key_value_heads,
	                                        static_cast<size_t>(model->head_dim), 2u};
	bool per_cap_unit_overflowed = false;
	for (size_t factor : per_cap_unit_factors) {
		if (factor != 0 && per_cap_unit_bytes > SIZE_MAX / factor) {
			per_cap_unit_overflowed = true;
			break;
		}
		per_cap_unit_bytes *= factor;
	}
	if (per_cap_unit_overflowed || per_cap_unit_bytes == 0 ||
	    blob_workspace_size % static_cast<uint64_t>(per_cap_unit_bytes) != 0) {
		return SSLM_SEQUENCE_KV_BUFFER_MISMATCH;  // malformed: not an integral context_cap
	}
	const uint64_t derived_context_cap = blob_workspace_size / static_cast<uint64_t>(per_cap_unit_bytes);
	if (derived_context_cap == 0 ||
	    derived_context_cap > static_cast<uint64_t>(model->context_cap)) {
		// derived_context_cap == 0: malformed (a real save never produces a zero-cap sequence).
		// derived_context_cap > model->context_cap: inadmissible -- this model cannot create a
		// handle large enough to hold what the blob claims to be. Both share the one disposition
		// every structural rejection in this function already uses (design Sec4.2/Sec21: "no new
		// status is minted for this").
		return SSLM_SEQUENCE_KV_BUFFER_MISMATCH;
	}

	// T-2113 (P2, design Sec4.2/Sec9/Sec22, routed `Claude/Poirot/50f3d5d-t2113-1p0-gpu-core-
	// build-review.md` Sec15, D-SLM3415): the identity check the N1 size-admissibility widening
	// (above) left open -- a foreign blob whose derived context_cap happens to be admissible
	// against a same-shape-but-different model would otherwise restore silently. Checked AFTER
	// the size-derivation ladder (a structurally malformed blob is rejected for that reason
	// first, before its content hash is even read) and BEFORE any device work -- the fresh
	// handle is not created until this passes. A blob whose header cannot even be re-read here
	// (should be unreachable: PeekGpuSeqBlobWorkspaceSize already validated the identical
	// magic/size preconditions above) falls under the same structural-malformed disposition
	// every other rejection in this function uses; a hash that IS read but does not match
	// `model`'s own returns the distinctly-named SSLM_RESTORE_MODEL_MISMATCH (design Sec9) --
	// never folded into the generic malformed-blob status, since a wrong size and a wrong model
	// are different facts a caller needs told apart.
	std::array<uint8_t, superslm::kIntegrityHashBytes> blob_model_hash{};
	if (!superslm_gpu::PeekGpuSeqBlobModelHash(blob, blob_size, &blob_model_hash)) {
		return SSLM_SEQUENCE_KV_BUFFER_MISMATCH;
	}
	if (blob_model_hash != model->content_hash) {
		return SSLM_RESTORE_MODEL_MISMATCH;
	}

	SslmGpuSequenceHandle* fresh = nullptr;
	const SslmGpuStatus create_status =
	    sslm_gpu_seq_create(ctx, model, static_cast<int64_t>(derived_context_cap), &fresh);
	if (create_status != SSLM_OK || !fresh) {
		return create_status;
	}
	// RestoreGpuSequenceState's own device round-trip (gpu_port.h/superslm_gpu.cpp,
	// D-SLM3080, unchanged): decodes the blob's header fields into `live_state` and the
	// K/V bytes into `host_kv_mirror` -- a malformed or size-mismatched blob returns
	// false, translated here the same way every other structural rejection in this file
	// is (SSLM_SEQUENCE_KV_BUFFER_MISMATCH, the create-time disposition this fresh
	// handle would have carried had its own sizing been the problem).
	// T-2114 (C1): hidden_codes_size = fresh->hidden_codes.size() -- fresh->live_state.hidden_codes
	// already aliases fresh->hidden_codes.data() (set in sslm_gpu_seq_create, above), so
	// RestoreGpuSequenceState writes the blob's residual-stream bytes directly into this
	// handle's own storage; no separate sync step is needed the way the scalar fields below
	// need one, because this is the same memory, not a mirror.
	//
	// T-2114 (S4, Claude/Poirot/50f3d5d-t2113-1p0-gpu-core-build-review.md): this call is now
	// inside a try -- it submits, fence-waits, and maps a readback entirely through
	// `SSLM_GPU_HR`, which throws on a real device/HR failure (the identical boundary B5 already
	// closed for `sslm_gpu_ready`/`RunLayerLoopGpuFinish`, cited in this same function's own
	// header comment above). Before this fix, an exception here escaped `sslm_gpu_seq_restore`
	// uncaught, crossing the status-returning API boundary design Sec4.1.1 establishes, and
	// stranded `fresh` with `ctx->live_handles`/`model->live_sequences` already incremented (by
	// the `sslm_gpu_seq_create` call above) with no release ever reached -- the context could
	// never afterwards be destroyed (`ContextHasLiveHandles`/`ModelHasLiveSequences`). The catch
	// below releases `fresh` exactly like the `ok == false` and `UploadResidentUavBufferSync`
	// failure paths immediately around it already do, and returns a status instead of unwinding.
	bool ok = false;
	try {
		ok = superslm_gpu::RestoreGpuSequenceState(blob, blob_size, &fresh->live_state,
		                                            fresh->hidden_codes.size(),
		                                            fresh->host_kv_mirror.data(),
		                                            fresh->host_kv_mirror.size());
	} catch (const std::exception&) {
		sslm_gpu_seq_release(ctx, fresh);
		return SSLM_DEVICE_LOST;
	}
	if (!ok) {
		sslm_gpu_seq_release(ctx, fresh);
		return SSLM_SEQUENCE_KV_BUFFER_MISMATCH;
	}
	// Upload the restored K/V bytes into this handle's own REAL resident buffer (design
	// Sec5.3: K/V is device-resident-only in the 1.0 API -- RestoreGpuSequenceState's own
	// host-memory round-trip is the substrate's pre-1.0 contract, reused verbatim for the
	// header/blob decode; the device residency itself is this call's own added step).
	try {
		fresh->kv_buf = UploadResidentUavBufferSync(ctx->device, fresh->host_kv_mirror.data(),
		                                             fresh->host_kv_mirror.size());
	} catch (const std::exception&) {
		sslm_gpu_seq_release(ctx, fresh);
		return SSLM_DEVICE_LOST;
	}
	fresh->kv_needs_resume_barrier = false;  // freshly (re)created in UNORDERED_ACCESS state
	fresh->hidden_scale = fresh->live_state.hidden_scale;
	fresh->layer_index = fresh->live_state.layer_index;
	fresh->kv_saturation_count = fresh->live_state.kv_saturation_count;
	fresh->context_length = fresh->live_state.context_length;
	*out_seq = fresh;
	return SSLM_OK;
}

SslmGpuStatus sslm_gpu_seq_reset(SslmGpuContext* ctx, SslmGpuSequenceHandle* seq) {
	if (!ctx || !seq || seq->ctx != ctx) {  // T-2114 (S2): ->destroyed read removed, see sequence
	                                         // handle's own field comment. T-2124 (D-SLM3446
	                                         // P1-4): `seq->ctx != ctx`.
		return SSLM_SEQUENCE_KV_BUFFER_MISMATCH;
	}
	const bool is_submitted = seq->state == superslm_gpu::SslmSequenceGpuState::Submitted;
	if (!superslm_gpu::CallProceedsOrBusy_SeqReset(is_submitted)) {
		return SSLM_BUSY;  // design Sec4.2's own state table, unchanged policy.
	}
	std::fill(seq->hidden_codes.begin(), seq->hidden_codes.end(), 0);
	seq->hidden_scale = superslm::CarriedScale{};
	seq->layer_index = 0;
	seq->kv_saturation_count = 0;
	seq->context_length = 0;
	seq->live_state.hidden_scale = seq->hidden_scale;
	seq->live_state.layer_index = 0;
	seq->live_state.kv_saturation_count = 0;
	seq->live_state.context_length = 0;
	std::fill(seq->host_kv_mirror.begin(), seq->host_kv_mirror.end(), 0);
	// G5-5 (T-2132, Brunel): mirrors `sslm_seq_reset`'s own extension (design Sec5,
	// src/sslm_abi.cpp) -- reset clears the DFA-walk-state back to the bound schema's own start
	// state (0) but PRESERVES the schema binding itself (`bound_schema_index` untouched); a
	// sequence with no schema bound stays at the unused sentinel, exactly the CPU ABI's own
	// disposition.
	if (seq->bound_schema_index >= 0) {
		seq->dfa_walk_state = 0u;
	}
	// G5-5 session 3 fix (T-2132, Brunel): mirrors `sslm_seq_reset`'s own identical
	// `ready_for_logits = false` clear (src/sslm_abi.cpp:1488) -- a reset sequence must not carry
	// a stale "finish without embedding" flag into whatever decode call comes next.
	seq->ready_for_logits = false;
	return SSLM_OK;
}

// T-2113 (B3): the bench-only accessor bodies declared in gpu_1p0_bench_bridge.h -- B5
// has now landed the real async path above; these accessors are kept only because
// tools/t2113_b3_sequence_smoke.cpp (a B3-era bench tool, not part of Curie's suite)
// still calls them, and retiring that tool is not part of this section's own scope.
// Every NEW caller (the B5 smoke tool, the red suite) drives the real API instead.
ID3D12Resource* SslmGpuSeqHandleKvBufferForBench(SslmGpuSequenceHandle* seq) {
	return seq ? seq->kv_buf.Get() : nullptr;
}
bool* SslmGpuSeqHandleKvResumeFlagForBench(SslmGpuSequenceHandle* seq) {
	return seq ? &seq->kv_needs_resume_barrier : nullptr;
}
int8_t* SslmGpuSeqHandleHiddenCodesForBench(SslmGpuSequenceHandle* seq) {
	return seq ? seq->hidden_codes.data() : nullptr;
}
superslm::CarriedScale* SslmGpuSeqHandleHiddenScaleForBench(SslmGpuSequenceHandle* seq) {
	return seq ? &seq->hidden_scale : nullptr;
}
uint32_t* SslmGpuSeqHandleLayerIndexForBench(SslmGpuSequenceHandle* seq) {
	return seq ? &seq->layer_index : nullptr;
}
uint64_t* SslmGpuSeqHandleKvSaturationForBench(SslmGpuSequenceHandle* seq) {
	return seq ? &seq->kv_saturation_count : nullptr;
}
int64_t* SslmGpuSeqHandleContextLengthForBench(SslmGpuSequenceHandle* seq) {
	return seq ? &seq->context_length : nullptr;
}
size_t SslmGpuSeqHandleHiddenSizeForBench(SslmGpuSequenceHandle* seq) {
	return seq ? seq->hidden_codes.size() : 0;
}
// T-2113 (N1, design Sec4.2/Sec21, Sec11 dim9's own product cell): exposes the handle's own
// context_cap so a bench driver can confirm sslm_gpu_seq_restore sized the FRESH handle to the
// blob's own recorded context_cap, not the model's -- the exact property N1's fix makes true and
// the property a red cell needs a real accessor, not a hardcoded assumption, to check.
int64_t SslmGpuSeqHandleContextCapForBench(SslmGpuSequenceHandle* seq) {
	return seq ? seq->context_cap : 0;
}

// -----------------------------------------------------------------------------------------
// G5-5 (T-2132, Brunel): gpu_1p0_g5_bridge.h's own body -- see that header for the full
// contract each function below implements. "No new arithmetic" (design Sec4) is enforced by
// construction here: every logits/mask/argmax step below is a direct call into
// superslm::RmsNormSite/LogitsSite/ApplyMaskAndArgmax/ArgmaxLowestIndexTieBreak
// (forward_sites.h) -- the SAME functions sslm_decode_step's own finishing block
// (src/sslm_abi.cpp) calls on its own hidden state, never a parallel reimplementation.
// -----------------------------------------------------------------------------------------

bool SslmGpuModelHasSchemasForG5Bridge(SslmGpuModelHandle* model) {
	return model != nullptr && model->schemas.Count() > 0;
}

int32_t SslmGpuSchemaLookupForG5Bridge(SslmGpuModelHandle* model, const char* name) {
	if (!model || !name) return -1;
	size_t index = 0;
	if (!model->schemas.ByName(name, &index)) return -1;
	return static_cast<int32_t>(index);
}

SslmGpuStatus SslmGpuSeqSetSchemaForG5Bridge(SslmGpuContext* ctx, SslmGpuSequenceHandle* seq,
                                              int32_t schema_index) {
	if (!ctx || !seq || !seq->model || seq->ctx != ctx) {
		return SSLM_SEQUENCE_KV_BUFFER_MISMATCH;
	}
	// Mirrors `sslm_seq_set_schema`'s own precondition (src/sslm_abi.cpp): valid ONLY when the
	// sequence's own DFA-walk-state is at its start -- kSslmGpuDfaWalkStateUnused (never bound)
	// or 0 (bound, unadvanced). SSLM_SEQUENCE_REJECTED is the closest existing per-sequence
	// structural-rejection disposition this bridge's own small surface has (gpu_1p0.h's own
	// SSLM_SEQUENCE_REJECTED comment; the CPU ABI's distinct SSLM_SCHEMA_BIND_REJECTED has no
	// counterpart on this file's own SslmGpuStatus enum, which this header deliberately does
	// not extend).
	if (seq->dfa_walk_state != kSslmGpuDfaWalkStateUnused && seq->dfa_walk_state != 0u) {
		return SSLM_SEQUENCE_REJECTED;
	}
	if (schema_index < 0) {
		seq->bound_schema_index = -1;
		seq->dfa_walk_state = kSslmGpuDfaWalkStateUnused;
		return SSLM_OK;
	}
	if (static_cast<size_t>(schema_index) >= seq->model->schemas.Count()) {
		return SSLM_SEQUENCE_REJECTED;
	}
	seq->bound_schema_index = schema_index;
	seq->dfa_walk_state = 0u;
	return SSLM_OK;
}

uint32_t SslmGpuSeqWalkStateForG5Bridge(SslmGpuSequenceHandle* seq) {
	return seq ? seq->dfa_walk_state : kSslmGpuDfaWalkStateUnused;
}

SslmGpuStatus SslmGpuSeqFinishTokenForG5Bridge(SslmGpuContext* ctx, SslmGpuSequenceHandle* seq,
                                                int32_t* out_token) {
	if (!out_token) return SSLM_SEQUENCE_KV_BUFFER_MISMATCH;
	*out_token = -1;
	if (!ctx || !seq || !seq->model || seq->ctx != ctx) {
		return SSLM_SEQUENCE_KV_BUFFER_MISMATCH;
	}
	if (seq->state != superslm_gpu::SslmSequenceGpuState::Idle) {
		return SSLM_BUSY;  // an in-flight Submitted call must be drained (sslm_gpu_ready) first.
	}
	SslmGpuModelHandle* model = seq->model;
	if (seq->layer_index != model->num_hidden_layers) {
		// This token's own layer loop has not reached full depth yet -- the caller's own
		// dispatch-budget loop must submit more sslm_decode_step_gpu calls first. No dedicated
		// enumerator exists for "not ready yet" on this bridge's own small surface; same
		// "no more specific status" disposition the whole production file already uses.
		return SSLM_SEQUENCE_REJECTED;
	}

	// final_norm + logits -- the EXACT two calls sslm_decode_step's own finishing block makes
	// (src/sslm_abi.cpp), on this sequence's own GPU-derived, already-read-back hidden_codes
	// (`seq->hidden_codes` -- `live_state.hidden_codes` aliases it directly, so sslm_gpu_ready's
	// own readback already landed here, no separate sync needed).
	std::vector<int8_t> final_codes(model->hidden_size);
	superslm::CarriedScale final_scale{};
	superslm::SslmForwardStatus fst = superslm::RmsNormSite(
	    seq->hidden_codes.data(), model->final_norm_gain.data(), model->hidden_size,
	    seq->hidden_scale, model->final_norm_site_constant, final_codes.data(), &final_scale,
	    "final_norm", /*token_index=*/0, /*trace_hook_state=*/nullptr,
	    /*external_wide_scratch=*/nullptr);
	if (fst != superslm::SslmForwardStatus::Ok) return MapDecodedStatusToGpuStatus(fst);

	std::vector<int64_t> wide_logits(static_cast<size_t>(model->vocab_size));
	std::vector<int32_t> logit_row(static_cast<size_t>(model->vocab_size));
	fst = superslm::LogitsSite(final_codes.data(), model->hidden_size, model->head_weights.data(),
	                            model->vocab_size, wide_logits.data(), logit_row.data());
	if (fst != superslm::SslmForwardStatus::Ok) return MapDecodedStatusToGpuStatus(fst);

	// G5-5: masking applies to int32 logits BEFORE argmax, indexed by this sequence's own
	// DFA-walk-state -- SAME code (superslm::ApplyMaskAndArgmax) as the CPU path's masked-argmax
	// step; no schema bound is byte-for-byte the pre-G5 path (plain ArgmaxLowestIndexTieBreak),
	// exactly the base-level GPU parity this bridge extends.
	int32_t produced = 0;
	if (seq->bound_schema_index >= 0) {
		const superslm::SchemaEntry* entry =
		    model->schemas.ByIndex(static_cast<size_t>(seq->bound_schema_index));
		if (!entry) return SSLM_SEQUENCE_REJECTED;
		const uint8_t* page = entry->mask_pages +
		                       static_cast<size_t>(seq->dfa_walk_state) * model->schemas.MaskPageBytes();
		superslm::ApplyMaskAndArgmax(logit_row.data(), page, model->vocab_size, &produced);
		uint32_t next_state = seq->dfa_walk_state;
		model->schemas.Transition(*entry, seq->dfa_walk_state, static_cast<uint32_t>(produced),
		                           &next_state);
		seq->dfa_walk_state = next_state;
	} else {
		produced = superslm::ArgmaxLowestIndexTieBreak(logit_row.data(),
		                                                static_cast<size_t>(model->vocab_size));
	}
	*out_token = produced;
	// Resting convention -- matches CPU's identical `seq->state.layer_index = 0` reset
	// (sslm_decode_stepImpl, src/sslm_abi.cpp) so the next embed starts a fresh token boundary.
	seq->layer_index = 0;
	seq->live_state.layer_index = 0;
	return SSLM_OK;
}

namespace {
// G5-5 session 3 fix (T-2132, Brunel): the SAME "drive one token's own layer loop to full depth,
// possibly across several sslm_decode_step_gpu calls" loop every bridge entry point below needs
// -- extracted once here (internal linkage) so `SslmGpuSeqPrefillPromptForG5Bridge` and
// `SslmGpuSeqDecodeStepForG5Bridge` (below) share one implementation rather than each hand-rolling
// its own copy, which is exactly the shape of duplication that let the ready_for_logits gap slip
// past this ticket's own first parity harness (session 2/3's own finding: the harness's `tools/
// t2132_g5_gpu_parity_gpu.cpp` had its own private copy of this loop, and its own private, buggy
// prompt-prefill composition around it).
bool DriveGpuSeqToFullDepthForG5Bridge(SslmGpuContext* ctx, SslmGpuSequenceHandle* seq,
                                        uint32_t dispatch_budget) {
	SslmGpuModelHandle* model = seq->model;
	uint32_t guard = 0;
	while (seq->layer_index < model->num_hidden_layers) {
		if (sslm_decode_step_gpu(ctx, seq, /*adapter_or_null=*/nullptr, dispatch_budget) != SSLM_OK) {
			return false;
		}
		int32_t ready = 0;
		SslmGpuStatus drained = SSLM_OK;
		if (sslm_gpu_ready(ctx, seq, /*block=*/1, &ready, &drained) != SSLM_OK) return false;
		if (drained != SSLM_OK) return false;
		if (++guard > 10000) return false;  // runaway-loop guard, never expected in practice.
	}
	return true;
}
}  // namespace

namespace {

// T-2169 (Rung 3/4, design Claude/Vitruvius/t2169-gpu-batched-prefill-design-2026-08-18.md Sec5/
// Sec5.1/Sec8, D-SLM3590/3595/3611-3613/3619-3622/3631-3634): the cause a chunk's own admission
// count is attributable to, resolved once per call by the pre-scan below -- reused by both G5
// bridge functions' own cause-routed return (Sec5's table).
enum class ChunkAdmissionCause { kNone, kDfa, kEmbed, kPositionCap };

// Host-side sequential pre-scan (design Sec5), run to completion over the intended chunk before
// any dispatch is recorded. `dfa_entry` is null for the prompt twin (no reachability check --
// `dfa_admit_count` defaults to `chunk_len`, design Sec5's own "the two GPU-schema-only terms
// default to chunk_len for the prompt twin"); non-null for the schema twin. Produces `admit_count`
// (the min of the applicable terms), the `cause` that produced it (DFA > EMBED > POSITION_CAP,
// matching the shipped per-token loop's own check order so a tie resolves identically to what one
// more per-token round-trip would find), every admitted token's own pre-packed embedding bytes
// (the `[0, SeqScaleOff(hidden_size)+16)` layout `SubmitChunkToFullDepthForG5Bridge` expects,
// `superslm_gpu::T2169SeqEmbeddingBlockBytes`), and -- schema twin only -- the DFA walk state
// after each admitted transition, so the caller can commit the state after exactly however many
// tokens actually end up committed (`admit_count`, or a device-derived shorter count, D-SLM3622).
struct ChunkPreScanResult {
	uint32_t admit_count = 0;
	ChunkAdmissionCause cause = ChunkAdmissionCause::kNone;
	SslmGpuStatus embed_reject_status = SSLM_OK;    // meaningful iff cause == kEmbed.
	std::vector<uint8_t> chunk_embedding_bytes;     // admit_count * block_bytes, packed.
	std::vector<uint32_t> dfa_states_after;         // schema twin only; dfa_states_after[t] is the
	                                                 // walk state after admitting token t, valid
	                                                 // for t in [0, dfa_admit_count).
};

ChunkPreScanResult RunChunkAdmissionPreScan(SslmGpuModelHandle* model, int64_t chunk_open_ctxlen,
                                             int64_t context_cap, const int32_t* tokens,
                                             int32_t chunk_len, const superslm::SchemaEntry* dfa_entry,
                                             uint32_t dfa_walk_state_start) {
	ChunkPreScanResult r;
	const uint32_t n = static_cast<uint32_t>(chunk_len);

	// D-SLM3613: position_admit_count, O(1) -- position advances by exactly 1 per token, so no
	// per-token loop is needed (design Sec5).
	int64_t cap_room = context_cap - chunk_open_ctxlen;
	if (cap_room < 0) cap_room = 0;
	const uint32_t position_admit_count = static_cast<uint32_t>(std::min<int64_t>(n, cap_room));

	// D-SLM3590: dfa_admit_count, run to completion over the intended chunk -- schema twin only.
	// Tracks the walk state after each admitted transition.
	uint32_t dfa_admit_count = n;
	if (dfa_entry) {
		r.dfa_states_after.reserve(n);
		uint32_t state = dfa_walk_state_start;
		for (uint32_t t = 0; t < n; ++t) {
			uint32_t next_state = 0;
			if (!model->schemas.Transition(*dfa_entry, state, static_cast<uint32_t>(tokens[t]),
			                                &next_state)) {
				dfa_admit_count = t;
				break;
			}
			state = next_state;
			r.dfa_states_after.push_back(state);
		}
	}

	// D-SLM3621: embed_admit_count, run to completion over the intended chunk -- the SAME pass
	// that produces the record-time embedding bytes 2b needs, for every candidate index.
	uint32_t embed_admit_count = n;
	SslmGpuStatus embed_reject_status = SSLM_OK;
	const uint32_t block_bytes = superslm_gpu::T2169SeqEmbeddingBlockBytes(model->hidden_size);
	const uint32_t scale_offset = block_bytes - 16u;
	std::vector<uint8_t> embed_bytes_cache;  // block_bytes per validated candidate index
	embed_bytes_cache.reserve(static_cast<size_t>(block_bytes) * n);
	for (uint32_t t = 0; t < n; ++t) {
		const int32_t token_id = tokens[t];
		if (token_id < 0 || token_id >= model->vocab_size) {
			embed_admit_count = t;
			embed_reject_status = SSLM_TOKEN_ID_OUT_OF_RANGE;
			break;
		}
		std::vector<int8_t> embed_codes(model->hidden_size);
		superslm::CarriedScale embed_scale{};
		const superslm::SslmForwardStatus est = superslm::EmbedEntry(
		    token_id, model->vocab_size, model->embed_weights.data(),
		    static_cast<size_t>(model->hidden_size), model->embed_site_constant, embed_codes.data(),
		    &embed_scale);
		if (est != superslm::SslmForwardStatus::Ok) {
			// EmbedEntry's own only rejection is the identical token_id bounds check already
			// performed above -- unreachable in practice, handled the same way
			// sslm_gpu_seq_embed_token handles it (this file, above).
			embed_admit_count = t;
			embed_reject_status = SSLM_TOKEN_ID_OUT_OF_RANGE;
			break;
		}
		const size_t base = embed_bytes_cache.size();
		embed_bytes_cache.resize(base + block_bytes, 0);
		for (uint32_t i = 0; i < model->hidden_size; ++i) {
			const int32_t v = static_cast<int32_t>(embed_codes[i]);
			std::memcpy(embed_bytes_cache.data() + base + static_cast<size_t>(i) * 4u, &v, 4);
		}
		std::memcpy(embed_bytes_cache.data() + base + scale_offset, &embed_scale.m, 8);
		std::memcpy(embed_bytes_cache.data() + base + scale_offset + 8, &embed_scale.e, 8);
	}

	// D-SLM3620: admit_count = min of the applicable terms. `admit_count == chunk_len` is checked
	// FIRST and is unconditionally NONE (every term trivially equals chunk_len when nothing is
	// rejected, so the priority chain below is only meaningful once an actual rejection -- a term
	// strictly less than chunk_len -- exists); otherwise cause is resolved by the shipped
	// per-token loop's own check order (DFA before embed before the submit-time position-cap
	// guard).
	r.admit_count = std::min({dfa_admit_count, embed_admit_count, position_admit_count});
	if (r.admit_count == n) {
		r.cause = ChunkAdmissionCause::kNone;
	} else if (dfa_entry && dfa_admit_count == r.admit_count) {
		r.cause = ChunkAdmissionCause::kDfa;
	} else if (embed_admit_count == r.admit_count) {
		r.cause = ChunkAdmissionCause::kEmbed;
		r.embed_reject_status = embed_reject_status;
	} else {
		r.cause = ChunkAdmissionCause::kPositionCap;
	}

	r.chunk_embedding_bytes.assign(
	    embed_bytes_cache.begin(),
	    embed_bytes_cache.begin() + static_cast<size_t>(r.admit_count) * block_bytes);
	return r;
}

// T-2169 (Rung 3/4, design Sec5 step 5/D-SLM3614/D-SLM3622): dispatches exactly `admit_count`
// pre-scanned, pre-embedded tokens through the chunk-submission primitive
// (`superslm_gpu::SubmitChunkToFullDepthForG5Bridge`, internally TDR-safe-split) and drains the
// final (sub-)chunk's own async token to completion, copying the decoded result back into this
// handle's own canonical fields -- the same copy-back `sslm_gpu_ready` performs, since this call
// bypasses that function entirely (this bridge owns the whole Submit/Finish pair for a chunk
// call). `*out_derived_count` is the device's own actually-committed count, derived from
// `seq->context_length`'s post-chunk delta against `chunk_open_ctxlen` -- never assumed to equal
// `admit_count` (D-SLM3614): a device-domain rejection the pre-scan could not see (the sticky-tag
// mechanism, P3(b)/S4(b)) or a mid-recording infrastructural fault (the tenth failure origin,
// D-SLM3634) both show up here as a short count, which the caller checks against `admit_count` to
// decide whether the device-computed override (D-SLM3622) applies -- this function itself reports
// no status of its own beyond the derived count.
void SubmitAdmittedChunkForG5Bridge(SslmGpuModelHandle* model, SslmGpuSequenceHandle* seq,
                                     const uint8_t* chunk_embedding_bytes, uint32_t admit_count,
                                     int64_t chunk_open_ctxlen, int64_t* out_derived_count) {
	*out_derived_count = 0;
	if (admit_count == 0) return;
	// Mirrors `sslm_gpu_seq_embed_token`'s own unconditional `layer_index = 0` reset for a VALID
	// token, which the shipped per-token loop always performs for token 0 before any guard or
	// dispatch runs -- this batched path bypasses `sslm_gpu_seq_embed_token` entirely (design
	// Sec5.1's own "bypassing that wrapper's own state-transition/host-mirror side effects"), so
	// without this, `PrepareGpuLayerLoopChunkOpenState`'s own `SequenceAlreadyComplete` guard
	// (`seq.layer_index >= num_hidden_layers`) would reject every call after the first: a
	// committed token's own last-layer commit leaves the DEVICE's `layer_index` at
	// `num_hidden_layers` (D-SLM3649's own finding), and nothing else on this path ever resets
	// the HOST mirror between separate calls. Only fires when a token WILL actually be dispatched
	// (admit_count > 0), matching embed_token's own reset firing only for a validating token.
	seq->layer_index = 0;
	seq->live_state.layer_index = 0;
	static const superslm::SslmTensorManifest kEmptyManifest;
	// T-2185 remedy N1 (Brunel fix round 2, D-SLM3674): the T-2184 remedy (S1) opened this window
	// AFTER `SubmitChunkToFullDepthForG5Bridge` returned, which is also after that call's own
	// internal loop has already submitted-and-synchronously-finished every non-final sub-chunk
	// (`SubmitChunkToFullDepthForG5Bridge`'s own header comment, superslm_gpu.cpp: "Every
	// non-final sub-chunk is finished SYNCHRONOUSLY, here" -- `RunLayerLoopGpuFinish(...,
	// block=1)` inside that loop). So the window covered only the FINAL sub-chunk's own fence
	// wait -- under 3.2% of the call's wall time at a 128-token chunk (T-2185 N1's own derivation)
	// -- and every earlier sub-chunk's fence wait ran with `model->submitted_sequences == 0`,
	// exactly the reachable consequence the T-2184 review named, still reachable across the
	// other 96.8% of the call. The window now opens HERE, before the first sub-chunk is ever
	// submitted, so every sub-chunk's own fence wait -- not only the final one -- falls inside it;
	// `seq->in_flight` is assigned once the call returns the final sub-chunk's own inflight token
	// (the only one handed back to this caller unfenced; every earlier one already completed
	// inside the call).
	//
	// T-2186 remedy P1 (Brunel fix round 3, D-SLM3682, confirmation
	// Claude/Poirot/8642652-t2186-t2169-fix2-confirmation.md): opening and closing this window as
	// two separate manual statements -- open here, close unconditionally below -- was NOT
	// symmetric with the shipped per-token path in the property that matters: the call in between
	// can throw. `SubmitChunkToFullDepthForG5Bridge`'s own tail
	// (`SubmitOneSubChunkToFullDepthForG5Bridge`, superslm_gpu.cpp:2591/2597/2598 --
	// `dev.list->Close()`, `dev.queue->Signal()`, `new GpuLayerLoopInFlight()`) sits OUTSIDE that
	// function's own try/catch, so a failed `Close()`/`Signal()` (`SSLM_GPU_HR` throws
	// `std::runtime_error` -- precisely the device-removed channel design Sec9 promises is
	// deliverable) or a `std::bad_alloc` from the `new` can escape this call with no catch
	// anywhere between here and the ABI boundary. A manual "close below" statement is never
	// reached on that path: `seq->state` stays `Submitted` and `model->submitted_sequences` stays
	// incremented forever, and `sslm_gpu_model_unmap` (`SslmGpuModelHandle::submitted_sequences`'
	// own check, above) returns `SSLM_BUSY` on every later call, permanently -- the model can
	// never be unmapped. The shipped per-token twin (`SubmitOneSequenceDecode`, this file,
	// `:1379-1381`'s own comment: "Rejected before submission (a guard, or an exception) -- seq/
	// host state untouched, still Idle") avoids this class entirely by opening its window only
	// AFTER submission returns; this batched path cannot do the same (design Sec9 wants the
	// window held across the whole chunk, submitted or not, not only after it succeeds), so it is
	// made safe by construction instead: an RAII guard opens the window in its constructor and
	// closes it in its destructor, which C++ runs on every exit from this function -- normal
	// return AND exception unwind alike -- so there is no longer any path between open and close
	// where a throw can leave the window stuck open. This replaces the manual open above and the
	// manual unconditional close that used to sit after the call, below; the close is no longer a
	// second statement that can be skipped, it is a property of the guard's lifetime.
	struct SubmittedWindowScopeGuard {
		SslmGpuSequenceHandle* seq;
		SslmGpuModelHandle* model;
		SubmittedWindowScopeGuard(SslmGpuSequenceHandle* s, SslmGpuModelHandle* m) : seq(s), model(m) {
			seq->state = superslm_gpu::SslmSequenceGpuState::Submitted;
			model->submitted_sequences += 1;
		}
		~SubmittedWindowScopeGuard() {
			seq->state = superslm_gpu::SslmSequenceGpuState::Idle;
			if (model->submitted_sequences > 0) model->submitted_sequences -= 1;
		}
	} submitted_window_guard(seq, model);
	superslm_gpu::GpuLayerLoopInFlight* inflight = nullptr;
	const superslm::SslmForwardStatus submit_status = superslm_gpu::SubmitChunkToFullDepthForG5Bridge(
	    seq->live_state, /*layers=*/nullptr, model->num_hidden_layers, model->hidden_size,
	    model->head_dim, model->num_key_value_heads, model->intermediate_size, seq->context_cap,
	    kEmptyManifest, seq->host_kv_mirror.data(), seq->host_kv_mirror.size(), chunk_embedding_bytes,
	    admit_count, seq->kv_buf.Get(), &seq->kv_needs_resume_barrier, model->weights_buf.Get(),
	    model->rope_cos_buf.Get(), model->rope_sin_buf.Get(), model->has_rope_tables,
	    model->rope_cos_elem_count, model->rope_sin_elem_count, /*adapter_bridge=*/nullptr, &inflight);
	if (submit_status == superslm::SslmForwardStatus::Ok && inflight) {
		// `SubmitChunkToFullDepthForG5Bridge` returns the FINAL (sub-)chunk's own inflight token
		// genuinely unfenced (its own header comment: "the caller's own async contract... only the
		// FINAL (sub-)chunk's own inflight token is handed back unfinished"), so the fence for that
		// last sub-chunk has not yet signaled at this point -- a concurrent caller on another
		// thread (e.g. `sslm_gpu_model_unmap`, `sslm_gpu_seq_save`) still genuinely observes the
		// window while `RunLayerLoopGpuFinish` below blocks on it.
		seq->in_flight = inflight;
		int32_t ready = 0;
		superslm_gpu::RunLayerLoopGpuFinish(inflight, seq->live_state, seq->host_kv_mirror.data(),
		                                      /*block=*/1, &ready);
		seq->in_flight = nullptr;
	}
	// `submitted_window_guard`'s destructor closes the window here, unconditionally, on this
	// normal-return path exactly as it would on an exception unwind -- `sslm_gpu_ready`'s own
	// "collapse Submitted -> Idle in one call" symmetry, now a property of the guard's lifetime
	// rather than a second statement a throw could skip. No adapter bookkeeping to mirror: this
	// path always submits with `adapter_bridge = nullptr` (no LoRA on the chunk path), so
	// `seq->in_flight_adapter` is never set here and needs no clearing, matching every other
	// adapter-less caller.
	//
	// Copy the (possibly partially-advanced, per the sub-chunk split's own synchronous-finish
	// discipline, D-SLM3596/3649) live_state back into this handle's own canonical fields.
	seq->hidden_scale = seq->live_state.hidden_scale;
	seq->layer_index = seq->live_state.layer_index;
	seq->kv_saturation_count = seq->live_state.kv_saturation_count;
	seq->context_length = seq->live_state.context_length;
	*out_derived_count = seq->context_length - chunk_open_ctxlen;
	if (*out_derived_count < 0) *out_derived_count = 0;  // defensive; not expected by construction.
}

// D-SLM3619: reproduces the shipped per-token loop's own incidental side effect at the
// position-cap boundary -- token[admit_count] is a VALID id (position_admit_count, never embed
// validity, is what capped it, since cause == kPositionCap implies embed_admit_count >
// admit_count), so the shipped loop's own unconditional `sslm_gpu_seq_embed_token(tokens[
// admit_count])` call ALWAYS succeeds before the device-side guard discovers the position-cap
// violation, leaving `hidden_codes`/`layer_index` at that token's own raw (never-dispatched)
// embedding -- exactly the state `SslmGpuSeqFinishTokenForG5Bridge`'s own "not full depth yet"
// check reads to decide whether a subsequent ready_for_logits-shortcut decode step is REJECTED.
// No dispatch is recorded for this token; this is a host-only write, matching
// `sslm_gpu_seq_embed_token`'s own exact effect. Called only when cause == kPositionCap and the
// device did not additionally override the outcome (the device-computed fallback is a distinct,
// untested-by-this-suite mechanism, design Sec6/Sec9's own named gap).
void ReproducePositionCapBoundaryEmbedSideEffect(SslmGpuModelHandle* model,
                                                  SslmGpuSequenceHandle* seq, const int32_t* tokens,
                                                  uint32_t admit_count) {
	const int32_t next_token = tokens[admit_count];
	std::vector<int8_t> embed_codes(model->hidden_size);
	superslm::CarriedScale embed_scale{};
	const superslm::SslmForwardStatus est = superslm::EmbedEntry(
	    next_token, model->vocab_size, model->embed_weights.data(),
	    static_cast<size_t>(model->hidden_size), model->embed_site_constant, embed_codes.data(),
	    &embed_scale);
	if (est != superslm::SslmForwardStatus::Ok) return;  // unreachable per this function's own
	                                                        // header comment; defensive only.
	std::copy(embed_codes.begin(), embed_codes.end(), seq->hidden_codes.begin());
	seq->hidden_scale = embed_scale;
	seq->layer_index = 0;
}

}  // namespace

// T-2184 remedy C1 (Brunel fix round 1, D-SLM3662): a bench-only entry point that reproduces the
// GENUINELY shipped (pre-T-2169, `e35edc1`) per-token prefill composition -- `sslm_gpu_seq_embed_
// token` followed by `DriveGpuSeqToFullDepthForG5Bridge`, one submit-and-fence round-trip per
// LAYER (28 for Qwen2.5-1.5B, `dispatch_budget=24`), issued once per token in a plain host loop.
// Body copied verbatim from `e35edc1:src/gpu/gpu_1p0.cpp`'s own
// `SslmGpuSeqPrefillPromptForG5Bridge` (including its own per-iteration busy check, `if (seq->
// state != Idle) return SSLM_BUSY;` inside the loop, and its own `if (count > 0) seq->ready_for_
// logits = true;` tail) -- not reconstructed from a description of it.
//
// WHY THIS EXISTS. The T-2184 review's own C1 finding: after Rungs 3/4 rewrote `SslmGpuSeqPrefill
// PromptForG5Bridge`'s body to the pre-scan-then-batch composition, `tools/t2180_rung6_tokps.cpp`'s
// own "shipped-per-token" bench arm -- N separate single-token calls into that SAME public bridge
// function -- silently became the batched primitive at `chunk_len=1`, not the genuinely shipped
// 28-round-trip-per-token path. `DriveGpuSeqToFullDepthForG5Bridge` (this file, above) still exists
// and is still exactly the shipped shape, but its only remaining production caller is
// `SslmGpuSeqDecodeStepForG5Bridge` -- neither prefill twin can reach it any more. This function is
// the one new caller that can, for bench purposes only.
//
// NOT part of the shipped 1.0 API surface (matching `gpu_1p0_bench_bridge.h`'s own "not part of
// the shipped GPU API surface, and not installed alongside it" convention) -- declared in that
// header, `tools/t2180_rung6_tokps.cpp`'s own intended includer alongside its existing bench
// accessors.
//
// T-2185 remedy N6/observation (Brunel fix round 2, D-SLM3677): this definition, and its own
// declaration in `gpu_1p0_bench_bridge.h`, are now gated behind
// `SUPERSLM_ENABLE_GPU_BENCH_PRE_BATCHING` -- previously defined unconditionally at file scope,
// so the symbol linked into every binary that compiled this translation unit, not only the one
// bench tool that calls it. Gated per this codebase's own established injection-seam convention
// (`SUPERSLM_O11_ALLOC_INJECTION`, `SUPERSLM_T2169_CHUNK_RECORDING_FAULT_INJECTION`, gpu_port.h):
// `build.bat`'s own `tools\t2180_rung6_tokps.cpp` compile line is the only place that defines the
// macro; every other target -- the red-suite cells, every other tool -- links `gpu_1p0.cpp`
// without it, so this symbol is absent from those binaries entirely (verified: `dumpbin
// /symbols` against a build of this file without the macro defined shows no
// `SslmGpuSeqPrefillPromptPreBatchingBenchOnly` entry).
#ifdef SUPERSLM_ENABLE_GPU_BENCH_PRE_BATCHING
SslmGpuStatus SslmGpuSeqPrefillPromptPreBatchingBenchOnly(SslmGpuContext* ctx,
                                                            SslmGpuSequenceHandle* seq,
                                                            const int32_t* tokens, int32_t count,
                                                            uint32_t dispatch_budget) {
	if (!ctx || !seq || !seq->model || seq->ctx != ctx || (count > 0 && !tokens) || count < 0 ||
	    dispatch_budget < 1) {
		return SSLM_SEQUENCE_KV_BUFFER_MISMATCH;
	}
	for (int32_t i = 0; i < count; ++i) {
		if (seq->state != superslm_gpu::SslmSequenceGpuState::Idle) return SSLM_BUSY;
		const SslmGpuStatus st = sslm_gpu_seq_embed_token(ctx, seq, tokens[i]);
		if (st != SSLM_OK) return st;
		if (!DriveGpuSeqToFullDepthForG5Bridge(ctx, seq, dispatch_budget)) return SSLM_DEVICE_LOST;
	}
	if (count > 0) seq->ready_for_logits = true;
	return SSLM_OK;
}
#endif  // SUPERSLM_ENABLE_GPU_BENCH_PRE_BATCHING

// G5-5 session 3 fix (T-2132, Brunel, Claude/Brunel/t2132-g5-build-2026-08-16.md session 3): the
// GPU-1.0 twin of `sslm_prefill(..., SSLM_SPAN_PROMPT, ...)` -- embeds and drives EVERY token in
// `tokens` (including the last) to full depth, exactly `PrefillWholeTokensImpl`'s own
// SSLM_SPAN_PROMPT branch (no walk-state touch, no masking -- design Sec5: "never advances the
// DFA walk, whatever schema is bound"). On success, sets `seq->ready_for_logits = true`,
// mirroring `sslm_prefill`'s own unconditional set on any successful prefill
// (src/sslm_abi.cpp:1607) -- this is the fix: a caller that then calls
// `SslmGpuSeqDecodeStepForG5Bridge` (below) for its first post-prompt decode step gets the SAME
// "reuse the already-computed final residual, do not re-embed" shortcut the CPU ABI already gives
// every real caller, closing the exact duplicate-KV-commit gap session 3 diagnosed in this
// bridge's own prior caller (`tools/t2132_g5_gpu_parity_gpu.cpp`'s hand-rolled `PrefillPrompt`).
SslmGpuStatus SslmGpuSeqPrefillPromptForG5Bridge(SslmGpuContext* ctx, SslmGpuSequenceHandle* seq,
                                                  const int32_t* tokens, int32_t count,
                                                  uint32_t dispatch_budget) {
	if (!ctx || !seq || !seq->model || seq->ctx != ctx || (count > 0 && !tokens) || count < 0 ||
	    dispatch_budget < 1) {
		return SSLM_SEQUENCE_KV_BUFFER_MISMATCH;
	}
	if (count == 0) return SSLM_OK;  // matches the shipped per-token loop's own vacuous-loop exit
	                                  // (the busy check below is never reached for count == 0
	                                  // either, on either path).
	if (seq->state != superslm_gpu::SslmSequenceGpuState::Idle) return SSLM_BUSY;

	// T-2169 (Rung 4, design Sec5/Sec8): the pre-scan-then-batch composition -- host-side
	// sequential admission, then ONE call into the chunk-submission primitive for exactly
	// `admit_count` tokens, replacing the per-token DriveGpuSeqToFullDepthForG5Bridge loop above.
	// No DFA pre-scan (the prompt twin never advances or checks the DFA walk, design Sec5).
	SslmGpuModelHandle* model = seq->model;
	seq->live_state.hidden_scale = seq->hidden_scale;
	seq->live_state.layer_index = seq->layer_index;
	seq->live_state.kv_saturation_count = seq->kv_saturation_count;
	seq->live_state.context_length = seq->context_length;
	const int64_t chunk_open_ctxlen = seq->context_length;

	const ChunkPreScanResult scan = RunChunkAdmissionPreScan(
	    model, chunk_open_ctxlen, seq->context_cap, tokens, count, /*dfa_entry=*/nullptr,
	    /*dfa_walk_state_start=*/0);

	int64_t derived_count = 0;
	SubmitAdmittedChunkForG5Bridge(model, seq, scan.chunk_embedding_bytes.data(), scan.admit_count,
	                                chunk_open_ctxlen, &derived_count);

	// D-SLM3622 (design Sec5 step 5): the device-computed fallback -- a rejection the pre-scan
	// could not see, discovered only via the post-chunk readback -- overrides whatever the
	// host-computable cause predicted, regardless of cause. ready_for_logits stays untouched.
	if (derived_count < static_cast<int64_t>(scan.admit_count)) {
		return SSLM_DEVICE_LOST;
	}

	switch (scan.cause) {
		case ChunkAdmissionCause::kEmbed:
			// ready_for_logits left untouched -- P2's own disposition (design Sec5's cause table).
			return scan.embed_reject_status;
		case ChunkAdmissionCause::kPositionCap:
			// D-SLM3619: refuses the whole call with the shipped path's exact observables --
			// ready_for_logits left untouched, and the next-rejected token's own incidental
			// host-side embed side effect is reproduced (never a dispatch for it).
			ReproducePositionCapBoundaryEmbedSideEffect(model, seq, tokens, scan.admit_count);
			return SSLM_DEVICE_LOST;
		case ChunkAdmissionCause::kDfa:
			// Unreachable for the prompt twin (no DFA pre-scan is run for it) -- defensive only.
			return SSLM_SEQUENCE_REJECTED;
		case ChunkAdmissionCause::kNone:
		default:
			seq->ready_for_logits = true;  // count > 0 already established above -- P4's own row.
			return SSLM_OK;
	}
}

// G5-5 session 3 fix (T-2132, Brunel): the recommended one-call-per-step entry point -- the GPU-
// 1.0 twin of `sslm_decode_step`'s own composition (embed-if-needed, layer-loop-to-depth, finish),
// including its `ready_for_logits` shortcut (src/sslm_abi.cpp:1708-1715) verbatim: when a prior
// `SslmGpuSeqPrefillPromptForG5Bridge`/`SslmGpuSeqPrefillSchemaContentForG5Bridge` call left this
// flag set, `token_to_embed_if_needed` is IGNORED and this call jumps straight to finishing the
// already-computed residual -- no embed, no layer loop, matching the CPU ABI's own "this is the
// ONE call that costs no layer work" comment exactly. Otherwise embeds `token_to_embed_if_needed`
// and drives it to full depth first. Either way, finishes via `SslmGpuSeqFinishTokenForG5Bridge`
// (masking if a schema is bound, walk-state advance, plain argmax otherwise). This is the shape a
// future bridge consumer should reach for FIRST -- the granular
// embed/`sslm_decode_step_gpu`/`sslm_gpu_ready`/finish primitives stay available (this bridge/
// `gpu_1p0.h` do not remove them) for a caller that genuinely needs to interleave other GPU work
// between layers, but composing them by hand is exactly what re-trips this session's own bug.
SslmGpuStatus SslmGpuSeqDecodeStepForG5Bridge(SslmGpuContext* ctx, SslmGpuSequenceHandle* seq,
                                               int32_t token_to_embed_if_needed,
                                               uint32_t dispatch_budget, int32_t* out_token) {
	if (!out_token) return SSLM_SEQUENCE_KV_BUFFER_MISMATCH;
	*out_token = -1;
	if (!ctx || !seq || !seq->model || seq->ctx != ctx || dispatch_budget < 1) {
		return SSLM_SEQUENCE_KV_BUFFER_MISMATCH;
	}
	if (seq->ready_for_logits) {
		// This sequence's hidden_codes already hold a fully-computed final hidden state (from a
		// prior prefill call) -- no embed, no layer loop this call; jump straight to finishing,
		// exactly sslm_decode_step's own ready_for_logits branch (src/sslm_abi.cpp:1708-1715).
		seq->ready_for_logits = false;
	} else {
		if (seq->state != superslm_gpu::SslmSequenceGpuState::Idle) return SSLM_BUSY;
		const SslmGpuStatus st = sslm_gpu_seq_embed_token(ctx, seq, token_to_embed_if_needed);
		if (st != SSLM_OK) return st;
		if (!DriveGpuSeqToFullDepthForG5Bridge(ctx, seq, dispatch_budget)) return SSLM_DEVICE_LOST;
	}
	return SslmGpuSeqFinishTokenForG5Bridge(ctx, seq, out_token);
}

SslmGpuStatus SslmGpuSeqPrefillSchemaContentForG5Bridge(SslmGpuContext* ctx,
                                                          SslmGpuSequenceHandle* seq,
                                                          const int32_t* tokens, int32_t count,
                                                          uint32_t dispatch_budget_per_token,
                                                          int32_t* consumed) {
	if (!consumed) return SSLM_SEQUENCE_KV_BUFFER_MISMATCH;
	*consumed = 0;
	if (!ctx || !seq || !seq->model || seq->ctx != ctx || (count > 0 && !tokens) || count < 0 ||
	    dispatch_budget_per_token < 1) {
		return SSLM_SEQUENCE_KV_BUFFER_MISMATCH;
	}
	if (seq->bound_schema_index < 0) {
		return SSLM_SEQUENCE_REJECTED;  // the GPU twin of SSLM_SCHEMA_SPAN_UNBOUND.
	}
	SslmGpuModelHandle* model = seq->model;
	const superslm::SchemaEntry* entry =
	    model->schemas.ByIndex(static_cast<size_t>(seq->bound_schema_index));
	if (!entry) return SSLM_SEQUENCE_REJECTED;
	if (count == 0) return SSLM_OK;  // matches the shipped per-token loop's own vacuous-loop exit.
	// T-2184 remedy M1 (Brunel fix round 1, D-SLM3662): restores the shipped per-token loop's own
	// ordering -- `Transition(tokens[0])` first, `SSLM_SEQUENCE_REJECTED` on failure, THEN the busy
	// check (`e35edc1`'s own per-token body). The batched composition below had the busy check run
	// FIRST, so a sequence left `Submitted` (reachable single-threaded through
	// `sslm_decode_step_gpu`, `fixture_common.h:174`'s own documented idiom) whose first token is
	// DFA-unreachable returned `SSLM_BUSY` here instead of the shipped `SSLM_SEQUENCE_REJECTED`.
	// `*consumed` stays 0 on this path -- no earlier admitted token exists in THIS call to set
	// `ready_for_logits`, matching the shipped loop's own first-iteration disposition exactly.
	{
		uint32_t reachability_probe_state = 0;
		if (!model->schemas.Transition(*entry, seq->dfa_walk_state, static_cast<uint32_t>(tokens[0]),
		                                &reachability_probe_state)) {
			return SSLM_SEQUENCE_REJECTED;
		}
	}
	if (seq->state != superslm_gpu::SslmSequenceGpuState::Idle) return SSLM_BUSY;

	// T-2169 (Rung 3, design Sec5/Sec8): the pre-scan-then-batch composition -- host-side
	// sequential admission (DFA reachability, embed validity, position cap, in that priority),
	// then ONE call into the chunk-submission primitive for exactly `admit_count` tokens,
	// replacing the per-token Transition/embed_token/DriveGpuSeqToFullDepthForG5Bridge loop above.
	// design Sec14.3/D-SLM3478's own partial-consumption contract is preserved: tokens admitted
	// before the cause that stops the call already ran their forward pass for real and *consumed
	// already counts them -- what changes is HOW they are dispatched (one chunk call), never
	// which tokens land.
	seq->live_state.hidden_scale = seq->hidden_scale;
	seq->live_state.layer_index = seq->layer_index;
	seq->live_state.kv_saturation_count = seq->kv_saturation_count;
	seq->live_state.context_length = seq->context_length;
	const int64_t chunk_open_ctxlen = seq->context_length;

	const ChunkPreScanResult scan = RunChunkAdmissionPreScan(model, chunk_open_ctxlen,
	                                                          seq->context_cap, tokens, count, entry,
	                                                          seq->dfa_walk_state);

	int64_t derived_count = 0;
	SubmitAdmittedChunkForG5Bridge(model, seq, scan.chunk_embedding_bytes.data(), scan.admit_count,
	                                chunk_open_ctxlen, &derived_count);

	// D-SLM3622 (design Sec5 step 5): the device-computed fallback overrides whatever the
	// host-computable cause predicted, regardless of cause -- ready_for_logits stays untouched,
	// and *consumed reflects the device's own actually-committed count, not admit_count.
	const bool overridden = derived_count < static_cast<int64_t>(scan.admit_count);
	*consumed = static_cast<int32_t>(overridden ? derived_count
	                                             : static_cast<int64_t>(scan.admit_count));

	// The DFA walk state advances exactly as far as the tokens that actually committed
	// (*consumed, whether that is admit_count or the device-derived shorter count) -- nothing
	// past the actually-committed prefix legitimately advanced the schema's own state either,
	// matching design Sec14.3's "the rejected token's own effects never land" extended to the
	// device-computed fallback.
	if (*consumed > 0) {
		seq->dfa_walk_state = scan.dfa_states_after[static_cast<size_t>(*consumed) - 1];
	}

	if (overridden) {
		return SSLM_DEVICE_LOST;
	}

	switch (scan.cause) {
		case ChunkAdmissionCause::kDfa:
			// S1's own disposition (design Sec5's cause table): ready_for_logits SET, iff
			// *consumed > 0 -- unlike EMBED/POSITION_CAP, this row is not "left untouched".
			if (*consumed > 0) seq->ready_for_logits = true;
			return SSLM_SEQUENCE_REJECTED;
		case ChunkAdmissionCause::kEmbed:
			// S3's own disposition: ready_for_logits left untouched.
			return scan.embed_reject_status;
		case ChunkAdmissionCause::kPositionCap:
			// D-SLM3619: refuses the whole call with the shipped path's exact observables --
			// S4(a)'s own disposition, ready_for_logits left untouched, and the next-rejected
			// token's own incidental host-side embed side effect is reproduced.
			ReproducePositionCapBoundaryEmbedSideEffect(model, seq, tokens, scan.admit_count);
			return SSLM_DEVICE_LOST;
		case ChunkAdmissionCause::kNone:
		default:
			// S5's own disposition: ready_for_logits set iff *consumed > 0 -- always true here
			// since admit_count == count > 0 on the all-admitted path.
			if (*consumed > 0) seq->ready_for_logits = true;
			return SSLM_OK;
	}
}

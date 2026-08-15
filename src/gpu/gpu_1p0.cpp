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
#include <cstring>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "d3d12_harness.h"
#include "superslm/gpu_port.h"       // T-2113 B2: GpuLayerLayout/ComputeLayerLayout/PackLayerWeightsBytes
#include "superslm/layer_marshal.h"  // T-2113 B2: MarshalLayer/LayerBacking (relocated, tools/sslm_marshal.h)
#include "superslm/model.h"          // T-2113 B2: the REAL superslm::SslmModelView this TU needs by value

// The real SslmGpuContext this handle type opaquely names to every 1.0 API caller.
// Owns everything B1's own gate requires be OWNED rather than reached through a
// process-global name: its own harness::Device (never harness::GetDevice()'s
// static), and its own PSO/root-signature caches, keyed exactly the way the
// pre-1.0 singleton caches were (by shader base name) so the later B-sections that
// port dispatch-recording code onto this object can reuse that code's own lookup
// shape unchanged -- only the cache's OWNER moves.
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

	std::unique_ptr<SslmGpuModelHandle> h(new SslmGpuModelHandle());
	try {
		h->weights_buf = UploadResidentBufferSync(ctx->device, lw_bytes.data(), lw_bytes.size());
		h->rope_cos_buf = UploadResidentBufferSync(ctx->device, cos_bytes.data(), cos_bytes.size());
		h->rope_sin_buf = UploadResidentBufferSync(ctx->device, sin_bytes.data(), sin_bytes.size());
	} catch (const std::exception&) {
		return SSLM_DEVICE_LOST;
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
SslmGpuStatus sslm_gpu_model_unmap(SslmGpuContext* ctx, SslmGpuModelHandle* model) {
	if (!model) {
		return SSLM_OK;  // same null-is-a-no-op reasoning as sslm_gpu_context_destroy(nullptr).
	}
	if (model->live_sequences > 0) {
		return SSLM_MODEL_HAS_LIVE_SEQUENCES;
	}
	if (ctx && ctx->live_handles > 0) {
		ctx->live_handles -= 1;
	}
	model->destroyed = true;
	delete model;
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
	if (!ctx || !model || model->destroyed) {
		return SSLM_SEQUENCE_KV_BUFFER_MISMATCH;
	}
	if (context_cap < 1) {
		// Mirrors RunLayerLoopGpu's own InvalidContextCap guard (superslm_gpu.cpp:914,
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
	const bool is_submitted = seq->state == superslm_gpu::SslmSequenceGpuState::Submitted;
	if (!superslm_gpu::CallProceedsOrBusy_SeqRelease(is_submitted)) {
		return SSLM_BUSY;
	}
	if (seq->model && seq->model->live_sequences > 0) {
		seq->model->live_sequences -= 1;
	}
	if (ctx && ctx->live_handles > 0) {
		ctx->live_handles -= 1;
	}
	seq->destroyed = true;
	delete seq;
	return SSLM_OK;
}

namespace {
// T-2113 (B5): `RunLayerLoopGpuSubmit`'s own guard ladder (superslm_gpu.cpp, unchanged
// from the pre-1.0 substrate) rejects for reasons the 1.0 API's own construction already
// makes unreachable in ordinary use -- context_cap/hidden_size/head_dim are validated
// once, at sslm_gpu_seq_create/sslm_gpu_model_map time, never re-supplied per decode
// call, so a live SslmGpuSequenceHandle/SslmGpuModelHandle pair cannot in practice
// present the substrate's own ladder with a geometry it rejects. Design Sec9's own table
// assigns no dedicated 1.0 status to any of these substrate-level guards (they were never
// meant to be reachable through this API), so every one takes the same "no more specific
// 1.0-level status exists" disposition B1/B2's own null-out-parameter cases already
// established: SSLM_DEVICE_LOST.
SslmGpuStatus MapSubmitRejectionToGpuStatus(superslm::SslmForwardStatus /*st*/) {
	return SSLM_DEVICE_LOST;
}
// T-2113 (B5): the FINAL, GPU-decoded per-call result (DecodeStickyTag's own return,
// design Sec4.2's own `sslm_gpu_ready`'s `out_status` channel) -- `Ok` is the only
// non-rejecting value; every rejecting sticky-tag status is, identically to the
// submission-time guards above, a substrate-level CPU-domain violation design Sec9's
// own table assigns no dedicated 1.0 status to. Same disposition, same reason.
SslmGpuStatus MapDecodedStatusToGpuStatus(superslm::SslmForwardStatus st) {
	return st == superslm::SslmForwardStatus::Ok ? SSLM_OK : SSLM_DEVICE_LOST;
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
	// Design Sec8/Sec10 B6: per-sequence adapter binding is B6's own delivery (B6 depends
	// on B5, not the reverse, design Sec10's own dependency table). A non-null adapter
	// here is accepted but produces base-only behavior today -- named, not silently
	// claimed as adapter support, per StandardsDocument.md Sec5.6.
	(void)adapter_or_null;
	if (!ctx || !seq || seq->destroyed || !seq->model || seq->model->destroyed) {
		return SSLM_SEQUENCE_KV_BUFFER_MISMATCH;  // no channel exists for a malformed handle
		                                           // at this call besides the structural-
		                                           // invariant status (same "no more specific
		                                           // status" disposition as the helpers above).
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

	// Sync the handle's own canonical scalar fields into `live_state` (its own
	// `hidden_codes` pointer was set once, at sslm_gpu_seq_create, and needs no
	// per-call refresh -- it already aliases `hidden_codes.data()` directly).
	seq->live_state.hidden_scale = seq->hidden_scale;
	seq->live_state.layer_index = seq->layer_index;
	seq->live_state.kv_saturation_count = seq->kv_saturation_count;
	seq->live_state.context_length = seq->context_length;

	// Never read: the external-rope bridge below supplies presence/size directly from
	// `model`'s own cached fields (B2), so this call has no live SslmTensorManifest of
	// its own and needs none.
	static const superslm::SslmTensorManifest kEmptyManifest;

	superslm_gpu::GpuLayerLoopInFlight* inflight = nullptr;
	const superslm::SslmForwardStatus submit_status = superslm_gpu::RunLayerLoopGpuSubmit(
	    seq->live_state, /*layers=*/nullptr, model->num_hidden_layers, layers_to_issue,
	    model->hidden_size, model->head_dim, model->num_key_value_heads, model->intermediate_size,
	    model->context_cap, kEmptyManifest, seq->host_kv_mirror.data(), seq->host_kv_mirror.size(),
	    seq->kv_buf.Get(), &seq->kv_needs_resume_barrier, &inflight, model->weights_buf.Get(),
	    model->rope_cos_buf.Get(), model->rope_sin_buf.Get(), model->has_rope_tables,
	    model->rope_cos_elem_count, model->rope_sin_elem_count);

	if (!inflight) {
		// Rejected before submission (a guard, or an exception) -- seq/host state
		// untouched, still Idle, exactly RunLayerLoopGpu's own existing contract.
		return MapSubmitRejectionToGpuStatus(submit_status);
	}
	seq->in_flight = inflight;
	seq->state = superslm_gpu::SslmSequenceGpuState::Submitted;
	return SSLM_OK;
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
	if (!ctx || !seq || seq->destroyed) {
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
// existing blob mechanism, D-SLM3080, unchanged) rather than a second implementation.
SslmGpuStatus sslm_gpu_seq_save(SslmGpuContext* ctx, const SslmGpuSequenceHandle* seq,
                                 void* out_blob, size_t* out_blob_size) {
	if (!ctx || !seq || seq->destroyed) {
		return SSLM_SEQUENCE_KV_BUFFER_MISMATCH;
	}
	const bool is_submitted = seq->state == superslm_gpu::SslmSequenceGpuState::Submitted;
	if (!superslm_gpu::CallProceedsOrBusy_SeqSave(is_submitted)) {
		return SSLM_BUSY;  // design Sec4.2's own state table, unchanged policy.
	}
	const bool ok = superslm_gpu::SaveGpuSequenceState(seq->live_state, seq->host_kv_mirror.data(),
	                                                    seq->host_kv_mirror.size(), out_blob,
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
	if (!ctx || !model || model->destroyed || !blob) {
		return SSLM_SEQUENCE_KV_BUFFER_MISMATCH;
	}
	// Design Sec5.3: "sslm_gpu_seq_restore allocates a fresh SslmGpuSequenceHandle...
	// never reusing an existing handle's" -- the SAME allocation sslm_gpu_seq_create
	// performs, at the model's own context_cap (design Sec4.2: restore takes no
	// context_cap of its own; the model handle is the caller-independent source, the
	// same one every sequence created against this model already uses).
	SslmGpuSequenceHandle* fresh = nullptr;
	const SslmGpuStatus create_status = sslm_gpu_seq_create(ctx, model, model->context_cap, &fresh);
	if (create_status != SSLM_OK || !fresh) {
		return create_status;
	}
	// RestoreGpuSequenceState's own device round-trip (gpu_port.h/superslm_gpu.cpp,
	// D-SLM3080, unchanged): decodes the blob's header fields into `live_state` and the
	// K/V bytes into `host_kv_mirror` -- a malformed or size-mismatched blob returns
	// false, translated here the same way every other structural rejection in this file
	// is (SSLM_SEQUENCE_KV_BUFFER_MISMATCH, the create-time disposition this fresh
	// handle would have carried had its own sizing been the problem).
	const bool ok = superslm_gpu::RestoreGpuSequenceState(
	    blob, blob_size, &fresh->live_state, fresh->host_kv_mirror.data(), fresh->host_kv_mirror.size());
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
	if (!ctx || !seq || seq->destroyed) {
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

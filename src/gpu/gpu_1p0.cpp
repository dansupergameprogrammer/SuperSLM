// T-2113 (B1): production implementation of the 1.0 GPU API's context lifecycle
// (Claude/Vitruvius/t2107-gpu-core-1p0-design-2026-08-14.md Sec4.1.1, Sec10 B1).
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

	// ModelHasLiveSequences guard (design Sec9, Sec4.2's own model_unmap row): incremented
	// by B3's own sslm_gpu_seq_create, decremented on release. No SslmGpuSequenceHandle
	// type is constructible yet (B3 lands it), so this is always 0 today -- real, exercised
	// guard code (reads the same field B3 will increment), matching B1's own
	// ContextHasLiveHandles precedent rather than a stand-in.
	int64_t live_sequences = 0;

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
Microsoft::WRL::ComPtr<ID3D12Resource> UploadResidentBufferSync(superslm_gpu::harness::Device& dev,
                                                                 const void* data, size_t bytes) {
	SSLM_GPU_HR(dev.alloc->Reset());
	SSLM_GPU_HR(dev.list->Reset(dev.alloc.Get(), nullptr));
	Microsoft::WRL::ComPtr<ID3D12Resource> upload = dev.Upload(data, bytes);
	Microsoft::WRL::ComPtr<ID3D12Resource> resident = dev.MakeBuffer(
	    bytes, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST);
	dev.list->CopyResource(resident.Get(), upload.Get());
	D3D12_RESOURCE_BARRIER barrier{};
	barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
	barrier.Transition.pResource = resident.Get();
	barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST;
	barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
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
// free the residency." No SslmGpuSequenceHandle exists yet (B3), so `live_sequences` is
// always 0 today -- see this handle's own header comment for why that is real, exercised
// guard code rather than a stand-in.
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

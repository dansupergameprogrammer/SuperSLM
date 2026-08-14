// T-1986 GPU-serial port -- implementation of the `superslm_gpu::*` contract
// surface T-2024's red suite (Claude/Curie/t2019-gpu-serial-red-suite-2026-08-
// 13.md, include/superslm/gpu_port.h) link-gates against. Built stepwise, in
// Sec11's own dependency order (B1 -> B2 -> B3 -> B4 -> B5 -> B6/B7 -> B8 ->
// B11 -> B12); see the build log (Claude/Brunel/t2025-gpu-serial-build-2026-
// 08-13.md) for which steps are real GPU-verified ports and which are named,
// dated stand-ins for a step not yet reached.
//
// B1 (Sec7.1): the four primitive-battery symbols below are real, GPU-
// dispatched, bit-exact ports of the shipped CPU functions of the same name
// (src/intmath.cpp, src/forward/checked_chain_funnel.cpp), executed on real
// D3D12 hardware via src/gpu/d3d12_harness.h and src/gpu/shaders/*.hlsl.
//
// Every symbol below B1 is a STUB pending its own B-step, marked "// STUB" at
// its definition, so this translation unit provides all 32 symbols the red
// suite link-gates against (StandardsDocument.md Sec5.6: a build must not
// silently omit part of what a plan/suite requires) -- allowing the suite to
// LINK and RUN from B1 forward, with the not-yet-built steps' own cells
// failing an assertion (visible, honest, ASSERTION-RED) rather than the whole
// binary failing to link. A stub never crashes the process and never returns
// a value it is not entitled to return without being wrong in an OBSERVABLE
// way -- CHECK_MSG must be able to catch it, not be accidentally satisfied by
// it.
#include "superslm/gpu_port.h"

#include <cstring>
#include <stdexcept>
#include <vector>

#include "d3d12_harness.h"

namespace superslm_gpu {
namespace harness {

// GetModuleFileNameA-derived directory of the running executable, plus
// "shaders\\<name>.cso" -- build.bat/CMakeLists.txt (this design's own build
// scripts, Sec5.7) place the compiled shaders there alongside the test
// binary.
std::string ShaderPath(const std::string& name) {
	char path[MAX_PATH]{};
	DWORD n = GetModuleFileNameA(nullptr, path, MAX_PATH);
	std::string dir;
	if (n > 0 && n < MAX_PATH) {
		std::string full(path, n);
		size_t slash = full.find_last_of("\\/");
		dir = (slash == std::string::npos) ? "." : full.substr(0, slash);
	} else {
		dir = ".";
	}
	return dir + "\\shaders\\" + name + ".cso";
}

}  // namespace harness
}  // namespace superslm_gpu

namespace superslm_gpu {

using harness::Device;
using harness::GetDevice;
using harness::GetOrBuildPipeline;

namespace {

void PutI64(std::vector<uint8_t>& buf, int64_t v) {
	uint64_t u = static_cast<uint64_t>(v);
	uint8_t b[8];
	std::memcpy(b, &u, 8);
	buf.insert(buf.end(), b, b + 8);
}

int64_t GetI64(const std::vector<uint8_t>& buf, size_t off) {
	uint64_t u = 0;
	std::memcpy(&u, buf.data() + off, 8);
	return static_cast<int64_t>(u);
}

}  // namespace

// ===========================================================================
// B1 (Sec7.1/Sec11 B1): per-primitive battery. Real GPU dispatch, bit-exact
// against the shipped CPU function of the same name.
// ===========================================================================

int64_t DynamicScaleReciprocalGpu(int64_t dn) {
	auto& pipe = GetOrBuildPipeline("dyn_recip");
	std::vector<uint8_t> in;
	PutI64(in, dn);
	auto out = GetDevice().DispatchOne(pipe.root_sig.Get(), pipe.pso.Get(), in, 8);
	return GetI64(out, 0);
}

int64_t RequantTokenCodeWideGpu(int64_t x_i, int64_t r, int s) {
	auto& pipe = GetOrBuildPipeline("requant_wide");
	std::vector<uint8_t> in;
	PutI64(in, x_i);
	PutI64(in, r);
	PutI64(in, static_cast<int64_t>(s));
	auto out = GetDevice().DispatchOne(pipe.root_sig.Get(), pipe.pso.Get(), in, 8);
	return GetI64(out, 0);
}

bool BiasReconcileWideGpu(int64_t b, int64_t q_b, int64_t r_a, int64_t e_a, int64_t* out) {
	auto& pipe = GetOrBuildPipeline("bias_wide");
	std::vector<uint8_t> in;
	PutI64(in, b);
	PutI64(in, q_b);
	PutI64(in, r_a);
	PutI64(in, e_a);
	auto result = GetDevice().DispatchOne(pipe.root_sig.Get(), pipe.pso.Get(), in, 16);
	bool ok = GetI64(result, 0) != 0;
	if (out) *out = GetI64(result, 8);
	return ok;
}

superslm::CarriedScale CombineCarriedScaleGpu(superslm::CarriedScale a, superslm::CarriedScale b) {
	auto& pipe = GetOrBuildPipeline("combine_carried");
	std::vector<uint8_t> in;
	PutI64(in, a.m);
	PutI64(in, a.e);
	PutI64(in, b.m);
	PutI64(in, b.e);
	auto out = GetDevice().DispatchOne(pipe.root_sig.Get(), pipe.pso.Get(), in, 16);
	superslm::CarriedScale r;
	r.m = GetI64(out, 0);
	r.e = GetI64(out, 8);
	return r;
}

// ===========================================================================
// B5 (Sec5.5/Sec11 B5): the two-schedule int64 abs-max reduction. STUB pending
// B5 -- returns a sentinel that diverges from the CPU oracle on any nonzero
// row so a false pass cannot occur.
// ===========================================================================

int64_t MaxAbsReduceWideGpuScheme0(const int64_t* x, size_t n) {
	(void)x;
	(void)n;
	// STUB (B5 not yet built): INT64_MIN can never equal a real MaxAbsReduceWide
	// result (that function's own range is [1, INT64_MAX], intmath.h:217), so
	// this sentinel is always an observable mismatch, never an accidental pass.
	return INT64_MIN;
}

int64_t MaxAbsReduceWideGpuScheme1(const int64_t* x, size_t n) {
	(void)x;
	(void)n;
	return INT64_MIN;  // STUB (B5 not yet built) -- see Scheme0's own note.
}

// ===========================================================================
// B2 (Sec6/Sec11 B2): the guard-path port, isolated tier. Real GPU dispatch,
// bit-exact against the shipped CPU predicate of the same name. B2 does not
// depend on B5 (Sec11's own dependency graph: B1 -> B2 -> B4 -> ...; B1 -> B5
// -> B4), so RequantChainCheckedGpu's own reduction is a self-contained,
// single-thread guard-tier port (requant_chain_checked.hlsl's own, not B5's
// two-schedule production reduction).
// ===========================================================================

superslm::SslmForwardStatus CheckRoundingDivideByPotExponentDomainGpu(int64_t q_B, int64_t e_a) {
	auto& pipe = GetOrBuildPipeline("check_rdp_exponent");
	std::vector<uint8_t> in;
	PutI64(in, q_B);
	PutI64(in, e_a);
	auto out = GetDevice().DispatchOne(pipe.root_sig.Get(), pipe.pso.Get(), in, 8);
	const int64_t tag = GetI64(out, 0);
	switch (tag) {
		case 3:
			return superslm::SslmForwardStatus::RoundingDivideByPotExponentOutOfDomain;
		default:
			return superslm::SslmForwardStatus::Ok;
	}
}

superslm::SslmForwardStatus CheckBiasAccumulateMagnitudeDomainGpu(int64_t acc_i, int64_t b,
                                                                    int64_t q_b, int64_t r_a,
                                                                    int64_t e_a) {
	// Not exercised by any T-2019 cell (grep of tests/test_main.cpp finds no
	// caller; the design's own B2(3) construction routes the equivalent
	// coverage through RunLayerLoopGpu instead, Curie casebook Sec6). Built
	// for real anyway, per B2's own declared scope (gpu_port.h) -- reuses the
	// already-proven BiasReconcileWideGpu dispatch (B1) for the wide
	// reconcile term, then reproduces CheckBiasAccumulateMagnitudeDomain's own
	// second-stage overflow test (checked_chain_funnel.cpp:450-478) in the
	// identical unsigned two's-complement form the CPU reference uses (no UB,
	// same technique host-side as device-side would use).
	int64_t term = 0;
	const bool fits = BiasReconcileWideGpu(b, q_b, r_a, e_a, &term);
	if (!fits) return superslm::SslmForwardStatus::BiasReconcileProductOutOfDomain;
	const uint64_t ua = static_cast<uint64_t>(acc_i);
	const uint64_t ub = static_cast<uint64_t>(term);
	const uint64_t sum = ua + ub;
	const bool same_sign_operands = ((ua ^ ub) >> 63) == 0;
	const bool sum_sign_differs = ((ua ^ sum) >> 63) != 0;
	if (same_sign_operands && sum_sign_differs) {
		return superslm::SslmForwardStatus::BiasReconcileProductOutOfDomain;
	}
	return superslm::SslmForwardStatus::Ok;
}

superslm::ChainResult RequantChainCheckedGpu(const int64_t* wide_row, size_t n,
                                              const superslm::CarriedScale* incoming,
                                              size_t n_incoming,
                                              superslm::CarriedScale site_constant) {
	static constexpr size_t kMaxRow = 32;
	static constexpr size_t kMaxIncoming = 32;
	if (n > kMaxRow || n_incoming > kMaxIncoming) {
		throw std::runtime_error(
		    "RequantChainCheckedGpu: fixture exceeds this guard-tier kernel's fixed capacity "
		    "(32 row elements / 32 incoming factors) -- B4's production shader (not this B2 "
		    "guard-only tier) is where the real per-site row widths are handled");
	}
	std::vector<uint8_t> in(16 + kMaxRow * 8 + kMaxIncoming * 16 + 16, 0);
	const int64_t n_val = static_cast<int64_t>(n);            // n <= kMaxRow, always representable
	const int64_t n_inc_val = static_cast<int64_t>(n_incoming);
	std::memcpy(in.data() + 0, &n_val, 8);
	std::memcpy(in.data() + 8, &n_inc_val, 8);
	for (size_t i = 0; i < n; ++i) {
		std::memcpy(in.data() + 16 + i * 8, &wide_row[i], 8);
	}
	const size_t incoming_off = 16 + kMaxRow * 8;
	for (size_t i = 0; i < n_incoming; ++i) {
		std::memcpy(in.data() + incoming_off + i * 16 + 0, &incoming[i].m, 8);
		std::memcpy(in.data() + incoming_off + i * 16 + 8, &incoming[i].e, 8);
	}
	const size_t site_off = incoming_off + kMaxIncoming * 16;
	std::memcpy(in.data() + site_off + 0, &site_constant.m, 8);
	std::memcpy(in.data() + site_off + 8, &site_constant.e, 8);

	auto& pipe = GetOrBuildPipeline("requant_chain_checked");
	auto out = GetDevice().DispatchOne(pipe.root_sig.Get(), pipe.pso.Get(), in, 8);
	const int64_t tag = GetI64(out, 0);
	switch (tag) {
		case 1:
			return superslm::ChainResult{superslm::SslmForwardStatus::CarriedScaleMantissaOutOfDomain};
		case 2:
			return superslm::ChainResult{superslm::SslmForwardStatus::ChainInputOutOfDomain};
		default:
			return superslm::ChainResult{superslm::SslmForwardStatus::Ok};
	}
}

// ===========================================================================
// B4/B7/B11 (Sec5.6/Sec11): the composed, device-resident, multi-layer
// forward. STUB pending B3/B4/B5/B6/B7/B11 -- the largest remaining unit.
// ===========================================================================

superslm::SslmForwardStatus RunLayerLoopGpu(superslm::SequenceLayerState& seq,
                                             const superslm::LayerWeights* layers,
                                             uint32_t num_hidden_layers, uint32_t layer_budget,
                                             size_t hidden_size, size_t head_dim,
                                             size_t num_key_value_heads, size_t intermediate_size,
                                             int64_t context_cap,
                                             const superslm::SslmTensorManifest& rope_tables,
                                             uint8_t* workspace, size_t workspace_size) {
	(void)seq;
	(void)layers;
	(void)num_hidden_layers;
	(void)layer_budget;
	(void)hidden_size;
	(void)head_dim;
	(void)num_key_value_heads;
	(void)intermediate_size;
	(void)context_cap;
	(void)rope_tables;
	(void)workspace;
	(void)workspace_size;
	// STUB (B3/B4/B5/B6/B7/B11 not yet built): `seq` is left untouched (no
	// GPU-side composition exists yet), and the returned status is one no
	// fixture in this suite expects (every real fixture's CPU oracle status is
	// Ok or one of the funnel/guard rejections; KvPrecisionUnsupported is
	// never a RunLayerLoop outcome this design's build target reaches) -- an
	// observable mismatch against every T-2019 B4/B6/B7/B8/B11 cell, not an
	// accidental pass.
	return superslm::SslmForwardStatus::KvPrecisionUnsupported;
}

const int8_t* KeyRowGpu(const uint8_t* workspace, uint32_t layer, int64_t context_cap,
                         size_t num_kv_heads, size_t head_dim, size_t kv_head, int64_t position) {
	(void)workspace;
	(void)layer;
	(void)context_cap;
	(void)num_kv_heads;
	(void)head_dim;
	(void)kv_head;
	(void)position;
	// STUB (B4/B6/B7 not yet built): a static zeroed row -- safe to dereference
	// (every B11 fixture reads exactly 2 bytes at this suite's own fixed
	// head_dim=2), and all-zero fails every "K present" assertion honestly
	// (the design's own real K rows are never all-zero once landed).
	static const int8_t kZeroRow[64] = {};
	return kZeroRow;
}

const int8_t* ValueRowGpu(const uint8_t* workspace, uint32_t layer, int64_t context_cap,
                           size_t num_kv_heads, size_t head_dim, size_t kv_head, int64_t position) {
	(void)workspace;
	(void)layer;
	(void)context_cap;
	(void)num_kv_heads;
	(void)head_dim;
	(void)kv_head;
	(void)position;
	static const int8_t kZeroRow[64] = {};
	return kZeroRow;  // STUB (B4/B6/B7 not yet built) -- see KeyRowGpu's own note.
}

// ===========================================================================
// B7 (Sec5.8/Sec11 B7): the dispatch_budget contract. STUB pending B7 -- pure
// host-side policy, but sequenced after B6 per Sec11's own dependency order
// (this design's build target is not fused ahead of it).
// ===========================================================================

SslmGpuStatus PlanDispatchBudgetGpu(uint32_t dispatch_budget, uint32_t num_hidden_layers,
                                     uint32_t current_layer_position, uint32_t* out_layers_to_issue) {
	(void)dispatch_budget;
	(void)num_hidden_layers;
	(void)current_layer_position;
	if (out_layers_to_issue) *out_layers_to_issue = 999999u;  // STUB (B7 not yet built) -- never a real layer count
	return SslmGpuStatus::Busy;  // never the status any T-2019 B7 cell expects (Ok / DispatchBudgetTooSmall)
}

// ===========================================================================
// Sec5.9 (D-SLM3076-3079): the asynchronous sequence lifecycle. STUB pending
// Sec5.9's own build (sequenced with B7/B8 per Sec11).
// ===========================================================================

bool CallProceedsOrBusy_DecodeStepGpu(bool sequence_is_submitted) {
	(void)sequence_is_submitted;
	return true;  // STUB (Sec5.9 not yet built) -- wrong on the Submitted=true cell (must be false)
}
bool CallProceedsOrBusy_SeqSave(bool sequence_is_submitted) {
	(void)sequence_is_submitted;
	return true;  // STUB (Sec5.9 not yet built)
}
bool CallProceedsOrBusy_SeqReset(bool sequence_is_submitted) {
	(void)sequence_is_submitted;
	return true;  // STUB (Sec5.9 not yet built)
}
bool CallProceedsOrBusy_SeqRelease(bool sequence_is_submitted) {
	(void)sequence_is_submitted;
	return true;  // STUB (Sec5.9 not yet built)
}
bool CallProceedsOrBusy_ModelUnmap(bool any_sequence_submitted) {
	(void)any_sequence_submitted;
	return true;  // STUB (Sec5.9 not yet built) -- wrong on the any_sequence_submitted=true cell
}

bool GpuReadySignalsCompletion(bool fence_signaled, int32_t* out_ready,
                                superslm::SslmForwardStatus* out_status) {
	(void)fence_signaled;
	// STUB (Sec5.9 not yet built): *out_ready pinned to a value neither the
	// "before signal" (want 0) nor "after signal" (want 1) cell accepts.
	if (out_ready) *out_ready = -1;
	if (out_status) *out_status = superslm::SslmForwardStatus::WorkspaceTooSmall;
	return false;
}

// ===========================================================================
// B8 (Sec11 B8): device residency round-trip. STUB pending B8.
// ===========================================================================

bool SaveGpuSequenceState(const superslm::SequenceLayerState& seq, const uint8_t* workspace,
                           size_t workspace_size, void* out_blob, size_t* out_blob_size) {
	(void)seq;
	(void)workspace;
	(void)workspace_size;
	(void)out_blob;
	if (out_blob_size) *out_blob_size = 0;
	return false;  // STUB (B8 not yet built)
}
bool RestoreGpuSequenceState(const void* blob, size_t blob_size, superslm::SequenceLayerState* out_seq,
                              uint8_t* out_workspace, size_t workspace_size) {
	(void)blob;
	(void)blob_size;
	(void)out_seq;
	(void)out_workspace;
	(void)workspace_size;
	return false;  // STUB (B8 not yet built)
}

// ===========================================================================
// B3 (Sec5.1/Sec11 B3): descriptor-table binding substrate. Real GPU
// dispatch: one shader-visible descriptor heap sized to n_arrays, one RAW
// buffer SRV per array, bound as a single DESCRIPTOR_TABLE root parameter
// (the SM6.2-compatible idiom -- a bound, sized-at-creation-time table with
// an unbounded array declared in the shader, Sec5.1), never N root
// parameters.
// ===========================================================================

namespace {

// Binds `array_pointers[0..n_arrays)` (each `array_element_counts[i]` int8
// elements) through one descriptor table and reads every one back into
// `*out_widened` (one int32 per element, in array order) -- the shared
// mechanism `BindDescriptorTableAndReadback` and
// `DescriptorHeapRegionIsCleanAfterHandleRelease` both drive.
bool RunDescriptorTableBind(const int8_t* const* array_pointers, const size_t* array_element_counts,
                             size_t n_arrays, std::vector<int32_t>* out_widened) {
	harness::Device& dev = GetDevice();
	if (!dev.available || n_arrays == 0) return false;

	std::vector<uint64_t> offsets(n_arrays);
	uint64_t total_elements = 0;
	for (size_t i = 0; i < n_arrays; ++i) {
		offsets[i] = total_elements;
		total_elements += array_element_counts[i];
	}
	if (total_elements == 0) return false;

	// Per-array data buffers (upload heap, GENERIC_READ -- valid directly as
	// an SRV source, no default-heap copy needed for this structural test),
	// each padded to a 4-byte multiple for a RAW buffer view's own NumElements.
	std::vector<Microsoft::WRL::ComPtr<ID3D12Resource>> array_bufs(n_arrays);
	for (size_t i = 0; i < n_arrays; ++i) {
		const size_t padded = ((array_element_counts[i] + 3) / 4) * 4;
		std::vector<uint8_t> bytes(padded, 0);
		std::memcpy(bytes.data(), array_pointers[i], array_element_counts[i]);
		array_bufs[i] = dev.Upload(bytes.data(), padded);
	}

	D3D12_DESCRIPTOR_HEAP_DESC hd{};
	hd.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
	hd.NumDescriptors = static_cast<UINT>(n_arrays);
	hd.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
	Microsoft::WRL::ComPtr<ID3D12DescriptorHeap> heap;
	SSLM_GPU_HR(dev.dev->CreateDescriptorHeap(&hd, IID_PPV_ARGS(&heap)));
	const UINT stride =
	    dev.dev->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
	D3D12_CPU_DESCRIPTOR_HANDLE cpu_start = heap->GetCPUDescriptorHandleForHeapStart();
	for (size_t i = 0; i < n_arrays; ++i) {
		const size_t padded = ((array_element_counts[i] + 3) / 4) * 4;
		D3D12_SHADER_RESOURCE_VIEW_DESC srv{};
		srv.Format = DXGI_FORMAT_R32_TYPELESS;
		srv.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
		srv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
		srv.Buffer.FirstElement = 0;
		srv.Buffer.NumElements = static_cast<UINT>(padded / 4);
		srv.Buffer.Flags = D3D12_BUFFER_SRV_FLAG_RAW;
		D3D12_CPU_DESCRIPTOR_HANDLE h{cpu_start.ptr + i * stride};
		dev.dev->CreateShaderResourceView(array_bufs[i].Get(), &srv, h);
	}

	std::vector<int64_t> counts64(n_arrays), offsets64(n_arrays);
	for (size_t i = 0; i < n_arrays; ++i) {
		counts64[i] = static_cast<int64_t>(array_element_counts[i]);
		offsets64[i] = static_cast<int64_t>(offsets[i]);
	}
	auto counts_buf = dev.Upload(counts64.data(), counts64.size() * 8);
	auto offsets_buf = dev.Upload(offsets64.data(), offsets64.size() * 8);
	auto out_uav = dev.MakeBuffer(total_elements * 4, D3D12_HEAP_TYPE_DEFAULT,
	                               D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
	                               D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
	auto readback = dev.MakeBuffer(total_elements * 4, D3D12_HEAP_TYPE_READBACK,
	                                D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST);

	static Microsoft::WRL::ComPtr<ID3D12RootSignature> s_root_sig;
	static Microsoft::WRL::ComPtr<ID3D12PipelineState> s_pso;
	if (!s_root_sig) {
		s_root_sig = dev.MakeRootSigDescriptorTable();
		auto cso = harness::ReadFile(harness::ShaderPath("descriptor_table_readback"));
		s_pso = dev.MakePSO(s_root_sig.Get(), cso);
	}

	SSLM_GPU_HR(dev.alloc->Reset());
	SSLM_GPU_HR(dev.list->Reset(dev.alloc.Get(), s_pso.Get()));
	ID3D12DescriptorHeap* heaps[] = {heap.Get()};
	dev.list->SetDescriptorHeaps(1, heaps);
	dev.list->SetComputeRootSignature(s_root_sig.Get());
	const uint32_t n_arrays_u32 = static_cast<uint32_t>(n_arrays);
	dev.list->SetComputeRoot32BitConstants(0, 1, &n_arrays_u32, 0);
	dev.list->SetComputeRootShaderResourceView(1, counts_buf->GetGPUVirtualAddress());
	dev.list->SetComputeRootShaderResourceView(2, offsets_buf->GetGPUVirtualAddress());
	dev.list->SetComputeRootDescriptorTable(3, heap->GetGPUDescriptorHandleForHeapStart());
	dev.list->SetComputeRootUnorderedAccessView(4, out_uav->GetGPUVirtualAddress());
	dev.list->Dispatch(static_cast<UINT>((n_arrays + 63) / 64), 1, 1);
	D3D12_RESOURCE_BARRIER b{};
	b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
	b.Transition.pResource = out_uav.Get();
	b.Transition.StateBefore = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
	b.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_SOURCE;
	b.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
	dev.list->ResourceBarrier(1, &b);
	dev.list->CopyResource(readback.Get(), out_uav.Get());
	SSLM_GPU_HR(dev.list->Close());
	ID3D12CommandList* lists[] = {dev.list.Get()};
	dev.queue->ExecuteCommandLists(1, lists);
	SSLM_GPU_HR(dev.queue->Signal(dev.fence.Get(), ++dev.fence_val));
	if (dev.fence->GetCompletedValue() < dev.fence_val) {
		SSLM_GPU_HR(dev.fence->SetEventOnCompletion(dev.fence_val, dev.fence_event));
		WaitForSingleObject(dev.fence_event, INFINITE);
	}

	out_widened->resize(total_elements);
	void* p = nullptr;
	D3D12_RANGE range{0, static_cast<SIZE_T>(total_elements * 4)};
	SSLM_GPU_HR(readback->Map(0, &range, &p));
	std::memcpy(out_widened->data(), p, total_elements * 4);
	D3D12_RANGE none{0, 0};
	readback->Unmap(0, &none);
	return true;
}

}  // namespace

bool BindDescriptorTableAndReadback(const int8_t* const* array_pointers,
                                     const size_t* array_element_counts, size_t n_arrays,
                                     int8_t* out_readback_concat, size_t out_readback_capacity) {
	std::vector<int32_t> widened;
	if (!RunDescriptorTableBind(array_pointers, array_element_counts, n_arrays, &widened)) {
		return false;
	}
	if (widened.size() > out_readback_capacity) return false;
	for (size_t i = 0; i < widened.size(); ++i) {
		out_readback_concat[i] = static_cast<int8_t>(widened[i]);
	}
	return true;
}

bool DescriptorHeapRegionIsCleanAfterHandleRelease() {
	// Sec10 dim 1, Sec14 Fold G1's own sequential-release-then-remap shape,
	// realized on this design's own binding substrate: map a first "handle"
	// (one descriptor-table bind), release every device object it created
	// (ComPtr scope exit below), then map a second, independently-created
	// handle with DIFFERENT content and confirm it reads back only its own
	// pattern -- a real property of this session's device/heap lifecycle,
	// not a tautology (a bug that reused a stale GPU virtual address, or
	// failed to fence-wait before releasing, would show up here).
	const int8_t first_pattern[8] = {1, 2, 3, 4, 5, 6, 7, 8};
	const int8_t* first_ptrs[1] = {first_pattern};
	const size_t first_counts[1] = {8};
	{
		std::vector<int32_t> widened;
		if (!RunDescriptorTableBind(first_ptrs, first_counts, 1, &widened)) return false;
		for (size_t i = 0; i < 8; ++i) {
			if (widened[i] != first_pattern[i]) return false;
		}
	}  // every device object from the first bind releases here

	const int8_t second_pattern[8] = {-1, -2, -3, -4, -5, -6, -7, -8};
	const int8_t* second_ptrs[1] = {second_pattern};
	const size_t second_counts[1] = {8};
	std::vector<int32_t> widened2;
	if (!RunDescriptorTableBind(second_ptrs, second_counts, 1, &widened2)) return false;
	for (size_t i = 0; i < 8; ++i) {
		if (widened2[i] != second_pattern[i]) return false;  // residue from the first bind
	}
	return true;
}

namespace {
bool g_tier_mock_armed = false;
int g_tier_mock_value = 3;
}  // namespace

void ArmResourceBindingTierMock(int tier) {
	g_tier_mock_armed = true;
	g_tier_mock_value = tier;
}
void ClearResourceBindingTierMock() { g_tier_mock_armed = false; }

bool MapModelGpuResidencyTierCheck() {
	// Sec11 B3, D-SLM3082 (Dan's amendment 5): a preflight run before any
	// descriptor-table/UAV-table build or device allocation is attempted --
	// this design's binding architecture requires Tier 3 (D-SLM3000).
	int tier;
	if (g_tier_mock_armed) {
		tier = g_tier_mock_value;
	} else {
		tier = static_cast<int>(GetDevice().QueryResourceBindingTier());
	}
	return tier >= static_cast<int>(D3D12_RESOURCE_BINDING_TIER_3);
}

// ===========================================================================
// B12 (Sec11 B12): deterministic allocation-failure injection. STUB pending
// B3/B12 -- kGpuResidencyAllocationCallCount is a real design quantity
// (Sec5.1's read+write resource-table count) that cannot be derived correctly
// until B3's resource list is final; set to 0 for now (0 allocation-index
// cells run rather than a made-up count silently misreporting the real one --
// named explicitly here, not left implicit) so TestT2019_B12_
// InjectedFailureAtEveryAllocationIndex's own loop is a no-op until B12 names
// the real value with its derivation stated, per this ticket's own brief.
// ===========================================================================

extern const uint32_t kGpuResidencyAllocationCallCount = 0;  // STUB -- see note above; derived at B12

void ArmAllocationFailureInjection(uint32_t index) { (void)index; }  // STUB (B12 not yet built)
void ArmLowBudgetInjection(uint64_t mocked_budget_bytes) { (void)mocked_budget_bytes; }  // STUB
void ClearAllocationInjection() {}                                                        // STUB
uint32_t LiveAllocationCount() { return 1;  }  // STUB (B12 not yet built) -- wrong (never 0)
uint32_t AllocationCallsAttempted() { return 1; }  // STUB (B12 not yet built) -- wrong (never 0)
bool MapModelGpuResidencyWithInjection(uint64_t required_bytes) {
	(void)required_bytes;
	return false;  // STUB (B12 not yet built) -- wrong on the "clean call" cells (must be true)
}

}  // namespace superslm_gpu

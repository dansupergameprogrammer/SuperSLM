// T-1986 GPU-serial port -- minimal D3D12 compute harness for the B1 (and
// later) `superslm_gpu::*` symbols. This is this design's OWN build artifact
// (Sec5.7: "this decision binds this design's own build scripts"), not the
// spike-tree scratch at Claude/Laplace/gpu-determinism/gpu.hpp -- the shape is
// deliberately the same proven pattern (root SRV/UAV/32-bit-constants binding,
// no descriptor heap yet; §5.1's table-based binder lands at B3), extended
// here with a per-shader PSO cache since this design issues many small,
// distinct dispatches rather than one experiment's single shader.
#pragma once
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <d3d12.h>
#include <dxgi1_6.h>
#include <wrl/client.h>

#include <climits>
#include <cstdint>
#include <cstdio>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace superslm_gpu {
namespace harness {

using Microsoft::WRL::ComPtr;

#define SSLM_GPU_HR(x)                                                                 \
	do {                                                                                \
		HRESULT _hr = (x);                                                               \
		if (FAILED(_hr)) {                                                               \
			std::fprintf(stderr, "superslm_gpu: HR FAIL 0x%08lx at %s:%d\n",               \
			             (unsigned long)_hr, __FILE__, __LINE__);                          \
			throw std::runtime_error("D3D12 call failed");                                 \
		}                                                                                 \
	} while (0)

struct Device {
	ComPtr<IDXGIAdapter1> adapter;
	ComPtr<ID3D12Device> dev;
	ComPtr<ID3D12CommandQueue> queue;
	ComPtr<ID3D12CommandAllocator> alloc;
	ComPtr<ID3D12GraphicsCommandList> list;
	ComPtr<ID3D12Fence> fence;
	HANDLE fence_event = nullptr;
	UINT64 fence_val = 0;
	bool available = false;
	std::string init_error;

	void Init() {
		ComPtr<IDXGIFactory6> factory;
		if (FAILED(CreateDXGIFactory2(0, IID_PPV_ARGS(&factory)))) {
			init_error = "CreateDXGIFactory2 failed";
			return;
		}
		for (UINT i = 0;; ++i) {
			ComPtr<IDXGIAdapter1> a;
			if (factory->EnumAdapterByGpuPreference(i, DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE,
			                                         IID_PPV_ARGS(&a)) == DXGI_ERROR_NOT_FOUND) {
				break;
			}
			DXGI_ADAPTER_DESC1 d;
			a->GetDesc1(&d);
			if (d.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) continue;
			if (SUCCEEDED(D3D12CreateDevice(a.Get(), D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&dev)))) {
				adapter = a;
				break;
			}
		}
		if (!dev) {
			init_error = "no D3D12 hardware compute adapter found";
			return;
		}
		D3D12_COMMAND_QUEUE_DESC qd{};
		qd.Type = D3D12_COMMAND_LIST_TYPE_COMPUTE;
		SSLM_GPU_HR(dev->CreateCommandQueue(&qd, IID_PPV_ARGS(&queue)));
		SSLM_GPU_HR(dev->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COMPUTE, IID_PPV_ARGS(&alloc)));
		SSLM_GPU_HR(dev->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_COMPUTE, alloc.Get(), nullptr,
		                                    IID_PPV_ARGS(&list)));
		SSLM_GPU_HR(list->Close());
		SSLM_GPU_HR(dev->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence)));
		fence_event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
		available = true;
	}

	ComPtr<ID3D12Resource> MakeBuffer(UINT64 bytes, D3D12_HEAP_TYPE heap, D3D12_RESOURCE_FLAGS flags,
	                                   D3D12_RESOURCE_STATES state) {
		D3D12_HEAP_PROPERTIES hp{};
		hp.Type = heap;
		D3D12_RESOURCE_DESC rd{};
		rd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
		rd.Width = bytes;
		rd.Height = 1;
		rd.DepthOrArraySize = 1;
		rd.MipLevels = 1;
		rd.Format = DXGI_FORMAT_UNKNOWN;
		rd.SampleDesc.Count = 1;
		rd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
		rd.Flags = flags;
		ComPtr<ID3D12Resource> r;
		SSLM_GPU_HR(dev->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &rd, state, nullptr,
		                                          IID_PPV_ARGS(&r)));
		return r;
	}

	ComPtr<ID3D12Resource> Upload(const void* data, UINT64 bytes) {
		auto r = MakeBuffer(bytes, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_FLAG_NONE,
		                     D3D12_RESOURCE_STATE_GENERIC_READ);
		void* p = nullptr;
		D3D12_RANGE none{0, 0};
		SSLM_GPU_HR(r->Map(0, &none, &p));
		memcpy(p, data, bytes);
		r->Unmap(0, nullptr);
		return r;
	}

	// Sec5.1's own binding architecture: one descriptor table (unbounded SRV
	// range at t0, space1) plus two root SRVs (Counts, Offsets metadata) plus
	// one root 32-bit-constant (NArrays) plus one root UAV (Out) -- B3's own
	// generic binding-substrate shape (TestT2019_B3_DescriptorTableBinding_
	// KnownPatternReadback's own N-arbitrary-arrays contract), the SM6.2-
	// compatible idiom (a bound, sized-at-creation-time root parameter with an
	// unbounded array declared in the shader), not SM6.6's ResourceDescriptorHeap[].
	ComPtr<ID3D12RootSignature> MakeRootSigDescriptorTable() {
		D3D12_DESCRIPTOR_RANGE range{};
		range.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
		range.NumDescriptors = UINT_MAX;  // unbounded
		range.BaseShaderRegister = 0;     // t0
		range.RegisterSpace = 1;          // space1
		range.OffsetInDescriptorsFromTableStart = 0;

		D3D12_ROOT_PARAMETER ps[5]{};
		ps[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
		ps[0].Constants.Num32BitValues = 1;
		ps[0].Constants.ShaderRegister = 0;  // b0
		ps[0].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
		ps[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_SRV;
		ps[1].Descriptor.ShaderRegister = 1;  // t1 Counts
		ps[2].ParameterType = D3D12_ROOT_PARAMETER_TYPE_SRV;
		ps[2].Descriptor.ShaderRegister = 2;  // t2 Offsets
		ps[3].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
		ps[3].DescriptorTable.NumDescriptorRanges = 1;
		ps[3].DescriptorTable.pDescriptorRanges = &range;
		ps[3].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
		ps[4].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;
		ps[4].Descriptor.ShaderRegister = 0;  // u0 Out

		D3D12_ROOT_SIGNATURE_DESC rs{};
		rs.NumParameters = 5;
		rs.pParameters = ps;
		ComPtr<ID3DBlob> blob, err;
		HRESULT hr = D3D12SerializeRootSignature(&rs, D3D_ROOT_SIGNATURE_VERSION_1, &blob, &err);
		if (FAILED(hr)) {
			if (err) std::fprintf(stderr, "%s\n", (char*)err->GetBufferPointer());
			throw std::runtime_error("descriptor-table root signature serialization failed");
		}
		ComPtr<ID3D12RootSignature> r;
		SSLM_GPU_HR(dev->CreateRootSignature(0, blob->GetBufferPointer(), blob->GetBufferSize(),
		                                      IID_PPV_ARGS(&r)));
		return r;
	}

	// D3D12_FEATURE_DATA_D3D12_OPTIONS::ResourceBindingTier -- Sec5.1's own
	// stated hardware floor (Tier 3, D-SLM3000). Not mocked here; the mock
	// override lives at the call site (superslm_gpu.cpp), which is what B3's
	// own red-suite cell arms/clears.
	D3D12_RESOURCE_BINDING_TIER QueryResourceBindingTier() {
		D3D12_FEATURE_DATA_D3D12_OPTIONS o{};
		dev->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS, &o, sizeof(o));
		return o.ResourceBindingTier;
	}

	// T-2032/T-2035: the composed pipeline's own root signature -- one 9-value
	// 32-bit-constants block (b0: layer_index, hidden_size, head_dim,
	// num_kv_heads, context_cap, position, num_attention_heads, width,
	// intermediate_size; T-2032 shaders read only the first six), seven root
	// SRVs (t0 LayerWeights, t1 Layout, t2 RopeInfo, t3 ModelConstants [the
	// i-exp derivation's own three compile-time constants, T-2035], t4
	// SiluLut, t5 RopeCosTable, t6 RopeSinTable [T-2035: RoPE's own real
	// rotation data, not just presence/extent]), three root UAVs (u0
	// SeqState, u1 LayerScratch, u2 KvCache). Shared by every shader the
	// composed dispatch issues -- a dispatch that does not use one of these
	// bindings simply never reads it; D3D12 does not require a PSO to
	// consume every root parameter its shared signature declares.
	ComPtr<ID3D12RootSignature> MakeRootSigComposed() {
		D3D12_ROOT_PARAMETER ps[11]{};
		ps[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
		ps[0].Constants.Num32BitValues = 10;  // 10th: num_hidden_layers (commit_site.hlsl only)
		ps[0].Constants.ShaderRegister = 0;  // b0
		ps[0].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
		ps[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_SRV;
		ps[1].Descriptor.ShaderRegister = 0;  // t0 LayerWeights
		ps[2].ParameterType = D3D12_ROOT_PARAMETER_TYPE_SRV;
		ps[2].Descriptor.ShaderRegister = 1;  // t1 Layout
		ps[3].ParameterType = D3D12_ROOT_PARAMETER_TYPE_SRV;
		ps[3].Descriptor.ShaderRegister = 2;  // t2 RopeInfo
		ps[4].ParameterType = D3D12_ROOT_PARAMETER_TYPE_SRV;
		ps[4].Descriptor.ShaderRegister = 3;  // t3 ModelConstants
		ps[5].ParameterType = D3D12_ROOT_PARAMETER_TYPE_SRV;
		ps[5].Descriptor.ShaderRegister = 4;  // t4 SiluLut
		ps[6].ParameterType = D3D12_ROOT_PARAMETER_TYPE_SRV;
		ps[6].Descriptor.ShaderRegister = 5;  // t5 RopeCosTable
		ps[7].ParameterType = D3D12_ROOT_PARAMETER_TYPE_SRV;
		ps[7].Descriptor.ShaderRegister = 6;  // t6 RopeSinTable
		ps[8].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;
		ps[8].Descriptor.ShaderRegister = 0;  // u0 SeqState
		ps[9].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;
		ps[9].Descriptor.ShaderRegister = 1;  // u1 LayerScratch
		ps[10].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;
		ps[10].Descriptor.ShaderRegister = 2;  // u2 KvCache

		D3D12_ROOT_SIGNATURE_DESC rs{};
		rs.NumParameters = 11;
		rs.pParameters = ps;
		ComPtr<ID3DBlob> blob, err;
		HRESULT hr = D3D12SerializeRootSignature(&rs, D3D_ROOT_SIGNATURE_VERSION_1, &blob, &err);
		if (FAILED(hr)) {
			if (err) std::fprintf(stderr, "%s\n", (char*)err->GetBufferPointer());
			throw std::runtime_error("composed-pipeline root signature serialization failed");
		}
		ComPtr<ID3D12RootSignature> r;
		SSLM_GPU_HR(dev->CreateRootSignature(0, blob->GetBufferPointer(), blob->GetBufferSize(),
		                                      IID_PPV_ARGS(&r)));
		return r;
	}

	ComPtr<ID3D12RootSignature> MakeRootSig1SrvUav() {
		D3D12_ROOT_PARAMETER ps[2]{};
		ps[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_SRV;
		ps[0].Descriptor.ShaderRegister = 0;
		ps[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;
		ps[1].Descriptor.ShaderRegister = 0;
		D3D12_ROOT_SIGNATURE_DESC rs{};
		rs.NumParameters = 2;
		rs.pParameters = ps;
		ComPtr<ID3DBlob> blob, err;
		HRESULT hr = D3D12SerializeRootSignature(&rs, D3D_ROOT_SIGNATURE_VERSION_1, &blob, &err);
		if (FAILED(hr)) {
			if (err) std::fprintf(stderr, "%s\n", (char*)err->GetBufferPointer());
			throw std::runtime_error("root signature serialization failed");
		}
		ComPtr<ID3D12RootSignature> r;
		SSLM_GPU_HR(dev->CreateRootSignature(0, blob->GetBufferPointer(), blob->GetBufferSize(),
		                                      IID_PPV_ARGS(&r)));
		return r;
	}

	ComPtr<ID3D12PipelineState> MakePSO(ID3D12RootSignature* rs, const std::vector<uint8_t>& cso) {
		D3D12_COMPUTE_PIPELINE_STATE_DESC pd{};
		pd.pRootSignature = rs;
		pd.CS.pShaderBytecode = cso.data();
		pd.CS.BytecodeLength = cso.size();
		ComPtr<ID3D12PipelineState> p;
		SSLM_GPU_HR(dev->CreateComputePipelineState(&pd, IID_PPV_ARGS(&p)));
		return p;
	}

	// One dispatch of numthreads(1,1,1): `in_bytes` bound as a root SRV, a UAV of
	// `out_bytes` size bound and read back. Matches every B1 shader's own shape
	// (Sec7.1: "one GPU dispatch per call").
	std::vector<uint8_t> DispatchOne(ID3D12RootSignature* rs, ID3D12PipelineState* pso,
	                                  const std::vector<uint8_t>& in_bytes, size_t out_bytes) {
		auto in_buf = Upload(in_bytes.data(), in_bytes.size());
		auto uav = MakeBuffer(out_bytes, D3D12_HEAP_TYPE_DEFAULT,
		                       D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
		                       D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
		auto readback = MakeBuffer(out_bytes, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_FLAG_NONE,
		                            D3D12_RESOURCE_STATE_COPY_DEST);
		SSLM_GPU_HR(alloc->Reset());
		SSLM_GPU_HR(list->Reset(alloc.Get(), pso));
		list->SetComputeRootSignature(rs);
		list->SetComputeRootShaderResourceView(0, in_buf->GetGPUVirtualAddress());
		list->SetComputeRootUnorderedAccessView(1, uav->GetGPUVirtualAddress());
		list->Dispatch(1, 1, 1);
		D3D12_RESOURCE_BARRIER b{};
		b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
		b.Transition.pResource = uav.Get();
		b.Transition.StateBefore = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
		b.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_SOURCE;
		b.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
		list->ResourceBarrier(1, &b);
		list->CopyResource(readback.Get(), uav.Get());
		SSLM_GPU_HR(list->Close());
		ID3D12CommandList* lists[] = {list.Get()};
		queue->ExecuteCommandLists(1, lists);
		SSLM_GPU_HR(queue->Signal(fence.Get(), ++fence_val));
		if (fence->GetCompletedValue() < fence_val) {
			SSLM_GPU_HR(fence->SetEventOnCompletion(fence_val, fence_event));
			WaitForSingleObject(fence_event, INFINITE);
		}
		std::vector<uint8_t> out(out_bytes);
		void* p = nullptr;
		D3D12_RANGE range{0, (SIZE_T)out_bytes};
		SSLM_GPU_HR(readback->Map(0, &range, &p));
		memcpy(out.data(), p, out_bytes);
		D3D12_RANGE none{0, 0};
		readback->Unmap(0, &none);
		return out;
	}
};

inline Device& GetDevice() {
	static Device dev = [] {
		Device d;
		try {
			d.Init();
		} catch (const std::exception& e) {
			d.available = false;
			d.init_error = e.what();
		}
		return d;
	}();
	return dev;
}

inline std::vector<uint8_t> ReadFile(const std::string& path) {
	FILE* f = std::fopen(path.c_str(), "rb");
	if (!f) throw std::runtime_error("cannot open shader: " + path);
	std::fseek(f, 0, SEEK_END);
	long n = std::ftell(f);
	std::fseek(f, 0, SEEK_SET);
	std::vector<uint8_t> b(static_cast<size_t>(n));
	if (n > 0) {
		size_t got = std::fread(b.data(), 1, static_cast<size_t>(n), f);
		(void)got;
	}
	std::fclose(f);
	return b;
}

// Locates `<exe_dir>/shaders/<name>.cso` -- built.bat/CMake copy this design's
// own compiled shaders next to the test binary (see build.bat / CMakeLists.txt).
std::string ShaderPath(const std::string& name);

struct CachedPipeline {
	ComPtr<ID3D12RootSignature> root_sig;
	ComPtr<ID3D12PipelineState> pso;
};

// One-SRV-one-UAV pipeline cache, keyed by shader base name (e.g. "dyn_recip").
// Every B1 shader shares this exact root-signature shape (Sec11 B1: a single
// scalar-argument buffer in, a single scalar-result buffer out).
inline CachedPipeline& GetOrBuildPipeline(const std::string& name) {
	static std::map<std::string, CachedPipeline> cache;
	auto it = cache.find(name);
	if (it != cache.end()) return it->second;
	Device& dev = GetDevice();
	CachedPipeline cp;
	cp.root_sig = dev.MakeRootSig1SrvUav();
	auto cso = ReadFile(ShaderPath(name));
	cp.pso = dev.MakePSO(cp.root_sig.Get(), cso);
	auto [inserted, ok] = cache.emplace(name, std::move(cp));
	(void)ok;
	return inserted->second;
}

// T-2032/T-2035: the composed pipeline's own PSO cache, keyed by shader base
// name -- every shader sharing MakeRootSigComposed()'s signature above (the
// 14 real per-layer dispatch shaders, attn_norm_site through commit_site). A
// distinct cache from GetOrBuildPipeline's B1-shaped one-SRV-one-UAV pool
// above, since the two families use different root signatures.
inline CachedPipeline& GetOrBuildComposedPipeline(const std::string& name) {
	static std::map<std::string, CachedPipeline> cache;
	// ONE shared root-signature object across every composed-pipeline PSO --
	// not a fresh MakeRootSigComposed() per shader. D3D12's command list binds
	// root parameters against whatever root signature was last set via
	// SetComputeRootSignature independent of which PSO SetPipelineState later
	// selects; this design's own host orchestration sets the root signature
	// ONCE and swaps only the PSO across a layer's dispatches (Sec5.4's
	// per-site dispatch shape), so every PSO here MUST share the identical
	// root-signature object the command list has bound, not merely a
	// structurally-identical distinct one.
	static ComPtr<ID3D12RootSignature> s_shared_root_sig;
	auto it = cache.find(name);
	if (it != cache.end()) return it->second;
	Device& dev = GetDevice();
	if (!s_shared_root_sig) s_shared_root_sig = dev.MakeRootSigComposed();
	CachedPipeline cp;
	cp.root_sig = s_shared_root_sig;
	auto cso = ReadFile(ShaderPath(name));
	cp.pso = dev.MakePSO(cp.root_sig.Get(), cso);
	auto [inserted, ok] = cache.emplace(name, std::move(cp));
	(void)ok;
	return inserted->second;
}

}  // namespace harness
}  // namespace superslm_gpu

// Disposable D3D12 compute plumbing for the GPU-determinism commission.
// Scratch construction (Laplace) — not product. Targets the REAL hardware adapter
// (the 2080 SUPER), never WARP: WARP would prove nothing about silicon.
#pragma once
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <d3d12.h>
#include <dxgi1_6.h>
#include <wrl/client.h>
#include <vector>
#include <string>
#include <cstdio>
#include <cstdint>
#include <stdexcept>

using Microsoft::WRL::ComPtr;

#define HR(x) do { HRESULT _hr = (x); if (FAILED(_hr)) { \
    fprintf(stderr, "HR FAIL 0x%08lx at %s:%d\n", (unsigned long)_hr, __FILE__, __LINE__); \
    throw std::runtime_error("HR"); } } while(0)

struct Gpu {
    ComPtr<IDXGIAdapter1> adapter;
    ComPtr<ID3D12Device>  dev;
    ComPtr<ID3D12CommandQueue> queue;
    ComPtr<ID3D12CommandAllocator> alloc;
    ComPtr<ID3D12GraphicsCommandList> list;
    ComPtr<ID3D12Fence> fence;
    HANDLE fenceEvent = nullptr;
    UINT64 fenceVal = 0;
    std::wstring adapterName;

    void init() {
        UINT flags = 0;
#ifdef _DEBUG
        { ComPtr<ID3D12Debug> dbg; if (SUCCEEDED(D3D12GetDebugInterface(IID_PPV_ARGS(&dbg)))) { dbg->EnableDebugLayer(); flags |= DXGI_CREATE_FACTORY_DEBUG; } }
#endif
        ComPtr<IDXGIFactory6> factory;
        HR(CreateDXGIFactory2(flags, IID_PPV_ARGS(&factory)));
        // LAPLACE_WARP=1 forces Microsoft's WARP software rasterizer — a SECOND, fully
        // independent D3D12 implementation (its own int64 emulation + DXIL lowering).
        // Not silicon; a second implementation of the same spec, for corroboration only.
        if (getenv("LAPLACE_WARP")) {
            ComPtr<IDXGIAdapter1> warp;
            HR(factory->EnumWarpAdapter(IID_PPV_ARGS(&warp)));
            if (SUCCEEDED(D3D12CreateDevice(warp.Get(), D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&dev)))) {
                adapter = warp; adapterName = L"WARP (software, Microsoft)";
            }
        } else {
            // Pick the highest-perf HARDWARE adapter (not WARP).
            for (UINT i = 0; ; ++i) {
                ComPtr<IDXGIAdapter1> a;
                if (factory->EnumAdapterByGpuPreference(i, DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE, IID_PPV_ARGS(&a)) == DXGI_ERROR_NOT_FOUND)
                    break;
                DXGI_ADAPTER_DESC1 d; a->GetDesc1(&d);
                if (d.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) continue;
                if (SUCCEEDED(D3D12CreateDevice(a.Get(), D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&dev)))) {
                    adapter = a; adapterName = d.Description; break;
                }
            }
        }
        if (!dev) throw std::runtime_error("no D3D12 device");

        D3D12_COMMAND_QUEUE_DESC qd{}; qd.Type = D3D12_COMMAND_LIST_TYPE_COMPUTE;
        HR(dev->CreateCommandQueue(&qd, IID_PPV_ARGS(&queue)));
        HR(dev->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COMPUTE, IID_PPV_ARGS(&alloc)));
        HR(dev->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_COMPUTE, alloc.Get(), nullptr, IID_PPV_ARGS(&list)));
        HR(list->Close());
        HR(dev->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence)));
        fenceEvent = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    }

    void reportFeatures() {
        D3D12_FEATURE_DATA_D3D12_OPTIONS1 o1{};
        dev->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS1, &o1, sizeof(o1));
        D3D12_FEATURE_DATA_D3D12_OPTIONS o0{};
        dev->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS, &o0, sizeof(o0));
        D3D12_FEATURE_DATA_SHADER_MODEL sm{ D3D_SHADER_MODEL_6_7 };
        dev->CheckFeatureSupport(D3D12_FEATURE_SHADER_MODEL, &sm, sizeof(sm));
        wprintf(L"# adapter: %s\n", adapterName.c_str());
        printf("# Int64ShaderOps: %d\n", (int)o1.Int64ShaderOps);
        printf("# WaveOps: %d  WaveLaneCountMin: %u  WaveLaneCountMax: %u\n",
               (int)o1.WaveOps, o1.WaveLaneCountMin, o1.WaveLaneCountMax);
        printf("# HighestShaderModel: 0x%x\n", (unsigned)sm.HighestShaderModel);
    }

    ComPtr<ID3D12Resource> makeBuffer(UINT64 bytes, D3D12_HEAP_TYPE heap,
                                      D3D12_RESOURCE_FLAGS f, D3D12_RESOURCE_STATES st) {
        D3D12_HEAP_PROPERTIES hp{}; hp.Type = heap;
        D3D12_RESOURCE_DESC rd{};
        rd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        rd.Width = bytes; rd.Height = 1; rd.DepthOrArraySize = 1; rd.MipLevels = 1;
        rd.Format = DXGI_FORMAT_UNKNOWN; rd.SampleDesc.Count = 1;
        rd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR; rd.Flags = f;
        ComPtr<ID3D12Resource> r;
        HR(dev->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &rd, st, nullptr, IID_PPV_ARGS(&r)));
        return r;
    }

    ComPtr<ID3D12Resource> upload(const void* data, UINT64 bytes) {
        auto r = makeBuffer(bytes, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ);
        void* p = nullptr; D3D12_RANGE none{0,0};
        HR(r->Map(0, &none, &p)); memcpy(p, data, bytes); r->Unmap(0, nullptr);
        return r;
    }

    ComPtr<ID3D12RootSignature> makeRootSig(UINT numRootConst, UINT numSRV, bool hasUAV) {
        std::vector<D3D12_ROOT_PARAMETER> ps;
        if (numRootConst) {
            D3D12_ROOT_PARAMETER p{}; p.ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
            p.Constants.Num32BitValues = numRootConst; p.Constants.ShaderRegister = 0;
            p.ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL; ps.push_back(p);
        }
        for (UINT i = 0; i < numSRV; ++i) {
            D3D12_ROOT_PARAMETER p{}; p.ParameterType = D3D12_ROOT_PARAMETER_TYPE_SRV;
            p.Descriptor.ShaderRegister = i; ps.push_back(p);
        }
        if (hasUAV) {
            D3D12_ROOT_PARAMETER p{}; p.ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;
            p.Descriptor.ShaderRegister = 0; ps.push_back(p);
        }
        D3D12_ROOT_SIGNATURE_DESC rs{}; rs.NumParameters = (UINT)ps.size(); rs.pParameters = ps.data();
        ComPtr<ID3DBlob> blob, err;
        HRESULT hr = D3D12SerializeRootSignature(&rs, D3D_ROOT_SIGNATURE_VERSION_1, &blob, &err);
        if (FAILED(hr)) { if (err) fprintf(stderr, "%s\n", (char*)err->GetBufferPointer()); throw std::runtime_error("rootsig"); }
        ComPtr<ID3D12RootSignature> r;
        HR(dev->CreateRootSignature(0, blob->GetBufferPointer(), blob->GetBufferSize(), IID_PPV_ARGS(&r)));
        return r;
    }

    ComPtr<ID3D12PipelineState> makePSO(ID3D12RootSignature* rs, const std::vector<uint8_t>& cso) {
        D3D12_COMPUTE_PIPELINE_STATE_DESC pd{}; pd.pRootSignature = rs;
        pd.CS.pShaderBytecode = cso.data(); pd.CS.BytecodeLength = cso.size();
        ComPtr<ID3D12PipelineState> p;
        HR(dev->CreateComputePipelineState(&pd, IID_PPV_ARGS(&p)));
        return p;
    }

    // One dispatch; read back `outBytes` from a UAV default buffer. Inputs bound as root SRVs.
    std::vector<uint8_t> dispatch(ID3D12RootSignature* rs, ID3D12PipelineState* pso,
                                  const std::vector<uint32_t>& rootConsts,
                                  const std::vector<ID3D12Resource*>& srvs,
                                  ID3D12Resource* uav, UINT64 outBytes,
                                  UINT gx, UINT gy = 1, UINT gz = 1) {
        auto readback = makeBuffer(outBytes, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST);
        HR(alloc->Reset()); HR(list->Reset(alloc.Get(), pso));
        list->SetComputeRootSignature(rs);
        UINT slot = 0;
        if (!rootConsts.empty()) { list->SetComputeRoot32BitConstants(slot++, (UINT)rootConsts.size(), rootConsts.data(), 0); }
        for (auto* s : srvs) list->SetComputeRootShaderResourceView(slot++, s->GetGPUVirtualAddress());
        list->SetComputeRootUnorderedAccessView(slot++, uav->GetGPUVirtualAddress());
        list->Dispatch(gx, gy, gz);
        D3D12_RESOURCE_BARRIER b{}; b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        b.Transition.pResource = uav; b.Transition.StateBefore = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
        b.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_SOURCE;
        b.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
        list->ResourceBarrier(1, &b);
        list->CopyResource(readback.Get(), uav);
        HR(list->Close());
        ID3D12CommandList* lists[] = { list.Get() };
        queue->ExecuteCommandLists(1, lists);
        HR(queue->Signal(fence.Get(), ++fenceVal));
        if (fence->GetCompletedValue() < fenceVal) { HR(fence->SetEventOnCompletion(fenceVal, fenceEvent)); WaitForSingleObject(fenceEvent, INFINITE); }
        std::vector<uint8_t> out(outBytes);
        void* p = nullptr; D3D12_RANGE range{0, (SIZE_T)outBytes};
        HR(readback->Map(0, &range, &p)); memcpy(out.data(), p, outBytes);
        D3D12_RANGE none{0,0}; readback->Unmap(0, &none);
        return out;
    }
};

inline std::vector<uint8_t> readFile(const char* path) {
    FILE* f = fopen(path, "rb"); if (!f) { fprintf(stderr, "cannot open %s\n", path); throw std::runtime_error("open"); }
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    std::vector<uint8_t> b(n); fread(b.data(), 1, n, f); fclose(f); return b;
}

// FNV-1a 64-bit over a byte span — the golden-hash oracle.
inline uint64_t fnv1a(const void* data, size_t n) {
    const uint8_t* p = (const uint8_t*)data; uint64_t h = 1469598103934665603ULL;
    for (size_t i = 0; i < n; ++i) { h ^= p[i]; h *= 1099511628211ULL; }
    return h;
}

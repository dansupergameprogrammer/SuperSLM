// Multi-row GPU plan (rev 1.3.2) cell 4.9 [every machine]: record the wave sizes. Prints
//   wave options1 WaveLaneCountMin=<n> WaveLaneCountMax=<n> WaveOps=<0|1> adapter="<description>"
//   wave probe group=256 lane_counts=<sorted distinct values> max_lane_index=<n>
// from D3D12_FEATURE_DATA_D3D12_OPTIONS1 and from mr2_wave_probe.hlsl (one 256-thread group, the widened sites'
// shape). Diagnostic only: it grades nothing, and the record files the numbers beside every run of 4.10/6.1.
// Usage: mr2_wave_probe.exe --cso=PATH     exit 0 when both readings were taken.
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <d3d12.h>
#include <dxgi1_4.h>

#include <cstdint>
#include <cstdio>
#include <set>
#include <string>
#include <vector>

namespace {
template <class T>
void Rel(T*& p) {
	if (p) p->Release();
	p = nullptr;
}
bool ReadAll(const std::string& path, std::vector<uint8_t>* out) {
	std::FILE* f = std::fopen(path.c_str(), "rb");
	if (!f) return false;
	std::fseek(f, 0, SEEK_END);
	const long n = std::ftell(f);
	std::fseek(f, 0, SEEK_SET);
	out->resize(n > 0 ? static_cast<size_t>(n) : 0);
	const bool ok = n > 0 && std::fread(out->data(), 1, out->size(), f) == out->size();
	std::fclose(f);
	return ok;
}
D3D12_RESOURCE_DESC BufDesc(UINT64 bytes, D3D12_RESOURCE_FLAGS flags) {
	D3D12_RESOURCE_DESC d{};
	d.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
	d.Width = bytes;
	d.Height = 1;
	d.DepthOrArraySize = 1;
	d.MipLevels = 1;
	d.SampleDesc.Count = 1;
	d.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
	d.Flags = flags;
	return d;
}
}  // namespace

int main(int argc, char** argv) {
	std::string cso;
	for (int i = 1; i < argc; ++i) {
		const std::string s = argv[i];
		if (s.rfind("--cso=", 0) == 0) cso = s.substr(6);
	}
	std::vector<uint8_t> code;
	if (cso.empty() || !ReadAll(cso, &code)) { std::printf("FAIL usage: mr2_wave_probe --cso=PATH\n"); return 2; }

	IDXGIFactory4* fac = nullptr;
	IDXGIAdapter1* ad = nullptr;
	ID3D12Device* dev = nullptr;
	if (FAILED(CreateDXGIFactory1(__uuidof(IDXGIFactory4), reinterpret_cast<void**>(&fac)))) { std::printf("FAIL dxgi\n"); return 1; }
	for (UINT i = 0; fac->EnumAdapters1(i, &ad) != DXGI_ERROR_NOT_FOUND; ++i) {
		if (SUCCEEDED(D3D12CreateDevice(ad, D3D_FEATURE_LEVEL_11_0, __uuidof(ID3D12Device), reinterpret_cast<void**>(&dev)))) break;
		Rel(ad);
	}
	if (!dev) { std::printf("FAIL no D3D12 device\n"); return 1; }
	DXGI_ADAPTER_DESC1 desc{};
	ad->GetDesc1(&desc);
	char name[256] = {};
	WideCharToMultiByte(CP_UTF8, 0, desc.Description, -1, name, sizeof name - 1, nullptr, nullptr);
	D3D12_FEATURE_DATA_D3D12_OPTIONS1 o1{};
	const bool have_o1 = SUCCEEDED(dev->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS1, &o1, sizeof o1));
	std::printf("wave options1 %s WaveLaneCountMin=%u WaveLaneCountMax=%u WaveOps=%d adapter=\"%s\"\n",
	            have_o1 ? "ok" : "unavailable", o1.WaveLaneCountMin, o1.WaveLaneCountMax, o1.WaveOps ? 1 : 0, name);

	// Root signature: one root UAV (u0).
	D3D12_ROOT_PARAMETER rp{};
	rp.ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;
	rp.ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
	D3D12_ROOT_SIGNATURE_DESC rsd{};
	rsd.NumParameters = 1;
	rsd.pParameters = &rp;
	ID3DBlob* blob = nullptr;
	ID3DBlob* err = nullptr;
	ID3D12RootSignature* rs = nullptr;
	ID3D12PipelineState* pso = nullptr;
	if (FAILED(D3D12SerializeRootSignature(&rsd, D3D_ROOT_SIGNATURE_VERSION_1, &blob, &err)) ||
	    FAILED(dev->CreateRootSignature(0, blob->GetBufferPointer(), blob->GetBufferSize(), __uuidof(ID3D12RootSignature),
	                                    reinterpret_cast<void**>(&rs)))) {
		std::printf("FAIL root signature\n");
		return 1;
	}
	D3D12_COMPUTE_PIPELINE_STATE_DESC pd{};
	pd.pRootSignature = rs;
	pd.CS = {code.data(), code.size()};
	if (FAILED(dev->CreateComputePipelineState(&pd, __uuidof(ID3D12PipelineState), reinterpret_cast<void**>(&pso)))) {
		std::printf("FAIL pipeline\n");
		return 1;
	}
	const UINT64 bytes = 256 * 8;
	D3D12_HEAP_PROPERTIES def{}, rb{};
	def.Type = D3D12_HEAP_TYPE_DEFAULT;
	rb.Type = D3D12_HEAP_TYPE_READBACK;
	ID3D12Resource* buf = nullptr;
	ID3D12Resource* back = nullptr;
	D3D12_RESOURCE_DESC bd = BufDesc(bytes, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
	D3D12_RESOURCE_DESC rd = BufDesc(bytes, D3D12_RESOURCE_FLAG_NONE);
	if (FAILED(dev->CreateCommittedResource(&def, D3D12_HEAP_FLAG_NONE, &bd, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr,
	                                        __uuidof(ID3D12Resource), reinterpret_cast<void**>(&buf))) ||
	    FAILED(dev->CreateCommittedResource(&rb, D3D12_HEAP_FLAG_NONE, &rd, D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
	                                        __uuidof(ID3D12Resource), reinterpret_cast<void**>(&back)))) {
		std::printf("FAIL buffers\n");
		return 1;
	}
	ID3D12CommandQueue* q = nullptr;
	ID3D12CommandAllocator* al = nullptr;
	ID3D12GraphicsCommandList* cl = nullptr;
	ID3D12Fence* fence = nullptr;
	D3D12_COMMAND_QUEUE_DESC qd{};
	qd.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
	if (FAILED(dev->CreateCommandQueue(&qd, __uuidof(ID3D12CommandQueue), reinterpret_cast<void**>(&q))) ||
	    FAILED(dev->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, __uuidof(ID3D12CommandAllocator), reinterpret_cast<void**>(&al))) ||
	    FAILED(dev->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, al, pso, __uuidof(ID3D12GraphicsCommandList),
	                                  reinterpret_cast<void**>(&cl))) ||
	    FAILED(dev->CreateFence(0, D3D12_FENCE_FLAG_NONE, __uuidof(ID3D12Fence), reinterpret_cast<void**>(&fence)))) {
		std::printf("FAIL queue\n");
		return 1;
	}
	cl->SetComputeRootSignature(rs);
	cl->SetComputeRootUnorderedAccessView(0, buf->GetGPUVirtualAddress());
	cl->Dispatch(1, 1, 1);
	D3D12_RESOURCE_BARRIER b{};
	b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
	b.Transition.pResource = buf;
	b.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
	b.Transition.StateBefore = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
	b.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_SOURCE;
	cl->ResourceBarrier(1, &b);
	cl->CopyResource(back, buf);
	cl->Close();
	ID3D12CommandList* lists[] = {cl};
	q->ExecuteCommandLists(1, lists);
	q->Signal(fence, 1);
	HANDLE ev = CreateEventA(nullptr, FALSE, FALSE, nullptr);
	fence->SetEventOnCompletion(1, ev);
	WaitForSingleObject(ev, 30000);
	uint32_t* p = nullptr;
	D3D12_RANGE r{0, static_cast<SIZE_T>(bytes)};
	if (FAILED(back->Map(0, &r, reinterpret_cast<void**>(&p)))) { std::printf("FAIL map\n"); return 1; }
	std::set<uint32_t> counts;
	uint32_t max_idx = 0;
	for (int t = 0; t < 256; ++t) {
		counts.insert(p[t * 2]);
		if (p[t * 2 + 1] > max_idx) max_idx = p[t * 2 + 1];
	}
	back->Unmap(0, nullptr);
	std::string cs;
	for (uint32_t c : counts) cs += (cs.empty() ? "" : ",") + std::to_string(c);
	std::printf("wave probe group=256 lane_counts=%s max_lane_index=%u\n", cs.c_str(), max_idx);
	return 0;
}

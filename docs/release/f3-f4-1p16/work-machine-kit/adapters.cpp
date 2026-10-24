// Lists the D3D12-capable hardware adapters in DXGI's EnumAdapters1 order, the order SSLM_GPU_ADAPTER_INDEX indexes:
// one line per adapter, "index vendor device dedicated_mb name". The F2 and F3 work-machine kits pick the RX 7900 XTX
// (vendor 1002) by its dedicated memory, since the work machine's Ryzen also has an AMD integrated GPU.
// (F3: the home box's pieces use it too, to pick the RTX 2080 SUPER by vendor 10de rather than by a fixed index.)
#include <cstdio>
#include <dxgi1_6.h>
#include <wrl/client.h>
#pragma comment(lib, "dxgi.lib")

int main() {
	Microsoft::WRL::ComPtr<IDXGIFactory1> f;
	if (FAILED(CreateDXGIFactory1(IID_PPV_ARGS(&f)))) {
		std::printf("ERROR CreateDXGIFactory1\n");
		return 1;
	}
	for (UINT i = 0;; ++i) {
		Microsoft::WRL::ComPtr<IDXGIAdapter1> a;
		if (f->EnumAdapters1(i, &a) == DXGI_ERROR_NOT_FOUND) break;
		DXGI_ADAPTER_DESC1 d{};
		a->GetDesc1(&d);
		if (d.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) continue;
		char name[128] = {};
		WideCharToMultiByte(CP_UTF8, 0, d.Description, -1, name, sizeof(name) - 1, nullptr, nullptr);
		std::printf("%u %04x %04x %llu %s\n", i, d.VendorId, d.DeviceId,
		            static_cast<unsigned long long>(d.DedicatedVideoMemory / (1024ull * 1024ull)), name);
	}
	return 0;
}

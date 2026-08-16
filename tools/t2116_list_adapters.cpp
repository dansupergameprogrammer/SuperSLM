// T-2116 (cross-vendor certification package): a dedicated, minimal adapter-enumeration
// tool for run_crossvendor.ps1. The battery tools (t2113_b*_smoke.cpp etc.) build their
// GPU device through superslm_gpu::harness::GetDevice(), whose Init() failure path
// (src/gpu/d3d12_harness.h) records a diagnostic string in Device::init_error but has no
// caller that surfaces it to stdout/stderr on a bad SSLM_GPU_ADAPTER_INDEX -- a battery
// tool run against a non-existent adapter index just reports ordinary check failures
// (sslm_gpu_context_create returning non-OK), indistinguishable in its own output from a
// real defect on a real adapter. Enumerating adapters by probing the battery tools with
// increasing indices therefore cannot reliably tell "no more adapters" apart from "this
// adapter is real but broken."
//
// This tool sidesteps that entirely: raw DXGI enumeration only, no superslm_gpu
// dependency, no model/artifact loading, no D3D12 device creation beyond the minimum
// needed to confirm an adapter is usable. Prints one line per adapter EnumAdapters1
// reports -- hardware and software (WARP) alike, each tagged SOFTWARE=0/1 -- in the
// identical raw index order SSLM_GPU_ADAPTER_INDEX selects by, then exits 0.
// run_crossvendor.ps1 parses this tool's own output to build its adapter list (a
// hardware adapter that failed device creation is USABLE=0, not omitted -- see that
// script's own header comment for why a hardware adapter is never silently dropped)
// -- it never re-derives adapter count from a battery tool's exit code.
//
// Usage: t2116_list_adapters   (no arguments; always exits 0)
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <d3d12.h>
#include <dxgi1_6.h>
#include <wrl/client.h>

#include <cstdio>
#include <cwchar>

using Microsoft::WRL::ComPtr;

int main() {
	ComPtr<IDXGIFactory6> factory;
	if (FAILED(CreateDXGIFactory2(0, IID_PPV_ARGS(&factory)))) {
		std::printf("ERROR: CreateDXGIFactory2 failed\n");
		return 0;
	}
	for (UINT i = 0;; ++i) {
		ComPtr<IDXGIAdapter1> a;
		HRESULT hr = factory->EnumAdapters1(i, &a);
		if (hr == DXGI_ERROR_NOT_FOUND) break;
		if (FAILED(hr)) {
			std::printf("ERROR: EnumAdapters1(%u) failed\n", i);
			break;
		}
		DXGI_ADAPTER_DESC1 d;
		a->GetDesc1(&d);
		if (d.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) {
			std::wprintf(L"INDEX=%u SOFTWARE=1 NAME=%s\n", i, d.Description);
			continue;
		}
		ComPtr<ID3D12Device> dev;
		bool usable = SUCCEEDED(D3D12CreateDevice(a.Get(), D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&dev)));
		std::wprintf(L"INDEX=%u SOFTWARE=0 USABLE=%d VENDOR=0x%04x DEVICEID=0x%04x DEDICATED_VIDEO_MB=%llu NAME=%s\n",
		             i, usable ? 1 : 0, d.VendorId, d.DeviceId,
		             (unsigned long long)(d.DedicatedVideoMemory / (1024ull * 1024ull)), d.Description);
	}
	return 0;
}

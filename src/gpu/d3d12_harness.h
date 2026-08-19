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
#include <cstdlib>
#include <cwchar>
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

	// T-2101 (dispatch-overhead decomposition, D-SLM3302/D-SLM3304's own follow-up): a GPU
	// timestamp query heap, created once and reused every call. `RunLayerLoopGpu` ends one query
	// PER DISPATCH BOUNDARY -- immediately before every `bind_and_dispatch` call, plus one final
	// query after the last -- so the resolved ticks measure GPU-BUSY time both for the composed
	// dispatch chain as a whole (boundary 0 to the last) AND per individual dispatch (consecutive
	// boundary deltas), excluding command list recording, submission, and readback (all measured
	// CPU-side, `GpuCallTiming` below). COMPUTE command lists support timestamp queries on every
	// D3D12 Tier-3-capable device this design targets (D-SLM3000); a device that cannot resolve
	// them would already have failed `dev.available` well upstream of this heap ever being read.
	//
	// Sized for `kMaxTimestampSlots` boundaries -- generous headroom over the real 1.5B tier's own
	// 28 layers * 22 sites/layer + 1 = 617 boundaries this ticket's own measurement round actually
	// uses (22, not 17, since T-2101's own fix round split five sites' GEMM step into its own
	// dispatch; `RunLayerLoopGpu` clamps to this capacity rather than overrunning the heap on a
	// larger `num_hidden_layers * layer_budget`).
	static constexpr UINT kMaxTimestampSlots = 8192;
	ComPtr<ID3D12QueryHeap> timestamp_heap;
	ComPtr<ID3D12Resource> timestamp_readback;
	UINT64 timestamp_frequency = 0;  // ticks per second, from the command queue

	// T-2116 (cross-vendor certification package): a machine used for cross-vendor
	// certification may have more than one D3D12-capable adapter installed (a discrete GPU
	// plus an iGPU, or two discrete GPUs). The default loop below picks whichever adapter
	// DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE ranks first -- silently, with no way for a caller
	// to name a different one. SSLM_GPU_ADAPTER_INDEX, when set to a non-negative integer,
	// overrides that default and selects by RAW EnumAdapters1 index instead (the same
	// enumeration order `run_crossvendor.ps1` uses to label results per adapter) -- a
	// requested index that does not exist, is software, fails device creation, OR IS NOT A
	// WELL-FORMED NON-NEGATIVE INTEGER (T-2116 fix round, Claude/Poirot's own S6: `std::atoi`
	// has no failure signal and silently read a malformed value as 0, selecting adapter 0
	// with no diagnostic -- the exact silent fallback this comment already promised never to
	// do) is a loud init failure (init_error names the index or the malformed value), never a
	// silent fallback to adapter 0 or to the default-preference path.
	//
	// The "# adapter: <name>" print (below, on a successful selection) is gated on the
	// override actually being requested -- i.e. SSLM_GPU_ADAPTER_INDEX was set at all, valid
	// or not -- per the same fix round's S2: this function is also reached from the public
	// 1.0 C API (`sslm_gpu_context_create`, gpu_1p0.cpp), once per context, on every
	// caller's own stdout, unconditionally, with no way to suppress it and no query to ask
	// which adapter a context landed on. `run_crossvendor.ps1` sets the env var for every
	// cell it runs, so certification output is unaffected; every other consumer of the 1.0
	// API (unset env var) gets byte-identical stdout to before this ticket.
	void Init() {
		// T-2169 (D-SLM3649's own owed evidence, Dan's review): SSLM_GPU_ENABLE_DEBUG_LAYER, when
		// set, turns on the D3D12 debug layer (and GPU-based validation, when the installed SDK
		// supports it) BEFORE any device is created -- the only order the API allows a debug
		// device to be produced in. This is what settles D-SLM3649's own driver-defect attribution
		// in either direction: replaying the chunk_tokens=8 crashing shape under this flag and
		// reading whether the validation layer reports anything on OUR command list, before the
		// driver's own recursion fires. Off by default (unset env var): zero behavioral change,
		// zero performance cost, matching this file's own established SSLM_GPU_ADAPTER_INDEX
		// convention for a diagnostic-only, opt-in override.
		{
			char buf[8] = {0};
			DWORD n = GetEnvironmentVariableA("SSLM_GPU_ENABLE_DEBUG_LAYER", buf, sizeof(buf));
			if (n > 0 && buf[0] == '1') {
				ComPtr<ID3D12Debug> debug0;
				if (SUCCEEDED(D3D12GetDebugInterface(IID_PPV_ARGS(&debug0)))) {
					debug0->EnableDebugLayer();
					ComPtr<ID3D12Debug1> debug1;
					if (SUCCEEDED(debug0.As(&debug1))) {
						debug1->SetEnableGPUBasedValidation(TRUE);
					}
					std::fprintf(stderr, "# SSLM_GPU_ENABLE_DEBUG_LAYER=1: D3D12 debug layer%s enabled\n",
					             debug1 ? " + GPU-based validation" : " enabled (GPU-based validation unavailable)");
				} else {
					std::fprintf(stderr,
					              "# SSLM_GPU_ENABLE_DEBUG_LAYER=1 requested but D3D12GetDebugInterface "
					              "failed -- the Windows 'Graphics Tools' optional feature is likely not "
					              "installed on this machine; proceeding WITHOUT the debug layer\n");
				}
			}
		}
		ComPtr<IDXGIFactory6> factory;
		if (FAILED(CreateDXGIFactory2(0, IID_PPV_ARGS(&factory)))) {
			init_error = "CreateDXGIFactory2 failed";
			return;
		}

		int override_index = -1;
		bool override_requested = false;
		{
			char buf[32] = {0};
			DWORD n = GetEnvironmentVariableA("SSLM_GPU_ADAPTER_INDEX", buf, sizeof(buf));
			if (n > 0) {
				override_requested = true;
				if (n >= sizeof(buf)) {
					// Too long to have fit -- GetEnvironmentVariableA truncates silently on a
					// too-small buffer, so a truncated value must never be parsed: it could
					// read as a different, shorter, entirely valid-looking index.
					init_error =
					    "SSLM_GPU_ADAPTER_INDEX is set but longer than this parser accepts (" +
					    std::to_string(sizeof(buf) - 1) +
					    " chars) -- refusing rather than parsing a truncated value";
					return;
				}
				char* endp = nullptr;
				long v = std::strtol(buf, &endp, 10);
				const bool well_formed = (endp != buf) && (*endp == '\0');
				if (!well_formed || v < 0 || v > static_cast<long>(INT_MAX)) {
					init_error = "SSLM_GPU_ADAPTER_INDEX=\"" + std::string(buf) +
					              "\" is not a well-formed non-negative integer -- refusing "
					              "rather than guessing which adapter was meant";
					return;
				}
				override_index = static_cast<int>(v);
			}
		}

		if (override_index >= 0) {
			ComPtr<IDXGIAdapter1> a;
			HRESULT hr = factory->EnumAdapters1(static_cast<UINT>(override_index), &a);
			if (hr == DXGI_ERROR_NOT_FOUND) {
				init_error = "SSLM_GPU_ADAPTER_INDEX=" + std::to_string(override_index) +
				              " does not exist (EnumAdapters1 DXGI_ERROR_NOT_FOUND)";
				return;
			}
			if (FAILED(hr)) {
				init_error = "SSLM_GPU_ADAPTER_INDEX=" + std::to_string(override_index) +
				              ": EnumAdapters1 failed";
				return;
			}
			DXGI_ADAPTER_DESC1 d;
			a->GetDesc1(&d);
			if (d.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) {
				init_error = "SSLM_GPU_ADAPTER_INDEX=" + std::to_string(override_index) +
				              " is a software adapter, not a hardware one";
				return;
			}
			if (FAILED(D3D12CreateDevice(a.Get(), D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&dev)))) {
				init_error = "SSLM_GPU_ADAPTER_INDEX=" + std::to_string(override_index) +
				              ": D3D12CreateDevice failed";
				return;
			}
			adapter = a;
		} else {
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
		}
		if (!dev) {
			if (init_error.empty()) init_error = "no D3D12 hardware compute adapter found";
			return;
		}
		if (override_requested) {
			// Only when SSLM_GPU_ADAPTER_INDEX was set -- see the Init() header comment (S2).
			DXGI_ADAPTER_DESC1 d;
			adapter->GetDesc1(&d);
			std::wprintf(L"# adapter: %s\n", d.Description);
			std::fflush(stdout);
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

		D3D12_QUERY_HEAP_DESC qhd{};
		qhd.Type = D3D12_QUERY_HEAP_TYPE_TIMESTAMP;
		qhd.Count = kMaxTimestampSlots;
		SSLM_GPU_HR(dev->CreateQueryHeap(&qhd, IID_PPV_ARGS(&timestamp_heap)));
		timestamp_readback = MakeBuffer(kMaxTimestampSlots * sizeof(UINT64), D3D12_HEAP_TYPE_READBACK,
		                                 D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST);
		if (FAILED(queue->GetTimestampFrequency(&timestamp_frequency))) timestamp_frequency = 0;

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

	// T-2032/T-2035: the composed pipeline's own root signature -- one
	// 10-value 32-bit-constants block (b0: layer_index, hidden_size,
	// head_dim, num_kv_heads, context_cap, position, num_attention_heads,
	// width, intermediate_size, num_hidden_layers [T-2045 (C3): the 10th
	// value, `commit_site.hlsl`'s own dispatch alone reads it -- corrected
	// 2026-08-14, T-2055, Claude/Poirot/db73b22-gpu-serial-port-final-
	// confirmation-review.md, P5/O7, superseding this paragraph's own
	// "9-value" claim, false against `Num32BitValues = 10` and
	// `bind_and_dispatch`'s own 10-element `consts[10]` array below since
	// T-2045 landed the 10th value and never updated this count]), eight
	// root SRVs (t0 LayerWeights, t1 Layout, t2 RopeInfo, t3 ModelConstants
	// [the i-exp derivation's own three compile-time constants, T-2035], t4
	// SiluLut, t5 RopeCosTable, t6 RopeSinTable [T-2035: RoPE's own real
	// rotation data, not just presence/extent], t7 ScratchLayout [T-2039: the
	// per-call, per-real-dims LayerScratch/WorkScratch byte offsets, computed
	// once host-side -- never re-derived shader-side, matching Layout's own
	// established host-computes/shader-reads discipline]), four root UAVs (u0
	// SeqState, u1 LayerScratch, u2 KvCache, u3 WorkScratch [corrected
	// 2026-08-14, T-2055, P5: this bracket used to describe WorkScratch as
	// carrying "attention-scores" -- false since T-2049's own N6 retired
	// that region (superslm_gpu.cpp's own WorkScratch layout comment, and
	// three sibling instances elsewhere in this tree, were corrected the
	// same round; this was the fourth live instance, missed because that
	// round's own sweep re-ran the PRIOR finding's phrase family rather than
	// grepping fresh for this one -- StandardsDocument.md §7/§6.6). WorkScratch
	// is the transient, per-call-sized WIDE_A/WIDE_B wide-row scratch every
	// real production-geometry GEMM-funneled site now streams through
	// instead of a fixed-capacity local array; the attention score/probs row
	// lives in LayerScratch's own persistent `scores` field instead
	// (`ScratchLayout` index 25, `superslm_gpu.cpp`'s `ComputeScratchLayout`)]).
	// Shared by every shader the composed dispatch
	// issues -- a dispatch that does not use one of these bindings simply
	// never reads it; D3D12 does not require a PSO to consume every root
	// parameter its shared signature declares.
	// T-2049 (N4, Claude/Poirot/34ef30f-gpu-serial-port-confirmation-review.md):
	// the single source of truth for how many resource bindings (SRV+UAV,
	// excluding the root-constants block) the composed pipeline's own root
	// signature carries -- 8 SRVs + 4 UAVs today. `kGpuResidencyAllocationCallCount`
	// (superslm_gpu.cpp) reads THIS constant rather than repeating the number,
	// so B12's own injection sweep width cannot silently drift from what the
	// root signature actually binds: the confirmation review found the prior
	// `12` presented as "§5.1's own real read+write resource-table count" when
	// it was actually this file's own current substrate, counted by hand and
	// duplicated into a second file -- a value that could disagree with this
	// one the moment either changed. §5.1's own RATIFIED architecture (one SRV
	// descriptor table + one UAV descriptor table, D-SLM3001/D-SLM2929) is a
	// DIFFERENT, smaller binding count than this one; this constant is
	// honestly the CURRENT substrate's own count, not a §5.1 count, and is
	// named as such everywhere it is used -- it must be re-derived, not
	// reused, when S2's own binding-architecture migration lands.
	// T-2113 (B6b, design Sec8): 12 -> 14 (10 SRV + 4 UAV). The two new SRVs (t8 LoraAB, t9
	// Fold) are the adapter-delta dispatch's own read-only inputs -- design Sec8's own
	// "addressed via a root-constant/root-descriptor the recording code sets per sequence,
	// per call." Every existing composed-pipeline shader (attn_norm_site through
	// commit_site, every *_gemm_site/*_site) declares no t8/t9 register and is unaffected --
	// D3D12 does not require a PSO's shader to reference every root parameter its shared
	// root signature carries, the same "subset usage" already proven by every GEMM-only
	// dispatch shader that skips t3-t6 (ModelConstants/SiluLut/RopeCosTable/RopeSinTable)
	// while sharing this identical signature.
	static constexpr UINT kComposedResourceBindingCount = 14;  // 10 SRV + 4 UAV

	ComPtr<ID3D12RootSignature> MakeRootSigComposed() {
		D3D12_ROOT_PARAMETER ps[1 + kComposedResourceBindingCount]{};
		ps[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
		ps[0].Constants.Num32BitValues = 27;  // 10th: num_hidden_layers (commit_site.hlsl only);
		                                      // 11th: T-2113 (B4) GEMM lanes (the six
		                                      // <site>_gemm_site.hlsl dispatches only);
		                                      // 12th-19th/20th-27th: T-2113 (B10 lever 1b) two
		                                      // adapter-delta slots (rank, a_offset, b_offset,
		                                      // fold_offset, adapter_u_off, in_base, wide_base,
		                                      // stage1_lanes each) -- the fused tail dispatch's own
		                                      // extension (site_common.hlsli's
		                                      // ApplyFusedAdapterDeltaGpu), set only by
		                                      // `bind_and_dispatch_tail` (superslm_gpu.cpp); every
		                                      // other `bind_and_dispatch` call still writes only the
		                                      // first 11 and leaves these unread, since no non-tail
		                                      // shader declares a cbuffer field past position 10.
		                                      // T-2113 (B10 lever 1b): 8th field per slot
		                                      // (stage1_lanes) added when lever 1's own 7-field
		                                      // slot (25 total) grew to 8 (27 total) -- the fused
		                                      // stage-1 reduction's own group-cooperative lane
		                                      // count, host-computed per slot from that slot's real
		                                      // rank (see `bind_and_dispatch_tail`).
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
		ps[8].ParameterType = D3D12_ROOT_PARAMETER_TYPE_SRV;
		ps[8].Descriptor.ShaderRegister = 7;  // t7 ScratchLayout
		ps[9].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;
		ps[9].Descriptor.ShaderRegister = 0;  // u0 SeqState
		ps[10].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;
		ps[10].Descriptor.ShaderRegister = 1;  // u1 LayerScratch
		ps[11].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;
		ps[11].Descriptor.ShaderRegister = 2;  // u2 KvCache
		ps[12].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;
		ps[12].Descriptor.ShaderRegister = 3;  // u3 WorkScratch
		// T-2113 (B6b, design Sec8): the adapter-delta dispatch's own two read-only inputs.
		ps[13].ParameterType = D3D12_ROOT_PARAMETER_TYPE_SRV;
		ps[13].Descriptor.ShaderRegister = 8;  // t8 LoraAB (adapter lora_A+lora_B bytes)
		ps[14].ParameterType = D3D12_ROOT_PARAMETER_TYPE_SRV;
		ps[14].Descriptor.ShaderRegister = 9;  // t9 Fold (adapter DeltaFoldScales+UFoldScales)

		D3D12_ROOT_SIGNATURE_DESC rs{};
		rs.NumParameters = 1 + kComposedResourceBindingCount;
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

// Locates `<exe_dir>/shaders/<name>.cso`. build.bat places its compiled
// shaders next to its own built test binary (out\superslm_tests.exe).
// CMakeLists.txt (T-2115, D-SLM3432) also compiles these shaders, behind
// SUPERSLM_BUILD_GPU, but builds no GPU-linked executable of its own to sit
// beside them -- see superslm_gpu.cpp's own ShaderPath() definition for the
// current, full account of where each build puts them.
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

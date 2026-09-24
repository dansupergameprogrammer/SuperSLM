// TE-425 -- the SuperSLM 1.8.0 GPU allocation-status red suite, every GPU cell as one mode of one
// binary. See te425_harness.h for the contract every cell asserts and the three fault sources.
//
// Usage: te425_cells.exe <mode> [--key=value ...]
// Artifacts:  --qwen3=PATH  qwen3-embedding-0.6b-1p5.sslm (SHA-256 0be28bf4...8f99fbf, the t2791 artifact)
//             --r15=PATH    qwen2.5-1.5b-instruct.sslm (the t2956 R15 artifact; device-resident head)
//             --adapter=PATH  qwen2.5-1.5b-shopkeeper-lora-v2-t2102-runtime.sslm
//             --g5=PATH     t2132_g5_fixture_1p5b.sslm (SHA-256 078df885...a019, schema-bearing)
// Exit status: 0 every leg conforms to plan Sec3.3; 1 at least one leg does not (RED);
//              3 setup failed; 4 a leg could not be driven (a hook never fired, a selector did not
//              resolve) -- an instrument that cannot decide never reads as a verdict.
// Every leg prints one line starting with its cell id; every mode ends with
//   SUMMARY <cell> ...   and   VERDICT <cell> GREEN|RED|INVALID
#include "te425_harness.h"

#include <functional>
#include <map>

#include <psapi.h>

using namespace te425;

// =============================================================================================
// Fault source 1: the replaced operator new family.
// =============================================================================================
static void* Te425Alloc(size_t n, size_t align, bool aligned) {
	if (g_new.counting.load(std::memory_order_relaxed)) {
		const uint32_t c = g_new.count.fetch_add(1) + 1;
		if (aligned) g_new.aligned_seen.fetch_add(1);
		if (g_new.label.load(std::memory_order_relaxed) && c < kMaxSites) {
			g_labels[c] = SiteLabel{static_cast<uint16_t>(g_list_open.load() ? g_window.load() : 0),
			                        static_cast<uint8_t>(g_last_event.load()), static_cast<uint8_t>(g_phase.load()),
			                        static_cast<uint8_t>(aligned ? 1 : 0)};
		}
		const uint32_t f = g_new.fail_at.load();
		if (f != 0 && c == f) {
			g_new.fail_at.store(0);
			g_new.fired.store(c);
			switch (static_cast<NewFault>(g_new.kind.load())) {
				case NewFault::LengthError: throw std::length_error("te425 injected length_error");
				case NewFault::Foreign: throw ForeignFault{c};
				case NewFault::RuntimeError: throw std::runtime_error("te425 injected runtime_error");
				default: throw std::bad_alloc();
			}
		}
	}
	void* p = aligned ? _aligned_malloc(n ? n : 1, align) : std::malloc(n ? n : 1);
	if (!p) throw std::bad_alloc();
	return p;
}
void* operator new(size_t n) { return Te425Alloc(n, 0, false); }
void* operator new[](size_t n) { return Te425Alloc(n, 0, false); }
void* operator new(size_t n, const std::nothrow_t&) noexcept {
	try {
		return Te425Alloc(n, 0, false);
	} catch (...) {
		return nullptr;
	}
}
void* operator new[](size_t n, const std::nothrow_t&) noexcept {
	try {
		return Te425Alloc(n, 0, false);
	} catch (...) {
		return nullptr;
	}
}
void* operator new(size_t n, std::align_val_t a) { return Te425Alloc(n, static_cast<size_t>(a), true); }
void* operator new[](size_t n, std::align_val_t a) { return Te425Alloc(n, static_cast<size_t>(a), true); }
void* operator new(size_t n, std::align_val_t a, const std::nothrow_t&) noexcept {
	try {
		return Te425Alloc(n, static_cast<size_t>(a), true);
	} catch (...) {
		return nullptr;
	}
}
void* operator new[](size_t n, std::align_val_t a, const std::nothrow_t&) noexcept {
	try {
		return Te425Alloc(n, static_cast<size_t>(a), true);
	} catch (...) {
		return nullptr;
	}
}
void operator delete(void* p) noexcept { std::free(p); }
void operator delete(void* p, size_t) noexcept { std::free(p); }
void operator delete[](void* p) noexcept { std::free(p); }
void operator delete[](void* p, size_t) noexcept { std::free(p); }
void operator delete(void* p, const std::nothrow_t&) noexcept { std::free(p); }
void operator delete[](void* p, const std::nothrow_t&) noexcept { std::free(p); }
void operator delete(void* p, std::align_val_t) noexcept { _aligned_free(p); }
void operator delete(void* p, size_t, std::align_val_t) noexcept { _aligned_free(p); }
void operator delete[](void* p, std::align_val_t) noexcept { _aligned_free(p); }
void operator delete[](void* p, size_t, std::align_val_t) noexcept { _aligned_free(p); }
void operator delete(void* p, std::align_val_t, const std::nothrow_t&) noexcept { _aligned_free(p); }
void operator delete[](void* p, std::align_val_t, const std::nothrow_t&) noexcept { _aligned_free(p); }

// =============================================================================================
// Fault source 3: D3D12 vtable hooks.
// =============================================================================================
namespace {
using PFN_CCR = HRESULT(STDMETHODCALLTYPE*)(ID3D12Device*, const D3D12_HEAP_PROPERTIES*, D3D12_HEAP_FLAGS,
                                            const D3D12_RESOURCE_DESC*, D3D12_RESOURCE_STATES,
                                            const D3D12_CLEAR_VALUE*, REFIID, void**);
using PFN_CLOSE = HRESULT(STDMETHODCALLTYPE*)(ID3D12GraphicsCommandList*);
using PFN_LRESET = HRESULT(STDMETHODCALLTYPE*)(ID3D12GraphicsCommandList*, ID3D12CommandAllocator*,
                                               ID3D12PipelineState*);
using PFN_ARESET = HRESULT(STDMETHODCALLTYPE*)(ID3D12CommandAllocator*);
using PFN_MAP = HRESULT(STDMETHODCALLTYPE*)(ID3D12Resource*, UINT, const D3D12_RANGE*, void**);
using PFN_PSO = HRESULT(STDMETHODCALLTYPE*)(ID3D12Device*, const D3D12_COMPUTE_PIPELINE_STATE_DESC*, REFIID,
                                            void**);
using PFN_ROOTSIG = HRESULT(STDMETHODCALLTYPE*)(ID3D12Device*, UINT, const void*, SIZE_T, REFIID, void**);
using PFN_SIGNAL = HRESULT(STDMETHODCALLTYPE*)(ID3D12CommandQueue*, ID3D12Fence*, UINT64);
using PFN_SETEVENT = HRESULT(STDMETHODCALLTYPE*)(ID3D12Fence*, UINT64, HANDLE);
using PFN_CQUEUE = HRESULT(STDMETHODCALLTYPE*)(ID3D12Device*, const D3D12_COMMAND_QUEUE_DESC*, REFIID, void**);
using PFN_CALLOC = HRESULT(STDMETHODCALLTYPE*)(ID3D12Device*, D3D12_COMMAND_LIST_TYPE, REFIID, void**);
using PFN_CLIST = HRESULT(STDMETHODCALLTYPE*)(ID3D12Device*, UINT, D3D12_COMMAND_LIST_TYPE,
                                              ID3D12CommandAllocator*, ID3D12PipelineState*, REFIID, void**);
using PFN_CFENCE = HRESULT(STDMETHODCALLTYPE*)(ID3D12Device*, UINT64, D3D12_FENCE_FLAGS, REFIID, void**);
using PFN_CQHEAP = HRESULT(STDMETHODCALLTYPE*)(ID3D12Device*, const D3D12_QUERY_HEAP_DESC*, REFIID, void**);

PFN_CCR o_ccr;
PFN_CLOSE o_close;
PFN_LRESET o_lreset;
PFN_ARESET o_areset;
PFN_MAP o_map;
PFN_PSO o_pso;
PFN_ROOTSIG o_rootsig;
PFN_SIGNAL o_signal;
PFN_SETEVENT o_setevent;
PFN_CQUEUE o_cqueue;
PFN_CALLOC o_calloc;
PFN_CLIST o_clist;
PFN_CFENCE o_cfence;
PFN_CQHEAP o_cqheap;

inline HRESULT ArmedHr() { return static_cast<HRESULT>(g_arm.hr.load()); }
inline bool CallThenFail() { return g_arm.behavior.load() == HB_CALL_THEN_FAIL; }
template <typename T>
inline void NullOut(T** p) {
	if (p) *p = nullptr;
}
inline void ReleaseOut(void** p) {
	if (p && *p) {
		static_cast<IUnknown*>(*p)->Release();
		*p = nullptr;
	}
}

HRESULT STDMETHODCALLTYPE H_ccr(ID3D12Device* self, const D3D12_HEAP_PROPERTIES* hp, D3D12_HEAP_FLAGS f,
                               const D3D12_RESOURCE_DESC* rd, D3D12_RESOURCE_STATES s, const D3D12_CLEAR_VALUE* cv,
                               REFIID iid, void** out) {
	const int heap = hp ? static_cast<int>(hp->Type) : 0;
	const uint64_t w = rd ? rd->Width : 0;
	const bool fault = HookEnter(M_CCR, heap, w);
	g_last_event.store(M_CCR);
	if (fault) {
		if (CallThenFail()) {
			(void)o_ccr(self, hp, f, rd, s, cv, iid, out);
			ReleaseOut(out);
		}
		NullOut(out);
		return ArmedHr();
	}
	return o_ccr(self, hp, f, rd, s, cv, iid, out);
}
HRESULT STDMETHODCALLTYPE H_close(ID3D12GraphicsCommandList* self) {
	const bool fault = HookEnter(M_CLOSE);
	g_last_event.store(M_CLOSE);
	if (fault) {
		if (CallThenFail()) {
			if (SUCCEEDED(o_close(self))) g_list_open.store(0);
		}
		return ArmedHr();  // HB_FAIL_NO_CALL: the list stays recording
	}
	const HRESULT r = o_close(self);
	if (SUCCEEDED(r)) g_list_open.store(0);
	return r;
}
HRESULT STDMETHODCALLTYPE H_lreset(ID3D12GraphicsCommandList* self, ID3D12CommandAllocator* a,
                                   ID3D12PipelineState* p) {
	const bool fault = HookEnter(M_LRESET);
	g_last_event.store(M_LRESET);
	if (fault) return ArmedHr();  // the list is not opened
	const HRESULT r = o_lreset(self, a, p);
	if (SUCCEEDED(r)) {
		g_list_open.store(1);
		g_window.fetch_add(1);
	}
	return r;
}
HRESULT STDMETHODCALLTYPE H_areset(ID3D12CommandAllocator* self) {
	const bool fault = HookEnter(M_ARESET);
	g_last_event.store(M_ARESET);
	if (fault) return ArmedHr();
	return o_areset(self);
}
HRESULT STDMETHODCALLTYPE H_map(ID3D12Resource* self, UINT sub, const D3D12_RANGE* range, void** data) {
	const D3D12_RESOURCE_DESC d = self->GetDesc();
	const bool fault = HookEnter(M_MAP, 0, d.Width);
	g_last_event.store(M_MAP);
	if (fault) {
		NullOut(data);
		return ArmedHr();
	}
	return o_map(self, sub, range, data);
}
HRESULT STDMETHODCALLTYPE H_pso(ID3D12Device* self, const D3D12_COMPUTE_PIPELINE_STATE_DESC* d, REFIID iid,
                               void** out) {
	const bool fault = HookEnter(M_PSO);
	g_last_event.store(M_PSO);
	if (fault) {
		NullOut(out);
		return ArmedHr();
	}
	return o_pso(self, d, iid, out);
}
HRESULT STDMETHODCALLTYPE H_rootsig(ID3D12Device* self, UINT node, const void* blob, SIZE_T n, REFIID iid,
                                   void** out) {
	const bool fault = HookEnter(M_ROOTSIG);
	g_last_event.store(M_ROOTSIG);
	if (fault) {
		NullOut(out);
		return ArmedHr();
	}
	return o_rootsig(self, node, blob, n, iid, out);
}
HRESULT STDMETHODCALLTYPE H_signal(ID3D12CommandQueue* self, ID3D12Fence* fence, UINT64 v) {
	const bool fault = HookEnter(M_SIGNAL);
	g_last_event.store(M_SIGNAL);
	if (fault) return ArmedHr();  // the fence is never signaled at v
	return o_signal(self, fence, v);
}
HRESULT STDMETHODCALLTYPE H_setevent(ID3D12Fence* self, UINT64 v, HANDLE e) {
	const bool fault = HookEnter(M_SETEVENT);
	g_last_event.store(M_SETEVENT);
	if (fault) return ArmedHr();
	return o_setevent(self, v, e);
}
HRESULT STDMETHODCALLTYPE H_cqueue(ID3D12Device* self, const D3D12_COMMAND_QUEUE_DESC* d, REFIID iid, void** out) {
	const bool fault = HookEnter(M_CQUEUE);
	g_last_event.store(M_CQUEUE);
	if (fault) {
		NullOut(out);
		return ArmedHr();
	}
	return o_cqueue(self, d, iid, out);
}
HRESULT STDMETHODCALLTYPE H_calloc(ID3D12Device* self, D3D12_COMMAND_LIST_TYPE t, REFIID iid, void** out) {
	const bool fault = HookEnter(M_CALLOC);
	g_last_event.store(M_CALLOC);
	if (fault) {
		NullOut(out);
		return ArmedHr();
	}
	return o_calloc(self, t, iid, out);
}
HRESULT STDMETHODCALLTYPE H_clist(ID3D12Device* self, UINT node, D3D12_COMMAND_LIST_TYPE t,
                                  ID3D12CommandAllocator* a, ID3D12PipelineState* p, REFIID iid, void** out) {
	const bool fault = HookEnter(M_CLIST);
	g_last_event.store(M_CLIST);
	if (fault) {
		NullOut(out);
		return ArmedHr();
	}
	return o_clist(self, node, t, a, p, iid, out);
}
HRESULT STDMETHODCALLTYPE H_cfence(ID3D12Device* self, UINT64 v, D3D12_FENCE_FLAGS f, REFIID iid, void** out) {
	const bool fault = HookEnter(M_CFENCE);
	g_last_event.store(M_CFENCE);
	if (fault) {
		NullOut(out);
		return ArmedHr();
	}
	return o_cfence(self, v, f, iid, out);
}
HRESULT STDMETHODCALLTYPE H_cqheap(ID3D12Device* self, const D3D12_QUERY_HEAP_DESC* d, REFIID iid, void** out) {
	const bool fault = HookEnter(M_CQHEAP);
	g_last_event.store(M_CQHEAP);
	if (fault) {
		NullOut(out);
		return ArmedHr();
	}
	return o_cqheap(self, d, iid, out);
}

template <typename Fn>
bool Patch(void* object, int slot, void* hook, Fn* original) {
	void** vt = *reinterpret_cast<void***>(object);
	DWORD old = 0;
	if (!VirtualProtect(&vt[slot], sizeof(void*), PAGE_EXECUTE_READWRITE, &old)) return false;
	if (vt[slot] != hook) {
		*original = reinterpret_cast<Fn>(vt[slot]);
		vt[slot] = hook;
	}
	VirtualProtect(&vt[slot], sizeof(void*), old, &old);
	return true;
}

// The probe objects whose vtables were patched; kept alive for the process.
ComPtr<ID3D12Device> p_dev;
ComPtr<ID3D12CommandQueue> p_queue;
ComPtr<ID3D12CommandAllocator> p_alloc;
ComPtr<ID3D12GraphicsCommandList> p_list;
ComPtr<ID3D12Fence> p_fence;
ComPtr<ID3D12Resource> p_buf;
}  // namespace

// SDK slot numbers (d3d12.h's C vtable order; ID3D12Device's two #if-duplicated entries at slots
// 25/26 counted once): Device 8 CreateCommandQueue, 9 CreateCommandAllocator, 11 CreateComputePipelineState,
// 12 CreateCommandList, 16 CreateRootSignature, 27 CreateCommittedResource, 36 CreateFence,
// 39 CreateQueryHeap; GraphicsCommandList 9 Close, 10 Reset; CommandAllocator 8 Reset; Resource 8 Map;
// CommandQueue 14 Signal; Fence 9 SetEventOnCompletion.
bool te425::InstallHooks(std::string* why) {
	if (p_dev) return true;
	if (FAILED(D3D12CreateDevice(nullptr, D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&p_dev)))) {
		*why = "D3D12CreateDevice failed";
		return false;
	}
	D3D12_COMMAND_QUEUE_DESC qd{};
	qd.Type = D3D12_COMMAND_LIST_TYPE_COMPUTE;
	D3D12_HEAP_PROPERTIES hp{};
	hp.Type = D3D12_HEAP_TYPE_UPLOAD;
	D3D12_RESOURCE_DESC rd{};
	rd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
	rd.Width = 256;
	rd.Height = 1;
	rd.DepthOrArraySize = 1;
	rd.MipLevels = 1;
	rd.SampleDesc.Count = 1;
	rd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
	if (FAILED(p_dev->CreateCommandQueue(&qd, IID_PPV_ARGS(&p_queue))) ||
	    FAILED(p_dev->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COMPUTE, IID_PPV_ARGS(&p_alloc))) ||
	    FAILED(p_dev->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_COMPUTE, p_alloc.Get(), nullptr,
	                                    IID_PPV_ARGS(&p_list))) ||
	    FAILED(p_dev->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&p_fence))) ||
	    FAILED(p_dev->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &rd, D3D12_RESOURCE_STATE_GENERIC_READ,
	                                          nullptr, IID_PPV_ARGS(&p_buf)))) {
		*why = "probe object creation failed";
		return false;
	}
	bool ok = true;
	ok = ok && Patch(p_dev.Get(), 8, reinterpret_cast<void*>(&H_cqueue), &o_cqueue);
	ok = ok && Patch(p_dev.Get(), 9, reinterpret_cast<void*>(&H_calloc), &o_calloc);
	ok = ok && Patch(p_dev.Get(), 11, reinterpret_cast<void*>(&H_pso), &o_pso);
	ok = ok && Patch(p_dev.Get(), 12, reinterpret_cast<void*>(&H_clist), &o_clist);
	ok = ok && Patch(p_dev.Get(), 16, reinterpret_cast<void*>(&H_rootsig), &o_rootsig);
	ok = ok && Patch(p_dev.Get(), 27, reinterpret_cast<void*>(&H_ccr), &o_ccr);
	ok = ok && Patch(p_dev.Get(), 36, reinterpret_cast<void*>(&H_cfence), &o_cfence);
	ok = ok && Patch(p_dev.Get(), 39, reinterpret_cast<void*>(&H_cqheap), &o_cqheap);
	ok = ok && Patch(p_list.Get(), 9, reinterpret_cast<void*>(&H_close), &o_close);
	ok = ok && Patch(p_list.Get(), 10, reinterpret_cast<void*>(&H_lreset), &o_lreset);
	ok = ok && Patch(p_alloc.Get(), 8, reinterpret_cast<void*>(&H_areset), &o_areset);
	ok = ok && Patch(p_buf.Get(), 8, reinterpret_cast<void*>(&H_map), &o_map);
	ok = ok && Patch(p_queue.Get(), 14, reinterpret_cast<void*>(&H_signal), &o_signal);
	ok = ok && Patch(p_fence.Get(), 9, reinterpret_cast<void*>(&H_setevent), &o_setevent);
	if (!ok) {
		*why = "VirtualProtect failed on a vtable slot";
		return false;
	}
	// Self-check: every hook this harness can reach without the engine must fire on a probe call. A
	// slot number off by one would call a different method through the hook and fail here (or be
	// caught by a leg's own "the hook never fired" refusal for the two pipeline-creation methods,
	// which need shader bytecode).
	uint32_t before[M_COUNT];
	for (int m = 0; m < M_COUNT; ++m) before[m] = g_calls[m].load();
	HookDisarm();
	void* mapped = nullptr;
	D3D12_RANGE none{0, 0};
	HANDLE ev = CreateEventW(nullptr, FALSE, FALSE, nullptr);
	ComPtr<ID3D12CommandQueue> q2;
	ComPtr<ID3D12CommandAllocator> a2;
	ComPtr<ID3D12GraphicsCommandList> l2;
	ComPtr<ID3D12Fence> f2;
	ComPtr<ID3D12QueryHeap> qh;
	ComPtr<ID3D12Resource> b2;
	D3D12_QUERY_HEAP_DESC qhd{};
	qhd.Type = D3D12_QUERY_HEAP_TYPE_TIMESTAMP;
	qhd.Count = 2;
	bool calls_ok = SUCCEEDED(p_list->Close()) && SUCCEEDED(p_alloc->Reset()) &&
	                SUCCEEDED(p_list->Reset(p_alloc.Get(), nullptr)) && SUCCEEDED(p_list->Close()) &&
	                SUCCEEDED(p_buf->Map(0, &none, &mapped)) && SUCCEEDED(p_queue->Signal(p_fence.Get(), 1)) &&
	                SUCCEEDED(p_fence->SetEventOnCompletion(1, ev)) &&
	                SUCCEEDED(p_dev->CreateCommandQueue(&qd, IID_PPV_ARGS(&q2))) &&
	                SUCCEEDED(p_dev->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COMPUTE, IID_PPV_ARGS(&a2))) &&
	                SUCCEEDED(p_dev->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_COMPUTE, a2.Get(), nullptr,
	                                                   IID_PPV_ARGS(&l2))) &&
	                SUCCEEDED(p_dev->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&f2))) &&
	                SUCCEEDED(p_dev->CreateQueryHeap(&qhd, IID_PPV_ARGS(&qh))) &&
	                SUCCEEDED(p_dev->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &rd,
	                                                         D3D12_RESOURCE_STATE_GENERIC_READ, nullptr,
	                                                         IID_PPV_ARGS(&b2)));
	if (mapped) p_buf->Unmap(0, nullptr);
	WaitForSingleObject(ev, 5000);
	CloseHandle(ev);
	// CreateRootSignature through a real, empty root signature.
	D3D12_ROOT_SIGNATURE_DESC rsd{};
	ComPtr<ID3DBlob> blob, err;
	ComPtr<ID3D12RootSignature> rs;
	calls_ok = calls_ok && SUCCEEDED(D3D12SerializeRootSignature(&rsd, D3D_ROOT_SIGNATURE_VERSION_1, &blob, &err)) &&
	           SUCCEEDED(p_dev->CreateRootSignature(0, blob->GetBufferPointer(), blob->GetBufferSize(),
	                                                IID_PPV_ARGS(&rs)));
	const int must_fire[] = {M_CLOSE, M_ARESET, M_LRESET, M_MAP, M_SIGNAL, M_SETEVENT, M_CQUEUE,
	                         M_CALLOC, M_CLIST, M_CFENCE, M_CQHEAP, M_CCR, M_ROOTSIG};
	for (int m : must_fire) {
		if (g_calls[m].load() == before[m]) {
			*why = std::string("hook self-check: ") + MethodName(m) + " did not fire on a probe call";
			return false;
		}
	}
	if (!calls_ok) {
		*why = "hook self-check: a probe call failed";
		return false;
	}
	g_list_open.store(0);
	g_window.store(0);
	return true;
}

// =============================================================================================
// Options, output, verdicts
// =============================================================================================
namespace {
std::map<std::string, std::string> g_opt;
std::string Opt(const char* k, const char* d = "") {
	auto it = g_opt.find(k);
	return it == g_opt.end() ? std::string(d) : it->second;
}
long OptInt(const char* k, long d) {
	auto it = g_opt.find(k);
	return it == g_opt.end() ? d : std::strtol(it->second.c_str(), nullptr, 10);
}
bool Has(const char* k) { return g_opt.count(k) != 0; }

struct Verdict {
	const char* cell;
	uint32_t legs = 0, pass = 0, fail = 0, invalid = 0;
	explicit Verdict(const char* c) : cell(c) {}
	void Leg(bool conforms) {
		++legs;
		if (conforms) ++pass;
		else ++fail;
	}
	void Invalid() {
		++legs;
		++invalid;
	}
	int Finish() const {
		std::printf("SUMMARY %s legs=%u conform=%u nonconform=%u invalid=%u\n", cell, legs, pass, fail, invalid);
		const char* v = invalid ? "INVALID" : fail ? "RED" : "GREEN";
		std::printf("VERDICT %s %s\n", cell, v);
		std::fflush(stdout);
		return invalid ? 4 : fail ? 1 : 0;
	}
};
int SetupFail(const char* cell, const char* what) {
	std::printf("SETUP-FAIL %s: %s\nVERDICT %s INVALID\n", cell, what, cell);
	std::fflush(stdout);
	return 3;
}
bool Hooks(const char* cell) {
	std::string why;
	if (!InstallHooks(&why)) {
		std::printf("SETUP-FAIL %s: %s\n", cell, why.c_str());
		return false;
	}
	return true;
}

// A route's reference: the frame of U3's loop, and (for decode routes) a token and the saved state.
struct Ref {
	Frame frame;
	int32_t token = -1;
	std::vector<uint8_t> blob;
	bool Same(const Ref& o) const { return frame.Same(o.frame) && token == o.token && blob == o.blob; }
};
std::vector<uint8_t> SaveBlob(SslmGpuContext* ctx, SslmGpuSequenceHandle* s) {
	size_t n = 0;
	(void)sslm_gpu_seq_save(ctx, s, nullptr, &n);
	std::vector<uint8_t> b(n);
	if (!n || sslm_gpu_seq_save(ctx, s, b.data(), &n) != OK) b.clear();
	return b;
}
bool WriteFileBytes(const std::string& p, const void* d, size_t n) {
	FILE* f = nullptr;
	if (fopen_s(&f, p.c_str(), "wb") || !f) return false;
	const bool ok = fwrite(d, 1, n, f) == n;
	fclose(f);
	return ok;
}
// A frame serialized as codes | m | e.
std::vector<uint8_t> FrameBytes(const Frame& f) {
	std::vector<uint8_t> b(f.codes.size() + 16);
	std::memcpy(b.data(), f.codes.data(), f.codes.size());
	std::memcpy(b.data() + f.codes.size(), &f.m, 8);
	std::memcpy(b.data() + f.codes.size() + 8, &f.e, 8);
	return b;
}
bool FrameFromBytes(const std::vector<uint8_t>& b, uint32_t hidden, Frame* f) {
	if (b.size() != hidden + 16u) return false;
	f->codes.assign(b.begin(), b.begin() + hidden);
	std::memcpy(&f->m, b.data() + hidden, 8);
	std::memcpy(&f->e, b.data() + hidden + 8, 8);
	f->status = OK;
	return true;
}
}  // namespace

// =============================================================================================
// Fixtures
// =============================================================================================
namespace {

// Two contexts on one artifact, each with its own map and sequence: the second context is the
// "second context" of plan Sec3.5 R3/R9/R10 -- it shares nothing with the first but the process's
// submission device, which is what the stuck-list oracle reads.
struct Pair {
	Artifact art;
	Stack s1, s2;
	std::vector<int32_t> toks;
	Frame ref, f;
	int64_t cap = 64;
	bool Open(const std::string& path, int len, int64_t c = 64, uint32_t flags = 0, bool reference = true) {
		cap = c;
		if (!art.Load(path)) return false;
		if (!s1.Open(art, cap, flags) || !s2.Open(art, cap, flags)) {
			std::printf("SETUP stack open failed on %s\n", path.c_str());
			return false;
		}
		toks = PromptFor(art, len);
		ref.Size(s1.hidden);
		f.Size(s1.hidden);
		if (!reference) return true;
		LoopOut o = PromptLoop(s1.ctx, s1.seq, toks, &ref);
		if (o.First() != OK) {
			std::printf("SETUP reference loop on ctx1: %s\n", St(o.First()));
			return false;
		}
		o = PromptLoop(s2.ctx, s2.seq, toks, &f);
		if (o.First() != OK || !f.Same(ref)) {
			std::printf("SETUP reference loop on ctx2: %s equal=%d\n", St(o.First()), f.Same(ref) ? 1 : 0);
			return false;
		}
		return true;
	}
	// The stuck-list oracle: the second context's next loop is OK and byte-equal.
	bool After2(SslmGpuStatus* st) {
		LoopOut o = PromptLoop(s2.ctx, s2.seq, toks, &f);
		*st = o.First();
		return o.First() == OK && f.Same(ref);
	}
	bool After1(SslmGpuStatus* st) {
		LoopOut o = PromptLoop(s1.ctx, s1.seq, toks, &f);
		*st = o.First();
		return o.First() == OK && f.Same(ref);
	}
};

// Schema-content DFA chain from the start state (tests/t2791 fixture_common.h's technique: an
// independent parse of the artifact's SchemaMasks section).
struct Chain {
	superslm::SslmModelView view{};
	superslm::SchemaMasksTable table;
	const superslm::SchemaEntry* entry = nullptr;
	static uint32_t Le32(const uint8_t* p) {
		return static_cast<uint32_t>(p[0]) | (static_cast<uint32_t>(p[1]) << 8) |
		       (static_cast<uint32_t>(p[2]) << 16) | (static_cast<uint32_t>(p[3]) << 24);
	}
	bool Build(const std::vector<uint8_t>& bytes, const std::string& name) {
		std::string err;
		if (superslm::SslmModel::Load(bytes.data(), bytes.size(), view, &err) != superslm::SslmModelStatus::Ok)
			return false;
		const superslm::SslmSectionView* s = view.Section(superslm::SslmSectionType::SchemaMasks);
		if (!s) return false;
		if (!superslm::SchemaMasksTable::Parse(s->data, s->byte_size, view.config.vocab_size, table, &err))
			return false;
		entry = table.ByName(name);
		return entry != nullptr;
	}
	// The first legal token at `state`, or -1.
	int32_t FirstLegal(uint32_t state) const {
		const uint32_t b = Le32(entry->state_offsets_le + static_cast<size_t>(state) * 4);
		const uint32_t en = Le32(entry->state_offsets_le + static_cast<size_t>(state + 1) * 4);
		return b < en ? static_cast<int32_t>(Le32(entry->transitions_le + static_cast<size_t>(b) * 8)) : -1;
	}
	std::vector<int32_t> Walk(int n) const {
		std::vector<int32_t> c;
		uint32_t st = 0;
		for (int k = 0; k < n; ++k) {
			const int32_t t = FirstLegal(st);
			uint32_t nx = 0;
			if (t < 0 || !table.Transition(*entry, st, static_cast<uint32_t>(t), &nx)) break;
			c.push_back(t);
			st = nx;
		}
		return c;
	}
};
const char* kSchemaName = "shopkeeper_intent_extraction";

// U3's per-document loop over the schema twin: reset, bind, schema content, read.
struct SchemaOut {
	LoopOut loop;
	int32_t consumed = -1;
	int64_t committed = -1;
};
SchemaOut SchemaLoop(SslmGpuContext* ctx, SslmGpuSequenceHandle* seq, int32_t idx, const std::vector<int32_t>& chain,
                     Frame* f) {
	SchemaOut o;
	g_phase.store(PH_RESET);
	o.loop.reset = sslm_gpu_seq_reset(ctx, seq);
	if (o.loop.reset == OK) o.loop.reset = SslmGpuSeqSetSchemaForG5Bridge(ctx, seq, idx);
	g_phase.store(PH_PREFILL);
	if (o.loop.reset == OK) {
		o.loop.prefill = SslmGpuSeqPrefillSchemaContentForG5Bridge(ctx, seq, chain.data(),
		                                                           static_cast<int32_t>(chain.size()), 64, &o.consumed);
		o.committed = *SslmGpuSeqHandleContextLengthForBench(seq);
	}
	g_phase.store(PH_READ);
	std::fill(f->codes.begin(), f->codes.end(), static_cast<int8_t>(0x5A));
	if (o.loop.reset == OK) {
		o.loop.read = sslm_gpu_seq_read_prefill_final_hidden(ctx, seq, f->codes.data(), f->codes.size(), &f->required,
		                                                     &f->m, &f->e);
	}
	f->status = o.loop.read;
	g_phase.store(PH_NONE);
	return o;
}

// One decode step through the per-token primitives: embed, step (full depth), ready(block).
struct StepOut {
	SslmGpuStatus embed = OK, step = OK, ready_ret = OK, out_status = OK;
	SslmGpuStatus First() const {
		return embed != OK ? embed : step != OK ? step : ready_ret != OK ? ready_ret : out_status;
	}
};
constexpr int32_t kStepToken = 785;
StepOut DecodeStep(SslmGpuContext* ctx, SslmGpuSequenceHandle* seq) {
	StepOut o;
	g_phase.store(PH_STEP);
	o.embed = sslm_gpu_seq_embed_token(ctx, seq, kStepToken);
	if (o.embed == OK) o.step = sslm_decode_step_gpu(ctx, seq, nullptr, 0xFFFFFFFFu);
	g_phase.store(PH_READY);
	if (o.embed == OK && o.step == OK) {
		int32_t ready = 0;
		o.ready_ret = sslm_gpu_ready(ctx, seq, 1, &ready, &o.out_status);
	}
	g_phase.store(PH_NONE);
	return o;
}

}  // namespace

// =============================================================================================
// The generic single-site sweep (R2, R3, R4, R5, R10, R15, R1): for each site k of a fault source,
// a clean call before (must be clean), the armed call, and the after-calls.
// =============================================================================================
namespace {
enum class Src { Seam, New };
struct SweepHooks {
	std::function<bool()> pre;              // establish state and prove it clean; false = isolation broken
	std::function<void()> armed;            // the armed call; records its own outcome
	std::function<bool(std::string*)> judge;  // the armed call's own outcome against the contract
	std::function<bool(SslmGpuStatus*)> after2;  // the second context's next call
	std::function<bool(SslmGpuStatus*)> after1;  // the same handle's next call
};

// Counting pass: the number of sites the armed call reaches, and a label per site.
uint32_t CountSites(Src src, const std::function<void()>& call, std::vector<TraceEv>* ccr_trace) {
	if (src == Src::Seam) {
		TraceStart();
		SslmGpuAllocCounterResetForTest();
		call();
		const uint32_t k = SslmGpuAllocCountForTest();
		std::vector<TraceEv> t = TraceStop();
		if (ccr_trace) {
			ccr_trace->clear();
			for (const TraceEv& e : t)
				if (e.method == M_CCR) ccr_trace->push_back(e);
		}
		return k;
	}
	TraceStart();
	NewCountOnly(true);
	call();
	NewStop();
	TraceStop();
	return g_new.count.load();
}
const char* HeapName(int h) {
	switch (h) {
		case D3D12_HEAP_TYPE_DEFAULT: return "DEFAULT";
		case D3D12_HEAP_TYPE_UPLOAD: return "UPLOAD";
		case D3D12_HEAP_TYPE_READBACK: return "READBACK";
		default: return "?";
	}
}

struct SweepTally {
	// key: "<own status>|after2 <clean|dirty>" (and heap for the seam source)
	std::map<std::string, uint32_t> classes;
	uint32_t fired = 0, not_fired = 0, pre_bad = 0;
	void Print(const char* cell) const {
		std::printf("TALLY %s fired=%u not_fired=%u pre_not_clean=%u\n", cell, fired, not_fired, pre_bad);
		for (const auto& kv : classes) std::printf("TALLY %s %6u  %s\n", cell, kv.second, kv.first.c_str());
	}
};

// Runs sites [from, to] of `src` over `h`. `kind` is the operator-new fault kind; `hr` the seam's.
// Returns the verdict object's exit status.
int Sweep(const char* cell, Src src, NewFault kind, long hr, uint32_t K, const std::vector<TraceEv>& ccr,
          uint32_t from, uint32_t to, const std::vector<uint32_t>* only, SweepHooks& h, Verdict& v,
          SweepTally& tally) {
	std::vector<uint32_t> sites;
	if (only) {
		sites = *only;
	} else {
		for (uint32_t k = from; k <= to && k <= K; ++k) sites.push_back(k);
	}
	for (uint32_t k : sites) {
		const bool pre_ok = h.pre();
		if (!pre_ok) ++tally.pre_bad;
		bool fired = false;
		if (src == Src::Seam) {
			SslmGpuAllocCounterResetForTest();
			ArmGpuAllocFaultAtOccurrence(k, hr);
			h.armed();
			fired = SslmGpuAllocCountForTest() >= k;
			ArmGpuAllocFaultAtOccurrence(0, S_OK);
		} else {
			NewArm(k, kind);
			h.armed();
			NewStop();
			fired = g_new.fired.load() != 0;
		}
		std::string own;
		const bool own_ok = h.judge(&own);
		SslmGpuStatus s2 = OK, s1 = OK;
		const bool a2 = h.after2 ? h.after2(&s2) : true;
		const bool a1 = h.after1 ? h.after1(&s1) : true;
		std::string label;
		if (src == Src::Seam) {
			label = k <= ccr.size() ? std::string("heap=") + HeapName(ccr[k - 1].heap) + " bytes=" +
			                              std::to_string(ccr[k - 1].width) + " win=" + std::to_string(ccr[k - 1].window)
			                        : std::string("heap=?");
		} else if (k < kMaxSites) {
			const SiteLabel& L = g_labels[k];
			label = std::string("win=") + std::to_string(L.window) + " ev=" + MethodName(L.last_event) +
			        " ph=" + PhaseName(L.phase) + (L.aligned ? " aligned" : "");
		}
		if (!fired) {
			++tally.not_fired;
			std::printf("%s k=%u %s fired=0 own=%s after2=%s after1=%s -> NOT-FIRED\n", cell, k, label.c_str(),
			            own.c_str(), St(s2), St(s1));
			continue;
		}
		++tally.fired;
		const bool conforms = pre_ok && own_ok && a2 && a1;
		v.Leg(conforms);
		std::string key = own.substr(0, own.find(' ')) + " | after2 " + (a2 ? "clean" : std::string("dirty:") + St(s2));
		if (src == Src::Seam && k <= ccr.size())
			key = std::string(ccr[k - 1].heap == D3D12_HEAP_TYPE_DEFAULT ? "VRAM " : "HOST ") + key;
		if (src == Src::New && k < kMaxSites) key = std::string(g_labels[k].window ? "window " : "outside ") + key;
		tally.classes[key] += 1;
		std::printf("%s k=%u %s fired=1 pre=%d own=%s after2=%s%s after1=%s%s -> %s\n", cell, k, label.c_str(),
		            pre_ok ? 1 : 0, own.c_str(), St(s2), a2 ? "" : "(!)", St(s1), a1 ? "" : "(!)",
		            conforms ? "PASS" : "FAIL");
		std::fflush(stdout);
	}
	return 0;
}
}  // namespace

// =============================================================================================
// R2 / R3 / R10: U3's prompt loop, every D3D12 allocation (seam) or every operator-new site.
// =============================================================================================
namespace {
std::string LoopDesc(const LoopOut& o) {
	return std::string(St(o.First())) + "@" + std::to_string(o.FirstStage()) + " read=" + St(o.read);
}

int CellPromptSweep(const char* cell, Src src, NewFault kind) {
	Verdict v(cell);
	if (!Hooks(cell)) return SetupFail(cell, "hooks");
	const int len = static_cast<int>(OptInt("len", 40));
	Pair p;
	if (!p.Open(Opt("qwen3"), len)) return SetupFail(cell, "fixture");
	LoopOut own;
	SweepHooks h;
	h.pre = [&] {
		LoopOut o = PromptLoop(p.s1.ctx, p.s1.seq, p.toks, &p.f);
		return o.First() == OK && p.f.Same(p.ref);
	};
	h.armed = [&] { own = PromptLoop(p.s1.ctx, p.s1.seq, p.toks, &p.f); };
	h.judge = [&](std::string* d) {
		*d = LoopDesc(own);
		if (kind == NewFault::Foreign) {
			// Plan Sec3.5 R10: a foreign exception is not an allocation failure -- SSLM_DEVICE_LOST.
			return own.First() == DEVICE_LOST && (own.FirstStage() != 1 || own.read == HIDDEN_UNAVAILABLE);
		}
		return AllocLoopConforms(own);
	};
	h.after2 = [&](SslmGpuStatus* s) { return p.After2(s); };
	h.after1 = [&](SslmGpuStatus* s) { return p.After1(s); };
	std::vector<TraceEv> ccr;
	const uint32_t K = CountSites(src, [&] { own = PromptLoop(p.s1.ctx, p.s1.seq, p.toks, &p.f); }, &ccr);
	std::printf("%s len=%d C=%lld sites=%u (aligned new calls=%u) counting-pass=%s equal=%d\n", cell, len,
	            static_cast<long long>(p.cap), K, g_new.aligned_seen.load(), St(own.First()), p.f.Same(p.ref) ? 1 : 0);
	if (own.First() != OK) return SetupFail(cell, "counting pass not clean");
	if (src == Src::Seam && ccr.size() != K)
		std::printf("%s NOTE: seam count %u != CreateCommittedResource calls %zu\n", cell, K, ccr.size());
	std::vector<uint32_t> sel;
	const std::string select = Opt("select", "all");
	if (select == "window-edges") {
		// Plan Sec3.5 R10: at least one site per sub-chunk block. The first and last operator-new site
		// inside each recording window of the prefill.
		std::map<int, std::pair<uint32_t, uint32_t>> w;
		for (uint32_t k = 1; k <= K && k < kMaxSites; ++k) {
			if (g_labels[k].window == 0 || g_labels[k].phase != PH_PREFILL) continue;
			auto it = w.find(g_labels[k].window);
			if (it == w.end()) w[g_labels[k].window] = {k, k};
			else it->second.second = k;
		}
		for (const auto& kv : w) {
			sel.push_back(kv.second.first);
			if (kv.second.second != kv.second.first) sel.push_back(kv.second.second);
		}
		std::printf("%s window-edges: %zu windows, %zu sites\n", cell, w.size(), sel.size());
		// --pick=i: only the i-th of those sites, so each runs in a fresh process (a foreign exception can
		// leave the process's GPU path failing for every later call, which would contaminate the rest).
		if (Has("pick")) {
			const size_t i = static_cast<size_t>(OptInt("pick", 0));
			if (i >= sel.size()) return SetupFail(cell, "--pick beyond the window-edge sites");
			sel = {sel[i]};
		}
	} else if (Has("sites")) {
		// An explicit site list, "a,b,c-d".
		std::string s = Opt("sites");
		size_t i = 0;
		while (i < s.size()) {
			size_t j = s.find(',', i);
			if (j == std::string::npos) j = s.size();
			const std::string tok = s.substr(i, j - i);
			const size_t dash = tok.find('-');
			if (dash == std::string::npos) sel.push_back(static_cast<uint32_t>(std::stoul(tok)));
			else
				for (uint32_t k = static_cast<uint32_t>(std::stoul(tok.substr(0, dash)));
				     k <= static_cast<uint32_t>(std::stoul(tok.substr(dash + 1))); ++k)
					sel.push_back(k);
			i = j + 1;
		}
	}
	SweepTally t;
	const uint32_t from = static_cast<uint32_t>(OptInt("from", 1));
	const uint32_t to = static_cast<uint32_t>(OptInt("to", 0x7FFFFFFF));
	Sweep(cell, src, kind, E_OUTOFMEMORY, K, ccr, from, to, sel.empty() ? nullptr : &sel, h, v, t);
	t.Print(cell);
	return v.Finish();
}
}  // namespace

// =============================================================================================
// R4: the schema twin, both fault sources.
// =============================================================================================
namespace {
int CellSchemaSweep(const char* cell, Src src) {
	Verdict v(cell);
	if (!Hooks(cell)) return SetupFail(cell, "hooks");
	const int len = static_cast<int>(OptInt("len", 5));
	Pair p;
	if (!p.Open(Opt("g5"), 5, 64, 0, /*reference=*/false)) return SetupFail(cell, "fixture");
	if (!SslmGpuModelHasSchemasForG5Bridge(p.s1.model)) return SetupFail(cell, "--g5 carries no SchemaMasks");
	Chain ch;
	if (!ch.Build(p.art.bytes, kSchemaName)) return SetupFail(cell, "schema parse");
	const std::vector<int32_t> chain = ch.Walk(len);
	const int32_t idx = SslmGpuSchemaLookupForG5Bridge(p.s1.model, kSchemaName);
	const int32_t idx2 = SslmGpuSchemaLookupForG5Bridge(p.s2.model, kSchemaName);
	if (idx < 0 || idx2 != idx) return SetupFail(cell, "schema lookup");
	std::printf("%s schema chain requested=%d walked=%zu\n", cell, len, chain.size());
	if (chain.empty()) return SetupFail(cell, "empty DFA chain");
	SchemaOut r = SchemaLoop(p.s1.ctx, p.s1.seq, idx, chain, &p.ref);
	SchemaOut r2 = SchemaLoop(p.s2.ctx, p.s2.seq, idx, chain, &p.f);
	if (r.loop.First() != OK || r2.loop.First() != OK || !p.f.Same(p.ref) ||
	    r.consumed != static_cast<int32_t>(chain.size()))
		return SetupFail(cell, "schema reference loop");
	SchemaOut own;
	auto loop1 = [&] { return SchemaLoop(p.s1.ctx, p.s1.seq, idx, chain, &p.f); };
	SweepHooks h;
	h.pre = [&] {
		SchemaOut o = loop1();
		return o.loop.First() == OK && p.f.Same(p.ref);
	};
	h.armed = [&] { own = loop1(); };
	h.judge = [&](std::string* d) {
		*d = LoopDesc(own.loop) + " consumed=" + std::to_string(own.consumed) + " committed=" +
		     std::to_string(own.committed);
		// Plan Sec3.5 R4: the oracle adds that *consumed counts only committed tokens.
		const bool consumed_ok = own.loop.FirstStage() != 1 || own.consumed == own.committed;
		return AllocLoopConforms(own.loop) && consumed_ok;
	};
	h.after2 = [&](SslmGpuStatus* s) {
		SchemaOut o = SchemaLoop(p.s2.ctx, p.s2.seq, idx, chain, &p.f);
		*s = o.loop.First();
		return *s == OK && p.f.Same(p.ref);
	};
	h.after1 = [&](SslmGpuStatus* s) {
		SchemaOut o = loop1();
		*s = o.loop.First();
		return *s == OK && p.f.Same(p.ref);
	};
	std::vector<TraceEv> ccr;
	const uint32_t K = CountSites(src, [&] { own = loop1(); }, &ccr);
	std::printf("%s sites=%u counting-pass=%s\n", cell, K, St(own.loop.First()));
	SweepTally t;
	Sweep(cell, src, NewFault::BadAlloc, E_OUTOFMEMORY, K, ccr, static_cast<uint32_t>(OptInt("from", 1)),
	      static_cast<uint32_t>(OptInt("to", 0x7FFFFFFF)), nullptr, h, v, t);
	t.Print(cell);
	return v.Finish();
}
}  // namespace

// =============================================================================================
// R5: one decode step, per-token primitives and the decode-step bridge.
// =============================================================================================
namespace {
struct DecodeFx {
	Pair p;
	std::vector<int32_t> prompt;
	Ref ref;
	bool bridge = false;
	SslmGpuStatus own_status = OK;
	int32_t own_token = -1;
	StepOut own_step;
	// Built after the armed call, never inside it: the armed call runs under operator-new counting,
	// and a harness allocation there would move every site index.
	std::string Desc() const {
		if (bridge) return St(own_status);
		return std::string(St(own_status)) + " (embed=" + St(own_step.embed) + " step=" + St(own_step.step) +
		       " ready=" + St(own_step.ready_ret) + " out_status=" + St(own_step.out_status) + ")";
	}
	// Establish the post-prompt state on stack s (reset + 5-token prefill; the bridge path also
	// spends the prefill's ready_for_logits with one clean bridge call, so the armed call drives a
	// token through the layers).
	bool Pre(Stack& s) {
		if (sslm_gpu_seq_reset(s.ctx, s.seq) != OK) return false;
		if (SslmGpuSeqPrefillPromptForG5Bridge(s.ctx, s.seq, prompt.data(), static_cast<int32_t>(prompt.size()), 64) != OK)
			return false;
		if (bridge) {
			int32_t tok = -1;
			if (SslmGpuSeqDecodeStepForG5Bridge(s.ctx, s.seq, kStepToken, 64, &tok) != OK) return false;
		}
		return true;
	}
	// The call under test.
	void Call(Stack& s) {
		if (bridge) {
			g_phase.store(PH_CALL);
			own_token = -1;
			own_status = SslmGpuSeqDecodeStepForG5Bridge(s.ctx, s.seq, kStepToken, 64, &own_token);
			g_phase.store(PH_NONE);
		} else {
			own_step = DecodeStep(s.ctx, s.seq);
			own_status = own_step.First();
		}
	}
	// Pre + Call + (token, state) on stack s.
	bool Run(Stack& s, Ref* out, SslmGpuStatus* st) {
		if (!Pre(s)) {
			*st = DEVICE_LOST;
			return false;
		}
		Call(s);
		*st = own_status;
		if (own_status != OK) return false;
		if (bridge) {
			out->token = own_token;
		} else {
			out->token = -1;
			if (SslmGpuSeqFinishTokenForG5Bridge(s.ctx, s.seq, &out->token) != OK) return false;
		}
		out->blob = SaveBlob(s.ctx, s.seq);
		return !out->blob.empty();
	}
};

int CellDecodeSweep(const char* cell, Src src, NewFault kind = NewFault::BadAlloc) {
	Verdict v(cell);
	if (!Hooks(cell)) return SetupFail(cell, "hooks");
	DecodeFx d;
	d.bridge = Opt("path", "step") == "bridge";
	if (!d.p.Open(Opt("qwen3"), 5)) return SetupFail(cell, "fixture");
	d.prompt = QwenPrompt(5);
	SslmGpuStatus st;
	Ref r2;
	if (!d.Run(d.p.s1, &d.ref, &st) || !d.Run(d.p.s2, &r2, &st) || !(r2.token == d.ref.token && r2.blob == d.ref.blob))
		return SetupFail(cell, "decode reference");
	SweepHooks h;
	h.pre = [&] { return d.Pre(d.p.s1); };
	h.armed = [&] { d.Call(d.p.s1); };
	h.judge = [&](std::string* s) {
		*s = d.Desc();
		// Plan Sec3.5 R10: a foreign exception is not an allocation failure -- SSLM_DEVICE_LOST.
		return d.own_status == (kind == NewFault::Foreign ? DEVICE_LOST : ALLOC_FAILED);
	};
	// "Another sequence's next step is clean" (plan Sec3.5 R5): the second context's next step, and the
	// same handle's, are SSLM_OK with the reference token and state.
	h.after2 = [&](SslmGpuStatus* s) {
		Ref o;
		return d.Run(d.p.s2, &o, s) && o.token == d.ref.token && o.blob == d.ref.blob;
	};
	h.after1 = [&](SslmGpuStatus* s) {
		Ref o;
		return d.Run(d.p.s1, &o, s) && o.token == d.ref.token && o.blob == d.ref.blob;
	};
	std::vector<TraceEv> ccr;
	d.Pre(d.p.s1);
	const uint32_t K = CountSites(src, [&] { d.Call(d.p.s1); }, &ccr);
	std::printf("%s path=%s sites=%u counting-pass=%s token=%d\n", cell, d.bridge ? "bridge" : "step", K,
	            St(d.own_status), d.ref.token);
	// --select=window-edges: the first and last operator-new site of each recording window (R10).
	std::vector<uint32_t> sel;
	if (Opt("select") == "window-edges") {
		std::map<int, std::pair<uint32_t, uint32_t>> w;
		for (uint32_t k = 1; k <= K && k < kMaxSites; ++k) {
			if (g_labels[k].window == 0) continue;
			auto it = w.find(g_labels[k].window);
			if (it == w.end()) w[g_labels[k].window] = {k, k};
			else it->second.second = k;
		}
		for (const auto& kv : w) {
			sel.push_back(kv.second.first);
			if (kv.second.second != kv.second.first) sel.push_back(kv.second.second);
		}
		std::printf("%s window-edges: %zu windows, %zu sites\n", cell, w.size(), sel.size());
		if (sel.empty()) return SetupFail(cell, "no recording-window site");
	}
	SweepTally t;
	Sweep(cell, src, kind, E_OUTOFMEMORY, K, ccr, static_cast<uint32_t>(OptInt("from", 1)),
	      static_cast<uint32_t>(OptInt("to", 0x7FFFFFFF)), sel.empty() ? nullptr : &sel, h, v, t);
	t.Print(cell);
	return v.Finish();
}
}  // namespace

// =============================================================================================
// R6: batch decode -- one sequence's allocation failure is per-sequence.
// =============================================================================================
namespace {
int CellBatch(const char* cell) {
	Verdict v(cell);
	if (!Hooks(cell)) return SetupFail(cell, "hooks");
	Pair p;
	if (!p.Open(Opt("qwen3"), 5)) return SetupFail(cell, "fixture");
	SslmGpuSequenceHandle* seqs[3] = {p.s1.seq, nullptr, nullptr};
	if (sslm_gpu_seq_create(p.s1.ctx, p.s1.model, 64, &seqs[1]) != OK ||
	    sslm_gpu_seq_create(p.s1.ctx, p.s1.model, 64, &seqs[2]) != OK)
		return SetupFail(cell, "batch sequences");
	const std::vector<int32_t> prompt = QwenPrompt(5);
	auto prep = [&](int i) {
		return sslm_gpu_seq_reset(p.s1.ctx, seqs[i]) == OK &&
		       SslmGpuSeqPrefillPromptForG5Bridge(p.s1.ctx, seqs[i], prompt.data(), 5, 64) == OK &&
		       sslm_gpu_seq_embed_token(p.s1.ctx, seqs[i], kStepToken) == OK;
	};
	SslmGpuStatus sts[3];
	int32_t toks[3];
	auto batch = [&] {
		for (auto& s : sts) s = DEVICE_LOST;
		g_phase.store(PH_CALL);
		const SslmGpuStatus call = sslm_decode_step_batch_gpu(p.s1.ctx, seqs, nullptr, 3, 0xFFFFFFFFu, sts);
		g_phase.store(PH_NONE);
		for (int i = 0; i < 3; ++i) {
			toks[i] = -1;
			if (call == OK && sts[i] == OK) (void)SslmGpuSeqFinishTokenForG5Bridge(p.s1.ctx, seqs[i], &toks[i]);
		}
		return call;
	};
	for (int i = 0; i < 3; ++i)
		if (!prep(i)) return SetupFail(cell, "batch prep");
	if (batch() != OK || sts[0] != OK || sts[1] != OK || sts[2] != OK) return SetupFail(cell, "clean batch");
	const int32_t ref_tok[3] = {toks[0], toks[1], toks[2]};
	// Sites inside sequence 1's submission: the seam's per-sequence count, and the operator-new sites in
	// the second recording window (each sequence is its own submission, drained before the next).
	if (Has("trace")) {
		for (int i = 0; i < 3; ++i) prep(i);
		TraceStart();
		batch();
		const std::vector<TraceEv> t = TraceStop();
		for (size_t i = 0; i < t.size(); ++i)
			std::printf("TRACE %zu %s win=%u heap=%u width=%llu\n", i, MethodName(t[i].method), t[i].window, t[i].heap,
			            static_cast<unsigned long long>(t[i].width));
	}
	for (int i = 0; i < 3; ++i) prep(i);
	SslmGpuAllocCounterResetForTest();
	batch();
	const uint32_t total = SslmGpuAllocCountForTest();
	const uint32_t per = total / 3;
	for (int i = 0; i < 3; ++i) prep(i);
	NewCountOnly(true);
	batch();
	NewStop();
	const uint32_t Knew = g_new.count.load();
	uint32_t new_site = 0;
	for (uint32_t k = 1; k <= Knew && k < kMaxSites; ++k)
		if (g_labels[k].window == 2) {
			new_site = k;
			break;
		}
	std::printf("%s seam allocations per batch=%u (per sequence %u) operator-new sites=%u seq1-window site=%u tokens=%d,%d,%d\n",
	            cell, total, per, Knew, new_site, ref_tok[0], ref_tok[1], ref_tok[2]);
	if (per * 3 != total || new_site == 0) return SetupFail(cell, "per-sequence site");
	struct Leg {
		const char* name;
		bool seam;
		long hr;
		bool removed_seam;
		SslmGpuStatus want1;
		bool want_rest_ok;
		const char* rule;
	};
	const Leg legs[] = {
	    {"R6.seam-oom", true, E_OUTOFMEMORY, false, ALLOC_FAILED, true, "rule 2"},
	    {"R6.new-bad_alloc", false, 0, false, ALLOC_FAILED, true, "rule 2"},
	    {"R6.mustreject-hr-removed", true, static_cast<long>(DXGI_ERROR_DEVICE_REMOVED), false, DEVICE_LOST, false, "rule 3"},
	    {"R6.mustreject-removed-seam", true, E_OUTOFMEMORY, true, DEVICE_LOST, false, "rule 1"},
	};
	for (const Leg& L : legs) {
		for (int i = 0; i < 3; ++i) prep(i);
		if (L.removed_seam) ArmGpuMapDeviceRemovedQueryInjection();
		if (L.seam) {
			SslmGpuAllocCounterResetForTest();
			ArmGpuAllocFaultAtOccurrence(per + 1, L.hr);
		} else {
			NewArm(new_site, NewFault::BadAlloc);
		}
		batch();
		NewStop();
		ArmGpuAllocFaultAtOccurrence(0, S_OK);
		bool ok = sts[0] == OK && toks[0] == ref_tok[0] && sts[1] == L.want1;
		if (L.want_rest_ok) {
			ok = ok && sts[2] == OK && toks[2] == ref_tok[2];
		} else {
			ok = ok && sts[2] == DEVICE_LOST;  // the must-reject: a lost device poisons the rest
		}
		// The armed batch's own outcome, kept before the usable-afterwards batch below overwrites it.
		const SslmGpuStatus armed_sts[3] = {sts[0], sts[1], sts[2]};
		const int32_t armed_toks[3] = {toks[0], toks[1], toks[2]};
		// The faulted sequence's handle stays usable (rule 2 legs): a clean batch afterwards.
		bool usable = true;
		if (L.want_rest_ok) {
			for (int i = 0; i < 3; ++i) prep(i);
			usable = batch() == OK && sts[1] == OK && toks[1] == ref_tok[1];
		}
		SslmGpuStatus s2 = OK;
		const bool a2 = p.After2(&s2);
		const bool conforms = ok && usable && a2;
		v.Leg(conforms);
		std::printf("%s %s (%s) slots=%s,%s,%s tokens=%d,%d,%d want slot1=%s rest=%s usable-after=%d after2=%s -> %s\n",
		            cell, L.name, L.rule, St(armed_sts[0]), St(armed_sts[1]), St(armed_sts[2]), armed_toks[0],
		            armed_toks[1], armed_toks[2], St(L.want1),
		            L.want_rest_ok ? "OK" : "DEVICE_LOST", usable ? 1 : 0, St(s2), conforms ? "PASS" : "FAIL");
	}
	sslm_gpu_seq_release(p.s1.ctx, seqs[1]);
	sslm_gpu_seq_release(p.s1.ctx, seqs[2]);
	return v.Finish();
}
}  // namespace

// =============================================================================================
// Hook legs (R9, R12, R7.1, R7.3, R11's other/stranded drives): one D3D12 call of a chosen family
// fails, at a site chosen from a clean trace of the same call.
// =============================================================================================
namespace {
enum Expect { EXP_ALLOC, EXP_LOST };
enum Rule { RULE0, RULE1, RULE2, RULE3 };
enum RouteId { RT_PROMPT, RT_DECODE, RT_SEQCREATE, RT_MAP, RT_RESTORE, RT_CTX, RT_DEVLOGITS, RT_ADAPTER, RT_SCHEMA,
               RT_PROMPT_FIRST, RT_MAPHEAD };
const char* RouteName(int r) {
	static const char* k[] = {"prompt5", "decode", "seq_create", "model_map", "seq_restore", "context_create",
	                          "finish_token_device_logits", "adapter_map", "schema5", "prompt5_first_in_process",
	                          "model_map_head_on_device"};
	return k[r];
}

// A call site, resolved against a clean trace of the route: the `nth` call of `method` made after the
// `anchor_n`-th `anchor` call (anchor M_NONE = from the start), optionally only inside recording
// window `window`, skipping calls on a buffer of `exclude_width` bytes (the timestamp readback, whose
// Map the engine does not check).
struct Sel {
	int method;
	int anchor;
	uint32_t anchor_n;
	uint32_t nth;
	uint16_t window;
	uint64_t exclude_width;
	bool before = false;  // the nth `method` call BEFORE the anchor_n-th anchor, counting backwards
};
uint32_t Resolve(const Sel& s, const std::vector<TraceEv>& t) {
	if (s.before) {
		size_t at = t.size();
		uint32_t seen = 0;
		for (size_t i = 0; i < t.size(); ++i)
			if (t[i].method == s.anchor && ++seen == s.anchor_n) {
				at = i;
				break;
			}
		if (at == t.size()) return 0;
		uint32_t matched = 0;
		for (size_t i = at; i-- > 0;) {
			if (t[i].method != s.method) continue;
			if (s.exclude_width != 0 && t[i].width == s.exclude_width) continue;
			if (++matched == s.nth) {
				uint32_t ordinal = 0;
				for (size_t j = 0; j <= i; ++j) ordinal += t[j].method == s.method;
				return ordinal;
			}
		}
		return 0;
	}
	uint32_t anchors = 0, matched = 0, ordinal = 0;
	for (const TraceEv& e : t) {
		if (e.method == s.method) ++ordinal;
		if (s.anchor != M_NONE && e.method == s.anchor) {
			++anchors;
			if (e.method != s.method) continue;
		}
		if (e.method != s.method) continue;
		if (s.anchor != M_NONE && anchors < s.anchor_n) continue;
		if (s.window != 0 && e.window != s.window) continue;
		if (s.exclude_width != 0 && e.width == s.exclude_width) continue;
		if (++matched == s.nth) return ordinal;
	}
	return 0;
}

struct HookLeg {
	const char* name;
	int route;
	Sel sel;
	long hr;
	int behavior;
	uint32_t n;
	Expect expect;
	Rule rule;
	bool isolate;       // run alone in a fresh process (rule 0, and first-in-process legs)
	const char* cells;  // which plan cells this leg realizes
};
constexpr long OOM = E_OUTOFMEMORY;
constexpr long FAIL = E_FAIL;
constexpr long REM = static_cast<long>(DXGI_ERROR_DEVICE_REMOVED);
constexpr uint64_t kTsWidth = 8192ull * 8ull;  // harness::Device::kMaxTimestampSlots * sizeof(UINT64)

#define SEL(m, a, an, nth, w) Sel{m, a, an, nth, w, kTsWidth}
#define SELB(m, a, an, nth) Sel{m, a, an, nth, 0, kTsWidth, true}
const HookLeg kLegs[] = {
    // ---- the device-logits bundle at a head-flag map (T1): phase A's staging Map and the bundle's
    //      own pipeline, then phase B's Close and Signal. Anchored on the bundle's root signature, the
    //      only one a map creates, so the number of uploads before it does not matter. ----
    {"H.map.oom", RT_MAPHEAD, SELB(M_MAP, M_ROOTSIG, 1, 3), OOM, HB_FAIL_NO_CALL, 1, EXP_ALLOC, RULE2, false, "R9 T1"},
    {"H.map.fail", RT_MAPHEAD, SELB(M_MAP, M_ROOTSIG, 1, 3), FAIL, HB_FAIL_NO_CALL, 1, EXP_LOST, RULE3, false, "R9 R11 T1"},
    {"H.rootsig.oom", RT_MAPHEAD, SEL(M_ROOTSIG, M_NONE, 0, 1, 0), OOM, HB_FAIL_NO_CALL, 1, EXP_ALLOC, RULE2, false, "R9 T1"},
    {"H.pso.oom", RT_MAPHEAD, SEL(M_PSO, M_NONE, 0, 1, 0), OOM, HB_FAIL_NO_CALL, 1, EXP_ALLOC, RULE2, false, "R9 T1"},
    {"H.close.A.oom", RT_MAPHEAD, SEL(M_CLOSE, M_ROOTSIG, 1, 1, 0), OOM, HB_FAIL_NO_CALL, 1, EXP_ALLOC, RULE2, false, "R9(A) T1"},
    {"H.close.B.oom", RT_MAPHEAD, SEL(M_CLOSE, M_ROOTSIG, 1, 1, 0), OOM, HB_CALL_THEN_FAIL, 1, EXP_ALLOC, RULE2, false, "R9(B) T1"},
    {"H.signal.oom", RT_MAPHEAD, SEL(M_SIGNAL, M_ROOTSIG, 1, 1, 0), OOM, HB_FAIL_NO_CALL, 1, EXP_ALLOC, RULE2, false, "R9 T1"},
    {"H.signal.fail", RT_MAPHEAD, SEL(M_SIGNAL, M_ROOTSIG, 1, 1, 0), FAIL, HB_FAIL_NO_CALL, 1, EXP_LOST, RULE3, false, "R9 T1"},
    // ---- R7.1's two remaining families: allocator Reset, and the context's own Init objects ----
    {"P.areset1.removed", RT_PROMPT, SEL(M_ARESET, M_NONE, 0, 1, 0), REM, HB_FAIL_NO_CALL, 1, EXP_LOST, RULE3, false, "R7.1"},
    {"X.cqueue1.removed", RT_CTX, SEL(M_CQUEUE, M_NONE, 0, 1, 0), REM, HB_FAIL_NO_CALL, 1, EXP_LOST, RULE3, false, "R7.1 T6"},
    // ---- the prompt twin, 5 tokens = two sub-chunks (T14, T15, T16) ----
    {"P.areset1.oom", RT_PROMPT, SEL(M_ARESET, M_NONE, 0, 1, 0), OOM, HB_FAIL_NO_CALL, 1, EXP_ALLOC, RULE2, false, "R9"},
    {"P.areset1.fail", RT_PROMPT, SEL(M_ARESET, M_NONE, 0, 1, 0), FAIL, HB_FAIL_NO_CALL, 1, EXP_LOST, RULE3, false, "R9"},
    {"P.lreset2.oom", RT_PROMPT, SEL(M_LRESET, M_NONE, 0, 2, 0), OOM, HB_FAIL_NO_CALL, 1, EXP_ALLOC, RULE2, false, "R9"},
    {"P.lreset2.fail", RT_PROMPT, SEL(M_LRESET, M_NONE, 0, 2, 0), FAIL, HB_FAIL_NO_CALL, 1, EXP_LOST, RULE3, false, "R9"},
    {"P.lreset1.removed", RT_PROMPT, SEL(M_LRESET, M_NONE, 0, 1, 0), REM, HB_FAIL_NO_CALL, 1, EXP_LOST, RULE3, false, "R7.1"},
    {"P.mapw1.oom", RT_PROMPT, SEL(M_MAP, M_NONE, 0, 1, 1), OOM, HB_FAIL_NO_CALL, 1, EXP_ALLOC, RULE2, false, "R9"},
    {"P.mapw1.fail", RT_PROMPT, SEL(M_MAP, M_NONE, 0, 1, 1), FAIL, HB_FAIL_NO_CALL, 1, EXP_LOST, RULE3, false, "R9"},
    {"P.mapw2.removed", RT_PROMPT, SEL(M_MAP, M_NONE, 0, 1, 2), REM, HB_FAIL_NO_CALL, 1, EXP_LOST, RULE3, false, "R7.1"},
    {"P.close1.A.oom", RT_PROMPT, SEL(M_CLOSE, M_NONE, 0, 1, 0), OOM, HB_FAIL_NO_CALL, 1, EXP_ALLOC, RULE2, false, "R9(A)"},
    {"P.close1.A.fail", RT_PROMPT, SEL(M_CLOSE, M_NONE, 0, 1, 0), FAIL, HB_FAIL_NO_CALL, 1, EXP_LOST, RULE3, false, "R9 R7.3"},
    {"P.close1.B.oom", RT_PROMPT, SEL(M_CLOSE, M_NONE, 0, 1, 0), OOM, HB_CALL_THEN_FAIL, 1, EXP_ALLOC, RULE2, false, "R9(B)"},
    {"P.close2.B.removed", RT_PROMPT, SEL(M_CLOSE, M_NONE, 0, 2, 0), REM, HB_CALL_THEN_FAIL, 1, EXP_LOST, RULE3, false, "R7.1"},
    {"P.close1.A2.oom", RT_PROMPT, SEL(M_CLOSE, M_NONE, 0, 1, 0), OOM, HB_FAIL_NO_CALL, 2, EXP_LOST, RULE0, true, "R9(A2)"},
    {"P.signal1.oom", RT_PROMPT, SEL(M_SIGNAL, M_NONE, 0, 1, 0), OOM, HB_FAIL_NO_CALL, 1, EXP_ALLOC, RULE2, false, "R9"},
    {"P.signal1.fail", RT_PROMPT, SEL(M_SIGNAL, M_NONE, 0, 1, 0), FAIL, HB_FAIL_NO_CALL, 1, EXP_LOST, RULE3, false, "R9 R7.3"},
    {"P.signal2.removed", RT_PROMPT, SEL(M_SIGNAL, M_NONE, 0, 2, 0), REM, HB_FAIL_NO_CALL, 1, EXP_LOST, RULE3, false, "R7.1"},
    {"P.rbmap1.oom", RT_PROMPT, SEL(M_MAP, M_SIGNAL, 1, 1, 0), OOM, HB_FAIL_NO_CALL, 1, EXP_ALLOC, RULE2, false, "R9 T16"},
    {"P.rbmap1.fail", RT_PROMPT, SEL(M_MAP, M_SIGNAL, 1, 1, 0), FAIL, HB_FAIL_NO_CALL, 1, EXP_LOST, RULE3, false, "R12 non-final"},
    {"P.rbmap2.oom", RT_PROMPT, SEL(M_MAP, M_SIGNAL, 2, 1, 0), OOM, HB_FAIL_NO_CALL, 1, EXP_ALLOC, RULE2, false, "R9 T16"},
    {"P.rbmap2.fail", RT_PROMPT, SEL(M_MAP, M_SIGNAL, 2, 1, 0), FAIL, HB_FAIL_NO_CALL, 1, EXP_LOST, RULE3, false, "R12 final"},
    {"P.rbmap2.removed", RT_PROMPT, SEL(M_MAP, M_SIGNAL, 2, 1, 0), REM, HB_FAIL_NO_CALL, 1, EXP_LOST, RULE3, false, "R7.1"},
    {"P.setevent1.oom", RT_PROMPT, SEL(M_SETEVENT, M_NONE, 0, 1, 0), OOM, HB_FAIL_NO_CALL, 1, EXP_LOST, RULE0, true, "R9 T16"},
    {"P.setevent1.fail", RT_PROMPT, SEL(M_SETEVENT, M_NONE, 0, 1, 0), FAIL, HB_FAIL_NO_CALL, 1, EXP_LOST, RULE0, true, "R12 non-final"},
    {"P.setevent2.fail", RT_PROMPT, SEL(M_SETEVENT, M_NONE, 0, 2, 0), FAIL, HB_FAIL_NO_CALL, 1, EXP_LOST, RULE0, true, "R12 final"},
    // ---- pipeline creation: the composed cache builds on the process's first prefill ----
    {"F.pso1.oom", RT_PROMPT_FIRST, SEL(M_PSO, M_NONE, 0, 1, 0), OOM, HB_FAIL_NO_CALL, 1, EXP_ALLOC, RULE2, true, "R9"},
    {"F.pso1.fail", RT_PROMPT_FIRST, SEL(M_PSO, M_NONE, 0, 1, 0), FAIL, HB_FAIL_NO_CALL, 1, EXP_LOST, RULE3, true, "R9"},
    {"F.rootsig1.oom", RT_PROMPT_FIRST, SEL(M_ROOTSIG, M_NONE, 0, 1, 0), OOM, HB_FAIL_NO_CALL, 1, EXP_ALLOC, RULE2, true, "R9"},
    {"F.rootsig1.removed", RT_PROMPT_FIRST, SEL(M_ROOTSIG, M_NONE, 0, 1, 0), REM, HB_FAIL_NO_CALL, 1, EXP_LOST, RULE3, true, "R7.1"},
    // ---- the per-token decode step (T12, T13, T16) ----
    {"D.areset1.oom", RT_DECODE, SEL(M_ARESET, M_NONE, 0, 1, 0), OOM, HB_FAIL_NO_CALL, 1, EXP_ALLOC, RULE2, false, "R9"},
    {"D.lreset1.fail", RT_DECODE, SEL(M_LRESET, M_NONE, 0, 1, 0), FAIL, HB_FAIL_NO_CALL, 1, EXP_LOST, RULE3, false, "R9"},
    {"D.mapw1.oom", RT_DECODE, SEL(M_MAP, M_NONE, 0, 1, 1), OOM, HB_FAIL_NO_CALL, 1, EXP_ALLOC, RULE2, false, "R9"},
    {"D.mapw1.removed", RT_DECODE, SEL(M_MAP, M_NONE, 0, 1, 1), REM, HB_FAIL_NO_CALL, 1, EXP_LOST, RULE3, false, "R7.1 decode"},
    {"D.close1.A.oom", RT_DECODE, SEL(M_CLOSE, M_NONE, 0, 1, 0), OOM, HB_FAIL_NO_CALL, 1, EXP_ALLOC, RULE2, false, "R9(A)"},
    {"D.close1.A.fail", RT_DECODE, SEL(M_CLOSE, M_NONE, 0, 1, 0), FAIL, HB_FAIL_NO_CALL, 1, EXP_LOST, RULE3, false, "R9 R7.3"},
    {"D.close1.B.oom", RT_DECODE, SEL(M_CLOSE, M_NONE, 0, 1, 0), OOM, HB_CALL_THEN_FAIL, 1, EXP_ALLOC, RULE2, false, "R9(B)"},
    {"D.close1.A2.oom", RT_DECODE, SEL(M_CLOSE, M_NONE, 0, 1, 0), OOM, HB_FAIL_NO_CALL, 2, EXP_LOST, RULE0, true, "R9(A2)"},
    {"D.signal1.oom", RT_DECODE, SEL(M_SIGNAL, M_NONE, 0, 1, 0), OOM, HB_FAIL_NO_CALL, 1, EXP_ALLOC, RULE2, false, "R9"},
    {"D.signal1.fail", RT_DECODE, SEL(M_SIGNAL, M_NONE, 0, 1, 0), FAIL, HB_FAIL_NO_CALL, 1, EXP_LOST, RULE3, false, "R9 R7.3"},
    {"D.rbmap1.oom", RT_DECODE, SEL(M_MAP, M_SIGNAL, 1, 1, 0), OOM, HB_FAIL_NO_CALL, 1, EXP_ALLOC, RULE2, false, "R9 T16"},
    {"D.rbmap1.fail", RT_DECODE, SEL(M_MAP, M_SIGNAL, 1, 1, 0), FAIL, HB_FAIL_NO_CALL, 1, EXP_LOST, RULE3, false, "R12 ready"},
    {"D.setevent1.fail", RT_DECODE, SEL(M_SETEVENT, M_NONE, 0, 1, 0), FAIL, HB_FAIL_NO_CALL, 1, EXP_LOST, RULE0, true, "R12 ready"},
    // ---- sequence create: UploadResidentBufferSyncTo on the context's own list (T7) ----
    {"C.map1.oom", RT_SEQCREATE, SEL(M_MAP, M_NONE, 0, 1, 0), OOM, HB_FAIL_NO_CALL, 1, EXP_ALLOC, RULE2, false, "R9 T7"},
    {"C.map1.fail", RT_SEQCREATE, SEL(M_MAP, M_NONE, 0, 1, 0), FAIL, HB_FAIL_NO_CALL, 1, EXP_LOST, RULE3, false, "R9 T7"},
    {"C.areset1.oom", RT_SEQCREATE, SEL(M_ARESET, M_NONE, 0, 1, 0), OOM, HB_FAIL_NO_CALL, 1, EXP_ALLOC, RULE2, false, "R9 T7"},
    {"C.lreset1.oom", RT_SEQCREATE, SEL(M_LRESET, M_NONE, 0, 1, 0), OOM, HB_FAIL_NO_CALL, 1, EXP_ALLOC, RULE2, false, "R9 T7"},
    {"C.close1.A.oom", RT_SEQCREATE, SEL(M_CLOSE, M_NONE, 0, 1, 0), OOM, HB_FAIL_NO_CALL, 1, EXP_ALLOC, RULE2, false, "R9(A) T7"},
    {"C.close1.A.fail", RT_SEQCREATE, SEL(M_CLOSE, M_NONE, 0, 1, 0), FAIL, HB_FAIL_NO_CALL, 1, EXP_LOST, RULE3, false, "R9 T7"},
    {"C.close1.B.oom", RT_SEQCREATE, SEL(M_CLOSE, M_NONE, 0, 1, 0), OOM, HB_CALL_THEN_FAIL, 1, EXP_ALLOC, RULE2, false, "R9(B) T7"},
    {"C.close1.A2.oom", RT_SEQCREATE, SEL(M_CLOSE, M_NONE, 0, 1, 0), OOM, HB_FAIL_NO_CALL, 2, EXP_LOST, RULE0, true, "R9(A2) T7"},
    {"C.signal1.oom", RT_SEQCREATE, SEL(M_SIGNAL, M_NONE, 0, 1, 0), OOM, HB_FAIL_NO_CALL, 1, EXP_LOST, RULE0, true, "R9 T7 (no retry)"},
    // ---- model map at flags 0: the uploads (T4) ----
    {"M.map1.oom", RT_MAP, SEL(M_MAP, M_NONE, 0, 1, 0), OOM, HB_FAIL_NO_CALL, 1, EXP_ALLOC, RULE2, false, "R9 T4"},
    {"M.lreset1.oom", RT_MAP, SEL(M_LRESET, M_NONE, 0, 1, 0), OOM, HB_FAIL_NO_CALL, 1, EXP_ALLOC, RULE2, false, "R9 T4"},
    {"M.lreset1.fail", RT_MAP, SEL(M_LRESET, M_NONE, 0, 1, 0), FAIL, HB_FAIL_NO_CALL, 1, EXP_LOST, RULE3, false, "R9 T4"},
    {"M.close2.A.oom", RT_MAP, SEL(M_CLOSE, M_NONE, 0, 2, 0), OOM, HB_FAIL_NO_CALL, 1, EXP_ALLOC, RULE2, false, "R9(A) T4"},
    {"M.signal1.oom", RT_MAP, SEL(M_SIGNAL, M_NONE, 0, 1, 0), OOM, HB_FAIL_NO_CALL, 1, EXP_LOST, RULE0, true, "R9 T4 (no retry)"},
    // ---- sequence restore: create's upload, the device round-trip on the process list (T8), the
    //      resident upload (T9) ----
    {"R.areset2.oom", RT_RESTORE, SEL(M_ARESET, M_NONE, 0, 2, 0), OOM, HB_FAIL_NO_CALL, 1, EXP_ALLOC, RULE2, false, "R9 T8"},
    {"R.areset2.fail", RT_RESTORE, SEL(M_ARESET, M_NONE, 0, 2, 0), FAIL, HB_FAIL_NO_CALL, 1, EXP_LOST, RULE3, false, "R9 T8"},
    {"R.map2.oom", RT_RESTORE, SEL(M_MAP, M_NONE, 0, 2, 0), OOM, HB_FAIL_NO_CALL, 1, EXP_ALLOC, RULE2, false, "R9 T8"},
    {"R.close2.A.oom", RT_RESTORE, SEL(M_CLOSE, M_NONE, 0, 2, 0), OOM, HB_FAIL_NO_CALL, 1, EXP_ALLOC, RULE2, false, "R9(A) T8"},
    {"R.rbmap.oom", RT_RESTORE, SEL(M_MAP, M_SIGNAL, 2, 1, 0), OOM, HB_FAIL_NO_CALL, 1, EXP_ALLOC, RULE2, false, "R9 T8"},
    {"R.rbmap.fail", RT_RESTORE, SEL(M_MAP, M_SIGNAL, 2, 1, 0), FAIL, HB_FAIL_NO_CALL, 1, EXP_LOST, RULE3, false, "R9 T8"},
    {"R.areset3.oom", RT_RESTORE, SEL(M_ARESET, M_NONE, 0, 3, 0), OOM, HB_FAIL_NO_CALL, 1, EXP_ALLOC, RULE2, false, "R9 T9"},
    {"R.signal2.oom", RT_RESTORE, SEL(M_SIGNAL, M_NONE, 0, 2, 0), OOM, HB_FAIL_NO_CALL, 1, EXP_LOST, RULE0, true, "R9 T8 (no retry)"},
    {"R.close2.A2.oom", RT_RESTORE, SEL(M_CLOSE, M_NONE, 0, 2, 0), OOM, HB_FAIL_NO_CALL, 2, EXP_LOST, RULE0, true, "R9(A2) T8"},
    // ---- context create: the context's own device setup (T6) ----
    {"X.cqueue1.oom", RT_CTX, SEL(M_CQUEUE, M_NONE, 0, 1, 0), OOM, HB_FAIL_NO_CALL, 1, EXP_ALLOC, RULE2, false, "R9 T6"},
    {"X.cqueue1.fail", RT_CTX, SEL(M_CQUEUE, M_NONE, 0, 1, 0), FAIL, HB_FAIL_NO_CALL, 1, EXP_LOST, RULE3, false, "R9 T6"},
    {"X.calloc1.oom", RT_CTX, SEL(M_CALLOC, M_NONE, 0, 1, 0), OOM, HB_FAIL_NO_CALL, 1, EXP_ALLOC, RULE2, false, "R9 T6"},
    {"X.clist1.oom", RT_CTX, SEL(M_CLIST, M_NONE, 0, 1, 0), OOM, HB_FAIL_NO_CALL, 1, EXP_ALLOC, RULE2, false, "R9 T6"},
    {"X.close1.oom", RT_CTX, SEL(M_CLOSE, M_NONE, 0, 1, 0), OOM, HB_FAIL_NO_CALL, 1, EXP_ALLOC, RULE2, false, "R9 T6"},
    {"X.cfence1.oom", RT_CTX, SEL(M_CFENCE, M_NONE, 0, 1, 0), OOM, HB_FAIL_NO_CALL, 1, EXP_ALLOC, RULE2, false, "R9 T6"},
    {"X.cqheap1.oom", RT_CTX, SEL(M_CQHEAP, M_NONE, 0, 1, 0), OOM, HB_FAIL_NO_CALL, 1, EXP_ALLOC, RULE2, false, "R9 T6"},
    // The context's setup allocates exactly the timestamp readback, so this one selector keeps it.
    {"X.ccr1.oom", RT_CTX, Sel{M_CCR, M_NONE, 0, 1, 0, 0}, OOM, HB_FAIL_NO_CALL, 1, EXP_ALLOC, RULE2, false, "R1 T6"},
    // ---- the finish on a device-resident head: RunDeviceLogits (T2; TE-422 S-2 site 3) ----
    {"L.areset1.oom", RT_DEVLOGITS, SEL(M_ARESET, M_NONE, 0, 1, 0), OOM, HB_FAIL_NO_CALL, 1, EXP_ALLOC, RULE2, false, "R9 T2"},
    {"L.lreset1.oom", RT_DEVLOGITS, SEL(M_LRESET, M_NONE, 0, 1, 0), OOM, HB_FAIL_NO_CALL, 1, EXP_ALLOC, RULE2, false, "R9 T2"},
    {"L.lreset1.fail", RT_DEVLOGITS, SEL(M_LRESET, M_NONE, 0, 1, 0), FAIL, HB_FAIL_NO_CALL, 1, EXP_LOST, RULE3, false, "R9 T2"},
    {"L.close1.A.oom", RT_DEVLOGITS, SEL(M_CLOSE, M_NONE, 0, 1, 0), OOM, HB_FAIL_NO_CALL, 1, EXP_ALLOC, RULE2, false, "R9(A) T2"},
    {"L.signal1.oom", RT_DEVLOGITS, SEL(M_SIGNAL, M_NONE, 0, 1, 0), OOM, HB_FAIL_NO_CALL, 1, EXP_ALLOC, RULE2, false, "R9 T2"},
    {"L.setevent1.oom", RT_DEVLOGITS, SEL(M_SETEVENT, M_NONE, 0, 1, 0), OOM, HB_FAIL_NO_CALL, 1, EXP_LOST, RULE0, true, "R9 T2"},
    // ---- adapter map (T5) ----
    {"A.map1.oom", RT_ADAPTER, SEL(M_MAP, M_NONE, 0, 1, 0), OOM, HB_FAIL_NO_CALL, 1, EXP_ALLOC, RULE2, false, "R9 T5"},
    {"A.lreset1.oom", RT_ADAPTER, SEL(M_LRESET, M_NONE, 0, 1, 0), OOM, HB_FAIL_NO_CALL, 1, EXP_ALLOC, RULE2, false, "R9 T5"},
    {"A.close1.A.oom", RT_ADAPTER, SEL(M_CLOSE, M_NONE, 0, 1, 0), OOM, HB_FAIL_NO_CALL, 1, EXP_ALLOC, RULE2, false, "R9(A) T5"},
    {"A.close1.A.fail", RT_ADAPTER, SEL(M_CLOSE, M_NONE, 0, 1, 0), FAIL, HB_FAIL_NO_CALL, 1, EXP_LOST, RULE3, false, "R9 T5"},
    // ---- the schema twin's finish side (R12) ----
    {"S.rbmap1.fail", RT_SCHEMA, SEL(M_MAP, M_SIGNAL, 1, 1, 0), FAIL, HB_FAIL_NO_CALL, 1, EXP_LOST, RULE3, false, "R12 schema non-final"},
    {"S.rbmap2.fail", RT_SCHEMA, SEL(M_MAP, M_SIGNAL, 2, 1, 0), FAIL, HB_FAIL_NO_CALL, 1, EXP_LOST, RULE3, false, "R12 schema final"},
    {"S.rbmap2.oom", RT_SCHEMA, SEL(M_MAP, M_SIGNAL, 2, 1, 0), OOM, HB_FAIL_NO_CALL, 1, EXP_ALLOC, RULE2, false, "R9 schema"},
    {"S.setevent1.fail", RT_SCHEMA, SEL(M_SETEVENT, M_NONE, 0, 1, 0), FAIL, HB_FAIL_NO_CALL, 1, EXP_LOST, RULE0, true, "R12 schema non-final"},
};
#undef SEL
#undef SELB

// The route driver: one process, the routes a group needs.
struct RouteFx {
	Pair p;             // qwen3 (or the R15/G5 artifact for their groups), two contexts
	Artifact adapter;   // adapter group
	std::vector<uint8_t> blob;  // restore route: a saved 5-token sequence
	int32_t schema_idx = -1;
	std::vector<int32_t> chain;
	Ref ref_decode, ref_devlogits;
	bool decode_ready = false;
	// Outcome of the last call.
	SslmGpuStatus status = OK, read = OK;
	int stage = -1;
	std::string desc;
	DecodeFx dec;

	bool Pre(int r) {
		switch (r) {
			case RT_DECODE: return dec.Pre(p.s1);
			case RT_DEVLOGITS:
				return sslm_gpu_seq_reset(p.s1.ctx, p.s1.seq) == OK &&
				       SslmGpuSeqPrefillPromptForG5Bridge(p.s1.ctx, p.s1.seq, p.toks.data(),
				                                          static_cast<int32_t>(p.toks.size()), 64) == OK;
			default: return true;
		}
	}
	// The call under test. Handle-creating routes release what they created.
	void Call(int r) {
		status = OK;
		read = OK;
		stage = -1;
		switch (r) {
			case RT_PROMPT:
			case RT_PROMPT_FIRST: {
				LoopOut o = PromptLoop(p.s1.ctx, p.s1.seq, p.toks, &p.f);
				status = o.First();
				stage = o.FirstStage();
				read = o.read;
				desc = LoopDesc(o);
				break;
			}
			case RT_SCHEMA: {
				SchemaOut o = SchemaLoop(p.s1.ctx, p.s1.seq, schema_idx, chain, &p.f);
				status = o.loop.First();
				stage = o.loop.FirstStage();
				read = o.loop.read;
				desc = LoopDesc(o.loop) + " consumed=" + std::to_string(o.consumed) + " committed=" +
				       std::to_string(o.committed);
				break;
			}
			case RT_DECODE:
				dec.Call(p.s1);
				status = dec.own_status;
				desc = dec.Desc();
				break;
			case RT_SEQCREATE: {
				SslmGpuSequenceHandle* h = nullptr;
				status = sslm_gpu_seq_create(p.s1.ctx, p.s1.model, 64, &h);
				desc = std::string(St(status)) + (h && status != OK ? " handle!=null" : "");
				if (h) sslm_gpu_seq_release(p.s1.ctx, h);
				break;
			}
			case RT_MAP:
			case RT_MAPHEAD: {
				SslmGpuModelHandle* h = nullptr;
				GpuResidencyConfig rc{};
				rc.flags = r == RT_MAPHEAD ? SSLM_GPU_RESIDENCY_HEAD_ON_DEVICE : 0u;
				status = sslm_gpu_model_map(p.s1.ctx, &p.art.view, rc, &h);
				desc = std::string(St(status)) + (h && status != OK ? " handle!=null" : "");
				if (h) sslm_gpu_model_unmap(p.s1.ctx, h);
				break;
			}
			case RT_RESTORE: {
				SslmGpuSequenceHandle* h = nullptr;
				status = sslm_gpu_seq_restore(p.s1.ctx, p.s1.model, blob.data(), blob.size(), &h);
				bool same = false;
				if (h && status == OK) same = SaveBlob(p.s1.ctx, h) == blob;
				desc = std::string(St(status)) + (status == OK ? (same ? " state-equal" : " state-DIFFERS") : "");
				if (status == OK && !same) status = KV_MISMATCH;  // a restore that restores the wrong state
				if (h) sslm_gpu_seq_release(p.s1.ctx, h);
				break;
			}
			case RT_CTX: {
				SslmGpuContext* c = nullptr;
				status = sslm_gpu_context_create(GpuContextConfig{}, &c);
				desc = std::string(St(status)) + (c && status != OK ? " handle!=null" : "");
				if (c) sslm_gpu_context_destroy(c);
				break;
			}
			case RT_DEVLOGITS: {
				int32_t tok = -1;
				g_phase.store(PH_CALL);
				status = SslmGpuSeqFinishTokenForG5Bridge(p.s1.ctx, p.s1.seq, &tok);
				g_phase.store(PH_NONE);
				desc = std::string(St(status)) + " token=" + std::to_string(tok);
				if (status == OK && tok != ref_devlogits.token) status = SEQ_REJECTED;  // wrong token
				break;
			}
			case RT_ADAPTER: {
				SslmGpuAdapterHandle* h = nullptr;
				status = sslm_gpu_adapter_map(p.s1.ctx, p.s1.model, &adapter.view, &h);
				desc = std::string(St(status)) + (h && status != OK ? " handle!=null" : "");
				if (h) sslm_gpu_adapter_unmap(p.s1.ctx, h);
				break;
			}
		}
	}
	// The second context's next call of the same kind (the stuck-list oracle).
	bool After2(int r, SslmGpuStatus* st) {
		if (r == RT_SCHEMA) {
			SchemaOut o = SchemaLoop(p.s2.ctx, p.s2.seq, schema_idx, chain, &p.f);
			*st = o.loop.First();
			return *st == OK && p.f.Same(p.ref);
		}
		return p.After2(st);
	}
	// The same handle's next call: pre + the route, clean and equal where it has an output.
	bool After1(int r, SslmGpuStatus* st) {
		if (!Pre(r)) {
			*st = DEVICE_LOST;
			return false;
		}
		Call(r);
		*st = status;
		if (status != OK) return false;
		if (r == RT_PROMPT || r == RT_PROMPT_FIRST || r == RT_SCHEMA) return p.f.Same(p.ref);
		if (r == RT_DECODE) {
			int32_t tok = -1;
			if (SslmGpuSeqFinishTokenForG5Bridge(p.s1.ctx, p.s1.seq, &tok) != OK) return false;
			return tok == ref_decode.token && SaveBlob(p.s1.ctx, p.s1.seq) == ref_decode.blob;
		}
		return true;
	}
};

bool Judge(const HookLeg& L, const RouteFx& fx) {
	const bool loop_route = L.route == RT_PROMPT || L.route == RT_PROMPT_FIRST || L.route == RT_SCHEMA;
	if (L.expect == EXP_ALLOC) {
		if (fx.status != ALLOC_FAILED) return false;
		if (loop_route && fx.stage == 1 && fx.read != HIDDEN_UNAVAILABLE) return false;
		return true;
	}
	if (fx.status != DEVICE_LOST) return false;
	if (loop_route && fx.stage == 1 && fx.read != HIDDEN_UNAVAILABLE) return false;
	return true;
}

int CellHooks(const char* cell) {
	Verdict v(cell);
	const std::string group = Opt("group");
	const std::string only = Opt("only");
	if (Has("list")) {
		for (const HookLeg& L : kLegs)
			std::printf("LEG %s route=%s method=%s hr=%s behavior=%d n=%u expect=%s isolate=%d cells=%s\n", L.name,
			            RouteName(L.route), MethodName(L.sel.method), Hex(static_cast<uint32_t>(L.hr)).c_str(),
			            L.behavior, L.n, L.expect == EXP_ALLOC ? "ALLOCATION_FAILED" : "DEVICE_LOST", L.isolate ? 1 : 0,
			            L.cells);
		return 0;
	}
	// The group's routes.
	auto in_group = [&](const HookLeg& L) {
		if (!only.empty()) return only == L.name;
		if (L.isolate) return false;  // isolated legs run only through --only
		const int r = L.route;
		if (group == "qwen3") return r == RT_PROMPT || r == RT_DECODE || r == RT_SEQCREATE || r == RT_MAP ||
		                             r == RT_RESTORE || r == RT_CTX;
		if (group == "r15") return r == RT_DEVLOGITS || r == RT_ADAPTER || r == RT_MAPHEAD;
		if (group == "g5") return r == RT_SCHEMA;
		return false;
	};
	std::vector<const HookLeg*> legs;
	for (const HookLeg& L : kLegs)
		if (in_group(L)) legs.push_back(&L);
	if (legs.empty()) return SetupFail(cell, "no legs selected (--group=qwen3|r15|g5, or --only=<leg>)");
	if (!Hooks(cell)) return SetupFail(cell, "hooks");
	bool need_r15 = false, need_g5 = false, first_in_process = false;
	for (const HookLeg* L : legs) {
		need_r15 |= L->route == RT_DEVLOGITS || L->route == RT_ADAPTER || L->route == RT_MAPHEAD;
		need_g5 |= L->route == RT_SCHEMA;
		first_in_process |= L->route == RT_PROMPT_FIRST;
	}
	RouteFx fx;
	const uint32_t flags = need_r15 ? SSLM_GPU_RESIDENCY_HEAD_ON_DEVICE : 0u;
	const std::string path = need_r15 ? Opt("r15") : need_g5 ? Opt("g5") : Opt("qwen3");
	// A first-in-process leg must see the pipeline caches empty, so no prefill may run before it: its
	// reference frame comes from a separate process (--ref=FILE, written by `refgen`).
	if (!fx.p.Open(path, 5, 64, flags, /*reference=*/!first_in_process && !need_g5)) return SetupFail(cell, "fixture");
	if (first_in_process) {
		std::vector<uint8_t> b;
		if (!ReadAll(Opt("ref"), &b) || !FrameFromBytes(b, fx.p.s1.hidden, &fx.p.ref))
			return SetupFail(cell, "--ref frame (run `refgen` first)");
	}
	if (need_g5) {
		Chain ch;
		if (!ch.Build(fx.p.art.bytes, kSchemaName)) return SetupFail(cell, "schema parse");
		fx.chain = ch.Walk(5);
		fx.schema_idx = SslmGpuSchemaLookupForG5Bridge(fx.p.s1.model, kSchemaName);
		SchemaOut o = SchemaLoop(fx.p.s1.ctx, fx.p.s1.seq, fx.schema_idx, fx.chain, &fx.p.ref);
		SchemaOut o2 = SchemaLoop(fx.p.s2.ctx, fx.p.s2.seq, fx.schema_idx, fx.chain, &fx.p.f);
		if (fx.chain.size() != 5 || o.loop.First() != OK || o2.loop.First() != OK || !fx.p.f.Same(fx.p.ref))
			return SetupFail(cell, "schema reference");
	}
	if (need_r15 && !fx.adapter.Load(Opt("adapter"))) return SetupFail(cell, "--adapter");
	fx.dec.prompt = QwenPrompt(5);
	bool has_decode = false, has_restore = false, has_devlogits = false;
	for (const HookLeg* L : legs) {
		has_decode |= L->route == RT_DECODE;
		has_restore |= L->route == RT_RESTORE;
		has_devlogits |= L->route == RT_DEVLOGITS;
	}
	if (has_decode) {
		SslmGpuStatus st;
		Ref r2;
		if (!fx.dec.Run(fx.p.s1, &fx.ref_decode, &st) || !fx.dec.Run(fx.p.s2, &r2, &st) ||
		    r2.token != fx.ref_decode.token || r2.blob != fx.ref_decode.blob)
			return SetupFail(cell, "decode reference");
	}
	if (has_restore) {
		if (sslm_gpu_seq_reset(fx.p.s1.ctx, fx.p.s1.seq) != OK ||
		    SslmGpuSeqPrefillPromptForG5Bridge(fx.p.s1.ctx, fx.p.s1.seq, fx.p.toks.data(), 5, 64) != OK)
			return SetupFail(cell, "restore blob");
		fx.blob = SaveBlob(fx.p.s1.ctx, fx.p.s1.seq);
		if (fx.blob.empty()) return SetupFail(cell, "restore blob");
	}
	if (has_devlogits) {
		if (!fx.Pre(RT_DEVLOGITS)) return SetupFail(cell, "device-logits prefill");
		if (SslmGpuSeqFinishTokenForG5Bridge(fx.p.s1.ctx, fx.p.s1.seq, &fx.ref_devlogits.token) != OK)
			return SetupFail(cell, "device-logits reference");
	}
	for (const HookLeg* Lp : legs) {
		const HookLeg& L = *Lp;
		// Resolve the site against a clean trace of this very call (also the leg's clean pre-call).
		uint32_t ordinal = 0;
		std::string pre_desc = "-";
		if (L.route == RT_PROMPT_FIRST) {
			ordinal = L.sel.nth;  // nothing ran before it in this process
		} else {
			if (!fx.Pre(L.route)) {
				v.Invalid();
				std::printf("%s %s pre-call not clean -> INVALID\n", cell, L.name);
				continue;
			}
			TraceStart();
			fx.Call(L.route);
			const std::vector<TraceEv> t = TraceStop();
			pre_desc = fx.desc;
			if (fx.status != OK) {
				v.Invalid();
				std::printf("%s %s clean trace call returned %s -> INVALID\n", cell, L.name, fx.desc.c_str());
				continue;
			}
			ordinal = Resolve(L.sel, t);
		}
		if (ordinal == 0) {
			v.Invalid();
			std::printf("%s %s selector did not resolve (%s) -> INVALID\n", cell, L.name, MethodName(L.sel.method));
			continue;
		}
		if (!fx.Pre(L.route)) {
			v.Invalid();
			std::printf("%s %s pre-call not clean -> INVALID\n", cell, L.name);
			continue;
		}
		HookArmCall(L.sel.method, ordinal, L.hr, L.behavior, L.n);
		fx.Call(L.route);
		const uint32_t fired = g_arm.fired.load();
		HookDisarm();
		if (fired == 0) {
			v.Invalid();
			std::printf("%s %s %s#%u never fired -> INVALID\n", cell, L.name, MethodName(L.sel.method), ordinal);
			continue;
		}
		const bool own_ok = Judge(L, fx);
		const std::string own = fx.desc;
		// After-calls. Rule 2 and rule 3: the second context's next call is clean (the stuck-list
		// oracle), and the same handle's next call is clean. Rule 0 (A2): the documented terminal case --
		// the same context's next call fails too. Other rule-0 legs (a fence wait or an unretried Signal
		// that cannot be confirmed) assert only the status.
		SslmGpuStatus s2 = OK, s1 = OK;
		bool after_ok = true;
		std::string after;
		if (L.rule == RULE2 || L.rule == RULE3) {
			const bool a2 = fx.After2(L.route, &s2);
			const bool a1 = fx.After1(L.route, &s1);
			after_ok = a2 && a1;
			after = std::string(" after2=") + St(s2) + (a2 ? "" : "(!)") + " after1=" + St(s1) + (a1 ? "" : "(!)");
		} else if (L.n == 2) {
			const bool a1 = !fx.After1(L.route, &s1);
			after_ok = a1;
			after = std::string(" after1(must fail)=") + St(s1) + (a1 ? "" : "(!)");
		}
		const bool conforms = own_ok && after_ok;
		v.Leg(conforms);
		std::printf("%s %s route=%s site=%s#%u hr=%s behavior=%s n=%u rule=%u want=%s own=%s%s cells=%s -> %s\n", cell,
		            L.name, RouteName(L.route), MethodName(L.sel.method), ordinal, Hex(static_cast<uint32_t>(L.hr)).c_str(),
		            L.behavior == HB_CALL_THEN_FAIL ? "call-then-fail" : "fail-no-call", L.n, static_cast<unsigned>(L.rule),
		            L.expect == EXP_ALLOC ? "ALLOCATION_FAILED" : "DEVICE_LOST", own.c_str(), after.c_str(), L.cells,
		            conforms ? "PASS" : "FAIL");
		std::fflush(stdout);
	}
	return v.Finish();
}
}  // namespace

// =============================================================================================
// refgen: a reference frame from a process of its own, for the first-in-process legs.
// =============================================================================================
namespace {
int CellRefgen(const char* cell) {
	Pair p;
	if (!p.Open(Opt("qwen3"), static_cast<int>(OptInt("len", 5)))) return SetupFail(cell, "fixture");
	const std::vector<uint8_t> b = FrameBytes(p.ref);
	if (!WriteFileBytes(Opt("ref"), b.data(), b.size())) return SetupFail(cell, "--ref write");
	std::printf("%s wrote %zu bytes to %s\nVERDICT %s GREEN\n", cell, b.size(), Opt("ref").c_str(), cell);
	return 0;
}
}  // namespace

// =============================================================================================
// R7.4: one .cso removed before the process's first prefill -> SSLM_DEVICE_LOST, never
// SSLM_GPU_ALLOCATION_FAILED (the E-3 split; te266 C10(d)).
// =============================================================================================
namespace {
int CellCsoRemoved(const char* cell) {
	Verdict v(cell);
	char exe[MAX_PATH] = {};
	GetModuleFileNameA(nullptr, exe, MAX_PATH);
	std::string dir(exe);
	dir = dir.substr(0, dir.find_last_of("\\/"));
	const std::string src = dir + "\\shaders";
	const std::string dst = Opt("tmp", "D:\\_te425\\tmp") + "\\cso_removed_" + std::to_string(GetCurrentProcessId());
	CreateDirectoryA(Opt("tmp", "D:\\_te425\\tmp").c_str(), nullptr);
	CreateDirectoryA(dst.c_str(), nullptr);
	const std::string removed = Opt("remove", "attn_norm_site");
	WIN32_FIND_DATAA fd;
	HANDLE h = FindFirstFileA((src + "\\*.cso").c_str(), &fd);
	if (h == INVALID_HANDLE_VALUE) return SetupFail(cell, "no shaders beside the executable");
	uint32_t copied = 0;
	bool saw_removed = false;
	do {
		const std::string name = fd.cFileName;
		if (name == removed + ".cso") {
			saw_removed = true;
			continue;
		}
		if (CopyFileA((src + "\\" + name).c_str(), (dst + "\\" + name).c_str(), FALSE)) ++copied;
	} while (FindNextFileA(h, &fd));
	FindClose(h);
	if (!saw_removed) return SetupFail(cell, "the named .cso is not in the shader set");
	Artifact a;
	if (!a.Load(Opt("qwen3"))) return SetupFail(cell, "--qwen3");
	GpuContextConfig cc{};
	cc.shader_dir = dst.c_str();
	SslmGpuContext* ctx = nullptr;
	SslmGpuModelHandle* model = nullptr;
	SslmGpuSequenceHandle* seq = nullptr;
	if (sslm_gpu_context_create(cc, &ctx) != OK || sslm_gpu_model_map(ctx, &a.view, GpuResidencyConfig{}, &model) != OK ||
	    sslm_gpu_seq_create(ctx, model, 64, &seq) != OK)
		return SetupFail(cell, "stack with shader_dir override");
	uint32_t hidden = 0;
	sslm_gpu_model_hidden_size(model, &hidden);
	Frame f;
	f.Size(hidden);
	LoopOut o = PromptLoop(ctx, seq, QwenPrompt(5), &f);
	const bool conforms = o.First() == DEVICE_LOST && o.FirstStage() == 1 && o.read == HIDDEN_UNAVAILABLE;
	v.Leg(conforms);
	std::printf("%s R7.4 %s.cso removed (%u copied): prefill=%s read=%s want DEVICE_LOST, never ALLOCATION_FAILED -> %s\n",
	            cell, removed.c_str(), copied, St(o.prefill), St(o.read), conforms ? "PASS" : "FAIL");
	return v.Finish();
}
}  // namespace

// =============================================================================================
// E-7 (plan Sec3.5 R1): the process's submission device, first touched by each route, in a fresh
// process. kind=alloc: its first-time setup runs out of memory (CreateCommittedResource of the
// timestamp readback, E_OUTOFMEMORY) -> SSLM_GPU_ALLOCATION_FAILED, and the repeat is SSLM_OK and
// byte-equal. kind=other: its first CreateCommandQueue returns E_FAIL -> SSLM_DEVICE_LOST on every call.
// =============================================================================================
bool Te425DirectGetDeviceFirstTouch(bool* threw, bool* available);  // te425_getdevice.cpp

namespace {
struct FirstTouch {
	Artifact art;
	Stack s;
	std::vector<uint8_t> blob;  // restore route
	int32_t schema_idx = -1;
	std::vector<int32_t> chain;
	SslmGpuSequenceHandle* bseq = nullptr;
	// One call of the route; returns its status and its output bytes.
	SslmGpuStatus Call(const std::string& route, std::vector<uint8_t>* out) {
		out->clear();
		if (route == "prefill") {
			Frame f;
			f.Size(s.hidden);
			LoopOut o = PromptLoop(s.ctx, s.seq, QwenPrompt(5), &f);
			if (o.First() == OK) *out = FrameBytes(f);
			return o.First();
		}
		if (route == "schema") {
			Frame f;
			f.Size(s.hidden);
			SchemaOut o = SchemaLoop(s.ctx, s.seq, schema_idx, chain, &f);
			if (o.loop.First() == OK) *out = FrameBytes(f);
			return o.loop.First();
		}
		if (route == "decode") {
			if (sslm_gpu_seq_reset(s.ctx, s.seq) != OK) return DEVICE_LOST;
			const StepOut o = DecodeStep(s.ctx, s.seq);
			if (o.First() != OK) return o.First();
			int32_t tok = -1;
			const SslmGpuStatus fin = SslmGpuSeqFinishTokenForG5Bridge(s.ctx, s.seq, &tok);
			if (fin != OK) return fin;
			*out = SaveBlob(s.ctx, s.seq);
			out->insert(out->end(), reinterpret_cast<uint8_t*>(&tok), reinterpret_cast<uint8_t*>(&tok) + 4);
			return OK;
		}
		if (route == "batch") {
			if (sslm_gpu_seq_reset(s.ctx, s.seq) != OK || sslm_gpu_seq_embed_token(s.ctx, s.seq, kStepToken) != OK)
				return DEVICE_LOST;
			SslmGpuSequenceHandle* seqs[1] = {s.seq};
			SslmGpuStatus st = DEVICE_LOST;
			const SslmGpuStatus call = sslm_decode_step_batch_gpu(s.ctx, seqs, nullptr, 1, 0xFFFFFFFFu, &st);
			if (call != OK) return call;
			if (st != OK) return st;
			int32_t tok = -1;
			const SslmGpuStatus fin = SslmGpuSeqFinishTokenForG5Bridge(s.ctx, s.seq, &tok);
			if (fin != OK) return fin;
			*out = SaveBlob(s.ctx, s.seq);
			out->insert(out->end(), reinterpret_cast<uint8_t*>(&tok), reinterpret_cast<uint8_t*>(&tok) + 4);
			return OK;
		}
		if (route == "restore") {
			SslmGpuSequenceHandle* h = nullptr;
			const SslmGpuStatus st = sslm_gpu_seq_restore(s.ctx, s.model, blob.data(), blob.size(), &h);
			if (st == OK && h) *out = SaveBlob(s.ctx, h);
			if (h) sslm_gpu_seq_release(s.ctx, h);
			return st;
		}
		return DEVICE_LOST;
	}
};

int CellFirstTouch(const char* cell, bool refgen) {
	Verdict v(cell);
	const std::string route = Opt("route");
	const std::string kind = Opt("kind", "alloc");
	if (!Hooks(cell)) return SetupFail(cell, "hooks");
	FirstTouch ft;
	const bool schema = route == "schema";
	if (!ft.art.Load(schema ? Opt("g5") : Opt("qwen3"))) return SetupFail(cell, "artifact");
	// Context create, map and sequence create use the context's own device, never the process's
	// submission device (plan Sec5.1 truth argument; gpu_1p0.cpp context create/map/seq create).
	if (!ft.s.Open(ft.art, 64)) return SetupFail(cell, "stack");
	if (schema) {
		Chain ch;
		if (!ch.Build(ft.art.bytes, kSchemaName)) return SetupFail(cell, "schema parse");
		ft.chain = ch.Walk(5);
		ft.schema_idx = SslmGpuSchemaLookupForG5Bridge(ft.s.model, kSchemaName);
		if (ft.chain.size() != 5 || ft.schema_idx < 0) return SetupFail(cell, "schema chain");
	}
	if (route == "restore") {
		ft.blob = SaveBlob(ft.s.ctx, ft.s.seq);  // host-only (SaveGpuSequenceState): no device touched
		if (ft.blob.empty()) return SetupFail(cell, "blob");
	}
	const std::string ref_path = Opt("ref");
	if (refgen) {
		std::vector<uint8_t> out;
		const SslmGpuStatus st = ft.Call(route, &out);
		if (st != OK || out.empty() || !WriteFileBytes(ref_path, out.data(), out.size()))
			return SetupFail(cell, "clean first touch");
		std::printf("%s route=%s clean first touch OK, %zu bytes -> %s\nVERDICT %s GREEN\n", cell, route.c_str(),
		            out.size(), ref_path.c_str(), cell);
		return 0;
	}
	std::vector<uint8_t> ref;
	if (!ReadAll(ref_path, &ref)) return SetupFail(cell, "--ref (run e7ref first)");
	// alloc: the setup's buffer allocation (the timestamp readback) runs out of memory.
	// queue_oom: the setup's CreateCommandQueue returns E_OUTOFMEMORY (plan Sec3.2 E-1's "device-object
	//   creation in Init": rule 2, through E-7's retry) -- the same expectation as alloc.
	// other: the setup's CreateCommandQueue returns E_FAIL -- a setup failure for a reason other than
	//   memory, SSLM_DEVICE_LOST on every call (E-7's `Other`).
	const bool alloc_kind = kind == "alloc" || kind == "queue_oom";
	if (kind == "alloc") {
		HookArmCall(M_CCR, 1, E_OUTOFMEMORY, HB_FAIL_NO_CALL, 1, D3D12_HEAP_TYPE_READBACK, kTsWidth);
	} else if (kind == "queue_oom") {
		HookArmCall(M_CQUEUE, 1, E_OUTOFMEMORY);
	} else if (kind == "other") {
		HookArmCall(M_CQUEUE, 1, E_FAIL);
	} else {
		return SetupFail(cell, "--kind=alloc|queue_oom|other");
	}
	std::vector<uint8_t> out;
	const SslmGpuStatus armed = ft.Call(route, &out);
	const uint32_t fired = g_arm.fired.load();
	HookDisarm();
	if (fired == 0) {
		v.Invalid();
		std::printf("%s route=%s kind=%s: the submission device's setup was never reached -> INVALID\n", cell,
		            route.c_str(), kind.c_str());
		return v.Finish();
	}
	std::vector<uint8_t> o1, o2;
	const SslmGpuStatus r1 = ft.Call(route, &o1);
	const SslmGpuStatus r2 = ft.Call(route, &o2);
	bool conforms;
	if (alloc_kind) {
		conforms = armed == ALLOC_FAILED && r1 == OK && o1 == ref && r2 == OK && o2 == ref;
	} else {
		conforms = armed == DEVICE_LOST && r1 == DEVICE_LOST && r2 == DEVICE_LOST;
	}
	v.Leg(conforms);
	std::printf("%s route=%s kind=%s armed=%s repeat1=%s%s repeat2=%s%s want=%s -> %s\n", cell, route.c_str(),
	            kind.c_str(), St(armed), St(r1), r1 == OK ? (o1 == ref ? " equal" : " DIFFERS") : "", St(r2),
	            r2 == OK ? (o2 == ref ? " equal" : " DIFFERS") : "",
	            alloc_kind ? "ALLOCATION_FAILED then OK+equal" : "DEVICE_LOST on every call",
	            conforms ? "PASS" : "FAIL");
	return v.Finish();
}

// E-7, a first touch that is not an entry point: harness::GetDevice() called first, as the
// SuperSLM-Unreal plugin's diagnostic does, with the setup's allocation failing.
int CellFirstTouchDirect(const char* cell) {
	Verdict v(cell);
	if (!Hooks(cell)) return SetupFail(cell, "hooks");
	FirstTouch ft;
	if (!ft.art.Load(Opt("qwen3")) || !ft.s.Open(ft.art, 64)) return SetupFail(cell, "stack");
	std::vector<uint8_t> ref;
	if (!ReadAll(Opt("ref"), &ref)) return SetupFail(cell, "--ref (run e7ref --route=prefill first)");
	HookArmCall(M_CCR, 1, E_OUTOFMEMORY, HB_FAIL_NO_CALL, 1, D3D12_HEAP_TYPE_READBACK, kTsWidth);
	bool threw = false, available = true;
	Te425DirectGetDeviceFirstTouch(&threw, &available);
	const uint32_t fired = g_arm.fired.load();
	HookDisarm();
	if (fired == 0) {
		v.Invalid();
		std::printf("%s GetDevice() did not reach its setup allocation -> INVALID\n", cell);
		return v.Finish();
	}
	std::vector<uint8_t> out;
	const SslmGpuStatus next = ft.Call("prefill", &out);
	const bool conforms = !threw && !available && next == OK && out == ref;
	v.Leg(conforms);
	std::printf("%s GetDevice() first, setup allocation fails: threw=%d available=%d; next prefill=%s%s "
	            "want no-throw, available=0, then OK+equal -> %s\n",
	            cell, threw ? 1 : 0, available ? 1 : 0, St(next), next == OK ? (out == ref ? " equal" : " DIFFERS") : "",
	            conforms ? "PASS" : "FAIL");
	return v.Finish();
}
}  // namespace

// =============================================================================================
// R14: restore as the recovery. Save, fault a mid-prefill recording-window site (a partial
// commit), restore on the same context, prefill the remainder; the read equals the clean reference.
// =============================================================================================
namespace {
int CellRestoreRecovery(const char* cell) {
	Verdict v(cell);
	if (!Hooks(cell)) return SetupFail(cell, "hooks");
	Pair p;
	if (!p.Open(Opt("qwen3"), 40, 64, 0, /*reference=*/false)) return SetupFail(cell, "fixture");
	const std::vector<int32_t> all = QwenPrompt(40);
	const std::vector<int32_t> head(all.begin(), all.begin() + 20), tail(all.begin() + 20, all.end());
	auto prefill = [&](SslmGpuSequenceHandle* s, const std::vector<int32_t>& t) {
		return SslmGpuSeqPrefillPromptForG5Bridge(p.s1.ctx, s, t.data(), static_cast<int32_t>(t.size()), 64);
	};
	auto read = [&](SslmGpuSequenceHandle* s, Frame* f) {
		return sslm_gpu_seq_read_prefill_final_hidden(p.s1.ctx, s, f->codes.data(), f->codes.size(), &f->required, &f->m,
		                                              &f->e);
	};
	// The clean reference: head then tail on one sequence.
	if (sslm_gpu_seq_reset(p.s1.ctx, p.s1.seq) != OK || prefill(p.s1.seq, head) != OK || prefill(p.s1.seq, tail) != OK)
		return SetupFail(cell, "reference prefill");
	p.ref.status = read(p.s1.seq, &p.ref);
	if (p.ref.status != OK) return SetupFail(cell, "reference read");
	const std::string src = Opt("src", "seam");
	// Site: inside the third sub-chunk's recording window of the tail prefill (two sub-chunks already
	// committed -- a partial commit), located on a clean trace of the tail prefill.
	auto setup_head = [&]() {
		return sslm_gpu_seq_reset(p.s1.ctx, p.s1.seq) == OK && prefill(p.s1.seq, head) == OK;
	};
	if (!setup_head()) return SetupFail(cell, "head");
	std::vector<uint8_t> blob = SaveBlob(p.s1.ctx, p.s1.seq);
	uint32_t site = 0;
	if (src == "seam") {
		TraceStart();
		SslmGpuAllocCounterResetForTest();
		prefill(p.s1.seq, tail);
		const std::vector<TraceEv> t = TraceStop();
		uint32_t k = 0;
		for (const TraceEv& e : t)
			if (e.method == M_CCR) {
				++k;
				if (e.window == 3) {
					site = k;
					break;
				}
			}
	} else if (src == "new") {
		NewCountOnly(true);
		prefill(p.s1.seq, tail);
		NewStop();
		for (uint32_t k = 1; k <= g_new.count.load() && k < kMaxSites; ++k)
			if (g_labels[k].window == 3) {
				site = k;
				break;
			}
	} else {  // close: the third sub-chunk's tail Close, driver behaviour (A), E_OUTOFMEMORY
		site = 3;
	}
	if (site == 0) return SetupFail(cell, "no site in the third window");
	if (!setup_head()) return SetupFail(cell, "head");
	blob = SaveBlob(p.s1.ctx, p.s1.seq);
	if (blob.empty()) return SetupFail(cell, "save");
	SslmGpuStatus faulted;
	if (src == "seam") {
		SslmGpuAllocCounterResetForTest();
		ArmGpuAllocFaultAtOccurrence(site, E_OUTOFMEMORY);
		faulted = prefill(p.s1.seq, tail);
		ArmGpuAllocFaultAtOccurrence(0, S_OK);
	} else if (src == "new") {
		NewArm(site, NewFault::BadAlloc);
		faulted = prefill(p.s1.seq, tail);
		NewStop();
	} else {
		HookArmCall(M_CLOSE, site, E_OUTOFMEMORY, HB_FAIL_NO_CALL, 1);
		faulted = prefill(p.s1.seq, tail);
		HookDisarm();
	}
	const int64_t committed = *SslmGpuSeqHandleContextLengthForBench(p.s1.seq);
	SslmGpuSequenceHandle* restored = nullptr;
	const SslmGpuStatus rs = sslm_gpu_seq_restore(p.s1.ctx, p.s1.model, blob.data(), blob.size(), &restored);
	SslmGpuStatus cont = DEVICE_LOST, rd = DEVICE_LOST;
	bool equal = false;
	if (rs == OK && restored) {
		cont = prefill(restored, tail);
		if (cont == OK) {
			rd = read(restored, &p.f);
			equal = rd == OK && p.f.codes == p.ref.codes && p.f.m == p.ref.m && p.f.e == p.ref.e;
		}
		sslm_gpu_seq_release(p.s1.ctx, restored);
	}
	const bool conforms = faulted == ALLOC_FAILED && rs == OK && cont == OK && equal;
	v.Leg(conforms);
	std::printf("%s src=%s site=%u faulted=%s (committed %lld of 40, head 20) restore=%s continue=%s read=%s equal=%d "
	            "want ALLOCATION_FAILED, restore OK, continue OK, equal -> %s\n",
	            cell, src.c_str(), site, St(faulted), static_cast<long long>(committed), St(rs), St(cont), St(rd),
	            equal ? 1 : 0, conforms ? "PASS" : "FAIL");
	return v.Finish();
}
}  // namespace

// =============================================================================================
// R15: the entry-point census. Disposition 3: a counted clean call reads 0 on both fault sources
// (and touches no hooked D3D12 method). Disposition 2: an operator-new (and HRESULT) sweep.
// =============================================================================================
namespace {
struct Counts {
	uint32_t news = 0, seam = 0, d3d12 = 0;
};
template <typename Fn>
Counts CountCall(Fn&& fn) {
	uint32_t before = 0;
	for (int m = 1; m < M_COUNT; ++m) before += g_calls[m].load();
	SslmGpuAllocCounterResetForTest();
	NewCountOnly(false);
	fn();
	NewStop();
	Counts c;
	c.news = g_new.count.load();
	c.seam = SslmGpuAllocCountForTest();
	uint32_t after = 0;
	for (int m = 1; m < M_COUNT; ++m) after += g_calls[m].load();
	c.d3d12 = after - before;
	return c;
}

int CellEntryZero(const char* cell) {
	Verdict v(cell);
	if (!Hooks(cell)) return SetupFail(cell, "hooks");
	const std::string set = Opt("set", "qwen3");
	// Runs the counted call itself, so the status it reports is the one this call returned.
	auto report = [&](const char* name, const std::function<SslmGpuStatus()>& call) {
		SslmGpuStatus st = DEVICE_LOST;
		const Counts c = CountCall([&] { st = call(); });
		const bool zero = st == OK && c.news == 0 && c.seam == 0 && c.d3d12 == 0;
		v.Leg(zero);
		std::printf("%s %s status=%s operator_new=%u device_allocations=%u d3d12_calls=%u -> %s\n", cell, name, St(st),
		            c.news, c.seam, c.d3d12, zero ? "PASS (disposition 3)" : "FAIL (moves to disposition 2)");
	};
	Artifact a;
	if (set == "qwen3") {
		if (!a.Load(Opt("qwen3"))) return SetupFail(cell, "--qwen3");
		Stack s;
		if (!s.Open(a, 64)) return SetupFail(cell, "stack");
		uint32_t hidden = 0;
		report("sslm_gpu_model_hidden_size", [&] { return sslm_gpu_model_hidden_size(s.model, &hidden); });
		sslm_parallel_for pf{};
		report("sslm_gpu_context_set_host_parallel_for", [&] { return sslm_gpu_context_set_host_parallel_for(s.ctx, &pf); });
		report("sslm_gpu_seq_reset", [&] { return sslm_gpu_seq_reset(s.ctx, s.seq); });
		size_t n = 0;
		(void)sslm_gpu_seq_save(s.ctx, s.seq, nullptr, &n);
		std::vector<uint8_t> blob(n);
		report("sslm_gpu_seq_save", [&] { return sslm_gpu_seq_save(s.ctx, s.seq, blob.data(), &n); });
		SslmGpuSequenceHandle* extra = nullptr;
		if (sslm_gpu_seq_create(s.ctx, s.model, 64, &extra) != OK) return SetupFail(cell, "extra sequence");
		report("sslm_gpu_seq_release", [&] { return sslm_gpu_seq_release(s.ctx, extra); });
		SslmGpuModelHandle* m2 = nullptr;
		if (sslm_gpu_model_map(s.ctx, &a.view, GpuResidencyConfig{}, &m2) != OK) return SetupFail(cell, "extra map");
		report("sslm_gpu_model_unmap", [&] { return sslm_gpu_model_unmap(s.ctx, m2); });
		SslmGpuContext* c2 = nullptr;
		if (sslm_gpu_context_create(GpuContextConfig{}, &c2) != OK) return SetupFail(cell, "extra context");
		report("sslm_gpu_context_destroy", [&] { return sslm_gpu_context_destroy(c2); });
		s.Close();
	} else if (set == "adapter") {
		if (!a.Load(Opt("r15"))) return SetupFail(cell, "--r15");
		Artifact ad;
		if (!ad.Load(Opt("adapter"))) return SetupFail(cell, "--adapter");
		Stack s;
		if (!s.Open(a, 64)) return SetupFail(cell, "stack");
		SslmGpuAdapterHandle* h = nullptr;
		if (sslm_gpu_adapter_map(s.ctx, s.model, &ad.view, &h) != OK) return SetupFail(cell, "adapter map");
		report("sslm_gpu_seq_bind_adapter", [&] { return sslm_gpu_seq_bind_adapter(s.ctx, s.seq, h); });
		(void)sslm_gpu_seq_bind_adapter(s.ctx, s.seq, nullptr);
		report("sslm_gpu_adapter_unmap", [&] { return sslm_gpu_adapter_unmap(s.ctx, h); });
		s.Close();
	} else if (set == "g5") {
		if (!a.Load(Opt("g5"))) return SetupFail(cell, "--g5");
		Stack s;
		if (!s.Open(a, 64)) return SetupFail(cell, "stack");
		const int32_t idx = SslmGpuSchemaLookupForG5Bridge(s.model, kSchemaName);
		if (idx < 0) return SetupFail(cell, "schema lookup");
		report("SslmGpuSeqSetSchemaForG5Bridge", [&] { return SslmGpuSeqSetSchemaForG5Bridge(s.ctx, s.seq, idx); });
		int32_t acc = -1, bound = -1;
		report("SslmGpuSeqSchemaAcceptingForG5Bridge",
		       [&] { return SslmGpuSeqSchemaAcceptingForG5Bridge(s.ctx, s.seq, &acc); });
		report("SslmGpuSeqSchemaBoundForG5Bridge", [&] { return SslmGpuSeqSchemaBoundForG5Bridge(s.ctx, s.seq, &bound); });
		s.Close();
	} else if (set == "disposition2") {
		// The commissioning must-reject (plan Sec3.5 R15 item 2, the planner's census): three entry
		// points the plan reads as allocating host memory. The zero-count census must refuse all three.
		if (!a.Load(Opt("qwen3"))) return SetupFail(cell, "--qwen3");
		Stack s;
		if (!s.Open(a, 64)) return SetupFail(cell, "stack");
		const std::vector<int32_t> t = QwenPrompt(5);
		report("sslm_gpu_seq_embed_token", [&] { return sslm_gpu_seq_embed_token(s.ctx, s.seq, kStepToken); });
		if (sslm_gpu_seq_reset(s.ctx, s.seq) != OK ||
		    SslmGpuSeqPrefillPromptForG5Bridge(s.ctx, s.seq, t.data(), 5, 64) != OK)
			return SetupFail(cell, "prefill");
		Frame f;
		f.Size(s.hidden);
		report("sslm_gpu_seq_read_prefill_final_hidden", [&] {
			return sslm_gpu_seq_read_prefill_final_hidden(s.ctx, s.seq, f.codes.data(), f.codes.size(), &f.required,
			                                              &f.m, &f.e);
		});
		int32_t tok = -1;
		report("SslmGpuSeqFinishTokenForG5Bridge", [&] { return SslmGpuSeqFinishTokenForG5Bridge(s.ctx, s.seq, &tok); });
		s.Close();
	} else {
		return SetupFail(cell, "--set=qwen3|adapter|g5|disposition2");
	}
	return v.Finish();
}

// Disposition 2: every operator-new site of one call.
int CellEntrySweep(const char* cell) {
	Verdict v(cell);
	if (!Hooks(cell)) return SetupFail(cell, "hooks");
	const std::string call = Opt("call");
	const bool r15 = call == "finish" || call == "finish_devlogits";
	Pair p;
	const uint32_t flags = call == "finish_devlogits" ? SSLM_GPU_RESIDENCY_HEAD_ON_DEVICE : 0u;
	if (!p.Open(r15 ? Opt("r15") : Opt("qwen3"), 5, 64, flags)) return SetupFail(cell, "fixture");
	int32_t tok = -1, ref_tok = -1;
	SslmGpuStatus own = OK;
	std::vector<int8_t> embed_ref;
	auto pre = [&](Stack& s) {
		if (call == "embed") return sslm_gpu_seq_reset(s.ctx, s.seq) == OK;
		return sslm_gpu_seq_reset(s.ctx, s.seq) == OK &&
		       SslmGpuSeqPrefillPromptForG5Bridge(s.ctx, s.seq, p.toks.data(), static_cast<int32_t>(p.toks.size()), 64) == OK;
	};
	auto run = [&](Stack& s) -> SslmGpuStatus {
		if (call == "embed") return sslm_gpu_seq_embed_token(s.ctx, s.seq, kStepToken);
		if (call == "read")
			return sslm_gpu_seq_read_prefill_final_hidden(s.ctx, s.seq, p.f.codes.data(), p.f.codes.size(), &p.f.required,
			                                              &p.f.m, &p.f.e);
		tok = -1;
		return SslmGpuSeqFinishTokenForG5Bridge(s.ctx, s.seq, &tok);
	};
	// The call's own observable output, for the after-call's equality.
	auto output_ok = [&](Stack& s) {
		if (call == "embed") {
			const int8_t* c = SslmGpuSeqHandleHiddenCodesForBench(s.seq);
			return std::vector<int8_t>(c, c + s.hidden) == embed_ref;
		}
		if (call == "read") return p.f.Same(p.ref);
		return tok == ref_tok;
	};
	if (!pre(p.s1) || run(p.s1) != OK) return SetupFail(cell, "reference call");
	ref_tok = tok;
	if (call == "embed") {
		const int8_t* c = SslmGpuSeqHandleHiddenCodesForBench(p.s1.seq);
		embed_ref.assign(c, c + p.s1.hidden);
	}
	SweepHooks h;
	h.pre = [&] { return pre(p.s1); };
	h.armed = [&] {
		g_phase.store(PH_CALL);
		own = run(p.s1);
		g_phase.store(PH_NONE);
	};
	h.judge = [&](std::string* d) {
		*d = St(own);
		return own == ALLOC_FAILED;
	};
	h.after2 = [&](SslmGpuStatus* s) { return p.After2(s); };
	h.after1 = [&](SslmGpuStatus* s) {
		if (!pre(p.s1)) {
			*s = DEVICE_LOST;
			return false;
		}
		*s = run(p.s1);
		return *s == OK && output_ok(p.s1);
	};
	pre(p.s1);
	const uint32_t K = CountSites(Src::New, [&] { own = run(p.s1); }, nullptr);
	std::printf("%s call=%s operator-new sites=%u counting-pass=%s\n", cell, call.c_str(), K, St(own));
	SweepTally t;
	Sweep(cell, Src::New, NewFault::BadAlloc, 0, K, {}, 1, K, nullptr, h, v, t);
	t.Print(cell);
	return v.Finish();
}
}  // namespace

// =============================================================================================
// R1 (operator new): every operator-new site of a handle-creating call. The call returns
// SSLM_GPU_ALLOCATION_FAILED with its out-handle null; the same context's next call succeeds.
// =============================================================================================
namespace {
int CellCreateSweep(const char* cell) {
	Verdict v(cell);
	if (!Hooks(cell)) return SetupFail(cell, "hooks");
	const std::string call = Opt("call");
	const bool r15 = call == "map_head" || call == "adapter";
	Pair p;
	if (!p.Open(r15 ? Opt("r15") : Opt("qwen3"), 5)) return SetupFail(cell, "fixture");
	Artifact ad;
	if (call == "adapter" && !ad.Load(Opt("adapter"))) return SetupFail(cell, "--adapter");
	std::vector<uint8_t> blob;
	if (call == "restore") {
		blob = SaveBlob(p.s1.ctx, p.s1.seq);
		if (blob.empty()) return SetupFail(cell, "blob");
	}
	SslmGpuStatus own = OK;
	bool handle_leak = false;
	auto run = [&]() -> SslmGpuStatus {
		void* h = nullptr;
		SslmGpuStatus st = DEVICE_LOST;
		g_phase.store(PH_CALL);
		if (call == "context") {
			SslmGpuContext* c = nullptr;
			st = sslm_gpu_context_create(GpuContextConfig{}, &c);
			h = c;
		} else if (call == "map0" || call == "map_head") {
			GpuResidencyConfig rc{};
			rc.flags = call == "map_head" ? SSLM_GPU_RESIDENCY_HEAD_ON_DEVICE : 0u;
			SslmGpuModelHandle* m = nullptr;
			st = sslm_gpu_model_map(p.s1.ctx, &p.art.view, rc, &m);
			h = m;
		} else if (call == "adapter") {
			SslmGpuAdapterHandle* a = nullptr;
			st = sslm_gpu_adapter_map(p.s1.ctx, p.s1.model, &ad.view, &a);
			h = a;
		} else if (call == "seq_create") {
			SslmGpuSequenceHandle* s = nullptr;
			st = sslm_gpu_seq_create(p.s1.ctx, p.s1.model, 64, &s);
			h = s;
		} else if (call == "restore") {
			SslmGpuSequenceHandle* s = nullptr;
			st = sslm_gpu_seq_restore(p.s1.ctx, p.s1.model, blob.data(), blob.size(), &s);
			h = s;
		}
		g_phase.store(PH_NONE);
		const bool was_counting = g_new.counting.load();
		g_new.counting.store(false);
		handle_leak = st != OK && h != nullptr;
		if (h) {
			if (call == "context") sslm_gpu_context_destroy(static_cast<SslmGpuContext*>(h));
			else if (call == "map0" || call == "map_head")
				sslm_gpu_model_unmap(p.s1.ctx, static_cast<SslmGpuModelHandle*>(h));
			else if (call == "adapter") sslm_gpu_adapter_unmap(p.s1.ctx, static_cast<SslmGpuAdapterHandle*>(h));
			else sslm_gpu_seq_release(p.s1.ctx, static_cast<SslmGpuSequenceHandle*>(h));
		}
		g_new.counting.store(was_counting);
		return st;
	};
	// The full retry (a whole map, for the two map calls) is the costly part; --retry-every=N runs it
	// on every Nth site and on the last, and the second context's cheap loop on every site.
	const uint32_t retry_every = static_cast<uint32_t>(OptInt("retry-every", 1));
	uint32_t site_index = 0;
	uint32_t K = 0;
	SweepHooks h;
	h.pre = [&] { return true; };
	h.armed = [&] { own = run(); };
	h.judge = [&](std::string* d) {
		*d = std::string(St(own)) + (handle_leak ? " handle!=null" : "");
		return own == ALLOC_FAILED && !handle_leak;
	};
	h.after2 = [&](SslmGpuStatus* s) { return p.After2(s); };
	h.after1 = [&](SslmGpuStatus* s) {
		++site_index;
		if (retry_every > 1 && site_index % retry_every != 0 && site_index != K) {
			*s = OK;
			return true;
		}
		*s = run();
		return *s == OK;
	};
	K = CountSites(Src::New, [&] { own = run(); }, nullptr);
	std::printf("%s call=%s operator-new sites=%u (aligned %u) counting-pass=%s retry-every=%u\n", cell, call.c_str(), K,
	            g_new.aligned_seen.load(), St(own), retry_every);
	if (own != OK) return SetupFail(cell, "counting pass not clean");
	SweepTally t;
	Sweep(cell, Src::New, Opt("kind") == "length_error" ? NewFault::LengthError : NewFault::BadAlloc, 0, K, {}, static_cast<uint32_t>(OptInt("from", 1)),
	      static_cast<uint32_t>(OptInt("to", 0x7FFFFFFF)), nullptr, h, v, t);
	t.Print(cell);
	return v.Finish();
}

// R1 (the seam) for context create: its own device setup's one counted allocation.
int CellContextSeam(const char* cell) {
	Verdict v(cell);
	if (!Hooks(cell)) return SetupFail(cell, "hooks");
	Pair p;
	if (!p.Open(Opt("qwen3"), 5)) return SetupFail(cell, "fixture");
	auto create = [&](SslmGpuStatus* st) {
		SslmGpuContext* c = nullptr;
		*st = sslm_gpu_context_create(GpuContextConfig{}, &c);
		const bool leak = *st != OK && c != nullptr;
		if (c) sslm_gpu_context_destroy(c);
		return leak;
	};
	SslmGpuStatus st;
	SslmGpuAllocCounterResetForTest();
	create(&st);
	const uint32_t K = SslmGpuAllocCountForTest();
	std::printf("%s context create: %u counted device allocation(s), clean=%s\n", cell, K, St(st));
	if (st != OK || K == 0) return SetupFail(cell, "clean context create");
	for (uint32_t k = 1; k <= K; ++k) {
		const long hrs[] = {E_OUTOFMEMORY, static_cast<long>(DXGI_ERROR_DEVICE_REMOVED)};
		for (long hr : hrs) {
			SslmGpuAllocCounterResetForTest();
			ArmGpuAllocFaultAtOccurrence(k, hr);
			const bool leak = create(&st);
			ArmGpuAllocFaultAtOccurrence(0, S_OK);
			SslmGpuStatus retry;
			create(&retry);
			SslmGpuStatus s2;
			const bool a2 = p.After2(&s2);
			const SslmGpuStatus want = hr == E_OUTOFMEMORY ? ALLOC_FAILED : DEVICE_LOST;
			const bool conforms = st == want && !leak && retry == OK && a2;
			v.Leg(conforms);
			std::printf("%s k=%u hr=%s own=%s%s retry=%s after2=%s want=%s -> %s\n", cell, k,
			            Hex(static_cast<uint32_t>(hr)).c_str(), St(st), leak ? " handle!=null" : "", St(retry), St(s2),
			            St(want), conforms ? "PASS" : "FAIL");
		}
	}
	return v.Finish();
}
}  // namespace

// =============================================================================================
// R11's removed-rule drives (rule 1): E_OUTOFMEMORY with the classifier's removed seam armed gives
// SSLM_DEVICE_LOST. One leg per process (at v1.7.1 the seam is consulted only at the map-time
// bundle, so an unconsumed arm would outlive its leg).
// =============================================================================================
namespace {
int CellRemoved(const char* cell) {
	Verdict v(cell);
	if (!Hooks(cell)) return SetupFail(cell, "hooks");
	const std::string site = Opt("site");
	const bool r15 = site == "devlogits";
	Pair p;
	if (!p.Open(r15 ? Opt("r15") : Opt("qwen3"), 5, 64, r15 ? SSLM_GPU_RESIDENCY_HEAD_ON_DEVICE : 0u))
		return SetupFail(cell, "fixture");
	SslmGpuStatus own = OK;
	std::string what;
	auto arm_seam_first_in_window = [&](const std::function<void()>& call) -> uint32_t {
		TraceStart();
		SslmGpuAllocCounterResetForTest();
		call();
		const std::vector<TraceEv> t = TraceStop();
		uint32_t k = 0;
		for (const TraceEv& e : t)
			if (e.method == M_CCR) {
				++k;
				if (e.window >= 1) return k;
			}
		return 0;
	};
	auto prompt = [&] {
		LoopOut o = PromptLoop(p.s1.ctx, p.s1.seq, p.toks, &p.f);
		own = o.First();
	};
	if (site == "prompt_window" || site == "decode_window") {
		DecodeFx d;
		d.prompt = QwenPrompt(5);
		auto call = site == "prompt_window" ? std::function<void()>(prompt) : std::function<void()>([&] {
			d.Call(p.s1);
			own = d.own_status;
		});
		if (site == "decode_window") d.Pre(p.s1);
		const uint32_t k = arm_seam_first_in_window(call);
		if (k == 0) return SetupFail(cell, "no in-window allocation");
		if (site == "decode_window") d.Pre(p.s1);
		ArmGpuMapDeviceRemovedQueryInjection();
		SslmGpuAllocCounterResetForTest();
		ArmGpuAllocFaultAtOccurrence(k, E_OUTOFMEMORY);
		call();
		ArmGpuAllocFaultAtOccurrence(0, S_OK);
		what = "TryMakeBuffer#" + std::to_string(k) + " in window";
	} else if (site == "prompt_close" || site == "prompt_readback" || site == "restore_reset" ||
	           site == "seq_create_map" || site == "devlogits" || site == "context_queue") {
		int method = M_CLOSE;
		uint32_t ord = 1;
		std::function<void()> call = prompt;
		std::vector<uint8_t> blob;
		if (site == "prompt_readback") {
			TraceStart();
			prompt();
			ord = Resolve(Sel{M_MAP, M_SIGNAL, 1, 1, 0, kTsWidth}, TraceStop());
			method = M_MAP;
		} else if (site == "restore_reset") {
			blob = SaveBlob(p.s1.ctx, p.s1.seq);
			method = M_ARESET;
			ord = 2;
			call = [&] {
				SslmGpuSequenceHandle* h = nullptr;
				own = sslm_gpu_seq_restore(p.s1.ctx, p.s1.model, blob.data(), blob.size(), &h);
				if (h) sslm_gpu_seq_release(p.s1.ctx, h);
			};
		} else if (site == "seq_create_map") {
			method = M_MAP;
			call = [&] {
				SslmGpuSequenceHandle* h = nullptr;
				own = sslm_gpu_seq_create(p.s1.ctx, p.s1.model, 64, &h);
				if (h) sslm_gpu_seq_release(p.s1.ctx, h);
			};
		} else if (site == "devlogits") {
			method = M_LRESET;
			call = [&] {
				int32_t tok = -1;
				own = SslmGpuSeqFinishTokenForG5Bridge(p.s1.ctx, p.s1.seq, &tok);
			};
			if (sslm_gpu_seq_reset(p.s1.ctx, p.s1.seq) != OK ||
			    SslmGpuSeqPrefillPromptForG5Bridge(p.s1.ctx, p.s1.seq, p.toks.data(), 5, 64) != OK)
				return SetupFail(cell, "prefill");
		} else if (site == "context_queue") {
			method = M_CQUEUE;
			call = [&] {
				SslmGpuContext* c = nullptr;
				own = sslm_gpu_context_create(GpuContextConfig{}, &c);
				if (c) sslm_gpu_context_destroy(c);
			};
		}
		if (ord == 0) return SetupFail(cell, "site did not resolve");
		ArmGpuMapDeviceRemovedQueryInjection();
		HookArmCall(method, ord, E_OUTOFMEMORY);
		call();
		const uint32_t fired = g_arm.fired.load();
		HookDisarm();
		if (!fired) {
			v.Invalid();
			std::printf("%s site=%s: %s#%u never fired -> INVALID\n", cell, site.c_str(), MethodName(method), ord);
			return v.Finish();
		}
		what = std::string(MethodName(method)) + "#" + std::to_string(ord);
	} else if (site == "map") {
		SslmGpuAllocCounterResetForTest();
		ArmGpuMapDeviceRemovedQueryInjection();
		ArmGpuAllocFaultAtOccurrence(1, E_OUTOFMEMORY);
		SslmGpuModelHandle* h = nullptr;
		own = sslm_gpu_model_map(p.s1.ctx, &p.art.view, GpuResidencyConfig{}, &h);
		ArmGpuAllocFaultAtOccurrence(0, S_OK);
		if (h) sslm_gpu_model_unmap(p.s1.ctx, h);
		what = "TryMakeBuffer#1";
	} else {
		return SetupFail(cell, "--site=prompt_window|decode_window|prompt_close|prompt_readback|restore_reset|"
		                       "seq_create_map|devlogits|context_queue|map");
	}
	const bool conforms = own == DEVICE_LOST;
	v.Leg(conforms);
	std::printf("%s site=%s fault=%s + removed seam (rule 1): own=%s want DEVICE_LOST -> %s\n", cell, site.c_str(),
	            what.c_str(), St(own), conforms ? "PASS" : "FAIL");
	return v.Finish();
}
}  // namespace

// =============================================================================================
// R8 (must-accept): one real process memory limit, no injection -- TE-419 leg D3's construction.
// =============================================================================================
namespace {
int CellRealLimit(const char* cell) {
	Verdict v(cell);
	if (!Hooks(cell)) return SetupFail(cell, "hooks");
	Artifact a;
	if (!a.Load(Opt("qwen3"))) return SetupFail(cell, "--qwen3");
	Stack s;
	if (!s.Open(a, 1024)) return SetupFail(cell, "stack");
	const size_t start = static_cast<size_t>(OptInt("start", 3072)) << 10;
	const size_t step = static_cast<size_t>(OptInt("step", 128)) << 10;
	const int steps = static_cast<int>(OptInt("steps", 48));
	std::vector<int32_t> long_toks;
	static const int32_t pool[] = {785, 279, 315, 323, 264, 13, 3974, 13876, 1879, 374, 11, 220};
	for (int i = 0; i < 899; ++i) long_toks.push_back(pool[i % 12]);
	long_toks.push_back(151643);
	const std::vector<int32_t> warm = QwenPrompt(40);
	Frame ref, f;
	ref.Size(s.hidden);
	f.Size(s.hidden);
	if (PromptLoop(s.ctx, s.seq, long_toks, &ref).First() != OK) return SetupFail(cell, "900-token reference");
	HANDLE job = CreateJobObjectW(nullptr, nullptr);
	if (!job || !AssignProcessToJobObject(job, GetCurrentProcess())) return SetupFail(cell, "job object");
	uint32_t failed = 0, passed = 0;
	for (int i = 0; i < steps; ++i) {
		const size_t room = start + step * static_cast<size_t>(i);
		SslmGpuSequenceHandle* sq = nullptr;
		if (sslm_gpu_seq_create(s.ctx, s.model, 1024, &sq) != OK) return SetupFail(cell, "create");
		PromptLoop(s.ctx, sq, warm, &f);
		PROCESS_MEMORY_COUNTERS_EX pmc{};
		pmc.cb = sizeof(pmc);
		K32GetProcessMemoryInfo(GetCurrentProcess(), reinterpret_cast<PROCESS_MEMORY_COUNTERS*>(&pmc), sizeof(pmc));
		JOBOBJECT_EXTENDED_LIMIT_INFORMATION li{};
		li.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_PROCESS_MEMORY;
		li.ProcessMemoryLimit = pmc.PrivateUsage + room;
		SetInformationJobObject(job, JobObjectExtendedLimitInformation, &li, sizeof(li));
		SslmGpuStatus st = sslm_gpu_seq_reset(s.ctx, sq);
		if (st == OK)
			st = SslmGpuSeqPrefillPromptForG5Bridge(s.ctx, sq, long_toks.data(), static_cast<int32_t>(long_toks.size()), 64);
		li.BasicLimitInformation.LimitFlags = 0;
		SetInformationJobObject(job, JobObjectExtendedLimitInformation, &li, sizeof(li));
		const LoopOut after = PromptLoop(s.ctx, sq, long_toks, &f);
		const bool equal = after.First() == OK && f.Same(ref);
		sslm_gpu_seq_release(s.ctx, sq);
		if (st == OK) {
			++passed;
			std::printf("%s headroom=%zu KiB prefill=OK re-encode=%s equal=%d\n", cell, room >> 10, St(after.First()),
			            equal ? 1 : 0);
			v.Leg(equal);
			continue;
		}
		++failed;
		const bool conforms = st == ALLOC_FAILED && equal;
		v.Leg(conforms);
		std::printf("%s headroom=%zu KiB prefill=%s re-encode=%s equal=%d want ALLOCATION_FAILED+equal -> %s\n", cell,
		            room >> 10, St(st), St(after.First()), equal ? 1 : 0, conforms ? "PASS" : "FAIL");
		std::fflush(stdout);
	}
	std::printf("%s cells=%d failed-under-limit=%u passed=%u\n", cell, steps, failed, passed);
	if (failed == 0) {
		v.Invalid();
		std::printf("%s no cell failed under the limit: the construction exercised nothing -> INVALID\n", cell);
	}
	return v.Finish();
}
}  // namespace

// =============================================================================================
// Plan Sec3.10 dimension 10: a real, non-injected VRAM exhaustion attempt (the ballast). A second
// D3D12 device in this process allocates DEFAULT buffers until the local segment's budget is
// exceeded by a bounded margin, then a C = 1,024 sequence create and a 900-token prefill run.
// Bounded so that it cannot drive the shared machine's system memory: the ballast stops at
// budget + --margin-mib, and WDDM demotion beyond that is recorded, not pushed.
// =============================================================================================
namespace {
int CellBallast(const char* cell) {
	Verdict v(cell);
	Artifact a;
	if (!a.Load(Opt("qwen3"))) return SetupFail(cell, "--qwen3");
	Stack s;
	if (!s.Open(a, 64)) return SetupFail(cell, "stack");
	std::vector<int32_t> long_toks;
	static const int32_t pool[] = {785, 279, 315, 323, 264, 13, 3974, 13876, 1879, 374, 11, 220};
	for (int i = 0; i < 899; ++i) long_toks.push_back(pool[i % 12]);
	long_toks.push_back(151643);
	Frame ref, f;
	ref.Size(s.hidden);
	f.Size(s.hidden);
	SslmGpuSequenceHandle* rs = nullptr;
	if (sslm_gpu_seq_create(s.ctx, s.model, 1024, &rs) != OK || PromptLoop(s.ctx, rs, long_toks, &ref).First() != OK)
		return SetupFail(cell, "900-token reference");
	sslm_gpu_seq_release(s.ctx, rs);
	ComPtr<IDXGIFactory6> factory;
	ComPtr<IDXGIAdapter3> adapter;
	ComPtr<ID3D12Device> dev;
	if (FAILED(CreateDXGIFactory2(0, IID_PPV_ARGS(&factory)))) return SetupFail(cell, "factory");
	{
		ComPtr<IDXGIAdapter1> a1;
		if (FAILED(factory->EnumAdapterByGpuPreference(0, DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE, IID_PPV_ARGS(&a1))) ||
		    FAILED(a1.As(&adapter)) || FAILED(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&dev))))
			return SetupFail(cell, "ballast device");
	}
	auto mem = [&](DXGI_MEMORY_SEGMENT_GROUP g) {
		DXGI_QUERY_VIDEO_MEMORY_INFO i{};
		adapter->QueryVideoMemoryInfo(0, g, &i);
		return i;
	};
	const uint64_t margin = static_cast<uint64_t>(OptInt("margin-mib", 512)) << 20;
	const uint64_t chunk = 64ull << 20;
	std::vector<ComPtr<ID3D12Resource>> ballast;
	DXGI_QUERY_VIDEO_MEMORY_INFO local = mem(DXGI_MEMORY_SEGMENT_GROUP_LOCAL);
	std::printf("%s before: local budget=%llu MiB usage=%llu MiB; non-local usage=%llu MiB\n", cell,
	            local.Budget >> 20, local.CurrentUsage >> 20, mem(DXGI_MEMORY_SEGMENT_GROUP_NON_LOCAL).CurrentUsage >> 20);
	HRESULT last = S_OK;
	while (true) {
		local = mem(DXGI_MEMORY_SEGMENT_GROUP_LOCAL);
		if (local.CurrentUsage >= local.Budget + margin) break;
		D3D12_HEAP_PROPERTIES hp{};
		hp.Type = D3D12_HEAP_TYPE_DEFAULT;
		D3D12_RESOURCE_DESC rd{};
		rd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
		rd.Width = chunk;
		rd.Height = 1;
		rd.DepthOrArraySize = 1;
		rd.MipLevels = 1;
		rd.SampleDesc.Count = 1;
		rd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
		ComPtr<ID3D12Resource> r;
		last = dev->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &rd, D3D12_RESOURCE_STATE_COMMON, nullptr,
		                                    IID_PPV_ARGS(&r));
		if (FAILED(last)) break;
		ballast.push_back(r);
		if (ballast.size() > 1024) break;  // 64 GiB: never reached on this box, a runaway guard
	}
	local = mem(DXGI_MEMORY_SEGMENT_GROUP_LOCAL);
	std::printf("%s ballast: %zu x 64 MiB; last HRESULT=%s; local usage=%llu MiB of budget %llu MiB; non-local usage=%llu MiB\n",
	            cell, ballast.size(), Hex(static_cast<uint32_t>(last)).c_str(), local.CurrentUsage >> 20, local.Budget >> 20,
	            mem(DXGI_MEMORY_SEGMENT_GROUP_NON_LOCAL).CurrentUsage >> 20);
	SslmGpuSequenceHandle* sq = nullptr;
	const SslmGpuStatus cs = sslm_gpu_seq_create(s.ctx, s.model, 1024, &sq);
	SslmGpuStatus pf = DEVICE_LOST;
	if (cs == OK) pf = PromptLoop(s.ctx, sq, long_toks, &f).First();
	ballast.clear();
	// Once the ballast is freed: the re-encode equals the reference.
	SslmGpuStatus rc = OK;
	bool equal = false;
	if (!sq) rc = sslm_gpu_seq_create(s.ctx, s.model, 1024, &sq);
	if (rc == OK && sq) {
		const LoopOut o = PromptLoop(s.ctx, sq, long_toks, &f);
		equal = o.First() == OK && f.Same(ref);
		sslm_gpu_seq_release(s.ctx, sq);
	}
	const bool exhausted = cs != OK || pf != OK;
	std::printf("%s under ballast: create(C=1024)=%s prefill(900)=%s; after free: re-encode equal=%d\n", cell, St(cs),
	            St(pf), equal ? 1 : 0);
	if (!exhausted) {
		std::printf("%s the ballast did not make any engine allocation fail (WDDM demoted instead): the real path is "
		            "unconstructible within the bound -> INJECTED ONLY\n", cell);
		std::printf("VERDICT %s UNCONSTRUCTIBLE\n", cell);
		return 5;
	}
	const bool conforms = (cs == ALLOC_FAILED || (cs == OK && pf == ALLOC_FAILED)) && equal;
	v.Leg(conforms);
	std::printf("%s want ALLOCATION_FAILED and an equal re-encode -> %s\n", cell, conforms ? "PASS" : "FAIL");
	return v.Finish();
}
}  // namespace

// =============================================================================================
int main(int argc, char** argv) {
	std::setvbuf(stdout, nullptr, _IOLBF, 1 << 16);
	if (argc < 2) {
		std::printf("usage: te425_cells.exe <mode> [--key=value ...]\n");
		return 2;
	}
	const std::string mode = argv[1];
	for (int i = 2; i < argc; ++i) {
		std::string a = argv[i];
		if (a.rfind("--", 0) != 0) continue;
		const size_t eq = a.find('=');
		if (eq == std::string::npos) g_opt[a.substr(2)] = "1";
		else g_opt[a.substr(2, eq - 2)] = a.substr(eq + 1);
	}
	const auto t0 = std::chrono::steady_clock::now();
	int rc = 2;
	if (mode == "r2") rc = CellPromptSweep("R2", Src::Seam, NewFault::BadAlloc);
	else if (mode == "r3") rc = CellPromptSweep("R3", Src::New, NewFault::BadAlloc);
	else if (mode == "r3len") rc = CellPromptSweep("R3.length_error", Src::New, NewFault::LengthError);
	else if (mode == "r10") rc = CellPromptSweep("R10", Src::New, NewFault::Foreign);
	else if (mode == "r4seam") rc = CellSchemaSweep("R4.seam", Src::Seam);
	else if (mode == "r4new") rc = CellSchemaSweep("R4.new", Src::New);
	else if (mode == "r5seam") rc = CellDecodeSweep("R5.seam", Src::Seam);
	else if (mode == "r5new") rc = CellDecodeSweep("R5.new", Src::New);
	else if (mode == "r10dec") rc = CellDecodeSweep("R10.decode", Src::New, NewFault::Foreign);
	else if (mode == "r6") rc = CellBatch("R6");
	else if (mode == "hooks") rc = CellHooks("R9");
	else if (mode == "refgen") rc = CellRefgen("REFGEN");
	else if (mode == "r7cso") rc = CellCsoRemoved("R7.4");
	else if (mode == "e7ref") rc = CellFirstTouch("E7REF", true);
	else if (mode == "e7") rc = CellFirstTouch("R1.E7", false);
	else if (mode == "e7direct") rc = CellFirstTouchDirect("R1.E7direct");
	else if (mode == "r14") rc = CellRestoreRecovery("R14");
	else if (mode == "r15zero") rc = CellEntryZero("R15.zero");
	else if (mode == "r15sweep") rc = CellEntrySweep("R15.sweep");
	else if (mode == "r1new") rc = CellCreateSweep("R1.new");
	else if (mode == "r1ctx") rc = CellContextSeam("R1.ctx");
	else if (mode == "removed") rc = CellRemoved("R11.removed");
	else if (mode == "r8") rc = CellRealLimit("R8");
	else if (mode == "ballast") rc = CellBallast("DIM10.ballast");
	else std::printf("unknown mode %s\n", mode.c_str());
	std::printf("WALL %s %.1f s\n", mode.c_str(),
	            std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
	std::fflush(stdout);
	// Leave without running static destructors: a rule-0 leg may leave D3D12 work unconfirmed, and the
	// verdict is already printed.
	std::fflush(stderr);
	TerminateProcess(GetCurrentProcess(), static_cast<UINT>(rc));
	return rc;
}

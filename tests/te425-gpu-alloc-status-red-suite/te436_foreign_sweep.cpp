// TE-436 -- the foreign-exception sweep: a non-standard exception thrown at EVERY host-allocation site
// of every GPU route that submits work, each leg judged by plan Sec3.3.
//
// Plan: Claude/Plans/te421-slm172-host-oom.md (records tree), Sec3.2 E-2 rule 3 and E-4, Sec3.3, Sec3.5
// R10. Record: Claude/Curie/te436-slm180-sweep-2026-09-25.md (records tree).
//
// THE CLASS PINNED. An exception type that a catch ladder does not contain escapes while the engine
// still holds submitted work or a handle's in-flight state. It surfaced three times, one region further
// out each time (the recording windows, the submission tails, the decode finish), because R10 drove the
// non-standard throw only at recording-window edges. This cell drives it at every operator-new site the
// route's call reaches, so the class is watched as a whole rather than region by region.
//
// THE ROUTES (--route). Each is the public call sequence a caller makes, on the real artifact:
//   prompt  U3's per-document loop: reset, SslmGpuSeqPrefillPromptForG5Bridge, read (--len, default 5:
//           two sub-chunks, so both the synchronously finished non-final sub-chunk and the final one run).
//   schema  the same loop over SslmGpuSeqPrefillSchemaContentForG5Bridge on the G5 schema fixture: its
//           caller frame differs from the prompt twin's (the DFA walk state and *consumed).
//   step    one decode step through the per-token primitives after a 5-token prefill: embed,
//           sslm_decode_step_gpu, sslm_gpu_ready(block=1).
//   bridge  one SslmGpuSeqDecodeStepForG5Bridge call after a 5-token prefill and one call that spends
//           ready_for_logits (--budget, default all layers in one submission: every allocation site of
//           the bridge's call is reached once).
//   batch   one sslm_decode_step_batch_gpu call over three sequences of one context, each embedded after a
//           5-token prefill.
//
// THE ORACLE, per leg (plan Sec3.3 for a foreign exception, which is rule 3):
//   1. the armed call's own status is SSLM_DEVICE_LOST (a failed prefill's read in the same loop
//      SSLM_PREFILL_HIDDEN_UNAVAILABLE; the schema twin's *consumed equals the committed count; a batch
//      marks the faulted sequence and every later one SSLM_DEVICE_LOST and no slot any other failure);
//   2. the next call on a SECOND context is SSLM_OK and byte-equal to the reference;
//   3. the same handle's next call -- after the sslm_gpu_seq_reset Sec3.3 prescribes -- is SSLM_OK and
//      byte-equal to the reference;
//   4. where the throw lands while a submission is outstanding (see below), that submission had completed
//      when the faulted public call returned.
//
// SITE SELECTION IS STRUCTURAL. A counting pass labels every operator-new call of the route's call with
// (phase, the ordinal of the last submit event -- list Close, list Reset, ExecuteCommandLists, Signal --
// the hash of the call stack that reached operator new, and which occurrence of that stack it is since
// that event). The armed pass fires at the call carrying the counted site's label; a label the armed call
// never reaches reads INVALID (exit 4), never a verdict. Absolute numbers are not the identity: the
// engine makes at least one allocation on some calls and not others, which shifts every later number
// (observed at c3b5412: 99 and 100 sites on two identical prompt loops). --site=K names the K-th site
// of the counting pass, which every process reaches with the same history.
//
// TIMING, MADE DETERMINISTIC. A site is OUTSTANDING when the last ExecuteCommandLists has been signalled
// and the engine has not yet observed that submission complete (neither GetCompletedValue at or past its
// value nor SetEventOnCompletion for it). Only there can an escaping exception race the GPU. At an
// outstanding site this cell gates the queue exactly as te433_tail_pin.cpp does: a hook inserts
// ID3D12CommandQueue::Wait on a harness fence ahead of that submission, and only the engine's own
// SetEventOnCompletion for it opens the gate. The completed value is read the instant the faulted public
// call returns, so an engine that returns without waiting the submission out reads RED on any host. A
// watchdog opens a gate still shut after 20 s and the leg reads INVALID, so a design hang cannot pass or
// stall a run. Every other site follows the engine's own confirmed wait, so its outcome does not depend
// on GPU timing.
//
// Usage: te436_foreign_sweep.exe --route=prompt|schema|step|bridge|batch (--count | --site=K |
//        --select=outstanding | --from=A --to=B) [--len=N] [--budget=N] [--expect-phase=NAME]
//        --qwen3=PATH [--g5=PATH]
// One site per process is the suite's shape (run_red_suite.py's fx-* jobs): a red leg can leave a handle
// or the process's submission list unusable, which would score every later leg on that damage.
// Exit status: 0 every leg conforms; 1 at least one does not (RED); 3 setup failed; 4 a leg could not be
// driven. Output ends with SUMMARY and VERDICT lines in the TE-425 suite's format.
#include "te425_harness.h"

#include <intrin.h>

#include <map>
#include <thread>

using namespace te425;

// =============================================================================================
// Submission tracking and the gate.
// =============================================================================================
namespace {
enum Ev : uint8_t { EV_NONE = 0, EV_CLOSE, EV_LRESET, EV_ECL, EV_SIGNAL, EV_SETEVENT, EV_GCV, EV_COUNT };
const char* EvName(int e) {
	static const char* const k[] = {"none", "Close", "ListReset", "ExecuteCommandLists", "Signal",
	                                "SetEventOnCompletion", "GetCompletedValue"};
	return (e >= 0 && e < EV_COUNT) ? k[e] : "?";
}
std::atomic<uint32_t> g_hook_calls[EV_COUNT];
// Submit events (Close, list Reset, ExecuteCommandLists, Signal) since Begin(). The two completion
// queries are not submit events: whether the engine calls SetEventOnCompletion depends on whether the
// GPU had already finished, so they never enter a site's label.
std::atomic<uint32_t> g_ev_ord{0};
std::atomic<int> g_ev_last{EV_NONE};
std::atomic<uint32_t> g_since{0};  // operator-new calls since the last submit event
std::atomic<uint32_t> g_ecl_n{0};  // ExecuteCommandLists calls since Begin()
// The outstanding submission: the Signal after the latest ExecuteCommandLists, until the engine observes
// it complete.
std::atomic<bool> g_await_signal{false};
std::atomic<bool> g_outstanding{false};
ID3D12Fence* g_out_fence = nullptr;
UINT64 g_out_value = 0;

void SubmitEvent(int e) {
	g_hook_calls[e].fetch_add(1);
	g_ev_ord.fetch_add(1);
	g_ev_last.store(e);
	g_since.store(0);
}

struct Gate {
	std::atomic<uint32_t> arm_ecl{0};  // the ExecuteCommandLists ordinal (since Begin) to gate; 0 = off
	Microsoft::WRL::ComPtr<ID3D12Fence> fence;
	UINT64 target = 0;
	std::atomic<bool> closed{false};
	std::atomic<bool> await_signal{false};
	ID3D12Fence* sub_fence = nullptr;
	UINT64 sub_value = 0;
	std::atomic<bool> opened_by_engine{false};
	std::atomic<long long> closed_at_ms{0};
	std::atomic<bool> watchdog_fired{false};
	HRESULT wait_hr = S_OK;
};
Gate g_gate;

long long NowMs() {
	return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch())
	    .count();
}

// Site labels. A site is identified by its phase, the ordinal of the last submit event, the hash of the
// call stack that reached operator new, and which occurrence of that stack it is since that event. The
// engine has at least one allocation that happens on some calls and not others (a container crossing its
// capacity): an absolute count, or a position counted since the last event, shifts every later site when
// it fires, while a call-stack identity does not. Addresses are compared only within one process.
struct SiteKey {
	uint8_t phase;
	uint32_t ev_ord;
	uint64_t stack;
	uint32_t occ;
	bool operator==(const SiteKey& o) const {
		return phase == o.phase && ev_ord == o.ev_ord && stack == o.stack && occ == o.occ;
	}
};
// Occurrences of each (event ordinal, stack) since Begin(): a fixed open-addressed table, no allocation.
constexpr uint32_t kOccSlots = 8192;
struct OccSlot {
	uint32_t ev_ord;
	uint64_t stack;
	uint32_t n;
};
OccSlot g_occ[kOccSlots];
uint32_t NextOccurrence(uint32_t ev_ord, uint64_t stack) {
	uint32_t i = static_cast<uint32_t>((stack ^ (static_cast<uint64_t>(ev_ord) * 0x9E3779B97F4A7C15ull)) % kOccSlots);
	for (uint32_t probe = 0; probe < kOccSlots; ++probe, i = (i + 1) % kOccSlots) {
		OccSlot& s = g_occ[i];
		if (s.n == 0) {
			s = OccSlot{ev_ord, stack, 1};
			return 1;
		}
		if (s.ev_ord == ev_ord && s.stack == stack) return ++s.n;
	}
	return 0;  // table full: never a match (no site count reaches this)
}
// The return address into the harness above ArmedPass(): the hash stops there, so the counting pass and
// the armed pass, which call ArmedPass() from different lines, hash the same frames.
void* g_stack_stop = nullptr;
uint64_t StackHash() {
	void* frames[32];
	const USHORT n = RtlCaptureStackBackTrace(2, 32, frames, nullptr);
	uint64_t h = 1469598103934665603ull;
	for (USHORT i = 0; i < n && frames[i] != g_stack_stop; ++i) {
		h ^= reinterpret_cast<uint64_t>(frames[i]);
		h *= 1099511628211ull;
	}
	return h;
}
struct FxLabel {
	SiteKey key;
	uint32_t since;  // position after the last submit event, for the reader
	uint8_t ev_last;
	uint8_t outstanding;
	uint32_t ecl;
	uint8_t aligned;
};
FxLabel g_fl[kMaxSites];

struct FireState {
	bool fired = false;
	uint32_t count = 0;
	FxLabel label{};
	bool gate_closed = false;
	bool have_sub = false;
};
std::atomic<bool> g_fire_armed{false};
SiteKey g_fire_key{};
FireState g_fire;
}  // namespace

// =============================================================================================
// Fault source: the replaced operator new family (TE-425's), throwing a foreign type at the armed label.
// =============================================================================================
static void* FxAlloc(size_t n, size_t align, bool aligned) {
	if (g_new.counting.load(std::memory_order_relaxed)) {
		const uint32_t c = g_new.count.fetch_add(1) + 1;
		const uint32_t since = g_since.fetch_add(1) + 1;
		const uint32_t ev_ord = g_ev_ord.load();
		const uint64_t stack = StackHash();
		const SiteKey key{static_cast<uint8_t>(g_phase.load()), ev_ord, stack, NextOccurrence(ev_ord, stack)};
		const FxLabel label{key, since, static_cast<uint8_t>(g_ev_last.load()), static_cast<uint8_t>(g_outstanding.load() ? 1 : 0),
		                    g_ecl_n.load(), static_cast<uint8_t>(aligned ? 1 : 0)};
		if (g_new.label.load(std::memory_order_relaxed) && c < kMaxSites) g_fl[c] = label;
		if (g_fire_armed.load() && key == g_fire_key) {
			g_fire_armed.store(false);
			g_fire.fired = true;
			g_fire.count = c;
			g_fire.label = label;
			g_fire.gate_closed = g_gate.closed.load();
			g_fire.have_sub = g_gate.sub_fence != nullptr && !g_gate.await_signal.load();
			throw ForeignFault{c};
		}
	}
	void* p = aligned ? _aligned_malloc(n ? n : 1, align) : std::malloc(n ? n : 1);
	if (!p) throw std::bad_alloc();
	return p;
}
void* operator new(size_t n) { return FxAlloc(n, 0, false); }
void* operator new[](size_t n) { return FxAlloc(n, 0, false); }
void* operator new(size_t n, const std::nothrow_t&) noexcept {
	try {
		return FxAlloc(n, 0, false);
	} catch (...) {
		return nullptr;
	}
}
void* operator new[](size_t n, const std::nothrow_t&) noexcept {
	try {
		return FxAlloc(n, 0, false);
	} catch (...) {
		return nullptr;
	}
}
void* operator new(size_t n, std::align_val_t a) { return FxAlloc(n, static_cast<size_t>(a), true); }
void* operator new[](size_t n, std::align_val_t a) { return FxAlloc(n, static_cast<size_t>(a), true); }
void* operator new(size_t n, std::align_val_t a, const std::nothrow_t&) noexcept {
	try {
		return FxAlloc(n, static_cast<size_t>(a), true);
	} catch (...) {
		return nullptr;
	}
}
void* operator new[](size_t n, std::align_val_t a, const std::nothrow_t&) noexcept {
	try {
		return FxAlloc(n, static_cast<size_t>(a), true);
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
// D3D12 vtable hooks. Slots are the SDK's C vtable order (Windows Kits 10.0.26100.0 d3d12.h):
// GraphicsCommandList 9 Close, 10 Reset; CommandQueue 10 ExecuteCommandLists, 14 Signal; Fence 8
// GetCompletedValue, 9 SetEventOnCompletion. te425_cells.cpp and te433_tail_pin.cpp patch the same slots.
// =============================================================================================
namespace {
using PFN_CLOSE = HRESULT(STDMETHODCALLTYPE*)(ID3D12GraphicsCommandList*);
using PFN_LRESET = HRESULT(STDMETHODCALLTYPE*)(ID3D12GraphicsCommandList*, ID3D12CommandAllocator*,
                                               ID3D12PipelineState*);
using PFN_ECL = void(STDMETHODCALLTYPE*)(ID3D12CommandQueue*, UINT, ID3D12CommandList* const*);
using PFN_SIGNAL = HRESULT(STDMETHODCALLTYPE*)(ID3D12CommandQueue*, ID3D12Fence*, UINT64);
using PFN_SETEVENT = HRESULT(STDMETHODCALLTYPE*)(ID3D12Fence*, UINT64, HANDLE);
using PFN_GCV = UINT64(STDMETHODCALLTYPE*)(ID3D12Fence*);
PFN_CLOSE o_close;
PFN_LRESET o_lreset;
PFN_ECL o_ecl;
PFN_SIGNAL o_signal;
PFN_SETEVENT o_setevent;
PFN_GCV o_gcv;

HRESULT STDMETHODCALLTYPE H_close(ID3D12GraphicsCommandList* self) {
	SubmitEvent(EV_CLOSE);
	return o_close(self);
}
HRESULT STDMETHODCALLTYPE H_lreset(ID3D12GraphicsCommandList* self, ID3D12CommandAllocator* a,
                                   ID3D12PipelineState* p) {
	SubmitEvent(EV_LRESET);
	return o_lreset(self, a, p);
}
void STDMETHODCALLTYPE H_ecl(ID3D12CommandQueue* self, UINT n, ID3D12CommandList* const* lists) {
	SubmitEvent(EV_ECL);
	const uint32_t ord = g_ecl_n.fetch_add(1) + 1;
	g_await_signal.store(true);
	const uint32_t arm = g_gate.arm_ecl.load();
	if (arm != 0 && ord == arm && !g_gate.closed.load()) {
		if (!g_gate.fence) {
			Microsoft::WRL::ComPtr<ID3D12Device> dev;
			if (SUCCEEDED(self->GetDevice(IID_PPV_ARGS(&dev)))) {
				(void)dev->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&g_gate.fence));
			}
		}
		if (g_gate.fence) {
			g_gate.target += 1;
			g_gate.wait_hr = self->Wait(g_gate.fence.Get(), g_gate.target);
			if (SUCCEEDED(g_gate.wait_hr)) {
				g_gate.sub_fence = nullptr;
				g_gate.sub_value = 0;
				g_gate.await_signal.store(true);
				g_gate.closed_at_ms.store(NowMs());
				g_gate.closed.store(true);
			}
		}
	}
	o_ecl(self, n, lists);
}
HRESULT STDMETHODCALLTYPE H_signal(ID3D12CommandQueue* self, ID3D12Fence* fence, UINT64 v) {
	SubmitEvent(EV_SIGNAL);
	if (g_await_signal.load()) {
		g_out_fence = fence;
		g_out_value = v;
		g_outstanding.store(true);
		g_await_signal.store(false);
	}
	if (g_gate.await_signal.load()) {
		g_gate.sub_fence = fence;
		g_gate.sub_value = v;
		g_gate.await_signal.store(false);
	}
	return o_signal(self, fence, v);
}
HRESULT STDMETHODCALLTYPE H_setevent(ID3D12Fence* self, UINT64 v, HANDLE e) {
	g_hook_calls[EV_SETEVENT].fetch_add(1);
	if (g_outstanding.load() && self == g_out_fence && v >= g_out_value) g_outstanding.store(false);
	if (g_gate.closed.load() && !g_gate.await_signal.load() && self == g_gate.sub_fence && v >= g_gate.sub_value) {
		// The engine asked to be told of the gated submission's completion: let the GPU run it.
		g_gate.opened_by_engine.store(true);
		g_gate.closed.store(false);
		(void)g_gate.fence->Signal(g_gate.target);
	}
	return o_setevent(self, v, e);
}
UINT64 STDMETHODCALLTYPE H_gcv(ID3D12Fence* self) {
	g_hook_calls[EV_GCV].fetch_add(1);
	const UINT64 r = o_gcv(self);
	if (g_outstanding.load() && self == g_out_fence && r >= g_out_value) g_outstanding.store(false);
	return r;
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

Microsoft::WRL::ComPtr<ID3D12Device> p_dev;
Microsoft::WRL::ComPtr<ID3D12CommandQueue> p_queue;
Microsoft::WRL::ComPtr<ID3D12CommandAllocator> p_alloc;
Microsoft::WRL::ComPtr<ID3D12GraphicsCommandList> p_list;
Microsoft::WRL::ComPtr<ID3D12Fence> p_fence;

bool InstallFxHooks(std::string* why) {
	D3D12_COMMAND_QUEUE_DESC qd{};
	qd.Type = D3D12_COMMAND_LIST_TYPE_COMPUTE;
	if (FAILED(D3D12CreateDevice(nullptr, D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&p_dev))) ||
	    FAILED(p_dev->CreateCommandQueue(&qd, IID_PPV_ARGS(&p_queue))) ||
	    FAILED(p_dev->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COMPUTE, IID_PPV_ARGS(&p_alloc))) ||
	    FAILED(p_dev->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_COMPUTE, p_alloc.Get(), nullptr,
	                                    IID_PPV_ARGS(&p_list))) ||
	    FAILED(p_dev->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&p_fence)))) {
		*why = "probe object creation failed";
		return false;
	}
	const bool ok = Patch(p_list.Get(), 9, reinterpret_cast<void*>(&H_close), &o_close) &&
	                Patch(p_list.Get(), 10, reinterpret_cast<void*>(&H_lreset), &o_lreset) &&
	                Patch(p_queue.Get(), 10, reinterpret_cast<void*>(&H_ecl), &o_ecl) &&
	                Patch(p_queue.Get(), 14, reinterpret_cast<void*>(&H_signal), &o_signal) &&
	                Patch(p_fence.Get(), 8, reinterpret_cast<void*>(&H_gcv), &o_gcv) &&
	                Patch(p_fence.Get(), 9, reinterpret_cast<void*>(&H_setevent), &o_setevent);
	if (!ok) {
		*why = "VirtualProtect failed on a vtable slot";
		return false;
	}
	// Self-check: each hook fires on a probe call (a slot off by one would call a different method).
	uint32_t before[EV_COUNT];
	for (int e = 0; e < EV_COUNT; ++e) before[e] = g_hook_calls[e].load();
	HANDLE ev = CreateEventW(nullptr, FALSE, FALSE, nullptr);
	ID3D12CommandList* lists[] = {p_list.Get()};
	bool calls_ok = SUCCEEDED(p_list->Close());
	if (calls_ok) p_queue->ExecuteCommandLists(1, lists);
	calls_ok = calls_ok && SUCCEEDED(p_queue->Signal(p_fence.Get(), 1)) &&
	           SUCCEEDED(p_fence->SetEventOnCompletion(1, ev)) && WaitForSingleObject(ev, 5000) == WAIT_OBJECT_0 &&
	           p_fence->GetCompletedValue() >= 1 && SUCCEEDED(p_alloc->Reset()) &&
	           SUCCEEDED(p_list->Reset(p_alloc.Get(), nullptr)) && SUCCEEDED(p_list->Close());
	CloseHandle(ev);
	for (int e = EV_CLOSE; e < EV_COUNT; ++e) {
		if (g_hook_calls[e].load() == before[e]) {
			*why = std::string("hook self-check: ") + EvName(e) + " did not fire on a probe call";
			return false;
		}
	}
	if (!calls_ok) {
		*why = "hook self-check: a probe call failed";
		return false;
	}
	return true;
}

// Opens a gate left shut after 20 s and marks the leg undriven: an engine that waits some other way
// would otherwise hang the process until the runner's timeout.
void Watchdog() {
	for (;;) {
		Sleep(250);
		if (g_gate.closed.load() && NowMs() - g_gate.closed_at_ms.load() > 20000) {
			g_gate.watchdog_fired.store(true);
			g_gate.closed.store(false);
			if (g_gate.fence) (void)g_gate.fence->Signal(g_gate.target);
		}
	}
}

// =============================================================================================
// Options, the reading, the release.
// =============================================================================================
std::map<std::string, std::string> g_opt;
std::string Opt(const char* k, const char* d = "") {
	auto it = g_opt.find(k);
	return it == g_opt.end() ? std::string(d) : it->second;
}
long long OptInt(const char* k, long long d) {
	auto it = g_opt.find(k);
	return it == g_opt.end() ? d : std::strtoll(it->second.c_str(), nullptr, 10);
}
bool Has(const char* k) { return g_opt.count(k) != 0; }
int SetupFail(const char* cell, const char* what) {
	std::printf("SETUP-FAIL %s: %s\nVERDICT %s INVALID\n", cell, what, cell);
	std::fflush(stdout);
	return 3;
}

void Begin() {
	g_ev_ord.store(0);
	g_ev_last.store(EV_NONE);
	g_since.store(0);
	std::memset(g_occ, 0, sizeof g_occ);
	g_ecl_n.store(0);
	g_await_signal.store(false);
	g_outstanding.store(false);
	g_out_fence = nullptr;
	g_out_value = 0;
}

struct Reading {
	bool taken = false;
	bool have_sub = false;
	UINT64 value = 0, completed = 0;
	bool gate_shut = false;
	bool Complete() const { return have_sub && completed >= value; }
};
Reading g_reading;
// Called after every public call of an armed route: the first one to return after the throw is the
// faulted call, and its return is the instant the gated submission's completion is read.
void AfterPublic() {
	if (!g_fire.fired || g_reading.taken) return;
	g_reading.taken = true;
	g_reading.have_sub = g_gate.sub_fence != nullptr && !g_gate.await_signal.load();
	if (g_reading.have_sub) {
		g_reading.value = g_gate.sub_value;
		g_reading.completed = o_gcv(g_gate.sub_fence);
	}
	g_reading.gate_shut = g_gate.closed.load();
}
// Opens a gate the engine left shut and waits the gated submission out, so the after-calls do not queue
// behind it. False when that wait cannot be confirmed.
bool Release() {
	bool ok = true;
	if (g_gate.closed.load()) {
		g_gate.closed.store(false);
		(void)g_gate.fence->Signal(g_gate.target);
	}
	g_gate.arm_ecl.store(0);
	g_gate.await_signal.store(false);
	if (g_gate.sub_fence && o_gcv(g_gate.sub_fence) < g_gate.sub_value) {
		HANDLE ev = CreateEventW(nullptr, FALSE, FALSE, nullptr);
		ok = SUCCEEDED(o_setevent(g_gate.sub_fence, g_gate.sub_value, ev)) && WaitForSingleObject(ev, 10000) == WAIT_OBJECT_0;
		CloseHandle(ev);
	}
	g_gate.sub_fence = nullptr;
	g_gate.sub_value = 0;
	g_gate.opened_by_engine.store(false);
	return ok;
}

std::vector<uint8_t> SaveBlob(SslmGpuContext* ctx, SslmGpuSequenceHandle* s) {
	size_t n = 0;
	(void)sslm_gpu_seq_save(ctx, s, nullptr, &n);
	std::vector<uint8_t> b(n);
	if (!n || sslm_gpu_seq_save(ctx, s, b.data(), &n) != OK) b.clear();
	return b;
}
SslmGpuStatus FirstNotOk(std::initializer_list<SslmGpuStatus> l) {
	for (SslmGpuStatus s : l)
		if (s != OK) return s;
	return OK;
}

// =============================================================================================
// The routes. Clean(i) runs the route's whole call on context i (1 or 2), reset first, and compares it
// with the reference; PreArmed() sets the state the armed call starts from on context 1; Armed() is the
// call under test on context 1 and calls AfterPublic() after each public call; Judge() holds its outcome
// to Sec3.3; ArmedStatus() is the armed call's first non-OK status (the counting pass must read OK).
// Nothing inside Armed() allocates on the harness's side, so every counted site is the engine's.
// =============================================================================================
constexpr int32_t kStepToken = 785;

struct Route {
	virtual ~Route() = default;
	virtual bool Open() = 0;
	virtual bool Clean(int which, SslmGpuStatus* st) = 0;
	virtual bool PreArmed() = 0;
	virtual void Armed() = 0;
	virtual bool Judge(std::string* d) = 0;
	virtual SslmGpuStatus ArmedStatus() const = 0;
};

// prompt: U3's loop on the prompt twin.
struct PromptRoute : Route {
	Artifact art;
	Stack s[2];
	std::vector<int32_t> toks;
	Frame ref, f;
	LoopOut own;
	bool Open() override {
		if (!art.Load(Opt("qwen3"))) return false;
		if (!s[0].Open(art, 64) || !s[1].Open(art, 64)) return false;
		toks = QwenPrompt(static_cast<int>(OptInt("len", 5)));
		ref.Size(s[0].hidden);
		f.Size(s[0].hidden);
		if (PromptLoop(s[0].ctx, s[0].seq, toks, &ref).First() != OK) return false;
		return PromptLoop(s[1].ctx, s[1].seq, toks, &f).First() == OK && f.Same(ref);
	}
	bool Clean(int which, SslmGpuStatus* st) override {
		const LoopOut o = PromptLoop(s[which - 1].ctx, s[which - 1].seq, toks, &f);
		*st = o.First();
		return *st == OK && f.Same(ref);
	}
	bool PreArmed() override { return true; }
	void Armed() override {
		Stack& x = s[0];
		own = LoopOut{};
		g_phase.store(PH_RESET);
		own.reset = sslm_gpu_seq_reset(x.ctx, x.seq);
		AfterPublic();
		g_phase.store(PH_PREFILL);
		if (own.reset == OK) {
			own.prefill = SslmGpuSeqPrefillPromptForG5Bridge(x.ctx, x.seq, toks.data(), static_cast<int32_t>(toks.size()), 64);
			AfterPublic();
		}
		g_phase.store(PH_READ);
		std::fill(f.codes.begin(), f.codes.end(), static_cast<int8_t>(0x5A));
		if (own.reset == OK) {
			own.read = sslm_gpu_seq_read_prefill_final_hidden(x.ctx, x.seq, f.codes.data(), f.codes.size(), &f.required,
			                                                  &f.m, &f.e);
			AfterPublic();
		}
		g_phase.store(PH_NONE);
	}
	bool Judge(std::string* d) override {
		*d = std::string("reset=") + St(own.reset) + " prefill=" + St(own.prefill) + " read=" + St(own.read);
		return own.First() == DEVICE_LOST && (own.FirstStage() != 1 || own.read == HIDDEN_UNAVAILABLE);
	}
	SslmGpuStatus ArmedStatus() const override { return own.First(); }
};

// schema: the same loop over the schema twin (tests/t2791 fixture_common.h's DFA walk, as te425_cells.cpp).
struct SchemaRoute : Route {
	Artifact art;
	Stack s[2];
	superslm::SslmModelView view{};
	superslm::SchemaMasksTable table;
	const superslm::SchemaEntry* entry = nullptr;
	std::vector<int32_t> chain;
	int32_t idx = -1;
	Frame ref, f;
	LoopOut own;
	int32_t consumed = -1;
	int64_t committed = -1;
	static uint32_t Le32(const uint8_t* p) {
		return static_cast<uint32_t>(p[0]) | (static_cast<uint32_t>(p[1]) << 8) | (static_cast<uint32_t>(p[2]) << 16) |
		       (static_cast<uint32_t>(p[3]) << 24);
	}
	bool Open() override {
		if (!art.Load(Opt("g5"))) return false;
		if (!s[0].Open(art, 64) || !s[1].Open(art, 64)) return false;
		if (!SslmGpuModelHasSchemasForG5Bridge(s[0].model)) return false;
		std::string err;
		if (superslm::SslmModel::Load(art.bytes.data(), art.bytes.size(), view, &err) != superslm::SslmModelStatus::Ok)
			return false;
		const superslm::SslmSectionView* sec = view.Section(superslm::SslmSectionType::SchemaMasks);
		if (!sec || !superslm::SchemaMasksTable::Parse(sec->data, sec->byte_size, view.config.vocab_size, table, &err))
			return false;
		const char* name = "shopkeeper_intent_extraction";
		entry = table.ByName(name);
		if (!entry) return false;
		uint32_t st = 0;
		for (int k = 0; k < static_cast<int>(OptInt("len", 5)); ++k) {
			const uint32_t b = Le32(entry->state_offsets_le + static_cast<size_t>(st) * 4);
			const uint32_t en = Le32(entry->state_offsets_le + static_cast<size_t>(st + 1) * 4);
			if (b >= en) break;
			const int32_t t = static_cast<int32_t>(Le32(entry->transitions_le + static_cast<size_t>(b) * 8));
			uint32_t nx = 0;
			if (!table.Transition(*entry, st, static_cast<uint32_t>(t), &nx)) break;
			chain.push_back(t);
			st = nx;
		}
		idx = SslmGpuSchemaLookupForG5Bridge(s[0].model, name);
		if (chain.empty() || idx < 0 || SslmGpuSchemaLookupForG5Bridge(s[1].model, name) != idx) return false;
		std::printf("schema chain walked=%zu\n", chain.size());
		ref.Size(s[0].hidden);
		f.Size(s[0].hidden);
		SslmGpuStatus st1 = OK;
		if (!Loop(0, &ref, &st1) || st1 != OK) return false;
		SslmGpuStatus st2 = OK;
		return Loop(1, &f, &st2) && st2 == OK && f.Same(ref);
	}
	// The whole loop on stack i, un-armed.
	bool Loop(int i, Frame* out, SslmGpuStatus* st) {
		Stack& x = s[i];
		SslmGpuStatus r = sslm_gpu_seq_reset(x.ctx, x.seq);
		if (r == OK) r = SslmGpuSeqSetSchemaForG5Bridge(x.ctx, x.seq, idx);
		int32_t c = -1;
		SslmGpuStatus p = OK, rd = OK;
		if (r == OK)
			p = SslmGpuSeqPrefillSchemaContentForG5Bridge(x.ctx, x.seq, chain.data(), static_cast<int32_t>(chain.size()), 64, &c);
		std::fill(out->codes.begin(), out->codes.end(), static_cast<int8_t>(0x5A));
		if (r == OK)
			rd = sslm_gpu_seq_read_prefill_final_hidden(x.ctx, x.seq, out->codes.data(), out->codes.size(), &out->required,
			                                            &out->m, &out->e);
		out->status = rd;
		*st = FirstNotOk({r, p, rd});
		return *st == OK && c == static_cast<int32_t>(chain.size());
	}
	bool Clean(int which, SslmGpuStatus* st) override { return Loop(which - 1, &f, st) && f.Same(ref); }
	bool PreArmed() override { return true; }
	void Armed() override {
		Stack& x = s[0];
		own = LoopOut{};
		consumed = -1;
		committed = -1;
		g_phase.store(PH_RESET);
		own.reset = sslm_gpu_seq_reset(x.ctx, x.seq);
		AfterPublic();
		if (own.reset == OK) {
			own.reset = SslmGpuSeqSetSchemaForG5Bridge(x.ctx, x.seq, idx);
			AfterPublic();
		}
		g_phase.store(PH_PREFILL);
		if (own.reset == OK) {
			own.prefill = SslmGpuSeqPrefillSchemaContentForG5Bridge(x.ctx, x.seq, chain.data(),
			                                                        static_cast<int32_t>(chain.size()), 64, &consumed);
			AfterPublic();
			committed = *SslmGpuSeqHandleContextLengthForBench(x.seq);
		}
		g_phase.store(PH_READ);
		std::fill(f.codes.begin(), f.codes.end(), static_cast<int8_t>(0x5A));
		if (own.reset == OK) {
			own.read = sslm_gpu_seq_read_prefill_final_hidden(x.ctx, x.seq, f.codes.data(), f.codes.size(), &f.required,
			                                                  &f.m, &f.e);
			AfterPublic();
		}
		g_phase.store(PH_NONE);
	}
	bool Judge(std::string* d) override {
		*d = std::string("reset=") + St(own.reset) + " prefill=" + St(own.prefill) + " read=" + St(own.read) +
		     " consumed=" + std::to_string(consumed) + " committed=" + std::to_string(committed);
		// Plan Sec3.5 R4: *consumed counts only committed tokens.
		const bool consumed_ok = own.FirstStage() != 1 || consumed == committed;
		return own.First() == DEVICE_LOST && (own.FirstStage() != 1 || own.read == HIDDEN_UNAVAILABLE) && consumed_ok;
	}
	SslmGpuStatus ArmedStatus() const override { return own.First(); }
};

// Shared by the three decode routes: two contexts, a 5-token prefill, and a reference token and state.
struct DecodeBase : Route {
	Artifact art;
	Stack s[2];
	std::vector<int32_t> prompt;
	// Reset (the reset Sec3.3 prescribes before reuse) and a 5-token prefill; the first refusal.
	SslmGpuStatus Prefill(Stack& x) {
		SslmGpuStatus st = sslm_gpu_seq_reset(x.ctx, x.seq);
		if (st == OK)
			st = SslmGpuSeqPrefillPromptForG5Bridge(x.ctx, x.seq, prompt.data(), static_cast<int32_t>(prompt.size()), 64);
		return st;
	}
	bool OpenStacks() {
		if (!art.Load(Opt("qwen3"))) return false;
		if (!s[0].Open(art, 64) || !s[1].Open(art, 64)) return false;
		prompt = QwenPrompt(5);
		return true;
	}
};

// step: embed, sslm_decode_step_gpu, sslm_gpu_ready(block=1).
struct StepRoute : DecodeBase {
	int32_t ref_token = -1;
	std::vector<uint8_t> ref_blob;
	SslmGpuStatus embed = OK, step = OK, ready_ret = OK, out_status = OK;
	bool Run(Stack& x, int32_t* tok, std::vector<uint8_t>* blob, SslmGpuStatus* st) {
		if ((*st = Prefill(x)) != OK) return false;
		SslmGpuStatus e = sslm_gpu_seq_embed_token(x.ctx, x.seq, kStepToken), p = OK, r = OK, o = OK;
		if (e == OK) p = sslm_decode_step_gpu(x.ctx, x.seq, nullptr, 0xFFFFFFFFu);
		if (e == OK && p == OK) {
			int32_t ready = 0;
			r = sslm_gpu_ready(x.ctx, x.seq, 1, &ready, &o);
		}
		*st = FirstNotOk({e, p, r, o});
		if (*st != OK) return false;
		*tok = -1;
		if ((*st = SslmGpuSeqFinishTokenForG5Bridge(x.ctx, x.seq, tok)) != OK) return false;
		*blob = SaveBlob(x.ctx, x.seq);
		return !blob->empty();
	}
	bool Open() override {
		if (!OpenStacks()) return false;
		SslmGpuStatus st;
		int32_t t2 = -1;
		std::vector<uint8_t> b2;
		return Run(s[0], &ref_token, &ref_blob, &st) && Run(s[1], &t2, &b2, &st) && t2 == ref_token && b2 == ref_blob;
	}
	bool Clean(int which, SslmGpuStatus* st) override {
		int32_t t = -1;
		std::vector<uint8_t> b;
		return Run(s[which - 1], &t, &b, st) && t == ref_token && b == ref_blob;
	}
	bool PreArmed() override { return Prefill(s[0]) == OK; }
	void Armed() override {
		Stack& x = s[0];
		embed = step = ready_ret = out_status = OK;
		g_phase.store(PH_STEP);
		embed = sslm_gpu_seq_embed_token(x.ctx, x.seq, kStepToken);
		AfterPublic();
		if (embed == OK) {
			step = sslm_decode_step_gpu(x.ctx, x.seq, nullptr, 0xFFFFFFFFu);
			AfterPublic();
		}
		g_phase.store(PH_READY);
		if (embed == OK && step == OK) {
			int32_t ready = 0;
			ready_ret = sslm_gpu_ready(x.ctx, x.seq, 1, &ready, &out_status);
			AfterPublic();
		}
		g_phase.store(PH_NONE);
	}
	bool Judge(std::string* d) override {
		*d = std::string("embed=") + St(embed) + " step=" + St(step) + " ready=" + St(ready_ret) + " out_status=" +
		     St(out_status);
		return ArmedStatus() == DEVICE_LOST;
	}
	SslmGpuStatus ArmedStatus() const override { return FirstNotOk({embed, step, ready_ret, out_status}); }
};

// bridge: one SslmGpuSeqDecodeStepForG5Bridge call that drives a token through the layers.
struct BridgeRoute : DecodeBase {
	uint32_t budget = 0xFFFFFFFFu;
	int32_t ref_token = -1;
	std::vector<uint8_t> ref_blob;
	SslmGpuStatus own = OK;
	int32_t own_token = -1;
	// Reset, prefill, and one bridge call that spends the prefill's ready_for_logits, so the next bridge
	// call embeds and drives.
	SslmGpuStatus Pre(Stack& x) {
		SslmGpuStatus st = Prefill(x);
		int32_t t = -1;
		if (st == OK) st = SslmGpuSeqDecodeStepForG5Bridge(x.ctx, x.seq, kStepToken, budget, &t);
		return st;
	}
	bool Run(Stack& x, int32_t* tok, std::vector<uint8_t>* blob, SslmGpuStatus* st) {
		if ((*st = Pre(x)) != OK) return false;
		*tok = -1;
		if ((*st = SslmGpuSeqDecodeStepForG5Bridge(x.ctx, x.seq, kStepToken, budget, tok)) != OK) return false;
		*blob = SaveBlob(x.ctx, x.seq);
		return !blob->empty();
	}
	bool Open() override {
		budget = static_cast<uint32_t>(OptInt("budget", 0xFFFFFFFFll));
		if (!OpenStacks()) return false;
		SslmGpuStatus st;
		int32_t t2 = -1;
		std::vector<uint8_t> b2;
		return Run(s[0], &ref_token, &ref_blob, &st) && Run(s[1], &t2, &b2, &st) && t2 == ref_token && b2 == ref_blob;
	}
	bool Clean(int which, SslmGpuStatus* st) override {
		int32_t t = -1;
		std::vector<uint8_t> b;
		return Run(s[which - 1], &t, &b, st) && t == ref_token && b == ref_blob;
	}
	bool PreArmed() override { return Pre(s[0]) == OK; }
	void Armed() override {
		own_token = -1;
		g_phase.store(PH_CALL);
		own = SslmGpuSeqDecodeStepForG5Bridge(s[0].ctx, s[0].seq, kStepToken, budget, &own_token);
		AfterPublic();
		g_phase.store(PH_NONE);
	}
	bool Judge(std::string* d) override {
		*d = std::string("bridge=") + St(own);
		return own == DEVICE_LOST;
	}
	SslmGpuStatus ArmedStatus() const override { return own; }
};

// batch: three sequences of one context, one sslm_decode_step_batch_gpu call.
struct BatchRoute : DecodeBase {
	SslmGpuSequenceHandle* seqs[2][3] = {};
	int32_t ref_tok[3] = {-1, -1, -1};
	std::vector<uint8_t> ref_blob[3];
	SslmGpuStatus call = OK, sts[3] = {OK, OK, OK};
	// Each sequence: reset, 5-token prefill, embed. The first refusal.
	SslmGpuStatus Prep(int i) {
		Stack& x = s[i];
		for (int j = 0; j < 3; ++j) {
			SslmGpuStatus st = sslm_gpu_seq_reset(x.ctx, seqs[i][j]);
			if (st == OK)
				st = SslmGpuSeqPrefillPromptForG5Bridge(x.ctx, seqs[i][j], prompt.data(), static_cast<int32_t>(prompt.size()), 64);
			if (st == OK) st = sslm_gpu_seq_embed_token(x.ctx, seqs[i][j], kStepToken);
			if (st != OK) return st;
		}
		return OK;
	}
	// A clean batch on context i, compared per sequence with the references (or recorded as them).
	bool Run(int i, bool record, SslmGpuStatus* st) {
		if ((*st = Prep(i)) != OK) return false;
		SslmGpuStatus b[3] = {DEVICE_LOST, DEVICE_LOST, DEVICE_LOST};
		const SslmGpuStatus c = sslm_decode_step_batch_gpu(s[i].ctx, seqs[i], nullptr, 3, 0xFFFFFFFFu, b);
		*st = FirstNotOk({c, b[0], b[1], b[2]});
		if (*st != OK) return false;
		for (int j = 0; j < 3; ++j) {
			int32_t t = -1;
			if ((*st = SslmGpuSeqFinishTokenForG5Bridge(s[i].ctx, seqs[i][j], &t)) != OK) return false;
			std::vector<uint8_t> blob = SaveBlob(s[i].ctx, seqs[i][j]);
			if (blob.empty()) return false;
			if (record) {
				ref_tok[j] = t;
				ref_blob[j] = blob;
			} else if (t != ref_tok[j] || blob != ref_blob[j]) {
				return false;
			}
		}
		return true;
	}
	bool Open() override {
		if (!OpenStacks()) return false;
		for (int i = 0; i < 2; ++i) {
			seqs[i][0] = s[i].seq;
			if (sslm_gpu_seq_create(s[i].ctx, s[i].model, 64, &seqs[i][1]) != OK ||
			    sslm_gpu_seq_create(s[i].ctx, s[i].model, 64, &seqs[i][2]) != OK)
				return false;
		}
		SslmGpuStatus st;
		if (!Run(0, true, &st)) return false;
		// The three sequences decode the same prompt, so they must agree with each other too.
		if (ref_tok[1] != ref_tok[0] || ref_tok[2] != ref_tok[0]) return false;
		return Run(1, false, &st);
	}
	bool Clean(int which, SslmGpuStatus* st) override { return Run(which - 1, false, st); }
	bool PreArmed() override { return Prep(0) == OK; }
	void Armed() override {
		for (auto& x : sts) x = DEVICE_LOST;
		g_phase.store(PH_CALL);
		call = sslm_decode_step_batch_gpu(s[0].ctx, seqs[0], nullptr, 3, 0xFFFFFFFFu, sts);
		AfterPublic();
		g_phase.store(PH_NONE);
	}
	bool Judge(std::string* d) override {
		*d = std::string("batch=") + St(call) + " slots=" + St(sts[0]) + "," + St(sts[1]) + "," + St(sts[2]);
		if (call == DEVICE_LOST) return true;  // the call itself refused: rule 3 at the boundary
		if (call != OK) return false;
		// Sec3.2 E-6: the faulted sequence reads SSLM_DEVICE_LOST and the batch marks every later one the
		// same; the sequences before it completed.
		int j = -1;
		for (int i = 0; i < 3; ++i)
			if (sts[i] != OK) {
				j = i;
				break;
			}
		if (j < 0) return false;
		for (int i = j; i < 3; ++i)
			if (sts[i] != DEVICE_LOST) return false;
		return true;
	}
	SslmGpuStatus ArmedStatus() const override { return FirstNotOk({call, sts[0], sts[1], sts[2]}); }
};

// The one frame both passes enter the route's call through (see g_stack_stop).
__declspec(noinline) void ArmedPass(Route* r) {
	g_stack_stop = _ReturnAddress();
	r->Armed();
}

// The phase names --expect-phase accepts are te425_harness.h's PhaseName values.
int PhaseByName(const std::string& n) {
	for (int p = 0; p <= 6; ++p)
		if (n == PhaseName(p)) return p;
	return -1;
}

std::string KeyDesc(const FxLabel& L) {
	return std::string("ph=") + PhaseName(L.key.phase) + " ev#" + std::to_string(L.key.ev_ord) + ":" +
	       EvName(L.ev_last) + " +" + std::to_string(L.since) + " stack=" + Hex(L.key.stack & 0xFFFFFFFFull) + "#" + std::to_string(L.key.occ) + " ecl=" + std::to_string(L.ecl) +
	       " outstanding=" + std::to_string(L.outstanding) + (L.aligned ? " aligned" : "");
}

// Where the process is, for the crash and terminate handlers.
std::atomic<int> g_stage{0};  // 0 setup, 1 armed (before the fire), 2 after the fire
std::atomic<uint32_t> g_stage_site{0};
const char* g_cell = "FX";

void OnTerminate() {
	const bool after = g_fire.fired;
	std::printf("%s k=%u TERMINATE: std::terminate was called %s the injected throw%s\n", g_cell, g_stage_site.load(),
	            after ? "after" : "before", after ? " (a foreign exception reached a noexcept frame)" : "");
	std::printf("SUMMARY %s legs=1 conform=0 nonconform=%d invalid=%d\nVERDICT %s %s\n", g_cell, after ? 1 : 0,
	            after ? 0 : 1, g_cell, after ? "RED" : "INVALID");
	std::fflush(stdout);
	TerminateProcess(GetCurrentProcess(), after ? 1u : 4u);
}
LONG WINAPI OnCrash(EXCEPTION_POINTERS* e) {
	const bool after = g_fire.fired;
	std::printf("%s k=%u CRASH: exception 0x%08lx %s the injected throw\n", g_cell, g_stage_site.load(),
	            static_cast<unsigned long>(e->ExceptionRecord->ExceptionCode), after ? "after" : "before");
	std::printf("SUMMARY %s legs=1 conform=0 nonconform=%d invalid=%d\nVERDICT %s %s\n", g_cell, after ? 1 : 0,
	            after ? 0 : 1, g_cell, after ? "RED" : "INVALID");
	std::fflush(stdout);
	TerminateProcess(GetCurrentProcess(), after ? 1u : 4u);
	return EXCEPTION_EXECUTE_HANDLER;
}

int CellSweep() {
	const std::string route = Opt("route", "prompt");
	static std::string cell_name;
	cell_name = "FX." + route;
	const char* cell = cell_name.c_str();
	g_cell = cell;
	std::string why;
	if (!InstallFxHooks(&why)) return SetupFail(cell, why.c_str());
	std::thread(Watchdog).detach();
	std::unique_ptr<Route> r;
	if (route == "prompt") r.reset(new PromptRoute);
	else if (route == "schema") r.reset(new SchemaRoute);
	else if (route == "step") r.reset(new StepRoute);
	else if (route == "bridge") r.reset(new BridgeRoute);
	else if (route == "batch") r.reset(new BatchRoute);
	else return SetupFail(cell, "--route must be prompt, schema, step, bridge or batch");
	if (!r->Open()) return SetupFail(cell, "fixture (reference calls on both contexts)");

	// Counting pass: every operator-new site of the armed call, labelled. It starts from exactly the state
	// every armed pass starts from -- a clean call on the same context, then the route's pre-state -- so
	// that the two passes number the same calls the same way.
	SslmGpuStatus count_pre = OK;
	if (!r->Clean(1, &count_pre) || !r->PreArmed()) return SetupFail(cell, "pre-state for the counting pass");
	Begin();
	NewCountOnly(true);
	ArmedPass(r.get());
	NewStop();
	const uint32_t K = g_new.count.load();
	if (K >= kMaxSites) return SetupFail(cell, "more sites than the label table holds");
	uint32_t outstanding_sites = 0;
	for (uint32_t k = 1; k <= K; ++k) outstanding_sites += g_fl[k].outstanding;
	std::printf("%s sites=%u outstanding=%u counting-pass=%s (len=%lld budget=%s)\n", cell, K, outstanding_sites,
	            St(r->ArmedStatus()), OptInt("len", 5), Opt("budget", "all").c_str());
	if (r->ArmedStatus() != OK) return SetupFail(cell, "counting pass not clean");
	if (Has("count")) {
		for (uint32_t k = 1; k <= K; ++k) std::printf("SITE %s k=%u %s\n", route.c_str(), k, KeyDesc(g_fl[k]).c_str());
		std::printf("SITES %s K=%u\nVERDICT %s COUNTED\n", route.c_str(), K, cell);
		std::fflush(stdout);
		return 0;
	}

	std::vector<uint32_t> sites;
	if (Has("site")) {
		sites.push_back(static_cast<uint32_t>(OptInt("site", 0)));
	} else if (Opt("select") == "outstanding") {
		// Every site at which a submission is outstanding: the gated legs.
		for (uint32_t k = 1; k <= K; ++k)
			if (g_fl[k].outstanding) sites.push_back(k);
		if (sites.empty()) return SetupFail(cell, "--select=outstanding: the route has no outstanding site");
	} else {
		const long long from = OptInt("from", 1), to = OptInt("to", K);
		for (long long k = from; k <= to && k <= K; ++k) sites.push_back(static_cast<uint32_t>(k));
	}
	const int expect_phase = Has("expect-phase") ? PhaseByName(Opt("expect-phase")) : -1;
	if (Has("expect-phase") && expect_phase < 0) return SetupFail(cell, "--expect-phase names no phase");
	uint32_t legs = 0, pass = 0, fail = 0, invalid = 0;
	for (uint32_t k : sites) {
		if (k < 1 || k > K) {
			std::printf("%s k=%u beyond the route's %u sites -> INVALID\n", cell, k, K);
			++legs;
			++invalid;
			continue;
		}
		const FxLabel want = g_fl[k];
		if (expect_phase >= 0 && want.key.phase != expect_phase) {
			std::printf("%s k=%u %s: not in phase %s -> INVALID\n", cell, k, KeyDesc(want).c_str(), Opt("expect-phase").c_str());
			++legs;
			++invalid;
			continue;
		}
		g_stage_site.store(k);
		// A clean call first: the leg starts from a sequence and a device proven clean.
		SslmGpuStatus pre_st = OK;
		if (!r->Clean(1, &pre_st) || !r->PreArmed()) {
			std::printf("%s k=%u pre-state not clean: %s -> INVALID\n", cell, k, St(pre_st));
			++legs;
			++invalid;
			continue;
		}
		const bool gated = want.outstanding != 0;
		g_fire = FireState{};
		g_reading = Reading{};
		g_gate.watchdog_fired.store(false);
		Begin();
		if (gated) g_gate.arm_ecl.store(want.ecl);
		g_fire_key = want.key;
		g_fire_armed.store(true);
		g_new.label.store(false);
		g_new.count.store(0);
		g_stage.store(1);
		g_new.counting.store(true);
		bool escaped = false;
		try {
			ArmedPass(r.get());
		} catch (const ForeignFault&) {
			escaped = true;  // the exception crossed a public entry point into its caller
			AfterPublic();
		}
		NewStop();
		g_fire_armed.store(false);
		g_stage.store(2);
		const bool released = Release();
		const bool watchdog = g_gate.watchdog_fired.load();

		// The label matched (that is what fires); the absolute number may differ by the engine's intermittent
		// allocation, and is printed, not required.
		const bool same_site = g_fire.fired && g_fire.label.key == want.key;
		const bool drove = same_site && !watchdog &&
		                   (!gated || (g_fire.gate_closed && g_fire.have_sub && g_reading.taken && g_reading.have_sub));
		std::string own;
		const bool own_ok = r->Judge(&own) && !escaped;
		const bool complete = !gated || g_reading.Complete();
		SslmGpuStatus s2 = OK, s1 = OK;
		const bool a2 = r->Clean(2, &s2);
		const bool a1 = r->Clean(1, &s1);
		++legs;
		if (!drove) {
			++invalid;
			std::printf("%s k=%u %s fired=%d fired-at=%u gated=%d gate-closed-at-fire=%d have-sub=%d watchdog=%d -> INVALID "
			            "(own=%s after2=%s after1=%s)\n",
			            cell, k, KeyDesc(want).c_str(), g_fire.fired ? 1 : 0, g_fire.count, gated ? 1 : 0,
			            g_fire.gate_closed ? 1 : 0, g_fire.have_sub ? 1 : 0, watchdog ? 1 : 0, own.c_str(), St(s2), St(s1));
			continue;
		}
		const bool conforms = own_ok && complete && a2 && a1;
		if (conforms) ++pass;
		else ++fail;
		std::string gate;
		if (gated) {
			gate = std::string(" completed-at-return=") + std::to_string(g_reading.completed) + "/" +
			       std::to_string(g_reading.value) + (complete ? "" : "(!)") + " gate-shut-at-return=" +
			       std::to_string(g_reading.gate_shut ? 1 : 0) + " release-confirmed=" + std::to_string(released ? 1 : 0);
		}
		std::printf("%s k=%u %s fired-at=%u own=%s%s%s%s after2=%s%s after1=%s%s -> %s\n", cell, k, KeyDesc(want).c_str(),
		            g_fire.count, own.c_str(),
		            own_ok ? "" : "(!)", escaped ? " ESCAPED-THE-API" : "", gate.c_str(), St(s2), a2 ? "" : "(!)", St(s1),
		            a1 ? "" : "(!)", conforms ? "PASS" : "FAIL");
		std::fflush(stdout);
	}
	std::printf("SUMMARY %s legs=%u conform=%u nonconform=%u invalid=%u\n", cell, legs, pass, fail, invalid);
	const char* v = invalid ? "INVALID" : fail ? "RED" : "GREEN";
	std::printf("VERDICT %s %s\n", cell, v);
	std::fflush(stdout);
	return invalid ? 4 : fail ? 1 : 0;
}
}  // namespace

int main(int argc, char** argv) {
	std::setvbuf(stdout, nullptr, _IOLBF, 1 << 16);
	for (int i = 1; i < argc; ++i) {
		std::string a = argv[i];
		if (a.rfind("--", 0) != 0) continue;
		const size_t eq = a.find('=');
		if (eq == std::string::npos) g_opt[a.substr(2)] = "1";
		else g_opt[a.substr(2, eq - 2)] = a.substr(eq + 1);
	}
	std::set_terminate(OnTerminate);
	SetUnhandledExceptionFilter(OnCrash);
	const auto t0 = std::chrono::steady_clock::now();
	const int rc = CellSweep();
	std::printf("WALL fx %.1f s\n", std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
	std::fflush(stdout);
	std::fflush(stderr);
	// Leave without running static destructors, as the TE-425 cells do: a leg may leave a handle wedged.
	TerminateProcess(GetCurrentProcess(), static_cast<UINT>(rc));
	return rc;
}

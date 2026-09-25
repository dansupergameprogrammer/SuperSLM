// TE-433 -- the deterministic pin for TE-432 change 2: each submission tail's final catch (...).
//
// Plan: Claude/Plans/te421-slm172-host-oom.md (records tree), Sec3.2 E-4 (a submission tail contains
// every exception type), Sec3.3 (the status contract), Sec3.5 R10 (a foreign exception).
// Record: Claude/Curie/te433-slm180-pins-2026-09-25.md (records tree).
//
// THE CLAIM. A foreign exception or a std::length_error thrown in a submission tail AFTER
// ExecuteCommandLists and the fence Signal -- at the in-flight token's allocation, the only operator
// new between the Signal and the tail's return -- must not leave the call before the submission it
// follows has completed: the stack is about to release buffers that work reads and writes. The two
// tails are RunLayerLoopGpuSubmit's (T13; reached by sslm_decode_step_gpu) and
// SubmitOneSubChunkToFullDepthForG5Bridge's (T15; reached by every prompt-prefill sub-chunk).
//
// WHY A GATE. R10's last-site legs detect a missing catch-all only through the device removal the
// early release causes, and only if the GPU is still running the submission when the unwind frees its
// buffers: a race the unfixed engine can win (TE-432 Sec5). This cell removes the race. When the armed
// leg reaches the tail's ExecuteCommandLists, a hook inserts ID3D12CommandQueue::Wait on a harness
// fence ahead of the tail's own command list, so the submission cannot start, let alone complete, on
// its own. The gate opens only when the engine itself asks to be told of that submission's completion
// (ID3D12Fence::SetEventOnCompletion on the tail's fence at a value no lower than the tail's Signal).
// The oracle reads the fence's completed value the instant the public call returns:
//   - an engine that waits the submission out asks for the event, the gate opens, the GPU runs the
//     submission, and the call returns after it completed: completed >= signalled value, on any host;
//   - an engine that lets the exception leave the tail never asks, the gate is still shut when the call
//     returns, and the submission cannot have completed: completed < signalled value, on any host.
// Neither reading depends on how fast the GPU is. After the reading the harness opens a still-shut
// gate itself and waits the submission out, so later calls in the process are not held behind it.
//
// Every leg also holds the plan's Sec3.3 status (foreign: SSLM_DEVICE_LOST; length_error:
// SSLM_GPU_ALLOCATION_FAILED; a failed prefill's read in the same loop SSLM_PREFILL_HIDDEN_UNAVAILABLE),
// and the second context's and the same handle's next call are SSLM_OK and byte-equal.
//
// WHAT THIS DOES NOT CLOSE. An engine that asks for the completion event and returns without blocking
// on it would open the gate and could read green if the GPU finished before the reading. That is not
// the defect this cell pins (a missing catch clause never asks), and no engine version has had it.
//
// Usage: te433_tail_pin.exe --route=prompt|step --kind=foreign|length_error [--tail=first|last]
//                            [--len=N] [--fire-delay-ms=N] --qwen3=PATH
// Exit status: 0 every leg conforms; 1 at least one does not (RED); 3 setup failed; 4 the leg could not
// be driven (the fault did not fire at the selected tail) -- an instrument that cannot decide never
// reads as a verdict. Output ends with SUMMARY and VERDICT lines in the TE-425 suite's format.
#include "te425_harness.h"

#include <map>

using namespace te425;

namespace {
// Tracked D3D12 events, in call order.
enum Ev : uint8_t { EV_NONE = 0, EV_CLOSE, EV_LRESET, EV_ECL, EV_SIGNAL, EV_SETEVENT, EV_COUNT };
const char* EvName(int e) {
	static const char* const k[] = {"none", "Close", "ListReset", "ExecuteCommandLists", "Signal",
	                                "SetEventOnCompletion"};
	return (e >= 0 && e < EV_COUNT) ? k[e] : "?";
}
std::atomic<int> g_ev_last{EV_NONE};
std::atomic<int> g_ev_prev{EV_NONE};
std::atomic<uint32_t> g_ecl_n{0};  // ExecuteCommandLists calls since Begin()
std::atomic<uint32_t> g_hook_calls[EV_COUNT];
std::atomic<uint32_t> g_since_ev{0};  // operator-new calls since the last tracked event
void Event(int e) {
	g_hook_calls[e].fetch_add(1);
	g_ev_prev.store(g_ev_last.load());
	g_ev_last.store(e);
	g_since_ev.store(0);
}

// Per operator-new site: the two tracked events before it, its 1-based position among the
// operator-new calls since the last of them, the ExecuteCommandLists ordinal, the phase.
struct PinLabel {
	uint8_t last, prev, phase;
	uint32_t since;
	uint32_t ecl;
};
PinLabel g_pl[kMaxSites];

// The gate (see the file comment).
struct Gate {
	std::atomic<uint32_t> arm_ecl{0};  // the ExecuteCommandLists ordinal (since Begin) to gate; 0 = off
	Microsoft::WRL::ComPtr<ID3D12Fence> fence;  // the harness's gate fence, on the engine's device
	UINT64 target = 0;                          // the value the gated queue waits for
	std::atomic<bool> closed{false};
	std::atomic<bool> await_signal{false};      // the next Signal after the gated list is the tail's
	ID3D12Fence* sub_fence = nullptr;           // the tail's fence and signalled value
	UINT64 sub_value = 0;
	std::atomic<bool> opened_by_engine{false};
	UINT64 engine_wait_value = 0;
	HRESULT wait_hr = S_OK;
};
Gate g_gate;

constexpr uint32_t kFireAfterGatedSignal = 0xFFFFFFFFu;
DWORD g_fire_delay_ms = 0;

// What the harness saw when the injected exception fired.
struct FireState {
	bool fired = false;
	uint32_t ecl = 0;
	int last = EV_NONE, prev = EV_NONE;
	bool gated = false;
	bool have_sub = false;
};
FireState g_fire;
}  // namespace

// =============================================================================================
// Fault source: the replaced operator new family (TE-425's, plus the tracked-event label).
// =============================================================================================
static void* PinAlloc(size_t n, size_t align, bool aligned) {
	if (g_new.counting.load(std::memory_order_relaxed)) {
		const uint32_t c = g_new.count.fetch_add(1) + 1;
		const uint32_t since = g_since_ev.fetch_add(1) + 1;
		if (g_new.label.load(std::memory_order_relaxed) && c < kMaxSites) {
			g_pl[c] = PinLabel{static_cast<uint8_t>(g_ev_last.load()), static_cast<uint8_t>(g_ev_prev.load()),
			                   static_cast<uint8_t>(g_phase.load()), since, g_ecl_n.load()};
		}
		// Armed (fail_at == kFireAfterGatedSignal): the first allocation after the gated submission's own
		// Signal, selected by structure rather than by number, since the allocation count before it is
		// not reproduced exactly from call to call.
		const uint32_t f = g_new.fail_at.load();
		const bool at_tail = f == kFireAfterGatedSignal && g_gate.sub_fence != nullptr &&
		                     !g_gate.await_signal.load() && g_ev_last.load() == EV_SIGNAL && since == 1;
		if (at_tail) {
			g_new.fail_at.store(0);
			g_new.fired.store(c);
			g_fire.fired = true;
			g_fire.ecl = g_ecl_n.load();
			g_fire.last = g_ev_last.load();
			g_fire.prev = g_ev_prev.load();
			g_fire.gated = g_gate.closed.load();
			g_fire.have_sub = g_gate.sub_fence != nullptr && !g_gate.await_signal.load();
			// --fire-delay-ms: a slow host, standing in for the delay that let the unfixed engine win the
			// R10 race (TE-432 Sec5). The gate keeps the submission from completing however long this is.
			if (g_fire_delay_ms) Sleep(g_fire_delay_ms);
			switch (static_cast<NewFault>(g_new.kind.load())) {
				case NewFault::LengthError: throw std::length_error("te433 injected length_error");
				case NewFault::Foreign: throw ForeignFault{c};
				case NewFault::RuntimeError: throw std::runtime_error("te433 injected runtime_error");
				default: throw std::bad_alloc();
			}
		}
	}
	void* p = aligned ? _aligned_malloc(n ? n : 1, align) : std::malloc(n ? n : 1);
	if (!p) throw std::bad_alloc();
	return p;
}
void* operator new(size_t n) { return PinAlloc(n, 0, false); }
void* operator new[](size_t n) { return PinAlloc(n, 0, false); }
void* operator new(size_t n, const std::nothrow_t&) noexcept {
	try {
		return PinAlloc(n, 0, false);
	} catch (...) {
		return nullptr;
	}
}
void* operator new[](size_t n, const std::nothrow_t&) noexcept {
	try {
		return PinAlloc(n, 0, false);
	} catch (...) {
		return nullptr;
	}
}
void* operator new(size_t n, std::align_val_t a) { return PinAlloc(n, static_cast<size_t>(a), true); }
void* operator new[](size_t n, std::align_val_t a) { return PinAlloc(n, static_cast<size_t>(a), true); }
void* operator new(size_t n, std::align_val_t a, const std::nothrow_t&) noexcept {
	try {
		return PinAlloc(n, static_cast<size_t>(a), true);
	} catch (...) {
		return nullptr;
	}
}
void* operator new[](size_t n, std::align_val_t a, const std::nothrow_t&) noexcept {
	try {
		return PinAlloc(n, static_cast<size_t>(a), true);
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
// D3D12 vtable hooks: Close, list Reset, ExecuteCommandLists, Signal, SetEventOnCompletion.
// Slots are the SDK's C vtable order (Windows Kits 10.0.26100.0 d3d12.h): GraphicsCommandList 9 Close,
// 10 Reset; CommandQueue 10 ExecuteCommandLists, 14 Signal; Fence 9 SetEventOnCompletion. The TE-425
// harness patches Close/Reset/Signal/SetEventOnCompletion at the same slots.
// =============================================================================================
namespace {
using PFN_CLOSE = HRESULT(STDMETHODCALLTYPE*)(ID3D12GraphicsCommandList*);
using PFN_LRESET = HRESULT(STDMETHODCALLTYPE*)(ID3D12GraphicsCommandList*, ID3D12CommandAllocator*,
                                               ID3D12PipelineState*);
using PFN_ECL = void(STDMETHODCALLTYPE*)(ID3D12CommandQueue*, UINT, ID3D12CommandList* const*);
using PFN_SIGNAL = HRESULT(STDMETHODCALLTYPE*)(ID3D12CommandQueue*, ID3D12Fence*, UINT64);
using PFN_SETEVENT = HRESULT(STDMETHODCALLTYPE*)(ID3D12Fence*, UINT64, HANDLE);
PFN_CLOSE o_close;
PFN_LRESET o_lreset;
PFN_ECL o_ecl;
PFN_SIGNAL o_signal;
PFN_SETEVENT o_setevent;

HRESULT STDMETHODCALLTYPE H_close(ID3D12GraphicsCommandList* self) {
	Event(EV_CLOSE);
	return o_close(self);
}
HRESULT STDMETHODCALLTYPE H_lreset(ID3D12GraphicsCommandList* self, ID3D12CommandAllocator* a,
                                   ID3D12PipelineState* p) {
	Event(EV_LRESET);
	return o_lreset(self, a, p);
}
void STDMETHODCALLTYPE H_ecl(ID3D12CommandQueue* self, UINT n, ID3D12CommandList* const* lists) {
	Event(EV_ECL);
	const uint32_t ord = g_ecl_n.fetch_add(1) + 1;
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
				g_gate.closed.store(true);
			}
		}
	}
	o_ecl(self, n, lists);
}
HRESULT STDMETHODCALLTYPE H_signal(ID3D12CommandQueue* self, ID3D12Fence* fence, UINT64 v) {
	Event(EV_SIGNAL);
	if (g_gate.await_signal.load()) {
		g_gate.sub_fence = fence;
		g_gate.sub_value = v;
		g_gate.await_signal.store(false);
	}
	return o_signal(self, fence, v);
}
HRESULT STDMETHODCALLTYPE H_setevent(ID3D12Fence* self, UINT64 v, HANDLE e) {
	Event(EV_SETEVENT);
	if (g_gate.closed.load() && !g_gate.await_signal.load() && self == g_gate.sub_fence && v >= g_gate.sub_value) {
		// The engine asked to be told of the gated submission's completion: let the GPU run it.
		g_gate.opened_by_engine.store(true);
		g_gate.engine_wait_value = v;
		g_gate.closed.store(false);
		(void)g_gate.fence->Signal(g_gate.target);
	}
	return o_setevent(self, v, e);
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

bool InstallPinHooks(std::string* why) {
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
	bool ok = Patch(p_list.Get(), 9, reinterpret_cast<void*>(&H_close), &o_close) &&
	          Patch(p_list.Get(), 10, reinterpret_cast<void*>(&H_lreset), &o_lreset) &&
	          Patch(p_queue.Get(), 10, reinterpret_cast<void*>(&H_ecl), &o_ecl) &&
	          Patch(p_queue.Get(), 14, reinterpret_cast<void*>(&H_signal), &o_signal) &&
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
	           SUCCEEDED(p_alloc->Reset()) && SUCCEEDED(p_list->Reset(p_alloc.Get(), nullptr)) &&
	           SUCCEEDED(p_list->Close());
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

// =============================================================================================
// Options and verdicts (the TE-425 suite's shape).
// =============================================================================================
std::map<std::string, std::string> g_opt;
std::string Opt(const char* k, const char* d = "") {
	auto it = g_opt.find(k);
	return it == g_opt.end() ? std::string(d) : it->second;
}
int SetupFail(const char* cell, const char* what) {
	std::printf("SETUP-FAIL %s: %s\nVERDICT %s INVALID\n", cell, what, cell);
	std::fflush(stdout);
	return 3;
}

// Start of a counted or armed call: operator-new counting and the ExecuteCommandLists ordinal both
// restart here, so the counting pass and the armed pass number the same calls the same way.
void Begin() {
	g_ecl_n.store(0);
	g_ev_last.store(EV_NONE);
	g_ev_prev.store(EV_NONE);
}

// The reading taken the instant the armed public call returns.
struct Reading {
	bool have_sub = false;
	UINT64 value = 0, completed = 0;
	bool gate_shut_at_return = false;
	bool opened_by_engine = false;
	bool Complete() const { return have_sub && completed >= value; }
};
Reading Take() {
	Reading r;
	r.have_sub = g_gate.sub_fence != nullptr && !g_gate.await_signal.load();
	if (r.have_sub) {
		r.value = g_gate.sub_value;
		r.completed = g_gate.sub_fence->GetCompletedValue();
	}
	r.gate_shut_at_return = g_gate.closed.load();
	r.opened_by_engine = g_gate.opened_by_engine.load();
	return r;
}
// After the reading: open a gate the engine left shut and wait the gated submission out, so later
// calls in this process do not queue behind it. Returns false when the wait cannot be confirmed.
bool Release() {
	bool ok = true;
	if (g_gate.closed.load()) {
		g_gate.closed.store(false);
		(void)g_gate.fence->Signal(g_gate.target);
	}
	g_gate.arm_ecl.store(0);
	g_gate.await_signal.store(false);
	if (g_gate.sub_fence && g_gate.sub_fence->GetCompletedValue() < g_gate.sub_value) {
		HANDLE ev = CreateEventW(nullptr, FALSE, FALSE, nullptr);
		ok = SUCCEEDED(g_gate.sub_fence->SetEventOnCompletion(g_gate.sub_value, ev)) &&
		     WaitForSingleObject(ev, 10000) == WAIT_OBJECT_0;
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

// =============================================================================================
// The two routes. prompt: U3's per-document loop (reset, prompt prefill, read) on a two-sub-chunk
// prompt -- the prefill's sub-chunks each end in SubmitOneSubChunkToFullDepthForG5Bridge's tail (T15).
// step: one decode step through the per-token primitives (embed, sslm_decode_step_gpu, ready) after a
// 5-token prefill -- the step ends in RunLayerLoopGpuSubmit's tail (T13).
// =============================================================================================
constexpr int32_t kStepToken = 785;

struct Fixture {
	Artifact art;
	Stack s1, s2;
	std::vector<int32_t> toks;  // the prompt (both routes' prefill)
	Frame ref, f;
	int32_t ref_token = -1;
	std::vector<uint8_t> ref_blob;
	bool step_route = false;

	bool Open(const std::string& path, int len, bool step) {
		step_route = step;
		if (!art.Load(path)) return false;
		if (!s1.Open(art, 64) || !s2.Open(art, 64)) return false;
		toks = QwenPrompt(len);
		ref.Size(s1.hidden);
		f.Size(s1.hidden);
		if (!step_route) {
			if (PromptLoop(s1.ctx, s1.seq, toks, &ref).First() != OK) return false;
			return PromptLoop(s2.ctx, s2.seq, toks, &f).First() == OK && f.Same(ref);
		}
		SslmGpuStatus st;
		int32_t t2 = -1;
		std::vector<uint8_t> b2;
		return Run(s1, &ref_token, &ref_blob, &st) && Run(s2, &t2, &b2, &st) && t2 == ref_token && b2 == ref_blob;
	}
	// The step route's state before the call under test.
	bool Pre(Stack& s) {
		if (sslm_gpu_seq_reset(s.ctx, s.seq) != OK) return false;
		return SslmGpuSeqPrefillPromptForG5Bridge(s.ctx, s.seq, toks.data(), static_cast<int32_t>(toks.size()), 64) == OK;
	}
	// The step route's whole call: embed, step, ready; the status and the finished token and state.
	bool Run(Stack& s, int32_t* tok, std::vector<uint8_t>* blob, SslmGpuStatus* st) {
		if (!Pre(s)) {
			*st = DEVICE_LOST;
			return false;
		}
		*st = sslm_gpu_seq_embed_token(s.ctx, s.seq, kStepToken);
		if (*st == OK) *st = sslm_decode_step_gpu(s.ctx, s.seq, nullptr, 0xFFFFFFFFu);
		SslmGpuStatus out = OK;
		if (*st == OK) {
			int32_t ready = 0;
			*st = sslm_gpu_ready(s.ctx, s.seq, 1, &ready, &out);
			if (*st == OK) *st = out;
		}
		if (*st != OK) return false;
		*tok = -1;
		if (SslmGpuSeqFinishTokenForG5Bridge(s.ctx, s.seq, tok) != OK) return false;
		*blob = SaveBlob(s.ctx, s.seq);
		return !blob->empty();
	}
	// A clean next call on stack s: SSLM_OK and byte-equal to the reference.
	bool Clean(Stack& s, SslmGpuStatus* st) {
		if (!step_route) {
			LoopOut o = PromptLoop(s.ctx, s.seq, toks, &f);
			*st = o.First();
			return *st == OK && f.Same(ref);
		}
		int32_t t = -1;
		std::vector<uint8_t> b;
		return Run(s, &t, &b, st) && t == ref_token && b == ref_blob;
	}
};

// The call under test on s1, with the reading taken as soon as the faulted public call returns.
struct Armed {
	LoopOut loop;             // prompt route
	SslmGpuStatus embed = OK, step = OK, ready = OK;  // step route
	Reading reading;
	bool read_taken = false;
};
void CallPrompt(Fixture& fx, Armed* a, bool take) {
	Stack& s = fx.s1;
	g_phase.store(PH_RESET);
	a->loop.reset = sslm_gpu_seq_reset(s.ctx, s.seq);
	g_phase.store(PH_PREFILL);
	if (a->loop.reset == OK) {
		a->loop.prefill = SslmGpuSeqPrefillPromptForG5Bridge(s.ctx, s.seq, fx.toks.data(),
		                                                     static_cast<int32_t>(fx.toks.size()), 64);
	}
	if (take) {
		a->reading = Take();
		a->read_taken = true;
	}
	g_phase.store(PH_READ);
	std::fill(fx.f.codes.begin(), fx.f.codes.end(), static_cast<int8_t>(0x5A));
	if (a->loop.reset == OK) {
		a->loop.read = sslm_gpu_seq_read_prefill_final_hidden(s.ctx, s.seq, fx.f.codes.data(), fx.f.codes.size(),
		                                                      &fx.f.required, &fx.f.m, &fx.f.e);
	}
	g_phase.store(PH_NONE);
}
void CallStep(Fixture& fx, Armed* a, bool take) {
	Stack& s = fx.s1;
	g_phase.store(PH_STEP);
	a->embed = sslm_gpu_seq_embed_token(s.ctx, s.seq, kStepToken);
	if (a->embed == OK) a->step = sslm_decode_step_gpu(s.ctx, s.seq, nullptr, 0xFFFFFFFFu);
	if (take) {
		a->reading = Take();
		a->read_taken = true;
	}
	g_phase.store(PH_READY);
	if (a->embed == OK && a->step == OK) {
		int32_t ready = 0;
		SslmGpuStatus out = OK;
		a->ready = sslm_gpu_ready(s.ctx, s.seq, 1, &ready, &out);
		if (a->ready == OK) a->ready = out;
	}
	g_phase.store(PH_NONE);
}

int CellTailPin(const char* cell) {
	std::string why;
	if (!InstallPinHooks(&why)) return SetupFail(cell, why.c_str());
	const std::string route = Opt("route", "prompt");
	const std::string kind_s = Opt("kind", "foreign");
	const std::string tail = Opt("tail", "last");
	const bool step = route == "step";
	if (!step && route != "prompt") return SetupFail(cell, "--route must be prompt or step");
	if (kind_s != "foreign" && kind_s != "length_error") return SetupFail(cell, "--kind must be foreign or length_error");
	const NewFault kind = kind_s == "foreign" ? NewFault::Foreign : NewFault::LengthError;
	const int len = static_cast<int>(std::strtol(Opt("len", "5").c_str(), nullptr, 10));
	g_fire_delay_ms = static_cast<DWORD>(std::strtoul(Opt("fire-delay-ms", "0").c_str(), nullptr, 10));
	Fixture fx;
	if (!fx.Open(Opt("qwen3"), len, step)) return SetupFail(cell, "fixture (reference calls on both contexts)");

	// Counting pass: every operator-new site of the call, labelled by the two tracked events before it.
	Armed count;
	if (step && !fx.Pre(fx.s1)) return SetupFail(cell, "step pre-state");
	Begin();
	NewCountOnly(true);
	if (step) CallStep(fx, &count, false);
	else CallPrompt(fx, &count, false);
	NewStop();
	const uint32_t K = g_new.count.load();
	const SslmGpuStatus count_status = step ? (count.embed != OK ? count.embed : count.step != OK ? count.step : count.ready)
	                                        : count.loop.First();
	if (count_status != OK) return SetupFail(cell, "counting pass not clean");
	// Tail sites: the FIRST operator-new call after an ExecuteCommandLists-then-Signal pair -- the
	// in-flight token's allocation, the only allocation between a tail's Signal and its return (later
	// allocations before the next D3D12 call belong to the caller) -- in the phase of the call that
	// submits (the prefill, or the step).
	const uint8_t want_phase = step ? PH_STEP : PH_PREFILL;
	std::vector<uint32_t> tails;
	for (uint32_t k = 1; k <= K && k < kMaxSites; ++k) {
		if (g_pl[k].last == EV_SIGNAL && g_pl[k].prev == EV_ECL && g_pl[k].since == 1 && g_pl[k].phase == want_phase)
			tails.push_back(k);
	}
	std::printf("%s route=%s kind=%s len=%d sites=%u tail-sites=%zu:", cell, route.c_str(), kind_s.c_str(), len, K,
	            tails.size());
	for (uint32_t k : tails) std::printf(" k=%u(ecl#%u)", k, g_pl[k].ecl);
	std::printf("\n");
	if (tails.empty()) return SetupFail(cell, "no operator-new site follows an ExecuteCommandLists+Signal pair");
	if (step && tails.size() != 1) return SetupFail(cell, "the step route must have exactly one submission tail");
	const uint32_t k = tail == "first" ? tails.front() : tails.back();
	const uint32_t ecl = g_pl[k].ecl;

	// Pre-state, proven clean, then the armed call.
	SslmGpuStatus pre_st = OK;
	if (!fx.Clean(fx.s1, &pre_st)) {
		std::printf("%s pre-state call not clean: %s\n", cell, St(pre_st));
		return SetupFail(cell, "pre-state not clean");
	}
	if (step && !fx.Pre(fx.s1)) return SetupFail(cell, "step pre-state");
	g_fire = FireState{};
	Armed a;
	Begin();
	g_gate.arm_ecl.store(ecl);
	NewArm(kFireAfterGatedSignal, kind);
	if (step) CallStep(fx, &a, true);
	else CallPrompt(fx, &a, true);
	NewStop();
	const bool released = Release();

	// The fault must have fired at the gated tail: after the gated ExecuteCommandLists and its Signal.
	const bool drove = g_fire.fired && g_fire.ecl == ecl && g_fire.last == EV_SIGNAL && g_fire.prev == EV_ECL &&
	                   g_fire.gated && g_fire.have_sub && a.read_taken && a.reading.have_sub;
	std::printf("%s counted-k=%u ecl#%u fired=%d at-site=%u fired-ecl#%u at=%s after %s gated=%d wait_hr=%s "
	            "signalled=%llu\n",
	            cell, k, ecl, g_fire.fired ? 1 : 0, g_new.fired.load(), g_fire.ecl, EvName(g_fire.last),
	            EvName(g_fire.prev), g_fire.gated ? 1 : 0,
	            Hex(static_cast<uint32_t>(g_gate.wait_hr)).c_str(), static_cast<unsigned long long>(a.reading.value));
	if (!drove) {
		std::printf("SUMMARY %s legs=1 conform=0 nonconform=0 invalid=1\nVERDICT %s INVALID\n", cell, cell);
		return 4;
	}

	// (1) The pin: the submission had completed when the faulted call returned.
	const bool complete = a.reading.Complete();
	// (2) The plan's Sec3.3 status.
	std::string own;
	bool own_ok;
	if (step) {
		const SslmGpuStatus s = a.embed != OK ? a.embed : a.step;
		own = std::string("embed=") + St(a.embed) + " step=" + St(a.step);
		own_ok = a.embed == OK && a.step == (kind == NewFault::Foreign ? DEVICE_LOST : ALLOC_FAILED);
		(void)s;
	} else {
		own = std::string("reset=") + St(a.loop.reset) + " prefill=" + St(a.loop.prefill) + " read=" + St(a.loop.read);
		own_ok = a.loop.reset == OK && a.loop.prefill == (kind == NewFault::Foreign ? DEVICE_LOST : ALLOC_FAILED) &&
		         a.loop.read == HIDDEN_UNAVAILABLE;
	}
	// (3) The second context's next call, then the same handle's: SSLM_OK and byte-equal.
	SslmGpuStatus s2 = OK, s1 = OK;
	const bool a2 = fx.Clean(fx.s2, &s2);
	const bool a1 = fx.Clean(fx.s1, &s1);
	const bool conforms = complete && own_ok && a2 && a1;
	std::printf("%s k=%u completed-at-return=%llu signalled=%llu -> %s (engine asked for the event: %s at %llu; "
	            "gate shut at return: %d; release wait confirmed: %d)\n",
	            cell, k, static_cast<unsigned long long>(a.reading.completed),
	            static_cast<unsigned long long>(a.reading.value), complete ? "COMPLETE" : "NOT-COMPLETE",
	            a.reading.opened_by_engine ? "yes" : "no", static_cast<unsigned long long>(g_gate.engine_wait_value),
	            a.reading.gate_shut_at_return ? 1 : 0, released ? 1 : 0);
	std::printf("%s k=%u own=%s%s after2=%s%s after1=%s%s -> %s\n", cell, k, own.c_str(), own_ok ? "" : "(!)", St(s2),
	            a2 ? "" : "(!)", St(s1), a1 ? "" : "(!)", conforms ? "PASS" : "FAIL");
	std::printf("SUMMARY %s legs=1 conform=%d nonconform=%d invalid=0\n", cell, conforms ? 1 : 0, conforms ? 0 : 1);
	std::printf("VERDICT %s %s\n", cell, conforms ? "GREEN" : "RED");
	std::fflush(stdout);
	return conforms ? 0 : 1;
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
	const auto t0 = std::chrono::steady_clock::now();
	const int rc = CellTailPin("TE433.tail");
	std::printf("WALL tail %.1f s\n", std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
	std::fflush(stdout);
	std::fflush(stderr);
	// Leave without running static destructors, as the TE-425 cells do.
	TerminateProcess(GetCurrentProcess(), static_cast<UINT>(rc));
	return rc;
}

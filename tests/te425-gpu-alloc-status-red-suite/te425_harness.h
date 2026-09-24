// TE-425 -- shared harness for the SuperSLM 1.8.0 (plan: "v1.7.2") GPU allocation-status red suite.
//
// Plan: Claude/Plans/te421-slm172-host-oom.md (records tree), Sec3.5 R1-R15, as folded by TE-424.
// Test-design record: Claude/Curie/te425-slm180-red-2026-09-24.md (records tree).
//
// WHAT EVERY CELL ASSERTS. The engine's status contract for an allocation failure on a live device
// (plan Sec3.3): a host allocation (std::bad_alloc, std::length_error), a D3D12 E_OUTOFMEMORY, or a
// failed first-time setup of the process's submission device returns SSLM_GPU_ALLOCATION_FAILED, the
// submission list is left closed (a SECOND context's next call is SSLM_OK and byte-equal), and the
// handle stays usable. Everything else a GPU call can fail with returns SSLM_DEVICE_LOST (rules 0, 1
// and 3 of plan Sec3.2 E-2). Every cell is written against that contract, so it is red against
// SuperSLM v1.7.1 (2a32042) wherever v1.7.1 departs from it, and green once the engine delivers it.
//
// THE THREE FAULT SOURCES, and why none of them is a seam the fix adds. The builder (TE-426) works in
// parallel and adds its own seams; a suite that needed them could not run at v1.7.1, so it could not
// be red there. Every fault source below exists at v1.7.1 and is independent of the change:
//   1. The global operator new family, replaced in te425_cells.cpp (plain, array, nothrow, and the
//      aligned forms), counting every call in a window and throwing at the k-th: std::bad_alloc,
//      std::length_error, or a foreign (non-std) type.
//   2. The engine's own test seams in the SUPERSLM_GPU_TEST_SEAMS build (gpu_port.h, T-2851):
//      ArmGpuAllocFaultAtOccurrence (the k-th TryMakeBuffer returns an HRESULT without calling D3D12)
//      and ArmGpuMapDeviceRemovedQueryInjection (the plan's rule-1 seam, E-2).
//   3. D3D12 vtable hooks (TE-419's shape), patched once per process on the runtime's own classes:
//      ID3D12Device::CreateCommittedResource/CreateComputePipelineState/CreateRootSignature/
//      CreateCommandQueue/CreateCommandAllocator/CreateCommandList/CreateFence/CreateQueryHeap,
//      ID3D12GraphicsCommandList::Close/Reset, ID3D12CommandAllocator::Reset, ID3D12Resource::Map,
//      ID3D12CommandQueue::Signal, ID3D12Fence::SetEventOnCompletion. Plan Sec3.5 R9 offered the
//      choice of this or a seam in SSLM_GPU_HR's failure branch; the hook is chosen because it is the
//      only one of the two that exists at v1.7.1, and because it fails the D3D12 call itself, so it
//      cannot tell "E-1 in the macro" from "E-1 in MakeBuffer" apart by construction -- which is the
//      mutant R9 exists to kill (plan Sec3.5 R9, "must-rejects").
//   Slot numbers are the SDK's own C vtable order (Windows Kits 10.0.26100.0 d3d12.h), and
//   InstallHooks() refuses to run a leg unless every hook it can exercise without the engine fires on
//   a probe object first.
#pragma once
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <d3d12.h>
#include <dxgi1_6.h>
#include <wrl/client.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <new>
#include <stdexcept>
#include <string>
#include <vector>

#include "superslm/gpu_1p0.h"
#include "superslm/gpu_1p0_bench_bridge.h"
#include "superslm/gpu_1p0_g5_bridge.h"
#include "superslm/model.h"
#include "superslm/schema_masks.h"

// The engine's T-2851 seams (SUPERSLM_GPU_ALLOC_FAULT_INJECTION build), declared at global scope
// exactly as tests/t2956-token-finish-red-suite declares them.
void SslmGpuAllocCounterResetForTest() noexcept;
uint32_t SslmGpuAllocCountForTest() noexcept;
void ArmGpuAllocFaultAtOccurrence(uint32_t k, long hr) noexcept;
bool SslmGpuAllocInBundleForTest(uint32_t k) noexcept;
void ArmGpuMapDeviceRemovedQueryInjection() noexcept;
void ArmGpuDeviceLogitsCloseFaultForTest(uint32_t consecutive_failures) noexcept;

namespace te425 {

using Microsoft::WRL::ComPtr;

// ---------------------------------------------------------------------------------------------
// Statuses
// ---------------------------------------------------------------------------------------------
constexpr SslmGpuStatus OK = SslmGpuStatus::SSLM_OK;
constexpr SslmGpuStatus DEVICE_LOST = SslmGpuStatus::SSLM_DEVICE_LOST;
constexpr SslmGpuStatus ALLOC_FAILED = SslmGpuStatus::SSLM_GPU_ALLOCATION_FAILED;
constexpr SslmGpuStatus SEQ_REJECTED = SslmGpuStatus::SSLM_SEQUENCE_REJECTED;
constexpr SslmGpuStatus KV_MISMATCH = SslmGpuStatus::SSLM_SEQUENCE_KV_BUFFER_MISMATCH;
constexpr SslmGpuStatus HIDDEN_UNAVAILABLE = SslmGpuStatus::SSLM_PREFILL_HIDDEN_UNAVAILABLE;

inline const char* St(SslmGpuStatus s) {
	switch (static_cast<uint32_t>(s)) {
		case 0: return "OK";
		case 1: return "DISPATCH_BUDGET_TOO_SMALL";
		case 2: return "BUSY";
		case 3: return "CONTEXT_HAS_LIVE_HANDLES";
		case 4: return "MODEL_HAS_LIVE_SEQUENCES";
		case 5: return "ADAPTER_MODEL_MISMATCH";
		case 6: return "ADAPTER_BASE_HASH_MISMATCH";
		case 7: return "KV_BUFFER_MISMATCH";
		case 8: return "DEVICE_LOST";
		case 9: return "BATCH_BUDGET_EXHAUSTED";
		case 10: return "TOKEN_ID_OUT_OF_RANGE";
		case 11: return "SEQUENCE_REJECTED";
		case 12: return "RESTORE_MODEL_MISMATCH";
		case 13: return "MODEL_HAS_LIVE_ADAPTERS";
		case 14: return "ADAPTER_HAS_BOUND_SEQUENCES";
		case 15: return "SHADER_BINARY_STALE";
		case 16: return "ALLOCATION_FAILED";
		case 17: return "OUTPUT_BUFFER_TOO_SMALL";
		case 18: return "PREFILL_HIDDEN_UNAVAILABLE";
		default: return "status#?";
	}
}

// ---------------------------------------------------------------------------------------------
// Fault source 1: the operator new family (definitions in te425_cells.cpp).
// ---------------------------------------------------------------------------------------------
enum class NewFault : int { BadAlloc = 0, LengthError = 1, Foreign = 2, RuntimeError = 3 };
// A type no engine catch clause names, and not derived from std::exception (plan Sec3.5 R10).
struct ForeignFault {
	uint32_t site;
};

// D3D12 events, used both as hook ids and as the "last event before this allocation" site label.
enum Method : int {
	M_NONE = 0,
	M_CCR,        // ID3D12Device::CreateCommittedResource
	M_CLOSE,      // ID3D12GraphicsCommandList::Close
	M_LRESET,     // ID3D12GraphicsCommandList::Reset
	M_ARESET,     // ID3D12CommandAllocator::Reset
	M_MAP,        // ID3D12Resource::Map
	M_PSO,        // ID3D12Device::CreateComputePipelineState
	M_ROOTSIG,    // ID3D12Device::CreateRootSignature
	M_SIGNAL,     // ID3D12CommandQueue::Signal
	M_SETEVENT,   // ID3D12Fence::SetEventOnCompletion
	M_CQUEUE,     // ID3D12Device::CreateCommandQueue
	M_CALLOC,     // ID3D12Device::CreateCommandAllocator
	M_CLIST,      // ID3D12Device::CreateCommandList
	M_CFENCE,     // ID3D12Device::CreateFence
	M_CQHEAP,     // ID3D12Device::CreateQueryHeap
	M_COUNT
};
inline const char* MethodName(int m) {
	static const char* const k[] = {"none",       "CreateCommittedResource", "Close",
	                                "ListReset",  "AllocReset",              "Map",
	                                "CreateComputePipelineState",            "CreateRootSignature",
	                                "Signal",     "SetEventOnCompletion",    "CreateCommandQueue",
	                                "CreateCommandAllocator",                "CreateCommandList",
	                                "CreateFence", "CreateQueryHeap"};
	return (m >= 0 && m < M_COUNT) ? k[m] : "?";
}

constexpr uint32_t kMaxSites = 1u << 16;
struct SiteLabel {
	uint16_t window;      // 1-based index of the command-list recording window the site ran in; 0 = none
	uint8_t last_event;   // the D3D12 event (Method) immediately before the site
	uint8_t phase;        // the harness's phase marker (see Phase)
	uint8_t aligned;      // an aligned operator new form
};
enum Phase : uint8_t { PH_NONE = 0, PH_RESET = 1, PH_PREFILL = 2, PH_READ = 3, PH_STEP = 4, PH_READY = 5,
                       PH_CALL = 6 };
inline const char* PhaseName(int p) {
	static const char* const k[] = {"none", "reset", "prefill", "read", "step", "ready", "call"};
	return (p >= 0 && p <= 6) ? k[p] : "?";
}

struct NewState {
	std::atomic<bool> counting{false};
	std::atomic<uint32_t> count{0};
	std::atomic<uint32_t> fail_at{0};  // 1-based; 0 = disarmed
	std::atomic<int> kind{0};
	std::atomic<uint32_t> fired{0};
	std::atomic<uint32_t> aligned_seen{0};
	std::atomic<bool> label{false};  // record SiteLabel per counted call
};
inline NewState g_new;
inline SiteLabel g_labels[kMaxSites];

// Window state, maintained by the Close/ListReset hooks.
inline std::atomic<int> g_list_open{0};
inline std::atomic<int> g_window{0};
inline std::atomic<int> g_last_event{0};
inline std::atomic<int> g_phase{0};

inline void NewArm(uint32_t k, NewFault kind) {
	g_new.kind.store(static_cast<int>(kind));
	g_new.fired.store(0);
	g_new.count.store(0);
	g_new.fail_at.store(k);
	g_new.counting.store(true);
}
inline void NewCountOnly(bool label) {
	g_new.fail_at.store(0);
	g_new.fired.store(0);
	g_new.count.store(0);
	g_new.aligned_seen.store(0);
	g_new.label.store(label);
	g_window.store(0);
	g_list_open.store(0);
	g_new.counting.store(true);
}
inline void NewStop() {
	g_new.counting.store(false);
	g_new.fail_at.store(0);
	g_new.label.store(false);
}

// ---------------------------------------------------------------------------------------------
// Fault source 3: D3D12 vtable hooks.
// ---------------------------------------------------------------------------------------------
enum HookBehavior : int {
	HB_FAIL_NO_CALL = 0,    // return hr without calling D3D12 (Close: the list stays recording)
	HB_CALL_THEN_FAIL = 1,  // call D3D12, then return hr (Close: the list really closed -- R9 (B))
};
struct HookArm {
	std::atomic<int> method{0};  // M_NONE = disarmed
	std::atomic<uint32_t> first{0};  // 1-based ordinal among MATCHING calls since arming
	std::atomic<uint32_t> n{0};      // consecutive matching calls to fail
	std::atomic<long> hr{0};
	std::atomic<int> behavior{0};
	std::atomic<int> heap{0};        // CreateCommittedResource filter: heap type, 0 = any
	std::atomic<uint64_t> width{0};  // CreateCommittedResource filter: buffer width, 0 = any
	std::atomic<uint32_t> seen{0};
	std::atomic<uint32_t> fired{0};
};
inline HookArm g_arm;
inline std::atomic<uint32_t> g_calls[M_COUNT];

struct TraceEv {
	uint8_t method;
	uint8_t heap;
	uint16_t window;     // recording window open when the call was made (0 = none)
	uint64_t width;
	uint32_t new_count;  // operator-new count at the time (for ordering against allocation sites)
};
constexpr uint32_t kMaxTrace = 1u << 16;
inline TraceEv g_trace[kMaxTrace];
inline std::atomic<uint32_t> g_trace_n{0};
inline std::atomic<bool> g_trace_on{false};

inline void HookDisarm() {
	g_arm.method.store(M_NONE);
	g_arm.first.store(0);
	g_arm.n.store(0);
}
inline void HookArmCall(int method, uint32_t first, long hr, int behavior = HB_FAIL_NO_CALL, uint32_t n = 1,
                        int heap = 0, uint64_t width = 0) {
	g_arm.method.store(M_NONE);
	g_arm.first.store(first);
	g_arm.n.store(n);
	g_arm.hr.store(hr);
	g_arm.behavior.store(behavior);
	g_arm.heap.store(heap);
	g_arm.width.store(width);
	g_arm.seen.store(0);
	g_arm.fired.store(0);
	g_arm.method.store(method);
}
inline void TraceStart() {
	g_trace_n.store(0);
	g_window.store(0);
	g_list_open.store(0);
	g_trace_on.store(true);
}
inline std::vector<TraceEv> TraceStop() {
	g_trace_on.store(false);
	const uint32_t n = std::min(g_trace_n.load(), kMaxTrace);
	return std::vector<TraceEv>(g_trace, g_trace + n);
}

// Called by every hook before it decides anything. Returns true when THIS call is to be faulted.
inline bool HookEnter(int m, int heap = 0, uint64_t width = 0) {
	g_calls[m].fetch_add(1);
	if (g_trace_on.load(std::memory_order_relaxed)) {
		const uint32_t i = g_trace_n.fetch_add(1);
		if (i < kMaxTrace) {
			g_trace[i] = TraceEv{static_cast<uint8_t>(m), static_cast<uint8_t>(heap),
			                     static_cast<uint16_t>(g_list_open.load() ? g_window.load() : 0), width,
			                     g_new.count.load(std::memory_order_relaxed)};
		}
	}
	if (g_arm.method.load() != m) return false;
	const int want_heap = g_arm.heap.load();
	const uint64_t want_width = g_arm.width.load();
	if (want_heap != 0 && want_heap != heap) return false;
	if (want_width != 0 && want_width != width) return false;
	const uint32_t seen = g_arm.seen.fetch_add(1) + 1;
	const uint32_t first = g_arm.first.load();
	if (seen >= first && seen < first + g_arm.n.load()) {
		g_arm.fired.fetch_add(1);
		return true;
	}
	return false;
}

bool InstallHooks(std::string* why);  // te425_cells.cpp

// ---------------------------------------------------------------------------------------------
// Artifacts and handles
// ---------------------------------------------------------------------------------------------
inline bool ReadAll(const std::string& p, std::vector<uint8_t>* out) {
	FILE* f = nullptr;
	if (fopen_s(&f, p.c_str(), "rb") || !f) return false;
	_fseeki64(f, 0, SEEK_END);
	const long long n = _ftelli64(f);
	_fseeki64(f, 0, SEEK_SET);
	out->resize(n > 0 ? static_cast<size_t>(n) : 0);
	const bool ok = n > 0 && fread(out->data(), 1, out->size(), f) == out->size();
	fclose(f);
	return ok;
}

struct Artifact {
	std::string path;
	std::vector<uint8_t> bytes;
	superslm::SslmModelView view{};
	bool Load(const std::string& p) {
		path = p;
		std::string err;
		if (!ReadAll(p, &bytes)) {
			std::printf("SETUP cannot read %s\n", p.c_str());
			return false;
		}
		if (superslm::SslmModel::Load(bytes.data(), bytes.size(), view, &err) != superslm::SslmModelStatus::Ok) {
			std::printf("SETUP load %s failed: %s\n", p.c_str(), err.c_str());
			return false;
		}
		return true;
	}
	int32_t Vocab() const { return static_cast<int32_t>(view.config.vocab_size); }
};

// One context with one mapped model and one sequence.
struct Stack {
	SslmGpuContext* ctx = nullptr;
	SslmGpuModelHandle* model = nullptr;
	SslmGpuSequenceHandle* seq = nullptr;
	uint32_t hidden = 0;
	bool Open(Artifact& a, int64_t cap, uint32_t residency_flags = 0) {
		if (sslm_gpu_context_create(GpuContextConfig{}, &ctx) != OK || !ctx) return false;
		GpuResidencyConfig rc{};
		rc.flags = residency_flags;
		if (sslm_gpu_model_map(ctx, &a.view, rc, &model) != OK || !model) return false;
		if (sslm_gpu_seq_create(ctx, model, cap, &seq) != OK || !seq) return false;
		return sslm_gpu_model_hidden_size(model, &hidden) == OK;
	}
	void Close() {
		if (seq) sslm_gpu_seq_release(ctx, seq);
		if (model) sslm_gpu_model_unmap(ctx, model);
		if (ctx) sslm_gpu_context_destroy(ctx);
		seq = nullptr;
		model = nullptr;
		ctx = nullptr;
	}
};

// U3's per-document prompt: n-1 ids from TE-421's pool, then Qwen's <|endoftext|> terminator, as U3
// appends it (TE-419/TE-421's construction). n = 1 is the terminator alone (one sub-chunk), n = 5 is
// two sub-chunks (the kT2169TdrSafeMaxChunkTokens = 4 boundary), n = 40 is the 10-sub-chunk document
// the must-reject population was measured on.
inline std::vector<int32_t> QwenPrompt(int n) {
	static const int32_t pool[] = {785, 279, 315, 323, 264, 13, 3974, 13876, 1879, 374, 11, 220};
	std::vector<int32_t> t;
	for (int i = 0; i + 1 < n; ++i) t.push_back(pool[i % 12]);
	t.push_back(151643);
	return t;
}
// The same shape for a vocabulary that is not Qwen3's (the 1.5B artifacts share Qwen's vocabulary,
// so this is only a guard).
inline std::vector<int32_t> PromptFor(const Artifact& a, int n) {
	if (a.Vocab() > 151643) return QwenPrompt(n);
	std::vector<int32_t> t;
	for (int i = 0; i < n; ++i) t.push_back((a.Vocab() / 7 + 37 * i) % a.Vocab());
	return t;
}

// The frame U3 reads, captured with every out-parameter. Buffers are preallocated so that a loop
// run under operator-new counting allocates nothing on the harness's side.
struct Frame {
	SslmGpuStatus status = DEVICE_LOST;
	std::vector<int8_t> codes;
	int64_t m = 0, e = 0;
	size_t required = 0;
	void Size(uint32_t hidden) { codes.assign(hidden, 0); }
	bool Same(const Frame& o) const {
		return status == OK && o.status == OK && codes == o.codes && m == o.m && e == o.e;
	}
};

// U3's whole per-document loop (TE-421 Sec2.2): reset, prompt prefill, read. Each status is kept;
// the read runs even after a failed prefill (plan Sec3.5 R2: "the read in the same loop returns
// SSLM_PREFILL_HIDDEN_UNAVAILABLE, never SSLM_OK").
struct LoopOut {
	SslmGpuStatus reset = OK, prefill = OK, read = OK;
	// The loop's verdict: the first non-OK status, or OK.
	SslmGpuStatus First() const { return reset != OK ? reset : prefill != OK ? prefill : read; }
	int FirstStage() const { return reset != OK ? 0 : prefill != OK ? 1 : read != OK ? 2 : -1; }
};
inline LoopOut PromptLoop(SslmGpuContext* ctx, SslmGpuSequenceHandle* seq, const std::vector<int32_t>& toks,
                          Frame* f, uint32_t budget = 64) {
	LoopOut o;
	g_phase.store(PH_RESET);
	o.reset = sslm_gpu_seq_reset(ctx, seq);
	g_phase.store(PH_PREFILL);
	if (o.reset == OK) {
		o.prefill = SslmGpuSeqPrefillPromptForG5Bridge(ctx, seq, toks.data(), static_cast<int32_t>(toks.size()),
		                                               budget);
	}
	g_phase.store(PH_READ);
	std::fill(f->codes.begin(), f->codes.end(), static_cast<int8_t>(0x5A));
	f->m = f->e = 0;
	if (o.reset == OK) {
		o.read = sslm_gpu_seq_read_prefill_final_hidden(ctx, seq, f->codes.data(), f->codes.size(), &f->required,
		                                                &f->m, &f->e);
	}
	f->status = o.read;
	g_phase.store(PH_NONE);
	return o;
}

// The oracle every armed loop is held to (plan Sec3.3, Sec3.5 R2/R3):
//   - the first non-OK status is SSLM_GPU_ALLOCATION_FAILED;
//   - when that failure was the prefill's, the read in the same loop refuses with
//     SSLM_PREFILL_HIDDEN_UNAVAILABLE (the prefill snapshot is empty).
inline bool AllocLoopConforms(const LoopOut& o) {
	if (o.First() != ALLOC_FAILED) return false;
	if (o.FirstStage() == 1 && o.read != HIDDEN_UNAVAILABLE) return false;
	return true;
}

inline std::string Hex(uint64_t v) {
	char b[32];
	std::snprintf(b, sizeof b, "0x%08llx", static_cast<unsigned long long>(v));
	return b;
}

}  // namespace te425

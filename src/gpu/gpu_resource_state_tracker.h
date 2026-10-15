// Paged-KV 1.13 (G4/G5 resource states): the recording helpers every D3D12 barrier, copy, root
// SRV/UAV binding and dispatch under src/gpu goes through, and the host-side resource-state tracker
// they feed in a test build.
//
// Why: on the box, GPU-based validation of the G4 red-suite runner reported resource-state errors
// (a dispatch reading an SRV in UNORDERED_ACCESS, a barrier whose StateBefore was not the buffer's
// state, copies whose source or destination was in the wrong state) that the G2/G3 runners do not.
// The cloud's software GPU (vkd3d-proton on lavapipe) has no D3D12 validation, so a defect of this
// class is invisible there unless the host checks it.
//
// Product builds: every helper is an inline forward to the one command-list call it names. Nothing
// is recorded and nothing is checked.
//
// Tracking builds (SUPERSLM_GPU_STATE_TRACKING): the tracker keeps an expected state per buffer, and
// checks each recorded command against it:
//   - a transition barrier's StateBefore must equal the buffer's expected state;
//   - a copy's source must be in a state that includes COPY_SOURCE (GENERIC_READ does), its
//     destination in COPY_DEST;
//   - a dispatch's root SRVs must be in a state that includes NON_PIXEL_SHADER_RESOURCE, its root
//     UAVs in UNORDERED_ACCESS.
// A buffer in COMMON satisfies any requirement once, by D3D12's implicit promotion for buffers; the
// promoted state lasts to the end of the submission and then decays to COMMON again. A state set by
// an explicit barrier (or given at creation) persists across submissions: that is the premise the
// engine's K/V resume latches are built on (the pool latch, plan §4 item 4), and whether the D3D12
// runtime decays such a buffer instead is the box's question (plan §7 3.12), not this model's.
// Each command list keeps its own pending states; they become the buffers' states when that list is
// executed (Device::ExecuteRecordedList), and are discarded when the list is reset without having
// been executed, so a recording a fault abandoned changes nothing.
//
// It also checks buffer lifetimes against the work that can still reach them. GPU-based validation
// resolves a root SRV/UAV and a copy by GPU virtual address, so a buffer released while recorded or
// in-flight work references it would make the layer charge that work's commands to whatever buffer
// the address belongs to next, in that buffer's state. Each buffer carries a sentinel in its private
// data, whose last Release is the buffer's destruction (OnDestroy), and:
//   - a command recorded on a released buffer is released_use;
//   - a list executed after a buffer it touched was released is released_recorded;
//   - a buffer released while an executed list that touched it is not known complete (its Signal
//     not made yet, or its fence value not yet seen complete by the device) is released_in_flight.
// A violation prints one line to stderr, "# GPU STATE VIOLATION [class] ...", with the recording
// site and the buffer's creation site, and is counted. StateTrackingViolations() reads the counts;
// a tracking build prints a summary at exit. SSLM_GPU_STATE_TRACKING_ABORT=1 aborts on the first
// violation instead. After a barrier violation the buffer takes the barrier's StateAfter, so one
// desync is reported once and not at every later use.
#pragma once

#include <d3d12.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <atomic>
#include <mutex>
#include <source_location>
#include <unordered_set>
#include <utility>
#include <vector>
#include <string>
#include <unordered_map>

namespace superslm_gpu {
namespace harness {

// The violation classes, one per message class of the box's GPU-based validation.
struct StateTrackingCounts {
	uint64_t barrier_before = 0;  // a transition's StateBefore is not the buffer's state
	uint64_t copy_source = 0;     // a copy reads a buffer that is not in COPY_SOURCE (or GENERIC_READ)
	uint64_t copy_dest = 0;       // a copy writes a buffer that is not in COPY_DEST
	uint64_t dispatch_srv = 0;    // a dispatch reads a root SRV that is not in NON_PIXEL_SHADER_RESOURCE
	uint64_t dispatch_uav = 0;    // a dispatch binds a root UAV that is not in UNORDERED_ACCESS
	// Lifetime classes: GPU-based validation finds a root SRV/UAV's or a copy's buffer by its GPU virtual
	// address, so a buffer released while GPU work can still reach it makes the layer charge that work's
	// commands to whatever buffer the address belongs to next, in whatever state that one is in.
	uint64_t released_use = 0;        // a command recorded on a buffer that was already released
	uint64_t released_recorded = 0;   // a list executed that references a buffer released after it was recorded
	uint64_t released_in_flight = 0;  // a buffer released while an executed list that references it is not complete
	uint64_t Total() const noexcept {
		return barrier_before + copy_source + copy_dest + dispatch_srv + dispatch_uav + released_use +
		       released_recorded + released_in_flight;
	}
};

#if defined(SUPERSLM_GPU_STATE_TRACKING)
namespace state_tracking {

struct BufferEntry {
	D3D12_RESOURCE_STATES state = D3D12_RESOURCE_STATE_COMMON;
	bool promoted = false;  // reached `state` by implicit promotion from COMMON (decays at the end)
	uint64_t bytes = 0;
	D3D12_HEAP_TYPE heap = D3D12_HEAP_TYPE_DEFAULT;
	const char* created_file = "";
	unsigned created_line = 0;
};

struct Binding {
	ID3D12Resource* resource = nullptr;
	bool uav = false;
};

struct ListEntry {
	std::unordered_map<ID3D12Resource*, BufferEntry> pending;  // every buffer the recording touched, its state
	std::unordered_map<UINT, Binding> bound;                   // root parameter -> SRV/UAV buffer
	std::vector<ID3D12Resource*> released_since_recorded;      // pending buffers released before the execute
	// What the GPU can still reach: the buffers of the executed submission whose Signal has not been made
	// yet, then those of each signalled submission, by fence value, until the device sees that value
	// complete (Device::CompletedFenceValue / WaitFenceEvent).
	std::unordered_set<ID3D12Resource*> executed_unsignaled;
	std::vector<std::pair<uint64_t, std::unordered_set<ID3D12Resource*>>> in_flight;
};

struct Tracker {
	std::mutex mutex;
	std::unordered_map<ID3D12Resource*, BufferEntry> committed;
	std::unordered_map<ID3D12Resource*, BufferEntry> released;  // until the address is a new buffer's
	std::unordered_map<ID3D12GraphicsCommandList*, ListEntry> lists;
	StateTrackingCounts counts;
	uint64_t buffers_created = 0;   // diagnostics: the lifetime classes see a release only through the
	uint64_t buffers_released = 0;  // sentinel, so a run that releases nothing has checked nothing
	bool abort_on_violation = false;
	Tracker() {
		const char* a = std::getenv("SSLM_GPU_STATE_TRACKING_ABORT");
		abort_on_violation = a != nullptr && a[0] == '1';
	}
	void PrintSummary() {
		std::lock_guard<std::mutex> lock(mutex);
		std::fprintf(stderr,
		             "# GPU STATE TRACKING: %llu violation(s): barrier_before %llu, copy_source %llu, copy_dest %llu, "
		             "dispatch_srv %llu, dispatch_uav %llu, released_use %llu, released_recorded %llu, "
		             "released_in_flight %llu (buffers created %llu, released %llu)\n",
		             static_cast<unsigned long long>(counts.Total()),
		             static_cast<unsigned long long>(counts.barrier_before),
		             static_cast<unsigned long long>(counts.copy_source),
		             static_cast<unsigned long long>(counts.copy_dest),
		             static_cast<unsigned long long>(counts.dispatch_srv),
		             static_cast<unsigned long long>(counts.dispatch_uav),
		             static_cast<unsigned long long>(counts.released_use),
		             static_cast<unsigned long long>(counts.released_recorded),
		             static_cast<unsigned long long>(counts.released_in_flight),
		             static_cast<unsigned long long>(buffers_created), static_cast<unsigned long long>(buffers_released));
	}
};

// Never destroyed: a buffer a static owns is released during exit, after any static Tracker would be gone,
// and its release still reaches OnDestroy. The summary prints from an atexit handler.
inline Tracker& Get() {
	static Tracker* t = [] {
		Tracker* made = new Tracker();
		std::atexit([] { Get().PrintSummary(); });
		return made;
	}();
	return *t;
}

inline const char* StateName(D3D12_RESOURCE_STATES s) {
	switch (s) {
		case D3D12_RESOURCE_STATE_COMMON: return "COMMON";
		case D3D12_RESOURCE_STATE_UNORDERED_ACCESS: return "UNORDERED_ACCESS";
		case D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE: return "NON_PIXEL_SHADER_RESOURCE";
		case D3D12_RESOURCE_STATE_COPY_DEST: return "COPY_DEST";
		case D3D12_RESOURCE_STATE_COPY_SOURCE: return "COPY_SOURCE";
		case D3D12_RESOURCE_STATE_GENERIC_READ: return "GENERIC_READ";
		default: return "OTHER";
	}
}

inline const char* FileTail(const char* f) {
	const char* tail = f;
	for (const char* p = f; *p; ++p) {
		if (*p == '/' || *p == '\\') tail = p + 1;
	}
	return tail;
}

// The entry a command in `list` sees for `r`: the list's pending entry, else the committed one
// (copied into the list's pending map, so a promotion or a transition lands there). Null for a
// buffer the tracker never saw created (none under src/gpu: every buffer comes from TryMakeBuffer).
inline void ReportLifetime(Tracker& t, uint64_t StateTrackingCounts::*cls, const char* cls_name, const BufferEntry& e,
                           const char* what, const std::source_location* at);

inline BufferEntry* Lookup(Tracker& t, ID3D12GraphicsCommandList* list, ID3D12Resource* r,
                           const std::source_location& at) {
	ListEntry& le = t.lists[list];
	auto p = le.pending.find(r);
	if (p != le.pending.end()) return &p->second;
	auto c = t.committed.find(r);
	if (c == t.committed.end()) {
		auto d = t.released.find(r);
		if (d != t.released.end()) {
			ReportLifetime(t, &StateTrackingCounts::released_use, "released_use", d->second,
			               "the command is recorded on it after its release", &at);
		}
		return nullptr;
	}
	return &(le.pending[r] = c->second);
}

inline void Report(Tracker& t, uint64_t StateTrackingCounts::*cls, const char* cls_name, const BufferEntry& e,
                   D3D12_RESOURCE_STATES have, D3D12_RESOURCE_STATES want, const std::source_location& at) {
	t.counts.*cls += 1;
	std::fprintf(stderr,
	             "# GPU STATE VIOLATION [%s] at %s:%u: buffer of %llu bytes created at %s:%u is in %s (0x%x), "
	             "the command needs %s (0x%x)\n",
	             cls_name, FileTail(at.file_name()), static_cast<unsigned>(at.line()),
	             static_cast<unsigned long long>(e.bytes), FileTail(e.created_file), e.created_line,
	             StateName(have), static_cast<unsigned>(have), StateName(want), static_cast<unsigned>(want));
	if (t.abort_on_violation) std::abort();
}

inline void ReportLifetime(Tracker& t, uint64_t StateTrackingCounts::*cls, const char* cls_name, const BufferEntry& e,
                           const char* what, const std::source_location* at) {
	t.counts.*cls += 1;
	if (at != nullptr) {
		std::fprintf(stderr, "# GPU STATE VIOLATION [%s] at %s:%u: buffer of %llu bytes created at %s:%u: %s\n", cls_name,
		             FileTail(at->file_name()), static_cast<unsigned>(at->line()),
		             static_cast<unsigned long long>(e.bytes), FileTail(e.created_file), e.created_line, what);
	} else {
		std::fprintf(stderr, "# GPU STATE VIOLATION [%s]: buffer of %llu bytes created at %s:%u: %s\n", cls_name,
		             static_cast<unsigned long long>(e.bytes), FileTail(e.created_file), e.created_line, what);
	}
	if (t.abort_on_violation) std::abort();
}

// Whether `e` satisfies `want` for a read (`want` a read state: every bit present) or a write (the
// exact state). COMMON satisfies anything once, by promotion, and records the promoted state.
inline bool Require(BufferEntry& e, D3D12_RESOURCE_STATES want, bool read) {
	if (e.state == D3D12_RESOURCE_STATE_COMMON) {
		e.state = want;
		e.promoted = true;
		return true;
	}
	if (read) return (e.state & want) == want;
	return e.state == want;
}

// The buffer's release, seen through a COM object stored in its private data: the runtime releases a
// resource's private-data interfaces when it destroys the resource, so the sentinel's last Release is
// the buffer's destruction. Then: the buffer moves to `released`; a recording that touched it and is
// later executed is a released_recorded violation (reported at the execute); and a submission the GPU
// may still be running that touched it is a released_in_flight violation, reported here.
inline void OnDestroy(ID3D12Resource* r) {
	Tracker& t = Get();
	std::lock_guard<std::mutex> lock(t.mutex);
	auto c = t.committed.find(r);
	if (c == t.committed.end()) return;
	const BufferEntry e = c->second;
	t.committed.erase(c);
	t.buffers_released += 1;
	t.released[r] = e;
	for (auto& kv : t.lists) {
		ListEntry& le = kv.second;
		if (le.pending.erase(r) != 0) le.released_since_recorded.push_back(r);
		bool reachable = le.executed_unsignaled.count(r) != 0;
		for (const auto& f : le.in_flight) reachable = reachable || f.second.count(r) != 0;
		if (reachable) {
			ReportLifetime(t, &StateTrackingCounts::released_in_flight, "released_in_flight", e,
			               "released while a submission that references it may still be running on the GPU", nullptr);
		}
	}
}

// {9A4C1E73-5B0D-4F6A-9C2E-7D18B3A6E541}
inline constexpr GUID kDestroySentinelGuid = {0x9a4c1e73, 0x5b0d, 0x4f6a, {0x9c, 0x2e, 0x7d, 0x18, 0xb3, 0xa6, 0xe5, 0x41}};

class DestroySentinel final : public IUnknown {
public:
	explicit DestroySentinel(ID3D12Resource* r) : r_(r) {}
	HRESULT STDMETHODCALLTYPE QueryInterface(REFIID iid, void** out) override {
		if (out == nullptr) return E_POINTER;
		if (IsEqualGUID(iid, __uuidof(IUnknown))) {
			*out = static_cast<IUnknown*>(this);
			AddRef();
			return S_OK;
		}
		*out = nullptr;
		return E_NOINTERFACE;
	}
	ULONG STDMETHODCALLTYPE AddRef() override { return refs_.fetch_add(1) + 1; }
	ULONG STDMETHODCALLTYPE Release() override {
		const ULONG n = refs_.fetch_sub(1) - 1;
		if (n == 0) {
			OnDestroy(r_);
			delete this;
		}
		return n;
	}

private:
	std::atomic<ULONG> refs_{1};
	ID3D12Resource* r_;
};

inline void AttachDestroySentinel(ID3D12Resource* r) {
	DestroySentinel* s = new DestroySentinel(r);
	(void)r->SetPrivateDataInterface(kDestroySentinelGuid, s);  // the resource holds its own reference
	s->Release();
}

inline void OnCreate(ID3D12Resource* r, uint64_t bytes, D3D12_HEAP_TYPE heap, D3D12_RESOURCE_STATES state,
                     const std::source_location& at) {
	Tracker& t = Get();
	std::lock_guard<std::mutex> lock(t.mutex);
	BufferEntry e;
	e.state = state;
	e.bytes = bytes;
	e.heap = heap;
	e.created_file = at.file_name();
	e.created_line = static_cast<unsigned>(at.line());
	t.committed[r] = e;  // a reused address is a new buffer: its old entry is replaced
	t.buffers_created += 1;
	t.released.erase(r);
	for (auto& kv : t.lists) {
		kv.second.pending.erase(r);
		kv.second.executed_unsignaled.erase(r);
		for (auto& f : kv.second.in_flight) f.second.erase(r);
	}
	AttachDestroySentinel(r);
}

inline void OnReset(ID3D12GraphicsCommandList* list) {
	Tracker& t = Get();
	std::lock_guard<std::mutex> lock(t.mutex);
	ListEntry& le = t.lists[list];
	le.pending.clear();
	le.bound.clear();
	le.released_since_recorded.clear();
}

inline void OnExecute(ID3D12GraphicsCommandList* list) {
	Tracker& t = Get();
	std::lock_guard<std::mutex> lock(t.mutex);
	ListEntry& le = t.lists[list];
	for (ID3D12Resource* r : le.released_since_recorded) {
		auto d = t.released.find(r);
		if (d != t.released.end()) {
			ReportLifetime(t, &StateTrackingCounts::released_recorded, "released_recorded", d->second,
			               "the executed list references it, and it was released after the recording touched it",
			               nullptr);
		}
	}
	le.released_since_recorded.clear();
	for (auto& kv : le.pending) {
		le.executed_unsignaled.insert(kv.first);
		BufferEntry e = kv.second;
		if (e.promoted) {
			e.state = D3D12_RESOURCE_STATE_COMMON;  // a promoted state decays when the list completes
			e.promoted = false;
		}
		t.committed[kv.first] = e;
	}
	le.pending.clear();
	le.bound.clear();
}

// The Signal made after the execute: the executed buffers stay reachable until `value` completes.
inline void OnSignal(ID3D12GraphicsCommandList* list, uint64_t value) {
	Tracker& t = Get();
	std::lock_guard<std::mutex> lock(t.mutex);
	ListEntry& le = t.lists[list];
	if (le.executed_unsignaled.empty()) return;
	le.in_flight.emplace_back(value, std::move(le.executed_unsignaled));
	le.executed_unsignaled.clear();
}

// The device saw its fence reach `completed`: every submission signalled at or below it is done.
inline void OnCompleted(ID3D12GraphicsCommandList* list, uint64_t completed) {
	Tracker& t = Get();
	std::lock_guard<std::mutex> lock(t.mutex);
	ListEntry& le = t.lists[list];
	size_t keep = 0;
	for (size_t i = 0; i < le.in_flight.size(); ++i) {
		if (le.in_flight[i].first > completed) {
			if (keep != i) le.in_flight[keep] = std::move(le.in_flight[i]);
			++keep;
		}
	}
	le.in_flight.resize(keep);
}

inline void OnTransition(ID3D12GraphicsCommandList* list, const D3D12_RESOURCE_TRANSITION_BARRIER& b,
                         const std::source_location& at) {
	Tracker& t = Get();
	std::lock_guard<std::mutex> lock(t.mutex);
	BufferEntry* e = Lookup(t, list, b.pResource, at);
	if (e == nullptr) return;
	if (e->state != b.StateBefore) {
		Report(t, &StateTrackingCounts::barrier_before, "barrier_before", *e, e->state, b.StateBefore, at);
	}
	e->state = b.StateAfter;
	e->promoted = false;
}

inline void OnCopy(ID3D12GraphicsCommandList* list, ID3D12Resource* dst, ID3D12Resource* src,
                   const std::source_location& at) {
	Tracker& t = Get();
	std::lock_guard<std::mutex> lock(t.mutex);
	if (BufferEntry* s = Lookup(t, list, src, at)) {
		const D3D12_RESOURCE_STATES have = s->state;
		if (!Require(*s, D3D12_RESOURCE_STATE_COPY_SOURCE, /*read=*/true)) {
			Report(t, &StateTrackingCounts::copy_source, "copy_source", *s, have, D3D12_RESOURCE_STATE_COPY_SOURCE, at);
		}
	}
	if (BufferEntry* d = Lookup(t, list, dst, at)) {
		const D3D12_RESOURCE_STATES have = d->state;
		if (!Require(*d, D3D12_RESOURCE_STATE_COPY_DEST, /*read=*/false)) {
			Report(t, &StateTrackingCounts::copy_dest, "copy_dest", *d, have, D3D12_RESOURCE_STATE_COPY_DEST, at);
		}
	}
}

inline void OnBind(ID3D12GraphicsCommandList* list, UINT index, ID3D12Resource* r, bool uav) {
	Tracker& t = Get();
	std::lock_guard<std::mutex> lock(t.mutex);
	t.lists[list].bound[index] = Binding{r, uav};
}

inline void OnDispatch(ID3D12GraphicsCommandList* list, const std::source_location& at) {
	Tracker& t = Get();
	std::lock_guard<std::mutex> lock(t.mutex);
	// Copied: Lookup may insert into the same list entry's pending map, never into `bound`.
	const std::unordered_map<UINT, Binding> bound = t.lists[list].bound;
	for (const auto& kv : bound) {
		BufferEntry* e = Lookup(t, list, kv.second.resource, at);
		if (e == nullptr) continue;
		const D3D12_RESOURCE_STATES have = e->state;
		if (kv.second.uav) {
			if (!Require(*e, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, /*read=*/false)) {
				Report(t, &StateTrackingCounts::dispatch_uav, "dispatch_uav", *e, have,
				       D3D12_RESOURCE_STATE_UNORDERED_ACCESS, at);
			}
		} else if (!Require(*e, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, /*read=*/true)) {
			Report(t, &StateTrackingCounts::dispatch_srv, "dispatch_srv", *e, have,
			       D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, at);
		}
	}
}

}  // namespace state_tracking

inline StateTrackingCounts StateTrackingViolations() {
	state_tracking::Tracker& t = state_tracking::Get();
	std::lock_guard<std::mutex> lock(t.mutex);
	return t.counts;
}
// A commissioning cell's must-reject constructions raise violations on purpose: it reads the counts
// before, checks its deltas, and puts the counts back, so a run's summary counts only unplanned ones.
inline void RestoreStateTrackingViolations(const StateTrackingCounts& c) {
	state_tracking::Tracker& t = state_tracking::Get();
	std::lock_guard<std::mutex> lock(t.mutex);
	t.counts = c;
}
inline constexpr bool kStateTrackingBuild = true;
#else
inline StateTrackingCounts StateTrackingViolations() { return StateTrackingCounts{}; }
inline constexpr bool kStateTrackingBuild = false;
#endif

// A debug name on a long-lived or per-submission buffer, so the box's debug layer and GPU-based
// validation name the buffer in each message instead of "Unnamed ID3D12Resource Object". No effect on
// what any submission does; a failure to set it is ignored.
inline void NameBuffer(ID3D12Resource* r, const wchar_t* name) {
	if (r != nullptr) (void)r->SetName(name);
}

// The default debug name Device::TryMakeBuffer gives every buffer while the debug layer is on:
// "SuperSLM buffer (<file>:<line>, <heap>)", the creating call's site (the caller's, through Upload,
// MakeBuffer, MakeInitializedUav and the resident-upload helpers, which forward their `at`) and its heap,
// which tells a resident buffer from its UPLOAD staging twin made at the same site. A NameBuffer call
// after the create replaces it with the buffer's role. So no GPU-based validation message names an
// "Unnamed" buffer, and every per-buffer count names the line that made the buffer.
inline void NameBufferBySite(ID3D12Resource* r, const std::source_location& at, D3D12_HEAP_TYPE heap) {
	if (r == nullptr) return;
	const char* file = at.file_name();
	for (const char* p = file; *p; ++p) {
		if (*p == '/' || *p == '\\') file = p + 1;
	}
	const char* heap_name = heap == D3D12_HEAP_TYPE_DEFAULT    ? "DEFAULT"
	                        : heap == D3D12_HEAP_TYPE_UPLOAD   ? "UPLOAD"
	                        : heap == D3D12_HEAP_TYPE_READBACK ? "READBACK"
	                                                           : "other heap";
	std::string ascii = "SuperSLM buffer (";
	ascii += file;
	ascii += ':';
	ascii += std::to_string(at.line());
	ascii += ", ";
	ascii += heap_name;
	ascii += ')';
	std::wstring name;
	for (const char c : ascii) name.push_back(static_cast<wchar_t>(static_cast<unsigned char>(c)));
	(void)r->SetName(name.c_str());
}

// ---- the recording helpers ----------------------------------------------------------------------

// One transition barrier on `r`, StateBefore -> StateAfter.
inline void RecordTransition(ID3D12GraphicsCommandList* list, ID3D12Resource* r, D3D12_RESOURCE_STATES before,
                             D3D12_RESOURCE_STATES after,
                             const std::source_location at = std::source_location::current()) {
	D3D12_RESOURCE_BARRIER b{};
	b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
	b.Transition.pResource = r;
	b.Transition.StateBefore = before;
	b.Transition.StateAfter = after;
	b.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
#if defined(SUPERSLM_GPU_STATE_TRACKING)
	state_tracking::OnTransition(list, b.Transition, at);
#else
	(void)at;
#endif
	list->ResourceBarrier(1, &b);
}

// The K/V resume latch (gpu_1p0.h; a dedicated handle's own, a pooled handle's its pool's) names one of
// the two states a K/V buffer rests in between submissions: set, the last executed submission's readback
// left it in COPY_SOURCE; clear, it is in UNORDERED_ACCESS. Every barrier that leaves a resting K/V
// buffer goes through RecordKvTransitionFromLatch, so the latch-to-state rule is written once. It
// records nothing when the buffer is already in `after`. The caller still owns the latch's writes: the
// readback sets it, the pool upload clears it once its wait returned, and a submission that did not
// execute restores the value it found.
inline D3D12_RESOURCE_STATES KvLatchState(bool latch_set) {
	return latch_set ? D3D12_RESOURCE_STATE_COPY_SOURCE : D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
}
inline void RecordKvTransitionFromLatch(ID3D12GraphicsCommandList* list, ID3D12Resource* kv, bool latch_set,
                                        D3D12_RESOURCE_STATES after,
                                        const std::source_location at = std::source_location::current()) {
	const D3D12_RESOURCE_STATES before = KvLatchState(latch_set);
	if (before != after) RecordTransition(list, kv, before, after, at);
}

// `n` barriers as recorded (transitions are tracked; UAV and aliasing barriers change no state).
inline void RecordBarriers(ID3D12GraphicsCommandList* list, UINT n, const D3D12_RESOURCE_BARRIER* barriers,
                           const std::source_location at = std::source_location::current()) {
#if defined(SUPERSLM_GPU_STATE_TRACKING)
	for (UINT i = 0; i < n; ++i) {
		if (barriers[i].Type == D3D12_RESOURCE_BARRIER_TYPE_TRANSITION) {
			state_tracking::OnTransition(list, barriers[i].Transition, at);
		}
	}
#else
	(void)at;
#endif
	list->ResourceBarrier(n, barriers);
}

inline void RecordCopyBufferRegion(ID3D12GraphicsCommandList* list, ID3D12Resource* dst, UINT64 dst_offset,
                                   ID3D12Resource* src, UINT64 src_offset, UINT64 bytes,
                                   const std::source_location at = std::source_location::current()) {
#if defined(SUPERSLM_GPU_STATE_TRACKING)
	state_tracking::OnCopy(list, dst, src, at);
#else
	(void)at;
#endif
	list->CopyBufferRegion(dst, dst_offset, src, src_offset, bytes);
}

inline void RecordCopyResource(ID3D12GraphicsCommandList* list, ID3D12Resource* dst, ID3D12Resource* src,
                               const std::source_location at = std::source_location::current()) {
#if defined(SUPERSLM_GPU_STATE_TRACKING)
	state_tracking::OnCopy(list, dst, src, at);
#else
	(void)at;
#endif
	list->CopyResource(dst, src);
}

// Root SRV / UAV bindings by buffer (the GPU address is the buffer's start, as every site binds).
inline void BindRootSrv(ID3D12GraphicsCommandList* list, UINT index, ID3D12Resource* r) {
#if defined(SUPERSLM_GPU_STATE_TRACKING)
	state_tracking::OnBind(list, index, r, /*uav=*/false);
#endif
	list->SetComputeRootShaderResourceView(index, r->GetGPUVirtualAddress());
}
inline void BindRootUav(ID3D12GraphicsCommandList* list, UINT index, ID3D12Resource* r) {
#if defined(SUPERSLM_GPU_STATE_TRACKING)
	state_tracking::OnBind(list, index, r, /*uav=*/true);
#endif
	list->SetComputeRootUnorderedAccessView(index, r->GetGPUVirtualAddress());
}

inline void RecordDispatch(ID3D12GraphicsCommandList* list, UINT x, UINT y, UINT z,
                           const std::source_location at = std::source_location::current()) {
#if defined(SUPERSLM_GPU_STATE_TRACKING)
	state_tracking::OnDispatch(list, at);
#else
	(void)at;
#endif
	list->Dispatch(x, y, z);
}

}  // namespace harness
}  // namespace superslm_gpu

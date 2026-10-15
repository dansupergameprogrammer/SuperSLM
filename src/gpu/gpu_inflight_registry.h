// GPU paged-KV plan (rev 16), step G3: the boundary token check's registry, the C ABI call scope,
// the boundary guard check's flag and the task trampoline. D3D12-free: this header and its one
// translation unit (gpu_inflight_registry.cpp) build on every platform, so the registry cell runs
// under ThreadSanitizer on Linux (plan §4.4, *The two boundary checks*, *The registry cell*).
//
// The pieces, in plan terms:
//   - an in-flight record (InFlightRecord) is what Submit() makes (submission_ledger.h). It is
//     registered in a list kept by the making thread, under that thread's current outermost call;
//   - every move of the record into a token (GpuSubmission or GpuInFlightToken), and the counting
//     accessor's lowering of its handle out of Submitted, re-registers it (Reregister) in the
//     moving thread's list under that thread's current outermost call. At depth 0 (no open scope,
//     or the empty binding a trampoline installs) a re-registration keeps the record's existing
//     registration and updates only its holder, under that list's lock (rev 14);
//   - GpuApiCallScope opens the outermost call (the scope at depth 0). When it closes, after every
//     local of the call has been destroyed, the walk visits every live record registered in this
//     thread's list under this call: a record whose holder is a handle's in_flight token with that
//     handle Submitted at that moment passes; any other is drained (non-throwing), freed, its
//     token emptied, and reported;
//   - one mutex per list (rev 13): every registration, unlink, walk and walk-read runs under the
//     lock of the list it touches; never two list locks at once; the walk collects under the lock,
//     releases it, then drains; a record is unlinked before it is freed; a list is held by a shared
//     pointer owned by its thread and by every record registered in it; a list-lock section calls
//     nothing outside the registry; a list reference is dropped only after its lock is released;
//   - a thread's registry binding is the list it registers in, the outermost call it registers
//     under, and its call depth (rev 14.2). GpuTaskTrampoline captures the calling thread's binding
//     when a SuperSLM call hands work to a host parallel-for `run`, and each invocation installs
//     all three for the task and restores what the thread had on every exit. With no scope open the
//     captured binding is empty, and the trampoline installs the empty binding (no call, depth 0,
//     the thread's own list), not nothing (rev 14.3);
//   - the boundary guard check (D4): the outermost scope clears a thread-local "guard passed" flag,
//     EntrySubmitGuard sets it (NoteEntrySubmitGuardRan), and BeginRecording() refuses a reset while
//     the depth is above 0 and the flag is clear (BoundaryGuardCheckRefuses).
//
// Lock order: the list mutexes are leaves; no thread holds two at once, and no list-lock section
// calls anything outside this registry (no fence wait, no free, no other lock).
#ifndef SUPERSLM_GPU_INFLIGHT_REGISTRY_H
#define SUPERSLM_GPU_INFLIGHT_REGISTRY_H

#include <cstddef>
#include <cstdint>
#include <memory>

namespace superslm_gpu {
namespace harness {
class SubmissionDevice;
}  // namespace harness

namespace inflight {

// A handle's identity for the walk's on-handle test. The handle's in_flight token is constructed
// with it, and it is never transferred by a move. SubmittedNow() is read by the walk under a list
// lock, possibly on another thread than the one that last wrote the handle's state, so it must be
// a plain (atomic) read that calls nothing outside the handle.
class InFlightHandleBinding {
public:
	virtual bool SubmittedNow() const noexcept = 0;

protected:
	InFlightHandleBinding() = default;
	~InFlightHandleBinding() = default;
	InFlightHandleBinding(const InFlightHandleBinding&) = default;
	InFlightHandleBinding& operator=(const InFlightHandleBinding&) = default;
};

// What holds a record: the outermost token (a GpuInFlightToken, or a GpuSubmission while no
// GpuInFlightToken carries it). The walk asks the holder for its handle binding (fixed at the
// holder's construction, never moved) and, for a record that fails the on-handle test, calls
// DrainFromWalk(), which drains the record through the non-throwing drain, frees it and empties
// the holder.
class InFlightHolder {
public:
	const InFlightHandleBinding* handle_binding() const noexcept { return binding_; }
	virtual void DrainFromWalk() noexcept = 0;

protected:
	explicit InFlightHolder(const InFlightHandleBinding* binding = nullptr) noexcept : binding_(binding) {}
	~InFlightHolder() = default;
	InFlightHolder(const InFlightHolder&) = delete;
	InFlightHolder& operator=(const InFlightHolder&) = delete;

private:
	const InFlightHandleBinding* const binding_;
};

namespace detail {
struct RegList;
}

// The record one Submit() makes: its device and fence value, and its registry node. Constructible
// only by harness::SubmissionDevice (Submit()). The node's fields are written only under the lock
// of the list the record is in (or, for list_, by the one thread that holds the record's token).
class InFlightRecord {
public:
	uint64_t fence_value() const noexcept { return value_; }
	harness::SubmissionDevice* device() const noexcept { return device_; }
	~InFlightRecord() = default;
	InFlightRecord(const InFlightRecord&) = delete;
	InFlightRecord& operator=(const InFlightRecord&) = delete;

private:
	friend class harness::SubmissionDevice;
	friend struct RegistryAccess;
	explicit InFlightRecord(harness::SubmissionDevice* device) noexcept : device_(device) {}

	harness::SubmissionDevice* const device_;
	uint64_t value_ = 0;
	std::shared_ptr<detail::RegList> list_;
	uint64_t call_ = 0;  // 0: registered under no call
	InFlightHolder* holder_ = nullptr;
	InFlightRecord* prev_ = nullptr;
	InFlightRecord* next_ = nullptr;
};

// ---- registration (called by the token types; not by verbs) --------------------------------

// Makes sure this thread's own list exists. Throws std::bad_alloc; Submit() calls it before
// ExecuteCommandLists, so an allocation failure submits nothing.
void ReserveRegistration();
// A record Submit() made: registered in this thread's current list, under its current outermost
// call (under no call at depth 0).
void RegisterNew(InFlightRecord* rec, InFlightHolder* holder) noexcept;
// A move into a token, or the accessor's lowering of the holder's handle: re-registered under the
// current binding; at depth 0 the registration is kept and only the holder updated.
void Reregister(InFlightRecord* rec, InFlightHolder* holder) noexcept;
// A record about to be freed (finished, waited or drained): unlinked under its list's lock.
void Unlink(InFlightRecord* rec) noexcept;

// ---- the scope and the boundary guard check ---------------------------------------------------

// Opened by every C ABI entry point before any other statement (InvokeGpuApiBoundary, and the
// entry points that do not go through it). Nested scopes only count depth; the outermost one runs
// the walk when it closes. `entry_point` names the entry for the report line.
class GpuApiCallScope {
public:
	explicit GpuApiCallScope(const char* entry_point) noexcept;
	~GpuApiCallScope() noexcept;
	GpuApiCallScope(const GpuApiCallScope&) = delete;
	GpuApiCallScope& operator=(const GpuApiCallScope&) = delete;
};

// EntrySubmitGuard calls this when it runs: some guard ran in the current outermost call.
void NoteEntrySubmitGuardRan() noexcept;
// True when BeginRecording() must refuse the reset: the thread is inside a C ABI call (depth > 0)
// and no EntrySubmitGuard has run in that outermost call. An internal-layer caller outside every
// C ABI call (depth 0) is exempt.
bool BoundaryGuardCheckRefuses() noexcept;

// ---- the task trampoline (decision 81) ----------------------------------------------------------

// A thread's registry binding: its list, its outermost call and its depth. Opaque.
class RegistryBinding {
public:
	RegistryBinding() noexcept = default;

private:
	friend struct RegistryAccess;
	std::shared_ptr<detail::RegList> list_;
	uint64_t call_ = 0;
	int depth_ = 0;
};

// The calling thread's binding; empty (no list, no call, depth 0) when no scope is open.
RegistryBinding CaptureRegistryBinding() noexcept;

// Installs `b` on the current thread (the empty binding when `b` is empty) and restores what the
// thread had when it is destroyed, on every exit.
class GpuTaskBindingScope {
public:
	explicit GpuTaskBindingScope(const RegistryBinding& b) noexcept;
	~GpuTaskBindingScope() noexcept;
	GpuTaskBindingScope(const GpuTaskBindingScope&) = delete;
	GpuTaskBindingScope& operator=(const GpuTaskBindingScope&) = delete;

private:
	RegistryBinding saved_;
};

// Every call of a host parallel-for hook's `run` passes this in place of the task:
//     GpuTaskTrampoline tramp(task, task_ctx);   // captures the calling thread's binding
//     hook.run(hook.host_ctx, count, &GpuTaskTrampoline::Invoke, &tramp);
// `run` returns only after every invocation has returned (parallel_for.h), so `tramp`, which lives
// in the calling SuperSLM call's frame, outlives every invocation. Invoke's signature is
// sslm_task_fn's.
class GpuTaskTrampoline {
public:
	using TaskFn = void (*)(void* task_ctx, int32_t task_index);
	GpuTaskTrampoline(TaskFn task, void* task_ctx) noexcept
	    : binding_(CaptureRegistryBinding()), task_(task), task_ctx_(task_ctx) {}
	GpuTaskTrampoline(const GpuTaskTrampoline&) = delete;
	GpuTaskTrampoline& operator=(const GpuTaskTrampoline&) = delete;
	static void Invoke(void* self, int32_t task_index) noexcept {
		const GpuTaskTrampoline* t = static_cast<const GpuTaskTrampoline*>(self);
		GpuTaskBindingScope bound(t->binding_);
		t->task_(t->task_ctx_, task_index);
	}

private:
	const RegistryBinding binding_;
	const TaskFn task_;
	void* const task_ctx_;
};

// ---- readings ---------------------------------------------------------------------------------

// Process-wide: records the walk drained and reported (gpu_port.h's BoundaryCheckReportsForTest()),
// and records it walked and passed on a Submitted handle (BoundaryCheckWalkedForTest()).
uint64_t BoundaryCheckReportCount() noexcept;
uint64_t BoundaryCheckWalkedCount() noexcept;

// Test readings: records alive (made by Submit() and not yet freed), the current thread's own
// list's live entries, and the current thread's depth.
int64_t LiveRecordsForTest() noexcept;
std::size_t OwnListEntriesForTest() noexcept;
int DepthForTest() noexcept;

}  // namespace inflight
}  // namespace superslm_gpu

#endif  // SUPERSLM_GPU_INFLIGHT_REGISTRY_H

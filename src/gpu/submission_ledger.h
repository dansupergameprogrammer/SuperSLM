// GPU paged-KV plan (rev 16), step G3: the submission ledger (§4.4, *The submission ledger*), the
// reset-site check PoolSubmitAssert, the device's submission surface (BeginRecording(), Submit(),
// Wait(v), Poll(v), Drain()) over abstract fence and queue primitives, and GpuSubmission, the
// draining token Submit() returns. D3D12-free: harness::Device (d3d12_harness.h) derives from
// SubmissionDevice and implements the primitives on its private queue, allocator and fence, so the
// ledger's rules are written once, here, and run under ThreadSanitizer on Linux (cell 3.6h).
//
// The ledger's four host-side values, each a std::atomic:
//   submitted_  the fence value of the last submission whose Signal succeeded;
//   observed_   the highest fence value the host has seen complete; it only rises (a
//               compare-and-swap maximum, so two observers that land out of order never lower it);
//   unsignaled_ set by Submit() after ExecuteCommandLists, cleared when a Signal for that work
//               succeeds; still set when Submit() returns or throws means stranded;
//   lost_       set when a wait on a signalled value cannot be armed or confirmed; never cleared.
//
// Submitters are serialised by P1's precondition (one thread at a time inside a reset, a
// submission or a blocking wait on a device). Observers are not: Poll(v) may run beside a Submit()
// on another thread, which the atomics and the monotonic maximum make safe.
//
// The test build (SUPERSLM_LEDGER_YIELD_HOOK, defined on the whole test target so every
// translation unit of the binary sees one definition of this class) carries a hook between the
// load of observed_ and its compare-and-swap, which 3.6h uses to force an interleaving inside one
// update.
#ifndef SUPERSLM_GPU_SUBMISSION_LEDGER_H
#define SUPERSLM_GPU_SUBMISSION_LEDGER_H

#include <atomic>
#include <cstdint>
#include <stdexcept>

#include "gpu_inflight_registry.h"

namespace superslm_gpu {
namespace harness {

// A submission the engine cannot confirm recovered: a command list that could not be confirmed
// Closed, or submitted work whose completion could not be confirmed (a failed fence Signal with no
// successful retry, or a failed SetEventOnCompletion). Every catch site reads it as rule 0 of the
// fault classification (d3d12_harness.h), whatever else is true. (Moved here from d3d12_harness.h
// at G3, unchanged, so the ledger can throw it without D3D12.)
class GpuSubmissionStrandedError : public std::runtime_error {
public:
	explicit GpuSubmissionStrandedError(const char* what) : std::runtime_error(what) {}
};

// The reset-site check's caller-contract failure: a reset while the host has not observed the
// device's last submission, or (the boundary guard check) a reset inside a C ABI call that passed
// no EntrySubmitGuard. A std::logic_error, as the tree's other caller-contract errors are, so the
// fault classifier never reads it as an allocation failure.
class PoolSubmitContractError : public std::logic_error {
public:
	explicit PoolSubmitContractError(const char* what) : std::logic_error(what) {}
};

#if defined(SUPERSLM_LEDGER_YIELD_HOOK)
// Test build only: called with the value being landed, between the load of observed_ and the
// compare-and-swap that raises it.
using LedgerYieldHookFn = void (*)(uint64_t landing_value);
inline std::atomic<LedgerYieldHookFn>& LedgerYieldHookForTest() noexcept {
	static std::atomic<LedgerYieldHookFn> hook{nullptr};
	return hook;
}
#endif

class SubmissionDevice;

class SubmissionLedger {
public:
	// The guard's ledger term (§4.4, *Scope*): Stranded is checked first, then Unobserved.
	enum class GuardTerm { Observed, Unobserved, Stranded };

	SubmissionLedger() noexcept = default;
	SubmissionLedger(const SubmissionLedger&) = delete;
	SubmissionLedger& operator=(const SubmissionLedger&) = delete;

	uint64_t submitted() const noexcept { return submitted_.load(); }
	uint64_t observed() const noexcept { return observed_.load(); }
	bool unsignaled() const noexcept { return unsignaled_.load(); }
	bool lost() const noexcept { return lost_.load(); }
	bool Stranded() const noexcept { return unsignaled_.load() || lost_.load(); }
	// submitted_ is read first: a poller raising observed_ between the two loads can then only
	// make the answer "observed", never a false "unobserved".
	bool Unobserved() const noexcept {
		const uint64_t s = submitted_.load();
		return observed_.load() < s;
	}
	GuardTerm ReadGuardTerm() const noexcept {
		if (Stranded()) return GuardTerm::Stranded;
		if (Unobserved()) return GuardTerm::Unobserved;
		return GuardTerm::Observed;
	}

private:
	friend class SubmissionDevice;

	// The compare-and-swap maximum. A failed exchange reloads `cur`, so a higher value that landed
	// in between is kept and the loop ends.
	void RaiseObserved(uint64_t v) noexcept {
		uint64_t cur = observed_.load();
#if defined(SUPERSLM_LEDGER_YIELD_HOOK)
		if (LedgerYieldHookFn hook = LedgerYieldHookForTest().load()) hook(v);
#endif
		while (v > cur && !observed_.compare_exchange_weak(cur, v)) {
		}
	}
	void MarkUnsignaled() noexcept { unsignaled_.store(true); }
	void MarkSignaled(uint64_t v) noexcept {
		submitted_.store(v);
		unsignaled_.store(false);
	}
	void MarkLost() noexcept { lost_.store(true); }

	std::atomic<uint64_t> submitted_{0};
	std::atomic<uint64_t> observed_{0};
	std::atomic<bool> unsignaled_{false};
	std::atomic<bool> lost_{false};
};

// The check, inside BeginRecording() before the reset. It reads host-side values only and never
// queries the fence, so whether it fires does not depend on GPU timing: it fires if and only if the
// host has not observed the last submission complete, or the ledger is stranded.
inline void PoolSubmitAssert(const SubmissionLedger& ledger) {
	if (ledger.Stranded()) {
		throw GpuSubmissionStrandedError("D3D12 device's submission ledger is stranded; no reset is made");
	}
	if (ledger.Unobserved()) {
		throw PoolSubmitContractError(
		    "reset while the device's last submission is unobserved (P1's precondition broken); no reset is made");
	}
}

class GpuSubmission;
template <class Record>
class BasicGpuInFlightToken;  // gpu_inflight_token.h

// What a stranded submission keeps alive (review-g23 F1). Once a wait or a Signal cannot be
// confirmed, the GPU may still be executing the submission, so nothing it reads or writes may be
// freed: the owner hands it to the device's stranded graveyard (SubmissionDevice::KeepAliveStranded)
// instead of destroying it. GpuInFlightPayload (gpu_inflight_token.h) derives from it, so a token
// whose drain strands moves its payload there. The graveyard is an intrusive list, so keeping
// something alive never allocates and never throws.
class StrandedKeepAlive {
public:
	virtual ~StrandedKeepAlive() = default;

protected:
	StrandedKeepAlive() noexcept = default;
	StrandedKeepAlive(const StrandedKeepAlive&) = delete;
	StrandedKeepAlive& operator=(const StrandedKeepAlive&) = delete;

private:
	friend class SubmissionDevice;
	StrandedKeepAlive* next_stranded_ = nullptr;
};

// The device's submission surface. harness::Device derives from it and implements the six
// primitives privately on its queue, allocator and fence; nothing else in the library reaches
// them. Every reset goes through BeginRecording() and every submission through Submit().
class SubmissionDevice {
public:
	SubmissionDevice(const SubmissionDevice&) = delete;
	SubmissionDevice& operator=(const SubmissionDevice&) = delete;

	// The boundary guard check, then PoolSubmitAssert, then the allocator's and the list's reset.
	// Throws GpuSubmissionStrandedError (stranded ledger) or PoolSubmitContractError (unobserved
	// ledger, or a C ABI call that passed no EntrySubmitGuard) before the reset.
	void BeginRecording() {
		if (ledger_.Stranded()) {
			throw GpuSubmissionStrandedError("D3D12 device's submission ledger is stranded; no reset is made");
		}
		if (inflight::BoundaryGuardCheckRefuses()) {
			throw PoolSubmitContractError(
			    "boundary guard check: a C ABI call reached a reset without passing EntrySubmitGuard; no reset is made");
		}
		PoolSubmitAssert(ledger_);
		ResetAllocatorAndList();
	}

	// ExecuteCommandLists, unsignaled_ set, Signal at a fresh value; see the definition below.
	[[nodiscard]] GpuSubmission Submit(bool retry = true);

	// Today's fence wait: returns without arming the event when the fence is already at `v`.
	// Raises observed_ to `v` only once the fence has reached it. A wait that cannot be armed or
	// confirmed sets lost_ and throws GpuSubmissionStrandedError.
	void Wait(uint64_t v) {
		if (!WaitNoThrow(v)) throw GpuSubmissionStrandedError("D3D12 fence wait could not be confirmed");
	}
	// Today's completed-value read: true, with observed_ raised to `v`, once the fence has reached
	// `v`. Never waits. Exempt from P1's serialisation (a poller).
	bool Poll(uint64_t v) noexcept {
		if (CompletedFenceValue() < v) return false;
		ledger_.RaiseObserved(v);
		return true;
	}
	// Wait(submitted_); on a stranded ledger it throws GpuSubmissionStrandedError.
	void Drain() {
		if (ledger_.Stranded()) throw GpuSubmissionStrandedError("D3D12 device's submission ledger is stranded");
		Wait(ledger_.submitted());
	}

	const SubmissionLedger& ledger() const noexcept { return ledger_; }

	// The stranded graveyard (review-g23 F1): takes ownership of `keep` and never destroys it while
	// this device lives. The device is lost by contract once a submission strands (rule 0), so what
	// the GPU may still touch stays allocated rather than be freed under it. Thread-safe (a token can
	// drain from another thread's walk); never allocates.
	void KeepAliveStranded(StrandedKeepAlive* keep) noexcept {
		if (keep == nullptr) return;
		StrandedKeepAlive* head = graveyard_.load(std::memory_order_relaxed);
		do {
			keep->next_stranded_ = head;
		} while (!graveyard_.compare_exchange_weak(head, keep, std::memory_order_release, std::memory_order_relaxed));
		stranded_keep_alives_.fetch_add(1, std::memory_order_relaxed);
	}
	// How many things this device has been handed to keep alive for a stranded submission (the
	// graveyard's entries and any deliberate keep-alive a derived device counts with
	// NoteStrandedKeepAlive). Test reading.
	uint64_t stranded_keep_alive_count() const noexcept { return stranded_keep_alives_.load(); }

protected:
	SubmissionDevice() noexcept = default;
	// The graveyard is released only here, at the device's destruction. A device whose queue may
	// still be executing at its destruction (harness::Device) abandons it first, a deliberate leak.
	virtual ~SubmissionDevice() { ReleaseStrandedKeepAlives(); }
	// Destroys the graveyard's entries now: only once nothing the stranded work uses can still run.
	void ReleaseStrandedKeepAlives() noexcept {
		StrandedKeepAlive* p = graveyard_.exchange(nullptr);
		while (p != nullptr) {
			StrandedKeepAlive* next = p->next_stranded_;
			delete p;
			p = next;
		}
	}
	// Leaves the graveyard's entries allocated for ever (the device is lost by contract).
	void AbandonStrandedKeepAlives() noexcept { graveyard_.store(nullptr); }
	// Counts a keep-alive a derived device makes without the graveyard (a deliberate reference leak).
	void NoteStrandedKeepAlive() noexcept { stranded_keep_alives_.fetch_add(1, std::memory_order_relaxed); }

	// The primitives (harness::Device: alloc->Reset() then list->Reset(alloc, nullptr);
	// [the queue gate's wait, fault-injection builds]; ExecuteCommandLists; queue->Signal(fence, v);
	// fence->GetCompletedValue(); SetEventOnCompletion and WaitForSingleObject).
	virtual void ResetAllocatorAndList() = 0;
	virtual void ExecuteRecordedList() = 0;
	// Returns false on a failed Signal, with the failure's code (an HRESULT on the device) in
	// `*failure`; ThrowSignalFailure(code) throws that failure as the classifier expects it
	// (GpuAllocationError for E_OUTOFMEMORY, std::runtime_error otherwise, or the T2169 seam's own).
	virtual bool SignalFence(uint64_t value, int32_t* failure) noexcept = 0;
	[[noreturn]] virtual void ThrowSignalFailure(int32_t failure) = 0;
	virtual uint64_t CompletedFenceValue() noexcept = 0;
	// Arms the fence event at `value` and waits for it; false when the wait cannot be armed or
	// confirmed (SetEventOnCompletion failed). Called only while the fence is below `value`.
	virtual bool WaitFenceEvent(uint64_t value) noexcept = 0;

	// The last fence value a Signal was attempted at (today's fence_val). Only submitters touch
	// it, under P1's precondition.
	uint64_t fence_value_counter() const noexcept { return fence_val_; }

private:
	friend class GpuSubmission;

	// The non-throwing drain every token's destructor uses: false when the wait strands, with
	// lost_ set.
	bool WaitNoThrow(uint64_t v) noexcept {
		if (CompletedFenceValue() >= v) {
			ledger_.RaiseObserved(v);
			return true;
		}
		if (!WaitFenceEvent(v)) {
			ledger_.MarkLost();
			return false;
		}
		ledger_.RaiseObserved(v);
		return true;
	}

	SubmissionLedger ledger_;
	uint64_t fence_val_ = 0;
	std::atomic<StrandedKeepAlive*> graveyard_{nullptr};
	std::atomic<uint64_t> stranded_keep_alives_{0};
};

// The draining token Submit() returns: move-only, holding the submission's record (its device and
// fence value). A synchronous site ends it with Wait(), today's throwing wait, which observes the
// value and empties it; destroyed while it still holds its record, it drains through the
// non-throwing drain and frees the record, so an early return, a jump or an exception between
// Submit() and the wait leaves nothing unobserved. The asynchronous submitters move it into the
// layer-loop record a GpuInFlightToken carries (gpu_inflight_token.h).
class [[nodiscard]] GpuSubmission final : public inflight::InFlightHolder {
public:
	GpuSubmission() noexcept = default;
	GpuSubmission(GpuSubmission&& other) noexcept : rec_(other.rec_) {
		other.rec_ = nullptr;
		if (rec_) inflight::Reregister(rec_, this);
	}
	// A move-assignment into a submission that still holds a record drains that record first.
	GpuSubmission& operator=(GpuSubmission&& other) noexcept {
		if (this != &other) {
			if (rec_) DrainNoThrow();
			rec_ = other.rec_;
			other.rec_ = nullptr;
			if (rec_) inflight::Reregister(rec_, this);
		}
		return *this;
	}
	~GpuSubmission() {
		if (rec_) DrainNoThrow();
	}

	bool holds() const noexcept { return rec_ != nullptr; }
	uint64_t fence_value() const noexcept { return rec_ ? rec_->fence_value() : 0; }
	// True once the fence has reached this submission's value (observed_ raised); never waits and
	// never empties the token. An empty submission reads as reached.
	bool ReadyNoWait() const noexcept { return rec_ == nullptr || rec_->device()->Poll(rec_->fence_value()); }

	// The throwing wait: observes the value, frees the record and empties the token; on a wait
	// that cannot be confirmed the ledger is stranded (lost_), the record is freed, and
	// GpuSubmissionStrandedError is thrown. The record holds no GPU resource, so freeing it is safe
	// either way; what the submission's work reads and writes is its owner's to keep alive on a
	// stranded wait (SubmissionDevice::KeepAliveStranded).
	void Wait() {
		if (rec_ && !DrainNoThrow()) throw GpuSubmissionStrandedError("D3D12 fence wait could not be confirmed");
	}
	// The device this submission was made on; null when empty.
	SubmissionDevice* device() const noexcept { return rec_ ? rec_->device() : nullptr; }
	// The non-throwing drain: false when the wait strands (lost_ set). Frees the record and empties
	// the token either way.
	bool DrainNoThrow() noexcept {
		inflight::InFlightRecord* rec = rec_;
		rec_ = nullptr;
		if (rec == nullptr) return true;
		const bool ok = rec->device()->WaitNoThrow(rec->fence_value());
		inflight::Unlink(rec);
		delete rec;
		return ok;
	}

private:
	friend class SubmissionDevice;
	template <class Record>
	friend class BasicGpuInFlightToken;

	explicit GpuSubmission(inflight::InFlightRecord* rec) noexcept : rec_(rec) { inflight::RegisterNew(rec_, this); }
	void DrainFromWalk() noexcept override { DrainNoThrow(); }

	inflight::InFlightRecord* rec_ = nullptr;
};

// Submit(retry): ExecuteCommandLists, then unsignaled_ set, then Signal at a fresh value. On
// success submitted_ is raised and unsignaled_ cleared, and the record is handed out in a
// GpuSubmission. On a failed Signal with `retry` (the default) Submit() signals once more at a
// fresh value, never the failed one; if that succeeds it raises submitted_, clears unsignaled_,
// drains the fresh value and rethrows the first failure (a recovered retry leaves nothing in
// flight); if the retry fails, or that drain strands, unsignaled_ (or lost_) stays set and it
// throws GpuSubmissionStrandedError (rule 0). Without `retry` a failed Signal is stranded at once.
// The record and this thread's registry list are allocated before ExecuteCommandLists, so an
// allocation failure submits nothing.
inline GpuSubmission SubmissionDevice::Submit(bool retry) {
	inflight::ReserveRegistration();
	inflight::InFlightRecord* rec = new inflight::InFlightRecord(this);
	try {
		ExecuteRecordedList();
	} catch (...) {
		delete rec;
		throw;
	}
	ledger_.MarkUnsignaled();
	int32_t first_failure = 0;
	const uint64_t first_value = ++fence_val_;
	if (SignalFence(first_value, &first_failure)) {
		ledger_.MarkSignaled(first_value);
		rec->value_ = first_value;
		return GpuSubmission(rec);
	}
	if (!retry) {
		delete rec;
		throw GpuSubmissionStrandedError("D3D12 fence Signal failed after ExecuteCommandLists");
	}
	int32_t retry_failure = 0;
	const uint64_t retry_value = ++fence_val_;
	if (!SignalFence(retry_value, &retry_failure)) {
		delete rec;
		throw GpuSubmissionStrandedError("D3D12 fence Signal and its retry both failed after ExecuteCommandLists");
	}
	ledger_.MarkSignaled(retry_value);
	rec->value_ = retry_value;
	{
		GpuSubmission recovered(rec);
		recovered.Wait();
	}
	ThrowSignalFailure(first_failure);
	// ThrowSignalFailure is [[noreturn]]; an override that returned anyway leaves the first failure
	// unreported, so the call still fails.
	throw GpuSubmissionStrandedError("D3D12 fence Signal failed; its failure could not be rethrown");
}

}  // namespace harness
}  // namespace superslm_gpu

#endif  // SUPERSLM_GPU_SUBMISSION_LEDGER_H

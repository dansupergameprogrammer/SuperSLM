// GPU paged-KV plan (rev 16), step G3: the host-only cells' fakes and model verbs.
//
// FakeDevice implements SubmissionDevice's six primitives on host atomics, so the ledger, the
// tokens and the registry under test are the library's own (src/gpu/submission_ledger.h,
// gpu_inflight_token.h, gpu_inflight_registry.cpp), and only the queue, allocator and fence are
// stand-ins. The model verbs (Decode, Ready, Stash, NonSubmitting) are the shape the plan gives
// the G3 verbs (§4.4): a guarded singleton call that submits and leaves its token on a Submitted
// handle; sslm_gpu_ready's finish, which moves the token off the handle only once the fence has
// reached its value and lowers the handle through the counting accessor; the file-scope stash
// (`fin_pipeline`, a must-reject the walk drains and reports); and a non-submitting call on
// another handle. They model the integration, which is another builder's; what they exercise is
// the registry, the ledger and the tokens.
#ifndef SUPERSLM_TESTS_G3_FAKES_H
#define SUPERSLM_TESTS_G3_FAKES_H

#include "gpu_inflight_registry.h"
#include "gpu_inflight_token.h"
#include "submission_ledger.h"

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace g3 {

using superslm_gpu::GpuInFlightPayload;
using superslm_gpu::harness::BasicGpuInFlightToken;
using superslm_gpu::harness::GpuSubmission;
using superslm_gpu::harness::GpuSubmissionStrandedError;
using superslm_gpu::harness::PoolSubmitContractError;
using superslm_gpu::harness::SubmissionDevice;
using superslm_gpu::harness::SubmissionLedger;
namespace inflight = superslm_gpu::inflight;

// The failure ThrowSignalFailure throws: the classifier's "first failure" stand-in.
struct FakeSignalFailure : std::runtime_error {
	int32_t code;
	explicit FakeSignalFailure(int32_t c) : std::runtime_error("fake Signal failure"), code(c) {}
};

class FakeDevice final : public SubmissionDevice {
public:
	// auto_complete: the fake GPU completes every submission at its Signal (the strike models'
	// device). Otherwise the fence moves only by Complete(v), or, with complete_on_arm, when a wait
	// is armed (work that finishes while the host waits).
	explicit FakeDevice(bool auto_complete = true, bool complete_on_arm = true)
	    : auto_complete_(auto_complete), complete_on_arm_(complete_on_arm) {}

	void Complete(uint64_t v) {
		uint64_t cur = completed_.load();
		while (v > cur && !completed_.compare_exchange_weak(cur, v)) {
		}
	}
	uint64_t completed() const { return completed_.load(); }
	void FailNextSignals(int n, int32_t code) {
		fail_signals_.store(n);
		fail_code_ = code;
	}
	void FailArms(bool on) { fail_arms_.store(on); }

	long resets() const { return resets_.load(); }
	long executes() const { return executes_.load(); }
	long arms() const { return arms_.load(); }
	// The values Signal was attempted at, in order (written only by the submitting thread).
	const std::vector<uint64_t>& signal_attempts() const { return signal_attempts_; }

private:
	void ResetAllocatorAndList() override { resets_.fetch_add(1); }
	void ExecuteRecordedList() override { executes_.fetch_add(1); }
	bool SignalFence(uint64_t value, int32_t* failure) noexcept override {
		signal_attempts_.push_back(value);
		if (fail_signals_.load() > 0) {
			fail_signals_.fetch_sub(1);
			*failure = fail_code_;
			return false;
		}
		if (auto_complete_) Complete(value);
		return true;
	}
	[[noreturn]] void ThrowSignalFailure(int32_t failure) override { throw FakeSignalFailure(failure); }
	uint64_t CompletedFenceValue() noexcept override { return completed_.load(); }
	bool WaitFenceEvent(uint64_t value) noexcept override {
		arms_.fetch_add(1);
		if (fail_arms_.load()) return false;
		if (complete_on_arm_) Complete(value);
		while (completed_.load() < value) std::this_thread::yield();
		return true;
	}

	const bool auto_complete_;
	const bool complete_on_arm_;
	std::atomic<uint64_t> completed_{0};
	std::atomic<int> fail_signals_{0};
	int32_t fail_code_ = 0;
	std::atomic<bool> fail_arms_{false};
	std::atomic<long> resets_{0}, executes_{0}, arms_{0};
	std::vector<uint64_t> signal_attempts_;
};

// The layer-loop record stand-in: constructible only by its token.
struct TestLoopRecord final : GpuInFlightPayload {
	int tag;

private:
	friend class superslm_gpu::harness::BasicGpuInFlightToken<TestLoopRecord>;
	TestLoopRecord(GpuSubmission&& s, int t) : GpuInFlightPayload(std::move(s)), tag(t) {}
};
using TestToken = BasicGpuInFlightToken<TestLoopRecord>;

enum class St { Idle, Submitted };

// A sequence handle stand-in: its state (written only through SetState, the counting accessor's
// stand-in) and its in_flight token, bound to it.
struct FakeHandle final : inflight::InFlightHandleBinding {
	std::atomic<St> state{St::Idle};
	TestToken in_flight{this};
	bool SubmittedNow() const noexcept override { return state.load() == St::Submitted; }
};

// The counting accessor's stand-in: a move out of Submitted re-registers the handle's record.
inline void SetState(FakeHandle* h, St s) {
	const bool lowering = h->state.load() == St::Submitted && s != St::Submitted;
	h->state.store(s);
	if (lowering) h->in_flight.ReregisterOnLowering();
}

enum class Status { Ok, Busy, DeviceLost };

inline const char* StatusName(Status s) {
	return s == Status::Ok ? "SSLM_OK" : s == Status::Busy ? "SSLM_BUSY" : "SSLM_DEVICE_LOST";
}

// A guarded singleton-reaching call (sslm_decode_step_gpu's shape): the guard (own handle
// Submitted, then the ledger term), the reset, the submission, the token onto the handle, the
// handle raised to Submitted.
inline Status Decode(FakeDevice& dev, FakeHandle* a) {
	inflight::GpuApiCallScope scope("decode");
	if (a->state.load() == St::Submitted) return Status::Busy;
	const SubmissionLedger::GuardTerm term = dev.ledger().ReadGuardTerm();
	if (term == SubmissionLedger::GuardTerm::Stranded) return Status::DeviceLost;
	if (term == SubmissionLedger::GuardTerm::Unobserved) return Status::Busy;
	inflight::NoteEntrySubmitGuardRan();
	dev.BeginRecording();
	TestToken t = TestToken::Make(dev.Submit(), 0);
	a->in_flight = std::move(t);
	SetState(a, St::Submitted);
	return Status::Ok;
}

// sslm_gpu_ready's shape: a not-ready poll returns before touching the token; otherwise the token
// moves into the finish's local, the handle is lowered, and the finish observes the value. Returns
// whether the handle was finished (or was not Submitted).
inline bool Ready(FakeHandle* a, bool block) {
	inflight::GpuApiCallScope scope("ready");
	if (a->state.load() != St::Submitted) return true;
	if (!block && !a->in_flight.ReadyNoWait()) return false;
	TestToken local = std::move(a->in_flight);
	SetState(a, St::Idle);
	local.Wait();
	return true;
}

// The file-scope stash: a submitter that keeps its token in a file-scope variable (fin_pipeline).
inline TestToken& StashSlot() {
	static TestToken slot;
	return slot;
}
inline void Stash(FakeDevice& dev) {
	inflight::GpuApiCallScope scope("stash");
	TestToken t = TestToken::Make(dev.Submit(), 1);
	StashSlot() = std::move(t);
}

inline void NonSubmitting(FakeHandle* b, std::atomic<long>* sink) {
	inflight::GpuApiCallScope scope("non_submitting");
	sink->fetch_add(b->state.load() == St::Submitted ? 1 : 0, std::memory_order_relaxed);
}

// A host parallel-for `run` stand-in: invokes every task index on a fresh host thread each, all at
// once, and returns after every invocation has returned (parallel_for.h's contract).
inline void HostRun(int32_t count, void (*task)(void*, int32_t), void* ctx) {
	std::vector<std::thread> ws;
	for (int32_t i = 0; i < count; ++i) ws.emplace_back([=] { task(ctx, i); });
	for (std::thread& w : ws) w.join();
}

}  // namespace g3

#endif  // SUPERSLM_TESTS_G3_FAKES_H

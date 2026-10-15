// GPU paged-KV plan (rev 16), step G3: GpuInFlightToken, the RAII in-flight token (§4.4, *Three
// states*, *The finish made structural*). D3D12-free.
//
// The asynchronous submitters (RunLayerLoopGpuSubmit, SubmitChunkToFullDepthForG5Bridge,
// SubmitOneSubChunkToFullDepthForG5Bridge) hand their work out in one, through a
// GpuInFlightToken* out-parameter; RunLayerLoopGpuFinish takes one by rvalue reference and moves
// from it only once the fence has reached the token's value (a not-ready poll returns before the
// move and leaves the token untouched). The token:
//   - is move-only (copy deleted; a move empties its source and re-registers the record it
//     carries, with the new token as its holder);
//   - drains on destruction: a token destroyed while it still holds work waits on the owning
//     device's fence through the non-throwing drain and frees the record, without a readback; a
//     drain that strands leaves the ledger stranded (lost_), hands the payload to the device's
//     stranded graveyard (never freed while the device lives; review-g23 F1) and returns;
//   - drains on move-assignment: assigning over a token that still holds a record drains and frees
//     that record first;
//   - carries the record only it can construct: the layer-loop record (GpuLayerLoopInFlight, in
//     the one source file that defines it) derives from GpuInFlightPayload, has a private
//     constructor, and names BasicGpuInFlightToken<itself> its friend; the payload holds the
//     GpuSubmission Submit() returned, so no work is in flight outside a token;
//   - is the holder of its record for the boundary token check while it carries it; a handle's
//     in_flight token is constructed with the handle's binding, which a move never transfers, so
//     the walk's on-handle test asks "is this token a handle's in_flight, and is that handle
//     Submitted now".
//
// The token is a template on the record type so this header needs nothing of the record's
// definition: GpuInFlightToken (below) names the forward-declared GpuLayerLoopInFlight, and only
// get()/Make(), used in the record's own source file, need it complete. Destruction goes through
// GpuInFlightPayload's virtual destructor.
#ifndef SUPERSLM_GPU_INFLIGHT_TOKEN_H
#define SUPERSLM_GPU_INFLIGHT_TOKEN_H

#include <cstdint>
#include <type_traits>
#include <utility>

#include "submission_ledger.h"

namespace superslm_gpu {

// The base of every record a token carries: the submission it is the work of. A StrandedKeepAlive,
// so a token whose drain strands hands the payload (and every resource its work reads and writes)
// to the device's graveyard instead of freeing it (review-g23 F1).
class GpuInFlightPayload : public harness::StrandedKeepAlive {
public:
	~GpuInFlightPayload() override = default;
	GpuInFlightPayload(const GpuInFlightPayload&) = delete;
	GpuInFlightPayload& operator=(const GpuInFlightPayload&) = delete;
	const harness::GpuSubmission& submission() const noexcept { return submission_; }

protected:
	explicit GpuInFlightPayload(harness::GpuSubmission&& submission) noexcept : submission_(std::move(submission)) {}

private:
	template <class Record>
	friend class harness::BasicGpuInFlightToken;
	harness::GpuSubmission submission_;
};

namespace harness {

template <class Record>
class BasicGpuInFlightToken final : public inflight::InFlightHolder {
public:
	BasicGpuInFlightToken() noexcept = default;
	// A handle's in_flight member: bound to the handle for the walk's on-handle test.
	explicit BasicGpuInFlightToken(const inflight::InFlightHandleBinding* handle) noexcept : InFlightHolder(handle) {}

	BasicGpuInFlightToken(const BasicGpuInFlightToken&) = delete;
	BasicGpuInFlightToken& operator=(const BasicGpuInFlightToken&) = delete;

	// A move carries the record, never the source's handle binding.
	BasicGpuInFlightToken(BasicGpuInFlightToken&& other) noexcept : payload_(other.payload_) {
		other.payload_ = nullptr;
		Adopt();
	}
	// Drains the record this token still holds, first; then carries `other`'s.
	BasicGpuInFlightToken& operator=(BasicGpuInFlightToken&& other) noexcept {
		if (this != &other) {
			if (payload_) DrainAndFreeNoThrow();
			payload_ = other.payload_;
			other.payload_ = nullptr;
			Adopt();
		}
		return *this;
	}
	~BasicGpuInFlightToken() {
		if (payload_) DrainAndFreeNoThrow();
	}

	// Constructs the record (only a token can) around the submission and carries it. Throws what
	// the record's constructor throws (std::bad_alloc included), in which case `submission` still
	// holds its record and drains when the caller's submission is destroyed.
	template <class... Args>
	static BasicGpuInFlightToken Make(GpuSubmission&& submission, Args&&... args) {
		static_assert(std::is_base_of<GpuInFlightPayload, Record>::value,
		              "a token's record derives from GpuInFlightPayload");
		BasicGpuInFlightToken t;
		t.payload_ = new Record(std::move(submission), std::forward<Args>(args)...);
		t.Adopt();
		return t;
	}

	explicit operator bool() const noexcept { return payload_ != nullptr; }
	bool holds() const noexcept { return payload_ != nullptr; }
	Record* get() const noexcept { return static_cast<Record*>(payload_); }
	Record* operator->() const noexcept { return get(); }
	// The record's fence value; 0 when empty or once its submission was waited.
	uint64_t fence_value() const noexcept { return payload_ ? payload_->submission_.fence_value() : 0; }
	// True once the fence has reached the token's value (observed_ raised). Never waits and never
	// empties the token: the not-ready poll's test.
	bool ReadyNoWait() const noexcept { return payload_ == nullptr || payload_->submission_.ReadyNoWait(); }
	// The throwing wait (blocking finish): observes the value and frees the submission's record,
	// keeping the payload (its readback buffers) for the caller; the token then carries no
	// in-flight work. Throws GpuSubmissionStrandedError when the wait cannot be confirmed; the
	// payload then goes to the device's stranded graveyard, never freed while the device lives, and
	// the token is left empty (review-g23 F1: the GPU may still be using what it holds).
	void Wait() {
		if (payload_ == nullptr || !payload_->submission_.holds()) return;
		SubmissionDevice* const dev = payload_->submission_.device();
		if (!payload_->submission_.DrainNoThrow()) {
			GpuInFlightPayload* const p = payload_;
			payload_ = nullptr;
			dev->KeepAliveStranded(p);
			throw GpuSubmissionStrandedError("D3D12 fence wait could not be confirmed");
		}
	}
	// The wait seam's wait (sslm_gpu_test_only_wait_in_flight): waits on the device's fence at the
	// token's value through the ledger's Wait(v), which observes it, and keeps the record and the
	// token as they are, so the holder stays Submitted with nothing executing. Throws
	// GpuSubmissionStrandedError when the wait cannot be confirmed (the ledger's lost_).
	void ObserveWithoutFinishing() {
		if (payload_ && payload_->submission_.rec_) {
			inflight::InFlightRecord* rec = payload_->submission_.rec_;
			rec->device()->Wait(rec->fence_value());
		}
	}
	// The counting accessor calls this when it moves the token's handle out of Submitted: the
	// record re-registers under the current outermost call (kept at depth 0), so the walk at that
	// call's return sees the handle no longer Submitted and drains and reports it.
	void ReregisterOnLowering() noexcept { Adopt(); }

private:
	void Adopt() noexcept {
		if (payload_ && payload_->submission_.rec_) inflight::Reregister(payload_->submission_.rec_, this);
	}
	// Frees the payload only once its submission is confirmed complete (or was already waited). A
	// drain that strands hands it to the device's graveyard instead (review-g23 F1): the GPU may
	// still read or write what it holds, and the device is lost by contract.
	void DrainAndFreeNoThrow() noexcept {
		GpuInFlightPayload* p = payload_;
		payload_ = nullptr;
		SubmissionDevice* const dev = p->submission_.device();
		if (p->submission_.DrainNoThrow()) {
			delete p;
		} else {
			dev->KeepAliveStranded(p);
		}
	}
	void DrainFromWalk() noexcept override {
		if (payload_) DrainAndFreeNoThrow();
	}

	GpuInFlightPayload* payload_ = nullptr;
};

}  // namespace harness

struct GpuLayerLoopInFlight;
using GpuInFlightToken = harness::BasicGpuInFlightToken<GpuLayerLoopInFlight>;

}  // namespace superslm_gpu

#endif  // SUPERSLM_GPU_INFLIGHT_TOKEN_H

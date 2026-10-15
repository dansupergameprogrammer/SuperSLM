// GPU paged-KV plan (rev 16), step G3, decision 80: the finish tail's "last call" timing snapshot
// (§4.4, *The not-ready poll*, *The finishing tail's shared state*). D3D12-free.
//
// From G3 a call's timing state is per call: the layer-loop record carries its own GpuCallTiming
// and per-dispatch vector, the submit writes record_ms into it and the tail the rest, and at the
// tail's end (and on the submit's and the tail's catch paths, as far as the call got) the call's
// timing is published into this snapshot under one mutex. Every singleton chunk open publishes an
// empty snapshot under the same mutex, and LastCallTiming() / LastCallPerDispatchTimingsMs() copy
// it under that mutex. So a finishing poller beside a chunk open, or beside another finishing
// poller, writes no shared state unlocked (1.8.0's globals, written in place, corrupt the heap
// there: strike round 6).
//
// The mutex is a leaf: the publish, the reset and the copies call nothing outside the snapshot
// while holding it. The data is an immutable block behind a shared pointer, built outside the lock
// and swapped in under it; a copy takes the pointer under the lock and reads the block after
// releasing it, so no allocation, copy or free runs under the lock either.
//
// What stays shared, and is not claimed: the snapshot is the last call to publish, which beside a
// concurrent caller may be another thread's. A diagnostic.
#ifndef SUPERSLM_GPU_CALL_TIMING_SNAPSHOT_H
#define SUPERSLM_GPU_CALL_TIMING_SNAPSHOT_H

#include <memory>
#include <mutex>
#include <utility>
#include <vector>

namespace superslm_gpu {

template <class Timing>
class LastCallTimingSnapshot {
public:
	LastCallTimingSnapshot() = default;
	LastCallTimingSnapshot(const LastCallTimingSnapshot&) = delete;
	LastCallTimingSnapshot& operator=(const LastCallTimingSnapshot&) = delete;

	// The tail's end (or a catch path): the call's own timing becomes the last call's. Allocates
	// the block, and throws what that allocation throws; `per_dispatch_ms` is moved from only once
	// the block is allocated, so a failed publish leaves the caller's vector as it was.
	void Publish(const Timing& timing, std::vector<double>&& per_dispatch_ms) {
		std::shared_ptr<const Block> fresh = std::make_shared<const Block>(timing, std::move(per_dispatch_ms));
		Swap(fresh);
	}
	// A singleton chunk open: an empty snapshot, held as no block (the copies read no block as the
	// empty timing and vector). Allocates nothing.
	void PublishEmpty() {
		std::shared_ptr<const Block> fresh;
		Swap(fresh);
	}
	Timing CopyTiming() const {
		std::shared_ptr<const Block> b = Load();
		return b ? b->timing : Timing{};
	}
	std::vector<double> CopyPerDispatchMs() const {
		std::shared_ptr<const Block> b = Load();
		return b ? b->per_dispatch_ms : std::vector<double>{};
	}

private:
	struct Block {
		Block() = default;
		Block(const Timing& t, std::vector<double>&& d) : timing(t), per_dispatch_ms(std::move(d)) {}
		Timing timing{};
		std::vector<double> per_dispatch_ms;
	};
	// The old block is released after the lock (it leaves with `fresh`).
	void Swap(std::shared_ptr<const Block>& fresh) {
		std::lock_guard<std::mutex> lock(mu_);
		current_.swap(fresh);
	}
	std::shared_ptr<const Block> Load() const {
		std::lock_guard<std::mutex> lock(mu_);
		return current_;
	}

	mutable std::mutex mu_;
	std::shared_ptr<const Block> current_;
};

}  // namespace superslm_gpu

#endif  // SUPERSLM_GPU_CALL_TIMING_SNAPSHOT_H

// GPU paged-KV plan (rev 16), step G3, decision 80: the finish tail's timing snapshot (§4.4, *The
// finishing tail's shared state*), the TAIL cell's D3D12-free part.
//
//   TAIL/poller-first  strike round 6's poller_first.cpp, rebuilt against the repaired publish: a
//                      finishing poller publishing its call's per-dispatch timings (a reallocation
//                      past the warmed capacity) races a singleton chunk open's reset.
//   TAIL/poller-pp     poller_pp.cpp, rebuilt likewise: two finishing pollers publish at once.
// Graded under ThreadSanitizer (clean) and AddressSanitizer (clean); the unrepaired tail, writing
// the shared state in place without the lock, is the mutant and must go red under
// ThreadSanitizer with halt_on_error=1 (run_g3_core.py). Each trial's final snapshot must also be
// one a serial order yields.
#include "g3_harness.h"

#include "gpu_call_timing_snapshot.h"

#include <atomic>
#include <thread>
#include <vector>

namespace g3 {
namespace {

struct TailTiming {  // GpuCallTiming's shape (gpu_port.h)
	double record_ms = 0.0;
	double submit_wait_ms = 0.0;
	double gpu_busy_ms = 0.0;
	double readback_ms = 0.0;
};

// The tail's end: the call's own timing, built per call, published under the snapshot's lock.
void FinishTail(superslm_gpu::LastCallTimingSnapshot<TailTiming>& snap, unsigned dispatches, double tag) {
	std::vector<double> mine(dispatches);
	for (unsigned d = 0; d < dispatches; ++d) mine[d] = d;
	TailTiming t;
	t.record_ms = tag;
	t.submit_wait_ms = tag;
	t.gpu_busy_ms = tag;
	t.readback_ms = tag;
	snap.Publish(t, std::move(mine));
}

void CellPollerFirst() {
	const long trials = G3Param("tail", 200);
	long bad = 0;
	for (long i = 0; i < trials; ++i) {
		superslm_gpu::LastCallTimingSnapshot<TailTiming> snap;
		FinishTail(snap, 16, 1.0);  // warmed by an earlier small call
		std::atomic<int> go{0};
		std::thread open([&] {
			go.fetch_add(1);
			while (go.load() < 2) {
			}
			snap.PublishEmpty();  // a singleton chunk open's reset
		});
		std::thread poller([&] {
			go.fetch_add(1);
			while (go.load() < 2) {
			}
			FinishTail(snap, 700, 7.0);
		});
		open.join();
		poller.join();
		const std::vector<double> d = snap.CopyPerDispatchMs();
		const TailTiming t = snap.CopyTiming();
		const bool empty = d.empty() && t.record_ms == 0.0;
		const bool full = d.size() == 700 && t.record_ms == 7.0 && d[699] == 699.0;
		if (!empty && !full) ++bad;  // neither serial order yields this
	}
	G3_CHECK_EQ(bad, 0);
}
G3_CELL("TAIL/poller-first", CellPollerFirst);

void CellPollerPP() {
	const long trials = G3Param("tail", 200);
	long bad = 0;
	for (long i = 0; i < trials; ++i) {
		superslm_gpu::LastCallTimingSnapshot<TailTiming> snap;
		FinishTail(snap, 16, 1.0);
		std::atomic<int> go{0};
		std::thread a([&] {
			go.fetch_add(1);
			while (go.load() < 2) {
			}
			FinishTail(snap, 300, 3.0);
		});
		std::thread b([&] {
			go.fetch_add(1);
			while (go.load() < 2) {
			}
			FinishTail(snap, 700, 7.0);
		});
		a.join();
		b.join();
		const std::vector<double> d = snap.CopyPerDispatchMs();
		const TailTiming t = snap.CopyTiming();
		const bool three = d.size() == 300 && t.record_ms == 3.0;
		const bool seven = d.size() == 700 && t.record_ms == 7.0;
		if (!three && !seven) ++bad;
	}
	G3_CHECK_EQ(bad, 0);
}
G3_CELL("TAIL/poller-pp", CellPollerPP);

}  // namespace
}  // namespace g3

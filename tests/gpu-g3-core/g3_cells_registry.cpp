// GPU paged-KV plan (rev 16), step G3: the registry cell (§4.4 *The registry cell*), rebuilt against
// the built registry (src/gpu/gpu_inflight_registry.cpp), the ledger and the tokens.
//
//   REGRACE/pollers   strike round 5's regrace.cpp, two-poller form (`poll ... stash`): the maker
//                     decodes and stashes, a block=0 poller on another thread finishes each decode
//                     (re-registering the maker's record into its own list and freeing it) while
//                     the maker walks its own list at the return of non-submitting calls on
//                     another handle. Graded under ThreadSanitizer: no race, no entry left in the
//                     maker's list, every decode walked and passed, every stash drained and
//                     reported. Mutant: the list mutex removed, which must go red (graded here).
//   REGRACE/serial    the serialised form (`serial ... stash`): one blocking waiter, serialised
//                     with the submits, beside the maker's non-submitting calls; 100,000 decodes.
//   REGISTRY/churn    a maker thread that exits before its record is finished on another thread
//                     (the round-6 churn shape), under AddressSanitizer: clean with the lists held
//                     by shared pointer; mutant: a plain thread_local list (heap-use-after-free).
//   GRID/depth0       the depth-0 grid (rev 14.1, decision 81): each registration event (Submit(),
//                     move-construction, move-assignment, the accessor's lowering), on a record
//                     made under the open call and on one made under an earlier call, performed on
//                     the calling thread, in a task under the trampoline, and on a host thread
//                     running no task: 21 cells, each green (reports 1, next guarded singleton
//                     call SSLM_OK) or named outside the reach (the host thread's Submit() and its
//                     events on the earlier call's record: reports 0, SSLM_BUSY).
//   GRID/tasks        four tasks registering at once in the calling thread's list, under
//                     ThreadSanitizer: clean, each call reporting 4.
//   GRID/empty-binding a task handed out with no scope open and run by a host thread inside its
//                     own call, while the handing-out thread takes the task's token back: clean
//                     under ThreadSanitizer with the empty binding (rev 14.2); mutant: a trampoline
//                     that installs nothing (red).
#include "g3_fakes.h"
#include "g3_harness.h"

#include <atomic>
#include <cstdio>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

namespace g3 {
namespace {

uint64_t Reports() { return inflight::BoundaryCheckReportCount(); }
uint64_t Walked() { return inflight::BoundaryCheckWalkedCount(); }

// ---- REGRACE ------------------------------------------------------------------------------------
// regrace.cpp's protocol: T1 (the maker) decodes the handle X and hands it to T2 only after the
// decode has returned; T2 finishes it through Ready. regrace models X by a record pointer; here X
// alternates between two real handles, so T1 reuses a handle only after T2 has taken the other
// one, which it does only after finishing the first (one thread per handle at a time).
struct RegraceResult {
	long decodes = 0, readies = 0, busy = 0;
	uint64_t reports = 0, walked = 0;
	size_t maker_list_left = 0;
	int64_t live_after = 0;
	bool observed_at_end = false;
};

RegraceResult RunRegrace(bool poll, long iters) {
	FakeDevice dev(/*auto_complete=*/true);
	FakeHandle X[2];
	FakeHandle Y;
	std::mutex serial;  // the host's serialisation of submitters and blocking waiters (P1)
	std::atomic<FakeHandle*> handed{nullptr};
	std::atomic<bool> stop{false};
	std::atomic<long> sink{0};
	RegraceResult res;
	const uint64_t r0 = Reports(), w0 = Walked();
	std::thread t2([&] {
		auto take = [&] {
			FakeHandle* h = handed.exchange(nullptr);
			if (!h) return false;
			if (poll) {
				while (!Ready(h, /*block=*/false)) std::this_thread::yield();
			} else {
				std::lock_guard<std::mutex> g(serial);
				(void)Ready(h, /*block=*/true);
			}
			++res.readies;
			return true;
		};
		while (!stop.load()) {
			if (!take()) std::this_thread::yield();  // idle: let a descheduled maker run (oversubscribed runners)
		}
		while (take()) {
		}
	});
	for (long i = 0; i < iters; ++i) {
		// Waiting for T2, the maker walks its own list at the return of non-submitting calls beside
		// the finisher (the race the lock closes); each wait step yields, so an oversubscribed runner
		// does not spin against a descheduled finisher.
		while (handed.load() != nullptr) {
			NonSubmitting(&Y, &sink);
			std::this_thread::yield();
		}
		FakeHandle* x = &X[i & 1];
		while (x->state.load() == St::Submitted) {  // T2 still finishing it
			NonSubmitting(&Y, &sink);
			std::this_thread::yield();
		}
		{
			// The stash first (its walk drains it, observing a value above X's), then the decode.
			std::unique_lock<std::mutex> g(serial, std::defer_lock);
			if (!poll) g.lock();
			Stash(dev);
			if (Decode(dev, x) != Status::Ok) ++res.busy;
		}
		++res.decodes;
		handed.store(x);
		for (int k = 0; k < 4; ++k) NonSubmitting(&Y, &sink);
	}
	while (handed.load() != nullptr) std::this_thread::yield();
	stop.store(true);
	t2.join();
	res.reports = Reports() - r0;
	res.walked = Walked() - w0;
	res.maker_list_left = inflight::OwnListEntriesForTest();
	res.live_after = inflight::LiveRecordsForTest();
	res.observed_at_end = !dev.ledger().Unobserved() && !dev.ledger().Stranded();
	return res;
}

void CheckRegrace(const RegraceResult& r, long iters, const char* mode) {
	std::printf("regrace %s: decodes=%ld readies=%ld busy=%ld reports=%llu walked_pass=%llu maker_list_left=%zu "
	            "live_records=%lld\n",
	            mode, r.decodes, r.readies, r.busy, static_cast<unsigned long long>(r.reports),
	            static_cast<unsigned long long>(r.walked), r.maker_list_left, static_cast<long long>(r.live_after));
	G3_CHECK_EQ(r.decodes, iters);
	G3_CHECK_EQ(r.readies, iters);
	G3_CHECK_EQ(r.busy, 0);
	G3_CHECK_EQ(r.reports, static_cast<uint64_t>(iters));  // every stash drained and reported
	G3_CHECK_EQ(r.walked, static_cast<uint64_t>(iters));   // every decode walked and passed
	G3_CHECK_EQ(r.maker_list_left, 0u);
	G3_CHECK_EQ(r.live_after, 0);
	G3_CHECK(r.observed_at_end);
}

void CellRegracePollers() {
	const long n = G3Param("regrace_poll", 20000);
	CheckRegrace(RunRegrace(true, n), n, "poll(block=0, exempt) stash");
}
G3_CELL("REGRACE/pollers", CellRegracePollers);

void CellRegraceSerial() {
	const long n = G3Param("regrace_serial", 100000);
	CheckRegrace(RunRegrace(false, n), n, "serialised(block=1) stash");
}
G3_CELL("REGRACE/serial", CellRegraceSerial);

// ---- REGISTRY/churn -------------------------------------------------------------------------------
void CellChurn() {
	const long n = G3Param("churn", 500);
	FakeDevice dev(/*auto_complete=*/true);
	FakeHandle A, B;
	std::mutex p1;
	std::atomic<bool> stop{false};
	std::atomic<long> sink{0};
	const uint64_t r0 = Reports();
	std::thread other([&] {
		while (!stop.load()) {  // an unrelated walker
			NonSubmitting(&B, &sink);
			std::this_thread::yield();
		}
	});
	long busy = 0, unfinished = 0;
	for (long i = 0; i < n; ++i) {
		std::thread maker([&] {
			std::lock_guard<std::mutex> g(p1);
			Stash(dev);
			if (Decode(dev, &A) != Status::Ok) ++busy;
		});  // the maker's thread exits with A's record still registered in its list
		maker.join();
		if (!Ready(&A, /*block=*/false)) ++unfinished;  // re-registers out of the dead thread's list
	}
	stop.store(true);
	other.join();
	G3_CHECK_EQ(busy, 0);
	G3_CHECK_EQ(unfinished, 0);
	G3_CHECK_EQ(Reports() - r0, static_cast<uint64_t>(n));
	G3_CHECK_EQ(inflight::LiveRecordsForTest(), 0);
}
G3_CELL("REGISTRY/churn", CellChurn);

// ---- GRID/depth0 ----------------------------------------------------------------------------------
TestToken g_worker_static;
std::unique_ptr<TestToken> g_worker_heap;

const char* const kEvents[] = {"submit", "move-construct", "move-assign", "accessor-lowering"};
const char* const kThreads[] = {"caller", "task", "host"};

// A submitter, run as one C ABI call (the caller holds P1's lock). `thr`: 0 the calling thread,
// 1 a SuperSLM task (the trampoline, through a host parallel-for run), 2 a host thread running no
// task.
struct GridCtx {
	FakeDevice* dev;
	FakeHandle* a;
	int ev;
	bool prior;
	TestToken fresh;
};
void GridEventBody(GridCtx& c) {
	if (c.ev == 0) {
		g_worker_static = TestToken::Make(c.dev->Submit(), 0);
		return;
	}
	TestToken& src = c.prior ? c.a->in_flight : c.fresh;
	if (c.ev == 1) g_worker_heap = std::make_unique<TestToken>(std::move(src));
	else if (c.ev == 2) g_worker_static = std::move(src);
	else SetState(c.a, St::Idle);
}
void GridTask(void* ctx, int32_t) { GridEventBody(*static_cast<GridCtx*>(ctx)); }

void GridEvent(FakeDevice& dev, FakeHandle* a, int ev, bool prior, int thr) {
	inflight::GpuApiCallScope call("grid_event");
	GridCtx c{&dev, a, ev, prior, TestToken()};
	if (ev != 0 && !prior) {
		if (ev == 3) {
			a->in_flight = TestToken::Make(dev.Submit(), 0);
			SetState(a, St::Submitted);
		} else {
			c.fresh = TestToken::Make(dev.Submit(), 0);
		}
	}
	if (thr == 0) {
		GridEventBody(c);
	} else if (thr == 1) {
		inflight::GpuTaskTrampoline tramp(&GridTask, &c);
		HostRun(1, &inflight::GpuTaskTrampoline::Invoke, &tramp);
	} else {
		std::thread w([&] { GridEventBody(c); });
		w.join();
	}
}

void CellGrid() {
	FakeDevice dev(/*auto_complete=*/true);
	std::mutex p1;
	int cells = 0, wrong = 0;
	for (int ev = 0; ev < 4; ++ev) {
		for (int prior = 0; prior < 2; ++prior) {
			if (ev == 0 && prior) continue;  // a Submit() makes a new record: no earlier-call cell
			for (int thr = 0; thr < 3; ++thr) {
				FakeHandle a, next;
				if (prior) {
					std::lock_guard<std::mutex> g(p1);
					if (Decode(dev, &a) != Status::Ok) {  // the earlier call: passes, on a Submitted handle
						G3_CHECK_MSG(false, "the earlier call's decode was refused");
						continue;
					}
				}
				const uint64_t r0 = Reports();
				{
					std::lock_guard<std::mutex> g(p1);
					GridEvent(dev, &a, ev, prior == 1, thr);
				}
				const uint64_t reports = Reports() - r0;
				Status st;
				{
					std::lock_guard<std::mutex> g(p1);
					st = Decode(dev, &next);  // the next guarded singleton-reaching call
				}
				const bool named = thr == 2 && (ev == 0 || prior == 1);  // P2 (d)'s named residue
				const bool ok = named ? (reports == 0 && st == Status::Busy) : (reports == 1 && st == Status::Ok);
				std::printf("grid: event=%s record=%s thread=%s reports=%llu next_guarded_call=%s%s%s\n", kEvents[ev],
				            prior ? "prior" : "fresh", kThreads[thr], static_cast<unsigned long long>(reports),
				            StatusName(st), named ? " (named outside the reach)" : "", ok ? "" : "  <-- WRONG");
				G3_CHECK_MSG(ok, "grid cell %s/%s/%s: reports %llu, %s", kEvents[ev], prior ? "prior" : "fresh",
				             kThreads[thr], static_cast<unsigned long long>(reports), StatusName(st));
				++cells;
				if (!ok) ++wrong;
				// Clean up so the next cell starts observed (the internal layer's DrainSubmissionsGpu).
				{
					std::lock_guard<std::mutex> g(p1);
					if (next.state.load() == St::Submitted) (void)Ready(&next, true);
					g_worker_static = TestToken();
					g_worker_heap.reset();
					a.in_flight = TestToken();
					a.state.store(St::Idle);
					dev.Drain();
				}
			}
		}
	}
	G3_CHECK_EQ(cells, 21);
	std::printf("grid: %d cells, %d wrong\n", cells, wrong);
	G3_CHECK_EQ(inflight::LiveRecordsForTest(), 0);
}
G3_CELL("GRID/depth0", CellGrid);

// ---- GRID/tasks -------------------------------------------------------------------------------------
TestToken g_task_tok[4];
struct TasksCtx {
	FakeDevice* devs;  // one per task: each task submits on its own device (P1 is per device)
	std::atomic<int> go{0};
};
void FourTask(void* ctx, int32_t i) {
	TasksCtx* c = static_cast<TasksCtx*>(ctx);
	c->go.fetch_add(1);
	while (c->go.load() < 4) std::this_thread::yield();
	TestToken t = TestToken::Make(c->devs[i].Submit(), static_cast<int>(i));
	g_task_tok[i] = std::move(t);
}

void CellTasks() {
	const long n = G3Param("tasks", 300);
	FakeDevice devs[4];
	long bad = 0;
	for (long i = 0; i < n; ++i) {
		const uint64_t r0 = Reports();
		{
			inflight::GpuApiCallScope call("four_tasks");
			TasksCtx c;
			c.devs = devs;
			inflight::GpuTaskTrampoline tramp(&FourTask, &c);
			HostRun(4, &inflight::GpuTaskTrampoline::Invoke, &tramp);
		}
		if (Reports() - r0 != 4) ++bad;
	}
	G3_CHECK_EQ(bad, 0);
	G3_CHECK_EQ(inflight::LiveRecordsForTest(), 0);
	for (FakeDevice& d : devs) G3_CHECK(!d.ledger().Unobserved());
}
G3_CELL("GRID/tasks", CellTasks);

// ---- GRID/empty-binding ------------------------------------------------------------------------------
// T hands a task out (with or without its own scope open); W, a host thread inside its own call,
// runs it; T takes the task's token back while W's call closes (plan review round 14, W1's model).
TestToken g_steal_slot;
struct StealCtx {
	FakeDevice* dev;
};
void StealTask(void* ctx, int32_t) {
	StealCtx* c = static_cast<StealCtx*>(ctx);
	g_steal_slot = TestToken::Make(c->dev->Submit(), 0);
}

long RunSteal(bool tscope, long n) {
	FakeDevice dev(/*auto_complete=*/true);
	for (long i = 0; i < n; ++i) {
		std::atomic<int> task_done{0}, w_close{0};
		TestToken mine;
		{
			std::unique_ptr<inflight::GpuApiCallScope> t_call(tscope ? new inflight::GpuApiCallScope("t_call") : nullptr);
			StealCtx c{&dev};
			inflight::GpuTaskTrampoline tramp(&StealTask, &c);
			std::thread W([&] {
				inflight::GpuApiCallScope w_call("w_call");
				inflight::GpuTaskTrampoline::Invoke(&tramp, 0);
				task_done.store(1);
				while (w_close.load() == 0) std::this_thread::yield();
			});
			while (task_done.load() == 0) std::this_thread::yield();
			w_close.store(1);
			mine = std::move(g_steal_slot);
			W.join();
		}
	}
	return static_cast<long>(inflight::LiveRecordsForTest());
}

void CellEmptyBinding() {
	const long n = G3Param("steal", 200);
	const uint64_t r0 = Reports();
	G3_CHECK_EQ(RunSteal(false, n), 0);  // no scope open: the empty binding
	G3_CHECK_EQ(Reports() - r0, 0u);     // W's walk never took the task's record
	const uint64_t r1 = Reports();
	G3_CHECK_EQ(RunSteal(true, n), 0);   // T's scope open: T's call walks (and drains) `mine`
	G3_CHECK_EQ(Reports() - r1, static_cast<uint64_t>(n));
}
G3_CELL("GRID/empty-binding", CellEmptyBinding);

// The trampoline restores the thread's binding on every exit, and a task run on the calling
// thread changes nothing.
void CellTrampolineRestores() {
	FakeDevice dev(/*auto_complete=*/true);
	G3_CHECK_EQ(inflight::DepthForTest(), 0);
	{
		inflight::GpuApiCallScope call("restores");
		struct Ctx {
			int depth_inside = -1;
		} c;
		auto task = [](void* p, int32_t) { static_cast<Ctx*>(p)->depth_inside = inflight::DepthForTest(); };
		inflight::GpuTaskTrampoline tramp(task, &c);
		inflight::GpuTaskTrampoline::Invoke(&tramp, 0);  // on the calling thread
		G3_CHECK_EQ(c.depth_inside, 1);
		G3_CHECK_EQ(inflight::DepthForTest(), 1);
		std::thread host([&] {
			{
				inflight::GpuApiCallScope own("host_own_call");
				G3_CHECK_EQ(inflight::DepthForTest(), 1);
				inflight::GpuTaskTrampoline::Invoke(&tramp, 0);
				G3_CHECK_EQ(inflight::DepthForTest(), 1);  // restored to the host's own call
			}
			G3_CHECK_EQ(inflight::DepthForTest(), 0);
		});
		host.join();
	}
	// With no scope open the trampoline installs the empty binding: depth 0 inside the task, even
	// on a host thread that is inside its own call.
	struct Ctx2 {
		int depth_inside = -1;
	} c2;
	auto task2 = [](void* p, int32_t) { static_cast<Ctx2*>(p)->depth_inside = inflight::DepthForTest(); };
	inflight::GpuTaskTrampoline tramp2(task2, &c2);
	std::thread host2([&] {
		inflight::GpuApiCallScope own("host_own_call");
		inflight::GpuTaskTrampoline::Invoke(&tramp2, 0);
		G3_CHECK_EQ(inflight::DepthForTest(), 1);
	});
	host2.join();
	G3_CHECK_EQ(c2.depth_inside, 0);
}
G3_CELL("GRID/trampoline-restores", CellTrampolineRestores);

}  // namespace
}  // namespace g3

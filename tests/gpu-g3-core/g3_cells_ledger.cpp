// GPU paged-KV plan (rev 16), step G3: the submission ledger's host-only cells.
//
//   3.6h/*            §7 cell 3.6h, the ledger's concurrency (one serialised submitter, concurrent
//                     observers landing Poll and Wait results in every order), with its two legs:
//                     whole landings in every order (named mutant: observed_ stored plainly), and
//                     the yield hook inside one update (named mutant: a load, a compare and a
//                     separate store). Run under ThreadSanitizer in the g++ 13.3 route.
//   G3/ledger/*       the ledger's rules of §4.4 that need no D3D12: Submit()'s fresh-value retry,
//                     the recovered and stranded outcomes, the check before the reset, Wait(v)'s
//                     fast path, the stranded wait, the not-ready poll, Drain(), and the boundary
//                     guard check inside BeginRecording().
#include "g3_fakes.h"
#include "g3_harness.h"

#include <algorithm>
#include <atomic>
#include <thread>
#include <vector>

namespace g3 {
namespace {

// Submits `n` times on `dev` without observing (no reset between: the cell drives the ledger, not
// a verb), keeping every submission so nothing drains until the caller drops them.
std::vector<GpuSubmission> SubmitN(FakeDevice& dev, int n) {
	std::vector<GpuSubmission> subs;
	for (int i = 0; i < n; ++i) subs.push_back(dev.Submit());
	return subs;
}

bool CheckFires(const SubmissionLedger& l) {
	try {
		superslm_gpu::harness::PoolSubmitAssert(l);
		return false;
	} catch (const PoolSubmitContractError&) {
		return true;
	} catch (const GpuSubmissionStrandedError&) {
		return true;
	}
}

// ---- 3.6h, leg 1: whole landings in every order ----------------------------------------------
void Cell36hOrders() {
	int perm[3] = {1, 2, 3};
	int orders = 0;
	do {
		FakeDevice dev(/*auto_complete=*/false);
		std::vector<GpuSubmission> subs = SubmitN(dev, 3);
		dev.Complete(3);
		G3_CHECK(CheckFires(dev.ledger()));  // observed_ 0 < submitted_ 3
		uint64_t high = 0;
		bool decreased = false;
		for (int k = 0; k < 3; ++k) {
			const uint64_t v = static_cast<uint64_t>(perm[k]);
			std::thread observer([&] { (void)dev.Poll(v); });  // each landing on its own thread
			observer.join();
			const uint64_t now = dev.ledger().observed();
			if (now < high) decreased = true;
			high = std::max(high, now);
		}
		G3_CHECK_MSG(!decreased, "order %d,%d,%d: observed_ decreased", perm[0], perm[1], perm[2]);
		G3_CHECK_MSG(dev.ledger().observed() == 3, "order %d,%d,%d: observed_ ends at %llu, not 3", perm[0], perm[1],
		             perm[2], static_cast<unsigned long long>(dev.ledger().observed()));
		G3_CHECK(!CheckFires(dev.ledger()));  // observed, nothing stranded: the check does not fire
		++orders;
	} while (std::next_permutation(perm, perm + 3));
	G3_CHECK_EQ(orders, 6);
}
G3_CELL("3.6h/orders", Cell36hOrders);

// ---- 3.6h: one serialised submitter, concurrent observers ------------------------------------
void Cell36hConcurrent() {
	const long submits = G3Param("submits", 4000);
	const int observers = 4;
	FakeDevice dev(/*auto_complete=*/false, /*complete_on_arm=*/false);
	std::atomic<bool> done{false};
	std::atomic<long> regressions{0}, above_fence{0};
	std::vector<GpuSubmission> subs;
	subs.reserve(static_cast<size_t>(submits));
	// The GPU: completes the fence in steps behind the submitter.
	std::thread gpu([&] {
		uint64_t c = 0;
		while (!done.load() || c < dev.ledger().submitted()) {
			const uint64_t s = dev.ledger().submitted();
			if (c < s) dev.Complete(++c);
			else std::this_thread::yield();
		}
	});
	// A monitor: observed_ never decreases, and never passes the fence.
	std::thread monitor([&] {
		uint64_t last = 0;
		while (!done.load()) {
			const uint64_t o = dev.ledger().observed();
			if (o < last) regressions.fetch_add(1);
			if (o > dev.completed()) above_fence.fetch_add(1);
			last = std::max(last, o);
		}
	});
	std::vector<std::thread> obs;
	for (int k = 0; k < observers; ++k) {
		obs.emplace_back([&, k] {
			uint64_t x = 0x9E3779B97F4A7C15ull * static_cast<uint64_t>(k + 1);
			while (!done.load()) {
				x ^= x << 13;
				x ^= x >> 7;
				x ^= x << 17;
				const uint64_t s = dev.ledger().submitted();
				if (s == 0) continue;
				const uint64_t v = 1 + x % s;
				if (k % 2 == 0) (void)dev.Poll(v);  // pollers
				else dev.Wait(v);                    // blocking observers (in the cell, distinct threads)
			}
		});
	}
	for (long i = 0; i < submits; ++i) subs.push_back(dev.Submit());  // the one submitter
	done.store(true);
	for (std::thread& t : obs) t.join();
	monitor.join();
	gpu.join();
	G3_CHECK_EQ(regressions.load(), 0);
	G3_CHECK_EQ(above_fence.load(), 0);
	G3_CHECK_EQ(dev.ledger().submitted(), static_cast<uint64_t>(submits));
	// The check fires if and only if observed_ < submitted_ (nothing is stranded here).
	const bool unobserved = dev.ledger().observed() < dev.ledger().submitted();
	G3_CHECK_EQ(CheckFires(dev.ledger()), unobserved);
	(void)dev.Poll(static_cast<uint64_t>(submits));
	G3_CHECK(!CheckFires(dev.ledger()));
	G3_CHECK(!dev.ledger().Stranded());
	subs.clear();  // every submission already observed: each drain returns without arming
}
G3_CELL("3.6h/concurrent", Cell36hConcurrent);

// ---- 3.6h, leg 2: the yield hook inside one update ---------------------------------------------
#if defined(SUPERSLM_LEDGER_YIELD_HOOK)
std::atomic<int> g_hook_paused{0};
std::atomic<int> g_hook_go{0};
void PauseLandingThree(uint64_t landing) {
	if (landing != 3) return;
	g_hook_paused.store(1);
	while (g_hook_go.load() == 0) std::this_thread::yield();
}
#endif

void Cell36hYieldHook() {
#if !defined(SUPERSLM_LEDGER_YIELD_HOOK)
	G3_CHECK_MSG(false, "built without SUPERSLM_LEDGER_YIELD_HOOK: the hook leg cannot run");
#else
	FakeDevice dev(/*auto_complete=*/false);
	std::vector<GpuSubmission> subs = SubmitN(dev, 5);  // submitted_ = 5
	dev.Complete(5);
	g_hook_paused.store(0);
	g_hook_go.store(0);
	superslm_gpu::harness::LedgerYieldHookForTest().store(&PauseLandingThree);
	// The observer landing 3 pauses after its load of observed_; the observer landing 5 runs to
	// completion there; then the first resumes.
	std::thread three([&] { (void)dev.Poll(3); });
	while (g_hook_paused.load() == 0) std::this_thread::yield();
	std::thread five([&] { (void)dev.Poll(5); });
	five.join();
	g_hook_go.store(1);
	three.join();
	superslm_gpu::harness::LedgerYieldHookForTest().store(nullptr);
	G3_CHECK_MSG(dev.ledger().observed() == 5, "observed_ ends at %llu, not 5",
	             static_cast<unsigned long long>(dev.ledger().observed()));
	// The operand: with nothing Submitted (every submission's fence reached and landed), the check
	// does not fire, so the guard's ledger term reads observed, not Busy.
	bool fired = false;
	try {
		superslm_gpu::harness::PoolSubmitAssert(dev.ledger());
	} catch (const PoolSubmitContractError&) {
		fired = true;
	}
	G3_CHECK_MSG(!fired, "the check throws PoolSubmitContractError with nothing Submitted");
	G3_CHECK(dev.ledger().ReadGuardTerm() == SubmissionLedger::GuardTerm::Observed);
#endif
}
G3_CELL("3.6h/yield-hook", Cell36hYieldHook);

// ---- 3.6h: stranded is terminal; lost_ once set stays set ---------------------------------------
void Cell36hStranded() {
	FakeDevice dev(/*auto_complete=*/false, /*complete_on_arm=*/true);
	{
		GpuSubmission s = dev.Submit();
		dev.FailArms(true);
		bool threw = false;
		try {
			s.Wait();
		} catch (const GpuSubmissionStrandedError&) {
			threw = true;
		}
		G3_CHECK(threw);
		G3_CHECK(!s.holds());
	}
	G3_CHECK(dev.ledger().lost());
	dev.FailArms(false);
	dev.Complete(1);
	(void)dev.Poll(1);  // a later observation completes the value but does not clear lost_
	G3_CHECK(dev.ledger().lost());
	G3_CHECK(dev.ledger().Stranded());
	G3_CHECK(CheckFires(dev.ledger()));
	G3_CHECK(dev.ledger().ReadGuardTerm() == SubmissionLedger::GuardTerm::Stranded);
}
G3_CELL("3.6h/stranded-terminal", Cell36hStranded);

// ---- G3/ledger ----------------------------------------------------------------------------------

// A failed Signal with retry: the retry signals at a fresh value, Submit() drains it and rethrows
// the first failure; the ledger is observed and the next reset is clean.
void CellSubmitRecovered() {
	FakeDevice dev(/*auto_complete=*/false, /*complete_on_arm=*/true);
	const int64_t live0 = inflight::LiveRecordsForTest();
	dev.FailNextSignals(1, 7);
	int32_t code = -1;
	bool other = false;
	try {
		GpuSubmission s = dev.Submit();
		(void)s;
	} catch (const FakeSignalFailure& e) {
		code = e.code;
	} catch (...) {
		other = true;
	}
	G3_CHECK_EQ(code, 7);
	G3_CHECK(!other);
	G3_CHECK_EQ(dev.signal_attempts().size(), 2u);
	if (dev.signal_attempts().size() == 2) {
		G3_CHECK_EQ(dev.signal_attempts()[0], 1u);
		G3_CHECK_EQ(dev.signal_attempts()[1], 2u);  // fresh, never the failed value
	}
	G3_CHECK_EQ(dev.ledger().submitted(), 2u);
	G3_CHECK_EQ(dev.ledger().observed(), 2u);  // drained inside Submit()
	G3_CHECK(!dev.ledger().unsignaled());
	G3_CHECK(!dev.ledger().Stranded());
	G3_CHECK_EQ(inflight::LiveRecordsForTest(), live0);  // nothing left in flight
	bool reset_threw = false;
	try {
		dev.BeginRecording();
	} catch (...) {
		reset_threw = true;
	}
	G3_CHECK(!reset_threw);
	G3_CHECK_EQ(dev.resets(), 1);
}
G3_CELL("G3/ledger/submit-recovered", CellSubmitRecovered);

void CellSubmitRetryStrands() {
	FakeDevice dev;
	const int64_t live0 = inflight::LiveRecordsForTest();
	dev.FailNextSignals(2, 9);
	bool stranded = false;
	try {
		GpuSubmission s = dev.Submit();
		(void)s;
	} catch (const GpuSubmissionStrandedError&) {
		stranded = true;
	}
	G3_CHECK(stranded);
	G3_CHECK_EQ(dev.signal_attempts().size(), 2u);
	G3_CHECK(dev.ledger().unsignaled());
	G3_CHECK(dev.ledger().Stranded());
	G3_CHECK(dev.ledger().ReadGuardTerm() == SubmissionLedger::GuardTerm::Stranded);
	G3_CHECK_EQ(inflight::LiveRecordsForTest(), live0);
	bool reset_stranded = false;
	try {
		dev.BeginRecording();
	} catch (const GpuSubmissionStrandedError&) {
		reset_stranded = true;
	} catch (...) {
	}
	G3_CHECK(reset_stranded);
	G3_CHECK_EQ(dev.resets(), 0);
	bool drain_stranded = false;
	try {
		dev.Drain();
	} catch (const GpuSubmissionStrandedError&) {
		drain_stranded = true;
	}
	G3_CHECK(drain_stranded);
}
G3_CELL("G3/ledger/submit-retry-strands", CellSubmitRetryStrands);

void CellSubmitNoRetry() {
	FakeDevice dev;
	dev.FailNextSignals(1, 5);
	bool stranded = false;
	try {
		GpuSubmission s = dev.Submit(/*retry=*/false);
		(void)s;
	} catch (const GpuSubmissionStrandedError&) {
		stranded = true;
	}
	G3_CHECK(stranded);
	G3_CHECK_EQ(dev.signal_attempts().size(), 1u);
	G3_CHECK(dev.ledger().Stranded());
}
G3_CELL("G3/ledger/submit-no-retry", CellSubmitNoRetry);

// The check runs before the reset, reads host values only, and fires while the last submission is
// unobserved, even though the fake GPU has finished it.
void CellCheckBeforeReset() {
	FakeDevice dev(/*auto_complete=*/true);
	GpuSubmission s = dev.Submit();
	G3_CHECK(dev.completed() >= 1);  // the GPU is done; the host has not observed it
	bool contract = false;
	try {
		dev.BeginRecording();
	} catch (const PoolSubmitContractError&) {
		contract = true;
	} catch (...) {
	}
	G3_CHECK(contract);
	G3_CHECK_EQ(dev.resets(), 0);
	G3_CHECK(dev.ledger().ReadGuardTerm() == SubmissionLedger::GuardTerm::Unobserved);
	G3_CHECK(dev.Poll(s.fence_value()));
	bool threw = false;
	try {
		dev.BeginRecording();
	} catch (...) {
		threw = true;
	}
	G3_CHECK(!threw);
	G3_CHECK_EQ(dev.resets(), 1);
}
G3_CELL("G3/ledger/check-before-reset", CellCheckBeforeReset);

// Wait(v) returns without arming the event when the fence is already at v.
void CellWaitFastPath() {
	FakeDevice dev(/*auto_complete=*/false, /*complete_on_arm=*/true);
	GpuSubmission a = dev.Submit();
	GpuSubmission b = dev.Submit();
	dev.Complete(1);
	dev.Wait(1);
	G3_CHECK_EQ(dev.arms(), 0);
	G3_CHECK_EQ(dev.ledger().observed(), 1u);
	dev.Wait(2);  // below the fence: armed once
	G3_CHECK_EQ(dev.arms(), 1);
	G3_CHECK_EQ(dev.ledger().observed(), 2u);
	a.Wait();
	b.Wait();
	G3_CHECK_EQ(dev.arms(), 1);
}
G3_CELL("G3/ledger/wait-fast-path", CellWaitFastPath);

void CellPollNotReady() {
	FakeDevice dev(/*auto_complete=*/false);
	GpuSubmission s = dev.Submit();
	for (int i = 0; i < 3; ++i) G3_CHECK(!dev.Poll(1));
	G3_CHECK_EQ(dev.ledger().observed(), 0u);
	G3_CHECK_EQ(dev.arms(), 0);
	dev.Complete(1);
	G3_CHECK(dev.Poll(1));
	G3_CHECK_EQ(dev.ledger().observed(), 1u);
}
G3_CELL("G3/ledger/poll-not-ready", CellPollNotReady);

void CellDrain() {
	FakeDevice dev(/*auto_complete=*/false, /*complete_on_arm=*/true);
	dev.Drain();  // nothing submitted: returns at once
	G3_CHECK_EQ(dev.arms(), 0);
	std::vector<GpuSubmission> subs = SubmitN(dev, 3);
	dev.Drain();
	G3_CHECK_EQ(dev.ledger().observed(), 3u);
	G3_CHECK(!CheckFires(dev.ledger()));
}
G3_CELL("G3/ledger/drain", CellDrain);

// The boundary guard check inside BeginRecording(): inside a C ABI call that passed no
// EntrySubmitGuard the reset is refused even with nothing pending; outside every call it is not.
void CellBoundaryGuard() {
	FakeDevice dev;
	auto reset_refused = [&] {
		try {
			dev.BeginRecording();
			return false;
		} catch (const PoolSubmitContractError&) {
			return true;
		}
	};
	G3_CHECK(!reset_refused());  // internal layer, depth 0: exempt
	{
		inflight::GpuApiCallScope call("unguarded_verb");
		G3_CHECK(reset_refused());  // nothing in flight, still refused
	}
	{
		inflight::GpuApiCallScope call("guarded_verb");
		inflight::NoteEntrySubmitGuardRan();
		G3_CHECK(!reset_refused());
	}
	{
		inflight::GpuApiCallScope call("next_call");  // the flag does not survive the call
		G3_CHECK(reset_refused());
	}
	{
		// Named blind spot (§4.4, D4): a guard run by an inner guarded verb counts for the outer call.
		inflight::GpuApiCallScope outer("unguarded_composite");
		{
			inflight::GpuApiCallScope inner("inner_guarded_verb");
			inflight::NoteEntrySubmitGuardRan();
		}
		G3_CHECK(!reset_refused());
	}
	G3_CHECK_EQ(inflight::DepthForTest(), 0);
}
G3_CELL("G3/ledger/boundary-guard", CellBoundaryGuard);

}  // namespace
}  // namespace g3

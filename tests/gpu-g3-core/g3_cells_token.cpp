// GPU paged-KV plan (rev 16), step G3: the tokens' host-only cells (GpuInFlightToken and
// GpuSubmission, §4.4 *The submission ledger* and *Three states*) and the scope's.
//
//   G3/token/*        move (empties its source, drains nothing), move-assign over a live token
//                     drains first, the not-ready poll never empties a token, drain on destruction,
//                     a destructor whose drain strands returns with the ledger stranded, and
//                     re-registration on every move and on the accessor's lowering (the walk follows
//                     the record wherever the move put it);
//   G3/submission/*   the synchronous site's early return drains, a move-assign drains first, a
//                     GpuSubmission kept in a file-scope variable is drained and reported;
//   G3/scope/*        the check runs at the outermost return only (a composite holding a local
//                     across an inner call), and counts per thread (a poller returning beside
//                     another thread's live local).
#include "g3_fakes.h"
#include "g3_harness.h"

#include <atomic>
#include <memory>
#include <thread>

namespace g3 {
namespace {

uint64_t Reports() { return inflight::BoundaryCheckReportCount(); }
uint64_t Walked() { return inflight::BoundaryCheckWalkedCount(); }

void CellTokenMove() {
	FakeDevice dev(/*auto_complete=*/false, /*complete_on_arm=*/true);
	TestToken a = TestToken::Make(dev.Submit(), 5);
	G3_CHECK(a.holds());
	const uint64_t v = a.fence_value();
	G3_CHECK_EQ(v, 1u);
	TestToken b(std::move(a));
	G3_CHECK(!a.holds());
	G3_CHECK(b.holds());
	G3_CHECK_EQ(b.fence_value(), v);
	G3_CHECK_EQ(b->tag, 5);
	G3_CHECK_EQ(dev.arms(), 0);  // a move drains nothing
	G3_CHECK_EQ(dev.ledger().observed(), 0u);
	TestToken c;
	c = std::move(b);  // assignment into an empty token: nothing to drain
	G3_CHECK(!b.holds());
	G3_CHECK(c.holds());
	G3_CHECK_EQ(dev.arms(), 0);
	dev.Complete(1);
}
G3_CELL("G3/token/move", CellTokenMove);

void CellTokenMoveAssignDrains() {
	FakeDevice dev(/*auto_complete=*/false, /*complete_on_arm=*/true);
	const int64_t live0 = inflight::LiveRecordsForTest();
	TestToken a = TestToken::Make(dev.Submit(), 1);  // value 1, incomplete
	TestToken b = TestToken::Make(dev.Submit(), 2);  // value 2, incomplete
	G3_CHECK_EQ(inflight::LiveRecordsForTest(), live0 + 2);
	a = std::move(b);  // a's record (value 1) is drained first
	G3_CHECK_EQ(dev.arms(), 1);
	G3_CHECK(dev.ledger().observed() >= 1u);
	G3_CHECK_EQ(inflight::LiveRecordsForTest(), live0 + 1);
	G3_CHECK(a.holds());
	G3_CHECK_EQ(a.fence_value(), 2u);
	G3_CHECK_EQ(a->tag, 2);
	G3_CHECK(!b.holds());
	// A handle's in_flight keeps its binding across an assignment (it is the handle's token).
	FakeHandle h;
	h.in_flight = std::move(a);
	G3_CHECK(h.in_flight.handle_binding() == &h);
	G3_CHECK(a.handle_binding() == nullptr);
	TestToken moved_out(std::move(h.in_flight));
	G3_CHECK(moved_out.handle_binding() == nullptr);  // a move never transfers the binding
}
G3_CELL("G3/token/move-assign-drains", CellTokenMoveAssignDrains);

// sslm_gpu_ready(block=0)'s finish: the fence below the token's value returns before the move.
enum class Finish { NotReady, Done };
Finish FinishModel(FakeHandle* h, bool block) {
	if (!block && !h->in_flight.ReadyNoWait()) return Finish::NotReady;
	TestToken local = std::move(h->in_flight);
	SetState(h, St::Idle);
	local.Wait();
	return Finish::Done;
}

void CellTokenNotReadyPoll() {
	FakeDevice dev(/*auto_complete=*/false, /*complete_on_arm=*/true);
	FakeHandle h;
	h.in_flight = TestToken::Make(dev.Submit(), 3);
	SetState(&h, St::Submitted);
	for (int i = 0; i < 3; ++i) {
		G3_CHECK(!h.in_flight.ReadyNoWait());
		G3_CHECK(FinishModel(&h, /*block=*/false) == Finish::NotReady);
	}
	G3_CHECK(h.in_flight.holds());  // never emptied
	G3_CHECK(h.state.load() == St::Submitted);
	G3_CHECK_EQ(dev.arms(), 0);  // never armed the shared fence event
	G3_CHECK_EQ(dev.ledger().observed(), 0u);
	dev.Complete(1);
	G3_CHECK(FinishModel(&h, /*block=*/false) == Finish::Done);
	G3_CHECK(!h.in_flight.holds());
	G3_CHECK_EQ(dev.arms(), 0);  // the fence was reached: the finish's wait arms nothing
	G3_CHECK_EQ(dev.ledger().observed(), 1u);
}
G3_CELL("G3/token/not-ready-poll", CellTokenNotReadyPoll);

void CellTokenDrainOnDestruction() {
	FakeDevice dev(/*auto_complete=*/false, /*complete_on_arm=*/true);
	const int64_t live0 = inflight::LiveRecordsForTest();
	{
		TestToken t = TestToken::Make(dev.Submit(), 0);
		G3_CHECK_EQ(inflight::LiveRecordsForTest(), live0 + 1);
	}  // dropped while holding incomplete work
	G3_CHECK_EQ(dev.arms(), 1);
	G3_CHECK_EQ(dev.ledger().observed(), 1u);
	G3_CHECK_EQ(inflight::LiveRecordsForTest(), live0);
	G3_CHECK(!dev.ledger().Unobserved());
	// An exception between the submit and the finish drains on the way out too.
	try {
		TestToken t = TestToken::Make(dev.Submit(), 0);
		throw 42;
	} catch (int) {
	}
	G3_CHECK_EQ(dev.ledger().observed(), 2u);
	G3_CHECK_EQ(inflight::LiveRecordsForTest(), live0);
}
G3_CELL("G3/token/drain-on-destruction", CellTokenDrainOnDestruction);

void CellTokenDestructorStrands() {
	FakeDevice dev(/*auto_complete=*/false, /*complete_on_arm=*/true);
	const int64_t live0 = inflight::LiveRecordsForTest();
	bool threw = false;
	try {
		TestToken t = TestToken::Make(dev.Submit(), 0);
		dev.FailArms(true);
	} catch (...) {
		threw = true;
	}
	G3_CHECK(!threw);  // the destructor never throws
	G3_CHECK(dev.ledger().lost());
	G3_CHECK(dev.ledger().ReadGuardTerm() == SubmissionLedger::GuardTerm::Stranded);
	G3_CHECK_EQ(inflight::LiveRecordsForTest(), live0);  // freed all the same
}
G3_CELL("G3/token/destructor-strands", CellTokenDestructorStrands);

// review-g23 F1: a drain that strands must not free what the submission's work may still use. The
// payload (a stand-in for the layer-loop record's buffers) counts its destruction: a token dropped
// while its drain strands, a throwing Wait() that strands, and a move-assignment over a token whose
// drain strands each hand the payload to the device's graveyard, never destroying it; a confirmed
// drain still frees at once; the graveyard is released only when the device is destroyed.
std::atomic<int> g_kept_payload_destroyed{0};
struct KeptRecord final : GpuInFlightPayload {
	~KeptRecord() override { g_kept_payload_destroyed.fetch_add(1); }

private:
	friend class superslm_gpu::harness::BasicGpuInFlightToken<KeptRecord>;
	explicit KeptRecord(GpuSubmission&& s) : GpuInFlightPayload(std::move(s)) {}
};
using KeptToken = BasicGpuInFlightToken<KeptRecord>;

void CellTokenStrandedKeepsPayload() {
	g_kept_payload_destroyed.store(0);
	const int64_t live0 = inflight::LiveRecordsForTest();
	{
		FakeDevice dev(/*auto_complete=*/false, /*complete_on_arm=*/false);
		// Control: a confirmed drain frees the payload at once.
		{
			KeptToken t = KeptToken::Make(dev.Submit());
			dev.Complete(1);
		}
		G3_CHECK_EQ(g_kept_payload_destroyed.load(), 1);
		G3_CHECK_EQ(dev.stranded_keep_alive_count(), 0u);
		dev.FailArms(true);
		// (a) dropped while its drain strands.
		{
			KeptToken t = KeptToken::Make(dev.Submit());
		}
		G3_CHECK(dev.ledger().lost());
		G3_CHECK_EQ(g_kept_payload_destroyed.load(), 1);
		G3_CHECK_EQ(dev.stranded_keep_alive_count(), 1u);
		// (b) the throwing wait strands: the token is emptied, the payload kept, the error thrown.
		{
			KeptToken t = KeptToken::Make(dev.Submit());
			KeptRecord* const rec = t.get();
			bool stranded = false;
			try {
				t.Wait();
			} catch (const GpuSubmissionStrandedError&) {
				stranded = true;
			}
			G3_CHECK(stranded);
			G3_CHECK(!t.holds());
			G3_CHECK(rec != nullptr);
		}
		G3_CHECK_EQ(g_kept_payload_destroyed.load(), 1);
		G3_CHECK_EQ(dev.stranded_keep_alive_count(), 2u);
		// (c) a move-assignment over a token whose drain strands.
		{
			KeptToken t = KeptToken::Make(dev.Submit());
			t = KeptToken{};
			G3_CHECK(!t.holds());
		}
		G3_CHECK_EQ(g_kept_payload_destroyed.load(), 1);
		G3_CHECK_EQ(dev.stranded_keep_alive_count(), 3u);
		G3_CHECK_EQ(inflight::LiveRecordsForTest(), live0);  // the records themselves hold nothing and are freed
	}  // the device goes: its graveyard is released here, and only here
	G3_CHECK_EQ(g_kept_payload_destroyed.load(), 4);
}
G3_CELL("G3/token/stranded-keeps-payload", CellTokenStrandedKeepsPayload);

// The walk follows the record wherever a move put it, and the accessor's lowering re-registers it.
TestToken g_file_scope_token;               // made at start-up, assigned into later
std::unique_ptr<TestToken> g_heap_token;    // a heap token

void CellTokenReregister() {
	FakeDevice dev(/*auto_complete=*/true);
	// (a) moved into a file-scope token made before the call: drained and reported at the return.
	uint64_t r0 = Reports();
	{
		inflight::GpuApiCallScope call("fin_pipeline_assign");
		TestToken t = TestToken::Make(dev.Submit(), 0);
		g_file_scope_token = std::move(t);
	}
	G3_CHECK_EQ(Reports() - r0, 1u);
	G3_CHECK(!g_file_scope_token.holds());
	G3_CHECK(!dev.ledger().Unobserved());
	// (b) onto a Submitted handle: walked and passed, not drained (the must-accept).
	FakeHandle h;
	r0 = Reports();
	uint64_t w0 = Walked();
	{
		inflight::GpuApiCallScope call("store_on_handle");
		h.in_flight = TestToken::Make(dev.Submit(), 0);
		SetState(&h, St::Submitted);
	}
	G3_CHECK_EQ(Reports() - r0, 0u);
	G3_CHECK_EQ(Walked() - w0, 1u);
	G3_CHECK(h.in_flight.holds());
	// (c) a later call moves the handle's token into static storage and lowers the handle.
	r0 = Reports();
	{
		inflight::GpuApiCallScope call("move_off_handle");
		g_file_scope_token = std::move(h.in_flight);
		SetState(&h, St::Idle);
	}
	G3_CHECK_EQ(Reports() - r0, 1u);
	G3_CHECK(!g_file_scope_token.holds());
	G3_CHECK(!dev.ledger().Unobserved());
	// (d) a later call lowers a Submitted handle and leaves its token on it.
	{
		inflight::GpuApiCallScope call("store_again");
		h.in_flight = TestToken::Make(dev.Submit(), 0);
		SetState(&h, St::Submitted);
	}
	r0 = Reports();
	{
		inflight::GpuApiCallScope call("lower_and_leave");
		SetState(&h, St::Idle);
	}
	G3_CHECK_EQ(Reports() - r0, 1u);
	G3_CHECK(!h.in_flight.holds());
	G3_CHECK(!dev.ledger().Unobserved());
	// (e) stored onto a handle that is not Submitted: drained and reported.
	r0 = Reports();
	{
		inflight::GpuApiCallScope call("store_not_submitted");
		h.in_flight = TestToken::Make(dev.Submit(), 0);
	}
	G3_CHECK_EQ(Reports() - r0, 1u);
	G3_CHECK(!h.in_flight.holds());
	// (f) moved onto the heap, and (g) into a lambda capture kept past the call.
	r0 = Reports();
	{
		inflight::GpuApiCallScope call("heap");
		g_heap_token = std::make_unique<TestToken>(TestToken::Make(dev.Submit(), 0));
	}
	G3_CHECK_EQ(Reports() - r0, 1u);
	G3_CHECK(g_heap_token && !g_heap_token->holds());
	auto make = [&] { return [t = TestToken::Make(dev.Submit(), 0)]() { return t.holds(); }; };
	using Capture = decltype(make());
	static std::unique_ptr<Capture> stored;
	r0 = Reports();
	{
		inflight::GpuApiCallScope call("capture");
		stored = std::make_unique<Capture>(make());
	}
	G3_CHECK_EQ(Reports() - r0, 1u);
	G3_CHECK(stored && !(*stored)());
	G3_CHECK(!dev.ledger().Unobserved());
	g_heap_token.reset();
}
G3_CELL("G3/token/reregister", CellTokenReregister);

// ---- GpuSubmission ----------------------------------------------------------------------------

int SyncSiteEarlyReturn(FakeDevice& dev, bool early) {
	inflight::GpuApiCallScope call("sync_site");
	GpuSubmission sub = dev.Submit();
	if (early) return 1;  // returns between Submit() and its wait
	sub.Wait();
	return 0;
}

void CellSubmissionEarlyReturn() {
	FakeDevice dev(/*auto_complete=*/false, /*complete_on_arm=*/true);
	const uint64_t r0 = Reports();
	const int64_t live0 = inflight::LiveRecordsForTest();
	G3_CHECK_EQ(SyncSiteEarlyReturn(dev, true), 1);
	G3_CHECK_EQ(Reports() - r0, 0u);  // the destructor drained it before the walk
	G3_CHECK_EQ(dev.arms(), 1);
	G3_CHECK(dev.ledger().ReadGuardTerm() == SubmissionLedger::GuardTerm::Observed);
	G3_CHECK_EQ(inflight::LiveRecordsForTest(), live0);
	G3_CHECK_EQ(SyncSiteEarlyReturn(dev, false), 0);
	G3_CHECK(dev.ledger().ReadGuardTerm() == SubmissionLedger::GuardTerm::Observed);
}
G3_CELL("G3/submission/early-return", CellSubmissionEarlyReturn);

void CellSubmissionMoveAssignDrains() {
	FakeDevice dev(/*auto_complete=*/false, /*complete_on_arm=*/true);
	GpuSubmission a = dev.Submit();
	GpuSubmission b = dev.Submit();
	a = std::move(b);
	G3_CHECK_EQ(dev.arms(), 1);
	G3_CHECK(dev.ledger().observed() >= 1u);
	G3_CHECK_EQ(a.fence_value(), 2u);
	G3_CHECK(!b.holds());
	GpuSubmission c(std::move(a));
	G3_CHECK(!a.holds());
	G3_CHECK_EQ(c.fence_value(), 2u);
	G3_CHECK_EQ(dev.arms(), 1);
}
G3_CELL("G3/submission/move-assign-drains", CellSubmissionMoveAssignDrains);

GpuSubmission g_kept_submission;
void CellSubmissionFileScope() {
	FakeDevice dev(/*auto_complete=*/true);
	const uint64_t r0 = Reports();
	{
		inflight::GpuApiCallScope call("kept_submission");
		g_kept_submission = dev.Submit();
	}
	G3_CHECK_EQ(Reports() - r0, 1u);
	G3_CHECK(!g_kept_submission.holds());
	G3_CHECK(!dev.ledger().Unobserved());
}
G3_CELL("G3/submission/file-scope", CellSubmissionFileScope);

// ---- the scope -------------------------------------------------------------------------------

void CellScopeOutermostOnly() {
	FakeDevice dev(/*auto_complete=*/true);
	uint64_t r0 = Reports();
	uint64_t at_inner = 0;
	{
		inflight::GpuApiCallScope outer("composite");
		TestToken local = TestToken::Make(dev.Submit(), 0);  // held across the inner call
		{
			inflight::GpuApiCallScope inner("inner_verb");
		}
		at_inner = Reports() - r0;
		G3_CHECK(local.holds());  // the inner return did not drain the composite's local
		local.Wait();
	}
	G3_CHECK_EQ(at_inner, 0u);
	G3_CHECK_EQ(Reports() - r0, 0u);
	// A record left off a handle by the composite is drained at the outer return, not the inner.
	r0 = Reports();
	{
		inflight::GpuApiCallScope outer("composite_stash");
		g_file_scope_token = TestToken::Make(dev.Submit(), 0);
		{
			inflight::GpuApiCallScope inner("inner_verb");
		}
		at_inner = Reports() - r0;
	}
	G3_CHECK_EQ(at_inner, 0u);
	G3_CHECK_EQ(Reports() - r0, 1u);
}
G3_CELL("G3/scope/outermost-only", CellScopeOutermostOnly);

void CellScopePerThread() {
	FakeDevice dev(/*auto_complete=*/true);
	const uint64_t r0 = Reports();
	{
		inflight::GpuApiCallScope call("long_call");
		TestToken local = TestToken::Make(dev.Submit(), 0);
		uint64_t during = 0;
		std::thread poller([&] {
			FakeHandle other;
			(void)Ready(&other, false);  // a poller's call returns beside this thread's live local
			during = Reports() - r0;
		});
		poller.join();
		G3_CHECK_EQ(during, 0u);
		G3_CHECK(local.holds());
		local.Wait();
	}
	G3_CHECK_EQ(Reports() - r0, 0u);
}
G3_CELL("G3/scope/per-thread", CellScopePerThread);

}  // namespace
}  // namespace g3

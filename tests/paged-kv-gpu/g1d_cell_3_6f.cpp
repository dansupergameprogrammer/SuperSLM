// Paged KV GPU slice, plan step G1 -- cell 3.6f (plan.md Sec7; Sec4.4 *The submission ledger*, *The queue gate*,
// *The not-ready poll*). Bracket: [G3]. Carrier rows T-poll and R10-G5 (the poll leg at source, the two
// token mutants, and the ledger-level Wait(v) leg) land here.
//
// THE LEDGER'S FEATURE ORACLE, WITH THE QUEUE GATE HELD, in the allocation-fault build. The gate keeps a
// submission incomplete for as long as the test holds it, so every step reads a list that is certainly
// pending. Every run has 3.6e's wall-clock timeout and watchdog (60 s; t2791 cell_prompt_guard_removal.cpp
// :20/:41), and a timeout fails the cell: behind a held gate an internal reset that runs records, submits
// and waits, so a mutant that lets it run hangs rather than fails. Each run releases the gate in teardown.
//
// Runs (--run=), each its own process:
//   first (default) -- in a process that creates NO POOL. A DEDICATED sequence A is made Submitted by
//       sslm_decode_step_gpu with the gate held. Then, in order:
//         Poll leg:      sslm_gpu_ready(A, block=0) returns not-ready (SSLM_OK, ready 0) and leaves A
//                        Submitted (sslm_gpu_seq_reset(A), which refuses only a Submitted handle, is
//                        SSLM_BUSY);
//         decode leg:    sslm_decode_step_gpu on a second DEDICATED sequence D2 returns SSLM_BUSY and
//                        changes nothing (D2's whole host-visible state, CaptureState, byte-equal);
//         internal-reset leg: RunLayerLoopGpu lets PoolSubmitContractError reach the test;
//         drain leg:     DrainSubmissionsGpu() on a second thread is still blocked after a bounded host
//                        wait of one second (the test's own thread only waits for it, which keeps the
//                        drain inside P1's precondition);
//         the debug layer is clean throughout.
//       After the gate is released: the drain returns (no exception); sslm_gpu_ready(A, block=1) observes
//       completion (ready 1, out_status SSLM_OK); D2's decode proceeds (SSLM_OK); and A's and D2's next
//       tokens byte-equal never-gated twins'. With A and D2 dedicated and no pool in the process, the count
//       terms are silent, and only the ledger term can refuse the decode leg (rev 10, CA8-M1).
//   stranded -- alone in a fresh process. te425's hook HookArmCall(M_SIGNAL, first=1, E_FAIL,
//       HB_FAIL_NO_CALL, n=2) fails Submit()'s Signal and its retry, with the gate held: the call returns
//       SSLM_DEVICE_LOST, the next guarded call returns SSLM_DEVICE_LOST, and an internal-layer reset throws
//       GpuSubmissionStrandedError. A stranded ledger is terminal for the process, so the run is alone in its
//       process, as te425 runs its rule-0 legs. The hook arms only after any trace (there is none here) and
//       first = 1 is the call's first queue Signal (carrier row HOOK-legs). A hook that does not fire twice
//       is INVALID, never a verdict.
//   waitnoarm -- (carrier row R10-G5) the ledger-level leg, in a fresh process with te425's hooks: with the
//       fence at the value, Wait(v) makes no SetEventOnCompletion call. Wait(v) is reached through
//       DrainSubmissionsGpu() (Drain() = Wait(submitted_)). The operand is te425's test-owned call counter
//       g_calls[M_SETEVENT]. Its instrument control: with the gate held the same drain does arm the event
//       (the counter rises), so a counter that never moves cannot pass the subject leg vacuously.
//
// The named mutants, and the leg that separates each (Sec7 3.6f; R10-G5):
//   - a Poll that reports ready before the fence reaches it: the Poll leg;
//   - a Poll that raises observed_ early but reports the truth: the decode leg (D2 proceeds) and the
//     internal-reset leg (no exception; behind the gate the reset's own wait hangs, the watchdog fails it);
//   - a Wait that does the same: the drain leg (the drain returns while the gate is held);
//   - a Drain() that clears without waiting: the drain leg (it returns at once);
//   - the check run after the reset instead of before: a reset under the pending list (debug layer; or the
//     reset's own wait hangs behind the gate);
//   - unsignaled_ never set: the stranded run (the next reset runs, no GpuSubmissionStrandedError);
//   - RunLayerLoopGpuFinish moves from the token before the fence test: the Poll leg (the moved local's
//     draining destructor waits behind the gate; sslm_gpu_ready(A, block=0) hangs, the watchdog fails it);
//   - a finish that empties the token without draining it: after release, sslm_gpu_ready(A) returns
//     SSLM_SEQUENCE_REJECTED (the Submitted-with-no-token exit) instead of observing completion;
//   - Wait(v) arming the event although the fence is at v: the waitnoarm run.
// The operands are statuses, exception types, the drain's blocked state, the debug layer, and (waitnoarm)
// te425's own hook counter; none is a count the ledger produces.
//
// RED NOW (63e7b0e): link-red. Unresolved: superslm_gpu::ArmQueueGateForTest, ReleaseQueueGateForTest,
// DrainSubmissionsGpu (G3). Green at G3. Debug layer on (SSLM_GPU_ENABLE_DEBUG_LAYER=1) for the first run.
//
// Build: the allocation-fault build, linked with te425_cells.cpp compiled with /Dmain=te425_cells_main
// (te425's InstallHooks, te425_cells.cpp:312), as carrier row HOOK-legs allows: "a binary that links
// te425's InstallHooks".
// Run: g1d_cell_3_6f.exe --model=<square artifact: R15> --run=first|stranded|waitnoarm
#define G1D_WITH_TE425_HOOKS 1
#include "g1d_helpers.h"

namespace {

using g1d::Bench;
using g1d::QueueGate;
using g1d::Thrown;

constexpr int kNextTokens = 4;

bool OpenModel(Bench& b) {
	const std::string model = g1d::Opt("model");
	if (model.empty()) {
		SKIP_MSG("3.6f needs --model=PATH (a square artifact: R15, qwen2.5-1.5b-instruct.sslm)");
		return false;
	}
	if (!b.Open(model)) {
		CHECK_MSG(false, "SETUP: %s did not load and map", model.c_str());
		return false;
	}
	return true;
}

bool Hooks() {
	std::string why;
	const bool ok = te425::InstallHooks(&why);
	CHECK_MSG(ok, "SETUP (INVALID): te425 hooks did not install: %s", why.c_str());
	return ok;
}

std::vector<int32_t> TwinTokens(Bench& b, int32_t embed, int32_t next) {
	SslmGpuSequenceHandle* t = b.Dedicated(64);
	if (!t || g1d::SubmitDecode(b, t, embed) != SSLM_OK || g1d::Ready(b.ctx, t, 1).First() != SSLM_OK) return {};
	return g1d::NextTokens(b, t, next, kNextTokens);
}

void RunFirst() {
	Bench b;
	if (!OpenModel(b)) return;
	g1d::InternalLoop il;
	if (!il.Build(b.fx.view, b.fx.has_qk_norm)) {
		CHECK_MSG(false, "SETUP: the internal-layer fixture did not build");
		b.Close();
		return;
	}
	const int32_t t0 = b.fx.SomeToken(), t1 = b.fx.OtherToken();
	SslmGpuSequenceHandle* A = b.Dedicated(64);
	SslmGpuSequenceHandle* D2 = b.Dedicated(64);
	const std::vector<int32_t> refA = TwinTokens(b, t0, t1);
	const std::vector<int32_t> refD = TwinTokens(b, t1, t0);
	CHECK_MSG(A && D2 && refA.size() == kNextTokens && refD.size() == kNextTokens, "SETUP: sequences and twins");
	if (!(A && D2 && refA.size() == kNextTokens && refD.size() == kNextTokens) || !g1d::RequireDebugLayer()) {
		b.Close();
		return;
	}
	CHECK(sslm_gpu_seq_embed_token(b.ctx, D2, t1) == SSLM_OK);  // before the gate: embed is not guarded
	const SeqState d2_before = CaptureState(D2);
	(void)g1d::DebugMessages();

	std::atomic<bool> drain_done{false};
	Thrown drain_thrown = Thrown::None;
	std::thread drainer;
	{
		QueueGate gate;
		gate.Arm();
		g1d::SetPhase("first: A's decode with the gate held");
		const SslmGpuStatus sub = g1d::SubmitDecode(b, A, t0);
		CHECK_MSG(sub == SSLM_OK, "SETUP: A's decode under the held gate returned %s", StatusName(sub));

		g1d::SetPhase("first: Poll leg");
		const g1d::ReadyOut p = g1d::Ready(b.ctx, A, 0);
		std::printf("    Poll leg: ret=%s ready=%d\n", StatusName(p.ret), p.ready);
		CHECK_MSG(p.ret == SSLM_OK && p.ready == 0,
		          "Poll leg: sslm_gpu_ready(A, block=0) behind the held gate returned ret=%s ready=%d, want SSLM_OK and "
		          "not ready",
		          StatusName(p.ret), p.ready);
		const SslmGpuStatus rs = sslm_gpu_seq_reset(b.ctx, A);
		CHECK_MSG(rs == SSLM_BUSY, "Poll leg: A is not left Submitted (sslm_gpu_seq_reset(A) returned %s, want SSLM_BUSY)",
		          StatusName(rs));

		g1d::SetPhase("first: decode leg");
		const SslmGpuStatus d = sslm_decode_step_gpu(b.ctx, D2, nullptr, b.FullDepthBudget());
		std::printf("    decode leg: %s\n", StatusName(d));
		CHECK_MSG(d == SSLM_BUSY,
		          "decode leg: a second dedicated sequence's decode with A's submission unobserved returned %s, want "
		          "SSLM_BUSY (the ledger term; no pool exists, so no count term applies)",
		          StatusName(d));
		CHECK_MSG(CaptureState(D2) == d2_before, "decode leg: the refused decode changed D2's state");

		g1d::SetPhase("first: internal-reset leg");
		std::string what;
		const Thrown t = g1d::CallInternal([&il] { (void)il.RunSync(); }, &what);
		std::printf("    internal-reset leg: %s %s\n", g1d::ThrownName(t), what.c_str());
		CHECK_MSG(t == Thrown::PoolSubmitContract,
		          "internal-reset leg: RunLayerLoopGpu let %s reach the test, want PoolSubmitContractError",
		          g1d::ThrownName(t));

		g1d::SetPhase("first: drain leg");
		drainer = std::thread([&drain_done, &drain_thrown] {
			drain_thrown = g1d::CallInternal([] { superslm_gpu::DrainSubmissionsGpu(); });
			drain_done.store(true);
		});
		std::this_thread::sleep_for(std::chrono::seconds(1));
		const bool blocked = !drain_done.load();
		std::printf("    drain leg: %s after 1 s\n", blocked ? "still blocked" : "RETURNED");
		CHECK_MSG(blocked, "drain leg: DrainSubmissionsGpu() returned within one second with the gate held");

		const size_t msgs = g1d::DebugMessages();
		CHECK_MSG(msgs == 0, "first: %zu debug-layer message(s) while the gate was held", msgs);
		g1d::SetPhase("first: release");
	}  // the gate is released here
	g1d::SetPhase("first: the drain returns after release");
	drainer.join();
	CHECK_MSG(drain_done.load() && drain_thrown == Thrown::None, "drain leg: after release the drain ended with %s",
	          g1d::ThrownName(drain_thrown));

	g1d::SetPhase("first: A observed, D2 proceeds");
	const g1d::ReadyOut r = g1d::Ready(b.ctx, A, 1);
	CHECK_MSG(r.ret == SSLM_OK && r.ready == 1 && r.out == SSLM_OK,
	          "after release: sslm_gpu_ready(A, block=1) ret=%s ready=%d out=%s, want completion observed (a finish that "
	          "emptied the token without draining it reads SSLM_SEQUENCE_REJECTED here)",
	          StatusName(r.ret), r.ready, StatusName(r.out));
	const SslmGpuStatus d2 = sslm_decode_step_gpu(b.ctx, D2, nullptr, b.FullDepthBudget());
	CHECK_MSG(d2 == SSLM_OK, "after release: D2's decode returned %s, want SSLM_OK", StatusName(d2));
	if (d2 == SSLM_OK) CHECK(g1d::Ready(b.ctx, D2, 1).First() == SSLM_OK);
	const std::vector<int32_t> ga = g1d::NextTokens(b, A, t1, kNextTokens);
	const std::vector<int32_t> gd = g1d::NextTokens(b, D2, t0, kNextTokens);
	CHECK_MSG(ga == refA, "A's next tokens [%s] differ from its twin's [%s]", g1d::Join(ga).c_str(), g1d::Join(refA).c_str());
	CHECK_MSG(gd == refD, "D2's next tokens [%s] differ from its twin's [%s]", g1d::Join(gd).c_str(), g1d::Join(refD).c_str());
	const size_t msgs = g1d::DebugMessages();
	CHECK_MSG(msgs == 0, "first: %zu debug-layer message(s) after release", msgs);
	b.Close();
}

void RunStranded() {
	Bench b;
	if (!OpenModel(b) || !Hooks()) {
		b.Close();
		return;
	}
	g1d::InternalLoop il;
	const bool have_il = il.Build(b.fx.view, b.fx.has_qk_norm);
	CHECK_MSG(have_il, "SETUP: the internal-layer fixture did not build");
	SslmGpuSequenceHandle* A = b.Dedicated(64);
	SslmGpuSequenceHandle* D2 = b.Dedicated(64);
	CHECK_MSG(A && D2, "SETUP: two dedicated sequences");
	if (!A || !D2 || !have_il) {
		b.Close();
		return;
	}
	const int32_t t0 = b.fx.SomeToken();
	CHECK(sslm_gpu_seq_embed_token(b.ctx, A, t0) == SSLM_OK);
	CHECK(sslm_gpu_seq_embed_token(b.ctx, D2, t0) == SSLM_OK);
	QueueGate gate;
	gate.Arm();
	g1d::SetPhase("stranded: both of Submit()'s Signals fail");
	te425::HookArmCall(te425::M_SIGNAL, 1, E_FAIL, te425::HB_FAIL_NO_CALL, 2);
	const SslmGpuStatus own = sslm_decode_step_gpu(b.ctx, A, nullptr, b.FullDepthBudget());
	const uint32_t fired = te425::g_arm.fired.load();
	te425::HookDisarm();
	std::printf("    stranding call: %s (Signal hook fired %u)\n", StatusName(own), fired);
	CHECK_MSG(fired == 2, "INVALID: the Signal hook fired %u time(s), want 2 (Submit()'s Signal and its retry)", fired);
	CHECK_MSG(own == SSLM_DEVICE_LOST, "stranded: the call that strands returned %s, want SSLM_DEVICE_LOST", StatusName(own));
	g1d::SetPhase("stranded: the next guarded call");
	const SslmGpuStatus next = sslm_decode_step_gpu(b.ctx, D2, nullptr, b.FullDepthBudget());
	CHECK_MSG(next == SSLM_DEVICE_LOST,
	          "stranded: the next guarded call returned %s, want SSLM_DEVICE_LOST (terminal; never SSLM_BUSY)",
	          StatusName(next));
	g1d::SetPhase("stranded: an internal-layer reset");
	std::string what;
	const Thrown t = g1d::CallInternal([&il] { (void)il.RunSync(); }, &what);
	CHECK_MSG(t == Thrown::Stranded, "stranded: an internal-layer reset let %s reach the test, want "
	                                 "GpuSubmissionStrandedError (%s)",
	          g1d::ThrownName(t), what.c_str());
	gate.Release();
	b.Close();
}

void RunWaitNoArm() {
	Bench b;
	if (!OpenModel(b) || !Hooks()) {
		b.Close();
		return;
	}
	SslmGpuSequenceHandle* A = b.Dedicated(64);
	CHECK_MSG(A != nullptr, "SETUP: a dedicated sequence");
	if (!A) {
		b.Close();
		return;
	}
	const int32_t t0 = b.fx.SomeToken();
	// Instrument control: with the fence below the value, the drain's Wait arms the event, and te425's
	// counter sees it.
	g1d::SetPhase("waitnoarm: control, the gated drain arms the event");
	{
		QueueGate gate;
		gate.Arm();
		CHECK(g1d::SubmitDecode(b, A, t0) == SSLM_OK);
		const uint32_t c0 = te425::g_calls[te425::M_SETEVENT].load();
		std::atomic<bool> done{false};
		std::thread drainer([&done] {
			(void)g1d::CallInternal([] { superslm_gpu::DrainSubmissionsGpu(); });
			done.store(true);
		});
		bool armed = false;
		for (int i = 0; i < 50 && !armed; ++i) {
			std::this_thread::sleep_for(std::chrono::milliseconds(100));
			armed = te425::g_calls[te425::M_SETEVENT].load() > c0;
		}
		CHECK_MSG(armed, "INVALID (instrument control): a drain blocked behind the held gate made no "
		                 "SetEventOnCompletion call that te425's hook counted");
		gate.Release();
		drainer.join();
		CHECK(done.load());
	}
	CHECK(g1d::Ready(b.ctx, A, 1).First() == SSLM_OK);
	g1d::SetPhase("waitnoarm: Wait(v) with the fence at v");
	const uint32_t c1 = te425::g_calls[te425::M_SETEVENT].load();
	std::string what;
	const Thrown t = g1d::CallInternal([] { superslm_gpu::DrainSubmissionsGpu(); }, &what);
	const uint32_t c2 = te425::g_calls[te425::M_SETEVENT].load();
	std::printf("    SetEventOnCompletion calls during a drain with the fence at the value: %u\n", c2 - c1);
	CHECK_MSG(t == Thrown::None, "waitnoarm: the drain threw %s %s", g1d::ThrownName(t), what.c_str());
	CHECK_MSG(c2 == c1, "waitnoarm: Wait(v) with the fence already at v made %u SetEventOnCompletion call(s), want 0",
	          c2 - c1);
	b.Close();
}

}  // namespace

int main(int argc, char** argv) {
	g1d::ParseOpts(argc, argv);
	g1d::StartWatchdog(60);
	const std::string run = g1d::Opt("run", "first");
	std::printf("cell 3.6f run=%s\n", run.c_str());
	if (run == "first") RunFirst();
	else if (run == "stranded") RunStranded();
	else if (run == "waitnoarm") RunWaitNoArm();
	else CHECK_MSG(false, "unknown --run=%s (first|stranded|waitnoarm)", run.c_str());
	g1d::PhaseDone();
	return FinishSuite("g1d_cell_3_6f");
}

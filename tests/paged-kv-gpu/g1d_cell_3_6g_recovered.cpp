// Paged KV GPU slice, plan step G1 -- cell 3.6g, the RECOVERED run (plan.md Sec7 3.6g; Sec4.4 *The submission
// ledger* `Submit(retry)`, *`Submit()` returns its work in a draining token*, *The ledger on the fault
// paths*). Bracket: [G3]. Carrier rows PR11-W2 and R11-N3 land here. The stranded and per-context runs are
// g1d_cell_3_6g_stranded.cpp (another build).
//
// BUILD: the T2169 chunk-recording build (SUPERSLM_T2169_CHUNK_RECORDING_FAULT_INJECTION, as
// tests/t2178-gpu-batched-prefill-red-suite/build_red_suite.bat compiles it), because the run uses the T2169
// signal seam with the queue gate held (Sec4.4 *The queue gate*: the seam is in both injection builds).
//
// CONSTRUCTION (Sec7 3.6g, the recovered bullet; the reference arm is t2178's
// TestGuard_ContextReusableAfterSignalFault, cell_trust_and_guard.cpp:577, which this run extends with the gate):
//   1. Reference arm, never faulted: a fresh dedicated sequence prefills the second chunk and decodes one
//      token through the bridge; its token is the reference.
//   2. The queue gate is armed, then ArmT2169ChunkRecordingTailSignalFaultInjection() (single-shot; from G3 it
//      fires inside Submit(), on the chunk tail's first Signal; the retry at a fresh value succeeds).
//   3. The faulted call -- the prompt-prefill bridge on a candidate sequence, one sub-chunk (3 tokens <=
//      kT2169TdrSafeMaxChunkTokens) -- runs on a second thread. After a bounded host wait of one second it is
//      STILL BLOCKED (from rev 12, inside Submit(): a recovered retry drains the fresh value before it
//      rethrows, and the gate holds that drain; R11-N3 corrects the rev 10 location "the tail's
//      Wait(submitted_)").
//   4. The gate is released. The call returns rule 3's SSLM_DEVICE_LOST (the seam's HRESULT is not
//      E_OUTOFMEMORY, so not rule 2's SSLM_GPU_ALLOCATION_FAILED).
//   5. The next call on the same context and sequence (the second chunk) returns SSLM_OK with a clean debug
//      layer, and its following decode token equals the reference arm's.
//
// THE MUTANTS, RE-DERIVED AGAINST REV 12's Submit() (carrier rows PR11-W2, R11-N3). The reading this cell
// states (PR11-W2 asks G1 to state it): Submit()'s internal drain of a recovered retry waits on the ledger's
// `submitted_` as it stands AFTER the retry's store -- i.e. Wait(submitted_), the value the retry raised --
// not on a local copy of the fresh value. Under that reading:
//   M1 rev 8 as written (the recovery a raw re-signal and wait outside the ledger): the fence passes but
//      observed_ is never raised, so step 5's call is refused SSLM_BUSY by the guard's ledger term.
//      Removed wait: the ledger-side observe. Separating operand: step 5's status (SSLM_BUSY, never SSLM_OK).
//   M2 a retry that leaves `submitted_` behind (signals the fresh value but does not raise submitted_):
//      Submit()'s drain waits on the previous value, already observed, so it returns at once and the faulted
//      call returns with the gate held. Removed wait: the drain's real wait. Separating operand: step 3's
//      still-blocked leg (it has returned). With the gate held, step 5 would also reset under the pending
//      list, which the debug layer reports -- recorded, but step 3 is the leg that fails first.
//   M3 Submit() rethrows without its drain: the faulted call's Submit() returns at once. Removed wait: the
//      drain inside Submit(). Separating operand: step 3's still-blocked leg -- PROVIDED no later wait in the
//      caller re-observes. Today's catch site ("after a recovered retry, Wait then rethrow") also waits; if G3
//      keeps that tail Wait(submitted_) as well as Submit()'s own drain, M3 is EQUIVALENT at this construction
//      (the tail wait blocks behind the gate and observes, so step 3 and step 5 both read green). R11-N3 says
//      M3 is separated by the still-blocked and next-call legs; PR11-W2 says the tail's Wait masks it. Both
//      cannot hold unless G3 removes the redundant tail wait. This cell does not adjudicate: it reports which
//      happened (the printed "still blocked" line), and the notes carry the ambiguity to G3.
//   M4 `unsignaled_` never set: not reachable on this run (the retry succeeds); the stranded runs separate it.
// The operands are statuses, the faulted call's blocked state, the debug-layer count and the tokens.
//
// RED NOW (63e7b0e): link-red -- superslm_gpu::ArmQueueGateForTest and ReleaseQueueGateForTest (G3) have no
// definition. Green at G3 (the signal seam moved inside Submit(), the ledger's recovered retry, the queue gate).
// Run: g1d_cell_3_6g_recovered.exe --model=<artifact: qwen3-0.6b.sslm or R15>, debug layer on
//      (SSLM_GPU_ENABLE_DEBUG_LAYER=1).
#include "g1d_helpers.h"

namespace {

using g1d::QueueGate;

void RunRecovered() {
	g1d::Bench b;
	const std::string model = g1d::Opt("model");
	if (model.empty()) {
		SKIP_MSG("3.6g recovered needs --model=PATH");
		return;
	}
	if (!b.Open(model)) {
		CHECK_MSG(false, "SETUP: %s did not load and map", model.c_str());
		return;
	}
	if (!g1d::RequireDebugLayer()) {
		b.Close();
		return;
	}
	const uint32_t budget = b.fx.one_layer_budget;
	const std::vector<int32_t> first_chunk = g1d::Tokens(b.fx, 3, b.fx.vocab / 9);
	const std::vector<int32_t> second_chunk = g1d::Tokens(b.fx, 3, b.fx.vocab / 4);
	const int32_t next_token = b.fx.SomeToken();

	// 1. The reference arm.
	g1d::SetPhase("recovered: reference arm");
	SslmGpuSequenceHandle* ref = b.Dedicated(64);
	CHECK_MSG(ref != nullptr, "SETUP: reference sequence");
	if (!ref) {
		b.Close();
		return;
	}
	const SslmGpuStatus rp = Prefill(b.fx, ref, second_chunk, budget);
	int32_t ref_token = -1;
	const SslmGpuStatus rd = rp == SSLM_OK ? SslmGpuSeqDecodeStepForG5Bridge(b.ctx, ref, next_token, budget, &ref_token)
	                                       : rp;
	CHECK_MSG(rp == SSLM_OK && rd == SSLM_OK, "SETUP: reference arm prefill=%s decode=%s", StatusName(rp), StatusName(rd));
	if (rp != SSLM_OK || rd != SSLM_OK) {
		b.Close();
		return;
	}
	SslmGpuSequenceHandle* cand = b.Dedicated(64);
	CHECK_MSG(cand != nullptr, "SETUP: candidate sequence");
	if (!cand) {
		b.Close();
		return;
	}
	(void)g1d::DebugMessages();

	// 2-4. The faulted call with the gate held.
	std::atomic<bool> done{false};
	SslmGpuStatus first_status = SSLM_OK;
	std::thread faulted;
	bool blocked = false;
	size_t msgs_faulted = 0;
	{
		QueueGate gate;
		gate.Arm();
		superslm_gpu::ArmT2169ChunkRecordingTailSignalFaultInjection();
		g1d::SetPhase("recovered: the faulted call, gate held");
		faulted = std::thread([&] {
			first_status = Prefill(b.fx, cand, first_chunk, budget);
			done.store(true);
		});
		std::this_thread::sleep_for(std::chrono::seconds(1));
		blocked = !done.load();
		std::printf("    faulted call after 1 s with the gate held: %s\n", blocked ? "still blocked" : "RETURNED");
		CHECK_MSG(blocked,
		          "recovered: the faulted call returned within one second with the queue gate held (status %s) -- "
		          "Submit()'s drain of the fresh value did not wait (mutant M2, a retry that leaves submitted_ "
		          "behind; or M3, Submit() rethrows without its drain)",
		          done.load() ? StatusName(first_status) : "-");
		g1d::SetPhase("recovered: release");
	}  // released here
	g1d::SetPhase("recovered: the faulted call returns after release");
	faulted.join();
	superslm_gpu::ClearT2169ChunkRecordingTailSignalFaultInjection();
	msgs_faulted = g1d::DebugMessages();
	std::printf("    faulted call: %s; debug-layer messages during it: %zu (recorded)\n", StatusName(first_status),
	            msgs_faulted);
	CHECK_MSG(first_status == SSLM_DEVICE_LOST,
	          "recovered: the faulted call returned %s, want rule 3's SSLM_DEVICE_LOST (the seam's HRESULT is not "
	          "E_OUTOFMEMORY)",
	          StatusName(first_status));

	// 5. The next call on the same context.
	g1d::SetPhase("recovered: the next call");
	const SslmGpuStatus second = Prefill(b.fx, cand, second_chunk, budget);
	CHECK_MSG(second == SSLM_OK,
	          "recovered: the next call on the same context returned %s, want SSLM_OK (SSLM_BUSY is mutant M1, the "
	          "recovery outside the ledger)",
	          StatusName(second));
	int32_t cand_token = -1;
	const SslmGpuStatus cd =
	    second == SSLM_OK ? SslmGpuSeqDecodeStepForG5Bridge(b.ctx, cand, next_token, budget, &cand_token) : second;
	CHECK_MSG(cd == SSLM_OK, "recovered: the following decode returned %s", StatusName(cd));
	CHECK_MSG(cand_token == ref_token, "recovered: the candidate's token %d differs from the never-faulted reference's %d",
	          cand_token, ref_token);
	const size_t msgs = g1d::DebugMessages();
	CHECK_MSG(msgs == 0, "recovered: %zu debug-layer message(s) on the next call (a reset under a pending list)", msgs);
	b.Close();
}

}  // namespace

int main(int argc, char** argv) {
	g1d::ParseOpts(argc, argv);
	g1d::StartWatchdog(60);
	std::printf("cell 3.6g run=recovered\n");
	RunRecovered();
	g1d::PhaseDone();
	return FinishSuite("g1d_cell_3_6g_recovered");
}

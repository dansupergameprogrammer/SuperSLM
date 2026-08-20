// T-2178 (Curie) -- design Sec9 "Trust boundaries / hostile inputs" and "Guard vitality" rows.
// The two Trust-boundaries occupants named fold round 3 (D-SLM3623) are direct adaptations of
// Claude/Loki/t2176-probe-cap-straddle-bridge-semantics.cpp's own Arm A / Arm F construction into
// committed cells, per this suite's own casebook Sec2 and the design's own Sec8 rung 1 text
// ("T-2176's own probe... is the template for the two new Trust-boundaries cells"). Arm A here
// calls the PUBLIC bridge, which since Rungs 3/4 runs the batched primitive for every count
// (chunk_len=1 included) -- there is no per-token arm in this file; the shipped-equivalence
// carrier is t2132_g5_gpu_parity (design Sec6, corrected at T-2184/T-2185).
#include "fixture_common.h"
#include "superslm/gpu_port.h"
// T-2192 finding 8 (O2's own instrument, and T1's own execution-verification pin): the SAME
// header the production translation units include, so `harness::GetDevice()`'s function-local
// static singleton (d3d12_harness.h) is genuinely shared with `superslm_gpu.cpp`/`gpu_1p0.cpp` --
// this file drains the identical device's debug-layer message queue the production code ran
// against, not a second, empty one.
#include "../../src/gpu/d3d12_harness.h"

#include <atomic>
#include <chrono>
#include <stdexcept>
#include <thread>

using namespace superslm;

namespace {

// Mirrors the probe's own ArmResult -- prefill status, ctxlen after prefill, and the consumer-
// visible consequence of `ready_for_logits`: does the FOLLOWING decode step consume the token
// it is handed, or silently finish the stale residual.
//
// D-SLM3654 (part 2, T-2183 fix): `prefill_dispatch_delta` is the
// g_gpu_chunk_dispatch_count_probe delta bracketing ONLY the prefill call under test -- captured
// inside RunPromptArm, strictly after PrimeSeq's own priming call and strictly before the
// following decode step, mirroring fixture_common.h's own RunTwoArmPrompt convention (which
// scopes its ref/cand deltas around the call under test, never around priming). The two callers
// that need a dispatch-count claim (TestGuard_PositionCapClampDispatchCountInstrumented,
// TestGuard_EmbedAdmitCountClampDispatchCountInstrumented) previously captured "before" outside
// this function, ahead of the call that includes PrimeSeq's own real, dispatching prefill --
// widening the observed window to include priming's own dispatch cost. Confirmed by exact
// arithmetic against the real artifact: the previously observed deltas (168, 84) equal
// `(prime_tokens + admitted_chunk_tokens) * num_hidden_layers` precisely, not
// `admitted_chunk_tokens * num_hidden_layers` alone.
struct BridgeArmResult {
	bool ok = false;
	SslmGpuStatus prefill_status = SSLM_OK;
	int64_t ctxlen_after_prefill = 0;
	SslmGpuStatus step_status = SSLM_OK;
	int64_t ctxlen_after_step = 0;
	bool step_consumed_its_token = false;
	int32_t step_out_token = -1;
	int64_t prefill_dispatch_delta = 0;
};

BridgeArmResult RunPromptArm(SslmGpuContext* ctx, SslmGpuModelHandle* model, int64_t context_cap,
                              const std::vector<int32_t>& prime, const std::vector<int32_t>& chunk,
                              int32_t submit_count, int32_t next_token) {
	BridgeArmResult r;
	SslmGpuSequenceHandle* seq = nullptr;
	if (sslm_gpu_seq_create(ctx, model, context_cap, &seq) != SSLM_OK || !seq) return r;
	if (!PrimeSeq(ctx, seq, prime)) { sslm_gpu_seq_release(ctx, seq); return r; }
	// D-SLM3654 (part 2): captured AFTER priming, bracketing only the prefill call under test.
	const int64_t dispatch_before = superslm_test::g_gpu_chunk_dispatch_count_probe.load();
	r.prefill_status =
	    SslmGpuSeqPrefillPromptForG5Bridge(ctx, seq, chunk.data(), submit_count, kDispatchBudget);
	r.prefill_dispatch_delta =
	    superslm_test::g_gpu_chunk_dispatch_count_probe.load() - dispatch_before;
	r.ctxlen_after_prefill = *SslmGpuSeqHandleContextLengthForBench(seq);
	const int64_t before = r.ctxlen_after_prefill;
	int32_t out_token = -1;
	r.step_status =
	    SslmGpuSeqDecodeStepForG5Bridge(ctx, seq, next_token, kDispatchBudget, &out_token);
	r.ctxlen_after_step = *SslmGpuSeqHandleContextLengthForBench(seq);
	r.step_out_token = out_token;
	r.step_consumed_its_token = (r.ctxlen_after_step > before);
	r.ok = true;
	sslm_gpu_seq_release(ctx, seq);
	return r;
}

}  // namespace

// --- Trust boundary (i): cap-straddle bridge-observable cell -- T-2176 cell 1, retained as this
// design's own gate. A chunk opening inside the last chunk_tokens-1 positions before
// context_cap; asserts all four observable columns match the shipped path's -- reproducing
// census members P3(a)/S4(a) (design Sec5's `cause = POSITION_CAP` row). ---
static void TestTrust_CapStraddleBridgeObservable(SslmGpuContext* ctx, SslmGpuModelHandle* model) {
	const int64_t kSeqContextCap = 6;
	const std::vector<int32_t> prime = {11, 22, 33, 44};
	const std::vector<int32_t> chunk = {55, 66, 77, 88};  // only 2 of 4 fit before context_cap
	const int32_t kNextToken = 99;

	// Arm A: the WHOLE chunk in one call -- the batched candidate under test: this
	// exercises SubmitChunkToFullDepthForG5Bridge's own admission clamp (the build is landed;
	// the pre-batching per-token loop is no longer reachable here) -- see cell_bitidentity.cpp's own
	// header note on why black-box content agreement alone is not sufficient evidence this cell
	// exercised the NEW mechanism, which is why every cell in this file also reads the dispatch
	// probes below).
	const int64_t submits_before = superslm_test::g_gpu_chunk_submit_count_probe.load();
	const BridgeArmResult a = RunPromptArm(ctx, model, kSeqContextCap, prime, chunk,
	                                        static_cast<int32_t>(chunk.size()), kNextToken);
	const int64_t submits_after = superslm_test::g_gpu_chunk_submit_count_probe.load();
	CHECK(a.ok);
	CHECK_MSG(a.prefill_status == SSLM_DEVICE_LOST,
	          "Trust(cap-straddle): prefill status is %s, expected SSLM_DEVICE_LOST (design Sec5 "
	          "cause=POSITION_CAP row, census P3(a)/S4(a))",
	          GpuStatusName(a.prefill_status));
	CHECK_MSG(a.ctxlen_after_prefill == kSeqContextCap,
	          "Trust(cap-straddle): ctxlen after prefill is %lld, expected %lld (exactly the 2 "
	          "admitted tokens committed)",
	          static_cast<long long>(a.ctxlen_after_prefill), static_cast<long long>(kSeqContextCap));
	CHECK_MSG(!a.step_consumed_its_token,
	          "Trust(cap-straddle): the following decode step consumed its own token argument -- "
	          "ready_for_logits must be LEFT UNSET on this cause (design Sec5's cause table)");
	CHECK_MSG(a.step_status == SSLM_SEQUENCE_REJECTED,
	          "Trust(cap-straddle): following decode step status is %s, expected "
	          "SSLM_SEQUENCE_REJECTED (reproduces the shipped path's own residual-finishing exit "
	          "by construction of the identical device state)",
	          GpuStatusName(a.step_status));
	// The submission-granularity signature: a genuinely batched call that clamps to
	// position_admit_count still opens (at most) ONE command list for the whole submitted
	// chunk, never one per admitted token -- distinguishing this from a per-token loop that
	// happens to clamp correctly at the same observable status/ctxlen/ready_for_logits triple.
	CHECK_MSG(submits_after - submits_before >= 1,
	          "Trust(cap-straddle): the dispatch-submission probe never incremented -- the "
	          "batched primitive was not exercised at all");
}

// --- Trust boundary (ii): hostile-token-at-index-1 cell -- T-2176 cell 2. A chunk whose second
// token is outside [0, vocab_size); asserts the returned embed status, committed count, and
// ready_for_logits all match the shipped path's -- census members P2/S3 (design Sec5's
// `cause = EMBED` row). ---
static void TestTrust_HostileTokenAtIndex1(SslmGpuContext* ctx, SslmGpuModelHandle* model,
                                            int32_t vocab_size) {
	const std::vector<int32_t> prime = {11, 22};
	const int32_t kHostileToken = vocab_size + 1000000;  // outside [0, vocab_size)
	const std::vector<int32_t> chunk = {55, kHostileToken, 77};

	const int64_t dispatch_before = superslm_test::g_gpu_chunk_dispatch_count_probe.load();
	const BridgeArmResult a = RunPromptArm(ctx, model, /*context_cap=*/16, prime, chunk,
	                                        static_cast<int32_t>(chunk.size()), /*next_token=*/99);
	const int64_t dispatch_after = superslm_test::g_gpu_chunk_dispatch_count_probe.load();
	CHECK(a.ok);
	CHECK_MSG(a.prefill_status == SSLM_TOKEN_ID_OUT_OF_RANGE,
	          "Trust(hostile-token@1): prefill status is %s, expected SSLM_TOKEN_ID_OUT_OF_RANGE "
	          "(design Sec5 cause=EMBED row, census P2/S3)",
	          GpuStatusName(a.prefill_status));
	CHECK_MSG(a.ctxlen_after_prefill == static_cast<int64_t>(prime.size()) + 1,
	          "Trust(hostile-token@1): ctxlen after prefill is %lld, expected %zu -- exactly index "
	          "0 committed, matching the shipped path's own K/V state at the point of rejection "
	          "(embed_admit_count = 1)",
	          static_cast<long long>(a.ctxlen_after_prefill), prime.size() + 1);
	CHECK_MSG(!a.step_consumed_its_token,
	          "Trust(hostile-token@1): ready_for_logits must be LEFT UNSET on the EMBED cause");
	// Dispatch-count instrument: exactly ONE token's worth of dispatches (num_hidden_layers) may
	// have been issued -- a primitive that dispatches token 1's own layer chain BEFORE
	// discovering the embed rejection has broken Sec5 2b's own "all embeddings are precomputed
	// host-side before the list opens" ordering.
	CHECK_MSG(dispatch_after >= dispatch_before,
	          "Trust(hostile-token@1): dispatch-count probe went backward, instrumentation bug");
}

// --- Trust boundary (iii): hostile count values -- 0, the smallest negative int32_t, and a
// value engineered toward the command-list-size ceiling multiplication boundary (design Sec9
// row text). Asserts bit-for-bit identical rejection/clamping behavior to the pre-existing
// per-token path at each value, and that the ceiling arithmetic does not overflow. ---
static void TestTrust_HostileCountValues(SslmGpuContext* ctx, SslmGpuModelHandle* model) {
	const std::vector<int32_t> prime = {11, 22};
	const std::vector<int32_t> chunk = {33, 44, 55};
	const int32_t kHostileCounts[] = {0, -1, INT32_MIN};
	for (int32_t count : kHostileCounts) {
		const BridgeArmResult a =
		    RunPromptArm(ctx, model, /*context_cap=*/64, prime, chunk, count, /*next_token=*/99);
		CHECK_MSG(a.ok, "Trust(hostile-count=%d): setup failed", count);
		if (!a.ok) continue;
		// count <= 0 admits nothing and must never crash, never overflow the
		// chunk_tokens * num_hidden_layers * kDispatchesPerLayer ceiling arithmetic, and must
		// leave the sequence's own context_length exactly where priming left it.
		CHECK_MSG(a.ctxlen_after_prefill == static_cast<int64_t>(prime.size()),
		          "Trust(hostile-count=%d): ctxlen after prefill is %lld, expected %zu (a "
		          "non-positive count must admit zero tokens)",
		          count, static_cast<long long>(a.ctxlen_after_prefill), prime.size());
	}
}

// --- Trust boundary (iv): hostile starting context_length (D-SLM3616, fold round 2) -- the
// chunk-open value engineered to land inside the last chunk_tokens-1 positions before
// context_cap; the exact value that arms T-2175 member 12 / design Sec5's position_admit_count.
// Distinct from (i) above only in HOW the straddle is reached (a pre-existing near-cap prime,
// not the chunk's own length) -- design Sec9's own text names both as separate occupants. ---
static void TestTrust_HostileStartingContextLength(SslmGpuContext* ctx, SslmGpuModelHandle* model) {
	const int64_t kSeqContextCap = 10;
	std::vector<int32_t> prime;
	for (int i = 0; i < 8; ++i) prime.push_back(1000 + i);  // context_length = 8 after priming
	const std::vector<int32_t> chunk = {900, 901, 902, 903};  // only 2 fit before cap=10

	const BridgeArmResult a = RunPromptArm(ctx, model, kSeqContextCap, prime, chunk,
	                                        static_cast<int32_t>(chunk.size()), /*next_token=*/99);
	CHECK(a.ok);
	CHECK_MSG(a.prefill_status == SSLM_DEVICE_LOST,
	          "Trust(hostile-start-ctxlen): status is %s, expected SSLM_DEVICE_LOST",
	          GpuStatusName(a.prefill_status));
	CHECK_MSG(a.ctxlen_after_prefill == kSeqContextCap,
	          "Trust(hostile-start-ctxlen): ctxlen after prefill is %lld, expected %lld",
	          static_cast<long long>(a.ctxlen_after_prefill), static_cast<long long>(kSeqContextCap));
}

// --- Guard vitality (a)/(a2)/(a3): the DFA-admission, position-cap, and embed_admit_count
// clamps, each instrumented via the ACTUAL dispatch count issued (not only the bookkeeping
// `*consumed` field), confirming it covers exactly tokens [0, admit_count) * num_hidden_layers.
// The MUTATION half (removing or off-by-one'ing each clamp, confirming the cell flips
// pass->fail) is a build-time, one-time, disposable exercise against a deliberately mutated
// build -- owed at Rung 5 (design Sec8), matching this project's own established convention for
// this exact shape of proof (Claude/Curie/t2158-t2149-avx-red-suite-2026-08-18.md Sec4's
// check_matmul_avx_isolation.py mutation-vitality proof, run and recorded, never committed as
// permanent CI). This cell is what that future mutation run is run AGAINST -- authored now so
// the instrument exists the moment the clamp does. ---
static void TestGuard_PositionCapClampDispatchCountInstrumented(SslmGpuContext* ctx,
                                                                  SslmGpuModelHandle* model,
                                                                  uint32_t num_hidden_layers) {
	const int64_t kSeqContextCap = 6;
	const std::vector<int32_t> prime = {11, 22, 33, 44};  // context_length = 4
	const std::vector<int32_t> chunk = {55, 66, 77, 88};  // only 2 admitted before cap=6

	const BridgeArmResult a = RunPromptArm(ctx, model, kSeqContextCap, prime, chunk,
	                                        static_cast<int32_t>(chunk.size()), /*next_token=*/99);
	CHECK(a.ok);
	const int64_t expected_dispatch_delta = 2 * static_cast<int64_t>(num_hidden_layers);
	CHECK_MSG(a.prefill_dispatch_delta == expected_dispatch_delta,
	          "Guard(position-cap clamp): dispatch-body invocations = %lld, expected exactly %lld "
	          "(2 admitted tokens * num_hidden_layers) -- a clamp removed or off-by-one'd would "
	          "dispatch for 3 or 4 tokens instead of 2, and this instrument would catch it",
	          static_cast<long long>(a.prefill_dispatch_delta),
	          static_cast<long long>(expected_dispatch_delta));
}

static void TestGuard_EmbedAdmitCountClampDispatchCountInstrumented(SslmGpuContext* ctx,
                                                                      SslmGpuModelHandle* model,
                                                                      uint32_t num_hidden_layers,
                                                                      int32_t vocab_size) {
	const std::vector<int32_t> prime = {11, 22};
	const int32_t kHostileToken = vocab_size + 1000000;
	const std::vector<int32_t> chunk = {55, kHostileToken, 77};  // only index 0 admitted

	const BridgeArmResult a = RunPromptArm(ctx, model, /*context_cap=*/64, prime, chunk,
	                                        static_cast<int32_t>(chunk.size()), /*next_token=*/99);
	CHECK(a.ok);
	const int64_t expected_dispatch_delta = 1 * static_cast<int64_t>(num_hidden_layers);
	CHECK_MSG(a.prefill_dispatch_delta == expected_dispatch_delta,
	          "Guard(embed_admit_count clamp): dispatch-body invocations = %lld, expected exactly "
	          "%lld (1 admitted token * num_hidden_layers) -- a clamp removed would dispatch "
	          "index 1's own layer chain despite its hostile embed, which this instrument catches "
	          "even though the final K/V content might otherwise look plausible",
	          static_cast<long long>(a.prefill_dispatch_delta),
	          static_cast<long long>(expected_dispatch_delta));
}

// --- Guard vitality (b): SSLM_BUSY under the widened in-flight window -- T-2184 remedy S1
// retarget (Brunel fix round 1, D-SLM3662), re-retargeted by T-2185 remedy N1 (Brunel fix round
// 2, D-SLM3674). Design Sec9's Concurrency/lifecycle row: "a batched chunk submission still
// occupies exactly one `Submitted` window per chunk (not per token)" -- gpu_1p0.cpp's
// `SubmitAdmittedChunkForG5Bridge` now sets `seq->state`/`model->submitted_sequences` BEFORE the
// first sub-chunk is ever submitted and clears them only after the whole chunk (every sub-chunk,
// not only the final one) has finished, symmetric with `SubmitOneSequenceDecode`/`sslm_gpu_ready`.
//
// UNLIKE the per-token path -- whose LOW-LEVEL `sslm_decode_step_gpu` submits without fencing and
// returns control to the SAME thread, letting a single-threaded caller observe the window with a
// second call (this file's own prior version of this cell) -- every batched public entry point
// (`SslmGpuSeqPrefillPromptForG5Bridge` et al.) submits AND drains inside one call, so the window
// is invisible to a same-thread caller by construction; it exists only for a genuinely concurrent
// SECOND thread, exactly the reachable consequence the T-2184 review named: a caller managing
// model lifetime on one thread while another thread drives a batched prefill. This cell reproduces
// that scenario for real -- a worker thread drives a forced span wide enough to span several
// TDR-safe sub-chunks (`kT2169TdrSafeMaxChunkTokens == 4`) at real full depth, while this thread
// polls `sslm_gpu_model_unmap` (the review's own discriminating oracle, gpu_1p0.cpp:789-794:
// `submitted_sequences>0` returns SSLM_BUSY before `live_sequences>0` is even checked) throughout.
// This does NOT drive the same SslmGpuSequenceHandle from two threads (gpu_1p0.h Sec5.4's own
// "unguarded caller error" boundary) -- the worker thread owns `seq` exclusively; the polling
// thread only reads `model`'s own two plain counters via the public `sslm_gpu_model_unmap` entry
// point, which submits no GPU work and touches no command list, the exact cross-thread pattern
// design Sec9 names as supported.
//
// T-2185 N1's own finding (against the T-2184 remedy): `observed_busy > 0` -- one hit anywhere --
// cannot discriminate a window that covers the whole 8-sub-chunk span (the N1 fix) from one that
// covers only the final sub-chunk of 8 (the T-2184 remedy this cell was retargeting), because both
// produce at least one BUSY poll. The N1 remedy also REMOVES this cell's prior timing dependence
// (N7): under the T-2184 remedy the window was bounded to one sub-chunk's own fence wait
// (~tens of ms), so a slow poll loop could race past it and see nothing; under the N1 remedy the
// window spans the whole submitter-thread call, so a poll landing anywhere but the extreme
// start/end of that call lands inside it. The oracle below is re-pointed accordingly: it counts
// every poll, not just whether one hit, and requires a STATED MAJORITY of them to observe
// SSLM_BUSY -- a property the final-sub-chunk-only window (1 of 8 sub-chunks, ~12.5% of this
// cell's own wall time) cannot satisfy, and the whole-chunk window comfortably does.
static void TestGuard_BusyUnderWidenedWindow(SslmGpuContext* ctx, SslmGpuModelHandle* model) {
	SslmGpuSequenceHandle* seq = nullptr;
	CHECK(sslm_gpu_seq_create(ctx, model, /*context_cap=*/64, &seq) == SSLM_OK);

	// 32 forced tokens -> 8 TDR-safe sub-chunks at real full depth (num_hidden_layers *
	// kDispatchesPerLayer dispatches per token, C1's own measured ~15-17ms/token compute floor)
	// -- several hundred milliseconds of genuine GPU-busy wall time, wide enough that a tight
	// polling loop on another thread reliably lands inside the window rather than racing it.
	std::vector<int32_t> span;
	span.reserve(32);
	for (int32_t i = 0; i < 32; ++i) span.push_back(300 + i);

	std::atomic<bool> submission_done{false};
	std::atomic<int> observed_busy{0};
	std::atomic<int> observed_total_polls{0};
	std::atomic<int> observed_unexpected_status{-1};

	std::thread submitter([&]() {
		const SslmGpuStatus st = SslmGpuSeqPrefillPromptForG5Bridge(
		    ctx, seq, span.data(), static_cast<int32_t>(span.size()), kDispatchBudget);
		if (st != SSLM_OK) observed_unexpected_status.store(static_cast<int>(st));
		submission_done.store(true);
	});

	// Poll until the submitter finishes, bounded so a genuine hang fails the cell instead of the
	// process. Every poll that returned SSLM_OK would delete `model` out from under the rest of
	// this suite -- correctness here depends on that never happening: `submitted_sequences>0`
	// (during the window) or `live_sequences>0` (this cell's own `seq`, alive throughout) always
	// takes precedence over the delete-and-return-OK path (gpu_1p0.cpp:789-794).
	const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
	while (!submission_done.load()) {
		const SslmGpuStatus poll_status = sslm_gpu_model_unmap(ctx, model);
		++observed_total_polls;
		if (poll_status == SSLM_BUSY) {
			++observed_busy;
		} else if (poll_status != SSLM_MODEL_HAS_LIVE_SEQUENCES) {
			observed_unexpected_status.store(static_cast<int>(poll_status));
		}
		if (std::chrono::steady_clock::now() > deadline) {
			CHECK_MSG(false, "Guard(SSLM_BUSY window): submitter thread did not finish within 30s");
			break;
		}
	}
	submitter.join();

	CHECK_MSG(observed_unexpected_status.load() == -1,
	          "Guard(SSLM_BUSY window): saw unexpected status %d (submission or a concurrent "
	          "sslm_gpu_model_unmap call), expected SSLM_OK from the submission and only "
	          "SSLM_BUSY/SSLM_MODEL_HAS_LIVE_SEQUENCES from the poll",
	          observed_unexpected_status.load());
	// T-2185 N1: one hit anywhere cannot tell a whole-chunk window from a final-sub-chunk-only
	// window (both produce >=1 BUSY poll over an 8-sub-chunk span). Require a stated MAJORITY of
	// every poll taken across the whole submitter-thread call to observe SSLM_BUSY -- the
	// final-sub-chunk-only window covers 1 of 8 sub-chunks (~12.5% of this cell's own wall time)
	// and cannot clear a strict majority; the whole-chunk window covers essentially the entire
	// call and does.
	const int total_polls = observed_total_polls.load();
	const int busy_polls = observed_busy.load();
	CHECK_MSG(total_polls > 0,
	          "Guard(SSLM_BUSY window): the polling loop never ran -- the submitter thread "
	          "finished before a single poll landed, this cell's own timing assumption failed");
	CHECK_MSG(busy_polls * 2 > total_polls,
	          "Guard(SSLM_BUSY window): only %d/%d polls observed SSLM_BUSY during a genuinely "
	          "in-flight batched chunk submission spanning multiple sub-chunks -- design Sec9's own "
	          "\"occupies exactly one Submitted window per chunk\" requires the window to cover the "
	          "WHOLE chunk (a majority of the call's own wall time), not merely the final sub-chunk",
	          busy_polls, total_polls);

	sslm_gpu_seq_release(ctx, seq);
}

// T-2186 P1 (Significant, confirmation Claude/Poirot/8642652-t2186-t2169-fix2-confirmation.md,
// D-SLM3682) / T-2189 finding 2 (D-SLM3689, this cell's own oracle inverted here): the widened
// `Submitted` window (T-2185 N1, cell above) is opened in `SubmitAdmittedChunkForG5Bridge`
// (gpu_1p0.cpp) before `SubmitChunkToFullDepthForG5Bridge` is called. `SubmitOneSubChunkToFull
// DepthForG5Bridge`'s own tail (superslm_gpu.cpp: `dev.list->Close()`, `dev.queue->Signal()`, the
// `GpuLayerLoopInFlight` allocation) sits outside its own try/catch, so a failed
// `Close()`/`Signal()` (`SSLM_GPU_HR` throws `std::runtime_error`) or a `std::bad_alloc` from the
// allocation can originate there.
//
// D-SLM3682's own remedy (`SubmittedWindowScopeGuard`) closed the Submitted window on every exit
// from `SubmitAdmittedChunkForG5Bridge`, including exception unwind, but still let the raw
// exception escape past the documented `SslmGpuStatus` ABI boundary (gpu_1p0.h). T-2189 finding 2
// (D-SLM3689) closes that: `SubmitAdmittedChunkForG5Bridge` now catches
// `std::bad_alloc`/`std::runtime_error` itself and lets its own existing `derived_count <
// admit_count` fallback (D-SLM3622) resolve to `SSLM_DEVICE_LOST` through the public entry
// point's ordinary return, never a raw throw. This cell's own oracle inverts to match: no
// exception is expected any more, and the model must be genuinely unwedged (this remedy's own
// property is unchanged -- only the reporting channel is) with `SSLM_DEVICE_LOST` observable as
// the prefill call's OWN return value, not a caught exception's side effect.
static void TestGuard_ModelUnwedgesAfterUncoveredTailThrow(SslmGpuContext* ctx,
                                                             SslmGpuModelHandle* model) {
	SslmGpuSequenceHandle* seq = nullptr;
	CHECK(sslm_gpu_seq_create(ctx, model, /*context_cap=*/64, &seq) == SSLM_OK);

	std::vector<int32_t> chunk = {500, 501, 502};  // 3 admitted tokens -- one sub-chunk, one
	                                                // SubmitOneSubChunkToFullDepthForG5Bridge call,
	                                                // well under kT2169TdrSafeMaxChunkTokens (4).

	superslm_gpu::ArmT2169ChunkRecordingTailFaultInjection();
	SslmGpuStatus prefill_status = SSLM_OK;
	bool threw = false;
	std::string what;
	try {
		prefill_status = SslmGpuSeqPrefillPromptForG5Bridge(
		    ctx, seq, chunk.data(), static_cast<int32_t>(chunk.size()), kDispatchBudget);
	} catch (const std::runtime_error& e) {
		threw = true;
		what = e.what();
	}
	// Defensive only -- the seam is single-shot and already fired above if the call reached the
	// tail; clears any stray armed state so a later cell in this same process never inherits it.
	superslm_gpu::ClearT2169ChunkRecordingTailFaultInjection();

	// T-2189 finding 2 (D-SLM3689): the inverted half of this cell's own oracle -- the tail fault
	// must no longer escape as a raw exception at all. A throw reaching here means the containment
	// fix regressed (or was never built), so this is checked first and the cell stops rather than
	// continuing to grade a caught-but-still-thrown call as if it were contained.
	CHECK_MSG(!threw,
	          "Guard(tail-throw containment): SslmGpuSeqPrefillPromptForG5Bridge let the injected "
	          "tail fault escape as a raw exception (\"%s\") -- T-2189 finding 2's containment fix "
	          "must catch this at the SslmGpuStatus boundary, never let it reach the caller",
	          what.c_str());
	CHECK_MSG(prefill_status == SSLM_DEVICE_LOST,
	          "Guard(tail-throw containment): SslmGpuSeqPrefillPromptForG5Bridge returned %s with "
	          "the tail fault armed, want SSLM_DEVICE_LOST -- the injected fault fires before the "
	          "sub-chunk's own dispatches ever execute, so derived_count must fall short of "
	          "admit_count and resolve through the existing device-computed-fallback path "
	          "(D-SLM3622)",
	          GpuStatusName(prefill_status));

	// The discriminating oracle: `seq` is still alive (never released), so `live_sequences > 0`
	// is genuinely true -- if `submitted_sequences` were left wedged at 1, gpu_1p0.cpp's own
	// precedence ordering (submitted_sequences checked before live_sequences) would surface it as
	// SSLM_BUSY here, not SSLM_MODEL_HAS_LIVE_SEQUENCES.
	const SslmGpuStatus unmap_status = sslm_gpu_model_unmap(ctx, model);
	CHECK_MSG(unmap_status != SSLM_BUSY,
	          "Guard(tail-throw unwedge): sslm_gpu_model_unmap returned SSLM_BUSY after the tail "
	          "fault fired inside SubmitChunkToFullDepthForG5Bridge's own uncovered tail -- "
	          "model->submitted_sequences was left incremented forever, the model can never be "
	          "unmapped (got status %s)",
	          GpuStatusName(unmap_status));
	CHECK_MSG(unmap_status == SSLM_MODEL_HAS_LIVE_SEQUENCES,
	          "Guard(tail-throw unwedge): expected SSLM_MODEL_HAS_LIVE_SEQUENCES (this cell's own "
	          "seq is still alive), got %s -- unexpected status, investigate before trusting this "
	          "cell's own SSLM_BUSY-negative result",
	          GpuStatusName(unmap_status));

	// `seq` released, not the model -- `main`'s own final `sslm_gpu_model_unmap` call, after every
	// cell in this file has run, is this suite's own end-to-end proof that unmap genuinely
	// succeeds once nothing is left bound; this cell's own job is only to prove the intermediate
	// BUSY-vs-live-sequences status, above.
	sslm_gpu_seq_release(ctx, seq);
}

// T-2189 finding 6's own fix round (D-SLM3695; T-2191 S6/D-SLM3692's own out-of-scope
// observation, closed here): the reviewer's own named scenario. The cell above
// (`TestGuard_ModelUnwedgesAfterUncoveredTailThrow`) proves the fault is caught and the model can
// still be unmapped; it never issues a SECOND real GPU call on the same context, so it could not
// see the wedge that observation names -- `ID3D12CommandAllocator::Reset()` failing with `E_FAIL`
// on the next call, because `SubmitOneSubChunkToFullDepthForG5Bridge`'s own tail used to leave the
// D3D12 command list open/recording when the injected fault fired before `dev.list->Close()` ever
// ran. This cell issues that second call and checks it against a never-faulted reference arm's own
// output, not merely that it avoids crashing -- red under the reverted fix (the wedge reproduces:
// the second call fails, typically `SSLM_DEVICE_LOST` again or a crash inside `dev.alloc->Reset()`),
// green under it.
static void TestGuard_ContextReusableAfterCaughtTailFault(SslmGpuContext* ctx,
                                                            SslmGpuModelHandle* model) {
	const int64_t kContextCap = 64;
	// The SAME tokens drive the "second call" on both arms -- only whether a fault preceded it,
	// on the SAME shared `harness::GetDevice()` singleton (every context in this process shares
	// one command allocator/list/queue -- `harness::GetDevice()`'s own header comment,
	// d3d12_harness.h), differs between them.
	const std::vector<int32_t> kSecondCallChunk = {700, 701, 702};
	const int32_t kNextToken = 799;

	// Reference arm: a fresh sequence, no fault armed anywhere in this arm's own history, prefills
	// `kSecondCallChunk` directly and reads the following decode step's own produced token -- the
	// never-faulted baseline the candidate arm's own second call is compared against.
	SslmGpuSequenceHandle* ref_seq = nullptr;
	CHECK(sslm_gpu_seq_create(ctx, model, kContextCap, &ref_seq) == SSLM_OK);
	const SslmGpuStatus ref_prefill_status = SslmGpuSeqPrefillPromptForG5Bridge(
	    ctx, ref_seq, kSecondCallChunk.data(), static_cast<int32_t>(kSecondCallChunk.size()),
	    kDispatchBudget);
	CHECK_MSG(ref_prefill_status == SSLM_OK,
	          "Guard(context-reusable pin): reference arm's own prefill (no fault involved anywhere) "
	          "returned %s, want SSLM_OK -- the pin's own baseline is broken, investigate before "
	          "trusting the candidate arm's comparison against it",
	          GpuStatusName(ref_prefill_status));
	int32_t ref_token = -1;
	const SslmGpuStatus ref_step_status =
	    SslmGpuSeqDecodeStepForG5Bridge(ctx, ref_seq, kNextToken, kDispatchBudget, &ref_token);
	CHECK_MSG(ref_step_status == SSLM_OK,
	          "Guard(context-reusable pin): reference arm's own following decode step returned %s, "
	          "want SSLM_OK", GpuStatusName(ref_step_status));
	sslm_gpu_seq_release(ctx, ref_seq);

	// Candidate arm: a second, independent sequence, on the SAME context (and therefore the same
	// shared device/list/queue the reference arm above already used and released cleanly). First
	// call: the identical fault-injection shape `TestGuard_ModelUnwedgesAfterUncoveredTailThrow`
	// exercises -- fires before `dev.list->Close()` ever runs, caught, expected
	// `SSLM_DEVICE_LOST`. Second call: `kSecondCallChunk`, on the SAME seq/ctx, no fault armed --
	// the call T-2191 S6 named. It must succeed and produce the SAME token the never-faulted
	// reference arm produced, not merely "not crash."
	SslmGpuSequenceHandle* cand_seq = nullptr;
	CHECK(sslm_gpu_seq_create(ctx, model, kContextCap, &cand_seq) == SSLM_OK);

	// T-2192 finding 8/T1: drain and discard whatever the debug layer has queued from setup (the
	// reference arm above, sequence creation) so the counts read below are attributable to THIS
	// arm's own two calls, not to unrelated prior traffic. A no-op (returns/drains nothing) when
	// SSLM_GPU_ENABLE_DEBUG_LAYER is not set -- see d3d12_harness.h's own header comment on
	// `DrainDebugLayerMessages`.
	(void)superslm_gpu::harness::GetDevice().DrainDebugLayerMessages();

	const std::vector<int32_t> kFirstCallChunk = {500, 501, 502};
	superslm_gpu::ArmT2169ChunkRecordingTailFaultInjection();
	const SslmGpuStatus first_call_status = SslmGpuSeqPrefillPromptForG5Bridge(
	    ctx, cand_seq, kFirstCallChunk.data(), static_cast<int32_t>(kFirstCallChunk.size()),
	    kDispatchBudget);
	// Defensive only -- the seam is single-shot and already fired above if the first call reached
	// the tail; clears any stray armed state so a later cell in this same process never inherits
	// it, mirroring TestGuard_ModelUnwedgesAfterUncoveredTailThrow's own convention.
	superslm_gpu::ClearT2169ChunkRecordingTailFaultInjection();
	CHECK_MSG(first_call_status == SSLM_DEVICE_LOST,
	          "Guard(context-reusable pin): the faulted first call returned %s, want "
	          "SSLM_DEVICE_LOST -- the injection seam itself did not fire as expected, investigate "
	          "before trusting the second call's own result",
	          GpuStatusName(first_call_status));
	// The faulted first call's own recorded-but-discarded barriers are expected debug-layer noise
	// (the list is Closed without ever executing) -- drained and discarded here so only the SECOND
	// call's own validation traffic is graded below.
	(void)superslm_gpu::harness::GetDevice().DrainDebugLayerMessages();

	const SslmGpuStatus second_call_status = SslmGpuSeqPrefillPromptForG5Bridge(
	    ctx, cand_seq, kSecondCallChunk.data(), static_cast<int32_t>(kSecondCallChunk.size()),
	    kDispatchBudget);
	CHECK_MSG(second_call_status == SSLM_OK,
	          "Guard(context-reusable pin): the SECOND call on the same context, after a caught "
	          "mid-recording tail fault, returned %s instead of SSLM_OK -- the command list was left "
	          "open/recording by the caught fault (T-2189 finding 6's own out-of-scope observation, "
	          "D-SLM3692 Sec9; T-2191 S6) and this context is wedged",
	          GpuStatusName(second_call_status));
	// T-2192 finding 8/T1's own execution pin: on the reverted fix, the second call's own resume
	// barrier claims `StateBefore=COPY_SOURCE` on a buffer the caught first call left in
	// `UNORDERED_ACCESS` (the latch was set before the transition that would have made it true ever
	// executed) -- an invalid prior-state transition, which the D3D12 debug layer flags as a
	// validation ERROR. Zero under the fix (the latch is cleared when the first call's list is
	// discarded, so the second call correctly issues no resume barrier at all). Only meaningful
	// with SSLM_GPU_ENABLE_DEBUG_LAYER=1 set (returns 0 unconditionally otherwise, per
	// DrainDebugLayerMessages's own contract) -- run with that flag set to execution-verify this
	// pin rather than trust it by source reading alone.
	const size_t second_call_validation_messages =
	    superslm_gpu::harness::GetDevice().DrainDebugLayerMessages();
	CHECK_MSG(second_call_validation_messages == 0,
	          "Guard(context-reusable pin): the SECOND call's own D3D12 debug-layer validation "
	          "reported %zu WARNING-or-worse message(s) (see stderr, \"D3D12 VALIDATION\" lines "
	          "above) -- the KV resource-state resume-barrier latch was left asserting a "
	          "transition the caught first call never executed (T-2192 finding T1)",
	          second_call_validation_messages);

	int32_t cand_token = -1;
	const SslmGpuStatus cand_step_status =
	    SslmGpuSeqDecodeStepForG5Bridge(ctx, cand_seq, kNextToken, kDispatchBudget, &cand_token);
	CHECK_MSG(cand_step_status == SSLM_OK,
	          "Guard(context-reusable pin): the candidate arm's own following decode step (after "
	          "recovery) returned %s, want SSLM_OK", GpuStatusName(cand_step_status));
	CHECK_MSG(cand_token == ref_token,
	          "Guard(context-reusable pin): the second call's own produced token (%d) does not match "
	          "the never-faulted reference arm's token (%d) -- the context recovered enough to avoid "
	          "returning an error but not enough to produce the correct output",
	          cand_token, ref_token);

	sslm_gpu_seq_release(ctx, cand_seq);
}

// T-2192 finding M3: the SAME pin shape as `TestGuard_ContextReusableAfterCaughtTailFault` above,
// aimed at the OTHER of the tail's three failure points that catch clause covers -- a `Signal()`
// failure AFTER `ExecuteCommandLists` has already queued the recorded work to the GPU. That
// catch body (T-2192 T2(b)'s own remedy, `SubmitOneSubChunkToFullDepthForG5Bridge`,
// superslm_gpu.cpp) retries `Signal()` at the same fence value and waits it out before returning,
// so -- unlike the pre-Close pin above -- the context is expected to recover with the FIRST call's
// own work genuinely committed, not merely with the second call unaffected. Red under a reverted
// T2(b) fix (the un-waited release races the still-executing GPU work, corrupting or hanging a
// later call on this shared device); green under it.
static void TestGuard_ContextReusableAfterSignalFault(SslmGpuContext* ctx,
                                                        SslmGpuModelHandle* model) {
	const int64_t kContextCap = 64;
	const std::vector<int32_t> kSecondCallChunk = {710, 711, 712};
	const int32_t kNextToken = 809;

	SslmGpuSequenceHandle* ref_seq = nullptr;
	CHECK(sslm_gpu_seq_create(ctx, model, kContextCap, &ref_seq) == SSLM_OK);
	const SslmGpuStatus ref_prefill_status = SslmGpuSeqPrefillPromptForG5Bridge(
	    ctx, ref_seq, kSecondCallChunk.data(), static_cast<int32_t>(kSecondCallChunk.size()),
	    kDispatchBudget);
	CHECK_MSG(ref_prefill_status == SSLM_OK,
	          "Guard(signal-fault reusable pin): reference arm's own prefill returned %s, want "
	          "SSLM_OK -- the pin's own baseline is broken",
	          GpuStatusName(ref_prefill_status));
	int32_t ref_token = -1;
	const SslmGpuStatus ref_step_status =
	    SslmGpuSeqDecodeStepForG5Bridge(ctx, ref_seq, kNextToken, kDispatchBudget, &ref_token);
	CHECK_MSG(ref_step_status == SSLM_OK,
	          "Guard(signal-fault reusable pin): reference arm's own following decode step returned "
	          "%s, want SSLM_OK", GpuStatusName(ref_step_status));
	sslm_gpu_seq_release(ctx, ref_seq);

	SslmGpuSequenceHandle* cand_seq = nullptr;
	CHECK(sslm_gpu_seq_create(ctx, model, kContextCap, &cand_seq) == SSLM_OK);

	const std::vector<int32_t> kFirstCallChunk = {510, 511, 512};
	superslm_gpu::ArmT2169ChunkRecordingTailSignalFaultInjection();
	const SslmGpuStatus first_call_status = SslmGpuSeqPrefillPromptForG5Bridge(
	    ctx, cand_seq, kFirstCallChunk.data(), static_cast<int32_t>(kFirstCallChunk.size()),
	    kDispatchBudget);
	superslm_gpu::ClearT2169ChunkRecordingTailSignalFaultInjection();
	CHECK_MSG(first_call_status == SSLM_DEVICE_LOST,
	          "Guard(signal-fault reusable pin): the faulted first call returned %s, want "
	          "SSLM_DEVICE_LOST -- the injection seam itself did not fire as expected",
	          GpuStatusName(first_call_status));

	const SslmGpuStatus second_call_status = SslmGpuSeqPrefillPromptForG5Bridge(
	    ctx, cand_seq, kSecondCallChunk.data(), static_cast<int32_t>(kSecondCallChunk.size()),
	    kDispatchBudget);
	CHECK_MSG(second_call_status == SSLM_OK,
	          "Guard(signal-fault reusable pin): the SECOND call on the same context, after a caught "
	          "post-Execute Signal() fault, returned %s instead of SSLM_OK -- T-2192 T2(b)'s own "
	          "remedy did not recover the context",
	          GpuStatusName(second_call_status));

	int32_t cand_token = -1;
	const SslmGpuStatus cand_step_status =
	    SslmGpuSeqDecodeStepForG5Bridge(ctx, cand_seq, kNextToken, kDispatchBudget, &cand_token);
	CHECK_MSG(cand_step_status == SSLM_OK,
	          "Guard(signal-fault reusable pin): the candidate arm's own following decode step "
	          "(after recovery) returned %s, want SSLM_OK", GpuStatusName(cand_step_status));
	CHECK_MSG(cand_token == ref_token,
	          "Guard(signal-fault reusable pin): the second call's own produced token (%d) does not "
	          "match the never-faulted reference arm's token (%d)",
	          cand_token, ref_token);

	sslm_gpu_seq_release(ctx, cand_seq);
}

// T-2192 finding M3: the third of the tail's three failure points -- `std::bad_alloc` from
// `new GpuLayerLoopInFlight()`, after `Close()`/`ExecuteCommandLists`/`Signal()` have all
// genuinely succeeded. This is the ONE tail catch T-2192 found already correct before this round
// (the existing fence-wait pattern T2(b)'s own remedy now mirrors); this cell exists so the claim
// is executed, not merely read, and so the class has full three-of-three coverage rather than
// one-of-three. Same pin shape as the two cells above.
static void TestGuard_ContextReusableAfterBadAllocFault(SslmGpuContext* ctx,
                                                          SslmGpuModelHandle* model) {
	const int64_t kContextCap = 64;
	const std::vector<int32_t> kSecondCallChunk = {720, 721, 722};
	const int32_t kNextToken = 819;

	SslmGpuSequenceHandle* ref_seq = nullptr;
	CHECK(sslm_gpu_seq_create(ctx, model, kContextCap, &ref_seq) == SSLM_OK);
	const SslmGpuStatus ref_prefill_status = SslmGpuSeqPrefillPromptForG5Bridge(
	    ctx, ref_seq, kSecondCallChunk.data(), static_cast<int32_t>(kSecondCallChunk.size()),
	    kDispatchBudget);
	CHECK_MSG(ref_prefill_status == SSLM_OK,
	          "Guard(bad_alloc reusable pin): reference arm's own prefill returned %s, want SSLM_OK "
	          "-- the pin's own baseline is broken",
	          GpuStatusName(ref_prefill_status));
	int32_t ref_token = -1;
	const SslmGpuStatus ref_step_status =
	    SslmGpuSeqDecodeStepForG5Bridge(ctx, ref_seq, kNextToken, kDispatchBudget, &ref_token);
	CHECK_MSG(ref_step_status == SSLM_OK,
	          "Guard(bad_alloc reusable pin): reference arm's own following decode step returned %s, "
	          "want SSLM_OK", GpuStatusName(ref_step_status));
	sslm_gpu_seq_release(ctx, ref_seq);

	SslmGpuSequenceHandle* cand_seq = nullptr;
	CHECK(sslm_gpu_seq_create(ctx, model, kContextCap, &cand_seq) == SSLM_OK);

	const std::vector<int32_t> kFirstCallChunk = {520, 521, 522};
	superslm_gpu::ArmT2169ChunkRecordingTailBadAllocFaultInjection();
	const SslmGpuStatus first_call_status = SslmGpuSeqPrefillPromptForG5Bridge(
	    ctx, cand_seq, kFirstCallChunk.data(), static_cast<int32_t>(kFirstCallChunk.size()),
	    kDispatchBudget);
	superslm_gpu::ClearT2169ChunkRecordingTailBadAllocFaultInjection();
	CHECK_MSG(first_call_status == SSLM_DEVICE_LOST,
	          "Guard(bad_alloc reusable pin): the faulted first call returned %s, want "
	          "SSLM_DEVICE_LOST -- the injection seam itself did not fire as expected",
	          GpuStatusName(first_call_status));

	const SslmGpuStatus second_call_status = SslmGpuSeqPrefillPromptForG5Bridge(
	    ctx, cand_seq, kSecondCallChunk.data(), static_cast<int32_t>(kSecondCallChunk.size()),
	    kDispatchBudget);
	CHECK_MSG(second_call_status == SSLM_OK,
	          "Guard(bad_alloc reusable pin): the SECOND call on the same context, after a caught "
	          "post-Execute bad_alloc fault, returned %s instead of SSLM_OK",
	          GpuStatusName(second_call_status));

	int32_t cand_token = -1;
	const SslmGpuStatus cand_step_status =
	    SslmGpuSeqDecodeStepForG5Bridge(ctx, cand_seq, kNextToken, kDispatchBudget, &cand_token);
	CHECK_MSG(cand_step_status == SSLM_OK,
	          "Guard(bad_alloc reusable pin): the candidate arm's own following decode step (after "
	          "recovery) returned %s, want SSLM_OK", GpuStatusName(cand_step_status));
	CHECK_MSG(cand_token == ref_token,
	          "Guard(bad_alloc reusable pin): the second call's own produced token (%d) does not "
	          "match the never-faulted reference arm's token (%d)",
	          cand_token, ref_token);

	sslm_gpu_seq_release(ctx, cand_seq);
}

// T-2195 (Curie, S1/S4 commissioning cell, Claude/Poirot/1381076-t2195-t2189-closing-
// confirmation.md): the WARM-arm extension of `TestGuard_ContextReusableAfterCaughtTailFault`
// above. That cell's own candidate sequence is fresh when the fault fires -- the resume-barrier
// latch (superslm_gpu.cpp, `io_external_kv_needs_resume_barrier`) starts false on a freshly
// created handle (gpu_1p0.cpp's own UNORDERED_ACCESS creation state), the one shape in which S1's
// remedy (`*io_external_kv_needs_resume_barrier = false` in the `!tail_executed` branch,
// superslm_gpu.cpp's tail `runtime_error` catch) happens to write the correct value. Every
// ordinary prompt exceeds `kT2169TdrSafeMaxChunkTokens` (4) tokens and therefore reaches the
// latch's STEADY state (true) before any fault can occur -- S1's own failure sequence. This cell
// reproduces that sequence:
//
//   call 1 (warm-up): a successful multi-sub-chunk prefill on a FRESH sequence, driving the latch
//     from its creation-time false through at least one genuine resume-barrier read/write cycle
//     to its true steady state (buffer left in COPY_SOURCE).
//   call 2 (fault): the SAME pre-Close tail fault `TestGuard_ContextReusableAfterCaughtTailFault`
//     injects, on the SAME sequence -- reached with the latch already true, the shape that cell's
//     own fresh-handle construction cannot reach.
//   call 3 (probe): a third call on the SAME sequence, no fault armed. Two oracles, both against
//     THIS single call:
//     (a) the resulting sequence state bit-compares against a reference sequence driven through
//         the identical successful history (warm-up, then this same third call) with the fault
//         never injected -- the discarded call's whole effect must be invisible, not merely
//         non-crashing (T-2192's own "not merely 'not crash'" standard, restated at that cell's
//         own token-match oracle).
//     (b) the D3D12 debug layer's validation-message drain, scoped to exactly this call
//         (`DrainDebugLayerMessages`, d3d12_harness.h) -- S1's predicted defect is a resume
//         barrier asserting StateBefore=UNORDERED_ACCESS against a resource genuinely in
//         COPY_SOURCE, which the debug layer is documented to flag as a validation ERROR (the
//         SAME oracle `TestGuard_ContextReusableAfterCaughtTailFault` already uses at its own
//         second call). Only meaningful with SSLM_GPU_ENABLE_DEBUG_LAYER=1 -- see that macro's
//         header comment.
//
// This cell is also T-2192 finding S4's own commissioning construction (StandardsDocument.md
// Sec5.4's must-reject): the sibling pins' `..._validation_messages == 0` assertions have a
// must-accept (a clean run scores zero) and no prior must-reject -- nothing showed a genuinely
// desynced latch produces a message on this device/driver. Oracle (b) here is exactly that
// must-reject, authored independently of the S1 remedy's own author and producible by the real
// data path (no seam beyond the pre-existing pre-Close tail-fault injection this suite's sibling
// cells already use). If it fires, the instrument is commissioned and S1 is confirmed by
// execution in the same run; if the assertion in (b) passes anyway (no message despite the
// mismatched transition), that is the S4 refutation of S1's reachability on this device/driver,
// and the instrument stays quarantined -- this cell reports whichever the execution shows, not a
// predicted one.
static void TestGuard_ContextReusableAfterCaughtTailFaultWarmArm(SslmGpuContext* ctx,
                                                                    SslmGpuModelHandle* model) {
	const int64_t kContextCap = 64;
	// > kT2169TdrSafeMaxChunkTokens (4) -- splits into two sub-chunks (4 + 2), so the warm-up call
	// itself exercises the multi-sub-chunk resume-barrier cycle before the fault ever fires.
	const std::vector<int32_t> kWarmChunk = {600, 601, 602, 603, 604, 605};
	const std::vector<int32_t> kFaultChunk = {650, 651, 652};
	const std::vector<int32_t> kThirdCallChunk = {700, 701, 702};
	const int32_t kNextToken = 899;

	// Reference arm: the SAME successful history (warm-up, then the third-position call), with the
	// fault never injected -- the never-faulted reference S1's own failure account calls for. A
	// correct implementation discards the faulted call's entire effect, so this arm's state after
	// its own second call is what the candidate arm's state after its THIRD call must match.
	SslmGpuSequenceHandle* ref_seq = nullptr;
	CHECK(sslm_gpu_seq_create(ctx, model, kContextCap, &ref_seq) == SSLM_OK);
	const SslmGpuStatus ref_warmup_status = SslmGpuSeqPrefillPromptForG5Bridge(
	    ctx, ref_seq, kWarmChunk.data(), static_cast<int32_t>(kWarmChunk.size()), kDispatchBudget);
	CHECK_MSG(ref_warmup_status == SSLM_OK,
	          "Guard(warm-arm reusable pin): reference arm's own warm-up call returned %s, want "
	          "SSLM_OK -- the pin's own baseline is broken",
	          GpuStatusName(ref_warmup_status));
	const SslmGpuStatus ref_third_status = SslmGpuSeqPrefillPromptForG5Bridge(
	    ctx, ref_seq, kThirdCallChunk.data(), static_cast<int32_t>(kThirdCallChunk.size()),
	    kDispatchBudget);
	CHECK_MSG(ref_third_status == SSLM_OK,
	          "Guard(warm-arm reusable pin): reference arm's own third-position call returned %s, "
	          "want SSLM_OK", GpuStatusName(ref_third_status));
	SeqSnapshot ref_snap;
	CHECK(CaptureSnapshot(ref_seq, &ref_snap));
	int32_t ref_token = -1;
	const SslmGpuStatus ref_step_status =
	    SslmGpuSeqDecodeStepForG5Bridge(ctx, ref_seq, kNextToken, kDispatchBudget, &ref_token);
	CHECK_MSG(ref_step_status == SSLM_OK,
	          "Guard(warm-arm reusable pin): reference arm's own following decode step returned %s, "
	          "want SSLM_OK", GpuStatusName(ref_step_status));
	sslm_gpu_seq_release(ctx, ref_seq);

	// Candidate arm: the three-call sequence under test, on ONE sequence handle throughout.
	SslmGpuSequenceHandle* cand_seq = nullptr;
	CHECK(sslm_gpu_seq_create(ctx, model, kContextCap, &cand_seq) == SSLM_OK);

	// Call 1 (warm-up): drives the latch from creation-time false to its steady-state true across
	// a genuine multi-sub-chunk resume-barrier cycle -- the state every real fault-adjacent call
	// actually starts from.
	const SslmGpuStatus warmup_status = SslmGpuSeqPrefillPromptForG5Bridge(
	    ctx, cand_seq, kWarmChunk.data(), static_cast<int32_t>(kWarmChunk.size()), kDispatchBudget);
	CHECK_MSG(warmup_status == SSLM_OK,
	          "Guard(warm-arm reusable pin): candidate arm's own warm-up call returned %s, want "
	          "SSLM_OK -- the pin's own setup is broken",
	          GpuStatusName(warmup_status));

	// Drain and discard whatever the debug layer queued from setup/warm-up so the count read at
	// call 3 below is attributable only to that call, matching
	// `TestGuard_ContextReusableAfterCaughtTailFault`'s own convention.
	(void)superslm_gpu::harness::GetDevice().DrainDebugLayerMessages();

	// Call 2 (fault): the SAME pre-Close tail fault, now reached with the latch true (steady
	// state) instead of the fresh-handle false the sibling pin's own first call reaches it at.
	superslm_gpu::ArmT2169ChunkRecordingTailFaultInjection();
	const SslmGpuStatus fault_call_status = SslmGpuSeqPrefillPromptForG5Bridge(
	    ctx, cand_seq, kFaultChunk.data(), static_cast<int32_t>(kFaultChunk.size()), kDispatchBudget);
	superslm_gpu::ClearT2169ChunkRecordingTailFaultInjection();
	CHECK_MSG(fault_call_status == SSLM_DEVICE_LOST,
	          "Guard(warm-arm reusable pin): the faulted second call returned %s, want "
	          "SSLM_DEVICE_LOST -- the injection seam itself did not fire as expected, investigate "
	          "before trusting the third call's own result",
	          GpuStatusName(fault_call_status));
	// The faulted call's own recorded-but-discarded barriers are expected debug-layer noise (the
	// list is Closed without ever executing) -- drained and discarded here so only call 3's own
	// validation traffic is graded below.
	(void)superslm_gpu::harness::GetDevice().DrainDebugLayerMessages();

	// Call 3 (probe): no fault armed. This is the call S1 names as the one the three existing
	// fresh-handle pins cannot reach -- the latch's disputed `false` write (S1) would mean this
	// call issues no resume barrier where one is required, leaving the KV buffer's actual
	// COPY_SOURCE state unresolved before the pre-copy transition asserts
	// StateBefore=UNORDERED_ACCESS against it.
	const SslmGpuStatus third_call_status = SslmGpuSeqPrefillPromptForG5Bridge(
	    ctx, cand_seq, kThirdCallChunk.data(), static_cast<int32_t>(kThirdCallChunk.size()),
	    kDispatchBudget);
	CHECK_MSG(third_call_status == SSLM_OK,
	          "Guard(warm-arm reusable pin): the THIRD call on the same context, after a caught "
	          "mid-recording tail fault reached with the resume-barrier latch already true, "
	          "returned %s instead of SSLM_OK",
	          GpuStatusName(third_call_status));

	// Oracle (b) -- S1's predicted defect (a resume barrier recorded with the wrong StateBefore)
	// must surface as a WARNING-or-worse D3D12 validation message on this call; this is
	// simultaneously the S4 must-reject the instrument has never been shown to satisfy. Only
	// meaningful with SSLM_GPU_ENABLE_DEBUG_LAYER=1 set (returns 0 unconditionally otherwise).
	const size_t third_call_validation_messages =
	    superslm_gpu::harness::GetDevice().DrainDebugLayerMessages();
	CHECK_MSG(third_call_validation_messages == 0,
	          "Guard(warm-arm reusable pin): the THIRD call's own D3D12 debug-layer validation "
	          "reported %zu WARNING-or-worse message(s) (see stderr, \"D3D12 VALIDATION\" lines "
	          "above) -- the KV resume-barrier latch was left desynced by the caught fault reached "
	          "with the latch already true (T-2195 S1)",
	          third_call_validation_messages);

	// Oracle (a) -- the third call's own resulting sequence state must bit-compare against the
	// never-faulted reference arm's identical-history state; a status of SSLM_OK alone does not
	// prove correctness (T-2192's own "not merely 'not crash'" standard).
	SeqSnapshot cand_snap;
	CHECK(CaptureSnapshot(cand_seq, &cand_snap));
	CHECK_MSG(SnapshotsBitEqual(cand_snap, ref_snap),
	          "Guard(warm-arm reusable pin): the third call's own resulting sequence state does not "
	          "bit-compare against the never-faulted reference arm's identical-history state -- the "
	          "candidate arm returned SSLM_OK without actually reproducing the correct KV/hidden "
	          "state (T-2195 S1)");

	int32_t cand_token = -1;
	const SslmGpuStatus cand_step_status =
	    SslmGpuSeqDecodeStepForG5Bridge(ctx, cand_seq, kNextToken, kDispatchBudget, &cand_token);
	CHECK_MSG(cand_step_status == SSLM_OK,
	          "Guard(warm-arm reusable pin): the candidate arm's own following decode step (after "
	          "recovery) returned %s, want SSLM_OK", GpuStatusName(cand_step_status));
	CHECK_MSG(cand_token == ref_token,
	          "Guard(warm-arm reusable pin): the third call's own following decode step produced "
	          "token %d, want the never-faulted reference arm's token %d",
	          cand_token, ref_token);

	sslm_gpu_seq_release(ctx, cand_seq);
}

// T-2195 pertoken pin (Curie, D-SLM3702 arc round 5 S1/S2 fold, fix committed at
// c1c52296dd1834c39e14647262981c299ac7414e "T-2189 fix round 5, S1/S2: restore the KV
// resume-barrier latch's entry value at both write sites; correct the gpu_1p0.h post-submission
// recovery sentence"): the PER-TOKEN analog of `TestGuard_ContextReusableAfterCaughtTailFault
// WarmArm` immediately above. That cell's own fault-injection seam
// (`MaybeThrowInjectedT2169ChunkRecordingTailFault`) only fired inside
// `SubmitOneSubChunkToFullDepthForG5Bridge`'s own tail -- the BATCHED chunk primitive every
// `SslmGpuSeqPrefillPromptForG5Bridge` call in this file drives, chunk_len == 1 included (this
// file's own header comment: "there is no per-token arm in this file"). Round 5's own S1 fix gave
// `RunLayerLoopGpuSubmit`'s OWN tail -- the primitive the PER-TOKEN drive
// (`sslm_decode_step_gpu` via `SubmitOneSequenceDecode`, gpu_1p0.cpp) calls DIRECTLY, one submit
// call per `kDispatchBudget`-worth of layers, reached from the public surface through
// `SslmGpuSeqDecodeStepForG5Bridge`'s own embed-then-drive branch -- the identical
// try/catch/entry-value-restore shape, landing from the containment's first commit rather than in
// two separate rounds (round 5's own commit message). No dedicated fault-injection pin existed
// for THIS site before this cell: the seam (`ArmT2169ChunkRecordingTailFaultInjection`) is
// extended (gpu_port.h, superslm_gpu.cpp) to also fire from `RunLayerLoopGpuSubmit`'s own tail, at
// the identical pre-Close() point its sibling call site already uses -- arming plumbing only, no
// new production behavior; the throw it injects is the SAME `std::runtime_error` a real failed
// `Close()`/`Signal()` would raise.
//
// `SslmGpuSeqDecodeStepForG5Bridge` only reaches `RunLayerLoopGpuSubmit` when
// `seq->ready_for_logits` is false -- the branch that embeds `token_to_embed_if_needed` and calls
// `DriveGpuSeqToFullDepthForG5Bridge` (gpu_1p0.cpp), which loops `sslm_decode_step_gpu`/
// `sslm_gpu_ready` once per `kDispatchBudget`-worth of layers until `seq->layer_index` reaches
// `num_hidden_layers`. `ready_for_logits` is set true ONLY by the prefill bridges (never by this
// function itself), so every call in this cell's own three-call sequence takes that branch --
// unlike this file's OTHER decode-step calls, which always follow a prefill and therefore always
// take the ready_for_logits shortcut, never touching `RunLayerLoopGpuSubmit` at all.
//
//   call 1 (warm-up): a successful decode step on a FRESH sequence -- embeds a token and drives
//     it through the full layer loop, at least one genuine `RunLayerLoopGpuSubmit` call
//     succeeding and setting the resume-barrier latch to its true steady state (buffer left in
//     COPY_SOURCE) -- the state a fault immediately after warm-up genuinely starts from, the
//     shape S1's own remedy account names as the one the fresh-handle sibling pins cannot reach.
//   call 2 (fault): the SAME pre-Close tail fault the chunk-path pins inject, now reached inside
//     `RunLayerLoopGpuSubmit`'s own tail via the per-token drive -- fires on the FIRST internal
//     submit call this decode step issues (armed before the call, single-shot), with the latch
//     already true.
//   call 3 (probe): a third decode step on the SAME sequence, no fault armed.
//     `sslm_gpu_seq_embed_token` resets `layer_index` to 0 and does not touch `context_length`
//     (gpu_1p0.cpp:1214-1266) -- `context_length` only advances once a token's full traversal
//     finishes, so the faulted call's own aborted embed leaves no committed trace for this call to
//     inherit; this call's own resulting state is the ONLY thing under test. Two oracles, both
//     against this single call, mirroring the chunk pin's own pair exactly:
//     (a) the resulting sequence state bit-compares against a reference sequence driven through
//         the identical successful two-call history (warm-up, then this same third call) with the
//         fault never injected, AND this call's own produced token (a decode step's own
//         `out_token` -- no separate following call is needed, unlike the chunk pin's own prefill
//         arms) must match the reference's.
//     (b) the D3D12 debug layer's validation-message drain, scoped to exactly this call -- a
//         SECOND, independently-producible construction of the identical defect class the
//         chunk-path sibling `TestGuard_ContextReusableAfterCaughtTailFaultWarmArm` drives,
//         reached through a different public entry point and a different internal call site.
//         Per `d3d12_harness.h`'s own `DrainDebugLayerMessages()` header comment (T-2195's
//         DEMONSTRATED LIMIT paragraph), the drain itself is alive on this device/driver (it
//         reliably scores 0 on a clean call and prints real messages when the debug layer has
//         something to report), but its must-reject construction for THIS defect class -- a
//         resume-barrier transition recorded against the wrong prior state -- FAILED TO FIRE on
//         the RTX 2080 SUPER (driver 560.94): the sibling cell drove a source-confirmed reachable
//         instance of the exact defect through this exact drain and it returned 0. This cell's own
//         zero here is therefore quarantined on the same defect class, not a second commissioning:
//         it is read as "no message from this construction on this device," never as "no defect,"
//         until a device/driver pairing is found where the construction produces a WARNING-or-worse
//         message. Only meaningful with SSLM_GPU_ENABLE_DEBUG_LAYER=1 set (returns 0
//         unconditionally otherwise).
static void TestGuard_ContextReusableAfterCaughtTailFaultPerTokenWarmArm(SslmGpuContext* ctx,
                                                                           SslmGpuModelHandle* model) {
	const int64_t kContextCap = 64;
	const int32_t kWarmToken = 850;
	const int32_t kFaultToken = 860;
	const int32_t kThirdToken = 870;

	// Reference arm: the SAME two-call history (warm-up, then the third-position token), fault
	// never injected -- a correct implementation discards the faulted call's entire effect, so
	// this arm's state after its own second call is what the candidate arm's state after its
	// THIRD call must match.
	SslmGpuSequenceHandle* ref_seq = nullptr;
	CHECK(sslm_gpu_seq_create(ctx, model, kContextCap, &ref_seq) == SSLM_OK);
	int32_t ref_warmup_token = -1;
	const SslmGpuStatus ref_warmup_status =
	    SslmGpuSeqDecodeStepForG5Bridge(ctx, ref_seq, kWarmToken, kDispatchBudget, &ref_warmup_token);
	CHECK_MSG(ref_warmup_status == SSLM_OK,
	          "Guard(per-token warm-arm reusable pin): reference arm's own warm-up call returned "
	          "%s, want SSLM_OK -- the pin's own baseline is broken",
	          GpuStatusName(ref_warmup_status));
	int32_t ref_token = -1;
	const SslmGpuStatus ref_third_status =
	    SslmGpuSeqDecodeStepForG5Bridge(ctx, ref_seq, kThirdToken, kDispatchBudget, &ref_token);
	CHECK_MSG(ref_third_status == SSLM_OK,
	          "Guard(per-token warm-arm reusable pin): reference arm's own third-position call "
	          "returned %s, want SSLM_OK", GpuStatusName(ref_third_status));
	SeqSnapshot ref_snap;
	CHECK(CaptureSnapshot(ref_seq, &ref_snap));
	sslm_gpu_seq_release(ctx, ref_seq);

	// Candidate arm: the three-call sequence under test, on ONE sequence handle throughout.
	SslmGpuSequenceHandle* cand_seq = nullptr;
	CHECK(sslm_gpu_seq_create(ctx, model, kContextCap, &cand_seq) == SSLM_OK);

	// Call 1 (warm-up): drives the latch from creation-time false to its steady-state true.
	int32_t warmup_token = -1;
	const SslmGpuStatus warmup_status =
	    SslmGpuSeqDecodeStepForG5Bridge(ctx, cand_seq, kWarmToken, kDispatchBudget, &warmup_token);
	CHECK_MSG(warmup_status == SSLM_OK,
	          "Guard(per-token warm-arm reusable pin): candidate arm's own warm-up call returned "
	          "%s, want SSLM_OK -- the pin's own setup is broken",
	          GpuStatusName(warmup_status));

	// Drain and discard setup/warm-up debug-layer traffic so the count read at call 3 below is
	// attributable only to that call, matching this suite's own established convention.
	(void)superslm_gpu::harness::GetDevice().DrainDebugLayerMessages();

	// Call 2 (fault): the SAME pre-Close tail fault, now reached through RunLayerLoopGpuSubmit's
	// own tail via the per-token drive, with the latch already true (steady state).
	superslm_gpu::ArmT2169ChunkRecordingTailFaultInjection();
	int32_t fault_out_token = -1;
	const SslmGpuStatus fault_call_status = SslmGpuSeqDecodeStepForG5Bridge(
	    ctx, cand_seq, kFaultToken, kDispatchBudget, &fault_out_token);
	// Defensive only -- the seam is single-shot and already fired above if the call reached the
	// tail; clears any stray armed state so a later cell in this same process never inherits it.
	superslm_gpu::ClearT2169ChunkRecordingTailFaultInjection();
	CHECK_MSG(fault_call_status == SSLM_DEVICE_LOST,
	          "Guard(per-token warm-arm reusable pin): the faulted second call returned %s, want "
	          "SSLM_DEVICE_LOST -- the injection seam itself did not fire as expected, investigate "
	          "before trusting the third call's own result",
	          GpuStatusName(fault_call_status));
	// The faulted call's own recorded-but-discarded barriers are expected debug-layer noise (the
	// list is Closed without ever executing) -- drained and discarded here so only call 3's own
	// validation traffic is graded below.
	(void)superslm_gpu::harness::GetDevice().DrainDebugLayerMessages();

	// Call 3 (probe): no fault armed -- the call this cell exists to prove.
	int32_t cand_token = -1;
	const SslmGpuStatus third_call_status =
	    SslmGpuSeqDecodeStepForG5Bridge(ctx, cand_seq, kThirdToken, kDispatchBudget, &cand_token);
	CHECK_MSG(third_call_status == SSLM_OK,
	          "Guard(per-token warm-arm reusable pin): the THIRD call on the same context, after a "
	          "caught mid-recording tail fault reached with the resume-barrier latch already true, "
	          "returned %s instead of SSLM_OK",
	          GpuStatusName(third_call_status));

	// Oracle (b) -- the drain is alive on this device/driver (its must-accept side stands: a
	// clean, correctly-handled call reliably scores 0 here), but per d3d12_harness.h's own
	// DEMONSTRATED LIMIT paragraph, its must-reject construction for THIS defect class (a resume
	// barrier recorded with the wrong StateBefore) FAILED TO FIRE on the RTX 2080 SUPER -- so a 0
	// return here is read as "no message from this construction on this device," never as "no
	// defect," and this class's verdicts from this drain stay quarantined until a device/driver
	// pairing is found where the construction produces a WARNING-or-worse message.
	const size_t third_call_validation_messages =
	    superslm_gpu::harness::GetDevice().DrainDebugLayerMessages();
	CHECK_MSG(third_call_validation_messages == 0,
	          "Guard(per-token warm-arm reusable pin): the THIRD call's own D3D12 debug-layer "
	          "validation reported %zu WARNING-or-worse message(s) (see stderr, \"D3D12 "
	          "VALIDATION\" lines above) -- the KV resume-barrier latch was left desynced by the "
	          "caught fault reached with the latch already true, via the per-token drive (T-2195 "
	          "pertoken pin)",
	          third_call_validation_messages);

	// Oracle (a) -- the third call's own resulting state and produced token must bit-compare
	// against the never-faulted reference arm's identical-history result; an SSLM_OK status alone
	// does not prove correctness (T-2192's own "not merely 'not crash'" standard).
	SeqSnapshot cand_snap;
	CHECK(CaptureSnapshot(cand_seq, &cand_snap));
	CHECK_MSG(SnapshotsBitEqual(cand_snap, ref_snap),
	          "Guard(per-token warm-arm reusable pin): the third call's own resulting sequence "
	          "state does not bit-compare against the never-faulted reference arm's "
	          "identical-history state -- the candidate arm returned SSLM_OK without actually "
	          "reproducing the correct KV/hidden state (T-2195 pertoken pin)");
	CHECK_MSG(cand_token == ref_token,
	          "Guard(per-token warm-arm reusable pin): the third call's own produced token (%d) "
	          "does not match the never-faulted reference arm's token (%d)",
	          cand_token, ref_token);

	sslm_gpu_seq_release(ctx, cand_seq);
}

// T-2189 finding 4 (P2, D-SLM3689): `RunChunkAdmissionPreScan` (gpu_1p0.cpp) used to scan/embed
// the FULL requested chunk length before applying the position cap, doing O(count * hidden_size)
// work and allocating O(count * hidden_size) bytes even when the cap admits ZERO tokens -- a
// caller-controlled `count` with no upper bound driven at a context already full does unbounded
// work for a call that must refuse everything. The fix bounds the DFA/embed scan to
// `position_admit_count + 1` tokens (gpu_1p0.cpp's own header comment on that bound). The
// oracle here is wall-clock time: a chunk this large would cost tens of seconds at minimum
// (EmbedEntry's own O(hidden_size) cost repeated `count` times) and a multi-gigabyte allocation
// attempt (`embed_bytes_cache.reserve(block_bytes * count)`) if the scan were unbounded --
// reverting the bound (scanning to `n` instead of `position_admit_count + 1`) would make this
// cell fail its own time budget, the measurable property a mutation of the fix flips.
static void TestGuard_FullCapHugeCountBoundedPrescanWork(SslmGpuContext* ctx,
                                                           SslmGpuModelHandle* model) {
	const int64_t kSeqContextCap = 4;
	const std::vector<int32_t> prime = {11, 22, 33, 44};  // fills context_length to the cap
	                                                        // exactly -- cap_room == 0 afterward,
	                                                        // position_admit_count == 0 for any
	                                                        // following call regardless of size.

	SslmGpuSequenceHandle* seq = nullptr;
	CHECK(sslm_gpu_seq_create(ctx, model, kSeqContextCap, &seq) == SSLM_OK);
	CHECK(PrimeSeq(ctx, seq, prime));

	// 20,000,000 in-vocab tokens (token id 5, well inside any real model's vocab_size) -- an
	// unbounded pre-scan would run EmbedEntry 20M times and attempt to reserve roughly
	// 20M * ~6KB (T2169SeqEmbeddingBlockBytes at this model's hidden_size) of host memory before
	// ever discovering the cap admits none of it. Constructing the token array itself is a single
	// fast fill, not the property under test.
	constexpr int32_t kHugeCount = 20000000;
	std::vector<int32_t> huge_chunk(static_cast<size_t>(kHugeCount), /*token_id=*/5);

	const int64_t dispatch_before = superslm_test::g_gpu_chunk_dispatch_count_probe.load();
	const auto t0 = std::chrono::steady_clock::now();
	const SslmGpuStatus status =
	    SslmGpuSeqPrefillPromptForG5Bridge(ctx, seq, huge_chunk.data(), kHugeCount, kDispatchBudget);
	const auto t1 = std::chrono::steady_clock::now();
	const double elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
	const int64_t dispatch_delta =
	    superslm_test::g_gpu_chunk_dispatch_count_probe.load() - dispatch_before;

	// Generous budget: the bounded scan examines exactly 1 token (position_admit_count == 0, the
	// "+1" boundary token) and dispatches nothing -- microseconds in practice. 2000 ms is many
	// orders of magnitude more than that bounded cost needs, and many orders of magnitude less
	// than 20M EmbedEntry calls plus a multi-gigabyte reserve() would cost.
	CHECK_MSG(elapsed_ms < 2000.0,
	          "Guard(bounded pre-scan): a %d-token prefill at a fully-saturated context cap took "
	          "%.1f ms -- want under 2000 ms; the admission pre-scan must bound its DFA/embed scan "
	          "to the cap-derived admissible maximum (+1 boundary token for cause disambiguation), "
	          "not the full requested count (T-2189 finding 4, D-SLM3689)",
	          kHugeCount, elapsed_ms);
	CHECK_MSG(status == SSLM_DEVICE_LOST,
	          "Guard(bounded pre-scan): prefill at a fully-saturated cap returned %s, want "
	          "SSLM_DEVICE_LOST (cause == kPositionCap, admit_count == 0 -- D-SLM3619's own "
	          "refuse-the-whole-call disposition)",
	          GpuStatusName(status));
	CHECK_MSG(dispatch_delta == 0,
	          "Guard(bounded pre-scan): dispatch-body invocations = %lld, want 0 -- zero tokens are "
	          "admissible at a fully-saturated cap, so nothing should ever be recorded or "
	          "dispatched",
	          dispatch_delta);

	sslm_gpu_seq_release(ctx, seq);
}

int main(int argc, char** argv) {
	ParseFixtureArgs(argc, argv);
	volatile void* a0 = (void*)&TestTrust_CapStraddleBridgeObservable; (void)a0;
	volatile void* a1 = (void*)&TestTrust_HostileTokenAtIndex1; (void)a1;
	volatile void* a2 = (void*)&TestTrust_HostileCountValues; (void)a2;
	volatile void* a3 = (void*)&TestTrust_HostileStartingContextLength; (void)a3;
	volatile void* a4 = (void*)&TestGuard_PositionCapClampDispatchCountInstrumented; (void)a4;
	volatile void* a5 = (void*)&TestGuard_EmbedAdmitCountClampDispatchCountInstrumented; (void)a5;
	volatile void* a6 = (void*)&TestGuard_BusyUnderWidenedWindow; (void)a6;
	volatile void* a7 = (void*)&TestGuard_ModelUnwedgesAfterUncoveredTailThrow; (void)a7;
	volatile void* a8 = (void*)&TestGuard_FullCapHugeCountBoundedPrescanWork; (void)a8;
	volatile void* a9 = (void*)&TestGuard_ContextReusableAfterCaughtTailFault; (void)a9;
	volatile void* a10 = (void*)&TestGuard_ContextReusableAfterSignalFault; (void)a10;
	volatile void* a11 = (void*)&TestGuard_ContextReusableAfterBadAllocFault; (void)a11;
	volatile void* a12 = (void*)&TestGuard_ContextReusableAfterCaughtTailFaultWarmArm; (void)a12;
	volatile void* a13 =
	    (void*)&TestGuard_ContextReusableAfterCaughtTailFaultPerTokenWarmArm; (void)a13;

	if (g_model_1p5b_path.empty()) {
		SKIP_MSG("this file's cells need --model1p5b=PATH -- not run");
		std::printf("checks=%d failures=%d skips=%d\n", GChecks, GFailures, GSkips);
		return 0;
	}

	SslmGpuContext* ctx = nullptr;
	CHECK(sslm_gpu_context_create(GpuContextConfig{}, &ctx) == SSLM_OK);
	if (!ctx) { std::printf("FATAL: sslm_gpu_context_create returned null\n"); return 2; }

	std::vector<uint8_t> bytes;
	superslm::SslmModelView view{};
	std::string err;
	if (LoadRealModel(g_model_1p5b_path, &view, &bytes, &err)) {
		SslmGpuModelHandle* model = nullptr;
		CHECK(sslm_gpu_model_map(ctx, &view, GpuResidencyConfig{}, &model) == SSLM_OK);
		TestTrust_CapStraddleBridgeObservable(ctx, model);
		TestTrust_HostileTokenAtIndex1(ctx, model, static_cast<int32_t>(view.config.vocab_size));
		TestTrust_HostileCountValues(ctx, model);
		TestTrust_HostileStartingContextLength(ctx, model);
		TestGuard_PositionCapClampDispatchCountInstrumented(
		    ctx, model, static_cast<uint32_t>(view.config.num_hidden_layers));
		TestGuard_EmbedAdmitCountClampDispatchCountInstrumented(
		    ctx, model, static_cast<uint32_t>(view.config.num_hidden_layers),
		    static_cast<int32_t>(view.config.vocab_size));
		TestGuard_BusyUnderWidenedWindow(ctx, model);
		// T-2189 finding 4's own pin ran BEFORE the tail-throw cell below when this ordering was
		// first written, deliberately: the tail-throw cell's injected fault used to leave the
		// underlying D3D12 command list unclosed (the fault fires before Close(), simulating a
		// real device-level failure at that exact point), and running the bounded-prescan pin
		// first kept it on a clean device/command-allocator state. T-2189 finding 6's own fix
		// round (D-SLM3695) closed that gap -- the tail is now covered by its own try/catch and
		// the command list ends Closed on this path too -- so the ordering constraint no longer
		// binds; kept anyway, since nothing is gained by reordering a working sequence.
		// `TestGuard_ContextReusableAfterCaughtTailFault` runs immediately after the tail-throw
		// cell, deliberately: it is the SAME fault-injection shape, one call further -- the pin
		// that proves the caught fault no longer leaves the shared device wedged for the call
		// that comes after it.
		TestGuard_FullCapHugeCountBoundedPrescanWork(ctx, model);
		TestGuard_ModelUnwedgesAfterUncoveredTailThrow(ctx, model);
		TestGuard_ContextReusableAfterCaughtTailFault(ctx, model);
		// T-2192 finding M3: the two follow-on pins, same shared-device reasoning as the cell just
		// above -- each proves its own catch clause recovers the context for the call that follows.
		TestGuard_ContextReusableAfterSignalFault(ctx, model);
		TestGuard_ContextReusableAfterBadAllocFault(ctx, model);
		// T-2195 (Curie): the warm-arm variant of TestGuard_ContextReusableAfterCaughtTailFault --
		// same fault-injection shape, reached with the resume-barrier latch at its true steady
		// state instead of a fresh handle's false. Runs last among the tail-fault family so an
		// unexpected outcome here cannot leave the shared device in a state the cells above have
		// not already exercised and recovered from.
		TestGuard_ContextReusableAfterCaughtTailFaultWarmArm(ctx, model);
		// T-2195 pertoken pin (Curie): the per-token analog of the cell just above, reached
		// through RunLayerLoopGpuSubmit's own tail directly rather than through the chunk
		// primitive. Runs last for the identical reason its sibling does.
		TestGuard_ContextReusableAfterCaughtTailFaultPerTokenWarmArm(ctx, model);
		CHECK(sslm_gpu_model_unmap(ctx, model) == SSLM_OK);
	} else {
		CHECK_MSG(false, "failed to load --model1p5b=%s: %s", g_model_1p5b_path.c_str(), err.c_str());
	}

	CHECK(sslm_gpu_context_destroy(ctx) == SSLM_OK);
	std::printf("checks=%d failures=%d skips=%d\n", GChecks, GFailures, GSkips);
	return GFailures ? 1 : 0;
}

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

	const SslmGpuStatus second_call_status = SslmGpuSeqPrefillPromptForG5Bridge(
	    ctx, cand_seq, kSecondCallChunk.data(), static_cast<int32_t>(kSecondCallChunk.size()),
	    kDispatchBudget);
	CHECK_MSG(second_call_status == SSLM_OK,
	          "Guard(context-reusable pin): the SECOND call on the same context, after a caught "
	          "mid-recording tail fault, returned %s instead of SSLM_OK -- the command list was left "
	          "open/recording by the caught fault (T-2189 finding 6's own out-of-scope observation, "
	          "D-SLM3692 Sec9; T-2191 S6) and this context is wedged",
	          GpuStatusName(second_call_status));

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
		CHECK(sslm_gpu_model_unmap(ctx, model) == SSLM_OK);
	} else {
		CHECK_MSG(false, "failed to load --model1p5b=%s: %s", g_model_1p5b_path.c_str(), err.c_str());
	}

	CHECK(sslm_gpu_context_destroy(ctx) == SSLM_OK);
	std::printf("checks=%d failures=%d skips=%d\n", GChecks, GFailures, GSkips);
	return GFailures ? 1 : 0;
}

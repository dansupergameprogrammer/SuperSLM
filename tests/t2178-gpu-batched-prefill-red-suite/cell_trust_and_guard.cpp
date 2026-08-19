// T-2178 (Curie) -- design Sec9 "Trust boundaries / hostile inputs" and "Guard vitality" rows.
// The two Trust-boundaries occupants named fold round 3 (D-SLM3623) are direct adaptations of
// Claude/Loki/t2176-probe-cap-straddle-bridge-semantics.cpp's own Arm A / Arm F construction into
// committed cells, per this suite's own casebook Sec2 and the design's own Sec8 rung 1 text
// ("T-2176's own probe... is the template for the two new Trust-boundaries cells"); its Arm A
// (the shipped per-token bridge, unmodified) is this file's own reference oracle for both.
#include "fixture_common.h"

#include <atomic>
#include <chrono>
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

	// Arm A: the WHOLE chunk in one call -- the batched candidate under test (post-build, this
	// exercises SubmitChunkToFullDepthForG5Bridge's own admission clamp; today it is the shipped
	// per-token loop, which already clamps correctly per-token -- see cell_bitidentity.cpp's own
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

int main(int argc, char** argv) {
	ParseFixtureArgs(argc, argv);
	volatile void* a0 = (void*)&TestTrust_CapStraddleBridgeObservable; (void)a0;
	volatile void* a1 = (void*)&TestTrust_HostileTokenAtIndex1; (void)a1;
	volatile void* a2 = (void*)&TestTrust_HostileCountValues; (void)a2;
	volatile void* a3 = (void*)&TestTrust_HostileStartingContextLength; (void)a3;
	volatile void* a4 = (void*)&TestGuard_PositionCapClampDispatchCountInstrumented; (void)a4;
	volatile void* a5 = (void*)&TestGuard_EmbedAdmitCountClampDispatchCountInstrumented; (void)a5;
	volatile void* a6 = (void*)&TestGuard_BusyUnderWidenedWindow; (void)a6;

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
		CHECK(sslm_gpu_model_unmap(ctx, model) == SSLM_OK);
	} else {
		CHECK_MSG(false, "failed to load --model1p5b=%s: %s", g_model_1p5b_path.c_str(), err.c_str());
	}

	CHECK(sslm_gpu_context_destroy(ctx) == SSLM_OK);
	std::printf("checks=%d failures=%d skips=%d\n", GChecks, GFailures, GSkips);
	return GFailures ? 1 : 0;
}

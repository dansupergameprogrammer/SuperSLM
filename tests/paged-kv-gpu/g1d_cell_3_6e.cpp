// Paged KV GPU slice, plan step G1 -- cell 3.6e (plan.md Sec7; Sec4.4 *The reset-site assert*, *The queue gate*).
// Bracket: [G3 the internal layer's resets on the singleton; G4 the pool upload].
//
// WHAT THE CELL ASSERTS. The assert fires on one thread, with no race to build. A pooled sequence A is
// made Submitted by sslm_decode_step_gpu WITH THE QUEUE GATE HELD, so A's list is still pending when the
// next call runs, whatever the GPU's speed. On the same thread the test calls one internal-layer function
// that reaches a reset site on the singleton. Each run asserts three things:
//   (1) the channel Sec4.4's assert table gives for the site: PoolSubmitContractError reaches the test (the
//       exception's type is the operand);
//   (2) no reset while A's list is pending: a clean debug layer;
//   (3) after the gate is released and sslm_gpu_ready(A), A's next tokens byte-equal an uninterrupted run
//       (a twin on the same pool, never gated).
// The separating operand is the channel: the named mutant (an assert whose condition is vacuous) runs the
// reset and throws nothing; behind the held gate the reset's own record-submit-wait then hangs, and the
// watchdog (60 s, t2791 cell_prompt_guard_removal.cpp:20/:41) fails the cell. Every run has the timeout.
//
// Parts (--part=):
//   g3 (default) [G3] RunLayerLoopGpu (the chunk-open reset), DynamicScaleReciprocalGpu (DispatchOne's
//       reset), BindDescriptorTableAndReadback (the bind reset), RestoreGpuSequenceState (the restore reset):
//       each lets PoolSubmitContractError reach the test. The control: SslmGpuDeviceLogitsBundleCreateForTest
//       (allocation-fault build), run WITH THE GATE STILL HELD: its per-context device has observed all its
//       own work, so it returns SSLM_OK inside the timeout -- which also shows the gate holds the singleton
//       only. Then the internal layer's remedy, DrainSubmissionsGpu() (carrier row T-stale: the leg re-aimed
//       at a token KEPT IN STORAGE, since from G3 a dropped local drains itself): a token from
//       RunLayerLoopGpuSubmit is moved into storage that outlives the leg and is not finished; a pooled
//       decode then returns SSLM_BUSY (the guard's ledger term sees the stored work unobserved); after
//       DrainSubmissionsGpu() the pooled decode returns SSLM_OK.
//   g4 [G4] The pool upload, UploadPoolRegionsSync, called through the test's own declaration of it (no
//       public header declares it), with the gate held: PoolSubmitContractError, then (2) and (3).
//
// Build: the allocation-fault build (SUPERSLM_GPU_ALLOC_FAULT_INJECTION, i.e. cmake
// -DSUPERSLM_GPU_TEST_SEAMS=ON), where SslmGpuDeviceLogitsBundleCreateForTest and the queue gate exist. The
// gate is released in teardown on every exit (QueueGate's destructor; the watchdog releases it before exit 7).
//
// RED NOW (63e7b0e): link-red. Unresolved: sslm_gpu_kv_pool_create, sslm_gpu_kv_pool_destroy,
// sslm_gpu_seq_create_budgeted (G3), superslm_gpu::ArmQueueGateForTest, ReleaseQueueGateForTest,
// DrainSubmissionsGpu (G3), and in the g4 part superslm_gpu::UploadPoolRegionsSync (G4). Green: G3 for the
// g3 part, G4 for the g4 part, each run inside its timeout, debug layer on (SSLM_GPU_ENABLE_DEBUG_LAYER=1).
//
// -DG1D_3_6E_G3_ONLY builds the g3 part alone, so it links at G3 before UploadPoolRegionsSync exists (G4).
// Run: g1d_cell_3_6e.exe --model=<square artifact: R15, qwen2.5-1.5b-instruct.sslm> [--part=g3|g4]
#define G1D_USES_POOLS 1
#include "g1d_helpers.h"

// The pool upload (Sec4 item 8): "UploadPoolRegionsSync(harness::GetDevice(), pool, regions)". The plan names
// the call, not the region type; this is the test's own declaration (Sec7 3.6e [G4]), and the G4 builder
// aligns the region type with it, or edits this one declaration and records the change.
#if !defined(G1D_POOL_UPLOAD_DECLARED) && !defined(G1D_3_6E_G3_ONLY)
namespace superslm_gpu {
struct PoolUploadRegion {
	uint32_t page = 0;     // page index in the pool
	uint64_t offset = 0;   // byte offset within the page
	uint64_t bytes = 0;    // bytes to upload from the pool's host mirror
};
void UploadPoolRegionsSync(harness::Device& dev, SslmGpuKvPoolHandle* pool,
                           const std::vector<PoolUploadRegion>& regions);
}  // namespace superslm_gpu
#endif

namespace {

using g1d::Bench;
using g1d::QueueGate;
using g1d::Thrown;

constexpr int kNextTokens = 4;

struct Ctx {
	Bench b;
	g1d::InternalLoop il;
	SslmGpuKvPoolHandle* pool = nullptr;
	SslmGpuSequenceHandle* A = nullptr;
	std::vector<int32_t> ref;  // the uninterrupted twin's next tokens
	int32_t t0 = 0, t1 = 0;
};

// One gated run: A Submitted behind the gate, then `site` on this thread; (1) the channel, then release,
// ready, (2) the debug layer, (3) the tokens. `control` runs with the gate still held, after the site.
void GatedRun(Ctx& c, const char* name, const std::function<void()>& site,
              const std::function<void()>& control = nullptr) {
	g1d::SetPhase(name);
	CHECK_MSG(sslm_gpu_seq_reset(c.b.ctx, c.A) == SSLM_OK, "%s: reset A", name);
	(void)g1d::DebugMessages();  // setup traffic is not this run's
	QueueGate gate;
	gate.Arm();
	const SslmGpuStatus sub = g1d::SubmitDecode(c.b, c.A, c.t0);
	CHECK_MSG(sub == SSLM_OK, "%s: SETUP: A's decode under the held gate returned %s, want SSLM_OK", name,
	          StatusName(sub));
	std::string what;
	const Thrown t = g1d::CallInternal(site, &what);
	std::printf("    %s: %s%s%s\n", name, g1d::ThrownName(t), what.empty() ? "" : " -- ", what.c_str());
	CHECK_MSG(t == Thrown::PoolSubmitContract,
	          "%s (1): the internal-layer call let %s reach the test, want PoolSubmitContractError before the reset",
	          name, g1d::ThrownName(t));
	if (control) control();
	gate.Release();
	const g1d::ReadyOut r = g1d::Ready(c.b.ctx, c.A, 1);
	CHECK_MSG(r.ret == SSLM_OK && r.ready == 1 && r.out == SSLM_OK,
	          "%s: sslm_gpu_ready(A, block=1) after release: ret=%s ready=%d out=%s", name, StatusName(r.ret), r.ready,
	          StatusName(r.out));
	const size_t msgs = g1d::DebugMessages();
	CHECK_MSG(msgs == 0, "%s (2): %zu WARNING-or-worse debug-layer message(s) (stderr) -- a reset ran under A's list",
	          name, msgs);
	const std::vector<int32_t> got = g1d::NextTokens(c.b, c.A, c.t1, kNextTokens);
	CHECK_MSG(got == c.ref, "%s (3): A's next tokens [%s] differ from the uninterrupted twin's [%s]", name,
	          g1d::Join(got).c_str(), g1d::Join(c.ref).c_str());
}

bool Setup(Ctx& c) {
	const std::string model = g1d::Opt("model");
	if (model.empty()) {
		SKIP_MSG("3.6e needs --model=PATH (a square artifact: R15, qwen2.5-1.5b-instruct.sslm)");
		return false;
	}
	if (!c.b.Open(model)) {
		CHECK_MSG(false, "SETUP: %s did not load and map", model.c_str());
		return false;
	}
	if (!c.il.Build(c.b.fx.view, c.b.fx.has_qk_norm)) {
		CHECK_MSG(false, "SETUP: the internal-layer fixture did not build");
		return false;
	}
	if (!g1d::RequireDebugLayer()) return false;
	c.pool = c.b.Pool(64);
	CHECK_MSG(c.pool != nullptr, "SETUP: sslm_gpu_kv_pool_create(64 pages) failed");
	if (!c.pool) return false;
	c.A = c.b.Budgeted(c.pool, 64);
	SslmGpuSequenceHandle* twin = c.b.Budgeted(c.pool, 64);
	CHECK_MSG(c.A && twin, "SETUP: two budgeted sequences of 64 on the pool");
	if (!c.A || !twin) return false;
	c.t0 = c.b.fx.SomeToken();
	c.t1 = c.b.fx.OtherToken();
	// The reference: the same calls on a twin, never gated, never interrupted.
	if (g1d::SubmitDecode(c.b, twin, c.t0) != SSLM_OK || g1d::Ready(c.b.ctx, twin, 1).First() != SSLM_OK) {
		CHECK_MSG(false, "SETUP: the twin's reference decode did not complete");
		return false;
	}
	SslmGpuStatus bad = SSLM_OK;
	c.ref = g1d::NextTokens(c.b, twin, c.t1, kNextTokens, &bad);
	CHECK_MSG(bad == SSLM_OK, "SETUP: the twin's reference tokens: %s", StatusName(bad));
	std::printf("    reference tokens: [%s]\n", g1d::Join(c.ref).c_str());
	return bad == SSLM_OK;
}

void PartG3(Ctx& c) {
	// The restore leg's blob: a dedicated sequence's own save, made before any gate is held.
	SslmGpuSequenceHandle* d = c.b.Dedicated(64);
	CHECK_MSG(d != nullptr, "SETUP: dedicated sequence for the restore leg's blob");
	std::vector<uint8_t> blob;
	if (d) {
		const std::vector<int32_t> p = c.b.fx.TokensP();
		CHECK(Prefill(c.b.fx, d, p) == SSLM_OK);
		blob = g1d::SaveBlob(c.b.ctx, d);
		CHECK_MSG(!blob.empty(), "SETUP: sslm_gpu_seq_save of the restore leg's blob");
	}

#if defined(SUPERSLM_GPU_ALLOC_FAULT_INJECTION)
	// The per-context control, with the gate held: SSLM_OK inside the timeout.
	auto control = [&c] {
		g1d::SetPhase("control: SslmGpuDeviceLogitsBundleCreateForTest with the gate held");
		const uint32_t V = 8, H = static_cast<uint32_t>(c.b.fx.hidden);
		std::vector<int8_t> head(static_cast<size_t>(V) * H);
		for (size_t i = 0; i < head.size(); ++i) head[i] = static_cast<int8_t>((i * 7) % 127 - 63);
		SslmGpuDeviceLogitsBundleForTest* bundle = nullptr;
		const SslmGpuStatus st = SslmGpuDeviceLogitsBundleCreateForTest(c.b.ctx, head.data(), V, H, &bundle);
		std::printf("    control: %s\n", StatusName(st));
		CHECK_MSG(st == SSLM_OK,
		          "control: the per-context bundle create with the singleton's gate held returned %s, want SSLM_OK "
		          "(a per-context device that observed its own work runs, and is never gated)",
		          StatusName(st));
		if (bundle) SslmGpuDeviceLogitsBundleDestroyForTest(bundle);
	};
#else
	auto control = [] {
		CHECK_MSG(false, "BUILD: 3.6e's control needs the allocation-fault build (SUPERSLM_GPU_ALLOC_FAULT_INJECTION)");
	};
#endif

	GatedRun(c, "G3 chunk-open reset: RunLayerLoopGpu", [&c] { (void)c.il.RunSync(); }, control);
	GatedRun(c, "G3 DispatchOne's reset: DynamicScaleReciprocalGpu", [] { g1d::CallDispatchOneSite(); });
	GatedRun(c, "G3 the bind reset: BindDescriptorTableAndReadback", [] { g1d::CallBindSite(); });
	if (!blob.empty()) {
		const size_t hidden = c.b.fx.hidden;
		GatedRun(c, "G3 the restore reset: RestoreGpuSequenceState", [&blob, hidden] { g1d::CallRestoreSite(blob, hidden); });
	}

	// The remedy (carrier row T-stale): a token kept in storage, then DrainSubmissionsGpu().
	g1d::SetPhase("G3 DrainSubmissionsGpu: a token kept in storage");
	CHECK(sslm_gpu_seq_reset(c.b.ctx, c.A) == SSLM_OK);
	g1d::InFlightToken* stash = nullptr;
	{
		QueueGate gate;
		gate.Arm();  // the stored work is certainly pending when it is stored
		g1d::InFlightToken tok{};
		const superslm::SslmForwardStatus st = c.il.Submit(&tok);
		CHECK_MSG(st == superslm::SslmForwardStatus::Ok, "drain leg: SETUP: RunLayerLoopGpuSubmit returned %s",
		          superslm::SslmForwardStatusName(st));
		stash = new g1d::InFlightToken(std::move(tok));  // moved into storage that outlives the leg; not finished
	}  // the gate is released here; the stored work completes on the GPU and nobody observes it
	const SslmGpuStatus before = g1d::SubmitDecode(c.b, c.A, c.t0);
	CHECK_MSG(before == SSLM_BUSY,
	          "drain leg: a pooled decode with the stored token unobserved returned %s, want SSLM_BUSY (the guard's "
	          "ledger term; nothing is Submitted, so no count term can refuse)",
	          StatusName(before));
	if (before == SSLM_OK) (void)g1d::Ready(c.b.ctx, c.A, 1);
	std::string what;
	const Thrown dt = g1d::CallInternal([] { superslm_gpu::DrainSubmissionsGpu(); }, &what);
	CHECK_MSG(dt == Thrown::None, "drain leg: DrainSubmissionsGpu() threw %s %s", g1d::ThrownName(dt), what.c_str());
	CHECK(sslm_gpu_seq_reset(c.b.ctx, c.A) == SSLM_OK);
	const SslmGpuStatus after = g1d::SubmitDecode(c.b, c.A, c.t0);
	CHECK_MSG(after == SSLM_OK, "drain leg: the pooled decode after DrainSubmissionsGpu() returned %s, want SSLM_OK",
	          StatusName(after));
	if (after == SSLM_OK) {
		const g1d::ReadyOut r = g1d::Ready(c.b.ctx, c.A, 1);
		CHECK(r.First() == SSLM_OK);
		const std::vector<int32_t> got = g1d::NextTokens(c.b, c.A, c.t1, kNextTokens);
		CHECK_MSG(got == c.ref, "drain leg: A's next tokens [%s] differ from the twin's [%s]", g1d::Join(got).c_str(),
		          g1d::Join(c.ref).c_str());
	}
	// Teardown: finish the stored token (its fence has passed; the finish only reads back).
	int32_t ready = 0;
	(void)c.il.Finish(std::move(*stash), 1, &ready);
	delete stash;
}

#if !defined(G1D_3_6E_G3_ONLY)
void PartG4(Ctx& c) {
	auto upload = [&c] {
		const uint64_t page_bytes = 64;  // one small region of page 0; the reset precedes any copy
		std::vector<superslm_gpu::PoolUploadRegion> regions{{0u, 0u, page_bytes}};
		superslm_gpu::UploadPoolRegionsSync(superslm_gpu::harness::GetDevice(), c.pool, regions);
	};
	GatedRun(c, "G4 the pool upload: UploadPoolRegionsSync", upload);
}
#endif

}  // namespace

int main(int argc, char** argv) {
	g1d::ParseOpts(argc, argv);
	g1d::StartWatchdog(60);
	const std::string part = g1d::Opt("part", "g3");
	std::printf("cell 3.6e part=%s\n", part.c_str());
	Ctx c;
	if (Setup(c)) {
		if (part == "g3") PartG3(c);
#if !defined(G1D_3_6E_G3_ONLY)
		else if (part == "g4") PartG4(c);
#endif
		else CHECK_MSG(false, "unknown --part=%s (g3|g4)", part.c_str());
	}
	g1d::PhaseDone();
	c.b.Close();
	return FinishSuite("g1d_cell_3_6e");
}

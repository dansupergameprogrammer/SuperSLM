// T-2124 (D-SLM3446): standalone executable witness for the adapter use-after-free fix (P0-3) and
// its own S2 finding (Claude/Poirot/435f730-t2124-adapter-uaf-review.md). Committed per that
// review's adopted recommendation: the committed suite pins for this ticket
// (tests/t2112-gpu-1p0-red-suite/dim2_hostile_red.cpp TestDim2_M4,
// tests/t2112-gpu-1p0-red-suite/dim5_failure_red.cpp TestDim5_M4) currently guard nothing on a
// machine where the real-content decode path is red end to end (D-SLM3388, entirely -- not
// intermittently, confirmed independently by the build seat and by review) -- this tool, isolated
// from both suite files' own unrelated instability, is the only construction that has actually
// executed the fixed contract. Mirrors tools/t2113_b*.cpp's own convention: built by build.bat,
// NOT auto-run (needs real .sslm artifacts this build does not assume exist on disk).
//
// Usage: t2124_adapter_uaf_repro <model1p5b.sslm> <adapter.sslm> [--concurrent]
//   (no flag)     Mode 1 -- single-threaded UAF pin (P0-3). Submit one decode with the adapter
//                 bound, do NOT drain, attempt unmap (must return Busy), drain, unmap again (must
//                 return Ok). This is the exact construction that, run against the pre-fix
//                 gpu_1p0.cpp (git stash), crashes the process with STATUS_HEAP_CORRUPTION
//                 (0xC0000374) -- the real use-after-free, not a modeled one -- and against the
//                 post-fix code exits 0. Exit 0 on pass, 1 on fail, matching every t2113_b* tool's
//                 own convention.
//   --concurrent  Mode 2 -- S2's own race shape: two OS threads, two sequences, ONE shared adapter.
//                 Design Sec5.4 requires a caller to serialize only the submit call itself
//                 (sslm_decode_step_gpu) -- NOT sslm_gpu_ready, which is the whole point of the
//                 Submit/Finish split (Sec6.2). Each thread here wraps ONLY its own
//                 sslm_decode_step_gpu call in a shared mutex (satisfying that requirement) and
//                 calls sslm_gpu_ready OUTSIDE the lock, unsynchronized against the other thread's
//                 submit -- exactly the shape the review's own Prediction names as where this class
//                 of bug actually surfaces (every concurrent cell the existing suite has drives
//                 submit+drain under one mutex together, which serializes both writers by accident
//                 and would never exercise this). Runs many submit/drain cycles per thread and
//                 reports how many completed before any decode call fails -- proving, on a machine
//                 where the real-content decode path works, that the shared adapter's own counter
//                 survives concurrent increment/decrement from two threads without a lost update
//                 (adapter_unmap succeeds exactly once, after both threads' sequences have both
//                 drained, never early and never stuck Busy forever).
#include <atomic>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "superslm/artifact.h"
#include "superslm/gpu_1p0.h"
#include "superslm/model.h"

static bool LoadRealModel(const std::string& path, superslm::SslmModelView* out_view,
                           std::vector<uint8_t>* out_bytes, std::string* out_err) {
	std::FILE* f = std::fopen(path.c_str(), "rb");
	if (!f) { *out_err = "could not open " + path; return false; }
	std::fseek(f, 0, SEEK_END);
	const long sz = std::ftell(f);
	std::fseek(f, 0, SEEK_SET);
	out_bytes->resize(sz > 0 ? (size_t)sz : 0);
	if (sz > 0) {
		const size_t n = std::fread(out_bytes->data(), 1, out_bytes->size(), f);
		if (n != out_bytes->size()) { std::fclose(f); *out_err = "short read on " + path; return false; }
	}
	std::fclose(f);
	const superslm::SslmModelStatus st =
	    superslm::SslmModel::Load(out_bytes->data(), out_bytes->size(), *out_view, out_err);
	return st == superslm::SslmModelStatus::Ok;
}

// Mode 1 -- see file header. Isolated: fresh ctx/model/adapter/seq, nothing shared with Mode 2.
static int RunUafPin(const superslm::SslmModelView& mview, const superslm::SslmModelView& aview) {
	SslmGpuContext* ctx = nullptr;
	SslmGpuStatus st = sslm_gpu_context_create(GpuContextConfig{}, &ctx);
	std::printf("[uaf] context_create: %d\n", (int)st);
	if (st != SSLM_OK) return 1;

	SslmGpuModelHandle* model = nullptr;
	st = sslm_gpu_model_map(ctx, &mview, GpuResidencyConfig{}, &model);
	std::printf("[uaf] model_map: %d\n", (int)st);
	if (st != SSLM_OK) return 1;

	SslmGpuAdapterHandle* adapter = nullptr;
	st = sslm_gpu_adapter_map(ctx, model, &aview, &adapter);
	std::printf("[uaf] adapter_map: %d\n", (int)st);
	if (st != SSLM_OK) return 1;

	SslmGpuSequenceHandle* seq = nullptr;
	st = sslm_gpu_seq_create(ctx, model, 64, &seq);
	std::printf("[uaf] seq_create: %d\n", (int)st);
	if (st != SSLM_OK) return 1;

	st = sslm_gpu_seq_embed_token(ctx, seq, 5);
	std::printf("[uaf] embed_token: %d\n", (int)st);
	if (st != SSLM_OK) return 1;

	// Submit ONE decode call with the adapter bound, deliberately do NOT drain.
	st = sslm_decode_step_gpu(ctx, seq, adapter, 24u);
	std::printf("[uaf] decode_step_gpu (adapter bound, undrained): %d\n", (int)st);
	if (st != SSLM_OK) {
		std::printf("[uaf] BLOCKED before the pin could run -- decode itself was rejected "
		            "(D-SLM3388-class real-content instability, not this ticket's own change)\n");
		return 2;
	}

	// THE PIN: attempt to unmap the adapter while its submission is still in flight.
	SslmGpuStatus unmap_while_submitted = sslm_gpu_adapter_unmap(ctx, adapter);
	std::printf("[uaf] adapter_unmap WHILE SUBMITTED: %d (expect SSLM_BUSY=%d)\n",
	            (int)unmap_while_submitted, (int)SSLM_BUSY);

	int32_t ready = 0;
	SslmGpuStatus out_status = SSLM_OK;
	while (!ready) st = sslm_gpu_ready(ctx, seq, /*block=*/1, &ready, &out_status);
	std::printf("[uaf] drain: st=%d out_status=%d\n", (int)st, (int)out_status);

	SslmGpuStatus unmap_after_drain = sslm_gpu_adapter_unmap(ctx, adapter);
	std::printf("[uaf] adapter_unmap AFTER DRAIN: %d (expect SSLM_OK=%d)\n",
	            (int)unmap_after_drain, (int)SSLM_OK);

	sslm_gpu_seq_release(ctx, seq);
	sslm_gpu_model_unmap(ctx, model);
	sslm_gpu_context_destroy(ctx);

	const bool pass = (unmap_while_submitted == SSLM_BUSY) && (unmap_after_drain == SSLM_OK);
	std::printf("[uaf] RESULT: %s\n", pass ? "PASS" : "FAIL");
	return pass ? 0 : 1;
}

// Mode 2 -- see file header. Submit-only mutex (design Sec5.4's own requirement); drain
// (sslm_gpu_ready) deliberately unsynchronized against the other thread's submit.
//
// `sslm_gpu_seq_create`/`sslm_gpu_seq_release` are NOT raced here -- design Sec5.4 blesses
// concurrent DECODE (submit-serialized) against disjoint sequence handles; it does not bless
// concurrent creation/release, and racing them hits the already-documented, unrelated
// ID3D12CommandAllocator::Reset()-while-executing hazard tools/t2113_b8_thread_smoke.cpp's own
// header comment names (D-SLM3384) -- confirmed by execution while authoring this tool (two
// threads calling sslm_gpu_seq_create concurrently reproduced that exact HR failure at
// src/gpu/gpu_1p0.cpp's own UploadResidentBufferSyncTo). Both sequences are created and released
// single-threaded, outside the race window, so the ONLY unsynchronized concurrent writers in this
// construction are the ones S2 names: SubmitOneSequenceDecode's increment (inside the submit lock)
// and sslm_gpu_ready's decrement (outside every lock) on the SHARED adapter's own counter.
static std::mutex g_submit_only_mutex;

struct ThreadResult {
	int cycles_completed = 0;
	int first_bad_status = -1;  // -1 == no failure observed
	const char* failed_at = "";
};

static void ConcurrentWorker(SslmGpuContext* ctx, SslmGpuSequenceHandle* seq,
                              SslmGpuAdapterHandle* adapter, int iterations, ThreadResult* out) {
	SslmGpuStatus st = SSLM_OK;
	for (int i = 0; i < iterations; ++i) {
		st = sslm_gpu_seq_embed_token(ctx, seq, 5);  // host-only, disjoint per sequence (Sec5.4)
		if (st != SSLM_OK) { out->first_bad_status = (int)st; out->failed_at = "embed_token"; break; }

		{
			// Submit-only critical section -- matches design Sec5.4's own requirement exactly,
			// and nothing more: sslm_gpu_ready below runs OUTSIDE this lock, on purpose.
			std::lock_guard<std::mutex> lock(g_submit_only_mutex);
			st = sslm_decode_step_gpu(ctx, seq, adapter, 24u);
		}
		if (st != SSLM_OK) { out->first_bad_status = (int)st; out->failed_at = "decode_step_gpu"; break; }

		int32_t ready = 0;
		SslmGpuStatus out_status = SSLM_OK;
		while (!ready) st = sslm_gpu_ready(ctx, seq, /*block=*/1, &ready, &out_status);
		if (st != SSLM_OK || out_status != SSLM_OK) {
			out->first_bad_status = (int)(st != SSLM_OK ? st : out_status);
			out->failed_at = "ready";
			break;
		}
		out->cycles_completed = i + 1;
	}
}

static int RunConcurrentRace(const superslm::SslmModelView& mview,
                              const superslm::SslmModelView& aview) {
	constexpr int kIterationsPerThread = 3000;

	SslmGpuContext* ctx = nullptr;
	SslmGpuStatus st = sslm_gpu_context_create(GpuContextConfig{}, &ctx);
	std::printf("[concurrent] context_create: %d\n", (int)st);
	if (st != SSLM_OK) return 1;

	SslmGpuModelHandle* model = nullptr;
	st = sslm_gpu_model_map(ctx, &mview, GpuResidencyConfig{}, &model);
	std::printf("[concurrent] model_map: %d\n", (int)st);
	if (st != SSLM_OK) return 1;

	SslmGpuAdapterHandle* adapter = nullptr;
	st = sslm_gpu_adapter_map(ctx, model, &aview, &adapter);
	std::printf("[concurrent] adapter_map: %d\n", (int)st);
	if (st != SSLM_OK) return 1;

	// Both sequences created single-threaded, before the race window opens -- see ConcurrentWorker's
	// own header comment for why concurrent creation is not part of what this test races.
	SslmGpuSequenceHandle* seq1 = nullptr;
	st = sslm_gpu_seq_create(ctx, model, 64, &seq1);
	std::printf("[concurrent] seq_create 1: %d\n", (int)st);
	if (st != SSLM_OK) return 1;
	SslmGpuSequenceHandle* seq2 = nullptr;
	st = sslm_gpu_seq_create(ctx, model, 64, &seq2);
	std::printf("[concurrent] seq_create 2: %d\n", (int)st);
	if (st != SSLM_OK) return 1;

	ThreadResult r1, r2;
	std::thread t1(ConcurrentWorker, ctx, seq1, adapter, kIterationsPerThread, &r1);
	std::thread t2(ConcurrentWorker, ctx, seq2, adapter, kIterationsPerThread, &r2);
	t1.join();
	t2.join();

	std::printf("[concurrent] thread 1: %d/%d cycles completed%s%s\n", r1.cycles_completed,
	            kIterationsPerThread, r1.first_bad_status >= 0 ? " -- first failure at " : "",
	            r1.first_bad_status >= 0 ? r1.failed_at : "");
	std::printf("[concurrent] thread 2: %d/%d cycles completed%s%s\n", r2.cycles_completed,
	            kIterationsPerThread, r2.first_bad_status >= 0 ? " -- first failure at " : "",
	            r2.first_bad_status >= 0 ? r2.failed_at : "");

	if (r1.cycles_completed == 0 && r2.cycles_completed == 0) {
		std::printf("[concurrent] BLOCKED before either thread completed one cycle -- "
		            "D-SLM3388-class real-content instability, not this ticket's own change "
		            "(first_bad_status thread1=%d thread2=%d)\n",
		            r1.first_bad_status, r2.first_bad_status);
		sslm_gpu_seq_release(ctx, seq1);
		sslm_gpu_seq_release(ctx, seq2);
		sslm_gpu_adapter_unmap(ctx, adapter);
		sslm_gpu_model_unmap(ctx, model);
		sslm_gpu_context_destroy(ctx);
		return 2;
	}

	// Both threads drained every cycle they completed before their next submit, and both sequences
	// are released single-threaded here, after both threads have joined -- so submitted_sequences
	// should read 0 on BOTH the model and the adapter, and unmap must succeed exactly here, proving
	// the shared adapter's own counter survived concurrent increment (one thread's submit) racing
	// the OTHER thread's decrement (its drain, and vice versa) with no lost update in either
	// direction.
	SslmGpuStatus seq1_release_st = sslm_gpu_seq_release(ctx, seq1);
	SslmGpuStatus seq2_release_st = sslm_gpu_seq_release(ctx, seq2);
	std::printf("[concurrent] seq_release 1/2: %d/%d (expect SSLM_OK=%d)\n", (int)seq1_release_st,
	            (int)seq2_release_st, (int)SSLM_OK);
	SslmGpuStatus adapter_unmap_st = sslm_gpu_adapter_unmap(ctx, adapter);
	std::printf("[concurrent] adapter_unmap after both threads joined: %d (expect SSLM_OK=%d)\n",
	            (int)adapter_unmap_st, (int)SSLM_OK);
	SslmGpuStatus model_unmap_st = sslm_gpu_model_unmap(ctx, model);
	std::printf("[concurrent] model_unmap: %d (expect SSLM_OK=%d)\n", (int)model_unmap_st,
	            (int)SSLM_OK);
	SslmGpuStatus destroy_st = sslm_gpu_context_destroy(ctx);
	std::printf("[concurrent] context_destroy: %d (expect SSLM_OK=%d)\n", (int)destroy_st,
	            (int)SSLM_OK);

	const bool pass = (seq1_release_st == SSLM_OK) && (seq2_release_st == SSLM_OK) &&
	                  (adapter_unmap_st == SSLM_OK) && (model_unmap_st == SSLM_OK) &&
	                  (destroy_st == SSLM_OK) && (r1.first_bad_status < 0) && (r2.first_bad_status < 0);
	std::printf("[concurrent] RESULT: %s (%d+%d cycles total)\n", pass ? "PASS" : "FAIL",
	            r1.cycles_completed, r2.cycles_completed);
	return pass ? 0 : 1;
}

int main(int argc, char** argv) {
	if (argc < 3) {
		std::printf("usage: %s <model1p5b.sslm> <adapter.sslm> [--concurrent]\n", argv[0]);
		return 2;
	}
	const bool concurrent = argc >= 4 && std::strcmp(argv[3], "--concurrent") == 0;

	std::vector<uint8_t> mbytes, abytes;
	superslm::SslmModelView mview{}, aview{};
	std::string err;
	if (!LoadRealModel(argv[1], &mview, &mbytes, &err)) {
		std::printf("model load failed: %s\n", err.c_str());
		return 2;
	}
	if (!LoadRealModel(argv[2], &aview, &abytes, &err)) {
		std::printf("adapter load failed: %s\n", err.c_str());
		return 2;
	}

	return concurrent ? RunConcurrentRace(mview, aview) : RunUafPin(mview, aview);
}

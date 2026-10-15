// Paged-KV plan (rev 16.2), step G1, group g1b: cell 11.4 [G3], the GPU pool size guard, with its G3
// part (rev 15.1), its bisection and the carrier's CEIL, PR13-N3, CEIL-15 and R12-G3 rows.
//
// Built in the fault-injection configuration (SUPERSLM_GPU_ALLOC_FAULT_INJECTION, the T-2851 seam
// build t2956 and te425 link), because its positive side "does not wait on the hardware to refuse"
// (11.4): the seam legs drive Device::TryMakeBuffer's countdown. Built without the define, the cell
// fails ("built without the fault-injection seams"): it never passes on its ceiling legs alone.
//
// RED NOW: link-red on sslm_gpu_kv_pool_create / sslm_gpu_kv_pool_destroy (G3), and on the two
// seams g1b_helpers.h declares for R12-G3 (SslmGpuAllocRecordForTest, ArmGpuPoolZeroFillCloseFaultForTest:
// names this group chose; G3's builder defines them).
// GREEN AT G3, on both GPUs (RTX 2080 SUPER and RX 7900 XTX), each run in its own process:
//   ceiling legs  (--m05, --m15) every check below; the log records each model's operative ceiling and
//                 every refusal's HRESULT and failing call;
//   seam legs     (--tiny) every check below;
//   leg Z (zero-fill Close refused; terminal) is selected by itself ("=11.4/G3:zerofill"), in a
//                 fresh process, because it leaves its context terminal.
// MUTANT (box-owed, 11.4 and CEIL-15): the create over 1.8.0's UploadResidentUavBufferSync (a
// full-size UPLOAD staging buffer through Device::Upload, then the DEFAULT pool), built with
// --gpu1p0=<mutant copy>: the staging leg below must read SSLM_DEVICE_LOST there, so the cell is red.
// R12-G3's mutants: (1) phase A without the rule-1 query -> leg 1 reads SSLM_GPU_ALLOCATION_FAILED;
// (2) 1.8.0's full-size Upload -> leg 2 reads an UPLOAD record of the pool's size; (3) one `try` over
// both phases -> leg Z reads SSLM_GPU_ALLOCATION_FAILED or a usable context.
#include "g1b_helpers.h"

namespace g1b {
namespace {

constexpr long kEInvalidArg = static_cast<long>(0x80070057L);
constexpr uint64_t kStagingCap = uint64_t{64} << 20;  // 64 MiB, §4 item 3 (rev 15)
constexpr uint32_t kHeapDefault = 1;                  // D3D12_HEAP_TYPE_DEFAULT
constexpr uint32_t kHeapUpload = 2;                   // D3D12_HEAP_TYPE_UPLOAD

// One create at `pages` on `ctx`, destroyed again if admitted, with this process's stderr captured
// so a refusal's HRESULT and failing call (SSLM_GPU_HR's "HR FAIL 0x... at file:line") are kept.
S CreateLogged(SslmGpuContext* ctx, SslmGpuModelHandle* model, uint32_t pages, std::string* log) {
	S st = S::SSLM_DEVICE_LOST;
	SslmGpuKvPoolHandle* p = nullptr;
	*log = CaptureStderr([&] { st = sslm_gpu_kv_pool_create(ctx, model, pages, &p); });
	if (st == S::SSLM_OK && p) sslm_gpu_kv_pool_destroy(ctx, p);
	G1B_CHECK_MSG(st == S::SSLM_OK || p == nullptr, "11.4: a refused create of %u pages left a non-null out-handle", pages);
	return st;
}

// "After each refusal a one-page create on the same context must return SSLM_OK, since phase A
// leaves the list Closed" (§4 item 3).
void OnePageAfterRefusal(SslmGpuContext* ctx, SslmGpuModelHandle* model, const char* after) {
	std::string log;
	G1B_CHECK_STATUS(CreateLogged(ctx, model, 1, &log), S::SSLM_OK, "11.4: one-page create after %s", after);
}

void LogRefusal(const char* model, uint32_t pages, const std::string& log) {
	std::printf("11.4 REFUSAL model=%s pages=%u status=SSLM_GPU_ALLOCATION_FAILED log:\n%s", model, pages,
	            log.empty() ? "  (nothing logged)\n" : log.c_str());
	// "each refusal's HRESULT and failing call logged" (CEIL-15): the library's own line.
	G1B_CHECK_MSG(!log.empty(), "11.4 %s at %u pages: the refusal logged no HRESULT or failing call", model, pages);
}

// ---- the ceiling legs, at each model (G3 part, rev 15.1) -------------------------------------------
// The largest page count whose byte size is at most 2^32 (the test's arithmetic: 43,690 at 0.5B and
// 18,724 at 1.5B, executed in the plan; 2^32 is a multiple of neither page size). One page more is
// refused SSLM_GPU_KV_POOL_TOO_LARGE. The create at the ceiling returns SSLM_OK or
// SSLM_GPU_ALLOCATION_FAILED, never SSLM_DEVICE_LOST on a live device and never a crash. On a
// refusal the operative ceiling is found by bisection on those two statuses only, with a one-page
// create after each refusal returning SSLM_OK, and each refusal's HRESULT and failing call logged.
void CeilingLeg(const char* name, const std::string& path, uint32_t plan_ceiling, uint64_t plan_page_bytes) {
	Artifact a;
	if (!a.Load(path)) {
		G1B_CHECK_MSG(false, "11.4 %s: artifact '%s' not supplied or unreadable (pass --%s=PATH)", name, path.c_str(),
		              plan_ceiling == 43690 ? "m05" : "m15");
		return;
	}
	const Geometry& g = a.geo;
	G1B_CHECK_MSG(g.page_bytes == plan_page_bytes && g.CeilingPages() == plan_ceiling,
	              "11.4 %s: page_bytes %llu (plan %llu), ceiling %u (plan %u): not the artifact the plan's figures are for", name,
	              static_cast<unsigned long long>(g.page_bytes), static_cast<unsigned long long>(plan_page_bytes),
	              g.CeilingPages(), plan_ceiling);
	G1B_CHECK_MSG((uint64_t{1} << 32) % g.page_bytes != 0, "11.4 %s: 2^32 is a multiple of the page size", name);
	SslmGpuContext* ctx = NewContext();
	SslmGpuModelHandle* model = ctx ? MapModel(ctx, a) : nullptr;
	if (!model) {
		G1B_CHECK_MSG(false, "11.4 %s: context or model map failed", name);
		if (ctx) sslm_gpu_context_destroy(ctx);
		return;
	}
	const uint32_t ceil = g.CeilingPages();
	{
		SslmGpuKvPoolHandle* p = reinterpret_cast<SslmGpuKvPoolHandle*>(uintptr_t{0x1});
		const S st = sslm_gpu_kv_pool_create(ctx, model, ceil + 1, &p);
		G1B_CHECK_STATUS(st, S::SSLM_GPU_KV_POOL_TOO_LARGE, "11.4 %s: ceiling + 1 (%u pages)", name, ceil + 1);
		G1B_CHECK_MSG(p == nullptr || st == S::SSLM_OK, "11.4 %s: ceiling + 1: out-handle not nulled", name);
		if (st == S::SSLM_OK && p) sslm_gpu_kv_pool_destroy(ctx, p);
	}
	std::string log;
	const S at = CreateLogged(ctx, model, ceil, &log);
	G1B_CHECK_MSG(at == S::SSLM_OK || at == S::SSLM_GPU_ALLOCATION_FAILED,
	              "11.4 %s: the create at the ceiling (%u pages) returned %s; only SSLM_OK or SSLM_GPU_ALLOCATION_FAILED pass, "
	              "and SSLM_DEVICE_LOST on a live device is not a bisection step",
	              name, ceil, StatusName(at));
	uint32_t operative = 0;
	if (at == S::SSLM_OK) {
		operative = ceil;
	} else if (at == S::SSLM_GPU_ALLOCATION_FAILED) {
		LogRefusal(name, ceil, log);
		OnePageAfterRefusal(ctx, model, "the ceiling refusal");
		// Invariant: lo admitted, hi refused. lo = 1 is admitted (the one-page create above).
		uint32_t lo = 1, hi = ceil;
		bool clean = true;
		while (hi - lo > 1 && clean) {
			const uint32_t mid = lo + (hi - lo) / 2;
			const S st = CreateLogged(ctx, model, mid, &log);
			if (st == S::SSLM_OK) {
				lo = mid;
			} else if (st == S::SSLM_GPU_ALLOCATION_FAILED) {
				hi = mid;
				LogRefusal(name, mid, log);
				OnePageAfterRefusal(ctx, model, "a bisection refusal");
			} else {
				G1B_CHECK_MSG(false, "11.4 %s: bisection at %u pages returned %s (only SSLM_OK and SSLM_GPU_ALLOCATION_FAILED are steps)",
				              name, mid, StatusName(st));
				clean = false;
			}
		}
		if (clean) operative = lo;
	}
	std::printf("11.4 OPERATIVE CEILING model=%s arithmetic=%u operative=%u (%.1f MiB)\n", name, ceil, operative,
	            static_cast<double>(operative) * static_cast<double>(g.page_bytes) / (1024.0 * 1024.0));
	sslm_gpu_model_unmap(ctx, model);
	sslm_gpu_context_destroy(ctx);
}

#if defined(SUPERSLM_GPU_ALLOC_FAULT_INJECTION)
// The occurrence of the pool create's UPLOAD staging buffer and of its DEFAULT pool buffer among the
// create's TryMakeBuffer calls, found on a clean create of `pages` through the size-recording seam.
// Leg 2 (R12-G3): exactly one UPLOAD allocation, of min(bytes, 64 MiB), and exactly one DEFAULT
// allocation, of exactly the pool's bytes.
struct Occurrences {
	uint32_t staging = 0, pool = 0;
};
Occurrences FindOccurrences(Env& e, uint32_t pages, const char* label) {
	Occurrences o;
	const uint64_t bytes = uint64_t{pages} * e.geo.page_bytes;
	SslmGpuAllocCounterResetForTest();
	SslmGpuKvPoolHandle* p = nullptr;
	G1B_CHECK_STATUS(sslm_gpu_kv_pool_create(e.ctx1, e.m1, pages, &p), S::SSLM_OK, "11.4 %s: clean create of %u pages", label, pages);
	const uint32_t n = SslmGpuAllocCountForTest();
	if (p) sslm_gpu_kv_pool_destroy(e.ctx1, p);
	uint32_t uploads = 0, defaults = 0;
	for (uint32_t k = 1; k <= n; ++k) {
		uint64_t b = 0;
		uint32_t heap = 0;
		if (!SslmGpuAllocRecordForTest(k, &b, &heap)) continue;
		std::printf("11.4 %s: allocation %u of %u: heap %u, %llu bytes\n", label, k, n, heap, static_cast<unsigned long long>(b));
		if (heap == kHeapUpload) {
			++uploads;
			o.staging = k;
			G1B_CHECK_MSG(b == std::min(bytes, kStagingCap), "11.4 leg 2 %s: the UPLOAD staging allocation is %llu bytes, want min(%llu, 64 MiB)",
			              label, static_cast<unsigned long long>(b), static_cast<unsigned long long>(bytes));
		} else if (heap == kHeapDefault && b == bytes) {
			++defaults;
			o.pool = k;
		}
	}
	G1B_CHECK_MSG(uploads == 1 && defaults == 1, "11.4 leg 2 %s: %u UPLOAD and %u pool-sized DEFAULT allocations, want one each", label,
	              uploads, defaults);
	G1B_CHECK_MSG(o.staging != 0 && o.pool != 0 && o.staging < o.pool,
	              "11.4 %s: phase A creates the staging buffer before the pool's DEFAULT buffer (staging %u, pool %u)", label, o.staging, o.pool);
	return o;
}

// Fault one occurrence of the create with E_INVALIDARG and read the status; the log must carry the
// injected HRESULT (SSLM_GPU_HR's line names the failing call's file and line).
S FaultedCreate(Env& e, uint32_t pages, uint32_t occurrence, bool removed_query, std::string* log) {
	SslmGpuAllocCounterResetForTest();
	ArmGpuAllocFaultAtOccurrence(occurrence, kEInvalidArg);
	if (removed_query) ArmGpuMapDeviceRemovedQueryInjection();
	S st = S::SSLM_OK;
	SslmGpuKvPoolHandle* p = reinterpret_cast<SslmGpuKvPoolHandle*>(uintptr_t{0x1});
	*log = CaptureStderr([&] { st = sslm_gpu_kv_pool_create(e.ctx1, e.m1, pages, &p); });
	ArmGpuAllocFaultAtOccurrence(0, 0);  // disarm, whatever fired
	G1B_CHECK_MSG(p == nullptr || st == S::SSLM_OK, "11.4: a faulted create left a non-null out-handle");
	if (st == S::SSLM_OK && p) sslm_gpu_kv_pool_destroy(e.ctx1, p);
	return st;
}

// The seam legs. The repaired create through the allocation-fault seam, E_INVALIDARG once at the
// staging buffer's create and once at the pool's: each reads SSLM_GPU_ALLOCATION_FAILED, logs the
// HRESULT, and a one-page create after it returns SSLM_OK. (The staging buffer's Map has no seam:
// covered by the probe's model only, §15.20.) R12-G3 leg 1: the removed-device query armed with the
// staging fault reads SSLM_DEVICE_LOST (rule 1 asked first). Leg 2: the allocation sizes, above and
// below 64 MiB, in FindOccurrences.
void SeamLegs(Env& e) {
	const uint32_t big = static_cast<uint32_t>((kStagingCap / e.geo.page_bytes) * 2);  // about 128 MiB of pool
	const uint32_t small = 100;                                                    // 600 KiB: below 64 MiB
	FindOccurrences(e, small, "below 64 MiB");
	const Occurrences o = FindOccurrences(e, big, "above 64 MiB");
	if (!o.staging || !o.pool) return;
	struct {
		const char* name;
		uint32_t occurrence;
	} legs[] = {{"staging buffer's create", o.staging}, {"pool buffer's create", o.pool}};
	for (auto& l : legs) {
		std::string log;
		G1B_CHECK_STATUS(FaultedCreate(e, big, l.occurrence, false, &log), S::SSLM_GPU_ALLOCATION_FAILED, "11.4 E_INVALIDARG at the %s", l.name);
		G1B_CHECK_MSG(log.find("0x80070057") != std::string::npos, "11.4 E_INVALIDARG at the %s: the HRESULT was not logged (log: %s)",
		              l.name, log.c_str());
		std::printf("11.4 SEAM %s log:\n%s", l.name, log.c_str());
		OnePageAfterRefusal(e.ctx1, e.m1, l.name);
	}
	std::string log;
	G1B_CHECK_STATUS(FaultedCreate(e, big, o.staging, true, &log), S::SSLM_DEVICE_LOST,
	                 "11.4 leg 1: removed-device query armed, E_INVALIDARG at the staging buffer's create");
}
#endif  // SUPERSLM_GPU_ALLOC_FAULT_INJECTION

void Cell114G3() {
	CeilingLeg("0.5B", GArgs().m05, 43690, 98304);
	CeilingLeg("1.5B", GArgs().m15, 18724, 229376);
#if defined(SUPERSLM_GPU_ALLOC_FAULT_INJECTION)
	Env& e = GetEnv();
	if (e.ok) SeamLegs(e);
#else
	G1B_CHECK_MSG(false, "11.4: built without SUPERSLM_GPU_ALLOC_FAULT_INJECTION: the seam legs cannot run, so the cell cannot pass");
#endif
}

// R12-G3 leg 3, in its own process: the zero-fill's Close refused (twice: the retry and the confirm
// step fail too, so the list is not confirmed Closed) reads SSLM_DEVICE_LOST and leaves the context
// terminal: the next call on it, a one-page create, reads SSLM_DEVICE_LOST too. Run on a context of
// its own, made here, so the shared environment's contexts are not the terminal one.
// (The plan's alternative trigger, the zero-fill's Signal failing without a retry through te425's
// Signal hook, is the same leg on another seam; this cell takes the Close seam.)
void Cell114G3ZeroFill() {
#if defined(SUPERSLM_GPU_ALLOC_FAULT_INJECTION)
	Env& e = GetEnv();
	if (!e.ok) return;
	SslmGpuContext* ctx = NewContext();
	SslmGpuModelHandle* model = ctx ? MapModel(ctx, e.tiny) : nullptr;
	if (!model) {
		G1B_CHECK_MSG(false, "11.4 leg 3: context or model map failed");
		return;
	}
	ArmGpuPoolZeroFillCloseFaultForTest(2);
	std::string log;
	G1B_CHECK_STATUS(CreateLogged(ctx, model, 64, &log), S::SSLM_DEVICE_LOST, "11.4 leg 3: the zero-fill's Close refused");
	ArmGpuPoolZeroFillCloseFaultForTest(0);
	G1B_CHECK_STATUS(CreateLogged(ctx, model, 1, &log), S::SSLM_DEVICE_LOST, "11.4 leg 3: the context is terminal after the stranded zero-fill");
	std::printf("11.4 leg 3 log:\n%s", log.c_str());
	// The terminal context is left as it is: its destroy is not this cell's subject.
#else
	G1B_CHECK_MSG(false, "11.4 leg 3: built without SUPERSLM_GPU_ALLOC_FAULT_INJECTION");
#endif
}

// review-g23 F2 (11.3's class at the GPU pooled release): HolderRelease consumes the holder whatever
// it returns, so a release that meets a refcount underflow must still finish -- the handle freed,
// the pool's live count dropped -- and report the underflow, as the CPU twin (ReleaseHolder) does.
// The seam zeroes the sequence's first reserve page's refcount; the release then reads
// SSLM_SEQUENCE_KV_BUFFER_MISMATCH, and the pool, with no sequence left, destroys SSLM_OK. Red before
// the fix: the release returned before dropping live_refs, so the destroy read
// SSLM_MODEL_HAS_LIVE_SEQUENCES for ever (and the handle kept a dangling holder).
void Cell113G3Release() {
#if defined(SUPERSLM_GPU_ALLOC_FAULT_INJECTION)
	Env& e = GetEnv();
	if (!e.ok) return;
	SslmGpuKvPoolHandle* pool = nullptr;
	G1B_CHECK_STATUS(sslm_gpu_kv_pool_create(e.ctx1, e.m1, 16, &pool), S::SSLM_OK, "11.3/G3 release: pool create");
	if (!pool) return;
	SslmGpuSequenceHandle* seq = nullptr;
	G1B_CHECK_STATUS(sslm_gpu_seq_create_budgeted(e.ctx1, e.m1, pool, 64, &seq), S::SSLM_OK,
	                 "11.3/G3 release: budgeted create");
	if (!seq) {
		sslm_gpu_kv_pool_destroy(e.ctx1, pool);
		return;
	}
	G1B_CHECK_MSG(ZeroGpuPooledSeqReservePageRefcountForTest(seq), "11.3/G3 release: the seam found no reserve page");
	G1B_CHECK_STATUS(sslm_gpu_seq_release(e.ctx1, seq), S::SSLM_SEQUENCE_KV_BUFFER_MISMATCH,
	                 "11.3/G3 release: the release meeting a refcount underflow");
	G1B_CHECK_STATUS(sslm_gpu_kv_pool_destroy(e.ctx1, pool), S::SSLM_OK,
	                 "11.3/G3 release: the pool's destroy after the underflowed release (the release finished)");
#else
	G1B_CHECK_MSG(false, "11.3/G3 release: built without SUPERSLM_GPU_ALLOC_FAULT_INJECTION");
#endif
}

}  // namespace

G1B_CELL("11.4/G3", "G3", Cell114G3);
G1B_CELL("11.4/G3:zerofill", "G3", Cell114G3ZeroFill);
G1B_CELL("11.3/G3:release", "G3", Cell113G3Release);

}  // namespace g1b

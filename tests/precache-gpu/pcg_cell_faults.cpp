// Precached-prefixes plan §7, step F1 (green at F2), the allocation-fault build (SUPERSLM_GPU_ALLOC_FAULT_INJECTION,
// SUPERSLM_GPU_TEST_SEAMS): the GPU restore's refusals after a draw.
//   5.3 [F2]   §4 item 3 step 12, through paged KV's fault seams: an allocation fault at the restore's
//              table buffer and at the pool upload's staging buffer -> SSLM_GPU_ALLOCATION_FAILED; the
//              staging fault with the removed-device query armed -> SSLM_DEVICE_LOST. Each: *out null,
//              every drawn page returned (fill probe), the pool still destroyable once the case's handles
//              are released, and the next restore succeeds.
//   5.11 [F2]  the GPU half (F0: the GPU share can refuse after the draw, as begin_from's does, and the
//              verb returns SSLM_SEQUENCE_KV_BUFFER_MISMATCH). The restore fault slot at its share
//              position, armed as an invalid argument and as a refcount underflow, on a nested blob with a
//              matching hint (s >= 1): SSLM_SEQUENCE_KV_BUFFER_MISMATCH, *out null, the slot's shared-entry
//              count 0, the free count unchanged, and the parent's pages back in the pool once the parent
//              is released. The map position (a root blob of L = 4B + 1): SSLM_GPU_KV_POOL_EXHAUSTED, the
//              same operands. The allocation position: SSLM_GPU_ALLOCATION_FAILED, shared-entry count s.
//              Every case ends with sslm_gpu_kv_pool_destroy returning SSLM_OK. 5.7's GPU pair for the
//              share refusal: armed at the share with a pool too short for the draw ->
//              SSLM_GPU_KV_POOL_EXHAUSTED and the slot never fires (the draw precedes the share).
//   The debug layer's clean reading for these cells is a box leg (SSLM_GPU_ENABLE_DEBUG_LAYER=1).
//
// RED AT F1: the save stub returns SSLM_DEVICE_LOST, so there is no blob; the fault slot stub never fires.
//
// Run: pcg_cell_faults.exe --fixture-dir=DIR (pkv_def).
#include "pcg_helpers.h"

using namespace pcg;

#if !defined(SUPERSLM_GPU_ALLOC_FAULT_INJECTION)
#error "pcg_cell_faults is built in the allocation-fault configuration"
#endif

namespace {

constexpr long kEOutOfMemory = static_cast<long>(0x8007000Eu);
constexpr long kEInvalidArg = static_cast<long>(0x80070057u);
constexpr uint32_t kHeapUpload = 2;   // D3D12_HEAP_TYPE_UPLOAD
constexpr uint32_t kHeapDefault = 1;  // D3D12_HEAP_TYPE_DEFAULT

struct Fresh : ProbeState {
	const Env& e;
	SslmGpuKvPoolHandle* pool = nullptr;
	std::vector<SslmGpuPrefixHandle*> prefixes;
	Fresh(const Env& env, uint32_t pages) : e(env) {
		const GS st = sslm_gpu_kv_pool_create(e.ctx, e.model(), pages, &pool);
		CHECK_MSG(st == GS::SSLM_OK, "pool of %u pages: %s", pages, StatusName(st));
	}
	SslmGpuKvPoolHandle* Pool() override { return pool; }
	SslmGpuContext* Ctx() override { return e.ctx; }
	SslmGpuModelHandle* Model() override { return e.model(); }
	void DrainAll() override {}
	// Releases every prefix, then destroys the pool; the destroy must return SSLM_OK (no live holder).
	void Finish(const char* what) {
		for (auto it = prefixes.rbegin(); it != prefixes.rend(); ++it) sslm_gpu_prefix_release(e.ctx, *it);
		prefixes.clear();
		if (pool) {
			const GS st = sslm_gpu_kv_pool_destroy(e.ctx, pool);
			CHECK_MSG(st == GS::SSLM_OK, "%s: sslm_gpu_kv_pool_destroy after every release: %s (a leaked holder?)", what,
			          StatusName(st));
			pool = nullptr;
		}
	}
	~Fresh() override { Finish("teardown"); }
};

GS Restore(Fresh& f, const std::vector<uint8_t>& blob, SslmGpuPrefixHandle* hint, SslmGpuPrefixHandle** raw) {
	SslmGpuPrefixHandle* out = OutSentinel<SslmGpuPrefixHandle>();
	const GS st = sslm_gpu_prefix_restore(f.e.ctx, f.e.model(), f.pool, blob.data(), blob.size(), hint, &out, nullptr);
	*raw = out;
	if (Real(out)) f.prefixes.push_back(out);
	return st;
}

SslmGpuPrefixHandle* Build(Fresh& f, const std::vector<int32_t>& t) {
	SslmGpuPrefixHandle* p = nullptr;
	GS st = sslm_gpu_prefix_begin(f.e.ctx, f.e.model(), f.pool, static_cast<int32_t>(t.size()), &p);
	if (st == GS::SSLM_OK) st = PrefixPrefillAll(f.e.ctx, p, t, 0, f.e.full_budget);
	if (st == GS::SSLM_OK) st = sslm_gpu_prefix_freeze(f.e.ctx, p);
	CHECK_MSG(st == GS::SSLM_OK, "prefix of %zu tokens: %s", t.size(), StatusName(st));
	if (Real(p)) f.prefixes.push_back(p);
	return st == GS::SSLM_OK ? p : nullptr;
}

// ---- 5.3 ---------------------------------------------------------------------------------------------

void Cell53(const Env& e, const std::vector<uint8_t>& root) {
	const int64_t P = e.geo.PrefixPages(Spx1Parse(root, e.geo).length);
	// The restore's allocations, from a clean run: the table buffer (DEFAULT) and the staging buffer (UPLOAD).
	uint32_t table_k = 0, staging_k = 0;
	{
		Fresh f(e, 16);
		SslmGpuAllocCounterResetForTest();
		SslmGpuPrefixHandle* raw = nullptr;
		CHECK(Restore(f, root, nullptr, &raw) == GS::SSLM_OK);
		const uint32_t n = SslmGpuAllocCountForTest();
		for (uint32_t k = 1; k <= n; ++k) {
			uint64_t bytes = 0;
			uint32_t heap = 0;
			if (!SslmGpuAllocRecordForTest(k, &bytes, &heap)) continue;
			if (heap == kHeapUpload && staging_k == 0) staging_k = k;
			if (heap == kHeapDefault && table_k == 0) table_k = k;
		}
		CHECK_MSG(staging_k != 0 && table_k != 0 && table_k < staging_k,
		          "5.3: the clean restore's allocations: table buffer at %u, staging at %u of %u", table_k, staging_k, n);
	}
	if (!staging_k || !table_k) return;
	struct Leg {
		const char* name;
		uint32_t k;
		long hr;
		bool removed;
		GS want;
	} legs[] = {
	    {"E_OUTOFMEMORY at the table buffer", table_k, kEOutOfMemory, false, GS::SSLM_GPU_ALLOCATION_FAILED},
	    {"E_OUTOFMEMORY at the upload's staging buffer", staging_k, kEOutOfMemory, false, GS::SSLM_GPU_ALLOCATION_FAILED},
	    {"E_INVALIDARG at the staging buffer, removed-device query armed", staging_k, kEInvalidArg, true, GS::SSLM_DEVICE_LOST},
	};
	for (const Leg& l : legs) {
		Fresh f(e, 16);
		SslmGpuAllocCounterResetForTest();
		ArmGpuAllocFaultAtOccurrence(l.k, l.hr);
		if (l.removed) ArmGpuMapDeviceRemovedQueryInjection();
		SslmGpuPrefixHandle* raw = nullptr;
		const GS st = Restore(f, root, nullptr, &raw);
		ArmGpuAllocFaultAtOccurrence(0, 0);
		CHECK_MSG(st == l.want && raw == nullptr, "5.3 %s: %s (want %s), out %s", l.name, Name(st), Name(l.want),
		          raw == nullptr ? "null" : "set");
		ProbeExactlyFreeOneState(e, f, 16, (std::string("5.3 ") + l.name + ": every drawn page returned").c_str());
		SslmGpuPrefixHandle* again = nullptr;
		CHECK_MSG(Restore(f, root, nullptr, &again) == GS::SSLM_OK, "5.3 %s: the next restore", l.name);
		(void)P;
		f.Finish((std::string("5.3 ") + l.name).c_str());
	}
}

// ---- 5.11's GPU half ------------------------------------------------------------------------------------

void Cell511(const Env& e) {
	const int64_t B = e.geo.B;
	const std::vector<int32_t> parent_t = PrefixTokens(static_cast<int32_t>(3 * B + 8), 41);  // 56: s = 3
	const std::vector<int32_t> child_t = PrefixTokens(12, 42);                                 // L = 68
	std::vector<uint8_t> nested, root;
	{
		Fresh f(e, 32);
		SslmGpuPrefixHandle* parent = Build(f, parent_t);
		SslmGpuPrefixHandle* child = nullptr;
		if (parent) {
			GS st = sslm_gpu_prefix_begin_from(e.ctx, parent, 12, &child);
			if (st == GS::SSLM_OK) {
				f.prefixes.push_back(child);
				st = PrefixPrefillAll(e.ctx, child, child_t, 0, e.full_budget);
			}
			if (st == GS::SSLM_OK) st = sslm_gpu_prefix_freeze(e.ctx, child);
			CHECK_MSG(st == GS::SSLM_OK, "5.11: the nested child: %s", StatusName(st));
			if (st == GS::SSLM_OK) nested = SavePrefix(e, child);
		}
		SslmGpuPrefixHandle* r = Build(f, PrefixTokens(static_cast<int32_t>(4 * B + 1), 43));  // L = 4B + 1
		if (r) root = SavePrefix(e, r);
	}
	if (nested.empty() || root.empty()) return;
	const int64_t L = Spx1Parse(nested, e.geo).length, s = (L - 12) / B;
	CHECK_MSG(s == 3, "5.11: the nested blob's origin gives s = %lld", static_cast<long long>(s));
	const int64_t parent_pages = e.geo.PrefixPages(static_cast<int64_t>(parent_t.size()));
	const uint32_t pool_pages = 32;
	// The share position, both refusals.
	for (int refusal : {2, 3}) {
		Fresh f(e, pool_pages);
		SslmGpuPrefixHandle* parent = nullptr;
		SslmGpuPrefixHandle* raw = nullptr;
		{
			// The parent restored from its own save of the same tokens: a matching hint (its rows are the blob's).
			Fresh tmp(e, 16);
			SslmGpuPrefixHandle* p0 = Build(tmp, parent_t);
			const std::vector<uint8_t> pb = p0 ? SavePrefix(e, p0) : std::vector<uint8_t>{};
			tmp.Finish("5.11 parent source");
			if (pb.empty() || Restore(f, pb, nullptr, &parent) != GS::SSLM_OK) {
				CHECK_MSG(false, "5.11: the parent");
				continue;
			}
		}
		const int64_t free_before = pool_pages - parent_pages;
		ArmGpuPrefixRestoreFaultForTest(1, refusal);
		const GS st = Restore(f, nested, parent, &raw);
		uint32_t shared_at = 99;
		const bool fired = GpuPrefixRestoreFaultFiredForTest(&shared_at);
		ArmGpuPrefixRestoreFaultForTest(0, 0);
		CHECK_MSG(fired && st == GS::SSLM_SEQUENCE_KV_BUFFER_MISMATCH && raw == nullptr && shared_at == 0,
		          "5.11 share (%s): fired %d, %s, out %s, shared entries at the fault %u",
		          refusal == 2 ? "invalid argument" : "refcount underflow", fired ? 1 : 0, Name(st), raw ? "set" : "null", shared_at);
		ProbeExactlyFreeOneState(e, f, free_before, "5.11 share: the free count after equals before");
		// The parent's refcounts unchanged: once it is released, its pages return and the pool reads whole.
		f.prefixes.pop_back();
		CHECK(sslm_gpu_prefix_release(e.ctx, parent) == GS::SSLM_OK);
		ProbeExactlyFreeOneState(e, f, pool_pages, "5.11 share: the parent's pages back once it is released");
		f.Finish("5.11 share");
	}
	// The map position: a root blob of L = 4B + 1 (5 pages).
	{
		Fresh f(e, pool_pages);
		SslmGpuPrefixHandle* raw = nullptr;
		ArmGpuPrefixRestoreFaultForTest(2, 0);
		const GS st = Restore(f, root, nullptr, &raw);
		const bool fired = GpuPrefixRestoreFaultFiredForTest(nullptr);
		ArmGpuPrefixRestoreFaultForTest(0, 0);
		CHECK_MSG(fired && st == GS::SSLM_GPU_KV_POOL_EXHAUSTED && raw == nullptr, "5.11 map: fired %d, %s", fired ? 1 : 0,
		          Name(st));
		ProbeExactlyFreeOneState(e, f, pool_pages, "5.11 map: the free count after equals before");
		f.Finish("5.11 map");
	}
	// The allocation position, nested with a matching hint: the slot records s shared entries.
	{
		Fresh f(e, pool_pages);
		Fresh tmp(e, 16);
		SslmGpuPrefixHandle* p0 = Build(tmp, parent_t);
		const std::vector<uint8_t> pb = p0 ? SavePrefix(e, p0) : std::vector<uint8_t>{};
		tmp.Finish("5.11 parent source");
		SslmGpuPrefixHandle* parent = nullptr;
		if (!pb.empty() && Restore(f, pb, nullptr, &parent) == GS::SSLM_OK) {
			SslmGpuPrefixHandle* raw = nullptr;
			ArmGpuPrefixRestoreFaultForTest(3, 0);
			const GS st = Restore(f, nested, parent, &raw);
			uint32_t shared_at = 99;
			const bool fired = GpuPrefixRestoreFaultFiredForTest(&shared_at);
			ArmGpuPrefixRestoreFaultForTest(0, 0);
			CHECK_MSG(fired && st == GS::SSLM_GPU_ALLOCATION_FAILED && raw == nullptr && shared_at == s,
			          "5.11 allocation: fired %d, %s, shared entries at the fault %u (want %lld)", fired ? 1 : 0, Name(st),
			          shared_at, static_cast<long long>(s));
			ProbeExactlyFreeOneState(e, f, pool_pages - parent_pages, "5.11 allocation: the free count after equals before");
			f.prefixes.pop_back();
			CHECK(sslm_gpu_prefix_release(e.ctx, parent) == GS::SSLM_OK);
			ProbeExactlyFreeOneState(e, f, pool_pages, "5.11 allocation: the parent's pages back once it is released");
		}
		f.Finish("5.11 allocation");
	}
	// 5.7's GPU pair for the share refusal: armed at the share, a pool too short for the draw -> the draw's
	// status, and the slot never fires.
	{
		Fresh f(e, static_cast<uint32_t>(parent_pages + 1));
		Fresh tmp(e, 16);
		SslmGpuPrefixHandle* p0 = Build(tmp, parent_t);
		const std::vector<uint8_t> pb = p0 ? SavePrefix(e, p0) : std::vector<uint8_t>{};
		tmp.Finish("5.7 parent source");
		SslmGpuPrefixHandle* parent = nullptr;
		if (!pb.empty() && Restore(f, pb, nullptr, &parent) == GS::SSLM_OK) {
			SslmGpuPrefixHandle* raw = nullptr;
			ArmGpuPrefixRestoreFaultForTest(1, 2);
			const GS st = Restore(f, nested, parent, &raw);  // needs ceil(68/16) - 3 = 2 pages, 1 free
			const bool fired = GpuPrefixRestoreFaultFiredForTest(nullptr);
			ArmGpuPrefixRestoreFaultForTest(0, 0);
			CHECK_MSG(!fired && st == GS::SSLM_GPU_KV_POOL_EXHAUSTED && raw == nullptr,
			          "5.7 GPU: share refusal + exhausted pool: fired %d, %s (want SSLM_GPU_KV_POOL_EXHAUSTED, slot not fired)",
			          fired ? 1 : 0, Name(st));
		}
		f.Finish("5.7 share pair");
	}
}

}  // namespace

int main(int argc, char** argv) {
	ParseFixtureArgs(argc, argv);
	{
		Env e;
		if (e.Open(FixturePath(argc, argv))) {
			std::vector<uint8_t> root;
			{
				Fresh f(e, 16);
				SslmGpuPrefixHandle* p = Build(f, PrefixTokens(40, 51));
				if (p) root = SavePrefix(e, p);
			}
			if (!root.empty()) Cell53(e, root);
			Cell511(e);
		}
	}
	return FinishSuite("pcg_cell_faults");
}

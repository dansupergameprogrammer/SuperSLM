// Paged-KV plan (rev 16.1) step G3, host-side cell for the GPU page pool's arithmetic and its use of
// the kv_pages module (§3.1, §3.4, §4 items 3, 6a, 6b and 7). CPU-only: it runs on every leg, with no
// device, against src/gpu/gpu_kv_pool_geometry.h (the D3D12-free header gpu_1p0.cpp and
// superslm_gpu.cpp use) and src/kv_pages.h.
//
// What it checks:
//   P1  the page geometry at the GPU's int8 element: the plan's executed figures (0.5B page 96 KiB,
//       ceiling 43,690 pages; 1.5B page 224 KiB, ceiling 18,724; the tiny fixture's 6,144-byte page,
//       ceiling + 1 = 699,051), B = cap when kv_block_size does not divide it, refusals for zero dims;
//   P2  the size refusal: the largest admissible count passes, one more and UINT32_MAX refuse, and a
//       page above 2^32 refuses any nonzero count;
//   P3  R(budget) = min(ceil(budget / B) + 1, ceil(cap / B)), the budget domain [1, cap] and the limit;
//   P4  the zero-fill's staging size (min(bytes, 64 MiB)) and its chunks cover [0, bytes) exactly once,
//       in order, each at most the staging size (64 chunks at the 0.5B ceiling);
//   P5  the coverage twin: its single-row form is the CPU's `context_length / B < mapped`, its chunk
//       form refuses exactly when a written row lies past the mapped pages, and a legacy one-page
//       view (B = cap, one page) is never refused below the cap, whatever the chunk length;
//   P6  §4 item 6b on the module the GPU pool reuses: over one base (the pool's host mirror), a
//       holder's unmap-all and release never change a neighbour's pages, and nothing zero-fills.
#include "gpu/gpu_kv_pool_geometry.h"
#include "kv_pages.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <vector>

namespace {
int g_checks = 0, g_failures = 0;
#define CHECK(c)                                                                  \
	do {                                                                          \
		++g_checks;                                                               \
		if (!(c)) {                                                               \
			++g_failures;                                                         \
			std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #c);              \
		}                                                                         \
	} while (0)

using namespace superslm_gpu::kv_pool;

void P1Geometry() {
	const GpuPageGeometry m05 = ComputeGpuPageGeometry(24, 2, 64, 4096, 16);
	CHECK(m05.ok && m05.page_positions == 16 && m05.page_bytes == 98304 && m05.cap_pages == 256);
	CHECK(LargestAdmissiblePageCount(m05.page_bytes) == 43690);
	const GpuPageGeometry m15 = ComputeGpuPageGeometry(28, 2, 128, 4096, 16);
	CHECK(m15.ok && m15.page_bytes == 229376 && LargestAdmissiblePageCount(m15.page_bytes) == 18724);
	const GpuPageGeometry tiny = ComputeGpuPageGeometry(2, 2, 48, 4096, 16);  // 2*2*2*16*48 = 6144
	CHECK(tiny.ok && tiny.page_bytes == 6144 && LargestAdmissiblePageCount(tiny.page_bytes) + 1 == 699051);
	const GpuPageGeometry nodiv = ComputeGpuPageGeometry(2, 2, 48, 100, 16);  // 16 does not divide 100
	CHECK(nodiv.ok && nodiv.page_positions == 100 && nodiv.cap_pages == 1);
	const GpuPageGeometry zero_b = ComputeGpuPageGeometry(2, 2, 48, 64, 0);
	CHECK(zero_b.ok && zero_b.page_positions == 64);
	CHECK(!ComputeGpuPageGeometry(0, 2, 48, 64, 16).ok);
	CHECK(!ComputeGpuPageGeometry(2, 0, 48, 64, 16).ok);
	CHECK(!ComputeGpuPageGeometry(2, 2, 0, 64, 16).ok);
	CHECK(!ComputeGpuPageGeometry(2, 2, 48, 0, 16).ok);
	CHECK(!ComputeGpuPageGeometry(2, 2, 48, int64_t{1} << 40, 16).ok);
}

void P2SizeRefusal() {
	for (size_t pb : {size_t{98304}, size_t{229376}, size_t{6144}, size_t{1}, size_t{4096}}) {
		const uint32_t ceil = LargestAdmissiblePageCount(pb);
		CHECK(!PoolBytesTooLarge(ceil, pb));
		if (ceil < std::numeric_limits<uint32_t>::max()) CHECK(PoolBytesTooLarge(ceil + 1, pb));
		CHECK(PoolBytesTooLarge(std::numeric_limits<uint32_t>::max(), pb) == (pb > 1));
		CHECK(!PoolBytesTooLarge(1, pb));
	}
	const size_t huge = static_cast<size_t>((uint64_t{1} << 32) + 1);
	CHECK(PoolBytesTooLarge(1, huge) && !PoolBytesTooLarge(0, huge));
	CHECK(!PoolBytesTooLarge(1, static_cast<size_t>(uint64_t{1} << 32)));
}

void P3Reservation() {
	const GpuPageGeometry g = ComputeGpuPageGeometry(2, 2, 48, 4096, 16);
	CHECK(ReservationPages(100, g, 4096) == 8);
	CHECK(ReservationPages(64, g, 4096) == 5);
	CHECK(ReservationPages(16, g, 4096) == 2);
	CHECK(ReservationPages(1, g, 4096) == 2);
	CHECK(ReservationPages(4096, g, 4096) == 256);
	CHECK(ReservationPages(4090, g, 4096) == 256);
	CHECK(ReservationPages(0, g, 4096) == 0 && ReservationPages(4097, g, 4096) == 0);
	CHECK(BudgetInDomain(1, 4096) && BudgetInDomain(4096, 4096));
	CHECK(!BudgetInDomain(0, 4096) && !BudgetInDomain(-1, 4096) && !BudgetInDomain(4097, 4096));
	CHECK(HolderLimit(0, 100, 4096) == 100 && HolderLimit(4000, 200, 4096) == 4096);
	CHECK(PagesToCover(0, 16) == 0 && PagesToCover(1, 16) == 1 && PagesToCover(16, 16) == 1 && PagesToCover(17, 16) == 2);
}

void P4ZeroFillChunks() {
	const uint64_t cap = uint64_t{64} << 20;
	const uint64_t sizes[] = {1, 6144, cap - 1, cap, cap + 1, 3 * cap + 7, uint64_t{43690} * 98304};
	for (uint64_t bytes : sizes) {
		const uint64_t s = ZeroFillStagingBytes(bytes);
		CHECK(s == (bytes < cap ? bytes : cap));
		const uint64_t n = ZeroFillChunkCount(bytes);
		CHECK(n == (bytes + s - 1) / s);
		uint64_t next = 0;
		bool ok = true;
		for (uint64_t i = 0; i < n; ++i) {
			uint64_t off = 0, len = 0;
			ZeroFillChunk(bytes, i, &off, &len);
			ok = ok && off == next && len > 0 && len <= s;
			next = off + len;
		}
		CHECK(ok && next == bytes);
	}
	CHECK(ZeroFillChunkCount(uint64_t{43690} * 98304) == 64);
	CHECK(ZeroFillChunkCount(0) == 0);
}

void P5Coverage() {
	// Single-row form against the CPU's.
	for (int64_t B : {1, 4, 16}) {
		for (uint32_t mapped = 0; mapped <= 4; ++mapped) {
			for (int64_t c = 0; c < 80; ++c) {
				const bool cpu = static_cast<uint64_t>(c) / static_cast<uint64_t>(B) >= mapped;
				CHECK(CoverageRefused(c, 1, 4096, B, mapped) == cpu);
			}
		}
	}
	// Chunk form: refused iff some written row r in [c, min(c + n, cap)) has r / B >= mapped.
	bool ok = true;
	for (int64_t cap : {int64_t{40}, int64_t{4096}}) {
		for (int64_t B : {1, 4, 16}) {
			for (uint32_t mapped = 1; mapped <= 3; ++mapped) {
				for (int64_t c = 0; c < 40 && c < cap; ++c) {
					for (uint32_t n = 1; n <= 40; ++n) {
						bool want = false;
						for (int64_t r = c; r < c + n && r < cap; ++r) want = want || (r / B >= mapped);
						ok = ok && CoverageRefused(c, n, cap, B, mapped) == want;
					}
				}
			}
		}
	}
	CHECK(ok);
	// The legacy one-page view: B = cap, one page, never refused below the cap.
	bool legacy = true;
	for (int64_t cap : {int64_t{1}, int64_t{17}, int64_t{4096}}) {
		for (int64_t c = 0; c < cap && c < 64; ++c) {
			for (uint32_t n : {1u, 4u, 1000u, 0xFFFFFFFFu}) legacy = legacy && !CoverageRefused(c, n, cap, cap, 1);
		}
	}
	CHECK(legacy);
}

void P6NeighbourInvariance() {
	using namespace superslm::kv_pages;
	const uint32_t pages = 12;
	const size_t page_bytes = 64;
	std::vector<uint8_t> base(pages * page_bytes, 0);  // the pool's host mirror, zeroed once at create
	PagePool* pool = nullptr;
	CHECK(PoolCreate(base.data(), pages, page_bytes, 4, &pool) == PageStatus::kOk);
	for (uint8_t b : base) CHECK(b == 0);  // create touches nothing
	PageHolder* a = nullptr;
	PageHolder* n = nullptr;
	CHECK(HolderCreate(pool, 8, 4, &a) == PageStatus::kOk);
	CHECK(HolderCreate(pool, 8, 4, &n) == PageStatus::kOk);
	for (int i = 0; i < 3; ++i) CHECK(HolderMapNext(a) == PageStatus::kOk);
	for (int i = 0; i < 3; ++i) CHECK(HolderMapNext(n) == PageStatus::kOk);
	for (uint32_t i = 0; i < 3; ++i) {
		std::memset(PageBase(pool, a->table[i]), 0xA0 + i, page_bytes);
		std::memset(PageBase(pool, n->table[i]), 0x50 + i, page_bytes);
	}
	const std::vector<uint8_t> before = base;
	auto neighbour_unchanged = [&] {
		bool same = true;
		for (uint32_t i = 0; i < 3; ++i) {
			const size_t off = static_cast<size_t>(n->table[i]) * page_bytes;
			same = same && std::memcmp(base.data() + off, before.data() + off, page_bytes) == 0;
		}
		return same;
	};
	// A pooled reset: unmap-all returns A's private pages to its own reserve and fills nothing.
	CHECK(HolderUnmapAll(a) == PageStatus::kOk);
	CHECK(base == before);
	CHECK(HolderMapped(a) == 0 && HolderReserve(a) == 4);
	// A pooled release: A's pages go back to the pool (poisoned iff dirty, never zeroed); N's rows stay.
	for (int i = 0; i < 2; ++i) CHECK(HolderMapNext(a) == PageStatus::kOk);
	CHECK(HolderRelease(a) == PageStatus::kOk);
	CHECK(neighbour_unchanged());
	// Every byte the release changed is the 0xCD poison: nothing zero-fills, nothing else is written.
	bool only_poison = true;
	for (size_t i = 0; i < base.size(); ++i) only_poison = only_poison && (base[i] == before[i] || base[i] == 0xCD);
	CHECK(only_poison && base != before);
	CHECK(HolderRelease(n) == PageStatus::kOk);
	uint32_t free_pages = 0, shared = 0;
	PoolCounts(pool, &free_pages, &shared);
	CHECK(free_pages == pages && shared == 0);
	PoolDestroy(pool);
}
}  // namespace

int main() {
	P1Geometry();
	P2SizeRefusal();
	P3Reservation();
	P4ZeroFillChunks();
	P5Coverage();
	P6NeighbourInvariance();
	std::printf("gpu_kv_pool_geometry_tests: %d checks, %d failures\n", g_checks, g_failures);
	return g_failures == 0 ? 0 : 1;
}

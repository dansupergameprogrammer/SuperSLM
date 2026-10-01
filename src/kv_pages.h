// Paged-KV plan (rev 16.1) §3.3, step C3: the page allocator -- one module, both backends.
//
// A pool is `page_count` pages of `page_bytes` bytes over caller memory, with a heap free list built
// once at create, a per-page `refcount` and `dirty` array, and the pool mutex. A holder (sequence or
// prefix) owns a page table of `table_entries` u32 entries allocated uninitialized, a `mapped` count,
// a private reserve stack, the `shared` count and the `materialized` count.
//
// The refcount law (§3.3's table): every read or write of `refcount` happens under the pool mutex,
// and the hot path (HolderMapNext: reserve -> table) never takes it and allocates nothing.
//
// What "shared" means is provenance, never the refcount: the leading `shared` entries of a table are
// the ones that entered it by sharing (at most floor(origin / B) of them; a restore without sharing
// and a private nested prefix have origin > 0 and fewer or none). They are followed by the
// `materialized` entries (a restore without sharing, §3.7), then by the holder's private pages.
// Unmapping a shared or materialized entry decrements it and frees it to the pool at 0; unmapping a
// private entry returns it to the holder's own reserve, refcount unchanged.
//
// Freeing a page: it has left every table; outside the lock it is poison-filled with 0xCD iff
// `dirty` (mapped since its last draw), `dirty` is cleared, and it is pushed under the lock. There
// is no zero fill anywhere.
//
// The test-facing surface is tests/paged-kv/pkv_module_api.h, which declares the same functions;
// the two must stay identical. The table-sentinel seam (cell 7.13) and the double-unmap seam (cell
// 11.3) are compiled only with SUPERSLM_ENABLE_KV_PAGES_TEST_SEAMS (the test libraries), so the
// production library's address read is a plain load.

#ifndef SUPERSLM_SRC_KV_PAGES_H
#define SUPERSLM_SRC_KV_PAGES_H

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <vector>

namespace superslm {
namespace kv_pages {

enum class PageStatus : int {
	kOk = 0,
	kExhausted = 1,         // the pool (or, for a map, the holder's reserve) has too few pages
	kInvalidArgument = 2,
	kRefcountUnderflow = 3, // §3.3: the release-build underflow check (cell 11.3)
	kAllocationFailed = 4,
};

// The value no page index takes (PoolCreate refuses a page_count that would reach it); the
// sentinel seam writes it into table entries at or above `mapped`.
constexpr uint32_t kNoPage = 0xFFFFFFFFu;

// The pool mutex, counting its acquisitions (§3.6's hot-path rule, cell 7.10: the ABI's test seam
// reads the count to show that prefill and decode never take it). One relaxed atomic increment per
// acquisition, and only lifecycle verbs acquire it.
class CountingMutex {
public:
	void lock() {
		m_.lock();
		acquisitions_.fetch_add(1, std::memory_order_relaxed);
	}
	bool try_lock() {
		if (!m_.try_lock()) return false;
		acquisitions_.fetch_add(1, std::memory_order_relaxed);
		return true;
	}
	void unlock() { m_.unlock(); }
	uint64_t acquisitions() const { return acquisitions_.load(std::memory_order_relaxed); }

private:
	std::mutex m_;
	std::atomic<uint64_t> acquisitions_{0};
};

struct PagePool {
	uint8_t* base = nullptr;
	uint32_t page_count = 0;
	size_t page_bytes = 0;
	int64_t page_positions = 0;  // B
	CountingMutex mutex;         // the pool mutex: guards free_list and refcount
	std::vector<uint32_t> free_list;  // stack; capacity page_count from create, so a push never allocates
	std::vector<uint32_t> refcount;
	// Written without the pool mutex by the holder that owns the page (reserve -> table), read and
	// cleared on the freeing path once the page has left every table. Ordered by the pool mutex.
	std::vector<uint8_t> dirty;
};

struct PageHolder {
	PagePool* pool = nullptr;
	uint32_t table_entries = 0;
	std::unique_ptr<uint32_t[]> table;  // uninitialized: entries at or above `mapped` are never read
	uint32_t mapped = 0;
	uint32_t shared = 0;        // leading entries that entered by sharing (provenance)
	uint32_t materialized = 0;  // entries after the shared ones, drawn beyond the reserve (§3.7)
	// Stack of the holder's own pages (refcount 1 each). Capacity is fixed at create to the pages it
	// can ever hold, so returning a private page to it never allocates.
	std::vector<uint32_t> reserve;
};

PageStatus PoolCreate(uint8_t* base, uint32_t page_count, size_t page_bytes, int64_t page_positions,
                      PagePool** out);
void PoolDestroy(PagePool* pool);
uint32_t PoolFreePages(PagePool* pool);
uint32_t PoolRefcount(PagePool* pool, uint32_t page);

PageStatus HolderCreate(PagePool* pool, uint32_t table_entries, uint32_t reserve_pages, PageHolder** out);
PageStatus HolderMapNext(PageHolder* holder);
PageStatus HolderShareLeading(PageHolder* dst, const PageHolder* src, uint32_t pages);
PageStatus HolderUnmapAll(PageHolder* holder);
PageStatus HolderRelease(PageHolder* holder);
// Every page of the holder's private reserve back to the pool (refcount := 0, poisoned iff dirty),
// its table untouched: a legacy prefix's first freeze (§3.4), and the ABI's reserve-drain seam
// (cell 11.2). *out_returned (optional) receives the count. Allocates nothing.
PageStatus HolderReturnReserve(PageHolder* holder, uint32_t* out_returned);

uint32_t HolderMapped(const PageHolder* holder);
uint32_t HolderShared(const PageHolder* holder);
uint32_t HolderReserve(const PageHolder* holder);

uint32_t HolderTableEntryForAddress(const PageHolder* holder, uint32_t i);

// The first byte of pool page `page` (no bounds check: callers pass a page a table maps).
inline uint8_t* PageBase(const PagePool* pool, uint32_t page) { return pool->base + size_t{page} * pool->page_bytes; }

// Acquisitions of `pool`'s mutex since PoolCreate (the ABI's cell-7.10 seam reads it).
uint64_t PoolMutexAcquisitions(const PagePool* pool);

#ifdef SUPERSLM_ENABLE_KV_PAGES_TEST_SEAMS
void SetTableSentinelForTest(bool enabled);
PageStatus DoubleUnmapForTest(PageHolder* holder, uint32_t i);
#endif

}  // namespace kv_pages
}  // namespace superslm

#endif  // SUPERSLM_SRC_KV_PAGES_H

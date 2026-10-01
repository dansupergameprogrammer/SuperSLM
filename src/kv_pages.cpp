// Paged-KV plan (rev 16.1) §3.3, step C3: the page allocator. See kv_pages.h for the state and laws.

#include "kv_pages.h"

#include <atomic>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <new>
#include <utility>

namespace superslm {
namespace kv_pages {

namespace {

constexpr uint8_t kPoison = 0xCD;

#ifdef SUPERSLM_ENABLE_KV_PAGES_TEST_SEAMS
std::atomic<bool> g_table_sentinel{false};
bool TableSentinelOn() { return g_table_sentinel.load(std::memory_order_relaxed); }
#endif

// One decrement of `page`'s refcount, caller holding the pool mutex. The underflow check is a real
// check in every build (§3.3, cell 11.3): a count already at 0 is left at 0, nothing is freed, and
// the caller reports kRefcountUnderflow. `*reached_zero` says the caller now owns freeing the page.
PageStatus DecrementLocked(PagePool* pool, uint32_t page, bool* reached_zero) {
	*reached_zero = false;
	uint32_t& rc = pool->refcount[page];
	if (rc == 0) return PageStatus::kRefcountUnderflow;
	rc -= 1;
	*reached_zero = rc == 0;
	return PageStatus::kOk;
}

// The fill half of freeing, outside the lock: the pages have left every table and nothing names
// them, so this holder alone touches their bytes and `dirty` flags. Each dirty page is filled whole
// and each clean page is left as it is; a run of consecutive list entries that are dirty and
// physically adjacent (ascending or descending, as a holder's table maps a fresh reserve) is filled
// by one memset over the run, the same bytes as one memset per page, so a full-length release
// writes the holder's pages as one stream rather than page_count short ones.
void PoisonIfDirty(PagePool* pool, const uint32_t* pages, size_t n) {
	size_t i = 0;
	while (i < n) {
		const uint32_t p = pages[i];
		if (!pool->dirty[p]) {
			++i;
			continue;
		}
		uint32_t lo = p, hi = p;
		int dir = 0;  // +1 ascending, -1 descending, 0 not yet known
		size_t j = i + 1;
		for (; j < n; ++j) {
			const uint32_t q = pages[j];
			if (!pool->dirty[q]) break;
			if (dir >= 0 && q == hi + 1) {
				hi = q;
				dir = 1;
			} else if (dir <= 0 && lo > 0 && q == lo - 1) {
				lo = q;
				dir = -1;
			} else {
				break;
			}
		}
		std::memset(pool->base + size_t{lo} * pool->page_bytes, kPoison, size_t{hi - lo + 1} * pool->page_bytes);
		for (size_t k = i; k < j; ++k) pool->dirty[pages[k]] = 0;
		i = j;
	}
}

// The push half of freeing, under the lock. free_list's capacity is page_count, and a page is on
// it at most once, so the push never allocates.
void PushFreeLocked(PagePool* pool, const uint32_t* pages, size_t n) {
	for (size_t i = 0; i < n; ++i) pool->free_list.push_back(pages[i]);
}

PageStatus Worse(PageStatus a, PageStatus b) { return a != PageStatus::kOk ? a : b; }

// A shared entry entering (+1) or leaving (-1) one table, caller holding the pool mutex: the
// per-page count and the distinct-page count sslm_kv_pool_stats reports (§3.6, provenance).
void SharedMapEnterLocked(PagePool* pool, uint32_t page) {
	if (pool->shared_maps[page]++ == 0) ++pool->shared_distinct;
}
void SharedMapLeaveLocked(PagePool* pool, uint32_t page) {
	uint32_t& n = pool->shared_maps[page];
	if (n == 0) return;  // never entered: nothing to count down (the underflow check is the refcount's)
	if (--n == 0) --pool->shared_distinct;
}

// The refusal half of a share, checked before any change: a mapped page at 0 is a count already
// lost (the underflow class); one at the maximum cannot take another reference.
PageStatus CheckShareableLocked(const PagePool* pool, const PageHolder* src, uint32_t pages) {
	for (uint32_t i = 0; i < pages; ++i) {
		const uint32_t rc = pool->refcount[src->table[i]];
		if (rc == 0) return PageStatus::kRefcountUnderflow;
		if (rc == std::numeric_limits<uint32_t>::max()) return PageStatus::kInvalidArgument;
	}
	return PageStatus::kOk;
}

}  // namespace

PageStatus PoolCreate(uint8_t* base, uint32_t page_count, size_t page_bytes, int64_t page_positions,
                      PagePool** out) {
	if (!out) return PageStatus::kInvalidArgument;
	*out = nullptr;
	// page_count < kNoPage keeps every page index distinct from the sentinel.
	if (!base || page_count == 0 || page_count == kNoPage || page_bytes == 0 || page_positions <= 0)
		return PageStatus::kInvalidArgument;
	if (page_bytes > std::numeric_limits<size_t>::max() / page_count) return PageStatus::kInvalidArgument;
	try {
		std::unique_ptr<PagePool> pool(new PagePool);
		pool->base = base;
		pool->page_count = page_count;
		pool->page_bytes = page_bytes;
		pool->page_positions = page_positions;
		pool->refcount.assign(page_count, 0);
		pool->dirty.assign(page_count, 0);
		pool->shared_maps.assign(page_count, 0);
		// Built once, as a stack that hands out page 0 first. No fill of the pool's memory.
		pool->free_list.reserve(page_count);
		for (uint32_t i = page_count; i-- > 0;) pool->free_list.push_back(i);
		*out = pool.release();
		return PageStatus::kOk;
	} catch (const std::bad_alloc&) {
		return PageStatus::kAllocationFailed;
	}
}

void PoolDestroy(PagePool* pool) { delete pool; }

uint32_t PoolFreePages(PagePool* pool) {
	if (!pool) return 0;
	std::lock_guard<CountingMutex> lock(pool->mutex);
	return static_cast<uint32_t>(pool->free_list.size());
}

uint32_t PoolRefcount(PagePool* pool, uint32_t page) {
	if (!pool || page >= pool->page_count) return 0;
	std::lock_guard<CountingMutex> lock(pool->mutex);
	return pool->refcount[page];
}

PageStatus HolderCreate(PagePool* pool, uint32_t table_entries, uint32_t reserve_pages, PageHolder** out) {
	if (!out) return PageStatus::kInvalidArgument;
	*out = nullptr;
	if (!pool || table_entries == 0) return PageStatus::kInvalidArgument;
	if (reserve_pages > pool->page_count) return PageStatus::kExhausted;
	std::unique_ptr<PageHolder> h;
	try {
		h.reset(new PageHolder);
		// Uninitialized: create fills nothing, so its time does not scale with the cap.
		h->table.reset(new uint32_t[table_entries]);
		h->reserve.reserve(reserve_pages);
	} catch (const std::bad_alloc&) {
		return PageStatus::kAllocationFailed;
	}
	h->pool = pool;
	h->table_entries = table_entries;
#ifdef SUPERSLM_ENABLE_KV_PAGES_TEST_SEAMS
	if (TableSentinelOn())
		for (uint32_t i = 0; i < table_entries; ++i) h->table[i] = kNoPage;
#endif
	{
		std::lock_guard<CountingMutex> lock(pool->mutex);
		if (pool->free_list.size() < reserve_pages) return PageStatus::kExhausted;  // nothing drawn
		// pool -> reserve: refcount := 1.
		for (uint32_t i = 0; i < reserve_pages; ++i) {
			const uint32_t p = pool->free_list.back();
			pool->free_list.pop_back();
			pool->refcount[p] = 1;
			h->reserve.push_back(p);  // within the capacity reserved above
		}
	}
	*out = h.release();
	return PageStatus::kOk;
}

// reserve -> table. The hot path: no pool mutex, no allocation, refcount untouched; sets `dirty`.
// Guarded by the holder's own lifecycle (one caller per holder at a time).
PageStatus HolderMapNext(PageHolder* holder) {
	if (!holder) return PageStatus::kInvalidArgument;
	if (holder->mapped >= holder->table_entries) return PageStatus::kInvalidArgument;
	if (holder->reserve.empty()) return PageStatus::kExhausted;
	const uint32_t p = holder->reserve.back();
	holder->reserve.pop_back();
	holder->pool->dirty[p] = 1;
	holder->table[holder->mapped++] = p;
	return PageStatus::kOk;
}

// Share into another table: refcount += 1 under the pool mutex. The pages enter `dst` as its
// leading shared entries (provenance).
PageStatus HolderShareLeading(PageHolder* dst, const PageHolder* src, uint32_t pages) {
	if (!dst || !src || dst == src || dst->pool != src->pool) return PageStatus::kInvalidArgument;
	if (dst->mapped != 0 || pages == 0 || pages > src->mapped || pages > dst->table_entries)
		return PageStatus::kInvalidArgument;
	PagePool* pool = dst->pool;
	std::lock_guard<CountingMutex> lock(pool->mutex);
	// Checked before any change, so a refusal leaves every count as it was.
	const PageStatus check = CheckShareableLocked(pool, src, pages);
	if (check != PageStatus::kOk) return check;
	for (uint32_t i = 0; i < pages; ++i) {
		const uint32_t p = src->table[i];
		pool->refcount[p] += 1;
		SharedMapEnterLocked(pool, p);
		dst->table[i] = p;
	}
	dst->mapped = pages;
	dst->shared = pages;
	dst->materialized = 0;
	return PageStatus::kOk;
}

// A budget adopt's unmap and share, every refcount change in one pool-mutex section (§3.3, rev 2).
// The leading shared/materialized entries take the pool path (decremented, freed at 0), the rest go
// back to the holder's own reserve; then `src`'s leading `pages` entries become `dst`'s shared ones.
// The increments come first, so a page `dst` already shares from `src` goes +1 then -1 and never
// reaches 0 in between. The freed pages are compacted into `dst`'s vacated leading entries, poisoned
// outside the section (nothing names them any more) and pushed; only then are the shared entries
// written into the table, which is `dst`'s own (its lifecycle lock), not the pool's.
PageStatus HolderAdoptShare(PageHolder* dst, const PageHolder* src, uint32_t pages) {
	if (!dst || !src || dst == src || dst->pool != src->pool) return PageStatus::kInvalidArgument;
	if (pages > src->mapped || pages > dst->table_entries) return PageStatus::kInvalidArgument;
	PagePool* pool = dst->pool;
	uint32_t* table = dst->table.get();
	const uint32_t mapped = dst->mapped;
	uint32_t pool_path = dst->shared + dst->materialized;
	if (pool_path > mapped) pool_path = mapped;
	PageStatus status = PageStatus::kOk;
	uint32_t to_free = 0;
	{
		std::lock_guard<CountingMutex> lock(pool->mutex);
		const PageStatus check = CheckShareableLocked(pool, src, pages);
		if (check != PageStatus::kOk) return check;  // nothing changed
		for (uint32_t i = 0; i < pages; ++i) {
			const uint32_t p = src->table[i];
			pool->refcount[p] += 1;
			SharedMapEnterLocked(pool, p);
		}
		for (uint32_t i = 0; i < pool_path; ++i) {
			const uint32_t p = table[i];
			if (i < dst->shared) SharedMapLeaveLocked(pool, p);
			bool zero = false;
			status = Worse(status, DecrementLocked(pool, p, &zero));
			if (zero) table[to_free++] = p;
		}
	}
	if (to_free > 0) {
		PoisonIfDirty(pool, table, to_free);
		std::lock_guard<CountingMutex> lock(pool->mutex);
		PushFreeLocked(pool, table, to_free);
	}
	// Private entries back to the reserve (they all came from it, so no allocation).
	for (uint32_t i = mapped; i-- > pool_path;) dst->reserve.push_back(table[i]);
#ifdef SUPERSLM_ENABLE_KV_PAGES_TEST_SEAMS
	if (TableSentinelOn())
		for (uint32_t i = 0; i < mapped; ++i) table[i] = kNoPage;
#endif
	for (uint32_t i = 0; i < pages; ++i) table[i] = src->table[i];
	dst->mapped = pages;
	dst->shared = pages;
	dst->materialized = 0;
	return status;
}

// A private restore's materialized pages (§3.7): reserve -> the first table entries, recorded as
// materialized so their unmap takes the pool path. Refcount unchanged (1), no pool mutex.
PageStatus HolderMapMaterialized(PageHolder* holder, uint32_t count) {
	if (!holder || holder->mapped != 0) return PageStatus::kInvalidArgument;
	if (count > holder->table_entries) return PageStatus::kInvalidArgument;
	if (count > holder->reserve.size()) return PageStatus::kExhausted;
	for (uint32_t i = 0; i < count; ++i) {
		const uint32_t p = holder->reserve.back();
		holder->reserve.pop_back();
		holder->pool->dirty[p] = 1;
		holder->table[i] = p;
	}
	holder->mapped = count;
	holder->shared = 0;
	holder->materialized = count;
	return PageStatus::kOk;
}

// Reset's unmap (§3.3's table, by provenance): shared and materialized entries are decremented and
// freed to the pool at 0; private entries go back to the holder's own reserve, refcount unchanged
// and `dirty` kept (a page that later leaves for the pool is still poisoned). Allocates nothing.
PageStatus HolderUnmapAll(PageHolder* holder) {
	if (!holder) return PageStatus::kInvalidArgument;
	PagePool* pool = holder->pool;
	uint32_t* table = holder->table.get();
	const uint32_t mapped = holder->mapped;
	uint32_t pool_path = holder->shared + holder->materialized;
	if (pool_path > mapped) pool_path = mapped;
	PageStatus status = PageStatus::kOk;
	// The pages that reach 0 are compacted into the table's own leading entries, which this unmap is
	// vacating: the freeing path needs no storage of its own.
	uint32_t to_free = 0;
	if (pool_path > 0) {
		std::lock_guard<CountingMutex> lock(pool->mutex);
		for (uint32_t i = 0; i < pool_path; ++i) {
			const uint32_t p = table[i];
			if (i < holder->shared) SharedMapLeaveLocked(pool, p);
			bool zero = false;
			status = Worse(status, DecrementLocked(pool, p, &zero));
			if (zero) table[to_free++] = p;
		}
	}
	if (to_free > 0) {
		PoisonIfDirty(pool, table, to_free);
		std::lock_guard<CountingMutex> lock(pool->mutex);
		PushFreeLocked(pool, table, to_free);
	}
	// Private entries back to the reserve, last mapped pushed first so a remap pops them in order.
	// They all came from this reserve, so its capacity holds them: no allocation.
	for (uint32_t i = mapped; i-- > pool_path;) holder->reserve.push_back(table[i]);
#ifdef SUPERSLM_ENABLE_KV_PAGES_TEST_SEAMS
	// The warm-table sentinel (decision 55): every entry this unmap vacated reads as a fresh one.
	if (TableSentinelOn())
		for (uint32_t i = 0; i < mapped; ++i) table[i] = kNoPage;
#endif
	holder->mapped = 0;
	holder->shared = 0;
	holder->materialized = 0;
	return status;
}

// Release: every mapped page −= 1 (a private page 1 -> 0 and freed; a shared page freed only by its
// last reference, so a prefix's release frees none of the pages its adopters still share); the
// unmapped reserve pages := 0 and freed. The holder is destroyed. Allocates nothing.
PageStatus HolderRelease(PageHolder* holder) {
	if (!holder) return PageStatus::kInvalidArgument;
	PagePool* pool = holder->pool;
	uint32_t* table = holder->table.get();
	std::vector<uint32_t>& reserve = holder->reserve;
	PageStatus status = PageStatus::kOk;
	uint32_t to_free = 0;  // mapped pages that reached 0, compacted into the table's leading entries
	size_t reserve_free = reserve.size();
	{
		std::lock_guard<CountingMutex> lock(pool->mutex);
		for (uint32_t i = 0; i < holder->mapped; ++i) {
			const uint32_t p = table[i];
			if (i < holder->shared) SharedMapLeaveLocked(pool, p);
			bool zero = false;
			status = Worse(status, DecrementLocked(pool, p, &zero));
			if (zero) table[to_free++] = p;
		}
		// A reserve page holds exactly this holder's reference. One already at 0 has been freed by
		// some other path; freeing it again would push it twice, so it is reported and kept off the
		// list (swapped to the tail, which is not freed).
		for (size_t i = 0; i < reserve_free;) {
			uint32_t& rc = pool->refcount[reserve[i]];
			if (rc == 0) {
				status = Worse(status, PageStatus::kRefcountUnderflow);
				std::swap(reserve[i], reserve[--reserve_free]);
				continue;
			}
			rc = 0;
			++i;
		}
	}
	PoisonIfDirty(pool, table, to_free);
	PoisonIfDirty(pool, reserve.data(), reserve_free);
	{
		std::lock_guard<CountingMutex> lock(pool->mutex);
		PushFreeLocked(pool, table, to_free);
		PushFreeLocked(pool, reserve.data(), reserve_free);
	}
	delete holder;
	return status;
}

// A legacy prefix's first freeze, and the drain seam: the reserve's pages go back to the pool by
// release's reserve rows (refcount := 0, poisoned iff dirty), the table untouched.
PageStatus HolderReturnReserve(PageHolder* holder, uint32_t* out_returned) {
	if (out_returned) *out_returned = 0;
	if (!holder) return PageStatus::kInvalidArgument;
	PagePool* pool = holder->pool;
	std::vector<uint32_t>& reserve = holder->reserve;
	size_t n = reserve.size();
	if (n == 0) return PageStatus::kOk;
	PageStatus status = PageStatus::kOk;
	{
		std::lock_guard<CountingMutex> lock(pool->mutex);
		// A reserve page holds exactly this holder's reference; one already at 0 was freed by
		// another path and is reported, never pushed twice.
		for (size_t i = 0; i < n;) {
			uint32_t& rc = pool->refcount[reserve[i]];
			if (rc == 0) {
				status = Worse(status, PageStatus::kRefcountUnderflow);
				std::swap(reserve[i], reserve[--n]);
				continue;
			}
			rc = 0;
			++i;
		}
	}
	PoisonIfDirty(pool, reserve.data(), n);
	{
		std::lock_guard<CountingMutex> lock(pool->mutex);
		PushFreeLocked(pool, reserve.data(), n);
	}
	// clear() keeps the capacity, so a later unmap's push back to this reserve never allocates.
	reserve.clear();
	if (out_returned) *out_returned = static_cast<uint32_t>(n);
	return status;
}

void PoolCounts(PagePool* pool, uint32_t* out_free, uint32_t* out_shared) {
	if (!pool) {
		if (out_free) *out_free = 0;
		if (out_shared) *out_shared = 0;
		return;
	}
	std::lock_guard<CountingMutex> lock(pool->mutex);
	if (out_free) *out_free = static_cast<uint32_t>(pool->free_list.size());
	if (out_shared) *out_shared = pool->shared_distinct;
}

uint64_t PoolMutexAcquisitions(const PagePool* pool) { return pool ? pool->mutex.acquisitions() : 0; }

uint32_t HolderMapped(const PageHolder* holder) { return holder ? holder->mapped : 0; }
uint32_t HolderShared(const PageHolder* holder) { return holder ? holder->shared : 0; }
uint32_t HolderReserve(const PageHolder* holder) {
	return holder ? static_cast<uint32_t>(holder->reserve.size()) : 0;
}

// The address path's table read. Under the seam, an entry holding the sentinel traps: a read at or
// above `mapped` (cell 7.13). Without it, the read is a plain load.
uint32_t HolderTableEntryForAddress(const PageHolder* holder, uint32_t i) {
#ifdef SUPERSLM_ENABLE_KV_PAGES_TEST_SEAMS
	if (i >= holder->table_entries || holder->table[i] == kNoPage) std::abort();
#endif
	return holder->table[i];
}

#ifdef SUPERSLM_ENABLE_KV_PAGES_TEST_SEAMS
void SetTableSentinelForTest(bool enabled) { g_table_sentinel.store(enabled, std::memory_order_relaxed); }

// A second unmap of entry `i` through the shared-page path, as a double unmap would: the module's
// own decrement, its own underflow check, and its own freeing at 0.
PageStatus DoubleUnmapForTest(PageHolder* holder, uint32_t i) {
	if (!holder || i >= holder->table_entries) return PageStatus::kInvalidArgument;
	PagePool* pool = holder->pool;
	uint32_t p = holder->table[i];
	if (p >= pool->page_count) return PageStatus::kInvalidArgument;
	bool zero = false;
	PageStatus st;
	{
		std::lock_guard<CountingMutex> lock(pool->mutex);
		st = DecrementLocked(pool, p, &zero);
	}
	if (zero) {
		PoisonIfDirty(pool, &p, 1);
		std::lock_guard<CountingMutex> lock(pool->mutex);
		PushFreeLocked(pool, &p, 1);
	}
	return st;
}
#endif

}  // namespace kv_pages
}  // namespace superslm

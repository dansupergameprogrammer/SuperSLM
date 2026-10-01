// Paged-KV plan (rev 16.1) step C1: the page module's test-facing surface (§3.3), which the C3
// cells (7.13's module part, 11.3, U1) call.
//
// The plan fixes the module's state and laws (§3.3) but not its C++ names, so the test author
// declares the narrowest surface those three cells need, and the C3 builder implements it in
// src/kv_pages.h / src/kv_pages.cpp (namespace superslm::kv_pages). The builder may rename by
// updating this file and the cells together, never the assertions. Red by link until C3.
//
// Every operation below is the module's own: no ABI handle, no model. A pool is `page_count` pages
// of `page_bytes` bytes over caller memory; a holder owns a page table of `table_entries` entries,
// a private reserve, and the `shared` count (§3.3, "What 'shared' means": the leading entries that
// entered by sharing).

#ifndef SUPERSLM_TESTS_PKV_MODULE_API_H
#define SUPERSLM_TESTS_PKV_MODULE_API_H

#include <cstddef>
#include <cstdint>

namespace superslm {
namespace kv_pages {

struct PagePool;
struct PageHolder;

enum class PageStatus : int {
	kOk = 0,
	kExhausted = 1,         // the pool (or, for a map, the holder's reserve) has too few pages
	kInvalidArgument = 2,
	kRefcountUnderflow = 3, // §3.3: the release-build underflow check (cell 11.3)
	kAllocationFailed = 4,
};

// Pool over `base` (page_count * page_bytes bytes, owned by the caller).
PageStatus PoolCreate(uint8_t* base, uint32_t page_count, size_t page_bytes, int64_t page_positions,
                      PagePool** out);
void PoolDestroy(PagePool* pool);
uint32_t PoolFreePages(PagePool* pool);  // takes the pool mutex
uint32_t PoolRefcount(PagePool* pool, uint32_t page);  // takes the pool mutex; test reading only

// A holder with a table of `table_entries` entries and `reserve_pages` pages drawn pool -> reserve
// (refcount := 1 each). kExhausted, with nothing drawn, when the pool has fewer free pages.
PageStatus HolderCreate(PagePool* pool, uint32_t table_entries, uint32_t reserve_pages, PageHolder** out);

// Reserve -> table: maps the next entry from the holder's own reserve (no pool mutex).
// kExhausted with nothing changed when the reserve is empty.
PageStatus HolderMapNext(PageHolder* holder);

// Shares `src`'s leading `pages` mapped entries into `dst`'s table as its leading shared entries
// (refcount += 1 each, under the pool mutex). `dst` must have nothing mapped.
PageStatus HolderShareLeading(PageHolder* dst, const PageHolder* src, uint32_t pages);

// Reset's unmap: private mapped pages back to the holder's own reserve; shared pages decremented
// and freed to the pool at 0 (§3.3's table, by provenance).
PageStatus HolderUnmapAll(PageHolder* holder);

// Release: every mapped page and the reserve, per §3.3's release rows. The holder is destroyed.
PageStatus HolderRelease(PageHolder* holder);

uint32_t HolderMapped(const PageHolder* holder);
uint32_t HolderShared(const PageHolder* holder);
uint32_t HolderReserve(const PageHolder* holder);

// The address path's table read (entry i, any i < table_entries): the page index the address
// formula would use. Under the sentinel seam it traps (std::abort) when the entry holds 0xFFFFFFFF.
uint32_t HolderTableEntryForAddress(const PageHolder* holder, uint32_t i);

// ---- test seams ----
// Cell 7.13: process-wide; with it on, every table created afterwards is filled with 0xFFFFFFFF and
// every entry an unmap releases is rewritten to it.
void SetTableSentinelForTest(bool enabled);
// Cell 11.3: unmaps table entry `i` a second time through the shared-page path (refcount -= 1 under
// the pool mutex), as a double unmap would. Returns what the module's own check returns.
PageStatus DoubleUnmapForTest(PageHolder* holder, uint32_t i);

}  // namespace kv_pages
}  // namespace superslm

#endif  // SUPERSLM_TESTS_PKV_MODULE_API_H

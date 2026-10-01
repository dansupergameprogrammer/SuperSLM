// Paged-KV plan (rev 16.1) §3.6 and decision 24, step C3: the page module's own u32 page-count
// bound, the module twin of 2.9b [C4] (review S1).
//
// The plan refuses a page count only when it EXCEEDS UINT32_MAX. A pool of exactly UINT32_MAX pages
// has indices 0 .. UINT32_MAX - 1, so the kNoPage sentinel (UINT32_MAX) never names a real page, and
// nothing in the module needs a smaller bound (its arrays hold page_count entries and every loop over
// them is `i < n` on u32). PoolCreate must therefore accept the count and go on to build the pool.
//
// The cell must not build it: the per-page arrays alone are 13 bytes a page, about 52 GiB. So this
// file replaces the global operator new for superslm_pkv_c3 with one that forwards to malloc and,
// only while a cell arms it, refuses any single request above kArmedCap with std::bad_alloc instead
// of allocating. PoolCreate catches its own bad_alloc and reports kAllocationFailed. So:
//   - a module that accepts UINT32_MAX passes its count checks, asks for its first per-page array
//     (UINT32_MAX entries), is refused by the cap, and returns kAllocationFailed, with the refused
//     request recorded here;
//   - a module that refuses the exact maximum returns kInvalidArgument and asks for nothing.
// The status and the recorded request are the verdict; nothing of the real size is ever allocated.
// No other c3_*.cpp may define an operator new.

#include "pkv_common.h"
#include "pkv_module_api.h"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <new>

namespace {
constexpr std::size_t kArmedCap = std::size_t{1} << 30;  // 1 GiB: far above any other C3 cell's request
std::atomic<bool> g_armed{false};
std::atomic<long long> g_refused{0};
std::atomic<std::size_t> g_first_refused{0};
}  // namespace

// Pairs malloc with free (GCC's -Wmismatched-new-delete cannot see that; as c4_dim7_contract.cpp, no
// function-pointer indirection, so the MSVC /MD import of std::malloc is never read before init).
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmismatched-new-delete"
#endif
void* operator new(std::size_t size) {
	if (size > kArmedCap && g_armed.load(std::memory_order_relaxed)) {
		std::size_t none = 0;
		g_first_refused.compare_exchange_strong(none, size, std::memory_order_relaxed);
		g_refused.fetch_add(1, std::memory_order_relaxed);
		throw std::bad_alloc{};
	}
	if (void* p = std::malloc(size ? size : 1)) return p;
	throw std::bad_alloc{};
}
void* operator new[](std::size_t size) { return operator new(size); }
void operator delete(void* p) noexcept { std::free(p); }
void operator delete(void* p, std::size_t) noexcept { std::free(p); }
void operator delete[](void* p) noexcept { std::free(p); }
void operator delete[](void* p, std::size_t) noexcept { std::free(p); }
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic pop
#endif

namespace {

using namespace superslm::kv_pages;

struct Armed {
	Armed() {
		g_refused.store(0, std::memory_order_relaxed);
		g_first_refused.store(0, std::memory_order_relaxed);
		g_armed.store(true, std::memory_order_relaxed);
	}
	~Armed() { g_armed.store(false, std::memory_order_relaxed); }
};

// 2.9b [C3]. PoolCreate at page_count = UINT32_MAX (page_bytes 1, B 1, over a 256-byte buffer the
// module never touches at create) passes the module's count check and reaches its first per-page
// allocation, which the armed cap refuses: kAllocationFailed, a null pool, the buffer untouched. The
// control at the same count with page_bytes one past SIZE_MAX / UINT32_MAX is refused by the
// module's own product check, kInvalidArgument, before any allocation.
void Cell29bModule() {
	PKV_CHECK_MSG(sizeof(std::size_t) >= 8, "2.9b/C3 needs a 64-bit size_t (the per-page arrays exceed 4 GiB)");
	if (sizeof(std::size_t) < 8) return;
	pkv::AlignedBuf mem(256, 0x5A);

	PagePool* pool = reinterpret_cast<PagePool*>(&mem);  // non-null; PoolCreate must null it
	PageStatus st;
	long long refused = 0;
	std::size_t first = 0;
	{
		Armed armed;
		st = PoolCreate(mem.p, UINT32_MAX, 1, 1, &pool);
		refused = g_refused.load(std::memory_order_relaxed);
		first = g_first_refused.load(std::memory_order_relaxed);
	}
	// kills: the module refusing page_count == kNoPage (== UINT32_MAX), which returns kInvalidArgument
	// before it sizes anything; decision 24 refuses only counts above UINT32_MAX
	PKV_CHECK_MSG(st == PageStatus::kAllocationFailed,
	              "PoolCreate(UINT32_MAX pages) -> %d, want kAllocationFailed (count admitted, then the "
	              "armed cap refused its per-page array)",
	              static_cast<int>(st));
	PKV_CHECK_MSG(refused >= 1 && first >= std::size_t{UINT32_MAX},
	              "the module asked for a per-page array of UINT32_MAX entries (refused %lld, first %zu bytes)",
	              refused, first);
	PKV_CHECK(pool == nullptr);
	if (pool && pool != reinterpret_cast<PagePool*>(&mem)) PoolDestroy(pool);
	bool untouched = true;
	for (std::size_t i = 0; i < mem.n; ++i) untouched = untouched && mem.p[i] == 0x5A;
	PKV_CHECK_MSG(untouched, "PoolCreate wrote into the pool memory at create");

	// Control: the product check still runs at the maximum count, and before any allocation.
	pool = reinterpret_cast<PagePool*>(&mem);
	{
		Armed armed;
		st = PoolCreate(mem.p, UINT32_MAX, SIZE_MAX / UINT32_MAX + 1, 1, &pool);
		refused = g_refused.load(std::memory_order_relaxed);
	}
	PKV_CHECK_EQ(static_cast<int>(st), static_cast<int>(PageStatus::kInvalidArgument));
	PKV_CHECK_EQ(refused, 0);
	PKV_CHECK(pool == nullptr);
	if (pool && pool != reinterpret_cast<PagePool*>(&mem)) PoolDestroy(pool);
}

PKV_CELL("2.9b/C3", "C3", Cell29bModule);

}  // namespace

// Paged-KV plan (rev 16.1) §7, step C3: the page module's cells -- 7.13's module part, 11.3, U1 and the
// module twins of 1.3 and 1.6 (§9's C3 row) (§3.3, the kv_pages module, through pkv_module_api.h).
//
// The module is driven directly: no ABI handle, no model. A pool is a few pages over the test's own
// memory. Page counts are graded by admission (a holder of k reserve pages is admitted and one of
// k + 1 is refused, nothing drawn), the module-level form of §8's fill probe; the per-page refcount
// is read through PoolRefcount, the module's test-reading surface, where the cell's text is the
// count itself (U1, 11.3).
//
// 7.13's trap is std::abort and 11.3's mutant is an assert, so those cases run in a child process
// (pkv_engine_child.h): POSIX forks; the box's MSVC leg re-runs this executable on the cell with the
// case named in the environment.

#include "pkv_common.h"
#include "pkv_engine_child.h"
#include "pkv_module_api.h"

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstring>
#include <thread>
#include <vector>

namespace {

using namespace superslm::kv_pages;
using pkv_engine::ChildEnd;
using pkv_engine::Passed;
using pkv_engine::RunInChild;

constexpr size_t kPageBytes = 256;
constexpr int64_t kPagePositions = 16;
constexpr uint32_t kSentinel = 0xFFFFFFFFu;

// A pool of `pages` pages over the test's memory.
struct Pool {
	pkv::AlignedBuf mem;
	PagePool* pool = nullptr;
	PageStatus status = PageStatus::kInvalidArgument;
	explicit Pool(uint32_t pages) : mem(kPageBytes * pages, 0x5A) {
		status = PoolCreate(mem.p, pages, kPageBytes, kPagePositions, &pool);
	}
	~Pool() {
		if (pool) PoolDestroy(pool);
	}
};

// The module's fill probe: exactly `k` pages are free iff a holder reserving k is admitted and,
// after its release, one reserving k + 1 is refused with kExhausted. Leaves the pool as found.
bool ExactlyFree(PagePool* pool, uint32_t k) {
	PageHolder* h = nullptr;
	const bool admits = HolderCreate(pool, 1, k, &h) == PageStatus::kOk;
	if (h) HolderRelease(h);
	h = nullptr;
	const PageStatus refused = HolderCreate(pool, 1, k + 1, &h);
	if (h) HolderRelease(h);
	PKV_CHECK_MSG(admits && refused == PageStatus::kExhausted, "fill probe at %u: admits %u -> %d, %u + 1 -> %d", k, k,
	              admits, k, static_cast<int>(refused));
	return admits && refused == PageStatus::kExhausted;
}

// 7.13 [C3]'s construction, up to the read: under the sentinel seam, a table of 8 entries mapped to
// 7 pages, unmapped (the pages return to the holder's reserve), then mapped to 3. Returns the holder
// (nullptr on a failed step, reported in the child's stderr).
PageHolder* SevenThenThree(Pool& p) {
	SetTableSentinelForTest(true);
	PageHolder* h = nullptr;
	if (p.status != PageStatus::kOk || HolderCreate(p.pool, 8, 7, &h) != PageStatus::kOk) return nullptr;
	for (int i = 0; i < 7; ++i)
		if (HolderMapNext(h) != PageStatus::kOk) return nullptr;
	if (HolderUnmapAll(h) != PageStatus::kOk || HolderMapped(h) != 0) return nullptr;
	for (int i = 0; i < 3; ++i)
		if (HolderMapNext(h) != PageStatus::kOk) return nullptr;
	return HolderMapped(h) == 3 ? h : nullptr;
}

// 7.13 [C3]: "table entries at or above `mapped` are never read". A read of entry 3 traps, warm table
// and fresh table alike; the entries below `mapped` read as pages.
void Cell713Module() {
	// The trap, warm: entry 3 held a page during the first mapping and was rewritten to the sentinel
	// by the unmap. The child must die by std::abort.
	{
		const ChildEnd end = RunInChild("7.13-warm", [] {
			Pool p(8);
			PageHolder* h = SevenThenThree(p);
			if (!h) {
				std::fprintf(stderr, "7.13 warm: the 7 / unmap / 3 construction failed\n");
				return false;
			}
			const uint32_t page = HolderTableEntryForAddress(h, 3);
			std::fprintf(stderr, "7.13 warm: entry 3 read as page %u, no trap\n", page);
			return false;
		});
		// kills: the unmap that skips the sentinel rewrite (entry 3 returns the stale page of the first
		// mapping and the child exits).
		PKV_CHECK_MSG(end.aborted, "7.13 warm table, read of entry 3: the child %s, not aborted", end.Describe().c_str());
	}
	// The trap, fresh: a new table under the seam is all sentinel; with 3 mapped, entry 3 traps.
	{
		const ChildEnd end = RunInChild("7.13-fresh", [] {
			SetTableSentinelForTest(true);
			Pool p(8);
			PageHolder* h = nullptr;
			if (p.status != PageStatus::kOk || HolderCreate(p.pool, 8, 3, &h) != PageStatus::kOk) return false;
			for (int i = 0; i < 3; ++i)
				if (HolderMapNext(h) != PageStatus::kOk) return false;
			const uint32_t page = HolderTableEntryForAddress(h, 3);
			std::fprintf(stderr, "7.13 fresh: entry 3 read as page %u, no trap\n", page);
			return false;
		});
		// kills: a create that leaves the table uninitialised under the seam (the read returns garbage).
		PKV_CHECK_MSG(end.aborted, "7.13 fresh table, read of entry 3: the child %s, not aborted", end.Describe().c_str());
	}
	// Control, at the same construction: entries 0..2 are pages of the pool, distinct, and reading them
	// does not trap. This is what shows the warm child above died at entry 3 and not before it.
	{
		const ChildEnd end = RunInChild("7.13-control", [] {
			Pool p(8);
			PageHolder* h = SevenThenThree(p);
			if (!h) return false;
			const uint32_t a = HolderTableEntryForAddress(h, 0), b = HolderTableEntryForAddress(h, 1),
			               c = HolderTableEntryForAddress(h, 2);
			const bool ok = a < 8 && b < 8 && c < 8 && a != b && b != c && a != c && a != kSentinel;
			if (!ok) std::fprintf(stderr, "7.13 control: entries 0..2 read %u %u %u\n", a, b, c);
			return ok && HolderRelease(h) == PageStatus::kOk;
		});
		// kills: a sentinel that is also written into mapped entries (the address path would trap on a
		// live row), and a remap that does not refill entries below `mapped`.
		PKV_CHECK_MSG(Passed(end), "7.13 control, entries 0..2: the child %s", end.Describe().c_str());
	}
}

// 11.3 [C3]: refcount underflow. A page shared into a second holder is unmapped there legitimately
// (its refcount reaches 0 and it is freed); the seam then unmaps the same entry a second time. The
// release-build check fires: the seam returns kRefcountUnderflow, the count stays 0 and the page is
// not freed twice. Run in a child, so an NDEBUG-style assert (which aborts in a debug library and is
// compiled out in a release one) is red either way.
void Cell113() {
	const ChildEnd end = RunInChild("11.3", [] {
		Pool p(4);
		if (p.status != PageStatus::kOk) return false;
		PageHolder* a = nullptr;
		PageHolder* b = nullptr;
		bool ok = HolderCreate(p.pool, 1, 1, &a) == PageStatus::kOk && HolderMapNext(a) == PageStatus::kOk;
		const uint32_t page = ok ? HolderTableEntryForAddress(a, 0) : 0;
		ok = ok && HolderCreate(p.pool, 1, 0, &b) == PageStatus::kOk && HolderShareLeading(b, a, 1) == PageStatus::kOk &&
		     PoolRefcount(p.pool, page) == 2;
		// a's release decrements the page to 1 (b still shares it); b's unmap takes it to 0 and frees it.
		ok = ok && HolderRelease(a) == PageStatus::kOk && PoolRefcount(p.pool, page) == 1;
		ok = ok && HolderUnmapAll(b) == PageStatus::kOk && PoolRefcount(p.pool, page) == 0;
		if (!ok) {
			std::fprintf(stderr, "11.3: the construction failed before the double unmap\n");
			return false;
		}
		const bool four_free = ExactlyFree(p.pool, 4);
		const PageStatus st = DoubleUnmapForTest(b, 0);
		const uint32_t rc = PoolRefcount(p.pool, page);
		// The page is still freed exactly once: 4 free, not 5 (a double push) and not 3.
		const bool still_four = ExactlyFree(p.pool, 4);
		const bool fired = st == PageStatus::kRefcountUnderflow && rc == 0;
		if (!fired) std::fprintf(stderr, "11.3: double unmap returned %d, refcount %u\n", static_cast<int>(st), rc);
		HolderRelease(b);
		return four_free && fired && still_four;
	});
	// kills: the underflow check as an assert (aborts the child in a debug library; compiled out in a
	// release one, where the count wraps to 0xFFFFFFFF or the page is pushed twice and the child exits 1).
	PKV_CHECK_MSG(Passed(end), "11.3 double unmap: the child %s", end.Describe().c_str());
	// Control: the same construction's single unmap is not an underflow (the check does not fire on a
	// legitimate decrement to 0).
	const ChildEnd ctl = RunInChild("11.3-control", [] {
		Pool p(4);
		PageHolder* a = nullptr;
		PageHolder* b = nullptr;
		const bool ok = p.status == PageStatus::kOk && HolderCreate(p.pool, 1, 1, &a) == PageStatus::kOk &&
		                HolderMapNext(a) == PageStatus::kOk && HolderCreate(p.pool, 1, 0, &b) == PageStatus::kOk &&
		                HolderShareLeading(b, a, 1) == PageStatus::kOk && HolderRelease(a) == PageStatus::kOk &&
		                HolderUnmapAll(b) == PageStatus::kOk && HolderRelease(b) == PageStatus::kOk;
		return ok && ExactlyFree(p.pool, 4);
	});
	// kills: a check that fires on every decrement to 0 (the legitimate unmap would report underflow).
	PKV_CHECK_MSG(Passed(ctl), "11.3 control, one legitimate unmap: the child %s", ctl.Describe().c_str());
}

// U1 [C3]: concurrent share and unshare of one page from 16 threads, with an exact final count.
// One source holder maps the page (refcount 1). Each of 16 threads owns a holder and runs 2,000
// share/unshare cycles against it, then shares once more: the count is then exactly 1 + 16. All 16
// unshare at once: exactly 1. The source's release frees it: the pool is whole again. The race is the
// TSan leg's (§9, C3's exit "module cells green under TSan"); this run grades the counts.
void CellU1() {
	constexpr int kThreads = 16;
	constexpr int kCycles = 2000;
	Pool p(4);
	PKV_CHECK_EQ(static_cast<int>(p.status), static_cast<int>(PageStatus::kOk));
	if (p.status != PageStatus::kOk) return;
	PageHolder* src = nullptr;
	PKV_CHECK_EQ(static_cast<int>(HolderCreate(p.pool, 1, 1, &src)), static_cast<int>(PageStatus::kOk));
	if (!src) return;
	PKV_CHECK_EQ(static_cast<int>(HolderMapNext(src)), static_cast<int>(PageStatus::kOk));
	const uint32_t page = HolderTableEntryForAddress(src, 0);
	PKV_CHECK_EQ(PoolRefcount(p.pool, page), 1);
	std::vector<PageHolder*> holders(kThreads, nullptr);
	for (auto& h : holders) PKV_CHECK_EQ(static_cast<int>(HolderCreate(p.pool, 1, 0, &h)), static_cast<int>(PageStatus::kOk));
	std::atomic<int> failures{0};
	std::atomic<int> ready{0};
	auto run = [&](int t, bool cycles) {
		ready.fetch_add(1);
		while (ready.load() < kThreads) std::this_thread::yield();
		if (cycles)
			for (int i = 0; i < kCycles; ++i) {
				if (HolderShareLeading(holders[t], src, 1) != PageStatus::kOk) failures.fetch_add(1);
				if (HolderUnmapAll(holders[t]) != PageStatus::kOk) failures.fetch_add(1);
			}
		if (cycles) {
			if (HolderShareLeading(holders[t], src, 1) != PageStatus::kOk) failures.fetch_add(1);
		} else if (HolderUnmapAll(holders[t]) != PageStatus::kOk) {
			failures.fetch_add(1);
		}
	};
	{
		std::vector<std::thread> ts;
		for (int t = 0; t < kThreads; ++t) ts.emplace_back(run, t, true);
		for (auto& t : ts) t.join();
	}
	PKV_CHECK_EQ(failures.load(), 0);
	// kills: a refcount changed outside the pool mutex (lost increments or decrements; TSan names the
	// race), and a share that does not count.
	PKV_CHECK_EQ(PoolRefcount(p.pool, page), 1 + kThreads);
	for (PageHolder* h : holders) {
		PKV_CHECK_EQ(HolderMapped(h), 1);
		PKV_CHECK_EQ(HolderShared(h), 1);
		PKV_CHECK_EQ(HolderTableEntryForAddress(h, 0), page);
	}
	ready.store(0);
	{
		std::vector<std::thread> ts;
		for (int t = 0; t < kThreads; ++t) ts.emplace_back(run, t, false);
		for (auto& t : ts) t.join();
	}
	PKV_CHECK_EQ(failures.load(), 0);
	// kills: an unshare that frees a page some other holder still maps (a count reaching 0 early).
	PKV_CHECK_EQ(PoolRefcount(p.pool, page), 1);
	// 3 pages free while the source holds its one; all 4 after its release (freed exactly once).
	ExactlyFree(p.pool, 3);
	for (PageHolder* h : holders) HolderRelease(h);
	PKV_CHECK_EQ(static_cast<int>(HolderRelease(src)), static_cast<int>(PageStatus::kOk));
	ExactlyFree(p.pool, 4);
}

// 1.3 [C3], the module twin of 1.3: after 1,000 mixed holder create / map / share / unmap / release
// cycles, the pool's free pages return exactly to the initial count -- no page leaked, none freed
// twice. Graded by the module's own admission (ExactlyFree), never by PoolFreePages.
void Cell13Module() {
	constexpr uint32_t kPages = 16;
	constexpr uint32_t kEntries = 4;
	constexpr int kCycles = 1000;
	Pool p(kPages);
	PKV_CHECK_EQ(static_cast<int>(p.status), static_cast<int>(PageStatus::kOk));
	if (p.status != PageStatus::kOk) return;
	ExactlyFree(p.pool, kPages);
	std::vector<PageHolder*> live;
	// A holder that has been shared from plays a prefix: §3.3 shares only a prefix's pages (budget adopt,
	// begin_from), and a prefix is never reset, so a source is released but never unmapped. (Unmapping
	// it would send a page its sharers still map back to its own reserve.)
	std::vector<PageHolder*> sources;
	auto is_source = [&](PageHolder* h) { return std::find(sources.begin(), sources.end(), h) != sources.end(); };
	uint32_t x = 0xC0FFEEu;
	auto rnd = [&](uint32_t n) {
		x = x * 1664525u + 1013904223u;
		return (x >> 8) % n;
	};
	int bad = 0, creates = 0, refused = 0, shares = 0, unmaps = 0, releases = 0;
	for (int c = 0; c < kCycles; ++c) {
		const uint32_t op = live.empty() ? 0 : rnd(5);
		if (op == 0) {  // create, reserving 1..3 pages; refused with nothing drawn when the pool is short
			PageHolder* h = nullptr;
			const PageStatus st = HolderCreate(p.pool, kEntries, 1 + rnd(3), &h);
			if (st == PageStatus::kOk) {
				live.push_back(h);
				++creates;
			} else if (st == PageStatus::kExhausted && h == nullptr) {
				++refused;
			} else {
				++bad;
			}
		} else if (op == 1) {  // map the next entry from the holder's own reserve
			PageHolder* h = live[rnd(static_cast<uint32_t>(live.size()))];
			if (HolderReserve(h) > 0 && HolderMapped(h) < kEntries && HolderMapNext(h) != PageStatus::kOk) ++bad;
		} else if (op == 2) {  // share a holder's leading pages into an empty holder
			PageHolder* src = live[rnd(static_cast<uint32_t>(live.size()))];
			PageHolder* dst = live[rnd(static_cast<uint32_t>(live.size()))];
			if (src != dst && HolderMapped(src) > 0 && HolderMapped(dst) == 0) {
				if (HolderShareLeading(dst, src, 1 + rnd(HolderMapped(src))) != PageStatus::kOk) {
					++bad;
				} else {
					++shares;
					if (!is_source(src)) sources.push_back(src);
				}
			}
		} else if (op == 3) {  // reset's unmap, on a holder that is not a source
			PageHolder* h = live[rnd(static_cast<uint32_t>(live.size()))];
			if (!is_source(h)) {
				if (HolderUnmapAll(h) != PageStatus::kOk) ++bad;
				++unmaps;
			}
		} else {  // release (a source's release frees none of the pages its sharers still map)
			const size_t i = rnd(static_cast<uint32_t>(live.size()));
			if (is_source(live[i])) sources.erase(std::find(sources.begin(), sources.end(), live[i]));
			if (HolderRelease(live[i]) != PageStatus::kOk) ++bad;
			live.erase(live.begin() + static_cast<std::ptrdiff_t>(i));
			++releases;
		}
	}
	PKV_CHECK_EQ(bad, 0);
	// The construction exercised what the cell names: sharing, unmapping and releasing with sharers
	// live, and creates refused at a short pool.
	PKV_CHECK_MSG(shares > 20 && unmaps > 20 && releases > 20 && refused > 0,
	              "1.3 mix: %d creates, %d refused, %d shares, %d unmaps, %d releases", creates, refused, shares, unmaps,
	              releases);
	for (auto it = live.rbegin(); it != live.rend(); ++it)
		if (HolderRelease(*it) != PageStatus::kOk) ++bad;
	PKV_CHECK_EQ(bad, 0);
	// kills: a shared page that reaches 0 on an unmap and is never pushed, or a release that leaks its
	// reserve (k is not admitted); and a shared page freed by every sharer instead of the last (pushed
	// twice: k + 1 is admitted).
	ExactlyFree(p.pool, kPages);
}

// 1.6 [C3], the module twin of 1.6's dirty accounting. Pool memory pre-filled 0x5A. A holder draws a
// reserve of 6 pages, maps 3 and the test writes them (through the page index the table holds, in
// the test's own memory), then the holder is reset (unmapped: the pages go back to its reserve) and
// released. Every page it wrote reads 0xCD; its 3 clean reserve pages, and the pool's undrawn pages,
// still read 0x5A.
void Cell16Module() {
	constexpr uint32_t kPages = 8;
	Pool p(kPages);
	PKV_CHECK_EQ(static_cast<int>(p.status), static_cast<int>(PageStatus::kOk));
	if (p.status != PageStatus::kOk) return;
	PageHolder* h = nullptr;
	PKV_CHECK_EQ(static_cast<int>(HolderCreate(p.pool, 6, 6, &h)), static_cast<int>(PageStatus::kOk));
	if (!h) return;
	std::vector<uint32_t> written;
	for (int i = 0; i < 3; ++i) {
		PKV_CHECK_EQ(static_cast<int>(HolderMapNext(h)), static_cast<int>(PageStatus::kOk));
		const uint32_t page = HolderTableEntryForAddress(h, static_cast<uint32_t>(i));
		PKV_CHECK(page < kPages);
		if (page >= kPages) return;
		std::memset(p.mem.p + size_t{page} * kPageBytes, 0x11 + i, kPageBytes);  // the K/V rows a write lands
		written.push_back(page);
	}
	PKV_CHECK_EQ(static_cast<int>(HolderUnmapAll(h)), static_cast<int>(PageStatus::kOk));
	PKV_CHECK_EQ(static_cast<int>(HolderRelease(h)), static_cast<int>(PageStatus::kOk));
	auto all = [&](uint32_t page, uint8_t v) {
		const uint8_t* b = p.mem.p + size_t{page} * kPageBytes;
		for (size_t i = 0; i < kPageBytes; ++i)
			if (b[i] != v) return false;
		return true;
	};
	int poisoned = 0;
	for (uint32_t page = 0; page < kPages; ++page) {
		const bool was_written = std::find(written.begin(), written.end(), page) != written.end();
		if (was_written) {
			// kills: the poison fill removed, or skipped for a page that went back through the reserve
			// before the release (the reset must not clear `dirty`).
			PKV_CHECK_MSG(all(page, 0xCD), "1.6: written page %u does not read 0xCD after the release", page);
			++poisoned;
		} else {
			// kills: the mutant that ignores the dirty flag and poisons every freed page (a clean reserve
			// page turns 0x5A -> 0xCD), and a fill that strays past its own page.
			PKV_CHECK_MSG(all(page, 0x5A), "1.6: clean page %u does not read 0x5A after the release", page);
		}
	}
	PKV_CHECK_EQ(poisoned, 3);
	ExactlyFree(p.pool, kPages);
}

PKV_CELL("7.13/C3", "C3", Cell713Module);
PKV_CELL("11.3", "C3", Cell113);
PKV_CELL("U1", "C3", CellU1);
PKV_CELL("1.3/C3", "C3", Cell13Module);
PKV_CELL("1.6/C3", "C3", Cell16Module);

}  // namespace

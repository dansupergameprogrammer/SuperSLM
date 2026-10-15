// GPU paged-KV plan (rev 16), step G3: the boundary token check's registry. D3D12-free; see the
// header for the rules this file implements, each named by its plan paragraph below.
#include "gpu_inflight_registry.h"

#include <atomic>
#include <cstdio>
#include <mutex>

namespace superslm_gpu {
namespace inflight {

// The list's mutex. One per list (rev 13; strike round 5).
using ListMutex = std::mutex;

namespace detail {
struct RegList {
	ListMutex mu;
	InFlightRecord* head = nullptr;
	std::size_t live = 0;
};
}  // namespace detail

using detail::RegList;

namespace {

std::atomic<uint64_t> g_call_counter{0};
std::atomic<uint64_t> g_reports{0};
std::atomic<uint64_t> g_walked{0};
std::atomic<int64_t> g_live_records{0};

// The thread's own list, held by the thread and by every record registered in it.
thread_local std::shared_ptr<RegList> t_own_list;
// The binding: the list registrations go to (null: the thread's own), the outermost call, the
// depth. A task trampoline installs a captured binding; GpuApiCallScope opens one.
thread_local std::shared_ptr<RegList> t_bound_list;
thread_local uint64_t t_call = 0;
thread_local int t_depth = 0;
// The boundary guard check's flag, and the outermost entry point's name for the report line.
thread_local bool t_guard_ran = false;
thread_local const char* t_entry_point = nullptr;

RegList* OwnList() {
	if (!t_own_list) t_own_list = std::make_shared<RegList>();
	return t_own_list.get();
}

std::shared_ptr<RegList> CurrentList() {
	if (t_bound_list) return t_bound_list;
	OwnList();
	return t_own_list;
}

}  // namespace

struct RegistryAccess {
	// Links `rec` into the current binding's list, under its call (no call at depth 0).
	static void Link(InFlightRecord* rec, InFlightHolder* holder) {
		std::shared_ptr<RegList> L = CurrentList();
		std::lock_guard<ListMutex> lock(L->mu);
		rec->list_ = L;
		rec->call_ = t_depth > 0 ? t_call : 0;
		rec->holder_ = holder;
		rec->prev_ = nullptr;
		rec->next_ = L->head;
		if (L->head) L->head->prev_ = rec;
		L->head = rec;
		++L->live;
	}
	// Unlinks `rec` from its list under that list's lock; the list reference this function took
	// is dropped after the lock is released (the lock_guard is destroyed first).
	static void UnlinkFromList(InFlightRecord* rec) {
		std::shared_ptr<RegList> L = rec->list_;
		std::lock_guard<ListMutex> lock(L->mu);
		if (rec->prev_) rec->prev_->next_ = rec->next_;
		else L->head = rec->next_;
		if (rec->next_) rec->next_->prev_ = rec->prev_;
		rec->prev_ = nullptr;
		rec->next_ = nullptr;
		--L->live;
	}
	// A depth-0 event keeps the record's registration; only its holder changes, under that list's
	// lock (rev 14; strike round 6, strike 1b).
	static bool KeepAtDepthZero(InFlightRecord* rec, InFlightHolder* holder) {
		if (t_depth != 0) return false;
		std::shared_ptr<RegList> L = rec->list_;
		std::lock_guard<ListMutex> lock(L->mu);
		rec->holder_ = holder;
		return true;
	}
	static void DropListReference(InFlightRecord* rec) { rec->list_.reset(); }

	// The walk at the outermost return (*The walk*): collect under the lock, release it, then drain.
	// The collection is a fixed batch, so the list-lock section allocates nothing and calls nothing
	// outside the registry; a full batch is followed by a re-scan (a drained record is unlinked, so a
	// re-scan finds only what is left). A full batch is followed by another scan, at most once per
	// batch of entries the list held when the walk began.
	static void Walk() {
		constexpr std::size_t kBatch = 16;
		RegList* L = t_own_list.get();
		if (L == nullptr) return;
		std::size_t scans_left = 0;
		bool first = true;
		for (;;) {
			InFlightHolder* holders[kBatch];
			std::size_t n = 0;
			uint64_t passed = 0;
			{
				std::lock_guard<ListMutex> lock(L->mu);
				if (first) scans_left = L->live / kBatch + 1;
				for (InFlightRecord* r = L->head; r != nullptr && n < kBatch; r = r->next_) {
					if (r->call_ != t_call) continue;
					InFlightHolder* h = r->holder_;
					const InFlightHandleBinding* b = h->handle_binding();
					if (b != nullptr && b->SubmittedNow()) {
						++passed;
						continue;
					}
					holders[n] = h;
					++n;
				}
			}
			first = false;
			// A scan that stopped short of the list's end (a full batch) is repeated, so only the scan
			// that reaches the end counts the records walked and passed.
			if (n < kBatch) g_walked.fetch_add(passed);
			for (std::size_t i = 0; i < n; ++i) {
				holders[i]->DrainFromWalk();
				g_reports.fetch_add(1);
#if defined(SUPERSLM_GPU_BOUNDARY_CHECK_STDERR)
				std::fprintf(stderr, "superslm_gpu: boundary token check: %s left in-flight work off a Submitted handle; drained\n",
				             t_entry_point ? t_entry_point : "(unnamed entry point)");
#endif
			}
			if (n < kBatch || scans_left-- == 0) break;
		}
	}

	static RegistryBinding Capture() {
		RegistryBinding b;
		if (t_depth > 0) {
			b.list_ = CurrentList();
			b.call_ = t_call;
			b.depth_ = t_depth;
		}
		return b;
	}
	static void Install(const RegistryBinding& b) {
		if (b.depth_ > 0) {
			t_bound_list = b.list_;
			t_call = b.call_;
			t_depth = b.depth_;
		} else {
			t_bound_list.reset();
			t_call = 0;
			t_depth = 0;
		}
	}
	static void Restore(const RegistryBinding& saved) {
		t_bound_list = saved.list_;
		t_call = saved.call_;
		t_depth = saved.depth_;
	}
	static RegistryBinding Save() {
		RegistryBinding b;
		b.list_ = t_bound_list;
		b.call_ = t_call;
		b.depth_ = t_depth;
		return b;
	}
};

void ReserveRegistration() { OwnList(); }

void RegisterNew(InFlightRecord* rec, InFlightHolder* holder) noexcept {
	RegistryAccess::Link(rec, holder);
	g_live_records.fetch_add(1);
}

void Reregister(InFlightRecord* rec, InFlightHolder* holder) noexcept {
	if (RegistryAccess::KeepAtDepthZero(rec, holder)) return;
	RegistryAccess::UnlinkFromList(rec);
	RegistryAccess::Link(rec, holder);
}

void Unlink(InFlightRecord* rec) noexcept {
	RegistryAccess::UnlinkFromList(rec);
	RegistryAccess::DropListReference(rec);
	g_live_records.fetch_sub(1);
}

GpuApiCallScope::GpuApiCallScope(const char* entry_point) noexcept {
	if (t_depth == 0) {
		// The outermost call: the thread's own list (a host thread's own call never runs under a
		// binding a trampoline installed: a task does not call into SuperSLM), a fresh call id,
		// and the guard flag cleared.
		OwnList();
		t_bound_list.reset();
		t_call = g_call_counter.fetch_add(1) + 1;
		t_guard_ran = false;
		t_entry_point = entry_point;
	}
	++t_depth;
}

GpuApiCallScope::~GpuApiCallScope() noexcept {
	if (--t_depth == 0) {
		RegistryAccess::Walk();
		t_guard_ran = false;
		t_entry_point = nullptr;
	}
}

void NoteEntrySubmitGuardRan() noexcept { t_guard_ran = true; }

bool BoundaryGuardCheckRefuses() noexcept { return t_depth > 0 && !t_guard_ran; }

RegistryBinding CaptureRegistryBinding() noexcept { return RegistryAccess::Capture(); }

GpuTaskBindingScope::GpuTaskBindingScope(const RegistryBinding& b) noexcept : saved_(RegistryAccess::Save()) {
	RegistryAccess::Install(b);
}

GpuTaskBindingScope::~GpuTaskBindingScope() noexcept { RegistryAccess::Restore(saved_); }

uint64_t BoundaryCheckReportCount() noexcept { return g_reports.load(); }
uint64_t BoundaryCheckWalkedCount() noexcept { return g_walked.load(); }
int64_t LiveRecordsForTest() noexcept { return g_live_records.load(); }
std::size_t OwnListEntriesForTest() noexcept {
	RegList* L = t_own_list.get();
	if (L == nullptr) return 0;
	std::lock_guard<ListMutex> lock(L->mu);
	return L->live;
}
int DepthForTest() noexcept { return t_depth; }

}  // namespace inflight
}  // namespace superslm_gpu

// T-2310 (Curie) -- Coverage Model dimension 7, capacity/allocation-count cell-group: Cells
// D-H (fold round 27) plus the calibration-currency cell, closing the fold-round-26/27 "S5"
// gap a code review found by execution: no cell in tests/ read capacity, slot count, or
// growth of any of the three growable types (Claude/Poirot/t2307-fold26-and-instrument-chain-
// review-2026-08-26.md, Finding 5), so reverting either half of fold round 26's own remedy
// (the `Grow()` formula correction or the `ContextEntry::counts{1}` hint) left the full
// 34,204-check regression green. Design of record:
// Claude/Vitruvius/t2265-superslm-fp-free-open-design-2026-08-24.md Sec7 dimension 7's fifth
// cell-group (Cells D-H, fold round 27) and sixth cell (calibration currency, fold round 27).
// Test-design record: Claude/Curie/t2310-fold27-growable-capacity-red-suite-2026-08-26.md.
//
// T-2312 UPDATE (2026-08-27, Claude/Curie/t2312-fold27-suite-repair-round-2026-08-27.md): fold
// round 27 landed for real (Claude/Brunel/t2311-fold27-build-2026-08-27.md, permanent -- `src/`
// and `include/` are read-only DELIVERABLES to this seat, not a state this suite reverts). Cells
// D/F/G/H are GREEN, unrevised. Cell E and Cell Calib both lost discrimination against the real
// landed build (T-2311 Sec10/Sec11, D-SLM4785/D-SLM4786) and were revised this round -- both
// GREEN again, at their own function definitions below, which carry the current, authoritative
// description of each cell's own method. The status table immediately below is T-2310's own
// authoring-time snapshot (fold 27 unbuilt) and is retained for the historical WHY; it does not
// describe current behavior for Cell E or Cell Calib.
//
// WHAT EACH CELL PROVED AT THIS SUITE'S OWN AUTHORING (T-2310, fold 27 not yet built):
//   Cell D -- GrowableIntMap's and GrowableContextMap's corrected `Grow()` formula (fold 26,
//             BUILT at this suite's authoring commit, 6041003) -- GREEN then, GREEN now.
//   Cell E -- `ContextEntry::counts{1}`'s own construction-site hint (fold 26, BUILT) --
//             GREEN then; lost discrimination once fold 27 landed for real, REVISED this
//             round (T-2312) -- see this cell's own function definition below.
//   Cell F -- GrowableIntSet's never-shrink compaction (fold 27, NOT YET BUILT at authoring)
//             -- RED then; fold 27 landed for real, GREEN now, unrevised.
//   Cell G -- GrowableContextMap's lazy construction (fold 27, NOT YET BUILT at authoring) --
//             RED then; fold 27 landed for real, GREEN now, unrevised.
//   Cell H -- order 1's own construction-site hint of 1 (fold 27, NOT YET BUILT at authoring)
//             -- RED then; fold 27 landed for real, GREEN now, unrevised.
//   Calibration-currency -- re-derives `AntiLmRetainedBytes`'s true understatement multiplier
//             at build/test time and asserts the published header range is a superset of it --
//             RED then (header stale); REVISED this round (T-2312) to read the published range
//             directly out of the header rather than a frozen copy -- see this cell's own
//             function definition below.
//
// T-2321 UPDATE (2026-08-27, Claude/Curie/t2321-cells-i-and-j-2026-08-27.md): Cell I authored,
// realizing design Sec7 dim 7's own seventh cell (fold round 28) -- see this cell's own
// function definition below for its full method and mutation-proof. Cell J follows in the next
// commit of this same round.
//   Cell I -- GrowableIntSet's own allocation count under steady erase-churn (fold 28,
//             D-SLM4796, NOT YET BUILT at this suite's current authoring commit -- Grow()
//             still carries fold round 27's formula) -- RED against the unmodified header,
//             for the design's own named reason (fold 27's formula does not fall with L);
//             flips GREEN under a temporary, reverted build of fold round 28's own target
//             formula.
//
// WHY CELLS F/G/H WERE ALLOWED TO BE RED AT T-2310's OWN HANDOFF (historical; fold 27 is landed
// now): fold round 27 was design-only as of this suite's authoring
// (`Claude/Vitruvius/t2265-fold27-superslm-fp-free-open-2026-08-26.md` Sec9, Verdict/status:
// "PROPOSED... neither T-2306's two adopted levers nor S1's repair is built"). T-2310's own exit
// condition was every cell reported RED-or-GREEN against 6041003 WITH its reason, and every
// falsifying mutation executed -- not every cell green. Cells F/G/H/Calib's own mutation-proof
// (T-2310, reported in that ticket's own test-design record) TEMPORARILY built fold 27's own
// printed fix in the engine tree, confirmed each cell flipped GREEN, then reverted the engine
// tree to the unbuilt state -- proving discrimination in BOTH directions without leaving fold 27
// built, which was not this seat's authority at that time (Curie.md, "Does not implement"; that
// ticket's own brief, Sec2 -- src/ and include/ are read-only to Curie as a DELIVERABLE; a
// temporary, reverted mutation to prove discrimination is the sanctioned method, used again this
// round for Cell Calib's own falsifying leg -- see that cell's own comment below).
//
// THE CAPACITY-OBSERVATION PROBLEM AND HOW THIS FILE SOLVES IT. None of GrowableIntSet,
// GrowableIntMap, or GrowableContextMap exposes a public capacity/slot-count accessor (by
// design -- adding one would be production code this seat does not write). Cells D/E/F/G/H all
// need to observe an EXACT slot count from outside the type. This file solves it with the same
// instrumented-global-allocator technique T-2299's code review and T-2302's build measured this
// exact codebase with (`Claude/Brunel/t2302-footprint-probe/footprint_probe.cpp`, global
// `operator new`/`operator delete` overrides, base vs. new, attribution per
// StandardsDocument.md Sec4/Sec7 -- this file's own instrumentation, windowing, and calibration
// are an independent construction built for this suite's own need, not a copy of that probe's
// text): every heap allocation's REQUESTED size is recorded while "armed," inside a WINDOW the
// test resets at each checkpoint. Since every one of these types' own `Grow()`/constructor does
// exactly one `std::vector::assign`/count-construction per allocation event, and that event's
// byte size is (new slot count) x sizeof(Slot), dividing an observed window's byte total by an
// independently CALIBRATED sizeof(Slot) recovers the exact slot count -- without ever reading a
// private member. Calibration is always taken at a KNOWN capacity (`BucketCountFor(hint)`,
// itself a public free function, called directly rather than hand-computed) so sizeof(Slot) is
// derived, never assumed. Every division below is checked for an exact (zero-remainder)
// quotient first -- a nonzero remainder means an assumption about what allocated in that window
// was wrong, and the CHECK fails loudly instead of silently rounding to a plausible-looking
// number.
//
// CELL H'S OWN MIRROR STRUCT. `AntiLmState::ContextEntry` (src/damped_greedy_antilm.cpp) is
// PRIVATE to that translation unit -- defined inside `AntiLmState`, itself an opaque type
// outside it (only reachable through the C-ABI-shaped free functions
// `AntiLmCreate`/`AntiLmUpdate`/`AntiLmPenalize`/`AntiLmRetainedBytes`/`AntiLmDestroy`). Cell H
// needs `sizeof(GrowableContextMap<ContextEntry>::Slot)` to interpret order 1's own outer-table
// allocation, and there is no way to obtain it without either a production accessor (not this
// seat's to add) or a field-for-field MIRROR of ContextEntry's own declaration, compiled by the
// same compiler under the same layout rules. `ContextEntryMirror` below is that mirror --
// copied by reading src/damped_greedy_antilm.cpp's own printed declaration directly (not
// guessed), same field types, same order, same default member initializers. It is a
// calibration instrument, never constructed as if it were the real type and never asserted
// equal to it (there is no way to check that from outside) -- if ContextEntry's own field set
// or order ever changes, this mirror must change with it or Cell H's own calibration silently
// measures the wrong type; this coupling is disclosed here rather than hidden.
//
// EXECUTION MODEL: no child-process isolation is needed here (unlike dim7_contract_red.cpp) --
// none of these cells crashes or traps; every one is an ordinary in-process assertion. This
// file does, however, override the process's global `operator new`/`operator delete` for its
// own binary (build_link_red.bat compiles each cell file into its own separate .exe, so this
// override is scoped to dim7_capacity_red.exe alone and does not affect any other cell file's
// binary).

#include "fixture_common.h"

#include "detail/int_hash.h"
#include "detail/context_hash.h"
#include "superslm/sslm_damped_greedy.h"

#include <atomic>
#include <cstdint>
#include <fstream>
#include <new>
#include <string>

// --- instrumented global allocator -------------------------------------------------------------
namespace t2310_alloc {

std::atomic<long long> g_live_bytes{0};
std::atomic<long long> g_window_bytes{0};
std::atomic<long long> g_window_events{0};
std::atomic<bool> g_armed{false};

// A 16-byte header in front of every allocation stores the requested size, so `operator
// delete(void*)` alone can find it again -- identical technique to
// Claude/Brunel/t2302-footprint-probe/footprint_probe.cpp.
constexpr std::size_t kHeader = 16;

inline void RecordAlloc(std::size_t sz) {
	if (!g_armed.load(std::memory_order_relaxed)) return;
	g_live_bytes.fetch_add(static_cast<long long>(sz), std::memory_order_relaxed);
	// g_window_bytes accumulates bytes NEWLY ALLOCATED since the last ResetWindow() -- it is
	// never decremented by a free. A Grow() event's own old-array free is accounted for by an
	// EARLIER window (the allocation that produced it); decrementing here would make a later
	// window's reading depend on an earlier window's own cleanup, which is not what "bytes
	// allocated during this interval" means.
	g_window_bytes.fetch_add(static_cast<long long>(sz), std::memory_order_relaxed);
	// g_window_events counts ALLOCATION EVENTS (calls to this function), not bytes -- added
	// for Cell I (below), which reads GrowableIntSet's own Grow() CALL COUNT rather than a
	// capacity. Reset alongside g_window_bytes, same lifetime.
	g_window_events.fetch_add(1, std::memory_order_relaxed);
}
inline void RecordFree(std::size_t sz) {
	if (!g_armed.load(std::memory_order_relaxed)) return;
	g_live_bytes.fetch_sub(static_cast<long long>(sz), std::memory_order_relaxed);
}

inline void Arm() { g_live_bytes = 0; g_window_bytes = 0; g_window_events = 0; g_armed = true; }
inline void Disarm() { g_armed = false; }
inline void ResetWindow() { g_window_bytes = 0; g_window_events = 0; }
inline long long ReadWindow() { return g_window_bytes.load(); }
inline long long ReadWindowEvents() { return g_window_events.load(); }
inline long long LiveBytes() { return g_live_bytes.load(); }

}  // namespace t2310_alloc

void* operator new(std::size_t sz) {
	void* raw = std::malloc(sz + t2310_alloc::kHeader);
	if (!raw) throw std::bad_alloc();
	*reinterpret_cast<std::size_t*>(raw) = sz;
	t2310_alloc::RecordAlloc(sz);
	return static_cast<char*>(raw) + t2310_alloc::kHeader;
}
void operator delete(void* p) noexcept {
	if (!p) return;
	char* raw = static_cast<char*>(p) - t2310_alloc::kHeader;
	std::size_t sz = *reinterpret_cast<std::size_t*>(raw);
	t2310_alloc::RecordFree(sz);
	std::free(raw);
}
void operator delete(void* p, std::size_t) noexcept { operator delete(p); }
void* operator new[](std::size_t sz) { return operator new(sz); }
void operator delete[](void* p) noexcept { operator delete(p); }
void operator delete[](void* p, std::size_t) noexcept { operator delete(p); }

// --- Cell H's calibration mirror (see this file's header comment) -------------------------------
struct ContextEntryMirror {
	// Field-for-field copy of `AntiLmState::ContextEntry`, src/damped_greedy_antilm.cpp:
	//   struct ContextEntry {
	//       GrowableIntMap<int32_t, int64_t, MixKeyS32> counts{1};
	//       int64_t total = 0;
	//   };
	superslm::detail::GrowableIntMap<int32_t, int64_t, superslm::detail::MixKeyS32> counts{1};
	int64_t total = 0;
};

namespace {

using superslm::detail::BucketCountFor;
using superslm::detail::ContextView;
using superslm::detail::GrowableContextMap;
using superslm::detail::GrowableIntMap;
using superslm::detail::GrowableIntSet;
using superslm::detail::MixKey64;
using superslm::detail::MixKeyS32;

// ==================================================================================
// CELL D -- GrowableIntMap's and GrowableContextMap's corrected Grow() formula (fold 26,
// BUILT). Design Sec7 dim 7 Cell D: drive to exactly 9 distinct keys from a construction hint
// of 8 (capacity 16) and assert the resulting capacity is exactly 32; drive to 17 and assert
// exactly 64. FALSIFYING MUTATION (executed and reverted this session, recorded in the
// test-design record): reverting the sizing line to `BucketCountFor((live_+1)*2)` must flip
// both assertions (9 distinct keys would then size to 64, not 32).
// ==================================================================================

void CellD_GrowableIntMap() {
	using GIM = GrowableIntMap<uint64_t, uint64_t, MixKey64>;
	t2310_alloc::Arm();

	GIM m(8);
	t2310_alloc::ResetWindow();
	m[0] = 0;  // first insert -- lazy alloc from hint 8 -> capacity BucketCountFor(8) = 16
	long long w0 = t2310_alloc::ReadWindow();
	uint64_t cap16 = BucketCountFor(8);
	CHECK_MSG(w0 > 0 && w0 % static_cast<long long>(cap16) == 0,
	          "GrowableIntMap first-insert allocation: %lld bytes not a clean multiple of "
	          "capacity %llu -- calibration assumption violated",
	          w0, (unsigned long long)cap16);
	long long sizeof_slot = w0 / static_cast<long long>(cap16);

	t2310_alloc::ResetWindow();
	for (uint64_t k = 1; k < 9; ++k) m[k] = k;  // keys 1..8 -> 9 distinct keys total
	long long w9 = t2310_alloc::ReadWindow();
	CHECK_MSG(w9 % sizeof_slot == 0,
	          "GrowableIntMap capacity-at-9 window (%lld bytes) not a clean multiple of "
	          "sizeof(Slot) (%lld) -- an unexpected allocation occurred",
	          w9, sizeof_slot);
	uint64_t cap_at_9 = static_cast<uint64_t>(w9 / sizeof_slot);
	CHECK_MSG(cap_at_9 == 32,
	          "GrowableIntMap capacity after 9 distinct keys (hint 8): got %llu, want 32 "
	          "(BucketCountFor(9))", (unsigned long long)cap_at_9);

	t2310_alloc::ResetWindow();
	for (uint64_t k = 9; k < 17; ++k) m[k] = k;  // keys 9..16 -> 17 distinct keys total
	long long w17 = t2310_alloc::ReadWindow();
	CHECK_MSG(w17 % sizeof_slot == 0,
	          "GrowableIntMap capacity-at-17 window (%lld bytes) not a clean multiple of "
	          "sizeof(Slot) (%lld)", w17, sizeof_slot);
	uint64_t cap_at_17 = static_cast<uint64_t>(w17 / sizeof_slot);
	CHECK_MSG(cap_at_17 == 64,
	          "GrowableIntMap capacity after 17 distinct keys (hint 8): got %llu, want 64 "
	          "(BucketCountFor(17))", (unsigned long long)cap_at_17);

	t2310_alloc::Disarm();
}

void CellD_GrowableContextMap() {
	using GCM = GrowableContextMap<uint64_t>;
	t2310_alloc::Arm();

	// Combined construction+first-insert window: robust to whether the constructor allocates
	// eagerly (current, pre-fold-27) or lazily (Cell G, fold 27) -- the array's own allocation
	// happens exactly once inside this window either way. STACK-allocated deliberately: a
	// heap-allocated `new GCM(8)` would add its own extra, unrelated allocation event
	// (sizeof(GCM) bytes for the object itself) into this window, contaminating the
	// calibration -- caught empirically during this suite's own authoring (a scratch
	// diagnostic trace showed an unexplained extra 40-byte event exactly matching
	// sizeof(GrowableContextMap<uint64_t>) before this was fixed).
	t2310_alloc::ResetWindow();
	GCM m(8);
	int32_t k0 = 0;
	m.FindOrEmplace(ContextView{&k0, 1}) = 0;
	long long w0 = t2310_alloc::ReadWindow();
	uint64_t cap16 = BucketCountFor(8);
	long long slot_bytes0 = w0 - 4;  // subtract the ctx_len=1 key-vector copy (1 x int32_t)
	CHECK_MSG(slot_bytes0 > 0 && slot_bytes0 % static_cast<long long>(cap16) == 0,
	          "GrowableContextMap construction+first-insert window (%lld bytes, -4 key-vector) "
	          "not a clean multiple of capacity %llu", w0, (unsigned long long)cap16);
	long long sizeof_slot = slot_bytes0 / static_cast<long long>(cap16);

	// GrowableContextMap::Grow()'s own rehash loop (unlike GrowableIntMap's/GrowableIntSet's)
	// re-CONSTRUCTS a brand-new `std::vector<int32_t>` key copy for every REHASHED slot too
	// (`FindOrEmplace(ContextView{s.key.data(), s.key.size()})`, context_hash.h), not only for
	// the newly-inserted key -- confirmed empirically during this suite's own authoring (a
	// scratch diagnostic trace of the 9th insert showed 10 NEW events, not 1: one 1280-byte
	// slots-array allocation plus nine 4-byte key-vector allocations, matching 8 rehashed old
	// keys + 1 new key). Each window below therefore subtracts `(new keys this window)*4`
	// PLUS, for the one window whose LAST key trips a Grow(), the rehashed-key term
	// `(live_ keys immediately before that Grow())*4` -- both counts are known exactly from
	// this cell's own population (Cell D's own dimension-1-derived thresholds, design Sec7).
	t2310_alloc::ResetWindow();
	for (int32_t k = 1; k < 9; ++k) {
		m.FindOrEmplace(ContextView{&k, 1}) = static_cast<uint64_t>(k);
	}
	long long w9 = t2310_alloc::ReadWindow();
	// 8 new distinct keys (1..8) + the Grow() at the 9th distinct key overall (k=8, the loop's
	// last iteration) rehashing the 8 keys already live at that instant (key 0, plus keys 1..7
	// inserted earlier in this same loop).
	long long key_vector_bytes9 = 8 * 4 + 8 * 4;
	long long slot_bytes9 = w9 - key_vector_bytes9;
	CHECK_MSG(slot_bytes9 % sizeof_slot == 0,
	          "GrowableContextMap capacity-at-9 window (%lld bytes, -%lld key-vectors) not a "
	          "clean multiple of sizeof(Slot) (%lld)", w9, key_vector_bytes9, sizeof_slot);
	uint64_t cap_at_9 = static_cast<uint64_t>(slot_bytes9 / sizeof_slot);
	CHECK_MSG(cap_at_9 == 32,
	          "GrowableContextMap capacity after 9 distinct contexts (hint 8): got %llu, want "
	          "32 (BucketCountFor(9))", (unsigned long long)cap_at_9);

	t2310_alloc::ResetWindow();
	for (int32_t k = 9; k < 17; ++k) {
		m.FindOrEmplace(ContextView{&k, 1}) = static_cast<uint64_t>(k);
	}
	long long w17 = t2310_alloc::ReadWindow();
	// 8 new distinct keys (9..16) + the Grow() at the 17th distinct key overall (k=16, this
	// loop's last iteration) rehashing the 16 keys already live at that instant.
	long long key_vector_bytes17 = 8 * 4 + 16 * 4;
	long long slot_bytes17 = w17 - key_vector_bytes17;
	CHECK_MSG(slot_bytes17 % sizeof_slot == 0,
	          "GrowableContextMap capacity-at-17 window (%lld bytes, -%lld key-vectors) not a "
	          "clean multiple of sizeof(Slot) (%lld)", w17, key_vector_bytes17, sizeof_slot);
	uint64_t cap_at_17 = static_cast<uint64_t>(slot_bytes17 / sizeof_slot);
	CHECK_MSG(cap_at_17 == 64,
	          "GrowableContextMap capacity after 17 distinct contexts (hint 8): got %llu, want "
	          "64 (BucketCountFor(17))", (unsigned long long)cap_at_17);

	t2310_alloc::Disarm();
}

// `AntiLmState::history_` (a `std::vector<int32_t>`) grows by one `push_back` per
// `AntiLmUpdate` call, UNCONDITIONALLY -- a term this file's own equations must account for
// separately from anything under test, since it has nothing to do with `tables_`/`counts`.
// Calibrated standalone (a bare `std::vector<int32_t>` reproduces the identical STL growth
// policy `history_` itself uses, independent of AntiLmState entirely) rather than hand-derived
// from the growth-factor formula, since the exact per-push allocation sizes are an incidental
// STL implementation detail this file should measure, not assume. Confirmed empirically during
// this suite's own authoring (a scratch diagnostic trace): push 1 allocates 4 bytes (capacity
// 0->1); push 2 allocates a further 8 bytes (capacity 1->2, freeing the earlier 4); this
// function's own window tracks NEW bytes only (matching `t2310_alloc`'s own accounting), so
// `HistoryPushBackBytes(1) == 4` and `HistoryPushBackBytes(2) == 4 + 8 == 12`.
long long HistoryPushBackBytes(int n_pushes) {
	t2310_alloc::ResetWindow();
	{
		std::vector<int32_t> v;
		for (int i = 0; i < n_pushes; ++i) v.push_back(0);
	}
	return t2310_alloc::ReadWindow();
}

// ==================================================================================
// CELL E -- `ContextEntry::counts{1}`'s own construction-site hint (fold 26, BUILT). Design
// Sec7 dim 7 Cell E: construct a fresh ContextEntry (through the real public surface -- see
// this file's header comment on why -- AntiLmCreate/AntiLmUpdate) and insert one candidate
// token; assert the resulting `counts` capacity is exactly 2, not the class default's 16.
// FALSIFYING MUTATION (executed and reverted this session, and again this round -- T-2312):
// reverting `counts{1}` to `counts{}` must flip this assertion (capacity 16 on the first
// insert).
//
// ISOLATION, T-2312-REVISED (D-SLM4785): the original method called `AntiLmCreate` UNARMED,
// relying on order 1's own outer GrowableContextMap<ContextEntry> table allocating EAGERLY at
// construction (true pre-fold-27) so that allocation happened before the armed window opened.
// Fold round 27 (D-SLM4779) made that table's own slot-array allocation LAZY -- first touch,
// not construction -- so it no longer happens inside the unarmed `AntiLmCreate` call at all; it
// now happens on the very same `AntiLmUpdate` call this cell measures, landing inside the same
// window as `counts{1}`'s own first-insert allocation and making the two terms inseparable by
// the old method (T-2311, Claude/Brunel/t2311-fold27-build-2026-08-27.md Sec10, executed both
// directions: both the correct build and a `counts{1}`->`counts{}` mutation FAILED at the same
// assertion with different, equally uninterpretable numbers -- zero discrimination). The fix
// keeps the window exactly where it was (opened after the unarmed `AntiLmCreate` call, around
// `AntiLmUpdate` alone -- `AntiLmCreate`'s own state-object and tables_-backing-array
// allocations are unconditional on every fold and correctly excluded either way) and instead
// SUBTRACTS order 1's own outer-table first-touch cost as an explicit, independently-calibrated
// term (BucketCountFor(1) slots of an independently-calibrated sizeof(Slot)), the same
// isolation Cell H already performs for order 1's own outer-table claim below. This isolation
// is fold-27-SPECIFIC: it assumes the outer table's slot array allocates lazily, inside this
// same `AntiLmUpdate` call, and subtracts that term out of the window it measures. That
// assumption is correct for this ticket's own permanently-built state (`src/`/`include/` are
// READ-ONLY deliverables to this seat; fold 27 is landed, not reverted, going forward) and is
// exactly what the mutation-proof below verifies by execution.
// ==================================================================================

void CellE_ContextEntryCountsHint() {
	using GIM_S32 = GrowableIntMap<int32_t, int64_t, MixKeyS32>;

	t2310_alloc::Arm();
	// Calibrate sizeof(GIM_S32::Slot) standalone, hint=8 -> capacity 16.
	t2310_alloc::ResetWindow();
	{
		GIM_S32 cal(8);
		cal[0] = 0;
	}
	long long w_cal = t2310_alloc::ReadWindow();
	uint64_t cap16 = BucketCountFor(8);
	CHECK_MSG(w_cal > 0 && w_cal % static_cast<long long>(cap16) == 0,
	          "GrowableIntMap<int32_t,int64_t,...> calibration window (%lld bytes) not a clean "
	          "multiple of capacity %llu", w_cal, (unsigned long long)cap16);
	long long sizeof_inner = w_cal / static_cast<long long>(cap16);

	// Calibrate sizeof(GrowableContextMap<ContextEntry>::Slot) -- order 1's own OUTER table's
	// own Slot, using the field-for-field ContextEntryMirror this file's header comment
	// establishes for Cell H (reused here, not redeclared: the Slot size depends on the value
	// type T, unlike GrowableContextMap<T>'s own fixed-size object -- calibrating against a
	// smaller T, e.g. uint64_t, would measure the wrong Slot size).
	t2310_alloc::ResetWindow();
	{
		GrowableContextMap<ContextEntryMirror> outer_cal(8);
		int32_t k = 0;
		outer_cal.FindOrEmplace(ContextView{&k, 1});
	}
	long long w_outer_cal = t2310_alloc::ReadWindow();
	uint64_t cap16_outer = BucketCountFor(8);
	long long slot_bytes_outer_cal = w_outer_cal - 4;  // subtract the ctx_len=1 key-vector copy
	CHECK_MSG(slot_bytes_outer_cal > 0 &&
	              slot_bytes_outer_cal % static_cast<long long>(cap16_outer) == 0,
	          "GrowableContextMap<ContextEntryMirror> calibration window (%lld bytes, -4 "
	          "key-vector) not a clean multiple of capacity %llu", w_outer_cal,
	          (unsigned long long)cap16_outer);
	long long sizeof_outer = slot_bytes_outer_cal / static_cast<long long>(cap16_outer);

	long long history_bytes = HistoryPushBackBytes(1);
	t2310_alloc::Disarm();

	// Real AntiLmState, single order. AntiLmCreate runs UNARMED -- its own state-object and
	// tables_-backing-array allocations happen unconditionally inside AntiLmCreate on every
	// fold (see this cell's own header comment); only order 1's own outer-table SLOT ARRAY
	// (accounted for above) moved into this AntiLmUpdate call under fold round 27's laziness.
	superslm::AntiLmState* state = superslm::AntiLmCreate(1);
	t2310_alloc::Arm();
	t2310_alloc::ResetWindow();
	superslm::AntiLmUpdate(state, /*token=*/0);
	long long w_full = t2310_alloc::ReadWindow();
	t2310_alloc::Disarm();
	superslm::AntiLmDestroy(state);

	// Subtract history_'s own unconditional push_back cost (see HistoryPushBackBytes's own
	// header comment above) and order 1's own outer-table first-touch allocation
	// (BucketCountFor(1) slots of sizeof_outer each) -- both unrelated to counts's own
	// capacity, but landing in the same window since AntiLmUpdate does all three in one call
	// under fold round 27's lazy construction (this cell's own header comment).
	long long order1_outer_bytes = static_cast<long long>(BucketCountFor(1)) * sizeof_outer;
	long long w = w_full - history_bytes - order1_outer_bytes;
	CHECK_MSG(w % sizeof_inner == 0,
	          "order 1's first-touch window (%lld bytes, -%lld history_ growth, -%lld order-1 "
	          "outer-table first-touch) not a clean multiple of sizeof(counts's own Slot) "
	          "(%lld) -- an unexpected allocation occurred",
	          w, history_bytes, order1_outer_bytes, sizeof_inner);
	uint64_t counts_capacity = static_cast<uint64_t>(w / sizeof_inner);
	CHECK_MSG(counts_capacity == 2,
	          "ContextEntry::counts's own capacity on its first insert: got %llu, want 2 "
	          "(BucketCountFor(1), the fold-round-26 construction-site hint `counts{1}`)",
	          (unsigned long long)counts_capacity);
}

// ==================================================================================
// CELL F -- GrowableIntSet's never-shrink compaction (fold 27, Sec3.5, NOT YET BUILT).
// Design Sec7 dim 7 Cell F: grow a table to a measured capacity C (any population sufficient
// to trigger at least one live_-driven growth), then erase down to a small live count and
// insert-and-erase enough distinct further keys to trip a tombstone-driven trigger; assert the
// post-trigger capacity is >= C, never smaller. FALSIFYING MUTATION: reverting to
// `BucketCountFor(live_+1)` alone (fold round 26's own formula, without the `max()` against
// current capacity) must flip this assertion -- the executed counterexample in the design's
// own Sec3.5 text (capacity 16, live_=1, tombstones_=7, shrinking to capacity 4) is the
// concrete instance this mutation reproduces.
//
// This cell needs no calibrated sizeof(Slot): GrowableIntSet<uint64_t, MixKey64>'s Key is POD
// (no per-key heap cost), so `t2310_alloc::LiveBytes()` is directly proportional to the
// table's own current capacity at every point in the sequence, and a `>=` comparison in bytes
// is exactly a `>=` comparison in capacity -- no division needed.
// ==================================================================================

void CellF_GrowableIntSetNeverShrink() {
	using GIS = GrowableIntSet<uint64_t, MixKey64>;
	t2310_alloc::Arm();

	GIS s(8);
	// Step 1: grow via a live_-driven trigger to a measured capacity C. Insert 9 distinct
	// keys (hint 8 -> capacity 16; the 9th insertion's own pre-check, live_=8: (8+1)*2=18>16,
	// trips Grow(), sizing to BucketCountFor(9)=32 -- matches Cell D's own arithmetic exactly,
	// no tombstone term since tombstones_=0 throughout this step).
	for (uint64_t k = 0; k < 9; ++k) s.InsertOrReclaim(k);
	long long capacity_c_bytes = t2310_alloc::LiveBytes();
	CHECK_MSG(capacity_c_bytes > 0, "GrowableIntSet step 1: expected a nonzero live footprint "
	                                "after 9 distinct inserts, got %lld", capacity_c_bytes);

	// Step 2: erase 8 of the 9, leaving key 0 alone -- live_=1, tombstones_=8, capacity
	// unchanged at C (erase never grows or shrinks).
	for (uint64_t k = 1; k < 9; ++k) CHECK(s.Erase(k));
	CHECK_MSG(t2310_alloc::LiveBytes() == capacity_c_bytes,
	          "GrowableIntSet step 2 (erase only): live footprint changed from %lld to %lld -- "
	          "erase alone must never reallocate", capacity_c_bytes, t2310_alloc::LiveBytes());

	// Step 3: insert-and-erase 40 further distinct keys (never previously used), one round at
	// a time -- each round's own erase deterministically creates a tombstone at the slot it
	// just occupied (Erase always tombstones on a hit, regardless of whether that round's own
	// insert reclaimed an existing tombstone or used a fresh empty slot), so tombstones_ is
	// monotonically non-decreasing across rounds and cannot get stuck below the trigger
	// threshold indefinitely -- 40 rounds against a threshold that trips at tombstones_ >= 15
	// (from (live_+tombstones_+1)*2 > 32 with live_=1) is a wide, empirically-confirmed
	// margin (see this file's own test-design record for the executed trace). The invariant
	// under test -- capacity never drops below C -- is checked after EVERY round, not only at
	// the end, so this cell does not depend on knowing exactly which round trips the trigger.
	//
	// "The trigger fired at least once" is checked via NEW allocation bytes (t2310_alloc's own
	// window, reset each round), never via a live-byte CHANGE: the corrected (fixed) formula's
	// own `max()` can leave the resulting capacity IDENTICAL to C (`max(BucketCountFor(2), 32)
	// == 32`), so a genuine Grow() -- a real reallocation, old array freed and a new one of the
	// same size allocated -- produces zero NET live-byte change even though it definitely ran.
	// A live-byte-change check would therefore wrongly conclude the trigger never fired on a
	// CORRECT build (caught empirically during this suite's own authoring, mutation-proofing
	// this exact cell: the fold-27 fix applied, `capacity_ever_changed` read false against a
	// build that was visibly, correctly holding the invariant).
	bool trigger_fired = false;
	for (uint64_t k = 100; k < 140; ++k) {
		t2310_alloc::ResetWindow();
		CHECK(s.InsertOrReclaim(k));
		CHECK(s.Erase(k));
		if (t2310_alloc::ReadWindow() > 0) trigger_fired = true;
		long long now = t2310_alloc::LiveBytes();
		CHECK_MSG(now >= capacity_c_bytes,
		          "GrowableIntSet tombstone-driven trigger: live footprint dropped from %lld "
		          "to %lld at key %llu -- capacity shrank below its own measured C",
		          capacity_c_bytes, now, (unsigned long long)k);
	}
	CHECK_MSG(trigger_fired,
	          "GrowableIntSet step 3: no round allocated anything -- the tombstone-driven "
	          "trigger never fired across 40 rounds, so this cell did not exercise the "
	          "property it exists to pin; widen the round count or re-check the trigger "
	          "arithmetic");

	t2310_alloc::Disarm();
}

// ==================================================================================
// CELL G -- GrowableContextMap's lazy construction (fold 27, Sec3.6, T-2306 Sec8.1 adopted,
// NOT YET BUILT). Design Sec7 dim 7 Cell G: construct a table and never call FindOrEmplace;
// assert both Find returns nullptr for any query and the table's own diagnostic state shows
// zero slot-array allocation. FALSIFYING MUTATION: reverting the constructor to its
// pre-fold-27 eager form (`mask_(...), slots_(mask_+1)`) must flip this assertion.
// ==================================================================================

void CellG_GrowableContextMapLazy() {
	using GCM = GrowableContextMap<uint64_t>;
	t2310_alloc::Arm();
	t2310_alloc::ResetWindow();
	{
		GCM m(8);
		int32_t k = 0;
		CHECK(m.Find(ContextView{&k, 1}) == nullptr);  // correctness half -- miss by construction
	}
	long long w = t2310_alloc::ReadWindow();
	CHECK_MSG(w == 0,
	          "GrowableContextMap construction, never touched: expected zero heap allocation "
	          "(fold-round-27 lazy construction), got %lld bytes", w);
	t2310_alloc::Disarm();
}

// ==================================================================================
// CELL H -- order 1's own construction-site hint of 1 (fold 27, Sec3.6, T-2306 Sec8.3
// adopted, NOT YET BUILT). Design Sec7 dim 7 Cell H: construct AntiLmState with any
// max_order >= 1 and touch order 1's table once; assert order 1's own resulting capacity is
// exactly 2, and every other order's own first-touch capacity is unaffected (still 16, the
// class default). FALSIFYING MUTATION: reverting order 1's construction expression to the
// class default must flip the FIRST assertion only.
//
// See this file's header comment for the calibration method (ContextEntryMirror) and the
// derivation this function implements: two independent AntiLmState instances (Test A, order 1
// alone; Test B, orders 1 and 2 together) with the arm window opened BEFORE AntiLmCreate in
// both, so each instance's own construction-or-first-touch allocation (whichever timing the
// current build uses) lands inside the measured window regardless. Order 1's own outer-table
// term, derived from Test A, is subtracted out of Test B's combined window to isolate order
// 2's own term.
// ==================================================================================

void CellH_Order1Hint() {
	using GIM_S32 = GrowableIntMap<int32_t, int64_t, MixKeyS32>;
	using GCM_Mirror = GrowableContextMap<ContextEntryMirror>;

	// sizeof(GrowableContextMap<T>) does not depend on T: the class holds only
	// `uint64_t mask_; uint64_t live_; std::vector<Slot> slots_;` -- Slot (T-dependent) lives
	// entirely inside the vector's own heap-allocated buffer, never inline in the object
	// itself. This is a compile-time fact (no calibration needed) and is what lets `tables_`'s
	// own backing-array cost (below) be computed for the REAL `GrowableContextMap<ContextEntry>`
	// via the generic, directly-instantiable `GrowableContextMap<uint64_t>`.
	constexpr long long kSizeofGcmObject = sizeof(GrowableContextMap<uint64_t>);
	static_assert(sizeof(GrowableContextMap<uint64_t>) == sizeof(GCM_Mirror),
	              "GrowableContextMap<T>'s own object size must not depend on T");

	// The (opaque) AntiLmState object's own heap allocation, from `AntiLmCreate`'s own
	// `return new AntiLmState(max_order);` (src/damped_greedy_antilm.cpp). AntiLmState cannot
	// be named or sizeof()'d from this file (it is defined entirely inside that translation
	// unit, forward-declared everywhere else) -- this is the one constant in this cell that is
	// NOT independently derivable from a public type, and it is stated here rather than
	// silently assumed. Justified two ways: (1) field-by-field, from AntiLmState's own printed
	// declaration -- `std::vector<GrowableContextMap<ContextEntry>> tables_` (24 bytes, a
	// vector's own control block), `std::vector<int32_t> history_` (24 bytes), `std::size_t
	// retained_bytes_` (8 bytes), `int max_order_` (4 bytes, padded to 8 for the struct's own
	// 8-byte alignment) = 64 bytes; (2) confirmed empirically during this suite's own
	// authoring, via a scratch diagnostic (not shipped) that logged every individual
	// allocation `AntiLmCreate(1)` makes: three separate events of 64, 40, and 1408 bytes --
	// the 64 attributable to nothing else in that call (the 40 matches kSizeofGcmObject above
	// exactly; the 1408 matches 16 * 88, confirming sizeof(Slot) for
	// GrowableContextMap<ContextEntry> is 88 bytes, exactly the figure this design's own text
	// states). This value does not change across fold 27: that fold touches
	// GrowableContextMap's own constructor and AntiLmState::tables_'s own construction
	// expression, never AntiLmState's own field set.
	constexpr long long kAntiLmStateObjectBytes = 64;

	t2310_alloc::Arm();

	// Calibrate sizeof(GIM_S32::Slot) (the counts sub-table's own Slot), hint=8.
	t2310_alloc::ResetWindow();
	{ GIM_S32 cal(8); cal[0] = 0; }
	long long w_inner_cal = t2310_alloc::ReadWindow();
	uint64_t cap16_inner = BucketCountFor(8);
	CHECK_MSG(w_inner_cal > 0 && w_inner_cal % static_cast<long long>(cap16_inner) == 0,
	          "counts sub-table calibration window (%lld bytes) not a clean multiple of "
	          "capacity %llu", w_inner_cal, (unsigned long long)cap16_inner);
	long long sizeof_inner = w_inner_cal / static_cast<long long>(cap16_inner);

	// Calibrate sizeof(GrowableContextMap<ContextEntryMirror>::Slot), STACK-allocated (no
	// object-size term to subtract, unlike a heap `new`), via a combined
	// construction+first-insert window (see this file's header comment: robust to eager or
	// lazy construction, and to whether Cell G has landed).
	t2310_alloc::ResetWindow();
	{
		GCM_Mirror cal(8);
		int32_t k = 0;
		cal.FindOrEmplace(ContextView{&k, 1});
	}
	long long w_outer_cal = t2310_alloc::ReadWindow();
	uint64_t cap16_outer = BucketCountFor(8);
	long long slot_bytes_outer_cal = w_outer_cal - 4;  // subtract the ctx_len=1 key-vector
	CHECK_MSG(slot_bytes_outer_cal > 0 &&
	              slot_bytes_outer_cal % static_cast<long long>(cap16_outer) == 0,
	          "GrowableContextMap<ContextEntryMirror> calibration window (%lld bytes, -4 "
	          "key-vector) not a clean multiple of capacity %llu", w_outer_cal,
	          (unsigned long long)cap16_outer);
	long long sizeof_outer = slot_bytes_outer_cal / static_cast<long long>(cap16_outer);

	long long history_1 = HistoryPushBackBytes(1);
	long long history_2 = HistoryPushBackBytes(2);

	// Test A: order 1 in isolation (max_order=1). Arm opened BEFORE AntiLmCreate, so order 1's
	// own allocation lands in this window whether it happens at construction (eager,
	// pre-fold-27) or at first touch (lazy, post-fold-27).
	t2310_alloc::ResetWindow();
	superslm::AntiLmState* state_a = superslm::AntiLmCreate(1);
	int32_t candidates[1] = {0};
	int64_t p_omega[1];
	superslm::AntiLmPenalize(state_a, candidates, 1, p_omega);
	superslm::AntiLmUpdate(state_a, /*token=*/0);
	long long w_a = t2310_alloc::ReadWindow();
	superslm::AntiLmDestroy(state_a);

	// w_a = kAntiLmStateObjectBytes (the AntiLmState object itself)
	//     + 1*kSizeofGcmObject (tables_'s own backing array, one element)
	//     + order1_outer_bytes (order 1's own GrowableContextMap<ContextEntry>::Slot array --
	//       the unknown this test solves for)
	//     + 2*sizeof_inner (counts{1}'s own first insert, capacity BucketCountFor(1)=2)
	//     + history_1 (history_'s own first push_back)
	//     + 0 (ctx_len=0 for order 1 always -- no key-vector cost, ever).
	long long order1_counts_first = 2 * sizeof_inner;
	long long order1_outer_bytes =
	    w_a - kAntiLmStateObjectBytes - kSizeofGcmObject - order1_counts_first - history_1;
	CHECK_MSG(order1_outer_bytes > 0 && order1_outer_bytes % sizeof_outer == 0,
	          "Test A window (%lld bytes) does not resolve to a clean order-1 outer-slot term "
	          "(-%lld state object, -%lld tables_ backing, -%lld counts, -%lld history_) -- got "
	          "%lld, not a multiple of sizeof(outer Slot) %lld",
	          w_a, kAntiLmStateObjectBytes, kSizeofGcmObject, order1_counts_first, history_1,
	          order1_outer_bytes, sizeof_outer);
	uint64_t order1_capacity = static_cast<uint64_t>(order1_outer_bytes / sizeof_outer);
	CHECK_MSG(order1_capacity == 2,
	          "order 1's own first-touch outer-table capacity: got %llu, want 2 "
	          "(BucketCountFor(1), the fold-round-27 construction-site hint)",
	          (unsigned long long)order1_capacity);

	// Test B: orders 1 and 2 together (max_order=2). Second AntiLmUpdate call reuses the SAME
	// token as the first -- this does NOT avoid a second allocation on order 1's own counts
	// sub-table: GrowableIntMap::operator[]'s own growth pre-check
	// (`(live_+1)*2 > slots_.size()`) fires on every call once live_ reaches the threshold,
	// whether or not the specific key already exists (int_hash.h, operator[], checked BEFORE
	// the probe that would find the existing key) -- so order 1's counts table (capacity 2,
	// live_=1 after call 1) unavoidably regrows to BucketCountFor(2)=4 during call 2. This is
	// accounted for explicitly below, not avoided.
	t2310_alloc::ResetWindow();
	superslm::AntiLmState* state_b = superslm::AntiLmCreate(2);
	superslm::AntiLmPenalize(state_b, candidates, 1, p_omega);
	superslm::AntiLmUpdate(state_b, /*token=*/0);
	superslm::AntiLmPenalize(state_b, candidates, 1, p_omega);
	superslm::AntiLmUpdate(state_b, /*token=*/0);  // same token both calls, by design (above)
	long long w_b = t2310_alloc::ReadWindow();
	superslm::AntiLmDestroy(state_b);

	uint64_t regrow_capacity = BucketCountFor(2);  // = 4
	long long order1_counts_total =
	    order1_counts_first + static_cast<long long>(regrow_capacity) * sizeof_inner;
	long long order2_counts_first = 2 * sizeof_inner;  // order 2's own counts{1}, its call
	long long order2_key_vector = 4;                   // ctx_len=1, one int32_t

	// w_b = kAntiLmStateObjectBytes + 2*kSizeofGcmObject (tables_'s own backing array, two
	// elements) + order1_outer_bytes (from Test A, assumed identical -- order 1's own hint
	// does not depend on max_order) + order2_outer_bytes (the unknown this test solves for) +
	// order1_counts_total + order2_counts_first + order2_key_vector + history_2.
	long long known_terms = kAntiLmStateObjectBytes + 2 * kSizeofGcmObject + order1_outer_bytes +
	                         order1_counts_total + order2_counts_first + order2_key_vector +
	                         history_2;
	long long order2_outer_bytes = w_b - known_terms;
	CHECK_MSG(order2_outer_bytes > 0 && order2_outer_bytes % sizeof_outer == 0,
	          "Test B window (%lld bytes) does not resolve to a clean order-2 outer-slot term "
	          "after subtracting known terms (%lld) -- got %lld, not a multiple of "
	          "sizeof(outer Slot) %lld",
	          w_b, known_terms, order2_outer_bytes, sizeof_outer);
	uint64_t order2_capacity = static_cast<uint64_t>(order2_outer_bytes / sizeof_outer);
	CHECK_MSG(order2_capacity == 16,
	          "order 2's own first-touch outer-table capacity: got %llu, want 16 (the class "
	          "default, unaffected by order 1's own construction-site hint)",
	          (unsigned long long)order2_capacity);

	t2310_alloc::Disarm();
}

// ==================================================================================
// CELL I -- GrowableIntSet's own allocation count under steady erase-churn (Sec3.5 above,
// fold round 28, D-SLM4796, closing T-2313 S1's cost half). Design Sec7 dim 7 Cell I:
// construct a table, insert keys until live_ reaches a fixed population L, then run 1,000
// rounds of one Erase (a key currently held) plus one InsertOrReclaim (a fresh key), counting
// every Grow() call from construction onward -- the counted window includes the initial
// fill's own Grow() calls, stated explicitly (T-2316 Note, fold round 29, D-SLM4807: only
// this reading gives K, below, anything to "cover," and both readings execute to the
// identical PASS/FAIL verdict, so this is a clarity fix, not a discrimination change). Assert
// the total count is <= ceil(1000 / (L + 1)) + K, K = 8, at L in {7, 31, 127, 511}.
//
// FALSIFYING MUTATION: reverting the sizing line to BucketCountFor(live_+1) alone (fold round
// 27's own formula, no 2x headroom) must flip this assertion at every tested L -- this IS the
// formula shipped at this suite's own build commit (Grow() carries fold round 27's formula,
// not fold round 28's target; design Sec2, T-2319's own audit Sec2), so this cell is RED
// against the unmodified header without any mutation, and flips GREEN under a temporary,
// reverted build of fold round 28's own BucketCountFor(2*(live_+1)) formula -- both executed,
// see this cell's own test-design record for the transcript.
//
// EVENT COUNTING, NOT BYTES. GrowableIntSet's own Grow() is the only place this type ever
// allocates once its constructor has returned (int_hash.h) -- every allocation EVENT this
// file's instrumented allocator records after construction is exactly one Grow() call, no
// division or sizeof(Slot) calibration needed (unlike Cells D/E/H/J, which read an EXACT
// capacity and therefore need byte-to-slot-count division). The table is constructed UNARMED
// so its own single construction-time allocation is never counted: Grow() is a private
// method reached only from InsertOrReclaim's own growth pre-check, never from the
// constructor, so the constructor's own allocation is not a "Grow() call" under this cell's
// own definition -- arming the window only after construction returns is what keeps it out.
// ==================================================================================

uint64_t CellI_GrowCallCount(uint64_t L) {
	using GIS = GrowableIntSet<uint64_t, MixKey64>;
	GIS s(8);  // unarmed -- see this cell's own header comment

	t2310_alloc::Arm();  // one continuous window: the initial fill AND all 1,000 churn rounds

	std::vector<uint64_t> live_keys;
	live_keys.reserve(L);
	for (uint64_t k = 0; k < L; ++k) {
		CHECK(s.InsertOrReclaim(k));
		live_keys.push_back(k);
	}

	uint64_t next_fresh_key = L;
	for (int round = 0; round < 1000; ++round) {
		uint64_t idx = static_cast<uint64_t>(round) % L;
		CHECK(s.Erase(live_keys[idx]));       // a key currently held
		uint64_t fresh_key = next_fresh_key++;
		CHECK(s.InsertOrReclaim(fresh_key));  // a fresh key, never used before
		live_keys[idx] = fresh_key;
	}

	long long events = t2310_alloc::ReadWindowEvents();
	t2310_alloc::Disarm();
	return static_cast<uint64_t>(events);
}

void CellI_GrowableIntSetChurnAllocationCount() {
	constexpr uint64_t kL[4] = {7, 31, 127, 511};
	constexpr uint64_t kRounds = 1000;
	constexpr uint64_t kK = 8;
	for (uint64_t L : kL) {
		uint64_t grow_calls = CellI_GrowCallCount(L);
		uint64_t bound = (kRounds + L) / (L + 1) + kK;  // ceil(1000/(L+1)) + K
		CHECK_MSG(grow_calls <= bound,
		          "GrowableIntSet Grow() call count under steady erase-churn (L=%llu, 1000 "
		          "rounds, counted from construction onward, including the initial fill's own "
		          "Grow() calls): got %llu, want <= %llu (ceil(1000/(L+1)) + %llu) -- the "
		          "corrected 2*(live_+1) formula's own amortized-cost bound",
		          (unsigned long long)L, (unsigned long long)grow_calls,
		          (unsigned long long)bound, (unsigned long long)kK);
	}
}

// ==================================================================================
// CELL CALIB -- the calibration-currency check (fold 27, D-SLM4778, S2). Design Sec7 dim 7,
// sixth cell: at build/test time, re-derive the true understatement multiplier using T-2299's
// own method (global operator new accounting, base vs. shipped construction) at max_order in
// {1, 3, 5}, and assert the published header range at each order is a superset of the freshly
// computed value.
//
// T-2312-REVISED (D-SLM4786): the original method compared against `kPublishedRanges`, a
// frozen C++ constant CITING the header's own text at this suite's authoring commit --
// deliberately hard-coded because "a C++ comment is not machine-readable." That is the wrong
// pin: it goes red the moment the header is edited for any reason (this exact suite's own
// fold-27 build round hit this -- the header was corrected to the TRUE ranges and the cell
// stayed red, comparing the correct new figures against the stale frozen copy), and if the
// header and the copy are ever edited TOGETHER by hand, the two can drift in lockstep with
// nothing to catch it -- the exact failure a currency check exists to prevent
// (Claude/Brunel/t2311-fold27-build-2026-08-27.md Sec11).
//
// The header IS machine-readable, at the one thing this cell actually needs: its own
// published-range lines have a fixed, disclosed text shape --
// "// max_order=<N>: <lo>x-<hi>x low" (three lines, AntiLmRetainedBytes's own comment,
// include/superslm/sslm_damped_greedy.h) -- and ParsePublishedRange below reads them directly
// out of the header file at test run time via `sscanf` on that exact shape, never a looser
// scan. This makes the header the single source: the only way to turn this cell green is to
// make the header state a range that actually contains the freshly measured figure. A header
// edit that changes a number is picked up automatically (no second file to remember to touch);
// a header edit that breaks the line's own text shape fails this cell loudly, at
// ParsePublishedRange's own CHECK_MSG below, rather than silently reading zero ranges and
// passing vacuously.
//
// FALSIFYING MUTATION: this cell must fail when the header names a range disjoint from the
// true figure -- executed and reverted this round (T-2312) the same way Cells F/G/H's own
// mutation-proofs touch production text (a temporary, reverted edit, comment-only here: the
// header's own published-range lines were rewritten to the pre-fold-27 stale figures --
// "1.31x-2.61x"/"3.85x-4.98x"/"4.54x-5.60x", the exact figures the ORIGINAL frozen-copy
// mechanism cited -- rebuilt, confirmed all three orders FAIL against the now-stale text, then
// reverted to the real, correct figures and confirmed green again; see this cell's own
// test-design record for the executed transcript).
//
// WORKLOAD: reuses T-2299's/T-2302's own primary three cells verbatim
// (Claude/Brunel/t2302-footprint-probe/footprint_probe.cpp -- max_order=1, 20k tok, vocab
// 4096; max_order=3, 20k tok, vocab 4096; max_order=5, 50k tok, vocab 8192), same fixed-seed
// xorshift64* stream construction, attribution per StandardsDocument.md Sec4/Sec7. Metric:
// live bytes / AntiLmRetainedBytes reading, measured at the end of the run -- the identical
// metric definition the header's own published ranges were derived from
// (Claude/Brunel/t2302-fp-free-open-lazy-alloc-build-2026-08-26.md Sec3's own table header,
// "measured range (live bytes / AntiLmRetainedBytes reading)").
// ==================================================================================

constexpr int kOrders[3] = {1, 3, 5};
constexpr int32_t kVocab[3] = {4096, 4096, 8192};
constexpr long long kTokenCounts[3] = {20000, 20000, 50000};

// Opens the header this cell reads its published ranges from. Two candidate relative paths,
// covering both known invocation shapes: build_link_red.bat (this suite's own build script)
// sets this exe's own working directory to this suite's own directory
// (tests/t2296-fp-free-open-red-suite) before running it (the first candidate resolves from
// there); a caller running the exe directly from the engine worktree root is covered by the
// second.
bool OpenPublishedRangesHeader(std::ifstream& f) {
	const char* candidates[] = {
	    "../../include/superslm/sslm_damped_greedy.h",
	    "include/superslm/sslm_damped_greedy.h",
	};
	for (const char* c : candidates) {
		f.open(c);
		if (f.good()) return true;
		f.clear();
	}
	return false;
}

// Reads `order`'s own published understatement range directly out of the header's text --
// see this cell's own header comment above for the exact line shape and why this is the
// single source of truth. Returns false if the header cannot be opened or no line for `order`
// matches the expected shape (a real finding, not a soft miss -- CHECK_MSG'd by the caller).
bool ParsePublishedRange(int order, double* lo, double* hi) {
	std::ifstream f;
	if (!OpenPublishedRangesHeader(f)) return false;
	char prefix[32];
	std::snprintf(prefix, sizeof(prefix), "max_order=%d:", order);
	std::string line;
	while (std::getline(f, line)) {
		std::size_t pos = line.find(prefix);
		if (pos == std::string::npos) continue;
		double parsed_lo = 0.0, parsed_hi = 0.0;
		if (std::sscanf(line.c_str() + pos, "max_order=%*d: %lfx-%lfx", &parsed_lo,
		                 &parsed_hi) == 2) {
			*lo = parsed_lo;
			*hi = parsed_hi;
			return true;
		}
	}
	return false;
}

uint64_t NextRand(uint64_t& state) {
	uint64_t x = state;
	x ^= x >> 12;
	x ^= x << 25;
	x ^= x >> 27;
	state = x;
	return x * 0x2545F4914F6CDD1DULL;
}

void CellCalib_AntiLmRetainedBytesCurrency() {
	for (int i = 0; i < 3; ++i) {
		uint64_t rng = 0x9E3779B97F4A7C15ULL ^ static_cast<uint64_t>(kOrders[i]) ^
		               (static_cast<uint64_t>(kVocab[i]) << 20) ^
		               (static_cast<uint64_t>(kTokenCounts[i]) << 1);
		if (rng == 0) rng = 0xDEADBEEFULL;

		t2310_alloc::Arm();
		superslm::AntiLmState* state = superslm::AntiLmCreate(kOrders[i]);
		int32_t candidates[8];
		int64_t p_omega[8];
		for (long long t = 0; t < kTokenCounts[i]; ++t) {
			for (int c = 0; c < 8; ++c) {
				candidates[c] = static_cast<int32_t>(NextRand(rng) % static_cast<uint64_t>(kVocab[i]));
			}
			superslm::AntiLmPenalize(state, candidates, 8, p_omega);
			superslm::AntiLmUpdate(
			    state, static_cast<int32_t>(NextRand(rng) % static_cast<uint64_t>(kVocab[i])));
		}
		long long live_bytes = t2310_alloc::LiveBytes();
		std::size_t reported = superslm::AntiLmRetainedBytes(state);
		superslm::AntiLmDestroy(state);
		t2310_alloc::Disarm();

		CHECK_MSG(reported > 0, "max_order=%d: AntiLmRetainedBytes reported 0 after %lld "
		                        "tokens -- cannot compute an understatement ratio",
		          kOrders[i], kTokenCounts[i]);
		double understatement = static_cast<double>(live_bytes) / static_cast<double>(reported);

		double lo = 0.0, hi = 0.0;
		bool parsed = ParsePublishedRange(kOrders[i], &lo, &hi);
		CHECK_MSG(parsed,
		          "max_order=%d: could not find or parse a 'max_order=%d: <lo>x-<hi>x low' "
		          "published-range line in include/superslm/sslm_damped_greedy.h -- the "
		          "header's own AntiLmRetainedBytes comment shape changed; update this cell's "
		          "own parse pattern to match",
		          kOrders[i], kOrders[i]);
		if (!parsed) continue;  // nothing to compare the measured figure against

		CHECK_MSG(understatement >= lo && understatement <= hi,
		          "max_order=%d: computed understatement %.4fx (live_bytes=%lld, reported=%zu) "
		          "falls outside the published header range [%.2fx, %.2fx] -- the header's own "
		          "calibration is stale for this cell",
		          kOrders[i], understatement, live_bytes, reported, lo, hi);
	}
}

}  // namespace

int main() {
	std::printf("=== dim7_capacity_red: Coverage Model dimension 7 capacity/allocation cells "
	            "(D-H, calibration currency) ===\n");

	std::printf("--- Cell D: GrowableIntMap/GrowableContextMap's corrected Grow() formula "
	            "(fold 26, built) ---\n");
	CellD_GrowableIntMap();
	CellD_GrowableContextMap();

	std::printf("--- Cell E: ContextEntry::counts{1}'s own construction-site hint (fold 26, "
	            "built) ---\n");
	CellE_ContextEntryCountsHint();

	std::printf("--- Cell F: GrowableIntSet's never-shrink compaction (fold 27, not yet "
	            "built) ---\n");
	CellF_GrowableIntSetNeverShrink();

	std::printf("--- Cell G: GrowableContextMap's lazy construction (fold 27, not yet "
	            "built) ---\n");
	CellG_GrowableContextMapLazy();

	std::printf("--- Cell H: order 1's own construction-site hint of 1 (fold 27, not yet "
	            "built) ---\n");
	CellH_Order1Hint();

	std::printf("--- Cell I: GrowableIntSet's own allocation count under steady erase-churn "
	            "(fold 28, not yet built) ---\n");
	CellI_GrowableIntSetChurnAllocationCount();

	std::printf("--- Cell Calib: AntiLmRetainedBytes calibration currency ---\n");
	CellCalib_AntiLmRetainedBytesCurrency();

	std::printf("\n%d checks, %d failures, %d skips.\n", GChecks, GFailures, GSkips);
	return GFailures == 0 ? 0 : 1;
}

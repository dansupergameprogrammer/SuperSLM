// Paged-KV plan (rev 16.1) §7, step C2: the engine's guard-vitality cells -- 11.1 (the coverage
// guard, §3.2 item 4) and 11.9's engine part (guard 5 at a multi-page view, §3.2 item 2).
//
// At the engine the test owns the page buffer, so "before any write" is graded by a raw read of it:
// the whole buffer is snapshotted before the call and compared after, and the page a guard-removed
// mutant would write is read raw on its own. Each refusal also leaves the sequence as it was. Each
// case has a control at the same construction with the one field corrected, which must be admitted
// and must write: that is what shows the construction separates the mutant (§7, the separation rule).
// The cases whose mutant writes through a null or wrapped address run in a child process
// (pkv_engine_child.h), so the mutant ends the child, not the suite.

#include "pkv_engine_child.h"
#include "pkv_engine_helpers.h"

#include <cstring>

namespace {

using namespace pkv_engine;

struct SeqSnap {
	int64_t context_length;
	uint32_t layer_index;
	std::vector<int8_t> hidden;
	uint64_t sat;
};
SeqSnap Snap(const Seq& s) {
	return {s.st.context_length, s.st.layer_index, s.hidden, s.st.kv_saturation_count};
}
bool Same(const SeqSnap& a, const Seq& s) {
	return a.context_length == s.st.context_length && a.layer_index == s.st.layer_index && a.hidden == s.hidden &&
	       a.sat == s.st.kv_saturation_count;
}

// A sequence resting at a token boundary at `context_length`, its hidden row the embedding of a fixed
// token. Rows [0, context_length) are whatever the pages hold (kFill): no guard reads them.
void Rest(const EngineFixture& e, Seq* s, int64_t context_length) {
	SslmForwardStatus st = Embed(e, 7, s->hidden.data(), &s->st.hidden_scale);
	PKV_CHECK_EQ(static_cast<int>(st), static_cast<int>(SslmForwardStatus::Ok));
	s->st.layer_index = 0;
	s->st.context_length = context_length;
}

// One chunk of `n` embedded tokens.
struct Chunk {
	std::vector<int8_t> codes;
	std::vector<CarriedScale> scales;
	Chunk(const EngineFixture& e, size_t n) : codes(n * e.Hidden()), scales(n) {
		for (size_t i = 0; i < n; ++i) Embed(e, static_cast<int32_t>(11 + i), codes.data() + i * e.Hidden(), &scales[i]);
	}
};

bool PageIsFill(PagedKv& kv, uint32_t physical) {
	const uint8_t* p = kv.Page(physical);
	for (size_t i = 0; i < kv.page_bytes; ++i)
		if (p[i] != kFill) return false;
	return true;
}

// 11.1 [C2]: a view one page short -> KvPageUnmapped before any write. The table entry one past
// `mapped` names the foreign page (a page of the test's buffer the view does not map); the
// guard-removed mutant writes its row there, and the raw read of that page catches it. Then the
// one-page view at context_length == cap still returns KvCapacityExhausted (§3.2 item 4's order).
void Cell111() {
	const EngineFixture& e = GetEngineFixture("pkv_def");
	if (!e.ok) return;
	for (int64_t B : {int64_t{4}, int64_t{16}}) {
		// Single-token: the row at 2B is the first row of logical page 2; two pages are mapped.
		{
			PagedKv kv(e, B);
			kv.table[2] = kv.ForeignPage();
			Seq s(e);
			Rest(e, &s, 2 * B);
			const std::vector<uint8_t> before = kv.Snapshot();
			const SeqSnap seq_before = Snap(s);
			const SslmForwardStatus st = StepToken(e, s.st, kv.View(2), 0);
			// kills: the coverage guard removed (the row lands in the foreign page).
			PKV_CHECK_MSG(st == SslmForwardStatus::KvPageUnmapped, "B=%lld single-token one page short: %s",
			              static_cast<long long>(B), StatusName(st));
			// The raw read of the test's own page buffer.
			PKV_CHECK_MSG(PageIsFill(kv, kv.ForeignPage()), "B=%lld: the foreign page was written",
			              static_cast<long long>(B));
			// kills: the guard placed after the first write (the K/V landing, a RoPE write-back).
			PKV_CHECK_MSG(kv.Equals(before), "B=%lld single-token: the page buffer changed under a refusal",
			              static_cast<long long>(B));
			PKV_CHECK_MSG(Same(seq_before, s), "B=%lld single-token: the sequence changed under a refusal",
			              static_cast<long long>(B));
			// Control: mapped = 3 admits the call, and its row lands in the foreign page.
			const SslmForwardStatus ok = StepToken(e, s.st, kv.View(3), 0);
			PKV_CHECK_MSG(ok == SslmForwardStatus::Ok && !PageIsFill(kv, kv.ForeignPage()),
			              "B=%lld control (three pages mapped): %s, foreign page written %d", static_cast<long long>(B),
			              StatusName(ok), !PageIsFill(kv, kv.ForeignPage()));
		}
		// Chunk: rows [2B - 3, 2B + 2) span logical pages 1 and 2; two pages are mapped.
		{
			PagedKv kv(e, B);
			kv.table[2] = kv.ForeignPage();
			Chunk c(e, 5);
			SequenceLayerState counters;
			const std::vector<uint8_t> before = kv.Snapshot();
			const std::vector<int8_t> codes_before = c.codes;
			const SslmForwardStatus st = StepChunk(e, c.codes.data(), c.scales.data(), 5, 2 * B - 3, kv.View(2), counters);
			// kills: the chunk path's coverage guard removed, or checking only the chunk's first row.
			PKV_CHECK_MSG(st == SslmForwardStatus::KvPageUnmapped, "B=%lld chunk one page short: %s",
			              static_cast<long long>(B), StatusName(st));
			PKV_CHECK_MSG(PageIsFill(kv, kv.ForeignPage()), "B=%lld chunk: the foreign page was written",
			              static_cast<long long>(B));
			// kills: a guard that fires per token, after the chunk's in-view rows [2B - 3, 2B) were written.
			PKV_CHECK_MSG(kv.Equals(before) && c.codes == codes_before,
			              "B=%lld chunk: the page buffer or the chunk changed under a refusal", static_cast<long long>(B));
			const SslmForwardStatus ok = StepChunk(e, c.codes.data(), c.scales.data(), 5, 2 * B - 3, kv.View(3), counters);
			PKV_CHECK_MSG(ok == SslmForwardStatus::Ok && !PageIsFill(kv, kv.ForeignPage()),
			              "B=%lld chunk control (three pages mapped): %s", static_cast<long long>(B), StatusName(ok));
		}
	}
	// The one-page view (B = cap, table {0}, page_bytes = the block): at context_length == cap both
	// guards would refuse, and KvCapacityExhausted fires first (test_main.cpp's flat cell keeps its
	// status, G30).
	{
		const int64_t cap = e.Cap();
		PagedKv kv(e, cap);
		PKV_CHECK_EQ(kv.logical, 1);
		PKV_CHECK_EQ(kv.table[0], 0);
		Seq s(e);
		Rest(e, &s, cap);
		const std::vector<uint8_t> before = kv.Snapshot();
		const SeqSnap seq_before = Snap(s);
		const SslmForwardStatus st = StepToken(e, s.st, kv.View(1), 0);
		// kills: the coverage guard ordered before KvCapacityExhausted (KvPageUnmapped here).
		PKV_CHECK_MSG(st == SslmForwardStatus::KvCapacityExhausted, "one-page view at the cap: %s", StatusName(st));
		PKV_CHECK_MSG(kv.Equals(before) && Same(seq_before, s), "one-page view at the cap: a write or a state change");
		Chunk c(e, 3);
		SequenceLayerState counters;
		const SslmForwardStatus cst = StepChunk(e, c.codes.data(), c.scales.data(), 3, cap - 2, kv.View(1), counters);
		// kills: the same misordering in the chunk loop (its guard 5 sits between the two, §3.2 item 4).
		PKV_CHECK_MSG(cst == SslmForwardStatus::KvCapacityExhausted, "one-page view, chunk past the cap: %s",
		              StatusName(cst));
		PKV_CHECK_MSG(kv.Equals(before), "one-page view, chunk past the cap: the page buffer changed");
		// Control: the last admissible write, row cap - 1, through the same view.
		Rest(e, &s, cap - 1);
		const SslmForwardStatus ok = StepToken(e, s.st, kv.View(1), 0);
		PKV_CHECK_MSG(ok == SslmForwardStatus::Ok && !kv.Equals(before), "one-page view at cap - 1: %s", StatusName(ok));
		PKV_CHECK_MSG(PageIsFill(kv, kv.ForeignPage()), "one-page view: the page after the block was written");
	}
}

// 11.9 [C2]: guard 5 at a multi-page view (B < cap). page_bytes one byte short -> WorkspaceTooSmall;
// a null pool_base -> WorkspaceTooSmall; an overflowing per-page product L * 2 * H_kv * B * D ->
// InvalidContextCap. Each before any write, in both loops. The guard-removed mutant writes with the
// short stride (caught by the raw read of the buffer), through the null base, or at a wrapped
// offset (each ends the child).
void Cell119Engine() {
	const EngineFixture& e = GetEngineFixture("pkv_def");
	if (!e.ok) return;
	for (int64_t B : {int64_t{4}, int64_t{16}}) {
		const std::string tag = "B=" + std::to_string(B);
		// The position written: B + 1, in logical page 1, with two pages mapped (the coverage guard is
		// satisfied, so only guard 5 can refuse).
		const int64_t at = B + 1;

		// page_bytes one byte short.
		{
			PagedKv kv(e, B);
			KvPageView v = kv.View(2);
			v.page_bytes = kv.page_bytes - 1;
			Seq s(e);
			Rest(e, &s, at);
			const std::vector<uint8_t> before = kv.Snapshot();
			const SeqSnap seq_before = Snap(s);
			const SslmForwardStatus st = StepToken(e, s.st, v, 0);
			// kills: guard 5's view meaning removed (the row lands at the short stride).
			PKV_CHECK_MSG(st == SslmForwardStatus::WorkspaceTooSmall, "%s single-token page_bytes - 1: %s", tag.c_str(),
			              StatusName(st));
			PKV_CHECK_MSG(kv.Equals(before) && Same(seq_before, s), "%s single-token page_bytes - 1: a write or a state change",
			              tag.c_str());
			Chunk c(e, 3);
			SequenceLayerState counters;
			const SslmForwardStatus cst = StepChunk(e, c.codes.data(), c.scales.data(), 3, at - 1, v, counters);
			PKV_CHECK_MSG(cst == SslmForwardStatus::WorkspaceTooSmall, "%s chunk page_bytes - 1: %s", tag.c_str(),
			              StatusName(cst));
			PKV_CHECK_MSG(kv.Equals(before), "%s chunk page_bytes - 1: the page buffer changed", tag.c_str());
			// Control: the exact page_bytes is admitted and writes.
			const SslmForwardStatus ok = StepToken(e, s.st, kv.View(2), 0);
			PKV_CHECK_MSG(ok == SslmForwardStatus::Ok && !kv.Equals(before), "%s control (exact page_bytes): %s",
			              tag.c_str(), StatusName(ok));
		}

		// A null pool_base: in a child, since the mutant writes through it.
		{
			const ChildEnd end = RunInChild(("11.9-null-" + tag).c_str(), [&]() {
				PagedKv kv(e, B);
				KvPageView v = kv.View(2);
				v.pool_base = nullptr;
				Seq s(e);
				Rest(e, &s, at);
				const SeqSnap seq_before = Snap(s);
				const SslmForwardStatus st = StepToken(e, s.st, v, 0);
				Chunk c(e, 3);
				SequenceLayerState counters;
				const SslmForwardStatus cst = StepChunk(e, c.codes.data(), c.scales.data(), 3, at - 1, v, counters);
				const std::vector<uint8_t> fill(kv.mem.n, kFill);
				const bool ok = st == SslmForwardStatus::WorkspaceTooSmall && cst == SslmForwardStatus::WorkspaceTooSmall &&
				                Same(seq_before, s) && kv.Equals(fill);
				if (!ok) std::fprintf(stderr, "null pool_base: single-token %s, chunk %s\n", StatusName(st), StatusName(cst));
				return ok;
			});
			// kills: the null check removed (the child faults or exits 1).
			PKV_CHECK_MSG(Passed(end), "%s null pool_base: the child %s", tag.c_str(), end.Describe().c_str());
		}
	}

	// The overflowing per-page product. A multi-page view needs B < cap, so the cap is 2^62 and B is
	// 2^61 (two pages): L * 2 * H_kv * B * D = 2 * 2 * 2 * 2^61 * 48 overflows size_t. The pool_base
	// is the test's real buffer. With the check removed, §3.1's in-page offset ((l*2 + half)*H_kv + h)
	// * B * D wraps to 0 mod 2^64 (B * D = 3 * 2^65), so every layer, half and head lands on the same
	// row of logical page 0 in the test's buffer, which the raw compare reads; any other wrap faults the
	// child. Because B < cap, this product overflows only where L * cap * H_kv * D * 2 does too, so a
	// build that still checks the flat product's overflow passes this case (equivalent here, and
	// harmless: the one-byte-short case above refuses any build that compares page_bytes to it).
	{
		const ChildEnd end = RunInChild("11.9-overflow", [&]() {
			PagedKv kv(e, 16);
			const int64_t big_cap = int64_t{1} << 62;
			const int64_t big_B = int64_t{1} << 61;
			KvPageView v = kv.View(2);
			v.page_positions = big_B;
			Seq s(e);
			Rest(e, &s, 1);
			const SeqSnap seq_before = Snap(s);
			const SslmForwardStatus st = superslm::RunLayerLoop(
			    s.st, e.layers.data(), e.L(), e.L(), e.Hidden(), e.D(), e.Hkv(), e.C().intermediate_size, big_cap,
			    e.view.rope_tables, v, superslm::OptionGKLandingMode::kLegacy, {}, 0, nullptr, e.QWidth(), GemmThreading{});
			Chunk c(e, 3);
			SequenceLayerState counters;
			const SslmForwardStatus cst = superslm::RunLayerLoopChunkBatched(
			    c.codes.data(), c.scales.data(), 3, e.layers.data(), e.L(), e.Hidden(), e.D(), e.Hkv(),
			    e.C().intermediate_size, big_cap, 0, e.view.rope_tables, v, false, &counters.kv_saturation_count, {},
			    nullptr, e.QWidth(), nullptr, nullptr, nullptr, nullptr, GemmThreading{});
			const std::vector<uint8_t> fill(kv.mem.n, kFill);
			const bool ok = st == SslmForwardStatus::InvalidContextCap && cst == SslmForwardStatus::InvalidContextCap &&
			                Same(seq_before, s) && kv.Equals(fill);
			if (!ok) std::fprintf(stderr, "overflow: single-token %s, chunk %s\n", StatusName(st), StatusName(cst));
			return ok;
		});
		// kills: the per-page product's overflow check removed (the child writes at a wrapped offset,
		// faults, or exits 1).
		PKV_CHECK_MSG(Passed(end), "overflowing per-page product: the child %s", end.Describe().c_str());
	}
}

PKV_CELL("11.1", "C2", Cell111);
PKV_CELL("11.9/C2", "C2", Cell119Engine);

}  // namespace

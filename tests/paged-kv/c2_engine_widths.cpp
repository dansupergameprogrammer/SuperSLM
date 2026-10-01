// Paged-KV plan (rev 16.1) §7, step C2: the engine's width and geometry matrix and its byte-equality
// spine -- cells 4.1 [C2], 4.6 [C2], 6.1 [C2] and 7.7.
//
// Each cell drives the view overloads of both layer loops (pkv_engine_api.h) over pages the test owns,
// through a scattered page table (pkv_engine_helpers.h), at B = 4, B = 16 and the one-page view
// (B = cap, table {0}: the flat wrappers' own view). Two drives per width: the chunk loop in 64-token
// calls then the single-token loop for the decode step (as sslm_prefill and sslm_decode_step drive
// the flat loops), and the single-token loop for every token. Both are graded against the same R0
// record: per-token RunLayerLoop and the chunk loop are bit-identical by design (forward_sites.h,
// RunLayerLoopChunkBatched's comment), so both must reproduce the `widths` scenario v1.11.0's ABI
// recorded. The runs are shared between the cells (computed once per process); each cell grades its
// own operands.

#include "pkv_engine_helpers.h"

#include <cstring>

namespace {

using namespace pkv_engine;

const Drive kDrives[] = {Drive::kChunked, Drive::kPerToken};

// The widths of 4.1 on this fixture, checked against the plan's numbers (B = 16, cap 4096).
bool WidthsArePlans(const pkv::Fixture& fx) {
	const std::vector<int64_t> w = Widths(fx);
	const std::vector<int64_t> plan = {1, 15, 16, 17, 32, 1064, 4095, 4096};
	PKV_CHECK_MSG(w == plan, "%s: 4.1's widths are not the plan's {1, 15, 16, 17, 32, 1064, 4095, 4096}",
	              fx.stem.c_str());
	return w == plan;
}

std::string Label(const pkv::Fixture& fx, int64_t B, Drive d, int64_t W) {
	return fx.stem + " B=" + std::to_string(B) + " " + DriveName(d) + " W=" + std::to_string(W);
}

// Tokens and statuses: what 4.1 and 4.6 grade.
void GradeTokens(const EngineFixture& e, int64_t B, Drive d, int64_t W) {
	const pkv::Fixture& fx = *e.fx;
	const WidthRun& r = RunWidth(e, B, d, W);
	const WidthRef ref = LookupWidth(fx, W);
	const std::string at = Label(fx, B, d, W);
	// kills: a coverage guard off by one at a page edge (each call maps exactly ceil(end / B) pages),
	// and a view loop that rejects a multi-page view at all.
	PKV_CHECK_MSG(r.fail.empty(), "%s: %s", at.c_str(), r.fail.c_str());
	if (!r.fail.empty()) return;
	// kills: a per-page score or context loop that drops, repeats or misaligns a page run (the ready
	// token is the softmax over all W positions). A build that ignores the table in its reads and its
	// writes alike is self-consistent and keeps every token; 6.1's raw rows are what catch it.
	PKV_CHECK_MSG(ref.ready && ref.ready->tokens == std::vector<int32_t>{r.ready},
	              "%s: ready token %d differs from v1.11.0's", at.c_str(), r.ready);
	if (W < e.Cap()) {
		// kills: a landing and an attention read that disagree on row W's table entry (the step attends
		// over its own row), and a last run of `rows = min(B, width - p0)` taken as B.
		PKV_CHECK_MSG(ref.decode1 && ref.decode1->tokens == std::vector<int32_t>{r.decode1},
		              "%s: decode token %d differs from v1.11.0's", at.c_str(), r.decode1);
		PKV_CHECK_EQ(r.final_length, W + 1);
	} else {
		// 4.1's "cap (the last admissible write)": row cap - 1 was written; the step past it is refused.
		// kills: the coverage guard ordered before KvCapacityExhausted (a full view at the cap would
		// report KvPageUnmapped), and any write at or past the cap.
		PKV_CHECK_MSG(r.cap_refused, "%s: %s", at.c_str(), r.cap_fail.c_str());
		PKV_CHECK_EQ(r.final_length, e.Cap());
	}
}

// Per-position K/V bytes: what 6.1 grades.
void GradeBytes(const EngineFixture& e, int64_t B, Drive d, int64_t W) {
	const pkv::Fixture& fx = *e.fx;
	const WidthRun& r = RunWidth(e, B, d, W);
	const WidthRef ref = LookupWidth(fx, W);
	const std::string at = Label(fx, B, d, W);
	PKV_CHECK_MSG(r.fail.empty(), "%s: %s", at.c_str(), r.fail.c_str());
	if (!r.fail.empty()) return;
	// kills: a K/V landing (or a K RoPE write-back, or a K read-before-rotate) that addresses the flat
	// layout or ignores the table: rows [0, W) gathered from the test's pages by §3.1's formula must
	// digest to the rows v1.11.0 saved.
	PKV_CHECK_MSG(ref.prefill && r.rows_prefill == ref.prefill->rows_sha, "%s: K/V rows [0, %lld) differ from v1.11.0's",
	              at.c_str(), static_cast<long long>(W));
	// kills: a saturating site counted twice or skipped on a page run (the counter is part of the state
	// the reference recorded at this stage).
	PKV_CHECK_MSG(ref.prefill && r.sat_prefill == ref.prefill->saturation, "%s: saturation %lld vs %lld", at.c_str(),
	              static_cast<long long>(r.sat_prefill), static_cast<long long>(ref.prefill ? ref.prefill->saturation : -1));
	if (W < e.Cap()) {
		PKV_CHECK_MSG(ref.decode1 && r.rows_decode1 == ref.decode1->rows_sha,
		              "%s: K/V rows [0, %lld) after the decode step differ from v1.11.0's", at.c_str(),
		              static_cast<long long>(W + 1));
		PKV_CHECK_MSG(ref.decode1 && r.sat_decode1 == ref.decode1->saturation, "%s: saturation after decode %lld vs %lld",
		              at.c_str(), static_cast<long long>(r.sat_decode1),
		              static_cast<long long>(ref.decode1 ? ref.decode1->saturation : -1));
	}
	// kills: a write outside the rows the calls owned -- into an unmapped table entry's page, the foreign
	// page, or another row's bytes (a page stride taken as B * D, a head stride taken as cap * D).
	PKV_CHECK_MSG(r.stray < 0, "%s: byte %lld of the test's page buffer was written outside rows [0, %lld)", at.c_str(),
	              r.stray, static_cast<long long>(r.final_length));
	// kills: view accessors that ignore the table or use the flat formula (§3.2 item 1).
	PKV_CHECK_MSG(r.accessors_ok, "%s: a view accessor does not address §3.1's formula (%s)", at.c_str(),
	              r.accessor_fail.c_str());
}

// 4.1 [C2]: widths 1, B-1, B, B+1, 2B, prefix + 64, cap-1 and cap (the last admissible write) at the
// engine, on pkv_def (B = 16, cap 4096), at B = 4, 16 and the one-page view, both loops. Graded by the
// R0 reference's tokens and by the loops' own statuses.
void Cell41Engine() {
	const EngineFixture& e = GetEngineFixture("pkv_def");
	if (!e.ok || !WidthsArePlans(*e.fx)) return;
	for (int64_t B : ViewPositions(e))
		for (Drive d : kDrives)
			for (int64_t W : Widths(*e.fx)) GradeTokens(e, B, d, W);
}

// 4.6 [C2]: both score branches run per page. pkv_def takes GemmInt8AccumulateRow (QK-norm stripped,
// heads of 48); pkv_qk keeps QK-norm (heads of 128) and takes the direct_qk branch, QkQ31Score per row
// (G2). The box's 0.5B (D 64) and 1.5B (D 128) artifacts are the ABI leg's (C4); in the cloud these two
// generated geometries carry the engine leg. Every width of 4.1, every view, both loops: tokens and
// rows equal the reference.
void Cell46Engine() {
	for (const char* stem : {"pkv_def", "pkv_qk"}) {
		const EngineFixture& e = GetEngineFixture(stem);
		if (!e.ok) continue;
		const bool want_direct = std::strcmp(stem, "pkv_qk") == 0;
		// The construction: the two fixtures really do take the two branches (else the cell separates
		// nothing for the branch it names).
		PKV_CHECK_MSG(e.DirectQk() == want_direct, "%s: direct_qk is %d, the construction needs %d", stem, e.DirectQk(),
		              want_direct);
		PKV_CHECK_EQ(e.D(), want_direct ? 128 : 48);
		if (!WidthsArePlans(*e.fx)) continue;
		for (int64_t B : ViewPositions(e))
			for (Drive d : kDrives)
				for (int64_t W : Widths(*e.fx)) {
					// kills (pkv_qk): a direct_qk branch that scores keys from the row's base without the page
					// run's offset, or reads past a page edge as if the keys were contiguous.
					GradeTokens(e, B, d, W);
					GradeBytes(e, B, d, W);
				}
	}
}

// 6.1 [C2]: tokens and per-position K/V bytes equal the reference at every width of 4.1, on both
// geometries, at every view, through both loops. The bytes are read raw from the test's own pages,
// addressed by §3.1's formula written in the test, and digested in the reference's order.
void Cell61Engine() {
	for (const char* stem : {"pkv_def", "pkv_qk"}) {
		const EngineFixture& e = GetEngineFixture(stem);
		if (!e.ok || !WidthsArePlans(*e.fx)) continue;
		for (int64_t B : ViewPositions(e))
			for (Drive d : kDrives)
				for (int64_t W : Widths(*e.fx)) {
					GradeTokens(e, B, d, W);
					GradeBytes(e, B, d, W);
				}
	}
}

// 7.7 [C2]: every "by construction" in the new engine code maps to 6.1 or 6.2. This cell runs neither
// (§9.3: "7.7 maps claims to 6.1 and 6.2 and runs neither"); it records the mapping and fails when a
// cell it maps to is missing from this executable.
//
//   claim (plan §)                                                          pinned by
//   one-page identity: the flat overloads are thin wrappers over the       6.1/C2, the one-page view leg
//     one-page view (pool_base = workspace, table {0}, mapped 1, B = cap,   (B = cap, table {0}, page_bytes =
//     page_bytes = workspace_size), so every legacy caller is unchanged      the block) equals v1.11.0 at
//     (§3.1, §3.2 item 2)                                                    every width, both loops
//   the one-page view's per-page loop runs once with rows = width, the     6.1/C2, the same leg
//     identical call made today (§3.2 item 3)
//   guard 5's statuses and their order are today's for the one-page view   11.1's one-page leg (status at
//     (§3.2 item 2)                                                          the cap); 6.1/C2 (no refusal
//                                                                            below it)
//   per-page attention is exact: the same int64 terms in the same          6.2 (chained runs equal one call
//     ascending order, associativity, G3's width-independent bound         at probs 2^15, values +-127,
//     (§3.2 item 3)                                                          width = cap); 6.1/C2 at B = 4, 16
//   the chunk loop is bit-identical to the single-token loop through a     6.1/C2: both drives against one
//     view (forward_sites.h, design §15.2)                                   reference record
//   page_bytes is only the stride between pages (§3.2 item 2)              6.1/C2 (rows read at the formula's
//                                                                            stride); 11.9/C2 (one byte short)
void Cell77() {
	for (const char* mapped : {"6.1/C2", "6.2"}) {
		bool found = false;
		for (const pkv::CellEntry& c : pkv::Registry()) found = found || std::strcmp(c.id, mapped) == 0;
		// kills: a mapping to a cell that no longer exists in the C2 suite.
		PKV_CHECK_MSG(found, "7.7 maps claims to %s, which this executable does not register", mapped);
	}
}

PKV_CELL("4.1/C2", "C2", Cell41Engine);
PKV_CELL("4.6/C2", "C2", Cell46Engine);
PKV_CELL("6.1/C2", "C2", Cell61Engine);
PKV_CELL("7.7", "C2", Cell77);

}  // namespace

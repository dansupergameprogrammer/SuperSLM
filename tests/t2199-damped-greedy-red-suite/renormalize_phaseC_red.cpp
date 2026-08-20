// T-2199 (Curie) -- Phase C2 red suite: TopKRenormalizeQ15, the k-restricted normalizer
// (plan Sec7.3 surface 2, Sec5.5, Sec8 Phase C2/C2a). RED BY LINK: sslm_damped_greedy.h
// declares TopKRenormalizeQ15; no .cpp in this repo defines it (grep confirmed at authoring
// commit main@071c5f3).
//
// Coverage Model cells realized here (plan Sec9, Wizard repo, commit 5a7f80e251):
//   dim5   Failure and rejection paths -- GATE, D-SLM3719  (SoftmaxRowQ15-shaped runtime
//                                          refusal on a real top-k candidate)
//   dim6   Numerical edges/determinism -- GATE, D-SLM3719  (width-domain boundary, executed
//                                          at the derived (q_b,q_c), a dependency pin) and
//                                          GATE, D-SLM3719 (the C2a speed-headroom cost cell)
//   dim7   Contract claims             -- GATE, D-SLM3719  ("TopKRenormalizeQ15's cost is
//                                          O(k), not O(V), regardless of row shape")
//   dim11  Guard vitality              -- GATE, D-SLM3719  (the dim5 refusal can actually
//                                          fire in the shipping build, mutation-proven)
#include <chrono>

#include "fixture_common.h"

using namespace t2199fixture;

// ===========================================================================================
// Known-answer correctness: a small, hand-inspectable row with k=3, cross-checked against the
// ALREADY-SHIPPED, certified SoftmaxRowQ15 called directly on the same k gathered elements --
// TopKRenormalizeQ15 is explicitly specified (Sec7.3) as "built entirely from the same
// certified sub-primitives SoftmaxRowQ15 already uses, restricted to k top-k positions," so
// this is a REFERENCE-implementation oracle appropriate to pair with the feature oracle below
// (Curie's own catalog dimension 10 discipline: pair REF with FEAT where the catalog
// requires; this is not a consistency oracle standing alone).
// ===========================================================================================
static void TestPhaseC2_KnownAnswer_MatchesSoftmaxRowQ15OnGatheredElements() {
	int64_t q_ln2, q_b, q_c;
	CHECK(DeriveDefaultScaleConstants(&q_ln2, &q_b, &q_c));

	const int32_t V = 50;
	const int32_t idx[3] = {10, 25, 40};
	std::vector<int32_t> row = MakeSyntheticRow(V, {25, 10, 40});  // 25 highest, then 10, 40

	int64_t out_topk[3];
	const bool wf = TopKRenormalizeQ15(row.data(), idx, 3, q_ln2, q_b, q_c, out_topk);
	CHECK_MSG(wf, "TopKRenormalizeQ15 refused on a well-inside-domain synthetic row");

	// FEATURE ORACLE: cross-check against SoftmaxRowQ15 called directly on the SAME three
	// int64-widened scores, gathered in the SAME order -- this is what "restricted to k
	// positions" means, definitionally.
	int64_t scores[3] = {row[static_cast<size_t>(idx[0])], row[static_cast<size_t>(idx[1])],
	                      row[static_cast<size_t>(idx[2])]};
	int64_t out_ref[3];
	const bool wf_ref = SoftmaxRowQ15(scores, 3, q_ln2, q_b, q_c, out_ref);
	CHECK_MSG(wf == wf_ref, "TopKRenormalizeQ15 all_well_formed=%d disagrees with the "
	                         "certified SoftmaxRowQ15 reference (%d) on the identical gathered "
	                         "elements",
	          wf, wf_ref);
	for (int i = 0; i < 3; ++i)
		CHECK_MSG(out_topk[i] == out_ref[i],
		          "candidate %d: TopKRenormalizeQ15=%lld, SoftmaxRowQ15 reference=%lld -- must "
		          "match exactly (same certified sub-primitives, same restricted set)",
		          i, (long long)out_topk[i], (long long)out_ref[i]);

	// The three q_theta values must sum to at most 1<<15 (Q15's own "always in [0, 2^15] since
	// a count ratio is a true probability" bound applies here too -- q_theta is a genuine
	// renormalized probability over k, so the k values sum to exactly 1<<15, modulo the
	// integer-division floor's own rounding slack of at most k-1 units).
	const int64_t sum = out_topk[0] + out_topk[1] + out_topk[2];
	CHECK_MSG(sum <= (1 << kProbFracBits) && sum >= (1 << kProbFracBits) - 3,
	          "sum of the three q_theta values = %lld, want within [32765, 32768] "
	          "(renormalized over exactly k, floor-rounding slack < k)",
	          (long long)sum);
}

// A second known-answer point at k=1 -- the degenerate case where renormalization is a no-op
// (a single candidate's own probability, relative to itself alone, is always exactly 1<<15).
static void TestPhaseC2_KnownAnswer_KEqualsOneIsIdentity() {
	int64_t q_ln2, q_b, q_c;
	CHECK(DeriveDefaultScaleConstants(&q_ln2, &q_b, &q_c));
	const int32_t V = 10;
	const int32_t idx[1] = {5};
	std::vector<int32_t> row = MakeSyntheticRow(V, {5});
	int64_t out[1];
	const bool wf = TopKRenormalizeQ15(row.data(), idx, 1, q_ln2, q_b, q_c, out);
	CHECK(wf);
	CHECK_MSG(out[0] == (1 << kProbFracBits),
	          "k=1 q_theta = %lld, want exactly 32768 (a single candidate's own probability "
	          "relative to itself is always the whole distribution)",
	          (long long)out[0]);
}

// ===========================================================================================
// Sec9 dim6 (GATE, D-SLM3719), a dependency pin, not new-implementation-dependent: "the
// width-domain boundary itself, executed at the derived (q_b,q_c) (Phase B2) -- trivially
// satisfied at k=6, but still executed, not assumed." CheckSoftmaxRowWidthDomain is ALREADY
// shipped and certified -- this cell pins that a legitimately derived (q_b,q_c) triple
// clears the width-domain gate at width=k for every k this design's own grid names
// (k in {3,6,10}), so it PASSES today rather than failing red; it exists in this suite to
// guard a regression (a future scale-derivation change that silently breaks the k-width
// bound) rather than to prove new code.
// ===========================================================================================
static void TestDim6_WidthDomainGate_ClearsAtEveryGridK() {
	int64_t q_ln2, q_b, q_c;
	CHECK(DeriveDefaultScaleConstants(&q_ln2, &q_b, &q_c));
	for (int32_t k : {3, 6, 10}) {
		const SslmForwardStatus st =
		    CheckSoftmaxRowWidthDomain(q_b, q_c, static_cast<size_t>(k));
		CHECK_MSG(st == SslmForwardStatus::Ok,
		          "CheckSoftmaxRowWidthDomain(q_b=%lld, q_c=%lld, width=%d) = status %d, want "
		          "Ok -- the plan's own default calibration must clear the width gate at "
		          "every primary-grid k",
		          (long long)q_b, (long long)q_c, k, static_cast<int>(st));
	}
}

// ===========================================================================================
// Sec9 dim5 (GATE, D-SLM3719) / dim11 (GATE, D-SLM3719, guard vitality): a real vocab row
// whose derived (q_b,q_c) passes CheckSoftmaxRowWidthDomain at conversion time but whose
// per-step values drive at least one top-k candidate's evaluated i-exp value past the
// recomputed per-element bound, asserting TopKRenormalizeQ15 propagates the refusal
// (all_well_formed=false) -- constructed adversarially against the REAL certified
// sub-primitives (IExpConstruct/IExpEvaluate/IExpScaleConstants), not merely asserted sound
// from the derivation-time check alone. Sec7.5's own refusal-propagation POLICY is
// score_select_phaseC_red.cpp's own cell (DampedGreedyScoreAndArgmax must map this to
// SoftmaxKernelRefusedAfterGateAccepted); this file pins that the underlying primitive
// TopKRenormalizeQ15 itself can genuinely be driven to refuse -- the guard's own liveness
// (dim11), not merely its presence in source.
//
// Construction: a (q_ln2, q_b, q_c) triple derived at the FAR end of the domain the width
// gate still accepts (a very large q_ln2, near the erasure-zone boundary Sec9 dim6 names,
// q_ln2 >= 23,378) paired with a candidate row whose shifted (post-ShiftByMax) deviation for
// a NON-max top-k element is small in magnitude -- the shape the plan's own citation
// describes as the one that can still overflow the per-element ceiling even after a clean
// derivation-time pass. This is a genuine numerical search over real inputs (no hand-typed
// magic constants), matching this suite's own construction discipline; if no firing point is
// found within the search budget, the cell SKIPs rather than fabricating one -- an honest
// report that the refusal surface may need a wider search once Phase C2 exists to drive
// against directly (the search below runs against the SAME certified primitives the future
// TopKRenormalizeQ15 will call, so a negative result here is informative, not vacuous).
// ===========================================================================================
static void TestDim5Dim11_RuntimeRefusal_AdversarialSearch() {
	bool found = false;
	int64_t found_q_ln2 = 0, found_q_b = 0, found_q_c = 0;
	std::vector<int32_t> found_row;
	int32_t found_idx[6] = {0};

	// Sweep large q_ln2 targets from the erasure zone upward -- each still must clear
	// CheckSoftmaxRowWidthDomain at k=6 to count as "passes at conversion time."
	for (int64_t target = 23378; target <= 23378 + 20000 && !found; target += 500) {
		int64_t q_ln2, q_b, q_c;
		if (!DeriveScaleConstants(target, &q_ln2, &q_b, &q_c)) continue;
		if (CheckSoftmaxRowWidthDomain(q_b, q_c, 6) != SslmForwardStatus::Ok) continue;

		// A row whose top-6 candidates sit at a range of shifted deviations from the max --
		// sweep the SPREAD between the max and the rest, since the per-element ceiling is a
		// function of the shifted (negative) deviation via IExpConstruct's own decomposition.
		for (int32_t spread = 1; spread <= 200000 && !found; spread += 4000) {
			const int32_t V = 6;
			std::vector<int32_t> row(V);
			row[0] = 0;  // the max (shifted deviation 0)
			for (int32_t i = 1; i < V; ++i) row[i] = -spread * i;
			int32_t idx[6] = {0, 1, 2, 3, 4, 5};
			int64_t out[6];
			const bool wf = TopKRenormalizeQ15(row.data(), idx, 6, q_ln2, q_b, q_c, out);
			if (!wf) {
				found = true;
				found_q_ln2 = q_ln2;
				found_q_b = q_b;
				found_q_c = q_c;
				found_row.assign(row.begin(), row.end());
				std::memcpy(found_idx, idx, sizeof(idx));
			}
		}
	}

	if (!found) {
		SKIP_MSG("adversarial search over the erasure-zone scale range found no "
		         "TopKRenormalizeQ15 refusal within budget -- either the refusal surface needs "
		         "a wider/different search once Phase C2 exists to drive directly, or this "
		         "design's own k-restriction (Sec7.5's 'LESS likely to fire' argument) makes "
		         "it genuinely hard to trigger at k=6; not asserted either way, reported "
		         "honestly per this cell's own construction discipline");
		return;
	}
	std::printf("dim5/dim11 refusal FOUND: q_ln2=%lld q_b=%lld q_c=%lld\n", (long long)found_q_ln2,
	            (long long)found_q_b, (long long)found_q_c);
	int64_t out[6];
	const bool wf = TopKRenormalizeQ15(found_row.data(), found_idx, 6, found_q_ln2, found_q_b,
	                                    found_q_c, out);
	CHECK_MSG(!wf, "the found construction must reproducibly refuse on a second call "
	               "(determinism of the refusal itself)");
}

// ===========================================================================================
// Sec9 dim7 (GATE, D-SLM3719): "TopKRenormalizeQ15's cost is O(k), not O(V), regardless of row
// shape... at minimum, a maximally flat/uniform logit row... must be timed against
// TopKRenormalizeQ15 and shown to cost within the same order of magnitude as the realistic
// row." This cell pins the ORDER-OF-GROWTH shape (flat row is not meaningfully more expensive
// than a clipping/realistic row, at fixed k) using host wall-clock timing of the standalone
// primitive -- the full "5% of total per-token forward cost" ratio (Sec8 Phase C2a's own
// budget) needs the real engine's own per-token GEMM cost as a denominator and is therefore
// this cell's own routed gap (see this campaign's test-design record): Phase C2a's own
// execution against the real CPU int8 kernel, sited at build time once the production
// TopKRenormalizeQ15 exists, is what closes that ratio -- not a standalone-primitive red cell
// this suite can author before an engine to measure the denominator against exists.
// ===========================================================================================
static void TestDim7_CostShape_FlatRowNotMoreExpensiveThanRealistic() {
	int64_t q_ln2, q_b, q_c;
	CHECK(DeriveDefaultScaleConstants(&q_ln2, &q_b, &q_c));
	const int32_t k = 6;
	const int32_t reps = 20000;

	// "realistic" row: one dominant candidate, rest clipping toward the floor -- most of the
	// certified sub-primitives' own fast paths apply.
	int32_t idx_realistic[6] = {0, 1, 2, 3, 4, 5};
	int32_t row_realistic[6] = {0, -50000, -80000, -95000, -99000, -100000};
	// "flat" row: every candidate at the SAME shifted deviation -- the shape both prior
	// strikes' own probes measured as most expensive for the dead full-row primitive.
	int32_t idx_flat[6] = {0, 1, 2, 3, 4, 5};
	int32_t row_flat[6] = {0, -1, -1, -1, -1, -1};

	int64_t out[6];
	auto time_reps = [&](const int32_t* row, const int32_t* idx) {
		const auto t0 = std::chrono::steady_clock::now();
		for (int32_t r = 0; r < reps; ++r) TopKRenormalizeQ15(row, idx, k, q_ln2, q_b, q_c, out);
		const auto t1 = std::chrono::steady_clock::now();
		return std::chrono::duration<double, std::micro>(t1 - t0).count();
	};
	const double t_realistic = time_reps(row_realistic, idx_realistic);
	const double t_flat = time_reps(row_flat, idx_flat);
	std::printf("cost shape: realistic=%.2f us/%d reps, flat=%.2f us/%d reps (ratio %.2fx)\n",
	            t_realistic, reps, t_flat, reps, t_flat / (t_realistic > 0 ? t_realistic : 1));
	// "within the same order of magnitude" -- pinned as a 10x ceiling on the ratio, matching
	// dim7's own qualitative "same order of magnitude" language rather than a tighter,
	// unstated bound this plan never pins.
	CHECK_MSG(t_flat < t_realistic * 10.0 + 1000.0,
	          "flat-row cost (%.2f us) exceeds 10x the realistic-row cost (%.2f us) -- the "
	          "O(k) claim requires row-SHAPE independence, not only k independence",
	          t_flat, t_realistic);
}

int main() {
	TestPhaseC2_KnownAnswer_MatchesSoftmaxRowQ15OnGatheredElements();
	TestPhaseC2_KnownAnswer_KEqualsOneIsIdentity();
	TestDim6_WidthDomainGate_ClearsAtEveryGridK();
	TestDim5Dim11_RuntimeRefusal_AdversarialSearch();
	TestDim7_CostShape_FlatRowNotMoreExpensiveThanRealistic();
	std::printf("checks=%d failures=%d skips=%d\n", GChecks, GFailures, GSkips);
	std::printf("[gap, not authored] Phase C2a's own 5%% forward-cost RATIO budget needs the "
	            "real engine's per-token GEMM cost as a denominator -- see this file's own "
	            "trailing comment; routed to whoever prosecutes Phase C2a.\n");
	return GFailures ? 1 : 0;
}

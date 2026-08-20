// T-2199 (Curie) -- Phase A red suite: the n-gram anti-LM (plan Sec7.2, Sec8 Phase A1/A2).
// RED BY LINK: sslm_damped_greedy.h declares AntiLmCreate/AntiLmDestroy/AntiLmUpdate/
// AntiLmPenalize/AntiLmRetainedBytes; no .cpp in this repo defines them (grep confirmed at
// authoring commit main@071c5f3). Every cell below therefore compiles clean and fails to
// link until Phase A lands.
//
// Coverage Model cells realized here (Claude/Plans/superslm-1p2-fsd-plan-2026-08-19.md Sec9,
// Wizard repo, commit 5a7f80e251):
//   dim1  Lifetime and reuse            -- GATE, D-SLM3719  (differential accumulation cell)
//   dim3  Concurrency (object-level)     -- GATE, D-SLM3719  (state isolation, no leak)
//   dim7  Contract claims                -- GATE, D-SLM3719  (determinism-vs-insertion-order,
//                                            memory O(distinct n-grams), p_omega in [0, 2^15])
//   Phase A2's own red-suite spec (Sec8): known-answer fixed-point cases, a determinism cell,
//   a memory-growth cell, a cross-reference cell against Loftus's own repaired driver.
#include "fixture_common.h"

using namespace t2199fixture;

// ===========================================================================================
// Phase A2 -- known-answer fixed-point case (order=1, unambiguous: no mixing weights involved,
// so this is the one case the plan's own formula ("num << 15) / total") pins to an EXACT Q15
// value with no rounding-rule ambiguity). MEASUREMENT-classed by dim6's own preamble rule
// (computation correctness of a numeric primitive), pinned here as a hand-computed oracle.
// ===========================================================================================
static void TestPhaseA_KnownAnswer_Order1() {
	AntiLmState* s = AntiLmCreate(/*max_order=*/1);
	// Prefix: A, B, A, A -- unigram context is the single empty context "", total count 4;
	// A appears 3 times, B appears 1 time, C never appears.
	const int32_t A = 100, B = 200, C = 300;
	AntiLmUpdate(s, A);
	AntiLmUpdate(s, B);
	AntiLmUpdate(s, A);
	AntiLmUpdate(s, A);
	const int32_t cands[3] = {A, B, C};
	int64_t out[3] = {-1, -1, -1};
	AntiLmPenalize(s, cands, 3, out);
	// Hand-computed, exact: (3 << 15) / 4 = 24576; (1 << 15) / 4 = 8192; C unseen -> 0.
	CHECK_MSG(out[0] == 24576, "p_omega(A) = %lld, want 24576 ((3<<15)/4)", (long long)out[0]);
	CHECK_MSG(out[1] == 8192, "p_omega(B) = %lld, want 8192 ((1<<15)/4)", (long long)out[1]);
	CHECK_MSG(out[2] == 0, "p_omega(C) = %lld, want 0 (never seen)", (long long)out[2]);
	AntiLmDestroy(s);
}

// A second known-answer point, order=1, a different split (1 of 5), pinning the formula's own
// exactness rather than one lucky ratio -- a mutation that rounded incorrectly (truncating
// toward zero vs. banker's rounding, say) would very likely still pass the first case
// (3<<15=98304 divides 4 evenly) but not this one (1<<15=32768 does not divide 5 evenly:
// exact floor is 6553).
static void TestPhaseA_KnownAnswer_Order1_NonDivisible() {
	AntiLmState* s = AntiLmCreate(1);
	const int32_t A = 100, D = 400;
	AntiLmUpdate(s, D);
	AntiLmUpdate(s, D);
	AntiLmUpdate(s, D);
	AntiLmUpdate(s, D);
	AntiLmUpdate(s, A);  // A: 1 of 5
	int64_t out = -1;
	AntiLmPenalize(s, &A, 1, &out);
	// (1 << 15) / 5 = 6553 (floor of 6553.6) -- pinned as an exact floor-division, per the
	// plan's own "(num << 15) / total" formula (Sec7.3 surface 1), matching this codebase's
	// own integer-division convention elsewhere (no rounding step is stated anywhere in Sec7).
	CHECK_MSG(out == 6553, "p_omega(A) = %lld, want 6553 (floor((1<<15)/5))", (long long)out);
	AntiLmDestroy(s);
}

// ===========================================================================================
// Sec9 dim7 (GATE, D-SLM3719): p_omega's own domain clamp, [0, 1<<15], adversarially
// constructed -- "a maximally lopsided count distribution: all mass on one order, one
// candidate" (plan Sec7.2/Sec9 dim7's own construction standard). Order=3, a purely
// repetitive two-token cycle drives EVERY order's conditional probability for the repeated
// candidate to exactly 1.0 -- the shape most likely to defeat an unclamped or unnormalized
// mixture, since every order's own Q15 ratio independently saturates at the ceiling before
// mixing even applies.
// ===========================================================================================
static void TestPhaseA_ClampGate_AllMassOneCandidate() {
	AntiLmState* s = AntiLmCreate(3);
	const int32_t LOOP = 1000, OTHER = 2000;
	// 30 repeats of the SAME token -- every order-1/2/3 context this produces has exactly one
	// observed continuation (LOOP), so every order's own conditional ratio is 1.0 exactly.
	for (int i = 0; i < 30; ++i) AntiLmUpdate(s, LOOP);
	const int32_t cands[2] = {LOOP, OTHER};
	int64_t out[2] = {-1, -1};
	AntiLmPenalize(s, cands, 2, out);
	CHECK_MSG(out[0] >= 0 && out[0] <= (1 << kProbFracBits),
	          "p_omega(LOOP) = %lld out of [0, 32768] under maximal lopsidedness",
	          (long long)out[0]);
	// Every order's own ratio is exactly 1.0 here, and the mixing weights are a convex
	// combination (Sec7.2's own normalized-weights design requirement) -- the mixed value
	// must therefore equal the ceiling exactly, not merely be bounded by it: a weighted
	// average of several copies of the SAME value (32768) is that value, exactly.
	CHECK_MSG(out[0] == (1 << kProbFracBits),
	          "p_omega(LOOP) = %lld, want exactly 32768 (every order's own ratio is 1.0, a "
	          "convex combination of identical values must equal that value)",
	          (long long)out[0]);
	CHECK_MSG(out[1] == 0, "p_omega(OTHER) = %lld, want 0 (never observed at any order)",
	          (long long)out[1]);
	AntiLmDestroy(s);
}

// A second construction at the OTHER extreme -- alternating two tokens forever, so every
// order-1 context conditions on exactly one continuation (deterministic alternation) but the
// order-1 UNIGRAM context ("") itself splits close to 50/50 -- exercises the clamp on a
// mixed-magnitude combination, not only the saturated case above.
static void TestPhaseA_ClampGate_AlternatingPattern() {
	AntiLmState* s = AntiLmCreate(2);
	const int32_t LOOP = 1000, ALT = 2000;
	for (int i = 0; i < 25; ++i) {
		AntiLmUpdate(s, LOOP);
		AntiLmUpdate(s, ALT);
	}
	const int32_t cands[2] = {LOOP, ALT};
	int64_t out[2] = {-1, -1};
	AntiLmPenalize(s, cands, 2, out);
	for (int i = 0; i < 2; ++i)
		CHECK_MSG(out[i] >= 0 && out[i] <= (1 << kProbFracBits),
		          "p_omega(cands[%d]) = %lld out of [0, 32768]", i, (long long)out[i]);
	AntiLmDestroy(s);
}

// ===========================================================================================
// Sec9 dim1 (GATE, D-SLM3719): the differential accumulation cell -- "a specific token is
// emitted once, early (e.g. step 3), never repeated, and generation is driven to a much later
// step (e.g. step 50); the cell asserts penalize's score for that token at step 50 differs
// measurably from what a per-step-reset anti-LM... would compute." A fresh instance queried
// immediately after a single AntiLmUpdate for only the MOST RECENT token is exactly what a
// per-step-reset anti-LM's own state looks like at query time (it never retains anything
// beyond the update immediately preceding the current step's own query, since its own history
// is discarded between steps) -- the accumulated instance retains the whole prefix.
// ===========================================================================================
static void TestPhaseA_Dim1_DifferentialAccumulation() {
	AntiLmState* accumulated = AntiLmCreate(2);
	const int32_t EARLY = 777;  // emitted exactly once, at step index 3
	// 50 steps: distinct filler tokens everywhere except step 3, so no context accidentally
	// re-creates EARLY's own n-gram elsewhere in the prefix.
	for (int step = 0; step < 50; ++step) {
		const int32_t tok = (step == 3) ? EARLY : (10000 + step);
		AntiLmUpdate(accumulated, tok);
	}
	int64_t p_accumulated = -1;
	AntiLmPenalize(accumulated, &EARLY, 1, &p_accumulated);
	CHECK_MSG(p_accumulated > 0,
	          "accumulated p_omega(EARLY) at step 50 = %lld, want > 0 (EARLY was observed once "
	          "in the 50-token history a warm anti-LM retains)",
	          (long long)p_accumulated);

	// The per-step-reset baseline: a fresh instance that has seen only the single update
	// immediately preceding this query (step 49's own token, NOT EARLY) -- exactly what an
	// anti-LM whose history is discarded between steps would carry into this query.
	AntiLmState* reset_each_step = AntiLmCreate(2);
	AntiLmUpdate(reset_each_step, 10000 + 49);
	int64_t p_reset = -1;
	AntiLmPenalize(reset_each_step, &EARLY, 1, &p_reset);
	CHECK_MSG(p_reset == 0,
	          "reset-each-step p_omega(EARLY) = %lld, want exactly 0 (a per-step-reset anti-LM "
	          "never retains step 3's own emission by step 50)",
	          (long long)p_reset);

	CHECK_MSG(p_accumulated != p_reset,
	          "accumulated (%lld) and per-step-reset (%lld) readings for the same candidate at "
	          "the same nominal step must differ -- this is the whole claim dim1 exists to pin",
	          (long long)p_accumulated, (long long)p_reset);
	AntiLmDestroy(accumulated);
	AntiLmDestroy(reset_each_step);
}

// A sequence pool reused across generations (mirroring the KV-block pool's own reuse
// pattern), with the anti-LM's own memory released/reset between sequences, asserted not to
// leak into the next sequence's counts (Sec9 dim1's second cell). AntiLmDestroy followed by a
// fresh AntiLmCreate is this suite's own "released/reset between sequences" construction --
// there is no separate reset verb in Sec7.2's own two-operation interface (update/penalize
// only), so release-and-recreate is the pooled-reuse shape this standalone primitive actually
// exposes; Phase D's own ABI-level sequence-pool reuse (mirroring the KV-block pool literally)
// is out of this suite's scope.
static void TestPhaseA_Dim1_NoLeakAcrossReleaseRecreate() {
	AntiLmState* first = AntiLmCreate(2);
	const int32_t SEEN_BY_FIRST = 555;
	for (int i = 0; i < 10; ++i) AntiLmUpdate(first, SEEN_BY_FIRST);
	AntiLmDestroy(first);

	AntiLmState* second = AntiLmCreate(2);  // a fresh sequence's own anti-LM, pool-reused slot
	int64_t p = -1;
	AntiLmPenalize(second, &SEEN_BY_FIRST, 1, &p);
	CHECK_MSG(p == 0,
	          "second sequence's own p_omega(SEEN_BY_FIRST) = %lld, want exactly 0 -- the "
	          "prior sequence's counts must not leak into a freshly created instance",
	          (long long)p);
	AntiLmDestroy(second);
}

// ===========================================================================================
// Sec9 dim3 (GATE, D-SLM3719), object-level slice: per-sequence state isolation and a
// teardown-during-flight proxy. The batched-ABI concurrency half (two sequences decoding in
// ONE batched sslm_decode_step call, and a real concurrent teardown against an in-flight
// mutation) is Phase D's own cell (the anti-LM is wired into a multi-sequence batched call
// only once Phase D lands) -- out of this suite's scope; what Phase A's own standalone
// primitive can pin today is that two independently-owned instances never share state, and
// that destroying one does not disturb a sibling still in use.
// ===========================================================================================
static void TestPhaseA_Dim3_TwoInstancesNoCrossContamination() {
	AntiLmState* seq_a = AntiLmCreate(2);
	AntiLmState* seq_b = AntiLmCreate(2);
	const int32_t ONLY_IN_A = 111, ONLY_IN_B = 222;
	for (int i = 0; i < 5; ++i) AntiLmUpdate(seq_a, ONLY_IN_A);
	for (int i = 0; i < 5; ++i) AntiLmUpdate(seq_b, ONLY_IN_B);

	int64_t p_a_sees_b = -1, p_b_sees_a = -1;
	AntiLmPenalize(seq_a, &ONLY_IN_B, 1, &p_a_sees_b);
	AntiLmPenalize(seq_b, &ONLY_IN_A, 1, &p_b_sees_a);
	CHECK_MSG(p_a_sees_b == 0, "seq_a's own p_omega(ONLY_IN_B) = %lld, want 0",
	          (long long)p_a_sees_b);
	CHECK_MSG(p_b_sees_a == 0, "seq_b's own p_omega(ONLY_IN_A) = %lld, want 0",
	          (long long)p_b_sees_a);

	int64_t p_a_self_before = -1;
	AntiLmPenalize(seq_a, &ONLY_IN_A, 1, &p_a_self_before);
	AntiLmDestroy(seq_b);  // teardown-during-flight proxy: seq_a stays live and in use
	int64_t p_a_self_after = -1;
	AntiLmPenalize(seq_a, &ONLY_IN_A, 1, &p_a_self_after);
	CHECK_MSG(p_a_self_before == p_a_self_after,
	          "seq_a's own reading changed (%lld -> %lld) after an UNRELATED sibling's "
	          "teardown -- state must be fully independent",
	          (long long)p_a_self_before, (long long)p_a_self_after);
	AntiLmDestroy(seq_a);
}

// ===========================================================================================
// Sec9 dim7 / Phase A2 (Sec8): "the anti-LM's determinism holds regardless of its internal
// table's iteration order." CORRECTED 2026-08-20 (builder's routed finding,
// `Claude/Brunel/t2199-phaseAC-build-2026-08-20.md` Sec6, ruled and confirmed here): the
// original construction permuted the token GENERATION sequence across two order-2 fixtures and
// asserted "same multiset -> same output," conflating hash-table iteration-order independence
// (Sec7.2's real claim -- the internal table is never iterated in its own container order)
// with token-SEQUENCE-order invariance, which is mathematically FALSE for any order>=2 n-gram
// model: context is defined by immediately-preceding tokens, not by set membership, so
// permuting the generation order changes which (context, candidate) pairs are observed at all.
// Hand-verified against the builder's own executed failure: `seq1={A,B,A,C,B,A,D,B}` and
// `seq2={D,B,A,B,C,A,B,A}` share the identical unigram multiset (A:3,B:3,C:1,D:1) but genuinely
// different bigram distributions (context `A` holds `{B:1,C:1,D:1}` in seq1 vs `{B:2}` in
// seq2) -- worse, the two instances are queried under DIFFERENT bigram contexts entirely
// (seq1's own history ends in `B`, seq2's in `A`), so the swap the builder measured
// (`p_omega(A)`: 23067 vs 5820, and the reverse for `p_omega(B)`) is the CORRECT behavior of a
// faithful order>=2 mixture, not a defect. No implementation of a genuine order->=2 model
// could pass the original cell as constructed. Ruling: the builder is right; the test was
// wrong. The claim this cell can actually pin as a black-box construction, without literally
// swapping the future implementation's container type (that mutation belongs to source-level
// review, not a black-box suite -- routed as a build-review item, not invented here):
//   (a) order=1 (unigram): context is ALWAYS the empty string, so ANY permutation of the same
//       token multiset is genuinely, provably order-invariant -- the one case where "same
//       multiset -> same output" is actually true. Tested directly.
//   (b) any order: `AntiLmPenalize`'s own OUTPUT must track candidate IDENTITY regardless of
//       the ORDER candidates are listed in the QUERY array -- a distinct, order-agnostic
//       determinism claim (a real defect class: an implementation that iterates a map and
//       writes results out of query order would fail this even though every single-candidate
//       query would individually look correct). Tested at order=3, reversing the query array.
// ===========================================================================================
static void TestPhaseA_Dim7_DeterminismAcrossInsertionOrder() {
	const int32_t A = 100, B = 200, C = 300, D = 400;

	// --- (a) order=1: ANY permutation of the SAME multiset must give IDENTICAL output. ---
	AntiLmState* order1a = AntiLmCreate(1);
	for (int32_t tok : {A, B, A, C, B, A, D, B}) AntiLmUpdate(order1a, tok);
	AntiLmState* order1b = AntiLmCreate(1);
	for (int32_t tok : {D, B, A, B, C, A, B, A}) AntiLmUpdate(order1b, tok);  // same multiset,
	                                                                          // different order
	const int32_t cands[4] = {A, B, C, D};
	int64_t out_a[4], out_b[4];
	AntiLmPenalize(order1a, cands, 4, out_a);
	AntiLmPenalize(order1b, cands, 4, out_b);
	for (int i = 0; i < 4; ++i)
		CHECK_MSG(out_a[i] == out_b[i],
		          "order=1, candidate %d: instance-a=%lld instance-b=%lld -- unigram counts "
		          "must be permutation-invariant (context is always empty at order 1)",
		          i, (long long)out_a[i], (long long)out_b[i]);
	AntiLmDestroy(order1a);
	AntiLmDestroy(order1b);

	// --- (b) order=3: the SAME generation sequence, queried with candidates listed in two
	// DIFFERENT array orders, must return values that track candidate IDENTITY, not array
	// position. ---
	AntiLmState* s = AntiLmCreate(3);
	for (int32_t tok : {A, B, A, C, B, A, D, B, A, C}) AntiLmUpdate(s, tok);
	const int32_t cands_fwd[4] = {A, B, C, D};
	const int32_t cands_rev[4] = {D, C, B, A};
	int64_t out_fwd[4], out_rev[4];
	AntiLmPenalize(s, cands_fwd, 4, out_fwd);
	AntiLmPenalize(s, cands_rev, 4, out_rev);
	for (int i = 0; i < 4; ++i)
		CHECK_MSG(out_rev[i] == out_fwd[3 - i],
		          "query-array-order independence: reversed-array position %d (token %d) = "
		          "%lld, canonical-array reading for the same candidate = %lld",
		          i, cands_rev[i], (long long)out_rev[i], (long long)out_fwd[3 - i]);
	AntiLmDestroy(s);
}

// Repeating the identical update sequence into a second, independent instance must reproduce
// bit-identical penalize() output -- the base "same update sequence twice -> identical state"
// cell Phase A2's own red-suite spec names explicitly (Sec8 Phase A2).
static void TestPhaseA_SameSequenceTwiceIdenticalState() {
	const std::vector<int32_t> seq = {10, 20, 10, 10, 30, 20, 10};
	AntiLmState* s1 = AntiLmCreate(3);
	AntiLmState* s2 = AntiLmCreate(3);
	for (int32_t t : seq) AntiLmUpdate(s1, t);
	for (int32_t t : seq) AntiLmUpdate(s2, t);
	const int32_t cands[4] = {10, 20, 30, 40};
	int64_t o1[4], o2[4];
	AntiLmPenalize(s1, cands, 4, o1);
	AntiLmPenalize(s2, cands, 4, o2);
	for (int i = 0; i < 4; ++i)
		CHECK_MSG(o1[i] == o2[i], "candidate %d: s1=%lld s2=%lld, want identical", i,
		          (long long)o1[i], (long long)o2[i]);
	AntiLmDestroy(s1);
	AntiLmDestroy(s2);
}

// ===========================================================================================
// Phase A2 (Sec8): the memory-growth cell -- "bounded by generation length" / Sec7.2's own
// "Memory is O(distinct n-grams seen so far) <= O(generation length)" claim (Sec9 dim7's
// mapped cell). A discriminating construction: replaying the IDENTICAL 40-token sequence a
// SECOND time (no new distinct n-gram is added; every context/candidate pair the second pass
// visits was already present after the first) must leave retained memory UNCHANGED --
// growth is a function of distinct n-grams, never of update CALL COUNT.
// ===========================================================================================
static void TestPhaseA_MemoryGrowth_BoundedByDistinctNgrams() {
	const std::vector<int32_t> loop_seq = {1000, 2000, 1000, 2000, 1000, 2000};
	AntiLmState* s = AntiLmCreate(2);
	for (int32_t t : loop_seq) AntiLmUpdate(s, t);
	const size_t bytes_after_one_pass = AntiLmRetainedBytes(s);
	for (int rep = 0; rep < 20; ++rep)
		for (int32_t t : loop_seq) AntiLmUpdate(s, t);  // 20 more passes, zero NEW n-grams
	const size_t bytes_after_many_passes = AntiLmRetainedBytes(s);
	CHECK_MSG(bytes_after_many_passes == bytes_after_one_pass,
	          "retained bytes grew from %zu to %zu across 20 further passes of an ALREADY-SEEN "
	          "cycle -- growth must track distinct n-grams, not update call count",
	          bytes_after_one_pass, bytes_after_many_passes);

	// A genuinely GROWING sequence (every token distinct, never repeated) must retain MORE
	// than the loop -- distinguishes "memory grows with distinct n-grams" from a vacuous
	// implementation that never grows at all (a cell that only checked the loop case above
	// would pass a stub that always returns 0).
	AntiLmState* growing = AntiLmCreate(2);
	for (int32_t t = 5000; t < 5000 + 40; ++t) AntiLmUpdate(growing, t);
	const size_t bytes_growing = AntiLmRetainedBytes(growing);
	CHECK_MSG(bytes_growing > bytes_after_one_pass,
	          "40 distinct-token updates retained %zu bytes, want strictly more than the "
	          "6-distinct-token loop's own %zu -- memory must actually grow with distinct "
	          "n-grams (a cell the loop-replay assertion alone cannot catch)",
	          bytes_growing, bytes_after_one_pass);
	AntiLmDestroy(s);
	AntiLmDestroy(growing);
}

// ===========================================================================================
// Phase A2 (Sec8), Sec9 dim7 -- the Loftus cross-reference cell. FILLED 2026-08-20 (Mendeleev's
// pre-build audit, Finding 1, `Claude/Mendeleev/t2199-red-suite-coverage-audit-2026-08-20.md`):
// the plan folded to `925ea281e3` and now carries a concrete, executed, reviewed fixture
// (Sec8 Phase A2) that did not exist when this suite was first authored against `5a7f80e251` --
// the routing note this comment used to carry is stale and is replaced by the cell itself.
//
// Reference: `NGram(input_ids=[5,12,7,5,12,9], n=3, vocab_size=50, beta=0.9, sw_coeff=0.0)`
// (plan Sec8 Phase A2, `925ea281e3`), three steps:
//   Step 1  penalize({5,12,7,5,12,9}, [5,12,7,9,3]) -- no update() yet:
//           5: 0.3333333432674408   12: 0.3333333432674408   7: 0.1666666716337204
//           9: 0.1666666716337204   3: 0.0
//   Step 2  update(7), penalize({5,12,7,5,12,9,7}, [5,12,7,9,3]):
//           5: 0.47857141494750977 12: 0.2857142984867096   7: 0.2857142984867096
//           9: 0.1428571492433548  3: 0.0
//   Step 3  determinism: a fresh instance, same construction + the same single update(7), same
//           penalize call -- must match step 2 exactly.
//
// BEHAVIORAL PARITY, NOT BIT PARITY, per the plan's own explicit framing (Sec8 Phase A2) --
// stated here as an engineering finding, not merely a disclaimer: this engine's own design
// (Sec7.2) states mixing weights are normalized "for each order i" but does not state whether
// the normalization denominator (`sum w_i`) runs over ALL orders 1..max_order unconditionally
// or is RE-normalized over only the orders with an observed context at the current query --
// the two conventions give numerically different results whenever some order is unobserved
// (exactly the shape both fixture steps exercise). Working the reference's own step-1 numbers
// backward (pure unigram, no scaling by any fractional order-1 weight) shows the REFERENCE
// algorithm renormalizes over active orders only; working step 2 backward (candidates 12/7/9/3
// landing at their EXACT raw unigram value while candidate 5 alone gains a boost from the
// bigram term) shows a mixing shape this suite's own header does not fully pin either. Rather
// than guess which convention this engine's own build will choose (inventing a precision
// the plan never specified would violate Curie's own "pin the documented claim, not invent"
// discipline), this cell checks the properties EVERY faithful implementation of "a smoothed
// n-gram mixture that gives more weight to higher, more specific orders when they have data"
// must satisfy, regardless of the exact normalization convention -- the same behavioral-parity
// standard the plan's own text names for this cell.
// ===========================================================================================
static void TestPhaseA_LoftusCrossReference_BehavioralParity() {
	const int32_t C5 = 5, C12 = 12, C7 = 7, C9 = 9, C3 = 3;
	const int32_t cands[5] = {C5, C12, C7, C9, C3};

	AntiLmState* alm = AntiLmCreate(/*max_order=*/3);  // matches the fixture's own n=3
	for (int32_t t : {5, 12, 7, 5, 12, 9}) AntiLmUpdate(alm, t);  // NGram's own input_ids

	// --- Step 1: no higher-order context has EVER been observed (neither (12,9) for the
	// trigram nor (9,) for the bigram was ever inserted as a context key by the six updates
	// above -- confirmed by hand-trace against this suite's own update() semantics, matching
	// Mendeleev's own independent re-derivation, task 2 of the audit). Every faithful mixture
	// therefore reduces to pure unigram counts (5:2, 12:2, 7:1, 9:1, 3:0 over 6 tokens) --
	// these tie/order relationships hold under ANY consistent weighting scheme, since a single
	// active order scales every candidate's own ratio by the identical factor. ---
	int64_t out1[5];
	AntiLmPenalize(alm, cands, 5, out1);
	CHECK_MSG(out1[0] == out1[1],
	          "step1: p_omega(5)=%lld != p_omega(12)=%lld -- both have unigram count 2 and no "
	          "order has ever seen higher-order context, so they must tie",
	          (long long)out1[0], (long long)out1[1]);
	CHECK_MSG(out1[2] == out1[3],
	          "step1: p_omega(7)=%lld != p_omega(9)=%lld -- both have unigram count 1, no "
	          "higher-order context observed, must tie",
	          (long long)out1[2], (long long)out1[3]);
	CHECK_MSG(out1[4] == 0, "step1: p_omega(3)=%lld, want exactly 0 -- candidate 3 is never "
	                         "observed at any order",
	          (long long)out1[4]);
	CHECK_MSG(out1[0] > out1[2],
	          "step1: the {5,12} group (count 2) must outrank the {7,9} group (count 1) -- "
	          "got p_omega(5)=%lld, p_omega(7)=%lld",
	          (long long)out1[0], (long long)out1[2]);
	CHECK_MSG(out1[2] > out1[4],
	          "step1: the {7,9} group (count 1) must outrank candidate 3 (count 0) -- got "
	          "p_omega(7)=%lld, p_omega(3)=%lld",
	          (long long)out1[2], (long long)out1[4]);

	// --- Step 2: update(7) makes bigram context (7,) active with exactly one completion, to
	// candidate 5 (the only (context, candidate) pair with any bigram evidence -- confirmed by
	// hand-trace: context (7,) was inserted once, from the original prefix's own token index 2
	// ("7" followed by "5"), and the newly appended token 7 only inserts a NEW context (9,)->7,
	// which does not touch this query). Candidates 12/7/9/3 gain NO bigram evidence (their own
	// bigram ratio is exactly 0 under context (7,) regardless of normalization), so they must
	// still tie/rank purely by their unigram counts (12:2, 7:2, 9:1, 3:0 over 7 tokens) --
	// EXACTLY the step-1 shape, one level shifted. Candidate 5 alone must rank STRICTLY above
	// its own unigram-only peers, because it is the only candidate any active higher order
	// favors -- this is the discriminating assertion: an implementation that dropped order>1
	// mixing entirely (collapsing to pure unigram always) would make candidate 5 tie with
	// {12,7} instead of beating them, failing this check for exactly the reason this cell
	// exists to catch. ---
	AntiLmUpdate(alm, 7);
	int64_t out2[5];
	AntiLmPenalize(alm, cands, 5, out2);
	CHECK_MSG(out2[1] == out2[2],
	          "step2: p_omega(12)=%lld != p_omega(7)=%lld -- both have unigram count 2 and "
	          "zero bigram evidence (context (7,) never completed to either), must tie",
	          (long long)out2[1], (long long)out2[2]);
	CHECK_MSG(out2[4] == 0, "step2: p_omega(3)=%lld, want exactly 0", (long long)out2[4]);
	CHECK_MSG(out2[0] > out2[1],
	          "step2: p_omega(5)=%lld must STRICTLY exceed p_omega(12)=%lld -- candidate 5 is "
	          "the sole beneficiary of the newly active bigram context (7,)->5; a build that "
	          "ignores order>1 evidence entirely would tie them instead",
	          (long long)out2[0], (long long)out2[1]);
	CHECK_MSG(out2[1] > out2[3],
	          "step2: the {12,7} group (unigram count 2) must outrank candidate 9 (count 1) -- "
	          "got p_omega(12)=%lld, p_omega(9)=%lld",
	          (long long)out2[1], (long long)out2[3]);
	CHECK_MSG(out2[0] > out2[3], "step2: p_omega(5)=%lld must exceed p_omega(9)=%lld",
	          (long long)out2[0], (long long)out2[3]);

	// --- Step 3: the determinism cross-check the fixture itself specifies -- a FRESH instance,
	// built with the identical construction (the same six-token prefix) plus the identical
	// single update(7), must match step 2's own reading EXACTLY. Unlike steps 1/2 above (which
	// compare against an EXTERNAL reference under an admittedly different, unpinned
	// normalization convention), this is a same-engine, same-construction comparison -- bit-
	// exact equality is the correct standard here, matching this suite's own
	// TestPhaseA_SameSequenceTwiceIdenticalState cell, now anchored to the SAME real fixture
	// data Mendeleev's audit reviewed rather than an arbitrary sequence. ---
	AntiLmState* fresh = AntiLmCreate(3);
	for (int32_t t : {5, 12, 7, 5, 12, 9}) AntiLmUpdate(fresh, t);
	AntiLmUpdate(fresh, 7);
	int64_t out3[5];
	AntiLmPenalize(fresh, cands, 5, out3);
	for (int i = 0; i < 5; ++i)
		CHECK_MSG(out2[i] == out3[i],
		          "step3 determinism: candidate index %d: accumulated=%lld fresh-replay=%lld -- "
		          "must match exactly",
		          i, (long long)out2[i], (long long)out3[i]);

	AntiLmDestroy(alm);
	AntiLmDestroy(fresh);
}

// ===========================================================================================
// Fix round 2026-08-20 (Poirot M4, `Claude/Poirot/7be9508-t2199-phaseAC-review.md`,
// `Claude/Brunel/t2199-phaseAC-build-2026-08-20.md` Sec12.1): `AntiLmCreate(max_order < 1)`
// now returns `nullptr` instead of the two pre-fix failure modes -- `AntiLmCreate(-1)`
// terminated the process (`0xC0000409`, a negative `max_order` casting to a huge `size_t` in
// `tables_`'s own vector constructor) and `AntiLmCreate(0)` was silently accepted, producing a
// state that always returns `p_omega = 0` (a permanently-disabled anti-LM, not a rejection).
//
// RED-FIRST: PARTIALLY EXECUTED, split by sub-case. `max_order = 0` is a genuine CHECK-based
// red/green pair -- pre-fix it silently returned a non-null, always-zero state (this cell's own
// `CHECK(state == nullptr)` would FAIL, printed, not crash); post-fix it returns nullptr
// (passes). Executed against both `c473dad` (pre-fix) and `826f607` (post-fix); see this
// campaign's own test-design record for the transcript. `max_order = -1` is NOT executed
// red against pre-fix code in THIS binary -- pre-fix, that call terminates the process before
// any CHECK can run or report, which would silently discard every OTHER cell's own result in
// this same file's `main()`. The crash itself (0xC0000409, confirmed in Poirot's own review,
// re-confirmed by a separate isolated scratch probe, not committed) IS the red evidence for
// this sub-case -- loud and unambiguous, just not a printed FAIL line -- so this cell is
// PINNED-GREEN for the negative-argument sub-case specifically, with the reason stated here
// rather than risking every other cell in this file behind an uncaught crash.
// ===========================================================================================
static void TestM4_AntiLmCreate_SubOneMaxOrderReturnsNullptr() {
	AntiLmState* zero = AntiLmCreate(0);
	CHECK_MSG(zero == nullptr, "AntiLmCreate(0) = %p, want nullptr (Poirot M4: max_order=0 was "
	                            "previously silently accepted as a permanently-disabled "
	                            "anti-LM, p_omega always 0, rather than a rejection)",
	          (void*)zero);
	AntiLmDestroy(zero);  // AntiLmDestroy(nullptr) must be safe (matches `delete nullptr`)

	// Positive control: max_order=1 (the domain floor) must still succeed -- a cell that only
	// ever saw nullptr would trivially "pass" a stub that always returns nullptr for everything.
	AntiLmState* one = AntiLmCreate(1);
	CHECK_MSG(one != nullptr, "AntiLmCreate(1) returned nullptr -- the domain floor itself must "
	                          "still succeed");
	AntiLmDestroy(one);

	// max_order = -1: NOT red-first executed via CHECK in this binary (pre-fix, this call
	// terminates the process -- see this cell's own header comment). Post-fix assertion only.
	AntiLmState* neg = AntiLmCreate(-1);
	CHECK_MSG(neg == nullptr, "AntiLmCreate(-1) = %p, want nullptr (Poirot M4: previously "
	                          "terminated the process, 0xC0000409, via a negative-to-size_t "
	                          "cast in the tables_ vector constructor)",
	          (void*)neg);
	AntiLmDestroy(neg);
}

// ===========================================================================================
// Fix round 2026-08-20 (Poirot S3, same casebook/build-log): `AntiLmRetainedBytes` recalibrated
// against measured process-memory deltas. The DOCUMENTED SCOPE, unchanged by the recalibration
// and pinned here rather than the recalibrated numeric constants themselves (which the
// production header explicitly calls "a MODEL... not a byte-exact accounting," already
// recalibrated once and disclaimed as subject to change again): the reported figure is a
// function of the TABLE ONLY (the per-order count structures) and DELIBERATELY EXCLUDES
// `history_` (the per-sequence generated-token buffer, which grows by one token on every
// `AntiLmUpdate` call regardless of repeats) -- stated explicitly in both the production
// header and `damped_greedy_antilm.cpp`'s own comment on `AntiLmRetainedBytes`. **This is the
// stated exclusion this cell's own construction is built to make break loudly**: if a future
// change folds `history_`'s own footprint into this same figure, the additivity and
// exclusion assertions below fail immediately, not silently.
//
// RED-FIRST: NOT executed against pre-fix code. S3 was not a domain-rejection defect (the
// pre-fix code always returned SOME reading; the finding was that the reading undercounted the
// true process-memory delta by 3-6x) -- this cell pins the DOCUMENTED SCOPE (table-only,
// additive per distinct n-gram, excludes history_), a property that held both before and after
// S3's own recalibration (only the numeric CONSTANTS changed, not the accounting SHAPE), so
// there is no pre-fix/post-fix behavioral difference for this cell to detect -- PINNED-GREEN,
// reason stated: this is a scope/shape pin, not a magnitude pin, and the magnitude is exactly
// what S3's own recalibration changed without changing the shape this cell checks.
// ===========================================================================================
static void TestS3_AntiLmRetainedBytes_TableOnlyDocumentedScope() {
	// (a) A freshly created state (zero updates) reports exactly zero -- the trivial base case
	// of "the table is empty."
	AntiLmState* s = AntiLmCreate(2);
	CHECK_MSG(AntiLmRetainedBytes(s) == 0, "fresh AntiLmState reports %zu bytes, want 0",
	          AntiLmRetainedBytes(s));

	// (b) Deterministic, reproducible per-entry cost: three structurally IDENTICAL insertions
	// (a brand-new order-1 candidate under the always-existing empty context, plus a brand-new
	// order-2 context-with-one-candidate under a token never seen as a context before) must
	// cost the IDENTICAL number of bytes each time, regardless of how much history precedes
	// them -- proving the accounting is a genuine per-entry sum (the documented "grows with
	// distinct n-grams" claim), not, e.g., a formula that happens to undercount later entries
	// or double-counts the context payload.
	const int32_t X1 = 1000, X2 = 2000, X3 = 3000;
	AntiLmUpdate(s, X1);
	const std::size_t after_1 = AntiLmRetainedBytes(s);
	AntiLmUpdate(s, X2);  // order1: new candidate X2 under ""; order2: new context (X1,) + X2
	const std::size_t after_2 = AntiLmRetainedBytes(s);
	AntiLmUpdate(s, X3);  // order1: new candidate X3 under ""; order2: new context (X2,) + X3
	const std::size_t after_3 = AntiLmRetainedBytes(s);

	CHECK_MSG(after_1 > 0, "after one update, retained bytes = %zu, want > 0 (a real table "
	                        "entry must cost something, not a vacuous stub)",
	          after_1);
	const std::size_t delta_2 = after_2 - after_1;
	const std::size_t delta_3 = after_3 - after_2;
	CHECK_MSG(delta_2 > 0 && delta_3 > 0, "delta_2=%zu delta_3=%zu, both must be strictly "
	                                      "positive (a genuinely new context+candidate at "
	                                      "each step)",
	          delta_2, delta_3);
	CHECK_MSG(delta_2 == delta_3,
	          "delta_2=%zu delta_3=%zu -- two structurally IDENTICAL insertions (one new "
	          "order-1 candidate, one new order-2 context-with-candidate) must cost the "
	          "IDENTICAL number of bytes; a mismatch means the accounting is not a "
	          "deterministic per-entry sum",
	          delta_2, delta_3);

	// (c) The stated exclusion, made concrete: replaying an ALREADY-SEEN 2-token cycle many
	// times (every context/candidate pair this replay visits was already counted above) must
	// leave retained bytes EXACTLY UNCHANGED, even though `history_` (excluded from this
	// figure by design) grows by one token per replay call -- the cell's own reason this
	// assertion exists: a future change that folds `history_` into this same figure would grow
	// `after_replay` past the priming baseline here, failing loudly.
	//
	// PRIMING, not a 2-token baseline: at order 2, a context is the SINGLE preceding token, so
	// a bare two-update baseline ({L1, L2}) only ever inserts context (L1,)->L2 -- the reverse
	// direction, context (L2,)->L1, is only reached on the THIRD update, which a short baseline
	// would misclassify as "new" growth during replay (an error this cell's own first draft
	// made, caught by executing it: 312 -> 440 bytes on the first replay step, exactly one new
	// order-2 context+candidate, `kContextBaseOverhead + kContextPerTokenOverhead +
	// kCandidateOverhead`). Three full cycles primes BOTH directions before the baseline is
	// read, matching this suite's own existing `TestPhaseA_MemoryGrowth_BoundedByDistinctNgrams`
	// cell's own construction exactly, for the identical reason.
	AntiLmState* loop = AntiLmCreate(2);
	const int32_t L1 = 5000, L2 = 6000;
	for (int i = 0; i < 3; ++i) {
		AntiLmUpdate(loop, L1);
		AntiLmUpdate(loop, L2);
	}
	const std::size_t loop_baseline = AntiLmRetainedBytes(loop);
	for (int rep = 0; rep < 50; ++rep) {
		AntiLmUpdate(loop, L1);
		AntiLmUpdate(loop, L2);
	}
	const std::size_t loop_after_replay = AntiLmRetainedBytes(loop);
	CHECK_MSG(loop_after_replay == loop_baseline,
	          "retained bytes grew from %zu to %zu across 50 further replays of an already-"
	          "primed 2-token cycle -- history_ (100 additional tokens by now) must NOT be "
	          "reflected in this table-only figure",
	          loop_baseline, loop_after_replay);

	AntiLmDestroy(s);
	AntiLmDestroy(loop);
}

int main() {
	TestPhaseA_KnownAnswer_Order1();
	TestPhaseA_KnownAnswer_Order1_NonDivisible();
	TestPhaseA_ClampGate_AllMassOneCandidate();
	TestPhaseA_ClampGate_AlternatingPattern();
	TestPhaseA_Dim1_DifferentialAccumulation();
	TestPhaseA_Dim1_NoLeakAcrossReleaseRecreate();
	TestPhaseA_Dim3_TwoInstancesNoCrossContamination();
	TestPhaseA_Dim7_DeterminismAcrossInsertionOrder();
	TestPhaseA_SameSequenceTwiceIdenticalState();
	TestPhaseA_MemoryGrowth_BoundedByDistinctNgrams();
	TestPhaseA_LoftusCrossReference_BehavioralParity();
	TestS3_AntiLmRetainedBytes_TableOnlyDocumentedScope();
	// M4's own crash-prone sub-case (AntiLmCreate(-1)) is called LAST, deliberately, so that
	// if it is ever run against a build where that regression has been reintroduced, every
	// earlier cell in this file has already reported its own result first (see that cell's
	// own header comment for why this suite does not otherwise re-execute a known crash).
	TestM4_AntiLmCreate_SubOneMaxOrderReturnsNullptr();
	std::printf("checks=%d failures=%d skips=%d\n", GChecks, GFailures, GSkips);
	return GFailures ? 1 : 0;
}

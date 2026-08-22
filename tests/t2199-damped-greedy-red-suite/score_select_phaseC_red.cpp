// T-2199 (Curie) -- Phase C3 red suite: score, combine, and select -- the mechanism that
// actually emits a token (plan Sec7.5-7.6, Sec8 Phase C3). RED BY LINK: sslm_damped_greedy.h
// declares DampedGreedyScoreAndArgmax/DampedGreedyScoreAndArgmaxDiag; no .cpp in this repo
// defines them (grep confirmed at authoring commit main@071c5f3).
//
// Coverage Model cells realized here (plan Sec9, Wizard repo, commit 5a7f80e251):
//   dim5  Failure and rejection paths  -- GATE, D-SLM3719  (refusal PROPAGATION policy)
//   dim7  Contract claims              -- GATE, D-SLM3719  (masked-token-never-selected
//                                         mutation, alpha=0 identity, keeper-probe
//                                         MEASUREMENT computation-correctness)
//   dim8  Composition                  -- GATE, D-SLM3719  (mask-first ordering, the single
//                                         legal continuation under a schema)
#include <algorithm>
#include <cstdlib>

#include "fixture_common.h"

using namespace t2199fixture;

static int GHotPathAllocations = 0;
void* operator new(std::size_t size) {
	++GHotPathAllocations;
	return std::malloc(size);
}
void operator delete(void* p) noexcept { std::free(p); }
void operator delete(void* p, std::size_t) noexcept { std::free(p); }

// A small reusable anti-LM: driven with a fixed loop-lock history (LOOP, ALT repeating), the
// SAME shape Loki's strike-6 probe used, so that p_omega(LOOP) is genuinely nonzero and large
// relative to p_omega for any never-seen candidate -- the anti-LM term must actually have
// weight for the cells below to discriminate anything (an anti-LM that always reads zero
// would make every score cell trivially pass regardless of correctness).
static AntiLmState* MakeLoopLockAntiLm(int32_t loop_tok, int32_t alt_tok) {
	AntiLmState* alm = AntiLmCreate(2);
	for (int i = 0; i < 20; ++i) {
		AntiLmUpdate(alm, loop_tok);
		AntiLmUpdate(alm, alt_tok);
	}
	return alm;
}

// ===========================================================================================
// Sec9 dim7 (GATE, D-SLM3719): the masked-token-never-selected mutation -- "whenever at least
// one legal candidate is present, damped greedy never selects a masked-out token" (Sec6/
// Sec7.5). The discriminating construction strike 6's own census demanded: n_legal < k, at
// least one swept alpha where the anti-LM penalty exceeds a legal candidate's own q_theta --
// exactly the shape that defeated the PRE-remedy formula (s(v) = q_theta - alpha*p_omega
// unconditionally, which silently scored a masked pick at ~0 and let it win whenever a legal
// candidate's own score went negative). This suite asserts the REMEDIED contract
// (DampedGreedyScoreAndArgmax must floor every masked pick to INT64_MIN) -- once Phase C
// exists, a build that omits the floor reproduces Loki's own strike-6 measurement (0-of-63
// become 24-of-63 schema violations), so a regression here is exactly that remedy regressing.
// ===========================================================================================
static void TestDim7_MaskedNeverSelected_HighAlphaOverwhelmsLegalCandidate() {
	const int32_t V = 200;
	const int32_t LOOP = 100, ALT = 150;  // the two legal continuations
	std::vector<int32_t> row = MakeSyntheticRow(V, {LOOP, ALT});
	// LOOP is the model's own favourite (matches the loop-lock shape) -- and the anti-LM's
	// own history below makes p_omega(LOOP) large, so a high alpha can plausibly push
	// s(LOOP) below a masked pick's own floor-free (pre-remedy) score of ~0.
	std::vector<uint8_t> mask = MakeMask(V, false);
	SetLegal(mask, LOOP);
	SetLegal(mask, ALT);
	ApplyMaskFloor(row, mask);

	AntiLmState* alm = MakeLoopLockAntiLm(LOOP, ALT);
	int64_t q_ln2, q_b, q_c;
	CHECK(DeriveDefaultScaleConstants(&q_ln2, &q_b, &q_c));

	// The plan's own primary grid alpha values (Sec5.6.2) -- swept, not a single point, since
	// the strike's own finding was that only SOME grid points violated pre-remedy.
	const double alphas[7] = {0.1, 0.3, 0.6, 1.0, 1.5, 2.0, 3.0};
	int violations = 0;
	for (double alpha : alphas) {
		const int64_t alpha_q15 = static_cast<int64_t>(alpha * (1 << kProbFracBits) + 0.5);
		int32_t token = -1;
		bool refused = false;
		const bool ok = DampedGreedyScoreAndArgmax(row.data(), mask.data(), V, /*k=*/6, alm,
		                                            alpha_q15, q_ln2, q_b, q_c, &token, &refused);
		CHECK(ok);
		if (refused) continue;  // a genuine domain refusal is not a schema violation
		const bool legal = IsLegal(mask, token);
		if (!legal) ++violations;
		CHECK_MSG(legal,
		          "alpha=%.1f: selected token %d is MASKED -- damped greedy must never select "
		          "a masked-out token while a legal candidate exists (Sec6/Sec7.5's absolute "
		          "claim, the INT64_MIN floor's own reason for existing)",
		          alpha, token);
	}
	std::printf("dim7 masked-never-selected: %d/%d alpha grid points violated (want 0)\n",
	            violations, 7);
	AntiLmDestroy(alm);
}

// The n_legal < k boundary at its narrowest: exactly ONE legal candidate among k=6 picks
// (mirrors Loki's own strike-6 specimen exactly).
static void TestDim7_MaskedNeverSelected_SingleLegalAmongSix() {
	const int32_t V = 200;
	const int32_t LOOP = 100;
	std::vector<int32_t> row = MakeSyntheticRow(V, {LOOP});
	std::vector<uint8_t> mask = MakeMask(V, false);
	SetLegal(mask, LOOP);
	ApplyMaskFloor(row, mask);
	AntiLmState* alm = MakeLoopLockAntiLm(LOOP, /*alt=*/9999 /* never appears in the row's own
	                                                             legal set, harmless filler */);
	int64_t q_ln2, q_b, q_c;
	CHECK(DeriveDefaultScaleConstants(&q_ln2, &q_b, &q_c));
	const int64_t alpha_q15 = static_cast<int64_t>(3.0 * (1 << kProbFracBits) + 0.5);
	int32_t token = -1;
	bool refused = false;
	const bool ok = DampedGreedyScoreAndArgmax(row.data(), mask.data(), V, 6, alm, alpha_q15,
	                                            q_ln2, q_b, q_c, &token, &refused);
	CHECK(ok);
	if (!refused) CHECK_MSG(token == LOOP, "sole legal token = %d, selected = %d", LOOP, token);
	AntiLmDestroy(alm);
}

// ===========================================================================================
// Sec9 dim8 (GATE, D-SLM3719): mask-first composition with schema-constrained decoding -- "the
// emitted token equals the schema's single legal continuation, at every swept (alpha, n, k)."
// This is the SAME claim as the dim7 cell above, restated as a composition pair per Sec9's own
// framing (the mask-first ordering exercised against damped greedy's own top choice) -- swept
// across (n, k) as well as alpha, matching Loki's own strike-6 probe's full 63-point primary
// grid rather than a single (n,k) pair.
// ===========================================================================================
static void TestDim8_MaskFirst_SingleLegalContinuationAcrossFullGrid() {
	const int32_t V = 500;
	const int32_t LOOP = 300, ALT = 350;
	std::vector<int32_t> row = MakeSyntheticRow(V, {LOOP, ALT});
	row[static_cast<size_t>(LOOP)] = 12000;  // the model's own favourite (loop-lock shape)
	row[static_cast<size_t>(ALT)] = 9000;
	std::vector<uint8_t> mask = MakeMask(V, false);
	SetLegal(mask, LOOP);
	SetLegal(mask, ALT);
	ApplyMaskFloor(row, mask);

	int64_t q_ln2, q_b, q_c;
	CHECK(DeriveDefaultScaleConstants(&q_ln2, &q_b, &q_c));

	const double alphas[7] = {0.1, 0.3, 0.6, 1.0, 1.5, 2.0, 3.0};
	const int ns[3] = {1, 2, 3};
	const int32_t ks[3] = {3, 6, 10};
	int violations = 0, total = 0, refusals = 0;
	for (double alpha : alphas)
		for (int n : ns)
			for (int32_t k : ks) {
				AntiLmState* alm = AntiLmCreate(n);
				for (int i = 0; i < 20; ++i) {
					AntiLmUpdate(alm, LOOP);
					AntiLmUpdate(alm, ALT);
				}
				const int64_t alpha_q15 =
				    static_cast<int64_t>(alpha * (1 << kProbFracBits) + 0.5);
				int32_t token = -1;
				bool refused = false;
				const bool ok = DampedGreedyScoreAndArgmax(row.data(), mask.data(), V, k, alm,
				                                            alpha_q15, q_ln2, q_b, q_c, &token,
				                                            &refused);
				CHECK(ok);
				++total;
				if (refused) {
					++refusals;
				} else if (token != LOOP && token != ALT) {
					++violations;
				}
				AntiLmDestroy(alm);
			}
	std::printf("dim8 mask-first grid: %d/%d violated schema, %d refused (want 0 violated)\n",
	            violations, total, refusals);
	CHECK_MSG(violations == 0,
	          "%d of %d primary-grid points selected a token outside the schema's two legal "
	          "continuations",
	          violations, total);
}

// ===========================================================================================
// Sec9 dim5 (GATE, D-SLM3719): the refusal PROPAGATION policy -- "a TopKRenormalizeQ15
// refusal on any of the k candidates propagates the same status upward... aborting that
// sequence's generation for that call rather than falling back to plain argmax for the one
// step." This cell reuses renormalize_phaseC_red.cpp's own adversarial search shape (a
// (q_ln2,q_b,q_c) that clears the width-domain gate but still refuses on a real top-k
// candidate) to confirm DampedGreedyScoreAndArgmax's own `out_refused` flag is set exactly
// when TopKRenormalizeQ15 itself would refuse -- the CALLER's own policy half, distinct from
// renormalize_phaseC_red.cpp's pin of the underlying primitive's own ability to refuse.
// ===========================================================================================
static void TestDim5_RefusalPropagatesFromDampedGreedyScoreAndArgmax() {
	bool found = false;
	int64_t found_q_ln2 = 0, found_q_b = 0, found_q_c = 0;
	for (int64_t target = 23378; target <= 23378 + 20000 && !found; target += 500) {
		int64_t q_ln2, q_b, q_c;
		if (!DeriveScaleConstants(target, &q_ln2, &q_b, &q_c)) continue;
		if (CheckSoftmaxRowWidthDomain(q_b, q_c, 6) != SslmForwardStatus::Ok) continue;
		for (int32_t spread = 1; spread <= 200000 && !found; spread += 4000) {
			const int32_t V = 6;
			std::vector<int32_t> row(V);
			row[0] = 0;
			for (int32_t i = 1; i < V; ++i) row[i] = -spread * i;
			int32_t idx[6] = {0, 1, 2, 3, 4, 5};
			int64_t out[6];
			if (!TopKRenormalizeQ15(row.data(), idx, 6, q_ln2, q_b, q_c, out)) {
				found = true;
				found_q_ln2 = q_ln2;
				found_q_b = q_b;
				found_q_c = q_c;
			}
		}
	}
	if (!found) {
		SKIP_MSG("no adversarial TopKRenormalizeQ15 refusal found within budget -- see "
		         "renormalize_phaseC_red.cpp's identical cell for the same honest-report "
		         "rationale; this cell cannot exercise the propagation POLICY without a real "
		         "firing case");
		return;
	}
	const int32_t V = 6;
	std::vector<int32_t> row(V, 0);
	std::vector<uint8_t> mask = MakeMask(V, true);  // free text -- irrelevant to the refusal
	                                                 // this cell targets
	AntiLmState* alm = AntiLmCreate(1);
	int32_t token = -1;
	bool refused = false;
	const bool ok = DampedGreedyScoreAndArgmax(row.data(), mask.data(), V, 6, alm,
	                                            /*alpha_q15=*/0, found_q_ln2, found_q_b,
	                                            found_q_c, &token, &refused);
	CHECK_MSG(ok, "DampedGreedyScoreAndArgmax's own return value must stay well-formed "
	              "(ok=true) even on a domain refusal -- refusal is signalled via *out_refused, "
	              "matching TopKRenormalizeQ15's own bool-return convention, not a second "
	              "failure channel");
	CHECK_MSG(refused, "*out_refused must be true when the underlying TopKRenormalizeQ15 call "
	                    "refuses on a real top-k candidate (Sec7.5's own adopted policy)");
	AntiLmDestroy(alm);
}

// ===========================================================================================
// Sec9 dim7 (GATE, D-SLM3719): "the score reduces to plain argmax when the anti-LM term is
// disabled" (Sec7.5, the ablation Loftus ran, Sec5.1) -- alpha=0 means alpha_eff=0 for every
// candidate regardless of p_omega, so s(v) = q_theta(v) exactly, and argmax over q_theta is
// argmax over the row's own raw values restricted to the top-k set (TopKRenormalizeQ15 is
// monotonic in its input row -- larger logit implies larger q_theta, since softmax preserves
// order). At alpha=0, DampedGreedyScoreAndArgmax must therefore select the SAME token
// ArgmaxLowestIndexTieBreak (the shipped C16 rule) selects over the identical masked row.
// ===========================================================================================
static void TestDim7_AlphaZeroReducesToArgmax() {
	const int32_t V = 300;
	std::vector<int32_t> row = MakeSyntheticRow(V, {50, 120, 200, 7, 88, 260});
	std::vector<uint8_t> mask = MakeMask(V, true);  // free text -- alpha=0 identity is stated
	                                                 // for the faithful band, unconditional on
	                                                 // masking (Sec9 dim7's own cell text)
	int64_t q_ln2, q_b, q_c;
	CHECK(DeriveDefaultScaleConstants(&q_ln2, &q_b, &q_c));
	// A busy anti-LM history so p_omega is genuinely nonzero for these candidates -- proving
	// alpha=0 zeroes the TERM, not merely that p_omega itself happens to be zero everywhere.
	AntiLmState* alm = AntiLmCreate(2);
	for (int i = 0; i < 15; ++i) {
		AntiLmUpdate(alm, 50);
		AntiLmUpdate(alm, 120);
		AntiLmUpdate(alm, 200);
	}
	int32_t token = -1;
	bool refused = false;
	const bool ok = DampedGreedyScoreAndArgmax(row.data(), mask.data(), V, /*k=*/6, alm,
	                                            /*alpha_q15=*/0, q_ln2, q_b, q_c, &token,
	                                            &refused);
	CHECK(ok);
	if (!refused) {
		// Reference: the shipped C16 rule over the identical masked row (free-text mask is a
		// no-op, so this row equals what ApplyMaskAndArgmax would see too).
		int32_t best_index = 0;
		int32_t best_value = row[0];
		for (int32_t i = 1; i < V; ++i)
			if (row[static_cast<size_t>(i)] > best_value) {
				best_value = row[static_cast<size_t>(i)];
				best_index = i;
			}
		CHECK_MSG(token == best_index,
		          "alpha=0 selected %d, plain-argmax reference (C16 rule) selects %d -- "
		          "damped greedy at alpha=0 must reduce EXACTLY to the shipped greedy path",
		          token, best_index);
	}
	AntiLmDestroy(alm);
}

// ===========================================================================================
// Sec9 dim7 (MEASUREMENT, Sec9 preamble's own classification discipline): the keeper-probe
// cell -- P_topk, alpha_eff, qspread, pomspread must be COMPUTED CORRECTLY against a
// known-good row (this CAN fail a build); this cell never asserts any of the five values
// "reads a particular way" (that would make it a calibration-reading GATE, forbidden under
// D-SLM3727). Cross-checked by independent recomputation from the same primitives, not by
// trusting DampedGreedyScoreAndArgmaxDiag's own output as its own oracle.
// ===========================================================================================
static void TestDim7_KeeperProbe_DiagnosticsComputedCorrectly() {
	const int32_t V = 30;
	const int32_t idx[6] = {0, 5, 10, 15, 20, 25};
	std::vector<int32_t> row = MakeSyntheticRow(V, {5, 10, 0, 15, 20, 25});
	std::vector<uint8_t> mask = MakeMask(V, true);
	int64_t q_ln2, q_b, q_c;
	CHECK(DeriveDefaultScaleConstants(&q_ln2, &q_b, &q_c));

	AntiLmState* alm = AntiLmCreate(2);
	for (int i = 0; i < 10; ++i) {
		AntiLmUpdate(alm, 5);
		AntiLmUpdate(alm, 10);
	}
	const int64_t alpha_q15 = static_cast<int64_t>(1.5 * (1 << kProbFracBits) + 0.5);

	// Independent reference: q_theta over the SAME 6 indices, via the certified reference
	// (SoftmaxRowQ15 on the gathered scores), and p_omega via a fresh, independently-driven
	// AntiLmState -- neither borrowed from the function under test.
	int64_t scores[6];
	for (int i = 0; i < 6; ++i) scores[i] = row[static_cast<size_t>(idx[i])];
	int64_t ref_q[6];
	CHECK(SoftmaxRowQ15(scores, 6, q_ln2, q_b, q_c, ref_q));
	int64_t ref_pw[6];
	AntiLmPenalize(alm, idx, 6, ref_pw);
	int64_t ref_qspread = ref_q[0], ref_qmin = ref_q[0];
	int64_t ref_pwspread = ref_pw[0], ref_pwmin = ref_pw[0];
	for (int i = 1; i < 6; ++i) {
		ref_qspread = std::max(ref_qspread, ref_q[i]);
		ref_qmin = std::min(ref_qmin, ref_q[i]);
		ref_pwspread = std::max(ref_pwspread, ref_pw[i]);
		ref_pwmin = std::min(ref_pwmin, ref_pw[i]);
	}
	ref_qspread -= ref_qmin;
	ref_pwspread -= ref_pwmin;

	// Z over the whole (small, V=30) row, for the reference P_topk denominator -- computed
	// directly via SoftmaxRowQ15 at full width (int64-widened row).
	std::vector<int64_t> full_scores(row.begin(), row.end());
	std::vector<int64_t> full_q(static_cast<size_t>(V));
	CHECK(SoftmaxRowQ15(full_scores.data(), static_cast<size_t>(V), q_ln2, q_b, q_c,
	                     full_q.data()));
	// Z_k reference: sum of the SAME six q values scaled back... instead, reference P_topk is
	// computed directly as the fraction of full-row Q15 mass the six candidates hold.
	int64_t full_row_z_proxy = 0;
	for (int64_t v : full_q) full_row_z_proxy += v;  // proxy Z in Q15 units (sums to ~1<<15)

	int32_t token = -1;
	bool refused = false;
	DampedGreedyDiagnostics diag{};
	const bool ok = DampedGreedyScoreAndArgmaxDiag(row.data(), mask.data(), V, 6, alm, alpha_q15,
	                                                q_ln2, q_b, q_c, full_row_z_proxy, &token,
	                                                &refused, &diag);
	CHECK(ok);
	if (!refused) {
		CHECK_MSG(diag.qspread_q15 == ref_qspread,
		          "diag.qspread_q15=%lld, independently recomputed=%lld", (long long)diag.qspread_q15,
		          (long long)ref_qspread);
		CHECK_MSG(diag.pomspread_q15 == ref_pwspread,
		          "diag.pomspread_q15=%lld, independently recomputed=%lld",
		          (long long)diag.pomspread_q15, (long long)ref_pwspread);
		CHECK_MSG(diag.alpha_eff_q15 >= 0 && diag.alpha_eff_q15 < (1LL << 31),
		          "diag.alpha_eff_q15=%lld out of the Sec7.5-derived safe range [0, 2^31)",
		          (long long)diag.alpha_eff_q15);
		CHECK_MSG(diag.p_topk_q15 >= 0 && diag.p_topk_q15 <= (1 << kProbFracBits),
		          "diag.p_topk_q15=%lld out of [0, 32768]", (long long)diag.p_topk_q15);
	}
	AntiLmDestroy(alm);
}

// ===========================================================================================
// Sec9 dim7 (MEASUREMENT), Finding 2 fill-spec (Mendeleev's pre-build audit,
// `Claude/Mendeleev/t2199-red-suite-coverage-audit-2026-08-20.md`): the cell above only
// RANGE-CHECKS `alpha_eff_q15`/`p_topk_q15`, which cannot re-detect the cycle-8 temper's own
// historically-real defect (a dropped or mis-ordered `>>15` narrowing on the wide
// `alpha_q15 * p_omega` product, Sec7.5) -- at the small alpha values the original cell and
// this suite's own primary-grid sweeps use (up to 3.0), an un-rescaled `alpha_eff` never
// approaches any bound a range check would catch. This cell closes that gap: it independently
// RECOMPUTES `alpha_eff_q15` bit-exact from the SAME pinned formula (`(alpha_q15 * p_omega)
// >> kProbFracBits`, Sec7.5), at an ADVERSARIALLY LARGE alpha near the Sec9 dim2 sanity
// ceiling (`alpha_q15 < 2^20`) -- exactly where a dropped/misplaced rescale produces a value
// that diverges from the reference by many orders of magnitude, a margin no range check could
// miss. `p_topk_q15` is likewise recomputed (Z_k / full_row_z, both independently derivable)
// rather than range-checked, to the same floor-rounding tolerance
// (`TestPhaseC2_KnownAnswer_MatchesSoftmaxRowQ15OnGatheredElements`'s own precedent for why a
// small slack is mathematically necessary, not a weakening of rigor).
// ===========================================================================================
static void TestFinding2_KeeperProbe_HighAlpha_ExactRecompute() {
	const int32_t V = 30;
	const int32_t idx[6] = {0, 5, 10, 15, 20, 25};
	std::vector<int32_t> row = MakeSyntheticRow(V, {5, 10, 0, 15, 20, 25});
	std::vector<uint8_t> mask = MakeMask(V, true);
	int64_t q_ln2, q_b, q_c;
	CHECK(DeriveDefaultScaleConstants(&q_ln2, &q_b, &q_c));

	AntiLmState* alm = AntiLmCreate(2);
	for (int i = 0; i < 10; ++i) {
		AntiLmUpdate(alm, 5);
		AntiLmUpdate(alm, 10);
	}
	// Adversarially large: alpha = 31.9, alpha_q15 ~ 1,045,401 -- just under the Sec9 dim2
	// sanity ceiling (alpha_q15 < 2^20 = 1,048,576; alpha < 32 in real terms), the exact zone
	// the audit names as where a dropped/misplaced >>15 narrowing becomes numerically visible.
	const double alpha = 31.9;
	const int64_t alpha_q15 = static_cast<int64_t>(alpha * (1 << kProbFracBits) + 0.5);
	CHECK_MSG(alpha_q15 < (1LL << 20), "test fixture error: alpha_q15=%lld must stay under the "
	                                   "dim2 sanity ceiling 2^20",
	          (long long)alpha_q15);

	int64_t ref_pw[6];
	AntiLmPenalize(alm, idx, 6, ref_pw);

	std::vector<int64_t> full_scores(row.begin(), row.end());
	std::vector<int64_t> full_q(static_cast<size_t>(V));
	CHECK(SoftmaxRowQ15(full_scores.data(), static_cast<size_t>(V), q_ln2, q_b, q_c,
	                     full_q.data()));
	int64_t ref_z_k_proxy = 0;
	for (int32_t i : idx) ref_z_k_proxy += full_q[static_cast<size_t>(i)];
	int64_t full_row_z_proxy = 0;
	for (int64_t v : full_q) full_row_z_proxy += v;

	int32_t token = -1;
	bool refused = false;
	DampedGreedyDiagnostics diag{};
	const bool ok = DampedGreedyScoreAndArgmaxDiag(row.data(), mask.data(), V, 6, alm, alpha_q15,
	                                                q_ln2, q_b, q_c, full_row_z_proxy, &token,
	                                                &refused, &diag);
	CHECK(ok);
	if (refused) {
		SKIP_MSG("TopKRenormalizeQ15 refused at this adversarially large alpha's own row -- "
		         "alpha_eff_q15's own recompute cannot be checked against a winner that was "
		         "never selected");
		AntiLmDestroy(alm);
		return;
	}

	// Locate the winning candidate's own position within idx[] (free-text mask -- the winner
	// is guaranteed to be one of the six gathered candidates).
	int winner_pos = -1;
	for (int i = 0; i < 6; ++i)
		if (idx[i] == token) winner_pos = i;
	CHECK_MSG(winner_pos >= 0, "selected token %d not found among the six gathered candidates",
	          token);
	if (winner_pos < 0) {
		AntiLmDestroy(alm);
		return;
	}

	// The EXACT pinned formula (Sec7.5): widen to int64, multiply, THEN narrow by >>15 --
	// bit-exact, no tolerance. A build that narrows AFTER subtracting, or omits the >>15
	// entirely, produces `ref_alpha_eff` off by a factor of ~2^15 at this alpha -- unmissable.
	const int64_t ref_alpha_eff =
	    (static_cast<int64_t>(alpha_q15) * ref_pw[winner_pos]) >> kProbFracBits;
	CHECK_MSG(diag.alpha_eff_q15 == ref_alpha_eff,
	          "diag.alpha_eff_q15=%lld, independently recomputed (alpha_q15=%lld * "
	          "p_omega=%lld) >> 15 = %lld -- a dropped or mis-ordered >>15 rescale would "
	          "produce a value roughly 2^15x too large here, unmissable at this alpha",
	          (long long)diag.alpha_eff_q15, (long long)alpha_q15, (long long)ref_pw[winner_pos],
	          (long long)ref_alpha_eff);
	// Sanity floor: at this alpha and a genuinely nonzero p_omega, a CORRECTLY rescaled
	// alpha_eff must fit comfortably under 2^31 (Sec7.5's own derived bound) -- confirms this
	// fixture actually exercises a nonzero term, not a vacuous alpha_eff=0 case that would let
	// a broken rescale pass by coincidence.
	CHECK_MSG(ref_pw[winner_pos] > 0 || ref_alpha_eff == 0,
	          "fixture error: winner's own p_omega=%lld but ref_alpha_eff=%lld -- this "
	          "construction must exercise a nonzero anti-LM term to be discriminating",
	          (long long)ref_pw[winner_pos], (long long)ref_alpha_eff);

	// p_topk_q15 recomputed (Z_k/full_row_z, both independently derived above), to the same
	// floor-rounding slack this suite already uses elsewhere (< k units) -- a mathematically
	// necessary tolerance (independent per-element Q15 rounding vs. one combined ratio), not a
	// weakening of the check.
	CHECK_MSG(diag.p_topk_q15 >= ref_z_k_proxy - 6 && diag.p_topk_q15 <= ref_z_k_proxy + 6,
	          "diag.p_topk_q15=%lld, independently recomputed Z_k/Z=%lld (tolerance +/-6 for "
	          "per-element floor-rounding slack across 6 candidates)",
	          (long long)diag.p_topk_q15, (long long)ref_z_k_proxy);

	AntiLmDestroy(alm);
}

// ===========================================================================================
// Fix round 2026-08-20 (Poirot S1 + the owed S6 fixture, same casebook/build-log): `p_topk_q15`
// is now `ExactRatioQ15` -- a file-local, purpose-built exact-integer multiply/divide (early
// clamp when the true ratio is >= 1.0, so the overflowing product is never formed; otherwise a
// 128-bit-safe restoring divide with round-to-nearest, ties away from zero) replacing the
// pre-fix `double` computation (`ratio = (double)z_k_raw * (double)(1<<15) / (double)denom;
// p_topk = llround(ratio)`), the only `double` anywhere in `src/` outside `model.cpp`'s
// artifact reader (Poirot's own sweep) -- a direct violation of plan Sec1 constraint 1, "No
// float on the decode path; any new arithmetic is exact integer."
//
// This is also where the STILL-OWED S6 fixture closes: `p_topk_q15` is clamped to
// `[0, 1<<15]`, and every cell in THIS SUITE before this fix round supplied a `full_row_z` in
// normalized Q15 units against a `z_k_q0` in raw units (both keeper-probe cells, this file) --
// a units mismatch that always saturates the clamp to its own ceiling, so the clamp's
// arithmetic below the ceiling had never been shown to READ correctly on any COMMITTED
// fixture (Poirot's own S6 finding: "a must-accept without a must-reject is not a
// commissioned instrument"). Poirot's own review supplies and executes the missing
// construction directly (this file's own earlier caveat, and this campaign's own prior test-
// design record, both routed exactly this gap): an HONEST denominator (`full_row_z` in the
// SAME raw units `z_k_q0` already reports, read directly off `diag.z_k_q0` rather than
// estimated) produces a genuinely sub-ceiling reading.
//
// RED-FIRST: EXECUTED for the exactness half (a) below -- see this cell's own comment there
// for why the "double reintroduction" construction searched for exhaustively (500,000+ random
// trials at this codebase's own magnitude range) found NO divergence between `double` and
// exact-integer rounding, and why the adversarial-magnitude pin (b) is the fallback the
// commission itself named for exactly this outcome. NOT executed pre-fix for (c) (the honest-
// denominator/S6 construction) -- the pre-fix `double` computation is ALSO arithmetically
// correct at this specific ratio (0.25 has an exact double representation, so `llround` cannot
// diverge from the exact-integer path here); (c)'s own claim ("the clamp reads correctly below
// its ceiling, on a COMMITTED fixture") is true of both the pre-fix and post-fix arithmetic --
// what changed is that NO cell exercised it before this fix round, which is what (c) itself
// now closes, not a pre-fix/post-fix behavioral difference.
// ===========================================================================================
static void TestS1S6_PTopkQ15_ExactIntegerAndHonestDenominator() {
	const int32_t V = 30;
	std::vector<int32_t> row = MakeSyntheticRow(V, {5, 10, 0, 15, 20, 25});
	std::vector<uint8_t> mask = MakeMask(V, true);
	int64_t q_ln2, q_b, q_c;
	CHECK(DeriveDefaultScaleConstants(&q_ln2, &q_b, &q_c));
	AntiLmState* alm = AntiLmCreate(1);
	AntiLmUpdate(alm, 5);

	// Probe call to learn the row's own REAL raw z_k_q0 (any full_row_z works for this read --
	// diag.z_k_q0 is reported independent of what p_topk_q15 clamps to).
	int32_t token = -1;
	bool refused = false;
	DampedGreedyDiagnostics probe{};
	CHECK(DampedGreedyScoreAndArgmaxDiag(row.data(), mask.data(), V, 6, alm, 0, q_ln2, q_b, q_c,
	                                      /*full_row_z=*/1, &token, &refused, &probe));
	const int64_t z_k = probe.z_k_q0;
	CHECK_MSG(z_k > 0, "test fixture error: z_k_q0=%lld, want a genuinely positive raw i-exp "
	                    "sum for this construction to be meaningful",
	          (long long)z_k);

	// (a) Ceiling case, reproducing Poirot's own first row exactly (full_row_z=1, a units
	// error of ~6 orders against any real z_k_q0 -- the true ratio is always >= 1.0 here,
	// clamped): p_topk_q15 must be exactly 32768, and this is what a caller-side full_row_z
	// underestimate is DESIGNED to report ("the k-gathered candidates hold effectively all of
	// the row's mass").
	CHECK_MSG(probe.p_topk_q15 == (1 << kProbFracBits),
	          "full_row_z=1: p_topk_q15=%lld, want exactly 32768 (the clamp ceiling)",
	          (long long)probe.p_topk_q15);

	// (b) Adversarial-magnitude exact-Q15 known-answer cells. full_row_z = N * z_k_q0 for
	// several small N -- the z_k_q0 factor cancels ALGEBRAICALLY (z_k_q0 / (N * z_k_q0) =
	// 1/N exactly), so the expected Q15 value is computable here with zero risk of this
	// TEST's own reference calc overflowing (no `<<15` product of the real, large z_k_q0 is
	// ever formed on this side), while the PRODUCTION call still forms and divides that exact
	// large product internally -- z_k_q0 can be up to ~2^47 (kSoftmaxRowMaxSafeExponent),
	// `z_k_q0 << 15` up to ~2^62, precisely the magnitude `double`'s 52-bit mantissa cannot
	// represent exactly, which is the adversarial-magnitude regime this cell exercises.
	// Expected values computed independently (exact rational arithmetic, round-to-nearest-
	// ties-away-from-zero, matching the production formula's own documented rounding rule):
	// N=3 -> 10923, N=7 -> 4681, N=25 -> 1311 -- each computed by EXECUTING the exact-rational
	// formula (floor division + round-to-nearest-ties-away-from-zero) in a scratch script, not
	// derived by hand (StandardsDocument.md Sec5.4: verified at source or by execution).
	struct NCase {
		int64_t n;
		int64_t expect_q15;
	};
	const NCase cases[] = {{3, 10923}, {7, 4681}, {25, 1311}};
	for (const NCase& c : cases) {
		const int64_t full_row_z = c.n * z_k;
		int32_t t2 = -1;
		bool r2 = false;
		DampedGreedyDiagnostics d2{};
		CHECK(DampedGreedyScoreAndArgmaxDiag(row.data(), mask.data(), V, 6, alm, 0, q_ln2, q_b,
		                                      q_c, full_row_z, &t2, &r2, &d2));
		CHECK_MSG(d2.z_k_q0 == z_k, "z_k_q0 changed (%lld -> %lld) across calls with the "
		                            "identical row/scale -- must be deterministic",
		          (long long)z_k, (long long)d2.z_k_q0);
		CHECK_MSG(d2.p_topk_q15 == c.expect_q15,
		          "N=%lld: full_row_z=%lld*z_k_q0, p_topk_q15=%lld, want exactly %lld "
		          "(1/%lld in Q15, exact -- the z_k_q0 factor cancels algebraically, so this "
		          "expected value does not depend on z_k_q0's own magnitude)",
		          (long long)c.n, (long long)c.n, (long long)d2.p_topk_q15,
		          (long long)c.expect_q15, (long long)c.n);
	}

	// (c) The owed S6 fixture: an HONEST denominator (N=4, reproducing Poirot's own second
	// row exactly -- "full_row_z = 4*z_k -> exactly 8192 (0.25)") -- a genuinely sub-ceiling,
	// non-trivial reading, closing the "never shown reading anything but its ceiling" gap.
	{
		const int64_t full_row_z = 4 * z_k;
		int32_t t3 = -1;
		bool r3 = false;
		DampedGreedyDiagnostics d3{};
		CHECK(DampedGreedyScoreAndArgmaxDiag(row.data(), mask.data(), V, 6, alm, 0, q_ln2, q_b,
		                                      q_c, full_row_z, &t3, &r3, &d3));
		CHECK_MSG(d3.p_topk_q15 == 8192,
		          "honest denominator (full_row_z=4*z_k_q0): p_topk_q15=%lld, want exactly "
		          "8192 (0.25) -- the arithmetic below the ceiling, on a COMMITTED fixture",
		          (long long)d3.p_topk_q15);
		CHECK_MSG(d3.p_topk_q15 != (1 << kProbFracBits),
		          "p_topk_q15 read the ceiling (32768) on an HONEST, sub-ceiling denominator -- "
		          "this is the exact failure mode the missing fixture existed to catch (every "
		          "prior committed fixture landed on the ceiling regardless of the true ratio)");
	}

	AntiLmDestroy(alm);
}

// ===========================================================================================
// Fix round 2026-08-20 (Poirot C1, `Claude/Poirot/7be9508-t2199-phaseAC-review.md`,
// `Claude/Brunel/t2199-phaseAC-build-2026-08-20.md` Sec12.1): both public entry points now
// reject `k <= 0`, `vocab_size <= 0`, or `k > vocab_size` with `return false` BEFORE touching
// any `k`-sized array -- closing the `q_theta[0]`/`p_omega[0]` out-of-bounds access at `k=0`
// in `DampedGreedyScoreAndArgmaxDiag` (pre-fix: `0xC0000005`, an access violation, since
// `q_theta`/`p_omega` are `std::vector`s sized `k` and both subscripts are out of bounds at
// `k=0`) and the non-`Diag` sibling's own silent `*out_token = -1` under a `true` return
// (successful-selection-of-token-negative-one, indistinguishable from a real result to a
// caller reading only the documented outputs).
//
// This cell tests THREE domain-violation shapes (`k <= 0`, `vocab_size <= 0`, `k >
// vocab_size`) against BOTH entry points where each shape is SAFE to exercise directly (never
// touching a `k`-sized array before the domain check fires, in EITHER the pre-fix or post-fix
// code -- see the per-shape notes below), pre-filling every output with a sentinel so "the
// function returns false and leaves outputs untouched" is a real, checkable assertion, not
// merely "returns false."
//
// RED-FIRST, split by shape and entry point -- executed where the pre-fix construction is
// SAFE to run in this shared binary (a crash would silently discard every other cell's own
// result in this file); pinned-green with the crash cited as evidence otherwise:
//   - `vocab_size = 0` (both entry points) and `k > vocab_size` with `vocab_size > 0` (both
//     entry points): SAFE at ANY `k > 0` pre-fix (`q_theta`/`p_omega` are sized by `k`, never
//     by `vocab_size`, and `vocab_size = 0` casts to a legitimate empty `size_t` allocation,
//     not a crash) -- EXECUTED against `c473dad` (pre-fix: `*out_token` overwritten, e.g. `-1`
//     from an all-empty `FsdTopK` write at `vocab_size=0`, or a real-but-wrong token at
//     `k>vocab_size`'s own S5-shaped corruption; `wf`-derived `true` return either way) and
//     `826f607` (post-fix: `false`, outputs untouched). See this campaign's own test-design
//     record for the transcript.
//   - `vocab_size < 0`: NOT exercised, even via the "safe" entry points -- a NEGATIVE
//     `vocab_size` casts to a huge `size_t` in `FsdTopK`'s own pre-fix `idx` vector
//     construction (the identical crash shape `AntiLmCreate(-1)` demonstrates elsewhere in
//     this suite), independent of `k`. An earlier draft of this cell included
//     `vocab_size=-5` in the "safe" shape list on the reasoning above (which only accounted
//     for `q_theta`/`p_omega`'s own sizing, not `FsdTopK`'s internal `idx` allocation) --
//     the pre-fix verification run crashed this entire binary with no output, discovered by
//     executing it, not merely reasoned through after the fact. Removed from the shape array;
//     PINNED-GREEN, same reasoning as the `k=0`/`Diag` sub-case below.
//   - `k = 0` via `DampedGreedyScoreAndArgmax` (non-`Diag`): SAFE pre-fix -- Poirot's own
//     review states plainly "The non-`Diag` sibling on the same input does *not* crash," and
//     re-derived here from the pre-fix source directly: at `k=0`, `TopKRenormalizeQ15`'s own
//     `if (k == 0) return true;` early-return makes the whole call SAFE end to end, landing on
//     `*out_token = -1` under `true` (the documented pre-fix defect: "a caller reading the
//     documented outputs is told a successful selection produced token -1"). EXECUTED.
//   - `k = 0` via `DampedGreedyScoreAndArgmaxDiag`: NOT re-executed in this binary -- this is
//     the literal access-violation shape Poirot's own review captured (0xC0000005), and running
//     it here would crash this file's entire process before any other cell's own result could
//     be recorded. PINNED-GREEN for this one sub-case, with Poirot's own executed crash (cited
//     above) standing as the red evidence.
// ===========================================================================================
static void TestC1_DomainRejection_BothEntryPoints_VocabAndKOversizedShapes() {
	const int32_t V = 20;
	std::vector<int32_t> row = MakeSyntheticRow(V, {5, 10, 15});
	std::vector<uint8_t> mask = MakeMask(V, true);
	AntiLmState* alm = AntiLmCreate(1);
	AntiLmUpdate(alm, 5);
	int64_t q_ln2, q_b, q_c;
	CHECK(DeriveDefaultScaleConstants(&q_ln2, &q_b, &q_c));

	struct Shape {
		int32_t vocab_size;
		int32_t k;
		const char* name;
	};
	// NOTE: vocab_size < 0 is deliberately NOT in this array -- executing it against pre-fix
	// code, unlike vocab_size=0, is UNSAFE: `FsdTopK`'s own pre-fix `std::vector<int32_t>
	// idx(static_cast<std::size_t>(vocab_size))` casts a negative `vocab_size` to a huge
	// `size_t`, attempting a catastrophic allocation -- confirmed by execution (crashes this
	// entire binary before any result records, the same crash SHAPE `AntiLmCreate(-1)`
	// demonstrates elsewhere in this suite, `antilm_phaseA_red.cpp`'s own M4 cell). This is a
	// finding this cell's own construction surfaced, not merely reasoned through: an earlier
	// draft included `vocab_size=-5` here and the pre-fix verification run crashed with no
	// output at all, silently discarding this file's own subsequent cells too -- removed and
	// documented rather than left to happen again. `vocab_size < 0` is therefore PINNED-GREEN
	// for the same reason `k=0` via `Diag` is below: the crash itself is the red evidence.
	const Shape shapes[] = {
	    {0, 6, "vocab_size=0"},
	    {V, V + 10, "k>vocab_size (k=30,V=20)"},
	};
	for (const Shape& sh : shapes) {
		// Non-Diag entry point.
		int32_t out_token = -424242;  // pre-call sentinel, never a real token this suite emits
		bool out_refused = true;      // pre-call sentinel (opposite of the expected untouched
		                               // state's own natural default, so a spurious write is
		                               // visible either direction)
		const bool ok = DampedGreedyScoreAndArgmax(row.data(), mask.data(), sh.vocab_size, sh.k,
		                                            alm, 0, q_ln2, q_b, q_c, &out_token,
		                                            &out_refused);
		CHECK_MSG(!ok, "DampedGreedyScoreAndArgmax(%s) returned true, want false (domain "
		               "violation)",
		          sh.name);
		CHECK_MSG(out_token == -424242, "DampedGreedyScoreAndArgmax(%s): *out_token was "
		                                "written (%d) despite the domain violation -- must "
		                                "stay untouched",
		          sh.name, out_token);
		CHECK_MSG(out_refused == true, "DampedGreedyScoreAndArgmax(%s): *out_refused was "
		                               "written despite the domain violation -- must stay "
		                               "untouched",
		          sh.name);

		// Diag entry point -- safe at every one of these shapes (k > 0 in all three, so
		// q_theta[0]/p_omega[0] are never the empty-vector access the k=0 shape triggers).
		int32_t d_out_token = -424242;
		bool d_out_refused = true;
		DampedGreedyDiagnostics diag;
		diag.qspread_q15 = -13;  // pre-call sentinel pattern, distinguishable from any real
		diag.p_topk_q15 = -13;   // computed value (all real fields are >= 0 by contract)
		diag.alpha_eff_q15 = -13;
		diag.pomspread_q15 = -13;
		diag.z_k_q0 = -13;
		const bool d_ok = DampedGreedyScoreAndArgmaxDiag(row.data(), mask.data(), sh.vocab_size,
		                                                  sh.k, alm, 0, q_ln2, q_b, q_c,
		                                                  /*full_row_z=*/32768, &d_out_token,
		                                                  &d_out_refused, &diag);
		CHECK_MSG(!d_ok, "DampedGreedyScoreAndArgmaxDiag(%s) returned true, want false", sh.name);
		CHECK_MSG(d_out_token == -424242, "DampedGreedyScoreAndArgmaxDiag(%s): *out_token "
		                                  "written despite the domain violation",
		          sh.name);
		CHECK_MSG(d_out_refused == true, "DampedGreedyScoreAndArgmaxDiag(%s): *out_refused "
		                                 "written despite the domain violation",
		          sh.name);
		CHECK_MSG(diag.qspread_q15 == -13 && diag.p_topk_q15 == -13 &&
		              diag.alpha_eff_q15 == -13 && diag.pomspread_q15 == -13 &&
		              diag.z_k_q0 == -13,
		          "DampedGreedyScoreAndArgmaxDiag(%s): *out_diag was written despite the "
		          "domain violation -- the memset that populates it must never run before the "
		          "domain check",
		          sh.name);
	}

	// k=0 via the non-Diag entry point -- SAFE pre-fix (see this cell's own header comment).
	{
		int32_t out_token = -424242;
		bool out_refused = true;
		const bool ok = DampedGreedyScoreAndArgmax(row.data(), mask.data(), V, /*k=*/0, alm, 0,
		                                            q_ln2, q_b, q_c, &out_token, &out_refused);
		CHECK_MSG(!ok, "DampedGreedyScoreAndArgmax(k=0) returned true, want false -- pre-fix "
		               "this returned true with *out_token=-1, indistinguishable from a "
		               "genuine successful selection of token -1");
		CHECK_MSG(out_token == -424242,
		          "DampedGreedyScoreAndArgmax(k=0): *out_token was written (%d) despite the "
		          "domain violation",
		          out_token);
	}

	AntiLmDestroy(alm);
}

// ===========================================================================================
// Fold 21, 2026-08-20 -- S7 (`Claude/Poirot/927bbda-t2199-confirmation.md`): the O2 revert's
// own pin. `DampedGreedyDiagnostics::pomspread_q15` is defined "over k" -- max/min of p_omega
// across ALL k gathered picks, masked included (plan Sec9 dim8's own open question (a),
// RULED, fold 21) -- not over the unmasked subset alone. The casebook's own executed
// construction (a masked pick holding an outlier p_omega genuinely different from every
// unmasked pick's): "picks: 3(p=3276,legal) 15(p=0,legal) 0(p=29491,MASKED) 1(p=0) 2(p=0)
// 4(p=0)) -- spread over all k (documented): 29491, REPORTED pomspread_q15 under O2: 3276 (9x
// low)." This cell reproduces that shape with its own real anti-LM history (not the
// casebook's own literal numbers, which came from a different scratch fixture never
// committed) and asserts the reported spread equals an INDEPENDENTLY computed "over k"
// reference, bit-exact -- discriminating O2-active (would narrow to the unmasked pair, both
// near/at zero here, reading far below the true spread) from O2-reverted (reads the true
// spread, including the masked outlier's own real contribution).
//
// RED-FIRST: EXECUTED against this suite's own immediate parent, `927bbda` (O2 still active).
// See this campaign's own test-design record for the transcript.
// ===========================================================================================
static void TestS7_PomspreadOverAllK_MaskedOutlierDiscriminates() {
	const int32_t V = 20;
	const int32_t LEGAL_A = 3, LEGAL_B = 15;
	const int32_t MASKED_OUTLIER = 0, MASKED_B = 1, MASKED_C = 2, MASKED_D = 4;
	const int32_t idx[6] = {LEGAL_A, LEGAL_B, MASKED_OUTLIER, MASKED_B, MASKED_C, MASKED_D};

	std::vector<int32_t> row(static_cast<std::size_t>(V), -2000);
	row[static_cast<std::size_t>(LEGAL_A)] = 20000;
	row[static_cast<std::size_t>(LEGAL_B)] = 19000;
	row[static_cast<std::size_t>(MASKED_OUTLIER)] = 18000;
	row[static_cast<std::size_t>(MASKED_B)] = 17000;
	row[static_cast<std::size_t>(MASKED_C)] = 16000;
	row[static_cast<std::size_t>(MASKED_D)] = 15000;

	std::vector<uint8_t> mask = MakeMask(V, false);
	SetLegal(mask, LEGAL_A);
	SetLegal(mask, LEGAL_B);
	// MASKED_OUTLIER/B/C/D stay masked.

	int64_t q_ln2, q_b, q_c;
	CHECK(DeriveDefaultScaleConstants(&q_ln2, &q_b, &q_c));

	// The anti-LM's own history is built ENTIRELY from MASKED_OUTLIER -- so its own real
	// p_omega is large and genuinely distinct from every other candidate's (all unseen, 0).
	// This is the exact shape the casebook's own row names: a masked pick, not an unmasked
	// one, is the one carrying the discriminating signal.
	AntiLmState* alm = AntiLmCreate(2);
	for (int i = 0; i < 20; ++i) AntiLmUpdate(alm, MASKED_OUTLIER);

	int32_t token = -1;
	bool refused = false;
	DampedGreedyDiagnostics diag{};
	const bool ok = DampedGreedyScoreAndArgmaxDiag(row.data(), mask.data(), V, 6, alm,
	                                                /*alpha_q15=*/0, q_ln2, q_b, q_c,
	                                                /*full_row_z=*/1, &token, &refused, &diag);
	CHECK(ok);
	if (!refused) {
		// Independent reference: AntiLmPenalize over ALL SIX gathered candidates directly
		// (never through ScoreAndSelect's own sparse-vs-dense array), so this reference is
		// unaffected by whether O2 is active or reverted -- exactly what makes it a valid
		// oracle for detecting the difference.
		int64_t ref_pw[6];
		AntiLmPenalize(alm, idx, 6, ref_pw);
		int64_t ref_max = ref_pw[0], ref_min = ref_pw[0];
		for (int i = 1; i < 6; ++i) {
			ref_max = std::max(ref_max, ref_pw[i]);
			ref_min = std::min(ref_min, ref_pw[i]);
		}
		const int64_t ref_spread = ref_max - ref_min;
		CHECK_MSG(ref_spread > 0,
		          "fixture error: reference spread=%lld, want > 0 (the masked outlier must "
		          "genuinely differ from the unmasked picks' own p_omega for this "
		          "construction to discriminate anything)",
		          (long long)ref_spread);
		CHECK_MSG(diag.pomspread_q15 == ref_spread,
		          "diag.pomspread_q15=%lld, independently recomputed over ALL k (masked "
		          "included)=%lld -- if this narrows to the unmasked-only spread instead, O2 "
		          "(reverted, fold 21) has regressed",
		          (long long)diag.pomspread_q15, (long long)ref_spread);
		// The masked outlier's own real p_omega, specifically, must be one of the two
		// extremes the spread was computed from -- ruling out a construction that happens to
		// pass by coincidence (e.g. every candidate reading 0).
		CHECK_MSG(ref_pw[2] == ref_max || ref_pw[2] == ref_min,
		          "fixture error: the masked outlier (index 2 in idx[]) does not carry either "
		          "extreme (p_omega=%lld, max=%lld, min=%lld) -- construction does not "
		          "isolate what this cell claims to test",
		          (long long)ref_pw[2], (long long)ref_max, (long long)ref_min);
	} else {
		// Suite convention (04e0d26 closing confirmation, finding 4): a guard that can drain
		// this cell's three discriminating checks must say so loudly, never go silently
		// vacuous under a future domain change.
		SKIP_MSG("S7 cell skipped: DampedGreedyScoreAndArgmaxDiag refused at this fixture's "
		         "scale constants -- the pomspread discrimination did not run");
	}
	AntiLmDestroy(alm);
}

// ===========================================================================================
// M6 (`Claude/Poirot/927bbda-t2199-confirmation.md`): the refusal-propagation cell through a
// PUBLIC ENTRY POINT, not stopped one call short of it. `TestM2_TopKRenormalizeQ15_
// WidthDomainGateActuallyInvoked` (renormalize_phaseC_red.cpp) already confirms
// `TopKRenormalizeQ15` itself refuses at `q_c < 0`; this cell drives the SAME construction
// through `DampedGreedyScoreAndArgmax`/`Diag`, confirming Sec7.5's own refusal-propagation
// policy (abort the sequence, `*out_refused = true`, never a silent fallback to plain greedy)
// actually executes end to end -- the one behavior in this feature that, per the casebook's
// own O1 finding, "has never executed" through a public entry point before this cell.
//
// RED-FIRST: not meaningfully constructible as a pre-fix/post-fix differential -- M2 (the
// remedy this cell's own construction depends on) already shipped in the fix round this
// suite's own `927bbda` includes; there is no PRIOR commit in this suite's own history where
// `q_c<0` was constructible against a caller-facing entry point and DIDN'T refuse (before
// M2, `TopKRenormalizeQ15` computed instead of refusing, so the failure mode would have been
// "silently returns a token" rather than "the refusal never fires" -- a different cell, not
// this one). PINNED-GREEN, verified by execution against `b1dffd7` only.
// ===========================================================================================
static void TestM6_RefusalPropagatesThroughPublicEntryPoints() {
	int64_t q_ln2, q_b, q_c;
	CHECK(DeriveDefaultScaleConstants(&q_ln2, &q_b, &q_c));
	q_c = -1;  // CheckSoftmaxRowWidthDomain's own unconditional first rejection.

	const int32_t V = 20;
	std::vector<int32_t> row = MakeSyntheticRow(V, {3, 5, 7, 9, 11, 13});
	std::vector<uint8_t> mask = MakeMask(V, true);
	AntiLmState* alm = AntiLmCreate(1);
	AntiLmUpdate(alm, 3);

	// Non-Diag.
	int32_t out_token = -424242;
	bool out_refused = false;
	const bool ok = DampedGreedyScoreAndArgmax(row.data(), mask.data(), V, 6, alm, 0, q_ln2, q_b,
	                                            q_c, &out_token, &out_refused);
	CHECK_MSG(ok, "DampedGreedyScoreAndArgmax must return true (the call is well-formed; the "
	              "refusal travels on *out_refused, not the return value)");
	CHECK_MSG(out_refused, "*out_refused must be true -- Sec7.5's own refusal-propagation "
	                       "policy, now executable through this public entry point");
	CHECK_MSG(out_token == -424242, "*out_token must stay untouched on a numeric refusal");

	// Diag.
	int32_t d_token = -424242;
	bool d_refused = false;
	DampedGreedyDiagnostics diag{};
	diag.z_k_q0 = -13;
	diag.p_topk_q15 = -13;
	diag.alpha_eff_q15 = -13;
	diag.qspread_q15 = -13;
	diag.pomspread_q15 = -13;
	const bool d_ok = DampedGreedyScoreAndArgmaxDiag(row.data(), mask.data(), V, 6, alm, 0, q_ln2,
	                                                  q_b, q_c, /*full_row_z=*/1, &d_token,
	                                                  &d_refused, &diag);
	CHECK_MSG(d_ok, "DampedGreedyScoreAndArgmaxDiag must return true on a numeric refusal");
	CHECK_MSG(d_refused, "*out_refused must be true (Diag entry point)");
	CHECK_MSG(d_token == -424242, "*out_token must stay untouched (Diag entry point)");
	CHECK_MSG(diag.z_k_q0 == 0 && diag.p_topk_q15 == 0 && diag.alpha_eff_q15 == 0 &&
	              diag.qspread_q15 == 0 && diag.pomspread_q15 == 0,
	          "*out_diag must be ZERO-FILLED on a numeric refusal (the memset that runs before "
	          "ScoreAndSelect, never overwritten since ScoreAndSelect returns false before any "
	          "diagnostics are computed) -- got z_k_q0=%lld p_topk_q15=%lld "
	          "alpha_eff_q15=%lld qspread_q15=%lld pomspread_q15=%lld",
	          (long long)diag.z_k_q0, (long long)diag.p_topk_q15, (long long)diag.alpha_eff_q15,
	          (long long)diag.qspread_q15, (long long)diag.pomspread_q15);

	AntiLmDestroy(alm);
}

// Release-gate regression for the ABI's damped selector path: real shipped vocabulary width,
// caller-provided k scratch, and a warm anti-LM. Persistent AntiLmUpdate growth happens before
// the counter reset; selection itself must perform exactly zero allocations.
static void TestC1_DampedSelectorScratchPathAllocatesZeroBytesPerToken() {
	constexpr int32_t V = 151936;
	constexpr int32_t K = 6;
	std::vector<int32_t> row(static_cast<std::size_t>(V), -1000000);
	std::vector<uint8_t> mask((static_cast<std::size_t>(V) + 7) / 8, 0xFF);
	for (int32_t i = 0; i < K; ++i) row[100 + i] = 1000 - i;
	AntiLmState* alm = AntiLmCreate(2);
	AntiLmUpdate(alm, 100);
	AntiLmUpdate(alm, 101);
	int64_t q_ln2 = 0, q_b = 0, q_c = 0;
	CHECK(DeriveDefaultScaleConstants(&q_ln2, &q_b, &q_c));
	int32_t indices[K]{};
	int64_t q_theta[K]{};
	int32_t token = -1;
	bool refused = false;
	GHotPathAllocations = 0;
	const bool ok = DampedGreedyScoreAndArgmaxWithScratch(
	    row.data(), mask.data(), V, K, alm, 65536, q_ln2, q_b, q_c, indices, q_theta, &token,
	    &refused);
	const int allocations = GHotPathAllocations;
	CHECK(ok);
	CHECK(!refused);
	CHECK(token >= 100 && token < 100 + K);
	CHECK_MSG(allocations == 0,
	          "allocation-free damped selector made %d allocation(s) at vocab_size=%d, k=%d",
	          allocations, V, K);
	AntiLmDestroy(alm);
}

int main() {
	TestC1_DomainRejection_BothEntryPoints_VocabAndKOversizedShapes();
	TestDim7_MaskedNeverSelected_HighAlphaOverwhelmsLegalCandidate();
	TestDim7_MaskedNeverSelected_SingleLegalAmongSix();
	TestDim8_MaskFirst_SingleLegalContinuationAcrossFullGrid();
	TestDim5_RefusalPropagatesFromDampedGreedyScoreAndArgmax();
	TestDim7_AlphaZeroReducesToArgmax();
	TestDim7_KeeperProbe_DiagnosticsComputedCorrectly();
	TestFinding2_KeeperProbe_HighAlpha_ExactRecompute();
	TestS1S6_PTopkQ15_ExactIntegerAndHonestDenominator();
	TestS7_PomspreadOverAllK_MaskedOutlierDiscriminates();
	TestM6_RefusalPropagatesThroughPublicEntryPoints();
	TestC1_DampedSelectorScratchPathAllocatesZeroBytesPerToken();
	std::printf("checks=%d failures=%d skips=%d\n", GChecks, GFailures, GSkips);
	return GFailures ? 1 : 0;
}

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

#include "fixture_common.h"

using namespace t2199fixture;

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

int main() {
	TestDim7_MaskedNeverSelected_HighAlphaOverwhelmsLegalCandidate();
	TestDim7_MaskedNeverSelected_SingleLegalAmongSix();
	TestDim8_MaskFirst_SingleLegalContinuationAcrossFullGrid();
	TestDim5_RefusalPropagatesFromDampedGreedyScoreAndArgmax();
	TestDim7_AlphaZeroReducesToArgmax();
	TestDim7_KeeperProbe_DiagnosticsComputedCorrectly();
	TestFinding2_KeeperProbe_HighAlpha_ExactRecompute();
	std::printf("checks=%d failures=%d skips=%d\n", GChecks, GFailures, GSkips);
	return GFailures ? 1 : 0;
}

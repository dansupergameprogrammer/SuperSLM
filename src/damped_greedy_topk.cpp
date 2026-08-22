// damped_greedy_topk.cpp -- T-2199 Phase C: the shared decode-step primitive,
// DampedGreedyScoreAndArgmax (plan Sec7.4-7.6, Sec8 Phase C1/C2/C2a/C3). Design of record:
// Claude/Plans/superslm-1p2-fsd-plan-2026-08-19.md (Wizard repo). Every certified
// sub-primitive this file calls (ShiftByMax, IExpConstruct, IExpEvaluate, SoftmaxRowQ15,
// CheckSoftmaxRowWidthDomain) is reused, never reimplemented.
//
// Fix round 2026-08-20 (Claude/Poirot/7be9508-t2199-phaseAC-review.md, FIX-THEN-SHIP):
// closes C1 (k=0 crash in the Diag entry point), S1 (double on a MEASUREMENT field), S5
// (FsdTopK's own "writes exactly k" claim), M1 (the bool return now carries the domain
// verdict), M2 (the documented SoftmaxRowQ15 gate is now invoked), M3 (GatheredRawIExpSum
// now applies the same per-element M-bound rejection SoftmaxRowQ15 does), and M4 (dead
// initializer documented, AntiLmCreate's own domain closes the sibling crash). O2 (querying
// the anti-LM only for unmasked picks) landed the same day and was REVERTED the day after
// (fold 21, plan commit dbda73ab31): see ScoreAndSelect's own comment below for why.
#include "superslm/sslm_damped_greedy.h"

#include <algorithm>
#include <cstring>
#include <vector>

#include "superslm/checked_chain_funnel.h"
#include "superslm/intmath.h"

namespace superslm {

namespace {

inline bool MaskBitSet(const uint8_t* mask_bits, int32_t idx) {
	return ((mask_bits[static_cast<std::size_t>(idx) >> 3] >> (idx & 7)) & 1u) != 0;
}

// Poirot C1/S5: the domain both public score-and-argmax entry points now enforce
// themselves. The plan sites the general (alpha/n/k) rejection test at Phase D1 (the ABI
// surface, out of this build's scope) -- until that lands, this header is a public surface
// with named external consumers and a caller crossing this boundary gets a `false` return,
// not an access violation or a corrupted candidate set.
inline bool KAndVocabInDomain(int32_t vocab_size, int32_t k) {
	return vocab_size > 0 && k > 0 && k <= vocab_size;
}

}  // namespace

// T-2199 Phase D closing-round residue, N2 (Claude/Poirot/a12bbdd-t2199-phaseD-closing.md): the
// single shared validator declared in sslm_damped_greedy.h -- see that declaration's own comment
// for the full reasoning. Exported (not in the anonymous namespace above, unlike
// KAndVocabInDomain) specifically so superslm::ValidateDampedGreedyParams (Phase D,
// damped_greedy_phaseD.cpp, a different translation unit) can call the SAME check rather than
// re-deriving the same [0, 2^20) bound a second time.
bool AlphaQ15InDomain(int64_t alpha_q15) noexcept {
	return alpha_q15 >= 0 && alpha_q15 < (int64_t{1} << 20);
}

// Phase C1 (Sec7.4).
//
// Domain: 1 <= k <= vocab_size. This function has no channel to report a domain violation
// (void return, matching the suite's own declared interface) -- `DampedGreedyScoreAndArgmax`
// and `DampedGreedyScoreAndArgmaxDiag` enforce the domain before ever calling this, and are
// the intended callers. A DIRECT caller that passes k > vocab_size still gets a defined
// result: exactly `min(k, vocab_size)` real vocabulary indices, most-favoured first, and
// every slot from `vocab_size` to `k-1` is written as -1 (never left uninitialized, never a
// valid vocabulary index, so it can never be silently mistaken for a real candidate
// downstream) -- correcting the prior "writes exactly k" claim, which was false whenever
// k > vocab_size (Poirot S5: the trailing slots were left untouched, a read of
// uninitialized memory for a direct caller).
void FsdTopK(const int32_t* masked_row, const uint8_t* mask_bits, int32_t vocab_size,
             int32_t k, int32_t* out_indices) {
	if (k <= 0 || vocab_size <= 0) return;
	const int32_t kk = std::min(k, vocab_size);
	const auto compare = [&](int32_t a, int32_t b) {
		if (masked_row[a] != masked_row[b]) return masked_row[a] > masked_row[b];
		const bool legal_a = MaskBitSet(mask_bits, a);
		const bool legal_b = MaskBitSet(mask_bits, b);
		if (legal_a != legal_b) return legal_a;  // unmasked outranks masked at equal value
		return a < b;
	};
	for (int32_t i = 0; i < kk; ++i) out_indices[i] = i;
	// A caller-owned k-element heap keeps its worst retained candidate at the root. This is
	// O(V log k), O(1) auxiliary storage, and avoids the former full-vocabulary allocation and
	// partial_sort. The final k-element sort restores the exact published candidate ordering.
	std::make_heap(out_indices, out_indices + kk, compare);
	for (int32_t candidate = kk; candidate < vocab_size; ++candidate) {
		if (!compare(candidate, out_indices[0])) continue;
		std::pop_heap(out_indices, out_indices + kk, compare);
		out_indices[kk - 1] = candidate;
		std::push_heap(out_indices, out_indices + kk, compare);
	}
	std::sort(out_indices, out_indices + kk, compare);
	for (int32_t i = kk; i < k; ++i) out_indices[i] = -1;  // k > vocab_size: defined, never UB
}

// Phase C2 (Sec7.3 surface 2, Sec5.5).
bool TopKRenormalizeQ15(const int32_t* row, const int32_t* indices, std::size_t k,
                         int64_t q_ln2, int64_t q_b, int64_t q_c, int64_t* out_q15) {
	if (k == 0) return true;
	// Poirot M2: `intmath.h` states twice that a caller gates this kernel with
	// `CheckSoftmaxRowWidthDomain(q_b, q_c, width)` before calling it -- the per-element
	// bound `SoftmaxRowQ15` enforces holds unconditionally either way, but the sum-overflow
	// bound the gate proves does not. Nothing was unsafe at this call site's own k (the sum
	// `k * kSoftmaxRowMaxSafeExponent` cannot overflow at any k this design's own grid
	// uses), but the documented contract is now honored explicitly rather than left as an
	// unenforced prose promise. On a gate failure, every output is written 0 (matching
	// `SoftmaxRowQ15`'s own "out_q15[i] is written for every element regardless of the
	// returned bool" contract) and this function returns false, exactly as a per-element
	// refusal from the kernel itself would.
	if (CheckSoftmaxRowWidthDomain(q_b, q_c, k) != SslmForwardStatus::Ok) {
		for (std::size_t i = 0; i < k; ++i) out_q15[i] = 0;
		return false;
	}
	int64_t row_max = row[indices[0]];
	for (std::size_t i = 1; i < k; ++i) row_max = std::max(row_max, int64_t{row[indices[i]]});
	IExpConstruction peak_construction;
	const IExpDomain peak_domain = IExpConstruct(0, q_ln2, q_b, q_c, &peak_construction);
	const bool peak_constructed = peak_domain != IExpDomain::kBadQ &&
	                              peak_domain != IExpDomain::kBadQLn2 &&
	                              peak_domain != IExpDomain::kBadQB;
	const int64_t peak = peak_constructed ? IExpEvaluate(peak_construction) : 0;
	const bool peak_usable = peak >= 1 && peak <= kSoftmaxRowMaxSafeExponent;
	int64_t total = 0;
	bool all_well_formed = peak_usable;
	for (std::size_t i = 0; i < k; ++i) {
		IExpConstruction construction;
		const int64_t shifted = static_cast<int64_t>(row[indices[i]]) - row_max;
		const IExpDomain domain = IExpConstruct(shifted, q_ln2, q_b, q_c, &construction);
		if (domain == IExpDomain::kBadQ || domain == IExpDomain::kBadQLn2 ||
		    domain == IExpDomain::kBadQB) {
			all_well_formed = false;
			out_q15[i] = 0;
			continue;
		}
		const int64_t value = IExpEvaluate(construction);
		if (!peak_usable || value < 0 || value > peak) {
			all_well_formed = false;
			out_q15[i] = 0;
			continue;
		}
		out_q15[i] = value;
		total += value;
	}
	const int64_t denom = total > 1 ? total : 1;
	for (std::size_t i = 0; i < k; ++i) {
		out_q15[i] = (out_q15[i] << kProbFracBits) / denom;
	}
	return all_well_formed;
}

namespace {

// The k gathered candidates' own raw (pre-Q15-divide) i-exp sum -- the same ShiftByMax
// reference SoftmaxRowQ15/TopKRenormalizeQ15 use internally. Since FsdTopK always selects
// the row's true global max among its k picks (that is what "top-k" means), ShiftByMax over
// just these k elements uses the identical reference point a full-row ShiftByMax would, so
// this raw sum is on the same scale as a genuine full-row raw total -- a diagnostic
// quantity for DampedGreedyScoreAndArgmaxDiag's own MEASUREMENT-classed keeper-probe
// fields, never used on the decode decision itself (DampedGreedyScoreAndArgmax, the
// production entry point, never calls this).
//
// Poirot M3: this duplicates SoftmaxRowQ15's own accumulation loop. Eliminating the
// duplicate would mean either exposing SoftmaxRowQ15's private internal total (touching a
// certified sub-primitive's own header/implementation, out of this build seat's remit -- no
// plan section specifies that change) or adding a second public entry point to the
// suite-pinned interface (which this build cannot do -- the interface's signatures are
// fixed by the red suite). Kept as a diagnostic-only duplicate, DISPOSITIONED rather than
// eliminated; what this fix round closes is the "weaker" half of the finding: this loop now
// applies the SAME per-element [0, M] bound rejection SoftmaxRowQ15 does (previously
// redundant only because this function is never reached except behind
// TopKRenormalizeQ15's own wf == true, which already guarantees every element here is
// in-bound -- this makes that guarantee load-bearing in the code, not merely in a caller's
// argument, so a future relaxation of the wf==true precondition cannot silently reintroduce
// an unbounded accumulation).
int64_t GatheredRawIExpSum(const int32_t* row, const int32_t* indices, std::size_t k,
                            int64_t q_ln2, int64_t q_b, int64_t q_c) {
	std::vector<int64_t> scores(k);
	for (std::size_t i = 0; i < k; ++i) scores[i] = static_cast<int64_t>(row[indices[i]]);
	std::vector<int64_t> shifted(k);
	ShiftByMax(scores.data(), k, shifted.data());

	// The same M = q_b*q_b + q_c per-element ceiling SoftmaxRowQ15 forms and judges
	// (intmath.cpp), recomputed here without a 128-bit type (this repo's own MSVC toolchain
	// carries no compiler `__int128`, which is exactly why SoftmaxRowQ15 reaches for its own
	// struct-based S128 rather than a compiler extension -- replicated in that same portable
	// spirit, via division rather than forming a product that could overflow):
	// `kSoftmaxRowMaxSafeExponent` (2^47) is the ceiling M must clear to be usable at all, so
	// `q_b*q_b` alone must not exceed it, checked by dividing rather than squaring first.
	const int64_t abs_qb = q_b < 0 ? -q_b : q_b;
	bool m_usable = q_c >= 0 && q_c <= kSoftmaxRowMaxSafeExponent &&
	                (abs_qb == 0 || abs_qb <= (kSoftmaxRowMaxSafeExponent / abs_qb));
	int64_t m = 0;
	if (m_usable) {
		const int64_t qb2 = abs_qb * abs_qb;
		m_usable = qb2 <= kSoftmaxRowMaxSafeExponent - q_c;
		if (m_usable) m = qb2 + q_c;
	}

	int64_t total = 0;
	for (std::size_t i = 0; i < k; ++i) {
		IExpConstruction construction;
		const IExpDomain d = IExpConstruct(shifted[i], q_ln2, q_b, q_c, &construction);
		if (d == IExpDomain::kBadQ || d == IExpDomain::kBadQLn2 || d == IExpDomain::kBadQB) continue;
		const int64_t value = IExpEvaluate(construction);
		if (!m_usable || value < 0 || value > m) continue;
		total += value;
	}
	return total;
}

// Poirot S1: exact-integer replacement for the prior `double` computation of
// `p_topk_q15 = round(z_k_raw * 2^15 / full_row_z)`, clamped to [0, 1<<15]. Plan Sec1
// constraint 1: "No float on the decode path; any new arithmetic is exact integer." This
// codebase already carries an exact 128-bit facility for exactly this shape (S128/SMul/
// SShrToI64, src/intmath.cpp), file-local and duplicated a second time in
// src/forward/checked_chain_funnel.cpp -- replicated here as a third, minimal, purpose-built
// copy rather than promoted out of either certified translation unit (Poirot S1: "promote
// or replicate per repo convention"; replicated, so as to touch neither certified file for
// one diagnostic field).
//
// z_k_raw and full_row_z are both non-negative by this call site's own contract (see the
// caller). The clamp-to-ceiling case (the true ratio >= 1.0) is detected WITHOUT forming
// the shifted product at all -- z_k_raw*2^15 >= full_row_z*2^15 iff z_k_raw >= full_row_z,
// since 2^15 cancels -- so only the sub-ceiling case ever needs the widened divide, and in
// that case the quotient is provably < 1<<15, which is what makes a 64-bit quotient exact
// here despite the (possibly 128-bit-wide) dividend.
uint64_t DivU128ByU64(uint64_t num_hi, uint64_t num_lo, uint64_t divisor, uint64_t* out_rem) {
	uint64_t quotient = 0;
	uint64_t remainder = 0;
	for (int bit = 127; bit >= 0; --bit) {
		const uint64_t bit_val =
		    (bit >= 64) ? ((num_hi >> (bit - 64)) & 1u) : ((num_lo >> bit) & 1u);
		remainder = (remainder << 1) | bit_val;
		if (remainder >= divisor) {
			remainder -= divisor;
			if (bit < 64) quotient |= (uint64_t{1} << bit);
		}
	}
	if (out_rem) *out_rem = remainder;
	return quotient;
}

int64_t ExactRatioQ15(int64_t z_k_raw, int64_t full_row_z) {
	const uint64_t denom = (full_row_z > 0) ? static_cast<uint64_t>(full_row_z) : 1;
	const uint64_t zk = (z_k_raw > 0) ? static_cast<uint64_t>(z_k_raw) : 0;
	if (zk >= denom) return int64_t{1} << kProbFracBits;  // true ratio >= 1.0 -- clamp
	const uint64_t num_lo = zk << kProbFracBits;
	const uint64_t num_hi = zk >> (64 - kProbFracBits);  // bits shifted out of num_lo
	uint64_t remainder = 0;
	uint64_t quotient = DivU128ByU64(num_hi, num_lo, denom, &remainder);
	// Round to nearest, ties away from zero (this codebase's own rounding convention --
	// RoundingDivideByPOT, src/intmath.cpp).
	if (remainder * 2 >= denom) ++quotient;
	if (quotient > (uint64_t{1} << kProbFracBits)) quotient = uint64_t{1} << kProbFracBits;
	return static_cast<int64_t>(quotient);
}

// Shared score/select body for both public entry points (Sec7.5-7.6): FsdTopK -> gate ->
// TopKRenormalizeQ15 -> AntiLmPenalize (ALL k gathered picks, uniformly) -> per-candidate
// score -> argmax. Returns false on a TopKRenormalizeQ15 refusal.
//
// REVERTED 2026-08-20 (fold 21, plan commit dbda73ab31, S7/S8 of
// `Claude/Poirot/927bbda-t2199-confirmation.md`): the fix round's own O2 change (query only
// the unmasked picks) is undone. That change made `DampedGreedyDiagnostics.pomspread_q15`
// read a narrower, unmasked-only spread under a bound schema (measured: 3,276 where the true
// all-k spread was 29,491; 16,384 where it was 0) while its own header still documented
// "over k" -- and pomspread is an operand of MODEL-GOVERNS, whose own worst-case margin is
// already 0.70x (Sec5.6.2). The ruling removes the term rather than adding an unmasked/all-k
// split to an already-thin criterion: `AntiLmPenalize` is queried for every one of the k
// gathered picks again, matching plan Sec7.5's own "a masked pick's own q_theta/p_omega reach
// IExpConstruct and the anti-LM lookup exactly like an unmasked pick's" (fold 20, the design
// of record). The score loop's own mask-floor branch below is unaffected: a masked pick's
// score is still INT64_MIN regardless of its now-real p_omega. Cost: AntiLmPenalize's own
// per-pick lookup is already inside plan Sec5.5's ~0.02-0.2%-of-the-head-GEMM figure for the
// whole k-wide pass, so this restores a cost the design was priced against from the start,
// not a new one.
struct ScoredCandidates {
	std::vector<int32_t> idx;
	std::vector<int64_t> q_theta;
	std::vector<int64_t> p_omega;  // real for every pick, masked and unmasked alike
	int32_t best_token = -1;
	int winner_pos = -1;
};

bool ScoreAndSelect(const int32_t* masked_row, const uint8_t* mask_bits, int32_t vocab_size,
                     int32_t k, const AntiLmState* anti_lm, int64_t alpha_q15, int64_t q_ln2,
                     int64_t q_b, int64_t q_c, ScoredCandidates* out) {
	out->idx.assign(static_cast<std::size_t>(k), -1);
	FsdTopK(masked_row, mask_bits, vocab_size, k, out->idx.data());

	out->q_theta.assign(static_cast<std::size_t>(k), 0);
	if (!TopKRenormalizeQ15(masked_row, out->idx.data(), static_cast<std::size_t>(k), q_ln2, q_b,
	                         q_c, out->q_theta.data())) {
		return false;
	}

	out->p_omega.assign(static_cast<std::size_t>(k), 0);
	AntiLmPenalize(anti_lm, out->idx.data(), static_cast<std::size_t>(k), out->p_omega.data());

	out->best_token = -1;
	out->winner_pos = -1;
	int64_t best_score = 0;  // never read before `have_best` is first set true, below
	bool have_best = false;
	for (int32_t i = 0; i < k; ++i) {
		const std::size_t si = static_cast<std::size_t>(i);
		int64_t score;
		if (MaskBitSet(mask_bits, out->idx[si])) {
			// T-2199 Phase D review fix O2 (Claude/Poirot/7a3b10a-t2199-phaseD-review.md): the
			// plan's own formula (Sec7.5) narrows alpha_eff through static_cast<int32_t>
			// before the subtraction -- omitted here previously (kept int64_t throughout).
			// Value-preserving, not merely spec-matching: C2's fix bounds alpha_q15 to
			// [0, 2^20) and p_omega to [0, 2^15] by construction (Sec7.2), so alpha_eff's own
			// true range is [0, 2^20) -- always representable in int32_t, so this cast changes
			// no computed value under the now-corrected domain, only makes the width the
			// design specifies explicit in code.
			const int32_t alpha_eff = static_cast<int32_t>(
			    (static_cast<int64_t>(alpha_q15) * out->p_omega[si]) >> kProbFracBits);
			score = out->q_theta[si] - alpha_eff;
		} else {
			score = INT64_MIN;
		}
		if (!have_best || score > best_score || (score == best_score && out->idx[si] < out->best_token)) {
			best_score = score;
			out->best_token = out->idx[si];
			out->winner_pos = i;
			have_best = true;
		}
	}
	return true;
}

bool ScoreAndSelectWithScratch(const int32_t* masked_row, const uint8_t* mask_bits,
                               int32_t vocab_size, int32_t k, const AntiLmState* anti_lm,
                               int64_t alpha_q15, int64_t q_ln2, int64_t q_b, int64_t q_c,
                               int32_t* indices, int64_t* q_theta, int32_t* out_token) {
	FsdTopK(masked_row, mask_bits, vocab_size, k, indices);
	if (!TopKRenormalizeQ15(masked_row, indices, static_cast<std::size_t>(k), q_ln2, q_b, q_c,
	                         q_theta)) {
		return false;
	}
	bool have_best = false;
	int64_t best_score = 0;
	int32_t best_token = -1;
	for (int32_t i = 0; i < k; ++i) {
		int64_t p_omega = 0;
		AntiLmPenalize(anti_lm, &indices[i], 1, &p_omega);
		const int64_t score = MaskBitSet(mask_bits, indices[i])
		                          ? q_theta[i] - static_cast<int32_t>(
		                                             (alpha_q15 * p_omega) >> kProbFracBits)
		                          : INT64_MIN;
		if (!have_best || score > best_score || (score == best_score && indices[i] < best_token)) {
			have_best = true;
			best_score = score;
			best_token = indices[i];
		}
	}
	*out_token = best_token;
	return true;
}

}  // namespace

// Phase C3 (Sec7.5-7.6).
//
// Domain: 1 <= k <= vocab_size (Poirot C1/S5). A domain violation returns false and leaves
// *out_token/*out_refused untouched -- distinct from *out_refused, which signals a
// TopKRenormalizeQ15 numeric refusal on an otherwise well-formed call (return true,
// *out_refused = true). The general (alpha/n/k) rejection Phase D1 will site at the ABI
// boundary supersedes this once built; until then this is the only domain guard between a
// caller and undefined behaviour.
bool DampedGreedyScoreAndArgmax(const int32_t* masked_row, const uint8_t* mask_bits,
                                 int32_t vocab_size, int32_t k, const AntiLmState* anti_lm,
                                 int64_t alpha_q15, int64_t q_ln2, int64_t q_b, int64_t q_c,
                                 int32_t* out_token, bool* out_refused) {
	if (!KAndVocabInDomain(vocab_size, k)) return false;
	// N2 fix: checked BEFORE ScoreAndSelect's own cast ever runs -- makes the sign-inversion at
	// alpha_q15 >= 2^31 unreachable from this public entry point.
	if (!AlphaQ15InDomain(alpha_q15)) return false;

	ScoredCandidates sc;
	if (!ScoreAndSelect(masked_row, mask_bits, vocab_size, k, anti_lm, alpha_q15, q_ln2, q_b, q_c,
	                     &sc)) {
		*out_refused = true;
		return true;
	}
	*out_refused = false;
	*out_token = sc.best_token;
	return true;
}

bool DampedGreedyScoreAndArgmaxWithScratch(
    const int32_t* masked_row, const uint8_t* mask_bits, int32_t vocab_size, int32_t k,
    const AntiLmState* anti_lm, int64_t alpha_q15, int64_t q_ln2, int64_t q_b, int64_t q_c,
    int32_t* indices_scratch, int64_t* q_theta_scratch, int32_t* out_token,
    bool* out_refused) {
	if (!KAndVocabInDomain(vocab_size, k) || !AlphaQ15InDomain(alpha_q15)) return false;
	if (!ScoreAndSelectWithScratch(masked_row, mask_bits, vocab_size, k, anti_lm, alpha_q15,
	                               q_ln2, q_b, q_c, indices_scratch, q_theta_scratch,
	                               out_token)) {
		*out_refused = true;
		return true;
	}
	*out_refused = false;
	return true;
}

// Same domain and return-value contract as DampedGreedyScoreAndArgmax above; on a domain
// violation *out_diag is also left untouched (the memset below never runs).
bool DampedGreedyScoreAndArgmaxDiag(const int32_t* masked_row, const uint8_t* mask_bits,
                                     int32_t vocab_size, int32_t k, const AntiLmState* anti_lm,
                                     int64_t alpha_q15, int64_t q_ln2, int64_t q_b, int64_t q_c,
                                     int64_t full_row_z, int32_t* out_token, bool* out_refused,
                                     DampedGreedyDiagnostics* out_diag) {
	if (!KAndVocabInDomain(vocab_size, k)) return false;
	// N2 fix: same shared check as DampedGreedyScoreAndArgmax above.
	if (!AlphaQ15InDomain(alpha_q15)) return false;

	std::memset(out_diag, 0, sizeof(*out_diag));

	ScoredCandidates sc;
	if (!ScoreAndSelect(masked_row, mask_bits, vocab_size, k, anti_lm, alpha_q15, q_ln2, q_b, q_c,
	                     &sc)) {
		*out_refused = true;
		return true;
	}
	*out_refused = false;
	*out_token = sc.best_token;

	// Diagnostics -- MEASUREMENT-classed (Sec9 dim7 preamble, D-SLM3727): computed correctly
	// against the actual k-gathered candidates, never asserted to "read a particular way."
	int64_t qmax = sc.q_theta[0], qmin = sc.q_theta[0];
	int64_t pmax = sc.p_omega[0], pmin = sc.p_omega[0];
	for (int32_t i = 1; i < k; ++i) {
		const std::size_t si = static_cast<std::size_t>(i);
		qmax = std::max(qmax, sc.q_theta[si]);
		qmin = std::min(qmin, sc.q_theta[si]);
		pmax = std::max(pmax, sc.p_omega[si]);
		pmin = std::min(pmin, sc.p_omega[si]);
	}
	out_diag->qspread_q15 = qmax - qmin;
	out_diag->pomspread_q15 = pmax - pmin;

	const int64_t winner_p_omega =
	    (sc.winner_pos >= 0) ? sc.p_omega[static_cast<std::size_t>(sc.winner_pos)] : 0;
	out_diag->alpha_eff_q15 = (static_cast<int64_t>(alpha_q15) * winner_p_omega) >> kProbFracBits;

	const int64_t z_k_raw = GatheredRawIExpSum(masked_row, sc.idx.data(),
	                                            static_cast<std::size_t>(k), q_ln2, q_b, q_c);
	out_diag->z_k_q0 = z_k_raw;

	// p_topk_q15 = z_k_q0 / full_row_z, in Q15, CLAMPED to [0, 1<<15] -- exact integer
	// (Poirot S1, ExactRatioQ15 above). The clamp's stated purpose (build log Sec8, Poirot
	// S6): full_row_z is caller-supplied and this function has no way to verify it carries
	// the true full-row total at the same scale as z_k_q0, so a caller-side underestimate
	// (yielding a raw ratio >= 1.0) is reported as "the k-gathered candidates hold
	// effectively all of the row's mass" rather than propagated past a probability's own
	// domain. Poirot S6's own second executed row (full_row_z = 4*z_k -> p_topk_q15 = 8192,
	// exactly 0.25) proves this arithmetic is correct below the ceiling; the still-owed item
	// is a COMMITTED fixture exercising that case (routed to Curie, build log Sec8 caveat),
	// not a code change here.
	out_diag->p_topk_q15 = ExactRatioQ15(z_k_raw, full_row_z);

	return true;
}

}  // namespace superslm

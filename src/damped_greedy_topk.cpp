// damped_greedy_topk.cpp -- T-2199 Phase C: the shared decode-step primitive,
// DampedGreedyScoreAndArgmax (plan Sec7.4-7.6, Sec8 Phase C1/C2/C2a/C3). Design of record:
// Claude/Plans/superslm-1p2-fsd-plan-2026-08-19.md (Wizard repo). Every certified
// sub-primitive this file calls (ShiftByMax, IExpConstruct, IExpEvaluate, SoftmaxRowQ15) is
// reused from intmath.h, never reimplemented.
#include "superslm/sslm_damped_greedy.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

#include "superslm/intmath.h"

namespace superslm {

namespace {

inline bool MaskBitSet(const uint8_t* mask_bits, int32_t idx) {
	return ((mask_bits[static_cast<std::size_t>(idx) >> 3] >> (idx & 7)) & 1u) != 0;
}

}  // namespace

// Phase C1 (Sec7.4).
void FsdTopK(const int32_t* masked_row, const uint8_t* mask_bits, int32_t vocab_size,
             int32_t k, int32_t* out_indices) {
	std::vector<int32_t> idx(static_cast<std::size_t>(vocab_size));
	for (int32_t i = 0; i < vocab_size; ++i) idx[static_cast<std::size_t>(i)] = i;
	const int32_t kk = std::min(k, vocab_size);
	const auto compare = [&](int32_t a, int32_t b) {
		if (masked_row[a] != masked_row[b]) return masked_row[a] > masked_row[b];
		const bool legal_a = MaskBitSet(mask_bits, a);
		const bool legal_b = MaskBitSet(mask_bits, b);
		if (legal_a != legal_b) return legal_a;  // unmasked outranks masked at equal value
		return a < b;
	};
	if (kk < vocab_size) {
		std::partial_sort(idx.begin(), idx.begin() + kk, idx.end(), compare);
	} else {
		std::sort(idx.begin(), idx.end(), compare);
	}
	for (int32_t i = 0; i < kk; ++i) out_indices[i] = idx[static_cast<std::size_t>(i)];
}

// Phase C2 (Sec7.3 surface 2, Sec5.5).
bool TopKRenormalizeQ15(const int32_t* row, const int32_t* indices, std::size_t k,
                         int64_t q_ln2, int64_t q_b, int64_t q_c, int64_t* out_q15) {
	if (k == 0) return true;
	std::vector<int64_t> scores(k);
	for (std::size_t i = 0; i < k; ++i) scores[i] = static_cast<int64_t>(row[indices[i]]);
	// "Built entirely from the same certified sub-primitives SoftmaxRowQ15 already uses,
	// restricted to k top-k positions" (plan Sec7.3) -- SoftmaxRowQ15 IS that composition
	// (ShiftByMax -> IExpConstruct/IExpEvaluate -> Q15 divide); calling it directly at width
	// k realizes the restriction exactly, with no separate implementation to drift from it.
	return SoftmaxRowQ15(scores.data(), k, q_ln2, q_b, q_c, out_q15);
}

namespace {

// The k gathered candidates' own raw (pre-Q15-divide) i-exp sum -- the same ShiftByMax
// reference SoftmaxRowQ15/TopKRenormalizeQ15 use internally. Since FsdTopK always selects
// the row's true global max among its k picks (that is what "top-k" means), ShiftByMax over
// just these k elements uses the identical reference point a full-row ShiftByMax would, so
// this raw sum is on the same scale as a genuine full-row raw total -- a diagnostic
// quantity, not a second certified kernel (the certified per-element domain refusal that
// TopKRenormalizeQ15/SoftmaxRowQ15 apply is not reproduced here; this helper is read only
// for DampedGreedyScoreAndArgmaxDiag's own MEASUREMENT-classed keeper-probe fields, never
// for the decode decision itself).
int64_t GatheredRawIExpSum(const int32_t* row, const int32_t* indices, std::size_t k,
                            int64_t q_ln2, int64_t q_b, int64_t q_c) {
	std::vector<int64_t> scores(k);
	for (std::size_t i = 0; i < k; ++i) scores[i] = static_cast<int64_t>(row[indices[i]]);
	std::vector<int64_t> shifted(k);
	ShiftByMax(scores.data(), k, shifted.data());
	int64_t total = 0;
	for (std::size_t i = 0; i < k; ++i) {
		IExpConstruction construction;
		const IExpDomain d = IExpConstruct(shifted[i], q_ln2, q_b, q_c, &construction);
		if (d == IExpDomain::kBadQ || d == IExpDomain::kBadQLn2 || d == IExpDomain::kBadQB) continue;
		const int64_t value = IExpEvaluate(construction);
		if (value < 0) continue;
		total += value;
	}
	return total;
}

}  // namespace

// Phase C3 (Sec7.5-7.6).
bool DampedGreedyScoreAndArgmax(const int32_t* masked_row, const uint8_t* mask_bits,
                                 int32_t vocab_size, int32_t k, const AntiLmState* anti_lm,
                                 int64_t alpha_q15, int64_t q_ln2, int64_t q_b, int64_t q_c,
                                 int32_t* out_token, bool* out_refused) {
	std::vector<int32_t> idx(static_cast<std::size_t>(k));
	FsdTopK(masked_row, mask_bits, vocab_size, k, idx.data());

	std::vector<int64_t> q_theta(static_cast<std::size_t>(k));
	const bool wf = TopKRenormalizeQ15(masked_row, idx.data(), static_cast<std::size_t>(k), q_ln2,
	                                    q_b, q_c, q_theta.data());
	if (!wf) {
		*out_refused = true;
		return true;
	}
	*out_refused = false;

	std::vector<int64_t> p_omega(static_cast<std::size_t>(k));
	AntiLmPenalize(anti_lm, idx.data(), static_cast<std::size_t>(k), p_omega.data());

	int32_t best_token = -1;
	int64_t best_score = 0;
	bool have_best = false;
	for (int32_t i = 0; i < k; ++i) {
		int64_t score;
		if (MaskBitSet(mask_bits, idx[static_cast<std::size_t>(i)])) {
			const int64_t alpha_eff =
			    (static_cast<int64_t>(alpha_q15) * p_omega[static_cast<std::size_t>(i)]) >>
			    kProbFracBits;
			score = static_cast<int64_t>(q_theta[static_cast<std::size_t>(i)]) - alpha_eff;
		} else {
			score = INT64_MIN;
		}
		if (!have_best || score > best_score ||
		    (score == best_score && idx[static_cast<std::size_t>(i)] < best_token)) {
			best_score = score;
			best_token = idx[static_cast<std::size_t>(i)];
			have_best = true;
		}
	}
	*out_token = best_token;
	return true;
}

bool DampedGreedyScoreAndArgmaxDiag(const int32_t* masked_row, const uint8_t* mask_bits,
                                     int32_t vocab_size, int32_t k, const AntiLmState* anti_lm,
                                     int64_t alpha_q15, int64_t q_ln2, int64_t q_b, int64_t q_c,
                                     int64_t full_row_z, int32_t* out_token, bool* out_refused,
                                     DampedGreedyDiagnostics* out_diag) {
	std::memset(out_diag, 0, sizeof(*out_diag));

	std::vector<int32_t> idx(static_cast<std::size_t>(k));
	FsdTopK(masked_row, mask_bits, vocab_size, k, idx.data());

	std::vector<int64_t> q_theta(static_cast<std::size_t>(k));
	const bool wf = TopKRenormalizeQ15(masked_row, idx.data(), static_cast<std::size_t>(k), q_ln2,
	                                    q_b, q_c, q_theta.data());
	if (!wf) {
		*out_refused = true;
		return true;
	}
	*out_refused = false;

	std::vector<int64_t> p_omega(static_cast<std::size_t>(k));
	AntiLmPenalize(anti_lm, idx.data(), static_cast<std::size_t>(k), p_omega.data());

	int32_t best_token = -1;
	int64_t best_score = 0;
	int winner_pos = -1;
	bool have_best = false;
	for (int32_t i = 0; i < k; ++i) {
		int64_t score;
		if (MaskBitSet(mask_bits, idx[static_cast<std::size_t>(i)])) {
			const int64_t alpha_eff =
			    (static_cast<int64_t>(alpha_q15) * p_omega[static_cast<std::size_t>(i)]) >>
			    kProbFracBits;
			score = static_cast<int64_t>(q_theta[static_cast<std::size_t>(i)]) - alpha_eff;
		} else {
			score = INT64_MIN;
		}
		if (!have_best || score > best_score ||
		    (score == best_score && idx[static_cast<std::size_t>(i)] < best_token)) {
			best_score = score;
			best_token = idx[static_cast<std::size_t>(i)];
			winner_pos = i;
			have_best = true;
		}
	}
	*out_token = best_token;

	// Diagnostics -- MEASUREMENT-classed (Sec9 dim7 preamble, D-SLM3727): computed correctly
	// against the actual k-gathered candidates, never asserted to "read a particular way."
	int64_t qmax = q_theta[0], qmin = q_theta[0];
	int64_t pmax = p_omega[0], pmin = p_omega[0];
	for (int32_t i = 1; i < k; ++i) {
		qmax = std::max(qmax, q_theta[static_cast<std::size_t>(i)]);
		qmin = std::min(qmin, q_theta[static_cast<std::size_t>(i)]);
		pmax = std::max(pmax, p_omega[static_cast<std::size_t>(i)]);
		pmin = std::min(pmin, p_omega[static_cast<std::size_t>(i)]);
	}
	out_diag->qspread_q15 = qmax - qmin;
	out_diag->pomspread_q15 = pmax - pmin;

	const int64_t winner_p_omega =
	    (winner_pos >= 0) ? p_omega[static_cast<std::size_t>(winner_pos)] : 0;
	out_diag->alpha_eff_q15 = (static_cast<int64_t>(alpha_q15) * winner_p_omega) >> kProbFracBits;

	const int64_t z_k_raw = GatheredRawIExpSum(masked_row, idx.data(), static_cast<std::size_t>(k),
	                                            q_ln2, q_b, q_c);
	out_diag->z_k_q0 = z_k_raw;

	// p_topk_q15 = z_k_q0 / full_row_z, in Q15 -- computed in double precision (this is a
	// MEASUREMENT field, not bit-pinned) to stay overflow-safe across the full raw-magnitude
	// range IExpEvaluate can produce, then CLAMPED to a valid probability's own [0, 1<<15]
	// range: full_row_z is supplied by the caller and this function has no way to verify it
	// carries the true full-row total at the same scale as z_k_q0, so a caller-side
	// underestimate of the true total (yielding a raw ratio above 1.0) is reported as "the
	// k-gathered candidates hold effectively all of the row's mass" rather than propagated
	// as a value outside a probability's own domain.
	const int64_t denom = (full_row_z > 0) ? full_row_z : 1;
	const double ratio = static_cast<double>(z_k_raw) *
	                      static_cast<double>(int64_t{1} << kProbFracBits) /
	                      static_cast<double>(denom);
	int64_t p_topk = static_cast<int64_t>(std::llround(ratio));
	if (p_topk < 0) p_topk = 0;
	if (p_topk > (int64_t{1} << kProbFracBits)) p_topk = int64_t{1} << kProbFracBits;
	out_diag->p_topk_q15 = p_topk;

	return true;
}

}  // namespace superslm

// T-2199 (Curie) -- shared Phase D fixture helper: derives a real, in-domain (q_ln2, q_b, q_c)
// triple via the certified IExpScaleConstants (intmath.h), targeting the plan's own default
// scale (q_ln2 = 493, Sec5.6.2). Factored out 2026-08-20 (conductor's dispute-resolution
// commission, dispute 3): three cells (TestD2_TokenDigest_CoversDampedGreedyTokens,
// phaseD2_wiring_red.cpp; the sole cell in phaseD2a_cost_ratio_red.cpp; the sole cell in
// phaseD3_teardown_red.cpp) each set params.q_ln2 = 493 alone and left q_b/q_c at their
// zero-init default -- (q_b=0, q_c=0) derives M = q_b^2 + q_c = 0, which fails
// CheckSoftmaxRowWidthDomain's own M >= 1 requirement, so every one of those cells' own
// sslm_decode_step_damped_greedy calls correctly REFUSED (matching plan Sec7.5's own adopted
// policy) rather than running to completion -- a fixture defect, not a code defect, per the
// builder's own build log (Claude/Brunel/t2199-phaseD-build-2026-08-20.md Sec4). The ONE
// sub-cell that got this right (TestD2_DampedGreedyMode_ProducesPrimitiveExactOutputThroughDecodeStep)
// derived all three together via this exact search; this header lifts that pattern out once so
// the three broken cells (and any future one) call it instead of re-deriving inline.
#ifndef SSLM_T2199_PHASED_FIXTURE_H
#define SSLM_T2199_PHASED_FIXTURE_H

#include "superslm/intmath.h"

namespace t2199phaseD {

// Searches for the plan's own corrected starting scale (Sec5.6.2, q_ln2 = 493 -- well inside
// the faithful calibration band Sec9 dim7's own alpha=0-identity cell cites, q_ln2 <= 7,893),
// matching this suite's own Phase A/C fixture_common.h::DeriveDefaultScaleConstants recipe.
// Returns false (leaves outputs untouched) only if the search space is exhausted without
// finding target_q_ln2 -- callers CHECK the return, never assume success silently.
inline bool DeriveDefaultScaleConstants(int64_t* out_q_ln2, int64_t* out_q_b, int64_t* out_q_c) {
	using namespace superslm;
	constexpr int64_t kTargetQLn2 = 493;
	for (int64_t e = -80; e <= 0; ++e) {
		for (int64_t m = (int64_t{1} << 30); m < (int64_t{1} << 31); m += (int64_t{1} << 20)) {
			int64_t a = 0, b = 0, c = 0;
			if (IExpScaleConstants(m, e, kIExpLn2Q, 30, kIExpBQ, 30, kIExpCaQ, 30, &a, &b, &c) !=
			    IExpScaleDomain::kOk)
				continue;
			if (a == kTargetQLn2) {
				*out_q_ln2 = a;
				*out_q_b = b;
				*out_q_c = c;
				return true;
			}
		}
	}
	return false;
}

// Same search, but returns the SOURCE (m, e) pair itself rather than the derived
// (q_ln2, q_b, q_c) -- for a cell that needs to round-trip the artifact-carried raw form
// (Phase B1's own (m, e), plan Sec8 B1) through SslmArtifact rather than the runtime-derived
// triple Phase B3 computes FROM it. Guarantees the returned (m, e) is genuinely in-domain
// (ReadDampedGreedyScaleConstants's own ScaleConstantsInDomain check, src/damped_greedy_
// phaseD.cpp, re-derives via this SAME IExpScaleConstants call and would otherwise reject an
// arbitrary hand-picked pair) -- added 2026-08-20 (conductor's dispute-resolution commission,
// dispute 1: TestD1b's own original fixture hand-picked an (m, e) never verified against the
// real domain check, which the domain-check-derived value below fixes at the source rather
// than by trial and error).
inline bool DeriveDefaultScaleConstantsSourceME(int64_t* out_m, int32_t* out_e) {
	using namespace superslm;
	constexpr int64_t kTargetQLn2 = 493;
	for (int64_t e = -80; e <= 0; ++e) {
		for (int64_t m = (int64_t{1} << 30); m < (int64_t{1} << 31); m += (int64_t{1} << 20)) {
			int64_t a = 0, b = 0, c = 0;
			if (IExpScaleConstants(m, e, kIExpLn2Q, 30, kIExpBQ, 30, kIExpCaQ, 30, &a, &b, &c) !=
			    IExpScaleDomain::kOk)
				continue;
			if (a == kTargetQLn2) {
				*out_m = m;
				*out_e = static_cast<int32_t>(e);
				return true;
			}
		}
	}
	return false;
}

}  // namespace t2199phaseD

#endif  // SSLM_T2199_PHASED_FIXTURE_H

// SuperSLM S3.2/S3.3 site compositions.
//
// See include/superslm/forward_sites.h for the contract
// (SuperSLM_S3a_WalkingSkeleton_Plan.md §11 S3.2, §11 S3.3; C31, C24/C25, C28,
// F-S3-8, C27, C33). The S3.2 bodies (FloorDivI64, RmsNormSite,
// ApplyWeightScaleFold, BiasReconcile, EmbedEntry) are the real green
// construction against the red suite authored in tests/test_main.cpp
// (Claude/Curie/superslm-s3.2-weightless-and-projection-sites-test-design-
// 2026-07-28.md §11). The S3.3 bodies below (LandingRescale, ClampRopeCode)
// are deliberately-wrong red-first STUBS (S3.3's header-contract commit) —
// see each function's own comment.
#include "superslm/forward_sites.h"

#include <vector>

#include "superslm/intmath.h"

namespace superslm {

namespace {

// pipeline.py:190: NORM_FRAC_BITS = 16 (SuperSLM_S3a_WalkingSkeleton_Plan.md
// §5.1's own pinned integer — the normalization divide and the per-element
// wide-row divide both shift by 2*NORM_FRAC_BITS).
constexpr int kNormFracBits = 16;

}  // namespace

int64_t FloorDivI64(int64_t a, int64_t b) {
	// C31 (§5.1): the greatest integer q with q*b <= a, for b > 0 (caller-
	// ensures). C++'s own `/` truncates toward zero; floor division differs
	// from it exactly when the divide is inexact and the numerator is negative
	// (F-S3-2) -- correct for that one case by decrementing the truncated
	// quotient.
	const int64_t q = a / b;
	const int64_t r = a % b;
	return (r != 0 && r < 0) ? q - 1 : q;
}

SslmForwardStatus RmsNormSite(const int8_t* h, const int32_t* g, size_t hidden_size,
                               CarriedScale /*incoming_scale*/, CarriedScale site_constant,
                               int8_t* out_codes, CarriedScale* out_scale,
                               std::string_view site, size_t token_index,
                               SslmTraceHookState* trace_hook_state) {
	// C31 (§5.1, §6.2 step 1/9): sumsq -> ISqrt(FloorDivI64(...)) ->
	// max(root,1) -> per-element FloorDivI64(h[i]<<2*NORM_FRAC_BITS, root)*g[i]
	// -> the funnel, with the incoming span EMPTY. `incoming_scale` is accepted
	// per the header's own contract but never folded in: the norm's carried
	// output scale is C23's gain-derived (scale-killing) form alone (Coverage
	// Model dim 7).
	int64_t sumsq = 0;
	for (size_t i = 0; i < hidden_size; ++i) {
		const int64_t hi = static_cast<int64_t>(h[i]);
		sumsq += hi * hi;
	}
	int64_t root =
	    ISqrt(FloorDivI64(sumsq << (2 * kNormFracBits), static_cast<int64_t>(hidden_size)));
	root = root > 1 ? root : 1;

	std::vector<int64_t> wide(hidden_size);
	for (size_t i = 0; i < hidden_size; ++i) {
		const int64_t hi = static_cast<int64_t>(h[i]);
		wide[i] = FloorDivI64(hi << (2 * kNormFracBits), root) * static_cast<int64_t>(g[i]);
	}

	// §11 S3.1a (D-SLM362): `site`/`token_index`/`trace_hook_state` are
	// forwarded unchanged -- this site never fixes its own name (the same
	// composition serves every RMSNorm instance in the per-layer forward), so
	// the caller's own site string, token index, and model-handle hook state
	// are exactly what reaches the funnel's own emission seam.
	const ChainResult result = RequantChainChecked(wide.data(), hidden_size,
	                                                std::span<const CarriedScale>{}, site_constant,
	                                                out_codes, out_scale, site, token_index,
	                                                trace_hook_state);
	return result.status;
}

int64_t ApplyWeightScaleFold(int64_t acc, int32_t identity, int32_t mult, int32_t shift) {
	// C24/C25 (§4.3, §6.2 step 2/6/10/12): identity==1 is the true pass-through
	// (acc unchanged, no multiply, no shift); identity==0 applies the already-
	// shipped MultiplyByQuantizedMultiplier. `identity` is load-time validated
	// to {0,1} (WeightScaleIdentityNotBool, model.cpp), so no third value ever
	// reaches this dispatch.
	if (identity != 0) return acc;
	return static_cast<int64_t>(
	    MultiplyByQuantizedMultiplier(static_cast<int32_t>(acc), mult, shift));
}

int64_t BiasReconcile(int64_t b, int64_t q_b, int64_t r_a, int64_t e_a) {
	// C28 (§4.4, §6.2 step 2): round_half_away_from_zero(B * R_a /
	// 2^(q_B + 62 + e_a)). RoundingDivideByPOT's int64 overload already ties
	// away from zero (C3), which is load-bearing here because B is signed
	// (§4.4). The composed exponent's domain is the caller's own check
	// (CheckRoundingDivideByPotExponentDomain, checked_chain_funnel.h) before
	// this function is ever invoked; it performs no check of its own, matching
	// every other funnel-adjacent compute in this tree.
	const int64_t exponent = q_b + 62 + e_a;
	return RoundingDivideByPOT(b * r_a, static_cast<int>(exponent));
}

int64_t LandingRescale(int64_t /*branch_code*/, int64_t /*m_a*/, int64_t /*r_t*/,
                        int64_t /*e_a*/, int64_t /*e_t*/,
                        uint64_t* /*out_saturation_count*/) {
	return 0;  // stub (S3.3 red-phase) -- matching this campaign's own precedent
	           // (S3.1/S3.2, a594dd2) for every int64_t-returning function: a
	           // fixed wrong sentinel so a follow-up red suite fails against the
	           // real construction rather than compiling against a missing symbol.
	           // `out_saturation_count` is an output parameter and is therefore
	           // left untouched on this path too, matching the funnel's own "on
	           // rejection, neither is touched" convention extended to every
	           // stub's output parameters in this campaign -- a caller that
	           // passes a non-null accumulator here sees it unmodified, which is
	           // itself deliberately wrong the moment a real hostile-value cell
	           // asserts a nonzero count.
}

int64_t ClampRopeCode(int64_t /*raw*/) {
	return 0;  // stub (S3.3 red-phase) -- same convention as LandingRescale above.
}

SslmForwardStatus EmbedEntry(int32_t token_id, int32_t vocab_size, const int8_t* embed_weights,
                              size_t hidden_size, CarriedScale site_constant, int8_t* out_codes,
                              CarriedScale* out_scale, std::string_view site, size_t token_index,
                              SslmTraceHookState* trace_hook_state) {
	// F-S3-8 (§4.8, §6.1): validate token_id against [0, vocab_size) BEFORE any
	// row of embed_weights is read -- a host-supplied id, never sanitized
	// upstream.
	if (token_id < 0 || token_id >= vocab_size) {
		return SslmForwardStatus::TokenIdOutOfRange;
	}
	const int8_t* row = embed_weights + static_cast<size_t>(token_id) * hidden_size;
	std::vector<int64_t> wide(hidden_size);
	for (size_t i = 0; i < hidden_size; ++i) {
		wide[i] = static_cast<int64_t>(row[i]);
	}
	// §11 S3.1a (D-SLM362): forwarded unchanged, same convention as RmsNormSite
	// above -- this entry has exactly one instance in the forward, but the
	// caller still supplies its site string ("embed") rather than this
	// function fixing it, matching RmsNormSite rather than special-casing.
	const ChainResult result = RequantChainChecked(wide.data(), hidden_size,
	                                                std::span<const CarriedScale>{}, site_constant,
	                                                out_codes, out_scale, site, token_index,
	                                                trace_hook_state);
	return result.status;
}

}  // namespace superslm

// SuperSLM S3.2 site compositions — the weightless and projection sites
// (SuperSLM_S3a_WalkingSkeleton_Plan.md §11 S3.2; C31, C24/C25, C28, F-S3-8).
//
// Declares the RMSNorm site (C31's floor-divide construction, §5.1), the WSC1
// identity/near-identity fold-apply dispatch (C24/C25), the C28 bias-
// reconciliation compute (§4.4), and the embed entry with its host-supplied
// token-id validation (F-S3-8, §4.8). C28's own (q_B, e_a) domain predicate,
// CheckRoundingDivideByPotExponentDomain, is declared in checked_chain_funnel.h
// instead — it is a derived-operand predicate in the funnel's own §7.2 second-limb
// family, not a site composition, and re-stages a declaration that already lived
// there once (see that header's own comment).
//
// THIS FILE IS S3.2's HEADER-CONTRACT STEP (red-first TDD, matching the S3.1/
// S2.1 precedent): the declarations below are the approved API surface (Claude/
// Curie/superslm-s3.2-weightless-and-projection-sites-test-design-2026-07-28.md
// §4/§9). Bodies in src/forward_sites.cpp are STUBS — they compile and link but
// return a fixed, deliberately wrong value when called. Curie's red suite is
// authored against this contract in a follow-up pass; the real construction
// lands at the next build step (Brunel green).
//
// PLACEMENT NOTE: this file's own translation unit (src/forward_sites.cpp) is
// deliberately NOT under src/forward/, even though these are forward-composition
// sites in the plan's own sense. tests/ci/check_no_forward_leaf_calls.py's
// _DEFAULT_FORWARD_GLOBS scans src/forward/**, and its own end-to-end test
// (test_main_end_to_end_against_the_real_default_glob_is_no_longer_vacuous,
// read-only to this campaign) asserts that glob matches EXACTLY ONE file,
// src/forward/checked_chain_funnel.cpp. Adding a second real file under
// src/forward/ would fail that exact-population assertion without a matching
// update to the check itself, which is out of this campaign's writable scope.
// Placing these sites at src/forward_sites.cpp (a sibling of src/model.cpp)
// keeps that test's assertion true; a future pass that owns tests/ can widen
// the check's glob/allowlist to bring this file under its coverage once the
// real construction is written, so a direct leaf call from a site composition
// is caught the same way one from the funnel's own file would be.
#ifndef SUPERSLM_FORWARD_SITES_H
#define SUPERSLM_FORWARD_SITES_H

#include <cstddef>
#include <cstdint>

#include "superslm/checked_chain_funnel.h"

namespace superslm {

// C31 (§5.1): a general signed-numerator, positive-denominator floor divide —
// the greatest integer q with q*b <= a, for b > 0 (caller-ensures). This is a
// different function from C's own `/`, which truncates toward zero and disagrees
// with this one on every negative, inexact divide (F-S3-2). Used by RmsNormSite
// below for both the normalization divide and the per-element wide-row divide.
int64_t FloorDivI64(int64_t a, int64_t b);

// C31's RMSNorm site (§5.1, §6.2 step 1/9): composes
//   sumsq   = sum_i (int64)h[i] * (int64)h[i]
//   root    = max(ISqrt(FloorDivI64(sumsq << (2*NORM_FRAC_BITS), hidden_size)), 1)
//   wide[i] = FloorDivI64((int64)h[i] << (2*NORM_FRAC_BITS), root) * (int64)g[i]
// then the funnel (RequantChainChecked) over `wide`, with the incoming span
// EMPTY — never `incoming_scale` — and `site_constant` as the sole factor
// (§6.2 step 1: "chain with empty incoming; carried scale is C23's gain-derived
// (scale-killing) form"). `incoming_scale` is accepted as a parameter solely so
// Coverage Model dimension 7's differential cell can vary it and assert it is
// annihilated: a conformant implementation's `out_codes`/`*out_scale` do not
// depend on it at all, and an implementation that instead folds it into the
// funnel's `incoming` span is non-conformant (§11 S3.2's own gate line).
// `h`/`g` each have `hidden_size` elements. On Ok, `out_codes` (hidden_size
// elements) and `*out_scale` are written; on any rejection, neither is touched
// (the funnel's own convention, §7.2).
SslmForwardStatus RmsNormSite(const int8_t* h, const int32_t* g, size_t hidden_size,
                               CarriedScale incoming_scale, CarriedScale site_constant,
                               int8_t* out_codes, CarriedScale* out_scale);

// C24/C25's WSC1 fold-apply dispatch (§4.3, §6.2 step 2/6/10/12): `identity == 1`
// is the true pass-through (returns `acc` unchanged — no multiply, no shift);
// `identity == 0` applies the already-shipped MultiplyByQuantizedMultiplier(acc,
// mult, shift). The identity/near-identity discrimination this dispatches between
// is the §11 S3.2 C24 cell: applying the near-identity fold where a true
// pass-through is owed changes the folded channel's own value by exactly one,
// and — because every element of a site's wide row shares one D' through the
// funnel — that one-unit difference changes another element's requantized code
// too, even though its own raw value never changed ("the divergence class is
// token-wide through the shared D'", SuperSLM_Plan.md:2194-2197).
int64_t ApplyWeightScaleFold(int64_t acc, int32_t identity, int32_t mult, int32_t shift);

// C28's bias-reconciliation compute (§4.4, §6.2 step 2): the reference's
// `bias_reconcile(B, q_B, R_a, e_a)` — round_half_away_from_zero(B * R_a /
// 2^(q_B + 62 + e_a)) (C3's tie rule, ties away from zero, load-bearing here
// because B is signed, unlike C30's positive-operand floor shifts). The
// composed exponent q_B + 62 + e_a must be checked against
// CheckRoundingDivideByPotExponentDomain (checked_chain_funnel.h) at the call
// site before this function forms the divide — this function itself performs
// no such check, matching every other funnel-adjacent compute in this tree
// (the domain check and the compute are separate calls).
int64_t BiasReconcile(int64_t b, int64_t q_b, int64_t r_a, int64_t e_a);

// The forward's entry (C23, §6.1, F-S3-8): validates `token_id` against
// `[0, vocab_size)` BEFORE any row of `embed_weights` is read (§4.8's standing
// obligation — a host-supplied id, never sanitized upstream), returning
// TokenIdOutOfRange on failure with `out_codes`/`*out_scale` untouched. On
// success, reads `embed_weights`'s row `token_id` (hidden_size int8 codes,
// row-major: `embed_weights + token_id * hidden_size`), widens each element to
// int64 (bound 127), and runs the funnel (RequantChainChecked) with the
// incoming span EMPTY and `site_constant` as the sole factor (§6.1: "incoming
// scale empty, site constant composition_constants[\"embed\"]").
SslmForwardStatus EmbedEntry(int32_t token_id, int32_t vocab_size,
                              const int8_t* embed_weights, size_t hidden_size,
                              CarriedScale site_constant, int8_t* out_codes,
                              CarriedScale* out_scale);

}  // namespace superslm

#endif  // SUPERSLM_FORWARD_SITES_H

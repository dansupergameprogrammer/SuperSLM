// SuperSLM S3.2/S3.3 site compositions — the weightless/projection sites and
// the attention-interior's landing/clamp sites
// (SuperSLM_S3a_WalkingSkeleton_Plan.md §11 S3.2, §11 S3.3; C31, C24/C25, C28,
// F-S3-8, C27, C33).
//
// Declares the RMSNorm site (C31's floor-divide construction, §5.1), the WSC1
// identity/near-identity fold-apply dispatch (C24/C25), the C28 bias-
// reconciliation compute (§4.4), the embed entry with its host-supplied
// token-id validation (F-S3-8, §4.8), the K/V landing composite's rescale
// (C27, §8.1), and C33's post-rotation clamp (§5.3). C28's own (q_B, e_a)
// domain predicate, CheckRoundingDivideByPotExponentDomain, and C32's own
// softmax-row width predicate, CheckSoftmaxRowWidthDomain, are declared in
// checked_chain_funnel.h instead — both are derived-operand predicates in the
// funnel's own §7.2 second-limb family, not site compositions.
//
// The S3.2 declarations' bodies (RmsNormSite, ApplyWeightScaleFold,
// BiasReconcile, EmbedEntry, FloorDivI64) are the real green construction
// (Claude/Curie/superslm-s3.2-weightless-and-projection-sites-test-design-
// 2026-07-28.md §4/§9). The S3.3 declarations below (LandingRescale,
// ClampRopeCode) are likewise real green constructions, against
// Claude/Curie/superslm-s3.3-attention-interior-test-design-2026-07-28.md
// §6.1/§6.3/§11's own red suite.
//
// PLACEMENT: this file's own translation unit now lives at
// src/forward/forward_sites.cpp, under the directory glob
// tests/ci/check_no_forward_leaf_calls.py's _DEFAULT_FORWARD_GLOBS already
// scans (src/forward/**/*.cpp). The sibling glob entry that module carries
// for this file's PRIOR path (src/forward_sites.cpp, a sibling of
// src/model.cpp) is now redundant with the directory glob rather than wrong
// (that module's own docstring names this outcome); no tests/ edit was
// needed to land the move, and none happened here (tests/ is read-only to
// this campaign).
#ifndef SUPERSLM_FORWARD_SITES_H
#define SUPERSLM_FORWARD_SITES_H

#include <cstddef>
#include <cstdint>
#include <string_view>

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
//
// `site`, `token_index`, and `trace_hook_state` (§11 S3.1a; D-SLM362) are
// forwarded UNCHANGED to the internal RequantChainChecked call, exactly the
// three trailing parameters that call already carries (checked_chain_funnel.h).
// This site's own name is never fixed here — the same composition serves
// every RMSNorm instance in the per-layer forward (attention-norm, FFN-norm,
// final-norm), each under its own site string ("layer3.attn_norm",
// "final_norm", ...) supplied by the caller that knows which one it is
// (§4.1's naming convention, Claude/Curie/superslm-s3.1a-trace-hook-test-
// design-2026-07-28.md §4.1). They default to an empty site, index 0, and
// nullptr so every existing call compiles unchanged and emits no trace record
// (RequantChainChecked's own default-argument convention, extended here).
SslmForwardStatus RmsNormSite(const int8_t* h, const int32_t* g, size_t hidden_size,
                               CarriedScale incoming_scale, CarriedScale site_constant,
                               int8_t* out_codes, CarriedScale* out_scale,
                               std::string_view site = {}, size_t token_index = 0,
                               SslmTraceHookState* trace_hook_state = nullptr);

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

// C27's K/V landing composite (§8.1, §11 S3.3 §6.1): the reference's
// `residual_reconcile(branch_code, m_a, r_t, e_a, e_t)` — the reciprocal-side
// reconciliation that lands a K/V branch value at its site's static per-head
// scale. Composition (not yet ported anywhere in this tree):
//   round_half_away_from_zero((branch_code * m_a * r_t) / 2^(62 - (e_a - e_t)))
// C3's tie rule (ties away from zero), load-bearing here because branch_code is
// signed. `(m_a, e_a)` is the incoming carried mantissa/exponent shared by the
// K and V branches at this site (the norm output); `(r_t, e_t)` are the
// OFFLINE per-(head,projection) reciprocal and exponent read straight from the
// artifact's `KvLandingReciprocals`/`KvLandingScales` sections — no runtime
// reciprocal exists at this site. The wide intermediate is C22-class
// (~2^94), carried the same way C22's own composition is (a 128-bit
// intermediate, never a narrower one) — this function's REAL body is where
// that carry is built; this declaration states the contract only. The call
// site composes `clamp(LandingRescale(...), -127, 127)` (§8.1); the clamp is
// the caller's, matching C33's own clamp-is-the-caller's convention below.
//
// `out_saturation_count` (T-518 / D-SLM201 option 2, §8.2; the PREDICATED-
// INCREMENT half of the saturation counter, staged now per the plan's own
// "written into the kernel as it is authored, not after" -- §8.2, D-SLM201).
// The clamp comparison the caller's own `clamp(..., -127, 127)` performs is
// evaluated once, internally, against this function's own about-to-be-
// returned raw value -- not a second, independently-derived comparison --
// and when it is out of `[-127, 127]`, `*out_saturation_count` is
// INCREMENTED by exactly one (never reset, never assigned): the accumulator
// is the CALLER's, owned across every call for one sequence (§8.2:
// "granularity: per sequence"; reset on sequence create / `sslm_seq_reset` is
// the caller's own responsibility, not this function's). This has NO EFFECT
// WHATSOEVER on the return value (§8.2's own non-negotiable property) --
// the raw, unclamped result is identical whether or not counting is
// requested. Defaults to `nullptr`, in which case nothing is read or
// written, exactly the convention `RmsNormSite`/`EmbedEntry` already use for
// their own optional trailing parameters -- every existing call (there are
// none yet) compiles unchanged. The REPORTING surface -- a field on the
// decode-step status that exposes this accumulator to the host -- is a
// separate, deferred obligation (S3.6/S3.7's own decode-step struct; see
// this pass's build log) and is NOT this parameter's job: this parameter is
// the counting mechanism only, never a report.
int64_t LandingRescale(int64_t branch_code, int64_t m_a, int64_t r_t, int64_t e_a,
                        int64_t e_t, uint64_t* out_saturation_count = nullptr);

// C33's post-rotation clamp (§5.3, §11 S3.3 §6.2 step 3): `RopeApplyPair`
// (intmath.h) returns its rotated pair UNCLAMPED and int64-wide by its own
// header's own contract ("clamping to the activation format is the caller's
// (site composition)") — this is that caller. Clamps `raw` (one component of
// a `RopePair`) to the pinned code range `[-127, 127]`, NOT the int8 storage
// range `[-128, 127]` — the two ranges differ at exactly the one value -128,
// and the design's own discriminating fixture (§5.3) exists to catch an
// implementation that clamps at the wrong one.
int64_t ClampRopeCode(int64_t raw);

// The forward's entry (C23, §6.1, F-S3-8): validates `token_id` against
// `[0, vocab_size)` BEFORE any row of `embed_weights` is read (§4.8's standing
// obligation — a host-supplied id, never sanitized upstream), returning
// TokenIdOutOfRange on failure with `out_codes`/`*out_scale` untouched. On
// success, reads `embed_weights`'s row `token_id` (hidden_size int8 codes,
// row-major: `embed_weights + token_id * hidden_size`), widens each element to
// int64 (bound 127), and runs the funnel (RequantChainChecked) with the
// incoming span EMPTY and `site_constant` as the sole factor (§6.1: "incoming
// scale empty, site constant composition_constants[\"embed\"]").
//
// `site`, `token_index`, and `trace_hook_state` (§11 S3.1a; D-SLM362) are
// forwarded UNCHANGED to the internal RequantChainChecked call, the same
// three trailing parameters that call already carries and the same default
// convention RmsNormSite above uses — an empty site, index 0, and nullptr, so
// every existing call compiles unchanged and emits no trace record. This
// entry has exactly one instance in the forward (§4.1's own example names it
// "embed"); the caller supplies that literal rather than this function fixing
// it, matching RmsNormSite's own convention rather than special-casing the
// single-instance site.
SslmForwardStatus EmbedEntry(int32_t token_id, int32_t vocab_size,
                              const int8_t* embed_weights, size_t hidden_size,
                              CarriedScale site_constant, int8_t* out_codes,
                              CarriedScale* out_scale,
                              std::string_view site = {}, size_t token_index = 0,
                              SslmTraceHookState* trace_hook_state = nullptr);

}  // namespace superslm

#endif  // SUPERSLM_FORWARD_SITES_H

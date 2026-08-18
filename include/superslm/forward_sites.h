// SuperSLM S3.2/S3.3/S3.4 site compositions — the weightless/projection
// sites, the attention-interior's landing/clamp sites, and the MLP half's
// SwiGLU activation site
// (SuperSLM_S3a_WalkingSkeleton_Plan.md §11 S3.2, §11 S3.3, §11 S3.4; C31,
// C24/C25, C28, F-S3-8, C27, C33, C34).
//
// Declares the RMSNorm site (C31's floor-divide construction, §5.1), the WSC1
// identity/near-identity fold-apply dispatch (C24/C25), the C28 bias-
// reconciliation compute (§4.4), the embed entry with its host-supplied
// token-id validation (F-S3-8, §4.8), the K/V landing composite's rescale
// (C27, §8.1), C33's post-rotation clamp (§5.3), the RoPE application
// site (§6.2 step 3, §11 S3.3's own gate line; D-SLM376, D-SLM383, D-SLM384),
// and the SwiGLU activation site (§5.4, §6.3 step 11; T-1345). C28's own
// (q_B, e_a) domain predicate, CheckRoundingDivideByPotExponentDomain, and
// C32's own softmax-row width predicate, CheckSoftmaxRowWidthDomain, are
// declared in checked_chain_funnel.h instead — both are derived-operand
// predicates in the funnel's own §7.2 second-limb family, not site
// compositions. C34's own derived-operand predicate,
// CheckSiluCompositionScaleDomain, is likewise declared in
// checked_chain_funnel.h and is already real and green (S3.1); MlpActSite
// below is the site that calls it, not the predicate itself.
//
// The S3.2 declarations' bodies (RmsNormSite, ApplyWeightScaleFold,
// BiasReconcile, EmbedEntry, FloorDivI64) are the real green construction
// (Claude/Curie/superslm-s3.2-weightless-and-projection-sites-test-design-
// 2026-07-28.md §4/§9). The S3.3 declarations LandingRescale and ClampRopeCode
// are likewise real green constructions, against
// Claude/Curie/superslm-s3.3-attention-interior-test-design-2026-07-28.md
// §6.1/§6.3/§11's own red suite. RopeApplySite below is likewise real and
// green — CheckPositionOverCap first, then the ROP1 table read, then
// RopeApplyPair+ClampRopeCode per pair — against
// Claude/Curie/superslm-s3.3-rope-application-site-test-design-2026-07-28.md
// §6's own red suite; D-SLM376 rules it is built here, in S3.3, as
// CheckPositionOverCap's own first act. MlpActSite below is S3.4's own real
// green construction — the four-step composition its doc comment states,
// replacing the declare-and-stub landed with
// Claude/Curie/superslm-s3.4-mlp-act-site-test-design-2026-07-29.md's red
// suite (Claude/Brunel/superslm-s3.4-mlp-act-site-body-build-2026-07-29.md).
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
#include "superslm/model.h"  // SslmTensorManifest (RopeApplySite's rope_tables parameter)

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
// `external_wide_scratch`, trailing, defaulted to nullptr (T-2139 closing round, coordinator's
// own "eliminate it honestly (into the workspace or off the path)" instruction, curie/
// t2138-abi-red-suite@11e7182's own recalibrated dim7 C1a cell): when non-null, MUST point at
// `hidden_size` writable `int64_t` elements -- this call's own `wide` intermediate is written
// there instead of a freshly heap-allocated std::vector, letting a caller that already owns a
// correctly-sized scratch region (the sslm_* consumer ABI's own workspace, src/sslm_abi.cpp) make
// this call with zero allocations of its own. nullptr (every existing call site, unchanged) keeps
// the original internally-allocated behavior exactly as before -- this is additive, not a
// behavior change for any caller that does not pass it.
SslmForwardStatus RmsNormSite(const int8_t* h, const int32_t* g, size_t hidden_size,
                               CarriedScale incoming_scale, CarriedScale site_constant,
                               int8_t* out_codes, CarriedScale* out_scale,
                               std::string_view site = {}, size_t token_index = 0,
                               SslmTraceHookState* trace_hook_state = nullptr,
                               int64_t* external_wide_scratch = nullptr);

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

// T-2021/T-2029 B2 (design Sec4, D-SLM2915): the runtime-additive LoRA delta-fold's own
// primitive -- `ApplyWeightScaleFold`'s sibling, over the SIGNED
// [kAmplifyingScaleExponentMin, kAmplifyingScaleExponentMax] = [-31,31] domain (model.h) rather
// than WSC1's own unsigned [0,31]. `identity == 1` is the same exact `rho == 1` pass-through;
// `identity == 0` and `exponent >= 0` is BIT-IDENTICAL to `ApplyWeightScaleFold`'s own
// non-identity branch (same two gemmlowp primitives, same order) -- only `exponent < 0`
// (amplification, `rho > 1`, the ratio `ApplyWeightScaleFold` cannot represent, T-1990's own
// fracture) is new arithmetic, via `SaturatingLeftShift32` (intmath.h). Used ONLY by the
// delta-fold and u-fold (design Sec4/Sec8); WSC1's own base fold keeps calling
// `ApplyWeightScaleFold` exactly as before, unmodified -- its ratio never needs amplification.
//
// Precondition, bounded by construction rather than asserted (design Sec4, D-SLM2983): `|acc| <
// 2^31` at both real call sites (`delta_raw[i]`, a sum of `r` int8x8 products, one per rank
// term; `u_acc[k]`, a sum over `d` contracting terms) -- `static_cast<int32_t>`
// on the first line silently TRUNCATES, not saturates, if this bound is exceeded. Bounded, not
// checked: at this project's own widest rank/dimension (Qwen2.5, `SuperSLM_Plan.md` Sec11),
// `|delta_raw[i]| <= r*127^2` and `|u_acc[k]| <= d*127^2` sit roughly two orders of magnitude
// below 2^31 (design Sec4's own executed figures).
int64_t ApplyAmplifyingWeightScaleFold(int64_t acc, int32_t identity, int32_t mult, int32_t exponent);

// C28's bias-reconciliation compute (§4.4, §6.2 step 2): the reference's
// `bias_reconcile(B, q_B, R_a, e_a)` — round_half_away_from_zero(B * R_a /
// 2^(q_B + 62 + e_a)) (C3's tie rule, ties away from zero, load-bearing here
// because B is signed, unlike C30's positive-operand floor shifts). The
// composed exponent q_B + 62 + e_a must be checked against
// CheckRoundingDivideByPotExponentDomain (checked_chain_funnel.h) at the call
// site before this function forms the divide — this function itself performs
// no domain check of its own; the domain check and the compute are separate
// calls. T-1657 Poirot N-1/N-7 (confirmation pass 5156477): the composed
// exponent IS checked one level down, inside BiasReconcileWide (intmath.h,
// D-SLM676), because that function must form 2^exponent to divide by — an
// out-of-domain exponent no longer risks undefined behavior, only a silently
// wrong or zero result with no diagnostic, which is why the call-site check
// above stays required. See BiasReconcileWide's own contract for the two
// distinct false-return paths this delegates to. (Several computes in this
// tree guard their own degenerate-length
// case inside the kernel instead of relying on a separate call-site gate —
// `SoftmaxRowQ15`, `RowBoundsWide` (intmath.h) — see each one's own contract
// for its guard; that is a different property from this function's own
// domain check, which stays the caller's. `MaxAbsReduceWide` is neither of
// these: it is total over `n`, including `n == 0` — the all-zero-row guard
// on its accumulated magnitude runs unconditionally and the element loop
// simply does not execute at `n == 0` — so it has no degenerate-length case
// to guard and no `n >= 1` precondition for a caller to ensure; see its own
// contract, intmath.h.)
int64_t BiasReconcile(int64_t b, int64_t q_b, int64_t r_a, int64_t e_a);

// C27's K/V landing composite (§8.1, §11 S3.3 §6.1): the reference's
// `residual_reconcile(branch_code, m_a, r_t, e_a, e_t)` — the reciprocal-side
// reconciliation that lands a K/V branch value at its site's static per-head
// scale. This function's own body (forward_sites.cpp) is that composition,
// real and green — not yet CALLED from any production forward call site in
// this tree (S3.4+ wires the actual call site in; this declaration states
// the contract, and the .cpp file is where it is already built):
//   round_half_away_from_zero((branch_code * m_a * r_t) / 2^(62 - (e_a - e_t)))
// C3's tie rule (ties away from zero), load-bearing here because branch_code
// AND m_a are both signed (Popper 2026-07-28 §3.1: a mid-composition carried
// mantissa is NOT positive by construction, only required to fit int32_t's
// range — see this function's own body for the corrected sign handling).
// `(m_a, e_a)` is the incoming carried mantissa/exponent shared by the
// K and V branches at this site (the norm output); `(r_t, e_t)` are the
// OFFLINE per-(head,projection) reciprocal and exponent, and BOTH are read
// straight from the artifact's `KvLandingReciprocals` section alone (word 2
// and word 1 respectively; Tools/superslm_spike/dynamic_engine.py:374-378's
// own unpack, `_m_t, e_t, r_t = model.kv_landing_reciprocals[...]`) — no
// runtime reciprocal exists at this site. `KvLandingScales` carries this
// site's separate LANDED target (m_target, e_target), never an argument
// to this function; naming the two sections together as if either supplied
// `(r_t, e_t)` is what put commit 1b0bd10's e_t domain floor on the wrong
// section's word (model.cpp; Poirot 2026-07-28 remediation-confirmation
// review, finding B; D-SLM372). The wide intermediate is C22-class
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
// returned raw value -- not a second, independently-derived comparison.
// **Neither e_t nor e_a carries a domain check anywhere in this tree**
// (Popper 2026-07-28 §3.2): an extreme composed exponent drives the negative-k
// branch's left shift past the 128-bit carry's own width, which would
// otherwise narrow to a silently wrong small `raw` with the counter never
// told. This function detects that loss internally (shift-then-verify) and
// counts it as a clamp event even when the narrowed `raw` itself does not
// look out of range -- the return value stays exactly as unreliable in that
// class as it always was (caller-ensures: neither e_t nor e_a is domain-
// checked here), but the counter is not fooled by it.
//
// The composed exponent `62 - (e_a - e_t)` ITSELF is computed without
// signed-integer-overflow UB regardless of `e_a`/`e_t`'s values (T-1596,
// closing Popper's own finding above): both subtractions use unsigned-
// wraparound overflow detection and saturate to `INT64_MIN`/`INT64_MAX`
// rather than overflow, and the result is clamped to a small magnitude
// before either branch below narrows it to `int` for a shift amount --
// closing a second, silent-truncation hazard the same extreme composed
// exponent reaches (an unclamped magnitude narrowed straight to `int`
// lands on an arbitrary small value, not one that reads as "past the
// 128-bit width"). This is UNDEFINEDNESS safety, not a domain check: the
// return value is exactly as unreliable, for the same reason stated above,
// as it always was. Whenever the
// return value itself IS out of `[-127, 127]`, or this internal detection
// fires, `*out_saturation_count` is INCREMENTED by exactly one (never reset,
// never assigned): the accumulator
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
//
// `out_magnitude_exceeded_int64` (T-1377 / D-SLM457, Sec7.2b, Sec14.14): a
// SECOND, DISTINCT out-parameter from `out_saturation_count` above -- the two
// conditions are not the same claim. `out_saturation_count`'s own internal OR
// (`magnitude_exceeds_clamp || raw < -127 || raw > 127`) is coupled to the
// K/V-landing caller's own `[-127, 127]` clamp range, which is not every
// caller's contract (`ResidualReconcileSite` below composes no clamp at all,
// Sec6.2 step 8). This parameter reports only "the true, pre-narrowing
// magnitude does not fit `int64`" -- in the non-negative-`k` branch, the
// 128-bit quotient exceeds `INT64_MAX` (Sec7.2b: "the divisor 2^(k+1) ... must
// bring that magnitude below 2^63" is this exact threshold, stated there as
// the reachability bound and enforced here as the check); in the negative-`k`
// branch, the left-shifted 128-bit magnitude either loses bits (the
// shift-then-verify mismatch already computed for the existing counter) or
// itself exceeds `INT64_MAX`. Neither branch's condition is the `> 127`
// clamp-range test the existing counter also performs -- a wide row
// legitimately carries values outside `+/-127` at this site, so that test
// would false-positive on every ordinary large-but-correct result. When
// non-null, written exactly once per call (never OR'd with a prior value,
// matching `out_saturation_count`'s own "caller passes a per-call flag"
// convention here, NOT its own "accumulate across a sequence" convention --
// this parameter has no accumulation semantics of its own; `ResidualReconcileSite`
// below is the one caller and it checks this flag per element, immediately).
// Defaults to `nullptr`: every existing call -- none of which passes this new
// parameter yet -- compiles unchanged and nothing is read or written through
// it. (Corrected 2026-07-31, Poirot 5eff945-t1380-t1381-t1382-review-2026-
// 07-31.md, Minor 1: the prior correction here cited existing calls by
// `file:line`, and the citation was stale the moment it was written -- an
// edit earlier in the same commit had already moved the lines it named. A
// `file:line` citation inside a source comment goes stale on every edit
// above it; the claim itself needs no citation at all.)
int64_t LandingRescale(int64_t branch_code, int64_t m_a, int64_t r_t, int64_t e_a,
                        int64_t e_t, uint64_t* out_saturation_count = nullptr,
                        bool* out_magnitude_exceeded_int64 = nullptr);

// C33's post-rotation clamp (§5.3, §11 S3.3 §6.2 step 3): `RopeApplyPair`
// (intmath.h) returns its rotated pair UNCLAMPED and int64-wide by its own
// header's own contract ("clamping to the activation format is the caller's
// (site composition)") — this is that caller. Clamps `raw` (one component of
// a `RopePair`) to the pinned code range `[-127, 127]`, NOT the int8 storage
// range `[-128, 127]` — the two ranges differ at exactly the one value -128,
// and the design's own discriminating fixture (§5.3) exists to catch an
// implementation that clamps at the wrong one.
int64_t ClampRopeCode(int64_t raw);

// S3.3's RoPE application site (§6.2 step 3, §11 S3.3's own gate line;
// T-1308, D-SLM376, D-SLM383, D-SLM384). Signature matches Claude/Curie/
// superslm-s3.3-rope-application-site-test-design-2026-07-28.md §6.1 exactly,
// so that record's §6 fixtures drop in unchanged once the real body below
// replaces this stub. `row`/`out_row` each have `head_dim` elements
// (`head_dim` even — the rotation pairs elements, load-time-rejected
// otherwise, per §6.2 step 3's own "head_dim odd is a load-time rejection").
// `rope_tables` is the loaded ROP1 view (`SslmModelView::rope_tables`,
// model.h) carrying the "cos"/"sin" tensors this site reads by row.
//
// The real body performs, in this order and no other:
//   1. CheckPositionOverCap(position, context_cap) (checked_chain_funnel.h) —
//      the site's documented FIRST ACT (D-SLM376). On rejection, return
//      PositionOverCap immediately; `out_row` is untouched and `rope_tables`'
//      "cos"/"sin" tensors are never read — "never a table read" (§11 S3.3's
//      own gate line, §6.2 step 3's own text).
//   2. Resolve `rope_tables.Tensor("cos")` / `Tensor("sin")`. Either absent
//      (nullptr) returns RopeTableTensorMissing — no caller can discharge
//      this precondition, because this function is the one that performs
//      the lookup (Poirot fa3189a-s3.3-rope-site-and-c32-softmax-review-
//      2026-07-28.md, Critical 1).
//   3. Bound `position` and `head_dim / 2` against `cos`/`sin`'s own
//      validated `elem_count` — `context_cap` (step 1) is a fact about CFG1,
//      never a fact about the ROP1 tensors' real shape, and no load-time
//      join ties the two (same review, Critical 2). Out-of-extent returns
//      RopeTableExtentExceeded before either tensor's row `position` is
//      touched.
//   4. Read the "cos"/"sin" tensors' row `position` from `rope_tables`
//      (`head_dim / 2` elements each).
//   5. For each pair `i` in `[0, head_dim/2)`: `RopeApplyPair(row[2i],
//      row[2i+1], cos_row[i], sin_row[i])` (intmath.h — interleaved even/odd
//      pairing, matching the reference's own `_rotate_rows`, not a
//      first-half/second-half split), then `ClampRopeCode` on each
//      component, written to `out_row[2i]` / `out_row[2i+1]`.
//
// The body is real (forward_sites.cpp), driven green against
// Claude/Curie/superslm-s3.3-rope-application-site-test-design-2026-07-28.md
// §6's red suite (the feature-oracle cell, the never-a-table-read ordering
// cell, and the ASan guard-vitality cell) and against
// Claude/Curie/fa3189a-s3.3-rope-site-and-c32-softmax-remediation-test-
// design-2026-07-28.md's remediation suite (the null-tensor and
// extent-exceeded cells for Critical 1 and Critical 2).
SslmForwardStatus RopeApplySite(const int8_t* row, size_t head_dim,
                                 int64_t position, int64_t context_cap,
                                 const SslmTensorManifest& rope_tables,
                                 int8_t* out_row);

// C34's SwiGLU activation site (§5.4, §6.3 step 11; T-1345). The declaration
// and a stub were landed first by the test-design pass that authored this
// site's red suite (Claude/Curie/superslm-s3.4-mlp-act-site-test-design-
// 2026-07-29.md), following the same declare-and-stub sequence the RoPE
// application site used (D-SLM384/385/386); the real body below replaced that
// stub and is green against that suite.
//
// The body performs, in this order and no other (§5.4, §7.2 second limb):
//   1. CheckSiluCompositionScaleDomain(gate_scale.m, gate_scale.e)
//      (checked_chain_funnel.h) -- the site's documented FIRST ACT, before
//      SiluSigmoidQ15 is ever called. On rejection, return
//      SiluCompositionScaleOutOfDomain immediately; out_codes/out_scale are
//      untouched and SiluSigmoidQ15 never evaluates -- the same
//      predicate-before-primitive ordering RopeApplySite's own
//      CheckPositionOverCap-before-table-read establishes as this tree's
//      house convention (§11 S3.3's own gate line).
//   2. For each i in [0, n): sig[i] = SiluSigmoidQ15(sigmoid_lut_table,
//      gate_code[i], gate_scale.m, gate_scale.e) -- C10's fixed-point LUT
//      construction (silu_lut.h:61), NEVER the i-exp-sigmoid construction
//      F-S3-1 found the reference computing before S3.0's own reconciliation
//      (this pinning is the reason the slot exists: an implementation
//      substituting i-exp-sigmoid here diverges from this construction on
//      real activation codes and, downstream, on at least one requantized
//      int8 code -- executed, Claude/Curie/superslm-s3.4-mlp-act-site-
//      test-design-2026-07-29.md's own negative control).
//   3. wide[i] = (int64_t)gate_code[i] * (int64_t)sig[i] * (int64_t)up_code[i]
//      -- the product's bound is 127 * 2^15 * 127 < 2^29 (§5.4), comfortably
//      int64-exact.
//   4. RequantChainChecked(wide, n, incoming={gate_scale, up_scale},
//      site_constant, out_codes, out_scale, site, token_index,
//      trace_hook_state) -- BOTH the gate and up carried scales fold into
//      the funnel's incoming span (Tools/superslm_spike/dynamic_engine.py:
//      546-548's own `_chain_record_vec(model, f"{prefix}.mlp_act", t, wide,
//      [gate_scale[t], up_scale[t]], trace)` -- neither scale is dropped, and
//      the order is gate then up, matching the reference's own list literal).
//
// `gate_code`/`up_code` each have `n` elements (`n == intermediate_size`).
// `gate_scale` is the runtime-derived per-token gate scale that both
// SiluSigmoidQ15's domain check and the funnel's fold consume -- never an
// artifact constant (§5.4). `sigmoid_lut_table` is the caller's materialized
// SIL1 table (`kSiluLutN + 1` native-endian int32 Q15 nodes), the same table
// every other SiluSigmoidQ15 caller in this tree already supplies. On Ok,
// `out_codes` (n elements) and `*out_scale` are written; on any rejection,
// neither is touched (the funnel's own convention, §7.2, extended here to
// this site's own domain check).
//
// `site`, `token_index`, and `trace_hook_state` (§11 S3.1a) are forwarded
// unchanged to the internal RequantChainChecked call, the same convention
// RmsNormSite/EmbedEntry already use -- an empty site, index 0, and nullptr,
// so every existing call compiles unchanged and emits no trace record.
SslmForwardStatus MlpActSite(const int8_t* gate_code, CarriedScale gate_scale,
                              const int8_t* up_code, CarriedScale up_scale, size_t n,
                              const int32_t* sigmoid_lut_table, CarriedScale site_constant,
                              int8_t* out_codes, CarriedScale* out_scale,
                              std::string_view site = {}, size_t token_index = 0,
                              SslmTraceHookState* trace_hook_state = nullptr);

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

// --- S3.5: the mid-token residual, C26's residual-reconciliation site, and
// the layer loop (SuperSLM_S3a_WalkingSkeleton_Plan.md §11 S3.5; §9.3; master
// plan §7; C26/D-SLM57; Claude/Curie/superslm-s3.5-residual-and-layer-loop-
// test-design-2026-07-29.md). ---------------------------------------------

// §9.3's mid-token residual (master plan §7, "the mid-token residual buffer
// and layer-position marker"): the sequence's own suspended state between
// layer-budget micro-steps. Lives in the sequence's own state, NEVER the
// per-call workspace -- a residual parked in shared scratch is a lifetime
// hazard and makes a sequence's bits depend on what else shared the
// workspace (master plan §8.2/W1).
//
// `hidden_codes` is caller-owned, `hidden_size` elements: the residual
// stream's own int8 codes. `hidden_scale` is the residual's own carried
// (m, e). `layer_index` is the layer-position marker -- the next layer
// RunLayerLoop resumes at, `0 <= layer_index <= num_hidden_layers`.
// `layer_index == num_hidden_layers` means the current token is complete;
// starting a fresh token resets it to 0 (§9.3: "a sequence resting between
// whole tokens carries a marker at layer 0 and a zero-length residual" --
// realized here as `layer_index == 0`, `hidden_codes` not yet meaningful).
//
// T-1590/T-1597: §13 dim 9 pins this struct as "addressable as a unit", an
// explicit invitation for a caller to save and restore one -- which means a
// value received in any of these five fields, by EITHER of the two
// functions below that take a `SequenceLayerState&`, carries no guarantee
// of having come from either loop's own resume path. The closed unit is
// the cross product of the two functions and the five fields, not the five
// fields against one function:
//
//   - `RunLayerLoop`: its own top-of-function guard block enumerates all
//     five fields directly; see it for what is validated, what is not, and
//     why in each case. `hidden_codes == nullptr` and `context_length < 0`
//     are rejected there. `layer_index` is already fully bounded by the
//     guard immediately following it. `kv_saturation_count` and
//     `hidden_scale` need no guard -- neither is ever read back as a size,
//     offset, or index.
//   - `RunGreedyDecodeLoop`: its own RunWholeToken step overwrites
//     `hidden_scale` and `layer_index` with fresh, non-restored values
//     BEFORE its first call into `RunLayerLoop` -- so a restored value in
//     either field is discarded, unread, on this path. That same step
//     writes THROUGH `hidden_codes`, dereferencing the caller's own
//     (not-yet-overwritten) pointer one call frame BEFORE `RunLayerLoop` is
//     ever entered. `context_length` and `kv_saturation_count` are never
//     touched directly by this function; both reach `RunLayerLoop`'s own
//     guard block above still carrying whatever value the caller restored.
//     So of the five fields, `hidden_codes` is the only one read, unguarded,
//     before `RunLayerLoop`'s own guard block runs -- `RunGreedyDecodeLoop`
//     therefore carries its own `hidden_codes == nullptr` guard
//     (forward_sites.cpp), checked immediately after its `kv_precision`
//     check and before any token is embedded: the one cell of the ten (two
//     functions x five fields) that needs a guard outside `RunLayerLoop`'s
//     own block.
struct SequenceLayerState {
	int8_t* hidden_codes = nullptr;
	CarriedScale hidden_scale;
	uint32_t layer_index = 0;

	// §8.2 (T-518 / D-SLM201 option 2): the K/V landing saturation counter,
	// "granularity: per sequence" -- owned by the caller across every call for
	// one sequence, exactly like `layer_index`. RunLayerLoop only increments
	// this (via `LandingRescale`'s own `out_saturation_count` parameter); it
	// never resets it -- reset on sequence create / `sslm_seq_reset` is the
	// caller's own responsibility, per LandingRescale's own header.
	uint64_t kv_saturation_count = 0;

	// S3.7 (§11 S3.7 "The mechanism", §9.4): the number of positions of this
	// sequence already committed to the K/V store -- distinct from
	// `layer_index`, which counts progress through the CURRENT token's layers
	// and resets to 0 every token. `context_length` is never reset by a fresh
	// token embed; it advances by exactly one, at RunLayerLoop's own single
	// existing commit point, when the layer that just committed is the
	// token's last (`layer_index == num_hidden_layers` after the increment).
	// A fresh sequence starts at 0, giving `width == 1` on its first token --
	// the exact case S3a already builds and gates.
	int64_t context_length = 0;
};

// C26's residual-reconciliation site (§6.2 step 8 "attn_residual", §6.3 step
// 13 "mlp_residual" -- "as (8)": the SAME function serves both, the caller
// supplying which via `site`, matching RmsNormSite's own multi-instance
// convention). Matches the vendored reference's `residual_reconcile`
// (tests/reference/superslm_spike/intmath.py:431-446), reusing the
// ALREADY-SHIPPED `LandingRescale` for its exact arithmetic -- C27's own
// residual_reconcile call and this one are the SAME primitive at different
// call sites (§8.1's static offline reciprocal there; here, a reciprocal
// derived at runtime from the STREAM's own mantissa, step 1 below) -- then
// folds the result through the funnel's already-proven left-associated
// order (D-SLM57).
//
// Performs, in this order and no other:
//   0. If `stream_scale.m` does not fit int32_t's own range, returns
//      `CarriedScaleMantissaOutOfDomain` immediately, with
//      `out_codes`/`*out_scale` untouched -- the funnel's own
//      `RequantChainChecked` step-0 precondition (`CarriedScaleMantissaFitsInt32`,
//      checked_chain_funnel.cpp), enforced here at the door instead of two
//      steps later (T-1604; Poirot 8f63577-t1602, Significant 2/3). This
//      step changes this function's returned status, relative to the
//      version documented before T-1604, on `stream_scale.m` in
//      `(kInt32Max, 2281701375]` per sign -- a band that was fully defined
//      before this step existed and previously reached step 2 or step 4
//      below.
//   1. r_h = DynamicScaleReciprocal(stream_scale.m) -- C19, over the
//      STREAM's own mantissa, never the branch's (§6.2 step 8's own text:
//      "R_h = C19 over the stream mantissa").
//   2. per element i: reconciled[i] = LandingRescale(branch_code[i],
//      branch_scale.m, r_h, branch_scale.e, stream_scale.e, nullptr) --
//      rescales the branch value into the stream's own scale, UNCLAMPED
//      (no saturation counting at this site -- T-518's counter is C27's
//      landing composite's own, not this one's).
//   3. wide[i] = reconciled[i] + (int64_t)stream_code[i] -- the wide add,
//      both operands now nominally at the stream's own scale.
//   4. RequantChainChecked(wide, hidden_size, /*incoming=*/{stream_scale},
//      site_constant, out_codes, out_scale, site, token_index,
//      trace_hook_state) -- the funnel's own left-associated fold of
//      {stream_scale, site_constant, D'-factor} (D-SLM57; RequantChainChecked
//      step 5), the SAME primitive every other funnel-based site already
//      uses. This is the D-SLM57 association-order pin AT THIS SITE: a right-
//      associated implementation of this call (or one that pre-folds
//      stream_scale and site_constant outside the funnel in a different
//      order) diverges from the funnel's own proven left-associated result
//      (superslm-s3.5-residual-and-layer-loop-test-design-2026-07-29.md §3's
//      witness triple).
//
// `branch_code`/`stream_code` each have `hidden_size` elements. `site_constant`
// is the artifact's KVC1 CompositionConstants entry for this residual site
// (`composition_constants["layerL.attn_residual"]` /
// `"layerL.mlp_residual"`, the trace-hook site-naming convention extended to
// this site, §4.1). On Ok, `out_codes`/`*out_scale` are written; on any
// rejection, neither is touched (§7.2's convention).
//
// **T-1377 / D-SLM457 (§7.2b, §14.14): step 2's own derived-operand predicate,
// reached only when step 0 above has not already rejected the call.**
// `LandingRescale`'s second out-parameter (`out_magnitude_exceeded_int64`) is
// checked for every element BEFORE step 4's funnel call runs. If any element's
// flag is set, this function returns `ResidualReconciliationMagnitudeOutOfDomain`,
// with `out_codes`/`*out_scale` untouched -- the same
// "reject leaves output untouched" contract every other funnel rejection
// already honours. This is a runtime check on `LandingRescale`'s own
// already-computed loss signal, not a static exponent-range threshold: no
// closed-form bound independent of layer count exists here (`CarriedScale.e`
// accumulates as an unbounded sum with no domain check anywhere in this
// tree), and the danger zone is reachable from a single, load-legal call
// (§7.2b's derivation, confirmed by execution).
SslmForwardStatus ResidualReconcileSite(const int8_t* branch_code, CarriedScale branch_scale,
                                          const int8_t* stream_code, CarriedScale stream_scale,
                                          size_t hidden_size, CarriedScale site_constant,
                                          int8_t* out_codes, CarriedScale* out_scale,
                                          std::string_view site = {}, size_t token_index = 0,
                                          SslmTraceHookState* trace_hook_state = nullptr);

// One layer's fully-resolved constants -- weights, WSC1 folds, and KVC1 site
// constants -- for RunLayerLoop below. Caller-resolved: RunLayerLoop performs
// NO WGT1/KVC1-by-name lookup of its own. This is deliberate, not an
// oversight -- D-SLM422 (SuperSLM_S3a_WalkingSkeleton_Plan.md §13.2, R5)
// found NO naming or shape convention committed anywhere in this repository
// for WGT1's per-layer projection tensors (q_proj/k_proj/v_proj/o_proj/
// gate_proj/up_proj/down_proj), while noting S3.4 and S3.5 are the first
// sub-slots scheduled to consume one. S3.4's MlpActSite resolved this by
// taking already-materialized code arrays, never a tensor name; RunLayerLoop
// does the same, at the whole-layer granularity, so authoring this sub-slot's
// red suite does not require inventing the missing naming convention --
// whichever later piece resolves a loaded artifact's WGT1 tensors into a
// `LayerWeights[]` (the host-facing entry point, S3.6/S3.7's own scope or an
// unassigned one) is a separate, downstream obligation this declaration does
// not discharge and does not block on.
//
// All weight matrices are row-major [out_channels, in_channels] (WGT1's own
// layout, matmul.h). `hidden_size == num_attention_heads * head_dim`.
// `k_weight`/`v_weight` are `[num_key_value_heads * head_dim, hidden_size]`,
// distinct from `q_weight`/`o_weight`/`gate_weight`/`up_weight`/`down_weight`
// (each `[hidden_size, hidden_size]` or `[intermediate_size, hidden_size]`/
// `[hidden_size, intermediate_size]` as already documented below) -- GQA
// group dispatch, including this shape, is owned by §11 S3.8a (T-1654,
// D-SLM619/D-SLM623).
//
// C28's bias reconciliation (§6.2 step 2's third component, "then C28's bias
// reconciliation where the site has a BIA1 entry", F-S3-4) is likewise a
// declared narrowing: this struct carries no bias field for any projection,
// so `ProjectAndFunnel` composes GemmInt8AccumulateRow -> the WSC1 fold ->
// the funnel and never calls `BiasReconcile` (already shipped in this same
// translation unit). Sound for a fixture with no BIA1 entries, which is this
// sub-slot's own scope (Poirot e4b398c review, Significant 3); the first
// artifact carrying a BIA1 section for a projection this loop calls is a
// separate, later obligation this declaration does not discharge.
// T-2021/T-2029 B1a (design Sec3/Sec4/Sec8/Sec9, D-SLM2843's `n==1` slot: base + at most ONE
// active runtime adapter): one projection's runtime-additive LoRA adapter state, for exactly one
// (layer, projection) pair. **Present iff `a_weight != nullptr`** -- the SAME "nullptr = absent"
// convention this file already uses for `q_bias`/`k_bias`/`v_bias` below. Two DISTINCT absent
// cases this design requires be exercised independently (design Sec8, D-SLM2884/D-SLM2919):
//   - The whole sequence has NO adapter bound at all (design's NULL-adapter case, B1b): the
//     caller sets `LayerWeights::adapter = nullptr` for EVERY layer.
//   - A bound adapter's own `target_modules` does not cover THIS projection at THIS layer
//     (design's third branch, B1c, the routine PEFT case -- q_proj/v_proj-only adapters are
//     common): `LayerWeights::adapter` is non-null, but this ONE `LayerAdapterProjection`
//     member's own `a_weight` is nullptr.
// Either way the delta-add step (`AddAmplifyingLoraDelta`, forward_sites.cpp) is a complete
// no-op -- `acc[]` is left byte-identical to the base-only path, design's own stated contract.
//
// `a_weight` is `[rank, in_channels]` row-major (PEFT's own `lora_A.weight = (r, in)`, folded
// per `SuperSLM_Plan.md` Sec11's converter text); `b_weight` is `[out_channels, rank]`
// (`lora_B.weight = (out, r)`) -- both already-quantized int8, PEFT scaling folded into the
// fold triples below, per design Sec4/Sec9.
//
// T-2041 (Poirot c81e48c review, Significant 1): `delta_fold_entry`/`u_fold_entry` carry B0b's
// own typed `SslmAmplifyingFoldEntry` (model.h) -- never raw `const int32_t*` -- so the
// insertion point reads triples ONLY through `SslmDeltaFoldScaleView::Identity/Mult/Exponent(entry,row)`
// / `SslmUFoldScaleView::...`, the structural disambiguation design Sec11 B0b was made a build
// blocker to buy ("the engine insertion point reads delta_*/u_* values through B0b's own typed
// accessors, not through a raw pointer this design would otherwise have to trust was pointed at
// the right section"). A `SslmTensorView` (WSC1's own entry type) cannot be assigned into either
// field and compile -- `SslmAmplifyingFoldEntry` is a distinct type. `delta_fold_entry`'s own
// `row_count` is the adapted projection's `out_channels` (one triple PER OUTPUT CHANNEL);
// `u_fold_entry`'s own `row_count` is `LayerAdapter::rank` (one triple PER RANK INDEX, design
// Sec4's extension) -- both already carried on the entry itself (B0b's own dimension
// validation, `ValidateAmplifyingFoldDimension`), not re-derived here.
struct LayerAdapterProjection {
	const int8_t* a_weight = nullptr;
	const int8_t* b_weight = nullptr;
	const SslmAmplifyingFoldEntry* delta_fold_entry = nullptr;  // DeltaFoldScales entry, out_channels rows
	const SslmAmplifyingFoldEntry* u_fold_entry = nullptr;      // UFoldScales entry, rank rows
};

// T-2021/T-2029 B1a (design Sec8, D-SLM2843): the active sequence's bound adapter state for ONE
// layer -- one `LayerAdapterProjection` per PEFT-adaptable projection (design's own seven:
// q/o/gate/up/down/k/v). `rank` is shared by every adapted projection in this bind (this
// design's own single-adapter, single-rank-per-module scope, D-SLM2843).
struct LayerAdapter {
	uint32_t rank = 0;
	LayerAdapterProjection q, o, gate, up, down, k, v;
};

struct LayerWeights {
	const int32_t* attn_norm_gain;  // hidden_size
	CarriedScale attn_norm_site_constant;

	// T-2021/T-2029 B1a (design Sec3/Sec8): nullptr means NO adapter is bound to the active
	// sequence at all for this layer (design's NULL-adapter case, B1b) -- the caller sets this
	// to nullptr on EVERY layer's LayerWeights when the sequence has no bound adapter. A
	// non-null pointer means an adapter IS bound; which of its SEVEN per-projection members
	// (above) are themselves populated (`a_weight != nullptr`) determines which of this layer's
	// projections are actually adapted (design's third branch, B1c). Owned by the caller (the
	// same lifetime convention as every other pointer field on this struct) -- RunLayerLoop
	// never mutates it.
	const LayerAdapter* adapter = nullptr;

	// q/k/v/o projections (§6.2 step 2/6/7): GemmInt8Accumulate against the
	// weight matrix, then the WSC1 fold, one (identity, mult, shift) triple
	// PER OUTPUT CHANNEL (T-1666: `q_fold_identity`/`q_fold_mult`/
	// `q_fold_shift` and the k/v/o siblings below, replacing the single
	// scalar triple every projection used to reuse). q_proj/o_proj
	// then funnel (RequantChainChecked, via `q_site_constant`/
	// `o_site_constant`); k_proj/v_proj do NOT funnel -- they land at the
	// static per-head scale through `LandingRescale` (§8.1), using
	// `kv_landing_r_t`/`kv_landing_e_t` (KvLandingReciprocals'/KvLandingScales'
	// own (r_t, e_t) for this head).
	const int8_t* q_weight;
	const int8_t* k_weight;
	const int8_t* v_weight;
	const int8_t* o_weight;
	// q/k/v/o projections' WSC1 fold, ONE (identity, mult, shift) TRIPLE PER OUTPUT
	// CHANNEL (T-1666, replacing the single shared scalar triple every projection
	// used to reuse -- the real artifact's WSC1 data is per-output-channel and not
	// degenerate, driver log Claude/Brunel/t1664-...-2026-08-02.md §4). Same
	// ownership convention as ctx_fold_identity/_mult/_shift (below) and
	// q_bias/k_bias/v_bias (below): caller-resolved, no runtime length field --
	// the caller's own out_channels/kv_hidden_size argument (already threaded
	// through ProjectAndFunnel and the K/V landing loop) is what bounds every read.
	const int32_t* q_fold_identity;  // hidden_size
	const int32_t* q_fold_mult;      // hidden_size
	const int32_t* q_fold_shift;     // hidden_size
	const int32_t* k_fold_identity;  // num_key_value_heads * head_dim
	const int32_t* k_fold_mult;      // num_key_value_heads * head_dim
	const int32_t* k_fold_shift;     // num_key_value_heads * head_dim
	const int32_t* v_fold_identity;  // num_key_value_heads * head_dim
	const int32_t* v_fold_mult;      // num_key_value_heads * head_dim
	const int32_t* v_fold_shift;     // num_key_value_heads * head_dim
	const int32_t* o_fold_identity;  // hidden_size
	const int32_t* o_fold_mult;      // hidden_size
	const int32_t* o_fold_shift;     // hidden_size
	CarriedScale q_site_constant;
	CarriedScale o_site_constant;

	// C28's dynamic-arm bias, one array per q/k/v projection this layer's
	// ProjectAndFunnel/K-V-landing calls consume (§6.2 step 2, T-1656, D-SLM622).
	// nullptr means this projection carries no BIA1 entry at this layer -- the
	// composition is unchanged from today's unbiased path, matching the reference's
	// own `model.dynamic_biases.get(site)` returning None. Non-null arrays hold
	// hidden_size (q_bias) or num_key_value_heads * head_dim (k_bias/v_bias)
	// elements, in the SAME projection-output-channel order GemmInt8AccumulateRow
	// already produces for that projection.
	const int64_t* q_bias = nullptr;  // hidden_size, or nullptr
	const int64_t* k_bias = nullptr;  // num_key_value_heads * head_dim, or nullptr
	const int64_t* v_bias = nullptr;  // num_key_value_heads * head_dim, or nullptr
	// §8.1: per-(head, projection) K/V landing reciprocal/exponent, from
	// KvLandingReciprocals'/KvLandingScales' own per-head rows -- K and V are
	// separate arrays because §8.1 states the reciprocal/exponent as
	// per-(head, projection), and the reference reads two distinct artifact
	// keys per head (`kv_landing_reciprocals[f"{prefix}.k_head{head}"]` and
	// the matching `v_head{head}` entry). Caller-resolved, like every other
	// field on this struct: `num_key_value_heads` entries each (T-1654,
	// S3.8a -- corrected from "num_heads entries each", a transcription
	// error the MHA-degenerate fixture population never exposed, since
	// num_heads == num_key_value_heads at every such fixture; the
	// reference has always read these per KV head, not per query head,
	// `Tools/superslm_spike/dynamic_engine.py:369-370`).
	const int64_t* kv_landing_r_t_k;  // num_key_value_heads
	const int64_t* kv_landing_e_t_k;  // num_key_value_heads
	const int64_t* kv_landing_r_t_v;  // num_key_value_heads
	const int64_t* kv_landing_e_t_v;  // num_key_value_heads

	// The D-SLM57 per-head ctx_fold dispatch (§6.2 step 6): WSC1's
	// `layer{L}.ctx_fold` row for EACH head, applied via ApplyWeightScaleFold
	// (C24/C25) to the raw GemmProbQ15Accumulate context-accumulate value,
	// before the funnel (empty incoming, per §6.2 step 6's own text).
	// Caller-resolved, `num_heads` entries each.
	const int32_t* ctx_fold_identity;  // num_heads
	const int32_t* ctx_fold_mult;      // num_heads
	const int32_t* ctx_fold_shift;     // num_heads
	// The funnel's own site constant at the "attn_ctx" call -- ONE per layer
	// (§6.2 step 6's funnel call folds the whole wide row, every head's
	// contribution together, in a single call after every head's own
	// per-head fold above has already run) -- never per-head, unlike the
	// three fields directly above.
	CarriedScale ctx_fold_site_constant;

	CarriedScale attn_residual_site_constant;

	// C30's per-query i-exp derivation (§7.2 second limb; T-1655, D-SLM620): the
	// artifact's CompositionConstants["{prefix}.softmax_khead{kv_head}"] entry, one
	// per KV head. num_key_value_heads entries each -- indexed by kv_head = h / group,
	// never by query head h. RunLayerLoop derives, once per distinct kv_head per
	// token per layer, carried_scale_product([q_scale, {iexp_softmax_khead_m[kv_head],
	// iexp_softmax_khead_e[kv_head]}]) and then C30's (q_ln2, q_b, q_c) triple from the
	// result (IExpScaleConstants, intmath.h) -- replacing this struct's prior fixed
	// per-LAYER q_ln2/q_b_iexp/q_c_iexp substitution, which this design's own header
	// comment named as standing in for this not-yet-built derivation.
	const int64_t* iexp_softmax_khead_m = nullptr;  // num_key_value_heads
	const int64_t* iexp_softmax_khead_e = nullptr;  // num_key_value_heads

	const int32_t* mlp_norm_gain;  // hidden_size
	CarriedScale mlp_norm_site_constant;
	const int8_t* gate_weight;  // intermediate_size x hidden_size
	const int8_t* up_weight;    // intermediate_size x hidden_size
	const int8_t* down_weight;  // hidden_size x intermediate_size
	// gate/up/down projections' WSC1 fold, one triple per output channel
	// (T-1666); the q/k/v/o siblings of these fields are declared above.
	const int32_t* gate_fold_identity;  // intermediate_size
	const int32_t* gate_fold_mult;      // intermediate_size
	const int32_t* gate_fold_shift;     // intermediate_size
	const int32_t* up_fold_identity;    // intermediate_size
	const int32_t* up_fold_mult;        // intermediate_size
	const int32_t* up_fold_shift;       // intermediate_size
	const int32_t* down_fold_identity;  // hidden_size
	const int32_t* down_fold_mult;      // hidden_size
	const int32_t* down_fold_shift;     // hidden_size
	// T-1355: gate_proj and up_proj each funnel in their own right (§6.3 step
	// 10, "each with the funnel"), so each owns a KVC1 site constant --
	// `layerL.gate_proj.requant` and `layerL.up_proj.requant`. The declaring
	// pass omitted both, which is not a shortcut that still composes: MlpActSite
	// consumes a per-operand carried scale for each of gate and up, and there is
	// no other field either scale could come from. Grounded at the reference,
	// which emits exactly these two chain records
	// (Tools/superslm_spike/dynamic_engine.py's `{prefix}.gate_proj.requant` and
	// `{prefix}.up_proj.requant`), never one shared record.
	CarriedScale gate_site_constant;
	CarriedScale up_site_constant;
	CarriedScale mlp_act_site_constant;
	CarriedScale down_site_constant;
	CarriedScale mlp_residual_site_constant;
};

// S3a's layer loop (§9.3, §11 S3.5): advances `seq` through up to
// `layer_budget` layers of its CURRENT token, composing, per layer and in
// this order (§6.2/§6.3): attn_norm (RmsNormSite) -> q/k/v projections
// (GemmInt8Accumulate + ApplyWeightScaleFold; q funnels through
// RequantChainChecked, k/v land at the static per-(head, projection) scale
// through `clamp(LandingRescale(...), -127, 127)`, §8.1 -- the clamp is this
// call site's own, per `LandingRescale`'s header, and reuses `ClampRopeCode`
// for it, the same pinned `[-127, 127]` code range) -> RoPE on q and k
// (RopeApplySite) -> attention
// (GemmInt8Accumulate for the score row, SoftmaxRowQ15, GemmProbQ15Accumulate
// for the context accumulate -- §6.2 step 5; no separate named site exists
// for this composition anywhere in this tree, so RunLayerLoop is where it is
// first composed, matching §3.4 of the plan's own text: "No C++ code in
// D:\SuperSLM performs... an attention pass... S3a is the first slot with a
// forward pass in it") -> the ctx_fold dispatch (ApplyWeightScaleFold) ->
// o_proj (as q_proj) -> attn_residual (ResidualReconcileSite) -> mlp_norm ->
// gate/up projections -> MlpActSite -> down_proj -> mlp_residual
// (ResidualReconcileSite) -> the resume-point update (`seq.layer_index += 1`
// per completed layer).
//
// Two contracts decided at §9.3, not invented here:
//   - `layer_budget == 0` is `InvalidLayerBudget`, and `seq` is left
//     bit-identical to its pre-call state -- `hidden_codes`, `hidden_scale`,
//     AND `layer_index` are all untouched. A step that consumes a call,
//     advances nothing, and returns "pending" is a host-visible livelock
//     (§11's reject-over-silently-degrade law).
//   - There is no `all` sentinel. "Every layer" is spelled
//     `layer_budget == num_hidden_layers`.
//
// `context_cap < 1` is `InvalidContextCap`, checked before the workspace size
// is computed from it -- `context_cap` sizes the required K/V workspace
// (`num_hidden_layers * context_cap * num_key_value_heads * head_dim * 2`),
// and every value outside `[1, INT64_MAX]` (zero or negative) is invalid for
// the same reason: neither is a legitimate count of cache positions. One
// check covers both, because both violate the identical domain requirement.
//
// A layer commits atomically. Every checked call between `attn_norm` and
// `mlp_residual` (inclusive) can reject; on ANY rejection inside a layer,
// `seq` is left exactly as it was before that layer's attempt started --
// `hidden_codes`, `hidden_scale`, AND `layer_index` all untouched, the same
// property the `layer_budget == 0` cell already requires one level up. §9.3
// states the resume point is a layer boundary and nowhere else, so no
// intermediate, mid-layer state may ever become visible in `seq`; a rejection
// therefore leaves the sequence ready to re-attempt the WHOLE layer, never a
// partially-applied one.
//
// The residual (`seq.hidden_codes`/`hidden_scale`) lives in `seq`'s OWN
// state, never in `workspace` -- poison-filling `workspace` between two
// calls that together advance the SAME sequence across a layer boundary must
// not change the final output (master plan §8.2/W1; this sub-slot's own red
// cell).
//
// This function performs NO WGT1/KVC1-by-name resolution (see LayerWeights'
// own header comment) and reads `context_cap`-worth of K/V state through
// S3.7's own `KeyRow`/`ValueRow` accessors, declared just below -- every
// position 0..context_cap-1 is reachable, not only position 0.
//
// `layers` has `num_hidden_layers` entries. `rope_tables` is the loaded ROP1
// view, shared across every layer (ROP1 is model-wide, never per-layer).
// `site_prefix`/`token_index`/`trace_hook_state` are forwarded into every
// internal funnel call, with each site's own per-layer name appended (e.g.
// `"layer0.attn_norm"`) -- the same convention RmsNormSite's own doc
// comment states, extended across the whole loop.
//
// S3.7 (§11 S3.7 "Fail fast on a full cache"): on EVERY call, when
// `seq.context_length >= context_cap`, this returns `KvCapacityExhausted`
// before any layer runs, `seq` left bit-identical -- "defined and resumable"
// (the status enum's own comment): the sequence is left valid and
// query-able, and a second call at the SAME state rejects the identical way
// rather than corrupting the handle. Checked without regard to
// `seq.layer_index`: a caller-restored `seq` (§13 dim 9 pins the struct as
// addressable as a unit, for exactly this save/restore use) is not bound to
// have come from this loop's own resume path, so a full cache at a non-zero
// `layer_index` is rejected here rather than reaching the landing write
// (Poirot 0d64462 review, Critical 1). S3a builds no eviction or truncation
// remedy for a full cache; that stays S4's.
SslmForwardStatus RunLayerLoop(SequenceLayerState& seq, const LayerWeights* layers,
                                 uint32_t num_hidden_layers, uint32_t layer_budget,
                                 size_t hidden_size, size_t head_dim, size_t num_key_value_heads,
                                 size_t intermediate_size, int64_t context_cap,
                                 const SslmTensorManifest& rope_tables, uint8_t* workspace,
                                 size_t workspace_size, std::string_view site_prefix = {},
                                 size_t token_index = 0,
                                 SslmTraceHookState* trace_hook_state = nullptr);

// T-1894 (design Sec31.2.1, round 4/D-SLM2426): the fused K-landing
// selector's own type. `const char* -> bool` is a standard boolean
// conversion and outranks `const char* -> std::string_view`'s user-defined
// conversion, so a caller passing a string literal at this overload's
// selector position (13, ahead of `site_prefix`) used to silently resolve to
// THIS overload -- enabling the fused path and discarding the label --
// rather than the pre-existing overload's own `string_view site_prefix`
// (T-1901 Significant 1). A scoped enum has no implicit conversion from
// `const char*`, `bool`, or `int`: a string literal at that position now
// finds no viable conversion here and correctly resolves to the 15-parameter
// overload instead, and a caller reaching for a bare `true`/`false` gets a
// hard compile error at both overloads rather than a silent semantic
// accident. Scoped to this ONE overload-resolution position --
// `RunGreedyDecodeLoop`'s own new parameter and `SslmModelView::
// option_g_fused_k_landing` (both above/model.h) stay `bool`, since neither
// is an overload-resolution site.
enum class OptionGKLandingMode : uint8_t { kLegacy = 0, kFused = 1 };

// T-1899 (Curie, red suite for T-1894 -- T-1822 design Sec31.2's production
// Option-G build). The production RunLayerLoop overload T-1894's own build
// adds: threads `option_g_k_landing_mode` (read by the CALLER from
// `SslmArtifact::OptionGFusedKLandingEnabled()`, design Sec31.2.1/Sec31.2.3 --
// "the artifact reference RunLayerLoop already threads for every other
// constant it reads") as an explicit parameter, rather than either (a)
// changing the EXISTING 15-parameter overload's own signature/body -- both
// are Brunel's real implementation work, not this suite's declare-only
// contract extension -- or (b) reaching for the disposable
// T-1891 spike's own env-var mechanism, explicitly retired by D-SLM2351/
// D-SLM2355. When `option_g_k_landing_mode == OptionGKLandingMode::kFused`,
// the two K/V-landing call sites rotate each head's wide pre-landing K
// accumulator pairwise (RopeApplyPairWide, below) at this token's own
// position, THEN land once (LandingRescale, already shipped) -- replacing
// the post-landing RopeApplySite read/rotate and its write-back loop (the
// EXISTING overload's own §6.2-step-3 K rotation) for K only; V and Q are
// unmodified by this parameter (design Sec31.2.2's own construction, carried
// from T-1891 §2, confirmed sound by T-1892). Declared, not defined -- the
// same declare-and-stub convention this header's own RopeApplySite/
// MlpActSite history already established (T-1839's own signature-change
// precedent, Claude/Curie/t1832-...-red-suite-test-design-2026-08-08.md
// Sec9.5, applied here to a NEW overload rather than an edited existing
// signature so the EXISTING, already-shipped, already-tested 15-parameter
// overload is left entirely untouched by this suite). The parameter's own
// TYPE changed from `bool` to `OptionGKLandingMode` in T-1894's own build
// round 4 (T-1901 Significant 1/D-SLM2418, D-SLM2426) -- see that enum's own
// comment, above.
SslmForwardStatus RunLayerLoop(SequenceLayerState& seq, const LayerWeights* layers,
                                 uint32_t num_hidden_layers, uint32_t layer_budget,
                                 size_t hidden_size, size_t head_dim, size_t num_key_value_heads,
                                 size_t intermediate_size, int64_t context_cap,
                                 const SslmTensorManifest& rope_tables, uint8_t* workspace,
                                 size_t workspace_size, OptionGKLandingMode option_g_k_landing_mode,
                                 std::string_view site_prefix = {}, size_t token_index = 0,
                                 SslmTraceHookState* trace_hook_state = nullptr);

// T-2147 (design §15.1/§15.2/§15.3, D-SLM3479/D-SLM3481/D-SLM3482/D-SLM3483): the chunk-batched
// prefill entry point -- runs `chunk_tokens` already-embedded tokens through every layer,
// streaming each layer's weight matrix ONCE across the chunk for the seven weight-heavy
// projection sites (q/o/gate/up/down/k/v), instead of once per token. Every non-GEMM step stays
// per-token/per-position, in position order, unchanged -- bit-identical to `RunLayerLoop` called
// once per token, by construction (design §15.2; see forward_sites.cpp's own header comment on
// this function for the full contract). `hidden_codes_chunk` is `chunk_tokens` rows of
// `hidden_size` int8 codes, row-major, already embedded by the caller; `hidden_scales` is one
// CarriedScale per row. `context_length_start` is the position token 0 of the chunk lands at.
// `kv_saturation_count` is the same per-sequence running counter
// `SequenceLayerState::kv_saturation_count` is. There is no partial/resumable layer_budget in
// this path -- every layer runs to completion for every token, or the call returns non-Ok.
SslmForwardStatus RunLayerLoopChunkBatched(int8_t* hidden_codes_chunk, CarriedScale* hidden_scales,
                                            size_t chunk_tokens, const LayerWeights* layers,
                                            uint32_t num_hidden_layers, size_t hidden_size,
                                            size_t head_dim, size_t num_key_value_heads,
                                            size_t intermediate_size, int64_t context_cap,
                                            int64_t context_length_start,
                                            const SslmTensorManifest& rope_tables,
                                            uint8_t* workspace, size_t workspace_size,
                                            bool option_g_fused_k_landing,
                                            uint64_t* kv_saturation_count,
                                            std::string_view site_prefix = {},
                                            SslmTraceHookState* trace_hook_state = nullptr);

// T-1899 (design Sec31.2's own "int64-input, __int128-intermediate sibling of
// the RoPE pair primitive, Q2.30 tables unchanged" -- Sec12 "Wide-RoPE
// overflow domain"). The rotated wide pair -- matches `RopePair`'s own shape
// (this file, above) but at int64 INPUT width (`RopePair`'s own inputs are
// int32; this primitive's `x`/`y` are the wide, pre-landing K accumulator
// values `RunLayerLoop`'s K/V-landing block already carries at int64,
// unnarrowed). Defined in forward_sites.cpp, on the same portable-128-bit
// substitution `LandingRescale`'s own C27 composite already uses (this
// toolchain has no native `__int128`) -- Brunel's build, not this suite's.
struct RopePairWide {
	int64_t x = 0;
	int64_t y = 0;
};

// Combination and rounding are IDENTICAL to `RopeApplyPair` (this file,
// above): `(x*cos - y*sin, x*sin + y*cos)`, ONE C3 (ties-away-from-zero)
// rounding at ROPE_FRAC_BITS -- only the input width and the intermediate's
// own width differ (int64/128-bit here vs. int32/int64 there).
// `cos_q30`/`sin_q30` are the SAME Q2.30 ROP1 table entries RopeApplySite
// already reads (|·| <= ROPE_ONE); this primitive imposes no additional
// domain on them. The result is UNCLAMPED -- clamping to the activation
// format is the caller's, through the EXISTING, already-shipped
// LandingRescale+ClampRopeCode pair (design Sec31.2.2's own construction);
// this primitive is not a second clamp. `*out_in_domain` is false (refuse,
// not wrap) whenever either rotated component's ROUNDED value -- the value
// this function returns, T-1892 Minor 1's own correction: checked on the
// value AFTER C3 rounding, never some unrounded "true" value the function
// never materializes -- does not fit int64_t
// (`OptionGWideRopeMagnitudeOutOfDomain`, checked_chain_funnel.h, this suite).
// Declared, not defined -- the design's own new primitive; Brunel's build.
RopePairWide RopeApplyPairWide(int64_t x, int64_t y, int32_t cos_q30, int32_t sin_q30,
                                bool* out_in_domain);

// S3.7 (§9.4, §11 S3.7 "The K/V store's real layout, and the accessor"): the
// K/V store is per-(layer, head)-major, position-minor --
// `offset(kv_head, position, d) = kv_head * context_cap * head_dim +
// position * head_dim + d`, inside the K half (starting at `workspace`, at
// `layer * context_cap * num_kv_heads * head_dim * 2`) or the V half (the
// same base plus `context_cap * num_kv_heads * head_dim`). `KeyRow`/`ValueRow`
// return a pointer to `head_dim` contiguous int8 codes at that address;
// `MutableKeyRow`/`MutableValueRow` return the same address non-const, for
// the landing write. `workspace`/`layer`/`context_cap`/`num_kv_heads`/
// `head_dim` are the identical values a caller already has from its own
// RunLayerLoop call over the same sequence -- no bounds checking is
// performed here (`position < context_cap` and `kv_head < num_kv_heads` are
// the caller's own responsibility, exactly as RunLayerLoop's guards already
// establish before any accessor call it makes).
//
// T-1654 (S3.8a): this parameter is named `num_kv_heads`, not `num_heads` --
// it is always the KV-head count, never the query-head count RunLayerLoop's
// own `num_heads` local names. The old name invited exactly this sub-slot's
// own defect: a call site written against a parameter named `num_heads` has
// every reason to pass the query head count. The arithmetic these four
// functions perform is unchanged; only the name is corrected.
const int8_t* KeyRow(const uint8_t* workspace, uint32_t layer, int64_t context_cap,
                      size_t num_kv_heads, size_t head_dim, size_t kv_head,
                      int64_t position) noexcept;
const int8_t* ValueRow(const uint8_t* workspace, uint32_t layer, int64_t context_cap,
                        size_t num_kv_heads, size_t head_dim, size_t kv_head,
                        int64_t position) noexcept;
int8_t* MutableKeyRow(uint8_t* workspace, uint32_t layer, int64_t context_cap,
                       size_t num_kv_heads, size_t head_dim, size_t kv_head,
                       int64_t position) noexcept;
int8_t* MutableValueRow(uint8_t* workspace, uint32_t layer, int64_t context_cap,
                         size_t num_kv_heads, size_t head_dim, size_t kv_head,
                         int64_t position) noexcept;

// --- S3.6: the head and the greedy decode loop (SuperSLM_S3a_WalkingSkeleton_
// Plan.md §11 S3.6; §9.1; master plan §6.4; C16, D-SLM35 row C16). This is
// the "host-facing entry point" LayerWeights' own header comment above names
// as S3.6/S3.7's scope or unassigned -- claimed here. ----------------------

// Master plan §6.4 step 15 ("Logits"): `GemmInt8Accumulate(final_codes,
// weights[tie ? "embed" : "lm_head"])`, then `NarrowRowChecked`
// (checked_chain_funnel.h's second entry point) -- never a bare
// `NarrowAccumulatorToI32`. `final_codes` is step 14's own output
// (`RmsNormSite` under site "final_norm"; its OWN carried scale is
// discarded -- "the head consumes codes", §6.4's own text, so this function
// takes no incoming CarriedScale at all). `head_weights` is a flat,
// row-major `[vocab_size, hidden_size]` int8 matrix -- tied-vs-dedicated is
// the CALLER's resolution (CFG1's `tie_word_embeddings`), matching every
// other site function in this file: EmbedEntry's `embed_weights` is the
// identical kind of raw, caller-supplied pointer, never a by-name tensor
// lookup, and this function performs no WGT1/embed-tensor-naming resolution
// of its own (LayerWeights' own header comment states the same declared-scope
// convention for the per-layer weights it carries).
//
// Performs, in this order and no other:
//   1. GemmInt8AccumulateRow(final_codes, head_weights, hidden_size,
//      vocab_size, wide_logits) -- matmul.h's already-shipped, order-free
//      int64 accumulation (design §4/§11 item 4), one row per vocabulary
//      entry.
//   2. NarrowRowChecked(wide_logits, vocab_size, out_logits) -- C35's
//      asymmetric int32 domain check (checked_chain_funnel.h); on rejection
//      returns LogitNarrowingOverflow and leaves `out_logits` untouched (the
//      funnel's own convention, extended here to this site). §6.4's own
//      derivation: the wide row's bound is `hidden_size * 127 * 127`; at
//      `hidden_size == 1536` that is 24,774,144, ~86.7x under 2^31 --
//      comfortable headroom at the model's own geometry, and the reason the
//      check is PERFORMED rather than assumed is that `hidden_size` is
//      per-artifact (§6.4) -- a smaller or adversarially-scaled artifact is
//      not bound by the 1536-wide headroom this comment cites for context.
//
// `final_codes` has `hidden_size` elements; `head_weights` has
// `vocab_size * hidden_size` elements, row-major (row v is the v-th
// vocabulary logit's own weight vector). `wide_logits` is caller-owned
// scratch, `vocab_size` elements (the composition's own wide intermediate,
// matching this file's existing convention of a caller-owned wide-row
// buffer for every funnel-adjacent site). `out_logits` receives `vocab_size`
// int32 logits on Ok and is untouched on rejection.
SslmForwardStatus LogitsSite(const int8_t* final_codes, size_t hidden_size,
                              const int8_t* head_weights, size_t vocab_size,
                              int64_t* wide_logits, int32_t* out_logits);

// C16 (master plan §6.8 row C16, D-SLM35; §6.4 step 16): the pinned argmax
// tie-break -- LOWEST token index. Caller-ensures `n >= 1` (the same
// caller-ensures convention as FloorDivI64/ShiftByMax -- an empty logit row
// is a caller defect, not a runtime-checked one; a conformant artifact's
// `vocab_size` is load-time rejected at 0, so no production call ever passes
// `n == 0`; `MaxAbsReduceWide` does not share this convention -- it is total
// over `n`, defined and safe at `n == 0`, see intmath.h). Scans left to
// right and keeps the
// FIRST (lowest-index) strict maximum -- a later element equal to the
// running maximum never replaces it, which is what makes the tie-break
// lowest-index rather than last-write-wins or highest-index.
int32_t ArgmaxLowestIndexTieBreak(const int32_t* logits, size_t n);

// G5-2/G5-5 (T-2132, Brunel): the ONE mask-application-before-argmax primitive design Sec4's
// architecture table names ("Table lookup + bitmask AND, int32 logits, pre-argmax") -- every
// caller that needs a schema-constrained token, CPU (`sslm_decode_step`'s own masked-argmax
// step, src/sslm_abi.cpp) or GPU (G5-5's parity bridge, src/gpu/gpu_1p0.cpp), calls this exact
// function on its own hidden-state-derived logits, never a parallel reimplementation --
// StandardsDocument.md Sec5.4's "one implementation, not two copies that could drift"
// discipline, applied here the same way D-SLM3352 already applied it to weight packing.
// Extracted from sslm_abi.cpp's own file-local ApplyMaskAndArgmaxImpl (G5-2) so a second TU
// (gpu_1p0.cpp) can call the identical bits rather than a lookalike -- bit-parity between the
// CPU and GPU constrained-decode paths is achievable BY CONSTRUCTION only if both feed their
// own (bit-identical, by the base kernel set's own already-proven GPU parity) hidden state
// through this SAME function, never through two independently-authored copies of it.
//
// `mask` is `entry`'s own page for the sequence's current DFA-walk-state (state_count *
// mask_page_bytes bytes total per schema, schema_masks.h) -- caller-selects the right page;
// this function does the table lookup ONLY in the sense of "look up whether bit t is set,"
// never in the sense of picking which page. Masked-out positions (bit not set) are forced to
// INT32_MIN so ArgmaxLowestIndexTieBreak's own lowest-index tie-break can never select them.
// NO defensive check for an all-masked page (D-SLM40's own positive requirement, design Sec3/
// Sec7 dim11) -- an all-zero mask degrades to picking the lowest index, exactly dim11's own
// negative-control cell. Mutates `logits` in place; caller-ensures `vocab_size >= 1` (the same
// caller-ensures convention ArgmaxLowestIndexTieBreak's own `n >= 1` already documents).
void ApplyMaskAndArgmax(int32_t* logits, const uint8_t* mask, int32_t vocab_size,
                          int32_t* out_token_id);

// §9.1's two terminal reasons a greedy decode loop stops for a reason OTHER
// than a rejection. Distinct from SslmForwardStatus (checked_chain_funnel.h),
// which names rejections; a loop that reaches either of these two reasons
// is not rejected, it is DONE, and §11 S3.6's own gate text ("terminates on
// each of the two stop conditions with a status naming which fired")
// requires the two to be distinguishable from each other and from Ok.
enum class SslmDecodeStopReason {
	MaxTokensReached = 0,  // the host's max-token count was reached with no stop id matched
	StopTokenMatched = 1,  // the produced token is a member of the host's stop-id set
};

// §9.1's greedy decode loop: "Prefill the prompt in one call; then repeat:
// one forward over the last token, argmax with C16's tie-break, append,
// until a stop condition." Prefills `prompt_tokens` (each embedded via
// EmbedEntry and run through RunLayerLoop with `layer_budget ==
// num_hidden_layers` -- §9.3's own "every layer" spelling, one WHOLE token
// per call; this loop's own scope is whole-token stepping, never the
// layer-range micro-step RunLayerLoop itself also supports), then repeats:
// embed the last produced (or last prompt) token, RunLayerLoop, RmsNormSite
// under site "final_norm" (its own carried scale discarded, §6.4 step 14),
// LogitsSite, ArgmaxLowestIndexTieBreak, append -- until a stop condition.
//
// Every host-supplied id -- EVERY element of `prompt_tokens` AND EVERY
// element of `stop_ids` -- is validated against `[0, vocab_size)` on the
// IDENTICAL rule EmbedEntry already applies to a single token (§9.1: "on the
// same rule as a host-supplied prompt token"): rejected with
// TokenIdOutOfRange. Every stop id is checked BEFORE the loop starts (a stop
// id the argmax can never produce is a silently-inert stop condition, never
// discovered by running the loop, §9.1's own reasoning) and every prompt id
// is checked as each is embedded, in encounter order, so a bad prompt id
// past a good prefix still leaves everything at and after it untouched.
// `stop_count == 0` is legal and means "run to `max_new_tokens`" (§9.1: "An
// empty set is legal"). The produced token is appended to `out_tokens`
// BEFORE the stop-id test runs (§9.1: "The produced token is appended before
// the stop test, so a stop token appears in the output sequence"), so a
// matched stop token is the LAST element of `out_tokens` on
// `SslmDecodeStopReason::StopTokenMatched`, never withheld.
//
// `out_logit_rows` (caller-owned, `out_tokens_capacity * vocab_size`
// elements) receives, at row i, the full int32 logit row LogitsSite computed
// immediately before the i-th produced token was selected -- the raw
// material for §10.1's final-logit digest (decode_digest.h); this function
// does not itself compute either digest (§10.2's exclusion table is a claim
// about what participates in the arithmetic and the hash alike, and keeping
// digesting a separate, pure function of the two output arrays this call
// already produces is what makes that claim checkable rather than asserted).
//
// On Ok, `*out_stop_reason` names which of the two conditions fired,
// `*out_tokens_produced` is the number of NEWLY produced tokens written to
// `out_tokens`/`out_logit_rows` (`prompt_tokens` is never echoed into
// `out_tokens`), and `seq` reflects the sequence's state after the last
// successfully produced token. On any rejection (a bad prompt id, a bad stop
// id, or any status RunLayerLoop/LogitsSite themselves can return),
// `out_tokens`, `out_logit_rows`, `*out_tokens_produced`, and
// `*out_stop_reason` are all left untouched, and `seq` is left in the state
// RunLayerLoop's OWN atomic-layer contract already guarantees (§11 S3.5: a
// rejected layer leaves `seq` exactly as it was before that layer's own
// attempt).
//
// `out_tokens`/`out_logit_rows` have `out_tokens_capacity` (rows, for the
// latter) elements -- caller-ensures `out_tokens_capacity >= max_new_tokens`
// (a workspace-sizing question this call does not scope, matching this
// file's existing caller-ensures convention for buffer sizes throughout).
// This function performs no per-layer WGT1/KVC1-by-name resolution of its
// own (LayerWeights' own header comment); `embed_weights`/`head_weights` are
// the same kind of raw, caller-resolved pointer EmbedEntry/LogitsSite above
// already take.
//
// S3.7 (§14.4, §11 S3.7 "The int16 narrowing"): `kv_precision` is checked
// FIRST, before `seq`, `workspace`, or any output parameter is touched, and
// before any token is embedded -- `SslmKvPrecision::Int16` is a defined,
// load-legal CFG1 value (§14.4 is a declared quality narrowing, not a
// hostile-input rejection) rejected here with `KvPrecisionUnsupported`;
// `SslmKvPrecision::Int8` is unaffected and proceeds exactly as before this
// parameter existed. T-1900 fix round 3: this parameter's own `= Int8`
// default is gone -- not because anything was found wrong with defaulting
// IT, but because C++ forbids a defaulted parameter from preceding a
// required one, and `option_g_fused_k_landing` (below) is made required by
// this same round. Every real call site already passes `kv_precision`
// explicitly (`tools/sslm_generate.cpp`, `tools/sslm_layer_trace.cpp`,
// `tests/test_main.cpp`'s `TestOptionGSelectionDispatch_EndToEndProductionPath`)
// except the two `tests/test_main.cpp` sites this round also makes explicit
// (`DecodeLoopCallFixture::Run`, `TestRunGreedyDecodeLoopRejectsInt16Kv
// PrecisionBeforeAnythingElse`'s positive control) -- this is a mechanical
// consequence, not a second independent decision.
// T-1894 (design Sec31.2.1, round 4/D-SLM2423, link 3 of 5): trailing
// REQUIRED parameter -- no default. The round-4 build (T-1901 fix round 1)
// gave this a defaulted `= false`, following `kv_precision`'s own precedent
// so every existing call site would compile unchanged; the round-2
// confirmation (T-1901 New 1) found that precedent wrong for THIS parameter:
// a default here is silently-wrong-by-default rather than silently-inert,
// because a caller that omits it does not get legacy-forever (harmless) --
// it gets whatever `sslm_layer_trace.cpp` actually got, a SECOND consumer of
// this function that kept compiling, kept passing its own self-check (its
// production leg and its manual-replay leg both silently ran legacy
// together, so bit-for-bit agreement proved nothing about which mode either
// leg was in), and shipped a flags=1 trace indistinguishable from a flags=0
// one. StandardsDocument S4: where a rule can be made structural, make it
// structural -- removing the default turns "the next omitted caller" into a
// compile error instead of a second silent-agreement instrument, which a
// reviewer-named CI assertion over call sites (the alternative considered)
// would not: a text-scan checker is exactly the class of structural-absence
// check this same build round already had to hunt a blind spot out of
// (`check_no_optionG_fused_site_skip.py`, this ticket), where the compiler's
// own overload resolution has none. The caller (`tools/sslm_generate.cpp`,
// link 4, and now `tools/sslm_layer_trace.cpp`'s own production leg) passes
// `model_view.option_g_fused_k_landing` (link 1/2) explicitly, the same way
// it already passes `model_view.config.kv_precision` as this signature's own
// previous last argument. Threaded through to the `RunWholeToken` lambda's
// own `RunLayerLoop` call inside this function's definition (link 5,
// `forward_sites.cpp`), which selects the 16-parameter overload's
// `OptionGKLandingMode` explicitly.
SslmForwardStatus RunGreedyDecodeLoop(
    SequenceLayerState& seq, const LayerWeights* layers, uint32_t num_hidden_layers,
    size_t hidden_size, size_t head_dim, size_t num_key_value_heads, size_t intermediate_size,
    int64_t context_cap, const SslmTensorManifest& rope_tables,
    const int32_t* prompt_tokens, size_t prompt_len,
    const int8_t* embed_weights, CarriedScale embed_site_constant,
    const int32_t* final_norm_gain, CarriedScale final_norm_site_constant,
    const int8_t* head_weights, int32_t vocab_size,
    const int32_t* stop_ids, size_t stop_count, size_t max_new_tokens,
    uint8_t* workspace, size_t workspace_size,
    int32_t* out_tokens, int32_t* out_logit_rows, size_t out_tokens_capacity,
    size_t* out_tokens_produced, SslmDecodeStopReason* out_stop_reason,
    SslmKvPrecision kv_precision, bool option_g_fused_k_landing);

}  // namespace superslm

#endif  // SUPERSLM_FORWARD_SITES_H

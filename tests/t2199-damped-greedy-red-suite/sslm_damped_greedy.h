/* sslm_damped_greedy.h -- T-2199 (Curie) red suite: the declared surface for damped greedy
 * decoding's Phase A (the n-gram anti-LM) and Phase C (the shared decode-step primitive,
 * DampedGreedyScoreAndArgmax) -- the two build phases this suite covers, per the conductor's
 * scoping brief.
 *
 * Transcribed from the design of record, Claude/Plans/superslm-1p2-fsd-plan-2026-08-19.md
 * (Wizard repo) at commit 5a7f80e251 -- Sec7 (the mechanism), Sec7.2 (the anti-LM, its own
 * exact-key-lookup determinism argument and its normalized-mixing-weight [0, 2^15] clamp),
 * Sec7.3 (the two numeric surfaces), Sec7.4 (top-k selection, corrected tie-break: mask
 * status before index), Sec7.5 (score/combine, the INT64_MIN masked-score floor, the >>15
 * Q15 rescale), Sec7.6 (final selection), and Sec8 Phase A/Phase C's own decomposition into
 * A1/A2/C1/C2/C2a/C3. Every citation in a function's own comment below names the exact plan
 * subsection it was transcribed from, per this repo's own T-2130/T-2112 promoted-header
 * convention (this directory's copy is canonical for this suite; regenerate from the design
 * of record if Sec7/Sec8 change).
 *
 * RED BY LINK: every symbol below is declared, not defined, anywhere in this repo (grep
 * confirmed at authoring commit main@071c5f3 -- no `AntiLm*`, `FsdTopK`, `TopKRenormalizeQ15`,
 * or `DampedGreedyScoreAndArgmax*` symbol exists under these names in src/ or include/). Every
 * cell in this suite that calls one of these symbols therefore compiles clean against this
 * header and fails to LINK (LNK2019) until the build seat lands Phase A/Phase C's own .cpp --
 * see build_link_red.bat.
 *
 * Scope boundary, stated so a reader does not look for it elsewhere in this header: Phase B
 * (the vocab-row logit scale artifact/converter work), Phase B0 (the empirical calibration
 * capture instrument), Phase D (ABI surface, both decode entry points, the GPU bridge), and
 * Phase E (the confirmatory acceptance run) are OUT of this suite's scope by the commission's
 * own brief -- this header declares only the Phase A / Phase C primitives those later phases
 * will wire into the ABI, not the ABI surface itself. `(q_ln2, q_b, q_c)` and `alpha_q15` are
 * therefore taken as plain function parameters here, exactly as Sec7.1's own "Inputs" list
 * states them, never read from an `sslm_decode_params` this suite does not touch.
 */
#ifndef SSLM_T2199_DAMPED_GREEDY_H
#define SSLM_T2199_DAMPED_GREEDY_H

#include <cstddef>
#include <cstdint>

namespace superslm {

// =====================================================================================
// Phase A -- the n-gram anti-LM (plan Sec7.2, Sec8 Phase A1/A2). No engine dependency;
// buildable and testable standalone, per the plan's own phase-ordering rationale.
// =====================================================================================

// Opaque per-sequence state -- "a warm-object class, not fresh-per-call" (Sec9 dim1). Never
// defined in this header; only AntiLmCreate/AntiLmDestroy know its real shape once Phase A
// lands.
class AntiLmState;

// Constructs a fresh anti-LM instance mixing orders 1..max_order (Sec7.2's own "smoothed
// n-gram model over orders 1..N"). Ownership: exactly one AntiLmDestroy per AntiLmCreate.
[[nodiscard]] AntiLmState* AntiLmCreate(int max_order);
void AntiLmDestroy(AntiLmState* state);

// Appends one token to the sequence's own generated prefix (x<t, never the prompt -- Sec7.2's
// own citation of Loftus's verified reference and the paper's own statement). Called once per
// emitted token; mutates every order's count table. Strictly sequential, in generation order
// (Sec7.2's own determinism argument for `update`).
void AntiLmUpdate(AntiLmState* state, int32_t token);

// p_omega(v) in Q15 (kProbFracBits, intmath.h) for each of the `k` candidate tokens named in
// `candidates` -- Sec7.2's own "two operations only... penalize(query, candidates)... returns
// p_omega(v) for the exactly-k candidate tokens under consideration, never the whole
// vocabulary." Exact-key lookup only: this function must never iterate the state's own count
// tables in their container order (Sec7.2's own determinism argument -- "removes the entire
// class of hazard... by construction"). Every output is a normalized convex combination of
// each order's own Q15 count ratio (weights w_i/(sum w_i), beta=0.9 decay) and is CLAMPED to
// [0, 1 << kProbFracBits] as a designed domain guarantee of `penalize`'s own implementation
// (Sec7.2, "CORRECTED 2026-08-20 (strike 7...)"), not an incidental property of the
// arithmetic -- Sec9 dim7's own GATE cell for this exact clamp is pinned against this
// function.
void AntiLmPenalize(const AntiLmState* state, const int32_t* candidates, std::size_t k,
                     int64_t* out_p_omega_q15);

// Bytes currently retained across every order's count table -- Sec7.2's own "Memory is
// O(distinct n-grams seen so far) <= O(generation length)" claim (Sec9 dim7's mapped cell,
// via Phase A2's own memory-growth red-suite cell, Sec8).
std::size_t AntiLmRetainedBytes(const AntiLmState* state);

// =====================================================================================
// Phase C -- the shared decode-step primitive, DampedGreedyScoreAndArgmax (plan Sec7.4-7.6,
// Sec8 Phase C1/C2/C2a/C3).
// =====================================================================================

// Phase C1 (Sec7.4): selects exactly `k` positions from `masked_row` (an int32 row whose
// masked positions have already been narrowed/floored to INT32_MIN by the caller, per the
// plan's own "mask first" convention, Sec6) -- descending by row value; ties broken FIRST by
// mask status read from `mask_bits` (an unmasked position always outranks a masked one at
// equal value -- the cycle-9 temper's own extension, closing the selection-step exclusion a
// legal INT32_MIN-valued token would otherwise suffer), THEN ascending token index within
// positions of the same mask status. `mask_bits` is the packed schema-mask array
// (1 bit/vocab position, LSB-first per byte) already used at the masking-half call site
// (checked_chain_funnel.h's own convention); an all-ones mask_bits array is free-text mode's
// own no-op form (Sec6). Writes exactly `k` vocabulary indices to `out_indices`, in the order
// selected.
void FsdTopK(const int32_t* masked_row, const uint8_t* mask_bits, int32_t vocab_size,
             int32_t k, int32_t* out_indices);

// Phase C2 (Sec7.3 surface 2, Sec5.5): `TopKRenormalizeQ15` -- q_theta(v) for exactly the `k`
// indices `FsdTopK` selected, built from the same certified ShiftByMax/IExpConstruct/
// IExpEvaluate sub-primitives SoftmaxRowQ15 (intmath.h) already uses, restricted to k
// elements -- "not a new numeric domain, its own domain proof proportionately smaller." Row
// values at `indices` are widened int32->int64 before the multiply, matching this codebase's
// own house style. Returns `all_well_formed` exactly as `SoftmaxRowQ15` does: false means at
// least one of the k gathered elements hit a per-element domain refusal (kBadQ/kBadQLn2/
// kBadQB, or the evaluated i-exp value exceeded the recomputed per-element ceiling `m`) --
// Sec7.5's own "Policy on a TopKRenormalizeQ15 runtime refusal" paragraph is what the CALLER
// (DampedGreedyScoreAndArgmax, below) does with a false return, not this function's own
// concern. `out_q15[i]` is written for every element regardless of the returned bool
// (SoftmaxRowQ15's own "written whenever the decomposition is well-formed" contract, mirrored
// here).
[[nodiscard]] bool TopKRenormalizeQ15(const int32_t* row, const int32_t* indices,
                                       std::size_t k, int64_t q_ln2, int64_t q_b, int64_t q_c,
                                       int64_t* out_q15);

// Phase C3 (Sec7.5-7.6): the full score/combine/select mechanism for one decode step.
//   1. FsdTopK selects the k candidate indices.
//   2. TopKRenormalizeQ15 computes q_theta(v) for the k candidates. A refusal (returned
//      false) is propagated to the caller (`*out_refused = true`, `*out_token` untouched) --
//      Sec7.5's own adopted policy, matching the shipped attention call site's
//      SoftmaxKernelRefusedAfterGateAccepted precedent, never a silent one-step fallback to
//      plain argmax.
//   3. AntiLmPenalize computes p_omega(v) for the k candidates.
//   4. For each UNMASKED candidate (mask bit 1): alpha_eff = (alpha_q15 * p_omega(v)) >> 15
//      (both widened to int64 before the multiply, the product narrowed by >>kProbFracBits
//      BEFORE the subtraction -- Sec7.5's own pinned formula), s(v) = q_theta(v) - alpha_eff.
//      For each MASKED candidate: s(v) = INT64_MIN exactly (Sec7.5's own strike-6 remedy,
//      the masked-score floor) -- read from `mask_bits`, never inferred from the row's own
//      int32 value (Sec7.5's own strike-7 remedy: a legal token can legitimately narrow to
//      exactly INT32_MIN, so testing the row value instead of the mask bit collides with it).
//   5. Argmax over the k scores, lowest-TOKEN-INDEX tie-break (Sec7.6, C16 extended to the
//      k-element set) -- the ONE argmax that actually selects the emitted token.
// `alpha_q15` is the ABI's own alpha_q15 field, taken directly as a parameter (Phase D's own
// wiring, out of this suite's scope, is what will read it off sslm_decode_params).
[[nodiscard]] bool DampedGreedyScoreAndArgmax(const int32_t* masked_row, const uint8_t* mask_bits,
                                               int32_t vocab_size, int32_t k,
                                               const AntiLmState* anti_lm, int64_t alpha_q15,
                                               int64_t q_ln2, int64_t q_b, int64_t q_c,
                                               int32_t* out_token, bool* out_refused);

// Sec9 dim7's own "keeper-probe cell" (MEASUREMENT, never a build gate per D-SLM3727/Sec9's
// own classification discipline) and Sec5.6.3's model-governance column, at per-step grain:
// reports P_topk, alpha_eff, qspread, pomspread for the step DampedGreedyScoreAndArgmax just
// resolved, so the red suite can pin these five values are COMPUTED correctly against a
// known-good or adversarially-constructed row (which CAN fail a build) without ever pinning
// that any of them reads a particular way (which cannot). `full_row_z` is Z over the WHOLE
// masked row (the same denominator TopKRenormalizeQ15's own Z_k is a subset of) -- supplied
// by the caller since computing it is a full-row softmax, explicitly the O(V) operation this
// design's own O(k) redefinition (Sec5.3) never performs inside the hot path; the red suite's
// own fixtures compute it directly from a known row for the correctness pin.
struct DampedGreedyDiagnostics {
	int64_t z_k_q0;         // sum of the k gathered i-exp values, pre-renormalize (raw units)
	int64_t p_topk_q15;      // Z_k / full_row_z, in Q15
	int64_t alpha_eff_q15;   // (alpha_q15 * p_omega(winning candidate)) >> kProbFracBits
	int64_t qspread_q15;     // max(q_theta over k) - min(q_theta over k)
	int64_t pomspread_q15;   // max(p_omega over k) - min(p_omega over k)
};
[[nodiscard]] bool DampedGreedyScoreAndArgmaxDiag(
    const int32_t* masked_row, const uint8_t* mask_bits, int32_t vocab_size, int32_t k,
    const AntiLmState* anti_lm, int64_t alpha_q15, int64_t q_ln2, int64_t q_b, int64_t q_c,
    int64_t full_row_z, int32_t* out_token, bool* out_refused, DampedGreedyDiagnostics* out_diag);

}  // namespace superslm

#endif  // SSLM_T2199_DAMPED_GREEDY_H

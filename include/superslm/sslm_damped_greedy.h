// sslm_damped_greedy.h -- T-2199 Phase A (the n-gram anti-LM) and Phase C (the shared
// decode-step primitive, DampedGreedyScoreAndArgmax) production surface.
//
// Transcribed from the design of record, Claude/Plans/superslm-1p2-fsd-plan-2026-08-19.md
// (Wizard repo) at commit 603f9bfdb3 -- Sec7 (the mechanism), Sec7.2 (the anti-LM, its own
// exact-key-lookup determinism argument and its normalized-mixing-weight [0, 2^15] clamp),
// Sec7.3 (the two numeric surfaces), Sec7.4 (top-k selection, corrected tie-break: mask
// status before index), Sec7.5 (score/combine, the INT64_MIN masked-score floor, the >>15
// Q15 rescale), Sec7.6 (final selection), and Sec8 Phase A/Phase C's own decomposition into
// A1/A2/C1/C2/C2a/C3. This is the production copy of the interface Curie's own red suite
// declares (tests/t2199-damped-greedy-red-suite/sslm_damped_greedy.h) -- signatures match
// exactly (that is what makes the suite link); this file's own comments are the
// implementation's, not a duplicate of the suite's citation trail.
//
// Scope boundary matching the build commission: Phase B (the vocab-row logit scale
// artifact/converter), Phase B0 (the empirical calibration instrument), Phase D (ABI
// surface, both decode entry points, the GPU bridge), and Phase E (the confirmatory
// acceptance run) are OUT of this header's scope -- `(q_ln2, q_b, q_c)` and `alpha_q15` are
// taken as plain function parameters here, never read from an `sslm_decode_params` this
// header does not touch.
#ifndef SUPERSLM_SSLM_DAMPED_GREEDY_H
#define SUPERSLM_SSLM_DAMPED_GREEDY_H

#include <cstddef>
#include <cstdint>

namespace superslm {

// =====================================================================================
// Phase A -- the n-gram anti-LM (plan Sec7.2, Sec8 Phase A1/A2).
// =====================================================================================

// Opaque per-sequence state -- a warm-object class, not fresh-per-call (plan Sec9 dim1).
class AntiLmState;

// Constructs a fresh anti-LM instance mixing orders 1..max_order (Sec7.2's own "smoothed
// n-gram model over orders 1..N"). Ownership: exactly one AntiLmDestroy per AntiLmCreate.
[[nodiscard]] AntiLmState* AntiLmCreate(int max_order);
void AntiLmDestroy(AntiLmState* state);

// Appends one token to the sequence's own generated prefix (x<t, never the prompt).
// Called once per emitted token; mutates every order's count table. Strictly sequential,
// in generation order.
void AntiLmUpdate(AntiLmState* state, int32_t token);

// p_omega(v) in Q15 (kProbFracBits, intmath.h) for each of the `k` candidate tokens named
// in `candidates`. Exact-key lookup only: never iterates the state's own count tables in
// their container order. Every output is a normalized convex combination of each order's
// own Q15 count ratio (weights w_i/(sum w_i) over the orders whose context has been
// observed at all -- an order whose context was never seen contributes neither numerator
// nor denominator, so the combination renormalizes over the active orders, matching the
// plan's own "genuine convex combination" requirement at every query, not only when every
// order happens to be active) and is CLAMPED to [0, 1 << kProbFracBits].
void AntiLmPenalize(const AntiLmState* state, const int32_t* candidates, std::size_t k,
                     int64_t* out_p_omega_q15);

// Bytes currently retained across every order's count table -- grows with the number of
// distinct n-grams observed, never with the number of AntiLmUpdate calls.
std::size_t AntiLmRetainedBytes(const AntiLmState* state);

// =====================================================================================
// Phase C -- the shared decode-step primitive, DampedGreedyScoreAndArgmax (plan Sec7.4-7.6,
// Sec8 Phase C1/C2/C2a/C3).
// =====================================================================================

// Phase C1 (Sec7.4): selects exactly `k` positions from `masked_row` (an int32 row whose
// masked positions have already been narrowed to INT32_MIN by the caller) -- descending by
// row value; ties broken FIRST by mask status read from `mask_bits` (an unmasked position
// always outranks a masked one at equal value), THEN ascending token index within positions
// of the same mask status. `mask_bits` is a packed schema-mask array (1 bit/vocab position,
// LSB-first per byte). Writes exactly `k` vocabulary indices to `out_indices`.
void FsdTopK(const int32_t* masked_row, const uint8_t* mask_bits, int32_t vocab_size,
             int32_t k, int32_t* out_indices);

// Phase C2 (Sec7.3 surface 2, Sec5.5): q_theta(v) for exactly the `k` indices `FsdTopK`
// selected -- built directly from the certified SoftmaxRowQ15 (intmath.h), restricted to
// the k gathered elements. Returns `all_well_formed` exactly as `SoftmaxRowQ15` does.
[[nodiscard]] bool TopKRenormalizeQ15(const int32_t* row, const int32_t* indices,
                                       std::size_t k, int64_t q_ln2, int64_t q_b, int64_t q_c,
                                       int64_t* out_q15);

// Phase C3 (Sec7.5-7.6): the full score/combine/select mechanism for one decode step. A
// `TopKRenormalizeQ15` refusal is propagated (`*out_refused = true`, `*out_token`
// untouched) rather than falling back to plain argmax. For each unmasked candidate:
// alpha_eff = (alpha_q15 * p_omega(v)) >> kProbFracBits (both widened to int64 before the
// multiply, narrowed by the shift before the subtraction), s(v) = q_theta(v) - alpha_eff.
// For each masked candidate: s(v) = INT64_MIN exactly. Argmax over the k scores,
// lowest-token-index tie-break.
[[nodiscard]] bool DampedGreedyScoreAndArgmax(const int32_t* masked_row, const uint8_t* mask_bits,
                                               int32_t vocab_size, int32_t k,
                                               const AntiLmState* anti_lm, int64_t alpha_q15,
                                               int64_t q_ln2, int64_t q_b, int64_t q_c,
                                               int32_t* out_token, bool* out_refused);

// Sec9 dim7's own "keeper-probe cell" (MEASUREMENT, D-SLM3727: never a build gate on any
// reported value's own reading) and Sec5.6.3's model-governance column, at per-step grain.
// `full_row_z` is supplied by the caller (computing a full-row total is the O(V) operation
// this design's own O(k) redefinition never performs inside the hot path).
struct DampedGreedyDiagnostics {
	int64_t z_k_q0;         // sum of the k gathered i-exp values, pre-renormalize (raw units,
	                         // the same ShiftByMax-relative-to-the-row-max scale FsdTopK's own
	                         // gathered top-k set shares with the row's true global max)
	int64_t p_topk_q15;      // z_k_q0 / full_row_z, in Q15, CLAMPED to [0, 1<<kProbFracBits] --
	                         // a probability cannot exceed 1.0 regardless of full_row_z's own
	                         // precision at the call site (see build log: this clamp is the
	                         // documented resolution of an underspecified caller-units question)
	int64_t alpha_eff_q15;   // (alpha_q15 * p_omega(winning candidate)) >> kProbFracBits
	int64_t qspread_q15;     // max(q_theta over k) - min(q_theta over k)
	int64_t pomspread_q15;   // max(p_omega over k) - min(p_omega over k)
};
[[nodiscard]] bool DampedGreedyScoreAndArgmaxDiag(
    const int32_t* masked_row, const uint8_t* mask_bits, int32_t vocab_size, int32_t k,
    const AntiLmState* anti_lm, int64_t alpha_q15, int64_t q_ln2, int64_t q_b, int64_t q_c,
    int64_t full_row_z, int32_t* out_token, bool* out_refused, DampedGreedyDiagnostics* out_diag);

}  // namespace superslm

#endif  // SUPERSLM_SSLM_DAMPED_GREEDY_H

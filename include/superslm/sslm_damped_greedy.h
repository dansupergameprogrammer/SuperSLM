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
//
// Fix round 2026-08-20 (Claude/Poirot/7be9508-t2199-phaseAC-review.md, FIX-THEN-SHIP):
// every function below is caller-ensures on its pointer parameters (`state`, `anti_lm`,
// `masked_row`, `mask_bits`, `out_*`) -- none is null-checked, matching this codebase's own
// established convention elsewhere in `src/`; a null pointer where a non-null one is
// documented is undefined behaviour, not a rejection. Two domains are now enforced by the
// functions themselves rather than left as unstated preconditions a caller could cross into
// undefined behaviour or a corrupted candidate set (Poirot C1/S5) -- see `AntiLmCreate` and
// `DampedGreedyScoreAndArgmax`/`Diag` below for the exact contracts. The general (alpha/n/k)
// rejection Phase D1 will site at the ABI boundary supersedes both once built.
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
// Domain: max_order >= 1. Returns nullptr for max_order < 1 (Poirot M4: a negative
// max_order previously terminated the process; a zero max_order was silently accepted as a
// permanently-disabled anti_LM, p_omega always 0, rather than a rejection).
// AntiLmDestroy(nullptr) is safe.
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
// distinct n-grams observed, never with the number of AntiLmUpdate calls. Recalibrated
// 2026-08-20 against measured process-memory deltas (Poirot S3: previously read
// 3.0-5.9x low); remains a MODEL of allocator overhead, not a byte-exact accounting.
// Deliberately EXCLUDES the per-sequence generated-token history (grows by one token per
// AntiLmUpdate call regardless of repeats, unlike the table this reports on) -- a caller
// pricing the anti-LM's TOTAL footprint adds `generation_length_so_far * sizeof(int32_t)`
// directly; see the implementation's own comment for why folding it into this figure would
// break this suite's own memory-growth cell.
std::size_t AntiLmRetainedBytes(const AntiLmState* state);

// =====================================================================================
// Phase C -- the shared decode-step primitive, DampedGreedyScoreAndArgmax (plan Sec7.4-7.6,
// Sec8 Phase C1/C2/C2a/C3).
// =====================================================================================

// Phase C1 (Sec7.4): selects positions from `masked_row` (an int32 row whose masked
// positions have already been narrowed to INT32_MIN by the caller) -- descending by row
// value; ties broken FIRST by mask status read from `mask_bits` (an unmasked position
// always outranks a masked one at equal value), THEN ascending token index within positions
// of the same mask status. `mask_bits` is a packed schema-mask array (1 bit/vocab position,
// LSB-first per byte).
//
// Domain: 1 <= k <= vocab_size. This function has no return channel to report a domain
// violation (void, matching the suite's own declared interface) -- `DampedGreedyScoreAndArgmax`
// and `DampedGreedyScoreAndArgmaxDiag` below enforce the domain before calling this, and are
// the intended callers. A DIRECT caller outside that domain still gets a DEFINED result
// rather than undefined behaviour: writes exactly `min(k, vocab_size)` real vocabulary
// indices, most-favoured first, then -1 (never a valid vocabulary index, never left
// uninitialized) for every remaining slot up to `k` (Poirot S5 -- corrects the prior
// "writes exactly k" claim, which was false whenever k > vocab_size: the trailing slots
// were left untouched, a read of uninitialized memory for a direct caller).
void FsdTopK(const int32_t* masked_row, const uint8_t* mask_bits, int32_t vocab_size,
             int32_t k, int32_t* out_indices);

// Phase C2 (Sec7.3 surface 2, Sec5.5): q_theta(v) for exactly the `k` indices `FsdTopK`
// selected -- built directly from the certified SoftmaxRowQ15 (intmath.h), restricted to
// the k gathered elements, behind the same `CheckSoftmaxRowWidthDomain(q_b, q_c, k)` gate
// `SoftmaxRowQ15`'s own header documents as the caller's contract (Poirot M2 -- previously
// unenforced prose). Returns `all_well_formed` exactly as `SoftmaxRowQ15` does, including on
// a gate failure (every `out_q15[i]` written 0, matching `SoftmaxRowQ15`'s own "written for
// every element regardless of the returned bool" contract).
[[nodiscard]] bool TopKRenormalizeQ15(const int32_t* row, const int32_t* indices,
                                       std::size_t k, int64_t q_ln2, int64_t q_b, int64_t q_c,
                                       int64_t* out_q15);

// Phase C3 (Sec7.5-7.6): the full score/combine/select mechanism for one decode step.
//
// Domain: 1 <= k <= vocab_size (Poirot C1: `DampedGreedyScoreAndArgmaxDiag` previously
// subscripted its own k-sized diagnostics arrays before any k>0 test, an access violation
// at k=0; this sibling's own prior k=0 behavior -- an empty scoring loop writing
// `*out_token = -1` under a `true` return, indistinguishable from a successful selection of
// a real token -1 -- is also closed by this same domain check). A domain violation
// **returns false** and leaves `*out_token`/`*out_refused` UNTOUCHED. This is distinct from
// `*out_refused`, which signals a `TopKRenormalizeQ15` NUMERIC refusal on an otherwise
// well-formed call (return true, `*out_refused = true`, `*out_token` untouched). The
// general (alpha/n/k) rejection Phase D1 will site at the ABI boundary supersedes this
// domain check once built; until then this is the only guard between a caller and undefined
// behaviour.
//
// On a well-formed, non-refused call: for each unmasked candidate, alpha_eff = (alpha_q15 *
// p_omega(v)) >> kProbFracBits (both widened to int64 before the multiply, narrowed by the
// shift before the subtraction), s(v) = q_theta(v) - alpha_eff. For each masked candidate:
// s(v) = INT64_MIN exactly. Argmax over the k scores, lowest-token-index tie-break.
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
	                         // EXACT INTEGER (Poirot S1: previously computed in double, the
	                         // only float on this path, against plan Sec1 constraint 1). A
	                         // probability cannot exceed 1.0 regardless of full_row_z's own
	                         // precision at the call site (see build log Sec8: this clamp is
	                         // the documented resolution of an underspecified caller-units
	                         // question; Poirot S6 confirms the arithmetic is correct below
	                         // the ceiling and routes the still-missing discriminating
	                         // fixture to the suite owner, not to this code).
	int64_t alpha_eff_q15;   // (alpha_q15 * p_omega(winning candidate)) >> kProbFracBits
	int64_t qspread_q15;     // max(q_theta over k) - min(q_theta over k)
	int64_t pomspread_q15;   // max(p_omega over k) - min(p_omega over k)
};
// Same domain (1 <= k <= vocab_size) and false-on-violation contract as
// DampedGreedyScoreAndArgmax above (Poirot C1). On a domain violation *out_diag is left
// UNTOUCHED (the return is false before anything is written); on a TopKRenormalizeQ15
// refusal (return true, *out_refused = true) *out_diag is zero-filled rather than computed.
// Either way, every field is only MEANINGFUL when the return is true and *out_refused is
// false.
[[nodiscard]] bool DampedGreedyScoreAndArgmaxDiag(
    const int32_t* masked_row, const uint8_t* mask_bits, int32_t vocab_size, int32_t k,
    const AntiLmState* anti_lm, int64_t alpha_q15, int64_t q_ln2, int64_t q_b, int64_t q_c,
    int64_t full_row_z, int32_t* out_token, bool* out_refused, DampedGreedyDiagnostics* out_diag);

}  // namespace superslm

#endif  // SUPERSLM_SSLM_DAMPED_GREEDY_H

// T-1822 activation-scale remedy — Stage A primitive contract (T-1832, Curie).
//
// Design of record: Claude/Vitruvius/t1822-activation-scale-remedy-design-2026-08-07.md
// (§3-§6 mechanism definitions, §12 Coverage Model). This header is the API surface
// §11 Stage A's new instrument tool builds against, at the VALUE level only — the
// per-group power-of-two scale mechanism (M1, §4) and the fixed-P outlier-peel
// mechanism (M2, §5), mirroring §4-§5's committed-value semantics exactly. No engine
// forward-pass integration lives here: that is gate A6's read/write/join sweep and
// the stage-A/D tooling, which is build-stage work this ticket routes onward
// (Claude/Curie/t1832-activation-scale-remedy-red-suite-test-design-2026-08-08.md
// §5's routed-gap table).
//
// Every function below is DECLARED, NOT DEFINED. The test suite in tests/test_main.cpp
// (the "T-1832" section) calls them; linking superslm_tests therefore fails with
// unresolved externals until Brunel implements this file's .cpp — the red-unimplemented
// state Curie's discipline requires (Claude/Curie's own spec, Phase 4).
//
// Declaring this contract is authoring the test's target, not the feature: no function
// body exists here, and none of §12's cells are discharged by this file.
#ifndef SUPERSLM_T1822_ACTIVATION_SCALE_REMEDY_H
#define SUPERSLM_T1822_ACTIVATION_SCALE_REMEDY_H

#include <cstddef>
#include <cstdint>

namespace superslm_t1822 {

// C_max = 2^14 — the checked bound M2's peeled-code primitive rejects above (§5 step 3).
inline constexpr int64_t kPeelCMax = 16384;

enum class RemedyStatus {
	Ok = 0,
	// M2 step 3 (§5): the 128-bit shifted composite's high 64 bits are non-zero, or its
	// low word exceeds kPeelCMax. Evaluated on the UN-narrowed 128-bit magnitude, before
	// any truncation to int64 (D-SLM1633/1663 — the re-sited guard, §12 boundary bullet).
	PeeledCodeMagnitudeOutOfDomain,
};

// ---------------------------------------------------------------------------
// M1 — per-group power-of-two scales (§4.1 steps 2-3)
// ---------------------------------------------------------------------------

// §4.1 step 2, standard (non-RoPE-transiting) sites: the largest k in [0, k_cap] with
// (group_max_abs << k) <= row_max_abs. group_max_abs is D'_g (>= 1, C20's guard);
// row_max_abs is D' (the row's own single max-abs reduction, unchanged by grouping).
int ComputeRefinementExponent(int64_t group_max_abs, int64_t row_max_abs, int k_cap);

// §4.1 step 2, RoPE-safe variant (site 3 only, §6.2, D-SLM1816): k = 0 is always
// admissible; otherwise the largest k in [0, k_cap] with
// 127 * (group_max_abs << k) <= 90 * row_max_abs. Distinct predicate from the standard
// site's — NOT the same function with a different k_cap.
int ComputeRefinementExponentRopeSafe(int64_t group_max_abs, int64_t row_max_abs, int k_cap);

// §4.1 step 3: the existing C22 primitive (superslm::RequantTokenCodeWide) applied to
// the exactly pre-shifted operand `wide_value << k_g`. Not a re-implementation of C22 —
// a call to it. `r` and `s` are the row's own (unchanged) NormalizeScale/
// DynamicScaleReciprocal outputs.
int8_t ComputeGroupedCode(int64_t wide_value, int k_g, int64_t r, int s);

// §11's A6-independent differential reference for step 2 (D-SLM1614(i)): authored
// without reading ComputeRefinementExponent's own body —
// `while ((group_max_abs << (k+1)) <= row_max_abs) ++k`, capped at k_cap.
int ReferenceRefinementExponent(int64_t group_max_abs, int64_t row_max_abs, int k_cap);

// §11's A6-independent differential reference for the RoPE-safe variant
// (Mendeleev F1, D-SLM1881): authored without reading
// ComputeRefinementExponentRopeSafe's own body — the largest k in [0, k_cap] with
// 127 * (group_max_abs << (k+1)) <= 90 * row_max_abs, k = 0 always admissible.
// §12 bullet 1's differential requirement reaches this predicate exactly as it
// reaches the standard path's -- it is a distinct declared function, not a
// parameterization of ComputeRefinementExponent, and a differential cell against
// one does not discharge the other.
int ReferenceRefinementExponentRopeSafe(int64_t group_max_abs, int64_t row_max_abs, int k_cap);

// §11 A6 / §12 shape row: is `group_size` an admissible offline group stride for a row
// of `row_width` channels — a power of two that evenly divides row_width?
bool IsGroupSizeAdmissible(int64_t row_width, int64_t group_size);

// §6.3b: is `k_cap` admissible at a norm consumer (k_cap <= 3)?
bool IsNormConsumerKCapAdmissible(int k_cap);

// §12 shape row / §8.5, §6.1: is grouping (k_cap > 0) admissible at the named site?
// site_id follows §6.6's eighteen-site numbering. Refuses site 1 (committed-state,
// D-SLM1672) unconditionally, and sites 11/18 (peeled-residual, consistent-grid bound,
// D-SLM1662) whenever k_cap_requested > 0 — K = 0 only is admissible there.
bool IsGroupingAdmissibleAtSite(int site_id, int k_cap_requested);

// §6.2, D-SLM1679: is `group_size == 1` admissible at the named site? False only for
// site 3 (the RoPE pair-in-group floor is 2); true elsewhere (subject to
// IsGroupSizeAdmissible's own divisibility rule).
bool IsG1AdmissibleAtSite(int site_id, int64_t group_size);

// ---------------------------------------------------------------------------
// M2 — fixed-P outlier peel (§5 steps 1-3)
// ---------------------------------------------------------------------------

struct PeelRecord {
	uint16_t index = 0;
	int64_t c_star = 0;
};

// §5 step 1: writes the `min(p, n)` largest-magnitude channels of wide_row[0..n) into
// out_records, lowest-index tie-break (ArgmaxLowestIndexTieBreak's own rule). Returns
// the count written. Re-derives the set fresh from THIS row — an implementation that
// caches a prior call's set is exactly the fault the discrimination cell (audit F9)
// exists to catch.
size_t SelectPeelIndices(const int64_t* wide_row, size_t n, int p, PeelRecord* out_records);

// Exact integer ceiling division by 2^r_cap: (x + 2^r_cap - 1) >> r_cap.
int64_t CeilDivPow2(int64_t x, int r_cap);

// §5 step 2: D'_grid = max(unpeeled_max_abs, CeilDivPow2(full_row_max_abs, r_cap)).
// unpeeled_max_abs already carries C20's >= 1 guard.
int64_t ComputePeelGrid(int64_t unpeeled_max_abs, int64_t full_row_max_abs, int r_cap);

// §5 step 3, peeled-channel code: C22's composite (identical grid, tie rule, 128-bit
// intermediate) at (r, s), returning int64 rather than clamping to int8. Rejects with
// PeeledCodeMagnitudeOutOfDomain when the UN-narrowed 128-bit magnitude has non-zero
// high 64 bits or a low word exceeding kPeelCMax; *out_c_star is left UNTOUCHED on
// rejection (the atomicity contract, §12's failure-path bullet).
RemedyStatus ComputePeeledCode(int64_t wide_value, int64_t r, int s, int64_t* out_c_star);

// §11 A6-independent reference for step 3 (D-SLM1633): the same composite, authored
// independently of ComputePeeledCode's own body, same rejection rule.
RemedyStatus ReferencePeeledCode(int64_t wide_value, int64_t r, int s, int64_t* out_c_star);

// §6.3c: is (P, r_cap) an admissible parameter pair in this design's swept range
// (P <= 7 at r_cap = 7; P = 8 exceeds the residual consumer's headroom, E13)?
bool IsPeelParamsAdmissible(int p, int r_cap);

// T-1834 F6 (D-SLM1896, fold 12, §12 primitive-tier domain-extremity bullet):
// IsPeelParamsAdmissible above is a swept-range rectangle (P<=7 && r_cap<=7),
// sound only at the pinned n=1536 this design sweeps at. §6.3c's own region is
// the coupled, n-DEPENDENT inequality P*C_max^2 + (n-P)*127^2 <= 2^31 - 1, with
// C_max = 2^r_cap * 2^7 (the checked peeled-code bound at that r_cap). This
// function evaluates that inequality directly, taking n as its own operand
// rather than assuming the pinned row width. Declared, not defined -- the F6
// remedy is stage-A build-seat work (Claude/Curie's own red-first discipline);
// this red suite calls it to pin the coupled-inequality claim independently of
// the swept-range rectangle above, which the two-argument form cannot express
// (T-1838, D-SLM1898).
bool IsPeelParamsAdmissibleAtN(int p, int r_cap, int64_t n);

// §5 step 4, a GEMM consumer's rank-P fix-up (site 16's down_proj, §12 composition
// bullet): for every output channel j, acc[j] += sum over the P peel records of
// records[p].c_star * weight[records[p].index * out_channels + j] -- weight is the
// consumer's own weight matrix, ROW-MAJOR [in_channels x out_channels]. acc is
// read-modify-write: it already holds the bulk accumulate on entry, and this call
// adds the peeled channels' contribution on top of it, never replaces it. Fixed op
// count: record_count * out_channels multiply-adds (§5's own "fixed op count"
// framing, mirrored at the fix-up site).
void ApplyPeelRankPFixup(int64_t* acc, size_t out_channels, const PeelRecord* records,
                          size_t record_count, const int8_t* weight, size_t in_channels);

}  // namespace superslm_t1822

#endif  // SUPERSLM_T1822_ACTIVATION_SCALE_REMEDY_H
